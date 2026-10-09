"""Medusa ShareGPT pretraining with SpecNaacl paths and resumable checkpoints."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler
from medusa.model import MedusaModel as FastGRPOModel
from helper.pretrain_data import DataCollator
from helper.checkpointing import capture_rng_state, restore_rng_state


def pretrain_loss(model, batch, loss_scale=1.):
    batch={k:v.to(model.device) for k,v in batch.items()}
    with torch.no_grad():
        target=model.target_model.get_base_model() if hasattr(model.target_model,'get_base_model') else model.target_model
        hidden=target.model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).last_hidden_state
    sums=torch.zeros(3,device=model.device)
    for h,ce,weight in model.draft_model.loss_chunks(hidden,batch['input_ids'],batch['attention_mask'],batch['loss_mask'],model.lm_head):
        sums[h]+=ce.detach()
        (ce*weight*loss_scale).backward()
    # Keep gradient collectives identical across ranks when a short sequence
    # has no labels for one of the deeper horizons.
    for head in model.draft_model.heads:
        for parameter in head.parameters():
            if parameter.grad is None:parameter.grad=torch.zeros_like(parameter)
    model.last_head_losses=sums.cpu().tolist()
    return sum((.8**h/sum(.8**j for j in range(3)))*x for h,x in enumerate(sums)),sums.sum()*0


def atomic_save(payload,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(payload,tmp);tmp.replace(path)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model_dir','dataset_dir','saved_model_dir','log_dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--version_name',default='medusa-pretrain')
    p.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    p.add_argument('--load_lora_path',default='')
    p.add_argument('--model_type',default='qwen2');p.add_argument('--num_epochs',type=int,default=5)
    p.add_argument('--batch_size',type=int,default=4);p.add_argument('--accumulation_steps',type=int,default=1)
    p.add_argument('--lr',type=float,default=5e-5);p.add_argument('--warmup_ratio',type=float,default=.05)
    p.add_argument('--max_length',type=int,default=2048);p.add_argument('--max_samples',type=int,default=0)
    p.add_argument('--num_workers',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--save_interval',type=int,default=500);p.add_argument('--resume',default='')
    p.add_argument('--max_steps',type=int,default=0,help='Stop after this optimizer step, save resumable state; 0 means all epochs')
    p.add_argument('--model_output_root',default='');p.add_argument('--dtype',default='bf16',choices=['bf16','fp16','fp32','auto'])
    p.add_argument('--attn_implementation',default='sdpa')
    a=p.parse_args(argv)
    if min(a.batch_size,a.accumulation_steps,a.num_epochs,a.max_length,a.save_interval)<1: p.error('positive training settings required')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    if world>1:dist.init_process_group('nccl')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    config=AutoConfig.from_pretrained(a.model_dir)
    dtype={'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32,'auto':'auto'}[a.dtype]
    target=AutoModelForCausalLM.from_pretrained(a.model_dir,torch_dtype=dtype,attn_implementation=a.attn_implementation).to(device).eval()
    if a.load_lora_path:
        from peft import PeftModel
        target=PeftModel.from_pretrained(target,a.load_lora_path).eval()
    for param in target.parameters():param.requires_grad_(False)
    config=deepcopy(config);config.num_hidden_layers=1;config.rope_scaling=None;config.torch_dtype=target.dtype
    model=FastGRPOModel(config,target).to(device)
    tokenizer=AutoTokenizer.from_pretrained(a.model_dir,padding_side='right')
    data=json.loads(Path(a.dataset_dir).read_text())
    if a.max_samples:data=data[:a.max_samples]
    sampler=DistributedSampler(data,num_replicas=world,rank=rank,shuffle=True,seed=a.seed)
    # A separate generator prevents iterator construction on resume from consuming model RNG.
    loader=DataLoader(data,batch_size=a.batch_size,sampler=sampler,
        collate_fn=DataCollator(tokenizer,a.max_length,a.model_type),num_workers=a.num_workers,
        persistent_workers=a.num_workers>0,generator=torch.Generator().manual_seed(a.seed))
    model.opd_projector.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.draft_model.parameters(),lr=a.lr)
    total_steps=a.num_epochs*((len(loader)+a.accumulation_steps-1)//a.accumulation_steps)
    scheduler=get_scheduler('cosine_with_min_lr',optimizer=optimizer,
        num_warmup_steps=min(int(a.warmup_ratio*total_steps),500),num_training_steps=total_steps,
        scheduler_specific_kwargs={'min_lr_rate':0.})
    output=Path(a.saved_model_dir);logs=Path(a.log_dir)
    output.mkdir(parents=True,exist_ok=True);logs.mkdir(parents=True,exist_ok=True)
    latest=output/(a.version_name+'-latest');latest.mkdir(exist_ok=True)
    step=accumulated=epoch_start=batch_start=0
    optimizer.zero_grad(set_to_none=True)
    if a.resume:
        path=latest/'training_state.pt' if a.resume=='auto' else Path(a.resume)
        if path.is_dir():path=path/'training_state.pt'
        if path.exists():
            state=torch.load(path,map_location='cpu',weights_only=False)
            if state['world_size']!=world:raise ValueError('resume world size mismatch')
            model.draft_model.load_state_dict(state['draft_model'])
            optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
            step,accumulated,epoch_start,batch_start=(state[k] for k in ('step','accumulated','epoch','next_batch'))
            local=state['ranks'][rank]
            for name,param in model.draft_model.named_parameters():
                if name in local['gradients']:param.grad=local['gradients'][name].to(param.device)
            restore_rng_state(local['rng'])
        elif a.resume!='auto':raise FileNotFoundError(path)
    def save(epoch,next_batch):
        local=dict(rng=capture_rng_state(),gradients={n:p.grad.cpu() for n,p in model.draft_model.named_parameters() if p.grad is not None})
        ranks=[None]*world
        if world>1:dist.all_gather_object(ranks,local)
        else:ranks=[local]
        if rank:return
        weights={'architecture':'medusa_parallel_3','draft_model':model.draft_model.state_dict()}
        atomic_save(weights,output/f'step{step}.pth');atomic_save(weights,latest/'draft.pth')
        atomic_save(dict(**weights,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            step=step,accumulated=accumulated,epoch=epoch,next_batch=next_batch,world_size=world,ranks=ranks),latest/'training_state.pt')
        # Full target config supplies vocabulary/head dimensions for the proposal tuner.
        target.config.to_json_file(latest/'target_config.json')
        if a.model_output_root:
            root=Path(a.model_output_root);root.mkdir(parents=True,exist_ok=True)
            for name,destination in [('latest_checkpoint',latest),('latest_target_config.json',latest/'target_config.json')]:
                link=root/name;temporary=root/(name+'.tmp')
                temporary.unlink(missing_ok=True);temporary.symlink_to(destination.resolve());temporary.replace(link)
    start=time.perf_counter()
    progress_disabled=rank!=0 or os.environ.get('TQDM_DISABLE','').strip().lower() in {'1','true','yes','on'}
    for epoch in range(epoch_start,a.num_epochs):
        sampler.set_epoch(epoch)
        progress=tqdm(enumerate(loader),total=len(loader),desc=f'Pretrain epoch {epoch+1}/{a.num_epochs}',
                      unit='batch',dynamic_ncols=True,disable=progress_disabled)
        for i,batch in progress:
            if epoch==epoch_start and i<batch_start:continue
            has_labels=torch.any(batch['loss_mask']==1).to(device=model.device,dtype=torch.int32)
            if world>1:dist.all_reduce(has_labels,op=dist.ReduceOp.MIN)
            if not has_labels:continue
            loss1,loss2=pretrain_loss(model,batch,1./a.accumulation_steps);loss=loss1+loss2
            valid=torch.isfinite(loss).to(torch.int32)
            if world>1:dist.all_reduce(valid,op=dist.ReduceOp.MIN)
            if not valid:continue
            accumulated+=1
            if accumulated%a.accumulation_steps:continue
            if world>1:
                for param in model.draft_model.parameters():
                    if param.grad is not None:dist.all_reduce(param.grad);param.grad.div_(world)
            optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
            if rank==0:
                row=dict(step=step,epoch=epoch,batch=i,loss=float(loss.detach()),loss1=float(loss1.detach()),loss2=float(loss2.detach()),wall_time_s=time.perf_counter()-start,**{f'head{h+1}_training_loss':v for h,v in enumerate(model.last_head_losses)})
                progress.set_postfix(step=step,loss=f"{row['loss']:.4f}",
                    h1=f'{model.last_head_losses[0]:.4f}',h2=f'{model.last_head_losses[1]:.4f}',h3=f'{model.last_head_losses[2]:.4f}',
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",refresh=False)
                for filename in ('metrics.jsonl',f'epoch_{epoch}.log'):
                    with (logs/filename).open('a') as f:f.write(json.dumps(row)+'\n')
            if step%a.save_interval==0:save(epoch,i+1)
            if a.max_steps>0 and step>=a.max_steps:
                if not progress_disabled:
                    progress.n=i+1
                    progress.refresh()
                progress.close()
                save(epoch,i+1)
                if world>1:dist.destroy_process_group()
                return
        progress.close()
        remainder=accumulated%a.accumulation_steps
        if remainder:
            for param in model.draft_model.parameters():
                if param.grad is not None:
                    param.grad.mul_(a.accumulation_steps/remainder)
                    if world>1:dist.all_reduce(param.grad);param.grad.div_(world)
            optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
            accumulated+=a.accumulation_steps-remainder
            if rank==0:
                values=model.last_head_losses
                row=dict(step=step,epoch=epoch,batch=len(loader)-1,
                    loss=sum((.8**h/sum(.8**j for j in range(3)))*x for h,x in enumerate(values)),
                    partial_accumulation=True,lr=optimizer.param_groups[0]['lr'],
                    wall_time_s=time.perf_counter()-start,
                    **{f'head{h+1}_training_loss':v for h,v in enumerate(values)})
                for filename in ('metrics.jsonl',f'epoch_{epoch}.log'):
                    with (logs/filename).open('a') as stream:stream.write(json.dumps(row)+'\n')
        save(epoch+1,0)
        if a.max_steps>0 and step>=a.max_steps:
            if world>1:dist.destroy_process_group()
            return
        batch_start=0
    save(a.num_epochs,0)
    if rank==0:
        (output/'pretrain_complete.json').write_text(json.dumps(dict(step=step,epochs=a.num_epochs,target_model_path=str(Path(a.model_dir).resolve()),target_adapter=a.load_lora_path))+'\n')
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
