#!/usr/bin/env python3
"""Bounded real-weight decoder/head CUDA smoke, not a GRPO/B200 benchmark."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from medusa.model import MedusaModel
from medusa.generate import speculative_generate
from medusa.training import train_heads


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',required=True)
    parser.add_argument('--heads',help='optional shared pretrained Medusa checkpoint')
    parser.add_argument('--target-adapter',help='optional common initial target LoRA')
    parser.add_argument('--responses',type=int,default=2)
    parser.add_argument('--new-tokens',type=int,default=12)
    parser.add_argument('--length-policy',choices=('specnaacl_compatible','per_response'),default='specnaacl_compatible')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available():parser.error('actual CUDA runtime required')
    if args.responses<1 or args.new_tokens<2:parser.error('responses>=1 and new-tokens>=2 required')
    torch.set_num_threads(1);torch.manual_seed(42)
    target=AutoModelForCausalLM.from_pretrained(args.model_dir,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).cuda().eval()
    if args.target_adapter:
        from peft import PeftModel
        target=PeftModel.from_pretrained(target,args.target_adapter).eval()
    for parameter in target.parameters():parameter.requires_grad_(False)
    model=MedusaModel(target.config,target).cuda()
    if args.heads:model.load_model(args.heads)
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True,padding_side='left')
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    questions=('What is 17 + 25?','Calculate the area of a rectangle with width 7 and height 12.')
    texts=[tokenizer.apply_chat_template([{'role':'user','content':p}],tokenize=False,add_generation_prompt=True)
           for p in questions]
    batch=tokenizer(texts,padding=True,return_tensors='pt').to('cuda')
    backbone=target.get_base_model().model if hasattr(target,'get_base_model') else target.model
    calls=[];hook=backbone.register_forward_hook(lambda *_:calls.append(1))
    records=[]

    def run(method,**options):
        calls.clear();model.opd_projector_grad_sum.zero_();model.opd_projector_grad_weight.zero_()
        torch.manual_seed(93);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        out=speculative_generate(model,batch.input_ids,batch.attention_mask,tokenizer,method=method,do_sample=True,
            repeated_generate_nums=args.responses,max_length=batch.input_ids.shape[1]+args.new_tokens,
            return_all_draft_input=True,opd_train_projector=True,generation_length_policy=args.length_policy,**options)
        assert len(calls)==out['target_forward_calls']==out['verification_batches']+1
        records.append(dict(method=method,options=options,generated_tokens=sum(out['response_generated_tokens']),
            target_forward_calls=out['target_forward_calls'],verification_batches=out['verification_batches'],
            aal=out['total_acc_length']/max(out['total_decoded_token_num'],1),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_s=time.perf_counter()-start,
            **{k:v for k,v in out.items() if k.startswith('head') or k.startswith('opd_head')}))
        return out,torch.cuda.get_rng_state().clone()

    baseline,rng=run('medusa')
    for options in ({'opd_enabled':False},{'opd_fast_lr':0}):
        output,other=run('medusa_reflex',**options)
        assert output['generated_token_ids']==baseline['generated_token_ids']
        assert torch.equal(rng,other)
    serial,_=run('medusa_reflex',opd_update_stream=False)
    serial_b=model.medusa_proposal_runtime.B_fast.clone()
    asynchronous,_=run('medusa_reflex',opd_update_stream=True)
    assert serial['generated_token_ids']==asynchronous['generated_token_ids']
    torch.testing.assert_close(serial_b,model.medusa_proposal_runtime.B_fast,atol=2e-7,rtol=.002)
    before=len(calls);torch.cuda.reset_peak_memory_stats()
    train_heads(model,asynchronous,batch.attention_mask,repeated_generate_nums=args.responses)
    assert len(calls)==before
    assert all(p.grad is None for p in target.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.draft_model.heads.parameters())
    report=dict(hardware=torch.cuda.get_device_name(),torch=torch.__version__,model=str(Path(args.model_dir).resolve()),
        real_target_weights=True,heads=args.heads or 'synthetic zero-residual initialization; not pretrained experiment heads',
        target_adapter=args.target_adapter or 'none; base-target decoder/head smoke only',
        prompts=2,responses_per_prompt=args.responses,max_length=batch.input_ids.shape[1]+args.new_tokens,
        generation_length_policy=args.length_policy,
        prompt_lengths=batch.attention_mask.sum(-1).cpu().tolist(),records=records,
        disabled_and_b0_output_rng_parity=True,serial_async_parity=True,head_losses=model.last_head_losses,
        head_training_peak_allocated_bytes=torch.cuda.max_memory_allocated(),no_extra_head_training_target_forward=True,
        limitations=['Not a GRPO/reward or 5-epoch real pretraining run','Not a B200 benchmark',
                     'Not full production-concurrency or seven-model OOM verification'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));hook.remove()


if __name__=='__main__':main()
