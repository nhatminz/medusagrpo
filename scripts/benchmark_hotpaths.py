#!/usr/bin/env python3
"""Measure the original and revised head trainer/KV compactor on identical inputs.

This isolated microbenchmark does not imply full-model or B200 speedups.
CUDA timings and peak memory are opt-in measurements outside production.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from medusa.model import MedusaModel
from medusa.training import train_heads
from medusa.generate import compact_kv
from helper.opd_static_cache import OPDStaticCache
from reference_online import train_heads as original_train_heads


def original_compact(cache,indices,past,width):
    safe=indices[:,:width].clamp_min(0)
    for layer in cache.layers:
        for pool in (layer.key_pool,layer.value_pool):
            source=pool[:safe.shape[0],:,past:layer.length]
            selected=source.gather(2,safe[:,None,:,None].expand(-1,source.shape[1],-1,source.shape[3]))
            pool[:safe.shape[0],:,past:past+width].copy_(selected)
        layer.length=past+width


def measure(fn,prepare,trials,device):
    for _ in range(2):prepare();fn()
    times=[];peaks=[]
    for _ in range(trials):
        prepare()
        if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter();fn()
        if device.type=='cuda':torch.cuda.synchronize();peaks.append(torch.cuda.max_memory_allocated(device))
        times.append((time.perf_counter()-start)*1000)
    return dict(median_ms=statistics.median(times),trials_ms=times,
                peak_allocated_bytes=max(peaks) if peaks else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--hidden',type=int,default=64);p.add_argument('--vocab',type=int,default=4096)
    p.add_argument('--responses',type=int,default=16);p.add_argument('--length',type=int,default=96)
    p.add_argument('--prompt',type=int,default=64);p.add_argument('--trials',type=int,default=7)
    p.add_argument('--output',type=Path)
    a=p.parse_args();device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():p.error('CUDA required')
    if min(a.hidden,a.vocab,a.responses,a.trials)<1 or not 1<=a.prompt<a.length:p.error('invalid dimensions')
    torch.set_num_threads(1);torch.manual_seed(42)
    dtype=torch.bfloat16 if device.type=='cuda' else torch.float32
    class Target(nn.Module):
        def __init__(self):
            super().__init__();self.lm_head=nn.Linear(a.hidden,a.vocab,bias=False,device=device,dtype=dtype)
        @property
        def dtype(self):return self.lm_head.weight.dtype
    model=MedusaModel(SimpleNamespace(hidden_size=a.hidden,vocab_size=a.vocab),Target()).to(device)
    states=[torch.randn(a.length-r%5,a.hidden,device=device,dtype=dtype) for r in range(a.responses)]
    ids=[torch.randint(a.vocab,(x.shape[0],),device=device) for x in states]
    output=dict(all_draft_input_states=states,all_draft_input_ids=ids,prompt_lengths=[a.prompt]*a.responses)
    mask=torch.ones(a.responses,a.prompt,device=device,dtype=torch.long)
    head={}
    for name,fn in [('before',original_train_heads),('after',train_heads)]:
        head[name]=measure(lambda:fn(model,output,mask),lambda:model.zero_grad(set_to_none=True),a.trials,device)
    cache=OPDStaticCache(256);b=a.responses;past=128;nodes=12;width=4
    values=torch.randn(b,8,past+nodes,64,device=device,dtype=dtype)
    for layer in range(8):cache.update(values,values,layer)
    indices=torch.tensor([0,1,3,7],device=device).expand(b,-1).contiguous()
    def prepare():
        for layer in cache.layers:layer.length=past+nodes
    kv={}
    with torch.inference_mode():
        for name,fn in [('before',original_compact),('after',compact_kv)]:
            kv[name]=measure(lambda:fn(cache,indices,past,width),prepare,a.trials,device)
    for component in (head,kv):component['median_speedup']=component['before']['median_ms']/component['after']['median_ms']
    report=dict(device=str(device),hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        torch=torch.__version__,synthetic_inputs=True,dimensions=vars(a)|{'output':str(a.output)},
        online_heads=head,kv_compaction=kv,
        note='Microbenchmark only; no B200/full-model claim. New trainer caps total vocabulary rows at MEDUSA_HEAD_LOGIT_ROWS (128 by default).')
    text=json.dumps(report,indent=2)+'\n'
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(text)


if __name__=='__main__':main()
