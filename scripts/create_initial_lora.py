#!/usr/bin/env python3
"""Create one explicit common initialization when no prior target LoRA exists."""
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model,LoraConfig,TaskType


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    a=p.parse_args()
    if a.output.exists():p.error('output already exists; choose a new initialization path')
    torch.manual_seed(a.seed)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(a.device)
    config=LoraConfig(task_type=TaskType.CAUSAL_LM,r=64,lora_alpha=32,lora_dropout=0.,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    model=get_peft_model(model,config)
    model.save_pretrained(a.output)
    print(f'Common initial LoRA: {a.output.resolve()}. Set TARGET_ADAPTER to this path for ALL baselines.')


if __name__=='__main__':main()
