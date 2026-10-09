#!/usr/bin/env python3
"""Compile new kernels for sm100 without executing them or claiming GPU validation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from triton.compiler import compile,ASTSource
from triton.backends.compiler import GPUTarget
from medusa.tree_kernels import _build
from medusa.opd_kernels import _all_updates,_all_end
from helper.tree_kernels import _tree_mask


def compile_all():
    specs=[
        (_build,dict(zip(('ROOT','IDS','Q','T','P','D','S','C'),('*i64','*i64','*fp32','*i64','*i64','*i64','*fp32','*i64'))),
         dict(BUDGET=12,HEADS=3,K=4,K0=4,K1=3,K2=2,BN=16,BK=4)),
        (_all_updates,{**{f'{n}{h}':t for h in range(3) for n,t in [('S','*i32'),('N','*i32'),('I','*i64'),('G','*fp32'),
             ('U','*fp32'),('W','*fp32'),('B','*fp32'),('T','*i32'),('A','*i32'),('C','*i32')]},'CONTEXTS':'*i64','ROWS':'i32'},dict(K=16,R=8,LR=.01)),
        (_all_end,{f'{n}{h}':t for h in range(3) for n,t in [('B','*fp32'),('T','*i32'),('A','*i32'),('C','*i32'),('W','*fp32'),('M','*fp64')]},dict(R=8,LR=.01)),
        (_tree_mask,{'PARENTS':'*i64','MASK':'*bf16','ROWS':'i32','PAST':'i32','WIDTH':'i32',
                     'PAST_MASK':'*i1','MASK_STRIDE':'i32'},dict(MINIMUM=-3.3895313892515355e38,BK=256,HAS_PAST_MASK=True)),
    ]
    for fn,sig,constants in specs:
        kernel=compile(ASTSource(fn,signature=sig,constexprs=constants),target=GPUTarget('cuda',100,32),
                       options={'num_warps':4,'enable_fp_fusion':False})
        print(f'{fn.__name__}: compiled sm100, warps={kernel.metadata.num_warps}; NOT executed')


if __name__=='__main__':compile_all()
