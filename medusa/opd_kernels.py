"""Batch all three head SGD updates in one grid, retaining SpecNaacl arithmetic."""
import triton
import triton.language as tl
from helper.opd_reflex_kernels import _update,_round_end


@triton.jit
def _all_updates(S0,N0,I0,G0,U0,W0,B0,T0,A0,C0,
                 S1,N1,I1,G1,U1,W1,B1,T1,A1,C1,
                 S2,N2,I2,G2,U2,W2,B2,T2,A2,C2,
                 CONTEXTS,ROWS,K:tl.constexpr,R:tl.constexpr,LR:tl.constexpr):
    head=tl.program_id(1)
    if head==0:
        _update(S0,N0,I0,G0,U0,CONTEXTS,W0,B0,T0,A0,C0,None,ROWS,1,K,R,LR,
                triton.next_power_of_2(2*K),triton.next_power_of_2(R),False)
    elif head==1:
        _update(S1,N1,I1,G1,U1,CONTEXTS,W1,B1,T1,A1,C1,None,ROWS,1,K,R,LR,
                triton.next_power_of_2(2*K),triton.next_power_of_2(R),False)
    else:
        _update(S2,N2,I2,G2,U2,CONTEXTS,W2,B2,T2,A2,C2,None,ROWS,1,K,R,LR,
                triton.next_power_of_2(2*K),triton.next_power_of_2(R),False)


@triton.jit
def _all_end(B0,T0,A0,C0,W0,M0,B1,T1,A1,C1,W1,M1,B2,T2,A2,C2,W2,M2,
             R:tl.constexpr,LR:tl.constexpr):
    head=tl.program_id(0)
    if head==0:_round_end(B0,T0,A0,C0,W0,M0,R,LR,128,triton.next_power_of_2(R))
    elif head==1:_round_end(B1,T1,A1,C1,W1,M1,R,LR,128,triton.next_power_of_2(R))
    else:_round_end(B2,T2,A2,C2,W2,M2,R,LR,128,triton.next_power_of_2(R))


def update_heads(states,contexts):
    b,q=contexts.shape;arguments=[];end=[]
    for s in states:
        arguments.extend((s.selected_ids,s.selected_count,s.union_ids,s.union_g,s.u_cache,s.round_weight,
                          s.B_fast,s.bitmap,s.active_ids,s.active_count))
        end.extend((s.B_fast,s.bitmap,s.active_ids,s.active_count,s.round_weight,s.counters))
    s=states[0]
    if s.fast_lr>0:
        _all_updates[(min(b*q,32),3)](*arguments,contexts,q,s.topk,s.rank,s.fast_lr,
                                   num_warps=4,enable_fp_fusion=False)
    _all_end[(3,)](*end,s.rank,s.fast_lr,num_warps=4)
