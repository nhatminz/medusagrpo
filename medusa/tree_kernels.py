"""One GPU program per response builds its whole sparse prefix tree."""
import torch
import triton
import triton.language as tl
from medusa.tree import SparseTree


@triton.jit
def _build(ROOT,IDS,Q,T,P,D,S,C,BUDGET:tl.constexpr,HEADS:tl.constexpr,K:tl.constexpr,
           K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    batch=tl.program_id(0)
    nodes=tl.arange(0,BN); ranks=tl.arange(0,BK)
    parents=tl.full((BN,),-1,tl.int32)
    depths=tl.full((BN,),0,tl.int32)
    tokens=tl.full((BN,),-1,tl.int64)
    scores=tl.full((BN,),-float('inf'),tl.float32)
    tokens=tl.where(nodes==0,tl.load(ROOT+batch),tokens)
    scores=tl.where(nodes==0,0.,scores)
    running=tl.full((),0.,tl.float32)
    for h in tl.static_range(HEADS):
        token=tl.load(IDS+(batch*3+h)*K)
        prob=tl.load(Q+(batch*3+h)*K)
        running+=tl.log(tl.maximum(prob,1.e-30))
        tokens=tl.where(nodes==h+1,token,tokens)
        parents=tl.where(nodes==h+1,h,parents)
        depths=tl.where(nodes==h+1,h+1,depths)
        scores=tl.where(nodes==h+1,running,scores)
    # Only O(budget * max_topk) candidates; never enumerate Cartesian paths.
    for slot in tl.static_range(HEADS+1,BUDGET):
        horizon=tl.minimum(depths,2)
        candidate=tl.load(IDS+(batch*3+horizon[:,None])*K+ranks[None,:],ranks[None,:]<K,other=-2)
        probs=tl.load(Q+(batch*3+horizon[:,None])*K+ranks[None,:],ranks[None,:]<K,other=0.)
        limit=tl.where(horizon==0,K0,tl.where(horizon==1,K1,K2))
        valid=(nodes[:,None]<slot)&(depths[:,None]<HEADS)&(ranks[None,:]<limit[:,None])&(ranks[None,:]<K)
        for existing in tl.static_range(slot):
            ep=tl.sum(tl.where(nodes==existing,parents,0),0)
            et=tl.sum(tl.where(nodes==existing,tokens,0),0)
            valid=valid&~((nodes[:,None]==ep)&(candidate==et))
        value=tl.where(valid,scores[:,None]+tl.log(tl.maximum(probs,1.e-30)),-float('inf'))
        best=tl.max(tl.max(value,1),0)
        order=nodes[:,None]*BK+ranks[None,:]
        index=tl.min(tl.min(tl.where((value==best)&valid,order,BN*BK),1),0)
        parent=index//BK; rank=index%BK
        depth=tl.sum(tl.where(nodes==parent,depths,0),0)
        token=tl.load(IDS+(batch*3+tl.minimum(depth,2))*K+rank)
        tokens=tl.where(nodes==slot,token,tokens)
        parents=tl.where(nodes==slot,parent,parents)
        depths=tl.where(nodes==slot,depth+1,depths)
        scores=tl.where(nodes==slot,best,scores)
    tl.store(T+batch*BUDGET+nodes,tokens,nodes<BUDGET)
    tl.store(P+batch*BUDGET+nodes,parents,nodes<BUDGET)
    tl.store(D+batch*BUDGET+nodes,depths,nodes<BUDGET)
    tl.store(S+batch*BUDGET+nodes,scores,nodes<BUDGET)
    tl.store(C+batch*BUDGET+nodes,tl.where(depths<HEADS,0,-1),nodes<BUDGET)


def build(root,ids,probs,plan):
    b=root.numel();n=plan.budget
    tokens=torch.empty((b,n),device=root.device,dtype=torch.long)
    parents=torch.empty_like(tokens);depths=torch.empty_like(tokens);contexts=torch.empty_like(tokens)
    scores=torch.empty((b,n),device=root.device)
    _build[(b,)](root,ids.contiguous(),probs.contiguous(),tokens,parents,depths,scores,contexts,
        n,plan.active_heads,ids.shape[-1],*plan.topk,triton.next_power_of_2(n),triton.next_power_of_2(ids.shape[-1]),num_warps=4)
    return SparseTree(parents,tokens,contexts,plan.active_heads,depths,scores)
