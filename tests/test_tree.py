from types import SimpleNamespace
import pytest
import torch
from medusa.tree import plan_tree,build_sparse_tree,select_feedback
from helper.tree_verification import trace_verified_path


@pytest.mark.parametrize('active,budget,max_nodes',[(1,41,41),(8,80,24),(128,512,24),(128,256,24)])
def test_prefix_budget_attention_and_horizons(active,budget,max_nodes):
    plan=plan_tree(active,budget,max_nodes)
    logits=torch.randn(active,3,17)
    q=logits.softmax(-1);p,ids=q.topk(4,-1)
    tree=build_sparse_tree(torch.zeros(active,dtype=torch.long),ids,p,plan)
    assert tree.tokens.shape[1]*active<=budget
    assert tree.tokens.shape[1]<=max_nodes
    assert plan.active_heads==min(3,budget//active-1)
    assert bool(plan.reason)==(plan.active_heads<3)
    mask=tree.attention_mask(2,torch.float32)
    for r in range(min(active,3)):
        for node in range(plan.budget):
            ancestors={node};cur=int(tree.parents[r,node])
            while cur>=0:
                assert cur<node
                ancestors.add(cur);cur=int(tree.parents[r,cur])
            allowed=set(torch.where(mask[r,0,node,2:]==0)[0].tolist())
            assert allowed==ancestors
            assert len(ancestors)==int(tree.depths[r,node])+1
        assert set(tree.depths[r].tolist())==set(range(plan.active_heads+1))
        children=set()
        for node in range(1,plan.budget):
            key=(int(tree.parents[r,node]),int(tree.tokens[r,node]))
            assert key not in children
            children.add(key)


def test_high_concurrency_cannot_override_global_memory_limit():
    assert plan_tree(128,512,24).active_heads==3
    assert plan_tree(128,256,24).reason
    with pytest.raises(ValueError,match='lower rollout concurrency'):plan_tree(513,512,24)


def test_three_heads_can_all_be_accepted():
    ids=torch.tensor([[[2,5,6,7],[3,5,6,7],[4,5,6,7]]])
    q=torch.tensor([[[.7,.15,.1,.05]]*3])
    tree=build_sparse_tree(torch.tensor([1]),ids,q,plan_tree(1,4,4))
    path=trace_verified_path(tree,torch.tensor([[2,3,4,9]]),eos_token_id=10)
    assert path.tokens.tolist()==[[2,3,4,9]]
    assert path.packed_indices.tolist()==[[0,1,2,3]]
    assert path.lengths.tolist()==[4]


@pytest.mark.parametrize('strategy',['visited_only','visited_capped_frontier','all_internal_weighted'])
def test_frontier_cap_unique_and_head_specific(strategy):
    ids=torch.tensor([[[2,5,6,7],[3,5,6,7],[4,5,6,7]]]*3)
    q=torch.tensor([[[.7,.15,.1,.05]]*3]*3)
    tree=build_sparse_tree(torch.tensor([1,1,1]),ids,q,plan_tree(3,123,41))
    samples=torch.full_like(tree.tokens,22);samples[:,0]=2;samples[:,1]=3
    path=trace_verified_path(tree,samples,23)
    for h in range(3):
        w,kind=select_feedback(tree,path,h,strategy)
        assert ((w>0)&(tree.depths!=h)).sum()==0
        if strategy=='visited_capped_frontier':assert ((kind==2).sum(1)<=2).all()
        if strategy=='visited_only':assert (kind==2).sum()==0
        assert ((kind==1).sum(1)<=1).all()
