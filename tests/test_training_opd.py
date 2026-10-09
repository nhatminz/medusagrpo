from types import SimpleNamespace
import torch
from medusa.training import train_heads
from medusa.opd import MedusaOPD
from medusa.tree import build_sparse_tree,plan_tree,select_feedback
from helper.tree_verification import trace_verified_path


def test_all_heads_have_supervision_and_update_only_owned_parameters(tiny_model):
    m=tiny_model
    for p in m.target_model.parameters():p.requires_grad_(True)
    tokens=torch.tensor([[1,2,3,4,5,6,7,8,9,10]])
    with torch.no_grad():hidden=m.target_model.model(tokens,use_cache=False).last_hidden_state
    before={k:v.clone() for k,v in m.target_model.state_dict().items()}
    opt=torch.optim.AdamW(m.draft_model.parameters(),lr=.01)
    heads_before=[h.fc.weight.clone() for h in m.draft_model.heads]
    out={'all_draft_input_states':[hidden[0]],'all_draft_input_ids':[tokens[0]]}
    train_heads(m,out,torch.ones(1,2,dtype=torch.long))
    assert min(m.last_head_losses)>0
    opt.step()
    assert all(not torch.equal(h.fc.weight,w) for h,w in zip(m.draft_model.heads,heads_before))
    assert all(p.grad is None for p in m.target_model.parameters())
    assert all(torch.equal(before[k],v) for k,v in m.target_model.state_dict().items())


def test_per_head_update_matches_specnaacl_sparse_gradient(tiny_model):
    m=tiny_model;b=2
    engine=MedusaOPD(m,b,24,rank=8,topk=16,fast_lr=.01,backend='auto',train_projector=True)
    torch.manual_seed(54)
    raw,features=m.draft_model.logits(torch.randn(b,16),m.lm_head)
    ids,q=engine.propose(raw,features,4)
    tree=build_sparse_tree(torch.tensor([1,1]),ids,q,plan_tree(b,24,12))
    samples=torch.full_like(tree.tokens,22)
    samples[:,0]=tree.tokens[:,1];samples[:,1]=tree.tokens[:,2]
    path=trace_verified_path(tree,samples,22)
    teacher=torch.randn(b,tree.tokens.shape[1],23).softmax(-1)
    expected=[]
    for h,state in enumerate(engine.engines):
        w,_=select_feedback(tree,path,h)
        qq=raw[:,h].detach().softmax(-1)[:,None,:].expand_as(teacher)
        ti=teacher.topk(16,-1).indices
        di=state.ids_cache[:b,0,None].expand(b,tree.tokens.shape[1],16)
        union=torch.cat((di,ti),-1)
        valid=torch.cat((torch.ones_like(di,dtype=torch.bool),~(ti[:,:,:,None]==di[:,:,None,:]).any(-1)),-1)
        grad=(qq-teacher).gather(-1,union)*w[:,:,None]*valid
        u=features[:,h].detach().float()@m.opd_projector.detach()
        delta=torch.zeros(23,8)
        delta.index_add_(0,union.flatten(),(grad[:,:,:,None]*u[:,None,None]).reshape(-1,8))
        expected.append(-.01*delta/w.sum().clamp_min(1))
    engine.feedback(tree,path,teacher)
    torch.testing.assert_close(engine.B_fast,torch.stack(expected),atol=2e-9,rtol=2e-5)
    assert engine.B_fast.shape==(3,23,8)
    engine.restart(b,24)
    assert engine.B_fast.count_nonzero()==0


def test_masks_supervise_future_label_and_do_not_cross_padding(tiny_model):
    m=tiny_model;ids=torch.arange(10)[None,:]
    attention=torch.tensor([[0,1,1,1,1,1,1,1,0,0]])
    mask=torch.tensor([[0,0,0,0,1,1,1,1,0,0]])
    hidden=torch.randn(1,10,16,requires_grad=True)
    losses=list(m.draft_model.loss_chunks(hidden,ids,attention,mask,m.lm_head,2))
    sum(x[1]*x[2] for x in losses).backward()
    assert hidden.grad is None
    assert all(h.fc.weight.grad is not None and h.fc.weight.grad.norm()>0 for h in m.draft_model.heads)


def test_no_feedback_skips_deep_head_updates(tiny_model):
    m=tiny_model
    engine=MedusaOPD(m,1,2,rank=8,topk=16,fast_lr=.01,backend='auto')
    raw,features=m.draft_model.logits(torch.randn(1,16),m.lm_head)
    ids,q=engine.propose(raw,features,4)
    tree=build_sparse_tree(torch.tensor([1]),ids,q,plan_tree(1,2,2))
    path=trace_verified_path(tree,torch.full_like(tree.tokens,22),22)
    engine.feedback(tree,path,torch.randn(1,2,23).softmax(-1))
    assert engine.B_fast[1:].count_nonzero()==0
    counters=engine.finish()
    assert counters['opd_head2_selected_states']==0
    assert counters['opd_head3_selected_states']==0


def test_short_pretrain_batch_keeps_all_head_gradient_collectives(tiny_model):
    from train_draft import pretrain_loss
    ids=torch.tensor([[1,2,3,4]])
    pretrain_loss(tiny_model,dict(input_ids=ids,attention_mask=torch.ones_like(ids),
                                  loss_mask=torch.ones_like(ids)))
    assert tiny_model.last_head_losses[0]>0
    assert tiny_model.last_head_losses[1]>0
    assert tiny_model.last_head_losses[2]==0
    assert all(p.grad is not None for h in tiny_model.draft_model.heads for p in h.parameters())
    assert all(p.grad.count_nonzero()==0 for p in tiny_model.draft_model.heads[2].parameters())
    assert all(p.grad is None for p in tiny_model.target_model.parameters())
