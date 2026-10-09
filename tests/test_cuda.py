"""Actual CUDA checks, explicitly skipped when a CUDA PyTorch runtime is absent."""
from types import SimpleNamespace
import pytest
import torch
from medusa.tree import plan_tree,build_sparse_tree
from medusa.generate import speculative_generate

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires actual CUDA PyTorch runtime')


def test_gpu_sparse_builder_matches_cpu_reference():
    torch.manual_seed(11)
    q,ids=torch.randn(16,3,31).softmax(-1).topk(4)
    root=torch.arange(16);plan=plan_tree(16,192,12)
    a=build_sparse_tree(root,ids,q,plan)
    b=build_sparse_tree(root.cuda(),ids.cuda(),q.cuda(),plan)
    for name in ('parents','depths','tokens'):assert torch.equal(getattr(a,name),getattr(b,name).cpu())
    torch.testing.assert_close(a.scores,b.scores.cpu(),atol=2e-6,rtol=2e-6)


def test_gpu_decoder_opd_async_and_zero_identity(tiny_model):
    m=tiny_model.to(device='cuda')
    m.target_model.bfloat16();m.draft_model.heads.bfloat16()
    x=torch.tensor([[1,2,3],[0,1,4]],device='cuda');mask=torch.tensor([[1,1,1],[0,1,1]],device='cuda')
    counts=[];hook=m.target_model.model.register_forward_hook(lambda *_:counts.append(1))
    def run(method,lr=.01,stream=True):
        counts.clear();torch.manual_seed(6)
        out=speculative_generate(m,x,mask,SimpleNamespace(eos_token_id=22),do_sample=True,
            repeated_generate_nums=2,max_length=20,method=method,opd_fast_lr=lr,
            opd_train_projector=True,opd_update_stream=stream)
        assert out['target_forward_calls']==len(counts)==out['verification_batches']+1
        return out
    baseline=run('medusa');zero=run('medusa_reflex',0)
    assert baseline['generated_token_ids']==zero['generated_token_ids']
    serial=run('medusa_reflex',stream=False)
    asynchronous=run('medusa_reflex',stream=True)
    assert serial['generated_token_ids']==asynchronous['generated_token_ids']
    assert asynchronous['opd_updates']>0
    assert asynchronous['head3_proposed_nodes']>0
    assert m.opd_projector_grad_sum.isfinite().all()
    hook.remove()
