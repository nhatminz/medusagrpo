"""Actual CUDA checks, explicitly skipped when a CUDA PyTorch runtime is absent."""
from types import SimpleNamespace
import os
import pytest
import torch
from medusa.tree import plan_tree,build_sparse_tree
from medusa.generate import speculative_generate

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires actual CUDA PyTorch runtime')
from config_smoke import MODELS as SUPPORTED_MODELS,smoke as config_smoke


@pytest.mark.parametrize('profile_mode',['none','explicit','automatic'])
def test_gpu_launcher_runtime_probe_supports_float32_target_autocast(profile_mode,monkeypatch,tmp_path):
    import json
    import warnings
    from helper.environment_checks import probe_training_runtime
    from helper.opd_profiles import execution_key,fingerprint
    key=execution_key(fingerprint(),151936,8,'bf16',16)
    profile=tmp_path/'qwen-production.json'
    if profile_mode!='none':
        profile.write_text(json.dumps(dict(execution_key=key,records=[dict(contexts=1,
            trials=[dict(slots=0,sparse=1.,fused=2.,gemm=3.)])])))
    monkeypatch.setenv('OPD_PROPOSAL_PROFILE',str(profile) if profile_mode=='explicit' else '')
    monkeypatch.setenv('OPD_PROPOSAL_PROFILE_DIR',str(tmp_path))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result=probe_training_runtime('cuda')
    assert not any('No exact compatible OPD proposal profile' in str(w.message) for w in caught)
    assert os.environ['OPD_PROPOSAL_PROFILE']==(str(profile) if profile_mode=='explicit' else '')
    assert result['architecture']=='medusa_parallel_3'
    assert result['target_forward_calls']>=2


@pytest.mark.parametrize('model',SUPPORTED_MODELS)
def test_seven_model_configs_native_architecture_cuda_smoke(model):
    config_smoke(model,'cuda')


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
    assert all(not s.teacher_tiles for s in m.medusa_proposal_runtime.engines)
    hook.remove()


def test_gpu_prefill_probability_precision_and_group_rng(tiny_model):
    from helper.opd_sampling import sample_target_with_metadata
    m=tiny_model.to('cuda');m.target_model.bfloat16();m.draft_model.heads.bfloat16()
    x=torch.tensor([[1,2,3],[2,1,4]],device='cuda');mask=torch.tensor([[1,1,1],[1,1,1]],device='cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        hidden=m.target_model.model(input_ids=x,attention_mask=mask,
            position_ids=(mask.cumsum(-1)-1).clamp_min(0)).last_hidden_state
        logits=m.lm_head(hidden[:,-1:])
    torch.manual_seed(9)
    first,p,_=sample_target_with_metadata(logits,do_sample=True,temperature=.8,top_p=.95,top_k=None,
        eos_token_id=22,mode='finite')
    assert p.dtype==torch.bfloat16
    expected_rng=torch.cuda.get_rng_state().clone()
    for method in ('medusa','medusa_reflex'):
        torch.manual_seed(9)
        out=speculative_generate(m,x,mask,SimpleNamespace(eos_token_id=22),method=method,
            do_sample=True,temperature=.8,top_p=.95,repeated_generate_nums=4,max_length=4,generation_length_policy='per_response')
        assert out['generated_token_ids']==[[int(first[r,0])] for r in range(2) for _ in range(4)]
        assert torch.equal(torch.cuda.get_rng_state(),expected_rng)


def test_gpu_online_bfloat16_gradient_parity(tiny_model,monkeypatch):
    from copy import deepcopy
    from medusa.training import train_heads
    from reference_online import train_heads as reference
    monkeypatch.setenv('MEDUSA_HEAD_LOGIT_ROWS','128')
    model=tiny_model.to('cuda');model.target_model.bfloat16();model.draft_model.heads.bfloat16()
    with torch.no_grad():
        for head in model.draft_model.heads:head.fc.weight.normal_(std=.03)
    old=deepcopy(model);new=deepcopy(model)
    states=[torch.randn(n,16,device='cuda',dtype=torch.bfloat16) for n in (19,8,15,10)]
    ids=[torch.randint(23,(x.shape[0],),device='cuda') for x in states]
    out=dict(all_draft_input_states=states,all_draft_input_ids=ids,prompt_lengths=[3,5])
    mask=torch.tensor([[1,1,1,0,0],[1,1,1,1,1]],device='cuda')
    reference(old,out,mask,repeated_generate_nums=2,draft_accumulation_steps=3)
    train_heads(new,out,mask,repeated_generate_nums=2,draft_accumulation_steps=3)
    torch.testing.assert_close(torch.tensor(old.last_head_losses),torch.tensor(new.last_head_losses),atol=.003,rtol=.003)
    for a,b in zip(old.draft_model.heads.parameters(),new.draft_model.heads.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=1e-4,rtol=.03)


def test_gpu_per_head_feedback_and_async_gradients_match_cpu(tiny_model):
    from copy import deepcopy
    from medusa.opd import MedusaOPD
    from helper.tree_verification import trace_verified_path
    m=tiny_model;gpu=deepcopy(m).to('cuda')
    cpu_engine=MedusaOPD(m,2,24,rank=8,topk=16,backend='auto',train_projector=True)
    gpu_engine=MedusaOPD(gpu,2,24,rank=8,topk=16,backend='auto',train_projector=True,update_stream=True)
    torch.manual_seed(74)
    anchor=torch.randn(2,16);teacher=torch.randn(2,12,23).softmax(-1)
    for _ in range(2):
        for model,engine,device in ((m,cpu_engine,'cpu'),(gpu,gpu_engine,'cuda')):
            raw,features=model.draft_model.logits(anchor.to(device),model.lm_head)
            ids,q=engine.propose(raw,features,4)
            tree=build_sparse_tree(torch.tensor([1,1],device=device),ids,q,plan_tree(2,24,12))
            samples=torch.full_like(tree.tokens,22);samples[:,0]=tree.tokens[:,1];samples[:,1]=tree.tokens[:,2]
            path=trace_verified_path(tree,samples,22)
            engine.feedback(tree,path,teacher.to(device))
        gpu_engine.wait()
        torch.testing.assert_close(cpu_engine.B_fast,gpu_engine.B_fast.cpu(),atol=2e-7,rtol=2e-4)
    torch.testing.assert_close(m.opd_projector_grad_sum,gpu.opd_projector_grad_sum.cpu(),atol=2e-6,rtol=2e-3)
    a=cpu_engine.finish();b=gpu_engine.finish()
    for h in (1,2,3):assert a[f'opd_head{h}_selected_states']==b[f'opd_head{h}_selected_states']
    gpu_engine.restart(2,24)
    assert not gpu_engine.B_fast.count_nonzero()


@pytest.mark.parametrize('family',['qwen2','qwen3','llama'])
def test_gpu_tree_kv_matches_recomputation_with_padding_rejection_and_bonus(family,strict_cuda_reference):
    from transformers import Qwen2Config,Qwen2ForCausalLM,Qwen3Config,Qwen3ForCausalLM,LlamaConfig,LlamaForCausalLM
    from helper.opd_static_cache import OPDStaticCache
    from helper import tree_kernels
    from medusa.generate import compact_kv
    config_cls,model_cls={'qwen2':(Qwen2Config,Qwen2ForCausalLM),'qwen3':(Qwen3Config,Qwen3ForCausalLM),
                         'llama':(LlamaConfig,LlamaForCausalLM)}[family]
    torch.manual_seed(83)
    config=config_cls(vocab_size=23,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1)
    config._attn_implementation='sdpa'
    target=model_cls(config).cuda().eval()
    prompt=torch.tensor([[1,2,3],[0,1,2]],device='cuda')
    mask=torch.tensor([[1,1,1],[0,1,1]],device='cuda')
    logical=mask.sum(-1);positions=(mask.cumsum(-1)-1).clamp_min(0)
    ids=torch.tensor([[[5,6,7,8],[9,10,11,12],[13,14,15,16]]]*2,device='cuda')
    q=torch.tensor([[[.7,.15,.1,.05]]*3]*2,device='cuda')
    tree=build_sparse_tree(torch.tensor([4,4],device='cuda'),ids,q,plan_tree(2,24,12))
    tree_mask=tree.attention_mask(3,torch.float32,kernels=tree_kernels,past_mask=mask.bool())
    reference_mask=tree.attention_mask(3,torch.float32,past_mask=mask.bool())
    assert torch.equal(tree_mask,reference_mask)
    cache=OPDStaticCache(32)
    with torch.inference_mode():
        target.model(input_ids=prompt,attention_mask=mask,position_ids=positions,past_key_values=cache,use_cache=True)
        verified=target.model(input_ids=tree.tokens,position_ids=logical[:,None]+tree.depths,
            attention_mask=tree_mask,
            past_key_values=cache,use_cache=True).last_hidden_state
        parents=tree.parents.cpu().tolist()
        paths=[]
        for row in range(2):
            row_paths=[]
            for node in range(12):
                path=[];cursor=node
                while cursor>=0:path.append(cursor);cursor=parents[row][cursor]
                path.reverse();row_paths.append(path)
                real=prompt[row:row+1,3-int(logical[row]):]
                sequence=torch.cat((real,tree.tokens[row:row+1,path]),1)
                expected=target.model(input_ids=sequence,use_cache=False).last_hidden_state[:,-1]
                torch.testing.assert_close(verified[row:row+1,node],expected,atol=2e-5,rtol=2e-4)
            paths.append(row_paths)
        # A deep accepted path and an immediate rejection commit unequal KV
        # suffix lengths; the final sampled bonus has no KV until this forward.
        chosen=[paths[0][-1],[0]];width=max(map(len,chosen))
        indices=torch.tensor([p+[-1]*(width-len(p)) for p in chosen],device='cuda')
        compact_kv(cache,indices,3,width)
        lengths=torch.tensor(list(map(len,chosen)),device='cuda')
        past_mask=torch.cat((mask.bool(),torch.arange(width,device='cuda')[None,:]<lengths[:,None],
                             torch.ones(2,1,device='cuda',dtype=torch.bool)),1)
        bonus=torch.tensor([[17],[18]],device='cuda')
        actual=target.model(input_ids=bonus,attention_mask=past_mask,position_ids=(logical+lengths)[:,None],
                            past_key_values=cache,use_cache=True).last_hidden_state
        for row,path in enumerate(chosen):
            real=prompt[row:row+1,3-int(logical[row]):]
            sequence=torch.cat((real,tree.tokens[row:row+1,path],bonus[row:row+1]),1)
            expected=target.model(input_ids=sequence,use_cache=False).last_hidden_state[:,-1:]
            torch.testing.assert_close(actual[row:row+1],expected,atol=2e-5,rtol=2e-4)
