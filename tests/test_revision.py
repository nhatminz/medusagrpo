"""Regression evidence for the correctness/fairness/performance revision."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from medusa.generate import speculative_generate,compact_kv
from medusa.training import train_heads
from medusa.tree import build_sparse_tree,plan_tree,restrict_feedback,select_feedback
from helper.tree_verification import trace_verified_path
from helper.opd_sampling import sample_target_with_metadata
from helper.opd_static_cache import OPDStaticCache
from reference_online import train_heads as original_train_heads
from baseline_support import baseline_root,require_baseline_files,source_contract_complete

ROOT=Path(__file__).resolve().parents[1]
from config_smoke import MODELS as SUPPORTED_MODELS,smoke as config_smoke


@pytest.mark.parametrize('model',SUPPORTED_MODELS)
def test_seven_model_configs_native_architecture_cpu_smoke(model):
    config_smoke(model,'cpu')


@pytest.mark.parametrize('row_budget',[9,33,128])
@pytest.mark.parametrize('accumulation',[1,3])
def test_online_loss_gradients_match_original_variable_responses(tiny_model,monkeypatch,row_budget,accumulation):
    monkeypatch.setenv('MEDUSA_HEAD_LOGIT_ROWS',str(row_budget))
    torch.manual_seed(17)
    for head in tiny_model.draft_model.heads:
        nn.init.normal_(head.fc.weight,std=.05)
        nn.init.normal_(head.fc.bias,std=.05)
    old=deepcopy(tiny_model);new=deepcopy(tiny_model)
    lengths=[11,7,15,9,6,4];prompts=[3,5,2]
    with torch.inference_mode():
        states=[torch.randn(n,16) for n in lengths]
        ids=[torch.randint(0,23,(n,)) for n in lengths]
    output=dict(all_draft_input_states=states,all_draft_input_ids=ids,prompt_lengths=prompts)
    mask=torch.tensor([[1,1,1,0,0],[1,1,1,1,1],[1,1,0,0,0]])
    reference=original_train_heads(old,output,mask,repeated_generate_nums=2,draft_accumulation_steps=accumulation)
    projections=[];native=torch.nn.functional.linear
    def linear(x,weight,bias=None):
        if weight.data_ptr()==new.lm_head.weight.data_ptr():projections.append(x.shape[0])
        return native(x,weight,bias)
    monkeypatch.setattr(torch.nn.functional,'linear',linear)
    actual=train_heads(new,output,mask,repeated_generate_nums=2,draft_accumulation_steps=accumulation)
    torch.testing.assert_close(torch.tensor(actual),torch.tensor(reference),atol=1e-6,rtol=2e-6)
    torch.testing.assert_close(torch.tensor(new.last_head_losses),torch.tensor(old.last_head_losses),atol=1e-6,rtol=2e-6)
    assert projections and max(projections)<=row_budget
    for (name,a),(other,b) in zip(old.named_parameters(),new.named_parameters()):
        assert name==other
        if a.grad is None:assert b.grad is None
        else:torch.testing.assert_close(a.grad,b.grad,atol=2e-7,rtol=2e-5)
    assert all(n>0 for n in new.last_head_supervised_tokens)


def test_first_token_grouping_rng_matches_specnaacl(tiny_model):
    x=torch.tensor([[1,2,3],[2,1,4]]);mask=torch.tensor([[1,1,1],[1,1,1]])
    with torch.no_grad():
        hidden=tiny_model.target_model.model(input_ids=x,attention_mask=mask,
            position_ids=(mask.cumsum(-1)-1).clamp_min(0)).last_hidden_state
        logits=tiny_model.lm_head(hidden[:,-1:])
    torch.manual_seed(57)
    first,_,_=sample_target_with_metadata(logits,do_sample=True,temperature=.8,top_p=.7,top_k=5,
                                         eos_token_id=22,return_probs=False,mode='finite')
    expected_rng=torch.get_rng_state().clone()
    for method,kw in [('medusa',{}),('medusa_reflex',{'opd_enabled':False}),('medusa_reflex',{'opd_fast_lr':0})]:
        torch.manual_seed(57)
        result=speculative_generate(tiny_model,x,mask,SimpleNamespace(eos_token_id=22),
            repeated_generate_nums=4,do_sample=True,temperature=.8,top_p=.7,top_k=5,
            max_length=4,method=method,generation_length_policy='per_response',**kw)
        assert result['generated_token_ids']==[[int(first[r,0])] for r in range(2) for _ in range(4)]
        assert torch.equal(torch.get_rng_state(),expected_rng)
        assert result['target_forward_calls']==1
        assert result['medusa_scheduling_syncs']==1
        assert 'opd_host_syncs' not in result


def test_zero_fast_adapter_preserves_all_proposals_and_tree_nodes(tiny_model):
    from medusa.opd import MedusaOPD
    raw,features=tiny_model.draft_model.logits(torch.randn(8,16),tiny_model.lm_head)
    off=MedusaOPD(tiny_model,8,96,enabled=False)
    on=MedusaOPD(tiny_model,8,96,enabled=True,fast_lr=0)
    a,qa=off.propose(raw,features,4);b,qb=on.propose(raw,features,4)
    assert torch.equal(a,b);torch.testing.assert_close(qa,qb,atol=0,rtol=0)
    root=torch.ones(8,dtype=torch.long);plan=plan_tree(8,96,12)
    ta=build_sparse_tree(root,a,qa,plan);tb=build_sparse_tree(root,b,qb,plan)
    for field in ('tokens','parents','depths','scores'):assert torch.equal(getattr(ta,field),getattr(tb,field))


def test_feedback_excludes_eos_prefixes_and_truncated_horizons():
    ids=torch.tensor([[[2,22,6,7],[3,22,6,7],[4,22,6,7]]]*3)
    q=torch.tensor([[[.7,.15,.1,.05]]*3]*3)
    tree=build_sparse_tree(torch.tensor([1,1,1]),ids,q,plan_tree(3,123,41))
    samples=torch.full_like(tree.tokens,21);samples[:,0]=2;samples[:,1]=3
    path=trace_verified_path(tree,samples,22)
    remaining=torch.tensor([1,2,3])
    path.packed_indices.masked_fill_(torch.arange(4)[None,:]>=remaining[:,None],-1)
    restrict_feedback(tree,remaining,22)
    for h in range(3):
        weights,kind=select_feedback(tree,path,h)
        assert ((kind==2).sum(1)<=2).all()
        for row,node in (weights>0).nonzero().tolist():
            assert int(tree.depths[row,node])<int(remaining[row])
            while node>=0:
                assert int(tree.tokens[row,node])!=22
                node=int(tree.parents[row,node])
    assert select_feedback(tree,path,2)[0][0].sum()==0
    assert select_feedback(tree,path,2)[0][1].sum()==0
    assert select_feedback(tree,path,2)[0][2].sum()>0


@pytest.mark.parametrize('eos,max_length,expected,heads,aal',[
    (7,10,[1,2,3,4,5,6,7],[2,1,1],3.),
    (2,10,[1,2],[0,0,0],1.),
    (7,4,[1,2,3],[1,0,0],2.),
    (1,10,[1],[0,0,0],0.),
])
def test_handcrafted_root_pending_bonus_and_head_accounting(monkeypatch,eos,max_length,expected,heads,aal):
    from medusa.model import MedusaModel
    class Backbone(nn.Module):
        def forward(self,input_ids,past_key_values=None,**kw):
            hidden=torch.nn.functional.one_hot(input_ids,8).float()
            if past_key_values is not None:past_key_values.update(hidden[:,None],hidden[:,None],0)
            return SimpleNamespace(last_hidden_state=hidden)
    class Target(nn.Module):
        def __init__(self):
            super().__init__();self.model=Backbone();self.lm_head=nn.Linear(8,8,bias=False)
            with torch.no_grad():
                self.lm_head.weight.fill_(-8)
                for token in range(8):self.lm_head.weight[(token+1)%8,token]=8
        @property
        def dtype(self):return torch.float32
    model=MedusaModel(SimpleNamespace(hidden_size=8,vocab_size=8),Target())
    def heads_fn(anchor,lm_head):
        features=torch.nn.functional.one_hot((anchor.argmax(-1)[:,None]+torch.arange(1,4)[None,:])%8,8).float()
        return lm_head(features),features
    monkeypatch.setattr(model.draft_model,'logits',heads_fn)
    monkeypatch.setenv('MAX_TREE_NODES_PER_SEQ','4')
    result=speculative_generate(model,torch.tensor([[0]]),torch.ones(1,1,dtype=torch.long),
        SimpleNamespace(eos_token_id=eos),do_sample=False,max_length=max_length,generation_length_policy='per_response')
    assert result['generated_token_ids']==[expected]
    assert [result[f'head{h}_accepted_tokens'] for h in (1,2,3)]==heads
    assert sum(heads)==result['total_accepted_draft_tokens']
    assert result['total_acc_length']==sum(result['response_accepted_length_sum'])
    assert result['total_decoded_token_num']==sum(result['response_verification_rounds'])
    assert result['total_acc_length']/max(result['total_decoded_token_num'],1)==aal
    assert all(result[f'head{h}_verified_tokens']==result[f'head{h}_proposed_nodes'] for h in (1,2,3))


@pytest.mark.parametrize('value_dtype',[torch.float32,torch.float16])
def test_kv_compaction_reuses_bounded_scratch_and_handles_nonmonotonic_path(value_dtype):
    cache=OPDStaticCache(32)
    values=torch.arange(2*2*9*3).reshape(2,2,9,3).float()
    for layer in range(3):cache.update(values+100*layer,(values+1000+100*layer).to(value_dtype),layer)
    path=torch.tensor([[0,4,6],[0,2,-1]])
    expected=[(layer.key_pool[:2,:,2:9].gather(2,path.clamp_min(0)[:,None,:,None].expand(2,2,3,3)),
               layer.value_pool[:2,:,2:9].gather(2,path.clamp_min(0)[:,None,:,None].expand(2,2,3,3))) for layer in cache.layers]
    compact_kv(cache,path,2,3)
    for layer,(keys,vals) in zip(cache.layers,expected):
        torch.testing.assert_close(layer.keys[:,:,2:],keys)
        torch.testing.assert_close(layer.values[:,:,2:],vals)
    pointers=[v.data_ptr() for k,v in cache._scratch.items() if k[0]=='medusa_suffix']
    assert len(pointers)==(1 if value_dtype==torch.float32 else 2)
    compact_kv(cache,torch.tensor([[0,1],[1,0]]),2,2)
    assert pointers==[v.data_ptr() for k,v in cache._scratch.items() if k[0]=='medusa_suffix']
    dtype_bytes=4+(2 if value_dtype==torch.float16 else 0)
    assert cache.statistics()['kv_compaction_workspace_bytes']<=2*2*3*3*dtype_bytes*2


def _target_objective_scope(root):
    scope={'torch':torch}
    for file,names in [('helper/fastgrpo_training.py',{'compute_target_loss'}),
                       ('grpo_speculative.py',{'compute_target_loss_and_backward'})]:
        tree=ast.parse((root/file).read_text())
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(root/file),'exec'),scope)
    return scope['compute_target_loss_and_backward']


def test_target_grpo_loss_gradients_match_actual_specnaacl(tiny_model):
    source=ROOT.parent/'SpecNaacl'
    if not source.is_dir():pytest.skip('SpecNaacl source is unavailable; parity not verified')
    from peft import get_peft_model,LoraConfig,TaskType
    target=get_peft_model(deepcopy(tiny_model.target_model),LoraConfig(task_type=TaskType.CAUSAL_LM,
        r=64,lora_alpha=32,lora_dropout=0.,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    for name,param in target.named_parameters():
        if 'lora_B' in name:
            with torch.no_grad():param.normal_(std=.02)
    targets=[deepcopy(target),deepcopy(target)]
    x=torch.tensor([[1,2,3,4,5],[1,3,6,0,0]])
    attention=torch.tensor([[1,1,1,1,1],[1,1,1,0,0]])
    mask=torch.tensor([[0,1,1,1,1],[0,1,1,0,0]])
    rewards=torch.tensor([[1.],[-1.]])
    reports=[]
    for model,root in zip(targets,(source,ROOT)):
        fn=_target_objective_scope(root);old=reference=None;losses=[]
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6)
        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss=fn(SimpleNamespace(target_model=model),x,attention,mask,rewards,.1,.04,iteration,
                    old_logps=old,ref_logps=reference,loss_scale=.3)
            old,reference=loss[-2:];losses.append(loss[:3]);optimizer.step()
        reports.append(losses)
    assert reports[0]==reports[1]
    for a,b in zip(targets[0].parameters(),targets[1].parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)


def test_generation_hot_path_has_no_dynamic_mask_compaction():
    tree=ast.parse((ROOT/'medusa/generate.py').read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('nonzero','item','synchronize') for n in ast.walk(tree))
    # Runtime-dependent strict sampler recovery is audited separately; this
    # source check only prevents reintroducing the confirmed generation sites.


def test_previous_teacher_storage_is_released_before_next_sampler(tiny_model,monkeypatch):
    import weakref
    import medusa.generate as generation
    native=generation.sample_target_with_metadata;references=[]
    def sampler(logits,**kwargs):
        assert all(ref() is None for ref in references)
        builder=kwargs.get('metadata_builder')
        if builder is not None:
            def capture(tokens,probs,metadata):
                references.append(weakref.ref(probs))
                return builder(tokens,probs,metadata)
            kwargs['metadata_builder']=capture
        return native(logits,**kwargs)
    monkeypatch.setattr(generation,'sample_target_with_metadata',sampler)
    result=speculative_generate(tiny_model,torch.tensor([[1,2,3]]),torch.ones(1,3,dtype=torch.long),
        SimpleNamespace(eos_token_id=22),method='medusa_reflex',do_sample=True,max_length=12)
    assert references and result['target_forward_calls']>1


def test_fairness_audit_effective_configs_all_seven_models():
    from scripts.check_fairness import audit,MODELS
    spec=baseline_root('SpecNaacl');pure=baseline_root('puregrpo')
    require_baseline_files(spec,'grpo_speculative.py','helper/fastgrpo_generate.py','helper/opd_generate.py')
    report=audit(spec,pure)
    assert [row['model'] for row in report['models']]==list(MODELS)
    assert all(not row['specnaacl_configuration_mismatches'] for row in report['models'])
    assert report['configuration_status']=='PASS'
    assert all(row['methods'][method]['inspection_status']=='inspected' for row in report['models']
               for method in ('medusa','medusa_reflex','fastgrpo','fastgrpo_reflex'))
    assert all(report['first_token_convention'][m]=='shared_per_prompt' for m in
               ('medusa','medusa_reflex','fastgrpo','fastgrpo_reflex'))
    assert report['initial_target_lora']['status']=='NOT VERIFIED'
    sources=[report['source_checks'][m] for m in ('medusa','medusa_reflex','fastgrpo','fastgrpo_reflex')]
    recipes=[s['lora_recipe'] for s in sources]
    assert all(recipe==recipes[0] for recipe in recipes)
    optimizers=[s['target_optimizer'] for s in sources]
    assert all(optimizer==optimizers[0] for optimizer in optimizers)


def test_available_puregrpo_source_fairness_contract():
    from scripts.check_fairness import audit
    pure=baseline_root('puregrpo')
    if not source_contract_complete(pure):
        pytest.skip('NOT VERIFIED: complete PureGRPO comparison/audit source unavailable at '+str(pure))
    report=audit(baseline_root('SpecNaacl'),pure)
    assert all(not [v for v in row['configuration_mismatches'] if v['method']=='puregrpo']
               for row in report['models'])
    assert report['first_token_convention']['puregrpo']=='independent_per_response'
    source=report['source_checks']['puregrpo'];reference=report['source_checks']['medusa']
    assert source['lora_recipe']==reference['lora_recipe']
    assert source['target_optimizer']==reference['target_optimizer']


def test_missing_baseline_source_is_not_reported_as_pass(tmp_path):
    from scripts.check_fairness import audit
    report=audit(ROOT.parent/'SpecNaacl',tmp_path/'missing',models=('qwen25_1p5b',))
    assert report['models'][0]['methods']['puregrpo']['status']=='NOT VERIFIED'
    assert report['source_checks']['puregrpo']['status']=='not verified'


@pytest.mark.parametrize('field,value,message',[
    ('runtime_semantics_version','old_runtime','semantics mismatch'),
    ('method','medusa_reflex','method mismatch'),
    ('opd_enabled',True,'OPD_ENABLED mismatch'),
    ('generation_length_policy','per_response','generation length policy mismatch'),
])
def test_resume_rejects_incompatible_runtime_or_ablation(tmp_path,field,value,message):
    from medusa.generate import RUNTIME_SEMANTICS_VERSION
    import torch.distributed as dist
    # Isolate the actual loader from the training CLI, and exercise refusal
    # before any weights/optimizers/RNG can be mutated.
    tree=ast.parse((ROOT/'grpo_speculative.py').read_text())
    loader=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_training_checkpoint')
    scope=dict(torch=torch,dist=dist)
    exec(compile(ast.Module(body=[loader],type_ignores=[]),'checkpoint_loader','exec'),scope)
    checkpoint=dict(format='medusa_grpo_checkpoint_v1',runtime_semantics_version=RUNTIME_SEMANTICS_VERSION,
                    method='medusa',opd_enabled=False,world_size=1,grpo_alignment_version='response_rows_v2')
    checkpoint[field]=value
    path=tmp_path/'resume.pt';torch.save(checkpoint,path)
    model=SimpleNamespace(_training_method='medusa',_opd_enabled=False)
    with pytest.raises(ValueError,match=message):
        scope['load_training_checkpoint'](path,model=model,optimizer_target=None,optimizer_draft=None)
