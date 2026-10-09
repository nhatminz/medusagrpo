"""Real target objectives and decoding for the final alignment/length correction."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import numpy as np
import torch
from peft import get_peft_model, LoraConfig, TaskType
from medusa.generate import speculative_generate

ROOT=Path(__file__).resolve().parents[1]


def module(path):
    spec=importlib.util.spec_from_file_location(path.parent.parent.name+'_'+path.stem,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def objective(root):
    if root.name=='puregrpo':return module(root/'helper/grpo_core.py').compute_target_loss_and_backward
    scope={'torch':torch}
    for file,names in [('helper/fastgrpo_training.py',{'compute_target_loss'}),
                       ('grpo_speculative.py',{'compute_target_loss_and_backward'})]:
        nodes=[n for n in ast.parse((root/file).read_text()).body if getattr(n,'name',None) in names]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(root/file),'exec'),scope)
    return scope['compute_target_loss_and_backward']


@pytest.mark.parametrize('repository',['MedusaGRPO','SpecNaacl','puregrpo'])
def test_sorted_response_identity_loss_and_gradients(repository,tiny_model):
    root=ROOT.parent/repository;alignment=module(root/'helper/response_alignment.py')
    data=dict(messages=[],rewards=[],std_rewards=[],response_metadata=[])
    ids=[[1,3,4,5,6],[1,7],[1,8,9,10]]
    masks=[[0,1,1,1,1],[0,1],[0,1,1,1]]
    rewards=np.array([1.2,.2,.7]);advantages=((rewards-rewards.mean())/rewards.std()).tolist()
    alignment.append_response_group(data, ['long','short','medium'], rewards.tolist(), advantages, (0,2,17,3))
    original=deepcopy(data)
    # Execute the actual trainer's sorting call, not a stand-in packing routine.
    source=root/('helper/train_ops.py' if repository=='puregrpo' else 'grpo_speculative.py')
    call=next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.Assign)
              and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)
              and n.value.func.id=='sort_training_rows')
    scope=dict(input_ids=deepcopy(ids),attention_mask=[[1]*len(x) for x in ids],
               loss_mask=deepcopy(masks),batch_data=data,sort_training_rows=alignment.sort_training_rows)
    exec(compile(ast.Module(body=[call],type_ignores=[]),str(source),'exec'),scope)
    assert data['messages']==['short','medium','long']
    assert data['rewards']==[.2,.7,1.2]
    assert data['std_rewards']==[advantages[1],advantages[2],advantages[0]]
    assert [m['response_id'] for m in data['response_metadata']]==[1,2,0]
    assert all(m['prompt_id']==(0,2,17,3) for m in data['response_metadata'])
    target=get_peft_model(deepcopy(tiny_model.target_model),LoraConfig(task_type=TaskType.CAUSAL_LM,
        r=64,lora_alpha=32,lora_dropout=0.,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    with torch.no_grad():
        for name,p in target.named_parameters():
            if 'lora_B' in name:p.normal_(std=.02)
    fn=objective(root);models=[target,deepcopy(target)];reports=[]
    for model,seqs,mask_rows,advantages in [(models[0],ids,masks,original['std_rewards']),
          (models[1],scope['input_ids'],scope['loss_mask'],data['std_rewards'])]:
        width=max(map(len,seqs))
        x=torch.tensor([x+[0]*(width-len(x)) for x in seqs])
        attn=torch.tensor([[1]*len(x)+[0]*(width-len(x)) for x in seqs])
        mask=torch.tensor([x+[0]*(width-len(x)) for x in mask_rows])
        old=ref=None;losses=[]
        for iteration in (0,1):
            model.zero_grad(set_to_none=True)
            result=fn(SimpleNamespace(target_model=model),x,attn,mask,torch.tensor(advantages)[:,None],
                      .1,.04,iteration,old_logps=old,ref_logps=ref,loss_scale=1/3)
            losses.append(result[:3]);old,ref=result[-2:]
        reports.append(losses)
    torch.testing.assert_close(torch.tensor(reports[0]),torch.tensor(reports[1]),atol=1e-6,rtol=1e-5)
    for a,b in zip(models[0].parameters(),models[1].parameters()):
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-5)


@pytest.mark.parametrize('method',['medusa','medusa_reflex'])
@pytest.mark.parametrize('near_limit',[False,True])
@pytest.mark.parametrize('device',['cpu']+(['cuda'] if torch.cuda.is_available() else []))
def test_true_prompt_budget_mixed_eos_and_pending(tiny_model,method,near_limit,device):
    # Native transformer/KV path with independent capped response lifetimes.
    tiny_model=tiny_model.to(device)
    with torch.no_grad():tiny_model.target_model.lm_head.weight.zero_()
    width=2047 if near_limit else 7
    lengths=[width,width-2,width-4]
    x=torch.ones(3,width,dtype=torch.long,device=device)
    mask=torch.zeros_like(x)
    for row,length in enumerate(lengths):mask[row,-length:]=1;x[row,:width-length]=0
    result=speculative_generate(tiny_model,x,mask,SimpleNamespace(eos_token_id=22),
        max_length=width+1,method=method,return_all_draft_input=True)
    assert list(map(len,result['generated_token_ids']))==[1,3,5]
    assert result['prompt_lengths']==lengths
    for row,ids in enumerate(result['all_draft_input_ids']):
        assert len(ids)==width+1
        assert ids.tolist()==[1]*lengths[row]+[0]*(width+1-lengths[row])
    assert result['medusa_scheduling_syncs']==1+result['medusa_round_scheduling_syncs']
    # Compare stored accepted KV hidden states with an ordinary full recompute.
    for row,ids in enumerate(result['all_draft_input_ids']):
        with torch.inference_mode(),torch.autocast(device,dtype=torch.float16,enabled=device=='cuda'):
            full=tiny_model.target_model.model(input_ids=ids[None],attention_mask=torch.ones_like(ids[None])).last_hidden_state[0]
        # Final pending token has not been run through the transformer.
        torch.testing.assert_close(result['all_draft_input_states'][row][:-1],full[:-1],
                                   atol=3e-3 if device=='cuda' else 2e-5,rtol=3e-3 if device=='cuda' else 2e-5)


def test_mixed_prompt_eos_removes_only_finished_rows(tiny_model):
    with torch.no_grad():tiny_model.target_model.lm_head.weight.zero_()
    x=torch.tensor([[1,1,1],[0,0,1]]);mask=torch.tensor([[1,1,1],[0,0,1]])
    for eos,expected in ((0,[1,1]),(22,[1,3])):
        result=speculative_generate(tiny_model,x,mask,SimpleNamespace(eos_token_id=eos),max_length=4)
        assert list(map(len,result['generated_token_ids']))==expected


def test_five_methods_load_same_lora_tensors_and_detect_mutation(tiny_model,tmp_path):
    from scripts.check_fairness import lora_recipe
    recipes=[(ROOT/'grpo_speculative.py','MedusaGRPO'),(ROOT/'grpo_speculative.py','MedusaGRPO'),
             (ROOT.parent/'SpecNaacl/grpo_speculative.py','SpecNaacl'),
             (ROOT.parent/'SpecNaacl/grpo_speculative.py','SpecNaacl'),
             (ROOT.parent/'puregrpo/helper/target.py','puregrpo')]
    shared=tmp_path/'shared';hashes=[]
    for index,(file,repository) in enumerate(recipes):
        torch.manual_seed(91+index)
        recipe=lora_recipe(file);recipe['task_type']=TaskType.CAUSAL_LM
        target=get_peft_model(deepcopy(tiny_model.target_model),LoraConfig(**recipe))
        if index==0:target.save_pretrained(shared)
        target.load_adapter(shared,adapter_name='default')
        audit=module(ROOT.parent/repository/'helper/shared_adapter.py')
        proof=audit.verify_loaded_adapter(target,shared)
        hashes.append(proof['loaded_tensor_sha256'])
        with torch.no_grad():next(p for n,p in target.named_parameters() if 'lora_A' in n).add_(1)
        with pytest.raises(ValueError,match='differs at'):audit.verify_loaded_adapter(target,shared)
    assert len(set(hashes))==1


@pytest.mark.parametrize('repository',['MedusaGRPO','SpecNaacl','puregrpo'])
def test_official_benchmark_requires_checkpoint(repository,monkeypatch):
    monkeypatch.setenv('GRPO_BENCHMARK','1')
    audit=module(ROOT.parent/repository/'helper/shared_adapter.py')
    with pytest.raises(ValueError,match='Official benchmark requires TARGET_ADAPTER'):
        audit.preflight_shared_adapter('')


def test_fairness_checker_never_promotes_missing_or_mismatched_proofs():
    from scripts.check_fairness import audit
    base=audit(ROOT.parent/'SpecNaacl',ROOT.parent/'puregrpo',models=('qwen25_1p5b',))
    row=base['models'][0]
    assert row['checks']['loaded_initial_target_lora']['status']=='NOT VERIFIED'
    reports={}
    for method in row['methods']:
        reports[('qwen25_1p5b',method)]={'initialization':{
            'target_lora':{'status':'PASS','loaded_tensor_sha256':'equal'},
            'draft':{'status':'PASS','loaded_tensor_sha256':'pair'},'alignment_version':'response_rows_v2'},
            'target_optimizer_steps':2}
    result=audit(ROOT.parent/'SpecNaacl',ROOT.parent/'puregrpo',models=('qwen25_1p5b',),runtime_reports=reports)
    assert result['models'][0]['checks']['loaded_initial_target_lora']['status']=='PASS'
    assert result['status']=='FAIL' # Known length/precision differences remain visible.
    reports[('qwen25_1p5b','puregrpo')]['initialization']['target_lora']['loaded_tensor_sha256']='wrong'
    result=audit(ROOT.parent/'SpecNaacl',ROOT.parent/'puregrpo',models=('qwen25_1p5b',),runtime_reports=reports)
    assert result['models'][0]['checks']['loaded_initial_target_lora']['status']=='FAIL'


def test_heterogeneous_eos_rounds_use_individual_budgets(monkeypatch):
    from torch import nn
    from medusa.model import MedusaModel
    class Backbone(nn.Module):
        def forward(self,input_ids,past_key_values=None,**kwargs):
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
    def heads(anchor,lm_head):
        features=torch.nn.functional.one_hot((anchor.argmax(-1)[:,None]+torch.arange(1,4)[None,:])%8,8).float()
        return lm_head(features),features
    monkeypatch.setattr(model.draft_model,'logits',heads)
    x=torch.tensor([[1,1,5],[0,1,2],[0,0,0]]);mask=torch.tensor([[1,1,1],[0,1,1],[0,0,1]])
    outputs=[]
    for method in ('medusa','medusa_reflex'):
        result=speculative_generate(model,x,mask,SimpleNamespace(eos_token_id=7),method=method,max_length=8)
        assert result['generated_token_ids']==[[6,7],[3,4,5,6,7],[1,2,3,4,5,6,7]]
        assert result['response_generated_tokens']==[2,5,7]
        outputs.append(result['generated_token_ids'])
    assert outputs[0]==outputs[1]
