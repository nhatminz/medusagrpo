"""Batch stopping follows actual SpecNaacl expressions; native Medusa KV paths."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from medusa.model import MedusaModel
from medusa.generate import speculative_generate,resolve_length_policy
from medusa.tree import build_sparse_tree,plan_tree,select_feedback
from helper.tree_verification import trace_verified_path

ROOT=Path(__file__).resolve().parents[1]
DEVICES=['cpu']+(['cuda'] if torch.cuda.is_available() else [])


def source_stop(prompt_lengths,counts,width,max_length):
    """Evaluate both actual baseline real-length assignments, without importing CLI."""
    result=[]
    for name in ('fastgrpo_generate','opd_generate'):
        file=ROOT.parent/'SpecNaacl/helper'/f'{name}.py'
        tree=ast.parse(file.read_text())
        assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id=='real_sequences_length' for t in n.targets))
        # Physical row width may exceed every actual response length after
        # accepted-path padding; source subtracts BOTH prompt and round padding.
        physical=max(counts)
        pads=[width-length+physical-count for length,count in zip(prompt_lengths,counts)]
        scope=dict(input_ids=torch.empty(len(counts),width),
            generated_sequences=[[0]*physical for _ in counts],
            padding_positions=[set(range(n)) for n in pads],
            history=SimpleNamespace(lengths={'generated_ids':physical}),pad_counts=pads)
        exec(compile(ast.Module(body=[assignment],type_ignores=[]),str(file),'exec'),scope)
        result.append(scope['real_sequences_length']>=max_length)
    assert result[0]==result[1]
    return result[0]


class CycleBackbone(nn.Module):
    def forward(self,input_ids,past_key_values=None,**kwargs):
        hidden=torch.nn.functional.one_hot(input_ids,32).float()
        if past_key_values is not None:past_key_values.update(hidden[:,None],hidden[:,None],0)
        return SimpleNamespace(last_hidden_state=hidden)


class CycleTarget(nn.Module):
    def __init__(self):
        super().__init__();self.model=CycleBackbone();self.lm_head=nn.Linear(32,32,bias=False)
        with torch.no_grad():
            self.lm_head.weight.fill_(-8)
            for token in range(32):self.lm_head.weight[(token+1)%32,token]=8
    @property
    def dtype(self):return torch.float32


def cycle(monkeypatch,device,accepted=True):
    model=MedusaModel(SimpleNamespace(hidden_size=32,vocab_size=32),CycleTarget()).to(device)
    def heads(anchor,lm_head):
        if accepted:
            tokens=(anchor.argmax(-1)[:,None]+torch.arange(1,4,device=device)[None,:])%32
        else:tokens=torch.full((len(anchor),3),17,device=device)
        features=torch.nn.functional.one_hot(tokens,32).float()
        return lm_head(features),features
    monkeypatch.setattr(model.draft_model,'logits',heads)
    monkeypatch.setenv('MAX_TREE_NODES_PER_SEQ','4')
    return model


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('method',['medusa','medusa_reflex'])
@pytest.mark.parametrize('accepted',[True,False])
@pytest.mark.parametrize('width,limit',[(7,8),(7,10),(8,8),(9,8),(2047,2048)])
def test_batch_round_limit_matches_actual_source(monkeypatch,device,method,accepted,width,limit):
    model=cycle(monkeypatch,device,accepted)
    prompts=[width,width-2,width-4]
    x=torch.zeros(3,width,dtype=torch.long,device=device);mask=torch.zeros_like(x)
    for r,n in enumerate(prompts):mask[r,-n:]=1
    out=speculative_generate(model,x,mask,SimpleNamespace(eos_token_id=31),method=method,
        max_length=limit,return_all_draft_input=True)
    advance=4 if accepted else 1
    counts=[1]*3;rounds=0
    # No prefill boundary check: source ALWAYS executes the first round.
    while True:
        counts=[n+advance for n in counts];rounds+=1
        if source_stop(prompts,counts,width,limit):break
    assert out['response_generated_tokens']==counts
    assert out['response_verification_rounds']==[rounds]*3
    assert out['target_forward_calls']==rounds+1
    assert out['medusa_scheduling_syncs']==rounds+1
    assert out['generation_length_policy']=='specnaacl_compatible'
    for r,seq in enumerate(out['generated_token_ids']):assert seq==list(range(1,counts[r]+1))
    for r,seq in enumerate(out['all_draft_input_ids']):
        assert len(seq)==prompts[r]+counts[r]
        assert seq.tolist()==[0]*prompts[r]+list(range(1,counts[r]+1))


@pytest.mark.parametrize('device',DEVICES)
def test_newly_eos_long_row_still_stops_batch_before_pruning(monkeypatch,device):
    model=cycle(monkeypatch,device)
    width=7;x=torch.zeros(2,width,dtype=torch.long,device=device);x[0,-1]=28
    mask=torch.tensor([[1]*7,[0]*4+[1]*3],device=device)
    out=speculative_generate(model,x,mask,SimpleNamespace(eos_token_id=31),max_length=10)
    assert out['generated_token_ids']==[[29,30,31],[1,2,3,4,5]]
    assert source_stop([7,3],[3,5],7,10)
    assert out['response_verification_rounds']==[1,1]


@pytest.mark.parametrize('device',DEVICES)
def test_eos_pruning_then_pending_bonus_continues_shorter_prompt(monkeypatch,device):
    model=cycle(monkeypatch,device)
    x=torch.tensor([[0,0,0,28],[0,0,0,0]],device=device)
    mask=torch.tensor([[1,1,1,1],[0,0,1,1]],device=device)
    out=speculative_generate(model,x,mask,SimpleNamespace(eos_token_id=31),max_length=9)
    assert not source_stop([4,2],[3,5],4,9)
    assert out['generated_token_ids']==[[29,30,31],list(range(1,10))]
    assert out['response_verification_rounds']==[1,2]
    assert source_stop([2],[9],4,9)
    assert out['head3_accepted_tokens']==2


def test_prefill_eos_historical_source_round_and_output_filter(monkeypatch):
    model=cycle(monkeypatch,'cpu',accepted=False)
    x=torch.tensor([[30]])
    # Source starts end_sig=0 regardless of prefill EOS. Output still truncates
    # at its first EOS, even when the internal history contains later samples.
    out=speculative_generate(model,x,torch.ones_like(x),SimpleNamespace(eos_token_id=31),max_length=3)
    assert out['generated_token_ids']==[[31]]
    assert out['response_verification_rounds']==[1]
    assert out['target_forward_calls']==2
    strict=speculative_generate(model,x,torch.ones_like(x),SimpleNamespace(eos_token_id=31),max_length=3,
        generation_length_policy='per_response')
    assert strict['generated_token_ids']==[[31]] and strict['target_forward_calls']==1


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('policy',['specnaacl_compatible','per_response'])
def test_native_kv_mixed_lengths_policy_and_disabled_rng_parity(tiny_model,device,policy):
    model=tiny_model.to(device)
    with torch.no_grad():model.target_model.lm_head.weight.zero_()
    x=torch.tensor([[1,1,1,1,1],[0,0,1,1,1]],device=device)
    mask=torch.tensor([[1]*5,[0,0,1,1,1]],device=device)
    outputs=[]
    for method,kw in [('medusa',{}),('medusa_reflex',{'opd_enabled':False}),('medusa_reflex',{'opd_fast_lr':0})]:
        torch.manual_seed(49)
        out=speculative_generate(model,x,mask,SimpleNamespace(eos_token_id=22),method=method,
            max_length=6,do_sample=True,temperature=.8,top_p=.95,return_all_draft_input=True,
            generation_length_policy=policy,**kw)
        rng=torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state()
        outputs.append((out['generated_token_ids'],rng))
        for ids,states in zip(out['all_draft_input_ids'],out['all_draft_input_states']):
            with torch.inference_mode(),torch.autocast(device,dtype=torch.float16,enabled=device=='cuda'):
                expected=model.target_model.model(input_ids=ids[None],attention_mask=torch.ones_like(ids[None])).last_hidden_state[0]
            torch.testing.assert_close(states[:-1],expected[:-1],rtol=3e-3 if device=='cuda' else 2e-5,
                                       atol=3e-3 if device=='cuda' else 2e-5)
    assert outputs[0][0]==outputs[1][0]==outputs[2][0]
    assert torch.equal(outputs[0][1],outputs[1][1]) and torch.equal(outputs[0][1],outputs[2][1])


def test_policy_env_default_and_validation(monkeypatch):
    monkeypatch.delenv('GENERATION_LENGTH_POLICY',raising=False)
    assert resolve_length_policy()=='specnaacl_compatible'
    monkeypatch.setenv('GENERATION_LENGTH_POLICY','per_response')
    assert resolve_length_policy()=='per_response'
    assert resolve_length_policy('specnaacl_compatible')=='specnaacl_compatible'
    with pytest.raises(ValueError,match='unsupported'):resolve_length_policy('wrong')


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('active',[32,64,128])
def test_three_head_paths_and_frontier_caps_at_production_concurrency(device,active):
    ids=torch.tensor([[[2,5,6,7],[3,8,9,10],[4,11,12,13]]],device=device).expand(active,-1,-1).contiguous()
    q=torch.tensor([[[.7,.15,.1,.05]]*3],device=device).expand(active,-1,-1).contiguous()
    plan=plan_tree(active,512,12)
    tree=build_sparse_tree(torch.ones(active,dtype=torch.long,device=device),ids,q,plan)
    assert active*tree.tokens.shape[1]<=512
    assert plan.active_heads==3
    assert all((tree.depths==h).any(1).all() for h in (1,2,3))
    samples=torch.full_like(tree.tokens,21)
    samples[:,0]=2;samples[:,1]=3;samples[:,2]=4
    path=trace_verified_path(tree,samples,22)
    assert torch.equal(path.lengths,torch.full((active,),4,device=device))
    assert torch.equal(path.packed_indices,torch.tensor([[0,1,2,3]],device=device).expand(active,-1))
    for h in range(3):
        weights,kind=select_feedback(tree,path,h)
        assert (kind==1).sum()==active # one distinct visited teacher per head/response
        assert ((kind==2).sum(1)<=2).all()
        assert torch.equal(weights>0,kind>0)
        assert not ((weights>0)&(tree.depths!=h)).any()


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('active',[32,64,128])
def test_real_generation_head_and_opd_counters(monkeypatch,device,active):
    from medusa.opd import MedusaOPD
    model=cycle(monkeypatch,device)
    monkeypatch.setenv('MAX_TREE_NODES_PER_SEQ','12')
    monkeypatch.setenv('CPEAK_NODES','512')
    monkeypatch.setenv('OPD_SELECTION','visited_capped_frontier')
    monkeypatch.setenv('OPD_MAX_FRONTIER_PER_HEAD','2')
    observations=[];native=MedusaOPD.feedback
    def observe(engine,tree,path,*args,**kwargs):
        # Test-only readback: production feedback/scheduling remains untouched.
        row=[]
        assert tree.tokens.numel()<=512
        for h in range(3):
            w,kind=select_feedback(tree,path,h)
            assert ((kind==2).sum(1)<=2).all()
            row.append(((tree.depths==h+1).sum().item(),
                        ((path.lengths-1)>h).sum().item(),
                        (kind==1).sum().item(),(kind==2).sum().item(),(w>0).sum().item()))
        observations.append(row)
        return native(engine,tree,path,*args,**kwargs)
    monkeypatch.setattr(MedusaOPD,'feedback',observe)
    calls=[];hook=model.target_model.model.register_forward_hook(lambda *_:calls.append(1))
    x=torch.zeros(active//8,3,dtype=torch.long,device=device)
    out=speculative_generate(model,x,torch.ones_like(x),SimpleNamespace(eos_token_id=31),
        repeated_generate_nums=8,max_length=10,method='medusa_reflex')
    hook.remove()
    assert out['target_forward_calls']==len(calls)==len(observations)+1
    assert out['medusa_scheduling_syncs']==len(observations)+1
    for h in range(3):
        proposed,accepted,visited,frontier,selected=map(sum,zip(*(r[h] for r in observations)))
        assert proposed>0 and accepted>0 and visited>0
        assert out[f'head{h+1}_proposed_nodes']==out[f'head{h+1}_verified_tokens']==proposed
        assert out[f'head{h+1}_accepted_tokens']==accepted
        assert out[f'opd_head{h+1}_visited_states']==visited
        assert out[f'opd_head{h+1}_frontier_states']==frontier
        assert out[f'opd_head{h+1}_selected_states']==selected==visited+frontier


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('verification',[False,True])
@pytest.mark.parametrize('top_p,top_k',[(None,None),(.95,None),(.95,5)])
def test_actual_fastgrpo_bf16_sampler_probabilities_and_rng(device,verification,top_p,top_k):
    import warnings
    from helper.opd_sampling import sample_target_with_metadata
    source=ROOT.parent/'SpecNaacl/helper/fastgrpo_generate.py'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='sampling')
    captured={}
    class TorchProbe:
        def __getattr__(self,name):return getattr(torch,name)
        def multinomial(self,probs,**kw):
            captured['probs']=probs.clone()
            return torch.multinomial(probs,**kw)
    scope={'torch':TorchProbe(),'F':torch.nn.functional,'warnings':warnings}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    torch.manual_seed(51)
    logits=torch.randn(3,4,32,device=device,dtype=torch.bfloat16)
    with torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda' and verification):
        torch.manual_seed(91)
        expected=scope['sampling'](logits,top_k=top_k,top_p=top_p,temperature=.8,eos_token_id=31)
        expected_rng=torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state()
        torch.manual_seed(91)
        actual,probs,_=sample_target_with_metadata(logits,do_sample=True,top_k=top_k,top_p=top_p,
            temperature=.8,eos_token_id=31,return_probs=True,mode='finite')
        actual_rng=torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state()
    assert torch.equal(actual,expected) and torch.equal(actual_rng,expected_rng)
    assert torch.equal(probs.flatten(0,1),captured['probs'])
    assert probs.dtype==(torch.float32 if device=='cuda' and verification else torch.bfloat16)


def test_config_layers_and_cli_defaults_are_actual(monkeypatch):
    from scripts.check_fairness import audit
    report=audit(ROOT.parent/'SpecNaacl',ROOT.parent/'puregrpo')
    requested={'qwen25_1p5b':('16','2'),'qwen25_14b':('4','8'),'qwen3_1p7b':('16','2')}
    assert report['configuration_status']=='PASS'
    for row in report['models']:
        assert row['configuration_status']=='PASS'
        if row['model'] in requested:
            config=row['model_config']['medusa']
            assert (config['BATCH_SIZE'],config['ACCUMULATION_STEPS'])==requested[row['model']]
        for method in ('medusa','medusa_reflex','fastgrpo','fastgrpo_reflex'):
            args=row['methods'][method]['parsed_arguments']
            assert (args['batch_size'],args['accumulation_steps'])==(8,4)
            assert args['max_grpo_steps']==0 and args['train_split']=='train'
    # Removing the wrapper profile selects model defaults and is correctly
    # reported as a discrepancy with today's effective source benchmark.
    result=audit(ROOT.parent/'SpecNaacl',ROOT.parent/'puregrpo',models=('qwen25_1p5b',),env={'LAUNCHER_ENV':'/dev/null'})
    assert result['configuration_status']=='FAIL'
