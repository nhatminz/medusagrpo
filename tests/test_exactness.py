from types import SimpleNamespace
import pytest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM,Qwen3Config,Qwen3ForCausalLM,LlamaConfig,LlamaForCausalLM
from helper.opd_static_cache import OPDStaticCache
from medusa.tree import build_sparse_tree,plan_tree
from medusa.generate import compact_kv,speculative_generate


@pytest.mark.parametrize('config_cls,model_cls',[(Qwen2Config,Qwen2ForCausalLM),(Qwen3Config,Qwen3ForCausalLM),(LlamaConfig,LlamaForCausalLM)])
def test_tree_attention_and_extracted_kv_match_recompute(config_cls,model_cls):
    torch.manual_seed(4)
    config=config_cls(vocab_size=23,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                     num_attention_heads=2,num_key_value_heads=1)
    target=model_cls(config).eval()
    ids=torch.tensor([[[5,6,7,8],[9,10,11,12],[13,14,15,16]]])
    probs=torch.tensor([[[.7,.15,.1,.05]]*3])
    tree=build_sparse_tree(torch.tensor([4]),ids,probs,plan_tree(1,12,12))
    prompt=torch.tensor([[1,2,3]]);cache=OPDStaticCache(32)
    with torch.no_grad():
        target.model(input_ids=prompt,past_key_values=cache,use_cache=True)
        verified=target.model(input_ids=tree.tokens,past_key_values=cache,use_cache=True,
            position_ids=3+tree.depths,attention_mask=tree.attention_mask(3,torch.float32)).last_hidden_state
        for node in range(tree.tokens.shape[1]):
            path=[node];parent=int(tree.parents[0,node])
            while parent>=0:path.append(parent);parent=int(tree.parents[0,parent])
            path.reverse()
            seq=torch.cat((prompt,tree.tokens[:,path]),1)
            expected=target.model(input_ids=seq,use_cache=False).last_hidden_state[:,-1]
            torch.testing.assert_close(verified[:,node],expected,atol=3e-6,rtol=3e-5)
        chosen=torch.tensor([[0,1,2,3]])
        compact_kv(cache,chosen,3,4)
        actual=target.model(input_ids=torch.tensor([[17]]),past_key_values=cache,use_cache=True).last_hidden_state
        expected=target.model(input_ids=torch.cat((prompt,tree.tokens[:,chosen[0]],torch.tensor([[17]])),1),use_cache=False).last_hidden_state[:,-1:]
        torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-5)


def test_disabled_and_zero_b_rng_identity_no_extra_target_forward(tiny_model):
    m=tiny_model;inputs=torch.tensor([[1,2,3],[0,1,4]]);mask=torch.tensor([[1,1,1],[0,1,1]])
    counts=[];hook=m.target_model.model.register_forward_hook(lambda *_:counts.append(1))
    def run(method,**kw):
        counts.clear();torch.manual_seed(42)
        out=speculative_generate(m,inputs,mask,SimpleNamespace(eos_token_id=22),do_sample=True,
             repeated_generate_nums=2,max_length=15,method=method,**kw)
        assert len(counts)==out['target_forward_calls']==out['verification_batches']+1
        return out
    a=run('medusa');b=run('medusa_reflex',opd_enabled=False);c=run('medusa_reflex',opd_fast_lr=0)
    assert a['generated_token_ids']==b['generated_token_ids']==c['generated_token_ids']
    assert a['target_forward_calls']==b['target_forward_calls']==c['target_forward_calls']
    d=run('medusa_reflex',opd_train_projector=True)
    assert d['opd_updates']>0
    assert all(d[f'opd_head{h}_frontier_states']<=2*d['active_response_rounds'] for h in (1,2,3))
    hook.remove()


def test_variable_accepted_paths_mask_padding_kv(tiny_model):
    target=tiny_model.target_model
    prompt=torch.tensor([[1,2,3],[1,2,3]])
    ids=torch.tensor([[[5,6,7,8],[9,10,11,12],[13,14,15,16]]]*2)
    q=torch.tensor([[[.7,.15,.1,.05]]*3]*2)
    tree=build_sparse_tree(torch.tensor([4,4]),ids,q,plan_tree(2,24,12))
    cache=OPDStaticCache(32)
    with torch.no_grad():
        target.model(input_ids=prompt,past_key_values=cache,use_cache=True)
        target.model(input_ids=tree.tokens,past_key_values=cache,use_cache=True,
            position_ids=3+tree.depths,attention_mask=tree.attention_mask(3,torch.float32))
        compact_kv(cache,torch.tensor([[0,1,2],[0,-1,-1]]),3,3)
        actual=target.model(input_ids=torch.tensor([[17],[18]]),past_key_values=cache,use_cache=True,
            attention_mask=torch.tensor([[1,1,1,1,1,1,1],[1,1,1,1,0,0,1]]),
            position_ids=torch.tensor([[6],[4]])).last_hidden_state
        for row,chosen in enumerate(([0,1,2],[0])):
            seq=torch.cat((prompt[row:row+1],tree.tokens[row:row+1,chosen],torch.tensor([[17+row]])),1)
            expected=target.model(input_ids=seq,use_cache=False).last_hidden_state[:,-1:]
            torch.testing.assert_close(actual[row:row+1],expected,atol=3e-6,rtol=3e-5)


def test_sparse_target_distribution_monte_carlo(tiny_model,monkeypatch):
    # Two-token joint target distribution, many independent responses. Statistical
    # coverage complements deterministic prefix/KV checks; no acceptance forcing.
    n=12000;monkeypatch.setenv('CPEAK_NODES',str(n*4));monkeypatch.setenv('MAX_TREE_NODES_PER_SEQ','4')
    m=tiny_model;prompt=torch.tensor([[1,2,3]])
    with torch.no_grad():
        p=m.target_model(prompt).logits[:,-1].float().div(.7).softmax(-1)[0]
        expected=[]
        for token in range(23):
            seq=torch.cat((prompt,torch.tensor([[token]])),1)
            expected.append(p[token]*m.target_model(seq).logits[:,-1].float().div(.7).softmax(-1)[0])
        expected=torch.stack(expected);expected[22]=0 # prefill EOS stops
    torch.manual_seed(133)
    prompt=prompt.expand(n,-1).clone()
    out=speculative_generate(m,prompt,torch.ones_like(prompt),SimpleNamespace(eos_token_id=22),
        repeated_generate_nums=1,do_sample=True,temperature=.7,top_p=1.,max_length=5)
    empirical=torch.zeros(23,23)
    for seq in out['generated_token_ids']:
        if len(seq)>1:empirical[seq[0],seq[1]]+=1/n
    # Compare marginals and low-dimensional joint moments without noisy 529-bin assertions.
    assert (empirical.sum(1)-expected.sum(1)).abs().max()<.008
    assert (empirical.sum(0)-expected.sum(0)).abs().max()<.008
    assert abs(float(empirical[:12,:12].sum()-expected[:12,:12].sum()))<.015


def test_eos_and_rejected_sample_become_pending_root(tiny_model,monkeypatch):
    import medusa.generate as generation
    seen=[]
    native=generation.build_sparse_tree
    def build(root,*a,**kw):seen.append(root.clone());return native(root,*a,**kw)
    monkeypatch.setattr(generation,'build_sparse_tree',build)
    calls=[]
    def sampler(logits,**kw):
        # First token 20; proposal topk never needs to contain sampled token 21.
        token=20 if not calls else (21 if len(calls)==1 else 22)
        calls.append(1)
        samples=torch.full(logits.shape[:-1],token,dtype=torch.long)
        return samples,None,None
    monkeypatch.setattr(generation,'sample_target_with_metadata',sampler)
    out=speculative_generate(tiny_model,torch.tensor([[1,2,3]]),torch.ones(1,3,dtype=torch.long),
        SimpleNamespace(eos_token_id=22),max_length=10)
    assert out['generated_token_ids']==[[20,21,22]]
    assert seen[0].tolist()==[20] and seen[1].tolist()==[21]
    assert out['target_forward_calls']==3


def test_head3_participates_in_real_generation(monkeypatch):
    from torch import nn
    from medusa.model import MedusaModel
    class Backbone(nn.Module):
        def forward(self,input_ids,past_key_values=None,use_cache=True,**kwargs):
            hidden=torch.nn.functional.one_hot(input_ids,8).float()
            if past_key_values is not None and use_cache:
                past_key_values.update(hidden[:,None],hidden[:,None],0)
            return SimpleNamespace(last_hidden_state=hidden)
    class Target(nn.Module):
        def __init__(self):
            super().__init__();self.model=Backbone();self.lm_head=nn.Linear(8,8,bias=False)
            with torch.no_grad():
                self.lm_head.weight.fill_(-8)
                for token in range(8):self.lm_head.weight[(token+1)%8,token]=8
        @property
        def dtype(self):return torch.float32
    m=MedusaModel(SimpleNamespace(hidden_size=8,vocab_size=8),Target())
    def accurate_heads(anchor,lm_head):
        features=torch.nn.functional.one_hot((anchor.argmax(-1)[:,None]+torch.arange(1,4)[None,:])%8,8).float()
        return lm_head(features),features
    monkeypatch.setattr(m.draft_model,'logits',accurate_heads)
    monkeypatch.setenv('MAX_TREE_NODES_PER_SEQ','4')
    out=speculative_generate(m,torch.tensor([[0]]),torch.ones(1,1,dtype=torch.long),SimpleNamespace(eos_token_id=7),
        do_sample=False,max_length=10)
    assert out['generated_token_ids']==[[1,2,3,4,5,6,7]]
    assert out['head3_accepted_tokens']==1
    assert out['target_forward_calls']==3
