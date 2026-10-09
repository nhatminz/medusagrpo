"""Small native-architecture smoke; never a real-weight/model OOM claim."""
from types import SimpleNamespace
import re
from pathlib import Path
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM,Qwen3Config,Qwen3ForCausalLM,LlamaConfig,LlamaForCausalLM
from medusa.model import MedusaModel
from medusa.generate import speculative_generate
from medusa.training import train_heads

ROOT=Path(__file__).resolve().parents[1]
MODELS=('qwen25_1p5b','qwen25_3b','qwen25_7b','qwen25_14b','qwen3_1p7b','qwen3_4b','llama31_8b')


def smoke(key,device):
    env=(ROOT/'configs'/key/'b200.env').read_text()
    family=re.search(r'MODEL_TYPE="\$\{MODEL_TYPE:-(\w+)\}"',env)[1]
    config_cls,model_cls={'qwen2':(Qwen2Config,Qwen2ForCausalLM),'qwen3':(Qwen3Config,Qwen3ForCausalLM),
                         'llama':(LlamaConfig,LlamaForCausalLM)}[family]
    torch.manual_seed(62)
    config=config_cls(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1)
    target=model_cls(config).to(device).eval()
    if device=='cuda':target.bfloat16()
    model=MedusaModel(config,target).to(device)
    x=torch.tensor([[1,2,3],[0,1,4]],device=device)
    mask=torch.tensor([[1,1,1],[0,1,1]],device=device)
    calls=[];hook=target.model.register_forward_hook(lambda *_:calls.append(1))
    def generate(method,**kw):
        calls.clear();torch.manual_seed(41)
        result=speculative_generate(model,x,mask,SimpleNamespace(eos_token_id=31),method=method,
            do_sample=True,repeated_generate_nums=2,max_length=9,return_all_draft_input=True,**kw)
        assert len(calls)==result['target_forward_calls']
        assert len(result['generated_token_ids'])==4
        return result
    baseline=generate('medusa');disabled=generate('medusa_reflex',opd_enabled=False)
    assert baseline['generated_token_ids']==disabled['generated_token_ids']
    active=generate('medusa_reflex')
    before=len(calls);train_heads(model,active,mask,repeated_generate_nums=2)
    assert len(calls)==before
    assert all(torch.isfinite(p.grad).all() for p in model.draft_model.heads.parameters())
    assert all(p.grad is None for p in target.parameters())
    assert len(model.last_head_losses)==3
    hook.remove()
