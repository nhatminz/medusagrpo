"""Native Medusa/SDPA/cache compatibility probe."""
def probe_training_runtime(device='cpu'):
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from medusa.model import MedusaModel
    from medusa.generate import speculative_generate
    from types import SimpleNamespace
    config=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1)
    target=Qwen2ForCausalLM(config).to(device).eval()
    model=MedusaModel(config,target).to(device)
    out=speculative_generate(model,torch.tensor([[1,2,3]],device=device),
        torch.ones(1,3,device=device,dtype=torch.long),SimpleNamespace(eos_token_id=31),
        max_length=5,do_sample=False,opd_backend='auto')
    return dict(architecture='medusa_parallel_3',target_forward_calls=out['target_forward_calls'])
