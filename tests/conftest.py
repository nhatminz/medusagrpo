import pytest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from medusa.model import MedusaModel

torch.set_num_threads(1)

@pytest.fixture
def tiny_model():
    torch.manual_seed(12)
    config=Qwen2Config(vocab_size=23,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1)
    return MedusaModel(config,Qwen2ForCausalLM(config).eval())
