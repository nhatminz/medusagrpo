import pytest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from medusa.model import MedusaModel
from torch.nn.attention import sdpa_kernel, SDPBackend

torch.set_num_threads(1)


@pytest.fixture
def strict_cuda_reference():
    """Pin arithmetic only for tight FP32 KV oracles; restore caller settings.

    Production CUDA/autocast tests keep the default backend. This oracle must
    not compare different-shape forwards under TF32 or hardware SDPA dispatch.
    """
    matmul = torch.backends.cuda.matmul
    modern = hasattr(matmul, 'fp32_precision')
    previous = matmul.fp32_precision if modern else torch.get_float32_matmul_precision()
    try:
        if modern:
            matmul.fp32_precision = 'ieee'
        else:
            torch.set_float32_matmul_precision('highest')
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        if modern:
            matmul.fp32_precision = previous
        else:
            torch.set_float32_matmul_precision(previous)

@pytest.fixture
def tiny_model():
    torch.manual_seed(12)
    config=Qwen2Config(vocab_size=23,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1)
    return MedusaModel(config,Qwen2ForCausalLM(config).eval())
