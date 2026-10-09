"""Parallel independent Medusa heads, ported from FlashGRPO.

Only residual Linear(LayerNorm) + SiLU transforms are owned by the head
optimizer. The shared output projection is supplied by the caller and detached
in the head objective, even when target LoRA parameters require gradients.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from helper.opd_reflex import initialize_projector


class PredictionHead(nn.Module):
    def __init__(self, hidden, dtype=None):
        super().__init__()
        self.fc = nn.Linear(hidden, hidden, dtype=dtype)
        self.norm = nn.LayerNorm(hidden, dtype=dtype)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = x.to(self.fc.weight.dtype)
        return x + F.silu(self.fc(self.norm(x)))


class MedusaHeads(nn.Module):
    num_heads = 3

    def __init__(self, hidden, vocab, dtype=None, lm_head=None, rank=8):
        super().__init__()
        self.hidden_size, self.vocab_size = hidden, vocab
        self.heads = nn.ModuleList([PredictionHead(hidden, dtype) for _ in range(3)])
        self.opd_projector = nn.Parameter(initialize_projector(
            hidden, rank, head=None if lm_head is None else lm_head.weight))
        self.register_buffer('opd_projector_grad_sum', torch.zeros(hidden, rank), persistent=False)
        self.register_buffer('opd_projector_grad_weight', torch.zeros(1), persistent=False)

    def project(self, hidden):
        # One shared LM-head GEMM for all three heads. Backbone states are frozen.
        return torch.stack([head(hidden.detach()) for head in self.heads], dim=1)

    def logits(self, hidden, lm_head):
        features = self.project(hidden)
        return F.linear(features.to(lm_head.weight.dtype), lm_head.weight.detach(),
                        None if lm_head.bias is None else lm_head.bias.detach()), features

    def loss_chunks(self, hidden, ids, attention, supervision, lm_head, chunk_size=128):
        """Yield bounded-vocabulary CE sums, with label offsets 2, 3, 4.

        h[t] predicts t+1 through the target head. Medusa head k therefore
        predicts t+k+1. Require an unbroken, real-token interval and supervise
        the future label's assistant/generated mask, including the last head.
        Counts are computed before chunks so chunk size cannot alter weighting.
        """
        counts, masks = [], []
        for k in range(3):
            shift = k + 2
            n = max(0, ids.shape[1] - shift)
            valid = supervision[:, shift:].bool().clone()
            for offset in range(shift + 1):
                valid &= attention[:, offset:offset+n].bool()
            masks.append(valid)
            counts.append(valid.sum().clamp_min(1))
        for k, head in enumerate(self.heads):
            shift = k + 2
            n = max(0, ids.shape[1] - shift)
            for start in range(0, n, chunk_size):
                end = min(start+chunk_size, n)
                features = head(hidden[:, start:end].detach())
                logits = F.linear(features.to(lm_head.weight.dtype), lm_head.weight.detach(),
                                  None if lm_head.bias is None else lm_head.bias.detach()).float()
                labels = ids[:, start+shift:end+shift].masked_fill(~masks[k][:, start:end], -100)
                ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
                                     ignore_index=-100, reduction='sum') / counts[k]
                yield k, ce, (0.8**k / sum(0.8**h for h in range(3)))


class MedusaModel(nn.Module):
    def __init__(self, config, target_model, path=None):
        super().__init__()
        self.target_model = target_model
        self.draft_model = MedusaHeads(config.hidden_size, config.vocab_size,
            target_model.dtype, target_model.lm_head, int(__import__('os').environ.get('OPD_RANK', '8')))
        self.last_head_losses = [0., 0., 0.]

    @property
    def lm_head(self):
        target = self.target_model.get_base_model() if hasattr(self.target_model, 'get_base_model') else self.target_model
        return target.lm_head

    @property
    def embed_tokens(self):
        return self.target_model.get_input_embeddings()

    @property
    def device(self): return self.lm_head.weight.device
    @property
    def dtype(self): return self.lm_head.weight.dtype
    @property
    def opd_projector(self): return self.draft_model.opd_projector
    @property
    def opd_projector_grad_sum(self): return self.draft_model.opd_projector_grad_sum
    @property
    def opd_projector_grad_weight(self): return self.draft_model.opd_projector_grad_weight

    def enable_opd(self, rank):
        if rank != self.opd_projector.shape[1]: raise ValueError('checkpoint OPD rank mismatch')

    def get_opd_projector(self, rank):
        self.enable_opd(rank)
        return self.opd_projector

    @torch.no_grad()
    def apply_opd_projector_gradient(self):
        self.opd_projector.grad = (self.opd_projector_grad_sum / self.opd_projector_grad_weight.clamp_min(1)).clone()
        self.opd_projector_grad_sum.zero_()
        self.opd_projector_grad_weight.zero_()

    @torch.no_grad()
    def load_opd_projector(self, value): self.opd_projector.copy_(value)

    def load_model(self, path):
        path = Path(path)
        if path.is_dir(): path = path/'draft.pth'
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state.get('architecture') != 'medusa_parallel_3':
            raise ValueError('expected MedusaGRPO parallel-three-head checkpoint')
        self.draft_model.load_state_dict(state['draft_model'], strict=True)

    def save_model(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name+'.tmp')
        torch.save(dict(architecture='medusa_parallel_3', draft_model=self.draft_model.state_dict()), tmp)
        tmp.replace(path)
