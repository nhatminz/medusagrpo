"""Reduce Medusa counters at the existing end-of-job logging boundary."""
import torch
import torch.distributed as dist


COUNTER_NAMES = (
    *(f'head{h}_{name}' for h in (1,2,3)
      for name in ('active_rounds','proposed_nodes','verified_tokens','accepted_tokens','supervised_tokens','gradient_batches','optimizer_updates')),
    *(f'opd_head{h}_{name}' for h in (1,2,3)
      for name in ('selected_states','visited_states','frontier_states','update_count')),
    'verification_nodes','target_forward_calls','opd_update_count',
)


def aggregate_medusa_totals(totals, device):
    values=torch.tensor([totals.get(name,0) for name in COUNTER_NAMES],
                        device=device,dtype=torch.int64)
    if dist.is_initialized():dist.all_reduce(values,op=dist.ReduceOp.SUM)
    result=dict(zip(COUNTER_NAMES,values.tolist()))
    for h in (1,2,3):
        result[f'head{h}_acceptance_rate']=(
            result[f'head{h}_accepted_tokens']/max(result[f'head{h}_proposed_nodes'],1))
    return result
