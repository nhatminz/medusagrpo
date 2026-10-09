# Frozen pre-optimization implementation for loss/gradient comparisons.
"""One common online/pretrain objective for Medusa and Medusa+Reflex."""
import torch
from tqdm.auto import tqdm
import torch.distributed as dist


def train_heads(model,outputs,prompt_mask,*,repeated_generate_nums=1,
                draft_accumulation_steps=1,chunk_size=128,**unused):
    states=outputs['all_draft_input_states'];ids=outputs['all_draft_input_ids']
    losses=torch.zeros(3,device=model.device)
    # The stored hidden states came from the same target verification. There is
    # no teacher transformer pass for either online heads or Reflex.
    rank=dist.get_rank() if dist.is_initialized() else 0
    progress=tqdm(range(len(ids)),desc='Online Medusa heads',unit='response',leave=False,
        disable=rank!=0 or len(ids)<32 or __import__('os').environ.get('TQDM_DISABLE')=='1',mininterval=1.)
    for r in progress:
        x=states[r].clone().unsqueeze(0);tokens=ids[r].clone().unsqueeze(0)
        attention=torch.ones_like(tokens)
        supervision=torch.ones_like(tokens)
        prompt_length=int(prompt_mask[r//repeated_generate_nums].sum())
        supervision[:,:prompt_length]=0
        for h,ce,weight in model.draft_model.loss_chunks(x,tokens,attention,supervision,model.lm_head,chunk_size):
            losses[h]+=ce.detach()/len(ids)
            (ce*weight/(len(ids)*draft_accumulation_steps)).backward()
    # Every rank must participate in the same head gradient collectives, also
    # when a short response leaves an entire future horizon without labels.
    for head in model.draft_model.heads:
        for parameter in head.parameters():
            if parameter.grad is None:parameter.grad=torch.zeros_like(parameter)
    model.last_head_losses=losses.cpu().tolist() # at training boundary, never a verification round
    return sum((0.8**h/sum(0.8**j for j in range(3)))*loss for h,loss in enumerate(model.last_head_losses)),0.


def pretrain_loss(model,batch):
    batch={k:v.to(model.device) for k,v in batch.items()}
    with torch.no_grad():
        target=model.target_model.get_base_model() if hasattr(model.target_model,'get_base_model') else model.target_model
        hidden=target.model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                            use_cache=False).last_hidden_state
    losses=torch.zeros(3,device=model.device)
    for h,ce,weight in model.draft_model.loss_chunks(hidden,batch['input_ids'],batch['attention_mask'],
                                                 batch['loss_mask'],model.lm_head):
        losses[h]+=ce.detach()
        (ce*weight).backward()
    model.last_head_losses=losses.cpu().tolist()
    return losses
