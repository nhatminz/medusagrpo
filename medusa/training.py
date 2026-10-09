"""One common online/pretrain objective for Medusa and Medusa+Reflex."""
import torch
from tqdm.auto import tqdm
import torch.distributed as dist
import os
from torch.nn import functional as F


def _training_rows(states,ids,prompt_lengths,shift,rows):
    """Pack only supervised anchors into bounded chunks, without padding.

    Keep the original per-response denominator even when a response spans
    several chunks. No full-response hidden-state copy is needed.
    """
    hidden_parts=[];label_parts=[];weight_parts=[];used=0
    for r in sorted(range(len(ids)),key=lambda i:ids[i].shape[0]):
        first=max(0,prompt_lengths[r]-shift)
        end=max(0,ids[r].shape[0]-shift)
        denominator=max(end-first,1)
        while first<end:
            take=min(rows-used,end-first)
            hidden_parts.append(states[r][first:first+take].detach())
            label_parts.append(ids[r][first+shift:first+shift+take])
            weight_parts.append(states[r].new_full((take,),1./denominator,dtype=torch.float32))
            used+=take;first+=take
            if used==rows:
                yield torch.cat(hidden_parts),torch.cat(label_parts),torch.cat(weight_parts)
                hidden_parts=[];label_parts=[];weight_parts=[];used=0
    if used:yield torch.cat(hidden_parts),torch.cat(label_parts),torch.cat(weight_parts)


def train_heads(model,outputs,prompt_mask,*,repeated_generate_nums=1,
                draft_accumulation_steps=1,chunk_size=128,**unused):
    states=outputs['all_draft_input_states'];ids=outputs['all_draft_input_ids']
    losses=torch.zeros(3,device=model.device)
    if not ids:raise ValueError('online head training needs at least one response')
    # Generation already supplied these host lengths in its scheduling packet.
    # Direct callers need one bulk transfer, never one GPU scalar read/response.
    lengths=outputs.get('prompt_lengths')
    if lengths is None:lengths=prompt_mask.sum(-1).cpu().tolist()
    prompt_lengths=[int(lengths[r//repeated_generate_nums]) for r in range(len(ids))]
    row_budget=int(os.environ.get('MEDUSA_HEAD_LOGIT_ROWS',str(chunk_size)))
    if row_budget<3:raise ValueError('MEDUSA_HEAD_LOGIT_ROWS must be at least 3')
    per_head=max(1,row_budget//3)
    streams=[iter(_training_rows(states,ids,prompt_lengths,h+2,per_head)) for h in range(3)]
    counts=[sum(max(0,ids[r].shape[0]-max(prompt_lengths[r],h+2)) for r in range(len(ids))) for h in range(3)]
    # The stored hidden states came from the same target verification. There is
    # no teacher transformer pass for either online heads or Reflex.
    rank=dist.get_rank() if dist.is_initialized() else 0
    steps=max((n+per_head-1)//per_head for n in counts)
    progress=tqdm(range(steps),desc='Online Medusa heads',unit='chunk',leave=False,
        disable=rank!=0 or steps<32 or os.environ.get('TQDM_DISABLE')=='1',mininterval=1.)
    for _ in progress:
        features=[];labels=[];weights=[];parts=[]
        for h,stream in enumerate(streams):
            batch=next(stream,None)
            if batch is None:continue
            x,target,weight=batch
            # Concatenation creates normal tensors from inference-mode storage.
            features.append(model.draft_model.heads[h](x))
            labels.append(target);weights.append(weight);parts.append((h,target.numel()))
        projected=torch.cat(features).to(model.lm_head.weight.dtype)
        logits=F.linear(projected,model.lm_head.weight.detach(),
            None if model.lm_head.bias is None else model.lm_head.bias.detach()).float()
        ce=F.cross_entropy(logits,torch.cat(labels),reduction='none')*torch.cat(weights)
        offset=0;objective=0.
        for h,n in parts:
            loss=ce[offset:offset+n].sum()/len(ids)
            losses[h]+=loss.detach()
            objective=objective+loss*(.8**h/sum(.8**j for j in range(3)))
            offset+=n
        (objective/draft_accumulation_steps).backward()
    # Every rank must participate in the same head gradient collectives, also
    # when a short response leaves an entire future horizon without labels.
    for head in model.draft_model.heads:
        for parameter in head.parameters():
            if parameter.grad is None:parameter.grad=torch.zeros_like(parameter)
    model.last_head_losses=losses.cpu().tolist() # at training boundary, never a verification round
    model.last_head_supervised_tokens=counts
    model.last_head_gradient_batches=[int(n>0) for n in counts]
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
