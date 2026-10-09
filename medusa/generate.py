"""Exact target-sampled sparse Medusa traversal with a pending bonus token.

A rejection sample is emitted immediately and becomes next round's root. Its
KV is computed as part of that verification, never a separate catch-up forward.
Every decision is sampled from target conditionals using SpecNaacl's sampler.
"""
import os
import time
import torch
from helper.opd_sampling import sample_target_with_metadata
from helper.opd_static_cache import persistent_cache
from helper.tree_verification import trace_verified_path
from helper.opd_attention import AttentionWorkspace
from medusa.tree import plan_tree,build_sparse_tree
from medusa.opd import MedusaOPD
TARGET_SAMPLER_MODE=os.environ.get('OPD_SAMPLER_MODE','finite')


def base_lm(model):
    return model.target_model.get_base_model() if hasattr(model.target_model,'get_base_model') else model.target_model


def compact_kv(cache, indices, past, width):
    """Gather only the verified suffix, keeping the accepted path's KV order."""
    safe=indices[:,:width].clamp_min(0)
    for layer in cache.layers:
        # Source is the verification suffix, so history is never copied here.
        for pool in (layer.key_pool,layer.value_pool):
            source=pool[:safe.shape[0],:,past:layer.length]
            selected=source.gather(2,safe[:,None,:,None].expand(-1,source.shape[1],-1,source.shape[3]))
            pool[:safe.shape[0],:,past:past+width].copy_(selected)
        layer.length=past+width


@torch.inference_mode()
def _speculative_generate(model,input_ids,attention_mask,tokenizer,do_sample=False,
        repeated_generate_nums=1,temperature=1.,top_p=.95,top_k=None,max_length=2048,
        return_all_draft_input=False,method='medusa',verification_capacity=512,
        opd_rank=8,opd_topk=16,opd_fast_lr=.01,opd_visited_weight=1.,opd_frontier_weight=1.,
        opd_update_stream=True,opd_profile=False,opd_diagnostics=False,opd_backend='auto',
        opd_train_projector=False,opd_enabled=None,**unused):
    if method not in ('medusa','medusa_reflex'):raise ValueError('unsupported Medusa method')
    enabled=method=='medusa_reflex' and (os.environ.get('OPD_ENABLED','1')=='1' if opd_enabled is None else opd_enabled)
    b,prompt=input_ids.shape;repeats=int(repeated_generate_nums or 1);total=b*repeats
    max_new=max_length-prompt
    if max_new<1:raise ValueError('max_length must exceed padded prompt length')
    eos=int(tokenizer.eos_token_id);device=input_ids.device
    cpeak=int(os.environ.get('CPEAK_NODES',str(verification_capacity)))
    max_nodes=int(os.environ.get('MAX_TREE_NODES_PER_SEQ','12'))
    topk=tuple(int(x) for x in os.environ.get('FIXED_TREE_TOPK_BY_DEPTH','4,3,2').split(','))
    initial_plan=plan_tree(total,cpeak,max_nodes,topk)
    # Common proposal workspaces/scanner also serve the baseline raw path.
    options=dict(rank=opd_rank,topk=max(opd_topk,max(topk)),fast_lr=opd_fast_lr,
        visited_weight=opd_visited_weight,frontier_weight=opd_frontier_weight,
        profile=opd_profile,diagnostics=opd_diagnostics,enabled=enabled,update_stream=opd_update_stream,
        backend=opd_backend if device.type=='cuda' else 'auto',train_projector=enabled and opd_train_projector)
    signature=(total,cpeak,str(device),model.dtype,tuple(sorted(options.items())))
    engine=getattr(model,'medusa_proposal_runtime',None)
    if engine is None or getattr(model,'medusa_proposal_signature',None)!=signature:
        engine=MedusaOPD(model,total,cpeak,**options)
        model.medusa_proposal_runtime=engine;model.medusa_proposal_signature=signature
    else:engine.restart(total,cpeak)
    started=time.perf_counter();target=base_lm(model)
    cache=persistent_cache(model,'medusa_target_cache',b,repeats,device,model.dtype)
    mask=attention_mask.bool()
    positions=attention_mask.long().cumsum(-1)-1;positions.clamp_min_(0)
    outputs=target.model(input_ids=input_ids,attention_mask=attention_mask,position_ids=positions,
                         past_key_values=cache,use_cache=True,return_dict=True)
    hidden=outputs.last_hidden_state
    anchor=hidden[:,-1].repeat_interleave(repeats,0)
    first_logits=model.lm_head(hidden[:,-1:]).repeat_interleave(repeats,0)
    pending,_,_=sample_target_with_metadata(first_logits,do_sample=do_sample,
        temperature=temperature,top_p=top_p,top_k=top_k,eos_token_id=eos,return_probs=False,mode=TARGET_SAMPLER_MODE)
    pending=pending[:,0]
    cache.batch_repeat_interleave(repeats)
    mask=mask.repeat_interleave(repeats,0)
    logical=attention_mask.sum(-1).repeat_interleave(repeats,0)
    global_ids=torch.arange(total,device=device)
    generated=torch.full((total,max_new),eos,device=device,dtype=torch.long)
    generated[:,0]=pending
    counts=torch.ones(total,device=device,dtype=torch.long)
    round_counts=torch.zeros_like(counts);accepted_sums=torch.zeros_like(counts)
    head_counts=torch.zeros((3,3),device=device,dtype=torch.long) # active,proposed,accepted
    ids_storage=states_storage=None
    if return_all_draft_input:
        ids_storage=torch.full((total,max_length),eos,device=device,dtype=torch.long)
        ids_storage[:,:prompt]=input_ids.repeat_interleave(repeats,0)
        ids_storage[:,prompt]=pending
        states_storage=torch.zeros((total,max_length,anchor.shape[-1]),device=device,dtype=model.dtype)
        states_storage[:,:prompt]=hidden.repeat_interleave(repeats,0)
    del outputs,hidden,first_logits
    workspace=getattr(model,'medusa_attention_workspace',None)
    if workspace is None:workspace=AttentionWorkspace(device);model.medusa_attention_workspace=workspace
    kernels=None
    if device.type=='cuda':
        import helper.tree_kernels as kernels
    rounds=nodes=active_rounds=0
    reasons={}
    # Remove prefill EOS before verification; do not sample past EOS.
    live=(pending!=eos)&(max_new>1)
    keep=live.nonzero().flatten()
    if keep.numel()!=total:cache.batch_select_indices(keep)
    pending=pending[keep];anchor=anchor[keep];mask=mask[keep];logical=logical[keep];global_ids=global_ids[keep]
    while pending.numel():
        active=pending.numel();past=cache.get_seq_length()
        plan=plan_tree(active,cpeak,max_nodes,topk)
        if plan.reason:reasons[plan.reason]=reasons.get(plan.reason,0)+1
        logits,features=model.draft_model.logits(anchor,model.lm_head)
        candidate_ids,candidate_q=engine.propose(logits,features,max(topk))
        tree=build_sparse_tree(pending,candidate_ids,candidate_q,plan)
        del logits,features,candidate_ids,candidate_q
        padding=(~mask).nonzero()
        attn=tree.attention_mask(past,model.dtype,padding,kernels=kernels,
                                  workspace=workspace.buffer('tree',(active,1,plan.budget,past+plan.budget),model.dtype).flatten())
        pos=logical[:,None]+tree.depths
        verified=target.model(input_ids=tree.tokens,attention_mask=attn,position_ids=pos,
            past_key_values=cache,use_cache=True,return_dict=True).last_hidden_state
        # All reachable nodes can be last accepted node, requiring a bonus
        # conditional. No anchor distribution is reused as a deep-head teacher.
        verify_logits=model.lm_head(verified)
        captured={}
        def retain_metadata(tokens,probs,metadata):
            captured['teacher']=probs
            captured['metadata']=metadata
            return None
        samples,_,_=sample_target_with_metadata(verify_logits,do_sample=do_sample,
            temperature=temperature,top_p=top_p,top_k=top_k,eos_token_id=eos,
            metadata_builder=retain_metadata if enabled else None,return_probs=False,mode=TARGET_SAMPLER_MODE)
        del verify_logits
        path=trace_verified_path(tree,samples,eos,kernels=kernels)
        # Enforce remaining token budget on device; no feedback after truncation.
        remaining=max_new-counts[global_ids]
        path.lengths.copy_(torch.minimum(path.lengths,remaining))
        slots=torch.arange(path.tokens.shape[1],device=device)[None,:]
        valid=slots<path.lengths[:,None]
        path.packed_indices.masked_fill_(~valid,-1)
        path.tokens.masked_fill_(~valid,-1)
        if enabled:
            teacher=captured['teacher']
            if teacher is None:teacher=torch.nn.functional.one_hot(samples,model.lm_head.weight.shape[0]).float()
            engine.feedback(tree,path,teacher,captured['metadata'])
        last=(path.lengths-1)[:,None]
        pending=path.tokens.gather(1,last).squeeze(1)
        finished=(pending==eos)|(path.lengths>=remaining)
        # One small scheduling/dispatch packet per round, no feedback readback.
        snapshots=torch.stack([s.dispatch_snapshot[0] for s in engine.engines]).long()
        packet=torch.cat((torch.stack((path.lengths,finished.long()),1).flatten(),snapshots)).cpu().tolist()
        lengths=packet[:active*2:2];done=packet[1:active*2:2]
        for h,state in enumerate(engine.engines):state.host_active_count=packet[active*2+h]
        width=max(lengths)
        for h in range(3):
            proposed=(tree.depths==h+1).sum()
            head_counts[h,0]+=((tree.depths==h+1).any(1)).sum()
            head_counts[h,1]+=proposed
            # The final decision is the pending bonus, not an accepted head token.
            accepted=(slots<(path.lengths-1)[:,None])&(slots==h)
            head_counts[h,2]+=accepted.sum()
        round_counts[global_ids]+=1;accepted_sums[global_ids]+=path.lengths
        dest=counts[global_ids,None]+slots
        # Invalid padding writes an EOS into a scratch column, never valid output.
        row_grid=global_ids[:,None].expand_as(dest)
        generated[row_grid[valid],dest[valid]]=path.tokens[valid]
        if return_all_draft_input:
            ids_storage[row_grid[valid],prompt+dest[valid]]=path.tokens[valid]
            source=path.packed_indices.clamp_min(0)[:,:,None].expand(-1,-1,verified.shape[-1])
            accepted_hidden=verified.gather(1,source)
            # KV rows correspond to root and accepted children, one token before
            # each sampled decision. Bonus hidden is intentionally not invented.
            state_dest=prompt+counts[global_ids,None]-1+slots
            states_storage[row_grid[valid],state_dest[valid]]=accepted_hidden[valid]
        counts[global_ids]+=path.lengths
        anchor=verified.gather(1,path.packed_indices.gather(1,last)[:,:,None].expand(-1,-1,verified.shape[-1])).squeeze(1)
        compact_kv(cache,path.packed_indices,past,width)
        mask=torch.cat((mask,slots[:,:width]<path.lengths[:,None]),1)
        logical+=path.lengths
        rounds+=1;nodes+=active*plan.budget;active_rounds+=active
        # Wait only on device before proposal cache row indices change.
        engine.wait()
        survivor=[r for r,flag in enumerate(done) if not flag]
        keep=torch.tensor(survivor,device=device,dtype=torch.long)
        if len(survivor)!=active:
            cache.batch_select_indices(keep)
            pending=pending[keep];anchor=anchor[keep];mask=mask[keep];logical=logical[keep];global_ids=global_ids[keep]
        del verified,samples,path,tree,attn,captured
    metrics=engine.finish() if enabled else dict(opd_backend='off',opd_updates=0,opd_profile_time_ms=0.)
    packet=torch.cat((counts,round_counts,accepted_sums,head_counts.flatten())).cpu().tolist()
    lengths=packet[:total];response_rounds=packet[total:2*total];response_acc=packet[2*total:3*total]
    hc=packet[3*total:]
    generated_ids=[generated[r,:lengths[r]].cpu().tolist() for r in range(total)]
    inputs=states=None
    if return_all_draft_input:
        inputs=[];states=[]
        real_prompt=attention_mask.bool().repeat_interleave(repeats,0)
        for r in range(total):
            valid=torch.cat((real_prompt[r],torch.ones(lengths[r],device=device,dtype=torch.bool)))
            inputs.append(ids_storage[r,:prompt+lengths[r]][valid].clone())
            states.append(states_storage[r,:prompt+lengths[r]][valid].clone())
    accepted=sum(hc[h*3+2] for h in range(3));proposed=sum(hc[h*3+1] for h in range(3))
    metrics.update(generated_token_ids=generated_ids,max_sequence_length=max(lengths),
        total_acc_length=sum(response_acc),total_decoded_token_num=sum(response_rounds),
        total_accepted_draft_tokens=accepted,total_proposed_draft_tokens=proposed,
        response_generated_tokens=lengths,response_accepted_length_sum=response_acc,response_verification_rounds=response_rounds,
        all_draft_input_ids=inputs,all_draft_input_states=states,total_time_cost=time.perf_counter()-started,
        prefill_time_cost=0.,target_time_cost=0.,draft_time_cost=0.,check_time_cost=0.,
        verification_batches=rounds,active_response_rounds=active_rounds,verified_tree_nodes=nodes,
        verification_nodes=nodes,target_forward_calls=rounds+1,
        tree_nodes_per_response=nodes/max(active_rounds,1),tree_depth_reached=max((h+1 for h in range(3) if hc[h*3]),default=0),
        head_limit_reasons=reasons,opd_host_syncs=rounds,opd_host_syncs_per_round=1. if rounds else 0.)
    for h in range(3):
        for i,name in enumerate(('active_rounds','proposed_nodes','accepted_tokens')):metrics[f'head{h+1}_{name}']=hc[h*3+i]
        metrics[f'head{h+1}_acceptance_rate']=hc[h*3+2]/max(hc[h*3+1],1)
    metrics.update({f'opd_target_{k}':v for k,v in cache.statistics().items()})
    cache.end_rollout(int(os.environ.get('OPD_KV_MAX_RETAINED_TOKENS','0')))
    return metrics


def speculative_generate(*args,**kwargs):
    model=args[0] if args else kwargs['model']
    # Match SpecNaacl's CUDA autocast region, including target softmax/sampling.
    dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16
    with torch.autocast(model.device.type,dtype=dtype,enabled=model.device.type=='cuda'):
        return _speculative_generate(*args,**kwargs)
