"""SpecNaacl ReflexOPD, with per-head rollout-local B and capped selection.

Each engine batches all selected responses/parents for a single horizon. Three
engines share A and a contiguous [3,V,R] B pool. There is no per-node Python
update, correction, teacher scan, or target forward.
"""
from types import SimpleNamespace
import os
import torch
from helper.opd_reflex import OPDReflex, OPD_COUNTER_NAMES, union_reference
from medusa.tree import select_feedback


class HeadOPD(OPDReflex):
    def _feedback_reference(self,tree,path,target,greedy,sampling_metadata=None):
        # CPU test oracle. Exact SpecNaacl sparse coordinate q-p update + tail KL.
        w,kind=tree.selected_weights,tree.selected_kind
        b,n=w.shape;k=self.topk
        head=self.head_cache[:b,0];u=self.u_cache[:b,0]
        raw=torch.nn.functional.linear(head,self.head.weight,self.head.bias).float()
        q=(raw+u@self.B_fast.T).softmax(-1)
        p=target.float() if not greedy else torch.nn.functional.one_hot(target,self.vocab).float()
        ti=p.topk(k,dim=-1).indices
        di=self.ids_cache[:b,0,None,:].expand(b,n,k)
        ids,valid,pu,qu,pt,qt,kl=union_reference(p.flatten(0,1),q[:,None,:].expand_as(p).flatten(0,1),ti.flatten(0,1),di.flatten(0,1))
        weights=w.flatten();total=weights.sum()
        g=torch.where(valid,(qu-pu)*weights[:,None],0.)
        if self.train_projector:
            v=(g[:,:,None]*self.B_fast[ids.clamp_min(0)]).sum(1)
            self.model.opd_projector_grad_sum.add_(head[:,None,:].expand(b,n,-1).flatten(0,1).float().T@v)
            self.model.opd_projector_grad_weight.add_(total)
        delta=torch.zeros_like(self.B_fast)
        delta.index_add_(0,ids.clamp_min(0).flatten(),(g[:,:,None]*u[:,None,None,:].expand(b,n,2*k,self.rank).reshape(b*n,2*k,self.rank)).flatten(0,1))
        self.B_fast.add_(delta,alpha=-self.fast_lr/float(total.clamp_min(1)))
        active=torch.nonzero(self.B_fast.abs().sum(-1)>0).flatten()
        self.active_count.fill_(active.numel());self.active_ids[:active.numel()].copy_(active)
        selected=weights>0;kc=kind.flatten()
        self.counters[:13].add_(torch.stack([total,selected.sum(),(selected&(kc==1)).sum(),(selected&(kc==2)).sum(),
            torch.where(selected&torch.isfinite(kl),weights*kl,0.).sum(),(valid&selected[:,None]).sum(),
            selected.sum(),torch.where(selected,pu[:,:k].sum(-1),0.).sum(),total*0,
            ((total>0)&(self.fast_lr>0)).float(),self.active_count[0].float(),total.new_tensor(1.),
            (selected&~torch.isfinite(kl)).sum()]).double())
        self.counters[15]=torch.maximum(self.counters[15],self.active_count[0].double())


class MedusaOPD:
    def __init__(self,model,batch,max_nodes,update_stream=True,**options):
        self.model=model
        self.selection=os.environ.get('OPD_SELECTION','visited_capped_frontier')
        self.cap=int(os.environ.get('OPD_MAX_FRONTIER_PER_HEAD','2'))
        self.engines=[]
        self.enabled=options.get('enabled',True)
        self.B_fast=(torch.zeros((3,model.lm_head.weight.shape[0],options.get('rank',8)),device=model.device)
                     if self.enabled else None)
        mapping=torch.arange(model.lm_head.weight.shape[0],device=model.device)
        for h in range(3):
            state=HeadOPD(**options)
            state.start(model,batch,mapping,model.lm_head.weight.shape[1],max_contexts=1,
                max_nodes=max_nodes,max_path=4,max_proposal_contexts=1)
            state.defer_fast_update=state.backend=='triton'
            if self.enabled:state.B_fast=self.B_fast[h]
            self.engines.append(state)
        self.stream=(torch.cuda.Stream(device=model.device) if self.enabled and model.device.type=='cuda' and update_stream else None)
        self.event=None

    def restart(self,batch,max_nodes):
        self.wait()
        self.event=None
        for h,state in enumerate(self.engines):
            state.start(self.model,batch,state.mapping,self.model.lm_head.weight.shape[1],max_contexts=1,
                max_nodes=max_nodes,max_path=4,max_proposal_contexts=1)
            if self.enabled:state.B_fast=self.B_fast[h]
        if self.enabled:self.B_fast.zero_()

    def wait(self):
        if self.event is not None:
            torch.cuda.current_stream(self.model.device).wait_event(self.event)

    def propose(self,logits,features,keep):
        self.wait()
        ids=[];probs=[]
        for h,state in enumerate(self.engines):
            # Batch all responses once/head/round; q is reused by every prefix.
            p,i,_=state.propose(logits[:,h:h+1],features[:,h:h+1],keep,state.mapping,root=True)
            probs.append(p[:,0].clone());ids.append(i[:,0].clone())
        return torch.stack(ids,1),torch.stack(probs,1)

    def feedback(self,tree,path,teacher,metadata=None):
        def work():
            for h,state in enumerate(self.engines):
                w,kind=select_feedback(tree,path,h,self.selection,self.cap,state.visited_weight,state.frontier_weight)
                adapted=SimpleNamespace(parents=tree.parents,feedback_contexts=torch.zeros_like(tree.parents),
                    selected_weights=w,selected_kind=kind)
                state.feedback(adapted,path,teacher,sampling_metadata=metadata)
            if self.engines[0].backend=='triton':
                from medusa.opd_kernels import update_heads
                ticket=self.engines[0].begin('opd_update_ms')
                update_heads(self.engines,adapted.feedback_contexts)
                self.engines[0].end(ticket)
        if self.stream is None:work()
        else:
            self.stream.wait_stream(torch.cuda.current_stream(self.model.device))
            # Prevent allocator reuse while feedback runs on its own stream.
            for tensor in (tree.parents,tree.depths,tree.scores,tree.feedback_contexts,path.packed_indices,teacher):
                tensor.record_stream(self.stream)
            if metadata is not None:
                for tensor in metadata:tensor.record_stream(self.stream)
            with torch.cuda.stream(self.stream):
                work();self.event=torch.cuda.Event();self.event.record(self.stream)

    def finish(self):
        self.wait()
        result={name:0. for name in OPD_COUNTER_NAMES}
        result['opd_backend']=self.engines[0].backend
        result['opd_profile_time_ms']=0.
        for h,state in enumerate(self.engines,1):
            metrics=state.finish()
            for name in OPD_COUNTER_NAMES:result[name]+=metrics[name]
            weight=metrics['opd_state_weight'];count=metrics['opd_selected_states']
            for name,source in [('selected_states','selected_states'),('visited_states','visited_states'),('frontier_states','frontier_states'),('active_rows','active_rows_max')]:
                result[f'opd_head{h}_{name}']=metrics['opd_'+source]
            result[f'opd_head{h}_kl']=metrics['opd_kl_sum']/max(weight,1.)
            result[f'opd_head{h}_target_mass_in_draft_top16']=metrics['opd_draft_topk_target_mass_sum']/max(count,1.)
            result['opd_profile_time_ms']+=metrics['opd_profile_time_ms']
            for phase in ('opd_state_select_ms','opd_teacher_extract_ms','opd_union_loss_ms','opd_update_ms','opd_feature_ms','opd_proposal_extra_ms'):
                result[phase]=result.get(phase,0.)+(metrics.get('opd_profile_sections_ms') or {}).get(phase,0.)
        result['opd_update_count']=result['opd_updates']
        result['opd_feedback_time']=None if not self.engines[0].profile else sum(result.get(p,0.) for p in ('opd_state_select_ms','opd_teacher_extract_ms','opd_union_loss_ms','opd_update_ms'))/1000
        result['opd_proposal_time']=None if not self.engines[0].profile else sum(result.get(p,0.) for p in ('opd_feature_ms','opd_proposal_extra_ms'))/1000
        return result
