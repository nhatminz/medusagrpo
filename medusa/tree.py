"""Budget-aware sparse prefix trees; no dense Cartesian construction."""
from dataclasses import dataclass
import torch
from helper.tree_verification import PackedTree


@dataclass(frozen=True)
class TreePlan:
    budget: int
    active_heads: int
    topk: tuple
    reason: str


def plan_tree(active_responses, cpeak_nodes=512, max_nodes=12, topk=(4,3,2)):
    if active_responses < 1 or max_nodes < 1 or cpeak_nodes < active_responses:
        raise ValueError('global verification budget must fit at least one root per active response; lower rollout concurrency')
    if len(topk) != 3 or min(topk) < 1: raise ValueError('three positive head top-k values required')
    budget = min(max_nodes, cpeak_nodes//active_responses)
    heads = min(3, budget-1)
    possible = 1
    branches = 1
    for k in topk[:heads]:
        branches *= k; possible += branches
    budget = min(budget, possible)
    return TreePlan(budget, heads, tuple(topk),
                    '' if heads == 3 else f'global_budget={cpeak_nodes},active_responses={active_responses},per_response={budget}; limited_heads={3-heads}')


@dataclass
class SparseTree(PackedTree):
    depths: torch.Tensor
    scores: torch.Tensor


def build_sparse_tree(root, candidate_ids, candidate_probs, plan, backend='auto'):
    b = root.shape[0]; n = plan.budget
    if root.is_cuda and backend != 'torch':
        from medusa.tree_kernels import build
        return build(root, candidate_ids, candidate_probs, plan)
    tokens = torch.empty((b,n), device=root.device, dtype=torch.long)
    parents = torch.full_like(tokens, -1); depths = torch.zeros_like(tokens)
    scores = torch.zeros((b,n), device=root.device)
    tokens[:,0] = root
    # Reserve one complete highest-probability path BEFORE width expansion.
    for d in range(plan.active_heads):
        tokens[:,d+1] = candidate_ids[:,d,0]
        parents[:,d+1] = d; depths[:,d+1] = d+1
        scores[:,d+1] = scores[:,d] + candidate_probs[:,d,0].clamp_min(1e-30).log()
    kmax = candidate_ids.shape[-1]
    rank = torch.arange(kmax, device=root.device)[None,None,:]
    row = torch.arange(b, device=root.device)[:,None]
    for slot in range(plan.active_heads+1,n):
        pd = depths[:,:slot]
        valid_parent = pd < plan.active_heads
        horizon = pd.clamp_max(2)
        ids = candidate_ids[row,horizon]
        lp = candidate_probs[row,horizon].clamp_min(1e-30).log()
        duplicate = ((parents[:,:slot,None,None] == torch.arange(slot,device=root.device)[None,None,:,None]) &
                     (tokens[:,:slot,None,None] == ids[:,None,:,:])).any(1)
        limits = torch.tensor(plan.topk,device=root.device)[horizon]
        valid = valid_parent[:,:,None] & (rank < limits[:,:,None]) & ~duplicate
        values = (scores[:,:slot,None] + lp).masked_fill(~valid, -torch.inf)
        choice = values.flatten(1).argmax(1)
        p, r = choice//kmax, choice%kmax
        tokens[:,slot] = ids[torch.arange(b,device=root.device),p,r]
        parents[:,slot] = p; depths[:,slot] = pd.gather(1,p[:,None]).squeeze(1)+1
        scores[:,slot] = values.flatten(1).gather(1,choice[:,None]).squeeze(1)
    contexts = torch.where(depths < plan.active_heads, torch.zeros_like(depths), -1)
    return SparseTree(parents,tokens,contexts,plan.active_heads,depths,scores)


def select_feedback(tree, path, head, selection='visited_capped_frontier', frontier_cap=2,
                    visited_weight=1., frontier_weight=1.):
    """Select decision parents on device before any teacher vocabulary scan.

    Head 1 is taught by root/depth0; head 3 by depth2, never anchor logits.
    Frontier consists of off-path states whose parent was visited. A prefix's
    accumulated log-proposal probability ranks them without target work.
    """
    if selection not in ('visited_capped_frontier','visited_only','all_internal_weighted'):
        raise ValueError('unsupported OPD_SELECTION')
    if not 0 <= frontier_cap <= 2: raise ValueError('frontier cap must be in [0,2]')
    rows = torch.arange(tree.parents.shape[1],device=tree.parents.device)[None,:]
    visited = (rows[:,:,None] == path.packed_indices[:,None,:]).any(-1)
    parent_visited = (tree.parents[:,:,None] == path.packed_indices[:,None,:]).any(-1)
    eligible = (tree.depths == head) & (tree.feedback_contexts >= 0)
    on = visited & eligible
    off = ~visited & eligible
    if selection == 'visited_only': off = torch.zeros_like(off)
    elif selection == 'visited_capped_frontier':
        off &= parent_visited
        scores = tree.scores.masked_fill(~off,-torch.inf)
        chosen = scores.topk(min(frontier_cap,scores.shape[1]),dim=1).indices
        capped = torch.zeros_like(off).scatter_(1,chosen,True)
        off &= capped
    weights = on.float()*visited_weight + off.float()*frontier_weight
    if selection == 'all_internal_weighted': weights *= tree.scores.exp()
    kinds = on.int() + off.int()*2
    return weights,kinds


def restrict_feedback(tree,remaining,eos_token_id):
    """Verified output is eligible only before EOS and within the token budget.

    A decision at depth d emits token d+1 after the pending root. Off-path
    prefixes containing EOS are terminal, including their descendants.
    This does not change proposals, attention, target sampling or traversal.
    """
    valid=tree.depths<remaining[:,None]
    cursor=torch.arange(tree.tokens.shape[1],device=tree.tokens.device).expand_as(tree.tokens)
    for _ in range(tree.max_depth+1):
        valid &= tree.tokens.gather(1,cursor)!=int(eos_token_id)
        cursor=tree.parents.gather(1,cursor).clamp_min(0)
    tree.feedback_contexts.masked_fill_(~valid,-1)
