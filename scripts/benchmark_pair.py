#!/usr/bin/env python3
"""Run paired, real GRPO training trials; select a shared tree by generation throughput.

All trials load the same initial LoRA/Medusa checkpoint, seed, prompts and
optimizer-step budget through the production model launchers. Results include
generation and complete training wall throughput, never an AAL-only winner.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
MODELS=('qwen25_1p5b','qwen25_3b','qwen25_7b','qwen25_14b','qwen3_1p7b','qwen3_4b','llama31_8b')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,default='qwen25_1p5b')
    p.add_argument('--steps',type=int,default=10)
    p.add_argument('--max-prompts',type=int,default=0)
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--budgets',default='512:12',help='global_nodes:per_response_nodes pairs; main benchmark keeps 512 nodes')
    p.add_argument('--topk',default='4,3,2')
    p.add_argument('--output',type=Path)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--profile',action='store_true',help='optional OPD event timings; adds profiling overhead')
    a=p.parse_args()
    if min(a.steps,a.trials)<1:p.error('positive steps and trials required')
    if not os.environ.get('TARGET_ADAPTER') and not a.dry_run:p.error('TARGET_ADAPTER must name the common initial LoRA')
    if not a.dry_run:
        import torch
        if not torch.cuda.is_available():p.error('CUDA runtime required; this tool does not fabricate GPU timings')
    output=(a.output or ROOT/'outputs/benchmarks'/f'{a.model}_{time.strftime("%Y%m%dT%H%M%S",time.gmtime())}').resolve()
    records=[]
    initialization_proofs={}
    for config in a.budgets.split(','):
        global_budget,max_nodes=map(int,config.split(':'))
        for trial in range(a.trials):
            for method in ('medusa','medusa_reflex'):
                name=f'nodes{global_budget}_max{max_nodes}_trial{trial}_method-{method}'
                run=output/name
                env=dict(os.environ,LAUNCHER_USE_ENV='1',RUN_DIR=str(run),RUN_NAME=name,RESUME='',DRY_RUN='true' if a.dry_run else 'false',
                    GRPO_BENCHMARK='1',GENERATION_LENGTH_POLICY=os.environ.get('GENERATION_LENGTH_POLICY','specnaacl_compatible'),PYTHON_BIN=sys.executable,CPEAK_NODES=str(global_budget),MAX_TREE_NODES_PER_SEQ=str(max_nodes),
                    FIXED_TREE_TOPK_BY_DEPTH=a.topk,MAX_TARGET_OPTIMIZER_STEPS=str(a.steps),MAX_ROLLOUT_PROMPTS=str(a.max_prompts),
                    OPD_SELECTION='visited_capped_frontier',OPD_MAX_FRONTIER_PER_HEAD='2',OPD_PROFILE='1' if a.profile else '0')
                suffix='medusa' if method=='medusa' else 'reflex'
                subprocess.run(['bash',str(ROOT/f'train_{a.model}_{suffix}.sh')],env=env,cwd=ROOT,check=True)
                if a.dry_run:continue
                summary=json.loads((run/'summary.json').read_text())
                proof=summary['initialization']
                if proof['target_lora']['status']!='PASS' or proof.get('draft',{}).get('status')!='PASS':
                    raise RuntimeError('Benchmark requires loaded target/draft tensor proofs')
                identity=(proof['target_lora']['loaded_tensor_sha256'],proof['draft']['loaded_tensor_sha256'])
                pair=(global_budget,max_nodes,trial)
                if pair in initialization_proofs and initialization_proofs[pair]!=identity:
                    raise RuntimeError('Paired benchmark initialization tensors differ')
                initialization_proofs[pair]=identity
                if not a.max_prompts and summary['target_optimizer_steps']!=a.steps:
                    raise RuntimeError('trial did not reach target optimizer budget; increase NUM_EPOCHS or inspect reward filtering')
                rows=list(csv.DictReader((run/'logs/rollout_timing.csv').open()))
                # Drop one warm-up rollout from generation throughput, keep actual
                # whole-run throughput separately (incl startup/training/reward).
                steady=rows[1:] or rows
                tokens=sum(float(r['iter_rollout_tokens']) for r in steady)
                generation=sum(float(r['iter_generation_time_s']) for r in steady)
                wall=sum(float(r['iter_wall_time_s']) for r in steady)
                record=dict(generation_length_policy=summary['generation_length_policy'],method=method,global_nodes=global_budget,max_nodes=max_nodes,trial=trial,
                    generation_tokens_per_s=tokens/max(generation,1e-9),end_to_end_tokens_per_s=tokens/max(wall,1e-9),
                    whole_run_tokens_per_s=summary['total_rollout_tokens']/max(summary['total_wall_time_s'],1e-9),
                    opd_feedback_time_s=sum(float(r.get('opd_feedback_time') or 0) for r in steady) if a.profile else None,
                    opd_proposal_time_s=sum(float(r.get('opd_proposal_time') or 0) for r in steady) if a.profile else None,
                    aal=summary['cumulative_aal'],target_optimizer_steps=summary['target_optimizer_steps'],
                    peak_vram_gb=summary.get('gpu_peak_allocated_gb'),
                    rollout_prompts=summary['rollout_prompts_seen'],**summary.get('medusa_totals',{}),run=str(run))
                records.append(record)
                output.mkdir(parents=True,exist_ok=True)
                (output/'results.json').write_text(json.dumps(records,indent=2)+'\n')
    if a.dry_run:return
    scored=[]
    for config in a.budgets.split(','):
        g,n=map(int,config.split(':'))
        # Choose one policy for both ablations by paired geometric mean of
        # measured generation throughput, with medians across trials.
        scores=[statistics.median(r['generation_tokens_per_s'] for r in records if r['global_nodes']==g and r['max_nodes']==n and r['method']==m)
                for m in ('medusa','medusa_reflex')]
        scored.append(((scores[0]*scores[1])**.5,g,n))
    score,g,n=max(scored)
    (output/'selected_tree.env').write_text(f'export CPEAK_NODES={g}\nexport MAX_TREE_NODES_PER_SEQ={n}\nexport FIXED_TREE_TOPK_BY_DEPTH={a.topk}\n')
    print(json.dumps(dict(selected_global_nodes=g,selected_max_nodes=n,paired_generation_tokens_per_s=score,
        report=str(output/'results.json'),settings=str(output/'selected_tree.env')),indent=2))


if __name__=='__main__':main()
