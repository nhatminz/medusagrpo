#!/usr/bin/env python3
"""Exercise production GRPO on a shared synthetic checkpoint, and report CPU timings."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/benchmarks/cpu_smoke')
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for method in ('medusa','medusa_reflex'):
        console=a.output/f'{method}.console.log'
        with console.open('w') as stream:
            subprocess.run([sys.executable,str(ROOT/'tests/tiny_runner.py'),str(a.output),method,method],
                stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT,
                env=dict(os.environ,OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false'),check=True)
        run=a.output/method
        summary=json.loads((run/'summary.json').read_text())
        rows=list(csv.DictReader((run/'logs/rollout_timing.csv').open()))
        tokens=sum(float(r['iter_rollout_tokens']) for r in rows)
        generation=sum(float(r['iter_generation_time_s']) for r in rows)
        wall=sum(float(r['iter_wall_time_s']) for r in rows)
        records.append(dict(method=method,device='CPU',synthetic_model=True,
            aal=summary['cumulative_aal'],generation_tokens_per_s=tokens/generation,
            end_to_end_tokens_per_s=tokens/wall,target_optimizer_steps=summary['target_optimizer_steps'],
            **summary.get('medusa_totals',{})))
    report=dict(note='Synthetic CPU smoke only. These timings are not B200 benchmarks or research results.',records=records)
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
