#!/usr/bin/env python3
"""Read-only, seven-model comparison of effective launchers and runtime bindings.

Exit status 0 means the audit ran, not that all five methods are fully fair.
Use --strict to fail on known differences, missing sources or unverified LoRA.
No baseline training entrypoint is imported or executed.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
MODELS=('qwen25_1p5b','qwen25_3b','qwen25_7b','qwen25_14b','qwen3_1p7b','qwen3_4b','llama31_8b')
COMMON=('model_dir','model_type','dtype','attn_implementation','load_lora_path','train_option',
    'dataset_path','train_data_fraction','train_subset_seed','max_train_samples','batch_size',
    'num_epochs','accumulation_steps','target_lr','temperature','top_p','max_length',
    'max_prompt_length','max_training_padding_gap','max_training_token','logps_chunk_size',
    'grpo_iteration_num','repeated_generate_nums','beta','epsilon','seed','nproc_per_node')


def launcher(root,key,suffix,env):
    path=root/f'train_{key}{suffix}.sh'
    if not path.is_file():return None
    result=subprocess.run(['bash',str(path)],cwd=root,env=env,text=True,capture_output=True,check=True)
    command=next(line.split(':',1)[1] for line in result.stdout.splitlines() if line.startswith('Command'))
    tokens=shlex.split(command);flags={}
    for i,token in enumerate(tokens):
        if token.startswith('--'):
            name,sep,value=token[2:].partition('=')
            flags[name]=value if sep else tokens[i+1]
    return flags


def _functions(path,names):
    if not path.is_file():return None
    tree=ast.parse(path.read_text())
    return {n.name:ast.dump(n,include_attributes=False) for n in tree.body
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names}


def lora_recipe(path):
    if not path.is_file():return None
    calls=[n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Name) and n.func.id=='LoraConfig']
    if len(calls)!=1:return 'not verified'
    result={k.arg:(ast.unparse(k.value) if k.arg=='task_type' else ast.literal_eval(k.value)) for k in calls[0].keywords}
    result['target_modules']=sorted(result['target_modules'])
    return result


def optimizer_settings(path):
    if not path.is_file():return None
    tree=ast.parse(path.read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
           and n.func.attr=='AdamW']
    if not calls:return 'not verified'
    return {'optimizer':ast.unparse(calls[0].func),'explicit_options':
        {k.arg:ast.unparse(k.value) for k in calls[0].keywords if k.arg!='lr'}}


def runtime_generation(path,flags,pure=False):
    """Resolve CLI-bound globals, then inspect the actual generation call."""
    tree=ast.parse(path.read_text());assignments={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name):assignments[target.id]=node.value
    def value(node,seen=()):
        if isinstance(node,ast.Constant):return node.value
        if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='args':
            return flags.get(node.attr)
        if isinstance(node,ast.Name) and node.id in assignments and node.id not in seen:
            return value(assignments[node.id],(*seen,node.id))
        return 'not verified: '+ast.unparse(node)
    name='generate' if pure else 'speculative_generate'
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==name]
    call=next(n for n in calls if any(k.arg=='repeated_generate_nums' for k in n.keywords))
    result={k.arg:value(k.value) for k in call.keywords
            if k.arg in ('do_sample','max_length','repeated_generate_nums','temperature','top_p','top_k')}
    if pure:result['top_k']=None if flags.get('top_k','0')=='0' else flags['top_k']
    else:result.setdefault('top_k',None)
    return result


def _prefill_convention(path,sample_name):
    if not path.is_file():return 'not verified'
    tree=ast.parse(path.read_text())
    samples=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call)
             and isinstance(n.func,ast.Name) and n.func.id==sample_name]
    repeats=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call)
             and isinstance(n.func,ast.Attribute) and n.func.attr=='repeat_interleave'
             and any(isinstance(x,ast.Name) and x.id in ('pending','target_next_token','logits','target_logits','generated_sequences') for x in ast.walk(n.func.value))]
    if not samples or not repeats:return 'not verified'
    return 'shared_per_prompt' if min(samples)<min(repeats) else 'independent_per_response'


def audit(spec,pure,env=None,models=MODELS):
    base={k:v for k,v in os.environ.items() if k in ('PATH','HOME','LANG','LD_LIBRARY_PATH')}
    if env is not None:base.update(env)
    base.update(DRY_RUN='true',PYTHON_BIN=sys.executable,RUN_NAME='fairness_audit',
                RUN_DIR=str(ROOT/'outputs/fairness_dry_run_no_writes'),RESUME='')
    base.pop('METHOD',None)
    entries={'medusa':(ROOT,'_medusa',ROOT/'grpo_speculative.py'),
        'medusa_reflex':(ROOT,'_reflex',ROOT/'grpo_speculative.py'),
        'fastgrpo':(spec,'_fastgrpo',spec/'grpo_speculative.py'),
        'fastgrpo_reflex':(spec,'',spec/'grpo_speculative.py'),
        'puregrpo':(pure,'',pure/'training.py')}
    source_checks={}
    for method,(root,_,training) in entries.items():
        if not training.is_file():
            source_checks[method]={'status':'not verified','reason':'source unavailable'};continue
        reward={name:(root/'helper'/name).is_file() and (spec/'helper'/name).is_file()
                and (root/'helper'/name).read_bytes()==(spec/'helper'/name).read_bytes()
                for name in ('rewards.py','get_QAs.py')}
        collator=_functions(root/('helper/grpo_core.py' if method=='puregrpo' else 'grpo_speculative.py'),{'TrainDataCollator'})
        source_collator=_functions(spec/'grpo_speculative.py',{'TrainDataCollator'})
        source_checks[method]=dict(status='inspected',reward_and_dataset=reward,
            collator_matches=collator==source_collator,
            lora_recipe=lora_recipe(root/'helper/target.py' if method=='puregrpo' else training),
            target_optimizer=optimizer_settings(training),
            target_loss_ast_matches=(None if method=='puregrpo' else
                _functions(training,{'compute_target_loss_and_backward'})==
                _functions(spec/'grpo_speculative.py',{'compute_target_loss_and_backward'})))
    rows=[]
    for key in models:
        methods={};differences=[]
        for method,(root,suffix,training) in entries.items():
            flags=launcher(root,key,suffix,base) if training.is_file() else None
            if flags is None:
                methods[method]={'status':'not verified','reason':'source or model launcher unavailable'};continue
            generation=runtime_generation(training,flags,pure=method=='puregrpo')
            methods[method]=dict(status='inspected',effective_flags={k:flags.get(k) for k in COMMON},
                generation_runtime=generation,top_k=generation['top_k'],
                tree_budget={k:base.get(k,v) for k,v in [('CPEAK_NODES','512'),('MAX_TREE_NODES_PER_SEQ','12'),('FIXED_TREE_TOPK_BY_DEPTH','4,3,2')]} if method.startswith('medusa') else None)
        reference=methods['medusa']
        for method,result in methods.items():
            if result['status']=='not verified':continue
            for field in COMMON:
                if result['effective_flags'][field]!=reference['effective_flags'][field]:
                    differences.append(dict(method=method,field=field,medusa=reference['effective_flags'][field],value=result['effective_flags'][field]))
            for field,value in result['generation_runtime'].items():
                if value!=reference['generation_runtime'].get(field):
                    differences.append(dict(method=method,field='runtime_generation.'+field,medusa=reference['generation_runtime'].get(field),value=value))
        rows.append(dict(model=key,methods=methods,configuration_mismatches=differences))
    conventions={
        'medusa':_prefill_convention(ROOT/'medusa/generate.py','sample_target_with_metadata'),
        'medusa_reflex':_prefill_convention(ROOT/'medusa/generate.py','sample_target_with_metadata'),
        'fastgrpo':_prefill_convention(spec/'helper/fastgrpo_generate.py','sampling'),
        'fastgrpo_reflex':_prefill_convention(spec/'helper/opd_generate.py','sample_target_with_metadata'),
        'puregrpo':_prefill_convention(pure/'helper/autoregressive.py','sample_from_probs')}
    initial=base.get('TARGET_ADAPTER','')
    checkpoint=Path(initial) if initial else None
    initial_status=dict(status='not verified',path=initial,reason='set TARGET_ADAPTER to the same existing initial checkpoint')
    if checkpoint is not None and (checkpoint/'adapter_config.json').is_file():
        initial_status=dict(status='path configured; full real-model tensor loading not verified',path=str(checkpoint.resolve()),
            files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir() if p.name in ('adapter_config.json','adapter_model.safetensors','adapter_model.bin')})
    requirements={}
    for method,(root,_,_) in entries.items():
        path=root/'requirements.txt'
        requirements[method]=dict(line.strip().split('==',1) for line in path.read_text().splitlines() if '==' in line and not line.lstrip().startswith('#')) if path.exists() else 'not verified'
    return dict(status='audit_complete_with_limits',models=rows,source_checks=source_checks,
        first_token_convention=conventions,initial_target_lora=initial_status,declared_packages=requirements,
        remaining_differences=[
            ('PureGRPO independently samples first tokens; four speculative methods share one first token/prompt.' if conventions['puregrpo']!='not verified' else 'PureGRPO source unavailable: its sampling convention is not verified.'),
            'Medusa uses a strict per-response padded-prompt length cap. Source FastGRPO stops a full verification batch using real lengths and can overshoot; PureGRPO uses real prompt lengths.',
            'All inspected trainers retain the inherited sequence-length sort without permuting advantages. This association defect is deliberately not fixed only in Medusa.',
            'Source max_grpo_steps is a legacy eligible-prompt step label; Medusa MAX_TARGET_OPTIMIZER_STEPS counts actual optimizer updates. Align actual updates/prompts from logs.',
            'SpecNaacl and Medusa prefill sampling occur outside autocast (BF16 probabilities for BF16 targets); verification sampling occurs inside autocast. PureGRPO explicitly computes FP32 probabilities.',
            'Pinned packages are declarations; deployed B200 environments and full real-model initial LoRA tensors are not verified by this read-only audit.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec-root',type=Path,default=ROOT.parent/'SpecNaacl')
    p.add_argument('--pure-root',type=Path,default=ROOT.parent/'puregrpo')
    p.add_argument('--model',choices=MODELS)
    p.add_argument('--use-environment',action='store_true')
    p.add_argument('--strict',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args();report=audit(a.spec_root.resolve(),a.pure_root.resolve(),dict(os.environ) if a.use_environment else None,(a.model,) if a.model else MODELS)
    text=json.dumps(report,indent=2)+'\n'
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(text)
    if a.strict:sys.exit(1) # Known sampling/length/initialization limits preclude a fully-fair PASS.


if __name__=='__main__':main()
