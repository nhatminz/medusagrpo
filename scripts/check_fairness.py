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
    'grpo_iteration_num','repeated_generate_nums','beta','epsilon','seed','nproc_per_node',
    'max_target_optimizer_steps','max_rollout_prompts')


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


def audit(spec,pure,env=None,models=MODELS,runtime_reports=None):
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
    report = dict(status='NOT VERIFIED',models=rows,source_checks=source_checks,
        first_token_convention=conventions,initial_target_lora=initial_status,declared_packages=requirements,
        remaining_differences=[
            'Medusa caps each response using actual prompt length; unchanged SpecNaacl/PureGRPO stop a whole batch using its longest real sequence. SpecNaacl may overshoot a verification round.',
            'PureGRPO independently samples first tokens and uses FP32 probabilities; speculative methods share first tokens per prompt and preserve source prefill/verification dtype conventions.',
            'Old GRPO checkpoints used mismatched advantages after length sorting; retrain all methods from shared initialization.',
            'Launcher/package declarations do not prove actual weights, deployed dependencies, prompt order or optimizer budgets. Supply completed runtime reports.'])
    return verify_runtime(report, runtime_reports or {})


def check(status,reason):
    return dict(status=status,reason=reason)


def verify_runtime(report, runtime_reports):
    """Accept only startup tensor proofs + completed summaries, per model/method.

    runtime_reports maps (model,method) to a production summary JSON dictionary.
    Missing evidence remains NOT VERIFIED, even when launcher flags match.
    """
    for row in report['models']:
        methods=row['methods'];checks={}
        available=all(v['status']=='inspected' for v in methods.values())
        checks['launcher_configuration']=check('FAIL' if row['configuration_mismatches'] else
            ('PASS' if available else 'NOT VERIFIED'),'Effective dry-run commands; runtime overrides checked separately')
        checks['generation_length']=check('FAIL','Medusa per-response actual length cap differs from unchanged baseline batch stopping/overshoot')
        evidence={m:runtime_reports.get((row['model'],m)) for m in methods}
        proofs={m:(v or {}).get('initialization',{}) for m,v in evidence.items()}
        loras=[p.get('target_lora',{}) for p in proofs.values()]
        known=[p for p in loras if p.get('status')=='PASS' and p.get('loaded_tensor_sha256')]
        hashes={p['loaded_tensor_sha256'] for p in known}
        checks['loaded_initial_target_lora']=check('FAIL' if len(hashes)>1 else
            ('PASS' if len(known)==5 else 'NOT VERIFIED'),'Compare every loaded target LoRA tensor, including dtype and shape, across five startup proofs')
        for pair in (('medusa','medusa_reflex'),('fastgrpo','fastgrpo_reflex')):
            values=[proofs[m].get('draft',{}) for m in pair]
            known_draft=[v.get('loaded_tensor_sha256') for v in values if v.get('status')=='PASS']
            checks['loaded_draft_' + pair[0]]=check('FAIL' if len(set(known_draft))>1 else
                ('PASS' if len(known_draft)==2 else 'NOT VERIFIED'),'Loaded pretrained proposal tensors must agree within the Reflex pair')
        settings=[p.get('effective_args',{}) for p in proofs.values()]
        fields=[k for k in COMMON if k not in ('load_lora_path','nproc_per_node')]
        mismatch=[k for k in fields if len({json.dumps(v.get(k)) for v in settings if v})>1]
        for name,field,content in (('loaded_target_backbone','target_backbone','loaded_tensor_sha256'),
                                   ('tokenizer_artifacts','tokenizer_files','hashes'),
                                   ('dataset_artifact','dataset','file_sha256')):
            values=[p.get(field,{}) for p in proofs.values()]
            known_values=[json.dumps(v.get(content),sort_keys=True) for v in values
                          if v.get('status')=='PASS' and v.get(content)]
            checks[name]=check('FAIL' if len(set(known_values))>1 else
                ('PASS' if len(known_values)==5 else 'NOT VERIFIED'),'Actual startup hashes across five methods')
        checks['effective_runtime_configuration']=check('FAIL' if mismatch else
            ('PASS' if all(all(k in v for k in fields) for v in settings) else 'NOT VERIFIED'),'Runtime common-argument differences: '+str(mismatch))
        environments=[{k:p.get(k) for k in ('packages','cuda','gpu','target_dtype','attention_implementation')} for p in proofs.values()]
        complete=all(p.get('packages') and p.get('target_dtype') for p in proofs.values())
        checks['deployed_environment']=check('FAIL' if complete and any(v!=environments[0] for v in environments) else
            ('PASS' if complete else 'NOT VERIFIED'),'Actual packages, GPU, dtype and resolved attention implementation')
        counts=[(v or {}).get('target_optimizer_steps',(v or {}).get('optimizer_steps')) for v in evidence.values()]
        checks['actual_optimizer_updates']=check('FAIL' if len({v for v in counts if v is not None})>1 else
            ('PASS' if all(v is not None for v in counts) else 'NOT VERIFIED'),'Completed optimizer.step counts: '+str(counts))
        # A startup report cannot prove executed prompt order/cadence or resume.
        order=[(v or {}).get('prompt_order_sha256') for v in evidence.values()]
        checks['executed_prompt_order']=check('FAIL' if len({v for v in order if v})>1 else
            ('PASS' if all(order) else 'NOT VERIFIED'),'Executed prompt-batch hash chains; compare same rank/world size')
        cadence=[(v or {}).get('optimizer_step_cadence') for v in evidence.values()]
        checks['actual_optimizer_cadence']=check('FAIL' if len({json.dumps(v) for v in cadence if v is not None})>1 else
            ('PASS' if all(v is not None for v in cadence) else 'NOT VERIFIED'),'Actual (prompts seen, optimizer updates) at training boundaries')
        checks['distributed_rank_coverage']=check('NOT VERIFIED','Per-rank startup proofs and complete multi-rank prompt/cadence traces have not been supplied')
        checks['checkpoint_resume']=check('NOT VERIFIED','Local tiny-model resume tests do not prove full-model B200 continuation')
        checks['sampling_precision']=check('FAIL','PureGRPO FP32 sampling and source speculative BF16 prefill probabilities differ; source conventions preserved')
        for pair in (('medusa','medusa_reflex'),('fastgrpo','fastgrpo_reflex')):
            left,right=(proofs[m] for m in pair)
            args_left=dict(left.get('effective_args',{}));args_right=dict(right.get('effective_args',{}))
            ignored=('method','version_name','log_file','timing_file','summary_file','saved_model_dir',
                     'saved_draft_model_dir','saved_statistics_dir','checkpoint_dir','opd_train_projector')
            common=(set(args_left)&set(args_right))-set(ignored)
            diff=[k for k in common if args_left[k]!=args_right[k]]
            env_left={k:v for k,v in left.get('environment',{}).items() if k!='OPD_ENABLED'}
            env_right={k:v for k,v in right.get('environment',{}).items() if k!='OPD_ENABLED'}
            checks['paired_recipe_'+pair[0]]=check('FAIL' if diff or (env_left and env_right and env_left!=env_right) else
                ('PASS' if args_left and args_right and env_left and env_right else 'NOT VERIFIED'),
                'Paired proposal/verifier/tree/online-training arguments and runtime environment; differences: '+str(diff))
        alignment=[p.get('alignment_version') for p in proofs.values()]
        checks['reward_advantage_alignment']=check('PASS' if all(v=='response_rows_v2' for v in alignment) else
            'NOT VERIFIED','All three trainers carry identity/reward/advantage through a common stable permutation; cross-repository loss/gradient tests required')
        for method,value in methods.items():
            value['inspection_status']=value['status']
            value['status']='NOT VERIFIED' if not evidence[method] else 'FAIL'
        row['checks']=checks
        row['status']='FAIL' if any(v['status']=='FAIL' for v in checks.values()) else 'NOT VERIFIED'
    lora_status=[row['checks']['loaded_initial_target_lora']['status'] for row in report['models']]
    report['initial_target_lora']['status']='FAIL' if 'FAIL' in lora_status else ('PASS' if all(v=='PASS' for v in lora_status) else 'NOT VERIFIED')
    report['status']='FAIL' if any(row['status']=='FAIL' for row in report['models']) else 'NOT VERIFIED'
    return report



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec-root',type=Path,default=ROOT.parent/'SpecNaacl')
    p.add_argument('--pure-root',type=Path,default=ROOT.parent/'puregrpo')
    p.add_argument('--model',choices=MODELS)
    p.add_argument('--use-environment',action='store_true')
    p.add_argument('--strict',action='store_true')
    p.add_argument('--output',type=Path)
    p.add_argument('--runtime-report',action='append',default=[],metavar='MODEL:METHOD:SUMMARY_JSON')
    a=p.parse_args();runtime_reports={}
    for item in a.runtime_report:
        model,method,path=item.split(':',2)
        if model not in MODELS or method not in ('medusa','medusa_reflex','fastgrpo','fastgrpo_reflex','puregrpo'):
            p.error('Unknown model/method in --runtime-report')
        runtime_reports[(model,method)]=json.loads(Path(path).read_text())
    report=audit(a.spec_root.resolve(),a.pure_root.resolve(),dict(os.environ) if a.use_environment else None,
                 (a.model,) if a.model else MODELS,runtime_reports)
    text=json.dumps(report,indent=2)+'\n'
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(text)
    if a.strict and report['status']!='PASS':sys.exit(1)


if __name__=='__main__':main()
