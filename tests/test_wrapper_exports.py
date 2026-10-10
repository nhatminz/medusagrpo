"""Exercise effective launcher defaults and caller overrides for every model."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
MODELS={
    'qwen25_1p5b':('Qwen2.5-1.5B-Instruct','qwen2',16,2),
    'qwen25_3b':('Qwen2.5-3B-Instruct','qwen2',8,4),
    'qwen25_7b':('Qwen2.5-7B-Instruct','qwen2',4,8),
    'qwen25_14b':('Qwen2.5-14B-Instruct','qwen2',2,16),
    'qwen3_1p7b':('Qwen3-1.7B','qwen3',16,2),
    'qwen3_4b':('Qwen3-4B','qwen3',8,4),
    'llama31_8b':('Llama-3.1-8B-Instruct','llama',4,8),
}


def parse_command(line):
    tokens=shlex.split(line.split(':',1)[1]);args={}
    for i,t in enumerate(tokens):
        if t.startswith('--') and t!='--standalone':
            key,sep,value=t[2:].partition('=')
            args[key]=value if sep else tokens[i+1]
    return args


def clean_env(**overrides):
    env={k:v for k,v in os.environ.items() if k in ('PATH','HOME','LANG','LD_LIBRARY_PATH')}
    env.update(DRY_RUN='true',PYTHON_BIN=sys.executable,RESUME='')
    env.update(overrides)
    return env


def command(wrapper,env):
    result=subprocess.run(['bash',str(ROOT/wrapper)],cwd=ROOT,env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    line=next(line for line in result.stdout.splitlines() if line.startswith('Command'))
    return parse_command(line),result.stdout


WRAPPERS=[('train','_medusa'),('train','_reflex'),('train',''),('pretrain','')]
DATASETS={
    'gsm8k':'gsm8k/main/train-00000-of-00001.parquet',
    'simplelr':'simplelr_abel_level3to5/train.parquet',
    'dapo':'DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet',
}


@pytest.mark.parametrize('key',MODELS)
@pytest.mark.parametrize('prefix,suffix',WRAPPERS)
def test_wrappers_keep_model_defaults(key,prefix,suffix,tmp_path):
    env=clean_env(RUN_DIR=str(tmp_path/'explicit_run'),RUN_NAME='explicit_run',RESUME='auto')
    args,stdout=command(f'{prefix}_{key}{suffix}.sh',env)
    name,family,pb,pa=MODELS[key]
    assert args['model_dir']==f'/workspace/storage-shared/models/{name}'
    assert args['model_type']==family
    assert args['load_lora_path']==str(ROOT/f'outputs/initial_target/{key}_seed42')
    assert args['nproc_per_node']=='1'
    assert 'CUDA_VISIBLE_DEVICES: 0' in stdout
    assert str(tmp_path/'explicit_run') in stdout
    if prefix=='train':
        assert args['adapter_path']==str(ROOT/f'outputs/pretrain/{key}/latest_checkpoint')
        assert (args['batch_size'],args['accumulation_steps'])==('8','4')
        assert args['generation_length_policy']=='specnaacl_compatible'
        assert args['dataset_path']=='/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/train.parquet'
    else:
        assert (args['batch_size'],args['accumulation_steps'])==(str(pb),str(pa))
        assert args['dataset_dir']=='/workspace/storage-shared/nlp/minhpn19/data/sharegpt/ShareGPT_V4.3_unfiltered_cleaned_split.json'
    assert not (tmp_path/'explicit_run').exists() # dry-run never writes outputs


@pytest.mark.parametrize('key',MODELS)
@pytest.mark.parametrize('prefix,suffix',WRAPPERS)
@pytest.mark.parametrize('dataset',DATASETS)
def test_gpu_and_dataset_overrides_without_opt_in(key,prefix,suffix,dataset,tmp_path):
    env=clean_env(CUDA_VISIBLE_DEVICES='2',DATA_ROOT=str(tmp_path/'data'),
                  DATASET=dataset,PRETRAIN_DATASET=dataset)
    args,stdout=command(f'{prefix}_{key}{suffix}.sh',env)
    assert 'CUDA_VISIBLE_DEVICES: 2' in stdout
    expected=str(tmp_path/'data'/DATASETS[dataset])
    if prefix=='train':
        assert args['dataset_path']==expected
        assert args['train_option']==dict(gsm8k='gsm8k',simplelr='simplelr_abel_level3to5',dapo='DAPO-math')[dataset]
    else:
        assert f'Dataset: {expected}' in stdout


@pytest.mark.parametrize('key',MODELS)
@pytest.mark.parametrize('prefix,suffix',WRAPPERS)
@pytest.mark.parametrize('legacy_flag',[False,True])
def test_explicit_environment_keeps_paths_and_training_overrides(key,prefix,suffix,legacy_flag,tmp_path):
    env=clean_env(MODEL='/custom/Qwen',MODEL_TYPE='qwen2',CUDA_VISIBLE_DEVICES='2,3',NPROC_PER_NODE='2',
        OUTPUT_ROOT=str(tmp_path/'output'),TARGET_ADAPTER='/custom/initial',DRAFT_CHECKPOINT='/custom/heads',
        DATASET='gsm8k',DATASET_PATH='/custom/gsm data.parquet',
        PRETRAIN_DATASET='sharegpt',PRETRAIN_DATASET_PATH='/custom/pretrain data.json',
        BATCH_SIZE='4',ACCUMULATION_STEPS='8',PRETRAIN_BATCH_SIZE='3',PRETRAIN_ACCUMULATION_STEPS='6',
        GENERATION_LENGTH_POLICY='per_response',OPD_FRONTIER_WEIGHT='0.75',TRAIN_SUBSET_SEED='7',
        RUN_DIR=str(tmp_path/'trial'),RUN_NAME='trial',MAX_TARGET_OPTIMIZER_STEPS='7')
    if legacy_flag:env['LAUNCHER_USE_ENV']='1'
    args,stdout=command(f'{prefix}_{key}{suffix}.sh',env)
    assert args['model_dir']=='/custom/Qwen'
    assert args['model_type']=='qwen2' and args['load_lora_path']=='/custom/initial'
    assert args['nproc_per_node']=='2' and 'CUDA_VISIBLE_DEVICES: 2,3' in stdout
    assert str(tmp_path/'trial') in stdout
    if prefix=='train':
        assert args['adapter_path']=='/custom/heads'
        assert args['dataset_path']=='/custom/gsm data.parquet'
        assert (args['batch_size'],args['accumulation_steps'])==('4','8')
        assert args['generation_length_policy']=='per_response'
        assert args['max_target_optimizer_steps']=='7' and args['seed']=='7'
        if suffix!='_medusa':assert args['opd_frontier_weight']=='0.75'
    else:
        assert args['dataset_dir']=='/custom/pretrain data.json'
        assert (args['batch_size'],args['accumulation_steps'])==('3','6')
        assert args['seed']=='7'


@pytest.mark.parametrize('key',MODELS)
@pytest.mark.parametrize('custom',[False,True])
def test_benchmark_pair_dry_run_preserves_shared_configuration(key,custom,tmp_path):
    env=clean_env(CUDA_VISIBLE_DEVICES='2',DATASET='gsm8k',TRAIN_SUBSET_SEED='17',
                  OPD_PROFILE='1') # benchmark profiling must still default off
    if custom:
        env.update(DATASET_PATH='/custom/train data.parquet',TARGET_ADAPTER='/custom/initial',
                   DRAFT_CHECKPOINT='/custom/heads',OPD_FRONTIER_WEIGHT='0.75')
    output=tmp_path/'benchmark'
    result=subprocess.run([sys.executable,str(ROOT/'scripts/benchmark_pair.py'),
        '--model',key,'--dry-run','--steps','2','--trials','1','--budgets','512:12',
        '--output',str(output)],env=env,cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    commands=[parse_command(line) for line in result.stdout.splitlines() if line.startswith('Command')]
    assert len(commands)==2
    medusa,reflex=commands
    assert medusa['method']=='medusa' and reflex['method']=='medusa_reflex'
    for setting in ('model_dir','adapter_path','load_lora_path','dataset_path','seed',
                    'train_subset_seed','batch_size','accumulation_steps','max_target_optimizer_steps',
                    'max_rollout_prompts','verification_capacity','generation_length_policy'):
        assert medusa[setting]==reflex[setting]
    assert medusa['model_dir']==f'/workspace/storage-shared/models/{MODELS[key][0]}'
    assert medusa['load_lora_path']==('/custom/initial' if custom else str(ROOT/f'outputs/initial_target/{key}_seed42'))
    assert medusa['adapter_path']==('/custom/heads' if custom else str(ROOT/f'outputs/pretrain/{key}/latest_checkpoint'))
    assert medusa['dataset_path']==('/custom/train data.parquet' if custom else '/workspace/storage-shared/nlp/minhpn19/data/'+DATASETS['gsm8k'])
    assert medusa['seed']=='17' and medusa['max_target_optimizer_steps']=='2'
    assert reflex['opd_frontier_weight']==('0.75' if custom else '0.5')
    assert reflex['opd_visited_weight']=='1.0' and reflex['opd_profile']=='0'
    assert result.stdout.count('CUDA_VISIBLE_DEVICES: 2')==2
    assert result.stdout.count('CPEAK_NODES=512 MAX_TREE_NODES_PER_SEQ=12 TOPK=4,3,2')==2
    assert not output.exists() # no fabricated benchmark artifacts from dry-run


def test_production_benchmark_admits_model_checkpoint_defaults(monkeypatch,tmp_path):
    """Reach the real launcher with its defaults, then stop before GPU work."""
    import torch
    from scripts import benchmark_pair
    monkeypatch.delenv('TARGET_ADAPTER',raising=False)
    monkeypatch.delenv('DRAFT_CHECKPOINT',raising=False)
    monkeypatch.setattr(sys,'argv',['benchmark_pair.py','--model','qwen3_1p7b',
        '--steps','1','--trials','1','--output',str(tmp_path/'benchmark')])
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    class LauncherReached(Exception):pass
    def stop_at_launcher(args,*,env,cwd,check):
        assert args==['bash',str(ROOT/'train_qwen3_1p7b_medusa.sh')]
        assert not env.get('TARGET_ADAPTER') and not env.get('DRAFT_CHECKPOINT')
        assert env['DRY_RUN']=='false' and env['OPD_PROFILE']=='0'
        raise LauncherReached
    monkeypatch.setattr(benchmark_pair.subprocess,'run',stop_at_launcher)
    with pytest.raises(LauncherReached):benchmark_pair.main()


def test_resume_auto_finds_model_method_checkpoint(tmp_path):
    env=dict(os.environ,DRY_RUN='true',LAUNCHER_USE_ENV='1',OUTPUT_ROOT=str(tmp_path),
        TRAIN_MODEL_ROOT=str(tmp_path/'train/qwen25_3b'),RESUME='auto')
    env.pop('RUN_DIR',None);env.pop('RUN_NAME',None)
    active=tmp_path/'train/qwen25_3b/existing__method-medusa_reflex'
    checkpoint=active/'checkpoints/resume/latest.pt'
    checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'fixture')
    (active.parent/'active_run_medusa_reflex').symlink_to(active)
    args,_=command('train_qwen25_3b_reflex.sh',env)
    assert args['resume_checkpoint']==str(checkpoint)
    assert args['summary_file']==str(active/'summary.json')
