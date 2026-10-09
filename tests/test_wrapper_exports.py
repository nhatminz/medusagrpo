"""Model wrappers replace stale model configuration without touching resume controls."""
import os
from pathlib import Path
import shlex
import subprocess
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


def command(wrapper,env):
    result=subprocess.run(['bash',str(ROOT/wrapper)],cwd=ROOT,env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    line=next(line for line in result.stdout.splitlines() if line.startswith('Command'))
    tokens=shlex.split(line.split(':',1)[1]);args={}
    for i,t in enumerate(tokens):
        if t.startswith('--') and t!='--standalone':
            key,sep,value=t[2:].partition('=')
            args[key]=value if sep else tokens[i+1]
    return args,result.stdout


@pytest.mark.parametrize('key',MODELS)
@pytest.mark.parametrize('prefix,suffix',[('train','_medusa'),('train','_reflex'),('train',''),('pretrain','')])
def test_wrappers_replace_previous_model_exports(key,prefix,suffix,tmp_path):
    env={k:v for k,v in os.environ.items() if k in ('PATH','HOME','LANG','LD_LIBRARY_PATH')}
    # Previous model config must not leak into the selected model's invocation.
    env.update(DRY_RUN='true',MODEL='/wrong/model',MODEL_TYPE='llama',
        OUTPUT_ROOT='/wrong/outputs',DATA_ROOT='/wrong/data',DATASET='gsm8k',DATASET_PATH='/wrong/train',
        TARGET_ADAPTER='/wrong/adapter',DRAFT_CHECKPOINT='/wrong/heads',
        PRETRAIN_MODEL_ROOT='/wrong/pretrain',MODEL_OUTPUT_ROOT='/wrong/pretrain_out',
        TRAIN_MODEL_ROOT='/wrong/train_out',COMMON_ENV='/wrong/common',MODEL_ENV='/wrong/config',
        LAUNCHER_ENV='/wrong/profile',PYTHON_BIN='/wrong/python',
        BATCH_SIZE='99',ACCUMULATION_STEPS='99',PRETRAIN_BATCH_SIZE='99',PRETRAIN_ACCUMULATION_STEPS='99',
        NPROC_PER_NODE='4',CUDA_VISIBLE_DEVICES='3',GENERATION_LENGTH_POLICY='per_response',
        PRETRAIN_DATASET='wrong',PRETRAIN_DATASET_PATH='/wrong/sharegpt',
        RUN_DIR=str(tmp_path/'explicit_run'),RUN_NAME='explicit_run',RESUME='auto')
    args,stdout=command(f'{prefix}_{key}{suffix}.sh',env)
    name,family,pb,pa=MODELS[key]
    assert args['model_dir']==f'/workspace/storage-shared/models/{name}'
    assert args['model_type']==family
    assert args['load_lora_path']==str(ROOT/f'outputs/initial_target/{key}_seed42')
    assert args['nproc_per_node']=='1'
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


def test_explicit_environment_mode_keeps_benchmark_overrides(tmp_path):
    env=dict(os.environ,LAUNCHER_USE_ENV='1',DRY_RUN='true',MODEL='/custom/Qwen',MODEL_TYPE='qwen2',
        OUTPUT_ROOT=str(tmp_path/'output'),TARGET_ADAPTER='/custom/initial',DRAFT_CHECKPOINT='/custom/heads',
        BATCH_SIZE='4',ACCUMULATION_STEPS='8',GENERATION_LENGTH_POLICY='per_response',
        RESUME='',RUN_DIR=str(tmp_path/'trial'),RUN_NAME='trial',MAX_TARGET_OPTIMIZER_STEPS='7')
    args,_=command('train_qwen25_3b_reflex.sh',env)
    assert args['model_dir']=='/custom/Qwen'
    assert args['load_lora_path']=='/custom/initial' and args['adapter_path']=='/custom/heads'
    assert (args['batch_size'],args['accumulation_steps'])==('4','8')
    assert args['generation_length_policy']=='per_response'
    assert args['max_target_optimizer_steps']=='7'


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
