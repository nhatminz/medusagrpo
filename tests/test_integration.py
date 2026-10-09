import ast
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
MODELS=('qwen25_1p5b','qwen25_3b','qwen25_7b','qwen25_14b','qwen3_1p7b','qwen3_4b','llama31_8b')


def run(*args):
    p=subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,text=True,capture_output=True,
                     env=dict(os.environ,OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false'),timeout=120)
    assert p.returncode==0,p.stdout[-3000:]+p.stderr[-3000:]
    return p


def test_copied_target_loss_reward_and_data_are_exact_source_snapshot():
    manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    for name in ('helper/fastgrpo_training.py','helper/rewards.py','helper/get_QAs.py','helper/pretrain_data.py','helper/checkpointing.py','helper/opd_sampling.py'):
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==manifest[name]['sha256']
    src=ROOT.parent/'SpecNaacl/grpo_speculative.py'
    if src.exists():
        original=ast.parse(src.read_text());current=ast.parse((ROOT/'grpo_speculative.py').read_text())
        for name in ('compute_target_loss_and_backward','TrainDataCollator'):
            a=next(n for n in original.body if getattr(n,'name',None)==name)
            b=next(n for n in current.body if getattr(n,'name',None)==name)
            assert ast.dump(a)==ast.dump(b)
        for model in MODELS:
            old=(ROOT.parent/'SpecNaacl/configs'/model/'b200.env').read_text()
            assert (ROOT/'configs'/model/'b200.env').read_text().startswith(old)


@pytest.mark.parametrize('model',MODELS)
def test_all_model_launchers_dry_run(model,tmp_path):
    env=dict(os.environ,DRY_RUN='true',PYTHON_BIN=sys.executable)
    for suffix in ('_medusa','_reflex',''):
        path=ROOT/f'train_{model}{suffix}.sh'
        p=subprocess.run(['bash',str(path)],env=env,text=True,capture_output=True)
        assert p.returncode==0,p.stderr
        assert 'MedusaGRPO/outputs/train' in p.stdout
        assert '/workspace/storage-shared/' in p.stdout
        assert 'medusa_reflex' in p.stdout if suffix!='_medusa' else '--method medusa ' in p.stdout
    p=subprocess.run(['bash',str(ROOT/f'pretrain_{model}.sh')],env=env,text=True,capture_output=True)
    assert p.returncode==0,p.stderr
    assert '--num_epochs 5' in p.stdout and '--max_length 2048' in p.stdout
    assert 'ShareGPT_V4.3_unfiltered_cleaned_split.json' in p.stdout


def nested_equal(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:nested_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):nested_equal(x,y)
    else:assert a==b


@pytest.mark.parametrize('method',['medusa','medusa_reflex'])
def test_real_training_checkpoint_resume_logging(method,tmp_path):
    fixture=ROOT/'tests/tiny_runner.py'
    run(fixture,tmp_path,method,'full')
    run(fixture,tmp_path,method,'split','--max_target_optimizer_steps','1')
    checkpoint=tmp_path/'split/resume/latest.pt'
    first=torch.load(checkpoint,weights_only=False)
    assert first['rank_states'][0]['batch_data']['target_optimizer_steps']==1
    run(fixture,tmp_path,method,'split','--resume_checkpoint',checkpoint)
    full=torch.load(tmp_path/'full/resume/latest.pt',weights_only=False)
    split=torch.load(checkpoint,weights_only=False)
    nested_equal(full['draft_model'],split['draft_model'])
    nested_equal(full['target_lora'],split['target_lora'])
    nested_equal(full['optimizer_target'],split['optimizer_target'])
    nested_equal(full['optimizer_draft'],split['optimizer_draft'])
    for name in ('full','split'):
        summary=json.loads((tmp_path/name/'summary.json').read_text())
        assert summary['target_optimizer_steps']==2
        rows=list(csv.DictReader((tmp_path/name/'logs/timing.csv').open()))
        assert len(rows)==2
        rollouts=list(csv.DictReader((tmp_path/name/'logs/rollout_timing.csv').open()))
        assert len(rollouts)==2
        assert int(rollouts[-1]['head3_proposed_nodes'])>0
        assert float(rollouts[-1]['head3_training_loss'])>0
        assert float(rollouts[-1]['iter_aal'])>=1
        for h in (1,2,3):
            for suffix in ('active_rounds','proposed_nodes','accepted_tokens'):
                key=f'head{h}_{suffix}'
                assert summary['medusa_totals'][key]==sum(int(r[key]) for r in rollouts)
        assert summary['medusa_target_forward_calls']==summary['medusa_totals']['target_forward_calls']


def test_sharegpt_pretrain_five_epochs_remainder_and_resume(tmp_path):
    run(ROOT/'tests/tiny_runner.py',tmp_path)
    def pretrain(out,*extra):
        return run(ROOT/'train_draft.py','--model_dir',tmp_path/'model','--dataset_dir',tmp_path/'sharegpt.json',
            '--saved_model_dir',tmp_path/out/'checkpoints','--log_dir',tmp_path/out/'logs','--model_output_root',tmp_path/out,
            '--num_epochs','5','--batch_size','2','--accumulation_steps','2','--dtype','fp32','--device','cpu',
            '--num_workers','0','--max_length','256','--load_lora_path',tmp_path/'initial_lora',*extra)
    output=pretrain('pretrain')
    assert 'Pretrain epoch' in output.stderr
    pretrain('split','--max_steps','1')
    pretrain('split','--resume','auto')
    a=torch.load(tmp_path/'pretrain/latest_checkpoint/draft.pth',weights_only=False)
    b=torch.load(tmp_path/'split/latest_checkpoint/draft.pth',weights_only=False)
    nested_equal(a,b)
    completed=json.loads((tmp_path/'pretrain/checkpoints/pretrain_complete.json').read_text())
    assert completed['epochs']==5 and completed['step']==10
    rows=[json.loads(s) for s in (tmp_path/'pretrain/logs/metrics.jsonl').read_text().splitlines()]
    assert all(row['head3_training_loss']>0 for row in rows)


def test_prompt_budget_with_reward_filtered_iterations(tmp_path):
    run(ROOT/'tests/tiny_runner.py',tmp_path,'medusa','filtered','--test_constant_rewards','--max_rollout_prompts','1')
    summary=json.loads((tmp_path/'filtered/summary.json').read_text())
    assert summary['rollout_prompts_seen']==1
    assert summary['target_optimizer_steps']==0
    rows=list(csv.DictReader((tmp_path/'filtered/logs/rollout_timing.csv').open()))
    assert len(rows)==1 and int(rows[0]['eligible_prompts'])==0
    assert int(rows[0]['total_prompts'])==1


def test_missing_pretrained_heads_never_falls_back_to_random(tmp_path):
    run(ROOT/'tests/tiny_runner.py',tmp_path)
    (tmp_path/'heads/draft.pth').unlink()
    p=subprocess.run([sys.executable,str(ROOT/'tests/tiny_runner.py'),str(tmp_path),'medusa','missing'],cwd=ROOT,
        text=True,capture_output=True,env=dict(os.environ,OMP_NUM_THREADS='1'))
    assert p.returncode!=0 and 'draft.pth' in p.stderr


def test_disabled_plugin_does_not_update_projector(tmp_path,monkeypatch):
    monkeypatch.setenv('OPD_ENABLED','0')
    run(ROOT/'tests/tiny_runner.py',tmp_path,'medusa_reflex','disabled')
    initial=torch.load(tmp_path/'heads/draft.pth',weights_only=True)['draft_model']
    checkpoint=torch.load(tmp_path/'disabled/resume/latest.pt',weights_only=False)
    assert torch.equal(initial['opd_projector'],checkpoint['draft_model']['opd_projector'])
    assert not checkpoint['opd_enabled']
