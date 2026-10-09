"""Reproduce incomplete PureGRPO deployment without touching sibling sources."""
from pathlib import Path
import re
import shutil
import pytest
from baseline_support import ROOT,baseline_root,require_baseline_files
from scripts.check_fairness import audit


@pytest.fixture
def partial_pure(tmp_path):
    source=baseline_root('puregrpo')
    needed=('training.py','grpo.py','train_qwen25_1p5b.sh','scripts/launch/train_model.sh',
        'helper/grpo_core.py','helper/target.py','helper/autoregressive.py',
        'helper/rewards.py','helper/get_QAs.py',
        'configs/_shared/b200_common.env','configs/qwen25_1p5b/b200.env')
    require_baseline_files(source,*needed)
    root=tmp_path/'partial_pure'
    for name in needed:
        dst=root/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,dst)
    return root


def test_missing_optional_source_has_explicit_not_verified_reason(tmp_path):
    with pytest.raises(pytest.skip.Exception,match='NOT VERIFIED'):
        require_baseline_files(tmp_path/'missing_pure','helper/shared_adapter.py')
    # Missing Medusa's own implementation must remain a failure.
    with pytest.raises(pytest.fail.Exception,match='NOT VERIFIED'):
        require_baseline_files(ROOT,'helper/does_not_exist.py')


def test_partial_pure_audit_keeps_real_configuration_failures(partial_pure):
    wrapper=partial_pure/'train_qwen25_1p5b.sh'
    original=wrapper.read_text()
    changed=re.sub(r'(?m)^(?:export )?TARGET_LR=.*$', 'export TARGET_LR=2e-6', original)
    assert changed!=original
    wrapper.write_text(changed)
    report=audit(baseline_root('SpecNaacl'),partial_pure,models=('qwen25_1p5b',))
    row=report['models'][0]
    assert report['configuration_status']=='PASS' # Medusa/SpecNaacl scope
    assert report['status']=='FAIL' # known Pure LR mismatch remains visible
    assert row['methods']['puregrpo']['configuration_status']=='FAIL'
    assert row['methods']['puregrpo']['status']=='NOT VERIFIED'
    assert row['checks']['baseline_initialization_audit_support']['status']=='NOT VERIFIED'
    assert row['checks']['loaded_initial_target_lora']['status']=='NOT VERIFIED'
    assert not row['specnaacl_configuration_mismatches']
    assert any(v['method']=='puregrpo' and v['field']=='target_lr' for v in row['configuration_mismatches'])


def test_missing_pure_cannot_make_configuration_audit_pass(tmp_path):
    report=audit(baseline_root('SpecNaacl'),tmp_path/'missing',models=('qwen25_1p5b',))
    row=report['models'][0]
    assert report['status']=='NOT VERIFIED'
    assert row['checks']['launcher_configuration']['status']=='NOT VERIFIED'
    assert row['methods']['puregrpo']['configuration_status']=='NOT VERIFIED'
    assert row['checks']['baseline_initialization_audit_support']['status']=='NOT VERIFIED'
