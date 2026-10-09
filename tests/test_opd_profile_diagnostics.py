"""A saved summary must not hide the reason an execution profile is rejected."""
import json

import pytest

from helper.opd_profiles import describe_profile_search,discover_profile,execution_key


def runtime_key():
    hardware=dict(gpu='NVIDIA B200',compute_capability=[10,0],torch='2.13.0+cu130',
                  triton='3.7.1',cuda='13.0',kernel_sha256='current-kernel')
    return execution_key(hardware,151936,8,'bf16',16)


def payload(key):
    return dict(execution_key=key,records=[dict(contexts=1,trials=[
        dict(slots=0,sparse=1.,fused=2.,gemm=3.)])])


@pytest.mark.parametrize('field,value',[
    ('dtype','torch.float32'),('kernel_sha256','old-kernel'),('torch','2.8.0'),
    ('gpu','NVIDIA A100'),('topk',8),
])
def test_mismatch_reports_actual_and_runtime_without_accepting_profile(tmp_path,field,value):
    key=runtime_key();saved=dict(key,**{field:value})
    (tmp_path/'measured.json').write_text(json.dumps(payload(saved)))
    (tmp_path/'models_summary.json').write_text(json.dumps(dict(models=[dict(profile='measured.json')])))
    assert discover_profile(tmp_path,key)==(None,None)
    reason=describe_profile_search(tmp_path,key)
    assert str(tmp_path) in reason and 'Expected profile:' in reason
    assert f'{field}: profile={value!r}, runtime={key[field]!r}' in reason
    assert 'models_summary.json: no execution_key' in reason


def test_renamed_matching_profile_is_still_discovered(tmp_path):
    key=runtime_key();path=tmp_path/'renamed.json';saved=payload(key)
    path.write_text(json.dumps(saved))
    assert discover_profile(tmp_path,key)==(path,saved)


def test_missing_empty_and_invalid_directories_have_actionable_diagnostics(tmp_path):
    key=runtime_key()
    assert 'does not exist' in describe_profile_search(tmp_path/'missing',key)
    assert 'No proposal JSON profiles found' in describe_profile_search(tmp_path,key)
    (tmp_path/'broken.json').write_text('{')
    (tmp_path/'invalid.json').write_text(json.dumps(dict(execution_key=key,records=[])))
    reason=describe_profile_search(tmp_path,key)
    assert 'broken.json:' in reason
    assert 'invalid.json: empty measured proposal profile' in reason
