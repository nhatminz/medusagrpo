"""Preflight must not validate a Qwen profile against its tiny synthetic model."""
import os
from pathlib import Path

import pytest

from helper.environment_checks import probe_training_runtime


@pytest.mark.parametrize('setting',['unset','empty','production'])
def test_probe_does_not_load_production_profiles_and_restores_environment(monkeypatch,tmp_path,setting,capsys):
    import medusa.generate
    names=('OPD_PROPOSAL_PROFILE','OPD_PROPOSAL_PROFILE_DIR')
    for name in names:
        if setting=='unset':monkeypatch.delenv(name,raising=False)
        else:monkeypatch.setenv(name,'' if setting=='empty' else str(tmp_path/name))
    before={name:os.environ.get(name) for name in names}
    original=medusa.generate.speculative_generate
    seen=[]
    def checked(*args,**kwargs):
        assert os.environ['OPD_PROPOSAL_PROFILE']==''
        directory=Path(os.environ['OPD_PROPOSAL_PROFILE_DIR'])
        assert directory.is_dir() and not list(directory.iterdir())
        seen.append(directory)
        return original(*args,**kwargs)
    monkeypatch.setattr(medusa.generate,'speculative_generate',checked)
    result=probe_training_runtime('cpu')
    assert result['architecture']=='medusa_parallel_3'
    assert result['target_forward_calls']>=1
    assert {name:os.environ.get(name) for name in names}==before
    assert len(seen)==1 and not seen[0].exists()
    lines=capsys.readouterr().out.splitlines()
    assert lines and all(line.startswith('[runtime probe] ') for line in lines)


def test_probe_failure_restores_profile_overrides(monkeypatch,tmp_path):
    import medusa.generate
    profile=str(tmp_path/'qwen-profile.json');directory=str(tmp_path)
    monkeypatch.setenv('OPD_PROPOSAL_PROFILE',profile)
    monkeypatch.setenv('OPD_PROPOSAL_PROFILE_DIR',directory)
    def failed(*args,**kwargs):
        assert os.environ['OPD_PROPOSAL_PROFILE']==''
        raise RuntimeError('synthetic execution failed')
    monkeypatch.setattr(medusa.generate,'speculative_generate',failed)
    with pytest.raises(RuntimeError,match='synthetic execution failed'):
        probe_training_runtime('cpu')
    assert os.environ['OPD_PROPOSAL_PROFILE']==profile
    assert os.environ['OPD_PROPOSAL_PROFILE_DIR']==directory
