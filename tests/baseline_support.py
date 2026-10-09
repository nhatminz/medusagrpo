"""Optional cross-repository evidence, separate from Medusa's own checks."""
import os
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def baseline_root(repository):
    if repository == 'MedusaGRPO':
        return ROOT
    variable = 'SPEC_ROOT' if repository == 'SpecNaacl' else 'PUREGRPO_ROOT'
    return Path(os.environ.get(variable, str(ROOT.parent / repository))).resolve()


def require_baseline_files(root, *relative_paths):
    missing = [str(root / name) for name in relative_paths if not (root / name).is_file()]
    if missing:
        message = 'NOT VERIFIED: cross-repository source files unavailable: ' + ', '.join(missing)
        if root.resolve() == ROOT:
            pytest.fail(message)
        pytest.skip(message)


def source_contract_complete(root):
    return all((root / name).is_file() for name in (
        'training.py', 'grpo.py', 'helper/target.py', 'helper/grpo_core.py',
        'helper/train_ops.py', 'helper/response_alignment.py', 'helper/shared_adapter.py'))
