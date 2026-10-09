#!/usr/bin/env python3
"""Validate supported installed packages and execute the FastGRPO training APIs.

requirements.txt remains the reproducible installation profile. Existing
environments may use compatible versions after import/API/execution checks.
"""

from __future__ import annotations

import argparse
import importlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
import sys
from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]


IMPORT_NAMES = {
    "cuda-tile": "cuda.tile",
    "huggingface-hub": "huggingface_hub",
    "latex2sympy2-extended": "latex2sympy2_extended",
    "math-verify": "math_verify",
    "openai-harmony": "openai_harmony",
    "pyyaml": "yaml",
    "typing-extensions": "typing_extensions",
}
# Interpreter admission is a lower bound, not a patch-release allowlist.
# The dependency pins/imports and CUDA checks below remain separate: accepting
# a newer interpreter does not guarantee wheels/runtime support for that version.
MIN_PYTHON_VERSION = (3, 10, 0)

# Bounded API families, including the user's existing B200 stack. Admission is
# followed by real import and execution checks; newer does not imply compatible.
COMPATIBLE_VERSIONS = {
    "torch": ">=2.5.1,<3", "transformers": ">=4.51.3,<6",
    "peft": ">=0.17.1,<0.22", "datasets": ">=4,<6",
    "accelerate": ">=1.10.1,<2", "safetensors": ">=0.6.2,<1",
    "numpy": ">=2.2.6,<3", "pandas": ">=2.3.2,<4",
    "tqdm": ">=4.67.1,<5", "math-verify": ">=0.8,<1",
    "latex2sympy2-extended": ">=1.10.2,<2", "sympy": ">=1.13.1,<2",
    "triton": ">=3.1,<4", "matplotlib": ">=3.10.6,<4",
    "packaging": ">=25,<27", "pytest": ">=8.4.2,<10",
}
REQUIRED_APIS = {
    "torch": ("save", "load", "_assert_async", "multinomial"),
    "transformers": ("AutoConfig", "AutoTokenizer", "AutoModelForCausalLM", "get_scheduler"),
    "triton": ("jit", "cdiv", "next_power_of_2"),
    "datasets": ("load_dataset",),
    "peft": (
        "get_peft_config", "get_peft_model", "LoraConfig", "TaskType", "PeftType",
        "get_peft_model_state_dict", "set_peft_model_state_dict", "PeftModel",
    ),
}


def version_matches(distribution, expected, actual, *, strict=False):
    actual = actual.split("+", 1)[0]
    if actual == expected:
        return True
    if strict:
        return False
    allowed = COMPATIBLE_VERSIONS.get(distribution.lower().replace("_", "-"))
    return allowed is not None and Version(actual) in SpecifierSet(allowed)


def validate_python_version():
    if sys.version_info[:3] >= MIN_PYTHON_VERSION:
        return
    minimum = ".".join(map(str, MIN_PYTHON_VERSION))
    recommended = ".".join(map(str, MIN_PYTHON_VERSION[:2]))
    environment = f"venv-py{recommended.replace('.', '')}"
    raise RuntimeError(
        f"Python >={minimum} is required; found {sys.version.split()[0]}\n"
        f"Interpreter: {sys.executable}\n"
        "Create a supported environment from the project directory:\n"
        f"  uv python install {recommended}\n"
        f"  uv venv --python {recommended} --seed {environment}\n"
        f"  source {environment}/bin/activate\n"
        '  export PYTHON_BIN="$(command -v python)"\n'
        "Install the project dependencies in this environment; see ENVIRONMENT.md."
    )


def pinned_requirements(path: Path):
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if match is None:
            raise RuntimeError(f"requirement is not exactly pinned: {line}")
        yield match.group(1), match.group(2)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", type=Path, default=REPO_ROOT / "requirements.txt")
    parser.add_argument("--strict-versions", action="store_true",
                        help="require the exact installation pins instead of compatible API families")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--python-only", action="store_true",
                        help="check the supported interpreter without importing dependencies")
    args = parser.parse_args(argv)
    validate_python_version()
    if args.python_only:
        return

    failures = []
    for distribution, expected in pinned_requirements(args.requirements):
        try:
            actual = version(distribution)
        except PackageNotFoundError:
            failures.append(f"{distribution}: not installed")
            continue
        if not version_matches(distribution, expected, actual, strict=args.strict_versions):
            supported = expected if args.strict_versions else COMPATIBLE_VERSIONS.get(distribution, expected)
            failures.append(f"{distribution}: supported {supported}, found {actual}")
            continue
        module_name = IMPORT_NAMES.get(distribution, distribution.replace("-", "_"))
        try:
            module = importlib.import_module(module_name)
            missing = [name for name in REQUIRED_APIS.get(distribution.lower(), ())
                       if not hasattr(module, name)]
            if missing:
                raise ImportError("required APIs missing: " + ", ".join(missing))
        except Exception as exc:
            failures.append(
                f"{distribution}: import {module_name} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if actual.split("+", 1)[0] != expected:
            print(f"{distribution}: compatible version {actual} (install pin {expected}); checking runtime APIs")
    if failures:
        raise RuntimeError("environment validation failed:\n- " + "\n- ".join(failures))

    torch = importlib.import_module("torch")
    if args.require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from helper.environment_checks import probe_training_runtime
        result = probe_training_runtime('cuda' if args.require_cuda else 'cpu')
    except Exception as exc:
        raise RuntimeError(f"FastGRPO runtime compatibility probe failed: {type(exc).__name__}: {exc}") from exc
    print(f"Runtime compatibility probe passed: {result}")
    print(
        "environment validation passed: "
        f"python={sys.version.split()[0]} torch={torch.__version__} "
        f"cuda={torch.version.cuda}"
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
