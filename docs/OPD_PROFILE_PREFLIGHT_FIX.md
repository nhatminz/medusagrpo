# OPD proposal profiles during environment validation

The launcher runs `scripts/validate_environment.py --require-cuda` before
loading the production model. Its compatibility probe constructs a synthetic
Qwen2 model with vocabulary 32 and FP32 weights. Previously it inherited
`OPD_PROPOSAL_PROFILE` and `OPD_PROPOSAL_PROFILE_DIR` from training.

This caused two misleading preflight results: automatic lookup warned that no
profile existed for the synthetic workload, and explicitly selecting a tuned
Qwen BF16 profile failed with `incompatible proposal profile: vocab, dtype`.
Neither result established that the production model lacked a compatible profile.

`helper/environment_checks.py` now runs the synthetic probe with an empty
temporary profile directory and no explicit profile. It suppresses only the
expected missing-calibration warning within this probe, while retaining actual
generation and auto-dispatch. All original profile environment values and warning
filters are restored even if the probe fails. Production profile discovery and
strict execution-key validation are unchanged, as are OPD algorithms and kernels.
The probe announces that it is using the synthetic V=32 FP32 workload and
prefixes its own output with `[runtime probe]` to distinguish it from training.

Local validation: **16 passed, 3 skipped**, plus Python compileall. The three
CUDA probe variants (no profile, automatic directory, explicit Qwen BF16 profile)
were skipped because the local PyTorch installation has no CUDA runtime; B200
correctness has not been verified by this run.

After copying the updated MedusaGRPO files to the server, run:

```bash
python -m pytest \
  tests/test_environment_probe_profiles.py \
  tests/test_opd_profile_diagnostics.py \
  tests/test_training_opd.py \
  tests/test_cuda.py::test_gpu_launcher_runtime_probe_supports_float32_target_autocast \
  -q -rs

bash train_qwen25_1p5b_reflex.sh
```

Retuning is not required to fix this preflight bug. If production lookup still
reports an incompatible profile, diagnose that production execution key separately.
