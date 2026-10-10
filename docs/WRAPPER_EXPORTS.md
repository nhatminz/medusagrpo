# Per-model shell configuration

All 21 train scripts and 7 pretrain scripts use caller environment values first.
`LAUNCHER_USE_ENV=1` remains compatible but is no longer required. Unset or empty
values use the same per-model defaults as before. Clear an old exported model
or checkpoint variable if you want the new model wrapper to choose its default.

Default initial target checkpoint:
`outputs/initial_target/<model_key>_seed42`.
Default pretrained heads:
`outputs/pretrain/<model_key>/latest_checkpoint`.
Create these once; wrappers do not create missing checkpoints. Train wrappers
default to `GRPO_BENCHMARK=1`, preserving actual loaded tensor hash checks.

Training batch/accumulation comes from the shared launcher's existing
`configs/<model>/launcher.env` layer (8/4) followed by model config. Explicit
`BATCH_SIZE`/`ACCUMULATION_STEPS` values win. `LAUNCHER_ENV=/dev/null` still
selects model-config defaults. Pretrain batch/accumulation keeps each model's
existing defaults and accepts `PRETRAIN_BATCH_SIZE`/`PRETRAIN_ACCUMULATION_STEPS`.

Model wrappers no longer assign `DATASET_PATH` or `PRETRAIN_DATASET_PATH`.
Shared launchers resolve training `DATASET=gsm8k|simplelr|dapo` and pretraining
`PRETRAIN_DATASET=sharegpt|gsm8k|simplelr|dapo` beneath `DATA_ROOT`. Explicit
paths are preserved, including paths containing spaces.

```bash
CUDA_VISIBLE_DEVICES=2 DATASET=gsm8k DRY_RUN=true \
  bash train_qwen3_1p7b_reflex.sh

CUDA_VISIBLE_DEVICES=2 DATASET=dapo DATASET_PATH=/custom/train.parquet \
  bash train_qwen25_3b_medusa.sh

CUDA_VISIBLE_DEVICES=2 PRETRAIN_DATASET=sharegpt \
  PRETRAIN_DATASET_PATH=/custom/sharegpt.json bash pretrain_qwen25_3b.sh

TARGET_ADAPTER=/custom/initial DRAFT_CHECKPOINT=/custom/heads \
  RESUME=auto bash train_qwen25_3b_reflex.sh
```

`train_<model>.sh` aliases select Reflex; `_medusa.sh` selects baseline Medusa.
`RESUME`, `RUN_DIR`, `RUN_NAME`, budgets and OPD options remain overrideable.
Existing model/method resume rules are unchanged.

The shared config defaults `OPD_FRONTIER_WEIGHT` to 0.5. Explicit environment
weights win. Visited weight stays 1.0, selection stays `visited_capped_frontier`,
and the frontier cap stays 2. No generation, ReflexOPD gradient, tree or kernel
implementation changes accompany this launcher update.

The paired benchmark inherits GPU, dataset, paths and weights, uses model-specific
checkpoint defaults if none are supplied, and retains paired loaded-tensor checks.
Both methods use the same initial target, heads, seed, tree and training budget.
Profiling remains off unless `--profile` is provided. Throughput computation,
`results.json` and `selected_tree.env` are unchanged.

```bash
CUDA_VISIBLE_DEVICES=2 DATASET=gsm8k python scripts/benchmark_pair.py \
  --model qwen3_1p7b --steps 10 --trials 3 --budgets 512:12 --dry-run
# Remove --dry-run for the actual GPU benchmark.
```

Regression tests cover defaults, GPU/dataset overrides, explicit paths and
training settings for all 28 wrappers; legacy `LAUNCHER_USE_ENV=1`; automatic
resume; and paired benchmark dry-runs for all seven models with both default and
explicit checkpoints. Dry-runs do not allocate a GPU or fabricate timing results.
The old `WRAPPER_EXPORTS_VALIDATION.json` describes the previous overwrite policy;
current validation is recorded in `LAUNCHER_OVERRIDES_VALIDATION.json`.
