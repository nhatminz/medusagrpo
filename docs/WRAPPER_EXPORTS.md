# Per-model shell configuration

All 28 train/pretrain wrappers for the seven supported models now contain an
explicit export block. Running a different model resets model, family, data,
initial LoRA, heads, output roots, active Python, CUDA device/world size,
BF16/SDPA, batch/accumulation, seed, length policy and configuration file paths.
Inherited values of these variables are replaced in the launched child shell;
the parent terminal is not modified. Model paths retain the server defaults
from the inspected configs. Outputs are relative to the Medusa repository.

Default initial target checkpoint:
`outputs/initial_target/<model_key>_seed42`.
Default pretrained heads:
`outputs/pretrain/<model_key>/latest_checkpoint`.
Create/pretrain these once; wrappers do not create missing checkpoints or
silently substitute another model's initialization. Training wrappers enable
`GRPO_BENCHMARK=1` so actual loaded target and heads hashes are recorded.

Activate the appropriate environment once, then run directly:

```bash
RESUME=auto bash train_qwen25_1p5b_reflex.sh
RESUME=auto bash train_qwen25_3b_reflex.sh
RESUME=auto bash train_qwen25_7b_reflex.sh
RESUME=auto bash train_qwen25_14b_reflex.sh
RESUME=auto bash train_qwen3_1p7b_reflex.sh
RESUME=auto bash train_qwen3_4b_reflex.sh
```

`train_<model>.sh` aliases select Reflex; `_medusa.sh` selects baseline Medusa.
`pretrain_<model>.sh` selects that model's initial target and pretraining batch.
Training batch/accumulation stays 8/4 to match actual SpecNaacl wrappers;
pretraining keeps each model's existing settings. No decoding/OPD code changes.

`RESUME`, `RUN_DIR`, `RUN_NAME`, `DRY_RUN`, training budgets and OPD options are
still accepted as run controls. Automatic resume uses the existing model/method
active-run link; an explicit RUN_DIR selects a specific run. A completed run or
missing compatible checkpoint may cause `RESUME=auto` to start a new run under
the existing launcher rules. Do not reuse RUN_DIR from a different model.

Edit a wrapper's export block to change its permanent configuration. Its paired
method wrapper must use the same target/head checkpoints for fairness. For
explicit environment overrides (custom experiments/ablation/tooling), opt in:

```bash
LAUNCHER_USE_ENV=1 TARGET_ADAPTER=/custom/initial DRAFT_CHECKPOINT=/custom/heads \
  bash train_qwen25_3b_reflex.sh
```

The paired benchmark tool opts in to preserve its explicit checkpoint paths,
tree settings and trial budgets. Default fairness audits inspect the real model
wrappers. They supply the SAME per-model initial target path to baseline dry-run
commands, record that binding as `shared_initialization_input`, and still
require loaded tensor proofs. This is an explicit input for comparing the
experiment recipe, not proof that baseline defaults select this checkpoint.
`--use-environment` opts in to environment mode for custom configuration audits.

Regression tests launch all 28 wrappers with deliberately stale exports and
check the actual generated CLI commands. They also cover environment opt-in,
explicit RUN_DIR preservation and automatic method-specific resume detection.
