# Medusa-only length and configuration fairness revision

Only MedusaGRPO was edited. The before/after SHA256 inventory checks 443
SpecNaacl/puregrpo files without changes. Target loss, reward alignment,
target optimizer, batched head training, tree construction, KV compaction,
OPD selection/updates and kernels retain their existing implementation.

## Configuration results

The checker executes dry-run wrappers, extracts actual CLI parser types and
defaults without importing trainers, resolves generation-call bindings, and
compares model configuration defaults separately. Reward/data bytes,
collator/loss AST, LoRA recipe and AdamW options match the inspected SpecNaacl.

| Model | Model config batch/accumulation | Effective launch batch/accumulation | Config vs SpecNaacl | Full runtime fairness |
|---|---:|---:|---|---|
| Qwen2.5-1.5B | 16/2 | 8/4 | PASS | NOT VERIFIED |
| Qwen2.5-3B | 8/4 | 8/4 | PASS | NOT VERIFIED |
| Qwen2.5-7B | 8/4 | 8/4 | PASS | NOT VERIFIED |
| Qwen2.5-14B | 4/8 | 8/4 | PASS | NOT VERIFIED |
| Qwen3-1.7B | 16/2 | 8/4 | PASS | NOT VERIFIED |
| Qwen3-4B | 8/4 | 8/4 | PASS | NOT VERIFIED |
| Llama3.1-8B | 8/4 | 8/4 | PASS | NOT VERIFIED |

The current upstream wrappers explicitly choose 8/4 for all seven models, before
sourcing model configs. Medusa mirrors each inspected wrapper in its own
`configs/<model>/launcher.env`, while preserving the requested model defaults
in `b200.env`. New convenience wrappers reset model-bound exports unless
`LAUNCHER_USE_ENV=1` is selected (see [WRAPPER_EXPORTS.md](WRAPPER_EXPORTS.md)). Selecting
`LAUNCHER_USE_ENV=1 LAUNCHER_ENV=/dev/null` uses Medusa's model defaults; the checker then reports
a mismatch against the current unmodified source launchers where appropriate.

The complete JSON is [length_fairness_report.json](length_fairness_report.json).
PASS here concerns shared configuration and source contracts. It does not
certify seven-model experiment initialization, executed budgets or hardware.

## Length behavior

Both methods use the same generator with
`GENERATION_LENGTH_POLICY=specnaacl_compatible` by default. It consumes the
entire verified path, truncating at EOS, emits the pending rejection/bonus,
and stops all active rows when any row's actual prompt plus generated length
reaches `max_length`. Rows that just hit EOS participate in this maximum before
pruning. Shorter prompts therefore stop with the batch. Round overshoot depends
on each method's accepted path; identical cross-method output lengths are not
guaranteed.

The source also unconditionally starts round one, even if the prefill sample is
EOS or already meets the limit. Compatible mode preserves that scheduling/RNG
convention and the source's final output truncation at the first EOS. Raw online
head history retains the source convention. Medusa's existing terminal feedback
exclusions remain. `per_response` preserves the previous individual length cap
for ablation. Checkpoint policy/version checks prevent accidental mixed resumes.

Length checks stay on GPU and reuse the existing scheduling packet. No new
generation-loop D2H copy, target forward, target sampling rule or kernel was
introduced. Compatible mode reserves four extra output/storage slots to hold
the final full path. Existing GPU scatter and suffix KV compaction remain.

## Sampling and initialization

Finite BF16 probability tensors and sampled tokens/RNG are tested directly
against SpecNaacl's actual legacy sampler with temperature, nucleus and top-k,
outside autocast for prefill and inside CUDA autocast for verification.
First tokens are still sampled once per prompt before response repetition.
The checker records the mode actually selected by shell configuration and the
loaded tokenizer EOS/padding metadata from startup reports.

FastGRPO's legacy sampler has invalid-logit recovery. Medusa's finite path
rejects nonfinite logits on device; its strict mode preserves recovery. Medusa
tree verification/acceptance and round-wide RNG consumption differ from FastGRPO
and have not been replaced. PureGRPO retains independent first samples and FP32
probabilities. These implementation differences remain explicit in the report.

Loaded target LoRA hashes include parameter names, shapes, dtypes and bytes;
checkpoint equality is checked after loading. Equal seeds alone do not pass.
Medusa's loaded draft hash includes all pretrained heads and shared projector A.
The checker refuses missing draft hashes, compares runtime arguments both across
methods and against parsed launch commands, and leaves missing PureGRPO source
or runtime/checkpoint evidence NOT VERIFIED.

## Validation

- 180 tests passed, zero failed/skipped, on Python 3.12 / torch 2.8.0+cu126 /
  RTX 3090. A subsequent four-test checker regression also passed after the
  final audit changes. Compileall: 61 Python files. Shell syntax: 46 scripts/envs.
- Mixed prompt lengths, limits at/beyond prompt width and 2048, rejection,
  EOS pruning, initial EOS, pending/bonus and native KV recomputation passed.
  Output/RNG parity with Reflex disabled and B0 passed for both length policies.
- CPU and native CUDA trees/generators at 32/64/128 active responses keep all
  three heads reachable. Budgets are 12/8/4 nodes per response, totals
  384/512/512. Proposed/verified/accepted and selected/visited/frontier counters
  match independent observations. Frontier cap remains two per head/row/round.
- Production training-loop smoke with synthetic native Qwen2 weights and
  controlled rewards completed two BF16 GPU optimizer updates for each method.
  Loaded target/draft hashes, prompt order, cadence and sampler/EOS match.
- Real Qwen2.5-3B weights passed decoder/head smoke for both policies,
  disabled/B0 output/RNG parity, serial/async parity and no extra head-training
  target forwards. Experiment pretrained heads and target LoRA were not supplied
  to this real-weight smoke; it uses initialized heads and the base target.
- B200 smoke/benchmark: SKIPPED because this host has RTX 3090, not B200.
  Multi-GPU, seven full-model checkpoints/production OOM, observed EOS/mode across
  all baselines and end-to-end benchmark performance remain NOT VERIFIED.

No weights, frontier weights, OPD LR or fused kernels were changed. The paired
benchmark's default now keeps the existing 512:12 budget; no larger-tree sweep
runs by default. No new speedup claim follows from these smoke timings.

## Commands

From `MedusaGRPO`, with the appropriate CUDA environment activated:

```bash
TRITON_CACHE_DIR="$PWD/outputs/triton_cache" .venv-cuda/bin/python -m pytest tests -q
.venv-cuda/bin/python scripts/check_fairness.py --output outputs/fairness.json
TRITON_CACHE_DIR="$PWD/outputs/triton_cache" .venv-cuda/bin/python scripts/smoke_model.py \
  --model-dir ../models/Qwen2.5-3B-Instruct --output outputs/qwen3b_smoke.json
```

For the main paired experiment, set real model/data paths, the common initial
target LoRA and the **same pretrained Medusa-head checkpoint** for both runs:

```bash
export LAUNCHER_USE_ENV=1  # explicit custom checkpoint/model paths below
export MODEL=/absolute/path/to/Qwen2.5-3B-Instruct
export DATA_ROOT=/absolute/path/to/data
export TARGET_ADAPTER=/absolute/path/to/shared_initial_target_lora
export DRAFT_CHECKPOINT=/absolute/path/to/pretrained_medusa_heads
export PYTHON_BIN="$PWD/.venv-cuda/bin/python"
export GENERATION_LENGTH_POLICY=specnaacl_compatible
export GRPO_BENCHMARK=1 CPEAK_NODES=512
bash train_qwen25_3b_medusa.sh
bash train_qwen25_3b_reflex.sh
```

To inspect experiment overrides use `scripts/check_fairness.py --model qwen25_3b
--use-environment`. Supply `--runtime-report MODEL:METHOD:SUMMARY_JSON` for each
completed method/model. `--strict` fails while any required evidence is missing.
For the length ablation export `GENERATION_LENGTH_POLICY=per_response` and use
fresh run directories; this is explicitly a length-fairness FAIL for the main
SpecNaacl-compatible benchmark.
