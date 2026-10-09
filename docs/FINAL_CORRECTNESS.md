# Final correctness audit (2026-10-09)

This revision supersedes the historical `REVISION_AUDIT.md` and its 85-test
results. The latest request explicitly authorizes the common training fix in
MedusaGRPO, SpecNaacl and puregrpo. No FastGRPO/Reflex generator, PureGRPO decoder,
reward formula, target loss, clipping, KL, or persistent draft objective changed.
The before/after SHA256 inventory is `final_correctness_changes.json`.

Final validation: **537 tests passed (104 MedusaGRPO, 360 SpecNaacl, 73 puregrpo),
0 failed/skipped**, compileall and 87 shell scripts passed. Actual BF16 GPU
training ran two updates per Medusa method with matching startup tensor hashes
and prompt order. Real Qwen2.5-3B decoder/head smoke passed disabled/B0 RNG and
serial/async parity. See `FINAL_VALIDATION.json`; GPU is RTX 3090, not B200.

## Confirmed bugs and fixes

1. All three trainers sorted token/mask rows but indexed advantages in the old
   order. They now apply one stable permutation to complete response records:
   `(prompt_id, response_id, messages, input_ids, attention_mask, loss_mask,
   reward, standardized advantage)`. Production prompt identity includes rank,
   epoch, loader batch and within-batch prompt. GRPO loss and gradients are tested
   against an unsorted reference using the actual objectives from all trainers.
2. Medusa subtracted the padded batch width from max_length. Each response now
   has budget `max_length - valid_prompt_length`; counts include the first
   sampled token and pending/rejection/bonus tokens. Physical storage includes
   left padding separately; the inherited prompt eligibility check still rejects
   batches whose padded prompt width is already at max_length. EOS/truncated feedback, scratch scatter and KV
   logical positions retain the existing paths. The existing prefill packet
   supplies allocation sizes; no extra generation-loop readback was added.
3. Source/PureGRPO previously lacked an actual optimizer-step budget; a batch
   boundary check alone can overshoot when GRPO iteration count exceeds one. All three trainers support MAX_TARGET_OPTIMIZER_STEPS and
   MAX_ROLLOUT_PROMPTS, stop at completed optimizer boundaries, save counters and
   prompt hash chains in checkpoints, and report actual update cadence. Defaults
   of zero preserve the original unlimited training cadence. Medusa retains its
   existing benchmark guard requiring grpo_iteration_num=1. Prompt budgets are
   global across ranks and must be divisible by world size; partial final batches
   are sliced consistently across all fields.
4. Seed/path equality did not prove shared initialization. The three loaders now
   compare every actual loaded LoRA tensor with saved weights, verify recipe,
   active adapter and trainability, and hash tensor names/shapes/dtypes/bytes.
   Official mode rejects missing shared initialization. Checkpoints reject old
   alignment versions and changed initial target tensor hashes before restoration.

**Retrain every old GRPO benchmark from the common initial adapter.** Old trained
GRPO checkpoints contain updates from wrong response advantages and cannot be
made equivalent merely by resuming. Pretraining checkpoints remain usable.
Medusa runtime semantics is now `actual_prompt_response_budget_v3`.

## Fairness is not fully established

The seven-model/five-method checker reports PASS/FAIL/NOT VERIFIED per condition.
A successful checker process means an audit ran; `--strict` fails unless all
conditions pass. Launcher flags alone do not certify runtime fairness.

**Known blocking difference:** unchanged SpecNaacl stops an entire batch when
its longest real sequence reaches max_length, possibly after a whole verification
round overshoots. PureGRPO also uses batch stopping. Independent response budgets
and exact agreement with this baseline behavior cannot both hold without
changing baseline decoding. The request also prohibits changing FastGRPO
speculative decoding. Medusa now implements the explicit independent-budget
requirement, while the checker reports generation-length FAIL. The user has been
asked which requirement should take precedence; no baseline decoder was changed.

PureGRPO retains independent first-token sampling and FP32 probabilities; four
speculative methods retain shared per-prompt first tokens and the source BF16
prefill convention. This numerical precision difference is reported explicitly.
Zero-supervision rows also differ historically (source NaN vs PureGRPO zero);
this revision does not silently change either objective.

No seven-model experimental initialization/runs or B200 are available here.
Unexecuted model/runtime conditions remain NOT VERIFIED. Actual CUDA tests use
RTX 3090, not B200. Tiny controlled-reward integration exercises production
optimizers/checkpoints; it is not real-data training evidence.

## Runtime evidence and commands

Use one common initial target adapter for all five methods for each target
model. Each draft pair uses its own common pretrained checkpoint. Medusa heads
and the FastGRPO draft intentionally have different architectures/objectives.

```bash
# Run from workspace root; choose a destination for the shared initialization.
MedusaGRPO/.venv-cuda/bin/python puregrpo/scripts/prepare_shared_target_adapter.py \
  --model_dir models/Qwen2.5-3B-Instruct \
  --output MedusaGRPO/outputs/shared_initial/qwen25_3b --dtype bf16

# Use the same TARGET_ADAPTER in ALL five launchers.
export GRPO_BENCHMARK=1
export TARGET_ADAPTER="$PWD/MedusaGRPO/outputs/shared_initial/qwen25_3b"
export MAX_TARGET_OPTIMIZER_STEPS=10
export MAX_ROLLOUT_PROMPTS=0
# Set MODEL, DATASET_PATH, DRAFT_CHECKPOINT and PYTHON_BIN for the deployment.
# Pretrained DRAFT_CHECKPOINT must be Medusa heads for the following pair.
bash MedusaGRPO/train_qwen25_3b_medusa.sh
bash MedusaGRPO/train_qwen25_3b_reflex.sh

# Flag/source audit of all 35 launchers; strict currently fails on known differences.
MedusaGRPO/.venv-cuda/bin/python MedusaGRPO/scripts/check_fairness.py \
  --output MedusaGRPO/outputs/fairness.json
# Add completed production summaries to check actual loaded tensors/updates:
# --runtime-report qwen25_3b:medusa:/absolute/run/summary.json
# Repeat for all five methods (and all models for a complete audit).
```

Official mode hashes actual target backbone tensors, loaded proposal tensors,
tokenizer artifacts and dataset bytes once at startup. Initialization evidence
is written as `initialization.rankN.json` and included in the final summary;
this startup cost is outside generation and is not a model-throughput result.
Hash chains/step cadence describe executed rank-local prompt batches. Full
multi-rank and full-model resume remain NOT VERIFIED without matching runs.

The previous isolated head/KV speedups are historical results, not before/after
measurements of this correctness revision. No kernels, tree budgets, OPD ranks,
frontier weights, learning rates, batching/stream optimizations were retuned.
