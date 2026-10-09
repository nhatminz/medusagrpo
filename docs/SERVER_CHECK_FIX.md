# Server check failures

The supplied server log contained 9 failures and 171 passes. The server reports
torch 2.13.0+cu130, transformers 5.12.1, NVIDIA B200, matmul precision `high`,
and TF32 enabled. Local validation uses torch 2.8.0+cu126, transformers 4.51.3,
and RTX 3090. The following changes affect Medusa tests, reporting and a wrapper;
the generator, target loss/optimizer, tree/KV kernels and ReflexOPD are unchanged.

## Changes

- The three tight FP32 KV recomputation tests previously relied on implicit
  precision/attention defaults. They now select IEEE FP32 (the newer per-backend
  API when available, otherwise `highest`), explicit SDPA and the math backend.
  Settings are restored at teardown. Tolerances stay 2e-5/2e-4. An exact GPU vs
  reference tree attention-mask check is added. The other production BF16 CUDA
  and async tests continue to use the caller's normal dispatch/settings.
- Optional cross-repository tests check required baseline files before loading
  them. Missing PureGRPO comparison/audit files become SKIPPED with an explicit
  `NOT VERIFIED` reason. The five-method loaded-LoRA test is parametrized so
  absent PureGRPO cannot prevent the four available methods from checking
  tensor equality and detecting mutations. Missing Medusa source still fails.
- The seven-model Medusa/SpecNaacl configuration test is separate from the
  complete PureGRPO contract test. The latter skips only when its source
  contract is incomplete; a complete source with mismatched settings still
  fails. Hash-only unit proofs preserve any real configuration failures rather
  than assuming the overall audit must be NOT VERIFIED on every machine.
- The checker now records configuration mismatches/status per method and
  missing initialization-audit source files. Known discrepancies keep the
  overall result FAIL. Missing runtime/checkpoint evidence stays NOT VERIFIED.
  `SPEC_ROOT`/`PUREGRPO_ROOT` can select external source directories for auditing.
- Default launcher tests use an isolated environment so exported experiment
  variables do not alter the intended defaults. The canonical Qwen3-1.7B
  pretrain wrapper resolves DATA_ROOT/dataset paths in the shared launcher.
  The server log's line-5 DATA_ROOT expansion is absent from this canonical
  wrapper; update that wrapper on the server.

TF32 and hardware/backend dispatch are plausible contributors to the reported
small FP32 differences. They were not reproduced on RTX 3090, even with TF32
allowed. A B200 rerun is required to confirm their resolution. Selecting a
controlled reference computation does not certify the B200 production decoder;
the normal BF16/autocast/async tests must also pass there.

Precision controls follow the official [PyTorch CUDA documentation](https://docs.pytorch.org/docs/2.9/notes/cuda.html).

## Validation in this workspace

- Full suite: 188 passed, 0 failed/skipped, RTX 3090.
- Missing-PureGRPO scenario: 20 passed, 4 skipped with NOT VERIFIED reasons.
- Partial-PureGRPO and preserved mismatch checks: 4 passed.
- Controlled KV tests: 3 passed; caller precision `high` and all SDPA enable
  flags were restored after the test session.
- B200 / torch 2.13 / transformers 5.12 rerun: NOT VERIFIED until the updated
  files run on the server. Historical 180-test reports remain unchanged.

## Run again on the server

Update `tests/`, `scripts/check_fairness.py` and `pretrain_qwen3_1p7b.sh` from
this Medusa revision. Do not modify the sibling baseline repositories.

```bash
cd /workspace/storage-shared/nlp/minhpn19/MedusaGRPO
export PYTHON_BIN="$(command -v python)"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs}"
mkdir -p "$OUTPUT_ROOT"
export TRITON_CACHE_DIR="$OUTPUT_ROOT/triton_cache"

"$PYTHON_BIN" -m pytest tests/test_cuda.py::test_gpu_tree_kv_matches_recomputation_with_padding_rejection_and_bonus -q
"$PYTHON_BIN" -m pytest tests -q -rs
"$PYTHON_BIN" scripts/check_fairness.py --output "$OUTPUT_ROOT/fairness_before_training.json" \
  > "$OUTPUT_ROOT/fairness_before_training_console.json"
```

Inspect actual configuration differences instead of converting a FAIL to PASS:

```bash
"$PYTHON_BIN" - "$OUTPUT_ROOT/fairness_before_training.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
print('Medusa/SpecNaacl configs:', report['configuration_status'])
print('Full fairness:', report['status'])
for row in report['models']:
    for difference in row['configuration_mismatches']:
        print(row['model'], difference)
PY
```
