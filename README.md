# MedusaGRPO

Repo độc lập cho `medusa` và `medusa_reflex`, dùng target GRPO của SpecNaacl.
Không import hoặc chạy source bên SpecNaacl/FlashGRPO/PureGRPO ở runtime.

Revision fairness mới nhất: [`docs/LENGTH_FAIRNESS.md`](docs/LENGTH_FAIRNESS.md).
`GENERATION_LENGTH_POLICY=specnaacl_compatible` là mặc định của cả hai methods:
kiểm tra maximum actual sequence length của batch sau khi hoàn tất verification
round, kể cả bonus/rejection token và rows vừa EOS, trước khi prune. Có thể vượt
`max_length` trong round cuối như SpecNaacl. `per_response` giữ lại cho ablation.
Checkpoint lưu policy và từ chối resume khi đổi policy.

Model configs đã khớp SpecNaacl: Qwen2.5-1.5B và Qwen3-1.7B là batch/accumulation
16/2, Qwen2.5-14B là 4/8. **Launchers SpecNaacl hiện ghi đè thành 8/4 ở cả 7
models**, nên `configs/<model>/launcher.env` của Medusa giữ cùng defaults thực.
Các wrapper `.sh` chứa defaults riêng cho từng model; biến môi trường người
dùng truyền vào luôn được ưu tiên, không cần `LAUNCHER_USE_ENV=1`. Initial LoRA tự chọn
`outputs/initial_target/<model_key>_seed42`; heads tự chọn
`outputs/pretrain/<model_key>/latest_checkpoint`. Chỉ cần activate môi trường rồi
chạy `RESUME=auto bash train_qwen25_1p5b_reflex.sh` (checkpoint phải có sẵn).
Truyền trực tiếp `CUDA_VISIBLE_DEVICES=2 DATASET=gsm8k` để đổi GPU/dataset.
Shared launcher tự chọn dataset path, hoặc giữ `DATASET_PATH` được truyền rõ ràng.
`LAUNCHER_USE_ENV=1` vẫn tương thích; `LAUNCHER_ENV=/dev/null` chọn model-config
defaults và checker vẫn báo lệch nếu
baseline chưa có cùng cấu hình. Xem [WRAPPER_EXPORTS.md](docs/WRAPPER_EXPORTS.md).

## Cấu trúc

```text
grpo_speculative.py          training loop port từ SpecNaacl
train_draft.py               ShareGPT pretrain, 5 epoch mặc định
medusa/model.py              3 independent residual heads, shared target lm_head
medusa/tree.py               sparse prefix planner và feedback selection
medusa/tree_kernels.py       GPU tree construction, một program/response
medusa/generate.py           exact verification, pending token, persistent KV
medusa/opd.py                shared A, per-head B_fast, async feedback
medusa/opd_kernels.py        batched updates cho cả 3 heads
medusa/training.py           online head objective, không forward target thêm
helper/                     loss/reward/data/sampling/OPD kernels đã vendor
configs/_shared/             defaults B200 và OPD
configs/<model>/             target settings tương ứng SpecNaacl
scripts/launch/              generic launchers, validate, resume, logging
scripts/benchmark_pair.py    paired real GRPO benchmark và tree selection
scripts/tune_opd_proposals.py sparse/fused/GEMM proposal tuning theo hardware/vocab
tests/                      CPU tests, checkpoint/resume, CUDA tests có điều kiện
docs/IMPLEMENTATION.md       audit, fairness, định nghĩa metrics và giới hạn kiểm chứng
docs/REVISION_AUDIT.md       bugs, sync audit, benchmarks và các khác biệt còn lại
scripts/check_fairness.py    audit 35 launchers và actual runtime bindings
scripts/benchmark_hotpaths.py trước/sau online heads và KV trên cùng tensors
```

Có 7 pretrain scripts, 14 method scripts và 7 `train_<model>.sh` aliases cho
Medusa+Reflex: `qwen25_1p5b`, `qwen25_3b`, `qwen25_7b`, `qwen25_14b`,
`qwen3_1p7b`, `qwen3_4b`, `llama31_8b`.

## Chạy Qwen2.5-1.5B trên B200

Các default model/data paths được giữ theo SpecNaacl:

```bash
cd /mnt/hdd/nhatminh/SpecDecode/MedusaGRPO
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match 'torch==2.8.0+cu128' -r requirements.txt
export PYTHON_BIN="$PWD/.venv/bin/python"
export MODEL=/workspace/storage-shared/models/Qwen2.5-1.5B-Instruct
export DATA_ROOT=/workspace/storage-shared/nlp/minhpn19/data
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
"$PYTHON_BIN" scripts/validate_environment.py --require-cuda
```

Đây là CUDA wheel cho môi trường B200; chọn driver tương thích trên server.
Các bản PyTorch 2.8/CUDA được liệt kê tại
[PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).
Môi trường kiểm tra RTX 3090 trong workspace là `.venv-cuda` với torch
`2.8.0+cu126`; `.venv` hiện là môi trường CPU. Không dùng wheel cu126 này
để suy ra khả năng chạy B200.

`TARGET_ADAPTER` phải là **cùng checkpoint LoRA khởi tạo mà các baseline khác
đang dùng**. Wrapper Medusa chọn đường dẫn theo model; SpecNaacl vẫn cần trỏ
`TARGET_ADAPTER` tới cùng checkpoint đó.

```bash
# Chỉ cần khi dùng checkpoint ngoài đường dẫn mặc định của wrapper.
export TARGET_ADAPTER=/absolute/path/to/common_initial_target_lora
```

Nếu chưa có initial LoRA, tạo **một lần** và dùng checkpoint này cho cả năm
phương pháp. Lệnh dưới đây không thay thế checkpoint của experiment đã chạy:

```bash
python scripts/create_initial_lora.py \
  --model /workspace/storage-shared/models/Qwen2.5-1.5B-Instruct \
  --output outputs/initial_target/qwen25_1p5b_seed42 --seed 42
# Khi chạy các baseline SpecNaacl/PureGRPO, cũng trỏ chúng tới checkpoint này.
```

Pretrain một lần, rồi giữ nguyên checkpoint cho cả hai ablation:

```bash
bash pretrain_qwen25_1p5b.sh
bash train_qwen25_1p5b_medusa.sh
bash train_qwen25_1p5b_reflex.sh
# Alias của Medusa+Reflex:
bash train_qwen25_1p5b.sh
```

Pretrain default: 5 epoch đầy đủ, max length 2048, seed 42, BF16, SDPA.
Target backbone và lm_head được freeze. Batch/accumulation theo model; các biến
`PRETRAIN_BATCH_SIZE` và `PRETRAIN_ACCUMULATION_STEPS` nằm trong từng wrapper;
environment override được áp dụng trực tiếp.
Checkpoint cuối nằm ở `outputs/pretrain/<model>/latest_checkpoint/draft.pth`.
Training xuất vào `outputs/train/<model>/<unique_run_name>`.

```bash
DRY_RUN=true bash train_qwen25_1p5b_medusa.sh
MAX_TARGET_OPTIMIZER_STEPS=10 bash train_qwen25_1p5b_medusa.sh
RESUME=auto bash pretrain_qwen25_1p5b.sh
RESUME=auto bash train_qwen25_1p5b_reflex.sh
# Hoặc resume một run cụ thể:
RUN_DIR=/absolute/path/to/run RESUME=auto bash train_qwen25_1p5b_reflex.sh
```

`MAX_TARGET_OPTIMIZER_STEPS` là budget **tổng** tính cả phần đã resume;
`MAX_ROLLOUT_PROMPTS` giới hạn tổng prompt đã rollout, kể cả reward-filtered.
Với multi-GPU, prompt budget phải chia hết cho world size. Có tqdm ở rank 0.

## Tuning và benchmark B200

Default cây là **provisional**, chưa được chọn bằng đo B200. Global node budget
512, max 12 nodes/response, top-k 4/3/2. Với 128 active responses, cây dùng 4
nodes/response và vẫn có một đường tới cả ba heads. Không nâng lên dense 41 nodes.

```bash
python scripts/tune_opd_proposals.py --models qwen25_1p5b \
  --target-config "$MODEL/config.json" --draft-checkpoint "$DRAFT_CHECKPOINT" \
  --rank 8 --dtype bf16 --topk 16 --shapes 1x1,8x1,32x1,64x1,128x1

python scripts/benchmark_pair.py --model qwen25_1p5b --steps 10 --trials 3 \
  --budgets 512:12
```

`benchmark_pair.py` dùng cùng defaults model/checkpoints như launcher, kế thừa
GPU/dataset/path và các OPD weights từ environment. Default
`OPD_FRONTIER_WEIGHT=0.5`, `OPD_VISITED_WEIGHT=1.0`, capped frontier vẫn là 2.
Không bật profiling mặc định; `--profile` chỉ dùng cho lượt đo riêng.

```bash
CUDA_VISIBLE_DEVICES=2 DATASET=gsm8k DRY_RUN=true \
  bash train_qwen3_1p7b_reflex.sh

CUDA_VISIBLE_DEVICES=2 PRETRAIN_DATASET=gsm8k \
  bash pretrain_qwen3_1p7b.sh

CUDA_VISIBLE_DEVICES=2 DATASET=gsm8k python scripts/benchmark_pair.py \
  --model qwen3_1p7b --steps 10 --trials 3 --budgets 512:12 --dry-run
# Bỏ --dry-run để chạy benchmark thật và xuất results.json, selected_tree.env.
```

Benchmark chạy production GRPO launchers với cùng checkpoint, seed, settings và
optimizer budget. Bảng kết quả gồm AAL, generation/E2E tokens/s, per-head
utilization, nodes và target forwards. `selected_tree.env` chọn một cấu hình dùng
chung, theo median generation throughput của từng phương pháp và geometric mean
của cặp. Dùng `--profile` ở một lượt riêng để đo OPD event times. Profiling không
bật trong lượt production mặc định.

```bash
source outputs/benchmarks/<benchmark_run>/selected_tree.env
bash train_qwen25_1p5b_medusa.sh
bash train_qwen25_1p5b_reflex.sh
```

Các OPD defaults: `OPD_SELECTION=visited_capped_frontier`,
`OPD_MAX_FRONTIER_PER_HEAD=2`, `OPD_RANK=8`, `OPD_TOPK=16`,
`OPD_FAST_LR=0.01`, `OPD_UPDATE_STREAM=1`. Ablations:

```bash
OPD_SELECTION=visited_only bash train_qwen25_1p5b_reflex.sh
OPD_SELECTION=all_internal_weighted bash train_qwen25_1p5b_reflex.sh
OPD_ENABLED=0 bash train_qwen25_1p5b_reflex.sh
```

## Kiểm tra

```bash
python -m pytest -q
python scripts/compile_kernels.py   # compile sm100; không thực thi GPU
python scripts/smoke_cpu.py         # synthetic CPU production-loop smoke
python scripts/validate_environment.py --require-cuda
python scripts/check_fairness.py --output docs/fairness_report.json
# Khi đã set MODEL/DATA_ROOT/TARGET_ADAPTER cho experiment:
python scripts/check_fairness.py --use-environment --output outputs/fairness.json
```

Fairness checker exit 0 nghĩa audit đã chạy. `--strict` trả exit 1 khi còn FAIL
hoặc NOT VERIFIED. Cấu hình và stopping policy của 7 models hiện PASS; hashes
checkpoint/runtime đầy đủ của các baseline và B200 chưa được cung cấp.
Lỗi sort response/advantage đã được sửa đồng nhất trong cả ba trainer; các
GRPO run cũ cần retrain từ initial LoRA chung. `GRPO_BENCHMARK=1` bắt buộc
checkpoint chung và xác minh tensor thực sau khi load. Xem báo cáo mới
[`docs/LENGTH_FAIRNESS.md`](docs/LENGTH_FAIRNESS.md) và
[`docs/LENGTH_FAIRNESS_VALIDATION.json`](docs/LENGTH_FAIRNESS_VALIDATION.json).

Kiểm tra thực tế dùng CPU Python 3.12/torch2.8 và RTX3090/torch2.8+cu126.
Suite gồm tiny native-architecture smoke của 7 configs, GRPO/head gradient
parity, CUDA KV recomputation, B0/disabled RNG, async, pretrain5epochs và resume.
Revision fairness hiện tại: **180 Medusa tests passed, 0 failed/skipped** trên
môi trường CUDA, cộng regression checker sau thay đổi cuối. Kết quả lịch sử
537 tests của ba repos nằm ở [`docs/FINAL_VALIDATION.json`](docs/FINAL_VALIDATION.json);
lần này chỉ sửa và chạy suite của Medusa.
[`docs/VALIDATION.json`](docs/VALIDATION.json) giữ kết quả revision trước.
[`docs/real_qwen25_1p5b_cuda_smoke.json`](docs/real_qwen25_1p5b_cuda_smoke.json)
giữ smoke Qwen2.5-1.5B trước đây. Lần sửa cuối đã kiểm tra weights thật
Qwen2.5-3B, xem [`docs/final_real_qwen25_3b_smoke.json`](docs/final_real_qwen25_3b_smoke.json),
với heads khởi tạo, disabled/B0 RNG và serial/async parity. Đây chưa phải full GRPO hoặc full-concurrency OOM test.

Microbenchmark RTX3090 (H1536, V151936) cho online heads: **76.29 → 19.82 ms**;
KV compaction: **0.620 → 0.509 ms**. Xem
[`docs/hotpath_cuda_benchmark.json`](docs/hotpath_cuda_benchmark.json).
Không suy rộng các số này thành end-to-end/B200 speedup. Máy không có B200;
full 5-epoch ShareGPT trên 7 weights thật, multi-GPU, full-model OOM và B200
throughput/backend tuning **chưa được kiểm chứng**.

Các lệnh kiểm tra lại trên máy hiện tại:

```bash
.venv/bin/python -m pytest -q
TRITON_CACHE_DIR="$PWD/outputs/triton_cache" .venv-cuda/bin/python -m pytest -q
TRITON_CACHE_DIR="$PWD/outputs/triton_cache" .venv-cuda/bin/python scripts/smoke_model.py \
  --model-dir ../models/Qwen2.5-1.5B-Instruct --output outputs/real_model_smoke.json
.venv-cuda/bin/python scripts/benchmark_hotpaths.py --device cuda \
  --hidden 1536 --vocab 151936 --responses 8 --length 128 --prompt 96 --trials 11 \
  --output outputs/hotpaths_cuda.json
```

Real-model smoke hỗ trợ `--heads "$DRAFT_CHECKPOINT"` và
`--target-adapter "$TARGET_ADAPTER"` khi đã có checkpoints; không truyền hai flags này chỉ kiểm
tra base target + heads khởi tạo. Có thể chạy lại với model directory của từng
config trên máy đủ VRAM; smoke ngắn không thay thế full-concurrency OOM checks.

Online heads mặc định chặn tổng LM-head projection ở 128 rows/chunk, dùng
chung cho hai methods. Có thể giảm `MEDUSA_HEAD_LOGIT_ROWS` (ít nhất 3) để giảm
vocabulary activation memory; normalization và optimizer cadence giữ nguyên.
