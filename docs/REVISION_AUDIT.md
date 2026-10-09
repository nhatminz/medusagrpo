# Correctness, performance và fairness revision

Chỉ sửa MedusaGRPO. Snapshot `revision_source_snapshot.json` bảo vệ 450 source/config
files của SpecNaacl, FlashGRPO, puregrpo và FastGRPO-main. Runtime vẫn độc lập;
fairness/tests chỉ đọc baseline sources. Cả `medusa` và `medusa_reflex` dùng chung
trainer, heads, planner, verifier, sampling, KV và online-head optimization.

## Bugs và khác biệt được xác nhận trước khi sửa

| Vấn đề | Bằng chứng và thay đổi |
| --- | --- |
| First-token grouping sai với SpecNaacl | Cả hai Spec generators sample trước repeat. `medusa/generate.py` nay sample một lần/prompt rồi repeat; output/RNG grouping test trên CPU và BF16 CUDA. |
| Prefill autocast làm đổi xác suất | Spec prefill sampler ở ngoài autocast. CUDA probe xác nhận BF16 softmax thành FP32 khi đặt trong autocast. Tách scope prefill, giữ verification sampling trong autocast. |
| Config literals khác actual wrappers | Spec wrappers ghi đè batch/accum thành8/4 cho cả7 models. Sửa configs Qwen2.5-1.5B/14B và Qwen3-1.7B; audit 35 launchers + runtime generation bindings. |
| Feedback vào terminal/truncated prefixes | `restrict_feedback()` loại node chứa EOS hoặc có ancestor EOS và depth vượt remaining budget trước selection/teacher extraction. |
| CUDA dynamic compaction trong generator | Bỏ `live.nonzero`, padding nonzero và boolean-index scatter; survivor mapping dùng scheduling packet sẵn có, history mask truyền trực tiếp vào tree attention kernel, integer scatter dùng 4 private scratch columns/response. |
| KV scratch giả định key/value cùng dtype | Actual CUDA launcher validator tái hiện FP32 keys/FP16 values khi FP32 target chạy dưới autocast. Scratch/layout nay theo dtype/shape của từng pool; regression unit và CUDA runtime probe đều kiểm tra trường hợp này. |
| Teacher storage còn sống sang vòng tiếp | Xóa local alias sau enqueue feedback; weakref regression xác nhận sampler sau không giữ teacher probability tensor cũ. Async tensor lifetimes vẫn được bảo vệ bằng `record_stream`. |
| Metric `opd_host_syncs` gây hiểu nhầm | Generator không trả metric này nữa; legacy CSV field để trống. Thay bằng explicit scheduling packet counters, không gọi là tổng số sync. |
| Resume từ method/runtime khác | Checkpoint lưu runtime semantics version và từ chối version cũ, khác method hoặc khác OPD_ENABLED trước khi restore weights/RNG. |

Điều kiện đếm `slots < path.lengths-1` **đúng**, không đổi để tăng acceptance.
`path.tokens[0]` là sample tại root, ứng với head1 nếu traversal match child;
slot1 ứng với head2, slot2 ứng với head3. Decision cuối luôn là token pending
cho vòng sau, gồm rejection/leaf bonus/EOS/length-boundary; không tính accepted
head. Handcrafted tests kiểm tra đầy đủ các trường hợp này.

`total_acc_length` là số decision tokens emit trong verification rounds, không
tính first token từ prefill. `total_decoded_token_num` là tổng response-rounds,
khác số batched target calls. AAL = hai tổng này chia nhau, cùng định nghĩa source.
`headN_verified_tokens` đếm candidate nodes đã đi qua target verification,
không phải accepted tokens. Target forwards luôn = 1 prefill + batched rounds.

## Audit synchronization và lifetimes

| Vị trí | Hành vi production |
| --- | --- |
| Prefill generator | Một packet `[prompt_lengths, first_tokens]` phục vụ EOS scheduling và metadata training. |
| Mỗi verification round | Một packet `[path_lengths, finished, backend_snapshots]`; CPU dùng max width và survivor indices, GPU giữ token/KV/path data. |
| Sparse builder/verifier | Triton fixed grids và bounded-depth Torch operations; không host scalar read để chọn nodes/traverse. |
| Proposal/feedback | Ba head calls dùng chung scheduling/async stream; không D2H trong feedback. B updates đã dùng batched head dimension. |
| Async | Update stream chờ target stream; event sau update; proposal/pruning chờ event trên device. `record_stream` bảo vệ teacher, metadata, parents/depths/scores và path storage. Không xóa các waits này. |
| Cuối rollout | Một contiguous OPD counter readback, một bookkeeping packet và một bulk generated-output copy; không copy blocking từng response. Nằm ngoài scheduling metric. |
| Init / CPU oracle | Projector initialization có một transfer nhỏ từ LM head; CPU-only oracle vẫn dùng scalar conversions/nonzero. CUDA không chạy oracle. |
| Optional modes | Strict sampler giữ recovery checks/dynamic valid-row indexing như source. Diagnostics có extra readbacks; profiling tạo timed events và synchronize sau rollout. |

`medusa_scheduling_syncs = 1 + verification_batches`, tách thành prefill1 và
round counter. Đây chỉ là explicit scheduling readbacks. Không đo tổng implicit
sync của PyTorch multinomial, allocator, model internals hoặc output copies;
không tuyên bố generator hoàn toàn không sync. Default finite sampler giữ finite
logits contract và on-device assert; invalid logits dùng opt-in strict fallback.

Training output slicing dựa trên contract **left padding** của TrainDataCollator.
History mask vẫn hỗ trợ các gap do KV compaction. Không coi arbitrary prompt
mask holes là supported training-input format.

## Online heads và KV

Giữ nguyên objective: shifts2/3/4, teacher tokens, response-specific supervision
denominator, chia số responses, weights `.8**h/sum(.8**j)` và gradient accumulation.
Pack riêng các supervised anchors theo response lengths, chunk mỗi head rồi ghép
LM-head GEMM/backward của ba heads. Không clone toàn bộ rollout states trước
training và không forward target thêm. Default tổng vocabulary rows <=128,
override `MEDUSA_HEAD_LOGIT_ROWS>=3`; đủ bounded activation memory khi V lớn.
Heads không có labels vẫn tham gia cùng DDP collectives với zero gradient.

Frozen reference objective ở `tests/reference_online.py`. CPU tests so losses và
mọi head FC/norm gradients với atol1e-6/2e-7; BF16 GPU tests dùng loss atol0.003
và gradients atol1e-4/rtol0.03 do GEMM/reduction grouping khác. Test nhiều response
lengths/prompt lengths, row budgets9/33/128, accumulation1/3, inference tensors.

KV compaction giữ source-compatible persistent scratch trong `cache._scratch`,
chỉ gather suffix tối đa4 accepted KV rows. Dùng `gather(out=...)` rồi copy trên
cùng stream, đọc xong mới ghi đè source. Reuse một buffer/layout cho tất cả
layer/key/value tương thích; không thêm custom KV kernel chưa benchmark.
Các pool khác dtype (FP32 key, FP16 value) có scratch riêng; không cast KV chỉ
để tái sử dụng buffer.
CPU/CUDA recomputation kiểm tra mọi tree prefix và continuation sau unequal paths,
padding, rejection, bonus ở native Qwen2/Qwen3/Llama.

RTX3090, torch2.8+cu126, 11 measured trials sau 2 warmups, cùng synthetic tensors:

| Isolated component | Before median | After median | Ratio |
| --- | ---: | ---: | ---: |
| Online heads: H1536/V151936, 8 responses, length128/prompt96 | 76.294 ms | 19.819 ms | 3.850× |
| KV: 8 layers, 8 KV heads, head_dim64, suffix12→4 | 0.620 ms | 0.509 ms | 1.220× |

Head peak allocation 828,233,216 → 828,617,728 bytes (thêm khoảng0.37MiB);
KV peak 553,056,768 → 553,024,000 bytes. Peak gồm resident benchmark model/tensors.
Report `hotpath_cuda_benchmark.json`; CPU riêng ở `hotpath_cpu_benchmark.json`.
Không chạy CUDA tests đồng thời với lượt benchmark được báo cáo. Đây không phải
full-model/end-to-end/B200 speedup hay phép đo production optimizer memory.

Reflex giữ shared persistent A, contiguous B `[3,V,8]` reset mỗi rollout, Top16
target ∪ Top16 draft + tail, source sparse forward-KL coordinate gradient,
per-head normalization. A nhận gradient tại head optimizer boundary. Proposal q
được reuse giữa mọi prefixes cùng response/head; selected teacher distributions
lấy đúng verified parent row. Default visited + tối đa2 frontier/head/response,
weights1/1 và async stream1. Teacher fallback scan workspace chỉ cấp phát khi
không có sorted sampler metadata. Ba proposal/feedback head calls chưa được
fuse thêm: không có B200 để chứng minh phương án mới thắng. Không đổi default
weight/tree budget để làm Reflex đẹp hơn.

## So sánh đủ 7 models / 5 methods

`scripts/check_fairness.py` đọc actual DRY_RUN commands và AST runtime bindings,
reward/data/collator/loss, LoRA recipe và target AdamW; không import trainer
baseline. Kết quả ở `fairness_report.json`: cả35 launchers inspected, không có
mismatch trong các target flags và generation bindings được kiểm tra.

| Setting | Medusa | Medusa+Reflex | Spec FastGRPO | Spec FastGRPO+Reflex | PureGRPO |
| --- | --- | --- | --- | --- | --- |
| Batch / accumulation, cả7 models | 8/4 | 8/4 | 8/4 | 8/4 | 8/4 |
| Responses / seed / temp / top-p | 8/42/1/.95 | same | same | same | same |
| Target dtype / attention / LR | BF16/SDPA/1e-6 | same | same | same | same |
| Beta / epsilon / LoRA | .04/.1/r64-alpha32 | same | same | same | same |
| First token | shared/prompt | shared/prompt | shared/prompt | shared/prompt | independent/response |
| Prefill probabilities với BF16 | BF16 | BF16 | BF16 | BF16 | explicit FP32 |
| Initial LoRA default | not verified | not verified | not verified | not verified | not verified |

Model/data/SimpleLR split paths, prompt handling, reward functions và filtering
đã đối chiếu. Actual initialization phải được chọn bằng `TARGET_ADAPTER` chung;
checker có thể hash checkpoint files khi `--use-environment`, không chứng minh
full tensor loading trên mọi model/server. Runtime package declarations khớp;
deployment environments trên B200 vẫn cần xác nhận thực tế.

Các khác biệt còn lại:

- PureGRPO independent first-token sampling và explicit FP32 sampling. Shared
  first token vẫn có đúng marginal target distribution nhưng responses trong
  cùng prompt bị correlated; không tuyên bố five-method trajectory identity.
- Medusa strict cap dựa trên padded prompt width. FastGRPO source dựa real
  lengths và có thể vượt cap theo cả verification round; Pure dùng real lengths.
- Reproducer xác nhận **sequence sort không permute advantages** ở cả Spec,
  Pure và Medusa. Giữ behavior chung theo yêu cầu; đây vẫn là lỗi baseline.
- Source legacy `max_grpo_steps` dựa eligible prompts, có thể lặp step label;
  Medusa bổ sung actual optimizer-step/prompt budgets. So sánh bằng actual logs,
  không coi cùng nhãn step là cùng số optimizer updates.
- Không có common experiment LoRA/heads checkpoint do người dùng cung cấp để
  xác minh mọi real run. Tiny production-loop tests dùng cùng fixture checkpoint.

Checker exit0 chỉ nghĩa audit hoàn tất. `--strict` exit1 do các giới hạn trên.
Repo thiếu source được báo `not verified`, có regression test cho trường hợp này.

## Validation và phần còn thiếu

Kết quả cuối cùng và commands ở `VALIDATION.json`. CPU và CUDA suite gồm exact
disabled/B0 output/RNG, B0 tree, GRPO loss/gradient/AdamW parity với source,
online loss/gradients, Monte Carlo target joint distribution, handcrafted
EOS/rejection/bonus/counters, KV recomputation, feedback/per-head B+sharedA
gradients, serial/async parity, 7 tiny native-architecture configs và concurrency
1/8/32/64/128, checkpoint refusal/full-vs-resume, synthetic ShareGPT5epochs.
Compile-only sm100 gồm4 kernels. Actual CUDA execution trên sm86 của RTX3090.

Suite CUDA cuối: **85 passed, 0 skipped**, 53.32s. CPU-only: **69 passed,
16 CUDA tests skipped**, 46.06s; cả16 đã thực thi thành công trong suite CUDA.
Compileall, bash syntax của30 launchers/scripts và environment validator đều qua.

Real target Qwen2.5-1.5B decoder/head smoke dùng 2 prompts × 2 responses × 12
new tokens, BF16/SDPA, zero-residual initialized heads. Disabled và B0 khớp
output/RNG, serial và async khớp output/B, online heads không thêm target forward;
peak khoảng3.32GiB. Report `real_qwen25_1p5b_cuda_smoke.json`. Head3 có186
verified nodes nhưng0 accepted,2 selected frontier states và1 B update ở active
run: log phản ánh coverage thấp, không nâng budget để che starvation. Heads này
chưa pretrain, nên không dùng AAL/elapsed của smoke để đánh giá algorithm speedup.

Reproduce real smoke bằng `scripts/smoke_model.py --model-dir <local-model-dir>
--output <report.json>`; optional `--heads` và `--target-adapter` dùng checkpoint
chung đã pretrain. Command cụ thể trong `VALIDATION.json` và README. Lượt smoke
không có target LoRA/pretrained heads được ghi rõ trong report.

**Chưa xác minh**: B200 execution/proposal-fusion benchmark, full production
concurrency và OOM của cả7 real models (đặc biệt7B/14B), actual 5-epoch ShareGPT
pretrain trên7 models, multi-GPU/rank-asymmetric runtime, real reward/GRPO paired
training với common experiment checkpoints. Không gọi production-ready/fully fair.

Pretrain/train mẫu, cùng checkpoint, giữ toàn bộ7 model scripts: xem README.
Trên B200 dùng `scripts/benchmark_pair.py` với3 sparse budgets và3 trials; đo
profiling overhead ở lượt `--profile` riêng. Chọn một shared tree cho cả cặp;
không chọn theo AAL hoặc thay generation budget giữa hai methods.
