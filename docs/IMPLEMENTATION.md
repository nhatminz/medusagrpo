# Audit và kiểm chứng

## Sources và fairness

Nguồn chính: `SpecNaacl/grpo_speculative.py`, `scripts/launch/{train,pretrain}_model.sh`,
`configs/_shared/b200_common.env`, 7 model envs, `helper/{fastgrpo_training,rewards,get_QAs,pretrain_data,checkpointing,rollout_metrics,step_metrics}.py`,
`helper/opd_{reflex,reflex_kernels,generate,sampling,profiles}.py`, tree/KV kernels.
Medusa architecture được đối chiếu `FlashGRPO/flashgrpo_b200/models/medusa_heads.py`;
tree/verifier/cache được đối chiếu `decoding/{medusa_tree,flash_medusa_decoder,acceptance,tree_attention,kv_extraction}.py`.

`SOURCE_MANIFEST.json` ghi hashes ban đầu lúc vendor, không phải manifest của
mọi file sau revision. `source_snapshot.json` ghi
407 code/config files của hai sources trước triển khai. Revision tiếp theo dùng
`revision_source_snapshot.json` cho 450 code/config files của SpecNaacl,
FlashGRPO, puregrpo và FastGRPO-main là snapshot lịch sử. Lần sửa cuối được
user cho phép sửa training trong cả ba repositories; inventory trước/sau ở
`final_correctness_changes.json`. Runtime không import các repositories khác.
Báo cáo hiện tại ở `FINAL_CORRECTNESS.md`; `REVISION_AUDIT.md` là lịch sử.

| Thành phần target | Đối chiếu SpecNaacl |
| --- | --- |
| Model và data paths | Copy các B200 envs; SimpleLR Abel 3–5 và ShareGPT cùng defaults |
| Tokenizer, chat template, prompt truncation | Cùng AutoTokenizer và nguyên `TrainDataCollator` |
| Initialization | Cùng PEFT API và LoRA r64/alpha32/dropout0/modules; bắt buộc explicit target adapter |
| Dataset subset/order | Vendor nguyên loader/subset; DistributedSampler seed42, cùng generator riêng |
| Reward/advantage/filtering | Nguyên reward code; 0.2 format + accuracy; NumPy std; bỏ nhóm reward constant |
| GRPO/KL/clipping/mask | Nguyên `compute_target_loss`; target backward/collator kiểm tra AST parity |
| Target optimizer/scheduler | Cùng AdamW và LR; không có target scheduler, giữ cadence của source |
| Batch/generation | Actual launchers của cả 7 models: batch8/accum4, 8 responses, temp1, top-p0.95, length2048 |
| BF16/SDPA/packages | Cùng install pins; native Qwen2/Qwen3/Llama target backbone |
| Online heads | Cả hai dùng future CE offsets2/3/4, decay0.8, LR1e-4 và cùng boundaries |
| Pretrained heads | Cả hai load strict cùng `draft.pth`; không random fallback |
| Tree/performance policy | Cùng sparse planner, budget, shared proposal scanner, KV và LM projection |
| Reflex | Một learned A; B `[3,V,8]` reset mỗi rollout; chỉ Reflex cập nhật B và A |

Source có hai chi tiết cần giữ rõ khi so sánh:

- `step = eligible_prompts // (batch_size * accumulation_steps)` là nhãn kế thừa.
  Source có thể cập nhật target optimizer nhiều lần với cùng nhãn. Repo giữ
  cadence, đồng thời ghi `target_optimizer_steps` thật và dùng nó cho
  `MAX_TARGET_OPTIMIZER_STEPS`/mỗi dòng `timing.csv`. `source_grpo_step` vẫn giữ
  nghĩa cũ. Không dùng nhãn `step` cũ để suy ra số updates khi so năm phương pháp.
- Source sắp tokenized rows theo length nhưng không hoán vị advantages tương ứng.
  Lỗi đã được sửa đồng nhất ở SpecNaacl, puregrpo và Medusa: tokens/masks,
  identity, rewards và standardized advantages dùng cùng permutation. Loss và
  gradient được so với tham chiếu không sort. Các GRPO run cũ cần retrain.
- SpecNaacl và hai Medusa methods sample một first token/prompt trước khi repeat
  responses. PureGRPO sample độc lập. Medusa dùng cap riêng từng response theo
  actual valid prompt length; source FastGRPO/PureGRPO dừng cả batch theo real
  lengths và source FastGRPO có thể
  vượt max_length theo cả verification round. Không gọi cả năm methods fully fair.

## Vì sao heads 2/3 bị starvation

Dense tree 4/3/2 cần `1+4+4*3+4*3*2 = 41` nodes. Non-adaptive concurrency planner
đi từ shallow tới deep: sau head1 đã dùng5, head2 cần thêm12, head3 cần thêm24.
Budget 10–16 không đủ; fixed planner cũng giảm depth cuối trước. Nhánh `sparse`
trong source chỉ warning rồi đổi thành `dense`. Ngoài ra, clamp budget lên
`min_tree_nodes_per_seq` có thể làm tổng nodes vượt `cpeak_nodes` khi concurrency cao.

Planner mới dùng `min(max_nodes, global_budget // active_responses)`, không clamp
vượt global cap. Nó dành trước root + top1/head thành một trunk (4 nodes cho cả
3 heads). Sau đó chỉ xét children của các prefixes đã tồn tại, chọn cumulative
log-proposal score cao nhất, loại child trùng. Cây luôn prefix-closed, không tạo
dense Cartesian candidates trước khi prune. Budget nhỏ hơn4 được ghi trong
`head_limit_reasons`; budget nhỏ hơn số roots trả lỗi rõ để giảm concurrency.

GPU builder thực hiện toàn cây trong một program/response; Torch implementation
là CPU oracle. Một ancestor-only attention mask được kiểm tra bằng independent
root-to-node recompute; KV chỉ gather phần suffix theo accepted path.

## Exactness và pending token

Prefill lấy một target sample/prompt rồi repeat responses. Mỗi verification tree gồm pending root và
candidates của ba heads từ anchor hidden **trước** root. Target forward tree
dùng causal ancestor mask và logical positions. Traversal chỉ đi vào child nếu
token target sample khớp; luôn giữ sample tại decision cuối, kể cả rejection
hoặc leaf bonus. Sample này được emit ngay và làm root chưa có KV của vòng sau.
KV committed chỉ chứa root và matched nodes; bonus không bị giả lập KV hoặc
forward riêng. EOS dừng traversal; giới hạn sequence truncate cả path và feedback.

Target sampling reuse SpecNaacl's temperature/nucleus/top-k sampler. Prefill
sampling nằm ngoài CUDA autocast (BF16 probabilities với BF16 target), còn
verification sampling nằm trong autocast, đúng source. Default `OPD_SAMPLER_MODE=finite` dùng contract logits hữu hạn,
giống launcher OPD mới của source; `strict` phục hồi invalid logits theo sampler
vendor. Không có typical acceptance, residual reweighting, hoặc forced acceptance.

## Reflex feedback và gradient

Mỗi active response/round tạo một q cho mỗi head; tất cả parent prefixes của
horizon đó reuse q. Head1 teacher ở depth0; head3 teacher ở depth2. Visited là
decision rows traversal thực sự ghé, có cả rejection row. Frontier là off-path
parent state có parent đã visited, chọn tối đa2/head/response bằng cumulative
proposal score đã có. Không duplicate row. `visited_only` và
`all_internal_weighted` dùng chung teacher/update path.

Selection diễn ra trên GPU trước teacher extraction. Reuse sorted probabilities
do target nucleus sampling đã tạo; compact teacher scan chỉ đọc selected rows.
Sparse union, per-head weighted normalization, active bitmap và projector kernels
được port từ SpecNaacl. Update trên union coordinates dùng q-p như source; tail
được giữ trong KL diagnostic. CPU gradient oracle đối chiếu chính coordinate
objective này. Ba B updates dùng một batched GPU grid; không loop Python qua nodes.
Shared A gradients được aggregate từ frozen B trước update và apply tại head
optimizer boundary. Heads CE detach target hidden/lm_head; target LoRA chỉ GRPO.

EOS-containing prefixes và horizons vượt remaining budget bị loại trước feedback.
Async stream đợi target outputs trên device, ghi event sau batched updates; main
stream đợi event trước khi dùng B lần sau. Một packet nhỏ/round phục vụ scheduling
và proposal backend snapshots; không thêm packet riêng cho feedback. Target
forward count = prefill + verification rounds, kể cả Reflex.

## Metrics

`logs/rollout_timing.csv` ghi mỗi DataLoader iter, gồm skipped/reward-filtered.
`logs/timing.csv` ghi mỗi actual target optimizer update; `metrics.jsonl` và
`console.log` qua launcher; summary và rank-local checkpoint/RNG states đầy đủ.

`iter_aal = iter_total_acc_length / iter_total_decoded_token_num`; cumulative
AAL là tổng numerator/tổng denominator, cùng source. Numerator là decision-path
tokens emit trong verification rounds; token prefill không được đưa vào numerator.
Acceptance = accepted Medusa tokens / proposed tree nodes, báo cả per-head.
Depth mới dùng root depth0 và headk candidate depthk; chưa mix với source's
root-depth1 notation. `tree_depth_reached=3` nghĩa cả ba proposal horizons có node.

Host counters theo dõi per-head active rounds, proposed nodes, accepted tokens,
head loss, verification nodes, forwards và reasons budget. OPD counters ở GPU chỉ
đọc cuối rollout, trong một contiguous packet cho cả ba heads. Thêm alias verified
nodes, head supervised tokens, gradient batches và optimizer updates; không chạy
kernel mới chỉ để tạo các alias này. `medusa_scheduling_syncs` chỉ đếm packet
scheduling prefill + rounds, không giả vờ đếm toàn bộ runtime/implicit sync.
Event time `opd_feedback_time`/`opd_proposal_time` chỉ có khi
`OPD_PROFILE=1`; production để blank, không tạo giả timing. Không print/token
hoặc print/verification round. Cumulative generation/E2E time cùng host basis
và target/head training phase timings giữ implementation SpecNaacl.

Rollout CSVs là rank-local, file `.rankN.csv` trên worker phụ; target shared
counters trong summary dùng source distributed reductions. `medusa_totals`,
verification nodes và target forwards được SUM toàn job tại final logging
boundary; acceptance rates tính lại từ tổng numerator/denominator. Snapshot
`medusa_metrics` vẫn là rollout cuối của rank0. Không thêm collective trong
generation để thống kê các counters này.

## Validation và giới hạn

CPU checks: sparse budget/prefix/ancestor mask; đủ3 heads; high concurrency;
tree attention/KV parity trên Qwen2/Qwen3/Llama; empirical two-token target joint
distribution; EOS/rejection/pending; disabled/B0 RNG identity; no extra target
forwards; per-head q-p gradients/frontier cap/reset; head3 pretrain/online loss,
actual participation bằng Markov target fixture; numerical resume cả target,
heads và optimizers; full5 epochs incl remainder; các model launchers; prompt
budget khi rewards bị filter; không random fallback.

`compile_kernels.py` compile sparse construction, batched update/end và padded
tree attention mask cho sm100 (compile-only). Suite CUDA đã thực thi trên RTX3090
với torch2.8+cu126 và Triton3.4; gồm BF16 RNG/gradient/async checks và GPU KV
recomputation ở Qwen2/Qwen3/Llama. Native tiny smoke chạy đủ 7 model configs;
weights thật Qwen2.5-1.5B cũng đã chạy decoder/head smoke. Các tiny fixtures không
chứng minh full-model OOM behavior. Full pretrained real-model GRPO, multi-GPU,
B200 throughput, memory và backend winner chưa được xác nhận. Kết quả và commands
được ghi tại `VALIDATION.json` và `REVISION_AUDIT.md`.

Synthetic CPU smoke comparison dùng production GRPO loop, cùng target adapter,
head checkpoint và seed; report tại `outputs/benchmarks/revision_cpu_smoke/comparison.json`.
Các số này chỉ chứng minh pipeline có thể chạy. Không dùng để báo research speedup.

Default sparse policy chưa được B200 tuned. `benchmark_pair.py` chạy đủ cặp với
ít nhất ba sparse budgets, chọn shared policy theo generation throughput, đồng
thời ghi actual E2E throughput/optimizer/prompts. Không tạo tuned profile giả,
không chọn default theo AAL, không thay đổi config nguồn.
