# DSV41 1M 上下文 prefill 实施方案

> 依据：prefill-1m 调研报告 + kv-page-design 报告（三处前提修正）。
> 仓库只读；本文档只做规划，未改任何代码。代码基线 2026-09-11。

## 0 结论

现状 DSV41 **没有 chunked prefill**：`prefill_chain` = `for t in ids { step(t,i) }`（`dsv41-run.rs:890`），
每步跑完整 40 层 **decode 形状**链（m=1 GEMV）。1M × ~10–16ms ≈ **2.8–4.4h**；叠加 indexer
逐 token 扫全池 → O(n²)。

目标：chunk=64 的 `prefill_chunk`，1M → **10–25 min（~20×）**；O(n²) 只剩 indexer 选点，Phase 3 消除。

前提：① ring/index_k 页化（1M 单序列 ≈8.4GB，×8 rank 全复制 ~67GB）；② 抬三个上限——
`DSV41_MAX_POS`（默认 64k，`chain_dev.rs:496`）、`kIdxMaxPos=65538`、`kIdxMaxRows=8`（后两者必须锁步）。

## 0.1 2026-09-11 复核（读码，未改代码）

- ✅ **§0 的现状描述确认**：`prefill_chain`（`dsv41-run.rs:890`）= `for (i,&t) in ids { chain.step(t,i) }`，
  每 token 走完整 40 层 decode 形状链。**`chain_dev.rs` 的 `step_body` 没有 `prefill_tokens>0` 分支**——
  prefill 与 decode 共用 `step_body`，唯一差别是 `decode_steps` 门控 CUDA graph（prefill 路径不 capture，
  `chain_dev.rs:1524-1560`）与 compressor 在 pos=0 的 mode 1。**不要去找一个不存在的 prefill 分支。**
- ✅ **§0 前提②的上限行号已变**：`DSV41_MAX_POS` 默认 64k 现在在 `chain_dev.rs:746-753`（读 `max_seq_len.min(env)`）。
- ⚠️ **§0 前提①的 8.4GB 是"唯一 store 的下界"，不是代码当前分配量**。`DevChain::new`
  （`chain_dev.rs:754-781`）**对全部 43 层逐层分配** `ring=(window+max_pos/ratio+2)*head_dim(512)` f32 +
  `index_k=(max_pos/ratio+2)*index_head_dim(128)` f32。1M 时每 rank：ratio-1 层 ≈2.5 GiB/层（20 层）、
  ratio-2 ≈1.25 GiB/层（18 层）、ratio-0 因 `compress_ratio(l).max(1)` 仍按 1M 分（5 层 ≈2.5 GiB）
  ⇒ **≈85 GiB/rank**（单序列、TP 下全复制），是 8.4GB 唯一 store 的 ~10×。页化/共享（consumer 不分配 ring）
  才能回到 8.4GB。B300 实测可用 ≈275GB/rank（`perf-roadmap.md:759` 的 213GB/275GB OOM），不是 180GB；
  85 GiB KV + 39 GiB 权重已吃紧，若再挂 189 GiB engram 表必 OOM（当前 `load.rs::skip_prefixes` 跳过）。
- ✅ **config 实测值**（`/tmp/dsv41/config.json`）：`index_n_heads=32`、`index_head_dim=128`、`index_topk=512`、
  `candidate_topk_blocks=2048`、`candidate_block_size=8`、`window_size=128`、`head_dim=512`、
  `compress_ratios=[0,0,2×18,1×20,0,0,0]`、`kv_source=[2,8,14,20]`、`index_source=[2,8,14,20,24,28,32,36]`。
- ⚠️ **indexer 上限必须锁步抬**：`kIdxMaxPos=65538`（`dsv41_kernels.cu:1886`）在 1M 下会静默截断候选
  （`indexer_score_kernel` 的 `if (n_pos > kIdxMaxPos) n_pos = kIdxMaxPos`，`:1901`），与 `DSV41_MAX_POS` 必须同步。

## 1 三问核对（读码，非推断）

| 问题 | 结论 | 证据 |
|---|---|---|
| `gemm_fp8_gemv_kernel` 有 m>1 分支？ | **没有**。GEMV 核 M=1-only；m>1 走**另一个**核 `gemm_fp8_kernel`（16×64 tile，SIMT） | `dsv41_kernels.cu:2006` / `:2033`；AR store 融合遇 m>1 直接返错 `:2030` |
| `hc_mixes_tail_kernel` 支持 rows>1？ | **支持**。grid=(rows,)，上限 `DSV41_HC_SPREAD_MAXR=2048`、`mix≤64` | `:3595` / `:3527` / `:2719` |
| `sparse_attn_pf_kernel` 要 chunk 版？ | **半支持**。已收 `(b,m)`、grid=(b·m,h)、idxs 每行独立；但 `clen` 是**单指针**、`n=window+*clen` 全批共享 | `:484-491` |

各段 m=1→m=chunk 的改动量：

- **embed / hc**：`embed_expand_dev(...,n)`、`hc_collapse_norm(rows)` 已带 rows → 只改调用（最省，Phase 1）。
- **线性/投影**：`lin()` = `quant1(rows=1)+gemm_fp8_mx(m=1)`（`chain_dev.rs:709`）→ 需新增 `quant_fp8(rows=n)` + `gemm_fp8_mx(m=n)`（落到 tile 核）。bf16 侧 `lin_bf16` 已走 cuBLAS `gemm_bf16(rows)`，天然 chunk。
- **MoE**：`route_topk(rows=1)`、`expert_gemv_fp4_batched(slots=topk)` 按 slot 切而非按行 → 需 slots×chunk 2D 批化；`swiglu_limit(rows)` 已支持行（`device.rs:1322`）。
- **indexer**：`indexer_score_kernel` 本就按行（`lens[mm]`，`:1254-1258`），天然支持 m 行；只卡在 `kIdxMaxRows=8`。

## 2 分块 attention（两段式，chunk=64）

单 chunk = `[start, start+64)`：

- **(A) 共享上下文段**：64 个 query 对 "chunk 之前" 的 window ∪ 压缩候选做一次 attention，
  同一份 idx 直接喂 `sparse_attn_pf_kernel(b=1,m=64)`（已支持）——chunk 的主要工作量。
- **(B) chunk 内因果段**：chunk 自身 64 行 KV 互相 attend（小 causal mask）；成本 O(64²·h·d) 可忽略。
- **收尾**：window ring 一次 append 64 行（`ring_append` 现为单行 → 加 rows）；
  compressor 按 ratio 批量提组，`state_kv/state_score` 只跨 chunk 边界 carry。

## 3 indexer O(n²) 的解法

`compress_len = (start_pos + seqlen) // ratio`（`ref_inference/model.py:734`；ratio ∈ {1,2}，
**不是 n/32**）。5 个 ratio=1 的 index-source 层（20/24/28/32/36）在 1M 时 ≈4G MAC/token，全序列 O(n²)。

1. **chunk 共享扫描**：一个 chunk 的 64 行共享同一段候选区间 → 64 次候选扫描合成一次
   **GEMM (m=64 × n_cand)**。FLOPs 不变，但 key 的读取量降 64×、launch 降 64×，且 GEMM 可上
   tensor core——瓶颈从"延迟/带宽"转"算力"。
2. **两层块预筛**（唯一把 O(n²)→O(n) 的路）：config 已有 `candidate_topk_blocks=2048`、
   `candidate_block_size=8`（候选上限 16384）。先用廉价块分（块内 key 池化后点积）选 top 块，
   再块内精确打分。⚠️ 与参考实现的"全量打分后选块"（`model.py:583`）**语义不等价**，必须与
   golden 逐层对拍后定案；当前 Rust 侧 `chain.rs:252` 传的是全零 logits，只是占位。

## 4 时间预估（B300，粗算）

| 方案 | 非 indexer 主体 | indexer | 合计 |
|---|---|---|---|
| 现状逐 token | ~4h | 尾部每 token ~ms，叠加 O(n²) | **2.8–4.4h** |
| Phase 1+2（m=64，GEMM+共享扫描） | 0.1–0.3 ms/token | ~51s 理想 / 5–20min 现实 | **10–25 min** |
| Phase 3（+块预筛） | 同上 | ~0.3s 理想 | **5–12 min** |

（算法：40 层 ≈32 GFLOP/token；fp8 GEMM 取 300–400 TFLOPS 有效。indexer 全量
= Σ 4096·p/ratio ≈ 1e16 MAC ≈ 2e16 FLOP。）

## 5 与 prefix hit 的协同

kv-page-design：P=128 token/page、链式哈希 `h_p = H(h_{p-1} ‖ tokens[p] ‖ model_salt)`；命中 L 页后
D2D 拷回 ring/index_k，直接跳到 `[L·P, n)`。

**chunked + prefix = 多轮 agent 的解**：第 k 轮只 prefill 尾部 m 个新 token，成本
≈ `O(m·L/ratio + m²)`；m ≤ P 时 chunk 数 = 1，退化为"单 chunk 增量 prefill"，稳态成本 ≈ O(m)，
与历史长度几乎无关。

## 6 实施分阶段

- **Phase 0（前提，先做）**：抬 `DSV41_MAX_POS`/`kIdxMaxPos`（锁步）/`kIdxMaxRows`（≥chunk）；
  ring+index_k 页化（PagePool+PageTable，可仿 `ferrite-kv/src/lib.rs:212`）。
- **Phase 1（最简，收益最大的机械改动）**：chunked embed + hc + 线性/投影 + MoE + 出口 norm/head。
  这些核已有 rows/GEMM 路径，`prefill_chain` 换成 `for c in ids.chunks(C)`。
  验收：四段文本与逐 token 路径对齐（首轮允许 1-ulp 级 GEMM 差异，但不许翻字）。
- **Phase 2**：chunked attention（§2 两段式）+ chunk 内 compressor/ring 批量 append + §3① indexer 共享扫描。
- **Phase 3**：§3② indexer 两层块预筛，与 ref 逐层对拍。

## 7 不要做

- ❌ 不要在逐 token 路径继续调 occupancy/launch（延迟受限已证，prefill 只会更糟）。
- ❌ 不要 MLA 吸收（分块后 prefill 转算力受限，省内存换 2× FLOPs 是负和）。
- ❌ 不要用 fp16/fp8 KV 换字节（decode 已实测回归）。
- ❌ 不要用常量钳位 hack 绕上限（indexer i0 教训）；`kIdxMaxPos` 与 `DSV41_MAX_POS` 必须锁步抬。
- ❌ 不要把 MTP/投机引入 prefill。
- ❌ 不要先做 CP/EP 大改（排在 Phase 1–3 之后）。
- ❌ 不要只抬 `kIdxMaxRows` 而不重构 `g_idx_score` 尺寸策略（64×1M×4B=256MB）；
  Phase 2 应把 score 收进 `kIndexerChunk` 内联循环，删掉全局 score 数组。
