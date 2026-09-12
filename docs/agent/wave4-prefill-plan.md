# Wave 4 实施文档 — 1M 上下文 prefill

> 基线：`/home/smith/src/ferrite`（只读侦察，未改任何代码）· 2026-09-12
> 上游输入：`/tmp/prefill_research_report.md`（P0 清单）、`docs/agent/dsv41-prefill-1m-plan.md`、`docs/agent/unified-engine-battle-plan.md`（Wave 2/3 状态）
> 本文件的所有 `file:line` 均已读码核对；凡推断均标 ⚠️。

---

## 0. 开工顺序（一句话）

**P0-B（GLM serve 接 chunked）→ P0-E（GDN 对齐）→ P0-F（显存预算表）→ P0-C（KV 口径统一）→ P0-A（抬上限锁步）→ P0-D（indexer 分块）→ P0-G（DSV41 chunked，与 B~D 并行）。**

理由见 §1 的依赖图。**P0-B 必须先做**：它是唯一「不碰 kernel、只改调度」的项，且 chunk 语义一旦落地，P0-D（chunk 共享扫描）、P0-G（多行链 chunk 化）、P0-A 的 tile 预算入院才有共同的坐标系。P0-A 必须排在 P0-C/P0-F 之后——**先抬上限再统一口径 = 把「启动拒」换成「跑一半 OOM」**。

---

## 1. P0 依赖图与排序

```text
P0-B GLM serve chunked ──┬──> P0-D indexer 分块（chunk 共享扫描）
   (零 kernel 改动)       │
                         ├──> P0-G DSV41 chunked（复用多行链）
P0-E GDN 对齐 ───────────┘   （与 B 同批验证，独立小项）

P0-F 显存预算表 ──┬──> P0-A 抬上限锁步
P0-C KV 口径统一 ─┘   （A 的前置：不统一口径，抬上限只会换一种崩法）
```

| 序 | 项 | 依赖 | 为什么这个位置 | 风险 |
|---|---|---|---|---|
| 1 | **P0-B** GLM serve 接 chunked | 无 | 只改调度；立刻拿到「整段 vs 分块」parity 基线；解锁后续所有 chunk 议题 | 低 |
| 2 | **P0-E** GDN chunk 对齐 | 无 | 与 B 同批（都靠 `prefill_chunk` 反复调用同一 seq）；只需连续性和数值验证 | 低 |
| 3 | **P0-F** 显存预算表 | 无（纯读码建表） | 是 P0-C 的目标函数、P0-A 的准入条件 | 无 |
| 4 | **P0-C** DSA KV 口径统一 | 无 | 解锁 A；也是 P2-4（KV fp8）的前置 | **高**（动 kernel 内存布局 + 数值对拍） |
| 5 | **P0-A** 抬上限锁步 | C + F | 口径统一后才能定 1M 的真实上限 | 中（锁步漏一个 = 静默截断） |
| 6 | **P0-D** GLM indexer 分块 | B | 1M 的**时间**可用性（O(n²)→O(n)） | **高**（新 kernel + 与 golden 逐层对拍） |
| 7 | **P0-G** DSV41 chunked | B（多行链） | 可并行；复用 Wave 2 的 `step_rows` 形状 | 高（`compress_rows`/`publish_index_key` 的已知缺口） |

---

## 2. 每项的具体改动点

### P0-B — GLM serve 接 chunked prefill

**现状（已核对）**：引擎是 n-可变的，只有调用方传整段。

- 拒绝上限：`crates/ferrite-serve/src/gpu_engine.rs:37`
  ```rust
  const MAX_CTX: usize = 8100;
  ```
  拒绝点：`gpu_engine.rs:165-171`（`submit()` 里 `prompt + max_new > MAX_CTX`）。
- **调用点（唯一要改的地方）**：`crates/ferrite-serve/src/gpu_engine.rs:227`
  ```rust
  self.cluster.prefill_chunk(cluster_seq, &prompt)?;   // ← 整段
  ```
  上下文是 `tick()` 的 admission 段（`gpu_engine.rs:221-241`）。改为：
  ```rust
  // budget 来自 SchedConfig::prefill_token_budget（默认 512）
  for chunk in prompt.chunks(budget) {
      self.cluster.prefill_chunk(cluster_seq, chunk)?;
  }
  ```
  ⚠️ 必须**在同一个 tick 内循环完**（否则 admission 与 decode 的交错语义变化），且 `g.prompt_len` 的记账（`gpu_engine.rs:229`）保持不变——`prefill_chunk` 内部 `ensure_seq_all`（`tp.rs:827`）是幂等的，但 `shards[0].ensure_seq`（`ferrite-exec/src/lib.rs:168-187`）只在第一次插入 `SeqRuntime.tokens`，**后续 chunk 不会 append tokens**。
  ⇒ **第二处改动**：`crates/ferrite-exec/src/lib.rs:168` 的 `ensure_seq` 需要区分「首次建 runtime」与「chunk 推进」（或新增 `advance_seq_tokens(seq, chunk)`）。这是本次改动最容易踩的坑：`tokens` 是 decode 的 KV 生命周期依据，第一遍只写了 prompt[0..C]，后面所有 chunk 都没进 `tokens`。
- 引擎侧已就绪，无需改：
  - `crates/ferrite-exec/src/tp.rs:826` `TpCluster::prefill_chunk(&mut self, seq: u64, chunk_tokens: &[u32])` —— 签名本来就是 chunk。
  - `crates/ferrite-exec/src/lib.rs:320` `Engine::prefill_chunk` —— 同样，`n = chunk_tokens.len()`。
  - `crates/ferrite-exec/src/lib.rs:226`（CPU/mock Engine 的驱动）**已经**是 chunk 循环 + `advance_prefill`，可作参考实现。
- 预算来源：
  - `crates/ferrite-dispatch/src/batch.rs:114` `prefill_token_budget: usize`（默认 `512`，注释「chunk ladder is 512」）。
  - `crates/ferrite-dispatch/src/batch.rs:132-146` `SchedConfig::default()`。
  - ⚠️ `GpuEngine` **不持有** `SchedConfig`（`gpu_engine.rs:59-79` 只有 `cluster/arena/live/queue/...`）。**要么**给 `GpuEngine` 加一个 `budget: usize` 字段（最小改动），**要么**把 Wave 3 的「GpuEngine 接 ferrite-dispatch」提前——后者是把 `SchedConfig` 里的 radix/hicache 一起接上（见 §6），建议 Wave 4 先取最小版本（加 `budget`），把调度器接入留给 Wave 3。
- 验收：同一 prompt 整段 vs 分块 → 四段文本一致（允许 1-ulp GEMM 差，**不许翻字**）。注意 `AGENTS.md` §测试纪律：只跑一轮，单轮读数即结论。

### P0-E — GDN chunk 长度对齐 prefill chunk

**现状（已核对）**：**实际上已经是对的**，属于「验证 + 确认」而不是「改代码」。

- 调用：`crates/ferrite-exec/src/lib.rs:659-661`
  ```rust
  self.backend.gated_deltanet_chunk(&q, &k, &v, &beta, &gate, a_log, &state_in, &mut core, &mut state_out)?;
  ```
- kernel：`crates/ferrite-kernel/src/cuda.rs:3012-3034` `gated_deltanet_chunk` → `ferrite_gdn_chunk_v2`（WYF-parallel chunkwise，内部 32-token chunk；`README.md:97-98`「32-token chunks, 32x fewer launches」）。
- **状态跨 chunk 边界已经连续**：
  - GDN state：`lib.rs:651-663`（读 `linear_states[(seq,layer)]` → `state_in`，写回 `state_out`）。
  - conv tail：`lib.rs:555-566`（`conv_tails[(seq,layer)]` carry）。
  ⇒ 每个 `prefill_chunk` 调用返回后 state 已落回 host map，下一个 chunk 自然续算。**C=512 的 chunk 内部会切 16 个 WYF 32-chunk，WYF 链在 kernel 内闭合**——P0-B 落地后直接可测。
- 需要做的：
  1. 验证 `n=512` 与 `16 × n=32` 与 `512 × n=1` 三条路径的 state/输出一致（`ferrite_gdn_chunk_v2` 的 tail-chunk 回退分支，`cuda.rs:3024-3026` 注释）。
  2. ⚠️ **容量**：`gated_deltanet_chunk` 每次调用都 `DevBuf::alloc(q.numel())` 等 6 个 buffer（`cuda.rs:3018-3027`）——n=512 时 q 是 `[512,64,128]` = 4.2M floats ≈ 16 MB × 6，**每次 chunk 重新 malloc/free**。进 CUDA graph 前必须池化（与 P2-1 prefill graph 同一前置）。

### P0-F — 显存预算表 + 启动自检

**这是纯读码建表 + 一个新模块**，是 P0-C 的目标函数。

模型（对标 SGLang `arg_groups/memory_hook.py:43-59` 的 `reserved_mem = chunked_prefill_size*1.5 + max_bs*2`）：

```
per_seq_bytes(n_max, chunk) =
      Σ_dsa_layers  KV_bytes_per_token(n_max)          # 见 §4
    + Σ_gdn_layers  gdn_state_bytes                    # 固定，与 n_max 无关
    + activation_bytes(chunk)                          # ∝ chunk，SGLang 的 1.5× 系数
    + weights_bytes                                    # 常量
```

- 激活项 ∝ chunk。GLM 的激活大头：`hc_mult * hidden` 的 4 流残差、MoE 的 `token × topk × 2*inter`（`chain_dev.rs:1336-1340` 的 `ex_act_r` 就是这形状）。
- **落点**：新建 `crates/ferrite-serve/src/budget.rs`（或 `ferrite-dispatch` 内），在 `GpuEngine::new`（`gpu_engine.rs:82`）里做启动自检，不合法**启动即报错**。
- 输入维度必须从 kernel 侧真实取：
  - `DsaCacheState` 分配（`crates/ferrite-kernel/src/cuda.rs:4144-4162`）。
  - `DsaKvPool` 维度（`crates/ferrite-kv/src/lib.rs:213-254`）。
  - `LayerCache` 分配（`crates/ferrite-models/src/dsv41/chain_dev.rs:1203-1214`）。

### P0-C — DSA KV 口径统一（latent 512 vs kernel 展开 41088）

**两套口径的精确落点（已核对）**：

| 口径 | 每 token 每 DSA 层 | 代码 |
|---|---|---|
| latent | `kv_lora_rank(512) + index_n_heads*index_head_dim(4096) = 4608` floats | `crates/ferrite-kv/src/lib.rs:239-241`（`latent = cfg.dsa.kv_latent_dim()`，`indexer = 32*128`） |
| kernel 展开 | `h*dk + h*dv + h + h + idm + idm = 16384+16384+64+64+4096+4096 = 41088` floats | `crates/ferrite-kernel/src/cuda.rs:4148-4153` |
| 比 | **8.916×** | |

展开式的六块分配（`cuda.rs:4148-4153`）：
```rust
let kn  = self.dsa_alloc(max_tokens * h * dk)?;   // 64*256  = 16384
let vv  = self.dsa_alloc(max_tokens * h * dv)?;   // 64*256  = 16384
let kns = self.dsa_alloc(max_tokens * h)?;        // 64      (fp8 scale)
let vss = self.dsa_alloc(max_tokens * h)?;        // 64
let ki_ = self.dsa_alloc(max_tokens * idm)?;      // 4096    (indexer key)
let kg  = self.dsa_alloc(max_tokens * idm)?;      // 4096    (kpool gate)
```

**关键事实：`dsa_alloc` 是 f32**
`crates/ferrite-kernel/src/cuda.rs:5029-5044`：`cudaMalloc(&mut p, floats * 4)` —— 注释里 `kns`/`vss` 标称 fp8 scale 但仍是 4 B/元素。⚠️ `AGENTS.md` 记载 fp8 KV 因 prefill/batched 格式分歧 bug 被回退到 f32（`a0e262d`）。**任何「1M 需要多少显存」在统一口径前无意义**（研究报报告 §A4-1）。

**统一的三个技术选项**：

1. **直接换 latent 口径**（推荐起步）：`k_nope`/`v` 改成存 `kv_lora_rank=512` 的 latent，`kv_b` 展开推迟到 attention 的 sparse kernel 内做（MLA 吸收）。⚠️ `docs/agent/dsv41-prefill-1m-plan.md:108` 明确「不要 MLA 吸收」——**那条禁令是对 DSV41 说的**（DSV41 分块后是算力受限，省内存换 2× FLOPs 负和）；GLM 侧是带宽受限（fp8 GEMV 路径注释 `cuda.rs:2392-2395` 承认 prefill 慢于 tiled GEMM），**GLM 侧的 MLA 吸收需要重新评估**，不能照搬 DSV41 的禁令。
2. **页化 + 按需展开**：保持 kernel 展开口径，但只在 attention 的候选区间内展开（topk 2048 而非全量）。与 P0-D 的候选扫描天然合并。
3. **fp8 KV**（P2-4）：口径统一后重做。

**受影响的消费方**（改口径必须全部同步）：
- `ferrite_dsa_cache_append`（`cuda.rs:4183-4189`，写 `k_nope`/`v`）
- `ferrite_kpool_compress`（`cuda.rs:4206-4209`，读 `k_idx`/`k_gate`）
- `ferrite_indexer_topk`（`cuda.rs:4228-4232`）
- `ferrite_pool_expand`（`cuda.rs:4245-4248`）
- `ferrite_sparse_attn_v2`（`cuda.rs:4264-4271`，读 `k_nope`/`v`）
- GPu batched 路径：`cuda.rs:4529` `dsa_layer_dev_batched`（同六块分配，按 B 行各自 t0）
- 主机侧镜像：`ferrite-exec/src/lib.rs:880-918`（CPU golden path 的 `k_nope`/`v`/`k_idx`/`k_gate` Vec）——**这是数值对拍的 golden，必须同步改，否则对拍失效**。

### P0-A — 抬上限「锁步」

**GLM 侧的三处上限（必须同抬）**：

| 常量 | 位置 | 当前值 | 说明 |
|---|---|---|---|
| `MAX_CTX` | `crates/ferrite-serve/src/gpu_engine.rs:37` | 8100 | serve 层直接拒（`gpu_engine.rs:165`） |
| `FERRITE_DSA_MAXT` | `crates/ferrite-kernel/src/cuda.rs:4144-4147` | 8192（env 默认） | 每 (seq, family) 的 cache 容量 |
| `max_t`（indexer smem） | `kernels/cuda/ferrite_kernels.cu:1669` | `2048` | ⚠️ **报告漏了这个**：`smem = max_t*5*4`，`max_t = max_npools`，即 **npools 上限 2048 → max_tokens 8192**。只抬 `FERRITE_DSA_MAXT` 而不抬它，kpool 压缩会被静默截断 |

**DSV41 侧的三处上限（必须锁步）**：

| 常量 | 位置 | 当前值 |
|---|---|---|
| `DSV41_MAX_POS` | `crates/ferrite-models/src/dsv41/chain_dev.rs:1180-1187` | 65536（env 默认） |
| `kIdxMaxPos` | `kernels/cuda/dsv41_kernels.cu:2575` | `65536 + 2` |
| `kIdxMaxRows` | `kernels/cuda/dsv41_kernels.cu:2574` | `8` |

⚠️ **静默截断的精确位置**：
- `dsv41_kernels.cu:2604`（`indexer_score_kernel`）与 `:2696`（同族）：`if (n_pos > kIdxMaxPos) n_pos = kIdxMaxPos;` —— **无报错**。
- `dsv41_kernels.cu:6730`：`if (b * m > kIdxMaxRows) return cudaErrorInvalidValue;` —— 这一处**是有报错的**（grid 装载点），但 `kIdxMaxRows=8` < prefill chunk C ⇒ **chunk 超过 8 行会被拒**，P0-G 必须先抬它。
- `kIdxMaxRows` 抬起会连带 `g_idx_score[kIdxMaxRows][kIdxMaxPos]`（`dsv41_kernels.cu:2576`）膨胀：8×65538×4 B = 2 MB 现在是放在 `__device__` 全局；抬到 512×1M 是 2 GB，**不可行**。⇒ `dsv41-prefill-1m-plan.md:113-114` 的「不要只抬 kIdxMaxRows 而不重构 g_idx_score 尺寸策略」必须遵守：**Phase 2 要把 score 收进 `kIndexerChunk` 内联循环，删掉全局数组**。
- `FERRITE_DSA_MAXT` 抬高的连带代价：`dsa_alloc`（`cuda.rs:5029`）是 per-(seq,family) 的整块 `cudaMalloc`，11 个 family × 每块 ~0.5 GB @ 8192（`cuda.rs:4146-4147` 注释原文「these huge (~0.5GB each) allocations」）⇒ 抬到 1M 是 11 × 64 GB/序列，**必须先页化**（P0-C 选项 1/2）。

### P0-D — GLM DSA indexer 分块（O(n²) → O(n)）

**现状（已核对）**：

- **CPU golden path**：`crates/ferrite-exec/src/lib.rs:922-1026`
  - kpool 压缩：`lib.rs:928-962`（**每次调用对全量 `total` 重算 npools**）
  - indexer topk：`lib.rs:969` `self.backend.indexer_topk(&qi, &pool_idx_all, &w_idx, select_k, ctx0/kpool, &mut idx_pools)`
  - pool 展开：`lib.rs:971-1018`
  - sparse attention：`lib.rs:1026`
- **device path**：`crates/ferrite-kernel/src/cuda.rs:4195-4274`
  - kpool 压缩：`cuda.rs:4200-4212`（`max_npools = (8192 + kpool-1)/kpool`，`npools` 由 pinned total 派生 —— **也是全量重算**）
  - indexer topk：`cuda.rs:4222-4235`
  - pool expand：`cuda.rs:4241-4252`
  - sparse attention：`cuda.rs:4254-4274`（`ferrite_sparse_attn_v2`，split-K 由 `splits = (256/(n*h)).clamp(1,32)`）
- **kernel**：`kernels/cuda/ferrite_kernels.cu:1539` `indexer_topk_kernel` —— 头注释 `:1536-1537` 自认「v1: full scan per row (t <= 1M tokens OK for correctness harness)」，**每行对全部 npools 打分**。

**复杂度**（GLM 全程 1M）：kpool 压缩每次 `O(total)`，`n/C` 个 chunk ⇒ **O(n²/(kpool·C))**；indexer topk 每 chunk `O(C·npools)` ⇒ **O(n²/kpool)**。两条都是 n²。

**改动设计（三段）**：

1. **增量 kpool 压缩**（不改 kernel，收益最大）：`cuda.rs:4200-4212` 现在每次都全量重算。改为只压缩「本 chunk 新覆盖的完整 pool」+ 追加到 `pool_keys` 尾部；`npools` 只增不减。⇒ 压缩总量 O(n)，而不是 O(n²/C)。
   - 需新增一个 kpool cursor（host 侧 `DsaCacheState` 加 `pool_count` 字段，`cuda.rs:4162`）。
   - ⚠️ 与 graph-safe 的 pinned total 机制（`cuda.rs:4172-4180`）兼容：cursor 也要 pinned 化，否则捕获后冻结。
2. **chunk 共享候选扫描**：一个 chunk 的 C 行共享同一段候选区间 ⇒ 把「C 次 row-wise 打分」合成「一次 GEMM `m=C × n_cand`」。
   - **候选预算是现成的**：`candidate_topk_blocks=2048` / `candidate_block_size=8`（`crates/ferrite-model/src/config.rs:259-260` ⚠️ 需确认 GLM 的 `Glm53FlashConfig` 是否有同名字段；DSV41 侧确认为 `crates/ferrite-models/src/dsv41/config.rs:76-77,259-260`）。
   - **kernel 已存在但未接线（GLM 侧要新写）**：DSV41 的 `dsv41_candidate_blocks`（`kernels/cuda/dsv41_kernels.cu:1826` kernel / `:6826` launcher）已在 device 注册（`crates/ferrite-models/src/dsv41/device.rs:266,895,2222`），Rust 侧 `ops::select_candidate_blocks`（`crates/ferrite-models/src/dsv41/ops.rs:396-441`）是 CPU golden。
   - **接线点（关键）**：
     - DSV41 device 侧 `Device::indexer_topk`（`crates/ferrite-models/src/dsv41/device.rs:2201-2219`）**已经有 `candidates` 参数 + `uses_candidates`**。
     - **真正没接线的地方是调用者**：`crates/ferrite-models/src/dsv41/chain_dev.rs:4764-4783`（`indexer_rows` 的 device 调用）传的是 `std::ptr::null()` + `false`。
     - 主机参考 `crates/ferrite-models/src/dsv41/chain.rs:249-259` 也传全零 logits（`let logits = vec![0f32; rows * compress_len];`）。
   ⇒ **P0-D 的 DSV41 接线点 = `chain_dev.rs:4769` 的 `std::ptr::null()` 换成真实 `mask` 指针 + `uses_candidates=true`**；mask 的产出点是 `Device::candidate_blocks`（`device.rs:2222`），其输入 logits 必须来自「先打分后选块」——**而 SGLang 的语义是块内池化预筛**（研究报报告 §C0#2 / `model.py:583`），⚠️ **两者语义不等价，必须逐层与 golden 对拍后定案**（`dsv41-prefill-1m-plan.md:70-73` 同款警告）。
3. **chunk 共享扫描 kernel**（GLM 侧新写）：把 `indexer_topk_kernel` 的 per-row full scan 改为「一次 pass 算 C 行 × npools 的分数矩阵 → 每行 topk」。⇒ key 读取量降 C 倍、launch 降 C 倍、可上 tensor core。

### P0-G — DSV41 chunked prefill

**现状（已核对，行号已漂移）**：

- `crates/ferrite-dsv41/src/serve.rs:668-674` `prefill_chain`（**不是报告说的 `dsv41-run.rs:890`**）：
  ```rust
  fn prefill_chain(chain: &mut DevChain<'_>, ids: &[u32]) -> Result<u32> {
      chain.reset()?;
      let mut next = 0u32;
      for (i, &t) in ids.iter().enumerate() { next = chain.step(t, i)?; }
      Ok(next)
  }
  ```
- 另一条路径：`crates/ferrite-dsv41/src/bin/dsv41-run.rs:188-190`（单机）/ `:389-391`（TP rank 循环），同一模式。
- **没有 prefill 分支**：`chain_dev.rs:2336 step_body` 只有单 token。`step_dev`（`chain_dev.rs:2224-2227`）也是。**不要去找一个不存在的 prefill 分支**（`dsv41-prefill-1m-plan.md:19-22`）。

**改动**：`prefill_chain` 的 `for t in ids` → `for c in ids.chunks(C)`，每 chunk 走 **多行链**（§3）。

- Phase 0（前提）：抬三个上限（P0-A 的 DSV41 部分）+ ring/index_k 页化。
- Phase 1（机械改动，收益最大）：chunked embed + hc + 线性投影 + MoE + 出口 norm/head。**这一步的多行 kernel 已经全部就位**（battle plan 风险表第 1 行已核对：`dsv41_sparse_attn` grid `(b*m,h)`、`hc_mixes(rows)`、`moe_route(rows)`、`compressor(b,seqlen)`、`gemm_fp8_mx(m,n,k)`、`embed_expand_dev(n)`）。
- Phase 2：chunked attention 两段式（§3）+ compressor/ring 批量化 + indexer 共享扫描。
- Phase 3：indexer 两层块预筛（见 P0-D 接线点）。

---

## 3. 多行链复用分析（Wave 2 的 `step_rows` vs prefill chunk）

### 3.1 `step_rows` 的结构（已逐行读）

`crates/ferrite-models/src/dsv41/chain_dev.rs:2671-2879`：

| 阶段 | 位置 | 形态 |
|---|---|---|
| pos_base D2H + `pos_rows` H2D | `:2690-2695` | 1 读 1 写 |
| `ids_r`/`premix_r` H2D | `:2693-2702` | |
| `embed_expand_dev(rows=m)` | `:2705-2713` | ✅ native rows |
| `engram_hash_step` 逐行 | `:2750-2766` | ⚠️ per-row（kernel 是单 token） |
| `engram_apply_rows` | `:4915` | ⚠️ gather per-row，collective 是 m 行 |
| **`layer_rows`** 循环 | `:2780-2785` | |
| ├ `attention_rows` | `:4339-4613` | 见下 |
| ├ `moe_rows` | — | ✅ native rows（pin GEMV arm，TODO#4） |
| final collapse+norm(rows=m) | `:2790-2805` | ✅ |
| `head_gemv_bf16_mrows` | `:2825-2832` | ✅ folded（m≤8） |
| 逐行 argmax + 1 次 D2H | `:2852-2868` | ⚠️ per-row |

`attention_rows`（`chain_dev.rs:4339-4613`）的逐行化清单：

| 步骤 | 位置 | 形态 | prefill chunk 需求 |
|---|---|---|---|
| `lin(wq_a)`/`lin(wkv)`/`rmsnorm`/`lin(wq_b)` | `:4356-4395` | ⚠️ **4×m launches** | ⇒ 应合成 `gemm_fp8_mx(m=C)` |
| `apply_rope(q)` | `:4402-4417` | ⚠️ 1×m | ⇒ 块形式（`off` 传 pos_base、`step=1`） |
| `rmsnorm(kv)` + `apply_rope(kv)` | `:4418-4439` | ✅ native rows（`step=1`） | 直接用 |
| `ring_append` + `window_idxs` **交错逐行** | `:4463-4476` | ⚠️ 2×m | ⇒ 需要 batched 版（见 3.2） |
| `compress_rows` | `:4800-4907` | ⚠️ **per-row pool+commit**，且**只建模 1 个完成的组** | ⇒ 必须泛化（见 3.3） |
| `indexer_rows` | `:4690-4785` | ⚠️ per-row query + per-row `indexer_topk` + `null` candidates | ⇒ 共享扫描 + 候选接线 |
| `sparse_attn` | `:4511-4527` | ⚠️ 1×m | ⇒ `sparse_attn_pf(b=1,m=C)`（kernel 已支持） |
| `apply_rope(o)` | `:4528-...` | ⚠️ 1×m | ⇒ 块形式 |

### 3.2 三类复用面

**（A）直接复用（形状层）——无需改动**

- 全部 `*_r` 缓冲的行索引约定（`pos_rows[r]`、`ids_r[r]`、`h_r[r*hc*dim]`）。
- `hc_mixes`/`hc_collapse`/`hc_post`/`rmsnorm`/`moe_rows`/`embed_expand_dev` 的 native `rows` 维度。
- `engram_apply_rows` 的 collective（按行 rank 序 reduce，逐行 bit-identical）。
- `layer_rows` 的 premix 线程（`pa` slot 0/1/2，`:2775-2786`）。
- 出口的 `head_gemv_bf16_mrows`（m 行折叠一次 head 权重流）。

**（B）复用但必须扩容/泛化（正确性层）**

| 缺口 | 位置 | 问题 | 修法 |
|---|---|---|---|
| **scratch 全部按 `VERIFY_ROWS=6` 分配** | `chain_dev.rs:1300-1345`（`h_r`…`kvp_r`/`scp_r`/`ex_act_r`…）+ `VERIFY_ROWS` 常量 `:84` | m=512 时全部越界 | 新增一套 chunk 尺寸的 scratch（不要改 VERIFY_ROWS，Wave 2 的 verify 依赖它）+ `step_chunk()` 入口 |
| **`compress_rows` 只建模 1 个完成的组** | `chain_dev.rs:4795-4799`（**文档自己承认的 TODO#1**） | chunk C=512、ratio=2 ⇒ 完成 256 组；state kernel 的 decode 分支只携带 1 行 | 新增 `compress_chunk()`：`compressor(b=1, seqlen=C)` 批量提组 + state 跨 chunk 边界 carry |
| **`publish_index_key` 只发 1 个 key** | `chain_dev.rs:4615-4630` | `indexer_owns_k` 时在 slot `*clen - 1` 写**一个** key；chunk 会产生 256 个 | 改成批发布（`lin_bf16(latent[C], wk)` + rmsnorm + rope，一次 C 行） |
| **`indexer_rows` 传 null candidates** | `chain_dev.rs:4769` | 候选预筛没接线 | 接 `Device::candidate_blocks`（`device.rs:2222`） |
| **`indexer_topk` 逐行** | `chain_dev.rs:4764-4783` | C 行 ⇒ C × launch | 共享扫描（`Device::indexer_topk` 已收 `b,m`） |
| **`ring_append`+`window_idxs` 逐行交错** | `chain_dev.rs:4463-4476` | 2C launches；且**不能简单换成「整块先 append」** | 见下 ⚠️ |

⚠️ **ring 的因果性陷阱（Wave 2 已踩过，别重踩）**：`chain_dev.rs:4450-4462` 的注释记录了 `verify_ring_win` 融合核的失败根因——「appended the whole block first and derived the indices from slot numbers, which broke exactly there: once `base+r >= window` the `v > start_pos` filter never fires (v is a SLOT, not a position) and every row but the last read the block's own future rows」。⇒ **prefill chunk 的 batched ring 版本必须用位置（不是 slot）做因果过滤**，或者采用 DSV41 1M plan §2 的**两段式**：
- **(A) 共享上下文段**：C 行对「chunk 之前」的 window ∪ 压缩候选做一次 attention（同一份 idx 喂 `sparse_attn_pf(b=1,m=C)`——kernel 已支持，`dsv41-prefill-1m-plan.md:52-58`）。
- **(B) chunk 内因果段**：chunk 自身 C 行互相 attend（小 causal mask，`O(C²)`）。
- 收尾：window ring 一次 append C 行；compressor 按 ratio 批量提组。
两段式的 (A) 段正好复用 `attention_rows` 的 idx 构造（只要把它从「逐行 append+window」换成「先算 chunk 前的 window ∪ 压缩候选」）。

**（C）不复用（性能层）——逐行化只为 parity，不是 prefill 的速度解**

- Wave 2 实测：verify 6 行朴素版 **45.3ms**（battle plan「Wave 2 状态快照」）= 7.55ms/行。**用这个速度做 1M prefill 是 7560 s ≈ 2.1 h** —— 和逐 token 的 2.8-4.4 h 同量级。
- ⇒ `step_rows` 的逐行化是 **parity 目标 / 形状地基**，不是速度路径。速度必须来自 `verify-perf-design` 在同做的批量化（`docs/agent/dspark-verify-perf-plan.md`：多行 GEMV bit-identity C1-C5，目标 verify 39→5ms）。
- **共享件**：`dspark-verify-perf-plan.md` 的多行 GEMV/图化优化 **= prefill chunk 的 Phase 1 优化**，同一份工作服务两个 Wave。

### 3.3 复用的最终结论

| 归入 | 内容 |
|---|---|
| **直接用** | 行索引约定、native-rows 的 elementwise/hc/MoE、`head_gemv_bf16_mrows`、`engram_apply_rows` 的 collective、`layer_rows` 的 premix 线程 |
| **扩容** | `*_r` scratch（VERIFY_ROWS → chunk 尺寸） |
| **泛化（正确性）** | `compress_rows`（多组）、`publish_index_key`（批发布）、`indexer_rows`（共享扫描+候选）、ring 的因果窗口（两段式） |
| **不复用** | 逐行 launch 形态（性能）；必须换成 batched kernel（`sparse_attn_pf(b=1,m=C)`、`gemm_fp8_mx(m=C)`、共享扫描 GEMM） |

---

## 4. 1M 显存预算表

### 4.1 GLM-5.3-Flash

配置（`crates/ferrite-model/src/config.rs:335-372`）：hidden 4096 · 45 层（前 3 dense）· **34 GDN + 11 DSA** · DSA `h=64` `dk=256` `dv=256` `kv_lora_rank=512` `q_lora=1536` `index_n_heads=32` `index_head_dim=128` `index_topk=2048` `index_kpool=4` · GDN 64 heads × 128 · `max_position_embeddings = 1_048_576`。

**A. DSA KV（11 层，单序列，TP 下各 rank 一份）**

| 口径 | /token/layer | 1M × 11 层 f32 | 备注 |
|---|---|---|---|
| **latent**（`ferrite-kv::DsaKvPool`，`crates/ferrite-kv/src/lib.rs:239-241`） | 4608 floats (512 + 4096) | **202.8 GB / 188.8 GiB** | 其中 **89% 是 indexer key**（4096/4608） |
| **kernel 展开**（`DsaCacheState`，`cuda.rs:4148-4153`） | 41088 floats | **1.81 TB / 1.64 TiB** | 显然不可行 |
| latent + fp8 KV（P2-4） | ~4608 B（未计 scale） | **~101 GB / ~94 GiB** | scale 另算 |
| **差** | **8.916×** | | |

⚠️ `dsa_alloc` 是 **f32**（`cuda.rs:5029-5044`，`floats * 4`），所以上面的 f32 列就是当前的真实字节。
⚠️ **per-family × 11**：`DsaCacheState` 按 `(seq, family)` 分配，family = 「第几个 DSA 层」（`ferrite-exec/src/lib.rs:1037-1045` `dsa_family_index`）⇒ GLM 有 11 个 family，**预算要乘 11**。

**B. GDN state（34 层，固定，与上下文无关）**

`crates/ferrite-model/src/layer.rs:28-38` `linear_state_elems = num_heads * head_dim * head_dim`
= 64 × 128 × 128 × 4 B = **4 MiB/层/序列** × 34 = **142.6 MB/序列**（固定）。

### 4.2 DSV41（`chain_dev.rs:1188-1215` 逐层分配）

公式（`chain_dev.rs:1190-1214`）：`ratio = compress_ratio(l).max(1)`；`max_comp = max_pos/ratio + 2`；
`ring = (window_size + max_comp) * head_dim(512) * 4 B`；`index_k = max_comp * index_head_dim(128) * 4 B`。

层分布（`crates/ferrite-models/src/dsv41/config.rs:14-15`，实测 `/tmp/dsv41/config.json`）：
`compress_ratios = [0,0, 2×18, 1×20, 0,0,0]` ⇒ ratio-2：18 层；ratio-1：20 层；ratio-0：5 层（因 `.max(1)` 仍按 ratio=1 尺寸分配）。

| ratio | 层数 | max_comp @1M | ring/层 | index_k/层 | 小计 |
|---|---|---|---|---|---|
| 2 | 18 | 500002 | 1.024 GB | 0.256 GB | **23.04 GB** |
| 1 | 20 | 1000002 | 2.048 GB | 0.512 GB | **51.20 GB** |
| 0 | 5 | 1000002 | 2.048 GB | 0.512 GB | **12.80 GB** |
| | | | | **合计 @1M** | **87.0 GB / 81.1 GiB per rank** |

对照 @默认 64k：ratio-2 1.52 GB + ratio-1 3.36 GB + ratio-0 0.84 GB = **5.72 GB / 5.33 GiB/rank**（15.2× 放大）。

⚠️ 与两个上游数字的口径差异：
- `dsv41-prefill-1m-plan.md:24-30` 估 **≈85 GiB/rank**（把 ratio-0 也算进去）——与本表 81.1 GiB 一致（差在 GB/GiB 与 ratio-0 是否计入）。
- prefill 研究报告 §A4 的 **~72.5 GiB** 是「只算 ratio>0 层」的数字（87.0 − 12.8 = 74.2 GB = 69.1 GiB ⚠️ 仍有小数差，以本表公式为准）。
- **TP 下全复制**（每 rank 一份 `DevChain`，`crates/ferrite-models/src/dsv41/tp.rs:1-18`）⇒ 1M 直接吃 ~81 GiB/rank。`perf-roadmap.md:759` 记 B300 实测可用 ≈275 GB/rank，39 GiB 权重 + 81 GiB KV 已吃紧；挂 engram 表（189 GiB）必 OOM（当前 `load.rs::skip_prefixes` 跳过）。

### 4.3 总账（单序列 1M，per rank）

| 模型 | 权重 | KV/ring | activation | 合计 | 可用 275 GB？ |
|---|---|---|---|---|---|
| GLM（latent 口径） | ~39-45 GiB ⚠️待核 | 188.8 GiB | ∝chunk（512 时 ~几 GiB） | **~230 GiB** | 勉强 |
| GLM（kernel 展开，现状） | 同上 | 1.64 TiB | — | **不可行** | ✗ |
| GLM（latent + fp8，P2-4） | 同上 | ~94 GiB | — | **~135 GiB** | ✅ |
| DSV41（现状全分配） | 39 GiB | 81.1 GiB | ∝chunk | **~120 GiB** | ✅（但 engram 189 GiB 一挂就爆） |
| DSV41（ring/index_k 页化 + consumer 不分配，P2-5） | 39 GiB | ~8.4 GiB | — | **~50 GiB** | ✅✅ |

---

## 5. 性能目标

### 5.1 必须花的 FLOPs（GLM，1M）

| 项 | 公式 | 量 |
|---|---|---|
| 主干线性/MoE | 1M × ~32 GFLOP/token | **3.2 × 10¹⁶ FLOP** |
| DSA indexer（O(n²) 形态） | Σ_t (t/4)·32·128·2 ≈ 2048 × 1M²/2 | **≈ 1.0 × 10¹⁵ FLOP** |
| DSA sparse attention | 1M × 11 × 2048 × 64 × 512 × 2 | **≈ 1.5 × 10¹⁵ FLOP** |
| GDN | 1M × 34 层 × 定长 state 更新（∝n，非 n²） | ~10¹⁴ FLOP |

（DSV41 口径见 `dsv41-prefill-1m-plan.md:83`：40 层 ≈32 GFLOP/token。）

### 5.2 现状 vs 目标

| 阶段 | 有效算力 | 1M 单请求 | 备注 |
|---|---|---|---|
| **现状（GLM）** | bf16 FMA tile（`matmul_tiled_bf16_kernel`，`kernels/cuda/ferrite_kernels.cu:133`，32×32 SIMT，**非 tensor core**）+ 逐 token DSA（延迟受限） | **3–6 h** | `matmul_dev` 的 fp8 快路注释 `cuda.rs:2392-2395` 自认 prefill 慢于 tiled GEMM |
| **现状（DSV41）** | 逐 token decode 链（m=1 GEMV），10–16 ms/token | **2.8–4.4 h** | `dsv41-prefill-1m-plan.md:8-12` |
| **P0-B/E + chunk=512（GLM，仍是 bf16 tile）** | ~100 TFLOP/s | **~320 s ≈ 5.3 min**（主干）+ indexer | indexer 未分块前仍小时级 |
| **P0-D 后（GLM，chunk 共享扫描 + fp8 tensor core）** | ~300–400 TFLOP/s | **主干 ~90 s + indexer ~10-30 s ≈ 2–3 min** | ⚠️ indexer 的 key 读取是 16 GB/层 × 11 = 176 GB，带宽是第二瓶颈 |
| **P0-D + 前缀命中（P1-3）** | — | **秒级** | 唯一把 1M 变成「几乎免费」的手段 |
| **P0-G（DSV41，Phase 1+2，chunk=64）** | — | **10–25 min** | `dsv41-prefill-1m-plan.md:80` |
| **P0-G（DSV41，+Phase 3 块预筛）** | — | **5–12 min** | 同上 `:81` |

**目标线（Wave 4 验收）**：冷 prefill **≤ 5 min @1M（GLM）/ ≤ 25 min（DSV41）**；热（前缀命中）**≤ 10 s**。

**关键杠杆排序**（研究报报告 §C1，本文件核对后仍成立）：
1. indexer 分块（O(n²)→O(n)）——**最大一笔（可能 >10×）**
2. chunk 共享候选扫描（key 读取降 C 倍 + 可上 tensor core）
3. prefill 主 GEMM 换 tensor core（`matmul_dev` 的 bf16 分支；B300 有 `docs/agent/expert-tcgen05-plan.md` 路线）
4. GDN / MoE 的 chunk 批化（`n>1` 走 GEMM）
5. 前缀命中（P1-3）

---

## 6. 与 Wave 2 / Wave 3 的共享件

| 共享件 | 提供方 | Wave 4 用法 | 冲突风险 |
|---|---|---|---|
| **多行链 `step_rows` + `layer_rows`/`attention_rows`/`compress_rows`/`indexer_rows`/`moe_rows`/`engram_apply_rows`** | Wave 2（`chain_dev.rs:2671,4198,4339,4800,4690,4915`） | P0-G 的 chunk 形状地基；P0-D 的 DSV41 接线点 | ⚠️ **scratch 扩容会与 verify 争内存**（`chain_dev.rs:1300-1345`）；建议**新增** chunk scratch，不动 VERIFY_ROWS |
| **`dspark_snapshot`/`dspark_rollback`**（verify write-set 保存） | Wave 2（`chain_dev.rs:2969`） | ⚠️ **prefill 不需要**（prefill 不回退）；但 `compress_rows` 的 `spec_capture` 分支（`:4843-4857`）会被 chunk 版复制，注意别把 spec 逻辑带进去 | 低 |
| **KV 前缀快照 `kv_snap_*`** | Wave 3（`chain_dev.rs:1360-1371` 分配；`serve.rs:808-833` 的 `prefill_chain` MISS 路径） | P1-3 前缀命中；**prefill 分块后，快照点从「整段后」变成「每个 chunk 后」**（更细粒度） | ⚠️ 快照语义要与 chunk cursor 对齐 |
| **`Admission.prefix_hit`** | Wave 3（`crates/ferrite-dispatch/src/batch.rs:217-223`；`gpu_engine.rs:239` 恒 0） | 接上 radix 后，prefill 从 `[L·P, n)` 开始 | 低 |
| **`SchedConfig`（页预算/radix/hicache 三层）** | Wave 3（`crates/ferrite-dispatch/src/batch.rs:109-146`） | P0-B 的 `prefill_token_budget` 来源 | ⚠️ `GpuEngine` 现在不持有它，见 P0-B |
| **megab 图池 / `decode_step_batched`** | Wave 2/5（`tp.rs:885+`） | P2-1 prefill graph 可复用「per-size 键控 + 指针表内容刷新」机制 | 低 |
| **`verify_ring_win` kernel 的血泪教训** | Wave 2（`chain_dev.rs:4450-4462`） | P0-G 的 batched ring 必须用位置过滤，不能用 slot | **必读** |

---

## 7. 风险 / 陷阱（改之前必读）

1. ⚠️ **`MAX_CTX` 不是 bug，是口径的产物**。`gpu_engine.rs:37` 的注释说它来自 kernel 的 `max_tokens`；`cuda.rs:4146-4147` 说每 cache ~0.5 GB。**先 P0-C 再 P0-A**。
2. ⚠️ **DSV41 的三个上限必须锁步抬**。`kIdxMaxPos` 的静默截断在 `dsv41_kernels.cu:2604`/`:2696`（无报错）；`kIdxMaxRows` 在 `:6730`（有报错但 chunk>8 直接拒）。**只抬一个 = 候选丢失且无报错**。
3. ⚠️ **`g_idx_score[kIdxMaxRows][kIdxMaxPos]` 不能跟着抬**（`dsv41_kernels.cu:2576`）：8×65538×4B=2 MB 现在还 OK，512×1M = 2 GB。必须先把 score 收进 chunk 内联循环（`dsv41-prefill-1m-plan.md:113-114`）。
4. ⚠️ **`ferrite-kernels.cu:1669` 的 `max_t = 2048` 是第三个 GLM 上限**（报告漏了）：`smem = max_t*5*4`，`max_t = max_npools` ⇒ **max_tokens 8192**。抬 `FERRITE_DSA_MAXT` 不抬它，kpool 被静默截断。
5. ⚠️ **`ensure_seq` 不 append tokens**（`ferrite-exec/src/lib.rs:168-187`）：P0-B 分块后第二个 chunk 的 tokens 不会进 `SeqRuntime.tokens`，而 decode 依赖它。
6. ⚠️ **`dsa_alloc` 是 f32 且 per-(seq,family) 整块 malloc**（`cuda.rs:5029`）：抬 `FERRITE_DSA_MAXT` 会让 11 个 family × 每块 64 GB 的整块分配，驱动记账先崩（`chain_dev.rs:1174-1179` 记了同一现象：256 MiB cudaMalloc 在 182 GB free 时失败）。**必须先页化**。
7. ⚠️ **`compress_rows` 的已知缺口**（`chain_dev.rs:4795-4799`）：state kernel 的 decode 分支 + mode-2 pool **只携带 1 行**，一个 block 只建模 1 个完成的组。prefill chunk（ratio=2、C=512）会完成 256 组 ⇒ **现状直接错**。
8. ⚠️ **`publish_index_key` 只发 1 个 key**（`chain_dev.rs:4615-4630`）：chunk 会产生 256 个 key，只发最后一个 ⇒ **部分提交的 block 会对着从未写过的 key 选点**（该函数自己的 doc 注释就写了这个后果）。
9. ⚠️ **`dsv41_candidate_blocks` 的语义不等于 SGLang**：SGLang 是「块内池化预筛」，ferrite 的 `ops::select_candidate_blocks`（`ops.rs:396-441`）是「先取块内 max 再选块」——**必须与 golden 逐层对拍后定案**（`dsv41-prefill-1m-plan.md:70-73`）。
10. ⚠️ **`dist.rs` 已删（Wave 1）**——prefill 研究报告的陷阱 #3（「`dist.rs` 与 `distributed.rs` 两份重复，必须同步改」）**已过期**：`crates/ferrite-exec/src/lib.rs:24` 只声明 `pub mod distributed;`，目录里没有 `dist.rs`。CP 改动只需改 `distributed.rs`。
11. ⚠️ **测速纪律**（`AGENTS.md` §测试纪律）：只跑一轮；跨版本比较必须整树切 commit + 双产物重编。上述 `file:line` 会随 Wave 2/3 的 commit 漂移——**开工前先 `git log` 确认基线**（当前 HEAD `569ca10`）。
12. ⚠️ **硬性禁令**：禁止 `git revert`/`git reset` 回退已提交改动；退化要改成默认关闭的 env 开关。

---

## 8. 验收清单（可直接当 checklist）

| # | 项 | 验收 | 命令/判据 |
|---|---|---|---|
| 1 | P0-B | 整段 vs 分块（512）文本一致 | 四段文本；允许 1-ulp GEMM 差，不许翻字 |
| 2 | P0-B | `SeqRuntime.tokens` 正确累计 | decode 后 `tokens.len() == prompt.len()+generated` |
| 3 | P0-E | `n=512` ≡ `16×n=32` ≡ `512×n=1` 的 state/输出 | 逐元素对拍 |
| 4 | P0-F | 非法配置启动即报错 | 故意配 chunk > 显存 → 启动 fail |
| 5 | P0-C | 单序列 1M KV ≤ 200 GB（f32）/ ~100 GB（fp8） | 打印分配表；与 golden 对拍 |
| 6 | P0-A | 1M 不拒、不静默截断 | 1M prompt 跑通；`*lens` 与 host mirror 一致 |
| 7 | P0-D | O(n²)→O(n·(n/chunk)) | launch 计数 + profile；逐层 golden 对拍 |
| 8 | P0-G | 四段文本逐字；chunk=64 的 1M ≤ 25 min | `dsv41-run` 计时 |

---

## 9. 文档维护（本次侦察发现）

- `docs/agent/dsv41-prefill-1m-plan.md:8`、`:19`、`:100` 引用的 `dsv41-run.rs:890 prefill_chain` **已漂移**——`prefill_chain` 现在在 `crates/ferrite-dsv41/src/serve.rs:668`（`dsv41-run.rs` 只有 467 行，prefill 循环在 `:188-190` / TP 路径 `:389-391`）。**建议更新**。
- 同上 `:15` 的 `DSV41_MAX_POS` 位置（文档写 `chain_dev.rs:496`，实际 `:1180-1187`）——该文档 §0.1 已自我修正过，但 §0 未改。
- prefill 研究报告 §E-3（`dist.rs`/`distributed.rs` 双份）**已过期**，见陷阱 #10。
- 本文件新增的两个事实（研究报报告漏项）：**`ferrite_kernels.cu:1669` 的 `max_t=2048` 第三个 GLM 上限**；**`ensure_seq` 不 append tokens**。

---

## 3. P0-F — 显存预算表（读码建表，2026-09-12）

**分配的来源**（chain_dev.rs:1228-1262）:
```rust
let max_pos  = cfg.max_seq_len.min(DSV41_MAX_POS.unwrap_or(65536));   // :1228-1234
let max_comp = max_pos / ratio + 2;                                    // :1239
ring:    fb((window_size + max_comp) * hd)          // window+压缩latents 连续
index_k: fb(max_comp * index_head_dim)             // pre-RoPE keys
state_kv/state_score: fb(ratio * hd)               // compressor carry
latent:  fb(hd)
```
**关键**：`max_pos` 默认 **65536**（不是 8100——8100 是 GLM 侧的 MAX_CTX；DSV41 的 KV 缓冲按 64k 预分配）。**1M 需要 `DSV41_MAX_POS=1048576` + `cfg.max_seq_len` 的界抬升**（P0-A）。

### DSV41 @ 1M（ratio=2 的 DSA 层，hd=512，index_head_dim=128，window=128）

| 项 | 每层 | 层数 | 合计 |
|---|---|---|---|
| ring（window+压缩 latents，f32） | (128+524288)×512×4B = **1.024 GB** | 11（DSA） | **11.3 GB** |
| index_k（f32） | 524288×128×4B = **256 MB** | 11 | **2.8 GB** |
| state_kv+state_score（compressor carry） | 2×2×512×4B = 8 KB | 11 | 88 KB |
| GDN 层（ring 256KB + state 4KB） | ~0.3 MB | 34 | **10 MB** |
| **合计/卡** | | | **≈14.1 GB** |

**@ 当前默认 64k**：ring 11×64MB + index_k 11×16MB ≈ **0.9 GB/卡**。
**结论**：1M 的 KV 内存 **~14 GB/卡**（8 卡各存全量 latent——DSA 的 KV 不是 TP 切分的），**180 GB 卡完全可行**。**内存不是 1M 的瓶颈**——时间（indexer 的 O(n²)）与 KV 口径（P0-C）才是。

### GLM @ 1M（读码待补——MAX_CTX=8100 是 DSA cache 的 per-family max_tokens 界）

GLM 的 DSA cache 分配在 `ferrite-kernel`（gpu_engine.rs:33 的注释："DSA cache allocation bound (ferrite-kernel's max_tokens per family)"）。**P0-A 的前置**：把该界抬到 1M 时，cache 的 per-family max_tokens 与显存的换算（`src/cuda.rs` 的 pool 布局）——本表的 GLM 列待补（P0-C 的口径统一后一起做）。

---

## 4. P0-E — GDN chunk 对齐（代码级结论，2026-09-12）

**`prefill_chunk` 的结构**（`ferrite-exec/src/tp.rs:826`）:
```rust
pub fn prefill_chunk(&mut self, seq: u64, chunk_tokens: &[u32]) -> Result<()> {
    self.ensure_seq_all(seq, chunk_tokens);
    let h0 = self.shards[0].embed(chunk_tokens);          // ← 每 chunk 独立 embed
    let mut h = if mhc { hc_expand(&h0, hc_mult) } else { h0 };
    for plan in &plans { h = self.layer_forward_tp(seq, plan.layer_idx, h, chunk.len())?; }
    Ok(())
}
```
**关键观察**：
1. **`h` 不跨 chunk 携带**（每 chunk 的 hidden 从本 chunk 的 token 起）——**正确**：causal 的上下文全在 seq 的 KV state 里（DSA ring/compressor）与 GDN 的 recurrent state（conv + S）里，hidden 只是"当前 chunk 这一批 token 的激活"。
2. **GDN 的状态随 seq 累积**（`layer_forward_tp(seq, …)` 取 seq 的 state）——**chunk 边界天然连续**（state 是 seq 的属性）。
3. **conv1d（k=4）的边界**：chunk 首个 token 需要的 3 个前驱在 seq 的 conv state 里——**同样跨 chunk 连续**。

**⇒ 设计上 chunked prefill 的 GDN 语义正确**（无需 kernel 改动）。**P0-E 的剩余工作 = 数值验证**：同一 prompt 整段 prefill vs `FERRITE_PREFILL_BUDGET=512` 分块——**逐 token 的输出必须逐位一致**（若有差 → 定位 chunk 边界的 state 提交时序）。

---

## 5. P0-A — 1M 上限的界清单（读码建表，2026-09-12）

**必须同时抬的界**（漏一个 = 静默截断或 OOM）：

| # | 位置 | 当前值 | 1M 需要 | 备注 |
|---|---|---|---|---|
| 1 | `ferrite-serve/src/gpu_engine.rs:37` `MAX_CTX` | **8100** | 1_048_576 | 提交时的拒绝界（`prompt + max_new > MAX_CTX` → InvalidArg）。**必须与 #2 的 pool 同步**（否则放进来了但 pool 装不下）|
| 2 | `ferrite-kernel` 的 DSA cache 池 `max_tokens` per family | 由 pool 布局定（MAX_CTX 的注释自认是它的界）| 1M | **真正的物理界**；`ferrite-kernel/src/cuda.rs` 的 pool 尺寸 + `PoolMiss` 的类 |
| 3 | DSV41 `DSV41_MAX_POS` | **65536**（`chain_dev.rs:1230` 的 `unwrap_or`）| 1_048_576 | KV buffer 的分配界（`max_comp = max_pos/ratio`——见 P0-F 的 14GB/卡）|
| 4 | DSV41 `cfg.max_seq_len`（config.rs:39）| checkpoint 值 | ≥1M | `min(max_seq_len, DSV41_MAX_POS)` |
| 5 | **索引/位置的 dtype** | `i32`（pos_ctr/pos_rows/idxs）| 1M < 2^31 ✓ | **2^31 是硬界**（4M token 时 i32 溢出——1M 安全）|
| 6 | **锁步的位置一致性** | 各 rank 的 pos_ctr 独立推进（device 侧）| 同 | **P0-A 的隐藏风险**：1M 下任何一处"按 host 的 pos 分支"（如 indexer 的 host 镜像）在 chunk 边界漏一拍 = **静默截断**——**chunked prefill（P0-B）必须与 pos 的推进严格同流** |

**验证顺序**（P0-A 的准入）: P0-F（内存表 ✓）→ P0-C（KV 口径统一）→ **P0-A**（抬 #1+#2+#3+#4 后跑 32k/128k/512k/1M 的逐步长验证 + `DSV41_PREFILL_BUDGET=512` 的整段-vs-分块 parity）。
