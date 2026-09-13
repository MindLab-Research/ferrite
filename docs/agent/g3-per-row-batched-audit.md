# G3 — per-row launch 批量化：最优栈真实发数审计 + ring/window rows 版

> 工部 · 2026-09-13 · 只读勘察 + 一处新内核（ring/window rows）+ 一处**方案冲突上报**。
> 基线 HEAD `fa63dfa`（工作树干净，`git status` 空）。**未跑 GPU/e2e**；远端 `nvcc compile-only` 已跑。
> 依据：`docs/agent/sglang-verify-model.md` §4（路线图）、`docs/agent/verify-amortization-lesion-audit.md` §6/§10。

**回答的正是任务四问**：①最优栈真实发数（哪些 gate 生效/哪些 fallback）②ring+window rows 版 diff
③compress grid —— **方案冲突，见 §3，未改代码** ④cargo check + nvcc ⑤GPU 验证手册。

---

## 0. 三条必须先看的结论（含一条**上报尚书省**的方案冲突）

1. **`ATTN_MROWS` 在稳态是 fallback，不是生效**。它的硬前置 `pos_base + m - 1 < win`
   （`chain_dev.rs:12489`）在 `window_size = 128`、`m = 6` 下等价于 **`pos_base < 123`**——即
   **只有前 ~123 个位置生效**。ring 一旦翻转，整块的 block-wide sparse-attn 就会读到本块的未来行
   （audit defect #2），所以 gate 主动 decline。**§10.2 记的 `ATTN_MROWS −2.98ms` 必须标注"前 123 位口径"**
   ——稳态（生产）下这 −2.98 不成立。**这是本次审计最重要的发现。**
2. **`COMPRESSOR_MROWS` 对 `ratio == 1` 的层恒 decline**（`mrows_compress_ok` 要求 `ratio > 1`，
   `chain_dev.rs:13899`）。全模型只有 4 个 comp source（layer 2/8/14/20），其中 **layer 20 是 ratio 1
   ⇒ 永远走逐行 `compress_row`（2 发/行 = 12 发）**；只有 2/8/14（ratio 2）真被折叠成 1 发。
3. **⚠️ 方案冲突（`compress_rows_fused` 的 grid=1 → grid=(rows)）**：任务书写"每行一块、逐位等价（行独立）"，
   但**行不独立**——`compressor_fused_mrows_kernel` 有两个硬跨行依赖（ratio 2 时行 r 与 r+2 共用
   `state` 槽；commit 的 `*clen` 是严格串行计数）。grid=(rows) 会**静默损坏**（非性能差异）。
   **故未按原文实施**，见 §3 的完整举证与安全替代方案。

---

## 1. 最优栈真实发数表 vs 理论最小发数表（交付 ①）

### 1.1 gate 生效/fallback 判决表（file:line 证据）

最优栈 = A0（含 `HC_FRONT_ROWS=1`）+ 1b（`MROWS_ACT_CPASYNC=1`）+ `VERIFY_ROPE_MROWS=1` +
`ATTN_MROWS=1` + `SH_PAIR_M=1` + `GATE_MROWS=1` + `COMPRESSOR_MROWS=1` + `COMPRESSOR_PROJ_MROWS=1`
+ `INDEXER_MROWS=1` + `VERIFY_HEAD_MROWS=1`（`VERIFY_OROPE` 默认 ON）。

| gate | 逐行发数（关时） | 是否真跳过 | 读点 | 跳过点（file:line） | 会 fallback 的条件 |
|---|---|---|---|---|---|
| `SH_PAIR_M=1` | shared 专家 5 发/行 ×6 = **30** | ✅ **真跳过** | `:1887` | 早返回 `Ok(true)` `:15370` → `if sh_mrows_done { break }` `:15065` | `sh_il%32≠0` / `dim%32≠0` / `m>8` / 无 `sh_exp_fused` 符号 → 退 `SH_EXP_MROWS` 或逐行 |
| `GATE_MROWS=1` | gate 1 发/行 ×6 = **6** | ✅ **真跳过** | `:1676`(alias `:1684`) | `gemv_bf16_v2_mrows` `:14620` → `if !gate_folded` 循环 `:14630` 跳过 | `gemv_bf16_v2_wanted(n_routed)` / 符号缺 → 逐行 `gemv_bf16` |
| `VERIFY_ROPE_MROWS=1` | q-rope 1 发/行 ×6 = **6** | ✅ **真跳过** | `:1358` | `apply_rope_mrows` `:12210` → `if !q_roped` `:12223` 跳过 | 符号缺 / `!supports_apply_rope_mrows` → 逐行 |
| `ATTN_MROWS=1` | sparse 1 发/行 ×6 = **6** | ⚠️ **条件性（仅 pos<123）** | `:3081` | block launch `:12838`；`if !mrows_attn` `:12737` 跳过 | **`pos_base+m-1 >= win` → decline（稳态！）**；`world>1` 且无 `rp` 符号 |
| `COMPRESSOR_MROWS=1` | pool+commit 2 发/行 ×6 = **12** | ⚠️ **仅 ratio>1 层** | `:1170` | `compress_rows_fused` `:12385` → 逐行 `compress_row` 跳过 | **`ratio==1` 恒 decline（layer 20）**；`pos_base==0`；`m<2` |
| `COMPRESSOR_PROJ_MROWS=1` | proj 2 发/行 ×6 = **12** | ✅ **真跳过** | `:1494` | `gemv_f32_mrows` ×2 `:13729/:13740` | `n<2048` / `k%4` / 符号缺 → 逐行 `lin_f32_on` |
| `INDEXER_MROWS=1` | front 4 发/行 ×6 = **24** | ✅ **front 真跳过；select 仍逐行（设计如此）** | `:1397` | `indexer_rows_m` `:12441`；`if !q_done` `:13576` | 无 indexer 权重 / `m` 越界 |
| `VERIFY_HEAD_MROWS=1` | head 6 发 | ✅ **真跳过** | `:2124` | v1_mrows 单发 | 非 bf16 / 符号缺 |
| `VERIFY_OROPE`（默认 ON） | o-rope+quant 2 发/行 ×6 = **12** | ✅ **真跳过** | `:3023` | `sparse_attn_orope*` 融合；`o_oroped_all` `:12913` | launcher 侧 shape decline（all-or-nothing） |
| **`RING_WIN_FUSE=1`** | ring+window 2 发/行 ×6 = **12** | ❌ **不在最优栈（默认 OFF）** | `:3132` | ——（`verify_ring_win_fuse()` 严格 `=="1"`，`unwrap_or(false)`） | 未显式开即走逐行 2 发 |

### 1.2 发数账：真实 vs 理论最小（每层，m=6，只算 §1.1 涉及的族）

| 族 | 最优栈真实 | 理论最小 | 差距根因 |
|---|---|---|---|
| ring+window（owner 层） | **12**（gate 未开） | **1** | `RING_WIN_FUSE` 未进栈；rows 版（本次交付）在无翻转窗口→1 |
| sparse attn | **6（稳态）/1（前 123 位）** | 1 | 翻转后 `ATTN_MROWS` 必须 decline（正确性） |
| compressor pool+commit | **12（layer 20）/1（layer 2/8/14）** | 1 | `ratio==1` 层恒 decline |
| compressor proj | 2（1 层 source） | 2 | 已最小 |
| shared expert | **2**（`quant_rows` + `sh_exp_fused`） | 2 | 已最小（SH_PAIR_M 生效） |
| gate | **1** | 1 | 已最小 |
| indexer front | **4**（idx source 层） | 4 | 已最小 |
| indexer select | **~5/行 ×6 = 30** | ~5/行 | 设计如此（publish 4 + topk 1，逐行是正确性要求） |
| q/kv rope | **1 + 2 = 3** | 3 | 已最小（q 折叠；kv 的 B4 未开） |
| wo_b | **6（quant）+1 = 7** | 1 | `VERIFY_WOB_MROWS_F32`（B6）未开（且数值红线，§10 判过） |

**判读**：任务书以为的"已开的五个 gate 都在最优栈里生效"**只对 3 个成立**
（`SH_PAIR_M` / `GATE_MROWS` / `VERIFY_ROPE_MROWS` / `COMPRESSOR_PROJ_MROWS` / `VERIFY_HEAD_MROWS` / `VERIFY_OROPE`
真生效）；`ATTN_MROWS`（稳态）与 `COMPRESSOR_MROWS`（ratio-1 层）**是 fallback**。
**per-row 残量的大头不在这些 gate，而在 `RING_WIN_FUSE` 未开（12）与 indexer select 的逐行（30，设计使然）。**

---

## 2. ring+window 的 rows 版（交付 ②）

### 2.1 关键事实：为什么"块宽 ring 融合"历史上被 revert

`verify_ring_win_kernel`（`dsv41_glue.cu:2059`）先 append 整块、再算 per-row 窗口，**ring 翻转后**
行 r 的窗口覆盖**全部 win 个槽**，任何后续行的 append 都会覆盖行 r 还需要的槽 —— 这就是 audit defect #2。
**唯一安全的窗口**：块不翻转（`pos_base + m - 1 < win`），此时行 r 的窗口只覆盖 `[0, pos_r]`（槽 `{0..pos_r}`，
与后续 append 的槽不相交）。**该前置与 `ATTN_MROWS` 的 `:12489` 完全同款** —— 两者在同一 regime 生效。

### 2.2 交付的改动（`DSV41_RING_WIN_MROWS=1`，默认 OFF）

**新内核**（`kernels/cuda/dsv41_glue.cu`，紧跟 `dsv41_ring_win_fuse_ph`）：

```c
__global__ void ring_win_fused_mrows_kernel(float* ring, const float* kv, const int* pos_ctr,
                                            int window, int hd, int m, int idx_stride, int32_t* idxs) {
    const int base = *pos_ctr;
    const int total = m * hd;
    for (int e = threadIdx.x + blockIdx.x*blockDim.x; e < total; e += gridDim.x*blockDim.x) {
        const int j = e / hd, i = e % hd;
        ring[(size_t)((base + j) % window) * hd + i] = kv[e];
    }
    if (blockIdx.x != 0) return;                       // idx 半在单块内
    for (int e = threadIdx.x; e < m * window; e += blockDim.x) {
        const int r = e / window, c = e % window;
        const int start_pos = base + r;                // == per-row 调用的 pos_rows[r]
        int idx; /* … decode 分支，逐项等于 ring_win_fused_kernel … */
        idxs[(size_t)r * idx_stride + c] = idx;        // ← 新参数：per-row 调用用 idxs_r + r*ist
    }
}
extern "C" int dsv41_ring_win_fuse_mrows(...);
```

与既有 `dsv41_verify_ring_win` 的**唯一差别**是 `idx_stride`：后者写 `idxs[r*window+c]`
（行距 = window），而 verify 的 `idxs_r` 行距是 `ist = win + index_topk`
（`sparse_attn` 按 `idxs_r + r*ist` 读）。传 `ist` 即逐字节复现 per-row 的 `idxs_r + r*ist` 基址。
- 每行 append/index 算术与 `ring_win_fused_kernel`（eager 的 1 发版）**逐项相同**（同一 `slot`、同一 decode 分支）。

**Rust 侧**：
- `device.rs`：新可选字段 `ring_win_fuse_mrows`（`ko!(rt, "dsv41_ring_win_fuse_mrows")`）+ 方法
  `ring_win_fuse_mrows(ring, kv, pos_rows, window, hd, m, idxs, idx_stride) -> Result<bool>`。
  旧 `.so` 无符号 → `Ok(false)` → 保留逐行路径。
- `chain_dev.rs`：
  - 新 gate `ring_win_mrows()`（`:3155`，严格 `=="1"`）。
  - `attention_rows`：在 per-row 循环**之前**加 hoist（`:12563`），前置 `owns_kv && m>1 &&
    ring_win_mrows() && pos_base+m-1 < win`；跳过点 `if owns_kv && !rw_mrows`（`:12603`）。

**发数**：owner 层 ring+window **12（或 6）→ 1**。

**诚实的价值评估**：因前置是"无翻转"，此臂**只在启动前 ~123 位（@win=128）生效**，稳态无用。
**稳态的真赢是 `DSV41_RING_WIN_FUSE=1`（零新代码，12 → 6）**——两者互补：
`RING_WIN_FUSE` 管稳态，`RING_WIN_MROWS` 管冷启动，且后者与 `ATTN_MROWS` 同 regime、可同开。

---

## 3. ⚠️ 方案冲突上报：`compress_rows_fused` 的 grid=1 → grid=(rows)（交付 ③）

**任务书原文**："`compress_rows_fused`（`compress_rows_fused grid=1 单块串行）… 改 grid=(rows)（每行一块——kernel 内的 `for r` 循环改 blockIdx 维度）——**逐位等价（行独立）**"。

**核验结论：行不独立，按原文实施会静默损坏。未改。** 证据：

`kernels/cuda/dsv41_kernels.cu:3408-3444`（kernel 头注释）+ `:3469-3556`（本体）明列两条**硬跨行依赖**：

1. **state 槽共享**：`slot = (start_pos + r) % ratio`（`:3471`）。`compress_ratios = [0,0,2×18,1×20,…]`
   ⇒ ratio 2 的层上**行 r 与 r+2 写同一槽**（`state_kv[dst]` `:3477`）。行 r+2 覆盖的行 0 组槽，正是
   行 r 的 pool 阶段（`:3486-3504`）要读的 —— grid 化后两块的执行顺序不定 ⇒ pool 会混两个组。
2. **commit 计数串行**：`len = *clen; … *clen = len + 1`（`:3529/:3549`），ring 目标行
   `window + len`（`:3534`）也是**逐行推进的计数器**。grid 化后多个块并发读写 `*clen` ⇒ 非确定。

**安全替代（供尚书省裁决，均需新立项，非本任务范围）**：
- **保序多块**：仍是串行语义，只能靠更大 `blockDim` 摊开 `hd` 维——但会改变 RMSNorm 归约树
  （`s_red[warp]` `:3510-3514`），**破坏逐位等价**，需重证。
- **拆两核**：stage-1 state carry（若 `ratio >= m` 则**每行槽互异**、可 grid 化）+ pool/commit（串行）。
  生产 `ratio = 2 < m = 6` ⇒ **stage-1 也不可 grid 化**，此路在生产形状下收益≈0。
- **真解**：把 pool/commit 换成"按 (row, channel-tile) 并行的无跨行耦合写法"需先重构 state 语义——
  属 G2/tensor-core 路线，不是 G3 的 wiring。

**即：`compress_rows_fused` 的 grid=1 是**正确性约束的结果，不是"最坏形态"的实现偷懒**。**
本次**不改代码**；建议在 per-layer-census 清单里把该项从"可 grid 化"改判为"跨行耦合，保留 grid=1"。

---

## 4. 推荐 gate 集 diff（给主 agent）

最优栈之上**还需要的**（本次审计新增/纠偏）：

```diff
  COMMON="DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 \
          DSV41_BF16_TRUNCATE=1 DSV41_SWALLOW_STEP=1 DSV41_SWALLOW_EPOCH_PAD=1 \
          DSV41_SH_EXP_MROWS=1 DSV41_SH_PAIR_M=1 \
          DSV41_GATE_MROWS=1 DSV41_VERIFY_HEAD_MROWS=1 \
          DSV41_INDEXER_MROWS=1 DSV41_COMPRESSOR_MROWS=1 \
          DSV41_VERIFY_ROPE_MROWS=1 DSV41_ATTN_MROWS=1 \
          DSV41_COMPRESSOR_PROJ_MROWS=1 DSV41_MROWS_ACT_CPASYNC=1"
+# 零新代码、稳态有效（本次审计第一优先）
+DSV41_RING_WIN_FUSE=1
+# 本次交付（冷启动期 effective；与 ATTN_MROWS 同 regime，可叠加）
+DSV41_RING_WIN_MROWS=1
```

**不要开的**（§10 已判死，勿重踩）：`VERIFY_WOB_MROWS_F32`（B6，数值红线）、`MROWS_MPAR`（二连败）、
`ATTN_LIN_FUSE`（`m==1` 专属，SWALLOW 空门）。

**纠偏**：`ATTN_MROWS` / `COMPRESSOR_MROWS` 的票面在稳态**不成立**，见 §1，别再按票面编预算。

---

## 5. 编译/测试证据（交付 ④）

- `cargo check -p ferrite-models`：**OK**（6 条既存 warning，与本改动无关）。
- `cargo check --workspace --all-targets`：**OK**。
- `cargo test -p ferrite-models --lib`：**92 passed / 0 failed**（2 ignored，既存）。
- 远端 `nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 -c dsv41_glue.cu`：
  **COMPILE_OK**，符号已生成：
  `T dsv41_ring_win_fuse_mrows` + `T ring_win_fused_mrows_kernel(float*, float const*, int const*, int, int, int, int, int*)`。

---

## 6. GPU 验证手册（交付 ⑤）

> 双门禁：每臂同时报 `step_ms`（`[dspark] steps=`）**AND** `mean-k`（A0 基线 1.34 / Fix-A 后 2.240）。
> **一臂一进程**（`OnceLock` 每进程读一次）；`/proc/<pid>/environ | grep DSV41_` 逐门回读（§10.1 幻影门纪律）。

### 6.1 本轮三臂（同栈、交错 A/B/A/B）

| 臂 | env 增量 | 期望 | 判据 |
|---|---|---|---|
| **R1** 稳态 ring 融合 | `DSV41_RING_WIN_FUSE=1` | ring+window owner 层 12→6 | verify 位移 + launch 计数（nsys `ring_win_fused_kernel` 实例 ×2→×1）；mean-k 不变 |
| **R2** rows 版（冷启动） | `+DSV41_RING_WIN_MROWS=1` | 前 123 位 6→1 | nsys 见 `ring_win_fused_mrows_kernel`；`_verify_ring_win`/逐行 `ring_append` 归零；mean-k 不变 |
| **R3** 前置校验（反证） | R2 + 强制 `pos>=win` 的 prompt | rows 臂应 **decline**（无 mrows kernel） | 证明前置真的挡住翻转期（正确性护栏） |

### 6.2 证据采集（禁止吞吐反推；per-kernel 用 nsys，死锁规避 AR_SAFE）

```bash
# 同栈基线（R0）与 R1/R2 背靠背；每臂重建双产物（build.sh 103a + cargo build --release）
DSV41_TIMING=1 DSV41_DSPARK_DEBUG=1 bash scripts/batched_400_v2.sh   # ← 基线矩阵
DSV41_RING_WIN_FUSE=1            bash scripts/batched_400_v2.sh      # R1
DSV41_RING_WIN_FUSE=1 DSV41_RING_WIN_MROWS=1 bash scripts/batched_400_v2.sh  # R2
# nsys 轮（只看 kernel 相对倍数）：AR_SAFE 模式，避免 side-stream 前置 decline
env -u FERRITE_P2P NCCL_NVLS_ENABLE=0 DSV41_AR_V5=0 DSV41_GRAPH_STEP=0 ~/nsys_dual.sh
```

### 6.3 红线（任一破即弃该臂）

1. 出师表 1000 token **逐字 + 零拉丁**（`BF16_TRUNCATE=1` 红线）；
2. `[dspark] mean-k` 逐位/Z_不变（掉出 2.240±噪声 ⇒ 数值回归）；
3. `faults=0`、无 `MISMATCH`；
4. `/proc/<pid>/environ` 回读确认 `DSV41_RING_WIN_*` 真进进程；
5. `nm -D libferrite_kernels.so | grep ring_win_fuse_mrows` 有符号（否则 rows 臂静默 decline）。

### 6.4 预期判读

- R1：verify **−0.2~0.5ms**（ring+window 在 nsys 占比小；主要是 launch/图节点数），mean-k 不变。
- R2：冷启动段额外 −0.x ms；**稳态与 R1 等价**（rows 臂 decline）。若 R2 在稳态仍见
  `ring_win_fused_mrows_kernel` ⇒ **前置失效，立即停**（正确性事故）。
- **战略提醒**：G3 的 per-row 残量大头（indexer select 30、wo_b 7）**不在本臂覆盖范围**；
  ring+window 只是"最快兑现"的一格。真正的 4×→1.3× 机制仍是 G2（M 进 GEMM tile）。

---

## 附：改动清单

| 文件 | 改动 |
|---|---|
| `kernels/cuda/dsv41_glue.cu` | +`ring_win_fused_mrows_kernel` + `dsv41_ring_win_fuse_mrows`（+69 行） |
| `crates/ferrite-models/src/dsv41/device.rs` | +`ring_win_fuse_mrows` 字段/注册/方法（+50 行） |
| `crates/ferrite-models/src/dsv41/chain_dev.rs` | +gate `ring_win_mrows()`；`attention_rows` hoist + 跳过点（+55 行） |
| **`compress_rows_fused` grid** | **未改（方案冲突，§3 上报）** |
