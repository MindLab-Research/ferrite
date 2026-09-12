# batched（m=6）到 12.5ms 的需求清单 + "structurally slower" 根因 + m=6 权重共享真实性判定

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 代码基线：工作树 HEAD（`kernels/cuda/dsv41_kernels.cu` / `dsv41_experts_mxf4.cu` /
> `crates/ferrite-models/src/dsv41/chain_dev.rs`）。
> 输入账本：`dspark-correctness-chain.md`（12.5ms 的原始出处 :3111-3114）· `verify-architecture-floor.md` ·
> `verify-marginal-cost.md` · `verify-ms-breakdown.md` · `sh-pair-template-m-design.md` ·
> `b6-mrows-f32-design.md` · `swallow-unlocked-next-plan.md` · `batched-400-v2-*`。
> **口径纪律**：每个 ms 都标注「实测 / 设计口径 / launch 账」；推算项显式标注。

---

## 0. 判决（先读五条）

1. **12.5ms 不是一个"多翻几个 flag"的目标——它就是 L5 架构地板本身。**
   `verify ≈ 8ms（L5 地板）+ draft 4.3 + commit 0.2 = 12.5ms`。L5（8~9ms，`verify-architecture-floor §3`）
   要求 **tcgen05 + 族级融合（N÷5）+ L4 占用/MLP + L5 cp.async 流水** 四件套全部兑现，
   其中 L4/L5（16~21 人日 + 8~12 人日）**仓内零实测背书**（v17→v21 四变体全中性）。

2. **"权重读一次"在 byte 层与 launch 层是真的，但它不是一个加速性质——因为这一族 kernel 根本不吃字节。**
   除 head（5.9TB/s、77% 峰值、只占 1.12ms/37ms）外，**全部族跑在 HBM 峰值的 0.7~4.9%**
   （shared expert 885MB/10.4ms = **85GB/s** = 1.1% 峰值）。省 5/6 的权重字节 =
   省"已经跑在 1% 峰值的那部分时间" ≈ 0（`verify-marginal-cost §4 机理1`）。

3. **mrows 折掉的是「权重解码」，不是「激活解码 + FMA」。**
   指令数 `(2 + 3M) / (5M)`：M=5 ⇒ 0.68×，M→∞ ⇒ 0.60×。**永远拿不到 1/M = 0.17×**（`verify-marginal-cost §4 机理2`）。

4. **m=6 下 warp 数不增、寄存器/SMEM 增**（`gemm_fp8_mrows_kernel`：一个 warp 一行输出，
   `row = blockIdx.x*nwarps + warp`；M 只进 `acc[M]` 累加器与激活 smem）⇒ **占用率随 M 下降**，
   而「3~4 warps/SM 的延迟暴露」正是两堵主墙之一。**这是 "structurally slower" 的核心机制。**

5. **唯一真正给 batched 加并行度的是 MoE 的 `grid.z = rows`**（CTA 数 ∝ rows，`dsv41_experts_mxf4.cu:2741`），
   而 routed experts 是设计上「天然 5×、不可压」的族，只能靠 tcgen05（且 down 无 tcgen05 核）。

---

## 1. m=6 权重共享的真实性判定（任务的关键问题）

### 1.1 现场事实（读码，非估计）

`gemm_fp8_mrows_kernel`（`dsv41_kernels.cu:5208-5368`）：

```cpp
const int row = blockIdx.x * nwarps + warp;      // ★ 一个 warp 拥有一个输出行
const uint8_t* wr = w + (size_t)row * (size_t)k; // 该行的权重行
dsv41_cp_async16(row_s, wr);                     // 该行权重 staged 一次
...
#pragma unroll 32
for (int kb = 0; kb < nb_k; ++kb) {
    const float wv = s_lut[s_w[warp*k + j]] * sb;          // C4: 权重解码 · 1 次
    #pragma unroll
    for (int r = 0; r < M; ++r) {
        const float av = s_lut[s_a[r*k + j]] * s_as[...];  // 激活解码 · M 次
        acc[r] += av * wv;                                 // FMA      · M 次
    }
}
```

- **权重行的 warp 分配只看 `n`**（grid = `ceil(n/nwarps)`），**与 M 完全无关**。
- **M-fold 只在 consume 内层**：同一 `(output row, kb)` 的 `wv` 解码一次，喂 M 条独立累加链 `acc[r]`。
- **warp 数 = n（每行一个 warp），与 M 无关。** M 增只加 `acc[M]` 寄存器 + `m*k` 激活 smem。

### 1.2 判定：分三个域，两真一假

| 域 | batched m=6 vs lazy 6×m=1 | 真实性 | 是否省时 |
|---|---|---|---|
| **权重字节** | 1× vs 6× | ✅ **真共享** | ❌ 该族跑 0.7~4.9% 峰值 ⇒ 省 ≈ 0 |
| **权重解码指令** | `2+3M` vs `5M`（≈0.60~0.68×） | ⚠️ **部分共享** | 仅 ~1/3 指令节省，且被占用率下降吃掉 |
| **并行度（warp 数）** | **相同（都 = n）** | ❌ **不共享** | ❌ M 不加 warp；smem/regs ↑ ⇒ 占用 ↓ |
| **launch 数** | 1 vs 6（每族） | ✅ 真共享 | 只在「per-step 族」上兑现（§2） |

**⇒ 结论：m=6 的"权重共享"是 byte 层与 launch 层的共享，不是 M 倍的加速。**
**"一个 warp 一行"这个映射决定了：M 只让每个 warp 多干活，不让机器多并行。**
**在延迟受限（3~4 warps/SM）的架构里，"少而重"的 warp 比"多而轻"的 warp 慢——这就是 m=6 比 6×m=1 慢的机制根源。**

### 1.3 现场反证（实测，全部支持上面的判定）

| 反证 | 数值 | 来源 |
|---|---|---|
| verify(m=5) vs 5×EAGER | **37.31 > 5 × 6.15 = 30.75** ⇒ folding 后的 batched **比 5 次独立单行 forward 还慢 25%** | `verify-marginal-cost §0` |
| 投影族（mrows 已生效） | 只到 **1.42× EAGER**，不是"5 行只贵 10%" | `verify-marginal-cost §5` |
| hc 链（已是 rows=m 原生） | **53GB/s = 0.07% 峰值**（全表最低） | `verify-ms-breakdown §1` |
| `SH_EXP_MROWS` | **两次实测零收益**（instruction-bound） | `verify-ms-breakdown §修正` |
| `{SH_EXP+GRAPH+ROPE+P3A}` 全开 | 实测 **−1.21ms**（预期 −24） | 同上 |

---

## 2. batched 的 launch 结构（"一次 forward 6 行" vs lazy"6 次 forward"）

### 2.1 关键不对称：per-step 族 vs per-row 族

| 族 | 性质 | batched(m=6) | lazy(k_emit≈2 行) | 可 fold? |
|---|---|---|---|---|
| hc 链 | **per-step**（rows=m 原生） | 400 发/步（6 行共享） | 400 × k_emit | ✅ 一次 |
| AR v5 | **per-step**（协议地板，m 无关） | 80 轮 × 2 核 = 160 | 160 × k_emit | ✅ 一次 |
| norm / rope / quant | per-row 但可 fold | 1 发/族 | × k_emit | ✅ |
| **attention append/window/select** | **per-row，因果强制** | **× 6（不可 fold）** | × k_emit | ❌ |
| indexer / compressor | per-row | × 6 | × k_emit | ⚠️ |
| **routed experts** | **per-row，天然 5×** | × 6（grid.z=rows） | × k_emit | ❌ |
| 投影 / MoE gate / shared / head | 可 fold | 1 发/族 | × k_emit（m=1 fold 退化） | ✅ |

**⇒ batched 的结构性优势 = "per-step 族只付一次" + "可 fold 族付一次"；**
**结构性劣势 = "per-row 因果族（attn/indexer/compressor）与 routed 仍 ×6"。**

### 2.2 launch 数（口径对齐）

- **现状 batched（SWALLOW 无图）**：~6224（m=5 口径）/ ~7000（m=6），几乎全是 GPU 执行时间
  （图化实测只 **−1.5ms** ⇒ submit 已被 CUDA async launch 隐藏，`verify-ms-breakdown §修正`）。
- **折叠后目标**：~1300（族级融合 N÷5）；任务口径的 ~800 是更激进的折法。
- **lazy**：6 × ~800 = **~4800**（每行一个完整 forward，per-step 族被重复付 6 次）。
- **⇒ 任务前提"batched ~800 vs lazy ~4800"成立于"batched 已折到 ~800"这一假设上。
  当前 batched 是 6224~7000，不是 800——折本身就是那 12~15 人日的族级融合。**

---

## 3. 到 12.5ms 的差距分析

### 3.1 缺口算式

```
目标：verify ≤ 8ms（12.5 − draft 4.3 − commit 0.2）
现状：verify(m=6) ≈ 37~39ms（SWALLOW 无图）；serve 墙钟步时 ~40ms（用户已两次判不可信）
缺口：−29 ~ −31ms
```

### 3.2 逐项叠加（沿用任务给的账，已按现场修正）

| # | 项 | 设计口径 | 实测/修正 | 状态 |
|---|---|---|---|---|
| B0 | batched 现状（SWALLOW 无图） | — | **~40ms（serve）/ verify ~37-39** | 实测 |
| B1 | + Plan B（图化） | −3~4 | ✅ **实测只 −1.5ms**（非 −15） | 已实施未测 |
| B2 | + SH_PAIR `template<M=6>` | −4.9~7.9 | ✅ **parity 根因已修**（哨兵 `0x5A→0x7F`，kernel 无 bug；见 `dspark-correctness-chain.md` §SH_PAIR Parity） | 待上 GPU 复跑 parity |
| B3 | + mrows 族（GATE/INDEXER/ROPE/HEAD/NORM/COMPRESSOR） | −4.5~5.8 | ❌ **实测 ≈0**（两次零收益） | gate 已就位 |
| B4 | + tcgen05 e4m3 grouped | −2 | ⚠️ **修正 −1.0~3.8**（down 无核） | 从未上 GPU |
| — | 小计（B1-B4） | −15 ~ −20 | 60% 兑现 ⇒ **−9 ~ −12** | |

```
40 − (9~12) = 28~31ms  （任务账）
40 − (15~20) = 20~25ms （设计口径全兑现）
```

**⇒ 两种口径下都到不了 12.5ms：**
- 任务账（只算 B1-B4）：**还差 ~16~19ms**
- 设计口径全兑现（再加 hc −1.3~1.7 / 投影 R1-R3 −1.25~2.8 / B6 −0.66~1.5）：
  **还差 ~9~12ms**（与任务结论一致）

### 3.3 那剩下的 9~12ms 在哪里（L3/L4/L5）

| 层 | 内容 | Δms | 成本 | 仓内实测背书 |
|---|---|---|---:|---|
| L3 | 族级融合（N÷5：6224→~1300） | −8~15 | 12~15 人日 | 部分（SH_PAIR 是其中最重一块） |
| **L4** | **占用/MLP：行进 grid（×M warps，非寄存器）+ K-split(KStep=64/NSTAGE=16) + TMA 深度** | **−5~8** | 5~8 人日 | **零**（v17→v21 四变体全中性） |
| **L5** | **kernel 内 cp.async 流水 + 满 wave + smem 阶段交接** | **−2~3** | 8~12 人日 | 零 |

**⇒ 12.5ms = L5 地板。它等价于要求"整个 backbone 的逐层 GEMV 有效带宽从 373GB/s（4.9% 峰值）翻倍"，
而 `NEXT-SESSION-HANDOVER §1.3` 明确判其为"无路径"（gemv 减半无解）。**

> 12.5ms 的 480 tok/s 假设（`dspark-correctness-chain.md:3111-3114`）原文是：
> 「6 行共享一次权重读（weight-stationary）→ 一次 forward ≈ EAGER + 边际 ≈ **7-8ms**」。
> **§1 的判定直接否证这个假设**：m=6 不共享 warp、激活侧 ∝M、字节不是约束 ⇒
> "一次 forward ≈ EAGER + 边际"要求激活侧几乎免费，而实测每行边际 = **1.31 × EAGER**（`verify-marginal-cost §0`）。

---

## 4. batched 路径到 12.5ms 的具体需求清单

### A. 零/低成本（已就位 gate，只差 A/B 或默认值）

1. `SH_PAIR_M=1 (+ SH_PAIR_M_FOLD=1)` —— **前置：parity 100%**（prod/m=6/nolimit 的两项已定名为
   **测试口径问题、kernel 无 bug**：phase-1 哨兵 `0x5A` 可产出 → 已改 `0x7F`；另一"失败"是
   `SH_CHECK ++g_fails` 与 `main += sh_case()` 的 double-count。**待上 GPU 复跑确认**；
   `dspark-correctness-chain.md` §SH_PAIR Parity）；`template<M=6>` 是 batched production 形状。
2. `GATE_MROWS` / `INDEXER_MROWS`(front) / `VERIFY_ROPE_MROWS` / `VERIFY_HEAD_MROWS`（**最后单独上**，
   历史 ar5-hang 组合）/ `NORM_MROWS` / `COMPRESSOR_MROWS`。
3. `HC_VERIFY_FUSE=1` + `HC_FRONT_ROWS=1`（**前置**：`chain_dev.rs::hc_mixes_auto` 的 verify 调用点
   `bf16_truncate()` → `false` —— A2 的 truncate 坑，与 A1-a 同构的一行修复）。
4. `VERIFY_GRAPH=1`（Plan B 后；**定位是顺序/正确性工具，−1.5ms**，不是性能项）。
5. `SWALLOW_STEP=1` + `SIDS_WRITEBACK=1`（结构性前提，硬依赖）。

### B. 中成本（新核 / 新接线）

6. B6 `dsv41_gemm_fp8_mrows_f32`（B 类公共祖先）→ B1–B5（−0.66~1.5ms B6 单项；全族 −2.8~4.9ms）。
7. 投影 R1（wo_a cp.async16）+ R2（m==1 复用 lin2+lin_rope_norm）+ R3（WO_PAIR）（−1.25~2.8ms）。
8. attention `b·m` 单发（**前置**：per-row `clen` 设备快照 + ring/window 的 r 升序合核）
   + indexer select 的 `out_stride` / `n_pos=max(lens)`。

### C. 高成本（换核范式 —— **12.5ms 的必要条件**）

9. **tcgen05 e4m3 grouped**（routed gate/up；down 无 tcgen05 核）—— 前置：e4m3 臂 GPU parity +
   4-gate 联合（`EXPERT_GROUPED=1` + `EXPERT_TCGEN05_E4M3=1` + `GATEUP_FUSE=0` + `EXPERT_ILV=0`）。
10. **L4 占用/MLP**：`行进 grid（×M warps，非寄存器）+ K-split(KStep=64/NSTAGE=16) + TMA 深度`
    —— **三者必须同时上**（v19/v21/v24 已证"只动一个因子无效"）。
11. **L5 kernel 内流水**：cp.async + 满 wave + smem 阶段交接。

**⇒ 12.5ms 的必要条件 = 9 + 10 + 11 全部兑现。**
**⇒ 1–8 只是"把 37 压到 20~25"的部分；即使 B2/B3/B4 的设计口径全部足额兑现，
落点也只是 verify ~20~24ms ⇒ 步时 ~24~28ms ⇒ ~210~250 tok/s（@accept 5）。**

---

## 5. "structurally slower" 的根因（逐条物理机制）

### 根因 1 — 架构不吃字节（权重共享省的是不重要的那部分）
除 head 外全族 0.7~4.9% 峰值。**mrows 省 5/6 权重字节 ⇒ 省的是 1% 峰值的时间 ≈ 0。**
反证：head 是唯一跑满带宽的族（5.9TB/s），而它只占 1.12ms/37ms = 3%。
**凡是字节有意义的族都不重要；凡是要紧的族字节都无意义。**

### 根因 2 — mrows 折权重解码、不折激活解码 + FMA
`per (row,kb)`：折叠后 `2 + 3M`，逐行 `5M` ⇒ M=6 时 `20/30 = 0.67×`。
**激活侧 3/5 的指令仍 ∝ M。** 且 smem 随 M 增长（`nwarps*k + 256f + M*nb_k*4 + M*k`）、
`acc[M]` 寄存器随 M 增长 ⇒ **占用率随 M 下降（二次惩罚）**。

### 根因 3 — warp 数不随 M 增长（"一个 warp 一行"）+ 占用率随 M 下降
- mrows：warp 数 = `n`（与 M 无关）；M 只加寄存器 + smem ⇒ 占用 ↓。
- 在飞 warp 需求 **4.6MB > 单呼叫权重 4.5MB** ⇒ **M=1 结构上打不满 MLP**（`STATUS:7943`）；
  而 mrows **不加 warp**，只把 6 行的活压给同一批 warp。
- ⇒ **batched 把 6 行"串行化"进一次 launch，lazy 把 6 行"并行化"成 6 次 launch（每次各占满 SM）。**
  延迟受限时后者更快 —— **这解释了实测 37.31 > 5 × 6.15（folding 后反而慢 25%）。**

### 根因 4 — 8/11 族没折到 1×（配置层）
- 3 族 flag 默认 OFF：shared(10.4) / gate(3.44) / head(1.12)
- 4 族无 mrows 路径：attn(2.8) / indexer(2.5) / compressor(0.55) / ring-window
- 1 族设计不可压：routed(8.3)
- **⇒ 即使全折，也因根因 1/2 拿不到 1/M。**

### 根因 5 — batched 的激活侧 ×6（lazy ×~2.2）
同一 accept 下 batched 恒跑 6 行激活，lazy 只跑 `k_emit ≈ 2` 行 ⇒ **batched 做 ~3× 的激活工作**。
⇒ `lazy iff (1+mean_k) < B/c`，阈值 `mean_k ≈ 3.55`：
**accept < 3.55 时 batched 结构性净亏**（对话 0.96 / 出师表 1.21 ⇒ 应走 lazy）。

### 逐 kernel 判定：m=6 下谁比 6×m=1 慢？

| kernel 族 | m=6 vs 6×m=1 | 机理 |
|---|---|---|
| 所有「mrows 折、但 M 不进 grid」族（投影/CATE/HEAD/hc） | **更慢或中性** | warp 数不增、占用随 M 降、激活侧 ×6 |
| attention append/window/select | **更慢** | 因果强制逐行（`read(r)→append(r)→read(r+1)`），M 不给并行、只加依赖链长度 |
| indexer / compressor | 更慢 | 逐行 + `indexer_topk` 单核地板（21GB/s） |
| routed experts | 中性（`grid.z=rows` 真加并行，但字节天然 ×5） | 唯一真正的 M 并行化 |
| **SH_PAIR / 三段一核** | **更快**（launch 1000→80/步） | **唯一把 M 进 grid 的折法**（phase-1 `M×9` block；phase-2 `acc[M]` 不损并行） |

**⇒ "structurally slower" 不是所有 kernel，而是"用 mrows 折、但没把 M 进 grid"的那一大类。**
**SH_PAIR 之所以是最大单项（−4.9~7.9ms），正因为它是唯一按设计 §2.3 把 M 进 grid 的折法
（phase-1 并行度 = `ceil(n1/32) × M`，M=6 → 54 block；而不是折进 warp 只剩 9 block）。**

---

## 6. 一句话交付

> **12.5ms = L5 架构地板**，需要 `tcgen05 + 族级融合 + L4 占用/MLP + L5 流水` 四件套（25~35 人日，
> L4/L5 仓内零实测背书）。
> **"权重读一次"在 byte/launch 层真实、在加速层不成立**：m=6 不增 warp（warp 数 = n），
> 占用反降，而这一族 kernel 根本不吃字节（0.7~4.9% 峰值）。
> **structurally slower 的机制** = 一个 warp 一行（M 不加并行）+ 激活侧 ∝M（指令下限 0.6~0.68×）
> + 延迟受限（3~4 warps/SM）。
> **现实落点**：全部已就位 gate（含 SH_PAIR/mrows/tcgen05 设计口径全兑现）⇒ verify ~20~25ms ⇒
> 步时 ~25~30ms ⇒ **~200~250 tok/s（@accept 5）**；要 400@accept5（步时 ≤15ms）也仍需 L4。

---

*工部 · 只读分析 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*所有 ms/launch 数均标注来源与口径（实测 / 设计口径 / launch 账）；与任务前提冲突处
（"m=6 权重共享 = 一次 forward ≈ EAGER + 边际 7-8ms"、"batched ~800 launch"、"图化 −3~4ms"）
已显式修正并给出依据。*
