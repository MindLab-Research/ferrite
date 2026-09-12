# verify 的架构地板分析（工部 · 架构分析）

> 任务：当前架构下 verify 的理论最低步时；400 需要什么架构变化。
> 方式：**只读**——读 `chain_dev.rs` / `dsv41_experts_mxf4.cu` / `ferrite_kernels.cu` / `build.sh`
> + 仓库内全部实测与账本（`verify-ms-breakdown` / `verify-calc-floor` / `verify-family-fusion` /
> `verify-marginal-cost` / `batched-400-v2-*` / `swallow-step-400-necessity` / `STATUS.md`）。
> 未执行 GPU 命令、未改动任何源码；本文件为唯一产出。凡推算项均标注。
> 基线：`DSV41_TIMING` verify(m=5) = **37.31ms / 6224 launches / 14.09GB**，HEAD `034e15c`。

---

## 0. 判决（先读七条）

1. **任务前提里的 `37.31 − 13~14 = 23~24ms` 是「只翻 flag + 换一颗核」的落点，不是架构地板。**
   它是**乐观**的（mrows 的 −3~−4ms 与实测 −1.21ms 冲突），同时也是**悲观**的（完全没有族级融合）。
2. **当前架构的真地板 ≈ 20~22ms**（族级融合 N÷5 + tcgen05，但保留今天的核效率）；
   **越过它必须换核范式**（tensor-core + 满 wave + kernel 内流水）→ ~8~9ms。
3. **地板的台阶不是线性的，是四级不连续**：37.3 → ~25（换核）→ ~21（融合）→ ~11~14（占用）→ ~8~9（流水/满 wave）。
   每一级换的是**不同的物理量**（L1TEX μop 数 / kernel 数 / 在飞 warp 数 / 消除 latency 暴露）。
4. **带宽从来不是约束**（折叠后 0.74ms = 目标的 8%），**launch submit 也不是**（图化只 −1.5ms）。
   约束是 **「tiny kernel 的 μop 数 + 3~4 warps/SM 的延迟暴露」**。这是"多而小的 SIMT kernel"这一设计点的必然代价。
5. **400 被两个独立缺口夹住**：`verify ≤9ms`（需 L5，无人开始）与 `accept ≥3`（实测 1.214）。
   **在当前 accept 下，400 甚至不需要优化——它需要一个 5.5ms/步的机器，那在物理上不存在。**
6. **lazy 今天比 batched 快（92 vs ~55 tok/s），而这不是 lazy 好**——是 accept 低到 lazy 只跑 2 行。
   accept 一旦升到 3+，lazy 必然要跑 4 行 × 6.15ms，**batched 的反转才会发生**。
   ⇒ **整个 400 计划成立的前提是 accept 先涨**，步时优化是第二位的。
7. **sglang 的 13ms 不是调出来的，是另一个设计点**：~15-20 kernel/层 vs ferrite 的 ~156 kernel/层；
   tensor-core 30-50% MFU vs SIMT GEMV 0.7-4.9% 峰值。**对标 sglang 等于重写 kernel portfolio，不是调参数。**

---

## 1. 现状的五个锚（把地板计算的基础钉死）

| # | 量 | 值 | 来源 |
|---|---|---|---|
| A1 | verify(m=5) | **37.31 ms** | 实测 `DSV41_TIMING` |
| A2 | launches | **6224** / 步（40 层 × ~156） | 代码审计；本轮核对：`moe_rows` 45 + `attention_rows` 21 + `layer_rows` 4 个静态站点，逐行循环 ×m 后落到 ~156/层 ✓ |
| A3 | 流量 | **14.09 GB**（未切 head 口径；切分后 8.30GB） | `verify-calc-floor §2` |
| A4 | 达成带宽 | **381 GB/s = 峰值的 5.0%** | A3/A1 |
| A5 | **每发平均** | **5.99 µs** | 37.31ms / 6224 |
| A6 | per-kernel 下限 | 单核 exec **1.408µs** / 图内 node dispatch **0.411µs** | `graph_bench` |
| A7 | 审计口径 | 2.9µs submit + 3.3µs 最小执行 = **6.2µs** | `dspark-perf-400-plan §六` |

**A5 ≈ A7（5.99 vs 6.2）是这个分析的支点**：一次 launch 的全价 —— 但**只有 ~1.4µs 是"发核"**，
其余 ~4.6µs 是核自己的 μop 执行与延迟暴露。**所以"少发核"能拿回的是 6224×1.4µs ≈ 8.7ms 量级，
不是 20.5ms 量级**（后者把 per-kernel 全价当成了纯 boundary，与 hc 族自相矛盾：见 §2.3）。

---

## 2. 三个"理论地板"，以及它们为什么都不是真地板

### 2.1 带宽地板（任务口径 2ms）

```
折叠后权重 5.66GB  ÷ 7.672TB/s = 0.74 ms      ← 真带宽地板
现状未折叠 14.09GB ÷ 7.672TB/s = 1.84 ms
```
**只有 head 一族真的碰过带宽**（5.9 TB/s，77% 峰值），而它只占 1.12ms/37ms = 3%。
**凡是字节有意义的族都不重要；凡是要紧的族字节都无意义。**（`verify-marginal-cost §4 机理1`）

### 2.2 launch 地板（任务口径 3.9ms）

```
1300 发 × 3.0µs（审计口径） = 3.9 ms
1300 发 × 1.4µs（图内单核） = 1.8 ms
```
都是真的，但**只在 1300 发这个前提成立时**。6224 发的今天，这个地板是 8.7~18.7ms —— 已经超过 10ms 预算。

### 2.3 架构地板（真地板）

```
t_verify = Σ_f  N_f × t_f ,   t_f ≥ max( bytes_f/BW_peak , μop_f/issue_peak , latency_f/occupancy )
```

**关键：第三项（latency/occupancy）在今天的 verify 里主导。** 三个现场证据：

| 族 | 字节 | 实测 ms | 达成带宽 | 峰值占比 | 性质 |
|---|---:|---:|---:|---:|---|
| routed experts | 3133 MB | 8.30 | 378 GB/s | 4.9% | **L1TEX μop 地板** |
| 投影族 | 955 MB | 3.70 | 258 GB/s | 3.4% | mrows 已生效，肉在固定项 |
| gate | 786 MB | 3.44 | 229 GB/s | 3.0% | 逐行 ×5 |
| indexer | 435 MB | 2.50 | 174 GB/s | 2.3% | 逐行 + topk 单核地板 |
| attention | 262 MB | 2.80 | 94 GB/s | 1.2% | 纯 launch/低占用 |
| shared expert | 885 MB | 10.40 | 85 GB/s | 1.1% | 5× 重读 + 低占用 |
| hc | 157 MB | 2.96 | 53 GB/s | 0.7% | 已是 rows=m，纯占用 |
| head | 6619 MB | 1.12 | **5.9 TB/s** | **77%** | 唯一跑满 |

**自相矛盾的检验**：若把 A7 的「3.3µs/发 boundary」当作独立成本，hc 族 = 400×3.3µs = 1.32ms boundary
+ 其内禀工作；但实测 2.96ms，且 `hc_mixes` 隔离微基准就是 7.8µs、400×7.8µs = 3.12ms ≈ 2.96ms ✓。
⇒ **hc 的 2.96ms 几乎全是核自己的时间，boundary 项≈0。**
**所以"6224 × 3.3µs = 20.5ms 的 boundary"这个模型高估了**；真实可回收量更接近
`Σ N_f × (t_f − t_f^min)`，而下界 `t_f^min` 由 μop/占用决定 —— **这正是真地板**。

---

## 3. 架构地板的四级台阶（精确计算）

| 级 | 变化 | 物理量 | N | ms | 依据 / 阻塞 |
|---|---|---|---:|---:|---|
| **L0** | 今天 | — | 6224 | **37.31** | 实测 |
| **L1** | 只翻 flag：全 mrows 族 + hc A1/A2 + graph | 无（同核） | ~6200 | **31~33** | 实测子集 `{SH_EXP,GRAPH,ROPE,P3A}` = **−1.21**；GATE_MROWS(−2.75 未验) + INDEXER front(−1.0~1.5) + hc(−1.3~1.7) |
| **L2** | **+ tcgen05**（routed 换核，e4m3 grouped） | μop 数（LUT 消失） | ~6200 | **25~26** | −6.8（计划口径 −5~−6.8）。阻塞 = 4 gate 联合：`EXPERT_GROUPED` 默认 OFF、`EXPERT_ILV=0`、`GATEUP_FUSE=0`、`E4M3⊥MXF4` + e4m3 臂 parity |
| **L3** | **+ 族级融合**（N÷5，P1/P2/P3） | kernel 数 | **~1300** | **20~22** | `verify-family-fusion §4.2 保守列`（只承认 N÷5 + 指令模型 0.68-0.75×）|
| **L4** | **+ 占用/MLP 修复**（行进 grid、K-split、TMA 深度） | 在飞 warp 数 | ~1300 | **11~14** | 目标列；`STATUS:7943` 的 MLP 分析：打满需 in-flight 4.6MB > 单呼叫权重 4.5MB ⇒ **M=1 结构上不可能，M=6 共享后降 6×** |
| **L5** | **+ kernel 内 cp.async 流水 + 满 wave + smem 阶段交接** | 延迟暴露 | ~1000 | **8~9** | 400 的算术地板（`swallow-step-400-necessity §2.1` 的 2+3.9+2~3） |

**回答任务的问题 1**：
> `37.31 − 13~14 = 23~24ms` ≈ **L2/L3 之间**。
> **但真地板是 L3 的 20~22ms**（族级融合本身还值 −3~−5ms，它对 tcgen05 有部分重叠）。
> **要更低必须走 L4/L5，那是换核范式，不是继续翻 flag。**

### 3.1 对前提里四项预期的诚实修正

| 前提 | 判定 | 证据 |
|---|---|---|
| mrows 3~4ms | **⚠️ 高估**：实测全开 −1.21ms；`SH_EXP` 两次实测零收益 | `verify-ms-breakdown §修正`；`batched-400-v2-prediction §0` |
| tcgen05 6.8ms | ✅ 量级可信，**但不能单独拿**（4 gate 联合） | `batch-400-v2-prediction §3.4` |
| hc 1.3ms | ✅ 可信，最高 ROI（一行 truncate + 2 gate） | `batched-400-v2-remaining-roi §3` |
| graph 1.5ms | ✅ 可信，**但它是正确性工具不是性能项**（理论 −15ms 已被推翻） | 同上 §5 |

---

## 4. launch 地板的突破：6224 → 1300 需要什么

### 4.1 三条原语 + 逐族映射

| 原语 | 机制 | 用于 | launch |
|---|---|---|---|
| **P1 行进 block** | 5 行激活进同一 block，权重 tile 只 decode 一次；smem + grid barrier 交接阶段 | shared expert（三段一核 `sh_exp_fused<M>`，骨架 `gemm_fp8_sh_pair_kernel` 已是 **dead code**）、gate | 1000→40；200→40 |
| **P2 行进 grid** | 5 行进 grid 维，每行一 block，一发 | attention（`b·m` launcher 已就绪）、indexer（`indexer_topk` grid 已是 `(m,b)`）、compressor | 880→160；230→40；80→16 |
| **P3 阶段融合** | 用 kernel 内 grid barrier 把「写 global → 下一发读回」换成「写 smem → barrier → 读 smem」 | shared expert 的 w2 需要全部 inter 列（跨 block 依赖不可消） | 内含于 P1 |
| **P4 换核** | tcgen05 swapAB，`kind::mxf4` 的**最小合法 N=8 恰好容下 m=5/6 行** | routed experts | 400→40~120 |
| 纯接线 | `src_stride` / 层内 m 合并 | norm/quant（o/wo quant）、投影族（每层 4 发而非 20 发） | 680→250；2000→~200 |

合计：**6224 → ~1200~1400** ✓

### 4.2 两个硬阻塞（都不是性能问题，是正确性问题）

1. **per-row clen 的因果序**（attention / indexer 的 5 个 kernel 体）。
   verify 块内是 `read(r) → append(r) → read(r+1) → append(r+1)`；`window=128` 且长上下文恒回绕，
   行 `0..m-2` 会读到块自己的未来行 = **audit defect #2**。
   `DSV41_ATTN_MROWS` 因此在 `world>1 || pos+m-1 ≥ win` 时**全部 decline**——
   生产（TP8 + 长上下文）里等于永不生效。
   **解法**：`clen_rows[m]` 设备快照（**部分已在**：`indexer_rows_one` 已带 `mrows_clen: Option<ptr>`）
   + ring/window 的 block 内 r 升序合核（或块前 ring 快照）。`kv_snap_ring` 已分配（10.5MB）。
2. **tcgen05 与 e4m3 激活互斥**：`kind::mxf4` 只吃 e2m1，`kind::f8f6f4` 无 block-scale 操作数
   ⇒ 换核必须先做 **e4m3 臂的 GPU parity**（`dsv41_expert_tcgen05_gate_up_e4m3` 骨架已在、默认编入）。
   `DSV41_GATEUP_FUSE`（默认 ON）会触发 grouped 路径的 decline #3；`DSV41_EXPERT_ILV`（默认 ON）两个 arm 都拒。

### 4.3 launch 地板本身能压到多少

```
1300 发 × 1.4µs（图内单核 exec）  = 1.8 ms
1300 发 × 3.0µs（审计全价）        = 3.9 ms
```
**但这是"完全重叠"的上界**。串行 40 层依赖链里，kernel 之间的 ramp/drain 不能互相掩盖
（图化只证明 CPU submit 重叠，不证明 GPU 侧重叠）。**可兑现值更接近 3~5ms。**

---

## 5. instruction-bound 的突破

### 5.1 为什么 mrows 权重共享无效（精确模型）

`gemm_fp8_mrows_kernel` 折叠的是**权重解码**，不是**激活解码 + FMA**：

```
per (row, kb):  1 w-decode + 1 w-scale-mul  +  M × (1 a-decode + 1 a-scale-mul + 1 FMA)
逐行:           M × 5 指令
折叠后:         2 + 3M
⇒ M=5: 17/25 = 0.68× ；M→∞: 0.60×      ← 物理下限，永远拿不到 1/M = 0.2×
```
gate 的实测模型更狠：NT=1 时 15 指令/行，5 行 75；融合后 `5 + 50 = 55` ⇒ **0.73×**。

**而实测连 0.68× 都没兑现**（−1.21ms），原因在下面第二条墙。

### 5.2 两道独立的墙，对应两种完全不同的解法

| 墙 | 证据 | 物理量 | 解法 |
|---|---|---|---|
| **μop/操作数供给墙**（routed experts） | expert_gateup 24.1µs；**IPC 0.8/4、80% issue 槽在停等**；480 块/148SM = 3.24 块/SM；443GB/s（≤4% HBM）。§4.1：每 group 26 条 L1TEX op（4 LDS.128 + 16 LDS.64 LUT + …） | **L1TEX μop 吞吐**：3133MB fp4 ≈ 6.3G 值 × ~1.6 op ≈ 10G μop ÷（148SM×4/cyc×1.9GHz ≈ 1.1T op/s）≈ **8.9ms ≈ 实测 8.30ms** ✓ | **只能删指令 → tensor cores（tcgen05）**。SIMT 侧已由 v17–v24 五个正交变体全部证伪 |
| **延迟/MLP 墙**（所有 GEMV 族） | SIMT gemv 10.94µs，**对 L2 不敏感**（compute-bound）；有效带宽 373GB/s = 4.9%；打满需 in-flight 4.6MB > 单呼叫权重 4.5MB ⇒ **M=1 结构上不可能**；n 小 ⇒ blocks 72~96 ⇒ **每 SM 仅 3~4 warps** | **在飞 warp 数 × 每 warp 环深度** | **行进 grid（×M warps）+ K-split KStep=64/NSTAGE=16 + TMA 深度**。注意 v21/v19/v24 已证明「只动一个因子」无效——必须 warps 与环深**同时**上去 |

**⇒ 回答任务的问题 3**：mrows 不动这两堵墙中的任何一堵（它加寄存器累加器、不加 warp）。
突破顺序是 **先换核（删 μop）→ 再上 warp（藏延迟）**，二者不可互换。

### 5.3 一个反直觉但关键的结论

**在今天的核效率下，batched verify 结构上慢于 lazy。**
lazy 每行 m=1，权重驻留、activation 侧只做 1 行；batched 一次权重读但 activation 侧 ×6。
既然字节不是约束（5% 峰值）、μop/延迟才是，**batched 就该比 lazy 贵**——
现场正是如此：lazy step 22.56ms（92 tok/s）vs batched step ~39ms（~55 tok/s）。

**lazy→batched 的反转只在两个条件之一成立时发生**：
(a) accept 升到 3+（lazy 被迫跑 4 行 × 6.15ms = 24.6ms + draft），或
(b) mrows/占用让权重共享真正省钱（即 L4）。

---

## 6. 400 的真实路径

### 6.1 预算恒等式

```
verify + draft + commit ≤ 10ms
accept 3 ⇒ 4 tok/step ⇒ 4 / 0.010 = 400 tok/s
```

### 6.2 verify ≤9ms 需要什么

| 项 | ms | 前置 |
|---|---:|---|
| launch（1300 发 × ~3µs） | 3.9 | 族级融合（含 clen 因果） |
| 权重读一次（5.66GB ÷ 7TB/s） | 0.8 | 8 族全部折到 1× |
| 6 行独立计算（attention/MoE 激活侧） | 2~3 | — |
| 其余（AR 协议地板、hc、engram） | ~1.5 | AR 已是 2-kernel，折无可折 |
| **合计** | **8.2~9.2** | **= L5** |

**内部自洽 ✓，但它把 L5 的全部收益一次花光，余量为零。**

### 6.3 与 accept 的耦合（这是真正的判决）

| accept | tok/step | 400 所需步时 | 现有架构可达？ |
|---|---:|---:|---|
| **1.214（实测）** | 2.21 | **5.54 ms** | ❌ **低于 L5 地板（8~9ms）⇒ 物理上不可达** |
| 2.0 | 3.00 | 7.50 ms | ❌（L5 刚够，L3 远不够） |
| **3.0** | **4.00** | **10.0 ms** | ⚠️ **L5 刚好，零余量** |
| 4.0 | 5.00 | 12.5 ms | ✅（L4 的量级即可） |
| 5.0（sglang） | 6.00 | 15.0 ms | ✅ |

**⇒ 400 是两个独立缺口的交集**：
- **缺口 A（步时）**：37.31 → ≤9ms，需要 L5 全量（tegen05 + 族级融合 + 占用 + 流水）≈ **25~30 人日**，且尚无一项开工。
- **缺口 B（accept）**：1.214 → 3.0，需要 draft 数值审计的 5 个缺陷修完（同一 MTP head sglang 能到 5，说明是数值 bug 不是能力上限）。

**任何一个单独补齐都到不了 400。** 真实落点：**accept 3 + L4 的 ~12ms → 333 tok/s；accept 2 + L5 的 ~9ms → 333 tok/s。**
**诚实的票面预期：250~350 tok/s**；400 需要两端同时落在乐观端，成功率低。

### 6.4 一条必须写进账的负面判据

若 accept 停在 1.2 附近，**任何步时优化都改变不了量级**：
`accept 1.21 + L5 的 9ms = 2.21/0.009 = 246 tok/s`。
**400 的第一优先级是 accept，不是 verify。** 这一条与 `dspark-correctness-chain §910` 的结论一致
（64% 首 token 拒绝是瓶颈）。

---

## 7. 与 sglang 的对比：他们凭什么 13ms

**sglang DSpark B=1 B300 TP8：383.7 tok/s、accept ~5、步时 13ms。**
校验：`383.7 × 0.013 = 4.99 tok/step` = accept 5 ✓（6 行 verify + 5 draft，一个完整 spec 步 13ms）。

| 维度 | sglang | ferrite | 比值 |
|---|---|---|---|
| kernel 数/层 | **~15~20**（cutlass/tilelang grouped GEMM + MLA + fused MoE） | **~156**（`moe_rows` 45 + `attention_rows` 21 + `layer_rows` 4 静态站点 ×m 循环） | **~10×** |
| 计算范式 | tensor-core（tcgen05/TMA/warp-specialized），MFU 30-50% | SIMT GEMV / fp4-LUT，**峰值 0.7~4.9%**，IPC 0.8/4 | **~10×** |
| 图 | **整步一张图**（draft+verify+commit） | 每形状一张 verify 图，**6224 node**（图不消除 GPU 侧 ramp） | — |
| attention | MLA absorbed KV、单 KV head | DSA sparse + ring + compressor + indexer，5 个 kernel 体 | — |
| MoE | grouped GEMM，M = 6×tokens，持久 CTA + TMA | per-(row, slot) tiny GEMV，480 块/148SM = 3.24 块/SM | — |
| 每行边际 | (13ms − draft)/6 ≈ **~1.8ms/行** | 37.31/5 = **7.5ms/行** | **~4×** |

**结论**：sglang 的 13ms **不是同架构下的调优结果**，是"少而大的 tensor-core kernel"设计点的产物。
ferrite 的 37ms 是"多而小的 SIMT kernel 串行链"设计点的**必然**。
**对标 sglang ⇒ 必须换 kernel portfolio（L2+L4+L5），翻 flag 无效。**

> 附注：sglang 的 accept 5 用的是**同一个 MTP head 权重** ⇒ ferrite 的 1.214 是 draft 数值缺陷，
> 不是模型能力上限。**这一点让缺口 B 是可解的——但解完也只是把 400 从"绝无可能"变成"刚好可能"。**

---

## 8. 架构变化清单（按影响排序）

| 排名 | 变化 | 物理量 | Δms | 成本 | 阻塞 / 前置 | 性质 |
|---|---|---:|---:|---|---|---|
| **1** | **tcgen05 e4m3 grouped MoE（gate/up + down），N=8 tile 容 m=6 行** | 删 L1TEX μop | **−5~−6.8** | 4~5 人日 | 4 gate 联合（`EXPERT_GROUPED=1` + `EXPERT_TCGEN05_E4M3=1` + `GATEUP_FUSE=0` + `EXPERT_ILV=0`）+ **e4m3 臂 GPU parity** | **唯一能越过"5% 峰值"这堵墙的动作** |
| **2** | **族级融合波（N÷5：6224→~1300）** | kernel 数 | **−8~−15**（保守） | 12~15 人日 | **per-row clen 因果**（attention/indexer）；每族逐位 parity | 唯一不依赖核效率假设的收益 |
| **2a** | └ shared expert 三段一核 `sh_exp_fused<M>` | — | −5~−8 | 3.5 | 骨架 `gemm_fp8_sh_pair_kernel` 是 **dead code**，需 `template<M>` + `co_res` cap 防 barrier 死锁 | |
| **2b** | └ 投影族层内 m 合并 + attention `b·m` + norm/quant `src_stride` | — | −3~−5 | 5 | clen 因果 / `row_pitch`（TP>1） | |
| **2c** | └ indexer 纯接线（`INDEXER_MROWS` front） | — | −1~−1.5 | 0（代码已就位） | select 半缺 `out_stride` + `n_pos=max(lens)` | **零成本项** |
| **3** | **占用/MLP 修复：行进 grid（非寄存器）+ K-split KStep=64/NSTAGE=16 + TMA 深度** | 在飞 warp 数 | **把 L3 的 21ms 拉到 L4 的 11~14** | 5~8 人日 | v19/v21/v24 已证「只动一个因子」无效 | 决定 L3→L4 的成败 |
| **4** | **hc 链 A1+A2 接线**（A2 的 `bf16_truncate=false` 一行修复 + 两个 gate） | launch/占用 | **−1.3~−1.7** | **0.5~1 人日** | A2 的 truncate 坑；`HC_VERIFY_FUSE` 反向默认 | **最高 ROI**（已部分落地：`6f5de2b`） |
| **5** | **GATE_MROWS + head v1 mrows + ROPE/NORM mrows** | 字节/launch | −3.5~−5 | 1~2 人日 | 只差默认值 + A/B（SH_EXP 已证零收益，gate 未测） | 低风险 |
| **6** | **SWALLOW_STEP**（anchor 折进 verify 块，m=6） | 步预算重构 | **整步 −4.55**（verify +1.6） | 0.5 人日 | `SIDS_WRITEBACK=1`；**禁** LAZY/SEED_ALIGN；（m=6 与 m=5 各占一图槽，已支持） | **≤10ms 预算的结构性前提**，不是 verify 优化 |
| **7** | **kernel 内 cp.async 流水 + 满 wave + smem 阶段交接** | 延迟暴露 | **L4→L5 的 −2~−3** | 8~12 人日 | 第二次 kernel 重写 | 400 的最后一段 |
| **8** | attention scratch ring 快照 + 复合视图核重构 | — | −1.3 | 3~4 人日 | 落在 defect #1/#2 最密的代码；拷贝成本可忽略（~1.4µs/步） | 高风险，排 tcgen05 之后 |
| **9** | AR v5 store 折进 producer 尾（2→1 发） | — | −0.2~−0.4 | 1.5~2 人日 | `AR_STORE_FUSE` 数值破功未解；**AR 已是 2-kernel 不是 3** | 协议地板 |
| — | verify 图化 | — | **≈0（已入账 −1.5）** | — | — | **正确性/顺序工具，不计入收益** |
| — | head fold | — | −0.24 | — | K 序 parity | 不做 |

**不要浪费机时的三条**：
1. 不要按图的 −15ms 编预算（实测 −1.5ms）。
2. 不要指望 SH_EXP_MROWS 的 −8.3ms（两次实测零收益，instruction-bound）。
3. 不要在缺 nsys 证据时对 mrows gate 下"declined"结论（5 个 gate 静默 `return Ok(false)`，
   只有 `VERIFY_HEAD_MROWS` / `ATTN_MROWS` 有 one-shot 日志）。

---

## 9. 交付前必须钉死的三个未知（否则上面的区间会偏）

| # | 未知 | 影响面 | 取证 |
|---|---|---|---|
| U1 | **per-kernel boundary 的真实可回收量** | L3 的 20~22ms 是保守还是乐观（±5ms） | 一次 nsys：按 kernel 名聚合 verify 段，看 `expert_gateup_fp4_batched` / `gemm_fp8_mrows` / `sparse_attn_split+merge` 的 **launch 数 + 每发实际 µs** |
| U2 | **routed experts 的 ±6ms**（V2，全表最大不确定项） | L2/L3 的关键输入 | 同上；`grid.z=rows` 后 CTA 数 ×5，达成带宽可能从 443 上移到 1.3TB/s ⇒ 8.30 → 2.4ms |
| U3 | **e4m3 臂 tcgen05 的 GPU parity** | L2 是否成立（−6.8ms 的开关） | `tests_tcgen05_mxf8f6f4_1x.cu` 的 (2)(3)(4) 从未上 GPU；先跑 parity 再谈收益 |

---

## 10. 一句话交付

> **当前架构的 verify 地板是 ~20~22ms（族级融合 + 换核，保留今天的核效率）**，
> 而不是任务前提的 23~24ms —— 后者既高估了 flag 的收益、又完全没算融合。
> **地板由「tiny kernel 的 L1TEX μop 数 + 3~4 warps/SM 的延迟暴露」决定**，
> 带宽（0.74ms）和 submit（−1.5ms）都早已出局。
> **400 的真实路径 = verify ≤9ms（L5：tcgen05 + 融合 + 占用 + 流水，~25~30 人日，未开工）
> ∩ accept ≥3（draft 数值审计，同 head sglang 能到 5）。**
> **在当前 accept 1.214 下，400 需要一个 5.5ms/步的机器——低于物理地板，不可达。**
> **sglang 的 13ms 不是同架构的调优上限，是另一个 kernel portfolio 的结果。**

---

*工部 · 只读分析，未执行 GPU 命令、未改动任何源码；本文件为唯一产出。*
*所有 ms/launch/字节数均标注来源（实测 / 账本 / 代码推导）；推算项集中在 §3 表与 §9。*
