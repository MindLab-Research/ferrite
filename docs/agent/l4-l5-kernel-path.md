# L4 / L5 kernel 优化路径 — 从族融合地板到 400 算术地板

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未改动任何源码、未执行 GPU 命令。
> 任务：`verify-architecture-floor.md` 确定 L3（族融合，~20-22ms）之后，拆解 **L4（+占用/MLP，~11-14ms）**
> 与 **L5（+流水/满 wave，~8-9ms）** 的**具体 kernel 工作**。
> 代码基线：工作树 HEAD；kernel 行号以**函数名**为准（仓库有行号漂移的历史）。
> 关键前置文档：`verify-architecture-floor.md` §5.2/§8 · `sh-pair-template-m-design.md` §3.3 ·
> `routed-expert-residual.md` §2.2/§3 · `tcgen05-e4m3-grouped-expectation.md` §5.2 ·
> `verify-eager-fusion-migration.md` §3.3 · `hc-chain-bandwidth-analysis.md` §3-B ·
> `dspark-perf-400-plan.md` §十 · `dsv41-persistent-arch.md` §6。

---

## 0. 口径校准（先钉死，否则清单会错位）

| # | 量 | 本文件取的口径 | 依据 |
|---|---|---|---|
| C1 | **L3 完成态** | arch-floor 口径 **~20-22ms**（族级融合 + tcgen05，保留今天的核效率）；任务给的「当前 ~15-18ms」是 **L3 部分兑现**的实测 | `verify-architecture-floor §3` |
| C2 | **L4 的物理量** | **在飞 warp 数 × 每 warp 环深**（不是字节、不是 launch） | `arch-floor §5.2` 第二道墙 |
| C3 | **L5 的物理量** | **延迟暴露的消除**（cp.async 多级流水 + 满 wave + smem 阶段交接） | `arch-floor §3` L5 行 |
| C4 | **两堵墙的顺序** | **先删 μop（换核）→ 再上 warp（藏延迟）**，不可互换 | `arch-floor §5.2` 判词 |
| C5 | **tcgen05 的边界** | L2 只换 **gate/up**；**down 无 tcgen05 kernel** → L2 的 −6.8ms 天然砍半 | `tcgen05-e4m3-grouped-expectation §5.1` |
| C6 | **L4/L5 不是翻 flag** | v19/v21/v24 已证「只动一个因子」无效；v17→v21 四变体全中性 | `arch-floor §5.2` / `§8 排名 3` |

**一句话边界**：L3 是「少发 kernel + 换核」的终点；**L4 让每个 kernel 内部有足够 warp 藏住 LDS/L2 延迟；
L5 让每个 kernel 内部的流水线不再暴露 staging 延迟**。两者都是 **kernel 重写**，不是配置。

---

## 1. L4 工作清单（占用 / MLP）——目标 20-22ms → 11-14ms

> L4 的判据统一用 **「在飞 warp 数」**：`blocks/SM × warps/block`。今天的 GEMV 族是 **3~4 warps/SM**
> （`verify-family-fusion §1.3 M-b`），148 SM 上大量 SM 空转。

### 1.1 kernel 内占用

| # | kernel（C 符号 / 落点） | 改动 | 预期节省 | 人日 | 把握 |
|---|---|---|---:|---:|---|
| **L4-1** | `gemm_fp8_mrows_kernel<M>` / launcher `dsv41_gemm_fp8_mrows` | **默认翻转 `DSV41_MROWS_SMALL_N_ADAPTIVE=1`**（代码已在树里，默认 OFF）：`dsv41_mrows_warps_for()` 让 n<256→1 row/block、n<512→2。shared w1/w3 形状 n=288 → grid 从 **72 块（49% SM）** 抬到 **144~288 块** | −0.2~−0.6ms | **1**（A/B + crossover 调参） | 中低（instruction-bound，`SH_EXP_MROWS` 两次实测零收益是反向证据） |
| **L4-2** | `gemm_fp8_sh_exp_pair_kernel<M>` / `dsv41_gemm_fp8_sh_exp_fused`（含既有 `gemm_fp8_sh_pair_kernel`） | **phase-1 加 K-split**：现在 phase-1 的并行度 = `9 i-tile × M`（fold_r=1 → 54 块，36% SM），phase-2 是 160 块；grid = `max(p1b, g2t)` ⇒ **106 个 block 在 barrier 上空转**。把 k1=5120 切成 ks 块（grid.x = 9×M×ks），加「按 ck 升序」的 partial reduce | −0.3~−0.8ms | **2**（K-split 骨架 + parity） | 低（**非逐位**：f32 结合序变；phase-1 权重是 L2 常驻，不是带宽项） |
| **L4-3** | `tc5::mxf4::expert_tcgen05_gateup_mxf4_kernel` / `m4_launch_gateup`（+ `tc5::e4::*`、`tc5::e4x::*`） | **加第三 grid 维做 K-split** + 升序 reduce（源码 `[K-SPLIT TODO]` 已标注）。现状 grid=(30, slots)，slots=1 → **30 CTA ≈ 15 SM**；kRing=8 只给 28KiB/CTA in-flight，**打不满 DRAM** | −1.0~−3.0ms | **3~4** | 中（骨架在、MMA 形式 GPU 已验 EXACT；reduce 非结合需 parity） |
| **L4-4** | **新核** `tc5::down::*`（不存在；对照 `expert_gemv_fp4_down_reduce_kernel`） | **down 的 swapAB tcgen05**（k=inter_local=320 → n=dim=5120）+ fused asc-slot reduce。现状 down **3.48ms 仍是纯 SIMT** | −2.0~−2.5ms | **3~4** | 中低（无骨架，从零写） |
| **L4-5** | `tc5::e4x` grouped kernel | **M=128 过量的对策**：`1-CTA kind::f8f6f4` 的 M 硬件固定 128，每个 expert 只占 1~3 行 ⇒ **~120× 过量 MMA**。对策不是改 tile，而是①grouped mask 把多 assignment 填满 128 行 / ②接受过量、靠 L4-3 的 CTA 数掩盖 | 0~−1.0ms | **2~3** | 低（研究探索，可能放弃） |

> **tcgen05 边界提示（C5）**：L4-3/L4-4 其实是 **L2 的收尾**——若当前「L2 已落地」仅指 gate/up 换核，
> 则 routed 仍在 ~3ms，L4-3+L4-4 是把它打到 1.0~1.5ms 的**必要路径**。若 L2 尚未落地，这两项归 L2 账。

### 1.2 MLP（multi-level parallelism）

| # | kernel / 落点 | 改动 | 预期节省 | 人日 | 把握 |
|---|---|---|---:|---:|---|
| **L4-6** | **发射层**（`chain_dev.rs::layer_rows` @ `moe_rows`/`attention_rows`；无新 kernel） | **把 EAGER 的三条侧流迁到 verify**：`dual_chain`（q 链 vs kv 链）· `compress_side`（compressor 4 发）· `moe_dual`（routed vs shared）。verify 的 `layer_rows` **全程单流**，EAGER 的三条 fork/join 一个都没接 | −0.5~−1.5ms | **3~4** | 中（`devrt.rs` 的侧流原语已在；风险在 m 行块的 stream 顺序 + CUDA graph 捕获） |
| **L4-7** | `hc_front_split` / `hc_mix_dots_kernel` / `hc_dots_late_kernel` / `hc_mixes_tail_kernel` | **hc 链的 side stream 遮蔽 sinkhorn**：接 A1/A2（`HC_VERIFY_FUSE`/`HC_FRONT_ROWS`，代码已就位）+ verify 侧 AR 折叠。`sinkhorn` 是 **6.46µs 的 warp0 串行链**，同核内无等长窗口，只能挪到 `dl` 流 | −1.3~−1.7ms（大部分是 L3 的 A1/A2 欠账；L4 增量 = 侧流遮蔽 −0.6ms） | **0.5~1**（truncate 修复 + A/B） | 中高（**最高 ROI**，但 A2 的 `bf16_truncate` 坑必须先修 false） |
| **L4-8** | `hc_dots_late_kernel`（`DSV41_HC_DL_KCHUNK`）+ `gemm_fp8_mrows` 的 P2/P3 | **占用率修复：K-split/环深与 warps 同时上**。v19/v21/v24 已证「只动一个因子」无效；`hc_dots_late` 的 3.84MB/发里 **1.92MB 是纯冗余**（每 block 把同一份 x 行拖一遍），换 48KB 双缓冲 | −0.2~−0.3ms（hc）+ 投影族 −0.3~−0.5ms | **1.5~2** | 中（`KCHUNK` 启动器自带对齐约束，默认 OFF） |

### 1.3 L4 小结

```
L4 总预期： −4.5 ~ −9.5ms（20-22 → 11-14 ✓ 与 arch-floor §3 的台阶吻合）
L4 总成本： 14 ~ 21 人日（含 tcgen05 收尾 L4-3/4 的 6~8 人日）
若把 tcgen05 收尾划归 L2 账：L4 纯占用/MLP 部分 = 6 ~ 13 人日 ≈ arch-floor §8 排名 3 的 5~8 人日
```

---

## 2. L5 工作清单（流水 / 满 wave）——目标 11-14ms → 8-9ms

> L5 的判据统一用 **「staging 延迟是否暴露」**：今天每个 kernel 的 K/权重流是
> 「issue → commit → wait → consume」的**单缓冲**，延迟全暴露。L5 把它们改成多级 ring。

### 2.1 kernel 内流水（cp.async double buffering）

| # | kernel（C 符号） | 现状流水 | 改动 | 预期节省 | 人日 | 把握 |
|---|---|---|---:|---:|---|
| **L5-1** | `expert_gemv_fp4_batched_kernel`（P4 `DSV41_GATEUP_CPASYNC`） | **单缓冲**：PDEPTH=1 的 weight-first prologue，注释明确「no depth above 1 is worth an instantiation」 | **PDEPTH=2~3 的 per-warp ring**（`cp.async.wait_group N` 是立即数 ⇒ 需模板实例化）。SMEM 代价 512B×nwarps×PDEPTH | −0.3~−0.6ms | **2** | 中低（⚠️ 必须与 L4-3 的 K-split 同时做，否则 occupancy 被吃掉） |
| **L5-2** | `gemm_fp8_mrows_kernel<M>` / `gemm_fp8_gemv_kernel`（P3 `g_gemv_cpasync`） | **单缓冲是刻意选择**：注释写「A separate double-buffer slot would cost another warps*k bytes (48512→4 blocks/SM)：**用 residency 换 overlap 是陷阱**」 | **re-balance：K-split + 双缓冲 + 更多 block 三者同时调**（v19/v21/v24 的铁律） | −0.3~−0.8ms | **2~3** | 中低（v19/v21/v24 全中性的历史） |
| **L5-3** | `tc5::e4x::*`（grouped masked） | **零流水**：`kStageAtoms=2` 单缓冲、每 stage `fence.proxy.async`+`__syncthreads`，**0 条 cp.async**；40 stage 全串行 | **抄 `tc5::mxf4` 的 kRing=8 TMA ring + mbarrier**（同文件已有完整先例） | −0.5~−1.5ms | **3~4** | 中（先例在、但 e4x 从未上过 GPU） |
| **L5-4** | `sparse_attn_warp_kernel`（已有 3 深预取）· `expert_gemv_fp4_down_reduce_kernel` | 部分已有预取；down 的 mode 3/4 受 40-reg 压 | **down 的 4-value 解码 @ 40 reg**（微基准 0.90×，生产因 56reg→1.51 wave 悬崖从未跑过）+ attn 预取加深 | −0.3~−0.6ms | **1.5~2** | 中（有微基准背书） |

### 2.2 满 wave（CTA 数 = SM 数的整数倍）

| # | kernel | 现状 CTA / wave | 改动 | 预期节省 | 人日 | 把握 |
|---|---|---|---:|---:|---|
| **L5-5** | `expert_gemv_fp4_batched_kernel` | **480 CTA / 148 SM = 3.24 块/SM**（尾波 24%） | rows(8/4/2) × ksplit(1/2) × K-split 的组合，使 CTA ∈ {444, 592}（3×148 / 4×148）；`DSV41_GATEUP_ROWS` 旋钮已在 | 含在 L5-1 | （合并） | 中 |
| **L5-6** | `tc5::*`、`gemm_fp8_mrows<M>`、`gemm_fp8_sh_exp_pair<M>`、`ferrite_p2p_ar_v5*`、`hc_*_kernel` | **AR=5 block**（n4=1280→5×256，5 SM）；hc=`rows` 块（5/148 SM）；sh_pair=`max(54,160)` | 逐 kernel 用 nsys 量 wave 数，把 grid 调成 148 的整数倍（或最小化尾波）；`co_res` cap 保护不可越 | −0.3~−1.0ms | **2** | 中（需逐 kernel wave 证据） |
| **L5-7** | 段核（`dsv41-persistent-arch`，`hc_front_persist*`） | P1d 实测 **+3.3ms 回归**（tail 串行链在整格 drain 期间跑） | **不推荐**；若要，只能用无自旋的角色分解（`hc_dots_late` 的 elected-last-block），**绝不 ticket 自旋**（hc-merge 教训 +3.2ms） | 0~−1.0ms | **3~5** | 低（P1d 已证回归） |

### 2.3 L5 小结

```
L5 总预期： −2.0 ~ −5.0ms（11-14 → 8-9 ✓ 与 arch-floor §3 的台阶吻合）
L5 总成本： 9 ~ 12 人日（不含段核 L5-7）
```

---

## 3. 从当前（L2-L3）到 L4-L5 的路径（先做什么后做什么）

### 3.1 依赖图（硬约束）

```
[L3 尚欠]  hc A1/A2 接线（truncate=false 已修）        ──→ L4-7 的侧流遮蔽
[L2 尚欠]  tcgen05 gate/up 落地 ──→ L4-3 K-split ──┬──→ L4-4 down 换核
                                                    └──→ L5-3 e4x kRing
[L4]       L4-1 nwarps 自适应（零代码）──→ L5-2 gemv 双缓冲
[L4]       L4-2 sh_pair K-split（独立）
[L4]       L4-6 多流发射 ──→ L4-7 hc 侧流（共用 devrt 原语）
[L5]       L5-1 gateup PDEPTH ──（必须与 L4-3 K-split 同批）
[L5]       L5-5/L5-6 满 wave ──（在 L4/L5 的 grid 形状定稿后收尾）
```

### 3.2 建议波次

| 波 | 内容 | 前置 | 人日 | 里程碑 |
|---|---|---|---|---|
| **W-L4a** | **L4-1 + L4-7**（零/低成本项）：`MROWS_SMALL_N_ADAPTIVE=1` A/B；hc A1/A2 接线（truncate 已修） | 无 | 1.5~2 | 先拿 −1.5~−2.3ms |
| **W-L4b** | **L4-3 tcgen05 K-split**（第三 grid 维 + 升序 reduce）——**L4 的最大单项** | L2 gate/up 落地 + 单层微基准门（22.2/17.2µs） | 3~4 | routed 从 ~3ms → ~1.5ms |
| **W-L4c** | **L4-4 tcgen05 down 换核**（从零写） | W-L4b 的形状定稿 | 3~4 | routed → 1.0~1.5ms（L4 达成） |
| **W-L4d** | **L4-6 verify 多流发射** + **L4-2 sh_pair K-split** | W-L4a（hc 侧流共用原语）+ L3 的 SH_PAIR | 5~6 | 关键路径遮蔽 + phase-1 满 SM |
| **W-L5a** | **L5-1 gateup PDEPTH** + **L5-3 e4x kRing** | W-L4b/W-L4c（grid 形状） | 5~6 | staging 延迟被 ring 藏住 |
| **W-L5b** | **L5-2 gemv 双缓冲** + **L5-4 down 40-reg** | W-L4a（nwarps 定稿） | 3.5~5 | 投影族 + down 流水 |
| **W-L5c** | **L5-5/L5-6 满 wave 收尾**（逐 kernel nsys wave 证据） | W-L4/L5 全部 | 2 | 8-9ms 地板 |

**关键路径**：`L2 tcgen05 gate/up → L4-3 K-split → L4-4 down → L5-3 e4x kRing` 是唯一的串行主干
（约 11~15 人日）；其余项可并行。

---

## 4. 成本汇总（交付物）

| 层 | 项 | kernel | 预期节省 | 人日 |
|---|---|---|---|---:|
| **L4** | L4-1 | `gemm_fp8_mrows_kernel<M>` | −0.2~−0.6 | 1 |
| | L4-2 | `gemm_fp8_sh_exp_pair_kernel<M>` | −0.3~−0.8 | 2 |
| | L4-3 | `tc5::mxf4::*`（K-split） | −1.0~−3.0 | 3~4 |
| | L4-4 | **新** `tc5::down::*` | −2.0~−2.5 | 3~4 |
| | L4-5 | `tc5::e4x::*`（tile 利用） | 0~−1.0 | 2~3 |
| | L4-6 | 发射层（`layer_rows` 多流） | −0.5~−1.5 | 3~4 |
| | L4-7 | `hc_front_split` / `hc_dots_late_kernel` | −1.3~−1.7 | 0.5~1 |
| | L4-8 | `hc_dots_late_kernel`(KCHUNK) + gemv P2/P3 | −0.5~−0.8 | 1.5~2 |
| | **小计** | | **−5.8 ~ −11.9** | **16.5~21.5** |
| **L5** | L5-1 | `expert_gemv_fp4_batched_kernel`(PDEPTH) | −0.3~−0.6 | 2 |
| | L5-2 | `gemm_fp8_mrows_kernel<M>`(双缓冲) | −0.3~−0.8 | 2~3 |
| | L5-3 | `tc5::e4x::*`(kRing TMA) | −0.5~−1.5 | 3~4 |
| | L5-4 | `expert_gemv_fp4_down_reduce_kernel`(40reg) + attn 预取 | −0.3~−0.6 | 1.5~2 |
| | L5-5/6 | 全族满 wave | −0.3~−1.0 | 2 |
| | L5-7 | 段核（**不推荐**） | 0~−1.0 | 3~5 |
| | **小计** | | **−1.7 ~ −4.5** | **9~13** |
| | **L4+L5 合计** | | **−7.5 ~ −16.4** | **25.5~34.5** |

> 与 arch-floor §8 对账：排名 3（占用/MLP）5~8 人日 + 排名 7（流水/满 wave）8~12 人日 = **13~20 人日**。
> 本表偏多的原因：**把 tcgen05 的 down 换核（3~4）与 K-split（3~4）显式列出**——arch-floor 把它们
> 记在 L2（排名 1）的 4~5 人日里。**若 L2 的 tcgen05 已完整落地，L4+L5 的成本收敛到 ~18~24 人日。**

---

## 5. 诚实校准（必须写在账上）

1. **L4/L5 的全部 ms 预期都是设计口径，仓库里没有任何一项有实测背书。**
   反向证据：`{SH_EXP_MROWS, VERIFY_GRAPH, VERIFY_ROPE_MROWS, DRAFT_P3A}` 全开只 **−1.21ms**（预期 −24）；
   `SH_EXP_MROWS` 两次实测零收益。**instruction-bound + 低占用的墙，靠「只动一个因子」是推不动的**
   （v19/v21/v24 四变体全中性）。
2. **L4 的成败集中在 routed experts（tcgen05 收尾）**：`L4-3 + L4-4` 占 L4 收益的 ~60%。
   其余七项（占用/MLP）加起来只有 −2~−4ms。**这正是 arch-floor §5.2 的「先换核、再上 warp」**。
3. **L5 的先例其实已经在树里**：`tc5::mxf4` 的 kRing=8 TMA ring 就是 L5 形态的样本；
   L5-3 是把它抄到 e4x grouped。**L5 的难点不在设计，在于「双缓冲 vs residency」的取舍**
   （`gemm_fp8_mrows` 的注释已经把陷阱写死：多一个 buffer 槽 = 少 1 block/SM）。
4. **必须上机取证的三件事**（否则清单会偏 ±5ms）：
   - **U1**：一次 nsys 按 kernel 名聚合 verify 段，量每个 kernel 的 **wave 数**（L5-6 的唯一输入）；
   - **U2**：tcgen05 K-split 前，先跑单层微基准门（gateup 22.2µs / down 17.2µs）——**不达标即止损**；
   - **U3**：e4m3 臂的 GPU parity（`tc5::e4x` 从未上过 GPU；两条 `[OPEN]` 是静默错值风险）。
5. **L4-5（M=128 过量）可能是错药**：mask 省不掉 MMA 工作量，`120× 过量`若不解决，
   tcgen05 的落点会是 2~3.5ms 而非 1.0~1.5ms（`tcgen05-e4m3-grouped-expectation §5.2`）。
   **建议先做 L4-3/L4-4 拿实测，再用实测决定 L4-5 是否值得投。**

---

*工部 · 只读分析，未执行 GPU 命令、未改动任何源码；本文件为唯一产出。*
*所有 kernel 名以 `__global__` 符号为准；ms 均标注「设计口径」（无实测）。*
