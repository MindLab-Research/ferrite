# L4 占用 / MLP 优化设计 —— 每个 kernel 的「在飞 warp 数」修复

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未改动任何源码、未执行 GPU 命令。
> 任务：综合路线图判定 **L4 是 400 的唯一一道门**（M6 = 430~545 tok/s @ 25~44 人日，
> `final-400-sprint-roadmap §5`）。本文件把 L4 从「−5~8ms」拆成**逐个 kernel 的施工图**：
> 当前占用 → 目标占用 → 改动方法 → 预期节省 → 实施成本。
> 代码基线：工作树 HEAD `2f59369`；kernel 名以 `__global__` / `extern "C"` 符号为准（仓库有行号漂移史）。
> 前置：`l4-l5-kernel-path.md`（清单骨架）· `verify-architecture-floor.md §5.2`（两堵墙）·
> `lazy-verify-optimization-path.md`（per-step 族 ×k_emit）· `sh-pair-template-m-design.md §3.3` ·
> `routed-expert-residual.md` · `expert-tcgen05-plan.md` · `hc-chain-bandwidth-analysis.md` ·
> `batched-400-v2-remaining-roi.md` · `final-400-sprint-roadmap.md`。

---

## 0. 口径与判据（先钉死，否则清单会错位）

| # | 量 | 本文件取的口径 | 依据 |
|---|---|---|---|
| C1 | **L4 的物理量** | **在飞 warp 数 / CTA 数**（不是字节、不是 launch） | `arch-floor §5.2` 第二道墙 |
| C2 | **L4 与 B 类的分野** | B 类（mrows fold）= 收益源是 **m**（行折叠）；**L4 = 收益源是 n（权重行数）与 CTA 数** | 本文件 §3 的核心结论 |
| C3 | **两堵墙的顺序** | **先删 μop（换核）→ 再上 warp（占延迟）**，不可互换 | `arch-floor §5.2` 判词 |
| C4 | **tcgen05 的边界** | `tc5::mxf4`/`tc5::e4` **只有 gate/up**；**down 无 tcgen05 核**（`tc5::down` 不存在） | `dsv41_experts_mxf4.cu`（`namespace tc5`：无 down 符号） |
| C5 | **L4 不是翻 flag** | v19/v21/v24 已证「只动一个因子」无效；`{SH_EXP_MROWS,VERIFY_GRAPH,VERIFY_ROPE_MROWS,DRAFT_P3A}` 全开只 −1.21ms | `arch-floor §5.2` / `verify-ms-breakdown §修正` |
| C6 | **所有 ms 均为设计口径** | 仓内 L4 无任何一项有实测背书 | `l4-l5-kernel-path §5.1` |

**生产形状（TP8，`configs/dsv41_flash.json` + `config.rs::production_shapes()` 断言）**：

```
dim = 5120 · moe_inter_dim = 2304 · world = 8
inter_local   = padded_inter(2304/8) = padded(288) = 320     # routed 每 rank 的 inter 行数
sh_il         = 2304/8 = 288                                  # shared expert 的 n1（w1/w3 行数）
topk = 6 · n_routed = 384
hc = 4 · hc_dim = 20480 · mix = hc*(2+hc) = 24
nh = 64 · nlh = 8 · hd = 512 · ql = 1280 · ol_local = 1024
window = 128
```

**硬件**：B300 · 148 SM · 227 KB smem/SM（opt-in）· 2048 threads/SM（64 warps）· 65536 regs/SM ·
HBM 7.672 TB/s · **TMEM 256 列/CTA ⇒ tcgen05 族 ≤ 2 CTA/SM**。

**占用度量**（本文件统一用两个数）：`CTA 数 / 148`（SM 覆盖率）与 `resident warps/SM`。
今天的 GEMV 族普遍是 **3~4 warps/SM**（`verify-family-fusion §1.3`）。

---

## 1. 占用现状总账（逐个 kernel，读码结论）

> 「当前」列：生产形状、**m=1（lazy）与 m=6（batched）两栏**——因为 L4 的靶子大多是 n 驱动的，
> m 只改变某些 kernel 的 workload，不改变 CTA 数（§3 的完整论证）。

| # | kernel（C 符号 / Rust 落点） | 形状 | **CTA（m=1 / m=6）** | SM 覆盖 | 绑定资源 | 目标 |
|---|---|---|---|---|---|---|
| K1 | `gemm_fp8_mrows_kernel<M>` / `dsv41_gemm_fp8_mrows` | sh w1/w3 n=288,k=5120 | **72 / 72** | 49% | block 数（n/nwarps） | 144~288 |
| K1b | 同上（wkv 投影） | n=512,k=5120 | **128 / 128** | 86% | 同上 | 256（需调 crossover） |
| K2 | `gemm_fp8_sh_exp_pair_kernel<M>` **phase-1** | n1=288 | **9 / 54** | 6% / 36% | grid（=n1t·nsr） | 9·ks / 54·ks |
| K2b | 同上 **phase-2** | n2=5120 | 160 / 160 | 108% | — | — |
| K3 | `tc5::mxf4::expert_tcgen05_gateup_mxf4_kernel` | rows=640,slots=6 | **(5,6)=30 / 30** | 10% | **TMEM 256 列 → ≤2 CTA/SM** | (5,6,ks) ks=8 → 240 |
| K4 | `expert_gemv_fp4_down_reduce_kernel` | dim=5120,k=320 | **(640,1,1)=640 / 3840** | 432% | regs（40 → 6 blk/SM） | 换核（`tc5::down`） |
| K5 | `e4m3_gemm_grouped_kernel`（`tc5::e4x`） | n_total=640,n_assign=6 | **(10,1,6)=60 / 60** | 40% | — | 重排 mask（过量为 128/m） |
| K6 | `hc_mixes_kernel` / `dsv41_hc_mixes` | grid = **rows = m** | **1 / 6** | **0.7%** | **grid=rows** | 折核（L1/A2）+ side stream |
| K7 | `hc_mix_dots_kernel` / `hc_dots_late_kernel` | grid = (mix, rows) | **(24,1)=24 / (24,6)=144** | 16% / 97% | smem 160 KB → **1 blk/SM** | KCHUNK → 4 blk/SM |
| K8 | `hc_mixes_tail_kernel` | grid = rows | **1 / 6** | 0.7% | grid=rows + 128 thr | 折进 dots/LATE |
| K9 | `dsv41_hc_collapse_norm_kernel`（A1-a） | grid = rows | **1 / 6** | 0.7% | grid=rows | 折核后仍 1：**待摊开** |
| K10 | `dsv41_rmsnorm_rows_kernel` | grid = rows | **1 / 6** | 0.7% | grid=rows | 同上 |
| K11 | `hc_post_kernel`（staging 形态） | 100×256 | 100 / 100 | 68% | — | `hc_post_inplace`（去 memcpy） |
| K12 | `cudaMemcpyAsync` D2D（h2→h） | — | 1 / 1 | 0.7% | — | 删除（A1-b） |
| K13 | `p2p_ar_store_v5_kernel` / `_pubred_v5_kernel` | n=5120 | **(20,world)=160 / 160** × 64thr；pubred **20** | 14%（pubred） | 协议地板 | 不投（§7） |
| K14 | `sparse_attn_split_kernel` / `_merge_kernel` | (split_c, b·m, h) | **(4,1,8)=32 / 192**；merge **(1,8)=8** | 22% / 5% | — | 归 L5（预取）；非 L4 主项 |
| K15 | `gemm_fp8_mrows_kernel<M>` 其余投影 | wq_a 1280 / wq_b 4096 / wo 1024 | 320 / 512 / 256 | 216%/346%/173% | — | — |

**读表要点（三条）**：

1. **最刺眼的是 K6/K8/K9/K10：grid = `rows` = m**。在 **lazy（m=1）下它们是 1 个 CTA —— 只占 1 个 SM**。
   这是 `hc 族 = 2.96ms / 全表最低 53 GB/s`（`hc-chain-bandwidth-analysis §2`）的**真正根因**，
   而不是「kernel 写得差」。**L4 在 lazy 下的最大单项就在这四行**，且被 `×k_emit` 放大（§3）。
2. **K3 只有 30 个 CTA**（rows/128=5 × slots=topk=6），受 **TMEM 256 列/CTA** 钉死在 ≤2 CTA/SM
   ⇒ 15 个 SM 在干活，**133 个 SM 空转**。kRing=8 只给 28 KiB/CTA in-flight，
   30 CTA × 28 KiB = 840 KiB，**只有 HBM 带宽-延迟积（≈5.6 MB）的 15%** ⇒ 打不满 DRAM
   （`dsv41_experts_mxf4.cu` 的 OCCUPANCY / DEPTH NOTE）。
3. **K1（shared expert）49% 覆盖**是 mrows 族唯一的低占用点；但**它在 lazy 下会被 L3（SH_PAIR_M）
   接管**，届时这个靶子消失（§4-L4-1 的互斥说明）。**K1b（wkv, n=512）落在 crossover 外**，
   需要把 `kMrowsSmallN2` 从 512 抬到 640 才能吃到。

---

## 2. L4 工作清单（施工图）

> 每项：**kernel 名 + 当前问题 + 优化方法 + 预期节省（设计口径）+ 实施成本 + 把握 + 前置**。
> 「把握」按仓内证据强度：高=有实测/骨架已验 · 中=有代码先例 · 低=纯外推。

### 2.1 kernel 内占用（7 项）

| # | kernel | 当前问题 | 优化方法（改动落点） | 预期节省 | 人日 | 把握 | 前置 |
|---|---|---|---|---:|---:|---|---|
| **L4-1** | `gemm_fp8_mrows_kernel<M>` / `dsv41_mrows_warps_for`（`dsv41_kernels.cu`） | n=288（sh w1/w3）时 `nwarps=4` ⇒ **72 CTA（49% SM）**；n=512（wkv）时 128 CTA（86%） | ① 翻 `DSV41_MROWS_SMALL_N_ADAPTIVE=1`（代码已在树，默认 OFF）；② **调 crossover 常量** `kMrowsSmallN1: 256→320`（吃 n=288 的 1 行/block = 288 CTA）、`kMrowsSmallN2: 512→640`（吃 wkv n=512 的 2 行/block = 256 CTA）。两个常量已命名，A/B 一处可改 | −0.2~−0.6 | **1** | 中低 | **无（但见 §4-L4-1 与 L3 互斥）** |
| **L4-2** | `gemm_fp8_sh_exp_pair_kernel<M>` / `dsv41_gemm_fp8_sh_exp_fused` **phase-1** | **bit-identical 的并行度上限 = `ceil(n1/32) × M`**（一个 block 必须独占一个 32 行 scale block，amax 树的前提）⇒ **m=1 时只有 9 个 block**（6% SM），而 phase-2 有 160 个 **在 grid barrier 上空转** | **phase-1 加 K-split**：`grid.x = n1t·nsr·ks`（`ks` 运行期参），block 走 `k1/ks` 切片 ⇒ 9·M·ks 个 block；partial 写 `[M][n1][ks]` scratch，按 **`ck` 升序**做 reduce（非结合 fp 加法，序即契约） | −0.3~−0.8 | **2** | 低 | 无（独立）；⚠️ **非逐位**（f32 结合序变） |
| **L4-3** | `tc5::mxf4::expert_tcgen05_gateup_mxf4_kernel` / `m4_launch_gateup`（+ `tc5::e4::*`、`tc5::e4x::*`） | `grid=(rows/128=5, slots=6)` = **30 CTA**，TMEM 钉死 ≤2 CTA/SM ⇒ 15 SM；kRing=8 只够 840 KiB in-flight（BDP 的 15%） | **加第三 grid 维做 K-split**：`grid=(5, slots, ks)`，block 走 atom 切片；partial 写 `[tile][slot][ks][2·inter]`，**升序 ck reduce**。源码已在 `:3694 / :4373 / :5115` 标了三处 `[K-SPLIT TODO]`，kernel 的 ring **不用改** | −1.0~−3.0 | **3~4** | 中 | **L2：tcgen05 gate/up 已落地**（否则归 L2 账） |
| **L4-4** | **新核** `tc5::down::*`（不存在；对照 `expert_gemv_fp4_down_reduce_kernel`） | down 仍是纯 SIMT：`grid=(dim/warps=640, 1, rows)`，**每 warp 读 320 float 激活 = 80 条 LDS.128，是权重 LDG 的 ~8×**；mode 3（56 reg）会掉到 1.51 waves | **down 的 swapAB tcgen05**：`D[M=权重行 dim=5120, N=激活列 8] = A[W2 5120×k=320] × B[act 320×8]` + **fused 升序 slot reduce**。M 侧 5120 整除 128（40 tile），`grid=(40, slots)` = 240 CTA | **−2.0~−2.5** | **3~4** | 中低 | L4-3 的形状定稿（无骨架，从零写） |
| **L4-5** | `e4m3_gemm_grouped_kernel`（`tc5::e4x`） | `1-CTA kind::f8f6f4` 的 M 硬件固定 128，而每个 expert 只占 1~3 行 ⇒ **~120× 过量 MMA**（m=1 时最差） | ①grouped mask 把多 assignment 填满 128 行（`grp_m_cap` 重排）；或 ②接受过量，靠 L4-3 的 CTA 数掩盖。**不是改 tile** | 0~−1.0 | 2~3 | 低 | L4-3 的实测（先用实测决定是否值得投） |
| **L4-8** | `hc_dots_late_kernel`（`DSV41_HC_DL_KCHUNK`）+ `gemm_fp8_mrows` 的 P2/P3 | `hc_dots_late` 一发搬 3.84 MB，其中 **1.92 MB 是纯冗余**（每个 block 把同一份 x 行拖一遍）；整份 staging (~160 KB) ⇒ **1 block/SM** | ①翻 `DSV41_HC_DL_KCHUNK=1`（默认 OFF，启动器自带 `chunk % lcm(96, mix*8)==0` 对齐约束）⇒ 48 KB 双缓冲 + cp.async 提前发下一 chunk ⇒ **4 blk/SM**；② `gemm_fp8_mrows` 的 warps 与环深**同时**上（v19/v21/v24 的铁律） | −0.2~−0.3（hc）+ −0.3~−0.5（投影族） | 1.5~2 | 中 | 无 |
| **L4-9** | `dsv41_hc_collapse_norm_kernel` / `dsv41_rmsnorm_rows_kernel`（K9/K10） | **grid = `rows` = 1（m=1）→ 1 个 CTA、1 个 SM**；1024 thr ⇒ 32 warps 在 1 个 SM 上、其余 147 个空转。这是 A1 融合后**唯一没被摊开**的一环（把 2 发折成 1 发，但还是 1 个 block） | **把 `rows` 折进 grid 的第二维不够——它已经是 1**。真正的解：**在 kernel 内把 `hc*dim` 维摊开**（一个 `(row, hc)` 一个 block，`grid=(hc, rows)=(4,1)=4` CTA）或把 `dim` 切成 2~4 段（**K-split 的变体，非逐位**）。可行性要单测 | −0.1~−0.3 | 1.5~2 | 低 | L1（A1 接线）先落地 |

### 2.2 MLP（multi-level parallelism，2 项）

| # | kernel / 落点 | 当前问题 | 优化方法 | 预期节省 | 人日 | 把握 | 前置 |
|---|---|---|---|---:|---:|---|---|
| **L4-6** | **发射层**：`chain_dev.rs::layer_rows` / `attention_rows` / `moe_rows`（**无新 kernel**） | **verify 的 `layer_rows` 全程单流**——EAGER `layer()` 的三条 fork/join 一个都没接：`dual_chain`（q 链 vs kv 链）· `compress_side`（compressor 4 发）· `moe_dual`（routed vs shared）。三个 devrt 原语**已在**（`device.rs:1357-1437` 的 `dual_chain_fork/join`、`compress_side_fork/join`） | 把三条 fork 按 `layer()` 的现有形态接到 `layer_rows`/`attention_rows`/`moe_rows`；`moe_dual` 复用 `dual_chain_fork`。**风险**：m 行块的 stream 顺序 + CUDA graph 捕获（fork/join 必须保持 `cudaEventRecord`/`WaitEvent` **相邻配对**） | −0.5~−1.5 | **3~4** | 中 | L4-7（hc 侧流共用同一批原语） |
| **L4-7** | `hc_front_split` / `hc_mix_dots_kernel` / `hc_mixes_tail_kernel` / `hc_dots_late_kernel`（K6~K8） | **hc_mixes 在 m=1 只有 1 个 CTA**；`sinkhorn` 是 **6.46 µs 的 warp0 串行链**，同核内没有等长窗口可藏（实测否决过「藏进 collapse」：窗口 0.46 µs ≪ 6.46 µs） | **接 A2（`DSV41_HC_FRONT_ROWS=1`）**：`layer_rows` 的两处 `hc_mixes` 改走 `hc_mixes_auto` ⇒ EARLY collapse/norm 折进去 + **dots 网格从 5 块变 `mix × rows`**（m=1 即 24 块）+ **dots/LATE 移到 `dl` 流**（遮蔽 sinkhorn）。`truncate=false` 已在 `chain_dev.rs:8967` 修好 | **−1.3~−1.7** | **0.5~1** | **中高** | 无（A1/A2 代码全在树里） |

> **L4-7 与 L1 的账务边界（不得相加）**：A1/A2 的**折核本身**（400 发 → 240 发）记在
> `lazy-verify-optimization-path §3-L1`（−2.9~4.2ms/步，×k_emit）。**L4 自己的增量 =
> 侧流遮蔽 sinkhorn + dots 网格 (5→24/120 块)，≈ −0.6ms/步**。本表写 −1.3~−1.7 是**A1+A2 的总账**，
> 引用时必须按 §5 的分栏处理，否则与 L1 重复计数。

### 2.3 明确不做（记账用）

| 项 | 为什么不做 |
|---|---|
| **K13 AR v5 的占用** | 已是 2-kernel（`store` + `pubred`），`store` 网格 `(20, world)=160`、`pubred` 20×64thr。它不是占用问题而是**协议地板**（17.3 µs/轮），减轮数才行——而那要 accept 改善，不是 L4 |
| **`DSV41_HC_MIXES_SPREAD`** | 它**非逐位**：同会话 A/B 实测 31.44 vs 32.07ms 且**答案发散**（'2' 变跑题）。**是诊断臂，不是落地臂** |
| **`tc5::e4x` 的 L5 流水（kRing TMA）** | 属 L5（`l4-l5-kernel-path §2.1`），L4 只碰 tile 与 CTA 数 |
| **段核 / persistent（`hc_front_persist*`）** | P1d 实测 **+3.3ms 回归**；`hc-merge` 的 ticket-spin 也 +3.2ms。L5-7 已判「不推荐」 |
| **`gemm_fp8_gemv_kernel` 的双缓冲** | 注释已把陷阱写死：多一个 buffer 槽 = 少 1 block/SM。属 L5，且需与 K-split 同批 |

---

## 3. 与 lazy verify 的交互 —— **L4 在 m=1 下有效吗？**

> 这是本设计的核心问题。**结论：L4 与 B 类（mrows fold）的收益源不同——B 类是 m 驱动，
> L4 是 n / CTA 驱动。lazy 的 m=1 杀死 B 类，杀不死 L4；对权重主导的族，L4 的收益在 lazy 下
> 反而被 ×k_emit 放大。**

### 3.1 一个必须先钉死的量化事实

```
lazy  c_row = 18.11 / k_emit(2.214) = 8.18 ms/行        （实测，lazy-verify §1.1）
batched c_row = 37.31 / 5           = 7.46 ms/行        （实测，arch-floor §1）
⇒ 每行的代价从 m=6 到 m=1 只降 ~10%
```

**这条数字的含义**：verify 的 per-row 族（shared/routed/proj/gate/…）是**权重主导**的
（每 assignment 2.61 MB 权重 vs active 侧 kB 级），所以 **kernel 的每次代价几乎与 m 无关**。
逐条对照 `lazy-verify §4.2` 的「权重读 k_emit 次（tiny-SIMT 下权重共享不省钱）」——同一个机理。

### 3.2 推论：L4 的收益在 lazy 下被 ×k_emit

```
设某 per-row kernel 每次耗时 T（m 无关，见 §3.1），L4 把它降到 T−δ。
   batched：每步 N_layer 次 ⇒ 省  N_layer · δ
   lazy   ：每步 k_emit · N_layer 次 ⇒ 省  k_emit · N_layer · δ     ← 2.214×
per-step 族（hc/AR，发数与 m 无关）：同样 ×k_emit（lazy-verify §2 的原始发现）
```

⇒ **L4 的 ms 账在 lazy 下要乘 ≈2.214**（对 per-row 与 per-step 族都成立）。
**这不是「只在 batched 有效」——恰恰相反。**

### 3.3 逐项判定表（m=1 是否有效）

| # | L4 项 | 靶子物理量 | m=1 有效？ | 理由 | lazy 下的倍数 |
|---|---|---|---|---|---|
| **L4-1** | mrows nwarps | `blocks = ceil(n/nwarps)` | ✅ 有效（**但被 L3 吸收**） | n 驱动，与 m 无关；**SH_PAIR_M=1 接管 shared expert 后靶子消失**，只剩 wkv（需调 crossover） | ×k_emit |
| **L4-2** | SH_PAIR phase-1 K-split | `p1b = n1t·nsr·ks` | ✅✅ **m=1 是最差情形** | bit-identical 并行度上限 = `ceil(n1/32)·M` = **9**（m=1）vs 54（m=6）；phase-2 恒 160 ⇒ **m=1 时 151 个 block 在 barrier 上空转** | ×k_emit |
| **L4-3** | tcgen05 gate/up K-split | `grid=(rows/128, slots, ks)` | ✅ 有效 | `rows = 2·inter_local`、`slots = topk`，**两者都与 m 无关** ⇒ 30 CTA 在 m=1/m=6 完全相同 | ×k_emit |
| **L4-4** | tcgen05 down 换核 | `grid=(dim/128, slots)` | ✅ 有效 | 同上（down 的 k=inter、n=dim 都由权重定） | ×k_emit |
| **L4-5** | e4x M=128 tile | 过量 = `128 / m_assign` | ❌ **反向：m=1 最差** | m=1 时每个 expert 只占 1 行 ⇒ **~128× 过量**（m=6 是 ~21×）。**只在 batched 有意义** | — |
| **L4-6** | verify 多流发射 | 流重叠窗口 | ✅ 有效 | **lazy 跑的就是 `layer_rows`（m=1）**——落点正是缺 fork 的那个函数；m=1 下与 EAGER `layer()` 同形，风险反而最低 | ×k_emit |
| **L4-7** | hc 侧流 + dots 网格 | `grid=(rows)` → `(mix, rows)` | ✅✅ **最有效** | m=1 时 `hc_mixes` 只有 **1 个 CTA**；A2 后 dots 变 24 个 block，sinkhorn 移到 `dl` 流 | **×k_emit（per-step 族）** |
| **L4-8** | hc_dots_late KCHUNK | smem → blocks/SM | ✅ 有效 | 与 m 无关（m=1 时 grid=24 仍是 24 个 block，只是每块 1→4 的 resident 不成瓶颈；收益转为延迟隐藏） | ×k_emit |
| **L4-9** | collapse_norm/rmsnorm 的 1-CTA | `grid=rows` | ✅✅ **m=1 是 1 个 CTA 的最坏情形** | 与 L4-7 同源：m=1 时 1 CTA / 1 SM | ×k_emit |

### 3.4 必须写进账的三条反例边界

1. **「mrows 族在 lazy 下恒为 0」不能外推到 L4。** 那句（`lazy-verify §0-7`）说的是 **mrows 的
   `m` 折叠**（把 m 行折进一发，m=1 时无东西可折）——**不是** mrows 的**占用旋钮**（n/nwarps 是 n 驱动）。
   两者物理量不同：前者是 launch 数，后者是 CTA 数。
2. **但 L4 的绝对收益在 lazy 下仍会缩水一部分**：m=1 时 per-row kernel 的 *active 侧* 工作量最小，
   而 active 侧（激活解码 + FMA）正是 L4 不修的那一半（L4 修的是权重侧并行度）。
   ⇒ §5 的节省表**按 batched 口径（m=6）给**，lazy 的折算另列。
3. **L4 与 L3（SH_PAIR_M）互斥的点只有一个**：shared expert 的 n=288。L4-1 的原靶子被 L3 接管后，
   L4-1 只剩 wkv（且需要调 crossover）。**顺序上 L4-1 应排在 L3 之后做，或与 L3 同一次 A/B 测量。**
4. **§3.2 的 ×k_emit 倍数按「族」计，不按「kernel 计」**，且各族的上界不同：
   - **per-step 族（hc 链 / AR）**：发数与 m 无关（`hc_mixes(..., m as i32, ...)` 把行数当 kernel 内的
     rows 维；`layer_rows` 每层调一次）⇒ **严格 ×k_emit**（`lazy-verify §2.1` 已论证）。
   - **per-row 族（shared / proj / gate / indexer / attention）**：lazy 逐行 `step_rows(m=1)` ⇒
     每层每行各调一次 ⇒ **×k_emit**，但每行的 active 侧工作量最小（§3.4-2 的缩水项）。
   - **routed experts 的 tcgen05 臂**：`moe_rows` 的 routed 段**没有逐行外层循环**（读码：14200-15200
     内唯一的 `for` 是 `for slot in 0..topk`），但它消费的 `ex_act_b` 是 `[topk][...]` 布局、
     `act` 是「ONE shared quantised activation row」——**「m 行如何进入这一发」这一点静态读码未能完全钉死**
     （可能由 kernel 内的 rows 维或上游 pool 承担）。⇒ **L4-3/L4-4 的 lazy 倍数必须由上机（U1 的
     launch 计数）确定，不得按 2.214 直接乘。**

---

## 4. 优先级排序（ROI + 依赖）

### 4.1 ROI 单排序（ms / 人日）

| 排名 | 项 | 中位节省 | 人日 | **ROI（ms/pd）** | 把握 |
|---|---|---:|---:|---:|---|
| **1** | **L4-7** hc 侧流 + dots 网格 | 1.5 | 0.5~1 | **1.5~3.0** | 中高 |
| **2** | **L4-8** hc_dots KCHUNK + P2/P3 | 0.65 | 1.5~2 | 0.33~0.43 | 中 |
| **3** | **L4-4** tcgen05 down 换核 | 2.25 | 3~4 | 0.56~0.75 | 中低 |
| **4** | **L4-3** tcgen05 gate/up K-split | 2.0 | 3~4 | 0.5~0.67 | 中 |
| **5** | **L4-9** collapse_norm/rmsnorm 摊开 | 0.2 | 1.5~2 | 0.10~0.13 | 低 |
| **6** | **L4-1** mrows nwarps + crossover | 0.4 | 1 | 0.4（**但 L3 后 →0**） | 中低 |
| **7** | **L4-2** SH_PAIR phase-1 K-split | 0.55 | 2 | 0.28 | 低 |
| **8** | **L4-6** verify 多流发射 | 1.0 | 3~4 | 0.25~0.33 | 中 |
| **9** | **L4-5** e4x M=128 tile | 0.5 | 2~3 | 0.17~0.25 | 低（batched only） |

> 单看 ROI，L4-7 是碾压性的（**0 代码 + ×k_emit**）；L4-4/L4-3 的绝对量大但成本高、把握低。
> **不要按 ROI 单调排施工**——L4-3/L4-4 有硬前置（L2 tcgen05 落地）且是**唯一能越过「5% 峰值」
> 那堵墙**的动作（`arch-floor §5.2`）。ROI 排序用于「先拿白捡的钱」，波次排序用于关键路径。

### 4.2 依赖感知的波次（建议）

| 波 | 内容 | 前置 | 人日 | 里程碑 |
|---|---|---|---|---|
| **W-L4a** | **L4-7 + L4-9 + L4-1**（零/低成本：A2 接线 + 两个 crossover 常量 + KCHUNK） | 无 | 3~4 | 先拿 −1.5~−2.3ms（lazy 下 ×2.2） |
| **W-L4b** | **L4-3 tcgen05 K-split**（第三 grid 维 + 升序 reduce）—— **L4 的最大单项** | L2 gate/up 落地 + 单层微基准门（gateup 22.2µs） | 3~4 | routed 从 ~3ms → ~1.5ms |
| **W-L4c** | **L4-4 tcgen05 down 换核**（从零写） | W-L4b 形状定稿 + down 微基准门（17.2µs） | 3~4 | routed → 1.0~1.5ms（**L4 达成**） |
| **W-L4d** | **L4-6 verify 多流发射** + **L4-2 SH_PAIR K-split** | W-L4a（hc 侧流共用原语）+ L3 的 SH_PAIR | 5~6 | 关键路径遮蔽 + phase-1 满 SM |
| **W-L4e** | **L4-5 e4x tile**（**先用 W-L4b/c 的实测决定是否投**） | W-L4c 实测 | 0~3 | 只在落点 >2ms 时才做 |

**关键路径**：`L2 tcgen05 gate/up → L4-3 K-split → L4-4 down → (L5-3 e4x kRing)` 是唯一串行主干
（~11~15 人日）；`L4-7 / L4-1 / L4-8 / L4-6` 可并行。

---

## 5. 预期节省分解（L4 的 −5~8ms 从哪些 kernel 来）

> **口径**：batched（m=6）视verify 段。lazy 下的折算见 §5.2。

### 5.1 主表

| 来源 kernel | L4 项 | 低 (ms) | 高 (ms) | 占 L4 收益 | 性质 |
|---|---|---:|---:|---:|---|
| `tc5::mxf4::*` gate/up | L4-3 K-split | 1.0 | 3.0 | 17~25% | 删 μop 后的带宽兑现 |
| **`tc5::down::*`（新）** | L4-4 down 换核 | **2.0** | **2.5** | **34~21%** | ← **单项最大** |
| `hc_front_split` / `hc_dots_late` | L4-7 + L4-8 | 1.5 | 2.5 | 26~21% | 占用 + 侧流遮蔽 |
| 发射层（`layer_rows`） | L4-6 | 0.5 | 1.5 | 9~13% | 关键路径遮蔽 |
| `gemm_fp8_sh_exp_pair_kernel<M>` ph-1 | L4-2 | 0.3 | 0.8 | 5~7% | phase-1 满 SM |
| `gemm_fp8_mrows_kernel<M>` | L4-1 | 0.2 | 0.6 | 3~5% | CTA 覆盖 |
| `tc5::e4x::*` | L4-5 | 0.0 | 1.0 | 0~8% | tile 利用（batched only） |
| `hc_collapse_norm` / `rmsnorm_rows` | L4-9 | 0.1 | 0.3 | 2~3% | 1-CTA 摊开 |
| **合计** | | **5.8** | **11.9** | | |

**与任务口径对账**：任务/路线图给 L4 = **−5~8ms** ⇒ 落在本表的**中低段**（5.8~8）。
取「把握 ≥ 中」的项（L4-3/4/6/7/8/1 = 5.2~10.4）≈ **−5.2~8.0**，与路线图吻合。

**三条必须写在账上的分解事实**：

1. **`L4-4 + L4-3` = L4 收益的 ~50~60%**。**其余七项（占用/MLP）加起来只有 −2~4ms。**
   这正是 `arch-floor §5.2` 的「先换核、再上 warp」——**L4 的成败集中在 routed experts 的 tcgen05 收尾**。
2. **若把 tcgen05 收尾（L4-3/L4-4，6~8 人日）划归 L2**，L4 的**纯占用/MLP** 部分只有
   **−2.8 ~ −5.3ms / 8~13 人日**。本表的完整性取决于这条账务归属，引用时必须声明。
3. **L4-7 的 −1.5ms 与 `lazy-verify §3-L1` 的 −2.9~4.2ms 部分重叠**（同一次 A1/A2 接线）。
   合并时**取 L1 的数，L4 只留 −0.6ms**（侧流遮蔽增量）。§5.1 已按「L4 全账」列，不再重复。

### 5.2 lazy（m=1）下的折算

```
L4 收益（lazy） ≈ Σ_family  k_emit · N_layer · δ_family
               ≈ 2.214 × (per-row 与 per-step 族的 L4 增量)          （§3.2）
但 per-row 族的 active 侧在 m=1 最小 ⇒ 折算系数 < 2.214。
保守取 1.5~2.0×：本表的 −5.8~11.9 ⇒ lazy 下 −8.7 ~ −23.8（上界过乐观）
```
⇒ **不要用折算值入预算**。正确用法：**L4 的绝对值按 §5.1（batched 口径）编，lazy 的 ×k_emit
是「为什么要把 L4 排在 lazy 上」的理由，而不是「L4 值 2×」的承诺。**

---

## 6. 上机取证与验证计划（L4 若无实测会偏 ±5ms）

| # | 会话 | 内容 | 判据 | 面向的 L4 项 |
|---|---|---|---|---|
| **U1** | nsys（1，nccl 追踪） | 按 **kernel 名聚合** verify 段：数 `gemm_fp8_mrows_kernel<M>` / `gemm_fp8_sh_exp_pair_kernel<M>` / `expert_*` / `hc_*` 的 **launch 数 + 每发 µs + wave 数** | 每族的 **CTA 数/148** 与尾波占比 | 全部（L4-1/2/9 的唯一输入） |
| **U2** | 单层微基准（1） | `scripts/dsv41_tcgen05_mxf4_verify.sh --step 3` 的 gateup 22.2µs / down 17.2µs | **不达标即止损**，不进集成 | L4-3 / L4-4 |
| **U3** | GPU parity（1） | `tests_tcgen05_mxf4.cu`（已 GPU 验 EXACT）在当前 checkpoint 尺度下复跑；`tc5::e4x` 的两条 `[OPEN]` | 逐位 / maxdiff=0 | L4-3 / L4-5 |
| **U4** | K-split parity（1） | 升序 ck reduce 与单发 reduce 的**逐位**对照（**这是 L4-2/3/4 的共同风险**） | 若无法逐位，走容忍度 A/B（四段文本 + `faults=0`） | L4-2 / L4-3 / L4-4 |
| **U5** | 侧流 A/B（1） | `DSV41_HC_FRONT_ROWS=0/1` 同会话背靠背；`DSV41_HC_DL_KCHUNK=0/1` | `verify=` 位移 + **零拉丁** + `faults=0` | L4-7 / L4-8 |

**铁律**（`dspark-correctness-chain` 会话教训）：
① 同一远端同时只有一个测试驱动；② 每个 gate 翻转**单独 commit + 读回确认**（R6 陷阱：
gate 设了但没生效）；③ `cargo check --workspace` 是本地硬门禁；
④ **nsys 按 kernel 名聚合**，不要只看 `verify=`（`verify-ms-breakdown §修正` 的教训）。

---

## 7. 诚实校准（必须写在账上的五件事）

1. **L4 的所有 ms 都是设计口径，仓内没有任何一项有实测背书**（`l4-l5-kernel-path §5.1`）。
   反向证据：`{SH_EXP_MROWS, VERIFY_GRAPH, VERIFY_ROPE_MROWS, DRAFT_P3A}` 全开只 **−1.21ms**
   （预期 −24）；`SH_EXP_MROWS` 两次实测零收益。**instruction-bound + 低占用的墙，
   靠「只动一个因子」推不动**（v17→v21 四变体全中性）。
2. **§3.2 的「×k_emit」是结构推论（§3.1 的 c_row 只差 10% 是实测锚点），不是实测 A/B。**
   本文件把它当作**理由**（L4 该排在 lazy 上），不当作**收益承诺**。
3. **L4-4 是从零写的新核**（`tc5::down` 不存在），无骨架、无 GPU 证据；L4-5 的
   `120× 过量`若不解决，tcgen05 的落点会是 2~3.5ms 而非 1.0~1.5ms ⇒
   **建议先做 L4-3/L4-4 拿实测，再用实测决定 L4-5 是否值得投**。
4. **L4-1 与 L3 互斥**（shared expert 的 n=288 被 SH_PAIR 接管）：若 L3 落地，L4-1 只剩 wkv，
   收益趋近 0。**不要按「−0.6ms × 2」编 L4-1 的预算。**
5. **本表把 tcgen05 收尾（L4-3/L4-4）显式列在 L4 内**，与 `arch-floor §8` 排名 1（记在 L2）冲突。
   两处口径**不得相加**——同一批 6~8 人日只算一次。

---

## 8. 对既有文档的修正（读码结论，非估计）

| 处 | 原文档 | 本文件 | 依据 |
|---|---|---|---|
| `l4-l5-kernel-path §1.1 L4-1` | 靶子 = shared expert n=288，`MROWS_SMALL_N_ADAPTIVE` 是 head-line | **保留但降级**：该靶子在 L3（SH_PAIR_M=1）落地后消失；真正的剩余是 **wkv n=512 需要调 `kMrowsSmallN2: 512→640`** | `chain_dev.rs:11746` 的 first-try arm 顺序 + `dsv41_kernels.cu:3869-3877` 的 crossover |
| `l4-l5-kernel-path §1.2 L4-6` | 「EAGER 的三条侧流迁到 verify」 | 确认且**量化了落点**：`layer_rows` 全程单流（8869-12400 内无 `dual_chain`/`compress_side`/`moe_dual` 引用），原语在 `device.rs:1357-1437` | 读码 |
| `l4-l5-kernel-path §2.2 L5-6` | 「AR = 5 block（n4=1280→5×256，5 SM）」 | **已过时**：HEAD 是 `ferrite_ar_v5_block_threads`=64 + `grid_blocks`=20 ⇒ **20 block × 64 thr**（store 是 `(20,world)=160`） | `ferrite_kernels.cu` 的两个 static inline |
| `lazy-verify §0-7` | 「mrows 族在 lazy 下恒为 0，不要投」 | **对 mrows 的「m 折叠」成立，对「n/nwarps 占用旋钮」不成立**（后者是 n 驱动） | §3.4-1 |
| `l4-l5-kernel-path §1.1 L4-2` | 只提 K-split | 补一条**结构性判词**：bit-identical 的 phase-1 并行度上限 = `ceil(n1/32)·M` ⇒ **m=1 恒 9**，除了 K-split 无机械解 | `sh-pair-template-m-design §3.4` + 读码 |
| （新增） | — | **L4-9**：`hc_collapse_norm` / `rmsnorm_rows` 在 m=1 也是 **1 CTA / 1 SM**——A1 融合后的漏网之鱼 | §1 K9/K10 + `dsv41_kernels.cu:9107 / :8685` 的 `<<<(unsigned)rows, ...>>>` |

---

*工部 · 只读分析 + 本文件（唯一产出）；未执行任何 GPU 命令、未改动任何源码。*
*所有 kernel 名以 `__global__` / `extern "C"` 符号为准；所有 ms 均为**设计口径**（无实测）。*
