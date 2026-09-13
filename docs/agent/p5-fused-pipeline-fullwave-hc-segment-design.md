# P5 设计：融合核内 cp.async 流水 + 满 wave + hc 段核

> 中书省 · 2026-09-13 · **纯设计 / 只读代码分析，未执行任何 GPU 命令、未改动任何源码**。
> 任务：把 P5（roadmap 口径 verify ~10-11 → ~7ms，−3~4ms）从「未验证假设」变成**可施工的设计 + 可判决的实验**。
> 基线：工作树 HEAD（`dsv41_kernels.cu` 现场核对，行号以**函数名/符号**为准——本仓有行号漂移史）。
> 输入判决：`verify-amortization-lesion-audit.md` §2/§9/§10.9 · `mrows-mpar-design.md` ·
> `mrows-mtile-design.md` + `mrows-mtile-fix.md` · `mrows-l2-bcast-design.md` ·
> `tensorcore-proj-design.md` §5 · `hc-chain-mrows-audit.md` · `hc-chain-bandwidth-analysis.md` ·
> `dsv41-persistent-arch.md` · `l4-l5-kernel-path.md` §2 · `l4l5-next-batch-implementation-plan.md`。

---

## §0 三条必须先说的判决（其中一条推翻任务前提）

**判决 1（纠正任务前提）：第三条路已经在树里了，而且已经有三条实现 + 一次实测。**

任务书说「第三条路 = M 真并行 + 权重只 stage 一次」是「待设计」。现场读码：

| 臂 | kernel | 载体 | M 的轴 | 每元素指令（M=6） | 实测 |
|---|---|---|---|---|---|
| legacy | `gemm_fp8_mrows_kernel<M>` :5616 | M-in-register（`float acc[M]`） | warp 内串行 | 1.00× | 5×/行（nsys） |
| ② MPAR | `gemm_fp8_mrows_mp_kernel<M>` :5925 | **M 作 warp 轴** | warp 布局 | 1.59× | **−0.52 / −1.28ms 二连败** |
| ③ ⑤a | `gemm_fp8_mrows_l2_kernel<M>` :6185 | M 进 grid、无 smem | grid 维 | 1.00× | **四档全负** |
| **④ MTile** | `gemm_fp8_mtile_kernel<M,AQ>` :6402 | **M 作 register tile 的行轴** | warp 内 tile | **0.65×(bn=2) / 0.47×(bn=4)** | **g2 版 +2.54ms（bn=1 对照），fix 版未上机** |

⇒ 用户要的「第三条路」的**骨架已落地**（`mrows-mtile-design.md` + `mrows-mtile-fix.md`），但：
* **MTile 的 fix 版（激活回 smem staging + per-warp slab/LUT）从未跑过 GPU**（fix 文档 §0 自陈「GPU 未验证」；
  audit §10.9 的 A/B 表里没有 mtile 臂）；
* **g2 版的判决是负向**，且负向被锁在两处**结构性**病灶上（不是指令数：bn1→bn4 砍半指令只换回 0.68ms）。
* ⇒ **P5 的第一件事不是写新核，是把已落树的 MTile fix 送上机**。下面 §2/§3 的设计都以「fix 版为正结果」为前提，
  并给出负结果时的备用腿（§3.5 mma 路线）。

**判决 2（新发现的、账上没有的一项成本）：MTile 把 **激活的 prologue 复制** 做成了新的头号 L2 项。**

MTile 的权重侧确实做到「只 stage 一次」（每 warp 只 stage 自己的 `bn` 行，`dsv41_kernels.cu:6493-6508`），
但**激活侧是 block-wide stage**（`:6456-6477`），而 block 只有 `nw` 个 warp、覆盖 `nw*bn` 个输出行
（`:6424`）。于是**每个 block 都要把整份 m 行激活（`m*k` + `m*nb_k*4`）搬进自己的 smem**：

| 形状 | n×k | m | nw(bn=2) | grid | smem 分解（权重 / LUT / 激活） | **激活 L2→smem 总量** | 权重 DRAM |
|---|---|---:|---:|---:|---|---:|---:|
| **wkv** | 512×5120 | 6 | **1** | **256** | 10.2 KB / 1 KB / **33.75 KB** | **8.85 MB** | 2.62 MB |
| wq_a | 1280×5120 | 5 | 4 | 160 | 40 KB / 4 KB / 28.1 KB | 4.6 MB | 6.55 MB |
| wq_b | 4096×1280 | 6 | 8 | 256 | 20 KB / 8 KB / 8.4 KB | 2.2 MB | 5.24 MB |
| wo_b | 5120×1024 | 5 | 8 | 320 | 16 KB / 8 KB / 5.6 KB | 1.8 MB | 5.24 MB |

- **wkv 是重灾区**：`dsv41_mrows_mtile_warps_for`（:6655）的 coverage 规则 `nw = n/(bn*SM)` 在 n=512 时把 `nw` 压到 **1**
  ⇒ 每个 block 只有 1 个 warp，却要付 33.75 KB 的激活 prologue，**激活 staged 量 = 权重的 3.4×**。
- 对照 legacy：wkv 是 `nwarps=4`、`fold_r=m` ⇒ grid=128、激活总量 4.42 MB（**仍是 MTile 的一半**）。
- **这正是 MPAR 的死因 (a)（prologue 按 grid 复制）在激活侧的复现**，而 MTile 的指令账表（`mrows-mtile-design.md §2.2`）
  只数了「每 warp 每 kb 指令」，**没有这一项**。bn=1 对照臂的 +2.54ms 里有没有它的份额，是必须由 A/B 分离的。

**判决 3（结构上限，决定「满 wave」能做什么、不能做什么）：bit-identical 约束下，投影族的 warp 数被 `n` 钉死。**

legacy 的总 warp 数 = `nt × nwarps = n`；MTile = `n/bn`；MPAR = `n·M`（唯一能填满机器的，代价 1.59× 指令）。
`k` 不产生并行度（`nt` 只来自 `n`，audit §2 的原判词成立）。于是：

| 形状 | n | legacy warps | MTile warps(bn=2) | warps/SM（148） |
|---|---:|---:|---:|---:|
| wkv | 512 | 512 | 256 | **1.7-3.5** |
| sh w1/w3 | 288 | 288 | 144 | **1.0-1.9** |
| wq_a | 1280 | 1280 | 640 | 4.3-8.6 |
| wq_b | 4096 | 4096 | 2048 | 13.8 |
| wo_b | 5120 | 5120 | 2560 | 17.3 |

⇒ **小 n 形状（wkv 512 / sh 288）在逐位约束下永远填不满 148 SM**。要填满只有两条路，都必须付代价：
1. **K-split**（把 k 切成 ks 份 → grid ×ks）：并改变部分和顺序 ⇒ **非逐位**（`dsv41-persistent-arch.md §1` 的
   「split 必须 = 1」铁律；`hc_front_persist_mb` 的实测先例）；
2. **tensor core**（`mma.m16n8k32` swapAB，grid=(n/16)·ks）：同样非逐位，但**内部逐位**（`tensorcore-proj-design §2.4.1`
   的 program-consistent parity）——这是唯一能同时「填满机器 + 保住行自洽」的路。

⇒ **本设计把「满 wave」拆成两问**：
* **P5-B（逐位，今天可做）**：把 grid 从「<148 的一波」改成「≥148 且无尾波」+ 把 prologue 从 per-block 变 per-launch
  （§3.2），**不追求填满 SM**，追求**消除波次/尾波浪费与暴露的 prologue**；
* **P5-E（换程序，需要 parity 门禁）**：mma/ks 路线，**唯一**能填满小 n 的路（§3.5，骨架已编译通过）。

---

## §1 当前 kernel 的完整分析（交付 ①）

### 1.1 `gemm_fp8_mrows_kernel<M>`（`dsv41_kernels.cu:5616-5833`）

**warp 组织**
```
__launch_bounds__(256)；blockDim = nwarps*32（nwarps = dsv41_mrows_warps_for(n)：n<2048 → 4，n≥2048 → 8）
grid  = nt × ng          nt = ceil(n/nwarps)  ng = ceil(m/fold_r)（fold_r 默认 = m ⇒ ng=1）
it = blockIdx.x % nt ; r0 = (blockIdx.x/nt)*fold_r ; row = it*nwarps + warp   （:5633-5700）
active = row < n
```
* **warp → 一个输出行**；该 warp 独立折叠**全部 M 个激活行**（`float acc[M]`，:5747）。
* M 不进 grid（fold_r 门 OFF 时）；M 也不进 warp 布局 ⇒ **M 完全落在单个 warp 的寄存器里**。

**寄存器 / 累加器布局（a32=1 臂，:5777-5799）**
```
for kb in 0..nb_k-1:                       # 升序，j = kb*32 + lane
    sb  = ue8m0_to_f(wsr[kb])              # ← 1 LDG（权重 scale 行）
    wv  = s_lut[s_w[warp*k + j]] * sb      # ← 2 LDS（权重字节 + LUT 项）+ 1 FMUL
    af[M]  : af[q] = s_lut[s_a[q*k+j]] * s_as[q*nb_k + (j>>5)]   # ← 3M LDS + M FMUL
    acc[q] += af[q] * wv                   # ← M FFMA（q=0..M-1）
```
* 每 (warp, kb) 指令数 = `5M + 4`（M=6 → **34**；`mrows-mtile-design §2.2` 的表）；其中 **24 条（71%）是激活侧**
  ——而激活只有 M=6 行、对全部 `n` 个 warp 是同一份数据 ⇒ **这 24 条被 n 个 warp 各付一遍**。
* 归约：每个 (warp, q) 一棵 `shfl_xor` off=16,8,4,2,1（:5822-5831），lane0 加 bias 写 `out[(r0+q)*out_stride + row]`。
* ptxas/Ncu：**40 regs**，`__launch_bounds__(256)` 下寄存器侧 6 blocks/SM。

**smem 布局（:5643-5648）——legacy 的三段式**
```
s_w   : nwarps * k               字节   每 warp 自己那一行的权重（cp.async16 单缓冲，:5730-5739）
s_lut : 256 * 4 = 1 KB                   e4m3 解码表（block-wide 一份，:5648）
s_as  : fold_r * nb_k * 4              激活 scale 行（block-wide）
s_a   : fold_r * k                     激活 fp8 行（block-wide，1b 后走 cp.async16，:5669-5691）
```
wkv（n=512,k=5120,m=6,nwarps=4）⇒ `4*5120 + 1024 + 6*160*4 + 6*5120 = 54.75 KB` ⇒ **smem 侧 4 blocks/SM**。

**PROLOGUE 顺序（legacy）**：`issue 激活（block-wide）→ wait_all → __syncthreads → issue 本 warp 权重行
→ （跨 barrier 在飞）→ 各 warp wait_all 自己那组 → consume`。
⇒ 权重组的 DRAM 往返**被 barrier 覆盖**；激活组是 block-wide 且**在 barrier 前退休**。

**为什么 m=6 退化成 latency-bound（数值对齐）**
* 总指令（wkv, M=6, n=512, nwarps=4, grid=128）：`34 × 512 warp × 160 kb = 2.78M warp-inst`
  ⇒ 148 SM × 4 scheduler ≈ 592 issue/cycle ⇒ **≈ 4.7k cycle ≈ 2.4 µs** 的 instruction-bound 地板。
* DRAM 地板：2.62 MB / 8 TB/s ≈ **0.33 µs**。
* NCU：DRAM 0.67-0.79%、Compute 12.8-13%、L1 14.1-14.6%、**占用 13.4%** ⇒ **没有任何一条管线饱和**。
* 实测（nsys，`gemm_fp8_mrows<5>` avg）：**52.1 µs**，且 `52.1/5 行 = 单行 gemv 的 5×`。
* ⇒ **离指令地板 ~20×、离 DRAM 地板 ~160×**。**结论：既不是 instruction-bound，也不是 DRAM-bound，
  而是「不饱和」——这一点必须写进设计的赌注里**（§5 的第三种病）。

**⚠️ 口径缺口（P5 的判决必须先补的一条）**：`52.1 µs` 是**四个形状的平均**（wq_a/wkv/wq_b/wo_b），
`5×` 也是平均值口径。**无法从中读出「哪个形状是病灶」**——而 P5 的三条修法对不同形状的杠杆完全不同
（§3.2 的 coverage 账在 wkv 与 wo_b 上相差 10×）。⇒ §4.0 的 Step-0 是**按形状分解的 m=1 vs m=6 阶梯**。

### 1.2 `gemm_fp8_mtile_kernel<M, AQ>`（:6402-6614）——第三条路的现存实现

```
BM = kMrowsMtileBM = 8（≥ VERIFY_ROWS=6，2 的幂；RN = min(M,BM)，无 pad 计算）
nw = blockDim.x>>5（来自 dsv41_mrows_mtile_warps_for，:6655）
warp → n_block = warp ；block 拥有 row0 = blockIdx.x*nw*bn 起的 nw*bn 个输出行
每 warp 的寄存器 tile: acc[RN][BN_MAX]，BN_MAX = 4（:6520-6524）
consume（:6529-6598）：
   for kb: av[RN] = s_lutw[s_a[q*k+j]] * s_as[q*nb_k + (j>>5)]      # 激活 decode 一次，服务 bn 行
           for nn<bn: wv = s_lutw[s_w[(n_block*bn+nn)*k + j]] * sb   # 权重 decode，服务 RN 行
                      acc[q][nn] += av[q]*wv
归约（:6599-6612）：每 (q, nn) 一棵 legacy 的 shfl_xor 树，lane0 写 out[q*out_stride + row]
```
**smem**：`s_w = nw*bn*k` | `s_lut = nw*256*4`（**每 warp 一份**，:6437/:6483，故只需 `__syncwarp`）|
`s_as = RN*nb_k*4` | `s_a = RN*k`（后两者 block-wide，**仅 AQ==0**）。
**PROLOGUE 顺序**：`issue 激活(block-wide) → 每 warp 建自己的 LUT（cover）→ wait 激活 →
issue 本 warp 的权重（bn 行）→ __syncthreads（激活那个，唯一的 block barrier）→ warp-local wait 权重 → consume`。
* **fix 版的两处修**：激活回 smem + LDS（短依赖链）；权重 per-warp 发布（消掉 block 尺寸的串行点）。
* ptxas（fix 后）：M=6 → **76 regs / 0 spill / 1 barrier**；寄存器侧 **3 blocks/SM**；smem 侧 wkv bn=2 是
  44.75 KB ⇒ **5 blocks/SM** ⇒ **wkv 的 binding resource 是 smem（= 判决 2 的激活 prologue）**。
* 指令账：每 (warp,kb) = `3bn + 3M + (M+bn) + M*bn` = 44（bn=2, M=6）；**总指令 0.65× legacy**。

---

## §2 P5 的目标与判决门（先把「能不能做」钉死）

**P5 目标**：把投影族（nsys `gemm_fp8_mrows` 15.2%、投影族 15-21% 口径）从「M 行 = 单行的 5×」
压到「M 行 ≈ 单行的 1.0-1.3×」，从而在 verify 上兑现 **−3~4ms**；hc 侧再拿 **−0.3~0.5ms**。

**三条止损门（写死，不许事后放宽）**
1. **微基准门（第一步，最高优先）**：`tests_dsv41_gemm_mrows.cu` 的 `t(m=6)/t(m=1)` 比值。
   legacy 是 2~4×；**目标 ≤1.3×**。**不达标 ⇒ 不进 e2e**（MPAR 的教训：先看符号，再看幅度）。
2. **NCU 门（第二步）**：fix 版 kernel 的 `sm__throughput` 应从 ~13% 抬起；`short_scoreboard` 停顿应下降；
   `l1tex__data_pipe_lsu_wavefronts_mem_shared` 应显著降（LUT 不再逐 warp 建 256 项）。
   **若 sm__throughput 仍 ~13% 且停顿不降 ⇒ 判「第三种病」（§5.4），回报，不要调参。**
3. **双门禁（e2e）**：每臂同时报 `step_ms`（`[dspark] steps=`）**AND** `mean-k`（现基线 2.240）。
   掉了即弃该臂，不解释。

---

## §3 第三条路的设计（交付 ②：M 分配 + 权重 staging + 计算组织）

### 3.1 M 分配：**不要 warp 级 M 分配**（那是 MPAR，二连败）；M 必须是 **warp 内的 tile 行轴**

| 方案 | 形态 | 判决 |
|---|---|---|
| warp 级 M 分配（每 warp 一对 (输出行, 激活行)） | = MPAR `gemm_fp8_mrows_mp_kernel` | ❌ **实测二连败**（①每元素 1.59× 指令；②prologue 按 grid 复制） |
| **M 作 register tile 的行轴**（每 warp 拥有全部 M 行 × bn 个输出行的整块） | = MTile | ✅ **唯一降总指令数的路**（0.65×），且**权重服务 M 行 + 激活服务 bn 行**两笔复用同时成立 |
| M 进 grid（×M 个块，无 smem） | = ⑤a | ❌ 四档全负（×m 的 L1/L2 请求 + 块数爆炸） |

**设计定稿：M 分配 = MTile 的 (RN × BN_MAX) 寄存器 tile，`RN = min(M, BM=8)`，`n_block = warp`。**
（`BLOCK_M=8` 不是「算 8 丢 2」——`RN=M` 时 pad 行不计算，见 :6413 的注释。）

### 3.2 权重 staging：**「只 stage 一次」要重新定义成「每字节每 launch 只被搬一次」——激活侧现在不满足**

**现状（fix 版）**：权重 ✅（每 warp 自己的 bn 行，总字节 = `n*k` = 1×）；**激活 ❌（每 block 一份 `m*k`，
总量 = `grid × m*k`，wkv 上是权重的 3.4×）**。

**修法 A（首选，逐位安全）：把「每 block 的激活 prologue」变成「每 launch 的激活常驻」——段核化的 grid-stride**

```
grid  = ceil(n / (nw*bn*T))          # T = 每个 block 内部的「输出行 tile」循环数
block = nw warps（nw 从 coverage 规则放开到 nw = 4..8，不再被 n/(bn*SM) 压到 1）
prologue : stage 一次激活（m*k + m*nb_k*4）→ 唯一的 block barrier
loop t = 0..T-1:                      # 每个 t 覆盖 nw*bn 个输出行
    t+1 的权重 slab 用 cp.async 预取（ping-pong 双缓冲）
    consume t 的 tile（weight slab 走 LDS、激活走常驻的 s_a）
```
* **收益账（wkv，bn=2）**：nw=4、T=2 ⇒ grid=32（<148，❌）；nw=1、T=2 ⇒ grid=128、激活 4.42 MB（= legacy 水平）；
  **nw=4、T=1 ⇒ grid=64**……**⇒ 小 n 上 T 与 coverage 直接对冲**（见 §3.4 的诚实结论）。
* **在 wq_b/wo_b 上账很正**：wo_b（n=5120, bn=2, nw=8, T=1 ⇒ grid=320；T=2 ⇒ grid=160，激活 1.84→0.92 MB）。
* **逐位**：只改「谁算哪个元素」与「字节从哪读」（拷贝宽度不可观测）⇒ C1-C6 全保持（`mrows-mtile-fix.md §3` 同款论证）。

**修法 B（逐位安全，把小 n 的激活 prologue 真正降下来）：cluster DSMEM 共享激活 slab**
* cluster 内 C 个 block 只由 rank0 stage 激活、其余 rank 经 `cluster.map_shared_rank` 读（`cluster-dsmem-design.md` +
  `tests_cluster_dsmem.cu` 是树内的原语）。激活 total = `grid/C × m*k` ⇒ wkv（grid 256、C=4）8.85 → 2.2 MB。
* 代价：DSMEM 跨 SM 的 LDS 延迟高于本地 smem（用于 *激活* 读、不用于 weight decode，风险可控）；
  cluster 占用受 `cudaOccupancyMaxActiveClusters` 约束。
* **这是「权重 stage 一次 + 激活也 stage 一次 + M 真并行 + 逐位」的唯一形态**（`tensorcore-proj-design §5.2` 认领的第三条路）。

**修法 C（非逐位，仅在 A/B 失败时启用）：K-split** — 见 §3.5。

### 3.3 计算组织：核内 cp.async 流水（双缓冲 ring）

现状是**单缓冲**：权重「issue →（barrier 覆盖）→ wait → consume」，**consume 期间没有字节在飞**。
（`l4-l5-kernel-path.md §2.1 L5-2` 的原判词；树内注释也写死「多一个 buffer 槽 = 少 1 block/SM」的陷阱。）

**设计：沿 k 的 chunk ring（ping-pong），consume 链不变、逐位安全**
```
KC    = k 的 chunk 数（chunk 取 512 B/行，wkv k=5120 → 10 chunks）
槽位  = (nw*bn + RN) × chunk × 2        # 权重 chunk + 激活 chunk，双缓冲
序    : issue(chunk 0) → commit → [issue(chunk1) → commit] → wait_group(1) → consume(chunk0)
        → issue(chunk2) …（每 consume 一个 chunk 之前只 wait_group(1)）
```
* **逐位**：consume 仍是**唯一一条升序 kb 串行链**（C1/C6），chunk 只决定「哪个字节何时到 smem」，
  不决定乘法顺序（拷贝宽度不可观测）。**这是「核内流水」在逐位约束下唯一合法的形态**（K-split 不是）。
* **smem 收益（wkv bn=2）**：44.75 KB → `(2+6)*512*2 + scale ≈ 9 KB` ⇒ **smem 侧 5 blocks/SM → 25 blocks/SM**，
  binding resource 从 smem 移回寄存器（76 regs ⇒ 3 blocks/SM @ 32 线程/块 = 26 blocks/SM）。
* **树内先例（可直接抄的机制）**：`dsv41_cp_wait_group1`（:13139，gemv P4 的「退休激活、留权重在飞」）与
  `dsv41_cp_wait_group2`（:13143，hc_dots_late 的 KCHUNK 双缓冲，`g_hc_dl_chunk` 768 float4 / 4 blocks/SM）。
* **⚠️ 与修法 A 的耦合**：ring 才是让 A 的 T>1 真正有收益的机制——**没有 ring，T>1 只是把暴露的 prologue
  从「每 block 一次」变成「每 tile 一次」**（MPAR 一败的死因原样复现）。**A 与 C 必须同批**。

### 3.4 满 wave：**诚实结论——小 n 在逐位约束下填不满，能做的是「消尾波 + 消暴露 prologue」**

| 形状 | 现状 grid（legacy） | 现状 warps/SM | MTile+T 的目标 | 能否「满」 |
|---|---:|---:|---|---|
| **wkv** 512×5120 | 128（0.86 波） | 3.5 | grid 调到 148/296 的整数附近 + 激活常驻 | ❌ 总 warp ≤ n/bn = 256 |
| **sh** 288×5120 | 72（0.49 波） | 1.9 | 同上（另见 §6：SH_PAIR 接管） | ❌ ≤ 144 |
| wq_a 1280×5120 | 320（2.16 波） | 8.6 | T=2 ⇒ 160 块、无尾波 | ⚠️ 半 |
| wq_b 4096×1280 | 512（3.46 波） | 13.8 | 尾波已 <1% | ✅ 已满 |
| wo_b 5120×1024 | 640（4.32 波） | 17.3 | 同上 | ✅ 已满 |

* **「128 blocks on 148 SM = 86%」的真相**：它不是「占用率病」，是**波的余数**——128 块跑一波，
  20 个 SM 空转一整波。真正的损失是 `(148-128)/148 = 13.5%` 的**这一波**，而 wkv 的一波只有 ~2-5 µs。
* **可做的**（P5-B，逐位）：把 `(nw, bn, T)` 解成 `grid = ceil(n/(nw*bn*T)) ∈ [148, 296] 或 |grid-148k| 最小`，
  并保证 `nw ≥ 2`（块内多 warp 互相藏延迟）。**但 wkv 上 T 与 coverage 对冲：T=2 ⇒ grid=128（0.86 波）**，
  ⇒ **wkv 的正解是把 nw 从 1 抬到 2、bn 保持 2、T=1 ⇒ grid=128 不变，但块内 2 warp**（waves 不变、块内 ILP 翻倍）。
* **填满 = 必须换程序**（§3.5）。

### 3.5 备用腿：tensor core（`mma.m16n8k32` swapAB）——唯一能填满小 n 的路

* **形态**（`tensorcore-proj-design.md §1`）：权重在 mma 的 **M=16**（满利用率）、**6 个激活行进 N=8 的列**
  ⇒ 利用率 6/8 = 75%（不是 pad-16 的 37.5%）；**每 MAC 指令数 0.177 → 0.0055（≈32×）**；
  grid = `(n/16) × ks`（wkv 用 ks=32 ⇒ **1024 warp**，wo_b 640）——**这就是满 wave**。
* **代价（不可论证掉的一条）**：**逐位不可能**（块内 32 项硬件求和顺序不可指定，§2.2 的证明）。
  唯一的救法是 **(b′) program-consistent parity**：m=1（eager/draft）**也**走同一个 mma kernel，
  于是「verify row r ≡ eager row r」逐位成立（D 的第 r 列只依赖 B 的第 r 列，§2.4.1），
  **前提是 ks 只是 (n,k) 的函数、绝不能是 m 的函数**。
* **现状**：骨架 `kernels/cuda/dsv41_proj_mma_skel.cu` 已落树、远端 **compile-only EXIT=0 / 8 个 M 特化 0 spill**；
  未并入 `build.sh`；Rust 侧 scratch（`swapab_part → [ks][M][n]`）**与 peer 改动区重叠，需协调**（ks=1 可完全绕开）。
* **树内先例**：`DSV41_SWAPAB`（M=1 decode 已用 tensor core，非逐位、用文本指纹验收）⇒ 换程序的纪律已建立。

### 3.6 三条路的排序（P5 的实施顺序）

| 顺序 | 项 | 逐位 | 预期 | 前置 |
|---|---|---|---|---|
| **①** | **MTile fix 上机（零代码）** + 按形状分解的 m=1/m=6 阶梯 | ✅ | 定符号（决定后面全部） | 无 |
| **②** | **A + C 同批**：grid-stride 常驻激活 + k-chunk ring | ✅ | −1.5~2.5ms（设计） | ①的符号 |
| **③** | **B**：cluster DSMEM 共享激活（小 n 专用） | ✅ | −0.5~1.0ms | ②的形状定稿 |
| ④ | **E**：mma swapAB + (b′) 成对 gate | ❌（内部逐位） | −3.5ms（天花板 −4.7） | ①-③ 后的实测；**独立臂** |

---

## §4 hc 段核设计（交付 ③）

### 4.1 现状（生产栈，`HC_FRONT_ROWS=1` 默认 ON）

```
verify（rows=m=6）/层：
  hc_front_split（:14454）→ 两次 launch（`DSV41_HC_DL_SIDE` 在时是两条流）：
     (1) hc_mixes_tail_kernel(EARLY)  grid=rows=6 块, block=1024, smem=512B    → collapse+rmsnorm+T1 fp8  (:14537)
     (2) hc_dots_late_kernel           grid=(mix,rows)=(24,6)=144 块, block=128 → dots + 选举的 LATE tail (:13970)
  hc_post：默认已折进 AR 的 epilogue（`DSV41_HCPOST_EPI` 默认 ON）
draft（rows=3）/层 ×3 mtp 层：**无条件走 raw `hc_mixes`**（`dspark_dev.rs` 直调，不受 HC_FRONT_ROWS 管）⇒ 6 发/步
```
* hc 族**每条 kernel 都已经是 rows-native**（`hc-chain-mrows-audit §2.1` 的签名级证据）⇒ **不需要 rows 化**，
  缺的是 **dispatch 覆盖率**；
* `hc_dots_late` grid=144/148 = **97%** ⇒ dots 侧已满；**EARLY 是 6 块（4% SM）**——这是 hc 侧唯一「spread 不足」的块。

### 4.2 四段（pre / collapse / post / tail）的融合与流水化设计

| # | 设计 | 数值性质 | 收益 | 判决 |
|---|---|---|---|---|
| **H1** | **消灭 draft 侧 6 发 raw `hc_mixes`**：开 `DSV41_P3LITE_HC_FRONT=1` / `DSV41_DRAFT_P3LITE_A_SEG=1`（核 `dsv41_draft_hc_front_kernel` 已在树，`dsv41_kernels.cu:11988` 起） | 逐位（row-independent，§2.1 论证） | **−0.31ms/步**（6 发 × 51.4µs，audit §5.4） | ✅ **首选**：零新核、前置是 draft parity（R1/R2/R3） |
| **H2** | **KCHUNK 重调 + 侧流**：`DSV41_HC_DL_KCHUNK`（默认 OFF、v22 略负）+ `DSV41_HC_DL_SIDE`（第四条流，:14515） | 逐位（构造上等价，:14116） | −0.2~0.3ms | ✅ 低风险，**先 A/B 重调 chunk（768→576 float4）再谈 flips** |
| **H3** | **把 EARLY 折进 dots 的同一 launch**（段核化）：单 launch 三角色（dots 块 / collapse 块 / 选举的 tail 块） | ⚠️ **非逐位** | 只省图节点（~1.5µs × 80 ≈ 0.12ms） | ❌ **不做**：block 尺寸冲突是硬阻塞——collapse 的 rmsnorm 归约树（`red[32]` 按 warp 数分组 + thread0 升序求和，:13345-13353）**依赖 blockDim=1024**；dots 用 128。改成统一 1024 会毁掉 dots 的并行度（或毁掉 collapse 的树）⇒ 换数值语义只值 0.12ms，不值。 |
| **H4** | **把 collapse 元素部分 dim-split 分块**（C=8 块/行 → +48 块） | ⚠️ 元素部分逐位，**rmsnorm 的 ss 跨块合并不逐位** | 0（ss 仍是一块/行，串行尾不变） | ❌ 不做（`hc_front_persist_mb` 的 P1d +3.3ms 已在同一形态上翻过车） |
| **H5** | **`hc_front_kernel`（`DSV41_HC_MERGE`，ticket 自旋）扩展到 rows=m** | ⚠️ **正确性隐患**（`hc-chain-mrows-audit §3`：grid=(25,6)=150 > 148 SM、1 块/SM ⇒ tail 自旋等 dot 块、dot 块等 SM ⇒ ~5s watchdog 后**静默错答**） | — | ⚠️ **先修正确性**：在 `dsv41_hc_front` 的 merge 分支加 `rows == 1` 前置（或改注释为事实）。**这是上报项，不是提速项** |
| **H6** | **把 hc_pre 折进前驱 AR 的 epilogue**（dots 的输入 = 前驱 hc_post 的残差） | 逐位（P1e 已证元素划分精确） | — | ❌ 不做：AR epilogue grid = `ceil(n4/256) = 5` 块 ⇒ 把 1.92MB 权重流压到 5 个 SM ⇒ 估算 +10µs/层 × 80 = **净亏 0.8ms**（`dsv41-persistent-arch §P1e` 已判） |

**hc 段核的定稿结论**：hc 族的**四段已经是「两条流 + 一个折进 AR 的 post」**，融合空间已被前面几轮吃干。
**P5 在 hc 侧只收 H1 + H2 = −0.5ms**；H3/H4/H5/H6 都不做（三条有数值/并行度硬阻塞，一条是正确性修复）。

---

## §5 与失败先例的差异论证（交付 ⑤）+ 预算（交付 ④）

### 5.1 逐条对照（每个先例的**实测死因** vs 本设计怎么避开）

| 先例 | 实测死因（原文口径） | 本设计为何不同 | 残余风险 |
|---|---|---|---|
| **fold_r**（6× 退化，63.8→10.3 tok/s） | M 进 grid ⇒ 每个 (i-tile, M 组) **重新 cp.async16 stage 同一份权重行到 private smem** ⇒ prologue 付 ng 次 | MTile **不把 M 放 grid**（且 launcher 强制 `fold_r == m`，:6758）；权重按 warp 自己的 bn 行 stage，总量 `n*k` = 1× | **激活侧的同类复制**（判决 2）——必须由修法 A/B 消掉 |
| **MPAR**（rpb=1 +0.52 / auto +1.28，二连败） | ① 每元素 **1.59× 指令**；② LUT+slab 按 grid 复制、prologue 完全暴露（首版 issue 后立刻 wait） | MTile **每元素 0.65×**（唯一 <1 的路）；LUT **每 warp 一份**（消 cross-warp 发布）；prologue 按 MPAR 的返工版顺序（issue→cover→wait→barrier） | 若 MTile 仍负 ⇒ 说明「既非 issue 也非 latency bound」（§5.4） |
| **⑤a L2-bcast**（四档全负） | **去掉 staging** ⇒ 每块直读 L1/L2、请求 ×m + 块数爆炸（3200-3840 块） | **保留权重 staging**（⑤a 的教训正确用法），只把激活的**读法**改成 LDS（fix 版） | 激活直读的诱惑必须抵抗（g2 的 bn=1 对照就是踩了这个：+2.54ms） |
| **g2 MTile**（bn=1 对照 +2.54ms） | **不是指令数**（bn1→4 砍半指令只换回 0.68ms），是两处结构：**(a) 激活 `__ldg` 直读 L1；(c′) 权重 slab 的 block 级发布** | fix 版把 (a) 换回 legacy 的 smem staging、(c′) 换成 per-warp 发布（唯一 block barrier = 激活那个 = legacy 自己的） | **fix 未上机** ⇒ §2 的三道门 |
| **v19/v21/v24 四变体全中性** | 「只动一个因子」推不动（instruction-bound + 低占用） | P5 是**多因子同批**（M 轴 + staging 组织 + 流水 + grid 形状），顺序照 `arch-floor §5.2`：**先删 µop（MTile）→ 再上 warp（ring/满 wave）** | 多因子同批 ⇒ **归因难**：必须一臂一变一 commit，A/C 只在同一臂内 |
| **P1d**（hc 段核 +3.3ms） | ① 丢 `ss_in`（tail 整行重读 20480 float）② `ck` 循环 8× L2 ③ tail 在整格 drain 期间跑串行链 | hc 设计**只取选举机制、不取 P1d 形状**（H3/H4 直接不做）；保留 `ss_in`、`split=1` | H1 依赖 draft parity |
| **hc-merge**（`DSV41_AR_STAMP_FOLD` 29.5s/步 + 代码存在性 +0.33ms） | 图 replay 下单调 arrival 机制坏掉 | P5 不碰 AR 协议 | H5 的 rows=m 隐患必须先修 |

### 5.2 ms 预算（全部标口径）

| 项 | 口径 | 收益 | 实现难度 | 人日 |
|---|---|---:|---|---:|
| **① MTile fix A/B**（零代码） | 微基准 + e2e 双门禁 | **−1.5 ~ −2.5**（若兑现） | **0 代码**（gate 已在树 `DSV41_MROWS_MTILE[_BN]`） | 0.5 |
| **② A+C**（常驻激活 + k-chunk ring） | 设计 | −0.5 ~ −1.5 | 中（改 mtile body，不新增核） | 2~3 |
| **③ B**（cluster DSMEM 激活共享） | 设计 | −0.3 ~ −1.0 | 中高（cluster 原语 + occupancy 约束） | 2~3 |
| **④ E**（mma swapAB + (b′)） | 设计（骨架已在树） | −3.5（区间 −2.5~−4.5） | **高**（换程序 + Rust scratch + 成对 gate） | 4~6 |
| **H1** draft raw hc 归零 | 账本（audit §5.4） | −0.31 | 低（gate 在树 + draft parity） | 0.5 |
| **H2** KCHUNK 重调 + 侧流 | 实测 v22 略负 | −0.2 ~ −0.3 | 低 | 0.5 |
| **H5** merge 分支 rows==1 前置 | — | 0（正确性） | 低 | 0.2 |
| **合计（P5-B 逐位腿）** | | **−2.8 ~ −5.6** | | 6~9.5 |
| **合计（不含 ④）** | | **−2.3 ~ −4.6**（**与 roadmap 的 −3~4ms 口径吻合**） | | 4~7 |

**诚实折扣**：仓史兑现率 60%（`l4l5-next-batch-implementation-plan §0-6`）⇒ **P5 的现实落点 −1.4 ~ −2.8ms**；
足额兑现才能把 verify 从 ~10-11 拉到 ~7。

### 5.3 必须上机取证的 U 项（否则清单可偏 ±5ms）

* **U1**：nsys **按形状**分解 `gemm_fp8_mrows` 的时间与 GridX（现在只有一个 52.1µs 的**平均**，见 §1.1 缺口）。
* **U2**：微基准 `t(m=1)` vs `t(m=6)` **逐形状**（legacy / fix-mtile bn=1,2,4）——§2 的门 1。
* **U3**：NCU（micro bench only）看 `sm__throughput` / `short_scoreboard` / `l1tex wavefronts`——§2 的门 2。
* **U4**：`/proc/<pid>/environ` 回读 + `[mrows-mtile] ARMED` 回执（本仓 #1 陷阱：gate 设了没生效；
  `DSV41_MROWS_MTILE*` 是 `.cu` 侧 `getenv` ⇒ **不在 Rust 侧 envchk 里**，判活的硬证据是回执行 + nsys 里
  `gemm_fp8_mtile_kernel` 的名字）。

### 5.4 第三种病（必须预先登记，否则会把负结果调参掩盖）

MTile 的赌注是 **issue-bound**；MPAR 的赌注是 **latency-bound**，已被两次实测证伪。
**若 fix 版 MTile 也负向** ⇒ 「既不是 issue-bound 也不是 latency-bound」成立，那就要找第三种病，候选：
① LDS 通道饱和（每 (warp,kb) 有 3bn+3M = 24 条 LDS）；② launch/drain 地板（每步数百发 × 1.5µs）；
③ 图的节点结构（AR_SAFE 表的放大）。
**处置：回报 + 转 ④（mma，指令数再降 30×），不调 bn 掩盖。**

---

## §6 影响范围

**修改文件（按实施顺序）**
| 文件 | 改动 | 备注 |
|---|---|---|
| `kernels/cuda/tests_dsv41_gemm_mrows.cu` | **不改**（已有 mtile resolution pin 与逐位/性能两腿）——只新增形状阶梯的打印 | Step-0 |
| `kernels/cuda/dsv41_kernels.cu` | ②：`gemm_fp8_mtile_kernel<M,AQ>` 的 body（grid-stride 循环 + k-chunk ring + 常驻激活）；`dsv41_mrows_mtile_warps_for` 的 coverage 规则加 T 与 nw 下限；③：cluster 变体（新 kernel `gemm_fp8_mtile_cluster_kernel<M>`）；H5：`dsv41_hc_front` 的 `rows==1` 前置 | **peer 冲突面**：该文件是热区 |
| `crates/ferrite-models/src/dsv41/chain_dev.rs` | ④（仅 mma 路线）：`swapab_part → [ks][M][n]` + `DSV41_PROJ_MMA` 与 `DSV41_SWAPAB` 的成对 arm | ⚠️ **peer 改动区，落地前必须协调**（ks=1 可绕开） |
| `kernels/cuda/dsv41_proj_mma_skel.cu` → 并入 `dsv41_kernels.cu` / `build.sh` | ④：骨架并 TU | 骨架已 compile-only 通过 |
| `scripts/*` | ②③ 的 A/B 臂（一臂一进程 + env 回读断言） | — |

**影响模块**：投影族（wq_a/wkv/wq_b/wo_b/wo_a_grouped/head 的 mrows 路径）；hc 前端（H1/H2/H5）；
`ferrite-dsv41` 的 scratch 与 FFI（仅 ④）。

**兼容性**：全部**默认 OFF**（`DSV41_MROWS_MTILE` unset ⇒ `dsv41_mrows_mtile_for` 返回 0 ⇒ 逐字节回 legacy）。
**无 ABI breaking change**（①②③ 都在既有 ABI 内；④ 复用 `swapab_part` 的 scratch 形态）。
**唯一数值面变化的是 ④**（且需要 (b′) 成对 gate）。

---

## §7 风险评估

| # | 风险 | 触发 | 应对 |
|---|---|---|---|
| 1 | **符号未定（头号）**：MTile 可能仍然负向 | fix 版 e2e 位移 < 40% 票面或无位移 | §2 门 1/门 2 前置；负向 ⇒ §5.4 第三种病 → 转 ④；**不投 bn 变体矩阵**（勿重演 v17→v21） |
| 2 | **口径缺口**：52.1µs 是四形状平均，可能病灶不在 wkv | U1 缺失 | Step-0 的按形状阶梯（0 代码，1 次 GPU） |
| 3 | **修法 A 与 coverage 对冲**（小 n 上 T>1 会把 grid 压到 <148） | wkv/sh | 小 n 只把 nw 抬到 2、T 保持 1；把 T>1 限定在 wq_b/wo_b；或走修法 B（cluster） |
| 4 | **ring 的 smem 反复**：chunk 太小 ⇒ cp.async 指令数上升；太大 ⇒ 回到低 blocks/SM | ② | chunk 512 B/行 起，扫 {256,512,1024}；用 `dsv41_cp_wait_groupN` 立即数 ⇒ **每档要模板实例化** |
| 5 | **激活 prologue 复制未被判为病灶**（判决 2 是我从代码推的，无实测） | ②的收益不出现 | U3 的 NCU 看 `l1tex wavefronts` 与 smem staging 量；必要时先做 B 的 micro bench |
| 6 | **第四个臂把 launcher 选路搞乱**（MTILE > ⑤a > MPAR > legacy，已点名 shadow） | 多 gate 同开 | 一臂一 gate；回执行必须打印；`fold_r == m` 前置已在（:6758） |
| 7 | **④ 的数值风险**（换程序，near-tie 边界移动） | 只开 PROJ_MMA 不开 SWAPAB = (a) 的 WOB 形状 | **成对 gate 硬约束**；`ks` 只是 (n,k) 的函数；双门禁 + 文本指纹 |
| 8 | **H5 的静默错答**（merge 分支 rows=m） | `HC_MERGE=1` + `rows=m` | **先修 `rows == 1` 前置**（0.2 人日）；这条是正确性，不是提速 |
| 9 | **peer 冲突**：`dsv41_kernels.cu` / `chain_dev.rs` 是热区 | ②④ 落地 | 落地前与 peer 协商；② 只动 kernel body（同文件但独立函数）；④ 的 Rust 侧延后到 ① - ③ 有结论 |
| 10 | **图捕获**：新增 T 循环/常驻激活不改核内分配（无 `cudaMalloc`），ring 用 `cp.async` ⇒ 可捕获 | — | 段核化不得引入 host 交互或 grid.sync（`cudaLaunchCooperativeKernel` 与图捕获不兼容） |

---

## §8 建议分工

| 部门 | 任务 | 为什么是它 |
|---|---|---|
| **工部（ministry-works）** | ① U1/U2/U4 的 Step-0 阶梯与回执；② 修法 A+C 的 kernel 改写（`gemm_fp8_mtile_kernel` body + coverage 规则）；H1/H2 的 A/B | 唯一有 kernel 编写与 A/B 执行力的部门；②③④ 全是 `.cu` 级工程 |
| **户部（ministry-revenue）** | 判决 2 的**独立复核**：按形状算「权重 / 激活 / LUT」三笔 smem staging 字节与 L2 请求量，验证 `wkv 上激活 = 权重 3.4×` 的账；ring 的 smem/占用曲线 | 这是纯资源账，且是 P5 收益的主要来源，必须独立算一遍 |
| **刑部（ministry-justice）** | 逐位契约的边界审查：②③ 的 C1-C6 是否真的只改「谁算哪个元素」；H5（`hc_front_kernel` rows=m）的**静默错答**路径复核；T 循环下的 `wrow/nvalid` 边界 | 本仓有「m 行核非逐位 ⇒ accept 崩」的先例（WOB_MROWS_F32），逐位断言是红线 |
| **兵部（ministry-defense）** | gate 纪律与幻影门审计：`.cu` 侧 getenv（MTILE*）不进 `/proc/environ`，必须靠回执 + nsys kernel 名判活 | 本仓 #1 陷阱（gate 设了没生效）已多次造成假结论 |
| **礼部（ministry-rites）** | 设计文档与实测回写的对齐：`mrows-mtile-fix.md` / 本文件 / `l4-l5-kernel-path.md §2` 三处口径统一（52.1µs 的「平均值」必须标注） | 口径混用是本仓的重复事故（roadmap §8） |
| **门下省（chancellery）** | 审查本方案的**判决门**是否够硬（门 1/2/3 与止损线）；审查「H3/H4 不做」的论证是否有漏洞 | 方案的风险面在「做与不做的边界」上 |
| **吏部** | — | 不涉及代码质量重构 |
| **尚书省（department-state）** | ②③④ 的波次排期与 peer 冲突协调（`dsv41_kernels.cu` / `chain_dev.rs` 热区） | 跨部门资源与冲突仲裁 |

**不建议分配给任何部门**：④（mma）在 ①-③ 出结论前不开工（避免与 §3.2 的形状定稿冲突）；
H3/H4/H6 不做（已论证）。

---

## §9 一句话总结

**P5 不是「待设计的第三条路」——第三条路（MTile）已经在树里，它的 g2 版被测为负、fix 版从未上机；
本设计把它补成可施工的三步（常驻激活的 grid-stride + k-chunk ring + cluster/full-wave），
并揭出账上没有的第二笔成本（激活 prologue 在 wkv 上是权重的 3.4×）与结构上限（逐位约束下小 n 填不满 148 SM，
填满必须换程序 = mma）。hc 侧只需 H1+H2（−0.5ms），其余四种融法都有数值或并行度硬阻塞。
P5 的现实预期 = −1.4 ~ −2.8ms（60% 兑现率），足额兑现 = −2.3 ~ −4.6ms，与 roadmap 的 −3~4ms 口径吻合。**
