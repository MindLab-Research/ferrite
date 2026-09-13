# M-tile 的 wo_a / head 扩展（G2 续）— `wo_a_grouped` 与 `gemv_bf16_v1_mrows` 的 N-tile 化

> 工部 · 2026-09-13 · **禁止 GPU/e2e**（任务书）；本机 `cargo check --workspace --all-targets`
> **EXIT=0**；远端 b300（CUDA 13.2, `ubuntu@43.202.208.136`, **无 GPU**）`nvcc compile-only`
> （见 §3）。
> 依据：`docs/agent/mrows-mtile-design.md`（G2 完整设计：warp 映射 / 指令账 / C1-C6 逐位论证，
> 其 §7.5 点名「wo_a_grouped / head 的 v1_mrows：同族，后续」）+ `wo-a-grouped-nwarps-verify-manual.md`
> （wo_a 现状：占用率旋钮）+ `head-act-f32vec-wo-a-cp16.md`（head 现状：v1-order m-row fold）。
> 载体：`kernels/cuda/dsv41_glue.cu`（head）+ `kernels/cuda/dsv41_kernels.cu`（wo_a）。
> Gate：`DSV41_HEAD_MTILE` / `DSV41_WO_A_MTILE`（**均默认 OFF**；unset/`0` = 逐字节回到现状）。

---

## §0 一句话

G2 把 M 从「warp 内寄存器串行」改成「GEMM 的 M-tile 维」并只覆盖了**投影族**（`wq_a/wkv/wq_b/wo_b`，
走 `dsv41_gemm_fp8_mrows`）。本交付把**同一条定理**（只换「哪个 warp 算哪个元素」，元素程序逐位不动）
套到同族但**不同入口**的两个 kernel：wo_a 的分组 GEMV（`wo_a_grouped_gemv_kernel`）与 head 的 bf16
m-rows GEMV（`gemv_bf16_v1_mrows_kernel`）。两个 kernel 现状都已经是「权重服务 M 行」的 weight-
stationary m-row fold，**缺的是 N 轴的 tile**：每个输出行仍独占一个 warp ⇒ 同一份激活的
`LDS + decode`（wo_a）或 f32 读（head）被 `n` 个 warp **各付一遍**。本交付把 N 轴变成 warp 内的
`bn` 宽 tile（`acc[RN][bn]`），让**一次激活 decode 服务 `bn` 个输出行**。

**指令账（每 warp 每 (r,kb) / 每 k 列）**：wo_a `1.00x(bn=1) / 0.80x(bn=2) / 0.70x(bn=4)`；
head `1.00x / 0.77x / 0.65x`。**权重读量两者都不变**（`n*k`，每输出行仍被恰好一个 warp 读一次）。

---

## §1 head：`gemv_bf16_v1_mrows_kernel` 的 N-tile（`DSV41_HEAD_MTILE`）

### 1.1 病灶（proj-head 的「摊薄失效」，audit §10.2）

`gemv_bf16_v1_mrows_kernel` 已把 m 行激活折进**一次权重扫描**（消掉 m 次权重重读），但每个输出行
仍是**自己的 warp** ⇒ 整个 `M x k` 的 f32 激活 slab 被 `n` 个 warp **各走一遍**。激活是 f32（4 B）
对权重 bf16（2 B）⇒ 每输出行的激活流是权重流的 `2M` 倍；head 形状 `(m=6, k=5120)` 下一个 8-warp
block 搬 `8*6*5120*4 = 960 KB` 激活 vs `8*5120*2 = 80 KB` 权重 —— 摊薄省下的权重字节在激活侧又付回去
（`head-act-f32vec` 从 **staging 读路径**攻击同一个病灶；本臂从 **tile 复用**攻击它，两者不同轴、
不互斥）。

### 1.2 方案（warp → (m_block=0, n_block=w)）

```
BN_MAX = 4                        // 寄存器 tile 的 N 上界
bn     = DSV41_HEAD_MTILE_BN      // 1..4，默认 2（warp tile 的 N 宽度）
nwarp  = 8                        // block = 256 线程（与 legacy 同块宽）
warp w → 输出行 [row0 + w*bn, row0 + (w+1)*bn)，row0 = blockIdx.x * nwarp * bn
grid   = ceil(n / (nwarp*bn))     // 无 strided walk：block 直接占住自己的 nwarp*bn 行
寄存器 tile: acc[RN][bn]，RN = M
```

每个 k 列：**激活解一次**（`xv[r] = x[r*k+c]`，M 个，hoist 出 nn 循环）→ 折叠 `bn` 个权重行
（`wv = __bfloat162float(w[row*k+c])`）→ `acc[r][nn] += wv * xv[r]`。

### 1.3 指令账（每 warp 每 k 列，M=6）

| 类别 | legacy `v1_mrows` | **M-tile bn=2** | **M-tile bn=4** |
|---|---:|---:|---:|
| 激活读（`x`） | M = 6 | **M = 6**（hoist，服务 bn 行） | 6 |
| 权重读（`w`） | 1 | bn = 2 | 4 |
| FMA | M = 6 | M·bn = 12 | 24 |
| **每 warp 每列** | **13** | **20** | **34** |
| warp 数 | `n` | `n/2` | `n/4` |
| **每列全 grid** | **13n** | **10n** | **8.5n** |
| **对 legacy 比值** | 1.00 | **0.77** | **0.65** |

**读表**：权重读总量恒为 `n`（每列）—— tile 没有改变权重流量，只把**激活读**从 `n*M` 降到
`n*M/bn`。**bn=1 是对照组**（比值 1.00 = 只换几何、无复用）。

> **诚实边界**：head 的 `n*k*2` = 165 MB（seg=16160 @ world=8）权重流是它 DRAM 侧的支配项，
> 本臂**不动**它。所以本臂的票面是**指令侧**（mrows-mtile 的 issue-bound 赌注），不是字节侧。
> 若 GPU 显示 head 纯权重-DRAM-bound ⇒ 本臂是 wash —— 那正是这次实验要回答的。

---

## §2 wo_a：`wo_a_grouped_gemv_kernel` 的 N-tile（`DSV41_WO_A_MTILE`）

### 2.1 病灶

`wo_a_grouped_gemv_kernel` 已是 weight-stationary m-row fold：每 warp stage **一条**权重行、
把 M 行激活都折上去（权重读摊到 M）。但**每个输出行仍独占一个 warp** ⇒ M 行激活的
`decode`（`s_lut[s_a[j]] * s_as[j>>5]`，以及其背后的 `s_a`/`s_as` staging）被本组 `n` 个 warp
**逐输出行各做一遍**。

### 2.2 方案（分组结构保留；warp → (m_block=0, n_block=w)）

```
BN_MAX = 4                        // 寄存器 tile 的 N 上界
bn     = DSV41_WO_A_MTILE_BN      // 1..4，默认 2
warp w → 组 g 内输出行 [row_base, row_base+bn)，row_base = blockIdx.x*nwarps*bn + w*bn
grid   = (ceil(n/(nwarps*bn)), groups)     // grid.y = groups = nlg（分组结构不动）
寄存器 tile: acc[RN][bn]，RN = M
smem   : nwarps*bn*k（权重 tile）+ 1KB(LUT) + nb_k*4(激活 scale) + k(激活行)
```

**分组结构不动**：`g = blockIdx.y`、`wg = w + g*n*k`、`wsg = w_scale + (g*n/32)*nb_k`、
`og = out + g*n` 全部逐字保留 —— M-tile 只在**每个组内**把 N 轴 tile 化。每 warp stage 自己
`bn` 条权重行（cp.async16，与 legacy 的单行 staging 同规则、同字节）；每 (r, kb) 的激活
decode `av` 只算一次，服务 `bn` 个输出行。

### 2.3 指令账（每 warp 每 (r, kb)，M=5）

| 类别 | legacy `wo_a_grouped` | **M-tile bn=2** | **M-tile bn=4** |
|---|---:|---:|---:|
| 激活 decode `av`（2×LDS + scale LDS + FMUL） | 4 | 4（hoist，服务 bn 行） | 4 |
| 权重 `wv`（LDS + LUT LDS + FMUL）+ `sb`（LDG+convert） | 6 | 6·bn = 12 | 24 |
| FMA | 1 | bn = 2 | 4 |
| **每 warp 每 (r,kb)** | **10** | **12** | **28** |
| warp 数 | `n` | `n/2` | `n/4` |
| **对 legacy 比值** | 1.00 | **0.80** | **0.70** |

### 2.4 几何解析（launcher）

- `bn = DSV41_WO_A_MTILE_BN`（1..4，默认 2）。
- `nwarps` 由 launcher**重新解析**（`wo_a_grouped_nwarps` 的占用率旋钮之后）：
  「**仍覆盖 SM 的最宽块**」+「权重 tile 装进 48 KB 静态预算」：
  ```
  mw = min(legacy_nwarps, max(1, n/(bn*SM)))      // 覆盖：grid=ceil(n/(mw*bn)) >= SM
  while (mw > 1 && mt_smem(mw) > 48KB) mw >>= 1   // 预算：装不下就缩到装得下
  ```
  `mt_smem(w) = w*bn*k + 1KB + (k/32)*4 + k`。
- **无整除要求**（尾行由 `active`/`nvalid` 谓词处理；barrier 在谓词之外，不死锁）——比 legacy 的
  `n % nwarps == 0` 更宽。
- 48 KB 超限 ⇒ **decline 回 legacy**（与 legacy 同一「不做 per-M `cudaFuncSetAttribute`」纪律）。

---

## §3 逐位论证（C1-C6 的同构）

**契约**：`out[q][row]` 与 legacy kernel（进而与 m=1 解码）**逐位相同**。两个 M-tile 都是
**同一个元素程序的重排** —— 唯一动的是「哪个 warp 算哪个元素」（两个 kernel 的头注释都明文写着
「行独立 ⇒ 谁拥有它不可观测」）。

| 契约 | wo_a legacy | wo_a M-tile | head legacy | head M-tile | 为什么相同 |
|---|---|---|---|---|---|
| C1 K 走序 | `#pragma unroll 32`, `kb` 升序, `j=kb*32+lane` | **同** | `for (c=lane; c<k; c+=32)`，**无 unroll** | **同** | 逐字保留（head 的 no-pragma 是 legacy 实测 pin） |
| C2a 权重字节 | `s_lut[row_s[j]]`，`row_s=s_w+warp*k` | `s_lut[row_s_nn[j]]`，`row_s_nn=s_w+(warp*bn+nn)*k` | `__bfloat162float(w[row*k+c])` | **同** | staged slab 是 `w[row0*k..)` 的逐字节拷贝，索引 1:1；head 直读同一全局地址 |
| C2b 权重 scale | `ue8m0_to_f(wsr[kb])` | `ue8m0_to_f(wsg[((row>>5)*nb_k)+kb])` | —（head 无 scale） | — | 同一 `w_scale` 字 |
| C2c 激活 | `s_lut[s_a[j]]*s_as[j>>5]` | **同**（每 (r,kb) 一次，跨 nn 复用） | `x[r*k+c]`（f32） | **同**（每 (r,c) 一次，跨 nn 复用） | hoist 出 nn 循环的值与 legacy 同一 load/同一表达式，复用不改变任何行的值 |
| C3 归约树 | `shfl_xor` off=16,8,4,2,1 | 每 (q,nn) 一棵，**同 off 序** | 同 | 每 (r,nn) 一棵，**同 off 序** | 每元素恰好 1 个 warp 算、1 棵树 |
| C4/C5 无跨行/跨 K 重组 | `acc` 单链 | `acc[q][nn]` 独立链 | `acc[r]` 独立链 | `acc[r][nn]` 独立链 | 从未跨 q/nn 相加；无 K-split |
| C6 累加式 | `acc += av*(s_lut[row_s[j]]*sb)` | `acc[q][nn] += av*wv`，`wv=s_lut[row_s_nn[j]]*sb` | `acc[r] += wv*x[r*k+c]` | `acc[r][nn] += wv*xv[r]` | **同一表达式、同一操作数次序**（wo_a：`av*(...)`；head：`wv*x`） |

**唯一变化的东西 = warp 映射**（legacy header 原话：the block geometry does not enter the parity
argument — rows are independent）。所以每行的 fma 链次序、归约树**逐行保持**。

**两个必须点名的细节**：
1. **`s_a`/`s_as`/权重 slab 的 staging 是纯拷贝**（wo_a：`s_a[i]=ar[i]` 与 `dsv41_cp_async16`，
   无算术；head 无 staging）⇒ staged 槽位与 legacy 读的是**同一全局地址的同一批字节**，位不可移动。
2. **head 的 `xv[r]` hoist / wo_a 的 `av` hoist**：legacy 也已把该值每 (r, ·) 算一次（wo_a 的 `av`
   本来就在 kb 循环里每 (r,kb) 算一次；head 的 `x[r*k+c]` 本来就在 r 循环里每 (r,c) 载一次）。
   本臂把它**跨 nn 复用** —— 同一个乘积、同一个操作数次序 ⇒ 任何一行的位不动。

> ⚠️ **head 不做 c 循环 unroll、不做向量化 body**：那会改变 lane→c 归属 ⇒ 改变每个部分和，正是
> `gemv_bf16_nt_kernel` 的 v2 程序（`head_gemv_bf16_mrows` 曾以 ~1e-3 偏差、33% 回声失败的
> 那条路，b8b67c0）。本臂保持 v1 的标量、v1 的次序。

---

## §4 编译验证（compile-only，无 GPU）

| 项 | 命令 | 结果 |
|---|---|---|
| Rust | `cargo check --workspace --all-targets` | ✅ **EXIT=0**（仅既存 warning：dead_code/unreachable/unused，与本次无关） |
| CUDA glue | 远端 `nvcc … -Xptxas -v -c dsv41_glue.cu` | 见 §4.1 |
| CUDA kernels | 远端 `nvcc … -Xptxas -v -c dsv41_kernels.cu` | 见 §4.1 |

### 4.1 远端 nvcc（CUDA 13.2, `ubuntu@43.202.208.136`, `/tmp/mtile_woahead_smith/`, sm_100a）

```
nvcc -gencode arch=compute_100a,code=sm_100a -O3 --use_fast_math -std=c++17 -Xptxas -v -c dsv41_glue.cu     -> GLUE_RC=0
nvcc -gencode arch=compute_100a,code=sm_100a -O3 --use_fast_math -std=c++17 -Xptxas -v -c dsv41_kernels.cu  -> KERNELS_RC=0
```
产物：`glue.o` 1,139,448 B；`kernels.o` 1,860,584 B。**0 error**。

**唯一的既存 warning**（非本次改动）：`dsv41_kernels.cu` 的 `#177-D` unused ×4 ——
`:9526 k1max` / `:6101 nt` / `:5832 nwarp` / `:108 e2m1_to_f`，全部是 HEAD 已有（行号因
wo-pair + 本次新增而位移）。`dsv41_glue.cu` **0 error / 0 warning**。

### 4.2 ptxas 数据（`-Xptxas -v`，全部 0 stack frame；**0 spill** 除 wo_a 的 M=3）

**head `gemv_bf16_v1_mtile_kernel<M, 4>`**（8/8 全部生成）：

| M | 1 | 2 | 3 | 4 | 5 | **6** | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| registers | 30 | 40 | 47 | 42 | 48 | **57** | 62 | 64 |
| spill (B) | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 0 |
| barriers | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**wo_a `wo_a_grouped_mtile_kernel<M, 4>`**（8/8 全部生成）：

| M | 1 | 2 | 3 | 4 | **5** | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| registers | 48 | 48 | 48 | 64 | **64** | 64 | 102 | 108 |
| spill (B) | 0 | 0 | **8** | 0 | **0** | 0 | 0 | 0 |
| barriers | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

**读表**：
- head 的**生产实例 `M=6` = 57 regs / 0 spill**；`__launch_bounds__(256)` 下 57 regs 允许
  ≥3 块/SM（寄存器侧），无 smem ⇒ **两个 resource 都不构成瓶颈**。0 spill 说明 (M×4) 的寄存器
  tile 没把编译器逼到本地内存。
- wo_a 的**生产实例 `M=5` = 64 regs / 0 spill / 1 barrier**（与 legacy `wo_a_grouped_gemv_kernel`
  实测的 32 regs 相比高——这是 tile 的代价，但仍在 64 regs 档，`__launch_bounds__(256)` 下 4 块/SM）。
- **唯一的小 spill**：`wo_a` 的 **M=3 → 8 B**（不在生产形状上；M=3 的 `RN×BN_MAX=12` 个 acc 与
  地址寄存器交错时编译器溢出 8 B）。**M=1..2 与 M=4..8 均 0 spill**。汇报，不改设计去掩盖
  （若尚书省要求 M=3 也 0 spill，可对 M=3 特化 `BN_MAX=1` 或把 c 循环分块，属独立小项）。

---

## §5 GPU 验证手册（主 agent 执行；本任务禁止 GPU/e2e）

> 纪律（audit §5.3）：**优先 micro bench**（`tests_dsv41_*`）做逐位门，**e2e 双门禁**只作最后一步。

### 5.1 前置（三证）

```bash
cd kernels/cuda && bash build.sh 100a          # .so 与 Rust 一起重建
cd ../.. && cargo build --release
nm -D kernels/cuda/libferrite_kernels.so | grep -c gemv_bf16_v1_mtile_kernel   # >= 1
```
> ⚠️ 两个 gate 都**新增独立 kernel 符号**（`gemv_bf16_v1_mtile_kernel` /
> `wo_a_grouped_mtile_kernel`），**不改 ABI、不加新 extern "C" 入口** ⇒ 旧 `.so` 与新二进制互不
> 干扰；但 gate 是**文件/函数内 static（load 时读一次）** ⇒ **一进程一值**，不能同进程 sweep。

### 5.2 逐位门（**先于性能**）

**head**：`tests_dsv41_head_mrows.cu` 已有 `dsv41_gemv_bf16_v1_mrows` 的 in-process 对照臂
（`tests_dsv41_draft_parity.cu:886` 的 `DRAFT_HEAD_FOLD v1`）。新增一臂：同一 `(w,x,m,n,k)` 跑
`DSV41_HEAD_MTILE=0` 与 `=1` 两次，`to_bits` 逐位比 ⇒ 期望 `maxdiff = 0x0`。

**wo_a**：`tests_dsv41_glue.cu` / `tests_dsv41_gemm_fp8.cu` 里有 `wo_a_grouped` 的对照臂。
新增一臂：同一 `(a,a_scale,w,w_scale,g,rows)` 跑 `DSV41_WO_A_MTILE=0/1`（`BN=1,2,4`），逐字节比
⇒ 期望 `0` 字节差。

```bash
# 一臂一进程（static gate）
for bn in 1 2 4; do
  DSV41_HEAD_MTILE=1 DSV41_HEAD_MTILE_BN=$bn /tmp/mtile/t_head_mrows --quick
  DSV41_WO_A_MTILE=1 DSV41_WO_A_MTILE_BN=$bn /tmp/mtile/t_glue
done
```
**判据**：每个臂 all-checks-passed 且打印 `[head-mtile] ARMED …` / `[wo-a-mtile] ARMED …` 一行
—— **没打印 = 没走 M-tile**，那次运行不算数（`armed but inert` 幻影门）。

### 5.3 性能门（micro bench，同形状；m=1 / bn=1 为基准）

| 臂 | head | wo_a |
|---|---|---|
| 基线 | `DSV41_HEAD_MTILE` 未设 | `DSV41_WO_A_MTILE` 未设 |
| 对照 | `bn=1`（几何变、无复用） | `bn=1` |
| 目标臂 | `bn=2`, `bn=4` | `bn=2`, `bn=4` |

**判据优先级**：先看**符号**（是否 < legacy），再看幅度。
- head 目标：`t(m=6, mtile)/t(m=1)` 进一步下降；若不动 ⇒ 确认「head 是权重-DRAM-bound」，上报。
- wo_a 目标：`wo_a_grouped_mtile_kernel` per-instance 时间下降（vs `wo_a_grouped_gemv_kernel`）；
  若上升 ⇒ 上报（**不要**硬调 bn 掩盖）。

### 5.4 e2e 双门禁（授权后）

```
# 一臂一进程；计数 200 tok，读 [dspark] 的 step_ms 分解
DSV41_HEAD_MTILE=1 DSV41_HEAD_MTILE_BN=2   ...   # 期望 verify 不变或下降
DSV41_WO_A_MTILE=1 DSV41_WO_A_MTILE_BN=2   ...   # 期望 verify 不变或下降
```
**双门禁**：
- `step_ms`（`[dspark]` 分解）；
- `mean-k` **不掉**（当前 SWALLOW 栈基线 **2.240**；**逐位等价 ⇒ 应完全相等**，任何 delta 说明
  实现有 bug，不是噪声）；
- 三段文本红线：**零拉丁 / 0 double-char / 「先帝创业未半」**。

### 5.5 回滚

```bash
unset DSV41_HEAD_MTILE DSV41_WO_A_MTILE     # 立即回到现状程序（逐字节）
```

---

## §6 与 peer 的冲突边界

- **wo-pair-rows peer**（`docs/agent/wo-pair-rows-design-and-gpu-manual.md`）：其 §1 明文
  「**未改**（刻意）：`wo_a_grouped_gemv_kernel` 体」；它改的是 `gemm_fp8_mtile_kernel<M,AQ>` +
  新入口 `dsv41_gemm_fp8_mrows_q_f32`（:6851-:6970 区）。本交付的 wo_a 区（:8098+）与它
  **文本不重叠**（开工 `git status` = 该文件 clean；它的 .cu 改动已在 HEAD 提交 `19aa607`）。
  它的 §4.5「将来 1 发 grid-sync pair 会复用 wo_a 体」留待有 GPU 的一轮 —— 届时若真动
  `wo_a_grouped_gemv_kernel` 体，本交付的新符号 `wo_a_grouped_mtile_kernel` 是**独立 kernel**，
  可单独 rebase。
- **hc-chain-rows / fusion-parity-rows peer**：改 `chain_dev.rs` / `device.rs` / `dspark_dev.rs`
  的 hc/attention 区 —— **不碰 .cu**，不冲突。本交付**未动 Rust 侧**（gate 全在 .cu launcher 内）。
- ⚠️ **本交付未改任何 Rust 文件**；若尚书省要把 gate 接到调用点（例如 chain_dev.rs 的
  `verify_wo_pair_mrows()` 那样），那是**另一个交付**（且会与 hc-chain peer 抢 chain_dev.rs）。

---

## §7 风险与坦白

1. **wo_a 的 smem 预算**：M-tile 的权重 tile 是 `nwarps*bn*k`（legacy 是 `nwarps*k`）⇒ bn=2 时
   同样 nwarps 下 smem 翻倍。launcher 用「覆盖 + 预算」双约束重解析 nwarps；对 verify 形状
   （n=1024, k=4096）bn=2 会解析到 `nwarps=3`（grid≈171 ≥ 148）或更小（视 SM 数），**会偏离
   已交付的 `DSV41_WO_A_NWARPS` sweep 结论** ⇒ **本臂下 wo_a 的 nwarps sweep 必须重跑**。
2. **head 的符号未定**：同 mrows-mtile —— 本臂赌 issue-bound。若减了指令仍不变/更慢，说明 head 是
   纯权重-DRAM-bound（或在 L2/依赖链上），**上报，不调参**。
3. **bn=4 的覆盖**：`cover = n/(bn*SM)`，bn=4 时 cover 更小 ⇒ nwarps 更小；head 的
   `grid = ceil(n/(8*4))` 在 n=16160 时 = 505 ≥ 148 ✓；wo_a 的 bn=4 在 n=1024 时 cover=1
   ⇒ grid = ceil(1024/4) = 256 ✓。
4. **两个新 kernel 都未在 GPU 上跑过**（本任务禁止 GPU）。本文档的指令账是**静态核算**，
   性能结论一律留给 §5。
