# M-tile 的逻辑修复 — 激活 smem staging + per-warp 权重 slab（bn=1 对照实验的判决落地）

> 载体：`kernels/cuda/dsv41_kernels.cu` 的 `gemm_fp8_mtile_kernel<M, AQ>`（**body 替换**，gate / ABI / 名字不变）。
> Gate：`DSV41_MROWS_MTILE`（**默认 OFF**）+ 调参轴 `DSV41_MROWS_MTILE_BN`（1..4，默认 2）——**沿用原 gate**，
> 本修复是「修复版替换」而非并存 A/B（见 §6 回滚）。
> 前序：`mrows-mtile-design.md`（g2 设计，§2.3 的「激活不进 smem」论证**被证伪**）、
> `mtile-woa-head-design.md`（wo_a/head 扩展）。
> 日期：2026-09-13。**编译已验证，GPU 未验证**——本文件是修复记录 + 验证手册。

---

## §0 一句话

g2 的 M-tile 在 **bn=1 对照臂**（指令/warp/smem 与 legacy 相等）仍然 **+2.54 ms**，把负向锁死在两个
**结构性**改动上：**(a) 激活 `__ldg` 直读 L1**（legacy 是 smem staging + LDS 消费）与
**(c′) 权重 slab 的跨 warp 发布**（block 级 `cp.async + wait_all + __syncthreads`）。
本修复把这两处换回 legacy 的形态：激活照 legacy stage 进 `s_a`/`s_as` 并读 LDS；权重改**每 warp
stage 自己的 `bn` 行**、LUT 改**每 warp 自建**——block 级 barrier 只剩激活那一个（= legacy 自己的那个）。
**`bn=1` 时本 kernel 在结构上就是 legacy 程序。**

---

## §1 病灶（bn=1 对照实验的判决，本任务书内嵌）

| 事实 | 值 |
|---|---|
| bn=1 臂 vs legacy 的**每 warp 指令数** | 相等 |
| bn=1 臂 vs legacy 的 **warp 数** | 相等 |
| bn=1 臂 vs legacy 的 **smem** | 相等 |
| bn=1 臂的实测 | **+2.54 ms（慢）** |
| bn1→4 砍半指令带来的改善 | 仅 0.68 ms ⇒ **非 issue-bound** |

⇒ 负向不是指令数，是**结构**（延迟/串行），且唯一的两处结构差异就是 (a) 与 (c′)。

**为什么 bn=1 对照能把病灶锁死**：bn=1 时 M-tile 的几何退化成「一 warp 一行输出、折叠 M 行激活」
——**与 legacy 的 (warp, 行) 映射逐一对齐**，指令账 §2.2 的表在 bn=1 正是 legacy 的 `5M+4`。所以
两臂之间**只剩**「激活怎么读」和「权重怎么发布」两处不同。慢 2.54ms 只能从这两处来。

### 1.1 病灶 (a)：激活 `__ldg` 直读 L1

- **legacy**：`s_a[q*k+j]`（LDS，staging 是 `cp.async16`/scalar 的纯拷贝）+ `s_as[q*nb_k+(j>>5)]`（LDS）。
  依赖链：**LDS → LUT LDS → FMUL**（两条 smem 往返，短）；
- **g2**：`s_lut[__ldg(a + q*k+j)] * __ldg(a_scale + q*nb_k + kb)`。
  依赖链：**LDG(L1) → LUT LDS → FMUL**——每条激活字节都挂在一次 **L1 往返**上，
  且 `a_scale` 的 `__ldg` 是 **per-(q,kb) 标量**读（32 lane 同址广播）。
  在 `nw` 个 warp 各自折叠同一份 M 行激活的情况下，这是一层被 `nw` 倍放大的 L1 依赖。

**g2 §2.3 的论证为什么错**：它说「激活的复用是 warp 内 bn 轴，block 内 stage 会按 block 复制 prologue」。
bn=1 对照证明：**复用不是决定读法的量，延迟才是**；而且 legacy **每个 block 也 stage 了激活**
（`M*k` 字节），这笔预算是 legacy 已经付过的，不是 M-tile 新引入的 prologue。

### 1.2 病灶 (c′)：权重 slab 的跨 warp 发布

- **legacy**：每 warp stage **自己那一行**（`for (i = lane; i < n16; i += 32)`），
  在激活 barrier **之后**只 `wait_all` 自己那一组 —— 单 warp 等待量 = `k` 字节；
- **g2**：slab 是 `nw*bn` 行**整块**由全 block 线程 `i += blockDim.x` 平铺 stage，然后
  `wait_all + __syncthreads` **块级发布**。于是「这个 block 的权重 tile 的 DRAM/L2 往返」
  成了一个 **block 尺寸的硬串行点**：单次等待量 = `nw*bn*k` 字节（wkv bn=1 且 nw=3 时是 legacy 的 3 倍），
  且**每个 warp 都被迫等齐其他 warp 的字节**。

---

## §2 修法（两处，均已落地）

### 修法 1 — 激活 smem staging（`AQ == 0` 专属）

把 `m` 行激活照 **legacy 的同一规则** stage 进 `s_a`/`s_as`，消费改读 LDS：

```cpp
// prologue（block-wide，legacy 的 stage 规则 verbatim）
for (int q = 0; q < RN; ++q) {
    for (int i = threadIdx.x; i < nb_k; i += blockDim.x) s_as[q*nb_k + i] = asr[i];
    if (a16) for (int i = threadIdx.x; i < n16a; i += blockDim.x)
                 dsv41_cp_async16(sar + (i<<4), ar + (i<<4));   // + k%16 tail
    else     for (int i = threadIdx.x; i < k;    i += blockDim.x) s_a[q*k + i] = ar[i];
}
...
if (a16) dsv41_cp_wait_all();            // 激活组在这里退休
if constexpr (AQ == 0) __syncthreads();  // ← 唯一的 block barrier（legacy 自己的那个）

// consume（legacy 的表达式 verbatim）
av[q] = s_lutw[s_a[q*k + j]] * s_as[q*nb_k + (j>>5)];
```

- **`AQ == 1`（wo-pair-rows）不 stage 激活**：它的激活是 **f32**（`a_stride` 行距）且在 warp 内量化，
  stage 一套 f32 tile 是 `m*k*4` 字节（wkv 形状 120 KB）⇒ 该臂保持 `__ldg(af + …)`（本来就没有
  激活 staging 的病：它是**另一条** arm，量化本身是它的工作）。
- **barrier 的数量与位置**：只剩激活那一个，**与 legacy 完全相同**；且权重传输被刻意留在
  **跨 barrier 在飞**（barrier 之后按 warp 退休），这正是 legacy prologue 的形状。

### 修法 2 — per-warp 权重 slab + per-warp LUT（block barrier 消失）

```cpp
// LUT：每 warp 自建**全部 256 项**（lane 跨步写），只给自己的 lane 用 ⇒ __syncwarp 发布
for (int i = lane; i < 256; i += 32) s_lutw[i] = e4m3_to_f((uint8_t)i);
__syncwarp();

// 权重：每 warp 只 stage **自己的 bn 行**（`i = lane`，步长 32 —— 不是 blockDim.x）
const uint8_t* wsrc = w + (size_t)row_base * k;
uint8_t*       s_ww = s_w + (size_t)n_block * bn * k;
if (wrows > 0) { for (int i = lane; i < wrows*n16; i += 32) cp_async16(s_ww + i*16, wsrc + i*16);
                 dsv41_cp_commit(); }
...
if (wrows > 0) { if (w16) dsv41_cp_wait_all(); __syncwarp(); }   // warp-local 退休
```

- **读法**：consume 的索引 `s_w[(n_block*bn + nn)*k + j]` **一个字都没改** ——
  而 `s_w + n_block*bn*k + nn*k + j` 正是 `s_ww + nn*k + j`。**槽地址、字节、1:1 平铺索引全部不变**；
- **LUT 读**：`s_lut[byte]` → `s_lutw[byte]`（= `s_lut[warp*256 + byte]`）。**值相同**（同一 `e4m3_to_f`），
  只是每个 warp 有自己完整的 256 项副本；
- **为什么这样能去掉 barrier**：一个 warp 的 slab 槽与它的 LUT 副本**只有它自己写、只有它自己读**
  ⇒ 发布只需 `__syncwarp`（warp 内 smem 可见性），**没有任何跨 warp 读 `s_w`/`s_lut` 残留**。

### 为什么不是「per-warp 冗余 stage 激活」

激活是**真的 block 共享**（`nw` 个 warp 都读全部 `RN` 行）。per-warp 冗余 stage 会让 `nw` 个 warp
各搬一遍 `M*k` 字节 —— 那正是 MPAR 的病根（重复的量是 prologue）。所以激活保持 **block-wide + 一个
barrier**，而 barrier 只 gate 激活（`M*k`，对每个 block 相同、L2 驻留），不再 gate 整块权重。

---

## §3 逐位红线（C1–C6 保持）

契约不变：`out[q][row]` 与 m=1 解码该行**逐位相同**。修复只改了「激活字节从哪读」和「权重字节谁发布的」，
**没有改任何值**。

| 契约 | 修复前 | 修复后 | 为什么相同 |
|---|---|---|---|
| C1 K 走序 | `kb` 升序，`j = kb*32+lane` | **同** | 未触碰 |
| C2a 权重字节 | `s_lut[s_w[(n_block*bn+nn)*k+j]]` | `s_lutw[s_w[(n_block*bn+nn)*k+j]]` | 槽地址不变；`s_lutw[b] == s_lut[b] == e4m3_to_f(b)`（同一函数建表，每 warp 一份完整副本） |
| C2b 权重 scale | `w_scale[(row>>5)*nb_k + kb]` | **同** | 未触碰 |
| C2c 激活 | `s_lut[__ldg(a+q*k+j)] * __ldg(a_scale+q*nb_k+kb)` | `s_lutw[s_a[q*k+j]] * s_as[q*nb_k+(j>>5)]` | `s_a`/`s_as` 是 `a`/`a_scale` 的**纯拷贝**（无算术、无重结合），拷贝的宽度不可观测 ⇒ 读到的是同一字节/同一个字；`j>>5 == kb` |
| C3 归约树 | 每 (warp,q,nn) 一棵 `shfl_xor` off=16,8,4,2,1 | **同** | 未触碰；每元素仍恰由一个 warp 算、一棵树 |
| C4/C5 无跨行/跨 K 重组 | `acc[q][nn]` 独立链 | **同** | 未触碰；无 K-split |
| C6 累加式 | `acc[q][nn] += av[q]*wv`，`#pragma unroll 32` | **同** | 未触碰 |
| a32 两臂 | 单一内联形式，对 `DSV41_GEMV_A32` 两个取值都逐位等价 | **同** | 未触碰（`av[q]` 的算式未变，只换了两个操作数的**读法**） |

**staging 是纯拷贝**这一点是整条论证的支点，而它也是 legacy 自己的支点（`mrows` kernel 的
`DSV41_MROWS_ACT_CPASYNC` 头注：scalar→cp.async16 的 A/B 是 bit-identical 的，因为 staging 不碰值）。
**barrier 只提供可见性，不提供算术**：`__syncthreads()` / `__syncwarp()` 都不是浮点操作。

**唯一可观测变化**：`s_lut` 从 1 份变 `nw` 份、`s_w` 的 stage 从 block 平铺变 warp 平铺 —— 全是
**传输/发布的调度**，不进入任何乘积、任何求和顺序。

---

## §4 smem 预算表

```
smem(fix) = nw*bn*k              // 权重 tile（每 warp 自己的 bn 行）
          + nw*256*4             // e4m3 表：每 warp 一份完整副本（nw KB）
          + m*(k/32)*4           // 激活 scale 行（仅 AQ==0）
          + m*k                  // 激活 fp8 行（仅 AQ==0）
nw = min( dsv41_mrows_warps_for(n), max(1, n/(bn*SM)) )      // SM = 148
```
`AQ == 1`（wo-pair-rows）：无后两项，`smem = nw*bn*k + nw*256*4`。

| 形状 | n × k | m | nw(bn=1/2/3/4) | bn=1 | bn=2 | bn=3 | bn=4 | legacy | 
|---|---|---:|---|---:|---:|---:|---:|---:|
| `wkv` | 512 × 5120 | 6 | 3/1/1/1 | **51.75 KB** | **44.75 KB** | **49.75 KB** | **54.75 KB** | 54.75 KB |
| `wq_a` | 1280 × 5120 | 5 | 4/4/2/2 | **52.13 KB** | **72.13 KB** | **60.13 KB** | **70.13 KB** | 49.13 KB |
| `wq_b` | 4096 × 1280 | 6 | 8/8/8/6 | 26.44 KB | 36.44 KB | 46.44 KB | **44.44 KB** | 19.44 KB |
| `wo_b` | 5120 × 1024 | 5 | 8/8/8/8 | 21.63 KB | 29.63 KB | 37.63 KB | **45.63 KB** | 14.63 KB |

**读表**：
- **没有形状越过 device ceiling**（`dsv41_smem_ceiling` ≈ optin max − 1 KB ≈ 226 KB；最宽的
  `wq_a` bn=2 是 72.13 KB）。`> 48 KB` 的形状（`wkv` bn∈{1,3,4}、`wq_a` 全臂、`wq_b` bn∈{3,4}）
  走 `cudaFuncSetAttribute(…MaxDynamicSharedMemorySize)` —— **launcher 已有这条分支，无需新增**。
- **bn=1 时 `wkv` 的 smem 是 51.75 KB < legacy 54.75 KB**（权重从 3×5120 变 3×5120 相同 + LUT 3 KB vs 1 KB，
  而 legacy 的权重是 nwarps=4 行）⇒ bn=1 对照臂的 smem **不大于** legacy，对照实验的前提在修复后依然成立。
- `wq_b`/`wo_b` 变大约 17/15 KB —— 全来自 **每 warp 一份 LUT**（nw=8 时 8 KB）与激活 slab。
  **不构成占用瓶颈**：这两个形状的寄存器（96 regs @ 256 线程）已经把块数限制在 **2 块/SM**，
  smem 侧 231424/37312 = 6 块/SM 不是 binding resource。
- `wq_a` bn=2 的 72 KB：3 块/SM（smem 侧），寄存器侧仍是 2 块/SM ⇒ 不受影响。

---

## §5 编译验证（compile-only，无 GPU）

### 5.1 本机 cargo check

```
cargo check --workspace --all-targets   → EXIT=0（仅既存 warning）
```

Rust 侧不编译 `.cu`（运行时 dlopen `libferrite_kernels.so`），所以本行只证明 ABI/头文件面未破。

### 5.2 远端 nvcc（CUDA 13.2, `ubuntu@43.202.208.136`, 无 GPU）

```
nvcc -gencode arch=compute_100a,code=sm_100a -O3 --use_fast_math -std=c++17 \
     -Xptxas -v -c dsv41_kernels.cu -o dsv41_kernels.o        # EXIT=0, 0 errors
```

### 5.3 ptxas 账（修复后，`-Xptxas -v`，全部 **0 stack / 0 spill**）

**`gemm_fp8_mtile_kernel<M, AQ=0>`（fp8 激活臂 —— 本次修复的主对象）**

| M | regs | spill | stack | barriers | g2 基线 regs | 变化 |
|---:|---:|---|---|---:|---:|---|
| 1 | 40 | 0 | 0 | 1 | 40 | = |
| 2 | 48 | 0 | 0 | 1 | 53 | **−5** |
| 3 | 63 | 0 | 0 | 1 | 63 | = |
| 4 | 60 | 0 | 0 | 1 | 73 | **−13** |
| 5 | 64 | 0 | 0 | 1 | 80 | **−16** |
| **6** | **76** | **0** | **0** | **1** | **96** | **−20** |
| 7 | 76 | 0 | 0 | 1 | 92 | −16 |
| 8 | 128 | 0 | 0 | 1 | 96 | +32 |

**`gemm_fp8_mtile_kernel<M, AQ=1>`（wo-pair-rows 臂 —— 继承修法 2）**

| M | regs | spill | stack | **barriers** |
|---:|---:|---|---|---:|
| 1 | 40 | 0 | 0 | **0** |
| 2 | 48 | 0 | 0 | **0** |
| 3 | 61 | 0 | 0 | **0** |
| 4 | 64 | 0 | 0 | **0** |
| 5 | 64 | 0 | 0 | **0** |
| **6** | **75** | **0** | **0** | **0** |
| 7 | 80 | 0 | 0 | **0** |
| 8 | 103 | 0 | 0 | **0** |

**读表（三条结论）**：

1. **M=6（生产 m）的寄存器从 96 降到 76**。修复不是「加 smem 换寄存器」而是**两者都省**：
   激活改读 `s_a`/`s_as` 后，`__ldg` 的地址算式（`q*k`、`q*nb_k`、两个 64 位全局基址）不再需要常驻，
   编译器把它们换成了两个 smem 基址 + 便宜得多的 LDS。⇒ §5.1 担心的「regs 上升掉占用」**没有发生**；
   256 线程 × 76 regs = 19456 regs/块 ⇒ `65536/19456 = 3` 块/SM（**比 g2 的 2 块/SM 更好**）。
2. **barriers 对 `AQ == 0` 仍是 1**（激活那一个 = legacy 自己的那个），对 `AQ == 1` **降到 0**
   （它没有激活 staging ⇒ 整块 kernel 再无 block barrier）。**barrier 数不是负向来源，它的语义才是**——
   修复前那一个 barrier gate 的是 `nw*bn*k`（整块权重 tile），现在 gate 的是 `m*k`（激活，每块相同、L2 驻留）。
3. **M=8 的 128 regs** 是 `__launch_bounds__(256)` 的上限（32768/块 ⇒ 2 块/SM）。生产 `m <= 6`，
   且 launcher 对 m 的 dispatch 上界就是 8，这个形状不在 verify 的路径上；**无 spill**，不是缺陷。

---

## §6 GPU 验证手册（主 agent 执行；本任务禁止 GPU）

纪律沿用 audit §5.3：**只跑 micro bench**（`tests_dsv41_gemm_mrows.cu` 二进制），不跑 e2e NCU。

### 6.1 构建

```bash
nvcc -gencode arch=compute_100a,code=sm_100a -O3 --use_fast_math -std=c++17 \
     -o /tmp/mtile_$USER/t_gemm_mrows kernels/cuda/tests_dsv41_gemm_mrows.cu
```

### 6.2 逐位等价验收（**先于性能**）

`g_mrows_mtile` / `g_mrows_mtile_bn` 是**文件级 static（load 时 getenv）** ⇒ **一进程一值**：

```bash
for bn in 1 2 3 4; do
  echo "== DSV41_MROWS_MTILE=1 DSV41_MROWS_MTILE_BN=$bn =="
  DSV41_MROWS_MTILE=1 DSV41_MROWS_MTILE_BN=$bn /tmp/mtile_$USER/t_gemm_mrows --quick
done
DSV41_MROWS_MTILE=1 /tmp/mtile_$USER/t_gemm_mrows        # 全量（含 dispatch/m、n%nwarps、tiny）
```

**判据**：每个臂 `RESULT: all checks passed`（m 行 launch 与 m 个 m=1 launch **逐位**相等，
sentinel 覆盖无空洞）。任一位差 ⇒ 停，回报。每个臂必须打印 `[mrows-mtile] ARMED …` 一行
（没打印说明没走 M-tile，那次运行不算数）。**`bn=1` 是本次修复的对照组**：它的期望是
「≈ legacy」，不是「更快」——它现在的结构就是 legacy。

### 6.3 性能验收（micro bench，m=1 为基准）

```bash
                      /tmp/mtile_$USER/t_gemm_mrows --quick                        # legacy
DSV41_MROWS_MTILE=1   /tmp/mtile_$USER/t_gemm_mrows --quick                        # fix bn=2
DSV41_MROWS_MTILE=1 DSV41_MROWS_MTILE_BN=1 /tmp/mtile_$USER/t_gemm_mrows --quick   # ← 对照臂
DSV41_MROWS_MTILE=1 DSV41_MROWS_MTILE_BN=4 /tmp/mtile_$USER/t_gemm_mrows --quick
```

**验收判据（按优先级）**：
1. **先看 `bn=1` 对照臂的符号**：修复前它是 **+2.54 ms**。修复若正确，它应落到 **≈ legacy（±1%）**——
   这是「两个结构病灶就是全部负向」的直接检验。
2. **再看 `bn=2/4` 的符号**：目标 `t(m=6)/t(m=1) ≈ 1.0~1.3`（legacy 是 2~4×）。
   **修复后 bn 的效应应该变干净**：bn=1 vs bn=2 vs bn=4 的差只应反映**指令数**（§2.2 的 0.65×/0.47×），
   不应再有 bn=1 那种「结构输了」的伪差。
3. **若 bn=1 仍显著慢** ⇒ 病灶不止 (a)/(c′)，回报（不要拿 bn 调参掩盖）。
4. **若 bn=1 变快但 bn≥2 反而慢** ⇒ 新引入的成本在 smem/占用侧，查 §5.3 的 regs（1 块/SM 的征兆）。

### 6.4 e2e 双门禁（主 agent，授权后）

```
DSV41_MROWS_MTILE=1 DSV41_MROWS_MTILE_BN={2,4}   （一臂一进程，计数 200 tok，读 [dspark] steps=50）
```
每个臂必须**同时**报告：`step_ms`（票面 verify −3~8 ms）与 `mean-k`（S1 修复后 2.240；**掉了 = 数值回归，弃用该 gate**）。

### 6.5 nsys 判据

`t(mrows kernel, m=6) / t(m=1) ≈ 1× 单行`（legacy ~5×），且 kernel 表里**只有** `gemm_fp8_mtile_kernel`。

### 6.6 回滚

```bash
unset DSV41_MROWS_MTILE     # 立即回到 M-in-register 程序（逐字节）
```

本修复是**修复版替换**（不是 `…_V2` 并存）：没有保留 g2 的旧 body，因为 bn=1 对照已判定它是
结构性负向；回滚点就是 gate OFF（走 legacy/⑤a/MPAR 原路径，逐字节不变）。

---

## §7 待定 / 风险

1. **符号仍未定（头号）**：修复去掉了两处结构性病灶，但 M-tile 的赌注仍依赖「指令数决定时间」。
   bn=1 对照修复后必须 ≈ legacy —— 若成立，说明 (a)+(c′) 就是 g2 的全部负向，bn≥2 的收益回到指令账；
   若不成立，第三种病（LDS 通道饱和/占用）仍未被排除。
2. **`bn=4` 的 SM 覆盖**：`wkv` 的 grid = ceil(512/4) = 128 < 148（nw 被 clamp 到 1）⇒ 掉覆盖率。
   sweep 必须带上 bn=4 看这个效应，别把它误读成「结论」。
3. **per-warp LUT 的 smem 成本**：`wq_b`/`wo_b` +8 KB。当前不 binding（寄存器已限 2 块/SM），
   但若未来寄存器压力下降，这 8 KB 会在 nw=8 的形状上变成占用约束。
4. **`a_scale` 的 `__ldg` 已随修法 1 消失**（改读 `s_as`）⇒ g2 §7.3 的「uncoalesced scalar LDG」
   风险项**关闭**。
5. **`AQ == 1` 未做同类修复**：它的激活是 f32 且在 warp 内量化，`__ldg` 是它的设计（不是病灶）；
   它继承的是权重侧与 LUT 侧的修法 2。
