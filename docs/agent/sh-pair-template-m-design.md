# SH_PAIR `template<M>` —— shared expert 三段一核的实现设计

> 工部 · 2026-09-12 · **只读分析 + 本文档（唯一产出）**。未改动任何源码、未执行任何 GPU 命令。
> 任务来源：`verify-operator-optimization-list.md` #6b（唯一真正的 5×→1× 融合，预估 −4.9~−7.9ms，3.5 人日）。
> 代码基线：HEAD `51235f6`。
> 上游设计：`verify-family-fusion.md` §3.1（W3）/ §2.1 P1+P3；`verify-architecture-floor.md` §5.1/§5.2。

---

## 0. 结论先行（五条）

1. **本文档给出的是一个「三段一核 + M 行进 grid」的 kernel，不是一个「M 行折叠进寄存器」的 kernel。**
   原因见 §3.3：phase 1（w1|w3）的 bit-identical 并行度上限是 `ceil(n1/32) × M`，把 M 折进 warp
   寄存器会把并行度从 **54 个 block 降到 9 个**（B300 148 SM），指令数省 48% 但 SM 覆盖掉 6×，
   **净亏 2.7×**。所以默认 `fold_r = 1`（M 进 grid），`fold_r` 作为 A/B 旋钮保留。

2. **现有 M=1 kernel 有一个必须先说清楚的缺陷**：`gemm_fp8_sh_pair_kernel` 的 grid 按
   `max(n1, n2)` 定尺（= 160 block），而 phase 1 的映射是 `row = blockIdx.x*32 + warp`、guard
   `base < n1` ⇒ **phase 1 只有 9/160 个 block 在干活，151 个 block 在 barrier 上空转**。
   `template<M>` 顺手把这件事修掉（phase 1 变成 `M × ceil(n1/32)` 个 block）。见 §2.3。

3. **收益的主项是 launch 数（1000 → 80/步），不是字节。**
   885 MB → 177 MB 只在 phase 2（w2）兑现（M-fold 免费，且那里并行度富余）；phase 1 的 w1|w3
   仍是 M 次读，但那是 **L2 命中**（2.95 MB 常驻），不是 HBM。这与「mrows 零收益」的实测一致
   （字节不是这族的天花板），因此本设计**不为了省字节牺牲并行度**。

4. **三段一核的 SMEM 只有 33.4 KB（RF=1）**，远超预期（设计文档口径的「30KB 激活 + 权重 tile」
   是 M-fold 形状；RF=1 只需 1 行激活 = 20 KB `s_af`）。occupancy 由**寄存器**而非 SMEM 决定，
   落在 **1~2 CTA/SM（32~64 warps/SM）**，`co_res = 148~296`。见 §4。

5. **数值域是「逐位相等」，不是「容差相等」**：本 kernel 的每一条链都是把
   `gemm_fp8_gemv_kernel` mode-4 + a32 的 consume 循环、`swiglu_limit_kernel` 的 clamp+silu、
   `swiglu_limit_q_kernel` 的 amax+emit、`gemm_fp8_gemv_kernel` 的 phase 2 **逐字转写**（C1–C8）。
   ⚠️ 但 `build.sh` 默认 `--use_fast_math`，**新写的表达式必须与旧核字面一致**（多路展开 +
   分裂累加器会漂 ~1 ULP/层，见 `build.sh:43` 的警告）。见 §5。

---

## 1. 现状精确解剖（读码结论，非估计）

### 1.1 `gemm_fp8_sh_pair_kernel`（`kernels/cuda/dsv41_kernels.cu:6511`）

```
签名  (a, a_scale, wg, wg_scale, wu, wu_scale, limit, n1, k1,
       act, aq, aqsc, w2, w2_scale, n2, out, cpasync)
block  __launch_bounds__(1024) = 32 warps（fp8 emit 要求「一块 = 一个 32 行 scale block」）
```

**phase 1**（`:6548`–`:6629`）

| 项 | 事实 |
|---|---|
| 映射 | `base = blockIdx.x*32; if (base < n1) { row = base + warp; }` —— **一个 warp 拥有一个 inter 行**，同时走 `w1[row,:]` 与 `w3[row,:]`（PAIR 结构，gate/up 共享同一个 `av`） |
| 并行度 | `ceil(n1/32)` 个 block。生产 `n1 = sh_il = 2304/8 = 288` ⇒ **9 个 block** |
| 激活 | `s_af[k1]` f32 **块级物化一次**（`s_lut[a[i]] * s_as[i>>5]`，`a32_direct` 一趟从 global 解码），32 个 warp 共享读 |
| 权重 | **直接从 global 读**（`g_row[j]` / `u_row[j]`），不进 SMEM |
| epilogue | `fminf` clamp → `(g/(1+expf(-g)))*u` → `act[row] = v`；`s_rows[warp] = v` → `__syncthreads()` → 32 值 shuffle 树 amax → `aqsc[blockIdx.x]`、`aq[row]` |

**grid barrier**（`:6631`–`:6648`）：module 级 `__device__ unsigned g_sh_arrive, g_sh_sense`（`:6508`），
sense-reversing，thread 0 到达、`__threadfence()` 释放、`__nanosleep(32)` 自旋。

**phase 2**（`:6650`–`:6718`）：`for (row = blockIdx.x*32+warp; row < n2; row += gridDim.x*32)`，
k = n1，`w2` 行用 `cp.async16` 预取进 `s_w[warp][n1]`，`s_af` 由 `aq/aqsc` 重新物化。

**launcher**（`:6737`–`:6808`）

```
warps = 32（硬钉）
gsmem = 32*n1 + 32*nb_k2_al + 256*4 + nb_kmax*4 + 32*4 + k1max*4     :6769
blocks = ceil(max(n1, n2)/32)，上限 co_res（cudaOccupancyMaxActiveBlocksPerMultiprocessor × SM 数）
decline（返回 2，绝不返回 1）: mode != 4 / a32 off / a32_staged / k1&31 / n1&31 / 任何指针为空
```

**生产形状下的具体数字**（`dim = 5120`，`inter = 2304`，`world = 8`，`sh_il = 288`）：

| 量 | 值 |
|---|---|
| `n1` = w1/w3 行数 = `sh_il` | **288** |
| `n2` = w2 行数 = `dim` | **5120** |
| `k1` = phase-1 收缩宽 = `dim` | **5120**（160 个 kb） |
| `k2` = phase-2 收缩宽 = `n1` | **288**（9 个 kb） |
| `nb_k1 / nb_k2 / nb_k2_al` | 160 / 9 / 24 |
| `gsmem` | **32256 B ≈ 31.5 KB** |
| `blocks` | `ceil(5120/32) = 160`，capped 到 `co_res` |
| **phase-1 实际干活 block** | **9**（151 个 block 跳过 phase 1 直接到 barrier） |

### 1.2 逐行路径的对照（本 kernel 要替换的 5 发）

`chain_dev.rs:11246`–`:11337`（`moe_rows` 的 per-row 循环）：

```
quant1(xn_r + r*dim)  →  gemm_fp8_mx2(w1|w3)  →  swiglu_limit  →  quant1  →  gemm_fp8_mx_add(w2)
```

**已接线的两条 mrows 路径**（`shared_expert_mrows`，`:11422`）：

| 路径 | 每层 launch 数（m=5） | 备注 |
|---|---:|---|
| per-row 循环 | 25 | 基线 |
| `DSV41_SH_EXP_MROWS`（A5 add 折进 w2） | **6** = `quant_rows` + `mrows(w1)` + `mrows(w3)` + `swiglu_q` + `mrows(w2)` + `add` | 实测零收益（`arch-floor`） |
| `DSV41_SH_EXP_FUSED + DSV41_SH_PAIR`（M=1 逐行发 SH_PAIR） | **7** = `quant_rows` + `m×1 sh_pair` + `add` | 未上机 A/B |
| **本设计 `template<M>`** | **2** = `quant_rows` + `sh_exp_fused<M>`（可再折到 1） | 40~80 发/步 |

---

## 2. 设计：`gemm_fp8_sh_exp_pair_kernel<M>`

### 2.1 总体结构

```
grid (G, 1, 1)，block (1024, 1, 1) = 32 warps，动态 SMEM

  ┌ phase 0（可选，日后）  fp8 quant of xn_r[m][dim] → SMEM/global
  │
  ├ phase 1   (i, r) 二维拆分：blockIdx.x < n1t*ceil(M/fold_r) 的 block
  │           block 拥有 (i-tile it = blockIdx.x % n1t, 激活行组 r0..r0+fold_r-1)
  │           warp w 拥有 inter 行 i = it*32 + w，对每个 r 走 w1[i,:] 与 w3[i,:]（PAIR）
  │           → clamp+silu →（block 级 32 值 amax 树）→ aq[r][i] fp8 + aqsc[r][i/32]
  │
  ├ grid barrier（复用 g_sh_arrive / g_sh_sense，sense-reversing）
  │
  └ phase 2   ALL blocks，grid-strided：warp 拥有 w2 输出行 j，acc[M] 链（M-fold）
              → out[r*n2 + j]（可带 epi_add：`o + (acc + bias)`）
```

### 2.2 kernel 签名

```cuda
// dsv41_kernels.cu —— 紧跟现有 gemm_fp8_sh_pair_kernel 之后
template <int M>
__global__ void __launch_bounds__(1024)
gemm_fp8_sh_exp_pair_kernel(const uint8_t* __restrict__ a,        // [M, k1]   fp8 e4m3
                            const float*   __restrict__ a_scale,  // [M, k1/32] f32
                            const uint8_t* __restrict__ wg, const uint8_t* __restrict__ wg_scale,
                            const uint8_t* __restrict__ wu, const uint8_t* __restrict__ wu_scale,
                            float limit, int n1, int k1,
                            float*   __restrict__ act,            // [M, 2*n1] f32  —— 可空（by-product）
                            int act_stride,
                            uint8_t* __restrict__ aq,             // [M, n1]    fp8  —— phase-1 出 / phase-2 入
                            float*   __restrict__ aqsc,           // [M, n1/32] f32
                            int aq_stride, int aqsc_stride,
                            const uint8_t* __restrict__ w2, const uint8_t* __restrict__ w2_scale,
                            int n2, int out_stride, int epi_add,
                            float* __restrict__ out, int fold_r, int cpasync) {
```

`M` 是**编译期**模板参（`acc[M]`、`s_rows[fold_r][32]` 所需的数组维）；`fold_r` 是**运行期**参
（A/B 旋钮，`1..M`）。

### 2.3 grid 设计（**这是本设计最关键的一处修正**）

```c
const int n1t  = (n1 + 31) >> 5;                       // phase-1 i-tile 数 = 9
const int nsr  = (M + fold_r - 1) / fold_r;            // 激活行组数（fold_r=1 ⇒ M）
const int p1b  = n1t * nsr;                            // phase-1 block 数 = 9*M
const int g2t  = (n2 + 31) >> 5;                       // phase-2 j-tile 数 = 160
int G = (p1b > g2t) ? p1b : g2t;                       // 合并 grid
if (G > co_res) G = co_res;                            // 硬保护
if (p1b > G) return 2;                                 // phase-1 覆盖不全 ⇒ decline（绝不截断）
```

角色判定（block 内 uniform）：

```c
const bool p1 = (blockIdx.x < p1b);
const int  it = p1 ? (blockIdx.x % n1t) : -1;
const int  rg = p1 ? (blockIdx.x / n1t) : -1;          // 激活行组
```

**为什么 phase 1 必须进 grid 而不是折进 warp**（→ §3.3 的完整账）：
`n1 = 288` 时，bit-identical 的并行度只有两个轴：`i`（288）与 `r`（M）。折进 warp 只剩 288 条链
（9 block）；进 grid 有 `M × 288` 条链（`M × 9 = 54` block）。B300 有 148 个 SM，9 block 的
SM 覆盖是 6%，54 block 是 36%。

**为什么 `p1b > G` 要 decline 而不是截断**：phase 1 的 32 个 warp 必须属于**同一个 32 行 scale
block**（amax 树的前提）。截断会把某个 scale block 拆到两个 launch 里，amax 就错了。

### 2.4 phase 1 明细

```cuda
if (p1) {
    const int r0 = rg * fold_r;
    const int rn = min(fold_r, M - r0);
    const int i  = it * 32 + warp;                     // 本 warp 的 inter 行
    const uint8_t* g_row = wg + (size_t)i * k1;        // w1 行（global，L2 命中）
    const uint8_t* u_row = wu + (size_t)i * k1;        // w3 行
    const uint8_t* gsc   = wg_scale + (size_t)(i >> 5) * nb_k1;
    const uint8_t* usc   = wu_scale + (size_t)(i >> 5) * nb_k1;
    float g[fold_r], u[fold_r];
    #pragma unroll
    for (int q = 0; q < fold_r; ++q) { g[q] = 0.f; u[q] = 0.f; }

    #pragma unroll 4
    for (int kb = 0; kb < nb_k1; ++kb) {
        const float sbg = ue8m0_to_f(gsc[kb]);
        const float sbu = ue8m0_to_f(usc[kb]);
        const int j = (kb << 5) + lane;
        const uint8_t wgv = g_row[j], wuv = u_row[j];          // 2 次 LDG.8
        #pragma unroll
        for (int q = 0; q < fold_r; ++q) {
            const float av = s_af[(size_t)q * k1 + j];         // RF=1: 1 次 LDS.32（块级共享）
            g[q] += av * (s_lut[wgv] * sbg);
            u[q] += av * (s_lut[wuv] * sbu);
        }
    }
    ...
}
```

- **`fold_r == 1` 时 `s_af[k1]` 是块级共享的 f32（20 KB）**，与 M=1 kernel **同一条读路径**
  （`s_af[j]`）—— 这是最强的逐位等价形式，也是 RF=1 的另一个理由。
- **`fold_r > 1`** 时 `s_af` 会变成 `fold_r × k1 × 4`（RF=2 = 40 KB，RF=6 = 120 KB 不可行），
  此时改用 **mrows 的寄存器物化**（`af[q] = s_lut[s_pa[q*k1+j]] * s_pas[q*nb_k1 + (j>>5)]`，
  见 `gemm_fp8_mrows_kernel:5331` 的 C2 论证）。**两条路径都逐位等价**，代价是 SMEM / 指令数互换。

epilogue（每 r 一次）：

```cuda
if (lane == 0) { /* clamp + silu，与 swiglu_limit_kernel 逐字 */ }
// s_rows[q][warp] = v;  __syncthreads();
// 每 q 一棵 32-lane amax 树（fmaxf 精确且可结合，与 swiglu_limit_q_kernel 同值）
// aqsc[r*aqsc_stride + it] = sc;  aq[r*aq_stride + i] = byte
```

⚠️ **`s_rows` 必须是 `[fold_r][32]`**：`fold_r > 1` 时一个 block 覆盖多行，每行一棵独立树，
`__syncthreads()` 不能省（`fold_r` 次树、一次 barrier 即可）。

⚠️ **`act`（f32 swiglu 行）是 by-product**：M=1 arm 自己也只用它保持布局（`downstream 无读者`）。
本设计把 `act` 设为**可空**（`nullptr` ⇒ 跳过 `act[...] = v` 的 store），省 32 个 f32 store / (block, r)。

### 2.5 phase 2 明细（M-fold，这里才"每行只读一次"）

```cuda
for (int row = blockIdx.x * 32 + warp; row < n2; row += G * 32) {
    // 1) w2 行 → s_w[warp][n1]（cp.async16，与 M=1 kernel 的 P3 预取同规则）
    // 2) 逐 kb（k2 = n1，9 个 kb）：
    //      wv = s_lut[s_w[warp*n1 + j]] * ue8m0_to_f(s_ws[warp*nb_k2_al + kb]);
    //      for r: acc[r] += s_af2[r][j] * wv;      // M 条独立链，wv 一次解码 M 行复用
    // 3) 每 r：shuffle 树 → out[r*out_stride + row] = epi_add ? (out[...] + v) : v;
}
```

- 并行度 = `ceil(n2/32) = 160` 条链，**M-fold 不损失并行度**（5120 行 vs 160 block）。
- `wv` 提升到 r 循环外 = `gemm_fp8_mrows_kernel` 的 C4（同一 (row, kb) 值的复用，
  不触碰结合律）。
- **`epi_add`（A5）**：把 `add_inplace_raw(moe_out_r, sh_out_r)` 折成 read-modify-write，
  表达式 `o + (acc + bias)` 与 `gemm_fp8_mx_add` 完全同序 ⇒ 逐位相同，省 1 发/层（40/步）。

### 2.6 launcher 签名与 dispatch

```cuda
extern "C" int dsv41_gemm_fp8_sh_exp_fused(
    const uint8_t* a, const float* a_scale,
    const uint8_t* wg, const uint8_t* wg_scale,
    const uint8_t* wu, const uint8_t* wu_scale,
    float limit, int n1, int k1, int m,
    int fold_r, float* act, int act_stride,
    uint8_t* aq, float* aqsc, int aq_stride, int aqsc_stride,
    const uint8_t* w2, const uint8_t* w2_scale,
    int n2, int out_stride, int epi_add, float* out, cudaStream_t s);
```

- **dispatch**：`switch (m) { case 1: …<1>; … case 8: …<8>; default: return 2; }`。
- **decline（返回 2，绝不返回 1 —— 1 与 `cudaErrorInvalidValue` 撞码，是 r42/r43 的教训）**：
  `m ∉ [1,8]`、`fold_r ∉ [1,m]`、`n1%32`、`k1%32`、`m > 8`、任何指针为空、
  `n1t*ceil(m/fold_r) > co_res`、`mode != 4 || !a32 || a32_staged`、`DSV41_NO_GEMV_FP8` 置位。
- **`cq.async` 不用传**：phase 2 一律用 cp.async16（`n1%16==0` 已由 `n1%32==0` 保证），
  M=1 kernel 的 `cpasync` 门在这里没有 A/B 价值。
- **每 M 特化必须各自设 SMEM 属性**：`gemm_fp8_mrows` 的 launcher 已经吃过这个亏
  （`dsv41_kernels.cu:5426-5439`："EVERY M specialisation needs its own attribute … setting only
  `<1>` left `<m>` at the 48KB default → cudaErrorInvalidValue at m=5"）。写成宏：

```cuda
#define FERRITE_SET_SHP_M_SMEM(k)                                              \
    do { if (cudaFuncSetAttribute(gemm_fp8_sh_exp_pair_kernel<k>,              \
            cudaFuncAttributeMaxDynamicSharedMemorySize,                       \
            dsv41_smem_ceiling(gemm_fp8_sh_exp_pair_kernel<k>)) != cudaSuccess) \
        { (void)cudaGetLastError(); return 2; } } while (0)
    switch (m) { case 1: FERRITE_SET_SHP_M_SMEM(1); break; /* … 8 */ }
```

- **`co_res` 缓存**：照抄 M=1 launcher（`:6404-6418`）—— 用
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 并且**按 (gsmem, warps) 键控缓存**（每层跑一次，
  每次都是 driver round trip）。
- **不接 PDL**：grid barrier 与 PDL 互斥（M=1 kernel 的 header 已说明）。

---

## 3. SMEM 布局与预算

### 3.1 布局（一个 pool，phase 1 / phase 2 分段）

```cuda
extern __shared__ uint8_t s_pool[];
float*   s_lut  = (float*)s_pool;                       // [256]      f32  1024 B   两 phase 共用
// ---- phase 1 ----
uint8_t* s_pa   = s_pool + 1024;                        // [fold_r][k1]     fp8   （仅 fold_r>1 用）
float*   s_pas  = (float*)(s_pa + (size_t)fold_r*k1);   // [fold_r][nb_k1]  f32
float*   s_af   = s_pas + (size_t)fold_r*nb_k1;         // [fold_r][k1]     f32   （仅 fold_r==1 用 → 直接复用）
float*   s_rows = s_af + (size_t)fold_r*k1;             // [fold_r][32]     f32
// ---- phase 2 ----
uint8_t* s_w    = (uint8_t*)(s_rows + (size_t)fold_r*32);  // [32][n1]      fp8  cp.async16 目标（16B 对齐）
uint8_t* s_ws   = s_w + (size_t)32*n1;                     // [32][nb_k2_al] ue8m0
uint8_t* s_aq   = s_ws + (size_t)32*nb_k2_al;              // [M][n1]        fp8
float*   s_aqs  = (float*)(s_aq + (size_t)M*n1);           // [M][nb_k2]     f32
// pool 总字节 = 1024 + fold_r*(k1 + nb_k1*4 + k1*4|0 + 128) + 32*n1 + 32*nb_k2_al + M*n1 + M*nb_k2*4
```

**对齐不变量**（必须与 launcher 的字节表达式**完全一致**，drift 一行就是静默指针偏移 —— M=1
launcher 的 header 已经写明）：
`n1 % 16 == 0` ⇒ `s_w` 的每个 warp 行 16B 对齐；`s_pool` 本身 16B 对齐 ⇒ `cp.async16` 合法；
`s_lut` 之后所有 slot 都是 4B 的整数倍。

### 3.2 生产形状的字节账（`dim = 5120`，`n1 = 288`，`k1 = 5120`，`n2 = 5120`，`nb_k1=160`，`nb_k2_al=24`）

| slot | RF=1（默认） | RF=2 | RF=6（M-fold） |
|---|---:|---:|---:|
| `s_lut` | 1 024 | 1 024 | 1 024 |
| `s_af`（RF=1：块级 f32） | 20 480 | — | — |
| `s_pa`+`s_pas`（RF>1：fp8+scale） | — | 10 752 | 34 560 |
| `s_rows` | 128 | 256 | 768 |
| `s_w` | 9 216 | 9 216 | 9 216 |
| `s_ws` | 768 | 768 | 768 |
| `s_aq` | 1 728 | 1 728 | 1 728 |
| `s_aqs` | 216 | 216 | 216 |
| **合计** | **33 560 B ≈ 32.8 KB** | **33 960 B** | **48 280 B ≈ 47.2 KB** |
| M=1 kernel 对标 | 32 256 B | | |

> **设计文档口径修正**：`verify-family-fusion.md` 写的「30 KB 激活」是 **M-fold 形状**的
> `M × dim × fp8`。RF=1 的形状里每 block 只持 **1 行**激活（但它持 f32 的 `s_af`，20 KB）。
> 两者字节接近，但并行度差 6×。

### 3.3 为什么不是「M 行折叠进 warp」（phase 1 的并行度账）

指令模型（每 warp 每 kb）：`2 (w-decode) + 2 (w-mul) + fold_r × (1 a-LDS + 1 a-mul + 2 FMA)`
= `4 + 4·fold_r`。链数 = `n1t × ceil(M/fold_r) × 32` warp，各 `nb_k1 = 160` kb。

| 策略 | warp 数 | 每 warp 指令 | 总指令 | **占 SM 数** | 模型周期（4 IPC/SM） |
|---|---:|---:|---:|---:|---:|
| `fold_r = 1`（M 进 grid） | 54×32 = 1 728 | 160×8 = 1 280 | **2.21 M** | **54** | **10 240** |
| `fold_r = 2` | 27×32 = 864 | 160×12 = 1 920 | 1.66 M | 27 | 15 360 |
| `fold_r = 6`（纯 M-fold） | 9×32 = 288 | 160×28 = 4 480 | 1.29 M | 9 | 35 840 |
| （参照）M=1 kernel 实况 | 288 | 160×(2 w + 1 a + 2 FMA + 2 mul) | — | **9** | ≈ 9 block 全速 |

**结论：`fold_r = 1` 快 1.5×（vs RF=2）、3.5×（vs RF=6）。** 省下的指令（−42%）被 SM 覆盖
（÷6）吃掉了。这与 `arch-floor §5.2` 的判词一致：「mrows（只折字节/只加寄存器累加器）不动
那两堵墙中的任何一堵」——它反而**降低了在飞 block 数**。

> ⚠️ **诚实的边界**：这个模型假设 1 CTA/SM 时 32 warps 足够喂满 4 IPC。真实数字必须由
> §7 的 A/B 定夺（`fold_r` 是 runtime 参，所以 A/B 不需要重编译）。

### 3.4 为什么不做 k-split

phase 1 的 k 维（5120 = 160 kb）是最大的可切轴，但**切了就不逐位等价**：
`acc = (((a0+a1)+a2)+a3)` vs 两段 `(a0+a1)+(a2+a3)` 在 f32 下**不等**（非结合）。
M=1 kernel 的并行度上限因此**结构性地**钉在 `n1 × M`。这是本设计的硬边界，也是 §3.3 必须
在「并行度 / 指令数」之间选的原因。**不做 k-split，不引入 tolerance 口径。**

---

## 4. Occupancy 分析（B300 / sm_100a）

| 约束 | 值 | 推出的 CTA/SM |
|---|---|---|
| SMEM | 227 KB/SM（opt-in），本核 32.8 KB | ⌊227/32.8⌋ = **6** |
| 线程 | 2 048 threads/SM，block = 1 024 | ⌊2048/1024⌋ = **2** |
| 寄存器 | 65 536 regs/SM，128 regs/thread 上限 | `⌊65536/(1024·R)⌋` |

寄存器估算（RF=1 路径）：phase-1 累加器 `g/u`（2）+ 循环/地址临时 ≈ **40~56**；phase-2 的
`acc[M] + s_af2`（2M = 12）+ 临时 ≈ **48~64**。两个 phase 不同时活跃，取 max：

| R | CTA/SM | threads/SM | warps/SM | 占用率（/64 warps） |
|---:|---:|---:|---:|---:|
| 32 | 2 | 2 048 | 64 | 100% |
| 40 | 1 | 1 024 | 32 | 50% |
| **64（预期）** | **1** | **1 024** | **32** | **50%** |

⇒ **`co_res = per_sm × 148 = 148 ~ 296`**（实际以 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
为准；M=1 kernel 也是这个量级）。

**对 grid 的影响**：

- `M = 6`（swallow）：`p1b = 9×6 = 54`，`g2t = 160` ⇒ `G = 160`。若 `co_res = 148` ⇒ `G = 148`
  ⇒ phase 2 的尾部 12 个 tile 走 grid-strided 的第二趟；phase 1 的 54 个 block 全部落在
  `blockIdx.x < 54 < 148` ✅。
- `M = 8`：`p1b = 72 ≤ 148` ✅。
- **必须保留的 guard**：`if (p1b > G) return 2;`（`G` 已被 `co_res` 截断）。

**提高占用的旋钮**（写进 kernel，但第一版不动）：

1. `__launch_bounds__(1024, 2)` 强压 regs ≤ 32 → 2 CTA/SM（有 spill 风险，需 A/B）。
2. block 降到 512 threads（16 warps），**每 warp 拥有 2 个 inter 行**（`s_rows[fold_r][32]` 的树
   仍然成立：16 warp × 2 值 = 32 值）。block 数 ×2（phase 1 = 54 个 512-thread block 不变 ——
   不对，i-tile 变小 ⇒ `n1t` 变 18，`p1b = 18×6 = 108` block）。**这是唯一能同时提高 block 数
   与 SM 覆盖的形状**，列为 A/B 的第二旋钮。

---

## 5. 数值域：与逐行版的逐位等价论证（C1–C8）

**验收口径不变**（`NEXT-SESSION-HANDBACK §2`）：四段文本逐字 + `faults=0` + 同会话背靠背 A/B。
下列每条都是「逐字转写」级别的论证，**不是容差**。

| # | 环节 | 论证 |
|---|---|---|
| **C1** | 激活 fp8（phase 边界） | 仍然由 `quant_rows`（或 M=1 arm 的 `gemm_fp8_sh_pair` 前的 `quant_rows`）生产 —— `quant_kernel` 一个 thread-group 一个 (row, block)、每行 amax 是自己的 32-lane shuffle 树 ⇒ 与 m 次 `quant1` 逐字节相同（该结论 `chain_store.rs` 的 `xq_r` 注释已记录）。本 kernel **只消费** fp8 字节。 |
| **C2** | 激活物化 | RF=1：读 `s_af[j]`，其填充是 `s_lut[a[i]] * s_as[i>>5]`（`a32_direct` 一趟），与 M=1 kernel `:6560-6570` **同一段代码**。RF>1：寄存器物化 `s_lut[s_pa[...]] * s_pas[...]`，与 `dsv41_a32_mat4` / 三个 scalar fill 路径的同值同序（`gemm_fp8_mrows_kernel` 的 C2 已论证「两个 arm 逐位相同 by construction」）。 |
| **C3** | K 走序 | `kb` 升序、`j = kb*32 + lane`，**每链单个串行 `+=`**。`#pragma unroll N` 只能重叠 load，不能重结合串行链。 |
| **C4** | 归约树 | `for (off=16; off>0; off>>=1) acc += __shfl_xor_sync(...)` —— 与 M=1 kernel `:6598`/`:6715` 逐字相同，每 (chain, r) 跑一次。 |
| **C5** | 无跨行重结合 | 每 r 一条独立链（`acc[r]` / `g[q]`），任何地方都不跨 r 相加。 |
| **C6** | swiglu epilogue | `limit > 0 ? fminf(g, limit) : g`、`fminf(fmaxf(u,-limit),limit)`、`(g/(1+expf(-g)))*u` —— 与 `swiglu_limit_kernel`（`dsv41_glue.cu:183-187`）逐字。 |
| **C7** | fp8 emit | 一个 block 恰好拥有一个 32 行 scale block（`nwarps == 32` 且 `n1 % 32 == 0`）；amax 是 `fabsf` + `fmaxf` 的 32-lane 树（`fmaxf` 精确且可结合 ⇒ 与 `swiglu_limit_q_kernel` 的同 32 值树同结果）；`fmaxf(fast_round_scale(am, 1/448), 1e-30f)`、`fminf(fmaxf(v*(1/sc), -448), 448)`、`__nv_fp8_e4m3` —— 逐字。 |
| **C8** | phase 2 与 add | `s_lut[s_w[j]] * sb`、`kb` 升序、同一棵树；`epi_add` 的 `out[...] + v` 与 `gemm_fp8_mx_add` 的 read-modify-write 同结合序（`o + (acc + bias)`）。 |

**⚠️ 唯一的真风险（R2）**：`build.sh` **默认 `--use_fast_math`**，且脚本里的实测警告是
「多路展开 + 分裂累加器 + 普通运算符 ⇒ ~1 ULP/层漂移，40 层后模型崩掉」。对冲手段：

1. **照抄**旧核的表达式与 `#pragma unroll` 力度，不写"更优化"的形式；
2. 任何不得不改的结合点，用 `__fmaf_rn` / `__fadd_rn` 钉死；
3. **parity 用例是硬门**（§7 步骤 4）—— 逐字节比较，不看容差。

---

## 6. 与 verify(m=6) / SWALLOW_STEP 的集成

### 6.1 调用点（`chain_dev.rs::shared_expert_mrows`，插在现有 M=1 fused arm **之前**）

```rust
// 新增 gate：DSV41_SH_PAIR_M（默认 OFF）
if sh_pair_m() && self.dev.supports_sh_exp_fused()
    && m != 0 && m <= VERIFY_ROWS && m <= 8
    && (sh_il % 32) == 0 && (dim % 32) == 0
{
    self.quant_rows(self.s.xn_r.ptr as *const f32, m, dim as i32)?;   // 保留（步骤 4 可折）
    let ok = self.dev.gemm_fp8_sh_exp_fused(
        self.s.xq_r.ptr as *const u8,          // a       [m, dim]
        self.s.xsc_r.ptr as *const f32,        // a_scale [m, dim/32]
        w1.as_u8(), w1s.as_u8(), w3.as_u8(), w3s.as_u8(),
        cfg.swiglu_limit, sh_il as i32, dim as i32, m as i32,
        fold_r_default(),                      // 1（A/B 旋钮）
        std::ptr::null_mut(),                  // act = nullptr（by-product 不要）
        0,
        self.s.sh_aq_r.ptr as *mut u8,         // aq    [m, sh_il]
        self.s.sh_aqsc_r.ptr as *mut f32,      // aqsc  [m, sh_il/32]
        sh_il as i32, (sh_il / 32) as i32,
        w2.as_u8(), w2s.as_u8(),
        dim as i32, dim as i32,
        1,                                     // epi_add = 1（折进 moe_out_r）
        (self.s.moe_out_r.ptr as *mut f32).wrapping_add(r0 * dim as usize),
        self.dev.stream(),
    )?;
    if ok { return Ok(true); }                 // 单发 ⇒ 不会半途 decline，无需 all_rows 逻辑
}
// ↓ 落到现有 M=1 fused arm（SH_EXP_FUSED+SH_PAIR）→ SH_EXP_MROWS → per-row 循环
```

### 6.2 缓冲与形状（**全部已存在，无需新分配**）

| 缓冲 | 尺寸 | 本设计用途 |
|---|---|---|
| `xq_r` / `xsc_r` | `[VERIFY_ROWS, dim]` / `[VERIFY_ROWS, dim/32]` | `a` / `a_scale`（`quant_rows` 的行距 = `dim`，见 `quant_rows` (#5)） |
| `sh_aq_r` / `sh_aqsc_r` | `[VERIFY_ROWS, inter]` / `[VERIFY_ROWS, inter/32]` | `aq` / `aqsc`；**行距传 `sh_il` 与 `sh_il/32`**（M=1 arm 同款） |
| `moe_out_r` | `[m, dim]` | `out`（`epi_add=1` 直接折进，免 `sh_out_r` + `add`） |
| `sh_out_r` / `sh_act_r` | `[m, dim]` / `[m, 2*sh_il]` | **本 arm 不用**（回退路径仍用） |

**m=6（SWALLOW_STEP）形状检查**：`VERIFY_ROWS = 6` 且 `VERIFY_ROWS == DSPARK_DRAFTS + 1`
（`chain_dev.rs:97` 的 const assert）⇒ `m = 6`、`M = 6`、`sh_aq_r` 的 6 行 slot 够 ✅；
`dim % 32 == 0` ✅；`sh_il = 288`（TP8）或 `2304`（replicated）都 `% 32 == 0` ✅。

⚠️ **与 `VERIFY_HEAD_MROWS` 无冲突**（那条是 `argmax_rows` 与 rows=6 的死锁，`192ae83`）；
本 kernel 的 M=6 在 `TEMPLATE_RANGE` 内，不需要 `argmax_rows`。

### 6.3 每层 launch 拓扑（目标）

```
M=6：  quant_rows(1)  →  gemm_fp8_sh_exp_fused<6>(1)        = 2 发/层  ⇒ 80 发/步
步骤 4 折 quant 后：  gemm_fp8_sh_exp_fused<6>(1)            = 1 发/层  ⇒ 40 发/步
（对照：per-row 25/层 = 1000/步；SH_EXP_MROWS 6/层 = 240/步；SH_PAIR M=1 7/层 = 280/步）
```

---

## 7. 实施步骤（3.5 人日）

| 步 | 内容 | 产物 | 人日 |
|---|---|---|---:|
| **1 · 骨架 + launcher + 接线** | `template<M>` kernel 壳（phase 1/2 用旧核函数体占位）；`dsv41_gemm_fp8_sh_exp_fused` 的 dispatch / decline / `co_res` 缓存 / per-M `FERRITE_SET_SHP_M_SMEM` 宏；`device.rs` 的 `ko!` + `supports_sh_exp_fused()` + FFI 方法；`chain_dev.rs` 的 `sh_pair_m()` gate + first-try arm；`cargo check --workspace` + `build.sh` 出 .so | 可编译、gate OFF 时逐位走老路径 | 0.5 |
| **2 · phase 1 实体** | (i,r) grid 拆分 + `fold_r`（1：块级 `s_af`；>1：寄存器物化）；PAIR 双链；`s_rows[fold_r][32]` + amax 树 + fp8 emit；`act` 可空 | phase 1 与 M=1 kernel 同字节 | 1.0 |
| **3 · phase 2 实体** | M-fold `acc[M]` + `cp.async16` w2 行 staging + `epi_add` 折进 `moe_out_r` | phase 2 与 `gemm_fp8_gemv_kernel` 同字节 | 0.5 |
| **4 · parity 用例** | 扩 `tests_dsv41_glue.cu` 的 sh_pair case 到 `M ∈ {1..6}`：M 发核的 row r 输出 **逐字节** == M=1 核的 row r；再补 `epi_add ∈ {0,1}` 两臂；再补 `fold_r ∈ {1,2,M}` 三臂 | 编译期断言级 parity | 0.5 |
| **5 · A/B** | 同会话背靠背：`DSV41_SH_PAIR_M=1` vs OFF；四段文本逐字 + `faults=0`；nsys 按 kernel 名聚合数 launch + 每发 µs；扫 `fold_r`；决定默认值 | 结论 + 默认值 | 0.5 |
| **6 · 可选折叠（缓冲）** | (a) 折 `quant_rows` 进 kernel 的 phase 0（`quant_kernel` 逐字，`M×k1/32 = 960` 个独立 (row, block) 用 block 的 1024 线程）；(b) 去掉 `act` 参数 | 1 发/层 | 0.5 |
| | | **合计** | **3.5** |

**每步的验收判据**（不得跳过）：编译通过 → parity 逐字节 → 四段文本 + `faults=0` → 同会话 A/B。
**任一步不达 ⇒ 停在那一步回退，不往上叠。**

---

## 8. 风险表

| # | 风险 | 触发条件 | 应对 |
|---|---|---|---|
| **R1** | **barrier 死锁**（最严重） | `G > co_res`（SMEM/寄存器让 co_res 变小）；`p1b > G` 被截断 | launcher 硬 cap + `if (p1b > G) return 2;`；**绝不**越过 co_res（M=1 header 的 DEADLOCK SAFETY note） |
| **R2** | **逐位等价破功** | `--use_fast_math` 重结合；多路展开 / 分裂累加器 | 照抄旧核表达式与 unroll 力度；必要的结合点用 `__fmaf_rn`；parity 是硬门（§5 C1–C8） |
| **R3** | **SMEM 指针偏移**（静默） | kernel 的指针算术与 launcher 的 `gsmem` 表达式 drift | 两侧写在紧邻位置 + 注释互指（M=1 launcher `:6767` 的做法）；launcher 里对总字节加 `static_assert` 式注释 |
| **R4** | **per-M SMEM 属性漏设** | 只对 `<1>` 设了 opt-in ⇒ `<6>` 回落 48 KB 默认 | `FERRITE_SET_SHP_M_SMEM` 宏在 dispatch 里逐个设（`gemm_fp8_mrows:5430` 的原话："observed on serve as … cuda error 1"） |
| **R5** | **`aq` 与 `a` 别名**（cross-block race） | 复用 `xq_r` 当 phase-1 输出 | 传 `sh_aq_r`（**已存在且刻意分离**，`:552-559` 记录了这条 race）。**不许**复用 |
| **R6** | **Rust gate 与 .so gate 不一致**（项目 #1 测量陷阱） | `supports_sh_exp_fused()` 假 ⇒ 两臂都测老路径 | `ko!` 符号探测 + `BUILD_ID` 同源强制 + 一次性 warning |
| **R7** | **收益高估** | `fold_r=1` 的模型（§3.3）过于乐观；phase 1 的 54 block 覆盖不够 | 接受设计口径降级：本设计的**确定性收益是 launch 数**（1000→80），kernel 内部效率是**待测项**。若 `fold_r=1` 中性，直接试 512-thread + 2 行/warp 的形状（§4 旋钮 2） |
| **R8** | **m=6 的 `sh_aq_r` 越界** | 行距传成 `inter` 而 slot 按 `sh_il` 切 | 行距显式传参（`aq_stride=sh_il`），与 M=1 arm 用法逐字一致 |
| **R9** | **barrier state 共享** | 模块级 `g_sh_arrive/g_sh_sense` 与 M=1 kernel 共用 | 两条路径同流串行，且每次都自复位（`:6502-6507`）。**不要**让两者并发 |

---

## 9. 明确**不做**的事（划清范围）

1. **不做 k-split / cross-warp 部分和**：破坏 f32 结合序，逐位等价不成立（§3.4）。
2. **不换核到 tcgen05**：那是 `verify-operator-optimization-list` #7（routed experts）的路径；
   shared expert 的 w1|w3/w2 形状（n=288/5120，k=5120/288）与 e4m3 grouped 的 M=128 tile 不匹配，
   且会换数值域。本设计保留 SIMT fp8 decode。
3. **不改 `quant_rows` / `add_inplace_raw` 的现有调用点**：它们服务 mrows 与 per-row 回退路径。
   步骤 6 的折叠是**新增 phase 0 并保留旧入口**，不是替换。
4. **不动 `grouped routing` / `tcgen05` / attention / indexer**——别的 subagent 的战场。
5. **不动 `sh_act_r` 的分配**：回退路径仍要它；本 arm 只是不写。

---

## 10. 交付物清单（本设计）

| 文件 | 改动 |
|---|---|
| `kernels/cuda/dsv41_kernels.cu` | 新 `gemm_fp8_sh_exp_pair_kernel<M>`（紧跟 `:6511` 的 M=1 kernel 之后）+ `dsv41_gemm_fp8_sh_exp_fused` launcher + per-M SMEM 属性宏 |
| `crates/ferrite-models/src/dsv41/device.rs` | `kernels.gemm_fp8_sh_exp_fused`（`ko!`）+ `supports_sh_exp_fused()` + FFI 方法 |
| `crates/ferrite-models/src/dsv41/chain_dev.rs` | `sh_pair_m()` gate + `shared_expert_mrows` 的 first-try arm + `fold_r` 默认值 |
| `crates/ferrite-dsv41/tests/`（`tests_dsv41_glue.cu`） | sh_pair parity case 扩到 `M ∈ {1..6}` × `epi_add ∈ {0,1}` × `fold_r ∈ {1,2,M}` |
| `docs/agent/sh-pair-template-m-design.md` | 本文档 |

**兼容性**：无 breaking change。新符号 + 新 gate（默认 OFF）+ 老 `.so` 靠符号探测自动回退。

---

*工部 · 只读分析 + 本文件（唯一产出）；未执行任何 GPU 命令、未改动任何源码。*
*所有代码行号以 HEAD `51235f6` 为准，读码时以函数名为准。*
*设计口径的 ms 均标注来源；§3.3/§4 的周期数是模型值，真值由 §7 步骤 5 的 A/B 给出。*

---

## 11. 实施状态（工部 · 2026-09-12 · 实施轮）

> 本节由**实施轮**追加。§1–§10 是设计（未改动）；本节记录落地了什么、与设计的两处偏差、
> 以及主 agent 必须跑的东西。**未执行任何 GPU 命令**（本机无 nvcc / 无 GPU）。

### 11.1 落地的文件

| 文件 | 改动 |
|---|---|
| `kernels/cuda/dsv41_kernels.cu` | `ShPairMSmem` + `sh_pair_m_smem()`（**host/device 共用**的布局函数）、`template<int M> gemm_fp8_sh_exp_pair_kernel`、`extern "C" dsv41_gemm_fp8_sh_exp_fused`（decline / per-M SMEM 属性 / per-M `co_res` 缓存 / grid 计算）。紧跟 M=1 的 launcher 之后（原 `:6808`） |
| `crates/ferrite-models/src/dsv41/device.rs` | `Kernels::gemm_fp8_sh_exp_fused`（`ko!`）、`supports_sh_exp_fused()`、FFI 方法 `gemm_fp8_sh_exp_fused(...)` |
| `crates/ferrite-models/src/dsv41/chain_dev.rs` | `sh_pair_m()` / `sh_pair_m_fold()` 两个 gate；`shared_expert_mrows` 的 **first-try arm**（在 M=1 fused arm 之前） |
| `kernels/cuda/tests_dsv41_sh_exp_mrows.cu` | **新增** parity 套件（见 §11.4） |
| 本文件 | §11 |

### 11.2 与 §3.1 的两处 SMEM 偏差（都是"安全侧"，已写进 kernel 注释）

1. **`s_pas` 在 `fold_r == 1` 时保留 1 行**（设计的字节表算它 0）。生产形状 640 B；换来的是
   `fold_r == 1` 分支能**逐字照抄** M=1 kernel 的 `s_af` 填充（那段代码读的是 staged 的
   `s_as[i>>5]`，不保留就得改成裸读 global `a_scale`）。
2. **`s_af2` 是独立 slot（`M*n1` f32）**，不与 `s_af` 复用。换来 phase 2 对**每一行**都用
   M=1 kernel 的**同一条读路径**（`s_af[j]`，一个 LDS 一个操作数）。生产形状 M=6 多 6 912 B：
   总 smem 41 112 B < 48 KB（**不到 opt-in 阈值**），而 occupancy 由线程数决定（2 CTA/SM，§4），
   所以不移动。

生产形状（`n1=288, k1=5120, n2=5120, M=6, fold_r=1`）的 `gsmem = 41 112 B`。
`fold_r > 1` 时 `s_pa`+`s_pas` 替代 `s_af`：RF=6 时为 `6*5120 + 6*160*4 = 34 560` B（与 §3.2 一致），
总 54 824 B **> 48 KB** ⇒ 走 `FERRITE_SET_SHP_M_SMEM`（那里已经为每个 M 各设一次，R4）。

### 11.3 落地的关键实现选择（设计未钉死的几处）

| 项 | 选择 | 理由 |
|---|---|---|
| `fold_r > 1` 的寄存器数组 | `float g[M], u[M]` + `#pragma unroll` 全展开 + `if (q < fold_r)` 谓词 | `fold_r` 是**运行期**参，`g[fold_r]` 会掉 local memory；`M ≤ 8` 常量下标才能留住寄存器 |
| phase 2 的 M-fold | `acc[M]` + `wv` 提到 r 循环外 | C4（同一个 `(row,kb)` 值复用，不碰结合律） |
| P3 预取（`cpasync`） | **不做**，行内 `cp.async16` staging | 纯 staging 重排、无数值含量；M=1 的 `!prefetched` 分支就是这段。少一个旋钮 |
| `s_lut` 的构建 | **所有 block**（`if (p1)` 之外） | 只做 phase 2 的 block（`blockIdx.x >= p1b`）也要解码表 —— 这是 M=1 kernel 没有的分支，漏了就是错值 |
| LUT 之后 | 追加一个 `__syncthreads()` | 非 `p1` block 在 grid barrier 之前唯一的发布点 |

### 11.4 parity 套件（§7 步骤 4）

**`tests_dsv41_sh_exp_mrows.cu`**（新文件，不是扩 `tests_dsv41_glue.cu` —— 那份里**并没有** sh_pair
case，§10 的假设不成立）。**参照系是 `dsv41_gemm_fp8_sh_pair`（M=1）**，不是五发链：M=1 kernel 的
header 已经论证了它就是五发链的逐位等价值，所以套件把两个claim 串起来。

覆盖：

| 维度 | 取值 |
|---|---|
| M | 1, 2, 3, 5, 6, 8（+ 全 dispatch 1..8） |
| `fold_r` | 1, 2, 3, 6, m（+ 在 M=8 上扫 1..8 全范围） |
| `epi_add` | 0, 1（1 的期望值由 host 的 `pre[i] + ref[i]` 给出，同一对操作数同一顺序） |
| `act` | `nullptr`（不写）与真 buffer（逐位等于 M=1 的 act 行） |
| 形状 | 生产形状（288/5120/5120）+ 小形状 + `n2 % 32 != 0` 的 grid-stride 尾巴 |
| decline | `m∉[1,8]`、`fold_r∉[1,m]`、`k1%32`、`n1%32`、`aq_stride%16`、`out_stride<n2`、null 操作数 |
| 覆盖性 | phase-1 每个字节 + phase-2 每个元素都必须被写（NaN / 0x5A 哨兵） |

判定是 **raw f32 bits 的 memcmp**（不是容差），与 `tests_dsv41_gemm_mrows.cu` 同口径。

### 11.5 主 agent 必须跑的东西（按顺序）

```bash
# (0) 双产物重编 —— .cu 改了，build.sh 必须先于 cargo build（AGENTS.md 硬纪律）
ssh ubuntu@43.202.208.136 'cd ~/ferrite && git fetch -q origin && git reset -q --hard origin/main && \
  cd kernels/cuda && bash build.sh 103a && cd ~/ferrite && source ~/.cargo/env && cargo build --release'

# (1) parity 套件（编译 + 跑，需要一块空闲 GPU）—— 这是硬门
ssh ubuntu@43.202.208.136 'cd ~/ferrite && nvcc -gencode arch=compute_103a,code=sm_103a -O3 \
  --use_fast_math -std=c++17 -o /tmp/t_sh_exp_mrows kernels/cuda/tests_dsv41_sh_exp_mrows.cu && \
  CUDA_VISIBLE_DEVICES=<free> /tmp/t_sh_exp_mrows'

# (2) A/B（同会话背靠背，单变量）
#     ARM base : 什么都不设（per-row 25 发/层）
#     ARM M    : DSV41_SH_PAIR_M=1（2 发/层）
#     ARM M/f2 : DSV41_SH_PAIR_M=1 DSV41_SH_PAIR_M_FOLD=2
#     ARM M/f6 : DSV41_SH_PAIR_M=1 DSV41_SH_PAIR_M_FOLD=6
# 读 [dsv41] step 行（verify_ms）+ 四段文本逐字 + faults=0
```

`fold_r` 是**运行期**参（kernel + launcher 双确认），所以 (2) 的 fold 扫描**不需要重编译**。

**nsys 聚合口径**：新 kernel 是 `gemm_fp8_sh_exp_pair_kernel<M>` 模板实例，与 M=1 的
`gemm_fp8_sh_pair_kernel` **名字不同**，可直接按 kernel 名分组数 launch / 每发 µs（§7 步骤 5 要的就是
这个）。两个 kernel 共享 `g_sh_arrive`/`g_sh_sense`，**不能并发**（同流串行是前提，R9）。

### 11.6 未做（明确留在设计里）

- §7 步骤 6（折 `quant_rows` 进 phase 0、去掉 `act`）**未做**：入口保留，仍是"2 发/层"。
- 未做 k-split、tcgen05、PDL（§9 的划界不变）。
- `fold_r > 1` 的 kb 循环同样用 `#pragma unroll 4`（与 M=1 一致）。⚠️ fast-math 下多路展开 +
  分裂累加器是 R2 的已知风险面 —— **fold_r > 1 的三个 arm 必须由 §11.5(1) 的 parity 先过**才可上机。

*实施轮 · 只写代码 + `cargo check --workspace`（EXIT=0）+ 静态审读；未跑 GPU、未跑 nvcc、未改设计正文。*
