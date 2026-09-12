# L4/L5「M 进 grid」第一步设计 —— `gemm_fp8_mrows_kernel<M>` 的 `fold_r` 化

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未改动任何源码、未执行任何 GPU 命令。
> 任务：400 路径上 L4/L5 的第一步具体设计（kernel + grid 划分 + 预期 + 验证）。
> 代码基线：工作树 HEAD `0cb018c`（`git log` 现场核对）；行号以**函数名/符号**为准（本仓有行号漂移史）。
> 输入文档：`l4-occupancy-mlp-design.md` · `l4-l5-kernel-path.md` · `dspark-correctness-chain.md`
> §「SH_PAIR "M 进 grid" 模式的推广前景」/ §「L4/L5 第一步实施计划」 ·
> `sh-pair-template-m-design.md`（模板与 §11 实施状态）· `projection-family-optimization.md` ·
> `mrows-swallow-batched-implementation-design.md` · `swallow-nsys-batched-analysis-framework.md`。
> **口径纪律**：每条 ms 标来源（**实测** / **代数** / **设计**）；本机无 GPU、无 nvcc。

---

## 0. 结论先行（七条，前两条纠正任务前提）

1. **第一步的 kernel = `gemm_fp8_mrows_kernel<M>`（投影族）。** 它是四个 L4 候选里**唯一还没有设计/骨架**的一项
   （`hc_dots` 的 `(mix, rows)` 网格**已经在跑**；`collapse_norm` 的 split 核**已在树里**；tcgen05 在测试中），
   也是全表**最大的「M 进 warp」族**（batched m=6 占 **15.1% / ~4.2ms**）。见 §2。

2. **❗ 三个「M 进 grid」候选里有两个**已经**是 grid 形态了 —— 不要重复投**：
   | 候选 | 现状（读码） | 判定 |
   |---|---|---|
   | `hc_mix_dots_kernel` / `hc_dots_late_kernel` | `<<<dim3(mix, rows), ...>>>`（`dsv41_kernels.cu` 的 `hc_dots_late` launcher / `hc_mix_dots` 的 `blockIdx.x=m, blockIdx.y=r`）⇒ m=6 时 **(24,6)=144 块、97% SM** | **已是 M 进 grid**，无增量；`l4-occupancy §2.2 L4-7` 的「5 块→mix×rows」是 **m=1** 口径（24 块） |
   | `hc_mixes_kernel` / `hc_mixes_tail_kernel` / `dsv41_hc_collapse_norm_kernel` / `dsv41_rmsnorm_rows_kernel` | `<<<rows, 1024, 0, s>>>`（`:9815` / `:10051` / `:9629`）⇒ m=6 时 **rows=6 块 = 4% SM** | **M 已经是 grid 的唯一维**；「把 M 再放进 grid」不可行（它已经在）⇒ 真正的解是**加第二个维**（`hc*dim`），= `l4-occupancy §2.1 L4-9`，且 `dsv41_hc_collapse_norm_split_kernel`（`:10157`，launcher `:10231`）**已在树里待 A/B** |
   ⇒ **「M 进 grid」这四个字对 `grid == rows` 的核是不适用的**。适用条件见 §1.3。

3. **`gemm_fp8_mrows_kernel<M>` 现在确实是「M 进 warp」**（任务洞察 #2 ✓）：
   `grid.x = ceil(n / nwarps)`（**与 M 无关**），M 只进 `float acc[M]` / `float af[M]` 寄存器
   （`:5303 / :5343`）⇒ M 只**加每 warp 工作量，不加并行度**。这正是 SH_PAIR §3.3 判词的另一端。

4. **但「权重读 1/M」这个卖点在 instruction-bound 下是 0 收益**（任务洞察 #3 ✓）。
   `projection-family-optimization §0-3` 的实测带宽账：投影族 **~4.4 ms/步 vs 0.26 ms/步带宽地板 = 17×**。
   ⇒ 本设计**不为省字节牺牲并行度/占用**；`fold_r` 的唯一目标是**在飞 warp 数**（`arch-floor §5.2` 第二道墙）。

5. **因此第一步的设计量是 `fold_r`（运行期「每 block 几行激活」旋钮）+ 一处纯指令修复，两者都逐位等价、都可单变量 A/B。**
   `fold_r = M`（现状）与 `fold_r = 1`（纯 M 进 grid）是**同一 kernel 的两个端点**，
   `fold_r` 做成运行期参 ⇒ **格子扫描不需要重编译**（照抄 `sh-pair-template-m-design §7 步骤 5` 的做法）。

6. **最大的一项其实是同核的一处非对称**：**权重行用 `cp.async16` staging，激活行却是标量逐字节拷贝**
   （`:5290` vs `:5249`）。这是 **8× 指令差**、且暴露在 barrier 前的关键路径上（§4）。
   它比 `fold_r` 更便宜、更安全、对**所有形状**成立 ⇒ **建议作为本设计编号 1b，先于 1a 落地。**

7. **预期（诚实口径）**：1a+1b 合计 **−0.5 ~ −1.0 ms/步（batched 口径）**，lazy 下按 `l4-occupancy §3.2` 的
   `×k_emit(≈2.214)` 放大（**该倍数是结构推论，不是实测承诺**）。**这不改变 L4 的排序**——
   L4 的绝对量大头仍在 tcgen05 收尾（L4-3/L4-4，`l4-occupancy §5.1` 的 ~50-60%）与 hc 侧流（L4-7）。
   本设计定位 = **「把 SH_PAIR 的模板固化成第二个可复用的核」**，不是 L4 的 ms 冠军。

---

## 1. 现状精确解剖（读码结论，非估计）

### 1.1 `gemm_fp8_mrows_kernel<M>`（`dsv41_kernels.cu:5223`）

```
grid  = ceil(n / nwarps)                    ← 只 n 驱动，与 M 无关（:5433）
block = nwarps * 32 线程，__launch_bounds__(256)
smem  = nwarps*k + 256*4 + M*nb_k*4 + M*k   ← :5437
映射：warp w 拥有输出行 row = blockIdx.x*nwarps + w      （:5256）
      每 warp：staging 自己的权重行 row_s = s_w + warp*k （cp.async16, :5290）
      每 block：staging M 行激活 s_a[r*k+i]（**标量逐字节**, :5249）
consume（a32 臂, :5334）：for kb: wv = s_lut[s_w[...]]*sb; af[r]=s_lut[s_a[r*k+j]]*s_as[..]; acc[r]+=af[r]*wv
epilogue：每 r 一棵 shfl_xor 树 → out[r*out_stride + row]（:5373）
```
* `nwarps = dsv41_mrows_warps_for(n)`（`:3885`）：`n<256→1`、`n<512→2`（需 `DSV41_MROWS_SMALL_N_ADAPTIVE=1`），
  否则 `dsv41_gemv_warps_for(n)`（`n>=2048→8`，其余 `4`；`g_gemv_warps` 默认 4，`:3814`）。
* **核头（`:5145-5192`）已经论证**：`acc[r]` 是独立链、任何地方不跨 r 重结合、每 (warp,r) 一棵树
  ⇒ **「the block geometry does not enter the parity argument -- rows are independent --
  it only decides how many rows share one block's staging」**（`:5430` 原话）。**这是 `fold_r` 的合法性来源。**

### 1.2 生产形状（batched m=6, `VERIFY_ROWS=6`）——**MROWS_SMALL_N_ADAPTIVE 已在 batched GATES 里**

| 投影 | n × k | `nwarps` | blocks = ceil(n/nw) | SM 覆盖 | smem (M=6) | smem 限制的 blocks/SM |
|---|---|---:|---:|---:|---:|---:|
| `wq_a` | 1280 × 5120 | 4 | 320 | 216% | 56 064 B | **4** |
| **`wkv`** | **512 × 5120** | **4** | **128** | **86%** | **56 064 B** | **4** |
| `wq_b` | 4096 × 1280 | 8 | 512 | 346% | 19 904 B | 11 |
| `wo_b` | 5120 × 1024 | 8 | 640 | 432% | 16 128 B | 14 |
| `sh w1/w3` | 288 × 5120 | 2 | 144 | 97% | 45 824 B | 4 |
| ~~`wo_a`~~ | 1024 × 4096（本 rank 1 组） | — | — | — | — | 走 `wo_a_grouped_gemv_kernel`，**不在本核** |

**读表要点（三条）**：
1. **`DSV41_MROWS_SMALL_N_ADAPTIVE=1` 已经把 sh w1/w3 的 72 → 144 块修掉了**（`n=288 < kMrowsSmallN2=512` ⇒ `nwarps=2`）。
   ⇒ **`l4-occupancy §2.1 L4-1` 的靶子只剩 `wkv`（128 块，86%）**，且它在 L3（SH_PAIR_M）之后才露出来。
2. **真正的低占用不是「块数不够」而是「每 SM 的 warp 数不够」**：`wq_a`/`wkv` 的
   **smem 56064 B ⇒ 只 4 blocks/SM × 4 warps = 16 warps/SM（25%）**——离 64 warps/SM 的机器上限差 4×。
   `smem` 的主要成分是 **`M*k` 的激活行（M=6,k=5120 ⇒ 30 720 B = 55%）**。
   ⇒ **这就是 `fold_r` 的靶子**（把 M 从「每 block 的 smem 行数」改成「grid 的一维」⇒ smem 减半、warp/SM 翻倍）。
3. **`wq_b`/`wo_b`（n≥4096）块数已 346%/432%**，`fold_r=1` 只会把权重 staging ×6 而不加并行度
   ⇒ **它们的默认应是 `fold_r = M`（不动）**。这是本设计必须写死的**形状选择规则**（§3.4）。

### 1.3 「M 进 grid」的**适用判据**（从 SH_PAIR 反推，可复用）

SH_PAIR §3.3 的模型给的是「并行度 / 指令数」的取舍表；抽象成三条：

```
① 把 M 折进 warp 后，块数（= 除 M 以外的并行度）是否 < 148（SM 数）？
   否 ⇒ M 进 grid 无并行度收益，只加权重重读 ⇒ 保持 fold_r = M。
② 折叠是否让每 block 的 smem/regs 跨过占用台阶？（本核：smem=M*k 主导）
   是 ⇒ M 进 grid 买到的是「在飞 warp 数」。
③ 权重/操作数是否是 L2 常驻（重读是 L2 命中而非 HBM）？
   否 ⇒ M 进 grid 会真的付 HBM 字节（本核：wq_a 6.55MB / wkv 2.62MB —— **是** L2 常驻）。
```
**本核三条全中**（`wq_a`/`wkv`：块数 320/128、smem 56 KB 跨台阶、权重 L2 常驻）
⇒ **SH_PAIR 的机制在本核成立，且靶子比 SH_PAIR 更大**（SH_PAIR phase-1 是 9 块、本核是 128~320 块但仍受 smem 压制 warp/SM）。

---

## 2. 为什么第一步是 mrows（候选逐项判决）

| 候选 | nsys 占比（batched m=6） | 状态 | 判决 |
|---|---:|---|---|
| gemm 投影族（**本设计**） | **15.1% / ~4.2 ms** | 只有「M 进 warp」；无 M-grid 骨架 | ✅ **第一步**（见 §3/§4） |
| hc_dots（L4-7） | 6.7% / ~1.9 ms | **已是 `(mix, rows)` = 144 块** | ❌ 无 M-grid 增量；其增量是 smem/KCHUNK（L4-8）与侧流，属别的工单 |
| collapse_norm（L4-9） | 含在 hc 链 | `dsv41_hc_collapse_norm_split_kernel` **已在树**（`:10157`） | ❌ 设计已备、待 A/B（前置 = A1 接线）；重设计无价值 |
| tcgen05（L4-3/L4-4） | 17.4% + 8.7% | TMA 对齐修复在跑 | ❌ 不是 grid 问题（TMEM 钉 ≤2 CTA/SM），且归 L2 的收尾 |
| AR（36%） | **6.6 ms** | A1a 失败，替代设计在跑 | ❌ 协议地板，不是 grid |

**一句话**：四个候选里，**只有 mrows 是「M 进 warp 且还没有 M-grid 版本」的**。
其余三个要么已经是 grid、要么已在树里待测、要么根本不是 grid 问题。
⇒ **第一步选 mrows 是「唯一性」结论，不是 ROI 排序结论**——这一点必须写进账（见 §10-1）。

---

## 3. 设计 1a：`fold_r`（M 进 grid 的运行时旋钮）

### 3.1 总体结构（以 SH_PAIR 为模板逐条对照）

```
grid (G, 1, 1)，block = nwarps*32，动态 SMEM

  nt  = ceil(n / nwarps)            // 输出行 tile 数（"i-tile"）—— 只 n 驱动
  ng  = ceil(M / fold_r)            // 激活行组数（"row group"）—— 就是 M 进 grid 的那一维
  G   = nt * ng
  it  = blockIdx.x % nt             // 本 block 的输出行 tile
  rg  = blockIdx.x / nt             // 本 block 的激活行组
```
| 项 | SH_PAIR `gemm_fp8_sh_exp_pair_kernel<M>` | 本设计 `gemm_fp8_mrows_kernel<M>`（+fold_r） |
|---|---|---|
| 并行度上限（逐位） | `n1t × nsr`，`nsr=ceil(M/fold_r)` | `nt × ng`，`ng=ceil(M/fold_r)` |
| block 拥有的东西 | (i-tile, 激活行组) | (输出行 tile, 激活行组) |
| warp 拥有的东西 | 1 个 inter 行 × `fold_r` 个激活行 | 1 个输出行 × `fold_r` 个激活行 |
| 跨 block 重复读 | 权重（L2 常驻，接受） | 权重（L2 常驻，接受） |
| 折进 warp 的反面教材 | `fold_r=6` ⇒ 9 块 | `fold_r=M`（现状）⇒ `nt` 块（与 M 无关） |
| **不能做的** | k-split（破 f32 结合序，§3.4） | **同样不能 k-split**（`l4-occupancy §2.1 L4-2` 已判「非逐位」） |

### 3.2 kernel 改动（**单核修订，不是新核**）

```cuda
// gemm_fp8_mrows_kernel<M>：新增一个尾部运行期参 fold_r
__global__ void __launch_bounds__(256)
gemm_fp8_mrows_kernel(const uint8_t* __restrict__ a, const float* __restrict__ a_scale,
                      const uint8_t* __restrict__ w, const uint8_t* __restrict__ w_scale,
                      const float* __restrict__ bias, float* __restrict__ out,
                      int n, int k, int out_stride, int a32, int fold_r) {
    ...
    const int nt = (n + nwarps - 1) / nwarps;
    const int ng = (M + fold_r - 1) / fold_r;
    const int it = blockIdx.x % nt;
    const int r0 = (blockIdx.x / nt) * fold_r;      // 本 block 覆盖的激活行起点
    const int rn = min(fold_r, M - r0);             // 尾组

    // staging：**只 stage 本 block 的 rn 行**（smem 随之从 M*k 降到 rn*k）
    #pragma unroll
    for (int q = 0; q < M; ++q) {                   // M 是编译期，循环上界仍用 M
        if (q >= rn) break;                         // 运行期谓词（保住寄存器）
        const uint8_t* ar = a + (size_t)(r0 + q) * k;
        const float*   as = a_scale + (size_t)(r0 + q) * nb_k;
        for (int i = threadIdx.x; i < nb_k; i += blockDim.x) s_as[q*nb_k + i] = as[i];
        for (int i = threadIdx.x; i < k;     i += blockDim.x) s_a[q*k + i]     = ar[i];
    }
    ...
    const int row = it * nwarps + warp;             // ← 只改这一行的上一级（原：blockIdx.x*nwarps + warp）
    ...
    float acc[M];                                    // 编译期数组（照 §11.3 的教训：不能是 acc[fold_r]）
    #pragma unroll 32
    for (int kb = 0; kb < nb_k; ++kb) {
        const float wv = s_lut[s_w[(size_t)warp*k + j]] * sb;
        #pragma unroll
        for (int q = 0; q < M; ++q) {
            if (q >= rn) break;
            const float av = s_lut[s_a[(size_t)q*k + j]] * s_as[q*nb_k + (j>>5)];
            acc[q] += av * wv;
        }
    }
    #pragma unroll
    for (int q = 0; q < M; ++q) {
        if (q >= rn) break;
        float a_q = acc[q];
        for (int off = 16; off > 0; off >>= 1) a_q += __shfl_xor_sync(0xFFFFFFFFu, a_q, off);
        if (lane == 0) out[(size_t)(r0+q)*out_stride + row] = a_q + (bias ? bias[row] : 0.f);
    }
}
```
* **`acc[M]` 保持编译期大小、`if (q >= rn) break` 做运行期谓词** —— 正是
  `sh-pair-template-m-design §11.3` 已落地的选择（「`fold_r` 是运行期参，`g[fold_r]` 会掉 local memory；
  `M ≤ 8` 常量下标才能留住寄存器」）。**照抄，不重新发明。**
* **launcher**：`dsv41_gemm_fp8_mrows` 的 C ABI **不变**；`fold_r` 由 launcher 内部算：
  ```c
  int fold_r = g_mrows_fold_r;                        // 来自 DSV41_MROWS_FOLD_R，默认 0 = 自动
  if (fold_r <= 0) fold_r = mrows_fold_r_for(n);      // 形状规则（§3.4）
  if (fold_r > m)  fold_r = m;
  if (fold_r < 1)  fold_r = 1;   // 绝不返回 1 当 decline 码
  ```
* **SMEM 属性零改动（重要）**：`FERRITE_SET_MROWS_SMEM(k)` 已对每个 M 设了 **`fold_r = M` 的上界**
  （`:5437` 的 `smem` 表达式用的是 `m`）。`fold_r <= m` ⇒ 实际 smem ≤ 已设上界 ⇒
  **不需要新的 `cudaFuncSetAttribute` 分支**（这是本设计比 SH_PAIR 便宜的地方——SH_PAIR 需要 per-M 宏，本核已有）。

### 3.3 占用账（生产形状，代数）

| 形状 | 现状（`fold_r=M`） | `fold_r=1` | 变化 |
|---|---|---|---|
| `wq_a`/`wkv` smem | 56 064 B | **27 264 B** | ÷2.06 |
| blocks/SM（smem 限） | 4 | **8** | ×2 |
| blocks/SM（线程限，128 thr） | 16 | 16 | — |
| **warps/SM** | **16（25%）** | **32（50%）** | **×2** |
| grid（`wkv`，nt=128） | 128 | **768** | ×6 |
| 权重 staging（`wkv`） | 1× | 6×（2.62 MB ⇒ 15.7 MB，**L2 常驻**） | 代价 |
| 激活 staging 指令 | 240/线程 | 40/线程（**总量不变**：`nt·M·k` 与 `fold_r` 无关，见 §4.1） | — |

> **诚实边界**：`fold_r=1` 的并行度收益是**结构性**的（smem ÷2 ⇒ warps/SM ×2），
> 但**每 block 的权重重读 ×6** 是真实成本。**净符号必须由 A/B 定**（§7.3 的扫格），
> 本文件只保证：**两端都逐位等价、且不需要重编译即可扫**。

### 3.4 形状选择规则（默认值，可由 `DSV41_MROWS_FOLD_R` 覆盖）

```c
static inline int mrows_fold_r_for(int n) {
    // 判据 = §1.3：块数 < 148 或 smem 压制 warp/SM ⇒ M 进 grid
    if (n <= 1024) return 1;      // wkv(512) → 1；wq_a(1280) 视 A/B 结果
    return -1;                    // -1 = 用 M（现状），wq_b/wo_b 不动
}
```
* **`n` 是运行期就能读到的**（launcher 的入参）⇒ 规则不引入新 ABI。
* **`wq_a`（n=1280）落在灰区**：块数 320 已够，但 smem 56 KB 仍压制 warp/SM。
  ⇒ **`fold_r` 是运行期参，扫格即可定**，不要在设计期替实测做决定。

---

## 4. 设计 1b：激活 staging → `cp.async16`（同核的一处非对称，**建议先做**）

### 4.1 现状（读码，非估计）

```cuda
// :5244  激活行 —— 标量逐字节（1 B/lane/次 ⇒ 32 B/warp-issue）
for (int i = threadIdx.x; i < k; i += blockDim.x) s_a[(size_t)r*k + i] = ar[i];
// :5290  权重行 —— cp.async16（16 B/lane/次 ⇒ 512 B/warp-issue）
for (int i = lane; i < n16; i += 32) dsv41_cp_async16(row_s + (i<<4), wr + (i<<4));
```
**同一个 kernel 的两条 staging 走两条规则**：权重 16 B/issue，激活 1 B/issue。
`dsv41_kernels.cu` 自己在权重侧把这条账写死了（`:5260-5265`：「32 B moved per warp-issue where the m=1
prologue moves 512 B … **SIXTEEN times the instructions**」）——**但只修了权重侧，激活侧从未修**。
同时 `DSV41_GEMV_ACT_CPASYNC`（P4，`:3961`）**只接了 m=1 的 `gemv`，没接 `mrows`**。

### 4.2 指令账（生产形状，代数）

| 形状 | 激活 staging（标量，现状） | 激活 staging（cp.async16） |
|---|---|---|
| `wkv`（128 thr, k=5120, M=6） | 6×5120/128 = **240 次/线程 × 2 指令 = 480 warp-instr** | 6×5120/16/128 = **15 warp-instr** |
| 相对 consume（`nb_k=160 × ~28`） | ~10% | ~0.3% |
| **关键路径** | 在 `__syncthreads()`（`:5298`）**之前**，完全暴露 | 与权重 cp.async 一起在 barrier 下重叠 |

* **对齐**：launcher 已拒绝 `k & 31` ⇒ `a + r*k` 16 B 对齐；smem 侧
  `s_a = s_as + fold_r*nb_k`（`nb_k*4 = 640`，`640 % 16 == 0`）且 `s_lut` 256 float、`nwarps*k` 均 16 B 对齐
  ⇒ **`cp.async16` 合法**（与权重行同一条论证，`:5279-5285`）。
* **数值**：纯拷贝，K-walk / `acc[r]` 链 / `shfl_xor` 树**一个字节不动**
  ⇒ **逐位等价**（与权重侧 staging 修复同一论证：「staging does not touch a value」）。

### 4.3 与 1a 的关系

**两者独立、可分别 A/B**：
* 1b 只改 staging 的**指令规则**，不改 grid/映射 ⇒ 对**所有形状**成立，且与 `fold_r` 正交。
* 1a 只改 grid/映射，**不减激活 staging 的总量**——`nt·M·k` 与 `fold_r` 无关
  （`fold_r=1` 时块数 ×M、每块 staging ÷M）⇒ **M 进 grid 不修 staging 指令，1b 才修**。
⇒ **1b 先做**（更大、更安全、无 A/B 依赖），1a 紧随（要扫格）。

---

## 5. 与现有 mrows 的关系（任务问题 3）

**互补，不是替代；而且严格说不是两个核，是同一个核的两个端点。**

| | `fold_r = M`（现状 mrows） | `fold_r = 1`（本设计的新端点） |
|---|---|---|
| 物理量 | 每 warp 的工作量（`acc[M]`） | 并行度（`nt × M` 块） |
| 适用 | 块数已 ≥148 且 smem 不压制（`wq_b`/`wo_b`） | 块数 <148 或 smem 压制（`wkv`/`wq_a`） |
| 权重读 | 1× | ×M（**L2 常驻**，见 §1.3-③） |
| 数值 | **逐位相同**（同一个 `(row, r)` 链） | 同左 |
| 谁替代谁 | **谁也不替代谁** | 同左 |

三条必须写进账的边界：
1. **`fold_r` 不替代 mrows 的「M 折叠」**——它为 mrows **补上被折叠掉的那个维度**。
   任务洞察 #2 的「M 只加每 warp 工作量」在 `fold_r=M` 时成立，在 `fold_r=1` 时不成立。
2. **与 L4-1（`MROWS_SMALL_N_ADAPTIVE`）正交**：L4-1 动的是 `nwarps`（**行/块**，n 驱动），
   本设计动的是 `fold_r`（**激活行/块**，M 驱动）。两者可以同开（`nwarps=2` + `fold_r=1`）。
3. **与 L3（SH_PAIR_M）的边界**：sh w1/w3（n=288）的 mrows 发数在 L3 落地后**消失**
   （改走 `gemm_fp8_sh_exp_pair_kernel<6>`）⇒ **本设计的靶子只剩 `wkv` / `wq_a` / `wq_b` / `wo_b`**，
   不要按 5 个形状编票面。

---

## 6. 数值域：逐位等价论证（C1–C7，照搬 SH_PAIR 的验收口径）

| # | 环节 | 论证 |
|---|---|---|
| **C1** | block↔行的映射 | mrows 核头（`:5145-5192`）已证 **rows are independent**、`acc[r]` 独立链、无跨 r 重结合；`:5430` 原话：「the block geometry does not enter the parity argument」。`it/rg` 的拆分只改「几行共享一个 block 的 staging」。 |
| **C2** | 激活物化 | `af = s_lut[s_a[q*k+j]] * s_as[q*nb_k + (j>>5)]` —— 与 `fold_r=M` 时的第 q 个操作数**同一表达式、同一字节来源**（`s_a` 的 slot 变短，读法不变）。 |
| **C3** | K 走序 | `kb` 升序、`j = kb*32 + lane`，每链**单个串行 `+=`**；`#pragma unroll 32` 只重叠 load，不重结合。 |
| **C4** | `wv` 复用 | `wv` 提到 q 循环外是**同一个 `(row,kb)` 值的复用**（`fold_r=M` 已经这么做，`:5339` 的 C4 注释）——不触碰结合律。 |
| **C5** | 归约树 | `for (off=16; off>0; off>>=1) acc += __shfl_xor_sync(...)` 逐字，每 (warp, q) 一次。 |
| **C6** | bias / store | `out[(r0+q)*out_stride + row] = a_q + (bias ? bias[row] : 0.f)` —— 与 `:5377` 同式同序。 |
| **C7** | staging | `cp.async16` 把**同样的字节**放进**同样的 slot**；consume 只读该 slot ⇒ 与标量拷贝逐位相同（`:5270-5277` 的原话）。 |

**唯一的真风险**：`--use_fast_math`（`build.sh` 默认）。对冲 = **照抄旧核表达式与 `#pragma unroll` 力度**，
任何不得不动的结合点用 `__fmaf_rn`。**不引入 k-split、不引入 tolerance 口径。**

---

## 7. 验证方法（任务问题 4）

### 7.1 本地硬门禁（0 GPU）
```bash
cargo check --workspace                       # 项目硬门禁
cd kernels/cuda && bash build.sh 103a         # .cu 变了 ⇒ 必须先于 cargo build（AGENTS.md 纪律）
```

### 7.2 parity（1 GPU，**raw-u32 memcmp，不是容差**）
扩 **`tests_dsv41_gemm_mrows.cu`**（已有套件，不是新文件）：
| 维度 | 取值 |
|---|---|
| `M` | 1,2,3,5,6,8（+ dispatch 1..8 全覆盖） |
| **`fold_r`** | 1, 2, 3, M（+ 在 M=8 上扫 1..8） |
| `a32` | 0, 1（本核两个臂都在树里，`:5306/:5351`） |
| 形状 | 生产（`k=5120`,`n∈{288,512,1280}`）+ 小形状 + **`n % nwarps != 0` 的尾块** + **`M % fold_r != 0` 的尾组** |
| 判据 | 对每个 `(row, r)`：`fold_r=k` 的输出 **逐位 == `fold_r=M` 的输出**（raw f32 bits memcmp） |
| decline 表 | `fold_r∉[1,m]`、`k&31`、`out_stride<n`、null 指针、`mode<3` |

### 7.3 A/B（1 GPU，**同会话背靠背，单变量**）
```bash
# ARM 0 : 现状（fold_r=M, 标量 staging）—— 什么都不设
# ARM 1b: 只翻 staging            DSV41_MROWS_ACT_CPASYNC=1
# ARM 1a: 只翻 fold_r             DSV41_MROWS_FOLD_R=1
# ARM 1a*: 形状规则（wkv/wq_a）    DSV41_MROWS_FOLD_R=auto
# ARM 1a+1b
```
* 读 `[dsv41] step` 行 + **四段文本逐字** + `faults=0`；红线 = **计数数字顺序 + 出师表零拉丁**。
* ⚠️ **必须与 `DSV41_SH_PAIR_M=1` 同臂测**（否则 `sh w1/w3` 的 mrows 靶子还在，混淆归因）。
* ⚠️ **每个 gate 翻转单独 commit + 读回确认**（`dspark-correctness-chain` 的 R6 陷阱：gate 设了但没生效）。

### 7.4 nsys 判据（**本设计特有的坑**）
**kernel 名不变**（是同一个 `gemm_fp8_mrows_kernel<6>`，不是新符号）
⇒ `swallow-nsys-batched-analysis-framework §4` 那套**「按 kernel 名数 launch」的判据在这里失效**。
必须改用 **`cuda_gpu_trace` 的 `GridX` 列**：
```python
# 取 decode 窗口内 gemm_fp8_mrows_kernel<6> 的 GridX 分布：
#   fold_r=M → GridX ∈ {144,128,320,512,640}     （只 n 驱动）
#   fold_r=1 → GridX ∈ {864,768,1920,...}        （nt × 6）
# 判据：GridX 出现 nt*6 的倍数 · 且 DSV41_MROWS_FOLD_R 的读回值一致
```
* 同 `§3.1` 的 CSV 口径：`python csv` + `r[-1]`，禁止 awk 切列。
* **不读绝对 ms**（AR-safe 栈下 `VERIFY_GRAPH` 会 decline，绝对 ms ≠ 生产）。

### 7.5 止损线
| 判据 | 动作 |
|---|---|
| parity 有任何一位不同 | **停**，回退，不改设计正文 |
| `fold_r=auto` 在 `wq_b`/`wo_b` 上被选中 | 停，检查 §3.4 规则（大 n 不该动） |
| ARM 1a 在 `wkv` 上中性或负 | 只保留 1b（它在所有形状上正），1a 记「中性」封存 |
| ARM 1b 中性 | 说明该核不是 staging 受限 ⇒ **重估 instruction-bound 归因**（上报） |

---

## 8. 风险表

| # | 风险 | 触发 | 应对 |
|---|---|---|---|
| **R1** | `fold_r` 掉 local memory | 写成 `float acc[fold_r]`（运行期大小） | **`acc[M]` 编译期 + `if (q>=rn) break` 谓词**（SH_PAIR `§11.3` 已证） |
| **R2** | 逐位破功 | `--use_fast_math` 下重结合 | 照抄表达式/unroll 力度；K-walk 与树一字不动；parity 是硬门 |
| **R3** | `cp.async16` 对齐 | smem 槽不在 16 B 边界 | 已在 §4.2 逐项算过（`nb_k*4=640`）；launcher 的 `k&31` 拒绝是前提 |
| **R4** | SMEM 属性失配 | 误以为要为新 `fold_r` 重设 | **不需要**：属性已按 `m` 设上界，`fold_r<=m` ⇒ 实际 ≤ 上界（§3.2） |
| **R5** | gate 没生效（项目 #1 陷阱） | `DSV41_MROWS_FOLD_R` 未读回 | 每次翻转读回 + nsys `GridX` 交叉验证（§7.4） |
| **R6** | 尾组越界 | `M % fold_r != 0`（如 M=6,fold_r=4） | `rn = min(fold_r, M-r0)` + parity 覆盖该组合 |
| **R7** | 大 n 误开 fold_r=1（权重 ×6） | `auto` 规则写错边界 | §3.4 规则 + `n>1024 ⇒ -1`；A/B 时逐个形状看 `GridX` |
| **R8** | 与 SH_PAIR 的 `g_sh_arrive/g_sh_sense` 冲突 | — | **本核无 grid barrier**，不共享任何模块级状态 ⇒ 无此风险（与 SH_PAIR 的关键差异） |

---

## 9. 明确**不做**的事（划清范围）

1. **不做 k-split / 跨 warp 部分和**：破 f32 结合序（`l4-occupancy §2.1 L4-2` 已判「非逐位」）。
2. **不动 `wq_b`/`wo_b` 的默认**（n≥4096，块数已 346%/432%）——`fold_r=M` 保持。
3. **不碰 `wo_a_grouped_gemv_kernel`**：它是另一个核（`:5536`），其 `cp.async16` 修复在
   `projection-family-optimization §3-R1` 单独立项。
4. **不碰 hc 链 / collapse_norm / tcgen05 / AR**——别的工单（§2）。
5. **不动 `quant_rows` / `add_inplace_raw` 的调用点**。
6. **不追求「权重读 1/M」**：instruction-bound（任务洞察 #3 ✓），字节不是天花板。

---

## 10. 预期与诚实校准（必须写在账上的四条）

1. **本设计是「唯一性」结论，不是「最高 ROI」结论。** 按 ROI 单排，
   `l4-occupancy §4.1` 的第一名是 **L4-7（1.5~3.0 ms/pd）**，本设计约 **0.4~0.8 ms/pd**。
   选它的理由是 §2：**四个候选里唯一还没有 M-grid 骨架的**，且它把 SH_PAIR 的模板**固化成第二个可复用核**。
   **不得把它写成 L4 的 ms 冠军。**
2. **预期 ms（batched 口径，设计值）**：
   | 项 | 来源 | 低 | 高 | 把握 |
   |---|---|---:|---:|---|
   | 1b 激活 staging cp.async16 | 指令账（§4.2，~10% of 族） | −0.3 | −0.5 | 中（姊妹核同款修复已在树） |
   | 1a `fold_r`（`wkv`/`wq_a`） | 占用账（§3.3，warps/SM ×2） | −0.2 | −0.5 | 中低（需扫格） |
   | **合计** | | **−0.5** | **−1.0** | |
   ⇒ 投影族 15.1% / 4.2 ms 中回收 **12~24%**。lazy 下按 `×k_emit(2.214)` **放大**是**结构推论**，
   **不是实测**（`l4-occupancy §3.2` 明说「不要用折算值入预算」）。
3. **1b 是本文件里把握最高的一条**：它是「隔壁已经做过的修复这边漏了」
   （`:5260-5265` 的权重侧账 + `DSV41_GEMV_ACT_CPASYNC` 只接了 gemv），不是外推。
4. **所有 ms 都是设计口径，仓内无实测背书**；反向证据仍在：
   `{SH_EXP_MROWS, VERIFY_GRAPH, VERIFY_ROPE_MROWS, DRAFT_P3A}` 全开只 **−1.21ms**（预期 −24）。
   **instruction-bound + 低占用**的墙，只能靠 §7.3 的同会话 A/B 推。

---

## 11. 交付物清单（本设计）

| 文件 | 改动 |
|---|---|
| `kernels/cuda/dsv41_kernels.cu` | `gemm_fp8_mrows_kernel<M>` 加 `fold_r` 尾部参 + `nt/ng/it/r0/rn` 映射 + 激活 staging 的 `cp.async16` 分支（`DSV41_MROWS_ACT_CPASYNC`）；`dsv41_gemm_fp8_mrows` 里的 `fold_r` 计算 + 形状规则。**C ABI 不变** |
| `kernels/cuda/tests_dsv41_gemm_mrows.cu` | 加 `fold_r` 轴（1/2/3/M）+ 尾块/尾组 + `a32∈{0,1}` 的 raw-u32 parity |
| `crates/ferrite-models/src/dsv41/chain_dev.rs` | 若要在 Rust 侧暴露 A/B 臂：`mrows_fold_r()` gate（**默认走 launcher 的自动规则**，不改调用点） |
| `docs/agent/l4-mgrid-first-step-design.md` | 本文档 |

**兼容性**：无 breaking change。新 gate 默认 OFF（`fold_r` 由 launcher 的自动规则给，默认对大 n 返回 `M`）。
`DSV41_MROWS_FOLD_R=0/unset` ⇒ `fold_r=auto`；`=M` ⇒ 与今天逐字节相同。

---

*工部 · 只读分析 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*所有 kernel 名以 `__global__` / `extern "C"` 符号为准；所有 ms 标来源（实测 / 代数 / 设计）。*
*代码基线 HEAD `0cb018c`；读码时以函数名为准。*
