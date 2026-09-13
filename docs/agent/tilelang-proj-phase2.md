# TileLang 投影 GEMM 原型 · 第二阶段（五形状 + wo_a 分组）

> 工部 · 2026-09-13 · **GPU 实测**（远端 B300 sm_103a，`ssh ubuntu@43.202.208.136`；micro bench，未跑 e2e serve）。
> 交付：五形状 TileLang route A 原型 + wo_a 分组形态决策 + M6/M1 完整 benchmark + 数值形态。
> 前置：`tilelang-proj-proto.md`（route A 原型 + §3.1 的 epilogue 直读踩坑清单）、
> `projection-family-optimization.md`（"投影族是 launch/指令受限，不是带宽受限"）、
> `tensorcore-proj-design.md`（§3.3 的 K-split 规则）。
> 代码基线：`kernels/cuda/dsv41_kernels.cu` HEAD（`gemm_fp8_mrows_kernel` / `wo_a_grouped_gemv_kernel` / `wo_a_grouped_fp8`）。

---

## §0 一句话判决

**第一阶段那条"M 缩进 mma 的 tile 维度 ⇒ M=6 的成本 ≈ M=1"的结论，在完整投影族上成立**：
五形状（含 **wo_a 分组**）实测 **M6/M1 = 0.99–1.01**（老 kernel 3.03–5.78×；wo_a 分组 4.61×/5.39×）。
判据 `<1.5×` **全通过**。

| # | 项 | 判决 |
|---|---|---|
| ① | 五形状 TileLang route A 原型（GPU 实测） | ✅ `proj_fp8_tilelang_phase2.py`（稠密 4 + wo_a 分组 2 形态） |
| ② | M6/M1 比值 | ✅ **0.99–1.01**（判据 <1.5×） |
| ③ | wo_a 分组形态 | ✅ **group 维进 grid.z**（`blockIdx.y` 语义），激活按 `a_stride` 跨组分段 |
| ④ | 五形状完整 benchmark | ✅ 见 §4（双侧 + 归一化净时间） |
| ⑤ | 数值形态 | ✅ vs f64 真值 **max_rel ≤ 3.9e-6**；vs **真实 ferrite kernel** max_rel ≤ **7.6e-7**（§5） |
| ⑥ | epilogue 直读的通用性 | ✅ 五形状 × `ns∈{1..4}` × 3 rep **逐位相同**（§3） |

> ⚠️ **任务书形状表与代码事实冲突（已按代码实现，请尚书省知悉）**：任务书写 wo_a 是
> `[nlg=1? × hd=512, dim=5120]`。ferrite 的真实 ABI（`dsv41_kernels.cu:8165` 的 kernel 头 +
> `chain_dev.rs:13270` 的调用点）是 **per-group `[n, k] = [o_lora_rank=1024, hpg*head_dim=4096]`**，
> 分组数 `G = nlg = o_groups/world`（verify@TP8 → **1**，draft@TP1 → **8**），
> 激活行距 `a_stride = nlh*head_dim`（≠ k 当 world<8）。本阶段按**代码事实**实现与 benchmark。
> 另：`proj_fp8_tilelang.py`（一阶段）本就已覆盖 wkv/wq_a/wq_b/wo_b 四形状；本阶段的**新增量**
> 是 wo_a 分组 + 五形状双侧对拍 + 确定性验证。

---

## §1 形状表（以代码为准）

| 调用 | n | k | M | 来源 |
|---|---|---|---|---|
| wkv | 512 | 5120 | 1/6 | `chain_dev.rs` verify/draft |
| wq_a | 1280 | 5120 | 同 | 同上（q_lora_rank × dim） |
| wq_b | 4096 | 1280 | 同 | 同上（nh·hd/world × q_lora_rank） |
| wo_b | 5120 | 1024 | 同 | 同上（dim × o_lora_rank） |
| **wo_a 分组** | **1024** | **4096** | 同 | `[o_lora_rank, hpg·head_dim]`，**G = o_groups/world** |

`dim=5120 · nh=64 · hd=512 · ql=1280 · olg=1024 · o_groups=8`（`config.rs` production）。
wo_a：`hpg = nh/o_groups = 8`，`k = hpg·hd = 4096`，`a_stride = nlh·hd = G·k`。

**wo_a 组布局**（block-diagonal，`dsv41_kernels.cu:8116-8173`）：

```
a        [rows, a_stride] fp8     —— 第 g 组的 k 字节段在 +g*k
a_scale  [rows, a_stride/32] f32  —— 第 g 组段在 +g*k/32
w        [G, n, k] fp8            —— 第 g 组在 +g*n*k
w_scale  [G*n/32, k/32] ue8m0     —— 第 g 组在 +(g*n/32)*nb_k
out      [rows, G*n] f32          —— 第 g 组列在 +g*n
```

> ⚠️ 组段的激活偏移是 **`g*k` 字节**而不是 `g*(a_stride)`——**rank 本地**的 `nlh` 个 head 顺序
> 排列着 `nlg` 个组，`nlg*k == nlh*hd == a_stride`，所以组段在行内是**连续堆叠**的。
> world=8 时 `nlg=1` ⇒ `a_stride == k`（strided 问题退化）；world=1 时 `nlg=8, a_stride=32768`
> 才是真正的"跨组不同 stride"形态。**两个都 benchmark 了**。

---

## §2 扩展原型 — deliverable ①

`kernels/tilelang/proj_fp8_tilelang_phase2.py`（一阶段的姊妹版；一阶段文件保持不动）。

| 函数 | 作用 |
|---|---|
| `proj_fp8_partial(N,K,bN,ks,threads,ns)` | 稠密 K-split 分片，grid `(N/bN, ks)`（**与一阶段逐行相同**） |
| `proj_fp8_reduce(ks,N)` | 确定性归约（kp 升序求和） |
| **`wo_a_grouped_partial(G,N,K,ASTRIDE,bN,ks,threads,ns)`** | **分组**分片，grid `(N/bN, G, ks)` |
| **`wo_a_grouped_reduce(ks,G,N)`** | 分组归约，写 `OUT[:, g*N + ...]` |
| `quant_ref(x)` | **复刻 ferrite `quant_kernel<0>`**（per-32 amax + `fast_round_scale` pow2 + e4m3），shim 侧量化的参照 |
| `selftest / determinism / benchmark / compare_old` | 数值、确定性、bench、与老 kernel 对拍 |

**route A 的核心（沿用一阶段，未改）**：

```python
T.gemm(A_sh, W_sh, C_p, transpose_B=True, clear_accum=True)   # raw fp8 乘积和
for i, j in T.Parallel(MPAD, bN):
    C_l[i, j] += C_p[i, j] * ASC[i, gko] * T.cast(WSC[..., gko], "float32")
```

**分组版只多了两处**（其余逐行相同）：

```python
with T.Kernel(T.ceildiv(N, bN), G, ks, threads=threads) as (bx, g, kp):
    T.copy(A[0, g * K + gko * 32], A_sh)          # ① 组段偏移（block-diagonal）
    T.copy(W[g, bx * bN, gko * 32], W_sh)         #    组维索引权重
    ... C_l[i,j] += C_p[i,j] * ASC[i, g*KELEM + gko] * cast(WSC[g, ...], f32)
    T.copy(C_l, P[kp, g, 0, bx * bN])
```

### §2.1 五个 TileLang 形状盲区（都验证过）

| # | 盲区 | 结论 |
|---|---|---|
| 1 | **`a_stride ≠ k`**（跨组步长与组内 K 不同） | ✅ 把 `a` 声明成 2D `[MPAD, ASTRIDE]`、组段取 `A[0, g*K + ...]` 即可——**不需要 materialise 成 3D**，直接复现 ferrite ABI |
| 2 | **group 维放哪** | ✅ `grid.z`（`T.Kernel(gx, gy, gz)` 的第三维）。每 block 单组、零跨组共享，与老 kernel 的 `blockIdx.y = g` 同形 |
| 3 | **组 × K-split 交叉** | ✅ `(N/bN, G, ks)` 三维 grid 直接表达；`P` 升到 `[ks, G, MPAD, N]` 后归约（组间无耦合） |
| 4 | **两段式 reduce** | 与稠密同构（组维进 reduce 的 grid.y）——集成时应换 ferrite 的单发 ctr 归约（proto.md §8#2） |
| 5 | **M 必须 pad 到 16** | 与稠密同（mma `m16n8k32`）——**这是唯一的集成侧 staging 改动**（§6） |

---

## §3 ③ epilogue 直读的通用性（"5a-slab" 形态对照）

一阶段 §3.1 的**必读踩坑**：在 pipelined loop 里自建 scale 共享缓冲 ⇒ `ns>1` 时跨 stage 覆盖 ⇒
**静默错值**。修法是 scale **直读全局**（L1/L2 命中）。

本阶段把它做成**可执行的通用性判据**：

| shape | ns=1 | ns=2 | ns=3 | ns=4 |
|---|---|---|---|---|
| wkv | ok | ok | ok | ok |
| wq_a | ok | ok | ok | ok |
| wq_b | ok | ok | ok | ok |
| wo_b | ok | ok | ok | ok |
| wo_a_nlg1 | ok | ok | ok | ok |
| wo_a_nlg8 | ok | ok | ok | ok |

审判规则：**同一输入**下，`ns∈{1,2,3,4}` × **3 次重复**共 12 次运行，结果两两**逐位相同**
（`torch.equal`）。⇒ **`=> ALL BIT-IDENTICAL ✅`**。

**读法**：epilogue 直读不是"wkv 上碰巧成立的技巧"——它在**五形状（含分组）× 四个 stage 数**
上都给出逐位确定的输出。若哪天有人把 scale 挪回自建 smem，这张表会立刻变红，**这就是它的价值**。
（自建缓冲在 `ns=1` 时是对的，所以旧写法在 `wkv` 的单 stage 快速验证里"看起来没问题"——
一阶段踩的就是这个坑。）

---

## §4 ④ 五形状完整 benchmark

### 4.1 方法

- **老 kernel**：`kernels/cuda/bench_proj_old.cu`（独立 TU，`#include "dsv41_kernels.cu"`，
  直接调 launcher 而非重写 kernel）；CUDA event、稠密每次 2000 iters（m=6 档 500 iters）。
  `nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a`。
- **TileLang**：route A + K-split（`ks=8`），`bN=128, threads=128, ns=3`，1000 iters。
  `partial-only` = 只发分片 kernel；`partial+reduce` = 两发。
- **launch 底噪**：空 TileLang kernel 同口径 **5.94 µs/call**（一阶段 5.52；同机共享环境波动）。
- ⚠️ 两侧的 event 都包住 launch 开销，**看比值要同时看归一化净时间**（§4.3）。

### 4.2 双侧原始表（µs/call）

**老 kernel**（`mx` = `dsv41_gemm_fp8_mx` 的 m=1 eager gemv；`mrows` = `dsv41_gemm_fp8_mrows`）

| shape | n | k | mx m=1 | mrows m=1 | mrows m=6 | **mrows M6/M1** | mrows/mx |
|---|---|---|---|---|---|---|---|
| wkv | 512 | 5120 | 6.71 | 14.72 | **85.17** | **5.78** | 2.19 |
| wq_a | 1280 | 5120 | 9.13 | 16.17 | **85.26** | **5.27** | 1.77 |
| wq_b | 4096 | 1280 | 7.72 | 8.38 | 25.99 | 3.10 | 1.09 |
| wo_b | 5120 | 1024 | 7.66 | 7.92 | 24.03 | 3.03 | 1.03 |
| **wo_a 分组** | | | | | | | |
| wo_a_nlg1 (G=1) | 1024 | 4096 | — | grp r1 = **10.33** | grp r6 = **47.67** | **4.61** | — |
| wo_a_nlg8 (G=8) | 8192 | 4096 | — | grp r1 = **29.80** | grp r6 = **160.76** | **5.39** | — |

> wo_a 的 `mx` 对照是**调用点的回退形态**（每 `(group,row)` 一次 m=1 `gemm_fp8_mx`）：
> `r1 = 6.76 / 54.05 µs`，`r6 = 40.58 / 330.65 µs`。
> ⇒ 老的"分组 GEMV"比回退快 **1.5×/1.8×**（G=1/G=8）——**分组 kernel 本身是赚的**；
> 它的问题是 **M 放大 4.6–5.4×**，不是绝对慢。

**TileLang route A**（`bN=128 ks=8 ns=3 thr=128`）

| shape | n | k | partial M=1 | M=6 | **M6/M1** | +reduce M=1 | M=6 | **M6/M1** | 净 M=1† | 净 M=6† |
|---|---|---|---|---|---|---|---|---|---|---|
| wkv | 512 | 5120 | 6.83 | 6.75 | **0.99** | 12.25 | 12.09 | **0.99** | 0.90 | 0.82 |
| wq_a | 1280 | 5120 | 6.69 | 6.67 | **1.00** | 12.19 | 12.11 | **0.99** | 0.76 | 0.73 |
| wq_b | 4096 | 1280 | 6.81 | 6.79 | **1.00** | 12.24 | 12.27 | **1.00** | 0.87 | 0.86 |
| wo_b | 5120 | 1024 | 6.75 | 6.73 | **1.00** | 12.20 | 12.43 | 1.02 | 0.82 | 0.79 |
| **wo_a_nlg1** (G=1) | 1024 | 4096 | 6.71 | 6.78 | **1.01** | 12.37 | 12.20 | **0.99** | 0.77 | 0.85 |
| **wo_a_nlg8** (G=8) | 8192 | 4096 | 18.94 | 18.95 | **1.00** | 21.71 | 21.73 | **1.00** | 13.01 | 13.02 |

† 净 = 值 − 5.94（launch 底噪）。

### 4.3 判决

1. **判据 `<1.5×`：五形状全过**（0.99–1.01）。老 kernel 的 3.03–5.78×（wo_a 4.61/5.39×）
   被压平 ⇒ **改善 3–5.8×**。
2. **归一化净时间**：稠密四形状 + `wo_a_nlg1` 的净时间都在 **0.73–0.90 µs**；老 `mrows` m=1
   的净时间（同减 5.94）为 **2.0–10.2 µs**（wq_b/wo_b 2.0–2.4；wkv/wq_a 8.8–10.2）。
   ⇒ **TileLang 单发在五形状上都不慢于、且在 `wkv/wq_a` 上快 ~10×**。
3. **`partial+reduce` 的两发 ≈ 2× 底噪 ⇒ reduce 第二发是纯开销**（一阶段结论复现）。
   集成走 ferrite 的单发 ctr 归约（proto.md §8#2）。
4. **`wo_a_nlg8` 是唯一的"非底噪主导"形状**：净 13.0 µs，权重流量 33.5 MB ⇒ **2.6 TB/s**
   （G=1 时 4.19 MB / 0.77 µs ⇒ 5.4 TB/s）。老 kernel 同流量 29.8 µs ⇒ 1.1 TB/s。
   ⇒ 分组版把 draft 侧的 wo_a **权重带宽利用率抬 2.4×**，且 M=6 免费。
5. **`ks` 的必要性**：`wo_a_nlg1` 的 grid 只有 `(8, 1, ks)`，`ks=8` ⇒ 64 blocks；
   148 SM 的机器上仍是欠覆盖。`ks` 可再抬（`ks=16/32` 归约成本不变，只是多发块）——
   集成时按"块数 ≥ SM 数"取（proto.md §5.1 的 ks 规则）。

---

## §5 ⑤ 数值形态

### 5.1 口径 A —— vs f64 真值（`selftest`，与 proto.md §4 同规）

统计 `|truth| > 0.05·max` 的元素（排除 near-zero 分母放大）：

| shape | n | k | \|C−ref\|max | p50 rel | p99 rel | **max rel** | mean rel |
|---|---|---|---|---|---|---|---|
| wkv | 512 | 5120 | 1.221e-04 | 2.093e-07 | 1.830e-06 | **3.906e-06** | 3.125e-07 |
| wq_a | 1280 | 5120 | 9.155e-05 | 2.048e-07 | 1.508e-06 | **3.261e-06** | 2.943e-07 |
| wq_b | 4096 | 1280 | 3.433e-05 | 1.084e-07 | 7.505e-07 | **1.681e-06** | 1.511e-07 |
| wo_b | 5120 | 1024 | 3.815e-05 | 1.027e-07 | 6.572e-07 | **1.744e-06** | 1.383e-07 |
| wo_a_nlg1 | 1024 | 4096 | 6.104e-05 | 7.957e-08 | 5.331e-07 | **1.080e-06** | 9.862e-08 |
| wo_a_nlg8 | 8192 | 4096 | 7.629e-05 | 8.071e-08 | 4.806e-07 | **1.455e-06** | 9.491e-08 |

`ref` = ferrite 表达式的 f32 实现 `(a·as) @ (w·ws)^T`（**不是**真值）。
`|C−ref|max`（1e-5~1e-4）是"换程序"的绝对差，相对差 1.1e-6~3.9e-6 ——**都是"正确到 f32"**。

### 5.2 口径 B —— vs **真实 ferrite kernel**（本阶段新增，`compare_old`）

`kernels/cuda/dump_proj_old.cu` 用**确定性 hash 输入**（两侧逐字节同源，脚本 `_hash_idx()`）
跑真 kernel（`dsv41_gemm_fp8_mrows` m=6 / `dsv41_wo_a_grouped_fp8` rows=6）dump 输出，
再与 TileLang route A 逐元素比：

| shape | max_abs | p50 rel | p99 rel | **max rel** | mean rel |
|---|---|---|---|---|---|
| wkv | 4.00 | 6.46e-08 | 2.10e-07 | **2.81e-07** | 5.60e-08 |
| wq_a | 4.00 | 6.54e-08 | 2.17e-07 | **3.43e-07** | 5.65e-08 |
| wq_b | 1.50 | 7.40e-08 | 2.46e-07 | **5.16e-07** | 7.24e-08 |
| wo_b | 1.00 | 7.72e-08 | 2.58e-07 | **5.43e-07** | 7.50e-08 |
| wo_a_nlg1 | 2.00 | 6.52e-08 | 2.18e-07 | **5.54e-07** | 5.70e-08 |
| wo_a_nlg8 | 4.00 | 6.55e-08 | 2.15e-07 | **7.56e-07** | 5.65e-08 |

**读法（这是本阶段最有用的一个数）**：
1. route A 与 **ferrite 现有 kernel** 的相对偏差 **≤ 7.6e-7，mean ≈ 6–7.5e-8**。
   输入是全正幅值（无抵消）的大和，`max_abs` 1–4 对应的其实是 ~1e-8 相对。
2. ⇒ **EAGER 对照的容差可以设得比 proto.md §2.2 猜的更紧**（proto 只对到 f32 参考，max_rel 1.5e-3
   是被 near-zero 放大出来的；这里同规 mask 后是 **e-7 量级**）。
3. **仍然不是逐位等价**（硬件求和顺序不可指定）⇒ **禁止 byte-compare**，走
   `swapab_parity.rs` 口径的**容差 + 文本指纹**双门禁（proto.md §7）。
4. wo_a 分组两形态与稠密四形状**同一量级**，说明**分组形态没有引入额外的数值退化**。

---

## §6 wo_a 分组版的形态决策 — deliverable ③

### 6.1 **量化在 shim 侧**（不做 route B / 不做 kernel 内量化）

老 `wo_a` 的**调用点**是 `quant_fp8(o_r) → wo_a_grouped_fp8`（`chain_dev.rs:13306-13333`，
f32 激活 → 量化 → 分组 GEMV）。**量化本来就在 kernel 外**（`dsv41_wo_a_grouped_fp8` 的 `a` 已是
`const uint8_t*` fp8 + `a_scale`）。所以：

> **TileLang 版吃 `(fp8 e4m3 + f32 per-32 scale)`，与 ferrite ABI 完全同形，shim 侧零新增。**
> 这一点让 wo_a 成为**集成成本最低的形状**——它不像稠密四形状那样需要论证"激活从哪来"。

`quant_ref()` 复刻了 `quant_kernel<0>` 的算术（per-32 `amax` → `fast_round_scale(amax, 1/448)`
→ pow2 scale → e4m3 饱和），用于对拍时**证明喂给 TileLang 的就是 ferrite 会产出的字节**。

**为什么不做 route B（dequant bf16）**：一阶段 §4 已判死（bf16 8 位尾数 ⇒ mean_rel ~1.7e-2）。
分组版没有让这条变好，反而多一层组循环 ⇒ **不复活**。

### 6.2 形态选择

| 决策 | 选择 | 理由 |
|---|---|---|
| group 维放哪 | **`grid.z`**（`blockIdx.z` 语义） | 每 block 单组、零跨组共享；与老 kernel `blockIdx.y = g` 同形，便于逐块对拍 |
| 激活怎么表达 | **2D `[MPAD, ASTRIDE]` + `A[0, g*K + ...]`** | 直接复现 ferrite ABI 的"行距 ≠ 组内 K"，**不 materialise 3D** |
| 输出怎么写 | 列偏移 `g*N`（归约时），partial 走 `P[kp, g, ...]` | 与 `out[(r*out_stride) + g*n + row]` 同构 |
| K-split | `ks` on `grid.z` 之外（即 3D grid 的第三维） | 组 × K-split 无耦合，块数 = `(N/bN)·G·ks` |
| 权重布局 | `W[G, N, K]` + `WSC[G, N/32, K/32]` | 老 ABI 是扁平 `+g*n*k`；3D 视图索引 1:1，无拷贝 |
| **唯一集成改动** | **激活 M pad 到 16**（`MPAD`） | mma `m16n8k32` 硬约束；与稠密四形状同 |

### 6.3 与老 kernel 的逐块关系

老 `wo_a_grouped_gemv_kernel` 是 **weight-stationary**（一 warp 一行，权重行 stage 一次，折 M 行）
的 SIMT GEMV；TileLang 版是 **tensor core + K-split**。两者**不是同一个程序**（累积顺序不同），
所以 §5.2 的 e-7 偏差是"换程序"的正常结果，**不是回归**。

**收益归因**：
- 老 kernel 的 M 开销来自 **M-in-register 串行折叠**（`acc` 每个 `(warp, r)` 一份，M 行重复 walk K）。
- 老 kernel 在 `nlg=1` 时 grid `(1024/8, 1) = (128, 1)` ⇒ **<1 block/SM**，latency 主导
  （`projection-family-optimization.md` §1.3 记的 72 GB/s ≈ 1% peak）。
- TileLang 版：**M 进 mma 的 tile 维度**（m16 一发覆盖 6 行）⇒ M 不再是迭代维度；
  **K-split 把块数填满机器**。两条合起来就是 §4.3 的 4.6×/5.4× → 1.01×。

---

## §7 集成形态（未来接线，本任务不做）

沿用一阶段的纪律（新增路径、不替换）：

- **gate**：`DSV41_GEMM_TILELANG`（默认 unset/OFF，严格 `== "1"`），分子旋钮 `_KS`，
  形状白名单（`k%32==0`、`n%32==0`、M pad 到 16）。
- **wo_a 专属**：`dsv41_wo_a_grouped_fp8` 的 ABI 已经是 `(a, a_scale, w, w_scale, bias, out,
  groups, rows, n, k, a_stride, out_stride)` ⇒ TileLang 版的 shim 就是**同一个调用点的参数**，
  **return 2 的 decline 语义可原样保留**（老 launcher 的 `2 = DECLINED` 约定，`:8166`）。
- **scratch**：分组 K-split 需要 `P[ks][G][16][n]` f32；用 ferrite 单发 ctr 归约则需 `ctr` u32。
  **必须 Rust 侧显式 sizing**（proto.md §7 同一坑）。
- **活性回执**（防幻影 gate）：`[proj-tilelang] ARMED m=.. n=.. k=.. G=.. ks=.. -> grid=..`。
- **数值门禁**：§5.2 的 **max_rel ≤ 1e-6** 可作为容差起点（实测 ≤7.6e-7，留 30% 余量）；
  **禁止 byte-compare**，走 `swapab_parity.rs` 的容差 + 文本指纹。

---

## §8 未做 / 风险

| # | 项 | 说明 |
|---|---|---|
| 1 | **swapAB（权重在 M、激活在 N=8）** | 一阶段 §8#1 的未做项。pad-16 的算力浪费在 latency 受限族里不体现（M6/M1 全 1.00），未验。 |
| 2 | **单发 K-split（ctr 归约）** | 本原型的 reduce 仍是第二发 launch（≈ 一个底噪）。集成时用 `swapab` 的 `atomicAdd(ctr)+最后到达者归约`。 |
| 3 | **NCU 判读** | micro bench 未上 NCU；`wo_a_nlg8` 的 2.6 TB/s 是算术反推，未验 `sm__throughput` / DRAM 计数。 |
| 4 | **e2e / 双门禁** | 只做 micro bench，**未跑 serve**。§5.2 的容差起点待集成后实测。 |
| 5 | **`wo_a` 的 `DSV41_WO_A_MTILE` 分支** | 树内还有 `wo_a_grouped_mtile_kernel`（M-into-tile 的 SIMT 版，默认 OFF）。本阶段**未与它对照**——TileLang 版是 tensor core 路线，与它不同族。 |
| 6 | **`bias`** | TileLang 版未做 bias epilogue（wo_a site 的 `bias` 恒为 null，`:8177` 明说）。四形状若需要 bias 需补。 |
| 7 | **per-128 scale 变体** | ferrite 真实是 per-32（`gemm_fp8_mrows_kernel` ABI + `chain.rs:28`）。per-128 下 bK 可放大 ⇒ 更快，但**与 ABI 不符**，未做。 |

---

## 附：复现脚本

远侧 `~/tl_proj/`（本阶段新增标 **+**）：

| 文件 | 作用 |
|---|---|
| `proj_fp8_tilelang_phase2.py` **+** | 五形状 + wo_a 分组原型（`selftest`/`det`/`cmp <outdir>`/`bench`/`all`） |
| `bench_proj_old.cu` **+** → `bench_proj_old` | 老 kernel 五形状对照（含 wo_a 分组 + 调用点回退形态） |
| `dump_proj_old.cu` **+** → `dump_proj_old` | 确定性 hash 输入下 dump 老 kernel 输出（给 `cmp` 用） |
| `dumps/old_*.f32` **+** | dump 产物（6 个形状） |
| `proj_fp8_tilelang.py` / `bench_tl.py` / `numerics.py` / … | 一阶段产物（未改） |

本地仓库对应：
- `kernels/tilelang/proj_fp8_tilelang_phase2.py`
- `kernels/cuda/bench_proj_old.cu`
- `kernels/cuda/dump_proj_old.cu`

编译（两个 TU 均约 2–2.5 min，`dsv41_kernels.cu` 14.8k 行）：

```bash
nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a bench_proj_old.cu -o bench_proj_old
nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a dump_proj_old.cu  -o dump_proj_old
./bench_proj_old 2000
mkdir -p dumps && ./dump_proj_old dumps
python3 proj_fp8_tilelang_phase2.py all dumps
```

---

*工部 · 2026-09-13 · 全部数字来自远端 GPU 实测；数值口径与 ms 来源已逐条标注。
wo_a 形状按 `dsv41_kernels.cu:8165` / `chain_dev.rs:13270` 的代码事实实现（非任务书标注），
差异已在 §0 点名。*
