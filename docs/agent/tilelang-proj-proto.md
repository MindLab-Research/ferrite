# TileLang 投影 GEMM 原型（fp8 e4m3，tensor core 路线）

> 工部 · 2026-09-13 · **GPU 实测**（远端 B300 sm_103a，`ssh ubuntu@43.202.208.136`；micro bench，未跑 e2e serve）。
> 交付：可运行的 TileLang fp8 投影 GEMM 原型 + M=1/6 比值 + 四形状 benchmark 表 + AOT 产物形态 + 数值形态。
> 前置：`tensorcore-proj-design.md`（mma 路线设计 + (b′) program-consistent parity）、
> `mrows-mtile-design.md`、`projection-family-optimization.md`（"投影族是 launch/指令受限，不是带宽受限"）。
> 代码基线：`kernels/cuda/dsv41_kernels.cu` HEAD（`gemm_fp8_mrows_kernel<M>` / `gemm_fp8_swapab_kernel`）。

---

## §0 一句话判决

**TileLang 的 `T.gemm` 能让 M=6 的投影成本 ≈ M=1**：四形状实测 **M6/M1 = 0.98–1.03**
（老 `gemm_fp8_mrows<M>` 是 **5.73 / 5.26 / 3.11 / 3.04×**）。判据 `<1.5×` **通过**，且优于
SGLang 的 1.2–1.3× 机制口径。机制是设计文档 §1.3 那条：**M 缩进 MMA 的 tile 维度后，
m=1 与 m=6 发射的 mma 指令数完全相同**（都是 1 个 m16 M-tile），M 不再是"要迭代的维度"。

| # | 项 | 判决 |
|---|---|---|
| ① | TileLang fp8 GEMM 原型（GPU 实测） | ✅ route A（原生 fp8 mma）与 route B（dequant bf16）都跑通 |
| ② | M=1/6 比值 | ✅ **0.98–1.03**（判据 <1.5×） |
| ③ | 四形状 benchmark | ✅ 见 §5 |
| ④ | AOT 产物形态 | ✅ `.cu`(5.9KB) + `.so`(77–123KB)；240 regs / 0 spill（§6） |
| ⑤ | 数值形态 | ✅ route A **比 f32 SIMT 参考更准**；route B 不可用（§4） |

**数值路线结论**：落地走 **route A（原生 fp8 mma + per-32-K-block scale 作用在 mma 输出上）**——
它镜像了树内已有前例 `gemm_fp8_swapab_kernel:800-813` 的 consume 模式，**数值上比 ferrite 的
f32 SIMT 表达式更接近真值**；route B（bf16 dequant）精度 ~1.7e-2，不可用（WOB 形状）。

---

## §1 环境验证 — deliverable ①

| 项 | 值 |
|---|---|
| host | `ubuntu@43.202.208.136`，8 × NVIDIA B300 SXM6 AC |
| compute capability | **10.3（sm_103a）** |
| Python / CUDA | 3.12.3 / CUDA 13.2（`nvcc V13.2.51`） |
| TileLang | **0.1.14**（`~/.local/lib/python3.12/site-packages/tilelang`） |
| `import tilelang` | ✅ |

**bf16 hello-GEMM（T.gemm 首次跑通 B300）**：`C[6,512] = A[6,5120] @ B[512,5120]^T`，
block (16,64,128)、`T.Pipelined(ns=3)`、128 threads ⇒ **max_abs_err 1.57e-3**（bf16 容差内）。
TileLang→B300 通路 OK。

---

## §2 fp8 API 与形状约束

**TileLang 0.1.14 的 fp8 dtype**（`tilelang.language`）：

| 用途 | TileLang dtype | torch dtype | 备注 |
|---|---|---|---|
| 权重 / 激活 e4m3 | `T.float8_e4m3fn` | `torch.float8_e4m3fn` | OCP e4m3，**与 ferrite `e4m3_to_f` 位模式一致**（sign1/exp4/mant3，bias 7） |
| 权重 block scale ue8m0 | `T.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `T.cast(x,"float32")` 直接得到 2^(b−127) |

**`T.gemm` 签名**（`tilelang.language.gemm`）：
```python
T.gemm(A, B, C, transpose_A=False, transpose_B=False, policy=GemmWarpPolicy.Square,
       clear_accum=False, k_pack=1, ...)
```
- `A/B` 为 fp8 shared/fragment，`C` 为 f32 fragment ⇒ 编译到 **tensor core**。
- 生成的 CUDA（`get_kernel_source()`）用 **TMA（`CUtensorMap`）+ warp specialization + mbarrier**，
  即 Blackwell 的 TMA/异步拷贝路径，不是朴素 SIMT。

**形状约束（硬）**：`T.gemm` 的 **M 维必须能被 16 整除**（mma `m16n8k32`），否则：
```
InternalError: Check failed: (M % kMPerWarp == 0): M must be divisible by 16, but got 6
```
⇒ 激活的 M 维 **pad 到 16**（`MPAD=16`）。这是设计文档 §1.2 的「形态 A：激活在 M」。

> **为什么 pad-16 不影响"M=6≈M=1"**：实测（§5）M=1 与 M=6 的耗时相同，因为
> **两者的 mma 指令数完全一样**（都发射 1 个 m16 M-tile、同样的 n8 子块数）。
> 「6/16 = 37.5% 利用率」是**算力浪费**，但本族是 **latency/指令受限**（projection-family §0-3），
> 空闲 lane 是免费的。设计文档 §1.3「每 MAC 指令数」才是杠杆，不是 tile 利用率。
> （若将来要吃掉那 37.5%，可换 swapAB 把权重放 M、激活放 N=8 列 —— §8 未做项。）

**scale 布局（以 ferrite 代码为准，非任务书的 per-128）**：
`gemm_fp8_mrows_kernel` 的 ABI 与 `chain.rs:28` 明确是 **ue8m0 32×32 块**：
- `a_scale [m, k/32]` f32（激活，per-32-K）
- `w_scale [n/32, k/32]` **ue8m0**（权重，32×32 块）
本原型的 scale 粒度**按 ferrite 真实布局（per-32）实现**。任务书写「per-128」与代码不符——
per-32 比 per-128 更难（scale 施加次数 ×4），因此原型在更严的粒度上通过了判据。

---

## §3 两条路线 — deliverable ①

### 3.1 route A —— 原生 fp8 mma + per-32-K-block scale（**推荐**）

镜像 `gemm_fp8_swapab_kernel:800-813`：mma 出 raw fp8 乘积和 `d`，再 `acc += d * sc`，
其中 `sc = a_scale[kb] * ue8m0_to_f(w_scale[kb])`（per-32-K block）。

```python
# 每个 32-K 块：raw mma -> 乘 scale 累加（bK=32 即 scale 块，与 mma 的 K=32 对齐）
for ko in T.Pipelined(Kc//32, num_stages=ns):
    T.copy(A[0, gko*32], A_sh); T.copy(W[bx*bN, gko*32], W_sh)
    T.gemm(A_sh, W_sh, C_p, transpose_B=True, clear_accum=True)   # C_p = Σ_{k∈32} a·w
    for i, j in T.Parallel(MPAD, bN):
        C_l[i,j] += C_p[i,j] * ASC[i,gko] * T.cast(WSC[bx*(bN//32)+j//32, gko], "float32")
```

**踩坑（必读）**：不要在 pipelined loop 里用 `T.alloc_shared` 自建 `asc_sh/wsc_sh` 缓冲
再在里面用 `T.Parallel` 写 —— TileLang **不为这种自建缓冲做多缓冲**，stage ko+1 的写会覆盖
stage ko 还没读的值 ⇒ **静默错值**（实测非确定性：ns=1 时 rel≈4e-2，多次运行结果漂移）。
**解法**：scale 在 epilogue 里**直接读全局**（L1/L2 命中，量极小），无中间缓冲、无竞态。
修复后跨 rep/num_stages 完全确定（2.747e-4 稳定复现）。

### 3.2 route B —— dequant bf16 + bf16 mma（备用，**数值不可用**）

把 fp8 按 per-element scale dequant 到 bf16（在 smem 里），再做 bf16 `T.gemm`。
结构更简单，但 **bf16 只有 8 位尾数** ⇒ 相对误差 ~2–5e-4 起步（§4）。**不推荐落地**。

---

## §4 数值形态 — deliverable ⑤

口径（**未来 EAGER 对照的标尺**）：wkv `N=512 K=5120 m=6`，权重/激活 fp8 e4m3（randU(−1,1)），
`a_scale∈[0.75,1.25]` f32、`w_scale` ue8m0∈[2^-4,2^0]；**与 f64 真值**比，
统计 `|truth| > 0.05·max` 的元素（排除 near-zero 分母放大）：

| 程序 | p50 rel | p99 rel | max rel | mean rel |
|---|---|---|---|---|
| `SIMT_f32`（ferrite 表达式 `(a·as)·(w·ws)` 逐元素 f32 累加） | 1.77e-07 | 1.40e-06 | 2.64e-06 | 2.60e-07 |
| **route A**（tensor core `(Σ32 a·w)·(as·ws)`） | **7.29e-08** | **6.42e-07** | **1.41e-06** | **1.14e-07** |
| route B（bf16 dequant） | — | — | — | **1.69e-02**（不可用） |

**读法**：
1. **route A 比 f32 SIMT 参考更接近真值**（约 2×）。原因 = 设计文档 §2.1 的 D1/D2：
   tensor core 在 32 元素块内以更高内部精度求和，且 scale 只施加一次（`(Σa·w)·scale`
   vs SIMT 的逐元素 `(a·as)·(w·ws)`——后者每元素 2 次舍入）。**这不是"换程序就更差"。**
2. **route A vs SIMT 的差**（max_rel ≈1.5e-3，同样由 near-zero 驱动；量级 f32-精确）满足
   设计文档 §2.2 的判定：**不是逐位等价**（硬件求和顺序不可指定），但都是"正确到 f32"。
   ⇒ EAGER 对照必须走**容差 + 文本指纹**（`swapab_parity.rs` 口径），**不要**尝试 byte-compare。
3. **route B 排除**：bf16 dequant 的 8 位尾数把精度掉到 ~1e-2 量级 ⇒ 正是 WOB
   （`DSV41_VERIFY_WOB_MROWS_F32`）那种"换数值语义"的形状，**不做**。

---

## §5 Benchmark — deliverable ③

### 5.1 方法

- 老 kernel：独立 TU `#include "dsv41_kernels.cu"`，直接调 `dsv41_gemm_fp8_mx`（m=1 eager gemv）
  与 `dsv41_gemm_fp8_mrows`（m=1 / m=6 verify），CUDA event、2000 iters、预分配 out。
- TileLang：route A + **K-split（ks=8，partial 写 P 后确定性 reduce，mirror 设计文档 §3.3 的 ks 规则）**，
  预分配 C/P，1000 iters。`bN=128, threads=128, num_stages=3, ks=8`。
- **为什么必须 K-split**：不 split 时 `wkv`（N=512）只有 `N/bN` = 4–8 块，148 SM 的机器上
  **并行度严重不足**（M=1 达 32 µs）。K-split 把块数 ×ks 填满机器（设计文档 §3.3 的 ks 规则）。

### 5.2 老 kernel 对照（iters=2000，µs/call）

| shape | n | k | `mx` m=1（eager gemv） | `mrows` m=1 | `mrows` m=6 | **mrows M6/M1** |
|---|---|---|---|---|---|---|
| wkv | 512 | 5120 | 6.71 | 14.83 | **84.95** | **5.73** |
| wq_a | 1280 | 5120 | 9.10 | 16.19 | **85.22** | **5.26** |
| wq_b | 4096 | 1280 | 7.74 | 8.37 | 26.01 | 3.11 |
| wo_b | 5120 | 1024 | 7.66 | 7.90 | 24.05 | 3.04 |

> 复现了任务书的对照口径：`mrows` 的 M=6 是 M=1 的 **5.7× / 5.3× / 3.1× / 3.0×**
> （M-in-register 串行 + 每行重读权重的结构）。

### 5.3 TileLang route A（µs/call）

| shape | **partial-only M=1** | **M=6** | **M6/M1** | partial+reduce M=1 | M=6 | M6/M1 |
|---|---|---|---|---|---|---|
| wkv | 6.47 | 6.47 | **1.00** | 11.63 | 11.65 | **1.00** |
| wq_a | 6.53 | 6.54 | **1.00** | 11.63 | 11.57 | **1.00** |
| wq_b | 6.45 | 6.37 | **0.99** | 11.67 | 11.70 | **1.00** |
| wo_b | 6.46 | 6.46 | **1.00** | 11.58 | 11.57 | **1.00** |

（`bN=128 ks=8 ns=3 thr=128`；`bN∈{64,128,256}`、`ks∈{8,16,32}` 全扫，比值都是 0.98–1.03。）

### 5.4 Launch 底噪校准（**关键**）

同一套 event 计时测 **空 kernel**：**5.52 µs/call**（本机共享环境的每次 launch 开销）。

⇒ **减掉底噪后的真实 kernel 时间**：

| shape | TileLang partial-only（净） | 老 `mrows` m=1（净，=值−5.52） |
|---|---|---|
| wkv | **≈0.95 µs** | ≈9.3 µs |
| wq_a | ≈1.01 µs | ≈10.7 µs |
| wq_b | ≈0.93 µs | ≈2.9 µs |
| wo_b | ≈0.94 µs | ≈2.4 µs |

**结论**：
1. 绝对时间在 ~1 µs 量级 ⇒ 5.3 表的 ~6.5 µs 里 **85% 是 launch 底噪**，比值必须看校准后的口径。
2. TileLang 单 kernel（partial-only）**在四形状上都不慢于老 mrows 的 m=1**，
   且 `wkv/wq_a` 快 ~10×。
3. `partial+reduce` 的两发 ~11.5 µs ≈ 2× 底噪 ⇒ **reduce 的第二发是纯开销**。
   集成时若用 ferrite 的「最后一个 ks 分片原子计数后归约」单发结构（`swapab` 的 `ctr` 模式），
   可回到单 launch。

**判据判定**：**M6/M1 = 0.99–1.00 < 1.5× ✅**（老 5.73× ⇒ 改善 ~5.7×）。

---

## §6 AOT 产物形态 — deliverable ④

`JITKernel` 提供 AOT 接口（`tilelang.jit.kernel.JITKernel`）：

| API | 产出 | 用途 |
|---|---|---|
| `get_kernel_source()` | device `main_kernel` 完整 CUDA C++ 源码 | **`.cu` dump** |
| `export_library(path)` | 预编译 `.so`（含 host 侧 launcher） | **FFI 集成** |
| `export_ptx()` / `export_sass()` | PTX / SASS | 审计 |
| `get_host_source()` | host 侧调用源码 | FFI 签名参考 |

**四形状产物**（route A，`bN=128 ks=8 thr=128 ns=3`）：

| shape | n | k | `.cu` | `.so` |
|---|---|---|---|---|
| wkv | 512 | 5120 | 5904 B | 122576 B |
| wq_a | 1280 | 5120 | 5907 B | 122576 B |
| wq_b | 4096 | 1280 | 5883 B | 81616 B |
| wo_b | 5120 | 1024 | 5883 B | 77520 B |

**资源占用**（`wkv_routeA.cu` 用 `nvcc -gencode arch=compute_103a,code=sm_103a -Xptxas -v`）：
```
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 240 registers, used 1 barriers, 1024 bytes smem
```
⇒ **0 spill**，与设计文档 §3.5 的 CUDA skeleton（64–72 regs / 0 spill）同族。

**生成的 kernel 形态**（`wkv`）：TMA `CUtensorMap` 描述符（A/W 各一）+ `extern __shared__`
动态 smem + `mbarrier_mem[6]` + **warp specialization**（`warpgroup_reg_dealloc/alloc`，
producer warp 发 `tma_load`，consumer warp 跑 mma）——即 Blackwell 的异步流水，不是 SIMT。

**Rust FFI 集成路径**：`.so` 是 CUDA runtime 标准产物（含 host launcher），
Rust 侧可 `dlopen` + 按 `get_host_source()` 的签名调；或把 `.cu` 并入
`kernels/cuda/build.sh` 的 nvcc 流程与现有 `ferrite-kernel` 的 `extern "C"` 符号并列。
本任务**不接线**（见 §7）。

---

## §7 集成形态（未来接线，本任务不做）

按任务纪律：这是**新增路径**，不替换。建议 gate（镜 `DSV41_PROJ_MMA` 的纪律）：

- **gate**：`DSV41_GEMM_TILELANG`（默认 unset/OFF，严格 `== "1"`），分子旋钮 `_KS`（显式 ks）、
  形状白名单（`k%32==0`、`n%32==0`、pad 到 MPAD=16 的激活缓冲）。
- **ABI**：与 `dsv41_gemm_fp8_mrows` 同形（`a/a_scale/w/w_scale/bias/out` + `m,n,k,out_stride`），
  激活侧需要 **M pad 到 16** 的 staging（或换 swapAB，见 §8）。
- **活性回执**（防幻影 gate）：`[proj-tilelang] ARMED m=.. n=.. k=.. ks=.. -> grid=..`
- **数值门禁**：route A 非逐位 ⇒ 必须走双门禁（`step_ms` AND `mean-k`）
  + 与 `swapab_parity.rs` 同规的文本指纹，**禁止** byte-compare。
- **scratch**：K-split 需要 `P[ks][16][n]` f32；若用 ferrite 的单发 ctr 归约则需
  `ctr[n/bN]` u32。**必须在 Rust 侧显式 sizing**（否则越界，与设计文档 §3.4 同一坑）。
- **第一刀建议**：先 `ks=1` + `bN` 取大（wq_b/wo_b 的 n 大、块数够），零 scratch、
  零 reduce 第二发，就能拿 n=4096/5120 两档的符号与数值结论。

---

## §8 未做 / 风险

| # | 项 | 说明 |
|---|---|---|
| 1 | **swapAB（权重在 M、激活在 N=8）** | 设计文档 §1.2 的形态 B（75% vs 37.5% 利用率）。本原型走 pad-16（形态 A）。因本族 latency/指令受限，pad-16 的浪费未体现（实测 M6/M1=1.00 已达标）；swapAB 是**吃算力**的后续优化，不是达标前提。 |
| 2 | **单发 K-split（ctr 归约）** | 本原型的 reduce 是第二发 launch（~5.5µs 底噪）。集成时用 `swapab` 的 `atomicAdd(ctr)+最后到达者归约`结构可省掉。 |
| 3 | **NCU 判读** | micro bench 未上 NCU。设计文档 §3.5 #5 的预期（`sm__throughput` 抬起、LUT 相关 LDS 消失）未验。 |
| 4 | **e2e / 双门禁** | 本任务只做 micro bench，**未跑 serve**。数值门禁（step_ms + mean-k）待集成后做。 |
| 5 | **per-128 scale 变体** | 任务书写 per-128；ferrite 真实是 per-32。per-128 下 bK 可放大到 128 ⇒ 迭代数 ÷4、更快；但**与 ferrite ABI 不符**，未做。 |
| 6 | **mode-0/1 等 SIMT 旋钮** | TileLang 程序不含 LUT / 不 materialise 激活 operand ⇒ 与 `DSV41_GEMV_A32` / `DSV41_MROWS_ACT_CPASYNC` 无关（不重读这两个 gate）。 |

---

## 附：复现脚本

远侧 `~/tl_proj/`：

| 文件 | 作用 |
|---|---|
| `hello_bf16.py` | §1 bf16 hello-GEMM |
| `protoA.py` / `protoB.py` | §3 两条路线的正确性 |
| `diagA.py` / `diag2.py` | §3.1 竞态定位与修复 |
| `bench_old.cu` | §5.2 老 kernel 对照（独立 TU，`#include "dsv41_kernels.cu"`） |
| `bench_tl.py` / `bench2.py` / `bench3.py` | §5.3/5.4 TileLang 基准 + launch 底噪校准 |
| `numerics.py` / `final_ev.py` | §4 数值形态 + §6 AOT 导出 |

---

*工部 · 2026-09-13 · 全部数字来自远端 GPU 实测；数值口径与 ms 来源已逐条标注。*
