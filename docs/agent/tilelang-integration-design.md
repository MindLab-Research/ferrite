# TileLang 产物 × ferrite Rust FFI 的集成架构 — 设计

> 角色：中书省（方案制定 / 架构决策）。**只读分析 + 设计**，本任务不落实现代码、不跑 GPU/e2e。
> 产出：`docs/agent/tilelang-integration-design.md`。
> 上游 peer：`tilelang-proj-proto`（投影族 GEMM 原型）、`tilelang-moe-grouped`（MoE grouped 专家原型）。
> 前置文档：`docs/agent/tensorcore-proj-design.md`（§2 逐位不可能 / §3 gate 契约 / §7 并入时机）、
> `docs/agent/proj-mma-verdict.md`（**判死先例**：verify 单侧换程序 ⇒ mean-k 2.240 → 0.020）、
> `kernels/cuda/build.sh`（build-stamp / same-source gate）、`crates/ferrite-kernel/src/devrt.rs`（dlopen 体系）。
> 日期：2026-09-13。
> 纪律：本任务**禁 GPU/e2e**；远端**只读 + compile-only** 实验已执行——§1 的每一段输出都是远端实测原文。

---

## §0 一句话

TileLang 0.1.14 的 kernel **不是**只能 JIT 跑的 Python 物件：它同时产出一份**自包含的 CUDA 源码**
（`~/.tilelang/cache/.../<sha>/device_kernel.cu`）与**一个自足的 `.so` 指针**（`get_kernel_source()`），
后者用现成 nvcc **独立编译 EXIT=0 且端到端数值正确**（§1.3）。
⇒ 集成路线取 **(a) 源码合入**：把生成物冻结进 `kernels/cuda/tilelang_gen/`，配一个**手写 launcher shim**
（ferrite 现有 `extern "C" int fn(..., CuStream)` 约定，`dsv41_proj_mma_skel.cu:428` 同形），
`build.sh` 把它当一个普通 TU 链进 `libferrite_kernels.so` —— **devrt / Rust 侧的 dlopen 体系零改动**，
build-stamp 自动覆盖新 TU（`build.sh:136-138` 的 `sha256(SRCS)`）。两条 gate（`DSV41_GEMM_TILELANG` /
`DSV41_MOE_TILELANG`）挂在**既有分派链的最前面**，回退链与 one-shot「armed-but-inert」提示照抄现有契约。

**五问判决**（对应验收 ①②③④⑤）：

| # | 问题 | 判决 |
|---|---|---|
| ① | 产物形态：TileLang 能否 dump 出自足 CUDA 源码并独立编译 | ✅ **实测通过**（§1）。`get_kernel_source()` / `export_sources()` / `export_ptx()` 三入口；裸指针 ABI 需 `TL_DISABLE_TMA_LOWER=1`（§1.2） |
| ② | 三条集成路线（源码合入 / 独立 .so / Python 常驻） | ✅ 推荐 **(a) 源码合入**；(b) 因 TileLang 缓存的 `.so` **依赖 TVM FFI 运行时、非自足**而出局；(c) 架构破坏，否决（§2） |
| ③ | gate / precedence / 回退链 | ✅ `TILELANG > PROJ_MMA > MPAR > MTILE > legacy`（投影族）；`MOE_TILELANG > EXPERT_TCGEN05_E4M3/EXPERT_GROUPED > per-(row,slot)`（MoE）（§4） |
| ④ | 数值对照验收协议 | ✅ 三层判据：**内部契约（先过）→ 数值容差 → 端到端文本红线**；把 proj-mma 的 mean-k 崩塌写进红线（§5） |
| ⑤ | 实施步骤清单 | ✅ §6，含 vendoring 足迹与 RACI（§6.4） |

---

## §1 产物形态调研 — deliverable ①

### 1.1 实验环境（远端实测）

| 项 | 值 |
|---|---|
| 机器 | `ubuntu@43.202.208.136`（b300-4，8× B300 SXM6 AC，compute_cap **10.3 = sm_103a**） |
| Python | `/opt/dlami/nvme/dsv41_venv/bin/python`（3.12） |
| TileLang | **0.1.14**（`tilelang-0.1.14.dist-info`；另有一份同版本 user-site） |
| nvcc | `/usr/local/cuda-13.2/bin/nvcc`，release 13.2, V13.2.51 |
| cache 根 | `~/.tilelang/cache/0.1.14/` |

### 1.2 产物三形态（实测）

**(i) JIT 模式** —— `@tilelang.jit` 返回 `JITKernel`；默认 `execution_backend == "tvm_ffi"`。

**(ii) cache 目录** —— 每个 kernel 一个 `<sha256>/` 目录，内含**四件套**：

```
~/.tilelang/cache/0.1.14/
├── linux-x86_64/kernels/<sha256>/
│   ├── device_kernel.cu     # ← 生成的 CUDA 设备源码（自包含，§1.3 独立编译通过）
│   ├── host_kernel.cu       # ← TVM-FFI host 胶水（调 TVMFFIFunctionCall，非裸指针）
│   ├── executable.so        # ← 缓存 .so（NEEDED 只有 libc，但见 §1.4 的 UNDEFINED）
│   ├── manifest.json        # ← 每个文件的 sha256 + size（可做产物指纹）
│   └── params.pkl
└── cuda-binaries/*.cubin    # 编译中间产物
```

**(iii) 三入口（0.1.14 实测签名）**：

| 入口 | 作用 |
|---|---|
| `JITKernel.get_kernel_source(kernel_only=True) -> str` | 直接返回设备 CUDA 源码（**这是路线 (a) 的入口**） |
| `JITKernel.export_sources(kernel_path=..., host_path=...)` | 落盘 device / host 两份源码 |
| `JITKernel.export_ptx(path)` / `export_sass` / `show_source` | 汇编级产物（审计用） |
| `JITKernel.export_library(path)` / `get_host_source()` | 库级 / host 源码 |

### 1.3 关键实验：dump → 独立编译 → 端到端启动（**全部实测通过**）

**实验 A（默认形态）**：`T.gemm(A_s, B_s, C_l)`，1024³ bf16，`block (128,128,64)`，`num_stages=3`。

```cpp
// get_kernel_source() 实测输出（节选）
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(
    __grid_constant__ const CUtensorMap A_desc,
    __grid_constant__ const CUtensorMap B_desc,
    bfloat16_t* __restrict__ C) { ... tl::mma_sync<...kBFloat16...> ... }
```

⚠️ **默认形态是 TMA 描述符 ABI**（`__grid_constant__ const CUtensorMap`）——host 侧必须
`cuTensorMapEncodeTiled` 才能启动，与 ferrite 的裸指针约定不同构。**这是本设计要绕开的那道坎。**

**实验 B（裸指针形态，**推荐形态**）**：同一 GEMM + `pass_configs={TL_DISABLE_TMA_LOWER: True, TL_DISABLE_WARP_SPECIALIZED: True}`：

```cpp
// 实测输出：与 ferrite 现有 kernel ABI 完全同构
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(
    const bfloat16_t* __restrict__ A,
    const bfloat16_t* __restrict__ B,
    bfloat16_t* __restrict__ C) { ... }
```

**独立编译（实验 A 与 B 各一次）**：

```bash
nvcc -arch=sm_103a -O3 -std=c++17 \
     -I <tilelang>/src -I <tilelang>/3rdparty/cutlass/include \
     -cubin -o out.cubin dumped_kernel.cu      # EXIT=0
```
> 仅 `-I <tilelang>/src` 会失败：`common.h:43` 拉 `cute/numeric/numeric_types.hpp`，
> `common.h:46` 拉 `cutlass/bfloat16.h` ⇒ **cutlass include 是硬依赖**（§1.4 量化）。

**端到端（dump 的 .cu → 独立 host harness → 启动 → 数值对照 fp32 参考）**：

```
launch err=no error
maxrel=1.385e-02  meanrel=1.410e-03  nbad(rel>1e-2)=1 / 1048576
```

⇒ 生成源码**自包含、可独立编译、可被外部 host 以裸指针启动、数值正确**（1/100 万元素 >1e-2，
符合 bf16 输入 × fp32 累加的舍入画像）。**路线 (a) 的最大不确定性被消除。**

### 1.4 依赖足迹与自足性（决定 vendoring 成本与路线取舍）

| 量 | 实测 |
|---|---|
| `tl_templates/`（生成码 `#include <tl_templates/cuda/...>`） | **46 文件 / 1.1 MB**（`cuda/` 29 头 + `hip/`） |
| `cutlass/include/`（`common.h` 的 `bfloat16.h`/`float8.h` + `cute/`） | **27 MB**（`3rdparty/cutlass` 共 30 MB） |
| 生成码传递闭包（`nvcc -M`） | 542 头；其中 **33 个来自 `cutlass/include`，9 个来自 `tl_templates/cuda`** |
| 生成码实际 `#include <tl_templates/...>` | 8 个：`copy.h` `cuda_bf16_fallbacks.cuh` `debug.h` `instruction/mma.h` `ldsm.h` `reduce.h` `scan.h` `threadblock_swizzle.h` |

**TileLang 缓存 `executable.so` 的自足性（路线 (b) 的死因证据）**：

```
readelf -d executable.so   →  NEEDED: libc.so.6  （只有这一条）
nm -D --undefined-only     →  U TVMBackendGetFuncFromEnv
                               U TVMFFIEnvTensorAlloc
                               U TVMFFIErrorSetRaisedFromCStr
                               U TVMFFIFunctionCall     ← 四个 TVM FFI 运行时符号
nm -D --defined-only       →  T __tvm_ffi_main   （唯一导出）
```

⇒ 缓存 `.so` **不是自足动态库**：它把 host 侧全部押在 **TVM FFI 运行时**（Python 进程内的
`tvm_ffi` 库）上。脱离 TVM 运行时 `dlopen` 它会因未解析符号失败，或者只能走 TVM FFI 的
`TVMFFIEnvTensorAlloc`/`TVMFFIFunctionCall` 协议（≠ ferrite 的裸指针 stream 约定）。

### 1.5 §1 判决

- **产物形态成立**：`get_kernel_source()` 给出**自包含 CUDA 源码**，可冻结、可审计、可独立编译。
- **ABI 可控**：`TL_DISABLE_TMA_LOWER=1`（+`TL_DISABLE_WARP_SPECIALIZED=1`）产出**裸指针 ABI**，
  与 ferrite devrt 的 `<<<grid, block, smem, stream>>>` 约定同构（对照 `dsv41_proj_mma_skel.cu:505`）。
- **成本**：vendoring `tl_templates`（1.1 MB，46 文件）+ `cutlass/include`（27 MB）——见 §6.1 的裁剪方案。

---

## §2 三条集成路线对比 — deliverable ②

| 维度 | **(a) 源码合入（推荐）** | (b) 独立 `.so` + 第二次 dlopen | (c) Python 常驻 JIT 服务 |
|---|---|---|---|
| 产物 | `device_kernel.cu` → 冻进 `kernels/cuda/tilelang_gen/` | TileLang cache 的 `executable.so` | Python 进程 + IPC |
| 编译期 | `build.sh`（现有 nvcc 一条命令） | 两条独立构建流水线 | 运行时 |
| 运行时 | **单 `.so`**，`devrt.rs:540,543` 一次 dlopen 不变 | 第二次 dlopen + 双 ABI 管理 | 常驻进程 + 序列化 |
| 自足性 | ✅ 只依赖 nvcc + vendored 头 | ❌ **实测依赖 TVM FFI 运行时**（§1.4）→ 必须链 `libtvm_ffi` 或自写 host glue（≡ 退化成 (a) 的 shim） | ❌ 依赖完整 Python/TileLang 环境 |
| build-stamp | ✅ **自动覆盖**（`build.sh:136-138` 的 `sha256(SRCS)` 含新 TU） | ❌ 需给第二产物另立 stamp + Rust 校验 | ❌ 版本漂移不可控 |
| graph capture | ✅ 普通 kernel launch，进 capture 无障碍 | ⚠️ 需自控 host glue 的 capture 语义 | ❌ **致命**：capture 期间不能有 Python 帧/GC/同步 |
| 迭代速度 | 慢（改 kernel 要重生成 + 重编） | 中 | 快 |
| 一致性 | ✅ kernel 名出现在现有 `err=0/faults=0` 回执链 | ⚠️ 两套日志 | ❌ |
| 维护性 | 生成物**是**生成物——不做人工可读性承诺（见 §7.1） | 差 | 最差 |

**推荐 (a)，理由（逐条锚定）**：

1. **契约最小**：`devrt.rs:540,543` 的 `DevRuntime::open` 只 dlopen 一个 `libferrite_kernels.so`，
   路线 (a) **一个字节都不改**；路线 (b) 要引入第二个句柄、第二套没找到符号的降级语义。
2. **build-stamp 白送**：`build.sh:136-138` 的 `CU_HASH` 已经把 `SRCS` 的 sha256 折进 `BUILD_ID`，
   新 TU 一旦进 `SRCS` 就自动进指纹——**same-source gate 不用改一行**（这是本项目 #1 测量偏置陷阱的护栏）。
3. **capture 安全**：现有 launcher 全是 `extern "C" int fn(..., CuStream s)` + `<<<...,s>>>`
   （`dsv41_proj_mma_skel.cu:505`），生成的 kernel 是普通 `__global__`，同一条路。
4. **(b) 已被实测排除**：TileLang 缓存的 `.so` 不自足（§1.4），走 (b) 要么拖进 TVM FFI 运行时，
   要么自己写 host glue——而自写 host glue **就是 (a) 的 launcher shim**。**(b) 相对 (a) 没有任何净收益。**

**否决 (c)**：JIT 进程 + IPC 与 serve 的 CUDA-graph 捕获（`devrt.rs:73-78` 的 `stream_begin_capture`）
结构性冲突；且每次 getenv/GC 都可能打断 capture。**serve 常驻路径只允许纯 device 代码。**


---

## §3 接线架构

### 3.1 架构图（data / build 两条流）

```
        远端 B300（一次性、离线）                     本仓 ferrite（可重复构建、可复现）
 ┌────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
 │ python + tilelang 0.1.14           │    │ kernels/cuda/                                        │
 │   @tilelang.jit(pass_configs={...})│    │   tilelang_gen/<name>_tl.cu      ← ★ 冻结的生成物      │
 │   T.prim_func: fp8/bf16 GEMM       │    │   tilelang_gen/<name>_launch.cu  ← ★ 手写 launcher shim│
 │        │  T.gemm → mma             │    │   tilelang_inc/tl_templates/     ← vendored (1.1 MB)  │
 │        ▼                           │    │   tilelang_inc/cutlass/          ← vendored (裁剪, §6.1)│
 │   JITKernel.get_kernel_source() ───┼───►│   build.sh  SRCS += *_tl.cu + *_launch.cu             │
 │   export_sources(manifest)         │    │        │  (nvcc -gencode sm_103a)                     │
 └────────────────────────────────────┘    │        ▼                                              │
                (人工 review + git commit)  │   libferrite_kernels.so   ← 单一 dlopen 目标           │
                                            └────────────┬─────────────────────────────────────────┘
                                                         │
 ┌───────────────────────────────────────────────────────▼──────────────────────────────────────────┐
 │ Rust                                                                                              │
 │  crates/ferrite-kernel/src/devrt.rs:540,543  DevRuntime::open  ── dlopen(libferrite_kernels.so)  [不变]│
 │  crates/ferrite-models/src/dsv41/device.rs                                                       │
 │     KernelDev { ..., gemm_fp8_mrows_tl: ko!(rt, "dsv41_tilelang_gemm_...") }  ← 新增一个 Option 符号│
 │  crates/ferrite-models/src/dsv41/chain_dev.rs                                                    │
 │     fn proj_mrows() { if Self::gemm_tilelang() { ...tilelang 臂... } if Self::proj_mma() {...} } │
 │     gate: DSV41_GEMM_TILELANG / DSV41_MOE_TILELANG  (OnceLock, default OFF)                       │
 └───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 产物落位（文件布局，git 可追踪）

```
kernels/cuda/
├── tilelang_gen/
│   ├── PROVENANCE.md              # ★ 生成清单：tilelang 版本 + pass_configs + 生成脚本 + 源 sha256
│   ├── proj_mrows_tl.cu           # device 源码（tilelang-proj-proto 交付）
│   ├── proj_mrows_tl_launch.cu    # 手写 shim：cudaFuncSetAttribute(INIT) + <<<grid,block,smem,s>>>
│   ├── moe_grouped_tl.cu          # device 源码（tilelang-moe-grouped 交付）
│   └── moe_grouped_tl_launch.cu
└── tilelang_inc/                  # vendored 头（版本锁定，随生成物一起 commit）
    ├── tl_templates/cuda/...      # 46 文件
    └── cutlass/                   # 裁剪子集（§6.1）
```

**为什么冻结而不在 build 时重生成**：build.sh 必须能在**没有 Python / 没有 TileLang** 的机器上跑
（现有纪律：`nvcc` compile-only，无 GPU 即可）。冻结生成物 = `build.sh` 只依赖 nvcc + vendored 头，
且生成物与 kernel 源码一样进 **build-stamp**（`build.sh:136-138`）。这与既有 TU 的处理方式**完全一致**。

### 3.3 launcher shim 契约（与现有 TU 逐条对齐）

生成物只含 `__global__` kernel，**不含** Launcher。需手写一个 shim，形如
（照 `dsv41_proj_mma_skel.cu:428,505` 的既有约定）：

```cpp
// tilelang_gen/proj_mrows_tl_launch.cu
#include "proj_mrows_tl.cu"

extern "C" int dsv41_tilelang_proj_mrows(
        const uint8_t* a, const float* a_scale, const uint8_t* w, const uint8_t* w_scale,
        float* out, int m, int n, int k, int out_stride, cudaStream_t s)
{
    // (1) 形状拒绝：任何 C 侧不能接受的形状 return 2（declined），
    //     让 Rust 回退到 legacy —— 与 dsv41_gemm_fp8_mrows_mma 同契约。
    if (m < 1 || m > 8 || n % 16 || k % 32) return 2;
    // (2) INIT-TIME 的 smem 属性（★ 绝不在 capture 内调用）。
    //     既有先例：dsv41_experts_mxf4.cu:4133/4137 的 "init-time, never capture-time"。
    static bool inited = false;
    if (!inited) { cudaFuncSetAttribute(tl_proj_mrows_kernel, ...); inited = true; }
    // (3) 网格/块/smem 来自生成物（编译期常量，写死在 shim 里并注释出处）。
    dim3 grid((n + TN - 1)/TN, (m + TM - 1)/TM), block(THREADS);
    tl_proj_mrows_kernel<<<grid, block, SMEM, s>>>(...);
    return 0;   // 0 = 已发射；2 = declined（回退）
}
```

**三条硬约束**（每条都有树内先例）：

| 约束 | 原因 | 先例 |
|---|---|---|
| `cudaFuncSetAttribute` 只在 INIT 期 | capture 期间调用会让 `cudaStreamEndCapture` 失败 | `dsv41_experts_mxf4.cu:4137,5743` |
| `<<<..., stream>>>` 用调用方传入的 `CuStream` | 必须进 capture / 与主 stream 同序 | `dsv41_proj_mma_skel.cu:505` |
| 形状不合规 `return 2`（不是 1/负数） | Rust 侧「decline 即未武装序列」契约 | `device.rs:4986`（mma 臂注释） |

### 3.4 `build.sh` 改动（精确到行）

```diff
 for f in "$DIR"/dsv41_kernels.cu ... "$DIR"/dsv41_proj_mma_skel.cu; do
     [ -f "$f" ] && SRCS+=("$f")
 done
+# --- TileLang 生成物（2026-09-13）：生成源码 + 手写 shim，各自 TU ---
+for f in "$DIR"/tilelang_gen/*_tl_launch.cu; do
+    [ -f "$f" ] && SRCS+=("$f")
+done
```
nvcc 命令行加两个 include（vendored，**不入 `3rdparty`**）：

```diff
 NVCC ... -gencode "arch=compute_${ARCH},code=sm_${ARCH}" \
+    -I "$DIR/tilelang_inc" \
     -DFERRITE_KERNEL_BUILD_ID=...
```
> ⚠️ `-I "$DIR/tilelang_inc"` 里含 `tl_templates/` 与 `cutlass/`、`cute/`，正好对上
> 生成码的 `#include <tl_templates/cuda/...>` 与 `common.h:43,46` 的 `<cute/...>` / `<cutlass/...>`。
> `CU_HASH`（`build.sh:136-138`）自动把新 TU 折进 `BUILD_ID` —— **same-source gate 免改**。

### 3.5 Rust 侧改动面（最小）

- `devrt.rs`：**零改动**。
- `device.rs`：`KernelDev`（`device.rs:1741` 附近）**新增一个 `Option<fn>` 字段**
  （`ko!(rt, "dsv41_tilelang_proj_mrows")`），外加一个 wrapper `gemm_fp8_mrows_tl(...)`——
  照抄 `gemm_fp8_mrows_mma`（`device.rs:4986`）的「符号缺失即返回 false」写法。
  MoE 侧同理加 `moe_grouped_tl` 符号。
- `chain_dev.rs`：两个 `OnceLock` gate 函数（§4.1）+ 分派链头部两个 `if`（§4.2）。

---

## §4 gate / precedence 设计 — deliverable ③

### 4.1 两个 gate（默认 OFF，OnceLock 读一次）

| gate | 作用域 | 语义 | 读法 | 先例 |
|---|---|---|---|---|
| `DSV41_GEMM_TILELANG` | **投影族 GEMM**（wq_a / wkv / wq_b / wo_b + indexer wq_b），**eager 与 verify 双侧同换** | 程序替换（非逐位） | 严格 `== "1"` | `DSV41_PROJ_MMA`（`chain_dev.rs:5779`）、`DSV41_SWAPAB` |
| `DSV41_MOE_TILELANG` | **MoE grouped 专家 GEMM**（gate/up + down） | 程序替换 | 严格 `== "1"` | `DSV41_EXPERT_TCGEN05_E4M3`（`chain_dev.rs:798`）、`DSV41_EXPERT_GROUPED`（`chain_dev.rs:945`） |

**「双侧同换」是 `DSV41_GEMM_TILELANG` 的定义的一部分**（不是实现细节）：eager(m=1) 与 verify(m≤6)
必须跑**同一个 TileLang 程序族**，这正是 `tensorcore-proj-design.md §2.4.1` 的
**(b′) program-consistent parity**——`row r of an M-row launch ≡ row r of the M=1 launch of THIS program`。
proj-mma 判死（`proj-mma-verdict.md`）的实锤原因就是它**只换了 verify 侧**（⇒ 两个不同程序 = WOB 形状
⇒ mean-k 2.240 → 0.020）。**TILELANG 臂从设计上就避开这个陷阱：两侧一起换。**

### 4.2 precedence（投影族）

在 `chain_dev.rs:5549` 的 `proj_mrows()` 头部，**TILELANG 排在第一**：

```
① DSV41_GEMM_TILELANG   ← 新增（最高）
② DSV41_PROJ_MMA        ← 既有（chain_dev.rs:5567）
③ DSV41_MPAR            ← 既有（在 C entry 内）
④ DSV41_MTILE           ← 既有
⑤ SIMT legacy           ← 兜底
```

**为什么 TILELANG 必须排在 PROJ_MMA 之前**：PROJ_MMA 是「单侧（verify）换程序」的已知危险形态，
TILELANG 是「双侧同换」。若允许二者共存并让 PROJ_MMA 先命中，就等于在 TILELANG 臂上
**又**把 verify 换成第三个程序 ⇒ 重新制造 WOB。⇒ **互斥且 TILELANG 优先**，并在两者同时武装时
打一条 one-shot 提示（"both armed, TILELANG wins, PROJ_MMA is inert for this process"），
因为「ON 的臂实际测的是老路」是本项目 #1 测量偏置陷阱。

### 4.3 precedence（MoE）

在 `moe_experts_grouped_gate_up`（`chain_dev.rs:14721`）/ `moe_rows` 头部：

```
① DSV41_MOE_TILELANG          ← 新增（最高）
② DSV41_EXPERT_TCGEN05_E4M3   ← 既有（+ DSV41_EXPERT_GROUPED 提供 grouped 布局）
③ per-(row, slot) legacy      ← 兜底
```
`DSV41_MOE_TILELANG` 复用 `DSV41_EXPERT_GROUPED` 产出的 grouped 布局（`moe_route_grouped`,
`chain_dev.rs:14596`）——**只换消费该布局的 GEMM**，不重建布局。二者互斥（同 §4.2 的理由），
one-shot 提示同款。

### 4.4 回退链（三条腿，缺一不可）

每次 arm 的判定是**三分支**，全部沿既有契约：

1. **gate OFF** → 下一级臂（无声，正常）。
2. **gate ON 但 `.so` 无符号**（陈旧构建）→ 打**一次** warning（"armed but the .so has no symbol;
   this run measures the OLD path"）→ 下一级臂。照抄
   `expert_grouped_skipped_note`（`chain_dev.rs:962`）的写法。
3. **gate ON、有符号、但形状 declined（C 侧 `return 2`）** → 无声回退到下一级臂
   （decline 是 per-call 的，合法形状会命中）。

**反面案例（必须避免的形态）**：`kernels/cuda/build.sh:87-99` 记录的
「symbol 不在 ∈ `.so` ⇒ `supports_*()` 为假 ⇒ 每次 A/B 静默测老路」——所以 vendored 生成物
**必须无条件编进 `.so`**（DEFAULT ON 的编译），只在 **运行时** 由 env gate 控制。


---

## §5 数值对照验收协议 — deliverable ④

### 5.1 为什么协议长这样（proj-mma 的先例是设计输入，不是脚注）

`proj-mma-verdict.md` 的实锤：**单侧换程序** ⇒ mean-k 2.240 → **0.020**（accept 崩 99%），
掉出接受带 ⇒ 直接弃臂。它**同时**给出两条可复用的判据来源：

1. **跨程序逐位不可能**（`tensorcore-proj-design.md §2.2` 已证）⇒ **禁止**把「与 SIMT 逐位」当验收门。
2. **唯一可做的 byte-compare 是「同一程序的跨 m 自洽」**（b′：`row r of M-row ≡ row r of M=1`），
   且 proj-mma 第一轮**从未跑过它**——所以那一轮连"程序自洽"的证据都没有。

⇒ **TileLang 的协议把「内部契约」列为第一道门（未过则弃臂，不看 e2e）**，第二道是数值容差，
第三道才是端到端文本。而且用户裁决的 **「老 eager 完整保留为对照」** 在这里恰好给了最强的外部对照：
我们有两个**各自内部自洽**的完整栈——`legacy 栈`（老 eager + 老 verify）与
`tilelang 栈`（TileLang eager + TileLang verify）——外部对照 = **两个栈的文本质量 + 吞吐**。

### 5.2 三层判据（顺序执行，前一层不过不看后一层）

#### 门 1 — 内部契约（program-consistent parity，**byte compare，必须先过**）

| 测什么 | 方法 | 判据 |
|---|---|---|
| m 自洽 | 同一 TileLang 程序：`m=1` launch 的 row 0 vs `m=6` launch 的 row 0（**激活相同**） | **逐位相同**（`memcmp`）。任一行不同 ⇒ 弃臂 |
| vs fp32 参考 | 同一输入喂 fp32 累加参考 | meanrel ≤ bf16 预算（实测参考值 **1.4e-3**，§1.3） |
| 重复性 | 同一次 launch 跑两遍 | 逐位相同（排除非确定性 atomic） |

> 这是**唯一**可做的逐位对照，也是「程序自洽」的唯一证据。它是 micro bench（单 kernel，无 e2e），
> 成本极低但**不可跳过**。

#### 门 2 — 数值容差（跨程序，禁止逐位）

TileLang MMA 与 SIMT 的差异是**跨程序**的（求和结合序硬件定义 + scale 施加位置不同，
`tensorcore-proj-design.md §2.2`）。⇒ 判「分布」不判「逐位」：

| 量 | 判据 |
|---|---|
| 相对误差分布 | `maxrel` / `meanrel` / `p99`，与门 1 的 fp32 参考同口径 |
| argmax 一致率（**近 tie 敏感**） | 同一激活下，legacy eager 的 logits 与 TileLang eager 的 logits，在 **129280 路 vocab** 上的 argmax 一致率。⚠️ 先例：commit `c42ab14` 证明 **f32 非结合性足以让近 tie argmax 翻转**——所以这一项**必须实测**，不能论证 |

#### 门 3 — 端到端文本红线（双侧同换后，对照 legacy 栈）

以基线为对照（`400-final-frontier-analysis.md` / `NEXT-SESSION-HANDOVER.md` 的既有口径）：

| 判据 | 红线 | 出处 |
|---|---|---|
| 出师表 | **零拉丁** + 四段文本逐字 | `NEXT-SESSION-HANDOVER.md:82` |
| 计数 | 1–200 数字顺序正确 | `400-final-frontier-analysis.md:100,295` |
| `k_acc` | 逐位不变（出师表序列 `4 0 0 0 3 0 1 1 0 …`） | 同上 |
| **mean-k** | **落在接受带 2–3**（clean 基线 **2.240**） | `proj-mma-verdict.md §1.1` |
| 错误 | `faults=0` + `err=0` | `NEXT-SESSION-HANDOVER.md:82` |
| 性能 | `[dsv41] step pos=N` 行的 step_ms（**禁段平均**） | `NEXT-SESSION-HANDOVER.md` |
| **kernel 名确认** | 日志 / trace 必须看到 **TileLang kernel 真的跑了** | `400-final-frontier-analysis.md:295`（#1 陷阱） |

> **mean-k 的读法（关键）**：proj-mma 的 2.240→0.020 是**单侧换程序**造成的 WOB。
> TILELANG 是**双侧同换** ⇒ 理论上 accept 关系保持（draft 与 verify 是同一程序族，
> 近 tie 的"谁赢"关系不变）。但这只**降低**风险，**不取消**门 3 —— `k_acc`/mean-k 仍必须实测落带。

### 5.3 对照矩阵（一次跑齐，避免段平均）

| 栈 | eager (m=1) | verify (m≤6) | 用途 |
|---|---|---|---|
| **L（legacy）** | 老 SIMT `gemm_fp8_mrows` | 老 SIMT | **基线**：正确性 + 性能 + 文本 |
| **T（tilelang）** | TileLang proj kernel | **同一** TileLang proj kernel | 被测臂 |
| 交错跑 | L → T → L → T | | 消除机器漂移（本项目 A/B 纪律） |

一次进程只 allow 一个臂（gate 读 OnceLock；混跑 = 测不到东西）。

### 5.4 弃臂条件（三条任一命中 ⇒ 弃）

1. 门 1 自洽失败（同一程序不同 m 不一致）；
2. 门 2 的 argmax 一致率低于 legacy 自身的重跑一致率；
3. 门 3 出师表/计数红线破 or mean-k 掉出 2–3。

---

## §6 实施步骤清单 — deliverable ⑤

### 6.1 阶段 0：vendoring（无 GPU，可立即做）

| # | 步骤 | 验收 |
|---|---|---|
| 0.1 | 从 `tilelang-0.1.14` 取 `src/tl_templates/{cuda,common*}` → `kernels/cuda/tilelang_inc/tl_templates/` | 46 文件 / 1.1 MB |
| 0.2 | 取 `3rdparty/cutlass/include/{cutlass,cute}` 的**传递闭包子集**（实测 33 头；用 `nvcc -M` 导出清单） | 27 MB → 目标裁剪到 ≤5 MB |
| 0.3 | 写 `tilelang_inc/VERSION`（tilelang 0.1.14 + cutlass commit） | 版本可追溯 |
| 0.4 | 把 `-I "$DIR/tilelang_inc"` 加进 `build.sh` 的 nvcc 行 | `bash build.sh 103a` 仍 EXIT=0（此时无新 TU） |

> ⚠️ 0.2 是唯一有裁剪风险的一步。**保守起点**：先整体 vendor `cutlass/include`（27 MB，一次性），
> 确保 EXIT=0；裁剪是后续优化（§7.3）。**不裁剪正确性优先**。

### 6.2 阶段 1：接线骨架（依赖 peer 交付 1 个 kernel）

| # | 步骤 | 验收 |
|---|---|---|
| 1.1 | 收 `tilelang-proj-proto` 的 `get_kernel_source()` 输出 → `tilelang_gen/proj_mrows_tl.cu` | 独立 `nvcc -cubin` EXIT=0 |
| 1.2 | 写 `proj_mrows_tl_launch.cu`（§3.3 的 shim 契约） | EXIT=0，符号 `dsv41_tilelang_proj_mrows` 可见 |
| 1.3 | `build.sh` `SRCS += tilelang_gen/*_launch.cu` | `.so` 里 `nm -D` 有符号；`BUILD_ID` 变化 |
| 1.4 | `device.rs` 加 `gemm_fp8_mrows_tl: Option<fn>` + wrapper | 陈旧 `.so` 时 `supports_*()` 为假 |
| 1.5 | `chain_dev.rs` 加 gate `DSV41_GEMM_TILELANG` + `proj_mrows()` 头部第一分支 | 默认 OFF，行为逐位不变（回归） |
| 1.6 | one-shot armed-but-inert 提示（三分支回退，§4.4） | 与 `expert_grouped_skipped_note` 同形 |

### 6.3 阶段 2：门 1 微测（需 GPU，单 kernel 无 e2e）

| # | 步骤 | 验收 |
|---|---|---|
| 2.1 | 写 `kernels/cuda/tests_tilelang_parity.cu`（m=1 vs m=6 逐位 + fp32 参考） | 门 1 全过 |
| 2.2 | argmax 一致率 micro bench（legacy vs tilelang 同激活） | 记录分布，作为门 2 基线 |

### 6.4 阶段 3：e2e 双臂（需 GPU；跑 §5.3 矩阵）

| # | 步骤 | 负责 |
|---|---|---|
| 3.1 | `scripts/dsv41_tilelang_ab.sh`（L/T 交错 + 文本红线抓取） | 工部 |
| 3.2 | `DSV41_GEMM_TILELANG=1` 跑门 3 全红线 | tester |
| 3.3 | MoE：收 `tilelang-moe-grouped` 产物，重复 1.1–1.6 / 2.x / 3.x（gate `DSV41_MOE_TILELANG`） | 工部 |
| 3.4 | 判决：过 / 弃（§5.4） | 中书省 + 门下省 |

### 6.5 RACI（§角色分工见文末）

- **吏部**（代码质量）：shim / gate 命名与既有约定一致性审查；
- **户部**（性能/资源）：vendoring 足迹、smem/寄存器/occupancy、step_ms；
- **礼部**（文档）：`PROVENANCE.md`、build.sh 注释、README；
- **兵部**（安全）：**生成代码的供应链审计**——vendored 头与生成物的 sha256 记录、
  "生成物禁止手改"纪律（§7.1）；
- **刑部**（Bug）：门 1 自洽、边界形状（m/n/k 不合规的 decline）；
- **工部**（实现）：§3 全部接线 + shim + A/B 脚本。

---

## §7 风险 / 待定

| # | 风险 | 应对 |
|---|---|---|
| 7.1 | 生成代码可读性差、被误当手写代码去改 | `PROVENANCE.md` + 每份生成物头部固定注释（"GENERATED — do not edit; regenerate via <script>"）+ CI 校验生成物 sha256 |
| 7.2 | TileLang 版本漂移（0.1.14 → 0.2.x 产物形态变） | 生成物**冻结** + vendored 头**冻结**；升级 = 重新生成 + 全量回归（一次显式动作，非隐式） |
| 7.3 | cutlass 子集裁剪漏头 | 阶段 0.2 保守起步（整包），裁剪单独一轮 + `nvcc -M` 清单比对 |
| 7.4 | 门 2 argmax 一致率低（近 tie 翻转） | **这正是门 2 存在的意义**：实测决定，不做论证；低到影响 accept ⇒ 弃臂（照 proj-mma 先例） |
| 7.5 | MoE grouped 布局契约不匹配（peer 产物 vs `dsv41_route_group` 的 `[g]` 行序） | §8 冻结契约：grouped 布局由 ferrite 侧 `moe_route_grouped` 提供，TileLang kernel **只消费**，不重建 |
| 7.6 | 双侧同换后 accept 仍崩 | 门 3 兜底；记录 mean-k，照接受带判决 |
| 7.7 | fp8 block-scale 语义（激活 f32 scale vs TileLang ue8m0） | 见 §8 的待冻结项——**这是 peer 交付前必须对齐的头号数值契约** |

---

## §8 给两个 peer 的交付契约（接口冻结清单）

`tilelang-proj-proto` / `tilelang-moe-grouped` 交付时必须**逐项冻结**（否则接线返工）：

1. **生成配置**：`pass_configs={TL_DISABLE_TMA_LOWER: True, TL_DISABLE_WARP_SPECIALIZED: True}`
   —— 保证**裸指针 ABI**（§1.3 实验 B）。若 peer 需要 TMA 性能，则必须同时交付 host 侧
   `cuTensorMapEncodeTiled` 的 shim（工作量不同，**提前定**）。
2. **kernel 签名与语义**（一次写死）：张量次序、dtype、`__restrict__`、grid/block 约定、
   smem 字节数（shim 要用）。
3. **scale 契约**：激活 scale 的 dtype/布局（`a_scale[m, k/32]` f32？ue8m0？）、权重 scale 同问
   —— **与 `tensorcore-proj-design.md §1.4` 的 f32 激活 scale 是否一致**，不一致则需 epilogue 适配。
4. **形状域**：m∈[1,8]、n/k 的对齐要求（决定 shim 的 `return 2` 条件）。
5. **数值画像**：`get_kernel_source()` 产物 + `nvcc -cubin` EXIT=0 + m=1 vs m=6 逐位自洽的
   **micro bench 证据**（门 1 的 peer 侧初判）。
6. **产物指纹**：`manifest.json`（TileLang 自己记的 sha256）随源码一起交付。

---

## 附：一句话总结

TileLang 的产物**天然适合源码合入**——`get_kernel_source()` 给自包含 CUDA 源码，
`TL_DISABLE_TMA_LOWER=1` 给裸指针 ABI，nvcc 独立编译 EXIT=0 且数值正确（§1.3 实测）。
ferrite 侧只需一个**手写 launcher shim** + `build.sh` 两行（SRCS 与 `-I`）+
`device.rs` 一个 `Option<fn>` + `chain_dev.rs` 两个 gate 头部 `if`；
**devrt / dlopen / build-stamp 体系零改动**。两条 gate 排在分派链**最前**，
`DSV41_GEMM_TILELANG` 的定义就是「eager 与 verify 双侧同换」——从设计上避开 proj-mma 的
「单侧换程序 ⇒ mean-k 2.240→0.020」死因；验收先把 **program-consistent parity**（逐位）当第一道门，
老 eager 全栈保留为外部文本对照。
