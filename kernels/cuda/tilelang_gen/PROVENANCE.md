# tilelang_gen/ — TileLang 生成物与 launcher shim 的出处

> 工部 · 2026-09-13 · 第一阶段（wkv 一个形状走通全链）。
> 上位设计：`docs/agent/tilelang-integration-design.md`（路线 (a) 源码合入 + 裸指针 ABI）。
> 原型：`kernels/tilelang/proj_fp8_tilelang.py`、`docs/agent/tilelang-proj-proto.md`。

---

## 1. 目录内容

| 文件 | 性质 | 说明 |
|---|---|---|
| `wkv_partial_tl.cu` | **生成物（禁止手改）** | K-split 分片内核，TileLang `get_kernel_source()` 原样 dump |
| `wkv_reduce_tl.cu` | **生成物（禁止手改）** | 确定性归约内核，同上 |
| `wkv_tl_config.txt` | 生成物 | 冻结几何（grid/block/smem/ks/bN/threads）+ 两个内核的签名 |
| `wkv_shim.cu` | **手写** | launcher shim（`dsv41_gemm_fp8_tilelang_wkv`），`#include` 两个生成物 |
| `moe_up_tl.cu` | **生成物（禁止手改）** | up（gate‖up）grouped GEMM，见 §7 |
| `moe_dn_tl.cu` | **生成物（禁止手改）** | down grouped GEMM，见 §7 |
| `moe_tl_config.txt` | 生成物 | MoE 冻结几何（grid/block/smem/SEG_CAP）+ 两个内核的签名 |
| `moe_bf16_shim.cu` | **手写** | MoE launcher shim + gather/scatter/dequant（3 个导出符号），见 §7 |

生成物头部各有一行 `GENERATED — do not edit` banner（banner 之外逐字节等于 dump）。
每个 dump 都定义 `extern "C" __global__ void main_kernel(...)`，所以每个 shim 用
`#define main_kernel ...` 改名后 include；**build.sh 只把 `*_shim.cu` 当一个 TU**。

---

## 2. 重生成（唯一合法路径）

远端 B300（sm_103a，tilelang 0.1.14）：

```bash
scp kernels/tilelang/gen_wkv_aot.py ubuntu@43.202.208.136:~/tl_proj/
ssh ubuntu@43.202.208.136 \
  'cd ~/tl_proj && /opt/dlami/nvme/dsv41_venv/bin/python gen_wkv_aot.py aot_gen'
scp ubuntu@43.202.208.136:'~/tl_proj/aot_gen/*' kernels/cuda/tilelang_gen/
# 然后给两份 .cu 加 banner（banner 是唯一的、可复现的本地改动）
```

生成参数（冻结）：

| 项 | 值 |
|---|---|
| 形状 | wkv `n=512, k=5120` |
| 几何 | `bN=128, ks=8, threads=128, num_stages=3`；reduce `bN=256, threads=256` |
| 数值路线 | **route A**：原生 fp8 mma + per-32-K-block scale（`a_scale` f32 × `w_scale` ue8m0）|
| ABI | 裸指针：`pass_configs={TL_DISABLE_TMA_LOWER:1, TL_DISABLE_WARP_SPECIALIZED:1}` |
| TileLang | 0.1.14（`~/.local/lib/python3.12/site-packages/tilelang`）|
| nvcc | `/usr/local/cuda-13.2/bin/nvcc`（V13.2.51）|

### 2.1 与原型的两处生成差异（动机：让 shim 直接吃 ferrite 的 `[m, k]` 激活）

1. **`m` 是运行期标量 + 激活 staging 带 `if i < m` 谓词**。原型把激活 pad 到 16 行；
   这里行 ≥ m 的 `A_sh` 写 0、**不读** `a` 的那一行 ⇒ 不需要 host 侧 16 行 pad 缓冲，
   且不会越界读。TileLang 会为此在写 `A_sh` 前后各插一个 `__syncthreads()`（`A_sh`
   退化为单缓冲 —— 安全，读到的仍是本迭代写入的值）。
2. **归约的 store 带同样的 `if row < m` 谓词** ⇒ 可直接写 ferrite 的 `out`，前提是
   `out` 的行 stride 恰为 `n`（shim 的形状门因此要求 `out_stride == n`）。
   ⇒ 不需要 `[16, n]` 输出 staging，也不需要 m 行回拷。

⇒ **每次调用 = 2 次 launch**（partial + reduce）+ 常驻 scratch `P[8][16][512] f32`（256 KiB），
不再有 pad 拷贝 / 回拷。

### 2.2 真实 dump 的 sha256（banner 之前，即 TileLang 原样输出）

```
9f3176338fa1e297f7daf80e17fa7ce1b3bef8cd1e1468d22a2668aefb925c97  wkv_partial_tl.cu (raw, 16957 B)
122f03e579d7a48f183565181f0b0619c392e57de2704cd8b547a643157495a5  wkv_reduce_tl.cu  (raw,  5478 B)
```

（仓库内文件 = banner + raw。raw 的 sha256 是审计基准：重生成后应与上表逐位一致。）

---

## 3. vendored 头 `kernels/cuda/tilelang_inc/`

| 子树 | 来源 | 文件数 |
|---|---|---|
| `tl_templates/` | `tilelang-0.1.14/src/tl_templates/` **全量** | 46（1.1 MB）|
| `cute/` + `cutlass/` | `tilelang-0.1.14/3rdparty/cutlass/include/` 的**传递闭包** | 37（1.6 MB）|

- ⚠️ **2026-09-13 闭包扩展（+2）**：第五阶段的 `moe_bs_up_tl.cu` 头部 `tl_templates/cuda/intrin.h`
  在 `#if __CUDA_ARCH_LIST__ >= 900` 下拉进 `cute/arch/mma_sm90_gmma.hpp`（它又拉
  `cutlass/arch/synclog.hpp`），两者不在第一阶段闭包内 ⇒ 只补这两个（各带 sha256 核对，
  取自同一 venv `3rdparty/cutlass/include/`）。补法仍是 `nvcc -M` 实测闭包。

- 生成码 `#include <tl_templates/cuda/...>`（8 个头）与 `common.h` 里的 `<cute/...>` /
  `<cutlass/...>`；一个 `-I kernels/cuda/tilelang_inc` 同时覆盖三个前缀。
- **cutlass 裁剪依据**：对两份生成物跑 `nvcc -M` 取传递闭包，只留实际被 include 的头
  （35/… 个），从整包 27 MB 降到 **0.6 MB**。裁剪清单可重放（见 §2 脚本 + `nvcc -M`）。
  ⚠️ 设计文档 §6.1 的保守起点是整包 vendor；这里用了实测闭包，因为闭包两次
  compile-only 都 EXIT=0（§4）。若将来换形状/换 pass_configs 触发新的 cutlass 头，
  按 `nvcc -M` 结果补进闭包即可。
- **无既有 TU 引用这些前缀**（`grep -rn '#include <cute/\|<cutlass/\|<tl_templates/' kernels/cuda/*.cu`
  为空）⇒ 新增 `-I` 不改变任何既有 TU 的 include 解析。

---

## 4. 验收证据（rebuild 记录）

| 检查 | 结果 |
|---|---|
| `nvcc -arch=sm_103a -cubin`（两生成物，仅 `-I tilelang_inc`） | EXIT=0 |
| `nvcc … -shared -fPIC`（`wkv_shim.cu`） | EXIT=0；`nm -D` 有 `dsv41_gemm_fp8_tilelang_wkv` |
| ptxas | 见 §5 微基准记录 |

---

## 5. 第二形状的扩展路径

wkv 走通后，把第二个形状（建议 `wq_a`：`n=1280, k=5120`，n 大 ⇒ 块数够）接上：

1. **生成**：在 `gen_wkv_aot.py` 里加/改一组 `(N, K, bN, ks)`（wq_a 可先试 `ks=1`
   —— `n/128 = 10` 个块，配 reduce 仍是 2 launch；或 `ks=4`/`8` 填满 SM），dump 出
   `<name>_partial_tl.cu` / `<name>_reduce_tl.cu`，落进本目录。
2. **shim**：复制 `wkv_shim.cu` 为 `<name>_shim.cu`，改 ①冻结几何常量、②形状门
   （`n == <N> && k == <K>`）、③`#include` 的两份生成物名与 `#define` 重命名前缀
   （`<name>_tl_partial_kernel`）、④导出符号 `dsv41_gemm_fp8_tilelang_<name>`。
   K-split 的 grid 由冻结几何直接给出（`(N/bN, ks)`），scratch 尺寸随 `<name>` 独立。
3. **build.sh**：把 `tilelang_gen/wkv_shim.cu` 的循环换成 `tilelang_gen/*_shim.cu`
   （生成物仍不作为独立 TU —— 它们都有 `main_kernel`，必须经 shim rename-include）。
4. **device.rs**：加一个 `Option<fn>` 字段 + `ko!(rt, "dsv41_gemm_fp8_tilelang_<name>")`
   + 一个 wrapper（照抄 `gemm_fp8_tilelang_wkv`）。
5. **chain_dev.rs**：`gemm_tilelang_wkv()` 的点位不用动 —— 分派的形状门在 C 侧
   （新形状在 wkv 的点位上会 decline）；新增形状在它自己的调用点（wq_a 在
   `proj_mrows`/`lin`）挂同样的「gate 命中 → 试 TileLang → decline 回退」臂。
6. **第二形状的 KS 规则**：镜 `dsv41_proj_mma_skel.cu:389` 的 `proj_mma_ks_for`
   —— 目标是 `(n/bN)*ks ≈ SM 数`，且 `(k/32) % ks == 0`。第一阶段把 wkv 钉死在
   `ks=8`（4×8 = 32 块）是同一个判据的手工解；多形状后应当做成生成期参数（AOT 时
   就 resolve，不留运行期分支）。

⚠️ **不做**的扩展：swapAB（权重在 M、激活在 N=8，75% vs 37.5% 利用率）。第一阶段
`pad-16` 形态已实测 M6/M1 = 1.00，没有任何判据要求 swapAB；它是吃算力的后续优化，
换形态要重生成全部形状 + 重跑门 1/2，独立一轮。

### 5.1 head（H3，bf16 稠密 GEMM）—— `gen_head_aot.py` 的生成 + ABI 复核

`head_bf16_shim.cu` 的 `__has_include` 守卫（`:102-108`）包含的正是本节两份生成物名
（`head_partial_tl.cu` / `head_reduce_tl.cu`，与生成器落在 `<outdir>/` 的文件名逐字相同
⇒ **文件名匹配**；缺席时该 TU 编成空单元、`.so` 无 `dsv41_head_bf16_tilelang`、
`supports_head_tilelang()` false，绝不静默测老路）。

**生成（主 agent，远端 B300 —— GPU/远端 python 操作，接线 agent 只写清单）**：

```bash
scp kernels/tilelang/gen_head_aot.py ubuntu@43.202.208.136:~/tl_bs/
ssh ubuntu@43.202.208.136 \
  'cd ~/tl_bs && mkdir -p aot_gen && /opt/dlami/nvme/dsv41_venv/bin/python gen_head_aot.py aot_gen'
# ↑ mkdir 是必需的：脚本直接 open(f"{outdir}/…")，目录不存在即 FileNotFoundError。
scp ubuntu@43.202.208.136:'~/tl_bs/aot_gen/head_partial_tl.cu' \
    ubuntu@43.202.208.136:'~/tl_bs/aot_gen/head_reduce_tl.cu' \
    ubuntu@43.202.208.136:'~/tl_bs/aot_gen/head_tl_config.txt' \
    kernels/cuda/tilelang_gen/
```

（`gen_head_aot.py` 自含：只 import `hashlib/sys/tilelang/tilelang.language`，**不需要**
`head_bf16_tilelang.py`；也不依赖 `vendor/apply_tilelang_patch.py`——那是
tcgen05-blockscaled（e4m3）臂的补丁，head 是普通 bf16 mma。脚本自己给两份 `.cu` 加
banner（banner 之外逐字节等于 dump），所以回传后**不需要**再手工贴 banner。）

**产出的期望值**（与 `head_bf16_shim.cu` 的冻结几何逐项对齐；任何不符 ⇒ 先停下对齐）：

| 项 | 期望 | 出处 |
|---|---|---|
| config 行 | `N=16160 K=5120 NPAD=16256 BN=128 KS=8 BK=64 NS=3 THREADS=128 MPAD=16 RED_BN=256 RED_THREADS=256` | `gen_head_aot.py:67-74` |
| partial grid / block | `(127, 8)` / `128` | `ceildiv(N,BN)=127`，与 `NPAD/BN=127` 同值 |
| partial smem | `55296` B | `NS*(MPAD+BN)*BK*2`（> 48 KiB ⇒ shim 的 `SetAttribute` 是必要条件） |
| reduce grid / block | `64` / `256` | `ceildiv(16160,256)=64`（⚠️ **不是** `NPAD/256=63`） |
| P / Xb scratch | `8.30 MiB` / `160 KiB` | `KS*MPAD*NPAD*4` / `MPAD*K*2` |

**ABI（形参序）—— 不许猜，且机制已知**。TileLang 在
`src/transform/split_host_device.cc` 的 `SortDeviceParams()` 里排 device 形参：

```cpp
sort_key = { !var->dtype.is_handle(),  var->name_hint };   // 指针在前，再按名 ASCII 升序；标量最后
```

⇒ head 的张量名是 `X` / `W` / `P`（`gen_head_aot.py:93-95`），故

| dump | 形参序 | shim 的调用 |
|---|---|---|
| `head_partial_tl.cu` | `(P f32*, W bf16*, X bf16*)` | `(g_part, (const bfloat16_t*)w, (const bfloat16_t*)g_xb)` |
| `head_reduce_tl.cu` | `(C f32*, P f32*, int m)` | `(out, g_part, m)` |

（旧版 shim 写的 `(X, P, W)` / `(P, C, m)` 是从 wkv 的 `(A, ASC, P, W, WSC)` 外推的
"P 前移到 W 之前"；wkv 的 `A` 恰好也是字母序最小项，那条证据分不出「原序首位」与
「按名排序」。依据 §9.3.1：形参序是 lowering 的产物，**只有 dump 说了算**。
⚠️ reduce 的两个指针都是 `float*`，**错位编译器不报错**：会把 P 当归约输出写回、
logits 永不更新——所以下面 ① 是硬闸，不是建议。）

**复核命令（生成后、重建前，硬闸）**：

```bash
# ① 两份签名行与实际 dump 逐字比对（config 的这两段就是生成器从 dump 里抓的）
sed -n '/--- head_partial_tl.cu signature ---/,/^$/p' kernels/cuda/tilelang_gen/head_tl_config.txt
sed -n '/--- head_reduce_tl.cu signature ---/,/^$/p'  kernels/cuda/tilelang_gen/head_tl_config.txt
grep -m1 "__launch_bounds__" kernels/cuda/tilelang_gen/head_partial_tl.cu
grep -m1 "__launch_bounds__" kernels/cuda/tilelang_gen/head_reduce_tl.cu
# 判据：partial = main_kernel(float* P, const bfloat16_t* W, const bfloat16_t* X)
#       reduce  = main_kernel(float* C, const float* P, int m)
# ② grid / smem（reduce_grid 必须 = 64；shim 的 kTLSmem 必须 == partial_smem_bytes）
grep -E "^(partial|reduce)_(grid|block|smem)" kernels/cuda/tilelang_gen/head_tl_config.txt
```

**编译验证（远端 compile-only，无 GPU）**：

```bash
ssh ubuntu@43.202.208.136 'cd ~/ferrite && nvcc -gencode arch=compute_103a,code=sm_103a \
  -O3 -std=c++17 --use_fast_math -c -I kernels/cuda/tilelang_inc \
  kernels/cuda/tilelang_gen/head_bf16_shim.cu -o /tmp/head_shim.o && \
  nm /tmp/head_shim.o | grep -E "head_tl_(partial|reduce)_kernel|dsv41_head_bf16_tilelang"'
```

**整编 + 符号三证**（`bash kernels/cuda/build.sh 103a` 之后，见 `c4-head-ab-manual.md` §4.0）：

```bash
nm -D kernels/cuda/libferrite_kernels.so | grep -c dsv41_head_bf16_tilelang   # ≥ 1（=0 ⇒ Arm 2 无意义）
```


---

## 6. 纪律

- 生成物**禁止手改**；改动一律走 §2 的重生成 + 全量回归（一次显式动作）。
- `tilelang_inc/` 与生成物**一起冻结、一起 commit**：TileLang 0.1.14 → 0.2.x 的产物形态
  会变，升级 = 重生成 + 重跑微基准（门 1）+ 重新 vendor。
- 生成物进 `build.sh` 的 `SRCS` ⇒ 自动进 `CU_HASH` ⇒ 进 `BUILD_ID`（same-source gate
  免改一行）。

---

## 7. 第二阶段：MoE grouped GEMM（bf16 臂）

> 原型与全部结论：`docs/agent/tilelang-moe-grouped.md`（GPU 实测 bf16 臂
> **70.0µs/层** = up 45.5 + dn 24.5 = SIMT 250µs 的 **28.0%**，判据 <40% 通过）。
> 接线设计与验证手册：`docs/agent/tilelang-moe-wiring.md`。

### 7.1 生成（唯一合法路径）

```bash
scp kernels/tilelang/gen_moe_aot.py ubuntu@43.202.208.136:~/tl_moe/
ssh ubuntu@43.202.208.136 \
  'cd ~/tl_moe && /opt/dlami/nvme/dsv41_venv/bin/python gen_moe_aot.py aot_gen'
scp ubuntu@43.202.208.136:'~/tl_moe/aot_gen/moe_up_tl.cu' \
                       :'~/tl_moe/aot_gen/moe_dn_tl.cu' \
                       :'~/tl_moe/aot_gen/moe_tl_config.txt' kernels/cuda/tilelang_gen/
# 再给两份 .cu 加 banner（banner 是唯一的、可复现的本地改动）
```

| 项 | 值 |
|---|---|
| 形状 | up `N=640 K=5120`；dn `N=5120 K=320`；`E=384`、`BM=16`、`NSEG=SEG_CAP=36` |
| 几何 | up `BN=256 BK=64 th=256 stg=3`（grid 3×36，smem 104448）；dn `BN=512 BK=64 th=256 stg=2`（grid 10×36，smem 135168）|
| 数值路线 | bf16 权重 × bf16 激活 → `mma.sync.m16n8k16` → fp32 累加（原型 §3 实测最优臂）|
| ABI | **裸指针**（`TL_DISABLE_TMA_LOWER:1, TL_DISABLE_WARP_SPECIALIZED:1`，与第一阶段同路线）|
| 签名 | `main_kernel(const bfloat16_t* A, float* C, const int* Eid, const bfloat16_t* W)` |
| TileLang | 0.1.14；nvcc CUDA 13.2，`-arch=sm_103a` |

**与原型 `moe_grouped_proto.py::k_bf16` 的两处生成差异**（都是为了接进 ferrite）：

1. **裸指针 ABI**。原型的默认 lowering 产出 `CUtensorMap` + TMA + warp specialization
   （原型 §6 的 89 行 device 形态）；那要求 host 用 driver API 的
   `cuTensorMapEncodeTiled` 造描述符——`build.sh` 不链 `-lcuda`，且每层每方向每次
   launch 都要重编码。裸指针 ABI 让 shim 直接吃 ferrite 的 device 指针，与第一阶段的
   wkv 同一取舍。
2. **`NSEG` 烘成上界 `SEG_CAP=36`**（`VERIFY_ROWS(6) × TOPK_MAX(6)`）。TileLang 的 grid
   维度是编译期常量，而真实 `nseg` 随路由逐位变化（本形状 35）。host 的
   `order`/`counts`/`eid` 表在 `nseg` 之后**零填充**（见 7.3），pad 段的输出不被
   scatter，代价是 `grid.y` 从 35 抬到 36（≈3% 空转）。

### 7.2 dump 的审计基准（banner 之前的 sha256）

```
7499f395cbc1e585cb8cb2838a062623b14f752079e43dcc7fbaac60cdd15d3c  moe_up_tl.cu (raw, 10766 B)
ec5d36b79bc31c7369c3a8085818585d8869c922fdbd55c336938822a6822dc6  moe_dn_tl.cu (raw,  7517 B)
```

### 7.3 shim（`moe_bf16_shim.cu`）：三个导出符号 + rc 契约

| 符号 | 作用 |
|---|---|
| `dsv41_moe_tilelang_gate_up_bf16` | gather(f32→bf16) + up grouped MMA + scatter（**RAW gate‖up**）|
| `dsv41_moe_tilelang_down_bf16` | gather + down grouped MMA + scatter（per-slot partial）|
| `dsv41_moe_fp4_to_bf16` | 加载期 fp4(e2m1+ue8m0) → bf16 副本 |

rc 契约与 wkv shim / `dsv41_proj_mma_skel.cu` 一致：`0` 已发射 / `2` DECLINED / 其它 =
`cudaGetLastError()`。Rust 只在 `rc == 2` 回退。

**内核不做 gather/mask/atomic**（原型 §2.2）：gather/scatter 在 shim 里，输入输出契约：

- `A`：`[SEG_CAP*BM, K]` bf16（host 已 gather + 每段 pad 到 16 行；pad 行写 0）
- `W`：`[E=384, N, K]` bf16，**K 连续**；up 的 N 前半 = gate(w1)、后半 = up(w3)
- `Eid`：`[SEG_CAP]` i32；`C`：`[SEG_CAP*BM, N]` f32
- `order[SEG_CAP*BM]`：该 (段, 段内行) 的 assignment 下标（`row*topk + slot`，pad = -1）
- `counts[SEG_CAP]` / `eid[SEG_CAP]`：段的 live 行数 / expert id，**`nseg` 之后零填充**

⚠️ **INIT 的 SetAttribute 是必要条件**（up 104448 B / dn 135168 B 均 > 48 KiB 默认上限）——
与 wkv 的「契约对齐」不同，这里不设就 launch 失败。

### 7.4 唯一的运行期限制：EAGER（非 capture）

`moe_align` 在 **host** 侧（`chain_dev.rs::moe_align_host`，纯函数：稳定按 expert 排序 +
排他前缀和，无 atomic、不依赖 block 调度 ⇒ 同一路由表逐位可复现），而 `route_idx*` 是
`route_topk` 写在 **device** 上的，所以这个臂需要一个 **D2H 回读 + 三个小 H2D 上行**，
在 CUDA-graph capture 内非法。⇒ 调用方在 `dev.capturing()` 时不派遣（decline + 一次性
提示）。**把 `nseg` 摊到 GPU 侧**是明确的后续项（原型 §8-3），届时才能进 graph。

### 7.5 显存预算（⚠️ 需拥有者拍板）

bf16 路径的前置是**加载期把专家权重 dequant 成 bf16 常驻**
（`DSV41_MOE_BF16_DEQUANT=1` → `dsv41_moe_fp4_to_bf16`），代价是专家权重的 fp4 池 ×4。

按 **ferrite 自己的加载布局**算（`load.rs::load_expert_pool` **不做 expert 并行**：每个
rank 持有全部 384 个专家，只对 `inter` 做 TP 切分）：

| 量 | 值 |
|---|---|
| per rank fp4 专家权重 | 40 层 × 384 专家 × (2·320·5120 + 5120·320) × 0.5 B ≈ **35–37 GiB** |
| bf16 副本 | ×4 ≈ **141–148 GiB** |
| **增量** | **≈ +105–113 GiB/rank** |

⚠️ **与原型文档 §4.3 的「+15GB/rank」不一致，且必须由拥有者拍板。** 那 15 GB 的算法是
「fp4 40GB → bf16 160GB，TP8 下 5GB→20GB/rank」——它隐含了 **8 路专家分片**（每 rank 48
个专家）。ferrite 的 loader 不做专家并行，所以每 rank 的增量是上述的 **~7 倍**。两种口径
的分母不同，部署前必须确认走哪一种：

- 若接受 +105~113 GiB/rank ⇒ bf16 臂（本 shim）就是最终形态；
- 若不可接受 ⇒ **唯一替代是 tcgen05 blockscaled 原生 fp4 MMA**
  （`T.tcgen05_gemm_blockscaled` + `T.make_blockscaled_gemm_layout` + `T.alloc_tmem` +
  mbarrier；原型 §4.3-2 / §8-1），预期 up 159.8µs → 15–25µs 量级；另一路 subagent 在探索。
  ⚠️ **不要**退回「读 fp4 → 内核里展开成 bf16 → MMA」：那条路已判死（去掉 dequant ALU
  仍 102.8µs > bf16 45.5µs，原型 §4.2）。

### 7.6 与 fp4 ILV 布局互斥

`DSV41_MOE_BF16_DEQUANT` 要求 gate/up 平面是**普通的连续 `[inter_local, dim]` 块**。在
`DSV41_EXPERT_ILV` 下 w1 的 view 别名那个交错的加倍区域，dequant 会把交错字节当成普通的
gate 平面读——**静默错值**。⇒ 两者互斥，`load.rs::ilv_ok` 在 bf16 dequant 打开时拒绝
交错（`load_expert_pool` 里也有同样的守卫，防漂移）。

### 7.7 验收证据（本阶段）

| 检查 | 结果 |
|---|---|
| `nvcc -O3 -std=c++17 -shared -fPIC -arch=sm_103a -I tilelang_inc`（`moe_bf16_shim.cu`）| **EXIT=0** |
| `nm -D` | `dsv41_moe_tilelang_gate_up_bf16` / `dsv41_moe_tilelang_down_bf16` / `dsv41_moe_fp4_to_bf16` 三个 T 符号 |
| `cargo check -p ferrite-models` / `--workspace` | **EXIT=0** |
| GPU e2e + parity | ⏳ 见 `docs/agent/tilelang-moe-wiring.md` §5（主 agent 职责）|

---

## 8. 第三阶段：五形状投影批量接线（wq_a / wq_b / wo_b + wo_a 分组）

> 前置：`docs/agent/tilelang-proj-phase2.md`（五形状原型的全部 GPU 数字，M6/M1 = 0.99–1.01）。
> 接线验证手册：`docs/agent/tilelang-proj-phase2-wiring.md`。

在 §5 的扩展路径上，把**剩余四个形状**全部接进生产链（wkv 不动，逐位保持第一阶段的产物）。

### 8.1 生成（唯一合法路径）

```bash
scp kernels/tilelang/gen_proj_shapes_aot.py ubuntu@43.202.208.136:~/tl_proj/
ssh ubuntu@43.202.208.136 \
  'cd ~/tl_proj && mkdir -p proj_gen && /opt/dlami/nvme/dsv41_venv/bin/python gen_proj_shapes_aot.py proj_gen'
scp ubuntu@43.202.208.136:'~/tl_proj/proj_gen/*' kernels/cuda/tilelang_gen/
# 再给 10 份 .cu 加 banner（banner 是唯一的、可复现的本地改动）
```

| 项 | 值 |
|---|---|
| 形状（稠密） | `wq_a (n=1280,k=5120)`、`wq_b (n=4096,k=1280)`、`wo_b (n=5120,k=1024)` |
| 形状（分组） | `wo_a`：`G∈{1,8}`、`n=1024`、`k=4096`、`a_stride∈{4096,32768}`、`OS=8192` |
| 几何 | `bN=128, ks=8, threads=128, ns=3`；reduce `bN=256, threads=256`（同第一阶段） |
| 数值路线 | route A（原生 fp8 mma + per-32 scale，与原型/第一阶段逐行一致） |
| ABI | 裸指针（`TL_DISABLE_TMA_LOWER:1, TL_DISABLE_WARP_SPECIALIZED:1`）|
| 产物 | `{wq_a,wq_b,wo_b}_{partial,reduce}_tl.cu` + `wo_a_g{1,8}_{partial,reduce}_tl.cu` + `proj_shapes_tl_config.txt` |

**与第一阶段唯一的生成差异：输出行 stride 由 `OS` 烘进来，不是 `n`。**
ferrite 的 `out_stride` 是本形状调用点真实的行距，且**不总等于 n**：

| 形状 | 调用点 | `OS`（= out_stride） | `n` | OS == n? |
|---|---|---|---|---|
| wkv | verify/eager | 512 | 512 | ✅ |
| wq_a | verify/eager | 1280 | 1280 | ✅ |
| **wq_b** | **verify / eager `lin`** | **nh·hd = 32768** | nlh·hd = 4096 | ❌（ColumnParallel：只填整行的前 nlh·hd 列）|
| wq_b | indexer（idx_wq_b） | idx_nh·idx_hd = 4096 | 4096 | ✅ —— 但 OS 已烘成 32768 ⇒ **decline** |
| wo_b | verify/eager | 5120 | 5120 | ✅ |
| **wo_a** | verify（`wo_a_grouped_fp8` 站点） | **ol_total = 8192** | 1024（每组宽）| ❌（ColumnParallel：nlg 组写进全局宽行）|

生成物把 OS 作为编译期常量写进归约声明 `C: T.Tensor((MPAD, OS))`，**store 仍是
`tl::store_global_256`（256-bit 向量化）**，且不需要第三发 launch 做 strided copy。
shim 的形状门据 OS 做判定（见 8.2 的 m==1 放宽）。

### 8.2 shim（四个新导出符号）

`tilelang_gen/{wq_a,wq_b,wo_b,wo_a}_shim.cu`，全部照 `wkv_shim.cu` 的契约（rc `0`/`2`、
三条硬约束、INIT 懒初始化、ARMED 一次性回执、两次 launch + 常驻 scratch、K-split）。

| 符号 | 形状门 | scratch |
|---|---|---|
| `dsv41_gemm_fp8_tilelang_wq_a` | `m∈[1,8] && n==1280 && k==5120 && out_stride==1280` | `P[8][16][1280]` = 640 KiB |
| `dsv41_gemm_fp8_tilelang_wq_b` | `m∈[1,8] && n==4096 && k==1280 && out_stride==32768` | 2 MiB |
| `dsv41_gemm_fp8_tilelang_wo_b` | `m∈[1,8] && n==5120 && k==1024 && out_stride==5120` | 2.5 MiB |
| `dsv41_gemm_fp8_tilelang_wo_a` | `rows∈[1,8] && n==1024 && k==4096 && out_stride==8192 && (groups,a_stride)∈{(1,4096),(8,32768)}` | `P[8][8][16][1024]` = 4 MiB（G=8 上界，两变体复用）|

公共门（四个都有）：`bias` 必须 null（生成物无 bias 通路，绝不静默丢 bias）；`a/a_scale/
w/w_scale` 16B 对齐、`out` 32B 对齐；INIT 失败 ⇒ decline（不半发射）。

**m==1（rows==1）时 `out_stride` 放宽**（四形状都有）：行 stride 只在 `m>1` 参与归约 store
的地址计算（第 0 行恒在 offset 0）⇒ `m==1` 时 `out_stride` 不参与计算。这条放宽是为了
eager 单行站点：`lin`/`lin2` 按 `n_out` 约定传 stride，而 wq_b 的真 OS 是 `nh*hd`；单行下
两者等价。`m>1` 仍严格要求 `out_stride == OS`。

### 8.3 接线

| 侧 | 点位 | 形状 | gate |
|---|---|---|---|
| verify | `proj_mrows`（⓪ wkv 臂**之后**新增一块）| wq_a / wq_b / wo_b | `DSV41_GEMM_TILELANG` |
| verify | `attention_rows` 的 `wo_a_grouped_fp8` 调用点 | wo_a | `DSV41_GEMM_TILELANG` |
| eager | `lin`（⓪ wkv 臂之后）| wq_a / wq_b / wo_b | `DSV41_GEMM_TILELANG && DSV41_GEMM_TILELANG_EAGER` |
| eager | `lin2`（`tilelang_dense_pick`）| wq_a / wq_b / wo_b | 同上 |
| eager | `layer()` 的 wo_a per-group 循环前 | wo_a | 同上 |

- **eager 侧统一二级 gate**（`DSV41_GEMM_TILELANG_EAGER=1`，默认 OFF）：T 臂挂起分诊的
  结论 —— eager `lin` 站点喂的是 `quant1` 的单行 `s.xq/s.xsc`，其 scale 布局审计未完成。
  verify 侧 parity 已证，`DSV41_GEMM_TILELANG` 单独即上线。
- **`out_stride` 的 decline 是设计内的**：wq_b 的 indexer 站点（`out_stride == n`）、
  wo_a 的变体不匹配，都静默回退到老 kernel（一个形状一个 dump，OS/几何是编译期常量）。
- **phantom-gate 回执**：新增 `proj_tilelang_skipped_note_phase2()` —— gate armed 但 `.so`
  一个 phase-2 符号都没有时，一次性提示（区别于"形状 decline 是正常的、静默的"）。

### 8.4 验收证据（本阶段）

| 检查 | 结果 |
|---|---|
| 10 份生成物 dump + `proj_shapes_tl_config.txt` | ✅ 见 8.1；OS 烘入已在 dump 中核对（wq_b reduce store `C + i*262144 + row*32768`）|
| `nvcc -O3 -std=c++17 -c -arch=sm_103a -I tilelang_inc -I tilelang_gen`（五个 shim）| ✅ **全 EXIT=0** |
| `cargo check -p ferrite-models` / `--workspace` | ✅ **EXIT=0** |
| GPU e2e + parity | ⏳ `kernels/cuda/tests_tilelang_proj_parity.cu`（五形状台架，主 agent 执行）|

⚠️ 本阶段**未跑 GPU**（用户指令：GPU 测量是主 agent 专属职责）。生成已在远端 B300 完成
（dump 落仓），CPU 侧 compile-only 全绿；GPU parity 台架已交付，见 §8.5。

### 8.5 GPU 验证手册（主 agent 执行）

```bash
# 编译（CPU compile-only，约 2–2.5 min；dsv41_kernels.cu 14.8k 行）
nvcc -arch=sm_103a -O3 -std=c++17 -I kernels/cuda -I kernels/cuda/tilelang_inc \
     -I kernels/cuda/tilelang_gen -o /tmp/tl_proj_parity \
     kernels/cuda/tests_tilelang_proj_parity.cu
# 运行（GPU）
/tmp/tl_proj_parity
```

判据（与第一阶段 wkv 台架同口径）：
1. 每形状 `shim rc` 必须为 **0**（2 = decline，说明接线或 OS 不对）；
2. **门 1**：`repeatability` 与 `m=6 row0 == m=1 row0` 必须 `BIT-IDENTICAL`；
3. **门 2**：`maxrel ≲ 1e-6`（原型 §5.2 实测 ≤7.6e-7），`argmax` 一致；**禁止 byte-compare**；
4. wo_a 的 `g=0..G-1` 每组 `maxrel` 同量级（分组形态不引入额外数值退化）。

接线侧的活性证据：`[proj-tilelang] ARMED {wq_a,wq_b,wo_b,wo_a} ...` 一次性回执（stderr）。

---

## 9. 第五阶段：tcgen05 block-scaled 原生 fp4 MoE-up（`DSV41_MOE_TILELANG_BS`）

> 原型与全部 GPU 结论：`docs/agent/tcgen05-blockscaled-proto.md`。
> 接线设计 + 验证手册：`docs/agent/tilelang-moe-bs-wiring.md`（**改这个臂之前先读它**）。

| 文件 | 性质 | 说明 |
|---|---|---|
| `moe_bs_up_tl.cu` | **生成物（禁止手改）** | up（gate‖up）block-scaled grouped GEMM |
| `moe_bs_up_tl_host.cu` | **生成物** | TileLang 自己的 host launcher = **CUtensorMap 的权威配方**（§9.3） |
| `moe_bs_tl_config.txt` | 生成物 | 冻结几何 + grid/block/smem + 参数签名 + tensormap 表 |
| `moe_bs_shim.cu` | **手写** | launcher shim（`dsv41_moe_tilelang_gate_up_bs` + `dsv41_moe_bs_pack_wsf`） |

### 9.1 与 bf16 臂（§7）的三点不同

1. **没有 bf16 镜像**。本臂直接吃 fp4 池 + 它自己的 ue8m0 面（零 dequant、零副本），
   所以**不需要** `DSV41_MOE_BF16_DEQUANT`，也不花 §7.5 那笔 `+105~113 GiB/rank`。
   代价是**装载期 pack 出 group-major 的 SF 池**：`+1.54 GiB/rank`（§8 的 1.5%）。
2. **权重按 expert 直取**（`W1/W3: [E, NP, K]`，`e = Eid[by]`），不是原型那套
   `[NSEG, N, K]` 的 per-segment 副本。
3. **ABI 是 TMA 描述符**（默认 lowering），不是 §7 的裸指针：
   blockscaled 的 A/B smem 必须是 `float4_e2m1_unpacked`，而 packed-global →
   unpacked-smem 只有 TMA 的 tensor 形式能做（`copy_analysis.cc:539`）。
   shim 因此用 `dlopen("libcuda.so.1")` + `dlsym("cuTensorMapEncodeTiled")`
   —— **`build.sh` 一行不改**（不引入 `-lcuda`，`BUILD_ID` 的 flag 集不动）。

### 9.2 重生成（唯一合法路径）

```bash
scp kernels/tilelang/gen_moe_bs_aot.py ubuntu@43.202.208.136:~/tl_bs/
ssh ubuntu@43.202.208.136 \
  'cd ~/tl_bs && mkdir -p aot_gen && /opt/dlami/nvme/dsv41_venv/bin/python \
   gen_moe_bs_aot.py aot_gen'
scp ubuntu@43.202.208.136:'~/tl_bs/aot_gen/*' kernels/cuda/tilelang_gen/
# 再给 moe_bs_up_tl.cu 加 banner（内容在 aot_gen/moe_bs_up_tl.banner）
```

| 项 | 值 |
|---|---|
| 形状 | up `n=640 (gate‖up), k=5120`；`E=384`、`NP=320`、`SEG_CAP=36` |
| 几何 | **`BM=128`**（认证几何）、`BN=128`、`BK=128`、`nh=64`、`grid=(5,36)`、`threads=128`、`stages=6`、`gran=32` |
| ⚠️ BM | `--bm 64`（"目标几何"）在 TileLang 0.1.14 上**不可用**：`tcgen05.cp.32x128b.warpx4` 要求 SF smem 行数是 128 的倍数 ⇒ trace 期即被库拒（`gen_moe_bs_aot.py:266 assert BM % 128 == 0`）。`BM_DEFAULT=64` 是历史遗留值，AOT **必须** `--bm 128`。shim 的 `kBm` 必须同步为 128（否则 A 的 box/行块、SFA 段步长 4608=36×128 全错）|
| 数值路线 | **原生 mxfp4**：e2m1 数据 + ue8m0 标度直进 tensor core（`kind::mxf8f6f4.block_scale`） |
| ABI | **TMA 描述符**（默认 lowering；见 §9.1-3） |
| 0.1.14 补丁 | `gen_moe_bs_aot.py::install_blockscaled_fix()`（缺 `ann["is_tcgen05"]`，上游 v0.1.14 与 main 都缺） |

### 9.3 描述符配方（**不许猜**）

`moe_bs_up_tl_host.cu` 是 TileLang 为同一 kernel 生成的 host launcher，里面有它对每个
张量调 `cuTensorMapEncodeTiled` 的完整实参。`moe_bs_shim.cu` 的 5 个 `spec_*` 就是那份
配方的转写；**唯一允许需要人工比对的是 box 首维（字节 vs 元素）与 swizzle 枚举**。
转写流程与自检见 `docs/agent/tilelang-moe-bs-wiring.md` §4.3 / §6.2–6.3。

⚠️ `moe_bs_shim.cu` 顶部的 `static_assert` 是**第一道闸**：若 dump 不是描述符形态
（例如有人加了 `TL_DISABLE_TMA_LOWER`），它会在编译期直接失败并指向 §4.3 ——
**不要绕过它**（绕过 = 一个「能编译但走错路」的 shim）。

### 9.3.1 kernel launch 的 ABI（形参序 **不许猜**）

权威配方 = `moe_bs_up_tl_host.cu` 里 TileLang 自己的调用
`TVMFFIFunctionCall(main_kernel, args, 14)`：`args[0..7]` 是 kernel 形参，`args[8..13]`
是 `grid(5,36) / block(128) / …/ smem=202752`。device 侧签名（同一份 dump）：

| # | 形参 | 类型 | 形态 |
|---|---|---|---|
| 0 | `A_desc` | `CUtensorMap` | TMA load |
| 1 | `C_desc` | `CUtensorMap` | TMA store —— ⚠️ **不是** `float*` |
| 2 | `Eid` | `const int*` | 裸指针（唯一随形参走的元数据） |
| 3 | `SFA` | `const uint*` | **裸指针**：SFA 走 `cp.async.bulk`（`tma_load(dst,src,barrier,size)`），**不需要描述符** ⇒ shim 的 `g_tmap_sfa` 是 dead 的 |
| 4 | `SFW1_desc` | `CUtensorMap` | |
| 5 | `SFW3_desc` | `CUtensorMap` | |
| 6 | `W1_desc` | `CUtensorMap` | ⚠️ **W 排在 SFW 之后** |
| 7 | `W3_desc` | `CUtensorMap` | |

⇒ 三处反直觉（C 是描述符 / SFA 是裸指针 / W 在最后）正是"错位即编译错误"的来源；
`<<<>>>` 的正确实参序见 `moe_bs_shim.cu` §6 的 `(2)` 段。
⚠️ **launch 的 smem 参数 = 202752**（= device dump 的 `buf_dyn_shmem` 最高偏移 + 尾区；
`C_sh` 与 `A_sh` **共享 offset 0**），**不是** config 里那笔未去别名的 268288（262 KiB
> sm_100 的 227 KiB/block 上限 ⇒ `cudaFuncSetAttribute` 必失败）。不符 ⇒ launch err 1。


### 9.4 验收证据

| 检查 | 结果 |
|---|---|
| `cargo check -p ferrite-models` / `cargo check --workspace` | **EXIT=0**（device.rs + weights.rs + load.rs） |
| `nvcc -O3 -std=c++17 --use_fast_math -c`（8 份 `*_shim.cu`，仅 `-I tilelang_inc`） | ✅ **8/8 EXIT=0、0 error**（2026-09-13，sm_100a；`moe_bs_shim.cu` regs 144/32/27） |
| `bash kernels/cuda/build.sh 100a`（compile **+ link**） | ✅ **EXIT=0**（16 个 TU 全绿，含 `moe_bs_shim.cu`） |
| `nm -D` 两个 T 符号 | ⏳ 待主 agent 在交付 .so 上核 |
| GPU e2e + parity + bench | ⏳ 见 wiring §6.5/§6.6（主 agent 职责；**空卡**才用绝对 µs） |
| raw sha256（`moe_bs_up_tl.cu`，banner 之前） | ✅ `ae21487c…1fa`（与 config 一致） |

**2026-09-13 ABI 修复（第 13 次重编的编译错误）**：`moe_bs_shim.cu` 的 kernel 调用按 §9.3.1
重排为 `(g_tmap_a, g_tmap_c, g_eid, g_sfa, g_tmap_sfw1, g_tmap_sfw3, g_tmap_w1, g_tmap_w3)`；
并同步 `kBm 64→128`、`kSmem 184832→202752`。AOT 产物恢复原名（`moe_bs_up_tl.cu` /
`moe_bs_up_tl_host.cu`，不再是 `.pending_abi_fix`）。

⚠️ **仍未做（属 §9.3 的独立转写比对任务，改前先读 wiring §4.3/§6.2–6.3）**：
`spec_a`/`spec_w` 的 dtype 与 box 语义（host 用 sub-byte dtype=14 + **按元素**的 box
`(128,128)`；shim 现用 `UINT8` + 按字节 `(64,·)`），以及 `spec_c` 的 box 应为 `(32,128)`
（现为 `(kBn=128,·)`；128×f32=512 B > 128 B swizzle ⇒ encode 必失败）。这两条不修，
运行期 `dsv41_moe_tilelang_gate_up_bs` 会在 tensormap 编码处 decline（返回 2 ⇒ 安全回退
老路径，**不是**错值）。

---

### 9.5 D2 精度修复：A operand e2m1 → **e4m3**（2026-09-13，工部）

> 设计 + 逐项改动：`docs/agent/moe-bs-e4m3-activation-design.md`。
> 重生成/验收命令清单：本目录 `REGEN-E4M3.md`（**给主 agent 的远端执行清单**）。
> 红线：官方 DeepSeek-V4.1 的 routed 激活是 `act_quant(fp8_block_size=32, ue8m0)` 的
> **e4m3**，权重才是 MXFP4 e2m1；本臂此前把两侧都当 e2m1，**激活少 4 bit 位宽**。

| 文件 | 性质 | 改了什么 |
|---|---|---|
| `kernels/tilelang/gen_moe_bs_aot.py` | 手改 | `A: T.Tensor((M,K), T.float8_e4m3fn)`；`A_sh = alloc_shared(..., T.float8_e4m3fn)`（不再是 `float4_e2m1_unpacked`）；config 增 D2 审计段（idesc/tx/模板/行距） |
| `moe_bs_up_tl.cu` + `_host.cu` + `moe_bs_tl_config.txt` | **生成物（待重生成）** | 预期 diff 见 §2.2 的 5 项审计清单；`idesc 144709248 → 144708608` |
| `moe_bs_shim.cu` | 手改 | `kABox` 分家（`kABoxA = BK` / `kABoxW = BK/2`）；`spec_a` 的 `gdim[0]=gstride[0]=K`、`box[0]=BK`；gather 的 `k2 → abytes`（e4m3 直读）；`g_a = SEG_CAP*BM*K`（23.6 MB）；**新增能力符号 `dsv41_moe_bs_act_e4m3_cap`** |
| `crates/ferrite-models/src/dsv41/device.rs` | 手改 | `moe_bs_act_e4m3_cap` 字段 + `ko!` 注册；`supports_moe_tilelang_bs()` 纳入该 cap；新增 `supports_moe_bs_act_e4m3()`；`xq4` 文档改 e4m3 |
| `crates/ferrite-models/src/dsv41/chain_dev.rs` | 手改 | `moe_tilelang_bs_ready`：`!(e4m3 && cap)` → **`e4m3 && caps`**（臂现在**要求** e4m3 激活）；REFUSED 文案同步 |

**不许改的**：W1/W3/SFW 的读取与 pack（权重仍是 packed e2m1 + ue8m0）、SFA/SFW 布局、
`gran`、BM/BN/BK/stages/threads/grid、`kSmem=202752`、down（w2）臂。
**旧 `.so` 的静默错值面**由 `dsv41_moe_bs_act_e4m3_cap` 关掉（`xq4` 的语义变了但 C ABI
形状没变 ⇒ 没有这个符号就**不 arm** 该臂）。




