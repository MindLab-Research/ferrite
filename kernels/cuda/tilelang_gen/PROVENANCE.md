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

生成物头部各有一行 `GENERATED — do not edit` banner（banner 之外逐字节等于 dump）。
两个 dump 都定义 `extern "C" __global__ void main_kernel(...)`，所以 `wkv_shim.cu` 用
`#define main_kernel ...` 改名后 include；**build.sh 只把 `wkv_shim.cu` 当一个 TU**。

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
| `cute/` + `cutlass/` | `tilelang-0.1.14/3rdparty/cutlass/include/` 的**传递闭包** | 35（0.6 MB）|

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

---

## 6. 纪律

- 生成物**禁止手改**；改动一律走 §2 的重生成 + 全量回归（一次显式动作）。
- `tilelang_inc/` 与生成物**一起冻结、一起 commit**：TileLang 0.1.14 → 0.2.x 的产物形态
  会变，升级 = 重生成 + 重跑微基准（门 1）+ 重新 vendor。
- 生成物进 `build.sh` 的 `SRCS` ⇒ 自动进 `CU_HASH` ⇒ 进 `BUILD_ID`（same-source gate
  免改一行）。
