# TileLang 投影族 · 第三阶段接线与验证手册（五形状）

> 工部 · 2026-09-13 · 把 `tilelang-proj-phase2.md` 的五形状原型接进生产链。
> 上位：`tilelang-integration-design.md`（路线 (a) 源码合入 + 裸指针 ABI）；
> 阶段记录：`kernels/cuda/tilelang_gen/PROVENANCE.md` §8。
> ⚠️ 本阶段**未跑 GPU**（用户指令：GPU 测量是主 agent 专属职责）。生成已在远端 B300
> 完成、dump 落仓，CPU 侧 compile-only 全绿；GPU parity 台架已交付，见 §5。

---

## 1. 交付物

| 文件 | 性质 |
|---|---|
| `kernels/tilelang/gen_proj_shapes_aot.py` | 生成器（唯一合法路径，四个形状） |
| `kernels/cuda/tilelang_gen/{wq_a,wq_b,wo_b}_{partial,reduce}_tl.cu` | 生成物（禁手改）|
| `kernels/cuda/tilelang_gen/wo_a_g{1,8}_{partial,reduce}_tl.cu` | 生成物（禁手改）|
| `kernels/cuda/tilelang_gen/proj_shapes_tl_config.txt` | 冻结几何 + 签名 |
| `kernels/cuda/tilelang_gen/{wq_a,wq_b,wo_b,wo_a}_shim.cu` | **手写** launcher shim（4 个导出符号）|
| `crates/ferrite-models/src/dsv41/device.rs` | 4 个 `Option<fn>` + wrapper + `supports_*` |
| `crates/ferrite-models/src/dsv41/chain_dev.rs` | verify 4 点 + eager 4 点 gate 臂 |
| `kernels/cuda/tests_tilelang_proj_parity.cu` | 五形状 GPU parity 台架（主 agent 跑）|

wkv（第一阶段）**逐位未动**。

---

## 2. 关键设计：输出行 stride `OS` 烘进生成物

ferrite 的 `out_stride` 是各调用点真实的行距，**不总等于 `n`**：

| 形状 | 调用点 | `OS` = out_stride | `n` | OS==n |
|---|---|---|---|---|
| wkv | verify / eager | 512 | 512 | ✅ |
| wq_a | verify / eager | 1280 | 1280 | ✅ |
| **wq_b** | **verify / eager `lin`** | **nh·hd = 32768** | nlh·hd = 4096 | ❌ |
| wq_b | indexer `idx_wq_b` | 4096 | 4096 | ✅ → **decline** |
| wo_b | verify / eager | 5120 | 5120 | ✅ |
| **wo_a** | verify（grouped 站点） | **ol_total = 8192** | 1024（每组）| ❌ |

生成器把 OS 作为编译期常量写进归约声明 `C: T.Tensor((MPAD, OS))`，**store 保持
`tl::store_global_256`（256-bit 向量化）**，不需要第三发 strided copy。

- **decline 是设计内的**：wq_b 的 indexer 站点（`out_stride == n ≠ OS`）、wo_a 的变体不匹配，
  静默回退到老 kernel（一个形状一个 dump，OS/几何编译期确定）。
- **`m==1` 放宽**：行 stride 只在 `m>1` 参与归约 store 地址计算（第 0 行恒在 offset 0）。
  eager 单行站点（`lin`/`lin2`）按 `n_out` 约定传 stride，而 wq_b 真 OS 是 `nh*hd` ——
  单行下两者等价，故 shim 在 `m==1` 时放宽 `out_stride`，`m>1` 仍严格要求 `== OS`。

---

## 3. 接线图（8 个 gate 臂）

| 侧 | 函数 / 点位 | 形状 | 额外条件 |
|---|---|---|---|
| **verify** | `proj_mrows`（⓪ wkv 臂之后新增一块）| wq_a / wq_b / wo_b | `DSV41_GEMM_TILELANG` |
| **verify** | `attention_rows` 的 `wo_a_grouped_fp8` 调用点 | wo_a | `DSV41_GEMM_TILELANG` |
| **eager** | `lin`（⓪ wkv 臂之后）| wq_a / wq_b / wo_b | `+ DSV41_GEMM_TILELANG_EAGER` |
| **eager** | `lin2`（经 `tilelang_dense_pick`）| wq_a / wq_b / wo_b | `+ DSV41_GEMM_TILELANG_EAGER` |
| **eager** | `layer()` 的 wo_a per-group 循环前 | wo_a | `+ DSV41_GEMM_TILELANG_EAGER` |

- **precedence 不变**：⓪ TILELANG(所有形状) > ① PROJ_MMA > ② MPAR > ③ MTILE > ④ SIMT。
  新增的 phase-2 臂与 ⓪ 同一 family gate，且在 wkv 臂之后试 —— 形状互斥（(n,k) 不同），
  只有一个能命中。
- **double swap 契约**：TileLang 程序把 M 放进 mma tile，但它是与 `gemm_fp8_mx` /
  `gemm_fp8_mrows` **都不同**的程序 ⇒ 只有 eager 与 verify **同时**取它才自洽。
  verify 侧 parity 已证（`DSV41_GEMM_TILELANG` 单独即上线）；eager 侧因单行 scale 布局
  审计未完成，**默认 OFF**，需 `DSV41_GEMM_TILELANG_EAGER=1` 才挂（T 臂挂起分诊的结论）。
- **phantom-gate 回执**：`proj_tilelang_skipped_note_phase2()`（一次性）—— gate armed 但
  `.so` 一个 phase-2 符号都没有时提示；"形状 decline" 是正常的、静默的。
- ⚠️ wkv 臂（`chain_dev.rs` 的 `lin` / `proj_mrows` 两处，T 臂 gate 修复所在）**未动**。

---

## 4. shim 契约（四个新符号）

rc：`0` 已发射 / `2` DECLINED（调用方保持老路径）/ 其它 = `cudaGetLastError()`（真失败）。
**永不返回 1**（1 是 `cudaErrorInvalidValue`，与真实发射失败不可区分）。

| 符号 | 形状门 |
|---|---|
| `dsv41_gemm_fp8_tilelang_wq_a` | `m∈[1,8] && n==1280 && k==5120 && out_stride==1280` |
| `dsv41_gemm_fp8_tilelang_wq_b` | `m∈[1,8] && n==4096 && k==1280 && out_stride==32768` |
| `dsv41_gemm_fp8_tilelang_wo_b` | `m∈[1,8] && n==5120 && k==1024 && out_stride==5120` |
| `dsv41_gemm_fp8_tilelang_wo_a` | `rows∈[1,8] && n==1024 && k==4096 && out_stride==8192 && (groups,a_stride)∈{(1,4096),(8,32768)}` |

公共：`bias` 必须 null；`a/a_scale/w/w_scale` 16B、`out` 32B 对齐；INIT 失败 ⇒ decline。
每次调用 = **2 次 launch**（partial + reduce）+ 常驻 scratch `P`。

---

## 5. GPU 验证手册（主 agent 执行）

### 5.1 编译（CPU compile-only，约 2–2.5 min）

```bash
cd kernels/cuda
INC="-I . -I tilelang_inc -I tilelang_gen"
for s in wkv wq_a wq_b wo_b wo_a; do
  nvcc -arch=sm_103a -O3 -std=c++17 $INC -c tilelang_gen/${s}_shim.cu -o /tmp/${s}_shim.o
done
nvcc -arch=sm_103a -O3 -std=c++17 $INC -o /tmp/tl_proj_parity \
     tests_tilelang_proj_parity.cu /tmp/{wkv,wq_a,wq_b,wo_b,wo_a}_shim.o
```

> ⚠️ 五个 shim **不能** include 进同一 TU（匿名命名空间的 `kTLN`/`g_part` 等同名）——
> 必须像生产 build.sh 一样各自一个 TU 再链接。

### 5.2 运行与判据

```bash
/tmp/tl_proj_parity        # GPU
```

1. 每形状 `shim rc` 必须 **0**（2 = decline ⇒ 接线或 OS 不对）；
2. **门 1**：`repeatability`、`m=6 row0 == m=1 row0` 必须 `BIT-IDENTICAL`；
3. **门 2**：`maxrel ≲ 1e-6`（原型 §5.2 实测 ≤ 7.6e-7），`argmax` 一致；**禁止 byte-compare**；
4. wo_a 的 `g=0..G-1` 每组 `maxrel` 与稠密同量级（分组不引入额外退化）；
5. 激活只分配 `m` 行（不 pad 到 16）—— 越界即 fault，是 runtime-m 谓词的硬证据。

### 5.3 接线活性证据

stderr 一次性回执（每个 shim 第一次命中时）：

```
[proj-tilelang] ARMED wq_a m=.. n=.. k=.. ks=8 os=1280 -> grid=(10,8)x128 ...
[proj-tilelang] ARMED wq_b ... / wo_b ... / wo_a G=.. rows=.. a_stride=..
```

graph capture 内跑：INIT 的 `cudaMalloc`/`SetAttribute` 只在首次调用做；若首次落在 capture 内
会 INIT 失败 ⇒ decline（不半发射），调用方保持老路径。

---

## 6. 未做 / 风险

| # | 项 | 说明 |
|---|---|---|
| 1 | **eager ABI 审计** | eager `lin` 的单行 `s.xq/s.xsc` scale 布局审计未完成（T 臂首次请求 sticky illegal access）。在另一 peer 手里；完成前 `DSV41_GEMM_TILELANG_EAGER` 默认 OFF。|
| 2 | **单发 K-split（ctr 归约）** | 仍是两发 launch（partial + reduce）。集成优化项，见 phase2 §8#2。|
| 3 | **wq_b indexer 站点** | `idx_wq_b`（`out_stride==n`）decline，保持老 kernel。要接需再生成一个 `OS=idx_nh*idx_hd` 的变体。|
| 4 | **draft wo_a** | draft 的 `layer()` 走 per-group `gemm_fp8_mx` 循环；TileLang grouped 臂挂在 `layer()` 的循环前（eager gate 下），非 draft 专用。|
| 5 | **`bias`** | 生成物无 bias epilogue；带 bias 一律 decline（四形状站点当前均 null）。|
| 6 | **swapAB** | 未做（第一/二阶段结论：pad-16 下 M6/M1 已 1.00，无判据要求）。|
