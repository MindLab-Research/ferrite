# AOT 重生成清单 —— D2 修复（fp4 MoE 臂的 A operand：e2m1 → e4m3）

> 工部 · 2026-09-13。**这份文件是给主 agent 在远端 B300 上逐条执行的命令清单**
> （工部不做任何远端/GPU 操作）。设计与逐项判据：`docs/agent/moe-bs-e4m3-activation-design.md`
> §2.2 / §3；接线手册：`docs/agent/tilelang-moe-bs-wiring.md`。
>
> 前置：工部已改完的 4 个手写件（`kernels/tilelang/gen_moe_bs_aot.py`、
> `kernels/cuda/tilelang_gen/moe_bs_shim.cu`、`crates/ferrite-models/src/dsv41/device.rs`、
> `chain_dev.rs`）。**生成物（`moe_bs_up_tl.cu` / `_host.cu` / `moe_bs_tl_config.txt` /
> `.banner`）未经重生成之前，仓库处于「shim 期望 e4m3、生成物仍是 fp4」的不一致态**
> —— 这是**预期**的中间态（shim 的 `static_assert` 只检查 ABI 形态，不检查 dtype），
> 但在 STEP 1–2 落盘之前**不要跑 GPU 数值验收**（会得到 idesc=5/5 的旧臂结果）。

---

## STEP 0 —— vendored 源码补丁（幂等，先打环境）

```bash
scp -r kernels/tilelang/vendor      ubuntu@43.202.208.136:~/tl_bs/vendor
scp kernels/tilelang/gen_moe_bs_aot.py ubuntu@43.202.208.136:~/tl_bs/
ssh ubuntu@43.202.208.136 \
  'cd ~/tl_bs && /opt/dlami/nvme/dsv41_venv/bin/python vendor/apply_tilelang_patch.py'
```

判据：输出含 `tcgen05_gemm_blockscaled.is_tcgen05 : fixed` + `[vendor] verified`。
（没修 ⇒ STEP 1 会直接 `RuntimeError`，脚本自己会打印补救命令。）

## STEP 1 —— 重新生成（BM=128 是唯一可用的认证几何）

```bash
ssh ubuntu@43.202.208.136 \
  'cd ~/tl_bs && mkdir -p aot_e4m3 && /opt/dlami/nvme/dsv41_venv/bin/python \
   gen_moe_bs_aot.py aot_e4m3 --bm 128'
```

geometry 判据（config 打印，**与修复前逐项相同**）：
`BM=128 BN=128 BK=128 NH=64 stages=6 gran=32`、`sf_words=40 sf_period=1 k_iters=40`、
`grid=(5, 36)`、`host_source=moe_bs_up_tl_host.cu`。

## STEP 2 —— 生成物审计（**机械比对，5 项逐条命中**；不命中就停）

```bash
ssh ubuntu@43.202.208.136 'cd ~/tl_bs/aot_e4m3 && \
  grep -n "144708608\|kFloat8_e4m3\|expect_transaction(16384)" moe_bs_up_tl.cu'
```

| # | 位置 | 期望值 | 判据 |
|---|---|---|---|
| 1 | MMA 模板实参 | `tcgen05mma_blockscaled_ss<tl::DataType::kFloat8_e4m3,false>` | 取自 **A** 的 dtype（旧值 `kFloat4_e2m1fn`） |
| 2 | idesc 常数 | **`144708608`（0x08A01400）** | 位分解 `a_format=0(E4M3) / b_format=5(E2M1) / K32 / E8M0`；**旧值 `144709248`（0x08A01680）** |
| 3 | A 的 TMA 事务 | `expect_transaction(16384)` | `BM*BK*1 B`；旧值 `8192`（= BM*BK/2，packed fp4） |
| 4 | A smem 描述符 | `initialize_tcgen05_descriptor(desc_a, A_sh, 1, 64, 0, 0, 2)` **预期不变** | A 行仍是 `BK=128 B` ⇒ 128B swizzle 不变；**LBO/SBO 若变了不算失败，但必须转写进审计记录并回改 shim** |
| 5 | A_sh stage 布局 | 偏移 0 / stride `16384` / `B_sh` 仍在 `98304` | 字节数与 B 的偏移都不变 ⇒ `kSmem=202752`、stages、占用率不变 |

⚠️ **R1 闸门**：若 #2 不是 `...08608`（例如仍是 5/5 编码，或变成别的值），**立刻停下上报**
——那意味着 TileLang 用单一 dtype 推了 `a_format`/`b_format`，必须走「vendor 源码补丁」的
路。**绝不许手改生成物的常数**（生成物是 `DO NOT EDIT`）。

## STEP 3 —— 回传生成物 + 摘要行

```bash
scp ubuntu@43.202.208.136:'~/tl_bs/aot_e4m3/moe_bs_up_tl.cu'      kernels/cuda/tilelang_gen/
scp ubuntu@43.202.208.136:'~/tl_bs/aot_e4m3/moe_bs_up_tl_host.cu' kernels/cuda/tilelang_gen/
scp ubuntu@43.202.208.136:'~/tl_bs/aot_e4m3/moe_bs_tl_config.txt' kernels/cuda/tilelang_gen/
scp ubuntu@43.202.208.136:'~/tl_bs/aot_e4m3/moe_bs_up_tl.banner'  kernels/cuda/tilelang_gen/
```

* 把 `moe_bs_up_tl.banner` 的内容**贴到 `moe_bs_up_tl.cu` 头部**（与现存形态一致，
  即 banner 3 行 + 空行 + 原 dump），并核对新 `raw sha256` 与新 config 的 `raw_sha256(up)=` 一致。
* config 里新增的 **D2 审计段**（生成器输出，逐行核对）：
  `idesc_blockscaled=144708608`、`a_tma_bytes_per_k_iter=16384`、
  `mma_template=...kFloat8_e4m3...`、`a_row_stride_bytes=5120`。

## STEP 4 —— compile-only（无 GPU，**别跳**）

```bash
nvcc -arch=sm_103a -cubin -O3 -std=c++17 -I kernels/cuda/tilelang_inc \
     -o /tmp/moe_bs_up_e4m3.cubin kernels/cuda/tilelang_gen/moe_bs_up_tl.cu
nvcc -O3 -std=c++17 -shared -fPIC -arch=sm_103a -I kernels/cuda/tilelang_inc \
     -o /tmp/moe_bs_shim.so kernels/cuda/tilelang_gen/moe_bs_shim.cu
nm -D /tmp/moe_bs_shim.so | grep -E \
  "dsv41_moe_tilelang_gate_up_bs|dsv41_moe_bs_pack_wsf|dsv41_moe_bs_act_e4m3_cap"
nm -D --undefined-only /tmp/moe_bs_shim.so | grep -c cuTensorMapEncodeTiled   # 期望 0
```

判据：4 个符号全部导出（含**新增的 `dsv41_moe_bs_act_e4m3_cap`**）；`libcuda` 仍无链接期依赖。

## STEP 5 —— 描述符转写比对（A 这次**期望出现**）

```bash
grep -n "tensormap_create_tiled" -A 40 kernels/cuda/tilelang_gen/moe_bs_up_tl_host.cu | head -60
```

A 的期望 dump（shim 的 `spec_a()` 已按此转写）：

```
rank=2, addr=A, gdim=(5120, 4608), stride=(1, 5120), box=(128, 128),
estride=(1,1), ilv=0(NONE), swz=3(128B), l2=2(L2_128B), oob=0
```

（修复前 = fp4 的视图：`gdim=(5120,4608)`、`stride=(1,2560)`、`dtype=14`（16U4_ALIGN16B）。）
W 侧不变：`gdim[0]=K/2=2560`、`box[0]=BK/2=64`。
若 swizzle ≠ 128B 或 A 的 `box[0] ≠ 128` ⇒ 按 dump 改 shim 的 `spec_a()`，回到 STEP 4。
运行期 `DSV41_MOE_BS_DEBUG=1` 会把 shim 实际用的 spec 打出来，与 host source 一对一 diff。

## STEP 6 —— 构建 + 数值验收（需 GPU）

```bash
bash kernels/cuda/build.sh 103a      # 或按现场 arch；compile + link 全绿
```

```bash
# 官方口径（e4m3 激活是前提，不是选项）
DSV41_MOE_TILELANG_BS=1 DSV41_EXPERT_ACT_E4M3=1 <run>
# 参考臂（SIMT e4m3 路径）：不设 DSV41_MOE_TILELANG_BS
```

判据：
* 回执必须出现 `[moe-bs] ARMED gate_up_bs ...`（否则这一跑测的是**老路径**）；
* gate‖up 输出与参考实现的相对误差落在 **e4m3 量化噪声量级**（而非 e2m1 量级——后者大约 8×）；
* 每次实测都报 up-GEMM 的 µs：预期 A 读字节 ×2（每 k-iter 每 CTA 17408 → 25600 B），
  折算**个位数 µs**（`docs/agent/moe-bs-e4m3-activation-design.md` §4.1）。显著更大 ⇒ 回 §4.1 复核。

## 一行速查（ARMED 与格式是否真的换了）

```bash
# shim 侧（编译期常量）：
grep -n "kABoxA\|kABoxW\|s.gdim\[0\] = (cuuint64_t)kDim;\|abytes" kernels/cuda/tilelang_gen/moe_bs_shim.cu
# 生成器侧（dtype）：
grep -n "float8_e4m3fn" kernels/tilelang/gen_moe_bs_aot.py
# Rust 侧（cap 探针 + 前置条件）：
grep -rn "moe_bs_act_e4m3_cap\|supports_moe_bs_act_e4m3" crates/ferrite-models/src/dsv41/
```
