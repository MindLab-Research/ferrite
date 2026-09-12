# expert tcgen05 fp4 swapAB 实施计划（下一会话）

> 前置：本会话 `tcgen05-probe` 已用 `kernels/cuda/tests_tcgen05_mxf8f6f4_1x.cu`
> 在 `-gencode arch=compute_103a,code=sm_103a` 下 ptxas 验收通过
> `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X`，操作数布局确认：
> A=fp4 smem descriptor，B=e4m3，D=TMEM，SFA/SFB=TMEM e8m0。使用（不重证）。
> 目标：expert 2.0ms/步 → ~1.0ms/步 ⇒ 步时 ~3.0ms ≈ 300+ tok/s（**唯一路径**，4-5 人日）。
> 代码基线：`kernels/cuda/dsv41_experts_mxf4.cu`（现有 `mxf4_gemm_kernel` / `expert_gemv_fp4*`）。

## 0. 为什么 expert 与 gemv 的 swapAB 不同（避免重蹈 serve 中性）

| | gemv swapAB（已关闭） | expert tcgen05（本项目） |
|---|---|---|
| 权重读取 | 延迟绑（并行度不足，10.8x 差） | **DRAM 绑**：0.9GB/步 ÷ 8TB/s = 0.11ms 地板，当前 2.0ms = **18x** 差距 |
| M=128 tile | n 侧列复制的固定成本 | **权重侧满填**：gateup 3840/128=30 tile、down 5120/128=40 tile |
| 结论 | 隔离 1.76-1.94x，serve 中性 | **有真实 DRAM 缺口**，但仍有翻译风险 ⇒ 必须走 6 条协议 |

**swapAB 语义**：`D[M=权重行, N=激活列] = A[权重 M×K] × B[激活 K×N]`。
M 侧 3840/5120 整除 128（满 tile），N 侧 decode=1 token → 补到 **N=8**（mxf8f6f4 最小合法 N），
仅激活侧 8x 冗余（可忽略，权重侧零冗余）。
- gateup：A=W1W3[3840, dim=5120]，B=act[5120, 8]，K=5120（160 个 32-块 ✓）
- down：A=W2[5120, inter=1920]，B=act[1920, 8]，K=1920（60 个 32-块 ✓）
- scale：K 粒度 32 e8m0，与 checkpoint per-(row,k/32) **零转换**复用。

## 1. Phase 0（0.5 天）探针扩展：真实权重 + 数值 parity

**目的**：把「ptxas 通过」升级为「数值正确」，在 GPU 上闭环。
1. 复用 `tests_tcgen05_mxf8f6f4_1x.cu` 脚手架，把 PROBE_K 从 32 扩到真实 K（5120/1920 分块循环）。
2. 输入改用**真实 checkpoint 权重**（某专家 W1/W3/W2 的 fp4 e2m1 + e8m0 per-32）；
   激活用同一 x，量化到 **e4m3（block 32, e8m0）**（注意：现 expert 路径量化到 fp4 e2m1；
   此处改 fp8 精度更高，数值风险降低）。
3. 判据（对齐 `tests_tcgen05_mxf4.cu` 口径）：
   - `max|diff| / max|ref| < 5e-2`（vs 当前 SIMT `expert_gemv_fp4` / mxf4 GEMV）
   - 每元素相对误差 p50/p90/p99/max < 5e-2（分母下限 1e-3·max|ref|）
   - CPU golden（同字节块缩放和）两路都 < 5e-2（防共享布局 bug）
   - `argmax` 一致 + 报 `max|diff| vs top1-top2 margin`
4. **验收**：GPU 上 `nvcc -gencode arch=compute_103a,code=sm_103a && ./t` 全绿。
   **若 parity 不过，不进入 Phase 1**（布局/scale 映射的 bug 必须在此拦下）。

## 2. Phase 1（1-2 天）gateup kernel

落点：`kernels/cuda/dsv41_experts_mxf4.cu` 新增 `mxf8f6f4_swapab_gemm_kernel`（保留旧 kernel）。
- **A 侧（权重）**：fp4 smem descriptor。**必须补多级异步 staging**（TMA bulk / cp.async）——
  现 `mxf4_gemm_kernel:348-421` 全程 LDG→STS、无 cp.async/TMA（STATUS:5038 的 16.8GB/s 0.2%
  带宽地板根因）。swapAB 只解决 M 钉死，**不补 staging 仍是 0.2% 地板**。
- **B 侧（激活）**：e4m3 8×K staged；N=8（col0=token，其余 0）。
- **D**：TMEM 累加器；epilogue 用 `tcgen05.ld` 取回，套用现有 epi_mode（gate/up clamp `limit`，
  down 的 row_weight，`epi_mode==3` 累加）。
- **scale**：SFA（权重 per-row per-32 e8m0）、SFB（激活 per-32）经 `tcgen05.st` 写 TMEM；
  idesc 用 `make_idesc_mxf8f6f4(a_fmt=E2M1=5, b_fmt=E4M3=0, sf_id...)`。
- 形状契约沿用：launcher 只校验 `k % 32 == 0`；`n % 128 == 0` 需新增（M 侧 tile）。
- **验收**：隔离微基准 + Phase 0 同款 parity（真实权重）。

## 3. Phase 2（0.5-1 天）down kernel + launcher + Rust FFI

- down：同 kernel 模板，K=1920、M=5120，套 `expert_gemv_fp4_down_reduce_kernel` 的
  **ascending-slot 定点序**（fp 加法不结合，顺序即契约）或产 scratch 走 `dsv41_moe_down_reduce`。
- launcher：新增 `DSV41_EXPERT_TCGEN05` 门禁（**OnceLock 读一次**，否则破坏 graph capture），
  默认 OFF；旧 GEMV 路径保留为 fallback 与 A/B 基准。M=1 时**不能**再走 128x 冗余的旧 `mxf4_gemm_kernel`
  （见 `launch_mxf4:2109` 的注释史）。
- Rust FFI：`crates/ferrite-models/src/dsv41/kernels.rs` 新增符号 + `device.rs` 包装
  （参照 `expert_gate_up_fp4_batched` / `expert_down_reduce_fp4_batched`），
  `chain_dev.rs` 的 `moe_batch()` 分支按门禁派发；batched 形状（grid.y=slot）必须保持 disjoint 输出。
- **验收**：`cargo check` + 符号在 `.so` 中存在（`supports_*` 版本探测，旧 .so 回退）。

## 4. Phase 3（0.5 天）serve A/B + 数值 parity + 四段文本

- 走 `scripts/dsv41_swapab_text_parity.sh` 同款：**两次独立启动**（gate 每进程只读一次），
  臂 = `DSV41_EXPERT_TCGEN05=0 / =1`。
- 判据（硬线，全部满足）：
  1. 四段文本（Paris / 静夜思 / 1+1=2 / 出师表）**正文逐字相同**（首 token 可翻，仅当正文相同）；
  2. `faults=0`；3. p50 下降（预期 6.17 → ~5.2ms，~190+ tok/s；若真达 3.0ms 则 ~330）。
- 单测：新增 `expert_tcgen05_parity.rs`（CPU golden + vs SIMT + argmax margin），无 checkpoint 可跑。

## 5. 风险缓解（按 6 条 serve-faithful 协议）

STATUS:7713 定案的 6 条，隔离评估必须全部满足，否则数据无效：
1. **生产构型**（含已 ON 的前置优化：ILV / W2 prewarm / DOWN_FUSE / gateup_fuse）；
2. **图模式 capture+replay** 计时（禁 direct-launch 计时）；
3. **侧流干扰注入**（复现 4 流并行的 SM 争抢）；
4. **L2 污染**（每轮换缓冲，禁驻留红利）；
5. **关键路径计费**（gate/fork kernel 按 `fork_ev` 语义）；
6. **判据分层**：隔离只做**淘汰**；**正向收益一律 serve A/B + 四段文本 + faults=0**。

**额外的 expert 专属风险**：
- ⚠️ **翻译风险最高的一类（第 7 类系统差异）**：swapAB 已在此处失败（隔离 1.76-1.94x → serve 中性）。
  **止损门**：若隔离微基准达标但 serve A/B 中性（Δ<0.05ms），**立即关闭路径、默认 OFF**，
  不再投入变体矩阵（勿重演 v17→v21 四变体全中性的浪费）。
- ⚠️ **fast-math 重结合**：`--use_fast_math` 默认 ON（build.sh:42），任何多点累加必须
  `__fadd_rn/__fmul_rn/fmaf` 钉死，否则 ~1ULP/层漂移、40 层后模型坍缩（已发生过）。
- ⚠️ **图捕获禁区**：门禁不可在 kernel 内/env 每调用读；`cudaFuncSetAttribute`/动态 smem
  在捕获内会 err 900/901（STATUS:916）。TMA descriptor 必须在捕获外构造。
- ⚠️ **err 716 / 16B 对齐**：smem descriptor 的 LBO/SBO 与 cp.async 需 16B 对齐；写错静默错值。
- ⚠️ **N=8 浪费**：仅激活侧，可接受；但别误把 topk 个专家塞进 N（A/权重不同，不可共享）。

## 6. 里程碑

| 里程碑 | 交付 | 工时 | 通过判据 |
|---|---|---|---|
| M0 | 探针扩展 | 0.5d | GPU parity 全绿 |
| M1 | gateup tcgen05 kernel | 1-2d | 隔离微基准 + parity |
| M2 | down + launcher + FFI | 0.5-1d | cargo check + 符号探测 |
| M3 | serve A/B | 0.5d | 四段文本 + faults=0 + p50 |
| **Gate** | 去留决策 | — | serve 中性 ⇒ 关闭默认 OFF；否则转默认 ON |

## 7. 建议改动点（按顺序）

1. `kernels/cuda/tests_tcgen05_mxf8f6f4_1x.cu` —— Phase 0 扩展（真实权重 + parity）。
2. `kernels/cuda/dsv41_experts_mxf4.cu` —— 新增 swapAB gateup/down kernel + launcher 门禁。
3. `kernels/cuda/build.sh` —— 无需改（新 kernel 同 TU，自动进 `.so` + build_id 哈希）。
4. `crates/ferrite-models/src/dsv41/kernels.rs` / `device.rs` —— FFI + 版本探测。
5. `crates/ferrite-models/src/dsv41/chain_dev.rs` —— `moe_batch()` 派发。
6. `crates/ferrite-dsv41/tests/expert_tcgen05_parity.rs` —— 新增单测。
