# verify 摊薄病灶审计汇总（2026-09-13 深夜，400 攻坚主战场文档）

> 权威模型：`mtp-verify-amortization-model.md`（verify(m) ≈ eager(1)+ε；400 = step ~8ms + acc 2-3）。
> 本文档汇总三份 subagent 审计判词（kernel 静态审计 / MoE 物理判决 / 图结构排除）+ 实验工具纪律，是摊薄修复的执行清单。

## 0. 总账（verify 28.17ms = eager 6.33ms 的 4.45×）

| 病灶群 | 未摊 ms | 载体 | 状态 |
|---|---|---|---|
| **根层：gemm_fp8_mrows<M> 的 M 进寄存器串列** | ~8-10 | kernel 设计修复（M 真并行） | subagent 设计中 |
| **表层：已实现未开的 per-row gate** | ~5-7 | **零代码 A/B**（快赢臂 B1） | 命令就绪 |
| compressor/engram 投影黑洞（无 gate） | ~3.5-5 | 新 kernel | subagent 实施中 |
| MoE（6× sweep 恒等式，非 bug） | ~1-2.7 | grouped 激活（GATEUP_FUSE=0） | 命令就绪 |
| draft 链（3.87ms → 1.5-2ms） | ~2 | 减肥 | subagent 分析中 |
| 图结构 | 0（排除） | — | 已判决 |

全部兑现 ⇒ verify ≈ 7-8ms + draft ~1.5-2 + commit 0.5 ⇒ **step ~9ms @ acc 2-3 ⇒ 333-444 tok/s**。

## 1. 表层病灶（快赢臂 B1 的 gate 清单，已核实前置）

| gate | 病灶 | 票面 | 前置核实 |
|---|---|---|---|
| `DSV41_ATTN_MROWS=1` | sparse attention 每行一发（q/o/xq/xsc 按行） | ~3-4ms | decline 一次性日志（:2643），看 stderr |
| `DSV41_VERIFY_ROPE_MROWS=1` | q rope 每行一发（:11012） | ~1.5-2ms | 无额外前置（:1220，`!= "0"`） |
| `DSV41_VERIFY_WOB_MROWS_F32=1` | wo_b 每行 quant + 1 发 proj（:11774） | ~2-3ms | `== "1"`（:4984）；**非逐位**（跳 fp8 往返更准）——acceptance 是红线非 byte compare |
| `DSV41_GATE_MROWS_ROUTE=1` | gate+route 两发 | ~0.13ms | `DSV41_GATE_MROWS=0` 是恢复门（:501） |
| `DSV41_MROWS_ACT_CPASYNC=1`（1b） | 激活 staging 标量 16× 指令 | ~0.3-0.5ms | 逐位等价已证；活性回执 `[mrows-act-cp16] ARMED` |

注意：B4（RMSNORM_ROPE_MROWS）依赖 VERIFY_FORK 同臂（lazy 栈语义）——SWALLOW 栈不开 FORK，B1 不含 B4。

## 2. 根层病灶（gemm_fp8_mrows 的设计死锁）

`dsv41_kernels.cu:5334 gemm_fp8_mrows_kernel<M>`：
- 并行度 `nt = ceil(n/nwarps)` 只来自 n，**与 M 无关**；M 行进 `float acc[M]`（单 warp 寄存器）⇒ m=1 的 memory-bound 在 m=6 退化为 latency-bound（每块串行链 ×M）。
- **fold_r 反证**：M 折进 grid（ng=6）⇒ 每 ng-block 重新 staging 权重（cp.async16 到 private smem）⇒ 权重流量 ×6 ⇒ 63.8→10.3（6× 实测）。
- **设计死锁**：M-in-register（串行）vs M-in-grid（重读权重）——需第三条路：**M 真并行 + 权重只 stage 一次**（warp 级 M 分配 / L2 复用 / 两阶段 smem / tensor core）。NCU 微基准区分三假说：串行 acc / smem staging / 占用低。

## 3. MoE 物理判决（非 bug）

- MoE 成本 ∝ sweep 数 = n_assign = m×topk = 36（eager 6）；两实测点 711/722 GB/s 自洽 ⇒ **6× 是恒等式**。
- 唯一杠杆 = expert 并集去重（|active| < 36）：grouped arm（`e4m3_gemm_grouped_kernel` 每 live expert 读一次）。
- **激活阻碍（未文档化的致命坑）**：`DSV41_GATEUP_FUSE` 默认 ON 恒真 ⇒ grouped 永远 decline（"e4x epilogue only clamps, never fuses"）。**必须显式 `DSV41_GATEUP_FUSE=0`**。完整组合：`GATEUP_FUSE=0 + EXPERT_ACT_E4M3=1 + EXPERT_TCGEN05_E4M3=1 + EXPERT_GROUPED=1`（ILV=0 是 grouped 的硬条件）。
- 票面：MoE 4.90 → 2.5-3.8ms（|active| 24-18）；天花板 2.17（|active|=6）。grouped 只覆盖 gate/up（2/3 字节），down 仍 36 sweeps（grouped down 是新 kernel 立项）。
- tcgen05 SF 根修（70377ae）与 grouped 无耦合（grouped 的 B = w1/w3，k=5120→160 本就对齐；根修救的是 down 的 w2.scale）。

## 4. 图结构判决（排除）

- verify 是 **1 张 m=6 大图**（VERIFY_GRAPH_SLOTS=3 = 3 种 m 形状），非 6 张单行图。
- 图调度 ≤2.9ms（≤10%）；node 倍数 4.4× ≈ 耗时倍数 4.45× ⇒ 病灶在图内 per-row 粒度，不在图本身。
- EAGER 整步有图（GRAPH_STEP 默认 ON）⇒ 6.33 vs 28.17 是同口径对比 ✓。

## 5. 实验工具纪律（用户裁决）

1. **吞吐/步时**：非 nsys 的 e2e serve，看 `[dspark] steps=` 的 draft/verify/commit 真实分解（**禁止吞吐反推**——受 prefill/accept 影响）。
2. **per-kernel 时间**：nsys（多跑）；**死锁规避**：`DSV41_AR_V5=0 DSV41_GRAPH_STEP=0`（host barrier）+ `env -u FERRITE_P2P` + `NCCL_NVLS_ENABLE=0` + 5 分钟 SIGINT 硬帽；nsys 轮只看 kernel 相对倍数（AR 形态已变），吞吐数字必须来自非 nsys 轮。
3. **kernel 深度分析**（占用/带宽/瓶颈）：NCU——**只跑特定 kernel 的 micro bench（tests 二进制），不能 e2e**。
4. e2e 一律 background 模式（serve 启动 ssh 会挂住前台）。

## 6. B1 快赢臂实测（2026-09-13 00:35，63.8 栈 + 五 gate + P3B + 1b）

| 项 | 基线 | B1 实测 | 判读 |
|---|---|---|---|
| verify | 28.17ms | **27.53ms** | 仅 −0.6ms（噪声级）——五 gate 兑现远低于票面 ~5-7ms |
| draft | 3.87ms | **3.44ms** | P3B −0.43ms ✓（票面 0.9-1.1 的一半） |
| mean-k | 1.34 | **0.75-0.92** | ⚠️ 数值红线：断崖从 line 62 提前到 52，B6（WOB_MROWS_F32 非逐位）头号嫌疑，二分进行中 |
| ATTN_MROWS | —（票面 3-4ms） | **declined: world != 1** | kernel 按 (row*h+hh)*d 索引 vs 行 pitch nh*hd——TP8 需要 row_pitch 参数（ABI 没有）⇒ attn-mrows-tp8 subagent 实施中 |
| SF 根修 | pitch=10 | **pitch=16 ✓ pool geometry OK** | tcgen05 根修装载验证通过（928 行全对齐） |
| 1b | scalar | **ARMED cp.async16** ✓ | m=5 n=288 k=5120 |

**教训**：快赢 gate 的票面是"每层节省 × 40 层"的代数，实际兑现受 launch 依赖链/图节点结构调制——**gate A/B 必须看 [dspark] 分解实测，票面只作排序用**。verify 的真大头仍待 nsys per-kernel 表定位（27.53ms 的构成）。

## 7. 断崖前移二分链（line 62 → 52，2026-09-13 00:40）

B1 系三臂的断崖都在 line 52（1..51 正确然后跳 62）+ mean-k 0.64-0.92，而 A0 基线（95d7083 二进制）是 line 62 + mean-k 1.34。二分：

| 刀 | 臂 | 结果 | 判定 |
|---|---|---|---|
| 1 | B1 全量（五 gate + P3B + 1b + SF 根修） | line 52, mean-k 0.75-0.92 | 数值被改变 |
| 2 | B1 − B6（WOB_MROWS_F32） | line 52, mean-k 0.64-0.88 | **B6 排除** |
| 3 | B1 − B6 − SF（SFPAD=0 逃生门，两侧 env 分支已核实完整） | line 52 | **SF 根修排除** |
| 4 | 新二进制 + A0 基线 gate 集复刻（含 AR_PROBE） | 待出 | 回 62 ⇒ 四新增 gate 之过；仍 52 ⇒ 二进制（1b 回执/nwarps）或基线漂移 |

注意：line 52/62 是模型退化边界的漂移（near-tie argmax 对 1ULP 敏感）——找到元凶 gate 后需评估"数值合法性"（逐位承诺 vs 实际）。

## 9. nsys 对比表实测（2026-09-13 01:00，v6 配方，swallow A0 栈 vs eager，AR_SAFE 模式）

**病灶倍数实锤**（instRatio = swallow 实例数 / eager 实例数；健康应该 ≈1-2×）：

| kernel | sw% | instRatio | 判定 | 修复载体 |
|---|---|---|---|---|
| apply_rope | 0.8 | **40.1×** | q/kv rope per-row 🔴 | VERIFY_ROPE_MROWS（弃用四 gate 之一——需查它为什么改 accept） |
| rmsnorm | 1.2 | **24.9×** | per-row norm 🔴 | 同上（B4 族） |
| compressor_pool/commit | 0.2+0.1 | **11.6×** | compressor per-row 🔴 | ✅ comp-engram-v2 已交付（COMPRESSOR_PROJ_MROWS） |
| quant_kernel | 2.1 | **9.3×** | per-row quant 🔴 | WOB_MROWS_F32（弃用）+ 折叠设计 |
| sparse_attn_split/merge | 2.8+2.3 | **3.2×** | sparse attn per-row 🔴 | ✅ attn-mrows-tp8 收尾中（TP8 row_pitch） |
| gemm_fp8_gemv | 15.4(#1) | 1.7× | draft 侧逐行 gemv + verify 并存 | draft-p3-lite-v2（段融合） |
| gemm_fp8_mrows<5> | 15.2(#2) | — | 52µs avg，M-in-register | ✅ MPAR 已交付（符号未定→NCU 判读） |
| hc_mixes | 8.0(#4) | swallow-only | 30272 发×51.4µs | hc 链待查 |
| wo_a_grouped<5> | 4.5 | swallow-only | 57.9µs | WO_PAIR 方向 |
| gemv_bf16_v1_mrows<5> | 2.5 | swallow-only | **1.36ms/发**×352（head） | head 侧 |
| AR（reduce/store/stamp） | 8.4 | 1.2× | 正常 ✓（A0 判决一致） | — |

**NCU 补充（m5.ncu-rep，mrows_bench 微基准）**：gemm_fp8_mrows<5/6> 的 **DRAM 0.67-0.79%、Compute 12.8-13%、L1 14.1-14.6%、Occupancy 13.4%、40 regs**——kernel 完全没跑满（latency-bound 特征）⇒ **MPAR（warp 并行 M）符号利好**；m=1 gemv（m4）DRAM 9.1%/L1 44.2% 同样 latency-bound。

## 10. best 系列实测（2026-09-13 01:10-01:40，双门禁 + 3 请求形态）

| 臂 | gate 增量 | verify | 判定 |
|---|---|---|---|
| best1 | +1b +VERIFY_ROPE | 27.53（−1.14） | ✓ 兑现（计数 mean-k 1.34 ✓） |
| best2 | +ATTN_MROWS（TP8 row_pitch） | **24.55（−2.98）** | ✓ 票面兑现（计数 1.30 ✓；partial decline："compressor commits per row and left no device snapshot" 只挡部分块） |
| best3 | +COMPRESSOR_PROJ +ENGRAM_PROJ | 24.45（−0.10） | ✗ 票面高估 10×（nsys 实占 0.3% vs 审计 3.5-5ms）；gate ARMED 生效但无肉 |
| mpar1 | +MROWS_MPAR=1 | 25.07（+0.52） | ✗ 负向（LUT 复制 ×5120 块 + 激活复制） |
| mparA | +MROWS_MPAR=auto（rpb=6） | 25.20-25.46（+0.7~0.9） | ✗ 负向（同上，LUT ×854 块）——**MPAR 需回炉：LUT 全局常量化/激活 smem 共享/中间 rpb 档** |
| moeg | +MoE grouped 四件套 + GROUPED_DOWN | **misaligned 挂**（8 rank 中 7 个） | 🔴 SF 根修后 tcgen05/grouped-down 的首次 e2e 触发 misaligned——诊断中（CUDA_LAUNCH_BLOCKING 定位 kernel） |

**当前最优栈 = best2**（verify 24.55ms，累计 −4.1ms；计数 acc 1.30 保持）。p50 口径首读：mparA 全步 p50=28.76ms（含 draft+verify+commit+间隙）。

**acc 任务依赖实测**：计数 1.34 / python 代码 ~0.58（draft 对代码预测差）——**400 的 8ms+acc2-3 判据在代码任务上不成立**，任务形态是 400 验证的关键变量。

**两条结构性教训**：
1. **票面必须用 nsys 时间占比算**（COMP/ENGRAM 高估 10×；ATTN_MROWS 兑现因为 nsys 占 5.1%）。
2. **mrows 零摊销实锤**（52.1µs/5 行 = 单行 gemv 的 5×）——**MPAR 前一切 mrows 化无时间收益**（gemv-lesion 的前置铁律）；共享专家 480 发/步的折叠（SH 族）也要等 MPAR 回炉后才有效。

**§10.1 env 腿判定修正（2026-09-13 01:55，envchk 臂 + sh-gate-receipts 交付）**：
- **/proc 回读判死：best 系列的 env 腿完全通**（7 个关键 gate 全在进程：ATTN/COMPRESSOR/HC_DEBUG/HC_FRONT_ROWS/INDEXER/SH_EXP/SH_PAIR 全 =1）。
- **v6 nsys 轮的 SH 幻影是那个脚本特有的 env 丢失**（nsys 包装的 env 传递问题）——**v6 的 nsys 病灶表是"SH/INDEXER/COMPRESSOR 全关"的口径**：raw hc_mixes 30272 发、480 发共享专家逐行、sparse_attn 3.2× 都是在 gate 没进进程下测的。**病灶时间占比需要按 best 系列（gate 全开）重测**——verify 24.49 的真实构成与 v6 表不同。
- **[hc-front] 无 note = hc split 在 best 系列成功运行**（off-note 和 declined-note 都没打）——**hc 的 −3ms 可能已在 best1 的 −1.14ms 里部分兑现**（v6/第四刀的 env 断裂才走 raw）。hc0 对照臂（HC_FRONT_ROWS=0 显式关）在跑，将给出 hc 的精确贡献。
- **教训**：任何 nsys/A/B 脚本必须带 `/proc/<pid>/environ` 回读断言（已加入 ab_best.sh）；nsys 包装的 env 传递（`env -u X ... nsys profile ...`）与直接 env 的差异是幻影门温床。

**§10.2 AR_SAFE 口径的 head 放大（proj-head-lesion 判决，读 nsys 表必带修正）**：
- **三处词表切片（eager/verify/draft head slice）全部以 `uses_v5()` 为必要条件**（chain_dev.rs:6112/:7109/:1925、dspark_dev.rs:3252）——AR_SAFE（AR_V5=0）下 head **每次 launch 读全量 1.323GB** 而非 165MB 切片（**8× 放大**）。
- v6 表的 head 族 6.3%（gemv_bf16 3.8% + v1_mrows 2.5%）**不是生产口径**——生产（v5 ON）下缩到 ~1/8。**wo_a_grouped / gemm_fp8 / v2 不受影响**。
- wo_a_grouped<5> 4.5%（2.49ms/步）：verify 40 发 grid 仅 128×1（<148 SM）——latency/占用率病（72GB/s ≈ 1% 峰值）；修法 = nwarps 可调（:7234 写死 8）+ WO_PAIR 未接 mrows。
- gemv_bf16_v1_mrows<5>（draft head 1.36ms）：AR_SAFE 下未切片 1.323GB/发；摊薄失效根因 = f32 激活 2× 权重字节 + 行重读。
- VERIFY_HEAD_MROWS 在 v6 轮也 env 断（v1 逐行 6 发/步为证）。

**§10.3 生产模式 gate 生效判死（shgate 臂，2026-09-13 02:15，[sh-gate] 回执）**：
- **SH 族在生产模式全部 ARMED**：`SH_PAIR_M=1: ARMED fused（ONE dsv41_gemm_fp8_sh_exp_fused<m> for the whole block, epi_add folding the w2 add）` + `COMPRESSOR_MROWS=1: ARMED（ONE fused launch）` + `INDEXER_MROWS=1: ARMED on 8 index-source layers`——**共享专家/压缩器/索引器的折叠收益已含在 verify 24.49-24.53 里**。
- **AR_SAFE nsys 表的"SH 幻影/raw hc_mixes 30272 发/sparse 3.2×"全是 AR_SAFE 口径的假象**（side-stream/event 前置在 AR_SAFE 下 decline）——nsys 病灶表对生产模式无效。
- **生产模式剩余已知病灶**：①ATTN_MROWS 的 compressor-snapshot fence（"commits per row and left no device snapshot"——部分块仍 decline）②wo_a 2.49ms（AR_SAFE 口径，生产待测）③MoE grouped 716（修复中）④MPAR（回炉中）⑤draft 3.62ms（p3lite）。
- **生产账（A/B 差分）**：真基线 ~32ms verify → hc −3.4 → 1b/ROPE/ATTN −4.1 = **24.49ms**（含 SH/COMPRESSOR/INDEXER 融合）。

**§10.4 acc 口径定谳（P0 实验，DSPARK_DEBUG k_acc 直方图，2026-09-13 02:35）**：
- 计数 66 token（30 步）的 k_acc 直方图：**17 步 × 0（57%）+ 6 步 × 5（20%）+ 7 步 × 1-4**；均值 ≈1.73（与 [dspark] mean-k 1.14+1 吻合）。
- **acc 是双峰（0 或 5）而非稳态 2-3**——400 的 acc 判据需要分布平坦化（大多数步 k_acc 2-3），不只是任务形态。
- k_acc=0 的 17 步根因待查（warm-up 步？数字 token 边界错位？draft 质量？）。
- 出师表 300 拉丁：EAGER 对照同出（biochemicalutan/protato 等）⇒ 模型行为非回归，红线通过。

**双门禁**：每个优化臂必须同时报告 `step_ms`（[dspark] 分解）**AND** `mean-k`（A0 基线 1.34；掉了 = 数值回归，立即弃用该 gate）。

1. **重编**：subagent 交付的 .cu/chain_dev 改动 → `build.sh 103a` + `cargo build --release`（双产物）+ 符号三证。
2. **逐 gate A/B**（一臂一进程，计数 200 tok，读 [dspark] steps=50）：
   - `DSV41_ATTN_MROWS=1`（TP8 row_pitch 修复后）——票面 −3-4ms
   - `DSV41_MROWS_MPAR=1`（M 真并行）——票面 −8-10ms（分批）
   - `DSV41_COMPRESSOR_PROJ_MROWS=1` / `DSV41_ENGRAM_PROJ_MROWS=1`——票面 −3.5-5ms
3. **nsys 复测**：最优臂的 kernel 表 vs eager——病灶 kernel 的倍数应从 ~6× 降到 ~1×。
4. **组合最优栈** → 全量计数 + 出师表红线 → 吞吐。
5. **B2 MoE grouped**：`GATEUP_FUSE=0 + EXPERT_ACT_E4M3 + EXPERT_TCGEN05_E4M3 + EXPERT_GROUPED`（票面 −1-2ms，tcgen05 SF 根修已验证 pitch=16）。

**nsys 轮实测注意**（v2 脚本 ~/nsys_dual.sh，2026-09-13 00:45）：AR_SAFE 模式（AR_V5=0 + GRAPH_STEP=0 + host barrier）下 verify=34.32ms / draft=5.72ms / **mean-k=1.820**（vs 生产模式 28.67/3.87/1.34）——**无图+host barrier 的代价 = verify +5.65ms + draft +1.85ms；mean-k 反而升**（时序变化影响 argmax 分布的信号，非生产口径）。nsys 轮数字只用于 kernel 相对倍数。
