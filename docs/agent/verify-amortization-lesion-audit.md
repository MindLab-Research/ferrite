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
- k_acc=0 的 17 步根因**已定位（token 模式交互）**：k_acc=5 的步 drafts 以 201（换行 token）开头——draft 善于预测分隔符；k_acc=0 的步 drafts 以数字 token 开头（14/15/16/17...）——draft 对数字序列预测差。计数任务 = [数字,换行] 交替 ⇒ acc 天然双峰。**非 bug 非 warm-up**——acc 2-3 稳态需要 draft 在非分隔 token 上的预测质量（模型能力/tap 输入质量）。
- 出师表 300 拉丁：EAGER 对照同出（biochemicalutan/protato 等）⇒ 模型行为非回归，红线通过。
- **任务普查（同栈 chat/trans/cont 三任务）**：chat mean-k=0.74、trans/cont ~0.3-0.5（混合累计 0.47）——**所有正常任务 acc < 1，无任务达 2-3 稳态**。计数任务的 1.34 已是最高（双峰 0/5）。
- **用户校准 acc 2-3 的来源待查**：lazy 栈（逐行 verify）的 acc 可能与 SWALLOW（块判定）不同——lazy 对照臂在跑；若 lazy 显著更高 ⇒ SWALLOW 的 spec_accept 块判定链有 accept bug。

**§10.5 accept 损失 0.78 的完整判决（swallow-accept-loss，2026-09-13 03:10）**：
- **判定链无 bug**：(a) 对齐逐 index 正确（无 off-by-one）、(b) spec_accept 是最长前缀匹配（无全或无）——两条栈调同一函数（:9361/:9853）。
- **DIFF_EAGER 的举证有洞**：它只比 token 不比隐藏量（tap/ring KV/compressor latent 字节从未比较）+ 自重放覆盖块写 + 从 commit 后状态出发（对状态差异免疫）⇒ "数值一致"结论被推翻——**嫌疑 (c) 以隐藏量级复活**。
- **机制**：SWALLOW 的 m=6 块在 per-row 隐藏量（tap + compressor latent）上与 lazy m=1 不等价——目标模型 argmax 稳健（DIFF_EAGER 48/48 none）但 **draft 是弱模型、消费的正是 tap**（import_tap → main_h）与 latent——计数任务 k_acc=0 集中在数字 near-tie 位 ⇒ 系统性 −0.78 且无 mismatch 行。
- **三条不等价点**：S1（头号）tap 产生路径——hc 三开门（HC_VERIFY_FUSE/HC_FRONT_ROWS/VERIFY_AR_FOLD）在 rows=m 的 hc_mixes_auto 前端 vs lazy staging；S2 compressor 状态来路（块快照+replay vs 每行 pool+commit——无护栏的 hidden 等价）；S3 comp_side 被 spec_capture 分叉。
- **旁证**：WOB_MROWS_F32（非逐位 mrows）⇒ mean-k 1.34→0.75-0.92（"m 行核非逐位 ⇒ accept 崩"先例）。
- **判定实验**：E0（V5_LEDGER 混合臂）；E1（两栈 DSPARK_DEBUG 的 tap 字节对比——决定性三分）；E2（SWALLOW × hc 三开门逐个关——最小单变量直打 S1）；E3（TAP_PARITY 护栏——hidden 等价红线，建议进验收门）。
- **过渡策略**：acc 侧以 lazy 2.120 为基准，SWALLOW 性能结论带 ±0.78 星号，直到 E1/E2 落地。

**§10.7 E5 判决（2026-09-13 03:40，两栈 DSPARK_DEBUG 的同位对比）**：
- **lazy k_acc 直方图：11×5 + 1×3（92% 步全对）** vs SWALLOW 17×0+6×5+7×1-4——同任务同模型同 env（除路径 gate）下 draft 质量天差地别。
- **同 pos drafts 分叉**（pos=52：lazy [201,511,201,397,201] k_acc=5 vs SWALLOW [426,397,201,20,201] k_acc=0；pos=64 同样）——**draft 的第一个预测就不同** ⇒ **(d) 成立：draft 输入状态差**（env 全同 + draft kernel 同族 ⇒ tap/latent 的系统性差异）。注意同 pos 的 token 历史已因前面 accept 差而不同——但 lazy 在数字位也预测对（换行开头）而 SWALLOW 预测错，方向性明确：**SWALLOW 的 tap 让 draft 变差**。
- **E2 的 hc 三门不改变 tap 写出**（它们只改 hc 前端 kernel）——**tap 写出端的块/行分叉**（:10767-10793 块写 tap_r+(slot*VERIFY_ROWS)*dim vs lazy staging+lazy_tap_commit）才是 S1 的正身。**TAP_PARITY 护栏（hc-tap-parity-fix 实施中）是下一步定位的关键**。

**§10.8 S2 护栏交付 + 静态分叉候选清单（comp-parity，2026-09-13 04:00）**：
- **交付**：`DSV41_COMP_PARITY=1`（默认 OFF，strict `=="1"`，与 `DSV41_TAP_PARITY` 同规）。入口 `dspark_commit`（chain_dev.rs:10277）——replay 之后、`set_pos_ctr` 之前（:10301），即正好夹住"replay 装好的那个状态"。探针 `comp_parity_probe`（:10577），取/放助手 `comp_state_take`/`comp_state_put`（:10472/:10502）。
- **它测什么**：把 SWALLOW 的**块快照+replay**状态（replay 是 **layer-major**：每层的 keep 行跑完再下一层，:10331）与**同块按 lazy 方式重放**的状态（**row-major**：每行过全部 compress source 再下一行，:10683）**逐位**比较（`to_bits`，不合并 -0.0/NaN）。两跑喂**同一份投影**（`spec_snap_kvp/scp`），所以报出的分歧在**状态机**（顺序、复原精确性、跨层活性、计数器），不在投影——投影那半是 S1（E1/E3）。
- **为什么非空转**：顺序是实体差异（layer-major vs row-major），且没有任何断言保证两者等价；replay 的正确性还依赖`dspark_rollback_keep(keep=m)`的复原精确性与"`state_kv`/`state_score`/`latent` 均按层私有、无跨层耦合"这一隐含前提。全部由这一跑直测。
- **tier-2 `proj-live`**：同一 take 里把 `spec_snap_kvp/scp[last]` 与**活的** `kvp_r/scp_r` 逐位比（最后一个 compress source 的块行在 commit 时仍是它自己的活值——层循环升序 `for layer in 0..n_layers` 已核实）。这直测 **replay 消费的快照 == 块自产出的逐行投影**（S2 的"来路"接线），且是**最后一个 compress source 上的 S1 采样探针**：若此处不等 ⇒ 该层的 m 行投影与逐行投影本身不同（S1 在压缩器输入端实锤）。
- **纪律**：take(状态+接线) → `dspark_rollback_keep(pos_base,m,m)`（`keep=m` 使 ring 复原循环为空 ⇒ 保住已接受前缀的 KV；压缩器复原无条件 ⇒ 整块回滚，:8108-8166）→ row-major 参考跑（`publish=false`，不碰 `index_k`）→ 读回 → **无条件 put-back**（三个 payload + device clen + out_rows + host mirror）→ 才比较/打印。replay 或下载失败一律在 put-back **之后**报 `Err`（照 `diff_eager_probe` 的 take/put-back 纪律）。`pos_ctr` 两跑都不写（pair 读 `pos_rows`）。ring 的 compressed rows 不复原：状态相等时参考跑在相同槽位写相同字节；不等时已打印 MISMATCH（诊断）。
- **成本/用法**：两趟全 source D2H + 每 spec round 多一次 keep 行重放。**只在独占进程跑，绝不进 A/B 计时臂**。启动收据新增一行 `[sh-gate] startup (diagnostic guards): COMP_PARITY=...`（env 回读纪律，§10.1）。
- **实现注记**：`compress_replay` 的每行体内提为 `compress_replay_row`（:10386）——launch/参数/顺序逐字不变（活路径逐位不变），护栏用 row-major 驱动**同一程序**，这是"非空转"的前提。

**§10.9 sweep1 完整判定（2026-09-13 04:05，四个新交付项的 A/B）**：
| 臂 | verify | 判定 |
|---|---|---|
| p3b（p3lite 段 B） | 25.41 | **无效**——R1 前置漏设 ATTN_PROJ_ALIGN=1（段 B 未发射；draft 3.79 无变化）；完整验证需 ALIGN 同臂 + tap 修复后重测 |
| woa4（wo_a nwarps=4） | 26.21 | **负向 +1.72ms** |
| woa2（wo_a nwarps=2） | 27.38 | **负向 +2.89ms**（梯度确认 nwarps=8 已是局部最优） |
| mpar2（MPAR 回炉 auto） | 25.77 | **负向 +1.28ms——MPAR 二连败**（rpb=1 +0.52 → 回炉后 +1.28；warp-M-并行的 1.59× 指令代价 > 收益，结构性天花板确认） |
| fence（fence 修复验证） | 24.62 | **✓ decline 消失验证通过**（无 "left no device snapshot" 行 + ARMED 全打 + mean-k 1.380 健康；只解锁 layer 20 一层故 +0.1ms 内符合预期） |

**步时优化的战略判定**：四个新交付项三负一无——**剩余有效路径**：①tap 修复（+0.78 acc = +33% 吞吐——最大杠杆，TAP_PARITY/COMP_PARITY 护栏已交付待跑）②MoE grouped（−1.5-2.5ms，716 修复中）③mrows 摊薄的真解 = tensor core 路线（m=6 fp8 mma——设计进行中）④draft 减肥（p3lite 全套——tap 修复后重测）。

**静态逐项比对（`compress_replay`+`compress_replay_row` :10331/:10386 vs `compress_row` :13675）——S2 分叉候选清单**：

| # | 候选点 | 位置（file:line） | 判定 |
|---|---|---|---|
| C1 | 遍历顺序：replay **layer-major** vs lazy **row-major**（每行一次 `step_rows(m=1)` → 全层） | :10341-10362 vs :13675 / 层循环 :11737 | 每层状态私有，`pos_rows`/`clen`/`out_rows` 均按层切片 ⇒ 理论上顺序无关；**无断言**——**tier-1 直测此点** |
| C2 | 投影来源：`spec_snap_kvp/scp`（replay）vs 活 `kvp_r/scp_r`（lazy 每行） | 写 :13638-13652（**仅 `spec_capture` 下写**）；消费 :10386 / :13675 | 同层同 stride（`layer*VERIFY_ROWS*hd + r*hd`）；**tier-2 proj-live 直测**（限最后一个 source） |
| C3 | 位置双源：`start_pos`（host `pos_base+r`）与 `pos_ctr`（device `pos_rows[r]`）是同一行的**两个独立来源** | pool 调用 :10414-10417 / :13695-13698；仅 row 0 有断言 `inv_pos_rows_first` :11177 | replay 重传 `pos_rows`（:10339）自洽；**m 行块 r>0 无断言** ⇒ 结构化候选 |
| C4 | **S3**：`comp_side` 被 `spec_capture` 排除 ⇒ SWALLOW 的投影走主 stream，lazy 可走 `side_stream3` | :11935-11942（+ fork/join :11946, :11933） | 同 kernel 同参数、仅 stream 不同 + `compress_side_join` ⇒ 数值应同，但**两栈的投影路径结构性不同源** |
| C5 | `publish_index_key` 位置：replay 内联在每行 commit 后 vs lazy 由调用方在 commit 与 select 之间 | :10450-10458 vs :11880 区 | `index_k` 不在护栏比较范围（护栏显式跳过 publish）；`clen` 按同一确定性规则推进 ⇒ 等价 |
| C6 | ring 的 compressed rows（`window+*clen`）不被 rollback 复原，由 replay 重写 keep 行 | :8108（只复原 window 行） | `clen` 复原后 `>= clen` 的槽不可达（`sparse_attn` 只读 `< clen`）⇒ 等价（已文档化） |
| C7 | m 行 hoisted `compress_rows_fused`（`fused_mrows` 单 launch）vs m 次 pair | :13782 vs :10386 | **本护栏不覆盖**（m≥2 字节被 replay 覆盖，§10.5 已判不泄漏）；需"pair-vs-fused"第三腿另立 |
| C8 | 单行 decode 真身 `compress_on`（fused，读 `cache.kvp/scp`，counter 用活 `pos_ctr`）vs pair（读 `s.kvp_r/scp_r`，counter 用 `pos_rows+r`） | :17500 vs :13675 | 两个不同 scratch + 两个不同 counter 源；`compressor_fused_on` "与 pair 逐位相同"仅为**文档断言、无护栏** |
| — | 层集合：`compress_sources()`（:7814-7818）vs `is_comp_src`（:11713） | — | 同一谓词（`compress_ratio>0 && is_kv_source`）⇒ **已核实等价** |

- **判决读法**：`[comp-parity] IDENTICAL` ⇒ S2 的**状态机**干净（顺序/复原/计数器/活性全等价）⇒ 剩余嫌疑回到 **S1（投影本身）**，E1/E3 正身；`MISMATCH layer/seg/row/elem` ⇒ S2 实锤，按坐标定位。
- **编译/测试**：`cargo check --workspace --all-targets` **EXIT=0**；`cargo test -p ferrite-models --lib` **92 passed / 0 failed**。`cargo test --workspace --lib` 有两处**与本改动无关的既存失败**：① `ferrite-exec` lib-test 链接失败（dev profile 未链 CUDA driver，`cudaSetDevice`/`cudaMalloc` undefined——`ferrite-kernel/cuda.rs`，非本次触碰）；② `ferrite-model::weights::tests::layout_production_counts` 断言 38287 vs 37398（该文件 `git diff` 为 0 行、纯 config 计数、与本改动不同 crate）。

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
