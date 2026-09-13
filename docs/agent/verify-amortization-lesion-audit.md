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
