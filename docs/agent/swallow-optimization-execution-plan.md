# SWALLOW 优化综合执行计划 —— 所有设计的单一时间线（从 58.3 到 400）

> 工部 · 2026-09-12 · **只读勘察 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 整合输入（现场核对）：
> - `swallow-ar-first-step-design.md`（AR 第一步：身份 + 三段账 + 四分支）
> - `mrows-swallow-batched-implementation-design.md`（mrows 族 S2：3 接线 + 3 新核）
> - `ar-step2-a1a-moe-store-implementation.md`（A1a MoE store fold 实施）
> - `swallow-unlocked-400-final-path.md`（400 最终路径：阶梯 + 兑现率门槛）
> - 旁证：`hc-chain-bandwidth-analysis.md`（hc A1/A2）· `b6-mrows-f32-design.md`（B6）·
>   `expert-tcgen05-plan.md` / `tcgen05-e4m3-grouped-expectation.md`（tcgen05）·
>   `dspark-correctness-chain.md`（尾部 nsys 表 + AR Step 2 测试记录）·
>   `scripts/batched_400_v2.sh`（权威 GATES/FORBIDDEN）。
> 代码基线 = 工作树 HEAD（clean，最新提交 `0b103a8`）。
> **口径纪律**：每条 ms/μs 标来源（**实测** / **账本** / **设计** / **代数** / **nsys**）；
> 行号漂移处以**函数名 + gate 名**为准。

---

## 0. 状态校准（先读八条。前四条**修正任务前提**，须上报）

1. **❗ AR Step 2 不是「GPU 验证中」，它已经验证完毕且失败。**（这是本计划与任务前提的**最大冲突**）
   提交 `0b103a8` 明文记录了 SWALLOW 上的 A1a 测试结果：**7.1 tok/s（vs 58.3 基线 = 8× 退化）**，
   且**前 61 行不再全对**（输出退化，数值不中性）。`LEN=228 零拉丁 ✓ / 0 panic / 0 ar5-hang ✓`。
   ⇒ `DSV41_AR_STORE_FUSE` **必须保持 OFF**（默认即 OFF，安全）。
   **后果**：`dspark-correctness-chain.md:6214-6219` 的「+AR Step 2 → 24ms → 250 tok/s」预测
   **已被实测证伪**；本计划不再把「AR −3~5ms」计入任何预算，改为一条**待收敛的回归**（G0）。
   【实测：`dspark-correctness-chain.md:6221-6236`】

2. **❗ 「AR 36% = 10.1ms/步」是一次错误的乘法。** 正确口径：AR 每步 = **84 轮 × 78.3μs = 6.58ms**。
   36% 是 **kernel-sum 窗口内的占比**，不是步时占比。自洽核对：`72.3ms ÷ 0.36 = 200.8ms` 可见总量
   ÷ 11 步 = **18.3ms kernel/步**（28ms 墙钟 − 18.3 ≈ 10ms 是 gap/宿主/prefill）。
   84 轮是结构常数（`chain_dev.rs:1894`）；`924 实例 ÷ 84 = 11 步`。
   【nsys 表 + 代数：`swallow-ar-first-step-design §0-1`】

3. **❗ AR 的「身份」尚未钉死，不查清不能投 AR。** nsys 表写 `p2p_ar_pubred_v5_hcpost`，
   而该 run 自己的配置写 `AR_V5=0`；`ar_v5() = GRAPH_STEP ∥ AR_V5`（`tp.rs:1027-1047`）
   ⇒ 两腿全 0 时**根本不上场 v5**，走 host-barrier 伪路径（`ar_store`/`ar_stamp`/`ar_reduce`）。
   二者只能有一个是真的。**这是投 AR 之前的硬前提（G1 的一部分）。**
   【读码 + 文档矛盾：`§0-2`】

4. **❗ mrows S2 不是「翻转 6 个 gate」的动作。** 按代码事实是两类：
   * **零代码接线**（kernel 已在树）：B1 / B2 / B3；
   * **新核**（设计已备、kernel 不存在）：B4 / B5 / B6。
   且 **B1 ⊂ B2**（K2 上场后 `q_norm_fused` 短路 B1 的整个调用点）⇒ **收益不可相加**；
   6 个 mrows gate（`SH_EXP/GATE/INDEXER/NORM/COMPRESSOR/VERIFY_HEAD_MROWS`）**已在权威配置里**
   ⇒ 58.3 基线**已经带着它们**，不计入 S2 增量。
   ⇒ S2 的现实票面 = **−640 发/步 = −2.1ms（保守）~ −4.0ms（全账），中位 ≈ −3.0ms**（不是 −4.5）。
   【读码：`mrows-swallow-batched-implementation-design §0-1/§0-2/§0-3`】

5. **❗ hc 链的 gate 与权威脚本的 FORBIDDEN 直接冲突。**
   `scripts/batched_400_v2.sh:175` 明写 `FORBIDDEN="DSV41_LAZY_VERIFY DSV41_HC_VERIFY_FUSE DSV41_HC_FRONT_ROWS"`，
   而 hc 链的设计增量（−1.3~1.8ms）**恰恰要开 `HC_VERIFY_FUSE` + `HC_FRONT_ROWS`**。
   ⇒ 上 hc 之前，**必须给脚本加一个 opt-in arm**（照 `tcgen05 e4m3 grouped` 的先例，`:177-190`），
   否则「hc 链」在权威矩阵里**永远测不到**。这是执行计划里一个**隐藏的前置件**。

6. **⚠️ 「58.3 tok/s」与「28ms 步时」不在同一把尺子上，且 400 只在 accept≥5 下成立。**
   `58.3 ÷ (1000/28) = 1.63 tok/步` ⇒ 隐含 accept ≈ 0.63（真实 prompt 的 e2e 口径）。
   而 400 的代数（`400-final-path §0-6`）是：`400 = 1000·k_emit/步时`，
   `accept 5 ⇒ k_emit 6 ⇒ 步时 ≤ 15.0ms`；`accept 3 ⇒ ≤ 10.0ms`；`accept 1.214（出师表）⇒ ≤ 5.54ms（物理不可达）`。
   ⇒ **400 的目标口径 = 计数 prompt（accept 5）**；「58.3 tok/s」不能直接线性外推到 400。

7. **⚠️ tcgen05 的 −6.8ms 是「swapAB 全 routed（gate/up + down）」的票面，现有 arm 只换 gate/up。**
   仓内**没有 tcgen05 down kernel** ⇒ 现实增量 **−1.0~3.8ms（mid 2.4）**。
   Phase 0 代码已落地、**GPU 未跑**；TMA `cp.async.bulk` 的 16B 硬对齐**两轮修复失败**
   ⇒ tcgen05 是一个 **go/no-go 未定**的项，不能按「已备好」排队。
   【读码：`tcgen05-e4m3-grouped-expectation §0`、`tcgen05-tma-bulk-align-design §0`】

8. **⚠️ 本机没有 `.so`。** `ls kernels/cuda/*.so` = No such file ⇒ 所有产物级验证（`nm -D`、
   kernel 名、Instances 计数）**必须在远端 `build.sh 103a` 之后**做。本机只能做源码级勘察与写码。

---

## 1. 依赖图与时间线总表

```
T0  现在：58.3 tok/s / 步时 ~28ms（全 gate）/ AR 身份未钉死 / A1a 回归已实测
 │
 ├──【G0】AR Step 2（A1a）回归收敛 ──── 修 bug 或判弃（gate 已 OFF，安全）
 │
 ├──【G1】AR Step A：身份钉死 + 三段账（探针）── 决定 AR 的四条分支（T1~T4）
 │        └─ 附带 0.5 人日诊断件：探针 site 分流（可不写，先用总量走分支）
 │
 ├──【G2】S0 基线实测钉死（3×run + 1 长跑 + ledger 逐 rank 对账）
 │
 ├──【E1】SH_PAIR M=6 ──────────────── 第一优先（最大单项、arm 已编译、唯一把 M 进 grid）
 │
 ├──【E2】mrows S2 ── Phase A：B2 → B3（零代码，一 gate 一 serve）
 │                  └ Phase B：B6（写码，**不占 GPU**）→ B5 → B4（新核）
 │
 ├──【E3】hc 链 A1+A2（**前置：给脚本加 opt-in arm 破 FORBIDDEN**）
 │
 ├──【E4】tcgen05 go/no-go（**前置：TMA 16B 对齐构造性修复**）
 │
 └──【E5】L4/L5 占用与 MLP（条件触发：仅当 S5 实测 >15ms 且 accept ≥3）
```

**并行度**（这是本路线图的关键——不是串行）：

| 轨道 | 内容 | 占 GPU？ |
|---|---|---|
| **GPU 轨** | G0/G1/G2 → E1 → E2-PhaseA → E3 → E4 | 是（互相抢卡，须交错排程） |
| **写码轨** | E2-PhaseB 的 B6（0.5 人日）/ B5 / B4 + E3 的 FORBIDDEN arm | **否**（可与 GPU 轨并行） |
| **诊断轨** | §1.5 探针 site 分流（0.5 人日） | 否（只在 B 臂同轮跑） |

> ⚠️ **一次只跑一条 GPU 臂**（`OnceLock` 每进程只读一次 gate；多 gate 同开 = 失去归因）。

---

## 2. 逐项施工单（序 / 依赖 / 预期 / 验证 / 风险）

### G0 —— AR Step 2（A1a）回归收敛 · **前置：无** · 成本 0.5~2 人日 · 风险 中

| 项 | 内容 |
|---|---|
| 现状 | +665 行已落树（6 文件、gated OFF）。GPU 实测：**7.1 tok/s，前 61 行退化** |
| 依赖 | 无（可立即做，且**不占 GPU 的代码复审优先**） |
| 可能根因（读码假设，待证） | ① 载体「last writer」论证在 batched 路径不成立（`ar_store_fuse_moe` 的 `shared_here` 判定与 `moe_rows` 实际写入者错配）；② store 在捕获段内、publish 在段外 ⇒ **流序错位**（8× 退化更像「同步被破坏」而非「多算」）；③ ADD_EPI/A5 覆盖缺口的 rank 混用两条路径 |
| 判据 | `DSV41_AR_STORE_FUSE=1` vs `=0`：① 逐 token 一致；② 步时 ≤ 基线；③ `p2p_ar_store_v5_kernel` Instances 下降。**三条全过才算修好** |
| 止损 | 2 人日无定位 ⇒ **判弃**，A1a 永久 OFF，AR 走 §G1 的 T3/T4 分支（不碰 store） |
| 为什么排最前 | 它同时是「一个已经烧过一次 GPU 会话的坑」和「AR 分支里唯一已实现的优化」。不收敛它，AR 的预算没法编 |

### G1 —— AR Step A：身份 + 三段账 · **依赖 G0 或 G0 判弃** · 成本 0.5 人日(+0.5 可选) · 风险 **0**

| 项 | 内容 |
|---|---|
| 目的 | 一次会话同时回答：(i) AR 是**工作**还是**等待**；(ii) 走哪条优化分支；(iii) 两个零代码项是否立刻兑现 |
| 四臂（同一 binary + 同一 `.so`） | **A** 参照 / **B** `DSV41_AR_PROBE=1`（三段账）/ **C** `=B + DSV41_AR_SINGLE_POLL=1`（960→8 poller）/ **D** `=C + DSV41_AR_STORE_FUSE=1`（**仅当 G0 修好才跑**，否则跳过） |
| 关键读数 | `[ar-probe] avg_stamp / avg_spin / avg_epi`（× site × **rank**）。B300 ~1.8GHz ⇒ 1μs ≈ 1800 cyc |
| 判据（决策树） | **T1** spin 大且 per-rank 不对称 ⇒ A2c 负重平衡；**T2** spin 大且均匀 ⇒ A4→A1a→A2d→PDL；**T3** spin 小但 `avg_epi ≫ 15μs` ⇒ AR 是工作绑定（转 fold/reduce 向量化，**不碰协议**）；**T4** spin≈地板 且 epi 正常 ⇒ **nsys 36% 是放大伪影** ⇒ 停 AR，转 MoE/投影/mrows |
| 硬前提 | 探针轮**禁 nsys**（nsys 把 v5 自旋放大 ~300×）；MAXTOK ≥120（探针每 512 轮/site 才打印一行，84 轮/步 ⇒ 需 ≥7 步出首行）；`DSV41_AR_TIMEOUT_TRAP=1`；`nm -D` 必须含 `ferrite_p2p_ar_v5_hcpost_add`（否则 site 翻倍） |
| 通过线 | 每 site ≥1 行 × 8 rank；`ar5-hang=0`；计数红线绿；`0 panic`。**缺证据不得下结论（exit 2）** |
| 身份钉死 | 独立 **nsys 计数轮**（`--trace=cuda --cuda-graph-trace=node`）：读**完整 kernel 名** + `Instances/步 ≈ 84`（40 attn + 40 MoE + ~4 engram）。**该轮 μs/占比一律不读** |

### G2 —— S0 基线实测钉死 · **依赖 G1（同会话顺带做）** · 成本 ~0.2 人日

| 项 | 内容 |
|---|---|
| 目的 | 把任务前提的「28ms」钉成有来源的实测值（400-final-path 的 S0=31ms 与 28ms 差 3ms，恰好跨过 400 判决线） |
| 做法 | 3× 独立运行 + 1 长跑；`ledger off 取吞吐 / on 取 ledger`；**一 prompt 一 serve**；`V5_LEDGER=0`（吞吐轮） |
| 判据 | 每一步的阶梯都以此为基准；`|Δsteady_median|` < 基线波动带（先测 3 次求带）则视为「未兑现」 |

### E1 —— SH_PAIR M=6 · **依赖 G2** · 成本 0（arm 已编译）· 风险 低

| 项 | 内容 |
|---|---|
| gate | `DSV41_SH_PAIR_M=1`（默认 OFF）+ `DSV41_SH_PAIR_M_FOLD` |
| 落点 | kernel `dsv41_kernels.cu:7617`（`gemm_fp8_sh_exp_pair_kernel<M>`）/ launcher `:7944`；gate `chain_dev.rs:1470/:1486`，调用点 `:13511` |
| 预期 | **−4.9~7.9ms（mid 6.4）**——shared expert 1000→80 发/步。**最大单项** |
| 验证 | ① `.so` 重建后 `nm -D`；② nsys 出现 `gemm_fp8_sh_exp_pair_kernel<6>`（**非 <1>**）；③ 步时位移；④ 计数红线 / 出师表零拉丁 |
| 备注 | 脚本当前矩阵**不含** SH_PAIR_M（`:145-156`）⇒ 解锁后基线是「无 SH_PAIR」，这是**纯增量** |

### E2 —— mrows 族 S2 · **依赖 G2（可部分并行写码）** · 成本 0（3 项）+ ~1.5 人日（3 新核）

**Phase A（零代码，一 gate 一 serve）**

| 序 | 项 | gate | 省发/步 | 预期 launch 账 @3.3μs / @6.2μs | 关键陷阱 |
|---:|---|---|---:|---|---|
| 1 | **B2（K2）** | `DSV41_ATTN_MROWS_ROPE_NORM=1` | −280 | **−0.92 / −1.74ms** | `qr_norm_out` **必须传 null**（否则 9.2 tok/s cliff）；写回交给后续 `norm_rows` |
| 2 | **B3（K1）** | `DSV41_ATTN_MROWS2=1` | −40 | −0.13 / −0.25ms | 与 R2 单行臂无冲突；K1 排在 `lin2` 之前 |
| 3 | B1（fallback） | `DSV41_VERIFY_ROPE_MROWS=1` | −200（**B2 后 = 0**） | −0.66 / −1.24ms（B2 前） | **不与 B2 同轮**（B2 上场后整段短路） |

**Phase B（新核，写码不占 GPU）**

| 序 | 项 | 省发/步 | 预期 | 备注 |
|---:|---|---:|---|---|
| 4 | **B6** `dsv41_gemm_fp8_mrows_f32` | −240 | −0.79 / −1.49ms | **非位等价**（跳过 fp8 往返，更准）⇒ 走**红线**验收；新增 `a_stride` 形参；0.5 人日，**可与 E1/E3 的 GPU A/B 并行** |
| 5 | **B5** `gemv_bf16_v2_mrows_route` | −40 | −0.13 / −0.25ms | 仿 EAGER 的 m=1 版；`ctr` 须 4B zeroed ONCE |
| 6 | **B4** `dsv41_rmsnorm_rope_mrows` | −40 | −0.13 / −0.25ms | **必须接 `kv_stream`**（否则静默把 kv 半链拖回主流，抵消 FORK）；A/B 须与 `VERIFY_FORK` 同臂 |

**族合计（不重复计账）**：`−640 发/步 = −2.1ms（保守）~ −4.0ms（全账），中位 ≈ −3.0ms`。
**天花板** = nsys「gemv 投影」族的 **4.2ms** ⇒ **账与天花板重合、无余量**：
任何一项的静默 decline 都会把对应份额直接吃掉 ⇒ **每项必须有 nsys 的 kernel 名证据**。

### E3 —— hc 链 A1 + A2 · **依赖 G2 + （前置）脚本 opt-in arm** · 成本 0（接线已落）+ 0.5 人日（arm）· 风险 低~中

| 项 | 内容 |
|---|---|
| 现状 | A1/A2 **已落树**（`chain_dev.rs` 的 `collapse_norm_rows` / `hc_post_rows` / `hc_mixes_auto(rows)`），但 gate 在 FORBIDDEN 里 |
| 根因（推翻既定判断） | verify 段**根本没走任何 hc 融合**——`layer_rows()` 走的是**原始 10 发链**（5 个 C 入口 × 2 侧）。400 发/步、2.96ms 全在这里 |
| gate | `DSV41_HC_VERIFY_FUSE=ON`（A1 总闸）+ `DSV41_FUSE_B1/FUSE_C=ON`（内层）+ `DSV41_HC_FRONT_ROWS=OFF→ON`（A2 总闸） |
| 预期 | 2.96 → 1.2~1.4ms（**−1.6~−1.8ms**）；launch 400 → 160/步；**bit-exact** |
| 前置件 | ① 脚本加 opt-in arm（破 FORBIDDEN 的正确做法，不是删 FORBIDDEN）；② A1-b 的 `hc_post_inplace(rows>1)` 需一次 `tests/hc_parity` 对拍 |
| 验证 | ① nsys 出现 `hc_collapse_norm` / `hc_post_inplace` / `hc_mixes_tail_kernel` 的**多行 grid**（`blockIdx.x = r`）；② hc 5 核的 Instances/步 从 400 掉到 ≤160；③ `k_acc` 逐位不变 |
| ⚠️ 历史 | `HC_VERIFY_FUSE` 曾在 `1ddff9c` 因拉丁回归被判死；`chain_dev.rs:14104-14106` 注明「默认 0，理解交互后再 re-arm」。**⇒ 这是「已知坑」，比新核优先级高但要带红线复验** |

### E4 —— tcgen05 go/no-go · **依赖 TMA 对齐修复** · 成本 4~5 人日 · 风险 **高**

| 项 | 内容 |
|---|---|
| 现状 | Phase 0（真实权重 + 数值 parity 套件，1565 行）**代码已落、GPU 未跑**；TMA `cp.async.bulk` 16B 硬对齐**两轮修复失败**（`byte-fallback` 原理无效） |
| 根因设计 | 修复形态 = **把 16B 从 launcher 运行时门提升成布局的构造性不变量**（源与目的构造时对齐）——不是「再加 fallback」 |
| 门现状 | **「四缺三」**（`tc5::e4` 完整；其余三处不完整） |
| 预期 | **−1.0~3.8ms（mid 2.4）**（仅 gate/up；**down 无核**） |
| go/no-go 判据 | Phase 0 parity 全绿 **且** TMA 对齐修复使门恒真 ⇒ 才进 Phase 1 |
| 止损 | 对齐修复第 3 轮仍失败 ⇒ **判 no-go**，MoE 走 SIMT 优化（`ferrite_gemv_bf16_v2_mrows_route` 一类的 launch 收敛） |
| 备注 | 权威脚本**已有 opt-in arm**（`:177-190`），这是 E3 破 FORBIDDEN 的模板 |

### E5 —— L4/L5 占用与 MLP · **条件触发** · 成本 16~21 人日 · 风险 高

| 项 | 内容 |
|---|---|
| 触发条件 | **仅当** S5 实测 >15ms **且** accept ≥3 |
| 预期 | **−5~8ms（mid 6.5）**——仓内**零实测背书** |
| 意义 | 唯一能把 **accept 3 拉进 400** 的层（→ 8ms ⇒ 500 tok/s）。accept 5 路径上不必先做 |

---

## 3. GPU 验证清单（一张表 · 每项最小证据集，缺一不算完成）

| 项 | ① 符号/env | ② 上场证据（唯一硬证） | ③ 位/数值 | ④ 性能 | ⑤ 红线 |
|---|---|---|---|---|---|
| **G0 A1a 回归** | `nm -D` 含 `ferrite_add_store` / `dsv41_moe_down_reduce_st` / `...pubred_v5_moe` / `..._hcpost` | nsys `p2p_ar_store_v5_kernel` **Instances 下降** | **ON vs OFF 逐 token 一致**（当前 = 失败） | 步时 ≤ 基线（当前 7.1 vs 58.3 = 失败） | 计数前 61 行 + `ar5-hang=0` |
| **G1 探针三段账** | `/proc/$(pgrep -x ferrite-serve)/environ` 回读 `DSV41_AR_PROBE=1` | `[ar-probe]` 每 site ≥1 行 × **8 rank**、`n ≥ 512` | — | — | **该轮无 nsys** + 计数红线 + `ar5-hang=0` |
| **G1 身份** | `nm -D` 符号表 | nsys **计数轮**的**完整 kernel 名** + `Instances/步 ≈ 84` | — | **不读 μs/占比** | — |
| **G2 基线** | — | — | 3 run 一致 | `steady_median`（丢前 10 步、skip=20）+ 波动带 | 计数 + 出师表零拉丁 |
| **E1 SH_PAIR** | `.so` 重建 + `nm -D` | nsys 出现 `gemm_fp8_sh_exp_pair_kernel<6>`（**非 <1>**） | 计数红线 | 位移 ≥ −0.8ms（止损线 = 设计 40%） | 出师表零拉丁 |
| **E2-B2 K2** | `nm -D` 含 `dsv41_gemm_fp8_mrows_rope_norm` | nsys 出现该 **kernel 名**（decline 是**静默** `Ok(false)`）；`proj_mrows` fp8 GEMV Instances/步 40 → 0 | `k_acc` **逐位不变** + 计数前 61 行 | `[dspark] verify_ms` / `steady_median` | 计数数字顺序 |
| **E2-B3 K1** | `nm -D` 含 `dsv41_gemm_fp8_mrows2` | nsys 出现该核；fp8 GEMV Instances/步 80 → 40 | `k_acc` 逐位 | 同上 | 同上 |
| **E2-B6** | `.so` 重建 + `nm -D` | nsys 出现 `gemm_fp8_mrows_f32_kernel<6>` 且 `dsv41_quant_fp8` 的 **wo_b 调用数归零** | **红线**（非 memcmp）：`k_acc` mode 不降 + 出师表零拉丁 | 位移 | 上左 |
| **E2-B5** | `.so` 重建 + `nm -D` | nsys 出现 `ferrite_gemv_bf16_v2_mrows_route`（单核）且 `route_topk` 40 → 0 | `k_acc` 逐位 | 位移 | 上左 |
| **E2-B4** | `.so` 重建 + `nm -D` | nsys 出现新核；kv 侧 `apply_rope_kernel` 40 → 0 | `k_acc` 逐位 | 位移（**与 `VERIFY_FORK` 同臂**） | 上左 |
| **E3 hc 链** | 脚本 opt-in arm + `/proc/environ` 回读两 gate | nsys hc 5 核 Instances/步 **400 → ≤160**；`hc_mixes_tail_kernel` grid 带行维 | **对拍** `tests/hc_parity`（A1-b）+ `k_acc` 逐位 | 位移 ≥ −0.5ms | 出师表零拉丁（**历史回归点**） |
| **E4 tcgen05** | `nvcc -gencode arch=compute_103a,code=sm_103a` | Phase 0 parity：`max|diff|/max|ref| < 5e-2` ×4 case ×2 SF 布局 | parity 全绿（cpu golden 两路） | 位移 | 计数 + 出师表 |

**通用纪律（每轮必守）**：
1. **一 gate 一 serve**（`OnceLock` 每进程只读一次）。
2. **一 prompt 一 serve**（`[dspark] steps=` 累加器跨请求不清零）。
3. **交错 A B A B**（抵消热漂）。
4. 吞吐轮 **`V5_LEDGER=0`**（10 D2H/步会吃掉全部位移）；探针轮**禁 nsys**。
5. **禁** `LAZY_VERIFY` / `SEED_ALIGN`（抢臂）；hc 两 gate 需先破 FORBIDDEN。
6. 每次 A/B 必录：**逐 token 一致 + 计数红线 + `ar5-hang=0`**；AR 改动**不得改轮数**（84 真 + 81 pad = 165，改动后用 `[dspark] steps=` epoch 增量复核）。
7. **止损线**：单项 |Δ| < 设计增量 40% ⇒ 判 instruction-bound，立刻转下一项（`SH_EXP_MROWS` 两次零收益先例）。

---

## 4. 400 的完整执行计划（58.3 → 400）

### 4.1 阶梯（以 G2 实测 S0 为准；下表按任务前提的 28ms 演算，若 G2 得 31ms 则整体右移）

| 阶段 | 项 | Δ(mid) | 累计步时 | accept 5 隐含 tok/s | 来源 |
|---|---|---:|---:|---:|---|
| **S0** | 全 gate 基线（含 6 mrows gate + 图 + SWALLOW，**无 SH_PAIR**） | — | **28.0ms** | **214** | 实测（待 G2 复核） |
| **S1** | + SH_PAIR M=6 | −6.4 | 21.6ms | 278 | launch 账 |
| **S2** | + mrows 族（B2+B3+B6+B5+B4） | −3.0 | 18.6ms | 323 | launch 账（中位） |
| **S3** | + hc 链（A1+A2） | −1.5 | 17.1ms | 351 | 带宽分析 |
| **S4** | + B6 独立计入 | −1.08 | 16.0ms | 375 | 第一性计数 |
| **S5** | + tcgen05（gate/up） | −2.4 | **13.6ms** | **441** | 修正口径 |
| ★ | **400 判决点** | — | **≤15.0ms ⇒ ✓** | — | 代数 |
| **S6** | + L4/L5 | −6.5 | 7.1ms | 845 | 零背书（条件） |

> **注**：S2 已含 B6 ⇒ S4 的 B6 不应重复计账。**不重复计账的阶梯**是：
> `S0 28.0 → S1 21.6 → S2(+mrows 含B6) 18.6 → S3(+hc) 17.1 → S4(+tcgen05) 14.7 → 判决 ✓`
> （上表把 B6 单列仅为与源文档对齐；执行时**按不重复口径**算。）

### 4.2 兑现率门槛（代数，机械计算）

```
400 @ accept 5 ⇒ 步时 ≤ 15.0ms
S0 = 28.0ms（实测）⇒ 需兑现 Σ_cash = 13.0ms
Σ_design(S1+S2+S3+tcgen05) = 6.4 + 3.0 + 1.5 + 2.4 = 13.3ms
⇒ 所需兑现率 = 13.0 / 13.3 = 97.7%   ← 近乎全额
若 S0 = 31ms（400-final-path 口径）⇒ 需 16.0/13.3 = 120%  ⇒ 本路线不可达，必须叠 L4
```

**60% 兑现**（历史值）⇒ `28.0 − 8.0 = 20.0ms ⇒ 300 tok/s ⇒ 差 25%`。

### 4.3 两条硬约束（不可协商）

1. **accept 是硬门槛，不是可选项。** `accept 3 ⇒ 步时 ≤10.0ms`，即便 S1–S5 全足额也只有 **276~300 tok/s**。
   ⇒ **400 是「counting 口径（accept≥5）+ 全足额」的目标**；accept 3 的 400 属于 **L4 之后**。
2. **AR 是协议地板**（账本 17.3μs/轮 × 84 = 1.45ms/步，5%）⇒ **减少 AR 轮数的唯一路 = accept↑**，
   不是 kernel 微优化（AR 天花板只剩 ~1ms/步）。这也再次说明 **G1 必须先做**：
   在 AR 身份未钉死前投 AR，等于在错误的账上编预算。

### 4.4 判决树（每步都是 gate，失败即停）

```
T0 现在 58.3 tok/s / 28ms
 ├─ G0 A1a 回归：修好 ⇒ 进 G1 的 D 臂；判弃 ⇒ A1a 永久 OFF，跳过
 ├─ G1 探针：T2 ⇒ 走 A4→A1a→A2d→PDL；T4 ⇒ 停 AR，预算全给 E1/E2
 ├─ G2 基线钉死（3×run）
 ├─ E1 SH_PAIR M=6   → 若 |Δ|<2.6ms（设计 40%）⇒ 记录并继续（它是最大单项，不做止损放弃）
 ├─ E2 mrows Phase A（B2→B3）→ 若均 <40%设计 ⇒ 判 instruction-bound，Phase B 减半
 ├─ E3 hc 链（先加 opt-in arm）→ 红线复验（历史回归点）
 ├─ E4 tcgen05 go/no-go → parity + TMA 对齐；no-go ⇒ 转 SIMT
 ★  400 判决：实测 ≤15.0ms ⇒ ✓（@accept 5）；>15ms ⇒ 触发 E5/L4
 └─ E5 L4（16~21 人日，零背书）——唯一能把 accept 3 拉进 400 的层
```

---

## 5. 下一 session 的**前 30 分钟**（读什么 / 跑什么 / 决定什么）

> 目标：**不浪费 GPU 会话**——先用 10 分钟把「设了没生效」的坑排掉，再上卡。

### 0–10 min · 读（只读，无 GPU）
1. 本文件 §0（八条校准 + §4.2 门槛）——**这是决策上下文**。
2. `swallow-ar-first-step-design.md` §0 + §1.2/§1.3（四臂矩阵与三段读数）。
3. `dspark-correctness-chain.md:6221-6236`（G0 的失败记录）。
4. 若要做 E3：`hc-chain-bandwidth-analysis.md` §7（gate 清单）+ `scripts/batched_400_v2.sh:175`（FORBIDDEN）。

### 10–20 min · 跑（**不上 GPU 也要做**，防幻影门）
```bash
# ① 本机无 .so —— 确认必须在远端构建
ls kernels/cuda/*.so || echo "NO SO → 远端 build.sh 103a"

# ② 关键源码自检（零成本，先做）
grep -n "DSV41_AR_STORE_FUSE" crates/ferrite-models/src/dsv41/chain_dev.rs   # 默认 OFF?
grep -n "AR5_TIMEOUT_TRAP\|DSV41_AR_PROBE\|DSV41_AR_SINGLE_POLL" crates/ferrite-models/src/dsv41/*.rs
grep -n "FORBIDDEN" scripts/batched_400_v2.sh                                 # 确认 hc 仍被禁

# ③ 远端（有 GPU 的机器）——双产物纪律
cd kernels/cuda && bash build.sh 103a
nm -D libferrite_kernels.so | grep -cE 'ferrite_p2p_ar_pubred_v5$|ferrite_p2p_ar_pubred_v5_moe|ferrite_p2p_ar_pubred_v5_hcpost'
nm -D libferrite_kernels.so | grep -cE 'ferrite_add_store|dsv41_moe_down_reduce_st|ferrite_p2p_ar_v5_hcpost_add'
```
**判据**：`ferrite_p2p_ar_v5_hcpost_add` **必须存在**——否则 `_hcpost` 会同时服务 ATTN 与 MoE，探针 site 翻倍。

### 20–30 min · 起一臂 · 决定
* 起 **G1 的 B 臂**（`DSV41_AR_PROBE=1` + `DSV41_AR_TIMEOUT_TRAP=1`），计数 prompt，**MAXTOK ≥120**。
* 三证回读：`tr '\0' '\n' < /proc/$(pgrep -x ferrite-serve|head -1)/environ | grep DSV41_AR_`。
* **30 分钟时的三个决定**：
  * **D1**：`.so` 符号齐否？（缺 `_hcpost_add` ⇒ 先补符号再测，否则 site 数据不可用）
  * **D2**：探针是否按 site × 8 rank 出数？（不出 ⇒ MAXTOK 太小 / gate 没生效）
  * **D3**：A 臂 `steady_median` 与 58.3 / 28ms 的口径是否一致？（不一致 ⇒ 先标口径，`56.6 tok/s` 就是一次口径混用的产物）

---

## 6. 决策点与仲裁请求（上报尚书省 / 用户）

| # | 决策 | 选项 | 建议 |
|---|---|---|---|
| **1** | **AR Step 2（A1a）修 or 弃？** | (a) 投 2 人日 root cause；(b) 判弃，永久 OFF | **先 (a) 但硬止损 2 人日**；2 人日无定位即 (b)。理由：它是 AR 分支唯一已实现项，但 8× 退化说明**协议级错位**嫌疑大 |
| **2** | **hc 链要不要破 FORBIDDEN？** | (a) 给脚本加 opt-in arm（照 tcgen05 先例）；(b) 暂不动 | **(a)** —— 不破 FORBIDDEN 的正确姿势是加 arm，不是删限制；且 `HC_VERIFY_FUSE` 有拉丁回归前科，**必须带红线复验** |
| **3** | **SWALLOW 常开 vs lazy⇄batched 路由？** | (a) 常开；(b) 按任务路由（Schmitt 滞回） | **(b)**。判据：`lazy 更好 ⟺ mean_k < B/c − 1`（`c = 6.15ms/row`）。B=28ms 时阈值 3.55 ⇒ **出师表/对话在 batched 下净亏**；优化本身会扩大 batched 适用面（B→15ms 阈值 1.44）。**这是产品决策（改 P50 分布）** |
| **4** | **tcgen05 go/no-go？** | (a) 投第 3 轮对齐修复；(b) 判 no-go 转 SIMT | 看 TMA 对齐修复的**设计可行性**（构造性不变量）再定；第 3 轮仍失败 ⇒ (b) |
| **5** | **400 的目标口径确认为 counting（accept≥5）？** | (a) 是；(b) 也要 accept 3 的 400 | **(a)** 为本路线；accept 3 的 400 **必须叠 L4**（+16~21 人日、零背书），列为 Tier 2 |

---

## 7. 一页纸摘要

1. **AR Step 2 已经失败**（7.1 tok/s，8× 退化）——它不是待兑现的 −3~5ms，是**一条待收敛的回归**（G0）。
2. **AR 的账必须先算清**：真实每步 **6.58ms**（不是 10.1ms），且**身份未钉死**（表 vs 配置矛盾）。G1 探针是**唯一**能切开「账本 1.45ms vs nsys 6.58ms（差 5.1ms）」的工具。
3. **mrows S2 的现实票面 = −3.0ms（中位）**，不是 −4.5；B1 被 B2 覆盖、B4/B5 靶子已被削半。
4. **hc 链有一个隐藏前置件**：它的 gate 在权威脚本的 FORBIDDEN 里 ⇒ 上 hc 前必须先加 opt-in arm。
5. **400 = accept≥5 + 近乎全额兑现（@S0=28ms 需 ~98%；@S0=31ms 需 >100% ⇒ 本路线不可达）**。
   60% 兑现 ⇒ 300 tok/s（差 25%）。**先钉死 S0 是 G2 的全部意义。**
6. **最便宜、先验最高的三件事**（同一次 GPU 会话可收）：
   `DSV41_AR_SINGLE_POLL=1`（960→8 poller）· `DSV41_AR_STORE_FUSE=1`（**仅当 G0 修好**）· `SH_PAIR M=6`（−6.4ms mid）。

---

*工部 · 只读勘察 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*所有 ms/μs 标来源（实测 / 账本 / 设计 / 代数 / nsys）；与任务前提冲突处已显式给出 file:line 依据（§0-1~§0-8）。*
