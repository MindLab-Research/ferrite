# SWALLOW 解锁后的 400 冲刺路线（第 10 次修复成功 ⇒ batched 解锁）

> 工部 · 2026-09-12 · **只读设计 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 情景：第 10 次修复（**真 epoch pad**，`dsv41_v5_epoch_pad_kernel` `ferrite_kernels.cu:9129`，
> wiring `chain_dev.rs:8633`，script gate `DSV41_SWALLOW_EPOCH_PAD=1` + `DSV41_V5_LEDGER=1`）
> 的 `batched_400_v2.sh` 测试正在跑，**结果 = 0 ar5-hang**。
> 输入（现场核对）：`swallow-unlocked-next-plan.md` · `batched-400-v2-prediction.md` ·
> `batched-400-v2-remaining-roi.md` · `batched-12-5ms-requirements.md` · `400-final-frontier-analysis.md` ·
> `400-fastest-path-roadmap.md` · `post-fix-400-roadmap.md` · `swallow-result-action-plan.md` ·
> `sh-pair-template-m-design.md` · `b6-mrows-f32-design.md` · `plan-b-swallow-readiness.md` ·
> `swallow-fix9-round-ledger-design.md` · `scripts/batched_400_v2.sh` · `chain_dev.rs` · `dsv41_kernels.cu`。
> **口径纪律**：每一条 ms 都标来源（实测 / launch 账 / 设计口径 / 代数）；修正任务前提处显式给出依据。

---

## 0. 判决（先读八条）

1. **解锁后第一步不是「上优化」，是「锁基线 + 证明解锁是真的」。** 0 hang **一次不算修好**
   （历史 gap 1→2→22→3 是竞态；第 9 次修复本身是**幽灵**——函数存在、零调用点、kernel/gate 全无）。
   判据是 **3 次独立运行 + 1 次长跑（跨过历史 hang 步数）+ ledger 逐 rank 对账**（§2 A1/A2）。

2. **叠加顺序（按「收益 ÷ 风险 ÷ 依赖」，不是按任务给的收益排序）**：
   ```
   B1 SH_PAIR M=6  →  B2 mrows 族  →  B3 hc 链  →  B4 B6  →  B5 tcgen05  →  [B6 L4 条件触发]
   ```
   SH_PAIR 是**唯一把 M 进 grid 的折法**（最大单项、parity 已确认虚警、零代码）；
   mrows 族是零代码但**兑现率存疑**（`SH_EXP_MROWS` 两次实测零收益 → instruction-bound 先例）；
   tcgen05 **当前仍 blocked**（冒烟 LEN=0 + misaligned，HEAD 刚上第 2 轮修复，**必须 go/no-go**）。

3. **任务表里第 3 行必须重新分类——lazy 的 +15.6% 不可直接继承到 batched。**
   `R2(ATTN_LIN_FUSE)` 是 **m=1 臂**，batched 下的对应物是 **K1/K2**（`ATTN_MROWS2` / `ATTN_MROWS_ROPE_NORM`），
   R2 在 batched 里要么 decline、要么与 K1/K2 竞争（`chain_dev.rs:2167`：两者都开时 K1 赢）；
   `LAZY_SDR` 是 **lazy-only**（名字即语义，batched 下 N/A）；只有 `MARKOV`（draft 侧）与
   `FORK/RING_WIN`（verify 侧）跨臂有效，但后两者会引入**侧流 → 翼分歧**风险，与图化同源。
   **这四项在 batched 下必须各自重新 A/B，不得按 lazy 的账直接记 +15.6%。**

4. **任务表第 2 行「mrows 族 B1-B6，kernel 已有」是两件不同的事。**
   - **已有的**是 6 个 mrows gate：`SH_EXP_MROWS / GATE_MROWS / INDEXER_MROWS / VERIFY_ROPE_MROWS /
     VERIFY_HEAD_MROWS / NORM_MROWS / COMPRESSOR_MROWS`（**零代码，已在脚本矩阵里**）；
   - **B1–B6 是另一族**（`routed-expert-residual` / `wo_b` 的 m-rows 融合），**除 B6 本体的设计文档外，
     `dsv41_gemm_fp8_mrows_f32` 尚未实现**（`b6-mrows-f32-design.md` 是设计，不是代码）。
   **二者不得相加**（会重复记账）。本路线把「6 gate」放 B2、「B1–B6」放 B4。

5. **400 的数学（代数硬约束，不可协商）**：
   ```
   τ/step ≥ 0.4 tok/ms
   accept 5（k_emit=6）⇒ 步时 ≤ 15ms
   accept 3（k_emit=4）⇒ 步时 ≤ 10ms
   accept 1.214（出师表）⇒ 步时 ≤ 5.54ms  ← 物理不可达（低于 L5 floor 8~9ms）
   ```
   ⇒ **400 是「counting 口径 + accept≥3」的目标**；且 **accept 是门槛**（τ<2.80 时即使步时压到 sglang 的
   verify 实测 7.3ms 也只有 303 tok/s）。

6. **诚实票面（设计口径 vs 60% 历史兑现率）**：
   - 设计口径全兑现：`S5（SH_PAIR+mrows+tcgen05+B6）≈ 14.5ms ≈ 414 tok/s` ⇒ **400 ✓**；
   - 60% 兑现（历史值）：`≈ 18~20ms ⇒ 300~340 tok/s` ⇒ **差 15~25%**；
   - **400 需要 B1/B2/B5/B4 四项同时足额兑现，且 accept ≥ 3**——仓史上前所未有。

7. **最大风险是「instruction-bound」不是「gate 没开」。** 该族 kernel 跑在 HBM 峰值 **0.7~4.9%**
   （shared expert 885MB/10.4ms = **85GB/s = 1.1% 峰值**），mrows 省的是「已经跑在 1% 峰值的时间」≈0。
   反向证据三条：`SH_EXP_MROWS` 两次零收益、`{SH_EXP+GRAPH+ROPE+P3A}` 全开只 **−1.21ms**（预期 −24）、
   v17→v21 四变体全中性。⇒ **每一个 gate 都必须单变量 A/B + 读 register 名确认变体真跑**，位移 <40% 即止损。

8. **红线判据必须用「计数数字顺序」（主探针），不是「零拉丁」。** 本 session 的核心教训：
   损坏的 base 就是「计数数字错但拉丁检查通过」。每一次 A/B 的判据 =
   **计数数字顺序正确 + 首 61 行 + `k_acc` 逐位 + 零额外拉丁**，四缺一不下结论。

---

## 0.5 现场钉死：第 10 次修复到底装了什么（读码，非推断）

| 组件 | 落点 | 证据 |
|---|---|---|
| pad kernel | `dsv41_v5_epoch_pad_kernel` `ferrite_kernels.cu:9129` | block0 推进 `epoch += pad`、stamp 每个 peer 的 ready 行、同步 A4 broadcast word |
| wiring（**唯一调用点**） | `chain_dev.rs:8633` `if swallow_epoch_pad() { self.v5_epoch_pad_swallow(swallow_missing_rounds(cfg.n_layers))? }` | 位于 `dspark_spec_swallowed` 的 snapshot→import_tap 边界（之间无 collective） |
| pad 量 | `swallow_missing_rounds(n_layers) = 2*n_layers + 1 = 81`（40 层） | `chain_dev.rs:1925`；ledger: `legacy==aligned==165`，`swallowed==84`，差 81 |
| 观测（D1） | `DSV41_V5_LEDGER=1` → 每步一行 `[v5-ledger]` + 4B D2H；pre-step `[v5-ledger-pre]` | `chain_dev.rs:1882/9461/9474`；`c69c71d` 的 pre-step print |
| 脚本 | `batched_400_v2.sh` GATES 已含 `DSV41_SWALLOW_EPOCH_PAD=1 DSV41_V5_LEDGER=1` | `scripts/batched_400_v2.sh:154-156` |
| 硬约束 | `DSV41_LAZY_VERIFY` 与 `SWALLOW_STEP` **互斥**（lazy footprint = `3+81*k_emit`，常数 pad 无法等价） | `chain_dev.rs:1906-1912`；脚本 `FORBIDDEN` :175 |

**推论（必须写进判读）**：`script GATES` 里**没有 `DSV41_SH_PAIR_M`**（:145-156）。
⇒ 若测试 0 hang，得到的是「**全 mrows + 图 + SWALLOW，但无 SH_PAIR**」的 batched 基线，
**SH_PAIR 是本路线第一步（B1）要加的第一个 gate**。

---

## 1. 步时阶梯（口径统一后的账，起点 = 待测）

> 三个计时器必须分清（`swallow-unlocked-next-plan §0 C1`）：serve 墙钟（不可信）、
> `[dspark] steps=` 的 verify/draft/commit（半真，含 host barrier + D2H sync）、
> nsys per-kernel GPU 时间（**唯一纯模型执行时间**）。**每一次比较都标 arm + m + timer。**

| 阶段 | 内容 | 增量 | 累计步时 | 依据强度 |
|---|---|---|---:|---|
| **S0** | batched 解锁基线（全 mrows + 图 + SWALLOW，**无 SH_PAIR**） | — | **33~40ms（待测，§2 A3）** | ⚠️ 必须实测 |
| **S1** | + **B1 SH_PAIR M=6** | **−4.9~7.9ms** | 27~34ms | launch 账（`sh-pair-template-m-design §1.1`）|
| **S2** | + **B2 mrows 族**（逐个） | **−4.5~5.8ms**（设计）/ **−0~2ms**（实测系） | 25~32 / 27~34 | 设计口径 vs `{四件套}=−1.21ms` 实测 |
| **S3** | + **B3 hc 链**（A1+A2，truncate 修复后） | −1.3~1.7ms | 24~31 | 设计口径（`hc-chain-bandwidth-analysis`）|
| **S4** | + **B4 B6**（实现后） | −0.66~1.5ms | 23~30 | 第一性计数（−200~240 发/步）|
| **S5** | + **B5 tcgen05**（gate/up，**down 无核**） | −1.0~3.8ms | 21~28 | **修正口径**（非 −6.8ms）|
| **S6** | + **B6 L4**（占用/MLP，条件触发） | −5~8ms | 13~23 | **仓内零实测背书**，16~21 人日 |

**设计口径全兑现** ⇒ `S5 ≈ 14.5ms ≈ 414 tok/s`（400 ✓）；
**60% 兑现率** ⇒ `≈ 18~20ms ⇒ 300~340 tok/s`（400 ✗）。

---

## 2. Step A（P0）：解锁判定 + 基线测量 —— **这一步不做，后面全部作废**

### A1 —— 0 hang 的**概率性**确认（3× + 1 长跑）

| 项 | 判据 | 止损 |
|---|---|---|
| 独立运行 ×3 | 三次 `ar5-hang` 计数 **全 0** | 任一 hang ⇒ 回 `SWALLOW_STEP=0`，**不得把 B1/B2 叠上去**（失去归因能力）|
| 长跑 ×1 | 跑过历史 hang 出现的步数（≥ 计数任务全程）后仍 0 hang | 长跑 hang ⇒ 第 10 次修复同样「假解锁」，转 §6 R1 |
| pad 真的跑了 | `nm -D` 有 `dsv41_v5_epoch_pad`；**无** `[swallow-pad] ... has no ...` 一行 | 出现该告警 = `.so` 是旧的 = 本次是**幽灵实验**（重演第 9 次）|

### A2 —— ledger 逐 rank 对账（证明 pad 让「臂选择对 epoch 不可见」）

判据来自 `swallow-fix9-round-ledger-design §3.2`：
1. `legacy == aligned == 165`、`swallowed == 84` 的**旧不对称消失** ⇒ 各 rank 的 per-step epoch delta **相等**；
2. **pre-step epoch 必须单调不减**（`f2c160a` 记录了「epoch 从 1497 降到 54」的反常——
   若第 10 次修复后仍出现下降 ⇒ pad 没治根，**epoch 被别处 reset**，转 §6 R1）；
3. 跨 rank 分歧的**起始步**（若有）必须能定位（ledger 的 pre/post 行格式）。

> 读法：`[v5-ledger-pre]` 是**步前**（即使 hang 也已落盘），`[v5-ledger]` 是**步后**。
> 吞吐测量用 `B400_V5_LEDGER=0`（ledger 有 1 次 D2H/步，是观测开销不是路径）。

### A3 —— 基线口径钉死（三件套）

| 量 | 来源 | 用途 |
|---|---|---|
| `verify_ms / draft_ms / commit_ms` | `[dspark] steps=`（需 `DSV41_TIMING=1`）| 阶梯 S0 的分项 |
| `steady_median`（**median 而非 mean**）| `[dsv41] step pos=` 的稳态统计 | 唯一可信步时 |
| `k_acc` 直方图 + `mean-k` + `route=` | `[dsv41] step pos=` 的 delta + `[dspark]` | accept 门槛判据（§0-5）|
| `[verify_graph] captured verify_graph_m6` | serve 日志 | **必须出现**（否则 6 行块退直发，SWALLOW 缩水）|
| nsys per-kernel 名 + m | `nm -D` / nsys 聚合 | **唯一能证 `gemm_fp8_sh_exp_pair_kernel<6>` 真跑的手段** |

### A4 —— 红线（四缺一不下结论）

计数数字顺序正确（主探针）+ 首 61 行 + `k_acc` 逐位（同 prompt/seed）+ `faults=0` + 零**额外**拉丁
（模型自身在 ~50-60 token 的自然退化不算，见 `AGENTS.md` 的 protocol v2）。

### A 的分支

| 观测 | 结论 | 动作 |
|---|---|---|
| 0 hang ×3 + 长跑 + ledger 对账 | **batched 解锁 ✓** | 进 Step B |
| 0 hang 但 ledger 显示 epoch 下降/分歧 | **pad 未治根** | 转 §6 R1（找 reset 源），**不投优化** |
| 任一 hang | **假解锁** | 回 nograph（`VERIFY_GRAPH=0` 已是可用形态，只是拿不到图那 −1.5ms），改走 Path B0 |
| 红线破 | 语义回归 | 二分：`SWALLOW_STEP=0` → `VERIFY_GRAPH=0` → `E4M3=0` → `SIDS_WRITEBACK=0` |

---

## 3. Step B：优化叠加（每一步：gate + 机制 + 预期 + 测试 + 判定）

> **铁律**：一臂一进程（`OnceLock` 每进程读一次）、**单变量**、**同会话背靠背交错**、
> 每门读回 `/proc/<pid>/environ` + `nm -D` + **nsys 数 kernel 名**（本项目 #1 陷阱：gate 设了没生效）。
> GPU 串行（8 卡单 serve）；代码/分析可并行。

### B1 —— SH_PAIR `template<M=6>`（**第一优先**）

| 项 | 内容 |
|---|---|
| gate | `DSV41_SH_PAIR_M=1`（+ `DSV41_SH_PAIR_M_FOLD=1`，`chain_dev.rs:1468/1483`），调用点 `:13242-13269` |
| kernel | `gemm_fp8_sh_exp_pair_kernel<M>`（`dsv41_kernels.cu:7617`，launcher `dsv41_gemm_fp8_sh_exp_fused` `:7944`）|
| 机制 | 三段一核：`quant_rows` + **ONE** `sh_exp_fused<6>`（`m` 行共享一次权重读，**M 进 grid**，phase-1 `M×ceil(n1/32)=54` block）；替换 `m×` 逐行（w1w3→swiglu→fp8 emit → barrier → w2）|
| parity | **已确认为虚警**：哨兵 `0x5A` 可产出（改 `0x7F`）+ `SH_CHECK ++g_fails` 与 `main += sh_case()` 的 double-count；**kernel 无 bug**（`dspark-correctness-chain` §SH_PAIR Parity）|
| 预期 | **−4.9~7.9ms**（launch 账：shared expert 10.4ms 族 → 3~5.4ms；1000 → 80 发/步）|
| 为何是第一 | **唯一把 M 进 grid 的折法**（phase-1 并行度 `ceil(n1/32)×M`；而非折进 warp 只剩 9 block）；其余 mrows 都是「warp-per-row，M 只加链长」|
| **测试** | 同会话 A/B：`base` vs `base + DSV41_SH_PAIR_M=1`（其余门冻结）；**先跑 `scripts/sh_pair_ab.sh` 只跑 parity 复跑**（`SH_PAIR_M_FOLD ∈ {1,2,6}` 四臂）|
| **判定** | ①`nm -D` 有 `dsv41_gemm_fp8_sh_exp_fused`；②nsys 出现 `gemm_fp8_sh_exp_pair_kernel` 且 **M=6 变体**（不是 `<1>`）；③`verify_ms` 位移 ≥ 预期的 40%；④k_acc 逐位不变 + 计数数字正确 |
| 止损 | parity 复跑仍 failed（非虚警）⇒ 冻结 `SH_PAIR_M` 默认 OFF，−5ms 从阶梯划掉（落点掉到下一格）|
| ⚠️ | **每 M 特化必须各自 `cudaFuncSetAttribute`**（`:7994-8002` 已用宏；历史上漏设 `<m>` 导致 m=5 `cudaErrorInvalidValue`）；本项是唯一真收益点，故 **排第一、不得与其它 gate 同轮** |

### B2 —— mrows 族（零代码；逐个，**一次只加一个**）

顺序（按预期收益，**`VERIFY_HEAD_MROWS` 最后单独上**——历史 ar5-hang 组合）：

| # | gate | 落点 | 预期 | 实测先例 |
|---|---|---|---|---|
| B2a | `GATE_MROWS`（=`ROW_FOLD_GATE`）| `row_fold_gate()` `:1282-1305`，调用 `:12577` | **−2.75ms**（设计，未单测）| 未测 |
| B2b | `INDEXER_MROWS`（front 半）| `indexer_mrows()` `:1246`，`indexer_front_rows` `:9790` | −1.0~1.5ms | 未测，无 decline 日志 |
| B2c | `VERIFY_ROPE_MROWS` | `:1207` gate `:1210` | −0.53ms（launch 账）| 未测 |
| B2d | `NORM_MROWS` / `COMPRESSOR_MROWS` | `:1273` / `:1070` | 小 | ⚠️ 无 decline 日志 |
| B2e | `VERIFY_HEAD_MROWS` | `:1618` gate `:1621`，调用 `:6815` | −0.7~0.9ms | ⚠️ **正是历史 ar5-hang 的组合，最后单独验证** |

| **测试** | 每个 gate 单独一轮：`base + <gate>` vs `base`；A/B 交错 |
| **判定** | ①`/proc/environ` 读回；②`nm -D` 对应符号（如 `dsv41_rmsnorm_rows`）；③**nsys kernel 名**确认变体真跑（`gemm_fp8_mrows_kernel<6>` vs 逐行）；④k_acc 逐位 + 计数数字正确 |
| **止损** | 任一 gate 位移 < 预期 40% ⇒ **停该 gate、转下一个**；**不在同一轮叠 gate 找感觉** |
| ⚠️ | 五个 gate（SH_EXP/GATE/INDEXER/NORM/COMPRESSOR）**静默 `return Ok(false)`，无 decline 日志**（`batched-400-v2-prediction §3.2`）⇒ 只能靠 nsys 取证，**缺证据不得下「declined」结论** |
| ⚠️ | `SH_EXP_MROWS` 已两次实测零收益（instruction-bound）——**不要为它编预算** |

### B3 —— hc 链（A1 + A2，**前置一行 truncate 修复**）

| 项 | 内容 |
|---|---|
| 前置修复 | `hc_mixes_auto`（`chain_dev.rs:11556-11714`）的 **verify 调用点**（`:11625`）把 `bf16_truncate()` 改成 `false`（A2 与 A1-a 同构的坑；A1-a 已修 `collapse_norm_rows`）|
| gate | `HC_VERIFY_FUSE=1` + `HC_FRONT_ROWS=1`（**`HC_VERIFY_FUSE` 是反向默认 `v=="1"`，注意**）|
| 机制 | verify 的 10 发/层 → hc 融合后 240 发/步（`hc_mixes`+`hc_collapse`+`norm_rows`+`hc_post`+`memcpy_d2d` 折核）|
| 预期 | **−1.3~1.7ms**（2.96 → ~1.3-1.7）|
| 测试 | 先落 truncate 一行修复（单 commit）；再 A/B `base + HC_VERIFY_FUSE=1 HC_FRONT_ROWS=1`；**重点看:是否带回 BF16_TRUNCATE 进 verify（历史破零拉丁）** |
| 判定 | `verify_ms` 位移 + 计数数字正确 + 零额外拉丁；`nm -D` 有 `ferrite_p2p_ar_v5_hcpost_rows` |
| 止损 | 出现额外拉丁/计数错 ⇒ 回退该 commit（A1/A2 是历史上打破零拉丁的组合）|

### B4 —— B6（`dsv41_gemm_fp8_mrows_f32`，**需实现**）

| 项 | 内容 |
|---|---|
| 现状 | **设计文档-only**（`b6-mrows-f32-design.md`），kernel/launcher 未实现 |
| 机制 | wo_b 的 `m × quant_fp8 + 1 × proj_mrows = m+1 发/层` → **1 发/层**；**fp8 权重 × raw f32 激活**（跳过量化往返，略更准）|
| 预期 | **−0.66~1.5ms**（−200~240 发/步）；**B1–B6 全族 −2.8~4.9ms**（8~12 人日，本期不投）|
| 正确性判据 | **不是**「与旧 `quant_fp8+proj_mrows` 逐位相同」，而是「**m 行核第 r 行 == M=1 f32 GEMV 第 r 行**」（= EAGER 第 r 行）|
| 成本 | 0.5 人日（B6 单项）；**可与 B1/B2 的 GPU A/B 并行写**（代码不占 GPU）|
| 止损 | parity 不过或 `verify_ms` 位移 <40% ⇒ 停 |

### B5 —— tcgen05（gate/up group，**go/no-go 门，当前仍 blocked**）

| 项 | 内容 |
|---|---|
| 现状 | **仍 blocked**：`750725f` 冒烟 LEN=0 + 1 misaligned；HEAD `4beb9a4` 上了第 2 轮对齐修复（+35/−12，4 处 byte-fallback 守卫，首嫌 `:5009` `expert_tcgen05_gateup_e4_kernel`）——**待重测** |
| gate 链 | `EXPERT_ACT_E4M3=1` + `EXPERT_TCGEN05_E4M3=1` + `EXPERT_GROUPED=1` + `GATEUP_FUSE=0` + `EXPERT_ILV=0`（**五门必须同开**，`ILV=0` 改权重布局，不能与 ILV=1 比绝对值）|
| 预期 | **−1.0~3.8ms**（**非 −6.8ms**：`down` 无 tcgen05 核，3.48ms 原样保留）|
| 执行顺序 | 符号预检（`nm -D`）→ dry-run → 短 prompt 冒烟 → 基线轮 → **门税对照轮**（只 `GATEUP_FUSE=0 ILV=0`）→ grouped 轮 |
| 判定 | nsys 同时见到 `e4m3_gemm_kernel`(prefill) 与 `e4m3_gemm_grouped_kernel`(verify)；告警缺席（`expert_grouped_skipped_note` 等）|
| 止损 | 冒烟拉丁/非法指令 ⇒ **立即关闭路径，不投变体矩阵**（勿重演 v17→v21 四变体全中性）；门税 ≈ +0.4~0.5ms，中性和退化先查告警 |

### B6 —— L4 占用/MLP（**条件触发，仅当 B1–B5 落点 >15ms**）

| 项 | 内容 |
|---|---|
| 内容 | L4-3 tcgen05 gate/up K-split + L4-4 tcgen05 down 换核 + L4-1 mrows nwarps/crossover + L4-9 collapse_norm 摊开 |
| 预期 | **−5~8ms**；但**三者必须同时上**（v19/v21/v24 已证「只动一个因子无效」）|
| 成本 | **16~21 人日**；**仓内零实测背书** |
| 触发 | 仅当 §1 的 S5 实测 >15ms 且 accept ≥3 时才投 |
| 判词 | **L4 是「跨过 400 之后」的层（把中 accept 也拉进 400），不是最快路** |

---

## 4. 性能预测（两条口径，显式标注）

> 起点 S0 = **待测基线**（33~40ms，从「无 SH_PAIR 的全 mrows + 图 + SWALLOW」测得）。
> 下表两种兑现率，counting 口径（accept 5，k_emit=6）。

| 阶段 | 设计口径累计 | tok/s | 60% 兑现累计 | tok/s | 400 |
|---|---:|---:|---:|---:|:---:|
| S0（待测） | 33~40 | 150~182 | 33~40 | 150~182 | ✗ |
| S1（+SH_PAIR） | 27~34 | 176~222 | 29~36 | 167~207 | ✗ |
| S2（+mrows） | 25~32 | 188~240 | 27~34 | 176~222 | ✗ |
| S3（+hc） | 24~31 | 194~250 | 26~33 | 182~231 | ✗ |
| S4（+B6） | 23~30 | 200~261 | 25~32 | 188~240 | ✗ |
| **S5（+tcgen05）** | **21~28** | **214~286** | **24~31** | **194~250** | ✗ |
| S6（+L4） | 13~23 | 261~462 | 19~26 | 231~316 | ⚠️ |

**设计口径全足额（`swallow-unlocked-next-plan §6` 的账）**：`S5 = 14.5ms = 414 tok/s` ⇒ 400 ✓。
**· 但要求 S1/S2/S4/S5 四项同时足额**——仓史上前所未有（从没有任何一轮把设计口径全兑现）。
**两句话的真话**：
1. **400 的临界点在「tcgen05 兑现」之后（≈15ms）**，且它只把 B1–B5 推到 **bordering**；
   L4 才是真正的余量来源，但成本 16~21 人日、零实测背书。
2. **步时不是唯一门槛**：`accept ≥ 3` 是硬门槛（τ=2.214 时 400 物理不可达）⇒ **B 与 accept 支线并行**。

---

## 5. 测试矩阵（一次 GPU 会话，背靠背交错）

**姿态**：一臂一进程（`OnceLock` 每进程读一次）；**交错 A B A B** 抵消时钟/热漂；一次会话 5~6 臂。

```bash
COMMON="DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 \
        DSV41_EXPERT_ACT_E4M3=1 DSV41_BF16_TRUNCATE=1 \
        DSV41_SH_EXP_MROWS=1 DSV41_MROWS_SMALL_N_ADAPTIVE=1 \
        DSV41_GATE_MROWS=1 DSV41_INDEXER_MROWS=1 DSV41_NORM_MROWS=1 \
        DSV41_COMPRESSOR_MROWS=1 DSV41_DRAFT_GRAPH=1 DSV41_DRAFT_P3A=1 \
        DSV41_VERIFY_GRAPH=1 DSV41_SWALLOW_STEP=1 DSV41_SWALLOW_EPOCH_PAD=1 \
        DSV41_TIMING=1 DSV41_DSPARK_DEBUG=1"

# 基线：脚本原样（B400_V5_LEDGER=0 取吞吐；另跑一次 =1 取 ledger 行）
S0 = bash scripts/batched_400_v2.sh                     # 吞吐（ledger off）
S0L= B400_V5_LEDGER=1 bash scripts/batched_400_v2.sh    # 仅读 [v5-ledger] 行

# 优化臂（各自 base + 单 gate；脚注：SH_PAIR_M / head_mrows 不在脚本默认矩阵，需手工或加 knob）
B1 = COMMON + DSV41_SH_PAIR_M=1                          # ★第一优先
B2 = COMMON + <单个 mrows gate>                          # 逐个：GATE→INDEXER→ROPE→NORM/COMPRESSOR→HEAD(最后)
B3 = COMMON + DSV41_HC_VERIFY_FUSE=1 DSV41_HC_FRONT_ROWS=1   # 前置：A2 truncate 一行修复
B5 = bash scripts/batched_400_v2.sh with B400_TCGEN05_E4M3_GROUPED=1  # 先跑 0a/0b/冒烟/门税
```

**为何必须一次会话多臂**：同一远端 8 卡单驱动，拆多次只是更长的串行；但**归因要求单变量**
⇒ 交错 + 每次只加一个 gate 是必须的。

**每臂必录（缺一不能下结论）**：
1. `/proc/<pid>/environ` 逐门读回（`<tag>.env`）；
2. `nm -D $SO` 符号存在性；
3. `[dspark] steps=` 的 `verify/draft/commit` 中位位移；
4. **nsys 按 kernel 名聚合**（`gemm_fp8_sh_exp_pair_kernel` / `gemm_fp8_mrows_kernel<M>` vs 逐行）；
5. 计数数字顺序 + 首 61 行 + `k_acc` 逐位 + 零额外拉丁；
6. `[verify_graph] captured verify_graph_m6`（否则 6 行退直发）。

**执行顺序（关键路径）**：
```
A1/A2（解锁判定 + ledger）
  → A3/A4（基线口径 + 红线）
  → B1 SH_PAIR（最大单项，先拿）
  → B2 mrows 族（逐个）
  → B3 hc 链
  → B5 tcgen05（go/no-go，独立会话）
  → B4 B6（可与 B1/B2 并行写代码）
  → [B6 L4，条件]
```

---

## 6. 风险与止损门

| ID | 风险 | 触发信号 | 止损 |
|---|---|---|---|
| R1 | **pad 未治根**（epoch 被别处 reset）| ledger 出现 epoch 下降（1497→54）/ 跨 rank 分歧 | 找 reset 源（`2aef683` 的 `epoch_dev = staging + ctr_at` 线索）；**不投优化** |
| R2 | **假解锁**（0 hang 一次不算）| 3 次里任一 hang / 长跑 hang | 回 nograph（`VERIFY_GRAPH=0`，已可用）；**不为 ar5 修复投入除非探针证 AR 等待 ≥2ms** |
| R3 | **SH_PAIR parity 非虚警** | parity 复跑仍 failed | 冻结 `SH_PAIR_M` OFF；−5ms 从阶梯划掉 |
| R4 | **instruction-bound ⇒ mrows 零兑现**（先例：SH_EXP 两次零收益） | 任一 gate 位移 <40% | 停该 gate、转下一个；**不在同一轮叠 gate** |
| R5 | **gate 设了没生效**（#1 陷阱） | `verify=` 无位移 | `/proc/environ` + `nm -D` + **nsys 数 kernel 名** |
| R6 | **红线回归**（A1/A2 带 BF16_TRUNCATE 进 verify）| 额外拉丁 / 计数错 | 回退该 commit；A2 truncate 一行修复先落 |
| R7 | **tcgen05 红线**（e4x 从未上机；`[OPEN]` ×2）| 冒烟拉丁/非法指令/LEN=0 | 立即关闭路径，**不投变体矩阵** |
| R8 | **口径混用**（步时/accept/timer 三者混乘）| 跨会话比较 | 每次标 arm + m + timer；同会话背靠背 |
| R9 | **L4 投入陷阱** | L4-1 单独 A/B 中性 | 不开变体矩阵，除非 S5 给出 routed 实测新地板 |
| R10 | **accept 不足**（τ=2.214 < 2.80）| `mean-k` 停在 ~1.2 | 400 物理不可达 ⇒ 与 B 并行开 **accept 审计**（sglang silent shared-expert loader gap 同构排查）|

---

## 7. gate / 行号速查

| 对象 | 位置 |
|---|---|
| **epoch pad** kernel / wiring / pad 量 | `ferrite_kernels.cu:9129` / `chain_dev.rs:8633` / `:1925`（`2*n_layers+1=81`）|
| ledger（D1）| `chain_dev.rs:1882` / `:9461`（post）/ `:9474`（pre）|
| SWALLOW 分派 / 实现 | `chain_dev.rs:6943` / `dspark_spec_swallowed :8074`；tap 传递 `carry_kept_tap :7999` |
| **SH_PAIR M** gate / 调用点 / kernel / launcher | `:1468`（+ fold `:1483`）/ `:13242-13269` / `dsv41_kernels.cu:7617` / `:7944` |
| mrows gate | `GATE_MROWS :1282` · `INDEXER_MROWS :1246` · `VERIFY_ROPE_MROWS :1207` · `VERIFY_HEAD_MROWS :1618` · `NORM_MROWS :1273` · `COMPRESSOR_MROWS :1070` · `SH_EXP_MROWS :1335` |
| hc 链 | `hc_verify_fuse :11801` · `hc_mixes_auto :11556-11714`（A2 truncate `:11625`）|
| tcgen05 | `expert_tcgen05_e4m3 :756` · `expert_grouped :903` · `gateup_fuse :1891`；kernel `dsv41_experts_mxf4.cu:5120`（e4 swapAB）/ `:6070`（e4x grouped）|
| R2/K1K2 | `ATTN_LIN_FUSE :2133` · `ATTN_MROWS2 :2170` · `ATTN_MROWS_ROPE_NORM :2236` |
| B6 设计 / 骨架来源 | `b6-mrows-f32-design.md` / `dsv41_kernels.cu:5208`（`gemm_fp8_mrows_kernel`）|
| 脚本 | `batched_400_v2.sh`（GATES `:145-156`、FORBIDDEN `:175`、TC5 `:205-209`）· `sh_pair_ab.sh` · `verify_mrows.sh` · `tcgen05_smoke.sh` · `tcgen05_bench.sh` |

---

## 8. 一句话交付

> **解锁后的第一件事是「证明解锁真 + 锁基线」（3× + 长跑 + ledger 对账），不是上优化。**
> **然后是 `SH_PAIR M=6`（唯一把 M 进 grid、最大单项）→ mrows 族（逐个、止损 40%）→ hc 链 →
> B6 → tcgen05（go/no-go）**；`R2/LAZY_SDR` 是 lazy-only，`FORK/RING_WIN` 在 batched 下须重 A/B。
> **设计口径全兑现 ⇒ 14.5ms / 414 tok/s（400 ✓）；60% 兑现 ⇒ 18~20ms / 300~340（差 15~25%）。**
> **400 需要四项同时足额 + accept ≥3；L4 是余量层（16~21 人日、零实测背书），列条件触发。**

---

*工部 · 只读设计 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*所有 ms 均标来源（实测 / launch 账 / 设计口径 / 代数）；与任务前提冲突处
（「SH_PAIR parity kernel 无 bug」→ 已确认虚警；「mrows 族 B1-B6 kernel 已有」→ B6 未实现；
「R2/MARKOV/FORK/RING_WIN +15.6% 可用于 batched」→ lazy-only/需重 A/B；
「tcgen05 −2ms 待重测」→ 仍 blocked、修正 −1.0~3.8ms）已显式给出依据。*
