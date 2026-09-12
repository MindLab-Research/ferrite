# SWALLOW 解锁后的 400 最终路径更新（户部 · 最终版）

> 户部 · 2026-09-12 20:34 · **只读勘察 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 输入（现场核对）：`swallow-unlock-400-sprint.md`（注：任务写作 `swallow-unlocked-400-sprint.md`，实际文件名无 `ed`）·
> `swallow-unlocked-next-plan.md` · `swallow-1token-fix-design.md` · `dspark-correctness-chain.md`（尾部）·
> `nsys-clean-stack-91-design.md` · `batched-400-v2-remaining-roi.md` · `session-user-final-report.md` ·
> `serve.rs` / `dsv41-run.rs` / `chain_dev.rs` / `scripts/batched_400_v2.sh` 现场读码。
> **口径纪律**：每条 ms 标来源（实测 / launch 账 / 设计口径 / 代数）；与任务前提冲突处显式给出依据。

---

## 0. 判决（先读七条）

1. **❗ SWALLOW 不是「即将解锁」，而是「OOB 已修，暴露了下一个独立故障」。**
   任务前提「OOB 修复验证成功 ⇒ SWALLOW 即将解锁」**不成立**。现场最新证据（`swallow-1token-fix-design.md`，
   20:31，晚于任务所依据的 OOB 分析）显示本次跑机结果 = **`LEN=1` / `completion=1` / `finish_reason=length`**：
   - 4 项 OOB 判据全过（CANARY=0 / GUARD=0 / RESET=0 / ar5-hang=0）✓ ——**这是 OOB 修复的成功**；
   - 但 **ledger 只到 `pos=15` 的 pre，没有 `[v5-ledger]` note** ⇒ **第一次 swallowed 轮（pos=15）没走完**。
   - 判定（`swallow-1token-fix-design §0`）：`length` 在本仓**不是「位置到顶」，是「prefill 后、首个 decode token 提交前就 FAULT」**的签名
     （`driver.rs:337-350` + `single_flight.rs:157-183`）。根因 A（pos=15 返回 `Err`，置信度高）/ B（hang → 1800s 超时，置信度中）。
   ⇒ **解锁门 = 先收敛 1-token 故障（A vs B），不是先测基线。** 见 §2 Step 0。

2. **engram 越界修复「实施中」——现场看已经落树（DONE，非进行中）。**
   `serve.rs:405-414` 与 `dsv41-run.rs:358-367` 两处**都已补** `eng_cols/eng_rows`：
   `ar_bytes = max(hc_dim, VERIFY_ROWS*dim, eng_rows) * 4`，新 slot = `max(20480,30720,36864)*4 = 147456 B`。
   与 `dspark-correctness-chain.md` 尾部的判定逐字一致。**该项不计入关键路径**（已完成）。

3. **nsys 实测把账推翻了一半：AR 27.1% 是 #1，高于设计预期的 17~20%。**
   这意味着**设计口径系统性低估了通信**。后果见 §4：设计口径「14.5ms 全兑现」的概率进一步下降，
   **400 的兑现率门槛被抬高**（§3 给出代数：@accept5 需 **≈97% 兑现**）。

4. **「SH_PAIR M=6 arm 编译产物存在」本机不可验证**：`kernels/cuda/*.so` **在本机不存在**（`ls` No such file）。
   源码侧 kernel/launcher 齐备（`dsv41_kernels.cu:7617` / `:7933` / launcher `:7944` 段的 `cudaFuncSetAttribute` 宏），
   gate 侧齐备（`chain_dev.rs:1470` / `:1486` / 调用点 `:13511`）。⇒ 产物验证须在**远端** `build.sh 103a` 后 `nm -D` 取。

5. **任务给的「batched 基线 33~40ms」与源文档自身的另一处锚点冲突。**
   `swallow-unlock-400-sprint.md §1` 表写 S0=33~40ms；但同文 §0-6 / §4 脚注写
   「设计口径全兑现 ⇒ S5 ≈ **14.5ms** / 414 tok/s」——按 §1 的逐项设计增量 Σ≈16.5ms 反推，**S0 只能是 ~31ms**。
   ⇒ 本文采纳 **S0 = 31ms（区间 27~36）**作为工作基线，并**要求 Step 1 实测钉死**（±5ms 恰好是「400 ✓ vs ✗」的分界）。

6. **400 的代数硬约束（不可协商）**：
   ```
   400 tok/s ⇒ 步时 ≤ 1000·k_emit/400 = 2.5·k_emit ms   （k_emit = 1 + accept）
   accept 5 → k_emit 6 → 步时 ≤ 15.0ms
   accept 4 → k_emit 5 → 步时 ≤ 12.5ms
   accept 3 → k_emit 4 → 步时 ≤ 10.0ms
   accept 1.214（出师表）→ 步时 ≤ 5.54ms  ← 低于 L5 floor 8~9ms，物理不可达
   ```

7. **叠加顺序按「收益 ÷ 风险 ÷ 依赖」，不按收益排序**：
   ```
   Step0 收敛1token → Step1 锁基线 → B1 SH_PAIR M=6 → B2 mrows族 → B3 hc链 → B4 B6 → B5 tcgen05 → [B6 L4 条件]
   ```

---

## 1. 现场钉死（读码证据，非推断）

| 项 | 落点 | 证据 |
|---|---|---|
| engram slot 修复（**已落**）| `serve.rs:410-414` / `dsv41-run.rs:363-367` | `eng_cols=(max_ngram-1)*n_heads`、`eng_rows=VERIFY_ROWS*eng_cols*head_dim`；`ar_bytes=max(...)*4` |
| VERIFY_ROWS = 6（=M，DSPARK_DRAFTS+1）| `chain_dev.rs:84` / `:97`（静态断言）| SWALLOW 把 m 从 5 推到 6 ⇒ M 进 grid 的前提成立 |
| SH_PAIR M gate / fold / 调用点 | `chain_dev.rs:1470` / `:1486` / `:13511` | `DSV41_SH_PAIR_M=1`（默认 OFF）+ `DSV41_SH_PAIR_M_FOLD` |
| SH_PAIR kernel / launcher | `dsv41_kernels.cu:7617`（`gemm_fp8_sh_exp_pair_kernel<M>`）/ `:7944`（`dsv41_gemm_fp8_sh_exp_fused`）| 每 M 特化各自 `cudaFuncSetAttribute`（`:7994-8002`）|
| 脚本当前矩阵**不含** SH_PAIR_M | `scripts/batched_400_v2.sh:145-156` | GATES 含 mrows/图/SWALLOW/PAD/LEDGER，**无** SH_PAIR_M ⇒ 解锁后基线是「无 SH_PAIR」 |
| FORBIDDEN | `:175` | `DSV41_LAZY_VERIFY / HC_VERIFY_FUSE / HC_FRONT_ROWS`（lazy 与 SWALLOW_STEP 互斥）|

---

## 2. Step 0（**新门，先于一切**）：收敛 1-token 故障

| 项 | 内容 |
|---|---|
| 目的 | 判定 `LEN=1` 的根因是 A（pos=15 `Err`）还是 B（hang→超时），**一步收敛** |
| 判据 | `grep 'spec step err at pos 15'`（`serve.rs:611-614`）命中 ⇒ A；仅 `grep 'a rank did not answer'`（`:256-258`）命中 / 两次 pos=15 间隔 ≥1800s ⇒ B |
| A 的修 | 拿错误文本定位（`swallow-1token-fix-design §2.A` 的 E1–E6 子表），**OOB 修好后的新故障点** |
| B 的修 | phase 断言 + 缩短超时（`:§2.B`）|
| 止损 | 两者都不命中 ⇒ 先给 `dspark_spec_step` 加 exit-tag 错误打印（零 GPU 取证），再判 |
| **不做会怎样** | 基线 S0 **测不出来**（请求根本跑不完），后面全部作废 |

> **为何是「新故障」而非「OOB 复发」**：4 项 OOB 判据全过，说明越界写已堵住；`LEN=1` 是**独立**的下游故障
> （首个 swallowed 轮未能完成）。这正是「OOB 修复成功」与「SWALLOW 可用」之间的 gap。

---

## 3. Step 1（基线）+ 优化叠加：设计口径 vs 60% 折算

### 3.1 各优化项的设计增量（口径统一，来源标注）

| # | 项 | gate | 设计增量 | 来源 |
|---|---|---|---:|---|
| B1 | **SH_PAIR M=6** | `DSV41_SH_PAIR_M=1` | **−4.9~7.9**（mid 6.4）| launch 账：shared expert 10.4ms 族 → 3~5.4ms；1000→80 发/步 |
| B2 | mrows 族（6 gate）| `GATE/INDEXER/ROPE/NORM/COMPRESSOR/HEAD_MROWS` | **−4.5~5.8**（mid 5.15）| 折核件数×per-launch 价；**实测系 −0~2ms（旁证偏负）** |
| B3 | hc 链（A1+A2，前置 truncate 一行）| `HC_VERIFY_FUSE + HC_FRONT_ROWS` | −1.3~1.7（mid 1.5）| `hc-chain-bandwidth-analysis` |
| B4 | B6（需实现）| `dsv41_gemm_fp8_mrows_f32` | −0.66~1.5（mid 1.08）| 第一性计数（−200~240 发/步）|
| B5 | tcgen05（gate/up only）| 五门同开 | **−1.0~3.8**（mid 2.4）| 修正口径（**非 −6.8**：down 无核）|
| B6 | L4 占用/MLP | 条件 | **−5~8**（mid 6.5）| 仓内**零实测背书**，16~21 人日 |

### 3.2 步时阶梯（S0 = 31ms 工作基线，counting 口径 accept 5 / k_emit 6）

| 阶段 | 增量(mid) | 设计口径累计 | tok/s | 60% 兑现累计 | tok/s | 400 |
|---|---:|---:|---:|---:|---:|:---:|
| **S0** 解锁基线（全 mrows+图+SWALLOW，**无 SH_PAIR**）| — | **31** | 194 | **31** | 194 | ✗ |
| S1 +SH_PAIR | −6.4 | 24.6 | 244 | 27.2 | 221 | ✗ |
| S2 +mrows | −5.15 | 19.5 | 308 | 24.1 | 249 | ✗ |
| S3 +hc | −1.5 | 18.0 | 333 | 23.2 | 259 | ✗ |
| S4 +B6 | −1.08 | 16.9 | 355 | 22.5 | 267 | ✗ |
| **S5 +tcgen05** | −2.4 | **14.5** | **414** | **21.6** | **278** | **✓ / ✗** |
| S6 +L4 | −6.5 | 8.0 | 750 | 17.7 | 339 | ✓ / ✗ |

> **自洽性**：设计口径 S5=14.5ms=414 tok/s 与 `swallow-unlock-400-sprint §0-6` 的锚点一致；
> **但 S0=33~40 的写法与之矛盾（见 §0-5），已修正为 31ms。**

### 3.3 兑现率门槛（代数，回答任务 Q3）

400 @ accept 5 需步时 ≤15ms ⇒ 需兑现 `Σ_cash = S0 − 15 = 16ms`，而设计总增量 `Σ_design(S1..S5) = 16.5ms`：

```
所需兑现率 = 16.0 / 16.5 = 96.8%
```

> **诚实结论**：**400 @ accept 5 要求 S1–S5 近乎全额兑现（≈97%）——仓史从无以 100% 兑现过任何一轮。**
> 60% 兑现（历史值）⇒ **21.6ms / 278 tok/s ⇒ 差 30%**。
> 叠加 **AR 实测 27.1% > 设计 17~20%**（§0-3）这一反向证据，**实际兑现率很可能 <60%**。

---

## 4. nsys 实测画像对路径的影响（AR 27.1% #1）

| 族 | nsys 实测 | 设计预期 | 读数 |
|---|---:|---:|---|
| **AR** | **27.1%（#1）** | 17~20% | **通信被低估 ~35-40%**；AR 是协议地板（17.3µs/轮），减次数的唯一路 = **accept↑** |
| MoE routed | 19% | 18~23% | 与设计一致（自身几乎不变，占比被动）|
| gemv/投影 | 18.5% | 11~14% | 比预期高 ⇒ R2 的兑现**低于设计** |
| SH_PAIR | ~8-9% | 8~9% | 中性；**B1 正是要压这一族** |

**三条推论**：
1. **AR 是 #1 且是协议地板 ⇒ 最大的剩余杠杆是「减少 AR 轮数」，即 accept**，不是 kernel 微优化。
   `VERIFY_AR_FOLD`（hc_post 折进 pubred 尾，−80 发/步 ≈ −0.24ms）与 store-fuse（2→1 核）是仅有的 AR 侧 kernel 空间，
   后者被 `DSV41_AR_STORE_FUSE`（round 19 破四文本）挡住。
2. **B1/B2 的真实价值被重新定性**：它们压的是 **launch 数**（shared expert 1000→80 发、mrows 6→1 发），
   而 AR 的 160 发/步与这些 launch 共享 submit 通路 ⇒ **减少 launch ⇒ 间接缓解 AR-相邻开销**。这仍是第一优先。
3. **设计口径的 14.5ms 建立在「通信占比 ≤20%」的旧画像上**；实测 27.1% ⇒ **设计口径应打折**（§3.3 兑现率门槛的真实含义）。

---

## 5. 与 lazy 的对比：什么 accept 下 batched 更好（任务 Q4）

判据（`swallow-unlocked-next-plan §3.3`，`lazy-batched-gate §0.7`）：

```
lazy 更好 ⟺ (1+mean_k) < B/c        ⟺  mean_k < B/c − 1
batched 更好 ⟺ mean_k > B/c − 1
B = batched 步时（ms），c = 6.15 ms/row（lazy 每行的边际成本）
```

| B（batched 步时）| 阈值 mean_k | 计数 5.0 | 出师表 1.214 | 对话 0.96 |
|---:|---:|:---:|:---:|:---:|
| **28ms**（当前 SWALLOW）| **3.55** | batched ✓ | lazy ✓ | lazy ✓ |
| 21ms（S2 后）| 2.41 | batched ✓ | lazy ✓ | lazy ✓ |
| **15ms**（S5 全兑现）| **1.44** | batched ✓ | lazy（1.214<1.44，勉强）| lazy ✓ |
| **10ms**（S6 L4）| 0.63 | batched ✓ | **batched ✓** | lazy（0.96>0.63 → batched ✓）|

**结论**：
- **当前形态（B≈28ms）：只有计数型（mean_k>3.55）走 batched 划算**；出师表/对话在 batched 下是**净亏**。
- **优化步时本身就在「扩大 batched 的适用面」**：B 从 28→15ms，阈值 3.55→1.44；B→10ms，阈值 0.63，**三任务全部 batched 更优**。
- ⇒ **SWALLOW 不能当全局默认**，正确形态是「SWALLOW 常开 + lazy⇄batched 按任务路由」（`DSV41_LAZY_VERIFY` 已有路由设计，带 Schmitt 滞回）。
  这是**产品决策**（改变 P50 分布），**提请用户/尚书省仲裁**（同 `swallow-unlocked-next-plan §3.3`）。

---

## 6. 400 可达性矩阵（任务 Q3）

| 场景 | 目标步时 | design-full 落点 | 结论 |
|---|---:|---|---|
| batched + 全优化 @ **accept 5**（计数）| ≤15.0ms | **14.5ms = 414 tok/s** | **✓ 条件可达**（需 ≈97% 兑现，无先例）|
| batched + 全优化 @ **accept 4** | ≤12.5ms | 14.5ms = 345 tok/s | ✗（差 14%）|
| batched + 全优化 @ **accept 3**（用户校准）| ≤10.0ms | 14.5ms = **276 tok/s** | **✗（差 31%）；需 S5+L4 全兑现（8.0ms=500）** |
| batched + 全优化 @ accept 1.214（出师表）| ≤5.54ms | — | ✗ 物理不可达（低于 L5 floor 8~9ms）|
| **60% 兑现 @ accept 5** | ≤15.0ms | 21.6ms = 278 tok/s | ✗ |
| **60% 兑现 @ accept 3** | ≤10.0ms | 21.6ms = 185 tok/s | ✗ |

**两句话**：
1. **accept 是硬门槛，不是可选项**：accept 3 时即便 S1–S5 全足额也只有 276 tok/s。**400 是 accept≥5（计数）+ 全足额 的目标。**
2. **accept 3 的 400 需要 L4/L5**（S5+L4 全足额 → 8ms → 500）。L4 是「跨过 400 之后」的层。

---

## 7. 最终判定：400 何时可达（任务 Q5）

**时间线（每步都是 gate，失败即停）**：

```
T0  现在：OOB 已修（CANARY=0 ✓）+ engram slot 已修（147456 B ✓）
        ↓
【Step 0】收敛 1-token 故障（A: pos=15 Err / B: hang）   ← 唯一真门，未过则一切作废
        ↓
【Step 1】SWALLOW 基线实测（预期 S0 ≈ 28~32ms，most likely ~30~31；区间 27~36）
        · 3× 独立运行 + 1 长跑 + ledger 逐 rank 对账（`§2 A1/A2`）
        · 门税对照：ledger off 取吞吐、on 取 ledger
        · 预期落点：~30ms ⇒ 200 tok/s @ accept 5（±15%）
        ↓
【B1】SH_PAIR M=6（第一优先：唯一把 M 进 grid、最大单项、arm 已编译）
        · 预期 → ~24.6ms（design）/ ~27ms（60%）
        · 判据：nsys 出现 `gemm_fp8_sh_exp_pair_kernel<6>`（非 <1>）
        ↓
【B2】mrows 族（逐个、止损 40%；HEAD_MROWS 最后单独上——历史 ar5-hang 组合）
        · 预期 → ~19.5ms（design）/ ~24ms（60%）；旁证偏负（SH_EXP 两次零收益）
        ↓
【B3】hc 链（前置 A2 truncate 一行修复）
【B4】B6（0.5 人日，可与 B1/B2 的 GPU A/B 并行写）
        ↓
【B5】tcgen05 go/no-go（当前 2 轮修复失败：TMA bulk 16B 硬对齐无法 fallback）
        · 预期 → 14.5ms（design）/ 21.6ms（60%）
        ↓
★★★ 400 判决点 ★★★
  · S1–S5 ≈97% 兑现  ⇒ 14.5ms ⇒ 414 tok/s ⇒ 400 ✓（@ accept 5）
  · 60% 兑现           ⇒ 21.6ms ⇒ 278 tok/s ⇒ 400 ✗（差 30%）
        ↓
【B6/L4】条件触发（仅当 S5 实测 >15ms 且 accept ≥3）
        · 16~21 人日、仓内零实测背书
        · 也是唯一能把 accept 3 拉进 400 的层（→ 8ms ⇒ 500）
```

### 最终判定（三句）

1. **SWALLOW 现在解锁不了**——先过 Step 0（1-token 故障）。**OOB 已修 ≠ SWALLOW 可用。**
2. **@ accept 5（计数任务）**：400 **条件可达**，但需 S1–S5 **≈97% 全额兑现 + accept≥5**。
   以实测 AR 27.1%（设计预期 17~20%）为反向证据，**现实落点更可能是 60% 兑现 = 21.6ms / 278 tok/s（差 30%）**。
3. **@ accept 3（用户校准口径）**：400 **本路线不可达**（全足额仅 276 tok/s），**必须叠 L4**（+16~21 人日、零背书）。
   ⇒ **400 是「counting 口径（accept≥5）+ 全足额」的目标；accept 3 的 400 属于 L4 之后。**

---

## 8. 对任务前提的修正汇总（户部职责内必须显式给出）

| 任务前提 | 现场 | 依据 |
|---|---|---|
| SWALLOW「即将解锁」| **❌ OOB 修好但 LEN=1，第一次 swallowed 轮未完成** | `swallow-1token-fix-design.md`（20:31）§0/§1；`driver.rs:337-350` |
| engram 越界修复「实施中」| **✅ 已落树（DONE）** | `serve.rs:405-414`、`dsv41-run.rs:358-367` |
| 基线 33~40ms | **⚠️ 与源文档自身 14.5ms 锚点矛盾，修正为 S0≈31ms（27~36）** | §0-5 代数反推 |
| SH_PAIR M=6 arm 产物存在 | **⚠️ 本机 `kernels/cuda/*.so` 不存在，须远端 `build.sh 103a` + `nm -D` 取** | `ls kernels/cuda/*.so` No such file |
| nsys AR 27.1% #1 | **✅ 采信；且它推翻了「设计口径通信≤20%」，⇒ 400 兑现率门槛抬高** | §4 |
| SH_PAIR &lt;4.9~7.9ms&gt; | 保留（launch 账），但**兑现率须按 §3.3 的 97% 门槛审** | — |

---

*户部 · 只读勘察 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*所有 ms 标来源（实测 / launch 账 / 设计口径 / 代数）；与任务前提冲突处已显式给出依据与 file:line。*
