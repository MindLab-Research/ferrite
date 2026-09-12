# 400 tok/s 冲刺：最终执行计划（HEAD → accept-3 下的 400）

> 中书省 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 基线：工作树 HEAD `206ea26`（含未提交的 mxf4 skeleton）· `DSV41_TIMING` **verify = 37.31ms（m=5）**。
> 输入：`final-400-config.md` · `verify-ms-breakdown.md` · `verify-calc-floor.md` · `dspark-perf-400-plan.md` ·
> `dspark-swallow-step-diff.md` · `routed-expert-residual.md` · `dspark-correctness-chain.md`。
> 本机无 GPU ⇒ 所有 ms 标来源，推算项显式标注。

---

## 0. TL;DR —— 五条判决（先读这个）

1. **任务给的步时分解在算术上不成立**（除非 tcgen05 落地）。
   `7.5 = EAGER 6.15 + draft 0.5 + verify 边际 0.5 + commit 0.35` 隐含假设"verify 的 5 行几乎免费"。
   但 **verify 的边际不是 0.5ms，是 ~5.2ms**——因为 **routed experts 的字节随行数 ×6**
   （**不同行路由到不同专家，不可折叠**；`verify-calc-floor.md §4.1`：30 个 assignment 选 29 个唯一专家，去重仅 3%）。
   **这是本文件对任务描述最重要的更正。** 0.5ms 边际只有在"routed 换成权重驻留核 + 命中 82% 峰值"时才成立——
   那不是可以"假设"的前提，是必须实测的门。

2. **瓶颈既不是 launch，也不是字节，是"算子效率"**。
   今天的事实②（**图化仅 −1.5ms**）直接排除了 launch 主导；
   账本（**纯带宽地板 1.84ms = 37ms 的 5%**）直接排除了字节约束；
   剩下的 **49%（≈18.3ms）是"每发 kernel 的固定延迟 + 低占用"**——verify 是 **6232 个 M=1 级 tiny kernel 的串行链**，
   `14.09GB / 37.31ms = 381GB/s = HBM 峰值的 5.0%`。

3. **唯一的结构性杠杆是 tcgen05 mxf4**（routed 8.3 → 1.5~2.5ms）。
   它把"30 个 per-slot 小 GEMV"变成"几个 M=128 的 masked GEMM"，是**唯一攻击"低占用"这个真实瓶颈**的动作。
   全部其余优化（mrows 折叠族）加起来也只能把 verify 从 37.3 推到 **~12ms ⇒ 250 tok/s**，不是 400。

4. **"把 verify 折进 draft" 是语义错误，必须否决**（§4）。
   draft 跑的是 **MTP 层**（window attention + 自己的 MoE + markov_head 采样），verify 跑的是 **40 层 backbone + head**；
   两者产出的 argmax 来自**不同的参数化**。折叠掉 verify = 去掉验证 = 破坏 spec decoding 的正确性。
   **唯一合法的"消除独立 forward"是 swallow（把主链步折进 verify 行 0）**——已在设计里（`SWALLOW_STEP`），净 −4.55ms。

5. **两个必须先钉死的前置**（否则全表 ±5ms 漂移）：
   - **accept 口径**：`k_acc=3`（⇒ tok/step=4，目标步时 **10.0ms**）还是 `tok/step=3`（目标 **7.5ms**）？
     两者差 2.5ms，**且前者在"无 tcgen05"下也可达、后者必须靠 tcgen05**。
   - **mrows 为什么没兑现**：`{SH_EXP_MROWS, VERIFY_GRAPH, VERIFY_ROPE_MROWS, DRAFT_P3A}` 全开只给 **−1.21ms**
     （`verify-ms-breakdown.md §修正`，预期 −24ms）。这一条决定 §5 走哪条路径。

---

## 1. 口径钉死

### 1.1 今天不可推翻的事实（本文件的地基）

| # | 事实 | 证据（`file:line`） | 对计划的作用 |
|---|---|---|---|
| F1 | 正确性达标：直接 e4m3 单趟 = 官方 MXFP4 语义 | `chain_dev.rs:696 expert_act_e4m3`；`dsv41_experts_mxf4.cu:3805` 起（`act_e4m3` 单字节解码，K 循环/树/epilogue 不变） | 不再是 perf 议题；`tcgen05` 与 e4m3 **互斥**（`chain_dev.rs:687`：`kind::mxf4` 是 e2m1×e2m1，吃不进 e4m3）——**这是路径 A 的硬冲突，见 §7-R1** |
| F2 | verify 37.31ms 是 **GPU kernel 执行时间**主导，不是 launch submit | `verify-ms-breakdown.md §修正`：图化在 pos=20 捕获成功、30/50 步 replay，**replay 35.5ms vs 裸链 37ms ⇒ −1.5ms** | **作废**"50% submit + 50% exec"模型；一切账按"执行时间"重算 |
| F3 | verify 逐族账（m=5）：routed 8.3 / shared 10.4 / head 1.12 / 投影 3.7 / attn 2.8 / hc 2.96 / gate 3.44 / indexer 2.5 / 其它 1.79 | `verify-ms-breakdown.md §1`（自校准 +1.3%） | §3 归因的基础 |
| F4 | SWALLOW_STEP 净 −4.55ms（主链 −6.15，verify +1.6） | `chain_dev.rs:6000-6015` timeline；`chain_dev.rs:1524 swallow_step()` | 唯一合法的"消除独立 forward" |
| F5 | EAGER = 6.15ms（162 tok/s），一次完整 40 层 forward | 任务给定 + `STATUS.md` | §2.2 边际核算的分母 |
| F6 | tcgen05 mxf4 骨架已默认编进 .so，dispatch 已接线，缺 env + 验证 | `kernels/cuda/build.sh:89`；Rust dispatch `chain_dev.rs:10803-10809`；C 入口 `dsv41_experts_mxf4.cu:4007` | 路径 A 的起点 |
| F7 | 用户口径：accept 3（5 中 3 正常） | `dspark-perf-400-plan.md §九` | 目标乘数 |

### 1.2 一处必须推翻的旧假设

`dspark-perf-400-plan.md §六` / `verify-ms-breakdown.md §1-2` 的 **"50% submit（18.7ms）+ 50% exec（18.3ms）"分解已被实测推翻**（F2）。
⇒ **任何"launch 数减半 ⇒ 时间减半"的账都不再成立。** 本文件全部时间账按 kernel 执行时间重算。

---

## 2. 目标分解与算术核对

### 2.1 任务给的分解

```
步时 7.50 = EAGER 6.15（anchor 行 = 主链步被吞）
          + draft 0.50
          + verify 边际 0.50       ← 本文件要挑战的就是这一行
          + commit 0.35
```

### 2.2 硬核对：verify 的边际是 **~5.2ms**，不是 0.5ms

**verify 相对 EAGER 到底多做了什么？**

| 族 | EAGER（m=1） | verify（m=6） | 边际性质 |
|---|---|---|---|
| **routed experts** | 6 assign × 2.6112MB × 40L = **0.63GB** | 36 assign × 2.6112MB × 40L = **3.76GB** | 🔴 **×6，不可折叠**（每行路由到**不同**专家——这是 MoE 的定义；去重仅 3%）|
| 共享专家 | 177MB（读一次）| 177MB（mrows 后）| 🟢 可折叠（同一专家对所有行）|
| 投影 / head / gate / hc / indexer | 各读一次 | 各读一次（mrows 后）| 🟢 可折叠 |
| 激活 / KV 写 | 1× | ~6× | 🟡 小（账本 0.18ms/1×）|

**⇒ verify 的边际 = routed 的 (3.76 − 0.63) = 3.13GB + 激活 ~0.5ms。**

按今天的达成带宽 378GB/s：**3.13GB → 8.3ms**（与 F3 的 routed 8.3ms 自洽）。
即使把 routed 核效率提升到 1.5TB/s，边际仍是 **2.1ms**；要压到 0.5ms 需要 **≥ 6.3TB/s = 峰值的 82%**。

> **结论（本文件最重要的一句）**：只要 routed experts 走 per-token 路由（MoE 的定义），
> **verify 的 6 行边际不可能 ≤0.5ms**。任务的 0.5ms 目标是"最佳情形下勉强成立"，不是可假设的前提。

### 2.3 修正后的步时账（三条候选）

| 路径 | verify | draft | commit | **步时** | @accept 3 | 前提 |
|---|---:|---:|---:|---:|---:|---|
| **P-B 保现核 + 全折叠**（tcgen05 不成）| ~11.3 | 0.5~3.6 | 0.35 | **~12.2** | **246** | 需先解释 §5-D1 的 mrows 异常 |
| **P-A + tcgen05**（routed → ~1.9ms）| ~7.6 | 0.5 | 0.35 | **~8.4** | **357** | T1 微基准达标 |
| **P-A + tcgen05 + 全折叠 + swallow**（理想）| 6.65 | 0.5 | 0.35 | **7.5** | **400** | 三者同时到位 |

**读法**：**tcgen05 一项贡献 −3.8ms（12.2 → 8.4），是其余所有项之和的 2 倍。**
8.4 → 7.5 的最后 −0.9ms 落在 draft（目标 0.5 vs 现状 3.6）与 verify 的激活边际上。

### 2.4 accept 口径的歧义（必须先问，差 2.5ms）

```
tok/s = (tok/step) × 1000 / 步时
```
- **若 `k_acc=3` 且 anchor 也算 1 个 token ⇒ tok/step = 4 ⇒ 目标步时 10.0ms。**
  此时**保现核 + 全折叠（12.2ms）只差 −2.2ms**，P2 折叠族 + draft 优化即可够到
  ⇒ **400 在"无 tcgen05"下也可达。**
- **若 tok/step = 3 ⇒ 目标 7.5ms** ⇒ **必须有 tcgen05。**

**⇒ 这是第一个要向用户确认的问题。它决定 tcgen05 是"必须"还是"锦上添花"。**

---

## 3. Q2：verify 37ms 的真正瓶颈（逐项排除）

### 3.1 三个候选，两个已被今天的事实排除

| 候选 | 判据 | 结论 |
|---|---|---|
| **launch submit 主导** | 图化上限 −15ms（6224 × 2.5µs），实测只给 **−1.5ms** | ❌ 排除 |
| **字节（HBM 带宽）主导** | 折叠后 5.66GB @7TB/s = **0.81ms** = 37ms 的 2% | ❌ 排除 |
| **算子效率 / kernel 粒度** | `14.09GB / 37.31ms = 381GB/s = 峰值 5.0%`；残差半 ≈18.3ms | ✅ **成立** |

### 3.2 瓶颈的精确定位："tiny kernel 的固定延迟 + 低占用"

`6232 launch × (2.9µs submit + 3.3µs 最小执行) = 18.1 + 20.6 = 38.7ms ≈ 实测 37.31ms`（`dspark-perf-400-plan.md §六`）。

- 每小时提交与执行**不重叠**（依赖串行 + 每发 tail 延迟）——但**图化证明 submit 半不是抓手**（F2）；
- **每发 kernel 有一个 ~3.3µs 的最小执行时间**（即便只读几 KB）——这是**固定项，与工作量无关**；
- 6232 × 3.3µs = **20.6ms = 37ms 的 55%**。

**⇒ 机理：不是"launch 开销贵"，是"每个 kernel 的启动/排空 tail 延迟"贵，且因为层与层、行与行之间的依赖串行而无法重叠。**
**这不叫"算子序列太长"，叫"算子序列太长 + 每个算子太小"。** 唯一解法是让每个算子**变大**（更多工作塞进同一发），不是减少发射。

### 3.3 逐族归因：哪些是"重复读"（可折叠）、哪些是"真字节"（不可折叠）

| 族 | 现状 ms | 性质 | 可折叠到 | 折叠手段 |
|---|---:|---|---:|---|
| 共享专家 | 10.40 | 🟢 同一专家 ×6 行读了 6 遍（**纯重复读**）| ~2.1 | `SH_EXP_MROWS`（`chain_dev.rs:952`）|
| **routed experts** | **8.30** | 🔴 **真字节**（36 个 assignment 是 36 个不同专家）| **不可折叠** | **只能换核效率**（tcgen05）|
| 投影族 | 3.70 | 🟡 已 mrows，但肉在固定项 | 1.0 | 已做 |
| **MoE gate** | 3.44 | 🟢 逐行 ×6（`ROW_FOLD_GATE` 默认 OFF，`chain_dev.rs:918`）| 0.69 | 一行 flag |
| hc 链 | 2.96 | 🟢 已 rows=m，但 157MB 跑 53GB/s（**纯占用**）| 1.0 | 并发/融合 |
| attention KV | 2.80 | 🟠 逐行发起（`grid=(b*m,h)` 已支持）| 1.1 | `b·m` 单发 |
| indexer | 2.50 | 🟢 逐行 ×6 + topk 单核地板 | 0.4 | `indexer_rows_one` → `proj_mrows` |
| 其它（norm/AR/压缩/engram）| 1.79 | ⚪ 协议地板 | 1.79 | 减次数 |
| **合计** | **37.1** | | **~9.0**（**不含 routed**）| |

### 3.4 机械地板（这就是为什么 400 需要 tcgen05）

```
EAGER 非 routed 部分        = 6.15 − routed_m1(~1.0)  ≈ 5.15ms   ← 权重读一次，不可再降
+ routed m=6（现有核）        3.76GB @378GB/s          ≈ 9.9ms    ← 只能靠核效率
+ 激活边际                                            ≈ 0.5ms
────────────────────────────────────────────────────────────────
verify 保现核地板                                     ≈ 15.5ms
```
与 `verify-calc-floor.md §5` 的 16.6ms 同量级 ⇒ 模型自校准通过。
**⇒ 即使把所有"每行重复读"全折叠掉，verify 仍 ≥15ms。要到 6.65ms，routed 一项必须从 9.9ms 降到 ~1.0ms。**
**这就是 400 的唯一一道门，而它恰好是 tcgen05 的靶心。**

---

## 4. Q3：能不能把 verify 折进 draft？（结论：**不能**；但有一个合法的替身）

### 4.1 两个 forward 是两套参数

| | draft | verify |
|---|---|---|
| 入口 | `dspark_dev.rs:784 draft_forward` | `chain_dev.rs:4067 step_rows` → `4470 step_rows_inner` |
| 计算 | `for s in 0..n_mtp_layers`：MTP 层（window attention + **自己的 MoE**，`dspark_dev.rs:1535 draft_moe`），block 间**串行** | **40 层 backbone**，m=6 行**并行**（`grid.z = rows`）|
| 出口 | `draft_head()`（`dspark_dev.rs:2060`）：collapse → norm → **markov_head** → 采样 `ids[1..=bs]` + confidence | backbone head + `argmax_sliced_rows` |
| 产出分布来自 | `mtp.last.markov_head.head.weight` | backbone `head.weight`（129280×5120）|

**`draft_forward` 产出的 logits 来自 `markov_head`，`verify` 产出的 argmax 来自 backbone 的 head。两者是不同的参数化：**
draft 的分布是对 backbone 输出的**近似预测**，**verify 的作用正是提供真值去校验它**。
任务描述里的"draft 已经在跑 5 行的 hidden——verify 只是重复 forward 一遍拿 argmax"——
**前半句对（draft 确实跑 5 行），后半句错（那不是同一套参数下的同一件事）。**

### 4.2 折叠 = 删掉验证

`spec_accept`（`chain_dev.rs:6011`，`ferrite_types::spec_accept`）比的是 `drafts[i] == verify_out[i]`
（`chain_dev.rs:6017-6029` 解释了 index 对齐）。
若把 `verify_out` 换成 draft 自己的 argmax，则 `drafts[i] == drafts[i]` **恒真 ⇒ accept 恒满**，
spec decoding 退化成"直接信 draft"。**这不是优化，是删掉正确性。**

### 4.3 合法的替身：swallow —— 把**主链步**折进 verify 的**行 0**

- **原理**（`chain_dev.rs:6000-6015` 的 timeline）：上一轮 verify 的 **行 0（anchor）**
  就是"本轮本该由 `step_dev` 提供的 forward"——它从不需要一次独立的 6.15ms 主链步。
- **现状**：`step_dev` 独立跑一次（6.15ms），`verify` 再跑 6 行。
- **改法**：verify 直接含 anchor 行（m=6），主链步完全省掉 ⇒ **净 −4.55ms**
  （主链 −6.15，verify +1.6；`dspark-swallow-step-diff.md`）。
- **⇒ 这才是任务描述里"去掉一次独立 forward"的正确实现**，且它已被设计与论证（`SWALLOW_STEP`，`chain_dev.rs:1524`）。

---

## 5. 最快可行方案

### 5.0 第 0 步（阻塞其他一切）：解释 mrows 为什么没兑现

**实测**：`{SH_EXP_MROWS, VERIFY_GRAPH, VERIFY_ROPE_MROWS, DRAFT_P3A}` 全开 ⇒ verify = **36.10ms**
（vs 基线 37.31，**仅 −1.21ms**；预期 −24ms）。

两个互斥假设，**必须先分辨**（这决定 §5.1 走 A 还是 B）：

- **H1（dispatch 失效）**：多行分支根本没进。
  - `shared_expert_mrows` 在 `chain_dev.rs:8556` 有 `if !sh_exp_mrows() { ... }` 的**早退**；
  - gate 的 `gemv_bf16_v2_mrows` 在形状/符号不满足时**返回 `Ok(false)`** 而静默保留逐行循环（`chain_dev.rs:8100-8119`）；
  - `proj_mrows` 的 a32 回退是**在 C 端 decline**（`dsv41_kernels.cu:4985`），Rust 侧只看到 `Ok(false)`，
    且**decline 前 staging 已发射**（`verify-calc-floor.md §2.1`：纯浪费 +1760 launch/步）。
  - ⇒ 若是 H1：**收益还在，只差接线/默认值**。工作量小，收益高。
- **H2（byte 模型错）**：mrows 进了，但**耗时不是带宽决定的，是延迟/占用决定的**——
  折叠了字节但没折叠延迟，所以省不下来。
  - ⇒ 若是 H2：**所有"按字节折算"的 ledger 预测（含 §3.3 的折算表）都要作废**，
    §2.3 的 11.3ms 也要上修，**计划的中心从"折叠"转向"换核"（tcgen05）**。

**判据（一次 nsys，不改码 —— 这是全计划性价比最高的一次 GPU 会话）**：
```
DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_TIMING=1 + mrows 全开，nsys profile serve
按 kernel 名聚合 verify 段，看三个名字是否存在、各占多少：
  shared_expert?/gemm_fp8_mrows  /  gemv_bf16_v2_mrows  /  expert_gate_up_fp4_batched(rows>1)
```
- 名字**不存在** ⇒ H1 ⇒ 先修 dispatch。
- 名字**存在但时间没降** ⇒ H2 ⇒ 全力 tcgen05。

### 5.1 三条路径

**路径 A（推荐主攻）：tcgen05 mxf4 routed 落地**
- 现状：kernel 已写（`dsv41_experts_mxf4.cu:3805`，`#ifdef DSV41_TCGEN05_GATEUP_MXF4_SKELETON`），
  `build.sh:89` **默认编译进 .so**，Rust dispatch **已接线**（`chain_dev.rs:10803-10809`），
  C 入口 `dsv41_experts_mxf4.cu:4007`（18-param ABI，含 `ids` per-slot 间接寻址 ⇒ CUDA-graph 安全）。
  只差 `DSV41_EXPERT_TCGEN05_MXF4=1` + 验证。
- 已知缺口：`dsv41_experts_mxf4.cu:4321 [K-SPLIT TODO]`——`slots=8` 下 `grid=(30,8)=240 CTA` 的占用是否够
  （kRing=8 供 28KiB/CTA in flight）；不够则加第三维 K-split + **升序 reduce**（fp 加法非结合，顺序即契约）。
- 门：**单层微基准**（gateup 22.2µs / down 17.2µs；`scripts/dsv41_tcgen05_mxf4_verify.sh --step 3`）——**不达标即止损**。
- 预期：verify **−3.8ms**（12.2 → 8.4）。

**路径 B（兜底，不依赖 tcgen05）：全折叠 + swallow**
- 把 §3.3 的"可折叠族"全部 mrows 化（SH_EXP / gate / hc / indexer / sparse_attn `b·m`），+ swallow。
- 预期：verify 37.3 → **~11~12ms**，步时 **~12.2ms ⇒ 246 tok/s @ accept 3**（或 ~330 @ accept 4）。

**路径 C（口径）：若 accept 口径是 tok/step = 4**
- 目标 10.0ms ⇒ **路径 B 已够**（12.2 − 2.2 = 10.0），tcgen05 降级为加速器。

### 5.2 决策树

```
Q0：accept 口径 = k_acc（⇒ tok/step=4，目标 10ms）还是 tok/step=3（目标 7.5ms）？
  ├─ 4 ⇒ 目标 10ms ⇒ 走路径 B 即可；tcgen05 作加速器
  └─ 3 ⇒ 目标 7.5ms ⇒ tcgen05 是必须
Q1：nsys 诊断（§5.0）= H1 还是 H2？
  ├─ H1 ⇒ 先修 dispatch（1 次 GPU 会话，可能白捡 −8ms）
  └─ H2 ⇒ 折叠族无望，全力 tcgen05
```

---

## 6. 执行清单（按优先级排序）

| # | 工作 | 落点（file:line） | 预期 ms | 工作量 | 前置 |
|---|---|---|---|---|---|
| **0** | **nsys 单次诊断**（mrows 是否 dispatch）| 只读 + `nsys` | 决定后续一切 | **0.5 天** | 无 |
| **1** | **tcgen05 单层微基准门**（go/no-go）| `scripts/dsv41_tcgen05_mxf4_verify.sh --step 3` | go ⇒ −6.8 / no ⇒ 止损 | 0.5 天 | 一份 .so |
| **2** | **tcgen05 集成**（routed 8.3 → ~1.9）| `chain_dev.rs:10803`；缺 `slots=8`/K-split → `dsv41_experts_mxf4.cu:4321` | **−3.8（verify）** | **2~4 天** | #1 达标 |
| **3** | **SWALLOW_STEP 转 ON**（吞主链步）| `chain_dev.rs:1524`；语义点见 §7-R3 | **−4.55（步时）** | 2 天 | `SIDS_WRITEBACK` |
| **4** | **全 mrows 折叠族收敛** | `chain_dev.rs:952`（SH_EXP）/`:918`（gate）/`:8543`（hc）/`:6826`（indexer）/`:7348`（`b·m`）| 名义 −15，**实测算**（取决于 #0）| 1 天 + A/B | #0 结论 |
| **5** | **正确性收口**（accept 的前提）| `SPARSE_OROPE` 对齐 EAGER；`DSV41_SIDS_WRITEBACK` | accept 0.83→3 的前置 | 2 天 | 见 §7-R6 |
| **6** | **draft 质量**（400 的乘数）| tap 层/形状、`main_proj` 顺序、markov 采样、窗口语义 | accept 0.83 → 3 | **3~5 天（最难）** | — |
| **7** | **draft 侧快**（0.5ms 目标）| `DRAFT_P3A`（`dspark_dev.rs:178`）、`MARKOV_SLICED`（:226）、`DRAFT_MOE_MROWS`（:124）| draft 3.6 → ~2.5 | 1 天 | 独立 A/B |
| **8**（可选）| **confidence-gated verify**（按 accept 期望只 verify k+1 行）| 新逻辑 | routed 字节 −25~30% | 3 天 | accept 稳定 |

**做完 #0~#5 ⇒ 步时 ~8.4ms、357 tok/s @ accept 3**（若 accept 口径是 tok/step=4 ⇒ **已过 400**）。
**400（tok/step=3）落在 #6 + #8 上。**

---

## 7. 影响范围与风险

### 7.1 影响范围

**修改文件**：`crates/ferrite-models/src/dsv41/chain_dev.rs`（gate 默认值 + swallow 语义）、
`dspark_dev.rs`（draft 侧）、`kernels/cuda/dsv41_experts_mxf4.cu`（tcgen05 / down）、
`kernels/cuda/build.sh`（skeleton flag）、`device.rs` / `kernels.rs`（若 FFI 变）。

**影响模块**：DSpark spec 步（verify/draft/commit）、CUDA graph 捕获层、TP8 AR v5 足迹、MoE routed 路径。

**兼容性**：**无 API breaking change**（全部 env gate + 默认值，`=0` 可逐一回退）；
**有语义变更**：`SWALLOW_STEP`（m 5→6、commit/tap/pos）；`VERIFY_GRAPH` 形状池**必须按 m=6 重建**。

### 7.2 风险清单

| ID | 风险 | 应对 |
|---|---|---|
| **R1** | **e4m3 × tcgen05 互斥**（`chain_dev.rs:687`：`kind::mxf4` 是 e2m1×e2m1，吃不进 e4m3；`chain_dev.rs:10803` 的 `!e4m3` 也印证）| **这是 F1（正确性）与路径 A（性能）的正面冲突**：开 tcgen05 就要回 e2m1 激活 ⇒ 回到 opa 问题。**必须先确认 tcgen05 支持 e4m3 激活（改 kernel 的 `act_e4m3` 分支），否则路径 A 与 F1 不可兼得**——**这一条必须在 #1 之前澄清** |
| **R2** | mrows bundle 只 −1.21ms（H2 风险）| #0 的 nsys 先分辨 H1/H2；H2 ⇒ 折叠族的全部预期作废，重算 §2.3 |
| **R3** | SWALLOW 的 4 个语义点（commit keep 语义、tap 行下标=bonus 行、`note_ctx_rows` 基址、AR 足迹）| `chain_dev.rs:6000-6058` 已把 commit/emitted/counter 的账写死；逐点纸面推演后同会话 A/B；`出师表逐字 + has_double_char + dspark_parity(verify_bad==0)` |
| **R4** | tcgen05 历史负结果（`STATUS:6607` 的 16.8GB/s = 0.2% 峰值）| **先跑单层微基准门**（gateup 22.2µs / down 17.2µs）；不达标立即关闭路径，**不投变体矩阵**（勿重演 v17→v21 四变体全中性）|
| **R5** | graph × 合并的重复计账 | 全部按"执行时间"重算（F2 已作废 submit 半）；**先做合并、后开 graph** |
| **R6** | 正确性红线（accept 的前提）：o-rope 融合是剩余 row-0 mismatch 根因（`SPARSE_OROPE=0` 把 18→1）；`s.ids` 无人回写 | 先对齐 verify 与 EAGER 的 o-rope 路径；`SIDS_WRITEBACK=1` |
| **R7** | gate 默认值翻转卫生（v13→v15 的 0.37ms 误诊）| 每个默认值翻转**单独 commit + 读回确认**；`git add <specific-files>`（**禁 `git add -A`**）|

---

## 8. 验证计划（最少 GPU 次数 · 单一测试驱动）

**铁律**（`dspark-correctness-chain.md` 会话教训）：同一台远端**同时只能有一个测试驱动**；
subagent 只做代码/分析，**GPU 测试由主 agent 串行执行**；后台输出自动注入，不轮询远端。

| # | 会话 | 内容 | 判据 |
|---|---|---|---|
| **T0** | 本地无 GPU | `build.sh 103a` → `cargo build --release` → `nm -D <so>` + `grep tcgen05` | 符号在、`.build_id` 一致 |
| **T1** | GPU-1（独立，隔离）| **nsys 诊断**（§5.0）| 三个 mrows 核名存在与否 ⇒ H1/H2 |
| **T2** | GPU-2 | **tcgen05 单层微基准** | go/no-go（22.2 / 17.2µs）|
| **T3** | GPU-3 | **swallow A/B**（语义改动，独立）| 出师表逐字 + `dspark_parity` + 步时 −4.5ms |
| **T4** | GPU-4 | **mrows 折叠族 A/B** | 证据 + `verify_ms` 降幅 + 四段文本一致 |
| **T5** | GPU-5 | **终局全开端到端** | 四段文本逐字 + `faults=0` + tok/s |

**止损门**：T1 = H2 ⇒ 关闭折叠路径；T2 不达标 ⇒ tcgen05 默认 OFF、退回 down 4-value。

---

## 9. 建议分工

- **户部（性能）**：**T1 的 nsys 诊断**（§5.0，全计划性价比最高的动作）+ §2.3 步时账的实测校准。
  **原因**：mrows 异常是当前最大的未知，且它决定路径选择；户部已有全部字节/带宽账本。
- **工部（实现）**：**tcgen05 集成**（`slots=8`/K-split reduce/`act_e4m3` 支持）+ **全折叠族 mrows 化**（#4）。
  **原因**：这是唯一能把 12ms 推向 8ms 的通道。
- **刑部（Bug 审查）**：**SWALLOW 的 4 个语义点**（R3）+ R6 正确性红线的合并前静态审计。
  **原因**：改语义处，逐位一致性是硬约束。
- **吏部（质量）**：R7 的 gate 默认值翻转卫生（函数名锚定 + 读回 + 单 gate 单 commit）。
  **原因**：v13→v15 误诊教训直接适用。
- **礼部（文档）**：默认值翻转后同步 `AGENTS.md` / `NEXT-SESSION-HANDOVER.md` / 本文件。
- **兵部（安全）**：**不分配**——本次改动无安全面（无输入解析/鉴权/凭据）。
- **太子（决策）**：**Q0（accept 口径）+ R1（e4m3 × tcgen05 互斥）** 两个问题需要拍板。

---

## 10. 需要太子/用户补充的信息

1. **accept 口径**：`k_acc=3`（⇒ tok/step=4，目标 **10.0ms**）还是 tok/step=3（目标 **7.5ms**）？——**差 2.5ms，决定 tcgen05 是必须还是可选**。
2. **R1 的裁决**：tcgen05 能否支持 e4m3 激活？若否，**性能与正确性正面冲突**，需要用户决定（开 tcgen05 回 e2m1 是否可接受）。
3. **mrows 未兑现的真值**：`{SH_EXP+GRAPH+ROPE+P3A}` 只 −1.21ms 的原因（T1 nsys）。
4. **37.31ms 这一次 `proj_mrows` 是否真生效**（a32 decline 在 C 端不可见）——决定投影族 3.70 还是 8.70ms。

---

*中书省 · 只读分析 + 本文件（唯一产出）；未执行任何 GPU 命令、未改动任何源码。*
*所有「ms」均标了来源；推算项与重叠计账已在 §2.2 / §3.4 / §7-R5 显式标注。*
*本文件对任务描述的核心更正：**verify 的 5 行边际是 ~5.2ms（routed ×6 的字节），不是 0.5ms**（§2.2）。*
