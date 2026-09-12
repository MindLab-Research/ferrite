# 400 tok/s 最终配置与剩余工作清单（"从这里到 400" 的最短路径）

> 中书省 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**，未执行任何 GPU 命令、未改动源码。
> 基线口径：`DSV41_TIMING` **verify=37.31ms（m=5）**，含「投影多行化 / head 词表切分 / o-rope 融合」。
> 输入账本：`dspark-perf-400-plan.md` · `verify-ms-breakdown.md` · `verify-calc-floor.md` ·
> `draft-perf-ledger.md` · `routed-expert-residual.md` · `dspark-correctness-chain.md`。
> 所有 ms 均标注来源；推算项显式标注（本机无 GPU）。

---

## 0. TL;DR（先看这五条）

1. **全部 gate ON + tcgen05 + swallow ⇒ 步时 ≈ 12.0ms**
   （verify 8.2ms @ m=6 + draft 3.6ms + commit 0.2ms；主链步被吞，= 0）。
2. **400 = accept × (1000 / step_ms)**（账本 §九的用户口径）：
   - `12.0ms @ accept 3 → 250 tok/s`（差 1.6×）
   - 12.0ms 下要 400 ⇒ **accept = 4.8**（≈ 5 个 draft 几乎全中，不现实）
   - accept 3 下要 400 ⇒ **步时 ≤ 7.5ms**（在 12.0ms 上还差 **−4.5ms**）
3. **两个乘数同时在短**：性能 12.0 vs 7.5（**1.6×**）；accept ~0.83（mean-k）vs 3（**3.6×**）。
   ⇒ **accept 是更大的缺口，也是 400 的真正乘数**（`dspark-perf-400-plan.md` §八/§九 同判）。
4. **7.5ms 触及架构地板**：verify 的 weight-stationary 下限 ≈ 主链单步 × 1.2~1.5 ≈ **7.4~9.2ms**，
   8.2ms **已在门内**。要到 3.7ms（=总步 7.5 − draft/commit）意味着**整个 forward 再快 2×**——
   **现有任何账本都不支持**（`NEXT-SESSION-HANDOVER.md` §1.3：200 tok/s 已是研究级）。
5. **现实落点带**：**250 tok/s @ (12.0ms, accept 3)**；**330~375 @ (10.0ms, accept 4~4.5)**。
   400 需要在 accept 3 下再砍 **−4.5ms**（§5 的"第二轮合并"），**或**把 accept 提到 **4.8+**。

---

## 1. 目标

**一句话**：在 accept ≥ 3（用户权威口径："预测 5 个能对 3 个比较正常，官方也是这么多"）下，
把 spec 步时压到 **≤ 7.5ms** ⇒ `3 × 133.3 = 400 tok/s`。

交付：
- (a) 全部已落地/已就位优化开启后的**步时计算**（§3）；
- (b) **到 400 的缺口**与 accept×step 组合网格（§4）；
- (c) **剩余工作优先级**：必须做 / 可跳过（§5）；
- (d) **逐 gate 风险清单**（§6）；
- (e) **最少 GPU 测试次数**的验证计划（§7，单一测试驱动）。

---

## 2. 现状分析（口径钉死）

### 2.1 已落地（含在 37.31ms 里）

| 优化 | 落点（file:line） | 默认 | 状态 |
|---|---|---|---|
| 投影多行化 | `chain_dev.rs::proj_mrows`（调用点 :6497） | ON | ✓ |
| head 词表切分 | `chain_dev.rs:1164` `VERIFY_HEAD_SLICED`（`unwrap_or(true)`）+ `verify_head_geom` | ON | ✓ |
| o-rope 融合（P1v） | `chain_dev.rs:1416` `VERIFY_OROPE`（`unwrap_or(true)`）→ `sparse_attn_orope` | ON | ✓ |
| MoE 行批 | `dsv41_experts_mxf4.cu:2487`（`grid.z = rows`，"rows==1 degenerates to previous launch"） | ON | ✓ |

### 2.2 已就位（默认 OFF，只差 A/B 或默认值）

| 优化 | 落点（file:line） | 默认 | 预期 | 备注 |
|---|---|---|---|---|
| 共享专家多行化 | `chain_dev.rs:958` `SH_EXP_MROWS`（实现 :8422/:8534） | OFF | **−8.30ms** | 885→177MB、1000→~200 发 |
| verify 图化 | `chain_dev.rs:1301` `VERIFY_GRAPH`；gate `verify_graph_gate` :2901+ | OFF | **−15ms**（submit 半） | 形状池就位、`publish_key` 已修 |
| verify rope mrows | `chain_dev.rs:907` `VERIFY_ROPE_MROWS` | OFF | −0.60ms | 与 `VERIFY_OROPE` 交互 |
| draft P3a 折叠 | `dspark_dev.rs:178` `DRAFT_P3A`（a1/a2/a4） | OFF | −0.30ms（draft） | 4 个 fold |
| markov 词表切分 | `dspark_dev.rs:228` `MARKOV_SLICED` | OFF | −1.00ms（draft） | draft 侧 |
| 吞主链步 | `chain_dev.rs:1530` `SWALLOW_STEP`（`spec_primed` :1896） | OFF | **净 −4.55ms**（主链 −6.15，verify +1.6） | 见 `dspark-swallow-step-diff.md` |
| e4m3 双趟 | `chain_dev.rs:713` `EXPERT_ACT_E4M3` | OFF | **正确性**（opa 消除），**非 perf** | 见 §2.4 口径校正 |
| tcgen05 mxf4 | `kernels/cuda/build.sh:89`（`-DDSV41_TCGEN05_GATEUP_MXF4_SKELETON=1`，**WORKING TREE 未提交**） | build 进 .so，**函数体未填** | **−6.80ms** | subagent 跑中 |
| down 4-value | `dsv41_experts_mxf4.cu:715-724`（mode 3 @40reg = 0.90×） | OFF | −0.35ms | **被 tcgen05 吞掉**（见 §3 注②） |

### 2.3 verify 的逐族分解（37.31ms 基线）

| 族 | 实测 ms | 对应优化 | 优化后 |
|---|---:|---|---:|
| routed experts | 8.30 | tcgen05（−6.8） | ~1.50 |
| shared expert | 10.40 | SH_EXP_MROWS（−8.3） | ~2.10 |
| head | 1.12 | **已切分 ✓** | 1.12 |
| 投影族 | 3.70 | **多行化 ✓** | 3.70 |
| attention KV | 2.80 | rope mrows + graph | ~2.20* |
| hc 链 | 2.96 | graph + fusion | ~2.96* |
| MoE gate | 3.44 | graph（**gate mrows 需新核**） | ~3.44* |
| indexer | 2.50 | graph | ~2.50* |
| 其余 | 1.79 | graph | ~1.79* |
| **合计** | **37.11**（实测 37.31，+0.5%） | | |

`*` = 这些族不单独标注优化后值，**graph 的 −15ms 是作为"submit 半"整体扣减**（见 §3 注①）。

### 2.4 两处必须钉死的口径（否则表会偏 ±5ms）

1. **graph 与合并的重复计账**：`verify-ms-breakdown.md` §2 的模型是
   `每发全价 ≈ 2.9µs submit + 3.3µs exec ≈ 6.2µs`（`37.31 / 6232 = 5.99µs`）。
   - **graph 只削 submit 半**（2.9 → 0.411µs，`graph_bench` 实测）；
   - **合并**（SH_EXP / tcgen05 的行批）**两半都削**。
   ⇒ 两者**有重叠区**：先 graph 后合并，合并每减一发只省 3.3µs（exec 半）；
   先合并后 graph，graph 的可削基数也随之变小。
   **本文件采用"逐项独立收益相加"（用户给的 gate 表口径），并把此重叠列为 §6-R0 的头号风险**——
   真实落点应在 **8~12ms 之间**，必须用一次 nsys + 一次 A/B 钉死（§7-T2）。
2. **e4m3 的收益性质校正**：任务表把 `EXPERT_ACT_E4M3` 标为"正确性（opa 消除）"——
   与代码一致（`chain_dev.rs:210` "the SECOND pass"、:716 "ARMED-but-undispatchable" 警告）。
   但 `routed-expert-residual.md` §3(c) 的 **−2.0~3.3ms** 是**另一件事**（把 `s_act` 从 f32 改 fp8、
   占用 2→6 CTA/SM、down 的 LDS 指令 ÷4）。**两者不可混算**：
   - 已落地的 e4m3 双趟 = **纯正确性**（本文件不计 perf）；
   - **fp8 激活占用优化** = 可选 perf 项，**−2.0~3.3ms**（§5-P2）。
3. **投影族 V7 口径冲突**：`verify-calc-floor.md` 第二轮开头判定——**a32 回退吃掉 head 的 −1.5ms，
   且还回 `proj_mrows` 的 weight-stationary 收益（net 亏 ~3.9ms）**。
   ⇒ 37.31ms 这一版里 **mrows 是否真在生效存疑**；若投影实际走逐行，投影族应 ~8.7ms 而非 3.70ms。
   **这一项 ±5ms，是 §4 缺口分析的最大不确定源**，必须由 T2 的 nsys 钉死。

---

## 3. 最优配置的步时计算

**配置**：`SH_EXP_MROWS=1`、`VERIFY_GRAPH=1`、`VERIFY_ROPE_MROWS=1`、
`DRAFT_P3A=1`、`MARKOV_SLICED=1`、`SWALLOW_STEP=1`，叠加 tcgen05 mxf4 + down 4-value。

### 3.1 verify（m=6，含 anchor 行）

```
verify(m=5, 现状)                                = 37.31
  − SH_EXP_MROWS        共享专家 10.40 → ~2.10    − 8.30
  − tcgen05 mxf4        routed 8.30 → ~1.50      − 6.80
  − VERIFY_GRAPH        submit 半                 −15.00
  − VERIFY_ROPE_MROWS   rope 行批                  − 0.60
  + SWALLOW 的 anchor 行（5 行 → 6 行）            + 1.60
  ─────────────────────────────────────────────────────
  = verify(m=6)                                  ≈ 8.21 ms
```

**注①**：−15ms 是 6232 发 × (2.9 − 0.411µs) ≈ 15.5ms 的 submit 半口径；
与 SH_EXP/tcgen05 的合并收益有重叠（§2.4-1）⇒ **8.21ms 偏乐观，保守取 8.2~10.5ms**。
**注②**：down 4-value（−0.35ms）**被 tcgen05 吞掉**（tcgen05 替换整段 routed，
落点 1.0~1.5ms 已含 down），**不重复计**；仅在 **tcgen05 不落地** 时作为降级补偿。

### 3.2 draft

```
draft(现状)                          = 4.90
  − DRAFT_P3A（a1+a2+a4 折叠）        − 0.30
  − MARKOV_SLICED（词表切分）          − 1.00
  = draft(最优)                     ≈ 3.60 ms
```
（draft 的字节地板 2.655GB → 0.35~0.89ms；4.9ms 里 83% 是"290 条串行小核"的依赖延迟。
`DRAFT_MOE_MROWS`（−0.2~0.35）与 draft 共享专家缺的 MROWS port（−0.5~0.8，`draft-perf-ledger.md` §3）
**未在 gate 表内**，列为 §5-P2 的可选补刀。）

### 3.3 总步时

```
verify(m=6)   8.21
draft         3.60
主链步        0.00   ← SWALLOW_STEP（anchor 行由 verify 承担，tap 从 bonus 行取）
commit        0.20
──────────────────────
步时        ≈ 12.01 ms  （保守区间 11~14ms）
```

**⇒ `tok/s = accept × (1000 / 12.01) = accept × 83.3`**

---

## 4. 到 400 的缺口

### 4.1 accept × step 网格（400 等值线）

| 步时 ms | 1000/step | **400 所需 accept** | 落点（@accept 3） | 落点（@accept 4.5） |
|---:|---:|---:|---:|---:|
| **12.0**（本文件投影） | 83.3 | **4.80** | **250** | 375 |
| 11.0 | 90.9 | 4.40 | 273 | 409 ✓ |
| 10.0 | 100.0 | 4.00 | 300 | 450 ✓ |
| 9.0 | 111.1 | 3.60 | 333 | 500 ✓ |
| 8.0 | 125.0 | 3.20 | 375 | 563 |
| **7.5** | 133.3 | **3.00** | **400 ✓** | 600 |
| 7.0 | 142.9 | 2.80 | 429 ✓ | 643 |

**读法**：
- 12.0ms 落点上，**accept 3 只有 250 tok/s**——这是"全部已就位优化做完"的现实值。
- **单靠 accept**：12.0ms 需要 accept **4.8**（≈ 5 个 draft 全中）——不可达。
- **单靠性能**：accept 3 需要 **7.5ms**（−4.5ms）——触架构地板（§4.3）。
- **两条腿并用**：`10.0ms + accept 4.0` 或 `9.0ms + accept 3.6` 都到 400——
  **这是唯一现实的组合**（性能再砍 −2ms 是机械可达的，见 §5-P2）。

### 4.2 缺口的两个乘数分解

```
400 = accept × (1000/step)
  现状（correctness 达标那一轮）：accept ≈ 0.83（mean-k），step ≈ 49.8ms → 16.7 tok/s
  全 gate ON 后：                   accept ≈ 0.83，step ≈ 12.0ms  → 69 tok/s
  要 400 ⇒ (accept, step) 必须同时到位：
    · accept 0.83 → 3.0   = 3.6×   ← 更大缺口；**数值/语义问题，非 kernel 问题**
    · step  12.0  → 7.5   = 1.6×   ← 触地板
```

### 4.3 7.5ms 的地板论证（为什么不能靠"再融合"到 7.5）

- **主链单步 = 6.15ms/token**（一次完整 40 层 forward，权重各读一遍）。
- verify 是 6 行的**批处理**，其中 **anchor 行就是主链步本身**；5 个额外行只加激活侧工作量。
  ⇒ verify 的**机械下限 = 主链单步 × (1 + 额外的行侧开销) ≈ 6.15 × 1.2~1.5 = 7.4~9.2ms**
  （`verify-calc-floor.md` §5：全折叠 + 保现核 ≈ 16.6ms，其中 routed 8.30 + 协议地板 10.9ms 已是 5.5ms 的 2 倍）。
- **3.21ms 的 verify 预算（=7.5 − 3.6 − 0.2）比主链单步还小 2×** ⇒
  等价于要求"整个 backbone 的逐层 GEMV 族有效带宽从 373GB/s（峰值 4.9%）翻倍"——
  `NEXT-SESSION-HANDOVER.md` §1.3 明确其为"**无路径**"（gemv 减半无解）。
- **⇒ 结论：`accept 3 + 400 tok/s` 在现有架构与全部已规划优化下不可达。**
  现实目标应写成 **`250~375 tok/s`**，或把 accept 目标改为 **4.0~4.5** 后取 `333~375`。

---

## 5. 剩余工作（按优先级）

### P0 — 必须做（载荷项，决定能不能到 ~12ms）

| # | 工作 | 落点 | 收益 | 前置/证据 |
|---|---|---|---|---|
| 1 | **SH_EXP_MROWS 转默认 ON** | `chain_dev.rs:958` `unwrap_or(false)` → `true` | −8.3ms | 只差 A/B + `dspark_parity` |
| 2 | **VERIFY_GRAPH 转默认 ON** | `chain_dev.rs:1301` | −15ms（submit 半） | `verify_graph_failed` 日志证据；`publish_key` 已修 |
| 3 | **tcgen05 mxf4 routed 落地** | `dsv41_experts_mxf4.cu` `tc5::mxf4` + `build.sh:89` + FFI | −6.8ms | **先过单层微基准门（22.2/17.2µs）** |
| 4 | **SWALLOW_STEP 转 ON** | `chain_dev.rs:1530` `spec_primed` :1896 | 净 −4.55ms | 需先修 §6-R3 的 4 个语义点 |
| 5 | **VERIFY_ROPE_MROWS + DRAFT_P3A + MARKOV_SLICED** | :907 / `dspark_dev.rs:178` / :228 | −1.9ms | 各自逐位 A/B |
| 6 | **正确性红线收口**（见 §6-R6） | `DSV41_SPARSE_OROPE` / `SIDS_WRITEBACK` | accept 0.58→3 的前提 | 出师表逐字 + parity |

**做完 P0 ⇒ 步时 ~12ms、250 tok/s @ accept 3**（P0 是"到 400 的 60%"）。

### P1 — 必须做（400 的第二条腿：accept）

| # | 工作 | 收益 | 依据 |
|---|---|---|---|
| 7 | **draft 数值/语义修复**：tap 层/形状、main_proj 顺序、markov 采样、窗口语义、dtype 路径 | accept 0.83 → **3.0** | `dspark-perf-400-plan.md` §八/§九；`draft-quality-research` |
| 8 | **o-rope 融合对齐 EAGER** | 消除 row-0 mismatch | `dspark-correctness-chain.md`：`SPARSE_OROPE=0` 把 mismatch 18→1（**决定性**） |
| 9 | **`s.ids` 回写默认 ON** | 修 k_acc≥1 后嵌旧 token | `DSV41_SIDS_WRITEBACK`（默认 OFF 待根因 6/7 验证） |

### P2 — 可做（把 12ms 压到 10ms，让 accept 4.0 即可 400）

| # | 工作 | 收益 | 把握 |
|---|---|---|---|
| 10 | **MoE gate 行折叠**（`gemv_bf16_mrows`，nrows=m，与 v2 同 K 序） | −2.75ms | 高（flag 已存在，`verify-calc-floor.md` §6-②） |
| 11 | **sparse_attn 的 `b·m` 单发**（launcher `grid=(b*m,h)` 已支持；前置：行本地 `clen_rows_r` 快照） | −1.7~2.1ms | 中（依赖 B1） |
| 12 | **hc 链融合**（mixes 尾/collapse/post 并核） | −1.96ms | 中 |
| 13 | **indexer 多行**（`indexer_rows_one` 的 lin → proj_mrows） | −0.6ms | 中 |
| 14 | **fp8 激活（占用优化）**：`s_act f32→fp8`，gateup 2→6 CTA/SM + down LDS ÷4 | −2.0~3.3ms | 中（**与 §2.4-2 的 e4m3 区分**） |
| 15 | draft 侧 `DRAFT_MOE_MROWS` + draft 共享专家 MROWS port | draft −0.7~1.15ms | 高 |

`P2` 的可达项合计 **−7~10ms**，但**与 graph 高度重叠**（§2.4-1）⇒
**保守取 −2~4ms ⇒ 步时 8~10ms ⇒ `accept 3` 下 300~375 tok/s**。

### P3 — 明确可跳过（别再投）

| 项 | 跳过原因 |
|---|---|
| **down 的 4-value 解码** | **被 tcgen05 吞掉**；tcgen05 不成时才启用（−0.35ms） |
| **head 折叠**（`VERIFY_HEAD_FOLD=1`） | 与 eager 的 K 序 parity 未证（`FOLD=0` 正是为此默认）；切分已拿 −1.5ms |
| **assignment 去重/合并**（(a)） | production 384 选 6 ⇒ **仅 3.7%**（`routed-expert-residual.md` §3(a)），只作 tcgen05 的附属 |
| **压权重字节的任何尝试** | fp4 已 2 value/B；瓶颈是"每 value 2.5 条 L1TEX 指令"，不是字节 |
| **swapAB(gemv 版) / PDEPTH>1 / w2 prewarm / DL K-chunk / cross-layer pipe** | `NEXT-SESSION-HANDOVER.md` §4 已关闭，**勿重试** |
| head/indexer 的 fp4/cvt 解码路线 | 已被分析否证 |

---

## 6. 风险清单（逐 gate）

| ID | gate / 项 | 已知风险 | 应对 |
|---|---|---|---|
| **R0** | **graph × 合并 的重复计账** | §2.4-1：graph 只削 submit、合并削两半 ⇒ 相加会高估；8.21ms 的 verify 可能落 8~12ms | 用一次 nsys 按 kernel 名聚合 + 一次 A/B 钉死（T2）；**先做合并、后开 graph**（顺序反了浪费 graph 上限） |
| **R1** | **tcgen05 mxf4** | ① `down` 的 swapAB + fused asc-slot reduce **未写**；② 240 CTA 需 `slots=8`，**与 verify 的 `topk=6` 冲突**（需 pad 或 K-split）；③ K-split reduce 未写；④ pool/ids 间接寻址 + Rust FFI 未接；⑤ 无 2D tensor TMA（每 stage 258 条 TMA issue）。历史负结果：`STATUS:6607` 的 16.8GB/s（0.2% 地板） | **先跑单层微基准门**（gateup 22.2µs / down 17.2µs，`slots=8`）；**不达标不进集成**；动态 smem `cudaFuncSetAttribute` **必须 init 期一次性**（捕获内禁用，否则与 VERIFY_GRAPH 冲突）；`build.sh:89` 当前已默认建 mxf4 骨架 ⇒ 骨架进 .so 但函数体未填，**须确认 FFI 返回"不可调度"而非静默错误** |
| **R2** | **VERIFY_GRAPH** | ① **捕获失败按设计静默回退**（"gate 关着/捕获失败/图在跑"三者对外曾无差别）——已修：`step_rows` 打印 `[verify_graph] captured …` / `capture FAILED (…): <reason>`（`chain_dev.rs:4190`）；② 前置 `verify_graph_gate`（:2901+）：`!eng_host && !stats_dbg && !phase_dbg && compress_branch_steady && supports_dspark_snapshot && supports_memset_async && (comm.is_none() || ar_v5)`；③ `ar_v5()` **不是 opt-in（默认恒真）**，不要去改成 opt-in（会让默认整步图失去设备侧 AR）；④ 形状池：`SWALLOW` 的 `m=6` 在 `VERIFY_ROWS=8` 内 ✓（**但 m 从 5→6 必须重建形状池**，否则 replay 命中错形状） | A/B 判据 = **启用证据（capture/replay 日志）+ verify_ms 降 ≥10ms + 双字数一致**；缺证据 exit 2 不出结论；m=6 的形状池按 `SWALLOW` 开关重建 |
| **R3** | **SWALLOW_STEP** | ① **commit keep 语义**：5 行块"保留 `0..k_acc` 行"vs 6 行块"保留 `0..=k_acc` 行"——必须逐行读 `dspark_commit`/`rollback_keep`/`compress_replay` 的行基址与循环边界；② **tap 行下标 = bonus 行**（**不是恒定的行 0**！`dspark-swallow-step-diff.md` §4 已标为"需精确处理的设计点"）；③ `note_ctx_rows(..., pos+1)` 的基址要改 `pos`；④ **AR 足迹**：aligned 首轮 `draft_forward(next, pos+1)` 恒 ≥1 ⇒ 3 次 draft AR，而 legacy 命中 `pos==0` 早退 ⇒ `need−cur=3`（v5 永久自旋死锁）——**已修**（`spec_primed` 引导：首轮走 legacy）；⑤ 依赖 `SIDS_WRITEBACK`；⑥ 首轮引导 `spec_primed` + `reset()` 清零；⑦ `dspark_snapshot(pos, m6)` | 逐点纸面推演后再写；**同会话 A/B** `SWALLOW_STEP=0/1`；出师表逐字 + `has_double_char` + `dspark_parity`；`[dspark] verify=` 应 +1.6ms、步时 −4.5ms |
| **R4** | **SH_EXP_MROWS** | bit-identity 是**假设**（两次 `gemm_fp8_mrows` + `swiglu_limit_q(rows=m)` + `add_inplace`）；mrows kernel **硬编码行距 = k**，调用点必须紧凑缓冲（`moe-rowfold-next` 审计：gemm_fp8_mrows/head_gemv_bf16_mrows 调用点全用紧凑缓冲，安全） | `dspark_parity` verify 行级对照（`verify_bad == 0`） |
| **R5** | **VERIFY_ROPE_MROWS** | 与 `VERIFY_OROPE`（默认 ON）交互：o-rope 侧已被 P1v 接管，**q rope 单独 A/B 不可行**（:899）；必须与 `v2` 的 `gemv_bf16` **同 K 序**（重蹈 FOLD/gemv_a32 覆辙的教训） | 与 ORORE 一起 A/B；行距（rows+step）显式传 |
| **R6** | **正确性红线（accept 的前提）** | ① **o-rope 融合是剩余 row-0 mismatch 的根因**（`SPARSE_OROPE=0` → 18→1，决定性）——"verbatim" 声称不成立，与 FOLD 同类；② `s.ids` 无人回写 ⇒ k_acc≥1 后下一轮嵌旧 token；③ 已修的 2 处行距（`quant_rows` 源行距、`note_ctx_rows` 的 `tap_r` 行距）；④ `DSV41_GEMV_A32=0` 臂因并发测试被 kill，**未确认**是否解释最后 1 个 mismatch | **先对齐 verify 与 EAGER 的 o-rope 路径**（或 EAGER 关融合）；`SIDS_WRITEBACK=1`；重跑 `GEMV_A32=0` 臂（**必须单驱动**） |
| **R7** | **e4m3 双趟** | ILV 冲突：`[inter]` vs `[2*inter]` 契约（round-18 修复）；当前是"两趟 e2m1×2"；默认 OFF 且 `ARMED-but-undispatchable` 一次性警告（:716）⇒ **"设了但不生效"静默** | 用警告日志确认 dispatch；`sub_dequant_fp4` 残差原语与精度口径一次改到位 |
| **R8** | **a32 回退（V7）** | 若 37.31ms 那一次 `proj_mrows` 其实被 a32 decline（`dsv41_kernels.cu:4985`，`g_gemv_a32` 默认 true），投影族应 ~8.7ms ⇒ 全表 42.8ms；**且 decline 后 staging 已发射又落逐行重做（纯浪费）** | T2 的 nsys 数 `gemm_fp8_mrows` vs 逐行 `gemm_fp8_mx`；定后再算缺口 |
| **R9** | **gate 卫生** | 改默认值时 `FileReplace` 必须按**函数名/上下文锚定**并**读回确认**（v13→v15 连续三轮 ~0.37ms 误诊 = `006bd0c` 裸 `int v = 2` 锚点改错同名对象） | 每个默认值翻转**单独 commit + 读回**；`git add <specific-files>`（**禁 `git add -A`**，并行 subagent 下危险） |

---

## 7. 测试计划（最少 GPU 次数 · 单一测试驱动 = 主 agent）

**铁律**（`dspark-correctness-chain.md` 会话教训 1-2）：
**同一台远端同时只能有一个测试驱动**（主 agent 或一个 subagent，不可并发；build-id mismatch 事故的成因）；
**subagent 一律只做代码/分析，GPU 测试由主 agent 串行执行**；后台输出自动注入，**不轮询远端**。

**构建顺序（唯一可行）**：`build.sh ARCH`（重写 `.build_id`）→ `touch crates/ferrite-kernel/build.rs` →
`cargo build --release`。**单独 `cargo build` 永远修不好**（增量 + build.rs 戳记）。

| # | 会话 | 内容 | 判据 | 可否合并 |
|---|---|---|---|---|
| **T0** | 本地（**无 GPU**） | `build.sh` 全 SKELETON flag → `cargo build --release` → `nm -D <so> | grep` 新符号（`dsv41_expert_tcgen05_gate_up_mxf4` 等）→ `cargo test` 编译 | 符号在、编译过、`.build_id` 一致 | 与 T1 同会话（先构建） |
| **T1** | **GPU-1** | **tcgen05 单层微基准**（`slots=8`，240 CTA；gateup 22.2µs / down 17.2µs 门 + parity vs golden） | **go/no-go**：达标才继续；不达标 ⇒ tcgen05 关闭、用 down 4-value 补偿 | 独立（隔离），**必须先于集成** |
| **T2** | **GPU-2** | **"perf bundle" serve 背靠背 A/B**：`{SH_EXP_MROWS, VERIFY_ROPE_MROWS, VERIFY_GRAPH, DRAFT_P3A, MARKOV_SLICED}` vs 全 OFF | `[verify_graph] captured` 证据 + `verify_ms` 降幅 vs 预期（−24~28ms）+ `draft_ms` 降幅 vs 预期（−1.3ms）+ **四段文本逐字相同** + `faults=0` | 一轮出 5 个 gate 的**族级** delta（verify_ms/draft_ms 是两个独立计数器） |
| **T3** | **GPU-3** | **swallow A/B**（**语义改动**，必须独立）：`SWALLOW_STEP=0/1` | 出师表逐字 + `has_double_char` + 数字任务 + `dspark_parity`（`verify_bad==0`）+ `[tick] total` 步时 −4.5ms、`[dspark] verify=` +1.6ms | 独立（m=6、commit/tap 语义，不与 T2 混） |
| **T4** | **GPU-4** | **tcgen05 集成 A/B**（T1 通过后）：在 T2+T3 稳定配置上 `EXPERT_TCGEN05=0/1` | `verify_ms` −6.8ms + 四段文本一致 | 可并入 T5 的终局 |
| **T5** | **GPU-5** | **终局全开端到端**：全部 P0 默认 ON，跑 `A/B` 脚本 | **四段文本逐字 + `faults=0` + p50 下降 + tok/s 记录**；与 EAGER 同水平 | 最终验收（必须独立、全开） |

**最少次数 = 4 个独立 GPU 会话**（T1 微基准 / T2 perf bundle A/B / T3 swallow A/B / T5 终局），
**若 tcgen05 集成不并入终局则 5 次**。任一 FAIL ⇒ 追加 1~2 次 bisect（按 R0 的顺序：先合并、后 graph）。

**止损门**：
- T1 微基准不达标 ⇒ tcgen05 默认 OFF、退回 down 4-value（**立即关闭路径，不投变体矩阵**，
  勿重演 v17→v21 四变体全中性）；
- T2 的 `verify_ms` 降幅 < 预期的 60% ⇒ 先查 R0（graph×合并重叠）与 R8（a32/V7），
  再决定是否拆开单 gate 重测。

---

## 8. 影响范围

**修改文件**（全部已提交 / 工作树）：
- `crates/ferrite-models/src/dsv41/chain_dev.rs`（默认值翻转 ×4 + swallow 语义）
- `crates/ferrite-models/src/dsv41/dspark_dev.rs`（draft gate 默认值）
- `kernels/cuda/build.sh`（**未提交**：mxf4 skeleton 默认建）
- `kernels/cuda/dsv41_experts_mxf4.cu`（**未提交**：tcgen05/down 4-value）
- `crates/ferrite-models/src/dsv41/device.rs` / `kernels.rs`（FFI 签名，若接 tcgen05）

**影响模块**：DSpark spec 步（verify/draft/commit 三段）、CUDA graph 捕获层、TP8 AR v5 足迹、MoE routed 路径。

**兼容性**：
- **无 API breaking change**；全部为 env gate + 默认值，`=0` 可逐一回退。
- **有语义变更**：`SWALLOW_STEP`（m 5→6、commit/tap/pos 语义）；
  `VERIFY_GRAPH` 的形状池 **必须按 m=6 重建**。
- **正确性 alignment**：`SPARSE_OROPE`（verify 与 EAGER 同核）——**这是一处语义对齐，不是纯 perf**。

---

## 9. 建议分工

- **工部**（实现）：
  P2 的第二轮合并（MoE gate 行折叠 `gemv_bf16_mrows` / sparse_attn `b·m` 单发 / hc 链融合 / indexer 多行）
  + tcgen05 的 5 个缺口（down swapAB、slots=8、K-split reduce、pool/ids FFI、2D TMA）——
  **这是唯一能把 12ms 推向 10ms 的通道**，也是 400 的第二条腿。
- **刑部**（Bug 审查）：
  §6-R3（swallow 的 commit/tap/AR 足迹四点）与 R6（o-rope alignment / s.ids / GEMV_A32 臂）
  的**合并前静态审计**——这些是"改语义"的地方，逐位一致性是硬约束。
- **户部**（性能）：
  §2.4/R0/R8 的**口径钉死**（一次 nsys 按 kernel 名聚合）与**accept × step 网格的实测校准**；
  维护 `verify-ms-breakdown.md` / `draft-perf-ledger.md` 的最终账本。
- **吏部**（质量）：
  §6-R9 的 **gate 默认值翻转卫生**（函数名锚定 + 读回确认 + 单 gate 单 commit）——
  v13→v15 误诊的教训直接适用。
- **礼部**（文档）：
  默认值翻转后同步 `AGENTS.md` / `NEXT-SESSION-HANDOVER.md` / 本文件；
  `dspark-correctness-chain.md` 的结论指针。
- **兵部**（安全）：**不分配**——本次改动无安全面（无输入解析/鉴权/凭据处理）。

---

## 10. 需要太子/用户补充的信息（不补则 §3~§4 有 ±5ms 漂移）

1. **a32/V7 的真值**：37.31ms 那一次 `proj_mrows` 是否真生效？
   （`DSV41_GEMV_A32` 默认 true，C 端 decline 在 Rust 侧不可见）——决定投影族 3.70 还是 8.70ms。
2. **`dim=7168 / inter_local=256 / topk=8` 的来源**（`routed-expert-residual.md` §0/§5 的口径冲突）：
   若确是新配置，routed 的字节/去重因子要整表重算（3133MB → 4679MB）。
3. **accept 的目标口径**：用户"能对 3 个"——是 `k_acc=3`（⇒ tok/step=4）还是 `tok/step=3`？
   两者对 400 的步时要求差 **2.5ms**（7.5 vs 10.0ms）。
4. **tcgen05 是否仍在关键路径**：T1 微基准未跑 ⇒ 若 NO，则 routed 只能收 (c)+(d) ≈ −2.4~3.7ms，
   §3 的 verify 要回退到 ~14.5ms。

---

*中书省 · 只读分析 + 本文件（唯一产出）；未执行任何 GPU 命令、未改动任何源码。*
*所有「ms」均标了来源；推算项与重叠计账已在 §2.4 / §6-R0 显式标注。*
