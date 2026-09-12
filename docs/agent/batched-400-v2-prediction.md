# batched_400_v2 预分析：结果解读框架 + 逐档行动预案

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行任何 GPU 命令、未改动任何源码。
> 本文件 **v2 修订**：修正了初版 §2 采用的 `VERIFY_GRAPH −15ms` 估算——**该估算已被同会话实测推翻**
> （实测 −1.5ms，见 §1.2）。初版曾被同伴的提交 `6f5de2b` 一并纳入，本版为准。
> 代码基线：工作树 HEAD `6f5de2b`。脚本：`scripts/batched_400_v2.sh`（667 行）。
> 输入：`final-400-config.md` · `dspark-correctness-chain.md` · `swallow-step-400-necessity.md` ·
> `verify-ms-breakdown.md（§修正）` · `batched-400-v2-remaining-roi.md`（同伴，14:50）· 以及 `chain_dev.rs` 现场核对。

---

## 0. 三个与任务前提冲突的现场事实

任务把本轮配置写成 **「SWALLOW_STEP + 全 mrows + tcgen05 + BF16_TRUNCATE」**。逐字核对后，**两项不成立**：

| 任务前提 | 脚本实际 | 证据 |
|---|---|---|
| tcgen05 参与本轮 | **❌ 完全不在矩阵里** | `grep -n "TCGEN05\|GROUPED\|EXPERT_ILV" scripts/batched_400_v2.sh` → **0 命中**；`GATES`（:134-144）只有 SPEC/DSPARK/SIDS_WRITEBACK/E4M3/BF16_TRUNCATE/六个 mrows/两个 graph/SWALLOW/TIMING/DSPARK_DEBUG |
| 本轮 accept ≈ 1.214 | **❌ 会是基线 ~1.022** | accept 杠杆 `DSV41_TAP_INPUT` / `DSV41_DRAFT_BF16_DOMAIN` **不在矩阵**；`dspark-correctness-chain.md:1346-1350` 自己写明「脚本不含 P0-3+P1-5 杠杆…accept 会是基线 ~1.022」 |
| 「全 mrows」隐含收益 | **⚠️ 实测几乎为零** | `{SH_EXP+GRAPH+ROPE+P3A}` 全开实测 **−1.21ms**（`verify-ms-breakdown.md:159`）；SH_EXP 单独已两次实测零收益（instruction-bound）|

**⇒ 这一轮测的是「无 tcgen05 的 batched 400」，而且 mrows 族的实测兑现度 ≈ 0。**
`final-400-config.md §3.1` 的 `verify(m=6) ≈ 8.21ms` **内含 tcgen05(−6.8) + SH_EXP(−8.3) + graph(−15)**——
**三项里本轮只带了一项，而那一项实测只有 −1.5ms**（§1.2）。**≤10ms 这一档在当前代码上不可达。**

---

## 1. 与同伴文档的对齐（两处必须钉死的实测）

### 1.1 图化：**不是性能杠杆**
同会话 A/B（`verify-ms-breakdown.md:157-168`）：
```
裸链 verify                                                    = 37.31ms
{SH_EXP_MROWS, VERIFY_GRAPH, VERIFY_ROPE_MROWS, DRAFT_P3A} 全开 = 36.10ms   ⇒ −1.21ms
图化 replay 步时 ≈ 35.5ms vs 裸链 37ms                                     ⇒ 约 −1.5ms
```
**原因**：CUDA async launch 已让 CPU submit 与 GPU 执行**重叠**——"50% submit + 50% exec" 的分解是错的。
`final-400-config.md` R2/§3.1 的 `−15ms`（"submit 半"）**是理论口径，已被实测推翻**。
⇒ 图化的定位是**正确性/顺序工具**，不要为它编性能预算。**任何拿 −15ms 算的步时预测都偏乐观 15ms。**

### 1.2 传导到本轮的预测
本轮矩阵 = 上面那组 `{SH_EXP+GRAPH+ROPE+P3A}`（≈ −1.2ms）+ `GATE_MROWS` + `HEAD/INDEXER/NORM/COMPRESSOR_MROWS` + `SWALLOW`。
- `SH_EXP/GRAPH/ROPE/P3A`：**已含在上述实测的 −1.21ms 里**。
- `GATE_MROWS`：预期 −2.75ms（**未实测**，`chain_dev.rs:10774+` 的 fold 逻辑完整）。
- `HEAD_MROWS`：死门已修（§3.3），但 head 族基线仅 1.12ms，上限小。
- `INDEXER/NORM/COMPRESSOR_MROWS`：**未实测，且没有 decline 日志**（§3.2）。
- `SWALLOW`：verify 从 m=5 → m=6，**+1.6ms**（anchor 行），整步净 −4.55ms（主链步消失）。

---

## 2. 步时预期（两个锚，含显式不确定度）

### 锚 A：batched（**本轮走的就是这条**，`verify-ms-breakdown.md` 文档锚）
```
裸链 verify(m=5)                                        = 37.31ms   （6232 发，几乎全是 GPU 执行）
  {SH_EXP+GRAPH+ROPE+P3A}                               − 1.21      （实测）
  GATE_MROWS                                            − 2.75      （未实测，预期）
  HEAD/INDEXER/NORM/COMPRESSOR_MROWS                    − 0 ~ 2     （未实测，保守）
  SWALLOW 的 anchor 行（m=6）                            + 1.6
  ─────────────────────────────────────────────────────────────────
  verify(m=6)                                           ≈ 34 ~ 37ms
  + draft (DRAFT_GRAPH + P3A) ≈ 3.6 ~ 4.9
  + commit                    ≈ 0.2
  ─────────────────────────────────────────────────────────────────
  步时                                                  ≈ 38 ~ 42ms
```
**但要注意**：37.31ms 是「batched verify」的文档基线。它与已测的 **lazy 步时 22.56ms**
（`410f57b`：92 tok/s = 22.56ms × accept 1.08）**并不矛盾**——lazy 每行 m=1（权重驻留），
batched 是 6 行一次权重读但激活侧 ×m，而 **mrows 的权重共享对 instruction-bound 核无效**
（SH_EXP 两次实测零收益）。⇒ **当前 batched 比 lazy 慢**，这正是 `25ac239` 记录的
「batched 停在 LEN=142、lazy 到 LEN=201」的同一现象。

### 锚 B：lazy（**不是本轮**，仅作量级参照）
`steady_median ≈ 22.6ms`（实测）。**若本轮读数接近这个值，先确认走的不是 lazy**
（脚本已硬校验 `DSV41_LAZY_VERIFY` 不在 env 里）。

### 落点判断
| 档 | 判据 | 与本轮证据的一致性 |
|---|---|---|
| A `≤10ms` | | **与全部实测矛盾**（39ms 量级的 verify + 0 兑现的 mrows），几乎不可能 |
| B `15-20ms` | | 只有 lazy 达到过；batched 若落这里，需 `vg_shapes` 含 m=6 + mrows 兑现证据 |
| C `>25ms` | | **与锚 A 一致 ⇒ 最可能** |

**⇒ 票面预期：`steady_median ≈ 35~42ms`、`mean_k ≈ 1.02`、`tok/s ≈ 50~60`、`latin=0`、`vg_shapes` 含 `m=5,m=6`。**
**若落在这里，是账本预期内的结果，不是失败**——真正的缺口是 **tcgen05 未 armed（−6.8ms，最大单项）** 与 **accept 无杠杆**。

---

## 3. 四个检查点的现场机制

### 3.1 m=6 形状池 capture —— 机制没问题，但三态必须分清
- `VERIFY_ROWS = 6`（`chain_dev.rs:84`），**m=6 恰好是上限**，不是超限。
- 池是 **per-shape 槽**（`VERIFY_GRAPH_SLOTS = 3`，:107；`verify_slot` :5340，`verify_graph_gate` :5349）：
  m=5（首轮 legacy bootstrap）与 m=6（之后 swallow）各自独立 DRY→CAPTURE→REPLAY，**不需要重建池**。
- **三态分辨**（脚本 :353-372）：① 无 `[verify_graph]` 行 = gate 没开；② `capture FAILED (m=6)` = 设计内降级（带原因）；③ `captured verify_graph_m6` = 真接上。
- **前置**（:5349-5363）：`pos_base >= 1`、`!eng_host && !stats_dbg && !phase_dbg`、`compress_branch_steady`（所有 compressor 层 `compress_len > 0`，**早期步不满足**）、`supports_dspark_snapshot`、`supports_memset_async`、`comm.none || ar_v5()`。脚本没设 `STATS/PHASE/ENG_HOST`，这几项应过。
- ⚠️ **即使 capture 失败，本轮也不该归因于「SWALLOW 没工作」**：图化只值 ~1.5ms（§1.1），SWALLOW 的价值在整步 −4.55ms。**两者要分开读。**

### 3.2 mrows 的 decline 日志 —— **只有两个 gate 有，其余静默**
任务说「用新的 decline 日志分析哪些 gate declined」。现场核对：**全局只有两条 mrows 相关 one-shot 日志**：

| gate | 可观测性 | 落点 |
|---|---|---|
| `VERIFY_HEAD_MROWS` | ✅ `verify_head_mrows_note` | :1612，调用点 :5731 / :5778 |
| `ATTN_MROWS` | ✅ `attn_mrows_decline` | :2050 |
| `SH_EXP` / `GATE` / `INDEXER` / `NORM` / `COMPRESSOR_MROWS` | **❌ 静默 `return Ok(false)`** | 例：SH_EXP 六条件 `if !sh_exp_mrows() \|\| … { return Ok(false) }`（:11435+），**不打印** |

⇒ 任务是「**部分不可行**」。这五个 gate 是 armed-but-silent-fallback，**日志里查不到**。
判读只能靠：(a) `.so` symbol 存在性；(b) 行为对照（verify_ms 是否移动）；(c) **一次 nsys 按 kernel 名聚合**
（`gemm_fp8_mrows` vs 逐行 `gemm_fp8`、`gemv_bf16_v2_mrows` vs `gemv_bf16`）。
**不要在缺证据时对某个 mrows gate 下「declined」结论**（项目铁律：缺证据 exit 2）。

### 3.3 HEAD 死门已修
`chain_dev.rs:5511-5519` 文档注释明确：`step_rows_inner` 的**未切分 arm 现在也用同一个 v1 kernel 折叠全词表 head**，
`None` 不再让 gate 静默 no-op；`940e895`（head v1 mrows）在树内。
⇒ HEAD 现在**可能**兑现收益（head 族基线 1.12ms，上限本就小）。

### 3.4 tcgen05 的 5-gate 链 —— **本轮 0/N，且被多重独立阻塞**
真实链（`chain_dev.rs:13950+` / :10540+）：
```
tc_e4m3  = e4m3 && expert_tcgen05_e4m3() && supports_expert_tcgen05_e4m3() && !ilv
tc_mxf4  = !e4m3 && expert_tcgen05_mxf4() && supports_expert_tcgen05_mxf4() && !ilv
grouped  = DSV41_EXPERT_GROUPED && tc_e4m3 && !gateup_fused && !ilv && shape && symbol
```
| 门 | 本轮 | 依据 |
|---|---|---|
| `DSV41_EXPERT_TCGEN05_E4M3`（严格 `starts_with('1')`）| **未设** | 脚本无 |
| `DSV41_EXPERT_TCGEN05_MXF4` / `_TCGEN05` | **未设** | 脚本无 |
| `DSV41_EXPERT_GROUPED` | **未设**（默认 OFF）| 脚本无 |
| `!ilv`（`DSV41_EXPERT_ILV=0`）| **未设 ⇒ ilv 默认 ON** | `gateup_ilv()` = `unwrap_or(true)`（`weights.rs:489`）|
| **`e4m3 ⊥ mxf4`** | 脚本设了 `E4M3=1` ⇒ `tc_mxf4` 恒 false | :13950 的 `!e4m3` |

⇒ **即使补 gate，ILV 默认 ON 也会拒绝 tcgen05**（两个 arm 都要 `!ilv`）——这是「5-gate 链」之外的**独立阻塞**。
⇒ **routed experts 本轮走 `expert_gate_up_fp4_batched`（e4m3 单趟）**，不含 tcgen05 的 −6.8ms。

---

## 4. 逐档行动预案

### 档 A：`steady_median ≤ 10ms`
- **先质疑测量**（与全部实测矛盾）：确认 `vg_shapes` 含 `m=6`、`dg_engaged=captured`、
  `<tag>.env` 里**没有** `LAZY_VERIFY`（脚本会 exit 2，但确认一遍）、`rounds` 数量与 1000 token 匹配。
- 若证据齐全：这是重大正面结果 ⇒ 立刻补两轮**独立**测试：
  1. **accept 轮**：`+DSV41_TAP_INPUT=1 DSV41_DRAFT_BF16_DOMAIN=1`（预期 accept → 1.214），单独一轮。
  2. **tcgen05 轮**：`DSV41_EXPERT_ACT_E4M3=1 DSV41_EXPERT_TCGEN05_E4M3=1 DSV41_EXPERT_GROUPED=1 DSV41_GATEUP_FUSE=0 DSV41_EXPERT_ILV=0`
     （四件套**必须成组**，`ILV=0` 改权重布局，不能与 ILV=1 的结果比绝对值）。

### 档 B：`steady_median 15-20ms`
- 只有 lazy 达到过（22.56ms）。若 batched 落这里，说明 mrows/graph 兑现度**远高于**实测历史 ⇒ 需要证据：
  - `vg_shapes` 必须含 `m=6`；否则 graph 没接上 6 行块。
  - 用 **nsys 一次**钉死 mrows 族真实 dispatch（§3.2 是唯一取证手段）。
- **动作**：不要改代码。先出「哪一项比实测好、好多少」的证据，再决定下一步。

### 档 C：`steady_median > 25ms` ← **最可能**
- **含义**：与锚 A 一致（mrows 零兑现 + 无 tcgen05）。**不是 bug**。
- **动作**（按 ROI，直接引用同伴 `batched-400-v2-remaining-roi.md` §6）：
  1. **hc 链**（−1.3~−1.7ms）：`HC_VERIFY_FUSE` + `HC_FRONT_ROWS`；**前置**是 A2 的 `truncate=false`——
     该修复已在 `6f5de2b` 落地（本轮矩阵仍不开，属下一轮）。
  2. **indexer front**（−1.0~−1.5ms）：`DSV41_INDEXER_MROWS` 代码已就位，零成本开。
  3. **tcgen05**（−5~−6.8ms）：唯一能越过地板的大象，但需 e4m3 parity 前置 + ILV=0。
- **止损**：不要在同一轮里改代码；先确认 §3.1/§4 的日志证据（swallow 是否真分派：`verify_ms` 是否 +1.6ms）。

### 档 D：出现拉丁 / 双字 / 缺句（红线破）
- **红线优先于性能**。二分隔离（每步单独一轮，`LOCK` 保证串行）：
  1. **`SWALLOW_STEP=0`**（保留其余）：swallow × `SIDS_WRITEBACK` 是本轮**唯一的新语义组合**，
     且 `SIDS_WRITEBACK` 被文档标注为「会放大 verify 数值误差 ⇒ 更早 collapse」⇒ **第一嫌疑**。
  2. **`VERIFY_GRAPH=0`**：capture/replay 若与 host mirror（`compress_lens`）不同步，会静默改状态。
  3. **`EXPERT_ACT_E4M3=0`**：`E4M3=1` + ILV 默认 ON 的组合**无历史 A/B 证据**。
  4. **`DSV41_SIDS_WRITEBACK=0`**：单独验证 swallow 的硬依赖假设。
- **`HC_VERIFY_FUSE` 排除**：已在 `46de662` 默认 OFF，脚本 `FORBIDDEN` 硬校验（:498-504）⇒ 本轮不可能是它。

---

## 5. 交付前 3 个零成本自检

1. **env 实证**：脚本把 `/proc/<pid>/environ` 落到 `<tag>.env`（:494）。
   `grep -E 'TCGEN05|GROUPED|EXPERT_ILV|GATEUP_FUSE' <tag>.env` ⇒ **应为空**（§0 的现场事实）。
   若非空，说明 shell export 泄漏（脚本只挡 3 个 FORBIDDEN，不挡这 5 个）⇒ 本轮配置与任务描述不同，读数作废。
2. **build-id 同源**：`embeds_id` 用 `strings | grep -cF`（不是 `-q`，避开 SIGPIPE 假失败）。
   `--no-build` 跑时先确认 `.build_id` 与 binary 内嵌 id 一致。
3. **`rounds` 与 1000 token 匹配**：tok/step≈2 ⇒ 期望 ~450 轮。若 <100，检查是否提前 EOS
   （`chars` 会同步变小）——**不是性能问题，是可比性问题**。

---

## 6. 判定树 + 票面预期

```
读 steady_median（唯一可信步时，median 而非 mean）：
  ≤10ms  → 先质疑测量（与全部实测矛盾）；证据齐全则补 accept 轮 + tcgen05 四件套
  15-20  → 确认 vg_shapes 含 m=6 + nsys 钉 mrows dispatch；不改代码
  >25    → 【最可能】与锚 A 一致；按 ROI 走 hc → indexer front → tcgen05
  红线破 → 二分：SWALLOW=0 → VERIFY_GRAPH=0 → E4M3=0 → SIDS_WRITEBACK=0
```

**票面预期**：`verify(m=6) ≈ 34~37ms`、`步时 ≈ 35~42ms`、`mean_k ≈ 1.02`、`tok/s ≈ 50~60`、
`vg_shapes` 含 `m=5,m=6`、`latin=0`。

**读数的三句真话**：
1. **图化不是性能项**（实测 −1.5ms），任何含 −15ms 的预测都偏乐观 15ms。
2. **mrows 族本轮几乎不兑现**（实测 −1.21ms / 预期 −8.3ms），且五个 gate 无日志可查。
3. **最大缺口是 tcgen05 未 armed（−6.8ms）**，而它被「脚本未设 + E4M3 与 mxf4 互斥 + ILV 默认 ON」三重阻塞。

---

*工部 · 只读分析 + 本文件（唯一产出）；未执行任何 GPU 命令、未改动任何源码。*
*所有 ms/launch 数均标注来源；推算项（§2 落点带）已显式标注。*
*v2 修订：§1 的实测 −1.5ms 取代初版采用的 −15ms 理论口径。*
