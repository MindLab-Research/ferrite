# batched_400_v2 预分析：结果解读框架 + 逐档行动预案

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行任何 GPU 命令、未改动任何源码。
> 代码基线：工作树 HEAD `92f12e4`（clean）。脚本：`scripts/batched_400_v2.sh`（667 行）。
> 输入：`final-400-config.md` · `dspark-correctness-chain.md`（14:45 版）· `swallow-step-400-necessity.md` ·
> `dspark-swallow-step-diff.md` · `expert-tcgen05-plan.md`，以及 `chain_dev.rs` / `dspark_dev.rs` / `build.sh` 的现场核对。

---

## 0. 三个与任务前提冲突的现场事实（先看这个）

任务描述把本轮配置写成 **「SWALLOW_STEP + 全 mrows + tcgen05 + BF16_TRUNCATE」**。
逐字核对脚本后，**其中两项不成立**：

| 任务前提 | 脚本实际 | 证据 |
|---|---|---|
| tcgen05 参与本轮 | **❌ 完全不在矩阵里** | `grep -n "TCGEN05\|GROUPED\|EXPERT_ILV" scripts/batched_400_v2.sh` → **0 命中**；`GATES` 变量（:134-144）只有 SPEC/DSPARK/SIDS_WRITEBACK/E4M3/BF16_TRUNCATE/六个 mrows/两个 graph/SWALLOW/TIMING/DSPARK_DEBUG |
| 本轮 accept ≈ 1.214 | **❌ 会是基线 ~1.022** | accept 杠杆 `DSV41_TAP_INPUT` / `DSV41_DRAFT_BF16_DOMAIN` **不在矩阵**；`dspark-correctness-chain.md:1346-1350` 自己写明「脚本不含 P0-3+P1-5 accept 杠杆…accept 会是基线 ~1.022」 |
| 「全 mrows」隐含收益 | **⚠️ 只有 GATE/HEAD 可能兑现** | SH_EXP_MROWS 已两次实测 **零收益**（instruction-bound，`dspark-correctness-chain.md:1251`）；HEAD 的死门已修（见 §3.3） |

**结论**：这一轮测的是 **「无 tcgen05 的 batched 400」**。
`final-400-config.md` §3.1 的 `verify(m=6) ≈ 8.21ms` **内含 tcgen05 的 −6.8ms**——
把这个前提去掉，**≤10ms 这一档在本轮基本不可达**（见 §2）。任务的「结果 1」预期需要重写。

---

## 1. 关键指标的解读框架

### 1.1 判据字段来源（脚本 parser，`.metrics`）

| 字段 | 来源 pattern | 读法 |
|---|---|---|
| `steady_mean/median/min/p10` | `[dsv41] step pos=N: X.XXms`（`serve.rs:460`）| **唯一可信的步时**：前 20 轮热身丢弃、末轮丢弃（`STEADY_SKIP=20`）。判据用 **median**，不看 mean |
| `verify_ms` / `draft_ms` / `commit_ms` | `[dspark] steps=… mean-k=…` 行（**须 `DSV41_TIMING=1`**）| serve 进程级均值；**不与 steady 混用** |
| `tok_step` | 同行 `tok/step=` | = `mean-k + 1`。**400 的分母关键** |
| `mean_k` / `kacc_mean` / `hist0..6` | 同行 + `[dsv41] step pos=` 的 **位置差 − 1** | 两路交叉验证 accept；不相等 = commit 语义漂了（swallow 的 commit 已核：`pos_ctr = pos + k_emit`，delta 恒等于 k_emit ⇒ 口径可比） |
| `vg_engaged` / `vg_shapes` | `[verify_graph] captured verify_graph_m{m}` | **必须看到 `m=6`**；只有 `m=5` = swallow 的 6 行块没进图 |
| `vg_failed_shapes` | `capture FAILED (m={m}` | 见 §3.1 三态分辨 |
| `dg_engaged` | `[draft_graph] captured … at pos=` | draft 图在 `pos >= win` 才 arm，短答可能永不 capture |
| `latin` / `dbl` / `has_kaishen` / `chars` | resp.json 正文 | **红线**：latin=0 ∧ dbl=0 ∧ 先帝创业未半=yes ∧ chars>0 |

### 1.2 三个必须避开的读数坑

1. **`[dsv41] step pos=` 行尾的 `(X tok/s)` 标错了**（`serve.rs:456-465` 打印的是 `1.0/dt` = **steps/s**）。
   spec 步每步 emit `k_acc+1 > 1` 个 token ⇒ 该标签在任何 accept>0 时都偏小。
   **真 tok/s = tok_step × 1000 / steady_median_ms**。脚本正文只印 step/s，别把它当吞吐。
2. **verify 绝对值变大 ≠ 失败**。SWALLOW 让 verify 从 m=5 变 m=6，`verify_ms` **应 +1.6ms**；
   判据是**整步**（主链步消失，净 −4.55ms）。拿 verify 绝对值当止损条件是误判源（`swallow-step-400-necessity.md` §5.3）。
3. **`mean_k` 是 serve 进程均值，不随请求重置** ⇒ 一轮 serve **只能跑一个请求**（脚本已用 `LOCK` + 单请求强制）。

### 1.3 accept → 吞吐的精确换算（任务里的算术需要修正）

```
tok/s = (mean_k + 1) × 1000 / step_ms
```

| mean_k | tok/step | 400 tok/s 所需 step | 200 tok/s 所需 step |
|---:|---:|---:|---:|
| 1.022（**本轮预期**）| 2.022 | **5.06ms**（不可达）| 10.11ms |
| 1.214（含 accept 杠杆）| 2.214 | 5.54ms（不可达）| 11.07ms |
| 2.0 | 3.0 | 7.50ms | 15.0ms |
| 3.0（用户上限）| 4.0 | **10.00ms** | 20.0ms |

⇒ **「步时 ≤10ms 即为 400」只在 accept=3 时成立**。本轮 accept≈1.022，
即使步时压到 10ms，也只有 **≈202 tok/s**。任务「结果 1」里写的「400 路径确认」**不成立**；
正确表述是「**性能侧确认**（10ms 到位），吞吐仍需 accept 杠杆 + MTP 对齐」。
反过来，「需要 accept ~2 才到 400」也不准确：accept 2（3 tok/step）在 10ms 下是 **300 tok/s**，需 step ≤7.5ms。

---

## 2. 步时预期（无 tcgen05 的重新估算）

从 `final-400-config.md` §3.1 的账本出发，**逐项撤掉本轮不成立的项**：

```
verify(m=5) 现状基线                              = 37.31
  − SH_EXP_MROWS（−8.30 名义）  实测零收益 ⇒       − 0.0     # dspark-correctness-chain.md:1251
  − tcgen05 mxf4（−6.80 名义）  【本轮未 armed】⇒  − 0.0     # 脚本无该 gate
  − VERIFY_GRAPH（submit 半）                     − 15.0     # 若 capture 成功
  − VERIFY_ROPE_MROWS                             − 0.6
  + SWALLOW 的 anchor 行（5→6 行）                + 1.6
  ────────────────────────────────────────────────────────
  verify(m=6)                                    ≈ 22~23 ms（悲观）/ 若 graph 与合并重叠少则更低
```

**说明**：−15ms 是「graph 只削 submit 半」的上限口径，与 SH_EXP/tcgen05 的行批收益**有重叠**（`final-400-config.md` §2.4-1 的头号风险 R0）。
本轮把 SH_EXP 与 tcgen05 的合并收益都拿掉之后，**重叠区反而变小，graph 的 −15 更接近可兑现**。
再加上 mrows 族的 **GATE（−2.75，应生效）+ HEAD/INDEXER/NORM/COMPRESSOR（未定）**，落点带：**verify ≈ 14~23ms**。

```
步时 = verify(m=6) + draft + commit
     ≈ 14~23 + 3.6~4.9 + 0.2
     ≈ 18~28 ms
```

**⇒ 最可能的档位是「15-20ms」或「>25ms」，而不是「≤10ms」。**
「≤10ms」需要 tcgen05（−6.8）**且** SH_EXP 兑现（−8.3）**且** graph 满额——三项里本轮只带了一项。

---

## 3. 四个检查点的现场机制（含任务描述里的一条误判）

### 3.1 m=6 形状池 capture —— 机制上没问题，但三态必须分清

- `VERIFY_ROWS = 6`（`chain_dev.rs:84`），**m=6 恰好是上限**，不是超限。
- 形状池是 **per-shape 槽**（`VERIFY_GRAPH_SLOTS = 3`，:107；`verify_slot` :5340，`verify_graph_gate` :5349）：
  m=5 与 m=6 各自独立 DRY→CAPTURE→REPLAY，**不需要重建池**（`swallow-step-400-necessity.md` §3.1）。
  首轮 `spec_primed=false` 走 legacy（m=5），之后才 swallow（m=6）⇒ **两个形状都会出现，脚本已按此解析**。
- **三态分辨**（脚本 :353-372）：① 无 `[verify_graph]` 行 = gate 没开；② `capture FAILED (m=6)` = **设计内静默降级**（会打印原因）；③ `captured verify_graph_m6` = 真接上。
- **可兑现性前置**（:5349-5363）：`pos_base >= 1`、`!eng_host && !stats_dbg && !phase_dbg`、
  `compress_branch_steady`（所有 compressor 层 `compress_len > 0`，**早期步不满足**）、`supports_dspark_snapshot`、
  `supports_memset_async`、`comm.none || ar_v5()`。脚本没设 `STATS/PHASE/ENG_HOST`，这几项应过；
  **`compress_branch_steady` 是「图 capture 晚于第一轮」的合法原因**，不是故障。

### 3.2 mrows 的 decline 日志 —— **只有两个 gate 有，其余是静默**

任务说「mrows gates 的 decline 日志（head-mrows-deadgate-fix 加的可观测性）」。
现场核对：**全局只有两条 mrows 相关的 one-shot 日志**：

| gate | 可观测性 | 落点 |
|---|---|---|
| `VERIFY_HEAD_MROWS` | ✅ `verify_head_mrows_note` | :1612，调用点 :5731 / :5778 |
| `ATTN_MROWS` | ✅ `attn_mrows_decline` | :2050 |
| `SH_EXP_MROWS` / `GATE_MROWS` / `INDEXER_MROWS` / `NORM_MROWS` / `COMPRESSOR_MROWS` | **❌ 静默 `return Ok(false)`** | 例：SH_EXP 的六条件在 `if !sh_exp_mrows() \|\| … { return Ok(false) }`（:11435+），**不打印任何东西** |

⇒ 任务里「**用新的 decline 日志分析哪些 gate declined**」**部分不可行**。
这五个 gate 是「armed-but-silent-fallback」，日志里查不到。
**判读办法**：只能靠 (a) 编译期证据（.so 里有无对应 symbol）、(b) 行为对照（m=5/m=6 的 verify_ms 是否移动）、
(c) nsys 按 kernel 名聚合（`gemm_fp8_mrows` vs 逐行 `gemm_fp8`）。
**不要在缺证据时对某个 mrows gate 下「declined」结论**（项目铁律：缺证据 exit 2）。

### 3.3 HEAD 死门已修（与任务描述的「head-mrows-deadgate-fix」对齐）

`chain_dev.rs:5511-5519` 的文档注释明确：`step_rows_inner` 的**未切分 arm 现在也用同一个 v1 kernel 折叠全词表 head**，
`None` 不再让 gate 静默 no-op。`940e895`（head v1 mrows）已在树内。
⇒ HEAD 现在**可能**兑现收益（`head` 族基线 1.12ms，上限本就小）。

### 3.4 tcgen05 的 5-gate 链 —— **本轮 0/N，且被两重独立阻塞**

真实链（`chain_dev.rs:13950+` / :10540+）：

```
tc_e4m3 = e4m3 && expert_tcgen05_e4m3() && supports_expert_tcgen05_e4m3() && !ilv
tc_mxf4 = !e4m3 && expert_tcgen05_mxf4() && supports_expert_tcgen05_mxf4() && !ilv
grouped = DSV41_EXPERT_GROUPED && tc_e4m3 && !gateup_fused && !ilv && shape && symbol
```

| 门 | 本轮状态 | 依据 |
|---|---|---|
| `DSV41_EXPERT_TCGEN05_E4M3`（严格 `starts_with('1')`）| **未设** | 脚本无 |
| `DSV41_EXPERT_TCGEN05_MXF4` / `DSV41_EXPERT_TCGEN05` | **未设** | 脚本无 |
| `DSV41_EXPERT_GROUPED` | **未设** | 脚本无（且默认 OFF） |
| `!ilv`（`DSV41_EXPERT_ILV=0`）| **未设 ⇒ ilv 默认 ON** | `gateup_ilv()` = `unwrap_or(true)`（`weights.rs:489`）；`ilv_ok` 的默认条件（moe_batch/gateup_fuse/fp4_mode=2/dim%512==0）全默认满足 |
| **`e4m3 ⊥ mxf4`** | **脚本设了 `E4M3=1`** ⇒ `tc_mxf4` 恒 false | :13950 的 `!e4m3` |

⇒ **即使补上 gate，ILV 默认 ON 也会拒绝 tcgen05**（两个 arm 都要 `!ilv`）。
这是任务描述里「tcgen05 5-gate 链」之外**另有一条独立阻塞**（`swallow-step-400-necessity.md` R1 / `dspark-correctness-chain.md:1367` 的隐藏坑同源）。

⇒ **routed experts 本轮走 `expert_gate_up_fp4_batched`（e4m3 单趟）**，即 `EXPERT_ACT_E4M3=1` 的收益（省一趟 GEMM），**不含** tcgen05 的 −6.8ms。

---

## 4. 逐档行动预案

### 档 A：`steady_median ≤ 10ms`
- **含义**：性能侧到 400 骨架（**不是** 400 吞吐——本轮 accept≈1.022 ⇒ ≈202 tok/s）。
- **动作**：
  1. 记录 `vg_shapes` 必须含 `m=6`、`dg_engaged=captured`（否则 10ms 不可信）。
  2. 立刻补 accept 杠杆做第二轮：`DSV41_TAP_INPUT=1 DSV41_DRAFT_BF16_DOMAIN=1`（预期 accept → 1.214），
     **单独一轮**，不要与性能矩阵混（`dspark-correctness-chain.md` 明确「最终 400 组合测试」才合并）。
  3. 补 tcgen05 的完整四件套（含 ILV）：`DSV41_EXPERT_ACT_E4M3=1 DSV41_EXPERT_TCGEN05_E4M3=1
     DSV41_EXPERT_GROUPED=1 DSV41_GATEUP_FUSE=0 DSV41_EXPERT_ILV=0`（`dspark-correctness-chain.md:1373`）。
     ⚠️ `ILV=0` 会改变权重布局，**必须在同一轮内成组开启**，不能与 ILV=1 的结果比绝对值。

### 档 B：`steady_median 15-20ms` ← **最可能**
- **含义**：部分优化未兑现。**优先查两项**（按杠杆大小）：
  1. **tcgen05 缺席**（−6.8ms）——这是本轮**最大**的单点缺口，且是脚本设计如此。**不是 bug**。
  2. **VERIFY_GRAPH 的 m=6 是否 capture**——看 `vg_shapes`。若只有 `m=5`：
     swallow 的 6 行块退回 direct launch，**graph 的一半收益没了**。查 §3.1 的六项前置。
- **动作**：
  - 若 `vg_shapes` 缺 `m=6`：查 `vg_failed_shapes` + 该轮日志里 `capture FAILED (m=6)` 的 reason 串；
    **不要改代码**（缺证据）。先确认 `compress_branch_steady` 与 `pos_base>=1` 在首轮 swallow 时是否成立。
  - 若 graph 正常：做 **nsys 一次**，按 kernel 名聚合 `gemm_fp8_mrows` / 逐行 `gemm_fp8` / `gemv_bf16*`，
    钉死 mrows 族真实 dispatch（这是 §3.2「静默 gate」唯一的取证手段）。
  - **下一轮**再补 tcgen05 四件套（档 A 第 3 步）。

### 档 C：`steady_median > 25ms`
- **含义**：SWALLOW 或图化没接上。**按顺序排查**：
  1. **SWALLOW 是否真的分派了**：首轮必须走 legacy（`spec_primed` 引导，:6943），
     若 `spec_primed` 因某步失败未置位，就**永远停在 legacy m=5**（每步多付 6.15ms 主链）。
     证据：`verify_ms` 是否 +1.6ms、`[dspark] steps=` 的 verify 是否接近 m=6 的值。**没有 +1.6 ⇒ swallow 没进**。
  2. **图是否 capture**：`vg_engaged ∈ {no, failed}` ⇒ 主链+verify 全走 direct launch。
     `> 25ms` 与「graph 全丢」的量级（−15ms）吻合。
  3. **commit 语义**（`dspark_commit(pos, 6, k_emit)`，:7328）：`keep` 越界会 `debug_assert` 或错位，
     表现为**位置跳跃异常**（`kacc` 直方图出现 delta>7 被丢弃 ⇒ `kacc_n` 骤小）。
- **动作**：**不要在同一轮里改代码**。先出证据（哪一项没接上），再决定是否回退 `SWALLOW_STEP=0` 做 A/B 定位。

### 档 D：出现拉丁 / 双字 / 缺句（红线破）
- **含义**：某 gate 破坏了基线（或引入了数值路径）。**红线优先于性能**。
- **二分隔离顺序**（每一步单独一轮，`LOCK` 保证串行）：
  1. **`SWALLOW_STEP=0`**（保留其余）：swallow × `SIDS_WRITEBACK` 是本轮**唯一的新语义组合**，
     且 `SIDS_WRITEBACK` 被文档标注为「会放大 verify 数值误差 ⇒ 更早 collapse」（`swallow-step-400-necessity.md` §4）。
     **第一嫌疑**。
  2. **`VERIFY_GRAPH=0`**：m=6 的 capture 若在 replay 时与 host mirror（`compress_lens`）不同步，
     会静默改状态。**第二嫌疑**。
  3. **`EXPERT_ACT_E4M3=0`**：`E4M3=1` + ILV 默认 ON 的组合在本轮**没有历史 A/B 证据**（§0/§3.4）。
  4. **`DSV41_SIDS_WRITEBACK=0`**：单独验证 swallow 的硬依赖假设。
- **注意**：`HC_VERIFY_FUSE` 已在 `46de662` 默认 OFF，脚本 `FORBIDDEN` 硬校验它不在 env 里（:498-504），
  所以**本轮不可能是 HC_VERIFY_FUSE 的锅**（除非 env 校验被绕过——脚本会 exit 2）。

---

## 5. 交付前 3 个零成本自检（跑之前/读日志时）

1. **env 实证**：脚本会把 `/proc/<pid>/environ` 落到 `<tag>.env`（:494）。
   先 `grep -E 'TCGEN05|GROUPED|EXPERT_ILV|GATEUP_FUSE' <tag>.env` ⇒ **应为空**（§0 的现场事实）。
   若非空，说明有 shell export 泄漏，**本轮配置与任务描述不同**，读数作废（脚本只挡 3 个 FORBIDDEN，不挡这 5 个）。
2. **build-id 同源**：脚本 `embeds_id` 用 `strings | grep -cF`（不是 `-q`，避开 SIGPIPE 假失败）。
   若 `--no-build` 跑，先确认 `.build_id` 与 binary 内嵌 id 一致。
3. **`kacc_n` 与 `rounds` 的比例**：1000 token、tok/step≈2 ⇒ 期望 ~450 轮。
   若 `rounds` 远小于此（如 <100），检查是否提前 EOS（`chars` 会同步变小）——**不是性能问题，是可比性问题**。

---

## 6. 一句话交给测试后的判定树

```
读 steady_median：
  ≤10ms  → 性能骨架到位；下一步补 accept 杠杆 + tcgen05（四件套含 ILV=0）
  15-20  → 【最可能】先看 vg_shapes 有没有 m=6；缺 tcgen05 是本轮设计如此，不是 bug
  >25    → 查 spec_primed（swallow 是否真分派）→ 查 vg_engaged（图是否丢）→ 查 kacc_n
  红线破 → 二分：SWALLOW=0 → VERIFY_GRAPH=0 → E4M3=0 → SIDS_WRITEBACK=0
```

**票面预期**：`verify(m=6) ≈ 14~23ms`、`步时 ≈ 18~28ms`、`mean_k ≈ 1.02`、`tok/s ≈ 70~110`、
`vg_shapes` 含 `m=5,m=6`、`latin=0`。
**若落在这一带，是账本预期内的结果，不是失败**；真正的缺口是 **tcgen05 未 armed（−6.8ms）** 与 **accept 无杠杆（1.022 vs 3）**。

---

*工部 · 只读分析 + 本文件（唯一产出）；未执行任何 GPU 命令、未改动任何源码。*
*所有 ms 均标了来源；「无 tcgen05 的步时重估」为推算项，已在 §2 显式标注。*
