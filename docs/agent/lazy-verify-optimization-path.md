# LAZY_VERIFY 的优化路径（22.56ms → 15ms）—— 400 的真正路径

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 代码基线：HEAD `3d9709b`（`crates/ferrite-models/src/dsv41/chain_dev.rs`，逐条 `file:line` 核对）。
> 输入账本：`lazy-batched-gate.md` · `verify-fusion-arch.md` · `swallow-nograph-optimization-plan.md` ·
> `swallow-unlocked-next-plan.md` · `verify-ms-breakdown.md` · `verify-eager-fusion-migration.md` ·
> `verify-operator-optimization-list.md` · `sh-pair-template-m-design.md` · `dspark-correctness-chain.md`。
> **本机无 GPU ⇒ 所有 ms 标了来源（实测 / 账本推算 / 设计口径）。**

---

## 0. 判决（先读七条 —— 两条修正任务前提）

1. **22.56ms 的分解是**：`verify 18.11 + draft 4.28 + commit 0.17`。
   `verify = k_emit × c_row`，`k_emit = 1 + mean_k = 2.214`（accept 1.214 / 出师表）⇒ **`c_row = 8.18 ms/行`**
   （`dspark-correctness-chain` lazy 段实测：4.28 + ~18 + 0.17，每行 ~8.2~8.65）。

2. **🔴 关键机制发现：lazy 的残差不是「每行更贵」，而是「per-step 族被 ×k_emit」。**
   verify 的开销分两类：
   - **per-row 族**（发数 ∝ 行数 m）：shared/routed/proj/gate/attention/indexer/head/compressor/engram；
   - **per-step 族**（发数 ∝ 层数，**与 m 无关**）：**hc 链**（10 发/层 ×40 = 400 发）与 **all-reduce v5**（80 轮/步）。
   batched 每步付 per-step 族 **一次**；lazy 每步付 **k_emit 次**。
   实测锚点：hc 族 = **2.96ms @ 53 GB/s**（全表最低带宽 ⇒ 纯 launch-bound，**bytes 随 m 缩而时间不缩**）。
   ⇒ lazy 每步的 hc ≈ `2.214 × ~2.7 = 6.0ms`（batched 只 2.96ms）；AR ≈ `2.214 × 1.40 = 3.1ms`（batched 1.40ms）。
   **两项合计 lazy 比 batched 多付 ~4.7ms/步。**

3. **⇒ lazy 的最佳杠杆不是「让每行更便宜」（那是 batched 的杠杆），而是「消灭 per-step 族的重复」。**
   靶子就是 `HC_VERIFY_FUSE` + `HC_FRONT_ROWS` + `VERIFY_AR_FOLD` 这一批**零代码 flag**
   （`verify-eager-fusion-migration §2.1/§3.1`）——而它们在 lazy 下的价值是 batched 下的 **×k_emit 倍**。

4. **🔴 修正任务前提 3（SH_PAIR）**：`shared expert ~10.4ms` 是 **m=5 口径**（25 发/层 × 5 行 = 1000 发）。
   在 **m=1** 下 shared expert 只有 5 发/层 = 200 发/步 ⇒ **≈ 2.08ms/步**。所以 SH_PAIR 在 lazy 下
   **不是 −7.9ms，而是 −1.1~1.3ms/步**：`template<M>` 的真正收益（phase 1 的 9 → 54 block 并行度修复）
   **只在 M ≥ 2 兑现**，`M = 1` 时 phase 1 仍是 9/160 个 block 在干活（`sh-pair-template-m-design §1.1/§3.3`）。
   M=1 的真实收益是「5 发 → 2 发 + 去掉 `sh_act_r` 的 global 往返 + swiglu 独立发」。

5. **per-row 能降到 EAGER 6.15ms 吗？能 —— 但只到 18.2ms。**
   `8.18 → ~6.2` 靠 §0-3 的 hc 融合 + per-row sync 收敛（详见 §3 的 L1/L2）。
   但 `2.214 × 6.2 + 4.28 + 0.17 = 18.2ms`。**要到 15ms，单行必须 < 5.5ms ⇒ 必须再加 SH_PAIR(M=1) + tcgen05。**
   任务里的「lazy + 全融合 15-19ms」：**19ms = 无 tcgen05 的诚实落点，15ms = tcgen05 兑现后的落点。**

6. **图 replay 的 overhead ≈ 0，不是差额来源。** m=1 图化实测 **−1.6ms**（24.15 → 22.56，`dspark-correctness-chain`）——
   replay 比裸链**更快**，它的 submit 已被 CUDA async launch 隐藏（`swallow-nograph-optimization-plan §2`）。
   每行的图成本只有一次性 DRY + CAPTURE（已摊薄）。

7. **mrows 族在 lazy 下恒为 0，不要投。** `GATE_MROWS` / `INDEXER_MROWS` / `VERIFY_HEAD_MROWS` /
   `VERIFY_ROPE_MROWS` / `ATTN_MROWS` 全是「把 m 行折成 1 发」——m=1 时 rows=1，**mrows ≡ 逐行**
   （`swallow-unlocked-next-plan §3.1`，commit `33af60c` 已显式更正）。
   `SH_PAIR template<M≥2>` 的 −4.9~7.9ms 同理属于 **batched**，不属于 lazy。

---

## 1. 22.56ms 的分解（口径钉死）

### 1.1 主表

| 段 | ms | 来源 | 说明 |
|---|---:|---|---|
| **verify** | **18.11** | 实测（`[dspark] verify=`） | `= k_emit × c_row`；lazy 逐行 `step_rows(m=1)` |
| **draft** | **4.28** | 实测 | 5 个 MTP 块 × 3 层；P3a/markov 未开 |
| **commit** | **0.17** | 实测 | `dspark_commit_lazy` 只做 `set_pos_ctr` + `inv_compress_len`（**无回滚、无 replay**，`chain_dev.rs:8037`） |
| **合计** | **22.56** | 实测 | 92 tok/s @ accept 1.08~1.214 |

`k_emit`（swallow 布局，`chain_dev.rs:2261`「lazy IMPLIES swallow」）：
```
k_emit = matched + 1 = 1 + mean_k          # 1..=6
rows_run = k_emit                           # `debug_assert_eq!(rows.len(), k_emit)` :8277
c_row  = 18.11 / 2.214 = 8.18 ms/行
```

### 1.2 `c_row = 8.18` vs EAGER `6.15` —— 残差 +2.03ms/行 的归因

| 来源 | 量级 | 证据 |
|---|---:|---|
| **hc 链未融合**（per-step 族，EAGER 融合后 ~4 发/层，verify 走 raw chain 10 发/层 ⇒ +6 发/层 = 240 发/块） | **+0.8 ~ +1.5ms/行** | `verify-eager-fusion-migration §2.1`（400 发/步，2.96ms），`HC_VERIFY_FUSE`/`HC_FRONT_ROWS` **默认 OFF**（`chain_dev.rs:12239/12293`） |
| **per-row host round-trip**：`set_pos_ctr` blocking H2D + `ids_r`/`pos_rows` H2D + **argmax D2H**（一次 stream drain）+ `lazy_tap_commit` 3×D2D | **+0.5 ~ +0.6ms/行** | `dspark-correctness-chain`「剩余 ~2.85ms/行」的第 2/3 项（barrier 已被 `b018101f` 的 barrier-batch A/B 排除：22.59 ≈ 22.56） |
| 其余（head 逐行、norm/quant 逐行、engram 逐行的固定项） | +0.1 ~ +0.3ms/行 | `verify-ms-breakdown §1` |
| **图 replay 本身** | **≈ 0（净负）** | m=1 图化 −1.6ms（24.15→22.56） |
| **host_barrier** | **≈ 0** | barrier-batch A/B 无变化（`b018101f`） |
| **合计** | **≈ +2.0ms/行** | 与实测残差一致（区间 1.4~2.4） |

### 1.3 一个已实现但鲜被引用的数据点（基线口径修正）

`dspark-correctness-chain` 记录了 **mrows staging 修复**后的 lazy：
**22.56 → 18.82ms（−3.7ms）**，`verify` 33.96→（batched 段）与 lazy 无截断 18.82ms / 91.3 tok/s。
⇒ 本条任务的 22.56ms 基线**尚未含该修复**；若该修复在 HEAD，`c_row` 应已落在 **6.5~7.1ms**。
**建议 A/B 前先确认 HEAD 的 lazy 实测值**（否则 §3 的所有增量会与基线重复计数）。

---

## 2. per-step 族的 ×k_emit —— lazy 的隐藏税（本文件的核心）

### 2.1 两个族的发数与 m 无关

| 族 | 发数（m=5） | 发数（m=1，懒） | 实测 ms（m=5） | 性质 |
|---|---:|---:|---:|---|
| **hc 链** | **400**（10 发/层 × 40） | **400/行 → 886/步** | **2.96** | 53 GB/s（全表最低）⇒ **纯 launch-bound** |
| **all-reduce v5** | **240**（80 轮/步，2 轮/层） | **80/行 → 177/步** | **1.40** | 协议地板（17.3µs/轮） |

代码依据：`hc_mixes(..., m as i32, ...)`（`chain_dev.rs:8978`）把**行数当作 kernel 内的 rows 维**，
launch 数只随层数增长；`hc_collapse` / `norm_rows` / `hc_post` / `memcpy_d2d` 同理（`verify-eager-fusion-migration §2.1`）。
AR：`lazy-batched-gate §4.2` 明写「**lazy：逐行 ⇒ 每步 `k_emit × 80` 轮**」。

### 2.2 结论

```
lazy 每步的 per-step 族成本 = k_emit × (hc_row + ar_row)
                           = 2.214 × (2.7 + 1.40) ≈ 9.1 ms     ← 占 verify 18.11 的 50%
batched 每步同一族         = 1 × (2.96 + 1.40)     ≈ 4.36 ms
⇒ lazy 的 per-step 族溢付 ≈ 4.7 ms/步（batched 的 2.1×）
```

**⇒ 这两族是 lazy 的第一优化目标，而不是 per-row 族。**
而且它们是 **flag / 小代码**即可动的（hc 融合零代码；AR 折核依赖 hc；AR 轮数本身不可减，见 §3-L1 注）。

---

## 3. 优化路径（22.56 → 15ms）

> 每项：**改动 + 预期节省 + 实施成本 + 依据**。`ms` 未标 (实测) 的均为账本/设计口径。

### L0 — 基线校准（必做，1 GPU 会话）

**改动**：无。同会话背靠背测 `DSV41_LAZY_VERIFY=1`（关所有新 gate）的 `[dspark] verify= / draft= / commit=`。
**为什么**：① 确认 HEAD 是否已含 mrows staging 修复（§1.3）；② 确认 `k_emit` 分布；
③ 确定 `c_row` 的真值（6.2 还是 8.2），**否则 §3 的所有增量都会与基线重复计数**。
**成本**：0 代码，1 GPU 会话。

### L1 — hc 融合（**零代码 flag，lazy 下 ×k_emit 倍**）★最大单项

**改动**（全在 env，代码已就位）：
```bash
DSV41_HC_VERIFY_FUSE=1     # A1-a hc_collapse_norm(rows=m) + A1-b hc_post_inplace_rows(rows=m)
DSV41_HC_FRONT_ROWS=1      # A2 hc_mixes_auto(rows=m) = hc_front_split
DSV41_VERIFY_AR_FOLD=1     # AR + hc_post 单发（依赖 A1）
```
前置条件**已满足**：`truncate=false` 修复已落地（`collapse_norm_rows` / `hc_post_rows` 硬编码 `false`，
`chain_dev.rs:8967` + `verify-eager-fusion-migration §2.1`）——这正是 `HC_VERIFY_FUSE` 被默认关掉的历史根因。

| 量 | 值 |
|---|---|
| 现状 | 10 发/层 ⇒ **400 发/块**，2.96ms/块，53 GB/s |
| 迁移后 | 4~6 发/层 ⇒ 160~240 发/块（launch 账 −160~−240 发） |
| **节省（batched，m=5/6）** | −1.3 ~ −1.9ms/块（`verify-eager-fusion-migration §4.2` 的 A1+A2+AR fold） |
| **节省（lazy，m=1）** | **×k_emit = −2.9 ~ −4.2ms/步** |
| **成本** | **0 代码 + 0.5 人日 A/B** |
| 判据 | 四段文本逐字 + `faults=0` + 零拉丁（`BF16_TRUNCATE` 不得再漏进 verify） |

> **L1 注（AR 的硬地板）**：`VERIFY_AR_FOLD` 只把 hc_post 折进 AR 的 store 尾（2 发 → 1 发），
> **不减 AR 轮数**。每层 2 轮（attention AR + MoE AR）在 lazy 下是 `2.214 × 1.40 = 3.1ms/步` 的**硬成本**
> ——它是 lazy 相对 batched 的结构性代价，无机械解（除非减行数 = accept 改善）。

### L2 — per-row sync 收敛（新代码，小）★lazy 专属

**改动**（`dspark_spec_lazy` 的循环 + `lazy_run_row`，~50 行）：
1. **两行一批（SDR, speculative double row）**：`row i` 与 `row i+1` **背靠背 launch**（无中间 D2H），
   一次 D2H 回读两个 argmax ⇒ **sync 次数 ÷2**。若 `k_emit` 为奇数，多跑的那一行（= 首个被拒 draft 所在行）
   用**既有的** `dspark_rollback_keep(pos, 2, 1, &host_mirrors)` 回滚（快照每轮已无条件拍，:8196）。
2. **`set_pos_ctr` + `ids_r`/`pos_rows` 合一次 H2D**：现在每行 3 次 blocking H2D（`chain_dev.rs:8104/5304/5305`），
   合成一条 12 B 的 `upload_bytes_at`（`pos_lo, id, pos_hi`），kernel 侧读同一结构。
3. **`lazy_tap_commit`**：`DSPARK_TAP_SLOTS=3` 次 D2D（:8064）合成一次（三 slot 在 staging 里本就连续）⇒ 3 发 → 1 发。

| 项 | 现状 | 优化后 | 节省 |
|---|---:|---:|---:|
| argmax D2H（~0.5ms/行） | 2.214 次 | ~1.2 次 | −0.5ms/步 |
| `set_pos_ctr`+`ids`+`pos_rows` H2D | 3×2.214 次 | 2.214 次 | −0.15ms/步 |
| `lazy_tap_commit` | 3×2.214 发 | 2.214 发 | −0.05ms/步 |
| **合计** | | | **−0.7ms/步** |

**成本**：1~2 人日；**风险**：SDR 破坏 `rows_run == k_emit` 不变量（`debug_assert_eq!` :8277）——
必须重新论证「多跑的行」在 keep 范围外、且回滚路径覆盖 compression/ring/tap 三处；
判据 = `dspark_parity` verify 行级对照（`verify_bad == 0`）+ `DSV41_INV_CHECK=1` 全绿。

### L3 — SH_PAIR M=1（代码已就位，parity 已修）★修正口径

**改动**：`DSV41_SH_PAIR_M=1`（first-try arm，`shared_expert_mrows` :11761）或旧的
`DSV41_SH_EXP_FUSED=1` + `DSV41_SH_PAIR=1`（:11824）。
**parity 状态**：kernel 缺陷（phase-1 tail-group phantom row 越界）+ 测试缺陷已在 `3d9709b` 双双修复。

| 量 | 现状（m=1） | 迁移后 |
|---|---|---|
| 发数/层 | 5（quant1 → mx2(w1\|w3) → swiglu → quant1 → mx_add(w2)） | **2**（quant_rows + `sh_exp_fused<1>`[+epi_add]） |
| ms/步 | ~2.08（= 10.40/5 的 m=1 折算） | ~1.4 ~ 1.6 |
| **节省** | | **−1.1 ~ −1.3ms/步** |
| **成本** | | **0 代码 + 0.5 人日 A/B** |

> **不要把它当 −7.9ms**：`template<M>` 的 phase-1 并行度修复（9 → 54 block，`sh-pair-template-m-design §3.3`）
> **在 M=1 下不成立**；M=1 与 M≥2 是两个不同的收益源。
> 另：`verify-operator-optimization-list §5.2` 已预警「M=1 版可能不比 mrows 快」——
> 若 A/B 中性，**不要下「融合无效」的结论**，M=1 的收益本就只有 launch 与 staging 往返。

### L4 — tcgen05 routed gate/up（换核，唯一的「越 5% 峰值」项）

**改动**：5-gate 链 `EXPERT_ACT_E4M3=1 EXPERT_TCGEN05_E4M3=1 EXPERT_GROUPED=1 GATEUP_FUSE=0 EXPERT_ILV=0`
+ **GPU e4m3 parity + 对齐守卫验证**（`verify-operator-optimization-list §2-#7`：骨架已编入 .so，**从未上机**）。
**预期**：routed 1.66 → ~0.7ms/行 ⇒ **−1.9 ~ −2.1ms/步**（修正口径 −1.0~3.8ms，**只 gate/up**；down 无 tcgen05 核）。
**成本**：**4~5 人日 + GPU parity 会话**（本轮最贵的一项）。
**阻塞**：`chain_dev.rs:687` 的 e4m3 × tcgen05 互斥未澄清；`slots=8` 与 verify `topk=6` 冲突需 pad/K-split。

### L5 — draft（flag 已就位）

**改动**：`DSV41_DRAFT_P3A=1` + `DSV41_MARKOV_SLICED=1`（+ P3c 图化）。
**预期**：4.28 → ~3.5 ⇒ **−0.7 ~ −0.8ms/步**（`final-400-config §3.2`：−0.30/−1.00ms + P3c）。
**成本**：0 代码 + 0.5 人日 A/B。

### 3.1 步时阶梯

| 阶段 | 增量 | 累计 | 依据强度 |
|---|---:|---:|---|
| **L0** lazy 基线（accept 1.214） | — | **22.56** | 实测 |
| **L1** + hc 融合（A1+A2+AR fold，×k_emit） | −2.9~4.2 | **18.4 ~ 19.7** | 账本（launch 账 −160~240 发/块 × 2.214） |
| **L2** + per-row sync 收敛（SDR） | −0.7 | **17.7 ~ 19.0** | 设计口径（D2H ~0.5ms/行，`b018101f` 排除 barrier） |
| **L3** + SH_PAIR M=1 | −1.1~1.3 | **16.4 ~ 17.9** | 代码就位 + parity 已修；账本口径 |
| **L5** + draft 折叠 | −0.7~0.8 | **15.7 ~ 17.2** | 账本（`final-400-config §3.2`） |
| **L4** + tcgen05（gate/up） | −1.9~2.1 | **13.6 ~ 15.3** | ⚠️ 从未上机 |

```
无 tcgen05 的诚实落点：  16 ~ 17ms   （任务说的「19ms」偏保守）
tcgen05 兑现后的落点：   14 ~ 15ms   ✓ 命中 15ms 目标
```

**⇒ 15ms 需要 L1+L2+L3+L5+L4 全部兑现；其中 L1/L2/L3/L5 是零到低代码，L4 是唯一的 4~5 人日项。**

### 3.2 明确不做（在 lazy 下恒为 0）

| 项 | 为什么 |
|---|---|
| `GATE_MROWS` / `INDEXER_MROWS` / `VERIFY_HEAD_MROWS` / `VERIFY_ROPE_MROWS` / `ATTN_MROWS` | m=1 ⇒ rows=1 ⇒ mrows ≡ 逐行（`swallow-unlocked-next-plan §3.1`） |
| `SH_PAIR template<M≥2>` 的 −4.9~7.9ms | phase-1 并行度收益只在 M≥2 兑现；属 batched 的账 |
| `VERIFY_GRAPH` 转默认 ON 作为「省 15ms」 | 图只值 −1.5ms（`swallow-nograph §2`），且 ar5-hang 的臂分歧根因（Plan B 已实施，见 §4.3） |
| 「加回 batched」（SWALLOW_STEP=1） | 高 accept 下 ar5-hang 未解（`92bceb69`：计数任务 937 hang）；且 batched 结构上更慢（恒 6 行 vs 2.214 行） |

---

## 4. 与 batched 的对比 —— 什么时候 batched 更好

### 4.1 选路判据（修正 `lazy-batched-gate §0.7` 的 c）

```
lazy iff (k_emit) × c_row < B           # c_row = 本文件的 8.18（实测），不是 6.15（EAGER）
⇔ mean_k < τ,  τ = B / c_row − 1
```
| 情形 | B | c_row | τ | 谁赢 |
|---|---:|---:|---:|---|
| 现状（batched 未融合） | ~37 | **8.18** | **3.5** | mean_k < 3.5 ⇒ lazy（对话 0.96 / 出师表 1.21 ✓）；计数 5.0 ⇒ batched |
| 现状（§0.7 旧口径） | 37 | 6.15 | 5.0 | 过于乐观（把 lazy 的 per-step 溢付算成了 0） |
| L1+L2 后 | ~37 | ~6.2 | ~5.0 | lazy 全胜（含计数） |
| batched 的 mrows 真兑现 | 8.2 | 6.2 | **0.32** | 几乎永远 batched |

**⇒ 两个反转条件都是未兑现的**：(a) batched 的 mrows 族历史兑现率 ≈ 0（`SH_EXP_MROWS` 两次零收益、
`{SH_EXP+GRAPH+ROPE+P3A}` 全开只 −1.21ms，`verify-ms-breakdown §修正`）；
(b) accept ≥ 3.5 只在计数任务上出现（任务依赖，`accept-1214-to-2-3-path`）。
**⇒ 在 accept 1.2 的出师表/对话任务上，lazy 是唯一可靠且更快的臂。**

### 4.2 结构对比（tiny-SIMT-kernel 架构下的必然）

| 维度 | lazy | batched（SWALLOW） |
|---|---|---|
| 每步 forward 行数 | **2.214**（k_emit） | 恒 **6** |
| per-step 族（hc/AR） | `k_emit ×` ⇒ **lazy 亏 4.7ms** | 1× ⇒ batched 赢 |
| per-row 族（shared/routed/…） | 2.214× | 6× ⇒ batched 亏 |
| 权重读 | k_emit 次（但 tiny-SIMT 下权重共享不省钱） | 1 次（**对 instruction-bound 核无效**，`arch-floor §5.2`） |
| 回滚 / replay | **无** | 有（snapshot→rollback→replay 三段） |
| 可靠性 | ✅ 无 ar5-hang（m=1 单形状） | ❌ 高 accept 下 ar5-hang |

**净账（accept 1.214）**：lazy 22.56 vs batched ~37.5（+图）⇒ **lazy 胜 15ms**。
其中 contrition 的分解：per-step 族 lazy 亏 4.7ms，per-row 族 batched 亏 ~(6−2.214)/2.214 × ... ⇒ 净差 15ms 主要由「行数 6 vs 2.214」贡献。

### 4.3 一个可选的混合臂（记录，不建议先做）

`前 2 行 lazy + 尾部一个 m=4 块`：两个前导行 lazy（早退覆盖 ~65% 的步），若两行全中再发一个 `m=4` 块
（per-step 族只付一次）。**理论收益**：把 per-step 族在「k_emit ≥ 3」的步上从 ×4 降到 ×1，约 −0.8~1.2ms/步。
**阻塞**：① 形状池只有 2 槽（`VERIFY_GRAPH_SLOTS=2`，:107），`m=1`/`m=6` 已占满，第三形状 `m=4` 会静默落 direct；
② 两臂的 AR 足迹在同一步内不同 ⇒ 需要 §4.3 的 rank 同步换臂；③ 正确性论证（混合块的 keep 语义）比 SDR 更重。
**排期**：L1~L5 之后再评估。

---

## 5. 验证计划（最少 GPU 次数，单一测试驱动）

| # | 会话 | 内容 | 判据 |
|---|---|---|---|
| **V0** | GPU（1） | L0 基线：`LAZY_VERIFY=1` + 全 gate OFF，测 `verify=/draft=/commit=` + `k_emit` 分布 | `c_row` 真值（6.2 还是 8.2，§1.3） |
| **V1** | GPU（1） | L1 A/B：`HC_VERIFY_FUSE=0/1` → `HC_FRONT_ROWS=0/1` → `VERIFY_AR_FOLD`（**一次只动一个**） | 四段文本逐字 + 零拉丁 + `faults=0`；`verify=` 应 −2.9~4.2ms |
| **V2** | GPU | L3 A/B：`SH_PAIR_M=0/1` | 同上；若中性，**记录为「M=1 无 phase-1 收益」而非「融合无效」** |
| **V3** | GPU | L5 A/B：`DRAFT_P3A` + `MARKOV_SLICED` | `draft=` 应 −0.7~0.8ms |
| **V4** | GPU | L2：`dspark_parity` 行级对照 + `DSV41_INV_CHECK=1` | `verify_bad == 0`；`rows_run == k_emit` 不变量在 SDR 下仍成立 |
| **V5** | GPU | L4：tcgen05 单层微基准门（`scripts/dsv41_tcgen05_mxf4_verify.sh --step 3`） | gateup 22.2µs / down 17.2µs；**不达标即止损**，不进集成 |

**铁律**（`dspark-correctness-chain` 会话教训）：同一远端同时只有一个测试驱动；subagent 只做代码/分析；
不轮询远端；`cargo check --workspace` 是本地硬门禁。

---

## 6. 诚实校准（必须写在账上的四件事）

1. **L1 的 ×k_emit 是「结构推论 + 实测锚点」，不是实测 A/B。**
   hc 族 2.96ms/400 发 @ 53GB/s 是**实测**；「k=1 时同 400 发、时间不缩」是**推论**
   （launch-bound，bytes 只占 1.1% 峰值）。**V1 是唯一能把它变成实测的会话。**
2. **§4.2 的「per-step 族亏 4.7ms」与 §1.2 的「残差 2.03ms/行 = 4.5ms/步」部分重叠。**
   两者是同一现象的两个切法（前者按族、后者按 EAGER 差），**不得相加**。
3. **L4（tcgen05）从未上机**，且与本轮正确性基线（e4m3 单趟）的兼容性未澄清（`chain_dev.rs:687`）。
   §3.1 的 15ms 是**条件落点**，无 L4 的诚实落点是 **16~17ms**。
4. **R6 历史教训**：本项目的 #1 测量陷阱是「gate 设了但没生效」（mrows 5 个 gate 静默 `return Ok(false)`、
   `SH_EXP_FUSED` 需要 `.so` 符号、`HC_VERIFY_FUSE` 反向默认）。
   ⇒ 每个 gate 翻转必须**读回确认 + 单独 commit**（`final-400-config §6-R9`），
   并优先看 `[verify_graph]` / launch 计数 / nsys kernel 名，而不是只看 `verify=`。

---

## 附：一句话总结

**lazy 的 22.56ms 里，一半（~9.1ms）是「per-step 族」（hc 链 + AR v5）被 ×k_emit 的重付——
所以 lazy 的第一杠杆是把 hc 链按 EAGER 的融合搬到 verify（零代码 flag，−2.9~4.2ms/步），
第二杠杆是把每行的 host round-trip 合并（−0.7ms），第三是 SH_PAIR(M=1)（−1.1~1.3ms）；
把每行压到 EAGER 的 6.15ms 只到 18.2ms，要落到 15ms 必须再加上 tcgen05 的 routed 换核。**

---

*工部 · 只读分析 + 本文件（唯一产出），未执行 GPU 命令、未改动任何源码。*
*所有 ms 标来源（实测 / 账本推算 / 设计口径）；§3.1 的阶梯每格都标了依据强度，
未实测项集中在 L2/L4 与 §6。*
