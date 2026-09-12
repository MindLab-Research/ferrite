# S1 修复验证测试 — 结果预分析

> 工部 · 2026-09-12 · **只读分析：未改动任何源码、未执行 GPU 命令；本文件为唯一产出**
> 输入：`accept-first-strategy.md`（S1 提出者）、`accept-gap-1214-to-3.md`（S1 的反对判词）、
> `dspark-sglang-real-data.md`、`dspark-correctness-chain.md`、`chain_dev.rs` / `dspark_dev.rs` 代码。
> 所有代码事实给出 `文件:行号`；所有推断显式标注置信度。

---

## 0. 先钉口径：这次测试到底在测什么

### 0.1 代码状态核对（✅ 已确认，HEAD = `aba9618`）

| 站点 | 行号 | 现状 |
|---|---|---|
| `rope_pos` 定义 | `dspark_dev.rs:1787` | `if seed_pos_fix() { pos + 1 } else { pos }` |
| q | `dspark_dev.rs:1833` | `rope_queries(q, rope_pos)` |
| kv | `dspark_dev.rs:1871` + `:1877` | `kv_pos = if seed_pos_fix() { pos+1 } else { pos }` → `rope_at(..., kv_pos, ...)` |
| o | `dspark_dev.rs:1973` | `rope_queries_inv(o, rope_pos)` |

**默认臂逐位不变**：gate OFF 时 `rope_pos == pos == kv_pos`，与历史调用完全相同（`git show HEAD` 无其他差异）。
**SEED_POS 臂**：seed@`pos`、q/o/kv@`pos+1+r` —— **S1 的内部自洽目标已达成**。

### 0.2 本测试 = 首个「干净单变量 A/B」（这是最大价值）

| 历史组合 | accept | 可归因？ |
|---|---|---|
| 基线（无杠杆） | 1.022 | 基准 |
| P0-3+P1-5（= `TAP_INPUT`+`DRAFT_BF16_DOMAIN`） | **1.214** | 当前最佳 |
| +SEED_POS+DRAFT_ATTN_BF16+TAP_BF16 | 0.898 | ❌ **3 gate 捆绑 + 修前 SEED_POS，不可归因** |

本次配置 = **P0-3+P1-5（1.214 基线）+ SEED_POS 单独** ⇒ 相对 1.214 的 delta **首次**可归因给 SEED_POS+S1。

### 0.3 ⚠️ 但测试计划缺一件关键东西：**同会话 baseline**

计划里只有 `SEED_POS=1` 一条臂。`1.214` 这个参照值来自**另一个会话/臂**：

- `dspark-sglang-real-data.md §2.2` 把 1.214 记作 **lazy verify 臂、无截断**；
- `dspark-correctness-chain.md §1334` 把它记作 **P0-3+P1-5**（是否含 `BF16_TRUNCATE` 未逐字写明）；
- `accept-gap §1` 的 **U1**：`1.022 / 1.214 / 0.898` 分别属于哪个 arm **从未记录**，legacy / lazy / aligned 的 `k_acc` 物理含义相差一格（`chain_dev.rs:1711-1718` 三行表）。

⇒ **没有一个同会话的 `SEED_POS=0` 对照臂，1.4 和 0.9 都无法判读**（可能与 arm / 版本漂移混淆，违反 AGENTS.md「跨版本比较同会话背靠背」）。
**建议：本次测试至少跑两条臂（`SEED_POS=0` 与 `SEED_POS=1`，其余 flag 完全相同），或直接跑 `0/1/0` 三连。**

---

## 1. 两派先验：S1 到底会不会有效（**文档本身分歧**）

| 来源 | 判词 | 依据 | 预期 accept |
|---|---|---|---|
| `accept-first-strategy.md §1.3` | S1 补齐后 = **官方臂**，一次 A/B 检验"是数值 bug" | `model.py:1055→1059/1061/1068` 同一 `freqs_cis` | **1.4 ~ 1.8** |
| `accept-gap-1214-to-3.md §2.4` | 补 S1 后**仍有 2 处错位**，仍是"半成品臂" | row-0 token + accept 链未修 | **仍退化**（≤1.214） |

两派都能自洽解释历史数据（079ffbaf 单开退化、10ba0e73 组合 0.898）。**分歧点必须靠本次 A/B 裁决，不能靠再推理。**
我的主观先验（见 §3）：**"不变 ~1.2" 是众数（40-50%），"提升到 1.4-1.8" 只有 25-35%**。

---

## 2. 🔴 关键代码风险（本次新增发现）：LAZY 臂与 SEED_POS 的记法可能不匹配

这一条会直接决定第 3、4 分支的读判，**测试前必须知道**。

### 2.1 代码事实

1. **`DSV41_LAZY_VERIFY` 隐含 swallow 布局**（`chain_dev.rs:2184-2190`：「lazy IMPLIES swallow」，块布局是 `[anchor, d1..d5]`）。
2. **LAZY 臂调用 `draft_forward(token, pos)`**（`chain_dev.rs:7834`）—— **与 legacy 同一调用约定，不是 aligned 的 `(next, pos+1)`**。
3. **LAZY / SWALLOW 臂都携带 tap**：`carry_kept_tap(...)` 在 swallow 是**无条件**的（`chain_dev.rs:7537`）、在 lazy 也是**无条件**的（`chain_dev.rs:7929`）。即 draft 拿到的 tap 是**上一轮携带的**（内容位于 `pos-1`），不是本轮 `step_dev` 的（内容位于 `pos`）。
4. `carry_kept_tap` 的注释确认这条规则：`keep - 1` 行「`pos_base + keep - 1 = pos_ctr_new - 1`，是那个位置 token 的精确 hidden」（`chain_dev.rs:7985-7998`）。

### 2.2 由此得到的两条对撞结论

- **accept-gap §2.2 臂表**：`SWALLOW`（继承 tap）默认 = `seed@pos-1` **且** 内容 `T'@pos-1` ⇒ **自洽、=官方 ✓**。
- **SEED_POS 的立论前提**（`dspark_dev.rs:1768-1779` 的注释）是 **legacy 约定**：tap 是本步的，内容在 `pos`，所以 seed 应在 `pos`。

⇒ **在 LAZY 臂下打开 SEED_POS，等于把已经自洽的 `(相位 pos-1, 内容 pos-1)` 推成 `(相位 pos, 内容 pos-1)`——内容与相位再次错配（这次是内容落后一格）**，与 legacy 下的错配方向相反。

### 2.3 置信度与处置

- **置信度：中**。代码指针是硬的（调用约定 + 无条件 carry），但"tap 内容的绝对位置"最终要 GPU 上的 `unit_dump` / `DIFF_EAGER` 才能钉死；注释本身在 `pos+1+r` 与 `pos+r` 之间有历史积压（见 `chain_dev.rs:6756-6776` 与 `:7482-7486` 的相邻矛盾描述）。
- **处置（不阻塞测试，但影响判读）**：
  1. **必须先跑 `SEED_POS=0` 的 LAZY baseline**（§0.3）。若它 ≠ 1.214，则 U1 成立，一切跨臂比较重算。
  2. 若 `SEED_POS=0` 的 LAZY baseline ≈ 1.214 而 `SEED_POS=1` **下降** → **不要**直接判"相位不是主因"：先排除"LAZY 臂与 SEED_POS 记法错配"这个混杂（§2.2）。
  3. 干净裁决应同时覆盖 **`SEED_ALIGN`（`chain_dev.rs:7269`，`draft_forward(next, pos+1)`）** 与 **`SWALLOW`**——accept-gap 认为这两个才是"=官方"的臂，且**两者的 accept 从未测过**。

---

## 3. 四分支预案 + 预期范围

> 预期范围为**主观先验**（因 §1 的两派分歧，无法给客观点估计）；每个分支给出「判读 → 先确认 → 下一步」。

### 分支 1：accept **1.4 ~ 1.8**（概率 ~25-35%）

- **判读**：S1 成功，P0-1 完整修复有效；相位是主因之一。`p: 0.56 → ~0.62-0.66`。
- **先确认（防假阳）**：同会话 `SEED_POS=0` 必须 ≈ 1.214（否则是 arm/版本漂移，不是 S1 效果）；文本零拉丁 + 出师表 LEN 一致。
- **下一步**：
  1. **S2 单变量**：在同一修好基线上分别单开 `TAP_BF16` / `DRAFT_ATTN_BF16`（各 1 轮，`accept-gap §5-S2`）。预期 ±0.1-0.2 each。
  2. **性能组合**：把 accept 杠杆带进 batched-400 v2 口径（`SWALLOW_STEP + mrows + tcgen05`），一次拿 accept + step 两个数。
  3. **若 p 到 0.66 仍 < 0.83**：转 S4（成对域对齐：draft KV `act_quant` ↔ backbone ring 同步；或 attn 内部 bf16 ↔ verify 同量，`accept-gap §2.5`）。

### 分支 2：accept **~1.2（不变）**（概率 ~40-50%，众数）

- **判读**：S1 对 accept **无可测效果**。两种可能：(a) q/o 基址不是主要因素；(b) §2.2 的 arm 记法错配把 S1 的收益抵消/淹没。
- **先确认**：`SEED_POS=0` baseline 是否确为 1.214；`k_acc` 直方图（首 token 拒绝率）是否与 1.214 那次同形（`accept-gap §1` U2）。
- **下一步**（按 ROI）：
  1. **改测从未测过的官方臂**：`DSV41_SEED_ALIGN=1`（S1a）与 `DSV41_SWALLOW_STEP=1`（S1b）的 accept A/B——`accept-gap §5` 判定这是真正的裁决实验，且 SWALLOW 是"一举两得"（同时 −4.55ms 步时）。
  2. **`DIFF_EAGER=1`** 在最佳组合上跑：mismatch 位置 = 分叉点（`dspark-correctness-chain.md §1479-1489`）。
  3. 若两臂都 ≤1.214 ⇒ 根因 A 降级，预算转 S4/S5。
- **不要把结论写成"相位不是主因"**——除非先排除了 §2.2 的 arm 混杂。

### 分支 3：accept **< 1.0（退化）**（概率 ~20-30%）

- **判读**：S1 引入新问题，或 SEED_POS 臂本身仍不完整（row-0 token / accept 链 / arm 记法）。
- **先确认（关键分叉）**：
  1. **`SEED_POS=0` 默认臂是否仍逐位正常**（accept ≈1.214、零拉丁）。若默认臂也退化 ⇒ **S1 代码有 bug**（触及了非 gate 路径），立刻回查 `aba9618` 的 diff，而非继续调 gate。
  2. 若默认臂正常、仅 SEED_POS 臂退化 ⇒ 问题在 SEED_POS 臂内部（§2.2 arm 记法 / `win_rows` `min(win,pos+1)` 配套 / row-0 token）。
- **下一步**：
  1. 隔离：`SEED_POS` × {`LAZY`, `batched`, `SEED_ALIGN`, `SWALLOW`} 的小矩阵，定位与哪个布局组合退化。
  2. 按 accept-gap §2.4 核对另外两处错位：row-0 token（`dspark_dev.rs:1153`，`ids[0]=t0`）与 accept 链（legacy vs swallow 的 index 差一格，`chain_dev.rs:7194-7214`）。
  3. 回退策略：SEED_POS 是 env-gate 默认 OFF，**不需要 git revert**（AGENTS.md 硬性禁令），保持 OFF、代码保留即可。

### 分支 4：**拉丁出现**（概率 <10%，前提是 `BF16_TRUNCATE=1` 生效）

- **判读**：SEED_POS 臂破坏了基线（window/win_rows 配套或 rope 基址仍有问题），不是"零拉丁"回归。
- **先确认**：`BF16_TRUNCATE` 是否真的生效（历史根因是 `HC_VERIFY_FUSE` 默认 ON 把截断带进 verify，`dspark-correctness-chain.md §1315`）；`SEED_POS=0` 是否零拉丁。
- **下一步**：
  1. 若 `SEED_POS=0` 零拉丁、`SEED_POS=1` 出拉丁 ⇒ 直接归因 SEED_POS 臂；隔离 `win_rows`（`dspark_dev.rs:3480-3485`，`n=min(win,pos+1)`、`s0=(pos+1-n)%win`）与 rope 基址，一次只动一个。
  2. 079ffbaf 的历史拉丁就是"seed 动了、window 没配套"——若复现，说明 `win_rows`/`ensure_idxs` 的 SEED_POS 分支仍与 `draft_attention` 的 copy 不一致。
  3. 与分支 3 的隔离矩阵合并跑。

---

## 4. 与 sglang 硬锚点的关系（给结果定坐标）

| 量 | sglang（DSV4，block 5） | ferrite | 比 |
|---|---|---|---|
| per-token p | **≈0.93** | **≈0.56** | 1.66× |
| mean-k | ~4.0 | 1.214 | 3.3×（被截断放大） |
| 步时（verify 硬锚点） | **7.3ms（实测）** | 22.56ms | 3.1× |
| τ（tok/step） | ~5.0 | 2.214 | 2.26× |

- **S1 只修了一个相位站点**。即使完全正确，单点修复的 p 增量有限：**乐观 p≈0.66 ⇒ mean-k≈1.9**，离 400 需要的 p≥0.83 仍远。
- **若 p 停在 <0.6**：相位不是主因，差距在别处——draft 的 MoE 路径（`EXPERT_ACT_E4M3` 已同源）、attention 数值域（§2.5 的成对项）、或结构层（draft 与 verify 不共用 kernel，`accept-gap §2.6`）。
- **400 的乘积约束不因 accept 改善而消失**：τ=2.214 时，**即使步时压到 sglang 的 verify 实测 7.3ms 也只有 303 tok/s**（`dspark-sglang-real-data.md §4.2`）。accept 是**门槛**，步时是**兑现**。

---

## 5. 无论哪个分支都必须做的三件事

1. **同会话 paired baseline**：`SEED_POS=0` 与 `=1` 背靠背（同 binary、同文本、同 flag），否则 U1（arm 身份）会把判读带偏。
2. **口径三件套**：arm 名 + **完整 `k_acc` 直方图** + `mean-k`/`tok/step`（`accept-gap §1` 明确要求；`dspark_verify.rs:248` 的 `[dspark] steps=… mean-k=…` 行每隔 50 步打印，取最后一行）。
3. **两侧都看文本**：零拉丁 + 出师表逐字 + LEN（`AGENTS.md` 正确性红线：不能重复、不能乱码）。

---

## 6. 一句话交付

> **本次测试是 SEED_POS 的首个可归因 A/B（P0-3+P1-5 基线 + SEED_POS 单开），但计划缺同会话 `SEED_POS=0` 对照，且配置的 `LAZY` 臂本身会携带 tap（`chain_dev.rs:7537/7929` 无条件 `carry_kept_tap`）——在 LAZY 布局下 SEED_POS 的立论前提（tap 在 `pos`）可能不成立（§2.2）。**
> **预期众数是"不变 ~1.2"（40-50%），"1.4-1.8"只有 25-35%**。
> **最该做的不是再推理，而是把 paired baseline 和 k_acc 直方图补齐，并把从未测过的 `SEED_ALIGN` / `SWALLOW` 两臂排进同一轮。**

---

*工部 · 只读分析，未执行 GPU 命令、未改动任何源码；本文件为唯一产出。*
*代码事实均给出 `文件:行号`；推断项按 §2.3 / §3 显式标注置信度。*
