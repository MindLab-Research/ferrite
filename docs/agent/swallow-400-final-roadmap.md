# SWALLOW 400 路线图 · 最终更新（最优配置 · 剩余路径 · 优先级 · 时间线）

> 户部 · 2026-09-12 21:47 UTC · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 本文件**取代**以下早期路线图（结论已被本 session 实测覆盖）：
> `swallow-unlocked-400-final-path` · `swallow-optimization-execution-plan` · `400-fastest-path-roadmap`
> （后两者的「AR Step 2 = −3~5ms」票面已被 A1a 实测证伪，见 §3）。
> 输入：`dspark-correctness-chain.md` 尾部（§6400-6511）· `session-final-handover` ·
> `l4l5-next-batch-implementation-plan` · `ar-further-optimization` · `ar-step3-r3-pdl-design` ·
> `ar-step2-a1a-fix-design` · `mrows-swallow-batched-implementation-design`。
> **口径纪律**：每条数字标来源 —— **✅实测 / 🟡设计 / ⚪理论**；tok/s 与步时**分别标注口径**
> （吞吐 = serve 墙钟实测；步时 = nsys/账本推算；`400` 的代数口径 = `1000·k_emit / step_ms`）。

---

## 0. 判决（先读五条）

1. **SWALLOW 最优配置已钉死**：batched m=6 **nograph** + **mrows b2+b3**（`ATTN_MROWS2` +
   `ATTN_MROWS_ROPE_NORM`）⇒ ✅实测 **63.8 tok/s**（+9.4% over 58.3 基线），**红线通过**
   （零拉丁 ✓ + 出师表前 100 字正确 ✓ + 0 panic / 0 hang ✓）。
2. **三条门永久 OFF**（红线）：`fold_r`（✗✗ 6× 退化）· `AR_STORE_FUSE`（✗✗ A1a bug，8× 退化 / 输出退化）
   · `B5/B4/1b`（✗ 中性、无收益、且 B4 依赖未验证的 `VERIFY_FORK`）。
3. **tcgen05 仍未解锁**：4 轮修复（含 +775/−69 TMA 修复）后 **rank 7 仍 `sync: misaligned address`**。
   根因疑为「rank 7 分片边界天然非 16B 对齐」⇒ 重启**必须先做根因定位，不投变体矩阵**。
4. **AR 是唯一已量化的杠杆**：**R2（attn+MoE 合并 −3.3ms）> R1（SINGLE_POLL −1~3ms）**；
   R3（PDL）只值 **0.17–0.4ms/步（≤1.5% 步时）** ⇒ 降为收尾。
5. **400 不在本段可达**：R1+R2 后 **22–24ms ⇒ 250–270 tok/s @ accept 5**；到 400（≤15ms）
   **还差 −7~9ms**，只能由 **L4/L5 全面 kernel 重写**闭合。

---

## 1. 实测成绩单（本 session 全部，✅ 实测）

| 配置 | 吞吐 | 判定 | 证据 |
|---|---:|---|---|
| **lazy 干净栈** | **91.1 tok/s** | ✅ 全局最佳（但 lazy 天花板 = EAGER，**非 400 路径**） | 实测 |
| **SWALLOW + mrows b2+b3** | **63.8 tok/s** | ✅ **batched 最佳 —— THE BEST**（+9.4%） | 计数 1-200 + 出师表红线双通过 |
| SWALLOW base（全 gate） | 58.3 tok/s | — 基线 | 实测 |
| SWALLOW + mrows b2（单） | 60.4 tok/s | ✓ 有效（+3.6%） | 实测 |
| SWALLOW + B5+B4+1b | 62.3 tok/s | ✗ **中性**（−2.3%，噪声内） | bisect（6d36693b） |
| SWALLOW + fold_r | 10.3 tok/s | ✗✗ **6× 退化**（6× 权重 re-staging） | bisect 确认退化源 |
| lazy + AR_STORE_FUSE | 90.9 tok/s **+ 输出退化** | ✗ A1a bug | 实测 |
| SWALLOW + AR_STORE_FUSE | 7.1 tok/s（**8× 退化**） | ✗✗ A1a bug | 实测 |

**mrows 家族的最终判定**：
```
+ mrows b2              → 60.4 (+3.6%)  ✓
+ mrows b2+b3           → 63.8 (+9.4%)  ✓✓ THE BEST
+ B5+B4+1b              → 62.3 (−2.3%)  ✗ 中性（batched m=6 kernel 已接近最优）
+ fold_r                → 10.3 (−84%)   ✗✗ 灾难（永久 OFF）
```
> **教训**：B5/B4/1b 在 batched 路径**不带来可测量收益**（每项 −0.13ms 的预期未兑现）——
> **batched m=6 的 kernel 已接近最优**，mrows b2+b3 抓住了大部分收益。

---

## 2. SWALLOW 最优配置（交付配置）

```
# —— SWALLOW 400 最优臂（63.8 tok/s，实测）——
SWALLOW_STEP=1                      # batched m=6
DSV41_VERIFY_GRAPH=0                # nograph（图 + SWALLOW 触发 ar5-hang）
DSV41_ATTN_MROWS2=1                 # mrows b3
DSV41_ATTN_MROWS_ROPE_NORM=1        # mrows b2
# —— 以下三条：不加入（中性 / 退化）——
# DSV41_GATE_MROWS_ROUTE=1          # B5  中性 → 不加
# DSV41_RMSNORM_ROPE_MROWS=1        # B4  中性 + 依赖 VERIFY_FORK（未验证）→ 不加
# DSV41_MROWS_ACT_CPASYNC=1         # 1b  中性 → 不加
# DSV41_MROWS_FOLD_R=...            # fold_r  6× 退化 → 永久 OFF
# DSV41_AR_STORE_FUSE=1             # A1a bug 8× 退化 → 永久 OFF
```

| 项 | 状态 | 依据 |
|---|---|---|
| batched m=6 nograph | ✅ 生产形态 | nograph 已实测 0 ar5-hang；图版本被 hang 阻塞 |
| mrows b2+b3 | ✅ 锁定 | 唯一兑现的 mrows 项（+9.4%） |
| B5/B4/1b | ❌ 不加 | 中性（−2.3%）；B4 需 `VERIFY_FORK`（默认 OFF，SWALLOW 下从未单独验证） |
| fold_r | ⛔ 永久 OFF | 6× 退化（ng=6 ⇒ 权重被每 ng block 重新 staging） |
| AR_STORE_FUSE | ⛔ 永久 OFF | A1a bug：SWALLOW 8× 退化 / lazy 输出退化（见 §3） |
| tcgen05 | ⏸ 锁定 | rank 7 misaligned 未解（见 §6） |

---

## 3. 永久 OFF 清单（红线 · 误开即回退）

| 门 | 实测后果 | 根因 | 处置 |
|---|---|---|---|
| `fold_r`（`MROWS_FOLD_R`/auto） | 63.8 → **10.3 tok/s（6×）** | auto ⇒ n≤1024 时 ng=M/1=6 ⇒ grid=nt×6，**权重行被 6 个 ng block 重复 staging（6× 权重读）** | **永久 OFF**；b2+b3 已吃下 mrows 收益 |
| `DSV41_AR_STORE_FUSE`（A1a MoE store fold） | SWALLOW **7.1 tok/s（8×）**；lazy 90.9 + **输出退化** | 同门同时开 attn 折（树内既有、round 19 已判坏）+ 静默切换 `wo_b` 数值路径 | **永久 OFF**（默认即 OFF，安全） |
| `B5` `GATE_MROWS_ROUTE` | 中性（并入 62.3） | 无收益 | 不加 |
| `B4` `RMSNORM_ROPE_MROWS` | 中性 | 依赖 `VERIFY_FORK`（默认 OFF，未单独验证） | 不加 |
| `1b` `MROWS_ACT_CPASYNC` | 中性 | 仅装载方式变化 | 不加 |

> ⚠️ **A1a 修复已被设计判定为「弃」**（`ar-step2-a1a-fix-design §0`）：其 MoE 载体在生产配置下
> **一个 launch 都没省**（被 `DOWN_FUSE`/`ADD_EPI` 挡在 `else` 分支），实测退化必出自 attn 折；
> 收益上限仅 −0.08~0.16ms/步 ⇒ **超 2 人日止损，不修**。

---

## 4. AR 账本与剩余路径（唯一已量化的杠杆）

**AR 现状（✅ nsys/账本）**：
- AR = **36%** kernel-sum，**6.58ms/步 = 84 轮 × 78.3µs**（`84` 是结构常数，`chain_dev.rs:1894`）
- **77.7µs/轮是「等待」**（跨 rank 自旋 / peer stamp 轮询 / host 侧 rank 漂移 / nsys 自旋放大），
  非工作量（同负载对照：`_hcpost_rows` 28.6µs vs `_hcpost` 6.1µs = 22.5µs 非工作量）

| 项 | 内容 | 收益 | 成本 | 状态 |
|---|---|---|---:|---|
| **R2** | **attn + MoE 两次 AR 合并为每层 1 轮**（当前每层 2 轮 × 40 层 = 80→84 轮；payload 2×dim，同步次数减半 ⇒ 84→44 轮/步） | **−3.3ms** | 中高（payload 变大、slot 扩容、数值顺序需重证） | **待设计落地 —— 真正的杠杆** |
| **R1** | `DSV41_AR_SINGLE_POLL`（A4：block 0 轮询 + 设备内广播字；960→8 poller/轮） | **−1~3ms** | 低（**代码已在树内、默认 OFF、零代码**） | **实施待 GPU 验证**；判据 = A0 探针 `avg_spin` 位移 |
| R3 | PDL（程序化依赖加载：AR 自身两发 node-gap + PTLC spin 窗口 + 后继 prologue 对折） | **0.17~0.4ms/步（≤1.5%）** | 中（1.5~3 人日） | **低优先级**；R2 后轮数减半 ⇒ 收益再减半 |

> **判词**：R3 只把「AR 的尾巴 + 邻居的头部」对折，**吃不掉 spin 主体**（§2.2 三条不可行证明）。
> 要 AR 离开 #1，杠杆在 **R2 的轮数**，不在 PDL。

---

## 5. 到 400 的缺口算术（口径已分离）

**400 的代数口径**（accept 5 ⇒ k_emit = 6）：
```
400 tok/s = 1000 · k_emit / step_ms  ⇒  step_ms ≤ 6000 / 400 = 15.0ms
```

| 阶段 | 内容 | 步时 | tok/s @ accept 5 | 来源 |
|---|---|---:|---:|---|
| **S0** | SWALLOW 当前最优（mrows b2+b3） | **~28ms** | ~214 | ✅实测 63.8（吞吐口径）+ nsys |
| **S1** | + **AR R2**（attn+MoE 合并） | ~24.7 | ~243 | 🟡设计（−3.3ms） |
| **S2** | + **AR R1**（SINGLE_POLL） | **22–24** | **250–270** | 🟡设计（−1~3ms），**待 GPU 验证** |
| **S3** | **缺口** | **还差 −7~9ms** | **还差 130–150** | — |
| **S4** | + **L4/L5 全面 kernel 重写** ⇒ 400 | **≤15** | **400 ✓** | ⚪🟡（见 §6/§7） |

> ⚠️ **口径警告**：63.8 tok/s 是 **serve 墙钟吞吐**（含 prefill/gap/真实 accept ≈1.6~1.8），
> **不等于** accept-5 归一的 214。两张表不可混读——本文件的「步时/tok/s @ accept 5」列
> 一律是**代数归一**，与实测吞吐口径分离标注。

---

## 6. 优先级排序（R2 > R1 > tcgen05 重启 > L4/L5）

| 优先 | 项 | 收益 | 人日 | ROI 理由 |
|---:|---|---|---:|---|
| **1** | **AR R2**（attn+MoE 合并） | **−3.3ms** | 2~3 | 单项最大、已量化、设计方向已定；84→44 轮直接砍同步面 |
| **2** | **AR R1**（`DSV41_AR_SINGLE_POLL`） | **−1~3ms** | 1 | **零代码**（树内已实现）、零风险；先跑 A0 探针归因再 A/B |
| **3** | **tcgen05 重启**（秩 7 对齐根因） | 解 gate/up（−1~3.8ms 🟡） | 3~5 | 已落树、CUDA 侧从未编译验证；**需先定位 rank 7 misaligned 根因**，不投变体矩阵 |
| **4** | **L4 / L5 全面 kernel 重写** | **−7~13ms** | 16~21+ | **唯一闭合 −7~9ms 缺口的通道**；但成本最高、**仓内零实测背书**（v17→v21 四变体全中性） |

**排序理由**：
1. R2/R1 是**低成本、已量化**的杠杆，且 R1 代码已在树内 —— 先兑现确定性收益。
2. tcgen05 排在 L4 前：它是**已落树、只差根因定位**的中成本项；而 **L4-3/L4-4 恰恰是 tcgen05 的收尾**，
   没有 tcgen05 解锁，L4 的 routed experts 靶心不满。
3. L4/L5 成本最高（16~21+ 人日）、风险最大（零实测背书）⇒ **必须在 R1/R2/tcgen05 给出现场后才有靶**，
   不做「先投 L4 找感觉」（勿重演 v17→v21 四变体全中性）。

---

## 7. 时间线（人日，非日历；GPU 测试串行）

| 阶段 | 内容 | 人日 | GPU 会话 | 步时落点 | tok/s @ accept 5 | 判据 |
|---|---|---:|---:|---:|---:|---|
| **T0** | SWALLOW 最优配置锁定（今晚已达成） | 0 | 0 | ~28 | ~214 | ✅ 63.8 + 出师表红线 |
| **T1** | **R2** 实施 + 验证 | 2~3 | 1~2 | ~24.7 | ~243 | 轮数 84→44 + 零拉丁 + k_acc 逐位 |
| **T2** | **R1** 实施 + 验证（A0 探针 → A/B） | 1 | 1 | **22~24** | **250~270** | `avg_spin` 位移 + k_acc 逐位 |
| **T3** | tcgen05 根因定位（秩 7 对齐） | 3~5 | 1~2 | — | — | `misaligned=0` + launch 计数 > 0 |
| **T4** | tcgen05 gate/up 落地 | 2~3 | 1~2 | ? | ? | 单层微基准 gateup 22.2µs |
| **T5** | **L4（占用 / MLP）** | 16~21 | 多 | −5~8 | ~300~340（60% 兑现） | 逐 kernel wave 证据 |
| **T6 🎯** | **L5（流水 / 满 wave）** | +2~5 | 多 | −2~5 | **~350~428（400 ✓ 计数口径）** | 满 wave、零暴露 |

> **总账**：到 400 约 **25~40 人日**；按仓史 **60% 兑现率**折票后，落点 **300~340 tok/s**，
> **400 需 L4/L5 足额兑现**。**400 的第一个可达点 = T5/T6（L4/L5 落地）**，且只对**计数口径（accept 5）**成立。

---

## 8. 风险与止损

| 风险 | 触发信号 | 止损 |
|---|---|---|
| **误开永久 OFF 门** | `fold_r` / `AR_STORE_FUSE` 被脚本带开 | 立即回退；两门列入权威脚本 FORBIDDEN |
| **R2 数值顺序回归** | payload 2×dim 后 fp 非结合加法差异 | parity 硬门（逐位），不通过即冻结轮数合并 |
| **R1 中性** | `avg_spin` 位移 < 1% | 不进默认值；转 R2 主攻 |
| **tcgen05 根因无解** | 秩 7 misaligned 复现且无单一根因 | 关闭路径，**不投变体矩阵**（勿重演 v17→v21） |
| **L4/L5 投入陷阱** | L4 单项 A/B 中性 | 不开新变体矩阵，除非 tcgen05 给出 routed 实测新地板 |
| **零拉丁回归**（用户硬要求） | 四段文本出现拉丁碎片 | 回退该 commit；单 gate 单 commit + `/proc/environ` 读回 |
| **口径漂移** | 吞吐 / 步时 / nsys 三计时器混用 | 一切账按 nsys；本文件 §5 已分离口径 |

---

## 9. 一句话交付

> **SWALLOW 最优 = batched m=6 nograph + mrows b2+b3（b2+b3 唯一兑现，B5/B4/1b 中性、fold_r 6× 永久 OFF）
> ⇒ ✅实测 63.8 tok/s + 红线通过。到 400 的唯一已量化路径 = AR R2（−3.3ms）+ R1（−1~3ms）
> ⇒ 22~24ms ⇒ 250~270 tok/s @ accept 5；剩余 −7~9ms 缺口只能由 L4/L5 全面 kernel 重写闭合。
> 优先级 = R2 > R1 > tcgen05 重启 > L4/L5；总账 25~40 人日，400 只对计数口径（accept 5，≤15ms）成立。**

---

*户部 · 只读分析 + 本文件（唯一产出）；未执行任何 GPU 命令、未改动任何源码。*
*所有 tok/s / ms / µs 均标来源（✅实测 / 🟡设计 / ⚪理论）；吞吐与步时/accept-5 归一两种口径已在 §5 显式分离。*
