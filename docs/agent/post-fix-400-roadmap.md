# 修复后的 400 路线图（干净 base → 400 tok/s）

> 中书省 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动源码。
> 代码基线：HEAD `0322561`（含 S1/S3 修复 `51edf7c`，逐条 `file:line` 核对）。
> 输入：`dspark-correctness-chain.md`（S1/S3 根因 + 重验清单）· `400-fastest-path-roadmap.md` ·
> `lazy-verify-optimization-path.md` · `400-final-frontier-analysis.md` · `swallow-unlocked-next-plan.md` ·
> `l4-l5-kernel-path.md` · `r2b-a4-combo-prediction.md` · `verify-specific-fusion-kernel-design.md` ·
> `s1-verification-preanalysis.md`。
> **本机无 GPU ⇒ 所有 ms/tok 标了来源（实测 / 账本推算 / 设计口径）。**

---

## 0. 判决（先读七条）

1. **本路线图的前提是一条"重验优先"的原则**：session 发现的 BASE 损坏（D1 = DIRECT 臂
   `compress_len` 双计，`chain_dev.rs:10459` 的 S1 修复）使**之前所有"优化损坏"的判定都不可作为
   证据**——它们测在同一个损坏的 base 上。**在干净 base 上，这些优化必须逐个重验，而不是直接判死。**
2. **重验的顺序不是按"改进潜力"排，而是按"重验成本 × 证据价值"排**：**R2 第一**（单 gate、
   历史 +6% 实测、修复后可能直接兑现），K1/K2 第二（同程序替代，结构更安全），
   MARKOV/LAZY_SDR 第三（draft 侧、单 gate），其余零代码件第四。
3. **修复后的起点口径**：`78.8 tok/s` 是 **e2e（含 ~1000ms prefill）**；同 run 的生成段
   `~33ms/步 ≈ 175 tok/s`，内部口径 lazy 步时 `22.56ms ≈ 98 tok/s`。**三个数在三把尺子上，
   全程必须标口径**（`400-final-frontier-analysis §1` 的口径警告）。
4. **R2 的 +6% 很可能本来就是真的**：`78.1 → 82.9`（`r2b-a4-combo-prediction §1`，同任务同 binary 相对位移）
   是**时间位移**，而 D1 损坏影响的是**正确性**（镜像跑飞 → indexer 读旧 slot），
   对 launch 账/耗时结构影响很小。⇒ **R2 的性能收益与"损坏"判定是两件独立的事**：
   修复后应重测"是否损坏"，若干净则 +6% 立即可兑现。
5. **lazy 的路走到 ~145 就到头，400 必须换臂**：`tok/s = k_emit/(k_emit·c_row + d + c)`，
   `c_row` 压到 EAGER 的 6.15ms（hc 融合 + sync 收敛 + SH_PAIR + tcgen05 全兑现）
   在 accept 5（k_emit=6）下 = `6×6.15+4.5 ≈ 41.4ms ≈ 145 tok/s`（代数地板）。
   400 要求步时 ≤15ms ⇒ `c_row ≤ 1.75ms/行` = **EAGER 的 1/3.5**（lazy 结构下不成立）。
   ⇒ **400 的路径是 batched（SWALLOW）或 L4/L5 kernel 重写，不是继续打磨 lazy。**
6. **SWALLOW 今天就能跑（nograph）**：`SWALLOW_STEP=1 + VERIFY_GRAPH=0` 已实测 0 ar5-hang + 零拉丁；
   被阻塞的只是"batched + CUDA graph"。⇒ 重验 R2 拿到 +6% 后，**关键路径立即转向 batched nograph 的
   口径校准 + 零代码 mrows/SH_PAIR A/B**（潜在 −9~13ms）。
7. **红线的读法必须升级**：**"零拉丁"是必要不充分条件**——损坏的 base 就是"计数数字错但拉丁检查通过"
   （`dspark-correctness-chain`）。**每一次重验的判据 = 计数数字顺序正确（主探针）+ 零拉丁 + k_acc 逐位。**
   三缺一不能下结论。

---

## 1. 起点钉死（假设 S1/S3 修复成功）

### 1.1 干净 base 的构成

| 组件 | gate / 落点 | 状态 |
|---|---|---|
| lazy 逐行 verify | `DSV41_LAZY_VERIFY=1`（`chain_dev.rs:2591`） | 已开 |
| SH_PAIR | `DSV41_SH_PAIR_M=1`（`chain_dev.rs:1392`） | 已开（**实测中性偏负 78.8→78.1，`r2b-a4-combo-prediction §0-4`**） |
| Wave 1 | hc 融合 + AR fold 等 | 已开 |
| e4m3 激活 | `DSV41_EXPERT_ACT_E4M3=1` | 已开（正确性 + 单趟） |
| BF16 截断 | `DSV41_BF16_TRUNCATE=1` | 已开（零拉丁的防线） |
| **S1 修复** | `chain_dev.rs:10459` `mirror_advanced`（DIRECT 不再双计） | ✅ 已提交 `51edf7c`，验证中 |
| **S3 修复** | `DSV41_LAZY_ROUTE_LOCK`（默认 ON，`chain_dev.rs:2625`） | ✅ 已提交，验证中 |

### 1.2 修复验证测试的判据（Step 0，正在跑）

配置 = **base + S1/S3 fix**（无 R2/K1/K2，`VERIFY_GRAPH` 可开）。判据：

1. **计数任务 1-200 全部行号自洽**（`数字正确 == 总行数`）——**主探针**；
2. **出师表零拉丁 + 逐字 + LEN 一致**；
3. **`k_acc` 序列 ~5（计数口径）**，与历史 lazy 序列可比；
4. `faults=0`、无 hang。

**分支**：
- **全 PASS** ⇒ base 干净，进入 §2 的重验序列（本路线图的正题）。
- **仍 line-62 重置** ⇒ D1 不是唯一根因（**indexer_topk 的 baked n_pos 是第二嫌疑**，`ba5cacc`），
  **立刻停止所有性能工作**，转 S2/indexer_topk 修复。**不得在损坏的 base 上重验任何优化**。

---

## 2. 重验序列（修复后的核心工作）

> **铁律**：一臂一进程（`OnceLock` 每进程读一次，`chain_dev.rs:2133` 等）、**单变量**、
> **同会话背靠背**、每门**读回 `/proc/<pid>/environ` + `nm -D $SO`**（本项目 #1 陷阱：gate 设了没生效）。
> 每一格的判据 = §1.2 的四条（缺一不可）。

### Step 1 — R2 重验（`DSV41_ATTN_LIN_FUSE=1`）★最高价值

| 项 | 内容 |
|---|---|
| gate | `DSV41_ATTN_LIN_FUSE=1`（`chain_dev.rs:2133`），含 bisect 臂 `=2`(lin2 only) / `=3`(rope_norm only) |
| 机制 | `attention_rows(m=1)` 复用 EAGER 的 `lin2`(wq_a+wkv) + `lin_rope_norm`(norm+wq_b+rope)，**7 发/层 → 2~3 发/层** |
| 之前判定 | "损坏"（82.9→84.0 无效）——**但测在损坏 base 上，判定不可靠** |
| **重验预期** | **若干净 ⇒ +6%（78.8 → ~83.5）立即可兑现**；bisect 臂可定位（若仍损坏） |
| 成本 | 0 代码 + 0.5 人日 A/B |
| 风险 | R2 复用**不同程序**（EAGER 的 m=1 `gemm_fp8_gemv` 族 vs mrows 的 `gemm_fp8_mrows`），
bit-identity 是"claimed but never measured"（`chain_dev.rs:2148-2157`）；且 R2 触及单行 scratch
`s.xq`/`s.xsc` 的 consume-once 语义（`xq_of_*` 标志）——**这是 FMA/程序差异的历史坑源** |

### Step 2 — K1+K2 重验（`DSV41_ATTN_MROWS2=1` + `DSV41_ATTN_MROWS_ROPE_NORM=1`）

| 项 | 内容 |
|---|---|
| gate | K1 `DSV41_ATTN_MROWS2`（`chain_dev.rs:2170`）；K2 `DSV41_ATTN_MROWS_ROPE_NORM`（`chain_dev.rs:2236`） |
| 机制 | **同程序替代 R2**：K1 = `gemm_fp8_mrows2`（两族合一发）；K2 = `gemm_fp8_mrows_rope_norm`（norm+quant+wq_b+rope 合一发）。每一段都**拷贝自 verify 今日在用的 kernel**，不换程序（`chain_dev.rs:2217` 的设计原则） |
| 之前判定 | "损坏"（与 R2 同模式）——**同样是在损坏 base 上** |
| **重验预期** | 若干净 ⇒ 与 R2 同量级（launch 7→2）；**结构上比 R2 更可信**（无 EAGER 程序复用、无 `s.xq` scrub 共享、无 `pos_ctr` 读取） |
| 成本 | 0 代码（已实施）+ 0.5 人日 A/B |
| 决策 | **若 R2 干净且 K1/K2 也干净**：二者性能等价时**选 K1/K2**（同程序 = 更安全的长期默认）；
**若 R2 损坏但 K1/K2 干净** ⇒ K1/K2 就是 R2 的替代，拿回同一笔收益 |
| 风险 | K1 与 R2 的 `lin2` 有优先级（`chain_dev.rs:2167`：两者都开时 K1 赢，R2 半不达）；
K2 的 decline 路径必须确认（stale .so / swapAB / shape 拒绝时**必须回落到四发序列而非逐行**，
`chain_dev.rs:2233`） |

### Step 3 — MARKOV_SLICED + LAZY_SDR 重验（draft + lazy 侧）

| 项 | gate / 落点 | 机制 | 之前判定 | 重验预期 |
|---|---|---|---|---|
| **MARKOV_SLICED** | `DSV41_MARKOV_SLICED`（`dspark_dev.rs:492`） | draft 词表切分 | "退化 accept 5.0→1.4" | −1.0ms（draft）；**退化可能是 base 损坏的表现** |
| **LAZY_SDR** | `DSV41_LAZY_SDR`（`chain_dev.rs:2685`） | per-row sync 收敛（`memcpy_d2d_2d` + H2D 合并） | "退化 accept 5.0→2.4" | −0.7ms/步 |

**⚠️ 关键疑点（必须隔离读）**：`chain_dev.rs:8811` 的注释本身记着 LAZY_SDR 与
"the verify argmax (k_acc 5.0→2.4)" 的关联——**这可能是真效果（SDR 的 argmax 合并改变了
早退行为），也可能是 base 损坏**。⇒ 重验时必须**同会话 `LAZY_SDR=0/1` 背靠背**，
并**同时读 k_acc 直方图 + 计数数字正确性**，不能只看 accept。若 accept 真的降但数字正确，
则 LAZY_SDR 是"性能件但换 accept"，需按 §4 的门槛判值不值。

### Step 4 — 其余零代码件逐个隔离

| 项 | gate | 默认 | 机制 / 落点 | 预期 | 风险 |
|---|---|---|---|---|---|
| **VERIFY_FORK** | `DSV41_VERIFY_FORK` | OFF（`chain_dev.rs:1165`） | verify 的 attn 双链（kv 侧流）+ MoE 双链（routed/shared 分叉） | 设计口径（未隔离测） | 侧流引入 → 与 CUDA graph 的臂分歧（ar5-hang 的结构同源）|
| **RING_WIN_FUSE (R3)** | `DSV41_RING_WIN_FUSE` | **verify 路径 OFF**（`chain_dev.rs:2430`，B2 主路径默认 ON `:2403`） | 窗口环 append + 每行 causal window 合一发 | 账本 | 历史被卷入 hang 组合；**最后单独上** |
| **wo_a / WO_PAIR** | `DSV41_WO_PAIR` / `DSV41_WOB_F32` | WO_PAIR OFF（`:4700`）/ WOB_F32 **ON**（`:4681`） | wo_a→wo_b 链对网格同步；wo_b f32 直读 | 小 | 链对同步 = 又一处 collective 形态 |
| **HCPOST_EPI / PROJ_FUSE / NORM_FUSE / OROPE_Q** | 各 gate | 多为 ON | 单核融合 | 小 | 与 hc 融合重叠计账 |

### Step 5 — 组合 + 红线收口

把 Step 1~4 判定"干净"的项**按单变量顺序叠加**（一次只加一个 gate，读回确认），
每叠一层跑一次**四段文本 + 计数 + k_acc**。组合后必跑：
`dspark_parity` 行级（`verify_bad == 0`）+ 出师表逐字 + `DSV41_DIFF_EAGER=1` 的 `[diff]` 行。

**诚实预期**：若 R2/K1K2/MARKOV/LAZY_SDR 全干净，**干净栈 ≈ 85~90 tok/s（e2e）**
（session 自估，`dspark-correctness-chain` 尾段）——这是本次重验的**上限收益**，
约 **+8~14% over 78.8**。但**这仍远不是 400**。

---

## 3. 重验之后的 400 路径（两堵墙）

> **重验只恢复 R2 一族的 +6% 与若干 ms；它不动 `k_emit × c_row` 这个乘积。**
> 400 的建设性工作全部在 §3 的两堵墙之后。

### 3.1 墙与路径（口径：生成速率 tok/s = k_emit/step）

| 阶段 | 内容 | step（计数口径，k_emit=6） | tok/s | 依据强度 |
|---|---|---:|---:|---|
| **S0** | 修复后干净 base | 22.56ms（lazy，accept 1.2）/ ~52.9ms（计数 accept 5） | **98 / 113** | 实测 |
| **S1** | + R2 或 K1/K2（重验干净） | ~21.6 | ~104 | 账本（+6% e2e 折算） |
| **S2** | + lazy 零代码（hc + SDR + SH_PAIR M=1） | ~17.4 | **127** | 账本 |
| **S2″** | lazy 天花板（c_row→6.15） | 40.7 | **145** | **代数（结构上限）** |
| **S3** | batched nograph + SWALLOW + mrows 族 + SH_PAIR<M> | ~19~22 | **273~316** | 设计口径 × 兑现率 |
| **S4** | + tcgen05（gate/up，**down 无核**） | ~15.5~18 | **330~387** | 修正口径 |
| **S5** | + B6 + MMA 全族化 / L4 | ~12~14 | **430~500** | 设计口径，仓内零实测 |

### 3.2 重验后的关键路径（判决）

**关键路径 = `R2 重验 → 转 batched nograph → 口径校准 nsys → 零代码 mrows/SH_PAIR A/B → tcgen05`。**

- **不做**继续在 lazy 上打磨到 400（数学不可能，§0-5）；
- **不做** ar5-hang 的系统性修复，除非 nsys/探针证明"AR 等待 ≥2ms/step"（图只值 −1.5ms，
  真正价值是 AR peer-stamp 等待的上界 −3.9ms，实测未知）；
- **不做** L4/L5（25~34 人日、仓内零实测背书）——它是"越过 400 之后"的层。

### 3.3 batched 这一步的具体动作（零代码优先）

| 序 | 动作 | gate | 预期 | 判据 |
|---|---|---|---|---|
| B0 | batched nograph 口径校准 | `DSV41_SWALLOW_STEP=1` + `DSV41_VERIFY_GRAPH=0` | 基线 | `[dspark] steps=` 的 verify/draft/commit + route + k_emit 分布 |
| B1 | m=6 mrows 族逐个 A/B | `GATE_MROWS` → `INDEXER_MROWS` → `VERIFY_ROPE_MROWS` → `VERIFY_HEAD_MROWS`(最后单独) | −4.5~5.8ms | 每个 gate：k_acc 逐位不变 + `verify_ms` 位移 + **kernel 名确认变体真的跑了**（`gemm_fp8_mrows_kernel<6>`）|
| B2 | SH_PAIR `template<M=6>` e2e | `DSV41_SH_PAIR_M=6` | −4.9~7.9ms | parity 已确认虚警（`d05bfde`）⇒ 只差 e2e A/B |
| B3 | tcgen05 gate/up | 5-gate 链 | −1.0~3.8ms | 先澄清 `e4m3 × tcgen05` 互斥（`chain_dev.rs:687`）+ 单层微基准门 |

---

## 4. 优先级总表（按"重验/实施顺序"）

| # | 步骤 | 类型 | 预期 | 成本 | 依赖 | 止损门 |
|---|---|---|---|---|---|---|
| **0** | S1/S3 修复验证（正在跑） | 验证 | base 干净 | 0（已跑） | — | 仍 line-62 ⇒ 停性能、修 indexer_topk/S2 |
| **1** | **R2 重验**（`ATTN_LIN_FUSE=1`） | 重验 | **+6%**（若干净） | 0.5 人日 | Step 0 | 仍损坏 ⇒ 转 bisect `=2/=3`，再转 K1/K2 |
| **2** | **K1+K2 重验** | 重验 | 同 R2 量级 | 0.5 人日 | Step 1 | decline 路径必须回落四发（非逐行） |
| **3** | MARKOV + LAZY_SDR 重验 | 重验 | −1.7ms/步 | 0.5 人日 | Step 0 | accept 真降且数字正确 ⇒ 按 §4 门槛判值 |
| **4** | VERIFY_FORK / RING_WIN / wo_a 等逐个 | 重验 | 分散小项 | 1 人日 | Step 0 | 任一 gate 位移 < 预期 40% ⇒ 停该项 |
| **5** | 组合 + 红线（四文本+计数+parity） | 验证 | 干净栈 85~90 e2e | 0.5 人日 | Step 1~4 | 出现拉丁/数字错 ⇒ 回退该单 commit |
| **6** | **转 batched nograph + 口径校准 nsys** | 转向 | 重钉 S3 基线 | 1 人日 | Step 5 | — |
| **7** | batched mrows 族 + SH_PAIR<M> A/B | 零代码 | −9~13ms | 2~3 人日 | Step 6 | 位移 < 40% ⇒ 转 tcgen05 |
| **8** | tcgen05（gate/up） | kernel | −1.0~3.8ms | 4~5 人日 | Step 7 | 单层微基准不达标 ⇒ 关闭路径 |
| **9** | B6 + MMA 全族化 / L4 | kernel | −5~8ms | 8~25 人日 | Step 8 | 零实测背书，最后一投 |

**最快落点（60% 兑现率，历史值）**：Step 0~8 ≈ **步时 18~20ms ⇒ 计数 300~340 tok/s**（差 400 约 15~25%）。
**400 需要 S3/S4/S5 三项同时足额兑现**——仓史上前所未有（`swallow-unlocked-next-plan §2`）。

---

## 5. 风险清单

| ID | 风险 | 触发 | 应对 |
|---|---|---|---|
| R0 | **S1/S3 修复不彻底** | Step 0 仍 line-62 | 停性能，修 indexer_topk baked n_pos（`ba5cacc`）/ S2 |
| R1 | **重验污染**（在未确认干净的 base 上重验） | Step 0 未 PASS 就跑 Step 1 | **硬门禁：Step 0 PASS 才开 Step 1** |
| R2 | **gate 设了没生效**（本项目 #1 陷阱） | `verify=` 无位移 | 每门读回 `/proc/environ` + `nm -D` + nsys 数 kernel 名 |
| R3 | **R2 真损坏**（非继承） | 重验后仍损坏 | 转 K1/K2（同程序替代）；R2 的 kernel 级等价性独立诊断 |
| R4 | **"零拉丁"假阴性** | 拉丁通过但计数错 | **判据升级：计数数字顺序为主探针**（本 session 的核心教训） |
| R5 | **LAZY_SDR 真的换 accept** | `k_acc` 5.0→2.4 且数字正确 | 按门槛 `lazy iff k_emit×c_row < B` 判值；不达标不进默认 |
| R6 | **mrows 族兑现率 0**（先例：SH_EXP 两次零收益） | 任一 gate 位移 < 40% | 停 mrows 线，转 tcgen05/MMA |
| R7 | **batched + graph 的 ar5-hang** | 图开启后 hang | 保留 **nograph 作生产形态**（已可用）；不进 ar5 修复除非探针证明 AR 等待 ≥2ms |
| R8 | **tcgen05 红线** | 冒烟拉丁/非法指令 | 立即关闭路径，不投变体矩阵（勿重演 v17→v21） |
| R9 | **口径混用**（步时/accept/timer 三者混在一个乘式） | 跨会话比较 | 每次比较标 **arm + m + timer**；同会话背靠背 |

---

## 6. 一句话总结

> **修复后第一件事不是"上优化"，是"重验"——因为之前所有"优化损坏"的判定都站在同一个损坏的 base 上，
> 不可作证据。重验序列 = R2（+6% 若干净）→ K1/K2（同程序替代）→ MARKOV/LAZY_SDR → 其余零代码件。
> 判据必须用计数数字（主探针）+ 零拉丁 + k_acc 逐位，三缺一不下结论。**
> **重验最多把干净栈抬到 85~90 tok/s（e2e，≈ +8~14%）——它不动 `k_emit × c_row` 这个乘积。**
> **400 的路径在重验之后：lazy 的代数地板是 ~145 tok/s（k_emit=6，c_row→6.15），400 要求
> `c_row ≤ 1.75ms/行` = EAGER 的 1/3.5——lazy 结构下不成立。⇒ 关键路径转 batched nograph
> （SWALLOW 今天就能跑，0 hang + 零拉丁）的零代码 mrows/SH_PAIR A/B，再叠加 tcgen05 与 B6。
> 60% 兑现率的现实落点是 300~340 tok/s；400 需要 S3/S4/S5 三项同时足额兑现。**

---

*中书省 · 只读分析 + 本文件（唯一产出）；未执行任何 GPU 命令、未改动任何源码。*
*所有 ms/tok 均标来源（实测 / 账本推算 / 设计口径 / 代数）；代码事实均附 `file:line`。*
