# SWALLOW 不用图的性能优化路径（只读分析）

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 任务：`SWALLOW_STEP=1 + VERIFY_GRAPH=0` 完全工作（0 ar5-hang / 零拉丁 / k_acc 正常），
> 但步时 ~40ms（serve 侧）——分析这条**不用图**的路径如何优化，并给出 Plan B 的实施建议与最终步时预测。
> 代码基线：工作树 HEAD `cbaa83e`。现场核对：`chain_dev.rs`（`swallow_step()`/`step_rows_sync`/`verify_graph_gate`/
> `dspark_spec_swallowed`/`lazy_run_row`/`dspark_spec_lazy`）。
> 账本输入：`verify-architecture-floor` · `verify-ms-breakdown` · `swallow-unlocked-next-plan` · `swallow-result-action-plan` ·
> `batched-400-v2-prediction/-remaining-roi` · `sh-pair-template-m-design` · `tcgen05-e4m3-grouped-expectation` ·
> `b6-mrows-f32-design` · `lazy-batched-gate` · `dspark-correctness-chain` · `l4-l5-kernel-path`。

---

## 0. 判决（先读七条——三条修正任务前提）

1. **🔴 ~40ms 不是「没有图」造成的。图（`VERIFY_GRAPH`）实测只值 −1.5ms**（同会话 A/B：replay 35.5ms vs 裸链 37ms，
   `batched-400-v2-prediction §1.1`、`verify-ms-breakdown §修正`、`verify-architecture-floor §3.1`）。
   "50% submit + 50% exec" 的分解是**错的**——CUDA async launch 早已让 CPU submit 与 GPU 执行重叠。
   ⇒ **「不用图」的惩罚 ≈ 1.5ms，不是 15ms。** 40ms 的构成里 38.5ms 与图无关。

2. **🔴 40ms ↔ 25ms 的 15ms 差，主项是「batched m=6 vs lazy ~2 行」，不是「图 vs 无图」。**
   实测：batched step ≈ **39ms** vs lazy step（含 m=1 图）≈ **22.56ms**（`verify-architecture-floor §5.3`、
   `dspark-correctness-chain` lazy 段）⇒ 结构差 **16.4ms**。叠加图的 −1.5ms 后正好 ≈ 任务的 15ms 差。
   ⇒ **任务里的「SWALLOW 应该比 lazy 更快」在这套（tiny SIMT kernel）架构下不成立**——
   `verify-architecture-floor §5.3` 的原话：*"在今天的核效率下，batched verify 结构上慢于 lazy"*。
   字节不是约束（折叠后 0.74ms = 5% 峰值），约束是 **μop 数 + 3~4 warps/SM 的延迟暴露**；
   batched 一次读权重省不到这两堵墙，却把 activation 侧做了 **6 行**。

3. **🔴 Plan B（unanimity-or-direct）的步时价值 ≈ 1.5ms，不是关键路径。**
   Plan B 让 SWALLOW+图工作，但图本身只值 −1.5ms。**SWALLOW 的性能杠杆全在融合/换核（SH_PAIR / tcgen05 / B 类 /
   mrows 族），而它们的开关与图无关**（mrows 是 kernel dispatch，SH_PAIR 是 kernel 本身，tcgen05 是 kernel 代数——
   都不需要 CUDA graph）。⇒ **Plan B 应排在融合之后，且要明确它的定位是「把图当正确性/顺序工具拿回来」，
   不是「让 SWALLOW 变快」。**

4. **SWALLOW 的「主链吞并」收益已经可以通过 `LAZY_VERIFY` 拿到——因为 lazy 蕴含 swallow。**
   `lazy_verify()` 的 row 0 **就是**被吞的主链步（`chain_dev.rs:2261-2267` "lazy IMPLIES swallow"；
   row 0 = `[anchor,d1..d5]` 的 anchor 行 = `step_dev` 本该做的 forward）。
   ⇒ 在 accept 1.2 的任务上，**lazy 路线同时拿到了 −4.55ms 的吞并收益** + 2 行的廉价经济。**这才是现状的正解。**

5. **「SWALLOW 真实价值 −9~10.4ms」是设计口径的乐观账，不能当基线。**
   −4.55（吞主链）+ −4.5~5.8（m=6 mrows 族）里，后半截**一分未兑现**（`{SH_EXP+GRAPH+ROPE+P3A}` 全开实测
   **−1.21ms**，`SH_EXP_MROWS` 两次零收益）。账本已把 60% 兑现率折算落点写成 ~18-20ms / 300-340 tok/s。

6. **不用图的优化必须按「削哪一列」排序（`verify-ms-breakdown §3.2`）**：
   只有 **融合/换核**（削 kernel 自己的 μop + 每发最小执行）能真正下台阶；**图化只削 submit 半，而 submit 已重叠 ⇒ 无肉**。
   ⇒ 顺序：**mrows 族（零代码）→ hc A1/A2 → SH_PAIR → tcgen05 → B 类**；**Plan B（图）最后**。

7. **本轮的绝对纪律**：任何 ms 数必须标口径（**实测 / launch 账 / ms 账 / 设计口径**）。
   当前 40ms 是 **serve 侧墙钟（用户已两次判不可信）**——它含 curl/HTTP/admission/tail 四项非模型开销
   （`serve.rs:450-455` 自述注释）。**要拿真值必须先跑一次 nsys**（`nsys-wave1-analysis-framework §1.4` 口径 A/B）。

---

## 1. 40ms 的精确归因（无图 + batched m=6）

### 1.1 只用「模型内部计时」拆

`[dspark] steps= … verify=X draft=Y commit=Z`（半真，含 host barrier + D2H sync，但比 serve 墙钟准）：

| 段 | 量级 | 来源 |
|---|---|---|
| verify(`m=6`, direct 裸链) | **~34~37ms** | `verify(m=5)=37.31` + 一行 ≈ +1.6ms 的设计账（`swallow-step-diff`）扣掉图 |
| draft（5 个 MTP 块 × 3 层） | **~4.3ms** | `dspark-correctness-chain` "draft 4.28"（P3a 未开时） |
| commit（rollback + compressor replay + counter） | **~0.2~0.8ms** | 同上 "commit 0.17~0.82"，稳定 |
| **合计** | **~39~42ms** | 与任务「~40ms」吻合 |

⇒ **40ms 的 90%+ 是 verify(m=6) 本身**。图在这里能拿回的只有 ~1.5ms。

### 1.2 逐族账（m=6 口径，设计/账本混合）

`verify-ms-breakdown §1` 的 m=5 逐族 × m=6 修正（Swallow 多一行 ⇒ activation 侧族 ×~1.2，权重/AR 族不变）：

| 族 | m=5 实测 ms | m=6 估计 ms | 性质（`verify-ms-breakdown §3.1`） | 无图可削？ |
|---|---:|---:|---|---|
| **shared expert** | 10.40 | **~12.5** | 核效率 + 5× 重读（85GB/s，1.1% 峰值） | ✅ SH_PAIR |
| **routed experts** | 8.30 | **~9.96** | L1TEX μop 地板（378GB/s，4.9%） | ✅ tcgen05 |
| head | 1.12 | ~1.12 | 唯一跑满带宽（5.9TB/s） | ❌ 无肉 |
| 投影族 | 3.70 | ~4.4 | launch/mrows（instruction-bound） | ⚠️ mrows 零收益 |
| attention | 2.80 | ~3.4 | 纯 launch/低占用 | ⚠️ b·m 生产恒 decline |
| hc 链 | 2.96 | ~3.0 | 纯占用（53GB/s，全表最低） | ✅ hc A1/A2 |
| MoE gate | 3.44 | ~4.1 | 逐行 ×6 + 核效率 | ✅ GATE_MROWS（未实测） |
| indexer | 2.50 | ~3.0 | 逐行 + topk 单核地板 | ✅ INDEXER_MROWS |
| compressor | 0.55 | ~0.7 | 小 | ✅ COMPRESSOR_MROWS |
| engram / norm / 其它 | ~0.6 | ~0.7 | 小 | ❌ |
| all-reduce v5 | 1.40 | 1.40 | 协议地板（m 无关） | ❌ 只能减次数 |
| **合计** | **37.31** | **~44（粗估）** | | |

> ⚠️ 上表 m=6 合计粗估 44ms 高于实测 40ms——说明「逐族 ×1.2」在权重复用族上过估（batched 一次读权重）。
> **以实测 ~39-42ms 为准，逐族只做「谁是大头」的相对排序**：**shared(12.5) + routed(10) = 22.5ms ≈ 56%**，
> 其后 gate 4.1 > 投影 4.4 ≈ attention 3.4 ≈ hc 3.0 ≈ indexer 3.0。
> ⇒ **无图路径的两头（shared + routed）正是 SH_PAIR 与 tcgen05 的靶子——这不是巧合。**

---

## 2. 图 vs 无图：把 −1.5ms 钉死（防止把 15ms 记到图头上）

| 口径 | 值 | 证据 |
|---|---|---|
| 理论（6224 发 × 2.9µs submit） | −18ms | `verify-ms-breakdown §2` 的旧模型 |
| **实测 A/B（m=5 batched）** | **−1.5ms** | replay 35.5ms vs 裸链 37ms；`{SH_EXP,GRAPH,ROPE,P3A}` 全开 = 36.10ms（−1.21ms） |
| 实测（lazy，m=1 图捕获 at pos=16） | −1.6ms（24.15→22.56ms） | `dspark-correctness-chain` |
| 机理 | **submit 与 GPU 执行重叠** ⇒ 图只削已被隐藏的 submit | `arch-floor §2.3`：hc 族实测 2.96ms ≈ `hc_mixes` 隔离微基准 7.8µs×400，**boundary 项≈0** |

**无图路径的实际副作用（才是「不用图」真正付出/省下的东西）**：

1. **省下了 capture 的两个 host barrier**，且 gate 关闭时 direct 臂**不再进 barrier**（`chain_dev.rs:5425`
   的 `!skip_barrier && verify_graph_want()`）⇒ **全 rank 都走 direct、0 barrier、臂唯一** ⇒
   **这是 0 ar5-hang 的机制**（无臂可分歧）。
2. **付出 ~1.5ms**（submit 未被完全隐藏的残余）。
3. **失去「顺序/正确性工具」**：图 replay 能保证 launch 顺序可复现；无图靠裸链的依赖序（本仓已验证裸链正确）。

⇒ **结论：不用图的「净代价」≈1.5ms；「净收益」= 结构上不可能发生臂分歧。这笔交易是划算的**（1.5ms 换掉一个
竞态类 hang）。后续优化不应以「把图加回来」为第一目标。

---

## 3. SWALLOW(batched) vs lazy：都用图 / 都不用图 的正面对比

### 3.1 结构（同一个 `[anchor,d1..d5]` 块，两种跑法）

| 维度 | SWALLOW（batched） | lazy（implied swallow） |
|---|---|---|
| 块 | `[anchor,d1..d5]` @ `pos..pos+5`，**一次 m=6 forward** | 同块，**逐行 m=1**，**首个 miss 即停** |
| 每步 forward 行数 | **恒 6** | **k_emit = k_acc+1**（accept 1.2 ⇒ ~2.2） |
| 权重读 | 1 次/块（activation ×6） | k_emit 次（activation ×k_emit） |
| launch/步（估） | ~7000（m=6 全行） | ~2500（~2.2 行 × m=1） |
| 回滚 | 有（rejected tail） | **无**（跑过的行全在 keep 范围，`§3.4`） |
| compressor replay | 有 | **无**（每行自己 commit） |
| 图形状 | `m=6` 一槽 | `m=1` 一槽 |

### 3.2 实测步时（**分「有没有图」两栏**）

| 臂 | 无图 | +图 | 备注 |
|---|---:|---:|---|
| **SWALLOW batched** | **~39-40ms** | **~37.5-38.5ms** | 本次任务实测 ~40ms（serve）；图回补 −1.5ms |
| **lazy**（accept 1.2） | ~24.1ms | **~22.56ms** | `dspark-correctness-chain`：m=1 图捕获后 24.15→22.56 |
| **lazy**（Wave 1 组合） | — | **~25.17ms**（serve 侧；内部口径 ~31-33ms） | `nsys-wave1-analysis-framework §4` 修正：25ms 是 serve 假值 |

⇒ **在同一 accept（1.2）下，lazy 全面胜出**（22.5 vs 37.5），**差距 15ms，与图无关**。
⇒ **lazy→batched 的反转点**由 `lazy-batched-gate §0.7` 给出：`lazy iff (1+mean_k) < B/c`，
   取 `B≈28ms`（SWALLOW 后）、`c=6.15` ⇒ **阈值 mean_k ≈ 3.55**。当前 accept：对话 0.96 / 出师表 1.21 / 计数 5.0
   ⇒ **对话与出师表应走 lazy；计数才轮到 batched。**

### 3.3 batched 的 launch 更少但每次更贵——这句在账本上是对的，但**省不到钱**

- launch 少：batched 7000 vs lazy 2500——但 **submit 已重叠**（§2），且融合还没把 kernel 数降下来。
- 每次更贵：batched 的 activation 侧 ×6（lazy ×2.2）——**在当前核效率下这一项主导**（`arch-floor §5.3`）。
- ⇒ **batched 的「launch 更少」在图的 −1.5ms 上都体现不出来**（submit 被隐藏），更别说弥补 activation ×6。
- **batched 唯一的结构性翻盘条件**：`mrows/占用让权重共享真正省钱`（= L4）或 `accept ≥3.55`。
  两者今天都不成立（mrows 实测零收益；accept 1.2）。

---

## 4. 不用图的优化路径（按 ROI 排序）

> 口径：launch 账（可确定，纯计数）/ ms 账（设计口径，未实测）。**表内所有 ms 均标来源。**

| P | 动作 | 文件 / gate | 削哪一列 | 预期 | 成本 | 状态 | 依赖图？ |
|---|---|---|---|---|---|---|---|
| **P0** | **先钉死真值**：nsys 一次（口径 A 窗口切分） | `scripts/nsys_wave1.sh` | — | 重钉 40ms 的真值 | 1 GPU 会话 | **必做**（否则后面全是设计口径） | — |
| **P0b** | **SWALLOW 无图正确性加固**：3/3 独立运行 + 一次长跑 | — | — | 防「假解锁」 | 3 GPU 会话 | 建议（0 hang 一次不算修好，`swallow-unlocked §7`） | — |
| **P1** | **m=6 mrows 族逐个 A/B**（零代码） | `GATE_MROWS` → `INDEXER_MROWS` → `COMPRESSOR_MROWS` → `VERIFY_ROPE_MROWS` → **`VERIFY_HEAD_MROWS` 最后单独上** | **两列**（launch N↓ + 每发变大吸收固定项） | **−4.5~5.8ms（设计）**；兑现率历史 ≈0（`SH_EXP` 两次零收益） | **0**（gate 已在） | 代码就位 | ❌ 不需要图 |
| **P2** | **hc 链 A1/A2 复核**（Wave 1 已含，确认无图下仍生效） | `HC_VERIFY_FUSE` + `HC_FRONT_ROWS`（`truncate=false` 已修） | launch + 占用 | **−1.3~1.7ms** | 0.5~1 人日 | 已落地，需复核 | ❌ |
| **P3** | **SH_PAIR `template<M>`**（**最大单项**） | `gemm_fp8_sh_exp_pair_kernel<M>` + `SH_PAIR_M` | launch（1000→80/步）+ 核效率 | **−4.9~7.9ms（设计）** | 2~4 人日（parity 硬门） | **parity 36 failed**（phase-1 aq 差异，`sh-pair-template-m-design`） | ❌ |
| **P4** | **tcgen05 gate/up**（routed 换核） | `EXPERT_GROUPED` + `EXPERT_TCGEN05_E4M3` + `GATEUP_FUSE=0` + `EXPERT_ILV=0` | 核效率（删 L1TEX μop） | **−1.0~3.8ms（修正口径，非 −6.8）**；down 无 tcgen05 核 | 4~5 人日 + 对齐/parity | misaligned 守卫已入，**从未上 GPU**，两条 `[OPEN]` | ❌ |
| **P5** | **B 类公共祖先 B6 `dsv41_gemm_fp8_mrows_f32`** | `b6-mrows-f32-design.md` | launch | **−0.66~1.5ms（B6 单项）**；B1–B6 全族 −2.8~4.9ms | 0.5 人日（B6）；8~12（全族） | **新增 kernel** | ❌ |
| **P6** | **L4 占用/MLP**（routed 的 tcgen05 收尾 + 逐 kernel 满 wave） | `l4-l5-kernel-path §1` | 在飞 warp 数 | **−5~8ms** | 16~21 人日 | **仓内零实测背书**（v17→v21 四变体全中性） | ❌ |
| **P7** | **Plan B（unanimity-or-direct）→ 拿回图** | `step_rows_sync` 的臂决策 | **只削 submit 半（已重叠）** | **−1.5ms** | 1~2 人日 | 未实施 | ✅ 它是「让图可用」的那一项 |

**排序理由（一句话）**：P1~P6 都是「**削 kernel 自己的执行**」——这是唯一能下台阶的列（`verify-ms-breakdown §3.2`）；
P7 只削已被隐藏的 submit，**收益 −1.5ms**，故排最后。**关键路径 = P1 → P3 → P4（+P2 并行）**，不含 P7。

### 4.1 一个必须先做的判读（否则 P1 会重蹈 mrows 零收益）

- **P1 的四个 gate 无 decline 日志**（`batched-400-v2-prediction §3.2`：只有 `VERIFY_HEAD_MROWS`/`ATTN_MROWS` 有
  one-shot note），⇒ **必须用 nsys 的 kernel 名判生效**：`gemm_fp8_mrows_kernel<M>` 的 `M` **就是 verify 行数**；
  若出现大量 `<1>` 实例，说明 mrows decline 回落 `M=1` 循环（`nsys-wave1 §2 注`）。
- **SWALLOW 无图下 `m` 恒 6** ⇒ `kernel<6>` 是判据；`kernel<5>` 只应出现在 bootstrap 轮。

### 4.2 与任务给的三个替代项对账

| 任务项 | 账本口径 | 一致性 |
|---|---|---|
| SH_PAIR template`<M>`：−5ms if parity fixed | −4.9~7.9ms（设计）**；当前 parity 36 failed = 0 收益** | ✅ 方向对；`−5ms` 是设计口径中值（`sh-pair-template-m-design §0` 估 −4.9~7.9） |
| tcgen05：−2ms if fixed | **−1.0~3.8ms（修正后，非 −6.8）** | ✅ 与「−2ms」吻合；**注意 down 无 tcgen05**，−2ms 只是 gate/up 半 |
| B 类核：−3ms | B1–B6 全族 −2.8~4.9ms（设计）；**B6 单项仅 −0.66~1.5ms** | ✅ `−3ms` 是**全族**口径，不是单项 |

⇒ **三项全兑现 ≈ −8~−16ms**。40 − (8~16) = **24~32ms**（见 §6）。

---

## 5. Plan B（unanimity-or-direct）的实施建议 + 优先级

### 5.1 ar5-hang 的根因（已确认）

`step_rows_sync` 的**四臂** `host_barrier` 计数不对称：**DRY=0、replay=1、capture=2、direct=0**
（`chain_dev.rs:5300-5304`）。臂决策的输入是 **per-RANK** 的（capture 拒绝 / `compress_branch_steady` 未收敛 /
形状槽被占），**TP8 的 8 个 rank 不保证选同一臂**；而 `SpinBarrier` 是**到打次数**的代计数器——少打一次 wait 的
rank 就关闭了 peers 从未进入的 epoch，此后每个 barrier 都错一代（**静默 misphase**）。
三次修复（守卫+DRY barrier=22 / 回退=3 / Plan A+C barrier 对称化+SWALLOW warmup=23）**都没修好**
（`dspark-correctness-chain` ar5 段）。**不用图实测 0 hang ⇒ 臂唯一 ⇒ 根因确认为「臂分歧」。**

### 5.2 Plan B 的设计（rank 同步的臂决策）

**做法**：在 `step_rows_sync` 的 arm 决策点之前，把「本 rank 想走哪一臂」做一次**cross-rank all-reduce
取一致**（unanimity：全 capture 才 capture；否则全 direct），让 **8 个 rank 永远进同一臂**。

- **代价**：多一次小 all-reduce（~17.3µs，`verify-ms-breakdown §1` AR 地板单价）或一次 host barrier。**可忽略。**
- **收益**：四臂的 barrier 计数天然一致 ⇒ 消除 misphase 类 hang。
- **风险**：① 引入的新 barrier **本身**要计入四臂计数（P 的 `verify_graph_want()` 条件得同步改）；
  ② unanimity 会让「一个 rank 不能 capture」拖累全组退 direct——**这是正确的保守语义**（图是优化，不是正确性）。
- **落点**：`chain_dev.rs::verify_graph_gate`（臂决策前）+ `step_rows_sync` 的臂分派（`base.rs` 同步原语）。

### 5.3 优先级判决：**MEDIUM，不是关键路径**

- **步时价值 ≈ −1.5ms**（图实测值，§2）。**不要把它排到 P1~P6 之前。**
- **但值得做**，理由有二（都不是步时）：
  1. **把图当「顺序/正确性工具」拿回来**——无图时 launch 顺序靠裸链依赖序保证，图 replay 能提供可复现的顺序
     （排查类工作的价值）。
  2. **为未来依赖图的优化清障**（draft 链图化 P3c 预期 −2ms、整步图），若这些未来项与 `step_rows_sync`
     共用屏障语义，Plan B 是前置。
- **止损门**：若 Plan B 后 hang 以新形态复现（换 gate 组合，尤其 `VERIFY_HEAD_MROWS`），
  记录为已知缺陷并**保留 SWALLOW 无图作为生产形态**（它已验证 0 hang）。

### 5.4 现状的最优配置（不做 Plan B 也能拿到的）

```
DSV41_LAZY_VERIFY=1      # 蕴含 swallow：row 0 = 被吞的主链步 ⇒ 拿到 −4.55ms
DSV41_VERIFY_GRAPH=1     # lazy 的 m=1 图：−1.5ms，且 lazy 路径无 ar5-hang（图槽 m=1）
DSV41_HC_VERIFY_FUSE=1 DSV41_HC_FRONT_ROWS=1
+ mrows 族（P1）
```
⇒ **accept 1.2 的任务走 lazy 路线（~22.5ms）**；**accept ≥3.55（计数任务）才切 batched**。
**「SWALLOW 常开 + lazy⇄batched 走路由」**才是正确形态（`lazy-batched-gate §2.3` 的 Schmitt 滞回已实现）。

> ⚠️ 这是一个**产品层面**的选择（会改按任务的 P50 分布），工部不自行拍板——提请尚书省/用户仲裁：
> 测试任务固定 counting 口径，还是按任务自适应路由。

---

## 6. 最终步时预测

### 6.1 任务问的那条：`SWALLOW + 图 + SH_PAIR + tcgen05 ≈ ?`

从**实测基线 ~40ms（无图 batched）**出发，逐项叠加（口径：实测优先，否则设计口径）：

| 阶段 | 增量 | 累计 | 依据强度 |
|---|---:|---:|---|
| **B0** SWALLOW 无图 batched（本次实测） | — | **~40ms** | 实测（serve）+ 内部口径 ~38-40 |
| **B1** + `VERIFY_GRAPH`（Plan B） | **−1.5** | **~38.5** | ✅ 实测（多次） |
| **B2** + SH_PAIR `template<M>`（m=6） | −4.9~7.9 | **~30.6~33.6** | ⚠️ 设计口径，**parity 36 failed** |
| **B3** + tcgen05（gate/up） | −1.0~3.8 | **~26.8~32.6** | ⚠️ 修正口径，**从未上 GPU** |
| **B4** + mrows 族（零代码） | −4.5~5.8 | **~21~28** | ❌ 设计口径，历史兑现率 ≈0（`SH_EXP` 两次零收益） |
| **B5** + hc A1/A2 | −1.3~1.7 | **~19.3~26.7** | ✅ 部分实测（Wave 1） |
| **B6** + B 类全族 | −2.8~4.9 | **~14.5~23.9** | ⚠️ 设计口径（B6 单项仅 −0.66~1.5） |

**诚实票面**（按历史 **60% 兑现率**折算，且把 mrows 按 ≈0 计）：
```
40 (实测)
 − 1.5  (图，实测)
 − 3.5  (SH_PAIR 60%)
 − 1.5  (tcgen05 60%)
 − 0.9  (hc)
 − 2.0  (B 类 60%)
────────────────────
≈ 30.6ms   ⇒ SWALLOW + 图 + SH_PAIR + tcgen05 的**现实落点 ~30~33ms**
```
**乐观端（全部足额兑现）**：~19~24ms。**任务预期的「25-30ms」需要 SH_PAIR + tcgen05 + mrows 三项同时足额——**
**而这在仓史上前所未有**（`swallow-unlocked-next-plan §2`）。

### 6.2 关键提醒：这条 line 走完，**仍然追不上 lazy**

| 配置 | 步时 | 说明 |
|---|---:|---|
| **SWALLOW + 图 + SH_PAIR + tcgen05**（本任务问的） | **~30ms**（现实）/~24ms（乐观） | batched m=6，恒 6 行 |
| **lazy + 图 + 同样的融合** | **~15~19ms** | lazy 只跑 ~2 行；tcgen05/B/hc 对它同样生效 |
| **lazy + 图**（现状，不做任何新融合） | **~22.5ms** | 已经优于 SWALLOW 全融合的现实端 |

⇒ **在 accept 1.2 下，把资源投给「SWALLOW+batched 的融合」不如投给「lazy 的融合」**——
因为同一份融合，lazy 的乘数是 `k_emit≈2.2`，batched 的乘数是 `6`。
⇒ **batched（含 SWALLOW+图）只在 accept ≥3.55 时才该被投入**（`lazy-batched-gate §0.7`）。
   对计数类高 accept 任务：SWALLOW+全融合 ≈ 6 tok/step ÷ 30ms = **200 tok/s**（现实）/~250（乐观）。

### 6.3 一张决策表（交付用）

| accept（任务） | 正确臂 | 不用图的优化重点 | 步时落点 |
|---|---|---|---|
| **≤1.2**（对话 0.96 / 出师表 1.21） | **lazy**（implied swallow） | tcgen05 + B 类 + hc（**同乘数小于 batched**） | ~15~19ms |
| **1.2~3.5**（中等熵/代码） | **lazy**（滞回窗口） | 同上 + 攻 accept | ~19~24ms |
| **≥3.55**（计数 5.0） | **batched（SWALLOW）** | SH_PAIR + tcgen05 + mrows + B 类 | ~24~30ms |
| **任一**，要拿回图 | + **Plan B** | — | **−1.5ms** |

---

## 7. 一句话交付

> **~40ms 不是「没有图」造成的——图实测只值 −1.5ms；40ms 是 batched m=6 verify 自身的 kernel 成本
> （shared+routed ≈ 22.5ms = 56%）。** 15ms 的差距来自 **batched 恒 6 行 vs lazy ~2.2 行**，
> 在这套 tiny-SIMT-kernel 架构下 batched 结构上就慢（`arch-floor §5.3`）。
> **不用图的优化 = 削 kernel 执行**（mrows 族 → hc → SH_PAIR → tcgen05 → B 类），**顺序不能反**；
> **Plan B 只值 1.5ms、不是关键路径**，它的定位是「把图当正确性工具拿回来」。
> **且 SWALLOW 的 −4.55ms 吞并收益早已能通过 `LAZY_VERIFY`（蕴含 swallow）拿到**——
> **在 accept 1.2 的任务上，正解是 lazy 路线（~22.5ms），不是把 batched SWALLOW 硬推到 40→30ms。**
> **SWALLOW + 图 + SH_PAIR + tcgen05 的现实落点 ≈ 30~33ms（乐观 24ms），仍慢于 lazy+同融合的 ~15~19ms。**

---

*工部 · 只读分析 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*所有 ms 数均标注来源与口径（实测 / launch 账 / ms 账 / 设计口径）；与任务前提冲突处
（「图省 15ms」「SWALLOW 应比 lazy 快」「Plan B 是关键路径」）已显式修正并给出依据。*
