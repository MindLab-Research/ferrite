# 从 22.56ms 到 ≤15ms 的最快实施路径（400 实施优先级路线图）

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 代码基线：HEAD `c13dabd`（逐条 `file:line` 现场核对）。
> 输入：`lazy-verify-95ms-gap-analysis` · `lazy-verify-optimization-path` · `l4-occupancy-mlp-design` ·
> `l4-l5-kernel-path` · `l5-draft-slice-ab-plan` · `tcgen05-smoke-redesign` · `swallow-unlocked-next-plan` ·
> `final-400-sprint-roadmap` · `verify-architecture-floor` · `dspark-correctness-chain`。
> **本机无 GPU ⇒ 所有 ms 标了来源（实测 / 账本推算 / 设计口径）。**

---

## 0. 判决（六条）

1. **实施顺序已确定**：`Step0 口径 → Step1 零代码 A/B → Step2 落默认值 → Step3 tcgen05 → Step4 L4`（§6）。
2. **⚠️「426 tok/s @ accept 5」必须先钉死口径**（§1）：22.56ms 是 **accept≈1.1** 的 lazy 步时，不是
   accept-5 的步时；`6/0.02256` 隐含「步时与 accept 无关」，而 **lazy 步时 = k_emit × c_row + draft + commit**，
   `k_emit = 1 + mean_k` ⇒ accept 5 要跑 **6 行**。这与「33ms」是**同一类口径混用**（方向相反）。
3. **零代码可立即兑现 = L1/L2/L3/L5**（+L6 仅长任务），合计 **−5.4~7.0ms**，一次 GPU 会话（多臂交错）可拿。
4. **≤15ms 在 Step3（tcgen05）达成，不需要 L4**（L4 = 16~21 人日、仓内零实测背书，**可延后**；
   L4 是「中 accept 任务」与「400 余量」的门）。
5. **L1~L6 全部是默认 OFF 的 env gate**。「已验证 / 已提交」≠「默认开」——Step2 必须显式落默认值
   （含 `HC_VERIFY_FUSE` 的反向默认修复），否则 400 配置只活在脚本里。
6. **GPU 测试串行**（同一远端单驱动）；**代码可并行**（tcgen05/L4 kernel 可在 GPU A/B 期间写）。

---

## 1. 口径校准（Step 0，必做，先于一切）

### 1.1 两个数各自是什么（关键修正）

| 数 | 含义 | 来源 |
|---|---|---|
| **22.56ms** | lazy m=1 步时 **@ accept≈1.08~1.214**（k_emit=2.214）= verify 18.11 + draft 4.28 + commit 0.17 | `lazy-verify-optimization-path §1.1`（✅实测）|
| **~33ms** | 计数任务（k_acc≈5）的**生成阶段**步时（serve 墙钟）| `dspark-correctness-chain`（✅实测）|

⇒ **22.56 与 33 不是同一 (任务, 臂) 的数**。前者是出师表口径（accept 1.1）的内部步时，后者是计数口径
（accept 5）的 serve 墙钟。把 22.56 当「真值」、把 33 一律当「含 prefill 的错值」，本身是一次口径混用。

### 1.2 为什么 `6/0.02256` 不成立

```
lazy 步时 = k_emit × c_row + draft + commit     k_emit = 1 + mean_k     c_row = 8.18ms
  accept 1.214 ⇒ k_emit 2.214 ⇒ 22.56ms   ✓
  accept 5     ⇒ k_emit 6     ⇒ 6×8.18 + 4.45 ≈ 53.5ms   （≈112 tok/s）
```

⇒ `6/0.02256 = 266` 把「accept-1.1 的步时」配上了「accept-5 的 tok/step」。这与「33ms」同类。

### 1.3 两个可能的正确基线（由 Step 0 裁决）

- **若 accept-5 走 lazy**（k_emit=6）：基线 ≈ **53ms**。
- **若 accept-5 走 batched**（SWALLOW，恒 6 行；路由阈值 `τ = B/c_row − 1 ≈ 3.5`，`mean_k=5 > τ`
  ⇒ 自动选 batched）：基线 ≈ **33~39ms**，且**与 accept 无关**。

**Step 0 = 同 binary、同 prompt、背靠背跑出 accept-5 的 `[dspark] verify=/draft=/commit=` + `[dspark] route=`
+ k_emit 分布**，把分母钉死。成本 0.5 人日、0 代码。

> ⚠️ **不做 Step 0 的后果**：阶梯每一格都会在此处重复记账——若 accept-5 实为 batched，§6 的懒系阶梯
> （L1/L3/L5 那一列）**整列不适用**，须换成 batched 阶梯
> （`SWALLOW −4.55 + mrows 族 −4.5~5.8 + SH_PAIR M≥2 −4.9~7.9 + tcgen05 + B6`，
> 见 `swallow-unlocked-next-plan §2/§6`）。

---

## 2. 立即 A/B（零代码）—— Step 1

### 2.1 清单（全部 env gate、全部默认 OFF；已逐条核对 HEAD）

| 项 | gate | 默认 | 位置 | 预期（lazy 口径）| 备注 |
|---|---|---|---|---|---|
| **L1** hc 融合 | `HC_VERIFY_FUSE=1` + `HC_FRONT_ROWS=1` + `VERIFY_AR_FOLD=1` | OFF（`HC_VERIFY_FUSE` 是**反向默认** `v=="1"`）| `chain_dev.rs:12378 / 12432 / 12412` | **−2.9~4.2ms**（×k_emit）| A2 的 `bf16_truncate` 坑**已在树里修复**（`layer_rows` 两处显式传 `false`，`:9106 / :9178`）|
| **L2** sync 收敛 | `DSV41_LAZY_SDR=1` | OFF | `chain_dev.rs:2358` | −0.7ms | 代码已提交（`69487d7`）：`memcpy_d2d_2d` + tap-commit 合并 + `set_pos_ctr` H2D 合并 |
| **L3** SH_PAIR M=1 | `DSV41_SH_PAIR_M=1` | OFF | `chain_dev.rs:1392` | −1.1~1.3ms | GPU 已验证正确（零拉丁 + k_acc 逐位相同，`57090747`）|
| **L5** draft | `DSV41_DRAFT_P3A=1` + `DSV41_MARKOV_SLICED=1` | OFF | `dspark_dev.rs:362 / 492` | −0.7~0.8ms | 诚实值：MARKOV 单独 **−0.3~0.5**（5 轮 v5 往返吃掉一半）；P3A −0.3 |
| **L6** draft 图化 | `DSV41_DRAFT_GRAPH=1` | OFF | `dspark_dev.rs:225` | 长任务 −3.3ms；**短任务 ≈0** | 需 `pos≥win`；计数任务（144 tok）覆盖率 ~17% ⇒ ≈0 |

### 2.2 组合 A/B 矩阵（回答任务 2）

**姿态**：一臂一进程（`OnceLock` 每进程读一次）；**交错 A B A B** 抵消时钟/热漂。建议**一次 GPU 会话**跑
5~6 臂：

```bash
COMMON="DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 \
        DSV41_EXPERT_ACT_E4M3=1 DSV41_TIMING=1 DSV41_DSPARK_DEBUG=1 \
        DSV41_LAZY_VERIFY=1 DSV41_VERIFY_GRAPH=1 DSV41_BF16_TRUNCATE=1"

A(base) = COMMON
B       = COMMON + DSV41_LAZY_SDR=1                                        # L2
C       = COMMON + DSV41_SH_PAIR_M=1                                       # L3
D       = COMMON + DSV41_DRAFT_P3A=1 DSV41_MARKOV_SLICED=1                 # L5
E       = COMMON + DSV41_HC_VERIFY_FUSE=1 DSV41_HC_FRONT_ROWS=1 DSV41_VERIFY_AR_FOLD=1   # L1
F(opt)  = COMMON + DSV41_DRAFT_GRAPH=1                                     # L6（长任务才测）
```

**判据（每臂必录，缺一不能下结论）**：
1. `/proc/<pid>/environ | grep DSV41_` **逐门读回**（本项目 #1 陷阱：gate 设了没生效）；
2. `nm -D $SO` 符号（L3 需 `dsv41_gemm_fp8_sh_pair`；L5 需 `dsv41_dspark_markov_head_sliced` +
   `dsv41_argmax_key_pub`；L1 需 `ferrite_p2p_ar_v5_hcpost_rows`）；
3. `[dspark] steps=` 的 `verify=/draft=/commit=` 中位位移；
4. 四段文本逐字 + **零拉丁** + `faults=0` + **k_acc 逐位不变**（出师表序列 `4 0 0 0 3 0 1 1 0 …`）；
5. 一次 nsys 按 **kernel 名**数折核（不只看 `verify=`）。

**为什么一次会话多臂**：GPU 位唯一（单驱动），拆多次会话只是把串行变成更长的串行。**但归因要求单变量**——
交错 + 每次只加一个 gate 是必须的；**不要在同一轮叠 gate 找感觉**。

**预期合计**：`L1 −2.9~4.2 + L2 −0.7 + L3 −1.1~1.3 + L5 −0.7~0.8 = −5.4~7.0ms`（lazy 口径）
⇒ 22.56 → **15.6~17.2ms**。

---

## 3. tcgen05 的正确重测（回答任务 3）

**前提修正**：上一轮「0 misaligned」是**空洞的**——全链路只有 decline 告警，「成功」没有正证据；且**一个 gate
同时武装两个未验证 kernel**（prefill 的 swapAB `tc5::e4` + verify 的 grouped `tc5::e4x`），GPU 首触是
swapAB。重测必须「**先造正证据、再按 kernel 隔离、再单 GPU 先行**」。

```
Phase 0（单 GPU，无 serve、无 TP）
  0a  dsv41-run --tp 1 + swapAB arm gates           → 证 swapAB（单行，变量最少）
      证据：exit 0 + 文本可读 + launch 证据（nsys 里 e4m3_gemm_kernel）
  0b  新增 tests/real_grouped_tcgen05.rs             → 证 grouped（唯一干净隔离）
      Device + Loader(world=1, rank=0) + DevChain + reset + step_rows(6 tokens)
      证据：Ok + nsys/日志出现 e4m3_gemm_grouped_kernel + 无 expert_grouped_skipped_note
Phase 1（TP8 serve，仅当 0a + 0b 都 PASS）
  R0  基线 env（无 tcgen05 门）                      → 引擎自证；失败则本轮作废
  R1  GATEUP_FUSE=0 ILV=0（无 tcgen05 门）           → 门税 / plain 布局自证（f15ecd37 的 misaligned 属这档）
  R2  R1 + EXPERT_TCGEN05_E4M3=1 EXPERT_GROUPED=1    → 目标
      证据：nsys 同时看到 e4m3_gemm_kernel(prefill) 与 e4m3_gemm_grouped_kernel(verify)
首轮一律 CUDA_LAUNCH_BLOCKING=1（+ compute-sanitizer memcheck）；
⏱ 计时与 profiling 不要同轮（nsys/sanitizer 会污染 ms 读数）。
```

- **正证据三条路**：**A = nsys**（金标准，零改动）；**B = env-gated launch 打印**（~10 行 `.cu`，
  `DSV41_TCGEN05_TRACE=1`，属源码改动，**需尚书省批准**）；**C = device 计数**（改 ABI，慎用）。
- **5-gate 链**：`EXPERT_ACT_E4M3=1 EXPERT_TCGEN05_E4M3=1 EXPERT_GROUPED=1 GATEUP_FUSE=0 EXPERT_ILV=0`。
  ⚠️ 门读语义：`EXPERT_TCGEN05_E4M3` / `EXPERT_GROUPED` **严格 `starts_with('1')`**；`GATEUP_FUSE=0`
  单独就蕴含 `ilv=false`。
- **预期** −1.9~2.1ms（**只 gate/up**；**down 无 tcgen05 核**，3.48ms 原样保留）。
- **止损门**：单层微基准 **gateup 22.2µs / down 17.2µs** 不达标 ⇒ 立即关闭路径，**不投变体矩阵**
  （勿重演 v17→v21 四变体全中性）。

---

## 4. L4 占用的 kernel ROI（回答任务 4）

按 `l4-occupancy-mlp-design §4.1` 的 ROI（ms 人日）排序：

| 排名 | 项 | kernel 落点 | 中位节省 | 人日 | **ROI** | 把握 |
|---|---|---|---|---:|---:|---|
| **1** | **L4-7** hc 侧流 + dots 网格 | `hc_mixes_auto` / `hc_dots_late` / `hc_front_split` | 1.5 | 0.5~1 | **1.5~3.0** | 中高（A1/A2 代码全在树；**×k_emit**）|
| 2 | L4-4 tcgen05 down 换核 | **新** `tc5::down::*` | 2.25 | 3~4 | 0.56~0.75 | 中低（从零写）|
| 3 | L4-3 tcgen05 gate/up K-split | `tc5::mxf4::*`（第三 grid 维 + 升序 reduce）| 2.0 | 3~4 | 0.5~0.67 | 中 |
| 4 | L4-8 hc_dots KCHUNK + P2/P3 | `hc_dots_late_kernel` | 0.65 | 1.5~2 | 0.33~0.43 | 中 |
| 5 | L4-1 mrows nwarps/crossover | `gemm_fp8_mrows_kernel<M>` | 0.4 | 1 | 0.4 | 中低（**L3 后趋近 0**）|
| 6 | L4-6 verify 多流发射 | `layer_rows`（无新 kernel）| 1.0 | 3~4 | 0.25~0.33 | 中 |
| 7 | L4-2 SH_PAIR phase-1 K-split | `gemm_fp8_sh_exp_pair_kernel<M>` | 0.55 | 2 | 0.28 | 低 |
| 8 | L4-9 collapse_norm/rmsnorm 摊开 | `dsv41_hc_collapse_norm_kernel` / `rmsnorm_rows` | 0.2 | 1.5~2 | 0.10~0.13 | 低 |
| 9 | L4-5 e4x M=128 tile | `tc5::e4x::*` | 0.5 | 2~3 | 0.17~0.25 | 低（**batched only**）|

**四条关键判词**：
1. **L4-3 + L4-4 = L4 收益的 ~50~60%**（tcgen05 收尾）——L4 的成败在 **routed experts**，不在其余七项
   （其余加起来只有 −2~4ms）。
2. **L4-7 是碾压性 ROI**（0 代码 + ×k_emit），但**它与 L1 的 A1/A2 是同一批接线**——**两者不得相加**
   （合并时取 L1 的数，L4 只留侧流增量 ≈ −0.6ms）。
3. **L4-1 与 L3 互斥**：shared expert 的 n=288 被 SH_PAIR 接管后，L4-1 只剩 wkv（需调
   `kMrowsSmallN2: 512→640`），收益趋近 0。**不要按「−0.6 × 2」编预算。**
4. **全部是设计口径、仓内零实测背书**；反向证据（`{SH_EXP,GRAPH,ROPE,P3A}` 全开只 −1.21ms、
   v17→v21 四变体全中性）要求**先用 Step3 的实测决定 L4-5 是否值得投**。

---

## 5. 并行性（回答任务 5）

| 资源 | 可否并行 | 说明 |
|---|---|---|
| **GPU 测试** | ❌ **串行** | 同一远端同时只能一个测试驱动（build-id mismatch 事故成因）；Step1 的多臂必须背靠背交错 |
| **代码 / 分析** | ✅ 并行 | Step3 的 tcgen05 FFI/parity、Step4 的 L4-3/L4-4 kernel、B6（`dsv41_gemm_fp8_mrows_f32`）可在 Step1 的 GPU 会话期间写 |
| **accept 支线** | ✅ 并行 | oracle tap 对照 / `SEED_ALIGN` A/B / draft e4m3 ——不占关键路径（但要排队等 GPU 位）|
| **关键路径** | — | `Step0 → Step1 → Step2 → Step3 → Step4` |

---

## 6. 阶梯与最快路径（交付物）

> **口径**：下表按 §1 的 lazy 口径（accept≈1.1 内部步时）；**@ accept 5 的步时须由 Step 0 钉死**（§1.3）。

| 阶段 | 内容 | 增量 | 累计步时 | 实施成本 | 依赖 |
|---|---|---|---:|---|---|
| **S0** | 当前（lazy m=1 + 图）| — | **22.56ms** | — | — |
| **S0.5** | **口径钉死**（accept-5 内部步时 + route + k_emit）| 0 | — | **0 代码 / 0.5 pd** | — |
| **S1** | + **L1** hc 融合（×k_emit）| −2.9~4.2 | 18.4~19.7 | 0 代码 / 0.5~1 pd | S0.5 |
| **S2** | + **L3** SH_PAIR M=1 | −1.1~1.3 | 17.1~18.6 | 0 代码 / 0.5 pd | S1 |
| **S3** | + **L5** draft P3A+MARKOV | −0.7~0.8 | 16.3~17.9 | 0 代码 / 0.5 pd | — |
| **S4** | + **L2** LAZY_SDR | −0.7 | **15.6~17.2** | 0 代码 / 0.5 pd | — |
| **S5** | + **L6** DRAFT_GRAPH（**仅长任务**）| 0（短）/ −3.3（长）| 15.6~17.2 / 12.3~13.9 | 0 代码 / 1 pd | `pos≥win` |
| **S6** | **+ tcgen05 gate/up** | −1.9~2.1 | **13.5~15.3 ✅** | **4~5 pd** | S1~S4 |
| **S7** | + L4 占用/MLP | −5~8 | 8.5~10.3 | **16~21 pd** | S6 |
| **S8** | 落默认值（S1~S6 全部转 ON）| 0 | — | 0.5 pd | S6 全绿 |

### 6.1 最快实施路径（一句话）

> **S0.5（口径，0.5 pd）→ S1+S2+S3+S4（零代码，一次 GPU 会话，0.5~1 pd）→ S8（落默认值）→ S6（tcgen05，4~5 pd）
> ⇒ 13.5~15.3ms，命中 ≤15ms。**
> **L4（16~21 pd）不在关键路径上**：它的价值是（a）给 400 余量，（b）把「中 accept（2-3）」也拉进 400。
> **最短可行集 = 零代码栈 + tcgen05 ≈ 5~6 人日 + 1~2 GPU 会话。**

### 6.2 两条分支

- **若 Step 0 判定 accept-5 走 batched**（阈值 τ≈3.5，`mean_k=5 > τ`，**概率较高**）：
  把 S1~S4 换成 **batched 阶梯**（`SWALLOW −4.55` + `mrows 族 −4.5~5.8` + `SH_PAIR M≥2 −4.9~7.9` +
  `tcgen05 −1~3.8` + `B6 −0.66~1.5`）——即 `swallow-unlocked-next-plan §2/§6` 的 S1~S5；
  其余（S6/S7/S8）不变。
- **若判定 accept-5 走 lazy**：沿用上表。

---

## 7. 依赖图

```
S0.5 口径
   ├─→ S1(L1 hc) ─→ S2(L3 SH_PAIR) ─┐
   │    S3(L5 draft) ───────────────┼─→ S6(tcgen05) ─→ S7(L4) ─→ 400 余量
   │    S4(L2 SDR) ─────────────────┘        │
   └─→ S5(L6 draft-graph, 长任务)             └─→ S8 落默认值
并行支线（不占关键路径）：accept（oracle / SEED_ALIGN / draft e4m3）· B6 · L4-7
```

---

## 8. 风险与止损

| 风险 | 触发 | 止损 |
|---|---|---|
| **口径混用**（本文件 §1）| accept-5 步时从未实测 | **必做 S0.5**；不做则阶梯可能整列作废 |
| **gate 设了没生效**（本项目 #1 陷阱）| `verify=` 无位移 | 每门读回 `/proc/environ` + `nm -D` + nsys 数 kernel 名 |
| **tcgen05 空洞重测** | 「0 misaligned」无 launch 证据 | 换成「launch 计数 > 0 且无 fault」；不达标即止损 |
| **mrows/SH_PAIR 兑现率 0**（先例：SH_EXP 两次零收益）| 单 gate A/B 位移 < 预期 40% | 停该项、转下一个；**不在同一轮叠 gate** |
| **L4 投入陷阱** | L4-1 单独 A/B 中性 | 不开变体矩阵，除非 S6 已给出 routed 实测新地板 |
| **零拉丁回归**（用户硬要求）| 四段文本出现拉丁 | 回退该 commit；**单 gate 单 commit + 读回** |

---

*工部 · 只读分析 + 本文件（唯一产出）；未执行任何 GPU 命令、未改动任何源码。*
*所有 ms 标来源（实测 / 账本推算 / 设计口径）；与任务前提冲突处（「426 @ accept 5」的口径）已在 §1 显式修正并给出依据。*
