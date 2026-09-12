# SWALLOW 解锁后的下一步（行动计划 + 步时阶梯）

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 输入（现场核对）：`chain_dev.rs`（gate 定义 + 分派点）· `dsv41_kernels.cu`（SH_PAIR 核 + launcher）·
> `tests_dsv41_sh_exp_mrows.cu`（parity 套件本体）· 账本 `nsys-wave1-analysis-framework` /
> `verify-architecture-floor` / `verify-ms-breakdown` / `lazy-batched-gate` / `sh-pair-template-m-design` /
> `tcgen05-e4m3-grouped-expectation` / `b6-mrows-f32-design` / `l4-l5-kernel-path` / `dspark-correctness-chain`。
> 在跑的测试：**SWALLOW_STEP + ar5 Plan A+C**（0 ar5-hang）。

---

## 0. 四条必须先钉死的前提（不钉死，下面的阶梯整体偏 5~16ms）

**C1 — 三个计时器在混用，必须先统一。**
| 计时器 | 值 | 可信度 |
|---|---|---|
| `[dsv41] step pos=N: Y ms`（serve 墙钟） | 25.17ms（Wave 1 后） | ❌ 用户已两次判为不可信（curl/HTTP/admission/tail 摊到尾部步） |
| `[dspark] steps= … verify=X` | ~31-33ms（Wave 1 后推定） | ⚠️ 半真（含 host barrier + D2H sync） |
| nsys per-kernel GPU 时间 | 37.31ms（**Wave 1 前**，m=5） | ✅ 唯一纯模型执行时间 |

⇒ **「Wave 1 = −8.4ms」这句话目前没有任何一个可信计时器背书**：−8.4ms 的实测落点
（33ms → 25.17ms）两个端点都是 serve 墙钟。阶梯的第一项就是把这个数重新钉一次（§5 P0-b）。

**C2 — 阶梯在混臂。** 37.31ms 是 **batched m=5** 口径；Wave 1 实测跑的是 `DSV41_LAZY_VERIFY=1`
（**逐行 m=1**）。把两者的差当作 Wave 1 的收益是错的：lazy 的 m=1 verify 本身就比 batched 便宜
（accept 低时只跑 2 行），差额里混着「换了臂」和「换了计时器」两个变量。**每一条阶梯数都必须标注臂 + m。**

**C3 — mrows 族在 m=1 下恒为 0，这是本轮最大的隐藏账。**
`VERIFY_ROPE_MROWS` / `VERIFY_HEAD_MROWS` / `GATE_MROWS` / `INDEXER_MROWS` 全部是
**「m 行折成 1 发」**的折核（`chain_dev.rs:1131/1542` 的 gate + `row_fold_gate()` `:1225`）。
lazy 每行恒 m=1 ⇒ 折核退化成自身 ⇒ **零节省**。旁证有两条：`{SH_EXP+GRAPH+ROPE+P3A}` 全开实测仅
**−1.21ms**（预期 −24），`SH_EXP_MROWS` 两次实测零收益（`verify-ms-breakdown:159`）。
⇒ **Wave 1 的 −8.4ms 几乎全部来自 HC A1/A2 + AR fold（真 fusion），mrows 那一半并未兑现**
（它在 lazy 下不可能兑现）。

**C4 — `SH_PAIR_M` 的 `template<M>` launcher 是按 m 实例化的**（`dsv41_kernels.cu:7299-7343`，
`gemm_fp8_sh_exp_pair_kernel<k>` k=m），所以它**同样只在 m>1 时才有意义**——SH_PAIR 与 SWALLOW
共享「必须有 m 行块」这个前提。

---

## 1. 核心结论：SWALLOW 的价值不是 −4.55ms，是 −4.55ms **加上一把钥匙**

SWALLOW 把 m 从 5 推到 **6**，并让 **verify 成为唯一的主链**。这把两个此前恒为 0 的账户同时打开：

```
SWALLOW 自身的账          −4.55ms   （主链吞进 verify；含 anchor 行 +1.6ms 的代价后净值）
+ 它解锁的 m=6 mrows 族   −4.5 ~ −5.8ms  ← 目前一分未兑现，且是零代码工作（现有 gate）
  ├─ GATE_MROWS           −2.75ms（设计口径，未单测；Wave 1 里已设但被 lazy 抹平）
  ├─ INDEXER_MROWS(front) −1.0 ~ −1.5ms（代码已就位）
  ├─ VERIFY_ROPE_MROWS    −0.53ms（launch 账）
  └─ VERIFY_HEAD_MROWS    −0.7 ~ −0.9ms（⚠️ 正是历史 ar5-hang 的那个组合，必须最后单独验证）
────────────────────────────────────────────────────────
SWALLOW 的真实杠杆        −9 ~ −10.4ms
```

⇒ **优先级判决：SWALLOW 的验证 > SH_PAIR 的 parity 修复。**
SH_PAIR 是 ms 账最大单项（−4.9~7.9ms）但要 2~4 人日的 kernel 修复 + parity 硬门；
SWALLOW 之后的 mrows 族是 **−4.5~5.8ms / 零代码 / 一次 A/B**。先拿零成本的。

---

## 2. 步时阶梯（口径统一后的账）

以 **Wave 1 后内部计时 31-33ms**（C1/C2 校正后的工作基线）为起点：

| # | 阶段 | 增量 | 累计步时 | 依据强度 |
|---|---|---|---:|---|
| **S0** | Wave 1（现状，内部口径） | — | **31-33ms** | ⚠️ 需 P0-b 重钉 |
| **S1** | + SWALLOW（+ar5 修复） | −4.55ms | **26.5-28.5ms** | 设计口径，待本次测试 |
| **S2** | + m=6 mrows 族 | −4.5~5.8ms | **21-24ms** | 设计口径（其中 −1.21ms 的族级旁证偏负面） |
| **S3** | + SH_PAIR `template<M>` | −4.9~7.9ms | **13-19ms** | 设计口径；**当前 parity 36 failed = 0** |
| **S4** | + tcgen05（gate/up only） | −1.0~3.8ms | **9.2-18ms** | 修正口径（**非 −6.8ms**：down 无 tcgen05 核，只剩一半） |
| **S5** | + B6（B 类公共祖先） | −0.66~1.5ms | **7.7-17.3ms** | 第一性计数（−200~240 发/步）；**B6 单项不是 −2.8~4.9ms**（那是 B1–B6 全族） |
| **S6** | + L4（占用/MLP） | −5~8ms | **~4-13ms** | 设计口径，**仓内零实测背书**，16.5~21.5 人日 |

**诚实读数**：S3 的 −4.9~7.9ms 与 S2 的 −4.5~5.8ms 都是**设计口径的设计口径**（前者来自
shared expert 10.4ms 的族账，后者来自折核件数 × per-launch 价）。
仓库的反向证据权重不可忽略：**7 次「隔离有效 → serve 失效」**、`SH_EXP_MROWS` 两次零收益、
`{四件套}` −1.21 vs 预期 −24、v17→v21 四变体全中性。**按 60% 兑现率折算**：

| 兑现率 | S2 | S3 | S4 | S5 | 落点 |
|---|---|---|---|---|---|
| 100%（设计口径） | 22 | 15 | 13.5 | 12.5 | 400 ✓（counting） |
| **60%（历史兑现率）** | 24 | 20 | 18.5 | 17.8 | **337 tok/s**（counting，差 16%） |
| 40% | 25.5 | 23 | 22 | 21.5 | 279 tok/s |

⇒ **SWALLOW 解锁后的现实票面是「步时 ~18-20ms、counting 任务 300-340 tok/s」**，
400 需要 S2/S3/S4/S5 **四项同时足额兑现**——这在仓史上前所未有（从没有任何一轮把设计口径全兑现）。

---

## 3. Q5：accept 兼容性（batched m=6 vs lazy m=1）

### 3.1 k_acc 必须逐位可比（这是验证判据，不是观察项）
- swallow 块 = `[anchor, d1..d5]` @ `pos..pos+5`（`chain_dev.rs:6284`）；lazy 块 = 逐行 @ `pos..`。
  `anchor` = **上一轮已提交的 token**，它必须被接受；若 anchor 被拒 = **状态机 bug 信号**（不是 accept 变化）。
- `spec_accept(anchor_is_in_block=true)`（`spec_step.rs:91`）+ `carry_kept_tap(keep=k_emit)`（`:7995` 附近）
  是两条臂共用的语义核心（`lazy-batched-gate §0.1`）⇒ **同一 prompt 下 k_acc 序列应逐元素相同**。
  判据：拿 Wave 1 的 lazy 序列 `4 0 0 0 3 0 1 1 0 0 0 1 2 0 0 5 0 0 0 1`（出师表）逐字对。
  ⚠️ 这条只在**同 prompt 同 seed** 下成立；换 prompt 比 k_acc 是无效判据。

### 3.2 head/rope mrows 在 batched 下的额外收益（任务问的那一项）
| 折核 | lazy(m=1) | batched(m=6) | 差额来源 |
|---|---|---|---|
| `VERIFY_HEAD_MROWS` | **0**（每行已是单发） | −0.7~0.9ms（6 行 → 1 发） | 见 `chain_dev.rs:5935` 的 sliced-head fold |
| `VERIFY_ROPE_MROWS` | **0** | −0.53ms（6 行 → 1 发） | `:1116` 的 q-rope fold |
| `GATE_MROWS` | **0** | −2.75ms | `gemv_bf16_nt_kernel` per-row 累加器契约 |
| `INDEXER_MROWS` | **0** | −1.0~1.5ms | `indexer_front_rows` 折进 `indexer_rows_one` |

⇒ 这四项**只在 batched 下存在**，是 S1→S2 那一格的构成，**不得重复计入 Wave 1**（Wave 1 是 lazy）。

### 3.3 低 accept 任务上 batched 是净亏（产品层面的分叉，需要仲裁）
`lazy iff (1+mean_k) < B/c`（`lazy-batched-gate §0.7`）。取 `B≈28ms`（SWALLOW 后）、`c=6.15`：
阈值 `mean_k < 28/6.15 − 1 = 3.55`。实测 accept：**对话 0.96 / 出师表 1.214 / 计数 5.0**。
⇒ **对话与出师表在 batched 下是净亏**（batched 28ms vs lazy 2 行 ≈12.3ms）。
⇒ **SWALLOW 不能当全局默认**；正确形态是 §1 的「SWALLOW 常开 + lazy⇄batched 走路由」
（`DSV41_LAZY_VERIFY` 已有路由设计，`lazy-batched-gate §2.3` 带 Schmitt 滞回）。
⚠️ 这是**设计决策**（会改变按任务的 P50 分布），工部不自行拍板——**提请尚书省/用户仲裁**：
是「按任务自适应」还是「测试任务固定 counting 口径」。

### 3.4 计数任务踩在阈值线上（一个必须知道的细节）
计数任务 `mean_k = 5.0`，阈值 3.55 ⇒ 稳定 batched ✓。但**出师表 1.21 与对话 0.96 稳定 lazy**，
所以**同一场测试里两臂会同时活跃**（`VERIFY_GRAPH_SLOTS = 2` 恰好两槽，不互驱逐 ✓）。
⇒ 判读时不能假设「整场只有一个臂」，必须按请求分桶（`[verify_graph] captured …_m1 / …_m6` 两行都可能出现）。

---

## 4. SH_PAIR parity 的定位（36 failed 的最短路径）

**先做一件事：把失败清单的完整 case 表打出来（5 分钟，零 GPU 成本）。**
`tests_dsv41_sh_exp_mrows.cu:318-386` 的判据分五支：(a) phase-1 aq 逐字节 (b) phase-1 scale 逐位
(c) phase-2 out（`epi_add` 期望在 host 侧施加）(d) 副产品 act (e) 覆盖度（`0x5A` / NaN sentinel）。
**已报的失败只覆盖 (a) 与 (d)**，且报的是 36 项里的 4 条——**先补齐分类**，否则修的可能是错的支。

### 4.1 已报失败的两个指纹（都是确定性的，不是数值噪声）
| 指纹 | 原文 | 读法 |
|---|---|---|
| **F1：只差符号位** | `aq byte diff at r=0 c=0: m-row 0xf9 m=1 0x79`（另例 `0x00` vs `0x80`） | 0xf9^0x79 = 0x80，0x00^0x80 = 0x80 ⇒ **差异恰好只有符号位**。这**不可能**是 amax 归约顺序造成的（归约序改不了符号，只会改尾数/scale）。所以先前记录的「amax 树归约顺序」假设**很可能是错的**。 |
| **F2：整缓冲 sentinel** | `act == nullptr but 512 slot(s) look unwritten`；`[fold/range] m=8 n1=64` ⇒ m·n1 = **512 = 全缓冲** | **整个 act 缓冲一个字节没写** ⇒ 形态是「kernel 对这条形状 decline / 没跑」，**不是**「行分布漏写」。而同一 tag 的 aq 却写了（F1）⇒ 说明 arm 跑了但**跳过了 act 写回分支**，两条指纹同源。另例 `480` 需回 case 表核 m·n1 是否恰好相等。 |

### 4.2 最短诊断树（每步都能否证一个分支）
1. **补 case 表**：把 36 项按 `(fold_r, n1, k1, n2, limit, epi_add, with_act)` 分类。
   判据：**`fold_r > 1` 是不是必要条件？`n2 % 32 != 0` 是不是必要条件？** 两个条件都出现 ⇒ 两个独立 bug。
2. **F1 的判别实验（零成本）**：对同一条 case 打印 **aq 前 8 字节两臂 + `aqsc` r=0**。
   - `aqsc` 逐位相等 + aq 只差符号位 ⇒ **量化器的符号路径**（不是 scale/amax）；
   - aq 是参考的**整体移位副本** ⇒ **行基址/stride bug**（`a_stride` 契约）；
   - `aqsc` 也不等 ⇒ 才回到 amax。
   ⚠️ 输入是**均匀填充**（`0x38` / `asc=1.0f`）——均匀正输入下出现负字节，说明 m-row 侧取了**别处的字节**或**符号处理反了**，二者用第 2 支即可区分。
3. **F2 的判别实验**：sentinel 计数 **== m·n1 恰好** ⇒ arm 未写 act（查 decline 告警：`sh_pair_m` 相关的 one-shot note）；
   **< m·n1** ⇒ 行覆盖漏写（真 row-mapping bug）。
4. **最后才是 parity 硬化**：`raw f32 bits memcmp`（非容差）+ 四臂 A/B（`SH_PAIR_M_FOLD ∈ {1,2,6}`）。

### 4.3 一个必须先排除的「测试口径」分支
judge (d) 的 else 分支在 **act == nullptr** 时要求 `sent == 0`（即「必须被写过」）。
这条断言的语义本身就是可疑的——**若 act 指针为 null，正确的期望应当是「全 sentinel / 不参与比较」**。
⇒ 在动 kernel 之前先核对该断言（本仓 #1 陷阱：**两臂都测旧路径 / 判据本身错**）。
`[tiny/m=2] act=buf` 通过而 `act=null` 用例失败，**高度符合「判据写反」而非「kernel 写漏」**。

---

## 5. 行动计划（按优先级 / 依赖）

| P | 动作 | 判据 | GPU | 预期 | 依赖 |
|---|---|---|---|---|---|
| **P0-a** | **SWALLOW 结果判读**（本次测试）：`[dspark] steps=` 的 `verify=` + `[verify_graph] captured …_m6` + k_acc 逐位对照 lazy 序列 + 0 hang | §0 的 E1~E7（沿用 `swallow-result-action-plan §1`） | 已跑 | 决定 S1 是否成立 | — |
| **P0-b** | **统一计时口径**：nsys 一次（`nsys_wave1.sh`，口径 A 窗口切分 / 口径 B 差分），按 kernel 名验证**哪个臂/哪个 m 真的跑了**（`gemm_fp8_mrows_kernel<M>` 的 M 就是行数） | Wave 1 的真实 ms 位移（不是 −8.4 的 launch 账） | 1 | 重钉 S0 | — |
| **P0-c** | **hang 的概率性**：0 ar5-hang **一次不算修好**（历史 gap 1-2 → 22 → 3 是竞态）。至少 3 次独立运行 + 一次长跑（跨过历史 hang 出现的步数） | 3/3 无 hang | 3 | 防「假解锁」 | P0-a |
| **P1** | **m=6 mrows 族逐个 A/B**（零代码）：`GATE_MROWS` → `INDEXER_MROWS` → `VERIFY_ROPE_MROWS` → **`VERIFY_HEAD_MROWS` 最后单独上** | 每个 gate：k_acc 逐位不变 + `verify_ms` 位移 | 4 | **−4.5~5.8ms** | P0-c（必须 batched 常开） |
| **P2** | **SH_PAIR parity 修复**（§4 诊断树） | parity 100% 逐位；再跑四段文本零拉丁 | 2~3 | **−4.9~7.9ms** | 可与 P1 并行（非 GPU 冲突） |
| **P3** | **tcgen05 重测**（符号预检 → 冒烟 → 门税对照 → grouped） | `≥3ms` 改善 = 足额；中性即止损（见 `tcgen05-e4m3-grouped-expectation §0`） | 4~5 | **−1.0~3.8ms**（**不是 −6.8ms**） | 对齐守卫修复已入 |
| **P4** | **B6 `dsv41_gemm_fp8_mrows_f32`**（B 类公共祖先） | 「m 行核第 r 行 == M=1 f32 GEMV 第 r 行」逐位 | 0.5 | **−0.66~1.5ms**（B6 单项）；B1–B6 全族 −2.8~4.9ms（8~12 人日） | P1 定稿形状 |
| **P5** | **L4 占用/MLP**（`L4-7` hc 侧流 → `L4-1` adaptive → `L4-3` tcgen05 K-split → `L4-4` down 换核） | 逐项 nsys wave 证据 | 多 | **−5~8ms** | P3 定稿（L4-3/4 是 tcgen05 的收尾） |

**关键路径（唯一串行主干）**：`P0-a/c（SWALLOW）→ P1（mrows 族）→ P3（tcgen05）→ P5（L4）`。
SH_PAIR（P2）与 B6（P4）**并行支线**，不占关键路径的 GPU 位。

---

## 6. 400 判定（counting 任务，accept=5 ⇒ 6 tok/step）

| 阶段 | 步时 | tok/s | 400 |
|---|---:| ---:|:---:|
| S1（Wave1 + SWALLOW） | 27 | 222 | ✗ |
| S2（+ mrows 族） | 22.5 | 267 | ✗ |
| **S3（+ SH_PAIR）** | **17.5** | **343** | ✗（差 14%） |
| **S4（+ tcgen05）** | **15.5** | **387** | ⚠️ 差 3% |
| **S5（+ B6）** | **14.5** | **414** | **✓** |
| S6（+ L4） | ~12 | 500 | ✓ |

**判决**：
1. **400 的临界点落在 S4→S5 之间（15ms）**，且**必须 S3/S4/S5 三项同时足额**。按历史 60% 兑现率，
   现实落点是 **~18-20ms ⇒ 300-340 tok/s**。
2. **步时侧不是唯一门槛**：accept 3.55（阈值）× 步时 15ms = 400；出师表/对话在本架构下
   物理不可达（需要 ≤5.5/5ms，低于 L5 地板 8-9ms）。**400 是 counting-口径目标。**
3. **纯步时路线的最短可行集** = `SWALLOW + mrows 族 + SH_PAIR + tcgen05 + B6`
   （P1+P2+P3+P4），**不含 L4/L5**（那 25-34 人日属于越过 400 之后的事）。
4. **L4 是唯一能把「中等 accept（2.5-3）」也拉进 400 的层**（12ms → 250-333），但成本 16~21 人日、
   零实测背书、且 v17→v21 四变体全中性的历史明确警告「只动一个因子无效」。

---

## 7. 风险与止损门

| 风险 | 触发信号 | 止损 |
|---|---|---|
| **SWALLOW 假解锁**（一次 0 hang 不代表修好） | 3 次里任一 hang / 长跑跨过历史步数后 hang | 回到 `DSV41_SWALLOW_STEP=0`；**不要**在 hang 未复现前把 P1 的 mrows 族叠上去（会失去归因能力） |
| **Plan A+C 是 workaround 不是根因** | 换 gate 组合（尤其 `VERIFY_HEAD_MROWS`）后 hang 以新形态复现 | 记录为已知缺陷，P1 的 head mrows 单独成轮，不与其它项混 |
| **mrows 族兑现率 0**（`SH_EXP_MROWS` 的两次零收益是同一机理的先例） | 每个 gate 单独 A/B 后 `verify_ms` 位移 < 预期的 40% | 停 P1，转 P2（ms 账更大）；**不要**在同一轮里叠 gate 找感觉 |
| **SH_PAIR parity 修不动**（36 项是 kernel 结构问题） | 诊断树 4.2 走完仍无单一根因 | 冻结 `SH_PAIR_M` 默认 OFF，SH_PAIR 的 −5ms 从阶梯里划掉（阶梯落点掉到 S4=15.5ms 的下一格：19ms） |
| **tcgen05 红线**（`tc5::e4x` 从未上过 GPU；两个 `[OPEN]`） | 冒烟拉丁/非法指令 | 立即关闭路径，不投变体矩阵（勿重演 v17→v21） |
| **L4 投入陷阱** | `L4-1` 单独 A/B 中性（adaptive nwarps 的预期只有 −0.2~0.6ms） | 不在 L4 上开新变体矩阵，除非 P3 已给出 routed 的实测新地板 |

---

## 8. 定位速查（本轮涉及的门与文件）

| 对象 | 位置 |
|---|---|
| `SWALLOW_STEP` 分派 / gate | `chain_dev.rs:7172` / `swallow_step()` `:2213`（env `DSV41_SWALLOW_STEP`） |
| swallow 块实现 / anchor 语义 | `dspark_spec_swallowed`（`lazy-batched-gate §1.1` 记为 `:6284`，块 `[anchor,d1..d5]`） |
| **ar5 方案 C**（前 N 个 verify block 走 direct） | `chain_dev.rs:5530-5551`（`swallow_step() && verify_blocks < SWALLOW_GRAPH_WARMUP_BLOCKS`） |
| lazy 路由 / 阈值 | `lazy_verify()` `:2271`、`lazy_verify_*` `:2772-2795` |
| `VERIFY_ROPE_MROWS` | `:1116` gate `:1131` |
| `VERIFY_HEAD_MROWS` | `:1511` gate `:1542`；调用点 `:5935-6007`；one-shot note `:1659` |
| `GATE_MROWS`（= `ROW_FOLD_GATE`） | `row_fold_gate()` `:1225` |
| `SH_PAIR_M` / `SH_PAIR_M_FOLD` | `:1392` / `:1407`；调用点 `:11606-11647` |
| SH_PAIR 核 + launcher（**按 m 实例化**） | `dsv41_kernels.cu:6933`（`gemm_fp8_sh_exp_pair_kernel<M>`）/ `:7299-7343` |
| SH_PAIR parity 套件 | `kernels/cuda/tests_dsv41_sh_exp_mrows.cu`（五支判据 `:318-386`；decline 表 `:418-426`） |
| tcgen05 门 / kernel | `expert_tcgen05_e4m3 :756`、`expert_grouped :903`、`gateup_fuse :1891`；`dsv41_experts_mxf4.cu`（`tc5::e4x` / `tc5::mxf4`） |
| B6 设计 | `docs/agent/b6-mrows-f32-design.md`（`:29` 的 −200~240 发/步第一性计数） |
| L4/L5 清单 | `docs/agent/l4-l5-kernel-path.md` §1/§2/§4 |
| nsys 读表口径 | `docs/agent/nsys-wave1-analysis-framework.md` §1.2/§1.4 + `scripts/nsys_wave1.sh` |

---

*工部 · 只读分析 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*所有 ms 数均标注来源与口径（launch 账 / ms 账 / 设计口径 / 实测）；与任务前提冲突处
（Step 阶梯的起点、"Wave 1 −8.4ms 已验证"、"tcgen05 −6.8ms"、"B6 −2.8~4.9ms"）已显式修正并给出依据。*
