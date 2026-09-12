# SWALLOW_STEP 测试结果后的性能路径规划（结果驱动行动计划 + 48h 步骤）

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 输入：`verify-architecture-floor.md` · `dspark-sglang-real-data.md` · `batched-400-v2-prediction.md` ·
> `batched-400-v2-remaining-roi.md` · `tcgen05-e4m3-grouped-expectation.md` · `swallow-step-400-necessity.md` ·
> `accept-gap-1214-to-3.md` · `final-400-config.md` · `draft-graph-p3c.md` · 以及 `chain_dev.rs` / `dspark_dev.rs` /
> `dsv41_experts_mxf4.cu` / `build.sh` / `scripts/*.sh` 现场核对。
> 在跑的测试：**8e78e3d6**（`SWALLOW_STEP=1` + 全 mrows + `VERIFY_GRAPH` + `BF16_TRUNCATE`，**batched 非 lazy**）。

---

## 0. 三条必须先钉死的前提（否则下面的矩阵会整体偏 15~20ms）

1. **本轮矩阵里没有 tcgen05。** `scripts/batched_400_v2.sh` 的 `TC5_GATES`（:180）只在
   `B400_TCGEN05_E4M3_GROUPED=1` 时才拼进 `GATES`（:134-144）——默认跑的那一条**不含任何 tcgen05 门**。
   ⇒ 本轮**不可能**出现「tcgen05 的 −6.8ms」，`≤15ms` 这一档在账本上**结构性不可达**。
2. **mrows 族的实测兑现度 ≈ 0。** `{SH_EXP+GRAPH+ROPE+P3A}` 全开实测 **−1.21ms**（`verify-ms-breakdown.md:159`），
   `SH_EXP` 两次实测零收益（instruction-bound）。⇒ 本轮里唯一可能真兑现的 mrows 项是 **`GATE_MROWS`（预期 −2.75ms，未实测）**。
3. **`VERIFY_GRAPH` 不是性能杠杆（实测 −1.5ms，非 −15ms）。** CUDA async launch 已让 submit 与执行重叠；
   任何拿 −15ms 编的预测都偏乐观 15ms。图化在本轮的定位 = **让 m=6 形状能被 replay 的顺序/正确性工具**。

**⇒ 本轮读数的合理票面**（`batched-400-v2-prediction.md §2 锚 A`，逐项已对齐）：
`verify(m=6) ≈ 33~36ms`、`步时 ≈ 37~40ms`、`mean_k ≈ 1.02~1.2`、`tok/s ≈ 50~65`、`latin=0`、`vg_shapes ⊇ {m=5,m=6}`。

---

## 1. 结果判读：零 GPU 的 7 个证据（先做，再谈行动）

| # | 证据 | 在哪 | 期望值（正常） | 缺失/异常的含义 |
|---|---|---|---|---|
| E1 | **`[dspark] steps=` 的 `verify=`** | serve 日志 / 脚本 metrics | **比 m=5 基线 +1.6ms**（≈38.9 起） | 没动 = m=6 块没分派（swallow 未生效） |
| E2 | **`[tick] total` 整步** | `DSV41_TIMING` | **比 legacy 整步低 ~4.55ms** | 没降 = 主链步没被吞 |
| E3 | **`[verify_graph] captured verify_graph_m6`** | serve 日志 | 必须出现 | 未出现 = 6 行块退直发（图没接上，swallow 缩水） |
| E4 | **`mean-k` + k_acc 直方图** | `[dspark] steps=` / 脚本 | legacy 语义，**与 lazy 可直接比** | 若 <1.0 = 本轮基线被打坏 |
| E5 | **`latin=0` / `dbl=0` / 四段文本逐字** | 脚本 metrics | 全过 | 任一破 = 红线，最高优先级 |
| E6 | **`<tag>.env` 逐门核对** | 脚本落盘 | 含 `SWALLOW_STEP=1`；**不含** `LAZY_VERIFY` / 任何 tcgen05 门 | 泄漏 = 本轮配置与描述不符，读数作废 |
| E7 | **`rounds` 与 1000 token 匹配** | 脚本 | ~450 轮（tok/step≈2） | <100 = 提前 EOS，可比性问题不是性能问题 |

**判读顺序**：E6 → E5 → E1/E2 → E3 → E4 → E7。**在缺 E1/E2/E3 时不要对 swallow 下"生效/未生效"结论**
（项目铁律：缺证据 exit 2）。

---

## 2. 结果驱动的行动计划矩阵

### 2.1 校准后的判读表（**替代任务口径的绝对阈值**）

> 任务给的表把 `≤15ms` 当作"SWALLOW+mrows 兑现"、`>35ms` 当作"完全未改善"。
> 按 §0 的前提，**本轮 batched 的真实落点在 35~40ms**，且它**不代表 swallow 失败**——
> batched verify 本身就是 33~36ms 量级（lazy 之所以 22.6ms 是因为 accept 低、只跑 2 行）。
> ⇒ 用**绝对步时**分档会误判，必须改用「**E1+E2 的联合位移**」分档。

| 档 | 联合判据（E1+E2） | 步时参考 | 含义 |
|---|---|---:|---|
| **A** | verify= 未动 **且** 整步未降 | 任意 | **swallow 未分派**（真问题，查 E3 + `spec_primed`） |
| **B** | verify= 动了(+1.6) **且** 整步降了 | 37~40ms | **swallow 生效**；缺口 = mrows 零兑现 + 无 tcgen05（账本预期，非 bug） |
| **C** | verify= 动了 **且** 整步降幅 >4.5ms | 30~35ms | **swallow + `GATE_MROWS` 同时兑现**（超预期，锁配置） |
| **D** | 步时 ≤25ms | — | **先质疑测量**（账本：无 tcgen05 时 batched verify 地板 ≈33+ms） |
| **R** | 任何拉丁/双字/缺句 | — | **红线优先**，立即二分隔离 |

### 2.2 每档的 48h 行动（具体到 gate 与文件）

#### 档 A — swallow 未分派（verify= 没 +1.6）
**根因候选（按概率）**：
1. `spec_primed` 引导没走通（首轮没置位 ⇒ `dspark_spec_swallowed` 永不进入）。
   落点：`chain_dev.rs:6943`（分派 `swallow_step() && self.spec_primed`）、`:7153`（置位）、`:7372`（swallowed 尾置位）、`reset()` `:3663` 清零。
2. `m=6` 形状池槽位没建（`VERIFY_GRAPH_SLOTS=3`，`verify_slot` `chain_dev.rs:5340`）——
   但 per-shape 槽是自动的，**只有在 `[verify_graph]` 出现 `capture FAILED (m=6 …)` 时才成立**。
3. `SIDS_WRITEBACK` 未开（硬依赖，`chain_dev.rs:1723`）⇒ swallow 轮后 `s.ids` 落后，**下一轮 legacy 嵌旧 token**，
   表现为 accept 塌 + 可能红线。

**48h 步骤**：
- `t+0~2h`：读 E1/E2/E3/E6；`grep -n "spec_primed" serve 日志`、`grep "verify_graph" serve 日志`。
- `t+2~6h`：**只读代码复核**（不改）：`carry_kept_tap` `:7999` 的 `keep=k_emit` 行下标是否 = block 里
  **已提交的最后一行**（`dspark-swallow-step-diff.md §4` 标注的"需精确处理的设计点"）。
- `t+6~10h`：**同会话背靠背 A/B**：`SWALLOW_STEP=0` 与 `=1`（其余 gate 冻结），
  判据 = `[tick] total` 的 Δ≈−4.5ms + `[dspark] verify=` 的 Δ≈+1.6ms。
- `t+10~24h`：若 A/B 仍无位移 ⇒ 转**档 B 的行动**（swallow 不是本轮主缺口，别继续挖）。

#### 档 B — swallow 生效，缺口是 mrows 零兑现 + 无 tcgen05（**最可能**）
**含义**：账本预期内的结果，**不是失败**。真缺口两项：
- **tcgen05 未 armed**（−5~−6.8ms，最大单项，被 4-gate 联合 + ILV 默认 ON 三重阻塞）；
- **accept 无杠杆**（本轮矩阵不含 `TAP_INPUT`/`DRAFT_BF16_DOMAIN`，accept ≈ 1.02）。

**48h 步骤（按 ROI）**：
| 时段 | 动作 | 文件 / gate | 预期 | 成本 |
|---|---|---|---|---|
| `t+2~6h` | **hc 链 A2 的 truncate 修复 + 开 `HC_FRONT_ROWS`**（一行改 verify 调用点传 `false`） | `chain_dev.rs::hc_mixes_auto` `:11556-11714`，调用点 `:11625`；gate `HC_FRONT_ROWS`（前置 `HC_VERIFY_FUSE` `:11801`） | **−1.3~−1.7ms** | 0.5~1 人日 |
| `t+6~8h` | **`GATE_MROWS` 的独立 A/B**（本轮已设但未单测） | `row_fold_gate()` `:1225`（双名 `DSV41_GATE_MROWS`/`DSV41_ROW_FOLD_GATE`） | −2.75ms（若真兑现） | 0（现有 gate） |
| `t+8~12h` | **`INDEXER_MROWS` front**（代码已就位，零成本） | `indexer_mrows` `:1173` / `indexer_front_rows` `:9790` | −1.0~−1.5ms | 0 |
| `t+12~24h` | **tcgen05 的隔离冒烟 + parity**（见 §3，**必须先于集成**） | `scripts/tcgen05_bench.sh` + 短 prompt 冒烟 | go/no-go | 1 GPU 会话 |
| `t+24~48h` | **tcgen05 grouped 集成 A/B** | 见 §3 | −1~−2.8ms（修正后口径，非 −6.8） | 1 GPU 会话 |

#### 档 C — swallow + `GATE_MROWS` 同时兑现（超预期）
- **动作**：**冻结配置**（单 commit 记 gate 组合），立即补两轮独立测试：
  1. **accept 轮**：`+DSV41_TAP_INPUT=1 DSV41_DRAFT_BF16_DOMAIN=1`（P0-3+P1-5，预期 accept→1.214）。
  2. **tcgen05 轮**：按 §3 的组合（四件套必须成组——`ILV=0` 改权重布局，不能与 `ILV=1` 比绝对值）。
- **禁止**：在同一轮里混入新的语义改动（swallow 已是语义改动）。

#### 档 D — 步时 ≤25ms（先质疑测量）
- **零成本自检**：`vg_shapes` 是否含 `m=6`、`<tag>.env` 是否泄漏 `LAZY_VERIFY`、`rounds` 是否匹配。
- 证据齐全才认；认了立刻锁配置 + 走档 C 的两个补测。

#### 档 R — 红线破（拉丁/双字/缺句）
**二分隔离（每步单独一轮，脚本 `LOCK` 保证串行）**：
1. `SWALLOW_STEP=0`（保留其余）——swallow × `SIDS_WRITEBACK` 是本轮**唯一的新语义组合**，第一嫌疑。
2. `VERIFY_GRAPH=0`——capture/replay 若与 host mirror（`compress_lens`）不同步会静默改状态。
3. `EXPERT_ACT_E4M3=0`——`E4M3=1` + ILV 默认 ON 的组合**无历史 A/B 证据**。
4. `SIDS_WRITEBACK=0`——单独验证 swallow 的硬依赖假设。
- **`HC_VERIFY_FUSE` 排除**：脚本 `FORBIDDEN`（:148）硬校验，本轮不可能是它。

### 2.3 batched vs lazy 的决策规则（**别在档 B 直接回退 lazy**）

账本事实：**lazy 今天更快（22.56ms vs batched ~39ms），但这是 accept 低到 lazy 只跑 2 行的假象**
（`verify-architecture-floor.md §5.3`）。回退 lazy 会**锁死 400**——lazy 每行 m=1，mrows/权重共享完全帮不上。

| accept | 正确路径 |
|---|---|
| **≤1.2**（现状） | lazy 更快，但 **400 物理不可达**（τ=2.21 < sglang 下界 2.80）⇒ **先攻 accept，别在 batched/lazy 之间选** |
| **≥2.5** | **batched 反转**（6 行共享权重），继续 batched + swallow |

---

## 3. tcgen05 的测试顺序（三件套 + ILV + GATEUP_FUSE 的正确组合）

### 3.1 正确的门组合（逐条核对，全部成立才 arm）

```
DSV41_EXPERT_ACT_E4M3=1       # 激活 e4m3（≠"0" 宽松读）
DSV41_EXPERT_TCGEN05_E4M3=1   # e4m3 tcgen05 家族运行时门（严格 starts_with('1')）
DSV41_EXPERT_GROUPED=1        # grouped 布局 + mask 路径
DSV41_GATEUP_FUSE=0           # ← 关键：默认 ON 会触发 grouped decline #3
DSV41_EXPERT_ILV=0            # belt-and-braces（GATEUP_FUSE=0 已蕴含 ILV=0）
```
- **「三件套」= 前 3 个；`GATEUP_FUSE=0` 是被默认 ON 阻塞的隐藏坑 #3；`ILV=0` 是隐含门**（`load.rs:767-778` 的
  `ilv_ok()` 把 `gateup_fuse()` 作为合取项 ⇒ **不能做「ILV=0 但 GATEUP_FUSE=1」的实验**）。
- 代码落点：`chain_dev.rs:756`（`expert_tcgen05_e4m3`）、`:903`（`expert_grouped`）、`:1891`（`gateup_fuse`）、
  `weights.rs:487`（ILV）；kernel `dsv41_experts_mxf4.cu:6070`（`dsv41_expert_gemm_e4m3_grouped`，`tc5::e4x`）。
- **成组纪律**：这 5 个门**必须同开**（`ILV=0` 改权重布局 ⇒ 不能与 ILV=1 的结果比绝对值）。

### 3.2 执行顺序（**先隔离、再门税对照、最后集成**）

| # | 步骤 | 命令 / 位置 | 判据 | GPU |
|---|---|---|---|---|
| **0a** | **符号预检**（零 GPU 成本） | `nm -D libferrite_kernels.so \| grep -E "dsv41_expert_gemm_e4m3_grouped\|dsv41_route_(group\|gather_rows\|scatter_rows)"` | 四个符号缺任一 ⇒ **不要跑 arm** | 无 |
| **0b** | dry-run | `bash scripts/batched_400_v2.sh --dry-run` | 打印行含 `B400_TCGEN05_E4M3_GROUPED` 那串 | 无 |
| **1** | **短 prompt 冒烟**（2 分钟） | 同矩阵 + `MAXTOK=100~200` | 抓「非法指令 / serve 起不来 / 拉丁」 | 1 |
| **2** | **基线轮（arm OFF）** | `bash scripts/batched_400_v2.sh` | 同次 rebuild 的 verify/step 参考；`.env` 里**不含**这 5 门 | 1 |
| **3** | **门税对照轮**（**关键**，脚本当前不支持，需手工 serve 或加 knob） | 只 `GATEUP_FUSE=0 EXPERT_ILV=0` | 测出「arm 的必付税」≈ +0.4~0.5ms | 1 |
| **4** | **tcgen05 grouped 轮** | `B400_TCGEN05_E4M3_GROUPED=1 bash scripts/batched_400_v2.sh` | 见 §3.3 落点表 | 1 |
| **5** | 条件触发二分 | 红线 ⇒ `GROUPED=0` → `TCGEN05_E4M3=0` | 不要先关 E4M3（那是激活格式，会引入第二变量） | 1 |

**为什么不把 arm 当第一个 GPU 接触点**：`tc5::e4x` **从未在任何 GPU 上执行过**，且源码自带两个 `[OPEN]`
（dense idesc format code、f8f6f4 的 fp4 packed/unpacked 假设）——二者任一错 = 静默错值或非法指令。
`tcgen05_bench.sh` 只覆盖 swapAB `mxf4` 入口，**没有 e4x 的隔离 harness** ⇒ 冒烟是它唯一的前置。

### 3.3 tcgen05 grouped 读数落点表（**已按修正口径，非任务前提的 −6.8ms**）

| 读数（相对同次 rebuild 基线） | 解释 | 动作 |
|---|---|---|
| **≥3ms 改善** | kernel 生效且高效 | 记录；考虑补 gateup 微基准 + 论证 down 臂 |
| **1~3ms 改善** | 生效但受成本结构限制（无 cp.async、M=128 固定 ⇒ ~120× 张量核过量） | 看 nsys kernel duration；**不要急着调参** |
| **无变化** | 要么 decline（查 §3.4 告警），要么被 +0.5ms 门税抵消 | **先查告警，再做步骤 3 对照轮** |
| **退化 0.3~0.7ms** | 高概率 = **回退（门税）**，不是 kernel 慢 | 查 §3.4 告警；告警在 ⇒ kernel 根本没跑 |
| **红线破** | `[OPEN]` 命中（最可能）或 R7 | 步骤 5 二分 |

### 3.4 归因三件套（每轮跑完必做）
1. **`verify_ms`/`steady_median` 相对步骤 2 基线**的位移（不是相对历史基线）；
2. `[verify_graph]` 三态（`captured` / `capture FAILED` / 无行）；
3. **`[gmo]`/`[phs]`**——确认窗口内是否触发 `moe()`（同一门也武装了单行 `moe()` 的 swapAB e4m3 kernel，R7）。
4. **一次性告警**任一出现 = arm 没生效：`expert_grouped_skipped_note` / `tcgen05_e4m3_ext_skipped_note` /
   `tcgen05_e4m3_skipped_note`（`chain_dev.rs:920/:789/:772`）。

---

## 4. S2（accept A/B）与性能测试的 GPU 串行策略

### 4.1 为什么不能并行
- `s2_ab_matrix.sh` 与 `batched_400_v2.sh` **都要求 TP8 + `GPU_LIST=0..7`**（`s2:95/:94`、`batched:94`）——
  8 张卡被一个 serve 独占，**两个 serve 同时跑不是噪声问题，是无意义问题**。
- 项目铁律：**同一台远端同时只能有一个测试驱动**；subagent 只做代码/分析，GPU 由主 agent 串行执行。

### 4.2 串行队列（一个 GPU 会话 = 一个 arm 集合）

| 队列位 | 会话 | arm 数 | 判据 | 依赖 |
|---|---|---|---|---|
| Q1 | **SWALLOW 结果判读**（在跑，8e78e3d6） | 1 | §1 的 E1~E7 | — |
| Q2 | **S2 accept 矩阵**（`s2_ab_matrix.sh`） | 4（A baseline / B TAP_BF16 / C DRAFT_ATTN_BF16 / D 双开） | k_acc 直方图 + mean-k；红线 | `BASE_SEED_POS=0` |
| Q3 | **新增 accept 臂**：`DSV41_DRAFT_HEAD_FOLD=0`（v1 逐行，与 verify head 同程序） | +1 臂 | mean-k 是否升 | `dspark_dev.rs:94` |
| Q4 | **tcgen05 步骤 1~4**（§3） | 4~5 | §3.3 落点表 | 符号预检 |
| Q5 | **hc/indexer 的 A/B**（档 B 的 t+2~12h） | 2~3 | `verify_ms` 位移 + 零拉丁 | A2 truncate 修复先落 |

### 4.3 并行的是**非 GPU 工作**（这才是"并行策略"）
- **Q1 在跑期间**：subagent 可并行做（零 GPU）：
  - `hc_mixes_auto` 的 A2 truncate 一行修复（`chain_dev.rs:11625` 调用点传 `false`）；
  - tcgen05 的符号预检 + dry-run；
  - 读 §1 的日志、写判读报告。
- **Q4 之前**：把 `GATE_MROWS`/`INDEXER_MROWS`/`NORM_MROWS` 的默认值翻转**单 gate 单 commit** 准备好（不落盘，
  等 A/B 结果），避免在 GPU 会话里改代码。

---

## 5. accept 与步时的资源分配（**accept 是门槛，步时是兑现**）

### 5.1 sglang 硬锚点给出的判决
```
verify 7.3ms（一手实测）⇒ step ≥ 7.3ms ⇒ τ ≥ 383.7 × 0.0073 = 2.80 tok/step   ← accept 的独立下界（非循环）
ferrite 现状 τ = mean-k + 1 = 2.214 < 2.80
⇒ 即使步时压到 sglang 的 verify 实测下界 7.3ms，也只有 303 tok/s —— 400 不可达
```

| 只动一轴 | 极限 | 结果 |
|---|---|---|
| 只提 accept（步时停 22.56ms） | τ 最大 6.0 | 265.9 tok/s ✗ |
| 只降步时（accept 停 2.214） | 步时压到 7.3ms | 303.3 tok/s ✗ |
| **双动**（τ=5，步时 13ms） | — | 383.7 tok/s ✓ |

**⇒ 精确表述：accept 是必要条件（门槛，mean-k ≥ 1.92 才谈得上 400），步时是充分条件的一半（兑现）。**

### 5.2 48h 的分配（accept 60% / 步时 40%）
**理由**：
- **accept 侧成本极低**：S2 是**现成 gate 的一次 A/B**（0.5 人日、零代码风险）；`DRAFT_HEAD_FOLD=0` 是
  一行 gate 的新臂；SWALLOW 本身**同时是 accept 的根因 A 修复**（tap 传递）——**它已经在 Q1 里出结果**。
- **步时侧成本高**：tcgen05 是 4~5 人日且**从未上 GPU**（正确性风险）；族级融合 12~15 人日。

| 分配 | 内容 | 验收 |
|---|---|---|
| **accept 60%** | Q2（S2 四臂）+ Q3（DRAFT_HEAD_FOLD）+ Q1 的 accept 判读 | mean-k 是否 ≥1.4（脱离链式陷阱的门槛 p>0.7） |
| **步时 40%** | Q5（hc+indexer front，−2.3~−3.3ms）+ Q4（tcgen05 冒烟与集成） | verify_ms 位移 + 零拉丁 |

**判据**：**只有当 Q2/Q3 把 mean-k 顶到 ≥1.5（τ≥2.5）时，才值得把步时预算加码到 tcgen05 的完整集成**；
若 accept 停在 ~1.2，tcgen05 的 −2.8ms 只是把 250 tok/s 变成 267 tok/s——**量级不变**。

### 5.3 400 等值线（本轮后的现实带）
| 步时 | accept 3 (τ=4) | accept 2 (τ=3) |
|---:|---:|---:|
| 40ms（本轮） | 100 | 75 |
| 12ms（全 gate + tcgen05 乐观） | 333 | 250 |
| **10ms（L4~L5）** | **400 ✓** | 300 |
| 9ms | 444 ✓ | 333 |

**⇒ 250~350 tok/s 是本战役的现实票面**；400 需要 `accept ≥3 且步时 ≤10ms`（或 `accept ≥4 且 ≤12.5ms`）。

---

## 6. 48 小时的具体步骤（时间轴）

> 以 Q1（8e78e3d6）出结果为 t+0。

| 时段 | GPU | 动作 | 产出 |
|---|---|---|---|
| **t+0~2h** | — | §1 的 E1~E7 判读；对照 §2.1 定档 | 档位结论（A/B/C/D/R） |
| **t+2~6h** | — | subagent 并行：A2 truncate 一行修复 + tcgen05 符号预检 + dry-run | 待测的代码 patch + go/no-go |
| **t+6~12h** | **GPU-1** | **S2 矩阵**（Q2，4 臂串行） | mean-k + k_acc 直方图 |
| **t+12~18h** | **GPU-2** | **Q3**（`DRAFT_HEAD_FOLD=0` 单臂，接 Q2 最佳基线上） | accept 新杠杆判定 |
| **t+18~24h** | — | 汇总 accept：若 mean-k <1.5 ⇒ 转 S4（成对域对齐）；若 ≥1.5 ⇒ 保留 | 路径分叉 |
| **t+24~32h** | **GPU-3** | **Q5**：hc A2（`HC_FRONT_ROWS`）+ `GATE_MROWS` 独立 A/B + `INDEXER_MROWS` | −2.3~−3.3ms 实测 |
| **t+32~40h** | **GPU-4** | **tcgen05 步骤 1~3**（冒烟 + 基线 + 门税对照） | 冒烟过 / 不过 |
| **t+40~48h** | **GPU-5** | **tcgen05 步骤 4**（grouped 集成 A/B）或按冒烟结果止损 | §3.3 落点 |

**止损门**：
- tcgen05 冒烟红线破 ⇒ **立即关闭路径**，不再投变体矩阵（勿重演 v17→v21 四变体全中性）；
- Q5 的 `verify_ms` 降幅 < 预期 60% ⇒ 先查「graph×合并重叠」与「a32/V7 的 proj_mrows 真值」，再拆单 gate 重测；
- accept 三轮后仍 <1.3 ⇒ 停步时投入，转 accept 的结构层（draft/verify 共用 attention kernel）。

---

## 7. 附：本轮涉及的文件与 gate 速查

| 对象 | 位置 |
|---|---|
| SWALLOW 分派 | `chain_dev.rs:6943`；实现 `dspark_spec_swallowed :7459`；tap 传递 `carry_kept_tap :7999` |
| `spec_primed` 置位/清零 | `:7153` / `:7372` / `reset :3663` |
| `SIDS_WRITEBACK`（swallow 硬依赖） | `chain_dev.rs:1723` |
| `VERIFY_GRAPH` 槽位/日志 | `VERIFY_GRAPH_SLOTS :107`、`verify_slot :5340`、capture 日志 `:5222/:5244` |
| `GATE_MROWS`（= `ROW_FOLD_GATE`） | `row_fold_gate() chain_dev.rs:1225` |
| `SH_EXP_MROWS` | `chain_dev.rs:1259` |
| `INDEXER_MROWS` | `:1173` / `indexer_front_rows :9790` |
| `NORM_MROWS` | `:1200` |
| `COMPRESSOR_MROWS` | `:1062` |
| hc A1/A2 | `hc_verify_fuse :11801`、`hc_mixes_auto :11556-11714`（A2 truncate 在 `:11625`） |
| tcgen05 门 | `expert_tcgen05_e4m3 :756`、`expert_grouped :903`、`gateup_fuse :1891`、`expert_tcgen05_mxf4 :723` |
| ILV | `weights.rs:487` |
| tcgen05 kernel | `dsv41_experts_mxf4.cu:5120`（`tc5::e4` swapAB）、`:6070`（`tc5::e4x` grouped） |
| build skeleton flags | `kernels/cuda/build.sh:51-60` |
| 脚本 | `batched_400_v2.sh`（gates `:134-144`、`TC5_GATES :180`、`FORBIDDEN :148`）、`s2_ab_matrix.sh`、`tcgen05_bench.sh`、`verify_mrows.sh`、`dsv41_tcgen05_mxf4_verify.sh` |
| accept 杠杆 | `tap_input :11945`、`draft_head_fold dspark_dev.rs:94`、`tap_bf16 :291`、`draft_attn_bf16 :300` |

---

*工部 · 只读分析 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*所有 ms/launch 数均标注来源；与任务前提冲突处（绝对步时阈值、tcgen05 的 −6.8ms）已显式修正并给出依据。*
