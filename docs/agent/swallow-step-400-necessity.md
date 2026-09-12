# SWALLOW_STEP 与 400 路径的必要性分析

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 代码基线：HEAD `51f2515` · `crates/ferrite-models/src/dsv41/chain_dev.rs`。
> 背景：`batched-400-final` 的 gate 矩阵（A 组 perf / B 组 correctness）**漏了 `DSV41_SWALLOW_STEP=1`**。
> 输入：`final-400-config.md` · `final-400-battle.md` · `verify-family-fusion.md` · `lazy-batched-gate.md` ·
> `dspark-swallow-step-diff.md` · `verify-marginal-cost.md`。

---

## 0. 判决（三句）

1. **swallow 不是"锦上添花的 perf gate"，而是 10ms 预算的结构性前提。**
   缺了它，步时 = 主链 forward(6.15) + verify + draft + commit ≥ 6.15 + 0.8 + 0.2 = **7.15ms 的地板，
   而 verify 的 5/6 行边际（≥5.2ms，见 §2）必然把总步时推到 12ms+** ⇒ accept 3 下 400 直接出局。
2. **swallow 的作用是把"anchor 的 forward"从一次独立的 m=1 主链步（6.15ms，权重读 1 遍）**
   **搬进 verify 的 6 行块（m=6，与 5 个 draft 行共享同一遍权重读）**——anchor 行的边际只剩 +1.6ms。
   净 **−4.55ms/步**（主链 −6.15，verify +1.6）。这是**唯一合法的"消除一次独立 forward"**（`final-400-battle.md` §4.3）。
3. **它必须进 A 组（perf）矩阵，且必须独立 A/B**：因为它改的是 **m=5→6 的块形状 + commit/tap/pos 语义**，
   不是纯默认值翻转。混在其它 mrows gate 里跑，性能账会被语义改动与 dispatch 失效互相污染。

---

## 1. 机制：swallow 到底做了什么

### 1.1 两条路径的对照（`chain_dev.rs`）

| | legacy 臂（默认） | swallowed 臂（`DSV41_SWALLOW_STEP=1`） |
|---|---|---|
| 入口 | `dspark_spec_step` :6702 | `dspark_spec_swallowed` :7253 |
| 分派条件 | — | `swallow_step() && self.spec_primed`（:6737） |
| 主链步 | **`step_dev(token, pos)` = 6.15ms**（:6791） | **无**（被吞） |
| snapshot | `dspark_snapshot(pos_ctr+1, 5)`（:6798） | `dspark_snapshot(pos, 6)`（:7269） |
| draft | `draft_forward(token, pos)` | 同（:7280） |
| verify 块 | `[d1..d5]` @ `pos+1..pos+5`，**m=5** | `[token, d1..d5]` @ `pos..pos+5`，**m=6**（:7290-7294） |
| accept | `next` 领判 + 5 行移位链（:6872-6875） | **索引对齐链** `spec_accept(drafts, rows, true)`（:7315） |
| commit | `dspark_commit(pos+1, 5, k_acc)`（:6881） | `dspark_commit(pos, 6, k_emit)`（:7328） |
| tap | `step_dev` 直接写 `dspark_tap` | `carry_kept_tap(keep=k_emit)` 从 block 第 `k_emit-1` 行拷（:7331） |

### 1.2 为什么 anchor 行可以"免费"进 verify 块

- legacy 的 `step_dev` 把 anchor token 在 `pos` 位置 forward 一遍，产出 `next=argmax` 和 tap。**这是 m=1，整份 40 层权重读一遍 = 6.15ms。**
- swallow 让 verify 的 **行 0 就是这个 anchor forward**（同一 token、同一位置、写同一 KV 行，:7185-7189），
  它的 argmax 就是 `next`。verify 本来就是一次 m 行批处理——**加一行不重读权重**，只加这一行的激活侧工作量（≈+1.6ms）。
- ⇒ **anchor 的 6.15ms 权重读被"折"进 verify 那一次权重读里**，这就是净 −4.55ms 的全部来源。

### 1.3 三个必须记住的语义点（已在实现里处理，A/B 时要盯）

1. **commit 的 `keep` 语义是"从行 0 数，保留 `0..keep`"**（`dspark_commit` :7816-7834 把两种布局写死）：
   - legacy：block 行 0 = `d1` @ `pos+1`，`keep=k_acc` ⇒ `pos_ctr = pos+k_acc+1`；
   - swallow：block 行 0 = anchor @ `pos`，`keep=k_emit=k_acc+1` ⇒ `pos_ctr = pos+k_emit`。
   两者是**同一个函数**，只是实参不同——没有新增分支。
2. **tap 跨轮传递**：swallow 轮不跑 `step_dev`，tap 只能来自**上一轮 verify 块的最后一行已提交行**
   （行 `k_emit-1`，被喂过真 token）。`carry_kept_tap`（:7793，`keep=k_emit`）把它拷进 `dspark_tap`。
   ⚠️ 这是本路径**唯一的近似输入**（位置比 legacy 的 tap 早一格）；若要 A/B 备选（行 `k_emit`，位置精确但喂的是被拒 draft），
   在两个调用点把 `keep` 加 1（:7244-7246）。
3. **AR 足迹**：swallow 轮 `draft_forward(token, pos)` 在 `pos>=1` 上恒跑 3 个 MTP block 的 AR；
   legacy 首轮在 `pos==0` 会命中 prefill 早退（0 AR）。**必须靠 `spec_primed` 引导：首轮走 legacy**（:6737 的 `&& self.spec_primed`，
   :6947 置位，`reset()` 清零），否则 `need−cur=3` 会让 AR v5 永久自旋（已修，见 `dspark-correctness-chain.md`）。

---

## 2. 必要性：没有 swallow 时，10ms 预算为什么必然出局

### 2.1 预算口径

```
verify ≤ 9.0  +  draft ≤ 0.8  +  commit ≤ 0.2  ≤  10.0 ms      （不含主链步）
```
（`verify-family-fusion.md` §718-730：accept~3 校准后，400 需要 accept 3 + 步时 ≤10ms，4 tok/step → 4/0.010 = 400。）

这个 **verify ≤9ms 的地板本身就是"6 行批处理 + 权重共享"的地板**（权重流 14GB 读一遍 2ms + launch 3.9ms + 6 行计算 2-3ms ≈ 8-9ms）。
换句话说：**这个预算已经默认 anchor 行在 verify 块里**（否则那个 9ms 就不是 6 行而是 5 行，且 anchor 的 6.15ms 无处安放）。

### 2.2 加上主链步后的账

| 路径 | 主链步 | verify | draft | commit | **步时** | @accept 3（4 tok/step） |
|---|---:|---:|---:|---:|---:|---:|
| **关掉 swallow**（legacy） | **6.15** | ≥5.2 边际（6 行）→ 现实 8~10 | 0.8 | 0.2 | **≥ 15.2~17.2** | **233~263 tok/s ✗** |
| **开 swallow** | **0**（折进 verify 行 0） | 6 行含 anchor，8~9 | 0.8 | 0.2 | **≈ 9.2~10.2** | **≈ 400 ✓（刚好）** |

- 关掉 swallow 时 verify 走 **m=5**（`[d1..d5]`），但 **anchor 的 6.15ms 主链步仍要独立跑**。
  即使假设 verify(m=5) 也只花 8ms，步时 = 6.15+8+0.8+0.2 = **15.15ms ⇒ 264 tok/s**，离 400 差 1.5×。
- **verify 的 5 行边际不是 0.5ms 而是 ~5.2ms**（`final-400-battle.md` §2.2：routed experts 的字节随行数 ×6，不可折叠；
  30 个 assignment 选 29 个唯一专家，去重仅 3%）。⇒ **主链 forward 之后，10ms 预算里根本塞不下一个 5~6 行 verify。**
- ⇒ **结论：swallow 是 400@accept3 的必要条件，不是可选项。** 把它从 gate 矩阵里漏掉，
  整轮 batched-400 测试测的是一条**结构上不可能到 400 的路径**（12ms+），结果必然是"mrows 全开也没改善"的假阴性。

### 2.3 与"lazy"的关系（避免选错臂）

- `DSV41_LAZY_VERIFY=1` **蕴含 swallow**（`lazy_verify()` :2023-2049 的注释 + :6751 分派，:6744-6745）：
  lazy 臂的行 0 就是被吞的主链步，它只是把 6 行块拆成逐行（m=1）跑。
- **但 lazy 到不了 10ms**：每行 m=1 独立 forward，**mrows 的权重共享完全帮不上**（mrows 需要 m>1），
  per-row ~7ms × 4 行 = 28ms verify。⇒ **400 路径必须跑 batched（不开 LAZY_VERIFY），用 swallow。**
- 这也和 `verify-family-fusion.md` §730 的"全栈测试应该跑 batched（不开 LAZY_VERIFY）"一致。

---

## 3. swallow 后的形状变化（m=5→6）及其影响面

### 3.1 图化形状池（VERIFY_GRAPH）

- `VERIFY_ROWS = 6`（:84），**m=6 恰好是分配上限**，不是超限（`step_rows` :4969 只拒 `m > VERIFY_ROWS`）。
- 形状池 `verify_graphs/verify_shapes` 是 **per-shape 槽**（`VERIFY_GRAPH_SLOTS = 3`，:107；`verify_slot` :5192）：
  **m=5 与 m=6 自动各占一槽，各自的 DRY→CAPTURE→REPLAY 独立**（:4915-4920 明确写死了这个场景）。
- ⇒ `final-400-config.md` R2 里"**m 从 5→6 必须重建形状池**"这句话**在当前代码里已经过时**：
  单槽 latch 的旧缺陷已由 per-shape 槽修掉。**不需要改代码，但必须用 `[verify_graph]` 日志确认 m=6 那一路真的 capture 成功**——
  否则 swallow 的 −4.5ms 会因为"6 行块退回 direct launch"而缩水成只有主链那部分。
- ⚠️ swallow 使一个请求**同时产生 m=5（首轮 bootstrap）和 m=6（后续轮）两个形状**：两槽都用上了。
  若同时开 lazy（m=1）就是第三槽——**3 槽刚好，但这是"别在同一轮测试里同时开 lazy 与 swallow"的另一个理由。**

### 3.2 分配缓冲

所有 verify 块相关缓冲按 `VERIFY_ROWS=6` 分配（`argmax_r`/`dspark_tap_r`/`pos_rows`/`ids_r`/`clr` 等），
legacy 只用 5 行、swallow 用满 6 行 ⇒ **无越界风险**。`DsparkSpecReport.verify_out` 仍是 `[u32; DSPARK_DRAFTS=5]`
（swallow 把 `rows[1..6]` 拷进去，:7320-7321），报告形状与 legacy 可比——统计口径（mean-k / tok/step）不漂。

### 3.3 mrows 族（A 组）对 m=6 的适配

- mrows 族一律以 **`rows = m` 为参数**（`proj_mrows`/`sh_exp_mrows`/`gemv_bf16_v2_mrows`/`hc_collapse_norm` rows 维…），
  **没有硬编码 5 的行上限**；`MROWS_SMALL_N_ADAPTIVE`（`dsv41_kernels.cu:3810`）还按 n 自适应 rows/block。
- ⇒ **m=6 与 m=5 对 mrows 是同一个程序多一行**，不引入新的形状分支。
  ⚠️ 唯一要留意的是：**mrows 的收益（−8.3ms 等）是在 m=5 下估的**，m=6 会多一行，
  实测 verify 应从 m=5 的值 **+1.6ms 左右**；A/B 时要按"swallow=1 的 verify 比 swallow=0 高 ~1.6ms、
  但整步低 ~4.5ms"来读，**不能拿 verify 绝对值变大当失败**。

### 3.4 与现有 gate 的组合风险

| 组合 | 风险 | 处置 |
|---|---|---|
| **swallow × VERIFY_GRAPH** | m=6 的 capture 必须发生（否则白吞）；两槽共享一个 pool，别再加第三形状 | 看 `[verify_graph] captured verify_graph_m6` 日志；本轮**不要同时开 LAZY_VERIFY** |
| **swallow × mrows 族** | m=6 多一行；mrows 若 decline（a32 等）则 verify 不降反升 | 先确认 mrows dispatch（`28515b6` 的 a32 decline 已由 `4c98b30` 移除），否则性能账作废 |
| **swallow × SIDS_WRITEBACK** | **硬依赖**（见 §4） | `SIDS_WRITEBACK=1` 必须与 swallow 同开 |
| **swallow × SEED_ALIGN** | 二者都改块布局（aligned 是 `[next,d1..d5]`@`pos+1`，swallow 是 `[token,d1..d5]`@`pos`） | **互斥**，一輪只开一个（分派顺序：swallow 在前 :6737，aligned 在后 :6785） |

---

## 4. SIDS_WRITEBACK 依赖（为什么"要求 SIDS_WRITEBACK=1"）

- swallow 轮**不跑 `step_dev`**，而 `step_body` 的 embedding 读 `s.ids` ⇒ 若无回写，
  一个 swallow 轮后 `s.ids` 会比新 `pos_ctr` 落后 `k_emit` 个位置。swallow 臂**自身不读** `s.ids`（:7361-7368 明确说明），
  但**下一轮可能是 legacy 臂**（`spec_primed` 未置位、或上一轮失败回退）——那个 `step_dev` 会 `embed s.ids` ⇒ 嵌旧 token。
  所以三条臂共用同一不变式 `s.ids == emitted.last()`，`sids_writeback()`（:1575）必须 ON。
- ⚠️ **这是 swallow 与正确性红线的耦合点**：`SIDS_WRITEBACK` 默认 OFF 的原因不是它错，而是
  **它会放大 verify 的数值误差**（回写更正确的 token → 更早 collapse；`SPARSE_OROPE` 对齐是根因，见 R6）。
  ⇒ **swallow 的 A/B 必须与 `SPARSE_OROPE` 对齐（或 EAGER 关融合）绑在一起**，否则会把
  "swallow 的语义问题"与"verify 数值问题"混淆，做出错误判决。

---

## 5. 推荐的测试组合

### 5.1 把 SWALLOW_STEP 补进 gate 矩阵

建议把 A 组（perf）矩阵补一行，并拆成"独立语义轮"（沿用 `final-400-config.md` §7-T3 的思路）：

```
A 组（perf，默认值/形状类）:  SH_EXP_MROWS / MROWS_SMALL_N / GATE_MROWS / VERIFY_HEAD_MROWS /
                              INDEXER_MROWS / NORM_MROWS / COMPRESSOR_MROWS / VERIFY_GRAPH / DRAFT_GRAPH
A' 组（swallow，独立语义轮）: SWALLOW_STEP=1   ← 补齐
B 组（correctness）:          BF16_TRUNCATE / E4M3 / SIDS_WRITEBACK
```

### 5.2 三个必须成对出现的开关

| 轮次 | 开关 | 理由 |
|---|---|---|
| **swallow 轮（隔离）** | `SWALLOW_STEP=1` + `SIDS_WRITEBACK=1` | 硬依赖（§4） |
| 同上 | **不开 `LAZY_VERIFY`** | lazy 蕴含 swallow 但走 m=1 行循环，是另一条路；且抢第三个形状槽 |
| 同上 | **不开 `SEED_ALIGN`** | 布局互斥（§3.4） |
| 同上 | verify 数值先对齐（`SPARSE_OROPE` 对齐 EAGER / 或 EAGER 关融合） | 否则 SIDS_WRITEBACK 会掩盖真实结果（R6） |

### 5.3 A/B 判据

- **正确性**：出师表逐字 + `has_double_char`（无相邻重复）+ 数字任务 + `dspark_parity` 行级对照（`verify_bad == 0`）。
- **性能**（同会话 `SWALLOW_STEP=0/1`）：
  - `[dspark] steps` 的 `verify=` **应 +1.6ms**（6 行 vs 5 行）——变大是**预期**，不是失败；
  - `[tick] total` 的整步 **应 −4.5ms**（主链步消失）——这才是 swallow 的判据；
  - `[verify_graph] captured verify_graph_m6` 必须出现（否则 graph 没接上 6 行块）。
- **反向判据（止损）**：若 `verify=` 没变（m=6 的 capture 没发生）或整步没降，先查 dispatch / graph 日志，
  **不要在没证据时出结论**（`final-400-config.md` R2 的"缺证据 exit 2"）。

### 5.4 顺序建议

1. 先把 A 组 mrows 的 dispatch 钉死（确认不是 H1）——否则 swallow 的 verify 基线本身是错的；
2. 再单跑 **A' 轮（swallow + SIDS_WRITEBACK，独立）**，拿到 −4.5ms/步 与正确性通过；
3. 最后做 **组合轮**：A ∪ A' ∪ B + tcgen05，量最终步时（目标 ≈ 9.2~10.2ms）。

---

## 6. 与既有文档的口径差异（诚实标注）

- 本文代码行号以 HEAD `51f2515` 为准（`dspark_spec_swallowed` = **:7253**；`swallow_step()` = **:2018**；
  `spec_primed` 分派 = **:6737**）。`final-400-config.md`/`dspark-swallow-step-diff.md` 里的
  `:6284`/`:1524`/`:1530`/`:1896` 是更早版本的锚点，已漂移，**读代码时以函数名为准**。
- `final-400-config.md` R2 的"m 从 5→6 必须重建形状池"**已被 per-shape 槽实现取代**（形状池自动支持两形状）——
  该风险项应改为"A/B 时确认 m=6 的 capture 日志"，而非"需要改代码重建"。

---

*工部 · 只读分析 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
