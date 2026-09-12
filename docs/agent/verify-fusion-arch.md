# verify 融合进 draft —— 架构设计与口径校正（含"逐行 lazy verify"）

> 中书省 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动源码。
> 基线：工作树含 `DSV41_TIMING` **verify = 37.31 ms（m=5）**、EAGER 主链步 **6.15 ms**。
> 输入：`final-400-battle.md` · `final-400-config.md` · `verify-ms-breakdown.md` ·
> `verify-calc-floor.md` · `dspark-swallow-step-diff.md` · `routed-expert-residual.md`。本机无 GPU ⇒ 所有 ms 标来源。
> 代码基线：`chain_dev.rs` / `dspark_dev.rs` / `spec_step.rs`（引用 `file:line` 逐条核对）。

---

## 0. 判决（先读这五条）

1. **"共享 draft 的 forward" 依旧不成立**（本文件 §2 用代码复核）。draft 跑的是 3 个 MTP 层
   （`dspark_dev.rs:784` `draft_forward`，`n_mtp_layers=3`，`config.rs:520`），出口是 `markov_head`；
   verify 跑的是 40 层 backbone（`chain_dev.rs:4067` `step_rows` → `:4470 step_rows_inner` → `:6805 layer_rows`），
   出口是 backbone `head`。**两套参数化，产出不可互换**。折叠 = 删掉验证（`spec_step.rs:91` 的 `spec_accept`）。

2. **"逐行 lazy verify" 是一个真实、高价值、且代码上更简洁的机制** —— 我把它完整设计出来了（§3）。
   它把 verify 从"整块的一次 forward"改成"**一行一次 m=1 forward，行数随 accept 早退**"。
   它的**最大收益出现在低 accept**（今天）：verify 37 → ~10.5 ms（**3.5×**），且**无快照/无回滚/无 compress_replay**（§3.4 的关键推导）。

3. **但 lazy 不是 400 的路径 —— 这是本文件对任务描述最重要的更正**（§4）。
   lazy 的成本 = `行数 × c`（c = 单行 forward ≈ 6.15 ms），而**行数 = 本步 emit 的 token 数**
   （swallow 布局：`k_emit = matched + 1`，见 `chain_dev.rs:6346`）。于是
   `tok/s = R × 1000 / (R×c + draft + commit) → 1000/c ≈ 162（tcgen05 后 ≈ 225）`——
   **lazy 的吞吐上限就是"一个 token 一次 forward"**，即 EAGER 的吞吐（162）。它**不可能**到 400。
   任务口径里 "1.5 行 → accept 3 → 214/366 tok/s" 把**当前 accept 的行数**与**目标 accept 的 token 数**混用了。

4. **400 的正确路径是"让 batched 的边际行成本掉到 ~1 ms"**（§5/§6）：把 6 行块的**可折叠族权重读一次**
   （`SH_EXP_MROWS` / `PROJ_MROWS` / `ROW_FOLD_GATE` / `b·m` sparse_attn）+ **routed expert 换核**（tcgen05）。
   这正是任务直觉里"共享"的**正确落点**——共享的是**权重读取**，不是 draft 的 MTP forward。
   `final-400-config.md:126` 的 `verify(m=6) ≈ 8.21 ms`（全 gate ON）+ P3c draft = 步时 ~9.6 ms ⇒
   `4 tok/step @10 ms ⇒ 417 tok/s` ✓（**这才是 400 的形态**）。

5. **两者正交，应共存**：lazy 是**低 accept 的自动快路**（`:crossover` ≈ 1.3 行，§4.4），
   batched 是**高 accept 的唯一解**。建议同一臂里按 accept 自动选路（§7 建议）。

---

## 1. 调研：本步到底在跑几个 forward（代码级）

### 1.1 三条 forward

| 路径 | 入口 | 计算 | 出口（产出分数） |
|---|---|---|---|
| **主链步** | `chain_dev.rs:3562 step_dev` → `:3590 step_impl` → `:3689 step_body` | 40 层 backbone，**1 行** | backbone `head` → `s.ids`（下一 token） |
| **draft** | `dspark_dev.rs:784 draft_forward` | **3 个 MTP 块**（`:942 for s in 0..n_mtp_layers`），每块 window attn + **自己的 MoE**（`:1535 draft_moe`），**bs=5 行** | `draft_head` → `markov_head` → `ids[1..=bs]`（`:696 drafts()`） |
| **verify** | `chain_dev.rs:4067 step_rows` | 40 层 backbone，**m 行批**（`:4470 step_rows_inner` → `:4551 for layer` → `:6805 layer_rows`） | backbone `head` + 逐行 argmax（返回 `Vec<u32>`） |

**draft 与 verify 是两套参数**：draft 的 5 行 hidden 是 **MTP 层**的（3 层，`dspark_dev.rs:56 DSPARK_TAP_SLOTS=3` 的 tap + 各 MTP 块自己的 MoE）；
verify 的 5 行 hidden 是 **backbone 40 层**的。**同一个 batch 形状，不同的模型。**
⇒ 任务描述"verify 只是重复 forward 拿 argmax"**前半句（形状）对，后半句（同一数学）错**。

### 1.2 为什么 verify 的 5 行不能折（final-400-battle 的主判；本文件复核）

`verify-calc-floor.md:54` + `verify-ms-breakdown.md:34`：verify 逐族账（m=5）里 **routed experts = 8.30 ms（22.2%）**，
字节 `40L × 30 assign × 2.6112 MB = 3.13 GB`——**30 个 assignment 是 30 个不同专家**（`routed-expert-residual.md:250`：384 选 6，去重仅 3.7%）。
**MoE 的定义就是 per-token 路由 ⇒ 字节随行数线性，batched 折不掉。**

> ⚠️ 但要**分清主次**：routed 只占 8.3/37 = 22%。真正没兑现的是**可折叠族**：
> shared expert 10.4（`verify-ms-breakdown.md:35`，`SH_EXP_MROWS` 默认 OFF，`chain_dev.rs:952`）、
> MoE gate 3.44（`ROW_FOLD_GATE` OFF，`chain_dev.rs:8090`）、投影 3.7、indexer 2.5、hc 2.96。
> 全部 gate ON 实测只 −1.21 ms（`verify-ms-breakdown.md:159`）⇒ **H1（mrows 未 dispatch）或 H2（延迟非字节）未定**（`final-400-battle.md:205`）。
> **这条未定项决定 lazy 与 batched 的相对收益，必须先钉死**（§8-R0）。

### 1.3 当前 verify 的行成本 ≈ 一行一次完整 forward

`37.31 / 6.15 = 6.07`：**5 行批 = 6× EAGER**。即**批处理今天几乎没有共享效果**
（理想应为 `6.15 + 4×marginal ≈ 12 ms`）。这给 lazy 提供了机会窗口：
**"既然批处理不共享，那么按行跑、只跑需要的行"就是严格更优的。**

---

## 2. 否决"共享 draft forward"（代码复核）

| 判据 | 证据 | 结论 |
|---|---|---|
| 入口不同 | `dspark_dev.rs:784` vs `chain_dev.rs:4067` | 两条链 |
| 层数不同 | draft：`n_mtp_layers=3`（`config.rs:520`）；verify：`cfg.n_layers=40` | **不同模型** |
| 出口不同 | draft：`markov_head.head.weight`；verify：backbone `head.weight`（129280×5120） | **不同参数化** |
| 折叠后果 | `spec_accept`（`spec_step.rs:91`）比 `drafts[i] == judges[i]`；换成 draft 自己的 argmax ⇒ `drafts[i]==drafts[i]` 恒真 ⇒ **accept 恒满 = 删掉正确性** | **不可折叠** |

**任务描述里的"共享 forward"直觉，正确落点只有一个：共享"权重读取"（§5），不是共享"模型"。**
`dspark-swallow-step-diff.md` 的 SWALLOW_STEP 是另一条**合法**的"消除独立 forward"——
把**主链步**折进 verify 的 anchor 行（净 −4.55 ms），本文件把它作为 lazy 的一部分统一（§3.5）。

---

## 3. 逐行 lazy verify：完整架构

### 3.1 行语义与早退判据（用 `spec_accept` 的**真实**代数）

采用 **swallow 布局**（`chain_dev.rs:6284 dspark_spec_swallowed` 的块）：块 = `[anchor, d1..d5]`，
`rows_in[i]` 在位置 `pos+i`，`rows[i] = argmax(row i)` 预测 `pos+1+i`；`drafts[i]` 提案 `pos+1+i`

⇒ **index 对齐**：`spec_accept(drafts, rows, /*anchor_is_in_block=*/true) = matched + 1`（`spec_step.rs:91-112`）。

```
matched     = 第一个 i 使 drafts[i] != rows[i]（0..=5；全中则 =5）
k_emit      = matched + 1                # 本步 emit 的 token 数 = 1..=6
rows_run    = matched + 1 = k_emit       # 逐行需要跑的行数（含 anchor 行 0）
keep        = k_emit                     # commit 保留的行数（见 §3.4）
tok/step    = k_emit                     # ← 这是关键：行数 == emit 的 token 数
```

- **早退**：跑到第一个 mismatch 的**那一行就停**（`rows[matched]` 是判断 mismatch 所需）。
- **最好情形** `drafts[0] != rows[0]` ⇒ `k_emit=1`，**只跑 anchor 行**（1 行，6.15 ms）。
- **最坏情形** 5 个 draft 全中 ⇒ `k_emit=6`，跑 6 行（36.9 ms）——**≈ 今天的 37 ms 批处理**。
- ⇒ **lazy 的代价上界 = 今天的 batched，均值远低**（§4）。**不会更差**。

### 3.2 逐行循环（落点级）

新增一个 arm（`DSV41_LAZY_VERIFY=1`，默认 OFF），它是 `dspark_spec_swallowed` 的**行循环版**：

```rust
// chain_dev.rs 新增：fn dspark_spec_lazy(&mut self, dspark, token, pos) -> Result<DsparkSpecReport>
// 结构 = dspark_spec_swallowed（行 6300 起）去掉"一次性 6 行块"，改成：
let mut rows: Vec<u32> = Vec::with_capacity(m);          // m = VERIFY_ROWS = 6
let mut rows_in: Vec<u32> = Vec::with_capacity(m);
rows_in.push(token);                                     // 行 0 = anchor（吞掉的主链步）
rows_in.extend_from_slice(&drafts);

let mut matched = 0usize;
for i in 0..m {
    // 逐行 forward：m=1，位置 pos+i。KV 因果链由顺序天然满足（§3.3）。
    self.spec_capture = true;
    let a = self.step_rows(&rows_in[i..=i])?;            // ← m=1 的单行 forward
    self.spec_capture = false;
    rows.push(a[0]);
    if i > 0 && rows_in[i] != a[0] { /* 注意索引：见下 */ }
    // 判据用 spec_accept 的代数：drafts[i] vs rows[i]
    if drafts.get(i).map(|d| *d != a[0]).unwrap_or(false) { break; }
    matched = i;
}
let k_emit = matched + 1;
```

**两处必须精确处理的索引**（`spec_step.rs:91` 的真实代数，别凭直觉）：

1. **row 0 的判据是 `drafts[0] != rows[0]`**，不是 `rows_in[0] != rows[0]`。
   `rows[0]`（anchor 对 pos+1 的预测）与 `drafts[0]`（draft 对 pos+1 的提案）**同级**。
   ⇒ 若 `drafts[0] != rows[0]`，`matched=0, k_emit=1`，**只跑 anchor 行**。
2. **`rows[i]` 判 `drafts[i]`，行 i 喂的是 `rows_in[i] = (i==0 ? token : drafts[i-1])`**。
   ⇒ 行 i 喂的 token 是"draft 的第 i-1 个提案"，**它在 `matched>=i` 时才是真 token**（§3.4 靠这个做"零回滚"）。

### 3.3 KV 因果链（为什么逐行天然正确）

- `step_rows`（`:4013` 文档）:"row r sees `[pos_base+r-window+1, pos_base+r]`, which includes the block's own rows `0..r` through the ring geometry"。
  ⇒ **行 r 的 attention 只依赖行 `0..r-1` 已写入 ring 的 KV**。
- 逐行按 `i = 0,1,2,...` 顺序跑 ⇒ 行 `i` 跑时，行 `0..i-1` 的 KV 已在 ring 里 ⇒ **因果链天然满足，无需任何额外同步**。
- **批处理 vs 逐行在位级等价**：两者对每行做的是同一套 per-row 算子（`layer_rows(m=1)` 是 `layer_rows(m=5)` 的退化），
  行的独立性由 `sparse_attn` 的 `grid=(b*m,h)`（`chain_dev.rs:6482` 附近）与 `moe_rows` 的 `grid.z=rows`
  （`:8064`，`rows==1 degenerates to previous launch`）保证。**`dspark_parity` 的 verify 行级对照是验收判据**（D4，§9）。

### 3.4 快照 / 回滚 / commit —— **lazy 的最大结构性红利：零回滚**

**推导（本文件最重要的机制发现）**：keep (= k_emit) 恰好等于 rows_run。

- 行 i 写入 ring 位置 `pos+i`，喂的 token 是 `rows_in[i]`（i=0 是 anchor token，i≥1 是 `drafts[i-1]`）。
- **行 i 被 KEEP 当且仅当 `rows_in[i]` 是真 token** ⇔ `i=0` 或 `drafts[i-1]` 被接受 ⇔ `i <= matched`（= rows_run−1）。
- ⇒ **所有跑过的行（0..matched）都在 keep 范围内**；**没跑的行从来没写过**。
- ⇒ **lazy 不需要 `dspark_snapshot` / `dspark_rollback_keep` / `compress_replay`**（对比 batched 的 `:6300/:6302/:6359`）。

**附带**：compressor 状态在跑完 `k_emit` 行后 = "恰好 k_emit 次单行 decode 之后"的状态
（逐行的 `compressor_pool_on` + `compress_commit_on` 与 `compress_replay` 用的**是同一对单行算子**，`chain_dev.rs:6581/:6598`）
⇒ **无需 replay**。**这是 lazy 相对 batched 在"正确性侧"的净收益**（batched 必须 snapshot→rollback→replay）。

**commit 落地**（与 batched 共用 `dspark_commit`，`chain_dev.rs:6513`）：

```rust
self.dspark_commit(pos, m, /*keep=*/k_emit, /*host=*/&[])?;   // host 传空 ⇒ rollback_keep 早退（:4938）
dspark.note_ctx_rows(self.s.dspark_tap_r.ptr as *const f32, k_emit, k_emit, pos)?;  // 回填 draft 的 window ring
Self::carry_kept_tap(self.dev, self.s.dspark_tap.ptr, ..., k_emit)?;                // 下一轮 draft 的 tap
```

> **注**：`dspark_commit` 无条件调 `dspark_rollback_keep(pos, m, keep, host)`；`host.is_empty()` 时它**早退**（`:4938-4940`）。
> lazy 传 `host = &[]` ⇒ 零回滚、零拷贝。**唯一要改的是把 `m` 传成 6 而 `keep=k_emit`**（`rollback_keep` 里 `keep>=m` 是合法 no-op，但我们本来也不需要它做事）。

**错误路径**：若循环中途 `step_rows` 返回 `Err`，此时环里已有 `i` 行的写入。**lazy 的兜底 = 让下一轮走 `spec_primed=false` 的 legacy 引导路**（`spec_primed` 只在成功尾部置位，`:5978`），
但**已写入的 KV 需要还原**——所以**保留一次"循环前快照"仅用于错误路径**（正常路径不用它，`host` 传空即绕过）。 이것은 §8-R3。

### 3.5 与 swallow step 的统一

- **lazy ⇔ swallow 的自然结合**：swallow 的块 `[anchor, d1..d5]` 的**行 0 就是被吞掉的主链步**
  （`chain_dev.rs:6216-6220`），lazy 的循环第 0 次迭代就是它 ⇒ **lazy 天然吞主链步**，无需额外分支。
  `dspark_spec_lazy` 直接取代 `dspark_spec_swallowed` 的"一次性 6 行块"，**主链步成本完全消失**（−6.15 ms）。
- **legacy（不 swallow）的 lazy 版本**（可选）：块 `[d1..d5]`，`rows_run = k_acc`（**可以为 0 行**：
  `drafts[0]!=next` ⇒ 不跑任何 verify 行，只 emit `next`）。但因为它仍要跑一次独立 `step_dev`（6.15 ms），
  **总成本 = 6.15 + k_acc×6.15 = swallow-lazy 的 (1+k_acc)×6.15 —— 完全相同**。⇒ **只做 swallow-lazy 一条路**，少一个语义分支。
- **tap 的跨轮传递**：沿用 `carry_kept_tap(keep=k_emit)`（`:6456`）——行 `k_emit-1`（最后一个 KEPT 行）
  是"最后一个已提交位置的 true hidden"（`:6441-6446`），lazy 下与 batched 同一规则。
- **`note_ctx_rows` 基址**：swallow 的 `pos`（`:6360`），lazy 不变。

### 3.6 CUDA graph / AR / TP 影响（必须先看的三条）

| 面 | 影响 | 处理 |
|---|---|---|
| **VERIFY_GRAPH**（`chain_dev.rs:4282 verify_graph_gate`） | lazy **永远以 m=1 调 `step_rows`** ⇒ 形状池只用 slot(m=1)（`verify_slot` `:4273`）；DRY 一次、capture 一次、之后 replay | 形状池按 m=1 建；**与 batched 的 m=6 slot 不冲突**（各自 slot）。⚠️ `pos_rows`/`ids` 每行刷新在 capture **之外**（`:4094-4099`）⇒ 合法 |
| **AR v5**（无 host rendezvous） | 每行发 1 次 argmax 的 v5 round（`argmax_sliced_rows`）+ attn/MoE 的 AR ⇒ 每步 AR 数 = `k_emit ×`（行内 AR 数）。**各行 rank 必须发相同的 AR 数** | `k_emit` 由 argmax 决定，而 argmax 是**跨 rank 归约后的同一值**（`dsv41_argmax_sliced_rows` 一次 v5 round）⇒ 各 rank 决策相同 ⇒ AR 足迹一致。**但这是 AR-v5 死锁事故的高危区**（`final-400-config.md:264` R3-④）：**必须同会话 A/B 验证**（§9） |
| **`spec_capture`**（`:1871`，控制 tap 收集 `:6927/:7848`） | 必须**逐行**包住（每行的 `layer_rows` 都要收 tap） | `for` 循环体内 set/clear（§3.2 伪码已示） |

**额外成本（诚实计账）**：每行一次 D2H（4 B argmax）+ 一次 H2D（ids/pos）+ 一次 host 分支 ⇒ ~几 µs/行；
k_emit 次 argmax v5 round（~17.3 µs/次，`verify-ms-breakdown.md:19`）⇒ `k_emit×17.3µs` ≈ 0.1 ms。**相对 6.15 ms/行可忽略。**

---

## 4. 性能账（**两条口径并列**：任务口径 vs 严格口径）

### 4.1 严格模型（本文件主张）

```
c        = 单行 (m=1) backbone forward 成本 ≈ 6.15 ms              （EAGER，实测）
c_tc     = c − 1.7 = 4.45 ms                                      （tcgen05：routed 8.3→1.5，按行摊 1.7/行）
R        = 平均跑的行数 = 平均 k_emit = 1 + mean_k               （swallow 布局）
verify   = R × c          （lazy）    |    = B(m) ≈ const（batched，实测 m=5: 37 ms）
step     = verify + draft + commit
tok/s    = R × 1000 / step                                        （tok/step = R）
```

**关键不等式**：`lazy step ≥ R×c` ⇒ `tok/s ≤ 1000/c`。
- `c = 6.15` ⇒ **lazy 上限 162 tok/s**
- `c_tc = 4.45` ⇒ **lazy+tcgen05 上限 225 tok/s**
- ⇒ **lazy 在物理上不可能到 400**（这是 §0-3 的算术）。

### 4.2 对照表（R 取"今天"与"accept 3"两档）

| 方案 | verify | draft | commit | 步时 | tok/step | **tok/s** |
|---|---:|---:|---:|---:|---:|---:|
| **现状 batched m=6** | 37.0+1.6 | 4.3 | 0.3 | ~39 | R | R/0.039 |
| **lazy（今天 R≈1.7）** | 10.5 | 4.3 | ~0 | **14.8** | 1.7 | **115** |
| **lazy（accept 3, R=4）** | 24.6 | 4.3 | ~0 | **28.9** | 4 | **138** |
| **lazy+tcgen05（今天 R≈1.7）** | 7.6 | 4.3 | ~0 | **11.9** | 1.7 | **143** |
| **lazy+tcgen05（accept 3, R=4）** | 17.8 | 4.3 | ~0 | **22.1** | 4 | **181** |
| **batched+全opts+tcgen05（accept 3）** | 8.2 | 3.6 | 0.2 | **12.0** | 4 | **333** |
| **batched+全opts+tcgen05+P3c** | 8.2 | 1.2 | 0.2 | **9.6** | 4 | **417** ✓ |

**读法**：
- **lazy 的相对收益 = 37 / (R×c)**。今天 `R≈1.7` ⇒ **3.5×**（41 → 115 tok/s 的 verify 部分）；
  accept 3 时 `R=4` ⇒ 只剩 1.5×（37 → 24.6 ms）。**lazy 的收益随 accept 单调下降**。
- **batched 的破口**：今天 37 ms ≈ 5×6.15（**没共享**）；全 opts 后 8.2 ms（**共享了**）⇒ 边际行成本 8.2→~0.4 ms/行。
  **330~420 tok/s 全部落在 batched 侧**。

### 4.3 任务口径（照录，并指出其内部不一致）

| 方案 | verify | 步时 | tok/s @accept3 |
|---|---:|---:|---:|
| tcgen05 only | ~31 | ~36 | 83 |
| lazy | ~9.2 | ~14 | 214 |
| lazy+tcgen05 | ~6.7 | ~8.2 | 366 |

**不一致点**："lazy 9.2 ms = 1.5 行 × 6.15" 用的是**当前 accept 的行数（R≈1.5）**，
而 "accept 3 → 214 tok/s" 用的是**目标 accept 的 token 数（3）**。
**同一格的两个数来自不同的 R**。若统一到 accept 3（R=4）：verify = 24.6 ms（lazy）或 17.8 ms（+tcgen05），
落点 **138 / 181 tok/s**，**均 < 400**。**建议在拿到 §10 的口径确认前，不使用 214/366 这两个数。**

### 4.4 lazy vs batched 的交叉点（选路判据）

lazy 仅在 `R×c < B` 时优于 batched。取 `B = 8.2 ms`（batched 全 opts，m=6）、`c = 4.45 ms`（tcgen05）：

```
R < B/c = 8.2 / 4.45 ≈ 1.84   ⇒   mean_k < 0.84
```

- **今天**（mean_k ≈ 0.7~0.83，`final-400-config.md:19,187`）**卡在交叉点附近** ⇒ lazy ≈ batched（都 ~140 tok/s）。
- **若不做 mrows（B 停在 37 ms）** ⇒ `R < 6` ⇒ **lazy 永远胜**（37 → R×c）。
- **若 mrows 兑现（B = 8.2）且 accept 升到 3** ⇒ batched 完胜（417 vs 181）。
⇒ **选路应看两个运行时量：`B`（mrows 是否 dispatch）与 `mean_k`（accept）**。§7 给出自动选路建议。

---

## 5. 对比方案：tcgen05（对 routed experts 的 −6.8 ms）

- **它是什么**：把 `routed experts` 的 30 个 per-slot 小 GEMV 变成几个 M=128 的 masked GEMM
  （`dsv41_experts_mxf4.cu:3805` `tc5::mxf4`，`build.sh:89` 默认编进 .so，Rust dispatch `chain_dev.rs:10803-10809` 已接线）。
- **落点**：`routed 8.30 → ~1.50 ms`（`final-400-config.md:72`）⇒ **verify −6.8 ms**（batched）。
- **对 lazy 的意义**：routed 是**纯 per-row**项 ⇒ lazy 的**每行**都省 1.7 ms ⇒ `c_tc = 4.45`（§4.1）。
  **这是 lazy 与 tcgen05 的天然叠加点**（tcgen05 修的正是 lazy 每行都要重付的那部分）。
- **互斥风险（必须先澄清）**：`DSV41_TCGEN05_GATEUP_MXF4` 是 **e2m1×e2m1**（`chain_dev.rs:687`），
  而今天的**正确性达标**靠 **e4m3 单趟**（`chain_dev.rs:696 expert_act_e4m3`，F1）。
  ⇒ **开 tcgen05 就要回 e2m1 激活 = 回到 opa 问题**。**必须先确认 tcgen05 支持 e4m3 激活**（改 `act_e4m3` 分支），
  否则 §4.2 表里所有 `c_tc` 行都不可用（`final-400-battle.md:300` R1）。

---

## 6. 联合方案（任务口径 vs 严格口径）

**任务口径**：`1.5 行 × (6.15−1.7) = 6.7 ms` + draft P3c 1.2 + commit 0.3 = **8.2 ms → 366 tok/s**。

**严格口径**（R 必须跟 accept 走，§4.1）：
```
accept 3 ⇒ R = 4:
  lazy + tcgen05        = 4 × 4.45           = 17.8 ms
  + draft(P3c) 1.2 + commit 0.2               = 19.2 ms   → 4 tok/step → 208 tok/s
batched + 全opts + tcgen05 + P3c:
  verify(m=6) 8.2 + draft 1.2 + commit 0.2    =  9.6 ms   → 4 tok/step → 417 tok/s  ✓ 400
```

⇒ **联合方案里，lazy 与 batched 不是"叠加"，而是"二选一"**（同一块行只能跑一次）。
`lazy+tcgen05` 是**低 accept 的快路**；`batched+tcgen05+P3c` 是**高 accept 的 400 路**。

---

## 7. 建议的最终形态（自动选路，二者共存）

```
DSV41_LAZY_VERIFY=1（默认 OFF）时，dspark_spec_step 的后续轮按运行时量选臂：
  if R_est × c < B_now   →  dspark_spec_lazy     # 低 accept / mrows 未兑现
  else                   →  dspark_spec_swallowed  # 高 accept / mrows 已兑现
  （R_est = 1 + 上一步的 k_acc 滑动均值；B_now = 上一步 verify_ms；c = 上一步 verify_ms/R）
```
- 两臂**共用** `dspark_commit` / `note_ctx_rows` / `carry_kept_tap` / `spec_accept`（§3.4/§3.5）⇒ 语义只有一个。
- **`spec_primed` 引导**（`:5798`）对两臂一致：首轮走 legacy（供 tap），后续轮按上述选路。
- **AR 足迹**：两臂每步的 AR 数**不同**（lazy = k_emit×行内、batched = 1 块）⇒
  **一旦选定某臂，本请求应保持该臂到结束**（或至少所有 rank 同步切换），否则 AR-v5 计数错位（§8-R2）。**这是选路策略的硬约束。**

---

## 8. 影响范围与风险

### 8.1 影响范围
- **修改文件**：`crates/ferrite-models/src/dsv41/chain_dev.rs`（新 arm `dspark_spec_lazy` + 选路）；
  若启用 lazy 的 `spec_capture` 行循环，**不动** `dspark_dev.rs`（draft/commit 接口不变）。
- **影响模块**：DSpark spec 步、verify CUDA graph 形状池、AR v5 足迹、compressor 提交路径。
- **兼容性**：**无 API breaking change**（纯 env gate + 新 arm，`=0` 回退旧路）。

### 8.2 风险清单

| ID | 风险 | 应对 |
|---|---|---|
| **R0** | **H1/H2 未定**（`final-400-battle.md:205`）：mrows 到底 dispatch 没有？决定 `B` 是 8.2 还是 37，**进而决定 lazy 是"永远胜"还是"只在低 accept 胜"** | **先跑 §9-T0 的 nsys 单次诊断**，不投 lazy 的集成工作直到 `B` 钉死 |
| **R1** | **tcgen05 × e4m3 互斥**（`chain_dev.rs:687`）——`c_tc` 全表的前提 | 先确认 tcgen05 支持 e4m3 激活（§5）；不支持则 lazy 只按 `c=6.15` 计（上限 162） |
| **R2** | **AR v5 足迹**：lazy 每步 AR 数 = `k_emit×` 行内，随 accept 变；**换臂会改变足迹** | 选定臂后**本请求不换臂**；同会话 A/B 验证 `dspark_parity` + 无死锁 |
| **R3** | **错误路径的半写状态**：lazy 循环中途 `Err` 时环里已写 `i` 行 | 保留**一次循环前快照**仅服务错误路径（正常路径 `host=&[]` 绕过）；或直接令该请求退回 legacy 并 rollback |
| **R4** | **逐行 vs 批的行级位一致性**未证（`dspark_parity`） | **判据 = `dspark_parity` verify 行级对照（`verify_bad==0`）**；bs=1/m=1 退化用例 + bs=5 |
| **R5** | **VERIFY_GRAPH 的 m=1 slot**：`pos_rows` residue 与 capture 冻结（`chain_dev.rs:2748`） | m=1 形状单独 slot；先 DRY 再 capture；A/B 判据含 `[verify_graph] captured m=1` 证据 |
| **R6** | **口径未确认**（§10）：accept 语义、mean-k 真值 | 未确认前**不把 214/366 写进任何进度承诺** |

---

## 9. 验证计划（最少 GPU 次数 · 单一测试驱动）

**铁律**（`dspark-correctness-chain.md`）：同远端**同时只有一个测试驱动**；subagent 只做代码/分析；
后台输出自动注入，**不轮询远端**。全部 A/B 同会话同 .so。

| # | 会话 | 内容 | 判据 |
|---|---|---|---|
| **T0** | GPU（1 次，**最高优先级**） | nsys 诊断 mrows dispatch（`final-400-battle.md:223` 的四个 kernel 名） | 名字在→H2，不在→H1；落定 `B` |
| **T1** | GPU（1 次） | `DSV41_LAZY_VERIFY=0/1` 同会话 A/B（低 accept 载荷） | `verify=` 应 ↓3×；`[dspark] verify` 字段；出师表逐字 + `has_double_char` |
| **T2** | GPU | `dspark_parity` verify 行级对照（`LAZY` vs batched） | `verify_bad == 0`（逐位） |
| **T3** | GPU | AR 足迹验证（TP8、变 accept） | 无死锁、无 epoch 错读（`DSV41_INV_CHECK=1` 全绿） |
| **T4** | GPU | tcgen05 单层微基准门（`scripts/dsv41_tcgen05_mxf4_verify.sh --step 3`） | gateup 22.2µs / down 17.2µs；**不达标即止损**，不进集成 |

---

## 10. 需要太子补充的信息（阻塞 §4 的定稿）

1. **accept 口径**：`accept 3` = **mean-k = 3**（⇒ tok/step = k_emit = 4，lazy 跑 4 行）还是 **tok/step = 3**？
   两者差 1 行 × 6.15 ms，**直接决定 §4.2 表的落点**。
2. **mean-k 现状真值**：`0.66`（本任务）还是 `0.83`（`final-400-config.md:19`）？决定 lazy 的 `R`（1.66 vs 1.83）。
3. **目标 accept 的到达路径**：draft 质量（P1，`final-400-config.md:226`）**何时能到 3**？
   lazy 的收益与它**反向**（§4.4）——若 accept 长期停在 ~0.8，lazy 应是**主路**；若 accept 会到 3，lazy 只是**过渡**。
4. **`B` 的真值**：先要 §9-T0，否则 §7 的选路判据无法定阈值。

---

## 11. 建议分工（依 §3 的实现面）

- **工部**：`dspark_spec_lazy` 的行循环 + 选路（`chain_dev.rs`，~150 行，参照 `dspark_spec_swallowed` 重写块内 `step_rows` 为循环）；
  `dspark_commit(keep=k_emit, host=&[])` 的调用点。**原因**：纯 Rust 机械改造，无新 kernel。
- **户部**：T0 的 nsys 单次诊断 + `B`/`c`/`R` 的实测口径（§4.4 交叉点）；§4.2 表的落点复核。**原因**：性能账是户部本职，且 T0 决定选路。
- **刑部**：`dspark_parity` verify 行级对照（逐位）+ 边界（bs=1 / m=1 / k_emit=1/6 / 换臂）。**原因**：lazy 的位一致性是正确性红线。
- **兵部**：AR v5 足迹的 rank 一致性审查（`tp.rs` 的 `ar_v5` 计数）+ 半写状态的回滚安全。**原因**：AR-v5 死锁是历史高危。
- **礼部**：本文件 + `NEXT-SESSION-HANDOVER.md` 的同步（口径校正、选路判据）。**原因**：避免 214/366 被当成已验证的数。
- **吏部**：`spec_capture` 行循环的 flag 卫生（逐行 set/clear）+ 新 gate 默认值翻转的读回确认（`final-400-config.md:270` R9）。**原因**：v13→v15 的 0.37 ms 误诊教训。
- **尚书省**：不分配（本文件是设计，无跨部门依赖的大工程）。

---

## 附：一句话总结

**"verify 融合进 draft" 的正确形态有两层**：
(1) **权重融合**（真正的 400 路径）——6 行块的可折叠族权重读一次（mrows）+ routed 换核（tcgen05），把 batched 的边际行成本压到 ~1 ms；
(2) **时机融合**（今天的 3.5× 路径）——**逐行 lazy + swallow**：按行跑、遇拒即停、**零回滚**，代价上界 = 今天的 batched。
**两者正交，应共存于同一臂的自动选路里。lazy 不是 400 的答案；它是把 37 ms 拆成"只跑需要的行"的答案。**

---
*中书省 · 只读分析 + 本文件（唯一产出），未执行 GPU 命令、未改动源码。*
*所有 ms 标来源（实测/推算）；任务口径与严格口径的差异已在 §4.3 显式列出。*
