# lazy ⇄ batched 动态选路 gate —— 架构设计

> 中书省 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动源码。
> 上游判定：`docs/agent/verify-fusion-arch.md`（lazy 低 accept 胜 3.5×；batched 高 accept 唯一解，417 tok/s 路径）。
> 代码基线：`crates/ferrite-models/src/dsv41/chain_dev.rs` @ HEAD，逐条 `file:line` 核对。

---

## 0. 判决（先读这八条）

1. **动态选路可行，且两臂共用一个语义核心**：swallow 布局（块 `[anchor, d1..d5]`）、`spec_accept(anchor_is_in_block=true)`（`spec_step.rs:91`）、`carry_kept_tap(keep=k_emit)`（`chain_dev.rs:6456`）。选路新增的唯一状态是**每请求的 mean-k 滑动窗口**。

2. **形状池天然支持两臂共存 —— 无需改池**：`VERIFY_GRAPH_SLOTS = 2`（`:107`），池按 `m` 键控（`verify_slot` `:4273`）。lazy 只用 `m=1` 槽、batched 只用 `m=6` 槽 ⇒ **恰好两槽，互不驱逐**。

3. **逐步自适应（而非每请求锁臂）是安全的**：选路基数 `k_acc` 来自**跨 rank 归约的 argmax**（`dsv41_argmax_sliced_rows`），所有 rank 得到同一整数 ⇒ 同一滑动窗口 ⇒ 同一决策 ⇒ AR v5 足迹逐轮对齐。**这是对 verify-fusion-arch §7「选定臂后本请求不换臂」的更正**——只要决策是确定性整数函数（不是浮点本地计时），换臂安全（见 §4.3）。

4. **lazy 的 commit 与 batched 的 commit 语义不同（关键修正）**：
   - batched：`step_rows` 内**逐行提交** compressor（`compress_row` `:7885`）→ `dspark_commit` 先 `dspark_rollback_keep` 把 compressor **整块回滚**（`:4986` `dspark_comp_restore`）→ `compress_replay` 只 replay keep 行（`:6559`）。三段缺一不可。
   - lazy：逐行 `step_rows` 提交的行**全在 keep 范围**（§3.4 推导）⇒ 提交即终态 ⇒ **禁止调 `compress_replay`**，否则 compressor 双重提交。
   - ⇒ verify-fusion-arch §3.4 的伪码 `self.dspark_commit(pos, m, k_emit, host=&[])` **会在 `:6530` 触发 `compress_replay(pos, k_emit)` ⇒ bug**。必须改用 §3.3 的 `dspark_commit_lazy`。

5. **lazy 的 m=1 verify 有两处"运行期量"必须显式校正**（此前文档未提）：
   - **位置计数器**：`step_rows` 从**设备计数器**读 `pos_base`（`:4091`）并把整块建立在 `pos_base..pos_base+m-1`。m=1 时若计数器停在 `pos`，每行都会在 `pos` 前向 ⇒ 必须**逐行推进计数器** `pos+i`。
   - **tap 行号**：`hc_collapse` 对 m=1 块**永远写 slot 的第 0 行**（`:6939-6946`，基址 `slot*VERIFY_ROWS*dim`）。而 `note_ctx_rows`（`dspark_dev.rs:647`）与 `carry_kept_tap` 按**行号 j** 索引 `(slot*VERIFY_ROWS+j)*dim`。⇒ m=1 逐行会把每行的 tap 覆盖在 index 0，**行 1..k_emit-1 丢失**。修法见 §3.2（deferred tap）。

6. **lazy 的净红利 = batched 的净成本**：无 rollback（`host=&[]` 早退 `:4938`）、**无 replay**、无 snapshot（成功路径；快照仅服务错误路径）⇒ verify-fusion-arch §3.4 说对一半（"零回滚"），漏了 compressor 的"零 replay"。

7. **选路判据落为 mean-k 阈值**：`lazy` 成本 `R×c`（`R=k_emit=1+mean_k`）、`batched` 成本 `B` ⇒ `lazy iff (1+mean_k) < B/c` ⇔ **`mean_k < τ`, `τ = B/c − 1`**。数值：`B=37,c=6.15 ⇒ τ≈5.0`（永远 lazy）；`B=8.2,c=6.15 ⇒ τ≈0.33`；`B=8.2,c=4.45 ⇒ τ≈0.84`（与 verify-fusion-arch §4.4 一致）。

8. **B 与 c 是"每进程常量"**（mrows 是否 dispatch 是 `.so` 属性；EAGER 步时也是常量）⇒ τ 可一次标定；**动态的只是 mean-k**。B 的刷新点 = 启动轮（legacy/batched）与任何被选中的 batched 轮。

---

## 1. 现状分析

### 1.1 三条 arm 的落点（当前代码）

| arm | 落点 | 块布局 | verify 调用 | commit |
|---|---|---|---|---|
| legacy | `dspark_spec_step` `:5827` 起 | `[d1..d5]` @ `pos_ctr+1..` | `step_rows(&drafts)` m=5（`:5858`） | snapshot→rollback→replay |
| aligned | `dspark_spec_aligned` `:5994` | `[next,d1..d5]` @ `pos+1..` | `step_rows(&rows_in)` m=6（`:6107`） | 同上 |
| **swallow（batched 臂）** | `dspark_spec_swallowed` **`:6284`** | `[anchor,d1..d5]` @ `pos..pos+5` | `step_rows(&rows_in)` **m=6**（`:6325`） | `dspark_commit(pos,m,k_emit,&host)`（`:6359`） |

分派点：`dspark_spec_step` `:5798`：
```rust
if swallow_step() && self.spec_primed {
    return self.dspark_spec_swallowed(dspark, token, pos);
}
```
`swallow_step()`（`:1524`）与 `seed_align()`（`:1570`）都是 `OnceLock` 缓存的 env gate（默认 OFF）。**lazy 应挂在同一分派点**，不新增第 4 条独立入口。

### 1.2 形状池与图化（lazy 的兼容性基础）

- `step_rows`（`:4067`）：`m = toks.len()`，`m=1` 合法（≤ `VERIFY_ROWS=6`）。每个 verify 调用：H2D `ids_r`/`pos_rows`（`:4098-4099`）→ `verify_graph_gate(m,pos_base)`（`:4105`）。
- 池：`verify_slot(m)`（`:4273`）——先找 `shapes==m` 的槽，否则找空的槽。DRY→CAPTURE→REPLAY 每形状一次（`:4106-4178`）。
- ⇒ **两臂 = 两个形状 = 两个槽**。lazy 的 m=1 图与 batched 的 m=6 图并存，A/B 日志分别打印 `[verify_graph] captured verify_graph_m1 / _m6`（`:4185`）。
- ⚠️ 池只有 2 槽：若 `dspark_parity` 自检再引入第三个 `m`，它只能走 direct 路径（不致命，但会静默丢掉该形状的图化——A/B 时看 `[verify_graph]` 行确认）。

### 1.3 commit / compressor 的两种语义（lazy 必须区分的根因）

一次 `step_rows` 在 `spec_capture` 下会（每 comp-source 层、每行）：
1. `compress_proj_rows`（`:7806`）——投影 kvp/scp，并在 `spec_capture` 时把本层各行存进 `spec_snap_kvp/scp`（`:7848`）；
2. `compress_row`（`:7885`）——`compressor_pool_on` + `compress_commit_on`，**就地提交**该行 compressor。

⇒ batched 的 `dspark_commit(pos,m,keep,host)`（`:6513`）必须：`dspark_rollback_keep`（把 compressor 整块恢复到 pre-verify，`:4986`）→ `compress_replay(pos,keep)`（用 `spec_snap_*` replay keep 行，`:6559`）→ `set_pos_ctr(pos+keep)`。**净结果 = 只提交 keep 行**。

⇒ lazy 逐行 `step_rows(m=1)` 时，每行提交一次、顺序提交，**净结果天然 = 只提交 `0..k_emit`**（§3.4）。所以 lazy **不需要** rollback 与 replay；多调一次 `compress_replay` 会**双倍提交**（compressor 的 `state_kv/state_score` carry 前进两步/pos）。

---

## 2. 方案 · 选路 gate

### 2.1 gate 与环境变量

| env | 默认 | 语义 |
|---|---|---|
| `DSV41_LAZY_VERIFY` | `0`（OFF） | master gate。ON 时启用路由；OFF 时行为与今天逐字一致（`swallow_step()` 决定 batched 臂）。 |
| `DSV41_LAZY_THRESHOLD` | `auto` | `auto` ⇒ `τ = B/c − 1`（下式）；显式浮点 ⇒ 直接作为 `τ`（mean-k 阈值）。 |

实现照抄本文件已有的 gate 模式（`OnceLock`，进程内读一次）：
```rust
fn lazy_verify() -> bool {                       // 仿 swallow_step() :1524
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_LAZY_VERIFY").map(|v| v != "0").unwrap_or(false))
}
fn lazy_threshold_override() -> Option<f32> {    // "auto" ⇒ None
    static F: OnceLock<Option<f32>> = OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_LAZY_THRESHOLD").ok()
        .filter(|s| s != "auto").and_then(|s| s.parse().ok()))
}
```
**lazy 蕴含 swallow**：`DSV41_LAZY_VERIFY=1` 时 batched 半仍走 `dspark_spec_swallowed`（swallow-lazy 是唯一有意义的 lazy，见 verify-fusion-arch §3.5）。分派点改为：

```rust
// chain_dev.rs :5798 附近
let swallow = swallow_step() || lazy_verify();
if swallow && self.spec_primed {
    if lazy_verify() && self.lazy_route_decide() {
        return self.dspark_spec_lazy(dspark, token, pos);      // §3
    }
    return self.dspark_spec_swallowed(dspark, token, pos);      // batched 臂
}
```

### 2.2 运行时量：mean_k / B / c

| 量 | 定义 | 来源 / 落点 |
|---|---|---|
| `mean_k` | 最近 **32** 步 `k_acc` 的均值（整数累加，**无浮点**） | 每请求状态；每轮 `spec_step` 后 push `report.k_acc` |
| `B` | batched verify 步时常量（ms） | **每进程** `OnceLock<f32>`；启动轮（legacy/batched）与任何被选中的 batched 轮的 `report.verify_ms` 更新 |
| `c` | EAGER 单行步时常量（ms） | 每进程常量，默认 `6.15`（STATUS.md 基线）；可由启动轮 `step_dev` 实测细化 |
| `τ` | `B/c − 1`（override 时可覆盖） | `OnceLock<f32>`，进程内一次标定 |

**为什么 mean_k 必须是整数累加**：选路决策要在所有 rank 上**逐位相同**。`k_acc` 是跨 rank 归约后的整数（argmax 一次 v5 round 的同一结果）⇒ 每 rank 的窗口内容逐元素相同 ⇒ `mean_k = sum/len` 相同（避免任何本地浮点计时进入判据）。`B`/`c` 存在**进程级** `OnceLock`（rank 是同一进程的线程，见 `:4180` 注释）⇒ 全 rank 共享同一 τ。

### 2.3 阈值与滞回（Schmitt 触发）

交叉点附近抖动会导致逐轮换臂（图/AR 足迹抖动）。用双阈值：

```
τ_hi = τ + δ      # 从 lazy 切回 batched 的门槛
τ_lo = τ − δ      # 从 batched 切到 lazy 的门槛
δ = 0.25（mean_k 单位）
```

当前臂为**粘滞**状态（`lazy_mode ∈ {Lazy, Batched}`），只在越过对应门槛时翻转：

```rust
fn lazy_route_decide(&mut self) -> bool {
    let tau = self.lazy_tau();                 // override 或 B/c-1
    let mk  = self.lazy_hist.mean_k();         // 整数窗口
    let (lo, hi) = (tau - HYST, tau + HYST);
    let use_lazy = match self.lazy_mode {
        None            => mk < tau,           // 冷启动：单阈值，样本≥1 即可
        Some(Lazy)      => mk < hi,            // 粘滞：回到 batched 需 mk ≥ hi
        Some(Batched)   => mk < lo,            // 粘滞：切到 lazy 需 mk < lo
    };
    self.lazy_mode = Some(if use_lazy { Arm::Lazy } else { Arm::Batched });
    use_lazy
}
```

冷启动：本请求第 1 轮永远是 legacy 引导（`spec_primed` 仅在成功尾部置位，`:5979`），它给窗口 1 个样本、给 B 一个样本（`verify_ms`）。第 2 轮起路由生效。若 `B` 仍未知（极端：`DSV41_VERIFY_GRAPH` 等使 batched 轮不可用），`τ` 退回 c=6.15 的保守常量。

### 2.4 状态与复位

`DevChain` 新增（`:1869` 附近，与 `spec_capture`/`spec_primed` 同区）：

```rust
lazy_mode: Option<Arm>,           // None=未决策；Some 为粘滞臂
lazy_hist: LazyHist,              // [u8;32] 环形 + len + sum_u32
```
```rust
struct LazyHist { buf: [u8; 32], head: usize, len: usize, sum: u32 }
impl LazyHist {
    fn push(&mut self, k_acc: usize) {
        let k = k_acc.min(31) as u8;
        if self.len == self.buf.len() { self.sum -= self.buf[self.head] as u32; } else { self.len += 1; }
        self.buf[self.head] = k; self.sum += k as u32;
        self.head = (self.head + 1) % self.buf.len();
    }
    fn mean_k(&self) -> f32 { if self.len == 0 { 0.0 } else { self.sum as f32 / self.len as f32 } }
}
```

**复位**：`reset()`（`:2695`，已复位 `verify_shapes`/`verify_graph_failed`/`verify_dry_done` 于 `:2759-2761`）追加 `lazy_mode = None; lazy_hist = LazyHist::default();`。每请求冷启动与 `spec_primed=false`（`:2764`）一致。`lazy_mode` 与 `lazy_hist` **必须与 `spec_primed` 同生命周期**：一轮失败退回 legacy 时，窗口不重置（历史仍有效），但 `lazy_mode` 保留（粘滞）。

**窗口更新点**：在 `spec_step` 的成功返回处（而非 arm 内部），保证两臂都更新：
```rust
let rep = if use_lazy { self.dspark_spec_lazy(..)? } else { self.dspark_spec_swallowed(..)? };
self.lazy_hist.push(rep.k_acc);
Ok(rep)
```

---

## 3. 方案 · `dspark_spec_lazy`（逐行 lazy arm）

### 3.1 逐行循环（落点级）

新 arm，结构 = `dspark_spec_swallowed`（`:6284`）去掉"一次性 6 行块"，改逐行 + 早退：

```rust
fn dspark_spec_lazy(&mut self, dspark: &mut DsparkDev, token: u32, pos: usize)
    -> Result<DsparkSpecReport>
{
    let m = VERIFY_ROWS;                                  // 6（块上界，仍是 [anchor,d1..d5] 语义）
    // ---- 1. snapshot：仅服务错误路径（成功路径 host=&[] 早退）----
    let host_mirrors = self.dspark_snapshot(pos, m)?;     // :4802
    // ---- 2. draft：与 swallow 逐字相同 ----
    let t = Instant::now();
    dspark.import_tap(self.s.dspark_tap.ptr as *const f32)?;
    dspark.draft_forward(token, pos)?;
    let drafts = dspark.drafts()?;
    let draft_ms = t.elapsed().as_secs_f32() * 1e3;

    // ---- 3. 逐行 verify + 早退 ----
    let mut rows_in: Vec<u32> = Vec::with_capacity(m);
    rows_in.push(token);                                   // 行 0 = anchor
    rows_in.extend_from_slice(&drafts);                    // 行 i≥1 = drafts[i-1]
    let t = Instant::now();
    let mut rows: Vec<u32> = Vec::with_capacity(m);
    let mut fail: Option<FerriteError> = None;
    let mut k_emit = 1usize;
    match self.lazy_run_row(&rows_in, 0, pos) {            // 行 0 @ pos
        Ok(a) => rows.push(a),
        Err(e) => fail = Some(e),
    }
    if fail.is_none() && drafts[0] == rows[0] {
        let mut matched = 1usize;
        for i in 1..DSPARK_DRAFTS {                        // i = 1..=4
            match self.lazy_run_row(&rows_in, i, pos) {    // 行 i @ pos+i
                Ok(a) => rows.push(a),
                Err(e) => { fail = Some(e); break; }
            }
            if drafts[i] != rows[i] { break; }
            matched = i + 1;
        }
        k_emit = matched + 1;
    }
    let verify_ms = t.elapsed().as_secs_f32() * 1e3;
    // ---- 3b. 错误路径：全块回滚（keep=0）----
    if let Some(e) = fail {
        let _ = self.dspark_rollback_keep(pos, m, 0, &host_mirrors)?;
        return Err(e);
    }
    debug_assert_eq!(rows.len(), k_emit, "lazy: rows_run == k_emit");
    ...
}
```

`rows_run == k_emit` 的证明（**与 `spec_accept` 的真实代数对齐**，`spec_step.rs:91`）：
- `k_emit = matched + 1`；`rows` 需要 `rows[0..=matched]`（`rows[matched]` 是判定 mismatch 的那一行）。
- ⇒ 跑的行数 = `matched + 1 = k_emit` ✓；`drafts[0] != rows[0]` ⇒ `matched=0, k_emit=1`，只跑行 0。

### 3.2 三处运行期量（lazy 必须显式校正）

**(a) 位置计数器逐行推进**。`step_rows` 从设备计数器读 `pos_base`（`:4091`），块建立在 `pos_base..pos_base+m-1`。m=1 时若计数器不动，每行都在 `pos` 前向。⇒ 逐行 `set_pos_ctr(pos+i)`（`:6631`，一次 4B H2D）：

```rust
fn lazy_run_row(&mut self, rows_in: &[u32], i: usize, pos: usize) -> Result<u32> {
    self.set_pos_ctr(pos + i)?;                            // :6631  ← 关键
    self.spec_capture = false;                             // 见 (c)
    let r = self.step_rows(&rows_in[i..=i]);               // m=1
    Ok(r?[0])
}
```
（`step_rows` 内部每次重读计数器做 `pos_rows` H2D `:4099`；图只 READ `pos_rows`/`pos_ctr`，故 m=1 图可复用、位置靠外部刷新。）

**(b) tap 行号（deferred tap）**。m=1 块 `hc_collapse` 永远写 slot 的第 0 行（`:6939-6946`）。lazy 在每行 `step_rows` 返回后，用 `h_r`（m=1 ⇒ 恰是本行 hidden）**自己写 tap 到行号 i**：

```rust
// 与 :6939 同一对算子；唯一差异是行基址 (slot*VERIFY_ROWS + i)
for layer in 0..cfg.n_layers {
    if let Some(slot) = cfg.dspark_target_slot(layer) {
        self.dev.hc_collapse(
            self.s.h_r.ptr as *const f32,
            self.s.dspark_pre_mean.as_f32(),
            (self.s.dspark_tap_r.ptr as *mut f32)
                .wrapping_add((slot * VERIFY_ROWS + i) * cfg.dim),
            1, cfg.hc as i32, cfg.dim as i32)?;
    }
}
```
随后 `note_ctx_rows(tap_r, m, k_emit, pos)` 与 `carry_kept_tap(keep=k_emit)` **原样**可用（它们按行号索引）。为使上者不重复写，需在 `layer_rows` 的 tap 写点（`:6927`）加一个抑制标志 `if self.spec_capture && !self.spec_tap_deferred`；lazy 置 `spec_tap_deferred=true`。

**(c) compressor 提交点**。lazy 设 `spec_capture=false`：`step_rows` 内**既不写 in-graph tap**（改由 (b) 外部写，且**这正是我们要的**），**也不存 `spec_snap_kvp/scp`**（lazy 无 replay，`:7848` 的保存是纯浪费）。`spec_capture` 的两处 consumer 只有 `:6927` 与 `:7848`（全文件 53 处匹配确认），故 `false` 是安全的——但被 (b) 的 `spec_tap_deferred` 取代 tap 写。

> ⚠️ (b) 的 tap 写是**图外 launch**，逐行 3 次（`n_target≤3`）。m=1 图的 `step_rows_inner` 因 `spec_capture=false` 不含 tap 写，故 capture/replay 与之一致，无需按行号重捕获——这是 deferred tap 相对"把行号烘进图"（需 k_emit 个图）的关键优势。

### 3.3 commit（无回滚、无 replay）

```rust
// 与 dspark_commit (:6513) 的唯一差异：跳过 compress_replay，且 host=&[]。
fn dspark_commit_lazy(&mut self, pos: usize, k_emit: usize) -> Result<()> {
    // 无 rollback（host=&[] 会早退 :4938，但 derive 上我们本就不需要它）
    // 无 compress_replay（逐行 step_rows 已按序提交 0..k_emit，§1.3）
    self.set_pos_ctr(pos + k_emit)?;                       // :6631
    self.inv_compress_len()?;                              // :6537 不变量
    Ok(())
}
```
调用点（对齐 swallow 的收尾）：
```rust
let t = Instant::now();
self.dspark_commit_lazy(pos, k_emit)?;
dspark.note_ctx_rows(self.s.dspark_tap_r.ptr as *const f32, m, k_emit, pos)?;   // 与 :6360 同
Self::carry_kept_tap(self.dev, self.s.dspark_tap.ptr,
                     self.s.dspark_tap_r.ptr as *const c_void, cfg.dim, k_emit)?; // 与 :6362 同
let commit_ms = t.elapsed().as_secs_f32() * 1e3;
```

### 3.4 "零回滚"推导（ring 侧，复核 verify-fusion-arch §3.4）

- 行 i 写 ring 槽 `(pos+i) % win`，喂的 token 是 `rows_in[i]`（i=0 是 anchor token；i≥1 是 `drafts[i-1]`）。
- 行 i 被 KEEP ⇔ `rows_in[i]` 是真 token ⇔ `i=0` 或 `drafts[i-1]` 被接受 ⇔ `i ≤ matched`（= `k_emit-1`）。
- ⇒ 跑过的行 `0..=matched`（共 `k_emit` 行）**全在 keep 范围**；未跑的行从未写。
- ⇒ **成功路径零回滚**。（错误路径见 §3.1 的 `dspark_rollback_keep(pos,m,0,&host_mirrors)`——快照的唯一用途。）

### 3.5 报告与 emit（与 batched 同形）

```rust
let k_acc = k_emit - 1;
let next = rows[0];
let mut verify_out = [0u32; DSPARK_DRAFTS];
verify_out[..k_acc].copy_from_slice(&rows[1..k_emit]);   // 仅 0..k_acc 有定义
let mut emitted = Vec::with_capacity(k_emit);
emitted.push(next);
emitted.extend_from_slice(&verify_out[..k_acc]);         // = rows[0..k_emit]
```
`DsparkSpecReport { next, drafts, verify_out, k_acc, emitted, draft_ms, verify_ms, commit_ms }` 形状不变 ⇒ 驱动侧统计（`serve.rs:569-597` 的 `dspark_acc_sum/dspark_steps`、`[dspark] mean-accept` `:665`）**自动可比**。
⚠️ `verify_out[j]`（`j ≥ k_acc`）在 lazy 下**未定义**（填 0）——batched 会填被拒 draft 的 argmax。`dspark_dump_step("spec", ...)` 的 golden diff 只应比较 `0..k_acc`，否则会报"布局差异"假阳性（`dspark_spec_swallowed` `:6408` 的同类注释）。

---

## 4. 兼容性

### 4.1 形状池 / 图化

| 面 | lazy | batched | 结论 |
|---|---|---|---|
| 形状 | `m=1`（`step_rows(&[tok])`） | `m=6` | 两槽并存（`VERIFY_GRAPH_SLOTS=2` `:107`） |
| 槽位 | `verify_slot(1)` | `verify_slot(6)` | `:4273` 按 m 键控，互不驱逐 |
| capture | DRY→CAPTURE→REPLAY 各一次（`:4106-4178`） | 同 | 首次用该形状时一次性 |
| 图内 host 依赖 | `pos_rows`/`ids_r` 每次刷新于图外（`:4098-4099`）✓ | 同 | 位置无关，可复用 |
| tap 写 | **图外**（deferred，§3.2b） | 图内（`:6939`） | lazy 的 `m=1` 图不含 tap，故不因行号重捕获 |
| `spec_capture` | `false` | `true` | lazy 跳过 in-graph tap 与 spec_snap |

**换臂时的图行为**：从 batched 首轮切 lazy，`m=1` 槽空 ⇒ 首个 lazy 行做 DRY（真跑，即行 0 的前向，非额外成本）→ 第 2 行 CAPTURE → 第 3 行起 REPLAY。首轮切到 lazy 的**总前向次数仍 = k_emit**（每行一次），DRY 不引入额外前向。✓

### 4.2 AR 足迹

- 每层 verify = attention AR + MoE AR = **2 轮**（`verify-ms-breakdown.md:19`：AR v5 80 次 = 2×40）。
- **batched**：整块一次 ⇒ 每步 `1 × 80 = 80` 轮。
- **lazy**：逐行 ⇒ 每步 `k_emit × 80` 轮。
- **共同点**：两臂**每层每行**都是 2 轮 AR，**结构相同**；差异只在"一次跑 6 行" vs "k_emit 次各跑 1 行"。
- **rank 一致性**：`k_emit` 由跨 rank 归约的 argmax 决定 ⇒ 所有 rank 跑同样多的行 ⇒ 每步 AR 轮数逐 rank 相同 ⇒ **AR v5（无 host rendezvous）不需要额外 epoch 同步**。

### 4.3 逐步换臂的安全性（更正 verify-fusion-arch §7）

verify-fusion-arch §7 的硬约束"选定臂后本请求不换臂"**可放宽为"每轮全 rank 同步换臂"**，理由：
1. 决策输入 `mean_k` 是**整数窗口**、`B`/`c` 存在**进程级** `OnceLock` ⇒ 全 rank 逐位同决策。
2. `k_emit` 逐 rank 相同 ⇒ 换臂当轮，各 rank 同数发起 AR。**跨轮不匹配才会死锁**；同轮同步换臂不产生不匹配。
3. 唯一风险是 `B`/`c` 的**本地计时**进入判据（抖动导致 rank 分歧）——本设计用**进程级共享 + 整数 mean_k** 消除。

⇒ lazy 可**逐轮自适应**（低 accept 自动切 lazy、accept 抬升自动切回 batched）。这是相对"每请求锁臂"的实质改进。

---

## 5. 影响范围

- **修改文件**：`crates/ferrite-models/src/dsv41/chain_dev.rs` 单文件（新 arm `dspark_spec_lazy` + `dspark_commit_lazy` + `lazy_run_row` + 路由 + `LazyHist` + 2 个 gate + `spec_tap_deferred` 字段 + `reset()` 一行）。
- **`dspark_dev.rs`**：**不改**（`note_ctx_rows`/`carry_kept_tap`/`import_tap`/`drafts` 接口不变）。
- **影响模块**：DSpark spec 步、verify CUDA graph 形状池、AR v5 足迹、compressor 提交路径。
- **兼容性**：**无 API breaking change**（纯 env gate + 新 arm；`DSV41_LAZY_VERIFY=0` 逐字回退旧路）。`DSV41_SWALLOW_STEP` 单独 ON 时仍只跑 batched 臂（行为不变）。

---

## 6. 风险评估

| ID | 风险 | 应对 |
|---|---|---|
| **R0** | `B` 未定（mrows 是否 dispatch 决定 lazy 是"永远胜"还是"仅低 accept 胜"） | 先跑 verify-fusion-arch §9-T0 的 nsys 单次诊断钉死 `B`；未定前默认 `τ` 用 c=6.15 的保守档 |
| **R1** | **compressor 双重提交**（verify-fusion-arch §3.4 伪码的 bug） | 用 §3.3 的 `dspark_commit_lazy`（**无** replay）；A/B 判据含"latent 不漂移"（`DSV41_DSPARK_DEBUG` 的 verify 字段 + 出师表逐字） |
| **R2** | **m=1 tap 覆盖 index 0**（§3.2b）导致 `note_ctx_rows`/`carry_kept_tap` 读错行 | deferred tap（外部按行号写）；判据 = `dspark_parity` verify 行级对照（`verify_bad==0`） |
| **R3** | **位置计数器**未逐行推进 ⇒ 每行前向都在 pos | `lazy_run_row` 先 `set_pos_ctr(pos+i)`；单测 `k_emit=6` 时 `pos_rows` 轨迹 = `pos..pos+5` |
| **R4** | lazy 快照成本（成功路径本可免，但为错误路径保留） | 快照走 P0 fused 核（`dspark_snapshot` `:4802` 的 `supports_dspark_snapshot()` 分支）；无 P0 时快照 ~2.7ms 会侵蚀 lazy 收益 ⇒ 记为 gate 的前置条件 |
| **R5** | AR v5 换臂足迹（历史高危，`final-400-config.md:264` R3-④） | §4.3 整数判据 + 进程级 τ ⇒ 全 rank 同步；T3 变 accept 同会话 A/B 验无死锁 |
| **R6** | 池仅 2 槽，第三形状（parity 自检）落 direct | 不致命；A/B 时核对 `[verify_graph] captured verify_graph_m1/_m6` 两行都在 |
| **R7** | 冷启动第 1 轮永远 legacy（`spec_primed` `:5979`），路由第 2 轮才生效 | 可接受（与 batched 臂同样的引导）；低频请求下 lazy 收益打折 ⇒ 文档显式记账 |

---

## 7. 验证计划（最少 GPU 次数）

| # | 会话 | 内容 | 判据 |
|---|---|---|---|
| T0 | GPU（1） | nsys 诊断 mrows dispatch（verify-fusion-arch §9-T0） | 定 `B` ∈ {8.2, 37} ⇒ 定 `τ` |
| T1 | GPU（1） | `DSV41_LAZY_VERIFY=0/1` 同会话 A/B（低 accept 载荷） | lazy 轮 `verify=` ↓、`k_emit` 分布与 batched 一致 |
| T2 | GPU | `dspark_parity` verify 行级对照（lazy vs batched） | `verify_bad == 0`（逐位） |
| T3 | GPU | 变 accept 的 AR 足迹 + 换臂 | 无死锁；`[verify_graph] captured m1/m6` 两行都在 |
| T4 | GPU | `k_emit=1` / `k_emit=6` 边界 | `pos_rows` 轨迹正确；`note_ctx_rows` 行号正确 |

**同会话 A/B 铁律**：同远端同一时刻只有一个测试驱动；subagent 只做代码/分析；不轮询远端。

---

## 8. 建议分工

- **工部**：实现 `dspark_spec_lazy` / `dspark_commit_lazy` / `lazy_run_row` / 路由 / `LazyHist` / `spec_tap_deferred`（`chain_dev.rs`，~200 行，参照 `dspark_spec_swallowed` `:6284`）。**原因**：纯 Rust 机械改造 + 一处 `layer_rows` tap 抑制标志，无新 kernel。
- **户部**：T0 的 nsys 诊断 + `B`/`c` 实测 + `τ` 标定（§2.2）；复核 §4.2 的 AR 轮数账。**原因**：性能账是户部本职，且 T0 决定 τ。
- **刑部**：`dspark_parity` verify 行级逐位对照 + 边界（`k_emit=1/6`、`drafts[0]` 首检失败、错误路径半写）。**原因**：R2/R3 是正确性红线。
- **兵部**：AR v5 的换臂 rank 一致性审查（`tp.rs` `ar_v5` 计数）+ R1 的 compressor 双提交安全。**原因**：AR-v5 死锁与状态漂移是历史高危。
- **吏部**：gate 默认值读回确认 + `LazyHist` 复位卫生（v13→v15 的 0.37ms 误诊教训）。**原因**：新 gate 的静默失效是本项目 #1 测量偏差陷阱。
- **礼部**：本文件 + `NEXT-SESSION-HANDOVER.md` 同步（更正 verify-fusion-arch §3.4 的 commit 伪码与 §7 的换臂结论）。**原因**：避免错误的 commit 伪码被直接照抄。
- **尚书省**：不分配（单文件设计，无跨部门大工程）。

---

## 附：一句话总结

**两臂不是"二选一的产品分支"，而是"同一臂按 mean-k 的自适应阀位"**：整数 mean-k 窗口 + 进程级 `τ=B/c−1` ⇒ 全 rank 同步决策；`m=1`/`m=6` 两形状天然占满 2 槽形状池；lazy 的 commit 必须**跳过 replay**（batched 的 snapshot→rollback→replay 在 lazy 下退化为"逐行提交即终态"）。lazy 是**低 accept 的自动快路**，batched 是**高 accept 的唯一解**——动态选路让两者共存在同一条 `dspark_spec_step` 里。

---
*中书省 · 只读分析 + 本文件（唯一产出），未执行 GPU 命令、未改动源码。*
*所有 ms 标来源（实测/推算）；关键更正（§0-4 commit 语义、§0-5 tap 行号、§4.3 换臂安全性）已在正文显式标注。*
