# SWALLOW 只生成 1 token 的修复方案设计（finish_reason=length 的真正签名）

> 工部 · 2026-09-12 · **只读勘察 + 设计；未改动任何源码、未执行 GPU 命令。**
> 输入（现场核对，行号以工作树 HEAD `a418f8c` 为准）：
> `crates/ferrite-models/src/dsv41/chain_dev.rs`、`crates/ferrite-dsv41/src/serve.rs`、
> `crates/ferrite-http/src/{single_flight.rs,driver.rs}`、`crates/ferrite-types/src/spec_step.rs`、
> `docs/agent/{oob-fix-result-analysis-framework.md,dspark-correctness-chain.md,swallow-fix11-design.md}`。
> 本文件职责：**给出 1-token 症状的根因判定 + 每种根因的修复 / 成本 / 验证**；不改源码。

---

## 0. 一页纸结论（先看这里）

**任务给的 4 条根因假设（max_pos / emission 传递 / k_emit / 提前退出）与观测证据都不自洽。**
`finish_reason=length` + `completion=1` 在本仓库里**不是"位置到顶"的信号，而是"请求在 prefill 之后、任何一个 decode token 被提交之前就 FAULT 了"的信号**——
`SingleFlight` 的任何 `decode` 错误都会把该 seq 以 `retired` 收走，而 `out` 里只剩 prefill 那一个 token；`driver` 见 `out.last()` 非 stop，于是报 `Length`。

⇒ **根因重排（置信度从高到低）**：

| # | 根因 | 与证据的关系 | 置信度 |
|---|---|---|---|
| **A** | **第一次 swallowed 轮（pos=15）返回 `Err`** | 完全自洽：ledger 有 `pos=10 pre`（legacy bootstrap 成功）与 `pos=15 pre`（swallowed 开始），但**没有 note**（note 只在 arm `Ok` 之后发），且 `DecodeRun` 的局部 `out` 在 `Err` 时被丢弃 ⇒ 只剩 prefill 的 1 token | **高** |
| **B** | **第一次 swallowed 轮 hang（非 `[ar5-hang]` 型）→ `STEP_TIMEOUT`(1800s) → `broadcast` 返回 "a rank did not answer"** | 与 A 的**可观测签名完全相同**（都是 `Length`+1 token）；`F=0`（`[ar5-hang]`=0）**不能**证明"无 hang"——它只是一个特定 watchdog 的 grep | 中 |
| **C1** | `max_new=1`（max_tokens 没传进 engine）→ `admit` 立即 retire | 与证据**矛盾**：会有 `Length`+1 token，但**不会**有 pos=15 的 decode ledger 行（admit 后不再 tick） | 低 |
| **C2** | emission 只传 1 token / `k_emit` 恒为 0 / 提前退出 | 与证据矛盾：这些都会**继续跑很多轮**（每轮 1 token），产出 ≫1 token；且 swallowed arm 必 `emitted.push(next)`，`len≥1` | 很低 |

**唯一下一步（一步收敛 A vs B）**：看运行日志。

```bash
LOG=<本次 SWALLOW 跑机日志>
# A 的签名
grep -n 'spec step err at pos 15' "$LOG"        # serve.rs:611-614
grep -n 'POISONING the whole request' "$LOG"
# B 的签名（A 无命中时）
grep -n 'a rank did not answer' "$LOG"          # serve.rs:256-258
# 时间缺口（B 的证据：一步卡住 ~1800s）
grep -nE 'pos=15|did not answer' "$LOG"
```

- A 命中 ⇒ 按 **§2.A** 修（先拿到错误文本，直接定位到 §2.A 的 E1–E6 子表）。
- 只有 B 命中 / 两次 `pos=15` 之间 ≥ ~1800s 的缺口 ⇒ 按 **§2.B** 修（phase 断言 + 缩短超时）。
- 两者都没命中 ⇒ 先补 §4 的**最低成本取证**（给 `dspark_spec_step` 加 exit-tag 错误打印）再判。

> ⚠️ **必须先做这一条**：`E: >60tok` 的判据（框架 §2）是在"生成正常"的前提下写的；本次观测 `LEN=1` 落进了框架表**之外**（既不是 R5 的"仍 6 token"，也不是 R7 的"ledger 零行"）。**不要把本次读数套进 R0–R7 判绿**。

---

## 1. 证据链（为什么 `length` 其实是 fault 签名）

### 1.1 driver：`Length` 的两条来路（`ferrite-http/src/driver.rs`）

```rust
// driver.rs:337-350  —— 请求被 retire 时的 finish_reason
let out     = self.engine.output(seq).unwrap_or_default();
let stopped = out.last().map(|t| self.engine.is_stop(*t)).unwrap_or(false);
let reason  = if stopped { FinishReason::Stop } else { FinishReason::Length };
let completion = out.len().saturating_sub(if stopped { 1 } else { 0 });
```

`Length` 只要求两件事：**seq 被 retire** ∧ **最后一个 token 不是 stop**。
它**不**要求 `out.len() == max_new`——这正是本仓库 `retired` 的**第三种**来路留下的洞。

### 1.2 SingleFlight：任意 decode 错误 = retire + 只有 prefill 那 1 个 token

```rust
// single_flight.rs:157-183  step_once
let next = self.engine.decode(token, pos)?;      //  ← 错误在这里
...
let retired = stopped || l.out.len() >= max_new; //  ← 错误时 out 只有 prefill push 过的 first
```
```rust
// single_flight.rs:187-198  fail()
for s in seq.into_iter().chain(self.live.take()) {
    if let Some(l) = self.arena.get_mut(s) { l.retired = true; }   // ← 直接 retire
}
```
```rust
// single_flight.rs:130-153  admit()：prefill 后 push(first)，这就是唯一活下来的 token
l.out.push(first);
```
```rust
// single_flight.rs:230-263  tick()：step_once 出错 → self.fail(Some(seq), e)
if let Err(e) = self.step_once(seq) { self.fail(Some(seq), e); }
```

⇒ **`decode` 一 Err，`out = [first]`（completion=1），`status = retired`，`is_stop(first)==false` ⇒ `finish_reason = Length`。**
**这就是 "LEN=1 / completion=1 / length" 的完整生成机制，不需要 `max_new` 参与。**

### 1.3 TpRankPool：错误把整个 `DecodeRun` 的已产出 token 全部丢掉

```rust
// serve.rs:565-757（节选）
while out.len() < n {
    if poisoned.load(Ordering::Acquire) { r = Err(poisoned_request(rank)); break; }
    let emitted = match chain.dspark_spec_step(d, t, p) {
        Ok(rep) => { ...; rep.emitted }
        Err(e) => {
            eprintln!("[dsv41] rank {rank} spec step err at pos {p}: {e} — POISONING ...");  // :611
            poisoned.store(true, Ordering::SeqCst);
            r = Err(e); break;                    // :616-617
        }
    };
    ...
    for &tok in &emitted { out.push(tok); ... }   // 已 push 的 token 在下面被丢弃
    p += step_len;
}
if res_tx.send((rank, r.map(|_| out))).is_err() { ... }   // :758  r==Err ⇒ out 丢弃
```

**关键**：一个 `DecodeRun{n:16}` 内部的**多次 step**共享一个局部 `out`。
若第 1 步（`pos=10`，legacy）成功、第 2 步（`pos=15`，swallowed）失败，
则 `pos=10` 已产出的 ~5 个 token **随 `out` 一起被丢弃**——所以 `LEN=1` 与"ledger 有两个 pre"**并存不矛盾**。

### 1.4 ledger：pre 在 arm 之前，note 在 arm 之后 ⇒ "有 pre 无 note" = arm 未正常返回

```rust
// chain_dev.rs:8157  dspark_spec_step 头部（arm 分派之前）
self.v5_ledger_pre(pos);
// chain_dev.rs:8183-8187  只有 swallowed arm 返回 Ok 才发 note
if swallow_step() && self.spec_primed_unanimous() && self.spec_primed {
    let rep = self.dspark_spec_swallowed(dspark, token, pos)?;   // ← `?`：Err 直接冒泡，note 永不执行
    self.v5_ledger_note(pos, "swallowed", rep.emitted.len());
    return Ok(rep);
}
```

**"ledger 只到 pos=15 的 pre、没有 swallowed 的 note"** ⇒ `dspark_spec_swallowed` 在 `pos=15` **既没有 `Ok` 也没有走到 note**。
（注意：legacy arm **本来就不发 note**——见 §1.4 末。）所以 `pos=10 pre` 无 note 是正常的（legacy bootstrap），而 `pos=15 pre` 无 note 才是异常。

**推论（与 §1.2/§1.3 合起来）**：`pos=10`（legacy，`step_dev` 在，`spec_primed` 由它置位）成功 ⇒ `pos=15`（**第一次 swallowed**）失败。
也就是说：**失败恰好发生在"第一次丢掉 `step_dev`"的那一轮**——它不是 OOB、不是 arm 相位，而是 swallowed 臂**新引入的那段工作**。

### 1.5 首次 swallowed 轮的"新增面"（§2.A 的错误子表就从这里来）

对照 `dspark_spec_step` 的 legacy 分支（`:8244-8416`）与 `dspark_spec_swallowed`（`:8708-8866`），
第一次 swallowed 轮相对 legacy 多做了这些**从未在一起跑过**的事：

| 新增面 | legacy | swallowed | 代码位置 |
|---|---|---|---|
| 快照行数 `m` | 5（`[d1..d5]`，row0=`pos+1`） | **6**（`[anchor,d1..d5]`，row0=`pos`） | `:8253` vs `:8724` |
| verify 块 | `step_rows(&drafts)`（5 行） | `step_rows(&[token, d1..d5])`（6 行） | `:8275` vs `:8761` |
| accept | `Self::accept`（`ANCHOR_IS_IN_BLOCK=false`） | `spec_accept(.., true)` | `:8330` vs `:8782` |
| commit `keep` | `k_acc`（**可为 0**） | `k_emit = k_acc+1`（**恒 ≥1**） | `:8336` vs `:8795` |
| carry tap 行 | `k_acc` | `k_emit` | `:8348` vs `:8803` |
| `note_ctx_rows(k)` | `k_acc`, pos+1 | `k_emit`, pos | `:8337` vs `:8796` |
| import_tap 时机 | `step_dev` 之后 | `step_dev` **之前**（tap 靠 carry） | `:8257` vs `:8742` |
| 退出不变量 | `inv_ids` | `inv_ids` | `:8386` vs `:8854` |

---

## 2. 修复方案（按根因）

### 2.A 根因 A：第一次 swallowed 轮返回 `Err`（首选）

**修复策略：先"定位到具体 `Err` 点"，再按点修。** 不要盲改。
下面的 E1–E6 是 §1.5 新增面里**确实会返回 `Err`** 的入口，按"最可能先命中"排序。

| 子项 | `Err` 入口 | 触发条件 | 修复 | 实施成本 | 验证 |
|---|---|---|---|---|---|
| **E1** | `inv_ids`（`:8854` → `:9881`） | `DSV41_INV_CHECK=1` ∧ `DSV41_SIDS_WRITEBACK` 关（默认关，`:1846`）⇒ swallowed 臂的 `s.ids` 未回写，`seen != emitted.last()` 必失败 | 要么把 `sids_writeback` 默认打开，要么让 `inv_ids` 与 `sids_writeback` 同门（**两者必须同生共死**：不变量与写入是一对） | **极小**（改 2 个 gate 的默认/互斥，~5 行） | `DSV41_INV_CHECK=1` 单跑：`pos=15` 不再出 `spec step err`；ledger 出现 `arm=swallowed` 的 note |
| **E2** | `step_rows`（`:8761`）→ `step_rows_sync`（`:5989`） | `m=6` 的 verify 块：`m > VERIFY_ROWS` 检查（`:5999`，当前 `VERIFY_ROWS` 需实测=6）、graph capture/DRY/REPLAY 的 slot 分支错误 | 先确认 `VERIFY_ROWS == DSPARK_DRAFTS + 1`；若不是，补 `m=6` slot（**这是 `sh-pair M=6` 那条线**，M 是编译期模板） | 中（若 `VERIFY_ROWS` 本已=6 则 0 成本，仅加断言） | 加 `assert_eq!(VERIFY_ROWS, DSPARK_DRAFTS+1)`；`DSV41_VERIFY_GRAPH=0` 直发跑一次 |
| **E3** | `dspark_commit`（`:8795`）→ `dspark_rollback_keep`/`compress_replay`/`inv_compress_len`（`:9438/9440/9447`） | **`keep=k_emit` 恒 ≥1 而 legacy 的 `k_acc` 可为 0** ⇒ swallowed **每轮**都走 `compress_replay`；若 replay 的 `rows`/`pos_base` 与 `out_rows` 计数不一致即失败 | 核对 `compress_replay(pos, keep)` 的 `pos_base` 语义（注释写 "`pos + 1`"，但 swallowed 传 `pos`！`:9440` vs `:8795`）——**这是最可疑的一处语义错位** | 小～中（改 1 个实参或补 `+1` 语义分支） | 单轮跑 `keep==1` 与 `keep==6` 两端点，看 commit 是否 Err |
| **E4** | `carry_kept_tap(..., k_emit)`（`:8798`） | `keep==0` 早退是 `Ok`；`k_emit≥1` 走 `row_off=(keep-1)*row_bytes`（`:9378`）——若 `k_emit > VERIFY_ROWS` 越界（`debug_assert` 在 release 下不拦） | `k_emit` 上下界收敛到 `1..=VERIFY_ROWS`（`spec_accept` 的返回值做一次 `clamp` + `debug_assert` 提到 release 的 `assert`） | **极小** | 断言 + 失败计数落 ledger |
| **E5** | `dspark.note_ctx_rows`（`:8796`）/ `import_tap`（`:8742`）/ `draft_forward`（`:8747`） | tap 靠 carry 而非 `step_dev`（新增时序）⇒ 第一轮的 tap 可能"半写"；`note_ctx_rows(k_emit, pos)` 的 `pos` 从 `pos+1` 变 `pos` | 对照 `carry_kept_tap` 的 `slot` 语义与 `note_ctx_rows` 的行基；把 `pos`/`m`/`k` 三参在两侧打同一份 trace 比对 | 中 | `DSV41_DSPARK_DEBUG=1` 打印两臂的同字段 diff |
| **E6** | `spec_accept` 返回值越界（`:8782`） | 若 accept 链在 `ANCHOR_IS_IN_BLOCK=true` 下返回 `0`，则 `k_emit=0` ⇒ `k_acc = k_emit-1` **下溢 panic**（`usize` 减法！） | `k_emit.max(1)`（并在 `spec_accept` 侧断言 `1..=6`） | **极小** | 单测 `spec_accept` 全边界；release 下 run 一次 |

**交付物（A 类修复）**：先跑 §4 的取证（一行 exit-tag 打印），拿到**确切错误文本**，再对号入座改 E1–E6 中的**一个**。
**编译 + 测试**：`cargo check -p ferrite-models` + `cargo test -p ferrite-types spec_step` + 400 短跑（MAXTOK=64）。

> **E3 的额外说明（最值得先看）**：`compress_replay` 的文档明确写 "`pos_base` is the block's ROW 0 position (`pos + 1`)"（`:9461-9462`），
> 而 `dspark_spec_swallowed` 调 `self.dspark_commit(pos, m, k_emit, ...)`（`:8795`，`pos_base = pos`）。
> 若 `compress_replay` 内部真的按 "`pos_base` 是 row0" 用，那 swallowed 传的 `pos` 恰好**也是** row0（因为它的块从 `pos` 开始）——**语义可能恰好自洽**。
> 但 legacy 传的是 `pos_ctr+1`（`:8336`，块的 row0 确实是 `pos+1`）⇒ **两处语义不同、必须逐字核对**。
> 这不是"风格问题"：`compress_replay` 若在别处被按 `pos+1` 的假设读过一次，swallowed 每轮都会踩。

### 2.B 根因 B：第一次 swallowed 轮 hang（非 `ar5-hang`）→ 超时转 Err

**为什么 `F=0` 不能排除它**：框架 §2 的 F 判据只 `grep '\[ar5-hang\]'`（2 个特定 watchdog）。
一个卡在**别的集合通信**里的 rank（或非 v5 的 barrier）不会打 `[ar5-hang]`——
它只会让 `broadcast` 在 `STEP_TIMEOUT`(1800s, `serve.rs:110`) 后返回 "a rank did not answer"，
**随后走 §1.2 的同一条路 ⇒ 同样是 `Length`+1 token**。

**修复（若 B 成立）**：

| # | 修复 | 位置 | 成本 | 验证 |
|---|---|---|---|---|
| **B1** | **缩短取证超时**：加 `DSV41_STEP_TIMEOUT_MS`（默认仍 1800s，取证时设 60_000），让 hang 在 1 分钟内现形 | `serve.rs:110` / `:244` | 小（`OnceLock` 读一次） | 一次 run：60s 内出 "did not answer" 且带 `pos` |
| **B2** | **给 `DecodeRun` 的失败补位置**：把 "did not answer" 改带 `pos`（现在只有 rank） | `serve.rs:255-258` | 极小 | 日志可直接定位到 pos=15 |
| **B3** | **首轮 phase 断言**：在 swallowed 臂的 4 个相位（snapshot / draft / verify / commit）各加一条 `eprintln!("[swallow-phase] pos=.. phase=..")`（`DSV41_SWALLOW_TRACE=1` 门控） | `chain_dev.rs:8724/8747/8761/8795` | 小 | 复现 run 里"最后一个 phase"即 hang 点 |
| **B4** | **按 hang 点修**：若卡在 `step_rows`（m=6 的 verify graph 首次 capture/DRY），见 **E2**；若卡在 `import_tap` 的早退（tap 未写），见 **E5** | — | 中 | 同 E2/E5 |

### 2.C 任务原假设的逐一裁决（保留，但排在 A/B 之后）

> 这 4 条与证据都不自洽；**若 §4 取证意外指向它们**，按此表修。否则不要动。

| 原假设 | 裁决 | 若不成立则忽略；若成立这样修 | 成本 | 验证 |
|---|---|---|---|---|
| **1. serve 层 max_pos 判定** | ❌ 本仓库**没有 MAX_POS 判定**：`Length` 来自 `single_flight.rs:144/172` 的 `out.len() >= max_new` 或 §1.2 的 fault；`submit` 只在 `prompt+max_new > max_ctx` 时**拒绝**（`:210`），不会截断 | 若确系位置到顶：检查 `max_new` 是否被传成小值；`TpRankPool::decode` 的 `n = LOOKAHEAD.min(max_ctx - pos).max(1)`（`:291`）在 `pos` 逼近 `max_ctx` 时退化为 `n=1`——把 `max_ctx` 与 `prompt+max_new` 的预算对齐 | 小 | `DSV41_DSPARK_DEBUG=1` 看 `p` 是否逼近 `max_ctx` |
| **2. emission 只传 1 token** | ❌ `rep.emitted` 在 `serve.rs:681` 整体返回（`Vec`，非 `[0]`）；已被 `DSV41_DSPARK_DEBUG` 的 `emitted={:?}`（`serve.rs:634`）打印 | 若确系：核对 `spec_accept` 的 `k_acc` 是否恒 0（→ 见 E6 的 `k_emit` 下溢） | 极小 | debug 行里 `emitted.len()` |
| **3. `k_emit` 值错误** | ❌ `k_emit = spec_accept(..,true)`，`emitted` 长度 = `k_acc+1 = k_emit`，二者**同源**；`k_emit=0` 会在 `k_acc = k_emit-1` 处 **usize 下溢 panic**（`chain_dev.rs:8783`），而不是"只 emit 1 token" | 见 **E6**：`k_emit.max(1)` + `assert!(1..=VERIFY_ROWS)` | **极小** | 单测全边界 |
| **4. swallowed 臂提前退出** | ❌ swallowed 臂**没有** early-exit 分支（`:8708-8866` 全线性，只有 `?` 冒泡）；"提前退出"只可能是 `?` 返回 Err（= 根因 A） | 无需修 arm 本身；按 A 修 `Err` 源 | — | — |

---

## 3. 实施顺序（照图施工）

```
1) §4 取证（1 行打印 + 1 次短跑 MAXTOK=64）
        │
        ├─ 命中 'spec step err at pos 15'  ──▶ §2.A：按错误文本选 E1..E6（先看 E3）
        ├─ 命中 'did not answer'            ──▶ §2.B：先 B1/B2/B3 定位 phase，再按 B4
        └─ 都没有                            ──▶ 回到 step 1，把取证门 DSV41_SWALLOW_TRACE 打开重跑
2) 改完 → cargo check -p ferrite-models -p ferrite-dsv41
3) cargo test -p ferrite-types spec_step（accept 全边界）
4) 复现 run：expect  ledger 出现 arm=swallowed 的 note 且 k_emit ≥ 1；LEN > 60
5) 若复现失败 → 把 §4 的 trace 行升级为 Err 文本直出（不 gate），重复 1)
```

---

## 4. 最低成本取证（实施前必须先加，`只读 + 设计` 阶段不改；此处给出补丁形状）

**为什么需要它**：现有日志只能证明"pos=15 的 arm 没返回"，**不能**区分 A（Err）与 B（hang）——除非日志里恰好留下了 `spec step err` 行。
给一个**只在失败路径**打印、不 gate（或 `DSV41_SWALLOW_TRACE` 默认开）的 exit-tag：

```rust
// chain_dev.rs::dspark_spec_swallowed —— 在 4 个相位边界插桩（示意）
let rep = (|| -> Result<DsparkSpecReport> {
    eprintln!("[swallow-phase] pos={pos} phase=snapshot");
    let host_mirrors = self.dspark_snapshot(pos, m)?;
    eprintln!("[swallow-phase] pos={pos} phase=draft");
    ...
    eprintln!("[swallow-phase] pos={pos} phase=verify");
    let rows = self.step_rows(&rows_in)?;   // 原为 match，这里只为取证
    eprintln!("[swallow-phase] pos={pos} k_emit={k_emit} phase=commit");
    self.dspark_commit(pos, m, k_emit, &host_mirrors)?;
    ...
    self.inv_ids(pos, &emitted)?;
    Ok(rep)
})();
if let Err(e) = &rep {
    eprintln!("[swallow-ERR] pos={pos} k_emit=? err={e}");   // ← 唯一需要的行
}
rep
```

> 若不想改 arm 内部，最小版本只需在 **`dspark_spec_step` 的 swallowed 分支**加：
> ```rust
> let rep = self.dspark_spec_swallowed(dspark, token, pos)
>     .map_err(|e| { eprintln!("[swallow-ERR] pos={pos} err={e}"); e })?;
> ```
> ——但**这解决不了 B（hang）**，因为 `?` 永不返回。所以**B 的判别仍需 B1/B2/B3**。

**判读**：
- 见 `[swallow-ERR] pos=15 err=...` ⇒ **A**，按错误文本修。
- 只见 `[swallow-phase] pos=15 phase=verify`（或更早）后无输出，且 ~1800s 后出现 "did not answer" ⇒ **B**，hang 点 = 最后打印的 phase。

---

## 5. 验证方法（每类修复的通过线）

| 修复 | 单测/静态 | 端到端（一次 run） |
|---|---|---|
| **E1**（inv/sids 门互斥） | `cargo check`；grep 两门引用点一致 | `DSV41_INV_CHECK=1` + `DSV41_SWALLOW_STEP=1`：无 `spec step err` |
| **E2**（VERIFY_ROWS=6） | `cargo test -p ferrite-models`；`assert_eq!(VERIFY_ROWS, DSPARK_DRAFTS+1)` | `DSV41_VERIFY_GRAPH=1`：第一轮 m=6 不 Err、不 hang；第二轮起 replay 命中 |
| **E3**（commit/replay `pos_base`） | 对 `compress_replay` 的 `pos_base` 语义加单测（row0 == pos_base） | `k_emit=1` 与 `k_emit=6` 两种端点各一轮，无 Err |
| **E4/E6**（k_emit 边界） | `cargo test -p ferrite-types spec_step`（全 accept 边界 + `k_emit≥1`） | release 下无 panic、无 `swallow-ERR` |
| **E5**（tap/ctx 时序） | 两臂 trace 字段 diff 为零 | `DSV41_DSPARK_DEBUG=1`：`emitted` 长度随 accept 变化 |
| **B1/B2/B3** | `cargo check` | 60s 超时下 hang 快速现形且带 `pos`/`phase` |

**共同通过线（本次专项）**：`ledger` 出现 `arm=swallowed` 的 **note** 且 `k_emit ≥ 1`；`LEN > 60`；`finish_reason == "stop"`（自然 EOS）或 `length`（真到 `max_new=300`）。
**不要再只报 `LEN=1`**——那是 fault 签名，必须连同 `finish_reason` 与 `spec step err`/`did not answer` 一起报。

---

## 6. 陷阱与边界（本次专属）

1. **`LEN=1` 不等于"生成 1 个 token"**：它是"请求 fault，且 `DecodeRun` 的局部 `out` 被 `r.map(|_| out)` 丢弃"。
   需要区分"引擎真的只产 1 个"与"产了 5 个但被丢"——后者由 §1.3 的 `out` 生命周期决定，**只能靠日志的 `spec step err` 行看出来**。
2. **`F=0`（无 `[ar5-hang]`）≠ 无 hang**：F 只覆盖两个 kernel watchdog。任何别的卡点都会先撞 `STEP_TIMEOUT`。
3. **`note` 的缺失是 arm 未返回的**结构性**证据**：legacy arm 本来就不发 note（`dspark_spec_step` 只有 swallowed/lazy 分支调 `v5_ledger_note`）。
   所以"`pos=10` 有 pre 无 note"正常，"`pos=15` 有 pre 无 note"异常——不要用"ledger 零行"的 R7 口径去套。
4. **`k_emit=0` 是 usize 下溢**（`chain_dev.rs:8783` `let k_acc = k_emit - 1;`），不是"emit 1 token"；它在 debug 下 panic、release 下变成巨大 `k_acc` ⇒ 会走进越界路径。**E6 必须先修成 `max(1)`**。
5. **`compress_replay` 的 `pos_base` 是"row0"**（`:9461`），legacy 传 `pos_ctr+1`、swallowed 传 `pos` —— 两者都自称 row0，**必须逐字核对**，不能因为"注释一致"就放过。
6. **`DSV41_INV_CHECK` 默认关**（`:2045`）：如果取证 run 开了它而 `DSV41_SIDS_WRITEBACK` 没开（`:1852`），`inv_ids` **必然**在 swallowed 第一轮失败。做 A/B 时**两门必须一起开或一起关**。
7. **`max_ctx` 预算**：`submit` 用 `prompt + max_new > max_ctx` 拒绝（`:210`）。若 `max_seq_len` 被配小，300 的 `max_tokens` 会在 **admission** 就被拒（那是 `Cancelled`，**不是** `Length`）——可用于排除 C1。

---

## 7. 交付给实施者（行动清单）

| 优先级 | 动作 | 依据 |
|---|---|---|
| **P0** | 加 §4 的 `[swallow-ERR]` 取证行 + 1 次 MAXTOK=64 短跑 → 拿错误文本 | §1.4 |
| **P0** | 同一次 run 里确认是否 `spec step err at pos 15`（A）还是 1800s 后 `did not answer`（B） | §2.A/§2.B |
| **P1** | 无论 A/B，先落 **E6**（`k_emit.max(1)` + `assert!(1..=VERIFY_ROWS)`）——**零风险防御**，排除 usize 下溢这条分支 | §2.C-3 |
| **P1** | 核对 **E3** 的 `pos_base` 语义（legacy `pos+1` vs swallowed `pos`）——最可能的**真**错位 | §2.A-E3 |
| **P2** | 落 **B1/B2**（超时门 + 带 `pos` 的 "did not answer"）——即使 A 成立也留着，防止下次 hang 又"隐身" | §2.B |
| **P2** | `inv_ids`/`sids_writeback` 的门统一（**E1**） | §2.A-E1 |
