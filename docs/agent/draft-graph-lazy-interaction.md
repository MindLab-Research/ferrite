# DRAFT_GRAPH × lazy verify 的交互分析（L6 的可行性与风险）

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 代码基线：工作区 HEAD（`crates/ferrite-models/src/dsv41/dspark_dev.rs` / `chain_dev.rs` / `tp.rs` /
> `crates/ferrite-kernel/src/devrt.rs`），逐条 `file:line` 核对。
> 输入：`docs/agent/lazy-verify-optimization-path.md`（L0–L5 阶梯）、`docs/agent/draft-graph-p3c.md`
> （P3c 实现归档）、`docs/agent/lazy-batched-gate.md`、`docs/agent/dspark-correctness-chain.md`。

---

## 0. 判决（先读五条）

1. **交互面是干净的：lazy 每轮只调 `draft_forward` 一次 ⇒ 图每轮只 replay 一次。**
   （`chain_dev.rs:8203` lazy / `:7909` swallowed，两臂各一次，块循环在 `draft_forward` 内部。）
   L6 不会退化成「一行一次 replay」。

2. **DRAFT_GRAPH 与 VERIFY_GRAPH 同一条 stream，但不会互锁。**
   两个图各自 `capture_begin/capture_end` + `graph_launch` 都落 `self.stream`（`devrt.rs:1461/1475/1529`），
   而两处捕获在**程序序上严格串行**（`draft_forward` 返回后 `step_rows` 才被调），不存在嵌套 capture，
   也不存在图与图之间的 replay 依赖环。**stream 不是风险源。**

3. **🔴 真正的风险是 host_barrier 的到达次数不对称，不是 stream。**
   DRAFT_GRAPH 的四个臂对 `Collective::host_barrier` 的到达次数是 **直发 0 / DRY 0 / replay 1 / capture 2**
   （`dspark_dev.rs:1206-1226` + `:1636-1644`），而 `SpinBarrier` 是**全局 generation 计数器**
   （`tp.rs:51-71`）。这与 `ar5-hang` 的成因**完全同类**——verify 侧为此付出了 `VerifyArm` 投票
   （`chain_dev.rs:1948-1960`、`verify_graph_gate` 的 `unanimous_i32`，`chain_dev.rs:5684-5690`）
   + `SWALLOW_GRAPH_WARMUP_BLOCKS` 两道补丁。**DRAFT_GRAPH 两道都没有**：
   `draft_graph_arm(&self, pos)` 是纯 `&self` 判据，**没有任何跨 rank 协商**。
   唯一的 per-rank 分歧源是 `graph_failed` latch（`:1677`）——一旦某个 rank 的 capture 失败，
   该 rank 此后走直发（0 次 barrier）、其余 rank 走 replay（1 次），**epoch 从此永久错位**。
   `draft-graph-p3c.md §6` 只论证了这种情况下「数值一致」，**漏了 barrier 计数这一条**。

4. **`pos >= win` 的覆盖率被任务前提高估了：`pos` 是 token 位置，不是「步」——spec 解码每轮推进
   `k_emit`（≈4.6）个位置。** 所以「出师表 ~130 步刚好够」不成立：按 accept 1.214 / 146 token 算，
   ~66 轮里只有最后 ~24 轮 replay。详见 §3。

5. **结论**：L6 的**收益在 long-generation 上真实**（replay 占比 →90%），但在仓库的标准探针
   （146–200 token）上只有 **~40–50% 的阶梯口径**，且 `tok/s` 必须按**全程平均**口径读，
   不能按 replay 段稳态读。**上机前必须先补 §4 的 barrier 对称化**，否则 TP8 有挂死风险。

---

## 1. 交互面核对：lazy verify 每轮调几次 draft_forward？

**一次。** 三个臂的调用点：

| 臂 | 调用点 | 次数/轮 |
|---|---|---|
| lazy（`dspark_spec_lazy`） | `chain_dev.rs:8203` | 1 |
| swallowed/batched（`dspark_spec_swallowed`） | `chain_dev.rs:7909` | 1 |
| legacy（`dspark_spec_step` 末尾） | `chain_dev.rs:7439` | 1 |

`draft_forward` 内部跑 `bs=5 × n_mtp=3` 的块循环（`dspark_dev.rs:1262-1558` 的 `draft_body`），
所以「120 发 → 1 发」是**每轮一次**的账，`graph_replays` 与轮数 1:1（`dspark_dev.rs:1223`）。

lazy 的「逐行」只作用于 **verify**（`lazy_run_row` → `step_rows_sync(m=1)`，`chain_dev.rs:8103-8125`），
**不影响 draft 的粒度**。⇒ DRAFT_GRAPH 的启动开销（DRY + capture）与 replay 都是一轮一次，
不会被 `k_emit` 放大（这一点与 L1 的 hc 族「×k_emit 税」是相反方向的）。

**但请注意 lazy 的路由**：`lazy_route_decide`（`chain_dev.rs:8052-8066`）按 `mean_k` 选择 lazy / batched，
计数任务（mean_k≈5.0）恰好压在 `τ = B/c − 1 = 37/6.15 − 1 = 5.02`（`chain_dev.rs:2404/2413/2418`）上，
带 ±0.25 的 Schmitt（`:2380`）⇒ **同一任务可能在两臂之间切换**。DRAFT_GRAPH 在**两臂里都在**
（两臂都只调一次 `draft_forward`），所以 L6 不受路由影响——但 A/B 时必须把路由状态（`[dspark]` 的
route 行）一并记录，否则「收益漂移」会被误判成图的问题。

---

## 2. DRAFT_GRAPH × VERIFY_GRAPH：同 stream 吗？capture/replay 冲突吗？

### 2.1 同一条 stream —— 是

`DsparkDev.dev: &'a Device`（`dspark_dev.rs:498`）与 `DevChain.dev` 是**同一个 `Device` 实例**
（`serve.rs:401/413`：先 `DevChain::new(&dev, ...)`，再 `DsparkDev::new(&dev, ...)`）。
`Device::capture_begin/end/graph_launch` 全部转发到 `devrt` 的 `self.stream`
（`device.rs:1592/1596/1611` → `devrt.rs:1461/1475/1529`）。

### 2.2 capture 时序 —— 无嵌套（程序序保证）

一轮的执行序是：

```
dspark.draft_forward(token, pos)      ← 可能 capture_draft（内部 capture_begin … capture_end）
dspark.drafts()                        ← D2H，图外
[comm.host_barrier()]                  ← lazy 的「唯一一次」pre-loop rendezvous（chain_dev.rs:8225-8227）
lazy_run_row(0) → step_rows_sync(m=1, skip_barrier=true)   ← 可能 capture_verify
…
```

`draft_forward` **返回之后**才会进入 `step_rows`，而 `capture_draft`/`capture_verify` 都是
「begin → body → end」的闭合块（`dspark_dev.rs:1701-1715`、`chain_dev.rs:5728-5756`），
且 `capture_end` 保证退出捕获态（`devrt.rs:1475-1490`，`capturing` 标志先清）。
⇒ **不存在「捕获中再捕获」**，`cudaStreamCaptureUnjoined(901)` 的路径不存在。

⚠️ 一个必须知道的口径差异：`capture_begin` 用的是 `cudaStreamCaptureModeRelaxed`（`devrt.rs:1466`），
只拒绝**捕获线程自己**的非法调用，TP8 的其它 rank 线程不受影响——这正是它当初从 Global 改过来的原因。
两个图的捕获都吃这条规则，所以互不影响。

### 2.3 replay 依赖 —— 有顺序依赖，无环

两个图的 replay 都只是往同一 stream 上 `cudaGraphLaunch`（`devrt.rs:1529-1536`），
**按 stream 序串行**：draft replay 在前、verify replay（一行一条，`k_emit` 条）在后——
这与直接发 kernel 时的顺序完全一致（图只是发射方式的替换），所以依赖关系不变、无环、无死锁。

### 2.4 ✅ 唯一的真实死锁面：host_barrier 到达次数（不是 stream）

| 臂 | 代码 | `host_barrier` 次数/轮 |
|---|---|---:|
| 直发（gate 拒 / latch） | `dspark_dev.rs:1206-1207` | **0** |
| DRY | `dspark_dev.rs:1208-1213` | **0** |
| replay | `dspark_dev.rs:1219-1222` | **1** |
| capture | `dspark_dev.rs:1636-1644` | **2** |

`SpinBarrier::wait`（`tp.rs:51-71`）是 `count/gen` 的**代数计数器**：`n` 个 rank 每轮必须各到达一次。
两个 rank 走不同次数的臂 ⇒ 后面的每个 barrier 都错一代 ⇒ **静默 misphase 或挂死**
（`chain_dev.rs:1931-1947` 对这条机制的描述，正是 `ar5-hang` 的复盘）。

**为什么 DRAFT_GRAPH 比 VERIFY_GRAPH 更容易踩这条：**

- verify 侧：`verify_graph_gate` 用 `unanimous_i32`（`chain_dev.rs:5684-5690`）把臂**投票统一**，
  不一致就全体退直发；再加 `SWALLOW_GRAPH_WARMUP_BLOCKS = 3`（`:2277/:5641`）把「两形状切换」的窗口关掉；
  再让**每个臂的 ENTRY 都吃一次 barrier**（`:5353-5357` 的 `!skip_barrier` 分支）——方法 A。
- draft 侧：`draft_graph_arm` **纯 per-rank**，无投票、无 warmup、无 entry 对齐。

**现在是否已经会出问题？** 稳态下不会——`draft_graph_arm` 的 7 条判据里，
`draft_graph_want()`（env）、`supports_ring_append/memset_async`（`.so` 符号）、`ar_v5()`、
`seed_align()`/`seed_pos_fix()`（env）、`unit_dump`（env）、`win`/`pos`（同一请求同一轮）
**全部是进程级或同轮同值** ⇒ 各 rank 同臂，次数对称。

**但 `graph_failed`（`dspark_dev.rs:1570` + `:1677`）是 per-rank 的**：
capture 只在 `pos≈win+K` 那一轮尝试**一次**，任何一次瞬时失败（驱动拒绝某个 op、instantiate 失败、
`.so` 在某 rank 上差异）都会在**那一个 rank**上永久 latch。此后该 rank 每轮 0 次 barrier，
peer 1 次 ⇒ **从下一轮起 epoch 永久错位**。这是「一次性、不可逆、单 rank」的组合，
正是最危险的形态。`draft-graph-p3c.md §6` 的非对称 latch 段对此**只做了数值论证**，漏了这条。

---

## 3. `pos >= win` 的覆盖率分析（任务前提需要修正）

### 3.1 gate 的确切条件

```rust
// dspark_dev.rs:1607-1613
if self.win < 1 || pos < self.win || seed_align() || seed_pos_fix() { return false; }
```

`win = cfg.window_size = 128`（`dspark_dev.rs:679`；DSV41 配置见 `crates/ferrite-dsv41/STATUS.md:465`）。
`win_rows(pos)`（`dspark_dev.rs:3587-3601`）在 `pos >= win` 后恒 `(win, 0)`——
这冻结了 `window→all_kv` 拷贝的 size / `s0` 分支 / `sparse_attn` 的 `n_win` / `idxs` 表（D3/D4）。

**启用时刻**（`dspark_dev.rs:1206-1226` 的 dispatch）：

| 轮 | 条件 | 臂 | 收益 |
|---|---|---|---|
| `pos < 128` | gate 拒 | 直发 | 0 |
| `pos` 首次 ≥128 | `!graph_dry_done` | **DRY**（真实直发） | 0 |
| 下一轮 | `graph_dry_done && graph.is_none()` | **capture**（录制+首发） | **0 或负**（多一次 instantiate） |
| 再往后 | 有 `graph` | **replay** | **−3.3ms/轮** |

注意 `graph_dry_done`/`graph_failed`/`graph` 都是 **`DsparkDev` 的字段、进程级、无 per-request reset**
（`dspark_dev.rs:819-822`，`impl Drop` 只释放 exec，`:3752-3760`；`DsparkDev` 在 `serve.rs:413`
**启动时构造一次**）。⇒ **多请求 serve 里 DRY+capture 只付一次**（第 1 个请求），
后续请求从自己的第一个 `pos>=128` 轮直接 replay。**但 `pos >= win` 的等待是每请求重付的**
（`pos_ctr` 每请求 reset）。

### 3.2 关键修正：`pos` 推进的是 `k_emit`，不是 1

`pos` 是**绝对 token 位置**（`dspark_spec_lazy` 的 `pos` 参数 = `pos_ctr`，`chain_dev.rs:7351/8184`），
每轮由 commit 推进 `k_emit`（`dspark_commit_lazy`，`:8037`；`k_emit = 1 + matched`，`:8276`）。
所以：

```
轮数            R  = ⌈L / K⌉
gate 打开的轮号 n0 = ⌈(win − P) / K⌉        （P = prefill 位置数，K = 平均 k_emit）
replay 轮数        = max(0, R − n0 − 2)      （−2 = DRY + capture）
收益               = replay轮数 × 3.3ms
```

### 3.3 三个场景（P≈40，win=128）

| 场景 | L | K | R | n0 | replay 轮 | replay 占比 | 节省 | 全程均值口径 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 计数 1..200，mean_k=5（K=6） | 200 | 6.0 | 34 | 15 | **17** | 50% | 56ms | −1.65ms/轮 |
| 计数 1..200，K=4.64（=411 tok/s ÷ 11.3ms 反推） | 200 | 4.64 | 44 | 19 | **23** | 52% | 76ms | −1.72ms/轮 |
| 出师表，accept 1.214（K=2.214） | 146 | 2.21 | 66 | 40 | **24** | 36% | 79ms | −1.20ms/轮 |
| 长生成（对照） | 1000 | 6.0 | 167 | 15 | **150** | 90% | 495ms | −2.97ms/轮 |

**⇒ 任务前提「出师表 ~130 步刚好够」把 `pos` 当成了步数。**
spec 解码下一轮跨 4.6 个位置，所以「跨过 128」并不等于「剩 130 步」——
出师表实际只有 146/K ≈ 66 轮，其中只有 ~24 轮 replay。
**「14.6ms → 411 tok/s」是 replay 段的稳态口径，不是全程平均口径。** 按全程平均：
- 计数（K=6）：`(17×11.3 + 17×14.6)/34 = 12.95ms` ⇒ **463 tok/s**（仍 >400，因为 replay 恰好占一半）
- 出师表（K=2.214）：`(24×11.3 + 42×14.6)/66 = 13.4ms` ⇒ **165 tok/s**（accept 才是这里的瓶颈，不是 draft）

**建议口径**：报 `tok/s` 时必须同时报 **replay 占比**（或用长生成测），否则 400 这个数会被
短探针的 warmup 摊薄而误读。`graph_replays`（`dspark_dev.rs:648`）**目前只累加、从不打印**，
上机时应加一行收尾输出，专门用来抓「captured 但从未 replay（短跑）」这类陷阱——
这与 `draft-graph-p3c.md §8.1` 的「capture 行不出现 = gate 没生效」是同一类防线的另一半。

### 3.4 两个额外的 gate 互斥（阶梯不要踩）

- `seed_pos_fix()`（`DSV41_SEED_POS`）与 `seed_align()`（`DSV41_SEED_ALIGN`）都让 gate **永久拒绝**
  （`dspark_dev.rs:1609-1610`）。⇒ **L6 与「P0-1 co-fix 臂」不可同时开**。
- `unit_dump::enabled()` / `self.unit.is_some()` 同理（`:1617`）⇒ 黄金 harness 逐 unit 落盘时**没有图**。
  这意味着 `DSV41_DSPARK_UNIT_*` 的 parity 跑**不能用来验证图路径**，必须另外用文本红线 + `DIFF_EAGER`。

---

## 4. 必须补的一条（上机前）：draft 臂的 barrier 对称化

**目标**：让 DRAFT_GRAPH 的四个臂（在 gate 开启时）对 `host_barrier` 的到达次数**跨 rank 一致**，
或在分歧时全体退直发。两条路，建议**先做 (a)**（小、稳、与 verify 同构）：

### (a) 每个臂 ENTRY 一次 rendezvous（verify 的「方法 A」搬过来）

只在 gate 已通过的分支里加，**默认路径（gate OFF）逐位不变**：

```
if !self.draft_graph_arm(pos) {          // 仅当 draft_graph_want() 为真时
    if let Some(c) = self.comm.as_ref() { c.host_barrier(); }   // ← 新增：直发也吃一次
    self.draft_body(pos, false)?;
} else if !self.graph_dry_done {
    if let Some(c) = self.comm.as_ref() { c.host_barrier(); }   // ← 新增：DRY 也吃一次
    self.draft_body(pos, true)?;
    self.graph_dry_done = true;
} else if let Some(e) = self.graph {
    …既有各一次…                                                  // replay = 1
} else {
    …既有 capture 前后各一次…                                      // capture = 2
}
```

改完的计数表：**直发 1 / DRY 1 / replay 1 / capture 2**。
capture 的 2 次那一轮是**确定性**的（同轮全体 `!graph_dry_done` ⇒ 全体走 DRY，
下一轮全体 `dry_done && graph.is_none()` ⇒ 全体 capture），
而**capture 失败者当轮仍已吃满 2 次**（`:1636/:1642` 在 `capture_draft` 前后，与成败无关），
⇒ latch 的 rank 从**下一轮**起吃 1 次、与 peer 的 replay 相同，**epoch 不再错位**。

⚠️ 加 barrier 的**前置条件**同 verify：必须在 `!draft_graph_want()` 时**完全不加**
（否则默认路径的 host barrier 数变化，会打破既有的 AR 相位）。

### (b) 跨 rank 投票（更强，verify 的 Plan B）

对 `draft_graph_arm` 的返回值走一次 `comm.unanimous_i32(arm_code)`，
不一致则全体退直发。代价：一次 rendezvous/轮（与 (a) 的 1 次同量），
但能覆盖 (a) 覆盖不了的**未来**分歧源（例如某个 per-rank 的 `.so` 判据）。

> 说明：**(a) 是「次数对齐」，(b) 是「臂对齐」**。verify 两个都做了（(a) 在 `:5353-5357`，
> (b) 在 `:5684-5690`）。(a) 已经能修掉当前唯一的 per-rank 分歧源（`graph_failed`），
> 建议先 (a)、(b) 作为后续。

---

## 5. 测试建议（最小 GPU 次数，单驱动）

> 铁律沿用 `lazy-verify-optimization-path.md §5`：同一远端同时只有一个测试驱动；
> subagent 只做代码/分析；不轮询远端；`cargo check --workspace` 是本地硬门禁。

| # | 会话 | 内容 | 判据 |
|---|---|---|---|
| **G0** | 单卡（TP1，非 TP8） | `DSV41_DRAFT_GRAPH=1` vs unset **同进程背靠背**；看 `[draft_graph] captured the draft chain at pos=…` 是否出现 | 不出现 = gate 没生效（先查 `pos>=win` 与 `.so` 的 `ring_append`/`async memset` 符号） |
| **G1** | 单卡 | 文本红线：出师表逐字 + `1..100` 数数（**不看乱码，看重复/错位**）+ `DSV41_DIFF_EAGER=1` | `[diff]` anchor 每轮一致；零拉丁；`k_acc` 直方图与 gate OFF 相同（图不改数值） |
| **G2** | 单卡 | **覆盖率验证（本轮新增的关键项）**：收尾打印 `graph_captures/graph_replays` + 首个 replay 的 `pos`；对「计数 1..200」记录 `replay 轮数 / 总轮数` | 与 §3.3 的预测同量级（≈50%）；若为 0 ⇒ 短跑陷阱，400 的账要按长生成重算 |
| **G3** | TP8 | **barrier 对称化之后**再跑；`DSV41_DRAFT_GRAPH=1` 连续 6 个请求背靠背（ar5-hang 的复现配方：单请求 bit-identical、第 4 个请求炸） | 无挂死；6/6 请求文本一致；日志 mtime 不停滞 |
| **G4** | TP8 | **故障注入**：人为让**一个 rank** 的 capture 失败（例如该 rank 用不带 `dsv41_ring_append` 的 `.so`，或临时把 `supports_ring_append` 返回 false 的调试开关），验证 (a) 后不挂死 | 该 rank 打 `[draft_graph] capture FAILED … (latched)`，**全请求完成**；这是 §4 的回归测试 |
| **G5** | 单卡 | L6 的联合 A/B：`DRAFT_P3A + MARKOV_SLICED + DRAFT_GRAPH` 三开 | `draft=` 应从 4.28 → ~1.1ms；与 L5（4.28→3.5）**不要重复计数**（L6 吃掉 L5 的大部分） |

**特别提醒 G5 的记账**：`lazy-verify-optimization-path.md §3.1` 的 L5 已经算了 P3c 的 −0.7~0.8ms。
L6 是同一段代码的**更强口径**（4.37ms 的 draft 里 ~3.3ms 是发射开销），所以
**L5 与 L6 不得相加**——开 L6 后，L5 的 P3A/MARKOV_SLICED 收益主要体现在**图内节点减少**（capture 质量），
而不是额外的 ms。

---

## 6. 诚实校准

1. **§2.4 的 barrier 不对称是代码事实（`file:line` 已核对），但「是否已经挂过」无法在本机确认**
   （无 GPU）。按 `ar5-hang` 的历史，它还依赖 `.so`/驱动的时序，**可能一直没触发**——
   所以这条不是「已经坏了」，而是「**上机前必须堵上的已知同类缺口**」。
2. **§3.3 的 ms 是账本口径**：K=4.64 由任务的 411 tok/s ÷ 11.3ms 反推，P≈40 是探针 prompt 的估计；
   G2 是唯一能把它变成实测的会话。**不把 §3.3 当实测。**
3. **−3.3ms/轮 本身未实测**（DRAFT_GRAPH 从未上机）。它来自「draft 4.28ms 里 ~150 发的发射开销」
   的账本推算；`draft-graph-p3c.md §5` 的口径是「launch ~120 → 1」。
   G0/G5 的 `draft_ms` 是唯一的验收。
4. **§3.4 的两条互斥（SEED_POS / UNIT_DUMP）是有意设计，不是缺陷**；
   但报告「L6 收益」时必须确认这些 gate 当时是关的——否则图根本没参与。

---

## 附：一句话总结

**L6（DRAFT_GRAPH）与 lazy 的交互是干净的（每轮一次 replay、同 stream 无互锁、无环），
但它继承了 verify 侧那条 `ar5-hang` 的同类缺口——四个臂的 `host_barrier` 到达次数
（0/0/1/2）没有任何跨 rank 对齐，而 `graph_failed` 是 per-rank 的一次性 latch；
上机前必须先做「每臂 ENTRY 一次 rendezvous」的对称化。
收益侧：`pos >= win` 是位置约束而 pos 每轮跨 k_emit（≈4.6）个位置，
所以「130 步」不等于「130 个位置」——短探针只有 ~50% 的轮能 replay，
`411 tok/s` 是 replay 段稳态，不是全程平均；长生成（≥1000 token）才能兑现 ~90% 的阶梯口径。**
