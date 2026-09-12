# SWALLOW ar5-hang 第 11 次修复预备设计 —— epoch 降级（1497→54）的两个世界

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 代码基线：工作树 HEAD `9dd346d`（`main`）。行号以该树为准。
> 每个结论都给出**源码位置**；无法从源码定论的标 `[未验证]` + 证伪条件。
> 前提：**第 10 次（真正的 epoch pad）的测试正在跑**。本文件是「如果它也失败」的应急设计。

---

## 0. 前提校正（三条——不先读这三条，第 11 次会重复第 1..10 次的错）

### 0.1 ❗D1 文档的摘要**删掉了 `rank=`**——而这正是两个世界的唯一分界

`v5_ledger_pre` / `v5_ledger_note` 的每一行都带 `rank=`（`chain_dev.rs:9500-9503` / `:9530-9533`）。
`dspark-correctness-chain.md:5271-5280` 的摘要写成：

```
pos=22: 所有 rank epoch=1497（同步 ✓）
pos=22 步后（swallowed 臂）: epoch=54   ← 这一行没有 rank
```

**「1497→54 的降级」这个结论，是从摘要里得出的，而摘要丢了判定所需的那一列。**
- 如果 54 的那一行 `rank=` 与 1497 的**不同** ⇒ 世界 B（per-rank 分歧）**或**读到了别的 rank 的地址；
- 如果相同 ⇒ 才需要世界 A（同一 rank 的字被写小）。

⇒ **第 11 次的第一件事不是改代码，是把原始日志按 rank 分组重新提取。** 这是零成本、且一次定性的动作。第 9 次栽在「没有账本」，第 11 次不能再栽在「有账本但摘要丢了主键」。

### 0.2 ❗`epoch` 在同一 rank 上**源码级不可递减**——所以「reset」不是一个自由的假设

全部写 `*epoch` 的 kernel（`grep '\*epoch\s*=' kernels/cuda/*.cu`，共 7 处）：

| kernel | 写入 | v5 路径下是否可达 | 出处 |
|---|---|---|---|
| `p2p_ar_pubred_v5_kernel` | `*epoch = e + 1u` | ✅（每一步 84~165 次）| `ferrite_kernels.cu:9068` |
| `dsv41_v5_epoch_pad_kernel` | `*epoch = e + pad` | ✅（第 10 次的上场者）| `ferrite_kernels.cu:9141` |
| `argmax_xchg_v5_kernel` | `*epoch = e + 1u` | ✅（单行 head）| `dsv41_kernels.cu:8472` |
| `argmax_xchg_v5_rows_kernel` | `*epoch = e + 1u` | ✅（verify 的 m 行 head）| `dsv41_kernels.cu:8538` |
| `p2p_ar_down_v2_kernel` | `*ctr = 0u` | ❌（`ctr` 传 null）| `ferrite_kernels.cu:8622` |
| `p2p_ar_fused_v3_kernel` | `*ctr = e` | ❌（同上）| `ferrite_kernels.cu:8704` |
| `p2p_ar_publish_v3_kernel` | `*epoch = e + 1u` | ❌（v3 不跑）| `ferrite_kernels.cu:8723` |

- `ar_v5()` 默认 **true**（`tp.rs:881-901`：`graph || AR_V5`，而 `DSV41_GRAPH_STEP` 默认 ON），所以 v2/v3 那三个 kernel 在 batched 路径上**不发射**；
- v2/v3 那两个 `*ctr = …` 的 `ctr` 实参在 `tp.rs` 里被显式传 null（`all_reduce_inplace_inner:791`）——**没有指针别名可乘**；
- 构造期唯一一次 `zero_at(staging.ptr, ctr_at + 64)`（`tp.rs:243`）在进程启动时执行一次，`staging` 终生不重分配、`chain.reset()` 不碰它（`chain_dev.rs:4218-4247` 只写 `premix_const` / `dspark_pre_mean` / `premix_r` / `pos_ctr` / `clen`）。

⇒ **在「纯 v5 + 单 Collective + 同上一条流」的前提下，同一 rank 的 `*epoch` 只可能 +1 或 +pad。它不可能变小。**
这条推论把世界 A 的形状**改写了**：不是「某个操作写了 54」，而是下面四种之一：

| 编号 | 机制 | 与「同一 rank 递减」是否相容 |
|---|---|---|
| **A1** | 不是同一个字/地址（读偏、读错 device、读错 Collective）| ❌ 不是真递减，是**观测错位** |
| **A2** | 不是同一条流 ⇒ pad 的 `*epoch = e + pad` 用**过期 e** 覆盖了更新的值 | ✅ 真递减（唯一的真·递减机制）|
| **A3** | 越界写踩到 `ctr_at`（payload 越过槽尾）| ✅ 真递减/乱值 |
| **A4** | 同一 buffer 上有第二条 `Collective`（第二个 `staging`）| ❌ 不是同一个计数器的递减 |

**`[未验证]` A2 的证伪条件**：`v5_epoch_pad`、`p2p_ar_v5`、`argmax_sliced_rows` 三者的 `cudaStream_t` 实参是否恒等（`device.rs` 的 `self.stream`）——若是，A2 死。**这是第 11 次必须先用一条 grep/一行 debug 断言的廉价检查。**

### 0.3 「世界 A vs 世界 B」缺了必须存在的**世界 C**

D1 摘要里的 54 可能压根不是这个进程的 epoch：`dsv41-run.rs:360-361` 建了**两个** `Collective`（`c_small` / `c_big`），`serve.rs:402` 只建一个。若某条路径读的是另一个 Collective 的 `epoch_dev()`，读到的是**另一个计数器**（各自从 0 开始、各自单调）——**两个都「单调」，跨读就「非单调」**。这就是世界 C：**不是谁写小了，是读错了对象。**

---

## 1. 问题 1：`epoch_dev` 是 per-rank 还是全局？

**答案：物理上 per-rank，逻辑上被协议约束为同值——这个「约束」正是全部 10 次失败所在。**

```rust
// tp.rs:329-333
pub fn epoch_dev(&self) -> *mut std::ffi::c_uint {
    (self.staging.ptr as *mut u8).wrapping_add(self.ctr_at) as *mut std::ffi::c_uint
}
```

- `staging` 是**本 rank 私有**的一次 `cudaMalloc`（`tp.rs:237`，大小 `ctr_at + 64`），`ctr_at` 是本 Collective 的固定偏移（`:227`）；
- 每个 rank 一个 `Collective`（`serve.rs:402` 在 rank 线程里建一次）⇒ **world 个各自独立的 u32**；
- 布局（`tp.rs:220-227`）：

```
[0 .. 2*world*bytes)      payload（parity 0 / parity 1）
[stamps_at .. +4*world)   「stored」stamp 行：slot w = rank w 已发布到第几轮
[reduced_at .. +4*world)  「reduced」标记行（v5 不用，恒 0）
[ctr_at .. ctr_at+4)      ★ v5 epoch（= v2 里的 ctr 位；v5 专用）
[ctr_at+4 .. +8)          A4 广播字（`DSV41_AR_SINGLE_POLL`，默认 OFF）
```

⇒ **没有全局 epoch。** 「所有 rank 同步」不是硬件事实，而是「每个 rank 每步发射同样多的轮次」这个**行为契约**的结果。契约一破，就永久分歧（下面的 §3）。

> ⚠️ 顺带记下检索结果：`epoch_dev()` 的调用点只有 **5 处**（`chain_dev.rs:5701` 单行 head、`:6923` m 行 head、`:9499/:9521` 账本、`:9565` pad、`dspark_dev.rs:3226` draft 的 markov 切片 head）。**全部取自 `self.comm`，没有第二来源** ⇒ 世界 C 在**当前**代码里尚无实锤，但 `dsv41-run.rs` 的双 Collective 是它的温床，必须用 §4.0 的 canary 一次性判死。

---

## 2. 问题 2：`ar5_wait_round` 比较的是谁的 epoch？

**答案：比较的是「我的 epoch」对「peer 写进我的 ready 行的 stamp」——即 `peer_epoch >= my_epoch`。两边各用各的本地值，谁也没有把 epoch 送出去。**

```cpp
// ferrite_kernels.cu:8964-8985（OFF 臂，默认；single_poll 默认 OFF，:9168-9175）
const int tr = (int)threadIdx.x;                       // tr = 远端 rank 号
unsigned cur = *(volatile unsigned*)&ready_local[tr];  // ★ 我的 row 里 peer tr 的 stamp
while ((int)(cur - (e + 1u)) < 0) { ... }              // e = *epoch = 我的 epoch
```

而 `cur` 是怎么来的？对称地，在**每一个** peer 上（`:9061-9064`）：

```cpp
if (blockIdx.x == 0 && threadIdx.x < (unsigned)world)
    atomicExch_system((unsigned int*)&ready_tbl[threadIdx.x][my_rank], e + 1u);
//                              peer tr 的 row      ★ 我的列号
```

即：**我把我自己的 `e+1` 写进每个 peer 的 `row[my_rank]`**。所以

| 量 | 语义 |
|---|---|
| `e` | **本 rank 的** epoch（本轮开始前） |
| `ready_local[tr]` | **peer `tr` 的** epoch+1（它写进我 row 的那一列） |
| 判据 `cur - (e+1) >= 0` | **`peer_epoch >= my_epoch`** |

**三条直接后果（第 11 次的设计必须建立在这三条上）**：

1. **等待是「单向不落后」检查，不是握手。** 我不需要 peer 和我「同时」到达，我只需要 peer **不落后于我**。
2. **失败是不对称的**：epoch **大的**那个 rank 永远等（`[ar5-hang]` 的报点）；epoch **小的**那个 rank 反而一路畅通，并且**静默**读它自己 parity 槽里的陈旧 payload（`(my_e&1)` 与 `(peer_e&1)` 不同则读到的是上一轮的列）。——**日志里 `need` 是 leader 的，`cur` 是 laggard 的。** 这是读 D1 数据时最容易搞反的一处。
3. **`e` 与 `cur` 参与的是同一个整数比较，所以任何「谁比谁大」的推断都只在两者是同一条时间轴上才有意义。** 世界 B 说「1497 是 rank A、54 是 rank B」——在这个语义下**完全自洽**：A 等 B 的 54 涨到 1498，永不到来。

⇒ **结论：世界 B 在架构上不但可能，而且是这套协议的「设计缺陷面」。** v5 用「绝对 stamp + 单调比较」换掉了 v2 的 `seen[]`，代价是**它把「所有 rank 的 round index 相等」从一个被检查的不变量，变成了一个只被祈祷的假设。**

---

## 3. 问题 3：54 的数学

先把不可能的先消掉（都是源码级判定）：

| 猜想 | 判定 | 依据 |
|---|---|---|
| u32 wrap | ❌ | 1497 ≪ 2^32，wrap 需要 ~42 亿轮 |
| 读偏到 stamp 行 | ❌（会读到 `0` 或 `peer_e+1`，都是 epoch 量级）| 布局 `tp.rs:220-227` |
| 读偏到 `reduced` 行 | ❌（恒 0）| v5 不写 `reduced`（`tp.rs:773` 记为 unused） |
| payload 越界踩 `ctr_at` | ⚠️ 需 `len > bytes + 8*world`（≈64 B）| 槽位= `bytes`，`ctr_at` 在 2*world 个槽之后（`tp.rs:227`；`ar_bytes = max(hc_dim, 6*dim)*4`，`serve.rs:400`）——**量级不对，但要留一条 canary** |
| `54 = 2 × 27`（= 2×第 9 次报的 `gap`）| ❌ 巧合 | 27 是 `need-cur` 的一次观测值（`fb4acf4`），不是任何几何常量；`81 = 2*n_layers+1` 一变，全部此类分解失效 |
| `54 = 9 × VERIFY_ROWS` / `= 2/3 × 81` | ❌ 巧合 | 无对应的源码常量 |

**保留的两个读数**（都能被 §4.0 的一次运行判死）：

- **读数 ①「清零后走了半步」**：epoch 被写过一次小值（0 或某基值），之后又推进了 54 轮。**证据支持：54 恰好落在「一个 step 的轮次足迹」这个量级内**（84 / 165 的一半上下）。这正是「某个东西把字清了，然后一个部分步把它推到 54」的形状。
- **读数 ②「冻结的 laggard」**：54 是某个 rank 在**很早**的时候停住的值（它 hang 了 ⇒ 它的设备时线冻结），而对齐的 rank 已经走到 1497。**证据支持：`dspark-correctness-chain.md:5277-5279` 的 `pos=22 步后 = 54` 与 `pos=23 步前 = 54` 是同一个值**——**一个「活着并在推进」的计数器不会在两行之间一动不动**。冻结是 laggard 的签名。

**这两个读数有一个共同的、更硬的量化判据**：

> 1497 − 54 = **1443**。
> 任何 **per-step 不对称**的单步上限是 **81**（臂边界，`swallow_missing_rounds = 2*n_layers+1`，`chain_dev.rs:1925`），累积也才 165/step。
> 若两行是**同一 rank 同一时刻**，-1443 需要 **≈9~17 个 step 的持续漂移**——与 `pos=22 之前 all rank = 1497（同步 ✓）` 直接矛盾。
> ⇒ **在同一 rank 的假设下，1443 这一跳在「漂移」类机制里是解释不通的；它必须来自一次「覆盖/清零」（世界 A 的 A2/A3），或来自「读到了别的计数器」（世界 C）。**
> ⇒ 反过来，如果两行 `rank=` 不同，则世界 B 成立，且**「all rank = 1497」那句摘要必须是错的或只覆盖了 leader 子集**。

**结论：54 本身不能反推根因，1443 这个差值可以。** 第 11 次必须先把「同一 rank 吗」这一列拿回来。

---

## 4. 第 11 次设计

### 4.0 `11.0` D1 加固（**必做，零行为改变，一次运行判死三个世界**）

第 9 次的教训是「没有账本」；D1 上了账本，但账本**只读了一个字**、而且**摘要丢了主键**。加固三件事：

**(a) 主键 + 单调断言（把「非单调」变成日志里的一行，而不是事后的推理）**

在 `DevChain` 加 `v5_ledger_seen: RefCell<HashMap<usize /*rank*/, u32>>`（进程内每 rank 一条线），`v5_ledger_note`/`v5_ledger_pre` 里：

```rust
let prev = seen.insert(self.rank(), epoch);
if let Some(p) = prev {
    if epoch < p {
        eprintln!("[v5-ledger-RESET] pos={pos} rank={} prev={p} now={epoch} drop={}",
                  self.rank(), p - epoch);
    }
}
```
> 成本：gate 关时 0；gate 开时一个 HashMap 写。**不引入新 gate**，复用 `DSV41_V5_LEDGER`。

**(b) 把「同一个字」证出来（canary）**

`Collective::new` 里预留的 `ctr_at + 8 .. ctr_at + 64` 目前是空的（`tp.rs:237` 只要求 `ctr_at + 64`）。在**同一个 pad/AR kernel 里**顺带写一个 canary：

```cpp
// dsv41_v5_epoch_pad_kernel 末尾（与 *epoch 同一线程、同一行序列）
*(volatile unsigned*)(epoch + 2) = e + pad;   // canary：与 epoch 恒等
```
账本**同时读** `epoch` 与 `epoch+2`：**两值恒等 ⇒ 读的是同一个字（排除世界 C 的读偏）；不等 ⇒ 立刻抓到读错对象。** 再顺带读 `ready_local[rank]`（我自己那一列 = 我自己上一次发布的 stamp）作第二个独立参照。

**(c) 把「同一条流」证出来（A2 的廉价证伪）**

在 `v5_epoch_pad_swallow` 首次调用时打印一次 `cudaStream_t` 的值，与 `p2p_ar_v5` / `argmax_sliced_rows` 的实参比对（或在 `device.rs` 里加一个 `debug_assert_eq!(s, self.stream)`）。**这是 A2 唯一可存在的机制，必须一次性排除。**

**(d) 输出格式改为机器可解析且不丢字段**：每行固定 `pos rank epoch arm delta canary self_stamp`。摘要由脚本 `awk` 生成，**禁止人工转述**（第 9/10 次两次转述都丢了关键列）。

### 4.1 世界 A：消除「覆盖/清零」源（若 4.0 的 canary 相等、且同 rank 出现 `[v5-ledger-RESET]`）

按可能性排序，逐条是**可执行的**：

| # | 机制 | 修法 | 风险 |
|---|---|---|---|
| **A-1** | **A2：pad 用过期 `e` 覆盖**（跨流/跨图重放）| 给 `v5_epoch_pad_kernel` 加**单调写**：`atomicMax(epoch, e + pad)`（u32 单调），并把 pad 与 AR 的流用 `debug_assert` 钉死同一 stream | 低（atomicMax 语义与 `e+pad` 在正常路径下等价）|
| **A-2** | **A3：越界写踩到 `ctr_at`** | 把 v5 epoch **搬出 payload 邻域**：在 `ctr_at + 8` 起单独留 8 B 作 `epoch_at`（`ctr_at` 从此只作 v2 的 ctr 位），并用 4.0(b) 的 canary 双向夹住；同时在 v5 的 `all_reduce_inplace` 补上 `publish()` 里那条**被漏掉的** `assert!(len <= self.bytes)`（`tp.rs:636` 只在 v2 路径有，v5 路径 `:715-736` 没有）| 中（改布局，需 micro-bench 回归）|
| **A-3** | **A4：第二个 Collective 的 epoch 被读到** | 在 `device.rs` 的 `epoch_dev()` 消费点加 `debug_assert`（同一进程只有一个 `staging_base()` 被 `chain.comm` 引用）；把 `dsv41-run.rs:361` 的 `c_big` 显式标注为「不得用于任何 v5 轮次」| 低 |
| **A-4** | **未枚举的写者** | 用 `cuda-memcheck`/`compute-sanitizer --tool memcheck` 跑一次短 SWALLOW：越界写会被点名（这是唯一能在没有新 hypothesis 时**枚举**写者的手段）| 低（需 GPU 机）|

> **判据**：A-1/A-2 修完，`[v5-ledger-RESET]` 必须为 0 行。

### 4.2 世界 B：让「轮次相等」从祈祷变成构造（若 4.0 显示跨 rank 分歧但每 rank 内部单调）

这是**更可能的那个世界**（§2 的三条后果 + §0.2 的「不可能递减」）。已有三条路，按推荐度：

#### **11-B ★ 主推：动态 pad（把第 10 次的常数 pad 升级为 per-step 的 epoch consensus）**

第 10 次的 pad 是 `swallow_missing_rounds(n_layers) = 2*n_layers+1`（`chain_dev.rs:1925`）——**一个写死的 81**，只覆盖「swallowed 少跑 step_dev」这**一种**不对称。世界 B 的枚举里还有别的源（第 9 次设计 §2.2 的 (a) head geom ±1、(d) 臂边界、capture 臂的两个 barrier、lazy route），**常数 pad 对它们无效**——这就是「第 10 次也可能失败」的结构性理由。

**改法：在每个 step 的同一位置，各 rank 公布自己的 epoch，取 max，把落后者 pad 到 max。** 复用**已经落地的 pad kernel**（`dsv41_v5_epoch_pad_kernel`），只换 `pad` 的来源：

```rust
// dspark_spec_step 的公共出口（chain_dev.rs:8041 那一带，pos_ctr D2H 旁边）
// ——每个 step 一次：公布 → 取 max → 落后者补差。常数项消失。
fn v5_epoch_consensus(&self) -> Result<()> {
    let Some(c) = self.comm.as_ref() else { return Ok(()) };
    if !c.uses_v5() { return Ok(()); }
    let me = self.dev.download_u32(c.epoch_dev() as *const c_void)?;
    let max = self.comm_max_epoch(me);   // 共享 AtomicU32[world] + 既有 SpinBarrier
    if me < max {
        let _ = self.dev.v5_epoch_pad(c.peer_stamps_u32(), c.epoch_dev(),
                                      self.world() as i32, self.rank() as i32, max - me)?;
    }
    Ok(())
}
```
- **为什么这是对的**：pad 的合法性论证（第 9 次设计 §3.2 的四条）**不依赖 81 是常数**——只需要「跳过去的轮次无 payload、无读者」+「parity 与下一轮真实 AR 一致」。取 `max - me` 满足前者；parity 由「所有 rank 最终同值」保证（不再是「81 是奇数」）。
- **cost**：每 step 1 次 4 B D2H + 1 次 H2D + 1 次 host barrier ≈ 几十 µs，相对 ~6 ms 的 step 可忽略；**且它一次覆盖 (a)/(d)/capture/lazy 全部不对称源**——因为它量的不是「谁少跑了几轮」，而是「谁落后了」，与**原因**无关。
- **对世界 A 也部分免疫**：如果某个 rank 被清零，consensus 会把**它**补到 max（laggard 被拉平），rift 当场闭合。**唯一的失效面**是「清零发生在 consensus 之后、下一轮 AR 之前」。

> ⚠️ **这条修法与 `host_barrier` 的兼容性必须验证**：consensus 是一种 host 会合；`SpinBarrier` 是到达世代计数器，**多一次会合 = 之后每个 barrier 错一代**（`tp.rs:32-38` 的注释就是为这个写的）。⇒ 实现时必须**复用同一个 barrier 调用点**，或走**独立的 `RankVote` 式独立世代**（`tp.rs:108-162` 已经提供了「独立世代」的现成原语）。**这是 11-B 唯一的真风险，必须写进验收。**

#### **11-B′ 备选：把 consensus 做成纯设备侧（graph 更友好）**

把 pad kernel 扩成 `dsv41_v5_epoch_sync_kernel`：`atomicExch_system(peer_board[r][my_rank], *epoch)` 公布 → 需要一个**与 epoch 无关的公共参照**才能等所有 peer 公布。可用的公共参照是「本轮 AR 的 round index」——这正是缺的东西。⇒ 除非引入第二个计数器（`step_seq`，由 host 每步 +1 并 H2D，与 epoch 无关），否则**纯设备侧共识做不到 lockstep-free**。`[未验证]`，但在 11-B 失败前不必走这条。

#### **11-B″ 不推荐：把 wait 从绝对改成相对**

`ar5_wait_round` 改回 v2/v3 的 `seen[] + (int)(cur - prev) > 0`（`ferrite_kernels.cu:8729-8734` 的形态）。**能解决「谁落后」**，但**把 v5 用绝对 stamp 换来的「无 per-block `seen[]` 状态、multi-block 安全」全部还回去**——`p2p_ar_pubred_v5_kernel` 的整个存在理由就是那一条（`:9044-9049`）。**只有在 11-B 也失败、且证据指向「同一 step 内先落后、后被 pad 追平」的时序问题时**才考虑。

### 4.3 世界 C：读错对象（若 4.0 的 canary 与 epoch **不等**）

修法即 4.0(b) 的 canary 落地 + `epoch_dev()` 消费点收敛到单一来源（4.1 的 A-3）。**这一支的修复量最小，但前提是 canary 先上场。**

---

## 5. 第 11 次的判定树（一张图，按 4.0 的输出走）

```
4.0 跑一次（SWALLOW + V5_LEDGER=1，短 prompt）
│
├─ 每行 [v5-ledger] 有 canary 且 canary == epoch，同 rank 无 RESET
│   └─ 跨 rank 差逐 step 累积 ⇒ ★世界 B ⇒ 11-B 动态 pad（主推）
│
├─ 出现 [v5-ledger-RESET]（同 rank epoch < prev）
│   ├─ canary == epoch ⇒ 真·字被写小 ⇒ 世界 A ⇒ 4.1（A-1 atomicMax 先上，再 A-2 布局隔离）
│   └─ canary != epoch ⇒ 世界 C ⇒ 4.3
│
└─ 一个 step 内 rank 间出现 O(1000) 的差（而非 O(81)）
    ⇒ 「漂移」解释不通（§3）⇒ 世界 A/C，且**优先怀疑清零**：直接上 A-2 的布局隔离 + canary 夹持
```

> 关键：**先跑 4.0，不要先猜世界。** 第 1..10 次里至少有 6 次是先有 hypothesis 再去凑证据。

---

## 6. 验收（第 11 次的通过线，与第 10 次共用一套）

1. `[v5-ledger-RESET]` = **0 行**；每 step 跨 rank 的 `epoch` **逐字相等**（这是 11-B 的直接验收：它不再依赖 81 这个常数，而是依赖「max 被取到」）。
2. `[ar5-hang]` = 0，且**分开计数**（`argmax_rows` 带 `rows=` 的 vs `pubred` 不带 `rows=` 的）——第 9 次设计 §2.3 的 10× watchdog 差必须一直被尊重。
3. `[verify_graph] captured` 与 `[draft_graph] captured` 在长跑中出现（`pos >= 128`，`dspark_dev.rs:1607-1613`），否则「图从未生效」与「图生效」外部不可区分。
4. 吞吐只用 `V5_LEDGER=0` 那一跑；`k_acc` 序列与 base 逐项对照；出师表 1000 tok 红线：零额外拉丁。
5. `cargo check --workspace` EXIT=0 + `bash scripts/batched_400_v2.sh`（**不要手改它的 gate 串**；`.so` 时间戳与 binary 一致）。

---

## 7. 改动清单（供尚书省分派；按「如果第 10 次失败」的优先级排）

| # | 文件 | 改动 | 优先级 | 风险 |
|---|---|---|---|---|
| 1 | `crates/ferrite-models/src/dsv41/chain_dev.rs` | D1 加固：`v5_ledger_seen` + `[v5-ledger-RESET]` + 读 `epoch+2`(canary) + 读 `ready_local[rank]` + 行格式带 rank | **P0（无论第 10 次成败都该做）** | 极低 |
| 2 | `kernels/cuda/ferrite_kernels.cu` | `dsv41_v5_epoch_pad_kernel` 写 canary `epoch+2`；`*epoch` 改 `atomicMax`；`dsv41_v5_epoch_pad` 首次打印 stream 值 | P0/P1 | 低 |
| 3 | `crates/ferrite-models/src/dsv41/chain_dev.rs` | `v5_epoch_consensus()`（11-B，复用 pad kernel + `RankVote` 独立世代）| P1（世界 B） | 中（barrier 世代，必须验） |
| 4 | `crates/ferrite-models/src/dsv41/tp.rs` | `ctr_at+8` 划出 `epoch_at`；v5 路径补 `assert!(len <= self.bytes)` | P2（世界 A/A3） | 中（布局） |
| 5 | `crates/ferrite-dsv41/src/bin/dsv41-run.rs` | `c_big` 标注「不得参与任何 v5 轮次」 | P2（世界 C） | 低 |
| 6 | `scripts/batched_400_v2.sh` | 摘要生成脚本化（`awk` 出 per-rank 逐 step 表），禁止人工转述 | P0（第 9/10 次两次都栽在转述） | 低 |
| 7 | `docs/agent/dspark-correctness-chain.md` | 追加：本设计 + 4.0 的判定树 | P1 | 低 |

**不做**：修法 C（统一 `argmax_rows` 行数）——第 9 次设计 §0.2 已证对轮次是 no-op；11-B″（相对等待）在 11-B 未证伪前不做。

---

## 8. 一句话总结

前 10 次都在问「**谁少发了几轮**」（于是去对齐臂、去补常数 81）。
`ar5_wait_round` 的源码（`ferrite_kernels.cu:8976`）说的却是另一件事：**它比的是「peer 的 epoch 是否 ≥ 我的」**——所以真正要问的是「**谁落后了**」，而落后的原因可以是臂、是 head geom、是 capture 臂、是清零、是读错对象。
**第 11 次的主修（11-B）就是把问题从「少了多少」改成「落后多少」**：不再假设 81，而是每步量一次 max，把落后者补平——**与原因无关，因此对 10 次里每一次的新 hypothesis 都免疫。**
而在此之前，**先把 D1 的 `rank=` 列拿回来**（§0.1）——否则两个世界永远分不开。

---

*工部 · 只读分析 + 本文件（唯一产出），未执行 GPU 命令、未改动任何源码。*
*所有行号以工作树 HEAD `9dd346d` 为准；无法从源码定论的推断均显式标注 `[未验证]` 并给出证伪条件。*
