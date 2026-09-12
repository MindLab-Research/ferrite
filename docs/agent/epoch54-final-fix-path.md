# epoch 54 的最终修复路径 —— 流断言重瞄 → 设备侧写证词 → 分支修复 → 验收

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 代码基线：工作树 HEAD（本日）。行号以该树为准。每个结论给**源码位置**；推断标 `[未验证]` + 证伪条件。
> 前序：`docs/agent/epoch-54-alternate-fix-design.md`（P0–P4 路线）与 `dspark-correctness-chain.md` 末尾判决。本文是**判决的最终收敛**，不重复前序，只改其必要处。

---

## 0. 摘要：对任务书两处前提的源码级修正

| # | 任务书前提 | 源码事实 | 后果 |
|---|---|---|---|
| **M1** | 「pad kernel 在 stream A，AR 在 stream B——加 `debug_assert_eq!(stream)` 一次运行判死」 | **所有 v5 入口都硬编码 `self.stream`**（`device.rs:3250 / 4568 / 4593 / 3208`）。在 Device 层加 `debug_assert_eq!(stream)` 是**拿字段和自己比**——恒真，永不触发。 | 该实验**按原样判不死**。断言必须重瞄到「有序域」而不是「字段相等」（§3）。 |
| **M2** | 「pad 的过期 e 覆盖 → 真降级」 | 全树写者都形如 `*epoch = (入口读到的 e) + k`，`k ≥ 1`（pad 是 `k = pad ≥ 1`）。∴ **任何写者写出的值 ≥ 它自己读到的 e + 1**。 | 「取过期 e」只能造成**滞后/卡住**，**不能**把 999 一步打成 54（§2.1）。要出 54，必须有写者读到 `≤ 53`——那是**另一个字**或**被清过的字**，不是「过期」。 |

**因此最终路径不是「验证流断言 → 修」，而是：**

1. **P0（0 代码）**：先解决证据自相矛盾——`RESET=0` 在给定数据下**源码级不可能**（§4）。不解决它，后面都在修一个误读。
2. **P1（一次运行判死）**：把流断言换成**能失败**的版本，并用**设备侧写证词**（kernel 自己记录 `e_read / e_wrote`）取代 host 侧推断——host 看不到 stream 序（§5）。
3. **P2**：按 P1 的证词选分支修复（§6）。
4. **P3**：验收（§7），并保留前序文档那条独立支线（epoch 只要求**全 rank 同值**，不要求值是多少 ⇒ 「epoch 54 ⇒ EOS」仍是假设）。

---

## 1. 源码事实（先钉死，后面全部引用它）

**F1｜v5 三入口共用同一个 stream 字段。**
- `Device::v5_epoch_pad`：`f(ready_tbl, epoch, world, rank, pad, self.stream)`（`crates/ferrite-models/src/dsv41/device.rs:3250`）
- `Device::p2p_ar_v5`：`…, self.stream`（`device.rs:4568`）；`p2p_ar_pubred_v5`（`:4593`）
- `Device::argmax_sliced_rows`：`…, self.stream`（`device.rs:3208`）
⇒ 在 Device 层任何 `debug_assert_eq!(x, self.stream)` 都恒真。**流不可能在 Device 层分叉。**

**F2｜主流是 blocking，side 流是 NonBlocking。**
- 主 stream：`cudaStreamCreate`（`crates/ferrite-kernel/src/devrt.rs:601`）——**blocking**（legacy 默认流语义）。
- side stream：`cudaStreamCreateWithPriority(…, CUDA_STREAM_NON_BLOCKING, …)`，fallback 才用 `cudaStreamCreate`（`devrt.rs:161` / `:173`；常量 `:96`）。
- ledger 的读是 **blocking `cudaMemcpy` D2H**（`devrt.rs:1302`）。
⇒ `cudaMemcpy` 与**主流有序**，与 **NonBlocking side 流无序**。**任何在 side 流上推进 epoch 的 kernel，ledger 都可能在它落盘前读到旧值，且与主流上的 pad/AR 无任何序**。这才是「流」在树里**真实存在**的漏洞面（今天是潜在的：v5 各入口都在主流，但 `DSV41_DUAL_CHAIN` 把 kv/MoE-shared 分到 `side_stream2`，`chain_dev.rs:10599/12800/14796/15830`）。

**F3｜pad 是 direct launch，位于全步 AR 之前，同一 stream。**
- `dspark_spec_swallowed` 步首调用 `v5_epoch_pad_swallow`（`chain_dev.rs:8734-8735`），随后才 `dspark_snapshot`→`step_rows`（`:8724` / `:8761` 的 AR）。
- 同一 stream 上 host 序 = device 序 ⇒ **pad 先于本步 AR**，这点源码级成立。

**F4｜pubred 在 kernel 开头就推进 epoch，在 wait 之前。**
`p2p_ar_pubred_v5_kernel`：`const unsigned e = *epoch;`（`kernels/cuda/ferrite_kernels.cu:9052`）→ 先 stamp（`:9062`）→ `*epoch = e + 1u;`（`:9067-9068`）→ 再 `ar5_wait_round(…)`（`:9071`）。
⇒ 一轮的 epoch 推进在**等待之前**完成。

**F5｜pad kernel：读 e 在头，写 e+pad 在尾。**
`dsv41_v5_epoch_pad_kernel`：`const unsigned e = *epoch;`（`:9134`）→ stamp `e+pad`（`:9136`）→ `*(epoch+1) = e+pad; *epoch = e+pad;`（`:9140-9141`）。

**F6｜ledger 的 RESET 是 per-rank 的下降断言。**
`v5_ledger_probe`：`let prev = self.v5_ledger_seen.borrow_mut().insert(rank, epoch); if let Some(p) = prev { if epoch < p { eprintln!("[v5-ledger-RESET] …") } }`（`chain_dev.rs:9661-9670`）。
`v5_ledger_pre` / `v5_ledger_note` **都**调用 `probe`（`:9674-9697`）⇒ 二者共享同一张 `seen` 表。

**F7｜布局：`ctr_at` 是 epoch 所在字，尾 64 B 是预留。**
`ctr_at = reduced_at + world*4`（`tp.rs:338`），`staging = alloc(ctr_at + 64)`，构造期 `zero_at(…, ctr_at+64)`（`:348-354`）；canary 写在 `ctr_at+8`（`:313-318`，`canary_dev()` `:463-467`）。**`ctr_at+0..+8` 是 epoch（+0）与 A4 word（+4）——被合法代码写**；`+8` 起才是无人写的尾。

---

## 2. 判决复核：流假设被**重瞄**，不是被推翻

### 2.1 不等式：单步下降 999 → 54 不可能由「过期 e」产生

设某写者入口读到的值为 `r`，写出的值为 `r + k`（`k ≥ 1`，F4/F5；argmax/融合 epilogue 同为 `e+1`）。
**任何写入的值 ≥ 该写者读到的值 + 1。** 对某一时刻观测到的 `v`，必存在一个写者读到 `r ≤ v − 1`。

若 `v = 54`，则必有一个写者读到 `r ≤ 53`。而同一字在**一步之前**是 999（`pos=15 pre`）。
⇒ **「手上有过期但仍在 999 量级的 e」只能让它写 1000…，不可能写 54。** 「过期」解释的是**滞后**（少算），不是**跌破**（变小）。

**这是判决的关键加强**：`epoch54-source-and-stop` 说的「pad 的 e+pad 覆盖了 999」在**算术上**要求 pad 读到 `≤ 53`——这已经是「读到了另一个字 / 被清过的字」，而**不是**「读到了同一个字的过期副本」。流假设要保留它的**价值**（F2 无序域确实存在），但它的**表述**必须换成：**「写者读到的是一个 ≈0 或来自别处的字」**。

### 2.2 那把「≈0 的读」从哪来？剩下的只有两条，且互斥

| 形状 | 机制 | 一句话判据 |
|---|---|---|
| **A2′ 同一个字被清零后重数** | 某个**越界写**（或一次性清零）把 `ctr_at` 写成 0，本步余下的真实轮次把它带到 **54**（54 < 单步 84/165 ⇒ 「带不到满」的截断值，与「从头计数」的观感一致） | `[v5-ledger-RESET]` 出现；canary 可能**仍为 magic**（见 2.3 的盲区） |
| **C′ 读到了另一个对象** | ledger 读的 `epoch_dev()` 与写者写的不是同一个字（第二 Collective / 别的 staging / side 流上的副本）| canary == magic、**无** RESET；`[v5-ledger-stream]` 的 stream 与写者所在流不在同一有序域 |

### 2.3 盲区复核（任务书的 CANARY=0 判断，源码级成立，但比表述更宽）

`ctr_at+8` 是 canary，而合法写只覆盖 `+0..+8`（F7）。锚点差 8 B。**但真正的盲区不止「差 8 B」**：
- 若越界来自 **`reduced` 数组的向上溢出**（`reduced` 的最后一个字在 `ctr_at-4`），那么 `reduced[world] → ctr_at+0`、`reduced[world+1] → ctr_at+4`、`reduced[world+2] → ctr_at+8 = canary`。
  ⇒ **宽度恰为 1–2 字的溢出：恰好踩死 epoch（和 A4 word），canary 完好。** 这不是「差 8 B」，是**结构性的**——`ctr_at` 与被踩对象之间**没有 guard**。
- 所以 `CANARY=0` **不能**排除 A2′（前序文档把它归到「无 CANARY ⇒ 世界 C」是过强的；本文修正）。

---

## 3. P0：先把证据的自相矛盾解决（**零代码**，必须第一步）

### 3.1 矛盾在哪

按 F6，同一步内两条行**必然**走同一张 `seen` 表：
`pos=15 pre` → `insert(rank, 999)`（prev = None，不报）→ `pos=15 note` → `insert(rank, 54)`（prev = **Some(999)**）→ `54 < 999` ⇒ **`[v5-ledger-RESET]` 必须打印**。

观测是 `RESET=0`。所以在给定数据下，**只有三种可能**，必须先定性：

1. **两行不是同一实例/同一序列**：同 rank 上存在**多个 `ChainDev`**（同一进程解多条序列/多请求），`pos` 归属不同实例 ⇒ 每个实例的 `seen` 各自单调，`RESET` 永不触发。**这会把「一步内 999→54」证伪为「两个序列在同一 pos 上的两行」**——判决的前提直接塌。
2. **note 的 probe 提前 return**：`!v5_ledger()` / `comm == None` / canary 或 epoch 读失败（`chain_dev.rs:9615-9657`）。但 read 失败会打 `[v5-ledger] … read failed`，`pos=15 note` 行本身就不会出现——除非被摘要吃掉。
3. **摘要丢列**（前序 fix-11 §0.1 点名的病）：`awk` 重提时把 `rank=`（和/或实例位置）丢了，把两条序列的行拼成了一条。

### 3.2 动作（不改一行代码）

```bash
# 原样重提，禁止人工转述；逐列保留，按 (pos, rank) 排序后人工看「同 rank 的两条序列」
grep -E '\[v5-ledger(-pre|-RESET|-CANARY|-stream)?\]' <原始日志> \
  | awk '{for(i=1;i<=NF;i++) printf "%s ", $i; print ""}' | sort -s -k2,2n -k3,3n
# 必须同时数出三个计数（同一次 awk，不要另开一跑）
grep -c '\[v5-ledger-RESET\]'   <原始日志>
grep -c '\[v5-ledger-CANARY\]'  <原始日志>
```

**门槛**：拿不到「`RESET` 行数 = 0 且两行同实例同序列」这两个同时成立的事实之前，**不进入 P1/P2**。

---

## 4. P1：把流断言改成**能失败**的版本 + 设备侧写证词（一次运行判死）

### 4.1 为什么 host 侧断言不够（F2 的直接后果）

host 能看到的只有「我传了哪个 stream 字段」。**它看不到 device 上的实际提交序**，也看不到 `cudaMemcpy` 与 NonBlocking side 流之间的无序。所以判死必须**在设备侧留下证词**。

### 4.2 断言重瞄（3 处，全部是新增，不改语义）

| 站点 | 断言 | 为什么它能失败 |
|---|---|---|
| `Device::v5_epoch_pad`（`device.rs:3238`）| `debug_assert_eq!(stream, self.rt.stream())` **+** `debug_assert!(!self.dev.capturing())` | 前者在 Device 层恒真（保留作**文档化不变量**）；**后者能失败**：pad 若被录进 capture 就是「第 9 种幻影修复」（录制不执行）。`capturing()` 已存在于 `devrt.rs:1496`。 |
| `Device::v5_epoch_pad` + `p2p_ar_v5*` + `argmax_sliced_rows` | `eprintln!("[v5-stream] {who} stream={:?} epoch_ptr={:p}")`（每入口一次，`OnceLock`）| 打印**指针**——指针不同就是世界 C′，同一 stream 也救不了 |
| 调用点 `v5_epoch_pad_rounds`（`chain_dev.rs:9736-9742`）| `debug_assert_eq!(c.epoch_dev(), ledger_epoch_ptr)`（把 ledger 读的那个指针与 pad 传来的指针做同一性比较）| **直接判 C′**：这是任务书「流断言」的**正确靶子**（写者与读者是否同一个字），而不是 stream 字段 |

### 4.3 设备侧写证词（**本次判死的核心**）

**为什么必须有**：只有 kernel 自己知道「我读到的 e 是多少」。host 的 ledger 只能看到「写完之后是什么」。当 §2.1 已证「过期 e 不可能产生 54」，我们需要的信息是**「到底有没有写者读到 ≤53」**——这只能在写点记录。

**设计**（全部 gate 在 `DSV41_EPOCH_WITNESS=1`，`OnceLock`，与全树纪律一致）：

```c
// 新增，per-rank 环形：[8] u32 一组 { writer_id, e_read, e_wrote, clock_lo }
// 每个 epoch 写者，在同一线程、同一次写之前/之后各记一条（atomicAdd 推进 head）
// writer_id: 1=pubred 2=pad 3=argmax_xchg 4=argmax_xchg_rows 5=融合epilogue
if (witness_enabled && threadIdx.x == 0) {
    unsigned s = atomicAdd(head, 1u) & (SLOTS - 1u);
    w[s].writer = ID; w[s].e_read = e; w[s].e_wrote = e + k; w[s].clk = (unsigned)clock64();
}
```

改动点（与 F4/F5 同处，各 +4 行）：
- `p2p_ar_pubred_v5_kernel`（`ferrite_kernels.cu:9067` 前后）
- `dsv41_v5_epoch_pad_kernel`（`:9134` / `:9141`）
- `argmax_xchg_v5_kernel` / `argmax_xchg_v5_rows_kernel`（`dsv41_kernels.cu:8472` / `:8538`）
- 融合 epilogue 的 gemv（`ferrite_kernels.cu:9395` / `:9523`；`dsv41_kernels.cu:4453-4455`）

**读取**：新增 `Device::epoch_witness_dump()`（一次 D2H，`head` + `SLOTS×4` u32），在 `v5_ledger_note` 之后、gate 开启时调用，一行一条 `[v5-witness] pos=… writer=… e_read=… e_wrote=… clk=…`。

### 4.4 判定表（**一次运行走完，互斥**）

```
P1 跑一臂：DSV41_V5_LEDGER=1 DSV41_EPOCH_WITNESS=1 SWALLOW_STEP=1 DYNAMIC_PAD=1（与失败跑的 gate 串逐字一致）

读 [v5-witness] 与 [v5-ledger*]
│
├─ 存在一条 witness 的 e_read ≤ 53（而同一时刻 ledger 曾是 999）
│     ⇒ 世界 A2′/C′：有写者读到了「另一个/被清过的字」
│     ⇒ 看它的 epoch_ptr（4.2 的打印）
│        ├─ 指针 == ledger 的 ⇒ 该字在设备上确实被清过 ⇒ 走 §5-A2′（越界/清零）
│        └─ 指针 != ledger 的 ⇒ 世界 C′（第二对象）⇒ 走 §5-C′
│
├─ 全部 witness 的 e_read 都是 999 量级、e_wrote 单调，但 ledger 读到 54
│     ⇒ **读者问题**（观测错位/读序）⇒ 走 §5-D（ledger 读改成 stream-ordered D2H）
│
├─ witness 里没有任何 54 的写入，且 ledger 的 54 行与 999 行**不同实例**（P0 的结论）
│     ⇒ 「同一步内降级」是拼行假象 ⇒ **判决前提被证伪**，退回 dspark-correctness-chain.md 重定义问题
│
└─ witness 抓不到任何异常，且所有写者/读者指针一致、流同一有序域
      ⇒ 保留 §5-D + 上 compute-sanitizer memcheck（枚举越界，不依赖假设）
```

**这一步的价值**：它把「流断言是否成立」从一个**永不触发的断言**，变成**一个可读的、设备侧的事实**。并且它**不依赖** pad 是否真的在别的 stream——无论答案是什么，输出都定向到唯一的下一分支。

---

## 5. P2：按 P1 分支修复

### 5-A2′（越界写 / 清零踩到 ctr_at）——**推荐先做的结构性加固**

1. **在 `reduced` 与 `ctr_at` 之间插入 8 B guard**（布局，`tp.rs:338`）：
   `ctr_at = reduced_at + world*4 + 8;`，guard 区写一个与 canary 同族的 magic，ledger 一并读。**这直接补上 §2.3 的结构性盲区**（1–2 字的向上溢出从此必被捕获）。
2. `ctr_at + 8` 的 canary **扩成 4 个字**（`+8/+16/+32/+48`，全在 64 B 尾内），全读全比。
3. **P1 的 witness 一旦指认某写者的 `e_read ≤ 53`**，立刻用 `compute-sanitizer --tool memcheck` 在短跑上枚举越界写（不需要先有 hypothesis）。
4. 顺手补 v5 路径缺失的长度断言 `assert!(len <= self.bytes)`（`tp.rs:715-736` 一带）。

### 5-C′（读到另一个对象）

1. 用 4.2 的 `epoch_ptr` 打印定位**所有** `epoch_dev()` 消费点；把 spec/verify 路径**收敛到唯一 `self.comm`**。
2. `dsv41-run.rs:360-361` 的 `c_small` / `c_big` 标「不得参与任何 v5 轮次」（serve 路径今天只有一个 `Collective`，`serve.rs:402`——但 run 路径有两个，**同源风险必须堵**）。

### 5-D（读序/观测错位）

把 ledger 的读从**阻塞 `cudaMemcpy`**（`devrt.rs:1302`，与 side 流无序）改成**在 `self.stream` 上有序的 D2H**：先 `record_event(self.stream)` + `stream_wait_event(主/legacy)`，或直接 `dev.sync()` 后再 `download_u32`。成本 = 一步一次 sync（gate 后才有，OFF 路径不变）。同时给每条 probe 加**自增序号**，让 `pre/note` 配对不再靠 `pos` 猜。

### 5-流（若 witness 真的显示 pad 与 AR 无序——今天源码级不可能，但留作已建好的分支）

先确认是**哪条路径把 AR 放到了 side 流**（`DSV41_DUAL_CHAIN` 分叉点，`chain_dev.rs:10599/12800/14796/15830`），然后二选一：
- **首选**：pad 与 AR **同 stream 同图**——把 pad 变成 verify graph 的一个 node（它已经在 `step_rows` 之前，只差录制位置）；
- **次选**：`record_event(AR 的 stream)` + `stream_wait_event(pad 的 stream)`。
**不要**用 host `sync()` 当同步（会把 SWALLOW 的收益吃掉）。

---

## 6. 验证测试

| # | 测试 | 断言 | 位置 |
|---|---|---|---|
| T1 | **单 rank 单元**：无 peer 下连续 pad N 次 | epoch 严格单调、终值 = Σpad；`e_read` 始终等于前一次 `e_wrote` | `crates/ferrite-dsv41/tests/`（参 `ar_micro.rs`）|
| T2 | **负测试（witness 有效性）**：故意在 side 流上发一次 pad | `[v5-witness]` 必须记到 `e_read` 落后于主流的值 | 同 T1，只加一个 gate 分支 |
| T3 | **parity/acceptance**：`SWALLOW_STEP=1` 长跑 | epoch 跨 step **单调**；每 step 跨 rank **逐字相等**；`[v5-ledger-RESET]` = 0 **且 P1 那一跑必须 ≠ 0**（否则你没抓到要修的东西）；`[v5-ledger-CANARY]` = 0；`[ar5-hang]` = 0 | `scripts/batched_400_v2.sh` |
| T4 | **EOS 独立证据**（与前序 §6.3 同）| `dspark_parity` 量「swallowed 臂与 plain engine 同 position 逐位相同」；`DSV41_INV_CHECK=1` 八条不变量在**第一个坏 step** 报 `[inv-fail]`；1000 tok 零拉丁红线、EOS 不再提前 | 同上 |
| T5 | 编译门 | `cargo check --workspace` EXIT=0（不要手改 `batched_400_v2.sh` 的 gate 串；`.so` 与 binary 同源）| CI |

**T3 的 gate 纪律**：`V5_LEDGER / EPOCH_WITNESS / DYNAMIC_PAD / SWALLOW_STEP` 全部 `OnceLock`；**不新增 gate**（新增 gate = 新增一处「以为在跑其实没跑」，第 9 种的病）。

---

## 7. 改动清单（供尚书省分派）

| # | 文件 | 改动 | 优先级 | 风险 |
|---|---|---|---|---|
| 1 | `scripts/epoch54_digest.sh`（新，或直接 awk）| **P0**：重提原始日志，数 `RESET/CANARY` 行数，按 (pos, rank) 复原序列 | **P0（0 代码）** | 无 |
| 2 | `kernels/cuda/ferrite_kernels.cu` + `dsv41_kernels.cu` | **P1**：5 个写点各写一条 witness；pad/pubred 保持 `*epoch = e+k` 语义不变 | P1 | 低（gate 后 4 行/kernel）|
| 3 | `crates/ferrite-models/src/dsv41/device.rs` | **P1**：3 处入口加 `[v5-stream]` 指针打印；`v5_epoch_pad` 加 `!capturing` 断言；新增 `epoch_witness_dump()` 与 witness buffer 传递 | P1 | 低 |
| 4 | `crates/ferrite-models/src/dsv41/chain_dev.rs` | **P1**：ledger 行尾加 `witness=`；probe 加自增序号；`v5_epoch_pad_rounds` 加 `epoch_ptr` 同一性断言 | P1 | 极低 |
| 5 | `crates/ferrite-models/src/dsv41/tp.rs` | **P2-A2′**：`reduced` 与 `ctr_at` 之间加 8 B guard；canary 扩 4 字；补 `assert!(len <= self.bytes)` | P2（**仅 A2′**）| 中（布局）|
| 6 | `crates/ferrite-models/src/dsv41/chain_dev.rs` | **P2-D**：ledger 读改成 stream-ordered D2H（gate 后）| P2（**仅 D**）| 低 |
| 7 | `docs/agent/dspark-correctness-chain.md` | 追加本文 + §4.4 判定表 | P1 | 低 |

**不做**：
- Device 层 `debug_assert_eq!(stream)`（恒真，见 M1）——只保留作不变量文档 + 加能失败的 `!capturing` 与指针同一性；
- 在 P1 之前改任何 kernel 语义（尤其「单调化/atomicMax」——见下）；
- 方案 C（另造同步）。

**与前序文档的一处分歧（须上报尚书省）**：前序 P1 建议把 `*epoch = e+k` 全改 `atomicMax`。**A2′（清零）下 `atomicMax` 会掩盖真相**（清零后 max 只会把旧高值粘住，54 就再也观察不到，症状伪装成「修好了」）。**建议顺序改为：先上本文 P1 的 witness（只增不改语义），拿到事实后再决定是否需要 atomicMax。**

---

## 8. 一句话总结

任务书的「流断言」是**对的方向、错的靶子**：Device 层三个入口都硬编码 `self.stream`（`device.rs:3250/4568/4593/3208`），照原样写出来的断言**恒真**；而且「过期 e」在**算术上**不可能把 999 打成 54（写者恒 ≥ 读值 + 1）。
真正能一次判死的，是把靶子换成**「写者和读者是不是同一个字、在不在同一个有序域」**，并让**设备侧自己作证**（kernel 记 `e_read / e_wrote`）——因为 `cudaMemcpy` 的 D2H 与 NonBlocking 的 side 流**本来就无序**（`devrt.rs:1302` vs `:161`），这正是「流」在树里唯一真实的漏洞面。
且在这一切之前，**`RESET=0` 在给定数据下源码级不可能**（`chain_dev.rs:9661-9670`）——先花零成本把它定性（多实例拼行？摘要丢列？），否则后面都在修一个误读。

---

*工部 · 只读分析 + 本文件（唯一产出），未执行 GPU 命令、未改动任何源码。*
*所有行号以工作树 HEAD 为准；无法从源码定论的推断均显式标注 `[未验证]` 并给出证伪条件。*
