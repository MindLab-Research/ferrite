# epoch 54 —— P1 witness 出结果后的修复模板（A/B/C/D 四分支，即可套用）

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 代码基线：工作树 HEAD `21e8a8a`（含 commit `b8833ea` 的 P1 witness）。行号以该树为准。
> 前序：`epoch54-final-fix-path.md`（P0–P3 路线 + §4.4 判定表）、`epoch-54-alternate-fix-design.md`（A′/B⁺/C/D′）。
> 本文**不重复**前序推导，只做一件事：**把 witness 的四类输出，各自钉到「判据 → 根因 → 具体代码改动方向 → 验证」的模板**，witness 数据一到就能照做。

---

## 0. 一分钟上手：witness 行长什么样，怎么机械分类

P1 witness 的落地形态（`b8833ea`）：每个 epoch **写点**在写之前/之后记一条，host 在 `v5_ledger_note` 之后 gate 开启时 D2H 读回，一行：

```
[v5-witness] pos=<n> rank=<r> writer=<id> e_read=<R> e_wrote=<W> clk=<c>
```

`writer=<id>` 映射（本树真实写点，见 §1 表）：

| id | kernel | 写入表达式 | k |
|---|---|---|---|
| 1 | `p2p_ar_pubred_v5_kernel` (`ferrite_kernels.cu:9169`) | `e + 1u` | 1 |
| 2 | `dsv41_v5_epoch_pad_kernel` (`:9255`) | `e + pad` | pad |
| 3 | `p2p_ar_pubred_v5_hcpost_kernel` (`:9528`) | `e + 1u` | 1 |
| 4 | `p2p_ar_pubred_v5_hcpost_rows_kernel` (`:9673`) | `e + 1u` | 1 |
| 5 | `argmax_xchg_v5_kernel` (`dsv41_kernels.cu:8472`) | `e + 1u` | 1 |
| 6 | `argmax_xchg_v5_rows_kernel` (`:8538`) | `e + 1u` | 1 |

**五条自检不变量**（先跑这五条，才能把结果归到 A/B/C/D）：

| # | 不变量 | 违反 ⇒ 直接结论 |
|---|---|---|
| **W1** | 每条 `e_wrote == e_read + k(id)` | 违反 = 结果 **A-2**（算术/记录错） |
| **W2** | 同 rank 同 step 内，写点按 `clk` 排序后 `e_read[i] == e_wrote[i-1]` | 违反 = 结果 **C**（读到别处的字） |
| **W3** | 同 pos 同 writer 跨 rank 的 `e_wrote` 逐字相等 | 违反 = 跨 rank rift |
| **W4** | 在 ledger 掉值的那个 pos（pos=15），存在 `e_wrote ≤ 999` 的写点 | 不存在 = 结果 **B** 或 **D** |
| **W5** | 任意 witness 行的 `e_read` ≥ 本 rank 之前已见过的最大 `e_wrote` | 违反 = 结果 **C**（读了旧字/别字） |

> **先决（P0，零代码）**：`RESET=0` 在给定数据下**源码级不可能**——`v5_ledger_probe`（`chain_dev.rs:9614`）被 `pre`/`note` **共享**，note 的 54 会经同一 `insert(rank, epoch)`（`:9661`）进入 `seen`，pos=16 pre 必报 `prev=999 cur=54`。所以 witness 到手前，先用原始日志数 `[v5-ledger-RESET]` / `[v5-ledger-CANARY]` 行数并按 `(pos, rank)` 复原序列；若 RESET 仍为 0，先怀疑「同 rank 多 `ChainDev` 实例拼行」或「摘要丢列」，**否则在修一个误读**。

---

## 1. 源码事实表（模板的靶子，全部核过）

**写点**（`*epoch =`，`grep -n '\*epoch\s*=' kernels/cuda/*.cu`）：
- `ferrite_kernels.cu:9169` pubred　　　　　`*epoch = e + 1u`　（v5 可达）
- `ferrite_kernels.cu:9255` pad　　　　　　`*epoch = e + pad`　（v5 可达，pad=rounds）
- `ferrite_kernels.cu:9528` hcpost　　　　 `*epoch = e + 1u`　（v5 可达）
- `ferrite_kernels.cu:9673` hcpost_rows　 `*epoch = e + 1u`　（v5 可达）
- `dsv41_kernels.cu:8472`　 argmax_xchg　　 `*epoch = e + 1u`　（v5 可达）
- `dsv41_kernels.cu:8538`　 argmax_xchg_rows `*epoch = e + 1u`（v5 可达）
- `ferrite_kernels.cu:8623` down_v2 / `:8723` publish_v3（**v5 不可达**，`ctr`/`epoch` 传 null）

**读点**：每个写 kernel 入口都 `const unsigned e = *epoch;`（`:9149/:9248/:9512/:9657`、`dsv41_kernels.cu:8462/:8526`）。
**融合 epilogue 只读不写**：`dsv41_kernels.cu:4453` `const unsigned ar_e = (epoch!=nullptr) ? *epoch : 0u;`，仅用它算 `ar_base`（parity 槽），**本树没有任何 `*epoch =` 在这里**。
⇒ ⚠️ **与前序设计的分歧**：`epoch54-final-fix-path.md §4.3` 把「融合 epilogue 的 gemv」列为 writer=5。本树核实：它**是读点，不是写点**。若 witness 真报出一条 writer=5 的写，那是**新增写点**或**标签错**，先核对 witness 的 id 映射，别直接当成已知写点。

**布局**（`tp.rs`）：
- `stamps_at = 2*world*bytes`（`:336`）→ `reduced_at = stamps_at + world*4`（`:337`）→ `ctr_at = reduced_at + world*4`（`:338`）
- `staging = alloc(ctr_at + 64)`（`:348`），构造期 `zero_at(staging.ptr, ctr_at+64)`（`:354`）
- epoch = `ctr_at + 0`；A4 广播字 = `ctr_at + 4`（`ferrite_kernels.cu:9090/9254`）；canary = `ctr_at + 8`（`tp.rs:318/463`）
- `epoch_dev()` = `staging + ctr_at`（`tp.rs:455`）；`canary_dev()` = `staging + ctr_at + 8`（`:463`）
- `epoch_max(v)` = `RankMax` rendezvous（`tp.rs:996`）

**host 观测**：
- `v5_ledger_probe`（`chain_dev.rs:9614`）：先读 canary（`:9634`）后读 epoch（`:9651`），per-rank RESET（`:9661`）
- `v5_ledger_pre`（`:9674`）/ `v5_ledger_note`（`:9683`）
- `v5_epoch_pad_rounds`（`:9738`，传 `c.epoch_dev()` `:9747`）
- `v5_epoch_consensus`（`:9804`）：`download_u32(c.epoch_dev())`（`:9814`）→ `epoch_max`（`:9815`）→ pad `max - me`（`:9817`）
- `download_u32` = **阻塞 `cudaMemcpy` D2H**（`devrt.rs:1302`）；side 流 = `NonBlocking`（`devrt.rs:161`）
- `epoch_dev()` 消费点：`chain_dev.rs:5788`（argmax sliced）、`:7010`（argmax rows）、`:9651`、`:9747`、`:9814`；`dspark_dev.rs:3226`
- 第二 Collective：`dsv41-run.rs:360 c_small` / `:361 c_big`，`chain.comm = Some(c_small)`（`:382`）

---

## 2. 结果 A：某写点的 `e_wrote = 54`

**判据**：某 witness 行 `e_wrote=54`。先看它的 `e_read`，分两支（互斥）。

> **A 的算术前情**：全树写者都形如 `*epoch = (入口读到的 e) + k`，`k ≥ 1`。∴ `e_wrote=54` ⇒ **必有一个写者读到 `≤ 53`**。而同一字一步前是 999 ⇒ 「过期 e」不可能（过期只会写 ≥1000）。所以 A 一定是**读到了另一个/被清过的字**，或**写的表达式根本不是 `e+k`**。

### 2.A-1　`e_read = 53`（k=1 写点读到 53）—— 世界 C′：读错对象

- **根因**：该 kernel 的 `epoch` 形参指向一个含 53 的字，**不是**本 rank 的 `staging + ctr_at`。它是「链条的中段」——真正的下降由更早一个写者完成（它读到 ≤52）。必须先找**链条的头**（clk 最小的那个 `e_wrote ≤ 999` 的写点）。
- **改动方向**：
  1. **指针同一性打印**（新增，不改语义）：在 `device.rs` 每个 v5 入口加一次性 `eprintln!("[v5-stream] {who} epoch_ptr={:p}", epoch)`——`v5_epoch_pad:3250`、`p2p_ar_v5:4568`、`p2p_ar_pubred_v5:4593`、`argmax_sliced_rows:3183`。同时在 `chain_dev.rs` 首次取 `c.epoch_dev()` 处打 `[v5-epoch-ptr] rank=… staging={:#x} ctr_at={} epoch_ptr={:p}`。
  2. **按 writer id 追调用点**（host 侧传 epoch 那一行）：
     - writer 1/3/4（pubred/hcpost/hcpost_rows）← `layer()` / `moe_reduce` 的 AR 调用；
     - writer 5/6（argmax）← `chain_dev.rs:5788` / `:7010`；
     - writer 2（pad）← `chain_dev.rs:9747`。
     逐一确认传的是 `c.epoch_dev()`，**不是** `c_big.epoch_dev()`，也不是某个 peer base。
  3. 若确认传的是第二个 Collective（`c_big`，`dsv41-run.rs:361`）→ **收敛到唯一 `self.comm`**（`chain.comm`），并在 `dsv41-run.rs:361` 给 `c_big` 加注「不得参与任何 v5 轮次」。
- **验证**：单 rank 单元测试 **T1**——无 peer 连续 pad N 次，witness 的 `e_read[i] == e_wrote[i-1]`（W2 链条连续），且所有 `[v5-stream]` 的 `epoch_ptr` 都等于 `epoch_dev()`。

### 2.A-2　`e_read = 999`（读到 999 却写 54）—— 写表达式/记录错

- **判据**：`e_read=999 e_wrote=54`，且 `54 ≠ 999 + k(id)` ⇒ **W1 单点违例**。
- **根因**（二选一，先排 (ii)）：
  - (i) 该 kernel **不是 `e + k` 写**，写了一个别的表达式（例如把 A4 广播字/局部 `need`/别的变量写进 `epoch`）；或
  - (ii) **witness 记录本身把 `e_wrote` 记错**（结构体字段打包 offset 错，把某个别的字读进 `e_wrote`）。
- **改动方向**：
  1. **算术审计（只有一条能从 999 量级一步跳到任意小值）**：`dsv41_v5_epoch_pad_kernel` 的 `*epoch = e + pad`（`:9255`），`pad` 来自 host `v5_epoch_consensus` 的 `max - me`（`chain_dev.rs:9817`）。`epoch_max(me as i32)`（`tp.rs:996`）是**有符号** i32：若某 rank 的 epoch 曾 > `2^31`，`me as i32` 变负，`max` 变负，`max - me` 在 u32 下**回绕成巨值**，`e + pad` 再回绕到任意小值。**检查 epoch 是否曾越过 2^31**。若是，修 `epoch_max` 用无符号比较（或钳制）。
  2. **逐个写点核表达式**：对照 §0 表的 6 个写点，把每个 kernel 的写入表达式与 `k` 并排打印（把 `k` 也记进 witness）。
  3. 若判定 (ii)：修 witness 结构体字段打包（`{writer, e_read, e_wrote, clk}` 的偏移），用 **T2** 验证。
- **验证**：**T2 负测试**——在一个 gate 分支里故意让 pad 写 `e + 7`，witness 必须显示 `e_wrote = e_read + 7` 逐字相等；否则 witness 记录层有 bug，A 的一切结论先作废。

---

## 3. 结果 B：所有写点 `e_wrote > 999`（没有 kernel 写 54）—— host 读错位置

**判据**：每条 witness 满足 W1（`e_wrote = e_read + k`）且 `e_wrote ≥ 1000`（W4 不成立），但 `[v5-ledger]` 同一 pos 读到 54。
**根因**：kernel 从未写 54 ⇒ **ledger 的 D2H 读到 54，或读的不是同一个字/同一时刻**（世界 C′/D：观测错位）。

- **改动方向**：
  1. **地址同一性**：打印 `epoch_dev() as usize`（一次）与 witness 写地址。若不等 ⇒ `epoch_dev()` 算错（`tp.rs:455`），复查 `ctr_at`（`tp.rs:338`）与 `staging` 分配（`:348`）。
  2. **读序修复**（前序 §5-D）：`download_u32`（`devrt.rs:1302`）是**阻塞 `cudaMemcpy`**，与 **NonBlocking side 流**（`devrt.rs:161`；`DSV41_DUAL_CHAIN` 的 `side_stream2`，`chain_dev.rs:10599/12800/14796/15830`）**无序**。若 epoch 在 side 流推进，ledger 会在写落盘前读到中间值。**改法**：ledger 的读改成 `self.stream` 上的有序 D2H——`record_event(self.stream)` + `stream_wait_event`，或 gate 后 `dev.sync()` 再 `download_u32`（成本 = 一步一次 sync，仅 gate 后；OFF 路径不变）。
  3. **时间同一性**：给 ledger/witness 行都加**自增序号**（probe seq）+ `clk`，按 seq 对齐；确认 ledger 的 54 行与 witness 是同一次 step、同一 pos。
  4. **设备侧交叉验证**：用一个小探针 kernel 在 ledger 点读 `*epoch` 写进 witness buffer，与 D2H 值对比——`device-read ≠ D2H-read` 则 D2H 观测错位**定案**。
- **验证**：**T3**——同一次跑里设备侧读与 D2H 读逐字相等；side 流 gate × stream-ordered 读组合下 ledger 不再出现 54。

---

## 4. 结果 C：某写点 `e_read` 很小（如 53）—— kernel 读到别的字

**判据**：某 witness 行 `e_read` 远小于当时 999 量级（W5 违反），`e_wrote = e_read + k`。
**根因**：该 kernel 读到的**不是本 rank 当时的 epoch**——交叉污染/错位。三种候选：

- **改动方向**（按候选逐一排除）：
  - **(a) 指针指向别的对象**：同 §2.A-1：按 writer id 追 `epoch` 参数来源，打印指针同一性。先排除。
  - **(b) 读到 A4 广播字或别的邻字**：广播字在 `ctr_at + 4`（`ferrite_kernels.cu:9090/9254`），`reduced[]` 尾字在 `ctr_at - 4`。若该 kernel 的 `epoch` 实参恰好是 `epoch_dev()+1` 或 `epoch_dev()-world*4`，它会读到一个「看起来像轮次」的小值。**打印指针即可分辨**。
  - **(c) `reduced` 向上溢出覆盖 epoch**（结构性盲区）：`reduced` 最后一个字在 `ctr_at-4`（`tp.rs:337-338`）；`reduced[world] → ctr_at+0`、`reduced[world+1] → ctr_at+4`、`reduced[world+2] → ctr_at+8=canary`。**宽度 1–2 字的溢出恰好踩死 epoch（和 A4 字），canary 完好** ⇒ 这正是 `CANARY=0` 的盲区。改法（P2-A2′，**仅此分支做**）：
    1. `reduced_at` 与 `ctr_at` 之间插 **8 B guard**：`ctr_at = reduced_at + world*4 + 8`（`tp.rs:338`），guard 写 canary 同族 magic，ledger 一并读；
    2. canary 从 `+8` 扩到 4 字（`+8/+16/+32/+48`，都在 64 B 尾内），全读全比；
    3. 顺手补 v5 路径缺失的 `assert!(len <= self.bytes)`（`tp.rs` 一带）。
  - **枚举越界（不依赖 hypothesis）**：短跑 `compute-sanitizer --tool memcheck`。
- **验证**：加入 guard 后短跑，`[v5-ledger-CANARY]` 或 guard 失配**必须触发**（不触发 ⇒ 溢出不在 `reduced` 尾）；**T1** 单 rank 连续写序列 `e_read` 严格等于前一条 `e_wrote`。

---

## 5. 结果 D：witness 正常（所有 `e_wrote = e_read + k`）—— 两个互斥分支，先分辨

**判据**：所有 witness 行满足 W1/W2/W4（pos=15 存在 ≤999 的写），但仍观测到 54。
**根因**：**必须先分辨**下面两个互斥分支，否则会修错东西：

### 5.D1　witness **未覆盖真正的写者**（把 54 写进 epoch 的路径不经过插桩的 `*epoch =`）
候选（witness 都看不到）：
- `reduced[]` 向上溢出直写 `ctr_at`（见 §4-(c)）——**最可能**；
- host `zero_at(staging.ptr, ctr_at+64)`（`tp.rs:354`，**构造期一次**，非 per-step）——若有人重放/重复构造，epoch 归零，余下轮次把它带到 54；
- 任何落在 staging 上的 `upload_from`/`cudaMemset`（`tp.rs` 当前无此类回写，需 `grep` 复核）；
- **A4 广播字被误写**：某 kernel 的 `epoch` 形参比真 epoch **低一个字**（`ctr_at-4`），它的 `*(epoch+1)` = `ctr_at` 就**直接覆盖真 epoch**（`ferrite_kernels.cu:9090/9254`）。

**改动方向**：
  1. **先算覆盖**：统计 pos=15 的 witness 行数 vs 期望写点数（本 step 的 AR 轮数 × 写点数）。**缺行即 D1**，扩 witness。
  2. **对候选逐一插桩**：
     - `reduced[]` 尾后加 guard（§4-(c)）；
     - ledger 同一行把 epoch 周围内存 dump 出来（`ctr_at-16 .. ctr_at+64` 的 u32 全读），**找 54 落在哪个偏移**：
       - 偏移 `+0` ⇒ epoch 被写；偏移 `-4` ⇒ `reduced` 溢出；偏移 `+4` ⇒ A4 广播字。
     - 单独读一行 A4 广播字（`ctr_at+4`）与 epoch 对比。
  3. `compute-sanitizer --tool memcheck` 短跑枚举越界写。

### 5.D2　host 侧 D2H 读错（设备侧写正常）
- 与 §3（结果 B）**同一套修法**：stream-ordered D2H（`devrt.rs:1302` → `self.stream` 有序读）+ 设备侧探针交叉验证。

**验证**：**T4 独立证据线**（与前序 §6.3 同）——`dspark_parity`（swallowed 臂 vs plain engine 同 position **逐位相同**）+ `DSV41_INV_CHECK=1` 八条不变量在**第一个坏 step** 报 `[inv-fail]`。这同时回答「epoch 54 是否真的导致 EOS」——**在没拿到这条独立证据前，「epoch 54 ⇒ EOS」只是假设**。

---

## 6. 通用验证矩阵（四分支共用）

| # | 测试 | 断言 | 位置 |
|---|---|---|---|
| **T1** | 单 rank 单元：无 peer 连续 pad N 次 | witness 链条 `e_read[i]==e_wrote[i-1]`（W2）；终值 = Σpad；`epoch_ptr` 全等 | `crates/ferrite-dsv41/tests/`（参 `ar_micro.rs`）|
| **T2** | witness 有效性负测试：加 gate 分支故意写 `e+7` | witness 必须逐字回读 `e_wrote=e_read+7`（W1） | 同 T1 |
| **T3** | parity/acceptance：`SWALLOW_STEP=1` 长跑 | epoch 跨 step 单调、每 step 跨 rank 逐字相等（W3）；`[v5-ledger-RESET]`=0 **且 P1 那一跑必须 ≠0**；`[v5-ledger-CANARY]`=0；`[ar5-hang]`=0 | `scripts/batched_400_v2.sh` |
| **T4** | EOS 独立证据 | `dspark_parity` 逐位相同；`DSV41_INV_CHECK=1` 首个坏 step 报 `[inv-fail]`；1000 tok 零拉丁红线 | 同上 |
| **T5** | 编译门 | `cargo check --workspace` EXIT=0（**不要手改** `batched_400_v2.sh` 的 gate 串；`.so` 与 binary 同源）| CI |

**gate 纪律**：`V5_LEDGER / V5_WITNESS / DYNAMIC_PAD / SWALLOW_STEP` 全部 `OnceLock`；**不新增 gate**（新增一个 gate = 新增一处「以为在跑其实没跑」，第 9 种的病）。
**不先改 kernel 语义**：尤其**不要**先把 `*epoch = e+k` 全改 `atomicMax`——若世界是 A2′（清零），`atomicMax` 会把旧高值粘住，54 再也观察不到，症状伪装成「修好了」。

---

## 7. 一句话总结

witness 一到手，**先跑 W1–W5 五条不变量机械分类**：
- **W1 单点违反 → A-2**（写表达式/记录错；先查 `pad=max-me` 的回绕与 witness 字段打包）；
- **W4 全 >999 → B**（host D2H 观测错位；修 stream-ordered 读）；
- **W5 违反 → C**（读到别字；先查 epoch 指针来源，再查 `reduced` 向上溢出的结构性盲区）；
- **全绿但仍 54 → D**，且**先分辨「witness 没覆盖到真写者（D1，最可能是 reduced 溢出/广播字被踩）」还是「host 读错（D2）」**——D1 不要误判成 D2。

且无论哪支，**P0 重提原始日志**（RESET/CANARY 行数 + per-(pos,rank) 序列）都在最前面；**EOS 需独立证据线**，不许继续挂在 epoch 名下当结论。

---

*工部 · 只读分析 + 本文件（唯一产出），未执行 GPU 命令、未改动任何源码。*
*所有行号以工作树 HEAD `21e8a8a` 为准；无法从源码定论的推断均显式标注并给出证伪条件。*
