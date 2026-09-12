# epoch 1328→54 的替代修复方案设计 —— 先证「是不是同一个字」，再选修法

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 代码基线：工作树 HEAD `5f86d6b`（`main`）。行号以该树为准。
> 每个结论都给**源码位置**；无法从源码定论的标 `[未验证]` + 证伪条件。

---

## 0. 摘要：一条被任务书采纳但**源码级为假**的前提

任务书把方案 A 建立在「SWALLOW 的 rollback 恢复了 staging（于是 epoch 掉了）」之上。**这条前提在源码里不成立**（§1），所以：

* **方案 A 按原样不可实施**（没有「被 rollback 覆盖的 epoch」可持久化）。**需上报尚书省**：这是任务书与实际代码的冲突项，不是实现选择。
* 但 A 的**意图**（让 epoch 不再变小）有一个廉价且正确的等价形式：**A′ = 把 epoch 的写改成单调写**（`atomicMax`）。它不解决跨 rank 相等，但能把「同一个字被写小」这条世界从**推测**变成**可判定**（§4.1）。

**推荐主轴（按次序，前一条是后一条的前提）：**

| 序 | 动作 | 成本 | 作用 |
|---|---|---|---|
| **P0** | **重提原始日志**（`rank=` / `canary=` 两列本来就在，见 §1.4） | 0（不跑、不改） | 一次定性三个世界 |
| **P1** | **A′ 单调化**（4 个 kernel 的 `*epoch = ` → `atomicMax`）+ **写标记字**（`ctr_at+12`） | 低 | 把「递减」从可能变不可能；写出「谁最后碰了这个字」 |
| **P2** | **B⁺ 高水位重锚（推荐修复）** | 中低（纯 host，复用已landed 的 pad + RankMax，**不新增 barrier 世代**） | 把 54 拉回，而不是接受 54 |
| **P3** | A-2 布局隔离（epoch 搬出 payload 邻域）——**仅当 canary 触发** | 中 | 世界 A3 |
| **P4** | 方案 C（SWALLOW 绕过 epoch）——**仅当 P2 后计算仍坏** | 高 | 兜底 |

**并且必须同时开一条独立支线**：§3.3 证明「全 rank 相等 ⇒ AR 自洽 ⇒ EOS 未必由 epoch 引起」。**在把 EOS 归因给 epoch 之前，先让 EOS 有自己的证据。**

---

## 1. 问题 1：SWALLOW 的 snapshot/rollback 是否恢复 staging？

### 1.1 答案：**不恢复。rollback 的手够不到 staging，更够不到 epoch。**

`dspark_snapshot`（`chain_dev.rs:7147-7238`）保存的清单，逐项：

| 保存的东西 | 目的地 | 出处 |
|---|---|---|
| ring owner 各层的窗口 ring 的 m 个槽 | `s.dspark_snap_ring` | `:7163-7180`（`dspark_ring_save`）|
| compress source 各层 `state_kv` / `state_score` / `latent` / `clen` / `out_rows` | `s.dspark_snap_state` / `_latent` / `_clen` / `_out_rows` | `:7183-7236`（`dspark_comp_save`）|
| host 侧 `(layer, compress_len)` 镜像 | 返回值 `Vec<(usize,usize)>` | `:7205` / `:7235` |

`dspark_rollback_keep`（`:7269-7378`）把**上面这份清单原样放回**；`dspark_commit`（`:9423-9448`）= `rollback_keep` + `compress_replay` + `set_pos_ctr`。

**三个 buffer 从头到尾没出现在任何一边：** `self.comm` 的 `staging`、`ctr_at`（epoch）、`stamps_at`（ready 行）。

### 1.2 还有哪些「看起来像恢复」的东西也够不到 staging

* `chain.reset()`：按 fix-11 §0.2 的记录只写 `premix_const` / `dspark_pre_mean` / `premix_r` / `pos_ctr` / `clen`（`chain_dev.rs:4218-4247` 一带）——**不含 staging**。
* `DevChain::kv_snapshot` / 前缀快照（`:7380-7405` 的注释块）：序列级 KV 前缀，同样只覆盖 ring/compressor/index_k/engram 表。
* 全树唯一的 `zero_at` 打在 staging 上：`tp.rs:354`，**构造期一次**（`dev.alloc(ctr_at + 64)` 紧跟 `zero_at`，`tp.rs:348-354`）。`staging` 终生不重分配。
* 全树**没有任何** host→device 的 staging 回写：`grep staging.ptr` 在 `tp.rs` 里没有 `ul_*` / `upload*` 命中。

### 1.3 对方案 A 的直接后果

> 「在 snapshot/rollback 之外保存 epoch，让 rollback 不碰它」——**rollback 本来就不碰它**。
> 把这条实现出来是一个 **no-op**：它既不改变任何一次观测，也不会改变 54。
> 唯一的好处是「以防万一」，代价是给热路径加一对多余的 save/restore。

⇒ **方案 A 按原样被源码证伪。按角色规则上报尚书省，不自行改成别的实现。**

### 1.4 顺带：判定所需的两个列**已经在 HEAD 里**（P0 是零成本的）

`v5_ledger_probe`（`chain_dev.rs:9614-9672`）**已经**带：

* `canary`：读 `canary_dev()` = `staging + ctr_at + 8`（`:9633-9646`），不是 magic 就报 `[v5-ledger-CANARY]`；
* `[v5-ledger-RESET]`：per-**rank** 的单调断言（`:9661-9668`），`rank=` 在列里；
* `[v5-ledger-stream]`：一次，reader 的 stream id（`:9623-9629`）；
* 行格式 `pos= rank= epoch= canary= arm= …`（`:9677` / `:9692-9696`）。

而 `dspark-correctness-chain.md:5271-5280` 的摘要把 `rank=` / `canary=` 都丢了——**这正是 fix-11 §0.1 点名的病**。**P0 = 重新 `awk` 一遍原始日志，不改一行代码。**

---

## 2. 问题 2：54 的可能写入点 —— 全树枚举

### 2.1 全部写 `*epoch` 的 kernel（`grep '\*epoch\s*=' kernels/cuda/*.cu`）

| kernel | 写入 | batched v5 路径可达 | 出处 | 能否产出「小值」 |
|---|---|---|---|---|
| `p2p_ar_pubred_v5_kernel` | `e + 1u` | ✅ | `ferrite_kernels.cu:9068` | 否（`e` 是入口读）|
| `dsv41_v5_epoch_pad_kernel` | `e + pad` | ✅ | `:9141` | **只有 `e≈0` 时才可能**（`pad` 本身 ≥0）|
| `argmax_xchg_v5_kernel` | `e + 1u` | ✅（draft 的 markov 步）| `dsv41_kernels.cu:8472` | 否 |
| `argmax_xchg_v5_rows_kernel` | `e + 1u` | ✅（verify 的 m 行 head）| `:8538` | 否 |
| 融合 epilogue 的 gemv（读 `*epoch` 定 parity 槽）| `e + 1u` | ✅ | `ferrite_kernels.cu:9395` / `:9523`；`dsv41_kernels.cu:4453-4455` | 否 |
| `p2p_ar_down_v2_kernel` | `e + 1u`（同行还有个 `*ctr = 0u`）| ❌（`ctr` 传 null）| `ferrite_kernels.cu:8622-8623` | 否（顺序救回）|
| `p2p_ar_fused_v3_kernel` | `*ctr = e`（无 epoch 写入）| ❌ | `:8704` | 否 |
| `p2p_ar_publish_v3_kernel` | `e + 1u` | ❌ | `:8723` | 否 |

### 2.2 所有「写 0 / 写小值」的候选，以及它们为什么**够不到** epoch

| 站点 | 写入 | 它的 `ctr` 实参 | 能否别名到 `epoch` |
|---|---|---|---|
| `p2p_ar_down_kernel`（v1）| `*ctr = 0u` | `ferrite_p2p_ar_oneshot` 的 `ctr` 参数 | **理论可以**——v1 **根本没有 epoch 参数**，若有人把 `epoch_dev()` 当 `ctr` 传进来，epoch 直接 = 0。可达面只有 micro-bench/测试（`ferrite-kernel/tests/gpu_smoke_nccl_graph.rs`）|
| `p2p_ar_down_v2_kernel` | `*ctr = 0u; *epoch = e+1u;` | 同上 | 顺序救回（`*epoch` 在后）|
| `dspark_markov_head_sliced_kernel` 等 glue | `*ctr = 0u` | `dsv41_glue.cu:1613` / `:1802` 的 `mk_ctr` | 否，独立 buffer |
| router / moe 的 self-resetting ctr | `*ctr = 0u` | `ferrite_kernels.cu:1424` / `:3092` / `:8522` | 否 |
| host `zero_at(staging.ptr, ctr_at+64)` | 0 | — | **只在构造期一次**（`tp.rs:354`）|

⇒ **可达的写者全部是 `e + delta`（delta ≥ 0）。** 这个字在一个 rank 上**源码级不可能变小**——除非 `e` 被读成 ≈0。这条把 fix-11 §0.2 的结论从「10 个固定点的推理」加固为「含读点的完整枚举」。

### 2.3 于是 54 只有三种形状（**互斥，一次运行可判死**）

| 形状 | 机制 | 判据（P0 的原始日志）|
|---|---|---|
| **A2「同一个字被写小」** | 某 kernel 以**过期的 `e≈0`** 执行了 `*epoch = e + pad`，pad 恰好把 0 推到 54 | 同 rank 出现 `[v5-ledger-RESET]` **且** canary == `0xDEADBEEF` |
| **A3「越界写踩到 ctr_at」** | payload 越过槽尾写进 epoch 的 64 B 尾 | canary != `0xDEADBEEF`（`[v5-ledger-CANARY]` 出现）|
| **C「读到了别的对象」** | 读的不是这个进程这个 Collective 的字（第二 Collective / 别的 staging / 别的 device）| canary == magic 且 **无** RESET（因为读的那个字自己也单调）；`[v5-ledger-stream]` 的 stream 与 AR kernel 的实参不一致 |

**`[未验证]` 的量化线索：**
* `1328 = 16 × 83`（恰好整除）——pre-prime 的 legacy 单 token 步 = **83 轮/步**；
* `1497 − 1328 = 169`（6 个 pos）——swallow 前夜的每步足迹 ≈169，与 fix-9 账本的 165 同量级；
* `1497 − 54 = 1443`——**单步不对称的上限是 81**（`swallow_missing_rounds = 2*n_layers+1`，`chain_dev.rs:1986-1988`），累积 169/步。**1443 在「漂移」类机制里需要 ≈8.5 步**，而两行观测之间只有一步 ⇒ **「漂移」被排除；必须是覆盖/清零或读错对象。** 这与 fix-11 §3 一致，本设计复核成立。

---

## 3. 问题 3：为什么「均匀降级」这个事实本身改变了整个判断

### 3.1 动态 pad 的调用点决定了「均匀」= 每个 rank 自己就降到了 54

`v5_epoch_consensus` 在 `dspark_spec_step` 的**步首**被调用（`chain_dev.rs:8150`），且**在 ledger 读之前**（`:8157`）。consensus 的语义（`:9795-9811`）：读本 rank epoch → 取 world max → **只 pad 落后者** → `me == max` 的 rank 一个 kernel 都不发。

推论链：

1. 若只有**一个 rank** 掉到 54，consensus 会把它拉到 1497 —— 而 ledger 是在 consensus **之后**读的，**它必然显示 1497**。
2. 观测到的是「**所有 rank = 54，delta = 0**」。
3. ⇒ **consensus 的输入 `me` 在每一个 rank 上本来就是 54**。降级发生在**上一个 step 的最后一次 AR 之后、本 step 的步首之前**，且**对每个 rank 独立发生、结果相同**。

**这是「均匀」真正的信息量：它把「rift（跨 rank 分歧）」排除掉了，把问题钉在「每个 rank 都执行到的、确定性的那个点」上。** 而唯一「每个 rank 都执行到、且只在这个 arm 里执行」的点，就是**第一次 swallowed 步体内**。

### 3.2 动态 pad 为什么必然修不了它（与观测一致）

高水位/取 max 的 pad 只能**向上**补。当**全员都低**时 max == me，**差额为 0，pad 无事可做**。这正是第 11 次「0 hang 但 epoch 仍 54」的机制性解释，不是巧合。

### 3.3 ❗最重要的一条：**全 rank 相等 ⇒ AR 自洽 ⇒ EOS 未必是 epoch 的错**

epoch 在这个协议里只有**两个职责**：

1. **等待判据**：`ar5_wait_round` 比 `(int)(cur - (e+1)) >= 0`，即 `peer_epoch >= my_epoch`（`ferrite_kernels.cu:8976` 一带；fix-11 §2 的推导）；
2. **payload parity**：`staging[(e & 1) * world + r]`（`ferrite_kernels.cu:9082` / `:9090`）。

两个职责都**只要求全 rank 同值**，不要求这个值是 1497 还是 54。**全员同步在 54 时，AR 的等待与 parity 都是自洽的，AR 的结果应当是正确的。**

⇒ **「epoch=54 → attention 读到错误上下文 → EOS」这条因果链，第 11 次并没有被证明，只是被假设。** 至少存在一个同源解释：

> 那个把 epoch 打到 54 的**事件**，同时也破坏了别的东西（例如 `dsv41_kernels.cu:4453-4455`：融合 epilogue 的 gemv 用 `*epoch` 算 `ar_base`，**用过期 epoch 会把 partial 写进错的 parity 半区**——静默的 payload 损坏）。epoch 只是**同源症状**，不是**原因**。

**设计含义：不要为了让 epoch「看起来对」而收工。** 必须给 EOS 一条独立证据线（§6.3）。

---

## 4. 方案评估（A / B / C / D）

### 4.1 方案 A → **A′（单调化）**

| | |
|---|---|
| **A 原样** | **不可实施**：rollback 不碰 staging（§1）。实现 = no-op。**上报尚书省**。|
| **A′ 改法** | 4 个可达 kernel 的 `*epoch = e + k` → `atomicMax(epoch, e + k)`（`ferrite_kernels.cu:9068/9141/9395/9523`、`dsv41_kernels.cu:8472/8538`）。**epoch 从此不可能被写小。** |
| **成本** | 每 kernel 一行；`atomicMax` 在正常路径下与直接赋值等价（单写者时）；无布局改动、无新 gate。|
| **风险** | 低。唯一语义变化：**真·递减变成不可能** ⇒ 于是「递减仍出现」就**只能**是读错对象/越界（世界 C/A3），**这就是判据**。|
| **不解决** | 跨 rank 相等性（那是 P2 的事）。|

### 4.2 方案 B → **B⁺（高水位重锚）** ← **推荐修复**

**B 原样（接受 54）已经被第 11 次证伪**：全员 54 时 AR 确实不 hang，但计算坏了 ⇒ **「接受」不是修复**。

**B⁺ 的核心改法（纯 host，不动 kernel、不动 barrier 世代）：**

```rust
// chain_dev.rs::v5_epoch_consensus() —— 把「world 的当前 max」换成「world 的高水位」
// 新增一个 host 侧高水位（`Cell<u32>`），与已有的 `v5_ledger_seen` 同族：
let me    = self.dev.download_u32(c.epoch_dev() as *const c_void)?;
let hwm   = self.v5_epoch_high_water.get().max(me);   // ← 关键：不低于自己见过的最大值
let anchor = c.epoch_max(hwm as i32) as u32;           // RankMax over HWM，不是 over raw epoch
if me < anchor {
    self.v5_epoch_pad_rounds(anchor - me, "DSV41_SWALLOW_DYNAMIC_PAD")?;
}
self.v5_epoch_high_water.set(anchor);
```

* **为什么这是对的**：pad 的合法性论证（fix-9 §3.2 四条）只要求「跳过的轮次无 payload、无读者」+「parity 与下一轮真实 AR 一致」。`anchor - me` 满足前者；后者由「所有 rank 最终同值」保证（fix-11 §4.2 已给出）。**而「同值」现在由 host 高水位提供，与设备上发生了什么无关。**
* **为什么它修得了「全员 54」**：raw max 是 54 时 pad 无事可做；**高水位是 1497**，于是**每个 rank 都被 pad 回 1497**，rift 当场闭合，且下一轮 AR 从 1497 继续。
* **成本**：复用动态 pad 已经付的那一次 D2H + 一次 `RankMax`（`tp.rs:207-259` / `:996-998`）。**不新增 `SpinBarrier` 会合**——这正是 fix-11 §4.2 点名的唯一真风险，B⁺ 用它**天然规避**（继续走 `RankMax` 自己的 generation）。
* **风险（必须写进验收）**：
  1. `anchor` 只在 `me < hwm` 时 pad；`me > hwm` 时高水位单调抬升，**不会误 pad 一个真领先的 rank**；
  2. 高水位是 per-rank 的 host 状态，**进程重启即归零**——重启后的第一次 consensus 用设备值初始化（与今天行为一致）；
  3. **它会掩盖症状**（把 epoch「修好」），所以 `[v5-ledger-RESET]` 必须保留并且**在 P0/P1 之后仍然为 0** 才算通过——否则你在修自己的修。
  4. gate 必须沿用 `DSV41_SWALLOW_DYNAMIC_PAD` 的 `OnceLock` 纪律；**不新增 gate**（新增一个 gate = 新增一处「以为在跑其实没跑」——第 9 次的病）。

### 4.3 方案 C（SWALLOW 绕过 v5 epoch）

| | |
|---|---|
| **成本** | **高。** 要把 epoch 从 SWALLOW 路径摘掉，就得给那条路径**另造一套同步**：40 层 × 2 次 AR + draft 的 MoE AR + head 交换，每一步几十次会合。v5 的**全部存在理由**就是把这些会合消成设备侧的绝对 stamp（fix-11 §4.2 11-B″ 的论证）。失去的是**这个**——不是「性能优势」这种泛泛之词，而是 **SWALLOW 相对 legacy 的净收益本身**。|
| **风险** | **高。** 一个 staging buffer 上跑两套协议 ⇒ 出现 fix-9 §3.2 之外的第三种 wedge；且新协议的 hang 面尚未被测过。|
| **结论** | **不在 P2 之前做。** 只在「B⁺ 后 0 hang 仍在、但计算仍坏」时，作为隔离实验（诊断）而不是修复。|

### 4.4 方案 D → **D′（廉价、确定性）**

D 原样（「找 54 的精确来源」）**没有错，但需要一个能自报家门的手段**。有两个：

**(a) `compute-sanitizer --tool memcheck`**（fix-11 §4.1 A-4 已列）
* 成本：一次 GPU 短跑 + 重工具；慢，但对**越界写这一类**是**枚举性**的（不需要先有 hypothesis）。
* 风险：低。**只在 canary 触发（世界 A3）时才值得付。**

**(b) `D′ = 写标记字（write-tag）`** ← **推荐，且与 A′ 一次做完**
* 布局里 `ctr_at + 8 .. ctr_at + 64` 是**已经预留、无人写**的尾（`tp.rs:344` 分配 `ctr_at + 64`，`tp.rs:361-366` 只在 `+8` 放 canary）。取 `ctr_at + 12` 作 tag 字。
* 每个 epoch 写者（`pubred` / `pad` / 两个 `argmax_xchg` / 融合 epilogue）在**同一个线程**里顺带写 `*(unsigned*)(epoch + 3) = (tag << 24) | (e & 0xFFFFFF);`。
* ledger 的同一行同时读 `epoch`、`epoch+2`（canary）、`epoch+3`（tag）、`ready_local[my_rank]`。
* **它给出的正是唯一能分开世界的那一列**：「这个字最后是被**谁**、从**哪个 `e`** 写下去的」。成本 = 每个写者一条 store + ledger 一行多读 4 B。风险 = 极低（尾部本来预留；canary 已经是同一族的 tripwire）。

---

## 5. 判定树（P0 一次运行走完）

```
P0：用 awk 重提现有原始日志（gate: DSV41_V5_LEDGER=1 / DYNAMIC_PAD=1 / SWALLOW_STEP=1）
   每行取 pos rank epoch canary arm delta
│
├─ 无 [v5-ledger-CANARY]、有 [v5-ledger-RESET]（同 rank epoch 变小）
│    ⇒ 世界 A2：同一个字被写小
│    ⇒ P1（A′ atomicMax + 写标记字）→ 若 tag 指向 pad/pubred 的过期 e，即定案
│
├─ 有 [v5-ledger-CANARY]（canary != 0xDEADBEEF）
│    ⇒ 世界 A3：越界写进 epoch 尾
│    ⇒ P3（A-2 布局隔离：epoch 搬到独立 8 B；顺带补 v5 路径漏掉的
│             `assert!(len <= self.bytes)`，tp.rs:715-736 一带）
│    ⇒ 若还想枚举写者，再上 compute-sanitizer
│
├─ canary == magic、无 RESET，但 [v5-ledger-stream] 与 AR kernel 实参的 stream 不同
│    ⇒ 世界 C：读错对象 ⇒ 收敛 epoch_dev() 消费点 + 标死 c_big（dsv41-run.rs:361）
│
└─ 以上皆无（数据里一切正常）⇒ 好：那 EOS 与 epoch 无关，立刻转 §6.3 的独立证据线
```

---

## 6. 实施清单（供尚书省分派）

### 6.1 改动清单

| # | 文件 | 改动 | 优先级 | 风险 |
|---|---|---|---|---|
| 1 | `scripts/*`（或新 `scripts/epoch54_digest.sh`）| **P0**：`awk` 出 per-rank 逐 step 表（`pos rank epoch canary arm delta`），**禁止人工转述** | **P0（零代码）** | 无 |
| 2 | `kernels/cuda/ferrite_kernels.cu` / `dsv41_kernels.cu` | **P1**：6 个可达写点 `*epoch = e+k` → `atomicMax(epoch, e+k)`；同一线程写 `epoch+3` 的写标记字 | P1 | 低 |
| 3 | `crates/ferrite-models/src/dsv41/chain_dev.rs` | **P1**：`v5_ledger_probe` 多读 `epoch+3`，行尾加 `tag=` | P1 | 极低 |
| 4 | `crates/ferrite-models/src/dsv41/chain_dev.rs` | **P2（推荐修复）**：`v5_epoch_consensus` 换成**高水位重锚**（新增 `v5_epoch_high_water: Cell<u32>`），**不新增 gate / 不新增 barrier 世代** | P2 | 中低 |
| 5 | `crates/ferrite-models/src/dsv41/tp.rs` | **P3**：`ctr_at+8` 划出独立 `epoch_at`；v5 路径补 `assert!(len <= self.bytes)` | P3（仅 canary 触发）| 中（布局）|
| 6 | `crates/ferrite-dsv41/src/bin/dsv41-run.rs` | **P3**：`c_big` 标注「不得参与任何 v5 轮次」 | P3 | 低 |
| 7 | `docs/agent/dspark-correctness-chain.md` | 追加本设计 + §5 判定树 | P1 | 低 |

**不做：** 方案 A 原样（前提为假）；方案 C（在 P2 未证伪前）。

### 6.2 P2 的验收（与第 10/11 次共用一套）

1. `[v5-ledger-RESET]` = **0 行**；每 step 跨 rank 的 `epoch` **逐字相等**。**并且**在 P2 之前的那一跑里 RESET 必须**非** 0——否则你并没有抓到要修的东西。
2. `[ar5-hang]` = 0，**分开计数**（`argmax_rows` 带 `rows=` vs `pubred` 不带）。第 9 次 §2.3 的 10× watchdog 差必须一直被尊重。
3. epoch 不再回落到一个「看起来像低水位」的值：跨 step 单调，且**未见任何 step 的 delta 为 0**（第 11 次的签名是 delta=0 + 值冻结）。
4. `[verify_graph] captured` / `[draft_graph] captured` 在长跑中出现（`pos >= 128`），否则「图从未生效」与「图生效」外部不可区分。
5. 吞吐只用 `V5_LEDGER=0` 那一跑；`k_acc` 序列与 base 逐项对照；**1000 tok 零拉丁红线**。
6. `cargo check --workspace` EXIT=0 + `bash scripts/batched_400_v2.sh`（**不要手改它的 gate 串**；`.so` 与 binary 同源）。

### 6.3 ⚠ 必须并行的独立支线：给 EOS 自己的证据

§3.3 已证「全 rank 相等时 AR 自洽」。**所以 EOS 不许继续挂在 epoch 名下当结论。** 建议同跑一臂：

* `DSV41_INV_CHECK=1` + `DSV41_SIDS_WRITEBACK=1`（`inv_*` 族，`chain_dev.rs:9546-9820`）——八条不变量在**第一个坏 step** 报出 `[inv-fail]`，而不是等到 EOS；
* **parity 自检**：`dspark_parity`（`dspark_spec_swallowed` 的「row 0 的 forward 就是 `step_dev` 会产生的行」契约，`chain_dev.rs:8751-8770` 一带）——它直接量「swallowed 臂与 plain engine 在同 position 是否逐位相同」，**这正是「attention 读到错误上下文」的正面测量**；
* **`ar_e` 那一支**：`dsv41_kernels.cu:4453-4455` 的融合 epilogue 用 `*epoch` 算 `ar_base`。**在 P1 的写标记字落地后**，若 tag 显示「某个 producer 在 epoch 已前进之后又以旧 `e` 取 parity 槽」，那 EOS 就与 epoch 54 是**同源**，B⁺ 会一并修掉它——这是**顺带**的收益，不是 P2 的验收条件。

---

## 7. 一句话总结

任务书的方案 A 建立在一个**源码级为假**的前提上（rollback 够不到 staging），按原样是 no-op——**这条要上报尚书省**。
真正的形状是：**「均匀降到 54」恰恰说明每个 rank 自己降到了 54**（consensus 在 ledger 之前跑，只向上补——全员低时它无能为力），所以这是**每个 rank 都走到的那个点上的确定性写**，不是 rift。
而 **`1343` 级别的一跳在「漂移」里需要 ≈8.5 步、单步上限只有 81 ⇒ 必须是覆盖或读错对象**（§2.3）。

**推荐：P0 免费定性 → P1 单调化 + 写标记字（把「递减」变成不可能，把「谁写的」变成可读）→ P2 高水位重锚（用 host 的高水位取代设备 max，纯 host、复用已landed 的 pad + RankMax、不动 barrier 世代）→ 仅 canary 触发才做 P3，仅 P2 后计算仍坏才碰方案 C。**
**并且：epoch 的两个职责（等待 + parity）只要求全 rank 同值，不要求那个值是多少——所以「epoch 54 ⇒ EOS」目前是假设而非结论，必须另开一条独立证据线。**

---

*工部 · 只读分析 + 本文件（唯一产出），未执行 GPU 命令、未改动任何源码。*
*所有行号以工作树 HEAD `5f86d6b` 为准；无法从源码定论的推断均显式标注 `[未验证]` 并给出证伪条件。*
