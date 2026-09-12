# SWALLOW ar5-hang 第 9 次修复设计 —— 基于重建的 v5 轮次账本

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 代码基线：工作树 HEAD `996db36`（`main`）。所有行号以该树为准。
> 任务前提若有源码级反例，显式标注 **❌前提** 并给出依据；无法从源码定论的标 `[未验证]` + 证伪条件。

---

## 0. 前提更正（三条——先读，否则会重复 8 次的错）

### 0.1 ❌ `docs/agent/swallow-systematic-rounds*.md` 不存在
全盘搜索结论：
- `docs/agent/` 下无任何 `*round*` / `*systematic*` 文件（只有 `plan-b-swallow-readiness.md`、`swallow-unlocked-next-plan.md`、`swallow-nograph-optimization-plan.md`、`swallow-result-action-plan.md`、`swallow-step-400-necessity.md`、`dspark-swallow-step-diff.md`）；
- `git log --all` 无该 topic 的 commit；
- 两个 peer worktree（`peer-dsv41-*` / `peer-scheduler-*`）的 `docs/agent/` 也无此文件。

⇒ **"轮次账本"没有被落盘过。** 本设计的第一步就是把账本从源码重建（§1），否则第 9 次仍然是"基于传闻的修补"。

### 0.2 ❌ "argmax_rows rows=6 vs rows=5 是轮次差 1"
**源码否定。** `argmax_xchg_v5_rows_kernel` 的契约就是 **一整块只付 1 个 v5 轮**（`dsv41_kernels.cu:8494-8504` 自述；实现 `:8516-8572` 对 `rows` 做一次 stamp/一次 `*epoch=e+1`）。`rows` 只影响两件事：
- staging 槽的占用字节 `rows*8`（decline 判据 `rows*8 > stride_bytes`，`:8659`）；
- 本地 per-row `argmax_kernel` 的循环次数（`:8663-8666`，**不是 collective**）。

⇒ **"修法 C（统一 argmax_rows 行数）在轮次上是 no-op"**，8 次失败的清单里没有它是对的，但它也不是第 9 次该做的事。

### 0.3 "lazy 上限 ~145 tok/s" —— 与文档一致
`lazy` 的数学下限 = `k_emit × c_row(EAGER 6.15ms) + draft + commit = 41.4ms → 145 tok/s`（`dspark-correctness-chain.md` §"400 的数学必然性"）。**400 只能走 batched（SWALLOW）**——这一点任务前提正确，本设计的靶心也在这里。

---

## 1. 重建的 v5 轮次账本（源码级，可逐条复核）

### 1.1 常量
| 量 | 值 | 出处 |
|---|---|---|
| `n_layers` | 40 | `config.rs:171` |
| `VERIFY_ROWS` | 6 | `chain_dev.rs:84` |
| `window_size` | 128 | `config.rs:213/:528` |
| `n_mtp_layers` | 3（draft 的 MTP block 数）| `dspark_dev.rs` draft_body 循环 |

### 1.2 每一个 v5 轮次的发射点
| 轮次源 | 数量 | 出处 |
|---|---|---|
| 每层 attention 的 wo all-reduce | 1/层 | `chain_dev.rs:11058-11067`（m-row）/ `:14884-14901`（单行） |
| 每层 MoE out all-reduce | 1/层 | `chain_dev.rs:12903-12905`（m-row）+ `end_round` |
| head 的跨 rank argmax 交换 | 1/verify block | `chain_dev.rs:6808` → `dsv41_kernels.cu:8669` |
| draft 每 MTP block 的 MoE all-reduce | 1/MTP block | `dspark_dev.rs:2810-2814` |
| engram gather 的 all-reduce（仅 engram 层）| 1/engram 层 | `chain_dev.rs:4973` |

**三条决定性事实**：
1. **argmax 与 AR 共用同一个设备 epoch**：`epoch` 就是 `c.epoch_dev()` = `staging + ctr_at` 那一个字（`device.rs:3156-3160`、`tp.rs:329-331`）。`argmax_sliced_rows` 的 `*epoch` 推进是**无条件**的（`device.rs:3154-3169`）。
2. **verify 的 argmax 对整块只付 1 轮**（§0.2）。
3. **`k_emit` 在 rank 间一致**（它来自跨 rank argmax 交换的输出；`plan-b-swallow-readiness.md` §0-5 已证）。⇒ 同一步内所有 rank 的足迹**本应**相同。

### 1.3 每步轮次表（轮/步）

| 臂 | anchor forward | draft | verify | **合计** |
|---|---|---|---|---|
| **legacy**（round 1 / 任何未 primed 的轮）| `step_dev` = 40×2 AR + head argmax = **81** | 3×1 = **3** | m=5：40×2 + argmax = **81** | **165** |
| **aligned**（`SEED_ALIGN`）| 81 | 3 | m=6：81 | **165** |
| **swallowed**（`SWALLOW_STEP`）| **0**（anchor 就是 verify 的 row 0）| 3 | m=6：81 | **84** |
| **lazy**（`LAZY_VERIFY`）| 0 | 3 | m=1 逐行：每行 81，共 `k_emit` 行 | **3 + 81·k_emit** |

**账本的三条推论**：
- **推论 L1**：`legacy == aligned == 165`；`swallowed == 84`；**臂边界落差 = 81 轮/步**。
  这解释了为什么 `aligned`（不吞）跑出 **0 ar5-hang**（`dspark-correctness-chain.md` §"Aligned 模式测试结果"）：它与 `legacy` 足迹完全相同，**根本没有臂边界**。
- **推论 L2**：`swallowed` 的足迹与 `accept` **无关**（恒 84）。
  ⇒ 历史上"出师表 0 / 计数 937"的 accept 相关性，**不可能**来自 swallowed 自身的足迹——它必须来自某个 *accept 相关的分支* 或 *accept 改变了 pos 的推进速度*（见 §2.4）。
- **推论 L3**：**只有 lazy 的足迹 ∝ `k_emit`**。`batched_400_v2.sh:148` 的 `FORBIDDEN` 明确排除了 `DSV41_LAZY_VERIFY`——说明官方 batched 矩阵里 lazy 不参与，L2 的"accept 无关"成立。

---

## 2. 8 次失败后仍然存在的轮次差异（按"是否 v5 设备侧轮次"分层——8 次就是把这两层混为一谈）

### 2.1 第一层：`host_barrier`（SpinBarrier）不对称 —— **不是 v5 轮次**
| 臂 | 会合次数 | 出处 |
|---|---|---|
| verify Direct | 1（仅当 `verify_graph_want()`）| `chain_dev.rs:5998-6002` |
| verify Dry | 1 | `:5895-5899` |
| verify Replay | 1 | `:5910-5914` |
| verify **Capture** | **2**（录制被两次 barrier 夹住）| `:5919-5942` |
| draft Direct/Dry | 0 | `dspark_dev.rs:1206-1213` |
| draft Replay | 1 | `:1219-1221` |
| draft **Capture** | **2** | `:1636-1644` |

- `SpinBarrier` 是**到达数**的世代计数器（`tp.rs:32-38`），一个臂多等一次 = 之后每个 barrier 错一代。
- **但这一层只影响 host 侧会合序**；而 8 次观测到的 hang 是**设备侧 v5**（`[ar5-hang] argmax_rows`，`dsv41_kernels.cu:8555-8557`）。
- **⇒ Plan A（对称化）/ Plan B（臂投票）/ Plan C（warmup）都在打这一层**。它们失败不奇怪：即使 host 会合完全对齐，设备 epoch 的 rift 依然存在（`spec_primed` 选出的臂改变了 84 vs 165 的设备足迹，票投对了只是"顺带"把足迹也投对了，而票投错的任何一次都留疤）。

### 2.2 第二层：v5 设备侧轮次的 ±N 差异源（枚举，才是会 hang 的）
逐条把"某 rank 多/少一轮"的可能来源列全：

| 源 | 是否改变 v5 轮数 | 判定 |
|---|---|---|
| (a) verify head 的 sliced/unsliced（`verify_head_geom` `:6485`）| **sliced=1 轮，unsliced=0 轮 → ±1/verify block** | **★ 唯一能产生 ±1 的分支** |
| (b) AR 折叠 decline（`ar_hc_post_fold[_rows]` `:5055/:5133`）| 折/不折都是 1 轮 | 否 |
| (c) draft graph 四臂（`draft_graph_arm` `dspark_dev.rs:1569`）| 四臂设备轮数相同（capture 只是把执行**延后**到 `graph_launch` `:1660`）| 否 |
| (d) **legacy↔swallowed 臂边界** | **81 轮/步** | 是（但只在边界） |
| (e) verify shape pool / DRY latch | 四臂设备轮数相同 | 否 |
| (f) `end_round()` | v5 下**直接 return**（`tp.rs:~856`）| 否 |
| (g) accept / `k_emit` | `swallowed` 下不影响轮数（L2）；`lazy` 下 ∝`k_emit`（L3）| 否（batched）|

**⇒ 在 batched 路径上，除臂边界(d) 外，唯一能"某 rank 少一轮"的是 (a)。**
`(a)` 的输入在源码上**全部是进程级**（env / head dtype / world / vocab / `.so` symbol / `slot_fit = VERIFY_ROWS*8 <= c.bytes`）——理论上 rank 对称。**这是一个 `[未验证]` 的关键空白**：如果 `slot_fit` 或 decline 在真实形状下对某个 rank 翻转，就会稳定产生"每步 ±1"。
**证伪条件**：全 rank 同一条代码路径 → (a) 死；否则 (a) 活。

### 2.3 ★ 本轮最重要的方法论发现：watchdog 把 8 次修复都引向了症状线
| kernel | watchdog | 出处 |
|---|---|---|
| AR v5 的 `ar5_wait_round` | **5,000,000** 自旋（≈0.5 s）| `ferrite_kernels.cu:8980` / `:9024` |
| `argmax_xchg_v5_rows_kernel` | **25,000,000** 自旋（≈5 s，**10×**）| `dsv41_kernels.cu:8544` / `:8555` |

**后果**：一个真实的 epoch rift 会先在 **AR** 上撞到 0.5 s 看门狗并打印 `[ar5-hang] rank=.. peer=.. need=.. cur=..`（**无** `rows=`），10 倍之后才轮到 `argmax_rows`。**8 份记录里反复引用的是 `rows=6` 那一行**——这正是"第二个（更响的）报点"，而真正的第一个报点被当成噪声丢掉了。
⇒ 第 9 次必须先建立**账本观测**（§3.1 D1），否则仍是在症状上修补。

### 2.4 `gap=25` 的算术（不装作已知）
最新指纹：`argmax_rows rank=3 peer=7 need=79 cur=54 rows=6`。由 `ar5_wait_round` 的语义（`ferrite_kernels.cu:8964-8985`）：`cur = ready_local[peer] = peer 的 epoch+1`，等待 `cur >= 我的 e+1`。
⇒ **peer 7 的 epoch 比 rank 3 落后 24~25 轮**，即 **peer 7 少发了 ~25 个 v5 轮**。

候选分解（每条都给出后续可判）：
| 分解 | 含义 | 判据 |
|---|---|---|
| 25 × 1/步 | 某 rank 每步少 1 轮（→ §2.2(a)，head geom 的 sliced/unsliced 翻转）| 账本里每步增量差 1 |
| ~8 × 3/步 | draft 的 3 个 MTP block 少发（→ 某个 draft 分支）| 增量差 3 且正比于 MTP |
| 25 × 1 单次 | 一次性事件（某步的 arm 边界）| 账本在某一步突变 81 |
> 我**无法**从源码把 `need=79` 反推成 (k, 每步轮数) 的整数解（试过 165/84/81/80/41/44 等口径均不整）：最可能的解释是**该请求不是从进程启动后的第一轮开始计数**（epoch 是**进程级**单调、运行时从不重置——`tp.rs:236`），因此 79/54 的绝对值不可用于反推，**只有 `cur` 与 `need` 的差（25）是信息**。这本身就是"没有账本"的代价——D1 上线后这个歧义一次运行即消。

---

## 3. 第 9 次修复设计

**主旨（一句话）**：前 8 次都在**强制 rank 对"选哪个臂"达成一致**；第 9 次把**足迹本身**做成不变式，并把账本做成可观测——**让臂选择对 v5 轮数不可见**，于是"分歧"这个前提被消掉。

### 3.1 修法 D1（观测，必做，零风险，0 行为改变）—— **v5 轮次账本的运行时读出**

**为什么必须先做**：8 次里没有一次能读出"每个 rank 每步的 epoch 增量"，全靠 watchdog 反推；而 watchdog 的 10× 差（§2.3）系统性地把结论引向 `argmax_rows`。

**代码位置**：`crates/ferrite-models/src/dsv41/chain_dev.rs`
- 新增 gate `fn v5_ledger() -> bool`（照抄 `inv_check()` `:1808` 一带的 env 读法，`DSV41_V5_LEDGER`，缓存 static）。
- 新增 `fn v5_ledger_note(&self, pos: usize, arm: &str, k_emit: usize)`：
  ```rust
  if !v5_ledger() { return; }
  let Some(c) = self.comm.as_ref() else { return };
  // epoch 是设备侧权威计数（tp.rs:329-331）
  let now = self.dev.download_u32(c.epoch_dev() as *const c_void)? as usize;
  eprintln!("[v5-ledger] pos={pos} rank={} epoch={now} arm={arm} k_emit={k_emit}", self.rank());
  ```
  成本：**每步 1 次 4B D2H**（与既有 `inv_ids`/`inv_compress_len` 同量级，`:9354/:9420`），只在 gate 打开时发生；不开 gate 时 1 个分支。
- 调用点：`dspark_spec_step`（`:7921`）的**三个臂的公共出口**——即 `:7976`（swallowed）、`:7996`（lazy 的 batched）、`:7992`（lazy）以及 `:8199`（legacy 的 `Ok(...)`）之前各插一行，`arm` 传 `"swallowed" / "lazy" / "legacy"`。

**它直接判定 §2.2 的 (a) 与 §2.4 的分解**：
- 若每步各 rank 增量 **相同** ⇒ 排除 (a)/(d)，rift 在**别处**（转 §3.4 的分支）；
- 若某 rank 增量**恒少 1** ⇒ 命中 (a)，修法是让 head geom 的决策带上一个**每步的 AR 轮数断言**（见 D4）；
- 若某一步**突变 81** ⇒ 命中 (d) 臂边界，D2 治它。

### 3.2 修法 D2（主修）—— **epoch pad：让 swallowed 与 legacy 的足迹相等**

**目标**：无论一个 rank 落在哪个臂，**每步都推进同样的 165 个 v5 轮**。臂边界（含 `spec_primed` 分歧、vote 失效、任何 per-rank 输入翻转）**不再能产生 rift**。

**内核**（`kernels/cuda/ferrite_kernels.cu`，紧随 `p2p_ar_pubred_v5_kernel` 之后新增）：
```cuda
// 一个"空转轮"内核：把 epoch 一次推进 pad 个轮次，并让所有 peer 的
// ready 行看到同一个最终 stamp。没有 payload、没有 staging 写入。
__global__ void dsv41_v5_epoch_pad_kernel(
    unsigned* const* __restrict__ ready_tbl, unsigned* __restrict__ epoch,
    int world, int my_rank, unsigned pad) {
    if (blockIdx.x != 0) return;
    const unsigned e = *epoch;
    if (threadIdx.x < (unsigned)world)
        atomicExch_system((unsigned int*)&ready_tbl[threadIdx.x][my_rank], e + pad);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *(volatile unsigned*)(epoch + 1) = e + pad;  // A4 广播字，保持单调
        *epoch = e + pad;                            // 只在这里推进
    }
}
```
C 入口 `dsv41_v5_epoch_pad(...)`，`.so` 无该符号时返回 `false`（照抄现有 decline 风格）。

**正确性论证（这是"能不能这么干"的关键）**：
1. **v5 的等待判据是绝对 stamp 且单调**：`(int)(cur - (e+1)) >= 0`（`ferrite_kernels.cu:8976/:9020`）。一次跳到 `e+pad` 满足**所有** `e_p <= e+pad-1` 的 peer 的等待，不需要逐个轮次重放。
2. **staging 无陈旧读者**：每个真实 AR 都先把自己这一列写进**所有** peer 的 staging（`e&1` parity），reduce 只读本轮写过的列（`:8688-8698`）。pad 不写 staging，且被 pad 的轮次**没有任何读者**。
3. **parity 安全**：pad 值取 **81（奇数）** ⇒ parity 翻转一次，与真实 AR 的交替一致；下一次真实 AR 在它自己的 parity 槽写入完整一列。
4. **对称性**：pad 在**所有** rank 的同一位置执行同样次数 ⇒ 足迹仍然对称。
5. **A4 单点轮询的广播字 `epoch+1` 必须一起跳**（上面代码已含），否则 `ar5_wait_round` 的 `single_poll` 臂会等一个永远不来的值（`:8991/:8998`）。

**接入点**（`chain_dev.rs`）：
- `dspark_spec_swallowed`（`:8491`）：在 `dspark_snapshot(pos, m)`（`:8507`）之后、`import_tap`（`:8513`）之前，插入
  ```rust
  const SWALLOW_MISSING_ROUNDS: u32 = (40 * 2 + 1) as u32; // 81 = step_dev 的 40×2 AR + head argmax
  self.v5_epoch_pad(SWALLOW_MISSING_ROUNDS)?;              // 缺什么补什么
  ```
- `dspark_spec_lazy`（`:8939`）：**不要**在此处 pad——lazy 的 81·k_emit 是**行数相关**的足迹。lazy 与 batched 混跑必须由 `DSV41_LAZY_ROUTE_LOCK`（默认 ON，`:2625`）继续封死；官方 batched 矩阵已 FORBIDDEN `LAZY_VERIFY`（`scripts/batched_400_v2.sh:148`）。把这条写进 gate 的注释。

**代价**：**1 次 launch / 步（~2 µs）**，相对 SWALLOW 的 −4.5 ms 可忽略；且它**不抵消性能收益**——pad 只跳 epoch，不搬运 AR payload（真实 AR 的数据量一分不减）。

**备选 D2'（零开销，若 tap 可来自 prefill）**：让 spec 路径**从第 1 轮就 swallowed**——prefill 末尾把最后一个 hidden 存成 tap，`spec_primed` 与 legacy 臂整段消失，足迹恒定 84 ⇒ 边界不存在。这是更干净的架构解，但前提是 prefill 的 tap 可得（`dspark-correctness-chain.md` 说首轮 tap 必须由 `step_dev` 供，需先验证 prefill 侧是否已有该 buffer）。`[未验证]`

### 3.3 修法 D3（配套，治 lazy 的 accept 相关足迹）
若将来要让 `lazy` 与 `batched` 混跑（当前禁止），lazy 的足迹 `3 + 81·k_emit` 必须 pad 到与 batched 相同的常量（对每行补 80 轮，或按行 pad）。**在 D2 未覆盖 lazy 之前，`LAZY_VERIFY` 与 `SWALLOW_STEP` 不能同时开**——建议把这一条从脚本注释提升为**运行期硬 gate**（`chain_dev.rs` 的 gate 函数里直接拒绝）。

### 3.4 兜底分支：若 D1 显示"各 rank 增量相同但仍 hang"
那 rift 不在轮次数，而在**轮次顺序**，落到两个候选：
1. **`verify_head_geom` 的 sliced/unsliced 在真实形状下 rank 翻转**（§2.2(a) 的残余）→ 修法：把 `argmax_sliced_rows` 的 decline（`:8659`）从"静默返回 1 由 caller 报 Config error"改成**带 rank/need/cur 的打印 + 与 `slot_fit` 一起在 capture 前断言**。
2. **`capture` 的第 2 个 host barrier 与设备 AR 的交错**（§2.1 唯一奇点）→ 修法：A1（强制 `DRY=0`，整类消除 Capture 臂，`plan-b-swallow-readiness.md` §6 的 A1）——它把一个**结构性奇点**整类消除，而不是追"为什么 rank 会分歧"。

---

## 4. 验证方法（prompt + 指标 + 矩阵）

### 4.1 Prompt（两个，缺一不可）
| 用途 | prompt | 为什么 |
|---|---|---|
| **主探针** | **计数 1–200**（高 accept，历史 937 hang）| 强制长生成、跨过模型退化边界（`dspark-correctness-chain.md` §"教训 2"：出师表自然停止会产生**假阴性**）；数字顺序可判真伪 |
| **红线** | **出师表 1000 tok** | 零**额外**拉丁（判据是"不超过 EAGER 对照"，见 §"最终判定"）|

### 4.2 指标（按"能不能判死"排序）
1. **`[v5-ledger]` 的每 rank epoch 增量逐 step 相等** —— **这是 D2 的直接验收**（账本不变式本身）。
   - 红线：任一步 rank 间增量差 ≠ 0 ⇒ 修复失败，且日志**直接点名**是哪个 rank、哪一步、差几轮。
2. **`[ar5-hang]` 行数 == 0，且按 `argmax_rows` / `pubred`（无 `rows=`）分开计数** —— 分开计数是 §2.3 的直接应用：若出现的全是 `argmax_rows` 而无 `pubred`，说明真 rift 在别处被更长的 watchdog 掩盖。
3. **`[verify_graph] captured` / `[draft_graph] captured` / `arm vote DISAGREED`** —— 判"图是否真生效"（历史上"一直一致"与"一直分歧所以图从未生效"外部不可区分，`chain_dev.rs:6315-6339`）。
4. 吞吐 + `k_acc` 序列 + `first_bad` —— 与 base 逐项对照。

### 4.3 最小充分矩阵
| # | 配置 | 目的 | 通过线 |
|---|---|---|---|
| **T0** | **D1 only**（`DSV41_V5_LEDGER=1`，无 D2），计数 | **零修改先定位**：账本给出 rift 源 | 判出 §2.2(a) / (d) / §3.4 之一 |
| **T1** | D1 + D2，计数 | D2 的判决 | 0 hang + 增量逐 step 相等 |
| **T2** | D1 + D2，出师表 1000 tok | 红线 | 零额外拉丁 + 0 hang + 内容正确 |
| **T3** | T1 胜者，长跑（跨 `pos>128` ⇒ draft 图激活点、跨 `pos>31` ⇒ 压缩激活点）| **防假解锁** | 0 hang，且 `[draft_graph] captured` 出现在长跑中 |
> T3 的依据：`swallow-unlocked-next-plan.md` §7 —— **一次 0 hang 不算修好**；且 `draft_graph_arm` 要求 `pos >= win = 128`（`dspark_dev.rs:1607-1613`），短输出根本到不了那一层。

### 4.4 命令与纪律
- `bash scripts/batched_400_v2.sh`（已内含：同源双产物校验 `.build_id`、单 serve 单 prompt、`EXIT=0` 检查、`FORBIDDEN` 三项防串路径）。**不要**手改它的 gate 串。
- 每次测试前确认 `.so` 时间戳与 binary 一致（历史 ATTN_PROJ_ALIGN 第一次测试被 stale binary 判成"无效"）。
- D1 的 D2H 每步 1 次 → 会略微降低吞吐；**吞吐数字只用 T1/T3（`V5_LEDGER=0`）的那一跑**，D1 那跑只读账本。

---

## 5. 与前 8 次的本质区别（四条）

| # | 前 8 次 | 第 9 次 |
|---|---|---|
| 1 | 治**症状层**：强制 rank 对"选哪个臂"一致（守卫 / 回退 / barrier 对称化 / 臂投票）| 治**守恒量层**：把臂选择对 v5 轮数的影响归零（D2）。**不需要 rank 一致，也不会产生 rift** |
| 2 | 靠 **watchdog 反推**根因；而 argmax 的 watchdog 是 AR 的 **10×**（`dsv41_kernels.cu:8544` vs `ferrite_kernels.cu:8980`），系统性地把结论引向 `argmax_rows` | 先建**账本观测**（D1），把"猜"变成"读"：每 rank 每步 epoch 增量直接落日志 |
| 3 | 修法与**层**错配：Plan A/B/C 打的是 `host_barrier`（SpinBarrier 世代），观测到的却是**设备侧 v5** epoch | 显式分层（§2.1 vs §2.2），只用设备侧不变量收口 |
| 4 | 沿用两个错误前提：a) `rows` 与轮次挂钩（修法 C）；b) 臂边界只是"对齐问题" | 用源码否定 a)（§0.2）；用账本把 b) 转成**可消除的常数项**（81 轮/步 → pad） |

**一句话**：8 次是"让所有 rank 走同一条路"；第 9 次是"让每条路都花同样的钱"——后者对 per-rank 输入翻转免疫，前者不免疫。

---

## 6. 要落地的改动清单（供尚书省分派）

| # | 文件 | 改动 | 风险 |
|---|---|---|---|
| 1 | `crates/ferrite-models/src/dsv41/chain_dev.rs` | 新增 `v5_ledger()` gate + `v5_ledger_note()`；在 `dspark_spec_step` 四个出口各插 1 行（`:7976/:7992/:7996/:8199`）| 极低（gate 关时 1 分支）|
| 2 | `kernels/cuda/ferrite_kernels.cu` | 新增 `dsv41_v5_epoch_pad_kernel` + C 入口 | 低（新符号，不影响既有路径）|
| 3 | `crates/ferrite-models/src/dsv41/chain_dev.rs` | `dspark_spec_swallowed` 里插入 `v5_epoch_pad(81)`（`:8507` 之后）| 中（改了轮次序列，必须 T1/T3 验证）|
| 4 | `crates/ferrite-models/src/dsv41/kernels.rs` + `device.rs` | 绑定新符号 + `fn v5_epoch_pad()` 薄封装 | 低 |
| 5 | `scripts/batched_400_v2.sh` | 加 `DSV41_V5_LEDGER` 的开关说明；把"`LAZY_VERIFY` 与 `SWALLOW_STEP` 不可同时开"提升为硬 gate | 低 |
| 6 | `docs/agent/dspark-correctness-chain.md` | 追加 SWALLOW 章：本账本 + T0 结果 | 低 |

**不做**：修法 C（统一 `argmax_rows` 行数）—— §0.2 已证轮次 no-op。

---

*工部 · 只读分析 + 本文件（唯一产出），未执行 GPU 命令、未改动任何源码。*
*所有行号以工作树 HEAD `996db36` 为准；无法从源码定论的推断均显式标注 `[未验证]` 并给出证伪条件。*
