# Plan B + SWALLOW batched 路径的就绪度分析（只读）

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 任务：Plan B（unanimity-or-direct，已实施、从未 GPU 测试）的就绪度 + 测试矩阵 + 无图模式 hang 的根因假设。
> 代码基线：工作树 HEAD `64940c5`（Plan B 提交）。
> 现场核对：`chain_dev.rs`（`verify_arm_local` :5702、`verify_graph_gate` :5809、`note_arm_dissent` :5843、
> `step_rows_sync` :5479、`dspark_spec_swallowed` :8074、`dspark_commit` :8738、`compress_replay` :8784）、
> `tp.rs`（`RankVote` :108、`unanimous_i32` :132、`SpinBarrier::wait` :51）、
> `dspark_dev.rs`（`draft_graph_arm` :1569、`draft_capture` :1632、`note_ctx_rows` :944）、
> `ferrite_kernels.cu`（`ar5_wait_round` :8960+）、`dsv41_kernels.cu`（`argmax_xchg_v5_rows_kernel` :7834）。

---

## 0. 判决（先读七条——三条修正任务前提）

1. **🔴 Plan B 在「无图模式」下是严格惰性的（no-op）。** `verify_graph_gate` 的第一句就是
   `if !verify_graph_want() { return idx; }`（`chain_dev.rs:5807`）：`DSV41_VERIFY_GRAPH=0` 时函数在**投票之前**返回，
   `RankVote` 一次都不走；同时 `verify_arm_local` 的首句 `if !verify_graph_want() || pos_base < 1 { return Direct }`
   （`:5695`）让唯一可能出现的臂就是 Direct。⇒ **Plan B 不是「无图 hang 的备选修复」，而是「有图 hang 的修复」——
   测试矩阵必须打开 `VERIFY_GRAPH=1` 才谈得上测 Plan B。**

2. **🔴 「无图 hang 不是臂分歧」这句话成立，但推论要走对方向。** `VERIFY_GRAPH=0` 时 Direct 臂的 `host_barrier` 被
   `&& verify_graph_want()` 关掉（`:5675`），lazy 的 `skip_barrier=true`（`:8368`）也把入口 barrier 全抑制——
   所以**无图 SWALLOW 路径上不存在任何 `host_barrier`**。因此无图 hang **不可能是 `SpinBarrier` 类 misphase**，
   只能落在 **AR v5 的设备侧 epoch 协议**（内核 `[ar5-hang]` 诊断，`ferrite_kernels.cu:8981/:9025`）。

3. **🔴 Plan B 的收益上限 ≈ −1.5ms（图实测值），不是关键路径。** `swallow-nograph-optimization-plan §2` 已实测：
   图只削已被 CUDA async launch 隐藏的 submit 半。Plan B 的定位是「把图当顺序/正确性工具拿回来」，
   不是「让 SWALLOW 变快」。**就绪度评估应按「解锁 batched 图化」而非「提速」来判。**

4. **🟡 Plan B 的就绪度：代码就绪，GPU 零验证；且它有一个未覆盖的同类风险面（draft 图）。**
   `RankVote` 4/4 单测覆盖的是原语本身；`verify_graph_gate` 的端到端语义、投票开销、以及
   **另一处结构完全相同的臂分歧源（`DSV41_DRAFT_GRAPH`，见 §3 H1）都不在投票覆盖内**。

5. **🟡 「高 accept → 更多 committed rows」** 这一条**在进程内是可证的对称**：`k_emit` 来自
   `spec_accept(drafts, rows)`，`rows` 是跨 rank argmax 交换（`argmax_sliced_rows`）的输出、`drafts` 来自
   `note_ctx_rows` 前复制的 draft 头 ⇒ **8 个 rank 的 `k_emit` 相同**。所以 accept 本身**不制造 rank 间计数差**——
   accept 只能通过「改变了哪个内核/哪些 host 分支被走到」间接影响 epoch。见 §3 的 H3/H4/H5。

6. **🟡 一个被任务假设排除、但必须测的混淆变量**：历史的「无图」run 只保证 `VERIFY_GRAPH=0`，
   而 `batched_400_v2.sh` 的正式矩阵里 `DSV41_DRAFT_GRAPH=1`。若历史 run 也带着 draft 图，
   则「无图 hang」其实是 **draft 图的臂分歧**（Plan B 结构上修不到）。**矩阵第一格就必须用日志判定它。**

7. **绝对纪律**：本轮所有现象描述都给出**代码/日志证据**；无法从源码定论的地方显式标注 `[未验证]`，
   不把「合理猜测」写成「根因」。真值判定依赖 §4 矩阵里的 4 条 stdout 判据。

---

## 1. Plan B 到底修的是什么（把臂会合数钉死）

### 1.1 `VerifyArm` 的会合数表（现场核对）

`step_rows_sync`（`:5479`）的四个臂，每个臂在一次 `verify block` 里调 `host_barrier()` 的次数：

| 臂 | 触发条件（`verify_arm_local`） | `host_barrier` / block | 位置 |
|---|---|---:|---|
| **Direct** | `!want() \|\| pos<1`、SWALLOW warmup（block<3）、无 slot、`verify_graph_failed[idx]`、`!armed` | **1**（仅当 `want()`；否则 **0**） | `:5675`（条件式） |
| **Dry** | 有 slot、未 dry、`armed`（`!eng_host && !stats_dbg && !phase_dbg && compress_branch_steady && supports_dspark_snapshot && supports_memset_async && (comm.is_none() \|\| ar_v5)`） | **1** | `:5572` |
| **Replay** | `verify_graphs[idx].is_some()` | **1** | `:5587` |
| **Capture** | `verify_dry_done[idx] && verify_graphs[idx].is_none()` | **2**（录制被两次 barrier 夹住） | `:5596` + `:5618` |

> 与代码注释里历史测量的「DRY=0 / replay=1 / capture=2 / direct=0」相比，**method A 已经给 Dry 和 Direct 各补了 1 次**，
> 于是当前真正的**唯一奇点只剩 Capture=2**。这正是 Plan B 要处理的：**8 个 rank 若有人走 Capture、有人走别的臂，
> 则每个 epoch 的到达数差 1 ⇒ 此后每个 `SpinBarrier` 都错一代（静默 misphase）**。

### 1.2 Plan B 的做法与隔离

- `RankVote`（`tp.rs:108`）：`arrived`/`gen` + **双缓冲 parity**（`votes[(g&1)*world ..]`），**独立的世代**，
  不碰 `SpinBarrier.count/gen` ⇒ 投票本身不改变 `host_barrier` 的 epoch 序列（`tp.rs:32-38` 的设计注）；
  单轮超车可被 parity 覆盖；`world<=1` 直接 `Some(value)`。
- `verify_graph_gate`（`:5809`）：先 `verify_arm_local` 得**本 rank 的臂码**，再 `unanimous_i32(arm as i32)`；
  **全一致才走该臂，任何分歧 → 全 direct**（保守：direct 在每个状态下都存在，且是真执行）。
- `note_arm_dissent`（`:5843`）：rank0、每进程前 4 条——**这条 printf 是 Plan B 是否真的在工作的唯一外部证据**
  （「一直一致」与「一直分歧所以图从未生效」否则不可区分）。

**投「臂码」而不是投 Some/None** 是对的关键：`Dry`/`Capture`/`Replay` 都返回一个 slot idx，
但 `Capture` 单独会合 2 次、`Dry`/`Replay` 1 次——把它们混成「有 slot / 无 slot」会让一个 DRY 的 rank 和一个
REPLAY 的 rank 被判为「一致」，而两者的到达数不同，misphase 原样保留。

### 1.3 就绪度判定

| 维度 | 状态 |
|---|---|
| 原语正确性（parity/单轮超车/world=1） | ✅ 4 单测（`vote_tests`） |
| `verify_graph_gate` 语义（臂码而非 Some/None） | ✅ 代码自洽，与 `verify_arm_local` 读同一 state（`:5692` 的「纯函数」前提） |
| 默认路径零影响（开关关 → 不投票） | ✅ `:5807` 早退 |
| 单 rank（`comm=None`）零影响 | ✅ `:5810` 早退 |
| **端到端 GPU 验证** | ❌ **从未跑过** |
| **投票是否引入新时序面** | ⚠️ `[未验证]`：投票是一次跨 rank 会合 ⇒ 要求各 rank 的 `step_rows_sync` 调用次数相同（`:2737` 自述），与既有 `host_barrier` 暴露面同类 |
| **是否覆盖 draft 图臂分歧** | ❌ **不覆盖**（Plan B 只管 verify gate，见 §3 H1） |

---

## 2. 无图模式下的路径盘点（为根因假设定界）

`SWALLOW + VERIFY_GRAPH=0` 的一次 round（`dspark_spec_swallowed`，`:8074`）：

```
1  dspark_snapshot(pos, 6)        设备拷贝，无 collective
2  import_tap + draft_forward     3 个 MTP block（每 block 1 个 MoE AR + markov 头交换）
3  step_rows([anchor,d1..d5])     m=6：每层 2 个 AR（attn wo / MoE）×40 层 + 头 argmax 交换 1 round
4  spec_accept                    纯 HOST（k_emit）
5  dspark_commit                  rollback_keep + compress_replay(rows=k_emit) + set_pos_ctr(H2D)
6  note_ctx_rows(tap_r, 6, k_emit) 纯设备：memcpy_d2d + project_main_x + seed_window ×keep
7  carry_kept_tap                 memcpy_d2d
```

**在这条路径上，`host_barrier` 的调用点为零**（`step_rows_sync` 的四个臂都关了）。
唯一的跨 rank 同步是 **AR v5 的设备侧 epoch**（`ar5_wait_round` 自旋 + 500 万次自旋看门狗 → `[ar5-hang]`）。

### 2.1 哪些是 accept 相关的

| 步骤 | accept 相关？ | 性质 |
|---|---|---|
| 3 `step_rows` m=6 | ❌（恒定 6 行） | 每步 AR round 数固定 |
| 3 头 argmax 交换 | ❌（`argmax_sliced_rows` 对整块只付 **1** round，非每行） | 固定 |
| 5 `compress_replay(pos, k_emit)` | ✅ 行数 = `k_emit` | 每行 `compressor_pool_on`+`compress_commit_on`；**完成组时**多一次 `index_k_publish` |
| 5 `dspark_rollback_keep(pos,6,keep)` | ✅ 行数 = `6-keep` | `memcpy_d2d` / `dspark_ring_restore` |
| 6 `note_ctx_rows(..., k_emit)` | ✅ 行数 = `k_emit` | `memcpy_d2d` + `gemm` + `rmsnorm` + `rope`（**无 AR**） |
| 7 `carry_kept_tap(k_emit)` | ✅ | `memcpy_d2d` |

**关键负结果（我在源码上确认过的）**：`index_k_publish_kernel`（`dsv41_glue.cu:927`）只读 `*clen` 然后
逐元素拷贝——**不碰 `epoch`、不发 stamp**。所以「更多 committed rows → 更多 `index_k_publish`」**不改变 epoch 计数**。
同理 `compressor_pool/commit`、`project_main_x`、`seed_window`、`carry_kept_tap` 都不是 collective。
⇒ **「高 accept 直接改变 AR round 数」这条最直觉的假设，在进程内是对称的，不成立**（见 §3 的修正版假设）。

---

## 3. 无图 hang 的根因假设（按证据强度排序）

> 前提澄清：**「无图」= `VERIFY_GRAPH=0`。`DRAFT_GRAPH` 是另一个开关（默认 OFF），必须从日志确认。**
> 全部假设都要能被 §4 矩阵的某一格**证伪**，否则不进「下一步」。

### H1 ★★★ 最强：不是 verify 臂分歧，而是 **draft 图的臂分歧**（Plan B 覆盖不到）

**机制**：`draft_forward`（`dspark_dev.rs:1206`）是**四臂**结构，和 verify 完全同类：

| draft 臂 | 触发 | `host_barrier` |
|---|---|---:|
| direct（`!draft_graph_arm`） | `!want \|\| graph_failed \|\| !supports_ring_append \|\| !supports_memset_async \|\| (comm.is_some() && !ar_v5) \|\| win<1 \|\| pos<win \|\| seed_align \|\| seed_pos_fix \|\| unit_dump` | **0** |
| dry（`!graph_dry_done`） | 同上通过 | **0** |
| replay（`graph.is_some()`） | 同上通过 | **1**（`:1219`） |
| capture（`draft_capture`） | 同上通过 | **2**（`:1636` + `:1642`） |

**每个 rank 独有的输入**：`graph_failed`（per-rank 捕获失败 latch，`:1677`）、`self.unit`（**只有 rank0 在 `pos>0` 且
`unit_dump` 开启时置位**，`:1100-1107`）。⇒ 一个 rank 的 capture 被驱动拒绝并锁存后，**它永久 direct（0 barrier）而
peers replay（1 barrier）**，这正是 `ar5_hang` 的 misphase。

**证据**：`tp.rs:826-837` 的 `Collective::host_barrier` 注释记录了完全同类的实测：*「graph 开时单请求逐字一致，
但 6 个连续请求的第 4 个以非法访存失败，rank 逐次漂移」*——即图臂分歧的签名。而 `batched_400_v2.sh:33` 的正式矩阵
带 `DSV41_DRAFT_GRAPH=1`。

**与「低 accept 不 hang / 高 accept hang」的衔接**：`draft_graph_arm` 要求 `pos >= self.win`。**低 accept（出师表 ~2.2 tok/step）
时 `pos` 爬得慢，整段请求可能都停在 `pos < win`（全 rank direct）**；**高 accept（计数 ~6 tok/step）时 `pos` 快，
跨过 `win` 后 draft 图才启臂** —— 一启臂就暴露 capture/replay/direct 的分歧。这**恰好解释**了「出师表 0 hang / 计数 937 hang」，
而且**与 verify 图无关**。

**证伪条件**：若计数 run 的日志里**没有** `[draft_graph] captured ...` 行（脚本 `batched_400_v2.sh:681` 专门检测这一点），
H1 死。

**若 H1 成立的意义**：Plan B 只投 verify 臂 ⇒ **Plan B 一定修不了这个 hang**，但它给出了正确的修法：
**把 `draft_graph_arm` 的臂也做一次 `RankVote`**（复用同一 `RankVote` 原语；或在 `draft_capture`/replay 的 barrier 前后
对称化并投票）。这是 Plan B 的**同构扩展**，成本很低。

### H2 ★★ 较强：**batched 头 argmax 交换（`rows=6`）本身是未测路径**

**机制**：`argmax_xchg_v5_rows_kernel`（`dsv41_kernels.cu:7834`）是 **`rows` 维度的 v5 交换**，它自己的注释就写着：
*「a hang that only ever happens at rows > 1 is the v5 round-count desync the batching exists to avoid, and that is worth being visible」*，
并在 2500 万次自旋后打印 `[ar5-hang] argmax_rows rank=.. need=.. cur=.. rows=%d`。

- **lazy（m=1）走的是单行 `argmax_sliced`**，已被验证（83 tok/s、0 hang）；
- **SWALLOW（m=6）走的是 `argmax_sliced_rows`**，**只有 batched 会走**。

**与 accept 的衔接**：`rows` 恒 6，所以它不是「accept 改变了 rows」；但**它是 batched 独有的 v5 epoch round**，
且它的 parity slot 用 `(e&1)*world + my_rank`（`:7846`）。**若该 kernel 的轮次/parity 在整块交换时有边界 bug，
则只要走到 batched 就 hang，与 accept 无关**——这会让「出师表 0 hang」这一观测**主要是因为它跑得更远/更久**，
而不是因为 accept 低。[未验证：我无法从源码判定该 kernel 是否真的错；但它是最应该用 `[ar5-hang] ... rows=` 行来定位的假设。]

**判据**：hang 时 grep `[ar5-hang] argmax_rows`。**若该行出现 ⇒ H2 命中**（且它比 H1 更靠近根因，因为它就在 verify 里）。

### H3 ★ 中等：**epoch 记账的「隐藏非对称」——跨 rank 走到的 AR 内核变体不同**

**机制**：AR v5 有多个入口（`all_reduce_inplace` / `_hcpost` / `_hcpost_rows` / `_add` / `_pubred_only`），
它们各自**都会推进一次 `*epoch`**（`device.rs:3078` 的注释：「`all_reduce_inplace` → `p2p_ar_v5` 推进一个 round」）。
选择哪一个入口由 `ar_hc_post_fold` / `ar_hc_post_fold_rows` / `add_epi_ready` 等决定——
这些判断依赖 `fuse_c()` / `hc_verify_fuse()` / `supports_ar_*`（**进程级，对称**）**以及 fold 的形状 decline（也对称）**。
**所以按源码 H3 的「计数不同」被排除**；但**变体之间的看门狗阈值/协议细节不同**（`_hcpost_rows` 的 `28.6µs` vs `_hcpost` 的 `6.1µs`，
见 ar-optimization-analysis：[peer stamp 轮询 + rank drift]）——**在高 accept 下 commit 段变长、rank 漂移变大**，
可能把一个「本来就临近阈值」的 peer poll 推过看门狗。[未验证]

**判据**：hang 时 grep `[p2p-hang]` / `[p2p-prev]` 行的 `prev=/cur=/myepoch=`；若 `cur == prev`（对端冻结）而不是差 1，
指向 H3/H5（漂移/看门狗），而不是臂分歧。

### H4 ★ 中等：**SWALLOW 吞掉主链步后的 epoch「总量」与 bootstrap 轮不一致**

**机制**：round 1 走 **legacy 臂**（`step_dev` 主链 80 个 AR + 5 行 verify 81 个 ⇒ ~161 round），
round 2+ 走 **swallowed**（6 行 verify 81 round，**不含 step_dev**）。**两者每步 AR round 数相差 ~80。**
这在**每个 rank 上都一样**（`spec_primed` 是链状态，`:7746/:7965/:8643` 在同一条件下置位），所以**本身不该 hang**。
但 `seed_align` 的注释（`:7556`）明确记录了这类「两臂每轮 AR 足迹不同」就是 v5 死锁的经典来源，
且 `spec_primed` 一旦在某个 rank 上因错误路径被重置（`dspark_spec_step` 失败会把 `spec_primed` 留在原处、
下一轮回到 legacy），**就会产生一个 rank 走 legacy、其余走 swallowed 的 80-round 缺口 ⇒ 立即 hang**。
**这就把「accept 相关」重新解释为「失败重试相关」**：高 accept 的 commit/note_ctx_rows 更重 ⇒ 更容易踩到某个错误分支 ⇒
`spec_primed` 分叉。[未验证：需要 stdout 里的 `[inv-fail]` 或 Err 行来确认。]

**判据**：hang 前是否有 `[inv-fail]` / `dspark_spec_step ... returned` / `step_rows returned ... rows` 之类的 Err 行。

### H5 ★ 较弱：**纯漂移/看门狗（非计数差）**

**机制**：高 accept ⇒ `note_ctx_rows(keep=6)`（6×`project_main_x`+`seed_window`）+ `compress_replay(rows=6)` 让
**host 发射序列显著变长**；若某个 rank 被调度饿死，peer poll 超时（500 万次自旋 ≈ 0.5s）打印 `[ar5-hang]` 后
**break 并读陈旧 parity**（不是真死锁，是「看门狗 + 静默降级」）。
**特征**：`cur` 单调追上（不是永久冻结），hang 行数与运行时长相关。
**反证**：任务里计数是**短输出**（LEN=60）却有 **937** 行——不像纯漂移。倾向 H1/H2。

### H1–H5 的判定树（一句话）

```
hang 时先看三行：
  [draft_graph] captured ...        → H1（draft 图臂分歧，Plan B 结构上修不到）
  [ar5-hang] argmax_rows ... rows=  → H2（batched 头交换，verify 内）
  [inv-fail]/Err 行                 → H4（spec_primed 分叉，AR 总量差 ~80）
  [p2p-hang]/[ar5-hang] cur 冻结     → H3（变体/漂移）
  都没有，且 0 hang                 → Plan B 成功
```

---

## 4. SWALLOW 的正确测试矩阵

**前置纪律（脚本已强制，别绕）**：
- **同源双产物**：`bash kernels/cuda/build.sh 103a` **且** `cargo build --release`；binary 内嵌 `.build_id` 必须等于 `.so` 的
  （`batched_400_v2.sh` 的门禁；历史上 ATTN_PROJ_ALIGN 的第一次测试就是被 stale binary 判成「无效」）。
- **一次 serve 一个 prompt**（`[dspark]` 累加器是进程级、不按请求重置）。
- **第一次 serve 前先确认 build EXIT=0**（不能用 `tail -1` 掩盖失败）。
- 每格记录：`verify(draft/commit)_ms`、`mean-k`、`k_acc` 直方图、**是否有 `[verify_graph] captured` / `[draft_graph] captured` / `arm vote DISAGREED`**、
  **`[ar5-hang]` 总行数（并分类 argmax_rows vs pubred）**、`step pos=` 序列是否推进。

### 矩阵

| # | 目的 | 关键 env（其余沿用 `batched_400_v2.sh:134-144` 的 GATE 矩阵，**去掉 `FORBIDDEN` 三项**） | Prompt | 判据 |
|---|---|---|---|---|
| **T0** | **基线定位**：确认历史「无图 hang」到底带不带 draft 图 | `DSV41_SWALLOW_STEP=1 DSV41_VERIFY_GRAPH=0 DSV41_DRAFT_GRAPH=0`（并跑一次 `=1`，同 prompt） | 计数（高 accept，46–64 tok） | `DRAFT_GRAPH=0/1` 两跑的 hang 行数差 ⇒ **判定 H1**（若 `=1` 才 hang ⇒ H1 命中） |
| **T1** | **Plan B 正确性 + 图是否真生效** | `+ DSV41_VERIFY_GRAPH=1`（**Plan B 首次上 GPU**） | 出师表（低 accept，1000 tok） | 必须看到 `[verify_graph] captured verify_graph_m6`；**若 `arm vote DISAGREED` 出现 ⇒ Plan B 在退化（图未生效），功能应仍正确**；0 hang |
| **T2** | **Plan B 的决定性一格** | 同 T1 | 计数（高 accept） | 0 hang ⇒ Plan B 解锁；`arm vote DISAGREED` 有/无决定「图是否真的跑了」 |
| **T3** | **H1/H2 判别**：无图复现 + 关 draft 图 | `SWALLOW_STEP=1 VERIFY_GRAPH=0 DRAFT_GRAPH=0` | 计数 | 若仍 hang ⇒ **排除 H1**（draft 图不是源），转 H2/H4；hang 行 `argmax_rows rows=6` ⇒ **H2** |
| **T4** | **Plan B 的保守语义验证**：人为制造臂分歧 | 同 T1，但用 `DSV41_ENG_HOST`/`DSV41_STATS_DBG` 之类让**只有部分 rank 的 `armed` 条件不成立**（若无法只影响部分 rank，则跳过并标注） | 出师表 | 期望：`arm vote DISAGREED` + **全 direct + 输出正确 + 0 hang**——这是 Plan B「分歧→direct」的语义实证 |
| **T5** | **长跑 / 假解锁防线** | 同 T1/T2 的胜者 | 长输出（≥1000 tok 计数或代码） | `swallow-unlocked-next-plan §7`：**一次 0 hang 不算修好**；需跨过历史步数（`pos > win` 之后）再确认 |

**矩阵的因果结构（为什么这样排序）**：
- **T0 先做**：它用最小的改动（一个开关）把「无图 hang」的归因空间一分为二（H1 vs 非 H1）。**不做 T0 就去做 T1/T2，
  等于在不知道混淆变量的情况下解释 Plan B 的成败**——这正是前三次失败记录里 `[未验证]` 最多的环节。
- **T1 → T2**：Plan B 只有在 `VERIFY_GRAPH=1` 时才存在（§0-1）。T1 用低 accept 先证明「不引入回归 + 图生效」，
  T2 才是「高 accept 的解锁」判决。
- **T3** 是 T0 的反向确认（若 T0 无法单变量，用 T3 独立复现「无图 hang」并直接抓 `argmax_rows` 行）。
- **T5** 是历史教训的硬门（937 行 hang 出现在短输出上 ⇒ 长跑必须跨过 `pos > win` 与压缩激活点 `pos≈31`）。

---

## 5. 测试预期（每格的判定与后续动作）

| 格 | 期望 good | 期望 bad（与动作） |
|---|---|---|
| **T0** | `DRAFT_GRAPH=1` 才 hang ⇒ **H1**。动作：**把 Plan B 的投票原语扩到 `draft_graph_arm`**（低成本、同构），再重跑 T1/T2 | `DRAFT_GRAPH=0` 也 hang ⇒ **排除 H1**，转 T3 抓 `argmax_rows` |
| **T1** | `captured verify_graph_m6` + 0 hang + 文本逐字正确 + k_acc 与 lazy 基线一致 | 出现 `capture FAILED` ⇒ 图未生效（读 cause 行）；`k_acc` 退化/拉丁 ⇒ **投票改了时序**（转 `DSV41_DIFF_EAGER=1` 对照） |
| **T2** | **0 hang** ⇒ batched 图化解锁（Plan B 成功）⇒ 下一步 SH_PAIR(template<M=6>) parity + mrows 族 + tcgen05 | 仍 hang ⇒ 见 §6 的分支；**同时比对 `arm vote DISAGREED` 行**：有 ⇒ 分歧被正确降级但根因在别处；无 ⇒ 图一致却仍 hang ⇒ **根因与臂分歧无关**（H2/H3/H4） |
| **T3** | `argmax_rows rows=6` 行出现 ⇒ **H2**（batched 头交换）；动作：审计 `argmax_xchg_v5_rows_kernel` 的 parity/round，或**临时用 `argmax_sliced` per-row ×6 替换**（牺牲 1 个 round/块，换正确性）验证 | 无 `argmax_rows` 行、hang 在 pubred ⇒ H3/H5（漂移/看门狗） |
| **T4** | `DISAGREED` + 全 direct + 正确 + 0 hang ⇒ Plan B 语义实证通过 | 分歧却仍 hang ⇒ **Plan B 的降级路径本身有 bug**（最高优先级修） |
| **T5** | 长跑 0 hang 且跨 `pos>win`、`pos>31`（压缩激活）⇒ 可标「已验证」 | 长跑 hang ⇒ 保留 SWALLOW 无图作生产形态（止损，`swallow-nograph-plan §5.3`） |

### 预期的一句话总纲

> **Plan B 的通过线不是「变快」，而是「`VERIFY_GRAPH=1` 与 `=0` 的 hang 行数都为 0，且图确实 captured」。**
> 若 T1/T2 给出 0 hang，则 batched 图化解锁——但**它只值 −1.5ms，真正的 400 杠杆仍是 SH_PAIR/tcgen05/mrows**；
> 若 T2 仍 hang 且 T0/T3 指向 H1，则「Plan B 失败」这个判词是错的——**它只是没修到那个臂**，把投票扩到 draft gate 即可。

---

## 6. 如果 Plan B 也失败：batched 的替代路径

按「不依赖臂投票」的强度排序（都只动 verify/commit 的发射结构，不动数值）：

| 方案 | 做法 | 代价 | 评价 |
|---|---|---|---|
| **A1. 强制 `DRY=0`** | 在 `verify_arm_local` 里让 `armed=false`（或加 `DSV41_VERIFY_DRY=0` 门），使 capture 臂永不出现；用一次**显式预热**（首次进请求时 host 端单独跑一遍形状）替代 DRY 的「把 lazy 首次代价移出 capture」 | 丢掉「首次代价隔离」；warmup 那一步仍在 direct | **消除 Capture 臂（唯一 2-barrier 臂）** ⇒ 即使 rank 分歧，Direct/Dry/Replay 都是 1 ⇒ **计数天然一致**。最便宜的兜底 |
| **A2. capture 一次后恒 replay** | 每个形状**只允许一次 capture**，且把它放在**所有 rank 都 direct 的窗口**（如 `pos < win` 或 `spec_primed==false` 的首轮），之后 `verify_arm_local` 只可能返回 Replay/Direct（都 1 barrier） | 需要一个「全员 direct 的捕获窗口」；`SWALLOW_GRAPH_WARMUP_BLOCKS` 已是这个形状 | 与现有 warmup 同构，把 warmup 从「关图」改成「在 warmup 里 capture」——**capture 的 2 次 barrier 落在全 rank 同臂的窗口内** |
| **A3. 显式统一臂（不靠投票）** | 用**已在 v5 协议里承载 payload 的机制**广播一次「全局是否允许 capture」（如 `epoch+1` 的 bcast 位，或一次 4B H2D 交换），比 `RankVote` 更靠近既有承重不变量 | 需要改 v5 协议（高风险，`SEED_ALIGN` 的教训） | 不推荐，除非 A1/A2 都失败 |
| **A4. 扩 Plan B 到 draft gate** | 把 `RankVote` 用在 `draft_graph_arm` 的 capture/replay/direct 上（H1 的直接修法） | 与 Plan B 同构，1 次 `RankVote` | **若 T0 命中 H1，这应是第一动作**，而不是放弃 Plan B |
| **A5. 放弃图，只留 SWALLOW 无图** | `VERIFY_GRAPH=0 DRAFT_GRAPH=0` + SWALLOW | −1.5ms | 只有在 T0/T3 判出「无图也 hang 且非 H1」时才需要；否则图化收益太小、不值得为它冒 hang |
| **A6. 放弃 batched，回 lazy** | 现状：lazy 平台 ~83 tok/s，数学上限 ~145 tok/s | 放弃 400 | 诚实兜底（但与本任务「400 只有 batched 可达」矛盾，故只作为止损） |

**推荐顺序**：`A1 →（若 H1）A4 → A2 →（兜底）A5/A6`。
A1 之所以第一，是因为它**把一个结构性奇点（Capture=2）整类消除**，而不是去追「为什么 rank 会分歧」——
后者有 6 次失败史，前者是工程上确定可达的。

---

## 7. 一句话交付

> **Plan B 修的是 verify 四臂里唯一的不对称（Capture=2 vs 其余 1），做法是臂码 unanimity、分歧即全 direct；
> 它在 `VERIFY_GRAPH=0` 时是严格 no-op，所以「无图 hang」在结构上就不是它能修的——测试矩阵必须先开 `VERIFY_GRAPH=1`。**
> **无图 hang 的最强假设不是臂分歧，而是 draft 图的同构臂分歧（H1：`DRAFT_GRAPH=1` + `pos` 跨过 `win` 后启臂）——
> 它天然解释「出师表不 hang / 计数 hang」，且 Plan B 覆盖不到（应把 `RankVote` 扩到 draft gate）。**
> **测试矩阵：T0（单开关判 H1）→ T1/T2（Plan B 首次上 GPU，低/高 accept）→ T3（抓 `argmax_rows` 行判 H2）→ T4（分歧降级语义）→ T5（长跑防假解锁）。**
> **Plan B 的通过线是「两侧都 0 hang 且图 captured」，不是变快——它只值 −1.5ms，400 的真杠杆仍是 SH_PAIR/tcgen05/mrows。**
> **若 T2 仍 hang：先按 A1 消除 Capture 臂（最便宜的结构性兜底），再按 H1 扩投票到 draft gate。**

---

*工部 · 只读分析 + 本文件（唯一产出），未执行 GPU 命令、未改动任何源码。*
*所有行号以工作树 HEAD `64940c5` 为准；无法从源码定论的推断均显式标注 `[未验证]` 并给出证伪条件。*
