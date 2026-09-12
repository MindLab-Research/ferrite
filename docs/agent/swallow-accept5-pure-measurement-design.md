# SWALLOW accept 口径分离的测量设计 —— 高 accept 纯净测量 + 步时直接测量 + nsys

> 工部 · 2026-09-12 · **只读勘察 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 输入（现场核对）：`swallow-unlocked-shpair-m6-throughput-plan.md` · `swallow-unlocked-400-final-path.md` ·
> `swallow-contingency-plan.md` · `dspark-correctness-chain.md`（尾部 5864–5937）· `session-final-handover.md` ·
> `AGENTS.md`；读码：`crates/ferrite-dsv41/src/serve.rs` · `crates/ferrite-models/src/dsv41/chain_dev.rs` ·
> `crates/ferrite-http/src/{api.rs,driver.rs,engine.rs}` · `scripts/{sh_pair_ab.sh,l49_ab.sh,batched_400_v2.sh,
> nsys_wave1.sh,dsv41_profile.sh}`。代码基线 HEAD `d9261bb`（工作树未改）。
> **口径纪律**：每条结论标来源 —— 【实测】/【读码 file:line】/【代数】/【设计】。

---

## 0. 判决（先读十条）

1. **`56.7` 不是性能，是口径。** `tok/s = 1000·Σk_emit / Σdt`；`56.7` 是把**高 accept 区（前 ~61 行，
   k_emit=6）**与**模型退化区（k_emit≈1–2）**的时间与 token 混在一个分母里。**必须重算，不能重读**。
   【代数 + 读码】

2. **测量原语已经存在，且只有一个是对的**：`[dsv41] step pos=<p>: <X.XX>ms (<Y.Y> tok/s)`
   —— **每个 committed round 一行**（`serve.rs:478-487` 的 `step_time` 闭包；调用点 `:556`、`:748`）。
   它的 `dt` 是「模型 + host barrier + launch」的**单 round 墙**（不含 curl/HTTP/SSE）。
   ⚠️ **它打印的 `tok/s` 是 `1/dt`**（`serve.rs:484: 1.0 / dt.as_secs_f64()`），即 **k_emit=1 口径**——
   accept 5 时低报 **6 倍**。**这一条就是 56.7 与 194 不矛盾的根源。**【读码】

3. **`k_emit` 可无损提取**：同一条日志里 `k_emit_i = p_{i+1} − p_i`，`accept_i = k_emit_i − 1`
   （`l49_ab.sh:444-448`、`sh_pair_ab.sh:444-451` 已这样做）。⇒ **同一条日志同时携带「每 round 步时」与
   「每 round 产出」**，口径分离在数据层已经可能，缺的只是**分区（region）**，不是仪器。【读码】

4. **步时与 accept 无关是结构性的**：SWALLOW 的 verify 块恒 `[anchor, d1..d5]` = **6 行、ONE batched
   forward**（`dspark_spec_swallowed`，`chain_dev.rs:8708` 起；`VERIFY_ROWS = 6` 于 `:84`，静态断言
   `VERIFY_ROWS == DSPARK_DRAFTS + 1` 于 `:97`）。⇒ 每 round 的 GPU 工作量固定，**变的只是 emitted 计数**。
   ⇒ `6 / dt` 是「同一步时、高 accept」的合法换算。【读码】

5. **现有脚本的口径与本次目标正面冲突**：`STEADY_SKIP=20`（`sh_pair_ab.sh:135`、`:425`；
   `l49_ab.sh:433`）会把 **accept-5 区整段删掉**——因为高 accept 区只有 **~10–15 round**
   （61 行 ≈ 61–90 token ÷ k_emit=6）。⇒ 纯净测量**必须**换成「按 `[verify_graph] captured … at pos=`
   排除 warm-up」+「按 k_emit 分区」。【读码 + 代数】

6. **`[dspark] steps=` 今天无法承载纯净区结论**：它 ①每 **50** 个 dspark step 才打一行（`serve.rs:650`）；
   ②累加器在**请求循环之外**（`:451-455`），一个进程内的两个请求会混（`chain_dev.rs:4406` 的
   `previous request: captures=…` 证明跨请求状态确实延续）；③纯净区只有 ~10 round ⇒ **永不打印**。
   ⇒ `verify_ms/draft_ms` 的**每 round** 值需要补**一行仪表**（设计项，见 §4.3）。【读码】

7. **观测税是同步点，不是字节**：`V5_LEDGER` 每 round 5 个 D2H（4 canary + epoch）× pre/note = **10 个
   同步点/round**（`chain_dev.rs:9666, 9689, 9712, 9721`；`download_u32` 是同步 pageable memcpy，
   `devrt.rs:1302`）⇒ 任何吞吐轮 `V5_LEDGER=0`。【读码】

8. **nsys 只能做「归因」，不能做「绝对吞吐」**：DSV41 serve 无 `profiler_stop` 钩子 ⇒
   `--capture-range=cudaProfilerApi` 不可用（`nsys_wave1.sh:47-53`）；且必须 pin `DSV41_AR_V5=0
   DSV41_GRAPH_STEP=0`（设备侧 AR 自旋在 node tracing 下被放大 ~300×，`nsys_wave1.sh:44-52`、
   `dsv41_profile.sh:53-58`）⇒ **被 profile 的配置 ≠ 生产配置**。nsys 出的是「每 kernel 的 GPU 时间占比 /
   ms-per-round」，用它验证 template 上场与相对归因。【读码】

9. **`lazy 91.1` 也是混合口径**（lazy 的 v5 footprint `3 + 81·k_emit` 与 k_emit 相关，`chain_dev.rs:1908`）
   ⇒ 「SWALLOW 2× 快于 lazy」的对比必须**在同一 counting prompt、同一区间口径**下重测，否则是拿两个不同
   口径相减。【读码 + 代数】

10. **一个必须先仲裁的事实冲突**：任务称「SH_PAIR M=6 已在 56.7 的 gates 里」。仓内证据**互相矛盾**：
    `scripts/l49_ab.sh:189` 的 BASE_ENV **含** `DSV41_SH_PAIR_M=1`；
    `scripts/batched_400_v2.sh:145-156` 的 GATES **不含**；
    `swallow-unlocked-400-final-path.md:65` 明说「脚本当前矩阵不含 SH_PAIR_M ⇒ 解锁后基线是**无 SH_PAIR**」。
    ⇒ **以该次运行的 `<tag>.env`（`/proc/<pid>/environ` 回读，`sh_pair_ab.sh:555-556, 570-575`）为唯一仲裁**。
    若无该产物 ⇒ 该臂**不得携带结论**（设计已把它列为强制交付物）。

---

## 1. 定义与代数（口径分离的骨架）

### 1.1 符号

从**单个服务进程、单个请求**的日志里取出：

```
round 序列      r_i = (p_i, dt_i)     i = 0..R-1     ← `[dsv41] step pos=` 行，原文顺序
每 round 产出   k_emit_i = p_{i+1} - p_i             （i < R-1；末 round 用 completion/ledger 补齐）
                accept_i = k_emit_i - 1
求和恒等式      Σ_i k_emit_i = completion_tokens     （口径自检：必须等于客户端 usage.completion_tokens）
```

### 1.2 两种口径的公式

| 口径 | 公式 | 用途 |
|---|---|---|
| **混合**（复现 56.7） | `tok/s_mixed = 1000 · Σ_all k_emit / Σ_all dt` | 与客户端 e2e 数字对齐，**证明解析器没写错** |
| **纯净 accept-5**（本次目标） | `tok/s_pure = 1000 · Σ_{i∈H} k_emit / Σ_{i∈H} dt`，`H = {k_emit = 6}` 的高 accept 前缀 | **S0 的票面** |
| 单 round 步时 | `step_ms(H) = median / mean / min / p10 of {dt_i : i ∈ H}` | 400 的代数输入 |

### 1.3 400 的代数（不变）

```
tok/s = 1000 · k_emit / step_ms
400 tok/s  ⇒  step_ms ≤ 2.5 · k_emit
accept 5 (k_emit=6)  ⇒  step_ms ≤ 15.0ms
S0 = 31ms  ⇒  193.5 tok/s          ← 任务里的 ~194 ✓
S0 = 31ms + 设计 Σ(16.5) = 14.5ms  ⇒  414 tok/s（需 ~97% 兑现，仓史无先例）
```

⇒ **本测量要交付的唯一硬数字是 `dt(H)` 的 median（accept 5 区）**，精度目标 ±1ms
（±3ms 就是 400 的 ✓/✗ 分界）。§9 给出这个精度**可不可以达到**的诚实评估。

---

## 2. 测量原语清单（读码，逐条带落点）

| # | 原语 | 落点 | 内容 | 用途 | 陷阱 |
|---|---|---|---|---|---|
| P1 | `[dsv41] step pos=<p>: <X>ms (<Y> tok/s)` | `serve.rs:478-487`（`:556`/`:748` 调用） | **每 round** 一行：`p` = 该 round 起始位置，`X` = 该 round 墙（ms） | **主仪器**：`dt` + `k_emit`（差分） | 打印的 `Y = 1/dt` 是 k_emit=1 口径，**不是吞吐** |
| P2 | `[dspark-dbg] pos=… k_acc=… emitted=[…] commit=…ms` | `serve.rs:637-649`（`DSV41_DSPARK_DEBUG=1`） | **每 round** 的 accept/emitted token id 序列/commit 时间 | round↔文本的**精确对齐**（前 61 行的 round 边界） | 不打印 `verify_ms`（只有 commit） |
| P3 | `[dspark] steps=… mean-k=… verify=… draft=… commit=…` | `serve.rs:650-661`（每 **50** step；累加器 `:451-455`） | 进程级均值 | 长跑到 ≥50 step 时的**佐证** | 纯净区（~10 round）**永不打印**；跨请求混算 |
| P4 | `[verify_graph] captured <name> at pos=<p>` | `chain_dev.rs:6165` | 6 行块**图捕获**完成的 pos | **warm-up 边界**（Direct→Replay 的精确切点） | 缺此行 ⇒ SWALLOW 缩水，整轮作废 |
| P5 | `[verify_graph] previous request: captures=… replays=…` | `chain_dev.rs:4406-4410` | 跨请求的图状态延续 | 决定「同进程预热 + 干净请求」策略 | 被忽略 ⇒ 误判 warm-up |
| P6 | `[v5-ledger] pos=… arm=… k_emit=… epoch=… canary=…` | `chain_dev.rs:9740`（`V5_LEDGER=1`） | 每 probe 的 arm/k_emit/epoch | 结构证明（穿过旧墙）+ k_emit 旁证 | **10 个 D2H 同步点/round** ⇒ 吞吐轮必须 OFF |
| P7 | 客户端 e2e / `usage.completion_tokens` | `api.rs` SSE；`/v1/chat/completions` | 墙钟 + token 数 | 混合口径的对照 | 含 TTFT/HTTP/LOOKAHEAD 量化（见 §3.4） |
| P8 | nsys kernel 表 | `nsys stats --report cuda_gpu_kern_sum --format csv` | 每 kernel 调用数/总时间 | **纯 GPU** 归因 + template 上场证据 | 需 AR pins；不能当绝对吞吐 |

**一句话**：P1 是主仪器，P2 是标尺（对齐），P3 今天不可用，P4 划 warm-up 界，P6 关掉，P7 做对照，P8 做归因。

---

## 3. 高 accept 区间的纯净测量

### 3.1 区间定义（**不用固定 skip 数**）

```
W  = `[verify_graph] captured verify_graph_m6 at pos=<p>` 所标记的 round 下标 + 1
     （之前 SWALLOW_GRAPH_WARMUP_BLOCKS = 3 个块走 Direct，chain_dev.rs:2730 / :6338）
H  = { i : i ≥ W 且 k_emit_i = 6 } 的**极大连续前缀**    ← 主区间
H' = { i ≥ W : k_emit_i = 6 }（允许被打断，散点）        ← 稳健性交叉检查
B  = H 之后的第一个 k_emit < 6 的 round 下标（退化边界）
D  = { i > B }（退化区）
```

**必报**：`W`、`B`、`|H|`、`Σ_{i∈H} k_emit`（token 数）、`Σ_{i∈D} k_emit`、`k_emit` 直方图。

判据纪律：
- `P4` 行缺失 ⇒ **exit 2**（图没捕获 = 这轮不是 SWALLOW 的稳态，plan §2.3 `S0-5` 同款）。
- `|H| < 5` ⇒ **exit 2**（样本不足，见 §9），退回「多 run 聚合」而不是「把 D 混进来」。
- **禁止** `STEADY_SKIP`（20 > |H|，会把区间删空）。

### 3.2 三种请求形态（V1/V2/V3）

| 形态 | prompt | `max_tokens` | serve 形态 | 用途 |
|---|---|---|---|---|
| **V1** | 计数（`请从 1 数到 200，每个数字单独一行，只输出数字本身，不要任何解释。`） | **200** | 单请求单进程 | 一条日志里同时含 **H（高 accept）** 与 **D（退化区）** ⇒ 复现 56.7（混合）+ 分区；**accept 无关性**（同 run 内 6 vs ≤2 的 `dt` 对比，§4.2） |
| **V2** | 同上 | **66**（≈11 round @ k_emit 6） | 单进程 **两个请求**：请求 1 = 抛头（`max_tokens=12`），请求 2 = 测量请求 | 请求 2 **从 round 0 就在 Replay 态**（图已在请求 1 捕获，`chain_dev.rs:4406` 的跨请求延续）⇒ 无需 warm-up 扣样本，`H` 可以覆盖几乎整段 |
| **V3** | 出师表（`请完整背诵《出师表》全文…`） | 200 | 单请求单进程 | **低 accept 对照**（`k_emit∈{1,2}`）——为 §4.2 的 accept 无关性提供**跨 prompt** 的第二个证据面；同时给 nsys 的对照臂 |

> 选 66 而不是 70/100：66 = 11×6，**恰好是 k_emit=6 的整数倍**，让「H 覆盖整段」这件事在 `p` 序列上
> 自证（`p` 的每一步 +6，最后一步落点可预期）。真正的边界还是由 §3.3 钉死，66 只是让 `H` 最大化。

### 3.3 边界钉死（三层，从粗到细）

**L1（结构层，必须）**：`H` = `k_emit==6` 的连续前缀。它**不依赖任何文本假设**，是唯一必过的边界。

**L2（文本层，必须）**：客户端答案按 `l49_ab.sh:471-480` 的口径算 `ok_lines` / `first_bad`
（前 61 行 = `"1".."61"`）。**报告 `Σ_{i∈H} k_emit` 与「61 行对应的 token 数」的关系**：
- 若 `Σ_H k_emit` 明显 **小于** 61 行的 token 数 ⇒ `H` 是文本退化**之前**的纯区（好，最保守）；
- 若明显**大于** ⇒ 说明 `k_emit` 已先掉、文本还没坏，区间的口径要以 `H` 为准并如实标注。
两者都**不修改 `H`**，只作标注——这是 avoid「用文本假设反推性能区间」的纪律。

**L3（tokenizer 层，可选、精确）**：把参考串 `"1\n2\n…61\n"` 的 token 数 `T61` 在**节点上**算一次
（`/opt/dlami/nvme/models/DeepSeek-V4.1-Flash/tokenizer.json`），则「第 61 行的 token 边界」=
累计 `k_emit` 首次 ≥ `T61` 的 round。⇒ 得到**精确的 round 下标**，可把 `H` 的最后一个 round 是否
「跨过 61 行」判清楚。
> 实现选项（无需改码）：节点上 `python3 -c 'from tokenizers import Tokenizer; …'`（若装有 `tokenizers`）；
> 或退一步用 **P2 的 `emitted=[ids]`** 落盘 + 离线对照。**L3 缺失不阻塞主结论**。

### 3.4 客户端 e2e 佐证（含量化误差）

对 V2 的测量请求：`tok/s_client = (completion_tokens − 1) / (t_last_frame − t_first_frame)`（**去 TTFT**）。
⚠️ **量化误差**：`LOOKAHEAD = 16`（`serve.rs:77`、`:287-301`）⇒ 驱动层一个 `DecodeRun` 一次算
**最多 16 个 token**（accept 5 时 ≈ 3 个 round），随后逐 token 从 lookahead 缓冲吐出 ⇒ **SSE 帧是
「一簇最多 16 个 token」的脉动**，不是逐 round。⇒ 客户端分母被量化为 **≤3 个 round ≈ ≤90ms**
（在 66 token / 11 round 的尺度上 ~5% 误差带）。
**纪律**：**主数字一律取 P1（serve 侧 per-round）**；`tok/s_client` 只做旁证，且必须带这个误差带。

### 3.5 纯净测量的输出（每个 arm、每个请求一份）

```
arm, prompt_kind, max_tokens, runs
W, B, R                       (round 数)
H_rounds, H_tokens            (|H|, Σ_H k_emit)
hist_k = {0..6}               (k_emit 直方图，H / D 分开)
dt_H = {median, mean, min, p10, max}
tokps_pure = 1000·H_tokens / Σ_H dt
tokps_mixed = 1000·Σ_all k_emit / Σ_all dt       ← 必须 ≈ 客户端 e2e（自检）
step_ms_accept6 / step_ms_accept_le2             ← §4.2 的两个样本
ok_lines, first_bad, latin, has_kaishen          ← 红线（AGENTS.md:100）
env_sha / env_readback                           ← 见 §6 强制项
```

---

## 4. 步时的直接测量（answer「31ms 对吗、±1ms 可得吗」）

### 4.1 主仪器：per-round `dt` **条件于 k_emit**

```
step_ms_accept5 = median{ dt_i : i ∈ H }        ← 本次任务要钉的数
吞吐换算式      tokps_pure = 6000 / step_ms_accept5     （k_emit=6）
```
理由：这是**同一 round 的墙**，零口径混用；`dt` 含 host barrier 与 launch，**属于「生产配置的真实步时」**，
这正是 400 的分子所需（不是纯 GPU 时间）。

### 4.2 accept 无关性检验（把「6/0.031」从假设变成证据）

同一条日志（V1 的 200-token 请求）内：

```
A = { dt_i : i ≥ W, k_emit = 6 }        (高 accept)
C = { dt_i : i > B, k_emit ≤ 2 }        (退化区)
判据  |median(A) − median(C)| ≤ 1.5ms  ∧  IQR(A) ∩ IQR(C) ≠ ∅
跨 prompt 佐证：V3（出师表，全段低 accept）的 median(dt) 与 A 比较，同判据。
```
通过 ⇒ 「步时与 accept 无关」在本机**实测成立**，`6/0.031 ≈ 194` 才是可引用的数字；
不通过 ⇒ **必须报告 `step_ms(accept5)` 而**不能**用混合区的步时外推——这正是 56.7 那次踩的坑。

### 4.3 `[dspark] steps=` 的可用边界 + **建议补的一行仪表**

现状（§0-6）：每 50 step 一行、进程级累加 ⇒ 纯净区拿不到 `verify_ms`。

**建议仪表（设计项，未实施）**：在 `serve.rs:637-649` 的现有 `[dspark-dbg]` 行里**再补两个字段**：

```
+ verify={:.2}ms    ← rep.verify_ms（DsparkSpecReport 已有该字段，chain_dev.rs:8770 计时、:141 结构体）
+ varm={:?}         ← 本 round 的 VerifyArm（Direct/Replay，chain_dev.rs:6165/6520 处的判定）
```
理由：`[dspark-dbg]` 已经是**每 round**一行、**rank 0 only**、**纯打印无同步**；补两个既有字段
不增加 D2H、不改路径、默认（无 `DSV41_DSPARK_DEBUG`）零成本。⇒ 补完后可以直接得到
「每 round 的 verify_ms，按 k_emit 与 Direct/Replay 分区」，把 §4.2 的**差分检验**升级为**直接测量**，
并让 warm-up 边界从「靠 `captured` 行的 pos」升级为「每 round 带 `varm` 标签」。

**在那行落地之前**：`[dspark] steps=` 的数字**不得**用于纯净区结论（无 `|H| ≥ 50` 的场景）。

### 4.4 nsys 的 GPU 时间（纯 GPU，去掉 host barrier）

P1 的 `dt` 含 host AR rendezvous；nsys 的 per-kernel 求和是**纯 GPU**。两者之差 = host/launch 开销。
**纪律**：nsys 只用于 ①相对归因（per-kernel 占比 / ms-per-round）②template 上场证据；
**不得**把 nsys 的绝对时间与 P1 的墙混在同一张表里（被 pin 的 AR 路径不同，§0-8）。

---

## 5. nsys per-kernel（batched vs lazy，含 accept 区隔离）

### 5.1 约束（照抄现网可用形态，不要绕）

```
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --force-overwrite=true -o <rep> \
  env CUDA_VISIBLE_DEVICES=0..7 DSV41_MODEL_DIR=… DSV41_KERNELS=… \
      DSV41_AR_V5=0 DSV41_GRAPH_STEP=0 \        ← 两个 pin 都是承重的
      <ARM gates…> ./target/release/ferrite-serve --model dsv41 --serve --tp 8 …
# 结束：SIGINT（finalize report）。**不要** --capture-range=cudaProfilerApi
# 解析：nsys stats --report cuda_gpu_kern_sum --format csv <rep>     （绝不 parse 表格文本）
```
（形态来源：`nsys_wave1.sh:44-53, 66-140`；`dsv41_profile.sh:2-35, 53-77`。）

### 5.2 区间隔离：三层差分（核心设计）

| 层 | 做法 | 隔离掉什么 | 得到什么 |
|---|---|---|---|
| **L-a 长度差分** | 同一 prompt、同一进程：`max_tokens=12` 的 profile − `max_tokens=66` 的 profile | 模型加载 + 权重 warm + **prefill** + 图捕获（Direct 的 3 块） | **accept-5 区**的每 kernel 净成本 |
| **L-b 图态差分** | 同上：第一个请求（含 capture）− 后续请求（Replay） | Direct vs Replay 的差 | 6 行块图化的真实收益 |
| **L-c 臂差分** | 同一对请求跑在 ①batched（`DSV41_SWALLOW_STEP=1`，无 `LAZY_VERIFY`）②lazy（`DSV41_LAZY_VERIFY=1`）| prompt/长度/进程 | `batched − lazy` 的 **per-kernel 表**（高 accept 下） |
| **L-d 退化区对照** | 同进程内 `66` vs `200` 的差 | —— | **退化区**（k_emit≈1–2）的 per-kernel 表 ⇒ §4.2 的 GPU 侧佐证 |

归一化（**必做，否则数字不可比**）：用每次 profile 日志里 P1 的 `Σ k_emit` 与 round 数做分母，
输出 **`ms/round`** 与 **`ms/token`** 两列；占比按 `ms/round` 算。

⚠️ **禁止** `LAZY_VERIFY` 与 `SWALLOW_STEP` 同开（`batched_400_v2.sh:175` 的 FORBIDDEN；lazy 的
footprint 与 k_emit 相关，pad 无法对齐）。

### 5.3 输出表（每层一张）

```
kernel | calls/round | ms/round | % of ARM | Δ vs 对照 (ms/round)
```
必看的 kernel 名（用于「上场证据」与「归因」）：
- `gemm_fp8_sh_exp_pair_kernel<6>`（SH_PAIR M 臂真跑的**唯一**证据；`<1>` 或旧 `gemm_fp8_sh_pair_kernel` = 没上场，plan §3.2）
- 6 行块 verify 的主项（`argmax_rows`、`step_rows` 路径、hc/mix 族）
- AR 相关（**注意**：pin `DSV41_AR_V5=0` 后设备侧 publish 核不在 trace 内，host barrier 也不在 ⇒ **AR 的占比在 nsys 里被系统性低估**，必须显式声明）

### 5.4 nsys 同时充当「三证」的上场证据

SH_PAIR A/B 的「真上场」只能靠 nsys 的 kernel 名证明（`sh_pair_m` 的 decline 是**无声**的，
`return Ok(false)` 回落逐行链，plan §3.2/§7-5）。⇒ §6 的 P6 臂**不是可选项**。

---

## 6. A/B 矩阵与执行序（一次 GPU 会话，严格串行、交错）

> 承重纪律（沿用现有脚本，不改）：一臂一进程、跑前 env 回读、臂间 `pkill -9 -x ferrite-serve` + 等 0 残留、
> `flock` 单写者、收尾 `POST /shutdown`。

| # | 臂 | 关键 gate（相对 base 的差） | prompt / max_tokens | 回答 |
|---|---|---|---|---|
| **P1** | as-run 复现 | **保持该次运行的 env 原样**（由 `<tag>.env` 回读） | 计数 / 200 | 复现 56.7（混合）+ 分区 → **`tokps_pure`、`dt(H)`**；accept 无关性 |
| **P2** | **SH_PAIR OFF** | `-DSV41_SH_PAIR_M` | 计数 / 200 | **Q3**：SH_PAIR M=6 在 accept-5 区的增量（Δ`dt(H)`、Δ`tokps_pure`） |
| **P3** | 正式臂（FOLD=1） | `DSV41_SH_PAIR_M=1 DSV41_SH_PAIR_M_FOLD=1` | 计数 / 66（V2：先 12-token 预热请求） | 出货口径的纯净票面 |
| **P4** | 正式臂 − SH_PAIR | `-DSV41_SH_PAIR_M` | 计数 / 66（同 V2 形状） | 同上，A/B 的另一半 |
| **P5** | **lazy** | `DSV41_LAZY_VERIFY=1`（**无** `SWALLOW_STEP`；FORBIDDEN 其余不动） | 计数 / 66（同 V2 形状） | **Q2**：SWALLOW vs lazy 在 **同一口径**下的对比（91.1 同为混合口径，不可直接比） |
| **P6** | nsys 对 | 同 P3/P4 的 gates × `{12, 66}`（+可选 `200`） | 计数 | §5 的 per-kernel 表 + `template<6>` 上场证据 |
| **P7**（可选） | 出师表低 accept 对照 | 同 P1 | 出师表 / 200（V3） | §4.2 的跨 prompt 佐证 |

**交错与样本量**（关键：`|H| ≈ 8–12`，单 run 太薄）：
```
臂序： P1 P2 P1 P2 P1 P2          （3 对，抵消热漂；报告 3 次的 median 与极差）
      P3 P4 P5 P3 P4 P5          （3 对）
      P6（单次，nsys 单独占机）
```
**每条结论的最小证据集**：

| 结论 | 最小证据 |
|---|---|
| `tokps_pure` | 3× run 的 `H` 区间指标 + `tokps_mixed` 与客户端 e2e 相符（解析器自检）+ `[verify_graph] captured` 存在 |
| `dt(H)` = X ± 1ms | 3–5× run 的 `median(A)`，报告跨 run 极差；`Σ_H k_emit` 与 61 行的 token 数一致（L2/L3） |
| 步时与 accept 无关 | 同 run 内 `A` vs `C` 的 median 差 ≤1.5ms（§4.2）+ 出师表臂佐证 |
| SH_PAIR 上场 | `nm -D` 符号 + `/proc/<pid>/environ` 回读 + nsys `…<6>` （三证缺一不可） |
| SH_PAIR 有效 | Δ`dt(H)`（或 Δ`tokps_pure`）超过噪声带（跨 run 极差），否则判 **instruction-bound / 无收益** |
| SWALLOW vs lazy（高 accept） | P3 与 P5 的 `tokps_pure`（同 prompt、同 V2 形状、同区间定义） |

---

## 7. 判据（合取；缺证据 ⇒ exit 2，不给结论）

| # | 判据 | 通过线 | 失败指向 |
|---|---|---|---|
| J0 | USABLE：每臂 P1 行存在（`DSV41_TIMING` 生效）+ env 回读与臂定义**逐门**一致 + `nm -D` 符号（SH_PAIR 臂） | 全绿 | 幻影门 / 陈旧 `.so` ⇒ 臂作废 |
| J1 | `k_emit` 直方图（计数 prompt）| 区前 mode = **6**；退化边界 `B` 存在且被报告 | mode < 6 ⇒ accept 支线未解（400 分母作废） |
| J2 | 解析器自检 | `tokps_mixed` 与客户端 e2e 相对差 ≤ 3%（口径一致） | 差 > 3% ⇒ 请求切分错（同进程双请求混算） |
| J3 | 纯净数字 | `tokps_pure` + `dt(H)` 带 `|H|`、`Σ_H k_emit`、跨 run 极差 | `|H| < 5` ⇒ 只报「样本不足」 |
| J4 | accept 无关性 | `|median(A) − median(C)| ≤ 1.5ms` | 超线 ⇒ 步时确实依赖 accept，194 需降级为区间 |
| J5 | 红线 | 前 61 行正确 + 出师表零**额外**拉丁 + 0 `ar5-hang` + 0 panic | 任一破 ⇒ 读数作废（先修再测） |
| J6 | nsys（P6） | `gemm_fp8_sh_exp_pair_kernel<6>` 出现（SH_PAIR 臂）＋ CSV 非空（≥5 行 kernel） | 空 CSV = 被 profile 的是 `env`（`dsv41_profile.sh:60-66` 的旧坑） |

---

## 8. 陷阱清单（本测量专属，全部有源码/文档理由）

1. **`[dsv41] step pos=` 的 `tok/s` 是 `1/dt`**（`serve.rs:484`）——k_emit=1 口径，accept 5 低报 6×。
   任何「从日志直接 grep tok/s」的做法都会重犯 56.7 的错。
2. **`STEADY_SKIP=20` 会删光高 accept 区**（`sh_pair_ab.sh:135/425`）⇒ 必须换成 `captured`-pos 划界。
3. **`[dspark] steps=` 每 50 step 才有、且进程级累加**（`serve.rs:650`, `:451-455`）⇒ 纯净区永不打印；
   两请求同进程必混。
4. **`V5_LEDGER` = 10 个同步点/round**（`chain_dev.rs:9666/9689/9712/9721`）⇒ 吞吐轮必须 OFF
   （`batched_400_v2.sh:157-167` 已这样写，照抄）。
5. **`LOOKAHEAD=16` 让 SSE 帧成簇**（`serve.rs:77/287-301`）⇒ 客户端逐帧计时不是逐 round 计时，
   纯净数字只能取 serve 侧。
6. **nsys 的配置 ≠ 生产配置**（`DSV41_AR_V5=0`+`DSV41_GRAPH_STEP=0`）⇒ nsys 数字只做归因；
   且此 pin 会让 **AR 占比被低估**（publish 核不在 trace）。
7. **`sh_pair_m` 的 decline 无声**（`return Ok(false)`）⇒ 没 nsys 的 `…<6>` 就**不得**说「已上场」。
8. **SH_PAIR 是否在 as-run env 里：仓内证据冲突**（`l49_ab.sh:189` 有 / `batched_400_v2.sh:145` 无 /
   `final-path §65` 说无）⇒ **以 `<tag>.env` 为准**，否则 P1/P2 的差会被记成「SH_PAIR 增量」而其实是
   两次不同配置。
9. **62+ 行的退化是模型行为，不是性能悬崖**（AGENTS.md:100；`dspark-correctness-chain.md:4373`）⇒
   `k_emit` 掉到 1–2 **不是**引擎回归，不要把它记成「步时退化」。
10. **`LAZY_VERIFY` 与 `SWALLOW_STEP` 不得同开**（`batched_400_v2.sh:175`）；gate 串逐字手写，
   不直接跑 `batched_400_v2.sh`（其 `:154` 仍写 `SWALLOW_EPOCH_PAD=1`，与 `t1t4` P4 的 over-pad 结论冲突）。
11. **`V2` 的第二请求必须显式记住「图已在请求 1 捕获」**（`chain_dev.rs:4406`）⇒ 不要把请求 1 的
    Direct 块算进 `H`；反之，若只跑单请求，前 3 个 verify 块**必然**是 Direct（`:2730/:6338`）。
12. **无 `<tag>.env` ⇒ 该臂不作数**：本仓 #1 陷阱就是「设了没生效 / 漏设」（plan §3.2）。

---

## 9. 精度上限（诚实声明，供给侧）

```
高 accept 区的**天然样本量**（每 run）：
  61 行 ≈ 61–90 token（"1\n2\n…61\n"）÷ k_emit 6  ≈ 10–15 round
  减 warm-up（单请求跑法）：3 round（SWALLOW_GRAPH_WARMUP_BLOCKS）
  ⇒ V2 跑法（预热请求 + 干净测量请求）可把 |H| 提到 ≈ 10–12；V1 跑法 ≈ 7–12
单 run 的 median 标准误差（设每 round dt 的 σ ≈ 1–2ms，GPU-bound）：
  SE ≈ σ/√|H| ≈ 0.3–0.6ms
3 run 聚合（≈30 样本）：SE ≈ 0.2–0.35ms
⇒ **±1ms 可达；±0.3ms 不可达**（除非 ≥5 run 或专门的固定负载微基准）
```
⇒ 结论写法：`dt(H) = X ms（3 run 极差 a–b）`，并把它与 400 的 15.0ms 门限并列，**不要**写成单个小数。
另外提醒：`dt` 含 host barrier（生产口径），**不是** GPU 下限；后者只有 nsys 能给（§4.4/§5）。

---

## 10. 交付清单（供尚书省分派；本文件只给设计）

| # | 项 | 内容 | 优先级 | 风险 |
|---|---|---|---|---|
| 1 | 脚本 `scripts/swallow_accept5_pure.sh` | 复用 `sh_pair_ab.sh` 的骨架（`:164-230` 锁/envchk、`:540-620` 串行 serve），**判据替换为 §3.5 的分区解析**；臂 = P1..P5，交错 3 轮 | **P0** | 无（纯测量） |
| 2 | 解析器 `scripts/accept5_judge.py` | `rounds=[(p,dt)]` → `k_emit` → `W`(captured pos) → `H` → `tokps_pure/mixed`、`dt(H)` 统计、`A/C` 无关性、直方图、红线 | **P0** | 低 |
| 3 | 仪表（一行，可选但强烈建议） | `serve.rs:637-649` 的 `[dspark-dbg]` 行补 `verify={}ms` + `varm={:?}`（字段已存在）⇒ 每 round verify 归因 | **P1** | 低（仅 debug 门内） |
| 4 | nsys 对（P6） | 长度差分 {12,66,(200)} × {batched, lazy}，CSV 落盘 + `… <6>` 上场证据 | **P1** | 中（AR 占比低估需声明） |
| 5 | 边界精确化（L3） | 节点上算 `T61`（`tokenizer.json`）或落盘 `emitted=[ids]` 离线对照 | P2 | 低 |
| 6 | gate 卫生 | `batched_400_v2.sh`: `EPOCH_PAD → DYNAMIC_PAD` + FORBIDDEN 增列「EPOCH 与 DYNAMIC 同设」 | P2 | 低（需重跑基线） |

**不做**：
- 不用 `STEADY_SKIP`（会删空区间）；
- 不在吞吐轮开 `V5_LEDGER`（10 同步点/round）；
- 不把客户端 SSE 逐帧时间当逐 round 时间；
- 不把 nsys 绝对时间与 serve 墙并列；
- 不把 SH_PAIR 与其它 gate 同轮（单变量）；
- **不在没有 `<tag>.env` 回读的情况下引用任何臂的数字**。

---

## 11. 一页纸结论（回答任务 Q1–Q3）

1. **Q1（SWALLOW 前 61 行吞吐）**：口径分离只需**一个已有原语**（`[dsv41] step pos=`）＋ **一次分区**
   （`H = k_emit==6 前缀`，warm-up 由 `[verify_graph] captured … at pos=` 划界）。
   `tokps_pure = 6000 / median(dt(H))`；在 `dt(H)=31ms` 的假设下 = **193.5 tok/s**——
   **但这必须是实测的 `median(dt(H))`，不是从 56.7 反推的。**
2. **Q2（SWALLOW vs lazy）**：**91.1 也是混合口径**，不能直接与 194 对比；必须让 lazy 在同一 counting
   prompt、同一区间定义下跑一遍（P5），比 `tokps_pure`。「SWALLOW 2× 快」在测量前**不成立**。
3. **Q3（SH_PAIR 的增量）**：① 先仲裁 as-run env（`<tag>.env`）——仓内证据冲突；② 变是**单变量**
   A/B（P2/P4 去掉 `DSV41_SH_PAIR_M`），读数取 **Δ`dt(H)`** 与 **Δ`tokps_pure`**；
   ③ 上场由 nsys 的 `gemm_fp8_sh_exp_pair_kernel<6>` 独占证明。
   ⇒ **A/B 分离是 SH_PAIR 的唯一验收方式**，且预期值按 **−0.8~2.8ms** 编（nsys 实测族占比 ~8–9%），
   不按 launch 账 −4.9~7.9。

---

*工部 · 只读勘察 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*所有数字标来源（实测 / 读码 file:line / 代数 / 设计）；行号以 HEAD `d9261bb` 为准；与任务前提冲突处已显式给出依据与 file:line（§0-10）。*
