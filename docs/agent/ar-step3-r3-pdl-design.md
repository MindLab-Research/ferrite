# AR R3（A1b PDL）设计 —— 程序化依赖加载能不能吃下 77.7µs 的等待

> 工部 · 2026-09-12 · **只读勘察 + 本文件（唯一产出）**。未改任何源码、未改任何 gate 默认、未执行任何 GPU 命令。
> 输入：`docs/agent/ar-step2-a1a-fix-design.md`（A1a 弃案 + R1/R2/R3 重定向）、
> `docs/agent/ar-l4l5-optimization-design.md`（A0/A1b 设计）、`docs/agent/ar-further-optimization.md`、
> `docs/agent/swallow-ar-first-step-design.md`（对 78.3µs 的四条修正）、
> `docs/agent/dsv41-persistent-arch.md` §4.2（PDL 串链现状）。
> 代码基线 HEAD 工作树；`file:line` 现场核对，**读码一律以函数名为准**（有 peer 在改 `chain_dev.rs`/`kedev`，行号会漂）。

---

## 0. 判决（先读五条）

**条件 GO —— 但 R3 不是 6.58ms 的杠杆，是 R2 之后的收尾项。**

1. **❗PDL 不能消掉那 77.7µs。** 77.7µs 是**跨 rank rendezvous**（等最慢 peer 的 stamp）；
   PDL 是**同一 stream 上相邻两个 grid 的 launch 重叠**机制。两者是正交的两张网（§2 给出三条不可行证明）。
   ⇒ 把 R3 写成"隐藏 77.7µs"的任务，前提不成立。
2. **❗那 77.7µs 本身还没有证据。** `0.6µs = 122,880B ÷ 200GB/s` 只算了**一份** payload，
   而每轮的远程写是 **8 peer × 120KB = 960KB**（`p2p_ar_store_v5_kernel`，`gridDim.y = world`），
   且 **store 是独立 kernel 行、不在那张 top-6 表里**（`swallow-ar-first-step-design.md` §0 已判过这条算法漏项）。
   ⇒ R3 的**第一步不是写 PDL，是拿 A0 探针**（`DSV41_AR_PROBE=1`，**非 nsys**）。
3. **PDL 真正能覆盖的只有三块，合计 ~2–5µs/轮**（§4）：AR 自身两发的 node-gap（~0.3–0.8µs）、
   **后继的 pre-GDS prologue 藏进 AR 的 spin**（1–3µs，本设计的核心）、AR 尾（reduce+fold）被盖住（~1–2µs）。
   ⇒ 84 轮 × ⇒ **0.17–0.4ms/步**（≈ 步时的 0.6–1.5%）。R2 把轮数砍半后，R3 同比例缩水。
4. **增量最大的那一条恰恰风险最高**：用 PTLC 把 spin 变成"免费 SM/HBM 窗口"，
   挂一个**故意不调 GDS** 的预取 filler（树内先例 `dsv41_experts_mxf4.cu:3022-3028` 的 w2 prefetch）。
   但本项目**已实测过反例**：`W2_PREWARM` 的 warmer 与 gateup 尾部争 SM，**+0.04ms（收益变负）**
   （`dsv41-session-final-report.md`）。⇒ R3-C **默认 OFF、必须实测、可无条件回退**。
5. **红线（从 A1a 的尸检里抄下来的）**：R3 的任何一项都**只允许改时序，不允许改数据流与数值**。
   不折 store、不动 reduce 的升序 rank、不改 epoch 语义、**不在两个 AR 之间放 PDL 边**。违反任何一条 ⇒ 退回。

**排序：R1 → R2 →（A0 探针）→ R3。** 若只能投一件，投 R2：它的量级是 R3 全部的 10 倍。

---

## 1. 现场事实（决策的前提，逐条核对）

### 1.1 AR 的 launch 形状

| 组件 | 位置 | launch 形式 | 图内? |
|---|---|---|---|
| store（peer 并行远程写） | `ferrite_kernels.cu` `p2p_ar_store_v5_kernel`（`dim3(blocks, world, 1)`） | plain `<<<>>>` | 是 |
| pubred（publish+reduce 融合） | `::p2p_ar_pubred_v5_kernel` | plain `<<<>>>` | 是 |
| pubred + hc_post（decode） | `::p2p_ar_pubred_v5_hcpost_kernel` | plain `<<<>>>` | 是 |
| pubred + hc_post（verify 多行） | `::p2p_ar_pubred_v5_hcpost_rows_kernel` | plain `<<<>>>` | 是 |
| 等待（三变体共用） | `::ar5_wait_round`（A4 单块臂 / OFF 惊群臂） | —（device helper） | — |
| Rust 入口 | `tp.rs` `all_reduce_inplace*` / `end_round`、`device.rs` `p2p_ar_pubred_v5*` | — | — |
| 调用点（decode） | `chain_dev.rs` `layer()`（attn，`:15693-15716`）+ `moe_reduce()`（MoE，`:5524-5588`） | — | MoE 侧注释仍写"图外"（历史） |

**6 个 `extern "C"` launcher 全部是 plain `<<<>>>`**（`ferrite_p2p_ar_v5` / `_add` / `_pubred_v5` /
`_pubred_v5_moe` / `_hcpost*`）—— 一处 PDL 属性都没有。

### 1.2 ★致命事实：pubred 的入口读就是生产者依赖

```cuda
__global__ void p2p_ar_pubred_v5_kernel(...) {
    const unsigned e = *epoch;          // ← 第 1 条语句：epoch 是【上一个 pubred】写的
    ...
    if (blockIdx.x == 0 && threadIdx.x < world) {
        atomicExch_system(&ready_tbl[threadIdx.x][my_rank], e + 1u);  // stamp：还要等 store 的写可见
```

两条结论（决定了 R3 的形状）：

- **(a)** 若把 **AR 自己**做成 PDL secondary，它的 `cudaGridDependencySynchronize()` 必须放在
  `*epoch` **之前**——而那是函数第一条语句 ⇒ **AR 作为 secondary 可被 hoist 的 prologue ≈ 0**
  （只剩 grid rasterisation / CTA 调度）。这与 `dsv41_kernels.cu:1147-1149`
  对 `indexer` 的判定同一个形态（"prologue 是生产者相关的，没东西可提"）。
- **(b)** 反过来把 **AR 当 primary**、让**后继**提前发射，才是唯一有肉的方向（§3.2）。

### 1.3 PDL 基建已经在树内（三份 file-static 副本 + 一条默认 ON 的生产链）

| TU | helper | gate | 现状 |
|---|---|---|---|
| `ferrite_kernels.cu:770` | `pdl_or_plain` | `FERRITE_PDL`（默认 **OFF**） | GLM 侧 ~4 个 launcher |
| `dsv41_kernels.cu:4211` | `dsv41_pdl_or_plain` | `DSV41_PDL`（默认 **ON**，`=0` 回退） | attention 投影链 8 点 |
| `dsv41_experts_mxf4.cu:919` | `dsv41_experts_pdl_or_plain` | `DSV41_PDL` | `quant_fp4 → gateup → down_reduce` 3 点 |

⇒ **PDL 属性在 capture 下存活已被本项目自证**：`ferrite_kernels.cu:8569-8608` 的 `ferrite_pdl_exp`
`mode 2/3` 就是"图捕获下 normal vs PDL"的对照，且 `DSV41_PDL` 链默认 ON 已在生产图里跑。
**这正是 R3-A 需要的先例**（不必再赌 capture 语义）。

**GDS 契约（必须照抄）**：`cudaGridDependencySynchronize()` 放在核**入口**、
`#if __CUDA_ARCH__ >= 900` 内无条件执行（plain launch 上是 documented no-op），
且必须在**任何**读生产者输出的语句**之前**。

### 1.4 AR 的模式/图边界

- 整步一张图：`DSV41_GRAPH_STEP` **默认 ON**（`chain_dev.rs` `step_impl`），
  且 `ar_v5() = GRAPH_STEP || DSV41_AR_V5`（`tp.rs::ar_v5`）——两者默认都 ON ⇒ **AR 在图内**。
- 图用 **stream capture**（`ferrite-kernel/src/cuda.rs` 的 `cuStreamBeginCapture` 驱动路径），
  **不是**手工 `cudaGraphAddKernelNode` ⇒ "cudaGraph 的 PDL 节点"只能**由 capture 隐式形成**（§3.4）。
- 侧流 fork/join（`cudaEventDisableTiming`）**就布在 AR 附近**（`chain_dev.rs` 的 `hc_tail_join()` 在 AR 前）——
  这是 §3.4 的图边风险来源。

### 1.5 A4（R1）已在树内

`ar5_wait_round(..., single_poll, ...)` 的双臂已实现（`ferrite_kernels.cu:9229`）：A4 臂 =
**block 0 轮询 8 个 peer + 写 `epoch+1` 广播字**，其余块只等**一个本地字**。
⇒ R1 落地后，**AR 的绝大多数 block 更闲**（只有 block 0 在盯 peer），
**spin filler（R3-C）的理论窗口反而更大**（§3.3）。

---

## 2. PDL 语义（精确版）——为什么它吃不掉 77.7µs

### 2.1 官方语义（照抄，别引申）

```
primary : 所有 block 都调用 cudaTriggerProgrammaticLaunchCompletion()
          ⇒ driver 可以在【所有 primary block 已 launch 且已执行 PTLC】后发射 secondary
          （primary 不调 ⇒ 隐式在【所有 CTA 退出】时触发）
secondary: 带 cudaLaunchAttributeProgrammaticStreamSerialization 发射
          ⇒ 可以在 primary 还没跑完时就被调度
          ⇒ 但必须在 cudaGridDependencySynchronize() 之前只做"不读 primary 输出"的工作，
             GDS 会 block 到【primary 完成并把结果刷进 global memory】
```

⇒ 一句话：**PTLC 管"什么时候能发射"，GDS 管"什么时候能读"**。两者都不触碰别的同步。

### 2.2 三条不可行证明（对"PDL 消掉 77.7µs"的逐条否定）

| # | 命题 | 为什么不行 |
|---|---|---|
| **N1** | "PDL 让 peer 更早 stamp" | stamp 的时点 = **对端 rank 自己的本地进度**（它的 producer 跑到哪儿）。PDL 是本 rank、本 stream 的机制，**不能让另一个进程的 device 代码提前**。 |
| **N2** | "PDL 把 spin 的时间让给别的 SM 做有用工作" | **只能让"同一 stream 的后续 kernel"提前发射**，且它的工作必须**不读 AR 输出**。AR 的输出是下一层全部计算的输入（`o`→hc_post→`h`）⇒ 后继能提前做的只有 **pre-GDS prologue**，不是整段工作。 |
| **N3** | "PDL 把 AR 的 spin 从 critical path 挪走" | spin 是**本 kernel 内部**的轮询；kernel 不结束 ⇒ stream 不推进。PDL 能让后继**开始**（N2 的窗口），但**不能**让 AR 这个节点消失。 |

> ⇒ **PDL 的收益上界 = 后继的 pre-GDS 工作 + AR 自身的 node-gap，与"peer 到达 skew"无关。**
> 反过来说：如果 A0 探针显示 spin 真的是 ~77µs，那它的成因是 **rank 负重不平**（A2c）或 **轮数**（R2），
> 不是 launch 机制——**别在 R3 上找它**。

### 2.3 一个必须说清的细节：为什么 R3 的窗口是"spin 期"而不是"尾"

若 AR 不调 PTLC，隐式触发在 **CTA 退出**时 ⇒ 后继只能在 AR 的**尾**（reduce+fold 的 ramp-down）重叠，窗口 ~1–2µs。
若 AR 在**入口**调 PTLC（每个 block 都调；AR 的 grid = `64 × 20`，天然全常驻）⇒ 后继在整个 spin 期就被调度，
它的 pre-GDS 工作**跑在 spin 里**，窗口 = **整个 spin**。这就是 `cudaTriggerProgrammaticLaunchCompletion`
在 R3 里的全部价值：**把"尾窗口"换成"spin 窗口"**。

**不变量（写进核头注释）**：AR 在 PTLC 点之后写下的任何东西（staging 读、`out` 写、hc_post 的 `hc_res` 写），
**都只能被认为在 GDS 之后才可见**。⇒ 这条自动成立，因为后继的 pre-GDS 阶段本来就不允许读 AR 的输出。
（若将来有人想让 filler 读 `epoch`/stamp，**立刻违规** —— 见 §6 红线 R-2。）

---

## 3. 设计

> 三项互相独立、各自一个 env 门、各自可回退。**默认全 OFF**。

### 3.1 R3-A：AR 自身两发的 PDL 串链（node-gap，最小项）

**改动**：在 `ferrite_kernels.cu` 加**第 4 份** `pdl_or_plain` 副本（file-static，不能跨 TU 链接），
名 `ferrite_ar_pdl_or_plain`，gate **`DSV41_AR_PDL`**（默认 OFF，`=0`/unset ⇒ 现状），
把 **pubred 一族**的 launch 从 `<<<>>>` 换成它：

```
p2p_ar_store_v5_kernel<<<dim3(blocks, world, 1), threads, 0, s>>>(...);   // primary：不动（隐式 PTLC）
ferrite_ar_pdl_or_plain(p2p_ar_pubred_v5_kernel, blocks, threads, 0, s, ...); // secondary：带 attr
```

**注意三件事**：
1. **store 不加 attr**（它是 primary；primary 只需要 PTLC，而 store 没有值得提前的点 ⇒ 用隐式触发）。
2. **pubred 不加 GDS** —— 除非它同时是别的 kernel 的 secondary。在 R3-A 里它只做 secondary of store，
   而 store 的写它并不读：它读的是 `staging_local`（**更早**的 producer 写的），以及 `*epoch`；
   而 store 也**只读**同一个 `*epoch`（`p2p_ar_store_v5_kernel` 的 `const unsigned e = *epoch;`），
   **两者都不写 epoch** ⇒ 相互之间无依赖 ⇒ **pubred 相对 store 不需要 GDS**。
   ⇒ R3-A 的 pubred **不加 GDS**，只加 attr（这反而让它真的"零语义"）。**这一点必须在实机上用 A/B 复核**
   （若发现 pubred 需要 GDS，说明有我们没读到的依赖，回退 R3-A）。
   ⚠️ **前置澄清**：此时 pubred 是"secondary of store"，而 store 若自己也是 secondary（它的 predecessor
   是 producer gemv），就变成三态链——**R3-A 只做 store→pubred 这一条边**，
   上游边（producer→store）留给 R3-B 按 site 审计，避免"一条链上多种 GDS 契约"混在一起。
3. launcher 有 6 个（`_v5`/`_add`/`_pubred_v5`/`_pubred_v5_moe`/`_hcpost`/`_hcpost_rows`/`_hcpost_add`），
   只改 **pubred 的那一发**，逐点改、逐点可关。

**预期**：~0.3–0.8µs/轮 ⇒ 84 轮 ⇒ **0.03–0.07ms/步**。
**判定**：太小，**单独不值得投**；并入 R3-B 一起做（同一个门、同一次 A/B）。

### 3.2 R3-B：PTLC + 后继的 spin 窗口（★本设计的核心）

**改动两步**：

**(B-1) 在 AR 里埋 PTLC**（`ferrite_kernels.cu`，全部 gated）：

```cuda
// 在 p2p_ar_pubred_v5_kernel / _hcpost_kernel / _hcpost_rows_kernel 的【入口】、
// 所有 block 都会执行到的位置（不是 blockIdx.x==0 里面）。
#if __CUDA_ARCH__ >= 900
    if (ar_pdl) cudaTriggerProgrammaticLaunchCompletion();
#endif
```

- 位置选择：**在 stamp 之前**。理由：PTLC 表示"我的后继可以发射了"，而后继的 pre-GDS 工作不读 stamp、
  不读 epoch、不读 staging ⇒ 越早触发窗口越大（§2.3）。
  ⚠️ 反过来说也成立：**PTLC 之后的 AR 写的一切，对未过 GDS 的后继都不可见** —— 这正是我们要的。
- **每个 block 都必须执行**：条件必须是"gate + 架构"，不能带 `blockIdx`。
- **gate 必须进程读一次**（`ferrite_ar_pdl()` 的 `OnceLock`/静态缓存范式，照
  `ferrite_ar_single_poll`/`ferrite_ar_probe`）：它选的是内核行为，**capture 与每次 replay 必须一致**
  （capture 里逐次 `getenv` 是会被烘进图的隐患，`ar-step1` §3 已写过这条）。

**(B-2) 后继加 attr + 入口 GDS**（落到真正读 AR 输出的那个 kernel）：

| AR site | AR 之后的下一个 kernel | 是否已是 PDL 覆盖 | R3-B 动作 |
|---|---|---|---|
| attn（`_hcpost`，`HCPOST_EPI=1`） | hc_post 已折进 AR 的 epilogue ⇒ 后继 = **MoE 前端**（rmsnorm/router） | 否 | 加 attr + 入口 GDS |
| attn（`HCPOST_EPI=0`） | 独立 `hc_post_inplace` | 否 | 同上 |
| MoE（`_hcpost_add`） | **下一层的 front**（`hc_collapse`/`rmsnorm`） | 否 | 同上 |
| verify rows（`_hcpost_rows`） | 下一行的 front / tap | 否 | 同上（收益 ×k_emit） |

- **只对"直接后继"加**（PDL 是相邻 pair 的机制，隔一个节点就失效）。
- **禁止 AR→AR 的 PDL 边**：若直接后继是另一个 AR 的 launch，**本 site 直接放弃 R3-B**
  （v5 的 epoch 邻接契约 + R1 bug 2 的教训：`const unsigned e = *epoch` 必须读到本轮的值）。
- **属于 R3-B 的 pre-GDS 工作清单**（这是收益的来源，必须逐 kernel 审计并**写进 patch 描述**）：
  地址/指针表计算、smem 初始化、LUT 建表、与生产者无关的权重预取、grid 常量推导。
  ⚠️ **权重预取是否"与生产者无关"要逐个看**：`dsv41_kernels.cu:4038-4042`、`1423-1427`
  两处都明确写了"`w` 可能是上一个节点写的，PDL 下提前读是 race"⇒ **这类"看权重"的 prologue 不算可提**。
  真正可提的是**不碰任何被生产者写过的 buffer** 的那部分。

**预期**：1–3µs/轮（后继 prologue）+ 1–2µs/轮（AR 尾被盖）⇒ 84 轮 ⇒ **0.08–0.25ms/步**。
**风险**：中。RL 上最大的一类风险是"**GDS 放错位置 ⇒ 静默读脏**" ⇒ 必须逐 kernel 用 parity/逐 token 验（§5）。

### 3.3 R3-C：无 GDS 的 "spin filler"（唯一能把 77.7µs 变成产出的路子，也是最高风险）

**机制**：AR 入口调 PTLC 后，挂一个**故意不调 GDS** 的小 kernel（PDL secondary），
在 spin 期间把**下一轮确定会读**的权重从 HBM 拉进 L2（或做别的无依赖工作）。

**树内先例（照抄的模板）**：`dsv41_experts_mxf4.cu:3022-3028` 的 w2 prefetch kernel ——
"deliberately does NOT call cudaGridDependencySynchronize(): it reads only `ids` (a router output
several kernels upstream) and the w2 pools (weights), never anything the gate/up launch wrote,
so it is legal for it to race the producer"。

**硬约束**：
1. **filler 不得读**：AR 的输出（`out`/`o`/`h`/`hc_res`）、`epoch`、`ready_*`、`staging_*`。只能读**权重 + 更早的元数据**。
2. **入口/出口护栏**：filler 必须在"下一轮的第一个 kernel 需要这些权重"之前**结束**，否则它挤占下一轮的带宽。
   ⇒ 小网格（跨 SM 数上限）+ 有界工作量 ⇒ 用 `DSV41_AR_FILLER_BLOCKS` 限制。
3. **默认 OFF**，且**只在 spin ≥ 阈值时发**（用 `probe`/设备侧观测；先不做，等 A0 数据）。
4. **数值安全**：纯预取只影响时序 ⇒ **无位级风险**（这是它相对 A1a 的最大优点，也是它唯一值得赌的地方）。

**反例警示（必须先读）**：本项目 `W2_PREWARM` 的 warmer 与 gateup 尾部**争 SM**，
PDL 重叠变成**争抢**，实测 **+0.04ms（负收益）**，最后关掉（`dsv41-session-final-report.md` §2 第 2 行）。
⇒ R3-C **不是"免费"的**，它的成败就是"窗口里到底有没有真空闲资源"。

**预期**：**0 ~ 0.3ms/步（未证）**。判据实验见 §5 的 `mode 4`。
**前置**：R3-B 的窗口被实测 ≥ 10µs 才做。

### 3.4 图捕获与 PDL 的合法性（必读，R3 的隐藏雷区）

- **属性在 capture 下存活**：有先例（§1.3）⇒ R3-A/B 的 attr 随 capture 进图，**不需要手工建图**。
- **但 programmatic 依赖边只能 kernel→kernel**：CUDA 要求 `cudaGraphDependencyTypeProgrammatic`
  必须配 `cudaGraphKernelNodePortProgrammatic` / `...PortLaunchCompletion`，且**只在 kernel node 之间**。
  ⇒ 若 AR 图的**直接后继节点是 memcpy / event record**（侧流 fork/join 恰好在 AR 附近，`hc_tail_join()`），
  PDL 边非法 ⇒ **instantiate 失败**（响亮，可回退）或退化成普通边（静默，白做）。
  ⇒ **落地第一条**：把 AR 在图里的**直接后继节点的类型**打印出来（capture 时 dump 一次），再决定 attr 挂哪。
- **fallback 路线（可选，更大改动）**：不靠 capture，改成 `cudaGraphKernelNodeSetAttribute` +
  `cuGraphAddDependencies_v2`（带 `cudaGraphDependencyTypeProgrammatic`）**手工改边**。
  项目现在是 stream capture（`cuStreamBeginCapture`）⇒ 这条只在 capture 路走不通时才考虑。
- **PTLC 在非 PDL 图里必须是 no-op** —— 这一条**没有现成证据**，必须实测（§5 的 `mode 5`）。

---

## 4. 预期收益（回答"77.7µs 里能重叠多少"）

把 77.7µs 拆成"PDL 够得着 / 够不着"两栏：

| 段的量级 | 估计 | PDL 能覆盖 | 依据 |
|---|---|---|---|
| 对等端到达 skew（**spin 主体**） | 未知 | **0（N1/N2/N3）** | §2.2 |
| store 的远程写（8 × 120KB） | 2–5µs | 0（数据搬运） | `p2p_ar_store_v5_kernel` |
| stamp（8 次 remote atomicExch_system + fence） | 0.2–0.5µs | 0 | `ar5_wait_round` |
| AR 自身两发的 node-gap | 0.4–1µs | **~0.3–0.8** | R3-A |
| AR 尾（reduce+fold，被后继 prologue 盖） | 1–2µs | **~1–2** | R3-B |
| 后继的 pre-GDS prologue（跑在 spin 里） | 1–3µs | **~1–3** | R3-B |
| spin 里的 filler（预取） | 0–? | **0 ~ ?** | R3-C（未证，有负收益先例） |
| **合计（PDL 覆盖）** | | **~2–5µs/轮 ⇒ 0.17–0.4ms/步（84 轮）** | |

**三个必须并排说清的数字**（避免又一次口径混淆）：

| 口径 | 值 | 含义 |
|---|---|---|
| 现状（nsys，SWALLOW） | 84 轮 × 78.3µs = **6.58ms/步** | 含 nsys 自旋放大，分母是 kernel-sum |
| 账本（v5 协议地板） | ~17.3µs/轮 = **1.45ms/步** | `lazy-verify-optimization-path.md` |
| **R3 的增量** | **0.17–0.4ms/步** | **≤1.5% 步时；R2 后减半（44 轮）** |

> **结论句**：R3 是"把 AR 的尾巴和邻居的头部对折一下"，不是"消掉等待"。
> 要 AR 离开 #1，杠杆在 **R2 的轮数**（−3.3ms）和 **A0 给的归因**，不在这里。

---

## 5. 执行顺序与验收（每一步都可执行；GPU 部分由有卡侧执行）

### 5.0 前置（**零代码成本，必须先做**）

1. **A0 探针**：`DSV41_AR_PROBE=1` + 生产 gate 集，**不要 nsys**，读 stdout 的 `[ar-probe]`
   （site 0=MoE / 1=attn / 2=verify）。判据（`ar-l4l5-optimization-design.md` §4-A0）：
   - `avg_spin` **≫** 31k 周期 ⇒ 78.3µs 主要是**追踪放大** ⇒ **R3 的靶子缩小，甚至整个 AR 方向缩小**；
   - `avg_spin` **≈** 31k ⇒ 等待是真的 ⇒ 走 A2c（rank 负重）/ R2；
   - `avg_spin` **≪** 31k ⇒ AR 已无肉 ⇒ **R3 直接终止**。
2. **`ferrite_pdl_exp` 的两个新 mode**（`ferrite_kernels.cu:8560` 旁加，**纯微基准，无生产影响**）——
   这是 R3 的**判决实验**：
   - **`mode 4`（"primary 长 spin + secondary 重 prologue"）**：primary = 一个**长 `__nanosleep` 的 spin kernel**
     （模拟 pubred），secondary = 一个**带重 prologue + 入口 GDS** 的 kernel。
     量：`t_normal` vs `t_pdl` ⇒ **直接读出"prologue 能不能藏进 spin"以及能藏多少**。
     **这是 R3-B 成立与否的唯一硬证据。**
   - **`mode 5`（"primary 调 PTLC + secondary 是 plain launch"）**：验证 **PTLC 在无 secondary 时是 no-op**
     （R3 的默认路径行为不变）。
3. **`mode 3` 复验**（graph-captured PDL）在**当前节点**上再跑一次（B300/sm_103a 的 driver 版本）。

### 5.1 R3-A（0.5 人日）

- 改动：`ferrite_ar_pdl_or_plain` + 6 个 launcher 的 pubred 那一发 + `DSV41_AR_PDL` gate。
- 验收：`DSV41_AR_PDL=0/1` 图 A/B；**四段文本逐字节相同**（主）+ e2e p50（次，>2% 才动默认）；
  **`nsys` 只读计数、不读 ms**（v5 自旋在 nsys 下放大 ~300×，项目铁律）。
- **不需要** parity（不改数据流）。

### 5.2 R3-B（1–1.5 人日）

- 前置：5.0-2 的 `mode 4` 必须给出 **≥ 3µs** 的可藏窗口，否则**放弃 R3-B**。
- 改动：3 个 pubred kernel 的入口 PTLC（gated）+ §3.2 表里**逐 site 审计过**的后继 attr+GDS。
- 验收（三层）：
  1. **位级**：`DSV41_AR_PDL=0/1` 的 `[toktr]` md5 相同（`DSV41_TOKTRACE=1`，段外打印、安全）；
  2. **探针**：`[ar-probe]` 的 `avg_epi` **不变**（工作地板不该被 PDL 改变）、`avg_spin` 允许变；
  3. **图**：capture 成功、无 instantiate 失败、`[graph]` 快照的节点数不变（PDL 不加节点）。
- **失败即回退**：`DSV41_AR_PDL` 未设 ⇒ 与今天逐字节等价（plain launch）。

### 5.3 R3-C（1–2 人日，**仅在 R3-B 实测窗口 ≥10µs 时**）

- 前置：`mode 4` 证明 spin 里确有**空闲**窗口（不是被 memory system 自己占着）。
- 改动：新 filler kernel（**不调 GDS**）+ 网格上限 env；只对 1–2 个 site 先开。
- 验收：`DSV41_AR_FILLER=0/1` A/B；**文本逐字节相同**（纯预取）；e2e p50 **必须为正收益**，
  否则照 `W2_PREWARM` 的先例**直接关掉、不留门**（项目纪律：`e337b1e`"failed experiment code
  should be removed immediately, not left gated-off in the tree"）。

---

## 6. 红线（违反任一条 ⇒ 立即退回，不讨论）

| # | 红线 | 来自 |
|---|---|---|
| **R-1** | **不折 store、不动 reduce 的升序 rank、不改 epoch 语义** —— R3 只改**时序** | A1a 尸检（`ar-step2-a1a-fix-design.md` §2.3） |
| **R-2** | **filler 不得读 AR 的输出 / `epoch` / `ready_*` / `staging_*`** | §3.3；PTLC 后 AR 的写对未过 GDS 者不可见 |
| **R-3** | **不在两个 AR 之间放 PDL 边**（epoch 邻接契约） | `ferrite_kernels.cu:9577-9579`、R1 bug 2 |
| **R-4** | **GDS 必须无条件在核入口、`__CUDA_ARCH__ >= 900` 内** | 三份 PDL helper 的 CONTRACT |
| **R-5** | **gate 进程读一次**（capture 与 replay 必须同一臂） | `ar-step1` §3 |
| **R-6** | **PTLC 必须是 gate + 架构条件，不能带 `blockIdx` 条件**（所有 block 都要执行） | PDL 语义（§2.1） |

## 7. 风险登记

| 风险 | 概率 | 应对 / 回退 |
|---|---|---|
| 把 PTLC 当"输出可见"用 ⇒ **静默错数** | 中 | R-2 + §5.2 的位级验收；filler 默认 OFF |
| AR 在图里的后继是 memcpy/event ⇒ PDL 边非法 | 中 | §3.4 先 dump 后继节点类型；instantiate 失败即回退 |
| **PDL 重叠变 SM 争抢**（w2-prewarm 重演，+0.04ms） | 中 | filler 小网格 + 入口护栏 + 默认 OFF + "负收益就删" |
| `mode 4` 证明窗口 ≈ 0（spin 里没有真空闲） | **中-高** | **R3-B/C 直接终止**，只保留 R3-A（~0.05ms，几乎无感） |
| 与 A2b 的 PARK 交互：AR park 永不返回 ⇒ 后继永久 GDS 等待 ⇒ 整图 wedge | 中 | **预期行为**（响亮失败），但 watchdog 文案要写清"由 AR park 引起"，别误判成 PDL 卡死 |
| **A0 证明 78.3µs 是 nsys 伪影** | 中 | 整个 AR 方向的预算重估；R3 随之作废（这正是 5.0-1 排在最前的原因） |

## 8. 输出物与分工

- **本文件** = R3 的设计（换行率、命令、判定表）。
- 实施顺序：`5.0（探针+mode4/5，0.5 人日）` → `5.1（R3-A，0.5）` → `5.2（R3-B，1–1.5）` → `5.3（R3-C，1–2，条件）`。
- 与 R1/R2 的协调：R3 **不与 R1 争文件**（R1 只碰 `ar5_wait_round` 的臂选择）；
  R3 会碰**同一函数**（PTLC 埋在`ar5_wait_round` 的调用者的入口）⇒ **R1 合入后再做 R3-B**，避免同函数冲突。
- R2 落地后**必须重定标** R3 的收益（轮数 84→44，per-round 收益不变 ⇒ 总量减半）。

## 9. 诚实边界（必须写在账上）

1. **本机无 GPU、无 nvcc**：全部 µs 都是**设计口径**，来源逐条标注（代码事实 / 历史账本 / 代数）。
2. **77.7µs 未证实**：`0.6µs` 的算法漏了 8 peer × 120KB 的远程写与 reduce 的 8×120KB 读，
   且 store 是独立 kernel 行、不在那张 top-6 表内（`swallow-ar-first-step-design.md` §0 第 4 条已判过）。
   **R3 的可行性判定必须先过 A0。**
3. **任务前提需修正**：本题"PDL 让 pubred 的 77.7µs 与其他 kernel 重叠"——
   **对"spin 主体"不成立**（§2.2）；成立的只是"后继 pre-GDS prologue + AR 尾"。
4. **R3 的量级（0.17–0.4ms/步）比 R2（−3.3ms）小一个数量级**。若目标是"AR 不再是 #1"，
   优先级应是 **R2 > A0 归因 > R1 > R3**。
5. `ferrite_pdl_exp` 的 `mode 3` 是**本项目**验证过 capture 的唯一证据，且在 GLM 的 TU 上；
   AR 的 TU 需要**同样的模式**（新 helper 是第 4 份副本，语义逐条同义，但**没在 AR 上跑过**）。

---

*工部 · 本文件为唯一产出；未改任何源码、未改任何 gate 默认、未执行任何 GPU 命令。*
*代码行号以当前工作树为准；引用以函数名为准（有 peer 正在改 `chain_dev.rs`，行号会漂）。*
