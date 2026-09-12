# AR（TP8 all-reduce）的 L4/L5 优化具体设计

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 代码基线：HEAD `e3622fc`（`kernels/cuda/ferrite_kernels.cu`、`crates/ferrite-models/src/dsv41/{tp.rs,chain_dev.rs}`、
> `crates/ferrite-dsv41/src/serve.rs`）；引用一律 `file:line`。
> 输入数据：最终干净栈（91.1）nsys 表（`dspark-correctness-chain.md:5485-5510`，commit `32e15cc4` 记录）、
> 无图全可见 profile（`781feec7`，`:2977-3013`）、图外可见 profile（`7e1516bd`，`:2953-2975`）、
> `ar-further-optimization.md`、`lazy-verify-optimization-path.md`、`batched-12-5ms-requirements.md`。

---

## 0. 先读：三条校准（不校准会优化错东西）

### 0.1 「13.2% → 27.1%」不是同一个 kernel，也不是同一个分母

| profile | 可见口径 | 排名 #1 的 AR kernel | 实例 | 总时间 | 平均 | 可见总量 |
|---|---|---|---:|---:|---:|---:|
| `781feec7`（**无图**，全可见） | 全栈 | `p2p_ar_pubred_v5_hcpost_**rows**`（verify 多行） | 7,136 | 204.4ms | 28.6μs | ≈1,528ms |
| `7e1516bd`（VERIFY_GRAPH，可见=图外） | 图外 | AR 未进 top-6（≤5.0% ⇒ ≤7.0ms） | ? | ≤7.0ms | — | 139ms |
| `32e15cc4`（干净栈） | 图外 | `p2p_ar_pubred_v5_**hcpost**`（**decode 单行**） | 933 | 49.8ms | 53.4μs | **184ms** |

三条硬事实：

1. **13.2% 是 verify 的多行 AR（`_hcpost_rows`），27.1% 是 decode 的单行 AR（`_hcpost`）**——两个不同的
   kernel（`ferrite_kernels.cu:9373` vs `:9503`）。把它们并列成"AR 翻倍"是**跨 kernel 比较**。
2. **两个 profile 的可见分母差 8.3×（1,528ms → 184ms）**，而这两个 profile 的可见 kernel 集合不同：
   `32e15cc4` 的 top-6 里**没有** `gemm_fp8_mrows_kernel<1>`（`781feec7` 的 #2，196.1ms）、
   没有 `gemm_fp8_sh_exp_pair_kernel<1>`（#3，150.9ms）、也没有 `_hcpost_rows`（#1，204.4ms）
   ⇒ 干净栈这一 profile 里 **verify 整个在捕获图里（kern_sum 看不见）**，可见窗口基本只剩图外的 decode 侧 kernel。
3. **AR 的绝对时间没有变大。** 同名 decode kernel 的历史值：`781feec7` 里 933 例 = **57.3ms**（见 0.2 的列校验），
   干净栈 933 例 = **49.8ms** ⇒ **−13%（略快）**。

⇒ **结论：27.1% 是"可见分母塌缩"的产物，不是 AR 变慢。** 任务的"从 13.2% 翻倍"不成立。

### 0.2 一处列校验（历史表的 AR 两行有 ×10 笔误）

`781feec7` 的 top-10 表里 **8 行 × 例数 × 平均 = 总时间** 完全闭合（如 #1: 7,136×28.6μs = 204.1ms ✓、
#2: 14,820×13.2 = 195.6 ✓ … #7: 5,448×15.0 = 81.6 ✓），**只有 AR 的两行不闭合**：

| 行 | kernel | 例数 | 总时间(ms) | 平均列 | 例数×平均 |
|---|---|---:|---:|---:|---:|
| #8 | `p2p_ar_pubred_v5_hcpost` | 933 | 57.3 | 6.1μs | 5.7ms ✗ |
| #9 | `p2p_ar_store_v5` | 957 | 49.2 | 5.1μs | 4.9ms ✗ |

两行的 `平均` 列都比 `总时间/例数` 小 **10 倍**：57.3/933 = **61.4μs**，49.2/957 = **51.4μs**。
**本文一律以 `总时间` 列为准**（它与全表其它 8 行的内部一致性证明它是那两行里可靠的一列）。
⇒ decode AR：**61.4μs（无图）→ 53.4μs（干净栈）**。若反过来采信 `平均` 列，会得到"6.1μs → 53.4μs，AR 慢了 8.7 倍"
的结论，而这个结论会被同表自身否掉（同负载的 `_hcpost_rows` 28.6μs 无法与 6.1μs 共存）。
**A0 探针（§4-A0）是这个分歧的唯一终审。**

### 0.3 nsys 的第三个失真：自旋被追踪放大

`scripts/nsys_wave1.sh:33-40` 自陈：nsys 逐节点追踪下 v5 的 publish 自旋被放大（实测 **240s / 69 步**）。
账本口径的 v5 轮成本是 **17.3μs**（`lazy-verify-optimization-path.md:98`：AR 1.40ms/步 ÷ 80 轮）。

⇒ **同一个 v5 轮有三个数**：工作地板 **~5μs** / 账本 **17.3μs** / nsys **53.4μs**。
**本设计一律以 μs/轮 为单位表达收益**，并在需要时并列口径；"% 占比"只作参考，因为它的分母本身受图与追踪影响。

### 0.4 AR 仍然是最大目标——理由与 "%" 无关

三条与口径无关的事实：

1. **每轮的工作只有 ~5μs**（§2），其余全是"非工作量"（等待 + 协议）；
2. **轮数是按层数的固定税**：decode **80 轮/步**、lazy verify **80 轮/行**；
3. **lazy 把 per-step 族乘 k_emit**：AR 每步付 `80 × k_emit` 轮（`batched-12-5ms-requirements.md:95` 明确列 AR v5 为 per-step 族、lazy 付 `160 发 × k_emit`）。

⇒ L4/L5 的靶子是**每一轮的等待**，以及**轮数 × 等待**这个乘积里的**轮数因子**（只有臂选择能动，§3）。

---

## 1. 实现现状（代码地图 + 一轮的时序）

### 1.1 组件

| 组件 | 位置 |
|---|---|
| store 内核（peer 并行写） | `ferrite_kernels.cu:8854` `p2p_ar_store_v5_kernel`（`gridDim.y = world`；ADD_EPI `bias`） |
| pubred（decode 单行） | `:9037` `p2p_ar_pubred_v5_kernel` |
| **pubred + hc_post（decode 折叠）← 本次 #1** | `:9373` `p2p_ar_pubred_v5_hcpost_kernel` |
| pubred + hc_post（verify 多行） | `:9503` `p2p_ar_pubred_v5_hcpost_rows_kernel` |
| 等待（三变体共用） | `:8964` `ar5_wait_round`（A4 单块轮询臂 / OFF 惊群臂） |
| A1 探针 | `:8947-8958` `g_ar5_probe_*` + `[ar-probe]` |
| epoch pad（SWALLOW 对齐） | `:9129` `dsv41_v5_epoch_pad_kernel` |
| block shape | `:9155` `ferrite_ar_v5_block_threads`（world ≤ 64 ⇒ **64**）、`:9159` grid = `ceil(n4/64)` |
| Rust 入口 | `tp.rs:549` `_hcpost`、`:711` `_hcpost_rows`、`:641` `_hcpost_add`、`:508` `_pubred_only`、`:1027` `ar_v5()`、`:1002` `end_round()` |
| **decode 调用点（2 处/层）** | `chain_dev.rs:15371`（attention，fold）+ `:14357` → `:5369` `moe_reduce`（MoE） |
| verify 调用点（2 处/层/行） | `chain_dev.rs:11538`（attention rows）+ `:13377`（MoE rows） |
| staging | `serve.rs:401` `ar_bytes = max(hc_mult·dim = 20,480, VERIFY_ROWS·dim = 30,720)×4 = 120KB/槽`；`tp.rs:498` `slot_stride_elems = bytes/4 = 30,720` |
| lazy 行循环 | `chain_dev.rs:9030` `lazy_run_row`（`step_rows_sync(&rows_in[i..=i])` = 每行一次 forward） |

### 1.2 一轮（v5 round）的时序 — 代码事实

```
① store      : world 个 peer 各写 n×4B（gridDim.y=world，每线程 1 次 16B 远程 store）  :8879-8893
② stamp      : block 0 的 thread r → peer r 的 ready 槽 atomicExch_system(e+1) + threadfence_system  :9389-9392
③ epoch++    : *epoch = e+1（下一轮 store 的 parity/轮号来源）                          :9394-9395
④ poll       : ar5_wait_round                                                          :8964-9035
               OFF 臂 = 每个 block 的 8 个线程都轮询（n4/64 = 20 块 ⇒ 160 轮询者/轮，惊群）
               A4  臂 = block 0 轮询 + 设备内广播字（epoch+1）
⑤ reduce     : 1280 个 float4 / 1280 线程（正好 1 迭代/线程）；升序 r=0..7 load+add    :9401-9410
⑥ fold       : ar5_hc_post_col4（hc_n 行 × (1 __fmul_rn + 4 __fmaf_rn)/列），写 res      :9317-9371
```

**关键：每一轮都是一次硬同步。** AR 的输出是下一段计算的输入（`o`→hc_post→`h`→下一层），
所以第 ④ 步的等待**不可用别的计算掩盖**（这也是 `ar-further-optimization.md` B5 把 PDL 判死的原因——
但那个判断对"prologue 级重叠"过强，见 §4-A1b）。

---

## 2. 「933 × 53.4μs」的分解（回答任务问题 1）

以 n = 5,120 f32（**20KB payload**）、world = 8、grid = 64×20 = 1,280 线程 = n4 为单位：

| 阶段 | 实际工作 | 设计口径 | 依据 |
|---|---|---:|---|
| store（独立 kernel；#1 表里不含它） | 160KB 远程写（8 peer × 20KB）+ 20KB 读 | 2–5μs | `:8861-8868`（串行旧版实测 5.9μs，peer 并行后降） |
| stamp | 8 次 remote atomic + 1 fence，**并行**发 | 0.2–0.5μs | `:9056-9064`（串行版 0.4–0.8μs） |
| **poll / wait** | 等**最慢** peer 进入本轮 | **46μs（nsys 口径）/ ~12μs（账本口径）** | `:8964-9035`；账本 17.3μs − 工作 5μs |
| reduce | 8×float4 load + 12 add / 线程，1 迭代 | 1–2μs | 1280 线程恰好覆盖 n4 |
| hc_post epilogue | 4 行 × (1 mul + 4 fma)/列；读+写 `res` 各 80KB | 1–2μs | `:9317-9371`（layout `[hc_n=4][hc_h=5120]`） |
| launch / 边界 / tail | 一次 kernel 边界 + 20 块的尾巴 | 1μs | — |
| **工作地板合计** | | **≈ 4–6μs** | 与 `781feec7` 里该 kernel 的 6.1μs 读数同量级 |

⇒ **非工作量占 90%（nsys 口径）或 ~70%（账本口径）**。两部分（工作 / 等待）都必须由 A0 探针拆开。

**带宽核对（用于关掉"数据量"方向）**：

- 每轮远程写 = 8 × 20KB = **160KB**；decode 80 轮/步 ⇒ **12.8MB/步/rank**；
  在数百 GB/s 的每 rank NVLink 有效带宽下 ⇒ **~30–60μs/步**（≈ 步时的 0.2–0.3%）。
- 每轮本地（staging）读 8×20KB + `out` 写 20KB + `res` 读/写各 80KB ⇒ ~340KB，全部 L2 级。
- **⇒ 数据量不是瓶颈**，bf16/fp8 payload 的收益上限是每步几十 μs，而风险是破坏"NCCL 升序逐位等价"契约
  （AR 结果直接进 residual stream）。**明确不做**（§4-C）。

---

## 3. 轮次账（回答任务问题 2）

### 3.1 每步多少轮

| 路径 | 轮/单位 | 推算 | 依据 |
|---|---|---|---|
| decode 步 | **80 轮/步** | 2 处/层 × 40 层 | `:15371` + `:14357`；`ferrite_kernels.cu:9098-9104`（"swallowed 84 轮 vs aligned 165 轮/步"） |
| verify（lazy，m=1/行） | **80 轮/行**（+1 argmax = 81/行） | 2 处/层 × 40 层 × 1 行 | `:11538` + `:13377`；`swallow-fix9-round-ledger-design.md:201`（`3 + 81·k_emit`） |
| verify（batched，m=6 块） | **80 轮/步**（payload = m·dim，轮数不变） | 2 处/层 × 40 层 | `batched-12-5ms-requirements.md:95`（AR v5 是 per-step 族） |
| lazy 步足迹 | `84 + 81·k_emit` | k_emit=2.214 ⇒ **263 轮/步**；k_emit=6 ⇒ **570 轮/步** | 同上 |
| batched 步足迹 | **165 轮/步**（与 k_emit 无关） | `dspark-correctness-chain:5478` | — |

### 3.2 可减性（逐条给判定，含代码理由）

1. **同层的 attn + MoE 两轮不可合并。** attn AR 的输出（`o` → `hc_post` → `h`）是 MoE 的输入
   （`:15371` → `:14357`），中间隔着整个 MoE。
2. **「两行一批」不可行（否决任务问题 2 的设问）。** row i+1 的输入 = row i 的 argmax：
   `lazy_run_row`（`:9030-9065`）每行调 `step_rows_sync(&rows_in[i..=i], …)`，即**行间是串行的早退链**，
   "固定配对"在拓扑上不存在。强行配对 ⇒ 行数从 k_emit 推到 `DSPARK_DRAFTS+1 = 6`
   （**行数 2.214 → 4.0，+1.79 行/步**）而**每行照付 80 轮** ⇒ 纯亏。
   （同一结论已在 `ar-further-optimization.md §4-B2` 得出，本次由 `:9030-9065` 重新确认。）
3. **「部分 AR 延迟到 commit」不可行。** AR 的结果是下一层的输入；延迟意味着在残差链里保存
   8 个 rank 的部分和并让下游读未求和值 ⇒ 等于重写整条残差链，收益为负。
4. **每轮的 2 发 → 1 发是唯一"不改轮数但改每轮成本"的机械项**：store 折进 producer epilogue
   （`AR_STORE_FUSE`，`:2202` gate；attn 侧 `gemm_fp8_mx_ar` 已就位 `:15332`，MoE 侧待做）⇒ 少一次 launch + 一次 kernel 边界。
5. **唯一真正的"轮数杠杆" = 回 batched 臂**：`80/行 × k_emit` → `80/步`，
   k_emit=2.214 时 **−65%**，k_emit=6 时 **−71%**。这不是 AR kernel 工作。

> **§3 的核心结论：AR 的轮数在 lazy 臂里是结构性不可减的（80 轮/行 × 行数）。**
> 能动的只有（a）每轮的固定开销、（b）每轮的等待、（c）臂选择。

---

## 4. 设计（每项：改动 / 预期 / 成本 / 风险 / 验收）

### A0 — 探针（**必做第一步**，不改默认路径）

**为什么必须有**：同一个 v5 轮有三个数（5 / 17.3 / 53.4μs，§0.3），差 10 倍；三个优化方向
（减固定项 / 减等待 / 减轮数）对应的预算完全不同。**没有这个数，所有设计都是猜。**

改动（`ferrite_kernels.cu`，全部 gated）：

1. **site 标签**：`ar5_probe_report(spin, site)`，两组计数器（site = 0 图外 MoE AR / 1 图内 attn AR / 2 verify rows）
   ⇒ 回答"80 轮里哪一半在等"（这一半决定 A 类改动的靶子）。
2. **每 rank 打印**：`printf("[ar-probe] rank=%d n=%llu avg=%llu max=%llu", …)`（现在没有 rank 字段）
   ⇒ 若**某个 rank 的 avg_spin 系统性为 0** 而其它 rank 都大 ⇒ 瓶颈是"最后一个到达者"（走 A2c）；
   若**所有 rank 的 avg_spin 都大** ⇒ 是同步开销本身（走 A2d / A1）。
3. **轮内时间三分解**（可选，成本 +0.5 人日）：在 store 结束 / stamp 后 / poll 后各取一次 `clock64()`
   ⇒ 直接得到 store / stamp 传播 / poll 三段，替掉 §2 的设计口径。

运行（**关键：不要在 nsys 下跑**）：`DSV41_AR_PROBE=1` + 生产 gate 集，直接 `./target/release/ferrite-serve`，
读 stdout 的 `[ar-probe]`（单位 = SM 周期，B300 ~1.8GHz ⇒ 17.3μs ≈ **31k 周期**）。

- 成本 **0.5 人日 + 1 GPU**；风险 0（默认 OFF，不碰默认路径）。
- **判据**：`avg_spin` vs 31k 周期（账本）：
  - `avg_spin ≫ 31k` ⇒ nsys 的 53.4μs 主要是追踪放大，AR 的真实占比 <15% ⇒ **L4/L5 应转投 MoE/投影**（这是本设计最重要的一个可能结论）；
  - `avg_spin ≈ 31k` ⇒ 账本口径成立，等待是主项 ⇒ 走 A2；
  - `avg_spin ≪ 31k`（接近工作地板）⇒ AR 已无肉。

### A1 — 缩短每轮的关键路径（固定项，与"等待"无关）

**(a) store 折进 producer（= `AR_STORE_FUSE`）** — 每轮 2 发 → 1 发。
- 改动：MoE 侧 producer（`expert_gemv_fp4_down_reduce` / `add_inplace`）的 epilogue 携带 staging 写；
  attn 侧已就位（`gemm_fp8_mx_ar`，`:15332`；decode 侧补丁在工作区 `ar-fuse-store/ar-v5-store-epilogue.patch`；
  gate `:2202`）。
- 预期：**−1~2μs/轮** ⇒ decode **−0.08~0.16ms/步**；lazy 再 ×(1+k_emit) ⇒ **−0.25~0.5ms/步**。
- 成本 2–3 人日 + 双 A/B；风险**中高**：载体的 last-writer 论证随 rank 变，选错 ⇒ 静默错数；
  GEMV 签名变更 ⇒ ptxas 漂移（补丁 README §5 的双产物 A/B）。
- 验收：`DSV41_AR_ST=0` 旧 .so vs 新 .so **逐 token 一致**；新 .so 上 `AR_ST=1/0` 一致。

**(b) PDL：把"后继的 prologue"藏进 AR 的尾巴** — 修正 `ar-further-optimization.md` B5 的过强否定。
- 论证：B5 说"AR 的后继都读 `h_r`，无合法重叠对"——这对**整体重叠**成立，对 **prologue 重叠**不成立：
  `cudaGridDependencySynchronize()` **之前**的地址计算、TMA 描述符、独立的权重预取都不读 `h_r`。
- 改动：AR 的后继 launch 带 `programmaticStreamSerializationAllowed`（`:8409` 已有为此写的 mode 2/3 实验），
  后继入口加 `cudaGridDependencySynchronize()`。
- 预期：**−1~3μs/轮** ⇒ decode **−0.08~0.24ms/步**。
- 成本 2–3 人日；风险中（**capture 下 PDL 属性是否存活**必须先跑 `ferrite_pdl_exp` 的 mode 3 实测）。
- 验收：先 `ferrite_pdl_exp` mode 2 vs 3；再 e2e 逐 token 一致。

### A2 — 减少"等待"本身

**(a) A4 单块轮询：保持 OFF，不加投资。**
A4 已实施（`:8897-8968`），端到端 A/B **中性**（82.9 → 82.6 tok/s，`dspark-correctness-chain:3426`）
⇒ 惊群（160 轮询者/轮）**不是**主导项。除非 A0 显示 `avg_spin` 很大**且**与 `ceil(n4/64)` 相关，否则不动。

**(b) A4 的超时语义是"真 bug"，且**两臂都有**（安全项，非性能项）——建议最先做。**
代码事实（`:8970-9009` OFF 臂、`:8968-8992` A4 臂）：
两个臂都在 `spins > 5,000,000`（~0.5s）时 `printf("[ar5-hang]")` 然后 **`break`**，随后**照常推进**：
- OFF 臂：`break` → `__syncthreads()` → **直接 reduce**（读 peer 从未 publish 的 staging）；
- A4 臂：`break` → block 0 **仍然**执行 `*(volatile unsigned*)(epoch+1) = e+1`（**把超时当成功广播**）→ 其它块照常 reduce。

⇒ 这不是"停住"，是**静默错数**：epoch 一旦裂开（D1 的 epoch 54 / gap=27 系列），AR 会带着错误数据继续跑。
它与 SWALLOW 臂"6 个 token 后 EOS"的观测链一致（`dspark-correctness-chain:5505`）。
- 改动（二选一，均在 `ar5_wait_round`）：①超时后**不推进 epoch**、设 device 侧 poison 计数，由 host watchdog 报错退出；
  ②超时后 `printf` 并**永久 spin**（保留诊断，杜绝错数）。**绝不允许超时后 publish**。
- 预期性能影响 **0**；风险收益：把"静默错数"变成"响亮的失败"——**这是所有 AR A/B 的前提**（否则"干净"无法证明）。
- 成本 **0.5 人日**；验收：人为制造 epoch 失衡时进程必须报错，而不是输出 token。

**(c) rank 不对称 ⇒ 负重平衡（若 A0 显示某 rank 系统性晚到）。**
候选来源：head/argmax 集中在 rank 0、engram 层、compressor 的 host 往返。
- 改动：把该 rank 的额外工作移出关键路径（**改的是工作分配，不是 AR**）。
- 预期：wait 直接下降"落后量" ⇒ 若落后 5μs，80 轮/步 = **−0.4ms/步**。
- 成本 1–3 人日（取决于 A0 的分布）；风险低（不碰协议）。

**(d) drift 的结构性削减：把 MoE AR 纳入捕获段。**
- 代码依据：`moe_reduce` 的 AR 位于捕获段**外**（`:5365-5368`"a CUDA graph cannot contain the host barrier"），
  但 **`ar_v5()` 下 AR 已经没有 host barrier**（`end_round` 直接 return，`tp.rs:1002-1007`），
  且 `moe_add_in[layer]` 已在**捕获时刻**读取并烘进图（`:5371-5375`）⇒ 段内无主机调用。
- 预期：40 轮/步 从"host 驱动、暴露 jitter"变成"图 replay、lockstep" ⇒ **−0.2~0.5ms/步**；
  这是**唯一能结构性消掉等待**的改动。
- 成本 3–5 人日；风险中（capture 合法性：段内不得有 cuda 主机调用/分配；`OnceLock` 已安全）。
- 验收：图回放 vs direct 逐 token 一致；`[verify_graph] captured` 出现。
  **⚠️ 副作用**：这会让本 profile 的 "#1 27.1%" 直接变成"不可见" ⇒ **必须同时用 `cuda_gpu_trace` 口径复测**，
  否则会误判为"AR 消失了"。

### A3 — 轮数（结论：lazy 不可减，见 §3.2）→ 转 B

### B — 结构：臂选择（最大的单项，非 AR kernel 工作）

| 臂 | AR 轮/步 | k_emit=2.214（17.3μs/轮） | k_emit=6（accept=5） |
|---|---:|---:|---:|
| lazy | `84 + 81·k_emit`（263 / 570） | **3.1ms/步**（实测账本 1.40 × k_emit） | **9.9ms/步** |
| batched | **165（固定）** | **1.40ms/步** | **2.9ms/步** |
| 差 | −65% / −71% | **−1.7ms/步** | **−7.0ms/步** |

- **关键洞察：accept 越高，lazy 的 AR 越贵（线性 ∝ k_emit），batched 的 AR 与 accept 无关。**
  ⇒ 任何"提高 accept / 让 draft 更准"的工作都会**把 AR 相对推成 #1**——这正是干净栈里看到的形态
  （其它族变快 + accept 变高 ⇒ AR 占比上升，**即使 AR 的绝对时间没变**，§0.1）。
- 成本：SWALLOW/step_dev 的修复（进行中，11 次修复的教训）；风险：epoch/arm 边界
  （缓解：A2b 的"响亮失败"+ `epoch_max` 动态 pad）。
- **这是能让 AR 彻底离开 #1 的唯一路径。** 诚实地说：**AR 是不是 #1，最终由臂决定，不是由 AR kernel 决定。**

### C — 数据量（明确不做）

bf16/fp8 payload：每轮 160KB 远程写减半，但整步只有 ~30–60μs 级（§2），
且破坏"NCCL 升序逐位等价"契约（AR 结果直接进 residual stream，±4e-3 相对误差影响 accept）。
**判定：不做**（与 `ar-further-optimization.md §4-B4` 同结论，本次由 §2 的带宽账再次确认）。

### D — kernel 微优化（低收益；仅当 A0 显示工作占比可观时做）

| # | 项 | 结论 |
|---|---|---|
| D1 | block shape `64 × ceil(n4/64)` = `64×20` | **已最优**：1,280 线程 = n4，1 迭代/线程，铺在 20 个 SM；n=5,120 时无尾波（`:9155-9162`） |
| D2 | stamp 的 8 次 `atomicExch_system` | 必须跨设备可见，scope 不可降；可与 store 的最后一个 block 融合（= A1a 的另一半） |
| D3 | hc_post epilogue 的 4 行写（80KB，且 k 循环读旧 `res`） | **保留**——逐位等价要求（`:9365-9368`） |
| D4 | reduce 加 `__ldg` / `ld.global.nc` | 无收益（全 L2 命中） |

---

## 5. 执行顺序与落点

```
1) A2b 超时语义（0.5 人日）        ← 先消除静默错数；否则后面所有 A/B 不可信
2) A0  探针（非 nsys，0.5 人日）   ← 唯一能定预算的实验
   ├─ avg_spin ≫ 31k 周期 ⇒ nsys 失真为主 ⇒ 转投 MoE(L4-3/4)/投影(L4-1)，AR 只做 A2b/A1a
   ├─ avg_spin ≈ 31k      ⇒ A2c(rank 负重) + A2d(MoE AR 入图)
   └─ avg_spin ≪ 31k      ⇒ AR 无肉，停止 AR 投入
3) A1a store 折进 producer（2-3 人日）  ← 与 wait 无关的固定项
4) A1b PDL（2-3 人日，先跑 mode 3 实测）
5) B  臂选择（SWALLOW）                 ← AR 离开 #1 的唯一结构性路径
```

**落点（两口径并列，避免混淆）**

| 阶段 | nsys 可见口径 | 账本 ms/步（lazy, k_emit=2.214） |
|---|---:|---:|
| 现状 | 27.1%（53.4μs/轮） | 3.1（"三件套"口径） |
| +A0 +A2b | 不变（安全项） | 3.1 |
| +A2c/A2d | 27.1% → **~18-20%** | 2.2–2.7 |
| +A1a/A1b | → **~15–18%** | 1.9–2.4 |
| 地板（纯工作 5μs/轮） | → **~4–6%** | 0.4 |
| +B（batched） | **不再是 #1** | 1.40（与 k_emit 无关） |

> **回答"27.1% → 目标 15%？"**：可达，**但 15% 这个数字本身意义有限**（分母受图与追踪影响）。
> 真实收益应表达为 **wait 从 ~12μs/轮 压到 5–7μs/轮**（账本 3.1 → ~1.8ms/步，**−1.3ms/步 ≈ −6% e2e**）。
> 若只能投一件：**投 B（臂）**，它的量级是 A 类全部的 3–5 倍。

---

## 6. 诚实边界

1. **本机无 GPU**：所有 μs/ms 都标了来源（代码事实 / 历史 profile / 账本推算）。A0/A1/A2c/A2d 的数值是**设计口径**。
2. **任务前提需修正**：AR 没有"绝对翻倍"（§0.1、§0.2）；翻倍的是它在**可见分母**里的份额。
   27.1% 也不能直接当作真实运行占比（§0.3 的追踪放大）。
3. **历史 profile 的 AR 两行有 ×10 列笔误**（§0.2）；本文所有推算以 `总时间` 列为准并已显式标注。
4. **A4 的"中性"是端到端 A/B（噪声内）**，不是 AR 级 A/B ⇒ 不能完全排除惊群，但足以说明它非主导。
5. **A2b 是代码事实**（两臂 `break` 后继续/照常广播，`:8970-9009`），不是推测；但它只在 epoch 裂开时触发，
   所以"生产路径静默错数"是**条件性**的。
6. **D1/D2 两个历史未解问题仍会影响账**（v5 为何在 `AR_V5=0` 下仍运行；verify 侧 store 计数），
   本文未依赖它们；但 A1a 的收益以"MoE 侧 store 尚未折进 producer"为前提——D2 若已折，A1a 的 MoE 半为 0。
7. **§3 的轮数（80/步、80/行）与实例数（933）之间存在一处口径张力**：933/40 层 ÷ 2 处 = 11.7 步，
   而 7e1516bd 记录的计数任务是 ~24 spec 步。两种读法（11.7 步 × 2 处/层 或 23.3 步 × 1 处/层）都不改变
   §3/§4 的任何结论（**只影响 µs/轮 的分母**，不影响轮数与改动清单），故未深挖；A0 的 site 标签会顺带判定它。

---

*工部 · 只读分析 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*代码行号以 HEAD `e3622fc` 为准；读代码时以函数名为准。*
