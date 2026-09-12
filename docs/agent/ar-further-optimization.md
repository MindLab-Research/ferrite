# AR（TP8 all-reduce）的进一步优化分析

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 代码基线：HEAD（`crates/ferrite-models/src/dsv41/{tp.rs,chain_dev.rs,device.rs}`、
> `kernels/cuda/ferrite_kernels.cu`、`scripts/nsys_wave1.sh`），引用一律 `file:line`。
> 输入数据：nsys **无图** per-kernel 表（`docs/agent/dspark-correctness-chain.md` 的
> 「nsys 无图完整 per-kernel 数据」节，commit `0bcb1e6` 记录）。

---

## 0. 判决（先读七条）

1. **AR 三件套的 20.8% 里，主导项不是"搬了多少字节"，而是"等了多久"。**
   最强证据是一对**同形状同负载**的 kernel：`_hcpost_rows`（28.6µs）与 `_hcpost`（6.1µs）。
   lazy verify 恒 m=1，两者的 `n`、grid、reduce 循环、hc_post epilogue **逐项相同**
   ⇒ 22.5µs/次 的差值是"非工作量"（peer stamp 轮询 + host 侧 rank 漂移 + nsys 自旋放大）。
   ⇒ 4.7ms/步的 AR 里最多只有 ~0.7ms/步 是搬运**地板**。

2. **任务里的调用次数模型要修正一处**：lazy verify 是**每层 2 处 AR**（attn `wo_b` + MoE），
   不是 1 处。按 2 处算，7,136 恰好 = 40 层 × 2 处 × **89 行**，
   而 89/2.214 = **40 个 verify 步** ▸ `chain_dev.rs:10141`（attn 侧）、`:11798`（MoE 侧）。
   933 = 40 层 × **23.3 个 decode 步**（decode 每层只有 1 处走 `_hcpost`）。
   40.3 + 23.3 = **63.6 步 × 22.56ms ≈ 1435ms** ≈ 观测 kernel 总时间 1494ms（310.9/0.208）✓ 闭合。

3. **"13.2%"是全 run 均摊，被 decode 步稀释了。** rows AR 的真实占比：
   204.4ms / 40.3 个 verify 步 = **5.07ms/verify 步 = verify 步（18.11ms）的 28%**。
   每步付 **80 处 AR × k_emit 2.214 = 177 次**（这就是"k_emit 税"的准确形状）。

4. **数据量不是杠杆。** 每次 AR：store 8 peer × 20KB = **160KB 远程写**，reduce 8 × 20KB 本地读 + 20KB 写；
   每 verify 步 177 次 ⇒ 28MB 远程写/步 ⇒ 400GB/s 级 NVLink 上仅 **~70µs/步**（0.4%）。
   **bf16 传输**（减半）最多省 ~35µs/步，还要赌位等价 ⇒ **结论：不做**（详见 §4-B1）。

5. **v5 的 pubred 把 v3 明确消掉的"惊群"又引回来了**（可查的代码事实）：
   `p2p_ar_pubred_v5_kernel` 的 poll 是 `if (threadIdx.x < world)` 且**无 `blockIdx` 限定**
   ⇒ **每个 block 的 8 个线程都在轮询同样 8 个 flag**（20 块 = 160 个轮询者）；
   而 v3 的注释写着"The publish + wait on ONE block (8 pollers, **no thundering herd**)"
   （`ferrite_kernels.cu:8792-8794`）。这是一处**低风险、可测量**的微优化机会。

6. **可折叠的 epilogue 还剩一个真实缺口：shared expert 的 merge。**
   `all_reduce_inplace_hcpost_rows` **没有 `_add` 变体**（对照单行侧的 `hcpost_add`，`tp.rs:504`），
   所以 verify 路径是 `[shared pair] + [add_inplace_raw] + [AR store + AR pubred]` 4 发/层
   （`chain_dev.rs:12092`），而 decode 侧早已把 merge 折进 AR store（ADD_EPI/E18）。
   ⇒ 新入口 `_hcpost_rows_add`，**机械镜像既有 `_hcpost_add`**，−88 发/步 ≈ **−0.27ms/步**。

7. **先做诊断，再谈优化**（两件都便宜，且都会改变预算）：
   - **D1**：`DSV41_AR_V5=0` + `DSV41_GRAPH_STEP=0` 都钉住了，为什么 v5 kernel 仍在跑 7,136 次？
     `ar_v5() = GRAPH_STEP ∥ AR_V5`（`tp.rs:869-889`），两腿全 0 ⇒ 必为 false ⇒
     `all_reduce_inplace_hcpost_rows` 会走 `Ok(false)` 回落 `tp.rs` 的 host-barrier 路径
     （`ar_store`/`ar_reduce2`），**一个 v5 kernel 都不该出现**。⇒ 这份数据的 AR 口径存疑
     （脚本自己的头注还写了"nsys 下 v5 自旋被放大 ~300×"，`nsys_wave1.sh:33-40`）。
   - **D2**：`p2p_ar_store_v5` 只有 957 例，但 HEAD 的 rows 入口**每次都发 store**
     （`ferrite_kernels.cu:9381`）⇒ 期望 ≥7,136 例。少掉的 7,120 例要么说明
     profiled 的 `.so` 已带"store 由 producer 携带"补丁，要么说明报告缺项。
     **这一项决定 §4-A5 的 36ms 收益是否存在。**

---

## 1. 调用次数账（核对 + 修正）

### 1.1 修正后的模型

| 路径 | AR 处/层 | 层数 | 单位数 | 实例 | nsys | 判定 |
|---|---:|---:|---:|---:|---:|---|
| lazy verify（`layer_rows`，m=1/行） | **2** | 40 | **89.2 行** | 7,120 | 7,136 | ✓ |
| decode 主链（`layer()`，1 token） | 1（`_hcpost`） | 40 | **23.3 步** | 920 | 933 | ✓ |
| store（decode MoE 侧） | 1 | 40 | 23.3 步 | 920 | **957** | 期望 verify 侧另有 7,136 ⇒ 见 D2 |

- 89 行 / k_emit 2.214 = **40.3 verify 步**；40.3 + 23.3 = 63.6 步；
  63.6 × 22.56ms = **1435ms** vs kernel 总和 310.9/0.208 = **1494ms**（差 4%，记账内）。
- ⇒ **任务里的「40 层 × 2.2 行 × 78 步」把"行"和"步"混在一个乘式里**，正确写法是
  `40 层 × 2 处 × 行数`，行数 = k_emit × verify 步数。

### 1.2 每步的准确税负

```
每次 verify 步的 rows AR 次数 = 80 处/行 × 2.214 行/步 = 177 次
rows AR 时间/verify 步        = 204.4ms / 40.3 = 5.07ms（= verify 步 18.11ms 的 28%）
三件套合计/步（全 run 均摊）   = (204.4+57.3+49.2) / 63.6 = 4.89ms（= 22.56ms 的 21.7%，与 20.8% 一致）
```

**AR 地板的估算**（若"非工作量"全部消失，见 §3）：
`7136 × ~5µs + 933 × 6.1µs + 957 × 5.1µs ≈ 46.6ms = 3.1%` ⇒ **潜在可回收 17 个百分点
≈ 3.9ms/步**。这是上界，不是预测——必须先由 §5-D3 探针把"真等待 / 真搬运 / 探针伪影"分开。

---

## 2. 数据量与契约（回答任务问题 3）

事实（`tp.rs:212-255`、`serve.rs:385-393`、`ferrite_kernels.cu:8853-8893/8951-8965`）：

| 项 | 值 | 说明 |
|---|---|---|
| payload `n` | **5,120 f32 = 20KB**（m=1） | `len = fb(m*dim)`，`dim=5120`（`config.rs` hidden_size=5120） |
| staging slot | **120KB** | `ar_bytes = max(hc_mult*dim=20480, VERIFY_ROWS*dim=30720) * 4` |
| `stride` | 30,720 elem | `bytes/4`；store/reduce 只覆盖 payload 的 5,120 elem |
| 每次 AR 远程写 | 8 peer × 20KB = **160KB** | peer-parallel store（`gridDim.y = world`） |
| 每次 AR 本地读 | 8 × 20KB + out 20KB | reduce 读的是**本 rank 的 staging**（非 NVLink 项） |
| 每 verify 步远程写 | 177 × 160KB = **28.3MB** | ⇒ ~70µs @400GB/s，**不是瓶颈** |
| 每 verify 步本地读带宽 | 177 × 180KB = 32MB | L2 级，**不是瓶颈** |

**⇒ 优化数据量的（bf16/fp16 传输）在收益侧没有空间，在风险侧却是最贵的**
（破坏"NCCL 升序逐位等价"契约，AR 结果直接进 residual stream，误差 4e-3 相对量级 ⇒ accept 率可能下降；
1% accept = 0.02 行/步 ≈ 0.16ms，足以吃掉全部收益）。**明确建议：不做**，除非 §5-D3 探针证明搬运 > 等待。

---

## 3. 28.6µs 是等待还是计算？（回答任务问题 4）

### 3.1 同负载对照（本文件的核心证据）

| | `_hcpost_rows`（verify） | `_hcpost`（decode） |
|---|---|---|
| `n` | 5120 f32 | 5120 f32 |
| grid | `ceil(1280/64)=20` 块 × 64 线程 | 同 |
| reduce 体 | 8 × float4 读 + 升序加 + 写 out | 同 |
| epilogue | `ar5_hc_post_col4`（hc_n=4） | 同（`hc_rows=1` ⇒ 同一 helper、同一 `__fmaf_rn` 链） |
| **实测** | **28.6µs** | **6.1µs** |

工作量逐项相同 ⇒ `28.6 − 6.1 = 22.5µs/次` **不是计算、不是搬运**。
它的三个候选来源（按可信度排序）：

1. **peer stamp 轮询的等待**（`if ((int)(cur-(e+1u)) < 0) { __nanosleep(ns); ... }`，
   `ferrite_kernels.cu:8925-8945`）：等待最慢 peer 进入本轮 pubred。无图 + host 逐发时，
   8 个 rank 线程抢 CPU、每行还有 1 次 blocking H2D + 1 次全设备 sync（`chain_dev.rs:8168/8182`），
   rank 漂移最大 ⇒ 等待最长。**decode 主链的 host 节奏紧，等待小 ⇒ 与 6.1µs 自洽。**
2. **nsys 自旋放大**：脚本头注称 per-node tracing 下该自旋"放大 ~300×"（`nsys_wave1.sh:33-40`）。
   两份数据在同一 trace 下测得，故**差值**较可信，**绝对值**（28.6µs）不可信。
3. **SM 争用**：verify 路径在同一窗口还跑 hc 侧流（`hc_mixes_auto` 的 dots/LATE）与 MoE 尾；
   AR 只占 20 块 × 64 线程（≈14 SM 的 1/10 占用），争用不至于产生 20µs 级差值。

### 3.2 额外结构发现：v5 的 poll 是**惊群**

- `p2p_ar_pubred_v5_kernel`：`if (threadIdx.x < world)` **无 blockIdx 守卫** ⇒ 每块的 8 个线程
  各轮询同样 8 个远端 flag（20 块 × 8 = **160 个轮询者**，100ns 节奏 ⇒ ~1.6G probe/s 打在 8 条 L2 行上）。
- 反证：v3 的 launch 注释 `// Publish + wait on ONE block (8 pollers, no thundering herd)`
  （`ferrite_kernels.cu:8792`）——v5 2026-09-10 合并 publish+reduce 时**把它丢掉了**
  （`:8902-8907` 的 "Saves one graph node" 说明），代价一直没测。

---

## 4. 优化方案（每项：改动 / 预期 / 成本 / 风险）

### A 类：指向"等待"（首选，因为 22.5µs/次的非工作量在这里）

| # | 方案 | 改动 | 预期 | 成本 | 风险 / 前置 |
|---|---|---|---|---|---|
| **A1** | **wait 探针**（必做第一步） | `p2p_ar_pubred_v5*`（三个变体）的 poll 循环加 `DSV41_AR_PROBE` gated 的 `clock64()` 统计（avg/max/直方图，site = attn/moe），host 每 512 次打印一行 `[ar-probe]`（沿用 `tp.rs:732-744` 的打印范式） | 把 20.8% 拆成 **真搬 / 真等待 / 伪影**；决定 A2/A3/A4 的全部投入 | 0.5 人日 + 1 GPU | 无（默认 0，不动默认路径；设计见工作区 `dsv41-cross-layer-pipe-p4.md §5`） |
| **A2** | **打开 verify 图**（`DSV41_VERIFY_GRAPH=1` 已存在，但 **lazy 臂目前不走它**：`lazy_run_row` 把 `spec_capture` 清掉、per-row 跑 `step_rows_sync`，`chain_dev.rs:8168-8200`） | 让 lazy 的 per-row `step_rows(m=1)` 也走捕获/回放（形状池已按 m=1 键控，`verify_graph_gate`） | 回放是全局 lockstep ⇒ **主机漂移→0**，peer stamp 等待趋近 0。图化已实测 **−1.6ms/步**（24.15→22.56），本 profile 正是**无图**口径 ⇒ 这是"20.8% 里最虚的一块" | 1-2 人日 + 1 GPU | 中：SWALLOW/verify 图历史有 `ar5-hang`（barrier 不对称），已有 `unanimous_i32` 对称化机制；须先确认 lazy 臂的 arrival 计数一致性 |
| **A3** | **消掉 per-row host round-trip**（`DSV41_LAZY_SDR=1`，**已实现、默认 OFF**，`chain_dev.rs:2356`） | 0 代码 | 账本 −0.7ms/步；**外加**（账本未计）rank 漂移↓ ⇒ **AR 等待↓**（与本文件同一杠杆） | 0.5 人日 A/B | 低：`DSV41_INV_CHECK=1` 全绿 + `rows_run == k_emit` 不变 |
| **A4** | **单块 poll + 设备标志广播**（治 §3.2 的惊群） | pubred 三个变体：`blockIdx.x==0` 做 stamp+poll，然后 `__threadfence_system()` + 写一个 per-round 的 device flag（用 staging 里现成的 `ctr_at` 邻位，按 `e&1` 双缓冲）；其余块 spin 在**本地** flag 上 | 轮询流量 160 → 8 线程/AR；减少 L2 争用与最后戳可见性延迟。保守 **−1~3µs/AR ⇒ −0.2~0.5ms/verify 步** | 1-2 人日 + parity | 中：跨块 release/acquire 必须正确（`__threadfence_system` + volatile flag + 双缓冲）；**与 A1 一起测**，无收益就不留 |
| **A5** | **store 节点折进 m-rows producer**（消除 verify 侧 store） | 让 `proj_mrows`/`expert_gemv_fp4_down`/`sh_exp` 的 epilogue 携带 staging 写（工作区已有 decode 侧补丁 `ar-fuse-store/ar-v5-store-epilogue.patch`，m-rows 载体需新写） | 若 HEAD 行为成立：7,120 × 5.1µs ≈ **36ms/run ≈ −0.57ms/步（−2.4%）**；若 D2 证明已折 ⇒ **0** | 2-3 人日 + parity | **中高**：载体选错 ⇒ 静默错数（last-writer 论证随 rank 变）；GEMV 签名变更 ⇒ ptxas 漂移需双 A/B（patch README §5）；**前置 D2** |
| **A6** | **tap 的 `hc_collapse` 折进 AR 的 hc_post epilogue** | `ar5_hc_post_col4` 已经对每一列遍历了 `hc_n` 行 ⇒ 顺手求 mean 写 tap 槽；仅 `cfg.dspark_target_slot(layer)` 命中的层 | 省 tap 层数 × 行数 发/步（约 4-8 层 ⇒ 9-18 发/步 ≈ **0.03-0.1ms/步**） | 1-2 人日 + parity | 中低：跨 TU 的 collapse 需 `__fmul_rn/__fmaf_rn` 钉死（照 `ar5_hc_post_col4` 的既有做法） |

### B 类：指向"结构"（收益小或风险大，择一）

| # | 方案 | 改动 | 预期 | 成本 | 风险 / 判定 |
|---|---|---|---|---|---|
| **B1** | **shared-expert merge 折进 rows AR 的 store epilogue**（ADD_EPI 的 m 行版 = 文档 B7） | 新入口 `ferrite_p2p_ar_v5_hcpost_rows_add`（镜像 `_hcpost_add`：store epilogue 发 `partial+bias`），`moe_rows` 的 `add_inplace_raw`（`chain_dev.rs:12092`）改为把 `sh_out_r` 当 bias 传入 | −88 发/步 ≈ **−0.27ms/步** + 去掉 20KB 读/20KB 写/层 | 1-2 人日 + parity | **低**（同操作数、同升序 rank ⇒ 逐位等价）；**推荐做**，是唯一"还剩的真 epilogue" |
| **B2** | **两行一批合并 AR**（任务问题 2 的设问） | — | **0**（不可行） | — | 已被设计否决且结论仍成立：早退依赖使 row i+1 的输入 = row i 的 argmax（`chain_dev.rs:8180-8200` 的 lazy_run_row 循环）；固定 pair 会把行数 2.214→4.0（**+1.79 行/步 ≈ +10ms**，`lazy-verify-optimization-path §L2`）。**减 AR 次数只有一条路：accept↑（k_emit↑）** |
| **B3** | **用 AR 窗口藏 hc 的 ⟨B⟩（dots+tail）** | 工作区已有完整设计：`dsv41-cross-layer-pipe-p4.md`（blockIdx 分区把 ⟨B⟩ 挂进 store/pubred 两次 launch，用 kernel 边界代替 ticket） | **−0.3~0.6ms/步**（窗口 5-8µs 时） | 3-5 人日 + parity | 中高：分区错/`step` 未改 ⇒ **全 rank 死锁**（有缓解：`ar_blocks` 显式参数 + watchdog）；**前置 A1 探针证明窗口 ≥7.7µs**；与 A5 互斥（store 节点消失 ⇒ dots 无宿主） |
| **B4** | **bf16 传输** | staging 以 bf16 发、reduce 读 bf16 累 f32 | 理论 −0.2~0.3ms/步 | 3-5 人日 | **高**：破坏"NCCL 升序逐位"契约（项目核心价值）；±4e-3 相对误差进 residual；accept 可能反噬。**§2 已证明搬运非瓶颈 ⇒ 不做** |
| **B5** | **PDL 重叠** | 后继 kernel 提前发射 | 0 | — | 依赖阻断：AR 的后继（下一层的 hc front / tap）都读 `h_r`，而 `h_r` 正是 AR 的 epilogue 写的 ⇒ 无合法重叠对 |

---

## 5. 执行力顺序（一页）

```
D1  v5 为何在 AR_V5=0 下仍运行（查 env 是否到 rank / 该 profile 的实际 flags）
D2  verify 侧 store 计数（7,120 期望 vs 957 实测）——决定 A5 是否还有 36ms
D3  A1 wait 探针（非 nsys 环境下跑一次，拿到 [ar-probe] 分布）    ← 唯一能定预算的实验
    ↓
A3  LAZY_SDR=1（0 代码）→ A2 verify 图（0-1 人日）→ 复测 AR 占比
    ↓
B1  rows_add（−0.27ms/步，低风险） + A4 单块 poll（若探针证明等待在 poll）
    ↓
A5（若 D2 成立）/ A6（小）/ B3（仅在窗口 ≥7.7µs 时）
```

**预期落点（诚实区间）**

| 阶段 | AR 占比 | 说明 |
|---|---|---|
| 现状（无图 + 全部 flag） | **20.8%**（4.89ms/步） | 本 profile 口径 |
| +A2/A3（去 host 漂移） | **~10-14%** | 图化 −1.6ms + SDR −0.7ms 已在账本；**AR 等待的同步下降是额外的**，量取决于 D3 |
| +B1/A4/A6 | **~8-12%** | −0.3~0.6ms/步 |
| +A5 | **~7-10%** | 再 −0.57ms/步（36ms/run） |
| **地板** | **~3-4%**（0.7-0.9ms/步） | 7136×5µs + 933×6.1 + 957×5.1 —— 只有"非工作量归零"才达得到 |

---

## 6. 诚实校准（必须写在账上）

1. **本文件的 28.6µs 是 nsys 无图口径**，而 nsys 对自旋 kernel 有已知放大（`nsys_wave1.sh:33-40`）。
   **差值（22.5µs）比绝对值可信**；绝对等待量必须用 §5-D3 的 in-kernel 探针（非 trace 环境）重测。
2. **D1 未解**：`DSV41_AR_V5=0` + `DSV41_GRAPH_STEP=0` 与 7,136 例 v5 kernel 直接矛盾。
   在这条澄清之前，"20.8%"的**口径**（是否含 v5 自旋放大）本身存疑。
3. **D2 未解**：verify 侧 store 缺失 7,120 例。A5 的 36ms 收益**以 D2 为条件**。
4. **A2 的收益不是纯机械的**：图化实测 −1.6ms/步是整个 verify 块的（含 launch 开销），
   不能把它与"AR 等待下降"相加；两者**部分重叠**记账（沿用本项目的重叠警告惯例）。
5. **B1 的 −0.27ms/步按 3µs/launch 折算**，与本项目 `~720 launches/step` 的 launch 账本同源；
   若 launch 更便宜（cuLaunchKernel ~2µs），收益等比缩小。
6. **本机无 GPU**：除 nsys 表外的一切 ms 都标了来源（实测 / 账本推算 / 设计口径），未实测项集中在 A2/A4 与 D3。

---

## 附：一句话总结

> **AR 三件套 20.8% 的主项不是通信量而是等待：同负载的 `_hcpost_rows`(28.6µs) 与 `_hcpost`(6.1µs)
> 差出 22.5µs/次的"非工作量"（peer stamp 轮询 × host 侧 rank 漂移 × nsys 放大）。
> 所以优先级是：先探针量化等待（A1/D3）→ 用图与去 host round-trip 把漂移压掉（A2/A3）→
> 再补两个机械折叠（shared merge B1、store 进 producer A5）→ 最后才考虑 AR 窗口复用（B3）。
> bf16 传输明确不做；两行合并 AR 明确不可行（早退依赖），减次数的唯一路是 accept↑。**

---

*工部 · 只读分析 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*代码行号以 HEAD 为准；读代码时以函数名为准。*
