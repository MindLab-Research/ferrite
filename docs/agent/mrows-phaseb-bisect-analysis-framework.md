# mrows Phase B bisect —— 分析框架（预置）

> 工部（ministry-works）· 2026-09-12 · **只读勘察 + 本文档（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 代码基线：HEAD `123f0ec`（`crates/ferrite-models/src/dsv41/{chain_dev,config,device}.rs`、
> `kernels/cuda/dsv41_kernels.cu`、`scripts/batched_400_v2.sh`）。
> 上位文档：`docs/agent/mrows-swallow-batched-implementation-design.md`（§3/§7/§8）、
> `docs/agent/b5-b4-b6-gpu-verification-design.md`、`docs/agent/l4-mgrid-first-step-design.md`（§3/§4）、
> `docs/agent/dspark-correctness-chain.md`（全栈 v2 结果段）。
> **口径纪律**：每条数字标来源（**实测 / 读码 / 代数**）。凡与任务前提冲突处显式给出 file:line。

---

## 0. 先读四条（其中两条**改写 bisect 的判读**）

1. **❗ fold_r=auto 的 `n <= 1024` 边界恰好把 `wo_b` 折了 —— 设计明写它不该被折。**
   设计 `l4-mgrid-first-step-design §3.4` 的形状规则把「不折」的名单写成 **`wq_b`/`wo_b`（n ≥ 4096）**；
   但**实测形状下 `wo_b` 的 n 不是 4096，而是 1024**：
   `n = ol_local = o_groups × o_lora_rank / world = 8 × 1024 / 8 = 1024`（`config.rs:511,518` 生产形状断言；
   `chain_dev.rs:11634` `ol_local = ol_total / world`；调用点 `chain_dev.rs:11785`）。
   ⇒ `dsv41_mrows_fold_r_for`（`dsv41_kernels.cu:3978-3985`，`n <= 1024 -> 1`）把**权重矩阵最大的那一个**
   （1024×5120 = 5.24 MB/rank）也折成 `fold_r = 1`、`ng = 6`、grid 从 256 → **1536**。
   **这是 auto 规则唯一一处与设计意图相反的折**，也是 fold_r 臂里放大面最大的一项。

2. **❗ 「6× 权重读 ⇒ 6× 慢」在字节/指令两个口径下都差 ~3 个数量级 —— 该假设的机理不成立。**
   fold_r=auto 实际只折 2 个投影（`wkv` n=512、`wo_b` n=1024），重读增量：
   `(6−1) × (2.62 + 5.24) MB = 39.3 MB/层 × 40 层 = +1.57 GB/步`（代数）。
   @ HBM 6–8 TB/s ⇒ **+0.20~0.26 ms/步**；而观测到的步时差是
   `1/10.3 − 1/63.8 ≈ 583 − 94 = +489 ms/步`（实测口径，m=6 → 6 tok/步）。
   ⇒ 该机制只解释了观测量的 **~0.05%**。指令账同向：staging warp-issue 增量 ≈ 3.1 M/步 ÷ ~888 G issue/s ≈ **3.5 µs/步**（代数）。
   **结论：fold_r 若真是 6× 源，机理必然是「非线性」的（同步/占用/L2 抖动/图），不是字节账。** 见 §3。

3. **B5/B4/1b 的票面合计 +0.5~0.8 ms（增益，不是退化）⇒ 三者都不可能造成 6×。**
   B5 −0.13 ms、B4 −0.13 ms（设计 §附-4）、1b −0.3~0.5 ms（`l4-mgrid §4.2` 指令账）。
   ⇒ 63.8 → 65~68 tok/s。**如果 bisect 落在 ~10，那不是「B5/B4/1b 的预期代价」，而是其中某一项有 bug。**

4. **⚠️ 全栈 v2 的两个 gate（`DSV41_MROWS_FOLD_R` / `DSV41_MROWS_ACT_CPASYNC`）在脚本里没有 arm。**
   `grep -n "MROWS_FOLD_R\|MROWS_ACT_CPASYNC" scripts/` = **0 命中**；而 serve 的 env 只由 `$GATES_ONELINE` 构成
   （`batched_400_v2.sh:730`，经 `rssh`）。⇒ 若 v2 是用本脚本起的，**这两个 gate 根本没上节点**（「幻影门」）。
   **bisect 前必须先看 `$LOGDIR/<tag>.env` 的 `/proc/<pid>/environ` 实读**：确认 fold_r 确实被关、
   且确认 v2 那一轮 fold_r 确实被开。**在拿到实读证据之前，v2 的 6× 不能被归因到 fold_r 上。**

---

## 1. 读码确认：auto 到底折了哪些调用点

`gemm_fp8_mrows` / `proj_mrows` 在 batched 路径的调用点（`attention_rows` = `chain_dev.rs:10645-11861`）：

| # | 投影 | 调用点 | n | k | nwarps | nt | **auto fold_r** | **ng** | **grid** |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `wq_a` | `chain_dev.rs:10765` | `ql` = 1280 | `dim` = 5120 | 4 | 320 | `m`=6（1280>1024） | 1 | 320 |
| 2 | **`wkv`** | `chain_dev.rs:10774` | `hd` = 512 | `dim` = 5120 | 4 | 128 | **1**（512≤1024） | **6** | **768** |
| 3 | `wq_b` | `chain_dev.rs:10953` | `nlh*hd` = (64/8)×512 = 4096 | `ql` = 1280 | 8 | 512 | `m`=6（4096>1024） | 1 | 512 |
| 4 | **`wo_b`** | `chain_dev.rs:11785` | `ol_local` = **1024** | `dim` = 5120 | 4 | 256 | **1**（1024≤1024） | **6** | **1536** |
| 5 | `idx_wq_b`（indexer） | `chain_dev.rs:12070` | `idx_nh*idx_hd`（**待确认**） | `ql` = 1280 | ? | ? | 若 ≤1024 则**折** | — | — |

形状与 warps 来源（读码）：
* `config.rs:511-520` 生产形状：`dim=5120 / n_heads=64 / head_dim=512 / q_lora_rank=1280 / o_lora_rank=1024 / o_groups=8`；
  `n_layers=40`（`config.rs:514`）。
* `chain_dev.rs:10648-10655` `ql = cfg.q_lora_rank`、`nlh = nh / world`；`world = TP`，脚本默认 `TP=8`（`batched_400_v2.sh:114`）。
* `dsv41_gemv_warps_for`（`dsv41_kernels.cu:3820-3823`）：n ≥ 2048 → 8，否则 4（`g_gemv_warps` 默认 4，`:3751-3756`）。
  `DSV41_MROWS_SMALL_N_ADAPTIVE` 默认 OFF ⇒ 不参与。
* fold 规则：`dsv41_kernels.cu:3978-3985`；grid = `nt * ng`（`:5591-5592`、`:5631`）。

**两项代数量级（读码+代数）**

| 投影 | 权重/rank | 基线 staging | auto staging | 增量 |
|---|---:|---:|---:|---:|
| `wkv` | 512×5120 = **2.62 MB**（与设计 §3.6 的 2.62 MB 逐字吻合） | 2.62 MB | 15.7 MB | +13.1 MB/层 |
| `wo_b` | 1024×5120 = **5.24 MB** | 5.24 MB | 31.5 MB | +26.2 MB/层 |
| 合计 | 7.86 MB | 7.86 MB | 47.2 MB | **+39.3 MB/层 = +1.57 GB/步** |

> 权重是 fp8（1 B/元素）。fp32 激活的 staging **总量不变**（`nt·m·k` 与 fold_r 无关，设计 §4.1 ✓ 已验证）。

---

## 2. B5 / B4 / 1b 的票面与「能不能造成 6×」

| 项 | gate | 替换 | 票面 | 能否 6× |
|---|---|---|---|---|
| **B5** | `DSV41_GATE_MROWS_ROUTE=1`（+`GATE_MROWS`） | gate GEMV + `route_topk` 两发 → 1 发/层 | −0.13 ms（20 发/步） | ❌ 票面太小；**但它引入了全 grid 选举（见 §3-M2）** |
| **B4** | `DSV41_RMSNORM_ROPE_MROWS=1` | `norm_rows_on` + `apply_rope_on` 两发 → 1 发/层 | −0.13 ms | ❌ 票面太小 |
| **1b** | `DSV41_MROWS_ACT_CPASYNC=1` | 激活 staging 标量→`cp.async16` | **−0.3~0.5 ms（增益）** | ❌ 方向相反 |
| 合计 | — | — | **+0.5~0.8 ms ⇒ 65-68 tok/s** | ❌ |

**一处对判读有影响的读码结论（修正设计 §3-B4 的措辞）**：B4 **不需要** FORK 才上场。
`chain_dev.rs:11043-11060` 的接线点**无条件**执行（`if rmsnorm_rope_mrows() { … }`），
`kv_stream` 在 FORK 未取时回落为主流（`:10825`）。⇒ **bisect 臂里 B4 是真在场的**（不是空门），
只是「FORK 未开 ⇒ 无法顺带验证 side-stream 红线」。这排除了「bisect 只测了两项」的误判。

---

## 3. 6× 的非线性机理候选（按嫌疑排序，需 nsys/证据判别）

> 判据：**看 kernel 的 TIME（不是 Instances）**。字节账只值 0.2 ms/步，所以真凶必然表现为
> 某个 kernel 的**单核时间**或**步内串行段**暴增，而不是 grid/instances 变大。

| # | 机理 | 为什么可疑 | 判别证据 |
|---|---|---|---|
| **M2** | **B5 的 route 选举 = 跨 block 软件栅栏** | **唯一引入「新同步」的改动**。EAGER 的 `ferrite_gemv_bf16_v2_route` 是「last-block election on `ctr`」；B5 把它扩到 `rows` 维（`mrows-swallow… §8-B5`：`route_ctr_r` 4B、build 时 zero 一次、kernel 自己复位）。若复位/选举判据在**图 replay 40 层**下出错，最后一个 block 会自旋等待 → 每层 +数 ms 完全可能。 | nsys：`ferrite_gemv_bf16_v2_mrows_route` 的**单核 duration**（不是次数）；`route_topk` 40→0 是否成立；`route_ctr_r` 的 zero-once 纪律 |
| **M4** | **1b 的 `cp.async` 组管理** | 1b 把激活 staging 换成 `dsv41_cp_async16` 并把 `dsv41_cp_wait_all()` 放在**权重 staging 之前**（`dsv41_kernels.cu:5369-5391`）。组泄漏/等待位置错会导致隐式串行化（数值不受影响——与「输出正确」一致）。 | nsys：`gemm_fp8_mrows_kernel<6>` 的 duration（次数不变、时间暴涨则是它）；`1b` 单变量 A/B |
| **M5** | **图捕获 × grid 变化 / counter 复位** | v2 同时改了 grid（fold_r）与新增 counter（B5）。CUDA graph 对「每次 replay 必须自复位」的 counter 极敏感（本仓 `route_ctr` 纪律反复强调）。 | 关图（`DSV41_VERIFY_GRAPH=0`）复测同一臂 |
| **M1** | **fold_r 的 grid ×6 触发 L2 抖动** | 方向可疑但量级不足（§0-2）。仅当 6 个 row-group 的时间错位使权重集超出 L2 才有意义；B300 L2 远大于 47 MB/层。 | nsys 的 L2/HBM 计数器；或纯 kernel 微基准（`tests_*`） |
| **M3** | **B4 × FORK stream** | B4 无条件在场；若臂里同时开了 FORK，新增一条 side stream 的同步面。 | nsys 的 stream 时间线；`apply_rope_kernel` kv 侧 40→0 |

**排除法要点**：M2/M4/M5 都**与 fold_r 无关**。⇒ **bisect（fold_r OFF）若仍 ~10，则 fold_r 直接出局**，
嫌疑收窄到 M2/M4/M5，且 B5 是唯一引入同步的那一项 ⇒ 下一步 bisect 必须**先拆 B5**。

---

## 4. 判定框架（bisect 结果 → 结论 → 下一步）

| bisect | 代数量级判读 | 结论 | 下一步（按顺序） |
|---|---|---|---|
| **~63-65 tok/s** | 6× 全部归 fold_r | **fold_r 是唯一 6× 源**；但机制**不是**字节账（§0-2）⇒ 按 M1/M5 收窄 | ① **先修 auto 规则的边界**：把 `wo_b`（n=1024）排除（阈值改 `< 1024`，或按实测形状表重写 `dsv41_mrows_fold_r_for`）→ 只折 `wkv`；② `fold_r` 单变量扫描 `auto / 2 / 3 / 6 / 1`（运行期参，免重编译）；③ 若「只折 wkv」仍慢 ⇒ 折本身有问题（M1/M5），退回 `fold_r=6` 并把 1b 作为独立收益项保留；④ B5/B4/1b 可以保留 |
| **~30-40 tok/s** | ≈2× 由 fold_r（非线性），≈2-3× 由 B5/B4/1b | **部分退化** | 下一层 bisect：**(a) fold_r=auto 单独**（B5/B4/1b 关）与 **(b) B5+B4+1b 单独**（fold_r=6）各一臂；用 §5 的 nsys 硬证判 M2 vs M4 |
| **~10 tok/s** | fold_r 差额 = 0 ⇒ **fold_r 出局** | **B5/B4/1b 是主退化源**（票面 +0.5~0.8 ms 与观测 −489 ms/步，差 3 个数量级 ⇒ **是 bug 不是代价**） | 逐项拆，顺序按嫌疑：**① B5**（唯一引入同步；先验 `route_ctr_r` 复位纪律）→ **② 1b**（cp.async 组管理）→ **③ B4**（last：无条件在场但无同步面，票面最小） |
| **~1-5 tok/s / 挂起** | — | 已不是「退化」而是「卡死/自旋」 | 直接查 B5 的 election 死锁（counter 未复位 / `__threadfence` 缺失）+ 1b 的 `cp.async` 组泄漏（err 716 家族） |

**与任务给定框架的差异**：任务把「~10 ⇒ B5/B4/1b 主源 + fold_r 可能中性」写成并列。
本框架把它改成**强结论**：bisect ~10 ⇒ fold_r **出局**（因为它的差额本该是 −0.2 ms，不可能掩盖任何东西）；
且「B5/B4/1b 是主源」= **它们之中有 bug**（票面是增益，不是 −490 ms/步）。

---

## 5. 每臂必录证据（缺一不能下结论）

| # | 证据 | 命令 / 判据 | 为什么是硬证 |
|---|---|---|---|
| V1 | **env 实读** | `tr '\0' '\n' < /proc/$(pgrep -x ferrite-serve)/environ \| grep -E 'DSV41_(MROWS_FOLD_R\|MROWS_ACT_CPASYNC\|GATE_MROWS_ROUTE\|RMSNORM_ROPE_MROWS\|VERIFY_FORK)'`（脚本已 dump 到 `$LOGDIR/<tag>.env`，`:741-743`） | **本仓 #1 陷阱**；且 §0-4 已指出 fold_r/1b **没有脚本 arm** ⇒ 这一条决定 v2 的 6× 到底能不能归因 |
| V2 | **符号存在性** | `nm -D libferrite_kernels.so \| grep -c ferrite_gemv_bf16_v2_mrows_route` / `... dsv41_rmsnorm_rope_mrows` | decline 是静默 `Ok(false)`；v2 的「符号检查 = 2」（`dspark-correctness-chain.md:6478`）已满足 |
| V3 | **上场证据（核名）** | nsys sum 表：`ferrite_gemv_bf16_v2_mrows_route` / `dsv41_rmsnorm_rope_mrows`（B4/B5 各有独立核名） | B5 的**设备核名与 plain fold 逐字相同**（同一 `gemv_bf16_nt_kernel<NT,WPR>`）⇒ B5 的硬证退化为 **`dsv41_route_topk` Instances/步 40→0**（`b5-b4-b6… §0-5`） |
| **V3b** | **fold_r 上场的硬证 = `gemm_fp8_mrows_kernel<6>` 的 grid** | auto：`wkv` 768 / `wo_b` 1536；OFF：128 / 256（§1 表） | 这是**唯一**能证明 fold_r 真跑了的证据；也是修 auto 规则后的回归判据 |
| **V3c** | **单核 duration（本轮最关键）** | nsys：上述各核的 **duration**，以及 `gemm_fp8_mrows_kernel<6>` / `route_topk` / `apply_rope_kernel` 的 duration 与次数 | 字节账只值 0.2 ms/步 ⇒ 真凶必表现为**时间**而不是**次数**（§3 判据） |
| V4 | **步时** | `[dspark] steps=` 的 `verify_ms` / `steady_median`（丢前 10，skip=20） | 与 tok/s 交叉校验 |
| V5 | **正确性** | 计数：数字顺序 + 前 61 行；出师表：零拉丁 + `先帝创业未半` + 无双字 | v2 已通过（`dspark-correctness-chain.md:6477`） |
| V6 | **逐位** | `k_acc` 序列逐位不变（B1/B2/B3/B4/B5 都声称 by construction） | B6 不在本轮矩阵；B5/B4/1b 都走 V6 |

---

## 6. bisect 跑着时就能做的零-GPU 预检（**建议立刻做**）

1. **核对 bisect 臂的 env 实读**（V1）：`$LOGDIR/<bisect_tag>.env`
   * 必须**不出现** `DSV41_MROWS_FOLD_R`（或 `=0`）；
   * 必须出现 `DSV41_GATE_MROWS_ROUTE=1`、`DSV41_RMSNORM_ROPE_MROWS=1`、`DSV41_MROWS_ACT_CPASYNC=1`；
   * **特别看 `DSV41_VERIFY_FORK`**：B4 无条件在场但 FORK 决定它走哪条 stream（§2）。
2. **核对 v2 那一轮的 env 实读**：若缺失 `DSV41_MROWS_FOLD_R=auto`，则 6× **不能被归因到 fold_r**，
   判读表要整体改写成「B5/B4/1b 的 bug 排查」（等同 §4 的 ~10 分支）。
3. **确认 auto 下的 grid 预期值**已写进本轮 notebook（768 / 1536），方便 nsys 表一眼比对（V3b）。
4. **确认 indexer 调用点**（`chain_dev.rs:12070`）的 `idx_nh*idx_hd` 实际值——若 ≤1024，auto 还要多折一个
   调用点（8/40 层），是 auto 规则缺陷的第二个实例。

---

## 7. 结论一句话

> **bisect 的结果不是「fold_r 有罪/无罪」的二选一，而是一次「字节账 vs 非线性账」的分诊**：
> 因为 fold_r 的字节账只有 0.2 ms/步（差 3 个数量级），**无论 bisect 落哪，
> 都不存在「fold_r 用 6× 权重读解释了 6× 退化」这条路径**。
> 因此：~63 ⇒ fold_r 有罪但机理是 M1/M5（且先修它的 `wo_b` 边界缺陷）；
> ~10 ⇒ fold_r 出局、B5 是头号嫌疑（唯一引入同步的改动）。
> **先拿 `/proc/<pid>/environ` 实读，再谈归因。**（§0-4）
