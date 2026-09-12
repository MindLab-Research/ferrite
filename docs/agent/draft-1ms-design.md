# draft 链 4.37ms → ≤1ms 的设计

> 口径：400 tok/s 要求 `verify ≤5.5ms + draft ≤1ms + 主链吞掉 −6.15ms`。
> 输入账本：`docs/agent/draft-perf-ledger.md`（290 launch / 2.66GB / 4.9ms，本机无 GPU，ms 为推算）。
> 本文档**只做规划**，不含实施；所有判定均带 `file:line`。仓库状态 HEAD `3e044f4`（dspark wave2）。
> 生产几何：dim 5120 / vocab 129280 / mr 256 / n_mtp_layers 3 / bs 5 / TP8 / MoE 128 routed + 1 shared。

---

## 0. 结论（TL;DR）

1. **纠正一个决定性前提**：**markov 的 5 步不做 head forward。**
   `draft_head` 只发 **1 次** `head_gemv_bf16_mrows`（`dspark_dev.rs:1793`），把 bs=5 行折叠成**一次权重驻留**读，
   `head.weight` 每步只读 **1 遍**（1262 MiB）。被读 5 遍的是 **`markov_head`**（`dspark_dev.rs:1832` 的
   `for step in 0..bs` × 每次全词表扫 `dsv41_glue.cu:1400`，126 MiB/次 = 631 MiB/步）。
   ⇒ **「一次权重读服务 5 步」这个命题，在 head 上已经成立（无需再做），在 markov 上是真命题且是唯一可优化项。**
2. **账本 §4 的表与正文自相矛盾**（红旗，请门下定夺）：Path 3 正文写 `−0.25 ~ −0.45ms`，
   而汇总表却写 `+Path3 → 1.0-1.5ms`（要求 P3 单独贡献 `−1.6 ~ −2.2ms`）。
   按账本**自己的**边际收益逐项相加：`4.37 − 1.5 − 0.6 − 0.35 = 1.9ms`，
   **三路全做落在 1.9-2.6ms，不是 1.0-1.5ms。缺的那 ~1ms 没有任何一条路径认领。**
3. **缺的 ~1ms 只能来自「段级 persistent 融合」**——而它的设计**已经在仓库里**
   （`dsv41-persistent-arch.md` 的 L1 段内核 + 图内 PDL 增量），只是从未应用到 draft。
   ⇒ 本文档给 **四条**路径，**P3（融合）才是决胜项**，前面的机械/图/切分是它的铺路。
4. **词表切分对 markov 不是「省字节」，而是「使能」**：
   全词表 markov 权重 126 MiB > 全设备片上内存（148 SM × 227 KB ≈ 33.6 MB）⇒ 权重**不可能**留在片内，
   5 步必然重读；切 1/8 后 = **15.8 MiB/rank ≤ 33.6 MB** 且 < L2 60MB ⇒ 5 次扫描**天然命中 L2**。
   这是「一次读服务 5 步」唯一的物理实现路径。
5. **优先级**：`P0 机械（S/M）→ P1 图（M）→ P2 词表切分（M/L）→ P3 段级融合（L）`。
   落点：`4.37 → 3.1-3.5 → 2.6-3.0 → 2.3-2.7 → 1.0-1.4ms`。

---

## 1. 核实（对账本的三个疑点）

### 1.1 markov 的循环结构 —— 不是 5 次 head forward

`draft_head`（`dspark_dev.rs:1722-1853`）的真实结构：

| 序 | 调用 | 行 | launch | 权重读 |
|---|---|---|---|---|
| 1 | `hc_collapse` | 1739 | 1 | — |
| 2 | `rmsnorm(dspark_norm)` | 1748 | 1 | — |
| 3 | `head_gemv_bf16_mrows(bs=5)` | **1793** | **1** | **`head.weight` [129280,5120] bf16 = 1262 MiB × 1** |
| 4 | `dspark_markov_head(step=0..4)` | **1832-1848** | **5** | **`markov_head` [129280,256] f32 = 126 MiB × 5 = 631 MiB** |

- 第 3 步已经是**权重驻留（weight-stationary）**形态：kernel 一次解码自己的权重 tile，对全部 5 行做折叠
  （`dspark_dev.rs:1767-1790` 注释原文："one block decodes its weight tile ONCE and folds all bs rows against it"）。
  **head 的 5 行不需要 5 次读，也不需要切词表来省读次数。**
- 第 4 步是**逐步全词表扫描**：`dsv41_glue.cu:1400`
  `for (v = blockIdx.x*nwarp + wid; v < vocab; v += gridDim.x*nwarp)`，
  每步重读整个 `markov_head`，且 `er = markov_embed + tok*mr`（`:1389`）每步都变 ⇒ **无跨步复用**。
- `markov_embed` 每步只读 1 行（256 f32 = 1 KB）⇒ 可忽略（`markov_embed` 的 126 MiB **不**在带宽账上）。

**结论**：账本「head + markov = 2019MB = 76%」的量级正确（实测应为 `1262.5 MiB + 631 MiB = 1894 MiB`，
占 2.66GB 的 ~72%），但**归因必须改**：head 是「1 次大读」，markov 是「5 次重复读」。
二者的优化手段**完全不同**（head → 切分省字节；markov → 切分使能 L2/片内驻留）。

> 单位勘误（低危，但会影响后续 A/B 的期望值）：账本 head 写 `1357 MB`，`dspark_dev.rs:1760` 注释写 `1.29 GiB`，
> `chain_dev.rs:1068` 写 `1262 MB`。实际 `129280 × 5120 × 2 B = 1 323 827 200 B = 1262.5 MiB = 1.23 GiB`。
> 以 1262.5 MiB 为准。

### 1.2 为什么不能靠 L2 / 片内驻留（全词表）

| 项 | 字节 | L2（60 MB 口径） | 全设备片上（148 SM × 227 KB ≈ 33.6 MB） |
|---|---|---|---|
| `head.weight` | 1262.5 MiB | ✗ | ✗ |
| `markov_head`（全词表） | 126 MiB | ✗ | ✗ |
| `markov_head`（TP8 切 1/8） | **15.8 MiB** | **✓** | **✓（每 rank 16.5 MB ≤ 33.6 MB）** |

⇒ **词表切分是「一次读服务 5 步」唯一的物理使能项**，不是可选的省字节技巧。
（切分可行性：`129280 / 8 = 16160` 整除 ✓；verify 侧 `verify_head_geom` 已有 `world | vocab` 的同一约束，
`chain_dev.rs:1094-1097`。）

### 1.3 「5 步一次算」的三条候选路线 —— 逐条判定

| 路线 | 判定 | 依据 |
|---|---|---|
| **(a) 多行批处理 5 步** | **✗ 不成立** | step s+1 的输入 `ids[s+1]` 就是 step s 的 argmax（`dsv41_glue.cu:1388` 读 `ids[step]`、`:1492` 写 `ids[step+1]`）⇒ 严格顺序依赖，5 行不是同一批 |
| **(b) 设备级全局栅栏 + 权重留块内（全词表）** | **✗ 不成立** | ① 全设备片内 33.6 MB < 126 MiB，权重量本身放不下；② 现 grid = 2020 blocks × 256 thr = 517K 线程 > 设备容量（148 × 2048 = 303K）⇒ 非全驻留网格上的全局栅栏**必然死锁** |
| **(c) 词表切分 + 权重驻留（本地 L2 或 cooperative 片内）** | **✓ 可行** | 15.8 MiB/rank：① 最省事形态——**5 次 launch 不变，但每次只在 15.8 MiB 上 grid-stride，4 次重读天然命中 L2**；② 激进形态——cooperative launch（grid ≤ 常驻容量）grid-stride，每块留 row-tile 于 smem，核内 `grid.sync()` ×4 |
| **(d) 推测式 markov（beam/speculative）** | **△ 可做，不作主路径** | 用 raw head argmax 猜 5 个输入 → 一次 mrows 算出 5 行。**省同样的字节但改变 draft token**：`dspark_parity` 的逐行门会红，accept 长度需重新 A/B。仅在 (c) 的交换延迟被实测证明超预算时作为备选 |

> **(c)-① 是本文档推荐的实现**：改动量最小（只改 vocab 几何 + 加一次跨 rank reduce），
> 却拿到 (c)-② 的绝大部分收益，且**不触碰** cooperative launch / 核内 NCCL 的坑。
> cooperative / PDL 相关的风险在 `dsv41-persistent-arch.md §0/§2` 已有明确记录（"绝不把跨 rank 同步塞进单核"）。

### 1.4 跨 rank argmax 的次数 —— 账本多算了一倍

账本 §4 Path 3 说「head 的 5 行与 markov 的 5 步**各需一次** 8-rank top-1 交换」（共 10 次）。**实际只需 5 次**：

- head 的 5 行 logits **没有被任何一步单独 argmax**——代码里唯一的 `ids` 写入者是
  `dspark_markov_head`（`:1492`）。raw logits 只被 markov 就地 bias（`:1421-1422`）后 argmax。
- 所以 head 切分后，5 行 logits 是 rank-local 的，由**同一 rank 的 markov 核**就地消费；
  只需要 **markov 每步 1 次**跨 rank 归约（5 次），不是 5+1 或 5+5。

⇒ P2 的交换延迟预算减半（估 5 × 5-10µs = 25-50µs，而非 50-100µs）。
**前提**：head 的切分与 markov 的切分必须是**同一个 vocab 分区**（否则 bias 行与 logits 行错位）。
这一条要写进代码注释，它是静默错误的高发点（同 `chain_dev.rs:420-426` 对
`logits_r` 行距「NOT in the type」的告警）。

### 1.5 共享专家：draft 侧确实没接 mrows

`dspark_dev.rs:1634` 明文 `for r in 0..bs { ... }`，逐行发 5 次（`gemm_fp8_mx` w1/w3/swiglu/quant1/w2），
**没有任何 mrows 门**。routed 侧有门（`DSV41_DRAFT_MOE_MROWS`，`dspark_dev.rs:124/1478`，**默认 OFF 待 A/B**）。
verify 的模板是现成的：`chain_dev.rs:8136-8226` 的 `shared_expert_rows`
（`quant_rows` + `gemm_fp8_mrows`×2 + `swiglu_limit_q` + `gemm_fp8_mrows` + `add` = **6 launch/block**）。

**移植坑（必须带走）**：`dspark_dev.rs:1637-1640` 的注释记录了一个 8 倍越界读 bug——
`sh_il`（`SHARED_TP` 下 = `inter/world` = 288 行）与 `inter_local`（320）**不是同一个数**，
按 `inter_local` 寻址会走 8 倍。移植时 w1/w3 的 n 必须是 `sh_il`。

### 1.6 图化的层级混淆（账本 Path 2 的口径需澄清）

- `FERRITE_DRAFT_GRAPH`（`tp.rs:1519-1606`）是 **exec 级**图：`mega_d{seq}_{i}`，每个 draft **token** 一个图
  （cast_store → embed_one_dev → 整条 MTP 层链 → argmax）。**它不是**账本 Path 2 说的
  「`draft_forward` 内 290 launch 的 device 级图」。
- 账本 Path 2 要的是 **device 级**：把 `draft_forward`（`dspark_dev.rs:668`）这一整个函数体捕获。
  模板在 `chain_dev.rs:3868-3900`（`DSV41_VERIFY_GRAPH`：DRY → CAPTURE → 首次立即 launch → 之后 replay；
  输入在捕获区外刷新到**同一地址**）。
- 两者**不冲突、可叠加**，但**必须分开做、分开 A/B**。把 exec 级图当成 device 级图的收益来源会得出错误结论。

---

## 2. 设计：一次 markov 权重读服务 5 步（含 head 的处理）

### 2.1 目标形态

```
draft_head():
  hc_collapse → rmsnorm                                    (2 launch, 不变)
  head_gemv_bf16_mrows(bs=5, n=seg)                        (1 launch, 158 MiB/rank ← 切分)
  for step in 0..5:
      dspark_markov_head_sliced(step, n=seg)               (5 launch, 15.8 MiB/rank, L2 命中)
        └─ 局部 argmax → publish u64 → v5 pubred → 全局 ids[step+1]   (5 次跨 rank 往返)
```

### 2.2 改动清单（P2 主体）

| # | 位置 | 改动 | 依据/模板 |
|---|---|---|---|
| 1 | `weights.rs:90` | `head.weight` 由 `Shard::Replicated` 改为按 vocab 行切 `Rows(vocab/world)`；`mtp.{s}.*` 的 `markov_head.head/embed` 同切 | `weights.rs:29/90`；`chain_dev.rs:1063-1107` 的 `verify_head_sliced` |
| 2 | `dspark_dev.rs:1793` | `head_gemv_bf16_mrows` 的 `vocab` 实参 → `seg = vocab/world`；`logits` 行距 → `seg` | `device.rs:3207` 签名不变（n 即行宽） |
| 3 | `dsv41_glue.cu:1374-1498` | markov 核的 `vocab` → `seg`；**核对 `mk_partial` 长度**：现在 `MARKOV_MAX_BLOCKS=2048`（`dspark_dev.rs:48/396`），seg=16160 → `16160/(8×8)=253` blocks，仍在上限内，无需改 | `glue.cu:1507-1510` |
| 4 | 新增（device.rs） | `dsv41_argmax_pub`（或复用 `dsv41_argmax_sliced` 单行形态）把局部 packed key 做 v5 一轮广播；`argmax_packed` 缓冲已在 `chain_dev.rs:253` 有先例 | `chain_dev.rs:253-257` |
| 5 | `dspark_dev.rs:1845` | `mk_ctr` 的**自重置语义必须保留**（`glue.cu:1493-1495`：下一个 step 的 launch 必须从 0 开始） | 同处注释 |
| 6 | `dspark_dev.rs` unit dumps | `logits_row0` 的 dump 尺寸由 `[vocab]` → `[seg]`，或改为跨 rank 拼回 | `dspark_dev.rs:1831` |
| 7 | 门禁 | `DSV41_DRAFT_HEAD_SLICED` 默认 **OFF**（house rule：新路径先做 A/B 臂），`world==1 / vocab%world / 非 BF16` 回退全词表 | `chain_dev.rs:1094-1097` 同款回退 |

### 2.3 预期收益（P2）

| 项 | 现状 | 切分后 | 省 |
|---|---|---|---|
| head 权重读 | 1262.5 MiB | 158 MiB | **1105 MiB** |
| markov 权重读 | 631 MiB | 15.8 MiB 强制 + 4 × L2 命中 | **~600 MiB** |
| 合计字节 | 2.66 GB | ~0.83 GB | **−1.7 GB** |
| 代价 | — | 5 次跨 rank argmax 往返 | +25~50 µs |

按 4.5 TB/s 有效 GEMV 带宽换算：`−1.7 GB / 4.5 TB/s ≈ 378 µs`，扣交换 ≈ **−0.33ms**。
与账本 `−0.25 ~ −0.45ms` 一致。**诚实提醒**：head 的 1262 MiB 在总时长里只值 ~0.3ms
（账本 §0.4 自己承认「head 是字节大头但不是时间大头」），所以 **P2 的上限就是 ~0.4ms，不可能单独达标**。

---

## 3. 四条路径：优先级 / 预期 ms / 工作量 / 风险

> 排序 = 执行顺序（后项依赖前项）。**四条必须全做**，且**只有 P3 能穿过 1ms 门槛**。

### P0 —— 机械搬运 verify 的既有优化　【工作量 S-M｜风险 低】

搬 **共享专家 mrows**（`dspark_dev.rs:1634` → verify 模板 `chain_dev.rs:8136`）＋
**routed mrows A/B**（门已在 `dspark_dev.rs:1478`，只差实测）＋ **rope 多行**（`apply_rope` 已支持 rows+step）
＋ `sparse_attn_orope`（launcher `device.rs:2250`，符号已挂 `:958`）＋ `hc_post_inplace`/去 memcpy（launcher `device.rs:4479`）＋ gate 多行（`head_gemv_bf16_mrows` 先例）。

| | launch | 字节 |
|---|---|---|
| 现 | 290 | 2.66 GB |
| 后 | **~130-150** | 2.60 GB |

自算削减：共享专家 78→18（−60）、routed 45→9（−36）、rope 30→6（−24）、orope 24→3（−21）、
memcpy 9→0（−9）、gate 15→3（−12）＝ **−162**（账本保守取 290→150，此处两者都记）。
**预期 4.37 → 3.1-3.5ms**（按每条 launch 省 10-15µs 的执行+依赖延迟）。
风险：低——每项一个独立门 + 逐位 parity（`dsv41-layer-fusion.md` 的三条硬性律）。

### P1 —— draft 的 device 级 CUDA graph　【工作量 M｜风险 中】

捕获 `draft_forward` 全函数体。必须先解账本 §4 的 4 个 host 依赖：
- D1 每步阻塞 H2D 的 `upload_i32(ids)`（`dspark_dev.rs:771`）→ 图外 H2D 到捕获录下的同一地址（verify 的 `ids_r` 纪律，`chain_dev.rs:3914-3919`）。
- D2 `seed_window` 的 `slot = pos % win` 主机算地址（`dspark_dev.rs:1892`）→ `Device::ring_append`（launcher `device.rs:3780`；注释 `dspark_dev.rs:1888-1891` 已点名）。
- D3 `memcpy_d2d(window→all_kv)` 的主机 size/branch（`dspark_dev.rs:1152-1174`）→ 定长 win 行 + device 派生 `n_win`。
- D4 `ensure_idxs` 的 H2D（`dspark_dev.rs:2065-2085`）→ device 侧 idxs 生成，或把捕获推迟到 `pos ≥ win`。

**预期 4.37 →（P0 后）2.6-3.0ms**。增量来源 = 节点间空隙；`graph_bench` 实测 2.904µs → 0.411µs/节点。
**PDL 是真正的增量**（`dsv41-persistent-arch.md §4`：图内 ~0.2-0.3µs/节点），
建议图捕获与 `FERRITE_PDL=1`（`ferrite_kernels.cu:725-753` 的 `pdl_or_plain`）**一起 A/B**。
风险：中——捕获合法性（无 `cudaMalloc`/`sync`）、per-request drop、AR epoch 与捕获的 rendezvous
（`chain_dev.rs:3930-3935` 的 `host_barrier` 对）。

### P2 —— head + markov 的词表切分　【工作量 M-L｜风险 中-高】

见 §2。**预期 2.6-3.0 → 2.3-2.7ms（−0.25~−0.4ms）**。
风险：中-高——
① 跨 rank 协议：`argmax_pub` 曾与 AR 暂存冲突**死锁**（账本 §3 注），且 `44f4956` 记录过
「watchdog 静默超时 → 归约出陈旧半量 → 发出一个看似合理的错 token」——**必须带 watchdog + 行数打印**；
② 数值：129280 路近邻 argmax，切分后 key 必须携带**全局 index**（`chain_dev.rs:1071-1073` 的设计），
否则 tie-break 在 rank 间不一致；
③ head/markov 分区必须同构（§1.4）。

### P3 —— draft 的段级 persistent 融合　【工作量 L｜风险 高】★决胜项

**这是唯一能填上账本那 ~1ms 缺口的路径。** 设计已在 `dsv41-persistent-arch.md`（L1 段内核），
draft 侧的段划分与主链同构，每 block 3 段：

```
段 A: hc_mixes → hc_collapse → rmsnorm → attention(wq_a/q_norm/wq_b/rope/wkv/kv_norm/
        rope/ring/copy/sparse_attn/rope⁻¹/quant/wo_a/wo_b)      → 1 核
段 B: hc_post → hc_mixes → hc_collapse_norm → MoE(gate/route/quant/
        experts/shared/add)                                      → 1 核 + AR(核外)
段 C: hc_post                                                      → 1 核
```

3 block × 3 段 + 3 AR + 一次性 6 ≈ **~18-20 launch**（从 290 降一个数量级）。

**为什么它能穿过 1ms，而 P0-P2 不能**：P0-P2 之后剩下的 ~130 个 launch 仍是**串行小核**，
每个 ~10-15µs 的执行/依赖延迟（账本 §0 的 83%）**不随字节减少而降**。
段内核把这些延迟压成**每段一次**，此时耗时重新由**字节**决定：P2 后字节 ~0.83 GB @ 4.5TB/s ≈ 185µs，
加上段内 ramp-down 与 AR ⇒ **1.0-1.4ms**。

风险：高。**三条铁律不可破**（`dsv41-layer-fusion.md` / `persistent-arch.md §5`）：
① 同编译单元或显式 `__fmaf_rn`（跨 CU 的 FMA 收缩差异 = 1 ULP = 确定性文本变化，`hc_post_parity.rs` 实测 2.98e-8）；
② 归约顺序逐指令照抄（down 升序 slot、AR 升序 rank、hc split=1）；
③ 每个融合核配 `*_parity.rs` 逐位门禁，先于 serve A/B。
**AR 绝不进核内**——`hc-merge` 已实测单核 + ticket 自旋 = **+3.2ms 回归**；段边界必须落在 AR。

---

## 4. 路劲汇总

| 路径 | 累计 draft | launch | 字节 | 工作量 | 风险 | 关键动作 |
|---|---|---|---|---|---|---|
| 现状 | **4.37ms** | 290 | 2.66 GB | — | — | — |
| **P0** 机械 | **3.1-3.5ms** | ~130-150 | 2.60 GB | S-M | 低 | 共享专家 mrows（+routed A/B、rope、orope、memcpy、gate） |
| **P1** 图 | **2.6-3.0ms** | 150 节点图化 | — | M | 中 | 解 D1-D4；PDL 同 A/B |
| **P2** 切分 | **2.3-2.7ms** | — | **0.83 GB** | M-L | 中-高 | head/markov 切 1/8 + 5 次 v5 argmax |
| **P3** 段融合 | **1.0-1.4ms** | **~18-20** | 0.83 GB | **L** | **高** | 3 段内核 + AR 留核外 + parity 门禁 |

**诚实结论**：账本「三路全做 = 1.0-1.5ms」**不成立**（其自身边际收益只到 ~1.9-2.6ms）。
**P0-P2 是必要条件、P3 是充分条件**。要稳进 ≤1ms，**必须在 P3 上投入**；P0-P2 的价值是
为 P3 把字节降到 0.83 GB、把图与切分的协议先跑通（P3 的段内核要在图里，且 P3 的 head 段依赖 P2 的切分）。

**顺序建议**：`P0（本周，低风险拿 1ms）→ P1 + P2 并行（协议+图）→ P3（长线，唯一能达标）`。
不要为 P2 的 0.3ms 推迟 P3 的立项——P2 在 P3 里是**前置件**（head 段的 158 MiB 是段内核的输入几何）。

---

## 5. 影响范围

- **修改文件**
  - `crates/ferrite-models/src/dsv41/dspark_dev.rs`（P0 共享专家/rope/gate/memcpy；P1 捕获；P2 切分几何）
  - `crates/ferrite-models/src/dsv41/device.rs`（P0 `gemm_fp8_mrows` 接线；P2 argmax publish 启动器）
  - `crates/ferrite-models/src/dsv41/weights.rs`（P2 `head.weight` / `markov_head` 分片声明）
  - `kernels/cuda/dsv41_glue.cu`（P2 markov 核 vocab→seg；P3 段内核）
  - `kernels/cuda/dsv41_kernels.cu`（P3 段内核，**同 CU 纪律**）
  - `crates/ferrite-models/src/dsv41/dspark_parity.rs`（逐位门禁，新增）
- **影响模块**：draft forward / MTP exec 图（`tp.rs`，P3 后节点数暴降但形状不变） / verify 的 argmax 复用面（P2 与 verify 共享 `argmax_sliced` 家族）
- **兼容性**：**无 API breaking change**。全部新增门（`DSV41_DRAFT_HEAD_SLICED` 等）**默认 OFF**；
  `world==1`、`vocab % world != 0`、非 BF16 head、`.so` 缺符号 一律回退现路径。
  **但**：P2 会改变 unit dump 的 `logits_row0` 形状（`[vocab]` → `[seg]`）——依赖该 dump 的测试需同步。

## 6. 风险评估

| 风险 | 应对 |
|---|---|
| P0 共享专家移植重复 `inter_local` vs `sh_il` 越界（`dspark_dev.rs:1637-1640` 的历史 bug） | 移植时显式断言 w1/w3 的 n == `sh_il`；parity 对齐 verify |
| P1 捕获记录了非法的分配/同步，或 AR epoch 错位 | DRY → CAPTURE → 立即 launch 的既有纪律；失败**必须打印**（`chain_dev.rs:3964-3979` 的「print, do not silently degrade」） |
| P2 跨 rank argmax 死锁 / 静默超时发出错 token | watchdog + 行数打印（`44f4956` 的教训）；先跑单测（global index packing / tie→最小全局 index / epoch 恰好 +1） |
| P2 切分后 draft 的 top-1 在 rank 间 tie-break 不一致 | packed key 必须带**全局 index**（`chain_dev.rs:1071-1073`） |
| P3 融合破坏逐位确定性（跨 CU FMA 收缩 = 1 ULP） | 同 CU 或显式 `__fmaf_rn`；每个融合核配 `*_parity.rs`；split 恒 = 1 |
| P3 把 AR 折进核内 → +3.2ms 回归 | 段边界锚在 AR；核内只允许「段首全块栅栏」形态（`persistent-arch.md §2`） |
| 账本 ms 均为推算，本机无 GPU，边际收益假设（10-15µs/launch）未实测 | **先做一次 nsys**（draft 段仅 ~290 节点，好定位）拆开「发射 / 字节 / 依赖延迟」三向，再定 P3 的投产 |

## 7. 建议分工

- **工部**：P0 全部实施（共享专家 mrows 移植 + routed 门 A/B + rope/orope/gate/memcpy）；P2 的 kernel 侧（markov 核 vocab→seg、argmax publish）
- **户部**：① 一次 nsys 实测填平账本 §6.1 的三向拆分；② P0 每项的 A/B（单开门，测 `DSV41_TIMING` 的 `draft=`）；③ P3 的字节地板复算
- **刑部**：P0 移植的逐位 parity 与边界用例（bs=1、world=1、`vocab%world!=0` 回退）；P2 的 tie-break/epoch 断言
- **兵部**：P2 跨 rank 协议的故障审查（死锁/超时/陈旧半量 → 静默错 token 的失败模式）
- **礼部**：P2/P3 落地后更新 `draft-perf-ledger.md`（修正 §4 表的自相矛盾）与本节知识文档
- 吏部：本轮无新增风格面，不分配

---

*中书省 · 基于 2026-09-12 仓库状态（HEAD `3e044f4`）。launch 计数与字节为代码推算值；ms 为边际估算，待 nsys 实测。*
