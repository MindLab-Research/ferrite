# moe-batch-fwd 第五路 A —— m=1 → m=6 的「每-value L1TEX 倒退」判决 + 微基准 + L2 改法设计

> 工部 · 2026-09-13 · **禁止 GPU/e2e**（全部结论为静态 + compile-only + 远端 nvcc）。
> 基线 HEAD `c520108`（工作树 3 个 peer 改动 + 1 个 untracked doc，均非本次触碰）。
> 交付物：①两调用点逐参数 diff 表 ②配置差异判定 ③（无）④NCU 微基准 ⑤L2 改法设计 ⑥cargo check + nvcc。

---

## 0. 五行结论（先看）

1. **两调用点的 launcher 参数逐位相同**，`rows`（1 vs m）与缓冲对象是**唯一**差异，而 `rows` 只变成
   `gridDim.z`（`dsv41_experts_mxf4.cu:1385`）。**候选 1「对齐两臂参数」在 batched 两臂这一读法下 = 空集，
   没有免费收益。**（§1）
2. **「条 L1TEX/value」是每-lane 的指令比，不是访存节拍。** 我用内核自己的记账口径手算生产 pair body：
   `(1 LDG.128 + 2 LDG.U8 + 16 LDS.32 + 8 LDS.64) / 每 lane 每 group 32 个 fp4 value`
   加上按 CTA 摊薄的前导 → **0.882 条/value**，与 「eager 0.88」**三位吻合**。（§3.1）
3. 而 `arow` 只平移基址 —— 内核的 ROW INDEPENDENCE 契约保证同一 lane 的指令流与 `rows` 无关。
   **⟹ 1.66× 不可能由 grid.z 几何产生。** 占用也反着指：m=6 的占用**更高**（§3.2），
   resident 窗口内的专家数只从 6 涨到 7.4（×1.23，§3.3）。**候选 2/3 都被构造性削弱。**
4. 于是 1.66× 只能来自**不同的指令流**。仓库里**确实存在**两条可达的分派不对称，能让两臂跑不同 kernel：
   `moe_rows()` 永不查 `moe_batch()`，而 `moe()` 查；`DSV41_NO_GEMV_FP4` 又把 rows==1 改派到
   `expert_gemv_fp4_kernel`（48 regs，另一支内核）。**这才是候选 1 的真身，而且不是免费收益**（§2）。
5. 交付：微基准 `scripts/expert_mrows_l1tex_bench.cu`（rows × ids 双因子，可分辨「指令流」与「miss」两种读法）
   + 三个 L2/局部性改法（含零代码 A/B）。**没有改任何生产代码**（无差异可修 + 不越权）。

---

## 1. 交付① 两调用点逐参数 diff（file:line）

两侧都是 `Device::expert_gate_up_fp4_batched` → `dsv41_expert_gate_up_fp4_batched`（21 参 ABI，
`device.rs:6530` / `dsv41_experts_mxf4.cu:3523`）。**同一 C 符号、同一 launcher。**

| # | C 形参 | eager `chain_dev.rs:18584`（decode, rows=1） | verify `chain_dev.rs:15218`（m 行批） | 同? |
|---|---|---|---|---|
| 1 | `a` | `self.s.xq4.as_u8()` | `self.s.xq4_r.as_u8()`（`:15217`） | 形状同，对象不同 |
| 2 | `a_scale` | `self.s.xsc4.as_f32()` | `self.s.xsc4_r.as_f32()` | 同上 |
| 3 | `out` | `self.s.ex_act_b.ptr` | `self.s.ex_act_r.ptr` | 同上 |
| 4 | `out_slot_stride` | `act_slot`（`:18572`） | `act_slot`（`:15049`） | ✅ 值同（见下） |
| 5 | `rows` | **`1`** | **`m`** | ❌ **唯一参数差异** |
| 6 | `dim` | `dim` | `dim` | ✅ |
| 7 | `inter` | `inter_local` | `inter_local` | ✅ |
| 8 | `limit` | `cfg.swiglu_limit` | `cfg.swiglu_limit` | ✅ |
| 9 | `slots` | `topk` | `topk` | ✅ |
| 10-17 | `w1/w3 (+s) base,stride` | `ld.experts[0]`/`[1]` 指针差（`:18343`） | 同式（`:15014`） | ✅ 同式同值 |
| 18 | `ids` | `self.s.route_idx.ptr`（`:18492`） | `self.s.route_idx_r.ptr`（`:15179`） | 布局同为 `[row][slots]`，对象不同 |
| 19 | `ilv` | `ilv = ld.experts_ilv`（`:18321`） | `ld.experts_ilv`（`:15237`） | ✅ 同字段 |
| 20 | `act_e4m3` | `e4m3`（`:18449`） | `e4m3`（`:15041`） | ✅ 同表达式 |

**`act_slot`（第 4 项）逐字对照**（生产 `dim=5120`、`mode=2`、`fuse` 默认 ON → 两侧都 `= inter_local`）：

```rust
// eager  :18568 / :18572   (★ 少 dim%512 项)
let gateup_fused = !ran_tc && gateup_fuse() && self.dev.supports_gateup_fuse()
                   && expert_fp4_mode() == 2;
let act_slot = if gateup_fused { inter_local as i64 } else { (2 * inter_local) as i64 };
// verify :15045 / :15049   (含 dim%512 项)
let gateup_fused = gateup_fuse() && self.dev.supports_gateup_fuse()
                   && expert_fp4_mode() == 2 && (dim % 512) == 0;
```

**同一个表达式的第二个副本**（eager `:18661`，用于跳过 swiglu）**含** `dim % 512 == 0`。
⇒ eager 侧存在**同一函数内两个不同的 `gateup_fused`**，`dim % 512 != 0` 时 `act_slot` 与 launcher 自己的
`fuse`（`:3565` 含 `dim % 512 == 0`）**不一致** → 见 §2.3（生产 `dim=5120` 不触发，属潜伏）。

**布局侧同样逐位相同**：`out` 行距 = `slots*out_slot_stride`、`ids` 行距 = `slots`、`a` 行距 = `k/2`
（内核 `:1365-1373` 的 LAYOUT CONTRACT 由 `gridDim.y` + 已有参数推导），两侧 `out` 行距都 = `6*320`。

**结论（交付②）：两臂没有配置差异。** `vec` 不可能不同 —— 它是 `.so` 内的 `g_expert_fp4_mode`（一个进程级
`static`），两个调用点都够不着它；`ilv` 同字段；`fuse` 生产同值。「m=1 侧走了 vec=3」在
**batched launcher 内没有对应开关**。

---

## 2. 候选 1 的真身：两条**实际可达**的分派不对称（都不是免费收益）

### 2.1 `rows == 1` 会被改派到**另一个内核**（`.cu:3274` / `:3412`）

```cpp
// launch_mxf4 / launch_mxf4_indirect —— 注意：不在 batched launcher 里
if (rows == 1 && getenv("DSV41_NO_GEMV_FP4") == nullptr) {
    expert_gemv_fp4_kernel<<<blocks, 8*32, k*4, s>>>(...);   // ← 另一支：48 regs, 有自己的 vec 表
    return cudaGetLastError();
}
```
`expert_gemv_fp4_kernel` **就是** `.cu:900-914` 那张「1.25 L1TEX op/value vs 2.5」表的宿主，
即 **「条 L1TEX/value」这个指标的定义处**；它**没有 rows/slot 网格维**（`grid=(ceil(n_total/8),1,1)`）。
`load.rs:969-983` 的 `ilv_ok` 又把它接到**载入期布局**：`DSV41_NO_GEMV_FP4` 一设，池子就不交织。

### 2.2 两臂对 `moe_batch()` 的态度**不同**（`chain_dev.rs:18315` vs `:15216`）

```rust
// moe()  —— 查
let batched = moe_batch() && topk > 0 && ne >= 2 && self.dev.supports_moe_batch();   // :18315/:18483
// moe_rows() —— 不查！只有 grp_gu 能顶掉它
if !grp_gu { self.dev.expert_gate_up_fp4_batched(...) }                             // :15216
```
⇒ **`DSV41_MOE_BATCH=0` 时**：eager 掉进逐-slot 顺序循环 → `expert_gate_up_fp4_indirect` →
`rows==1` → **`expert_gemv_fp4_kernel`**；而 verify **照旧** 走 `expert_gemv_fp4_batched_kernel<ILV,1>`。
**两条臂跑的是两个不同的内核** —— 这正是任务书候选 1 的字面意思（「别的 decode 形态」），
而且是**唯一**能给出「每-value 指令比」1.66× 种子的机制。

**它为什么不是免费收益**：`expert_gemv_fp4_kernel` 的网格里**没有 rows 维**（`blocks = ceil(n_total/8)`，
一个 CTA 一批输出行，没有多激活行维），所以 rows>1 结构上无法复用它；要「对齐」就得给
`expert_gemv_fp4_kernel` 造一个 rows 变体（= 新内核 + 新 parity），不是改一个参数。

### 2.3 ⚠️ 上报：eager 的 `act_slot` 潜伏事故（`:18572` vs `:18661` vs launcher `:3565`）

`dim % 512 != 0` 且 `DSV41_GATEUP_FUSE` ON 时：eager 传 `act_slot = inter`（融合 pitch），
而 launcher 的 `fuse` 归 0（`:3565` 带 `dim%512==0`）→ 内核写**原始 `[2*inter]` 对**到 `inter`-pitch 的槽里，
越界进下一个 slot；同时 `:18661` 的 `gateup_fused == false` 让 host **仍然**跑 swiglu 并按下调的
`act_slot` 读 —— **静默错答**。`moe_rows` 侧无此问题（`:15045` 带该项，且注释明确写了
「the single-row call site omits」）。生产 `dim=5120` 不触发，**未改代码，交尚书省裁决**。

---

## 3. 计量学：为什么 1.66× **不可能**是 grid.z 的几何效应

### 3.1 `条/value` 是**每-lane 指令比**，且手算正好 0.88

用内核自己的口径（＝ `.cu:900-914` 定义 `mode 3 = 5 ops/4 values = 1.25` 的同一把尺），
生产 pair body（ILV, fuse, ksplit=2）每 lane 每 k-group：

| 指令 | 条数 | 来源（`.cu` 行） |
|---|---|---|
| `LDG.128`（gate+up 一对） | 1 | `:1792` |
| `LDG.U8`（`gsc`, `usc`） | 2 | `:1748`, `:1858` |
| `LDS.32`（`sa[0..15]`，两链共用一次） | 16 | `:1746` |
| `LDS.64`（`s_lut2` 4+4） | 8 | `:1832-1847` |
| **小计** | **27** | |

value 数：一个 group = 512 B = 1024 fp4 value，/32 lane = **32 value/lane**。
→ 权重环 **27/32 = 0.844**。
加按 CTA 摊薄的前导：LUT build 256 STS.64 + 激活 staging `k/32=160` 次 `(LDG.128+LDG.32+16×STS.32)`
= 2880 → `+3136`；CTA 值数 = 8 输出行 × 10 240 value = 81 920
→ **(16×5×32×27 + 3136)/81 920 = 0.882 条/value**。

**与 eager 的 0.88 三位吻合** ⇒ 「条/value」是 **L1TEX 指令 / fp4 value 的指令混比**，
而 eager 的 0.88 **就是生产 pair body 自己的数**。

### 3.2 占用反证（候选 3 被否定）

远端 `ptxas -v`（本源码，sm_103a，CUDA 13.2）：

| 内核 | regs | 备注 |
|---|---|---|
| **`expert_gemv_fp4_batched_kernel<true,1>`（生产 ILV）** | **64**（0 spill） | = `__launch_bounds__(1024)` 的硬顶 |
| `expert_gemv_fp4_batched_kernel<false,1>` | 62 | |
| `expert_gemv_fp4_kernel`（§2.1 另一支） | 48 | |
| `expert_gemv_fp4_gate_up_grouped_kernel` | 63 | |
| `mxf4_gemm_kernel<true|false>` | 62 / 56 | 25 616 / 24 592 B **static** smem |

生产 `blockDim = rows*ksplit*32 = 8*2*32 = 512`，`smem = dim*4 + 256*8 + nwarps*8 + nwarps*512 = 30 848 B`：
寄存器限 65536/(512·64) = **2 CTA/SM**（线程限 4，smem 限 ≥7）⇒ 32 warps/SM。

| | CTA 总数 | resident（2×148） | 平均 CTA/SM |
|---|---|---|---|
| rows=1 | 240 | 240（**全部在飞**） | **1.62** |
| rows=6 | 1440 | 296（4.87 波） | 2.00 |

**m=1 的占用更低（1.62 < 2.00）、SM 都没填满，却是快的那一臂** ⇒
「占用 / smem carveout」这条（候选 3）与观测**反向**。任务书里的 `63 regs / 22.5 KB / 3.24 block/SM`
对应的是 `ksplit=1 + cpasync=off` 的 **非生产形态**（22 528 = `dim*4+2048`，既无 pf ring 也无 `s_ks`），
不能拿来推生产占用 —— 这是候选 3 前提里的一处口径漂移。

### 3.3 足迹反证（候选 2 也被大幅削弱）

resident 窗口内的「不同专家数」= resident CTA / `ctas_x` = 296/40 = **7.4**（rows=6）
vs 240/40 = **6**（rows=1）→ **只有 ×1.23**，不是 ×6（因为每个 (slot,arow) 的 40 个 CTA 已经天然成团）。
再叠上 §3.1：**指令比是 `arow` 无关的**（内核 `:1379-1384` 的 ROW INDEPENDENCE 明写
「the only thing that changes with `arow` is a base pointer」）。

⇒ 候选 2/3 都无法单独给出 1.66×。**剩下的只有「两臂指令流不同」**（§2），
或 0.88/1.46 这两个数**不是同一把尺量的**。**NCU 必须同时给指令列与 wavefront 列来分开这两种读法。**

---

## 4. 交付④ NCU 微基准设计（主 agent 跑；文件已就位）

**可编译就位的探针（新文件，不动任何现有脚本）**：
`scripts/expert_mrows_l1tex_bench.cu`
（链接生产 `.so`；960 MiB 级专家池保持冷流；`fill_rand` 让 LUT 索引数据相关；
`cudaProfilerStart/Stop` 窗口配 `ncu --profile-from-start off`；把 ILV 的 per-expert 块布局
逐字节复制自 `load.rs::load_expert_pool`）。

### 4.1 双因子表（这是本设计的要点：把 `rows` 与 `footprint` 解耦）

| 档 | `rows` | `DSV41_BENCH_IDS` | 窗口内专家数 | 判定 |
|---|---|---|---|---|
| A | 1 | distinct | 6 | 基线（应复现 0.88 / ~700 GB/s） |
| B | 3 | distinct | ~7.4 | |
| C | 6 | distinct | ~7.4 | **生产 verify 形态**（目标：377 GB/s / 1.46） |
| D | 6 | **same** | **6**（= rows=1 的足迹，但 grid 仍是 (40,6,6)） | **判决点** |
| E | 1 | `indirect` 模式 | 1 slot/发 | 探 `expert_gemv_fp4_kernel`（§2.1）的指令比 |

* **D ≈ A ⇒ 代价纯粹是足迹（候选 2）** → 走 §5 的局部性改法，几何本身无辜。
* **D ≫ A 且 D ≈ C ⇒ 足迹常量化后代价仍在 ⇒ 是 grid.z/占用（候选 3）** → 走 smem carveout / CTA 形状。
* **E 的指令比若 ≈ 0.88 而 C 是 1.46 ⇒ 1.66× 的两个数来自不同内核/不同尺，候选 1 成立。**

### 4.2 三列 + 判别列（ncu `--metrics`，文件头已给完整命令行）

要求的三列：`smsp__issue_active.avg.pct_of_peak_sustained_elapsed`（issue_active）、
`l1tex__throughput.avg.pct_of_peak_sustained_elapsed`（l1tex throughput）、
`smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct`（warp_issue_stalled）。

**必须一起抓的判别列（用来分开「指令比」与「miss 膨胀」两种读法）**：
```
smsp__sass_inst_executed_op_global_ld.sum, smsp__sass_inst_executed_op_shared_ld.sum,
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,
l1tex__data_pipe_lsu_wavefronts_mem_global.sum, l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum, l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,
l1tex__t_sector_hit_rate.pct, lts__t_sector_hit_rate.pct,
dram__throughput.avg.pct_of_peak_sustained_elapsed, gpu__time_duration.sum,
launch__registers_per_thread, launch__shared_mem_per_block_dynamic,
launch__occupancy_limit_registers, launch__occupancy_limit_shared_mem,
launch__grid_size, launch__block_size, launch__waves_per_multiprocessor
```
`条/value` = `smsp__sass_inst_executed_op_{global,shared}_ld.sum` 之和 / `(rows*6*320*5120)`；
`wavefront/value` 用 `l1tex__data_pipe_lsu_wavefronts_*` 同除。
**两者分开动 = 指令流不同（候选 1）；只有后者动 = miss 膨胀（候选 2）。**

### 4.3 必须钉住的 env（否则数字不迁移；`.so` 全部一次性 cache）

```
DSV41_GATEUP_KSPLIT=2 DSV41_GATEUP_ROWS=8 DSV41_GATEUP_CPASYNC=1 DSV41_GATEUP_PIPELINE=1
DSV41_EXPERT_ILV=1 DSV41_EXPERT_FP4_MODE=2 DSV41_GATEUP_FUSE=1 DSV41_MOE_BATCH=1 DSV41_PDL=0
```
（`DSV41_PDL` 默认 OFF，但显式写；`EXPERT_NCU_ITERS` 之类不改窗口外的东西。）
⚠️ 顺手报一个既有缺陷：`scripts/expert_ncu_bench.cu:169-175` 仍在声明 **19 参旧 ABI**，
而 `.so` 现在是 **21 参**（尾部多了 `int act_e4m3`）—— 它会拿 `stream` 当 `act_e4m3` 传。
新文件用的是 21 参版本，并在头部写明这条。

---

## 5. 交付⑤ L2 局部性的改法设计（三档，按代价排序）

### 5.1 【零代码，先跑】`DSV41_GATEUP_KSPLIT=4 | 8` —— 直接缩小并发专家数

resident 窗口内的专家数 = `9472/(ksplit·n_total)`（与 `warps` **无关**，`warps` 在分子分母相消）：

| ksplit | blockDim | resident CTA | 窗口内专家数 |
|---|---|---|---|
| 2（生产） | 512 | 296 | **7.4** |
| 4 | 1024 | 148 | **3.7** |
| 8 | — | — | （`warps*ksplit ≤ 32` 卡在 4） |

⇒ **`DSV41_GATEUP_KSPLIT=4` 把并发专家区**（唯一能压的足迹量）**砍半**。零代码、可立刻 A/B。
代价：`ksplit>1` 改求和序（`(g0..g4)+(g5..g9)`，~1e-7/layer，需文本 A/B），且 `blockDim=1024` 顶到
`__launch_bounds__(1024)` 的 64-reg 硬顶。**若 R4 的 `rows=6/ksplit=4` 的 条/value 掉回 0.88 附近并涨带宽，
候选 2 成立，且这是最便宜的兑现方式。**

### 5.2 【bit-identical，小改】`blockIdx.x` 按 `blockIdx.z` 错位（skew）—— 打散跨行 L2 set 别名

现状：`row_base = blockIdx.x * rows_per_cta + row_local`（`:1499`）。
`rows` 行的 CTA 在同一时刻处理**各自专家内的同一偏移**；专家 stride 是固定大常数，
6 行同时打同一批 L2 set → 别名冲突（m=1 无此现象）。

```cpp
// dsv41_experts_mxf4.cu:1499 附近（pair_body 与 split body 共用的唯一行映射）
const int xb = (gridDim.z > 1)
    ? (int)((blockIdx.x + (unsigned)blockIdx.z * ((gridDim.x + gridDim.z - 1) / gridDim.z))
            % gridDim.x)          // skew = ceil(ctas_x / rows)
    : (int)blockIdx.x;
const int row_base = xb * rows_per_cta + row_local;
```
* **数值无关**：每个 `(x, slot)` 的行块仍被覆盖恰好一次（`xb` 是 `[0,ctas_x)` 上的置换），
  行之间无共享输出/累加器/smem（内核 ROW INDEPENDENCE）→ **bit-identical**。
* `rows==1` 时 `xb == blockIdx.x`，**旧路径逐位不变**。
* 需要同改 `:1501` 的 `g_begin`/`g_end`？不用 —— 它们只依赖 `half/nv2f/ksplit`。
* 风险：若 `ctas_x % rows != 0`，skew 步长要取 `ceil`（式子里已取），置换性质保持。
* **在 §4.1 的 D vs C 对比里，这个改动是直接的 A/B 候选。**

### 5.3 【设备端置换，中改】按专家 id 聚类 (slot,arow) 分派

只有当「同一专家被不同行命中」确实存在时才有收益（生产 36 个 assignment / 384 专家，
期望重复 ≈ 36²/2/384 ≈ 1.7 对）。做法：m×topk ≤ 8×8 = 64 项，一个 CTA 的**稳定计数排序**
（`dsv41_moe_row_perm`，新符号）产出 `perm[slot*rows + arow]`；内核把
`arow = blockIdx.z` 读成 `arow = perm ? perm[slot*gridDim.z + blockIdx.z] : blockIdx.z`（多一个 ABI 尾参）。
同样是纯置换 → bit-identical。**代价/收益比不如 5.1/5.2，建议排在 NCU 之后。**

### 5.4 【否决】候选 3 的 smem carveout

§3.2 已证 `rows=1` 在**更低**占用下更快 ⇒ carveout 调整没有可解释的作用面；
`cudaFuncAttributePreferredSharedMemoryCarveout` 不是这一刀。

---

## 6. 交付⑥ 编译验证

| 项 | 命令 | 结果 |
|---|---|---|
| cargo | `cargo check -p ferrite-dsv41` | ✅ `Finished`，0 error（仅既有 dead_code warning） |
| nvcc（生产 TU） | 远端 `nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a -Xptxas -v -c dsv41_experts_mxf4.cu` | ✅ RC=0，0 error；regs 表见 §3.2 |
| nvcc（新微基准） | 远端 `nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a -c scripts/expert_mrows_l1tex_bench.cu` | ✅ `NVCC_RC=0`，产出 37 848 B `.o` |

**回归护栏**：`cargo check -p ferrite-dsv41` 与生产 TU 的 nvcc 均 RC=0（改动前基线同）；
本任务**未触碰** `chain_dev.rs` / `dsv41_experts_mxf4.cu`，工作树里那两个 M 是 peer 的改动。

**未改任何生产代码**：§1 判定「无配置差异」→ 任务书第 3 步（若发现差异则修）不适用；
§5 的三档是**设计**（任务书第 5 步要求「改法设计」），其中 5.1 是 env A/B、5.2 是 bit-identical 小改，
都留待 NCU 指向后由尚书省定夺再施工。

---

## 7. 交尚书省的条目（不是我该改的）

1. §2.3 潜伏事故：eager `act_slot`（`:18572`）缺 `dim % 512 == 0`，与 launcher `fuse`（`:3565`）
   及同函数的 `:18661` 不一致 —— `dim%512!=0` 时静默错答。生产 `dim=5120` 不触发。
2. §2.2 不对称：`moe_rows` 不查 `moe_batch()` 而 `moe()` 查 —— `DSV41_MOE_BATCH=0` 会让 verify 与
   decode 跑两个不同内核（前者没有 fallback 顺序路径）。是否有意为之，需裁决。
3. §4.3 既有脚本 ABI 过期：`scripts/expert_ncu_bench.cu` 19 参 vs `.so` 21 参。
4. 任务书里 `63 regs / 22.5 KB / 3.24 block/SM` 对应非生产形态（§3.2），后续引用需换生产口径。
