# TileLang MoE grouped GEMM 接线（工部 · 工程实现）

> 上位原型：`docs/agent/tilelang-moe-grouped.md`（GPU 实测 bf16 臂 **70.0µs/层** = up 45.5 +
> dn 24.5 = SIMT 250µs 的 **28.0%**，判据 <40% 通过；fp4 原地 dequant 判死）。
> 第一阶段模式来源：`kernels/cuda/tilelang_gen/wkv_shim.cu` + `PROVENANCE.md`。
> 本文档 = **接线契约 + 显存预算 + GPU 验证手册**。生成物/出处见
> `kernels/cuda/tilelang_gen/PROVENANCE.md` §7。
>
> 工部 · 2026-09-13 · 本阶段**未跑 GPU e2e、未跑完整 build.sh**（主 agent 职责）；
> 已做：单文件 compile-only（EXIT=0）+ `cargo check`（EXIT=0）。

---

## 1. 交付物与改动文件

| 文件 | 性质 | 改动 |
|---|---|---|
| `kernels/tilelang/gen_moe_aot.py` | 新增 | MoE AOT 生成器（裸指针 ABI，照 `gen_wkv_aot.py`）|
| `kernels/cuda/tilelang_gen/moe_up_tl.cu` | 生成物（远程 dump）| up grouped GEMM，N=640 K=5120 |
| `kernels/cuda/tilelang_gen/moe_dn_tl.cu` | 生成物（远程 dump）| down grouped GEMM，N=5120 K=320 |
| `kernels/cuda/tilelang_gen/moe_tl_config.txt` | 生成物 | 冻结几何 + 签名 |
| `kernels/cuda/tilelang_gen/moe_bf16_shim.cu` | **新增（手写）** | launcher shim + gather/scatter/dequant，3 个导出符号 |
| `kernels/cuda/tilelang_gen/PROVENANCE.md` | 改 | §1 表 + 新 §7 |
| `crates/ferrite-models/src/dsv41/device.rs` | 改 | 3 个 `Option<fn>` 字段 + `ko!` + 3 个 wrapper + 2 个 `supports_*` 探测 |
| `crates/ferrite-models/src/dsv41/chain_dev.rs` | 改 | `moe_tilelang()` / `moe_bf16_dequant()` gate、`moe_align_host` 纯函数、双臂（verify/eager）|
| `crates/ferrite-models/src/dsv41/load.rs` | 改 | `DevExpert::{up_bf16,dn_bf16}` + `expert_bf16_pool` + 加载期 dequant；`ilv_ok` 互斥项 |

`build.sh` **未改**：第一阶段已加 `tilelang_gen/*_shim.cu` 的 glob，新 shim 自动进 `SRCS`
⇒ 自动进 `CU_HASH` ⇒ 进 `BUILD_ID`（same-source gate 免改一行）。

---

## 2. 数据流与布局契约

```
route_idx_r[m][topk] (device, route_topk 写)
        │  D2H 回读（m*topk i32）
        ▼
moe_align_host(idx, topk)  ── 纯函数：稳定按 expert 排序 + 排他前缀和
        │
        ├─ order[SEG_CAP*BM]  assignment 下标（row*topk+slot），pad = -1
        ├─ counts[SEG_CAP]    段的 live 行数（pad 段 0）
        ├─ eid[SEG_CAP]       段的 expert id（pad 段 0）
        └─ nseg
        │  （shim 内部 3 次小 H2D 上行到常驻 scratch）
        ▼
dsv41_moe_tilelang_gate_up_bf16
   gather: xn_r[rows, dim] f32 ──bf16──▶ A[SEG_CAP*16, dim]，每段 pad 到 16 行
   MMA   : grid (3, 36) × 256th, smem 104448 → C[SEG_CAP*16, 640] f32
   scatter: C ──▶ ex_act_r[row][slot][2*inter]  （**RAW gate‖up**）
        ▼
dsv41_swiglu_limit_batched（既有，未改；本臂把 gateup_fused 强制为 false）
        ▼
dsv41_moe_tilelang_down_bf16
   gather: ex_act_r（槽距 act_pitch = 2*inter）──bf16──▶ A[SEG_CAP*16, inter]
   MMA   : grid (10, 36) × 256th, smem 135168 → C[SEG_CAP*16, 5120] f32
   scatter: C ──▶ ex_down_r[row][slot][dim]
        ▼
dsv41_moe_down_reduce（既有，未改；固定升序 slot 合并 —— 数值契约不变）
```

**关键设计选择：本臂只替换 GEMM，不替换 swiglu / reduce。** shim 写 RAW gate‖up，
swiglu 与 down-reduce 仍是既有 kernel。这样：

1. 改动面最小（下游 buffer/偏移/数值契约一字未动）；
2. A/B parity 可检查（§5 门 1：与 `DSV41_MOE_TILELANG=0` 的对应中间量逐元素比）。

**权重布局**（`DSV41_MOE_BF16_DEQUANT=1` 产出，每层一块连续池）：

| 池 | 布局 | 每专家字节 | 下标 |
|---|---|---|---|
| `up_bf16` | `[E, 2*inter_local, dim]` bf16 | `2*320*5120*2` = 6.55 MB | 行 `[0,320)` = w1/gate，`[320,640)` = w3/up |
| `dn_bf16` | `[E, dim, inter_local]` bf16 | `5120*320*2` = 3.28 MB | w2 |

shim 用 `base + e*N*K` **算术**推每个专家的基址，所以池必须严格规则；
`chain_dev.rs::moe_tilelang_weights` 在每次发射前校验相邻专家的 stride 恰为 `N*K*2`
字节，不满足就 decline（非均匀池会变成静默错值，不是 fault）。

⚠️ **N 内部次序 = [gate ‖ up]** 是本接线选定的约定。原型用随机权重，无法反推；若实际
权重按 [up ‖ gate] 装载，parity 门会给出**明显错值**（不是细微差异），不会漏检。

---

## 3. 门与回退（三态门禁）

| 门 | 默认 | 语义 |
|---|---|---|
| `DSV41_MOE_TILELANG=1` | **OFF** | 派遣 up + down grouped GEMM 臂 |
| `DSV41_MOE_BF16_DEQUANT=1` | **OFF** | 加载期把专家 fp4 展开成 bf16 副本（§4）|

**域检查**（`chain_dev.rs::moe_tilelang_ready`），任一不满足 ⇒ decline 回既有路径 +
一次性提示（armed 的臂绝不静默测老路 —— 本项目 #1 测量偏置陷阱）：

- `.so` 同时有 up/down 两个符号（`supports_moe_tilelang`）；
- **不在** CUDA-graph capture 内（`dev.capturing()`，见 §6）；
- 冻结形状：`n_routed == 384`（生成物的 `e < 384` 守卫是 bake 的）、`dim == 5120`、
  `inter_local == 320`、`topk ∈ [1,6]`、`rows ∈ [1,6]`；
- 两个 bf16 池存在且 stride 规则。

**双侧同换**（契约）：verify 侧 `moe_rows`（`m` 行，`ex_act_r`/`ex_down_r`）与 eager 侧
`moe()`（单行，`ex_act_b`/`ex_down_b`）走**同一份** gate、**同一份** `moe_align_host`
与**同一对** shim 符号；单行只是多行布局的 `rows == 1` 实例，因此两侧 bit-for-bit 是
同一个计算。任何一侧 decline，另一侧也 decline（同一 `moe_tilelang_ready`）。

---

## 4. 显存预算（⚠️ 需拥有者拍板）

见 `PROVENANCE.md` §7.5。摘要：**ferrite 的 loader 不做专家并行**（每 rank 持全部 384
专家，只 TP 切 `inter`），所以 bf16 副本的增量 ≈ **+105~113 GiB/rank**，而不是原型文档
§4.3 的「+15GB/rank」（那个数字隐含 8 路专家分片）。

- **接受** ⇒ bf16 臂（本 shim）即最终形态；
- **不接受** ⇒ 唯一替代是 **tcgen05 blockscaled 原生 fp4 MMA**（原型 §4.3-2 / §8-1；
  另一路 subagent 在探索）。⚠️ 不要退回「读 fp4 → 内核展开 bf16 → MMA」（已判死）。

---

## 5. GPU 验证手册（主 agent）

### 前置

```bash
cd kernels/cuda && bash build.sh 103a          # 完整 build（主 agent）
# 产物 .so 应含：nm -D libferrite_kernels.so | grep dsv41_moe
#   dsv41_moe_tilelang_gate_up_bf16 / dsv41_moe_tilelang_down_bf16 / dsv41_moe_fp4_to_bf16
```

构建前先确认新 shim 已被 glob 纳入：`bash build.sh 103a` 的输出里
`from .../tilelang_gen/moe_bf16_shim.cu` 应出现在 SRCS 列表中。

### 门 0 — ARMED 回执（先证明「ON 的臂真的跑了新程序」）

```bash
DSV41_MOE_TILELANG=1 DSV41_MOE_BF16_DEQUANT=1 <serve> 2>&1 | grep moe-tilelang
# 期望（每个符号各一次）：
#   [moe-tilelang] ARMED gate_up rows=.. dim=5120 inter=320 topk=6 nseg=.. -> grid=(3,36)x256 smem=104448 + gather/scatter
#   [moe-tilelang] ARMED down    rows=.. dim=5120 inter=320 topk=6 nseg=.. -> grid=(10,36)x256 smem=135168 + gather/scatter
# 任一只出现 "ARMED but REFUSED/skipped" ⇒ 该 run 测的是老路径，不要采信时间。
```

### 门 1 — parity（数值门，先于性能门）

在**同一台机、同一权重、同一路由**上跑两次，比对**对应中间量**：

1. `DSV41_MOE_TILELANG=0`（老路径）→ dump `ex_act_r`（swiglu 后）与 `ex_down_r`；
2. `DSV41_MOE_TILELANG=1 DSV41_MOE_BF16_DEQUANT=1` → 同样 dump。

判据：

| 量 | 期望 |
|---|---|
| `ex_act_r`（swiglu 前，RAW gate‖up）| 与老路径**同阶**；差异只来自 bf16 权重/激活量化（原型 §7：up `max|err|` 2.0e-5 vs fp32）|
| `ex_down_r` | 同上 |
| `moe_out_r`（reduce 后）| 相对误差 < 1e-2（bf16 输入量化的预期量级）；**不得**出现 1e2 级或 NaN |
| 路由表 | `moe_align_host` 是纯函数：同一 `route_idx_r` 重复调用必须逐位相同（可单独单测）|

⚠️ 若 `ex_act_r` 的差异**远大于** bf16 量级（例如整段错位），先查 §2 的
`[gate ‖ up]` 次序约定与 `wq_pitch/ws_pitch`（`DSV41_SF_STRIDE_PAD` 默认 ON，
`w2.scale` 的物理行距是 16 B 而不是 10 B）。

**门 1 不过就不看门 2。**

### 门 2 — 端到端（双门禁）

```bash
# (a) 只开 dequant（证明它本身不改行为：老路径仍读 fp4，bf16 池只是多占显存）
DSV41_MOE_BF16_DEQUANT=1 DSV41_MOE_TILELANG=0 <serve>   # 输出应与两个门全 OFF 完全一致（逐位）
# (b) 两个门都开
DSV41_MOE_TILELANG=1 DSV41_MOE_BF16_DEQUANT=1 <serve>
# (c) 对照
DSV41_MOE_TILELANG=0 DSV41_MOE_BF16_DEQUANT=0 <serve>
```

判据（b vs c）：40 层 MoE 的段总时间应从 ~250µs/层 降到 ~70µs/层（原型 §3.3），
**且输出不退化**（同一 prompt 的 argmax 序列一致；接受率不下降）。

### 门 3 — eager vs verify 的同位性

同一权重下，单行路径（`moe()`）与 m 行路径（`moe_rows()`）在 `m == 1` 时应给出**相同
bits**（`ex_act_b` vs `ex_act_r` 的 row 0）。这是「双侧同换」契约的直接检验。

---

## 6. 已知限制 / 后续项（按 ROI）

1. **EAGER-only**：`moe_align` 在 host，需要 D2H 回读 `route_idx_r`，capture 内非法
   （`dev.capturing()` 时本臂 decline）。**把 `moe_align` 摊到 GPU 侧**（SGLang 的
   `moe_align_block_size` CUDA kernel 可直接借语义）是进 graph 的前提（原型 §8-3）。
2. **显存**：§4，需拍板；替代 = tcgen05 blockscaled fp4。
3. **`NSEG` 上界 36**：真实 `nseg` 逐位变化，`grid.y` 固定 36 ⇒ ≈3% 空转；进 graph 时
   可与 device-side align 一起重生成（把 `SEG_CAP` 换成实际值不现实，因为它是运行时量）。
4. **down 的反向 tile**：down 是 `N=5120, K=320`（K 很短），原型 §9-4 建议 up/down 共享
   A/B staging 再省一次 smem 往返 —— 换形态要重生成 + 重跑门 1/2，独立一轮。
