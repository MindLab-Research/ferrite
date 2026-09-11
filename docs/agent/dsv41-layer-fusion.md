# DSV4.1 每层 tile 对齐融合算子 — 设计

**用户指令（2026-09-11）**："做算子融合，我要每层一个 tile 对齐的融合算子"；追问后裁定：
**① 三段一起做（不分段交付）② tile 由 smem 容量反推**。

**动机（实测，非估计）**：更正后的剖析（见 `STATUS.md` 的分解节）显示每步每卡
**~700 次 kernel 启动 ≈ 17.5 次/层** ✗，且单核实测 ~40µs —— 其中绝大部分是**核固定成本
＋中间量落回 global 再读回** ✗，不是算术 ✓。三大件即占 12.5 次/层、26.1ms/32ms。

---

## 1. 硬约束：每层 2 处 all-reduce（物理边界，非取舍）

| AR | 位置 | 出处 | reduce 缓冲 | 字节数 |
|---|---|---|---|---|
| **AR#1** | `attention()` 里 wo_b 的 `lin` **之后立即**（steps 16 → 17）| `chain_dev.rs:1434-1437` | `s.o` `[1,dim]f32`（wo_b 的 RowParallel 输出）| `fb(dim)` = **20480 B** |
| **AR#2** | `layer()` 里 `moe()` **之后、`hc_post` 之前** | `moe_reduce()` `chain_dev.rs:638-646`（调用点 `:1096`）| `s.o` `[1,dim]f32`（routed + 共享专家已 add）| **20480 B** |
| （额外）| `step_body()` 块前，仅 engram 层（1、14）| `:596-599` | `s.eng_rows` | 非每层 |

两处 AR 均为设备侧 v5（`Collective::all_reduce_inplace`），**跨 rank 集合通信无法放进 tile 对齐的核内** ✗
⇒ 严格"每层 1 个算子"不存在 ✓，可行形态 = **每层 3 段** ✓：

```
段A: hc_mixes → hc_collapse → rmsnorm → attention(投影/rope/ring/compress/indexer/sparse_attn/o_proj)
  ── AR#1 (s.o, 20 KB) ──
段B: hc_post → copy_h_back → hc_mixes → hc_collapse → rmsnorm → MoE(gate/route/专家/共享专家)
  ── AR#2 (s.o, 20 KB) ──
段C: hc_post → copy_h_back
```

---

## 2. 单层算子清单（精确，带出处）

模型实参（`crates/ferrite-models/configs/dsv41_flash.json`）：`dim=5120, hc=4, head_dim=512, n_heads=64,
rope_head_dim=64, q_lora_rank=1280, o_lora_rank=1024, o_groups=8, window_size=128, index_n_heads=32,
index_head_dim=128, index_topk=512, moe_inter=2304, n_routed_experts=384, topk=6, 40 层`。
TP8 ⇒ `nlh = 8`、`inter_local = padded(2304/8)`、`ol_local = 128`、`nlg = 1`、`hpg = 8`。

### 段 A（`layer()` :973-1016 → `attention()` :1154-1439）

| # | 调用 | 出处 | 形状 |
|---|---|---|---|
| 1 | `hc_mixes` | :973 | `[1,hc*dim]f32` → pre_b `[1,hc]`、post `[1,hc]`、comb `[1,hc*hc]`；smem **160 B** |
| 2 | `hc_collapse` | :1000 | `[1,hc*dim]` → `[1,dim]`；grid=(rows*dim+255)/256 |
| 3 | `rmsnorm` | :1008 | `[1,dim]` |
| 4 | `lin wq_a`（=quant1+`gemm_fp8_mx`, m=1 ⇒ **GEMV**）| attention :1166 | k=5120 → n=1280 |
| 5 | `rmsnorm`（q_norm，原地）| :1174 | `[1,1280]` |
| 6 | `lin wq_b` | :1182 | k=1280 → n=`nlh*hd`=4096 |
| 7 | `apply_rope` | :1196 | rows=nlh, dim=rd=64, half=32 |
| 8 | `lin wkv` | :1215 | k=5120 → n=hd=512 |
| 9 | `rmsnorm`（kv_norm，原地）| :1223 | `[1,512]` |
| 10 | `apply_rope` | :1231 | rows=1 |
| 11 | `ring_append`（仅 owner 层）| :1284 | `[hd]` → ring |
| 12 | `window_idxs`（每步无条件）| :1325 | `idxs[win+index_topk+8]i32` |
| 13 | `compress()`（kv-source 层）| :1308-1314 → :1589-1649 | pool + commit 两核 |
| 14 | `indexer()`（仅 8 个 source 层）| :1332 → :1446-1584 | idx_k（512→128）+ norm + rope + publish + idx_wq_b（1280→4096）+ rope + idx_weights（5120→32）+ `indexer_topk`（smem **26688 B**，与 per-step `n_pos` 解耦 ✓）|
| 15 | `comp_placeholder`（无 indexer 时）| :1340 | 写 `idxs[win..]` |
| 16 | `sparse_attn` | :1348 | b=1,m=1,h=nlh,d=512, window=128, index_topk=512 → `s.o` |
| 17 | `apply_rope`（**反向**）| :1363 | rows=nlh, inverse=true |
| 18 | `quant1`（o 量化）| :1392 | 长度 `nlh*hd`=4096 |
| 19 | `gemm_fp8_mx` ×`nlg`（分组 o 投影）| :1409 | k=`hpg*hd`=4096, n=`olg`=1024 |
| 20 | `lin wo_b` | :1426 | k=`ol_local`=128 → n=5120 |
| 21 | **AR#1** | :1434 | `s.o`, 20480 B |

### 段 B（`layer()` :1017-1096 + `moe()` :2147-2593；行号 2026-09-11 复核：`fn moe` = 2147，`fn layer` = 1295，`fn attention` = 1567）

| # | 调用 | 出处 | 形状 |
|---|---|---|---|
| 22 | `hc_post`（注意力）| :1017 | `[1,hc*dim]` → `s.h2` |
| 23 | `copy_h_back` | :1027 | **纯 D2D memcpy**，`hc*dim*4` = 80 KB |
| 24 | `hc_mixes`（ffn）| :1034 | 同 #1，写 premix_slot(2) |
| 25 | `hc_collapse`（用 premix_slot(1)）| :1053 | 同 #2 |
| 26 | `rmsnorm`（ffn_norm）| :1061 | `[1,dim]` |
| 27 | `lin_bf16 gate` | moe :1693 | k=5120 → n=384 |
| 28 | `route_topk` | :1700 | score_func=2；smem **3096 B** |
| 29 | `zero` ×2 | :1728/:1729 | memset |
| 30 | `quant_fp4`（激活）| :1783 | rows=1, cols=5120, block=32 |
| 31 | **batched 路径（`moe_batch()` 代码默认 ON**，:188 `unwrap_or(true)`）| :2317-2432 | `expert_gate_up_fp4_batched`（smem **20480 B**）→（`gateup_fused` 开时**跳过** `swiglu_limit_batched`；`gateup_fused = DSV41_GATEUP_FUSE!=0 && supports_gateup_fuse() && expert_fp4_mode()==2`，:2344/:2383 —— 必须与 `.cu:1367` 的 `g_fuse && g_expert_fp4_mode==2 && dim%512==0` 逐字镜像）→ **down 方向二选一**：`DSV41_DOWN_FUSE`（:200，默认 ON）⇒ `expert_down_reduce_fp4_batched` **一次启动**（grid `⌈dim/8⌉`、串行升序 slot、`out` 覆盖写，替代下两行）；否则 `expert_down_fp4_batched`（smem `inter_local*4`）+ `moe_down_reduce`（定序求和 ✓）|
| 31' | sequential 回退（逐 slot ×topk）| :2433-2477 | `expert_gate_up_fp4_indirect` / `swiglu_limit` / `expert_down_fp4_indirect` |
| 32 | 共享专家（`shared_rank`：`DSV41_SHARED_TP` 时 = **所有 rank**，各自 `inter/world` 切片；否则仅 rank 0）| :2504-2588 | gate/up 二选一：`DSV41_SH_EXP_MX2`（默认 ON）⇒ **一次 `gemm_fp8_mx2`**(w1,w3)；否则两次 `gemm_fp8_mx`。前置 `quant1(xn)`（`sh_via_mixed` 时由 `gemm_bf16_fp8x2` 顺带完成）。后接 `swiglu_limit` + `quant1(ex_act)` + `gemm_fp8_mx`(w2) + `add_inplace(o, ex_out)` |
| 33 | **AR#2** | `moe_reduce()` :638 | `s.o`, 20480 B |

### 段 C

| # | 调用 | 出处 |
|---|---|---|
| 34 | `hc_post`（ffn）| :1100 |
| 35 | `copy_h_back` | :1110 |

---

## 3. tile 尺寸反推（用户裁定：由 smem 容量反推）

**带 dynamic smem 的核只有 4 类**（其余全为 0）：

| 核 | smem | 出处 |
|---|---|---|
| `hc_mixes` | `(mix + hc*hc)*4` = **160 B** | kernels.cu:1749-1750 |
| `indexer_topk` | `kIndexerChunk*5 + topk*12 + 64` = **26688 B**（编译期常量，与 per-step 计数解耦 ✓）| kernels.cu:1414-1415 |
| `route_topk` | `n_experts*8 + topk*4` = **3096 B** | route.cu:147 |
| `expert_*_fp4(_batched)` | **`k*sizeof(float)`** = 激活行驻 smem（gate/up: `dim*4`=20480 B；down: `inter_local*4`），另加 `256*sizeof(float2)` 的 LUT | experts_mxf4.cu:1210/1284/1370/1392 |

**推论（关键）：**最大的一项是 `expert_*` 把**激活行整条**（`k*4`）驻 smem ✓ —— 这已经是"tile 对齐"的既有范式 ✓。
若融合核沿用"整条激活行 + 输出 tile"的布局：

- 激活行 f32 = 5120×4 = **20 KB**（或 fp8 5 KB）；
- 每阶段中间量按 tile 计：T 列 × 4 B × ~4 个活跃缓冲 = 16T B；
- 227 KB/block（Blackwell 上限，见 kernels.cu:1246）⇒ **单看 smem，T 可达 ~1.2 万列** ✗ 远大于 dim=5120 ✓
  ⇒ **tile 实际不由 smem 卡住，而由并行度卡住** ✗✓：bs=1 下需要 ≥ ~2 blocks/SM × 148 SM ⇒
  **T ≤ 5120/296 ≈ 17 列（沿 dim）或 ≤ 20480/296 ≈ 69 列（沿 hc_dim）**。

**⇒ 设计取值：沿 hc_dim 取 T = 64 列（= 四分之一个 hc=4 的"head"），得到 320 blocks** ✓
（既能放满 148 SM 的 2 波，又让每阶段的中间量都远小于 smem 预算 ✓）。**这是把"由 smem 反推"
落成的实际含义：smem 给出上界，并行度给出下界，取两者之间的 T=64** ✓。

---

## 4. 融合核结构（段 A / B / C 各一个）

**通用形态：每 block 拥有 hc_dim 上的一段 tile（T=64 列），块内顺序走完该段的阶段链，
中间量在 smem/寄存器里交接、不落 global；跨 block 的少数归约沿用既有模式。**

**跨 block 归约的现成模式**（必须复用，不要另创）：
- `hc_mixes` 的两阶段求和（warp shuffle → `wpart[32]` → 跨 warp）✓ kernels.cu:590-601；
- 需要全局归约时用"设备全局 partials + is_last 选举"（`hc_mixes_ss_kernel` 的做法 ✓，
  `hc_mixes_rows_kernel` + `post` 三段式 ✓）—— 注意 **split=8 会改变部分和顺序** ✗
  （已实测：文本与基线的差异即由此而来 ✓），融合核若要求逐位等价须 **split=1** ✓。

| 段 | 阶段链（块内） | 跨 block 的点 | 段末 |
|---|---|---|---|
| **A** | hc_mixes(160B) → hc_collapse → rmsnorm → wq_a → q_norm → wq_b → rope(q) → wkv → kv_norm → rope(kv) → ring_append → window_idxs → compress → indexer → sparse_attn → rope⁻¹ → quant1(o) → wo_a → wo_b | hc 的 ss/comb 归约（既有模式 ✓）；rope/kv 的 row 级操作可 tile 内完成 ✓ | **AR#1** |
| **B** | hc_post → copy_h_back → hc_mixes → hc_collapse → rmsnorm → gate → route_topk → quant_fp4 → 专家（gate/up → swiglu → down → 定序 reduce）→ 共享专家 | route 的 top-k（单 block ✓ 既有一块）；专家的 slot 归约（既有定型 ✓） | **AR#2** |
| **C** | hc_post → copy_h_back | 无 | — |

**可分离的开关**：每段一个 env（`DSV41_FUSE_A` / `_B` / `_C`），默认 **OFF**，A/B 通过才翻 ✓。
**copy_h_back 是纯 memcpy** ✓（非 kernel）⇒ 关掉它可省 2 次 80 KB D2D ✓（融合后 `s.h2` 可直接别名）。

---

## 5. 验证与等价性判据

1. **每段单独 A/B**：同二进制、同会话背靠背（`scripts/dsv41_serve_ab.sh <tag> [ENV]` ✓），
   逐步直打 p50 + `faults` ✓。
2. **五段全文对照**（Paris / Tokyo / 1+1= / 静夜思 / 出师表，`max_tokens=96`，读 200 字符 ✓）：
   判据 = **正文（引文本身）逐字正确** ✓；前言级游离片段可接受（**与 base 同类瑕疵**，
   已有用户认可的 `gemm3` 先例 ✓，见 `STATUS.md` 的向量化判定节 ✓）。
3. **数值参照**：能保持顺序就保持（同一 lane→元素映射 ✓）；顺序必变的地方（tile 重排）
   记录在案并以文本判定 ✓。**不做**"看着差不多"的推断 ✓。

## 6. 已知风险 / 待确认

- **`gateup_fused` 曾与 `.cu` 融合条件不一致（已修复 2026-09-11）** ✗→✓：Rust 侧原来只判
  `DSV41_GATEUP_FUSE` + `supports_gateup_fuse()`，漏了 `.cu:1367` 的 `g_expert_fp4_mode == 2`。
  当 `DSV41_EXPERT_FP4_MODE=0/1` 时 kernel 写满 `2*inter` 不融合，而 host 仍按融合推进
  `act_slot=inter` 并跳过 swiglu ⇒ **静默数据错位**（非性能问题）。现两处均加
  `&& expert_fp4_mode() == 2`（`chain_dev.rs:2344`/`:2383`，helper 见 `:218`，OnceLock 缓存、
  未设默认 2、非法值按 atoi 语义取 0）。
  残留：`.cu` 还有 `(dim % 512) == 0` 这一项未镜像——`dim` 恒为 5120（`%512==0` 恒真），
  故当前为惰性；若未来支持非 512 对齐的 dim，需一并镜像。
- **`DSV41_MOE_BATCH` 默认值的注释与实现矛盾** ✗（注释 :173 写 DEFAULT OFF，实现 :179 是 `unwrap_or(true)` ✓）
  ⇒ 融合核按 **batched 路径为准**（代码为准 ✓），并顺手修正注释 ✓。
- **`DSV41_GRAPH_MOE` 是死路径** ✗（`moe_graph_armed` 全仓无赋值点 ✓）⇒ 其注释/字段应清理 ✓；
  我先前把它当作"段 B 边界定义"是错的 ✗（已在本文件更正 ✓）。
- **`s.o` 在段 A 内有双重生命周期** ✗：sparse_attn 的输出（`nlh*hd`=4096）与 wo_b 的输出（`dim`=5120）
  ⇒ 融合时不能混用 ✓（`s.xq/s.xsc` 按 `max(...)` 申请，见 :348-352 ✓）。
- **整个 `step_body` 默认在图捕获区内** ✓（`chain_dev.rs:729-740`）⇒ 融合核必须**可捕获**：
  核内不得有 `cudaMalloc`／`cudaStreamSynchronize`／host 交互 ✓，任何分配都要在预热期完成 ✓
  （`kernels.cu:1174-1194` 的 per-device 预热 scratch 是既有正确范式 ✓）。
- **MoE 的 TP 切分轴是 `inter` 而非 expert-parallel** ✓（`moe()` 正文 :1685-1688 ✓；
  `moe_reduce()` 注释 :641 写 expert-parallel ✗ 与实现矛盾 ⇒ 以正文为准 ✓）。
