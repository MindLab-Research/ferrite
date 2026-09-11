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
| 11 | `ring_append`（仅 owner 层）| :1284 | `[hd]` → ring。**B2 已与 #12 融合**（见 §6）|
| 12 | `window_idxs`（每步无条件）| :1325 | `idxs[win+index_topk+8]i32`。**B2**：`ring_win_fuse` 一次启动同时做 #11+#12，`ring==null` 时仍写 idxs（非 owner 层的独立 `window_idxs` 也消失）|
| 13 | `compress()`（kv-source 层）| :1308-1314 → :1589-1649 | pool + commit 两核 |
| 14 | `indexer()`（仅 8 个 source 层）| :1332 → :1446-1584 | idx_k（512→128）+ norm + rope + publish + idx_wq_b（1280→4096）+ rope + idx_weights（5120→32）+ `indexer_topk`（smem **26688 B**，与 per-step `n_pos` 解耦 ✓）|
| 15 | `comp_placeholder`（无 indexer 时）| :1340 | 写 `idxs[win..]`。**B3**：已折进 `ring_win_fuse` 的 epilogue（`DSV41_COMP_PLACEHOLDER_FUSE` 默认 ON），独立 launch 消失 —— 见 §6 ③ |
| 16 | `sparse_attn` | :1348 | b=1,m=1,h=nlh,d=512, window=128, index_topk=512 → `s.o` |
| 17 | `apply_rope`（**反向**）| :1363 | rows=nlh, inverse=true。**B2**：`apply_rope_q` 在 epilogue 直出 #18 的 fp8（见 §6）|
| 18 | `quant1`（o 量化）| :1392 | 长度 `nlh*hd`=4096。**B2**：由 #17 的 epilogue 承担，本行在融合路径消失 |
| 19 | `gemm_fp8_mx` ×`nlg`（分组 o 投影）| :1409 | k=`hpg*hd`=4096, n=`olg`=1024 |
| 20 | `lin wo_b` | :1426 | k=`ol_local`=128 → n=5120 |
| 21 | **AR#1** | :1434 | `s.o`, 20480 B |

### 段 B（`layer()` :1017-1096 + `moe()`；行号 2026-09-11 复核：`fn moe` = 2210，`fn layer` = 1314，`fn attention` = 1586，`fn step_body` = 930，`fn moe_reduce` = 795 —— ⚠️ 本节表格内的 `chain_dev.rs` 调用点行号基于更早的修订，使用前需以当前文件复核）

| # | 调用 | 出处 | 形状 |
|---|---|---|---|
| 22 | `hc_post`（注意力）| :1017 | `[1,hc*dim]` → `s.h2` |
| 23 | `copy_h_back` | :1027 | **纯 D2D memcpy**，`hc*dim*4` = 80 KB |
| 24 | `hc_mixes`（ffn）| :1034 | 同 #1，写 premix_slot(2) |
| 25 | `hc_collapse`（用 premix_slot(1)）| :1053 | 同 #2 |
| 26 | `rmsnorm`（ffn_norm）| :1061 | `[1,dim]` |
| 27 | `lin_bf16 gate` | moe :1693 | k=5120 → n=384 |
| 28 | `route_topk` | :1700 | score_func=2；smem **3096 B**。**已融合**：`DSV41_ROUTE_FUSE`（默认 ON）时由 #27 的 gate GEMV（`ferrite_gemv_bf16_v2_route`）last-block epilogue 顺带完成，本行消失（老 `.so` / MIX_GATE / CUBLAS_M1 自动回退到本行）|
| 29 | `zero` ×2 | :1728/:1729 | memset |
| 30 | `quant_fp4`（激活）| :1783 | rows=1, cols=5120, block=32。**已融合**：`DSV41_QUANT_FP4_FUSE`（默认 ON）时 `dsv41_quant_fp4` 发射单核 `quant_fp4_fused_kernel`（量化+打包一步，bit-exact），旧 `quant_kernel<1>` + `fp4_pack_kernel` 两段路径保留为回退（env=0，或 block 奇数/>256）。**再融合（`DSV41_QUANT_FOLD`，**默认 OFF**——v12 serve A/B 实测 +0.39ms 回归，2026-09-11 结论见下）**：FFN 半的 hc EARLY collapse epilogue 在写 `s.xn` 的同一趟里直出 fp4 到 `s.xq4`/`s.xsc4`（`hc_mixes_tail_kernel` 的 `xq4`/`xsc4` 形参），本行的 launch 被跳过；同 32-block 算式，bit-exact（见下文） |
| 31 | **batched 路径（`moe_batch()` 代码默认 OFF**，:190 `unwrap_or(false)`）| :2317-2432 | `expert_gate_up_fp4_batched`（smem **20480 B**）→（`gateup_fused` 开时**跳过** `swiglu_limit_batched`；`gateup_fused = DSV41_GATEUP_FUSE!=0 && supports_gateup_fuse() && expert_fp4_mode()==2`，:2344/:2383 —— 必须与 `.cu:1367` 的 `g_fuse && g_expert_fp4_mode==2 && dim%512==0` 逐字镜像）→ **down 方向二选一**：`DSV41_DOWN_FUSE`（:200，默认 OFF）⇒ `expert_down_reduce_fp4_batched` **一次启动**（grid `⌈dim/8⌉`、串行升序 slot、`out` 覆盖写，替代下两行）；否则 `expert_down_fp4_batched`（smem `inter_local*4`）+ `moe_down_reduce`（定序求和 ✓）。**W2 L2 预热（`DSV41_W2_PREWARM`，默认 ON）**：gate/up 启动之后、down 启动**之前**插一次 `dsv41_w2_l2_prewarm`（`device.rs::w2_l2_prewarm`，`supports_w2_prewarm()` 探测符号，老 `.so` 直接跳过），对 `slots` 个专家的 w2（`dim*(inter/2)`）与 scale 行（`dim*(inter/32)`）发射 `cp.async.bulk.prefetch.L2.global`；fire-and-forget、不写任何字节、恒返回 0，故逐位不变。动机：down 是**延迟受限**（286-385 GB/s vs ~7 TB/s），w2 在本步之前从未被触碰，把它提前放进 L2 等于把延迟从 ~600 ns 降到 ~200 ns（同层 9.17+0.57 MB < L2 的 10%；跨层 367 MB/step 不行）。**不要**把这套预热塞进 down kernel 内部做软件流水：每个 warp 每 slot 的算术只有 ~30 周期（~20 ns），对 ~600 ns 的 HBM 延迟差两个数量级，预热必须发生在消费者之外的空闲窗口 |
| 31' | sequential 回退（逐 slot ×topk）| :2433-2477 | `expert_gate_up_fp4_indirect` / `swiglu_limit` / `expert_down_fp4_indirect` |
| 32 | 共享专家（`shared_rank`：`DSV41_SHARED_TP` 时 = **所有 rank**，各自 `inter/world` 切片；否则仅 rank 0）| :2504-2588 | gate/up 二选一：`DSV41_SH_EXP_MX2`（默认 ON）⇒ **一次 `gemm_fp8_mx2`**(w1,w3)；否则两次 `gemm_fp8_mx`。前置 `quant1(xn)`（`sh_via_mixed` 时由 `gemm_bf16_fp8x2` 顺带完成）。swiglu 二选一（A4）：`DSV41_SWIGLU_Q`（**默认 ON**，`chain_dev.rs:301-331`）⇒ `swiglu_limit_q` 一次启动直出 `(xq,xsc)`（warp=1 个 32-block，amax 一次 shuffle；`inter%32` 不满足则返回 1 回退）；否则 `swiglu_limit` + `quant1(ex_act)`。⚠️ 它曾被记为 "round-18 数值 bug 暂缓"，实为**误归因**：A4 代码首次出现在 `f3b1be1`（其 commit message 报告的正是 round-18 那次跑分），而 `f3b1be1^` 里根本没有 `swiglu_limit_q`/`act_q`——round-18 的乱码与 A4 无关（根因是 gateup/down 融合的 `.cu`/Rust 默认值分裂，见下文），A4/A5 只是被 `f6a1c08` 连带批量 gate OFF。逐项核对：f32 写回同一个寄存器 `v`（无 global 回读）、amax 是同一组 32 值的 fmaxf 树（warp 恰好覆盖一个量化块，无需跨 warp 归约）、scale/字节算式与 `quant_kernel<0>`(block=32, round_scale=1) 逐项同式 ⇒ 逐位相同；证据 = `kernels/cuda/tests_dsv41_glue.cu` 的 `swiglu_q` 用例（对真实 `dsv41_quant_fp8` 比 xq/xsc/f32 逐位）。w2 二选一（A5）：`DSV41_MOE_EPI_ADD`（**默认 OFF**，`chain_dev.rs:292-299`；与 A4 一起被 `f6a1c08` 批量 gate OFF，`=1` 可重开）⇒ `gemm_fp8_mx_add` 把 `add_inplace` 折进 GEMV 的 lane-0 epilogue，**直写 `s.o`**（结合律 `o+(acc+bias)` 不变 ⇒ 逐位相同）；否则 `gemm_fp8_mx`(w2→`ex_out`) + `add_inplace(o, ex_out)`。两处 gate 均 OnceLock 缓存、`supports_*()` 探测 `.so` 符号（老 `.so` 自动回退） |
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

- **✅ B2：attention 尾部的两个小 kernel 融合（2026-09-11，DSV41_OROPE_Q / DSV41_RING_WIN_FUSE 默认 ON）**：
  - **① o-rope fp8 epilogue** —— `apply_rope_kernel`（`dsv41_kernels.cu`）新增 `xq`/`xsc` 两个可选尾参
    （null = 跳过，原 `dsv41_apply_rope` 行为逐位不变）；新符号 `dsv41_apply_rope_q` 在旋转后**再走一趟
    全行量化**（`[0, rows*row_len)`），用 `quant_kernel<0>` 的算术逐字（amax 一次 full-warp shuffle /
    `fast_round_scale(·,1/448)` / clamp±448 / `__nv_fp8_e4m3`），因此与它替代的 `quant1(s.o)` 逐位相同，
    `wo_a` 直读同一 `s.xq`/`s.xsc`。⚠️ **纠正一个直觉误区**：rope 循环是 `i<half`（half=32）且每 lane 处理
    **一对** (2i,2i+1)，32 lane 覆盖 64 列 = **2 个量化块**，**不能**让 rope 循环自己充当量化块；而且它只碰
    每 head 尾部 `rope_head_dim` 列，而 `quant1(s.o)` 量化的是**整行** `nlh*hd`。所以 epilogue 必须是
    **barrier 之后的第二趟**（一个 warp 一个 32-block）；这不是 T1 的"写回趟顺带"。launcher 在
    `rows*row_len % 32 != 0` 时返回 1 → Rust 回退 `(apply_rope, quant1)`。每步省 40 次 launch。
  - **② `ring_append` + `window_idxs` 合并** —— 两者相邻、都只依赖 `*pos_ctr`、互不消费对方输出
    （append 写 ring，idxs 只读计数器），故可合成一个 `dsv41_ring_win_fuse`（grid = `max(window,hd)/128`）。
    `ring==null` 表示非 owner 层（不拥有自己的 store）仍执行 idxs 半边 ⇒ **每个** layer 的独立
    `window_idxs` 都消失。调用点从原 `window_idxs` 位置（compress **之后**）上移到 `ring_append` 位置
    （compress **之前**）——安全，因为二者之间没有代码写 `idxs[0,win)`，唯一读者是 `sparse_attn`。
    每步省 40 次 launch。
  - **③ `comp_placeholder` 折进 ring_win_fuse 的 epilogue**（B3，`DSV41_COMP_PLACEHOLDER_FUSE` 默认 ON）——
    #15 写的 `idxs[win, win+take)`（`take = min(*clen, index_topk)`，**设备端**派生）与 #12 写的
    `idxs[0,win)` 是同一缓冲的**不相交**两块，唯一读者同为 `sparse_attn`；#12 与 #15 之间唯一的写者
    （indexer）自己拥有 `[win, ..)`，所以它走同一入口但**关掉** placeholder 半边。新符号
    `dsv41_ring_win_fuse_ph`（grid = `max(window, hd, index_topk)/128`，覆盖原独立 launch 的
    `ceil(index_topk/128)` 块），`clen==nullptr` 时与 `dsv41_ring_win_fuse` 逐字节相同。
    `chain_dev.rs` 的 `ph_ok` 额外排除 **compress source**（`clen[layer]` 正被本步 compressor 推进 ⇒
    在 ring_win 处读会 race；生产配置里 `kv_source ⊆ index_source`，那些层本就不需要 placeholder）。
    每步省 30 次 launch / 30 个图节点。
  - 收益：合计 −80 个图节点（若每次 launch ~1.5-2µs，则 −0.10~0.15ms 量级）；**尚未上机实测**。
  - 回退：`DSV41_OROPE_Q=0` / `DSV41_RING_WIN_FUSE=0`，或老 `.so`（`ko!` 符号探测自动回退）。
  - ⚠️ 与 sparse 段的并行改动无关：apply_rope 在 `dsv41_kernels.cu:1131`，sparse 段在 `:2839+`。

- **✅ wo_b 直读 f32 激活（DSV41_WOB_F32，默认 ON）** ✓（2026-09-11，待上机 parity）：
  `gemm_fp8_gemv_kernel` 新增**可选尾参** `const float* a_f32 = nullptr`（追加在 `qr_eps` **之后**，
  与 NORM_FUSE 的 `qr_raw` 相邻但独立参数位）。非 null 时 staging 直接
  `s_af[i] = a_f32[i]`，**跳过** `s_lut[a[i]] * s_as[i>>5]` 的 fp8 解码 + scale 乘；consume 循环
  （`acc += s_af[j] * (s_lut[row_s[j]] * sb)`）一字未动，**权重侧仍走 fp8 解码**，所以 `s_lut` 照建。
  新 C 符号 `dsv41_gemm_fp8_mx_f32(a_f32, w, w_scale, bias, out, n, k, s)`（stream **在最后**，
  该符号无 C++ 默认尾参，**不照抄** `gemm_fp8_mx` 的「stream 在 shape 后」ABI）。Rust：
  `device.rs` 的 `supports_gemm_fp8_f32()` + `gemm_fp8_mx_f32()`；`chain_dev.rs` 的 `wob_f32()`
  与 wo_b 调用点。
  - **为什么不会重蹈 B1**：B1 把融合做进 wo_a 的 epilogue，被迫 32 warps/block（32 连续行 = 一个
    quant block）⇒ grid = n/32、32 warps，**148→32 活跃 SM**，实测 +0.24ms。本方案**只改数据路径，
    不改 grid 形态**：block = 常规 `g_gemv_warps`(默认 4) / `ceil(n/warps)`，与普通 `gemm_fp8_mx`
    GEMV 同形，无 SM 利用率损失。
  - **⚠️ 非逐位**：直读 f32 跳过了 quantize→dequantize 往返，**没有 4-bit 尾数损失，精度更高**，
    与 fp8 路径**不逐位一致**。下游是 wo_b 行 partial → AR 求和 → hc_post，精度提升方向正确，
    但**上机必须做一次 parity**（`DSV41_WOB_F32=1` vs `=0` 的 text/fingerprint 对比）。
  - **限制**：`vec >= 3`（`s_af` 物化只在该分支）；无法与 AR store 融合共存（该融合在 fp8 launcher
    的 epilogue，故 f32 路径仅在 `ar_store_fused == false` 时启用）；`wo_fused`(B1) 优先。任一不满足
    或 `.so` 无符号 → 回退 `(quant1(s.wo), gemm_fp8_mx)`。收益 = 每步省 40 次 `quant1` launch
    （~0.06ms + 40 图节点）**尚未上机实测**。
  - 回退：`DSV41_WOB_F32=0`，或老 `.so`（`ko!` 符号探测自动回退）。
  - **验证**：`cargo check -p ferrite-models` ✓；远端 `nvcc -gencode arch=compute_103a,code=sm_103a
    -O3 -std=c++17 --use_fast_math -Xptxas -v -c dsv41_kernels.cu` **EXIT=0、无 error** ✓（仅既有的
    `ap`/`per`/`e2m1_to_f` 未引用 warning，非本次引入）。
  - ⚠️ 改动落在 gemv 段 `dsv41_kernels.cu:1965-2760`（含 NORM_FUSE 的 `qr_raw` prologue 区域），
    未碰 hc（:3700+）/sparse（:2839+）段；并行 agent 的 `xq_of_qr_valid` 修复在 `chain_dev.rs` 另一处，
    与本改动共存无冲突。

- **✅ engram 的 wkv 投影直读 f32（复用 `supports_gemm_fp8_f32`，无新 gate）** ✓（2026-09-11）：
  `engram_apply()` 的 `wkv` 投影原本是 `quant_fp8(eng_rows) → gemm_fp8_mx`，现在先试
  `gemm_fp8_mx_f32(eng_rows, wkv, wsc, ..., n=(hc+1)*dim, k=n_cols*ehd)`——与 wo_b 同一个符号、
  同一套回退契约，仅多一个调用点。省 **2 次/步** `quant_fp8` launch（L1 + L14 两个 engram 层），
  并省掉 `eng_xq`/`eng_xsc` 的 4-bit 尾数损失（**更精确**，与 wob-f32 同论证）。
  - **语义边界**：读的是 **AR 之后** 的 f32 行（`all_reduce_inplace` 在 f32 上求和后，本直读就是那次
    求和的逐元素拷贝）。quant-final-sweep 的警告「engram 的 fp8 不能由 `engram_gather` 直出」针对的是
    **AR 之前** 发射 fp8（`fp8(Σ rows) ≠ Σ fp8(row)`），与本改动的 POST-AR 读取是两回事，不冲突。
  - **回退**：老 `.so`（无 `dsv41_gemm_fp8_mx_f32` 符号）或 shape decline（`Ok(false)`，如
    `k = n_cols*ehd` 非 32 倍数）→ 自动落回 `(quant_fp8, gemm_fp8_mx)` 对。生产形状
    `k = 24*256 = 6144`（32 倍数 ✓，`vec` 分支满足）。两个分支都是 host 侧 + shape 确定性的，
    故捕获的 decode graph 跨 replay 一致。
  - **无新 env gate**（`DSV41_WOB_F32` 只管 wo_b 调用点，不影响本路径）——回退靠符号探测/形状。
  - **验证**：`cargo check -p ferrite-models` ✓（尚未上机 parity；`DSV41_WOB_F32` 的 A/B 轮次可顺带
    覆盖 engram 四段文本）。

- **✅ engram_apply 向量化 + 256 线程；engram_hash_step 按列并行** ✓（2026-09-11，已验证）：
  两个 kernel 都是「小 grid + 单线程/长串行扫描」的病（同 `gated_rmsnorm` 那类），实测不是带宽而是延迟：
  - `engram_apply_kernel`：grid 只有 `(rows=1, hc=4)` = **4 个 block**，每 block `dim=5120` 的 40 轮
    标量扫描 + `block_sum3`。改成 **float4 体（5 轮）+ blockDim 128→256**（256 = 8 warps，
    `block_sum3` 的 `red[3][8]` 硬上限，不可再宽）。保持标量回退分支（`dim % 4 != 0` 或四个基址
    任一非 16B 对齐）。
  - `engram_hash_step_kernel`：原本 **1 线程** 串行走完 `n_layers*n_cols = 48` 列，每列一次
    **64-bit 取模**（emulated，数百 cycle）→ 改成一列一线程（`<<<1,128>>>` + grid-stride），
    token gather 每 lane 重算（≤ max_ngram 次 L1 读）。整数 + 每元素单写者 ⇒ **bit-identical**。
  - **实测**（B300，event-timed，isolated）：`engram_apply` 16.36 → **4.72µs**（空 kernel 地板 3.09µs），
    `engram_hash_step` 15.64 → **3.37µs**。⇒ 每步省 ≈ **0.023 + 0.012 = 0.035ms**。
    （新加 `bench_floor.cu`/`parity.cu` 的注意：`#include "dsv41_glue.cu"` 会被同目录的旧副本劫持，
    比较 old/new 必须把两份分别放进独立目录。）
  - **正确性**：`parity.cu` 对 5 个形状（含生产 1×4×5120）与 f64 参考的 relmax 与改动前**逐位相同**；
    48 列 hash id `mismatch=0`；仓库自测 `tests_dsv41_glue.cu` 结果与基线一致（唯一 FAIL 是既有的
    `swiglu_q rows=3`，与本改动无关，old/new 同样 FAIL）。
  - ⚠️ `engram_apply` 的规约次序变了（float4 分组 + 256 线程 stride），**非 bit-identical**；
    与 engram f32 直读同一个取舍，文本级验证仍需一次 `dsv41-run` A/B。

- **✅ q rope / idx_q rope 已折进 GEMV epilogue（DSV41_ROPE_FUSE，默认 ON）** ✓（2026-09-11）：
  新的两个 C 符号 `dsv41_gemm_fp8_mx_rope`（单族，q rope）与 `dsv41_gemm_fp8_mx2_rope`
  （两族，wq_b 的 q rope + idx_wq_b 的 idx_q rope 各用各自 head 宽度 `rope_hd1/hd2`）把
  `apply_rope_kernel` 的旋转表达式逐字搬进 `gemm_fp8_gemv_kernel` 的行循环之后的 epilogue
  （`dsv41_kernels.cu` 的 rope 段）。**不复用 `dsv41_gemm_fp8_mx` 的 ABI**（该符号有固定 ABI +
  6 个调用点，融合形态另立符号，老 `.so` 由 `supports_rope_fuse()` 符号探测回退 ✓）。
  关键结构：rope 的对 (2i, 2i+1) 必落在同一 block 的**相邻 warp**（head 宽是 32 倍数、对起点偶数），
  launcher 强制 32 warps/block、grid=n/32（同 B1），pair-head warp 读 B1 的 `s_rows[warp+1]` 后
  在 lane 0 写回；`v = acc + bias` 即 rope kernel 会读回的同一 f32。launcher 校验
  `n%32==0`、`rope_hd%32==0`、`rope_rd` 偶且 `<=rope_hd`、`g_gemv_fp8_mode>=3`，任一不过返回 1 →
  Rust 侧回退 `lin`/`lin2` + 独立 `apply_rope`（逐位等价）。Rust：`chain_dev.rs` 的 `lin_rope`/
  `lin2_rope`，`attention()` 用 `q_roped`/`idx_q_roped` 记录合并的旋转，`idx_q_rope` 这个
  one-shot Cell 让 `indexer()` 跳过它自己的 `apply_rope(s.idx_q)`。收益 = 每步省 40（q）+ ~7
  （idx_q）次 launch。**唯一未在无 GPU 环境验证的点**：fast-math 下两处的 `x0*c - x1*s` 收缩
  是否逐位一致（同 TU 同表达式，按 rmsnorm_rope 的同款论证）——需一次 parity 测试确认 ✓。
  - ⚠️ **ABI 参数错位曾让 rope fusion 完全静默失效（2026-09-11 修复）** ✗→✓：`device.rs` 的两个
    FFI 函数指针类型把 `CuStream` 放在了第 9 个参数位（紧跟 `n, k`），照抄了 `gemm_fp8_mx` 的
    「stream 在 shape 之后、可选尾参之前」写法；但 `dsv41_gemm_fp8_mx_rope` / `..._mx2_rope` 没有
    可选尾参，C 侧 stream 是**最后一个**参数。于是 C 读到的形参整体错位一格：
    `rope_cos <- 真 stream`、…、`rope_rd <- 真 rope_inverse(0)` ⇒ launcher 的
    `rope_rd <= 0` 判定成立 → 返回 1（decline）→ Rust `gemm_fp8_mx_rope` 返回 `Ok(false)` →
    调用点静默回退到 `lin`/`lin2` + 独立 `apply_rope`。表现为「DSV41_ROPE_FUSE 默认 ON、
    `supports_rope_fuse()` 符号探测为真、kernel 侧实现正确，但 nsys 里 apply_rope 仍是 88/步」，
    且 round-37 的 f32v2 臂「中性」（既没省 launch 也没加 epilogue 开销）——正是静默回退的特征。
    **教训**：`Option<unsafe extern "C" fn(...)>` 的参数顺序是手写转录，编译器不会对着 `.cu` 校验；
    新符号的 FFI 类型必须逐参对照 C 原型（本仓库主流约定是 stream 放最后，只有带 C++ 默认尾参的
    `dsv41_gemm_fp8_mx` 例外）。修复：device.rs 两处 fn 类型与两处调用把 `self.stream` 移到末尾。
- **✅ rmsnorm_q + wq_b gemv 的生产者/消费者融合（DSV41_NORM_FUSE，默认 ON）** ✓（2026-09-11，待上机 parity）：
  新的 C 符号 `dsv41_gemm_fp8_mx_rope_norm`（`dsv41_kernels.cu`，紧跟 `dsv41_gemm_fp8_mx_rope` 之后）
  把 `rmsnorm_q_kernel` 的 **norm + fp8 encode** 搬进 `gemm_fp8_gemv_kernel` 的 **prologue**
  （`if (qr_raw != nullptr)` 分支），消费点自己做生产：每 block 用
  `blockDim` 跨步循环 + `__shfl_down_sync` 树 + thread0 汇总算出 `inv`，再逐元素
  `v = qr_raw[i]*inv*qr_w[i]` → warp 内 `__shfl_xor_sync` 求 32-block amax →
  `fmaxf(fast_round_scale(am,1/448),1e-30)` → clamp ±448 → `__nv_fp8_e4m3`，直接写进 smem 的
  `s_a`(字节) / `s_as`(scale)。之后原有的 a32 物化（`s_af[i] = s_lut[s_a[i]] * s_as[i>>5]`）与
  GEMV 主循环一字不改 ⇒ wq_b 输出 = `rmsnorm_q + gemm_fp8_mx_rope` 的输出。
  - **为什么改 grid 形态为 0**：融合点只是 prologue，`blocks = n/32`、32 warps/block 与 rope
    launcher 完全相同，行循环与 epilogue 未动。
  - **两块开销**：每 block 多读 k=1280 个 f32 + 一次 1280 元素归约；wq_b 的 grid 是 64 block
    （n=2048/32），共 ~82K 元素读取 + 归约，GPU 上 <1µs，可忽略（**不是**原分析里的
    "2048 blocks × 1280"——请以此处 Grid 数为准）。
  - **两条强约束**（launcher 强制，违反则返回 1 → Rust 回退旧双 launch 对）：
    ① block 必须 32 warps，因为归约树要跟 1024-thread 的 `rmsnorm_q_kernel` 逐位对齐；
    ② mode 强制为 4（唯一把 activation 留在 smem 的模式），**`DSV41_GEMV_FP8_MODE` 对该符号无效**。
  - Rust：`device.rs` 的 `supports_gemm_fp8_norm()` + `gemm_fp8_mx_rope_norm()`；
    `chain_dev.rs` 的 `lin_rope_norm()`，`attention()` 的 wq_b 调用点（`norm_fused`/`q_roped`）
    与 `indexer()` 的 idx_wq_b 调用点。收益 = 每步省 40 次 `rmsnorm_q` launch（nsys v5: 0.13ms）
    + 40 个图节点。
  - ⚠️ **`qr` 在该路径上保持 RAW**：融合是把 norm 推迟到消费者，所以 `qr` 不再被原地归一化。
    凡在 wq_b 之后读 `qr` 的地方都必须走同一个融合 launch —— 目前只有 `indexer()` 的 idx_wq_b
    （`DSV41_IDX_FUSE` 默认为 **OFF**：`idx_fuse()` 是 `.unwrap_or(false)`，旧注释与 STATUS 里
    "默认 ON" 是**过期**描述；只有 IDX_FUSE=ON 时两族 launch 共用一个 `xq`，才不可能让 `qr` 变 raw）。
    协调用的 one-shot Cell 是 `Scratch::qr_raw`：`attention()` 每条路径都写它，`indexer()` 消费并清零；
    idx_wq_b 侧若融合 launch 意外 decline，会先补一次 `rmsnorm(qr)` 再走旧 `quant1` 路径（防御）。
    另：融合路径下 `xq_of_qr_valid` 显式清 0，避免残留 `true` 让后续某个 `quant1(qr)` 静默跳过。
  - 回退：`DSV41_NORM_FUSE=0`，或老 `.so`（`ko!` 符号探测自动回退）。
  - ⚠️ **唯一未在无 GPU 环境验证的点**：`--use_fast_math` 下 prologue 的 `ss += t*t` 归约链
    与 `rmsnorm_q_kernel` 是否仍逐位一致（同 TU、同表达式、同 blockDim ⇒ 预期一致；但
    build.sh 已明确警告 fast-math 可重结合，长内核的 unroll/双累加器可能改变顺序）。
    上机时先做一次 parity（`DSV41_NORM_FUSE=1` vs `=0` 的 text/fingerprint 对比），不一致立即
    以 `=0` 回退。**改动只落在 gemv 段与 rmsnorm_q 相关区域，未碰 hc（:3700+）/sparse（:2839+）段。**
- **`gateup_fused` 曾与 `.cu` 融合条件不一致（已修复 2026-09-11）** ✗→✓：Rust 侧原来只判
  `DSV41_GATEUP_FUSE` + `supports_gateup_fuse()`，漏了 `.cu:1367` 的 `g_expert_fp4_mode == 2`。
  当 `DSV41_EXPERT_FP4_MODE=0/1` 时 kernel 写满 `2*inter` 不融合，而 host 仍按融合推进
  `act_slot=inter` 并跳过 swiglu ⇒ **静默数据错位**（非性能问题）。现两处均加
  `&& expert_fp4_mode() == 2`（`chain_dev.rs:2344`/`:2383`，helper 见 `:218`，OnceLock 缓存、
  未设默认 2、非法值按 atoi 语义取 0）。
  残留：`.cu` 还有 `(dim % 512) == 0` 这一项未镜像——`dim` 恒为 5120（`%512==0` 恒真），
  故当前为惰性；若未来支持非 512 对齐的 dim，需一并镜像。
- **⚠️ safe3 的 +8.7ms 退化根因 = `f3b1be1` 误翻 7 个既有门默认值（2026-09-11 定案）** ✗：
  该提交的本意只是「关掉 gateup/down 两组新融合」，但其 diff 同时把 7 个**与 A4A5 无关的既有已验证门**
  从 `unwrap_or(true)` 改成 `unwrap_or(false)`（`chain_dev.rs`）：
  `moe_batch`(:190)、`nr_fuse`(:231)、`sh_exp_mx2`(:237)、`mix_gate_shared`(:261)、
  `head_slice`(:274)、`fuse_c`(:1305)、`fuse_b1`(:1311)。
  对照 10.16ms 基线（`314f5df`，round-16）这 7 个全是 `unwrap_or(true)` ⇒ 以 **ON 为准**。
  主因是 `moe_batch=false`：批化 4 launch/层 → 顺序 topk(=6) × 3 launch/层，
  40 层共多 ~560 次 launch/步（本 workload 是 per-call-fixed-cost bound，~10-15µs/次）⇒ 就是那 ~8.7ms。
  修复：把这 7 个默认值还原 `true`，只保留新融合门（GATEUP/DOWN/AR_STORE/SWIGLU_Q/MOE_EPI_ADD）默认 OFF。
  注意 `DSV41_MOE_BATCH` 的 doc 注释仍写 "DEFAULT OFF"，与 round-16 基线以及 -3.2ms 的实测收益矛盾，应一并订正。
- **✅ gemv kernel 签名 +6 参数（epi_add + AR 5 个）已排除，与 +8.7ms 无关（2026-09-11 ptxas 实测）** ✓：
  `gemm_fp8_gemv_kernel` 14 → 22 参数（当次实验值），最终长到 **37 个位置参数**：
  epi_add + AR 5 个 + B1 的 `xq`/`xsc` + rope 7 个 + NORM_FUSE 3 个 + f32 的 `a_f32`。
  **2026-09-11 `gemv-struct-pack` 已取代该签名**：kernel 现在只收 4 个按值 struct
  （`GemvCore`/`GemvRope`/`GemvFusion`/`GemvEpi`，定义在 `dsv41_kernels.cu:2759-2886`，
  kernel 在 `:2915`），原参数名由 kernel 顶部一段 re-binding 块从 struct 取回，kernel body 逐字节未动；
  launcher 侧只构造 struct 再交给 `dsv41_pdl_or_plain` ⇒ 「Ex 转发 36+ 参数」这一失败模式从根上消失。
  位置参数个数的可维护性讨论到此结束。
  ⚠️ **该打包必须同时保留 `__grid_constant__ const` 与 `__launch_bounds__(1024)`**，
  两者都是承重限定符（实测 sm_103a / CUDA 13.2 / `-O3 --use_fast_math -Xptxas -v`）：
  37 标量 = **56** regs；纯 by-value struct = **72** regs；+`__grid_constant__` = **64** regs；
  +`__grid_constant__`+`__launch_bounds__(1024)` = **56 regs / 0 spill**（与标量版完全相同）。
  72 regs 时 1024 线程/块（5 条强制 32 warps 的 launcher：mx_rope / mx_rope_norm / mx2_rope /
  B1 epilogue / norm-fuse）需要 73728 > 65536 寄存器 ⇒ **launch 直接失败 `cudaErrorInvalidValue`**，
  即本文件追踪的 "cuda error 1"。另外 `__restrict__` **不是**机制：把 HEAD 的 17 处 restrict
  全部删掉仍是 56 regs（struct 成员加 restrict 也仍是 72）。
  以下为标量签名时代的对照。标量签名时代以生产 flag
  （`-O3 --use_fast_math -gencode arch=compute_103a,code=sm_103a -Xptxas -v`）对比 `314f5df` 与 HEAD：
  **寄存器 48 → 40（不升反降），两版均 0 spill，34 个 kernel 中只有这一个的寄存器数变了**。
  occupancy 也不是寄存器约束：mode 4 的 `gsmem` = 5·5120+22784 = **48384 B**（mode 3 = 43264 B），
  按 smem 算约 4 blocks/SM，而寄存器上限 48 regs→10 blocks、40 regs→12 blocks，**均非绑定项** ⇒ occupancy 不变。
  launcher 侧的每次调用开销也**未增加**：`dsv41_gemm_fp8_mx` 的两次 `cudaFuncSetAttribute`
  与 `cudaGetLastError` 与基线逐字节同构，唯一新增是 tile 路径上的 `staging_tbl != nullptr` 早退
  （`:2032`，M=1 热路径不经过）；`dsv41_gemm_fp8_mx2` 仅是 kernel 实参多传 `0/nullptr`。
  lane-0 的 `epi_add` 三元与 AR 循环都在 `staging_tbl == nullptr` 下短路，且每 warp 只 1 次。
  ⇒ 定位退化**不要**再看这里，看 launch 次数（见上一条）。
  注意 smem 只剩 768 B 余量（49152 − 48384），任何给 mode 4 加 smem 的改动都会
  越过 48 KB 门槛而触发 `cudaFuncSetAttribute`，需一并复核 launcher。
  - ⚠️ **`gemm_fp8_gemv_kernel` 的 SetAttribute 上限在同一文件里自相矛盾（2026-09-11 代码复核发现）**：
    7 个 launcher（如 `dsv41_kernels.cu:3580` rope_norm、`:3651` mx_rope、`:3713` mx2_rope）请求
    **232320 = 232448 − 128**（假设静态 smem = `s_norm_red[32]` = 128 B），但同一个 kernel 在
    `dsv41_gemv_occupancy`（`:4036-4039`，由 `4c995f2` struct-pack 提交从 232448 改成）请求
    **231676 = 232448 − 772**（假设静态 smem = 772 B）。两者只能有一个成立；若 772 是实测值，
    则所有 232320 请求都超过该 kernel 的动态上限 → `cudaFuncSetAttribute` 返回
    `cudaErrorInvalidValue`(=1) → launcher 在 launch **之前** `return (int)e` →
    `dsv41_gemm_fp8_mx_rope_norm: cuda error 1`。这也解释了为何 marshal 层修复
    （cudaLaunchKernel / PDL 关 / struct pack）全部无效：它们与根因无关。
    **正确做法**：不要硬编码，按 device+kernel 运行时推导
    `cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)` −
    `cudaFuncGetAttributes().sharedSizeBytes`，失败时降级为 decline(2)；用
    `cuobjdump -res-usage libferrite_kernels.so | grep -A2 gemm_fp8_gemv_kernel` 直接读出静态 smem 核对。
- **`DSV41_GRAPH_MOE` 是死路径** ✗（`moe_graph_armed` 全仓无赋值点 ✓）⇒ 其注释/字段应清理 ✓；
  我先前把它当作"段 B 边界定义"是错的 ✗（已在本文件更正 ✓）。
- **`s.o` 在段 A 内有双重生命周期** ✗：sparse_attn 的输出（`nlh*hd`=4096）与 wo_b 的输出（`dim`=5120）
  ⇒ 融合时不能混用 ✓（`s.xq/s.xsc` 按 `max(...)` 申请，见 :348-352 ✓）。
- **整个 `step_body` 默认在图捕获区内** ✓（`chain_dev.rs:729-740`）⇒ 融合核必须**可捕获**：
  核内不得有 `cudaMalloc`／`cudaStreamSynchronize`／host 交互 ✓，任何分配都要在预热期完成 ✓
  （`kernels.cu:1174-1194` 的 per-device 预热 scratch 是既有正确范式 ✓）。
- **MoE 的 TP 切分轴是 `inter` 而非 expert-parallel** ✓（`moe()` 正文 :1685-1688 ✓；
  `moe_reduce()` 注释 :641 写 expert-parallel ✗ 与实现矛盾 ⇒ 以正文为准 ✓）。
- **hc tail split 的 side stream 在图回放里"有边无并发"（2026-09-11 修复，待上机实测）** ✗→?：
  `DSV41_HC_TAIL_SPLIT` 把 tail 拆成 EARLY（collapse+rmsnorm+fp8，1.7µs，主流）与 LATE
  （ss+mixes+sigmoid+sinkhorn+comb，10.7µs，side stream），fork/join 用 `cudaEventRecord` +
  `cudaStreamWaitEvent`（`dsv41_kernels.cu` 的 `dsv41_hc_front_split`；Rust 侧 `devrt.rs` 建
  side stream、`device.rs::hc_tail_join` 在主流的 hc_post 前 wait）。第 41 轮实测只兑现 −0.20ms
  （理论 −0.86ms）。**根因（两条并存，均与"图回放不继承流优先级"有关）**：
  ① **`cudaStreamCreate` 建的 side stream 是默认优先级** ⇒ LATE 的 1-block 节点与主流上千个投影
     块同优先级竞争 SM，排在队尾；
  ② **`cudaGraphInstantiate` 第三参传 0** ⇒ 即使 side stream 有优先级，图回放仍让**所有节点跑
     在 launch stream 的优先级**上（CUDA 头文件原文：node priority "copied from stream priority
     during stream capture"，只有 `cudaGraphInstantiateFlagUseNodePriority`(=8) 才会用 per-node 优先级）。
  修复：`devrt.rs` 用 `cudaStreamCreateWithPriority(..., cudaStreamNonBlocking, greatest)` 建 side
  stream（greatest 由 `cudaDeviceGetStreamPriorityRange` 查得），并在 `graph_instantiate` 传
  `cudaGraphInstantiateFlagUseNodePriority`（被拒则清错回退 flags=0，只是提示，不影响正确性）；
  另把 LATE 半的 block 从 1024 降到 128（`DSV41_HC_LATE_T`）——LATE 只有 1 个 warp 有活干，
  1024 线程意味着 6 次 32-warp barrier + 向满负荷的 SM 要一个 1024 线程 slot；**逐位不变**
  （全是 elementwise + warp0 sinkhorn），唯一例外是 `DSV41_HC_SS=0` 的自算 ss 路径（读
  `wpart[threadIdx.x>>5]`，需要 warps 0..mix-1）⇒ launcher 在该路径上仍用 1024。
  开关：`DSV41_HC_TAIL_PRIO=0` / `DSV41_GRAPH_NODE_PRIORITY=0` / `DSV41_HC_LATE_T=1024`。
  - **2026-09-11（同日）EARLY 曾搬上 side stream（hc-early-opt），随后回退** ↩：
    EARLY 半（collapse+rmsnorm+fp8）对 dots 的输出零数据依赖（不读 `g_hc_part`），当时据此把
    它挪到 side stream 头部与 dots 并发。**但主流的 wins 是 0**：main 仍要等 side 的
    `early_ev`（即仍付 EARLY 的 1.7µs，只是从"本地 launch"变成"跨流往返"），代价却是
    **每个 front 多 4 个图节点**（record/wait `in_ev` + record/wait `early_ev`）。
    **B（2026-09-11 22:00 曾实施，同日 22:23 由 `9558491` 回退 —— 见块末 ⚠️）**：EARLY 回到主流——主流自己的程序序就同时钉住了它的生产
    （上一段 hc_post / AR fold）与消费（紧随的投影链），**零 event**；dots+LATE 仍在 side。
    发射序列回到**单一 fork/join 对**：`main: EARLY -> record(fork_ev) [-> 投影链 …]`；
    `side: wait(fork_ev) -> dots -> LATE -> record(join_ev)` ⇒ 每 front **净 −2 图节点**
    （×2 front/层 ×40 层 = 80 front ≈ **−0.2ms**）。`in_ev`/`early_ev` 作为**死参数**保留在
    ABI（Rust 侧已不再创建，传 null；`supports_hc_tail_split` 也不再要求它们非空）；
    `fork_ev` 恢复为**必需**（它现在是唯一的 main→side 边：dots 与 LATE 都读 `x`=`s.h`，
    必须钉在主流写之后）。side 链变为 dots(4.9)+LATE(10.7) ≈ **15.6µs**（EARLY 不再在它里面），
    仍远小于 ~50µs 的 hc 投影窗口 ⇒ join 依旧"到达即满足"。
    - ⚠️ **复核更正（HEAD `0c6fa1a`）**：上面 B 块（EARLY 回主流）**已被回退**——`9558491`「EARLY 恢复」后 **EARLY 在侧流头部**发射（`dsv41_kernels.cu:7500` 用 `side`），同一个 `fork_ev` 承载 main→side 输入边与 side→main EARLY-done 边**两条边**（`:7486/7491/7511/7516`），`join_ev` 在 `:7577`；STATUS 实测 B 使 p50 **回退 +0.14ms**（`STATUS.md:6859`）。故每 front 的 event 节点比 B 多 2 个。下方 dots-on-side / DL-merge 两块的表述才是现行状态。
    - **2026-09-11（同日）dots 也搬上 side stream（dots-on-side，保留）**：投影链的第一步
    `lin2`（`chain_dev.rs:2430`）**只读 EARLY 的输出**（`s.xn` = `out`，以及 `xq_of_xn_valid`
    时的 `xq`/`xsc`），而 dots 只写 `g_hc_part`——它唯一的读者是 `hc_mixes_tail_kernel` 的
    LATE 分支（`dsv41_kernels.cu` 的 `HC_TAIL_LATE`）。二者在同一条 side stream 上，
    **流内顺序**即保证 `g_hc_part` 的 publish happens-before LATE 的读 ⇒ 不需要第二条 fork。
    ⇒ main 不再付 dots 的 4.9µs（只付 EARLY 1.7µs；dots-on-side 的净收益 ≈ 3.2µs/front ≈ −0.26ms
    上界，其中 EARLY 那 1.7µs 已被 B 放回主流、与"EARLY 在 side 并被 early_ev 等"等价）。
    ⚠️ **正确性依赖两点**：(a) main 上没有任何 kernel 读 `g_hc_part`；(b) side 对 `s.h` 的读
    （dots 与 LATE 都读 `x`=`s.h`）必须早于 main 对 `s.h` 的写（AR fold 的 `hc_res` / 下一段
    `hc_post`）—— 后者由 `ar_hc_post_fold`/`layer` 里的 `hc_tail_join()`（wait `join_ev`）保证；
    前者由 `fork_ev`（记录在主流 EARLY 之后）保证。
    env gate **复用 `DSV41_HC_TAIL_SPLIT`**（未新增开关；A/B = tail-split on/off）。
    - **2026-09-11（同日）dots 与 LATE 合并成单节点（hc-dl-merge，默认 ON）✓ 待实测**：
      side 链上 dots(4.9µs) 与 LATE(10.7µs) 本是一个依赖对（LATE 读 dots 写的
      `g_hc_part`），却占**两个图节点**（审计 1355 节点 ≈ 2.0ms，~1.5µs/节点 ⇒ 每 front
      省 1 节点 = 2/层 × 61 层）。合并核 `hc_dots_late_kernel`
      （`dsv41_kernels.cu:6632`），grid=`(mix, rows)`、block=`DSV41_HC_DOTS_T`(128)：
      **每个 dot 块 publish 后 `__threadfence()` + `atomicAdd(&g_hc_dl_done[r],1)`，
      看到最后一个计数的块跑 LATE 半**——`hc_pre_persist_mb_kernel` 的机制，
      **无 ticket、无自旋**（这正是 `hc_front_kernel` 的 tail 自旋 +3.2ms 的反例）。
      门：`DSV41_HC_DL_MERGE`（默认 ON；`=0` 回两 launch 作 A/B 臂）。
      **逐位等价**：dot 分支是 `hc_mix_dots_kernel` 逐句照抄（同一 cp.async staging、
      同一 warp0 float4 三累加器 lane 链、同一 ss replay 残数 `m*32`/步长 `mix*32`）；
      tail 是 `hc_mixes_tail_kernel` 的 LATE 分支逐句照抄且 `ss_in==1`（读 dots 的 ss
      分块，只需 warp 0 ⇒ 在 128 线程下依然成立；自算 ss 路径需 ≥mix*32 线程，故
      `DSV41_HC_SS=0` 时 launcher 自动回两 launch）。**K-split 不用**：`ck` 槽会与 ss
      分块槽（第三维 `DSV41_HC_SPREAD_S=8`）冲突且 `split>1` 非逐位 ⇒ 本核固定 split=1。
      **无死锁**：没有任何块等另一个块（选举不是自旋），grid 精确 `(mix, rows)` 无边界
      早退 ⇒ 计数不可能漏块；被选举块在**最后一次读 `g_hc_part` 之后**把计数器复位为 0
      （图回放安全，与 `g_hc_ticket`/`g_hc_mb_done` 同一纪律）。
      **EARLY 不动**：EARLY+LATE 合并已证不可行（main 的等待从 1.7µs 变 17µs，因为 tail
      只能排在整格 drain 之后），EARLY 保持侧流头部的独立 launch。
  ⚠️ 若上机后仍只有 −0.2ms，下一个怀疑对象是**图节点开销本身**（审计：1355 节点 ≈ 2.0ms，
  ~1.5µs/节点；split 每次多 1 个 kernel 节点 + 2 个 event 节点 ⇒ 约 0.3-0.5ms/步），
  而不是调度——判据：nsys 看 tail_late 的 span 是否与投影时间轴重叠。
- **注意力双链 `DSV41_DUAL_CHAIN`（默认 ON，2026-09-11 实现，待上机实测）** ✓：
  `attention()` 里 `lin2`（wq_a+wkv mx2）之后，q 链（`NORM_FUSE`/`lin_rope` + wq_b gemv + rope）
  与 kv 链（`rmsnorm_rope(s.kv)`；`NR_FUSE=0` 或老 `.so` 时退化为 `rmsnorm` + `apply_rope`）互不
  读写，把 kv 链挪到**第二条 side stream** 与 q 链并行。实现要点：
  - **真正的汇合点是 ring append，不是 sparse_attn** ✗（题面如此，但代码里 `ring_win_fuse`/
    `ring_append` 立即读 `s.kv` 写 ring，`sparse_attn` 只读 ring）⇒ join 落在 kv 链之后、
    `ring_win_fuse` 之前（也就在 layer==0 的 debug 读 `s.kv` 之前）。
  - `devrt.rs`：加 `side_stream2`（**默认优先级**——kv 链是填充、q 链才是关键路径，不能反过来抢占）
    + `fork2_ev`/`join2_ev`（`cudaEventDisableTiming`，图捕获内合法），并暴露 `record_event`
    （tail split 的 fork/join 在 C 侧做，双链的两半由 Rust 发，需要这个原语）。
  - **必须独立于 `side_stream`** ✗：hc tail split 的 LATE 半在 `lin2` **之前**就已 fork
    （`hc_mixes_auto`），此刻仍在 side stream 上，共用一条会把两者串行。
  - `device.rs`：`supports_dual_chain` / `dual_chain_fork` / `dual_chain_join`，以及
    `rmsnorm_rope_on` / `rmsnorm_on` / `apply_rope_on`（带 stream 参数；原方法委托，调用点不变）。
  - `chain_dev.rs::attention()`：`lin2` 后 fork → kv 链发 `side_stream2` → 紧接 join；
    `kv_stream = if dual { side2 } else { main }` ⇒ 非 dual 路径与旧行为逐 launch 相同。
  - **位级一致** ✓：kernel 与操作数不变，只是发射流不同；两链共享的只有只读的 `cos/sin/pos_ctr`。
  - 两个静默回退门：① `kv_early` 必须为真——未融合的 `lin(wkv)` 会写**共享**的 `s.xq`/`s.xsc`，
    与 q 链的 `lin_rope`/`lin2_rope` 竞争，该路径保持串行；② `supports_dual_chain()`。
  - ⚠️ **收益预期须修正** ✗：v3 清单实测 `rmsnorm_rope_kernel` = **2.5µs/次**（40 次 = 0.10ms/步）；
    且代码里**不存在名为 `kvb` 的投影**（absorbed MLA：kv_b 已被 wq_b 吸收，`n = nlh*hd`）⇒
    这条改动能兑现的是 **≈0.10ms**，不是题面的 0.42ms。0.42ms 对应的是"把 wkv 的 GEMV 从 `lin2`
    里拆出来一起挪到侧流"（≈9.5µs + 2.5µs），那要先**解融合 lin2**，属另一项改动。
  - 开关：`DSV41_DUAL_CHAIN=0` 回退串行（同二进制 A/B）。
- **MoE 双链 `DSV41_MOE_DUAL`（默认 ON，2026-09-11 实现，待上机实测）** ✓：
  `moe()` 里 routed experts 链（`quant_fp4` → `expert_gate_up_fp4_batched` → `swiglu` →
  `expert_down_reduce_fp4_batched`，写 `s.o`）与 shared expert 链（`quant1(xn)` → `gemm_fp8_mx2`
  w1/w3 → `swiglu_limit_q` → w2 GEMV）互不依赖：**输入都是 `xn`，但走不同量化路径**
  （routed 读 `s.xq4`/`s.xsc4` 的 fp4，shared 读 `s.xq`/`s.xsc` 的 fp8，含 T1 缓存）。
  把 shared 链整段发到**第二条 side stream**（复用 `side_stream2` + `fork2_ev`/`join2_ev`），
  与 routed 链并行。实现要点：
  - **fork 点在 `moe()` 顶部**（`sh_w` 算完之后、gate 之前）✓：shared 链只依赖 `xn`（及其 T1
    fp8），不依赖 gate/route，所以 fork 越早越好；host 侧的发射顺序（fork → routed → shared →
    join）不影响重叠——重叠由"fork event 在 routed 链之前记在主流派"保证。
  - ⚠️ **输出缓冲竞态** ✓：routed 的 `down_reduce` **覆写 `s.o`**，`moe_down_reduce` 也是覆写；
    shared 的 w2 若仍用 A5 融合 epilogue 写 `s.o` 就与 routed 并发写同一 buffer。⇒ `MOE_DUAL`
    下**强制关掉 A5 融合**，w2 恒写**独立的 `s.ex_out`**，join 之后在主流派跑
    `add_inplace(&s.o, &s.ex_out)`——这正是串行非融合路径原有的那条 add，**操作数与顺序完全不变
    ⇒ 位级一致**（A5 注释本就说 fused 与 pair 位级等价）。
  - ⚠️ **`s.ex_act` 共享** ✓：batched routed 链写 `s.ex_act_b`（`[topk][2*inter]`），而**串行
    fallback 循环复用 `s.ex_act`** ——与 shared 链同一个 buffer。⇒ `MOE_DUAL` 只挂在 `batched`
    路径上（`moe_batch() && topk>0 && ne>=2 && supports_moe_batch()`）。
  - ⚠️ **MIX_GATE 必须关** ✓：混合 gate+w1+w3 launch 在主流派写 `s.xq`/`s.ex_act`，而 shared 链的
    `quant1(xn)` 也写 `s.xq` ⇒ 该配置下强制串行（`DSV41_MIX_GATE` 默认本就 OFF）。
  - ⚠️ **`sh_w` 不含 w2** ✓：`sh_w` 只有 w1/w3，shared 下半段还需要 `shared_w2`/`w2_scale` 才会
    真正执行。缺 w2 时整段被跳过，而 join 仍会把**陈旧的 `ex_out`** 并进 `s.o` ⇒ 谓词里必须一并
    检查 `shared_w2(_scale).is_some()`。
  - `device.rs`：新增带 stream 参数的一族 `*_on`（`quant_fp8_on` / `gemm_fp8_mx_on` /
    `gemm_fp8_mx2_on` / `gemm_fp8_mx_add_on` / `swiglu_limit_on` / `swiglu_limit_q_on`；原方法
    委托，调用点不变），`chain_dev.rs` 加 `quant1_on`（同 T1/T2 指针门控）。
  - **复用 `side_stream2`（不新开 side_stream3）** ✓：两条双链同层内**时间窗不重叠**（注意力的 kv
    链在 `lin2` 之后、MoE 的 shared 链在 attention 之后），且 `fork2/join2` 事件本就被设计为
    "每次 record/wait 对在图内按程序序消歧"（hc tail split 已 40×/捕获复用 `fork_ev`/`join_ev`）。
  - 开关：`DSV41_MOE_DUAL=0` 回退串行（同二进制 A/B）。
  - ⚠️ **收益预期须修正** ✗：题面 `−0.88ms(40×22µs)` 假设共享链 22µs 可完全隐藏；实际可隐藏
    部分 ≤ min(routed, shared) 且受 SM/带宽争抢影响。上机后须用 `DSV41_MOE_DUAL=0/1` 同会话背靠背
    单轮实测（`scripts/dsv41_serve_ab.sh`），并同时验证 `opcheck`/faults 与文本。
- **QUANT_FOLD `DSV41_QUANT_FOLD`（实现 2026-09-11；**默认 OFF** —— v12/v13 serve A/B 实测回归，`chain_dev.rs:2403` 读作 `v == "1"`，只能 opt-in）** ✗：
  **回归机制（已定案，勿再归因于占用率）**：`fork_ev` 是 **kernel 级**事件——EARLY 整核跑完才 record，
  而主流的投影组就 wait 在这个 event 上 ⇒ 加进 EARLY 的**任何**工作都直接落在关键路径上
  （fp4 直出 ≈1.8µs × 80 front = +0.14ms，换回的 quant launch 只有 −0.072ms）。
  ⚠️ **占用率不是机制**：`hc_mixes_tail_kernel` 在 decode 是 **grid = rows = 1**（`chain_dev.rs:2186`
  传字面量 1；launcher `dsv41_kernels.cu:8143/8210` 用 `<<<rows, 1024>>>`），整个 grid 只有 **1 个 block**
  ⇒ 「blocks/SM」不是变量（1 个 SM 忙、147 空）。即便多块，1024 线程/块下寄存器阈值只有
  32（2 块）/64（1 块）/超 64=**launch 失败 701**（本文件 `:3570-3590` 记录的 gemv 72-reg 事故），
  而本核带着 LATE sinkhorn + collapse 不可能 ≤32 ⇒ 已钉在 1 块的下界，**没有更小的正整数**。
  次要点：`mags[8]` 是 `const` + `#pragma unroll`，会被折成立即数，**不占 8 个寄存器**。
  实测复核命令（本地无 nvcc 时必须上机）：`nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 --use_fast_math -Xptxas -v -c kernels/cuda/dsv41_kernels.cu`。**

  MoE 的 `quant_fp4(s.xn)`（#30）唯一的输入就是 hc tail **EARLY** collapse 写出的 `s.xn`，
  而 EARLY 已经在同一趟 epilogue 里直出 fp8（T1）。⇒ 在 `hc_mixes_tail_kernel` 的 EARLY
  epilogue 再加一路 **fp4 直出**：`xq4`/`xsc4` 形参非空时，写完 `o_r[c]` 后同址算出该 32-block
  的 e2m1 nibble 与 scale，`chain_dev::moe` 里那次 launch 连同 `xn` 的全局回读一起省掉
  （≈1.8µs × 40/步）。实现要点：
  - **位一致（构造性）** ✓：amax 循环、`fast_round_scale(amax, 1.0f/maxv)`（`maxv=6.0f`，
    与 `quant_fp4_fused_kernel` 逐字同式，注意保留 `1.0f/maxv` 而不是字面折叠）、`±6` clamp、
    最近 e2m1 表循环、`(lo & 0xF) | (hi << 4)`（lo = 偶数下标元素）全部照抄；`v` 就是
    `quant_fp4` 会从 `s.xn` 读回的那个浮点值。**证据** = `crates/ferrite-dsv41/tests/hc_dl_merge_parity.rs`
    在两条 hc 前端路径上各自再跑一次独立 `quant_fp4` 并逐字节比 fp4 输出。
  - **block↔warp 不变量** ✓：`blockDim.x=1024`、`dim=5120` 使每次 pass 里一个 warp 的 32 个 lane
    恰好覆盖一个 32 元素 block（block 下标 = `warp + 32*pass`，与 T1 fp8 的 `xsc[c>>5]` 同映射）
    ⇒ amax 一次 shuffle、无 barrier；nibble 对 `(2t,2t+1)` 不跨 warp，奇数 nibble 用
    `shfl_down(…,1)` 取、只有偶数 lane 落字节。**要求 `dim % 32 == 0`**（否则拒绝折叠，launcher
    静默置空、Rust 侧 `fp4_armed` 也门控同一条件）。
  - **只挂在 FFN 半** ✗：attention 半的 collapse 输出被 fp8 投影（T1）消费，而 `s.xn` 会被
    FFN 半**覆写**，MoE 量化的是覆写后的值 ⇒ attn 调用点传 `null_mut()`，只由 FFN 调用点穿
    `s.xq4`/`s.xsc4`。
  - **流序** ✓：EARLY 在 `side` 上，但 split 的 side→main `fork_ev`（edge 2）在 EARLY 之后
    记录、main 在其上 wait，因此 `xq4`/`xsc4` 与 out/xq/xsc 一起对 main 可见；MoE 在该 wait 之后。
  - **consume-once 标志** `s.xq4_of_xn_valid`（同 T1 指针门控）：`hc_mixes_auto` 按实际走到的
    分支置位（split / 单 launch `hc_front` 为真；persistent 形态没有 fp4 epilogue ⇒ 假），
    `moe()` 用 `replace(false)` 消费。
    - ⚠️ `dsv41_hc_front` 的**合并单核形态**（`DSV41_HC_MERGE=1`，默认 OFF）用的是
      `hc_front_kernel` 的 collapse，没有 fp4 epilogue ⇒ 该 gate 打开且请求了 fp4 时强制走
      两段 launch，否则会读陈旧 `xq4`。
  - **ABI 变更**：`hc_mixes_tail_kernel` / `dsv41_hc_front` / `dsv41_hc_front_split` 均在
    `xsc` 之后插入 `uint8_t* xq4, float* xsc4`；`device.rs` 的函数指针类型与 wrapper、以及
    `hc_persist_parity.rs` / `hc_dl_merge_parity.rs` 两个测试的调用点须同步。旧 `.so` 与新
    Rust 二进制**不兼容**（同一 ABI 版本内变更）⇒ 需重编 `libferrite_kernels.so`。
  - 开关：`DSV41_QUANT_FOLD=0` 回退（两处传 null ⇒ 独立 launch 照跑，纯 A/B 臂）。
- **compress 侧流 `DSV41_COMPRESS_SIDE`（默认 ON，2026-09-11 实现，待上机实测）** ✓：
  kv-source 层（2/8/14/20）的 4 个 compress launch（`lin_f32` kvp/scp + `compressor_pool` +
  `compress_commit` ≈ 30µs/层）与 q 链、kv 链都零依赖，整段挪到**第三条 side stream**
  （`side_stream3`）与主流并行。实现要点：
  - **必须新开 `side_stream3`，不能复用 `side_stream2`** ✓：kv 链与 compress 的**时间窗完全重叠**
    （都在 `lin2` 之后、各自消费者之前），共用一条会把 10.6µs 的 kv 链串到 30µs 的 compress 后面
    ⇒ 40.6µs 远超 ~14µs 的窗口。第三条流默认优先级（compress 是填充，不能抢占 q 链）。
  - **fork 点 = `lin2` 之后、与 `dual_chain_fork` 同点**（`chain_dev.rs::attention()` 内两条
    fork 相邻）：此刻 `s.xn` 已定稿（pre-attention rmsnorm 写入），compress 只读它 + `pos_ctr` +
    本层 state。
  - ⚠️ **join 点不是 sparse_attn 前，而是 indexer 之前** ✗（题面给的是 `sparse_attn:2822`）：
    indexer 的 key 发布读**本层 `latent`**（`compressor_pool` 写，`chain_dev.rs::indexer` 里
    `lin_bf16(self.layers[layer].latent, ...)`）以及 **compress 推进的 `clen` 设备计数器**
    （`index_k_publish`/`apply_rope` 的 base=`clen+layer`）⇒ 对 2/8/14/20 这些既是 kv_source
    又是 index_source 的层，join 必须早于 indexer。最终落在 **`window_idxs` 之后、indexer 之前**
    （`window_idxs` 只读 `pos_ctr`，可与 compress 并行）——这是"仍早于首个消费者"的最晚点。
    非 index-source 的 kv layer（本模型没有）才可能推到 sparse_attn 前。
  - **缓冲不相交** ✓：compress 只写本层私有 `kvp`/`scp`/`state_kv`/`state_score`/`latent`/`out_rows`
    + ring 的**压缩行**（`window + clen`）+ `clen[layer]`；q 链是 `qr`/`q`/`xq`/`xsc`，kv 链是 `s.kv`，
    `ring_win_fuse`/`ring_append` 只碰 ring 的**窗口行**（`pos % win`）与 `idxs[0,win)` ⇒ 零冲突，
    因此三流并发仍**位级一致**（kernel 与操作数不变，只是发射流不同）。
  - `devrt.rs`：加 `side_stream3` + `fork3_ev`/`join3_ev`（`cudaEventDisableTiming`，图捕获内合法）
    + `zero_at_on`；`device.rs`：`supports_compress_side` / `side_stream3` / `compress_side_fork` /
    `compress_side_join`，以及 `gemv_f32_on` / `compressor_pool_on` / `compress_commit_on` /
    `zero_on`（原方法委托，调用点不变）；`chain_dev.rs`：`compress_side()` 门 + `lin_f32_on` +
    `compress_on(layer, pos, stream)`（`compress()` 变成传主流派的薄包装）。
  - 静默回退门：`cublas_m1()`（cuBLAS-M1 只有一个 handle、绑定主流派，不能发侧流；默认 OFF）、
    `supports_compress_side()`、以及本层 `comp_wkv`/`comp_norm` 是否存在（与 `compress()` 自身的
    early-return 同谓词）。
  - **收益预期** ⚠️：compress ≈30µs > 窗口 ~14µs（q 链 13.5µs + `ring_win_fuse`/`window_idxs` ~1µs），
    只能藏住窗口那部分 ⇒ 预期 **−0.05~0.06ms/步**（4 层 × ~14µs），不是 30µs×4。
  - **gate 分析（2026-09-11）**：join 不能后移——`indexer` 的 key 发布是本层 `latent` + `clen` 的
    首个消费者，而后续的 `sparse_attn` 读 ring 的压缩行；两者都在 join 之后，故 join 已是
    "最晚的合法点"。三条 2/8/14（ratio=2，kv+index source）各暴露 `30 − 13.5 ≈ 16.5µs`。
    真正可行的只有**缩短 compress 关键路径**：(a) `COMPRESS_FUSE` 省 2 个 1-block launch；
    (b) 把 `join` 拆成"latent 就绪"与"ring 行就绪"两条边（当前实现里 commit 紧跟 pool，
    拆了也只省一条边）；(c) 让 sparse_attn 用**上一步**的压缩行（滞后一步，非位级一致，需慎重）；
    (d) fork 提前——不可行，`s.xn`（pre-attention rmsnorm）就是 compress 的输入，fork 点已在它之后。
  - 开关：`DSV41_COMPRESS_SIDE=0` 回退串行（同二进制 A/B）。上机须同会话背靠背单轮实测，并
    同时验证 `opcheck`/faults 与文本。
- **compress 3-launch 合一 `DSV41_COMPRESS_FUSE`（默认 ON，2026-09-11）** ✓：
  decode 下（`b == seqlen == 1`、`ratio > 1`、`pos > 0`）`compressor_state`（decode 分支）+ 
  `compressor_pool`（mode 2）+ `compress_commit` 是**同一条严格串行链**（写 state → 读 state 写
  latent/out_rows → 读 latent/out_rows 写 ring/clen），且**每级本来就只有一个 block**
  （state 用 2 block 只因 512 元素 / 256 线程），因此合成**单 kernel `compressor_fused_kernel`
  （`dsv41_kernels.cu:2638`，`<<<1,128>>>`）**、级间用 `__syncthreads()`。
  - **位级一致的三条理由**：① state 是逐元素 store，目标集合相同、顺序无关；② pool 保持
    `nthr == 128` + 同一 `for (c = tid; c < hd; c += nthr)` 通道归属 ⇒ 同一 `s_red[32]`
    warp + 顺序跨 warp 的 RMSNorm 树（**换 block 大小会重分组求和、字节就变了**，故锁 128）；
    ③ commit 的 rope/store 循环与 thread-0 `*clen++` 是 `compress_commit_kernel`
    （`dsv41_glue.cu:658`）的**逐字拷贝**——两处必须同步改。
  - ⚠️ 原三核的**早退不能照搬成 `return`**（后面还有 `__syncthreads()` ⇒ UB）：改为用统一的
    `out_rows_val` 守卫（未完成一组时：只做 state 搬运，跳过 pool 写与整个 commit，语义同原早退）。
  - 回退：`DSV41_COMPRESS_FUSE=0`，或旧 `.so` 无 `dsv41_compressor_fused`（`supports_compress_fuse()`
    为假）⇒ 走原 3-launch；`ratio == 1` / `pos == 0` 也走原路径（state 映射不同）。
  - 收益：每 kv-source 层省 2 次 1-block launch（≈2–4µs）×3 层（2/8/14）；注意**它同时缩短
    compress 的 30µs 关键路径**，这正是 §compress join 暴露窗口的直接缓解（见下）。
- **三条侧流的优先级分配（`devrt.rs::create_side_stream`，每流独立 env gate）** ✓：
  优先级只在**图回放**且节点 READY 时决定谁先拿 SM（`cudaGraphInstantiateFlagUseNodePriority`，
  全局一个标志），因此只有"最长且最晚被消费 = 真正卡窗口"的链才值得 greatest。
  - `DSV41_HC_TAIL_PRIO`（tail dots+LATE ~15.6µs，hc 投影窗口 ~50µs）：**默认已改为 default(0)**
    （C，2026-09-11）。原为 greatest，后判为**过度分配**：LATE 是**单 block** 核（`g_hc_late_t`
    个 warp 里只有 1 个 warp 有活干），被 greatest 提前放行只会占住一个它用不满的 SM、并推迟
    它本该藏在其下的块并行工作；它又有 ~34µs 余量，延迟本就不敏感。A/B：`=greatest`（或 `=1`）恢复旧行为。
  - `DSV41_DUAL_PRIO`（kv 链 ~10.6µs vs q 链 13.5µs；MoE shared ~22µs vs routed ~42µs）：默认
    **default(0)**。它两条链都在主流自己的串行路径上，压主流不划算；需要时可 `=mid`/`=greatest`。
  - `DSV41_COMPRESS_PRIO`（compress ~30µs，是三条侧流里最长的，且汇合点最晚——在 indexer/
    `sparse_attn` 之前）：默认 **greatest**。它是窗口被 SM 打满时唯一真正 gate 住 attention 的链。
  - ⚠️ 节点优先级标志的开启条件已从"tail 流有优先级"改为"**任一**侧流有非默认优先级"，否则
    把 `HC_TAIL_PRIO=0` 会连带静默关掉 side_stream2/3 的优先级。
  - 判据：nsys `--cuda-graph-trace=node` 看三条侧链的 span 是否落在各自窗口内；stderr 的
    `[hc_tail]/[dual_chain]/[compress_side] side stream priority = N` 是优先级真的进了 capture 的唯一证据。

---

## 7. launcher decline-code 约定 + 全量审计（2026-09-11，r42/r43）

**背景**：r42/r43 的 serve 崩溃根因是 decline 哨兵与 CUDA 错误码**语义撞车**：
`cudaErrorInvalidValue == 1`，而多个融合 launcher 用 `return 1` 表示"形状不支持，请回退"。
后果有两个方向——
1. **误判为回退**：C 侧真实错误恰好是 1 时，Rust 的 `rc == 1` 把它当成优雅 decline 静默吞掉；
   launcher 若不 `cudaGetLastError()` 清 sticky，错误还会**泄漏到下一个 launcher** 并让 serve 崩
   （r42 的实际机制：SetAttribute 失败 → sticky 存活 → 下一个 `quant_fp8` 的 kerr 报错）。
2. **误判为硬错**：Rust 写 `rc == 2` 而 C 侧仍 `return 1`（r42 的 rope_norm/f32 就是这样），
   真 decline 被当错误抛出 ⇒ 融合路径永远不生效。

**约定（新符号起）**：**decline 哨兵一律用 `2`，永不用 `1`**（2 在实际路径上远不可能被
`cudaError_t` 真实返回，1 是最常见的 InvalidValue）。C 侧 `return 2;` ↔ Rust 侧 `if rc == 2 { Ok(false) }`。

### 7.1 审计结果（全部 launch 的 decline 点）

| launcher（C） | decline 码 | Rust 检查 | 判定 |
|---|---|---|---|
| `dsv41_gemm_fp8_mx_rope` | `2`（r43 修） | `gemm_fp8_mx_rope: rc==2`（r43 修） | ✅ 已迁移 |
| `dsv41_gemm_fp8_mx2_rope` | `2`（r43 修） | `gemm_fp8_mx2_rope: rc==2`（r43 修） | ✅ 已迁移 |
| `dsv41_gemm_fp8_mx_add` | `2`（r43 修） | `gemm_fp8_mx_add_on: rc==2`（r43 修） | ✅ 已迁移 |
| `dsv41_gemm_fp8_mx_rope_norm` | `2` | `rc==2` | ✅ r42-fix（4 处） |
| `dsv41_gemm_fp8_mx_f32` | `2` | `rc==2` | ✅ r42-fix（2 处） |
| `dsv41_apply_rope_q` | `1`（字面量） | `apply_rope_q: rc==1` | ⚠️ 旧符号，注释标注，未改 |
| `dsv41_rmsnorm_q` | `1`（字面量） | `rmsnorm_q: rc==1` | ⚠️ 旧符号，注释标注，未改 |
| `dsv41_swiglu_limit_q` | `1`（字面量） | `swiglu_limit_q_on: rc==1` | ⚠️ 旧符号，注释标注，未改 |
| `dsv41_gemm_fp8_mx`（xq/staging/shape） | `cudaErrorInvalidValue` | `gemm_fp8_mx_q: rc==1` | ⚠️ 旧符号；且 2542/2580 的 SetAttribute 失败路径**不清 sticky**（同 r42 crash 类）→ 已注释 |
| `dsv41_gemm_fp8_mx2`（shape） | `cudaErrorInvalidValue` | `gemm_fp8_mx2_on: rc==1` | ⚠️ 旧符号（r42 已给其 SetAttribute 路径加 sticky clear） |
| `dsv41_gemm_bf16_fp8x2`（shape） | `cudaErrorInvalidValue` | `gemm_bf16_fp8x2: rc==1` | ⚠️ 旧符号；3015 的 SetAttribute 失败路径不清 sticky → 已注释 |
| `dsv41_argmax_sliced`（shape） | `cudaErrorInvalidValue` | `argmax_sliced: rc==1` | ⚠️ 旧符号 |
| `ferrite_p2p_ar_v5_hcpost`（shape） | `cudaErrorInvalidValue` | `p2p_ar_v5_hcpost: rc==1` | ⚠️ 旧符号 |
| `dsv41_hc_front{,_split,_persist,_persist_mb}` | `cudaErrorInvalidValue`（gate off / 形状） | `rc==1` | ⚠️ 旧符号（gate-off 就是靠 InvalidValue 表意）；但 `_split` 的 3 处事件失败路径已在 r44 加 sticky clear（见 §7.2） |
| `dsv41_interleave_gateup_fp4`（r43 新增） | 无 decline（坏形状直接 `cudaErrorInvalidValue`） | `kerr`，无 rc 检查 | ✅ 硬报错语义，安全 |
| `dsv41_quant_fp4_fused`（r42 新增） | 无 decline（不满足条件就走 legacy 两 launch） | — | ✅ 安全 |

**结论**：一致性矩阵**全部匹配**（无 "C 返回 1 / Rust 查 2" 或 "C decline 无 Rust 检查" 的破口）。
所有残余风险收敛为同一类：**旧符号的 decline 码 = `cudaErrorInvalidValue`**，C 侧真实错误 1 会被
Rust 的 `rc==1` 读成回退。按"rounds 37-41 已冻结、改动有回归风险"的原则**保持现状，仅在源码注释里
标注隐患**（`device.rs:973` / `:1821`、`dsv41_kernels.cu` apply_rope_q/rmsnorm_q 头注释、
`dsv41_glue.cu` swiglu_limit_q 头注释）。

**下次改动纪律**：
- 新增/重触任何带 decline 的 launcher ⇒ 用 `2`，Rust 配 `rc == 2`；**不要**再出现 `return 1` 哨兵。
- 若某旧符号要重新启用（如 `DSV41_WO_QUANT_FUSE=1` 重新打开 `gemm_fp8_mx_q`），先把它迁到 2，
  否则 r42 的 sticky-leak 崩溃会复现。
- `dsv41_kernels.cu` gemv 段（约 :2500-2900）有其它 agent 在飞改动时，只动 `return 1→2` 的行。

### 7.2 event-sticky-fix（r44，2026-09-11）

**背景**：b300-4 驱动 wedge 事故的 first-forward-trace 追到 `dsv41_hc_front_split`：3 处事件调用
（`cudaEventRecord(fork_ev,s)` / `cudaStreamWaitEvent(side,fork_ev,0)` / `cudaEventRecord(join_ev,side)`）
的失败路径都写 `if (e != cudaSuccess) return (int)e;`，**不调用 `cudaGetLastError()`**。驱动状态
异常时这些调用失败 → sticky 存留 → 被下一个 launcher（`quant_fp8`）的 `kerr` 读到 → **归因错位**
（与 r42 的 SetAttribute sticky-leak 同类）。

**修复**：这 3 处改为
```c
if (e != cudaSuccess) {
    (void)cudaGetLastError();   // clear the sticky flag before reporting
    return (int)e;
}
```
同文件 r42 已给 `cudaFuncSetAttribute` 的失败路径加过同款 clear（`hc_front` / `hc_front_split` /
`hc_front_persist` / `hc_front_persist_mb`），本次只补齐事件调用这一类。

**审计**：`dsv41_kernels.cu` 内**只有** `dsv41_hc_front_split` 一个 launcher 直接调 `cudaEventRecord` /
`cudaStreamWaitEvent`；attention/moe 双链的 `fork2/join2` 事件由 **Rust** 侧发起
（`device.rs::dual_chain_fork/join` → `devrt.rs::record_event/stream_wait_event` → `kerr`），
没有 C 侧 `return (int)e` 形态，无需同款修改（Rust 侧返回 `Result`，由调用链传播）。

**纪律**：任何**非 `cudaGetLastError()` 来源**的 CUDA 调用（`cudaEventRecord`、`cudaStreamWaitEvent`、
`cudaFuncSetAttribute`、`cudaMalloc` …）在返回错误码前都必须先 `(void)cudaGetLastError()` 清 sticky，
否则错误会泄漏给下一个 launcher 造成归因错位。
