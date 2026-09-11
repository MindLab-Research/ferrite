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
| 15 | `comp_placeholder`（无 indexer 时）| :1340 | 写 `idxs[win..]` |
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
| 30 | `quant_fp4`（激活）| :1783 | rows=1, cols=5120, block=32。**已融合**：`DSV41_QUANT_FP4_FUSE`（默认 ON）时 `dsv41_quant_fp4` 发射单核 `quant_fp4_fused_kernel`（量化+打包一步，bit-exact），旧 `quant_kernel<1>` + `fp4_pack_kernel` 两段路径保留为回退（env=0，或 block 奇数/>256） |
| 31 | **batched 路径（`moe_batch()` 代码默认 OFF**，:190 `unwrap_or(false)`）| :2317-2432 | `expert_gate_up_fp4_batched`（smem **20480 B**）→（`gateup_fused` 开时**跳过** `swiglu_limit_batched`；`gateup_fused = DSV41_GATEUP_FUSE!=0 && supports_gateup_fuse() && expert_fp4_mode()==2`，:2344/:2383 —— 必须与 `.cu:1367` 的 `g_fuse && g_expert_fp4_mode==2 && dim%512==0` 逐字镜像）→ **down 方向二选一**：`DSV41_DOWN_FUSE`（:200，默认 OFF）⇒ `expert_down_reduce_fp4_batched` **一次启动**（grid `⌈dim/8⌉`、串行升序 slot、`out` 覆盖写，替代下两行）；否则 `expert_down_fp4_batched`（smem `inter_local*4`）+ `moe_down_reduce`（定序求和 ✓）|
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
  `gemm_fp8_gemv_kernel` 14 → 22 参数（`dsv41_kernels.cu:1722`：epi_add + AR 5 个 + B1 的
  `xq`/`xsc`），但以生产 flag
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
