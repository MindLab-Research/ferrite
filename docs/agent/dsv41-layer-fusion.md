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
| 30 | `quant_fp4`（激活）| :1783 | rows=1, cols=5120, block=32 |
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
