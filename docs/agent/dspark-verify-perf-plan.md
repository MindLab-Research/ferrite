# verify 链性能优化路线（户部 · 资源与性能）

> 目标：verify 38.9ms/步 → ≤5ms；draft 7.2ms → ≤3ms。400 tok/s 的必要条件。
> 方法：只读代码 + 用仓库内实测数字核算。**无法在本机跑 GPU（无 nvidia-smi/B300）**，
> 所有时间均为「launch 计数（代码精确）× 仓库已实测的每核成本」的推算，
> 凡未经本仓库实测支撑的条目已在 §9 标为「待验证」。

---

## 【分析范围】

| 文件 | 内容 |
|---|---|
| `crates/ferrite-models/src/dsv41/chain_dev.rs` | `step_rows` / `layer_rows` / `attention_rows` / `moe_rows` / `indexer_rows` / `compress_rows` / `engram_apply_rows` / `dspark_snapshot` / `dspark_rollback` |
| `crates/ferrite-models/src/dsv41/dspark_dev.rs` | `draft_forward` / `draft_attention` / `draft_moe` / `draft_head` / `rope_at` / `ensure_idxs` |
| `crates/ferrite-models/src/dsv41/device.rs` | 全部 launcher（launch 计数依据） |
| `crates/ferrite-kernel/src/cuda.rs` / `ferrite-exec/src/tp.rs` | CUDA graph 捕获/回放纪律、`small_n_rows` 先例 |
| `kernels/cuda/dsv41_kernels.cu` / `dsv41_glue.cu` / `ferrite_kernels.cu` | `gemm_fp8_gemv_kernel` / `gemv_bf16_kernel` / `gemv_bf16_v2_kernel`(nrows!) / `sparse_attn_*` / `hc_mixes` |

生产配置（`/tmp/dsv41/config.json` + `compress_ratios`）：
40 层 + 3 MTP；dim 5120 / hd 512 / nh 64 / 1 KV head / q_lora 1280 / o_lora 1024 / o_groups 8；
window 128 / index_topk 512 / 384 routed + 1 shared expert / topk 6 / hc 4；
vocab 129280；TP8（`nlh=8, nlg=1`）；verify `m = DSPARK_DRAFTS+1 = 6`。

---

## 1. verify 38.9ms 的分解（launch 数 × 每 launch 成本）

### 1.1 先给结论：它是一条 **~7200 次 launch 的裸流（无 CUDA graph）**

`step_rows` 里没有 `graph_capture_begin/end`，而且开头有 3 次 H2D、结尾有 1 次 D2H
（`ul_i32`×2 / `ul_f32` / `download_u8`），**它根本无法被捕获成图** ——
而主链 `step_dev` 是 `DSV41_GRAPH_STEP=1`（默认）的整步图。
这是 verify 与主链最大的结构性差异。

按代码逐条计数（TP8，m=6，含 `quant1`）：

| 段 | kernels | 次数/层 | ×40 层 |
|---|---|---|---|
| 投影 wq_a (`quant1`+`gemm`) | 逐行 | 12 | 480 |
| 投影 wkv | 逐行 | 12 | 480 |
| `rmsnorm(q_norm, rows=1)` | 逐行 | 6 | 240 |
| 投影 wq_b | 逐行 | 12 | 480 |
| `apply_rope(q)` | 逐行 | 6 | 240 |
| `rmsnorm(kv, rows=6)` + `apply_rope(kv, rows=6)` | **已多行** | 2 | 80 |
| `ring_append` + `window_idxs` | 逐行交替 | 12 | 480 |
| `sparse_attn`（split+merge） | 逐行 | 12 | 480 |
| `apply_rope(o)` 反 rope | 逐行 | 6 | 240 |
| wo：`quant1(o)`+`wo_a`+`quant1(wo)`+`wo_b` | 逐行 | 24 | 960 |
| attention AR（store+reduce+stamp） | | 3 | 120 |
| hc 链：`hc_mixes`×2 / `hc_collapse`×2 / `rmsnorm`×2 / `hc_post`×2 / `memcpy_d2d`×2 | | 10 | 400 |
| MoE gate `gemv_bf16` | 逐行 | 6 | 240 |
| `route_topk` + `quant_fp4` | | 2 | 80 |
| expert `gate_up` | 逐行 | 6 | 240 |
| expert `down` | 逐行 | 6 | 240 |
| 共享专家 `quant1`+`gemm_mx2`+`swiglu`+`quant1`+`gemm_add` | 逐行×5 | 30 | 1200 |
| MoE AR | | 3 | 120 |
| **小计** | | **~170** | **~6800** |
| indexer（8 层）`publish`×4 + 逐行×5 | | +34 | +272 |
| compressor（4 层）逐行×4 | | +24 | +96 |
| head `gemv_bf16` + `argmax`（逐行） | | — | 12 |
| engram（2 层）`gather`×6 + AR + quant + gemm + apply | | — | 24 |
| `embed_expand` + 末尾 `hc_collapse`+`rmsnorm` | | — | 3 |
| **总计** | | | **≈7220** |

**每 launch 平均 38.9ms / 7220 = 5.4µs。**
对照本仓库实测（`graph_bench`，B300 sm_103，2000 节点）：

| 量 | 实测 |
|---|---|
| 流式 host submit / launch | **2.904µs** |
| 图内空节点 dispatch 地板 | **0.411µs/node** |
| 图比流每节点省的开销 | **2.493µs/node** |
| 单核 exec（globaltimer） | 1.408µs |

⇒ 5.4µs/launch = 2.9µs 提交 + ~2.5µs 尾延迟/执行，**与「裸流发射」的画像完全吻合**。

### 1.2 按段的时间分解（用仓库实测每核中位数）

实测基准（`STATUS.md` 6.15ms TP8 基线，median/次）：
`gemm_fp8_gemv` 10.94µs · `expert gateup` 24.1µs · `expert down` 17.4µs ·
`gemv_bf16_v2` 9.1µs · `sparse_attn`(3 深预取) ≈10.5µs · `hc_mixes` ≈7.8µs ·
`quant_e4m3` 2.4µs · `rmsnorm` 3.5µs · AR v5 = 7.5+5.6+4.2 = 17.3µs ·
`lm_head`(bf16, 全词表 129280) **298µs** / (切分 16160) 48.5µs。

| 段 | launch/步 | 估 ms | 占比 | 备注 |
|---|---|---|---|---|
| **投影族**（wq_a/wkv/wq_b/wo_a/wo_b + 其 `quant1`） | 2400 | **10.5** | 27% | gemm 1200 次×~8µs + quant 1200 次×2.4µs；**权重被读 6 遍** |
| **MoE 路由专家**（gate+gate_up+down） | 800 | **12.1** | 31% | 240×24.1 + 240×17.4 + 240×9.1；字节数本身就是 6× |
| **MoE 共享专家** | 1200 | **4.0** | 10% | 5 launch/行×6 行×40 层，逐行 |
| 稀疏注意力 | 720 | **3.2** | 8% | 逐行调用 + 行内 split/merge 两发 |
| head + argmax | 12 | **2.2** | 6% | head **未切分**：6×298µs = 1.79ms ← 单点最大 |
| indexer + compressor | 368 | **1.7** | 4% | 8 层 + 4 层 |
| AR v5（80 次 × 3 核） | 240 | **1.4** | 4% | 17.3µs/次，协议地板 |
| hc 链 + memcpy | 400 | **1.6** | 4% | 7.8µs/次 |
| rmsnorm | 360 | **1.3** | 3% | |
| RoPE | 360 | **1.1** | 3% | |
| 因果窗口 ring+window | 480 | **1.0** | 3% | 逐行交替（因果正确性要求） |
| engram + embed + 收尾 | 33 | **0.5** | 1% | |
| **合计** | **≈7220** | **≈38.9** | 100% | ✓ 与实测吻合 |

### 1.3 根因归纳（三条，而非三条并列）

1. **没有 CUDA graph**：7220 次裸流发射，每次 ~2.9µs = **~21ms 的发射开销**，
   其中相当一部分没有被 kernel 执行掩盖（平均核只有 ~2.5µs）。
2. **权重按行重复读**：投影族每层每次都重读 wq_a/wkv/wq_b/wo_a/wo_b
   （每行每层 23.8MB fp8，×6 行 ×40 层 = **5.7GB**，而主链只读 952MB）。
   按仓库实测的 gemv 有效带宽 373 GB/s ⇒ 15.3ms —— 与 1.2 表的「投影族 10.5ms」同量级。
3. **head 未切分**：`step_rows` 直接用 `cfg.vocab_size` + `head.ptr()`（`Shard::Replicated`，全量 1.32GB/rank），
   6 行 × 298µs = **1.79ms**。主链早已默认切分（48.5µs/次），verify 没跟。
   （题面里「158MB 权重读 ×6」是**切分后**的尺寸；当前 verify 读的是**未切分**的全量 —— 这一条要更正。）

---

## 2. 多行 GEMV kernel 设计（数值域安全的加速主路径）

### 2.1 先例在仓库里已经存在（不要重造）

`kernels/cuda/ferrite_kernels.cu:2880` 的 `gemv_bf16_v2_kernel` 已经带 `nrows` 参数，
注释明确写着：

> "the weight row is loaded and fp8->half2 converted **ONCE and reused for both tokens' dots** …
> **Each token keeps its own accumulation order -> bit-identical output**."

且 `crates/ferrite-exec/src/tp.rs:3268` 用它把 MTP verify 链的 n 行合成 **ONE launch**
（`small_n_rows` → "was n single-row launches, the MTP verify chain's 19880 small-graph-node cause"）。

**dsv41 的 `step_rows` 恰恰没有走这条路** —— 它把 `device.rs` 的 M=1 launcher 逐行调。

### 2.2 kernel 设计：weight-stationary + 行独立累加

以 `gemm_fp8_gemv_kernel`（M=1）为蓝本，新增 `m` 维：

```
// grid: (ceil(out_f / rows_per_block), )   block: 256
// 每一 block 负责 out 的一个 tile；权重 tile 只 decode 一次
for (row = tile_start; row < tile_end; ++row) {
    wrow = s_w[row - tile_start];          // LUT 解码一次（原有路径）
    float acc[MAX_M];
    #pragma unroll
    for (r = 0; r < m; ++r) acc[r] = 0.f;
    for (c = lane; c < k; c += 32) {        // ← 与 M=1 版本逐字相同的 c 序列
        const float wv = wrow[c];
        #pragma unroll
        for (r = 0; r < m; ++r)             // ← 行是独立的累加链
            acc[r] = __fmaf_rn(wv, x[r*k + c], acc[r]);
    }
    #pragma unroll
    for (r = 0; r < m; ++r) {               // ← 每条链跑同一个归约树
        float a = acc[r];
        for (off = 16; off; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0) out[r*out_f + row] = a;
    }
}
```

寄存器预算是有先例的：`gemv_bf16_v2_kernel` 已经做到 `R=8` 行/组 + `float4 xc[2][4]`，
m=6 在预算内（`__launch_bounds__(256,4)` 的 64 寄存器墙需要复核 → §9）。

### 2.3 **数值域与单行 GEMV 逐位一致的条件（证明）**

设单行 kernel 对输出 (row) 计算：

```
y(row) = T( Σ_{c ≡ lane (mod 32), c ↑} w[row][c] * x[c] )
```
其中 `T` 是固定的 shuffle 归约树（`off = 16,8,4,2,1`），`Σ` 是 lane 内**严格升序**的 FMA 链。

多行 kernel 对每个 `(row, r)` 计算：

```
y(row, r) = T( Σ_{c ≡ lane (mod 32), c ↑} w[row][c] * x[r][c] )
```

**逐位一致成立，当且仅当下列条件全部满足：**

| # | 条件 | 本设计 |
|---|---|---|
| C1 | 每个 (row, r) 的 **K 迭代顺序**与单行版逐字相同（lane 步长 32、升序） | ✓ 共用同一个 c-loop |
| C2 | 权重 decode 出的 f32 值相同，且在表达式中的位置相同 | ✓ 同一 `s_w`/LUT 路径，同一 FMA |
| C3 | 归约树 `T` 逐位复现（相同 shfl 次序与 mask） | ✓ 每 r 各跑一遍同一个 T |
| C4 | **不做跨行重结合**（不得把两行相加、不得跨行共用部分和） | ✓ `acc[r]` 互相独立 |
| C5 | K-split 变体（WPR 分片 + smem fold）若保留，则每个 r 的 fold 次序不变 | ✓ 只给累加器数组加一行维度，fold 结构不动 |
| C6 | 编译器不得重排/重结合 c-loop（禁 `#pragma unroll` 改变次序、禁 fast-math 重结合） | 需 code review + 逐位回归（§9） |

⇒ **结论**：只要「同 K 序 + 同归约树 + 同 decode」，多行 kernel 对**任意 m** 都与单行 GEMV 逐位一致。
这比 GLM 的 `small_n_rows`（n≤3 退回逐行 GEMV 以保数值域）**更强**：
GLM 是「小 n 用逐行来避坑」，我们是「把逐行 kernel 的行维度显式化，从而对所有 m 都逐位一致」。
两个先例佐证：`tp.rs` 注释 "Per-row accumulation order (warp shuffle + WPR root) is unchanged,
so the greedy argmax is bit-identical; only the launch/graph-node count drops n×"。

### 2.4 收益

| 项 | 现状 | 多行后 | 依据 |
|---|---|---|---|
| 投影 launch | 2400/步 | **~600/步**（5 gemm + 5 quant 化到每层 1 发，×40） | 计数 |
| 投影权重读 | 5.7GB/步 | **952MB/步** | 23.8MB/层 |
| MoE 共享专家 launch | 1200 | **240** | |
| MoE 路由专家 launch | 720 | **120**（grid 加 y/z 维放 rows） | 见 §2.5 |
| head | 6×298µs = 1.79ms | **1 发 ≈ 0.30ms**（未切分）/ **0.05ms**（+切分） | 全词表 298µs、切分 48.5µs |

### 2.5 路由专家的多行（注意：**字节数不减**）

`dsv41_expert_gate_up_fp4_batched` / `_down_reduce_fp4_batched` 的 `rows` 参数
**不进 grid**（grid 是 `(n_total/warps, slots)`），所以现在必须逐行发。
修法：把 rows 放进 `grid.z`（或折叠进 y），每个 `(row, slot)` 独立 —— 同样是 C1–C6 的拷贝，
`moe_rows` 的文档已经说明「one activation row per launch」，改 grid 只是一次机械扩展。

**但必须诚实**：每行选自己的 top-6/384 专家 ⇒ **专家权重字节数真的 ×6**
（6 行 ×6 专家 ×2.46MB ×40 层 = **3.5GB**）。多行只省 launch 数与提高 issue 效率，
不省字节。这块的收益上限取决于 `expert-cpasync-full` / tcgen05（§5）。

---

## 3. CUDA graph 的 verify 图（DRY → rollback → CAPTURE）

### 3.1 捕获前必须解掉的 4 个「主机侧依赖」

| 现状（step_rows 内） | 问题 | 修法 |
|---|---|---|
| `ul_i32(ids_r)` / `ul_i32(pos_rows)` / `ul_f32(premix_r)` | 3 × H2D，捕获非法 | `ids_r`/`premix_r` 每步不变（premix 已是常量）+ `pos_rows` 改由**一个 tiny kernel** 从 `*pos_ctr` 生成 `pos_ctr + r` |
| `dspark_snapshot` / `rollback` 的 host 计算 slot `(pos+1+j)%win` → ~520 次小 D2D | 主机地址计算，捕获非法 | 换成 device-side `ring_save` / `ring_restore` 两个 kernel（grid=层×行），顺带把 520 次发射压成 2 次 |
| 末尾 `download_u8(argmax_r)` | D2H，捕获非法 | 移出图：捕获只到 argmax 写 device，回放后单独 D2H |
| 80 次 `all_reduce_inplace` | 需要 device-side epoch | v5 协议**本来就是** capture-safe（`graph || env` 强制 `ar_v5`，mega-graph 已验证）→ 直接可用 |

### 3.2 纪律（照抄仓库已验证的 mega-graph 模式）

```
1) DRY   : 真跑一遍完整 verify（真实执行）——预热每个 kernel、建好每个 lazy 缓冲、
           让 cudaMalloc 全部发生（**捕获期禁止任何分配**）
2) ROLLBACK: 用 ring_restore 把 DRY 推进过的环形/压缩器/clen 状态复原
           （`dspark_snapshot/rollback` 这套机制已存在，正是为这个场景写的）
3) CAPTURE: graph_capture_begin(); step_rows(); graph_capture_end("verify6");
           —— 记录不执行，所以回到 rollback 后的状态
4) REPLAY : 每步 graph_launch(verify6) + 图外一次 D2H
```

捕获触发点：**decode 第一步之后、且 `pos >= window` 之后**（`ring_append` 的 slot 由 device
counter 决定，本身安全；但 `ensure_idxs` 那类 n_win 缓存切换必须发生在捕获之外）。
每请求重置时 drop（与 `DevChain::reset()` 里 `step_graph.take()` 同样的两条理由：
host 分支被烘进图 + 地址可能被别的东西复用）。

### 3.3 收益

7220 节点：`0.411µs/node` dispatch 地板 vs `2.904µs/launch` 流式提交 ⇒
**上限 ~18ms**；扣除 kernel 真正执行的重叠后，保守 **8–14ms**（§9 待验证）。
注意：图还顺带吃掉 ~2.493µs/node 的 ramp-down 尾延迟。

---

## 4. draft 7.2ms 的分解

`draft_forward`（bs=5，3 个 MTP block，逐条计数）：

| 段 | kernels/block | ×3 | 估 ms | 依据 |
|---|---|---|---|---|
| **`draft_head` 的 head GEMV** | — | 5 | **1.49** | `draft_head` 注释自陈 "the head is read bs times: ~5 x 350 us. **That single term is most of the draft budget**"；实测全词表 298µs/次 |
| **`wo_a` 分组投影** | 8 组 × 5 行 = 40 | 120 | **~1.0** | n=1024,k=4096 fp8 = 4.19MB/发；120 发 + 120 次发射开销 |
| MoE 路由专家（gate_up+down） | 10 | 30 | 0.55 | 5 行 ×6 专家 ×2.46MB = 73.8MB/block；443GB/s ⇒ 0.17ms/block |
| 共享专家 | 25 | 75 | 0.30 | 逐行 5 发/行 |
| `rope_at` 的 **blocking H2D** | 12 | 36 | **0.25 +** | `upload_bytes_at` = 同步 `cudaMemcpy`；**每次强制流水线排空** |
| markov head（5 步串行） | 5 | 15 | 0.25 | 全词表扫描，且第 s 步依赖 `ids[s]` |
| 投影 gemm（main_proj/wq_a/wq_b/wkv/wo_b） | 5 | 15 | 0.15 | |
| hc 链 | 9 | 27 | 0.21 | 7.8µs/次 |
| 其余小核（quant/rmsnorm/rope/sparse/memcpy） | ~35 | ~105 | 0.35 | |
| **发射开销** | ~128 | **~400** | **~1.2** | 400 × 2.9µs |
| `drafts()` 末尾 D2H | 1 | 1 | — | 20 字节，但**阻塞整条链**（7.2ms 是端到端延迟，不是吞吐） |
| **合计** | | **~406** | **≈7.2** | |

**排序**：head (1.5) > wo_a (1.0) > 发射开销 (1.2，与上面并列) > 专家 (0.55) > H2D+排空 (0.25+)。

三大可砍点：
1. **head 多行化**（§2.4）：5 发 → 1 发，权重只读一遍 ⇒ 7.2 → **6.0ms**（−1.2）。
2. **分组输出投影融合**：`draft_attention` 自己写着 "A fused grouped-output kernel
   (or a group-major repack) is the obvious optimisation"。40 发 → 1 发 ⇒ **−0.9ms**。
3. **draft 位置计数器 device 化**（去掉 `rope_at` 的 36 次 blocking H2D，正是代码注释
   "⚠️ The upload is a blocking H2D per call; a device-side draft position counter would
   remove both it and the capture hazard"）⇒ **−0.3ms 且解锁 draft 图**。
4. **draft 图化**（形状固定 bs=5/3 blocks）⇒ 400 发 → 1 次 replay ⇒ **−1.1ms**。

四项合计 7.2 → **~2.6–3.0ms** ✓ 达标。

---

## 5. snapshot / rollback（2.7ms）与 commit

`dspark_snapshot` 逐层逐 slot：`ring_owners()`（默认 `DSV41_RING_OWNER=0` ⇒ 全部 40 层）×6 = 240 次 D2D，
加 4 个 compress source ×5 = 20 次；rollback 镜像一遍 ⇒ **~520 次小 D2D**。
按 ~5µs/次 = **2.6ms** ✓ 与实测 2.7ms 吻合。

⇒ 换成 device-side `ring_save` / `ring_restore`（grid = 层 × m）：
**520 发 → 2 发，2.7ms → ~0.05ms**。这是全表**性价比最高**的一项（§8 排 P0）。

commit（真 spec 模式下）= 把 accept 前缀的 KV/压缩器状态保留 + 推进 pos_ctr + 撤销尾部：
在 `ring_save/restore` 化之后，commit 就是「不还原」的那 1 次 ring_restore 跳过 —— 目标 ~1ms 内。

---

## 6. 优化后的步时预估

| 项 | 现状 | 优化后 | 主要手段 |
|---|---|---|---|
| **verify** | 38.9 | **4.5–6.5ms** | 图化（−8~14ms）+ 多行 GEMV（投影 10.5→2.0）+ head 多行+切分（2.2→0.1）+ MoE rows 批化 |
| **draft** | 7.2 | **2.6–3.0ms** | head 多行 + wo_a 融合 + device pos + 图化 |
| **snapshot/rollback** | 2.7 | **~0.05ms** | ring_save/restore kernel |
| **commit** | — | **~0.5–1.0ms** | 同上 |
| **步时** | ~55 | **8–10.5ms** | |

**tok/s @ accept=4（4 token/步）**：`4 / 步时`

| 步时 | tok/s |
|---|---|
| 5.5ms | **727** |
| 8ms | **500** |
| 10ms | **400** ✓ 达标线 |
| 12ms | 333 ✗ |

⇒ **400 tok/s 需要步时 ≤10ms，本路线给出 8–10.5ms 的落点，边际达标**；
若 MoE 的 `expert-cpasync-full`/tcgen05 不同时兑现，MoE 那 3.5GB 字节会把 verify 拖到 7–9ms，
总步时 ~12ms（333 tok/s）—— 这一条依赖必须显式记为风险（§8 R1）。

---

## 7. 优化建议（按收益排序）

1. **P0 · snapshot/rollback 核化**：`ring_save`/`ring_restore`（grid=层×m）替换 520 次 host-slot D2D。
   预期收益：**−2.6ms**／实施成本 **低**（~0.5 人日）／风险 **低**（纯搬运，
   数值域无涉；但 slot 算术必须与 `(pos+1+j)%win` 逐字一致 → 编码为 device 端算术）。
   同时也是 §3 图化的**前置条件**（host slot 计算不可捕获）。

2. **P1 · head 多行 + 词表切分**：一个 kernel 算 6 行 × 切分后 16160 行词汇 + 多行 argmax。
   预期：**−1.7ms**（1.79 → 0.05–0.30）／成本 **低-中**（head 已有多行先例 `nrows`；
   切分需跨 rank argmax 的**多行版**——`step_rows` 的 TODO#3 已点名 "a multi-row sliced argmax is the TODO"）／
   风险 **中**（跨 rank 交换缓冲 + 一次 barrier，历史上 `argmax_pub` 与 AR 暂存冲突导致死锁过一次）。

3. **P1 · 投影多行 GEMV（m=6）**：`gemm_fp8_gemv_kernel` 加 m 维（§2.2/2.3）。
   预期：**−7~8ms**（权重读 5.7GB→952MB + 2400→600 launch）／成本 **中**（新 kernel + WPR/K-split 路径）／
   风险 **中**（C6：任何重结合即破坏数值域 → 必须过逐位回归）。

4. **P2 · verify 图化**（§3）。预期 **−8~14ms**／成本 **中**（先做 P0/P1 清掉 host 依赖）／
   风险 **中**（捕获期分配、graph-drop-per-request、AR epoch 同步——mega-graph 的坑已记录）。

5. **P2 · MoE rows 进 grid**（§2.5）。预期 launch 720→120，**−1~2ms**（issue 效率）／成本 **低**／风险 **低**。

6. **P2 · sparse_attn 多行**：`sparse_attn` 的 launcher 已支持 `b*m`（grid=(b·m, h)，`kAttnMaxBM=8`），
   verify 却逐行发 ⇒ 直接 `b=1, m=6` 一发 + 1 次 merge。预期 **−0.5ms**／成本 **低**／风险 **低**。

7. **P3 · draft 三项**（head 多行 / wo_a 融合 / device-side draft pos counter）：
   预期 **−2.4ms**（7.2→4.8）／成本 **低-中**／风险 **低**（draft pos counter 是代码注释自陈的 TODO）。
8. **P3 · draft 图化**：−1.1ms／成本 中（依赖 #7 的 device pos）／风险 中。
9. **P3 · indexer/compressor 多行**：368 → ~80 launch，−1ms／成本 中／风险 中
   （`indexer_topk` 的 output stride 是运行期值 `min(topk,n_pos)`，多行需显式给 stride）。

### 不该做的（避免过度优化）
- **不动 `ring_append`/`window_idxs` 的「逐行交替」**：这是因果窗口修复的**正确性**要求
  （代码注释已记录 fused 版本因 `v > start_pos` 过滤失效而 "verify outputs garbage"）。
  多行化只能做「同一个 kernel 内按行顺序处理」，不能改成交替外的一次性 append。
- **不为省字节去动 expert 的 fp4**：3.5GB 是 6 个 token 的真实工作量，不是浪费；
  在 `expert-cpasync-full` 落地前，任何「减字节」改动都会把它推进 latency 墙
  （STATUS 的 gemv-l2-analysis 已用 swapAB 的实验证明过这条）。

---

## 8. 实施优先级总表

| 优先 | 项 | 收益 | 工作量 | 风险 | 依赖 |
|---|---|---|---|---|---|
| **P0** | snapshot/rollback 核化 | −2.6ms | 低 | 低 | — |
| **P1** | head 多行 + 切分 + 多行 argmax | −1.7ms | 低-中 | 中 | 跨 rank argmax 协议 |
| **P1** | 投影多行 GEMV (m=6) | −7~8ms | 中 | **中**（数值域回归） | — |
| **P2** | verify CUDA graph | −8~14ms | 中 | 中 | P0 + P1 |
| **P2** | MoE rows 进 grid | −1~2ms | 低 | 低 | — |
| **P2** | sparse_attn 多行（b=1,m=6） | −0.5ms | 低 | 低 | — |
| **P3** | draft head 多行 | −1.2ms | 低 | 低 | — |
| **P3** | draft wo_a 融合 | −0.9ms | 低-中 | 低 | — |
| **P3** | draft device pos counter | −0.3ms | 低 | 低 | — |
| **P3** | draft CUDA graph | −1.1ms | 中 | 中 | 上一项 |
| **P3** | indexer/compressor 多行 | −1.0ms | 中 | 中 | 显式 stride |

**R1（最大风险）**：MoE 3.5GB 专家字节。若 `expert-cpasync-full` / tcgen05 不兑现，
verify 落点从 4.5–6.5ms 退化到 7–9ms，400 tok/s 变成 333 tok/s。
这是**唯一一条不靠 launch 优化就能翻盘或翻车的项**。

**R2**：多行 GEMV 的逐位一致性（C6）。建议把 `dspark_parity` 那一套扩展成
「单行 vs 多行 GEMV 的 bit_diff」回归，作为 kernel 合入门禁。

---

## 9. 待验证（本机无 GPU，未经本次实测）

1. verify 的真实 nsys 分解（7220 launch 的 CPU/GPU 侧拆分）——本报告的分解是**推算**，
   虽与 38.9ms 总量吻合（表 1.2 合计 38.9），但单段值未直接测。
2. CUDA graph 对 verify 的实际收益（推算 8–14ms）——需 graph-on/off 两臂实测；
   注意 `graph_bench` 的 2.493µs/node 是在空核上测得，verify 的核平均 2.5µs，重叠率未知。
3. 多行 GEMV 在 m=6 时的寄存器占用（是否触到 `__launch_bounds__(256,4)` 的 64 寄存器墙）
   与是否真的把有效带宽从 373GB/s 推高（MLP 从 1 行变 6 行）。
4. MoE rows 进 grid 后的 issue-stall 改善（当前 80% issue 槽停等）。
5. draft `rope_at` blocking H2D 的真实代价（推算 36 次 × 数 µs + 排空，未单独测）。

---

## 10. 实施记录

### 2026-09-12 · P1 的 GEMV 一半落地（verify 的 head 多行）

**已实施**：`dsv41_head_gemv_bf16_mrows`（`kernels/cuda/dsv41_glue.cu`，紧跟
`gemv_bf16_kernel` 之后）+ `Device::head_gemv_bf16_mrows`（device.rs）+ `step_rows`
head 段改单发（chain_dev.rs）。**未实施**（本增量故意不做）：词表切分与多行
argmax —— 二者同属「跨 rank argmax 协议」，argmax 仍是逐行 6 发（~30µs，可接受）。

**纠正本文档 §1.3 与任务书的一处口径**：verify 的 head **确实未切分**——`head.weight`
是 `Shard::Replicated [129280, 5120]` bf16（weights.rs:90），`step_rows` 用
`cfg.vocab_size` + `head.ptr()` 寻址，即**每次 1324 MB**、6 行 6 次。任务书里的
「16160 行 / 165MB」是 `DSV41_HEAD_SLICE` 的**切分后**单 rank 尺寸，那条路径只在
`step_dev` 里生效（`step_rows` 没有 sliced 分支）。所以多行前的单点成本是
6 × 298µs = 1.79ms（与 §1.2 表一致），不是 990MB 的字节账。

**为什么不复用 `gemv_bf16_v2`（nrows>1）**：读了它的签名与 kernel 体
（ferrite_kernels.cu:2880）。v2 的 nrows 是**并行维度**不是**复用维度**——
`rowg = blockIdx.x*rpb + warp/WPR; token = rowg/out_f; row = rowg - token*out_f`，
权重 `w + row*in_f` 仍被每个 token 重读一遍。它省 launch 数（tp.rs 的 `small_n_rows`
用的就是这个），**不省字节**。头要的是 weight-stationary，所以新写了一个 kernel，
并把 c 步长**逐字保持 v1 的 `c = lane, lane+32, ...`**（向量化 K 会重排 lane 内
累加序 → 直接破坏 C1，故本 kernel 的收益只能来自权重复用，不能来自 load 宽度）。

**数值域**：kernel 头注释给出 C1-C5 的逐条论证；C6 用两处 codegen 钉死——
累加用 `__fmaf_rn`（v1 的 `acc += (float)wr[c]*x[c]` 在 fmad=on 下就是一条 FFMA，
同 `gemv_f32_v2` 头注释的论证），shuffle 归约树保持与 v1 逐字相同的源码形式。
`--use_fast_math` 默认 ON，其 build.sh 注释记录过「plain operator + 多路展开
≈1 ULP/层漂移」，m 路独立累加器正是最容易被重结合的形状，故不留给编译器。
`m` 是模板参数（分发 1..=8），m>8 或缺符号都回退到逐行循环（stale .so 安全）。

**待办（门禁 / 验收）**：
1. **R2 的逐位回归还没写**：建议在 `kernels/cuda/tests_dsv41_glue.cu` 加一个
   `head_gemv_mrows vs m 次 dsv41_gemv_bf16` 的 bit_diff case（本机无 nvcc，
   写不了也编不了，留给有 CUDA 工具链的节点）。在它绿之前，这次改动只能靠
   下面这条端到端回归兜底。
2. **端到端**：`dspark_parity`（「verify parity — the iron rule」，
   `cargo test -p ferrite-models --lib dspark_parity -- --ignored --nocapture`）。
   多行 kernel 若破坏数值域，`step_rows(truth)[r]` 会偏离但位置 argmax。
3. **实测（§9 待验证 #3）**：m=6 的寄存器占用是否触到 64-寄存器墙，以及单发
   一行是否真到 ~0.30ms（若因 6 份 x 载入而变成 load-bound，收益会低于 5/6，
   但不会低于 2 倍）。draft 侧（P3）未动，`draft_head` 仍是逐行。

---

*户部 · 基于 2026-09-12 仓库状态（HEAD 含 gpqa/v13 系列分支）*
