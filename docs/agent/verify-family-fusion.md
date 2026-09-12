# verify 路径的「族级融合」架构设计

> 中书省 · 2026-09-12 · **只读分析 + 本文档（唯一产出）**。未执行任何 GPU 命令、未改动任何源码。
> 基线：`DSV41_TIMING` **verify = 37.31 ms（m=5，6224 launches，14.09 GB）**，HEAD `49c6fc3`。
> 输入：`verify-5rows-marginal.md`（判定）、`verify-ms-breakdown.md`（逐族账）、`verify-calc-floor.md`、
> `dspark-verify-perf-plan.md`、`routed-expert-residual.md`、`final-400-battle.md`、`expert-tcgen05-plan.md`。
> 代码基线：`crates/ferrite-models/src/dsv41/chain_dev.rs`、`kernels/cuda/dsv41_kernels.cu`、
> `kernels/cuda/dsv41_glue.cu`、`kernels/cuda/dsv41_experts_mxf4.cu`、`kernels/cuda/ferrite_kernels.cu`。
> **本文档只做规划，不含实施**；所有新增 kernel 一律「先 parity、后 A/B、默认 OFF」。

---

## 0. 判决（先读这六条）

1. **mrows 的失败已被解释清楚，不是 dispatch 问题**。`gemm_fp8_mrows_kernel`（`dsv41_kernels.cu:4958`）
   折叠的是**权重解码**，激活解码 + FMA 仍 ∝ M。指令模型 `(2+3M)/(5M)`：M=5 → **0.68×**，
   M→∞ → 0.60×。这是**物理下限**，永远拿不到 1/M = 0.2×。实测（a32 修复后 38.34 = 36.10 + e4m3 代价）
   证明折叠确实上线了、贡献 ≈ 0 —— 因为这一族根本**不是指令吞吐 bound，也不是带宽 bound**，
   而是 **per-kernel 固定成本 + 低占用（每 SM 3–4 warps）** bound。

2. **因此「族级融合」的目标函数不是「少读字节」，而是「少发 kernel、多发 warp」**。
   同一个 mrows 折叠放进**一个 block 内**、把 M 行激活**放进同一个 grid 维度**、把**跨阶段串成一条
   kernel 内的流水线**——这三件事同时做，才叫族级融合。只做第一件（mrows）在 verify-5rows-marginal
   里已被证伪。

3. **8 个 5× 族里，有 6 个的融合内核的「骨架已经存在，只是停在 M=1」**：
   `gemm_fp8_sh_pair_kernel`（shared expert 三段一体）、`gemv_bf16_fp8x2_kernel`（gate + shared w1|w3）、
   `gemv_bf16_nt_kernel<NT,WPR>`（gate mrows）、`sparse_attn` 的 `b*m` grid（attention）、
   `indexer_topk` 的 `(m,b)` grid（indexer）、`head_gemv_bf16_mrows`（head）。**本设计的 80% 是
   「把既有骨架从 M=1 抬到 M=5」，不是发明新数学。**

4. **换核（tcgen05）与融合在同一处天然汇合**：`kind::mxf4` 的最小合法 N = 8，而 verify 的 m = 5
   **恰好塞进一个 MMA 的 N 维**（`dsv41_experts_mxf4.cu:3813`）。即「Routed experts 的 5 行」
   = 「一个 swapAB tile 的 8 列里的 5 列」。这是唯一一条能把 routed 8.30ms 打到 1.5–2.5ms 的路径。

5. **收益的**上限**受制于一个未钉死的量**：per-kernel 固定成本。verify-ms-breakdown 的分解
   （50% launch + 49% 残差）被图化 A/B 否掉了「submit 半」，但**「GPU 侧 ramp/drain 半」仍在**：
   6224 × 3.3µs ≈ 20.5ms。融合把 launch 数砍到 ~1500 时，**这 20.5ms 的边界成本会掉到 ~5ms**
   ——这是本设计最大的、也最可预测的一笔。**它不依赖任何核效率假设。**

6. **诚实结论**：族级融合 + tcgen05 的落点，**保守 18–22ms、目标 8–11ms**（细节见 §4）。
   到 5.5ms 需要融合核内部也接近带宽/占用地板（cp.async 流水 + 满 wave），那是**第二轮**
   的事，不在本设计的承诺内。

---

## 1. 现状分析：8 个 5× 族的 kernel 结构

### 1.1 逐族账（m=5，TP8，40 层；来源 `verify-ms-breakdown.md §1` + 代码复核）

| # | 族 | launch/步 | 实测 ms | 达成 BW | 每发 µs | 5× 的根因（代码） |
|---|---|---:|---:|---:|---:|---|
| 1 | **shared expert** | 1000 | 10.40 | 85 GB/s | 10.4 | `moe_rows` 逐行 5 发：`quant1`+`gemm_fp8_mx2`+`swiglu_limit`+`quant1`+`gemm_fp8_mx_add`（`chain_dev.rs:9122-9205`）|
| 2 | **routed experts** | 400 | 8.30 | 378 GB/s | 20.8 | 已 rows 进 grid.z（`expert_gemv_fp4_batched_kernel`，`dsv41_experts_mxf4.cu:1217`），但字节 ×5 不可压 + SIMT LUT gather |
| 3 | **MoE gate** | 200 | 3.44 | 229 GB/s | 17.2 | `moe_rows` 逐行 `gemv_bf16`（`chain_dev.rs:8895`）；`ROW_FOLD_GATE` 默认 OFF |
| 4 | **indexer** | 230 | 2.50 | 174 GB/s | 10.9 | `indexer_rows_one` 逐行：`lin`(2 发)+`apply_rope`(1)+`lin_bf16`(1)+`indexer_topk`(1)（`chain_dev.rs:8392`）|
| 5 | **attention KV/sparse** | 880 | 2.80 | 94 GB/s | 3.2 | `attention_rows` 行循环内 `ring_append`+`window_idxs`+`sparse_attn_orope`，各 ×m（`chain_dev.rs:7938-8100`）|
| 6 | **compressor** | 80 | 0.55 | 667 GB/s | 6.9 | `compress_proj_rows` 逐行 `lin_f32`×2（`:8523`）+ `compress_row` 逐行 pool+commit（`:8602`）|
| 7 | **head** | 10 | 1.12 | 739 GB/s | 112 | 逐行 `gemv_bf16`（`:4950-4960`）；`VERIFY_HEAD_FOLD` 默认 OFF（K 序不是 v1）|
| 8 | **norm/cast/quant** | 680 | (0.20) | — | — | 逐行 `apply_rope`(q)、`quant_fp8`(o)、`quant_fp8`(wo)——**launch 成本被折进它们服务的族** |

**合计 8 族 = 31.35 ms / 3480 launches**（routed 8.30 含在内）。剩余 8.68ms 是投影 3.70 / hc 2.96 /
engram 0.42 / AR 1.40 / 激活 0.20 —— 这 5 项**已折到 1×**，不在本设计范围（但 hc 的 53 GB/s 是另一个坑）。

### 1.2 三条被代码证实的关键事实

**(F1) attention 的 `b*m` 支持是「半成品」，缺的是 per-row clen。**
`dsv41_sparse_attn` launcher 已经是 `dim3 grid(b * m, h)`（`dsv41_kernels.cu:7154`），且
`kAttnMaxBM = 8`（`:1449`）覆盖 m=5。但四个 kernel 体都读**单个标量** `*clen`：
`sparse_attn_kernel:948`、`sparse_attn_warp_kernel:1037`、`sparse_attn_pf_kernel:1134`、
`sparse_attn_split_kernel:1491`、`sparse_attn_orope_kernel:2040`。
verify 的因果交错（`chain_dev.rs:7938` 的注释，audit defect #1/#2）使 **row r 的 clen 各不相同**
（row r 自己 commit 的组数）。⇒ **b·m 单发的前置条件 = 一个 `clen[m]` 数组 + 一个显式 `idx_stride`。**
kernel 里 `irow = idxs + (bb*m+mm)*topk` 用的是 `topk = win + min(clen, index_topk)`，而 verify 的
`idxs_r` 行距是固定的 `ist = win + index_topk`（`chain_dev.rs:7950`）——两个 stride 不一致，必须显式传。

**(F2) indexer 的两个核已经是 m 行原生的，调用点却是 m=1。**
`indexer_topk_kernel` 的 grid 是 `dim3 grid((unsigned)m, (unsigned)b)`（`dsv41_kernels.cu:7375`），
且 `cl = lens[mm]` 是**逐行读 clen**（`:2874`）；`indexer_score_kernel` 同理（`:2657`）。
`indexer_rows_one`（`chain_dev.rs:8392`）却逐行调用、每次 `b=1, m=1`，还把 `clen.ptr + key_owner`
当标量传进去。**这一族的融合几乎是纯接线工作。**

**(F3) shared expert 的三段一体 kernel 已经存在，且是 dead code。**
`gemm_fp8_sh_pair_kernel`（`dsv41_kernels.cu:6218`）+ launcher `dsv41_gemm_fp8_sh_pair`（`:6443`）
做的是：phase 1（w1|w3 GEMV + swiglu + fp8 emit）→ **grid barrier** → phase 2（w2 GEMV），
全部 bit-identical。`sh_pair()` gate（`chain_dev.rs:1073`）默认 OFF，**且 grep 全仓无调用点**。
它被写出来却从未接线，且只支持 M=1（单激活行）。**本设计的第一块砖就在这里。**

### 1.3 为什么这 8 族「折了也不快」——三个机理的精确表述

| 机理 | 内容 | 对本设计的要求 |
|---|---|---|
| **M-a 每 kernel 固定成本** | 每发 kernel 的 GPU 侧 ramp（grid 起、smem 分配、warp 起）+ drain（尾 warp 退出）+ 边界排空。审计口径 3.3µs/发（`dspark-perf-400-plan.md §六`）。6224 发 ≈ **20.5ms**。 | **所有融合核必须「一层一发」**，把 5 个行循环砍成 1 发。这是**唯一不依赖核效率假设的收益**。 |
| **M-b 低占用** | n=288（shared 每卡）/ n=384（gate）/ n=512（wkv）这些 n 下，`g_gemv_warps=4` → blocks = 72–96 → **每 SM 仅 3–4 warps**（`dsv41_kernels.cu:3553` 的 "latency-bound family" 注释自陈）。SM 上没有任何东西可以互相掩盖 LDS 延迟。 | 融合核要把 **M 行进 grid 或进 block**，把在飞 warp 数 ×M。`mrows` 只加寄存器累加器，**不加 warp** —— 这是它与族级融合的分水岭。 |
| **M-c 跨阶段 global 往返** | 链上每一段把中间量写 global 再读回（`sh_act_r`、`xq/xsc`、`logits_r`）。融合后用 smem + barrier 交接。 | **阶段融合**（F2 方案）把 5 段变成 1 段；中间量留在 smem/寄存器。 |

---

## 2. 方案：三条融合原语 + 一个总架构

### 2.1 原语

**P1 · 行进 block（row-in-block）** — 5 行激活放进同一个 block，权重 tile 只 decode 一次。
用于 instruction-bound 的 GEMV 族（shared expert、gate）。收益 = 权重解码 ÷M + 一次 staging 服务 M 行。

**P2 · 行进 grid（row-in-grid）** — 5 行放进 grid 的一个维度，每行一个 block，但**一次 launch**。
用于 n 小、本来就 block 少、且行之间无数据依赖的族（attention 的 `b*m`、indexer、compressor）。
收益 = 边界成本 ÷M + block 数 ×M（占用 ×M）。

**P3 · 阶段融合（stage fusion）** — 用 kernel 内 grid barrier 把「写 global → 下一发读回 global」
换成「写 smem → barrier → 读 smem」。仅在**跨 block 依赖不可消**时使用（shared expert 的 w2 需要
全部 inter 列）。

**P4 · 换核（kernel swap）** — 对 routed experts 用 tcgen05 swapAB 换掉 SIMT fp4 LUT gather。

### 2.2 总架构：每层的 launch 拓扑（目标）

```
attention block（每层）：
  [A1] hc_mixes + hc_collapse + rmsnorm                （既有，rows=m）
  [A2] 投影 fused：quant_rows + proj_mrows(wq_a|wkv)    （既有，M=1→M 行）
  [A3] q/kv norm + rope（mrows）                        （既有骨架）
  [A4] ★ attn_fused_rows：ring_append+window_idxs+compress+indexer+sparse_attn_orope
       → 拆成 3 发：ringwin_rows(1) / compress_pool_commit_rows(1) / attn_orope_bm(1,x2 split+merge)
  [A5] wo_a_grouped + wo_b mrows + o-quant（1 发）       （既有 + src_stride）
MoE block（每层）：
  [M1] ★ gate_sh_gateup_fused：gate GEMV + route_topk-epilogue + sh w1|w3 + swiglu + fp8
  [M2] routed experts（tcgen05 e4m3，N=8 容 m=5 行）
  [M3] ★ sh_exp_down_fused：w2 GEMV + add epilogue
  [M4] 两个 AR
head：
  [H1] ★ head_gemv_bf16_mrows_v1（v1 K 序）+ argmax_sliced_rows（已 m 行）
```

每层 launch 数从 ~170 降到 **~25–30**；每步从 6224 降到 **~1200–1400**。

---

## 3. 逐族设计（族 × 融合方案 × kernel 落点）

### 3.1 shared expert —「三段一核」（10.40ms → 目标 2.5–5.5ms）

**现状**（`chain_dev.rs:9122-9205`）：逐行 5 发 = `quant1` → `gemm_fp8_mx2`(w1|w3) →
`swiglu_limit` → `quant1` → `gemm_fp8_mx_add`(w2)。25 发/层 × 40 = 1000 发/步。

**设计**：把 `gemm_fp8_sh_pair_kernel`（`dsv41_kernels.cu:6218`）从 M=1 抬到 `template<int M>`，
即新符号 `dsv41_sh_exp_fused<M>`：

```
grid   = (max(ceil(2*sh_il/warps), ceil(dim/warps)), 1)，block = 1024（32 warps），
         grid ≤ co_res（沿用 sh_pair 的 residency cap，否则 barrier 死锁）
phase A：warp w 拥有 inter 行 i = blockIdx.x*nwarps + w（w1/w3 的同一行）
         ① 一次性 stage xq[0..M-1][dim]（M 行激活，fp8）+ 各自 scale 到 smem
         ② w1[i,:]、w3[i,:] 各 decode 一次
         ③ for r in 0..M：acc_g[r] / acc_u[r] 累加（M 个独立归一化链，C1–C6 逐位成立）
         ④ for r in 0..M：swiglu_limit 的 clamp+silu，fp8 emit 到 aq[r][i] / aqsc[r][i/32]
grid barrier（复用 g_sh_arrive/g_sh_sense，`dsv41_kernels.cu:6214-6215`）
phase B：warp w 拥有输出行 j；for r in 0..M：acc[r] = Σ_i w2[j,i]*aq[r][i]；
         ① 写 out[r][j]（+ add epilogue 折进 lane 0，等价 A5）
```

**关键不变量**：
- 每个 `(r, ·)` 的 K 序、归约树、decode 表达式与 `gemm_fp8_mx` 的 M=1 程序**逐字相同** ⇒ 逐位等价。
- 中间量 `aq/aqsc` 走 smem（不再落 global），M 行共用一次 activation staging。
- grid ≤ co_res 是**硬约束**：barrier 只对同时驻留的 block 成立。launcher 必须沿用
  `dsv41_gemm_fp8_sh_pair` 的 `co_res_cached` 逻辑（`dsv41_kernels.cu:6490-6508`）。

**launch 数**：25/层 → **1/层**（1000 → 40/步）。
**预期 ms**：分两种情景（见 §4）：保守 5.5ms，目标 2.5ms。
**Rust 落点**：`shared_expert_mrows`（`:9274`）改成先试 `sh_exp_fused`（新 `supports_sh_exp_fused()` +
`sh_exp_fused()` gate），无论成败都保留现有 mrows / per-row 两级回退。
**工作量**：3–4 人日（新 kernel + `tests_dsv41_glue.cu` 扩 M 行的 `sh_pair` parity case）。

---

### 3.2 MoE gate —「gate + 共享专家 gate/up 合一」（3.44ms → 0.8–1.5ms）

**现状**（`chain_dev.rs:8895`）：逐行 `gemv_bf16`（n=384, k=5120 bf16）。5 发/层 × 40 = 200 发/步。
**已存在但低效**：`ROW_FOLD_GATE`（`:969`）→ `ferrite_gemv_bf16_v2_mrows` → `gemv_bf16_nt_kernel<NT,WPR>`
（`ferrite_kernels.cu:3419`）。**为什么只到 0.68–0.73×**：每 k-step（uint4 = 8 个 bf16）里
`1 weight load + 4 cvt + NT×(2 x-load + 8 FMA)`，NT=1 是 15、5 行合计 75，融合后 `5 + 50 = 55`，
**55/75 = 0.73** —— 与 5× 的目标差得远。**因为它把 M 行放进寄存器累加器，没放进 grid：warp 数不变、
在飞 warp 数不变**（M-b 机理没被攻击）。

**设计**：复用一个**已经存在的三合一 kernel** —— `gemv_bf16_fp8x2_kernel`（`dsv41_kernels.cu:6532`，
launcher `dsv41_gemm_bf16_fp8x2:6691`），它在 `moe()`（单行路径）里同时算
**bf16 gate + 两个 fp8 family（w1/w3）**。把它抬到 M 行：

```
dsv41_gate_sh_gateup_fused<M>:
  grid = (ceil((nb + 2*nf)/rpb), 1)，block 256
  每个 warp 拥有一个「family 行」（gate / w1 / w3 的统一行空间）
  for t in 0..M：独立累加链（C1–C6）
  gate 部分额外带 route_topk 的 fuse epilogue（last-block 选举，照抄
  ferrite_gemv_bf16_v2_route:3369 的 gv2_route_epilogue 模式，扩到 M 行）
```

- 输入：`xn_r`（m 行，同一份）；这是 gate 与 shared expert **唯一共享的输入**——这就是融合的合法性来源。
- 输出：`scores_r`（gate）、`aq/aqsc`（shared expert 的 swiglu 后 fp8）。
- launch 数：5（gate）+ 10（w1|w3） → **1/层**（200 + 400 → 40/步）。

**回退**：老 `.so` 无符号 / gate 默认 OFF → 回到 `ROW_FOLD_GATE` → 回到逐行。
**工作量**：2 人日。

---

### 3.3 indexer —「纯接线级融合」（2.50ms → 0.6–1.0ms）

**现状**（`indexer_rows_one`，`chain_dev.rs:8392`）逐行 5 发：`lin`(=`quant1`+`gemm_fp8_mx`) ×1、
`apply_rope` ×1、`lin_bf16`(wp) ×1、`indexer_topk` ×1、`publish_index_key` ×1。
8 层 × 5 行 × 5 ≈ 230 发/步。

**设计**（**不需要新 kernel**——全部是既有多行入口的接线）：

| 步骤 | 现状 | 融合后 | 依据 |
|---|---|---|---|
| idx_wq_b | 逐行 `lin`（2 发/行） | `quant_rows(qr_r, m, ql)` + `proj_mrows(idx_wq_b)`（2 发/层） | `chain_dev.rs:3183` / `:3237` |
| rope | 逐行 `apply_rope`（1 发/行） | `apply_rope_mrows`（1 发/层） | `device.rs`（`dsv41_apply_rope_mrows`）|
| idx_weights | 逐行 `lin_bf16`（1 发/行） | `gemv_bf16_nt(nrows=m)`（1 发/层） | `ferrite_kernels.cu:3575` |
| topk | 逐行 `indexer_topk(b=1,m=1)` | `indexer_topk(b=1, m=5)`（1 发/层） | grid 已是 `(m,b)`，`lens[mm]` 已逐行 —— `dsv41_kernels.cu:2874/7375` |
| publish | 逐行 | 保留逐行（`*clen - 1` 写 key，顺序敏感）| `chain_dev.rs:8314` |

**前置**：把 `indexer_rows_one` 里 `clen.ptr + key_owner`（标量）换成一个 **`clen_rows[m]` 快照**。
快照来源：`compress_row` 已经把 host mirror `layers[l].compress_len` 逐行推进（`:8700`），
且 verify 的 `pos_rows` 逐行可用——需要一个极小的 `snapshot_clen_rows` kernel（或由 compress 的
epilogue 顺带写）。

**launch 数**：25/层 → **5/层**（230 → 40/步）。
**预期 ms**：2.50 → 0.6–1.0（主要是边界 + 权重只读一次）。
**工作量**：1.5 人日。

---

### 3.4 attention —「b·m 单发 + 逐行 clen + ring/window 合核」（2.80ms → 1.0–1.5ms）

**现状**（`chain_dev.rs:7938-8100`）：行循环内 `ring_append`(1) + `window_idxs`(1) +
`sparse_attn_orope`(1，split+merge = 2) → 每行 ~4 发。880 发/步，3.2µs/发（**纯边界 bound**）。

**设计（三件事）**：

**(a) `clen` 数组化 + `idx_stride` 参数化。**
新增 launcher 变体 `dsv41_sparse_attn_bm` / `dsv41_sparse_attn_orope_bm`：

```c
// 新增两个尾参（追加在最后，老调用点不变）
const int* clen_rows,   // [m]，nullptr => 退化为旧的 *clen 行为（逐位不变）
int         idx_stride  // idxs 的行距；verify 传 win + index_topk（= ist）
```
kernel 体内：`const int cl = clen_rows ? clen_rows[mm] : *clen;`，
`const int32_t* irow = idxs + (size_t)(bb*m+mm) * (idx_stride ? idx_stride : topk);`
**逐行 r 的 `n`/`topk` 全部由 `cl` 派生** ⇒ 与「逐行单发」逐位一致（每行的 idxs 是同一份）。

覆盖 5 个 kernel 体：`:948`、`:1037`、`:1134`、`:1491`、`:2040`；以及 merge（`:1747`，它不读 clen，无需改）。

**(b) ring_append + window_idxs → 一发（且必须保持 r 升序）。**
现有的 `dsv41_verify_ring_win`（`dsv41_glue.cu:1765`）**不能直接用**——它先 append 整块再推 idxs，
在 `base+r >= window` 时用 SLOT 当 position 过滤，正是 audit defect #2。**正确设计**：
`dsv41_ring_win_rows` 用 **grid = (max(win,hd)/128)，block 内 `for r in 0..m` 升序循环**，
每次先 `ring[((pos+r)%win)*hd + c] = kv[r*hd+c]` 再推 `idxs[r*ist + …]`。
单块内 r 升序 = 与逐行完全相同的顺序；block 间按列切分、互不干扰（`:882` 的
`ring_win_fused_kernel` 已是这个形状，只需加 M 行内循环）。**10 发/层 → 1 发/层。**

**(c) 因果交错的其余部分保持「行内」**：`compress_row`（pool+commit）与 `publish` 必须**按 r 升序**
（状态累积），所以它们在 kernel 内部用 r 循环（见 §3.5），而不是 grid 维度。

**launch 数**：880 → **~160/步**（ringwin 1 + compress 1 + attn_bm 2 + indexer 5 + 其余）。
**预期 ms**：2.80 → 1.0–1.5（主要是边界：880 × 3.2µs ≈ 2.8ms → 160 × ~6µs ≈ 1.0ms）。
**风险**：`b*m` 路径要求 `b*m ≤ kAttnMaxBM = 8`——m=5 安全，m=6（spec 步）**也安全**，但 m 再涨即 decline。
**工作量**：3 人日（5 个 kernel 体 + merge 路径的回归）。

---

### 3.5 compressor —「rows=m 的 pool/commit + f32 mrows 投影」（0.55ms → 0.25–0.35ms）

**现状**：`compress_proj_rows`（`:8523`）逐行 `lin_f32`×2（wkv/wgate，f32 权重 10.49MB/个）；
`compress_row`（`:8602`）逐行 `compressor_pool_on` + `compress_commit_on`。4 源层 × 5 行 × 4 ≈ 80 发。

**设计**：
- **投影**：新增 `dsv41_gemv_f32_mrows<M>`（把 `dsv41_gemv_f32` 的 M=1 体加一维行累加器，
  C1–C6 逐位论证同 `gemm_fp8_mrows`）。或直接用 `gemm_fp8_mx_f32`（`device.rs` 有
  `supports_gemm_fp8_f32`）配 `quant_rows`——但那会引入 fp8 精度变化，**不推荐**。
- **pool+commit**：现有 `dsv41_compressor_fused`（`dsv41_kernels.cu:7460`）**显式拒绝**
  `b != 1 || seqlen != 1`。新增 `seqlen = m` 臂：块内 `for r in 0..m` 升序（state_kv/state_score
  的累积按位置序，天然串行），`out_rows`/`clen` 写回后由 host mirror 比对。
- **前置**：`compress_proj_rows` 的 `spec_capture` D2D 快照（`:8560`）要按 `[layer][m][hd]` 一次拷。

**launch 数**：80 → **16/步**（4 源层 × (1 投影 + 1 pool/commit) × …）。
**工作量**：1.5 人日。

---

### 3.6 head —「v1 序 mrows」（1.12ms → 0.25–0.40ms）

**现状**（`:4950-4960`）：逐行 `gemv_bf16(head_ptr, xn_r + r*dim, logits_r + r*stride, seg, dim)`，
5 发；随后 `argmax_sliced_rows`（已支持 m 行，`:4979`）。

**关键**：head 是 **8 族里唯一真正带宽受限的**（未切分实测 5.9 TB/s；切分后 827.5MB / 1.12ms
= 739 GB/s）。**5 行各读一遍 165.5MB/卡 的权重 = 827.5MB，而它本可以只读 165.5MB。**
⇒ 这是**唯一一个「读一次权重」直接等于 5× 的族**。

**已有的 `head_gemv_bf16_mrows`（`dsv41_glue.cu:481`）为什么不能用**：它的 K 序是 **v2**
（`gemv_bf16_nt_kernel` 的 WPR==1 体转写），而 eager 路径的 head 走的是 **v1**
（`gemv_bf16_kernel`，`ferrite_kernels.cu:3029`→`dsv41_kernels.cu:2950` 区段）。v1/v2 的
求和顺序不同（v1 是单 warp shuffle 树，v2 是 K-slice partial + smem fold）⇒ 不是逐位等价。

**设计**：新增 `dsv41_head_gemv_bf16_mrows_v1`——把 `gemv_bf16_kernel` 的体**逐字转写**成
`template<int NT>` 多累加器版（`for c = lane; c < k; c += 32` 的标量 bf16 walk 保持不变，
每行一条独立链 + 各自的 shuffle 树）。⇒ head 5 发 → 1 发，权重 827.5 → 165.5MB。
**预期**：1.12 → 0.25–0.40ms（若带宽保持 739 GB/s，165.5MB/739GB/s = 0.22ms）。
**工作量**：1.5 人日（含 `tests_dsv41_head_mrows.cu` 的 v1-parity case）。

---

### 3.7 norm + quant —「行化收尾」（680 发 → ~250 发）

**现状**：这一族的 680 发里，`rmsnorm` 其实**已经是 rows=m**（`chain_dev.rs:7758/7843/7531/7577`），
真正的逐行项是：
- `apply_rope(q)` 逐行（`:7791-7800`），`DSV41_ROW_FOLD_ROPE` / `VERIFY_ROPE_MROWS` 默认 OFF；
- `apply_rope(o)` 逐行（`:8122-8131`），但 `VERIFY_OROPE` 默认 ON 已把它折进 `sparse_attn_orope`；
- `quant_fp8` 逐行（o 的 `:8180`、wo_b 的 `:8255`）——**不是不想折，是不能**：
  `quant_rows`/`quant_kernel` 从 `cols` 推源行距（`src = x + r*cols`，`:126-133`），
  而 `o_r` 的行距是 `nh*hd`、`wo_r` 是 `ol_total`（TP8 下 8× 差）。

**设计**：给 `dsv41_quant_fp8`（`dsv41_kernels.cu:3333`）加一个显式 `src_stride` 尾参
（nullptr/0 → 旧的 `cols` 行为，逐位不变）。kernel 体改一行：
`const float* src = x + (size_t)r * (src_stride ? src_stride : cols) + b*block;`。
⇒ o-quant / wo-quant 各 5 发 → 1 发。

**launch 数**：680 → **~250/步**（q rope 200→40，o/wo quant 400→80，余下已是 m 行）。
**ms**：账面上只有 0.20ms（被折进别族），但**它的边界成本（~1.4ms）会从被服务的族里消失**。
**工作量**：1 人日（两处加参 + 三个调用点 + A/B）。

---

### 3.8 routed experts —「换核 tcgen05 e4m3，N=8 容 m=5 行」（8.30ms → 1.5–2.5ms）

**现状**：400 发/步，3133MB（5× 不可压），378 GB/s。已被 `expert_gemv_fp4_batched_kernel`
（`dsv41_experts_mxf4.cu:1217`）折成 rows 进 grid.z + slots 进 grid.y。

**换核设计**（骨架已存在，`expert-tcgen05-plan.md` 已有完整计划，此处只做架构定位）：

- **两臂**：`kind::mxf4`（e2m1×e2m1，`dsv41_experts_mxf4.cu:4008`，
  `build.sh` **默认编入**，runtime gate `DSV41_EXPERT_TCGEN05_MXF4` 默认 OFF）；
  `kind::mxf8f6f4`（e4m3 激活，`:3381`，opt-in 宏）。
- **e4m3 变体必须存在的原因（硬约束）**：`kind::mxf4` 的 `b_format` 只接受 E2M1
  （`dspark-correctness-chain.md:419` 的判词）⇒ **tcgen05 硬件路径与 e4m3 激活互斥**。
  而今天的正确性达标靠 e4m3 单趟（`expert_act_e4m3`，`chain_dev.rs:747`）。
  ⇒ **换核必须先做 e4m3 臂的 GPU parity**，否则换核 = 回退到 opa 的数值域。
  骨架里 e4m3 臂的 extern "C" 入口已存在：`dsv41_expert_tcgen05_gate_up_e4m3`
  （`dsv41_experts_mxf4.cu:5120`，`DSV41_TCGEN05_GATEUP_E4M3_SKELETON` 默认编入）。
- **架构契合点（本设计的新发现）**：`kind::mxf4` 的最小合法 N = 8
  （`dsv41_experts_mxf4.cu:3813` "minimum legal N"），而 verify 的 **m = 5 ≤ 8**。
  当前 kernel 是「N=8，只用 column 0」（`:3118` "only column 0 is kept"）。
  ⇒ **把 M 行激活放进 N 的第 0..M-1 列，一个 MMA 就吃下整个 verify 块**。
  这同时完成「融合」（5 行一发）和「换核」（SIMT→tensor core），是**唯一的 5×→1× 路径**。
  代价：N 侧从 1 列变 5 列 ⇒ 激活侧 smem 与 B 描述符要按 N=8 布局
  （`dsv41_experts_mxf4.cu:3352` 的 `unit16(n,kb) = (n & 7) + 8*kb` 已是 8 行布局）。
- **down 方向**：`expert_gemv_fp4_down_reduce_kernel`（`:2055`）目前只有 SIMT。
  第一轮可只换 gate/up（~60% 的时间），down 保留 SIMT；第二轮补 down 的 tcgen05。

**预期**：8.30 → 1.5–2.5ms（plan 文档口径）。
**工作量**：4–5 人日（e4m3 臂的 GPU parity + N=5 行布局 + Rust dispatch + 四段文本终门）。

---

## 4. 性能模型与预期 ms

### 4.1 模型（**基于指令数/占用率，不是带宽**）

```
t_family = N_kernel × c_k  +  W_family
    c_k    = per-kernel GPU 侧固定成本（grid ramp + drain + 边界排空）
           ≈ 3.3 µs（dspark-perf-400-plan §六 的「最小执行」口径）
    W_family = 族的内禀工作量：
               (a) 指令数模型：mrows 折叠给 (2+3M)/(5M) → M=5 是 0.68×（下限，不可突破）
               (b) 占用模型：warp 数 ×M ⇒ 有效 IPC 上升，W 可低于指令模型
               (c) staging 共享：一次 activation staging 服务 M 行 ⇒ W 里固定的那段 ÷M
```
**注意**：图化 A/B 已证明 **submit 半是重叠的**（只省 1.21ms）⇒ 上式里的 `c_k` **不是** 2.9µs
submit，而是 GPU 侧那一段（~3.3µs）。**融合的收益主要来自把 N_kernel 除以 5，而不是来自 submit。**

### 4.2 逐族预期（**两情景**）

| 族 | N 现状 | ms 现状 | W 现状(≈ms−N×3.3µs) | N 融合后 | W 融合后 | **保守 ms** | **目标 ms** |
|---|---:|---:|---:|---:|---|---:|---:|
| shared expert | 1000 | 10.40 | 7.10 | 40 | 0.75×W（P1+P3） | **5.5** | **2.5** |
| MoE gate | 200 | 3.44 | 2.78 | 40 | 0.73×W（P1） | **2.1** | **1.2** |
| indexer | 230 | 2.50 | 1.74 | 40 | 1.0×W（P2） | **1.9** | **0.9** |
| attention | 880 | 2.80 | ~0.5 | 160 | 1.0×W | **1.0** | **0.7** |
| compressor | 80 | 0.55 | 0.29 | 16 | 1.0×W | **0.34** | **0.25** |
| head | 10 | 1.12 | 1.09 | 2 | **0.20×W**（真 5×） | **0.25** | **0.22** |
| norm/quant | 680 | (0.20) | — | 250 | 1.0×W | **−0.5 摊回** | **−1.4 摊回** |
| **routed（换核）** | 400 | 8.30 | 6.98 | 40–120 | **0.25×W** | **2.5** | **1.5** |
| **8 族小计** | **3480** | **31.35** | **20.4** | **~570** | | **~13.5** | **~7.3** |
| 其余（投影/hc/engram/AR/激活） | ~2744 | 8.68 | 7.8 | 不变 | 1.0×W | 8.7 | 8.7 |
| **verify 合计** | **6224** | **37.31** | | **~1200–1400** | | **~22** | **~11** |

**读法**：
- **保守列**只承认「少发 kernel」（N ÷5）与已证明的指令模型（0.68–0.75×）——
  即使核效率一点不涨，也从 37.3 → 22ms（**−15ms**）。**这笔账不依赖任何未验证假设。**
- **目标列**额外承认「占用率 ×M ⇒ 有效 IPC 上升」与「head 的真 5× 字节」——
  落到 ~11ms。
- 与 `final-400-config.md` 的口径对照：它给 verify(m=6) ≈ 8.21ms（全 gate ON）。
  **8.21ms 是「目标列 + 全部 flag 都兑现」的乐观值**；本设计把「兑现」的机制写死成
  族级融合，而不是指望 flag 默认值。

### 4.3 与 5.5ms 的距离（诚实记账）

400 tok/s @ accept 3 ⇒ verify ≤ 5.5ms。本设计（族级融合 + 换核）落在 **11–22ms**，
**到 5.5ms 还差第二轮**：
1. **融合核内部**做 cp.async 多级流水（P4/P3 的既有模式）+ 满 wave 调优；
2. **hc 链**（2.96ms, 53 GB/s）与**投影/hc 的固定项**（3.70ms）——本设计未覆盖；
3. **AR v5**（1.40ms）是协议地板，只能减次数（§2.2 的拓扑已把它从 6 发/层压到 4 发/层）。

⇒ **本设计的定位：把 37ms 的地板抬到 ~11ms，并把「5.5ms 需要什么」变成可测的清单。**

---

## 5. 影响范围

| 文件 | 改动 |
|---|---|
| `kernels/cuda/dsv41_kernels.cu` | 新 `dsv41_sh_exp_fused<M>`（扩 `gemm_fp8_sh_pair_kernel`）；`sparse_attn*` 的 5 个 kernel 体加 `clen_rows`/`idx_stride`；新 `dsv41_ring_win_rows`；`dsv41_quant_fp8` 加 `src_stride`；`dsv41_compressor_fused` 加 `seqlen=m` 臂 |
| `kernels/cuda/dsv41_glue.cu` | 新 `dsv41_head_gemv_bf16_mrows_v1`（转写 `gemv_bf16_kernel` 体） |
| `kernels/cuda/dsv41_experts_mxf4.cu` | e4m3 臂的 N=5 行布局（`kNTile=8`，激活列取 M 行） |
| `kernels/cuda/ferrite_kernels.cu` | `gemv_bf16_nt_kernel` 的 M 行进 grid 变体（gate） |
| `crates/ferrite-models/src/dsv41/device.rs` | 新 launcher + `supports_*` 符号探测（老 `.so` 自动回退） |
| `crates/ferrite-models/src/dsv41/chain_dev.rs` | `attention_rows` 行循环拆分；`indexer_rows_one` 行化；`moe_rows` 的 gate/sh 段；`compress_proj_rows`/`compress_row`；head 段；新增 `clen_rows` 快照 |
| `crates/ferrite-dsv41/tests/` | 每族的 M 行 parity case（扩展 `tests_dsv41_glue.cu` / `tests_dsv41_attn.cu` / `tests_dsv41_head_mrows.cu`） |

**兼容性**：**无 breaking change**。所有新入口「追加尾参」/「新符号」；老 `.so` 靠 `ko!` 符号探测
自动回退；所有新 gate 默认 **OFF**，逐个 parity + A/B 后翻转。

---

## 6. 风险评估

| 风险 | 触发条件 | 应对 |
|---|---|---|
| **R1 · barrier 死锁** | `dsv41_sh_exp_fused<M>` 的 grid > co_res（smem 变大后 co_res 下降） | 沿用 `dsv41_gemm_fp8_sh_pair` 的 `co_res_cached` 硬 cap（`:6507`）；**绝不**越过；老 `.so` 回退 |
| **R2 · 逐位等价破功** | 融合核里 M 行共用一次 decode，编译器重排/重结合 | 每行独立累加器 + 每行独立归约树；**禁** `--use_fast_math` 影响 FMA 序（核对 `build.sh:43` 的 `FAST_MATH_FLAG`）；逐位回归是门槛 |
| **R3 · attention 的因果顺序** | 把 ring_append 放进 grid 维度（如 `verify_ring_win` 那样）会重演 defect #2 | **只允许** block 内 `for r in 0..m` 升序；禁止把 r 放进 grid.y |
| **R4 · per-row clen 快照缺失** | `indexer_topk`/`sparse_attn` 用 block-final clen ⇒ 读到自己/别人的未来组 | 快照必须「row r commit 之后、row r 的 reader 之前」；用 host mirror（已逐行推进）+ 一次 device 写 |
| **R5 · tcgen05 × e4m3 互斥** | 用 mxf4（e2m1）换核 ⇒ 数值域回到 opa | **换核的前置 = e4m3 臂 GPU parity**；不通过不换；text 四段 + `faults=0` 是终门 |
| **R6 · 收益高估（H1/H2 未定）** | 若 `c_k` 实际 << 3.3µs，则「保守列」的 −15ms 缩水 | **上机前的判别实验**（§7-R0）：一次 nsys 按 kernel 名聚合，数 4 个数（mrows 核、sh expert 段、expert batched、sparse_attn） |
| **R7 · 参数误接（项目 #1 陷阱）** | Rust gate 与 `.so` gate 不一致 ⇒ 两臂都测老路径 | 每个 gate 照 `build.sh` 的 BUILD_ID 机制；`.so` 符号探测 + 一次性 warning |

---

## 7. 建议分工与执行顺序

### R0（**阻塞一切，0.5 人日**）· 一次 nsys 判别
`DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_TIMING=1` + nsys，按 kernel 名聚合 verify 段，数：
`gemm_fp8_mrows` / shared-expert 段 / `expert_gate_up_fp4_batched` / `sparse_attn_split+merge` / `quant_kernel`。
**判据**：launch 数是否如预期下降、每发实际 µs。
**决定 §4.2 用保守列还是目标列。**

### 实施波次

| 波 | 内容 | 部门 | 人日 |
|---|---|---|---|
| **W1** | **indexer 行化 + norm/quant 行化 + head v1 mrows**（纯接线，最低风险，最快验证融合框架） | 工部 | 4 |
| **W2** | **attention `b·m` + per-row clen + ring_win_rows**（kernel 体改动最多，单独 parity） | 工部 | 3 |
| **W3** | **shared expert 三段一核**（`sh_exp_fused<M>`，最大单项） | 工部 | 3.5 |
| **W4** | **gate + sh gate/up 合一** + compressor rows（依赖 W3 的 kernel 形状） | 工部 | 3.5 |
| **W5** | **tcgen05 e4m3 臂的 N=5 行布局**（换核，最大风险） | 工部 | 4.5 |
| **并行** | 每波 parity 用例 + 逐位回归 | 刑部 + 户部 | 3 |
| **并行** | 每族 A/B（同会话背靠背）+ nsys 复核 | 户部 | 2 |
| **并行** | 数值等价论证复核（C1–C6 逐条） | 门下省 | 1.5 |
| **并行** | 命名/注释/文档（每个新 kernel 的 header 必须写清「bit-identity 论证」） | 吏部 + 礼部 | 1.5 |

**总计 ≈ 22–25 人日**（单人 4.5–5 周）；W1–W4 可在 2 周内拿到「保守列」的 ~20ms。

### 每波的验收判据（统一）

1. **逐位**：M 行融合核的 row r 输出 == M=1 核（或 eager 单行核）的 row r 输出，**逐字节**；
2. **launch 数**：nsys 计数下降到设计值；
3. **文本**：四段（Paris/Tokyo/1+1/静夜思）正文逐字正确 + `faults=0`；
4. **A/B**：同会话、同二进制、背靠背（`scripts/dsv41_serve_ab.sh`），报 p50 + `faults`；
5. **回退**：任一 gate OFF / 老 `.so` ⇒ 走既有路径，输出与今天**逐位相同**。

---

## 8. 关键架构决策（供后续引用）

1. **族级融合 ≡ 「N_kernel ÷ M」+「warp-in-flight ×M」+「阶段间走 smem」**；mrows（只做第一条）已被实测证伪。
2. **行放 grid 还是放 block，取决于该族是否 instruction-bound**：GEMV 族（shared/gate）放 block（省权重 decode）；
   行独立族（attention/indexer/compressor）放 grid（加 block 数）。
3. **attention 的 r 维永远不能进 grid.y**——它是因果顺序，只能进 block 内的串行循环。
4. **per-row clen 是 attention/indexer 行化的唯一前置**；没有它，`b·m` 在语义上就是错的（defect #1 会复现）。
5. **tcgen05 与融合在同一处汇合**：N=8 的最小 tile 天然容纳 m=5 行；e4m3 臂的 parity 是换核的前置。
6. **收益的第一性来源是 per-kernel 固定成本（~3.3µs × 6224）**，不是 submit（已证明重叠）、不是带宽（0.7–4.9% 峰值）。

---

*中书省 · 只读分析 + 本文档（唯一产出），未执行任何 GPU 命令、未改动任何源码。*
*所有 ms/launch 数均标了来源（实测/推算）；未实测项集中在 §4.3 与 §6-R6。*
*代码引用基于 HEAD `49c6fc3`；文档撰写期间该文件仍在被并行修改，行号以函数名为准。*

## W2 补充：attention 族 clen_rows[m] 的具体设计（主 agent 代码分析）

**当前限制**（kernels/cuda/dsv41_kernels.cu:7244/:7338）：attention kernel 体读单个 `const int* clen`——verify 的每行 clen 可能不同（行 r 的 compressor 状态在行 r-1 commit 后才确定）。

**b·m grid 已就绪**：`kAttnMaxBM=8`（:1449），`g_attn_part[kAttnMaxBM][...]`（:1452）的 part buffer 已有 b·m 维，`split_c` 路径（:7274/:7364）已检查 `b*m <= kAttnMaxBM`。

**改动清单**（约 3 人日）：
1. `sparse_attn_kernel` 等 5 个 kernel 体加 `const int* clen_rows` 参数——`clen = clen_rows ? clen_rows[blockIdx.y / b] : *clen`（或统一为 clen_rows[m] 指针，eager 传同一值的数组）
2. Rust 侧 `attention_rows` 构造 `clen_rows: [i32; VERIFY_ROWS]`（每行的 compressor len 快照）
3. **r 绝不进 grid.y**（因果序——audit defect #2 的教训）：grid = (b, m×h)，行维度在 blockIdx.y 的低段
4. ring/window 合核：`ring_append` 与 `window_idxs` 的 m 行批版

### W2 实施记录（工部，2026-09-12）——三项方案补正 + 两个硬前置

**已实施（ABI 5，`.so` 必须同源重建）**：5 个 attention kernel 体（`sparse_attn_kernel` /
`_warp_` / `_pf_` / `_split_` / `_orope_`；merge 不读 clen 未动）加尾参
`const int* clen_rows, int idx_stride`（orope 另加 `int row_step`），body 为
`const int cl = clen_rows ? clen_rows[mm] : *clen;`（`mm = row % m`，**不是** `blockIdx.y`——
见下方第 1 项）；launcher `dsv41_sparse_attn` / `dsv41_sparse_attn_orope` 同步加尾参
（stream 前）；`device.rs` 方法与 5 个调用点（verify 逐行 ×2、eager ×2、draft 的 `b=1,m=bs` ×1）
同步；`FERRITE_KERNEL_ABI_VERSION` / `EXPECTED_ABI` 4 → 5。`cargo check --workspace` EXIT=0。

**补正 1 · 还缺 `idx_stride`（否则语义错）**：per-row 调用传的 `idxs` 是
`idxs_r + r*(win+index_topk)`，即行距是 **`ist`**；而 `clen` 一旦逐行，`topk = win + min(cl, index_topk)`
也逐行，kernel 里历史的 `(bb*m+mm)*topk` 行基址会串到下一行。批版必须传 `idx_stride = ist`
（`idx_stride = 0` 保留历史 `topk` 行距）。

**补正 2 · orope 的 rope 位置要有行项**：`tt = (*base)*mul + off + hh*step` 里**没有行号**，
单行调用靠 `off = r` 表达；批版必须 `+ mm*row_step`（verify 传 `base=pos_ctr, mul=1, off=0,
step=0, row_step=1` ⇒ `tt = pos_ctr + mm`，与逐行的 `off = r` 逐位同一公式）。

**补正 3 · 行距在 `world > 1` 下不是 `h*d`**：kernel 用 `(row*h + hh)*d` 索引 q/out，而 verify 的行
在完整的 `[m, nh*hd]` 缓冲里（行距 `nh*hd`），只有 `h = nlh = nh/world` ⇒ **`world == 1` 时相等**。
批版在 TP>1 还需要一个 `row_pitch` 尾参（5 个 attention kernel + **merge**（split 是默认臂））。

**硬前置 A（因果序，本波最关键）**：`clen_rows` 只解决 defect #1（块末 `*clen`）。块内后续行的
`ring_append` 会覆盖前面行窗口读到的最旧 m-1-r 个 slot——**只有 `pos + m - 1 < window`（环未回绕）时**
`window_idxs_kernel` 的 `idx > start_pos` 过滤才仍然把「未写过的 slot」置 -1（此时 slot == position），
每行读到的 ring 镜像才与逐行相同。一旦 `pos + r >= window`，该过滤永不触发，行 `0..m-2` 会把块自己的
未来行当历史读——**正是 defect #2**（`verify_ring_win` 因此被回退）。长上下文要批 `b·m`，必须先做
§3.4(b) 的 ring/window 合核（或给 kernel 传「块前 ring 快照」把被覆盖的最旧行读回来）。

**因此当前接线（`DSV41_ATTN_MROWS=1`，默认 OFF）在 `world == 1` 且 `pos + m - 1 < window` 时
才取单发**，其余情况走逐行并**一次性打印 decline 原因**（避免「gate ON 但什么都没变」）。
OFF / 老路径的 `nullptr, 0, 0` 参数使输出逐位不变。

**要让它在生产（TP8 + 长上下文）真正生效，还差**：(1) `row_pitch` 尾参（补正 3）；
(2) ring/window 的块内 r 升序合核或块前 ring 快照（硬前置 A）。两条都不是本波的范围内改动。

## 全景修正：37.31ms 的完整分解（verify-ms-breakdown 的账本）——"其他 7.7ms"的谜底

之前我只看了族清单的 ~29.6ms，漏掉的 7.7ms 是：
- **投影族（wq_a/wkv/wq_b/wo）：3.70ms**（2000 launch，launch 主导）——verify 的每层 4 投影 × 5 行 × 40 层 = 2000 发。mrows 版已存在（`proj_mrows`）但 launch 数没减（还是逐投影逐层发）。**W6 候选：投影的层内 m 合并**（每层 4 发 mrows 而不是 20 发单行）。
- **all-reduce v5：1.40ms**（240 次 × 17.3µs）——TP8 的 3 核/层 × 40 层 × 2（投影后 + MoE 后）。协议地板。
- **hc 链：2.96ms**（400 launch，53GB/s 全表最低带宽）——hc_mixes 的 dots+sigmoid+sinkhorn 链。已有 HC_FRONT 融合（tail split）但主链还有 10 发/层。**W7 候选**。
- engrave 0.42ms、compressor 0.55ms、norm 0.20ms——小项。

**修正后的完整落点**（全部融合兑现后）：
| 族 | 现在 | 融合后 |
|---|---|---|
| routed experts | 8.30 | 8.30（tcgen05 阻塞）|
| shared expert | 10.40 | ~3.0（staging 修复 + sh_pair + occupancy）|
| 投影族 | 3.70 | ~1.2（W6 层内 m 合并）|
| attention | 2.80 | ~1.5（W2 clen_rows）|
| hc 链 | 2.96 | ~2.0（HC_FRONT 已有 + W7）|
| gate | 3.44 | ~1.5（GATE_MROWS 已有）|
| indexer | 2.50 | ~1.5（W1 front 已提交）|
| head | 1.12 | ~0.4（v1 mrows 进行中）|
| AR v5 | 1.40 | 1.40（协议地板）|
| 其他 | 1.17 | 1.17 |
| **合计** | **37.79** | **~22.0** |

→ 22ms + draft 1ms（P3c）+ commit 0.2 = ~23ms/步 → accept 3 时 **130 tok/s**。
**400 的缺口**：routed experts 的 8.3ms 是最大单项——tcgen05 是唯一路径（−6.8ms → 15.2ms/步 → 197 tok/s @ accept 3）。**accept ≥4 或 batched verify 的真正 weight-stationary（5 行共享 routed 权重读）才能到 400**。

## W5 tcgen05 e4m3 的 scale 外提设计（代码级，读 dsv41_experts_mxf4.cu 骨架后）

**硬件约束**（已确认）：
- `kind::mxf4` 的 MMA 有 block-scale 操作数但 **只吃 e2m1** 激活（opa 根因，不可回退）
- `kind::f8f6f4` 吃 e4m3 激活但 **无 block-scale 操作数**——e8m0 per-32 scale 无法进 MMA

**外提方案**（官方 tilelang 同构——`C_local_accum += C_local * scale_a * scale_b` 在 inner loop 的累加步）：
```
for each K-atom (64 fp4 元素 = 2 个 32-块):
    MMA(A_atom[fp8], B_atom[fp4]) → tmem 的部分积 C_local[128×64]
    对 C_local 的每 32-K 块: C_accum += C_local × scale_a[k_blk] × scale_b[k_blk]
```
**代价**：每个 K-atom 需要 1 次 tmem 读 + 2 次乘 + 1 次加（vs 单次大 MMA 的 1 次终读）。K=2304/rank → 36 atoms → 36 次 tmem 往返。
**收益模型**：SIMT 377GB/s（每 fp4 值 2.5 条 L1TEX 解码指令）→ tensor core 路径的解码消失。tcgen05 mxf4 骨架已按 masked M=128 tile 写好（kMTile=128/kNTile=64/kAtomK=64），tile 机制可直接复用。
**关键风险**：36 次 tmem 往返的延迟 vs SIMT 的 377GB/s——需要微基准定夺（预期 tensor core + tmem 往返仍快 3-5×）。

## 超越 400 的完整数学（用户目标：明显超过 400）

**当前基线**（lazy verify + m=1 图 + e4m3）：
- 步时 22.56ms，tok/step 2.08（accept 1.08），实际吞吐 ~92 tok/s
- k_acc 直方图 {0:55, 1:13, 2:10, 3:4, 4:3, 5:1}——**64% 的步 k_acc=0**（首个 draft 就被拒）

**400 需要**：吞吐 4.3× → accept 4.6 @ 22.56ms 或 accept 1.08 @ 5.2ms 或组合
**500+ 需要**：吞吐 5.4× → accept 5.8 @ 22.56ms 或 accept 5.4 @ 4.2ms 或组合

**工程侧（步时压缩，可确定）**：
| 项 | 现在 | 落地后 |
|---|---|---|
| verify | 37.31 | ~15（全融合 + tcgen05）|
| draft | 4.29 | ~1（P3c 图化）|
| commit | 0.20 | 0.20 |
| **步时** | **~24-42** | **~16** |

**accept 侧（乘数，关键未知数）**：
- sglang DSpark B=1 达 accept ~5 → 383.7 tok/s @ 步时 13ms
- ferrite 当前 accept 1.08——**draft 首token 拒绝率 64% 是瓶颈**
- bf16 截断可能提升 accept（backbone 对齐官方 → draft 更准）——等 GPU 验证
- **如果 bf16 截断把 accept 拉到 3+**：16ms × accept 3 = 187 tok/s；accept 5 = 312 tok/s
- **如果 accept 保持 1**：再怎么压步时也到不了 400（需要 2.5ms 步时——不可能）

**结论**：400 的成败在 **accept 率**，不在步时。bf16 截断（backbone 对齐）+ draft 数值路径对齐是关键。sglang 的 accept 5 说明 MTP head 有这个能力——ferrite 的 draft 实现有数值偏差压低了它。

## W4 compressor 族现状——compress_proj_rows 已存在（mrows 版）

`compress_proj_rows`（chain_dev.rs:9188）是 compressor 投影的 m 行版本（per-row `comp_wkv`/`comp_wgate` 投影的批版）。80 launch/步已部分批化。`compressor_fused` 有 NOTE："NOT used"（:9263）——有 kernel 但未启用，原因需查（可能是数值域或形状约束）。剩余优化空间：0.55ms → 0.25-0.35ms（fused 启用 + f32 mrows）。

## 全栈测试计划（段错误修复后的 gate 矩阵）

**第一轮（正确性——用户红线）**：
- `DSV41_BF16_TRUNCATE=1 DSV41_TAP_BF16=1`（双截断）
- 判据：零拉丁字符 + accept 提升（k_acc 直方图的 0-模态占比下降）

**第二轮（性能 gate 逐个开启）**：
| gate | 预期收益 | 风险 |
|---|---|---|
| DSV41_SH_EXP_MROWS=1 + MROWS_SMALL_N | -1~8ms（staging 修复后）| 低（位等价）|
| DSV41_SH_EXP_FUSED=1 | 额外 -2~5ms | 中（grid barrier）|
| DSV41_GATE_MROWS=1 | -2ms | 低（v2 位等价已断言）|
| DSV41_VERIFY_HEAD_MROWS=1 | -0.7ms | 低（v1 序逐位）|
| DSV41_INDEXER_MROWS=1 | -1ms | 低（front 融合）|
| DSV41_NORM_MROWS=1 | -0.1ms | 低 |
| DSV41_VERIFY_GRAPH=1 | -1.5ms | 低（已验证）|
| DSV41_LAZY_VERIFY=1 | 步时 2×（vs batched）| 与 mrows 收益重叠 |

**第三轮（组合判定）**：
- lazy vs batched 在全 mrows 下的对比（mrows 让 batched 的边际行成本降低，可能反转 lazy 的优势）
- 目标：步时 ≤16ms → accept 3+ 时 187+ tok/s；accept 5 时 312+

**AR v5 的 1.40ms 是协议地板**（TP8 3-kernel × 40 层 × 2 = 240 launch × 17.3μs）——不可再压。

## W2 attention m-rows 的因果序根本约束（attn-clen-rows 的发现深化）

**问题**：verify 块内行间有因果链——`read(r) → append(r) → read(r+1) → append(r+1) → ...`。行 r+1 的 read 依赖行 r 的 append。ring buffer 只有 `window` 个 slot，行 r+1 的 append 会覆盖行 r 的 read 窗口中最老的 slot。

**何时安全**：`pos + m - 1 < window`（环未回绕）——此时 `window_idxs_kernel` 的 `idx > start_pos` 过滤把未写过的 slot 置 -1（此刻 slot == position），画像与逐行一致。**长上下文的常态是环已回绕**——过滤永不触发，行 0..m-2 读到块自己的"未来"。

**为什么不能"先全部 read 再全部 append"**：行 1 的 read 需要行 0 的 append（因果依赖）。批化 read 会破坏这个依赖。

**可能的解法**（按复杂度）：
1. **scratch ring 快照**：每行 read 前把 ring 复制到 scratch（行 r 的视图 = scratch + 行 0..r-1 的 append 应用）——copy 成本 ~window×dim×2 bytes × m 行，可能得不偿失
2. **append 延迟 + 窗口偏移**：块内所有行共享同一个"旧" ring（不含本块的 append），每行的窗口起点偏移 r——但这不等价（行 r 应该看到行 0..r-1 的 KV）
3. **接受损失**：在某些场景（短上下文）安全，长上下文关闭 gate——动态选路

**结论**：attention 的 m-rows 在当前 ring 架构下**根本上受限**。1.3ms 的收益需要架构改动（scratch ring 或 KV 布局重构），ROI 低于 tcgen05（-6.8ms）和 draft P3c（-3.3ms）。**优先做 tcgen05 和 P3c**。

## 400 目标的最终数学（全部优化落地后）

**基线**（无截断 lazy，mrows staging 修复后）：18.82ms/步

**subagent 优化清单**（跑中）：
| 项 | 预期 | 状态 |
|---|---|---|
| hc verify 接线（A1+A2）| −1.3~1.7ms | hc-verify-wiring 跑中 |
| draft P3c 图化 | −3.3ms | draft-p3c-graph 跑中 |
| grouped routing + tcgen05 | −6.8ms | grouped-routing 跑中 |
| **合计** | **−11.4ms** | → 步时 ~7.4ms |

**accept 乘数**（关键变量）：
| accept | tok/step | 吞吐 @ 7.4ms | 400 达标 |
|---|---|---|---|
| 1.02（当前）| 2.02 | 273 tok/s | ✗（差 47%）|
| 2.0 | 3.0 | 405 tok/s | ✓（刚过）|
| 3.0 | 4.0 | 540 tok/s | ✓✓ |
| 5.0（sglang 级）| 6.0 | 810 tok/s | ✓✓✓ |

**结论**：步时 7.4ms + accept ≥2 即可达 400。accept 的三个杠杆：
1. **draft attention 截断**（DSV41_DRAFT_ATTN_BF16，已实现待测）
2. **tap 截断**（DSV41_TAP_BF16，已实现）
3. **bf16 主链截断**（DSV41_BF16_TRUNCATE，已验证零拉丁 + accept 43%k0）

**截断代价**：~14% 平均吞吐（78.8 vs 91.3 tok/s）——但换来零拉丁（红线）。净效应在 accept ≥2 时为正。

## W4 compressor_fused 未启用的原因（已查明）

`compressor_fused` 的 launcher **硬拒 `b != 1 || seqlen != 1`**（chain_dev.rs:4735/:9601）——它是单行单 token 的融合（pool+commit 对的 parity target），verify 的多行块不能用。pool+commit 对是 verify 的正确路径。**W4 的 0.3ms 优化需要多行版 compressor_fused（seqlen=m）**——低优先级（0.3ms）。
