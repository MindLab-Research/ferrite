# 投影族（wq_a / wkv / wq_b / wo_a / wo_b）优化分析

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 代码基线：HEAD `0bcb1e6`（`kernels/cuda/dsv41_kernels.cu` + `crates/ferrite-models/src/dsv41/{chain_dev,dspark_dev}.rs`，逐条 `file:line` 核对）。
> 输入数据：nsys 无图 per-kernel 表（`0bcb1e6` 提交的 19.4% / 22.56ms 口径）。
> **本机无 GPU ⇒ 所有 ms 标了来源（nsys 实测 / 结构推算 / 账本口径）。**

---

## 0. 判决（先读五条 —— 其中一条修正任务前提）

1. **🔴 `gemm_fp8_mrows_kernel<1>` 在 m=1 下确实退化为 per-row，且「权重共享」在 m=1
   没有收益可收 —— 但要分清两件事。** kernel 的权重行是**每个输出行一份**
   （`row_s = s_w + warp*k`，:5274），跨行共享的只有**激活行**
   （`s_a[r*k + i]`，:5231-5236）。「weight-stationary over the row batch」指的是
   「**同一个权重行**被 **M 个激活行**复用」，M=1 时 M 个变 1 个 ⇒ **复用次数为 0**。
   核里唯一跨 M 共享的量 `wv`（:5325/:5349，每个 `(row,kb)` 解一次、喂给全部 r）
   在 M=1 下也退化成「每行解一次」。**任务前提成立。**

2. **但 mrows 核在 m=1 仍比旧的逐行 GEMV 快，靠的是另一样东西：cp.async16 权重装载
   （:5272-5281）。** 核注释自己记了这笔账（:5244-5263）：这条 staging 以前是
   「1 字节/lane/次 ⇒ 32 B/warp-issue」，而 m=1 prologue 是「16 字节/lane/次 ⇒ 512 B」，
   **16 倍指令差**；并且明确写「**folding the m-fold weight traffic into this kernel
   bought nothing: the staging it was supposed to amortise was the cost**」。
   ⇒ **mrows 族在 m=1 的价值 = 一次 staging 修复，不是权重共享。**
   这条对下面的 ROI 排序是决定性的：**投影族的瓶颈是 launch + 指令，不是带宽。**

3. **🔴 带宽账证明：投影族没有「共享权重就能省」的空间。**
   每层 fp8 权重总量 = wq_a 6.55 MB + wkv 2.62 MB + wq_b 5.24 MB + wo_a(本 rank 1 组)
   4.19 MB + wo_b 5.24 MB = **23.84 MB**（`dsv41_flash.json` 尺寸 × fp8 1 B）。
   × 40 层 × k_emit 2.214 = **2.11 GB/步**。B300 HBM ≈ 8 TB/s ⇒ **~0.26 ms/步**。
   实测投影族 **~4.4 ms/步**（19.4% × 22.56）⇒ **≈ 17× 带宽地板**。
   ⇒ **投影族是 launch/指令受限（instruction-bound），不是带宽受限。**
   「mrows 权重共享」即便在 m≥2 兑现，也只动那 0.26 ms 里的一小部分。
   **唯一的杠杆是：更少的 launch、每次 launch 更少的指令、更好的占用。**

4. **🔴 wo_a 是 fp8，不是 f32 —— 任务前提 #3 证伪。**
   `wo_a_grouped_gemv_kernel` 的 `a`/`w` 都是 `const uint8_t*`（fp8 e4m3）+ `ue8m0`
   块 scale（:5537-5543），accumulator/output 才是 f32。仓库自己的账本也记过这条：
   `dspark-verify-perf-plan` 「P0-4 wo_a 格式 → 证伪；checkpoint 的 wo_a 确为 fp8」。
   ⇒ 「换 fp8 / bf16」这一项**不成立**：fp8→fp8 是 no-op；fp8→bf16 是**回归**
   （权重流量 ×2：本 rank 切片 4.19→8.4 MB，draft 全 8 组 33.5→67 MB）。

5. **⇒ wo_a 真正的问题是它没吃到 mrows 核拿到的那次 cp.async 修复。**
   wo_a 的权重行仍是**旧的标量装载**：`for (i = lane; i < k; i += 32) row_s[i] = wr[i];`（:5569），
   k=4096 时 **128 次迭代**；mrows 核同一段已是 `dsv41_cp_async16`（:5276）**8 次迭代**。
   另外 wo_a **没有 `__launch_bounds__`**（mrows 核有 `__launch_bounds__(256)`，:5209）。
   **这是全表 ROI 最高、最便宜的一项（R1）。**

---

## 1. 事实清单（尺寸与发数）

### 1.1 形状（`dsv41_flash.json` + TP8）

`dim=5120 · layers=40 · nh=64 · hd=512 · rd=64 · ql=1280 · olg=1024 · o_groups=8 ·
dspark_block_size=5`；TP8 ⇒ `world=8, nlh=8, nlg=o_groups/world=1`。

| 投影 | n × k（m=1） | fp8 权重 | 使用者 | 核 |
|---|---|---|---|---|
| `wq_a` | 1280 × 5120 | 6.55 MB | verify + draft | `gemm_fp8_mrows`（`proj_mrows`） |
| `wkv` | 512 × 5120 | 2.62 MB | verify + draft | 同上 |
| `wq_b` | 4096 × 1280 | 5.24 MB | verify + draft | 同上 |
| `wo_a` | 本 rank **1 组**：1024 × 4096 | 4.19 MB | verify | `wo_a_grouped_gemv` |
| `wo_a` | draft **8 组**：8 × (1024 × 4096) | 33.5 MB | draft | 同上 |
| `wo_b` | 5120 × 1024 | 5.24 MB | verify + draft | `gemm_fp8_mrows` / `gemm_fp8_mx_f32`(WOB_F32) |

> 发数口径注意：nsys 表 14820 / 3528 两个计数**行数一致**（14820/40/4 = 92.6，
> 3528/40 = 88.2），说明两个核的采样窗口覆盖同一批 ~90 行。
> 但「40 层 × ~78 步」与「22.56ms/步 × N 步」不能同时成立（78 步 ⇒ 22.56×78 = 1760ms
> vs 表推总时长 196.1/0.126 = 1556ms）。**本文件的 ms 以「每实例 μs × 每步实例数」推算，
> 不依赖步数口径；发数口径的 ±30% 不确定性见 §5。**

### 1.2 `gemm_fp8_mrows_kernel<M>` 结构（`dsv41_kernels.cu:5208-5368`）

```
一个 block = nwarps 个 warp，一个 warp = 一个输出行 row = blockIdx.x*nwarps + warp
每个 warp：装载【自己的】权重行 row_s = s_w + warp*k        (:5273-5281, cp.async16)
每个 block：【M 个】激活行 s_a[r*k+i] / s_as[r*nb_k+i]      (:5230-5236, 跨 warp 共享)
consume（a32=1 臂，:5319-5336）：
    for kb in 0..nb_k:
        wv = s_lut[s_w[warp*k + j]] * sb          ← 每个 (row,kb) 解一次
        for r in 0..M: af[r] = s_lut[s_a[r*k+j]] * s_as[...]; acc[r] += af[r]*wv
    for r: shfl_xor 树 → out[r*out_stride + row]
```
- **跨输出行的共享 = 0**（每个 warp 自己的权重行）。
- **跨激活行的共享 = `wv`**（权重方的解码），M=1 ⇒ 复用 0 次。
- m=1 时 `af[0]` 就是「materialised `s_af`」，与 m=1 GEMV 逐位相同（C1-C6 论证，:5145-5192）。

### 1.3 `wo_a_grouped_gemv_kernel<M>` 结构（`dsv41_kernels.cu:5536-5599`）

```
grid = (n/nwarps, groups)，nwarps = 8（n>=8）
每个 warp：装载自己的权重行 —— 【标量】for (i=lane; i<k; i+=32) row_s[i]=wr[i]   (:5569)
每个 block：装载本组的激活行（标量、block 宽，:5580-5581），然后 for r in 0..M
    for kb: av = s_lut[s_a[j]]*s_as[j>>5]; acc += av * (s_lut[row_s[j]]*sb)       (:5585-5590)
    shfl_xor → og[r*out_stride + row]
    __syncthreads()   ← 每 r 一次（:5597）
```
- **与 mrows 核的唯一结构差异就是权重装载没上 cp.async16**（且没有 `__launch_bounds__`）。
- a32 形式不同（wo_a 内联折 `av*wv`，mrows 用 a32 臂物化），仓库记为按构造逐位相同（:5509-5519）。

---

## 2. 逐问回答

### Q1 —— `gemm_fp8_mrows` 在 m=1 下是否退化为 per-row？权重读是否共享？

**退化，且权重读完全无共享。** 见 §0-1 / §1.2。三点结论：

1. **权重行不共享**：`row_s` 按下标 `warp` 分配（:5274），一个 warp 一行，与 M 无关。
2. **跨行唯一的共享量 `wv` 在 M=1 复用 0 次**（:5325）。
3. **mrows 在 m=1 仍优于旧 GEMV 的唯一原因是 cp.async16 staging**（:5276），
   与「行批」无关 —— 这一点核注释自己已记录（:5244-5263）。

⇒ 对 lazy（m=1）而言：**「mrows 化」这个词是误导的**。核选得对（且它是 m=1 最优的
SIMT GEMV 形态），但**没有可收的权重共享红利**；能收的只有「launch 数」与「每发指令数」。

### Q2 —— 融合机会

**(a) `wq_a + wkv` 合并为一次 GEMM（共享 K-walk / 共享激活）—— EAGER 已有，verify 没用。**

- 两者消费**同一个激活** `xn_r`（dim=5120）：`chain_dev.rs:9310-9350`。
- EAGER 单行路径**已经融合**：`lin2`（`chain_dev.rs:4221`）→ `dsv41_gemm_fp8_mx2`
  （`dsv41_kernels.cu:7587`），gate **`DSV41_PROJ_FUSE` 默认 ON**。
  一次 quant1 + 一次 GEMV（行 `< n1` 属 family1，其余 family2），两路输出分别落 `qr`/`kv`。
- **verify 路径（`attention_rows`）没有用它**：`proj_mrows(wq_a)`（:9333）+
  `proj_mrows(wkv)`（:9342）是**两次独立 mrows launch**。
  注意 quant 已经是共享的（`quant_rows(xn_r)` 一次，:9332），
  ⇒ 融合省的是**恰好 1 次 launch**（+1 个 graph 节点），不是字节。
- ⚠️ **「共享 K-walk」在 m=1 不产生字节收益**：激活行只有 5120 B，已经 block 内共享；
  收益全在 launch 数上（§0-3）。

**(b) 投影 + rope 融合 —— EAGER 已有 `lin_rope_norm`，verify 没用；这是最大的一项。**

- 核：`dsv41_gemm_fp8_mx_rope`（:5746）、`dsv41_gemm_fp8_mx_rope_norm`（:5832，norm+gemm+rope 一发）。
- EAGER：`attention()`（:12932）在 `norm_fuse()`（`DSV41_NORM_FUSE` 默认 ON，:13037-13065）
  下走 `lin_rope_norm` —— **rmsnorm_q + fp8 编码 + wq_b GEMV + rope 合 1 发**，
  替代 `rmsnorm_q + quant1 + gemv + rope` 的 4 发。注释记「40 per step」的省。
- verify：`attention_rows` 走的是
  `norm_rows(qr_r)`（:9383）→ `quant_rows(qr_r)`（:9396）→ `proj_mrows(wq_b)`（:9397）
  → `apply_rope_mrows(q_r)`（:9436）**= 4 发**。
- ⇒ **q 链在 m=1 下 verify 比 EAGER 多 ~3 发/层/行**（§Q4）。
- 另一个相关核：`dsv41_gemm_fp8_mx2_rope`（:5902）—— 两 family + rope，if 要一次 fuse wq_a+wkv+wq_b 的路线可用。

**(c) `wo_a → wo_b` 段核 —— 已有 `dsv41_gemm_fp8_wo_pair`（:6361），gate `DSV41_WO_PAIR` 默认 OFF。**

- grid-sync 两段核（设备级 barrier），每行仍是同一 lane 序 ⇒ 逐位相同；省 1 launch + 1 graph 节点/层。
- 当前只验证了 decline/fallback 路径（`chain_dev.rs` 记「Default OFF because it is a NEW kernel
  whose only verified property so far is the decline/fallback path」）。

**(d) fp8 族内没有更多 mx2 可合并对**（`dsv41-kernel-inventory-v3` §105 已记），
剩余的只有 (a)(b)(c) 三类 + `wo_a→wo_b` 的段核（= (c)）。

### Q3 —— wo_a 的优化

**先证伪任务前提**：wo_a 已是 fp8（§0-4）。「换 fp8」不成立；「换 bf16」是权重流量 ×2 的回归。
真正可做的：

1. **cp.async16 权重装载（★R1）**：`:5569` 的标量循环 → mrows 核 `:5272-5281` 的形式。
   - 量：k=4096 ⇒ 装载指令 128 → 8 次/ lane（16×）。
   - 结构占比：装载 ≈ 128×(LDG.U8+STS.U8) = 256 warp-inst；consume ≈ nb_k=128 × ~8 = ~1024 warp-inst
     ⇒ **装载约占 ~20%**。
   - 对齐：`k & 31 == 0` 由 launcher 保证（:5610），`row*k` 与 `s_w + warp*k` 都是 16B 对齐 ⇒ 可直接上。
   - 数值：纯拷贝，K-walk / `acc` 链 / shfl 树不动 ⇒ 逐位不变（与 mrows 的同款修复同论证）。
2. **补 `__launch_bounds__`**：wo_a 当前没有（mrows 有 `__launch_bounds__(256)`，:5209），
   寄存器分配不受约束，可能压占用。
3. **占用/小 n 自适应**：wo_a 在 verify（nlg=1）的 grid = `(1024/8, 1) = (128, 1)`，
   B300 148 SM ⇒ **128 < 148，20 个 SM 空转**。`DSV41_MROWS_SMALL_N_ADAPTIVE`
   （`dsv41_mrows_warps_for`，:5409-5418）已给 mrows 族做了同类行/块臂，wo_a 可照抄
   （nwarps 4 ⇒ grid 256 > 148，两波，尾更均匀）。

> **不建议**：把 wo_a 的激活从 fp8 换成 f32 以省掉 `quant_fp8(o_r)`（:10014）。
> 实测上这个 quant 多数行已被 `sparse_attn_orope` 的 phase-3 顺手做掉（:10005-10009），
> 且在 verify nlg=1 下 wo_a 的激活字节数（4096/组）远小于权重（4.19 MB）——
> 性价比远低于 R1。

### Q4 —— 与 EAGER 的对比：EAGER 的投影更快吗？差在哪？

**更快，且差距来自「verify 在 mrows 化时丢掉了单行融合」。**

| 每层每行（m=1） | EAGER `attention()` | verify `attention_rows(m=1)` |
|---|---|---|
| wq_a + wkv | **`lin2` = 1 发**（PROJ_FUSE ON，含 quant） | `quant_rows`? + `proj_mrows`×2 |
| q norm + wq_b + rope | **`lin_rope_norm` = 1 发** | `norm_rows` + `quant_rows` + `proj_mrows` + `apply_rope_mrows` = **4 发** |
| 小计（q 链） | **2 发** | **6~7 发** |

`attention_rows` 的完整 q 链发数（:9332/:9333/:9342/:9383/:9396/:9397/:9436）：
`quant_rows(xn)` → `proj_mrows(wq_a)` → `proj_mrows(wkv)` → `norm_rows(qr)` →
`quant_rows(qr)` → `proj_mrows(wq_b)` → `apply_rope_mrows(q)` = **7 发**（m=1）。

- **差距根因**：`attention_rows` 是为 m≥2 的 batched 形态写的（mrows 是它的主收益），
  单行融合（`lin2` / `lin_rope_norm` 都是 **m=1 专用**核 —— `gemm_fp8_gemv_kernel` 家族）
  在 m≥2 下不适用，于是 mrows 化时被整体丢掉了。**对 lazy（m=1）而言这是净损失。**
- **EAGER 的 c_row = 6.15 ms** 对 lazy `c_row` 的对照（`lazy-verify-optimization-path §1.2`）
  里，残余 2.03 ms/行的归因清单**没有列入「投影单行融合缺失」**——
  它是一个被漏记的、可机械回收的项（见 R2）。

### Q5 —— 具体建议（按 ROI 排序）

见 §3。

---

## 3. 优化建议（按 ROI 排序）

| # | 项 | 改动 | 预期节省 | 成本 | 依据强度 |
|---|---|---|---|---|---|
| **R1** ★ | **wo_a cp.async16 权重装载 + `__launch_bounds__`** | `dsv41_kernels.cu:5569` 标量循环 → mrows 核 `:5272-5281` 形态；加 `__launch_bounds__(256)` | **−0.3 ~ −0.5 ms/步**（105.1ms × ~15-20% ≈ 16-21ms / 步推 ~0.4ms） | ~0.5 人日 + 1 GPU parity | 姊妹核同款修复已在树内（:5244-5281） |
| **R2** ★ | **m=1 的 q 链融合（verify 抄 EAGER）** | `attention_rows` 加 `if m == 1` 分支，复用 `lin2`(:4221) + `lin_rope_norm`(:4359)，decline 落回现路径 | **−0.7 ~ −1.9 ms/步**（7 发 → 2 发，省 ~5 发 × 40 层 × k_emit；并消掉 wq_b 的 mrows 实例 = 12.6% 的 1/4） | ~1-2 人日 + GPU parity | 核已在树内且 EAGER 默认 ON；仅需 m=1 分支 |
| **R3** | **`DSV41_WO_PAIR` 启用**（wo_a → wo_b 段核） | 开 gate + A/B（核 :6361 已在） | **−0.25 ~ −0.4 ms/步**（−1 launch/层/行，−1 graph 节点） | ~0.5-1 人日 A/B | 核仅验证过 decline 路径 |
| **R4** | **wo_a 行/块臂（小 n 自适应）+ 补占用** | 照抄 `dsv41_mrows_warps_for`(:5418) 的形态给 wo_a launcher | **−0.1 ~ −0.3 ms/步**（128→256 block，148 SM 不再空 20 个） | ~0.5 人日 + GPU | 结构推算（未实测） |
| **R5** | **wq_a+wkv 单发（mrows2）** | 仿 `gemm_fp8_mx2`(:7587) 写 `gemm_fp8_mrows2`；**若 R2 落地则已被 `lin2` 覆盖，勿重复** | −0.2 ~ −0.35 ms/步（单独看） | ~1 人日 | 机械合并；R2 会先覆盖 |
| **R6** | **（结构性，暂不做）行批 ×k_emit 的消除** | 让投影在 m≥2 下一次付费 —— 即 batched 臂 | 理论最大，但触发 accept 依赖的 `ar5-hang` 与 speculative 行浪费 | 高 | `lazy-verify-optimization-path §L2` 已分析并否决（speculative 行 +1.79 行/步 ≫ 省的 launch） |

**ROI 判读**：R1 是「同样的修复在隔壁核已落地、这边漏了」的确定性收益，最便宜 → **先做**。
R2 是最大单项，且核/launcher 都已在树内、EAGER 默认 ON，只需在 verify 加一个 m=1 分支
→ **紧随其后**。R3/R4 是低风险增量。R6 明确不在本轮。

---

## 4. 与既有账本的关系（防重复计数）

- `verify-family-fusion §552` 的「投影族 3.70ms（2000 launch，launch 主导）→ W6 层内 m 合并」
  是 **batched（m=5）口径**：那里 launch 数 = 4 投影 × 5 行 × 40 层。
  本文件的 R1/R2 是 **lazy（m=1）口径**，两者**不可相加**：
  lazy 下 mrows 的行批收益恒为 0（§0-1），veriy 的收益源是「单行融合」而非「层内 m 合并」。
- `lazy-verify-optimization-path §1.2` 的「残余 +2.03ms/行」三行归因（hc 未融合 / per-row
  host round-trip / head·norm·engram 固定项）**未包含投影单行融合缺失** ——
  R2 属于该清单之外的新增量，不与 L1/L2 重叠。
- 投影族的 **~4.4ms/步** 与「per-step 族 ×k_emit」（hc+AR ~9.1ms/步）**是两个独立族**，
  合计不改变 `c_row` 的口径。

---

## 5. 诚实校准（必须写在账上）

1. **发数口径不确定**：nsys 的 14820 / 3528 与「~78 步」的乘法不自洽（§1.1 注）。
   本文件的 ms **按「每实例 μs × 每步实例数」推算**，若真实步数不同，绝对值线性缩放，
   **相对排序与「节省占比」不变**。
2. **wo_a 的 29.8μs 到底是哪个形态未定**：若为 **draft 的 8 组形态**（权重 33.5 MB、
   grid 1024 block），则 29.8μs 合理（带宽地板 ~4.2μs，指令主导）；
   若为 **verify 的 1 组形态**（4.19 MB、grid 128 block），则 29.8μs 有 ~8× 的地板余量
   —— 无论哪种，**R1 的 cp.async 修复对两者都成立**。
3. **R2 的数值等价是「按构造」的断言，不是实测**：`gemm_fp8_mx_rope*` 与
   「rmsnorm_q + quant + gemm + rope」的逐位等价由内核注释承担（`chain_dev.rs:13037-13049`、
   `dsv41_kernels.cu` 对应段），**需 `dspark_parity` 行级对照实证**。
   ⚠️ 另：`lin_rope_norm` 会令 `qr` 保持未归一化（`s.qr_raw`），
   `indexer()` 必须跟随同一融合（:13046-13049）——即 R2 的 m=1 分支要连 `indexer` 的读数一起接。
4. **R1 的 15-20% 是结构占比推算，不是实测**：装载 256 vs consume ~1024 warp-指令。
   若 wo_a 实际受限于其他因素（如 launch 尾/占用），收益会低于此。
5. **未执行任何 GPU 命令**（本机无 GPU）；未改任何源码。

---

## 6. 验证计划（最少 GPU 次数）

| # | 会话 | 内容 | 判据 |
|---|---|---|---|
| **P1** | GPU（1） | R1：wo_a cp.async A/B（`DSV41_WOA_CPASYNC=0/1`）| 文本逐字 + `faults=0`；`wo_a_grouped_gemv` nsys 时间应 −15~20% |
| **P2** | GPU（1） | R2：`attention_rows` m=1 融合 A/B | `dspark_parity` 行级 `verify_bad == 0`；q 链 launch 数 7→2（看 nsys / `[verify_graph]`） |
| **P3** | GPU | R3：`DSV41_WO_PAIR=0/1` | 同上；−1 launch/层 |
| **P4** | GPU | R4：wo_a 小 n 自适应 A/B | 同 P1；128→256 block |
| **V0** | GPU | 先确认 nsys 采样窗口的真实步数/行数 | 定 §5-1 的绝对口径 |

**铁律**：同一远端同时只有一个测试驱动；`cargo check --workspace` 是本地硬门禁；
每个 gate 翻转必须读回确认（R6 历史教训：gate 设了但没生效）。

---

## 附：一句话总结

**投影族的 ~4.4ms/步是 launch/指令受限（17× 带宽地板），不是带宽受限——
所以「mrows 权重共享」在 m=1 无收益可收；能收的三笔是：
① wo_a 漏掉的 cp.async16 staging（R1，最便宜）；
② m=1 下 verify 丢掉的 EAGER 单行融合（R2，7 发→2 发，最大单项）；
③ wo_a→wo_b 段核与占用（R3/R4）。wo_a 本就是 fp8，不存在「换 fp8/bf16」的空间。**

---

*工部 · 只读分析 + 本文件（唯一产出），未执行 GPU 命令、未改动任何源码。*
*所有 ms 标来源（nsys 实测 / 结构推算 / 账本口径）；未实测项集中在 §5。*
