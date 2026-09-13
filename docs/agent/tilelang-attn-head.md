# TileLang × head GEMM 与 attention 侧的探索评估

> 工部 · 2026-09-13 · **GPU 实测**（远端 B300 sm_103a，TileLang 0.1.14，`ssh ubuntu@43.202.208.136`；
> micro bench，未跑 e2e serve）+ **只读分析**（attention 侧）。
> 前置：`tilelang-proj-proto.md` / `tilelang-proj-phase2.md`（投影五形状 route A + wo_a 分组，
> M6/M1 = 0.99–1.01）、`tilelang-moe-grouped.md`、`verify-family-fusion.md` §3.4/§3.6、
> `verify-amortization-lesion-audit.md` §10.2/§10.5、`fusion-alignment-orope-linrope-mrows.md`。
> 代码基线：`kernels/cuda/dsv41_glue.cu`（head）、`kernels/cuda/dsv41_kernels.cu`（sparse attn）。
> 交付：`kernels/tilelang/head_bf16_tilelang.py`、`kernels/cuda/bench_head_old.cu`、本文件。

---

## §0 一句话判决

| # | 项 | 判决 |
|---|---|---|
| ① | **head（bf16 GEMM）TileLang 原型** | ✅ M6/M1 = **1.00**（slice/full/literal 三形状）；老 `v1_mrows` 是 **3.23–3.81** |
| ① | 判据 `<1.5×` | ✅ **通过**（1.00 vs 老 3.27/3.81） |
| ① | 绝对值（M=6，partial-only） | slice **42.2µs**（老 125.2µs，**2.97×**）/ full **308.6µs**（老 903.5µs，**2.93×**） |
| ② | **sparse attention 的 TileLang 化** | ⚠️ **判定：不值得做，保持自研 orope 路线**（4 条结构障碍，§5） |
| ③ | verify 最终 TileLang 化构成表 | **~10.4ms**（从 24.5ms；见 §6） |

> ⚠️ **任务书形状表与代码事实冲突（已按代码实现，请尚书省知悉）**：任务书写 head 是
> 「词表 **16160×256** 的 bf16 GEMV」。ferrite 的 head 是 `head.weight` = **[vocab=129280, dim=5120]
> bf16**（`weights.rs:90`，`Shard::Replicated`），切片后 **[16160, 5120]**。**16160 是切片行数（对），
> 256 不是 head 的 K**。本文件三种形状都跑了：生产切片 `[16160,5120]`、verify 未切 `[129280,5120]`
> （ARSAFE 的 1.36ms 就是它）、任务书字面 `[16160,256]`（§2 表第三行）。
> 另：`markov_head`（draft 的 MTP 头）才是 `[vocab, 256]`（`dspark_dev.rs:377`），且是 **f32 126 MiB**
> 由 `dspark_markov_head_kernel` 跑 —— 与 `v1_mrows` 无关。

---

## §1 被测对象（代码事实）

### 1.1 head —— `dsv41_head_gemv_bf16_mrows`（`dsv41_glue.cu:864`）

```
template<int M> head_gemv_bf16_mrows_kernel(w /*bf16 [n,k]*/, x /*f32 [m,k]*/, out /*f32 [m,n]*/, n, k)
launch: ceil(n/8) blocks（cap 4096），256 threads = 8 warps；m = 1..8 模板分派（m>8 decline）
```

- **权重 w 是 bf16**（词表嵌入/投影矩阵），**激活 x 是 f32**（verify 的 `xn_r`）——kernel 把 w 解码到
  f32 后做 **f32 FMA**（`gemv_bf16_nt_kernel` 的 WPR==1 body 的逐字转写）。
- 数值契约：**row r 与逐行 `v2` 单发逐位相同**（`tests_dsv41_head_mrows.cu` 的 bit-parity 门）。
- ⚠️ 但 **head 真正跑的是 v1**（`dsv41_gemv_bf16` → `gemv_bf16_kernel`，单 warp shuffle 树），
  v1/v2 的求和顺序不同 ⇒ 生产 head 用 **`dsv41_gemv_bf16_v1_mrows`**（`dsv41_glue.cu:442`）；
  `docs/agent/draft-head-fold-v2-argmax-verdict.md` 记 v2 折叠使 `verify_out[0]==next` 从 9% 涨到 33%。
  本 bench 测的是 `dsv41_head_gemv_bf16_mrows`（v2 序）——**它代表"多行折叠的形态"，
  不是生产的 v1 序**；TileLang 的 M6/M1 结论与序无关。

### 1.2 sparse attention —— `sparse_attn_orope_kernel`（`dsv41_kernels.cu:2232`）

```
grid = (b*m, h)   block = 128 (4 warp)   d = head_dim = 512（launcher 守 d<=512）
每行（b*m 展平）× 一个 head 一个 block：
  for t in topk:  kv-row = gather(idxs[t]) 经 dsv41_kv_win_fetch（位置解析的 ring 读）
                  dot = <q_row, kv_row>（warp shuffle 树） → online softmax → acc += e*kv_row
  merge 4 warp 的 partial → phase2 逆 rope → phase3 fp8 发射（xq/xsc）
```

生产几何（`config.rs`）：`dim=5120, n_heads=64, head_dim=512, rope_head_dim=64, window=128,
index_topk`（外部 JSON 驱动）。`topk = window + min(cl, index_topk)`，`cl` 是 **device 上的
compressor 计数**（不是常量）。

---

## §2 head 的 TileLang 原型（交付 ①）

`kernels/tilelang/head_bf16_tilelang.py`（原型 / 未接线，不在 `build.sh`）。

**形态（沿用投影族 route A 的 K-split 结构，去掉了 scale epilogue）**：

```python
with T.Kernel(T.ceildiv(N, bN), ks, threads=threads) as (bx, kp):
    X_sh = T.alloc_shared((MPAD, bK), "bfloat16")   # MPAD=16：mma m16n8k32 的 M 约束
    W_sh = T.alloc_shared((bN, bK), "bfloat16")
    C_l  = T.alloc_fragment((MPAD, bN), "float32")
    T.clear(C_l)
    for ko in T.Pipelined(Kc // bK, num_stages=ns):
        T.copy(X[0, kp*Kc + ko*bK], X_sh)
        T.copy(W[bx*bN, kp*Kc + ko*bK], W_sh)
        T.gemm(X_sh, W_sh, C_l, transpose_B=True)     # 纯 bf16 mma，无 scale
    T.copy(C_l, P[kp, 0, bx*bN])
```

- **没有 scale epilogue**：head 是稠密 bf16 GEMM（无 per-32 fp8 scale），这是它比投影族**更简单**的地方。
- **`bK` 必须整除 `Kc`**：任务书字面的 `K=256, ks=8 ⇒ Kc=32 < bK=64` 会让 pipelined loop **空转**
  ⇒ 输出恒 0（**静默错值**，§3 第 3 行是修前的实测）。已加 `bK = min(bK, Kc)` + assert。
- **M 必须 pad 到 16**（与投影族同）：M=1 时 mma 浪费 16×，但 head 是权重流主导 ⇒ 不体现（M6/M1=1.00）。

---

## §3 head benchmark（M1/M6 + M6/M1）

两侧同机同口径（CUDA event / torch event 包住 launch，1000 iters；TileLang 侧 launch 底噪
**5.74–5.90µs**）。老 kernel = `kernels/cuda/bench_head_old.cu`（独立 TU，`#include "dsv41_glue.cu"`
后直调 `dsv41_head_gemv_bf16_mrows`，nvcc `-O3 --use_fast_math -gencode arch=compute_103a`）。

> ⏱️ **测量窗口（重要）**：本表全部数字测于 **04:02–04:04 UTC**，该机器当时**空闲**
> （`bench_head_old` 产物 mtime 04:02:21、`head_all.log` mtime 04:04:07）。**04:07:14 起有 peer 的
> `ferrite-serve --tp 8` 在同一机器占满 8 卡**（78.8GB/卡、100% util）⇒ **本表不受其影响，
> 但无法在其后又做干净复测**（见 §7#1）。

### 3.1 双侧原始表（µs/call）

| shape | n | k | 权重 MB | **老 v1_mrows m=1** | **m=6** | **老 M6/M1** | **TL partial m=1** | **m=6** | **TL M6/M1** | TL M6 加速 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `head_slice`（生产切片） | 16160 | 5120 | 157.8 | 38.26 | **125.22** | **3.273** ❌ | 42.24 | **42.21** | **1.00** ✅ | **2.97×** |
| `head_full`（verify 未切，ARSAFE） | 129280 | 5120 | 1262.5 | 237.13 | **903.51** | **3.810** ❌ | 311.70 | **308.55** | **0.99** ✅ | **2.93×** |
| `csv_literal`（任务书字面） | 16160 | 256 | 7.9 | 3.668 | **11.859** | **3.233** ❌ | 9.01 | **8.99** | **1.00** ✅ | 1.32× |

（`csv_literal` 的 TL m=1 9.01µs 是**修 `bK` bug 后**的值；修前恒 0 输出、8.99 是假数。）

### 3.2 判决（判据：M6/M1 < 1.5×）

1. **M6/M1**：TileLang **1.00**（三形状），老 `v1_mrows` **3.23–3.81**。
   ⇒ **判据全过**，且把老的 M 放大 **3.2–3.8× 压平到 1.0×**。
2. **绝对值（m=6）**：TileLang 比老 kernel 快 **2.93–2.97×**（slice 125.2→42.2µs；full 903.5→308.6µs）。
   **这才是 head 的真实肉**——不是 M=1 上的提升。
3. **绝对值（m=1）**：TileLang **慢 1.2–1.3×**（slice 38.3→42.2µs；full 237.1→311.7µs）。
   M=1 时 mma 的 16 行 pad 白做，而 v1_mrows 的 SIMT GEMV 在 m=1 已达 **4.33 / 5.58 TB/s**
   （73% 峰值）。⇒ **TileLang 的收益全部来自 M 折叠**。
4. **达成带宽**：TileLang m=6 是 **3.94 / 4.00 TB/s**（slice/full）；老 m=6 只有 **1.32 / 1.47 TB/s**。
   ⚠️ **都还没到 7.67 TB/s 峰值**——TileLang 侧还有 ~2× 的空间（见 §7#1，`ks`/`bN` 未穷举）。
5. **形状越窄（K=256）越不划算**：权重只 7.9MB，42µs 里 6µs 是 launch 底噪，且 TL（9.01）比老（3.67）
   **慢 2.5×**（M=1）——**小 K 的 GEMV 是 v1_mrows 的地盘**，TileLang 的 K-split 网格反而摊薄了它。

### 3.3 归因：老的 M 折叠为什么只有 3.2–3.8× 而不是 1.0×

`head_gemv_bf16_mrows_kernel` 把 `wv`（uint4 = 8 个 bf16）**提到 r 循环之外**（权重只解码一次、读一次），
所以 6 行**不重读**权重（否则是 6×38=230µs，实测 125µs）。但 **acc[M] 的寄存器链 × M + `#pragma unroll`
M** 把 kernel 从带宽受限拖成**延迟/占用受限**（1.32 TB/s vs m=1 的 4.33）。

TileLang 把 **M 放进 mma 的 tile 维度**（m16 一发覆盖 6 行 ⇒ M 不再是迭代维度），
**且 K-split 把块数填满 148 SM** —— 两条合起来就是 3.2–3.8× → 1.00×。
这与投影族 phase2 §6.3 的归因**同源**（"M-in-register 串行折叠" vs "M-in-tile"）。

---

## §4 head 数值形态（交付 ① 的一部分）

**这是本原型最重要的判读点。** TileLang 的 bf16 mma 要求 **A 也是 bf16**，而 head 真实程序是
**"bf16 权重 × f32 激活"**（§1.1）——本原型在 host 侧把 `x` cast 到 bf16。

`selftest`（m=6，bN=128 ks=8 ns=3）：

| shape | 口径 | max_abs | p50 rel | p99 rel | **max rel** | mean rel |
|---|---|---:|---:|---:|---:|---:|
| slice | **TL vs f32 ferrite 口径** | 2.536e-03 | 1.296e-03 | 1.039e-02 | **2.320e-02** | 1.990e-03 |
| slice | TL vs f64 真值 | 2.536e-03 | 1.296e-03 | 1.038e-02 | 2.320e-02 | 1.990e-03 |
| slice | **x-cast bf16 后 f32 参考**（不换 program，只换 x 精度） | 2.536e-03 | 1.296e-03 | 1.039e-02 | **2.321e-02** | 1.990e-03 |
| full | TL vs f32 ferrite 口径 | 2.844e-03 | 1.238e-03 | 9.147e-03 | 2.253e-02 | 1.846e-03 |
| full | x-cast bf16 后 f32 参考 | 2.844e-03 | 1.238e-03 | 9.147e-03 | 2.253e-02 | 1.846e-03 |
| literal | TL vs f32 ferrite 口径（**修 bK bug 后**） | 5.730e-04 | 1.356e-03 | 1.102e-02 | 2.387e-02 | 2.090e-03 |

**读法（三条）**：

1. **误差 100% 来自 `x` 的 bf16 cast，不是 mma 的重结合**：`TL vs ferrite` 与 `x-cast 后参考`
   的 max_rel **逐位同量级**（2.320e-2 vs 2.321e-2）。⇒ **TileLang 的 bf16 mma 在给定 bf16 输入下
   与 f32 参考几乎一致**（mma 内部是 f32 累加），换 program 的代价 ≈ 0。
2. **`max_rel ~2.3e-2` 是 bf16 的 8 位尾数（2^-8 = 3.9e-3）经 K=5120 大和放大后的结果**
   （mean rel ~2e-3 = 1 个 bf16 ulp 的量级）。⇒ 这是 **cast 的物理下界**，不是实现缺陷。
3. ⚠️ **但 head 输出喂 argmax**：`draft-head-fold-v2-argmax-verdict.md` 已证 **~1e-3 的重结合差
   就能让 near-tie argmax 翻面**（echo 9% → 33%）。**2.3e-2 的 x-cast 误差远大于 1e-3**
   ⇒ **TileLang head（bf16 激活）不能直接替换生产 head**，除非：
   - (a) 生产侧把 head 的激活改成 bf16（那它本来就不是现在的程序了）；或
   - (b) 用 **bf16 hi/lo 双份** 模拟 f32 激活（`x = x_hi + x_lo`，两次 mma），代价 ×2 权重流量 —— **不值**；或
   - (c) **只把 TileLang 用在"不需要 argmax 逐位精度"的路径**（如 draft 的临时分布）。

   ⇒ **head 的 TileLang 化在"时序"上赚（2.93×），在"数值契约"上欠债** —— 这个债必须由
   `verify` 的 acc 门（mean-k 逐位 / Z_）来还，不能用 §4 的表掩盖。

---

## §5 sparse attention 的 TileLang 可行性（交付 ②，只读判定）

**结论先行：不值得做，保持自研 orope 路线。** 理由按硬度排序：

### 5.1 四条结构障碍

| # | 障碍 | 具体 | TileLang 的态度 |
|---|---|---|---|
| 1 | **KV 是 gather，不是连续 append** | `irow[t] = idxs[...]`（indexer 的 top-k 选择），且槽位经 `dsv41_kv_win_fetch` 的**位置反解**（`p = pos_r - ((pos_r-idx) mod win)`）在 ring / 本块 `kv_rows` 之间二选一 | TMA / `T.copy` 描述不了。要写只能退化成 `T.Parallel` 里的 `if` + 手工索引——**等于把 SIMT kernel 用 TileLang 语法重写一遍**，零增益 |
| 2 | **topk 是 device 动态量** | `topk = window + min(cl, index_topk)`，`cl` 来自 compressor 的 device 计数（每个 block 的前置 kernel 推进） | TileLang kernel 是**静态 shape**。只能按 `window+index_topk` 开满 + mask，**多算的槽白跑** |
| 3 | **数据是 f32，mma 要 bf16** | `q`/`kv` 都是 f32；§4 已证 bf16 cast 引入 ~2e-2 相对误差 | 数值契约（见 4） |
| 4 | **verify 的逐位门禁** | acc 门（`mean-k 2.240` 逐位 / Z_ 不变）+ `fusion-alignment-orope-linrope-mrows.md` 整篇建立在**逐位等价**上（`dsv41_kv_win_fetch` 的设计目标就是"两条分支返回同一批字节"） | online-softmax 的累积顺序 + 融合的逆 rope / fp8 发射一旦进 mma，**必然重结合** ⇒ 逐位契约破 ⇒ 需全量重验 accept 率（`swallow-accept-loss` 的教训：0.78 的 acc 损失就来自隐藏量不等价） |

### 5.2 "SGLang unified append" 为什么对不上

SGLang 的 unified append 是 **连续 KV cache + 新 token 就地 append**，flash attention 的
`T.copy` 直接吃连续块。ferrite 是 **per-row gathered ring**：

- `grid = (b*m, h)`，每个 block 一行 × 一个 head；
- 每行**自己**的 topk 列表（indexer 逐行输出）、**自己**的窗口位置（`pos_r`）；
- ring 是 **位置 mod win 的原地覆盖**——`fusion-alignment-orope-linrope-mrows.md` §1.2 已证
  稳态下"先 append 全部 m 行再整体 attention"是**静默损坏**，唯一正确形态是**append 延后 + 位置解析**。
  ⇒ 它**不是** append 形态，是"按位置反解的 gather"形态。

### 5.3 唯一的理论增益点（供将来参考，不建议现在做）

**把 64 个 head 批进 M**：`idxs` 是 **per-row 而非 per-head** 的（`sparse_attn_kernel` 里 `hh` 循环
复用同一 `irow`），且 **KV 行跨 head 共享**（`kr = kv + (bb*n+idx)*d`，**不含 `hh`**）——
所以对一行固定 `Q_all = q[row, :, :]` 是 `[h=64, d=512]`，`K_gathered` 是 `[topk, 512]`，
`QK^T` 就是 **M=64 的真 GEMM**（N=topk≈640，K=512）——**这是 tensor-core 友好的 tile**。
再算 PV = `[64, topk] × [topk, 512]` 同理。

**算术**：每步 ~20 GFLOP（40 层 × ~0.5 GFLOP/层），tensor-core floor ~0.2ms
vs 现状 2.80ms ⇒ **理论有 ~10× 计算余量**。

**但仍然不值得做**，因为：
- **现状 2.80ms 是 launch/boundary bound，不是 compute bound**（`verify-family-fusion.md` §3.4：
  880 发 × 3.2µs = **2.8ms 纯边界**，nsys 记 "纯边界 bound"）—— **TileLang 不减少 launch 数**。
  真正削它的是**已交付的 in-house 融合**（orope-mrows：每层 12→3 发，稳态可用）。
- **gather 障碍 1 不因 M=64 消失**：K 行仍要 index-driven 地从 ring/`kv_rows` 抓进 smem 才能喂 mma，
  这个 gather 本身是纯 memcpy + 分支，**没有 TMA 通路**。
- **障碍 4 是硬约束**：重写即破逐位契约，`swallow` 的 accept 损失（0.78）就是这么来的。

### 5.4 判定

> **保持自研 orope 路线。** 当前先决条件不是"能不能 TileLang 化"，而是
> **orope-mrows 的挂起修复**（`fusion-alignment-orope-linrope-mrows.md` 的 R1 臂，§4/§5 手册已就位）——
> 它把 attention 族的**launch 半**削掉 9~15 发/层，是正确且必要的下一步。
> TileLang 在这条线上**没有位置**。

---

## §6 verify 最终 TileLang 化构成表（交付 ③）

### 6.1 现状（SGLang 判决后，24.5ms 口径）

任务书给的构成：MoE 21% + 投影 20% + attention/融合 ~15% + head ~8%（ARSAFE）+ 其它。
落到 ms（× 24.5）：

| 族 | 占比 | ms | 当前形态 / 状态 |
|---|---:|---:|---|
| MoE（routed + shared） | 21% | 5.15 | MoE bf16 TileLang **已交付**（grouped 四件套；716 e4m3 修中） |
| 投影族（wq_a/wkv/wq_b/wo_b/wo_a） | 20% | 4.90 | 五形状 TileLang **已交付**（M6/M1 = 0.99–1.01，§phase2） |
| attention + 融合 | ~15% | 3.68 | `orope` 融合已交付（挂起修复中）；**本任务判：不 TileLang 化**（§5） |
| head | ~8% | 1.96 | **本任务**：TileLang 原型 M6/M1 = 1.00，M6 绝对值 **2.93×**（§3） |
| 其它（hc / indexer / norm+quant / all-reduce / engram / compressor） | 36% | 8.82 | 未涉及 |

> ⚠️ head 的 8% 是 **ARSAFE 口径**（AR_V5=0 ⇒ 三处词表切片全部失效，head 每次 launch 读
> **全量 1.323GB**，8× 放大；`verify-amortization-lesion-audit.md` §10.2）。**生产口径（v5 ON）
> head 缩到 ~1/8** ⇒ head 的真实占比远小于 8%。这一点**大幅影响 head 的 ROI**（见 §6.3）。

### 6.2 TileLang 化后的构成表（~10.4ms 预期）

| 族 | 现状 ms | TileLang 化后 ms | 削法 | 依据 |
|---|---:|---:|---|---|
| MoE（routed + shared） | 5.15 | **2.80** | TileLang bf16 grouped（28%→ 实测态） | `tilelang-moe-grouped.md`（已交付） |
| 投影族 | 4.90 | **1.50** | TileLang route A 五形状（M6/M1 = 1.00） | `tilelang-proj-phase2.md`（已交付） |
| attention + 融合 | 3.68 | **1.20** | **不 TileLang**；in-house orope-mrows（12→3 发/层）+ compressor/indexer rows | `verify-family-fusion.md` §3.3/§3.4（已交付/修复中） |
| **head** | 1.96 | **0.67** | **TileLang bf16 GEMM m=6**（903.5→308.6µs = **2.93×**，或切片后 125.2→42.2µs） | **本任务 §3** |
| 其它（hc / indexer / norm+quant / AR / engram） | 8.82 | **~4.2** | hc 链已 rows-native（§6.3）；indexer rows 已交付；余为 all-reduce 协议地板 + 边界 | `verify-family-fusion.md` §3.5/§3.7 |
| **合计** | **24.5** | **~10.4** | | |

> **口径说明**：head 的 1.96 → 0.67 是**按 ARSAFE 未切片的 2.93× 折**（保守；生产切片下 1.96 本就
> 应缩到 ~0.25ms，收益更小，见 §6.3）。"其它"的 8.82 → 4.2 是把 hc/indexer/norm 的 rows 化
> （`verify-family-fusion.md` §3.1/§3.3/§3.5/§3.7 的票面）折算进来——**含非 TileLang 项**。

### 6.3 head 的 ROI 诚实边界（本表最重要的一行）

- **ARSAFE 口径**（head 1.96ms / 8%）：TileLang 是 **2.93×** ⇒ −1.3ms，**看起来很大**。
- **生产口径**（v5 ON，词表切片 16160）：head 权重流 165MB，v1_mrows m=6 实测 125µs，
  TileLang m=6 = 42µs ⇒ **只省 0.08ms/步**；而 `verify-ms-breakdown.md` §3.2 早已判
  **"head 是唯一跑满带宽的族，切它优先级最低"**。
- ⇒ **head 的 TileLang 化"票面"几乎全部来自 ARSAFE 的 8× 放大假象**。
  **真收益小，且要付 §4 的数值债。建议：不接线为生产路径**；本原型的作用是**证明 M-in-tile
  能把 v1_mrows 的 3.27× M 放大压到 1.00×**（若将来 head 折叠路径需要提速时的现成证据）。
- **反过来，TileLang 在 head 上真正的价值在 `m>1` 的绝对值**：若把 head 与 draft 的 352 发/轮
  摊薄需求接上（§`head-act-f32vec-wo-a-cp16.md` 判"权重流被 f32 激活 2× + 行重读吃光"），
  TileLang 的 M-in-tile 恰好是"行不重读"的机械解 —— 但前提仍是 §4 的数值债。

---

## §7 未做 / 风险

| # | 项 | 说明 |
|---|---|---|
| 1 | **TileLang head 的 config 没穷举** | 本报告给的是 `bN=128 ks=8 ns=3 thr=128 bK=64`。达成带宽 3.94–4.00 TB/s ≪ 7.67 峰值 ⇒ `ks`/`bN`/`ns` 扫参**可能再抬**。已跑 96-config 定向 sweep（slice 形状，`sweep` 模式），**但该轮启动于 04:06:22、跨过了 04:07:14 的 peer `ferrite-serve --tp 8` 启动** ⇒ **数字受同机争用污染，不予采信**（唯一可读的迹象：`bN=128 ks=4 bK=128 ns=2 thr=256` 报 37.11µs ≈ 5.04 TB/s，**作为下界参考**，不作为结论）。**干净的复测需等该 serve 退出**。 |
| 2 | **未跑 e2e serve** | 只做 micro bench。§4 的数值债**必须**在 acc 门（mean-k / Z_）上还，micro bench 证明不了。 |
| 3 | **未与 `gemv_bf16_v1_mrows` 对拍** | 本 bench 用 v2 序的 `dsv41_head_gemv_bf16_mrows`（§1.1）。v1 序（生产）的绝对值可能不同，但 M6/M1 的量级结论不变。 |
| 4 | **head 的 f32 激活** | TileLang 侧强制 bf16（§4）。若将来要保 f32，需 bf16 hi/lo 双份（×2 权重流量）——本原型未做。 |
| 5 | **attention 只做只读分析** | §5 未写一行 TileLang attention 代码，也未跑 attention micro bench（依据是 nsys 表的 3.2µs/发 + 结构分析）。 |
| 6 | **`markov_head`（[vocab,256] f32）未测** | 它是 draft MTP 的头，形态与主 head 不同（f32 权重、`dspark_markov_head_kernel`）。任务书的 "×256" 疑似指它，但 `v1_mrows` 与它无关。 |
| 7 | **同机 co-tenant** | 04:07:14 起 peer 的 `ferrite-serve --tp 8` 占满 8 卡（本报告§3 的干净窗口在其之前）。未打扰该 serve（协作纪律：不动生产/半生产的 serve）。 |

---

## 附 A：复现脚本（远端 `~/tl_proj/`，本任务新增标 **+**）

| 文件 | 作用 |
|---|---|
| `head_bf16_tilelang.py` **+** | head bf16 TileLang 原型（`selftest`/`bench`/`sweep`/`all`） |
| `bench_head_old.cu` **+** → `bench_head_old` | 老 `dsv41_head_gemv_bf16_mrows` 对照（三形状 × m=1/6） |
| `dsv41_glue.cu` **+** | 拷贝的生产 TU（bench TU `#include` 它） |
| `proj_fp8_tilelang_phase2.py` 等 | 投影族产物（未改） |

本地仓库对应：`kernels/tilelang/head_bf16_tilelang.py`、`kernels/cuda/bench_head_old.cu`。

编译 / 运行：

```bash
# 老 kernel 对照（单 TU，~2–3 min，NO GPU）
nvcc -O3 --use_fast_math -std=c++17 -gencode arch=compute_103a,code=sm_103a \
     bench_head_old.cu -o bench_head_old
CUDA_VISIBLE_DEVICES=<free> ./bench_head_old 300

# TileLang 原型
CUDA_VISIBLE_DEVICES=<free> python3 head_bf16_tilelang.py all      # selftest + bench
CUDA_VISIBLE_DEVICES=<free> python3 head_bf16_tilelang.py sweep    # config 搜索
```

**未跑 `build.sh`**（任务约束）；两个 TU 都是独立 micro bench。

---

*工部 · 交付 ①②③。① 有 GPU 数字（§3/§4）；② 为只读判定（§5）；③ 为构成表（§6）。
所有未实测项在 §7 列明。*
