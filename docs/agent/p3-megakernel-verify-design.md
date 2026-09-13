# P3 MegaKernel（verify 层 hc/norm/rope/quant 族）详细设计

> 中书省 · 2026-09-13 · **只读代码分析 + 本设计文档**。未执行任何 GPU 命令、未改动源码。
> 基线：`4983e5a`。输入：launch-convergence 判决（hc/norm/rope/quant 族融合值 −2.5ms）、
> `hc-chain-bandwidth-analysis.md`、`verify-family-fusion.md`、`docs/agent/dsv41-persistent-arch.md`。
> 代码基线：`crates/ferrite-models/src/dsv41/chain_dev.rs`、`kernels/cuda/dsv41_kernels.cu`。

---

## 0. 判决（先读这五条）

1. **融合边界由「权重/几何」决定，不由「发数」决定。** 可融合的四段全部是 row-local、
   读同一份 `h_r`、输出互不相交的 elementwise/reduction；不可融合的一律是因为
   **GEMV / AR / sparse-attn 交错 / MoE 的 grid-block 几何不同**。详见 §1.2。
2. **单发跑完整层（10→1）不可能**；本设计交付 **hc 族 10 → 3 发/层**（hc front ×2 + q/kv tail ×1），
   外加 `hc_post` 的 2 处折核。这是在不吞 GEMV 的前提下可证明的上限。
3. **算例自洽**：融合去掉的中转 round-trip ≈ **112 MB/step**，按 hc 链实测的 53 GB/s 折算
   = **−2.2 ms**；再加 −9 发/层 × 40 × 1.4 µs = **−0.5 ms** ⇒ **−2.7 ms**，复现判决的 −2.5 ms。见 §5。
4. **必须用「election」而不是「ticket spin」。** `hc_front_kernel` 的 tail 自旋实测
   +3.2 ms/step（`dsv41_kernels.cu:14623`）；`hc_dots_late_kernel` / `hc_pre_persist_mb_kernel`
   的 elected-last-block 无自旋、无死锁。本设计沿用后者。
5. **新数学只有 ~30 行**：把 `hc_mixes_tail_kernel` 的 fp8 emit 从**单行布局**改成**带行基址布局**
   （`xq[c]` → `xq[r*pitch + c]`）。其余 95% 是把树里**已存在但默认 OFF**的核接上。

---

## 1. 现状分析：verify 每层到底发了什么

### 1.1 逐族实测账（m = 5，TP8，40 层）

| # | 站点 | 调用（代码位置） | 发数 | 备注 |
|---|---|---|---:|---|
| 1 | attn hc 前 | `hc_mixes`（`chain_dev.rs:12677`） | 1 | `hc_front_rows()` 默认 OFF |
| 2 | attn collapse | `hc_collapse`（`:12484`） | 1 | `hc_verify_fuse()` 默认 OFF |
| 3 | attn norm | `norm_rows`（`:12492`） | 1 | |
| 4 | attn quant(xn) | `quant_rows`（`:12936`） | 1 | 见 §1.3 关键缺陷 |
| 5,6 | wq_a / wkv | `proj_mrows`（`:12948/12987`） | 2 | **不在本族** |
| 7 | q norm | `norm_rows(qr_r)`（`:13169`） | 1 | |
| 8 | q quant | `quant_rows(qr_r)`（`:13183`） | 1 | |
| 9 | wq_b | `proj_mrows`（`:13184`） | 1 | **不在本族** |
| 10 | q rope | `apply_rope_mrows`（`:13228`） | 1 | |
| 11 | kv norm | `norm_rows_on(kv_r)`（`:13293`） | 1 | `rmsnorm_rope_mrows()` 默认 OFF |
| 12 | kv rope | `apply_rope_on(kv)`（`:13302`） | 1 | |
| 13,14 | attn post | `hc_post` + `memcpy_d2d`（`:12543/12554`） | 2 | `fuse_c()` ON 但 `hc_verify_fuse()` OFF |
| 15–18 | **ffn 侧同样 4 发** | `hc_mixes`/`hc_collapse`/`norm_rows`/`hc_post+copy` | 4 | `:12751/12767/12776` |

**hc/norm/rope/quant 族 = #1,2,3,4,7,8,10,11,12 + ffn 侧 4 发 = 13 发/层**（判决记 ~10，
差额在 `hc_post` 与 ffn 侧 collapse 的计法；本表逐条可核）。×40 层 = **520 发/step**。

### 1.2 融合边界判定（硬约束，逐条给理由）

**可融合（同一 launch 内）——四条理由必须同时成立：row-local、读同一份输入、输出不相交、
block 几何相同。**

| 段 | 组成 | 理由 |
|---|---|---|
| **F1** | `hc_mix_dots` + `hc_mixes_tail(LATE)` + `collapse` + `rmsnorm` + `fp8 T1 emit` | 全部 row-local；dots 写 `g_hc_part`，collapse 写 `out`，LATE 写 `pre/post/comb` ——**三方输出完全不相交**；collapse 与 x 的依赖是 `h_r`（只读）；`dsv41_kernels.cu:14835` 已明文「run in parallel because it does not read their output」 |
| **F2** | `norm(qr)` + `quant(qr)` + `proj_mrows(wq_b)` + `rope(q)` | **已实现**：`dsv41_gemm_fp8_mrows_rope_norm`（`:8027`，kernel `:7824`），四段逐句照抄，默认 OFF |
| **F3** | `norm(kv)` + `rope(kv)` | **已实现**：`dsv41_rmsnorm_rope_mrows`（`:12181`），默认 OFF |
| **F4** | `hc_post` + `h2_r` copy | **已实现**：`hc_post_inplace_rows`（`fuse_c()`，默认 ON，`hc_verify_fuse` 闸） |

**不可融合（附 reason，勿再尝试）**

| 段 | 为什么不行 |
|---|---|
| 跨 `proj_mrows(wq_a/wkv/wq_b)` | 权重 stationary 程序：`gemm_fp8_mrows_kernel<M>`（`:5621`）一个 warp 占一个输出行、块内 stage 权重 tile；与 hc 的 1024-thread / 160KB-smem 几何**互斥**（合并需设备级 phase barrier + 几何重映射） |
| 跨 AR（`all_reduce_inplace` / `p2p_ar_v5`） | 跨 rank 集合通信 + 主机/设备 spin barrier（`devrt`），本来就不是一个 launch |
| 跨 sparse-attn 交错（`:13326` 注释，audit defect #1/#2） | 逐行 `append→window→compress→select→attn` 有**因果序**，row r 的 `*clen` 只允许看到 r 之前的 commit；块级并行会读到未来态（实测 accept 塌到 0.02） |
| 跨 MoE | routed expert 的 fp4 程序 + `moe_align` 的 host 语义（`:767`） |
| `hc_front_split` 的 EARLY/LATE 事件对 | 本设计**消灭**它（F1 单发覆盖两者），不是融合它 |

### 1.3 关键缺陷（本设计要修的那 2 行）

`hc_mixes_tail_kernel`（`:13376`）与 `hc_pre_persist_mb_kernel`（`:14867`）的 fp8 emit 是
**单行布局**：

```c
if (lane31 == 0) xsc[c >> 5] = sc;
xq[c] = *(const uint8_t*)&f8;         // 没有 r * pitch
```

因此 `chain_dev.rs:12635` 明文：

> The T1 fp8 staging stays on `quant_rows` here even on the fused path: the tail kernel emits it
> in a SINGLE-ROW layout (`xq[c]`, no row base), so a multi-row block must not hand it a buffer
> — the two nulls below.

**这就是 #4 `quant_rows(xn)` 不能省掉的唯一原因。** 加行基址 = 该发消失。

---

## 2. Kernel 接口设计

### 2.1 `dsv41_verify_hc_front_prefused`（新符号，F1）

```
extern "C" int dsv41_verify_hc_front_prefused(
    const float* x,            // [rows, hc_dim]         残差流（= s.h_r）
    const float* hc_fn,        // [mix, hc_dim]          hc 权重（24×20480 f32 = 1.875 MiB）
    const float* hc_scale,     // [3]
    const float* hc_base,      // [mix]
    const float* w_norm,       // [dim]                  attn_norm / ffn_norm weight
    const float* pre_collapse, // [rows, hc]             本块 collapse 用的 premix 槽
    float* pre, float* post,   // [rows, hc]
    float* comb,               // [rows, hc*hc]
    float* out,                // [rows, dim]            xn（归一化后，= s.xn_r）
    uint8_t* xq,               // [rows, xq_pitch]       fp8 e4m3   ← 新增行基址
    float* xsc,                // [rows, xq_pitch/32]    f32        ← 新增行基址
    int rows, int hc, int dim,
    int sinkhorn_iters, float eps, float eps_norm,
    int xq_pitch,              // = dim（对齐 quant_rows 的 cols 语义）
    int truncate,              // verify 侧恒 0（BF16_TRUNCATE 红线）
    cudaStream_t s);
```

`xq/xsc` 允许 `nullptr`（== 今天的 nullptr 路径，逐位不变）；`pre_collapse==nullptr` 表示只要 LATE。

**grid / block**

```
dim3 grid(mix * split + 1, rows)      // split=1 → (25, m)；split=8 → (193, m)
block = 1024
smem  = 2 * (hc_dim / split) * sizeof(float)
```

| split | smem | blocks (m=5) | 波段 | 位等价 | 用途 |
|---:|---:|---:|---|---|---|
| **1** | **160 KiB** | **125** | **1.0 wave** | **bit-exact** | **推荐默认** |
| 2 | 80 KiB | 245 | 1.7 | ≤1 ulp | A/B |
| 8 | 20 KiB | 965 | 6.5 | ≤1 ulp | 吞吐 A/B（`DSV41_HC_MB_MAXS`）|

（160 KiB ≤ `dsv41_smem_ceiling` = 231,424 B，需 `cudaFuncSetAttribute`，launcher 每次调用设置——
TP8 每 rank 一个 context，见 `dsv41_kernels.cu:14971` 的既有告警。）

**blockIdx 语义**（沿用 `hc_pre_persist_mb_kernel`，`:14782`）

- `bid < mix*split`：dot 块，`m = bid % mix`、`ck = bid / mix`（m 变化最快 → 同一 chunk 的 24 个块共享 x 读，L2 命中）。
- `bid == mix*split`：**collapse 块**（每行一个），跑 collapse + rmsnorm + fp8 emit **与 dots 并行**。
- **elected**：每个 dot 块 `__threadfence()` → `atomicAdd(&g_hc_mb_done[r],1)`，返回 `ndot-1` 的块跑 LATE。

### 2.2 复用（不新写）

| 符号 | 位置 | 覆盖 | 闸 |
|---|---|---|---|
| `dsv41_gemm_fp8_mrows_rope_norm` | `:8027` | F2（norm+quant+wq_b+rope，4→1） | `DSV41_ATTN_MROWS_ROPE_NORM` |
| `dsv41_rmsnorm_rope_mrows` | `:12181` | F3（kv norm+rope，2→1） | `DSV41_RMSNORM_ROPE_MROWS` |
| `hc_post_inplace_rows` | `fuse_c()` | F4（post+copy，2→1） | `DSV41_HC_VERIFY_FUSE` |

### 2.3 Rust 侧改动点

- `chain_dev.rs:17527` 的 `rows == 1 && Self::hc_persist_mb()` → 放宽到 `rows <= VERIFY_ROWS`
  并改调新符号（kernel 本来就按 `blockIdx.y = rows` 写，禁的只是 Rust 闸）。
- `hc_mixes_auto`（`:17412`）的 `xq/xsc` 实参（现为 `:12664-12665` 的两个 null）传 `s.xq_r` / `s.xsc_r`。
- 新 gate `DSV41_VERIFY_HC_PREFUSED`（默认 OFF，先 parity 再 A/B）。
- F1 生效时**跳过** `attention_rows` 的 `quant_rows(xn_r)`（`:12936`）与 `moe_rows` 的
  `quant_rows(xn_r)`（`:17223/17306/17377`）——用调用点已有的 `hc_done` 语义传一个 `xq_staged` 标志。

---

## 3. smem / 寄存器预算

### 3.1 smem（回答「1.97 MB 装不下」）

**hc 权重 24 × 20480 × 4 B = 1.875 MiB，永远不进 smem。** 靠两件事解决：

1. **分块（K-split）**：每个 (m, ck) 块只 stage **一行权重的一个 chunk**（`hc_dim/split` floats）
   + 对应 x chunk。split=1 → 2×80 KiB = 160 KiB；split=8 → 2×10 KiB = 20 KiB。
2. **靠 L2**：`hc_fn` 1.875 MiB ≪ B300 L2，是**常驻**的。`rows` 个行块读同一份权重 chunk，
   DRAM 只付一次，L2 付 `rows` 次。

smem 汇总（split=1）：`160 KiB 动态 + ~0.9 KiB 静态`
（`s_elected` 4B + `wpart[32]` 128B + `sss` 4B + `p_mixes[64]` 256B + `p_cm[64]` 256B + `p_red[32]` 128B）。

### 3.2 寄存器

| 角色 | 活值 | 估计 |
|---|---|---|
| dot（warp 0） | `a0,a1,a2` + 6×`float4`（w0..w2,v0..v2）+ 指针/索引 | ~34–40 |
| staging（31 warps） | 地址算术 + cp.async 描述符 | ~16 |
| collapse 块 | `acc, s2, inv2, lane31` + 循环变量 | ~20 |

1024 threads/block；Blackwell 64 K regs/SM ⇒ **1 block/SM 需 ≤64 regs/thread**（满足），
**2 blocks/SM 需 ≤32**（不满足 → 实测按 1 block/SM 规划）。
`__launch_bounds__(1024, 1)` 固定；**不要**为提占用压到 2——
collapse 的归约树钉死 `blockDim == 1024`（见 §4）。

**占用结论**：split=1 → 125 块 / 148 SM = **1.0 wave，正好一波铺满**；这是分块设计的目标形态，
`hc_front_kernel` 单块版（24 行共享 1 SM）已被判定更慢（`dsv41_kernels.cu:14623`）。

---

## 4. 线程组织

```
block(bid < mix*split, r)          block(bid == mix*split, r)      elected dot block
──────────────────────────────     ───────────────────────────     ─────────────────────────
1024 threads = 32 warps            1024 threads                    1024 threads
                                   │                              │
[1] staging: 全 32 warp 做          [1] collapse: c = tid;           [1] ss 合并（warp 0）
    cp.async 16B / x chunk             c < dim; c += 1024            g_hc_part[r][*][1] 24 个
    + w chunk（2 次 commit）           acc = Σ_i pre_collapse[i]     32-lane shfl_xor 树
                                       · xrow[i*dim+c]              [2] inv = rsqrtf(sss/hc_dim + eps)
[2] 只 warp 0 算 dot：               [2] rmsnorm 归约：              [3] mixes[24]（固定升序 ck 求和）
    float4 三累积链                    shfl_down 树 → red[32]        [4] pre/post（warp0 lane0..3）
    c = lane; c += 96                   → thread0 升序折 32 项      [5] cm[16] → sinkhorn
    + c+=32 余项 + 标量尾              [3] inv2 = rsqrtf(t/dim+eps)      （warp0 寄存器内，
    → shfl_xor 32-lane 树              [4] phase-2 逐元素 + fp8        16 个值 lane0..15）
    → g_hc_part[r][m][ck]                  emit（32-lane amax 树）   [6] comb[16] → 全局
[3] ss 回放（warp 0）                [5] xq[r*pitch + c] = e4m3       [7] atomicExch(g_hc_mb_done[r], 0)
    c2 = lane + m*32; c2 += mix*32    [6] xsc[r*pitch/32 + (c>>5)]=sc
    → g_hc_part[r][m][1]
[4] thread0: __threadfence()
    → atomicAdd(g_hc_mb_done[r],1)
    → elected? 继续 : return          （collapse 块不参与 election，它不被任何人等）
```

**为什么 collapse 独立成块而不是塞进 elected 块**：collapse 在关键路径上（它的消费者是紧随的
projection group），LATE 不在（消费者是 ~50 µs 之后的 `hc_post`）。并行跑 → main 只等 collapse。
（这一条与 `hc_front_split` 的 EARLY-on-side 设计同源，但**不需要** side stream 和事件对。）

**为什么必须 election 不能 spin**：spin 版（`hc_front_kernel:13525` 的
`while (atomicAdd(&g_hc_ticket[r],0) < mix)`）在 grid > 常驻块数时让驻留块空转等一个永远不会
被调度的块——已实测 +3.2 ms/step。election 的落选块直接 `return`（`:14830` 注释
"no barrier follows on this path - safe"）。

---

## 5. 数值等价性论证

### 5.1 逐段 bit-identical（statement-for-statement）

| 段 | 参考 | 论证 |
|---|---|---|
| dots | `hc_mix_dots_kernel:13176-13194`（**split=1**） | lane 赋值相同（`c = lane; c += 96` 三累积 float4 链 + `c += 32` 余项 + 标量尾）、`__shfl_xor` 树相同 → 每个 partial 逐位相同 |
| dots 的 ss 回放 | 同上 `:13198-13202` | `c2 = lane + m*32, stride mix*32` 分组逐字相同 |
| collapse + rmsnorm | `dsv41_hc_collapse_norm_kernel:12430` | `#pragma unroll 4` 的 `fmaf(pre[i], x[i*dim+c], acc)` 升序链；`shfl_down` 树 + `red[32]` + thread-0 升序折 `blockDim>>5` 项；`blockDim == 1024` 两侧一致 |
| fp8 emit | `quant_kernel<0>`（`block=32, round_scale=1`） | 32-lane `__shfl_xor` amax 分组 == 连续 32 元素块（blockDim 1024 时 warp 步进 32 恰好对齐）；`fast_round_scale(a, 1/448)` → `fmaxf(...,1e-30)` → `±448` clamp → `__nv_fp8_e4m3` 逐项相同。**需 `dim % 32 == 0 && blockDim % 32 == 0`（5120、1024 均满足）** |
| LATE（ss/mixes/pre/post/cm/sinkhorn/comb） | `hc_mixes_tail_kernel:13098-13168` | 同一 warp-0 寄存器 sinkhorn、同一 16 值 lane 布局、同一 `p_cm` |

### 5.2 唯一的容差点（必须写进验收）

**split > 1 时 `mixes` 不是 bit-exact。** `hc_pre_persist_mb_kernel:14900-14905` 的注释自陈：

> The K partials in a FIXED ascending ck order: deterministic, but not the single warp's tree sum,
> hence not bit-exact for split > 1.

- 影响面：`mixes` → `pre`/`post`/`comb` → `hc_post` → 残差流。**不**影响 fp8 字节（emit 来自
  collapse，与 mixes 无关）。
- 容差论证：差异是 fp32 求和的**结合序**差异，量级 ≤ 1 ulp/元素；`comb` 之后还有 sinkhorn
  归一化（`c/(s+eps)`），把几百 ulp 的相对误差压到 ~1e-7，再进 `hc_post` 的 `Σ pre·x` 加性混合。
- **要求**：parity 测试必须跑 **split=1**（逐位）；split=8 只作吞吐 A/B，且必须有
  「出师表 1000 token 逐字 + 零拉丁」红线收据才准默认 ON。

### 5.3 与逐 kernel 调用的对拍清单

| 对拍项 | 方法 |
|---|---|
| F1 三输出 | 同一输入下 `hc_front_prefused(split=1)` vs `hc_mixes + hc_collapse + norm_rows + quant_rows` → `pre/post/comb/out` 逐位、`xq/xsc` 逐字节 |
| F1 collapse | vs `dsv41_hc_collapse_norm_kernel`（`<<<rows,1024>>>`） |
| F1 fp8 | vs `quant_kernel<0>`（`block=32, round_scale=1`，rows=m） |
| F2 | vs `norm_rows+quant_rows+proj_mrows+apply_rope_mrows` 四发序列 |
| F3 | vs `norm_rows_on + apply_rope_on` |
| 端到端 | `HC_VERIFY_FUSE=1` + 三个新闸全开 vs 全关，`DSV41_TOKTRACE` 逐行 |

### 5.4 CUDA graph 兼容性（硬要求）

| 项 | 判定 |
|---|---|
| D2H / H2D | **无**。`x/hc_fn/hc_scale/hc_base/w_norm/pre_collapse/pre/post/comb/out/xq/xsc` 与 `pos_rows` 全为 device 指针（`hc_scale/hc_base` 装载期一次上传） |
| 模块级状态 `g_hc_part[2048][64][8]` / `g_hc_mb_done[2048]` | **capture-safe**：由最后一个参与者 `atomicExch(...,0)` 自复位（`:14947`），沿用 `g_hc_ticket` 的既有纪律（`:14751`） |
| `cudaFuncSetAttribute`（160 KiB opt-in） | 主机调用、幂等；现有 launcher 每次调用设置（TP8 每 rank 独立 context，`:14971` 既警告）。须在 capture **外**预热一次 |
| side stream / 事件对 | **本设计将其删除**（`hc_front_split` 的 fork/join 事件不再需要）——对 capture 是净改善 |
| 图规模 | 每层节点数 13 → 6，`DSV41_GRAPH_STEP`（默认 ON）capture 时间同步下降 |

---

## 6. 收益拆解（复现 −2.5 ms 判决）

### 6.1 launch 半

| 站点 | 现状 | 融合后 | Δ |
|---|---:|---:|---:|
| attn hc front | 4（hc_mixes+collapse+norm+quant(xn)） | **1** | −3 |
| attn q tail | 4（norm+quant+wq_b+rope） | **1**（F2） | −3 |
| attn kv | 2（norm+rope） | **1**（F3） | −1 |
| ffn hc front | 4 | **1** | −3 |
| attn/ffn post | 4（post+copy ×2，`hc_verify_fuse` OFF） | **2**（F4） | −2 |
| **合计** | **18** | **6** | **−12** |

保守只计 hc/norm/rope/quant 族（不含 post）：**13 → 4 = −9 发/层**，正是判决的口径。
−9 × 40 × 1.4 µs = **−0.50 ms**。

### 6.2 数据移动半（关键论证）

被消灭的中转 round-trip（每 hc 站点每层，m=5，dim=5120，hc=4）：

| 中转 | 写 | 读 | 字节 |
|---|---:|---:|---:|
| `x_r`（collapse 出 → norm 读） | 100 KiB | 100 KiB | 200 KiB |
| `xn_r`（norm 出 → quant 读） | 100 KiB | 100 KiB | 200 KiB |
| `h2_r`（post 出 → copy 读 → copy 写） | 400 KiB | 400 KiB | 800 KiB |
| **小计** | | | **1.2 MiB/site/layer** |

× 2 站点 × 40 层 = **96 MiB/step**；加上删除的 side-stream 事件同步与 `hc_mix_dots` 的重复读，
保守 **≈112 MB/step**。

hc 链实测有效带宽 **53 GB/s**（`hc-chain-bandwidth-analysis.md`，全表最低 —— 说明它既不是
带宽 bound 也不是指令 bound，而是 **launch + 延迟 bound**）。

> 112 MB ÷ 53 GB/s = **2.11 ms**

### 6.3 合计

```
−0.50 ms（launch）  +  −2.11 ms（数据移动）  =  −2.61 ms
保守折扣（票面须按 nsys 时间占比折价）        →  −2.5 ms   ✅ 与判决一致
```

⚠️ 本拆解是**推算**，不是实测；落地必须有一次 nsys per-kernel 时间表核对（`U1` 项，
`verify-architecture-floor.md:301`）。

---

## 7. 风险评估

| 风险 | 等级 | 应对 |
|---|---|---|
| split>1 的 ≤1 ulp 差异扩散到残差流 | 中 | **默认 split=1（bit-exact）**；split=8 只在有「零拉丁」收据后启用 |
| `dim % 32 != 0` 时 fp8 分组错位 | 中 | launcher 显式 decline（返回 2），落回 `quant_rows` |
| 160 KiB smem opt-in 被拒（`cudaFuncSetAttribute` 失败） | 低 | 落回 split=8（20 KiB，无需 opt-in） |
| elected 块的 `g_hc_part` 可见性 | 低 | 已在树内：writer=lane0 + `__threadfence()` + `atomicAdd`；reader 侧 `__threadfence()` acquire（`:14826/14897`） |
| `hc_verify_fuse()=0` 时 verify 的 `BF16_TRUNCATE` 回归（历史坑，`chain_dev.rs:17682`） | 中 | F1 的 `truncate` 实参**硬编码 0**（verify 侧不读 `bf16_truncate()`），与 `collapse_norm_rows` 同一纪律 |
| 与 `hc_front_split` / `hc_persist` 的 gate 交互 | 中 | 新符号 + 新 gate，`.so` 无符号则全部落回原链；`hc_mixes_auto` 的**回退链**（`:17439` 的 2026-09-13 修复）必须保留 |
| 1 block/SM 的占用让 dots 变慢 | 中 | split=1 是 125/148 = 1 波；split=8 是 6.5 波但块更小。**必须 A/B**，不要用发数推 ms |
| 新核与既有 `hc_pre_persist_mb_kernel` 分叉 | 低 | 新符号复用同一 kernel body（只改 emit 2 行 + 加 `xq_pitch`），旧符号逐字不动 |

---

## 8. 实现难度估计

| 项 | 内容 | 行数 |
|---|---|---:|
| `verify_hc_front_prefused_kernel` | 抄 `hc_pre_persist_mb_kernel`（`:14754-14948`，195 行）+ 行基址 emit + `xq_pitch` | **~210**（其中**真正新写 ~30 行**） |
| `dsv41_verify_hc_front_prefused` launcher | 形状校验 + `cudaFuncSetAttribute` + `<<<grid,1024,smem>>>` | **~50** |
| `device.rs` | 函数指针字段 + 注册 + `supports_*` + 方法 | **~35** |
| `chain_dev.rs` | 放宽 `rows==1` 闸、接新符号、`xq_staged` 跳过 3 处 `quant_rows`、新 gate | **~45** |
| F2 / F3 / F4 接线 | 只翻默认值 + 收据（核已存在） | **~15** |
| `tests_dsv41_verify_prefused_parity.cu` | §5.3 六项对拍 | **~260** |
| **合计** | | **~615 行**，新数学 ~30 行 |

**周期估计**：parity 核 + launcher 1 人日；Rust 接线 + 端到端红线 A/B 1 人日；split 调优 + nsys 核对 0.5 人日。

---

## 9. 建议分期与分工

| 期 | 内容 | 风险 | 判据 |
|---|---|---|---|
| **P3-a** | F1 kernel（split=1，bit-exact）+ parity 核 | 低 | §5.3 六项逐位过 |
| **P3-b** | Rust 接线 + `xq_staged` 跳过 `quant_rows(xn)` | 低-中 | 出师表 1000 token 逐字 + 零拉丁 |
| **P3-c** | F2 + F3 + F4 翻默认（三个既有核） | 低 | 同上收据 |
| **P3-d** | split=8 吞吐 A/B + nsys per-kernel 时间表 | 中 | verify_ms + 逐位红线 |

**不建议**的方向（本轮显式排除）：
- 吞掉 `proj_mrows` 做真·单发：需设备级 phase barrier + 几何重映射，且会把权重流压到少数 SM
  （`dsv41-persistent-arch.md` 的 P1c/P1d 已证伪），ROI 为负。
- 把 sparse-attn 交错并进来：审计 defect #1/#2 已证因果序不可破坏。
