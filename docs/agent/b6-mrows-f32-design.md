# B6 实施方案：`dsv41_gemm_fp8_mrows_f32`（wo_b 的 m-rows f32 GEMM）

> 工部（ministry-works）· 2026-09-12 · **只读 + 设计**。未执行 GPU 命令、未改动任何源码（本文件是唯一产出）。
> 代码基线：`kernels/cuda/dsv41_kernels.cu`、`crates/ferrite-models/src/dsv41/{chain_dev.rs,dspark_dev.rs,device.rs,weights.rs,config.rs}`。
> 上位文档：`docs/agent/verify-eager-fusion-migration.md`（B 类清单、Wave 4 gate 名、节省账）、
> `docs/agent/dspark-correctness-chain.md`（draft-verify 程序审计）、`docs/agent/dspark-perf-400-plan.md`。

---

## 0. 判词（先读这四条）

1. **B6 不是「wo_b 的新核」，是整条 B 类共用的 m-rows GEMM 骨架。**
   B1–B5 全部是「把一个逐行 stage 折进 m-rows GEMM」的同类需求，它们的共同前提是
   **m-rows GEMM 必须被拆成三个可插拔相位：prologue（激活的生产）/ consume（M 条独立累加链）/ epilogue（每行写回）**。
   在 B6 之前，仓里唯一的 m-rows GEMM（`gemm_fp8_mrows_kernel`）是一个**封闭**的核：
   prologue 硬编码 fp8 解码、consume 硬编码 fp8 操作数、epilogue 硬编码 `acc + bias`。
   **B6 是第一个必须打破这三处硬编码的条目**——因为它的激活不是 fp8（没有量化子、没有 scale、没有 `s_a`/`s_af` 物化），
   它必须引入第二种**激活域**（raw f32）。引入第二种域 = 把 prologue 变成参数 = 骨架成立。
   ⇒ B6 的交付物**同时**是「一个 wo_b 核」和「B1–B5 的地基」。

2. **B6 是唯一一个「数值上更准、而非位等价」的 B 类条目**——这是设计上的特性不是缺陷，
   而且它正好把 verify 的 wo_b 对齐到 **EAGER 已经在跑的那条程序**上（`dsv41_gemm_fp8_mx_f32`，`DSV41_WOB_F32` 默认 ON）。
   所以 B6 的 parity 判据**不是**「和旧的 `quant_fp8 + proj_mrows` 逐位相同」（做不到，也不需要），而是
   **「m 行核的第 r 行 == M=1 f32 GEMV 的第 r 行，逐位相同」**——即 `verify row r == EAGER row r`。

3. **收益的第一性来源是 launch 数，不是带宽**：verify 的 wo_b 现在是
   `m × quant_fp8` + `1 × proj_mrows` = `m+1` 发/层；B6 后 **1 发/层**。
   40 层 × `m`（m ∈ [1, VERIFY_ROWS=6]，生产常见 5）⇒ **−200 ~ −240 发/步**，
   按 `verify-ms-breakdown` 的 3.3µs（执行半）/~6.2µs（提交+执行）⇒ **−0.66 ~ −1.5ms**，与判词的 −0.6~1.0ms 一致。
   权重侧 DRAM 流量**不变**（`proj_mrows` 本来就已经是 weight-stationary，wo_b 权重全程只读一遍）。

4. **风险排序：程序一致性 > 数值 > 收益高估。**
   B6 本身没有新 K 序（它复刻 M=1 的 K 序），没有 grid barrier，没有跨块依赖 ⇒ 风险显著低于 B1/B2/B5。
   真正的坑是**接线类**：`.so` 不同源、gate 误接、"两臂都测旧路径"（本仓 #1 陷阱）、
   以及「跳过 `quant_fp8` 后 `xq_r` 留下陈旧字节被别的读者读到」。

---

## 1. 现状：wo_b 的三条路径（代码级）

wo_b = attention 的输出投影，**RowParallel**（k 维被 TP 切分，各 rank 出部分和，随后 all-reduce）。
它是「分组低秩输出投影」的第二级：`wo_a`（block-diagonal，groups 个 group 各投自己的 heads）→ `wo`（[rows, groups·o_lora] f32）→ `wo_b` → `o`（[rows, dim] f32）。

### 1.1 EAGER 主链（`chain_dev.rs::attention()`，12008 起）

| 项 | 值 | 锚点 |
|---|---|---|
| 调用 | `gemm_fp8_mx_f32(wo, wo_b, wo_b_scale, null, o, dim, ol_local)` | `:13454-13464` |
| gate | `DSV41_WOB_F32`（**默认 ON**）+ `.so` 有符号 | `Self::wob_f32()` / `supports_gemm_fp8_f32()` |
| m | 1 | — |
| 形状（TP8） | n = dim = 5120，k = ol_local = 1024 | — |
| 兜底 | `B1`(wo_q 融合) → `quant1(s.wo)` + `gemm_fp8_mx` / `gemm_fp8_mx_ar` | `:13465-13500` |

**要点**：EAGER 已经是 **fp8 权重 × raw f32 激活** 的 M=1 GEMV（`gemm_fp8_gemv_kernel` 的 `gf.a_f32` 分支）。
它的位等价契约写在 kernel 头：`s_af IS the value`（`:4702-4707` 的 `s_af[i] = a_f32[i]` 是纯拷贝），
consume 里 `a32=1` 读 `s_af[j]`、`a32=0` 读 `a_f32[j]`，**两者按构造位相同**（`:4882-4890`）。

### 1.2 verify（`chain_dev.rs::attention_rows()`，9017 起）

| 项 | 值 | 锚点 |
|---|---|---|
| 前置 | `let mrows = supports_gemm_fp8_mrows() && !swapab() && m <= VERIFY_ROWS` | `:9047` |
| 激活 | **m × `quant_fp8` 逐行**（`rows=1, cols=ol_local, block=32, round_scale=true`）写进 `xq_r`/`xsc_r` | `:9806-9816` |
| GEMM | `proj_mrows(wo_b, wo_b_scale, wo_out_r, m, dim, ol_local, dim)` | `:9817-9825` |
| 兜底 | 逐行 `quant1(wo_r + r*ol_total, ol_local)` + `gemm_fp8_mx_or_swap(..., dim, ol_local)` | `:9829-9846` |
| m | ≤ `VERIFY_ROWS` = **6** | `:84` |
| 形状（TP8） | n = 5120，k = ol_local = 1024，out_stride = dim | — |

⇒ **每层 `m+1` 发**（m=5 时 6 发/层 = 240 发/步；SWALLOW 的 m=6 时 280 发/步）。

**行距已修的历史坑**（必须继承，不能在 B6 里复现）：`quant_rows`/`quant_fp8` 的**源行距从 `cols` 推**，
而 `wo_r` 的真实 pitch 是 `ol_total`（TP8 下是 `ol_local` 的 8 倍）——所以 verify 已经在**逐行打包**
（`:9802-9816` 的注释记录了 "row 0 恒对、r≥1 全错" 的 diff-probe 指纹来自这里）。
B6 用 **显式 `a_stride` 参数**（见 §3.2）把这件事变成 kernel 的显式契约，而不是调用方的隐式约定。

### 1.3 DSpark draft（`dspark_dev.rs::draft_attn_out()`）

| 项 | 值 | 锚点 |
|---|---|---|
| bf16 域 | `DSV41_DRAFT_BF16_DOMAIN`（默认 OFF）：对 `wo` 做 in-place bf16 round-trip | `:2166-2169` |
| 激活 | `quant1(wo, bs*ol_total)` | `:2170` |
| 优先 | `DSV41_ATTN_PROJ_ALIGN`（默认 OFF）+ `proj_attn_mrows` → `gemm_fp8_mrows`（**m=1 GEMV 程序**） | `:2176-2185` |
| 兜底 | `gemm_fp8_mx(xq, xsc, wo_b, wo_b_scale, null, o, bs, dim, ol_total)`（**m=bs → 16 行 TILE 程序**） | `:2187-2197` |
| m | `bs` = `DSPARK_DRAFTS` = **5** | `dspark_dev.rs:65` |
| 形状 | n = 5120，k = **ol_total = 8192**（draft 的 wo_b 是 `Shard::Replicated`），out_stride = dim | `weights.rs:330` |

**这就是 draft-verify audit 的第 2 号不匹配**：draft 的四个 attention 投影走 `gemm_fp8_mx`（TILE，结构性不同的求和序），
verify 走 `proj_mrows`（m=1 GEMV）——同 head 级严重度。`ATTN_PROJ_ALIGN` 是**已有的 A/B 臂**（默认 OFF）。

### 1.4 dtype / 形状确认（任务里问的「或 bf16？」）

| 张量 | dtype | 形状（全局） | 局部（TP8） | 证据 |
|---|---|---|---|---|
| wo_b **权重** | **fp8 e4m3** | [5120, 8192] | [5120, 1024]（`Shard::Cols` 切 k） | `weights.rs:128`、`:701`、`:781` |
| wo_b **权重 scale** | **ue8m0**，32×32 block | [160, 256] | [160, 32] | `weights.rs:129` `load.rs:844` |
| wo_b **激活** | **f32**（`wo_r` / draft `wo`） | [m, 8192] | [m, 1024] | `chain_dev.rs:415,9808` `dspark_dev.rs:769` |
| wo_b **输出** | **f32** | [m, 5120] | [m, 5120] | `wo_out_r`、draft `o` |

**结论**：**不是 f32 权重，也不是 bf16 权重**。B6 的正确命名是「**fp8 权重 × f32 激活**」——
所以符号名 `dsv41_gemm_fp8_mrows_f32`（`fp8` 指权重格式，`f32` 指激活/输出格式）是准确的，与 `dsv41_gemm_fp8_mx_f32` 同一命名法。
DSV41-Flash 配置：`dim=5120, o_lora_rank=1024, o_groups=8, ol_total=8192`（`config.rs:1-14`）。

---

## 2. 可复用的既有 m-rows 模式（对照物）

`gemm_fp8_mrows_kernel<M>` / `dsv41_gemm_fp8_mrows`（`dsv41_kernels.cu:5208` / `:5373`）——B6 的骨架来源：

```
一个 block 拥有 nwarps 个输出行（权重行 cp.async16 暂存一次）
  → 每 (输出行, 激活行 r) 一条独立的串行 acc[r] 累加链
  → kb 升序、j = kb*32 + lane、shfl_xor 16..1 原样
  → lane 0 写 out[r*out_stride + row]
```
- **C1–C6 位等价论证**（同文件 `:5145-5192`）是 B6 必须照抄的模板：同 K walk / 同操作数同字节 / 同规约树 /
  **无跨行重组** / 无 K-split 无 smem fold / **codegen pin**（单一串行链 + `#pragma unroll 32` 源形，禁止 split-accumulator 优化）。
- **launcher 契约**：`0 = launched / 2 = DECLINED（永不 1，1 与 cudaErrorInvalidValue 冲突）`；decline 表是「形状/mode 属性」；调用方保留逐行循环。
- **smem 属性阶梯**：`smem > 48KB` 时**每一个 M 特化都要单独 `cudaFuncSetAttribute`**（`:5425-5451` 两次记录了漏设 `<M>` 导致 m=5 `cudaErrorInvalidValue` 的事故）。
- **几何不进入 parity**：`nwarps` 只决定「多少行共享一次 staging」，行是独立的 ⇒ `dsv41_mrows_warps_for(n)`（`:3871`）可以自由换臂。

**B6 相对它的三处差异**（这就是「第二种激活域」的全部内容）：

| 维度 | fp8 mrows（现状） | **B6（f32）** |
|---|---|---|
| 激活暂存 | `s_a` = M 行 × k **字节**（fp8） | **无**——raw f32 直接 global 读 |
| 激活 scale | `s_as` = M 行 × k/32 f32 | **无**（f32 路径没有量化 scale） |
| 操作数 | `s_lut[s_a[r*k+j]] * s_as[r*nb_k + (j>>5)]` | `a_f32[r*a_stride + j]`（**恒等**，即 M=1 核 a32=0 臂的表达式） |
| smem | `nwarps*k + 256f + m*(k/32)f + m*k` | `nwarps*k + 256f` |
| 位等价基准 | `dsv41_gemm_fp8_mx`（m=1） | `dsv41_gemm_fp8_mx_f32`（m=1） |

**为什么「无激活暂存」是正确的（不是偷懒）**：fp8 核必须暂存 `s_a`，是因为它要做 **`s_af` 物化**
（LUT 解码 + scale 乘，1.55µs/call 的实测成本，`dsv41_a32_mat4` 头注释）。而 **f32 路径的「物化」是恒等函数**
（`s_af[i] = a_f32[i]`，`:4707`）——所以那个 smem 槽位在这条域上**没有任何东西可摊薄**：
- 唯一数据量 = `m*k*4` = 24KB（m=6, k=1024），全部 L2 常驻（640 个 block 读同一份，DRAM 只读一遍）；
- 暂存到 smem 反而多一次 smem 写 + 一个 barrier，并且把 draft 形状（k=8192）直接撑到 160KB 爆 smem。
⇒ **直接 global 读 = 更简单 + 更省 + 两个形状都能跑**。

---

## 3. B6 kernel 设计

### 3.1 命名与 ABI（照上位文档的 Wave 4 清单，不另起名）

```cpp
// dsv41_kernels.cu —— 与 dsv41_gemm_fp8_mrows 同族、同 ABI 形状
extern "C" int dsv41_gemm_fp8_mrows_f32(
    const float*   a_f32,      // [m, a_stride] f32 激活，行 r 在 + r*a_stride
    const uint8_t* w,          // [n, k] fp8 e4m3 权重
    const uint8_t* w_scale,    // [n/32, k/32] ue8m0（32×32 block）
    const float*   bias,       // [n] 或 null（两个调用点都传 null）
    float*         out,        // [m, out_stride] f32，行 r 的元素 row 在 + r*out_stride + row
    int m, int n, int k,
    int a_stride,              // ★ 新增：激活行距（字节数/4 个 f32 元素）
    int out_stride,
    cudaStream_t s);
// 返回 0 = launched；2 = DECLINED（调用方保留逐行/旧路径，永不返回 1）
```

**为什么必须有 `a_stride`**（这是 B6 相对 fp8 兄弟的**唯一 ABI 扩展**）：
verify 的 `wo_r` 行距是 `ol_total`（`chain_dev.rs:9808` 用的是 `r*ol_total`）而 k 只有 `ol_local`；
draft 的 `wo` 行距恰好 = k = `ol_total`。fp8 兄弟核把这层"源行距 vs cols"的错配**推给了调用方**
（于是有了 verify-value-hunt 那个确定性根因 F1/F2）。B6 把它升成显式参数：

- 传 `a_stride = ol_total` ⇒ verify 直接读 `wo_r`，**不需要逐行打包、不需要 `xq_r` 的 m 次 quant**；
- 传 `a_stride = k` ⇒ draft / 紧凑缓冲。

这是**接线层的净胜**：它一次性消灭了「quant_rows / quant_fp8 的源行距 = cols」这一整类 bug 的复发面
（`spec-invariants.md` 族 B 的 B2 条、`dspark-correctness-chain.md` 的根因 F1/F2）。

### 3.2 设备核（模板 + 三个相位槽）

```cpp
// 相位槽 ①  PROLOGUE：生产本 block 的激活操作数
//   现状 fp8 核：LUT 解码 + scale 乘（写 s_a / s_af）
//   B6：      恒等（操作数就是 a_f32 本身）—— 见 §2 的论证
// 相位槽 ②  CONSUME：M 条独立串行链，kb 升序，j = kb*32 + lane（C1/C4/C6 原样）
// 相位槽 ③  EPILOGUE：每行 lane-0 写回；B6 = acc + bias；B1 在此插 rope；B5 在此插 route 选举
template <int M>
__global__ void __launch_bounds__(256)
gemm_fp8_mrows_f32_kernel(const float* __restrict__ a_f32,
                          const uint8_t* __restrict__ w, const uint8_t* __restrict__ w_scale,
                          const float* __restrict__ bias, float* __restrict__ out,
                          int n, int k, int a_stride, int out_stride) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    const int nb_k = k >> 5;
    extern __shared__ uint8_t smem[];
    uint8_t* s_w   = smem;                                     // [nwarps][k]
    float*   s_lut = reinterpret_cast<float*>(s_w + (size_t)nwarps * (size_t)k);  // [256]
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_lut[i] = e4m3_to_f((uint8_t)i);

    const int row = blockIdx.x * nwarps + warp;
    const bool active = row < n;
    if (active) {                                              // cp.async16，与 fp8 核同规则
        const uint8_t* wr = w + (size_t)row * (size_t)k;
        uint8_t* row_s = s_w + (size_t)warp * (size_t)k;
        const int n16 = k >> 4;
        for (int i = lane; i < n16; i += 32) dsv41_cp_async16(row_s + (i << 4), wr + (i << 4));
        for (int i = (n16 << 4) + lane; i < k; i += 32) row_s[i] = wr[i];
        dsv41_cp_commit();
    }
    __syncthreads();
    if (active) {
        dsv41_cp_wait_all(); __syncwarp();
        const uint8_t* __restrict__ wsr = w_scale + (size_t)(row >> 5) * (size_t)nb_k;
        float acc[M];
        #pragma unroll
        for (int r = 0; r < M; ++r) acc[r] = 0.f;
        #pragma unroll 32
        for (int kb = 0; kb < nb_k; ++kb) {
            const float sb = ue8m0_to_f(wsr[kb]);
            const int j = kb * 32 + lane;
            const float wv = s_lut[s_w[(size_t)warp * (size_t)k + j]] * sb;   // C4：一次解码，M 行复用
            float av[M];
            #pragma unroll
            for (int r = 0; r < M; ++r) av[r] = a_f32[(size_t)r * (size_t)a_stride + j];  // 槽 ① 恒等
            #pragma unroll
            for (int r = 0; r < M; ++r) acc[r] += av[r] * wv;                 // C6：单一串行链
        }
        #pragma unroll
        for (int r = 0; r < M; ++r) {                                        // C3：规约树原样
            float a_r = acc[r];
            for (int off = 16; off > 0; off >>= 1) a_r += __shfl_xor_sync(0xFFFFFFFFu, a_r, off);
            if (lane == 0) out[(size_t)r * (size_t)out_stride + row] =       // 槽 ③
                a_r + (bias != nullptr ? bias[row] : 0.f);
        }
    }
}
```

### 3.3 grid / block / smem

| 项 | 取值 | 依据 |
|---|---|---|
| `nwarps` | `dsv41_mrows_warps_for(n)`（复用**同一个** helper，`:3871`） | 与 fp8 兄弟核同语义；n=5120 ≥ 2048 ⇒ 8 |
| `block` | `nwarps * 32` = **256**（`__launch_bounds__(256)`，与 fp8 核同） | `:5209` |
| `grid` | `dim3(ceil(n / nwarps))` = ceil(5120/8) = **640** | 640 block × 148 SM ≈ 4.3 波，与 M=1 f32 GEMV 的 grid **完全一致** |
| `smem` | `nwarps*k + 256*4` = 8KB+1KB = **9216 B**（verify）/ 65KB+1KB = **66560 B**（draft） | verify 形状不需要 opt-in；draft 形状走 `> 48KB` 的属性阶梯 |
| M 特化 | 1..=8 全部 `cudaFuncSetAttribute`（照抄 `FERRITE_SET_MROWS_SMEM` 宏） | `:5435-5451` 的事故记录 |
| 对齐要求 | `k % 32 == 0`（cp.async16 需要 `row*k` 16B 对齐，由 k%32 ⇒ k%16 保证）；`a_stride % 4 == 0`（f32 行首 4B 对齐） | `:5265-5271` |

### 3.4 数值域（数值等价口径）

| 环节 | 表达式 | 来源 |
|---|---|---|
| 激活 | `av[r] = a_f32[r*a_stride + j]`（**raw f32，无量化/无 scale**） | = M=1 核 a32=0 的 `a_f32[j]`，= a32=1 的 `s_af[j]`（纯拷贝） |
| 权重解码 | `wv = s_lut[s_w[warp*k + j]] * sb`，`sb = ue8m0_to_f(wsr[kb])` | `e4m3_to_f` / `ue8m0_to_f` 与 M=1 核同一个 device helper |
| K walk | `kb` 升序 0..nb_k−1，`j = kb*32 + lane`（**保序形**，非 vectorised/scalar 臂） | C1（`g_gemv_fp8_mode < 3` 时 launcher decline） |
| 累加 | 每 `(输出行, r)` 一条 `acc[r] += av[r] * wv` 串行链，`#pragma unroll 32` | C4/C6（**禁止** split-accumulator / 多路展开的重结合） |
| 规约 | `shfl_xor` off = 16,8,4,2,1 原样，每 `(warp, r)` 跑一次 | C3 |
| 跨行 | **无**任何跨 r 的求和 | C4 |
| 写回 | `out[r*out_stride + row] = acc[r] + bias[row]`；lane 0 | 与 M=1 核 `:4922-4929` 同形 |
| 编译期 | 同一 TU（`dsv41_kernels.cu`）、同一 `--use_fast_math`、`-O3` | 同源 codegen（build.sh） |

**位等价结论**：`B6 的第 r 行 ≡ dsv41_gemm_fp8_mx_f32(row r) ≡ EAGER 主链 wo_b 的第 r 行`（逐位）。
**非位等价**：B6 ≠ 旧的 `(m × quant_fp8 + proj_mrows)` —— 它**跳过 fp8 量化→反量化的往返**，
激活不带 4-bit 尾数损失，行部分和**略更准**；下游是 `wo_out_r → AR 求和 → hc_post`，更紧的值是正确方向
（与 `dsv41_gemm_fp8_mx_f32` 的头注释 `:6048-6053` 同一句话）。

### 3.5 decline 表（返回 2，永不 1）

| 条件 | 理由 |
|---|---|
| `m <= 0 \|\| m > 8` | 模板特化范围（调用方保留逐行） |
| `n <= 0 \|\| k <= 0 \|\| (k & 31)` | kb walk / LUT 的 32 块前提 |
| `a_stride < k` | 行距小于有效宽度 = 静默错读（**显式拒绝，不猜**） |
| `out_stride < n` | 同上 |
| `a_f32 == nullptr \|\| w == nullptr \|\| w_scale == nullptr \|\| out == nullptr` | 空指针 |
| `g_gemv_fp8_mode < 3` | M=1 参考核在 mode 0/1 走重排臂，不是同一程序 |
| `DSV41_NO_GEMV_FP8` 已设 | M=1 参考核会走 TILE MMA，是完全不同的表达式 |
| **不** decline `g_gemv_a32` | f32 路径的 a32=1 与 a32=0 **按构造位相同**（`:4882-4890`），无需分臂——这是 B6 比 fp8 兄弟**更简单**的一点 |
| **不** decline `DSV41_GEMV_A32_STAGED` / `_CPASYNC` / `_ACT_CPASYNC` | 纯 staging 机制，不动值 |

---

## 4. 为什么 B6 是 B 类的公共祖先 + B1–B5 的复用路径

### 4.1 B6 建立的四样共享资产

| # | 资产 | B6 的具体形态 | 谁继承 |
|---|---|---|---|
| **A1** | **行谱约定** | `m ∈ 1..=8` 的模板派发；`rows = m` 的 launcher 语义；`out[r*out_stride + row]` 的行谱写回；显式 `a_stride` | B1–B5 全部（含 B4/B5 这两个**非 fp8 GEMM** 的条目——它们只需要这套 launcher/行谱约定） |
| **A2** | **激活域参数化（prologue 槽）** | 新增 `ActF32`（恒等）形态，与既有 `ActFp8`（LUT+scale）并列 | **B2 直接继承**（它的 prologue 就是 `gemm_fp8_mx_rope_norm` 对 raw f32 的 `qr_raw` 做 rmsnorm——同一个"raw f32 行"域）；B1/B3 沿用 fp8 域 |
| **A3** | **epilogue 槽** | `acc + bias` 的每行 lane-0 写回被显式标记为槽位 | **B1（rope 尾）**：EAGER 的 `gemm_fp8_mx_rope` 尾相位（`s_rows[warp] = v` + 配对 warp 旋转）已经是 M=1 的成熟形态，B1 = 「M=1 的 rope epilogue × B6 的行谱」；**B5（route 选举）**也落在这个槽（但需要跨块通道，见 §4.3） |
| **A4** | **parity 契约 + 验收脚手架** | `tests_dsv41_gemm_mrows_f32.cu`（raw-u32 memcmp、NaN 哨兵全覆盖、decline 表、gate 不变性） | B 类每个新核照抄这套三臂验收（`verify-eager-fusion-migration.md` R3 的硬要求） |

### 4.2 逐条复用路径（诚实版：哪些真继承、哪些只借约定）

| 条目 | 新核 | B6 给的 | 还需要 B6 **之外的**什么 |
|---|---|---|---|
| **B1** `mrows_rope`（wq_b+rope） | `dsv41_gemm_fp8_mrows_rope` | A1 + A3 + A4 | 激活域仍是 **fp8**（沿用现成 `s_a`/`s_as` 槽）；epilogue 加 `pos_rows`/`rope_cos`/`rope_sin`/`row_pitch` 尾参——**是 B6 定义的 epilogue 槽的第一个消费者，不是 B6 本体** |
| **B2** `mrows_norm_rope` | `dsv41_gemm_fp8_mrows_norm_rope` | **A2 直接继承** + A1 + A3 + A4 | prologue 槽换成「rmsnorm(raw f32 row) → fp8 编码」，逐字复刻 `rmsnorm_q_kernel`；**依赖 B1** |
| **B3** `mrows2`（wq_a+wkv 同激活） | `dsv41_gemm_fp8_mrows2` | A1 + A2(ActFp8) + A4 | **weight-family 槽**（第二族的 n2/k2/out2 寻址 + `mx2` 契约的 m 行推广）——B6 是单族，没给这个 |
| **B4** `rmsnorm_rope_mrows` | `dsv41_rmsnorm_rope_mrows` | A1 + A4 | **它不是 GEMM**（纯逐行 elementwise 融合），不继承 A2/A3；只继承行谱 launcher 约定与验收脚手架 |
| **B5** `mrows_route`（gate+route） | `ferrite_gemv_bf16_v2_mrows_route` | A1 + A4 | 权重是 **bf16**（`gemv_bf16_nt_kernel` 家族，不是 fp8 家族）；route 选举需要**跨块通道**（last-block 选举），B6 骨架里没有。B5 的真正父本是已存在的 `gemv_bf16_v2_mrows`（`DSV41_GATE_MROWS`） |

**一句话**：**B6 是「骨架」的公共祖先，不是「核体」的公共祖先**——
它把 m-rows GEMM 从「一个封闭的 fp8 核」变成「prologue / consume / epilogue 三相位的模板」，
并用自己的实施证明了三件事：**第二种激活域可行（A2）**、**epilogue 可以外挂（A3）**、**行谱/parity 契约可以复用（A1/A4）**。
B1 是 A3 的第一个消费者、B2 是 A2 的直接继承者、B3/B5 需要 B6 骨架**之外的**新槽、B4 只借约定。
**这就是把 B6 排在整个 B 类之前做的理由**：它是唯一一个「必须先把骨架立起来才能写」的条目。

### 4.3 骨架的扩展点（B6 实施时就要留好的三个空位）

1. **epilogue 槽位签名**：即便 B6 只写 `acc + bias`，核内的写回也要**单独成一小段**并显式标注槽位，
   外加一条「rope 相位要在这里（`s_rows[warp] = v` → 配对 warp 旋转）」的注释——直接映射 EAGER 的 `:4948-4959`。
   **不要**把 bias 写进 `acc` 的累加链尾（会污染 C6 的 codegen pin）。
2. **prologue 槽位注释**：写清「f32 路径 = 恒等；fp8 路径 = LUT+scale；B2 的 rmsnorm 是第三种形态」，
   并把 `s_a`/`s_as`/`s_af` 在 fp8 兄弟核里的位置与大小记在注释里（跨形态移植是 B1/B2/B3 的实际工作量）。
3. **`a_stride` 的语义**：把它写成 kernel 的**显式契约**（不是调用方的隐式约定），
   并在 launcher 的 decline 表里加 `a_stride >= k`——这样 B3 的第二族 `a_stride` 有现成的先例可扩。

---

## 5. 接线（gate / 调用点 / 优先级序）

### 5.1 gate

**`DSV41_VERIFY_WOB_MROWS_F32`，默认 OFF**（照上位文档 Wave 4 的 gate 名，不另起名）。
一次读取 + `OnceLock` 缓存（这一段在 CUDA graph capture 内，40×/步）。

**一个 gate 同时管两个调用点（verify + draft）是有意的**：B6 的交付物之一就是
「draft 和 verify 的 wo_b 走同一条程序」，两个点必须一起翻——否则 A/B 会得到一个"半对齐"的中间态，
正是 `dspark_dev.rs:2171-2175` 那段注释担心的 "half-aligned attention chain"。
（若 A/B 需要分离，再加 `DSV41_DRAFT_WOB_MROWS_F32` 作为第二臂；**默认不引入**。）

### 5.2 verify 调用点（`chain_dev.rs::attention_rows`，`:9801` 附近）

当前优先级：`mrows → 逐行`。B6 插到**最前**：

```rust
let took_wob = if Self::wob_mrows_f32() && self.dev.supports_gemm_fp8_mrows_f32()
             && mrows                      // 复用同一个 m<=VERIFY_ROWS / 无 swapab 的前提
{
    // ★ 不再有 for r in 0..m { quant_fp8(...) } 的循环（−m 发/层）
    // ★ 也不再需要逐行打包：a_stride 把 wo_r 的真实行距 ol_total 交给 kernel
    self.dev.gemm_fp8_mrows_f32(
        self.s.wo_r.ptr as *const f32,
        ld.wo_b.as_ref().unwrap().as_u8(),
        ld.wo_b_scale.as_ref().unwrap().as_u8(),
        std::ptr::null(),
        self.s.wo_out_r.ptr as *mut f32,
        m as i32, dim as i32, ol_local as i32,
        ol_total as i32,            // a_stride = wo_r 的行距
        dim as i32,                 // out_stride
    )?
} else if mrows { /* 现状：quant_fp8 循环 + proj_mrows */ }
else { /* 现状：逐行 quant1 + gemm_fp8_mx_or_swap */ };
```

**必须一起核对的"陈旧读者"检查（接线红线）**：现在的 `mrows` 臂会把 wo_b 的 fp8 激活写进 `xq_r`/`xsc_r`，
B6 臂不再写。**已核**：`attention_rows`（9017–~10100）内 `xq_r` 的最后一次使用就是 `:9809` 那次 quant，
之后到本函数结束没有 `xq_r` 读者；层内后续的 `moe_rows` 里的共享专家（`:11637`）**自己先 `quant_rows` 再读**
（`:11634-11640`），不依赖 wo_b 留下的字节。⇒ 跳过是安全的。
**实现时必须用同样的方式再核一遍**（`grep -n "xq_r" backup` + 确认读者顺序），不要凭这份文档。

### 5.3 draft 调用点（`dspark_dev.rs::draft_attn_out`，`:2166` 附近）

当前优先级：`bf16 round-trip → quant1 → (ATTN_PROJ_ALIGN ? proj_attn_mrows : gemm_fp8_mx)`。B6 插在 `quant1` **之前**：

```rust
if draft_bf16_domain() { /* 保留：in-place bf16 round-trip 仍要跑（它改 wo 的 f32 位，f32 直读正好读到它） */ }
let took = if Self::wob_mrows_f32() && self.dev.supports_gemm_fp8_mrows_f32() {
    // ★ 不调 quant1(wo)（−1 发/块 ×3 块/步）
    self.dev.gemm_fp8_mrows_f32(
        self.wo.ptr as *const f32, wo_b.as_u8(), wo_b_s.as_u8(), std::ptr::null(),
        self.o.ptr as *mut f32,
        bs as i32, dim as i32, ol_total as i32,
        ol_total as i32,      // draft 的 wo 是紧凑 [bs, ol_total]
        dim as i32,
    )?
} else if /* 现状 */
```

**形状核对**：draft `n=5120, k=8192, m=5` ⇒ `nwarps=8` ⇒ smem 66560 B ⇒ **走 >48KB 属性阶梯**
（`FERRITE_SET_MROWS_SMEM` 全套 M；这是 fp8 兄弟核已踩过的坑）。
`self.o` 的分配是 `bs*nh*hd`（`dspark_dev.rs:768`，远远大于 `bs*dim`），`[bs, 5120]` 的行谱写回落在分配内，与现状一致。

**draft 的收益是"一致性"而非 launch 数**：省 1 发/块 × 3 块/步 = 3 发/步（可忽略）；
它的价值是**把 draft 的 wo_b 从 TILE 程序搬到与 verify/EAGER 同一条 m-rows f32 程序上**
（`ATTN_PROJ_ALIGN` 的默认 OFF 臂因此也不再需要为 wo_b 兜底）。

### 5.4 EAGER 调用点（`chain_dev.rs::attention()`，`:13454`）——**不改**

EAGER 的 wo_b 已经跑 `dsv41_gemm_fp8_mx_f32`（m=1，WOB_F32 默认 ON）。
**B6 的 parity 判据正是「m 行核的第 r 行 ≡ 这条 M=1 路径的第 r 行」**，所以 EAGER 无需改动即可达成
「EAGER / verify / draft 三者在 wo_b 上同程序」。

**可选的后续（不在本次范围）**：把 EAGER 的 m=1 也切到 `dsv41_gemm_fp8_mrows_f32(m=1)`
（数值上是 no-op：同一程序、同一 grid；但会让代码路径字面一致）。这要单独 A/B，**默认不做**。

### 5.5 三条路径对齐后的效果

```
                wo_b 输入          程序                        状态
EAGER   :  raw f32 s.wo        gemm_fp8_mx_f32 (m=1)          已有（WOB_F32 ON）
verify  :  raw f32 s.wo_r      gemm_fp8_mrows_f32 (m≤6)       ★ B6
draft   :  raw f32 s.wo        gemm_fp8_mrows_f32 (m=5)       ★ B6
          ↑ 同一个 f32 域、同一条 m=1 GEMV 求和序、同一套 LUT/ue8m0 解码
```

---

## 6. 验收（parity suite）

新文件 `kernels/cuda/tests_dsv41_gemm_mrows_f32.cu`（照 `tests_dsv41_gemm_mrows.cu` 的结构，`#include "dsv41_kernels.cu"`）：

| # | 臂 | 判据 |
|---|---|---|
| 1 | **BIT-IDENTITY** | m 行调用 vs m 次 `dsv41_gemm_fp8_mx_f32`（同一 `a_f32`、同一 `w`/`w_scale`），**raw u32 memcmp 整块**（NaN 哨兵 ⇒ 未写与写小值可区分；`out_stride > n` 时 gap 两侧都必须是哨兵） |
| 2 | **行覆盖** | 每个 `(row, r)` 都被写（NaN 哨兵零残留）——防"只算了 row 0" |
| 3 | **decline 表** | `m ∉ 1..=8`、`k%32`、`a_stride<k`、`out_stride<n`、null、`mode<3`、`NO_GEMV_FP8` ⇒ 全部返回 **2**（不是 1，不是真 launch 错） |
| 4 | **形状矩阵** | verify `(m=6, n=5120, k=1024, a_stride=8192, out_stride=5120)`、draft `(m=5, n=5120, k=8192, a_stride=8192, out_stride=5120)`、小形状 `(m=1..8, n=32/144, k=64/512)`、`out_stride>n`、`a_stride>k` |
| 5 | **gate 不变性** | `DSV41_GEMV_A32=0/1` 两臂都必须等于**同一个** M=1 参考（证明 f32 路径两臂位相同）；`DSV41_MROWS_SMALL_N_ADAPTIVE=0/1` 不变（几何不进 parity） |
| 6 | **诊断（不做 pass/fail）** | 打印 B6 vs 旧 `(quant_fp8 + dsv41_gemm_fp8_mrows)` 的最大相对差——**只记录**，用于量化"更准"的方向与幅度（R3 的「新程序不能声称位等价」） |

**端到端验收**（由主 agent 按仓库纪律执行，工部不跑 e2e）：
1. 双产物重编（`.cu` 变了必须先 `build.sh 103a`，再 `cargo build --release`），记 `.so` md5；
2. `DSV41_VERIFY_WOB_MROWS_F32=0/1` **同会话背靠背** A/B（`scripts/dsv41_serve_ab.sh`）；
3. 四段文本逐字（Paris/Tokyo/1+1/静夜思）+ `faults=0` + 数字任务数数；
4. `DSV41_TIMING=1` 的步时 + nsys 的 per-kernel 计数：
   `dsv41_quant_fp8` 的 wo_b 调用数必须**归零**、`dsv41_gemm_fp8_mrows_f32` 计数 = 40+3/步。
   **launch 计数是首要判据**（R6：ms 账是设计口径）。

---

## 7. 风险与工作量

| # | 风险 | 触发 | 影响 | 应对 |
|---|---|---|---|---|
| B6-R1 | **数值改动被当成回归** | 翻 gate | verify 的 wo_b 输出与旧路径不同（更准） | 明确标为 **intentional**；判据是 M=1 f32 逐位 + 四段文本，不是旧 verify 位；`DSV41_DIFF_EAGER=1` 的 mismatch 数应**不变或下降**（wo_b 与 EAGER 同程序后反而更一致） |
| B6-R2 | **`.so` 不同源 / gate 误接** | 新符号 + 旧 `.so` | 两臂都跑旧路径，A/B 全无效果（本仓 #1 陷阱） | `supports_gemm_fp8_mrows_f32()` 符号探测 + 一次性 warning；`build.sh` BUILD_ID 核对 |
| B6-R3 | **draft 形状撑爆 smem** | k=8192 → 66560 B | `cudaErrorInvalidValue`（=2 以外的错误码） | 照抄 `>48KB` 属性阶梯 + **每个 M 特化单独 SetAttribute**；parity case 4 覆盖 draft 形状 |
| B6-R4 | **`xq_r` 陈旧读者** | 跳过 quant | 静默错答案 | §5.2 的读者核对（已核 + 实现时复核）；不依赖注释 |
| B6-R5 | **`a_stride` 语义写反** | verify 传 `ol_local` 而非 `ol_total` | 读到 row 0 的尾巴（= row≥1 全错的指纹） | launcher decline `a_stride < k`（挡不住 =k 但行距真值更大）⇒ parity case 4 必须带 `a_stride > k` 的臂，且端到端用 `DSV41_DIFF_EAGER` 看 r≥1 |
| B6-R6 | **收益高估** | ms 账 | 实测 < 设计 | 以 launch 计数为准；先做一次 nsys 按 kernel 名聚合 |

**工作量**：**1 人日**（与上位文档的 Wave 4 估值一致）。
- kernel + launcher：~130 行（`dsv41_kernels.cu`，骨架照抄 fp8 兄弟）
- parity suite：~300 行（照 `tests_dsv41_gemm_mrows.cu`）
- Rust 接线：FFI 字段 + 符号 + `supports_*` + launcher 方法 + 两个调用点 + 一个 gate（~80 行）
- **不需要**新外部依赖、不需要新 buffer（`wo_r`/`wo`/`wo_out_r`/`o` 已存在）

**预期收益**：verify −200~−240 发/步（−0.66 ~ −1.5ms，按 3.3/6.2µs 两口径）；
draft −3 发/步（一致性为主）。判词的 −0.6~1.0ms 落在保守口径内。

---

## 8. 改动文件清单（实施时）

| 文件 | 改动 |
|---|---|
| `kernels/cuda/dsv41_kernels.cu` | 新增 `gemm_fp8_mrows_f32_kernel<M>` + `dsv41_gemm_fp8_mrows_f32` launcher（放在 `dsv41_gemm_fp8_mrows` 之后，共享 LUT/ue8m0 helper 与 smem 属性宏） |
| `kernels/cuda/tests_dsv41_gemm_mrows_f32.cu` | **新建** parity suite（§6 六臂） |
| `crates/ferrite-models/src/dsv41/device.rs` | FFI 字段 `gemm_fp8_mrows_f32: Option<unsafe extern "C" fn(...)>` + `ko!(rt, "dsv41_gemm_fp8_mrows_f32")` + `supports_gemm_fp8_mrows_f32()` + `gemm_fp8_mrows_f32(...)` 方法 |
| `crates/ferrite-models/src/dsv41/chain_dev.rs` | `attention_rows` 的 wo_b 调用点插入 B6 臂（优先级最高）+ `fn wob_mrows_f32()` gate |
| `crates/ferrite-models/src/dsv41/dspark_dev.rs` | `draft_attn_out` 的 wo_b 调用点插入 B6 臂（`quant1` 之前）+ 共用同一个 gate |
| `docs/agent/` | 本文件（设计）+ 实施后的 parity/A-B 记录 |

**不改动**：`dsv41_gemm_fp8_mrows` / `gemm_fp8_mrows_kernel`（在位生产核，B6 与它**同骨架不同实例**，
不共用函数体——避免为"复用"去动一个已被 C1-C6 + 多轮 parity 钉死的核）；
`dsv41_gemm_fp8_mx_f32`（EAGER 的现状路径，B6 的 parity 基准）；
`weights.rs` / `load.rs`（dtype 与分片不变）。

---

## 附：本方案的事实来源

| 事实 | 来源 |
|---|---|
| wo_b = fp8 e4m3 + ue8m0 32×32，形状 [5120, 8192]/[160, 256] | `weights.rs:128-129`、`:701-702`、`load.rs:159-160,843-844` |
| TP8 下 k 切到 1024 | `weights.rs:781`（`local_shape(..., 8, 0) == [5120, 1024]`） |
| verify：`quant_fp8` 逐行 + `proj_mrows` | `chain_dev.rs:9801-9828` |
| EAGER：`gemm_fp8_mx_f32`（WOB_F32 默认 ON） | `chain_dev.rs:13442-13464`、`dspark_dev.rs` 注释 `:136-150` |
| draft：`quant1` + (`ATTN_PROJ_ALIGN` ? `proj_attn_mrows` : `gemm_fp8_mx`) | `dspark_dev.rs:2154-2198`、`:3658-3697` |
| `VERIFY_ROWS = 6`，`DSPARK_DRAFTS = 5` | `chain_dev.rs:84`、`dspark_dev.rs:65` |
| m-rows 骨架与 C1–C6 / decline / smem 属性阶梯 | `dsv41_kernels.cu:5125-5466`、`:3871`、`:3806` |
| f32 路径的 M=1 程序（`s_af IS the value`） | `dsv41_kernels.cu:4702-4707`、`:4882-4890`、`:6031-6114` |
| B 类清单 / Wave 4 gate 名 / 节省账 | `verify-eager-fusion-migration.md:186-196,208-246,308-317` |
| draft-verify 程序不匹配（第 2 号） | `dspark-correctness-chain.md:1939-1944` |
| 行距类根因（源行距 vs cols） | `chain_dev.rs:9802-9816`、`spec-invariants.md:72` |
