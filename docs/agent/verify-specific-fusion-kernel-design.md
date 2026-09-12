# verify 专用融合 kernel 设计（R2 的替代方案）

> 工部 · 2026-09-12 · **只读勘验 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 基线：HEAD `5f3b1d4`（逐条 `file:line` 核对 `kernels/cuda/dsv41_kernels.cu`、
> `crates/ferrite-models/src/dsv41/{chain_dev,device}.rs`）。
> 输入前提：R2（`DSV41_ATTN_LIN_FUSE`）已确认损坏（计数 61/65、出师表 ~100 字后"序列重置"）。

---

## 0. 先读三条（其中一条**修正任务前提**，必须上报尚书省）

### 0.1 ⚠️ 任务前提 #1「根因是 rope 位置处理」在 `m == 1` 下被**证伪**

任务描述："EAGER 用 `pos_ctr` 的语义 vs verify 需要每行 `pos + r` 的语义"。代码事实：

**kernel 侧位置公式**（两处，逐字对照）：

| 位置 | 公式 | file:line |
|---|---|---|
| 逐行 `apply_rope_kernel` | `t = (*base) * mul + off + r * step`（`r` = `blockIdx.x` = **head 行号**）| `dsv41_kernels.cu:1943` |
| 融合 GEMV rope epilogue | `t = (*rope_base) * rope_mul + rope_off + h * rope_step`（`h` = **head 号**）| `dsv41_kernels.cu:4976` |
| `apply_rope_mrows_kernel`（ROW-FOLD）| `t = pos_rows[r]`（r = **verify 行号**）| `dsv41_kernels.cu:2012` |

**R2 的调用参数**（`chain_dev.rs:4579`，`lin_rope_norm` 内）：

```
rope_base = s.pos_ctr,  rope_mul = 1,  rope_off = 0,  rope_step = 0
→ t = *pos_ctr + 0 + h*0 = *pos_ctr          （对全部 nlh 个 head 同一个位置）
```

**verify 的逐行调用参数**（`chain_dev.rs:9844-9857`）：

```
base = pos_ctr, mul = 1, off = r, step = 0
→ t = *pos_ctr + r
```

**关键等式链**（lazy 路径，`m == 1`）：

```
chain_dev.rs:8462   set_pos_ctr(pos + i)                      // H2D，阻塞，设备静止
chain_dev.rs:8472   step_rows_sync(&rows[i..=i], true, Some(pos + i))
chain_dev.rs:5540   pos_base = hint = *pos_ctr                // 不再 D2H，值由构造相等
chain_dev.rs:5544   pos_rows[r] = pos_base + r
⇒  m == 1 时  pos_rows[0] = pos_base = *pos_ctr
```

**⇒ R2 的 `t = *pos_ctr` 与逐行调用的 `t = pos_rows[0]` 是同一个整数。位置没有错。**

head 索引也对得上：融合 epilogue 的 `h = re / rhs`（`rhd = rope_hd = hd`，
`dsv41_kernels.cu:4967`）与逐行 `row + h*row_len + (row_len - dim)`（`row_len = hd`，
`dsv41_kernels.cu:1944`）索引同一个 head 基址；rope 列区间都是 `[hd - rd, hd)`。

> 这一结论与仓库里最后一次独立分析一致（`7ade4de`："rope_base = pos_ctr with
> mul=1/off=0/step=0; the lazy path's set_pos_ctr(pos+i) should make it correct"）。
> **R2 的 rope 位置不是 bug。** 因此**把 `off=r` 换成 `pos_rows[r]` 并不会修好 R2** ——
> 新设计必须建立在"逐段等价于 verify 自己的分离 kernel"上，而不是"修 rope 位置"。

（另：`dsv41_kernels.cu:4967-4976` 中 `h` 是 family 内的 head 号，
`re = e - roff`，family 边界由 `n1` 切；`lin_rope_norm` 是单 family（`gc.n1 = n`），
所以 `roff = 0` 恒成立，没有 family 边界问题。）

### 0.2 排除项（读码结论，供根因组复用）

以下 R2 声称的"按构造逐位等价"**逐条核对成立**，可以排除：

| 环节 | 参照 | 结论 |
|---|---|---|
| NORM_FUSE 归一化 prologue | `rmsnorm_rows_kernel`（verify 的 `norm_rows`）| **同**：blockDim 1024 的 `for i = tid; i < k; i += 1024` + `__shfl_down` 树 + thread0 跨 warp 顺序求和 + `rsqrtf(t/k + eps)`（`dsv41_kernels.cu:4611-4640` vs `:8709-8731`）|
| NORM_FUSE fp8 发射 | `quant_kernel<0>`（verify 的 `quant_rows`）| **同**：`fast_round_scale(amax, 1/448)` + `max(...,1e-30)` + clamp ±448 + `__nv_fp8_e4m3`，32-lane shuffle-xor amax（`:4630-4639` vs `:144-167`）|
| 融合 rope | `apply_rope_kernel` | **同**：`t*(rope_rd>>1)+i` 表索引、`x0/x1` 对、旋转式（`:4974-4984` vs `:1946-1950`）|

⇒ **R2 的两个融合 kernel 与其分离参照的逐元素运算是等价的。**
最强剩余嫌疑因此**不是**任何单段的算术，而是：
（a）`gemm_fp8_mx2` / `gemm_fp8_gemv_kernel(mode 4, a32)` **与**
`gemm_fp8_mrows` 是两个不同 kernel 程序（各自宣称等价于 m=1 GEMV）——1 ULP 级差异
× 40 层 × 60+ 步，正是"前 60 token 正确、之后翻转"的形状；
（b）**共享 scratch `s.xq` / `s.xsc`**：R2 的 `lin2` 走 `quant1`（`chain_dev.rs:4429`），
而 `s.xq`/`s.xsc` 是**单行 scratch**，verify 的 rows 路径平时完全不碰它（它用 `s.xq_r`）；
MoE 的 shared half（`DSV41_MOE_DUAL`）与 o/wo 量化都以 `s.xq` 为写点
（`chain_dev.rs:1089, 10428, 14845`）——R2 在 rows 路径里**新引入了一个 `s.xq` 写者**；
（c）`quant1` 的 T1/T2 consume-once 旗标（`xq_of_xn_valid` / `xq_of_qr_valid`）
以 `s.xn` / `s.qr` 的**指针**为键（`chain_dev.rs:4108, 4118`）——R2 传的是 `s.xn_r`/`s.qr_r`，
不触发 elision，但把两套 scratch 的旗标语义搅在同一层里。

**最高价值的 30 分钟实验（P0，建议先做）**：写一个纯 kernel parity 测试
（`kernels/cuda/tests_dsv41_gemm_mrows.cu` 的形态），在同一组输入上分别跑
① `lin2` + `lin_rope_norm` 与 ② `quant_rows + 2×proj_mrows + norm_rows + quant_rows +
proj_mrows + apply_rope_mrows`，逐字节 diff `qr_r`/`kv_r`/`q_r`。
- 若**逐字节相同** ⇒ 损坏在调用侧的共享状态/图交互，**本设计的 K1/K2 天然免疫**（见 §2 原则 3）。
- 若**不同** ⇒ 拿到第一处差异的 (row, col) 与差值，直接定位 (a)。**这是把"猜测"变成"事实"的唯一廉价路径。**

### 0.3 本设计对"根因未定"的立场（诚实校准）

本设计**不能保证**修好 R2 的损坏——因为 §0.2 的方案 (a)/(b)/(c) 在无 GPU 的读码层面
无法判定。但本设计**在构造上排除了这三条**（§2 原则 1/2/3），并且把"位置语义"
从"隐式读设备计数器"改成"显式 `pos_rows` 入参"，从而**不论** R2 的根因是哪一条，
都不再落在本设计的参数面上。**建议：先做 P0，再决定是否值得投入 K1/K2（§6 成本）。**

---

## 1. 现场：R2 做了什么、发数账

### 1.1 verify 的 q 链现状（`attention_rows`，lazy `m == 1`）

| # | 调用 | file:line |
|---|---|---|
| 1 | `quant_rows(xn_r)` → `xq_r`/`xsc_r` | `chain_dev.rs:9636` |
| 2 | `proj_mrows(wq_a)` → `qr_r` | `:9637-9645` |
| 3 | `proj_mrows(wkv)` → `kv_r` | `:9646-9654` |
| 4 | `norm_rows(qr_r)` 就地 | `:9769-9778` |
| 5 | `quant_rows(qr_r)` → `xq_r`/`xsc_r` | `:9784` |
| 6 | `proj_mrows(wq_b)` → `q_r` | `:9785-9793` |
| 7 | `apply_rope_mrows(q_r)`（或逐行 `apply_rope`）| `:9827-9858` |

**= 7 发/层/行**（`proj_mrows` 拒绝时退化为 `2m`/`4m` 逐行发射）。

R2 = 把 1+2+3 换成 EAGER 的 `lin2`（`:9619`）、4+5+6+7 换成 EAGER 的 `lin_rope_norm`（`:9743`）
⇒ 2 发。EAGER 的这两个 kernel 是 **`m = 1` 专用程序**（`gemm_fp8_gemv_kernel` 家族，
`dsv41_kernels.cu:5887 / 7642`）。

### 1.2 形状（`crates/ferrite-models/configs/dsv41_flash.json` + TP8）

```
dim=5120  n_heads=64  head_dim=512  rope_head_dim(rd)=64  q_lora(ql)=1280
o_lora=1024  o_groups=8  layers=40  sliding_window=128  index_topk=512
index_n_heads=32  index_head_dim=128  dspark_block_size=5
world=8 ⇒ nlh = 64/8 = 8,  nlg = 8/8 = 1
```

⇒ wq_b：`n = nlh*hd = 4096`, `k = ql = 1280`, `out_stride = nh*hd = 32768`；
wq_a：`n = 1280`, `k = 5120`；wkv：`n = 512`, `k = 5120`。

---

## 2. 设计原则（为什么不复用 EAGER 就安全）

1. **不复用 EAGER 的任何 kernel 程序。** 每一个融合段都以 **verify 自己已经在用的**
   kernel 为参照：`quant_rows`（`quant_kernel<0>`）、`proj_mrows`（`gemm_fp8_mrows_kernel<M>`）、
   `norm_rows`（`rmsnorm_rows_kernel` / `rmsnorm_kernel`）、`apply_rope_mrows`
   （`apply_rope_mrows_kernel`）。融合只搬动代码，不换程序。
   → 消除 §0.2(a) 的整类风险（`gemm_fp8_mx2` / `lin_rope_norm` vs mrows 链的跨族等价断言）。
2. **位置是入参，不是隐式设备读。** 新 kernel 只接受 `pos_rows`（设备指针数组），
   在 kernel 内 `for r ascending` 读 `pos_rows[r]` —— **逐字复制 `apply_rope_mrows_kernel:2012`**。
   任何新 kernel 里**没有 `pos_ctr`、没有 `mul/off/step`**。
   → 消除"单行位置语义 vs 每行位置语义"这一类错误的**结构可能性**（而不只是修一个点）。
3. **不碰单行 scratch。** 新 kernel 只消费 `s.xq_r`/`s.xsc_r`/`s.qr_r`/`s.xn_r`，
   **绝不触碰 `s.xq`/`s.xsc`/`s.xn`/`s.qr`**，也不经过 `quant1`（因而完全不碰
   `xq_of_xn_valid`/`xq_of_qr_valid`）。
   → 消除 §0.2(b)/(c) 的整类风险。
4. **decline = 原路返回。** 每个新 C 入口在形状/`.so` 不支持时返回 **2**（不是 1），
   调用侧原样走今天的 7 发序列。**新符号而不是改签名**：仓库的 `.so` 兼容约定是
   "一个固定 ABI ⇒ 陈旧 `.so` 退化为一次 `dlsym` 探测"（`device.rs:505-532` 的注释，
   以及 `dsv41_gemm_fp8_mx_add`/`_rope_norm` 的"独立入口点"既有先例）。
   ⇒ **不要**给 `dsv41_gemm_fp8_mrows` 加"第二 family"可选参数（任务描述里的备选方案）——
   那会让旧 `.so` 静默接受一个它不认识的 ABI。
5. **m ≥ 2 免费获得。** 两个新 kernel 都是 mrows 形态（`template <int M>` + r 循环），
   lazy（m=1）只是 M=1 的特例。**这不是本轮的验收目标，但结构上不会像 R2 那样被
   "m=1 专用"锁死**（R2 的 `lin2`/`lin_rope_norm` 在 m≥2 下不可用，这正是 R2 被认为
   "net loss for the batched arm" 的原因）。

---

## 3. K1 —— `dsv41_gemm_fp8_mrows2`（wq_a + wkv 一次发射）

### 3.1 目标
把 §1.1 的 #2+#3（两次 `proj_mrows`）合成**一次**发射；激活 `s.xq_r`/`s.xsc_r` 本就被
`quant_rows` 共享（`chain_dev.rs:9636`），所以**省的是恰好 1 发 + 1 个 graph 节点**，
不是字节（`docs/agent/projection-family-optimization.md §Q2(a)` 已记）。

### 3.2 CUDA 侧签名

```cpp
// template <int M> 与 gemm_fp8_mrows_kernel 同构；只多一个"按 warp 选 family"。
template <int M>
__global__ void __launch_bounds__(256)
gemm_fp8_mrows2_kernel(
    const uint8_t* __restrict__ a, const float* __restrict__ a_scale,
    const uint8_t* __restrict__ w1, const uint8_t* __restrict__ w1_scale,
    const float* __restrict__ bias1, float* __restrict__ out1, int n1,
    const uint8_t* __restrict__ w2, const uint8_t* __restrict__ w2_scale,
    const float* __restrict__ bias2, float* __restrict__ out2, int n2,
    int k, int out_stride, int a32);

// 返回 0 = 已发射；2 = 拒绝（调用侧退回两次 proj_mrows）。
// ABI: stream LAST（与 dsv41_gemm_fp8_mrows / _mx_rope_norm 一致）。
extern "C" int dsv41_gemm_fp8_mrows2(
    const uint8_t* a, const float* a_scale,
    const uint8_t* w1, const float* w1_scale, const float* bias1, float* out1, int n1,
    const uint8_t* w2, const float* w2_scale, const float* bias2, float* out2, int n2,
    int m, int k, int out_stride, cudaStream_t s);
```

### 3.3 kernel 结构（与 `gemm_fp8_mrows_kernel<M>` 逐行同构，只有 4 处 `family` 选择）

```
grid  = (ceil((n1 + n2) / nwarps), 1)     nwarps = dsv41_mrows_warps_for(n1 + n2)
block = nwarps * 32                        （与 mrows 完全相同的策略）
smem  = nwarps*k | 256*4 (lut) | M*nb_k*4 (s_as) | M*k (s_a)     ← 与 mrows 逐字节相同
row   = blockIdx.x * nwarps + warp
const bool fam1 = (row < n1);
const int  rrow = fam1 ? row : row - n1;              // ① family 内行号
const uint8_t* wf  = fam1 ? w1 : w2;                  // ② 权重基址
const float*   bs  = fam1 ? bias1 : bias2;            // ③ bias 基址
float*         of  = fam1 ? out1 : out2;              // ④ 输出基址
// 之后：wr = wf + rrow*k；wsr = (fam1?w1_scale:w2_scale) + (rrow>>5)*nb_k；
//       写 out[r*out_stride + rrow] = acc[r] + bs[rrow]
// consume 循环（kb 升序、j = kb*32 + lane、acc[r] += av*wv、shfl_xor 树）一字不改。
```

* `warp 级 family 选择` ⇒ **family 边界落在 block 内也正确**（每个 warp 各算自己那行），
  所以 `n1` 不需要是 `32` 的倍数（无 rope ⇒ 无 pair 约束）。
* `a`/`a_scale` 多行 staging、`s_lut`、`a32` 臂、`__launch_bounds__(256)`、
  cp.async16 权重装载 —— **全部照抄**（`:5272-5281`）。

### 3.4 拒绝条件（继承 `dsv41_gemm_fp8_mrows`，`:5373-5390`）
`m ∉ 1..=8` / `k & 31` / `out_stride < max(n1,n2)` / 空指针 / `g_gemv_fp8_mode < 3` /
`DSV41_NO_GEMV_FP8` / `n1 <= 0 || n2 <= 0` ⇒ 返回 2。

### 3.5 调用侧（Rust）

```rust
// device.rs：与 gemm_fp8_mrows 同形的 Option<unsafe extern "C" fn(...)> 字段 + ko! 加载
pub fn gemm_fp8_mrows2(&self, a, a_scale, w1, ws1, b1, out1, n1,
                       w2, ws2, b2, out2, n2, rows, k, out_stride) -> Result<bool>
// rc == 2 → Ok(false)（decline），其余走 kerr。
pub fn supports_gemm_fp8_mrows2(&self) -> bool
```

```rust
// chain_dev.rs：新增 proj_mrows2，与 proj_mrows(:4266) 同构
fn proj_mrows2(&self, ld: &LayerW, out_qr: *mut f32, ql: i32,
               out_kv: *mut f32, hd: i32, rows: usize, k: i32) -> Result<bool>
// 内部只传 self.s.xq_r / self.s.xsc_r —— 与 proj_mrows 同一个激活源。
```

`attention_rows` 的改动（§1.1 的 #2/#3）：

```rust
let mrows2 = self.dev.supports_gemm_fp8_mrows2() && !Self::swapab() && m <= VERIFY_ROWS;
let took_akv = if mrows2 {
    self.quant_rows(self.s.xn_r.ptr as *const f32, m, dim as i32)?;
    self.proj_mrows2(ld, self.s.qr_r.ptr as *mut f32, ql as i32,
                     self.s.kv_r.ptr as *mut f32, hd as i32, m, dim as i32)?
} else { /* 现有 mrows 分支，一字不改 */ };
```

（`quant_rows(xn_r)` 仍是独立 1 发 —— 见 §5 的可选 K1'。）

---

## 4. K2 —— `dsv41_gemm_fp8_mrows_rope_norm`（norm + quant + wq_b + rope 一次发射）

### 4.1 目标
把 §1.1 的 #4+#5+#6+#7 合成**一次**发射（4 → 1），且**逐段**等价于这四个 kernel。

### 4.2 CUDA 侧签名

```cpp
template <int M>
__global__ void __launch_bounds__(1024)          // ← 1024 是承重约束，见 §4.4
gemm_fp8_mrows_rope_norm_kernel(
    const float* __restrict__ qr_raw,            // [M, k] f32 —— wq_a 的 RAW 输出
    const float* __restrict__ qr_w,              // [k]
    float qr_eps,
    float* __restrict__ qr_norm_out,             // [M, k] f32 —— 可为 nullptr（见 §4.6）
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ w_scale,
    const float* __restrict__ bias, float* __restrict__ out,
    int n, int k, int out_stride,
    const float* __restrict__ cos, const float* __restrict__ sin,
    const int* __restrict__ pos_rows,            // ← 位置：设备数组，行 r 的位置
    int rope_rd, int rope_hd, int rope_inverse, int a32);

// 返回 0 / 2（拒绝）。ABI: stream LAST。
extern "C" int dsv41_gemm_fp8_mrows_rope_norm(
    const float* qr_raw, const float* qr_w, float qr_eps, float* qr_norm_out,
    const uint8_t* w, const uint8_t* w_scale, const float* bias,
    float* out, int m, int n, int k, int out_stride,
    const float* rope_cos, const float* rope_sin, const int* pos_rows,
    int rope_rd, int rope_hd, int rope_inverse, cudaStream_t s);
```

### 4.3 kernel 结构（三段，每段一个参照 kernel）

```
grid  = (ceil(n / 32), 1);  block = 1024 (32 warps)         ← §4.4
smem  = 32*k (s_w) | 256*4 (s_lut) | M*nb_k*4 (s_as) | M*k (s_a) | M*32 (s_rows 交换槽)

─ 段 1: PROLOGUE（参照 rmsnorm_rows_kernel + quant_kernel<0>，逐行 r 循环）
  for r in 0..M:
      // 1a 归一化：rmsnorm_q_kernel:8626-8639 的算术，term for term
      ss = 0; for (i = tid; i < k; i += 1024) { x = qr_raw[r*k+i]; ss += x*x; }
      lane = ss; for (off=16..1) lane += __shfl_down_sync(~0u, lane, off);
      if ((tid & 31) == 0) s_red[tid>>5] = lane;  __syncthreads();
      if (tid == 0) { t = Σ_{w=0..31} s_red[w]; s_red[0] = rsqrtf(t / k + qr_eps); }
      __syncthreads();  inv = s_red[0];
      // 1b 写回（可选，见 §4.6）+ (1c) fp8 发射：rmsnorm_q_kernel:8641-8652 的算术
      for (i = tid; i < k; i += 1024) {
          v = qr_raw[r*k+i] * inv * qr_w[i];
          if (qr_norm_out) qr_norm_out[r*k+i] = v;              // ← §4.6
          a = fabsf(v); for (off=16..1) a = fmaxf(a, __shfl_xor_sync(~0u,a,off));
          sc = fmaxf(fast_round_scale(a, 1/448), 1e-30f);
          if ((tid & 31) == 0) s_as[r*nb_k + (i>>5)] = sc;
          q = fminf(fmaxf(v * (1/sc), -448.f), 448.f);
          s_a[r*k + i] = *(const uint8_t*)&__nv_fp8_e4m3(q);
      }
      __syncthreads();

─ 段 2: CONSUME（参照 gemm_fp8_mrows_kernel<M>，逐字复制）
  wr = w + row*k;  cp.async16 装载进 s_w + warp*k;  (逐行 r: av = s_lut[s_a[r*k+j]] * s_as[...])
  for kb in 0..nb_k:  wv = s_lut[wr_local[j]] * sb;  for r: acc[r] += av[r] * wv;
  for r: shfl_xor 树 → t[r] = acc[r] + bias[row];

─ 段 3: ROPE EPILOGUE（参照 apply_rope_mrows_kernel:2005-2023 + gemv 的 pair 交换 :4948-4984）
  for r: s_rows[r*32 + warp] = t[r];                 // 每 warp 一行，跨 warp 交换
  __syncthreads();
  h = row / rope_hd;  lane_in = row % rope_hd;  sect = rope_hd - rope_rd;
  if (active && warp + 1 < 32 && lane_in >= sect && ((lane_in - sect) & 1) == 0) {
      i = (lane_in - sect) >> 1;
      if (lane == 0)
        for r in 0..M:
            const int tpos = pos_rows[r];                        // ← 每行位置，设备数组
            c = cos[tpos*(rope_rd>>1) + i];
            s = sin[tpos*(rope_rd>>1) + i] * (rope_inverse ? -1.f : 1.f);
            x0 = s_rows[r*32 + warp]; x1 = s_rows[r*32 + warp + 1];
            p = out + r*out_stride + row;
            p[0] = x0*c - x1*s;   p[1] = x0*s + x1*c;
  }
```

> **注意**：`out[r*out_stride + row]` 的前 `n` 列由段 2 写、后 `rope_rd` 列由段 3 覆写；
> 段 2/3 写在 `__syncthreads()` 两侧，无重排。

### 4.4 ⚠️ 承重约束：blockDim 必须是 1024（32 warps）

`rmsnorm_rows_kernel` / `rmsnorm_kernel` / `rmsnorm_q_kernel` 的 reduction 树都是
**blockDim 尺寸**的（`for i = tid; i < k; i += blockDim` + `blockDim>>5` 个部分和由
thread 0 顺序相加）。`gemm_fp8_mrows_kernel<M>` 是 `__launch_bounds__(256)`、blockDim=256。

**如果 K2 用 256 线程做 prologue，部分和的组合顺序就变了**（k=1280 时：
256 线程 = 每线程 5 个元素 / 8 个部分和；1024 线程 = 每线程 1~2 个元素 / 32 个部分和），
浮点加法不结合 ⇒ **1 ULP 级差异**。这正是 R2 的 `lin_rope_norm` 强制
`warps = 32`（`dsv41_kernels.cu:5901`，注释明说"the prologue's reduction tree is
rmsnorm_q_kernel's 1024-thread tree"）的那条约束。

**K2 因此固定 block = 1024 / 32 warps。** 段 2 的几何（nwarps=32）**不影响** 结果：
mrows 的 parity 论证不依赖 block 几何（"rows are independent"，`:5390-5400`），
每个 warp 的 k-walk 与 shfl 树与自己行绑定，跨 warp 只共享**字节相同的** staging。
⇒ grid = `ceil(n/32)`（wq_b: 128 blocks；idx_wq_b: 128 blocks）。

代价（诚实标注）：128 blocks < 148 SM ⇒ 部分 SM 空转；`__launch_bounds__(1024)` 对
`acc[M]` 的寄存器压力更敏感。**这是正确性优先的取舍**，与 EAGER 的 `lin_rope_norm`
（同样 `blocks = n/32`、`warps = 32`）同档，不引入新的风险面。

### 4.5 段 3 的 pair 不跨 block（承重论证）

段 3 需要输出行对 `(row, row+1)` 落在同一 block。因 `block` 覆盖 32 个**连续**输出行
（`row = blockIdx.x*32 + warp`），且 `rope_hd % 32 == 0`、`rope_rd` 为偶数：

```
head h 的 rope 区间 = 行 [h*rope_hd + rope_hd - rope_rd,  h*rope_hd + rope_hd)
最后一个 pair = 行 (h*rope_hd + rope_hd - 2, +1)
该行在其 block 内的 warp 偏移 = (rope_hd - 2) % 32 = 30   （rope_hd % 32 == 0）
⇒ warp + 1 = 31 < 32，pair 不跨 block。                              ✓
```

（wq_b: `rope_hd = hd = 512`, `sect = 448`；idx_wq_b: `rope_hd = idx_hd = 128`,
`sect = 64`。两者都满足。）⇒ 拒绝条件里加 `nwarps == 32`（本 kernel 恒真）、
`rope_hd % 32 == 0`、`0 < rope_rd <= rope_hd`、`rope_rd` 为偶数、`out_stride >= n`。

### 4.6 `qr_norm_out`：与 R2b 的兼容（**默认写回，不加 flag**）

R2 的 `lin_rope_norm` 把归一化结果留在 shared memory、**不写回 `qr_r`**，因此
`qr_r` 保持 RAW，于是必须引入一个消费侧旗标（`s.qr_raw`（单行）/ `s.qr_raw_r`（行批））
把"RAW"这件事传播给 indexer（`indexer()` 的 `lin_rope_norm(idx_wq_b)`，`chain_dev.rs:14510`；
R2b 的 `indexer_front_rows`/`indexer_rows_one`，`:10787`）。

**这条旗标链有三个真实成本：**
1. 它是"隐藏状态"，跨函数、跨 indexer/attention 两个调用者；
2. R2b 的消费侧用的是 **`m = 1` 专用**的 `lin_rope_norm`，却由
   `debug_assert!(m == 1, ...)`（`:10786`）**而非硬门禁**保护 —— release 下 m>1
   会静默用单行程序跑多行（今天不可达，因为 R2 只在 m==1 置位；但这是一个**结构性陷阱**）；
3. 它把 indexer 的读数语义与 attention 的融合实现耦合。

**K2 的设计选择：`qr_norm_out = qr_r`（写回）。** 段 1b 的那一句 `qr_norm_out[r*k+i] = v`
就是 `norm_rows(qr_r, q_norm, qr_r, ...)` 的逐字节等价物（同一个 `v`、同一个 `inv`、
同一块内存、每元素恰好一次写）。于是：

* `qr_r` 离开 K2 时**已经是归一化的** ⇒ indexer 的 q 半边（`indexer_front_rows` /
  `indexer_rows_one`）**一行都不用改**，与今天（R2 关闭时）的读数完全一致；
* `s.qr_raw_r` **不需要**，`DSV41_INDEXER_QR_RAW` **不需要**，那个 `debug_assert` 陷阱消失；
* 代价：每行多 `k` 个 f32 的 store（wq_b: 1280 f32 = 5 KB/行；`m=6` 时 30 KB/块）。
  与"wq_b 权重 5.24 MB/层"相比是噪声。

> 若后续确认 §0.2(a) 类差异不存在、且要看 R2b 那 1 发/层的收益，K2 留了 `qr_norm_out = nullptr`
> 的第二种用法：此时 `qr_r` 保持 RAW，由**新的、mrows 形态的** indexer 入口
> （`dsv41_gemm_fp8_mrows_rope_norm` 的同一次调用即可复用，传 `qr_norm_out = nullptr`
> + `w = idx_wq_b` + `rope_hd = idx_hd`）来消费 —— 这比 R2b 复用 m=1 程序**结构上更正确**。
> **建议先落地"写回"版本**（零耦合），把这一项留作后续 A/B。

### 4.7 拒绝条件

`m ∉ 1..=8` / `k & 31` / `n & 31` / `out_stride < n` / `rope_hd & 31` /
`rope_rd <= 0 || rope_rd & 1 || rope_rd > rope_hd` / 空指针
（`qr_raw`/`qr_w`/`w`/`w_scale`/`out`/`cos`/`sin`/`pos_rows`）/
`g_gemv_fp8_mode < 3` / `DSV41_NO_GEMV_FP8` ⇒ 返回 2。

### 4.8 调用侧（Rust）

```rust
// device.rs
pub fn gemm_fp8_mrows_rope_norm(&self, qr_raw, qr_w, eps, qr_norm_out,
                                w, ws, bias, out, rows, n, k, out_stride,
                                cos, sin, pos_rows, rope_rd, rope_hd, inverse) -> Result<bool>

// chain_dev.rs：与 lin_rope_norm(:4551) 同构，但位置参数显式传 pos_rows
fn mrows_rope_norm(&self, a_raw: *const f32, qw: *const f32, eps: f32, k: i32,
                   w: &DevTensor, ws: &DevTensor, n_out: i32, out: *mut f32,
                   rows: usize, out_stride: i32, pos_rows: *const c_int,
                   rope_rd: i32, rope_hd: i32) -> Result<bool>
```

`attention_rows` 的改动（§1.1 的 #4~#7）：

```rust
let took_b2 = if mrows2 {
    self.mrows_rope_norm(
        self.s.qr_r.ptr as *const f32, ld.q_norm.as_ref().unwrap().as_f32(), cfg.norm_eps,
        ql as i32, ld.wq_b.as_ref().unwrap(), ld.wq_b_scale.as_ref().unwrap(),
        (nlh * hd) as i32, self.s.q_r.ptr as *mut f32,
        m, (nh * hd) as i32,
        self.s.pos_rows.ptr as *const c_int,      // ← 位置：verify 自己的表
        rd as i32, hd as i32)?
} else { false };
// 只有 !took_b2 时才跑现有的 norm_rows + quant_rows + proj_mrows + rope 四发（一字不改）
```

**K2 之后 `q_roped = true`**，所以 `apply_rope_mrows` / 逐行 `apply_rope`（`:9827-9858`）
必须跳过 —— 与 R2 的 `q_norm_fused` 处理同形，但位置来自 `pos_rows[r]`
（在 kernel 内读），**不是** `*pos_ctr`。KV 半边（`norm_rows_on` + `apply_rope_on`，
`:9864-9887`）**不动**。

---

## 5. 发数账（lazy `m == 1`，每层每行）

| 方案 | q 链发射序列 | 发数 |
|---|---|---|
| 今天（mrows） | `quant_rows(xn)` → `proj_mrows(wq_a)` → `proj_mrows(wkv)` → `norm_rows` → `quant_rows(qr)` → `proj_mrows(wq_b)` → `apply_rope_mrows` | **7** |
| **R2（损坏）** | `lin2` → `lin_rope_norm` | 2 |
| **本设计（K1+K2）** | `quant_rows(xn)` → **`mrows2`** → **`mrows_rope_norm`** | **3**（−4）|
| 本设计 + 可选 K1' | **`mrows2_norm`** → **`mrows_rope_norm`** | **2**（−5，与 R2 同档但零 EAGER 依赖）|

**K1'（可选，不在本轮推荐）**：给 K1 也加一个 f32 prologue（读 `s.xn_r` + 用
`rmsnorm_q` 的算术 + fp8 发射），把 `quant_rows(xn)` 也吃掉。**不推荐先做**：
它引入第二个 prologue（多一份 `blockDim` 承重约束、多一个 `M*k` smem 槽），
收益只有 1 发/层，而 §5 的 3 发版本已经把 7→3 走完。等 K1+K2 的 parity 过了再谈。

**graph 节点**：`VERIFY_GRAPH=1` 下节点数同比例下降（3 发 ⇒ 每层 3 个节点 vs 7 个）。

---

## 6. 数值等价论证（逐段，参照 verify 自己的 kernel）

| 融合段 | 参照 kernel（verify 今日在用）| 等价依据 |
|---|---|---|
| K1 family 选择 + staging + consume | `gemm_fp8_mrows_kernel<M>`（`proj_mrows`）| 只多 4 处按 `row < n1` 的基址选择；`row=`、`rrow=`、`wr/wsr/of/bs` 的算式与 mrows 逐字相同；k-walk / `acc[r] += av*wv` / shfl 树 / `out[r*out_stride+row]` 一字不改 ⇒ **逐位** |
| K2 段 1a 归一化 | `rmsnorm_rows_kernel`（`norm_rows`）| 同 blockDim=1024、同 strided 元素循环、同 `__shfl_down` 树、同 thread-0 跨 warp 顺序求和、同 `rsqrtf(t/k+eps)` ⇒ **逐位** |
| K2 段 1b 写回（可选）| `norm_rows` 的就地写 | 同 `v`、同 `inv`、同地址、每元素一次写 ⇒ **逐位** |
| K2 段 1c fp8 发射 | `quant_kernel<0>`（`quant_rows`）| 同 32-lane `__shfl_xor` amax、同 `fast_round_scale(a,1/448)`、同 `max(...,1e-30)`、同 clamp ±448、同 `__nv_fp8_e4m3`、同 block 边界索引 ⇒ **逐位** |
| K2 段 2 consume | `gemm_fp8_mrows_kernel<M>`（`proj_mrows`）| 同 K1（单 family），激活来源 `s_a[r*k+i]`/`s_as` 与 mrows 的 staging 布局逐字节一致 ⇒ **逐位** |
| K2 段 3 rope | `apply_rope_mrows_kernel`（`apply_rope_mrows`）| 同 `t*half+i`（`half = rope_rd>>1`）表索引、同 `x0/x1` 对、同旋转式；列区间同为 head 的 `[rope_hd-rope_rd, rope_hd)`；位置同为**行 r 的 `pos_rows[r]`** ⇒ **逐位** |

**位置语义（§2 原则 2）单独强调**：

```
逐行参照（chain_dev.rs:9844）  apply_rope(q_r + r*nh*hd, ..., base=pos_ctr, mul=1, off=r, step=0)
                                → t = *pos_ctr + r
ROW-FOLD 参照（:9829）          apply_rope_mrows(..., pos_rows)  → t = pos_rows[r]
本设计 K2 段 3                  t = pos_rows[r]                  ← 与上面两者同一个整数
```

且 `pos_rows` 在每步的 `step_rows_sync` 里由 host 写死：
`pos_rows[r] = pos_base + r`（`chain_dev.rs:5544`），`pos_base` = 该步 `*pos_ctr`
（`:5540-5543`）⇒ 与"逐行 `off=r`"构造性相同（`apply_rope_mrows` 的 header 已给出同一论证，
`dsv41_kernels.cu:1981-1998`）。**新 kernel 内没有任何 `pos_ctr` 读取、没有 `mul/off/step`。**

---

## 7. 实施成本

| 项 | 内容 | 工作量 |
|---|---|---|
| K1 | kernel `gemm_fp8_mrows2_kernel<M>`（~70 行，mrows 的 family 化）+ C 入口（~35 行，含 8 个 M 特化的 smem 属性设置）+ `device.rs` 字段/加载/wrapper/supports（~60 行）+ `proj_mrows2`（~35 行）| **1.0–1.5 人日** |
| K2 | kernel `gemm_fp8_mrows_rope_norm_kernel<M>`（~130 行：prologue/consume/rope 三段）+ C 入口（~40 行）+ device wrapper（~50 行）+ `mrows_rope_norm`（~45 行）| **1.5–2.0 人日** |
| 接线 | `attention_rows` 两个分支 + `q_roped` 跳过逻辑 + 两个 gate（`DSV41_MROWS_FUSE` 默认 OFF，`DSV41_MROWS_ROPE_NORM` 默认 OFF）+ decline 回退 | **0.5 人日** |
| 测试 | ① K1 vs `2× proj_mrows` 行级 parity；② K2 vs `norm_rows+quant_rows+proj_mrows+apply_rope_mrows` 行级 parity（**m=1 与 m=6 各一组，位置用非平凡 `pos_rows`**）；③ decline 路径（陈旧 `.so` 符号为 None / 形状拒绝）；④ `cargo check --workspace` + `cargo test` | **1.0 人日** |
| **合计** | | **4–5 人日 + 1~2 次 GPU** |

**构建注意（仓库既有坑）**：
* 每个 `M` 特化要**各自** `cudaFuncSetAttribute`（`gemm_fp8_mrows` 的 `FERRITE_SET_MROWS_SMEM`
  宏，`:5385-5400`；现成先例，照抄）。
* `kernels/cuda/tests_dsv41_gemm_mrows.cu` 是 K1/K2 的天然测试宿主（已有 `M` 特化的
  CPU/GPU 对照框架）。
* smem 预算：K2 在 `M=6, k=1280` 时 `32*1280 + 1024 + 6*160 + 6*1280 + 6*32 ≈ 50.6 KB`
  > 48 KB ⇒ 必须走属性 opt-in 路径；`M=1` 时 ~42 KB 免 opt-in。

---

## 8. 验证计划（GPU，最少次数）

| # | 会话 | 内容 | 判据 |
|---|---|---|---|
| **P0** ★ | GPU(1) | **纯 kernel parity**（§0.2）：同一输入下 `lin2+lin_rope_norm` vs 分离链，逐字节 diff `qr_r`/`kv_r`/`q_r` | 找到/排除 R2 的 kernel 级根因；**决定本设计是否值得投入** |
| P1 | GPU(1) | K1：`DSV41_MROWS_FUSE=0/1`，`tests_dsv41_gemm_mrows` 行级 parity + `dspark_parity` | `verify_bad == 0`；文本逐字；q 链发数 7→6 |
| P2 | GPU(1) | K2：`DSV41_MROWS_ROPE_NORM=0/1`；**m=1 与 m=6 两种 block**，`pos_rows` 取非平凡值（如 `[1000,1001,...]`）| `verify_bad == 0`；计数 1..N 与出师表全文逐字；发数 6→3 |
| P3 | GPU | 组合 K1+K2 + 既有 lazy 栈；出师表 / 计数 / 长文三任务 | 零拉丁、无"序列重置"、`k_acc` ≈ 5.0 |

**铁律**：同一远端同时只有一个测试驱动；每个 gate 翻转必须读回确认
（仓库历史：gate 设了但没生效）。

---

## 9. 风险与回退

| 风险 | 说明 | 缓解 |
|---|---|---|
| 根因不在 kernel（§0.2 (b)/(c)）| P0 若判定"kernel 逐字节相同"，则本设计**不会**"修好"R2；但它**天然免疫**那两条（不碰 `s.xq`/`s.xsc`、不走 `quant1`、不带旗标）| P0 先行；若 (b)/(c) 成立，损坏应在**别的 gate**下也可见，应单独立案 |
| blockDim=1024 的开销 | 128 blocks < 148 SM；`acc[M]` 寄存器压力 | 正确性优先；先用 32 warps，A/B 后再谈 `nwarps` 策略（**改 nwarps 需重新论证 prologue 的 reduction 树**）|
| 段 3 pair 跨 block | 已论证不跨（§4.5），但这是**形状承重** | 拒绝条件里加 `rope_hd % 32 == 0 && nwarps == 32`；不满足直接 decline |
| 新 kernel 的 smem opt-in | `M=6` 时 > 48 KB | 照抄 `FERRITE_SET_MROWS_SMEM`（每个 M 特化各自设）|
| 与 batched 臂的关系 | K1/K2 是 mrows 形态 ⇒ m≥2 也能用，但**本轮不作为验收目标** | gate 默认 OFF，A/B 通过后再评估 |

**回退**：两个 gate 默认 OFF；`supports_*` 为 false（陈旧 `.so`）时自动落到今天的 7 发序列，
**一字不改**。

---

## 10. 待上报尚书省的冲突项（工部职责）

1. **前提冲突**：任务前提"根因 = rope 位置处理"在 `m == 1` 下**不成立**
   （§0.1，三条公式 + 四条 `file:line` + lazy 时序）。
   ⇒ 若 R2 的修复策略是"把 `off=r` 改成 `pos_rows[r]`"，**那是等价变换，不会修好**。
   建议把 R2 的根因判定改挂到 P0 上。
2. **建议先做 P0（1 次 GPU，~30 分钟）再决定 K1/K2 的投入**。P0 的产物是把
   "kernel 级是否等价"从断言变成事实，直接决定后面 4–5 人日是否值得花。
3. **R2b 的结构性陷阱**（顺带上报）：`indexer_front_rows` 的 RAW 消费分支用的是
   `m = 1` 专用 kernel，仅由 `debug_assert`（`chain_dev.rs:10786`）保护。
   今天不可达（`qr_raw_r` 只在 R2 的 m==1 臂置位），但**任何把该臂扩到 m>1 的改动
   都会静默跑错程序**。K2 采用"写回归一化行"（§4.6）正是为了彻底删掉这条链。

---

## 附：一句话总结

**R2 的 rope 位置在 m==1 下是对的（`*pos_ctr == pos_rows[0]`），所以"修 rope 位置"修不了它；
本设计换一个轴：不碰 EAGER 的任何 kernel，把 verify 自己的 7 发序列
（`quant_rows` / `proj_mrows`×3 / `norm_rows` / `apply_rope_mrows`）按"同程序、仅搬动"
的方式融成 3 发（K1: wq_a+wkv；K2: norm+quant+wq_b+rope），
位置从 `pos_rows[r]` 显式入参（kernel 内不读 `pos_ctr`、不碰 `s.xq`/`s.xsc`、不带 RAW 旗标），
每一段都以 verify 今日在用的 kernel 为逐位参照。成本 4–5 人日 + 1~2 次 GPU；
**但请先花 1 次 GPU 做 P0，把 R2 的 kernel 级是否等价定成事实。**
