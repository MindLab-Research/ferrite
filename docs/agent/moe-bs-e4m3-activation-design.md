# D2 精度修复设计：fp4 MoE 臂的激活从 **e2m1** 改为 **e4m3**

> 工部 · 2026-09-13。上位：`docs/agent/tcgen05-blockscaled-proto.md`（原型实测）、
> `docs/agent/tilelang-moe-bs-wiring.md`（接线手册）、`kernels/cuda/tilelang_gen/PROVENANCE.md §9`。
> **本文是设计 + 代码分析，不含任何 GPU 操作**（远端生成/验证命令见 §5，留给主 agent）。
>
> **红线**：精度完全对齐官方 DeepSeek-V4.1。官方 `act_quant(x, fp8_block_size=32, ue8m0)`
> 的**激活是 FP8 e4m3**，**权重是 MXFP4 e2m1**。本臂当前把**两侧都当 e2m1**（A 读 packed
> fp4 nibble），激活侧精度低于官方 ⇒ 本设计把 A operand 改成 e4m3。

---

## 0. 结论摘要（TL;DR）

| 项 | 结论 |
|---|---|
| 改动面 | **5 个文件**：AOT 生成脚本（2 行 dtype）、生成物（重新生成，勿手改）、shim（descriptor + gather + scratch + cap 符号）、`device.rs`（cap 探针 + 注释）、`chain_dev.rs`（bs 臂改成**要求** e4m3 激活） |
| MMA 指令 | **不变**：`tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale`（e2m1 的模板特化本来就转发到 e4m3 的那个 asm） |
| 真正变的量 | **指令描述符（idesc）**：`a_format` 5(E2M1) → **0(E4M3)**，`b_format` 保持 5(E2M1) ⇒ 常数 `144709248 (0x08A01680)` → **`144708608 (0x08A01400)`** |
| smem / 占用率 | **完全不变**（A_sh = BM×BK = 16384 B/stage；`kSmem=202752`；stages=6；threads=128） |
| SFA 布局 | **完全不变**（本来就是 per-32 ue8m0、group-major、40 words/row）——任务书里的 4(c) **已经满足** |
| 性能 | A 的**全局读取字节 ×2**（0.5→1 B/value）；每 CTA 每个 k-iter 的操作数 load 量 **+47%**；每次 MoE 调用 +59 MB L2 读 / +11.8 MB DRAM ⇒ 预估**个位数 µs** 量级，必须实测 |
| 最大风险 | TileLang 对 **A/B 混合格式**（a_format≠b_format）的 lowering —— 已在 0.1.14 源码中确认**前端放行 + idesc 分别编码**（§4.5），但必须以「新 dump 的 idesc 常数 == 144708608」为验收闸 |

---

## 1. ① 现状：A operand 的 dtype 与读取方式（file:line）

### 1.1 AOT 生成脚本 `kernels/tilelang/gen_moe_bs_aot.py`

| 行 | 内容 | 说明 |
|---|---|---|
| 240 | `"""grouped block-scaled fp4 (e2m1+ue8m0) up-GEMM ...` | 冻结点：**两侧都 e2m1** |
| 244 | `A : [NSEG*BM, K] float4_e2m1fn packed，段内行已 gather + BM-padding` | 契约文档 |
| **284** | `def main(A: T.Tensor((M, K), T.float4_e2m1fn), W1: ...float4_e2m1fn, W3: ...)` | **A 的 dtype 定义（要改）** |
| **294** | `A_sh = T.alloc_shared((stages, BM, BK), T.float4_e2m1_unpacked)` | **A 的 smem dtype（要改）**；`_unpacked` = 1 B/元素 |
| 295 | `B_sh = ... float4_e2m1_unpacked` | B（权重）**不动** |
| 287 | `SFA: T.Tensor((sf_words * M,), T.uint32)` | 激活标度：40 words/row（per-32 ue8m0，**不动**） |
| 185 / 273 | `GRAN = 32` / `sf_words = K/(gran*4)` = 40 | block=32，**与官方一致** |
| 320-321 | `T.tma_copy(A[by*BM:(by+1)*BM, k*BK:(k+1)*BK], A_sh[st, :, :], barrier=loaded[st])` | A 走 TMA（默认 lowering） |
| 346-354 | `T.tcgen05_gemm_blockscaled(..., sf_a_granularity_k=gran, sf_b_granularity_k=gran)` | gran 双侧 32，**不动** |

### 1.2 生成物 `kernels/cuda/tilelang_gen/moe_bs_up_tl.cu`（GENERATED，勿手改）

| 行 | 内容 | 现状值 |
|---|---|---|
| 23/24 | 参数 0 = `__grid_constant__ const CUtensorMap A_desc` | A 是 TMA 描述符 |
| 26/28 | `A_sh = buf_dyn_shmem + 0`，`B_sh = +98304` | A 每 stage = **16384 B**（= BM×BK×1 B） |
| 87-88 | `expect_transaction(8192)` + `tma_load(A_desc, ..., A_sh + (g%6)*16384, g*128, by*128)` | A **每次搬 8192 B**（= 128 行 × **64 B** packed fp4） |
| 115 | `initialize_tcgen05_descriptor(desc_a, A_sh, 1, 64, 0, 0, 2)` | smem 描述符（K-major / 128B swizzle） |
| 119-122 | `for ki<4: tcgen05mma_blockscaled_ss<tl::DataType::kFloat4_e2m1fn,false>(desc_a+ki*32, desc_b+ki*32, ..., idesc = ((144709248 \| (ki<<29)) \| (ki<<4)), sfa_data+0, sfa_data+4)` | **A 的模板 dtype + idesc** |

### 1.3 shim `kernels/cuda/tilelang_gen/moe_bs_shim.cu`（手写件）

| 行 | 内容 | 现状 |
|---|---|---|
| 273 | `constexpr cuuint32_t kABox = (cuuint32_t)(kBk / 2);` | A/W **共用** box 首维 = 64（字节视图） |
| 283-299 | `spec_a()`：dtype `UINT8`、`gdim[0] = kDim/2`、`gstride[0] = kDim/2`、`box[0] = kABox`、`box[1] = kBm` | A 按**打包字节**描述 |
| 309-328 | `spec_w()`：同上但 rank3、`box[1] = kNh`、`gstride[1] = w_stride` | W 不动 |
| 331-344 | `spec_sfa()`：`UINT32` rank1，`box[0] = kBm`（sf_period==1） | **不动** |
| 347-363 | `spec_sfw()`：`UINT32` rank2 | **不动** |
| 472-477 | `tl_moe_bs_gather_kernel(..., int k2, int nsc, int sf_words, ...)` | `k2` = **dim/2 = 2560** |
| 486-493 | `(a) fp4 nibble`：`adst[i] = asrc[i]`，`i < k2`（纯 memcpy 语义，pad 行写 0） | **A 的 gather（要改）** |
| 494-505 | `(b) 标度`：`tl_bs_f_pow2_to_ue8m0(xsc4[idx][g*4+b])` 装成 u32，`sfa[g*m + row]` | **不动**（已经就是 per-32 ue8m0） |
| 450-456 | `tl_bs_f_pow2_to_ue8m0()`（与 `dsv41_experts_mxf4.cu:287` 逐字同源） | **不动** |
| 583 | `cudaMalloc(&g_a, kSegCap*kBm*(kDim/2))` | A scratch = **11.8 MB** |
| 710-711 / 833-834 | `gather_kernel<<<...>>>(xq4, xsc4, g_a, g_sfa, order, counts, kDim/2, kDim/32, kSfWords, ...)` | 传 `kDim/2` |
| 192-193 | `kGran = 32` / `kSfWords = kDim/(kGran*4)` = 40 | 不动 |
| 210 | `kSmem = 202752` | **不动** |

### 1.4 上游激活生产者（Rust）

| 位置 | 内容 |
|---|---|
| `chain_dev.rs:16839-16859` | `if e4m3 { dsv41_quant_fp8(xn_r, xq4_r, xsc4_r, m, dim, 32, /*round_scale=*/true) } else { quant_fp4(...) }` —— **e4m3 生产者已存在**（`DSV41_EXPERT_ACT_E4M3`） |
| `chain_dev.rs:16495` | `moe_tilelang_bs_ready()` 里 **`!(expert_act_e4m3() && ...)`** —— **当前明确「拒绝 e4m3 激活」**（要反过来） |
| `chain_dev.rs:4847-4848` | `xq4_r = alloc(VERIFY_ROWS*dim)`、`xsc4_r = alloc(VERIFY_ROWS*dim/32+8)` —— **容量已按 e4m3 算**（1 B/value） |
| `dsv41_kernels.cu:3791-3805` | `dsv41_quant_fp8(x, y, scale, rows, cols, block, round_scale, s)`；语义见 `:273-277`（`act_quant` 的 `2^ceil(log2(amax/448))`） |
| `kernels/cuda/dsv41_experts_mxf4.cu:735-800, 1575-1600` | SIMT 回退路径的 `act_e4m3` 分支（`s_act[j] = e4m3_to_f(a[j]) * a_scale[j>>5]`）—— 回退臂**已经会读** e4m3 ⇒ 与 bs 臂格式一致 |

---

## 2. ② 修改方案（逐文件 diff 要点）

### 2.1 `kernels/tilelang/gen_moe_bs_aot.py`（2 处 dtype + 文档 + 审计常量）

```diff
-    def main(A: T.Tensor((M, K), T.float4_e2m1fn),
+    def main(A: T.Tensor((M, K), T.float8_e4m3fn),        # A: fp8 e4m3（官方激活）
             W1: T.Tensor((E_, NP_, K), T.float4_e2m1fn),
             W3: T.Tensor((E_, NP_, K), T.float4_e2m1fn),
@@
-            A_sh = T.alloc_shared((stages, BM, BK), T.float4_e2m1_unpacked)
+            A_sh = T.alloc_shared((stages, BM, BK), T.float8_e4m3fn)
             B_sh = T.alloc_shared((stages, BN, BK), T.float4_e2m1_unpacked)
```

要点：
1. **A 侧不再需要 `_unpacked`**：e4m3 天然 1 B/元素，`float4_e2m1_unpacked` 那个「packed
   global → unpacked smem」的展开路径（`copy_analysis.cc:539`）对 A **整体消失**。
   副产物：shim 的 `VERIFY #1`（box 首维「字节 vs 元素」的二义性）在 A 上**自动消解**
   （1 元素 = 1 字节）。
2. **`A_sh` 的 stage 字节数不变**（BM×BK×1 B = 16384）⇒ `kSmem`/stages/占用率**不需要重调**。
3. `sf_a_granularity_k = gran = 32` **不动**：`kind::mxf8f6f4` 的 SF 粒度**硬要求 K=32**
   （`tcgen5_meta.h:332` 的 `k_size = 0 (K32)`），我们本来就是 32。
4. SFA/B/SFW/C 的声明、k 循环、BK/stages/threads、assert 全部**不动**
   （`assert BK % gran == 0 and BK <= 4*gran` 依旧成立）。
5. 文档/审计文本要同步（防"读文档当现状"）：
   - 头部 §0/§2/§3 的「mxfp4: e2m1 + ue8m0」→「A=e4m3 / B=e2m1 混合 + ue8m0」；
   - `A : [NSEG*BM, K] float4_e2m1fn packed` → `float8_e4m3fn`（**行距 K 字节**，不再是 K/2）；
   - config 文件的 tensormap 段（`:487-489`）与 layout 段（`:497-500`）加
     `A 行距 = K 字节（1 B/value）`；
   - 建议在 config 里**新增两行审计值**（见 §4.5），供 shim/主 agent 机械比对：
     ```
     idesc_blockscaled=144708608 (a_format=0 E4M3, b_format=5 E2M1, K32, E8M0)
     a_tma_bytes_per_k_iter=16384 (BM*BK, was 8192)
     ```

### 2.2 生成物 `kernels/cuda/tilelang_gen/moe_bs_up_tl.cu`（**重新生成，禁止手改**）

预期的 5 处 diff（**审计清单**，逐项必须命中）：

| # | 位置 | 期望 |
|---|---|---|
| 1 | MMA 模板 | `tcgen05mma_blockscaled_ss<tl::DataType::kFloat4_e2m1fn,false>` → **`<tl::DataType::kFloat8_e4m3,false>`**（模板实参取自 **A** 的 dtype，`tcgen05_macro_generator.py:1177` 的 `a_dtype_abbrv`） |
| 2 | idesc 常数 | `144709248` → **`144708608`**（`0x08A01400`；§4.5 的位分解） |
| 3 | A 的 TMA 事务 | `loaded[..].expect_transaction(8192)` → **`16384`** |
| 4 | A smem 描述符 | `initialize_tcgen05_descriptor(desc_a, A_sh, 1, 64, 0, 0, 2)` —— **预期不变**（smem tile 仍是 128 行 × 128 B、128B swizzle；LBO/SBO 值以新 dump 为准，**变了也不算失败，但必须转写进 shim 之外的审计记录**） |
| 5 | A_sh stage 布局 | 偏移 0 / stride 16384 / B_sh 仍在 98304：**全部不变** |

> ⚠️ 若 #2 没有从 `...09248` 变成 `...08608`（例如仍是 5/5 或变成别的值），
> **立刻停下上报**：那意味着 lowering 用单一 dtype 推了 a_format/b_format，
> 必须走「TileLang 源码补丁」的路（与本目录既有的 `vendor/` 模式并列），
> **不许**改生成物常数（生成物是 `DO NOT EDIT`）。

### 2.3 `kernels/cuda/tilelang_gen/moe_bs_shim.cu`

**(a) A 的 box/gdim（`spec_a`，`:273, 283-299`）** —— A 与 W 的 box 首维**从此分家**：

```diff
-// fp4 的「逻辑元素」是 4 bit。TMA 没有 sub-byte 的 data type，所以全局张量按
-// 打包后的字节描述：dim0 = K/2 个 UINT8。
-// ⚠️ VERIFY #1：box 首维是按字节（K/2）还是按元素（K）给。
-constexpr cuuint32_t kABox = (cuuint32_t)(kBk / 2);
+// A（激活）= e4m3：1 元素 = 1 字节，**没有 sub-byte 二义性**（原 VERIFY #1 在 A 上消解）。
+// W（权重）= packed e2m1：仍按打包字节描述（1 行 = K/2 B，box 首维 = BK/2）。
+constexpr cuuint32_t kABoxA = (cuuint32_t)kBk;      // A: BK 字节（= BK 元素）
+constexpr cuuint32_t kABoxW = (cuuint32_t)(kBk / 2); // W: BK/2 字节（packed fp4）
```
```diff
 TmapSpec spec_a(const void* a) {
     ...
     s.dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
     s.rank = 2;
-    s.gdim[0] = (cuuint64_t)(kDim / 2);
+    s.gdim[0] = (cuuint64_t)kDim;            // 5120 字节/行（1 B/value）
     s.gdim[1] = (cuuint64_t)(kSegCap * kBm);
-    s.gstride[0] = (cuuint64_t)(kDim / 2);
-    s.box[0] = kABox;
+    s.gstride[0] = (cuuint64_t)kDim;         // 行距 5120 B
+    s.box[0] = kABoxA;                       // 128 B
     s.box[1] = (cuuint32_t)kBm;
```
`spec_w()` 只把 `kABox` 换成 `kABoxW`（其余不变：`gdim[0]=kDim/2`、`gstride[1]=w_stride`）。

**(b) gather（`:458-506`）**：`k2` 的语义从「打包字节数」变成「行字节数」：

```diff
-    const int live = counts[seg];
+    const int live = counts[seg];
     ...
-    // (a) fp4 nibble（2 值/字节 …）
-    uint8_t* adst = a + (int64_t)row * k2;
+    // (a) e4m3 直读（1 B/value，纯 memcpy；pad 行写 0x00 = +0.0，内核无 mask）
+    uint8_t* adst = a + (int64_t)row * abytes;
     if (idx < 0) {
-        for (int i = threadIdx.x; i < k2; i += kMovThreads) adst[i] = 0;
+        for (int i = threadIdx.x; i < abytes; i += kMovThreads) adst[i] = 0;
     } else {
-        const uint8_t* asrc = xq4 + (int64_t)idx * k2;
-        for (int i = threadIdx.x; i < k2; i += kMovThreads) adst[i] = asrc[i];
+        const uint8_t* asrc = xq4 + (int64_t)idx * abytes;
+        for (int i = threadIdx.x; i < abytes; i += kMovThreads) adst[i] = asrc[i];
     }
     // (b) 标度：**一字不改**（per-32 f32 pow2 → ue8m0，group-major，40 words/row）
```
参数 `int k2` → `int abytes`（两处 launch 传 `kDim`）。`nsc = kDim/32` 与 SFA 循环**不动**。

**(c) scratch（`:583`）**：`kSegCap*kBm*(kDim/2)` → **`kSegCap*kBm*kDim`**（11.8 → 23.6 MB/rank，
可忽略）。`kSmem`、16B 对齐门（`:653-655`）、两条臂的 launch 序列**不动**。

**(d) 能力符号（新，防「旧 .so 静默错值」）**：`xq4` 指针的**语义**变了（fp4 打包 →
e4m3 直排），但 C ABI 形状没变 ⇒ 旧 `.so` 会**静默**把 5120 B 的行当 2560 B 的 fp4 读。
按本项目对 `dsv41_expert_act_e4m3_cap`（`device.rs:7981-7988`）的既有做法，新增：

```c
extern "C" int dsv41_moe_bs_act_e4m3_cap(void) { return 1; }   // A operand = e4m3 激活
```
Rust 侧把它加进 cap 集合（§2.4）。**没有这一条，D2 就是一个静默错值面。**

### 2.4 Rust 接线

| 文件:行 | 改动 |
|---|---|
| `chain_dev.rs:16495` | `&& !(expert_act_e4m3() && self.dev.supports_expert_act_e4m3())` → **`&& expert_act_e4m3() && self.dev.supports_expert_act_e4m3() && self.dev.supports_moe_bs_act_e4m3()`**（bs 臂现在**要求** e4m3 激活） |
| `chain_dev.rs:16453`（`moe_bs_ready` 文档） | 把「the activation must be **PACKED fp4**」改成「must be **e4m3**（1 B/value）」 |
| `chain_dev.rs:16839-16859` | **不改**：`if e4m3 { quant_fp8 }` 分支会自然命中（`e4m3` 局部量已由同一个 gate 定义） |
| `chain_dev.rs:16983-16985` | REFUSED 文案里「an e4m3 activation」这条理由删掉（它现在是**前提**而不是拒绝理由） |
| `device.rs:7958-7960` | `supports_moe_tilelang_bs()` 增加 `self.kernels.moe_bs_act_e4m3_cap.is_some()`；`device.rs:2252` 附近加 `ko!(rt, "dsv41_moe_bs_act_e4m3_cap")` |
| `device.rs:7819-7851 / 7876-7908` | 两个 wrapper 的 doc：`xq4` 从 `[rows*topk][dim/2] u8 fp4 nibbles` → `[rows*topk][dim] u8 **e4m3**` |
| `load.rs` / `weights.rs` | **不动**（权重仍是 fp4，`wsf1/wsf3` 装载期 pack 与 `sha256` 无关） |
| `dspark_dev.rs` | **不受影响**（它不调 bs shim；`xq4` 容量 `bs*dim` 已够 e4m3） |

**关于「谁来决定激活格式」的设计选择（明确取舍）**：
本设计**不**让 bs 臂自己强制 `quant_fp8`，而是**要求**算子已经开了
`DSV41_EXPERT_ACT_E4M3=1`。理由：`xq4_r/xsc4_r` 是 bs 臂与 SIMT 回退臂
（`expert_gate_up_fp4_batched` 的 `act_e4m3` 分支，`dsv41_experts_mxf4.cu:1575+`）**共用**的
存储；让「激活格式」只有一个决定点（`expert_act_e4m3()`），两条臂才不会在回退时读错格式。
bs 臂只做「格式不合就 decline」，符合本项目「armed 但走了老路必须出声」的既有纪律。

### 2.5 明确**不做**的事（边界）

- 不改权重侧（W1/W3 保持 packed e2m1 + ue8m0）；不改 W 的 SF pack（装载期路径零改动）。
- 不改 SFA/SFW 的布局、不改 `gran`、不改 BM/BN/BK/stages/threads/grid。
- 不引入「一个 kernel 同时吃两种激活格式」的双模开关（审计面翻倍，红线只要 e4m3）。
- 不改 down（w2）臂：它的 A 来自 swiglu 后的**自己的**量化，与本设计无关。
- 不手改任何 `GENERATED` 文件。

---

## 3. ③ AOT 重生成命令（**给主 agent**，远端 B300，含无 GPU 的 compile-only）

> 生成只需要 lowering+codegen（`get_kernel_source()`），**不 launch、不碰 GPU 计算**；
> 但必须在**有 tilelang 0.1.14 + nvcc + sm_103a** 的那台机器上跑（远端 `dsv41_venv`）。

```bash
# ---- STEP 0：先把 vendored 源码补丁打上（幂等）----
scp -r kernels/tilelang/vendor      ubuntu@43.202.208.136:~/tl_bs/vendor
scp kernels/tilelang/gen_moe_bs_aot.py ubuntu@43.202.208.136:~/tl_bs/
ssh ubuntu@43.202.208.136 \
  'cd ~/tl_bs && /opt/dlami/nvme/dsv41_venv/bin/python vendor/apply_tilelang_patch.py'
# 判据：tcgen05_gemm_blockscaled.is_tcgen05 : fixed  +  [vendor] verified

# ---- STEP 1：重新生成（BM=128 是唯一可用的认证几何）----
ssh ubuntu@43.202.208.136 \
  'cd ~/tl_bs && mkdir -p aot_e4m3 && /opt/dlami/nvme/dsv41_venv/bin/python \
   gen_moe_bs_aot.py aot_e4m3 --bm 128'
# 判据（config 打印）：BM=128 BN=128 BK=128 NH=64 stages=6 gran=32
#                      sf_words=40 sf_period=1 k_iters=40 grid=(5, 36)
#                      host_source=moe_bs_up_tl_host.cu
# 判据（device dump 的 §2.2 审计清单）：idesc 常数 = 144708608
#                      tcgen05mma_blockscaled_ss<tl::DataType::kFloat8_e4m3, false>
#                      A 的 expect_transaction = 16384
grep -n "144708608\|kFloat8_e4m3\|expect_transaction(16384)" ~/tl_bs/aot_e4m3/moe_bs_up_tl.cu

# ---- STEP 2：回传三份 + banner（banner 由 _banner() 产出，是唯一的本地改动）----
scp ubuntu@43.202.208.136:'~/tl_bs/aot_e4m3/moe_bs_up_tl.cu'      kernels/cuda/tilelang_gen/
scp ubuntu@43.202.208.136:'~/tl_bs/aot_e4m3/moe_bs_up_tl_host.cu' kernels/cuda/tilelang_gen/
scp ubuntu@43.202.208.136:'~/tl_bs/aot_e4m3/moe_bs_tl_config.txt' kernels/cuda/tilelang_gen/
scp ubuntu@43.202.208.136:'~/tl_bs/aot_e4m3/moe_bs_up_tl.banner'  kernels/cuda/tilelang_gen/
# 把 banner 内容贴到 moe_bs_up_tl.cu 头部（与现有形态一致），并核对新 raw sha256

# ---- STEP 3：compile-only（无 GPU，先做，别跳）----
nvcc -arch=sm_103a -cubin -O3 -std=c++17 -I kernels/cuda/tilelang_inc \
     -o /tmp/moe_bs_up_e4m3.cubin kernels/cuda/tilelang_gen/moe_bs_up_tl.cu
nvcc -O3 -std=c++17 -shared -fPIC -arch=sm_103a -I kernels/cuda/tilelang_inc \
     -o /tmp/moe_bs_shim.so kernels/cuda/tilelang_gen/moe_bs_shim.cu
nm -D /tmp/moe_bs_shim.so | grep -E "dsv41_moe_tilelang_gate_up_bs|dsv41_moe_bs_pack_wsf|dsv41_moe_bs_act_e4m3_cap"
nm -D --undefined-only /tmp/moe_bs_shim.so | grep -c cuTensorMapEncodeTiled   # 期望 0

# ---- STEP 4：描述符转写比对（wiring §4.3 / §6.3 的 5 分钟检查）----
grep -n "tensormap_create_tiled" -A 40 kernels/cuda/tilelang_gen/moe_bs_up_tl_host.cu | head -60
#   逐项对齐 shim 的 spec_a / spec_w：**A 这次期望 dump 出现**
#     rank=2, addr=A, gdim=(5120, 4608), stride=(1, 5120), box=(128, 128),
#     estride=(1,1), ilv=0(NONE), swz=3(128B), l2=2(L2_128B), oob=0
#   （现状 = fp4 的 16U4_ALIGN16B 视图：dtype=14, gdim=(5120,4608), stride=(1,2560), box=(128,128)）
#   若 swizzle ≠ 128B 或 box[0] ≠ 128 → 按 dump 改 shim，再回到 STEP 3。

# ---- STEP 5（数值验收，需 GPU；A/B 必须与官方口径比）----
#   官方口径：DSV41_EXPERT_ACT_E4M3=1（act_quant block=32, round_scale）
#   两臂：DSV41_MOE_TILELANG_BS=1（新 bs 臂） vs 不设（SIMT e4m3 路径）
#   判据：bs 臂 ARMED（回执行 "ARMED gate_up_bs ..."），且与参考实现的
#         gate‖up 相对误差落在 e4m3 量化噪声量级（而非 e2m1 量级 —— 后者大 ~8x）
```

---

## 4. ④⑤ 影响与对齐论证

### 4.1 预期性能影响（逐字节算清）

每次调用：`grid = (5, 36) = 180 CTA`，`k_iters = 40`。

| 量 | fp4（现状） | **e4m3（本设计）** | 变化 |
|---|---|---|---|
| A 每个 k-iter 每 CTA 的全局读 | 8192 B（128 行 × 64 B） | **16384 B**（128 行 × 128 B） | **×2** |
| W（w1 半块 + w3 半块）每 k-iter | 8192 B | 8192 B | 0 |
| SFA + SFW 每 k-iter | 512 + 512 B | 512 + 512 B | 0 |
| 每 k-iter 操作数总读 | 17408 B | **25600 B** | **+47%** |
| 每 CTA 全 K 读 | 696 KB | **1.02 MB** | +327 KB |
| 每次调用 L2 侧操作数读 | ~125 MB | **~184 MB** | **+59 MB** |
| 唯一 A（DRAM，L2 吸收 5 个 N-tile 的复用） | 11.8 MB | **23.6 MB** | +11.8 MB |
| gather 的 A 拷贝 | 11.8 MB | 23.6 MB | +11.8 MB |
| `g_a` scratch | 11.8 MB | 23.6 MB | +11.8 MB |
| smem / stages / 占用率 / grid | — | **完全不变** | 0 |

- 结论：**激活读取带宽 ×2（字节级）**，但激活在总 load 量里只占一半左右（权重也是 packed fp4
  的一半），所以 **CTA 级 load 量只 +47%**；DRAM 侧唯一 A 只 +11.8 MB/次。
  按原型实测口径（up-GEMM contended 72.6 µs、静默折算 ≈28 µs），预估**个位数 µs** 的回归
  （L2 +59 MB @ ~10-20 TB/s ≈ 3-6 µs，DRAM +11.8 MB @ ~5-6 TB/s ≈ 2 µs，部分可重叠）。
- **必须实测**（本设计不承诺数字）：这也是「精度红线 vs 带宽」的取舍——红线在前。
- 附带成本：`dsv41_quant_fp8` 取代 `quant_fp4`（同为 band-limited 量化，量级相同）；
  gather 的拷贝翻倍（118 MB→236 MB 的搬运，仍是 µs 级）。

### 4.2 与官方的精度对齐论证

| 环节 | 官方 DeepSeek-V4.1 | 本设计 | 是否逐位一致 |
|---|---|---|---|
| 激活量化 | `act_quant(x, fp8_block_size=32, ue8m0)` → **e4m3 值 + per-32 幂次标度** | `dsv41_quant_fp8(..., block=32, round_scale=1)`（`dsv41_kernels.cu:3791`，语义注释 `:273-277`：`2^ceil(log2(amax/448))`） | ✅ 同一方案（该路径已是参考对齐路径，见 `tests_dsv41_gemm_fp8.cu`） |
| 激活数据位宽 | e4m3（1 B/value） | **e4m3（1 B/value）** | ✅ |
| 激活标度粒度 | block = 32 | `gran=32`（`sf_words=40 = 5120/128`，一个 u32 = 4×32K） | ✅ |
| 标度类型 | **ue8m0** | gather 把 f32 幂次 → ue8m0 字节（`f_pow2_to_ue8m0`，与 `dsv41_experts_mxf4.cu:287` 逐字同源） | ✅（f32 幂次 → ue8m0 是**无损**位转换） |
| 权重 | MXFP4 e2m1 + ue8m0, block=32 | packed e2m1 + `SFW`（装载期 pack，**不动**） | ✅ |
| MMA | fp8 激活 × fp4 权重，block-scale | `tcgen05.mma.kind::mxf8f6f4.block_scale`，**a_format=0(E4M3) / b_format=5(E2M1) / scale_format=1(E8M0)** | ✅ 硬件路径同一（`tcgen5_meta.h:284-334`） |
| 累加 | f32 | f32（`C_tmem`/`C` 均 f32） | ✅（仅**求和顺序**不同 ⇒ 1e-7 量级） |
| pad 行 | n/a | e4m3 `0x00` = **+0.0**，SFA=0 ⇒ 贡献恰为 0（内核无 mask，故必须真 0） | ✅ |

⇒ 本设计之后，**gate‖up 的输入精度（激活 8 bit / 权重 4 bit / 标度 per-32 ue8m0）与官方完全同构**，
剩下的唯一差异是 tensor-core 的累加顺序（不是精度损失）。当前 e2m1 激活与官方差着**整个 4 bit 位宽**
（这正是一开始被记为 D2 的那条）。

### 4.3 风险与检查点

| # | 风险 | 判据 / 处置 |
|---|---|---|
| R1 | TileLang 是否允许 **A/B 混合格式** 的 blockscaled | **已查证 0.1.14 源码**：`dtypes.py:308-335`（`is_f8f6f4_family` 对 e4m3 与 fp4 都 True）+ `cuda/op/gemm/gemm_tcgen05.py:53-55`（`allow_f8f6f4_mixed_dtypes = True`）⇒ **前端放行**；idesc 由 `GetTCGEN5BlockScaledInstrDesc(a_dtype, b_dtype, ...)` **分别编码** a_format / b_format（`tcgen5_meta.h:307-323`）⇒ **可表达**。**验收=新 dump 的 idesc == 144708608**（§2.2 #2）；不符则回报，走 vendor 源码补丁，**不许改生成物** |
| R2 | A 的 smem swizzle 可能随 elem_bits 4→8 变化 | 以新 `moe_bs_up_tl_host.cu` 的 A descriptor 为准（STEP 4）；shim 的 `kSwzAB` 预期仍是 128B（A 行 = BK = 128 B，两态相同） |
| R3 | **旧 .so 静默错值**（`xq4` 语义变了但 ABI 形状没变） | 新增 `dsv41_moe_bs_act_e4m3_cap` + Rust cap 探针（§2.3d/§2.4）。**这是硬要求** |
| R4 | 官方 vs 本臂的 A/B 数字差异被误读 | 验收必须开 `DSV41_EXPERT_ACT_E4M3=1` 作为参考臂（否则比的是 e2m1 老路） |
| R5 | SFA 的 SF-ID 循环（`(k/32)%4` → `ki<<29`） | **不受影响**：gran/micro_size_k 都是 32，sf_period=1，A 与 B 的 SF 覆盖格点相同 |
| R6 | 性能回归被忽略 | 每次实测都报 up-GEMM 的 µs；预期个位数 µs，若显著更大要回到 §4.1 复核 A 的 load 是否成为瓶颈 |

### 4.4 交付物清单（工部视角）

| 文件 | 性质 |
|---|---|
| `kernels/tilelang/gen_moe_bs_aot.py` | 手改（2 行 dtype + 文档/审计文本） |
| `kernels/cuda/tilelang_gen/moe_bs_up_tl.cu` + `_host.cu` + `moe_bs_tl_config.txt` | **生成物**（远端重生成后回传，禁止手改） |
| `kernels/cuda/tilelang_gen/moe_bs_shim.cu` | 手改（spec_a / gather / scratch / cap 符号 / 文档） |
| `crates/ferrite-models/src/dsv41/device.rs` | 手改（cap 探针 + `ko!` 注册 + 注释） |
| `crates/ferrite-models/src/dsv41/chain_dev.rs` | 手改（bs 臂 gate 反转 + 文案） |
| 本文件 | 设计记录 |

> 复述纪律：本文**不含**任何 GPU 操作；§3 的命令清单是给主 agent 的。生成物与 shim 的常数
> 必须**机械比对**（§2.2 / STEP 4），不凭推断。
