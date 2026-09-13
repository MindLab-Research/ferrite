# TileLang MoE grouped GEMM 原型（工部 · 工程实现 + 远端实测）

> 目标：用 tensor core 打破 ferrite verify MoE 的 **36-sweep FMA 恒等式**。
> 老 kernel 是 SIMT FMA-issue bound（36 sweep = 物理下限，40 层 ≈ 10ms/步 ⇒ **250µs/层**）；
> SGLang 的 Triton fused MoE 靠 `moe_align_block_size` + `tl.dot`(MMA) + swapAB 拿到 1.2–1.3× eager。
> 本文档交付 **TileLang 侧 grouped GEMM 原型**（远端 B300 sm_103a 实测）+ **判据结论** + **AOT/数值形态**。
>
> 工部 · 2026-09-13 · 远端 `ssh ubuntu@43.202.208.136`，TileLang 0.1.14 + CUDA 13.2 + B300 (sm_103a)。
> 全部数字来自 GPU micro-bench（非 e2e，未加载真实权重）。

---

## 0. 一句话结论（先看这个）

**成立——但成立的形式不是"fp4 原地 dequant 的 grouped GEMM"。**

| 臂 | up (N=640,K=5120) | down (N=5120,K=320) | **合计/层** | vs SIMT 250µs |
|---|---|---|---|---|
| **TileLang grouped MMA，权重 bf16** | 45.5 µs | 24.5 µs | **70.1 µs** | **28.0%** ✅ |
| TileLang grouped MMA，权重 fp4(e2m1+ue8m0) 原地 dequant | 159.8 µs | 55.7 µs | 215.5 µs | 86.2% ❌ |
| ↑ 同上但**去掉 dequant ALU**（纯 uint8→bf16 拓宽，只留结构） | 102.8 µs | — | — | — |
| SIMT 36-sweep 代理（同框架、朴素 fp4 GEMV，up 路径） | 1159.9 µs | — | — | 464% |

**判据（grouped GEMM expert 段总时间 < SIMT 36 sweep 的 40%=100µs/层）：bf16 臂 70.1µs = 28.0% → 通过。**

**关键负结果（比正结果更重要）**：fp4 原地 dequant **不是** bandwidth-bound。fp4 权重只读 57MB（bf16 是 229MB，4×），
但把 114.7M 个 fp4 展开成 bf16 的 **shared-memory 扩展流水**（`Bq → LUT → B_sh → ldmatrix/MMA`）本身就比"多读 4× 字节的 bf16"更贵
（结构地板 102.8µs > bf16 全量 45.5µs）。⇒ **本 GPU/本形状下，fp4 必须走原生 blockscaled MMA（tcgen05），不能走"读 fp4 展开成 bf16"**。

---

## 1. SGLang fused_moe_triton 读解（语义基线）

源码：`/tmp/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/{fused_moe.py,moe_align_block_size.py}`、
`/tmp/sglang/python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py`。

### 1.1 `moe_align_block_size` 的分组语义（本文档的布局契约来源）

- 输入 `topk_ids [num_tokens, top_k]`，扁平化后按 **expert 升序**（同 expert 内保持原顺序）排序；
- 每个 expert 分到的 token 数 **向上取整到 `BLOCK_SIZE_M` 的整数倍**（尾部补 padding token，输出端被 mask 掉）；
- 产出三元组：`sorted_token_ids`（已排序的 token 下标，含 padding）、`expert_ids`（**每个 M-block 属于哪个 expert**）、
  `num_tokens_post_padded`。
- 核心作用：让"同一个 expert 的 token"成为 **连续行块** ⇒ 每个 expert 的一整块 operand 可以喂给一次 dense block-GEMM
  （expert-centric），从而把 per-assignment 的小 GEMM 摊薄成 MMA。
- 上限：`max_num_tokens_padded = numel + (num_experts+1)*(block_size-1)`（或 `numel*block_size` 当 `numel < E+1`）。
- **确定性**：纯排序/kernel 内串行扫描，无 atomic，不依赖 block 调度 ⇒ CUDA-graph replay 逐位可复现。
- small-batch 快路径：`topk_ids.numel() <= SMALL_NUMEL_LIMIT` 且 `num_experts+1 > 64` 时走单 CTA 的 `moe_align_small_numel`。

### 1.2 BLOCK 选择（`fused_moe_triton_config.py`）

`try_get_optimal_moe_config(w1_shape, w2_shape, top_k, dtype, M, ...)`：按 (E,N,K,top_k,dtype,M) 查/回退，
`M <= E` （即小 batch）走小 block。几个代表值：

| dtype | M 关系 | BLOCK_M | BLOCK_N | BLOCK_K | GROUP_SIZE_M | num_warps | num_stages |
|---|---|---|---|---|---|---|---|
| fp8_w8a8 | M > E | 64 | 128 | 256 | 64 | 4 | 2 |
| fp8_w8a8 | `M <= E` | 64 | 128 | 128 | 1 | 4 | 4 |
| 通用/bf16 回退 | 默认 | 64 | 64 | 32 | 8 | — | — |
| 通用/bf16 回退 | `M <= E` | **16** | **32** | 64 | 1 | — | — |

**重要约束**：up / down 两个 kernel **共用同一次 `moe_align_block_size` 排序**，所以 `down_config["BLOCK_SIZE_M"]` 会被
**强制覆盖成 up 的 BLOCK_SIZE_M**（不一致时告警 + override，`:324-339`）。

### 1.3 swapAB 条件（原文，`fused_moe_triton_kernels.py:60-69`）

```python
@functools.lru_cache(maxsize=8)
def should_enable_swap_ab(BLOCK_SIZE_M: int, BLOCK_SIZE_N: int) -> bool:
    if not _is_cuda or is_batch_invariant_mode_enabled():
        return False
    return is_sm90_supported() and BLOCK_SIZE_M < 64 and BLOCK_SIZE_N >= 64
```

- 语义：accumulator 变成 `[BLOCK_N, BLOCK_M]`，循环内 `a,b = tl.trans(b,(1,0)), tl.trans(a,(1,0))`，结尾再 `tl.trans` 回来
  （`:539-606`）。作用是把**小的 M 塞进 MMA 的 N 维**，让 MMA 的 M 维是那个大的 N（640 / 5120）。
- **只在 sm90 启用**。B300 是 **sm_103a ⇒ swap_ab 关掉**。实测也一致：同配置下 swap=1 反而略慢
  （`BN=256 BK=128 th=256 stg=3`：swap0 **154.1µs** vs swap1 161.7µs；`BN=128 BK=256`：158.3 vs 164.4）。
  ⇒ **本原型不做 swapAB**，M 直接 pad 到 16 交给 `mma.m16n8k16`。

---

## 2. 复刻设计（ferrite verify MoE 形状）

形状：`dim=5120`、`inter_local=320`、`n_routed=384`、`m=6 行 × topk=6 = 36 assignments/层`。

| 路径 | GEMM | N | K | 每 assignment 权重 |
|---|---|---|---|---|
| up（gate‖up） | `[M_e, 5120] × [5120, 640]` | 640 | 5120 | w1‖w3 fp4 = 1.74MB |
| down | `[M_e, 320] × [320, 5120]` | 5120 | 320 | w2 fp4 = 0.87MB |

### 2.1 host 侧 moe_align（交付项 ③）

`kernels/tilelang/moe_grouped_proto.py::moe_align()` — 与 SGLang 同语义的 host 实现：

```python
flat   = topk_ids.reshape(-1)                     # [36] assignment -> expert
order  = torch.argsort(flat, stable=True)         # (expert 升序, assignment 下标升序)
uniq, counts = torch.unique_consecutive(flat[order], return_counts=True)
seg_start = cumsum(counts) - counts               # 排他前缀和
# 产出：order / seg_expert[nseg] / seg_start[nseg] / counts[nseg] / nseg
```

性质（与 `docs/agent/grouped-routing-design.md` 的布局契约一致）：
1. expert `e` 的 assignments 占据**连续区间** `[seg_start[e], seg_start[e]+counts[e])` ⇒ 一段连续 dense 行块；
2. 纯 `route_idx` 的函数，**无 atomic、不依赖 block 调度** ⇒ graph replay 逐位可复现；
3. 实测本形状：`nseg=35`（36 assignments 里只有 35 个不同 expert），`counts ∈ {1,2}` ⇒ **每段 M_e ≤ 2**。

> M_e ≤ 2 是这条路的形状本质：MMA 最小 M=16 ⇒ **BM=16 padding 浪费 ≥8×**。
> 但这条路的瓶颈不是 MMA throughput（见 §5），所以 padding 浪费可以接受。

### 2.2 grouped GEMM kernel（交付项 ①）

grid = `(ceil(N/BN), nseg)`，每个 block 认领一个 `(expert 段, N-tile)`：

```
e   = Eid[by]                                  # 该段的 expert id
for k in Pipelined(K/BK, stages):
    T.copy(A[by*BM, k*BK], A_sh)               # 该段的行（已按 expert 收拢 + BM padding）
    T.copy(W[e, bx*BN, k*BK], B_sh)            # 该 expert 的权重 tile
    T.gemm(A_sh, B_sh, C_l, transpose_B=True)  # C[M=16, N] = A @ W^T
T.copy(C_l, C[by*BM, bx*BN])
```

- `A` 是 **已 gather + BM-pad 的激活**（`[nseg*BM, K]`），`Eid` 是 per-segment expert 表 ⇒ 内核里没有 gather、没有 masks、
  没有 atomic；gather/scatter 在 host（或前置 kernel）完成，正是 §1.1 的布局契约。
- **每段权重只读一次**（`nseg` 次，不是 36 次重复）：这是"摊薄"的全部来源。

---

## 3. 实测结果

环境：B300 SXM6（275GB），driver 595.91.07，CUDA 13.2，`-arch=sm_103a`（TileLang 自动），TileLang 0.1.14。
计时：CUDA event 之外用 `perf_counter` 包 400 次 launch + `synchronize`（含 launch 开销，故为保守估计）。

### 3.1 主表

| 臂 | 配置 | up µs | dn µs | 合计 | 判据(<100µs) |
|---|---|---|---|---|---|
| **bf16 grouped MMA** | up `BN=256,BK=64,th=256,stg=3` / dn `BN=512,BK=64,th=256,stg=2` | **45.5** | **24.5** | **70.1** | ✅ 70% 冗余 |
| fp4 grouped（pair-LUT 原地 dequant） | up `BN=256,BK=64,th=256,stg=3` / dn `BN=256,BK=64,th=256,stg=2` | 159.8 | 55.7 | 215.5 | ❌ |

配置扫过的点（up 路径，fp4 与 bf16 同表对照）：

| BN | BK | th | stg | bf16 µs | fp4-LUT µs |
|---|---|---|---|---|---|
| 128 | 128 | 128 | 3 | 49.2 | 276.8 |
| **256** | **64** | **256** | **3** | **45.4** | 159.8 |
| 64 | 128 | 128 | 3 | 53.9 | 215.1 |

down 路径：bf16 `BN=512,BK=64,th=256,stg=2` → **24.9µs**（`BN=128` 33.0 / `BN=256` 28.3）。

### 3.2 与 SIMT 基线的交叉验证

- 文档基线（用户给定）：**250µs/层**（40 层均摊，SIMT FMA issue-bound）。
- 我在同一张卡上搭了一个**朴素 SIMT 36-sweep 代理**（`k_simt_fp4`：一个 block 管一个 assignment，per-output 串行 K 循环 + fp4 dequant，无 tensor core）：
  **1159.9 µs/层（up 路径）**。朴素实现比量产 SIMT 慢 ~4.6×，量级自洽 ⇒ **250µs 是可用的对照基线**（且我的对照是"朴素 SIMT 更慢"，不是"我更强"）。

### 3.3 判据

```
判据：grouped GEMM 的 expert 段总时间 < SIMT 36 sweep 的 40%
      bf16 臂 70.1µs  <  0.40 × 250µs = 100µs      ✅（实测 28.0%）
      fp4  臂 215.5µs >  100µs                     ❌（实测 86.2%）
```

---

## 4. fp4 路径：正确，但被"扩展流水"卡死（负结果 + 根因）

### 4.1 数值上完全正确

- **pair-LUT 版（最终采用的 fp4 实现）**：`err = 0.00000`，与 **fp4-roundtrip 参考逐位一致**（不是"近似一致"）。
- 先试过的两条更差的写法（记录以备后来人跳过）：
  - **bit-trick**（`pos = (e==0)? m*0x3F00 : ((e+126)<<7)|(m<<6)`，纯整型构造 bf16，无 transcendental）：
    正确，最好 154.1µs；
  - **shared 16-entry LUT 逐元素查表**：**错值（inf）+ 更慢（237µs）**——shared 动态索引 + 逐元素路径不可用。
- 最终 pair-LUT：**一个 packed fp4 byte（2 个权重）→ 一次 `Lut[byte]`（uint32，含 2 个 bf16）**，再按 ue8m0
  用**指数位移**（`<<7`）代替浮点乘、用 `p & 0xFFFF / p >> 16` 取上下半 ⇒ 每 2 个权重 ≈ 1 LDS + 4 ALU + 1 STS。

### 4.2 根因：瓶颈不是带宽，是 fp4→bf16 的 shared 扩展

| 量 | fp4 | bf16 |
|---|---|---|
| 每层 up 权重字节 | 35 × 640×5120×0.5 = **57MB** | **229MB** |
| 理论带宽地板 @8TB/s | ~7µs | ~29µs |
| 实测 | **159.8µs**（含 dequant） | **45.5µs** |
| 实测（**去掉 dequant ALU**，只留 `uint8→bf16` 拓宽 + smem 往返 + MMA） | **102.8µs** | — |

**去掉 dequant ALU 后 fp4 仍然 102.8µs > bf16 45.5µs**。也就是说：
把 114.7M 个权重元素从 fp4 展开成 bf16、写回 shared、再被 `ldmatrix` 吃掉的这套流水，
**比直接多读 4× 字节的 bf16 更贵**（结构地板就超了）。bf16 臂 45.5µs 对应 ~5.0TB/s ≈ 峰值的 60%，是健康的内存受限；
fp4 臂则完全不在带宽曲线上。

### 4.3 由此得到的工程结论

1. **不要**做"读 fp4 → 内核里展开成 bf16 → MMA"（无论 LUT 还是 bit-trick，无论 swapAB 与否）。
2. 想真正吃到 fp4 的 4× 带宽，必须走 **原生 blockscaled fp4 MMA**：TileLang 0.1.14 已有
   `T.tcgen05_gemm_blockscaled(A, B, C, SFA_tmem, SFB_tmem, k_start=…, sf_a_granularity_k=…, sf_b_granularity_k=…)`
   + `T.make_blockscaled_gemm_layout(C, A, transpose_A)`（`language/gemm_op.py:297/580`），配
   `T.alloc_tmem` / `T.alloc_barrier` / `T.mbarrier_wait_parity`。这是明确的后续项（见 §8）。
3. 若暂时不走 tcgen05：**把 MoE 专家权重在加载期一次性 dequant 成 bf16 存储**，per-step 直接跑 bf16 grouped GEMM ⇒ 70.1µs/层。
   代价是权重常驻显存 4×（fp4 40GB → bf16 160GB，TP8 下 5GB→20GB/rank，275GB/card 可容纳）。

---

## 5. 附：为什么"摊薄"是真的（不是 padding 假象）

M_e ≤ 2 却 pad 到 BM=16 ⇒ MMA 的 M 维**浪费 ≥8×**。但这条路不靠 MMA throughput：

- **bf16 臂 45.5µs 的带宽账**：35 段 × 640×5120×2B = 229MB / 45.5µs = **5.0 TB/s**（B300 峰值 ~8TB/s，60%）⇒ **内存受限，不是计算受限**。
- 老 SIMT 是 **FMA-issue bound**（每元素 1 条 FMA + dequant 指令，m=1 无 reuse）；MMA 把 640×N_tile×BK 的乘加压进
  `mma.sync.m16n8k16` 一条指令，**指令数不再是瓶颈**，于是掉到带宽墙上。
- 所以真正的收益来源是 **"每段权重只读一次 + 用 MMA 替掉 FMA 指令流"**，padding 的 8× 计算浪费被带宽余量吸收。

---

## 6. AOT 产物形态（交付项 ④）

导出：`Kernel.export_library(path)`（`tilelang/jit/kernel.py:738`）+ `Kernel.get_kernel_source()`。
落盘：`kernels/tilelang/aot/{up,dn}_{bf16,fp4}.{cu,so}`（.cu = device source，.so = 可加载 host wrapper）。

`up_bf16.cu` 的实际形态（89 行 device 代码）：

```cuda
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(
    __grid_constant__ const CUtensorMap A_desc, float* __restrict__ C,
    const int* __restrict__ Eid, __grid_constant__ const CUtensorMap W_desc)
```

| 特征 | 值 |
|---|---|
| 线程/CTA | **384**（warp-specialized：`threadIdx.x < 128` = producer warpgroup，其余 256 = consumer） |
| 寄存器调度 | producer `warpgroup_reg_dealloc<24>()`，consumer `warpgroup_reg_alloc<240>()` |
| 数据搬运 | **TMA**（`CUtensorMap` descriptor + `tl::tma_load`），not cp.async |
| 同步 | `mbarrier`（6 个，3-stage 生产者/消费者 parity） |
| 乘法指令 | **`tl::mma_sync` m16n8k16 bf16×bf16→fp32** + `ldmatrix.x4` |
| 动态 smem | `B_sh` 3×32768B = 98304B + `A_sh` 3×2048B ⇒ **~102KB/CTA** |
| 网格常量 | `Eid[blockIdx.y]` ⇒ 每段查一次 expert，权重 tile TMA 的第三个坐标就是 `e` |

**注意（留给下一轮的 headroom）**：TileLang 0.1.14 在 sm_103a 上把 `T.gemm` lower 成 **`mma.sync`（sm80 风格）**，
**没有**用 `wgmma`/`tcgen05`（`grep tcgen05` = 0）。B300 的 tcgen05 路径（尤其 blockscaled fp4）尚未被这条 lowering 用上。

`~/.tilelang/cache/0.1.14/` 下同时留有 `cuda-binaries/*.cubin`（裸 cubin）与 `linux-x86_64/*.so`（JIT 加载产物）。

---

## 7. 数值形态（交付项 ④ 之二）

| 项 | bf16 臂 | fp4 臂（pair-LUT） |
|---|---|---|
| 累加器 | fp32（`T.alloc_fragment((BM,BN), "float32")`） | 同左 |
| up `max\|err\|` (vs fp32 参考) | **2.0e-5** | **0.0**（vs fp4-roundtrip 参考，逐位） |
| down `max\|err\|` | **0.0** | **0.0** |
| 误差来源 | 仅 bf16 输入量化 | 仅 fp4 输入量化（内核 dequant 本身无损） |
| 权重布局 | `W[e, N, K]` bf16（K 连续） | `Wq[e, N, K/2]` uint8（**低 nibble = 偶数 k**）+ `Ws[e, N, K/32]` ue8m0；scale 语义 `2^(byte-127)` |
| 激活布局 | `A[assignment, K]` bf16，收拢到 `[nseg*BM, K]` + 尾部补零 | 同左 |
| 输出布局 | `C[nseg*BM, N]` fp32（段内顺序 = expert-grouped 顺序，需要一次 scatter 回 `(row, slot)`） | 同左 |

> fp4 的 `max|err| = 0.0` 说明：**内核里的 dequant 与参考实现在位级等价**；误差只来自 fp4 量化本身，与 grouped 结构无关。

---

## 8. 复现步骤

```bash
# 1) 上传原型（本仓库）
scp kernels/tilelang/moe_grouped_proto.py ubuntu@43.202.208.136:~/tl_proj/
# 2) 远端跑（B300，无需加载真实权重；随机权重 + fp4 打包）
ssh ubuntu@43.202.208.136 'cd ~/tl_proj && python3 moe_grouped_proto.py'
# 3) 产物：~/tl_proj/aot/{up,dn}_{bf16,fp4}.{cu,so}（本仓库 kernels/tilelang/aot/ 已放一份拷贝）
```

脚本内容：host `moe_align` → up/down 两条 grouped kernel（bf16 + fp4 两臂）→ 正确性对照 → 计时表 → SIMT 代理 → AOT 导出。

---

## 9. 交付项对照 + 下一步

| 验收项 | 状态 | 位置 |
|---|---|---|
| ① TileLang grouped GEMM 原型（远端实测） | ✅ | `kernels/tilelang/moe_grouped_proto.py`（§2/§3） |
| ② vs SIMT 比值（<40% 判据） | ✅ bf16 28.0%（fp4 86.2% 为负结果） | §3.3 |
| ③ moe_align 分组实现 | ✅ host 侧（TileLang 内核侧免 gather，只认 per-segment `Eid`） | §2.1 |
| ④ AOT 产物形态 + 数值形态 | ✅ | §6 / §7 |
| ⑤ 本文档 | ✅ | 本文件 |

**下一步（按 ROI 排序）**

1. **原生 tcgen05 blockscaled fp4 MMA**（拿回 4× 带宽）：`T.tcgen05_gemm_blockscaled` + `T.make_blockscaled_gemm_layout`
   + `T.alloc_tmem` + mbarrier。预期 up 从 159.8µs 掉到 ~15-25µs 量级（fp4 57MB 的带宽地板 ~7µs）。
2. **bf16 落地的工程化**：加载期把专家权重 dequant 成 bf16（§4.3-3），per-step 直接跑本原型的 bf16 臂 ⇒ 直接吃 70.1µs/层。
   需要与拥有者确认显存预算（TP8 下 +15GB/rank）。
3. **把 `nseg` 摊到 GPU 侧**：当前 host `moe_align` 每层一次（36 个元素），若要进 CUDA graph 需上 device（SGLang 的
   `moe_align_block_size` CUDA kernel 可直接借用语义）。
4. **down 路径的反向 tile**：down 是 `N=5120, K=320`（K 很短），`BN=512` 最好；若将来能在 up/down 间共享 A/B staging，
   可再省一次 smem 往返。
