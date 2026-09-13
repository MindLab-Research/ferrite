# tcgen05 block-scaled fp4 MoE 原型（工部 · 工程实现 + 远端实测）

> 承接 `docs/agent/tilelang-moe-grouped.md` §8「下一步 1」：那个文档把 **fp4 原地 dequant 判死**
> （159.8µs，结构地板 102.8µs > bf16 全量 45.5µs），并给出 headroom 假设——
> **B300 原生 blockscaled MMA（e2m1 数据 + ue8m0 scale 直接进 tensor core，无需 dequant）
> 能把 up 从 160µs 拉到 15-25µs**。
> 本文档就是对这个假设的**实测判决**：API 可用性 ✅、数值逐位精确 ✅、**速度 ❌（未达 15-25µs）**，
> 并给出根因（不是 tensor core、不是 scale 路径，是每 CTA 的 TMA 吞吐墙 ~20 GB/s）。
>
> 工部 · 2026-09-13 · 远端 `ssh ubuntu@43.202.208.136`，TileLang 0.1.14 + CUDA 13.2 + B300 sm_103a（148 SM）。
> **⚠️ 计时口径警告（必读）**：本次所有 GPU 测量期间，机器上有 co-tenant
> （`ferrite-serve --tp 8`，8 卡 100% util / 78.8GB/card，04:07:14 启动）。
> 同刻标定：D2D copy 1GB = **2.39 TB/s**、bf16 dense 8192³ = **627 TFLOPS**
> ⇒ 机器被拖慢约 **2.6×**。因此本文档**只用「同一时刻背靠背测得的三臂比值」做判据**，
> 绝对 µs 一律标注「(contended)」。静默会话的绝对值留给主 agent 复测。

---

## 0. 一句话结论

**API 完全支持「e2m1 + ue8m0 原生 fp4 blockscaled」，且数值逐位精确；但 TileLang 0.1.14 的
`T.tcgen05_gemm_blockscaled` 有一处注解缺陷会直接编译失败；修掉之后性能没到 15-25µs，
而是比 bf16 grouped MMA 快 1.6×（同刻 0.62×），瓶颈定位在「每 CTA ~20 GB/s 的 TMA 管道」，
不是 tensor core。**

| 臂（同刻背靠背，contended box） | up µs | vs bf16 |
|---|---|---|
| bf16 grouped MMA（`T.gemm` → `mma.sync.m16n8k16`） | 116.8 | 1.00× |
| fp4 原地 dequant + `mma.sync`（旧臂） | 416.5 | 3.57× |
| **fp4 tcgen05 block-scaled（BN=128, BK=128, stg=6, gran=32）** | **72.6** | **0.62×** ✅ |
| ↑ stg=4 | 75.0 | 0.64× |
| ↑ gran=128（粗粒度 scale 对照） | 74.2 | 0.64× |

> 静默会话（上一会话、无 co-tenant）的 bf16 臂 = **45.5µs**、旧 fp4 臂 = **159.8µs**。
> 按同刻比值折算，blockscaled 的静默绝对值 ≈ 72.6 / 2.57 ≈ **28µs** —— 落在 15-25µs 的同一个量级，
> 但**这是折算值不是实测值**，需主 agent 在空卡上复测确认。

判据（<50µs）：**contended 口径下 72.6µs 不通过；折算口径 ~28µs 通过。裁决留给空卡复测。**

---

## 1. ① API 可用性判定（e2m1 + ue8m0 是否直接支持）

### 1.1 结论：**是，直接支持，无需任何 dtype 妥协**

判定链（三层互相印证）：

1. **指令描述符编码**（`src/op/tcgen5_meta.h::GetTCGEN5BlockScaledInstrDesc`）：
   ```cpp
   } else if (dtype.is_float4()) {
     return 5u; // E2M1 (packed or f8f6f4/mxf8f6f4 unpacked storage)
   }
   ...
   desc |= set_bits(1, 23, 1);   // scale_format = 1 (E8M0)   ← 硬编码
   ```
   `a_format = b_format = 5 (E2M1)`、`scale_format = E8M0` ⇒ **e2m1 + ue8m0 是这条 ISA 的原生组合**。
   （可选 dtype：E4M3=0 / E5M2=1 / E2M3=3 / E3M2=4 / **E2M1=5**；scale 固定 E8M0。）

2. **lowering 落地**（`cuda/intrinsics/macro/tcgen05_macro_generator.py:537` 起、
   `cuda/op/gemm/gemm_tcgen05.py::_lower_blockscaled`）：生成的 CUDA 实测长这样
   （`moe_bs_dense.cu`，本仓库可复现）：
   ```cuda
   tl::tcgen05mma_blockscaled_ss<tl::DataType::kFloat4_e2m1fn, false>(
       uint64_t(desc_a + (ki * 32)), uint64_t(desc_b + (ki * 32)),
       (*reinterpret_cast<uint32_t*>(C_tmem)) + 0,
       ((0 < ki) ? 1 : ((k == 0) ? 0 : 1)),          // 累加使能
       static_cast<uint32_t>(((144709248 | (ki << 29)) | (ki << 4))),  // idesc: a_sf_id/b_sf_id
       (*reinterpret_cast<uint32_t*>(sfa_data)) + 0,
       (*reinterpret_cast<uint32_t*>(sfa_data)) + 4);
   ```
   即 PTX `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale`，**A/B 都是 e2m1**。
   `_initialize_k_dim` 对 `bits==4` 取 `k_dim = 32`（MXFP4 的 K-atom）也自动生效。

3. **实测**：所有编译成功的配置下 `max|err| = 0.00000`（见 §3）⇒ 不是「能跑但算错」。

**⇒ 不存在「需要退化成 e4m3+scale」的 gap。** 最近支持形态 = 我们跑的就是原生形态。

### 1.2 ⚠️ 0.1.14 的 API 缺陷：`tcgen05_gemm_blockscaled` 漏设 `is_tcgen05`（必须先打补丁）

直接调用会硬失败（这是我第一次跑原型就撞上的）：

```
tvm.error.InternalError: T.mma_gemm_blockscaled() requires an SM120 CUDA target,
    but got target={"kind":"cuda",...,"arch":"sm_103a"}
```

根因：`language/gemm_op.py` 里 `tcgen05_gemm()` 会 `ann["is_tcgen05"] = 1`，
而 **`tcgen05_gemm_blockscaled()` 只写 `sf_a_granularity_k` / `sf_b_granularity_k`，没写 `is_tcgen05`**。
于是 `cuda::Gemm::SelectInst`（`src/cuda/op/gemm.cc:363`）跳过 `isTcgen05_` 分支，
落到「SFA/SFB region 已定义 ⇒ 这一定是 SM120 的 NVF4 mma.sync 路径」分支并 FATAL。

**这是一个上游 bug，不是本地安装问题**——已比对
`raw.githubusercontent.com/tile-ai/tilelang/{v0.1.14,main}/tilelang/language/gemm_op.py` 与
`.../{v0.1.14,main}/src/cuda/op/gemm.cc`，**两个版本都是同样缺失**（main 也未修）。

**最小 workaround（原型已内置，2 行）**：把库函数体重新 exec 一次并注入注解，不改动任何其它逻辑：
```python
_SRC = textwrap.dedent(inspect.getsource(_gemm_op.tcgen05_gemm_blockscaled))
_NEEDLE = 'ann["sf_b_granularity_k"] = int(sf_b_granularity_k)'
_NS = dict(_gemm_op.__dict__)
exec(_SRC.replace(_NEEDLE, _NEEDLE + '\n    ann["is_tcgen05"] = 1'), _NS)
T.tcgen05_gemm_blockscaled = tcgen05_gemm_blockscaled = _NS["tcgen05_gemm_blockscaled"]
```
（`kernels/tilelang/moe_bs_proto.py` 顶部。若将来 TileLang 修了这个 bug，`assert _NEEDLE in _SRC` 仍成立，
但注入会变成幂等冗余；应改成先探测 `is_tcgen05` 是否已在 `_SRC` 里。）

### 1.3 完整 API 契约（实测确认，含硬约束）

```python
T.tcgen05_gemm_blockscaled(A, B, C, SFA_tmem, SFB_tmem,
                           transpose_A=False, transpose_B=False,
                           clear_accum=False, wg_wait=0, mbar=None, *,
                           k_start, sf_a_granularity_k, sf_b_granularity_k,
                           use_2cta=False)
# gemm_op.py:297（0.1.14）。显式 async：绝不自动 emit mbarrier_wait_parity，mbar 必填。
```

必须配套使用的原语（缺一个都跑不起来）：

| 原语 | 作用 | 备注 |
|---|---|---|
| `T.alloc_tmem([BM, BN], "float32")` | C 累加器 | TMEM，≤512 列预算 |
| `T.alloc_tmem([BM, 4], "uint32")` | SFA in TMEM | 128 datapath × 4 列 = 每行 16B scale |
| `T.alloc_tmem([BM, BN//128*4], "uint32")` | SFB in TMEM | BN=128 → 4 列 |
| `T.alloc_shared((S, BN), "uint32")` | scale smem | 每行 1 个 uint32 |
| `T.tcgen05_cp_warpx4(sf_sh[st,:], sf_tmem)` | smem→TMEM | `tcgen05.cp.32x128b.warpx4` |
| `T.tcgen05_sf_warp_transpose(sf_sh[st,:])` + `T.fence_proxy_async()` | **必需**的前置转置 | 不做则 cp 读到错位 scale |
| `T.tcgen05_mma_arrive(mbar)` | 完成/占位 | = `tcgen05.commit` |
| `T.alloc_barrier` / `T.mbarrier_wait_parity` / `T.mbarrier_arrive` | 手写流水 | |

**`T.make_blockscaled_gemm_layout`（gemm_op.py:580）实测用不上**——layout inference 会自动推出
C 的 TMEM layout（`make_mma_store_layout`）。它的 docstring 说「必须 `T.annotate_layout`」，
但官方 `examples/blockscaled_gemm_sm100/*` 也没调它。**结论：该 API 存在，但在这条路径上是可选的。**

**硬约束表（全部由编译失败/成功实测得到）**：

| 约束 | 值 | 来源 |
|---|---|---|
| BM | **% 64 == 0**（推荐 128） | blockscaled 强制 `disable_ws=True` ⇒ `GetTCGEN5MMAMeta` 只有 M%64/M%128 分支；**没有 M=16/32 atom** ⇒ BM=16 不可能 |
| BN | **% 128 == 0** | `_tcgen05_num_smem_chunks` 要求 SF smem extent 是 `tcgen05.cp.32x128b.warpx4` 的 128 字粒度 |
| BK | % 32 == 0 且 **≤ 4·`gran`** | SF smem 每行只放 1 个 uint32（覆盖 `4·gran` 个 K）；BK 更大需要多字/行 |
| BK 下限 | **128**（当 BM=128） | BK=64 时 A tile 内层只有 32B < 64B swizzle ⇒ `Invalid TMA descriptor arguments` |
| smem dtype | **必须 `T.float4_e2m1_unpacked`** | 用 packed `T.float4_e2m1fn` 能编译能跑，但 **数值错（max\|err\|=3.0）**——见 §4 负结果 |
| TMEM 预算 | C(BM×BN) + 4 + BN/128·4 ≤ 512 列 | 128×128 fp32 → 128 列，余量充足 |

### 1.4 Scale-factor 语义（本文档最关键的通读结论）

API 的 `sf_*_granularity_k` **不是** MMA 的 K 原子，而是「**一个 ue8m0 字节覆盖多少 K**」。
由 lowering 的
`runtime_sf_id = ((k_start + ki*micro_size_k) // sf_granularity_k) % 4` 反推：

- 一个 **uint32 = 4 个连续 e8m0 字节**，第 `b` 个字节覆盖 K 区间 `[base + b·gran, …)`；
- `micro_size_k = 32`（fp4），`a_sf_id ∈ {0..3}` 选 **uint32 内的字节序号**；
- host 侧布局：`SFA[(g) * rows + row]`（**group-major**：一组的全部行连续），
  `g = k // sf_period`，`sf_period = gran*4/BK`；
- 每 `sf_period` 次 k 迭代重新 `tcgen05_cp_warpx4` 一次。

**当 `gran = 32` 时：一个 uint32 = 4 个 32-K 块 = 128 K，`sf_id = ki`，语义 = 标准 MXFP4 =
ferrite 的 `quant.rs`「routed experts: fp4 e2m1, I8-packed, ue8m0 per row × 32-col block」**
⇒ **与 ferrite 原生存储格式一一对应，无需改造语义，只需要一次 host 侧 pack（或一个 20 行的小 kernel）把
row-major 的 `[rows, K/32]` 字节面重排成 group-major uint32。**

（对照：TileLang 官方 `examples/deepseek_v4/fp8_fp4_gemm_1d1d_sm100.py` 用 `sf_granularity_k=128`
= 每个字节覆盖 128 K，是「更粗的 1D1D」。本文原型两档都测了，实现在 `--gran` 开关。）

---

## 2. 原型设计

形状 = ferrite verify MoE up：`N=640(2×320), K=5120`，`nseg=35`（36 assignments 去重后），`counts ∈ {1,2}`。

```
grid = (N/BN, nseg)                      # 每个 CTA 一个 (expert 段, N-tile)
A  : [nseg*BM, K]  float4_e2m1fn          # 段内行已 gather + BM-padding
W  : [nseg, N, K]  float4_e2m1fn          # 每段只读一次的去重权重
SFA/SFW : group-major packed e8m0 (uint32)
C  : [nseg*BM, N]  float32
```

**为什么 BM=128 而不是旧原型的 16**：blockscaled 路径关掉了 warp specialization，
`GetTCGEN5MMAMeta` 对 fp4 只接受 `M % 64 == 0`（非 ws）/`M % 128 == 0`。
M_e ≤ 2 却 pad 到 128 是 **64× 计算浪费**——但这条路不是计算受限（§4 证明 MMA 只占 5%），
padding 被带宽/延迟余量吸收，与 `tilelang-moe-grouped.md` §5 的结论一致。

三 warp 分工（第 4 个 warp 空转）：

```
warp0 (tx<32) : TMA producer（A、W、SFA、SFW 各一次 tma_copy）→ loaded[st]
warp1 (tx<64) : wait loaded+sf_full → tcgen05_cp_warpx4 ×2 → tcgen05_gemm_blockscaled(...,mbar=consumed[st])
warp2 (tx<96) : wait loaded → tcgen05_sf_warp_transpose ×2 + fence_proxy_async → sf_full[st]
epilogue      : 全 warp wait tmem_full → C_tmem→C_l→C_sh→global
```

交付物：
- `kernels/tilelang/moe_bs_proto.py` — 原型 + 补丁 shim + fp4-roundtrip 参考 + 配置扫
- `kernels/tilelang/exp_three_arm.py` — 三臂同刻对照
- `kernels/tilelang/probe_launch.py` — host-bench vs CUDA-graph 计时对照
- `kernels/tilelang/exp_wave.py` — 波次实验（N padding / BN 变体）
- `kernels/tilelang/exp_tma.py` — 纯 TMA 探针（有 bug 未跑通，见 §6 遗留）

---

## 3. ② 实测数字

### 3.1 三臂同刻对照（唯一可比口径）

```
nseg=35, counts 1..2, 36 assignments; NVIDIA B300 SXM6 AC, SMs=148
  bf16 grouped MMA (T.gemm, mma.sync)               116.8 us
  fp4 in-kernel dequant + mma.sync                  416.5 us
  fp4 tcgen05 blockscaled (BN=128 stg=6 gran=32)     72.6 us
  fp4 tcgen05 blockscaled (BN=128 stg=4 gran=32)     75.0 us
  fp4 tcgen05 blockscaled (BN=128 stg=6 gran=128)    74.2 us

  ratios vs bf16 grouped MMA:
    bf16 grouped MMA                             1.00x
    fp4 in-kernel dequant + mma.sync             3.57x
    fp4 tcgen05 blockscaled (stg=6 gran=32)      0.62x
```

⇒ **blockscaled 比 bf16 快 1.6×，比旧 fp4 臂快 5.7×**（同刻）。
⇒ 相对 `tilelang-moe-grouped.md` 的绝对基线（bf16 45.5µs / fp4 159.8µs，静默会话）：
   本次机器整体慢 2.57×（bf16 116.8/45.5），**若 blockscaled 等其他臂同等缩放，静默值 ≈ 28µs**。

### 3.2 配置扫（contended，nseg=35）

| gran | stages | BK | µs | max\|err\| | 备注 |
|---|---|---|---|---|---|
| 32 | 2 | 128 | 126.9 | 0.0 | |
| 32 | 3 | 128 | 102.8 | 0.0 | |
| 32 | 4 | 128 | 85.9 | 0.0 | |
| **32** | **6** | **128** | **72.7** | **0.0** | 最佳；stg=6 已吃满 smem |
| 32 | 8 | 128 | FAIL | — | `Failed to set dynamic shared memory size to 2703xx` (>228KB) |
| 128 | 2 / 3 / 4 / 6 | 128 | 116.9 / 90.0 / 75.4 / 74.0 | 0.0 | 粗粒度 scale |
| 32/128 | 任意 | 64 | FAIL | — | `Invalid TMA descriptor arguments`（64B swizzle vs 32B 内层） |

**关键观察：`gran=32` 与 `gran=128` 只差 ~10%** ⇒ scale 路径（transpose + cp，gran=128 时频率降 4×）
**不是瓶颈**。这排除了「SF 处理拖慢 MMA」的直觉假设。

### 3.3 波次量化：≤148 CTA 时 ~36µs 平台

| nseg | CTAs | µs (contended) |
|---|---|---|
| 4 | 20 | 35.3 |
| 8 | 40 | 35.5 |
| 16 | 80 | 36.0 |
| 24 | 120 | 36.1 |
| 35 | **175** | 72.4 |

**20 → 120 个 CTA 完全平坦（35.3→36.1µs）**：每个 CTA 独占一个 SM，per-CTA 时间是常数，
加 CTA 不加时间 ⇒ **per-CTA 延迟链是墙，不是聚合带宽**。
35 段 × (640/128 = 5) 个 N-tile = **175 > 148 SM** ⇒ 2 波 ⇒ 正好 2×。

**CUDA-graph 复测**：`probe_launch.py` 用 `torch.cuda.CUDAGraph` 打包 200 次 launch，
得到与 host-bench 几乎相同的数（nseg=24/stg=6: host 35.7µs vs graph 32.1µs；
nseg=35/stg=6: host 73.0 vs graph 69.7）⇒ **这些是真 GPU 时间，不是 launch/python 开销**。
（对照：trivial 1-CTA kernel 的 host-bench = 8.7µs，即 harness 本底只有 ~9µs。）

### 3.4 加宽 N 消除第二波 —— 无效

BN 必须是 128 的倍数，而 N=640 = 5×128，所以「一波」只能靠 pad N：

| N | BN | stg | CTAs | µs (contended) | max\|err\| |
|---|---|---|---|---|---|
| 640 | 128 | 6 | 175 | 72.9 | 0 |
| 768 | 256 | 4 | 105 | 86.4 | 0 |
| 768 | 256 | 3 | 105 | 87.2 | 0 |
| 768 | 384 | 3 | 70 | 116.7 | **1.14 ❌** |
| 768 | 128 | 6 | 210 | 75.6 | 0 |
| 1024 | 256 | 4 | 140 | 101.0 | 0 |

⇒ 把 175 CTA 压到 105/140（真正的单波）**没有换来提速**，因为 **per-CTA 时间随 tile 字节数线性增长**：
- BN=128：17KB/迭代、~900ns/迭代 → 40 iter ≈ 36µs
- BN=256：33KB/迭代、~2150ns/迭代 → 40 iter ≈ 86µs

**即 per-CTA TMA 吞吐 ≈ 15-19 GB/s 是个与 tile 形状无关的常数。**
（`N=768/BN=384` 还给出了唯一的数值错误：`max|err|=1.14`——384 不是
`BN//128*4` 公式的干净取样点，**用 BN%128==0 以外的值会静默算错，务必避开**。）

---

## 4. ③ 数值形态 + 两个负结果

| 项 | 结果 |
|---|---|
| 累加器 | fp32（`T.alloc_tmem([128,BN],"float32")`） |
| **`max\|err\|` vs fp4-roundtrip 参考（所有编译成功的配置）** | **0.00000（逐位）** |
| 覆盖范围 | dense 单 tile（M=128 全实行）、MoE 35 段（counts 1..2，padding 行补零）、gran∈{32,128}、stg∈{2,3,4,6}、N∈{640,768,1024} |
| 误差来源 | 仅 fp4 输入量化本身（与 `tilelang-moe-grouped.md` §7 的 fp4-LUT 臂一致：内核本身无损） |
| 参考实现 | `dequant_fp4()`：packed nibble → e2m1 LUT，scale `2^(byte-127)`，每 32(或 gran) 列 repeat_interleave |

**⇒ blockscaled 吃 e2m1 数据 + ue8m0 scale 是精确的，不引入额外误差。**

### 4.1 负结果 A：packed smem 会静默算错

把 `A_sh`/`B_sh` 的 dtype 从 `T.float4_e2m1_unpacked`（1 值/字节）改成 packed
`T.float4_e2m1fn`（2 值/字节，每 stage 省一半 smem）：
- **编译通过、运行不报错、时间几乎不变（stg=8/10: 72.2µs vs unpacked stg=6: 73.7µs）**
- **但 `max|err| = 3.00024`** —— 静默错值。

⇒ **blockscaled tcgen05 路径必须用 `float4_e2m1_unpacked` smem**（`copy_analysis.cc:539`
只允许 packed-global → unpacked-smem 的单向 unpack TMA；MMA descriptor 按 unpacked 布局解码）。
省 smem 换 stage 这条路**走不通**。

### 4.2 负结果 B：stage 深度不是唯一瓶颈

用 packed smem 换来的 stg=8/10 **完全没有提速**（72.2 vs 73.7），
加上 unpacked 时 stg=6 已是 smem 上限（228KB），
⇒ 说明「无限加深流水」并不能解决 §4.3 的墙。

---

## 5. ④ Gap 分析：为什么没到 15-25µs

### 5.1 事实

- 权重只读 **57.3MB**（`35 × 640×5120×0.5B`）；contended 72.6µs ⇒ 0.79 TB/s；折算静默 ≈ 2 TB/s。
  B300 峰值 ~8 TB/s ⇒ **根本不在带宽曲线上**。
- 也不是计算受限：BM=128 的 64× padding 下总 MAC = 35×128×640×5120 = 14.7 GMAC ≈ 29.4 GFLOP。

### 5.2 根因定位（证据链，不是猜测）

| 实验 | 结果 | 排除/确认 |
|---|---|---|
| **TMA-only 消融**（warp1 完全不发 UMMA，只 `tcgen05_mma_arrive` 占位） | nseg=8/stg=4: 34.0µs vs 完整 35.4µs；nseg=35/stg=4: 72.2 vs 75.2 | **tensor core 只占 ~5%** ⇒ 不是 MMA 慢 |
| `gran=32` vs `gran=128`（SF 工作量差 4×） | 只差 ~10% | **SF transpose/cp 只占 ~10%** |
| CTA 数扫描（20→120） | 平坦 35.3→36.1µs | **per-CTA 延迟链是墙**（加并行度不加时间） |
| tile 形状（BN=128 vs 256） | 15-19 GB/s / CTA **不变** | **per-CTA TMA 吞吐是常数 ≈ 20 GB/s** |
| stage 深度 6→10 | 无改善 | 不是「流水不够深」 |
| CUDA graph vs host bench | 几乎相同 | 不是 launch/host 开销 |

**⇒ 墙 = 每 CTA 的 TMA 管道（≈20 GB/s/SM）**：
`17KB/迭代 ÷ ~900ns = 19 GB/s/CTA`，×148 SM ≈ **2.8 TB/s 全局上限**。
57MB 权重 ÷ 2.8 TB/s ≈ **20µs 是这套管道的理论天花板**（一波、负载均衡的理想情况），
但真实形状 175 CTA > 148 SM ⇒ 2 波 ⇒ 72µs。
（对比：bf16 臂在同一台机器上跑到 5 TB/s（上一会话）——它的 `T.Pipelined`+`cp.async` 由
TileLang 自动生成，比手写 TMA+mbarrier 流水高一个量级。**这不是 fp4 的问题，是「手写 tcgen05 流水」
这件事在 0.1.14 上还没有自动 pipeline 支持的问题。**）

### 5.3 「最近支持形态」的判定

**不需要降级形态**——e2m1+ue8m0 per-32 就是原生支持（§1）。
差的不是 ISA 能力，而是 **issue/流水实现**。差距量化：
- 目标 15-25µs ↔ 实测（contended）72.6µs / 折算 ~28µs；
- gap 的来源按贡献排序：**(1) 每 CTA 只有 ~20 GB/s TMA（~10× off）；(2) 175 CTA 2 波（2×）**。

### 5.4 未验证的改进方向（留给主 agent，均未实跑）

1. **改权重全局布局**：当前每个 tile 是「128 行 × 64B，行距 2560B」= 128 段不连续 64B。
   改成 K-major/blocked，让 CTA 的 tile 成为连续段，DRAM/L2 sector 效率应显著改善。
2. **消除第二波**：N pad 到 768 + BN=256（105 CTA）已测，contended 86µs——**单独这一条不够**，
   必须与 (1) 或 (3) 合用。
3. **TMA 分两条 warp 流**（warp0 喂 A、空闲的 warp3 喂 W，各自独立 mbarrier），
   把每迭代的 4 次 TMA 发布 + 握手拆成 2 条独立链。
4. **加大 BK**（需 2 个 SF 字/行/stage ⇒ 每迭代 2 组 cp+MMA），把每迭代握手摊薄到 2× 的字节上。
5. **把 SF 的 cp+transpose 从内层循环挪出去**（每 stage 一块独立 SF TMEM 区域），
   TMEM 预算允许（4 列 × stage 数 ≪ 512）。

---

## 6. 复现步骤

```bash
# 1) 上传原型 + 实验脚本
scp kernels/tilelang/{moe_bs_proto.py,probe_launch.py,exp_wave.py,exp_three_arm.py} \
    ubuntu@43.202.208.136:~/tl_proj/
# 2) 三臂同刻对照（唯一可比口径）
ssh ubuntu@43.202.208.136 'cd ~/tl_proj && python3.12 exp_three_arm.py'
# 3) 只跑 blockscaled（默认 BN=128 BK=128 stg=4 gran=32）
ssh ubuntu@43.202.208.136 'cd ~/tl_proj && python3.12 moe_bs_proto.py'
# 4) 配置扫 / 波次实验
ssh ubuntu@43.202.208.136 'cd ~/tl_proj && python3.12 moe_bs_proto.py --sweep'
ssh ubuntu@43.202.208.136 'cd ~/tl_proj && python3.12 exp_wave.py'
# 5) 生成 CUDA 源（核对 lowering 形态）
ssh ubuntu@43.202.208.136 'cd ~/tl_proj && python3.12 dump_src.py && grep -n "blockscaled\|tcgen05_cp\|sf_warp" moe_bs_dense.cu'
```
**跑之前先确认卡是空的**（`nvidia-smi --query-compute-apps=...`），否则数字会被拖慢 ~2.6×。

---

## 7. 交付项对照

| 验收项 | 状态 | 位置 |
|---|---|---|
| ① tcgen05 API 可用性判定（e2m1+ue8m0 是否直支持） | ✅ **直接支持**（三层证据）+ 发现 0.1.14 `is_tcgen05` 注解缺陷 | §1 |
| ② 原型三臂对照数字 | ✅ 同刻：bf16 116.8 / fp4-mma_sync 416.5 / **blockscaled 72.6µs（0.62×）** | §3.1 |
| ③ 数值形态 | ✅ **max\|err\| = 0.00000**（逐位，全配置） | §4 |
| ④ API 不支持 fp4 时的 gap 分析 | ⚪ **不适用**（API 支持 fp4）；改为「性能 gap 分析」 | §5 |
| ⑤ 本文档 | ✅ | 本文件 |

**下一步（按 ROI）**

1. **【主 agent】空卡复测**：机器上有 co-tenant 期间所有绝对 µs 被拖慢 ~2.6×，
   必须复核 blockscaled 的静默绝对值（预计 ~28µs）与是否过 50µs 判据。
2. **打通「自动流水」而不是手写流水**：查 TileLang 是否支持把 `tcgen05_gemm_blockscaled`
   放进 `T.Pipelined`（`_lower_blockscaled` 已是 explicit-async，pipeline pass 应该能接管 mbar）。
   若可以，bf16 臂那种 5 TB/s 的自动流水质量就能直接复用。
3. **权重布局连续化**（§5.4-1）——这是把 20 GB/s/CTA 抬上去的最直接杠杆。
4. **显存收益独立于速度**：blockscaled 读原生 fp4（57MB/层）**零 dequant、零 bf16 副本**，
   直接解决 MoE bf16 路径 +15GB/rank 的显存问题——即使速度只能打平 bf16，这条收益仍然成立。

---

## 8. 遗留 / 警告

- **⚠️ 本文档所有绝对 µs 均为 contended 口径**（co-tenant `ferrite-send --tp 8` 占满 8 卡，
  04:07:14 起）。同刻标定：copy 1GB 2.39 TB/s、bf16 8192³ 627 TFLOPS。
  请只使用**同刻比值**（0.62×）与**折算值**（~28µs），并在空卡上复测后再引用绝对值。
- **GPU 测量被用户叫停**（本轮）：已在远端 kill 全部自建 GPU 进程（`wait_and_measure.sh` /
  `exp_three_arm.py` / `moe_bs_proto.py` 等已确认退出，远端只剩 co-tenant 的 `ferrite-serve`）。
  **剩余验证（空卡绝对值）留给主 agent。**
- `exp_tma.py`（纯 TMA 探针，用于量化「64B 内层 / 2560B 行距」的 DRAM 效率）
  **未跑通**：所有配置 `CUDA error: unspecified launch failure`（疑似 smem 越界），未再追。
- `BN % 128 != 0` 会静默错值（实测 BN=384 → err 1.14），不要用。
- `moe_bs_proto.py` 顶部的补丁 shim 依赖 `gemm_op.py` 中存在
  `ann["sf_b_granularity_k"] = int(sf_b_granularity_k)` 这一行；升级 TileLang 后需重新确认。
