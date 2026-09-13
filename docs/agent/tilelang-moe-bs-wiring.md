# tcgen05 block-scaled fp4 MoE 的工程化接线（工部 · 代码件 + 验证手册）

> 上游原型与全部 GPU 结论：`docs/agent/tcgen05-blockscaled-proto.md`（**先读它**）。
> 同族先例：`docs/agent/tilelang-moe-grouped.md` §7/§8（bf16 臂）、
> `docs/agent/tilelang-moe-wiring.md`（bf16 臂的验证手册）、
> `kernels/cuda/tilelang_gen/PROVENANCE.md` §7（bf16 臂的出处）。
>
> 本文件是 **tcgen05 blockscaled（原生 fp4）MoE-up 臂**从原型到生产件的落地方案。
> 工部 · 2026-09-13。**纯代码工作：本文档不含任何 GPU 实测数字**——所有 GPU 数字
> 都来自上游原型（contended 口径，见其 §8 的警告），新臂的实测留给主 agent（§6）。

---

## 0. 交付物与边界

| 件 | 性质 | 位置 |
|---|---|---|
| AOT 生成器（含 0.1.14 `is_tcgen05` 缺陷的固化 shim） | **新** | `kernels/tilelang/gen_moe_bs_aot.py` |
| launcher shim（2 个导出符号 + gather/scatter/pack） | **新** | `kernels/cuda/tilelang_gen/moe_bs_shim.cu` |
| device 符号 + 包装器 + `supports_*` | 已落地 | `crates/ferrite-models/src/dsv41/device.rs` |
| arm 门（`DSV41_MOE_TILELANG_BS`）+ SF 池尺寸助手 | 已落地 | `crates/ferrite-models/src/dsv41/weights.rs` |
| 装载期 SF pack 池 | 已落地 | `crates/ferrite-models/src/dsv41/load.rs` |
| 运行期分派（`moe_rows` / `moe` 的 gate/up 点位） | **建议 patch**（不落盘，peer 在改 chain_dev.rs） | 本文 §5.3 |
| 生成物（`.cu` / host source / config） | **未生成**（需远端 tilelang，主 agent 执行 §6.1） | `kernels/cuda/tilelang_gen/` |

**本臂是什么**：routed gate/up（`expert_gate_up_fp4_batched`，SIMT FMA-issue bound）的
tensor-core 替代，**直接吃 ferrite 原生的 fp4 权重池 + 它自己的 ue8m0 面**——
**零 dequant、零 bf16 副本、零权重拷贝**。与 bf16 臂（`DSV41_MOE_TILELANG`）的根本
区别就在这里：bf16 臂的前提是装载期把专家权重展开成 bf16 常驻（实测 **+105~113
GiB/rank**，PROVENANCE §7.5），本臂不花那笔显存。**即使速度只打平 bf16，这条收益
独立成立**（原型 §7 第 4 条）。

**本臂不覆盖**：down（`w2`）。不是遗漏——是**形状上不可达**：down 的 `K = inter_local = 320`，
而 blockscaled 路径要求 `BK ≥ 128` 且 `BK ≤ 4·gran = 128`（即 `BK ≡ 128`）且
`BK | K`；320 不是 128 的倍数（`320/128 = 2.5`）。⇒ routed down 继续走
`expert_down_fp4_batched`。**不要**试着用 `BK=64` 绕（内层 32 B < 64 B swizzle，
原型 §3.2 实测 `Invalid TMA descriptor arguments`）。

---

## 1. 交付 ②：BM tile × 36 assignments 的段落映射

### 1.1 事实起点（来自原型 §1.3/§2，勿重推）

* `moe_align_host`（`chain_dev.rs:790`）把 `route_idx_r[m][topk]` **按 expert 稳定排序**，
  产出**每 expert 一段**的 `order`/`counts`/`eid`，段数 `nseg ≤ SEG_CAP = 36`。
  生产形状 `counts ∈ {1,2}`、`nseg = 35`（36 个 assignment 去重后）。
* blockscaled 强制 `disable_ws` ⇒ `GetTCGEN5MMAMeta` 只有 `M % 64 == 0` / `M % 128 == 0`
  两个分支，**没有 M=16/32 原子** ⇒ `BM=16`（bf16 臂用的那一档）**不可能**。
* `T.Kernel` 的 grid 维度必须是编译期常量 ⇒ `grid.y ≡ SEG_CAP = 36`（真实 35 时最后
  一段空转，见 §1.4）。

### 1.2 结论：**每 expert 段独立 pad 到 BM=64 行**，沿 **M 轴**，不是 K 轴

```
grid = (N_UP/BN, SEG_CAP) = (5, 36)          # 180 CTA，148 SM ⇒ 2 波
A : [SEG_CAP*BM, K]  段内行已 gather，r >= counts[seg] 的行**全 0**
W1/W3 : [E, NP, K]   按 eid[seg] 直取（每 expert 只读一次的去重权重）
C : [SEG_CAP*BM, N_UP]
```

**为什么 M 轴 pad、为什么不是「K 维度切」**：

1. **K 轴承载不了段身份。** 所有段共享同一个 `K = dim`；段与段的区别在 **W 的哪一行**
   （不同的 expert 权重面）。把 K 切开只会把一个段内部拆成两次 MMA，对「段的行数 ≤ 2」
   这件事毫无帮助。⇒ 「BK 维度切」不是一个可行的映射，它解决的是别的问题（每迭代
   握手摊薄，原型 §5.4-4），与段映射正交。
2. **段之间无法合并进一个 M-tile。** 一段 = 一个 expert（`moe_align_host` 已按 expert
   去重连续），同一 M-tile 内的行必须共用同一个 B 操作数；两个不同 expert 的行合并在
   一起就需要两次 MMA，反而更贵。⇒ 每段一个 M-tile 是**唯一**的映射。
3. **M 的浪费只落在 M 上**：
   * **权重带宽完全不受 BM 影响**——每个 `(段, N-tile)` 无论如何都只读一次自己的 W tile；
   * **MMA 只是预算的 ~5%**（原型 §5.2 的 TMA-only 消融：不发射 UMMA 只慢 1.4/3.0µs），
     `BM=64` 的 32× M 浪费落在 5% 上 ⇒ ~2.5%；
   * **A 的流量与 BM 成正比**：`BM=64` ⇒ A = 36×64×2560 B = **5.9 MB/层**（与 bf16 臂
     `BM=16` 的 `36×16×5120×2 B` 恰好相同）；`BM=128` 是 11.8 MB。
   ⇒ **BM=64 比 BM=128 省 ~20% 总流量**（97 MB → 77 MB/层），并且 smem 余量换来更深的
   流水（`stages=6` 时 184832 B vs BM=128 的 ~228 KB 顶格）。这就是选 64 的理由。

4. **pad 行必须真的全 0**（nibble 与标度都 0）：内核**不做 mask**，`A_sh` 的 pad 行会进
   MMA。e2m1 的 nibble 0 = +0.0，乘任何标度都是 0 ⇒ 只要 nibble 为 0 就安全；shim 仍把
   SFA 的 pad 字也写 0（省得将来有人改 nibble 语义）。**段内 pad，不发散、不越界**。

### 1.3 N 轴：为什么是「两个 64 宽的半块」而不是一个 640 宽的连续面（**这是本方案唯一的
结构性设计选择，且被 ferrite 的池布局强制**）

ferrite 的专家池（`load.rs:827-861`）把 6 个面按 **128 B 对齐**塞进一个 block，plain 顺序是

```
[w1][w1.scale][w3][w3.scale][w2][w2.scale]
```

⇒ **不存在**一个连续的 `[2*NP, K]`（= `[640, 5120]`）gate‖up 权重面：`w1.scale` 插在中间。
而 blockscaled 的 `BN % 128 == 0`，`320` 整除不了 128 ⇒ 也不能「一个面起一个 kernel」
（`320/128 = 2.5`，第 3 个 N-tile 会越界读）。

**解**：把 `BN = 128` 的 tile 理解成两个 `NH = 64` 宽的半块（`NH = NP / grid.x = 320/5`）：

```
B_sh[st, 0:NH]   <- W1[e, bx*NH : (bx+1)*NH, k*BK : (k+1)*BK]
B_sh[st, NH:2*NH] <- W3[e, bx*NH : (bx+1)*NH, k*BK : (k+1)*BK]
```

`640 = 2 × 320 = 2 × 5 × 64` ⇒ `grid.x = 5` **恰好**覆盖全部 640 列，**无 pad、无越界、
无 id 掩码**，且 grid 与原型完全一致（原型 `W:[NSEG,640,K]`/`BN=128` 也是 `grid.x=5`）。

**代价（必须记住）**：`C` 的**列序是交错的**。tile `bx` 的第 `j` 列（`j ∈ [0,128)`）：

```
j <  64 : gate 的第 bx*64 + j 列      -> out 的 n = bx*64 + j
j >= 64 : up   的第 bx*64 + j - 64 列 -> out 的 n = NP + bx*64 + (j - 64)
```

scatter 按这个映射解交错（`moe_bs_shim.cu::tl_moe_bs_scatter_kernel`）。**这是本臂唯一
一个「C 的字节顺序与老路径不同」的地方**，所以 §6 的 parity 检查**必须逐元素比**，
不能只比行和/范数。

**smem 子切片的安全性**：`B_sh` 的 smem 行 = `BK` 字节 = 128 B（unpacked fp4，1 B/元素），
swizzle 原子恰好是一行；半块起点 `64 × 128 B = 8192 B` 是原子整数倍 ⇒ 两次 TMA 落进
同一个 swizzled 缓冲是自洽的。`SFW_sh` 的半块 = 64 个字 = 256 B，16 B 对齐 ✓。

### 1.4 36 vs 35：最后一段空转是**故意的**

`grid.y` 必须是编译期常量（TileLang），而真实 `nseg` 随路由逐位变化（本形状 35）。
host 把 `order`/`counts`/`eid` 在 `nseg` 之后**零填充**（`moe_align_host` 已这样做），
pad 段 `counts=0` ⇒ gather 写全 0、scatter 不写。代价 `grid.y` 35→36 ≈ **3% 空转**
（180 CTA 里 5 个），与 bf16 臂完全相同。

### 1.5 段内行的来源：**已经是 fp4 的激活**，不是 f32

本臂的 `A` **不是** f32 激活转换来的——它直接读 routed expert 的**已量化 fp4**：

| 源 | 布局 | 说明 |
|---|---|---|
| `xq4_r`（`chain_dev.rs:569`） | `[rows*topk][dim/2]` u8 | e2m1 打包，低 4 位 = 偶 k（与 `quant.rs::fp4_pack_byte` 同构） |
| `xsc4_r`（`:570`） | `[rows*topk][dim/32]` **f32** | `fast_round_scale6` 的输出（**2 的幂**） |

⇒ gather 只是一次 **memcpy 语义的搬字节**（零解码），标度做一次 `f32 → ue8m0` 的位转换
（§2.3）。**比 bf16 臂的 gather 少 8× 的读流量**（2560 B/行 vs 20480 B/行）。
既有量化 pass **不动**（本臂替换的是它下游的 batched gate/up launch）。

---

## 2. 交付 ③：SF（scale factor）的 pack 设计与落点

### 2.1 语义回顾（原型 §1.4，逐字）

`sf_*_granularity_k = gran` 是「**一个 ue8m0 字节覆盖多少 K**」，不是 MMA 的 K 原子。
一个 **uint32 = 4 个连续 e8m0 字节** = `4·gran = 128` 个 K；host 侧布局 **group-major**：
`SF[g * rows + row]`，`g = k // 128`。`gran = 32` 时这正是 ferrite 的原生
per-(row, 32-col block) MXFP4 ⇒ **语义一一对应，不需要任何换算**。

```
sf_words  = K / (4*gran) = 5120/128 = 40     # 每行的 SF 字数
sf_period = 4*gran / BK  = 128/128  = 1      # 每 k-iter 一组 cp+transpose
```

### 2.2 权重的 SF：**装载期一次性**（结论）

| 落点 | 成本 | 判据 |
|---|---|---|
| **装载期 pack**（采纳） | **+1.54 GiB/rank** 常驻；热路径 0 额外 launch | ✅ |
| 每次调用 pack | 0 额外显存；热路径 +3.7 MB/层读 + 3.7 MB/层写 + 1 个 launch | 备选 |

选装载期的三条理由：

1. **SF 是权重的一部分，永不变化。** 一次转换换热路径零开销——这是**唯一**一种
   「花一次静态成本消除每步动态成本」的机会，而每步的成本落在已经被定位为墙的
   每-CTA 延迟链上（原型 §5.2）。
2. **1.54 GiB/rank 相对本臂的立意可以忽略**：对比 bf16 臂的 **+105 GiB/rank**（1.5%）；
   对比它旁边 35 GiB 的 fp4 池（4%）。而本臂存在的全部理由就是**不花那 105 GiB**。
3. **它是无损的**：`dst[g*rows + row] = u32(src[row*(k/32) + g*4 .. +4])` 是**纯字节
   置换**——ue8m0 字节不被解释、不被重舍入（值仍是 `2^(b-127)`）⇒ 与原型
   `pack_sf_group_major` **逐位同构**。这不是「近似」，是恒等变换。

计算（生产形状）：

```
每 expert 每面: sf_words(40) × NP(320) × 4 B = 51 200 B
池总量:        51 200 × 2 面 × 384 expert × 40 层 ≈ 1.54 GiB/rank
```

### 2.3 激活的 SF：**每次调用**，在 gather kernel 里顺带做

激活每步都变，所以它的标度**必然**是每调的。但不能为此多起一个 launch：gather 已经在
搬 `xq4_r` 的 2560 B/行，顺带写 40 个 uint32 是免费的。

```
SFA[g * M + seg*BM + r] = b0 | b1<<8 | b2<<16 | b3<<24      (M = SEG_CAP*BM)
  bb = f_pow2_to_ue8m0( xsc4_r[assign][g*4 + b] )
```

`f_pow2_to_ue8m0`（`moe_bs_shim.cu::tl_bs_f_pow2_to_ue8m0`）与生产件
`dsv41_experts_mxf4.cu:287` **逐字同源**：`fast_round_scale6` 的输出是 2 的幂，ue8m0 的
字节就是它的偏置指数 ⇒ **一个移位**，无舍入、无查表。⇒ 本臂的**量化误差与老路径逐位
相同**（这也是 §6 parity 能逐元素比的前提）。

### 2.4 备选（**未采纳，留给后续**）：把 W 面重排成连续

如果接受改 `load.rs` 的 `poff` 顺序为 `[w1][w3][w1.scale][w3.scale][w2][w2.scale]`，
则 `w1‖w3` 变成一个**连续的 `[640, K/2]` 面**（零副本、零额外显存——只是偏移表变了，
`block`/`poff` 仍是 128 B 对齐：`w1b = 819200 B` 是 128 的倍数）。收益：每 k-iter 的
W 搬运从 2 次 TMA 变 1 次、`B_sh` 无需子切片、`C` 的列序恢复成 `[gate‖up]` 直序
（scatter 变平凡）。风险：`poff` 是**所有既有 expert kernel 共享**的几何，被
`DSV41_ALIGN_AUDIT` 与 `check_bulk_geometry` 盯着，改动语义超出本任务范围。
⇒ **本方案不依赖它**；若 §6 发现子切片 TMA 在 lowering 侧不成立，这是首选替代。

---

## 3. 精确布局契约（host 必须逐字节对齐）

| 张量 | 形状 | dtype | 行距 / 来源 |
|---|---|---|---|
| `A` | `[SEG_CAP*BM, K/2]` | u8（packed e2m1，低 4 位 = 偶 k） | `K/2 = 2560` B，**连续**；pad 行全 0 |
| `W1`/`W3` | `[E, NP, K/2]` | u8 | 行距 `K/2 = 2560` B（`ExpertRows` 面无 pitch 例外）；expert 面 stride `NP*K/2` |
| `SFW1`/`SFW3` | `[E, sf_words*NP]` | u32 | 装载期 pack 池（§2.2） |
| `SFA` | `[sf_words * SEG_CAP*BM]` | u32 | 每调 pack（§2.3） |
| `Eid` | `[SEG_CAP]` | i32 | host 数组，上行 |
| `C` | `[SEG_CAP*BM, 2*NP]` | f32 | 列序**交错**（§1.3）；cta 常驻 scratch |

**三个「静默错值」陷阱**（写代码时逐条对照）：

1. `xsc4_r` 的行距是 **`dim/32 = 160` 个 f32**（不是字节）。`xsc4_r` 是
   `fb(VERIFY_ROWS*dim/32 + 8)` —— `+8` 只是余量，**pitch 就是 `dim/32`**
   （`chain_dev.rs` 的 batched 调用按 `arow*(k/32)` 寻址，即同一约定）。
2. `w1.scale`/`w3.scale` 是 `Shard::ExpertRows`，**`DSV41_SF_STRIDE_PAD` 不重排它们**
   （只重排 `w2.scale` 的 10 B 行）。所以源 pitch = `k/32 = 160` B。若哪天这条变了，
   `weights::moe_bs_sf_src_pitch` 是唯一需要改的地方（`load.rs` 已改用它，并带
   `debug_assert_eq!(pitch, k/32)`）。
3. `C` 是**交错列序**（§1.3）——scatter 之外任何地方按 `[gate‖up]` 读它都是错的。

---

## 4. 交付（隐含）：描述符 ABI 和「唯一需要转写的地方」

### 4.1 事实：默认 lowering 产出 TMA 描述符形态

`docs/agent/tilelang-integration-design.md` §1.3 实验 A 已在**本仓库**实测过默认形态：

```cpp
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(
    __grid_constant__ const CUtensorMap A_desc,
    __grid_constant__ const CUtensorMap B_desc,
    bfloat16_t* __restrict__ C) { ... }
```

**裸指针 ABI（实验 B 的 `TL_DISABLE_TMA_LOWER=1`）在这条路上走不通**：blockscaled 的
A/B smem **必须**是 `float4_e2m1_unpacked`（packed smem 能编译能跑但静默错值，err=3.0，
原型 §4.1），而 packed-global → unpacked-smem 这个「展开」只有 TMA 的 tensor 形式能做
（`copy_analysis.cc:539`）。⇒ **host 必须建描述符**。

### 4.2 但 `build.sh` 不链 `-lcuda` ⇒ shim 用 `dlopen`

`kernels/cuda/build.sh:164` 的 nvcc 命令行只有 `-I tilelang_inc`，没有 `-lcuda`。直接
引用 `cuTensorMapEncodeTiled`（`cuda.h` 里的声明）会引入链接期未定义符号 ⇒ **`.so` 连
不上**。本 shim 的处理：`dlopen("libcuda.so.1")` + `dlsym("cuTensorMapEncodeTiled")`，
用**自带的 typedef**（签名与 `cuda.h` 逐字一致）承接。

* **零 build-system 改动**：`build.sh` 一行不改 ⇒ `BUILD_ID` 的 `CU_HASH`/flag 集不变，
  不触发「both products 要重建」的连锁（build.sh 头注释里那个踩过的坑）。
* **可降级**：driver 不可用时 `rc = 2`（decline），不留半发射状态。
* `libcuda.so.1` 在任何 CUDA 进程里已被 cudart 加载 ⇒ `dlopen` 必然命中。

### 4.3 「唯一需要人工转写的地方」= `moe_bs_encode_tmaps()`

描述符的 **dims / strides / box / swizzle** 是 TileLang 内部推断的（它按 smem layout
选 swizzle，fp4 的 sub-byte 展开方式只有它的 lowering 知道）。**不许猜。**
本仓库没有任何生成物可作为依据（`kernels/cuda/` 下没有 bs dump）。

⇒ **生成器同时 dump TileLang 自己的 host launcher**（`moe_bs_up_tl_host.cu`），
那里面有它对每个张量调 `cuTensorMapEncodeTiled` 的**完整实参**——那就是权威配方。
`gen_moe_bs_aot.py::_dump_host()` 通过 `get_host_source()`（失败则退到
`export_sources(host_path=)`）拿到它。

**转写流程（5 分钟，主 agent 在 §6.3 做）**：

```bash
cd kernels/cuda/tilelang_gen
grep -n "cuTensorMapEncodeTiled" -A 12 moe_bs_up_tl_host.cu | head -140
# 逐项对齐 moe_bs_shim.cu 的 spec_a / spec_w / spec_sfa / spec_sfw / spec_c：
#   globalDim / globalStrides / boxDim / elementStrides / swizzle / interleave
# 只有三类可能不符：
#   (i)   box 首维是「字节(K/2=64)」还是「元素(K=128)」  -> kABox
#   (ii)  swizzle 枚举                                    -> kSwzAB / kSwzC
#   (iii) fp4 是否走了别的 load_mode                      -> box/rank 整体形状
```

**参数序错误不会静默**：形参里描述符（`CUtensorMap`）/ `float*` / `int*` 是**不同类型**，
任何错位都是**编译错误**（nvcc 在 shim 那一 TU 就报）。这正是这个 ABI 唯一的救赎——
也是 §6.2 把「compile-only」放在跑之前的原因。

**W1/W3/SFW 的描述符每层都要重建**：ferrite 的专家池是**每层一个 allocation**
（`LayerDev.expert_pool`，40 层各一份）⇒ 基址每层都变。shim 用
`ensure_w_tmaps()` 按基址缓存重建（不重建就是**静默读到别层的权重**）。
A/SFA/C 指常驻 scratch（INIT 期 `cudaMalloc`）⇒ `encode_fixed_tmaps()` 建一次。

---

## 5. 交付 ④：Rust 接线（已落地部分 + chain_dev.rs 的 patch 建议）

### 5.1 arm 门语义（`weights.rs::moe_tilelang_bs`，已落地）

```rust
pub fn moe_tilelang_bs() -> bool      // DSV41_MOE_TILELANG_BS，默认 OFF，每进程读一次
```

| `DSV41_MOE_TILELANG`（bf16 臂） | `DSV41_MOE_TILELANG_BS` | 行为 |
|---|---|---|
| off | off | 不变（老路径） |
| off | on  | **本臂**服务 routed gate/up |
| on  | off | bf16 臂（不变） |
| on  | on  | **本臂胜**，bf16 臂被 shadow（一次性提示）——两者读不同权池、不同生命周期，同开只是白花 bf16 镜像的显存 |

为什么门放在 `weights.rs` 而不是 `chain_dev.rs` 的 `moe_tilelang()` 旁边：**装载期也要
读它**（`load.rs` 决定是否建 SF 池），而从运行期分派点读装载期决策正是会产生漂移的形状。
`weights.rs` 是 `sf_stride_pad` 已经待着的地方，`chain_dev.rs` 的调用点用
`crate::dsv41::weights::moe_tilelang_bs()` 即可（**不改 chain_dev.rs 也能编译**）。

### 5.2 已落地的 Rust（可 `cargo check`）

| 文件 | 改动 |
|---|---|
| `device.rs` | 2 个函数指针字段（`moe_tilelang_gate_up_bs` 16 参 / `moe_bs_pack_wsf` 5 参）+ `ko!` 注册 + 2 个包装器（`rc==2 → Ok(false)`）+ `supports_moe_tilelang_bs()` |
| `weights.rs` | `moe_tilelang_bs()` 门 + `moe_bs_sf_words(k)`（`k%128!=0 → 0`）+ `moe_bs_sf_src_pitch()`（与 `sf_pitch_plane` 同源，防漂移） |
| `load.rs` | `DevExpert.wsf1/wsf3`（`Option<DevTensor>`）+ `LayerDev.expert_bs_pool` + `want_bs` 门 + **装载期 pack 块**（59 行，见 §2.2）+ 返回值 5 元组（两个调用点同步） |

**验收**：`cargo check -p ferrite-models` ✅ / `cargo check --workspace` ✅（2026-09-13）。

### 5.3 建议 patch：`chain_dev.rs`（**不落盘**，peer 在改）

改动点与 bf16 臂**同构**，共 4 处：

**(1) 就绪门**（放在 `moe_tilelang_ready` 旁边）：

```rust
    /// Whether the BLOCK-SCALED (native fp4) arm can serve THIS step.
    /// Same three families of terms as `moe_tilelang_ready`, but:
    ///   * it needs the PACKED-SF pool instead of the bf16 copies;
    ///   * `DSV41_EXPERT_ILV` must be OFF (under ILV the w1 view aliases the
    ///     doubled region, so `w1` is not a clean [NP, K/2] plane — the arm would
    ///     read interleaved bytes as a gate plane: silent wrong answer);
    ///   * `DSV41_MOE_TILELANG` (bf16 arm) being armed does NOT disable it — the
    ///     block-scaled arm wins, and the caller reports the shadowing once.
    fn moe_bs_ready(
        &self,
        ld: &LayerDev,
        m: usize,
        topk: usize,
        dim: usize,
        inter_local: usize,
        n_routed: usize,
    ) -> bool {
        crate::dsv41::weights::moe_tilelang_bs()
            && !self.dev.capturing()
            && self.dev.supports_moe_tilelang_bs()
            && !ld.experts_ilv                       // <-- 唯一的额外硬条件
            && n_routed == 384
            && dim == 5120
            && inter_local == 320                    // NB: ==TILELANG_* 的 inter_local
            && (1..=6).contains(&topk)
            && (1..=6).contains(&m)
            && ld.experts.len() >= 2
            && ld.experts[0].wsf1.is_some()
            && ld.experts[1].wsf1.is_some()
    }
```

**(2) 权重基址**（照 `moe_tilelang_weights` 的形状；**4 个 base + stride 校验**）：

```rust
    /// The routed experts' fp4 gate/up planes and their LOAD-TIME packed SF, as the
    /// bases the shim indexes (`base + e*NP*K/2`, `base + e*sf_words*NP*4`).
    /// `None` on a non-uniform pool: the shim derives the expert offset
    /// ARITHMETICALLY, so a ragged pool would be a SILENT wrong answer.
    fn moe_bs_weights(
        ld: &LayerDev,
        dim: usize,
        inter_local: usize,
    ) -> Option<(*const c_void, *const c_void, *const c_void, *const c_void)> {
        let (a, b) = (&ld.experts[0], &ld.experts[1]);
        let (w1, w3) = (a.w1.as_ref()?, &b.w1.as_ref()?);
        let (s1, s3) = (a.wsf1.as_ref()?, b.wsf1.as_ref()?);
        let (u1, u3) = (a.w3.as_ref()?, b.w3.as_ref()?);
        let (t1, t3) = (a.wsf3.as_ref()?, b.wsf3.as_ref()?);
        let k2 = (dim / 2) as i64;
        let w_expect = (inter_local as i64) * k2;
        let sf_expect = ((dim / 128) * inter_local * 4) as i64;
        if (w3.ptr() as i64) - (w1.ptr() as i64) != w_expect
            || (u3.ptr() as i64) - (u1.ptr() as i64) != w_expect
            || (s3.ptr() as i64) - (s1.ptr() as i64) != sf_expect
            || (t3.ptr() as i64) - (t1.ptr() as i64) != sf_expect
        {
            return None;
        }
        Some((w1.ptr(), u1.ptr(), s1.ptr(), t1.ptr()))
    }
```

**(3) 分派**（`moe_rows` 的 gate/up 点位，`tl_ready` 那一块**之前**插入；eager 侧
`~19481` 同样插一份）：

```rust
            // ---- TILELANG BLOCK-SCALED (DSV41_MOE_TILELANG_BS): native fp4 -------
            // Mutually exclusive with the bf16 arm above; the block-scaled arm WINS
            // when both are armed (they read different weight pools, so running both
            // would pay the bf16 mirror for nothing). One-shot shadow notice.
            let bs_ready = self.moe_bs_ready(ld, m, topk, dim, inter_local, n_routed);
            let mut bs_gu = false;
            if crate::dsv41::weights::moe_tilelang_bs() && !bs_ready {
                Self::moe_bs_skipped_note(
                    "the step is outside the arm's domain (a CUDA-graph capture, an \
                     INTERLEAVED pool, a frozen-shape mismatch, or the .so lacks the shim \
                     symbols / the packed-SF pool)",
                    true,
                );
            } else if bs_ready && !grp_gu {
                if moe_tilelang() {
                    Self::moe_bs_shadowed_note();   // one shot: BS wins over bf16
                }
                if let (Some(tables), Some(w)) =
                    (self.moe_tilelang_tables(m, topk)?, Self::moe_bs_weights(ld, dim, inter_local))
                {
                    let (order, counts, eid, nseg) = tables;
                    bs_gu = self.dev.moe_tilelang_gate_up_bs(
                        self.s.xq4_r.as_u8(),          // <-- fp4 激活，不是 xn_r
                        self.s.xsc4_r.as_f32(),        // <-- f32 标度
                        self.s.ex_act_r.ptr as *mut f32,
                        w.0, w.1, w.2, w.3,
                        eid.as_ptr(),
                        order.as_ptr(),
                        counts.as_ptr(),
                        *nseg as i32,
                        m as i32, dim as i32, inter_local as i32, topk as i32,
                    )?;
                }
                if bs_gu {
                    gateup_fused = false;          // RAW gate‖up（照 bf16 臂）
                    act_slot = 2 * inter_local;
                }
            }
```

**(4) 下游两处必须改成「`grp_gu || tl_gu || bs_gu`」**（否则本臂的结果会被老路径覆盖——
这正是 bf16 臂踩过的「armed gate 测成老路」陷阱）：

*   `chain_dev.rs:16044` 的 `if !grp_gu && !tl_gu {`（batched gate/up 兜底 launch）
    ⇒ `if !grp_gu && !tl_gu && !bs_gu {`
*   同理把它加进 `moe_rows`/`moe` 里任何以 `tl_gu` 为条件的 bookkeeping。

**不动的东西**：`swiglu_limit_batched`（本臂写 RAW gate‖up，与 bf16 臂同一取舍）、
`expert_down_fp4_batched`（down 不在本臂域内，§0）、`moe_down_reduce`。

---

## 6. 交付 ⑤：主 agent 的 GPU 验证手册

> **前置**：一台**空**的 B300（原型的所有绝对 µs 都是 contended 口径，被拖慢 ~2.6×；
> 跑之前 `nvidia-smi --query-compute-apps=...` 确认无 co-tenant）。
> **本手册是清单，不是脚本**——每一步的判据都在「通过/不通过」栏里。

### 6.1 STEP 0：生成 AOT（唯一合法路径）

```bash
# 本地 → 远端（生成器 + 原型，原型是 exp_* 脚本的依赖）
scp kernels/tilelang/gen_moe_bs_aot.py ubuntu@43.202.208.136:~/tl_bs/
scp kernels/tilelang/moe_bs_proto.py   ubuntu@43.202.208.136:~/tl_bs/   # 参考对照，可选

# 生成（BM=64 = 目标档）
ssh ubuntu@43.202.208.136 \
  'cd ~/tl_bs && mkdir -p aot_gen && /opt/dlami/nvme/dsv41_venv/bin/python \
   gen_moe_bs_aot.py aot_gen'

# 回传（**三份都要**：device / host 配方 / config）
scp ubuntu@43.202.208.136:'~/tl_bs/aot_gen/*' kernels/cuda/tilelang_gen/
# 然后给 moe_bs_up_tl.cu 加 banner（banner 是唯一的、可复现的本地改动）
#   -> _banner() 的内容在 aot_gen/moe_bs_up_tl.banner
```

**通过判据**：`aot_gen/moe_bs_tl_config.txt` 打印出
`is_tcgen05_fix_patched=1`（**若为 0，检查是不是升级到已修版本**，两种都可用）、
`grid=(5, 36)`、`smem_bytes=184832`、`host_source=moe_bs_up_tl_host.cu`。
**不通过**：lowering 报错（见 §7 的 BM=64 风险 → 退 `--bm 128`）。

### 6.2 STEP 1：compile-only（**无 GPU**，先做，别跳）

```bash
# 1) 生成物单独 compile-only（head/symbol 自足性）
nvcc -arch=sm_103a -cubin -O3 -std=c++17 -I kernels/cuda/tilelang_inc \
     -o /tmp/moe_bs_up.cubin kernels/cuda/tilelang_gen/moe_bs_up_tl.cu

# 2) shim（含 dlopen 的 driver 绑定；**不需要 -lcuda** —— 这正是设计目标）
nvcc -O3 -std=c++17 -shared -fPIC -arch=sm_103a -I kernels/cuda/tilelang_inc \
     -o /tmp/moe_bs_shim.so kernels/cuda/tilelang_gen/moe_bs_shim.cu
nm -D /tmp/moe_bs_shim.so | grep -E "dsv41_moe_tilelang_gate_up_bs|dsv41_moe_bs_pack_wsf"
```

**通过判据**：两条 nvcc 都 EXIT=0；`nm -D` 有**两个** `T` 符号；且
`nm -D --undefined-only /tmp/moe_bs_shim.so | grep -c cuTensorMapEncodeTiled` = **0**
（若 ≠ 0，说明 shim 直接引用了 driver 符号 —— 回到 §4.2）。
**若 shim 报「形参类型/个数不匹配」**：这就是 §4.3 的参数序/描述符形态 —— 按
`moe_bs_tl_config.txt` 的 signature 行与 `moe_bs_up_tl_host.cu` 对齐（**这是编译错误，
不是运行期问题**）。

### 6.3 STEP 2：描述符转写（§4.3 的 5 分钟检查）

```bash
grep -n "cuTensorMapEncodeTiled" -A 12 kernels/cuda/tilelang_gen/moe_bs_up_tl_host.cu
```

逐项对齐 `moe_bs_shim.cu` 的 5 个 `spec_*`。**唯一允许不符的三处**：box 首维按字节/元素、
swizzle 枚举、load_mode。改完重跑 6.2。
（运行期 `DSV41_MOE_BS_DEBUG=1` 会打印本 shim 实际用的 spec，可与 host source 一对一 diff。）

### 6.4 STEP 3：装进 `.so` 并确认符号在

```bash
# build.sh 已经把 tilelang_gen/*_shim.cu 当 TU 收进来（build.sh:52-54），无需改动。
# 但 shim 依赖同目录的 moe_bs_up_tl.cu（生成物）——先确认它已放好。
./kernels/cuda/build.sh 100a && nm -D libferrite_kernels.so | grep -E "moe_tilelang_gate_up_bs|moe_bs_pack_wsf"
```

**通过判据**：`supports_moe_tilelang_bs()` 为 true（两个符号都在）。
**不通过**：Rust 侧会 `REFUSED` 一次性提示，且**跑的是老路径** —— 别把它当成功。

### 6.5 STEP 4：parity（**逐元素**，这是本臂唯一真正要紧的检查）

三臂对照，**同一进程、同一路由表、同一激活**：

| 臂 | 环境 |
|---|---|
| 参考 | `DSV41_MOE_TILELANG_BS=0`（老路径 `expert_gate_up_fp4_batched`） |
| 本臂 | `DSV41_MOE_TILELANG_BS=1` |
| （对照）bf16 臂 | `DSV41_MOE_TILELANG=1 DSV41_MOE_BF16_DEQUANT=1` |

**检查点（逐条）**：

1. **`ex_act_r`（RAW gate‖up）逐元素比**：本臂与老路径**必须逐位相同**——
   两边的输入 nibble 与 ue8m0 标度是同一份（§2.3），内核本身无损（原型 §4 实测
   `max|err| = 0.00000`），所以**不允许有任何容差**。⚠️ 这一条同时验证了 §1.3 的
   交错列序映射：映射错了 C 的列会整体错位，逐元素比会**立刻**炸。
2. **`DSV41_MOE_BS_DEBUG=1` 的 spec 打印**：与 §6.3 的转写结果一致。
3. **首 token / 长文 prompt 的输出一致性**：≥2 个 prompt（一个短一个长），
   与老路径的采样序列一致（这是端到端的回归网）。
4. **形状门**：`nseg=36`（`m=6, topk=6` 的满段）与 `nseg=1`（`m=1, topk=1`）两个极端都要过；
   后者验证 pad 段（`counts=0`）确实被 gather 写 0、scatter 跳过。
5. **decline 路径**：`DSV41_MOE_TILELANG_BS=1` + `DSV41_EXPERT_ILV=1` ⇒ 必须
   `REFUSED` 一次性提示 + 老路径结果正确（**不是**错值）。

### 6.6 STEP 5：bench（**空卡**，绝对 µs 才是判据）

```bash
# 同刻三臂（唯一可比口径），200-400 次 launch，host-bench 与 CUDA-graph 各一次
#   A: DSV41_MOE_TILELANG_BS=1                 (本臂, 原生 fp4)
#   B: <unset>                                 (老路径 expert_gate_up_fp4_batched)
#   C: DSV41_MOE_TILELANG=1 DSV41_MOE_BF16_DEQUANT=1   (bf16 臂)
```

**判据**（判据阈值对齐 bf16 臂的既有约定，见 tilelang-moe-grouped.md §3.3）：

| 项 | 判据 |
|---|---|
| vs 老路径（SIMT 250µs/层的 up 部分） | **< 40%** |
| vs bf16 臂 | ≲ 1.0×（原型同刻 0.62×；BM=64 应更好） |
| 绝对 µs | **< 50µs**（空卡） |
| 显存 | 专家权重增量 = **+1.54 GiB/rank**（对比 bf16 臂的 +105 GiB/rank） |

**必测的两个额外项**（决定 §7 的两条风险）：

* **BM 扫**：`--bm 64` 与 `--bm 128` 各生成一次、各跑一次 parity+bench。
  若 64 的 lowering 不通 ⇒ 用 128（**功能等价、流量 +20%**），并把 config 的
  `smem_bytes` 抄进 shim（§7-1）。
* **`stages` 扫**：6（默认）→ 5/7。BM=64 的 smem 余量允许 7
  （`7*(8192+16384) + 7*192*4 + 32768 = 210176 B`，仍 < 227 KiB）。

### 6.7 STEP 6：回填

把三条数字（本臂 / 老路径 / bf16）写回**本文件 §6.6 的下方**，并把
`moe_bs_up_tl.cu` 的 raw sha256 写进 `PROVENANCE.md` 的新 §9（照 §7.2 的格式）。
生成物、banner、shim 一起 commit。

---

## 7. 已知风险与回退（按「先炸概率」排序）

1. **`BM=64` 的 lowering 有效性（唯一没被原型覆盖的点）。** 原型所有成功的配置都是
   `BM=128`。`BM=64` 满足已定谳的 `BM%64==0`，但仍有两处可能不被接受：
   (a) `GetTCGEN5MMAMeta` 的非 ws 分支是否真接受 `M=64`；
   (b) `_tcgen05_num_smem_chunks` 对 **SFA** 的「128 字粒度」约束（SFA 每个 stage 是
   `BM` 个字 = 64 个，**不足 128**；SFB 每个 stage 是 `BN=128` 个字，恰好满足）。
   ⇒ **回退 = `--bm 128`，一个 flag**：功能完全等价，代价是 A/C 流量 +20%
   （97 MB vs 77 MB/层）与 smem 顶格（`stages` 只能 6）。
   **注意**：回退后 shim 的 `kBm`/`kSmem` 两个常量必须同步（生成器的 config 里都有）。
2. **`B_sh` 子切片 TMA 的 lowering。** 两次 TMA 落进同一 swizzled 缓冲的半块（§1.3）。
   若 lowering 拒绝部分切片 ⇒ 用 §2.4 的**池面重排**（`[w1][w3][...]`，零副本零显存），
   或退到「W1/W3 各一个 kernel、N=320 再 pad」的形态（不推荐，要 384 行的 pad 面）。
3. **描述符转写错项**（§4.3）。表现：`CUDA error 716` / 结果乱。`DSV41_MOE_BS_DEBUG=1`
   的 spec 打印 + host source diff 是第一诊断手段。
4. **`dlopen` 在某些沙箱里被禁。** 表现：INIT 期 `tensormap init FAILED` 一次性提示 +
   永久 decline（**安全失败**，不会静默错）。替代：给 `build.sh` 加 `-lcuda`
   （一行，但会改 `BUILD_ID` 的 flag 集，要重建 both products）。
5. **EAGER 限制（继承自 bf16 臂）。** `moe_align` 要 D2H 回读 ⇒ capture 内 decline。
   本臂**不增加**这条限制（用的还是同一套 host table），但也不解决它。

**Plan B（若 TMA 描述符路线整体不可行）**：装载期把权重从 packed fp4 **展开成
`float4_e2m1_unpacked`（1 B/元素）的副本**（+35 GiB/rank），则全局操作数变成 1 B/元素，
可以走**纯 1D bulk（无 tensormap）**——正是 `kernels/cuda/dsv41_experts_mxf4.cu` 已在
生产用的那条路（`tc5_bulk_g2s`，`dsv41_experts_mxf4.cu:4381` 的注释）。代价：比本方案
多 35 GiB/rank（但仍比 bf16 臂的 +105 GiB/rank 少 3×），且与「零副本」的立意相悖。
**只在 §6.2/§6.3 反复失败时才考虑。**

---

## 8. 纪律

* 生成物（`moe_bs_up_tl.cu` / `moe_bs_up_tl_host.cu` / `moe_bs_tl_config.txt`）
  **禁止手改**；改动一律走 §6.1 的重生成。shim 是**手写件**，可以改——但 §4.3 的
  描述符必须**对 host source 转写**，不许猜。
* 生成物与 shim 一起 commit（shim `#include` 生成物；缺一编译不过）。
* 逐位可比性依赖两件事，任一变就必须重跑 §6.5：**（i）** `f_pow2_to_ue8m0` 与
  `dsv41_experts_mxf4.cu:287` 同源；**（ii）** TileLang 的 ue8m0 语义（`2^(b-127)`）
  不变。升级 TileLang 时先看 `gen_moe_bs_aot.py::install_blockscaled_fix` 的返回
  （`upstream-fixed` 是**好消息**：上游把 `is_tcgen05` 补上了）。
* 本臂与 `DSV41_EXPERT_ILV` **互斥**（§5.3-(1)）。与 `DSV41_MOE_BF16_DEQUANT`
  **不互斥但无需要**（本臂不读 bf16 副本；同开只是白花显存）。
