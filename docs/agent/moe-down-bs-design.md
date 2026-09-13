# MoE down 方向的 blockscaled（tcgen05 fp4）实现 — 设计与对拍

> 状态：**设计 + 实现 + 编译 + 数值（CPU）已完成**；GPU 对拍/微基准与 e2e 待主 agent 执行（§7 清单）。
> 隔离 worktree：`/tmp/moe-down-bs`（branch `moe-down-bs`，基线 `fad9d76`；本文所有 file:line 以该快照为准）。
> 门控：**`DSV41_MOE_DOWN_BS`（默认 OFF）**。
> 上位：`docs/agent/moe-bs-crash-investigation.md` §47（fp4 smem 语义定谳）、§100（缺口结论）。

---

## §1 一句话结论

`expert_gemv_fp4_down_reduce_kernel`（v3 profile **1.00 ms/步、10.3%、385 GB/s、occupancy-bound**）
可以被 **同一套 §47 定谳的 tcgen05 block-scaled fp4 语义**替换：down 的 A/B 角色与 gate/up **相同**
（A = e4m3 激活、B = packed fp4 权重），换的只是 **(N,K) = (dim, inter) = (5120, 320)**
（gate/up 是 (640, 5120)）⇒ K 从 40 个 128-K 迭代缩到 **3 个 K-span**。

**但有一个必须先说清的硬事实（§5）**：tensor core 的 A 操作数**只能是 8/6/4-bit 浮点**，
所以 down 的输入**必然**经历一次 **e4m3(block 32, e8m0) 量化** —— 这与官方参考
（`ref_inference/model.py:846-849` 的 `act_quant(fp8_block_size=32)`）**一致**，与
`DSV41_ROUTED_DOWN_QUANT` 要做的是**同一件事**，但与我们**当前**的 SIMT arm（吃 f32 激活）
**不一致**。⇒ 本臂是**精度门**（数值会变，且是朝官方对齐的方向变），**不是逐字节门**。
数值量级已用 CPU 模型定标：量化差 **2.47% of rms**，实现自身的 fp 序差 **4.6e-7 of rms（max 元素相对 5.7e-5）**。

---

## §2 down 现状与几何（读码结论）

### 2.1 现状代码

| 位置 | 内容 |
|---|---|
| `kernels/cuda/dsv41_experts_mxf4.cu:2258` | `expert_gemv_fp4_down_reduce_kernel<STAGED>` —— 被替代的融合核（down GEMV + ascending slot sum 一次发射） |
| `:2164` | `moe_down_reduce_kernel` —— 非融合路径的固定序求和（本臂仍然要用它做最终合并） |
| `:3833` | `dsv41_expert_down_reduce_fp4_batched` —— 生产入口（`DSV41_DOWN_FUSE` 默认 ON 时走它） |
| `:3734` | `dsv41_expert_down_fp4_batched` —— 非融合入口（写 `[slot][dim]` partial） |
| `:926` | `g_down_fp4_mode`：down 专用 lane 映射，生产默认 **mode 3**（4 值/ lane） |
| `:165` | `dsv41_sf_pitch(k) = (k/32 + 15) & ~15` ⇒ **w2 的 e8m0 面物理行距 = 16**（`k=320` 时逻辑只有 10 字节） |
| `chain_dev.rs:23480`（eager）/ `:18713`（`moe_rows` verify） | 两处 dispatch：`down_fuse()` ON → 融合核；OFF → (down + reduce) 对 |
| `chain_dev.rs:1220-1250` | `DSV41_ROUTED_DOWN_QUANT`（默认 OFF）：官方 down 输入流水线（rw → bf16 → e4m3(32) 往返） |
| `kernels/cuda/dsv41_glue.cu:2756` | `routed_down_prep_kernel` —— 上述门的实现（= 官方 `act_quant` 语义） |

### 2.2 形状与语义（down 生产形状）

```
y[dim] = Σ_s rw[s] · ( Σ_t  x[s][t] · W2[ids[s]][dim, t] )        t = contraction = inter
dim = 5120 (= kDim，gate/up 的 K)      inter = 320 (= kNp = inter_local)
slots/topk ≤ 6，rows ≤ 6，SEG_CAP = 36，BM = 128（BS 臂的段几何，与 gate/up 同表）
w2 池：u8 [E, dim, inter/2]  行距 160 B（packed fp4，低 nibble = 偶 k）
w2s 面：u8 [E, dim, 16]      行距 16（pad 面；逻辑 10 个 e8m0 字节 = 每 32 个 K 一个标度）
输出：out[row][n]，row ∈ [0,dim)，n = 1（eager）/ m（verify）个 token 的 assignment
```

**与 gate/up 的关系（这是"方向相反"的准确表述）**：
gate/up 权重是 `[inter, dim]`（N = inter = 320，K = dim = 5120）；
down 权重是 `[dim, inter]`（**N = dim = 5120，K = inter = 320**）。
⇒ N 与 K 互换，**A/B 的身份不变**（详见 §3.1）。

### 2.3 为什么它是"最刺眼的一条"

* SIMT 核实测 **385 GB/s**（w2 每层 29.5 MB / 23.8 us），远低于 HBM 峰值 ⇒ **latency-bound**；
  报告里同时写明 issue slot ~80% 停在 FMA 口（`dsv41_experts_mxf4.cu:2218-2255` 的占用率分析）。
* 每步 40 层 × 1 次 = **1.00 ms/步**；eager（plain decode）路径。
* 它读的 29.5 MB/层 = 36 个 assignment × 5120 行 × 160 B —— **与 `|active|` 无关**，
  是 per-assignment 的重复读（专家并集去重是另一条线，见 §8）。

---

## §3 设计

### 3.1 A/B 角色推导（为什么仍然是 A = 激活）

数学上 down 与 gate/up 的 (N,K) 互换，M 的角色也不同（down 的 M = 段内 assignment 行，
gate/up 的 M = token 行）。两条备选：

| 方案 | A 操作数 | B 操作数 | C 的写出形态 | 判断 |
|---|---|---|---|---|
| **1（采用）** | e4m3 **激活** `[M=128 assignments, K=320]` | packed fp4 **权重** `[N=128 dim 行, K=320]` | `C[row=assign][col=dim]` ⇒ **每行 dim 上连续 128 个输出** | ✓ 每 assignment 一次 512 B 连续写（4 个满 128 B sector） |
| 2（swapAB） | packed fp4 **权重** `[M=128 dim 行, K=320]` | e4m3 **激活** `[N=128 assignments, K=320]` | `C[row=dim][col=assign]` ⇒ 一行 = 128 个**不同 assignment** | ✗ 单元素散写，写合并崩掉 |

⇒ **down 的 A/B 角色由"写出去的那一维"唯一决定**（gate/up 没有这个约束：它的 M = token 行、
N = 权重行都能自然写出，所以两朝向都能用）。

**第二个（更隐蔽的）理由：L2 常驻**。若 A = 权重，A 面每层 29.5 MB，会被 40 个 N-tile
（= `dim/128`）各读一遍 ⇒ **1.2 GB/层**，直接击穿 HBM；若 A = 激活，A 面每层只有
`SEG_CAP × BM × inter` = **1.47 MB**（且大部分是 pad 零），40 次重读全部落在 L2
⇒ 代价 ~0（B300 L2 远大于 1.5 MB）。**A 必须选小的一侧。**

### 3.2 几何与 K-span 分解

```
M = 128（段内 assignment 行 = BS 臂的 BM）
N = 128（dim 的 N-tile）⇒ grid.x = dim/128 = 40，grid.y = SEG_CAP = 36 ⇒ 1440 CTA
K = 320 —— **不是 128 的整数倍**：320 = 128 + 128 + 64
   ⇒ **3 个 K-span**：span0/1 各 4 个 K-block(32)，span2 只 2 个 ⇒ 共 **10 个 MMA**
     （gate/up 是 40 迭代 × 4 = 160 个）
   ⇒ **不做 K 补零到 384**：补零要求"pad 区的数据与 SF 字节都是 0"（0 × 2^128 = NaN 的静默风险），
     而 10 个 MMA 与 12 个 MMA 的代价相同（瓶颈在 B 的带宽，不在 MMA 条数）。
每个 span = 一个独立的 128×128B SW128 tile ⇒ span 内用定谳的 `ki*32B` 递进，
span 之间切 tile 基址 ⇒ **写公式/描述符/idesc 一字不改**（§47 的直接复用）。
```

### 3.3 smem 布局与算式

```
单 stage（NS=1）：A 16384 | B 16384 | SFA 512 | SFB 512 | mbar 8   = 33800 B
NS=2（cp.async 双缓冲，默认）：2×33792 + 8 = 67592 B ⇒ 3 CTA/SM（202776 B < 227 KiB）
A tile 写：uint4 到 dn_sw128_16b(m,c) = (m>>3)*1024 + (m&7)*128 + (((c^(m&7))&7)<<4)
          源：act_e4m3 + (seg*BM+m)*320 + sp*128 + c*16   （320 % 16 == 0 ⇒ 16 B 对齐）
          ⚠️ 末 span 只有 64 B/行 ⇒ 只搬 c < 4
B tile 写：uint2 到 dn_pack_sw128(n, c*8) = (n>>3)*1024 + (n&7)*128 + (((c^(n&7))&7)<<4)
          源：w2 + e*w2_stride + (n_tile*128+n)*160 + sp*64 + c*8
          ⚠️ **8 B 粒度，不能用 16 B**：16 B 会连着写容器里硬件不读的 8..15 并破坏容器 swizzle
描述符：dn_make_desc(tile, lbo=1, sbo=64, layout=2)；MMA ki 的 A/B 描述符 = base + ki*2（16 B 单位）
idesc  ：dn_make_idesc(128, 128, a_fmt=0(E4M3), b_fmt=5(E2M1), sf_id=ki)
enable_d：只在 (sp==0 && ki==0) 为 0
```

`__launch_bounds__(128, 3)`：epilogue 用 4 批 × `tcgen05.ld.32x32b.x32`（不是 x128）
—— x128 一次吃 128 个寄存器，在 170 regs/thread 的 (128,3) 预算下会 spill，
而 spill 的量正好等于**整个输出**（gate/up 手写核是 (128,1)，没有这个约束）。

**TMEM 预算（3 CTA/SM 的硬约束，已核）**：每 CTA `tcgen05.alloc` 128（C）+ 32（SF）= **160 列**，
TMEM 共 **512 列** ⇒ 3 × 160 = **480 ≤ 512** ✓。若把 launch_bounds 提到 4 CTA/SM，
第 4 个 CTA 的 alloc 会**阻塞**（不是报错）⇒ **死锁**。⇒ 改 occupancy 时这条必须一起算。

### 3.4 SF 的打包与投递（哪些层需要新拉平）

| 层 | gate/up 臂的做法 | down 的做法 | 需要新拉平吗 |
|---|---|---|---|
| 激活 SF（SFA） | **每调用** gather 期把 f32 标度打成 **group-major u32**（字 g 覆盖 128 K，byte j = K-block j） | 同：`dn_qgather_kernel` 量化时**顺手**写 `SFA[(b>>2)*SEG*BM + seg*BM + r]` 的第 `(b&3)` 字节 | 不需要（同构） |
| 权重 SF（SFB） | **装载期**一次 pack 成 `[E, sf_words*NP]`（`dsv41_moe_bs_pack_wsf`） | **直接读 w2 的原始 e8m0 面**：`*(u32*)(w2s + e*w2s_stride + row*16 + 4*sp)` | **不需要新池、不需要 loader 改动** ✓ |
| 投递 | smem 内 `dn_sf_transpose`（4×32 u32 块内转置）→ `tcgen05.cp.32x128b.warpx4` → TMEM SF 列 → MMA 的 `sf_id` 选 byte | 逐字复用 | 不需要 |

**为什么 w2 不需要装载期 repack**：w1/w3 的 SF 面**面内行距 = dim/32 = 160**（= 40 个字/行），
四字节并不天然成字；而 w2 的面是 **pad 到 16 B 的物理面**（`dsv41_sf_pitch`）⇒ 每 4 个逻辑标度
字节天然构成一个对齐的 u32。于是"读原始面 + 一个 u32 load"就是 group-major 字，
**省掉 1.3 GB 的常驻池与它的 loader 改动**。

**末 span 的 pad 字节安全性**：K=320 的 group 2 = 字节 8..11，其中 10/11 落在 pad 区。
本 kernel 每个 span 只发 `ki < nki` 个 MMA（末 span `nki=2`）⇒ `sf_id ∈ {0,1}`
**永远选不到 pad 字节** ⇒ 不需要 pad 区为零、也不会出现 `0 × 2^128 = NaN`。

### 3.5 reduce 的落点（对标 SGLang 第 12 步"专家加权归约 + 共享专家 add + all-reduce 合并"）

本臂**只做 SGLang 那一步的前半**，且是**刻意**的：

```
本臂输出  = per-assignment partial（= rw × MMA 结果），散写到 out[order[seg*BM+r] * dim + n]
            布局与 dsv41_expert_down_fp4_batched 写出的 [slot][dim] **逐格相同**
⇒ 复用既有 `moe_down_reduce`（ascending slot 的固定序求和）
⇒ 再往下沿用既有的 共享专家 add / all-reduce 载波（`moe_down_reduce_st` 的 AR 路径不变）
```

**不把 reduce 融进本 kernel 的理由（与 gate/up 臂同一条）**：一段 = 一个专家，一个输出行
（assignment）的 slots 落在**不同段的 CTA** 里 ⇒ 合并需要 atomic（非确定序，**契约禁止**）
或第二遍。既有 bf16 down 臂（`moe_tilelang_down_bf16_dev`，`chain_dev.rs:23392`）也是这个分工。
**"融合到 all-reduce 载波"这一步是既有的**（A1a：`s.o` 最后写者携带 AR store），本臂不改。

### 3.6 数值语义（本设计的核心判断）

A 操作数必须是 e4m3（`kind::mxf8f6f4` 的 a_fmt，硬件要求）⇒ **输入必然量化**。
两种 rw 落点由门控选择（见 §6）：

| 模式 | gather 做什么 | epilogue 做什么 | 语义对应 |
|---|---|---|---|
| **默认（`RWOP=0`）** | 只做 e4m3(block32) 量化 | `out = __fmul_rn(C, rw)` | 与 SIMT **同求和序**（rw 在 partial 上乘，再由 ascending reduce 合并），**唯一差别 = 输入量化** |
| `RWOP=1` | `v *= rw; v = bf16(v); 再量化` | 不乘 | **官方顺序** = `DSV41_ROUTED_DOWN_QUANT`（两者**互斥**，见 §6） |

**两条都比 SIMT 更接近官方**（SIMT 吃 f32 输入、且在 w2 之后才乘 rw；官方是先乘 rw、
再 bf16、再 e4m3 量化）。⇒ 本臂**顺带把 `DSV41_ROUTED_DOWN_QUANT` 的精度修复做进去**了。

---

## §4 实现与改动清单

### 4.1 新增文件（worktree 内，全部带 `dn_` 唯一前缀）

| 文件 | 内容 | 行数 |
|---|---|---|
| `kernels/cuda/tilelang_gen/moe_bs_dn_handwritten.cu` | down 的 tcgen05 fp4 blockscaled 核 + 复用的 VERIFIED 原语（`dn_*`）+ 有界等待 + cp.async staging | ~600 |
| `kernels/cuda/tilelang_gen/moe_bs_dn_shim.cu` | `dn_qgather_kernel`（量化+gather）+ `dsv41_moe_bs_down_dev`（launcher，DEVICE 段表）+ `dsv41_moe_bs_down_cap`（能力符号）+ INIT + 门控 | ~430 |
| `kernels/cuda/tests_dn_bs_parity.cu` | GPU 对拍 + 微基准 + **OFF 等价性断言**（§7） | ~390 |
| `scripts/dn_bs_cpu_ref.py` | 纯 CPU 数值参考与三档拆解（§5 的表就是它跑出来的） | ~330 |
| `docs/agent/moe-down-bs-design.md` | 本文 | — |

**为什么新建 shim 文件就自动进构建**：`build.sh:63-65` 用 `tilelang_gen/*_shim.cu` 通配
⇒ 新 TU 自动进 `SRCS`（仍按 `-O3`、**不带** `--use_fast_math` 的 tilelang 分支编译），
**build.sh 一行不用改**。

### 4.2 关键实现点（file:line，worktree）

* `moe_bs_dn_handwritten.cu:113` `dn_pack_sw128` = §47 的 `hw_pack_sw128`（逐字）
* `:245` `dn_tc_cp` / `:232` `dn_sf_transpose` / `:201` `dn_make_desc` / `:216` `dn_make_idesc`
  —— 与 `moe_bs_handwritten.cu` 的 VERIFIED 原语逐字同源（只加前缀）
* `:395` `dn_load_span`（顺序版）/ `:440` `dn_issue_span`（cp.async 版）—— 同一批地址两种传输
* `:520` `moe_bs_dn_kernel` —— 3 span 循环 + SF 转置/cp + per-span MMA + 有界 mbarrier 等待
* `:600` epilogue —— 4×`dn_tc_ld_x32` + `__fmul_rn` + float4 散写
* shim `:120` `dn_qgather_kernel`（量化/gather/pad 归零）
* shim `:250` `dsv41_moe_bs_down_dev`（门控 → 形状闸 → gather → MMA → 返回）

### 4.3 Rust 侧接线（待主 agent 合入；本文给出确切位置）

```rust
// device.rs: Kernels 结构体加一个字段（Option，旧 .so 解析为 None ⇒ 不 arm）
moe_bs_down_dev: Option<unsafe extern "C" fn(
    *const f32, c_long, *mut f32, *const u8, c_long, *const u8, c_long,
    *const c_int, *const c_int, *const c_int, *const c_int, *const f32, c_long,
    c_int, c_int, c_int, c_int, CuStream) -> c_int>,
// + ko!(rt, "dsv41_moe_bs_down_dev")，+ supports_* 探 dsv41_moe_bs_down_cap
// chain_dev.rs: 在 23480（eager）与 18713（verify）的 down dispatch **之前**插入：
if down_bs() && self.dev.supports_moe_bs_down() && bs_gu && !q_on {
    let dn = self.dev.moe_bs_down_dev(
        self.s.ex_act_b.ptr as *const f32, act_slot,
        self.s.ex_down_b.ptr as *mut f32,
        w2_base, w2_stride, w2s_base, w2s_stride,
        self.s.bs_eid.ptr, self.s.bs_order.ptr, self.s.bs_counts.ptr, self.s.bs_nseg.ptr,
        rw_eff, 1, /*rows=*/1, dim, inter_local, topk)?;
    if dn { /* 落到既有的 moe_down_reduce(ex_down_b, o, dim, topk) 分支 */ }
}
```
**两个必须同时满足的前置**：① `bs_gu`（BS 段表已由 gate/up 臂建好、`order/counts/eid/nseg`
都在 `s.bs_*`）；② `!q_on`（`RWOP=0` 时由 epilogue 乘 rw，`q_on` 时 rw 已在操作数里
⇒ **同时开 `DSV41_ROUTED_DOWN_QUANT` 会重复乘**，本臂**必须拒绝**，见 §6）。
⚠️ `moe_bs_weights()` 只给 w1/w3 的 SF 池；`w2/w2s` 指针要从同一作用域的
`(w2_base, w2_stride, w2s_base, w2s_stride)`（`chain_dev.rs:22846` eager / `:18200` verify）取。

---

## §5 数值对拍

### 5.1 CPU（已跑；`scripts/dn_bs_cpu_ref.py`，numpy 逐位建模两边的算术）

| 档 | 比什么 | max\|d\| | rms(d) | max 元素相对 | 相对 rms(ref) |
|---|---|---|---|---|---|
| 自检 | BS(f32) vs BS(f64 金标准) | 4.41e-4 | 7.47e-5 | 4.37e-5 | **4.14e-7** |
| **A** | **纯量化差**：SIMT(量化操作数) vs SIMT(f32) | 1.10e+2 | 2.63e+1 | 8.11e+1 | **1.03e-1**（= 2.47% of rms） |
| **B** | **纯 fp 序差（硬指标）**：BS vs SIMT(**同一份**量化操作数) | 4.88e-4 | 9.50e-5 | **5.74e-5** | **4.59e-7** |
| C | 合计：BS vs SIMT(f32)（默认臂） | 1.10e+2 | 2.63e+1 | 8.11e+1 | 1.03e-1 |
| **D** | **官方顺序**：BS(RWOP) vs SIMT(官方操作数) | 4.88e-4 | 9.39e-5 | 3.10e-4 | **4.58e-7** |
| E | 合计：BS(RWOP) vs SIMT(f32) | 1.23e+2 | 2.95e+1 | 2.08e+2 | 1.15e-1 |

**读法（这是本任务"relerr ≈ 0"的准确含义）**：

* **B 档 = 实现正确性**：把同一份 e4m3 操作数喂给 SIMT 核（求和结构完全相同），
  差异只剩 f32 累加序 ⇒ **rms 相对 4.6e-7、元素级 max 5.7e-5**。**这是"relerr ≈ 0"**：
  它是**纯 fp 序差**，量级 = fp32 的机器精度 × sqrt(累加深度)。
* **A/C 档 = 算法差（e4m3 输入量化）**：2.47% of rms —— **不是实现缺陷**，
  是 tensor core 的输入格式要求，且**与官方一致**（官方 `act_quant(e4m3, block=32)`）。
  它**不可能**通过调 kernel 消除；要"零算法差"只能不用 tensor core。
* **D 档**证明：`RWOP=1` 时本臂**逐项复现 `DSV41_ROUTED_DOWN_QUANT` 的语义**
  （同一个量化地板 1e-4、同一条 `fast_round_scale`、同一个 bf16 边界、同一个 rw 时机）
  ⇒ 两者之间**只剩累加序 4.6e-7**。⇒ 本臂 = **"把 `ROUTED_DOWN_QUANT` 的精度修复
  与 down 的 kernel 化合并成一步"**（也正因如此它是精度门，见 §6）。

### 5.2 GPU（手册在 §7；`tests_dn_bs_parity.cu`）

harness 的三个臂与 CPU 表**一一对应**（A/B/C），另有 **OFF 断言**（arm D）与 `--bench`。
`arm B` 是硬判据（脚本里 `DN_CHECK(sB.max_rel < 1e-4, ...)`）。

### 5.3 我**没有**验证的（诚实清单）

* 本 kernel 的**指令级**正确性（描述符/idesc/SF 字节序/tcgen05.cp 布局）**只在 CPU 上做了
  公式级核对与逐字节复用**；§47 的 relerr=0 是**同源原语**的硬件实证，但**新组合**
  （3-span + 末 span 2 MMA + 原始 w2s 面直读）**必须在 GPU 上跑一次 arm B 才能定谳**。
* 末 span 的 `ki∈{0,1}` + `sf_id∈{0,1}` 与 4-K-block 字的关系由 §47/§44 的既有结论推出，
  **未在硬件上单独测过**（这是本设计**风险最高**的一处，见 §8）。
* 微基准是 L2-hot，**不能**推断 e2e；`--bench` 的数字只作上界参考。

---

## §6 门控与 OFF 等价性

| 门 | 默认 | 语义 |
|---|---|---|
| `DSV41_MOE_DOWN_BS` | **OFF** | 总门。未设/非 `1` ⇒ `dsv41_moe_bs_down_dev` **立刻 `return 2`（DECLINED）**，一个字节都不写 |
| `DSV41_MOE_DOWN_BS_CPASYNC` | `1` | cp.async 双缓冲；`0` = 顺序 LDG/STS（同一批地址、同一批字节） |
| `DSV41_MOE_DOWN_BS_RWOP` | `0` | `1` = rw 进操作数（官方顺序；与 `DSV41_ROUTED_DOWN_QUANT` **互斥**） |
| `DSV41_MOE_DOWN_BS_WAITDBG` | `0` | MMA 有界等待超时的详细打印（**有界性本身无门**） |

**OFF 等价性论证（三层，逐层可查）**

1. **调用侧**：Rust 分支由 `down_bs()`（默认 false）短路 ⇒ 门关时**不产生任何新的 kernel
   launch、不改变任何指针**（`rw_eff`/`ex_down_b`/`w2*` 与改动前逐字相同），
   走的分支就是今天出货的那条（`down_fuse()` ON ⇒ 融合核）。
2. **入口侧**：即使有人误接线，入口第一件事是读门 ⇒ 非 `1` 直接 `return 2`，
   **在任何 cudaMalloc/SetAttribute/launch 之前**（因此不会污染流、不会进 capture）。
3. **可执行证据**：`tests_dn_bs_parity.cu` 的 **arm D** 用 sentinel 哨兵断言
   "rc == 2 且 `out` 一个字节都没变" ⇒ 把 OFF 等价性变成**机器可检查**的断言，而不是文字承诺。

**`.so` 层面**：新 TU 是"编译进来但运行期门控"（与 gate/up BS 臂、proj_mma、moe_align 同契约）
⇒ 符号始终在 `.so` 里，`supports_moe_bs_down()` 探的是**能力符号** `dsv41_moe_bs_down_cap`
（旧 `.so` 解析为 None ⇒ 不 arm ⇒ 绝不静默测老路径）。
⚠️ 这会让 `.so` 的 `BUILD_ID` 变化（`build.sh` 的 CU_SHA 覆盖 `SRCS`）⇒ **双产物必须同时重编**
（主 agent 的既有纪律）。

**与 `DSV41_ROUTED_DOWN_QUANT` 的互斥（必须写进判读）**：
`ROUTED_DOWN_QUANT=1` 时 rw 已被搬进激活；本臂默认又在 epilogue 乘一次 ⇒ **重复乘**。
⇒ 接线条件里带 `!q_on`；两个门同时开时**本臂 decline**（一次性 note）。
反过来说：`RWOP=1` 的臂**等价于** `ROUTED_DOWN_QUANT` 的精度效果（§5 D 档），
所以"精度转正"这件事**可以只做一次**（二选一）。

---

## §7 上机验证清单（GPU；由主 agent 执行）

### 7.1 编译（远端，双产物）

```bash
cd ~/ferrite && git fetch -q origin && git checkout <moe-down-bs 合入后的 main> 
cd kernels/cuda && bash build.sh 103a | tee /tmp/dnbs_build.log; echo BUILD_RC=${PIPESTATUS[0]}   # 必须 0
cd ~/ferrite && source ~/.cargo/env && cargo build --release
nm -D kernels/cuda/libferrite_kernels.so | grep -E 'dsv41_moe_bs_down_(dev|cap)'                  # 两个符号都在
```
**判据**：`BUILD_RC=0` 且两个符号存在（缺符号 ⇒ 这一轮的 e2e 全部作废）。

### 7.2 kernel 级对拍 + 微基准（**先做这个**）

```bash
cd ~/ferrite/kernels/cuda
nvcc -gencode arch=compute_103a,code=sm_103a -O2 -std=c++17 \
     -Ikernels/cuda/tilelang_gen -o /tmp/t_dnbs kernels/cuda/tests_dn_bs_parity.cu
CUDA_VISIBLE_DEVICES=<free> /tmp/t_dnbs --bench
```
**判据（按序）**：
1. `[arm D] gate OFF -> rc=2, out untouched=1` ⇒ OFF 等价性 ✓
2. `arm B ... max_rel < 1e-4`（期望 ~1e-5，`rms/max` ~1e-6）⇒ **实现数值正确** ✓
3. `arm A` 的 `max/rms_ref` ∈ **[1e-2, 5e-2]**（CPU 预测 2.47%）⇒ 与"输入量化"这唯一解释一致；
   **若 arm A 远大于 5e-2 ⇒ 不是量化，是 kernel 错**（先查 arm B）。
4. `--bench` 的 `speedup` 只作上界：L2-hot，生产 HBM-cold。
   （若 speedup < 1.0，说明 occupancy/pipeline 不够 —— 那时先看 nsys 的
   `moe_bs_dn_kernel` 时长与 SM 占用，再决定是否上 cp.async 调优或分组去重。）

### 7.3 e2e（门开 vs 门关；**一次一个变量、同会话背靠背**）

```bash
# 门关（对照臂；也是当前出货状态）
~/arm_run.sh DNBS_off DSV41_MOE_TILELANG_BS=0            # 或保持出货配置不动
# 门开（BS 臂；需要 gate/up 也走 BS 臂 ⇒ 段表才有）
~/arm_run.sh DNBS_on  DSV41_MOE_TILELANG_BS=1 DSV41_MOE_DOWN_BS=1
~/arm_run.sh DNBS_on_rwop DSV41_MOE_TILELANG_BS=1 DSV41_MOE_DOWN_BS=1 DSV41_MOE_DOWN_BS_RWOP=1
~/verify_correct.sh <port> DNBS_on                       # 1..61 数字 + 拉丁探针 + step p50
```
**判据（本臂是精度门 + 性能门，两类判据都要）**
* **性能门**：`[dsv41] step pos=` 行的 **p50**（同会话背靠背，一次一个变量）。
  预期：down 那一项 1.00 ms/步 → 目标 ≤0.6 ms/步（§8 的算术），总 p50 的改善按此折算；
  nsys 复核 `expert_gemv_fp4_down_reduce_kernel` 应**消失**、`moe_bs_dn_kernel` 出现。
* **精度门**（**不要**找逐字节一致）：文本红线（不重复、不乱码、1..100 / 出师表）+
  `DSV41_DIFF_EAGER=1` 的 anchor 与 EAGER 一致 + accept 不下降。
  ⚠️ 期望"数值会变"，且**方向应与 `DSV41_ROUTED_DOWN_QUANT=1` 一致**（§5 D 档）——
  若 `DNBS_on_rwop` 与 `ROUTED_DOWN_QUANT=1` 的文本/指标互相印证，则两个门可以合并转正。
* **归因**：任何文本异常先看是"BS 臂没对"还是"这个门有害"——用 `DSV41_MOE_DOWN_BS=0`
  与 `DSV41_MOE_TILELANG_BS=0` 两个开关把变量拆开（本战役已踩过"两变量混挂"的坑）。

---

## §8 不确定点与预期收益依据

### 8.1 收益算术（可复核）

```
现状：SIMT down = 29.5 MB/层（36 assign × 5120 × 160 B）@ 385 GB/s = 23.8 us/层 → 0.95 ms/步
新臂：读 w2 29.5 MB + A 1.47 MB(L2) + 写 partial 0.74 MB + gather 读写 ~3 MB ≈ 34 MB/层
      若达到 1.5~3 TB/s（cp.async 双缓冲 + 3 CTA/SM，仍低于 7 TB/s 峰值）⇒ 11~23 us/层
⇒ 预期 -0.0 ~ -0.5 ms/步。**下限是"打平"，上限 -0.5 ms/步**（占总 step ~10% 的那一项的 2×）
⇒ 不能承诺"一定赢"；这正是必须先做 §7.2 微基准的原因。
```
**本臂没有减少权重流量**（仍是 per-assignment 重复读）——真正的流量杠杆是
`DSV41_EXPERT_GROUPED_DOWN`（专家并集去重，另一条线）。本臂与它是**正交**的：
分组去重把 29.5 MB 压到 ~|active|/n_assign 倍，本臂把剩下的带宽用满。

### 8.2 不确定点（按风险排序）

1. **末 span（K 256..319，只 2 个 MMA）的 SF/描述符**：由 §47/§44 推出，未单独硬件验证。
   若 arm B 的误差是**系统性**的（不是 1e-7 级），第一嫌疑就是这里：
   先试"K 补零到 384 + 4 个 MMA"（8.3 的开关），能分清是几何还是别处。
2. **A 面 40 次重读是否真的落 L2**：1.47 MB 远小于 L2，但如果被 w2 的 29.5 MB 流挤出去，
   会退化成 HBM 流量（+59 MB/层）。缓解：`cp.async.ca`（本实现用的是 `.ca`/`.cg` 混合，
   A 走 `.cg` = bypass L1，不影响 L2）。**若微基准显示 A 的重读代价大**，改成
   "CTA 内循环 N-tile、A 常驻"（smem 49 KB，grid 变 (4, 36)）—— 已在设计里留了路。
3. **cp.async 的 3 CTA/SM 是否够**：67592 B × 3 = 202776 < 227 KiB ✓（与 gate/up 同数），
   但 A 的两份 stage 是 49 KB/span 的大块，LDG 延迟可能仍是瓶颈。
4. **`__launch_bounds__(128,3)` 的寄存器**：epilogue 拆成 4×x32 后预计 ~60-80 regs，
   若 ptxas 报 spill，先降 (128,2)（smem 允许 3，占用率会掉）。
5. **e2e 除 down 外的其它变量**（rw 时机 / bf16 边界）**只在本臂的 RWOP 模式下对齐**；
   默认模式仍与官方差一个"rw 在 bf16 之后"的顺序 ⇒ 默认臂的精度收益**小于** RWOP 臂。

### 8.3 未做（明确留白，避免被误读为"已完成"）

* **verify（m>1）侧的接线**：kernel 本身对 rows>1 是通用的（段表用 flat assignment index），
  但 `chain_dev.rs:18713` 的 verify dispatch 需要同样的 `bs_*` 段表作用域确认 —— 留给主 agent。
* **K 补零到 384 的对照臂**（用 `DN_SPANS=4` 编译期变体）——用于 §8.2 的第 1 项定位。
* **装载期 w2 SF 池**：当前直读原始面（省 1.3 GB）；若 e2e 显示 SFB 的 4 B 随机读是瓶颈，
  再考虑装载期 repack（成本 1.3 GB/rank）。
* **分组去重合并**（`DSV41_EXPERT_GROUPED_DOWN` 的 BS 版）。
* **PDL**：本臂用普通 launch（生产者是 gate/up；PDL 只是优化，不是正确性所需）。

---

## §9 与 gate/up 臂的逐项对照（复用清单）

| 项 | gate/up | down（本设计） | 复用方式 |
|---|---|---|---|
| fp4 smem 语义 | `hw_pack_sw128` | `dn_pack_sw128` | 逐字 |
| 描述符/idesc/KS 递进 | `(1,64,2)` + `ki*2` | 同 | 逐字 |
| SF 投递 | `sf_transpose` + `tcgen05.cp.32x128b` | 同 | 逐字 |
| MMA 形状 | M=128, N=128, K=128/span ×4 | M=128, N=128, K=128/span（末 span 2 个 MMA） | 同形 |
| A/B 身份 | A = e4m3 激活 | A = e4m3 激活 | 同 |
| staging | cp.async 双缓冲（`DSV41_MOE_BS_CPASYNC`） | cp.async 双缓冲（默认 ON） | 同构 |
| 有界等待 | `whp_*`（§93） | `dn_*` | 同构 |
| 段表 / `nseg` | DEVICE 表（`dsv41_moe_align_from_group`） | **同一个表** | 直接用 |
| 输出 | scatter 回 RAW gate‖up | **直接散写 partial（省掉 scatter pass）** | 简化 |
