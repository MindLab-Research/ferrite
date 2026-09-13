# P3-lite **段 B（l4 / K2）** 实施 + GPU 验证手册 —— draft q 链 tail 段融合

> 工部 · 2026-09-13 · **未执行 GPU/e2e**（任务禁止）。
> 上游设计：`docs/agent/draft-p3lite-segment-fusion.md` §2.2 / §4（段 B = priority 3）。
> 性能口径：`docs/agent/mtp-verify-amortization-model.md`（唯一权威）。
> 改动面：**1 文件**（`crates/ferrite-models/src/dsv41/dspark_dev.rs`，纯 Rust 接线，**零 .cu 改动**）。
> 本机 `cargo check --workspace --all-targets` **EXIT=0**；远端 nvcc 13.2 `sm_103a` compile-only 见 §6。

---

## 0. 结论先行（五条）

1. **已实施段 B（l4 / K2）**：draft 的 q 链 tail
   `rmsnorm(qr)` + `quant1(qr)` + `gemm(wq_b)` + `rope(q)` → **1 发**
   `dsv41_gemm_fp8_mrows_rope_norm`（K2）。门 = `DSV41_P3LITE_Q_ROPENORM`（默认 OFF，
   跟总开关 `DSV41_DRAFT_P3LITE`）。
2. **R1 同程序律是代码里的前置条件，不是注释**：l4 **只在 `DSV41_ATTN_PROJ_ALIGN` 为真时发射**。
   K2 的 gemm 段是 `gemm_fp8_mrows_kernel<M>`（= `proj_attn_mrows` 的程序），而未对齐的 draft
   `wq_b` 走 `gemm_fp8_mx@m=bs`（16 行 TILE MMA）。单开 l4 = **跨两个程序且换掉一个**，
   正是 P3B 那一类复合主张 ⇒ 由代码显式拒绝。
3. **票面必须按 `DSV41_DRAFT_P3A` 是否同开分两档报**（本手册相对设计文档 §2.2 的修正）：
   | 配置 | q 链参考发数 | l4 后 | 净省 |
   |---|---|---|---|
   | P3a **OFF**（默认，rope = `×bs` = 5 发） | 10 | 3 | **−7/block = −21 发/步** |
   | P3a a4 **ON**（`DSV41_DRAFT_P3A`，rope = `rope_mrows` = 1 发） | 6 | 3 | **−3/block = −9 发/步** |
   **两折在 rope 上重叠**（l4 把 rope 折进 K2，a4 把 rope 折成 1 发）⇒ 全栈 A/B 时看到的是 **−9/步**，
   不是 −21/步。做 A/B 时**必须两个配置各跑一次**，否则会把 a4 的已入账收益算进 l4。
4. **段 A（hc 前端 `dsv41_draft_hc_front`）本轮未落，且是硬阻塞**：它需要**新核 + `device.rs` 的
   launcher / 符号探测**，而 `device.rs` 在本轮冲突防护白名单之外（设计文档 §7-3 同判）。
   §7 给出可直接执行的落地规格，等一个 `device.rs` 窗口。
5. **未引入任何新符号 / 新依赖 / 新分配**：K2 与它的 Device 方法
   （`gemm_fp8_mrows_rope_norm`，`device.rs:4817`，`Option` + 符号探测）**已在 HEAD**
   —— 这是选段 B 而非段 A 的第二个理由（零 FFI 风险）。

---

## 1. 改动清单（唯一实施项）

| 文件 | 改动 |
|---|---|
| `crates/ferrite-models/src/dsv41/dspark_dev.rs` | ① `DraftP3Lite` 加 `q_ropenorm: bool` + 字段文档 + `draft_p3lite()` 解析 `DSV41_P3LITE_Q_ROPENORM`；② `draft_p3lite()` 的文档表加 l4 行并改写"WHAT IS NOT HERE"（K2 从"不在"变"在"，wo_b/B6 保持不在）；③ `draft_attention` 的 q 链 tail 折叠 |
| `kernels/cuda/dsv41_kernels.cu` | **未改**（K2 `dsv41_gemm_fp8_mrows_rope_norm` 已在 HEAD，:7055） |
| `device.rs` / `chain_dev.rs` / `*.cu` 其余 | **未碰** |

### 1.1 折叠形式（`draft_attention`）

```
参考（OFF 臂，逐位基线）                         l4（ON 臂）
─────────────────────────────────            ─────────────────────────────────
⑥ quant1(xn)               1 发              ⑥ quant1(xn)                 1 发
⑦ wq_a: proj_attn_mrows    1 发              ⑦ wq_a: proj_attn_mrows      1 发
⑧ rmsnorm(qr, q_norm)      1 发              K2: [1a norm + 1b write-back
⑨ quant1(qr)               1 发                   + 1c quant + 2 gemm(wq_b)
⑩ wq_b: proj_attn_mrows    1 发                   + 3 rope(q)]             1 发
⑪ rope_queries(q)×5        5 发  (a4 ON 时 1 发)
                          ─────                                          ─────
                          10 发 (a4 ON: 6)                                3 发
```

发射条件（三者全真才发射；任一假 ⇒ 走参考臂）：

```rust
if draft_p3lite().q_ropenorm          // env，捕获期常量
    && attn_proj_align()              // R1：同程序（env，捕获期常量）
    && self.pos_dev == Some(rope_pos as i32)   // 位置源契约（步不变）
```

---

## 2. 逐位等价的**指令级**论证（段 B）

K2 的四个段在核头 `dsv41_kernels.cu:6797-6825` 逐段给出参照核；本节把每一段落到
**fma 链顺序 / 累积器状态 / launch 边界**三件事上，并逐条对上被替换的 launch。

### 2.1 段 1a/1b（norm + 写回）—— 替换 `self.dev.rmsnorm(qr, q_norm, qr, bs, ql, eps)`

| 项 | 参考：`ferrite_rmsnorm`（`ferrite_kernels.cu:329`，另一 TU） | K2 段 1a/1b | 判据 |
|---|---|---|---|
| launch 形状 | `<<<n, 1024>>>`（`:335-337`） | `__launch_bounds__(1024)`，`nwarps=32`（`:6863`） | **同 blockDim ⇒ 同树宽** |
| 求和走序 | `for (i = tid; i < dim; i += blockDim.x) ss += xr[i]*xr[i]`（`:305-307`） | `for (i = tid; i < k; i += blockDim.x) ss = __fmaf_rn(xr[i], xr[i], ss)`（`:6900`） | 走序相同；**每项舍入被 pin 成 fma.rn** |
| warp 折 | `__shfl_down_sync(f, off=16..1)`（`:310`） | 同（`:6903`） | 同树、同 off 序 |
| 跨 warp 折 | `red[32]`；`tid==0` 时 `for (i < blockDim.x>>5) t += red[i]`（`:314-320`） | `s_red[32]`；`tid==0` 时 `for (i < nwarps) t += s_red[i]`（`:6906-6911`） | **升序、同宽**（blockDim 同为 1024 ⇒ 32 个部分和） |
| inv | `rsqrtf(t/dim + eps)`（`:320`） | `rsqrtf(t/k + qr_eps)`（`:6911`） | 同式 |
| 写回 | `or_[i] = xr[i]*inv*w[i]`（`:325`） | `qr_norm_out[r*k+i] = v`，`v = xr[i]*inv*qr_w[i]`（`:6925-6926`） | **同表达式、同左结合序；每元素一次写**（`qr_norm_out == qr_raw` 的 in-place 就是参考臂的 `out == x`） |

**⚠️ 唯一的跨 TU 项**（本段最需要回执的地方，R4）：`ss` 的每项舍入。
参考在 `ferrite_kernels.cu` 里写的是**普通** `ss += xr[i]*xr[i]`，nvcc 默认 `-fmad=true` 会收缩成
`fma.rn`（**一次**舍入）；K2 在一个大融合核里被内联，编译器**可以拒绝**收缩而发 `mul.rn`+`add.rn`
（**两次**舍入）。K2 用显式 `__fmaf_rn` 把这一项从"取决于内联上下文"变成"与上下文无关"。
⇒ 这在源码级是**已论证**的（`:6886-6898`），在数值级是**已实测**的（R2 的 parity `T4.norm-ferrite`：
`rmsnorm_rows == ferrite_rmsnorm`）。**本条必须有 parity 回执才算过**（§5 的 `p3lite_q_norm_parity`）。
`fast_round_scale` 是 2 的幂量化器：`ss` 差 1 ULP → `inv` 差 1 ULP → 窗口 32 lane 的 `amax` 落在
`2^k` 边界附近时 **`sc` 翻到下一个 2 的幂、整组 32 字节重量化** —— 这就是为什么这一条不能只靠读码。

### 2.2 段 1c（fp8 发射）—— 替换 `self.quant1(qr, bs*ql)`

- 参考：`quant_kernel<0>`，block = **32**（一线程组 = 一个 warp 的 32 lane），
  `amax` 由 32 lane `__shfl_xor` 蝶形求出，`sc = max(fast_round_scale(a,1/448), 1e-30)`。
- K2：**逐句同体**（`:6924-6935`）——`i += blockDim.x` 的走序、同一蝶形、同一 `fast_round_scale`、
  同一 `max(...,1e-30)`、同一 ±448 夹取、同一 `__nv_fp8_e4m3`；`i>>5` 就是 `quant_kernel<0>` 的
  字节组号 `b`，本 warp 的蝶形正好覆盖 `[32b, 32b+32)`。
- **累积器状态**：段 1c 把 fp8 字节与 scale 写进 **smem**（`s_a` / `s_as`），**不写 global `xq/xsc`**
  （见 §3 的写集表）。⇒ 融合臂下 `xq/xsc` 停在 ⑥ 的 `xn` 量化上，与参考臂在这一行**不同**，
  但**没有任何读者**：下一处使用是 `:2070` 的 `quant1(xn)`（kv 链），它**重写** `xq/xsc`。
  已按 code 复核 2022-2070 之间无 `xq/xsc` 读者（唯一插入点 `q_pre_rope` dump 读的是 `self.q`）。
- **launch 边界**：参考的 `quant_kernel<0>` 是**独立 launch**，块间无耦合（每 32 字节组自洽）；
  K2 把它放进同一个 block（1024 线程）里做，`__syncthreads()`（`:6941`）保证 `s_a/s_as` 在
  段 2 stage 之前完整。**每元素仍只被一个线程写一次**，无跨行/跨块归约。

### 2.3 段 2（wq_b GEMV）—— 替换 `self.proj_attn_mrows(wq_b, ...)`

即 `gemm_fp8_mrows_kernel<M>`（`m=bs=5`），K2 的段 2 是**逐句照抄**（C1-C6）：

- **同 K 走序**：`kb = 0..nb_k-1` 升序、`j = kb*32 + lane`（`:6980-6981`）。
- **同单链累积**：`acc[r] += av * wv`（`:6988`），**无 K-split、无多累加器**。
- **同权重 / 激活来源**：权重经 cp.async16 进 `s_w + warp*k`；激活 = 段 1c 写下的 `s_a`/`s_as`
  字节，与 `proj_mrows` 读 global `xq/xsc` 的字节**同一批值**（段 1c 的 `xq/xsc` 语义词相同）。
- **同归约**：`shfl_xor` 树只在 32 lane 内（`:6995`），与 blockDim 无关。
- **同写出**：`out[r*out_stride + row] = a_r + bias`（`:7001`），`out_stride = nh*hd`、`bias=nullptr`
  （参考臂同样传 `nullptr`）。
- **launch 边界**：`__syncthreads()`（`:6956` / `:7008`）只定序；段 2 的每一行由**一个** warp 独占
  （`row = blockIdx.x*nwarps + warp`），无跨块归约。

> **R1 检查**：参考臂在 ALIGN 下就叫 `proj_attn_mrows` ⇒ `self.dev.gemm_fp8_mrows(..., rows=5, n=nh*hd, k=ql, out_stride=nh*hd)`。
> K2 的段 2 是**同一个 kernel 符号**（`gemm_fp8_mrows_kernel<5>`），只是 launch 边界从"独立一发"
> 变成"融合核里的一个相位"。**无程序更换** ✓

### 2.4 段 3（rope epilogue）—— 替换 `self.rope_queries(q, rope_pos)`（`×bs` 或 `rope_mrows`）

| 项 | 参考 | K2 段 3 | 判据 |
|---|---|---|---|
| 位置源 | `rope_at`：`t = (*base)*mul + off + r*step`，`off = rope_pos - pos_dev` ⇒ `t = rope_pos + r`；`rope_mrows`：直接读 `pos_rows[r] = pos_dev + r` | `t = pos_rows[r]`（`:7021`），**无 `pos_ctr`/`mul`/`off`/`step`** | 三者给**同一个整数**（`rope_queries` 的 `rope_at(rope_pos+r)` 与 `rope_mrows` 的 `pos_rows[r]` 在 `pos_dev == rope_pos` 时同为 `rope_pos + r`）。**因此 l4 显式要求 `pos_dev == rope_pos`**（§1.1 第三个条件） |
| 旋转区 | `apply_rope` 的尾部 `2*half = rope_rd` 列 | head 的 `[h*rope_hd + rope_hd - rope_rd, +rope_rd)`（`:7012-7014`，`sect = rope_hd - rope_rd`） | **同区**（尾部 rd 列） |
| 旋转式 | `x0*c - x1*s` / `x0*s + x1*c`，`inverse` 对 `sin` 取负 | 同（`:7028-7029` + `:7024` 的 `rope_inverse ? -1 : 1`） | 同式；本调用 `inverse = false` |
| 表索引 | `cos[t*(rope_rd>>1) + i]` | 同（`:7022`） | 同 |
| 无归约 | 每元素只写一次 | 每元素只写一次（pair 由偶偏移 warp 写） | **线程映射不可能移动值** |

- **launch 边界**：段 3 **只能**在同一 block 内做——pair `(row, row+1)` 跨 warp，`rope_hd % 32 == 0`
  保证 pair 不跨 block（`:6827-6834`，launcher 对 `rope_hd & 31` 返回 2）。第二个
  `__syncthreads()`（`:7008`）保证"段 2 的全部写 happens-before 段 3 的覆写"，
  否则同一地址两个无序写 = **未定义**（不是"通常正确"）。

### 2.5 P3B 三个洞的堵法（逐洞对账）

| 洞 | P3B 的破法 | 段 B 的堵法 | 可证形式 |
|---|---|---|---|
| **① 污染回退输入**（b1：`swiglu_limit_q` 覆写 `xq` 里那份"两臂之外只量化一次"的 `xn`，fallback 读到被污染输入） | 融合尝试**部分成功**后回退 | **K2 在 decline 时不发射任何 kernel**：launcher 的全部 `return 2` 判定（`:7060-7077`）都在 `switch` 之前，`cudaFuncSetAttribute` 的失败也 `return (int)e`（`:7112`）而不发射。⇒ 融合的写集为空。回退臂随后 `rmsnorm(qr)`（重写 qr）、`quant1(qr)`（重写 `xq/xsc`）、`gemm`（重写 `q`）、`rope_queries`（重写 `q`）—— **回退是完整的第三条路径**，不读任何被污染的中途状态 | 代码级（launcher 结构 + 写集表） |
| **② 条件式等价**（b3 的 "bit-identical **PROVIDED** the per-row path would take v2"；同族的 `head_gemv_bf16_mrows` 已被实测证伪） | 把"同 program"当读码可得 | 段 B 的等价**不依赖任何"per-row 会走哪条路"的条件式**：`attn_proj_align()` 是**显式 env**（不是 shape 推断），且 l4 由代码要求它为真；语义上"OFF 臂 = ALIGN 下的参考臂"是**同一门**控制的同一条臂。段 1c/2/3 的等价性是**同符号同体**（§2.2-2.4），段 1a 的跨 TU 项**必须拿 parity 回执**（§5），不接受纯源码论证（R4） | 需实测回执（parity + 双门禁） |
| **③ 图捕获期分支永久化**（b1 的 host 分支在捕获期解析，replay 不再判；replay 期日志无法区分融合臂与半融合 fallback 臂） | 捕获期选了 fallback，被永久写进图 | 三个发射条件**全部是"步不变"的**：`q_ropenorm`（env 常量）、`attn_proj_align()`（env 常量）、`pos_dev == rope_pos`（同一 arm 内两者同步推进 ⇒ 要么恒真要么恒假）。⇒ **捕获期与回放走同一条路径**，不存在"捕获时选了 A、回放时该选 B"的窗口。K2 的 launcher decline 也是 (shape, env) 的**纯函数**（bs=5/ql=1280/nh*hd=32768/rope_hd=512 固定）⇒ 捕获与回放一致 | 代码级（不变性论证） + GPU 单变量二分（R5） |

**R2（launch 级 all-or-nothing）**：l4 是**一次调用**接管整条相位链（要么 1 发 K2，要么 0 发 + 参考 4 发），
没有"试发 4 发、第 4 发失败再回退"。
**R3（回退不读被污染输入）**：见 §3 的写集/读集表。
**R4（离散消费者）**：段 B 的输出 `q` 的下游是 `sparse_attn` → `o` → … → head **argmax** ⇒
l4 **必须**过 parity + 双门禁 + 五段全文，不能只凭源码（§5）。
**R5（一折一门）**：`DSV41_P3LITE_Q_ROPENORM` 独立可切（`=0` 单独关）。

---

## 3. 写集 / 读集（R3 的对账表）

**K2 的 global 写集（读码复核，`dsv41_kernels.cu:6924-7033`）**：

| 地址 | 内容 | 谁来写 |
|---|---|---|
| `qr_norm_out[r*k + i]` | 归一化后的 qr（**in-place 时就是 `qr`**） | 段 1b（`:6926`） |
| `out[r*out_stride + row]` | wq_b 的 GEMV 输出（= `q`） | 段 2（`:7001`） |
| `out[...]` 尾部 `rope_rd` 列 | rope 后的 pair | 段 3（`:7028-7029`） |
| `s_lut/s_as/s_a/s_red/s_rows` | **smem**，随 block 消亡 | — |

**global `xq`/`xsc` 不在写集里** ✓

**参考臂（OFF）的写集**：`qr`（rmsnorm in-place）、`xq/xsc`（quant1）、`q`（gemm）、`q`（rope）。
**回退臂的读集**：`qr`（段 1a 的输入 = ALIGN 下 `proj_attn_mrows` 刚写下的 raw qr）、`q_norm`、
`xq/xsc`（段 1c 的输入 = 段 1b 刚写的 normed qr）、`wq_b`、`pos_rows`、`cos/sin`。
**交叉污染检查**：K2 的写集只含 `qr`/`q`；回退臂的**第一个动作**就是重写这两者，且回退臂的输入
（raw `qr`）在 decline 时**未被任何 kernel 动过** ⇒ 与 P3B 洞 ① 的区别正是这里。

**与既有两个 P3-lite 折的缓冲不相交**：l1/l2 写 `mk`/`kv`，l3 写 `o`/`xq(o)`，l4 写 `qr`/`q`。
l4 之后 `xq/xsc` 的下一个写者是 `:2070`（kv 链的 `quant1(xn)`）——**顺序保持**。

---

## 4. 回退契约（OFF 臂逐位不变）

| 情形 | 行为 | 结果 |
|---|---|---|
| `DSV41_P3LITE_Q_ROPENORM` 未设 + `DSV41_DRAFT_P3LITE` 未设 | `q_ropenorm = false` | 走参考 4 发/10 发 —— **与今天逐位相同** |
| 门 ON 但 `DSV41_ATTN_PROJ_ALIGN` OFF | 拒绝发射（R1） | 走参考臂；**这是设计要求的配对语义**，不是降级 |
| 门 ON、ALIGN ON，但 `pos_dev != rope_pos`（`DSV41_SEED_POS` arm） | 拒绝发射 | 走参考臂；两者在 arm 内**恒不相等** ⇒ 捕获/回放一致（洞 ③） |
| 门 ON 但 `.so` 缺 `dsv41_gemm_fp8_mrows_rope_norm` | `Ok(false)`（`device.rs:4838`） | 走参考臂 |
| 门 ON 但 K2 形状 decline（返回 2） | `Ok(false)`（`device.rs:4868`） | 走参考臂 |
| K2 返回**其它** cuda 错误 | `Err`（`kerr`） | **整步失败**，不静默降级 |

> 注：`xq/xsc` 在融合臂下停在 ⑥ 的 `xn` 量化（K2 不写它们），与参考臂的 `qr` 量化不同；
> 但 2022-2070 之间无读者，`:2070` 重写。**已按 code 复核**（§2.2）。

---

## 5. 验证手册（双门禁 + parity）

### 5.1 本机（无 GPU）已做

```
1. cargo check --workspace --all-targets      → EXIT=0（本次）
2. 远端 nvcc compile-only（.so 必须与 Rust 同轮重建）：
   nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 -Xptxas -v \
        -c kernels/cuda/dsv41_kernels.cu
   ⇒ 段 B **未改 .cu**，这一步是"树可建"的回归门（含 peer 的 mpar 改动），不是本折的必需项
3. 符号三证（沿用既有两个折的门禁）：
   nm -D $SO | grep -c dsv41_gemm_fp8_mrows_rope_norm    # 必须 = 1
```

### 5.2 parity（**先于 serve A/B**；R4）

| 测试 | 对比 | 判据 |
|---|---|---|
| `p3lite_q_rope_parity` | `gemm_fp8_mrows_rope_norm(bs=5, n=nh*hd, k=ql, out_stride=nh*hd, rope_rd, rope_hd=hd, inverse=false)` vs `rmsnorm(qr) + quant1(qr) + proj_mrows(wq_b) + rope_queries(q)` | `qr` **逐位** + `q` **逐位** |
| `p3lite_q_norm_parity`（**本折的头号门**） | K2 段 1a 的 `ss`/`inv`/`v` vs `ferrite_rmsnorm` 的对应量（`bs=5, dim=ql=1280`） | `ss`/`inv`/`v` **逐位**（跨 TU 的 `fma.rn` 收缩项，`__fmaf_rn` pin 的验证点） |
| `p3lite_q_quant_parity` | K2 的 `s_a`/`s_as` vs `quant_kernel<0>` 的 `xq/xsc` 字节 | 字节**逐位**（32 字节组 + scale） |
| 边界 | `m=1`（退化）/ `m=8`（K2 的 dispatch 上界）/ `rope_rd = rope_hd = hd`（全头旋转）/ `rope_rd = 2`（最小偶） | 不崩 + 逐位 |
| decline 边界 | `m=0` / `m=9` / `k & 31 != 0` / `n & 31 != 0` / `out_stride < n` / `rope_hd & 31 != 0` / `DSV41_GEMV_FP8_MODE=0/1` / `DSV41_NO_GEMV_FP8=1` | 返回 2（**不发射、不写**）+ 参考臂逐位 |

**`p3lite_q_norm_parity` 是本折唯一"源码不够、必须实测"的门**：它是跨编译单元的
`a*b+c` 收缩问题（§2.1 ⚠️），且下游是 2 的幂量化器（1 ULP 会翻整组 scale）。

### 5.3 serve A/B（**一臂一进程，200 tok**）

> ⚠️ **两个基线，不是两个臂**：段 B 的参考臂**本身**随 `DSV41_DRAFT_P3A` 变化（§0-3）。

```
序列 1（P3a OFF，验证 −21 发/步 这一档）
  B0  ALIGN=1, P3A=0, P3LITE_Q_ROPENORM=0   → draft=?, mean-k=A0基线(1.34), 全文
  B1  ALIGN=1, P3A=0, P3LITE_Q_ROPENORM=1   → draft 应降 ~0.20-0.30ms(推), mean-k ≥ 1.34

序列 2（P3a ON，验证 −9 发/步 这一档；也是生产全栈的真实增量）
  C0  ALIGN=1, P3A=1, P3LITE_Q_ROPENORM=0   → draft=?, mean-k
  C1  ALIGN=1, P3A=1, P3LITE_Q_ROPENORM=1   → draft 应降 ~0.10-0.15ms(推), mean-k

同时必须跑（R1 的反面）：
  D   ALIGN=0, P3A=0, P3LITE_Q_ROPENORM=1   → **必须与 B0 完全一致**（拒绝发射 ⇒ 零影响）
```

**双门禁（`lesion-audit §8`）**：每臂同时报 `[dspark] steps=` 的 `draft=` 字段 **AND** `mean-k`。
**`draft_ms` 降 且 `mean-k` 不掉（≥ 基线）才算过**；`mean-k` 掉 = 数值回归 ⇒ **立即弃用该门**。
**每个臂都要看五段全文**：`lesion-audit §7` 的教训是"断崖漂移"（line 62 → 52）比 mean-k 更能暴露
数值问题。`[dspark] draft=` 的分解（seed/q/kv/attn/o/wo/moe/head）是**第一指标**。

### 5.4 判据与止损

| 观测量 | 通过 | 止损 |
|---|---|---|
| `draft=` 分解里 q 链的份额 | 降到 K2 应有的量级 | 若 `draft=` 不降 ⇒ 大概率门未生效（`.so` 旧 / ALIGN OFF / `pos_dev != rope_pos`）——先查 D 臂 |
| `mean-k` | ≥ 基线 | 掉 ⇒ 弃用门，跑 §5.2 的 `p3lite_q_norm_parity` 定位 |
| 五段全文 | 无 latin fragment / 无断崖漂移 | 有 ⇒ 同上 |
| 数值形态 | 与 OFF 臂**逐位**同（若 golden harness 可用） | 任何 ULP 级差 ⇒ 段 1a 的跨 TU 项是头号嫌疑 |

---

## 6. 编译门状态

| 门 | 状态 |
|---|---|
| `cargo check --workspace --all-targets` | ✅ **EXIT=0**（本次，唯一 warning 均为既有的 dead_code/unreachable） |
| 远端 `nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 -Xptxas -v -c dsv41_kernels.cu` | ✅ **rc=0 / errors=0**（`dsv41_glue.cu` 同轮 rc=0；段 B 未改 .cu，此门覆盖树内 peer 的 mpar 改动） |
| 新符号 / 新 FFI / 新依赖 | **零**（K2 + `device.rs::gemm_fp8_mrows_rope_norm` 已在 HEAD） |
| ABI | **不变**（K2 的 C ABI 与 `.so` 均未动；Rust 侧只新增一次调用） |

---

## 7. 未落项：段 A（hc 前端）的硬阻塞与落地规格

**为什么没落**：段 A（设计 §2.1）要**新核** `dsv41_draft_hc_front` + 它在 `device.rs` 的
launcher / 符号探测（`hc_mixes`×1 + `hc_collapse` + `rmsnorm` → 1）。`device.rs` 在本轮的
冲突防护白名单之外，且设计文档 §7-3 同判（"peer 区，需一个合并窗口"）。
**不落它**是遵守给定文件集，而不是认为它没价值（§4 表：priority 4，等价置信度"高"，风险"中"）。

**落地规格（可直接执行，待 `device.rs` 窗口）**：

1. `kernels/cuda/dsv41_kernels.cu` 尾部新增 `__global__ void dsv41_draft_hc_front_kernel(...)`，
   `<<<rows, 1024, smem>>>`：
   - **phase 1 = `hc_mixes`（逻辑宽度 768）**：只让 `threadIdx.x < 768` 参与，
     `c += 768`（**显式常量**，不要写 `blockDim.x`）；`shfl_xor` 树、`wpart[32]` 跨 warp 折、
     sinkhorn 单 warp 寄存器 —— 逐句照抄 `hc_mixes_kernel:2462-2541`；
     `nwarp` 的取值必须是 **24**（768 派生），不能取 `blockDim.x>>5 = 32`。
     **decline 前置**：`DSV41_HC_MIXES_ACC4` / `DSV41_HC_MIXES_SPREAD` / `DSV41_HC_MIXES_THREADS`
     任一被设 ⇒ 返回 2（这三个都会改 `hc_mixes` 的答案，见 `:2501`/`:10871`/`:10867`）。
   - **phase 2 = `hc_collapse_norm`（宽度 1024）**：body 逐句照抄
     `dsv41_hc_collapse_norm_kernel:11093-11117`（`fmaf` 升序 hc + `shfl_down` 树 + **升序**跨 warp 折）。
   - 相位间 `__syncthreads()`（**只定序，不改值**）；smem = `mixes[24] + cm[16] + red[32]`（独立槽位，~288 B）。
2. `device.rs`：加 `draft_hc_front: Option<unsafe extern "C" fn(...)>` + `ko!(rt, "dsv41_draft_hc_front")`
   + `pub fn draft_hc_front(...) -> Result<bool>`（`Ok(false)` = 无符号 / decline）。
3. `dspark_dev.rs::draft_body`：在 `:1431` 的 `self.hc_mixes(...)` 之后接融合调用，
   带 `let mut fused = false;` 的两臂写法（与 l1/l2/l3 同构）。
4. 门：`DSV41_P3LITE_HC_FRONT`（默认跟总开关）。
5. 验收同 §5（parity 对比 `hc_mixes + hc_collapse + rmsnorm` 的 `pre/post/comb/h` 四组逐位）。

---

## 8. 未决 / 需上裁

1. **段 B 与 `DSV41_ATTN_PROJ_ALIGN` 的配对是否把 ALIGN 升格为 P3-lite 的组成部分**
   （设计 §7-2）。本实现的选择是**代码级拒绝**（不配 ALIGN 就不发射），把决定权留给 A/B 编排；
   若上裁要求"ALIGN 由 P3-lite 自动打开"，改一行发射条件即可。
2. **`p3lite_q_norm_parity` 的覆盖**：段 1a 与 `ferrite_rmsnorm` 跨 TU，建议把它加进
   `dsv41_r2_parity` 的 T4.norm 家族，成为一个常驻门而不是一次性测试。
3. **票面口径**（§0-3）：段 B 的收益与 `DSV41_DRAFT_P3A` 的 a4 在 rope 上重叠。建议所有
   P3-lite 的记账统一"按已开门的净增量"，避免重复计账。

---

*工部 · 基于 2026-09-13 仓库工作树（`58e9f36` + peer 未提交改动）。*
*launch 计数与 `file:line` 为代码精确值；ms 为推算并标「(推)」；本机无 GPU，未经 e2e。*
