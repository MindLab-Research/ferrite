# m=6 投影 GEMM 的 tensor-core（mma）路线 — 设计

> 载体（实施框架骨架，已远端 compile-only 通过）：`kernels/cuda/dsv41_proj_mma_skel.cu`
> —— `gemm_fp8_mrows_mma_kernel<M>` + launcher `dsv41_gemm_fp8_mrows_mma` + gate `DSV41_PROJ_MMA`。
> 前置：`mtp-verify-amortization-model.md`（verify(m) ≈ eager(1)+ε）、
> `verify-amortization-lesion-audit.md` §2/§9/§10（mrows 零摊销 + MPAR 二连败）、
> `mrows-mpar-design.md`（M-in-register vs M-in-grid 死锁 + MPAR 全设计）。
> 日期：2026-09-13。**GPU 未验证**——本文是设计与验证手册，不是实测结论。
> 纪律：本任务禁止 GPU/e2e；远端 nvcc compile-only 允许（已用）。

---

## §0 一句话

mrows 摊薄的死结不是"M 没并行"，是**每条 (元素, k) 乘积都要单独发射一条指令**：
`gemm_fp8_mrows_kernel<M>` 每 warp 每 32-k 块发 ~34 条指令只retire 6×32 = 192 个 MAC
（DRAM 0.7% / Compute 13% / L1 14% / occupancy 13.4%——**没有任何一条管线饱和**，
典型 latency-bound）。tensor core 用**一条** `mma.sync.m16n8k32` 退休 16×8×32 = 4096 个 MAC
——**每 MAC 指令数降 ~30-45×**，且加权重的 cp.async 深环把 DRAM 延迟从 accumulate 上摘下来。
映射取 **swapAB**（权重在 MMA 的 M、激活在 N=8 列）⇒ 6 行只占 8 列 ⇒ **利用率 6/8 = 75%**，
不是 pad-16 的 37.5%。**这条路不能逐位等于 SIMT**（§2 证明），但能满足工程真正要的那条契约：
*m 行 launch 的 row r ≡ 同一程序 m=1 的 row r*（§2.4，逐位成立，且树内已有先例 `DSV41_SWAPAB`）。

**五问判决**：

| # | 问题 | 判决 |
|---|---|---|
| ① | mma 路线可行性（tile/利用率/带宽账） | ✅ **可行**。`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`，swapAB，6/8=75%，权重流量 1×，网格 640-1024 warps。tcgen05 四条硬理由出局（§1.4） |
| ② | 数值等价性两选项 | ❌ 逐位不可能（§2.2）。(a) 论证稳健 = 只能实测、风险高（§2.3）；(b) "非判定路径"**不存在**（§2.4）——但有 **(b′) program-consistent parity**（§2.4），它才是真正可交付的那条 |
| ③ | 实施框架 | kernel 签名 / gate `DSV41_PROJ_MMA`（默认 OFF）/ 与 legacy & MPAR 的互斥切换 / ks 规则 / scratch 约定（§3） |
| ④ | 收益估算 | **−3.5ms 中央值**（verify 24.5 → 21.0），区间 [−2.5, −4.5]（§4） |
| ⑤ | 备选路线再评估 | ⑤a **L2 无 smem 广播**（逐位安全、最简）、⑤b **cluster DSMEM 权重广播**（逐位安全，正面回答 MPAR 的 prologue 死因）；两条都比 fold_r/MPAR 强（§5） |

---

## §1 B300（sm_103a）的 mma 路线调研 — deliverable ①

### 1.1 指令可用性（树内已实测的清单，不重猜）

来源：`dsv41_kernels.cu:13-38`（sm_103a / CUDA 13.2，带显式 `-gencode` 与 `.target sm_103a` 验证过）+ `dsv41_experts_mxf4.cu:8-16`。

| 指令 | sm_103a 判决 | 出处 |
|---|---|---|
| `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` | **OK（唯一可用的 fp8 mma.sync）** | `dsv41_kernels.cu:17`，实际使用于 `gemm_fp8_kernel:317`、`gemm_fp8_swapab_kernel:801` |
| `mma.sync ... kind::f8f6f4` / 任何 fp4 e2m1 `mma.sync` | **REJECTED**（"Instruction 'mma with FP6/FP4 floating point type' not supported on .target 'sm_103a'"） | `dsv41_kernels.cu:18` |
| `wgmma`（Hopper 族） | **sm_103a 无此 ISA**——Blackwell 用 tcgen05 取代 wgmma（树内零使用、零声明） | 反面证据：全树 grep 无 `wgmma` |
| `tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X` | OK，但 **fp4 专用**（tmem 累加器 + smem 操作数 + tmem block-scale 描述符 + mbarrier 完成） | `dsv41_experts_mxf4.cu:14` |
| `tcgen05.mma ... kind::mxf8f6f4.block_scale.scale_vec::1X` | OK（skeleton，Phase 0 已数值验证） | `dsv41_experts_mxf4.cu:3681,3968` |

**结论**：dense fp8 GEMM 在 sm_103a 上的 tensor-core 入口**只有** `m16n8k32` 这一条
（fp8 的 tcgen05 入口是 block-scaled 的 `mxf8f6f4`，见 §1.4）。

### 1.2 tile 形态三选一的账

m=6 激活 × 5120 列权重 × 输出 n。三种映射：

| 形态 | M tile | N tile | m=6 利用率 | 网格（wkv n=512 / wo_b n=5120） | 判决 |
|---|---|---|---|---|---|
| **A：激活在 M**（任务书的 pad-16 假设） | 16（6→16 pad） | 8 或 16 | **6/16 = 37.5%** | wkv: 512/8 = 64 warps | ❌ 次优 |
| **B：swapAB，权重在 M**（本设计） | 16（满） | 8（6→8 pad） | **6/8 = 75%** | wkv: 512/16 = 32 tiles；wo_b: 320 tiles | ✅ 选定 |
| C：tcgen05 | 128（硬件钉死） | ≥8 | 6/128 = 4.7%（若激活在 M）；权重在 M 时 M 满但**网格 n/128** | wkv: 4 CTA；wo_b: 40 CTA | ❌ 见 §1.4 |

**B 相对 A 的四项结构性优势**（不是微调，是映射本身）：

1. **利用率 75% vs 37.5%**——activation-on-M 必须把 6 行 pad 到 16；
   swapAB 把 6 行放进 N=8 的列，只 pad 2 列。
2. **网格 2× 更多 warp**（n/16 tiles vs n/8 tiles）——latency-bound 的机器上 warp 数是钱。
3. **权重 scale 在 tile 内是常量**：16 个权重行必然落在同一个 32 行 scale 块里
   （`gemm_fp8_swapab_kernel:660` 已记录 "16 rows always sit in one 32-row block"），
   所以每 k 块的权重 scale 是**一次** `ue8m0_to_f`，不需要 per-row 查表；
   激活 scale 则按 C 的**列**（= 激活行）取。
4. **它是树内已有前例的直接推广**：`gemm_fp8_swapab_kernel`（M=1 decode 上 tensor core，
   :606，gate `DSV41_SWAPAB`）+ tcgen05 swapAB gate/up skeleton
   （`dsv41_experts_mxf4.cu:3666`）。本设计的 kernel = 那个 kernel 的 M≤8 泛化，
   **只改两处**：(i) B fragment 从 `[m, kc]` 暂存块的**第 gid 列**读（而非只读第 0 列），
   (ii) epilogue 写全 M 列（而非只写第 0 列）。

### 1.3 为什么 swapAB 在"gemv 5× 摊薄 = 20% 利用率"之上还赢

任务书的对比口径是「pad-16 的 37.5% vs gemv 的 5× 摊薄（20%）⇒ 仍 2× 好于现状」。
**这个口径低估了 tensor core 的杠杆**——MMA 的收益不在 tile 利用率，在**每 MAC 发射的指令数**。
利用率只决定"tensor core 算力有多少被浪费"，而现状的病灶是**根本没喂饱任何管线**：

| 量 | `gemm_fp8_mrows_kernel<M=6>` | `gemm_fp8_mrows_mma_kernel<6>` |
|---|---|---|
| 每 warp 每 32-k 块的指令 | **34**（2+3M LDS + 1 LDG + 1+2M FP，MPAR 文档 §2.3 的表） | **~17**（4 LDS A-frag + 2 LDS B-frag + 1 MMA + ~10 epilogue/scale） |
| 每块退休的 MAC | 6 行 × 32 k = **192** | 16 × 8 × 32 = **4096**（实际用 16×6×32 = 3072） |
| **每 MAC 指令数** | **0.177** | **0.0041（含死列）/ 0.0055（只算活列）** |
| 网格 warp（wkv n=512） | 128 块 × 4 = 512 | (512/16)×32 = **1024**（ks=32 规则） |
| 权重流量 | n·k（1×） | n·k（**1×**，每 warp 走 16 行 × kc，全网格合计 (n/16)·ks·16·kc = n·k） |
| DRAM 延迟覆盖 | 每 warp 自己 stage 一行 → 串行依赖 | cp.async **深环（NStage=8）**——在飞字节覆盖往返延迟 |

**每 MAC 指令数 0.177 → 0.0055 ≈ 32×**（活列口径）。这就是"5× 摊薄"之外的那层杠杆：
即使 pad 掉 25% 的 N 列，也是 0.0055 vs 0.177 —— 差 32 倍，不是 2 倍。

### 1.4 tcgen05 为什么出局（四条硬理由，均为树内事实）

| # | 理由 | 事实出处 |
|---|---|---|
| 1 | **M tile 钉死 ≥ 64/128** | "M=128 (1-CTA), N in [8,256]"（`dsv41_experts_mxf4.cu:17`）；`constexpr int kMTile = 128; // MMA M, pinned by the instruction`（:107、:3864）。m=6 的激活放 M 是 4.7% 利用率；权重放 M 则网格 = n/128：**wkv n=512 → 4 个 CTA**（148 SM 的机器），wo_b → 40 |
| 2 | **TMEM 容量 ⇒ ≤2 CTA/SM** | 每 CTA 分 256/512 列 TMEM（`:3767-3773`），且 `tcgen05.alloc` 是 CTA 级、`relinquish/commit` 是 mbarrier 完成协议 |
| 3 | **block-scale 的 SF 格式不匹配** | `mxf8f6f4` 的 SF 是 **UE8M0**（scale_vec::1X，32 元素粒度）；本工程的激活 scale 是 **f32**（`a_scale[m, k/32]`）。喂进去必须 f32→e8m0 二次量化（把 scale 圆到 2 的幂，最坏 √2 相对误差）——这已经不是 1ULP 级的差，是**换数值语义**。想避开就只能 per-k-block 回读 TMEM 做 epilogue 缩放（160 次/tile），把收益吃回去 |
| 4 | **fp4 解包** | `mxf8f6f4` 消费**解包后**的 fp4（一字节一元素，`dsv41_experts_mxf4.cu:3713-3724`）；本工程投影权重是 **fp8 e4m3 全字节**，不涉及解包，但这条说明 tcgen05 路的 smem 开销结构完全不同 |

**判定**：投影族的形状画像（**小 n、大 K、latency-bound、每 step 数百发**）与 tcgen05 的
设计点（大 M tile、TMEM 累加、CTA 级协议、few-CTA）**正交**。选 `m16n8k32`。

---

## §2 数值等价性 — deliverable ②

### 2.1 差异的正身（逐条，不模糊）

参考程序（`gemm_fp8_mrows_kernel<M>` 的 consume loop，C1-C6）：

```
for kb in 0..nb_k-1:                       # 升序
    sb = ue8m0_to_f(wsr[kb])               # 2 的幂 ⇒ 精确
    j  = kb*32 + lane
    wv = s_lut[s_w[..j]] * sb              # e4m3→f32 精确, ×2^e 精确 ⇒ 精确
    av = s_lut[s_a[..j]] * s_as[..kb]      # a_scale 是**真 f32** ⇒ 1 次舍入
    acc += av * wv                         # 1 次 FFMA 舍入
一棵 shfl_xor 树 (16,8,4,2,1)               # 5 次舍入
```

MMA 程序：

```
D = Σ_{k∈块} a_k · w_k                      # 32 个乘积在 tensor core 内部精确; 求和顺序/内部精度**硬件定义**
acc += D * (a_scale[row,kb] * w_scale[..,kb])   # 2 次舍入
```

**三处实体差异**：

| # | 差异 | 性质 |
|---|---|---|
| D1 | **scale 的分配律位置**：SIMT 是 `(a·as)·(w·sb)` 逐元素乘；MMA 是 `(Σ a·w)·(as·sb)` 作用在**和**上 | 数学等价（as/sb 在 k 块内是常量），**浮点上不等价** |
| D2 | **求和结合律**：SIMT = 每 lane 一条 160 长串行 FFMA + 5 级 lane 树；MMA = 块内 32 项硬件顺序求和 + 160 次 f32 累加 + ks 分区的升序归约 | 不同结合 ⇒ 不同末位 |
| D3 | **内部精度**：e4m3×e4m3 的乘积是精确的（4 位尾数 × 4 位尾数 ≤ 8 位，f32 装得下），但**块内 32 项的求和是否在扩展精度里做、以什么顺序**是 NVIDIA 未承诺的实现细节 | 不可控 |

量级：两者都是"正确到 f32"，相对差 ~几 ULP（1e-7）。**但 near-tie argmax 对 1 ULP 敏感**
（audit §7：line 52/62 的漂移就是模型退化边界随 1ULP 移动）。

### 2.2 逐位不可能 — 结论与证明

> **命题**：对任意 mma 路线，都不存在能让 `gemm_fp8_mrows` 的输出与 SIMT 程序**逐位**相等的
> 实现（在不改参考程序的前提下）。
>
> **证明**：逐位相等要求 (i) 相同的乘积集合、(ii) 相同的结合顺序、(iii) 相同的每步舍入。
> 参考程序的结合顺序是"lane 内 160 长串行链 + 5 级 lane 树"（C1/C3/C6 明确锁定且**禁止**
> 任何重结合，见 kernel header C6 与 `build.sh:40-45` 的 fast-math 警告）。
> `mma.m16n8k32` 的块内 32 项求和顺序由硬件决定且**不可指定**；把 K 拆成 32 项以外的任何
> 粒度都只改变"哪个硬件顺序"，不改变"存在一个硬件顺序"。因此 (ii) 不可能满足。∎
>
> **推论**：任何"用 mma 加速 m=6 投影"的方案都必须放弃与 SIMT 的逐位契约，
> 转而用**别的判据**（EAGER 对照 / 文本指纹 / 双门禁）验收。这不是可以论证掉的东西。

### 2.3 选项 (a)：论证 verify 的投影对 mma 数值差稳健 —— 评估

**支持证据（弱）**：
- DIFF_EAGER 48/48 全 "none"，目标模型 argmax 对隐藏量扰动稳健（audit §10.5）；
- accept 是**双峰**（0 或 5）且**由 token 模式驱动**（数字 vs 换行 token，§10.4），
  不是由数值噪声驱动——17/30 步 k_acc=0 的根因是 draft 对数字序列预测差。

**反证证据（强）**：
- DIFF_EAGER 的举证**有洞**（audit §10.5）：它只比 token 不比隐藏量，且从 commit 后状态出发
  ⇒ "数值一致"的结论已被推翻，嫌疑以隐藏量级复活；
- **WOB 教训**：`DSV41_VERIFY_WOB_MROWS_F32`（非逐位 mrows）⇒ mean-k 1.34 → 0.75-0.92
  （audit §1 表 + §10.5 引为先例）；
- 但——audit §7 的二分链**第 2 刀把 B6（WOB_MROWS_F32）排除了**："B1 − B6 ⇒ line 52,
  mean-k 0.64-0.88"——即**去掉非逐位的 WOB 并没有把边界挪回 62**。

**WOB 教训的正确读法（本文档的裁决）**：把这条件当"非逐位 ⇒ accept 崩"的**铁律**是
**过度归因**（§7 已自证）；但把它当"数值无关"也是错的（§7 同时证明**某些 1ULP 改变确实
移动退化边界**）。正确的读法是：

> **非逐位改动**是**充分危险**（可能移动 near-tie 边界），而非**必然致命**；
> 它**无法用静态论证排除**，只能由双门禁（step_ms + mean-k）实测排除。

⇒ **选项 (a) 不是"论证题"，是"实验题"**。它的成本 = 一次真实的 A/B 臂（一臂一进程、
计数 200 tok、读 `[dspark] steps=50`），不是几页论证。

### 2.4 选项 (b)：mma 只用于"非判定路径" —— 为什么这个说法不成立，以及真正成立的那条

**任务书原文**：「mma 只用于 verify 的非判定路径（如 wo_a/投影的中间量）而判定路径（head/logits）保持逐位」。

**判决：不存在这样的路径。** 判定（accept）消费的是**最终 argmax / logits**；
而 wq_a → wkv → wq_b → （attention）→ wo_b → MoE → head，**每一个投影的输出都在判定路径上**
（它们共同决定 hidden state，hidden state 决定 logits）。wo_a 的"中间量"也一样：
它是 attention 的输出投影，直接进 hidden。所以按字面执行 (b) 只是"少改了几个投影"，
**不改变风险类型**——只是把风险按代理量占比缩小。

**但 (b) 有一个正确的重述，它是本设计真正推荐的那条**：

> **(b′) program-consistent parity**：让 mma 程序成为投影族的**唯一**程序——
> m=1（eager/draft）与 m=2..8（verify）**都走同一个 `gemm_fp8_mrows_mma_kernel<M>`**。
> 于是"verify row r ≡ eager row r"**逐位成立**，因为 **D 的第 r 列只依赖 B 的第 r 列与 A**，
> 与哪些别的激活行同时在场**无关**（§2.4.1 给出论证）。

#### 2.4.1 为什么 (b′) 能逐位成立（这是本设计最值钱的一条）

`mma.m16n8k32` 的语义：`D[16][8] = A[16][32] · B[32][8]`。**D 的第 r 列 = A · B[:, r]**。
把 B 的列 r 绑定到**激活行 r**（swapAB 的 B 列 = gid = 激活行，见骨架
`dsv41_proj_mma_skel.cu` 的 B fragment），则：

- `D[·][r]` 的**全部输入**是 `A`（权重 tile）与 `B[:, r]`（激活行 r 的 k 片）；
- 其它激活行（r' ≠ r）**只出现在别的列**，其数据不进 `D[·][r]` 的任何乘加；
- 每 k 块的 epilogue scale 也按列取：`sc_r = a_scale[r][kb] * w_scale[·][kb]`；
- 累加次序：`acc += D * sc` 在 `kb` 升序上串成链，与 m 无关。

⇒ 对任意 `M ∈ [1,8]`、任意 `ks`，`row r of the M-row launch == row r of the M=1 launch`
**逐位相等**。**唯一前提：`ks` 必须是 (n, k) 的函数，绝不能是 m 的函数**
（ks>1 的归约按 `kp` 升序求和，`ks` 变了结合律就变了）。这条已硬写进
`skeleton:proj_mma_ks_for` 的注释与签名（只吃 `n, k, sms`）。

**这条契约为什么重要**：verify 的**全部**摊薄收益都建立在
「m 行共享同一份权重读」上，而它的正确性前提一直是 kernel header 的 C1-C6
「row r of an m-row launch == the m=1 decode of row r」。MPAR 花了整个设计去保住它
（逐位重排）。mma 路线**用 B 的列布局免费拿到它**——只是参考对象从 SIMT 程序换成了
mma 程序。这是"换程序"而非"破契约"。

#### 2.4.2 (b′) 的树内先例与代价

| 项 | 事实 |
|---|---|
| 先例 | `DSV41_SWAPAB`（`chain_dev.rs:5530-5539`）已经**把 M=1 decode 的投影族换成非逐位的 tensor-core 程序**，其 kernel header 明写 "BIT-EXACTNESS: NOT bit-identical ... Parity is judged by text/fingerprint"（`dsv41_kernels.cu:381-385`）。**换程序 + 文本指纹验收**的纪律在树内已建立 |
| 验收载体 | `crates/ferrite-dsv41/tests/swapab_parity.rs`（交换 A/B 的 parity harness）——(b′) 只是把它的 `m=1` 参数扩到 `m ≤ 8` |
| 代价 | (b′) 要同时动 **eager/draft 的 m=1 投影**（现在走 SIMT `gemm_fp8_gemv_kernel`）。这把改动面从"verify 一条臂"扩到"整栈数值基线"，**风险面变大**，但**风险类型变对**：不再有"verify 与 eager 不同源"的隐性分叉，只剩"整栈换程序 ⇒ near-tie 边界移动"这一个显性问题 |
| 备选（更保守） | 只让 m≥2 走 mma（不动 eager）：**风险类型更差**——verify 与 eager 变成两个不同程序，正是 WOB 的形状 |

### 2.5 §2 判决与验收门

| 方案 | 逐位与 SIMT | verify ≡ eager | 风险 | 建议 |
|---|---|---|---|---|
| (a) 只改 m≥2 + 实测 | ✗ | ✗（跨程序） | 高（WOB 形状） | 仅作为**大范围 sweep 的第一步**，不作首选 |
| (b) 非判定路径 | 不存在这样的路径 | — | — | ❌ 按字面不成立 |
| **(b′) 全 m 走 mma** | ✗ | **✅ 逐位** | 中（整栈数值基线移动，可用双门禁度量） | ✅ **推荐；默认 OFF 落地，双门禁判活** |

**硬性验收门（不可协商）**：

1. `DSV41_PROJ_MMA` 默认 **unset/OFF** ⇒ `dsv41_gemm_fp8_mrows` 走原路径，逐字节不变；
2. 任何 PROJ_MMA 臂必须**同时**报告 `step_ms`（[dspark] 分解）**AND** `mean-k`
   （A0 基线 1.34）——**掉了即弃该臂**，不解释；
3. **禁止**拿性能数在数值过关之前汇报（MPAR 的教训：先看符号，再看幅度）；
4. `ks` 一旦作为 (n,k) 的函数固定，**同一臂内不得按 m 改 ks**（§2.4.1 的唯一前提）。

---

## §3 实施框架 — deliverable ③

> 骨架已落地并**远端 compile-only 通过**：`kernels/cuda/dsv41_proj_mma_skel.cu`
> （独立 TU、不在 `build.sh` 里、默认不参与构建；正式实施时并入 `dsv41_kernels.cu`）。

### 3.1 kernel 签名与几何

```cuda
template <int M>            // M = 激活行数，1..8（= mma 的 N tile 8）
__global__ void __launch_bounds__(32)
gemm_fp8_mrows_mma_kernel(const uint8_t* __restrict__ a,      // [m, k]     e4m3
                          const float*   __restrict__ a_scale, // [m, k/32]  f32
                          const uint8_t* __restrict__ w,       // [n, k]     e4m3
                          const uint8_t* __restrict__ w_scale, // [n/32, k/32] ue8m0
                          const float*   __restrict__ bias,    // [n] or null
                          float*         __restrict__ out,     // [m, out_stride]
                          int n, int k, int out_stride,
                          int ks,                              // K 分区数（n,k 的函数）
                          float*   __restrict__ partial,       // [ks][M][n]（仅 ks>1）
                          unsigned* __restrict__ ctr);         // [n/16]（仅 ks>1）
```

| 几何量 | 值 / 规则 |
|---|---|
| grid | `(n/16) * ks`，**一个 warp 一个 (16 行权重 tile, K 分区)** |
| block | 32（=1 warp；沿用 `gemm_fp8_swapab_kernel` 的 `kSwapabWarps=1`，让"每 SM 驻留块数"由 smem 而不是块内 warp 数决定） |
| M tile | 16（权重行 = 输出通道） |
| N tile | 8（激活行；6 活 + 2 死） |
| K step / 环深 | 128 / 8（沿用 `DSV41_SWAPAB_KSTEP/NSTAGE` 的实测组合；`main`:128/8 是实测最优档） |
| 行 pad | 权重环 `KStep+16`、激活暂存 `kc+16` —**bank-conflict 旋钮**，不是对齐旋钮（`kSwapabRow` 的注释） |
| smem | `[M][kc+16]` 激活 fp8 + `[M][nb]` 激活 scale(f32) + `[nb]` 权重 scale(u8) + `[NStage][16][KStep+16]` 权重环。**没有 LUT**（tensor core 直接解 fp8）——比 SIMT 程序省掉每块 256 项 LUT build |
| epilogue | C fragment：`d0=C[gid][2tg]`、`d1=C[gid][2tg+1]`、`d2=C[gid+8][2tg]`、`d3=C[gid+8][2tg+1]`；**C 行 = 输出通道，C 列 = 激活行** ⇒ `out[ar*out_stride + ch]`，4 次 store/线程（列 ≥M 丢弃） |

### 3.2 逐 kb 指令账（每 warp 每 32-k 块）

| 类别 | SPEC（`gemm_fp8_mrows_kernel<6>`） | MMA（本设计） |
|---|---:|---:|
| LDS（权重 A-frag） | 2 + 3M = 20 | **4** |
| LDS（激活 B-frag） | （含在上面） | **2** |
| LDS（scale） | — | **3**（`s_ws` ×1 + `s_as` ×2） |
| LDG | 1 | 0（环里已在飞） |
| FP（FMUL/FMA） | 1 + 2M = 13 | **~10**（scale 乘 + acc 累加 + `ue8m0_to_f`） |
| MMA | — | **1** |
| **合计** | **34** | **~17** |
| **退休 MAC** | 6×32 = 192 | 16×8×32 = 4096 |
| **指令/MAC** | **0.177** | **0.0041** |

**带宽账（wkv，n=512、k=5120、ks=32）**：

| 量 | 值 |
|---|---|
| 权重流量 | 网格 `(512/16)×32 = 1024` warp × 16 行 × kc(160) B = **2.62 MB = n·k = 1×**（无放大） |
| 激活流量 | 1024 × M(6) × kc(160) = **0.98 MB**（= (n/16)·M·k，L1/L2 复用；相对权重 +38%） |
| DRAM 底线（每 call） | 2.62 MB / 8 TB/s = **0.33 µs** |
| 现状实测（同族） | `gemm_fp8_mrows<5>` 52.1 µs ⇒ **离 DRAM 底线 160×** ⇒ latency-bound 实锤 |
| 目标（对齐 swapAB 实测 1.3-1.5 TB/s） | 2.62 MB / 1.4 TB/s ≈ **1.9 µs** |

### 3.3 gate / ks 规则 / 切换关系

**gate**：`DSV41_PROJ_MMA`（默认 OFF；严格 `== "1"`，与 `DSV41_TAP_PARITY`/`DSV41_COMP_PARITY` 同规），
子旋钮 `DSV41_PROJ_MMA_KS`（显式 ks，sweep 用）；另需 caller 侧 `pmma_n`（scratch 上界承诺，
镜像 `swapab_n` 的纪律）。

**ks 规则**（填满"smem 受限驻留"，而不是"每 SM 一块"）：

```c
ks = 最大 2 的幂 ≤ ceil(SM数 * 8 / (n/16))  且  (k/32) % ks == 0
```

| 投影 | n × k | tiles=n/16 | ks | warps | kc | nb |
|---|---|---:|---:|---:|---:|---:|
| `wq_a` | 1280 × 5120 | 80 | 8 | 640 | 640 | 20 |
| **`wkv`** | **512 × 5120** | **32** | **32** | **1024** | **160** | **5** |
| `wq_b` | 4096 × 1280 | 256 | 4 | 1024 | 320 | 10 |
| `wo_b` | 5120 × 1024 | 320 | 2 | 640 | 512 | 16 |
| `sh w1/w3` | 288 × 5120 | 18 | 32 | 576 | 160 | 5 |

> ⚠️ **为什么要 32 而不是随 `kSwapabKSplit=8`**：出厂的 swapAB launcher 在小 n 上
> **decline**（`if (n < 1664) return 2`，:7448），而 n≤576 恰是 mrows 最差的两档。
> 那个阈值是"ks=8 下的固定开销观察"，**不是物理律**——同一文件里 ks=10 (5.72µs) 打得过
> ks=8 (7.31µs)，方向就是"更多 warp"。ks 规则是这条路在小 n 上成不成的主旋钮。

**切换关系**（`dsv41_gemm_fp8_mrows` 内的选路，三分支 + 一个互斥）：

```
if (PROJ_MMA armed && 形状可发[k%32==0, n%16==0, out_stride>=n, 16B 对齐, n<=pmma_n, m<=8])
        -> gemm_fp8_mrows_mma_kernel<m>          # ① 非逐位臂（numerics-changing）
else if (MPAR armed && fold_r == m)
        -> gemm_fp8_mrows_mp_kernel<m>           # ② 逐位臂（M 作 warp 轴）
else    -> gemm_fp8_mrows_kernel<m>              # ③ 逐位臂（默认，M-in-register）
```

- **① 与 ② 互斥**（两个 gate 同时 armed 时 ① 赢并打一次性 receipt，② 的 receipt 不出现）：
  它们改变的是**同一条乘积链**，混在一臂里 A/B 什么都测不出来。
- **① 与 `fold_r > 1` 无关**：mma 路自己带 K 分区（ks），grid 里**没有** M 轴。
  （MPAR 需要 `fold_r == m` 是因为它把 M 放进了 warp 布局；mma 把 M 放进了 B 的列。）
- **① 与 `DSV41_MROWS_ACT_CPASYNC`(1b) / `DSV41_GEMV_A32` 无关**：那两条是 SIMT 程序的
  `s_a` staging 与 materialise 旋钮，mma 程序不需要 LUT、不 materialise 激活 operand。
  → 落地时**不重读**这两个 gate（骨架里确实没读）。
- **`DSV41_NO_GEMV_FP8`** 存在时 ① 也要 decline（与 legacy 臂同一理由：那会让 m=1 参考
  变成另一种程序，破坏 (b′) 的同源前提）。

### 3.4 scratch（Rust 侧）与活性回执

| 项 | 现状 | mma 路需要 | 说明 |
|---|---|---|---|
| `swapab_part` | `[kSplit=8][n]` f32（`chain_dev.rs:401-411`） | `[ks][M][n]` | 因为**一个 warp 一次产出全部 M 行**，slot 必须带激活行维。ks=32、M=8、n=5120 ⇒ 32×8×5120×4 = 5.2 MB（可接受，但**必须显式改 sizing**，否则越界） |
| `swapab_ctr` | `[n/16]` u32 | 同（`[n/16]`，一 tile 一票） | 复用 |

- **实现顺序建议**：第一轮先跑 **ks=1**（`DSV41_PROJ_MMA_KS=1`）——**不需要任何 scratch**，
  零 Rust 改动、零越界风险，就能拿到 `n=5120/4096` 两档（wo_b/wq_b，320/256 块）的
  符号与数值结论；scratch widening 作为第二刀。
- **活性回执**（防幻影 gate，本树反复踩过）：
  ```
  [proj-mma] ARMED m=.. n=.. k=.. ks=.. -> grid=.. warps, block=32, smem=..
  ```
  没打印 = 没走 mma（更早的 decline：gate/shape/对齐/scratch/`NO_GEMV_FP8`）。

### 3.5 验收清单（给有 GPU 的机器；本任务**不执行**）

| # | 项 | 判据 |
|---|---|---|
| 0 | 远端 compile-only | ✅ **已通过**：`nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 -Xptxas -v -c` **EXIT=0**，8 个 M 特化全部 **0 spills**（64-72 regs） |
| 1 | micro bench（`tests_dsv41_gemm_mrows.cu` 形态） | `t(m=6, mma) / t(m=1, mma) ≈ 1.0-1.2`（**现状 2-4×**）；绝对值对标 swapAB 的 1.3-1.5 TB/s |
| 2 | **内部契约**（逐位） | `m=1..8 每个 m 的 row r` vs `m=1 launch 的 row r` **逐位相等**（按 §2.4.1；`ks` 固定）。**这是唯一可做的 byte-compare**，必须先过 |
| 3 | 与 SIMT 的关系 | **只测容差 + 文本指纹**（`swapab_parity.rs` 的判据），**不要**尝试 byte-compare（§2.2） |
| 4 | 双门禁 | 一臂一进程，计数 200 tok：`[dspark] steps=50` 的 step_ms **AND** mean-k；`DSV41_PROJ_MMA=1` vs 基线，**同时**开 `DSV41_SWAPAB`（(b′)：m=1 也要走 tensor core，否则退回 (a) 的 WOB 形状） |
| 5 | NCU 判读（micro bench only） | `sm__throughput` 应从 ~13% 抬起、`smsp__warp_issue_stalled_short_scoreboard` 应降；`l1tex__data_pipe_lsu_wavefronts_mem_shared` 应显著降（LUT 消失） |
| 6 | 回滚 | `unset DSV41_PROJ_MMA`（逐字节回 legacy） |

---

## §4 收益估算 — deliverable ④

**估算基准（全部来自已记录实测，不新造数）**：verify（生产栈）= 24.49-24.55ms（best2）；
投影族在 nsys v6 表里的占比 = `gemm_fp8_gemv` 15.4%(#1) + `gemm_fp8_mrows<5>` 15.2%(#2)；
任务书口径取**投影族 ~21%**；`gemm_fp8_mrows<5>` 52.1µs = 单行 gemv 的 5×。

### 4.1 三层效应（自下而上，逐项带折扣）

> 口径说明：nsys v6 表的百分比是 **AR_SAFE 总时长**（34.32ms）的分母，且 §10.1 已判
> 该轮 env 断裂（SH/INDEXER/COMPRESSOR 全关）。所以本文**不直接搬百分比**，而是把
> 「投影族占 verify 的份额」当成一个区间 **[15%, 21%]**（下限 = 只算 verify 侧的 mrows；
> 上限 = 任务书的投影族口径），在 24.5ms 的 verify 上换算。

| 效应 | 机制 | 原始（份额 15.2% ⇒ 3.72ms） | 折扣 | 计入 |
|---|---|---:|---:|---:|
| **E1：M 折叠变免费** | m 行 launch = m=1 成本（§2.4.1）⇒ 该 share ÷6 | −3.10ms | ×0.80（小 n 两档的 ks 风险） | **−2.5** |
| **E2：单行成本也坍缩** | 单行从"160×(20 LDS + 13 FP)"降到 mma 的"160×~17 + 深环" | −0.40ms | ×0.75 | **−0.3** |
| **E3：(b′) 下 m=1/draft 同源** | gemv 侧也走 mma：swapAB 对 SIMT 实测 1.76-1.94×（n≥1664）、0.73-0.97×（n≤576，**ks 未调**）；份额按 15.4% ⇒ 3.77ms | −1.50ms | ×0.45（swapAB 在小 n 实测输过） | **−0.7** |
| | | | **合计** | **−3.5ms** |

**verify 24.5 → 21.0ms**。E1/E2/E3 覆盖**不相交**的 kernel 实例（verify 侧 mrows / 同一 share
的单行部分 / draft-side gemv），不重复计。

### 4.2 区间与风险

| 情形 | 收益 | 触发条件 |
|---|---|---|
| 天花板 | **−4.6ms** | 投影族压到自身 DRAM 底线：846MB/step ÷ 1.4TB/s ≈ 0.60ms ⇒ 5.1 → 0.6ms（见下"单点重算"） |
| 乐观 | **−4.2ms** | E1/E2 ×0.95 + E3 ×0.85（小 n 两档被 ks 规则救活 + swapAB 阈值修好） |
| **中央** | **−3.5ms** | §4.1 表 |
| 保守 | **−2.2ms** | E3 归零（小 n 仍输）+ E1/E2 ×0.7——但 **E1 的 M 折叠收益与 n 无关**，兜住下限 |

**单点重算（846MB 的口径）**：每层投影权重 ≈ wq_a 6.55 + wkv 2.62 + wq_b 5.24 +
wo_b 5.24 + sh 1.47 = **21.1 MB**；40 层 ⇒ **846 MB/step**。现状投影族 5.1ms ⇒ 有效带宽
**166 GB/s（2% of 8TB/s）**。MMA 路的目标带宽 1.4 TB/s ⇒ 0.60ms ⇒ **−4.5ms 量级天花板**
（这也是 §4.2 天花板 −4.6 的出处：5.1 − 0.6 = 4.5，再加 E2 的单行残余）。

**最大不确定性**：**小 n 形状**。`wkv`(512) 与 `sh`(288) 的 tile 数只有 32/18，
**必须**靠 ks=32 把网格顶到 1024/576 warp；如果这条不成立（例如 K 不是 32·ks 的倍数而
ks 掉档，或小 n 的固定开销压过收益），这两档会退回 ~1×，收益落到保守区间。
这也是 §3.3 把 ks 规则单列的原因。

---

## §5 备选路线再评估 — deliverable ⑤

前提：**若 §2 的双门禁判 (b′) 不通过**（mean-k 崩），mma 路线作废，但"mrows 摊薄"仍是
SH 族（共享专家 480 发/步）的前提。下面两条**全程逐位安全**的路因此值得留。

### 5.1 ⑤a —— L2 常驻 + 无 smem 的权重广播（fold_r 的"真身"）

**fold_r 实测 6× 退化的真因**（audit §2 + MPAR §1.3）：M 进 grid 后**每个 M-组重新
cp.async16 stage 同一份权重行到自己的 private smem**——重复的是 **prologue 的 DRAM
往返**，不是权重字节。

**⑤a 的做法**：M 进 grid（`ng = m`），**但权重不再进 smem**——每 lane 直接
`w[row*k + kb*32 + lane]` 的 32 B 合并读（`__ldg`/普通 LDG），LUT decode 在寄存器里做。

| 维度 | fold_r（已败） | ⑤a |
|---|---|---|
| 权重 staging | 每 M-组 private smem slab（prologue ×ng） | **无**（直读 L1/L2） |
| 权重 DRAM 字节 | ×ng（L2 未必接得住） | **1×**（同组 m 个 block 的读落在同一 L2 行；`blockIdx` 排布让同组 block 相邻 ⇒ 大概率同波驻留） |
| L1/L2 请求量 | 1× | **×m**（这是它的代价，也是它的收益来源：**warp 数 ×m**） |
| 指令/MAC | 不变 | **不变**（consume 表达式逐字不动） |
| 数值 | **逐位** | **逐位**（我读的 byte 与 smem slab 里的 byte 同源同值；C2 的"拷贝宽度不可观测"直接适用） |

**与 MPAR 的关系**：MPAR 是"1.59× 指令换 6× 并行度"，**外加** per-block prologue；
⑤a 是"**1× 指令**换 6× 并行度，零 prologue"——它把 MPAR 失败的两个成本项**同时**去掉。
它的新增成本是 ×6 的 L1/L2 请求，而 mrows 的 L1 利用率只有 14%，**余量充足**。

**风险**：网格从 `ceil(n/nwarps)` 变成 `m×` 倍块数（wo_b：640 → 3840 块）——
块调度/波次的固定开销可能重新吃掉收益（MPAR 的 L-A 教训）。**先做 micro bench 的
wave/occupancy 判读，再谈 e2e。**

### 5.2 ⑤b —— cluster DSMEM 权重广播（对 MPAR 死因的正解）

**做法**：`blockIdx` 的 M 组组成一个 **thread-block cluster**（Blackwell 支持
`cluster.map_shared_rank` 的分布式共享内存）：
- cluster 里 **rank 0** 建 256 项 LUT + stage 权重 slab 到**自己的 smem**；
- 其余 rank 通过 DSMEM **读同一片 slab**（`cluster.map_shared_rank(s_w, 0)`）；
- 加一层 cluster barrier 发布。

| 死因（MPAR 实测） | ⑤b 怎么解 |
|---|---|
| "LUT 复制 ×5120 块" | LUT 每 cluster **建一次**（grid/ng 次） |
| prologue 完全暴露（issue→build→wait 顺序错） | slab 每 cluster **stage 一次**；且可照 MPAR 的"issue→commit→建表（cover）→wait→barrier"重排 |
| 1.59× 指令（decode 被 M 个 warp 各做一次） | **仍在**——这是 ⑤b 的天花板。但 1.59× 的代价 vs 6× 的并行度，在 DRAM 0.7% / occupancy 13% 的机器上**本应赢**（MPAR 输在 prologue，不在指令） |

**数值**：**逐位**（同一份 smem byte 被 M 个 warp 读；行独立）。
**代价**：DSMEM 延迟高于本地 smem（跨 SM 的 LDS 走 cluster 网络），且 cluster 尺寸受
`cudaOccupancyMaxActiveClusters` 约束（B300 上 8 块/cluster 是常见上限，m=6 够用）。
**这是唯一一条同时满足"权重只 stage 一次 + M 真并行 + 逐位"的路**——
即 MPAR 文档 §1.4 认领的"第三条路"，MPAR 用 warp 布局去近似，cluster 用硬件原语直接给。

### 5.3 三条路的排序

| 路 | 逐位 | 收益上限 | 风险 | 建议 |
|---|---|---|---|---|
| **§1-§3 mma（swapAB）** | ✗（但 §2.4.1 内部逐位） | **−3.5ms（天花板 −4.7）** | 数值验收（双门禁） | ✅ 主攻，默认 OFF |
| ⑤a L2 无 smem 广播 | **✅** | 未知（MPAR 的 1.59× 被消掉后可能转正） | 块数爆炸的调度开销 | ✅ 并行小实验（零数值风险） |
| ⑤b cluster DSMEM | **✅** | 同 MPAR 上限 × prologue 修复 | DSMEM 延迟 / cluster 占用 | ⚠️ ⑤a 的结论出来后再说 |

> **关键关系**：⑤a/⑤b **不与 mma 竞争**——它们逐位安全，可以作为 mma 的"数值不过关"退路，
> 也可以作为独立臂并行验证。而 mrows 的摊薄（无论哪条路）是 **SH 族 480 发/步折叠**的先决条件。

---

## §6 风险 / 待定

1. **数值（头号）**：(b′) 要求 m=1 也走 mma；若只开 `DSV41_PROJ_MMA` 不开 `DSV41_SWAPAB`，
   就是 (a) 的 WOB 形状。**gate 组合必须成对检查**，并在文档/回执里写明。
2. **ks 的合法性**：`(k/32) % ks == 0` 对所有生产 k（5120/1280/1024）都成立，
   但 `k` 若出现非 32 倍数的形状（launcher 已拒）或 `k/32 < 32`（则 ks 掉档），
   小 n 的两档会退化。**首轮必须打 ks receipt 并核**。
3. **store 的合并度**：epilogue 是 4 次分散 store（每线程落到 2 个激活行 × 2 个通道段）。
   每 store 指令的合并粒度 = 32 B（8 个 gid 线程连续通道）。若 NCU 显示 STG 成为瓶颈，
   备选是走 smem 转置收尾（多一次 smem 往返换 128 B 合并）。
4. **激活暂存 vs 直读**：现在把 `M×kc` 激活 stage 进 smem（M=6 时是权重的 +38% 流量）。
   备选是 B-frag 直读 global/L1（每 lane 4 B，L1 高复用）——列为 follow-up 旋钮。
5. **scratch widening 是 Rust 侧改动**（`chain_dev.rs` 的 `swapab_part` 大小），
   **与 peer 的改动区重叠**——落地前需协调（第一轮 ks=1 可完全绕开）。
6. **tcgen05**：本文只判它"不适合投影族"；e4m3 的 tcgen05 (`mxf8f6f4`) 对
   **MoE grouped gate/up** 仍是活跃路线（大 M tile 恰好匹配 expert 的形状），不要因本文而放弃。

---

## §7 交付物 / 改动清单

| 文件 | 内容 | 状态 |
|---|---|---|
| `docs/agent/tensorcore-proj-design.md` | 本文件（①-⑤） | ✅ 本次交付 |
| `kernels/cuda/dsv41_proj_mma_skel.cu` | 实施框架骨架：`gemm_fp8_mrows_mma_kernel<M>` + `dsv41_gemm_fp8_mrows_mma` + `proj_mma_ks_for` + gate + receipt。**独立 TU、不在 build.sh、默认不构建** | ✅ 本次交付，远端 compile-only **EXIT=0 / 0 spills** |
| `kernels/cuda/dsv41_kernels.cu`（未来） | 骨架并入 + `dsv41_gemm_fp8_mrows` 的三分支选路 + ①/② 互斥 receipt | ⏳ 待实施（本任务未改，避免与 peer 冲突） |
| `crates/ferrite-models/src/dsv41/chain_dev.rs`（未来） | `swapab_part` → `[ks][M][n]`；`DSV41_PROJ_MMA` 的 arm（与 `DSV41_SWAPAB` 成对） | ⏳ 待实施（**peer 改动区，需协调**；第一轮 ks=1 不需要） |
| `kernels/cuda/build.sh`（未来） | 并入 TU 时机：骨架验证后从独立 TU 移入 | ⏳ |

**远端 compile-only 复现命令**（已验证）：

```bash
nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 \
     -Xptxas -v -c kernels/cuda/dsv41_proj_mma_skel.cu -o /tmp/proj_mma.o
# 期望：EXIT=0；8 个 `gemm_fp8_mrows_mma_kernel<1..8>` 全部
# "0 bytes spill stores, 0 bytes spill loads"，64-72 registers
# （实测：M=1/2/8 -> 64-65 regs；M=3..7 -> 72 regs）
```

---

## 附：与 MPAR 的关系，一句话

MPAR 证明了「**M 作 warp 轴**」换不来收益（1.59× 指令 + per-block prologue）；
本设计证明**换轴不是出路、换程序才是**——把 M 放进 mma 的 **B 的列**，
一行 CLA 都不留给它（M 从"要迭代的维度"变成"一次指令退休的维度"），
于是 m 行的成本 = 1 行的成本，且 row r 的自洽性由 B 的列布局**免费保证**。
