# gemm_fp8_mrows 的 M-tile（G2）— 把 M 从「寄存器串行」改成「GEMM 的 M-tile 维」

> 载体：`kernels/cuda/dsv41_kernels.cu` 的 `gemm_fp8_mtile_kernel<M>` + `dsv41_gemm_fp8_mrows` launcher。
> Gate：`DSV41_MROWS_MTILE`（**默认 OFF**，unset/`0` = 逐字节回到 M-in-register 程序）；
> 调参轴：`DSV41_MROWS_MTILE_BN`（warp tile 的 N 宽度，1..4，默认 2）。
> 权威模型：`sglang-verify-model.md`（SGLang verify = 一次 forward，m 个 token 进 GEMM 的 batch 维，实测 1.2-1.3×）。
> 前车之鉴：`mrows-mpar-design.md`（MPAR 两败）+ `verify-amortization-lesion-audit.md` §10.9（⑤a 四档负向）。
> 日期：2026-09-13。**GPU 未验证**——本文件是设计与验证手册，不是实测结论。

---

## §0 一句话

`gemm_fp8_mrows_kernel<M>` 把 M 藏在**单个 warp 的寄存器**里（`float acc[M]`），每个 warp 只认领
**一行输出**，于是 `n` 个 warp 各自把**同一份 M 行激活折叠一遍**——M 的代价被 `n` 个 warp 各付一次。
M-tile 把 M 变成 **warp 内部 (M × bn) 寄存器 tile 的行轴**：一个 warp 同时算 `M` 个激活行 ×
`bn` 个输出行的**一整块**，于是同一份**权重字节服务 M 行**（继承 legacy 的复用）、同一份**激活字节
服务 bn 行**（新增的复用，legacy 没有）。

**为什么这次能成（一句话）**：MPAR / ⑤a 的指令数分别是 legacy 的 **1.59×** 和 **1.00×**（都没降），
M-tile 是 **0.56×**（bn=2）/ **0.47×**（bn=4）——它是三条路里**唯一降低每输出元素指令数**的一条。

---

## §1 病灶核算：legacy 的指令去哪了

生产形状 `wkv`（n=512, k=5120, m=6, nwarps=4）。**每 (warp, kb)** 的指令账
（口径沿用 `mrows-mpar-design.md` §2.3，1 个 LDS/LDG/FMUL/FMA = 1 条 warp 指令）：

| 类别 | legacy `mrows<M>`（每 warp = 1 输出行 × M 激活行） |
|---|---:|
| LDS 权重字节 + LUT decode | 2 |
| LDS 激活字节 + LUT decode + 激活 scale | **3M = 18** |
| LDG 权重 scale | 1 |
| FMUL（权重 decode 1 + 激活 decode M） | 1 + M = 7 |
| FMA（`acc[q] += av*wv`，q=0..M-1） | 6 |
| **每 warp 每 kb 合计** | **34** |
| warp 总数 | `n` = 512 |
| **每 kb 全 grid** | **34n = 17408** |
| **每输出元素** | 34/6 ≈ **5.67** |

**读表**：34 条里有 **18 + 6 = 24 条（71%）是「激活侧」**——而激活行只有 M=6 行、对所有 `n` 个
输出行是**同一份数据**。legacy 的结构决定了：这 24 条被 `n` 个 warp **各付一遍**。这就是
「M-in-register 串行」的实体：**不是 acc 依赖链，是 M 的折叠工作量随输出行数线性复制**。

**与实测对齐**：`m=1` 时每 warp 每 kb 是 `3·1 + 2 + 1 + (1+1) + 1 = 9`，`m=6` 是 34 ⇒
**3.8×** 指令；实测 `52.1µs/5行` 对单行是 **5×**（含饱和/延迟放大）。**指令账与实测同号同量级**
⇒ 病灶是**总指令数**，不是「acc 依赖链」（这同时解释了 MPAR 为什么输：它把总指令抬到 1.59×）。

---

## §2 设计：warp → (m_block, n_block) 二维 tile

### 2.1 几何

```
BLOCK_M = 8        // M tile（≥ VERIFY_ROWS=6，2 的幂，且 m 的 dispatch 上界就是 8）
bn      = warp tile 的 N 宽度（1..4，DSV41_MROWS_MTILE_BN，默认 2）
nw      = 每 block 的 warp 数（coverage 规则，见 §2.4）
nwarp   = ceil(n / (nw * bn))            // grid，每 block 覆盖 nw*bn 个输出行
warp w  → (m_block = 0, n_block = w)     // BM=8 覆盖全部 M ⇒ m 轴恰好 1 块
          拥有输出行 [row0 + w*bn, row0 + (w+1)*bn)，row0 = blockIdx.x * nw * bn
寄存器 tile: acc[RN][bn]，RN = min(M, BLOCK_M)
```

**轴的含义**（这是"SGLang 机制"的 ferrite 落地）：

| 轴 | legacy | MPAR（已败） | **M-tile（本设计）** |
|---|---|---|---|
| M | warp 内 `acc[M]` 串行折叠 | **warp 轴**（每 warp 一对 (行,激活行)） | **tile 的行轴**（每 warp 一整块 M × bn） |
| N | warp 轴（每 warp 一行输出） | 同 legacy | **tile 的列轴**（每 warp `bn` 行输出） |
| K | lane 轴（`j = kb*32+lane`，上升） | 同 | **同**（逐位红线，见 §3） |
| 权重读 | 每 warp 每 kb 1 字节，**服务 M 行** ✓ | 每 warp 每 kb 1 字节，只服务 **1** 行 ✗ | 每 warp 每 kb 1 字节，**服务 M 行** ✓ |
| 激活读 | 每 warp 每 kb **M** 字节，服务 **1** 行 ✗ | 每 warp 每 kb **1** 字节，服务 1 行 | 每 warp 每 kb **M** 字节，服务 **bn** 行 ✓ |
| warp 数 | `n` | `n·M` | `n/bn` |

**一句话**：M-tile = legacy 的「权重服务 M 行」**保住**，再补上 MPAR 没有的「激活服务 bn 行」。

### 2.2 指令账（同 §1 口径，`wkv` M=6）

| 类别 | legacy | MPAR | **M-tile (bn=2)** | **M-tile (bn=4)** |
|---|---:|---:|---:|---:|
| LDS 权重（字节+LUT+scale） | 3 | 3 | 3·bn = 6 | 12 |
| LDG/LDS 激活（字节+LUT/scale） | 3M = 18 | 3 | 3M = 18 | 18 |
| FMUL（两侧 decode） | 1+M = 7 | 3 | M+bn = 8 | 10 |
| FMA | M = 6 | 1 | M·bn = 12 | 24 |
| **每 warp 每 kb** | **34** | **9** | **44** | **64** |
| warp 总数 | `n` | `n·M` | `n/2` | `n/4` |
| **每 kb 全 grid** | **34n** | **54n** | **22n** | **16n** |
| **每输出元素** | **5.67** | 9 | **3.67** | **2.67** |
| **对 legacy 比值** | 1.00 | **1.59（实测两败）** | **0.65** | **0.47** |

（⑤a 的每元素指令与 legacy **相同**——它只把权重 LDS 换成 LDG、去掉 staging，指令数不变；
它输在 `×m` 的 L1/L2 请求量与块数爆炸。）

**结论**：只有 M-tile 的比值 < 1。`bn=2` 每元素 −35%，`bn=4` −53%。
**FMA 是唯一随 M 增长的项且不可压缩**（`n·M·k` 是 GEMM 的算术下界）——M-tile 把
「激活的 `LDS byte + LUT decode + scale`（3 条/元素）」从「服务 **1** 行」抬到「服务 **bn** 行」，
把「权重的 `byte + LUT + scale + decode`（4 条/元素）」摊到 **M** 行，
于是每元素 `(4M/bn + 4 + M)` 替代了 legacy 的 `5M + 4`（M=6：`4M/bn` 项是节省的全部来源）。

**一个实现级的取舍必须点名**（它决定这张表的符号）：激活的**解码**用 `s_lut`（1 条 LDS）
而**不是** 内联 `e4m3_to_f`（⑤a 家族的形式，~7 条 ALU）。M-tile 有 smem（权重 slab），
1 KB 的 LUT 是免费的；若用内联解码，激活侧每元素要 10 条而不是 4 条，**总指令会反超 legacy**
（bn=2 时 40n vs 34n）——即"照抄 ⑤a 的 consume 形式"会把本设计的正号变成负号。


### 2.3 smem 预算

```
s_w   : nw * bn * k     字节   权重 tile，cp.async16 一次 stage，warp 间 + M 行间共享
s_lut : 256 * 4 = 1 KB         e4m3 解码表
```

**激活不进 smem**——这是与 legacy 唯一的结构性减法，也是 §4 里 ⑤a 教训的正确用法：

- legacy stage 了 `M*k` 字节的激活行（wkv 30 KB）+ `nwarps*k` 的权重（20 KB）+ LUT ≈ 51 KB；
- M-tile **只 stage 权重 tile**（`nw*bn*k`），激活**直读 L1**（`__ldg` 的 32 B/warp/kb 合并读）。
  理由：M-tile 的激活复用是 **warp 内的 bn 轴**，不是 block 内的 `nw` 轴；
  若把激活再 stage 成 block 共享，就**按 block 复制了一整套 prologue**
  （`fold_r` / MPAR 的病根：重复的量是 **prologue**，不是字节）。

生产预算（bn=2）：

| 形状 | n × k | nw | smem(M-tile) | smem(legacy) | 变化 |
|---|---|---:|---:|---:|---|
| `wq_a` | 1280 × 5120 | 4 | 4·2·5120 + 1K = **41 KB** | 56 KB | −27% |
| **`wkv`** | **512 × 5120** | **1** | **10 KB + 1K = 11 KB** | **56 KB** | **−80%** |
| `wq_b` | 4096 × 1280 | 8 | 8·2·1280 + 1K = **21 KB** | 20 KB | ≈0 |
| `wo_b` | 5120 × 1024 | 8 | 8·2·1024 + 1K = **17 KB** | 16 KB | ≈0 |

（>48 KB 的形状走 per-M `cudaFuncSetAttribute`，与 legacy/MPAR 同一纪律。）

### 2.4 coverage 规则（**MPAR auto 的教训直接用上**）

MPAR 的第一次败因之一是 `auto` = 「最宽块」，把 `wkv`（n=512）压到 103 块、`sh` 压到 58 块，
**低于 148-SM**，整片空转。M-tile 采用**反过来的规则**：

```
nw = min( dsv41_mrows_warps_for(n),  max(1, n / (bn * SM_COUNT)) )     // grid = ceil(n/(nw*bn)) >= SM
```

即「**仍然覆盖 SM 的最宽块**」：`wkv` ⇒ nw=1（grid 256），`wq_a` ⇒ nw=4（160），
`wq_b`/`wo_b` ⇒ nw=8（256 / 320）。**launcher 必须传 n**（测试 pin 走默认 n=0 时退回 legacy 宽度）。

---

## §3 数值红线（逐位等价论证）

**契约**：`out[q][row]` 与 m=1 解码该行**逐位相同**（沿用 legacy 的 C1-C6）。M-tile 与
`gemm_fp8_mrows_kernel<M>` **是同一个元素程序的重排**：只换了「哪个 warp 算哪个元素」。

| 契约 | legacy | M-tile | 为什么相同 |
|---|---|---|---|
| C1 K 走序 | `kb` 升序，`j = kb*32 + lane` | **同** | 逐字保留 |
| C2a 权重字节 | `s_lut[s_w[warp*k+j]]` | `s_lut[s_w[(n_block*bn+nn)*k+j]]` | `s_w` 的 slab 是 `w[row0*k .. (row0+nrows)*k)` 的**逐字节拷贝**，索引 1:1（`(n_block*bn+nn) = row-row0`） |
| C2b 权重 scale | `wsr[kb]` | `w_scale[(row>>5)*nb_k + kb]` | 同一 `w_scale` 字 |
| C2c 激活 | `s_lut[s_a[q*k+j]] * s_as[q*nb_k+(j>>5)]` | `e4m3_to_f(__ldg(a+q*k+j)) * __ldg(a_scale+q*nb_k+kb)` | `s_lut[b] ≡ e4m3_to_f(b)`（表就是用它建的）；`s_a` 是 `a` 的纯拷贝（宽度不可观测 ⇒ 读源即同值）；`j>>5 == kb`；`s_as` 是 `a_scale` 的纯拷贝 |
| C3 归约树 | 每 (warp, q) 一棵 `shfl_xor` off=16,8,4,2,1 | 每 (warp, q, nn) 一棵，**同 off 序列** | 每元素恰好由 1 个 warp 算、1 棵树 |
| C4/C5 无跨行/跨 K 重组 | `acc[q]` 独立链 | `acc[q][nn]` 独立链 | 从未跨 q 或 nn 相加；无 K-split |
| C6 累加式 | `acc[q] += av*wv`，`#pragma unroll 32` | `acc[q][nn] += av[q]*wv`，`#pragma unroll 32` | 同一表达式 |
| a32 两臂 | a32=1 材料化 `af[q]`，a32=0 内联 | 内联（M-tile 只承载一个形式） | 两臂是同一 FMUL 的同一值（legacy header 的 bit-identical by construction）⇒ **M-tile 对两个 gate 取值都逐位相等，不重读 `DSV41_GEMV_A32`** |

**唯一变化的东西是「哪个 warp 算哪个元素」**（legacy header 原话：*the block geometry does not
enter the parity argument — rows are independent*）。所以每行的 fma 链顺序、归约树**逐行保持**。

**一个必须点名的细节**：`av[q]` 在 M-tile 里**每 (q, kb) 只算一次**、跨 `nn` 复用。legacy 的
a32=1 臂也把 `af[q]` 每 (q, kb) 算一次（材料化到寄存器）后折叠——**同一个乘积、同一个操作数顺序**；
a32=0 臂在内联处算同一个乘积。所以 M-tile 的单一内联形式对**两个 a32 取值**都逐位等价
（`mrows-mpar-design.md` §3 的同一个论证）。

---

## §4 与 MPAR / ⑤a 的失败模式对照（为什么这次避开）

| 失败模式 | MPAR（实测两败） | ⑤a（实测四档负向） | **M-tile 怎么避开** |
|---|---|---|---|
| **① 总指令数上升** | per-element 9 vs legacy 的 9，但 **warp 数 ×M** ⇒ 总指令 **1.59×**（audit §10.9：「1.59× 指令代价 > 收益，结构性天花板确认」） | 指令**不变**（只把 LDS 换成 LDG） | **总指令 0.65×（bn=2）/ 0.47×（bn=4）**——唯一降指令的路（§2.2） |
| **② prologue 复制 grid 次** | LUT build + slab staging 按 grid 复制，且首版顺序错（issue 后立刻 wait），DRAM 往返完全暴露 | 无 prologue（也没 staging） | staging **只发生在权重 tile**（`nw*bn*k`，总字节 = `n*k` 仍 1×）；激活**不 stage** ⇒ 无 per-block 激活 prologue；prologue 顺序**直接照抄 MPAR 的返工版**：ISSUE → BUILD 表(cover) → WAIT → BARRIER |
| **③ 权重 decode 被 M 个 warp 各做一次** | `s_lut[rs[j]]*sb` 每 warp 做一次（M 倍） | 每 warp 做一次（1×） | **每 warp 每 (nn,kb) 一次**，该 `wv` 服务 **M 行** ⇒ decode 摊到 M |
| **④ 无 staging ⇒ L1/L2 请求 ×m** | — | **本设计要避的正靶**：权重直读 L1 让 3200-3840 块的 L2 请求 ×m，四档全负 | **权重 stage 到 smem 再读**（⑤a 的教训：staging 值得）；只有**激活**直读 L1，而激活的读量是 `M`（6）不是 `n`（5120），且每读一字节服务 bn 行 |
| **⑤ 块数爆炸 / SM 覆盖不足** | `auto` 把 wkv 压到 103 块 < 148 SM | rpb=1 时 3200-3840 块，调度开销吃掉收益 | coverage 规则（§2.4）：**grid ≥ SM 数**且取「仍覆盖 SM 的最宽块」，`wkv` 256 块、`wo_b` 320 块 |
| **⑥ 数值破契约** | 逐位 ✓（但没赢） | 逐位 ✓（但没赢） | **逐位 ✓**（§3，同一元素程序的重排） |

**三句话判决**：
- MPAR 的错是「**用更多指令换更多 warp**」；
- ⑤a 的错是「**去掉 staging 换零 prologue**」；
- M-tile 走的是第三条：**用 tile 复用把指令降下来**（权重服务 M 行 + 激活服务 bn 行），
  staging 保留在权重侧，prologue 按 block 只付一次。

**代价（诚实登记）**：warps 从 `n` 降到 `n/bn`（wkv bn=2：512 → 256）。M-tile 的赌注与 MPAR
**相反**——它赌的是「issue/吞吐 bound，指令数决定时间」，MPAR 赌的是「latency bound，warp 数决定时间」，
而 MPAR 的两次实测已经把后者证伪。**若 M-tile 在 GPU 上也负向，则说明「不是 issue-bound 也不是
latency-bound」，那是第三种病（例如 LDS 通道饱和 / 依赖链），必须回报而不是调参。**

---

## §5 实施（deliverable ②）

| 文件 | 改动 |
|---|---|
| `kernels/cuda/dsv41_kernels.cu` | ① 新增 `gemm_fp8_mtile_kernel<M>`；② 新增 gate `g_mrows_mtile` / `g_mrows_mtile_bn` + `dsv41_mrows_mtile_for()` / `dsv41_mrows_mtile_warps_for()`；③ launcher 加 **MTILE 优先级最高**的选路分支 + `[mrows-mtile] ARMED` 活性回执 + per-M smem 属性 |
| `kernels/cuda/tests_dsv41_gemm_mrows.cu` | 新增 MTILE contract pin（resolution + coverage 不变式）+ 头注释（load-time const ⇒ 一进程一值） |
| `docs/agent/mrows-mtile-design.md` | 本文件 |

**优先级**：`MTILE > ⑤a(L2BCAST) > MPAR > legacy`（任务书：M-tile 是设计正解）。
两条以上 armed 时，receipt 里**点名被遮蔽的臂**（本树被「armed but inert」咬过多次）。

**默认 OFF**：`DSV41_MROWS_MTILE` 未设 ⇒ `dsv41_mrows_mtile_for` 返回 0 ⇒ 走 ⑤a/MPAR/legacy 原路径，逐字节不变。

### 5.1 编译验证

- `cargo check --workspace --all-targets` → 见 §5.3。
- 远端 `nvcc` compile-only（`-gencode arch=compute_100a,code=sm_100a -O3 --use_fast_math -std=c++17`,
  远端 CUDA 13.2，`ubuntu@43.202.208.136`，私有目录，**无 GPU**）+ `-Xptxas -v` 看 regs/spill。

> ⚠️ `chain_dev.rs` / `device.rs` / `dspark_dev.rs` **不在本任务范围**（peer 改动区）。
> 本任务只写 `kernels/cuda/dsv41_kernels.cu` + `tests_dsv41_gemm_mrows.cu` + 本文件。

### 5.2 活性回执

首次 ARMED launch 打印一行：
```
[mrows-mtile] ARMED m=.. n=.. k=.. bn=.. nw=.. -> block=.. warps, grid=.., smem=.. [遮蔽: ...]
```
gate 未设时**不打印**。armed 而日志无此行 ⇒ launcher 更早就 decline 了
（mode<3 / NO_GEMV_FP8 / 形状拒绝 / `fold_r != m` / smem 超 ceiling）。

### 5.3 编译结果（本机 + 远端，compile-only，无 GPU）

**远端 nvcc（CUDA 13.2, `ubuntu@43.202.208.136`, `/tmp/mtile_smith/`, `sm_100a`, `-Xptxas -v`）**：

```
nvcc -gencode arch=compute_100a,code=sm_100a -O3 --use_fast_math -std=c++17 \
     -Xptxas -v -c dsv41_kernels.cu -o dsv41_kernels.o        # EXIT=0
nvcc -gencode arch=compute_100a,code=sm_100a -O3 --use_fast_math -std=c++17 \
     -o t_gemm_mrows tests_dsv41_gemm_mrows.cu                # EXIT=0
```

`gemm_fp8_mtile_kernel<M>` 的 ptxas 账（**全部 0 spill / 0 stack**）：

| M | regs | spill | barriers |
|---:|---:|---|---:|
| 1 | 40 | 0 | 1 |
| 2 | 53 | 0 | 1 |
| 3 | 63 | 0 | 1 |
| 4 | 73 | 0 | 1 |
| 5 | 80 | 0 | 1 |
| **6** | **96** | **0** | 1 |
| 7 | 92 | 0 | 1 |
| 8 | 96 | 0 | 1 |

**读表**：96 regs @ M=6（= `RN=6 × BN_MAX=4 = 24` 个 acc + `av[6]` + 地址/索引 + LUT/slab 索引）。
`__launch_bounds__(256)` 下 96 regs 允许 **2 块/SM**（寄存器侧）；smem 侧 wkv 是 11 KB（20 块/SM）、
wq_a 41 KB（5 块/SM）——**两个 resource 都不构成瓶颈**（唯一的瓶颈是 grid 的 SM 覆盖，见 §2.4）。
`0 spill` 说明寄存器 tile 没有把编译器逼到本地内存，这是 M-tile 可行的前提。
（LUT 版比内联解码版高 1 reg：多一个 `s_lut` 基址，但省掉每元素 ~6 条 ALU——**这笔换算是本设计的正号来源**，见 §2.2。）

**唯一的既存 warning**（非本次改动，前序 commit 已有）：`k1max` / `nwarp` / `nt` 未引用。

> 本机 `cargo check --workspace --all-targets`：**EXIT=0**（Rust 侧不编译 `.cu`——运行时 dlopen
> `libferrite_kernels.so`，其校验走远端 nvcc；`.cu` 不在 cargo 编译图内）。

---

## §6 GPU 验证手册 — deliverable ④

> 前置：本任务**禁止 GPU**。以下由主 agent 在有空闲 GPU 的机器执行。
> 纪律（audit §5.3）：**只跑 micro bench**（`tests_dsv41_gemm_mrows.cu` 二进制），**不跑 e2e NCU**。

### 6.1 构建

```bash
nvcc -gencode arch=compute_100a,code=sm_100a -O3 --use_fast_math -std=c++17 \
     -o /tmp/mtile_$USER/t_gemm_mrows kernels/cuda/tests_dsv41_gemm_mrows.cu
```

### 6.2 逐位等价验收（**先于性能**）

`g_mrows_mtile` 是**文件级 static（load 时读 getenv）**，同进程无法 sweep ⇒ **一进程一值**：

```bash
for bn in 1 2 3 4; do
  echo "== DSV41_MROWS_MTILE=1 DSV41_MROWS_MTILE_BN=$bn =="
  DSV41_MROWS_MTILE=1 DSV41_MROWS_MTILE_BN=$bn /tmp/mtile_$USER/t_gemm_mrows --quick
done
DSV41_MROWS_MTILE=1 /tmp/mtile_$USER/t_gemm_mrows        # 全量（含 dispatch/m、n%nwarps、tiny）
```

**验收判据**：每个臂 `RESULT: all checks passed`（m 行 launch 与 m 个 m=1 launch **逐位**相等，
sentinel 覆盖无空洞）。任一位差 ⇒ **停，回报**（不要拿性能数）。
每个臂应打印 `[mrows-mtile] ARMED ...` 一行——**没打印说明没走 M-tile**，那次运行不算数。

### 6.3 性能验收（micro bench，同一形状，m=1 为基准）

```bash
                      /tmp/mtile_$USER/t_gemm_mrows --quick                        # legacy
DSV41_MROWS_MTILE=1   /tmp/mtile_$USER/t_gemm_mrows --quick                        # M-tile bn=2
DSV41_MROWS_MTILE=1 DSV41_MROWS_MTILE_BN=4 /tmp/mtile_$USER/t_gemm_mrows --quick   # M-tile bn=4
```

- **目标**：`t(m=6, M-tile) / t(m=1) ≈ 1.0~1.3`（legacy 是 2~4×；MPAR 实测更差）。
- **bn sweep**：`{1,2,3,4}`。`bn=1` 退化成「legacy 几何 + 无激活 staging」——**它是本设计的对照组**
  （预期 ≈ legacy；若 bn=1 反而更快，说明激活 staging 才是 legacy 的成本）。
- **判据优先级**：先看**符号**（是否 < legacy 的 1.0×），再看幅度。**若 M-tile 更慢** ⇒ 上报，
  不要硬调 bn 掩盖（§4 末段的第三种病假设）。

### 6.4 e2e 双门禁（主 agent，授权后）

```
DSV41_MROWS_MTILE=1 DSV41_MROWS_MTILE_BN={2,4}   （一臂一进程，计数 200 tok，读 [dspark] steps=50）
```
**双门禁**（每个臂必须同时报告）：
- `step_ms`（`[dspark]` 分解）——票面 **verify −3~8 ms**；
- `mean-k`（A0 基线 1.34 / S1 修复后 **2.240**）——**掉了 = 数值回归，立即弃用该 gate**。

### 6.5 nsys 判据

```
nsys profile --stats=true ... → 看 gemm_fp8_mtile_kernel 的 kernel 时间
```
**判据**：`t(mrows kernel, m=6) / t(m=1) ≈ 1× 单行`（legacy 是 ~5×）。
且 kernel 表里应**只有** `gemm_fp8_mtile_kernel`（不应同时出现 legacy/MPAR/⑤a 的名字）。

### 6.6 回滚

```bash
unset DSV41_MROWS_MTILE     # 立即回到 M-in-register 程序（逐字节）
```

---

## §7 待定 / 风险

1. **符号未定（头号）**：M-tile 赌 issue-bound（指令数决定时间）。MPAR 的两次实测证伪了
   latency-bound，但没有直接证明 issue-bound ⇒ 本臂是**定符号实验**。负向即回报。
2. **bn 的 SM 覆盖**：coverage 规则保证 `grid ≥ SM`，但 `bn=4` 时 `wkv` 的 grid = 128 < 148
   （nw 被 clamp 到 1 后 `ceil(512/4) = 128`）⇒ **bn=4 在小 n 上会掉覆盖率**，首轮 sweep 必须带上。
3. **激活直读 L1 的请求量**：每 warp 每 kb 读 M 字节 + M 个 scale = 2M 条 LDG。bn=2 时 warp 数为
   `n/2`，总激活读 = legacy 的 1/2 —— **比 legacy 少**，不构成 ⑤a 的请求爆炸。但 `a_scale` 的
   per-(q,kb) 标量 LDG 是 uncoalesced 的（32 lane 同址 ⇒ 1 次广播），需 NCU 确认不是瓶颈。
4. **`BLOCK_M=8` vs `M=6` 的 2 行 pad**：本设计用 `RN = min(M, BM) = M` 做循环上界 ⇒ **零浪费**
   （不是「算 8 行丢 2 行」）。BM=8 只是 tile 的**声明上界**与对齐条款。
5. **wo_a_grouped / head 的 v1_mrows**：同族，后续（任务书 §5）。
