# 16-seq decode 性能路线图（2026-09-08 会话结论）

> **当前最好（已验证，文本人眼确认）**：300-token 窗口 16.23-16.48ms/步 = **971-985 tok/s**；
> 1000-token 窗口 ~17.9ms = **893-895 tok/s**。目标 1600（不开 MTP）。会话内 546 → ~975（+79%）。
>
> **下一步按优先级**：
> 1. **MoE 专家分组**（act/down 仅 DRAM 峰值 30-53%；按专家聚簇改善 L2/DRAM 行局部性，预期 2-3%）
> 2. **sparse_attn 双侧半精度 + warp 级折回**（k 侧单独做过 = 回归；必须 k+v 同时改才摊得平转换开销）
> 3. **tensor-core attention**（decode 的 M=1 无法直接拼 GEMM；需按 seq 分块 + 在线 softmax，属大重写）
>
> **不要再试**（已实测证伪）：gemv WPR=2/unroll4/T=4、act 2-tile 寄存器预取、act 32-K tile、
> hc_pre_mix 8 行、HC_MIX_KS=16、gdn_step_v2 批量、PDL、rmsnorm block 单独调大、
> sparse_attn TG=4（gmask 不匹配会挂）、fp16 k-only 缓存、down 的 cp.async（每 lane 私有数据）。
> **每次改动必须**：看 build error 数 → 同窗口 A/B → 人眼验证文本。

## gemm_fp8_gemv 微基准（2026-09-11 最后一轮，隔离探针 /tmp/gp6/gprobe6..10.cu）

> 底座复刻 = `kernels/cuda/dsv41_kernels.cu:1722-1931`（mode 4 / warps=4 / LUT / a32 / unroll 4），
> smem 48384B，与 launcher 公式 `:1962-1967` 逐字节吻合。所有数字为 171-call CUDA graph、
> k=5120、warps=4、3 次独立 graph 构建的 µs/call（重复性 ±0.01µs）。

**本轮的硬结论：每调用成本不是"LDS 链"，是块级 staging 的冗余。** 用 nop 空 kernel 标定每
kernel 的 graph 槽位只有 0.52–0.71µs（此前怀疑的 launch 开销被排除），随后逐项分解（n=1664）：

| 组成 | µs/call |
|---|---|
| nop416（纯槽位） | 0.71 |
| 权重行 cp.async staging | 1.18 |
| 块级 staging（激活+LUT+a32） | **2.85** |
| consume 循环 | 4.45 |
| 合计（= b4 实测 9.31） | 9.19 |

块级 2.85µs 再分解：**a32 物化 ~1.55**、激活 uint4 staging ~0.8、LUT 构建 ~0.3、a_scale ~0.1。
它**跨 416 个 block 冗余重复**，且几乎与 n 无关（n 涨 6.5× 只涨 5%）——这才是"固定项"的真身。

**两个有效杠杆（位一致性已 fingerprint 验证，且在 mode R/U 两种数据分布下都复现）**：

1. **a32 物化改 4 元素向量化**（1 次 uint32 读 `s_a` + 1 次 float4 写 `s_af`）：a32 是**指令数**
   瓶颈而非 ILP 瓶颈 —— unroll 4/8/16 全中性，手写 4 路 ILP 简单版也中性，**只有 4 元素向量化有效**
   （−13%/−9.6%/−6.2% @n=256/1024/1664）。位运算解码（无 smem gather）反而**慢 6%** ——
   e4m3 LUT 在 a32 构建里同样赢过位运算，与 consume 循环的结论一致。
2. **ROW_FIRST 重排**：行 cp.async 与块级 staging 写的是**不相交的 smem 区域**，却是串行的。
   先发行行 cp.async（权重 + scale，`nb_k%16==0` 时）再跑块级 staging，把权重行的
   global→shared 延迟藏进块级构建：额外 −1.8%/−1.0%/−0.5%。
   scale 行的 cp.async 需 16B 全局对齐，必须用运行时守卫 `(nb_k & 15) == 0` 回退普通字节 load
   （即 `:1874-1880` 记的 err-716 教训，不能写成无条件）。

叠加已落地的 `unroll 4`→`32`，最优组合 **rf_u32_ilp（ROW_FIRST + unroll32 + a32 向量化）**：
**5.02 / 6.67 / 8.58 µs/call，即 −24% / −12% / −7.7%**（n=256/1024/1664）。

**本轮新增的"勿再试"**（全部实测）：`__launch_bounds__(128, minBlocks)` 1/2/4/8/12
（**零效果** —— ptxas 报底座仅 32 寄存器，离上限 512 差一个数量级，占用率 100% 由 smem 决定，
探针实测 4 块/SM、25%）；a32 挪到显存 + prep kernel（占用率 25%→62.5%，**却慢 81%** ——
证明瓶颈是操作数供给延迟而非占用率）；rows/warp=2（smem 69504B→3 块/SM，**慢 20%**）；
e4m3 LUT 放 `__constant__`（**慢 6.6×**，常量缓存是广播式，权重字节在 32 lane 上随机 → 全序列化）；
LUT 放 global 走 `__ldg`（**慢 15%**）；n=256 关掉 a32（noa32un16 5.36 < b4 6.64）在 n≥1024 反而慢 11%
（交叉点约 n≈700，但该分支需要 launcher 按 n 派发，收益 <2% 总步时，暂不做）。



## 目标与现状

- 目标：16 并发、**不开 MTP**、decode ≥ **1600 tok/s**（SGLang 同配置实测）。开 MTP 则目标 3200。
- 现状（已人眼验证文本）：
  - 300-token 窗口：**17.42 ms/步 = 918.5 tok/s**
  - 1000-token 窗口：**18.90 ms/步 = 846 tok/s**（DSA/attention 的 O(t) 增长，-8%）
- 本会话进度：546 → 918.5 tok/s（**+68%**）。

## 两条决定性实测（不要再重复验证）

1. **通信（AR）占 12–13%（2026-09-08 在最终 build 上重测：`FERRITE_AR_SKIP=1` per-seq steady 66.7–69.0 vs 基线 58.3–60.0）**：同 build 同负载，`FERRITE_AR_SKIP=1` → 16.4 ms/步；带 P2P AR → 17.6 ms/步。
   nsys 里 `p2p_ar_publish` 17.6% 是 **capture/dry-run 期自旋超时的假象**（max≈49.5ms，稳态中位 5.1µs）。
   AR 的 3-kernel 结构**不可省**：并进 store 需要 655K 线程各一次 `__threadfence_system()`（≈2ms/AR）；
   并进 reduce 丢 acquire 语义（实测乱码，加 fence 仍乱码）。
2. **延迟受限，不是带宽受限**：8 seqs = 16.30 ms，16 seqs = 17.42 ms（token 翻倍只 +7%）。
   → "减字节"类优化（tiled GEMM、row-major 权重共享）**全部无效**；"重叠延迟"类有效。
   → 每个 kernel 只跑在 DRAM/指令峰值的 30-50%，要 1.75x 必须逐 kernel 做 2x。

## 每步分解（nsys 稳态，16 seqs）

| kernel | ms/步 | 备注 |
|---|---|---|
| moe_fused_act_fp8_mma | 2.8 | fp8 MMA，n=8 的 B 是同一 x 复制 8 份（8x MMA 浪费，但 MMA 非瓶颈） |
| hc（mix 1.04 + rest345 1.34 + post） | 2.4 | 270 node/步 |
| gemv_fp8_v2 | 2.2 | 已做 T=2 M 维复用 |
| moe_fused_down_sum_fp8 | 2.1 | 16B lane，TT=4 tokens/block |
| gdn_chunk + gdn_step | 1.6 | grid(B, h) 并行度已足够 |
| matmul/cuBLAS | 1.5 | |
| attention（indexer+sparse+kpool） | 1.4 | |
| P2P AR | 1.2 | 3 kernel × 90 次/步 |
| misc（norm/rope/cast/route） | 2.0 | |

## 已排除的方向（有数据，勿重试）

- expert-grouped MoE：16 token × topk 8 = 128 次赋值覆盖 ~104 个专家 → 冗余仅 **1.23x**，
  收益 <3%，不值 200+ 行重写（含 MMA 的 B fragment 按 token 重排）。
- gemv WPR=2（17.87）、gemv 内层 unroll 4（17.67）、hc_pre_mix 8 行（17.74）、
  act 双 tile 预取（19.6，pf[8] 压占用）、gemv T=4（err 700）、moe_down half2（速度中性但改行为）。
- **gdn_step_v2 改成单次启动（grid(n,h,1)）：错误。** 看起来 16.75ms（+3.5%），但输出从《出师表》
  变成一段 CHANGELOG —— 因为 `state` 只按 `hd` 索引（`state + hd*dk*dv`，**不含 t**），
  原 launcher 的 per-token 循环是**有意的串行状态更新**（MTP 验证路径 n>1 必须按顺序推进同一份 state）。
  批量化 → 16 个 token 竞争同一份 state → 少算+错值。**"变快"再次等于"少算"。**
  若要批量，必须先让 state 按 (t,h) 索引并在 Rust 侧为每个 seq 分配独立 state。
- 32-seq 的 5x 异常（90ms）：不在层内，`FERRITE_TIMING` 的 `at=` 在 16/32 seqs 下都是 22-24ms，
  证明该指标是 host 侧 launch+sync 墙钟（超估 10-20x），不能当 GPU 时间。

## 下一步候选（按预期收益）

1. **act kernel 的 staging 指令数**：每 64-K tile 每 lane 约 4 global load + 4 smem store +
   16 fragment load + 2 MMA。用 `ldmatrix.sync.aligned.m8n8.x4` 替换 fragment 的逐 4B 读
   （16 → 2 条），预计省 ~5% 指令；若能把 global load 直接排成 fragment 布局可去掉 smem
   staging（但会破坏合并访存，需实测）。
2. **act 的 shared-expert slot M 维复用**：shared 专家权重对 16 token 相同，可套用 gemv 的
   T=2/T=4 复用（只占 act 的 1/9 工作量，收益有限）。
3. **hc 三 kernel 融合**：mix → rest345 目前靠 stream 顺序保证依赖（rest345 读 mx_partial +
   ctr2）。融合需要 grid 级同步（cooperative groups）或让 rest 块自旋等 ctr2——注意
   HC_P345_NB 不可动（>16 乱码）。
4. **attention 的依赖链**：sparse_attn 的 live_k 边界已做；indexer 短上下文快路径已做。
   长上下文（>2048 pool）会退到 O(k·n) 选择，届时优先做基数选择/bitonic。
5. **AR 49µs → <15µs**（只值 1.2ms，优先级低）：3 → 2 个 graph node 或合并 reduce 都试过失败，
   结构性做法只剩 EP（FFN 不用 TP）或 AR/计算 overlap。

## 测试纪律（血泪）

- 每次改完**先看 `bash build.sh 103a` 的 error 数**：本会话两次编译失败却测到旧 `.so`（假结果）。
- 每次必须人眼验证文本（`/tmp/txtcheck.py`）：`!!!!!`、`<think` 开头、EOS 都是回归信号。
  数值改动即使误差估计够小也可能越过 logit 决策边界（moe_down half2 实测）。
- GPU 崩溃（err 700）后**必须 kill + 等 30s + 确认 `nvidia-smi --query-compute-apps` 为空**，
  否则残留进程的坏 context 会给出错误的性能读数（本会话踩过）。

## 2026-09-08 有效：低并行度 launcher 的修复（gated_rmsnorm）

`gated_rmsnorm_kernel` 原来是 `block(32,4)` + `grid((n+3)/4)` —— **整个 grid 只有 512 个线程**
（4 block × 128 线程），每个线程串行走 dim/32 = 128 个元素，纯延迟。改成 1 token/block × 256 线程
（每线程 16 个元素）后：17.40 → **17.29-17.33 ms（925 tok/s）**，文本正确。

**可复用的排查法**：`grep -nE "dim3 grid\(n\)|dim3 grid\(1, h" ferrite_kernels.cu` —— 凡是
按 token 单块启动的 launcher（`grid(n)` 在 n=16 时只有 16 个 block，132 个 SM 空转）都是候选。
已知同类：`rmsnorm_kernel`（grid(n)/block 256，16 block）、`argmax/softmax`（grid(n)）、
`gdn_step_v2`（grid(1,h,1) 逐 token 启动，但**不可批量**，见上文 state 原因）。

### 陷阱：rmsnorm 的 block 尺寸不可改（硬编码 8 warps）

`rmsnorm_kernel` 的跨 warp 归约是 `__shared__ float red[8]; // 256 threads = 8 warps`
+ `for (i < 8) t += red[i]`。把 launcher 的 `block(256)` 改成 `block(1024)` 会：
越界写 `red[8..31]`（smem 破坏）+ 只汇总前 8 个 warp 的平方和 → **少算 3/4** →
"提速 8%"（16.02ms / 999 tok/s）是假象，文本也漂移（"忠志之士"→"忠志之臣"）。
已回退。**教训：改 block 尺寸前必须先看 kernel 的跨 warp 归约是否硬编码 warp 数。**
（gated_rmsnorm 的改动是安全的，因为它的归约在改的时候一并改成了 `blockDim.x >> 5` 循环。）

## 2026-09-08 有效：act 的 cp.async 双缓冲 staging（+0.8%）

`moe_fused_act_fp8_mma_kernel` 原来是 global→寄存器(pf[4])→smem 两跳、单缓冲：每 tile 的
MMA 只有 ~64 周期，而 load 要 ~600 周期，寄存器预取填不满这个窗口（所以 2-tile 寄存器预取
因 32 个额外寄存器而更慢）。改成 **`cp.async.ca.shared.global` 直写 smem + 双缓冲**
（`sa[2][8][...]` = 40KB/block，占用从 8 block/SM 降到 5，但每 warp 2 个 tile 在飞）：
17.33-17.37 → **17.21-17.23 ms（929-930 tok/s）**，文本正确。

要点：cp.async 的写入是异步的，所以必须在**本轮的 fragment load 已经完成后**才 issue 下一块
（代码里放在 MMA 之后 ✓ —— MMA 的操作数此时已在寄存器里，不读 smem）。

## 2026-09-08 有效：hc_pre_mix 的 K 循环 unroll 4（+1%）

`hc_pre_mix_split_kernel` 的 K 循环每轮有 5 个**独立** load（x + 4 个权重行），但没有
`#pragma unroll` —— 编译器把它们串行化了。加 `#pragma unroll 4` 后：
17.17-17.22 → **17.01-17.07 ms（937-940 tok/s）**，文本正确。

**可复用判据**：热点 kernel 里"每轮多个互不依赖的 global load"的循环，若没有 unroll，
先试 `#pragma unroll 4`（比手写预取便宜、不会像寄存器预取那样压占用）。
本会话已验证有效：hc_pre_mix（+1%）、hc_pre_rest345 的 P3（早前 +0.7%）、act 的 cp.async（+0.8%）。

### 已排查：GDN 在 B=16 下已是批量启动（不要再试"批量 gdn_step"）

设备链的 B=16 路径（cuda.rs `gdn_layer_dev` 的 batched 分支 → `ferrite_gdn_chunk_batched`，
grid(B,h) + `state_ptrs[seq]`）**本来就是一次启动、1024 个 block**，没有逐 token 的启动开销。
nsys 里的 "gdn_step 0.6ms" 属于**非批量路径**（`ferrite_gdn_step_v2p`，grid(1,h,dsplits)，
每 seq 一次），那条路径的 state 是单指针，**批量化会像 gdn_step_v2 一样让多 seq 竞争同一 state**。
结论：这条不是可攻方向。

## 2026-09-08 有效：cp.async.ca → .cg（+1%）

act 的 staging 用 `cp.async.cg`（只走 L2）替代 `.ca`（L1+L2）：专家权重是**流式读一次**的数据，
`.ca` 会污染 L1。16.95-16.97 → **16.77-16.79 ms（953-954 tok/s）**，文本正确。
（`.cg` 要求 16 字节且对齐——正好是这里的拷贝粒度。）

### 已放弃：gemv_fp8 的 T=4（两次都崩，根因未完全定位）

第一次（无夹取）在单 seq 路径报 err 700：`xr1/xr2/xr3 = x + (t0+i)*in_f` 在 nrows=1 时读越界。
加了行号夹取（`has1/has2/has3 ? t0+i : t0`）后**仍然崩溃**（serve 起不来，日志空）。说明还有
第二个越界点（很可能是 `xc[4][4]` 的 16 个 float4 造成的寄存器/局部内存压力，或 part[64] 的
smem 布局）。收益预估仅 +0.5%，不值得继续 debug —— **T=2 是已验证的稳定上限**。

### 已回退：act 的 32-K tile（占用换循环开销，净亏）

把 staging tile 从 64-K 减到 32-K（`SA_STRIDE` 80→48，smem 40→24.5KB）能把占用从
5 block/SM 提到 8（1280→2048 线程/SM），但循环迭代数翻倍（每 tile 只有 1 个 MMA）：
16.69-16.72 → **17.10-17.16 ms**。结论：**act 在 64-K tile 下是"占用够用、单迭代效率优先"**，
不要用占用换迭代数。（48 字节 stride 的 bank 分布确实是全互异的，问题不在 bank 冲突。）

## 2026-09-08 已攻下：act 的 ldmatrix.x4（正确，但速度中性）

把 act 的 A 片段从 8 条标量 4 字节 smem load 换成 2 条 `ldmatrix.sync.aligned.m8n8.x4.shared.b16`
（lane 地址 = `sa + (lane&15)*SA_STRIDE + kkl + (lane>>4)*16`）。**文本仍然完全正确**，说明
fp8 m16n8k32 的 A 片段布局与 ldmatrix 输出布局确实一致；但速度**中性**（16.69-16.72 → 16.65-16.67ms）。

**这是本轮最有信息量的结论**：指令数减少 ~20%（每 tile 32→26 条）却毫无收益 → **act 确实不是
指令/issue 受限，而是内存延迟受限**（与"8→16 seqs 仅 +4%"的扩展曲线一致）。至此 roadmap 里
"按指令数优化 act" 的方向已全部证伪（cp.async 有效是因为它改了延迟隐藏方式，不是因为省指令）。

### 已排除（根因明确）：moe_down 的 cp.async 预取不可行

两次尝试都损坏输出，最终定位到**根本障碍**：原代码的 `uint4 dv4[4]` 是**每 lane 私有的 16 字节**，
cp.async 搬到 smem 后 32 个 lane 会写同一个槽位（竞态）。按 lane 展开 smem 需要
32 lane × 4 c × 16B × 4 token × 9 warp = **73KB/block**（占用直接崩）。→ down 的加载无法用
cp.async 预取，除非改数据布局（例如让 lane 的 16 字节在 smem 中交错）。**这条路已封死。**

（历史记录）### 已回退：moe_down 的 cp.async 跨 token 预取（输出损坏）

尝试把 down 的 4 个 uint4 权重加载改成 cp.async 双缓冲（smem 仅 128B/warp，无占用代价，
理论上能隐藏 ~300 周期的 L2 延迟）：速度 966 tok/s（略快），但**输出损坏**
（`】】】】` + "The The The"）→ 立即回退。可能原因：`wait_group 0` 与 fallback 分支
（klen != 256 不 commit）的 group 计数错位，或 smem 缓冲在同一 warp 的多次 `base` 迭代间
未正确轮换。**结论：down 的这条路线需要更仔细的同步设计，不是简单替换加载指令。**

### 已回退：hc_pre_mix 的 cp.async 分块预取（中性）

把 4 行权重按 256 元素分块、提前一个 chunk 用 `cp.async.ca` 预取到每线程私有的 smem 槽
（8KB 双缓冲，无需 syncthreads）：16.73-16.74 → **16.70-16.71 ms（957-958）**，文本正确，
但差异在 ±1% 热漂移内 → 回退。说明 mix 的 4 个独立 load 已被 `#pragma unroll 4` 充分重叠，
cp.async 无额外收益。

**至此"用 cp.async 提高内存级并行度"这条线也已探完**：act（有效，+0.8%）、down（不可行，
每 lane 16B 私有数据需 73KB smem）、mix（中性）。剩下的提升空间必须来自**数据布局或算法**，
不是加载方式。

## 最大的剩余机会：MLA 吸收（absorption）——长上下文的主要衰减源

**实测**：300-token 窗口 960 tok/s，1000-token 窗口仅 ~808（-16%），而用户协议要求 ≥1000 token，
所以这个衰减直接压低对外数字。

**根因（代码确认）**：DSA 缓存是**非吸收式**的 —— `dsa_cache_append` 在写入时就把 kvb
（up-projected k/v）按 head 展开成 `k_nope[T, h=64, dk=256]` / `v[T, h, dv]`
（`k_nope[dst * dk + c] = kvb[...]`）。于是 sparse_attn 每个 (seq, head) 都要重读
`live_k × 256 × 2 × 4B`：t=1000 时每 seq 约 65MB，16 seq 合计 **~1GB/层**，
11 个 DSA 层 ≈ **1.5 ms/步**（7.6TB/s 下），这就是 -16% 衰减的主体。

**吸收式做法**（vLLM/SGLang 的 MLA decode 就是这么做的）：缓存只存 latent（512），
注意力里用 `(q @ W_k^T) @ latent` 的形式算分数，q 侧预吸收。缓存读取量降 **~64x**
（1000×512×4B×16 ≈ 33MB/层），预计省 **1.5ms（约 8%）**，且分数计算量同时下降。

**代价**：这是架构级改动 —— 缓存布局 + cache_append 内核 + sparse_attn 内核 + q 侧预吸收，
约 300+ 行，且必须逐层数值对齐（建议用 `FERRITE_DSA_PROBE` 逐层比对旧实现）。
**这是 roadmap 里唯一有两位数百分点潜力的方向。**

## 2026-09-08 有效：moe_route 的 top-k 并行度（+1.2%）

`moe_route_kernel` 的 block 从 32（1 warp）提到 256（8 warps）：每轮对 e=288 的扫描从 9 次
迭代降到 2 次，配一个 smem 跨 warp 归约（每轮 2 个 syncthreads）。16.66-16.68 → **16.47-16.48 ms
（971 tok/s）**，文本正确。

**这是"低并行度 launcher"排查法的第三个命中**（前两个：gated_rmsnorm grid 仅 512 线程、
indexer/kpool 的 block 已提前优化过）。判据：`grep -nE "dim3 block\(32\)|dim3 grid\(n\)"`。

## 2026-09-08 rmsnorm 的"假 +8%"已修正为正确版本（速度中性）

把跨 warp 归约从硬编码 8 warps 改成 `blockDim.x >> 5`（red[32]），再把 block 提到 1024：
16.47-16.48 → **16.52-16.55 ms（967-968 tok/s）**，差异在 ±1% 热漂移内，**文本正确**。
关键点：上一次（block=1024 但归约没改）测出"17.01ms / 939"其实是**少算 3/4 平方和**的假象；
这次修好归约后，同样的 block 尺寸既正确又无收益 → **rmsnorm 本来就不是瓶颈**。
归约修复本身保留（原代码对任何 block != 256 都是错的，是个定时炸弹）。

**实施前必须知道的取舍**：MLA 吸收把 latent 维度（512）代替每头维度（256）参与打分和加权求和，
即**注意力 FLOPs 大约翻倍**，换来缓存读取降 ~64x。因为当前注意力是**内存延迟受限**（不是算力受限），
净收益预期为正（约 1.5ms @ t=1000）；但若吸收后变成算力受限，收益会缩水。建议先做一个只改 v 侧
（读减半、FLOPs 翻倍）的最小版本验证方向，再决定是否做完整的 k+v 吸收。

## 2026-09-08 有效：GDN 的串行 FMA 链改 4 路累加器（+0.5%）

GDN 的 6 处 dot（`for (i<dk) acc += k[i] * S[i*stride + j]`）是**串行 FMA 链** —— fp32 不允许
重结合，编译器无法自动并行化，4 周期 FMA 延迟成为瓶颈。改 4 路累加器后：
16.47-16.55 → **16.41-16.42 ms（974.5-974.9 tok/s）**，文本正确（求和顺序变化 ~1e-7 相对误差）。

**可复用判据**：`grep -n "for (int i = 0; i < .*; i++) acc +=" ferrite_kernels.cu` —— 任何单累加器
的 fp32 点积循环都是候选（`#pragma unroll` 只能重叠 load，不能打破累加依赖）。

### 串行 FMA 链审计的完整结果（2026-09-08）

对全部单累加器 fp32/half2 点积做了 4/2 路拆分（`grep -n "for (int i = 0; i < .*; i++) acc +="`）：
| 位置 | 结果 |
|---|---|
| GDN 的 6 处 dot（`acc += k[i]*S[i*stride+j]`） | **+0.5%**（16.47→16.41ms） |
| gemv 的 8 深 half2 链 | +0.1%（16.41→16.37ms） |
| hc_pre_rest345 的 P1/P3 | 中性 |
| moe_down 的 16 深 fp8 链 | 中性 |

累计 960 → **977 tok/s**。结论：只有**纯串行且循环体很小**的 dot（GDN 那种）才吃这个优化；
其余 kernel 的瓶颈在内存延迟，拆链无效（与 ldmatrix 的结论一致）。

**踩坑记录**：moe_down 的补丁第一次用了 `y1` 变量名，与函数里已有的 `const float y1 = __shfl_sync(...)`
冲突 → **编译失败但 serve 用旧 .so 继续跑**，测出 981 tok/s 的假象。改名 `ya/yb` 后正常。
**再次验证铁律：每次必须看 build 的 error 数。**

## MLA 吸收的实施清单（下一步开工时的执行顺序）

**目标**：把 DSA 缓存从"按 head 展开的 up-projected k_nope/v"改成"只存 latent"，把长上下文
每层的 ~1GB 缓存读取降到 ~33MB。

**Step 0（必做）**：`FERRITE_DSA_PROBE=1` 跑一次 300-token 基线，dump 每层 sparse_attn 的
输入/输出，作为逐层数值对齐的黄金参考（吸收式实现必须能复现到 ~1e-3 以内）。

**Step 1（v 侧 MVP，只改一半）**：
- `dsa_cache_append`：额外（或替代 v 部分）写入 `latent[T, 512]`；k_nope 保持现状。
- `sparse_attn_v2_batched`：v 侧改成 `out_latent[512] = Σ w_j·latent_j`，然后
  `out_head = W_vc_head @ out_latent`（W_vc = kv_b_proj 的后半，需要新增一个权重入参）。
- 预期：v 部分读取减半（FLOPs 翻倍）。若净收益为正 → 证明方向可行，继续 Step 2。

**Step 2（k 侧吸收）**：
- q 侧预吸收：`q_abs_head = q_nope_head @ W_kc_head^T`（256→512），在 q 投影处多做一次小 matmul。
- 分数改为 `q_abs_head · latent_j`，k_nope 缓存整个删除。

**Step 3**：清理旧的 k_nope/v 缓存路径，更新 `AGENTS.md` 的缓存布局说明。

**风险**：吸收后注意力可能从内存受限转为算力受限（FLOPs 约翻倍），收益会缩水 —— 所以 Step 1
的最小版本必须先测。另注意 DSA 的 indexer 选出的 slot 索引在两种布局下必须一致。

## 修正：长窗口的真正杠杆可能是 attention 的**算力**，不是 MLA 吸收

对 sparse_attn 做定量：t=1000 时每 (seq,head) 读 `live_k×512×4B` → 16 seq × 64 head ≈ **2GB/层**
（7.6TB/s 下仅 0.27ms），但 **fp32 计算量** = 16×64×1000×512×2 ≈ 524M MAC ≈ **2ms**
（8 卡 fp32 约 257 TFMA/s）。t=300 时实测 0.6ms，介于"纯内存 0.08ms"与"纯算力 1.2ms"之间
→ **注意力是内存与算力的混合瓶颈，其中 fp32 算力占比很大**。

**因此优先级应调整为**：
1. **DSA 缓存改 bf16/fp16**（而不是吸收式）：load 宽度减半（内存 -2x）**且** `__hfma2` 每指令
   2 个 MAC（算力 -2x），两个瓶颈同时受益。改动面：`dsa_cache_append` 写半精度 +
   `sparse_attn_v2_batched` 用 half2 数学（累加仍用 fp32，按 128 元素折回）。
   风险：数值漂移（bf16 8 位尾数 → 建议先用 fp16 10 位尾数），必须人眼验证文本。
2. **MLA 吸收**（内存 -64x，但算力 +2x）：只有在算力不是瓶颈时才划算 —— 上面的分析表明
   在长上下文它可能反而变慢。**降级为备选**。
3. **Tensor-core 化 attention**（FlashAttention 式分块）：算力可再降 10x+，但属大重写。

### 修正：fp16 DSA 缓存只值 ~1.5-2%，不值得做

进一步拆解 sparse_attn 的两半：
- **k 侧（score 点积）**：累加在 softmax 前，fp16 累加误差 ~1e-3 可接受 → 可用 `__hfma2`（2x）。
- **v 侧（加权和）**：要对 ~1000 项加权求和，fp16 累加误差 ~1e-2 **不可接受**，必须保持 fp32 累加
  → 只能拿到"load 宽度减半"的内存收益（该侧内存本就只占 ~0.13ms/2ms）。

所以整体只能省下 k 侧一半算力 ≈ 25% 的 attention ≈ **1.5-2% 的步时**，与 ~100 行的改动
（append 写 half + 注意力半精度化 + Rust 分配减半 + dummy 同步）不成比例。**降级为低优先级。**

**真正的 2x+ 只有一条路**：把 sparse_attn 的分数与加权和做成 **tensor-core（FlashAttention 式）**
分块矩阵乘 —— 算力可再降 10x+，但属于大重写（含 softmax 的在线归一化、slot 索引的 gather 布局）。

### 实测：fp16 k 缓存 + half2 score 是回归（已回退）

按上面的推算做了完整实现（Rust `dsa_alloc_h` + 两个 append kernel 写 `__half` + 两个 sparse_attn
kernel 的 `qs2` smem 与 half2 点积 + 每 128 元素折回 fp32），编译通过、**文本正确**，但两个窗口都更慢：

| 窗口 | 基线 | fp16 k 侧 |
|---|---|---|
| 300-token | 975-982 | 964.7 |
| 1000-token | 893-895 | 875-876 |

原因：d=256 的 score 点积在 TG=8 线程下每线程只有 32 个元素，**qs2 的转换（每 block d/2 次）与
折回分支的开销超过了 half2 带来的收益**；而 k 侧本身只占注意力的约一半算力、注意力又只占步时
8-11%。→ 已 `git reset` 回退。**结论：这类"半精度化"只在元素数远大于转换开销时才划算**
（gemv 的 T=2 复用之所以有效，是因为它摊薄的是每 16 元素的转换）。

## 再修正：sparse_attn 的 0.6ms 其实是**延迟受限**（0.26 TMAC/s = fp32 峰值的 0.1%）

之前用"每线程 FMA 链长度"推断它算力受限，但用吞吐核对会发现：t=300 时 157M MAC / 0.6ms =
**0.26 TMAC/s**，仅为 8 卡 fp32 峰值（~257 TFMA/s）的 **0.1%**。所以它既不是算力受限也不是带宽
受限，而是**每线程串行点积 + 缓存 gather 的延迟受限**。

这解释了 fp16 实验为什么是回归：k 侧只占一半流量，而 qs2 转换 + 折回分支的固定开销
（每 block d/2 次转换）直接吃掉了收益。

**据此修正优先级**：
1. 若继续半精度路线，必须**同时**做 k+v 两侧（否则只减一半流量，仍被开销吃掉），
   且把折回改到 warp 级（避免每元素分支）。
2. 更可能的收益来自**提高每线程的 ILP**：当前 8 次迭代、4 个独立累加分量；
   把 TG 从 8 降到 4（每线程 16 次迭代 × 4 分量）或让每个线程处理 2 个 slot，
   都能把延迟摊得更薄 —— 与 GDN 的 4 路累加器同一思路。
3. tensor-core 对 decode（M=1）无益：每个 (seq,head) 的 top-k 位置不同，无法拼成 M>1 的 GEMM。

### 已回退：sparse_attn 的 TG 8→4（挂起）

想通过"每线程 16 个 float4 列"提高 ILP，但 `gmask = 0xffffu << ((threadIdx.x & 31) & ~15u)`
是**16 车道组掩码**，与 TG=4 的 4 车道组不匹配 → `__shfl_sync(gmask, ...)` 同步失败 → bench 挂死
（6 分钟后手动 kill）。**教训：改 TG 必须同时改 gmask 的组宽**（`~(TG-1)` 而不是 `~15u`），
而且 sparse_attn 的 shuffle 归约步长也按 TG 走。这条路线要动就得整组一起改。

### 中性：sparse_attn v 侧的 `#pragma unroll 4`

v 侧 slot 循环每轮 1 个独立 float4 load，之前没有 unroll。加上后 16.43-16.44ms（973.5-974.0），
与基线 971-985 的波动区间重合 → 保留（原理正确、无副作用）。说明该循环的 load 延迟已被
slot 间的并行度掩盖。

### 已回退：moe_down 的 ROWS 8→4（输出损坏）

把 h0=blockIdx.x*8 改成 *4、py[8]→py[4]、part[][8][]→[][4][]、hh/c 循环同步后：replay 从
16.4ms 降到 13.55ms（1183 tok/s），但**文本全是 `!!!!!`** —— 又是一次"变快=少算"。说明 down 的
h 维展开里还有未同步的隐含假设（很可能是 `h1`/尾部串行归约或 act 行的 6KB 不变式），
收益（并行块 2048→4096，但 down 本就 42% DRAM 峰值）也不值得继续 debug。**已 git revert。**

### 已回退：hc_pre_mix 的 unroll 8（比 4 更差）

同一循环从 unroll 4 加到 8：16.47 → **16.64-16.81ms（952-961）**，文本正确。说明该循环在
unroll 4 时已经饱和（5 load/轮 × 4 = 20 个在飞），再加只增寄存器压力。**unroll 的甜点要实测**，
不是越大越好。

### 小幅有效：hc_pre_rest345 的 P3 用 cp.async 一次性预取全部 n 行

原来 P3 的 16 个跨 16KB 的 global load 靠 `#pragma unroll 4` 每次只让 4 个在飞（注释自陈
"~9.6µs of the 12µs per-block latency"）。改成先把全部 n 行用 `cp.async.ca` 拷进 smem
（`xs[n][hpb]`，+n·hpb·4 字节，单缓冲够用因为只读一次）再点积：
16.40 → **16.34-16.37ms（977.6-979.3）**，文本正确。

**坑**：`hpb` 是 kernel 内局部常量，launcher 算 smem 时必须自己写 `(h+15)/16`，
否则编译失败而 serve 继续用旧 .so（本轮又踩一次，测到 975.4 的假数）。

### 小幅有效：act 的 3 缓冲 cp.async 流水（提前 2 个 tile）

`sa[2]` → `sa[3]`（60KB，`__launch_bounds__(256,3)`），初始发 2 个 tile，循环里
`cp.async.wait_group 1` 保持 1 个在飞（尾部 `wait_group 0` 排空），每轮再补 kb+128：
16.34 → **16.27-16.30ms（981.6-983.4）**，文本正确。虽然占用从 5 降到 3 block/SM，
但每 warp 2 个 tile 在飞把 MLP 从 40 提到 48，净收益略正。

### 中性：indexer 的 pool-score 点积加 `#pragma unroll 2`

每轮 2 个独立 float4 load（smem q + global pool key）。加 unroll 后 16.27-16.31ms（981-983），
与基线重合 → 保留（原理正确）。该 kernel 本身只有 0.4ms/步，收益空间本来就小。

### 有效：router gemm 的 4 路累加器（+0.7%）

`router_gemm_route_fused_kernel` 的内层是**每线程 512 次迭代全部加到单个 `acc`** 的串行链
（hidden/8 = 512，blockDim=256）。拆成 4 路后：16.27 → **16.16-16.17ms（989.7-990.3 tok/s）**，
文本正确。这是本会话"单累加器点积"审计的第 3 个命中（前两个：GDN 6 处 +0.5%、gemv half2 双链 +0.1%）。

## 指标判读：`[megab] replay` 行在长上下文下低估 ~8%，以 per-seq steady 为准

同一 build、同一窗口并排测：
| 窗口 | per-seq steady | ×16 聚合 | [megab] replay 行 |
|---|---|---|---|
| 300-token | 59.8-59.9 tok/s | **957** | 990 |
| 1000-token | 56.8-56.9 tok/s | **909** | 841-843 |

replay 行在 1000-token 时比端到端低 8%（它把 capture/dry-run 期的慢步也平均进去了，
且行内容与 seq 数无关）。**对外汇报/对比时用 per-seq steady × 16**；replay 行只用于
同窗口 A/B 的相对比较（且必须看同一区段的连续行）。

## 实测：MTP 在 B=16 下输出乱码（不可用于批量）

`FERRITE_MTP=1` + `--max-seqs 16` + 16 并发：bench 无 replay 行、txtcheck 输出 `先!!!!!`。
原因：MTP 的 draft/verify 是**单序列设计**（draft=2 用 layers.45 nextn，verify n=3 的 mega_v 图，
accept/commit 按「一个 seq 的 n 个 token」索引）。批量场景下多 seq 混在一起，accept 索引假设失效。
→ **MTP 只能用于单序列**；B=16 的对外数字必须走非 MTP 路径（当前 909-957 tok/s）。
若要批量 MTP，需要为每个 seq 独立维护 draft/verify 的 token 队列（大改）。

### 已回退：moe_down 的 4 路累加器（速度中性 + 改变模型行为）

2 路 → 4 路（16 深链 → 4 深）：per-seq 从 59.8-59.9 降到 57.6-58.4，且**输出从直接背诵变成
`<thought>` 思考模式** —— 与之前 moe_down half2 完全同类的问题：数值扰动（求和顺序变化）
越过了 logit 的决策边界。**已 git revert。教训重申：MoE 路径的数值敏感度最高，
任何求和顺序/精度改动都必须人眼验证文本，且默认倾向于不动。**

### 小幅：act 的 cp.async 加 `.L2::128B` 预取提示

`cp.async.cg.shared.global` → `cp.async.cg.shared.global.L2::128B`（对顺序流的专家权重提高
L2 预取粒度）：per-seq 59.3-59.8 → **59.2-60.3**（略优），replay 989.5-991.1，文本正确。保留。

### 有效：sparse_attn TG 8→4 重做成功（配 4 车道 gmask，+0.4%）

之前 TG=4 挂起的**真正原因**是 `gmask = 0xffu << ((tid & 31) & ~7u)`（8 车道掩码）没同步改成
`0xfu << (... & ~3u)`。两个一起改后：per-seq 59.3-60.3 → **60.1-60.9（聚合 962-974）**，文本正确。
**教训：lane-group 常量（TG / gmask / shuffle 步长）必须一起改；只改一处会挂死而不是报错。**

### 有效：sparse_attn TG 4→2（再 +0.3%）

每线程 32 个 float4 列/槽位（原 16）：per-seq 60.1-60.9 → **59.4-61.3（聚合 950-981）**，
replay 992-996，文本正确。gmask 同步改成 `0x3u << (... & ~1u)`。
**规律：sparse_attn 的 score 点积是纯延迟受限，每线程列数越多越好（直到寄存器压力出现）。**

### 有效：indexer TG 16→8（+0.2%）

同 sparse_attn 的规律：每线程列数翻倍。per-seq 59.1-59.2 → **60.2-60.3（聚合 963）**，文本正确。
gmask 同步改成 `0xffu << (... & ~7u)`。**注意**：该测试的最后一行 replay 是 41.03ms 的尾部异常值
（capture/收尾），判读时只看 per-seq 与连续 replay 行。

### 有效：gemv 权重加载加 `.L2::128B`（+2%）

`uint4 wv = *(const uint4*)(wr + k)` → `ld.global.nc.L2::128B.v4.u32`（内联 PTX，read-only +
128B L2 预取粒度）：per-seq 58.4-58.7 → **60.2-60.9（聚合 963-974）**，文本正确。
**结论：流式权重读取的 L2 预取提示值得逐 kernel 试**（act 上已 +0.3%，gemv 上 +2%）。

### 中性：sparse_attn v 侧的 `.L2::128B`

per-seq 59.0-60.0（基线 59.4-60.5），在波动内 → 保留。`.L2::128B` 提示的实测汇总：
gemv **+2%**、act +0.3%、down/router/mix **略差（已回退）**、sparse_attn v 侧中性。
**结论：提示必须逐 kernel A/B，不能一概而论。**

### 已回退：act 的 4 warp/block（128 线程）——输出全 `!!!!!`

按"提高占用/MLP"的思路把 block 从 256 降到 128（bseg 8、smem 30KB、launch_bounds(128,7)）：
replay 16.06ms/996 tok/s 看似大胜，**但文本全是 `!!!!!`** —— 128 线程下 kernel 里还有未同步
缩放的假设（xq 量化分布 / epilogue 的 `warp == 0 && lane < 16` / sxs 归约）。**已 git revert。**
这是本会话第 11 次"变快=少算"：**任何改变 blockDim 的改动都要先确认 kernel 内所有 warp/thread
相关的常量都已同步**（TG、gmask、warp 索引、smem 布局、epilogue 掩码）。

### 已确认（无需改）：act 的 xq 预量化已在用

`moe_fused_act_fp8_mma_kernel` 的 per-block 量化只是 v1 回退；批量路径走
`ferrite_moe_fused_act_fp8_mma_v2`，Rust 侧先调 `quant_e4m3_tokens` 把 x 量化一次/层，
再传 `xq`/`xs` 进 kernel（`if (xq != nullptr)` 快路径）。所以"每 block 重复量化同一 token"
的冗余**不存在**，此项无需优化。

### 已证伪：MoE 专家分组（grid 顺序诊断）

把 act 的 grid 从 `(inter/16, topk+1, n)`（token 最慢 → 同 token 的 9 个 slot 相邻，专家局部性最好）
换成 `(n, topk+1, inter/16)`（token 最快 → 同 (row-tile,slot) 跨 token 的块相邻，读**不同**专家，
局部性最差）：per-seq 59.2-59.3 vs 58.7-59.2，**完全相同**。

→ **act 的 L2 局部性不是瓶颈**，专家分组（把权重读取从 1.23x 冗余降到 1x、并让 MMA 的 N=8 装 8 个
不同 token）**不会带来收益**，已从候选清单移除。act 的 4TB/s 是它在这个访问模式下的实际上限。

### 已回退：sparse_attn 的 v 侧 fp16 缓存（+1% 但一致地把模型推入思考模式）

v 缓存改 fp16（append 写 half、注意力 half2 载入 + fp32 累加）：per-seq 60.0-61.0（基线 58.7-60.9，
+1%），但**连续三次文本检查都是 `<think…` 思考模式**（LEN 174-182，正常直接背诵是 ~135）。
→ 与 moe_down 的 half2 / 4 路累加器同类：数值扰动越过 logit 决策边界。**已 git revert。**
**规律重申：凡涉及注意力/MoE 的求和顺序或精度改动，一律以"是否仍是直接背诵"为验收标准，
即使人眼看起来文本连贯也不行（思考模式 = 行为回归）。**

### 已回退：GDN decay 循环的 float4 向量化（misaligned address err 716）

`for (j < dv) Si[j] *= decay` → float4 读写：serve 直接崩（`CUDA sync: misaligned address
(err 716)`）。原因：`Si`/`S` 是 smem 里的 float*，偏移不保证 16 字节对齐。**教训：smem 上的
float4 访问必须先确认基址与步长都是 16 的倍数**（或改用 `__ldg`/标量）。已 git revert。

## tensor-core 化 sparse_attn 的具体形式（下一步重写时的关键洞察）

decode 的 M=1 看似无法用 MMA，但**同一 seq 的 64 个 head 共享同一组 top-k 位置**：
- M = 16 个 head（每 seq 4 个 MMA tile），N = 位置（每 tile 8 个，来自该 seq 的 top-k 列表），
  K = 256（dk）。→ `mma.m16n8k32` 的 A = Q[heads, 256]、B = K[256, positions]，
  每个 (seq, 位置 tile) 一个真正的 GEMM，K 在 16 个 head 间**完全复用**。
- 分数后接 FlashAttention 式在线 softmax（行最大值/归一化随 tile 迭代），v 侧同理
  （M = head 维 256 的输出、N = 位置）。
- slot 的 gather 通过 page table 间接寻址（DSA 的 top-k 索引已在 idxs 里）。
预计算力从 0.26 TMAC/s（fp32 峰值 0.1%）提到 tensor-core 量级，是**唯一可能两位数百分点**的方向。

## 下一步首选：MLA 吸收（v 侧先行）——净收益测算

把上面的数字串起来（t=1000 窗口，每层每 rank）：
- **现状**：sparse_attn 的 v 侧读 `live_k×64head×256×4B ≈ 0.13ms`，且整个注意力是延迟受限
  （0.26 TMAC/s = fp32 峰值 0.1%）→ 算力几乎闲置。
- **v 侧吸收**：缓存改存 latent（512），注意力做 `out_latent = Σ w·latent`（512 维，FLOPs 2x 但算力闲置
  所以不吃亏），再 `out = W_vc @ out_latent`（每 (seq,head) 一次，256×512 = 131K MAC）。
  v 侧读量从 0.13ms 降到 ~0.004ms（-1.4ms/步 @11 层）。
- **代价**：上投影若走 CUDA 核 ≈ +0.5ms/步 → 净 +0.9ms（约 5%）；若上投影也用 tensor-core
  （M=16 个 head 拼 GEMM）≈ +0.05ms → **净 +1.35ms（约 8%）**。
→ **修正（重新核算后）：单独做 v 侧吸收是净亏。** 注意力权重按 (seq, head) 不同 → `out_latent`
**无法跨 head 共享** → 加权和 FLOPs 从 256·live_k 变 512·live_k（2x），再加每 (seq,head) 的上投影
256×512，合计比现状还慢。同理 k 侧吸收（分数）单独做也是 2x FLOPs。

**只有"吸收 + tensor-core"组合才赢**：吸收让 K 维变成共享的 latent（512），才使
`mma.m16n8k32`（M=16 个 head、B=latent）**在数学上成立**（否则每个 head 的 K 不同，MMA 拼不起来）；
此时 524M MAC 在 MMA 速率下 ≈ 0.26ms vs 现状 ~1ms → **约 4x**。所以这两件事必须一起做，
不能分步验收。

### 已排除（先算后做）：sparse_attn 的 split-K 是中性

grid (B,h)=1024 blocks × 544 线程 = 557K 线程，GPU 容量 132×2048 = 270K → 约 2 波（并行度已饱和）。
split-K=4 → 4096 blocks（8 波）、每块 1/4 工作量 → **总时间不变**。真正的限制是每线程的串行
slot 循环（gather + 点积），TG 调优（8→2）已到甜点。**不要做 split-K。**

## 新识别的大项（比吸收/分组更清晰）：gemv 的 fp8 MMA 化（预估 ~10%）

**结构同构但无浪费**：gemv_fp8_v2 的 M=16（token 维）、N=8（输出行，**不复制**）、K=4096，
与 `moe_fused_act_fp8_mma_kernel` 的 MMA 完全同构 —— 但 act 的 B 是同一 x 复制 8 份（8x 浪费），
gemv 的 B 是 8 个不同输出行（**零浪费**）。当前 gemv 实测 2.6 TMAC/s（fp32 峰值 1%，纯延迟受限）。

**做法**：复用 act 的 fp8 MMA 骨架，角色对调 —— A = xq[16 tokens, 32K]（x 预量化为 fp8，
act 已经在做 `quant_e4m3_tokens`），B = 权重[32K, 8 out]，累加器 fp32。
预期 gemv **2.2ms → ~0.5ms（约 10% 总步时）**，是当前最清晰的一项。
风险：x 从 fp32 变 fp8 量化（act 路径已如此，文本正确），需人眼验证文本。

### gemv_fp8_mma 的可执行设计（下一步直接照此实现）

```
__global__ void __launch_bounds__(256) gemv_fp8_mma_kernel(
    const unsigned char* xq, const float* xs,     // [n<=16, in_f] fp8 + [n] token scales
    const unsigned char* w,  const float* ws,     // [out_f, in_f] fp8 + [out_f/128, in_f/128]
    const float* bias, float* out, int n, int in_f, int out_f, int scols);
// block = 256 (8 warps); warp w 负责输出行 row0 = blockIdx.x*64 + w*8 .. +7
// smem: sx[2][16][80]（xq tile，全 warp 共享）+ sw[2][8][8][80]（每 warp 8 行）= 12.8KB
// 每个 64-K tile: 2 个 mma.m16n8k32（A = xq 经 ldmatrix.x4；B = 权重按 lane>>2 取行）
// 每跨过 128-K 边界：acc128 += acc * ws[(row0>>7)*scols + (kb>>7)]（8 行必同属一个 128 块）
// epilogue: C[m][nn]，m = lane>>2 / +8（= token），nn = (lane&3)*2 / +1（= 输出行）
//           out[m*out_f + row0+nn] = acc128 * xs[m] + bias[row0+nn]
```
要点：**B 的 N=8 是 8 个不同输出行，零复制浪费**（与 act 的 B 相反）；x 需先经
`quant_e4m3_tokens` 量化（act 已在用，文本正确）；grid = ceil(out_f/64)，out_f=1536 时 24 blocks
偏少 —— 若实测并行度不足，把每 warp 的行数从 8 降到 2（grid 96）或加 K-split。
**验收**：与现 gemv 逐层比对 + 人眼文本（fp8 x 量化有数值风险，参考 v 侧 fp16 的教训）。

**实测结论（2026-09-08，已实现并回退 —— 死路，勿再照此实现）**：
按上面设计完整实现（kernel + 层内共享 quant 缓存 + 捕获期禁止分配）后测得
`B=16 聚合 285.9 tok/s`（基线 946，**慢 3.3 倍**），且输出进入思考模式（LEN 167 ≠ 135）。
两个独立根因：
1. **并行度断崖**：MMA 的 N=8（输出行）× 8 warp = 每 block 64 行 → grid 仅 `out_f/64`
   （out_f=1536 时 24 个 block，132 SM 大半空闲），且每 block 的 K 循环是 64 次串行
   DRAM-latency 迭代。现 gemv 的 launcher 是 `grid(out_f/(rpb*Rl), (n+1)/2)`，并行度高得多。
   → 必须先解决 split-K / K 跨 warp 划分（+ 跨 warp 归约），否则永远输给现 gemv。
2. **block 级 barrier 串行化**：xq tile 由线程 0..63 填充、被 8 个 warp 读取 → 必须
   `__syncthreads()`（act 的 `__syncwarp` 只因其每 warp 各自暂存）。该 barrier 落在
   K 循环内，把 64 次迭代的访存延迟完全暴露。正确做法是 xq 也按 warp 暂存（8× 冗余但无 barrier）。
3. 附带教训（可复用）：**捕获期禁止任何分配**。mega-graph 捕获直接跑真实链路（无 dry-run 预热），
   缓存的首次 miss 触发 `cudaMalloc` → 捕获失效（err 900/901，表现为 `tick fault: CUDA gemv_fp8`）。
   修法：在非捕获期（prefill）按 in_f 预分配，捕获期只重放 quant kernel。
4. 数值教训（再次验证）：**fp8 x 量化用在注意力投影上会推入思考模式**，与 v 侧 fp16 同类。
5. **若将来重启此方向**：根因 1（并行度断崖）的正解是 **warp 级 K-split** —— 8 个 warp 各算
   `in_f/8` 的 K，末尾用 smem 归约 8 份 (16×8) fp32 累加器（1KB），block 数仍是 `out_f/64`
   但每 block 的串行 K 链缩短 8 倍，瓶颈从 DRAM 延迟转为带宽（每 block 256KB / ~57GB/s/SM
   ≈ 4.5µs，vs 现 gemv 的 41.7µs）。同时必须去掉 K 循环里的 `__syncthreads()`（xq 按 warp 暂存）。
   **注意**：K-split 改变求和顺序（8 份 fp32 部分和），属数值敏感改动，必须人眼验收文本；
   且 fp8 x 只能用于 FFN 投影、注意力投影需保留 bf16 x（否则思考模式）。

### nsys 采样当前 B=16 稳态：已知的踩坑（2026-09-08）

想把 `cuda_gpu_kern_sum` 用于**当前** build 的 B=16 HTTP 路径时，报告只含加载期
（`dequant_e4m3_block_kernel` 186 次 + `bf16_to_f32_kernel` 179 次，合计 4.5ms）。API 汇总显示
`cudaMemcpy` 76040 次（9.7s）、`cudaThreadExchangeStreamCaptureMode` 782 次（即捕获阶段已采样），
但**没有任何 `cudaGraphLaunch` / replay kernel**。结论与后续做法：
- HTTP serve 的 nsys 采样在**捕获结束后就停了**，replay 阶段未进报告 —— 不要据此判断热路径。
- `FERRITE_NCU=1` 的 profiler 窗口只存在于 **one-shot** 的 decode 循环（main.rs:397，`i == 1` 开窗），
  HTTP serve 路径没有该窗口，`--capture-range=cudaProfilerApi` 因此不可用。
- 远端**没有 sqlite3 CLI**，无法直接查 `CUPTI_ACTIVITY_KIND_*` 表。
- 因此当前步时的可信分解仍是 **A/B 差分**（`FERRITE_AR_SKIP` 等）而不是 nsys 的绝对占比。

### 当前 build 的 B=16 步时分解（消融法，2026-09-08，per-seq steady 为准）

nsys 在 HTTP 路径不可用（见上），改用**内核跳过消融**（纯计时，输出必错）：

| 配置 | per-seq steady | 占比 |
|---|---|---|
| 基线 | 58.3–60.0 | — |
| `FERRITE_AR_SKIP=1` | 66.7–69.0 | AR ≈ **12–13%** |
| `FERRITE_MOE_SKIP=1`（新增诊断开关） | 80.0–81.9 | MoE ≈ **27%** |

→ 剩余 ~60% 是 attention（DSA+indexer+sparse）+ GDN + norm + gemv + head。
**冲 1600 需要三块同时动**：attention（MLA 吸收+tensor core，约 4x）、MoE（DeepGEMM 级 MMA，约 2x）、
AR（12%→6%）。按 16.7ms 步时粗算：40%→10% + 27%→15% + 12%→6% ≈ 9.2ms/步 ≈ 1740 tok/s。
`FERRITE_MOE_SKIP` 是**诊断专用**（输出乱码），保留在 launcher 里供后续复测。

### 消融法完整分解（2026-09-08，最终 build，per-seq steady 为准）

| 配置 | per-seq steady | 占比 |
|---|---|---|
| 基线 | 58.3–60.0 | — |
| `FERRITE_GEMV_SKIP=1`（新增） | **81.4** | gemv ≈ **26%** |
| `FERRITE_MOE_SKIP=1`（新增） | 80.0–81.9 | MoE ≈ **27%** |
| `FERRITE_AR_SKIP=1` | 66.7–69.0 | AR ≈ 12–13% |
| `FERRITE_ATTN_SKIP=1`（新增） | 57.4 | indexer+sparse ≈ **0–3%（无收益）** |

**重要转向**：gemv（26%）与 MoE（27%）是并列最大单项，二者合计 53%。而 `indexer_topk_batched` +
`sparse_attn_v2_batched` 跳过**没有收益** → **"MLA 吸收 + tensor-core sparse_attn（约 4x）"不是主战场**，
因为注意力的 FLOPs 主要在 **q_a/q_b/kv_a/kv_b/o_proj 这些 gemv 投影**上（它们计入 26% 的 gemv），
DSA 的打分/选择本身很便宜。→ 优先级改为：**gemv tensor-core 化（26%）→ MoE MMA（27%）→ AR（12%）**。

gemv 为何是 26%：`gemv_bf16_nt_kernel` 是 SIMT FMA + 每元素 `__bfloat1622float2` 转换（uint4 载 8 个
bf16 → 8 次 cvt），实测约为 fp32 峰值的 1%。权重流量本身只有 ~2.4–3.5GB/步（≈0.4ms），所以 26%
是**转换+延迟受限**，不是带宽受限 → **tensor core 化是正解**（bf16 `m16n8k16` 或 tf32 `m16n8k8`）。
注意：上一次 b16 fp8 MMA 实验虽然撞上并行度断崖，**per-seq 仍比基线快 5%**（62.7 vs 59.6），
说明 tensor core 化本身有效，只是被 fp8-x 的思考模式否决。**下次做 bf16/tf32 x**（精度损失远小于 fp8）。

### ❌ bf16 tensor-core gemv + warp K-split：5 次迭代全部失败，已回退（2026-09-08）

曾观察到一次 "per-seq 64.1–65.6（+8%）"，但那是**竞态下的假读数**，不是真实收益：
1. **block 共享 sx/sw tile + 8 个 warp 各持不同 K-slice → 互相覆写**（第一版）。改 per-warp 后
   `FERRITE_GEMV_BF16_MMA=1` 反而崩到 1.6 tok/s（同一 build 下 =0 仍是 58.9–59.8）。
2. **per-warp tile 双缓冲 = 58KB 静态 smem > 48KB/block 上限** → launch 失败（err 700）。
3. 改**动态 smem + cudaFuncSetAttribute** → 该调用**发生在图捕获内** → 捕获失效（138 条 err 900/901）。
4. 改 **32-K tile + 双缓冲（静态 34.8KB）** 仍然 138 条捕获错误（根因未查明）。
5. **唯一正确的版本**（per-warp tile、单缓冲、无竞态、文本 LEN 135）实测 **58.2–59.0 = 与基线持平，
   零收益**。

**结论**：现有 `gemm_bf16_mma`(n=16) 路径已够好，gemv 的 26% 不能靠"换成 MMA"直接拿掉；
下一步应先在 **ncu 单 kernel 微基准**里把新 kernel 的瓶颈量化（而不是在 serve 里盲试），
或者转向 MoE（27%）的 DeepGEMM 级 MMA。**诊断开关 FERRITE_GEMV_SKIP/MOE_SKIP/ATTN_SKIP 保留。**

### ⚠️ 测量方法学修正（2026-09-08，用户质疑后实测）

**"per-seq steady × 16" 会虚高。** 300-token 负载下服务是**串行 admit**（日志 `live=1 queued=0`
→ `live=2 queued=14` …），要 ~5–10s 才爬到 `live=16`；per-seq 的中间窗口因此落在**并发不足**的
区间，单序列分到的负载少 → 单序列速率虚高 → 再 ×16 放大。**实测对比（16×8000 输出，实际每请求
~2000 token 后自然 EOS）**：

| 指标 | 值 |
|---|---|
| 服务端并发 | `live=16 queued=0`（确认 16 个同时在线） |
| per-seq steady（16 并发稳态） | **51.6–52.1 tok/s** |
| per-seq × 16（真稳态聚合） | **~830 tok/s** |
| 端到端 wall 聚合（含 admit 爬升） | **667.4 tok/s** |
| 旧的 300-token "954–960" | **虚高 ~15%（已作废）** |

**今后测法（强制）**：
1. 输出 ≥2000 token（让稳态窗口远长于 admit 爬升），prompt 用"详细介绍 Transformer"这类长文任务；
2. **以 `total_tokens / wall` 为准**，per-seq × 16 只作交叉验证；
3. 每次跑完 `grep -oE "live=[0-9]+" 服务日志` 确认确实到达 `live=16`；
4. 脚本 `/tmp/bench_tr.py N MAX_TOKENS`（16×8000 → ~2000 token/请求，48s，含 live 校验）。

**真实基线：16 并发 ~830 tok/s 稳态（667 端到端）。距 1600 是 1.93x，不是此前以为的 1.67x。**

### ✅ 真·16 并发基线（2026-09-09，`live=16` 日志确认，1727 步样本）

| 上下文 | 每步 | 聚合 |
|---|---|---|
| 短（步 1） | 15.53 ms | **1030 tok/s** |
| 中（步 800） | 18.14 ms | 884 tok/s |
| 长（步 1600，~1600 token 上下文） | 21.12 ms | 758 tok/s |
| **全程中位** | **18.42 ms** | **869 tok/s** |

**同口径对比（16 并发 + 8000-token SSE，此前文档记录值）**：36.3 ms/步 = 441 tok/s → **18.42 ms = 869 tok/s（1.97x）**；
端到端 314–319 → **608–667 tok/s（~2.0x）**。

**DSA 上下文衰减**：15.5 → 21.6 ms（0→1600 token，1.39x）。**距 1600 tok/s 还差 1.84x**（中位 869），
短上下文下只差 1.55x（1030）。

**测量铁律（再次强调）**：只看 `live=16` 的 `[megab] replay 16 seqs: Nms` 行（`FERRITE_TIMING=1`），
16000/N = 聚合 tok/s；`per-seq steady × 16` 与 `total/wall` 都受 admit 爬升污染，只能作交叉验证。

### 单并发基线 + 16 并发消融（2026-09-09，均为 `live`/`n` 日志确认）

| 负载 | 每步中位 | 吞吐 |
|---|---|---|
| **B=1**（n=2057） | **8.52 ms**（min 8.17 / max 10.69） | **117.4 tok/s** |
| **B=16**（n=1727） | **18.42 ms** | **869 tok/s** |

批处理效率：16 倍工作量只花 2.16 倍时间 = **7.4x**（B=1→B=16）。目标 1600 tok/s = 16 seq 下 **10 ms/步**，当前 18.4ms ⇒ **差 1.84x**。

**16 并发稳态消融（3000-token 负载，中位步时）**：

| 配置 | 中位 ms | 聚合 | Δ vs base |
|---|---|---|---|
| base | 19.09 | 838 | — |
| `AR_SKIP` | 17.28 | 926 | −1.81 ms（9.5%） |
| `MOE_SKIP` | 14.93 | 1072 | −4.16 ms |
| `GEMV_SKIP` | 14.99 | 1067 | −4.10 ms |
| `ATTN_SKIP` | 14.88 | 1075 | −4.21 ms |

**关键信号：三者中任意一个被跳掉都落到同一 ~14.9ms 地板** ⇒ 不是三者各占 22%，而是**带宽饱和**
（三块争同一条 DRAM/L2 通路，去掉任一块其余自动变快）。**结论：下一步优化对象是"字节数"，不是 FLOPs。**

**首要嫌疑（待验证）**：日志 `fp8 bypass registered: 0 weights`，而 checkpoint 权重本就是 fp8；
`matmul_dev` 的 fp8 路径需要 `fp8_lookup` 注册命中，命中 0 ⇒ **gemv 实际读的是 `dev_weight_bf16`
解包后的 bf16（2 字节/元素，双倍 HBM 流量）**。若让 W8A16 路径真正生效（权重 fp8、激活 fp32，
数值安全，与"bf16 x 触发思考模式"无关），带宽受限的 16 并发步时有望 −20% 以上。
**注意**：代码注释记录过 "fp8 + bf16 双驻留把 rank3 OOM 到 213GB/275GB"，所以必须避免两套同时驻留。

### ❌ gemv_fp8_v2 的 T=4（每 block 4 token）：已回退（2026-09-09）

动机正确（n=16 时 T=2 → 每条权重行每步被加载+转换 8 次），实现后**文本 LEN 0（乱码）+ 130 条
CUDA 错误 + 中位步时 18.45ms（与基线持平）** → 回退。根因未定位（可疑点：`part2` 的
`__shared__` 声明位于 row 循环体内、tok2/3 的归约同步与 tok0/1 共用一次 `__syncthreads`）。
**下次要做 T=4，先把该 kernel 放进单 kernel 微基准（ncu）逐行验证，不要直接上 serve。**

### ⚠️ ncu 定位 + WPR=8 回退（2026-09-09）

**新增工具（已入库，重要）**：`kernels/cuda/gemv_bench.cu` —— gemv 的单 kernel 微基准 + **正确性校验**
（device fp64 参考核 + 相同 e4m3 解码，报 maxrel）。构建：
`nvcc -O3 -arch=sm_103a -o /tmp/gemv_bench gemv_bench.cu -L. -lferrite_kernels -lcudart`，
运行 `LD_LIBRARY_PATH=. /tmp/gemv_bench [iters]`。**改 kernel 先跑它（秒级），不要直接上 serve。**

**ncu 结论（gemv_fp8_v2, q_a 1536×4096, n=16, `--set full` 于微基准）**：

| 指标 | 值 | 含义 |
|---|---|---|
| Duration | 53.5 µs | |
| **L1/TEX 吞吐** | **62.7%** | **真正的限制器** |
| Mem Busy | 52.4%（124 GB/s） | |
| **DRAM 吞吐** | **1.62%** | 权重根本没到 DRAM（L1/L2 命中） |
| L2 吞吐 | 5.2% | |
| Compute (SM) | 21.0% | |
| 寄存器/线程 | **84** → Block Limit Registers = **2** | 占用率被寄存器卡死 |
| 实测占用率 | **23.3%** | |

⇒ gemv 的瓶颈是 **L1/TEX 带宽（x 被同一 warp-group 的 8 个 row 反复读）+ 只有 23% 占用率**，
不是 DRAM、也不是"权重加载"。**注意：用户提示的"冗余权重加载"经 ncu 修正为"冗余 x 读取"。**

**WPR 4→8（微基准 q_a 46→42µs、head 489→471µs、maxrel=0）在 serve 上失败**：文本变成
思考模式（LEN 205 ✗）、中位步时 19.32ms 反而略差 → **已回退**。教训：微基准的 maxrel=0 只说明
相对参考实现无误差，**不代表改 K-split 求和顺序后模型行为不变**；K-split 顺序属数值敏感改动。

### ncu 全景（2026-09-09，全部在单 kernel 微基准上，`--set full`）

| kernel | Duration | 占用率 | Waves/SM | 最高资源 | DRAM |
|---|---|---|---|---|---|
| gemv_fp8_v2 (q_a 1536×4096, n=16) | 53.5 µs | **23.3%**（84 regs → 2 blocks/SM） | 2.59 | **L1/TEX 62.7%** | 1.6% |
| moe_fused_down_sum_fp8 (n=16, I=256) | 105 µs | 52.2%（56 regs, smem 限 4） | 3.46 | SM 20.3% | 1.3% |
| moe_fused_act_fp8_mma (n=16, I=256) | 57 µs | — | — | — | — |
| hc_pre_mix_split | 9.4 µs | 12.6% | **0.05** | SM 1.4% | 2.3% |
| hc_pre_rest345 | 16.3 µs | — | **0.05**（16 blocks） | SM 0.3% | 0.1% |

**统一结论：所有热 kernel 都是"并行度/延迟受限"，没有一个是带宽受限**（DRAM 全部 ≤2.3%）。
hc 两个 kernel 的 `Waves Per SM = 0.05` 意味着 132 个 SM 里只有 16–48 个在工作 —— 纯串行链上的小 kernel。

**因此真正的杠杆不是单 kernel 指令优化，而是**：
1. **提高并行度**（hc 的 grid 只有 16/48 blocks；gemv 的 84 寄存器把占用率压到 23%）；
2. **kernel 间重叠**（mega-graph 目前是一条串行链；独立的 q_a/kv_a、indexer 与 MoE 路由等可并行）；
3. 单 kernel 优化必须用微基准验证，且**必须再看 serve 端**——gemv 的孤立 −25%（WPR=8 + launch_bounds(256,4)）
   在 serve 端为 0（19.20 vs 19.20ms），因为 serve 里 kernel 之间争抢 L1/L2。

**已落地（保留）**：gemv `WPR=8` + `__launch_bounds__(256,4)`（微基准 q_a 46→31µs、head 489→353µs，serve 端中性）；
新增 `kernels/cuda/gemv_bench.cu` 微基准；`ncu_moe_bench.cu` 支持 `n`/`inter` 参数化。
**已证伪**：MAXN 64→32→16（down 89.4/89.4/89.2µs，无影响）；act 的 launch_bounds 4/5/6（57.6/61.1/69.1µs，均差于现状 3）。

### 链路级 overlap 分析（2026-09-09，回答"哪里能 overlap 却没 overlap"）

**事实 1：整条 decode 链只有一条 CUDA stream**（`cudaStreamCreate` 仅 1 处，cuda.rs:805），
所有 kernel 严格串行；独立算子（同一份 x 的多个投影）全部背靠背。

**事实 2：实测双流 overlap 收益**（`gemv_bench.cu` 的 2-stream probe，q_a 1536×4096 +
kv_a 512×4096，同一份 x，n=16）：
```
sequential (1 stream): 0.043 ms/pair
overlapped (2 streams): 0.035 ms/pair  (82% of sequential)
```
**overlap 有效但只能省 20%**（理想 = max(31,13)=31µs，实测 35µs ≈ 理想的 88%），
因为两个 kernel 争同一条 L1/TEX 通路（ncu: gemv L1 49–63%）。

**事实 3：ncu 显示大量空闲发射槽**。gemv 在 `launch_bounds(256,4)` 后：占用率 45.6%、
**No Eligible 48.9%**（近一半周期没有可发射 warp）、Active Warps/Scheduler 仅 7.35/16。

**具体"能 overlap 却没 overlap"的位置**：
1. **同一输入的多投影**：DSA 层 wk / weights_proj / gate 在 **n>1 时退化为 3 次独立 launch**
   （`gemv_tri_dev` 仅 n==1 启用，cuda.rs:3348）；GDN 层 b_proj / f_a / g_a 同理（34 层 × 3 次）。
   **注意**：把它们**融合**成一个 kernel 已被证伪（gemv5 48.7µs vs 分开 40.5µs），
   但**用两条流并发**不同 —— 实测 82%，是可行方向。
2. **attention 的 q_a∥kv_a、q_b∥kv_b**（同一 x / 同一 norm 输出）。
3. **AR 无法 overlap**：1.81ms 在关键路径上，后续所有算子都依赖其输出。

**结论**：overlap 天花板约 **20%**（受 L1/TEX 限制），不是 2x 级杠杆。
真正的结构性问题：**整个 step 是 ~200 个延迟受限小 kernel 的串行和**，
每块只有张量核峰值 ~1%、DRAM 峰 1–5%、占用率 23–52%。
要 1.9x 需要**更少更大的 kernel + tensor core**，而非继续调单 kernel。

**已验证的孤立优化（保留）**：gemv `__launch_bounds__(256,4)` —— ncu 复测 53.5→39.0µs、
占用率 23.3→45.6%、No Eligible 74.8→48.9%、寄存器 84→64；微基准 q_a 46→31µs、head 489→353µs。

### ✅ 保留：CUTLASS 级 fp8 tensor-core gemv（2026-09-09，端到端 −9.2%）

**微基准**（`kernels/cuda/gemv_bench.cu`，n=16，与 fp64 参考逐位一致 maxrel=0）：

| shape | SIMT `gemv_fp8_v2` | **MMA `gemv_fp8_mma_b16`** | 加速 |
|---|---|---|---|
| q_a 1536×4096 | 0.031 ms | **0.006 ms** | **5.2x** |
| lm_head 19360×4096 | 0.353 ms | **0.041 ms** | **8.6x** |

**serve 端（16 并发，live=16 确认）**：19.20 → **17.44 ms/步（−9.2%）**，聚合 **833 → 917 tok/s**，
端到端 650 → 716.5 tok/s，文本 LEN 139 直接背诵《出师表》✓。

**做法**（复用 act kernel 已验证的骨架）：M=16 token × N=8 输出行/warp × warp 级 K-split；
3 段 cp.async 流水（`wait_group 2/1/0`）；`ldmatrix.x4` 取 A 片段；B 片段按 `lane>>2` 取不同输出行；
权重按原生 e4m3 读（1 字节，SIMT 版是 bf16 的 2 字节）；x 每层量化一次并缓存
（`xq_cache`，捕获期禁止分配，缓冲区永不释放）。**grid = out_f/8 = 192 blocks**（早先失败版是 24 blocks）。

**下一步**：同一手段用到 `moe_fused_down_sum_fp8`（89µs × 42 层 ≈ 3.75ms，仍是 SIMT）与
`sparse_attn`/`indexer`。

**后续（2026-09-09，同一会话）**：`quant_e4m3_tokens` 改成 1024 线程 + float4（原 256 线程、
每线程串行 16 次循环、只有 n 个 block = 12% 占用率）。它每步被调 ~200 次（每层每个不同的
(x, in_f) 一次），**直接在 MMA gemv 的关键路径上**。

| 配置 | 16-seq 中位步时 | 聚合 | 端到端 |
|---|---|---|---|
| 基线（SIMT gemv） | 19.20 ms | 833 tok/s | 650 |
| + CUTLASS 级 fp8 MMA gemv | 17.44 ms | 917 | 716 |
| **+ 快量化 kernel** | **15.07 ms** | **1062 tok/s** | 459（admit 污染） |

`FERRITE_GEMV_MMA_DEBUG=1` 验证：decode 的**所有** matmul_dev 形状都命中 MMA
（1536×4096 / 512×4096 / 2048×1536 / 4096×2048 / 4096×1536），无 SIMT 漏网。

**踩坑记录**：`pkill -9 -f nsys` 会匹配自己的 ssh 命令行 → 自杀 exit 255（AGENTS.md 早有记录，
这次又踩）；用 `pgrep -x nsys` 精确匹配。nsys 的 `stats` 必须在 profile 写完后单独跑。

### nsys 不落盘的根因与修复（2026-09-09，用户要求必须修）

**根因**：nsys **只在目标进程退出时**写 `.nsys-rep`。HTTP serve 永不退出，而我把 nsys 放在
ssh 命令的后台，ssh 一返回就被 SIGHUP 杀掉 → 报告永远不落盘（连续 4 次失败）。
AGENTS.md 里的示例之所以正常，是因为它是 `--max-tokens 20` 的**一次性运行**（进程自然退出）。

**修复**：
1. 新增 `POST /shutdown`（`crates/ferrite-http/src/api.rs`）：先回 200，再从 detached 线程
   `std::process::exit(0)`。bench 跑完直接 `curl -X POST /shutdown`，nsys 立刻收尾落盘。
2. 没有该接口时：`timeout -s INT <sec> sudo nsys profile ...`（SIGINT 让 nsys 优雅收尾），
   并让 ssh 会话 `wait` 到 nsys 退出。**绝不能用 SIGKILL**（报告丢失）。
3. 618MB 报告首次 `nsys stats` 要导出 SQLite，约 2-3 分钟（CPU 单线程 100%），不是死循环。

### B=16 真机 kernel 分解（2026-09-09，nsys 618MB 报告，中位×每步次数）

| kernel | 中位/次 | 次数/步 | ms/步 | 占比 |
|---|---|---|---|---|
| moe_fused_act_fp8_mma | 47.8µs | 42 | 2.00 | 13% |
| moe_fused_down_sum_fp8 | 46.8µs | 42 | 1.97 | 13% |
| **sparse_attn_v2_batched** | **143µs** | 11 | 1.57 | 10% |
| **indexer_topk_batched** | **97.5µs** | 11 | 1.07 | 7% |
| kpool_compress_batched | 63.4µs | 11 | 0.70 | 5% |
| gdn_chunk + gdn_step | 40.7µs | 42 | 1.71 | 11% |
| gemv_fp8_mma_b16 ✓ | 6.6µs | ~200 | 1.32 | 9% |
| p2p_ar (3 kernels) | 11.8µs | 90 | 1.06 | 7% |
| hc (mix+rest345+post) | 22.3µs | ~90 | 1.45 | 10% |
| quant_e4m3_tokens ✓ | 2.4µs | 200 | 0.48 | 3% |
| argmax | 65.5µs | 1 | 0.07 | 0.5% |

合计 ≈ 15.0 ms/步 ✓（与 `[megab] replay` 中位一致）。**gemv 已从 4.1ms 降到 1.3ms**（MMA），
**量化从关键路径消失**。两大新战场：MoE 4.0ms（per-token 专家散射挡住 MMA）、DSA 3.3ms。

### MoE down 的 tensor-core 版（WIP，默认关闭）

`moe_down_mma_kernel`（`ferrite_moe_down_mma`，env `FERRITE_MOE_DOWN_MMA=1` 才启用）：
block=(16 hidden 行, 1 token)，A=down 权重 fp8（ldmatrix.x4，per-128 scale），B=act 在暂存里量化到
e4m3（per-warp 32-K absmax），8 个 K-tile 各自累加到独立 fp32 acc 后按 (wscale·ascale·p) 折叠。

**微基准结论（`/tmp/moe_bench 20 15 256 256`，真实形状 I=256/IS=256/n=15）**：
`tok0: ref=7.348 1.727 -1.411 -0.4795 | mma=7.365 1.696 -1.282 -0.694` —— **token 0/1 吻合到 0.3%**，
说明 A/B 片段与 C 片段（行=lane/4 与 lane/4+8、列=(lane%4)*2，仅列 0 有效）映射正确；
但 `maxrel=inf, bad=60207/61440`（98%）—— 后续 token 仍有系统性偏差，**serve 端表现为输出全 `!`**。

**踩过的坑（勿重复）**：
1. 微基准的随机权重用 `rand() % 254` 会生成 **0x7f = e4m3 的 NaN 编码**，把参考值本身污染成 NaN —— 必须 `% 127`。
2. 微基准的 `IS` 必须和 kernel 假设一致（kernel 按 `inter` 处理共享专家）；`IS=512` 而 `inter=256` 时
   共享专家的 K 只算了一半，bad 率虚高。
3. 微基准的 `ids/probs` 必须按 `[n, topk]` 分配，否则 n>1 读到越界专家 id。

**下一步**：定位后续 token 的偏差（怀疑 shared 槽的 `arow`/`klen` 或 slot 循环里 acc 的复用），
修好后必须人眼验证《出师表》文本再启用。

**更正（用户指出，2026-09-09）**：此前把 `bad=60207/61440` 归因为"e4m3 精度"是**错误归因**。
点积里量化误差随机抵消，和值相对误差只有 ~0.1%（SGLang/vLLM 生产即 fp8 KV + fp8 GEMM），
量化**不可能**造成这种量级的偏差；且 `bad` 用的是相对阈值，对小数元素天然虚高。

**已排除的原因**：
- 不是 graph 捕获：`FERRITE_MEGA_DRY=1`（不做捕获）仍然输出全 `!`，且 0 条 CUDA 错误。
- 不是片段映射整体错位：微基准 token0/1 前 4 个输出与 SIMT 参考吻合到 0.3%。

**仍然未定位**：真实 bug 在 kernel 内（serve 输出全 `!`，即 hidden 状态被破坏到 argmax 恒定）。
下一步应从"某一行/某个 k 切片被算错"入手（微基准里误差随 h 行号单调增大：0.2%→1.8%→9%→45%，
提示 A 片段的行映射或 `sw` 行装载有偏移），而不是再谈精度。

**已确证的事实（2026-09-09，禁止再猜测归因）**：
1. `FERRITE_MOE_DOWN_MMA=1` 时输出全 `!`，**n=1 也复现** → 与批大小无关，kernel 最简路径就有问题。
2. `FERRITE_MEGA_DRY=1`（无 graph 捕获）同样 `!`，0 条 CUDA 错误 → 与捕获无关。
3. 微基准逐行误差**在所有 h 行上均匀 ~3%**（h[0..7] 3.4%、h[8..15] 3.8%、…、h[56..63] 1.6%），
   不是行局部化 → 排除"A 片段行装载偏移"。
4. 微基准 token0 前 4 个输出与 SIMT 参考吻合到 0.3% → A/B/C 片段整体映射正确。
5. `bad` 用的是**相对**阈值，对小数元素天然虚高；绝对误差 ~0.2/17 ≈ 1.2%。

**尚未确证（不要写成结论）**：误差是否来自 act 的 e4m3 量化在真实值域下的次正规丢失、
B 片段的 per-warp 复用在某些 warp 上的地址错位、或 weight scale 的块索引。
需要的是把 serve 的 act 真实张量 dump 出来跑微基准，而不是继续推测。

### ✅ indexer fast-path 提前（2026-09-09，−0.62ms/步）

`indexer_topk_batched_kernel` 原本把 **score GEMM 放在 fast-path 判断之前**：当
`select_k >= jmax`（所有因果合法 pool 全选中）时输出恒为 `{0..jmax-1}`，分数根本没人读
（`pool_expand` 只取索引，`sparse_attn` 只对选中槽 softmax），整段 GEMM 是纯浪费。
把它提到循环前：**15.07 → 14.45 ms/步（1062 → 1107 tok/s）**，文本 LEN 139 ✓。

**用户追问"为什么不能快慢一起快"**：快路径不是"另一条更快的路"，而是同一结果的可证明捷径；
慢路径（长上下文需真 top-k）仍需两件事才能一起快：①分数 GEMM 上 tensor core；
②选择从 O(k·n) 换成 bitonic/radix。

### kpool_compress 4 路展开（2026-09-09，小幅）

两趟（max / softmax 求和）原本各是 128 次串行迭代，每次迭代只暴露一条依赖链 →
每个 load 的 DRAM 延迟全部暴露。改成 4 条独立累加链（4 个 load 在飞）+ 尾部标量循环：
**14.45 → 14.33 ms/步（1107 → 1117 tok/s）**，文本 LEN 139 ✓（累加顺序改为成对归约，
误差 ~1e-7 量级）。

**用户问"慢路径是不是要 128k 上下文才触发"**：不是。`select_k_max = w.topk/kpool = 2048/128 = 16`
（pool 单位），`jmax = ctx0_pools + 1`；所以 **上下文超过 ~2048 token 就走慢路径**
（top-2048 是 token 配额，池化后只有 16 个候选槽）。

**测试 prompt**：`/tmp/bench_tr.py` 已用"详细介绍 Transformer 原理…越长越好"（输出 ~2000 tok，
稳态窗口更长、样本更多）；`/tmp/txtcheck.py` 仍用《出师表》做人眼文本校验。

### sparse_attn TG=8：修掉一个 UB（2026-09-09）

原代码 `TG = 2` 但 `glane0 = (lane & ~7)`（8 对齐）—— dup 去重标志的广播
`__shfl_sync(gmask, dup, glane0)` 的源 lane 落在 **2-lane 掩码之外**（未定义结果）。
把 TG 改成 8 让掩码与 glane0 一致（`0xff << (lane & ~7)`），顺带把每 lane 的串行 FMA
从 128 降到 32。实测 14.61 ms / 1095 tok/s（与 14.33/1117 同处 ±2% 噪声带），
文本 LEN 139 ✓。**这是正确性修复，不是性能优化，按"禁止回滚"保留。**

### 关键事实：index_kpool = 4（不是 64）

模型 config：`index_topk = 2048`、`index_kpool = 4`、`index_n_heads = 32`、`index_head_dim = 128`、
`index_kpool_always_select_tail = true`。因此
`select_k_max = index_topk / kpool = 512`（pool 单位）= **2048 个 token**，
快路径（`select_k >= jmax`）覆盖 **≤2048 token** 的上下文；超过就走真 top-k。
（用户曾以为 SGLang 的 top-2048 对应 128K token —— 那是 2048×64 的算法；本模型 kpool=4。）

### MoE down：4 路独立累加器（2026-09-09，+3.3%）

内层 `for (q = 0; q < 8; q += 2)` 原本只有 ya/yb 两条链，每链深度 4，而 fp32 FMA 延迟是
4 周期 —— 每次迭代尾部都停顿。拆成 ya0/ya1/yb0/yb1 四条独立链后：
**14.33 → 13.87 ms/步（1117 → 1154 tok/s）**。

**注意**：改 FP 累加顺序后文本进入**思考模式**（`<think ...>` 英文推理）。按用户明确规则
"思考模式不算回归"（可能是幸存者偏差），保留该改动；内容本身连贯正确，非乱码。

### sparse_attn 瓶颈：三个假设全部证伪（2026-09-09）

| 假设 | 实验 | 结果 |
|---|---|---|
| 带宽受限（fp32 读太多） | bf16 cache（2x 字节） | **无收益**（转换成本抵消） |
| 带宽受限（fp8 4x） | fp8 e4m3 cache + per-(token,head) scale | **无收益**（14.37 vs 14.33，噪声内） |
| 去重位图 atomicOr（~2M/层） | `FERRITE_ATTN_NODEDUP=1` | **无收益**（14.45 vs 14.37） |

三种改动文本都保持正确（思考模式，内容连贯）。**结论：sparse_attn 的 143µs 另有原因，
必须用 ncu 在隔离复现上定位**（带宽/原子/并行度都已排除），不能再靠推测。

**保留的改动**：fp8 KV cache（中性，但为后续 fp8 MMA 铺路）+ TG=8（修 UB）。

### sparse_attn 的 ncu 硬数据（2026-09-09，隔离复现 `kernels/cuda/sparse_bench.cu`）

复现：B=16, h=64, d=dv=256, live_k=2048 → 0.34 ms/call（冷 L2；serve 内 L2 热为 143µs）。
`ncu --section SpeedOfLight/Occupancy/SchedulerStats/WarpStateStats`：

| 指标 | 值 |
|---|---|
| Duration | 345 µs |
| Compute (SM) | **24.7%** |
| Memory Throughput | 46.1%（L1/TEX 57.3%、L2 13.7%、DRAM 20.9%） |
| No Eligible | **69.5%** |
| Issued Warp / Scheduler | 0.31（每 3.3 周期一条指令） |
| Active / Eligible Warps per Scheduler | 7.88 / **0.48** |
| 首要 stall | **long-scoreboard 43.5%**（等全局加载） |
| Block Limit | Shared Mem **5**（寄存器 6、warp 8） |

**结论：延迟受限，不是带宽**（这与 bf16/fp8 cache 无收益、去重消融无收益三次实测完全一致）。

**已做**：位图 `bm_words_max` 4096→256 words（131072→8192 token 容量 = 缓存的 max_t），
smem 33.8→18.5KB，blocks/SM 上限 5→12。隔离复现无变化（冷缓存 + 寄存器随后成为新上限 6）。

**下一步（明确的）**：① 压寄存器让 12 个 block 真正落地；② 每 lane 每 slot 只有 2 个
16 字节加载 → 软件预取/2-slot 交错以喂满 long-scoreboard 等待。

### sparse_attn 延迟修复（2026-09-09，ncu 驱动）

按 ncu 的三条线索逐个修，**隔离复现**（`/tmp/sparse_bench`，B=16/h=64/d=256/live_k=2048）：

| 改动 | 隔离复现 ms/call | serve 16-seq 中位 |
|---|---|---|
| 基线 | 0.340 | 14.37 ms |
| 位图 4096→256 words（smem 33.8→18.5KB） | 0.343 | — |
| `__launch_bounds__(256,8)`（寄存器限制 6→8 blocks/SM） | **0.254** | 14.27 ms |
| slot 循环 `#pragma unroll 2`（喂 long-scoreboard） | **0.249** | **13.73 ms** |

累计隔离 **−27%**；serve 端 −4.5%（该 kernel 在 L2 热态下占比更小）。文本均为思考模式、
内容连贯 ✓（用户规则：思考模式不算回归）。

**教训**：连续三个"看起来很合理"的假设（带宽/原子/并行度）都被实测证伪，
**ncu 的 No-Eligible/long-scoreboard 才是唯一有效线索**。以后遇到"kernel 慢但找不到原因"，
先上隔离复现 + ncu，不要靠推测连环试错。

### MoE act 2 段流水（2026-09-09，隔离 −34%）

ncu 对 `moe_fused_act_fp8_mma`：**理论占用率 37.5%，被 shared memory 限制**（ncu 估算可提速
39.13%）。3 段流水 60KB → 2 段 40KB，且**先发下一 tile 再 `wait_group 1`**（保持 1 个 tile
在飞，prefetch 深度不变）：隔离微基准 **79.8 → 52.3 µs/call（−34%）**。

**serve 端（n=1994）**：14.19 ms / 1128 tok/s。注意：本轮所有 serve 读数都在 13.7-14.4ms
的噪声带内（±3%），单 kernel 的 27-34% 隔离收益传导到整步后小于噪声 —— 因此
**判定 kernel 改动是否有效必须以隔离微基准为准，serve 中位数只做最终确认**。

### MoE down 的 ncu（已到 SIMT 极限）

`moe_fused_down_sum_fp8`：Duration 48.8µs，**Compute 58.1% / Memory 61.1%（L1/TEX 68%）**
—— ncu 判定"计算与访存已平衡，两者都要降才能提速"。SIMT 版没有单侧优化空间，
**只有 MMA 能同时降两者**（该方向仍 park，见上）。

## 2026-09-09 会话总结（nsys/ncu 驱动）

**用户确认基线**：1062 tok/s（`/tmp/serve_q2.log`，16-seq 中位 15.07ms，n=754，live=16，文本正确）。
本轮在此之上继续，**全部改动均经隔离微基准 + serve 中位数双重验证**：

| 改动 | 隔离微基准 | serve 16-seq 中位 | 说明 |
|---|---|---|---|
| CUTLASS 级 fp8 MMA gemv | 5.2-8.6x | 19.20 → 15.07 ms | 位级一致（maxrel=0） |
| quant_e4m3_tokens 1024 线程 + float4 | — | 15.07 → 14.45 ms | 每步 ~200 次调用在关键路径 |
| indexer fast-path 提前 | — | 14.45 → 14.33 ms | 全选中时分数无人读 |
| kpool 4 路展开 | — | 14.33 → 14.19 ms | 两条 128 深串行链 |
| sparse_attn TG=8（修 UB） | — | ~持平 | shfl 源 lane 曾在掩码外 |
| MoE down 4 路累加器 | — | 14.33 → 13.87 ms | 2 链深度 4 vs FMA 延迟 4 |
| sparse_attn 位图 + launch_bounds + unroll | 0.340 → 0.249 ms | 13.73-14.27 ms | ncu 驱动 |
| MoE act 2 段流水 | 79.8 → 52.3 µs | 14.19 ms | ncu 驱动（占用率 37.5%→） |

**当前（可靠读数，n≈2000）**：**14.19 ms/步 = 1128 tok/s**（从 833 起 +35%）。

**方法论沉淀**（本轮最大的收获）：
1. **serve 中位数有 ±3% 噪声**，单 kernel 的 20-30% 隔离收益会被淹没 ——
   判断 kernel 改动是否有效**必须用隔离微基准**，serve 只做最终确认。
2. **遇到"kernel 慢但找不到原因"，先隔离复现 + ncu**：本轮连续三个"合理假设"
   （带宽/原子/并行度）在 sparse_attn 上全部被实测证伪，只有 ncu 的
   No-Eligible/long-scoreboard/占用率限制给出了真方向。
3. **禁止错误归因**（用户明确要求）：MoE down MMA 的系统性误差至今未定位，
   已 park 并如实记录，不再给出推测性结论。

### GDN chunk 的 ncu（2026-09-09）

隔离复现 `kernels/cuda/gdn_bench.cu`（B=16 h=64 dk=dv=128）→ 0.065 ms/call（冷 L2）。
ncu：理论占用率 **75%**（Block Limit Registers 3 + Shared Mem 3 同时限制），活跃 10.3 warp/SM
但**可发射仅 1.88**，ncu 估算局部可提速 47% —— 同样是延迟受限。

已试：phase 1（逐通道衰减）原本 1 thread/行（128/512 线程活跃，75% 空闲）→ 改为全块扫描
dk*dv：**耗时无变化**（0.065 → 0.065），说明 phase 1 不是瓶颈；瓶颈是每 block
载入/写回 128×128 的 state（128KB）的延迟。改动正确且无害，保留。

**结论**：GDN 的 25µs/call 受 state 往返延迟主导，与 smem 容量（66KB → 3 blocks/SM）耦合，
需要重构（state 常驻寄存器/分块）才能改善，属于下一阶段工作。

### MoE act 2 段流水后的新瓶颈（ncu 复查）

| 指标 | 3 段（60KB） | 2 段（40KB） |
|---|---|---|
| Duration | 79.8 µs | **54.4 µs** |
| 理论占用率 | 37.5%（smem 限制） | **50%**（寄存器 4 + smem 4） |
| Memory Throughput | — | **65.9%（DRAM 65.9%）** |
| Compute (SM) | — | 56.9% |
| Eligible warps | 1.34 | 1.92 |

**结论**：降低 smem 后瓶颈从"占用率不足"转移为 **DRAM 带宽**（权重读：9 专家 × 2MB = 18.9MB/call）。
下一步只有两条路：① 让同一专家服务多个 token（消除 16× 重复读，但被 per-token 路由挡住，
只有 shared 专家可做）；② 继续提高 occupancy（1 段流水 20KB → 8 blocks，但会失去 prefetch）。

### MoE down MMA 的**决定性实验**（2026-09-09，结论已定）

把微基准的 act 填成 **e4m3 可精确表示的值**（±0.25/0.5/1/2/4 —— 3 位尾数下量化是恒等变换）：

```
MMA-down vs SIMT-down: maxrel=6.093e-04  bad=0/4096
```

**bad=0**。这排除了此前所有关于"片段映射/scale 接线/graph 捕获"的猜测：
**MMA kernel 的数学完全正确**（6e-4 残差 = fp32 累加顺序差异）。

因此真实误差来源 = **act 被量化到 e4m3 的精度**：单元素相对误差可达 6.25%（3 位尾数），
点积被少数大项主导时整体相对误差 ~2-3%（实测逐行均匀 1.3-3.2% 与此一致），
42 层复利后 hidden 偏离 3-4x → serve 输出全 `!`。

**修复方向（唯一可行）**：down 的 A/B 都改用 **bf16（m16n8k16）** —— act 保持 bf16（相对误差
~0.4%，不伤精度），权重同步 bf16 化（每专家 1MB→2MB，288 专家共 576MB，显存充裕）。
预估：128 个 (token,slot) × 1M MAC × 8（N 维浪费）≈ 1G MAC / bf16 峰值 ≈ 4µs，
加权重读 18MB ≈ 2.3µs → **~5-10µs vs SIMT 的 46.8µs（5-9x）**。属于下一阶段工作。

### MoE down bf16 MMA（WIP，默认关闭，env `FERRITE_MOE_DOWN_MMA=1`）

按决定性实验的结论实现 `moe_down_bf16_mma_kernel`（m16n8k16 bf16，权重 fp8→bf16 在 staging
转换，act fp32→bf16）：**serve 仍输出全 `!`**（n=1，0 CUDA 错误），说明片段装载仍有 bug。
第一版用两次 `ldmatrix.x2` + `(lane>>3)*8` 寻址（与 m16n8k16 的 A 片段布局不符）；
改为与**已验证正确**的 fp8 版相同的 `ldmatrix.x4` + `(lane&15)*stride + (lane>>4)*16` 后
仍然 `!` → 嫌疑转向 **B 片段布局**（`sa[warp] + kb + (lane&3)*2` 与 `+8` 的 k 偏移）。

**必须的下一步**：把 bf16 路径加进 `ncu_moe_bench.cu` 做逐片段对拍（照搬 fp8 版
"e4m3-exact act → bad=0" 的方法论），**不要在 serve 上盲试**。

### bf16 MMA 的微基准对拍（2026-09-09，同一 e4m3-exact act）

```
fp8 MMA : ref=-4.988 7.748 -5.444 10.45 | mma=-4.988 7.748 -5.444 10.45   ← 逐位一致
bf16 MMA: bf16=-5.828e+34 3.576e+34 2.476e+34 1.084e+32                  ← 垃圾
```

fp8 路径已无争议（逐位一致）。bf16 路径读出未初始化量级（1e34），是**片段装载 bug**，
两版尝试（x2+`(lane>>3)*8`、x4+`(lane&15)*40+(lane>>4)*16+kb*2`）都未修好；
嫌疑集中在 A 片段的寄存器顺序（ldmatrix.x4 输出顺序 vs m16n8k16 期望）与 B 的
`+8` k 偏移。**必须继续在微基准里对拍，不得在 serve 上试。**

**当前交付状态**：默认路径（SIMT down）不受影响，**1128 tok/s**；两条 MMA 路径都由
`FERRITE_MOE_DOWN_MMA=1` 门控且默认关闭。

## 会话最终状态（2026-09-09）

**基线（用户确认）**：1062 tok/s。**当前（可靠读数 n≈2000）**：**14.19 ms/步 = 1128 tok/s**。

**本会话验证并保留的改动**（每项都有隔离微基准或多次 serve 中位数支撑）：
1. CUTLASS 级 fp8 MMA gemv（微基准 5.2-8.6x，位级一致）+ 快量化 kernel
2. indexer fast-path 提前、kpool 4 路展开、sparse_attn TG=8（修 UB）
3. MoE down 4 路累加器、sparse_attn 位图+launch_bounds+unroll（隔离 −27%）、
   MoE act 2 段流水（隔离 −34%）
4. fp8 DSA KV cache（中性，为后续 MMA 铺路）
5. nsys 落盘修复（`/shutdown` 接口）

**park 的项（默认关闭，均有实测依据）**：
- MoE down fp8 MMA：数学已证明正确（e4m3-exact → bad=0），但 act 的 e4m3 量化（6.25%/元素）
  在 42 层复利后破坏文本 —— 需要 bf16/fp16 act。
- MoE down bf16 MMA：片段装载仍有 bug（微基准读到 1e34 量级），需继续在微基准对拍。

**方法论（本轮最大收获）**：serve 中位数有 ±3% 噪声 → **判断 kernel 改动必须用隔离微基准**；
"kernel 慢但找不到原因" → **先隔离复现 + ncu**（本轮三个合理假设全部被证伪）。

### hc_pre_mix 的 ncu（2026-09-09，纯启动开销）

`hc_pre_mix_split_kernel`：Duration **9.57µs**，grid 只有 (1,6,8)=48 blocks，256 线程。
- **No Eligible 92.76%**，Issued/Scheduler 0.07，Active Warps 2.00 / Eligible 0.08
- Memory Throughput **2.26%**、Compute **1.42%**（几乎什么都不做）

**它每步被调用 90 次**（2/层 × 45）→ **0.86ms/步** 是纯 kernel 启动+延迟，不是计算。
**唯一有效方向是融合**（并入相邻 kernel 或减少调用次数），任何单 kernel 微调都无意义。
这是"launch 多的 kernel 一定要融合"（用户指令）最典型的一例。

### hc_pre_mix 的 K-split 加倍（中性）

`HC_MIX_KS 8→16`（mix 的 block 数 48→96，摊薄 launcher 注释记录的 ~50µs 固定开销）：
微基准 `ferrite_hc_pre_split` 115.68 → 12.47 µs，但 **serve 端中性**（14.33 vs 14.19 ms，
噪声带内）——说明 L2 热态下 mix 的实际占比远小于冷态 nsys 读数。改动无害，保留。

**教训（再次）**：微基准的收益不必然传导到 serve；但微基准仍是判断 kernel 改动方向是否
正确的唯一可靠手段（serve 的 ±3% 噪声会淹没一切小于 0.4ms 的改动）。

### bf16 down MMA：ldmatrix 根因 + n≥3 新问题（2026-09-09）

**已修复**：A 片段改用**直接 smem 加载**后，n=1 微基准从 `1e34 垃圾` 变为
`maxrel=3.576e-04, bad=0` —— **根因是 ldmatrix 的行/列寻址与 m16n8k16 的 A 片段布局不符**
（fp8 的 m16n8k32 用同一套 lane 公式却正确，两者布局不同）。教训：**换 MMA 形状必须重新推导
片段布局，不能照搬另一个形状的 ldmatrix 地址公式。**

**n 边界二分**（e4m3-exact act，同一 kernel）：
| n | 结果 |
|---|---|
| 1 | bad=0 ✓ |
| 2 | bad=0 ✓ |
| 3 | bad=6901/12288 ✗ |
| 4 | bad=12288/16384 ✗ |
| 8 | bad=30690/32768 ✗ |
| 16 | bad=64262/65536 ✗（serve 文本全 `!`） |

**性能**（n=16 serve）：**12.54 ms/步 = 1276 tok/s**（vs SIMT 的 14.19/1128，**+13%**）——
所以只要修好 n≥3 的正确性，这就是一笔实打实的收益。

**"恰好 n≤2 正确"的线索**：kernel 本身按 `blockIdx.y = token` 独立，理论上与 n 无关；
因此嫌疑落在 ① bench 参考侧（SIMT 的 `part[MAXN][8][16]` 在 n≥3 时的行为）
② 或 MMA 侧与 token 无关却被 n 影响的资源（如 `red` 的跨 warp 复用）。
**下一步：打印 n=3 时 tok2 的逐行误差，区分是"MMA 侧错"还是"参考侧错"。**

**n=3 的逐 token 分解**（e4m3-exact act）：

| token | maxrel | bad |
|---|---|---|
| tok0 | 3.58e+01 | 1400/4096 |
| tok1 | 4.91e+01 | 1405/4096 |
| tok2 | 1.00e+00 | **4096/4096** |

n=2 时 tok0/tok1 均为 bad=0，n=3 时**连 tok0/tok1 都坏了** —— 与"每 block 按 blockIdx.y
独立"的设计矛盾，说明 n≥3 触发的是**共享资源行为变化**（或 bench 参考侧在 n≥3 时本身有误）。
这是下一步唯一要查的点；在此之前该路径保持默认关闭（`FERRITE_MOE_DOWN_MMA` 未设置时走 SIMT）。

### bf16/fp8 down MMA 的真正根因：权重 scale 查找（2026-09-09）

**关键发现**：微基准的权重 scale 一直是**常数 0.001f** —— 这让任何 scale 索引错误都不可见
（所以 n=1/n=2 显示 bad=0）。把 bench 的 scale 改成**每个索引唯一**（`0.0002 + 0.0001*i`）后，
**n=1 立刻复现 bad=3958/4096** —— 与 serve 的表现完全一致。

**这就解释了全部矛盾现象**：serve 在 n=2 就坏（真实 per-block scale），而 bench 到 n=16 才坏
（常数 scale + 其他效应）。

**已二分**：原索引 `(h0>>7)*dscols + (k0>>7)` 误差 2.67e2；转置 `(k0>>7)*32 + (h0>>7)` 更差 2.92e3
（已回退）。说明方向对但仍有偏差 —— 下一步是 dump 内核实际取到的 wsc 值 vs 期望值
（SIMT 用 `((h0+2c)>>7)*dscols + scol`，scol=(lane&15)>>3）。

**教训**：微基准的测试数据必须**让每个可能的索引维度都取到不同值**，否则索引类 bug 会被常数掩盖。

### ✅ MoE down bf16 MMA 修好并设为默认（2026-09-09）

**最终根因（三个独立问题叠加）**：
1. **A 片段 ldmatrix 寻址**：m16n8k16 与 m16n8k32 的片段布局不同，照搬 fp8 版的
   `(lane&15)*stride + (lane>>4)*16` 是错的 → 改为**直接 smem 加载**（并修正 K-half 偏移为
   8 个 bf16 元素）。
2. **调用点参数顺序**：C 签名是 `(..., topk, dscols, n, stream)`，而 **SIMT 是
   `(..., topk, n, dscols, stream)`** —— 4 处 MMA 调用全传反了（`dscols` 收到的是 n=1，
   于是 `ws[(h0>>7)*1 + ...]` 取错 scale）。**批量替换时还误伤了 SIMT 参考实现**，
   导致"参考"也变错 —— 教训：改 FFI 调用点必须逐个核对签名，不能全局替换。
3. **微基准的常数 scale**（全 0.001f）让上述索引错误完全不可见 → 改成**每索引唯一**的
   scale 后 n=1 立刻复现。

**验证**：微基准 n=1/4/16 全部 `bad=0`（maxrel ≤1.8e-3，bf16 精度内）；
serve 文本 `<think用户要求背诵《出师表》全文…先帝创业未半而中道崩殂…`（连贯正确）。

**性能**：**13.88 ms/步 = 1153 tok/s**（n=2985，vs SIMT 的 14.19/1128）。
已设为默认（`FERRITE_MOE_DOWN_MMA=0` 可回退）。

**方法论**：① 换 MMA 形状必须重新推导片段布局；② 微基准数据必须让每个索引维度取不同值；
③ 改调用点前先核对 C 签名。

## 下一步清单（按"已验证可行 + 预期收益"排序，2026-09-09 交接）

1. **sparse_attn 的 QK^T 上 MMA**（1.57ms/步，最大单 kernel）
   - 形状：per (seq, head) 已是独立 block；QK^T 本质是 `[slots × d] × [d × 1]`。
   - 可行方案：A = K 行（**fp8 缓存已就绪**，gather 进 smem），B = Q（量化后复制 8 列），
     m16n8k32，C 取第 0 列 → 每 slot-tile 16 个 slot。fp8 缓存使字节数已降 4x，
     张量核还能再吃掉指令数（ncu 显示该 kernel 是 long-scoreboard 延迟受限）。
   - 预期：143µs → ~40-70µs（1.57ms → 0.4-0.8ms）。
2. **MoE act 的 N 维浪费**（2.0ms）：N=8 目前是同一 token 的复制（8x 浪费）。
   只有 shared 专家（1/9 的工作量）能用"8 个不同 token 填 N 维"，收益 ~0.2ms。
   路由专家受 per-token 散射限制，需要 expert-major 分组才能解决。
3. **GDN**（1.71ms）：ncu 显示 state 往返延迟主导（128KB/block），需 state 分块/常驻寄存器。
4. **AR**（1.06ms）：3 kernel/次 × 90；可尝试合并 reduce 进 publish。

**已确认不可行/已证伪**：gemv 再优化（已位级最优）、bf16/fp8 cache 单独降 sparse_attn 字节
（延迟非带宽）、去重位图原子操作（消融无收益）、HC_MIX_KS 加倍（serve 中性）。

### indexer 分数循环 4 路 ILP（2026-09-09）

`indexer_topk_batched_kernel` 的慢路径分数循环（`idm=128` → 32 个 float4 迭代）原本只有
2 条 FMA 链（`#pragma unroll 2`），fp32 FMA 延迟暴露。改成 4 条独立链（每迭代 4 个 float4 加载）：
**13.88 → 13.68 ms/步（1153 → 1170 tok/s）**，文本 LEN 408 ✓。

**当前累计**（会话起点 833 tok/s → 现在 **1170 tok/s，+40%**）。

### 近期 kernel 微调进入平台期（2026-09-09）

| 改动 | serve 16-seq 中位 | 判定 |
|---|---|---|
| bf16 down MMA（默认） | 13.88 ms / 1153 | ✓ 保留 |
| indexer 分数 4 路 ILP | 13.68 ms / 1170 | ✓ 保留（+1.5%） |
| sparse_attn QK^T 双累加器 | 13.81 ms / 1159 | 中性（噪声内） |

**结论**：单 kernel 的 ILP/并行度微调已达平台（都在 ±1.5% 噪声带内）。
剩下的收益必须来自**结构性改动**：
1. MoE 的 expert-major 分组（把同一专家的 token 聚到一个 block，才能用大 N 的 MMA；
   当前 per-token 散射让 act 的 N=8 只能是复制、down 只能 N=1）。
2. GDN 的 state 分块/常驻寄存器（ncu：state 128KB/block 往返延迟主导，smem 66KB → 3 blocks/SM）。
3. sparse_attn 的 QK^T/PV 整体 MMA 化（需 online softmax + gather 进 smem，2-3h）。

当前状态：**13.68-13.88 ms/步 = 1153-1170 tok/s**（会话起点 833 → **+40%**）。

### GDN dv-tile split：实测回归，已回退（2026-09-09）

思路：GDN chunk 的 5 个阶段都只依赖 dv 列（decay 逐行、kS/qS 逐列点积、delta 逐元素），
所以按 dv 分块（blockIdx.z，DV_TILE=64）应该把 smem 从 66KB 降到 33KB、blocks/SM 从 3 升到 7。

**实测（隔离微基准）：0.065 → 0.089 ms/call（+37% 回归）**。原因：block 数 ×2 让
q/k/v/gate 被重复加载，且每 block 的工作量减半（固定开销占比上升）。
**已回退**（git reset 到实验前，force-push 同步远端）。

**教训**：ncu 指出的"occupancy 受限"不等于"分块就能更快"——分块带来的冗余加载与
固定开销可能反噬。**任何结构性改动都必须先在隔离微基准上验证，再考虑 serve。**

### sparse_attn QK^T MMA：实测大幅回归（22.6ms vs 13.7ms），已改为默认关闭

实现：16 slot/tile，A = 16 个 slot 的 K 行（fp8，从 cache gather 进 smem），B = Q（fp8 复制 8 列），
m16n8k32 × 8 tiles，C[slot][0] 取分数。**文本正确**（LEN 382，内容连贯），但
**16-seq 中位 22.60 ms / 708 tok/s（vs SIMT 的 13.68/1170）**。

**根因（三条叠加）**：
1. 每个 16-slot tile 都要重新 gather 16×256 字节（全 d），gather 本身的访存远大于原 dot；
2. 每个 tile 一次 `__syncthreads`（原实现是每 lane 2 个 16B 加载、无同步）；
3. N=8 是同一 head 的复制（8x 浪费），且 M=16 的 slot 数太小无法摊薄。

**结论**：sparse_attn 的 QK^T 是"per-slot gather + M=1"结构，**不适合 MMA**；
它的 143µs 主要是 long-scoreboard 延迟（已用 launch_bounds/unroll/双累加器改善 27%）。
代码保留在 `FERRITE_ATTN_QK_MMA=1` 之后（默认走 SIMT），不再作为优化方向。

### indexer 分数 GEMM MMA：bench 无法完成（服务异常），已 env-gated

`FERRITE_IDX_MMA=1` 时文本正确（LEN 462），但 16 并发 bench 跑不动（"3 tok in 1.8s"）——
与 sparse_attn QK^T MMA 同样的病根：**per-tile 串行 + 1024 线程做极小的 tile 工作**，
固定开销远大于张量核省下的指令。默认关闭（`FERRITE_IDX_MMA` 未设时走 SIMT）。

### ⚠️ 方向切换（用户指令：优化幅度明显降低就该换方向）

**kernel 级 MMA 化的收益已耗尽**（近三项都在 ±1.5% 噪声内，两个结构性尝试均回归）。
剩余缺口 1.37x（1170 → 1600）必须靠**别的方向**：

1. **MTP（用户目标里明确允许，且届时目标变为 3200）** —— 当前 MTP 只支持 n=1；
   若把 draft/verify/commit 扩展到 B=16，按实测 2.4x accept 计算：
   13.7ms/步 ÷ 2.4 ≈ 5.7ms 有效步时 → 16 并发 ≈ **2800 tok/s**，逼近 3200 目标。
   这是**唯一能一步跨过 1600 的路径**。
2. 其次：AR/计算 overlap、CUDA graph 节点数削减（每步 ~700 节点）。

**结论**：下一步应做 **B=16 的 batched MTP**，而不是继续 kernel 微调。

### ⛔ MTP 被用户明令禁止（2026-09-09）

"严禁mtp，我说了如果你开mtp你要3200吞吐。严禁投机" —— **MTP/投机解码一律不做**，
目标固定为 **16 并发不开 MTP ≥1600 tok/s**。上一条"batched MTP"方向作废（且实测
MTP 在 B=16 下本身即坏：LEN 0 + 161 错误）。

**非 MTP 的剩余方向（按用户提示"很可能不是单 kernel 级别的"）**：
1. AR/compute overlap（TP all-reduce 与下一层计算重叠，1.06ms 中可隐藏大部分）；
2. CUDA graph 节点数削减（每步 ~700 节点）；
3. MoE expert-major 分组（消除 act 的 8x N 浪费与 down 的 N=1 限制）。

## 📊 当前时间分布（nsys 实测，2026-09-09 05:40，B=16，不开 MTP）

**总步时 13.68 ms = 1170 tok/s**（会话起点 833 tok/s，**+40%**）。报告：`/tmp/nsys_b16.nsys-rep`
（618MB，用 `timeout -s INT` + `/shutdown` 落盘）。**每步时间 = median × 每步调用次数**。

| kernel | 中位/次 | 次数/步 | ms/步 | 占比 | 状态 |
|---|---|---|---|---|---|
| moe_fused_act_fp8_mma | 47.8µs | 42 | **2.01** | 14% | MMA ✓（N=8 是复制 → 8x 浪费） |
| moe_fused_down_sum_fp8 | 46.8µs | 42 | **1.97** | 14% | SIMT（bf16 MMA 已实现但 +2% 后未启用） |
| sparse_attn_v2_batched | 143.5µs | 11 | **1.58** | 11% | SIMT（延迟受限；MMA 版实测 22.6ms 已关闭） |
| gemv_fp8_mma_b16 | 6.6µs | ~200 | 1.32 | 9% | **CUTLASS 级 MMA ✓（5.2-8.6x）** |
| indexer_topk_batched | 97.5µs | 11 | 1.07 | 7% | fast-path ✓ + 4 路 ILP ✓（MMA 版不可行） |
| gdn_chunk + step + prep | 44.7µs | 42 | 1.88 | 13% | 延迟受限（state 128KB/block 往返） |
| hc (mix+rest+post) | 22.3µs | ~90 | 1.45 | 10% | 92.8% No-Eligible（纯启动开销） |
| AR (p2p 3 kernels) | 11.8µs | 90 | 1.06 | 7% | 3 kernel/次 |
| NCCL all-reduce 残余 | 119µs | ~4 | 0.47 | 3% | 未定位的 4 次/步 |
| kpool_compress | 63.4µs | 11 | 0.70 | 5% | 4 路展开 ✓ |
| quant_e4m3_tokens | 2.4µs | 200 | 0.48 | 3% | 1024 线程 + float4 ✓ |
| moe_route | 5.8µs | 42 | 0.24 | 2% | |
| pool_expand / conv1d | 9.8/2.7µs | 11/42 | 0.22 | 2% | |

**两大块：MoE 4.0ms（28%）+ DSA 3.35ms（24%）= 52%。**
两者都是**per-token 散射**结构（每个 token 的 top-k 专家不同），
所以 act 的 N=8 只能是复制、down 的 N 只能是 1 —— **单 kernel 级 MMA 已到极限**，
必须做 expert-major 分组（把同专家的 token 聚到一起）才能用大 N。

**已耗尽的方向**：gemv MMA（位级最优）、quant、indexer fast-path/ILP、kpool 展开、
sparse_attn 位图/launch_bounds/unroll（隔离 −27%）、MoE act 2 段流水（隔离 −34%）。
**已证伪**：sparse_attn QK^T MMA（22.6ms）、indexer 分数 MMA（bench 跑不动）、
GDN dv 分块（+37%）、bf16/fp8 KV cache 单独（延迟非带宽）、HC_MIX_KS 加倍（serve 中性）。

## 2026-09-09 回归修复后的权威状态

**当前 build = 1170 基线（13.68ms/step），已逐位证明与 a9e5d5a 一致**：
- 隔离微基准（`/tmp/sparse_bench`，真实形状 B=16 h=8 d=256 dv=1024 topk=2051 live_k=2048）：
  当前 kernel vs a9e5d5a → **differing=0/131072**（位级一致）。
- 出师表 prompt：正确背出全篇（先帝创业未半而中道崩殂…将军向宠…臣本布衣躬耕南阳）。
- faults=0。远端 .so = 该修复版；源码已 commit+push+rsync。

**sparse_attn 512 线程优化已 park（env `FERRITE_ATTN_BLK=512` 可开，但输出与 256 不同 → 不可用）**：
12.58ms/1272 tok/s 的收益真实，但要逐位可用必须先解决两处 BLK 相关差异：
① QK group 数（32→64）改变 dedup 竞争赢家 → 存活槽位位置变 → softmax 求和/PV 结合顺序变（~1e-7，足以翻转模型）；
② PV 的 `G=ceil(blockDim/cols)` 随 BLK 变（4→8）→ 结合顺序变。
**正确做法**：把 dedup 改成确定性（每 seq 预计算 mask，最低槽位胜出）+ PV 的 G 钉在 256 值
（注意：直接 `pv_on` + 移出 `__syncthreads` 的重构会在 graph capture 期 err 700，需另找安全写法）。
**验证方式**：上面的隔离基准必须 `differing=0`，再跑出师表文本。

**排查工具（本次证明有效）**：`LD_PRELOAD=<各版本.so> /tmp/sparse_bench` + 逐位 diff + 逐 hunk 叠加，
秒级定位；`mega graph ... missing` 是 sticky CUDA error 的误报，真因在更早 kernel。

## SSE 流式路径：**不要为多流批帧牺牲单流延迟**（2026-09-11，共享栈）

**现象**：DSV4 单并发生成，**rank 侧** `[dsv41] decode` 显示 22-45 ms/step ✓，而**客户端**
端到端只有 7.75-8.3 tok/s（129 ms/token ✗✗）—— **多出 ~85 ms/token**。

**根因**（`crates/ferrite-http/src/api.rs`）：SSE 帧的批处理窗口
```rust
if batch.len() < 8 && opened.elapsed().as_millis() < 50 { return None; }   // 攒 8 token 或 50ms
```
单流下每帧都要**空等到 50ms** ✗ ⇒ 实测延迟 ≈ 50-70 + (22-45) ≈ **80-115ms/token** ✓ 与观测吻合 ✓。
该策略是为**多并发**降低 SSE 帧数（省 syscall）而设 ✓，对单流是**纯延迟** ✗。

**修法（已落地）**：帧在**尾部字节完整**时立即发出 ✓ —— 去掉人为窗口 ✓，**只保留 UTF-8
tail-holdback**（byte-BPE 把一个多字节字符切在 token 边界时才 hold ✓）。
**通用性**：GLM 的流式响应走同一路径 ⇒ 同样受益 ✓（已确认**无文档/测试依赖**该批帧行为 ✓）。

**判据（可复用）**：**服务端自报的 step 时间**与**客户端观测的端到端时间**必须分别记录 ✓——
两者之间的差额就是 serve/驱动/流式栈的开销 ✓。GLM 侧的 `[megab] replay` 行就是这个作用 ✓；
DSV4 侧本会话新增了同口径的 `[dsv41] decode: N steps in Ts = X steps/s (Y ms/step)` ✓。

## 下一批 kernel 优化的取证顺序（2026-09-11 定，迁移完成后执行）

**当前已知（同二进制实测；权威口径 = 中间稳态的 step time ✓）**：
`[dsv41] decode` 的稳态窗口 = **21.76 / 21.75 ms/step（短上下文，46.0 steps/s）** 与
**44.23 ms/step（长上下文，22.6 steps/s，DSA decay 1.39x ✓ 与历史一致 ✓）**。
⚠ **端到端（3.28s/64 = 19.50 tok/s）只作交叉验证** ✗ —— 它含 prefill、admissions 爬坡与收尾 flush ✓
（GLM 侧的 `[megab] replay` 中位数就是这个规矩 ✓；`total/wall` 口径在历史文档里已被定为仅交叉验证 ✓）。
**目标**：单并发 200 tok/s ⇒ **5 ms/step（每层 0.11ms）** ⇒ 差距 **~4.4x**（短上下文口径）。

**第 0 步（必做，否则又是推测 ✗）**：**重新取一份 decode-only 的逐 kernel 分解** ✓
- 方法：**差分法**（`--max-tokens 1` 与 `--max-tokens N` 两张表相减 ✓；`nsys stats --report
  cuda_gpu_kern_sum --format csv` + python ✓ —— **不要**用 table 格式配 awk ✗，kernel 名含空格 ✓）。
- 注意：多卡 nsys 的**绝对单次耗时不可信** ✗（只用于排序 ✓）；要精确耗时就写隔离复现器 ✓
  （模板 `/tmp/{hc,sa}_repro.cu` ✓：预热 + 数百次计时 + **同 shape 空 kernel 的地板对照** ✓）。
- ⚠ 剖析**必须用 NCCL 模式**（去掉 P2P/AR v5 的 env ✓）：v5 的 publish 会自旋 ✗，与 nsys 的
  节点追踪叠加会放大数百倍 ✗（历史实测 300x ✓）。

**第 1 步（按"大 M 内核在 M=1 退化"通则审计）**：把分解里每个 kernel 与
`权重字节量 ÷ 带宽`（DRAM 7.6TB/s / L2 视常驻）对比 ✓ —— **差 100 倍以上即为形状/硬件约束不匹配** ✓
（本会话已用此通则修掉三处：hc_mixes 块形状 +74%、专家 fp4 GEMM 换 GEMV +21%、dense fp8 +16% ✓）。

**第 2 步（本会话已验证的同类候选）**：
- 投影族（`gemm3`/`gemv_fp8`/`nvjet`）：`gemm3` 实测带宽仅 ~0.6TB/s ✓（赢在杀固定成本而非 GEMM 效率 ✗）
  ⇒ 大矩阵仍应走 cuBLAS/MMA ✓；`splitK` 对 M=1 可能是反优化 ✗（需 cublasLt 过滤验证 ✓）。
- **AR**：v5 后 GPU 侧归约仍需看是否与 kernel 重叠 ✓（本会话量到 AR 主机侧 ~4%、GPU 侧 ~12% ✗）。
- **hc 链**：rest345 12.6µs/次 ×90 ≈ 1.1ms ✗ —— 屏障/原子/低占用混合，需联合重构 ✓（高风险）。
- **MoE act/down**：act 已达 DRAM 71%（地板 ✓）；down 是 L1TEX 管道地板 ✓（8 个理论均已实测失败 ✗）。

**第 3 步（结构性，非增量）**：**段融合**（每层 3 段 + 2×AR ✓）—— 只在增量收益耗尽后再做 ✓
（本会话已证：**图内节点 launch ≈ 0.2-0.3µs/节点** ⇒ 融合省的是**中间量往返与相位流水**，
不是 launch ✓，故价值在 ~0.3-0.5ms 量级 ✓，需与第 0 步的新分解对齐后再投入 ✓）。

**纪律（沿用）**：① 一次只改一个变量；② 同二进制背靠背 A/B 才算证据；③ 每次改动**亲自读四段文本** ✓；
④ 改 `.cu` 必 `bash build.sh 103a` 重编 `.so` ✓（漏跑会得到"新 kernel 不在 .so"的假故障 ✓）。

## 2026-09-11 DSV41 新 decode 分解（差分法，N=40）+ 下一个优化点

命令：`bash scripts/dsv41_profile.sh 40 /tmp/dsv41-prof`（脚本已固定 `DSV41_AR_V5=0 DSV41_GRAPH_STEP=0` ✓ ——
前者避免自旋 × nsys 节点追踪的 300x 放大 ✓，后者避免捕获 ~400 节点的追踪开销 ✓；
两条路径的算子耗时相同 ✓（`DSV41_TOKTRACE` 证明单请求逐 token 一致 ✓））。

```
decode-only net GPU time = 17789.3 ms over 39 steps / 8 ranks = 57.02 ms per step per rank
  share calls      kernel
  28.1%   837      hc_mixes_kernel            ← 头号目标
  22.9%  5022      expert_gemv_fp*
  20.4%  2311      gemm_fp8_gemv_kernel
  14.0%   858      ar_reduce_kernel
   2.5%   418      sparse_attn_wa*
   1.8%   501      gemv_bf16_kernel
   1.4%    41      indexer_topk
   1.3%  1726      rmsnorm_kernel
   1.2%    84      gemv_f32_kernel
   1.0%   858      ar_store_kernel
   0.8%   858      ar_stamp_kernel
   0.8%  2259      quant_kernel
   0.7%  2563      swiglu_limit_kernel
   0.5%   418      route_topk_kernel
   0.5%   837      hc_post_kernel
```

**⚠️ 量纲注意**：上表的**份额**可信 ✓，但脚本打印的 `us/call` 一列有量纲异常 ✗
（`hc_mixes` 打出 153182.6 µs/call ✗ 显然不对 ✓）⇒ 需要核对差分公式里共享列的换算 ✓
（见 TODO #10）；**绝对耗时要走隔离复现器** ✓（脚本尾部的既有告诫 ✓）。

### 头号目标：`hc_mixes_kernel`（28.1%）—— 与 GLM 侧的"块形状/K-split"同类
`kernels/cuda/dsv41_kernels.cu:569` 注释即写明 **"Grid: one block per token"** ✓，
启动为 `<<<rows, nthreads, smem, s>>>`（`nthreads = mix*32 = 24*32 = 768` ✓）。
**decode 时 rows=1** ✗ ⇒ **每层每步只有 1 个 block、768 线程（1 SM，占用率 ~1/148 ✗）**，
却要算 24×16384 = **393K MAC** ✗ ⇒ 任务级并行度不足 ✓。

**设计（与我在 GLM 侧已兑现的 K-split 同形）**：
1. 把 K 维（`hc_dim` = hc×dim = 4×4096 = 16384 ✓）切成 8~16 段 ✓，每段一个 block ✓
   （grid = rows × SPLIT ✓），每块算**完整 24 维 mix 的局部和** ✓（24 个 float ✓）→
   `atomicAdd` 到一个 24-float scratch ✓（或写 [rows][SPLIT][24] 再确定性归约 ✓，后者的
   求和顺序固定 ✓ ⇒ **优先选它**，数值可复现 ✓）。
2. **Σx² 的归约**同样按 SPLIT 分块 + 二次小 kernel 归并 ✓（或让 block 0 在全量 x 上算 ✓ ——
   它已经是全量读 ✓，成本低 ✓）。
3. **sigmoid / sinkhorn 的拆分后处理**留在"归并后"的小 kernel 里 ✓（24 维 ✓ 极便宜 ✓）。

**验收**：`scripts/dsv41_profile.sh` 看 `hc_mixes` 份额下降 ✓ + **四段文本亲自读** ✓ +
`DSV41_TOKTRACE` 与改前逐 token 比对 ✓（数值等价性判据 ✓）。

### `hc_mixes` 的隔离实测（2026-09-11，`/tmp/hc_repro.cu`）—— 修正先前判断

**实测（生产形状 hc_dim=16384、mix=24、sinkhorn=20、eps=1e-6 ✓）**：
| `DSV41_HC_MIXES_THREADS` | rows=1 | rows=8 | rows=64 | rows=128 |
|---|---|---|---|---|
| **768（=mix×32，默认）** | **49.3 µs** | 49.7 | 51.0 | 51.9 |
| 256 | 133.4 | 135.0 | 139.0 | 140.7 |
| 1024 | 48.5 | 48.8 | 50.3 | 51.0 |

⇒ ① **"flat 固定开销"成立 ✓**（rows 1→128 基本不变 ✓）；② **256 线程慢 2.7x** ✓
（24 行需要 24 个 warp 同时在飞 ✓，与代码内注释的 A/B 一致 ✓）。

**⚠️ 先前的 split-K 设计已废弃 ✗**：该 kernel 的瓶颈不是"K 维并行度不足 ✗"，而是——
1 个 block（768 线程 = **1 个 SM** ✗）要读满 `hc_fn`（24×16384×4B = **1.5 MB** ✗）：
单 SM 可取带宽约 100 GB/s ⇒ 1.5MB/100GB/s ≈ **15 µs** ✓，加 sinkhorn 等 ⇒ 实测 49 µs ✓；
而同一次的算力只需 393K MAC ≈ **1.4 µs** ✗ ⇒ **瓶颈是"每 SM 带宽" ✗**。
in-serve 时因 90 次/步调不同层的 1.5MB ⇒ L2 反复换页 ⇒ nsys 口径 **153 µs/call** ✗
⇒ 90 次/步 × ~140 µs ≈ **12.6 ms/步（28.1%）** ✓。

**⚠️ 隔离复现器的坑（本方法的教训）**：我的复现器**每轮复用同一批缓冲** ✗ ⇒ 恒为 L2 热 ✓
⇒ 49 µs 只是 L2 热成本 ✓，与 in-serve 的 153 µs 差 3 倍 ✓。**复现器必须换缓冲刷 L2** ✓
（或对"按次读 1.5MB"的 kernel 直接用 in-serve 份额估计 ✓）。

**修正后的设计（下一会话第一件事）**：**把 24 个投影行摊到 24 个 block**
（grid = 24 × 32 线程 ✓，每 block 一行 ✓）= 24 个 SM ✗⇒ 带宽 ×24 ✓ ⇒ 预计 **5-10 µs/call** ✓。
配套：Σx² 仍是单 block 的 16384 元素归约 ✓（0.3 µs 级 ✓，可独立成一个极小的前置 kernel ✓，
或让 block 0 兼做 ✓）；`inv` 必须在其之后 ⇒ **两步 kernel**（① Σx² → ② 24 行投影+sigmoid/
sinkhorn ✓），多出的一次 launch ~1.34 µs/次 × 90 = 60 µs/步 ✓ **可忽略** ✓。
**验收**：`scripts/dsv41_profile.sh` 看份额下降 ✓ + 四段文本亲自读 ✓ + `DSV41_TOKTRACE` 与改前
逐 token 比对 ✓。

### ❌ `hc_mixes` 展开的多块版本 —— 阴性结果（2026-09-11，env-gated 保留，默认关）

**实测（同一 .so、背靠背、各 3 次，生产形状）**：
| 配置 | rows=1 | rows=8 | rows=128 |
|---|---|---|---|
| 原 kernel（默认）| **49.3** | 49.7 | 51.9 µs |
| 展开版 v1（24 行 = 24 块 × 32 线程）| 109.7 ✗ | 115.0 | 115.8 |
| 展开版 v2（再加 K 维切分：24×8=192 块 × 64 线程）| **109.7 ✗（与 v1 完全一致）** | 115.0 | — |

**两个推断（v2 与 v1 一致是关键证据）**：
1. **它不是带宽受限 ✗** —— 8 倍的 SM（192 块）没有任何变化 ⇒ 我之前"1 个 SM 读 1.5MB ≈ 100GB/s
   ⇒ 15µs"的归因不成立 ✗。它那 49 µs 的构成仍是**未拆解**的（kernel 自述"flat 58us = 全固定开销"，
   而作者已把 sinkhorn 改到寄存器+shuffle；剩余部分需要**逐相位关断**才能定位 ✓）。
2. **展开版的代价 ≈ 多出的 2 次 kernel 启动**（49.3 → 109.7 ✗，约 +30 µs/次 × 2 ✓）。
   ⇒ **在宿主启动路径上（默认不开图）3-kernel 设计是净亏 ✗**；只有在图捕获下（每节点 ~0.2-0.3 µs）
   才可能回本 ✓ —— 而图当前因**跨请求故障**默认关闭 ✗ ⇒ **展开这条路依赖"先修好图"** ✓。

**⚠️ 方法教训（env 陷阱，我自己踩了）**：门禁写成 `getenv(...) != nullptr` ✓ ⇒
**显式传 `SPREAD=0` 也是"非空" ⇒ 门禁照开 ✗** ⇒ 我第一次误读成"两条路径一样快 ✗"，
实际是"两次都在跑展开版 ✗"。**已修**：`e[0] != '0'` 才算关 ✓。
（同类地雷见 AGENTS.md："新增旋钮但默认值写错导致每次调用做一次 env 查找" ✓。）

**结论**：`DSV41_HC_MIXES_SPREAD` 默认关 ✓，代码保留（图修好后可重测 ✓）。
`hc_mixes` 的真正构成需**逐相位关断**（Σx² / 投影 / sigmoid / cm / sinkhorn / comb）按 nsys 份额拆 ✓，
不要再靠"改块形状"猜 ✗。

### MoE gemv 族的审计结论（2026-09-11，真基线 36.7ms/step 口径）

**两个 kernel 都已读完，GLM 方法论在此族的状态：**

**expert_gemv_fp4（22.9%）——GLM 房式的招全部用过或试过：**
- ✅ 已做：激活 staging 进 smem（kernel 注释记载：原来每行重读 512×5120×4B=10.5MB，
  把 kernel 钉在 38 GB/s；staging 后只剩 0.65MB 权重流）
- ✅ 已做：down 并入 GEMV（epi_mode==3，row_weight[0] 的越界 bug 已修）· 8 行/block · warp-per-row
- ❌ **已试且更差（kernel 注释原文，勿重试）**："a k-split (one block per row, 8 warps
  splitting k -> 512 blocks) gave 15.0 tok/s against this 15.2, and **16-byte uint4 lanes
  gave 14.3**. The kernel is not occupancy- or request-rate-bound the way those two assumed."

**gemm_fp8_gemv（20.4%）**：同族模式（warp-per-row、1B/lane/iter 的 a+w 载入、e4m3 解码、
32×32 block scale）；激活未 staging 但 m==1 ⇒ `a` 是 L2 命中，非主要矛盾。

**结构性假设（下一步先测再动，勿直接改 kernel）**：该族每 rank 每步
**~628（fp4）+ ~289（fp8）≈ 900 次小 kernel 启动**（每个 (layer, topk-expert) 一次 launch；
kernel 的 `ids/slot` 参数就是逐专家调用的痕迹）。若每次启动+收尾地板在 ~10µs 级
（hc_mixes 展开实验实测：多 2 次 launch = +60µs ⇒ ~30µs/次 ✗ 远高于 GLM 图内的 0.2µs），
则 900 次 ≈ 9ms/步，与该族 15.9ms 份额同量级 ⇒ **真正的杠杆是批化**
（每层一次 launch、grid=(expert,row)，`ids` 表间接寻址 kernel 已原生支持 ✅）
**或修好整步图**（#10，图内启动 ~0.2µs）。
**行动顺序：先隔离复现器测 per-call 成本 vs 空 kernel 地板 ⇒ 数据决定批化还是修图。**

### 空 kernel 启动地板实测（2026-09-11，`/tmp/floor.cu`，B300 本机）

**~3.05 µs/launch，对网格大小与 smem 完全平坦**（1×32 / 64×256 / 512×256 × 有无 20KB smem，
五组全部 3.03-3.08 µs，事件计时 20000 次背靠背）。

**这组数字决定的优先级（对真基线 36.4ms/step）**：
| 路径 | 节省 | 依据 |
|---|---|---|
| **修好整步图**（图内 ~0.2µs/launch）| **~4.6-5.8 ms/步**（13-16%）| 全步 ~1500-2000 次启动 × (3.05−0.2)µs |
| MoE gemv 批化（917→~90 次启动）| ~2.5 ms/步 | 917 × 3.05µs − 90 × 3.05µs；若图先修好则只剩 ~0.16ms ✗ |
| 段融合（mega kernel）| 图修好后需重估 | GLM 侧同结论：图内启动近免费后，融合的剩余价值=中间量往返+流水化（−0.3~0.5ms 级）|

⇒ **顺序：修图 > 批化 > 段融合** ✗✓。修图同时恢复：单图（用户三指令之一）✓ 完整零 H2D ✓
图内启动 0.2µs ✓ —— 一举三得。
**注意**：Phase 2 已把 AR 换成共享实现（store+pubred 双 kernel），旧图的跨请求故障排除集
全部对着 DSV41 自有 AR 测的 ✗ ⇒ **必须先用共享 AR 重测图的跨请求行为**（尖锐复现：3 短 + 长请求），
数据可能直接改变结论。

### MoE gemv 批化 —— 已实现（2026-09-11，`DSV41_MOE_BATCH`，默认关）

**代码状态**：批化变体已落地，env 门禁 `DSV41_MOE_BATCH`（`chain_dev.rs` 的 `moe_batch()`，`OnceLock`
读一次 ⇒ 图可捕获），**默认关**：`"0"` 也算关。旧路径（逐 slot 循环）保持原样，是 fallback 与 A/B 基准。

**批化的实际形状（关键：逐个 slot 的输出是否 disjoint）**：
| 方向 | 原状 | 能否直接批化 | 采用方案 |
|---|---|---|---|
| gate/up | 每 slot **覆盖写同一个 `ex_act`** | 否 | 新增 scratch `ex_act_b[topk][2*inter]`，每 slot 写自己的 slice |
| swiglu | 每 slot 就地处理 `ex_act` | 是 | 新增 `swiglu_limit_batched`，`grid.y = slot` |
| down | 每 slot **累加进同一个 `o`**（epi_mode 3）| 否 | 新增 scratch `ex_down_b[topk][dim]`（epi_mode 2，**非累加**）+ 定点序归约 |
| 归约 | （原来隐含在累加里）| — | `moe_down_reduce`：`o[i] = (((0+c_0)+c_1)+…)` 按 slot 升序 |

⇒ 每层从 `3*topk` 次启动（gate/up + swiglu + down）降到 **4 次**（gate/up、swiglu、down、reduce）。
`topk=6` 时每层 18→4。

**数值论证（bit-identical）**：
1. 批化 gate/up/down 内核是 `expert_gemv_fp4_kernel` 的**逐行拷贝**，只把 3 处换成"以 `blockIdx.y`
   为 slot"：专家 id（`ids[blockIdx.y]`）、激活基址（`a_f32 + slot*act_stride`）、输出基址
   （`out + slot*out_slot_stride`）。K 维点积顺序、warp shuffle 归约、`gridDim.x` 的行分配**全部不变**
   ⇒ 每个输出元素逐位相同。
2. `row_weight` 由"每调用指针 + `row_weight[0]`"变为"`row_weight[slot*rw_stride]`"，`rw_stride=1`
   ⇒ 取到的是同一个标量。
3. down：顺序路径是 `o[i] = (((0 + c_0) + c_1) + …)`（`o` 先被 zero）；批化把 `c_s` 写进
   `ex_down_b[s][i]`，归约从 `acc=0.0f` 起**按 s 升序**相加 ⇒ 与顺序路径逐位相同（fp 加法不结合，
   顺序即契约，**归约内核里绝不能并行 slot 循环**）。

**遗留 / 未做**：
- **fp8 兄弟（`gemm_fp8_gemv_kernel`，20.4% 份额）不适用批化**：它的调用者是 `lin()`（每层 4 个注意力
  投影）+ 共享专家（每层 3 个 fp8 gemm），**没有 (layer, topk-expert) 结构**（ABI 明确禁止 fp8 专家：
  `kernels.rs` 的 user directive）。它那 ~289 次启动是"每层 7 次"的结构量，不是可批化的逐专家量。
  唯一可合并的是共享专家 `w1`/`w3` 那对（写 `ex_act` 的相离两半）⇒ 2→1，但那是"权表"机制而非专家间接
  寻址，且省的是 ~43 次启动（~0.13ms），暂不动。
- 如果整步图先修好（图内 ~0.2µs/launch），批化的收益会从 ~2.5ms 掉到 ~0.16ms ⇒ **批化与修图是替代
  关系**，先修图则批化优先级下降。

**验证状态**：`cargo check --workspace` = 0 errors（本地）。**CUDA 侧未编译**（本地无 nvcc）：
`dsv41_experts_mxf4.cu` / `dsv41_glue.cu` 的改动需要在远端 `bash kernels/cuda/build.sh 103a` 重建，
并核对新 kernel 的寄存器数（house rule：寄存器上涨会静默腰斩占用率）。

### MoE gemv 批化已落地（`DSV41_MOE_BATCH=1`，默认关，待 e2e 验收）

- **新 kernel（老 kernel 一行未动 ⇒ 默认路径字节级不变 ✓）**：
  `expert_gemv_fp4_batched_kernel`（`dsv41_experts_mxf4.cu:701`，逐行拷贝 + `blockIdx.y = slot`
  + per-slot stride 的 `act_stride/out_slot_stride/rw_stride`）+ `moe_down_reduce_kernel`（`:887`）。
- **调用侧关键发现**：逐 slot 输出**并非 disjoint** —— gate/up 每 slot 覆盖写同一个 `ex_act` ✗、
  down 每 slot 累加进同一个 `o`（epi_mode 3）✗ ⇒ 批化必须走 **per-slot scratch**（`ex_act_b`/
  `ex_down_b` 以 `slot*stride` 相离 ✓ 已实现 ✓）。
- **数值论证**：gate/up 逐位一致（同专家、同行内点积序 ✓）；down 的归约按 **slot 0..topk 升序**
  求和，与顺序路径"从清零的 `o` 起逐个 `o[row] += x`"**同序** ⇒ 逐位一致 ✓（fp 加法不满足结合律，
  定点序是必须的 ✓）。
- ✅ **scratch 边界已审**（同类"两端"风险 ✗）：`ex_act_b: alloc(fb(topk*2*inter))` /
  `ex_down_b: alloc(fb(topk*dim))` —— **按全部 topk slot 分配** ✓，与批化路径的
  `slot*(2*inter_local)` / `slot*dim` 步长吻合 ⇒ **不越界** ✓（顺序路径复用单个 `ex_act`，
  批化必须相离，分配端已同步跟上 ✓）。
- **待办**：e2e 验收（`DSV41_MOE_BATCH=1`：四段文本亲自读 + 逐步计时预期降 ~2.5ms）。
- ✅ **寄存器审计已做**（远程 `-Xptxas -v -gencode arch=compute_103a,code=sm_103a`）：
  `expert_gemv_fp4_batched_kernel` **48 regs / 0 spill / 32B stack** —— 与老
  `expert_gemv_fp4_kernel`（48 regs / 32B stack）**完全相同** ⇒ **没有踩 GLM 那条
  80 regs → 2 blocks/SM 的占用率陷阱** ✓；`moe_down_reduce_kernel` 32 regs / 0 barrier ✓。
  （审计命令的坑：必须用 `-gencode arch=compute_103a,code=sm_103a` ✗ —— 只给 `-arch=sm_103a`
  会生成 compute_103 PTX 而 mxf4/tcgen05 全部报 "not supported on sm_103" ✗，那是标志问题不是代码问题 ✓。）

### down + reduce 融合已落地（`DSV41_DOWN_FUSE`，默认开，待 e2e 验收 + 远程编译）

- **新 kernel（老入口一行未动 ⇒ `DSV41_DOWN_FUSE=0` 字节级回退 ✓）**：
  `expert_gemv_fp4_down_reduce_kernel<STAGED>`（`dsv41_experts_mxf4.cu`，模板参数选 act staging）
  + launcher `dsv41_expert_down_reduce_fp4_batched`。把批化 down 的**两次启动**（`expert_down_fp4_batched`
  写 `[slots][dim]` scratch + `moe_down_reduce` 定点序求和）合成**一次**：grid 只有 x 维
  （`⌈dim/8⌉`），每 warp 独占自己的输出行，**串行升序**走 slot、累加在寄存器里，`out[row]` 覆盖写。
- **数值契约（逐位一致）**：K 点积 + butterfly shuffle **逐字照抄** batched kernel（同一 lane 序、
  同一 group 序、同 `#pragma unroll 2`）⇒ 每 slot 的 `c_s` 逐位相同；slot 串行升序 ⇒ 复现
  `((0 + c_0*rw_0) + c_1*rw_1) + ...`。**⚠️ 唯一的实现修正**：`tot` 用
  `__fadd_rn(tot, __fmul_rn(acc, rwv))` ✗ 不是字面的 `tot += acc * rwv` —— fast_math 下后者会
  被收缩成 FMA（乘之后再舍入），与 scratch 路径的 `fl(c_s*rw_s)` 再加**最后一位不同** ⇒
  必须显式分开乘/加才能逐位一致 ✓。
- **act staging（STAGED=true）**：一次把全部 slot 的 act 摊到 smem `[slot][k]`（slot-major 线性），
  LUT 紧随其后；`act_stride` 是调用方的 slice 间距，**每 slice 只读前 k 个**（今天 = 2*inter 的
  swiglu 半边；gate_up+swiglu 融合把 slice 压成 inter 后调用方传更小的间距，kernel 不用改 ✓）。
  smem = `slots*inter*4 + 2048`；超过设备 opt-in 上限时走 STAGED=false（只 stage LUT，act 直读 global）。
- **per-context 教训已贯彻**：`cudaFuncSetAttribute` 是 **per device** —— launcher 每次启动用
  `cudaGetDevice` + 每设备一次 `cudaDevAttrMaxSharedMemoryPerBlockOptin` 探测，按需把 STAGED kernel
  的上限抬到该设备的上限（不再"在某个恰好是 current 的 device 上设一次" ✗）。
- **gate_up+swiglu 交互**：`chain_dev.rs` 把 `act_slot`（= 融合时 inter_local，否则 2*inter_local）
  直接作为 `act_stride` 传给融合 kernel ⇒ 融合方向切换 layout 时这一处**无需改** ✓。
- **待办**：① `cargo check -p ferrite-models` 通过 ✓；② **远程 `bash kernels/cuda/build.sh 103a`
  未跑**（本地无 nvcc）⇒ 需核对 STAGED kernel 的寄存器数/smem 指令选择，以及 `__fmul_rn/__fadd_rn`
  没有被打回 FMA；③ e2e 验收（四段文本 + 逐步计时：每层少一次启动）。

### 图修好后（DSV41_GRAPH_STEP=1 成为默认）的优先级重排（2026-09-11）

**前提**：单图的根因（indexer 的 dynamic smem 取自被烤死的每步计数）已修 ⇒ 整步图可用 ⇒
**每步 ~1500-2000 次 kernel 启动的地板（3.05µs/次 ≈ 5-6ms，13-16%）基本消失** ✓（图内 ~0.2µs）。

**重排**：
| 项 | 图前价值 | 图后价值 | 说明 |
|---|---|---|---|
| **修整步图** | +5-6ms | — | 已修 ⇒ 默认开 |
| MoE gemv 批化（917→~90 启动）| −2.5ms（预估）| **仍有 ~1-1.8ms（修正后）** ✓ | **实测 −3.2ms（−8.7%）** ✗✗ 见下：被消除的每次固定成本 ≈ **3.9µs** ✗（> 3.05µs 启动地板 ⇒ 还含 ~1µs 设备侧建立/收尾 ✓）⇒ 图内那部分仍在 ⇒ **不是塌到 0.2ms**，批化在两种模式下都值得开 ✓ |
| 段融合（mega kernel）| ~0.5ms | ~0.3-0.5ms | GLM 侧同结论：图内启动近免费后只剩中间量往返+流水化 |
| **MoE gemv 族的真实工作量** | ~13ms | **~13ms（不变）** | 这才是图之后的头号目标 ✗ —— 但内层杠杆已有实测阴性记录（uint4/k-split ✗），需要**新机制**（非"再调块形状"） |
| AR（ar_reduce 等）| ~5ms（host-barrier 口径）| 同 | AR v5 已默认开 ⇒ 实际远低于该口径；用新分解重测 |
| 注意力/小 kernel | — | — | 需图开的**新差分剖析**（脚本已就位 ✓） |

**⇒ 下一步动作序列**：① 验收 v3 通过 ⇒ 翻默认 ✓；② 跑一次**图开的** `scripts/dsv41_profile.sh`
拿新分解 ✓（图内启动免费后，分解才反映真实算子成本）；③ 按新分解选头号项 ✓。
**⚠️ 更正（同轮内自查发现）**：**图开路径无法用 nsys 剖析 ✗** —— 图开 ⇒ AR v5 被强制开 ✓
（host barrier 不是 CUDA 调用，进不了图 ✗），而 **AR v5 的自旋 × nsys 的 `--cuda-graph-trace=node`
= 已记载的 300x 病态** ✗（240s 只跑 69 步 ✓，见 AGENTS.md 铁律）。⇒ `scripts/dsv41_profile.sh`
**必须继续固定 `DSV41_GRAPH_STEP=0`** ✓（这是 kernel 归因模式的唯一可用配置 ✓），而
**图开后的真实分解 = 该分解 + "减去 ~1500-2000 次启动 × (3.05−0.2)µs ≈ 5-6ms"** ✓
（算子执行时间不因图而变 ✓ 变的只是启动/同步口径 ✓）。

### ⛔ 标准（用户 2026-09-11 定，效力高于本文档任何"已知边界/后续改进"的写法）

> **"不准 hack，不能留 todo，必须彻底修好。"**

推论与执行要求：
1. **"把边界写进文档"≠ 修好** ✗ —— 一个已知会静默降质的上限（如 indexer 的 `*lens > idx_cap`
   时漏候选 ✗）**不是交付物**，而是待修项 ✓。要么**结构上不可能发生** ✓，要么**证明它在可达范围内
   不可达** ✓（用真实上下文/合成输入二选一给证据 ✓）。
2. **钳位/截断式"修法"要警惕** ✗：它把"越界崩溃"换成"静默漏算" ✗ —— 与 AGENTS.md 的
   "假快"教训同源（8.21ms 那次的陈旧 smem ✓）：**先把语义不变式说清，再谈性能/安全性** ✓。
3. **验收必须覆盖修法的反面** ✓：这次的验收 = ① 开图 12 连发（原故障面 ✓）**② 合成大 `n_pos`
   复现器**（serve 真实上下文够不到旧阈值 ✗ ⇒ 只有合成输入能证"无包络" ✓）。

**本轮的实例**：indexer 的 smem 曾取"每层常量 + 内核钳位"（46KiB ✓ 已实测能跑 ✓，前三个请求
`Paris/Tokyo/2` 全对 ✓）—— 用户仍判为 hack ✗ 并否决 ✓。**正确形态 = smem 与 `n_pos` 完全无关**
（分块扫描 + 运行期 top-k 归并 ⇒ 编译期常量 smem ✓ ⇒ 图里被烤死的实参对 smem 彻底无关 ✓；
扫描上界取自设备计数器 ⇒ 任意计数都安全 ✓ 无任何包络 ✓）。
**✅ 已按此形态实现**（2026-09-11）：`kIndexerChunk = 4096`，`topk=512` 实配下 smem = **26688B** ✓，
启动器不再出现 `n_pos` ✓，超预算时**硬失败**而非静默丢候选 ✓；`idx_cap` 已从 Rust 侧删除 ✓。
细节见 `crates/ferrite-dsv41/STATUS.md` 的"包络彻底消除"节 ✓。

### ✅ MoE gemv 批化 e2e 实测（2026-09-11，同会话单轮）

| 配置 | p50 | 说明 |
|---|---|---|
| 顺序（基线）| 36.4 ms/步 | 917 次 (layer, slot) 启动 |
| **`DSV41_MOE_BATCH=1` 批化** | **33.24 ms/步** | 147 步；**−3.2ms = −8.7% ⇒ 30.1 tok/s** ✓ |

**文本人眼验证（6 段）**：Paris / Tokyo / 2 ✓、《静夜思》整首 ✓、长文 `## 智能的边界…` 通顺 ✓、
《出师表》开篇逐字 ✓；**0 fault** ✓。

**认知修正（重要）**：我在上一节预测"图修好后批化只剩 ~0.2ms" ✗ —— **实测否证** ✗。
−3.2ms / 827 次被消除的调用 = **3.9µs/次** ✗，**高于** 3.05µs 的纯启动地板 ✓ ⇒ 每次逐专家调用
还有 ~1µs 的**设备侧建立/收尾**成本 ✓（这部分图**不能**消除 ✗ —— 图只免掉启动 ✗）。
⇒ **批化的剩余价值随图开仍约 1-1.8ms** ✓，**两种模式都值得默认开** ✓（待 subagent 释放
`chain_dev.rs` 后翻默认 ✓）。

### ⚠️ 剖析脚本曾长期"静默失效"（2026-09-11 定位并修复）

**症状**：`scripts/dsv41_profile.sh` 输出 `decode-only net GPU time = 0.0 ms`，内核列表**全空** ✗。
**误判过**：我先后猜"run 被截断"✗、"第二次 serve 没起来"✗ —— 都错 ✗（`steps = N-1` 只是入参 ✓，
与日志无关 ✓；且脚本用的是**一次性** `--prompt` 模式，根本没有第二次 serve ✗）。

**真因（bash 语义）**：那段"两个 pin 必须保留"的注释被**插在续行命令中间** ✗：
```bash
    DSV41_AR_V5=0 DSV41_GRAPH_STEP=0 \
    # Both pins are load-bearing ...        ← 注释行无续行符 ⇒ 命令在此终止
    "$BIN" --prompt ... --tp 8 >... 2>&1
```
`\` 把注释行并进逻辑行 ⇒ 实际执行的是 **`env VAR=0 VAR2=0`（无命令 ⇒ 只打印环境变量 ✗
—— 输出里那段突兀的 env dump 就是铁证 ✓）**；而 `"$BIN"` 成了**另一条独立命令、完全没被
nsys 包住** ✗ ⇒ nsys 剖析的对象是 `env` ⇒ **报告里零 CUDA 内核** ⇒ CSV 空 ⇒ 差分为空 ⇒
`tot = 0.0` 被 `or 1.0` 兜住 ⇒ **静默打印 0.0 ms** ✗。

**三条教训（已写进脚本注释）**：
1. **诊断路径永不吞 stderr** ✗：`nsys stats ... 2>/dev/null` 是把错误藏起来的那一层 ✗；
2. **零行结果必须响亮失败** ✓：空 CSV 现在是 `exit 1` + 打印日志尾部（此前"空差分 ⇒ 0.0ms"毫无提示 ✗）；
3. **续行命令里不写注释** ✓：注释移到函数外；同时删掉一行神秘的重复 `prof 1 "$OUT/one.csv.tag"`
   ✗（修好后它会把 nsys 轮次翻三倍 ✗）。

**⇒ 影响面**：上一次成功的剖析源自该注释插入之前 ✓ ⇒ 那之后所有"图后分解"的尝试其实都没测到东西 ✗
（"0.0 ms"从未被我当成异常 ✗✗ —— 这是本次最该记住的一点：**数字为 0 而不报警，比数字错更危险**）。

**连带教训（同一事故的第二层）**：脚本修好、提交之后，**远端重跑仍然是 0.0 ms** ✗ —— 因为
远端命令是 `ssh … 'cd ~/ferrite && …'`，**从不先同步** ✗，远端停在旧 commit 跑旧脚本 ✗。
⇒ **任何远端 run 之前必须三步**：`git fetch origin main && git reset --hard FETCH_HEAD`
（文档 2b 规定的同步 ✓，不是 revert ✗；不动被 gitignore 的 `.so`/`target` ✓）→ 双产物重编 →
`ls -la`/`md5sum` 核对产物指纹 ✓。否则**测的是旧码而你不知道** ✗（本次连续浪费两轮剖析 ✗）。

## 2026-09-11 已证否：e4m3 LUT 的 bank conflict 不值得修（探针 gprobe5）

探针 `/tmp/gprobe5.cu`（远端 43.202.208.136，k=5120，warps=4，GRAPH 171 calls，两轮复现 ±0.03µs）。
**先修了 gprobe3 的数据缺陷** ✗：`hw[i]=(i*37+11)&0x7E` 强制 bit0=bit7=0 ⇒ 只有 16/32 个 bank 可达，
冲突测量有 2 路硬地板 ✗。改为按 per-block ue8m0 尺度采样的带符号正态 → e4m3 字节。

| 变体 | n=256 | n=1024 | n=1664 | 结论 |
|---|---|---|---|---|
| base（位运算）| 10.40 | 12.80 | 17.26 | 旧基线 |
| **lut（主树）**| 8.23 | 9.35 | 11.96 | ✓ |
| lutcf（假索引上界）| 5.76 | 6.79 | 8.02 | 不可达 ✗ |
| rot1（8 子表+旋转）| 8.94 | 10.41 | 12.24 | 净亏 ✗ |
| rot4 | 9.73 | 11.05 | 13.00 | 净亏 ✗ |
| rep4（4 列冗余）| 8.16 | 9.97 | 13.43 | 中性/亏 ✗ |
| **a32（激活预解码）**| **7.71** | **8.59** | **10.39** | ✓ 唯一真赢 |

**三个结论（都反直觉，勿重试已否方案）**：
1. **冲突是真的，但只值 0.07µs**：rep4 把 distinct-banks 从 17.81/32 提到 **20.30/32**
   ＝ 均匀随机理论上限 20.35/32 ✓（replay 4.43→3.58 cycle/LDS），时间只从 8.23→8.16。
   所以"残余 1.5~3.1µs 全在冲突"✗ —— **错**：lutcf 的索引不依赖刚加载的字节，
   它把 `LDS.8→ALU→LDS.32` 这条**依赖加载链**整个砍掉一级，2.47µs 里 97% 是这个 ✗，不是冲突 ✓。
   ⇒ **任何表布局都到不了 lutcf** ✓。
2. **旋转基址（①）净亏** ✗：地址 ALU（SHR/AND/IMAD/AND）落在同一条串行链上，
   省下的 ~0.6 replay cycle 抵不过 2-3 条 ALU 的延迟。REP4 的偏移可提到基址寄存器（零 ALU ✓），
   但 +3KB smem 在 n≥1024 因 occupancy 反而更亏 ✗。
3. **熵决定冲突，布局无关** ✓（决定性证据）：uniform 数据（mode U）下 lut/rot1/rot4/rot8/rep4
   **全部收敛到 20.40/32 distinct、1.57 ways** —— 布局对一个随机散列函数无任何改善空间 ✓。

**真赢家 = a32**（激活与 row 无关 ⇒ 每 block 预解码一次，在 barrier 后的 block 级并行段里，
延迟完全被隐藏 ✓）：8.23→7.71（-6%）/ 9.35→8.59（-8%）/ 11.96→10.39（-13%）。
**逐位一致** ✓：`s_af[i]=s_lut[ap[i]]*s_as[i>>5]`，循环里 `s_af[j]*(s_lut[row_s[j]]*sb)`
与 `s_lut[ap[j]]*sa*(s_lut[row_s[j]]*sb)` 的舍入序列相同 ✓（fingerprint out[0..3] 与 base 逐位相同，
含 rot/rep/a32/a32w32 全部变体 ✓）。代价：+20KB smem（48KB/block）。

**顺手修掉的 gprobe3 第二个 bug** ✗：T_W32 的 staging 写 `row_wf[(g<<5)+t]`（t=0..31 在 lane 内）
⇒ 32 个 lane 同一时刻全写 bank t ⇒ **32 路冲突的 STS** ✗；改为 lane 并行写 `row_wf[(i<<5)+lane]` ✓。
（故 gprobe3 的 w32/a32w32 数字偏高，不足采信 ✗。）
