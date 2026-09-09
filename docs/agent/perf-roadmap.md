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
