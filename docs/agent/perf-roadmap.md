# 16-seq decode 性能路线图（2026-09-08 会话结论）

## 目标与现状

- 目标：16 并发、**不开 MTP**、decode ≥ **1600 tok/s**（SGLang 同配置实测）。开 MTP 则目标 3200。
- 现状（已人眼验证文本）：
  - 300-token 窗口：**17.42 ms/步 = 918.5 tok/s**
  - 1000-token 窗口：**18.90 ms/步 = 846 tok/s**（DSA/attention 的 O(t) 增长，-8%）
- 本会话进度：546 → 918.5 tok/s（**+68%**）。

## 两条决定性实测（不要再重复验证）

1. **通信（AR）只占 7%**：同 build 同负载，`FERRITE_AR_SKIP=1` → 16.4 ms/步；带 P2P AR → 17.6 ms/步。
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
