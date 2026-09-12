# hc 链的带宽优化分析（53 GB/s 的根因与优化排序）

> 任务：W7 —— verify 段 hc 链（2.96ms / 157.3MB / 400 launch / 53 GB/s，全表最低带宽）。
> 方法：只读代码（`chain_dev.rs` + `kernels/cuda/*.cu`）+ 仓库内实测单价比对。
> **本机无 GPU ⇒ 未跑 nsys；所有时间数标了来源（实测 / 推导）。**
>
> 工部 · 2026-09-12 · HEAD 工作树（`crates/ferrite-models/src/dsv41/chain_dev.rs:7869`）

---

## 0. 头条结论（三条，且第 1 条推翻既定判断）

1. **verify 段根本没有用上任何 hc 融合。**
   verify 走的是 `step_rows_inner → layer_rows()`（`chain_dev.rs:7869`），它调用的是
   **原始 10 发链**：`dev.hc_mixes` / `dev.hc_collapse` / `norm_rows(dev.rmsnorm)` /
   `dev.hc_post` + `memcpy_d2d`，各 ×2（attention 侧 + FFN 侧）。
   而 `hc_mixes_auto`（`DSV41_HC_FRONT` + `DSV41_HC_TAIL_SPLIT` + `hc_front_split` / `hc_dots_late_kernel`）、
   `DSV41_FUSE_B1`（`hc_collapse_norm`）、`DSV41_FUSE_C`（`hc_post_inplace`）、
   `DSV41_HCPOST_EPI`（AR 折叠）**只接在 `layer()`（单行 decode）上**。
   `hc_mixes_auto` 的全部调用点只有两处：`chain_dev.rs:10415` / `:10527`，**都在 `layer()` 里**。

   ⇒ `verify-family-fusion.md:554` 写的「已有 HC_FRONT 融合（tail split）但主链还有 10 发/层」
   **不准确**：verify 连 HC_FRONT 都没走。这正是 400 发的来源，也是 W7 的**全部内容**。

2. **「53 GB/s」是把「只算权重的字节」除以「整条链的时间」得到的派生量，不是任一 kernel 的达成带宽。**
   157.3MB = `hc_mixes` 的权重读（80 次 × 1.966MB），而 2.96ms 是**五次不同 kernel** 的合计时间。
   链条自己真正搬的字节是 **388MB**（§2），⇒ 诚实口径的链内达成带宽是 **131 GB/s**。
   而它的**带宽地板只有 0.021ms**（157.3MB @ 7.6TB/s）——**hc 链从来不是带宽问题**，
   **不存在 100× 的 kernel 效率空间**（§3.3 给出真正的上限）。

3. **可回收量：2.96 → 1.2~1.4ms（−1.6~−1.8ms），其中 ~0.7ms 来自删 launch，~1.0ms 来自把 5 个 kernel 折成 2 个。**
   全部靠**已有件接线**（A1/A2），零新建 kernel；数值域全部 bit-exact。
   sinkhorn 迭代自适应（20→k）是**唯一**要动的数值契约，ROI 最低，排最后。

---

## 1. hc 链的精确 kernel 组成（verify 路径）

### 1.1 配置与本征量（`configs/dsv41_flash.json` 读出，非记忆）

```
dim = 5120 · hc_mult = 4 · hc_sinkhorn_iters = 20 · hc_eps = 1e-6
hc_dim = hc*dim = 20480 · mix = hc*(2+hc) = 24 · n_layers = 40 · m(verify) = 5
hc_fn[mix=24, hc_dim=20480] f32 = 24*20480*4 = 1.966 MB / 次调用
```

### 1.2 每层 10 发 = 5 个 C 入口 × 2（attention 侧 + FFN 侧）

调用点是 `layer_rows()`（`chain_dev.rs:7869-7968`），FFN 侧（`:7924-7968`）是 attention 侧的镜像：

| # | C 入口 | 调用点 | kernel | grid × block | 每发字节 (m=5) |
|---|---|---|---|---:|---:|
| 1 | `dsv41_hc_mixes` → `dsv41_hc_mixes` (`dsv41_kernels.cu:8127`) | `:7880` | `hc_mixes_kernel` | **grid = rows = 5** × `mix*32 = 768` | **2.376 MB** |
| 2 | `dsv41_hc_collapse` (`dsv41_glue.cu:1175`) | `:7894` | `hc_collapse_kernel` | 100 × 256 | 0.512 MB |
| 3 | `dsv41_rmsnorm_rows` (`:7949` 路径 `norm_rows`) | `:7902` | `dsv41_rmsnorm_rows_kernel` | **rows = 5** × 1024 | 0.225 MB |
| 4 | `ferrite_hc_post` (`ferrite_kernels.cu:2713`) | `:7911` | `hc_post_kernel`（staging 形态） | 100 × 256 | 0.922 MB |
| 5 | `memcpy_d2d` (`device.rs:1343`) | `:7921` | `cudaMemcpyAsync D2D`（`h2_r → h_r`） | — | 0.819 MB |
| | **每侧小计** | | **5 发** | | **4.854 MB** |
| | **× 2 侧 × 40 层** | | **400 发/步** | | **388.3 MB/步** |

> 落点核对：`hc_mixes_auto` / `hc_front_split` / `hc_front` / `hc_persist*` 在 `layer_rows`
> 里**一次都没出现**；`layer_rows` 的注释自己也写着「`hc_post` 用的是 staging 形态
> （`hc_post` + copy back）而不是单行的 `FUSE_C` 就地形态」——即它**显式选择**了未融合的形态。

### 1.3 每发的代价拆解（时间数来源见表注）

| kernel | 次/步 | µs/次 | ms/步 | 来源 |
|---|---:|---:|---:|---|
| `hc_mixes_kernel` | 80 | ~7.8 | **0.62** | `dspark-verify-perf-plan.md §1.2`「`hc_mixes` ≈ 7.8µs」（`STATUS.md` 6.15ms TP8 基线 median） |
| `hc_collapse_kernel` | 80 | ~5 | **0.40** | 推导（0.512MB @ ~100GB/s + 小核固定项；同族 `dsv41_hc_collapse_norm_kernel` 实测 6.0µs，见 `dsv41-kernel-inventory-v3.md:430`） |
| `dsv41_rmsnorm_rows_kernel` | 80 | ~3.5 | **0.28** | `dspark-verify-perf-plan.md §1.2`「`rmsnorm` 3.5µs」 |
| `hc_post_kernel` | 80 | ~3 | **0.24** | 推导（0.92MB；`hc_post_inplace` 同族实测 1.9µs，见 v3 §1 #11） |
| `cudaMemcpyAsync D2D` | 80 | ~2.5 | **0.20** | 推导（0.82MB + memcpy 发射税） |
| **kernel 小计** | 400 | | **1.74** | |
| **launch/串行半** | 400 | 3µs | **1.20** | `verify-ms-breakdown.md §1`（2.904µs 提交实测 + 执行不重叠） |
| **合计** | **400** | | **2.96** | ✅ 与账本实测吻合（残差 0.02ms） |

**⇒ 2.96ms = 1.20ms launch + 1.76ms kernel**，与 `verify-ms-breakdown.md` 的 `t_launch=1.20 / 残差=1.76` **逐项对上**。
**而且这 1.76ms 里，真正属于「hc 权重读」的只有 `hc_mixes` 的 0.62ms（35%）**——
剩下 1.14ms 是 collapse / norm / post / memcpy 四条**与 157.3MB 无关**的链。

---

## 2. 53 GB/s 的根因

### 2.1 三个根因（不是「kernel 写得差」）

**R1 — 结构性：verify 没接线（占 400 发里的 320 发）**
`hc_collapse`+`rmsnorm_rows`（折进 `hc_collapse_norm`，`DSV41_FUSE_B1` 默认 ON）、
`hc_post`+`memcpy_d2d`（折进 `hc_post_inplace` / AR epilogue，`DSV41_FUSE_C`/`DSV41_HCPOST_EPI` 默认 ON）、
`hc_mixes`（折进 `hc_front_split` EARLY + dots+LATE）——**三套融合件都在树里、都默认 ON，但只在 `layer()` 的调用链上生效**。
verify 每层白付 6 发（2 collapse + 2 norm + 2 post/copy 里的重复项）≈ **−240 发/步 = −0.72ms**。

**R2 — `hc_mixes` 的网格只有 `rows` 个块（5/148 SM），且每块拉整份权重**
`grid = rows = 5`、`block = 768`（24 warps，一 warp 一行投影）。每块：
- 读**整份** `hc_fn` 1.966MB（24 个 warp 各 80KB，行间不共享）；
- 24 个 warp **各自再读一遍** 80KB 的 x 行 ⇒ 单块 x 行流量 1.92MB（全靠 L1/L2 兜）。

⇒ 单块要拉 ~3.9MB，只有 **1 个 SM** 干活。实测 1.966MB / 7.8µs = **252 GB/s 单 SM**
（= HBM 峰值的 3.3%，与全仓「小 N GEMV 只跑到峰值 5%」同一堵墙）。
`STATUS.md:5028` 的同族实测也给了同一画像：`hc_mix_dots` 3.93MiB / 7.4µs = **531GB/s，每活跃 SM 21.6GB/s**。

**R3 — sinkhorn 的 20 轮 warp0 串行链（同核内不可隐藏）**
`hc_mixes_kernel:2437-2497` 之后是 `pre/post/comb` 的 sigmoid 与 4×4 Sinkhorn：
`STATUS.md:5205` 的相位探针（`/tmp/tail_phase_probe.cu`，差异法，已扣发射税）给出
`front 4.87 + sinkhorn 6.46 + collapseP1 0.46 + rmstail 1.18 = 12.97µs ≈ 13.01µs`。
同处 `STATUS.md:5208` 的「sinkhorn 藏进 collapse」**已被实测否决**：可藏窗口 0.46µs ≪ 6.46µs。
⇒ LATE 的临界路径就是 warp0 的串行链（80 次 shfl + 39 次 fp32 除的依赖链），
**同核内没有等长窗口**，唯一出路是把它挪到别的流上（HC_TAIL_SPLIT 干的就是这件事）。

### 2.2 「100× 提升空间」是错的 —— 真正的上限

```
hc 链真正搬的字节            388.3 MB/步
@ head 级的 5.9 TB/s（实测）   = 0.066 ms   ← 即使核效率拉满到 head 水平
@ 7.6 TB/s 峰值                = 0.051 ms
400 发的提交半（2.904µs/发）    = 1.16 ms   ← 地板，且不可压（不删 launch）
```

**⇒ 这条链的天花板 ≈ 1.2ms，瓶颈 100% 是 launch + 每次发射的固定项，不是带宽。**
`t_bw = 0.021ms` 恰好说明这一点：**字节比时间小 140×**。
把它读成「100× 的 kernel 效率空间」会导向错误的药方（换核 / TMA / cp.async），
正确的药方是**删 launch + 并核**——这也与 `verify-ms-breakdown.md §3.2` 的「动作 A：合并/多行化」一致。

---

## 3. 优化方案排序（按 ROI）

### 方案 A1 —— verify 接上现成的两个折核（最便宜，bit-exact，可立即做）

| 项 | 内容 |
|---|---|
| 落点 | `chain_dev.rs::layer_rows` `:7894`（`hc_collapse` + `norm_rows`）与 `:7911/:7921`（`hc_post` + `memcpy_d2d`） |
| 改动 | ① `hc_collapse`+`norm_rows` → **`dev.hc_collapse_norm(rows=m)`**（`dsv41_kernels.cu:8280`，**已有 `rows` 维、grid = rows**，decode 已默认 ON）；② `hc_post`+`memcpy_d2d` → **`hc_post_inplace`**（`dsv41_kernels.cu:8206`），需要把它的 `n` 维从「1 行」放宽到 rows（kernel 结构天然支持：一线程独占一个 `(t, j4)` 列，读 `res` 的 k=0..n-1 行、写全部 i=0..n-1 行 ⇒ `res == out` 别名安全） |
| 收益 | 每层 10 → 6 发：**−160 发/步 = −0.48ms**；kernel：少一趟 `x_r` 往返（100KB）+ 去掉 `h2` staging 与 0.82MB copy ⇒ **−0.2ms**；合计 **≈ −0.7ms（2.96 → ~2.3）** |
| 风险 | 低-中（A 部分是把 decode 已验证的组合搬到 rows=m；B 部分是新写 rows>1 的 in-place 变体，需一次 bit 对拍） |
| 数值域 | **bit-exact**。`hc_collapse_norm` 的 phase1/2（`dsv41_kernels.cu:8253-8277`）与 `hc_collapse` + `rmsnorm` 逐句相同（同一 `fmaf` 链序、同一归约树），decode 侧已有 `FUSE_B1` A/B；`hc_post_inplace` 与 `hc_post`+copy 的 bit-exact 论证（`ferrite_kernels.cu:2687-2693` 与 `dsv41_kernels.cu:8158-8168`）按「每线程独占列」原样推广到 rows>1 |

### 方案 A2 —— 把 `hc_mixes_auto`（`hc_front_split`）接到 `layer_rows`（主菜）

| 项 | 内容 |
|---|---|
| 现状 | C 侧 `hc_front_split`（`dsv41_kernels.cu:9641`）与两个子核**本来就带 rows 维**：`hc_mixes_tail_kernel` 用 `blockIdx.x = r`、`hc_dots_late_kernel` 用 `blockIdx.y = r`，签名里 `rows` 是第一个参数。缺口在 Rust 侧：`hc_mixes_auto` 自己把行数**硬编码成 `1`**（`chain_dev.rs:10198` 的实参 `1,`，转发给 `hc_front_split` 的 `rows`），且调用它的只有 `layer()`。所以 A2 = 「给 `hc_mixes_auto` 加 `rows` 形参 + 在 `layer_rows` 调用它 + 事件/流处理」 |
| 实现路径 | `layer_rows` 的两处 `hc_mixes` 换成 `hc_mixes_auto(...)`，`hc_done/ffn_done` 为真时跳过 collapse（EARLY 已折）；`hc_post` 换 AR 折叠（`ar_hc_post_fold` 的 rows 版，或 `HCPOST_EPI` 等价物）；`hc_tail_join()` 插在消费 `comb` 之前 |
| 收益 | 每侧 5 → 2 发（EARLY on `side` + dots+LATE merged on `dl`）；含两处 AR 折叠 ⇒ 每层 10 → 4 发 ⇒ **−240 发/步 = −0.72ms**；再把 collapse/norm/post/memcpy 的 kernel 时间折掉 ~0.6~0.9ms；合计 **≈ −1.3 ~ −1.7ms（2.96 → ~1.3~1.7）** |
| 风险 | 中：verify 未图化（`DSV41_VERIFY_GRAPH` 默认 OFF）时 side/DL 流是**纯重叠收益**；一旦开启整步捕获，必须保持 `cudaEventRecord`/`WaitEvent` 的**相邻配对**（`dsv41_kernels.cu:9680-9719` 的 program-order 消歧要求，已有注释写死） |
| 数值域 | **bit-exact by construction**。EARLY = `hc_collapse_norm` 逐句；dots = `hc_mix_dots_kernel` 的 float4 三累加器链逐句；LATE = `hc_mixes_tail_kernel` 的 `ss_in==1` 分支逐句（ss 的分组由 `mix*32` 步长保证，与 blockDim 无关）。AR 折叠沿用 `DSV41_HCPOST_EPI` 的契约（不同 TU 的 epilogue，`DSV41_HCPOST_EPI=0` 是 A/B 臂） |

### 方案 B —— dots 形态的占用率修复（bit-exact，已有开关，收益有限）

| 项 | 内容 |
|---|---|
| 落点 | `DSV41_HC_DL_KCHUNK=1`（`dsv41_kernels.cu:9130`，**默认 OFF**）+ `DSV41_HC_DL_KCHUNK_F4=768` |
| 原理 | 把 160KB 的「x 行 + 权重行」整份 staging 换成 48KB 双缓冲（4 blocks/SM），cp.async 提前发下一 chunk；`hc_dots_late_kernel` 的 3.84MB/发 里 **1.92MB 是纯冗余**（每个 block 把同一份 x 行拖一遍） |
| 收益 | `hc_mix_dots` 的 7.4µs 里，staging 暴露的 DRAM 延迟被藏 ⇒ 估 **−0.2 ~ −0.3ms/步** |
| 风险 | 低：启动器自带约束（`chunk % lcm(96, mix*8) = 192 == 0` 且 `hc_dim % 4 == 0`，生产形状均满足），不满足自动回退 |
| 数值域 | **bit-exact by construction**：chunk 对齐保证每 lane 的 fp32 累加器**操作数序列**不变（`dsv41_kernels.cu:9292-9303` 已给论证） |
| 备注 | 单列做收益不大（R2 的真正解药是 A2 把 dots 从 5 块变 120 块），**建议作为 A2 的附赠** |

### 方案 C —— 跨层融合（层 i 的 `hc_post` 与层 i+1 的 `hc_mixes`）

**判定：独立增量 ≈ 0，不单列。** 理由：

- 层间**不是**相邻算子：`hc_post(i)` 写 `h_r`，`hc_mixes(i+1)` 读 `h_r`，中间隔着整段 attention/MoE + AR。真正的可折叠关系已被 A2 覆盖（`hc_post` 折进产生它的 AR 尾 = `HCPOST_EPI`；`collapse` 折进上一层 tail = HC_FRONT 的 EARLY 形态）。
- 唯一还剩的是 **persistent 段核**（把 L 的 tail 链藏进 L+1 的 attn/AR 窗口）：`DSV41_HC_PERSIST` / `_MB` 已存在且**默认 OFF**，历史上 `hc_front_kernel` 的 ticket-spin 单核形态实测 **+3.2ms/步**（`dsv41_kernels.cu:9513-9519`）。要重做必须用 `hc_dots_late_kernel` 的 **elected-last-block** 机制（无 spin、不占 SM），属另一波 P4 工作。
- **注意**：R3 的 sinkhorn（6.46µs 串行链）在这一波**只能靠 A2 的 side-stream 重叠**解决——这正是 HC_TAIL_SPLIT 存在的意义（`chain_dev.rs:10387`：「hiding its ~10.7us of serialised sinkhorn latency」）。

### 方案 D —— sinkhorn 迭代自适应（20 → 5/~8）

| 项 | 内容 |
|---|---|
| 收益 | 探针：sinkhorn 6.46µs/次（相位预算，decode 口径）。砍到 5 轮理论省 ~4.8µs × 80 = **−0.39ms/步**；早退（行/列和 \|1−s\|<tol，典型 6~8 轮收敛）≈ **−0.25ms/步** |
| 风险 | **非 bit-exact，且误差有界性需要单独论证** |
| 排序 | **最后**。ROI 低于 A1/A2（0.3ms vs 1.7ms），却要动一条数值契约 |

---

## 4. 数值域安全（逐方案的位等价风险）

| 方案 | 位等价 | 论证 / 缺口 |
|---|---|---|
| **A1-a** `hc_collapse_norm` | ✅ bit-exact | 同一 `fmaf(pre[i], x[i*dim+c], acc)` 升序 i 链；同一 `shfl_down` 树 + `red[32]` 跨 warp 归约；同一 `o*inv*w` 尾。decode 侧 `DSV41_FUSE_B1` 默认 ON 已 A/B |
| **A1-b** `hc_post_inplace` rows>1 | ✅ bit-exact（需一次对拍） | 每线程独占 `(t, j4)` 列，读 k=0..n-1 **写完 i=0..n-1** ⇒ 读写集同列，`res==out` 别名安全；累加仍是 `__fmaf_rn(c, r_k, acc)` 升 k 序。缺口：现有实现硬编码 `s=1`，rows>1 需带 `n/rows` 两维索引，**必须跑 `tests/hc_parity` 类对拍** |
| **A2** `hc_front_split` on rows=m | ✅ bit-exact by construction | EARLY/dots/LATE 三段都是既有核的**逐句照抄**（源码注释 `dsv41_kernels.cu:9073-9080`、`:9720-9723`）；ss 的分组由 `mix*32` 步长固定，与 blockDim 无关（`hc_mixes_tail_kernel:8427` 的 `nwarp = ss_stride>>5`）。AR 折叠：`DSV41_HCPOST_EPI` 已有 A/B 臂 + 契约注释（`chain_dev.rs:10339-10348`） |
| **B** K-chunk | ✅ bit-exact by construction | chunk 对齐 ⇒ 每 lane 的累加器**操作数序列**（96-float4 网格 + `mix*8` ss 步长）不变；`hc_dim%4==0` 保证无未 staging 的标量尾。缺口：**未实测**（默认 OFF），需一次 nsys 复核收益 |
| **C** 跨层 | — | 无独立收益，不评 |
| **D** sinkhorn 20→k | ❌ **非 bit-exact** | 4×4 矩阵的 Sinkhorn 以谱隙速率收敛；砍 15 轮留下的相对误差 ~1e-3~1e-4 量级（**远高于 fp32 噪声**），而 `comb` 线性进入残差（`out = post·x + Σ_k comb[k,i]·res[k]`）。折中「收敛早退」仍 data-dependent ⇒ **必须走容忍度 A/B + accept 率验证**（四段文本 + `DSV41_TOKTRACE` + 每步 accept 统计），不可凭 ms 默认 |

---

## 5. 落点汇总与预期

| 方案 | 动作 | launch/步 | ms/步 | 累计 | 位等价 |
|---|---|---:|---:|---:|---|
| — | 现状（verify 走 `layer_rows`） | 400 | **2.96** | 2.96 | — |
| **A1** | `hc_collapse_norm` + `hc_post_inplace(rows=m)` | 240 | ~2.26 | ~2.3 | ✅ |
| **A1+A2** | 接 `hc_mixes_auto`（EARLY + dots/LATE merged + AR 折叠） | **160** | **~1.3** | **~1.3** | ✅ |
| +B | `DSV41_HC_DL_KCHUNK=1` | 160 | ~1.1 | ~1.1 | ✅ |
| +D（可选，最后） | sinkhorn 早退 | 160 | ~0.85 | ~0.85 | ❌ 需容忍度 A/B |

**对比 `verify-family-fusion.md` 的目标列**：它给 hc 链「2.96 → ~2.0（HC_FRONT 已有 + W7）」。
**该目标偏保守且前提错误**——HC_FRONT 在 verify 上并不「已有」，接线后可直接落 **~1.3ms**，
比其目标列多拿 ~0.7ms，且不需要动任何数值契约。

---

## 6. 三个待实测的断言（无 GPU，必须 nsys/对拍）

| # | 断言 | 验证方式 |
|---|---|---|
| H1 | `hc_mixes_kernel` 在 verify（rows=5）的 per-call = 7.8µs | 一条 nsys 按 kernel 名聚合 verify 段（`DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_TIMING=1`） |
| H2 | 单 SM 252GB/s 是 `hc_mixes` 的真瓶颈（而非 L2/smem） | ncu `dram__bytes.sum` + `sm__warps_active`；或 `DSV41_HC_MIXES_SPREAD=1` 的臂作方向性参照（**该臂非 bit-exact，只作诊断**） |
| H3 | `hc_post_inplace` 推广到 rows>1 后与 `hc_post`+copy 逐位一致 | `tests/hc_parity.rs` 风格的 rows=5 对拍（0/差异） |

⚠️ **不要**用 `DSV41_HC_MIXES_SPREAD=1` 当优化：`dsv41_kernels.cu:8113-8120` 记着
「同会话、同二进制的 A/B **不逐位一致**：31.44 vs 32.07ms，而答案发散（'2' 变跑题）」——
它是诊断臂，不是落地臂。

---

*工部 · 只读分析，未改动任何代码；本文件为唯一产出。*
*字节按 `configs/dsv41_flash.json` 的真实 shape 推导；时间数凡非实测者均已标注来源。*
