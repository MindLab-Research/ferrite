# DSV4.1-Flash 会话最终报告（定稿 v2）

> **状态：已定稿**（2026-09-12 01:20，HEAD `6fc6113`）。本文件是会话的权威汇总，
> 数字全部来自本会话已验证的 serve A/B 或隔离复现器；未验证项已显式标注。
>
> **会话终态**：**13.28 → 6.15–6.17ms（+115.4%），75.3 → 162.1–162.6 tok/s**。
> 最优读数 v24a **6.15ms / 162.6 tok/s**、v17a **6.17ms / 162.1 tok/s**，四段文本全对、`faults=0`。
> **"唯一数量级路径" swapAB 经 5 变体 serve 全矩阵中性，已于 v24 正式关闭**（§3.5）；
> DL K-chunk 亦于 v22 serve 判**略负**（第 10 次失败）。
> **200 tok/s 判定见 §9：tcgen05 + gemv 突破 + hc/AR 突破三者同时成立才可能——研究级。**

---

## 0. 一句话结论

会话把单请求 decode 步时从 13.28ms 砍到 **6.17ms**（162.1 tok/s，+115.4%），
**彻底解开**了 v13→v15 连续三轮的 6.6ms 谜团——根因**不是**"热点 kernel 代码存在性"
（residue-hunt 的误诊），而是 commit `006bd0c` 的 `FileReplace` **改错了同名变量**：
把 K-split（已验证 **−0.33ms** 收益）误关、把 PDEPTH pipeline（**+0.04ms** 回归）误开，
净 **+0.37ms 隐藏回归**。这是一条**代码卫生 > 性能分析**的教训。

**第二个同等重要的结论**：曾被评为"唯一数量级路径（预估 ~238 tok/s）"的 **swapAB**，
经 v17→v21 四个变体（全量 / 形状分发 / memset 消除 / TMA bulk staging）的完整 serve 矩阵验证，
**全部中性**。隔离口径的 1.76–1.94x 在 serve 的 SM 争抢 + L2 竞争下**完全不兑现**——
这是本会话第 7 次、也是最昂贵的一次"隔离→生产失效"。**swapAB 路径正式关闭**（§3.5）。
教训升级为：**没有任何隔离收益可以不经 serve A/B 就直接写进路线图**。

---

## 1. 优化链全景表（13.28 → 最终）

| # | 优化 | p50 ms | 累计收益 | commit | 验证 |
|---|---|---|---|---|---|
| 0 | 会话基线 | 13.28 | 75.3 tok/s | — | — |
| 1 | shared expert TP 切分 | 11.85 | +10.8% | `9cddfb8` | serve A/B，四段全对 |
| 2 | sparse 3-deep key-split + e4m3 LUT + 融合系列 | 9.38 | +29.4% | `710c107`、`54f470b`、`9e4e3df`、`2d7eead` | serve A/B |
| 3 | gate v2 + gateup MLP unroll + down revert | 8.60 | +35.2% | `65acfb1`、`73e4a50`、`5013268` | serve A/B |
| 4 | hc tail split + swiglu_q | 8.23 | +38.0% | `7d623ad` | serve A/B |
| 5 | allv2（K-split + P1 a32 + B+C + dead slots + launch_bounds） | 6.84 | +48.5% | `f019237`、`a6c4ed3`、`d76a183`、`c12ff60` | serve A/B |
| 6 | cp.async 权重先行 + warps8 + EARLY 侧流 | 6.66 | +50.5% | `f6aa8c8`、`a7600db`、`9558491` | serve A/B |
| 7 | down 回归修复 + dots-LATE merge | 6.48 | +51.4% | `96f169b`、`1b265a8` | serve A/B |
| 8 | AR store 并行化（grid.y=world） | 6.47 | +51.5% | `f911abd` | serve A/B |
| 9 | P4 act-cpasync（默认 ON） | **6.24** | **+112.8%** | `fdd70d4` | serve A/B，160.3 tok/s |
| — | *v10–v13 回归区（见 §4）* | *6.31→6.63* | *回退* | — | — |
| 16 | **gate 错配修复**（ksplit=2 / pipeline=1） | **6.26** | **+112.2%** | `4bed9f6` | serve A/B，159.7 tok/s ✓ |
| 17 | **会话终态复测**（v17a 基线，同 v16 配置） | **6.17** | **+115.4%** | — | serve A/B，162.1 tok/s ✓ |

> v9（P4 act-cpasync，6.23–6.24ms / 160.5 tok/s）与 v17a（6.17ms / 162.1）是会话中的两个最优读数；
> v16 的 6.26ms 与 v17a 的 6.17ms 是**同配置**，0.09ms 的差异是 serve 读数漂移（噪声内）。
> **"最终值"取 6.17ms / 162.1 tok/s（+115.4%）。** 后续 v18–v21 的全部 swapAB 变体均未超过此基线。
>
> ⚠️ **重要**：§1 表只记录**已落地并 serve 验证**的优化。v17–v21 期间实施但默认 OFF 的
> 三条新路径（swapAB / TMA / DL K-chunk）见 §2.2，它们**不构成收益**。

---

## 2. 落地优化清单（全部 serve A/B 验证）

| # | 优化 | 机制 | 来源 |
|---|---|---|---|
| 1 | **shared expert TP 切分** | 共享专家从"仅 rank0 串行"改为所有 rank 各切 `inter/world` | `9cddfb8` |
| 2 | **sparse 3-deep 预取 × key-split** | 同一 kernel 加 key 分块，grid=(C,b·m,h)，per-slot 78→10.7ns | `710c107` |
| 3 | **e4m3 LUT** | fp8 解码从位运算改 256-entry smem LUT | `54f470b`/`9e4e3df` |
| 4 | **gate v2** | MoE gate 的 bf16 gemv（n=384，延迟受限 3% 峰值带宽）向量化 + K-split | `65acfb1` |
| 5 | **gateup MLP unroll** | `#pragma unroll 2→4` + uint2 宽加载 | `73e4a50` |
| 6 | **hc tail split** | tail 拆成 EARLY（主流）/LATE（侧流）两半，侧流 fork/join | round-41 |
| 7 | **allv2（P1 a32 dead-slot + B+C）** | a32 物化直写 `s_af`（消 `s_a` 中转，smem 48512→43392B，4→5 blk/SM） | `f019237`/`a6c4ed3` |
| 8 | **cp.async 权重先行 + warps8 + EARLY 侧流** | prologue 与权重读重叠；n≥2048 时 warps=8 减半 block | `f6aa8c8`/`a7600db` |
| 9 | **down 回归修复 + dots-LATE merge** | 去 `__launch_bounds__(256,4)`（4→6 blk/SM）+ 恢复 ILV；dots 与 LATE 合一个 launch | `96f169b`/`1b265a8` |
| 10 | **AR store 并行化** | `grid.y=world`（5→40 blocks，3%→25% SM），每 block 只写一个 peer | `f911abd` |
| 11 | **P4 act-cpasync（默认 ON）** | 激活 staging 用 cp.async，消除 a32 物化的串行 LDG | `fdd70d4` |
| 12 | **COMPRESS_FUSE 3→1** | decode compressor 的 state+pool+commit 合一 launch，逐字节不变 | `595437d` |
| 13 | **gate 错配修复** | 恢复 K-split=2（−0.33ms）、PDEPTH pipeline=1（去 +0.04ms） | `4bed9f6` ✓ |

**旁证**：`175462d` 的 fold 物理清理与 `3570524` 的 hot-kernel-restore 本身也是"落地"——
它们不是为兑现收益，而是**修复缺陷 gate、消除 ABI 风险、恢复 np1 形态**（见 §4）。

### 2.2 已实施但默认 OFF（未兑现收益，勿计入）

| # | 实现 | commit | 默认 | serve 结果 |
|---|---|---|---|---|
| 1 | **swapAB kernel 族**（`gemm_fp8_swapab_kernel`，形状分发 + last-block reduction） | `166acc5`/`9696afa`/`b7744a1` | OFF | **中性**（4 变体全中性，§3.5） |
| 2 | **TMA Phase 1**（1D `cp.async.bulk` staging + mbarrier，`DSV41_SWAPAB_TMA`） | `0f34001` | OFF | **中性**（v21t，§3.5） |
| 3 | **DL K-chunk cp.async 流水**（`hc_dots_late_kchunk_kernel`，`DSV41_HC_DL_KCHUNK`） | `6fc6113` | OFF | **略负**（v22：6.25 vs 6.17ms，第 10 次失败） |

> 三条都是**完整的、编译通过、位一致已论证**的实现，保留在树内供后续升级后重测。
> 但**它们都不构成会话收益**——报告的数字口径以 §1 的 serve A/B 为准。

---

## 3. 失败关停清单（全部有机制级解释）

| # | 优化 | 预期 | 实际 | 根因 | 处置 |
|---|---|---|---|---|---|
| 1 | **PDEPTH pipeline**（2/5） | −0.48ms | **+0.04ms** | 占用率损失 > 延迟隐藏（42.8KB smem → 2 blocks/SM）；隔离时长 kernel 掩盖占用率 | 默认 `GATEUP_PIPELINE=1`（OFF） |
| 2 | **w2 L2 prewarm** | −0.25ms | **+0.04ms** | warmer 与 gateup 尾部争 SM（PDL 重叠变争抢） | `W2_PREWARM=0`（OFF） |
| 3 | **quant fold** | −0.072ms | **+0.39ms**（与 swiglu 合计） | fork_ev 是 **kernel 级**——EARLY 加工作 = 加到 main 关键路径；且 gate 缺陷使 fold 一直生效 | 代码物理删除（`175462d`） |
| 4 | **swiglu fold** | −0.068ms | ↑ 同上 | w2 prologue 的 swiglu 在 GEMV 上下文比独立 kernel 贵（寄存器/布局） | 代码物理删除 |
| 5 | **AR stamp fold** | −0.15ms | **29.5s/step** | 单调 counter 在图 replay 下机制坏掉（poll 自旋等永远不来的 stamp） | 代码物理删除（`175462d`） |
| 6 | **sparse-merge 选举折叠** | −0.05~0.10ms | **中性**（v14a=v14b=6.61ms） | 真中性；但其代码存在性被误判为 +0.34ms 回归源 | 整体回退（`3570524`） |
| 7 | **ringwin 全折叠（RW_FOLD）** | −0.02~0.05ms | 未上机 | 未验证即进树；给热点 `rmsnorm_rope_kernel` 加 6 尾参 | 整体回退（`3570524`） |
| 8 | **a32-vec4** | −0.13ms（隔离） | **中性** | P4 的 cp.async 已隐藏 staging，vec4 在其上增益≈0 | 保留（无害） |
| 9 | **AR reduce grid** | −0.08~0.16ms | **中性** | 8µs 主导项是 stamp/poll 的 NVLink 往返（~6µs），网格形状只影响 1–2µs | 保留 |
| 10 | **wo-pair 两段核** | −0.13ms | **+0.46ms** | 共享 smem 池按 max(k)=5120 分配 → phase2 占用率 96→20 warp/SM（4.8× 崩塌） | `WO_PAIR` 保持 OFF |
| 11 | **sh-pair** | −0.26ms | 未上机 | 与 wo-pair 同根因且更极端（k 比 8:1） | 审查后放弃 |
| 12 | **M>1 expert batching** | — | 价值 = 0 | 单请求无第二个 token；层内 6 专家已在一个 launch | 关闭 |
| 13 | **expert MMA 化** | 绕过指令地板 | 前提不成立 | sm_103a 无 M=8/16 的 fp4 MMA（tcgen05 mxf4 的 M 钉死 128）；M=128 masked 实测 0.2% 峰值 | 关闭 |
| 14 | **Stage C persistent 段核** | −1.0~1.5ms | 最小版仅 0.1~0.2ms | smem 池按 max(k) 分配 → 占用率崩塌 + `grid.sync` 与整步图捕获不兼容 + 段内近全串行 | 关闭 |
| 15 | **cross-layer pipe** | −0.3~0.5ms | −0.05~0.15ms | 80% 被路由依赖挡死（L+1 专家取哪 6/384 由 gate(xn) 决定） | 关闭 |
| 16 | **bf16-lut（a32 物化的 bf16 LUT）** | −0.1ms | **分析否证（未落地）** | 位序错配：`*(bf162*)(lut + b4)` 取 `LUT[c0]`/`LUT[c0+1]`，而 `o.y` 要 `LUT[c1]` ⇒ 49.8% 元素错 | 关闭（勿再提案） |
| 17 | **cvt 解码路线** | 省 smem gather | **分析否证（未落地）** | 转换管 16 results/clk/SM < LDS 32 值/clk/SM；且丢位一致 | 关闭（见 §5） |
| 18 | **swapAB（SIMT 形状分发）** | −2.0ms（隔离 1.76–1.94x） | **中性**（v18s 6.20 vs 6.17） | 隔离收益在 serve 的 SM 争抢 + L2 竞争中消失（第 7 次隔离→生产失效） | 默认 OFF；**路径正式关闭**（§3.5） |
| 19 | **swapAB + memset 消除** | −0.5µs/call | **中性**（v19s 6.18） | memset 本非瓶颈（图节点已被 last-block reduction 消除） | 随 §3.5 关闭 |
| 20 | **swapAB + TMA bulk staging** | 突破 staging 1.3TB/s 墙 | **中性**（v21t 6.18） | bulk DMA 不占 LSU ≠ 免于 SM 争抢；结论与 staging 方式无关 | 默认 OFF；**路径正式关闭** |
| 21 | **DSV41_HC_DOTS_T = 256/512** | −1~1.5µs/launch | **中性**（6.21 / 6.19） | 默认值 128 已是优；DL 的瓶颈不在 dot block 尺寸 | 保持默认 128 |
| 22 | **mx2 swapAB 变体**（wq_a+wkv 合并过阈值） | −0.28ms（理论） | **未上机（预判失效）** | 与 §3.5 同源；基于同失效模式，理论需大幅打折 | 不实施 |

---

## 3.5 swapAB 路径的完整验证矩阵（正式关闭）

> swapAB 曾被 §5 评为"唯一数量级路径（预估 ~238 tok/s）"，并投入 5 个迭代。
> 这是本会话**最昂贵的一次隔离→生产失效**，其完整证据链必须留档，防止未来重启。

| 变体 | 消除的变量 | p50 | tok/s | 判定 |
|---|---|---|---|---|
| v17a（基线，SIMT） | — | **6.17ms** | **162.1** | 基准 |
| v17s（全量覆盖 swapAB） | — | 6.63ms | 150.8 | **回归**——~146 个小 n 调用（0.73–0.97x）吃掉 ~100 个大 n 收益 |
| v18s（形状分发 n≥1664） | 小 n 回归 | 6.20ms | 161.3 | **中性**（Δ=+0.03） |
| v19s（+ memset 消除，last-block reduction） | memset 图节点 | 6.18ms | 161.8 | **中性**（Δ=+0.01） |
| v21t（+ TMA bulk staging，1D `cp.async.bulk` + mbarrier） | staging 路径 | 6.18ms | 161.8 | **中性**（Δ=+0.01） |
| v24r（+ 环满几何 KStep=64/NSTAGE=16，8x warps 真正 in-flight） | ring 几何 | 6.17ms | 162.1 | **中性**（Δ=+0.02） |

**逐项排除（排除法定位根因）**：
- **不是小 n 回归**（v18 形状分发已隔离）
- **不是 memset**（v19 已消除）
- **不是 staging 方式**（v21 TMA vs cp.async 同结果）
- **不是 ring 未填满**（v24 KStep=64/NSTAGE=16 使环真正填满、8x warps 成为真 in-flight，仍中性）
- ⇒ **是深层系统差异**：serve 的 4 条 side stream 造成 SM 争抢 + L2 竞争，
  使隔离口径的 cp.async staging 1.3TB/s 与"混合形状下的 SIMT 实际耗时"两头失真。
- ⇒ **5 个正交维度的修复全部无效 ⇒ swapAB 在 serve 的中性是绝对系统性的。**

**结论：swapAB 路径正式关闭。** 代码保留（`DSV41_SWAPAB` 默认 OFF，`166acc5`/`9696afa`/`b7744a1`/`0f34001`），
供未来架构变化（如去掉 side stream、增大 L2 隔离）后重测，但**不得再计入路线图收益**。

---

## 4. v10 → v16 完整侦探链（本会话最有价值的知识）

> 这段是本会话的核心：一条 **0.37ms 的隐藏回归**如何在 5 个 commit 里被反复误诊，
> 最终从"性能问题"变成"代码卫生问题"。

### 4.1 v10（07:00）：三项组合回归 +0.08ms

| 臂 | p50 | tok/s | 判定 |
|---|---|---|---|
| v10（PDEPTH=5 + w2warm + bf16cpasync） | 6.31ms | 158.5 | **+0.08 回归** |

四段全对，faults=0。**operand-supply 理论在 serve 中未兑现**（第三次隔离→生产失效）。

### 4.2 v10 bisect（07:30）：三项各自 +0.04 / +0.04 / 中性

| 臂 | p50 | 判定 |
|---|---|---|
| v10np1（pipeline OFF） | 6.27ms | pipeline 贡献 +0.04 |
| v10np2（PDEPTH=2） | 6.30ms | 浅 pipeline 也 +0.03 |
| v10nw（w2warm OFF） | 6.27ms | warm 贡献 +0.04 |
| bf16cpasync | — | 中性（两臂一致推出） |

**处置**：PDEPTH 默认 → 1（OFF）、W2_PREWARM 默认 → 0（OFF）、bf16 保持 ON。
> ⚠️ **这次"翻默认值"的提交 `006bd0c` 就是后续所有谜团的种子**——见 §4.7。

### 4.3 v11 崩溃（08:00）：部分提交的 FFI 错位（709）

`git add -A` 的 docs 提交扫入了 quant-early-fold subagent 的**部分实施**：
新 `.cu`（`hc_mixes_tail_kernel` 多 `xq4/xsc4`）+ 旧 Rust（旧 FFI）= 参数错位 = **cuda error 709**。
修复 `106db01` 补上 Rust 侧。**这是第 4 次部分提交事故，第一次真正崩溃**。

### 4.4 v12（08:30）：三个 fold 全部失败

| 臂 | p50 | 判定 |
|---|---|---|
| v12（quant fold + swiglu fold ON） | 6.59–6.62ms | **+0.36~0.39 回归** |
| v12sf（+ stamp fold ON） | **29.5s/step** | **灾难性失败（~4700× 慢）** |

- **quant fold**：fork_ev 在 EARLY kernel **完成时**记录 ⇒ fp4 直出 +1.8µs × 80 fronts = +0.14ms，
  省的 quant launch 只有 −0.072ms ⇒ 结构性净负。
- **stamp fold**：单调 arrival counter 在图 replay 下机制坏掉（poll 等永不到来的 stamp）。
- 处置：QUANT_FOLD / SWIGLU_FOLD / AR_STAMP_FOLD 全部默认 OFF。

### 4.5 v13（09:30）：fold 代码全 OFF，回归仍在

| 臂 | p50 | 判定 |
|---|---|---|
| v13（fold 代码全 OFF + compress-fuse ON） | 6.63ms | +0.40 回归 |
| v13nc（COMPRESS_FUSE=0） | 6.62ms | compress-fuse 不是源 |
| v13nq（全 gate 显式 OFF） | 6.60ms | gate 无关——初判"代码存在性"是源 |

**寄存器假设证伪**：remote `nvcc -Xptxas -v` 显示 pre-fold 与当前都 40 registers（无膨胀）；
hc_mixes_tail 在 decode 只有 1 个 block——占用率不是变量。

**fold-correctness-audit（10:00）**：quant-fold 的 **gate 有缺陷**——`=0` 时 `layer()` 仍传非空 `xq4`
（`chain_dev.rs`），kernel 只看 `xq4 != nullptr` ⇒ **fp4 直出照做**。所以 v13 ≈ fold 一直生效。

**`175462d`**：fold 代码物理清理（847 删除 / 14 文件），修复 gate 缺陷 + 消除 ABI 风险。

### 4.6 v14（11:00）：清理无效！——侦探链的分水岭

| 臂 | p50 | 判定 |
|---|---|---|
| v14（fold 清理 + sparse-merge + compress-fuse） | **6.60ms** | **清理未恢复**（回归仍在） |
| v14a（`SPARSE_MERGE_FOLD=0`） | 6.61ms | — |
| v14b（`SPARSE_MERGE_FOLD=1`） | 6.61ms | sparse-merge **真中性** |

**residue-hunt（假设 A）**：+0.34ms = "热点 kernel 编译产物重量"——
归因于 `sparse_attn_split_kernel` 的 **+12 运行时参数 + 选举代码**（gate OFF 无法编译期消除），
以及 `rmsnorm_rope_kernel` 的 6 尾参同款风险。
**sparse-gate-analysis**：gate 有效（选举不执行）但"代码存在性"有害。

**`3570524`（hot-kernel-restore）**：据此回退 **499 删除**，两个 kernel 恢复 np1（`d546139`）形态；
`c4220bb` 修复误删的 `rmsnorm_rope` launch 行（构建修复）。

> ⚠️ **假设 A 是误诊**——v15 未能恢复（见下）。

### 4.7 v15（00:12）：错误默认值仍 6.59ms → gate-hygiene-audit 揭穿真相

| 臂 | p50 | tok/s | 配置 |
|---|---|---|---|
| v15 | 6.59ms | 151.7 | ksplit=1（**误关**）+ pipeline=2（**误开**） |

**真相（gate-hygiene-audit）**：commit `006bd0c` 的 `FileReplace` 以 `int v = 2` 作 `old_string`，
**命中了错误的同名变量**：
- 打中 `dsv41_gateup_ksplit`（`dsv41_experts_mxf4.cu:869`，**已验证 −0.33ms 收益**）→ 被改成 `1`（关闭）
- 而本该改的 `dsv41_gateup_pipeline`（`:965`，**+0.04ms 回归**）原样停在 `2`（开启）

净效果 = **−0.33ms 收益丢失 + 0.04ms 回归保留 = +0.37ms 隐藏回归**，
这就是 v13/v14/v15 全部 6.6ms 的原因。**"热点 kernel 代码存在性"理论是误诊。**

**`4bed9f6`**：恢复 `dsv41_gateup_ksplit` 默认 = 2、`dsv41_gateup_pipeline` 默认 = 1。

### 4.8 v16（00:20）：基线恢复，谜团彻底解开 ✓

| 臂 | p50 | tok/s | 配置 |
|---|---|---|---|
| v15（错误默认） | 6.59ms | 151.7 | ksplit=1 + pipeline=2 |
| **v16（gate 修复）** | **6.26ms** | **159.7** | ksplit=2 + pipeline=1 |

**恢复 0.33ms**（K-split 的验证收益）。四段全对，`faults=0`。

### 4.9 侦探链的元教训

1. **两次"代码存在性"假设都失败了**（v13 的 fold 代码 → 实为 gate 缺陷；v14 的 hot kernel 代码 → 实为 gate 错配）。
   两次真根因都是 **gate 相关**。
2. **"改默认值"是高风险操作**：`FileReplace` 用变量声明（如 `int v = 2`）作锚点时，
   同文件里若有多个同形声明，会静默改错对象。改 gate 默认值必须**按函数名/上下文锚定**，
   改完**读回确认**。
3. **hot-kernel-restore（499 删除）是错诊下的过度清理**——但清理本身无害
   （sparse-merge 确实中性、ringwin 未验证，都不值得保留），所以**不需要回滚**。

---

## 5. lut-floor-research 定案（突破 LUT-gather 地板的研究）

**背景**：200 tok/s 判定指出剩余 ~1.05ms 必须突破 **LUT gather 地板**
（gemm a32 物化 1.55µs/call × 246 + expert gateup LUT-gather）。

| 结论 | 内容 |
|---|---|
| **业界无"激活解码层"** | FlashInfer / DeepGEMM 等对 M=1 decode 没有任何把 LUT 解码当卖点的 kernel——本仓库的 e4m3 LUT 物化是自创路径，业界走 MMA 直吃 fp8/fp4。 |
| **swapAB 是唯一的数量级路径（2×）** | `mma.m16n8k32` A/B 对调：权重做 A（M=输出行）、激活做 B（N=token）。利用率 **1/16 → 1/8**（16 行 M 只 1 token 有效 → 8 列 N 只 1 token 有效）。布局**零转置**（权重已 row-major `[n,k]`、激活 k 连续 = col-major B 的列 0），per-32-block scale 方案**已存在**（dense kernel `:320-329`）。a32 物化 + LUT 解码**整体删除**。目标 ~1.1–1.5µs/call；预估步时 4.2ms ≈ **238 tok/s**。工作量：新 kernel ~150 行 + launcher + Rust dispatch + parity test（fuse 族迁移另计）。 |
| **a32 的 416x（实为 640x）块级冗余** | a32 物化是**每 block 重算**：416 blocks（n=3328, warps=8）× k=5120 = 2.13M 次解码，而该 token 只有 5120 个唯一值 ⇒ **416× 冗余**。生产大 shape（n=5120, warps=8）为 **640 blocks ⇒ 640× 冗余**。这才是 1.55µs/call 的真身。 |
| **cvt 解码被否证** | 用 `cvt.rn.f16x2.e4m3x2` 替代 smem gather：转换管吞吐 **16 results/clk/SM**（Table 4 "All other type conversions"）< shared memory 的 **32 值/clk/SM**（LDS.32）；且每元素还需 `cvt.f32.f16` 展回 f32 ⇒ ≤10.7 元素/clk/SM，**严格更慢**；位一致也无法保持（scale 重结合改舍入序）。 |

> ⚠️ **本节结论已被 serve 否证（2026-09-12 20:00，见 §3.5）**：上表预测 swapAB 是
> "唯一数量级路径（~238 tok/s）"，但 v17–v21 的四个变体 serve 全矩阵**全部中性**
> （6.18–6.20ms vs 基线 6.17ms）。隔离口径的 1.76–1.94x 在 serve 的 SM 争抢 + L2 竞争中
> 完全消失。**本节的"swapAB 是唯一路径"判断作废**——保留它是因为其中
> **a32 的 416×/640× 块级冗余诊断仍然成立**（那是 1.55µs/call 的真身），
> 只是"换成 MMA 就能兑现"这一步不成立。

**bf16-LUT 同族否证**：`s_lut` 由 256×f32 改 256×bf16 的提案**物理不成立**——
`*(bf162*)(lut + (b4&0xFF))` 取到 `LUT[c0]`/`LUT[c0+1]`，而 `o.y` 要的是 `LUT[c1]`（码是任意字节，无相邻关系）；
随机枚举实测 **49.8% 元素错**，直接破坏位一致。且 LDS 不减半、smem 512B 不跨驻留边界。**勿再提案。**

---

## 6. row-stationary 定案（a32 冗余的另一种修法）

**提案**：把 a32 解码改成 row-stationary——在寄存器里复用解码后的激活，消掉每 block 的重算（§5 的 416×/640× 冗余）。

| 维度 | 结论 |
|---|---|
| 可行性 | **可行**（数学上正确，布局不需重排） |
| 收益 | **< 1µs/call** —— 相比 swapAB 的 ~1.5µs/call 目标低一个量级 |
| 阻碍 1 | **寄存器 64 墙**：`gemm_fp8_gemv_kernel` 的 register cap（launch_bounds）只有 64；row-stationary 要把整行/多行解码结果常驻寄存器，直接撞墙 |
| 阻碍 2 | **epilogue 契约**：kernel 的融合 epilogue（B1 fp8 行发射 / rope / AR-v5 / xq 发射）要求特定的行布局与 thread 映射；row-stationary 的重排会破坏这些契约 |
| 判定 | **性价比低于 swapAB** —— 同为"消 a32 冗余"，swapAB 收益更高、且顺带删掉 LUT 解码；row-stationary 收益 <1µs 却要动 epilogue 契约，风险/收益比差 |

**结论：不做 row-stationary。** 原判定理由是"a32 冗余的正确修法是 swapAB（§5）"——
**该理由已因 swapAB 关闭而作废**（§3.5），但**结论不变且更强**：row-stationary 的
<1µs/call 收益同样要穿越隔离→serve 的翻译损失（7 次失效的先验），且还要破坏 epilogue 契约，
**收益低于噪声、风险高于收益，双重不该做**。a32 的 640× 冗余在当前架构下**保持不修**。

---

## 7. 方法论铁律（8 条）

### 7.1 八条铁律

1. **隔离探针只用于淘汰，正向收益必须 serve A/B。**
   隔离探针无法复现 serve 的四个条件：**SM 争抢（侧流并行）、L2 竞争、占用率敏感、graph replay 模式**。
   本会话共确认 **7 次**"隔离→生产失效"（完整清单与三分类见 §7.2）。
   **推论（本会话最贵的教训）**：**任何隔离收益不得直接写进路线图**——
   swapAB 正是凭隔离 1.76–1.94x 被写进"唯一数量级路径"，最终 4 个变体全部中性。
2. **fork_ev 是 kernel 级事件，不是 block 级。**
   `fork_ev` 在 EARLY kernel **完成时**记录 ⇒ 给 gating kernel 加任何工作 = 加到 main 的关键路径。
   小 kernel 合并的收益必须 **> gate 语义的代价**（省 launch 的收益 < 加到 gate kernel 的代价时是净负）。
3. **失败实验的代码立即物理删除，不留 gated-off。**
   gated-off 死代码也可能影响 ABI 与代码布局；且"gate 可探测 ≠ gate 生效"。
   **gate 必须验证"OFF 时是否真的回退"**（§4.5 的 quant-fold gate 缺陷即反例）。
4. **FFI 边界（.cu ↔ Rust）是原子性单位，必须同一 commit。**
   `git add -A` 在 subagent 并行工作时是危险的——docs 提交会扫入进行中的实施（§4.3 的 709 崩溃）。
   docs 提交用 `git add <specific-files>`。
5. **给热点 kernel 加运行时参数/分支 = 编译产物变重 = 回归风险（模板或独立 kernel）。**
   ⚠️ **修正**：v13/v14 的 +0.34ms 曾被归为此条，但 gate-hygiene-audit 证明真因是 **gate 错配**（§4.7）。
   此条作为**设计偏好**保留（参数少、编译期可消除更安全），但**不再作为那 0.34ms 的解释**。
6. **小 kernel 合并、folding 的收益分析必须考虑 gate/fork 语义**（见 2）。三条 fold（quant/swiglu/stamp）全败。
7. **位一致是硬约束。** 所有优化必须通过 fingerprint / 四段文本 / 逐位一致性验证。
   bf16 LUT 的 **49.8% 元素错**不可接受（§5）；swapAB 的 ks=1 逐位一致、ks>1 rel~5e-8 是它唯一通过的门槛。
8. **gate 卫生（最决定性）：改 gate 默认值时，`FileReplace` 必须按函数名/上下文锚定，改完读回确认。**
   用 `int v = 2` 这类裸变量声明作锚点会静默改错同名对象——
   这一个错误造成了 v13→v15 连续三轮、~0.37ms 的误诊与 499 行无谓清理。

### 7.2 隔离→生产失效的 7 次分类

| 类型 | 次数 | 案例 | 机制 |
|---|---|---|---|
| **测量有偏** | 3 | a32-vec4 / AR reduce grid / quant+swiglu fold | 分母是旧构型（P4 cp.async 已吸收）/ 理论高估 / 计数漏掉 fork_ev 关键路径 |
| **serve 条件改变** | 3 | PDEPTH pipeline / AR stamp fold / w2 L2 prewarm | 占用率在 serve 被暴露 / graph replay 机制崩 / PDL 窗口与 side stream 争 SM |
| **系统差异** | 1 | **swapAB**（最贵） | L2 争用使 staging 慢于隔离 + 混合形状下 SIMT 实际更快，净亏 |

**系统性差异权重排序**：**SM 争抢（最高）> L2 竞争 > graph replay 模式 > 构型漂移**。

### 7.3 "serve-faithful" 隔离协议（6 条，新提案）

若必须做隔离实验，必须满足以下 6 条，否则结果只能当**淘汰信号**，不能当收益证据：

1. **生产构型**——含所有已 ON 的前置优化（否则分母有偏，如 a32-vec4）。
2. **图模式 capture + replay**——禁止 direct-launch 计时（stamp fold 的 29.5s 灾难即图模式特有）。
3. **侧流干扰注入**——主动复现 SM 争抢（本会话最高权重的差异源）。
4. **L2 污染**——每轮换缓冲，禁止驻留红利（swapAB 的 staging 墙在 serve 更严）。
5. **关键路径计费**——gate/fork kernel 按 `fork_ev` 语义计入 main 关键路径。
6. **判据分层**——隔离只做淘汰；正向收益一律 **serve A/B + 四段文本 + `faults=0`**。

---

## 8. 关键地板清单（当前架构不可逾越）

| 地板 | 值 | 依据 |
|---|---|---|
| **gemm a32 物化** | 1.55µs/call（640× 块级冗余的 LUT gather + consume） | consume 地板 4.45µs（LDS 延迟受限）+ launch 0.71µs ≈ 5.2µs；生产 mean ~7.5–8µs → 246 次 ≈ 1.85–2.0ms。~~swapAB 可删掉此地板~~ **——swapAB 已关闭（§3.5），此地板在当前架构下保持不可逾越**。 |
| **expert gateup LUT-gather** | — | cp.async 无效已证 5 次；IPC 0.8/4，80% issue 槽停等（操作数供给受限，非 FMA 吞吐） |
| **expert down L1TEX** | — | 8 项理论优化全部失败（k-split/uint4/加 blockDim/降 rows…） |
| **AR NVLink 协议** | ~8µs/call（往返 ~6µs 是 stamp/poll） | 网格形状只影响 1–2µs；AR 总量 ~0.65ms |
| **CUDA graph node dispatch** | **0.411µs/node** | `kernels/cuda/graph_bench.cu` 实测；1800 节点 ≈ 0.74ms |

**已过时的旧地板**：roadmap 的 "5.3ms 真实工作地板" 写于 operand-supply 发现与侧流优化系列之前，不再成立。

---

## 9. 200 tok/s 的最终判定与前行路径

| 量 | 值 |
|---|---|
| **会话终态（v24a，serve 验证）** | **6.15ms / 162.6 tok/s（+115.4%）** |
| 目标 | 5.00ms / 200 tok/s |
| Gap | **−1.15ms 仍缺** |

### 9.0 终态每步分解（v24a 基线 nsys）

| 段 | 步时占比 | 状态 |
|---|---|---|
| **gemv**（`gemm_fp8_gemv` + `gemv_bf16_v2`，246 次） | **2.70ms（44%）** | SIMT **compute-bound**（isolated 11µs 不变）——a32-vec4 / P4 / swapAB×5 全部无效；**无路径**|
| **expert**（`expert_gateup` 24.1µs + `expert_down` 17.4µs） | **2.00ms（32%）** | **DRAM-bound**：0.9GB/步 ÷ 8TB/s = 0.11ms 地板，当前 18× 差距——**tcgen05 唯一目标** |
| **hc / AR / misc**（`hc_dots_late` / `hc_mixes_tail` / `ar_*` / misc） | **1.45ms（24%）** | 各自地板：hc 屏障、AR NVLink 协议（~8µs/call）、misc 已优化；**无路径** |
| 合计 | 6.15ms | 162.6 tok/s |

### 9.1 判定：旧框架已全部关闭，新框架只有"半个"

- **"逐 kernel 抠 SIMT 地板"框架**：全部已识别的小优化路径（节点削减 / gemv_bf16 / w2 warm /
  down cp.async / epilogue folding）即使全部兑现 → ~6.05ms ≈ 165 tok/s，**达不到 200**。
- **swapAB 换范式框架**：曾被判定为"唯一数量级路径（~238 tok/s）"，经 v17s→v24r **5 变体
  serve 全矩阵中性/回归**（§3.5），**正式关闭**——与它绑定的 ~1.8ms gemv 收益、a32 地板拆除、
  row-stationary 替代等**全部作废**。gemv 的 2.7ms **确认无路径**。
- **DL K-chunk 框架**：v22 serve 判**略负**（6.25 vs 6.17ms），第 10 次失败。
- ⇒ **会话结束时唯一存活的路径只剩 expert tcgen05（§9.2）**，且它只覆盖 32% 的步时。

### 9.2 tcgen05-perf-model 定案（200 tok/s 可达性）

> 把 expert 的 2.0ms 作为唯一变量，其余段（gemv 2.7 + hc/AR/misc 1.45 = 4.15ms）视为不可动，
> 推演 tcgen05 兑现后单请求 decode 步时的可达区间。

**天花板推演（expert = 唯一变量）**：

| 场景 | expert 步时 | 步时 | tok/s | 说明 |
|---|---|---|---|---|
| 当前（SIMT） | 2.00ms | 6.15ms | 162.6 | 基线 |
| tcgen05 折中 | 1.00ms | 5.15ms | **194** | 计划书原预期 |
| **tcgen05 最乐观** | **0.50ms** | **4.65ms** | **215** | 理论天花板（DRAM 地板 0.11ms 的 ~4.5×，已放宽） |

⇒ **即使 tcgen05 100% 完美兑现，天花板也只有 ~4.65–5.15ms ≈ 194–215 tok/s**，
仅仅"擦线"越过 200（且只有 0.5ms 场景才真正越过）。

**叠加 serve-translation 折扣后的现实预期**：

本会话已积压 **7 次隔离→生产失效 + 10 次 serve 验证失败/中性**（§7.2、§3.5），
`swapAB` 隔离 1.76–1.94× → serve **完全中性**是最新的、也是最贵的一次先验。
tcgen05 属**第 7 类系统差异**（profiled 与 served 的 SM 争抢 / L2 竞争 / 图 replay 不同），
计划书 §5 已自认其"翻译风险最高的一类"。按 **60–100% 折扣**折算：

- 乐观（折扣 60%）：expert 2.0 → 1.4ms ⇒ 步 ~5.55ms ≈ **180 tok/s**
- 中性（折扣 80%）：expert 2.0 → 1.6ms ⇒ 步 ~5.75ms ≈ **174 tok/s**
- 悲观（折扣 100%，同 swapAB）：expert 不变 ⇒ **162.6 tok/s**

⇒ **现实预期 ~170–175 tok/s，200 tok/s 触不到。**

**200 的数学（为什么必须"三者同时"）**：

- 需要：6.15 → 5.00ms = **−1.15ms**
- tcgen05 即使给出理论最大 **−1.5ms**（expert→0.5ms），也只是把天花板抬到 215 tok/s，
  且该点要求 expert 达到 DRAM 地板 4.5× 内、翻译零损失——两者都未被任何证据支持。
- 要在**现实折扣下**稳过 200，还须**同时**：
  1. **gemv 的 2.7ms 减半**（−1.35ms）——**无路径**（SIMT compute-bound，5 变体 swapAB 全中性）；
  2. **hc/AR 的 1.45ms 再砍**（−0.3ms+）——**已在地板**（AR NVLink 往返 ~6µs、hc 屏障、misc 已优化）。
- 三者**同时成立**的概率，按本会话失败率与翻译风险外推，属**研究级**，非工程量级。

### 9.3 结论一句话

**当前架构上 200 tok/s 很可能不可达**：唯一存活的 expert tcgen05 即使完美兑现，
天花板也只有 194–215 tok/s，考虑 serve-translation 折扣后**现实预期 170–175 tok/s**；
要过 200 必须 **tcgen05 + gemv 突破 + hc/AR 突破三者同时成立**——
而 gemv 与 hc/AR 两条在当前架构下**均已无路径**。**这是一个研究级目标，不是下一会话的工程量。**

---

## 附：本报告的口径与来源

- serve A/B：`scripts/dsv41_serve_ab.sh`，同二进制背靠背，判据 = 四段文本（Paris/Tokyo/1+1=/静夜思/出师表）
  逐字 + `faults=0` + p50。
- 本报告覆盖的 commit 区间：`9cddfb8` … `b5bed8a`（HEAD）。关键 commit：
  `4bed9f6`（gate 修复）、`166acc5`/`9696afa`/`b7744a1`（swapAB 三阶段）、`0f34001`（TMA Phase 1）、
  `6fc6113`（DL K-chunk）、`7fe3ede`（v22 DL K-chunk 判负）、`06a1285`/`297a79a`/`1b27bc6`/`ab8b2ca`
  （v23/v24 swapAB 环几何系列）、`003a4ab`（v24 定案，swapAB 绝对关闭）、`246d1be`（终态 nsys 分解）、
  `b5bed8a`（tcgen05 Phase 1 骨架 + 环几何否证回退）。
- 相关文档：`crates/ferrite-dsv41/STATUS.md`（逐 commit 记录，v17–v24 全部定案在此）、
  `docs/agent/dsv41-methodology.md`（方法论手册）、`docs/agent/dsv41-layer-fusion.md`
  （DL K-chunk 与 dots+LATE merge）、`docs/agent/dsv41-persistent-arch.md`（选举/持久化段核）、
  `docs/agent/perf-roadmap.md`（gemv/a32/swapAB 分析）、`docs/agent/dsv41-nsys-v14-plan.md`（nsys 分解）、
  `docs/agent/expert-tcgen05-plan.md`（tcgen05 四阶段实施计划）、`docs/agent/roadmap-200-tokps.md`（执行计划）。
- 未验证项标注：§2.2 三条默认 OFF 实现（swapAB/TMA 已 serve 判**中性**；DL K-chunk 已于 v22 判**略负**）；
  §9.2 tcgen05 的 194–215 tok/s 为**理论天花板**，170–175 tok/s 为**含翻译折扣的现实预期**，两者均未上机。
