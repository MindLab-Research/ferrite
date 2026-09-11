# DSV4.1 decode kernel 清单 v3 — post-CSE 基线（剖析口径 10.58ms / 生产口径 9.64ms）

**一句话**：在 **HEAD `db2917501`（gateup CSE + rmsnorm_q 融合）默认 env** 下采两次 nsys（`/tmp/dsv41-prof-v3` + 复测 `/tmp/dsv41-prof-v3b`，两次逐项一致）：
剖析口径 **10.58 ms/步/rank**，换回生产 v5 AR 口径 = **9.64ms**，与同机同 tree 的 serve arm `ab_cse`（9.55–9.60ms）吻合 **0.7%**。
**与 v2 的净变化只有 −0.07ms**，但内部有一条 **+0.31ms 的隐藏回归** 和一条 **−0.28ms 的 env 默认门变化**互相抵消——
**这条 +0.31ms（`expert_gemv_fp4_down_reduce`）是可复现的、可直接回收的，本表最大单项机会**。

> ⚠️ **与 v2 表不可直接逐行相减**：v2 采集时 tree 是 `60a01a5`，`DSV41_MIX_GATE` 默认还是 **ON**；
> `ef77bc7`（round-25 定案）已把它翻成 **OFF** 并成为生产默认。kernel **混合**变了（1 核拆 2 核），见 §3(A)。

---

## 0. 采集状态与口径（读之前必看）

| 项 | 值 |
|---|---|
| GPU 状态 | 16:48 检查 **空闲**（8×B300，0% / 0 MiB）→ 采集前 16:53 复核仍空闲。采集期间 16:50:09–16:51:00 曾有一个同伴 serve arm 占用 GPU（已结束，未与本次采集重叠） |
| 远端 tree | `/home/ubuntu/ferrite @ db2917501` — 与 `origin/main` **逐字相同**（`HEAD == FETCH_HEAD`），工作区干净（仅未跟踪的 `.build_id`）。**未执行 `git reset --hard`**：远端已有并发使用者，reset 会误伤其未提交改动 |
| 二进制 | `kernels/cuda/libferrite_kernels.so`（`build.sh 103a`，build_id `db2917501…-dirty+cu722df5804131bcb3`，nvcc **13.2**）+ `target/release/dsv41-run`，均 16:53 构建 |
| env | **HEAD 默认，无任何显式覆盖**：`GATEUP_FUSE=1`、`DOWN_FUSE=1`、`SHARED_TP=1`、**`MIX_GATE=0`**；剖析档另加脚本内置的 `DSV41_AR_V5=0`、`DSV41_GRAPH_STEP=0` |
| 采集 | `bash scripts/dsv41_profile.sh 30 /tmp/dsv41-prof-v3`（16:54–16:56）；**复测** `/tmp/dsv41-prof-v3b`（16:59–17:01） |
| 复测结论 | 两次**逐项一致**（top 表同名同值：`gemm_fp8_gemv` 9.5µs、`gemv_bf16` 22.4µs、`down_reduce` 24.9µs、`hc_mixes_tail` 12.4µs），总计 10.58 vs 10.53（差在 AR 伪影行）。**下表数字不是单次噪声** |

### 口径三定律（沿用 v2，仍然成立）

1. **AR 行是 host-barrier 伪影**：`ar_reduce 0.796 + ar_store 0.445 + ar_stamp 0.355 = 1.596ms` 是 `AR_V5=0` 路径。
   生产 v5 的绝对值**本次测不到**（v5 是 device-side 自旋，nsys 下 300x 病态）。
   本表沿用 v2 约定取 **0.66ms**，并按 §4 用 serve 实测**反证**了这个取值的正确性。
2. **"ms/步/rank" 是吞吐口径**：8 rank 求和再除 8；只有部分 rank 干活的代码收益会被平均掉（v2 的共享专家教训）。
3. **多设备 nsys 的绝对 per-call 不可全信**（脚本原话）：份额排序可信，绝对值需隔离微基准复核。
   **本次新增**：同一 population 里混着两种调用时，**任何单一统计量都会骗人**（见 §6 `gemv_bf16`）。

---

## 1. 生产口径 kernel 分解表（post-CSE，9.64ms）

`次/步` = 每步每 rank 调用数；`µs/次` = nsys **mean**（= 总时长/次数，聚合口径唯一正确的量）；非 AR 行 `ms/步` 取剖析值，AR 行取 v5 0.66。

| # | kernel | 次/步 | µs/次 | ms/步 | % | v2（MIX_GATE ON）| Δ |
|---|---|---|---|---|---|---|---|
| 1 | `gemm_fp8_gemv_kernel` | **246** | 9.5 | **2.33** | 24.2% | 206 / 9.6 / 1.98 | **+0.35** |
| 2 | `gemv_bf16_kernel`（共享专家 gate + lm_head + engram）| **49** | 22.4* | **1.10** | 11.4% | 9 / 45.0 / 0.41 | **+0.69** |
| 3 | `expert_gemv_fp4_batched_kernel`（gate_up+swiglu 融合）| 40 | 25.3 | **1.01** | 10.5% | 40 / 25.7 / 1.03 | −0.02 |
| 4 | `expert_gemv_fp4_down_reduce_kernel<true>` | 40 | **24.9** | **1.00** | 10.3% | 40 / **17.2** / 0.69 | **+0.31** ⚠️ |
| 5 | `hc_mixes_tail_kernel` | 80 | 12.4 | **0.99** | 10.3% | 同 | 0 |
| 6 | **AR v5**（store + pubred）| — | — | **0.66** | 6.8% | 0.66 | 0 |
| 7 | `hc_mix_dots_kernel` | 80 | 7.0 | **0.56** | 5.8% | 同 | 0 |
| 8 | `sparse_attn_pf_kernel` | 40 | 8.5 | **0.34** | 3.5% | 同 | 0 |
| 9 | `route_topk_kernel` | 40 | 5.3 | **0.21** | 2.2% | 40 / 5.2 | 0 |
| 10 | `quant_kernel<0>` | **126** | 1.5 | **0.19** | 2.0% | 166 / 1.6 / 0.26 | **−0.06** |
| 11 | `dsv41_hc_post_inplace_kernel` | 80 | 1.9 | **0.15** | 1.5% | 同 | 0 |
| 12 | **`rmsnorm_q_kernel`（新核）** | 40 | 3.2 | **0.13** | 1.3% | — | **+0.13** |
| 13 | `gemv_f32_kernel` | 7 | 16.4 | **0.12** | 1.2% | 同 | 0 |
| 14 | `apply_rope_kernel` | 88 | 1.3 | **0.11** | 1.2% | 同 | 0 |
| 15 | `rmsnorm_rope_kernel`（NR_FUSE）| 40 | 2.5 | **0.10** | 1.0% | 同 | 0 |
| 16 | `indexer_score_kernel`（Step A）| 4 | 19.7 | **0.08** | 0.8% | 同 | 0 |
| 17 | `quant_kernel<1>` †† | 40 | 1.6 | **0.07** | 0.7% | 同 | 0 |
| 18 | `engram_apply_kernel` | 2 | 32.6 | **0.07** | 0.7% | 同 | 0 |
| 19 | `argmax_kernel`（切片 + 跨 rank）| 1 | 59.1 | **0.06** | 0.6% | 同 | 0 |
| 20 | `add_kernel` | 40 | 1.4 | **0.05** | 0.6% | 同 | 0 |
| 21 | `swiglu_limit_kernel`（共享专家 w2 前）| 40 | 1.2 | **0.05** | 0.5% | 同 | 0 |
| 22 | `fp4_pack_kernel` †† | 40 | 1.1 | **0.05** | 0.5% | 同 | 0 |
| 23 | `ring_append_kernel` | 40 | 1.1 | **0.05** | 0.5% | 同 | 0 |
| 24 | `window_idxs_kernel` | 40 | 1.0 | **0.04** | 0.4% | 同 | 0 |
| 25 | `rmsnorm_kernel` | **4** | 1.9 | **0.01** | 0.1% | 44 / 2.8 / 0.12 | **−0.12** |
| 26 | 其余 14 项（comp/engram/embed/…）| — | — | **~0.29** | ~3.0% | — | 见 §5 |
| | **合计（生产口径）** | | | **9.64** | 100% | 9.69 | **−0.05** |
| | _合计（剖析口径，AR 用 host-barrier 1.60）_ | | | _10.58_ | | _10.65_ | _−0.07_ |

\* **`gemv_bf16` 的 22.4µs 是两个 population 的混合**：40 次共享专家 gate（≈17.3µs）+ 9 次 lm_head/engram（≈45µs）。
只报均值会把 45µs 那一族藏起来——这是 §6 的新陷阱。

† **`route_topk_kernel` 已被融合（`DSV41_ROUTE_FUSE`，默认 ON）**：route 由 gate GEMV
（`gemv_bf16_v2_kernel`，n=384 → **WPR=8 / rpb=1 / 384 blocks**，不是 WPR=4/192）的 **last-block
epilogue** 顺带完成，40 次独立 launch 与 40 个图节点消失。语义 bit-exact（同一份 f32 scores、同一
tie-break、同一 renorm）。真正省下的只有 launch/节点开销（route 的 ~3µs 执行时间仍在 gemv 末尾的
关键路径上）⇒ 预期 **−0.06~0.10ms/step**，而非表列的 0.21ms。仅 bf16-gate 路径可融（`DSV41_MIX_GATE=1`
的 fp8x2 gate 与 `DSV41_CUBLAS_M1=1` 保持两段式）。

†† **`quant_kernel<1>` + `fp4_pack_kernel` 已融合为 `quant_fp4_fused_kernel`（`DSV41_QUANT_FP4_FUSE`，默认 ON，
2026-09-11）**：`dsv41_quant_fp4` 现在一次 launch 完成"量化 + 打包"，中间 `g_q4nib` scratch（rows*cols 字节）
不再分配、不再读写，40 个图节点消失（80 → 40）。字节 bit-exact：同一份 amax/shfl 归约、
同一条最近 e2m1 查表、同一 `(lo&0xF)|(hi<<4)` nibble 装配（lo = 偶数元素）。**成立前提是 `block` 为偶数**
（nibble 对不会跨 block 边界，故 pair t 的落点是 block 的纯函数）；`block > 256` 或奇数 block 时 launcher
自动回退到旧两段路径（`DSV41_QUANT_FP4_FUSE=0` 是显式回退开关）。生产形状 rows=1/cols=5120/block=32 走融合路径。
⇒ 表列的 0.07 + 0.05 仍计在执行时间上，融合只回收 launch/节点开销（≈0.06ms/step 量级），
且省掉一次 rows*cols 的 scratch 往返带宽。

### `gemm_fp8_gemv` 246 次的代码级分解（2026-09-11 只读代码审计，`chain_dev.rs`）

单层 attention 侧 4 次（**2026-09-11 dual-chain 重构后复核，行号已更新**）：
`lin2(wq_a,wkv)`（`xn`, k=5120, n=1280+512, **mx2 已融合**）:2331；
`wq_b`（`qr`, k=1280, n=nlh·hd=4096；8 个 index-source 层用 IDX_FUSE mx2 带上 idx_wq_b）:2410/2507；
`wo_a` 循环 :2884/2922（`o`, k=hpg·hd=4096, n=olg=1024，`nlg=o_groups/world=8/8=`**1**）；
`wo_b` :2955/2990（`wo`, k=ol_local=1024, n=dim=5120）。
单层 MoE 侧 2 次（共享专家，SHARED_TP=1 全 rank）：`w1/w3` mx2 :3804（k=5120, n=2·sh_il=512）；
`w2` :3903（k=sh_il=256, n=dim=5120；`MOE_EPI_ADD` 默认 OFF 且 `MOE_DUAL` 下被强制关闭，
故走末行写 `s.ex_out` 而非 mx_add :3891）。

⇒ 6 × 40层 = 240，再加 engram 的 fp8 gemv :899 在 L1/L14 = **2** ⇒ **代码推导 242**。
与下表 246 差 4，且 `gemv_bf16` 反向差 −4（代码推 40 gate + 4 idx_wk + 8 idx_w + 1 HEAD_SLICE = 53 vs 表 49）：
**镜像差指向 4 次调用的符号归属，而非 4 个隐藏调用**——需一次逐符号 CSV 交叉核对定案，勿直接改动本表的 246。

**融合空间（同一审计）**：attention 侧"同激活 + 同 k"的对已被 A1(mx2)/A2(idx mx2)/M1(mx2) 用完 ⇒
**无任何可 mx2 合并的对**。剩余只有两类：① 链式两段核（wo_a→wo_b，k 4096→1024，跨 stage 同步；
wq_a→wq_b / w1w3→w2 同构但需 grid 级同步，不可行），② 异激活融合需先加第二 fp8 激活缓冲
（`s.xq/xsc` 是**单缓冲**，T1/T2 一次性旗标建立在"只存最近一次量化"之上），且 mx2 只有单一 k。
⇒ 单层 6 次 → 1 次只能走 Stage C persistent 段核。

**dual-chain 重构后的复核（2026-09-11，结论：旧结论仍成立，但理由变了）**：`DSV41_DUAL_CHAIN`
在 `lin2(wq_a,wkv)`（:2331）**之后**才 fork（:2365-2368），因为 `lin2` 已把 wkv 与 wq_a 一起算完
（`kv_early`），所以 **kv 侧链上根本没有 gemv** —— 只有 `rmsnorm_rope`（:2585）一个非 gemv 的小核。
⇒ "wq_b ∥ kvb 异激活合并"的候选**前提不成立**（kvb 早已并入 fork 前的 mx2）。MoE 侧同理：
routed 链上是 fp4 expert 核（`expert_gemv_fp4_*`），shared 链上是 w1/w3 mx2 + w2，
**没有任何一条跨流 gemv 对**可供合并；两条链各自的 gemv 都是"链内依赖"（w1w3→swiglu→w2）。
⇒ 每层 5 次的唯一路径仍是 Stage C 段核（或把 wo_a→wo_b / w1w3→w2 做成 grid-sync 两段核）。

### 家族汇总（生产口径 9.64ms 为分母）

| 家族 | ms/步 | % | 备注 |
|---|---|---|---|
| **GEMV 族**（gemm_fp8 + gemv_bf16 + gemv_f32）| **3.54** | **36.7%** | 见 §5：91% 是固定成本 |
| expert fp4 家族（gate_up + down_reduce）| **2.01** | 20.8% | 其中 0.31 待回收（§3B） |
| hc 链（tail + dots + hc_post_inplace + collapse_norm）| **1.70** | 17.7% | tail 0.99 是 warp0 串行链 |
| attention/norm/quant 杂项（sparse/quant/rope/rmsnorm*/indexer）| **~1.15** | 11.9% | |
| AR v5 | 0.66 | 6.8% | 协议地板（口径分歧见 §4）|
| 其它 14 小项 | ~0.29 | 3.0% | |

---

## 2. v2 → v3 的 delta 归因（账要能对上）

两次采集之间，`kernels/` 下**只有 4 处**代码变更（`git diff 60a01a5..HEAD -- kernels/`）：

| 变更 | commit | 实测效果（µs/次）| Δms/步 |
|---|---|---|---|
| **(A) `MIX_GATE` 默认 ON→OFF** — 混合核不再被调用；gate 回 `gemv_bf16`，共享专家 w1/w3 回 `gemm_fp8_gemv` | `ef77bc7` | fp8x2 **40 次消失**；gemv_bf16 9→**49**；gemm_fp8_gemv 206→**246** | **−0.28** |
| **(B) down 方向 vec-320 循环**（`nv8 = k>>8` 的 256 值/组 uint32 主循环）| `01291b2` | `down_reduce` 17.2 → **24.9** | **+0.31** ⚠️ |
| **(C) gateup CSE**（`sa[16]` 一次装载喂两条 FMA 链）| `db2917501` | gate_up 25.7 → 25.3 | **−0.02** |
| **(D) `rmsnorm_q_kernel`**（rmsnorm + fp8 量化融合，同 commit 顺带上车）| `db2917501` | rmsnorm 44→4 次；quant<0> 166→**126** 次；新核 40 次 | **−0.05** |
| | | **合计** | **−0.04** |

实测总差 −0.07（10.65→10.58），余量在 AR 伪影行的抖动（−0.03）。**账对得上。**

### (A) 的再解释：v2 的 #2 kernel 不是"消失"，是被拆成两个更快的
- v2：`gemv_bf16_fp8x2` 40 次 × 33.0µs = **1.32ms**（gate + 共享专家 w1/w3 一次 launch）
- v3：40 次 `gemv_bf16`（≈17.3µs，bf16 gate）+ 40 次 `gemm_fp8_gemv`（9.5µs，fp8 w1/w3）= **1.07ms**
  ⇒ **−0.25ms**，与 round-25 的 serve A/B 方向/量级一致（同 tree 连续两 arm：`mg0` 9.47 vs `mg1` 9.78 = **−0.31**）。
  **结论**：混合核（即使 `9e4e3df` 给它补了 LUT+a32）仍是净亏，MIX_GATE=OFF 是对的。

### (B) 已定案：nv8 是**寄存器/占用率**回归，已回退 ⚠️→✅
- 事实：同一个核、同样 40 次/步，per-call **17.2 → 24.9µs（+45%）**，**两次独立采集复现**；同期未变的核（argmax 59.1/59.1、indexer_score 19.8/19.7、hc_tail 12.4/12.4）**没漂**，所以不是机器/温漂。
- 该 commit 的注释声称新循环把"整条 per-slot dot 从 5 次标量尾巴"换成"1 次 uint32 迭代 + 1 次尾巴"（**指令更少**），理论上应更快。实测相反。
- **隔离微基准定案（本机无 nvcc/GPU ⇒ 远端 nvcc 13.2，形状 `dim=7168,k=320,slots=8,vec=2`，证据 `/tmp/dv320/`）**：

  | 版本 | regs/thread | blocks/SM | waves | per-call |
  |---|---|---|---|---|
  | `01291b2^` | 40 | 6 | 1.01 | 26.1µs |
  | `01291b2` | **54** | **4** | **1.51** | **38.8µs（+49%）** |
  | 回退后 | 40 | 6 | 1.01 | 26.1µs（逐位复原） |

  **根因 = 占用率悬崖**：+14 寄存器把驻留从 6 块/SM 压到 4 块/SM，896-block 的 grid 从 1 个
  wave 变 1.5 个 wave（1.51× ≈ 1.49× ≈ nsys 的 1.45×）。**这个核是占用率/延迟受限，不是
  issue 也不是 smem**；`LDG.U8` → `LDG.32` 省下的指令远小于丢掉的并行度。
- **附带**：01291b2 还改了 k=320 的求和顺序，而 `expert_gemv_fp4_batched_kernel` 的 vec==2
  分支（`DOWN_FUSE=0` 回退路径）没有同步 ⇒ 破坏了 `:1015-1029` 的逐位一致契约。回退后复原。
- **矛盾解释**：serve A/B 判 "neutral"（9.42 vs 9.38）落在 §3 的 ±0.3ms 机时漂移带内，
  **不是有效反证**。nsys 的逐核 per-call 才是可信信号。

### (C) 的教训：这个核不是 smem-load-bound
CSE 把每 lane 每组的 `LDS.32` 从 32 降到 16（源码 + SASS 双确认），**预期 −0.15ms，实收 −0.02ms（12%）**。
⇒ `expert_gemv_fp4_batched` 的关键路径**不在 LDS**。下一刀只能打 **FMA/issue**（4 累加器→2、或换 warp 内归约），
或者打 **launch 数**——继续在 smem 侧找收益是白费（v2 的那条"再省一遍 LUT/a32"已经失效，这条现在也失效了）。

### (D) 被 commit message 藏起来的第二个优化
`db2917501` 的标题只讲 CSE，但同一个 commit 还塞进了 `rmsnorm_q_kernel`（`dsv41_kernels.cu` +67 行、`device.rs` +39 行）：
40 次 rmsnorm_q 吃掉了 40 次 rmsnorm + 40 次 quant<0>，净 **−0.05ms**。
⇒ **看 diff，别只看 commit message。** 另外这直接回答了 `0877f04` 的悬案："rmsnorm_q_kernel 已存在但未接入"——**实测证明它已接入且在跑**（40 次/步）。

---

## 3. 与 serve 实测交叉验证（本节的用处：定 AR 口径）

| 来源 | 数值 | 说明 |
|---|---|---|
| 本表生产换算 | **9.64ms** | 10.58 − 1.596(host-barrier AR) + 0.66(v5) |
| 同 tree serve arm `ab_cse`（16:50，HEAD `db2917501`）| **9.55–9.60ms**（p50≈9.57）| 差 **0.7%** ✅ |
| v2 的同类校验（先例）| 9.69 换算 vs `ab_fon` 9.73 | 差 0.4% ✅ |
| **若按 `86af349` 的 AR=1.49ms 口径** | 10.47ms | 与 9.57 **差 9%** ❌ |

**结论**：**`86af349` 提出的"AR 真实成本 1.49ms（比 0.66 低估 2.3x）"应判为 host-barrier 口径误用**——
`ar_reduce/ar_store/ar_stamp` 是 `AR_V5=0` 的 host barrier 路径，脚本自己就警告"多设备 nsys 的绝对 per-call 不可信"，
把它当成生产成本会**高估 0.9ms 并错排 Stage C 的优先级**。生产 AR 仍是 **0.66ms**（本节的 serve 反证）。
`86af349` 的**结构性**论点仍然成立且有用：**AR 每步 246 个图节点**（82 次 × 3 核），这是 Stage C persistent 的最强论据——只是它的**时间**不是 1.49ms。

⚠️ **机时漂移警告**：同一台机 16:25–16:50 内同伴的 8 个 arm 落在 **9.46 / 9.47 / 9.49 / 9.57 / 9.78** 之间。
**±0.3ms 内的单 arm 比较没有意义**（例如不能用 `ab_cse` 9.57 去反驳 STATUS 的基线 9.38）。本表因此只做**同一采集内**的逐核比较。

---

## 4. Top 剩余机会（按可回收量排序）

| 排名 | 机会 | 目标 | 预期 | 依据 / 风险 |
|---|---|---|---|---|
| **1** | **回收 (B) 的 0.31ms**：✅ **已完成**——`nv8` 循环已回退（保留 gateup CSE），隔离微基准 38.8 → 26.1µs 复原 | expert fp4 家族 2.01ms | **+0.31** | 已完成；回退时一并恢复了与 `DOWN_FUSE=0` 路径的逐位一致 |
| **2** | **GEMV 族 launch 数**：`gemm_fp8_gemv` **246 次/步 × 9.5µs = 2.33ms（24.2%）**，约 **91% 是固定成本**（per-call 已到 9.5µs）。**v2 的 "206 次" 基线已作废** | 2.33ms | **−0.25~0.5** | xn-megafuse 只能按 246 计；`s.xn` 复用缓冲让"5 族一 launch"不可能（v2 已驳回）|
| **3** | **hc 链**：tail 0.99（warp0 串行链，**探针已否决**体内重叠：可藏窗口 0.46µs ≪ sinkhorn 6.5µs）；`hc_post_inplace` 0.15 | 1.70ms | **−0.15~0.25** | 只剩跨层流水 / Stage C 段核 |
| **4** | **AR 节点数**（不是 AR 时间）：246 节点/步，生产 0.66ms 是协议地板 | 节点尾延迟 | **−0.3~0.5** | Stage C persistent 把 3 核/层 → 1 核/段；**建议先用图节点数直接量残余**（v2 遗留待办）|
| **5** | **FMA-side 打 gate_up**：CSE 已证明 smem 不是瓶颈 | 1.01ms | **−0.1~0.2** | 风险：数值契约（4 累计器→2 改变求和顺序 ⇒ 需 parity 测试）|
| **6** | **`gemv_f32` 的 v2 化**（compressor 的 kvp/scp，n=128×k=5120 / 7 次/步）| 0.12ms | **−0.08~0.10** | ✅ **已落地**（2026-09-11）：v1 只有 16 blocks / 128 warps（148 SM 的 11%）+ 160 次串行 4B load ⇒ 16.4µs = 48x 内存地板（0.34µs），与 gate 修前同病。新增 `gemv_f32_v2_kernel`（`dsv41_glue.cu`，模板 WPR：float4 16B/load + K-split + smem fold，`__fmaf_rn` 钉住 FFMA 舍入）→ n=128 走 WPR=8 = 1024 warps。`device.rs::gemv_f32` 按 `n < GEMV_F32_V2_MAX_N=2048` 分派，`DSV41_GEMV_F32_V2=0` 回退 v1。**待实测**：16.4µs 的改善幅度（预期对齐 bf16 gate v2 的 3-5µs 档）|

---

## 5. 小核全清单（< 0.1ms，防漏算）

| kernel | 次/步 | µs/次 | ms/步 |
|---|---|---|---|
| `indexer_score_kernel` | 4 | 19.7 | 0.079 |
| `quant_kernel<1>` ††（已并入 `quant_fp4_fused_kernel`）| 40 | 1.6 | 0.065 |
| `engram_apply_kernel` | 2 | 32.6 | 0.065 |
| `argmax_kernel` | 1 | 59.1 | 0.059 |
| `add_kernel` | 40 | 1.4 | 0.054 |
| `swiglu_limit_kernel` | 40 | 1.2 | 0.049 |
| `fp4_pack_kernel` ††（已并入 `quant_fp4_fused_kernel`）| 40 | 1.1 | 0.046 |
| `ring_append_kernel` | 40 | 1.1 | 0.045 |
| `window_idxs_kernel` | 40 | 1.0 | 0.042 |
| `comp_placeholder_kernel` | 30 | 1.0 | 0.030 |
| `indexer_topk_kernel`（Step B）| 4 | 6.8 | 0.027 |
| `compressor_pool_kernel` | 4 | 5.4 | 0.021 |
| `engram_hash_step_kernel` | 1 | 17.7 | 0.018 |
| `embed_expand_dev_kernel` | 1 | 11.8 | 0.012 |
| `compress_commit_kernel` | 4 | 2.1 | 0.009 |
| `rmsnorm_kernel`（残余）| 4 | 1.9 | 0.008 |
| `dsv41_hc_collapse_norm_kernel` | 1 | 6.0 | 0.006 |
| `engram_gather_kernel` | 2 | 2.7 | 0.005 |
| `index_k_publish_kernel` | 4 | 1.4 | 0.005 |
| `compressor_state_kernel` | 3 | 1.7 | 0.005 |
| `bf16_to_f32_kernel` | 0 | — | 0.000 |

⚠️ `kpool_compress` 仍不在本 profile 里（它只服务 batched DSA 链 `cuda.rs:4645`，与单序列 `sparse_attn_pf_kernel` 不是同一条数据链）。

---

## 6. 陷阱与注意事项（改这块代码前必读）

- ⚠️ **mean 还是 median**：nsys 报告同时给 mean/median。**聚合 ms/步必须用 mean×次数**（总时长是相加的）。
  `gemm_fp8_gemv` 的 median 7.9µs vs mean 9.5µs **差 0.39ms**（= 全表的 4%）。
  `86af349`/STATUS 表用的是 median，**不要把那张表和本表逐行比**。
- ⚠️ **异质 population 不能报单一 per-call**：`gemv_bf16` 现在 49 次里混着 ≈17.3µs（共享专家 gate）和 ≈45µs（lm_head/engram）两族。
  想优化 lm_head 就不能看 22.4µs 这个均值。
- ⚠️ **env 默认门翻转会改变 kernel 混合**：`MIX_GATE` ON→OFF 让一个核换成两个核。
  **跨 commit 比较 profile，等于同时比较 env 默认值**。本文头部必须记 env，v2 就是踩了这个（它的表是 MIX_GATE=ON 口径）。
- ⚠️ **一个 commit 可能塞进多个优化**：`db2917501` 标题只写 CSE，实际还含 `rmsnorm_q` 融合（−0.05ms）。
  归因时一律 `git show <commit>` 看 hunk。
- ⚠️ **`/tmp/dsv41-prof-v3` 是 v2 的证据目录**。本次采集前后已把 v2 的数据改存为
  **`/tmp/dsv41-prof-v2-mixgate`**（勿删）——否则 v2 文档的全部数字失去溯源。
- ⚠️ **远端是共享机器**：采集窗口内检测到同伴的 serve arm（16:50:09–16:51:00 GPU 被占）。
  **profile 前必须 `pgrep -x dsv41-run`**（脚本自身会拒绝启动，但要自己先看）。
- ⚠️ **长任务必须 detach**：agent 的 ssh 调用会在 120s 被杀，但远端进程会存活——
  用 `setsid nohup … > log 2>&1 < /dev/null &` + 轮询日志，别依赖前台返回。
- ⚠️ **本机 nvcc 是 13.2**：`sm_103a` 的 SASS 只能在远端复现；CSE 那类"源码等价、SASS 不等价"的判断**不能在本机验证**。

---

## 7. 复现 / 待实测

```bash
# 1) 检查 GPU 独占（远端有并发使用者时尤其重要）
ssh ubuntu@43.202.208.136 'pgrep -x dsv41-run || echo idle'
# 2) 同步到最新 main（远端已有干净 tree 时不要 reset，避免误伤并发使用者）
cd ~/ferrite && git fetch origin main -q && git rev-parse HEAD FETCH_HEAD   # 应相等
# 3) 重建（.so 与 binary 都要：build id 嵌在 binary 里）
(cd kernels/cuda && bash build.sh 103a) && cargo build --release
# 4) 采集（env 全默认 = 生产口径）
setsid nohup bash -c 'cd ~/ferrite && bash scripts/dsv41_profile.sh 30 /tmp/dsv41-prof-v3c > /tmp/prof_v3c.log 2>&1' </dev/null >/dev/null 2>&1 &
# 5) 全量差分（脚本只印 top-15；/tmp/v3data/kdiff.py 印全部 38 行）
python3 kdiff.py /tmp/dsv41-prof-v3c/one.csv /tmp/dsv41-prof-v3c/many.csv 30
```

**待实测清单**（本次未能定案的，按优先级）：

0. **a32 / 占用率（−0.74ms 的赌注）**：`gemm_fp8_gemv_kernel` 的 M=1 路径带一张 block 级
   预解码激活表 `s_af`（"a32"，k×f32 = 20KB @ k=5120），它把 smem 从 ~27.4KB 抬到 ~47.4KB
   （k=5120/warps=4：mode4 48512B，去掉 a32 28032B），即 blocks/SM 4 → 8。a32 的
   −6/−8/−13% 只在 n=256/1024/1664 探针上测过，从未在生产 k=5120 复测。
   **开关语义（易错，已核码）**：`DSV41_GEMV_FP8_MODE` 0=标量 / 1=向量化 / 3=保序分段 /
   4=保序分段+块级**激活** staging（`s_a`，k 字节）。**a32（`s_af`，4k 字节）在 mode 3 也物化**
   （mode 3 只是把 fp8 激活从 global 读而非读 `s_a`）⇒ `MODE=3` **不是** a32 开关。
   a32 的独立门是 **`DSV41_GEMV_A32`**（1=默认保留；0=丢弃，逐位等价——同一乘积
   `s_lut[ap[j]] * s_as[j>>5]`，只是内联回消费循环；smem 少 20KB）。
   工具：`scripts/dsv41_a32_bench.sh`（隔离基准，打印每 shape 的 smem/blocks-per-SM/µs +
   跨 arm 指纹校验）+ `scripts/dsv41_recovery_verify.sh`（哨兵→base→A/B 一键）。
   `dsv41_a32_bench.cu` 经 `dsv41_gemv_gsmem` / `dsv41_gemv_occupancy` 两个 host 探针把
   "20KB⇒4→8 blocks/SM" 从估算变成实测。
1. **`down_reduce` +0.31ms 的定案**：隔离微基准（同一 kernel，HEAD vs `01291b2^`），或 revert 后重采 profile。**这是唯一挡住 0.31ms 回收的事。**
2. **AR v5 的隔离绝对值**：0.66（v2 约定，被 §3 反证支持）vs 1.49（`86af349`，host-barrier 口径）。需要 device-side v5 的隔离测量。
3. **每步图节点数**：`hex/window` 类小核 + AR 246 节点 + 节点尾延迟（≈0.9ms，v2 遗留）——用 `cuda_gpu_trace` 或图节点数直接量。
4. **`gemv_bf16` 两族拆分**：把 9 次 lm_head/engram（≈45µs）单独归因，看 HEAD_SLICE 是否还有空间。
5. **`gemm_fp8_gemv` 的 mean/median 尾巴**（9.5 vs 7.9µs）来自哪几步：若是前几步 warmup，则生产口径应再降。

---

_事实来源：`/tmp/dsv41-prof-v3` + `/tmp/dsv41-prof-v3b`（本次两次 nsys，16:54–17:01，tree `db2917501`，HEAD 默认 env）；
`/tmp/dsv41-prof-v2-mixgate`（= 原 `/tmp/dsv41-prof-v3`，v2 文档的采集目录，tree `60a01a5`，MIX_GATE=ON）；
serve 交叉验证 `/tmp/ab_cse.log`（9.55–9.60）、`/tmp/ab_mg0|mg1`（MIX_GATE A/B）、`/tmp/ab_dv320`、`/tmp/ab_r27`；
代码差 `git diff 60a01a5..db2917501 -- kernels/`；
`crates/ferrite-dsv41/STATUS.md`（`86af349` 的 nsys-v3 段、`530153f` 的 round-29/30 判定、round-28 的 dv320）；
`docs/agent/dsv41-kernel-inventory-v2.md`、`docs/agent/roadmap-200-tokps.md`、`docs/agent/stage-b-execution.md`。_
