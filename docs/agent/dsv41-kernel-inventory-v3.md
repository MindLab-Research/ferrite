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
| 4 | `expert_gemv_fp4_down_reduce_kernel<true>` | 40 | **24.9** | **1.00** | 10.3% | 40 / **17.2** / 0.69 | **+0.31** ⚠️✅ |
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
| expert fp4 家族（gate_up + down_reduce）| **2.01** | 20.8% | ⚠️ 此为 **db2917501 profile 口径**（早于 667c6f6 的回退）⇒ 含 0.31 回归；代码已回退，实际应 ≈ **1.70**（待重采确认，§3B/§4 rank1） |
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

#### (B-修正, 2026-09-11 晚)：**「占用率红线」不成立；4 值 uint16 才是正确宽度**

在 bench 节点用同一个隔离微基准（`/tmp/dv320`，生产形状 `dim=7168, k=320, slots=8, 256 线程,
896 块`，5 轮交错；基线 = 2 值尾循环）重测了三种宽度：

| 臂 | regs | blocks/SM | waves | 相对基线 |
|---|---|---|---|---|
| 2 值尾（mode 2，基线）| 40 | 6 | 1.01 | 1.00 |
| **4 值 uint16（mode 3）+ `__launch_bounds__(256,6)`** | 40 | 6 | 1.01 | **0.90** |
| **4 值 uint16（mode 3，无 launch_bounds）** | 56 | 4 | 1.51 | **0.87（最快）** |
| 8 值 uint32（mode 4 = 01291b2 的形状）| 40 | 6 | 1.01 | 1.03 |
| 8 值 uint32（mode 4）| 62 | 4 | 1.51 | 0.97 |

- **输的是「8 值」这个宽度，不是「宽加载」**：4 值/lane 的工作集（1×uint16 + 1×float4 +
  2×float2 L1TEX 操作）仍能塞进 40 寄存器的调度；8 值省下的指令抵不过它吃掉的寄存器。

> 🚨 **本节下方「56 regs / 4 blocks/SM 是全场最快 ⇒ 占用率归因不成立」的结论已于 2026-09-12 推翻。**
> 该结论只对**这个隔离微基准**成立：微基准把同一组 buffer 连跑 N 次，整个工作集
> （w2 = 8 slots × dim × k/2 = 9.17 MB + 40KB act）**常驻 L2**；生产里 w2 在本步此前从未被读过
> （gate/up 只读 w1/w3），9.17 MB/层 × 40 层 = 367 MB/步 ≫ L2，**必然来自 HBM**。
> 实测 9.17 MB / 23.8 µs = 385 GB/s，远低于 HBM 峰值 ⇒ 生产是**延迟受限**，此时
> 常驻线程数（4 vs 6 blocks/SM = 1024 vs 1536 threads/SM）与 wave 数（1.51 vs 1.01）才是决定项。
> 所以「隔离 0.87x 更快」与「生产 +38% 更慢」可以同时为真——**两者测的不是同一个内核状态**。
> 见 §4 的 down 定案与 `dsv41_experts_mxf4.cu` 的 `__launch_bounds__` 注记。

- ~~**56 regs / 4 blocks/SM 那一臂是全场最快** ⇒ 上面「40 寄存器是单 wave 红线，多寄存器必然
  +49%」的归因**不成立**。01291b2 的 +45% 与占用率无关~~ —— **错**。01291b2 的 40 → 54 regs 就是
  6 → 4 blocks/SM = 1.01 → 1.51 waves 的悬崖，与 nsys 的 +45% 吻合；隔离微基准没有复现生产的
  缓存状态，因此不构成反证。**教训：任何用「连跑同一组 buffer」的隔离基准去否定占用率/带宽归因
  之前，先确认它的工作集是否落在 L2 内；L2-hot 基准天然对占用率不敏感。**
- **已落地**：`vec == 3` = 4 值/lane（1×`LDG.U16` 权重 + 1×`LDS.128` 激活 + 2×`LDS.64` LUT，
  1.25 op/值 vs 原 2.5；scale 覆盖整 4 值组 ⇒ 每组只做一次 scale-FMA）。两个核
  （`expert_gemv_fp4_down_reduce_kernel` 与 `expert_gemv_fp4_batched_kernel`）各有一份
  **逐字镜像**，闸门 `DSV41_DOWN_VEC4`（默认 ON，`=0` 回退到 `DSV41_EXPERT_FP4_MODE`）。
  只作用于 down 启动：gate/up 的 swiglu 融合要求 `mode == 2`。
- ⚠️ **寄存器分配是 per-function**：vec==3 分支的 56 regs 覆盖整个
  `expert_gemv_fp4_down_reduce_kernel`，**连 vec==2 实例也一起被拖到 4 blocks/SM** ⇒
  `DSV41_DOWN_VEC4=0` 并不能恢复占用率（分支还在函数里）。唯一恢复手段是
  `__launch_bounds__(256, 6)`（把 regs 压到 42 上限；bench 上该臂实测 40 regs 无 spill）。
  这正是 `9fe0766` 引入、`287d2b7` 未修复的 down_reduce 17.2 → 23.8 µs 回归的根因。
- ⚠️ mode 3 的 lane→k 映射与 mode 2 不同 ⇒ **融合/未融合的逐位一致契约靠「镜像分支」维持，
  改一边必须同时改另一边**（这正是 01291b2 回退的第二个理由，现在用镜像解决了）。

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
| **5** | **FMA-side 打 gate_up**：CSE 已证明 smem 不是瓶颈 | 1.01ms | **❌ 否决：预期 ≈0**（2026-09-11 复核，见 §4.2）| 原估 −0.1~0.2 不成立：issue 利用率仅 ~8%、warp 数受 grid 限制 ⇒ 减指令/减寄存器都换不到时间；且 4→2 改求和顺序 = 纯 parity 风险 |
| **6** | **`gemv_f32` 的 v2 化**（compressor 的 kvp/scp，n=128×k=5120 / 7 次/步）| 0.12ms | **−0.08~0.10** | ✅ **已落地**（2026-09-11）：v1 只有 16 blocks / 128 warps（148 SM 的 11%）+ 160 次串行 4B load ⇒ 16.4µs = 48x 内存地板（0.34µs），与 gate 修前同病。新增 `gemv_f32_v2_kernel`（`dsv41_glue.cu`，模板 WPR：float4 16B/load + K-split + smem fold，`__fmaf_rn` 钉住 FFMA 舍入）→ n=128 走 WPR=8 = 1024 warps。`device.rs::gemv_f32` 按 `n < GEMV_F32_V2_MAX_N=2048` 分派，`DSV41_GEMV_F32_V2=0` 回退 v1。**待实测**：16.4µs 的改善幅度（预期对齐 bf16 gate v2 的 3-5µs 档）|

### 4.2 gateup 的 FMA 累加器结构（2026-09-11 分析）：**4 累加器→2 否决**

**当前结构**（`dsv41_experts_mxf4.cu:1060-1232`，`ILV=true` 且 fuse 默认 ON ⇒ 这是生产路径；
`k = dim = 5120` → `nv2f = k>>9 = 10` 组/lane；`n_total = inter`，TP8 下 320 行/slot，6 slot）：

- **组内 4 个临时累加器**，不是跨组累加器：gate 链 `gp0..gp3`（`:1131`），每个串 **4 个 `fmaf`**（链深 4），
  再 `(gp0+gp1)+(gp2+gp3)` 两两树（3 个 FADD），最后 `g = fmaf(gsc, ·, g)`（`:1156`）折进**唯一的**跨组累加器 `g`。
  up 链 `up0..up3` 同构（`:1166-1191`）。⇒ 每组每链 ~20 条 FMA 类指令、组链深 ≈ 4+2+1 = 7；10 组串行 ⇒ `g` 链 ≈70。
  每 lane 每组还有 16×`LDS`(sa) + 16×`LDS.64`(LUT) + 1×`LDG.128`(ILV) ⇒ ~105 条指令/组/lane。

**为什么 4→2 是死路（三条独立证据）**：

1. **核不是 FMA/issue 绑定**：~1050 条指令/lane-row × 1920 warp = 2.0M warp-instr ÷ 148 SM ÷ 4 issue/cycle
   ≈ **3.4K cycle ≈ 1.9µs**，实测 **25µs** ⇒ issue 利用率 **~8%**，与源码注释里的"issue ~6%、
   94% 周期停在 K 的 LDG"（`:994-999`）完全一致。4→2 只省 4 个寄存器 + 每链每组 2 个 FADD（**~4% 指令**），
   在 8% 的 issue 占用下**不可能量出来**。
2. **寄存器也换不到占用率**：1920 warp / 148 SM = **13 warp/SM**，是 **grid 造成的 warp 饥饿**
   （240 CTA / 148 SM = 1.62），不是寄存器上限 —— 这正是 §4.1 已经论证过的同一件事。
   减寄存器/加 unroll 都无法凭空多出 warp。
3. **反向（4→8）也不成立**：缩链深只对 latency 链敏感的核有用；这里的停等是 LDG 延迟而非 FMA 相关链。

**结论**：4→2（以及任何"FMA-side 减指令/减寄存器"）**预期收益 ≈0** ⇒ 不值得用 parity 风险去换。
唯一能**同时**加 warp 数与 MLP 的杠杆是 **K-split**（把一行的 K 切给 2 个 warp），它才是正对着
"94% 停 LDG"的那把刀 —— 但注意 K-split **天然改求和顺序**（partial_A + partial_B ≠ 原 10 组链序），
所以它不是"先看累加器"能顺带解决的，需要独立的逐位对拍设计（或接受 tolerance A/B）。

⚠️ **顺带发现：源码契约注释已失效**（`:973-980`）说"each K walk below is the vec==2 shape of the unfused body
… same **single scale multiply per accumulator**, so gate/up accumulate to the **same floats** the unfused rows do"。
**这句与现在的代码不符**：unfused vec==2 体（`:1134-1183`）是 `a0..a3` **跨组持久**、每组各做 4 次 scale-FMA，
最后 `acc = (a0+a1)+(a2+a3)`；fused 体是**每组 1 次** scale-FMA 作用在 4 元树上。
`7100ebfe` 写下该注释，`667c6f66` 重写了 fused 体却留着注释（`git blame :974-980` = 667c6f66）。
⇒ 任何"改 accumulated 顺序仍逐位一致"的推理**不能引用这段注释**，两边本来就不逐位。
（`tests_tcgen05_mxf4.cu::run_ilv_case` 只对 ILV vs plain 逐位，不覆盖 fused vs unfused。）

### 4.1 `DSV41_GATEUP_ROWS`（2026-09-11 已落地）：行拆分**不是**占用率修复——它是该假说的证伪探针

**代码**（`kernels/cuda/dsv41_experts_mxf4.cu`）：gate/up batched launcher 的 CTA 形状从硬编码
`warps=8` 改为 `dsv41_gateup_rows()`（env **`DSV41_GATEUP_ROWS`**，默认 **8 = HEAD 原形**，范围 1..32）。
`blockDim = rows*32`、`grid = (ceil(n_total/rows), slots)`；kernel 的 row 窗口循环本来就闭合
（`row = blockIdx.x*nwarps + warp; row += gridDim.x*nwarps`，`nwarps` 由 `blockDim` 推导），
所以**除 launcher 的三行外 kernel 无需改行数**。**逐位一致**：每行的 dot 全在单个 warp 内完成
（shfl 树 + lane 0 写 `out[row]`），行间无归约，输出布局与 pitch 都不变 ⇒ down_reduce 的输入假设不受影响。
顺带修掉一个隐形地雷：LUT 构建原写作 `if (threadIdx.x < 256)`，rows<8（blockDim<256）时会漏建
表项 128..255 → 改成 256 步长的 stride 循环（≥256 线程时逐位相同）。
⚠️ down 方向（`dsv41_expert_down_fp4_batched` / `dsv41_expert_down_reduce_fp4_batched`）**保持 8 warp 不变**：
它的 grid 是 (⌈5120/8⌉,6) = 3840 CTA，本来不欠填充，且 `expert_gemv_fp4_down_reduce_kernel` 里
同样留着 `threadIdx.x < 256` 的写法（其 launcher 恒定 256 线程，故目前安全——改它的 blockDim 前必须先改那一行）。

**为什么预期 ≈0（别把它当 −0.5ms 的刀）**：
`rows × slots` 就是全部工作切分（DSV4.1：320 行 × 6 slot = **1920 个 row-dot，一 warp 一个**），
把同样的 1920 个 warp 重新打包成 240 / 480 / 960 个**更小的 CTA**：
- **warp 总数不变 ⇒ 每 SM 驻留 warp 数不变**（1920/148 = **13 warp/SM，与 rows 无关**）。
  延迟遮盖由"驻留 warp 数"决定，不由"驻留 CTA 数"决定 ⇒ "1.6 → 3.2 → 6.5 blocks/SM"不产生任何新并行度；
  若寄存器是驻留上限，半宽 CTA 让 blocks/SM 翻倍、warp/SM 不变，**严格等号**。
- **每 warp 的 MLP 不变**（仍 10 组 × unroll 4 × 1 LDG.128/组）⇒ in-flight 字节不变。
- 两个副作用是**负**的：`s_act`（k floats = 20KB）+ 256 项 LUT 是 **per-CTA**、被该 CTA 的行共享，
  行数减半 ⇒ **每行的 prologue 翻倍**（另有 smem store 量翻倍）；CTA 越小，能遮盖该 prologue 的
  warp 越少。⇒ rows=2/1 很可能是**净负**。

**该假说其实已被现成数据部分否证**：down 方向的 CTA 有 **3840 个**（gateup 的 16x）、warp **30720 个**
（16x），搬的字节只有 gateup 的一半 —— 若"CTA/warp 数 → 有效带宽"成立，down 应碾压 gateup；
实测 down v2 = 17.2µs / 4.92MB = **286 GB/s**，gateup 25.3µs / 9.83MB = **389 GB/s**：
**down 每字节效率更低**。⇒ 这个家族的限制量是**每 warp 的 in-flight 字节（MLP）与每 warp 的固定开销**，
不是 CTA 数。（388 GB/s vs 7.6TB/s 峰值 = 5%，DRAM 地板 1.3µs/层 vs 25.3µs = 19x ⇒ 不是带宽墙。）

**真正的决策量（本机无 nvcc/GPU，必须先到节点上量）**：`expert_gemv_fp4_batched_kernel<true|false>` 的
**regs/thread**（`cuobjdump --dump-resource-usage`）+ ncu `sm__warps_active.avg.pct_of_peak_sustained_active`。
两种情形**药方相反**：
- **regs ≤ 64/线程** ⇒ 每 SM 可驻 4 CTA = 32 warp，而现在只有 13 ⇒ grid 确实欠填充，加 warp 有效
  （但**只能用 K-split**：行拆分永远加不出 warp）。
- **regs ≥ 128/线程** ⇒ 每 SM 只能驻 2 CTA = 16 warp，现在 13/16 = **已达可驻留上限的 81%** ⇒
  瓶颈是**寄存器**（药方是减寄存器 / 加 MLP / 加 unroll 深度），K-split 多出的 warp 会挤成第二波，
  收益 ≈ −20% 或归零。参照同族 down 核：**40 regs/thread ⇒ 6 CTA/SM**（§3B 的隔离微基准实测）。

**执行顺序**：(1) `DSV41_GATEUP_ROWS=4/2` A/B（证伪 CTA 数假说；逐位一致，唯一变量是 CTA 形状）；
(2) 同轮抓 regs/thread；(3) 按结论选 K-split（加 warp）还是 MLP/寄存器路线。

**K-split 设计（✅ 2026-09-11 已实施，env 门 `DSV41_GATEUP_KSPLIT`，默认 1 = OFF）**：

> **落地状态（2026-09-11）**：`kernels/cuda/dsv41_experts_mxf4.cu` 的 fused gate/up 分支已实现
> K-split。env `DSV41_GATEUP_KSPLIT`（默认 **1**，范围 1..8，`dsv41_gateup_ksplit()` 缓存读取）；
> launcher `dsv41_expert_gate_up_fp4_batched` 里 `int ksplit = fuse ? dsv41_gateup_ksplit() : 1;`
> （**非 fused 分支强制 1**——只有 fused 体实现了跨 half 合并，否则两个 half 会各自算整行并 race 同一个
> `out[row]`），并在 `warps*ksplit > 32` 时递减（blockDim ≤ 1024）。grid.x **仍按 rows 算**
> （`ceil(n_total/rows)`），blockDim = rows*ksplit*32，即：
> - **默认 rows=8 + ksplit=2 ⇒ 240 CTA / 16 warps（512 线程）**，smem 仅多出 `rows*ksplit*8B = 128B`
>   （ksplit==1 时一字节不加 ⇒ 占用率与 HEAD 完全一致）；
> - `DSV41_GATEUP_ROWS=4 DSV41_GATEUP_KSPLIT=2` ⇒ 480 CTA / 3.2 per SM（即任务书里那个形状，仍然可达，
>   但**不是推荐形状**——它把 per-CTA prologue 翻倍，见下面的"形状修正"）。
>
> 合并走 **CTA 内 smem + `__syncthreads`**（不是跨 block scratch）：一半的 lane0 写 `s_ks[warp]`，
> 一次 barrier 后 `half==0` 的 lane0 按 **升序 half** 用 `__fadd_rn` 折叠（`ksplit=2` 时就是
> `__fadd_rn(g_half0, g_half1)`，即 `(g0..g4)+(g5..g9)`）**在 clamp/silu 之前**。窗口循环在
> ksplit>1 时退化为**单趟**（`row_stop = row_base+1`），且越界 warp 用 `active` 守卫（**不能**用
> loop 边界守卫——barrier 必须被 CTA 内所有 warp 到达，否则死锁；越界 warp 的 load 重定向到 row 0）。
> 验证：远端 `nvcc 13.2 -gencode arch=compute_103a,code=sm_103a --use_fast_math` **编译干净**；
> `cargo check -p ferrite-models` 通过。**parity A/B（文本）与 nsys 微基准尚未跑**——翻转前必须先做。

- 结构：`grid=(ceil(n_total/4), slots)` + `blockDim=8 warps`，warp `w` 与 `w+4` 认领**同一行**，
  各做 g2 ∈ [0,5) / [5,10)（k=5120 → k>>9 = 10 组，切在 32-值 scale block 边界上），各自做 warp 内
  shfl 树，两个 partial 落 smem（4 行 × 2 方向 × 2 partial = 32B），`__syncthreads()` 后由一侧合并并写 `out[row]`。
- ⚠️ 合并必须在 **clamp/silu 之前**（`g = a_g + b_g`、`u = a_u + b_u`，再 `fminf/fmaxf` + silu），否则语义错；
  固定"先 a 后 b" ⇒ 仍确定，但**不再与单 warp 路径逐位一致** ⇒ 需要 parity/容忍度测试（这是方向 2 的真实代价，
  也是先做方向 3 的原因）。另注意 build.sh 的 `--use_fast_math` 会重排浮点：新代码的每个 add/mul 要用
  `__fadd_rn`/`__fmul_rn` 或 `fmaf` 钉住（2026-09-11 的 4-路展开漂 1ULP 事故）。
- 结构优势：CTA 仍是 240 → **per-row 的 s_act prologue 不翻倍**（行拆分做不到这点）；代价是每 warp 的
  in-flight 字节减半且 warp 数翻倍 ⇒ 净效果取决于上面那个 regs 结论（0.81 波 → 1.62 波时 ≈ −19% 而非 −50%）。
- **形状修正（2026-09-11 核对代码，比上面的 4-行/8-warp 方案更优，已按此落地）**：`blockDim=8 warps + rows_per_cta=4`
  会把 CTA 数从 240 翻到 480，per-CTA 的 `s_act`（20KB）+ LUT prologue **随之翻倍**。更优形状 =
  **保持 rows=8/CTA、blockDim 加到 16 warps（512 线程）**：warp `w` → `row_local = w / ksplit`、`half = w % ksplit`，
  grid 仍 `(40, 6) = 240 CTA` ⇒ warp 数 1920 → 3840（**26/SM**）而 prologue 不翻倍。每 warp 仍
  `#pragma unroll 4`（可升 5）× 1 LDG.128（ILV）⇒ **每行 in-flight 字节 64 → 128B**，这才是 MLP 增益的来源
  （不是"每 warp 更深"——每 warp 只有 5 组，比原来浅）。
- **切点/合并实现**：half A = `g2 ∈ [0,5)`、half B = `[5,10)`（沿 **512 值 group 边界**连续切；每 warp 覆盖
  2560 个连续激活，scale block 不跨界）。ILV 地址 `g_row + 2*q`（`q=(g2<<8)+(lane<<3)`）对任意 g2 都是 16B 对齐，
  **两半的地址算术无需改**，只改循环上下界（`g_begin = half*nv2f/ksplit`，`g_end = (half+1)*nv2f/ksplit`）。两半各跑**同形 5 步 shfl 树** → lane0 写 `s_ks[warp]`，
  **加一次 `__syncthreads`**（因此把 `:1017` 那个 grid-stride 窗口循环在 ksplit>1 时改成单趟；launcher 已按
  `ceil(n_total/rows)` 定 grid，grid.x 按 **rows（8）** 而非 warps(16) 算），
  最后由 `half==0` 的 lane0 做 `g = __fadd_rn(g_half0, g_half1)`、`u = __fadd_rn(u_half0, u_half1)`，**再** clamp + silu 写 `out[row]`。
- `DSV41_GATEUP_KSPLIT=4` 不可取：`nv2f = k>>9 = 10` 不被 4 整除（只能 3/3/2/2 不均衡）⇒ 取 **2**（或 5）。内核的通用公式对任意 ksplit/rows 都闭合，但**只有 2 是设计点**。

---

## 5. 小核全清单（< 0.1ms，防漏算）

| kernel | 次/步 | µs/次 | ms/步 |
|---|---|---|---|
| `indexer_score_kernel`（Step A，v2 已落地见下）| 4 | 19.7 | 0.079 |
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

⚠️ **misc 残余四项的最终判定（2026-09-11 explore，正式关闭 misc 优化线）**：
`rope_precompute` 的 **16 次不是 per-step**——代码里只有 **2 个调用点**，都在 `DevChain::new` 内
（`chain_dev.rs:892/899`，主流 θ=10000 + compress 流 θ=160000），表**一次性覆盖整个上下文**
（`table = max_pos`，`DSV41_MAX_POS` 默认 64k，`chain_dev.rs:780-904`），`reset()` 不重建
（`chain_dev.rs:946`）。⇒ 16 次 = profiled 窗口内 **8 次链构造**（每进程 3 个 `DevChain::new` 调用点：
`dsv41-run.rs:199/394/802`），per-step 成本 **0**；"缓存"已是用尽（整段表即缓存）。
`gemv_f32_v2`（7×, 0.029ms）与 compress 的 `compressor_pool`/`compress_commit` 都在 `side_stream3` 上
（`lin_f32_on` → `gemv_f32_on`，`device.rs:2860`），与 q/kv 链重叠 ⇒ 非关键路径（v2 已落地，
`dsv41_glue.cu:969-1038`，4.1µs/次 = 2.6MB 权重的 ~12× HBM 地板，但藏得住）。
全表真正留在关键路径的 misc 残余只有 `embed_expand_dev`（1×, 0.012ms，`ferrite_kernels.cu:10057`，
`chain_dev.rs:1693`）：n=1 单块、读 1 行 + hc 展开，DAG 头，11.6µs 是**单核 launch 延迟地板**。
⇒ **misc 线关闭；关键路径残余 ≈ 0.012ms（embed 地板），无值得做的项。**

⚠️ **本表是 `db2917501`（09-11 16:48）的**剖析快照**，不是当前代码的 launch 清单**。表里四个 elementwise 小核
此后都已被融合/消除（2026-09-11 explore 逐核复核；数字仍保留供溯源）：

| 表内条目 | 当前状态（代码 + env 默认） | 出处 |
|---|---|---|
| `add_kernel` 40× | ✅ **ADD_EPI 已实施**（`DSV41_ADD_EPI`，默认 ON）：`moe` 的共享专家合并 `s.o += s.ex_out` 折进 MoE AR#2 的 **store** epilogue（`ferrite_p2p_ar_v5_add` / `..._hcpost_add`）。旧 `.so`/符号缺失/未跑共享专家 → 自动回退独立 `ferrite_add` | `chain_dev.rs::add_epi` / `moe_reduce` / `tp.rs::all_reduce_inplace_add` |
| `swiglu_limit_kernel` 40× | ✅ **已无独立 launch**：routed batched 路径由 gate/up 融合的 epilogue 承担（`gateup_fused` 默认 ON ⇒ `swiglu_limit_batched` 被跳过）；共享专家由 A4 `swiglu_limit_q`（默认 ON）直出 fp8 | `chain_dev.rs:3838 / 4027-4047` |
| `fp4_pack_kernel` 40× †† | ✅ **已融合**：`g_q4_fuse` 默认 1 ⇒ 单核 `quant_fp4_fused_kernel`；本表 tree（16:48）早于落地它的 `2d7eead`（18:39）⇒ 该行是**融合前**数据 | `dsv41_kernels.cu:2310-2332` |
| `ring_append_kernel` + `window_idxs_kernel` | ✅ **已合并为一个** `dsv41_ring_win_fuse`（`DSV41_RING_WIN_FUSE` 默认 ON，每层 1 次、非 owner 层仍写 idxs）；本表 tree 早于落地它的 `710c107`（17:54）⇒ 两行都是**合并前**数据 | `chain_dev.rs:2802` |
| `compressor_pool_kernel`/`compress_commit_kernel`（4×/步）| ✅ **已移出关键路径**（`DSV41_COMPRESS_SIDE` 默认 ON，`ec439e3` 19:42）：kv-source 层的 4 个 compress launch 在 `lin2` 后 fork 到 `side_stream3`，join 延到本层尾部（`window_idxs` 之前），与 q 链（主流 ~13.5µs）+ kv 链（side2）重叠。本表 tree（16:48）**早于**该 commit ⇒ 表里这两行是**串行时代**数据。🔁 2026-09-11 起 `DSV41_COMPRESS_FUSE`（默认 ON）再把 decode 的 `state`+`pool`+`commit` 合一为 `compressor_fused_kernel`（`dsv41_kernels.cu:2638`）⇒ 每层 4→3 launch、`compress_commit_kernel` 仍是 `dsv41_glue.cu:658`（fused kernel 内是它的逐字拷贝，两处须同步） | `chain_dev.rs:364/382/3801-3880`；`device.rs:826-880/2450-2560/3150-3200` |

### indexer_score v2（Step A，✅ 2026-09-11 已实施，**待 A/B**）

> **病**（v1，`dsv41_kernels.cu` 的 `indexer_score_kernel`）：与修好前的 gate 同构——每个 FMA 配一次**标量**
> load（q、k 各 128 次/lane/候选）、128 深依赖 FMA 链、头归约是 O(nh)=32 次 broadcast shuffle 且累加器在 lane 0，
> 而 grid 只有 **256 CTA × 8 warp = 2048 warp**（148 SM ⇒ **13.8 warp/SM = 21% 占用**；注意不是"几个 block"，
> 是 1.7 CTA/SM）。
>
> **修**（`gemv_bf16_v2` 模式直接套用，新增 `indexer_score_kernel_v2<WPR>`）：
> (a) **float4 (16B) q/k load + 4 个独立累加器** ⇒ 32 个向量步替代 128 个标量步，每条累加器链降到 8；
> (b) **WPR 个 warp 对同一候选切 hd**，同 block smem 折叠（`s_part[warp][lane]`，无 atomic、无第二内核；
> relu/权重必须在**完整 hd 点积之后**，所以折叠发生在 relu 之前）；
> (c) 头归约改 **5 步 `shfl_down` 树**（32 次 shuffle + lane0 上 32 长依赖加链 ⇒ 5+5）；
> (d) CTA 数不再钉死 256：`kIdxScoreBlocksV2 = 1024`（8192 warp = 55/SM 请求）。
>
> **env**（`dsv41_indexer_topk` launcher 内各读一次）：`DSV41_IDX_SCORE_V2`（0 = 逐字回退 v1）、
> `DSV41_IDX_SCORE_WPR`（1/2/4/8，默认 **1**）、`DSV41_IDX_SCORE_BLOCKS`（覆盖 grid.x）。
> 前提 `hd % 4 == 0 && 0 < nh <= 32`（float4 行对齐 + 树折叠 32 lane），否则自动回落 v1。
>
> **WPR 默认 1 的理由**：K-split 是**延迟**修复不是吞吐修复（总工作量不变），而它的折叠每轮要两次
> `__syncthreads`。index-source 层 `compress_ratio == 1`（`configs/dsv41_flash.json`）⇒ **n_pos 随序列长度走（万级）**，
> 单 warp 一候选的映射已足够填满机器，此时 barrier 开销超过它隐藏的延迟。WPR=2/4/8 留给短上下文那种
> n_pos 撑不满 8192 warp 的情形。
>
> ⚠️ **K-split 分支的 barrier 在循环体内 ⇒ 必须整 CTA 统一轮数**：`p = base + r*pstep + g`（g 随 warp 变）会让
> 低编号 group 多跑一轮、`n_pos` 不整除 stride 时**死锁**。实现改为由 block 的 base 算 `nround`，越界 group
> 用 `live` 守卫（load 重定向到候选 0、跳过 store、但仍到达 barrier）。
>
> **已核**：远端 `nvcc 13.2 -gencode arch=compute_103a,code=sm_103a` 编译干净（仅既有 warning）；
> `ptxas -v`：WPR=1 **40 regs / 0 barrier / 0 smem**（v1 32 regs），WPR≥2 40 regs / 1 barrier / 1024B smem。
> 40 regs ⇒ 每 SM 驻 6 CTA = **48 warp/SM（75%）**，比 v1 的 13.8 高 3.5x；`cargo check -p ferrite-models` 通过。
>
> **未做（翻转默认前必须）**：parity A/B（文本）与 nsys 隔离微基准
> （`scripts/dsv41_indexer_bench.cu`，n_pos ∈ {32,128,512,2048,4096}）。求和顺序变了（float4 lane、K 切片
> partial、树折叠）⇒ 分数 ~1e-7 漂移，远低于 top-k 选择边际；golden 是 1 元素点积（`ops.rs`）。
> 下一步若还要压：这个 score 本质是 `[nh,hd] × [hd,n_pos]` 的 GEMM（N 极大），CUDA core 的 128 warp-FMA/候选
> 是硬地板，再往下只能走 tensor core + 列归约 epilogue。


---

## 6. 陷阱与注意事项（改这块代码前必读）

- ⚠️ **mean 还是 median**：nsys 报告同时给 mean/median。**聚合 ms/步必须用 mean×次数**（总时长是相加的）。
  `gemm_fp8_gemv` 的 median 7.9µs vs mean 9.5µs **差 0.39ms**（= 全表的 4%）。
  `86af349`/STATUS 表用的是 median，**不要把那张表和本表逐行比**。
- ⚠️ **异质 population 不能报单一 per-call**：`gemv_bf16` 现在 49 次里混着 ≈17.3µs（共享专家 gate）和 ≈45µs（lm_head/engram）两族。
  想优化 lm_head 就不能看 22.4µs 这个均值。
- ⚠️ **env 默认门翻转会改变 kernel 混合**：`MIX_GATE` ON→OFF 让一个核换成两个核。
  **跨 commit 比较 profile，等于同时比较 env 默认值**。本文头部必须记 env，v2 就是踩了这个（它的表是 MIX_GATE=ON 口径）。
- ⚠️ **profile 的 .so 版本 ≠ 代码版本**（ADD_EPI 复核时踩到）：nsys 表里的 `swiglu_limit`/`fp4_pack`/`ring_append`+`window_idxs`
  在本表 tree（`db2917501`）之后才被融合，**照表去"实施融合"会去重做已完成的事**。判据一律是
  `git merge-base --is-ancestor <落地 commit> <profile tree>`，不是核名字还在不在表里。
- ⚠️ **ADD_EPI 的 host 决策在捕获时被烘焙**：整个 step 被捕获成一个 CUDA graph（`step_body`，`chain_dev.rs:1617-1646`），
  `moe()` 里的 host 分支只在**捕获那次**执行，AR 的参数（含 bias 指针）被写进图。所以 `moe_add_in` 必须
  **per-layer 且只读不消费**（`Vec<Option<*const f32>>` + `moe_reduce(layer)` 读）——写成 `take()` 会让每次
  replay 丢融合，写成单字段会让 per-layer MoE 图（`DSV41_GRAPH_MOE`）互相覆盖。
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

   **P1 a32 死槽消除（2026-09-11 已落地代码，待实测）——a32 ON 时 `s_a` 是死槽**：mode 4 的
   k 字节激活 staging（`s_a`）在 `a32==1` 时**只被读一次**——就是 `s_af` 的物化循环
   （`s_lut[ap0[i]] * s_as[i>>5]`，`ap0 = s_a`）；消费循环在 a32 分支只碰 `s_af`，`ap` 在该
   分支是死代码。**已实施**：把"uint4 跨步拷贝 → 第二趟逐字节解码"合成**一趟**
   （global uint4 宽读 → LUT 查表 → ×`s_as` → 直写 `s_af`），`s_a` 槽仅在需要时分配。
   收益：k=5120/warps=4 时 gsmem **48512 → 43392B**（blocks/SM 4 → 5，+25% 驻留）+ 少一趟 k
   遍历；`s_af[j]` 仍是同一乘积 ⇒ **逐位等价**（`s_a` 只是 `a` 的副本）。

   实现要点（改动都在 `dsv41_kernels.cu`，无 Rust 侧改动）：
   - 内核 `gemm_fp8_gemv_kernel`：新增 `const bool a32_direct = (a32 != 0) && (vec == 4) &&
     (qr_raw == nullptr);`（~`:3080`）；`s_ws = s_a + (vec == 4 && !a32_direct ? k : 0)`，
     即清掉 `s_a` 槽后整段尾部（`s_ws/s_as/s_lut/s_af/s_rows`）一起下移 k，**尾部内部相对
     偏移不变**（`s_rows` 的 `a32 ? s_af + k : s_lut + 256` 无需改）。
   - staging 循环（`:3153`）guard 加 `&& !a32_direct`；物化循环（`:3195`）新增
     `else if (a32_direct)` 分支做 global→`s_af` 的单趟融合；`s_af` 之后的
     `__syncthreads()` 位置不变（barrier 语义与旧路径一致）。
   - launcher 侧新增 `dsv41_gemv_sa_bytes(k, norm_fuse)`（~`:2613`），与内核的 `a32_direct`
     **必须保持同步**。除 `dsv41_gemm_fp8_mx_rope_norm`（`qr_raw` 非空：prologue 写 `s_a`、
     物化仍读它 ⇒ 保留 `(warps+1)*k`）外，其余 6 个 launcher（mx / rope / mx2_rope /
     mx_add / mx_f32 / mx2）都 `qr_raw == nullptr`，mode 4 改为
     `warps*k + dsv41_gemv_sa_bytes(k, false) + scale_bytes`；host 探针
     `dsv41_gemv_gsmem`（mode 4）同步。
   - `a32=0` 路径**不受影响**：`dsv41_gemv_sa_bytes` 返回 k ⇒ gsmem 与旧式 `(warps+1)*k`
     逐位一致，且消费循环仍读 `s_a`（`fuse_a32` 为假，staging 与物化循环原样执行）。
   - `mode 3` **未改**（无 `s_a` 可言，仍逐字节从 global 读），保持 A/B 基线可比。

   验证：远端 `nvcc -arch=sm_103a -Xptxas -v` 编译通过，`gemm_fp8_gemv_kernel` 的
   `32 regs / 32B spill / 128B static smem` 与 HEAD **完全一致**（未引入新 spill）。
   ⚠️ 反证风险：`MODE=3` 已等价于"不 staging、从 global 读"且实测更慢（19.96 vs 19.24）——
   但那是把 global 读留在**物化循环里逐字节**；本方案保留 uint4 宽读，只是把两趟并成一趟，
   属于 mode 3/4 之间的第三点，**仍需实测**（`scripts/dsv41_a32_bench.sh` 的 smem/blocks-per-SM
   一栏现在应打印 43392B / 5）。
   ✅ 同类机会（**2026-09-11 已同法落地**）：`gemv_bf16_fp8x2_kernel`
   （`dsv41_gemm_bf16_fp8x2`）有完全相同的死槽——其 `s_a` 也只被物化循环读，且该核**没有**
   a32 门/参数（消费循环无条件读 `s_af`，等价 a32 恒 ON），所以合并是**无条件**的。kernel
   `s_lut` 基址改 `s_w + nwarps*k`（原 `s_a + (vec == 4 ? k : 0)`；mode 3 本就 offset 0，
   布局不变）、删 staging 循环、物化循环改 global uint4 → LUT → 直写 `s_af`；launcher gsmem
   去掉 `(vec == 4) ? (warps + 1) * k` 的额外行 ⇒ 默认形状（k=5120/warps=4/mode 4）gsmem
   **47104 → 41984 B**。签名/门不变，无 Rust 改动。⚠️ 该核**没有** a32=0 回退臂，因此一旦
   回归只能 revert（不同于单族 `gemm_fp8_gemv_kernel` 有 `DSV41_GEMV_A32=0` 对照臂）。
0.5 **P2 自适应 warps（2026-09-11 已落地代码，待实测）——每 block 行数按 n 选择**：
   每调用的固定成本是**块级 prologue**（激活 uint4 staging + LUT 构建 + a32 物化），每 block 付一次、
   近似与 n 无关（perf-roadmap 2026-09-11：2.85µs/call 跨 416 blocks，n 涨 6.5× 只涨 ~5%）。
   P1 后 mode 4 / k=5120 / warps=4 的 gsmem = 43392B ⇒ 232448/43392 = 5.35 → **5 blocks/SM**
   （~30% 余量）。大 shape 可以让每 block 处理 **8 行**：gsmem = 8×5120 + 23552 = **64512B**
   ⇒ 232448/64512 = 3.6 → 3 blocks/SM，但 block 数**减半**，固定 prologue 摊到 2 倍行数上。
   延迟敏感的**小 shape**（`wq_a+wkv` 合并 n=1280+512=1792、`sh_w13` n=640）反之需要更多 block
   （更多在飞 warp）⇒ 用 n 分档。
   - **实现**：`dsv41_kernels.cu` 新增 `g_gemv_warps_adaptive`（env **`DSV41_GEMV_WARPS_ADAPTIVE`**，
     默认 ON，`=0` 回退固定 `DSV41_GEMV_FP8_WARPS`）+ 常量 `kGemvWarpsBigN=2048` + 内联
     `dsv41_gemv_warps_for(n)`。**只改 launcher 的 warps 选择，kernel
     不变**（`gemm_fp8_gemv_kernel` 早已按 `blockDim` 参数化）。
   - **P2b（2026-09-12 已落地，待实测）大 n 专用旋钮**：大 n 臂的 warps 从常量 `kGemvWarpsBig=8`
     改为 `g_gemv_warps_big`（env **`DSV41_GEMV_WARPS_BIG`**，默认 8、合法 4..32、缺失/越界回退 8）。
     **动机 = 去混淆**：`DSV41_GEMV_FP8_WARPS` 只喂小 n 臂（n<2048），拿它扫大 n 根本扫不动；而
     `DSV41_GEMV_WARPS_ADAPTIVE=0` 会把**两个臂一起**塌到 `g_gemv_warps`，小 shape 跟着变 ⇒ 要么扫不到、
     要么混淆。新变量是**大 n 臂唯一读取源**，serve 可固定小 shape、单独扫 4..32。
     ⚠️ 自适应门仍然守在前：`DSV41_GEMV_WARPS_ADAPTIVE=0` 时大 n shape 回退 `g_gemv_warps`，
     `DSV41_GEMV_WARPS_BIG` **失效**（那是回退路径，不是 A/B 臂）。未设 env 时逐字节等价旧行为。
     ⚠️ warps 越大 gsmem 越大（mode4：`warps*(k+nb_k_al) + …`；k=5120 时 warps=8→64512B、
     32→191232B，仍低于 232448 天花板且 32 warps=1024 线程=块上限），但 **k 更大 + warps=32** 有撞
     `dsv41_smem_ceiling` 的风险——越界值 launcher 返回非 0，调用方需确认有 fallback。
   - **覆盖面 = 4 个原本选 `g_gemv_warps` 的 launcher**：`dsv41_gemm_fp8_mx` /
     `_mx_add` / `_mx_f32` / `_mx2`（mx2 用 **n1+n2 总行数**，两族共享同一 block grid）。
     B1（`xq != nullptr`）仍在其后**强制 32**，不受影响。
   - **rope 族不动**（`mx_rope` / `mx_rope_norm` / `mx2_rope` 保持 nwarps=32）：rope epilogue
     的 pair 交换依赖 `e = blockIdx.x*nwarps + warp` 且 `grid*nwarps == n`；`mx_rope_norm`
     的 NORM_FUSE prologue 归约树还必须跑在参照的 1024 线程上才逐位等价。
     ⇒ **wq_b 在默认 NORM_FUSE 路径下仍走 32**；只有它的非融合 fallback 走 `dsv41_gemm_fp8_mx`
     时才吃到 8 行/block。**wo_b（f32, n=5120）/ w2（add, n=5120）/ 其余 n≥2048 的 `lin`** 均已覆盖。
   - ⚠️ 待实测：`DSV41_GEMV_WARPS_ADAPTIVE=0` vs `=1` 的同窗口 A/B（文本 + nsys 的
     `gemm_fp8_gemv` 每步 ms）。无 Rust 改动，`cargo check -p ferrite-models` 通过。
0.6 **P3 cp.async 权重先行（2026-09-11 已落地代码，待实测）——prologue 里唯一可动的串行段**：
   每 block 的固定成本是块级 prologue（激活 uint4 staging + LUT + a32 物化），而**权重行**的
   cp.async 原本在 `__syncthreads()` **之后**才发射 ⇒ 传输延迟（k=5120 时每 warp 5KB）完全
   暴露在 dot 前面。权重行不依赖任何激活产物（`w`/`w_scale` 是模型常量），所以把它提到
   staging **之前**发射、让 staging + barrier 做覆盖，loop 第 0 次迭代只 `wait_all` 即可。
   - **实现**：`g_gemv_cpasync`（env **`DSV41_GEMV_CPASYNC`**，默认 ON，`=0` 回退旧发射顺序）
     → `GemvCore.cpasync`；内核 prologue 在 `s_rows` 之后新增 `pf_row = blockIdx.x*nwarps+warp`
     的预取块（同一套 family/指针选择，逐字照抄 row 循环），loop 内 `const bool prefetched =
     (row == pf_row)` 决定是否跳过本轮发射；`wait_all` 无条件保留（多迭代 launch 仍正确）。
   - **smem 不变**：预取写进 row 循环本来就要写的同一个 per-warp 槽（`s_w + warp*k`），
     gsmem 仍是 43392B @ k=5120/warps=4/mode4 ⇒ **5 blocks/SM 保持**。分析里"+5KB 权重
     staging buffer"的双缓冲版本才会掉到 4 blocks/SM（自付费），故未采用；`s_ws` 的 scale 行
     也刻意不预取（同步字节 load，提到 barrier 前会顶住 barrier）。
   - **必须在 `cudaGridDependencySynchronize()` 之后发射**：`w` 可能由流上前一个节点写
     （requant 链），PDL 下 producer 还在跑 ⇒ 提前读是竞态。所以这是"循环内暂存的重新排序"，
     不是新的提前读。
   - **逐位等价**：同字节、同槽、同消费顺序，只有发射时机变了；新增状态只有一个 commit group，
     由 loop 的 `wait_all` 回收。残留风险：`pf_row` 的活跃区间跨整个 prologue（+1 寄存器），
     `__launch_bounds__(1024)` 把它压在 64 regs 内 ⇒ 只会 spill 不会 701；需实测 regs 数。
   - 验证：本机无 nvcc（`cargo check -p ferrite-models` 通过；.cu 需远端 `build.sh 103a` 重编）。
     ⚠️ `.cu` 改动后必须 `.so` 重编，否则运行期 build-id 门禁拒启。
   - 预期（分析口径）：~2-3µs/call × 246 call ⇒ −0.5~0.7ms。**待 user 亲自单轮 A/B**：
     `DSV41_GEMV_CPASYNC=1`（默认）vs `=0`，同窗口，文本 + nsys `gemm_fp8_gemv` 每步 ms。
0.6b **P3 → 混合核 `gemv_bf16_fp8x2_kernel` 的复制（2026-09-11 已落地代码，待实测）**：
   nsys v9 口径下该核（MIX_GATE=ON 时的 gate+共享专家三合一 launch）med **9.1µs × ~38/step
   = 0.35ms**，是 0.6/0.58 之外唯一还带"fp8 权重行在 barrier 之后才发射"的 gemv 族内核。
   - **实现**：新增 host gate `g_bf16fp8x2_cpasync`（env **`DSV41_BF16_CPASYNC`**，默认 ON，
     `=0` 回退），launcher 以**新增末位 kernel 参数 `cpasync`** 传入（该核原末参 `vec` 在
     kernel 内**未被引用**，仅作 mode 标记保留）。kernel prologue 在 `s_af` 基址算出后、**LUT
     构建之前**插入预取块，loop 内 `const bool prefetched = (row == pf_row)` 跳过本轮发射。
   - **比单族 gemv 的 P3 覆盖窗口更长**：单族的 prologue 只有 LUT + 激活 staging；本核还有
     **P1 的 `s_af` 物化**（k/16 次 global uint4 + 每次 16 个 LUT 查表）⇒ 预取早了整个 a32
     段 + 两道 barrier，隐藏窗口更大。
   - **fp8-only**：`row < nb` 的 bf16 gate 行**不 staging**（4-deep ILP 的直读 LDG，没有可
     重叠的 smem 拷贝；且它一行是 2k 字节，塞不进 k 字节的 fp8 槽）。所以条件是
     `pf_row >= 0 && pf_row >= nb`；`nb == 0` 时全 fp8，无影响。
   - **commit 无条件**（bf16 首行 warp / 门关时提交空 group），保证全 block 的 per-thread
     commit-group 计数一致；loop 的 `wait_all` 在预取那次迭代回收它。**无新增 smem 槽**，
     gsmem 公式不变。
   - **不需要 `cudaGridDependencySynchronize()`**：本核 launcher 是 plain `<<<>>>`（非 PDL），
     与 `gemm_fp8_gemv_kernel` 的 PDL 前提不同。
   - 验证（2026-09-11）：远端 `nvcc 13.2 -gencode arch=compute_100a,code=sm_100a -Xptxas -v`
     **编译通过、无 error**（仅 3 条既有 warning）；`gemv_bf16_fp8x2_kernel` 寄存器
     **44 → 46（+2）**、**0 spill / 0 stack**，占用率不受影响（reg 远非瓶颈，smem 才是）。
     本机无 nvcc ⇒ `cargo check -p ferrite-models` 通过（本改动**无 Rust 改动**，`extern "C"`
     签名不变）。
     ⚠️ `.cu` 改动后必须重编 `.so`（`kernels/cuda/build.sh 100a`），否则运行期 build-id 门禁拒启。
   - 预期（分析口径）：−0.09ms（0.35ms × 20~25%）。**待实测 A/B**：`DSV41_BF16_CPASYNC=1`
     （默认）vs `=0`，同窗口，文本 + nsys `gemv_bf16_v2` 每步 ms。
     ⚠️ 注意 `MIX_GATE` 默认值：本核只在 MIX_GATE=ON 时被调用（见 §0.2 (A)），
     **若同窗口 MIX_GATE 是 OFF，本门完全不生效**——A/B 前先确认该核仍在 profile 里出现。
0.58 **P4 cp.async 权重先行 —— expert gateup 版（2026-09-11 已落地代码，待实测）**：
   同一个 prologue 串行问题的 expert 侧复制。`expert_gemv_fp4_batched_kernel<ILV>` 的
   prologue = LUT(256 项) → `cudaGridDependencySynchronize()` → 激活 staging（uint4 解码）
   → `__syncthreads()` → row loop；**FUSED gate/up body 每 warp 一行、按 `k>>9` 个 512 B
   group 走**，group 0 的权重 LDG 就落在 barrier 之后、自己的 dot 之前，没有任何东西遮盖它。
   - **实现**：env **`DSV41_GATEUP_CPASYNC`**（默认 ON，`=0` 回退）→ host gate
     `dsv41_gateup_cpasync()` → **新增末位 kernel 参数 `pf`**（3 个 launch 点全部接线：
     gateup 的 ILV/非 ILV 传 `fuse && gate`，down batched 传 0）→ 内核 prologue 在
     `cudaGridDependencySynchronize()` 之后、激活 staging 之前，每 lane 发 **1 条
     `__pipeline_memcpy_async(...,16)`（cp.async.cg）**，把该 warp 的 group 0（512 B =
     gate 256 B + up 256 B，ILV 时是一整块 512 B）拷进新增 per-warp 槽 `s_pf`；
     循环内 `from_pf = pf_ok && (row == row_base) && (g2 == g_begin)` 时改从 smem 读
     （plain：gate 在 `+lane*8`、up 在 `+256 + lane*8`；ILV：`uint4` 在 `+lane*16`）。
   - **smem 布局耦合（新）**：`s_pf` 夹在 `s_lut2` 与 `s_ks` 之间，大小 `nwarps*512 B`
     （rows=8/ksplit=2 时 8 KB）。launcher 的 `smem` 公式与内核的
     `s_pf/s_ks` 指针**必须同时改** —— 同 §0.55 的布局耦合教训，错一边就整体平移越界。
   - **不整行 staging**：整行 5 KB/warp（rows=8/ksplit=2 ⇒ 80 KB/CTA）会把占用率换掉，
     正是 gemm-prologue-overlap 分析里判定为"自付费"的那条路；这里只买得起 prologue
     能覆盖的那 512 B。默认形状 CTA 是**线程受限**（512 线程、4 CTA/SM），8 KB 不损占用。
   - **逐位等价**：同字节、同 lane 偏移、同消费顺序，只有"从哪块内存读"变了。⚠️ plain
     布局下 16 B 的拷贝块宽于单 lane 的 8 B 读 ⇒ 消费者 lane ≠ 拷贝 lane，故 `wait` 必须是
     每线程 `wait_prior(0)` 且**紧跟既有的 pre-loop `__syncthreads()`**（由 barrier 发布），
     不能只依赖 lane 内顺序。
   - ⚠️ 风险：① 16 B 对齐是 cp.async.cg 的硬要求，代码里带运行时守卫（与 gemv scale 行的
     err-716 教训同源），未对齐就不发（自动退回 gmem 读）；② `pf_ok/pf_slot/row_base/g_begin`
     跨整个 prologue 存活（+1~2 寄存器），`__launch_bounds__(1024)` 的 64 regs 上限只会
     spill 不会 701，需 cuobjdump 复看；③ 收益上界 = 激活 staging 的时长（~1-3µs 量级），
     **不是** task 里的 −0.16ms 那种量级——那只在"整行先行"下才成立，而整行会被占用率吃回去。
   - 验证：本机无 nvcc（`cargo check -p ferrite-models` 通过；`.cu` 需远端 `build.sh 103a`
     重编，⚠️ 不重编 `.so` 会被 build-id 门禁拒启）。A/B：`DSV41_GATEUP_CPASYNC=1`（默认）
     vs `=0`，同窗口，人眼文本 + nsys `expert_gemv_fp4_batched_kernel` 每步 ms。
0.57 **P4.2 完整 cp.async 流水线 —— expert gateup 多组在飞（2026-09-11 已落地代码，待实测）**：
   §0.58（P4）只预取 group 0；主循环第 2..nv2f 组仍是串行 LDG，而 gateup 的实测症状是
   **22.2µs/call、443 GB/s、IPC 0.8/4（80% issue 槽停等）**，地板 1.3µs ⇒ 19-26x 差距，
   根因是操作数供给（每 warp 一个 group 只有 ~120 条指令的工作量，却要等 ~600 cycle 的 HBM
   载入）。寄存器型 unroll 救不了：LDG 目的寄存器挂在消费者 scoreboard 上，且
   `__launch_bounds__(1024)` 把 regs 钉在 64/thread。**cp.async 把载入从寄存器 scoreboard
   上摘下来**，等待变成 `cp.async.wait_group N` 的组计数器。
   - **实现**：env **`DSV41_GATEUP_PIPELINE`**（1 = P4 现状 / 2..5，**默认 2**，clamp 到
     `kGateUpPfDepthMax=5`）→ host `dsv41_gateup_pipeline()` → **新增内核模板参数
     `PDEPTH`**（`cp.async.wait_group N` 的 N 是立即数 → 编译期常量；launcher 按 clamp 后
     的深度选实例化，共 `ILV × PDEPTH(1..5)` 10 个）→ 每 warp 的 `s_pf` 变成 **PDEPTH 个
     512 B 槽的 ring**，`s_ks` 顺延 `nwarps*512*PDEPTH`。
   - **流水线语义**：prologue 发 PDEPTH 组（每组一次 commit，越界也 commit）；循环第 i 组
     `wait_prior(PDEPTH-1)` → 读 slot `i%PDEPTH` → **立刻**把第 i+PDEPTH 组发进刚空出的
     槽 → commit。尾部用**空 commit 补位**，保证"消费第 i 组前已 commit 恰好 PDEPTH+i 组"
     这一等式成立（否则 `wait_prior(PDEPTH-1)` 在尾部覆盖不到第 i 组）。循环尾 `wait_prior(0)`
     收掉空 commit（立即返回）。
   - **lane-local 拷贝（关键设计）**：`dsv41_gateup_pf_group<ILV>` 让**每个 lane 只拷自己
     要读的字节**——ILV：lane L 拷 `src + gi*512 + L*16` 的 16 B（一条 cp.async.cg）；plain：
     lane L 拷 gate `gi*256 + L*8` 与 up `gi*256 + L*8` 各 8 B（两条 cp.async.**ca**，因为
     cp.async.cg 只有 16 B 一种）。⇒ **不需要每个 group 的 `__syncthreads()`/`__syncwarp()`**，
     拷贝完成即对消费者可见。这也是把 P4 的 plain 路径从"跨 lane 16 B 拷贝 + barrier 发布"
     换成 8 B lane-local 的原因（smem 内容逐字节相同）。
   - **原始 PTX 而非 `__pipeline_*`**：CUDA header 的三个原语**没有 "memory" clobber**，
     编译器可以把 smem 读挪到 copy 发射之前——而 ring 的 copy 目标正是刚读过的那个槽，
     顺序反了会静默读错组（不崩、不报错）。故 `pf16/pf8/commit/wait_prior<N>` 全部自写
     asm 并带 `: "memory"`。
   - **smem 与占用率（本次复核推翻了旧结论）**：`smem = dim*4 + 256*8 [+ nwarps*8] +
     nwarps*512*PDEPTH`。生产形状（dim=5120、rows=8、ksplit=2 ⇒ nwarps=16、nv2f=10、
     每 warp 切片 5 组）：D=1 **30848 B（与 P4 完全一致）**、D=2 39040、D=3 47232（<48 KB
     默认上限）、D=4 55424、D=5 63616（>48 KB，需 opt-in）。⚠️ **占用率不是约束**：生产
     grid = (inter/rows, slots) = (40, 6) = **240 CTA / 148 SM ≈ 1.6 CTA/SM**——是 **grid
     受限**，不是资源受限，所以 ring 长出来的是没人用的余量。§0.58 里"整行 staging 会把占用率
     换掉"那句写在满 grid 的形状上，**在这个形状不成立**，这正是 D 可以开到 5（= 整个切片在飞，
     warp 级 MLP 上限）的原因。
   - **D=4/5 的 opt-in**：`dsv41_gateup_pf_smem_cap(ilv)` 按 **(device, ilv)** 缓存
     `cudaDevAttrMaxSharedMemoryPerBlockOptin` 并 `cudaFuncSetAttribute(<ILV,4|5>)`——照抄
     `expert_gemv_fp4_down_reduce_kernel` 的 per-device carve-out 修法（`cudaFuncSetAttribute`
     是 **per-context**，只设当前 device 会让其余 7 个 rank 停在默认值）。opt-in 失败 ⇒
     cap = 48 KB，launcher 把 pd clamp 回去，**launch 永不失败**。深度还会被"本 warp 切片组数
     `ceil(nv2f/ksplit)`"再 clamp 一次（超过切片的深度纯浪费）。
   - **逐位等价**：同字节、同 lane 偏移、同消费顺序、同 fma 链顺序，只有"从哪块内存读 / 何时
     发射拷贝"变了。PDEPTH=1 的代码路径与 §0.58 **逐字保持**（含 plain 的延迟 uw 载入），
     供 A/B 基线。
   - 验证（本次）：远端 nvcc 13.2 `-gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math`
     **10 个实例化全部编过**；寄存器：`<ILV,1>` 64/0 spill（= P4）、`<ILV,2..5>` 64 regs +
     8-12 B spill（2-3 寄存器，`__launch_bounds__(1024)` 的 64 上限所致，与 §0.58 同源）、
     `<plain,*>` 63-64/0 spill。PTX 复核：`<ILV,1>` 只有 1×`wait_group 0` + 1 条
     `cp.async.cg`；`<ILV,2>` `{wait 0 + wait 1×5}` + 7 条 cg16 + 7 commit；`<ILV,5>`
     `{wait 0 + wait 4×5}` + 10 条 cg16；`<plain,5>` 20 条 `cp.async.ca` 8 B。
     ⚠️ 本机无 GPU/nvcc、远端无 cuobjdump ⇒ **没有实跑、没有 SASS spill 定位**。
   - 待 user 实测：`DSV41_GATEUP_PIPELINE=1|2|3|5`（`=4` 可省）同窗口 A/B，看
     nsys `expert_gemv_fp4_batched_kernel` 每步 ms + `ncu --set full` 的 issue/停等分解。
     ⚠️ **必须重编 `.so`**（`bash kernels/cuda/build.sh 103a`，`.cu` 改了不重编会被 build-id
     门禁拒启）。
0.55 **P1 staged gate 的落地修正（2026-09-12）**：`6fc9a1a` 引入的 `DSV41_GEMV_A32_STAGED`
   当时只写了**调用**（`a32_direct = ... && !dsv41_gemv_a32_staged();`），该函数在全仓库
   **没有任何定义**（内核里也不能读 env）⇒ HEAD 的 `.cu` 实际**编不过**，且即便编过，
   launcher 侧 `dsv41_gemv_sa_bytes` 没有同门 ⇒ 内核多算一个 k 字节槽、host 不 reserve，
   尾部所有指针短 k 字节（越界）。现已按既有模式补齐：host gate `g_gemv_a32_staged`
   → `GemvCore.a32_staged`（6 个 M=1 launcher 全部接线）→ 内核 `&& !a32_staged`，
   并把 `dsv41_gemv_sa_bytes(k, norm_fuse)` 加上 `&& !g_gemv_a32_staged`（**布局耦合，
   两边必须同时改**）。
1. **`down_reduce` +0.31ms 的定案**：隔离微基准（同一 kernel，HEAD vs `01291b2^`），或 revert 后重采 profile。**这是唯一挡住 0.31ms 回收的事。**
2. **AR v5 的隔离绝对值**：0.66（v2 约定，被 §3 反证支持）vs 1.49（`86af349`，host-barrier 口径）。需要 device-side v5 的隔离测量。
3. **每步图节点数**：`hex/window` 类小核 + AR 246 节点 + 节点尾延迟（≈0.9ms，v2 遗留）——用 `cuda_gpu_trace` 或图节点数直接量。
4. **`gemv_bf16` 两族拆分**：把 9 次 lm_head/engram（≈45µs）单独归因，看 HEAD_SLICE 是否还有空间。
5. **`gemm_fp8_gemv` 的 mean/median 尾巴**（9.5 vs 7.9µs）来自哪几步：若是前几步 warmup，则生产口径应再降。
6. **gateup 的 CTA 粒度 vs warp 数（`DSV41_GATEUP_ROWS`，已落地，见 §4.1）**：`=8|4|2|1` 的 A/B
   （逐位一致，唯一变量是 CTA 形状；预期 flat——flat 即证伪 "1.6 blocks/SM 是真瓶颈" 的读法），
   **同轮**抓 `cuobjdump --dump-resource-usage`（`expert_gemv_fp4_batched_kernel<true|false>` 的 regs/thread）
   + ncu `sm__warps_active.avg.pct_of_peak_sustained_active`。这两个数决定下一步是 K-split（regs 低、grid 欠填充）
   还是 MLP/寄存器路线（regs 高、已近可驻留上限）。

---

## 8. 最终机会扫描补遗（2026-09-11 晚，纯代码复核）

逐核过了一遍 §1/§5 全表 + `chain_dev.rs::step_body`（:1688-1940）与 `layer()` 的全部 `self.dev.*` 调用点（162 处），
**除下面 3 条外，每个核都能对上一条已落地优化或一条明确的关闭决策**：

| 未覆盖项 | 现状（代码 + profile） | 判定 |
|---|---|---|
| `argmax_kernel` + `argmax_xchg_v5_kernel` | §1 #19 记 **1 次 / 59.1µs = 0.059ms**（**该行是两核聚合**：`dsv41_argmax_sliced`（`dsv41_kernels.cu:3930-3943`）连续发 `argmax_kernel<<<1,1024>>>`（:3936）+ `argmax_xchg_v5_kernel<<<1,1>>>`（:3939），nsys 必有**两行**，`argmax_xchg_v5_kernel` 在 §1/§5 表里查无此行）。**代码级预算（2026-09-11 复核）**：local 归约 16160 = 1024×15+800（尾部仅 1 次迭代失衡，可忽略），单 CTA shuffle+32 项 smem ⇒ ≈5µs ✓ 已在地板；xchg 的机械部分 = 8 store + **8 `atomicExch_system`** + 3 fence + epoch + 8 poll + 8 读，按本项目自测标定（`ferrite_kernels.cu:8369-8373`：**8 次串行 atomicExch_system 仅 0.4-0.8µs**）⇒ **机械总计 ≈5-10µs**（含 `<<<1,1>>>` 图节点）。**故 ≥45µs 是 poll 等最慢 rank** = 每步唯一"lm_head slice gemv 之后的跨 rank 汇合点"，协议地板。可回收项只剩 :3894 原子改 store（~0.5µs，与 AR 的 :8368-8377 等价）和 :3901 `__nanosleep(200)`→32ns 自适应（对齐 AR :8393-8396），合计 ≤1µs ⇒ **关**。若仍要复核，探针应量 **gemv_bf16 各 rank 的起止时间差**（漂移来源），而非只拆两核 | **关闭（可回收 ≤0.001ms）** |
| `quant_kernel<0>` 残差 126× | §1 #10 = 0.19ms，per-call 1.5µs 已在 launch/图节点地板（5120 元素实算 ≈0.1µs）。`rmsnorm_q`（40）、`o-rope-q`（40）、`sparse-o-rope` 已各吃掉一批；**剩下的 ~3 次/层只能走"生产核 epilogue 直出 fp8"**（同一模式已被证明 3 次）——单点 ≤0.02ms，是**模式级残差**不是单核 | **模式级，边际** |

> ⚠️ **2026-09-11 explore 逐点复核（读码）——"126 / ~3 次每层"是 `db2917501`（16:48）快照，已被后续 5 个 producer-fp8 commit 清掉。**
> `db2917501` 树里 **没有** `DSV41_OROPE_Q`（17:49）/`DSV41_NORM_FUSE`（18:12）/`DSV41_SPARSE_OROPE`（19:14）/`DSV41_WOB_F32`（19:22）/engram-f32（f102b21，19:22）/`DSV41_HC_TAIL_SPLIT`（18:06）——正是它们把 §1 的 126 打下去。
> HEAD 逐 site 核对（`chain_dev.rs`）：`xn`(wq_a/wkv + moe shared w1/w3)→T1（`hc_front_split` EARLY，:5730）；`qr`(wq_b + idx_wq_b)→`NORM_FUSE`（`lin_rope_norm`，:1273）或 rmsnorm_q T2（:2682）；`o`(wo_a)→`sparse_attn_orope`（:3094）/`apply_rope_q`（:3146）；`wo`(wo_b)→`gemm_fp8_mx_f32`（:3282）；`ex_act`(shared w2)→`swiglu_limit_q`（:4209）/**`DSV41_SWIGLU_FOLD`**（默认 ON：shared w2 的 gemv prologue 现算，该 launch 整体消失）；`eng_rows`→`gemm_fp8_mx_f32`（:1409）。**六个生产者全部直出 fp8，默认 ON，符号在树内**（`dsv41_kernels.cu:5730/3347/4033/3520`、`dsv41_glue.cu:877`）。

### SWIGLU_FOLD（2026-09-12，small-kernel-merge #2）
- **对象**：共享专家的 `swiglu_limit_q`（`dsv41_glue.cu:217`，1.7µs × 40/步）——读 `ex_act` f32 [2·`sh_il`] → silu+clamp → fp8 `(xq,xsc)`，其唯一消费者是共享 w2 的 M=1 GEMV。
- **做法**（NORM_FUSE 同构）：新 launcher `dsv41_gemm_fp8_mx_swiglu`（`dsv41_kernels.cu`，紧随 `dsv41_gemm_fp8_wo_pair`）+ `GemvFusion.gu`/`swiglu_limit` 两个字段；`gemm_fp8_gemv_kernel` 里 `gu != nullptr` 的 prologue 分支把 swiglu+量化写进 `s_a`/`s_as`。`a32_direct`/`act_async`/`s_as` 拷贝/`s_a` staging 四处 guard 都加了 `gu == nullptr`（与 `qr_raw` 并列）。
- **block 宽度**：**不**强制 32 warps（参考 amax 是 per-warp 的 32-lane 树，无跨 warp 归约）⇒ 保留 `dsv41_gemv_warps_for(n)`，与调用方原本的 standalone `gemm_fp8_mx` 同 grid/同 warps ⇒ `out` 逐位。smem 用 `dsv41_gemv_sa_bytes(k, true)`（NORM_FUSE 的 norm_fuse 语义：`s_a` 槽常驻）。
- **ABI**：`(gu, limit, w, w_scale, bias, out, n, k, epi_add, stream)`，stream 末位；decline 返回 2（stale .so / mode≠4 / `k%32` / null）。
- **接线**（`chain_dev.rs`）：`DSV41_SWIGLU_FOLD`（默认 ON，`=0` 回退）⇒ `gemm_fp8_mx_swiglu_on(ex_act, cfg.swiglu_limit, w2, w2s, null, dst, dim, sh_il, epi_add, sh_st)`；`dst`/`epi_add` 复刻 A5 决策（`!dual && moe_epi_add()` ⇒ 直接进 `s.o`，否则写 `s.ex_out` 再合并）。它**同时顶替 A4 与 A5**：`act_q`/`fused` 都被 `sw_folded` 短路。只覆盖共享专家（routed 由 `DSV41_GATEUP_FUSE` 负责，且 routed down 读 `ex_act_b`，与本路径无关）。
- **注意**：**未**回写 `ex_act` 的 f32（参考 kernel 会 `row[i]=v`）——`ex_act` 在该 kernel 之后无读者（已验证 `chain_dev.rs` 全文件），是死值。
- **与 sh_pair 的关系**：`dsv41_gemm_fp8_sh_pair`（chain-pair-batch 链2，`DSV41_SH_PAIR` 默认 OFF）把 `w1w3+swiglu+w2` 三合一；本折叠只合 `swiglu+w2`（保留已优化的 `gemm_fp8_mx2` 出 w1w3）。两者互斥于同一 launch，sh_pair 需 grid-sync 驻留、尚未接线（`sh_pair()` 无调用点）。
> `STATUS.md:6393` 的 quant-final-sweep（同为读码口径）独立得出同一结论：**剩余 = engram 2（已由 f102b21 消）+ 4 次未知（候选 idx-source fallback）≈ 尾巴清扫级，总收益 <0.01ms**。⇒ 本项**无可摘的 >20 次**，不实施；要收尾只剩一次 HEAD 上的 nsys 复核（旧 126 已失效）。
| `comp_placeholder_kernel` | §5 列 30 次 / 1.0µs = **0.030ms**，无决策。它写的是 `idxs[win+j]`，与 `ring_win_fused_kernel` 写的 `idxs[0,win)` **同一缓冲、同一 win**（`chain_dev.rs:3001`）⇒ **可折进 ring_win_fuse 的 epilogue**，30 个 launch/节点消失 | **✅ 已落地（B3，`DSV41_COMP_PLACEHOLDER_FUSE` 默认 ON）**：新符号 `dsv41_ring_win_fuse_ph`（`dsv41_glue.cu:794/825`）+ `device.rs::ring_win_fuse_ph` + `chain_dev.rs` 的 `ph_ok`/`ph_fused`（:2935/:3075）。本表数据 tree 早于落地它的 `2a16b48` ⇒ §5 的 30 行已消失。⚠️ 需重建 `.so` 才生效（旧 `.so` 无该符号 ⇒ 自动回退，行为不变） |

**以下为逐条排除（已覆盖 / 不适用）**：`gated_rmsnorm_kernel`、`layernorm_affine_kernel` —— 不在 DSV4.1 解码链
（只在 `ferrite-exec`/`ferrite-kernel` 的其他模型路径）；`bf16_to_f32_kernel` —— **0 次/步**，仅 load-time（`load.rs:477/579`）；
`f32_to_bf16` —— 只在 `DSV41_CUBLAS_M1=1` 的 `lin_bf16` 里（默认 OFF）；`window_idxs`+`ring_append` —— 已并入
`ring_win_fused_kernel`；`kpool_compress(_batched)` —— 不在单序列 profile（batched DSA 链 `cuda.rs:4645`）；
`sparse_attn_pf`（0.34ms）—— 已审（STATUS `5085`：0.34% 占用；key-split × 3 深预取已落地为默认）；
`indexer_topk_kernel`（Step B，0.027ms）/`compressor_pool`（0.021ms）—— 略过阈，未单独优化，潜力 ≤0.02 量级；
`embed_expand_dev`（0.012）、`engram_gather`（0.005）、`index_k_publish`（0.005）、`compressor_state`（0.005）—— **关闭（<0.02）**。

---

_事实来源：`/tmp/dsv41-prof-v3` + `/tmp/dsv41-prof-v3b`（本次两次 nsys，16:54–17:01，tree `db2917501`，HEAD 默认 env）；
`/tmp/dsv41-prof-v2-mixgate`（= 原 `/tmp/dsv41-prof-v3`，v2 文档的采集目录，tree `60a01a5`，MIX_GATE=ON）；
serve 交叉验证 `/tmp/ab_cse.log`（9.55–9.60）、`/tmp/ab_mg0|mg1`（MIX_GATE A/B）、`/tmp/ab_dv320`、`/tmp/ab_r27`；
代码差 `git diff 60a01a5..db2917501 -- kernels/`；
`crates/ferrite-dsv41/STATUS.md`（`86af349` 的 nsys-v3 段、`530153f` 的 round-29/30 判定、round-28 的 dv320）；
`docs/agent/dsv41-kernel-inventory-v2.md`、`docs/agent/roadmap-200-tokps.md`、`docs/agent/stage-b-execution.md`。_
