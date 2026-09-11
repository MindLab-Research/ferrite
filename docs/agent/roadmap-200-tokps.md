# 200 tok/s 执行计划 — 操作层

**口径**：B=1 decode 稳态 p50，**同二进制背靠背 A/B**；判据 = 四段文本（Paris/Tokyo/1+1=/静夜思/出师表）逐字 + `faults=0` + p50。
剖析固定 `DSV41_AR_V5=0 DSV41_GRAPH_STEP=0`（v5 自旋 × nsys = 300x 病态）。目标：5ms/step（每层 0.11ms）。

## 1. 当前状态

当前正确基线 **10.16ms / 98.4 tok/s**（会话起点 13.28ms，+30.7%）。
融合路径 GATEUP+DOWN 全开 = **9.44ms / 105.9 tok/s**（数值 bug 已定位：两侧默认值分裂，已统一 OFF，见 Stage A P0）。

| 已落地（默认 ON）| 生效点 |
|---|---|
| shared expert TP 切分（`DSV41_SHARED_TP`）| chain_dev.rs:2210 `moe`；−1.43ms |
| lm_head 词表切分 + 跨 rank argmax（`DSV41_HEAD_SLICE`）| −0.35ms |
| sparse 3 深预取（`DSV41_ATTN_PF`）| sparse_attn_pf_kernel dsv41_kernels.cu:484；−1.09ms |
| **sparse 3 深预取 × key-split（`DSV41_SPARSE_SPLIT`，默认 C=4）** | 同一 kernel 加了 key 分块：grid=(C,b·m,h) + merge kernel；C=1 与 pf **位级一致**，微基准 per-slot 78→10.7ns。**2026-09-11 sparse-attn-v8**：`sparse_attn_orope` 的 split 分支不再 decline（rope+fp8 epilogue 已折进 `sparse_attn_merge_kernel`），故 split 与两个融合可共存；`DSV41_ATTN_PF_SPLIT` 显式设置时仍优先。**2026-09-11 sparse-attn-v9**：merge 选举折叠（`DSV41_SPARSE_MERGE_FOLD`，默认 ON）——split 每组 (b·m,h) 的 C 个 block 用 per-group ticket 选最后完成者跑 merge，每层 2 launch → 1，位级一致（台架 fold ON/OFF 直接对拍）；=0 回退两 kernel |
| P1 route_topk hist 死码 + P2 zero 冗余 | −0.27ms |
| MoE 批化 / NR / SH_EXP_MX2 / MIX_GATE / FUSE_C / FUSE_B1 | 均为默认 ON（勿再误关）|

剩余分解（10.65ms 口径，AR=v5）：

| 项 | ms/步 | 状态 |
|---|---|---|
| gemm_fp8_gemv_kernel (dsv41_kernels.cu:1722) | 2.18 | **指令级地板**：91% 固定项（~11.6µs × 171）|
| expert_gemv_fp4_batched (dsv41_experts_mxf4.cu:701) | 1.78 | **L1TEX 管道地板**，内层杠杆全阴性 |
| hc 链（hc_mix_dots :3048 + hc_mixes_tail :3109）| 1.59 | sinkhorn 可藏 |
| gemv_bf16_kernel (dsv41_glue.cu:317) | 0.74 | lm_head 切分后 |
| AR v5（store ferrite_kernels.cu:8324 + pubred :8366）| 0.66 | **NVLink 协议地板**（store 的 160KB 远程写 + stamp 传播）；stamp 已改**并行写**（thread r → peer r，barrier 后再 fence/推进 epoch）+ poll 首轮 32ns 自适应退避（2026-09-11）。**但 reduce 不是 NVLink 项**：`staging_local` 是本 rank 自己的 buffer，8 个 peer 的 partial 在 store 阶段已被写进来 ⇒ reduce 是**本地读**（40KB/rank 级），其 1.5-2µs 主要是 **grid 失衡**——n=5120 ⇒ n4=1280 float4，旧 launch 是 `ceil(5120/1024)=5` block × 1024 线程、映射 `i4 = blockIdx*1024+tid, step=5120` ⇒ block0 干 80%、block1 干 20%、block2-4 全空，8192 次 load 压在单 SM。**2026-09-11 已修**：三个 launcher（`ferrite_p2p_ar_v5` / `ferrite_p2p_ar_pubred_v5` / `ferrite_p2p_ar_v5_hcpost`）改成 `threads = 256`（`world > 256` 时回退 1024，stamp/poll 要求 `blockDim.x >= world`）+ `blocks = ceil(n4/threads)` ⇒ n=5120 得 5 个满块 × 256 线程 = 1280 线程、每线程恰好 1 float4 × world rank，摊在 5 个 SM。**2026-09-11 二次收窄（ar-pubred-optim）**：256→64 线程 ⇒ `blocks = ceil(n4/64)` = 20 个满块 × 64 线程 = 同样 1280 线程，但摊到 **20 个 SM**（reduce 是纯本地读的延迟受限项，4× memory-level parallelism）。5 个 launcher 统一走 `ferrite_ar_v5_block_threads(world)` / `ferrite_ar_v5_grid_blocks(n, threads)`（`world ≤ 64` 取 64；`world > 64` 回退旧的 256/1024，守住 stamp/poll 的 `blockDim.x >= world`）。**逐位等价**：i4 的所有权与升序 rank 求和序都没变（无跨线程归约）。预期 −0.08~0.16ms。store kernel 同一映射（同 n4=1280，见 `:8418`）同改。**2026-09-11 store peer 并行化**：store kernel 的 `for (rr...)` 串行 8 次远程写改为 `gridDim.y == world`（`dim3(blocks, world, 1)`），每 block 只写 1 个 peer ⇒ 5 → 40 blocks、每线程 1 次远程写（原先单线程串行 8 次，纯 latency-bound，5.9µs × 82 = step 里最大的非 gemm 项）。`step` 仍只跨 gridDim.x，故 (i4, rr) 地址/值逐位不变。4 个 launcher（`:8463/:8492/:8684/:8711` 的 store 调用）改 grid；pubred 保持 1-D grid 不变（改它会重复归约）。**未覆盖融合路径**：`DSV41_AR_STORE_FUSE=1` 时 store 折进 producer epilogue（`dsv41_kernels.cu:3667`，仍串行 rr）。预期 −0.2ms。**2026-09-11 stamp 折进 store（`DSV41_AR_STAMP_FOLD=1`，默认 OFF，A/B 臂，已实施待验）**：stamp 从 pubred 的 block 0 移到 store kernel 的**最后一个 block**（`p2p_ar_store_v5_kernel` 末尾），pubred 退化为 poll+reduce（`stamp_in_store=1`）——省下 store→pubred 的 kernel 边界 + 一次 stamp 往返（stamp 原先在数据已 peer-visible 之后一晚才发出）。末块判定用**永不 reset 的单调 arrival 计数**：`arrive[1]` = 原子递增总到达数，`arrive[0]` = 上一轮末块写入的"本轮起点"，末块 = `prev == base + total - 1`（`total = gridDim.x*gridDim.y`）——**单调比较而非清零**（v2 `:8077` 的 in-kernel reset 竞态正是三次死锁根因），且按**本轮自己的 total** 比较，故 b1/b2/b4/b16 不同 grid、engram AR 不同 n 都能跨轮切换。`arrive` 两字落在 staging 预留的 `ctr_at+4/+8`（alloc 已 `zero_at`，只在 store kernel 内写，无 host 流量、无需 reset；`argmax_xchg_v5` 轮不碰它，因只有 store 轮递增）。正确性：每线程 `__threadfence_system()`（`p2p_ar_down_kernel` 同形）保证 stamp 不越过任何 peer store；parity 双缓冲安全性论证不变（stamp(k+1) 仍由 rank i 的 store(k+1) 发出，stream-序在 pubred(k) 之后）。GLM 共享路径传 `nullptr`/0 保持旧协议。预期 −0.1~0.25ms。|
| sparse_attn | 0.32 | 3 深后 |
| 残差（节点尾延迟）| ~1.15 | 700 节点 × ~1.5µs ramp-down，**只能靠减少节点数消** |

## 2. Stage A：增量优化 10.16 → 8.8ms

**P0（必须先做）**：统一融合 env 两侧默认——`.cu` 的 `g_fuse` return 0 + Rust `unwrap_or(false)`（chain_dev.rs:2407/2446）；`DSV41_AR_STORE_FUSE`（:1923）逐位验后翻 ON。铺路 −0.08ms。

5 项融合补丁（均已产出，默认 OFF；逐个 A/B，**通过才翻 ON**）：

| # | env | 融合点 | 预期 | 验证口径 |
|---|---|---|---|---|
| 1 | `DSV41_GATEUP_FUSE` | expert_gemv_fp4_batched(:701) 出口 2·inter→inter，swiglu 做 epilogue；删 swiglu_limit_batched(dsv41_glue.cu:727) | −0.4 | 四段文本 + faults；**逐位**：K 循环照抄、g/u 双独立归约链、clamp 幂等 |
| 2 | `DSV41_DOWN_FUSE` | expert_gemv_fp4_down_reduce(dsv41_experts_mxf4.cu:1027) 合并 down(:701 写 scratch)+moe_down_reduce(:986) | −0.3 | **逐位**：`__fadd_rn/__fmul_rn` 显式分开乘/加（fast_math 防 FMA 收缩），slot 升序累加 |
| 3 | `DSV41_AR_STORE_FUSE` | p2p_ar_store_v5(:8117) 折进 producer epilogue（attn=gemm_fp8_gemv / moe=add_inplace 或 moe_down_reduce）| −0.08（−80 节点）| ⚠️ GEMV 加参数触发重编译 ⇒ 新 .so `AR_ST=0` vs 旧 .so token **逐位一致**（fast_math ptxas 可能改结合）|
| 4 | `DSV41_SWIGLU_Q` | swiglu_limit_q(dsv41_glue.cu:176) 直出 (xq,xsc) | −0.06 | **逐位**（`tests_dsv41_glue.cu` 的 swiglu_q 用例：xq/xsc/f32 对 `swiglu_limit`+`dsv41_quant_fp8` 逐位）；**已翻默认 ON**——原"round-18 数值 bug"是误归因（A4 代码 `f3b1be1` 才进树，而该 commit 报的正是 round-18 那次跑分）|
| 4b | `DSV41_SWIGLU_FOLD`（small-kernel-merge #2）| **共享专家**的 `swiglu_limit_q` 整体搬进 w2 GEMV **prologue**（`dsv41_gemm_fp8_mx_swiglu`，chain_dev.rs 的 sh_w2 调用点）⇒ 该 launch（1.7µs × 40/步）与它的图节点消失 | **−0.068**（预估）| **逐位**：prologue 用 `swiglu_limit_q_kernel` 的逐项算式（clamp + silu + per-warp 32-lane amax 树 + `fast_round_scale` + clamp + e4m3），只因 `k%32==0` 且 blockDim%32==0 而 lane→element 映射相同；consume loop 与 standalone `gemm_fp8_mx` 同 grid/同 warps ⇒ `out` 逐位。**默认 ON**，decline（stale .so / mode≠4 / `sh_il%32≠0`）回退 A4/A5 老路径 |
| 5 | `DSV41_MOE_EPI_ADD` | gemm_fp8_mx_add 把 add_inplace 折进 lane-0 epilogue，直写 s.o | −0.08 | 结合律 `o+(acc+bias)` 不变 ⇒ **逐位** |

**补差（实测后只剩一项）**：`quant_kernel` 生产者直出 fp8（−0.28）；~~hc sinkhorn 藏进 collapse 重叠（−0.21，探针 /tmp/tail_probe）~~ **实测否决 → 0.018ms/步**（`/tmp/tail_phase_probe.cu`：可藏窗口 collapse P1 = 0.46µs ≪ sinkhorn 6.5µs）。
⇒ 5 补丁 −0.92 + 补差 −0.28 ≈ **落点 9.0ms（~112 tok/s）**。

## 3. Stage B：结构性 8.8 → 7.2ms

- **xn-megafuse（2 launch 替代 3）**：读 `s.xn` 的 5 投影族不能一 launch（`s.xn` 复用缓冲，两簇被 AR#1 隔开）。现实 = 2 launch：A 簇 wq_a+wkv+idx_wp（1824 行）、B 簇 gate+sh w1/w3（960 行）。真正省的是 idx_wp 的 8-block 小 slot ⇒ **−0.25~0.5ms**（非 −1.0）。激活 staging 3 遍→1 遍。
- **跨层流水（cross-layer-pipe）**：字面"ffn mixes 挪到 L+1"不可行（届时 s.h 已被 hc_post 覆写）。修正 = 每次 front 拆 ⟨A⟩collapse_norm（留原位，MoE 依赖 xn）+ ⟨B⟩mixes dots+tail（推迟，与 AR 合并进同一 launch——AR v5 只 5 个 1024 线程 block，poll 窗口上百 SM 空闲）。**−0.3~0.5ms**。
  ⚠️ 陷阱：AR 核 `step = gridDim.x*blockDim.x` 会因新增 block 错位 ⇒ 必须按 `blockIdx.x < ar_blocks` 分区；post/comb 单份共享缓冲需双缓冲（单层流水 3 槽 i%3 安全）。

## 4. Stage C：persistent 7.2 → 5.3ms

段级 persistent（AR 是跨 rank 硬边界，整模型一核不可能）：**40 层 × 3 段 = 120 节点**（vs ~700），残差 1.15 → 0.18ms。段核按 `dsv41-layer-fusion.md` 段 A/B/C（hc 族 T=64 → 320 blocks）。

| 阶段 | 内容 | 预期 | 风险 |
|---|---|---|---|
| P1 | 段 C 融合 hc_post+copy_h_back → hc_post_inplace（省 80KB D2D ×2/层）| −0.15~0.25 | 低 |
| P2 | 段 A 融合（hc 链 + attention 投影链一核，中间量留 smem）| −0.6~1.0 | 中 |
| P3 | 段 B MoE cooperative（gate→route→quant→gate_up→swiglu→down→reduce 一核）| −0.4~0.7 | 中 |
| P4 | 跨层流水（同 Stage B）| −0.3~0.5 | 中 |
| P5 | 零 AR 节点（store+stamp 折段末、pubred 折下段首，v5 协议改动）| −0.15 + 40 节点 | **高** |

回收 ≈ −2.5~3.0ms ⇒ **落点 ~7.2-7.7ms（130-139 tok/s）**。
铁律：AR 永远保留独立 launch 或 PDL 重叠；任何"省 kern"先证不产生**尾块长自旋**（hc-merge 单核 + ticket 自旋 = **+3.2ms** 回归，见第 14 轮）；融合核同编译单元或显式 `__fmaf_rn`（跨 CU 收缩差 = 1 ULP = 确定性文本变化）；每核配 `*_parity.rs`。

## 5. Stage D：5ms 突破

persistent 单独到不了 5ms——**真实工作地板 ≈ 5.3ms**（expert_gemv_fp4 1.78 + hc 链 1.59 + gemv_bf16 0.74 + AR 0.66 + sparse 0.32，全部已到指令/管道/协议地板）。
5ms 需在 Stage C 之上**叠加**：① 段间流水（段 X 的 post 与段 X+1 的 pre 重叠）；② expert 内核机制重构（换非 GEMV 的新机制，非"再调块形状"——uint4/k-split/launch_bounds 均已实测阴性）；③ hc 链逐相位关断后重构（12.6µs × 90 次待拆）。**不承诺**。

## 6. 每阶段"不做"清单

- **Stage A**：不做 xn-megafuse / 段融合（先拿满增量收益）；不批量 sed 翻默认（`f3b1be1` 误关 7 门 = +8.7ms）；不给 kernel 加 blockDim（sparse/quant 会错结果）。
- **Stage B**：不做字面跨层搬运（会读错 s.h）；~~不给 sparse_attn 加 `DSV41_ATTN_SPLIT`（实测更差）~~ → **2026-09-11 修正**：当年更差是**丢了 3 深预取**，不是 key-split 无效——split 版已补上预取并成为默认（`DSV41_SPARSE_SPLIT`，默认 C=4；`DSV41_ATTN_PF_SPLIT` 显式值仍优先，见 §1）；不共用 `s.o` 的双重生命周期。
- **Stage C**：不做整模型单核；不把跨 rank 同步塞进单核；不拆 split=8（改部分和顺序）；不捕获含 `cudaMalloc`/host 交互的核。
- **Stage D**：不做 MTP（用户明令禁止）；不重试 k-split/uint4/T=4（均阴性）；不"变快=少算"（必逐位/text 验证）。
