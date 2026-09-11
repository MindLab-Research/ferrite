# DSV4.1 decode kernel 清单 v2 — 9.65ms 基线（融合后）

**一句话**：用远端 nsys（`/tmp/dsv41-prof-v3`，2026-09-11 16:1x 采集）拿到融合路径的**真实**per-kernel 分解：
剖析口径 10.65 ms/步/rank，换回生产 v5 AR 口径 = **9.69ms**，与同环境生产 p50（`ab_fon` 9.73ms）吻合 0.4%。
**最大发现**：`gemm_fp8_gemv` 仍是第一单项（1.98ms / 20.4%），但**共享专家核 `gemv_bf16_fp8x2` 已跃居第二（1.32ms / 13.6%）**——
它此前被"per-rank 平均"的 nsys 口径藏成 0.275ms（rank0 串行），TP 切分后才暴露为每 rank 的真实成本。
**第二大发现**：`gemm_fp8_gemv` 调用数 171 → **206 次/步（+35）**，gap-analysis 的 171 已过时，来源待归因。

---

## 0. 采集状态与口径（读之前必看）

| 项 | 值 |
|---|---|
| GPU 状态 | 检查时 **BUSY**（第 23 轮 def + 后续 A/B，8 卡 46-50% util，`dsv41-run` 存活）；16:08 该批 `ALLDONE` 后**转空闲**，随即抓取 |
| 采集 | `DSV41_GATEUP_FUSE=1 DSV41_DOWN_FUSE=1 scripts/dsv41_profile.sh 30 /tmp/dsv41-prof-v3`（= `fon` 融合路径）|
| 远端 tree | `/home/ubuntu/ferrite @ 60a01a5`（比本地 HEAD 落后 2 个 commit；**默认门仍是 OFF**，故必须显式开融合） |
| 二进制 | `target/release/dsv41-run` + `libferrite_kernels.so`，16:03 构建 |
| 剖析口径 | `DSV41_AR_V5=0 DSV41_GRAPH_STEP=0`（v5 自旋 × nsys = 300x 病态，必须关）|

### 口径三定律（本表所有数字的前提）

1. **AR 行是 host-barrier 伪影，不是生产成本。** v3 的 `ar_reduce 0.836 + ar_store 0.443 + ar_stamp 0.346 = 1.625ms` 是 `AR_V5=0` 路径。
   生产默认 v5 = **0.66ms**（store + pubred，gap-analysis 定案）。
   ⇒ 生产口径 = 10.65 − 1.625 + 0.66 = **9.69ms**；同环境生产实测 `ab_fon` p50 = **9.73ms**（差 0.4%，换算成立）。
2. **"ms/步/rank" 是吞吐口径，不是关键路径口径。** nsys 把 8 个 rank 的耗时求和再除 8。
   rank0 独占的串行工作（共享专家旧路径 = rank0 40×55µs = 2.2ms）在平均口径下只剩 0.275ms。
   ⚠️ **凡"只有部分 rank 干活"的优化，收益在平均口径下会消失，必须看关键路径。** 共享专家 TP 切分（−1.43ms）就是这样被藏的。
3. **多设备 nsys 的绝对耗时不可全信**（脚本原话）：逐次调用时间用隔离微基准复核；本表用于**份额排序**，不用于绝对定论。

---

## 1. 生产口径 kernel 分解表（融合路径，9.69ms ≈ 9.65ms 基线）

`calls/stp` = 每步每 rank 调用数；`µs/call` = nsys 逐次均值；`ms/stp` 非 AR 行取剖析值（与生产吻合），AR 行取 v5 定案值。

| # | kernel | 次/步 | µs/次 | ms/步 | % | 13.3ms 时代 | Δ | 变化原因 |
|---|---|---|---|---|---|---|---|---|
| 1 | `gemm_fp8_gemv_kernel` | 206 | 9.6 | **1.98** | 20.4% | 2.90 (171@17.0) | −0.92 | ① a32 smem 修复（mx launcher 补 `k*4`）② e4m3→f32 **LUT**（float2，一次 LDS.64 出两 nibble）③ per-row **scale staging**。**⚠️ 调用数 +35 待归因** |
| 2 | `gemv_bf16_fp8x2_kernel`（MIX_GATE: gate + 共享专家 w1/w3） | 40 | 33.0 | **1.32** | 13.6% | 0.275 (5@55)**\*** | +1.05\* | **shared expert TP 切分**：rank0 独算全 inter → 每 rank 算 `inter/8`。\*平均口径上升但关键路径 2.2ms→1.32ms（−0.88） |
| 3 | `expert_gemv_fp4_batched_kernel`（gate_up+swiglu 融合） | 40 | 25.7 | **1.03** | 10.6% | 1.78 (80@22.2) | −0.75 | gateup+swiglu 融合：每 warp 产 (gate_i,up_i)、swiglu 做 epilogue、写 `inter` 而非 `2·inter`；down 分出去成独立核 |
| 4 | `hc_mixes_tail_kernel` | 80 | 12.4 | **0.99** | 10.2% | 0.97 | +0.02 | **未动**（hc-merge 第 14 轮 +3.2ms 回归后 gate OFF；sinkhorn 尚未藏进 collapse）|
| 5 | `expert_gemv_fp4_down_reduce_kernel<true>`（down+reduce 融合） | 40 | 17.2 | **0.69** | 7.1% | 新增 | — | down+reduce 合一：删 `grid.y` slot 维、升序 slot 累加（= reduce 数值契约）、`ex_down_b` 全程留寄存器；替代旧 down + `moe_down_reduce` |
| 6 | **AR v5**（store + pubred） | 80 | — | **0.66** | 6.8% | 0.66 | 0 | NVLink 协议地板 |
| 7 | `hc_mix_dots_kernel` | 80 | 7.0 | **0.56** | 5.8% | 0.59 | −0.03 | 4 warp 协同 staging |
| 8 | `gemv_bf16_kernel`（lm_head 切片 + engram） | 9 | 45.0 | **0.41** | 4.2% | 0.98 (44@22.3) | −0.57 | ① lm_head 1/8 词表切分 ② **MIX_GATE**（`chain_dev.rs:2324` `sh_via_mixed = mix_gate_shared() && gemm_bf16_fp8x2(...)`）把 gate+w1/w3 合成 `gemv_bf16_fp8x2` → 调用 **44→9**，与 fp8x2 的 **5→40** 数值对应（−35/+35）；精确拆分待代码/历史核 |
| 9 | `sparse_attn_pf_kernel` | 40 | 8.5 | **0.34** | 3.5% | 0.42 (10.5µs) | −0.08 | 3 深预取 |
| 10 | `quant_kernel<0>` | 166 | 1.6 | **0.26** | 2.7% | 0.28 (176) | −0.02 | P2 冗余 zero 跳过（调用 −10）；**纯固定成本，未到预测的 0.15** |
| 11 | `route_topk_kernel` | 40 | 5.2 | **0.21** | 2.2% | 0.21 | 0 | P1 只删了 `hist` 死码（`route_hist` 已消失），固定成本未降 |
| 12 | `dsv41_hc_post_inplace_kernel` | 80 | 1.9 | **0.15** | 1.5% | 0.15 | 0 | — |
| 13 | `rmsnorm_kernel` | 44 | 2.8 | **0.12** | 1.3% | 0.13 | 0 | — |
| 14 | `gemv_f32_kernel` | 7 | 16.4 | **0.12** | 1.2% | 0.11 | 0 | — |
| 15 | `apply_rope_kernel` | 88 | 1.3 | **0.11** | 1.1% | 0.11 | 0 | — |
| 16 | `rmsnorm_rope_kernel`（NR_FUSE） | 40 | 2.6 | **0.10** | 1.0% | 0.10 | 0 | — |
| 17 | `indexer_score_kernel`（Step A） | 4 | 19.8 | **0.08** | 0.8% | 新增 | — | indexer 两步走：2048 warp 在飞，替代 32 warp×64 波串行 |
| 18 | `quant_kernel<1>` | 40 | 1.6 | **0.07** | 0.7% | 0.07 | 0 | — |
| 19 | `engram_apply_kernel` | 2 | 32.5 | **0.07** | 0.7% | 0.07 | 0 | 仅 engram 层（1、14）|
| 20 | `argmax_kernel`（切片 + 跨 rank） | 1 | 59.1 | **0.06** | 0.6% | 0.06 | 0 | HEAD_SLICE 配套 |
| 21 | `add_kernel` | 40 | 1.4 | **0.06** | 0.6% | — | — | A5 未开，`add_inplace` 仍独立 |
| 22 | `swiglu_limit_kernel`（共享专家 w2 前） | 40 | 1.2 | **0.05** | 0.5% | 0.007 (5) | +0.04 | TP 切分后全 rank 执行 |
| 23 | `fp4_pack_kernel` | 40 | 1.1 | **0.05** | 0.5% | 0.05 | 0 | — |
| 24 | `ring_append_kernel` | 40 | 1.1 | **0.05** | 0.5% | 0.05 | 0 | — |
| 25 | `window_idxs_kernel` | 40 | 1.0 | **0.04** | 0.4% | 0.04 | 0 | — |
| 26 | `indexer_topk_kernel`（Step B） | 4 | 6.7 | **0.03** | 0.3% | 0.20 (50.1µs) | −0.17 | 打分段换成一次 load（Stage B 见 STATUS）|
| 27 | 其余 12 项（comp/engram/embed/…） | — | — | **~0.10** | ~1.0% | — | — | 见 §3 |
| | **合计** | | | **9.69** | 100% | 13.28 | −3.6 | |

### 家族汇总（给优化排序用）

| 家族 | ms/步 | % | 备注 |
|---|---|---|---|
| GEMV 族（gemm_fp8 + gemv_bf16×2 + gemv_f32）| **3.83** | 39.5% | gemm_fp8_gemv 91% 固定项已定论 |
| expert fp4 家族（gate_up+swiglu + down+reduce）| **1.72** | 17.7% | 13.3 时代为 1.918（1.78+0.054+0.084），净 −0.20 |
| hc 链（tail + dots + hc_post_inplace）| **1.70** | 17.5% | tail 单 block 串行是瓶颈 |
| AR v5 | 0.66 | 6.8% | 协议地板 |
| attention 杂项（sparse + quant + rope + rmsnorm + indexer）| ~1.1 | 11% | |

---

## 2. Top-5 剩余优化机会（按预期收益排序）

| 排名 | 机会 | 目标项 | 预期收益 | 阶段 | 依据 / 风险 |
|---|---|---|---|---|---|
| **1** | **MoE cooperative 段核**（gate→route→quant→gate_up→swiglu→down→reduce 一核，中间量留 smem） | expert 家族 1.72ms | **−0.4~0.7** | Stage C P3（但收益最大，可提前）| 省 ~440KB/层 global 往返 + 4 个节点尾延迟。⚠️ slot 升序累加是数值契约；同编译单元否则 1 ULP |
| **2** | **hc 链两招**：① sinkhorn 藏进 collapse 重叠（探针已部署 `/tmp/tail_probe`）② 段 C `hc_post`+`copy_h_back` → `hc_post_inplace`（省 80KB D2D×2/层）| hc 链 1.70ms（tail 0.99 + dots 0.56）| **−0.36~0.46** | ②= Stage C P1 | ② parity 既有，低风险；① tail 是**单 block 串行**（1 块/1024 线程 = 0.68% 占用，20 轮 sinkhorn + 7 barrier），探针测重叠可回收 ~0.2 |
| **3** | **cross-layer-pipe**：每次 front 拆 ⟨A⟩collapse_norm（留原位）+ ⟨B⟩mixes dots+tail（推迟，与 AR 合并进同一 launch——AR v5 仅 5×1024 线程 block，poll 窗口上百 SM 空闲）| 残差/节点延迟 | **−0.3~0.5** | **Stage B** | 字面"ffn mixes 挪到 L+1"**不可行**（届时 `s.h` 已被 hc_post 覆写）。⚠️ AR 核 `step = gridDim.x*blockDim.x` 会因新增 block 错位 ⇒ 必须按 `blockIdx.x < ar_blocks` 分区；post/comb 单份共享缓冲需双缓冲 |
| **4** | **xn-megafuse（2 launch）**：A 簇 `wq_a+wkv+idx_wp`（1824 行）、B 簇 `gate+sh w1/w3`（960 行）| gemm_fp8_gemv 1.98ms（206 次）| **−0.25~0.5** | **Stage B** | 5 族**不可能**一次 launch（`s.xn` 是复用缓冲，两簇被 AR#1 + 整段 attention 隔开）。真正省的是 `idx_wp` 那个 8-block 小 slot；`gemv_bf16_fp8x2` 不建 LUT/a32，可省的只有 fp8 激活 staging 一遍 |
| **5** | **quant 生产者直出 fp8**（−0.28）+ **AR store 融合**（−80 节点，−0.08）| quant 0.26ms + 节点数 | **−0.28~0.36** | Stage A 尾巴 | quant 是**纯固定成本**（1.6µs/次 × 166 = 0.26ms），只能靠消除调用；AR store 是纯逐元素拷贝，可融进 producer epilogue（补丁在 `~/.xbot/users/web-4/workspace/ar-fuse-store/`）|

**排序说明**：Top-2 是"当前表里最大且未被攻过"的两块（hc 链 + expert 家族合计 3.4ms，占 35%）；
Top-3/4 是 roadmap 定义的 Stage B 本体（结构性，8.8→7.2 段）；Top-5 是 Stage A 的零风险尾巴。
若严格按 roadmap 的 Stage B 范围，则优先级 = #4 → #3（两者合计 −0.55~1.0ms）。

---

## 3. 小核全清单（< 0.1ms，防止漏算）

| kernel | 次/步 | µs/次 | ms/步 |
|---|---|---|---|
| `engram_hash_step_kernel` | 1 | 17.3 | 0.017 |
| `embed_expand_dev_kernel` | 1 | 11.6 | 0.012 |
| `compressor_pool_kernel` | 4 | 5.3 | 0.021 |
| `comp_placeholder_kernel` | 30 | 0.9 | 0.026 |
| `compress_commit_kernel` | 4 | 2.1 | 0.008 |
| `dsv41_hc_collapse_norm_kernel` | 1 | 6.0 | 0.006 |
| `engram_gather_kernel` | 2 | 2.8 | 0.006 |
| `index_k_publish_kernel` | 4 | 1.2 | 0.005 |
| `compressor_state_kernel` | 3 | 1.6 | 0.005 |
| `rope_precompute_kernel` / `bf16_to_f32_kernel` | 0 | — | 0.000 |

⚠️ **`kpool_compress` 不在 `kernels/cuda/dsv41_kernels.cu`**（该文件只有单序列链，`dsv41_compressor_pool` / `compress_commit`）：它住在 `kernels/cuda/ferrite_kernels.cu:5851`（`kpool_compress_kernel`）与其 batched 版 `:6090`（launcher `:6702`），唯一调用点是 **batched DSA 链** `cuda.rs:4645`（`dsa_layer_dev_batched`）。同文件里的 `sparse_attn_pf_kernel`（`dsv41_kernels.cu:484`，launcher `:2413`）则是 **单序列** 路径（`chain_dev.rs:1899`），两者不共享数据链、不同口径，做重叠评估前先确认它们是否出现在同一次 profile 里。

---

## 4. 与 gap-analysis 推演对比：哪些准了 / 偏了

| 项 | gap-analysis 预测 | 实测（本表） | 判定 |
|---|---|---|---|
| `gemm_fp8_gemv` | 2.18 | **1.98** | ✅ **准，略优于预测**——LUT 比估计更狠（17.0→9.6µs，预测 11.6µs）。91% 固定项结论不变 |
| `expert_gemv_fp4_batched` | 1.75 | **1.03 + down_reduce 0.69 = 1.72** | ✅ 准（家族合计）；但**结构被融合拆成两核**，单行预测已不适用 |
| `gemv_bf16` | 0.74 | **0.41** | ✅ **优于预测**——MIX_GATE 把 40 次 gate 调用移走，额外 −0.33 |
| hc 链 | 1.59 | **1.55** | ✅ 准（tail 0.99 + dots 0.56）|
| AR v5 | 0.66 | **0.66** | ✅ 准（协议地板）|
| `sparse_attn` | 0.32 | **0.34** | ✅ 准 |
| `quant` | 0.15 | **0.26** | ⚠️ **偏乐观**——T1 未真正生效，只靠 P2 掉了 0.02；纯固定成本 |
| `route_topk` | 0.15 | **0.21** | ❌ **偏乐观**——P1 只删了 `hist` 死码，5.2µs/次固定成本未动 |
| `indexer_topk` | 0.10 | **0.03 + score 0.08 = 0.11** | ✅ 准（两步走后重新分配）|
| 残差（节点尾延迟）| ~1.15（700 节点×1.5µs）| 未单独成行（混在小核/间隙里，~0.9）| ⚠️ **口径不同**，需新采集用图节点数直接量 |
| **xn-megafuse 收益** | **−1.0** | 修正为 **−0.25~0.5** | ❌ **高估 2-4 倍**——"5 族一 launch"已被前提修正否决（`s.xn` 复用缓冲）|
| MoE cooperative | −0.3~0.5 | 未做 | — （本表验证其目标 1.72ms 真实存在）|
| 跨层流水 | −0.3~0.5 | 未做 | — |
| 零风险快赢合计 | −0.40~0.55 | 仅 P1+P2 = −0.27 落地 | ⚠️ 部分 |
| 融合路径落点 | 9.44（第 18 轮，含 bug）/ 预估 −0.7 | **9.65（第 22 轮）/ 9.73（远端复测）** | ⚠️ 偏乐观 0.2-0.3——实测融合门对门 **−0.51**（predict −0.4 与 −0.3 之和 = −0.7）|

### gap-analysis **漏掉**的两项（本次新发现）

1. **`gemv_bf16_fp8x2`（共享专家）1.32ms = 13.6%，全表第二**。gap-analysis 的 top 列表完全没提它——
   因为 rank0 串行时它在"per-rank 平均"口径下只有 0.275ms。**TP 切分后才暴露为每 rank 的真实成本**。
   ⇒ 这是 Stage B 排序必须纳入的新目标（此前被系统性低估）。
2. **`gemm_fp8_gemv` 调用数 171 → 206（+35）**。gap-analysis 的 "171 次 × 11.6µs" 算术已过时。
   来源待归因（可能：HEAD_SLICE 配套的切片 argmax 路径 / PROJ_FUSE / 共享专家切分后的额外 launch）。
   **在归因前，"减少 launch 数"类优化（xn-megafuse）的基线应按 206 而非 171 计。**

---

## 5. 复现与待实测

```bash
# 融合路径（gateup+down 全开），务必显式开——远端 tree 60a01a5 的默认门仍是 OFF
ssh ubuntu@43.202.208.136
cd /home/ubuntu/ferrite
export DSV41_GATEUP_FUSE=1 DSV41_DOWN_FUSE=1
bash scripts/dsv41_profile.sh 30 /tmp/dsv41-prof-v3
# 差分（脚本已内置；N 必须与实参一致——prof2 当时用 40、我按 30 算会整体 ×1.3）
python3 /tmp/kdiff.py /tmp/dsv41-prof-v3/one.csv /tmp/dsv41-prof-v3/many.csv 30
```

**待实测清单**（本次未做）：
1. **+35 次 `gemm_fp8_gemv` 的归因**（对照关闭 HEAD_SLICE/PROJ_FUSE/SHARED_TP 的 launch 计数）。
2. **残差/节点数**：用 `cuda_gpu_trace` 或图节点数直接量（当前只有 gap-analysis 的~1.15×700 估计）。
3. **生产口径的 absolute per-call**：多设备 nsys 不可信 ⇒ 对 gemm_fp8_gemv(206)、gemv_bf16_fp8x2(33µs)、hc_mixes_tail(12.4µs) 做隔离微基准。
4. 新采集应在**本地 HEAD**（`55dc747`：env 缓存 20 个热路径查询，去掉每步数百次 `env::var`）上做——
   远端 `60a01a5` 尚未含此 commit，**当前 9.73 可能已不是最新基线**。

---

## 6. 陷阱与注意事项（改这块代码前必读）

- ⚠️ **AR 行永远是伪影**：剖析必须 `AR_V5=0`，但其行值是 host-barrier，**不能直接进生产表**。换 v5 0.66ms。
- ⚠️ **"ms/步/rank" 隐藏关键路径**：任何"只有部分 rank 干活"的代码（如旧共享专家 rank0-only），
  在平均口径下看起来只有 1/8 成本。**优化收益必须用关键路径（max over ranks）判定。**
- ⚠️ **剖析的 N 必须与差分脚本的 N 一致**：否则整个表按比例失真（prof2 曾被我按 30 算 → 全部 ×1.3）。
- ⚠️ **default-value 分裂**：`.cu` 的 `g_fuse` 与 Rust 的 `unwrap_or` 必须逐字镜像，否则 kernel 融了但 host 没融 → 乱码（第 18-21 轮 6 轮排查的根因）。
- ⚠️ **批量 sed 翻默认值 = clobber**：`f3b1be1` 用 `unwrap_or(true)→false` 批量改，误关 7 个老门（MOE_BATCH 一项就 +560 launch/步 = +8.7ms）。
  子代理改共享文件后必须 `git diff` 检查**所有**改动。

---

_事实来源：`/tmp/dsv41-prof-v3`（one/many nsys，2026-09-11 16:1x）+ `/tmp/dsv41-prof2`（13.28 基线，逐项与 `STATUS.md:4932-4948` 吻合）；
`ab_fon` p50=9.73ms（`/tmp/ab_fon.log`）；`STATUS.md`（gap-analysis 5291-5316、第 22-23 轮 5527-5546）；`docs/agent/roadmap-200-tokps.md`、`dsv41-persistent-arch.md`、`dsv41-layer-fusion.md`。_
