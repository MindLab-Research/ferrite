# SGLang 的 verify 模型 vs ferrite 的 4× —— 对比判决（2026-09-13 03:00）

> 来源：sglang-dspark-compare（web+源码双线）+ 主 agent 源码追踪（eagle_worker_common.py:461 → tp_worker.py:593 → model_runner.py:1819）+ eager-verify-diff（ferrite 侧差分）+ vg-shape + ar-round-census + moe-sweep-anatomy。
> 本文档是 400 攻刺 step 侧的**新主战场路线图**（取代已判死的 MPAR/mma/⑤a 摊薄三路）。

## 1. SGLang 的 verify 是一次 forward，m 个 token 是 batch 维

| 组件 | SGLang（代码证据） | ferrite 现状 |
|---|---|---|
| 整体 | `bs×m` 拍平一个张量，`TARGET_VERIFY`（extend 形态），单次 `model.forward`（eagle_worker_v2.py:1670） | 6 行块但 kernel 链按行组织 |
| GEMM | M 维=bs×m，权重读 1 次（cuBLAS/torch 标准） | `gemm_fp8_mrows`：M 进寄存器串行（warp 数=n 与 M 无关） |
| MoE | `moe_align_block_size`→token 按 expert 排序分块→grouped kernel，每 expert 权重块服务名下所有行（fused_moe_triton_kernels.py:150-215）——**纯调度，不需要 tcgen05** | 逐 (row,slot) 36 sweep |
| attention | unified decode+extend：一次 append m 行 KV + m 行 query 单 kernel（tree mask 穿进） | 逐行 sparse_attn（3.2× 实例比） |
| graph | verify 复用 decode graph 桶（padding） | 按形状分槽 |
| vLLM | 同形态（cu_num_scheduled_tokens 拍平 + logits_indices 挑采样位，gpu_model_runner.py:2832-2870） | — |

## 2. 实测比：1.2-1.3×（不是 1.0×，更不是 4×）

- **EAGLE 论文 Table 8 反算**（step/eager = τ/speedup）：Vicuna 13B 1.23× / LLaMA2-70B 1.20× / 平均 **1.2-1.3×**。
- DeepSeek-V3 官方 MTP：1.8× TPS（若 verify=4× eager 则 1.8× 数学上不可能）。
- 内部 B300 SGLang 生产：64 并发 **TPOT 8.4ms @ acc 2.2-3.75**（用户的 8ms 口径 = TPOT）；fused_moe_kernel 占 verify 31.5% 且是 grouped。
- Medusa 论文的机理前提 = 用户的模型原文："predominantly memory-bandwidth-bound… each forward pass transfers the complete model parameters" ⇒ 多带 m 行激活 ≈ 免费。

## 3. ferrite 4× 的完整分解（无单一元凶，系统性零摊销）

1. **verify 用 eager 的"未融合 pair 形式"**：q norm+rope+wq_b 4 发 vs eager 1 发（lin_rope_norm）；sparse_attn 逐行 vs eager orope 1 发；collapse+norm 2 发 vs 0 发（折叠在 hc 前脸）。
2. **M-in-register 串行**：mrows warp 数=n 与 M 无关。
3. **MoE 36 sweep**：无 expert 去重。
4. **per-row 逐行发**：rmsnorm 25×/rope 40×/quant 9.3×。
5. AR 不是来源（verify=eager=81 轮，m 无关，payload ×6）✓。
6. host 侧多余 ~0.5-2ms（pos_ctr D2H + 2 H2D + 投票）——不解释 4×。

## 4. 修复路线（按 ROI）

| # | 改动 | 可移植性 | 预期 |
|---|---|---|---|
| G1 | **MoE 纯调度 grouped**（排序 per-expert 段——解耦 tcgen05；ferrite 的 GROUPED 骨架已存在只是绑在 tcgen05 门链） | ⭐⭐⭐⭐⭐ | 权重读 36→~16 hit experts；−1.5-2ms |
| G3 | **per-row launch batched**（row 进 grid 维） | ⭐⭐⭐⭐⭐ | launch/实例比归零 |
| G2 | **M 进 GEMM tile**（warp→(m_block,n_block)，BLOCK_M≥6） | ⭐⭐⭐⭐ | 4×→1.3× 的机制来源 |
| 融合对齐 | verify 吃 eager 融合形态的 rows 版（orope/lin_rope_norm/wo_pair） | ⭐⭐⭐⭐ | 4 发→1 发/层 |

**反向提醒**：ferrite kernel 跑在 HBM 0.7-4.9%（issue-bound）——收益来自**并行度∝tokens + 权重流次数**，不是"共享权重"本身。

## 5. Parity v2 判定（a4 修后首跑）

- 10/12 臂 OK（CTL/a1-a4(inv 双)/l1/l2/l3/MOE-MROWS 逐位 ✓）
- **l4/K2 FAIL（6400/6400 全差）**——P3LITE 段 B 的"逐位"声称是假的（parity 抓到；解释 p3b2 的 acc −0.22）
- DSPARK_DRAFTS 是编译期常量（DSV41_DSPARK_DRAFTS env 不存在）——DRAFTS=3/2 臂无效（m 没变）
