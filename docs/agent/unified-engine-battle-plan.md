# Ferrite 统一引擎战役计划 — 全方面超越 SGLang

**制定日期**: 2026-09-12（用户指令：统一架构 → MTP 400 tok/s → KV/radix → 1M prefill → 多并发）
**基线**: DSV41 单并发 6.15ms/步 = 162.6 tok/s（tag `dsv41-6.15ms-162toks`）；GLM B=16 13.4ms
**对标**: sglang DSpark 383.7 tok/s @ B=1 B300 TP8（V4-Pro, accept~5, LMSYS 2026-07-06 博客）

## 0. 用户目标 → 工程指标

| 用户要求 | 工程指标 | 依赖 |
|---|---|---|
| 统一 GLM/DeepSeek，零重复 | 单一二进制 `ferrite-serve --model {glm53,dsv41}`；五层重复（服务/TP/图/kernel/加载）逐层收敛 | dup-audit 报告 |
| MTP 单并发峰值 ≥400 | dspark block-5：draft(1 图) + verify(n=6) + commit ≤ ~12ms，accept ≥4 | MTP 调研报告 |
| KV 管理 + 前缀命中 | radix tree（dispatch 已有 SGLang parity）+ hicache 三层接上 GPU 引擎 | kv 调研报告 |
| 1M 上下文 prefill | chunked prefill + indexer O(n²)→O(n) + KV 口径统一（8.9x 差） | prefill 调研报告 |
| 多并发不掉单流 | DSV41 batched（megab_b{size} 图池）+ ragged verify | 架构统一 + MTP |
| 超越 sglang | 以上全部 + 生产级稳定性 | — |

## 1. 五份侦察的核心事实（浓缩）

### 1.1 重复度（dup-audit，480 行报告 /tmp/dup_audit_report.md）
- 唯一共享层 = ferrite-http。五层各写一份：服务(60%) / TP-AR(70%) / 图捕获(75%) / kernel(30-35%) / 权重加载(65%)
- AR v5 kernel 已共享；devrt 是图原语超集（侧流/事件/节点优先级）；quant.rs 是量化基座
- `ferrite-exec/src/dist.rs` 是死文件（未被 lib.rs 声明）；ferrite-batch/scheduler 仅 CPU Engine 用
- **ferrite-dispatch 是活的调度基座**：radix.rs（SGLang parity 611 行）+ state.rs（三层 hicache）+ mtp.rs（MTP×batch 协议）+ batch.rs（SchedConfig）
- hc（hyper-connection）两套是同一数学（GLM mhc.rs ↔ DSV41 hc_*），重复最严重

### 1.2 DSpark/MTP（mtp-research，509 行报告 /tmp/mtp_research_report.md）
- **DSV41 checkpoint 自带完整 dspark 权重**（mtp.{0,1,2}，128-expert MoE topk3 + Markov rank256 + confidence），加载器已读，引擎从不执行
- sglang 机制：gamma=5（draft tokens），verify 宽度 = gamma+1 = 6 行，线性链（非树），num_steps 强制 1
- **verify 行数在延迟绑定下几乎免费**（GLM 经验：n=3 verify 16.6ms ≈ n=1 decode 17ms）→ block-6 verify 预计 ≈ 1.2-1.8× 单步
- 400 预算表：accept 4 → T≤10ms；accept 5 → T≤12.5ms。步时估算 8-13ms → **385-750 tok/s，可行性由实测 accept 决定**
- GLM MTP 的 N-unified 机制（mega_v N 行图 + graph_run_ids + ferrite_mtp_commit 单核）直接复用
- **旧"严禁mtp/投机"禁令已被用户 2026-09-12 指令明确解除**（本战役目标即 MTP 400）
- 最大 POC 风险：DSV41 sparse attn 的 n=6 多行支持（未验证）；small_n_rows 数值域对 accept 的影响

### 1.3 KV/radix（kv-research，报告待出）
- ferrite-dispatch 已有 radix（页对齐/lock_ref/LRU/三层）+ state（GDN snapshot + DSA page lease）
- GPU 路径（GpuEngine/TpRankPool）未接 dispatch —— prefix_hit 恒 0（死字段）
- GLM DSA KV 两套口径差 8.9×（latent 512 vs kernel 展开 41088）—— 1M 显存讨论的前提

### 1.4 Prefill/1M（prefill-research，427 行报告 /tmp/prefill_research_report.md）
- GLM：引擎 chunk-ready（prefill_chunk 存在）但 serve 整段传；MAX_CTX=8100 vs 模型声明 1M
- DSV41：无 prefill 分支（逐 token 走 decode 链）；indexer O(n²) 是最大瓶颈
- **多行链（m>1）是 dspark verify(n=6) 与 prefill(chunk) 的共享基础设施** ← 关键架构洞察
- CP 基础设施全有（axes.rs/shard.rs/dist.rs）但未接生产
- P0 清单：抬上限锁步 / KV 口径统一 / GLM chunked 接 serve / indexer 分块 / GDN 对齐 / 显存预算表 / DSV41 chunked

## 2. 战役分波（依赖序 + 价值序）

### Wave 1 — 快速架构统一（低风险高价值，先做）
1. 删 `ferrite-exec/src/dist.rs` 死文件（747 行，无引用）
2. GLM `run_serve` 改用 `ferrite_http::serve::launch`（删 80 行复制）
3. `Dsv41Frame` 上移 ferrite-models（与 GlmFrame 对称）
4. CLI 统一（合并两套 arg 解析）
5. 单一二进制 `ferrite-serve --model {glm53,dsv41}`（dsv41-run 变 alias）
6. 文档：更新 roadmap-200-tokps.md / perf-roadmap.md 的 MTP 禁令记录（用户 2026-09-12 解除）

**验收**: 两个模型各自 serve 文本不变（四段文本）；单一二进制跑通两个模型。

### Wave 2 — dspark MTP（最高价值目标，400 tok/s）
按 mtp-research 的 M0-M4：
- M0: 远端核对 checkpoint mtp.* key（ssh 读 index）
- M1: device dspark draft 前向（3 block + Markov + gumbel；对照 host 参考对齐 argmax）
- M2: verify 图 n=6 + **DSV41 多行链 POC**（sparse attn n=6、compressor n=6、GEMV/GEMM 数值域）
- M3: accept（贪心最长前缀）+ commit（复用 ferrite_mtp_commit, mtp_n=6）+ 文本校验
- M4: 性能调优 ≥400 tok/s（accept 实测 → 步时分解 → 优化）

**关键设计**: spec 框架写成模型无关（DraftHead trait：propose(graph) / VerifyEngine: verify(n rows) / Commit），DSV41 dspark 是第一个实现。这同时服务后续 GLM MTP batched 和多并发 ragged verify。

### Wave 3 — KV/radix 接线（前缀命中）
1. GpuEngine 接 ferrite-dispatch 调度（SchedConfig 页预算 + Admission.prefix_hit 活化）
2. radix + hicache 三层接 DSV41 ring/index_k（快照 = state 恢复）
3. DSA indexer topk 表 + GDN state 的前缀恢复路径

### Wave 4 — 1M prefill
P0 清单执行（见 prefill-research §D）：多行链（与 Wave 2 共享）→ indexer 分块 → 口径统一 → 显存预算表

### Wave 5 — 多并发 batched
1. DSV41 batched 图池（megab_b{1,2,4,8,16,32} 键控 + STABLE 设备地址表）——复用 GLM 已验证机制
2. MTP × batched（ragged verify 总量键控分层，sglang 方案）
3. pad 策略（dummy state 共享 + membership 变化刷新）

### 深度统一（与 Wave 3-5 并行推进，有 400 基线保护后做）
- 设备层收敛：cuda.rs → devrt（GLM 获得侧流/节点优先级）
- kernel 库合并：dsv41_glue/route 并入 ferrite_kernels.cu；PDL 公共头
- hc 数学合并（GLM mhc ↔ DSV41 hc_*，需数值 parity）
- 描述化：LayerDesc schema，两个 chain*.rs 层循环变数据
- TP host 协议收敛：TpCluster → Collective

## 3. 关键技术决策（已定）

1. **多行链先行**: DSV41 的 layer 函数支持 rows>1 是 Wave 2/4/5 的共同基础。verify(6) 和 prefill chunk 走同一套。
2. **spec 框架模型无关**: DraftHead/Verify/Commit trait 在引擎层，dspark 是实现。
3. **图策略**: draft 1 张图（block 一次出 5 token，无 h 链 relay）+ verify 1 张图（n=6, graph_run_ids 喂 24B ids）+ commit 单核。DRY→rollback→CAPTURE 纪律 + 图间 sync 护栏（GLM 教训）。
4. **贪心 accept 先行**: 采样拒绝路径（sglang AcceptSampling）后补。
5. **数值域铁律**: draft 与 verify 同数值域（small_n_rows 的 n=6 取舍必须实测 accept 后定）。
6. **不 git revert**: 旧禁令文档更新为"已解除"，不删历史记录。

## 4. 风险清单

| 风险 | 影响 | 缓解 |
|---|---|---|
| ~~DSV41 sparse attn 不支持 n=6~~ | ~~verify 无法一次 6 行~~ | **✅ 已排除（2026-09-12 侦察）**：kernel 层全支持多行——`dsv41_sparse_attn` 的 `dim3 grid(b*m, h)`、`hc_mixes(rows)`、`moe_route(rows)`、`compressor(b, seqlen)`、`gemm_fp8_mx(m,n,k)`、`embed_expand_dev(n)`。改造集中在 chain 层（step_body 单 token 假设） |
| accept < 4 | 400 不可达 | M1 后立即离线测 accept 分布（host 参考在真实文本上）；confidence 头给出预期 |
| 多行链重构破坏 6.15ms 基线 | 回归 | env gate（DSV41_SPEC=1 才走新路径）；单行路径逐位不变 |
| 架构统一破坏两模型 | 回归 | 每步单一文本验收；tag 保护（dsv41-6.15ms-162toks） |
| 图捕获池 miss（TP=4 教训） | 崩溃 | DRY 预热纪律 + FERRITE_POOL_MISS 诊断 |

## 5. Wave 2 详细设计（2026-09-12 补充，M0 核对后）

### 5.1 关键架构事实（侦察定案）

1. **M0 ✅**：远端 checkpoint `mtp.*` 2401 key 全在（mtp.{0,1,2} 各 attn 13 + ffn 777 + hc 6 + norm，mtp.0 有 main_proj/main_norm，mtp.2 有 markov_head.{embed,head} + confidence_head + norm）；config：block_size=5、target_layers=[37,38,39]、markov_rank=256、noise_token_id=128799、128 experts topk3。
2. **kernel 层多行全支持**（见风险表第 1 行）——verify(n=6) 无 kernel 障碍。
3. **dspark draft block 与主链 block 算子同构**（权重命名对称：attn.{wq_a,q_norm,wq_b,wkv,kv_norm,wo_a,wo_b,attn_sink} / ffn.{gate,experts,shared_experts} / hc_* / *_norm）→ **draft device 化 = 复用 Device 现有 kernel 方法**，仅 Markov 循环头 + confidence 头需新 kernel。
4. **DSV41 状态回退比 GLM GDN 快照更轻**：compressor state（state_kv/state_score，固定 ratio*hd floats）快照便宜；ring/index_k 写入位置是 device counter → 回退 = counter 减法 + 覆写。
5. **sglang 语义**（复刻基准）：gamma=5 draft tokens，verify = gamma+1 = 6 行（anchor + 5），线性链因果（行 i 看 t0..d_{i-1}），贪心 accept 最长前缀，num_steps 强制 1。

### 5.2 三大实施件

**① 多行链 `step_body_rows`（我亲自做，最高风险）**
- chain_dev.rs 新增：`step_body_rows(&mut self, toks: &[u32], pos: usize) -> Result<Vec<u32>>`
- embed_expand_dev(n=len) → 每层 hc_mixes(rows=n) / attention（sparse_attn b=1,m=n，idxs 含块内因果）/ compressor(b=1, seqlen=n) / MoE(rows=n, moe_batch) → head n 行 argmax
- **与 prefill chunk 共享**（Wave 4 直接受益）；单 token 路径 `step_body` 保持不动（env gate DSV41_SPEC 控制 verify 走新路径）

**② dspark draft device 前向（subagent 写草案，我审）**
- `chain_dev.rs` 新增 `dspark_draft(&mut self, t0: u32, pos: usize) -> Result<([u32;5], [f32;5])>`：forward_embed（main_proj 投影 + noise embed 混排）→ 3× dspark block（窗口 attention 用 sparse_attn + dspark_topk_idxs 索引矩阵；MoE 复用）→ Markov 循环头（新 kernel：顺序 5 步，读已采样 token 的 markov_embed bias logits，gumbel/argmax 采样）→ confidence
- 关键缓冲：main stream 层 37/38/39 的 attention 输入导出（chain 里在 layer() 加 hook 缓存 h_mean），draft 窗口 ring（window_size=128）
- 新 CUDA kernel 仅 2 个：`dsv41_dspark_markov_head`（Markov 循环，单 block 顺序 5 步）+ confidence 可并入

**③ spec 编排 + accept + commit（我做）**
- `dspark_spec_step`：draft(1 次) → step_body_rows([t0,d1..d5], n=6) → device argmax[6] → accept 最长前缀 k → commit（compressor state 快照恢复 + ring/index_k counter 回退 6-k）→ 下一 token = argmax[k]
- 图策略：draft 图 + verify 图分开捕获（DRY→rollback→CAPTURE 纪律）；accept 前一次 D2H 24B（argmax 数组）
- env gate：`DSV41_DSPARK=1`

### 5.3 里程碑（细化）

| 阶段 | 内容 | 验收 |
|---|---|---|
| M1a | step_body_rows 多行链 + 单测（n=6 vs 6 次单 token，argmax 一致） | cargo test + 远端文本 |
| M1b | dspark draft device 前向 + 与 host 参考对齐（argmax 级） | 新 tests_dsv41_dspark.cu |
| M2 | spec_step 端到端（贪心 accept + 回退）+ 图捕获 | 四段文本逐字 + accept 打印 |
| M3 | 性能：accept 实测 → 步时分解 → 调优 | ≥400 tok/s |

### 5.4 spec 编排设计（2026-09-12 定案）

**`dspark_spec_step`（DevChain 方法，chain_dev.rs）**：
```
输入：t0（上一步 argmax 输出，pos_ctr = pos_base）
1. draft：DsparkDev::draft_forward(t0, pos_base) → draft_ids[5]（device，Markov 循环）
2. verify：step_rows([t0, d1..d5]) → argmax_r[6]（verify 的 argmax 循环传 null pos_ctr 不递增）
3. D2H：24B（argmax_r + draft_ids）
4. accept（host）：k = 1 + 最长前缀（d_i == argmax[i-1]，i=1..5）；输出 tokens = argmax[0..k-1]（k 个新 token）
5. commit（device）：
   - k == 6：全部接受（快路径，仅 pos_ctr += 6——下一步 argmax 自然递增）
   - k < 6：恢复状态——compressor state（state_kv/state_score 快照 D2D 恢复）+ clen 回退 + ring 窗口无需回退（覆盖写）+ pos_ctr 设 pos_base+k（H2D 4B）+ 重放前 k 行 compressor（用 x_r 前 k 行，从快照状态续算 clen/index_k 一致性）
6. 返回 k 个 token；下一步 t0 = argmax[k-1]
```

**状态快照**（verify 前，每 kv-source 层）：state_kv/state_score D2D 拷到快照缓冲（~10MB 总量）+ clen host 侧记录。第一版用逐层 memcpy（~40 次 launch），M2 融合为单 kernel。

**GPU 验证进度（2026-09-12）**：
- dsv41_dspark_markov_head：5/5 case 过（含 full 129280×256，与 CPU 参考逐位一致）
- dsv41_verify_ring_win：7/7 case 过（因果窗口索引 + 块 append，含窗口边界）
- 主 .so 远端编译通过（build_id fe3243ca）


## Wave 2 状态快照（2026-09-12 06:20 UTC，第 8 修后）

**已落地 8 个修复**（全部同会话验证）：
| # | 修复 | 效果 |
|---|---|---|
| 1-2 | RC0 错误显形 + RC1 snapshot 时机 | rank 错误不再吞；rollback 恢复正确基线 |
| 3 | mtp 权重 shard 越界（attn Replicated + MoE TP 几何 + AR） | cuda-700 消除 |
| 4 | all_kv 紧凑布局 + rope 位置 pos+r | drafts 从随机样 → 有信号 |
| 5 | verify 因果窗口逐行化 + compressor 逐行化 | verify 链 parity 达成 |
| 6 | tap 口径（层输出非输入） | verify[0]==next 频率 1/12→5/12 |
| 7 | draft head collapse 对齐 attn_pre + 撤销 mk 恢复官方块语义 | — |
| 8 | **时序对齐（anchor=bonus 官方结构）** | **结构里程碑：verify anchor 行正确预测下一 token**（verify[0]=1767==pos12 next 等）；draft 块 [next,noise×4]@pos+1、verify 6 行、accept drafts[j] vs verify_out[j] |

**性能实测**（影子模式，出师表 150 步）：主链 6.15ms + draft 7.4ms + verify 45.3ms（6 行朴素版）+ snapshot/rollback ~2.7ms ≈ 55ms/步。

**剩余瓶颈**：draft 匹配率 ~6%（0.3 匹配/步 vs 正常 60-80%）——draft 数值还有系统性偏差（结构已全部对齐官方：时序/窗口/块语义/rope/ctx/层序 ✓ 全部核对）。模式分析：无错位（shift1≈0）、d_in_v≈m（对/错二值分布）、部分位置匹配。

**进行中**（3 subagent）：
- audit-draft-residual：draft 剩余数值风险的系统性清单（noise id/premix/q_norm/MoE 路由等 10 项对照）
- commit-impl：真 commit（DSV41_SPEC=1，方案 A 快照+重放）
- verify-perf-design：verify 39→5ms 的优化路线（多行 GEMV + 图化）

**下一步**：audit-draft-residual 的清单 → 修 draft 的最后数值 bug → accept 跃升验证 → 真 commit → 性能优化至 ≥400。
