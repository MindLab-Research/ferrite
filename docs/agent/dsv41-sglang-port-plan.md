# DSV4.1 DSpark 移植总案：SGLang 算子 → ferrite（2026-09-13 定稿）

> 用户指令链：① 参考官方 PyTorch 彻底改好正确性；② MTP block-5 step ≈ 7ms 即达标（同口径 > sglang 873.6 tok/s）；③ eager 之外的算子判垃圾，**照抄 SGLang**（BBuf/sglang@835c3909，V4.1-Flash 真源；master 无 V4.1）；④ 移植→确认正确→优化到比他们快。
> 验证机：AWS b300-4（ubuntu@43.202.208.136，8×B300，模型 /opt/dlami/nvme/models/DeepSeek-V4.1-Flash）。

## 🎯 当前战况（2026-09-14 深夜四——kv_gemm 已逐位对齐（T1 发射 bf16 修复）；剩余=attn_o 3.3-3.5% 的其它源；读这节就够）

**本轮三大突破（全部默认配置验证）**：
1. **kv_gemm = 0.00%（逐位一致！）**——根因=**T1 融合发射**（`hc_mixes_tail_kernel` 的 EARLY 半段内联产 xn 的 fp8）**跳过了 bf16 化**：写回 xn 无 bf16 舍入 + amax/量化用未舍入 f32，而官方 xn 是 bf16（hc_pre 末尾 `.to(x.dtype)`）。修复=内核加 `bf16_xn` 参数（`g_tail_bf16_xn` static 镜像 DSV41_BF16_XN）+ 发射循环写回前舍入 + 3 调用点传参（commit 5d0d4c2f）。**方法论黄金链**（复用于后续猎杀）：xn 逐位同（kind-0）→ 独立 quant 逐字节同官方 act_quant（真实 xn 微测 0/5120+0/160）→ GEMM 内核恒等（SIMT gemv≡swapab≡mx2≡官方 tilelang，模输出 bf16 cast 0.15%）→ FUSE_B1=0 隔离实锤（kv_gemm 0.495%→0.000%）。**教训：所有"融合发射=独立调用"的声明必须逐字节验证**（本声明失败造成 0.495% 隐蔽偏差，且 NORM_FUSE/PROJ_FUSE 门都不控制它）。
2. **专家域三阶段 + sparse_attn 内部 bf16**（前两轮，见下表）。
3. **官方栈澄清**：官方=PyTorch 模型代码（model.py）+ **tilelang JIT 算子**（kernel.py 的 @tilelang.jit）——参考值来自这套栈（dump 生成源码确认 `tl::mma_sync<m16n8k32,e4m3>` 与我们同指令；"GEMM 累加序不同"的旧理论**已死**）。

**当前逐算子发散（默认配置，teacher-forced）**：xn 0.00% ✓ / **kv_gemm 0.00% ✓** / kv_RT 0.55-1.19% / attn_o 3.29-3.47% / moe_o 3.29-10.43%（上游泄漏+shared 内部）。数数 1..51 后断于 52（75 行）。

**attn_o 剩余源（TODO #4，按优先级）**：①kv_RT 残余 0.55-1.19%（norm/rope/RT 链内部——**警惕同款 T1 式融合跳边界**：kv 链有 gemm_fp8_mx_rope_norm 融合！）②q 链（wq_b/rope/q_norm）③indexer 选择（topk_idxs vs 官方）④o-proj（wo_a/wo_b）。方法论同上：零重建微测+官方对拍+env 门隔离。

## 🎯 T1 模式连环修复（2026-09-14 深夜六——四例已修验证；o 链 4 倍降！indexer D1 构建中）

**T1 模式定义**：融合内核内联产激活/量化但跳过官方有的 bf16 边界（官方每个算子输出都是 `.to(dtype)`=bf16）。四例全部修复：
1. **T1 原版**（hc_mixes_tail_kernel 的 xn fp8 发射）✓ kv_gemm 0.00%
2. **第二例**（rmsnorm_rope_kernel 的 kv norm+rope 融合）✓ kv_RT 1.19→0.08%
3. **第三例 Q-1**（gemm_fp8_gemv_kernel 的 NORM_FUSE prologue）✓
4. **第四例**（p2p_ar_pubred_v5_hcpost_kernel 的 AR 输出+hc_post 输出）✓

**o 链三修复（重大突破，37e197f0 验证）**：wo_a f32 切换（官方纯 bf16×bf16 einsum 无量化——fp8×2^e 权重解码在 f32 精确=官方 bf16 权重，s.o 已 bf16 值域，gemm_fp8_mx_f32 零成本复用）+ WOB_F32 默认 OFF（官方 wo_b 有 act_quant，我们跳过=精度过高）+ AR+hc_post 内核舍入（残差流 bf16 域）。**attn_o 3.17-3.41%→0.76-0.94%（4 倍降）！moe_o 10.4→1.89%（专家翻转消失）！**

**indexer 战况**：D2+D3（分数链 bf16+scale 折叠，30978c7c）已验证——**attn_o 无变化**（分数链舍入贡献小）。**D1（fp4 e2m1 往返）已验证通过**：数数 1..51→**1..63（历史最佳）**！op 级持平但序列级大幅改善（topk 选择对齐）。

## 🎯 EOS 提前根因定谳（2026-09-15 凌晨二——链路闭环，修复中）

**完整根因链（全部实测证据自洽）**：
1. **层间系统偏差**（attn_o 0.76-0.94%，主要源=kv_RT 0.08-0.57% 的 softmax 放大泄漏；moe_o 1.76-1.89% 几乎全是 attn_o 上游泄漏）
2. **× 40 层 hc 残差流放大**：层 0 输入 h 0.000% → **层 39 输入 h 74-98%**（kind-40/41 hc0 块对比有效——s.h 布局 [row][hc][dim] 由 hc_collapse 内核定谳）
3. **→ head 输入差 74-98% ⇒ logits 偏差 66-117%**（kind-42 实测：连 step 0 都 75.6%，但 argmax 大多匹配——两个"不同但合理"的 hidden 的 top-1 常一致）
4. **→ EOS 系统性膨胀 +8~11**（kind-42 实测 our[EOS] vs ref[EOS]）⇒ 数数 token 的 ~19-27 margin 被侵蚀 ⇒ **边界步翻转 = 提前 EOS**（崩溃点随序列变化 ✓ 与"1-100"断 63/51/34、"50-150"断 76 全部吻合）

**已修（构建中）**：rope cos/sin 表改 **host 双精度预计算**（消 --use_fast_math 的 cosf/sinf/powf/logf ~2⁻²¹ 近似；官方 torch.polar 是 libm 精度；启动时一次同步拷贝零成本）。

**进行中**：②kv_RT pos100 离群 0.57% 源头 ③q 链/o-proj 残余。目标 attn_o <0.3% ⇒ 40 层放大后 logits 偏差 <10% ⇒ EOS 不再翻转。

**⚠️ 运行间方差（竞态嫌疑）**：数数断点 63→51→34 跨轮变化（代码几乎相同）——dump 的 D2H 同步扰动或真竞态（kv_stream 侧流嫌疑）。待层间偏差压到 <0.3% 后复查。

**⚠️ engram 实证**：不带 SKIP_ENGRAM_WEIGHTS 数数退回 1..51（更差）——SKIP=1 保持（write-back 有害，用户裁决 engram 非正确性因子）。

**T1 模式定义**：融合内核内联产激活/量化但跳过官方有的 bf16 边界（官方每个算子输出都是 `.to(dtype)`=bf16）。三例：
1. **T1 原版**（hc_mixes_tail_kernel 的 xn fp8 发射）——已修 ✓ kv_gemm 0.00%
2. **第二例**（rmsnorm_rope_kernel 的 kv norm+rope 融合：norm 输出无 bf16 舍入、rope 读未舍入值）——已修 ✓（bf16_norm 参数穿线 + 独立路径中间舍入 bf16_round_inplace_on）待验证
3. **第三例 Q-1**（gemm_fp8_gemv_kernel 的 NORM_FUSE prologue：q_norm 输出 f32 直出 fp8）——已修 ✓（prologue 内加舍入）待验证

**YaRN rope 边界修复**（floor/ceil 对齐官方 model.py:382-383）——已验证无效果（kv_RT 离群 3/5/10 个仍在）但无回归，保留。

## 三份审计报告的发现（subagent 只读调研，2026-09-14）

**qchain-audit（q 链）**：官方 q 链 4 个 bf16 边界（wq_a 出/q_norm 出/wq_b 出/rope 出）；我们有 B1(qr)/B4(q post-rope) 两处，**Q-1（q_norm 出，NORM_FUSE prologue）已修**；wq_b 输出（pre-rope q）的 bf16 边界仍缺（低优先——rope 后有 B4 舍入）。

**indexer-audit（选择链）**：🔴D1 官方对 indexer 的 k 和 q 都做 `fp4_act_quant(x,32,inplace)`（fp4 e2m1 往返写回 dequant 值）——我们全 f32 无量化！🔴D2 官方 index_score 全程 bf16（einsum/逐头乘/sum 都 bf16）——我们 f32。🟡D3 scale 折叠位置（官方折进 per-head weights）。🟡D4 candidate blocks 未启用（仅 seqlen>16384）。**D1+D2 是大工程（内核改造），修好后 topk_idxs 应与官方一致。**

**oproj-audit（输出链）**：官方输出链全程 bf16 残差流。🔴wo_a 是纯 bf16×bf16 einsum（无量化！）——**我们把激活额外压 fp8（quant1→gemm_fp8_mx_q）= 精度过低**；wo_b 后 AR f32→bf16（我们无边界）；hc_post f32 混合→bf16（我们无边界）。**wo_a 修复=bf16 gemv 路径（fp8 权重精确反量化到 bf16 + f32 累加 + 输出 bf16 舍入）——gemv_bf16_kernel（glue.cu）可参考。**

**数数**：1..51 后断于 52（但继续输出至 98 行——"在数"而非退化）。正确性锚=官方 PyTorch（MP8 teacher-forced 对拍，kinds 0-21 dump 工具链）。

**已落库的域对齐（全部默认 ON）**：
| 域 | 状态 |
|---|---|
| `bf16_xn`/`bf16_kv`/`bf16_q`/`win_kv_quant_rt`/`gate_gemv_f32` | xn 逐位 0.00% ✓，kv_gemm 0.49%，kv_RT 1.1-1.5% |
| **专家域三阶段**（e4m3 激活 + down 链对齐 + ILV 修复） | moe_o 4.7-10.9%→**6.7-14.3%（受 attn 泄漏主导）**；40× 爆炸根因=ILV 交错池+强制 sequential+未清零 s.o（已修） |
| **sparse_attn 内部 bf16**（P·V 概率 acc_s_cast + 输出/rope bf16，DSV41_ATTN_PBF16） | attn_o 3.5-3.84%→**3.46-3.66%（微降）**——P·V 概率非主源 |

**剩余发散的因果链（定谳）**：`kv_gemm 0.49%（fp8 gemm 累加序≠官方 tilelang 结构）→ kv_RT 1.1-1.5% → attn_o ~3.5%（注意力放大 kv 误差）→ moe_o 7-14%（FFN 侧泄漏）`。**根子 = fp8 GEMM 的分块/累加序**——官方用 tilelang 的 fp8_gemm_kernel（确定性但 tile 特定），我们的 gemm 累加序不同 ⇒ ~0.5%/层，经链路放大。

**下一步（按序）**：
1. **kv_gemm 逐位对齐**：对照官方 tilelang fp8_gemm_kernel 的分块/累加结构（kernel.py:280-305），重排我们的 gemm 累加序（或融合边界）——目标 kv_gemm 0.49%→~0.1%。**这是正确性的最后一座大山**（其后 kv/attn/moe 全链坍缩）。
2. 数数 1..100 → spec_gt 复测 → S2-S7 性能线（450 tok/s / bench >908.9）。
3. 性能线注意：e4m3+sequential（ILV 排除）是正确性配置；性能需给 batched 内核加 e4m3 臂（ILV 恢复）。

**关键工具链（远端）**：`~/ref_diff.py`（teacher-forced，kinds 0-6 hooks）+ `DSV41_GT_XDUMP`（kinds 0-21，注意分析脚本需带全 kind 尺寸——`~/`下历次对拍脚本可复用）；官方 MP8 `/opt/dlami/nvme/dsv41_mp8`；微测试 `/tmp/tests_gemv2.cu`、`/tmp/tests_down.cu`（链接 .so 逐位验证内核）。

## ⚠️ 战况修订（2026-09-13 深夜，实测推翻假设）

1. **tag 的 eager 文本也坏**：port-dspark（=tag）数数 1..51 后跳 60（与 HEAD 同型同位；p50=6.22ms 与文档 6.15 吻合）。tag 时代"好"的判据是四段短文本（max_tokens=48），长程腐蚀从未测过。⇒ **正确性锚不是 tag，是官方 PyTorch 语义**（用户原话的字面执行）。
2. **E1**（HEAD + MOE_BATCH=0 + 8 parity 门 + bf16 边界）：仍在 52 行坏（token 变 62）⇒ parity 门不是（唯一）病因；**MOE_BATCH=0 可恢复前 61 行**（但 p50 6.2→16.3ms）⇒ batched MoE 路径确实另有腐蚀。
3. **one-shot 判别**：绕过整个 serve 层，腐蚀一模一样 ⇒ **病灶在模型/内核层**。
4. **W2 三角定位**（改 prompt 长度）：首坏 token 的序列位置 117/95/101（prompt 15/27/13），形态还不同（跳数 vs 跑偏+EOS）⇒ **累积漂移**，非硬边界（window=128 回卷公式手工验证是正确的）。
5. **漂移猎杀工具链（已就绪）**：官方 MP8 分片在 `/opt/dlami/nvme/dsv41_mp8`（WORLD_SIZE=8 NCCL 可跑官方 model.py）；`~/ref_count.py`（官方贪心数数=能力金标）、`~/ref_diff.py`（teacher-forced 逐步 logits dump）、`~/compare_logits.py`（按"预测位置"对齐，报首个 argmax 分歧 + 幅度形态）、我们侧 `DSV41_GT_LOGITS_DUMP` 探针（rank0 逐步全量 logits append，诊断专用）。
6. **S3a（真值 spec）已实现**（`spec_step_gt`：verify=6 串行 golden 步 + 每行快照 premix/state_kv/state_score/clen/compress_len + commit 恢复 snap[k+2]；draft 模式 noise/self/cycle）——金标准 = **spec 流 ≡ eager 流**。

## ⚠️ 战况修订二（2026-09-13 深夜二，S0 + spec_gt 实测）

1. **S0 同口径参考落袋：908.9 tok/s 中位**（sglang TP8、模拟 acc 5.49、random 4k/1k、cuda graph ✓、`--disable-flashinfer-autotune`——首启 autotune 在 TP8 死锁 17 分钟 GPU 全闲，禁用后 6 分钟起服务）。**这是同条件要击败的数字**。
2. **sglang 数数金标 = 完美 1..100**（101 行、first-61 干净、尾部 86..100；9 个"重复形态"恰为 11/22/.../99 正常值）⇒ 模型会数数，**我们的 52 行腐蚀确凿是我们的 bug**（与官方数值的发散）。
3. **sglang 自己的背诵输出也乱序**（静夜思"疑是地上疑霜"、出师表串词）= 重度量化 checkpoint 的背诵天花板，与数数红线无关。
4. **spec_gt 金标准 = 部分通过**：忠实复现 eager 流到第 58 行（含其腐蚀形态，字节级），第 59 行起发散——发生在模型漂移开始（52 行）之后的 marginal logits 区。**待漂移修复后复测才是干净判据**（漂移让 top-2 贴近时，任何合法微差都会放大）。p50=65.7ms/spec 步（cycle 模式 ~11 次 forward）与 eager ~6ms/步（无图）自洽。
5. **ref 工具链的 encode bug 已修**：transformers 5.17 的 `encode(text, False)` 第二位置参是 `text_pair` 而非 `add_special_tokens` → 改关键字传参。hunt.sh 步骤 1/1b 因旧码崩（模型加载后崩于 encode），需在队列后重跑；步骤 3+ 用修好的 ref_diff.py。
6. **parity 内核源码已提取备好**（HEAD 的 `win_kv_quant_rt_kernel`（含 G3 bf16 写回修）与 `glue_latent_fp4_block16`（e4m3 标度 + e2m1 码 + bf16 写回））——NO-FP4 判别一旦证实量化域假设即开干移植。

## ⚠️ 战况修订三（2026-09-14 凌晨，op 级下钻定谳）

1. **h-dump 步号差一的重大修正**：我们的 `(s,L)` 记录=位置 s（step_count 在 step_body 内未自增），ref 侧=位置 s-1 ⇒ 此前所有"逐层发散"在比不同位置。正确对齐后：embeddings 完全一致 ✓。
2. **op 级三路对拍（layer 0，位置 20/50/100）**：xn（前端 collapse 输出）rmsdiff **0.18%**（corr 0.999998，RMSNorm 输出 rms 恒定）——**前端混合完全正确**；attn_o rmsdiff **3.85-4.51%**；moe_o rmsdiff **5.38-10.84%**。两个子层都是域级中度发散，叠加 80 次/步 → logits ~2-5% → marginal 处翻 argmax（117）。
3. **NO-FP8KV 判别**：官方去掉窗口 KV fp8 量化仍干净数到 99 ⇒ fp8-KV 域差非腐蚀驱动（官方对此域鲁棒），但 attn_o 的 ~4% 正是该域差 —— **win_kv_quant_rt 已移植+接线（默认 ON）**，A/B 待 routehunt 验证坍缩。
4. **路由嫌疑**：`dsv41_route.cu` 文件头自曝"生产形状 384/6 下 12/12 assignments 与 CPU 参考不同（非近 tie 伪影）"——但该 STATUS 段可能早于 -INF 消费标记修复，真伪待 kinds 3/4（专家集+权重）对拍。官方 Gate 在 **f32** 算分数（x.float()@weight.float()/gate_temp），我们走 bf16 gate GEMM——精度域差可能翻转近 tie 的专家选择。
5. **官方 MoE 语义已读全**：scores+bias 选专家、原始 scores 取权重、/(sum+1e-20) 归一、×route_scale、权重作参数传 Expert（乘在 w2 **之前**）、shared 在 AR 后加、输出转 bf16。score_func=sqrtsoftplus（两边一致）。

## 0. 基座决策（已定）

- **基座 = tag `dsv41-6.15ms-162toks`（8a5a952，09-12 02:16）**：纯 eager 基座，chain_dev.rs 仅 4668 行，无任何 dspark/mrows 机器；eager 正确（用户锚点）、step 6.15ms。
- tag 之后 1206 提交（+2 万行 spec 机器）= 判垃圾的"两套实现"病：`step_body/layer/moe`（m=1）与 `step_rows_inner/layer_rows/moe_rows`（m=6）并行漂移。HEAD 实测 eager 已坏（1..51 后跳数；单门 triage 7 臂无一恢复 ⇒ 多门/无门重写污染）。
- **架构北极星（SGLang 的根本优势，E 报告）**：verify 走**同一个 forward 路径**（TARGET_VERIFY = 同一 backend 的 extend 元数据），只有 qo_indptr/paged_kernel_lens 变，kernel 不分叉。⇒ 移植核心 = **一套 m=1..6 统一执行核**，eager 与 verify 共用。
- tag 自述"kernels are all batched-ready; 单 token 约束在 host 编排"（tag chain_dev.rs:27）⇒ m 行核 = 新写 host 编排 + 修真正不 m-row-safe 的 kernel，而非重写全部。

## 1. 语义契约（13 条不变量，E 报告精缩；违反任何一条 = 幽灵 bug）

| # | 不变式 | sglang 锚点 | 备注 |
|---|---|---|---|
| I1 | verify row r ∈ [anchor, d1..d5]，position=base+r；emitted[i]=pos+1+i 的 token | dspark_worker_v2.py:763-765; planner.py:835 | 6 行**一次 forward**（= 我们 SWALLOW 臂布局） |
| I2 | accept 判据：draft[j] 对**前一行** argmax（绝不同行） | dflash_utils.py:794 | 布局换偏移必须跟着换 |
| I3 | bonus = target_predict[correct_len] | dflash_utils.py:790 | |
| I4 | commit 指针 = prefix + correct_len + 1 | dspark_accept.py:724-725 | +1 是 anchor/bonus |
| I5 | verify KV **先写后验**（row j 要看到 row j-1） | dflash_info.py:160 | |
| I6 | reject 行不得进 draft ring（commit-gated inject，col<commit_len 才写） | dspark_kv_inject.py:126-134 | |
| I7 | s.ids == emitted.last()，**所有臂单一出口** | worker_v2.py:908-911 | 我们历史坑：只写一臂 |
| I9 | replay 前刷新图输入（ids/pos_rows/verify_lens/inject_gate） | dspark_verify.py:560-578 | |
| I10 | compressor/ring 状态：整体恢复+重放，无 per-row undo | 我们特有 compress_replay | 不能盲抄 pointer-only 回滚 |
| I11 | k>1 commit 时 draft ring 必须补齐 rows 0..keep 的 target hidden | dspark_verify.py:316-358 | 没有它 accept 随多 token 步退化 |
| I12 | 图不得冻结随 pos 变化的 host 值；输出缓冲 warmup 期分配 | dspark_verify.py:641-645 | |
| I13 | 各臂 collective 足迹逐字对齐（投票一致才进图） | dspark_verify.py:144-151 | |
| — | draft = 3 stage MoE 块 + Markov head **自回归 5 步**（无并行解） | dspark.py:68-79 | 两边 SAME |
| — | STATIC 模式（873.6 用的）：固定 6 行、无 confidence head、图按 shape 分桶 | planner.py:421-422 | 照抄这个，别追 compact |

**accept 规则（greedy）**：`argmax(target_logits)` 比token不比logits；spec 的 greedy 输出流 **必须逐 token 等于 eager 输出流**——这是移植全程的正确性金标准（每阶段必过）。

## 2. 算子移植工作清单（D 报告精缩；源码全在 in-repo，唯一黑盒 = MoE 本体）

**源**：triton = `python/sglang/kernels/ops/attention/dsv4/*.py`；CUDA = `kernels/jit/csrc/deepseek_v4/*.cuh`；TileLang = `kernels/ops/layernorm/mhc.py`；MoE 本体 = flashinfer pip（github.com/flashinfer-ai/flashinfer，已克隆 /home/smith/src/flashinfer）。

| 优先 | 算子 | 源 | 复杂度 | 数值红线 |
|---|---|---|---|---|
| V1 | indexer 后处理融合（score检查+无效过滤+页翻译一步，74 行 triton） | indexer_postprocess.py:9 | S | top-k 顺序保持；int64 页算术；NaN 拒/+inf 合法 |
| V2 | Q RoPE 并入 buffer 写（46 行） | q_rope_store.py:9 | S | FMA 序 even=fma(v,cos,-p*sin)；只 r>=448 旋转 |
| V3 | WO-A bf16 split-K（M=6，8 split） | wo_a_bf16_small_batch.py:37 | S~M | partial 全 fp32、split=8 固定 |
| V4 | split-K 归约内 MXFP8 量化 | wo_a_bf16_small_batch.py:94 | M | **最高危**：bf16→f32 两跳 amax；UE8M0 正舍入（含 subnormal）；swizzle off=(col//4)*512+row*16+col%4 |
| V6 | MoE 激活+量化融合 | silu_and_mul_masked_post_quant.cuh | M | clamp 在 SiLU 后量化前；我们 DSV41_SWIGLU_Q 是等价物先对拍 |
| V8 | 候选 mask 融合 | candidate_blocks.py:88 | M | last 块强制保留 |
| V5 | C2 verify 压缩融合（ratio2+norm+rope+量化+KV写单核） | c2.py:156→c2.cuh | M~L | **in-block pairing**：首行读 ring、其余行读块内前一行（同 launch 写 ring 无序）；ring_size>draft_len |
| V7 | mHC 四残差流融合 | mhc.py:358（TileLang） | L | Sinkhorn 20 轮序；别开 WARP_SPECIALIZED/TMA_LOWER（源码显式关） |
| V11 | MoE runner（flashinfer cutlass_fused_moe） | flashinfer pip | L | 无 align/mask，路由在核内；shared=额外 expert 槽；AR 在 MoE 后防双算 |
| V10 | MoE TP padding（TP8: 288→384） | mxfp4.py:470-499 | S | pad 落 gate/up 分割点；scale pad 填 UE8M0_ONE 非 0 |
| V9 | mHC/多流 overlap | deepseek_v4.py:1259-1321 | S | 纯编排，qkv_a_ready/q_lora_ready 两 event |
| P1 | RoPE+FP4 融合（plain） | fp4_rope.py:58 | M | **两级量化不可合并**（floor 6*2^-126 vs 1e-4 在 ÷6 两侧） |
| P4/P5 | C2 decode 池化 / GEMV+norm 融合 | pair_pool_decode.py / wo_a_bf16_gemv.py | S | enable_fp_fusion=False 是显式契约 |

**attention 本体**（DSA sparse，verify 批 6 行）：`sparse_attn_v4_paged_decode`（unified_kv_kernels/paged_decode.py:895，triton，split-K 三段）。**mHC 注意**：我方已是 3 核族（hc_mixes/hc_collapse/hc_post）与 sglang 一一对应——V7 可后置，先用现有族跑正确。

## 3. parity 清单（B 报告；官方 = ref_inference/{model,kernel}.py；用户红线"不能高也不能低"）

默认配置是"最大偏离"配置，8 个对齐点（大多一行 default 或小核已备）：
1. `DSV41_EXPERT_ACT_E4M3`：routed 激活必须 e4m3（我们默认 e2m1，差 8 倍）— top1 风险
2. `DSV41_WINDOW_KV_QUANT`：窗口 KV fp8(block32,e8m0) 往返（我们存 f32=更精）
3. `DSV41_INDEXER_FP4_RT`：indexer q/k fp4 往返（离散选择敏感：topk 换批位置）
4. `DSV41_ATTN_P_BF16`：P·V 概率 bf16
5. `DSV41_ROUTED_DOWN_QUANT`：路由权重在 w2 **前**乘 + down 输入 e4m3 量化
6. `DSV41_COMPRESS_LATENT_QUANT`：压缩 latent fp4(b16+e4m3 非幂次标度)
7. `DSV41_ACTQ_FLOOR`：amax 下限 1e-4（非 scale 下限 1e-30）
8. `DSV41_SEQ_ALIGN`：MoE down 归约升序 expert id（非 slot 序）
**注意**：移植的 sglang 算子自带其中大部分语义（V4 的 UE8M0 正舍入、P1 的两级 fp4、P bf16 等）⇒ 随算子落地，勿重复造门。tag 基座上这些行为需手工补（HEAD 的 glue.cu 内核语义正确可参考：win_kv_quant_rt/glue_latent_fp4_block16 等）。
head 折叠必须 v1 序（`head_gemv_bf16_v1_mrows`）；lm_head fp32 累加 K 序已 MATCH。

## 4. 图化与 stall 教训（C 报告）

- "图开 6 步停" 最可能 = **AR 轮损坏→attention 错→提前 EOS**（H1）或**捕获冻结 host 分支**（H4，如 compressor mode/grid 由 host start_pos 决定）；判别实验：`VERIFY_GRAPH=1 GRAPH_STEP=0`（只 verify 图、host AR）。
- `DSV41_AR_V5=0` 在 `GRAPH_STEP=1` 下是**无效变量**（唯一读者 tp.rs:1279 `graph || AR_V5`）——历史"铁律"是错的；图臂 vs 图关臂实际差 {图, AR 协议} 两变量。
- 新架构按 sglang 设计规避：accept/finalize 整块进 target 图 tail hook、device 内完成、host 事后读（dspark_verify.py:494-735）；槽位池按 shape 分桶（STATIC 恒 6 行=单槽）。
- verify 图 4 臂状态机（DRY/CAPTURE/REPLAY + 投票）在 HEAD 是复杂度来源；新核重写时保持"全员投票一致才进图"（I13）+ 捕获外刷新（I9）。

## 5. 分阶段落地（每阶段过"spec==eager 逐 token"红线才进下一阶段）

| 阶段 | 内容 | 完成判据 |
|---|---|---|
| **S0** | sglang-bbuf 上 b300-4 跑通（TP8 + STATIC + 模拟 acc5.5），拿同口径 tok/s + 1..100 金标文本 | 他们的数字落袋 |
| **S1** | `port-dspark` 分支（自 tag）+ 构建基建 cherry-pick（并行编译/戳记/cargo 重跑）+ 远端双产物构建 | eager 1..100 完美 + p50≈6.15ms 复现 |
| **S2** | m=1..6 统一执行核（host 编排 + pos_rows + 块内 causal + KV 先写后验；kernel 缺口按 V 清单补） | eager 输出逐字节不变；6 行手工步进 verify（ground truth）== eager |
| **S3** | 6 行单次 forward verify（I1-I5 全落地）+ accept/commit（I2-I4, I7）| spec==eager 全探针 |
| **S4** | draft 模型（3 stage + Markov + target lm_head；权重 loader 扩展）+ commit-gated inject（I6, I11） | accept≈2.2+；spec==eager 仍过 |
| **S5** | 算子替换按 V 清单序（V1/V2/V3/V10 → V4/V6/V8 → V5 → V7/V11），每步数值对拍 + p50 记账 | step 逐步下降；正确性红线不破 |
| **S6** | 整步图（SGLang tail-hook 设计）+ 多流 overlap（V9） | step p50 ≈ 7ms |
| **S7** | 同口径 bench（random 4k/1k + 模拟 acc5.5，我们需实现 SIM_ACC 门）vs S0 数字 | > 他们 |
| **S8** | parity 收尾（S5 未覆盖的 eager 对齐点）+ 文档 | 官方对齐 |

## 6. 纪律（沿用 + 本战役新增）

- 测量：step 只认 `[dsv41] step pos=` p50；e2e 一律 background；同会话背靠背 A/B；一次一个变量。
- 构建：双产物同源（build.sh 103a → touch build.rs → cargo build --release）；`check_artifacts.sh` 门。
- 正确性金标准 = **spec 流 == eager 流（greedy 逐 token）**，每阶段必测；文本判据用 `cat -A` 看字节级原文。
- 探针（D2H+同步类）绝不进默认路径（多 rank lockstep 死锁）。
- 移植的 kernel 一律带 env 门（`DSV41_PORT_*`），默认 OFF 验证后转正；ON/OFF 双路可测。
- 参考实现三份：官方 ref_inference（数值权威）> sglang-bbuf（工程真源）> 我们 tag（eager 锚）。docs/agent 旧结论一律不作证据。

## 🎯 kind 50/51 定谳 + o-proj 修复计划（2026-09-15 上午——TODO #9）

**kind 50/51 dump 结果（pos 20, layer 0）**：
| 链段 | 误差 | 说明 |
|---|---|---|
| q_post_rope (kind 50) | **0.073%** | q 链几乎完美 ✓ |
| attn_pre_woa (kind 51) | **0.254%** | attention 计算加了 ~0.18%（softmax 正常放大） |
| attn_o (kind 1) | **0.79%** | **o-proj 链 3× 放大 ← 最大来源！** |

**o-proj 链的 3× 放大分解**（每步约翻倍，f32 不同累加序的典型复合）：
wo_a (0.254%→~0.4%) → wo_b (→~0.6%) → AR (→~0.7%) → hc_post (→0.79%)

**修复方案：用 cuBLAS 替代自定义 gemv**——官方的 einsum/Linear 底层就是 cuBLAS（PyTorch matmul → cuBLAS）。用相同的库、相同的 tile 参数、相同的累加序 ⇒ 位级一致。
- wo_a: 官方 einsum("bsgd,grd->bsgr") → cuBLAS bf16 GEMV。我们当前用 gemm_fp8_mx_f32（自定义 warp K-split gemv）。替换为 cuBLAS 调用（bf16 权重 + bf16 激活）。
- wo_b: 官方 RowParallelLinear → cuBLAS fp8 GEMM。我们当前用 gemm_fp8_mx。替换为 cuBLAS。
- AR: 官方 NCCL vs 我们 P2P v5——求和序可能不同。可用 NCCL 替代测试。

**实施步骤**：
1. 在 chain_dev.rs 的 wo_a 路径（WOA_F32 分支）添加 cuBLAS 调用选项
2. 权重需从 fp8 反量化到 bf16（启动时一次性或按需）
3. 激活已 bf16 值域 ✓（bf16_o 舍入）
4. 用 cublasGemmEx with BF16 data type
5. A/B 验证 attn_o 降幅

**注意**：ref 的 kind=6 prefill 记录是 7680 f32s（wkvpost 写 [1,15,512] 一条合并记录），解析需变长处理。ref 的 kind 50/51 n=4096 ✓。

## wo_a cuBLAS 路径实施完成（2026-09-15 上午，commit a2ee46c9）

**实施内容**：
1. `kernels/cuda/dsv41_glue.cu`：新增 `dsv41_dequant_fp8_ue8m0` 内核（fp8 e4m3 + ue8m0 块 scale → f32）
2. `crates/ferrite-models/src/dsv41/device.rs`：FFI 类型声明 + 符号加载 + `dequant_fp8_ue8m0` wrapper
3. `crates/ferrite-models/src/dsv41/chain_dev.rs`：`woa_cublas()` gate（DSV41_WOA_CUBLAS，默认 OFF）+ cuBLAS 分支
   - 每次调用反量化权重到 f32（测试路径，后续可缓存）
   - 逐 group 调用 `gemm_f32`（cuBLAS GEMV，匹配官方 einsum 的累加序）

**AR 排除测试（62a1c849）**：DSV41_AR_V5=0（NCCL 路径）attn_o = 0.77%——与 P2P v5（0.76-0.94%）相同
⇒ **AR 求和序不是 o-proj 3× 放大的主源**，wo_a/wo_b gemv 的计算顺序是剩余嫌疑

**待验证**：DSV41_WOA_CUBLAS=1 的 attn_o 是否从 0.79% 下降（验证轮 0e231742 在飞）

**注意**：ref 的 kind=6（kv_gemm）prefill 记录是 7680 f32s（wkvpost 写 [1,15,512] 一条合并记录），
解析需变长处理。ref 的 kind 50/51 n=4096 ✓。

## wo_a cuBLAS OOM 修复（2026-09-15 上午，commit 0767df4e）

**第一版 bug**：每次 wo_a 调用都 `alloc` 新 DevBuf（40 层 × 200 步 = 8000 次分配）
→ `cudaMalloc(16.0 MiB) failed, this rank has already allocated 192.3 GiB`（OOM 崩溃，RC=1）。

**修复**：thread_local HashMap 按权重指针缓存 + `Box::leak`（权重是常量，
40 层 × ~512 KiB ≈ 20 MiB 总量）。反量化只做一次。

**关键确认**：wo_quant_fuse 默认 OFF（`unwrap_or(false)`）→ WOA_F32 块（含 cuBLAS 分支）
确实被进入。前一轮的 attn_o 0.79% 是 cuBLAS 路径生效时的读数（与基线相同）
—— **wo_a 的 gemv vs cuBLAS 累加序差异不是 3× 放大的主源**（但前轮 OOM 崩溃可能污染了输出，
需清洁复测）。

## 🎯 rope 表位级修复——BIT_EXACT 定谳（2026-09-14 04:40，commit 40b812ea）

**单元测试链路（用户指示"直接证明不猜"后的完整闭环）**：
1. C 测试程序（/tmp/rope_test.c）dlopen .so 调用 dsv41_rope_precompute，dump cos/sin
2. 官方侧 Python 逐行照抄 precompute_freqs_cis（model.py:369-389），torch.polar 生成表
3. 逐位对比

**定谳数据**：
| 版本 | cos 差异 | sin 差异 | 结论 |
|---|---|---|---|
| 设备端 cosf（--use_fast_math） | 6436/8192 (78.6%) | 7693/8192 (93.9%) | 差 1-4 ulp |
| host 完全照抄官方（40b812ea） | **0/8192** | **0/8192** | **BIT_EXACT** |

**根因**：官方 precompute_freqs_cis 在 **CPU** 上运行（torch.arange 无 device 参数 → CPU；
torch.polar → glibc cosf/sinf）。CUDA 设备端的 cosf/sinf 是完全不同的实现（fast math 下更是
快速近似）。**位级一致唯一路径 = host 上用 glibc 的 cosf/sinf。**

**修复的三个关键点**（之前 host 双精度版本失败的原因）：
1. `1.0f / powf(base, q)`（正指数再倒数）——不是 `powf(base, -q)`（负指数），浮点结果不同
2. `(float)tt * freqs[i]`（f32 乘法）——不是 `(double)tt * f`（double 乘法），舍入不同
3. `cosf(a)/sinf(a)`（host glibc）——不是 `std::cos((double)a)`（虽然 glibc 内部也是 double，
   但直接调 cosf 保证走完全相同的代码路径）

**启动时一次 H2D**（与权重上传同类，非推理热路径）——用户确认 rope 表提前算好即可。

**下一验证**：数数 1-100（rope 位级修复后 kv_RT 的 4/512 元素差应消除 → attention 0.254%
应大幅下降 → attn_o 0.79% 应显著改善 → 数数断点应推进）。

## ⚠️ wkvpost 解析 bug 定谳（2026-09-14 04:55）

**发现**：ref_diff.py 的 wkvpost hook 写 kind=6 时用 `oo = output`（shape [1,s,512]），
`oo.shape[0]=1`（batch 维）→ 每条 forward 写 **一条 [s,512]=7680 f32s 记录**（prefill），
而我们引擎写 per-position 512 f32s。**SIZES 表按 512 解析 → prefill 后所有记录错位**。

**影响范围**：kind=6 变成 7680 之后的所有轮次，xn/kv_gemm/kv_RT MISS、attn_o/moe_o 的
"数字"以及 kind 50/51 暴力扫描的 110% 差异——**全部是错位伪影，不可信**。

**修复**：wkvpost 改为 `o[0]` 取 [s,512] 再逐行写 per-position [512] 记录（与 xpost 同构）。

**教训**：hook 写 dump 时必须先 `o[0]` 去掉 batch 维再逐行迭代（oo.shape[0] 是 batch 不是
seqlen）；解析侧 MISS 或荒谬数字（110%）时先查记录格式，不要直接采信。

## 🎯 用户定谳：YaRN 参数用错是 kv_RT 差异的根因（2026-09-14 05:00，决定性证据）

**证据链（列级精确吻合）**：
- 表对比 pos=20：我们的表（YaRN 开）vs 官方滑窗表（YaRN 关）——**仅 i≥21（列 490-511）分叉**，i≤20（列≤488）完全相同
- kv_RT 的 4 个差异列（491/497/504/510）**完全落在 DIFF 集合 {490..511} 内**，混合区外零差异

**根因**：官方 model.py:688-700 **按层类型分叉**——纯滑窗层（compress_ratio=0）用 `original_seq_len=0`（**YaRN 关**）+ `rope_theta=10000`；压缩层用 `original_seq_len=65536` + `compress_rope_theta=160000`。我们的主表（query + window KV）错误传了 `cfg.original_seq_len`（65536）⇒ **YaRN 被错误开启**。

**修复（一行）**：主表 `original_seq_len` 传 `0`（commit 已提交）。cos_comp/sin_comp 保持 65536 + 160000 不变（与官方压缩层一致 ✓）。

**自证循环教训**：之前 rope 单元测试 BIT_EXACT 的"官方侧"是我自己写的 Python 复现（也带 YaRN）——两边同一个公式必然 0 差异。**真正的参照必须来自模型的实际表参数（每层的 original_seq_len 分叉），不能用自己复现的公式当参照。**

**两点更正（用户指出）**：
1. `apply_rotary_emb` 是 **in-place** 的（`y.copy_(x)`，docstring 明写 "Rotate x in place"）——返回值被丢弃不表示没生效，**q 确实被 rope 了**
2. "q 前 448 列 100% 不同 ⇒ wq_b 根本差异"**不成立**：attn_o 仅 0.79%（≈2 bf16 ulp）证明 q 无根本差异；kind 50 的 106% 是 **dump 布局不可比的伪影**（head 级余弦"多对一"是错位特征）

**已排除（单元测试 BIT_EXACT 全过）**：rope 表（glibc cosf/sinf 照抄公式）、rmsnorm、norm+rope、RT。**已排除（op-level 0.000%）**：xn、kv_gemm。o-proj 自身 0.03%。
