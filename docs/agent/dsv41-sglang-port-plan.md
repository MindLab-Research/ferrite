# DSV4.1 DSpark 移植总案：SGLang 算子 → ferrite（2026-09-13 定稿）

> 用户指令链：① 参考官方 PyTorch 彻底改好正确性；② MTP block-5 step ≈ 7ms 即达标（同口径 > sglang 873.6 tok/s）；③ eager 之外的算子判垃圾，**照抄 SGLang**（BBuf/sglang@835c3909，V4.1-Flash 真源；master 无 V4.1）；④ 移植→确认正确→优化到比他们快。
> 验证机：AWS b300-4（ubuntu@43.202.208.136，8×B300，模型 /opt/dlami/nvme/models/DeepSeek-V4.1-Flash）。

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
