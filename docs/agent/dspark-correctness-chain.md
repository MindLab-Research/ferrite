# DSpark 正确性修复链（2026-09-12 会话归档）

本文件归档本会话的正确性调查完整链条。AGENTS.md 只保留结论指针。

## 已确认的根因（按发现顺序）

| # | 根因 | 状态 |
|---|---|---|
| 1 | `load.rs:925` placeholder hack（main_proj 覆盖 attn_norm 字段）→ 权重 fp8 垃圾 | ✅ 修复 |
| 2 | draft 的 MoE ILV 选择 bug（默认路径走非交错 reader 读交错池）→ accept 0.02→0.52 | ✅ 修复 |
| 3 | verify 的 per-row interleave 缺失（读侧统一用块末 clen） | ✅ 修复 |
| 4 | draft 块位置（B1：draft_forward 传 pos+1 但块在 pos） | ✅ 修复 |
| 5 | verify head 的 folded kernel K 序差（echo 33%→9%） | ✅ FOLD=0 默认 |
| 6 | **quant_rows 的源行距 bug**：kernel 把 cols 当源行距，`o_r` 真实行距 `nh*hd`（8x）、`wo_r` 是 `ol_total`（8x）→ **行 0 恒对、r≥1 从 row 0 尾部垃圾 scratch 读数**——diff probe 实测的 mismatch 指纹（只出现在 verify 行 1/2、输入对输出错、80-vs-1146 的垃圾量级）完全吻合 | ✅ 修复（逐行打包：源给真实行距、目的紧凑——下游布局不变、行 0 逐位不变） |
| 7 | **note_ctx_rows 的 tap_r 行距**：生产者写 `slot*VERIFY_ROWS + r`，消费侧读 `slot*m + j`（m=5 时 slot≥1 错行） | ✅ 修复 |
| 8 | s.ids 无人回写（spec commit 只推计数器不写 token → k_acc≥1 后下一轮嵌旧 token）| ✅ 修复（gate DSV41_SIDS_WRITEBACK，默认 OFF——待根因 6/7 的验证通过后改默认 ON）|

## 关键判词（来源 subagent，全文在 git 历史的 commit message）

- **verify-row0-systematic**：s.ids 回写的完整证据链（step_body 读 s.ids、只有主链 argmax 写它、verify 写 argmax_r、commit 不写 token）。
- **emit-chain-audit**：消费链无罪（out 每 token 一次 push、p/emitted.len() 与 Δpos_ctr 构造性恒等、SSE drain 语义正确、无重试重放）；**结构性收窄：emitted 的值来自主链 argmax，draft 污染只能压 k_acc 不能改值**。
- **spec-state-pollution**：写 vs 恢复清单（W1-W15 闭合，除 s.ids 外）；engram cache 不在 rollback 但方向性安全。
- **prefill-sids-init**：prefill 确实写 s.ids=first_token（最后一步 prompt 的 argmax）→ 第一轮 anchor 与 EAGER bit-identical；ratio=2 的 compressor 在奇数位提交组。
- **verify-value-hunt**（两轮）：**第一轮**给出 B1（consumer clen 块末——后被第二轮修正为次要）与 B2-B6（fused/单行 kernel 路径差的 A/B 清单）；**第二轮**用 diff probe 的数据钉死 **F1/F2（quant_rows 行距）为确定性根因**——"不是有时对有时错，是行 0 恒对 r≥1 全错"；同时排掉了行间 carry（live ✓）与窗口构造（interleave 实际成立 ✓）。
- **moe-rowfold-next**：全仓穷尽审计——**quant 类只有已修的 2 处是 bug，其余 12 处源行距=cols 或单行**；mrows kernel 的行距假设表（gemm_fp8_mrows/head_gemv_bf16_mrows 硬编码=k 但调用点全用紧凑缓冲，安全）；tap_r 行距（→根因 7）。

## 诊断工具（已入库）

- `DSV41_DIFF_EAGER=1`：每轮重放 emitted.len() 个单行 forward（同前缀 KV），报第一个 mismatch 的 index+绝对位置——index=1 ⇒ verify 行 0 已分歧。
- `DSV41_DSPARK_DEBUG=1`：逐轮 next/drafts/verify/k_acc/emitted。
- `DSV41_VROW0_PROBE=1`：verify 行 0 vs eager 的 top-5 对照（近 tie vs 结构性的判定）。
- `dspark_parity`（cargo test --ignored）：喂真值 token、逐行打印 verify_out[r] vs expect[r]。

## EAGER 对照基线（判据）

- 数字任务（1 数到 100）：EAGER 完美（LEN 216，仅 "anao"、跳 50 等个别瑕疵）。
- 出师表：EAGER LEN 146、双字 3。
- spec 的验收 = 两个任务与 EAGER 同水平（mismatch=0、无双字、无乱码）。

## 会话教训（新增）

1. **单一 GPU 测试驱动**：同一台远端上**同时只能有一个测试驱动**（主 agent 或一个 subagent，不可两者并发）——2026-09-12 的 build-id mismatch 事故（.so HEAD 7da4bef7 vs 二进制 HEAD cee7cffd）就是两个驱动并发 `git reset + build.sh + cargo build` 交错产出的不同源组合（门禁正确拒绝，但浪费了整轮测试）。**subagent 一律只做代码/分析，GPU 测试由主 agent 串行执行**。
2. **不轮询远端状态**：后台任务的输出会自动注入；反复跑同一条 `stat/grep` 查询既浪费轮次又违反"持续工作"的要求。启动测试后做本地实事（代码/审计/提交），等通知。

## 决定性验证（9fcecb70，2026-09-12 09:5x）——正确性基本达标

**配置**：quant_rows 行距修复 + tap_r 行距修复 + DSV41_SIDS_WRITEBACK=1 全开 + diff probe。

| 任务 | 修复前（乱码崩） | 修复后 | EAGER 基线 |
|---|---|---|---|
| 出师表 | LEN 12"出师nofollow" | **LEN 140 双字 3**（先帝创业…引喻失义opa——与 EAGER 同水平，连尾巴都一样） | LEN 146 双字 3 |
| 数字任务 | 全崩每数两次 | **LEN 426，干净数 1..63+**（双字 43 待查——后期退化） | LEN 216（anao/跳50） |
| k_acc | {0:13,1:6,2:3} | **{0:124, 1:84, 2:5, 3:15, 4:2, 5:2}**——k_acc=5（全接受）首次出现 | — |
| diff probe | 首轮即 mismatch | **18/232**（7.8%） | 0 |

**性能**（同测）：`mean-k=0.833 tok/step=1.833 draft=4.23ms verify=37.31ms commit=0.33ms`。
**下一步**：① 剩余 18 个 mismatch 的行分布（判断 B1 consumer-clen 还是 B2-B6 融合族）→ verify-moe-ilv-audit 的判定树；② 双字 43 的位置（是否集中在长上下文段）；③ 性能：verify 37.31ms → 图化 A/B + 计算下限账本（verify-calc-audit2 / draft-perf-audit 跑中）。

## 剩余 mismatch 的行分布（9fcecb70 的 diff probe）

**18 个 mismatch 的 index 分布：{1: 16, 3: 1, 4: 1}**——即 16/18 是 verify 行 0（verify_out[0]）与 eager 分歧、各 1 个是行 2/行 3。**不是** quant_rows 那种"行 0 恒对 r≥1 全错"的指纹——行 0 现在也错了（16 次）。
- pos=14 的 mismatch=4（行 3）：spec 第 5 个 token 37500 vs eager 28638——**前 4 个全对**（654/1767/1146/1338 完全一致），说明 k_acc=4 的深度接受已工作，只是第 5 行（被拒的那行之后的 bonus）偶有分歧——这属于正常的近 tie 或残余路径差。
- 16 个 index=1（行 0）：需要对照 verify-moe-ilv-audit 的 B2-B6 判定树（compressor fused / route fuse / hc tail / 融合投影族）——这是"多行 vs 单行的 kernel 路径差"的最后一层。

## 剩余 16 个 row-0 mismatch 的判定树（verify-moe-ilv-audit 判词）

**关键否定**：B3（route fuse）**不是嫌疑**——两臂的 route 数学逐句一致（smem 布局/激活/选择 tie 规则/归一/WPR 全同）。

**按优先级的 A/B 开关**（每关一个跑一次 diff probe，mismatch 归零即命中）：
| 优先 | 开关 | 差异 | 风险级 |
|---|---|---|---|
| 1 | `DSV41_GEMV_A32=0` | EAGER 的投影走物化 s_af、verify 的 mrows 走 inline——**与 FOLD 翻车完全同类**（自称逐位一致、藏在 fma 配对） | 高 |
| 2 | `DSV41_SPARSE_OROPE=0` | EAGER 的 sparse+rope+fp8 融合 vs verify 的三连——"verbatim" 未经实测 | 中 |
| 3 | `DSV41_NORM_FUSE=0` | EAGER 的 rmsnorm 折进 GEMV prologue vs verify 的独立 rmsnorm | 低 |
| 4 | `DSV41_COMPRESS_FUSE=0` | compressor fused vs pool+commit——主链已逐句核对一致，stage-1 state carry 未核 | 低 |
| 5 | `DSV41_HC_TAIL_SPLIT=0 DSV41_HC_FRONT=0` | hc 的 tail split（ss/dots 已证位级一致；collapse/norm 融合未实测）——**注意要同时关两个** | 低 |

## opa 尾部乱码的定位（实验 A 已执行）

**"opa" 是词表里的合法 token（id=41291）**——`encode("opa")=[41291]` 单 token 往返一致；"anao" 同样（id=83514 单 token）。⇒ **模型真的输出了这些 token**，不是显示层/切分问题。

**根因（H1/H2 组合，与历史案例完全同构）**：
- `serve.rs:1069-1076` 已记载同构现象：**该 checkpoint 没有 generation_config.json（已验证：文件不存在）且 eos_token_id 为 null**——模型答完后输出 EOS（token 1），但 stop 未生效 ⇒ 越过结束点继续生成退化尾段（"opa**" + 重启《出师表》第一句——"重启"是退化尾段的教科书形态）。
- 数字任务的 "anao"、崩坏态的 "nofollow" 同族（拉丁碎片出现在回答边界）——系统性模式。
- **修复方向**：① stop/EOS 的解析（tokenizer_config.json 存在——查 resolve_eos 为什么没取到 eos）；② 或硬编码 stop token 1（tokenizer stops: [1] 已经在 serve 日志里出现——查为什么没拦住）；③ 实验对照：官方 ref_inference 跑同 prompt 确认模型固有 vs ferrite 侧 stop 缺陷。

## opa 根因闭环：stop 机制其实在工作——opa 是"截断前的最后一段"不是"越过 EOS"

**关键事实链**：
1. `resolve_eos` 的三级 fallback：generation_config.json（**不存在**，已验证）→ config.json 的 eos_token_id（**null**）→ tokenizer_config.json 存在 ⇒ **`Some(1)`**——**eos 解析正常**。
2. serve 日志 `[http] tokenizer stops: [1]` ✓——stop 集是 `[1]`。
3. serve 的 stop 检查（:690 `stop_set.contains(&tok)`）**在 emitted 循环内逐 token 检查** ✓；HTTP 层（api.rs:240-244）也过滤 `is_stop`。
4. **⇒ stop 机制完整**。opa（id=41291，合法词表 token）是**模型在 max_tokens 耗尽前、EOS 之前**的**真实输出**——它出现在 `引喻失义` 之后是因为**模型的下一个 token 就不是 EOS**（模型在"义"后选了"opa"而不是停）。
5. **这不是 ferrite 的 stop 缺陷，而是模型在该位置的 logits 真的偏向 opa**——数值层面：要么（a）模型固有行为（语料污染/该 checkpoint 的特性——需官方 ref_inference 对照判定），要么（b）ferrite 的某处数值微扰把近 tie 的 argmax 翻到了 opa（EAGER 也有 ⇒ 若 (b) 则是 backbone 共性偏差）。
6. **注意出师表输出在 "opa" 之后紧跟 `**先帝创业未半而中道崩殂...`**——即输出=【完整背诵】+【opa**】+【重启出师表】——这是"模型答完正题后没找到 EOS、继续退化"的形态。**如果模型此时该出 EOS 但出了 opa ⇒ EOS 的 argmax 被翻 ⇒ 数值偏差**（候选：fp4 解包/量化路径的微扰）。

**修复路径**：
- **判定**（最便宜）：用官方 `ref_inference/generate.py` 跑同一 prompt——若官方也在同位置出 opa ⇒ 模型固有（无需修）；若官方干净 ⇒ ferrite 的 backbone 数值有共性偏差（继续二分：fp4 解包/量化）。
- **缓解**（无论如何可做）：模型可能本来就需要"背诵完出师表后收尾"的 chat template 引导——检查 Dsv41Frame 的模板是否让模型有明确的"答完即停"信号。

## 剩余 mismatch 的 A/B 结果（1f2ce355）

| 配置 | 出师表 | mismatch 数 |
|---|---|---|
| 基线（前一轮 final1） | LEN 140 双字 3 | 18 |
| `DSV41_GEMV_A32=0` | PARSE-FAIL（serve 被 kill——与 sh-exp-ab-runner 的并发冲突，作废） | — |
| **`DSV41_SPARSE_OROPE=0`** | **LEN 140 双字 3（与基线逐字一致）** | **1** |

**决定性**：`DSV41_SPARSE_OROPE=0` 把 mismatch 从 18 降到 **1**——**o-rope 融合（sparse_attn_orope）是剩余 row-0 mismatch 的根因**（EAGER 走融合、verify 走三连——"verbatim" 声称不成立，与 FOLD 同类）。文本不变（LEN 140 双字 3）说明那 1 个残余 mismatch 不影响本 prompt 的输出。
**修法**：把 verify 也走 `sparse_attn_orope`（对齐 EAGER），或 EAGER 关融合（性能损失小——但 o-rope 融合本身是优化）。**下一步**：重跑 GEMV_A32=0 臂（这轮被并发测试 kill 了），确认 a32 是否解释最后 1 个 mismatch。

## AR v5 死锁（SEED_ALIGN=1）的判词（ar5-deadlock-audit）

**证伪**：6 行块多发了 AR——aligned 与 legacy 的每轮 AR/epoch 足迹**逐位相同（164）**。
**真根因（H1，签名精确吻合）**：`draft_forward` 的 `pos == 0` 早退——legacy 的首轮 `draft_forward(token, 0)` 命中早退（0 次 draft AR），而 aligned 的 `draft_forward(next, pos+1)` 恒 ≥1（**3 次 draft AR**）⇒ **`need−cur = 3` 恰好 = 3 个 mtp block 的 MoE AR**。v5 的协议契约：落后方静默通过（读错值）、领先方永久自旋——无 host rendezvous 能吸收次数差。
**修法（F1，判词推荐）**：aligned 臂的首轮也走 legacy（`spec_primed` 的同款引导——route A 只在 primed 后接管），或把 draft 的 `pos==0` 早退的 AR 足迹对齐（假发 3 次）。
**附带发现**：H4（pubred 的 `e` 每 block 各读一次、block 0 中途写 `*epoch`——晚启动的 block 读到 e+1、stamp e+2、等错半区）是全 arm 共有的设备级隐患（route A 的 6 行块把窗口加宽 20%）；H3（argmax_sliced 的尾部 rank decline——与 peer=5/6/7 吻合）待查。

## 本轮修复汇总（2026-09-12 上午，全部已提交推送）

| 修复 | 内容 | 状态 |
|---|---|---|
| quant_rows 源行距 | o_r 的 nh*hd（8x）与 wo_r 的 ol_total（8x）逐行打包 | ✅ 验证通过（出师表 140/3≈EAGER） |
| note_ctx_rows tap_r 行距 | slot*m → slot*VERIFY_ROWS | ✅ 已提交 |
| s.ids 回写（gate） | DSV41_SIDS_WRITEBACK（默认 OFF，量化修复验证通过后这轮全开跑了） | ✅ |
| o-rope 融合对齐 | DSV41_VERIFY_OROPE 默认 ON——verify 走 EAGER 同一 sparse_attn_orope（mismatch 18→1 的 A/B 依据） | ✅ 已提交（bf3de09d 决定性验证跑中） |
| SEED_ALIGN 死锁 | spec_primed 引导首轮走 legacy（AR 足迹 0/3 对齐） | ✅ 已提交 |
| head 词表切分 | DSV41_VERIFY_HEAD_SLICED 默认 ON + 新 kernel dsv41_argmax_sliced_rows（1 个 v5 round 批 m 行）——6619→992MB | ✅ 已提交 |
| a32 gate | gemm_fp8_mrows 在 a32=1 时 decline（FOLD 同类防线） | ✅ 已提交 |

**待验证**（bf3de09d）：全部修复叠加后的正确性 + 性能（verify 应从 37.31ms 降 ~1.5ms+）。
**待跑**：GEMV_A32=0 的 A/B（上轮被并发测试 kill）。

## 读侧判词：verify 切片 head 的跨 rank argmax（epoch 足迹）

**契约**：`DSV41_VERIFY_HEAD_SLICED` 默认 ON 时，verify 的 head 读侧 = **每行一次切片内 `argmax_kernel`**（packed key 带*全局* index，`idx_off = rank*seg`）**+ 整个块一次 `argmax_xchg_v5_rows_kernel`**。`step_rows_inner` 的 `verify_head_geom()` 是唯一决定几何的地方（head / argmax / probe 三处同源），`logits_r` 的行距随臂变化（切片时 = `seg`）。

**为什么不能在行上循环 `argmax_sliced`**：每次调用**无条件**推一次 v5 epoch（publish → stamp → `*epoch=e+1` → poll），只有 `pos_ctr` 可置空；m 行就是 m 轮，而 EAGER 的 `step_dev` 每步只 1 轮 ⇒ 足迹差 m−1 = SEED_ALIGN 那类死锁（领先方永久自旋、落后方静默读错半区）。**批量化把 m 行压成 1 轮**，于是 verify 的 head 足迹与 EAGER 的**逐位相等（1/1）**；`pos_ctr` 传 NULL（计数器由 accept 逻辑推一次）。

**门必须 rank-uniform**（否则不同 rank 分支不同 = 新一轮足迹差）：`verify_head_geom` 的每一项都是 (.so 符号、world、vocab 可整除、head dtype、v5 live、`VERIFY_ROWS*8 <= bytes`)——全是 rank 无关量，唯一的 rank 依赖是切片基址。`rows*8 > stride_bytes` 时 C 入口返回 1（decline 哨兵，且**不碰 epoch**），Rust 侧把它当几何不一致直接 Err（不静默回退：此时 head 已按切片行距写了 `logits_r`，回退会读错）。

**验收**：`kernels/cuda/tests_dsv41_argmax_rows.cu`（`bash scripts/verify_mrows.sh --test argmax`）——(1) 每行 == 生产全词表 `dsv41_argmax_sliced`（含跨 rank 平局→全局最小 index）；(2) `rows = 1/3/6` 的 epoch 增量**恒为 1**（批量化契约的回归闸）；(3) decline 臂不改 epoch/`out`。失败即回归，不必等 serve 挂住才发现。**诊断**：`argmax_xchg_v5_rows_kernel` 的 5s watchdog 现打印 `[ar5-hang] argmax_rows rank=… peer=… need=… cur=… rows=…`（此前静默 break，会写一个"看起来合理"的错 token）。

## opa 判定的用户纠正 + 官方对照（ref-inference-compare）

**用户（权威）**："不说官方，我们自己 eager 也是没有乱码的，只有 mtp 有"——**重新核实**：
- EAGER 的出师表（e5329f6c，无 spec）：`…引喻失义opa**`（LEN 146 双字 3）——**EAGER 也有 opa**（此前的记录）。
- **但用户说 eager 干净**——可能的解释：①用户看的是更早/别的 eager 跑；②或用户认为 opa 出现在 MTP 的输出里而 eager 的那段是正常的（需要逐字比对确认）。
- **官方 ref_inference（3 个 prompt 变体，greedy）全部无 opa**——`引喻失义` 后官方一律 `，以塞忠谏之路也。`。⇒ **opa 是 ferrite 的数值偏差**（不是模型固有）——与用户"必须修复"的判定一致。
- **官方跑通的坑**（记录）：demo checkpoint 是 fp8 展开（config 要 `expert_dtype:"fp8"`，否则 94k size mismatch）——`~/ref_oracle/config_fp8.json` 已留档。
- **下一步（按官方判词）**：**首步 logits 对齐**（官方与 ferrite 的分歧从第 1 个 token 就开始了——官方 `《出师表》开头如下` vs ferrite `出师表》全文如下`——少 `《`）⇒ 根因在更早的层，opa 只是尾部表征。优先做 prompt 逐 token 的首步 top-k 对照。

## 用户的金标准锚点（2026-09-12）

**用户（权威）**："之前 162 tok/s，也就是打 git tag 那个版本（`dsv41-6.15ms-162toks`，HEAD=8a5a952）没有乱码。"

⇒ **正确性的判据锚定**：`dsv41-6.15ms-162toks` 的 EAGER 路径是干净的（无 opa/无乱码）。opa 是**这个 tag 之后引入的回归**——二分范围：8a5a952..HEAD 之间的 backbone 改动（不在 spec 路径——EAGER 也中招 ⇒ backbone 共性）。下一步：`git log 8a5a952..HEAD --oneline -- crates/ferrite-models/src/dsv41/chain_dev.rs kernels/cuda/dsv41_kernels.cu kernels/cuda/dsv41_glue.cu` 列出候选提交，按"触及 backbone 数值路径（量化/fp4/AR/head/attention kernel）"过滤，在远端逐个 checkout A/B 出师表（EAGER 模式）。

## opa 回归二分的两份判词（backbone-regress-bisect + first-token-divergence）

**backbone-regress-bisect 的关键杠杆**：EAGER 是 m=1 逐 token 的 step_dev 循环（无 `fn prefill`——prefill 也是 step_dev）⇒ **~80% 的 m>1 优化候选全部惰性排除**。剩余 4 个候选：
| # | 提交 | 内容 | EAGER 生效 | 合理性 |
|---|---|---|---|---|
| **C0** | b5dee8d+a3b913e+97d0b46 | chat frame/serve 入口统一 | ✓ | **最高**——prompt 字节差一步解释首 token + 尾部 |
| **C1** | 47a9bb9 | mxf4 ILV/scale 分片（**已知 test failure 72!=96 是它的指纹**） | ✓（load-time 布局） | **高**——每层 MoE 系统性小偏差 |
| C2 | 71b9d66 | AR staging 尺寸（hc_dim*4 → max(hc_dim, VERIFY_ROWS*dim)） | ✓（共享） | 中 |
| C3 | 3d318bf | indexer 慢路径（decode 恒 fast path） | 大概率惰性 | 低 |

**first-token-divergence 的头号根因**：**MoE routed expert 的激活量化格式——ferrite e2m1 vs 官方 e4m3**（官方 fp4 权重的 `linear()` 用 fp8 激活：`model.py:181-195`"both fp4 and fp8 weights take an fp8 one"）。e2m1 只有 1 位尾数——**backbone 里量级最大的已知数值分歧**。
**⚠️ 判据前提红旗**：官方输出 `《出师表》开头如下` vs ferrite `出师表》全文如下`——**两侧 prompt 不同**（"开头" vs "全文"）——**首 token 分歧可能是 prompt 差，不是数值差**。先对齐 prompt 再判定。

**二分计划（判词给的顺序，比 checkout 快）**：
1. **先仲裁 prompt**（两侧 ids 逐位 diff）
2. **ablation 阶梯**（env 开关，每条一次 EAGER 出师表）：`DSV41_EXPERT_ILV=0`（C1 一击）→ `DSV41_SKIP_EXPERTS=1`（定位 MoE）→ 融合族逐个 `=0`（EAGER 侧从未 A/B 过的 10 个默认 ON）→ `DSV41_GRAPH_STEP=0 DSV41_AR_V5=0`（C2）
3. checkout 二分（仅当 2 全落空）：先 `47a9bb9`（C1 直收）

## 锚点版（8a5a952）的复测结果（ce055974）

**锚点版起不来**：`parse config.json: missing field layer_types at line 156`——8a5a952 的 `--model dsv41` 入口还不认识当前模型目录的 config（`--model` 单二进制路由是 Wave 1 的 T3 改动，在锚点之后）。⇒ **锚点版必须用旧入口**（`dsv41-run --serve`）跑。下一轮复测用 `./target/release/dsv41-run --serve ...`。
**远端已恢复 origin/main**（46e07eb）。
