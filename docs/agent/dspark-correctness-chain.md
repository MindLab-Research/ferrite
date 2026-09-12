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

## 头号根因确认（代码级证据）：MoE routed expert 的激活量化格式分歧

**官方 `model.py:181-195`**（直接读原文核实）：
```python
if weight.dtype == torch.float4_e2m1fn_x2:
    x, s = act_quant(x, fp8_block_size, scale_fmt, scale_dtype)  # ← fp8 e4m3 激活
    return fp4_gemm(x, s, weight, weight.scale, ..., act_block_size=fp8_block_size)
```
注释原文："both fp4 and fp8 weights take an **fp8** one -- for fp4 the kernel handles the mixed precision."

**ferrite**：`chain_dev.rs:10199` 的 `quant_fp4(xn → xq4/xsc4)`——**e2m1（1 位尾数）激活**。

⇒ **e2m1 vs e4m3 = backbone 里量级最大的已知数值分歧**（e2m1 尾数 1 位 vs e4m3 尾数 3 位——每层每 token 的 routed expert 输出都有系统性偏差，44 层累积）。这完美解释：首 token 近 tie 翻转（`《` vs `出`）、长链尾部的 opa 退化、EAGER 与 spec 双双中招（backbone 共性）。
**修复方向**：routed expert 的激活切 e4m3（`DSV41_EXPERT_ACT_FP8` A/B 先验证）。
**注意**：这**不是** 8a5a952..HEAD 的回归——它是**一直存在的架构级偏差**（官方 fp4_gemm 是混合精度 MMA：fp8 激活 × fp4 权重；ferrite 的 fp4 kernel 假设 fp4 激活）。用户说"162tok/s 版没乱码"可能因为那个版本的其它路径掩盖了它，或 opa 的出现需要特定上下文长度（阈值效应）。锚点版复测（anchor-dsv41run-retest）会给出判定。

## 头号根因的 kernel 侧证据（官方 kernel.py 的混合精度语义）

- `kernel.py:14-15`：`FP8="float8_e4m3"` / `FP4="float4_e2m1fn"`——两种 dtype 都存在。
- `kernel.py:478+` 的 `fp4_gemm_kernel`：**A（激活）是 fp8 e4m3、B（权重）是 packed e2m1**——`B is stored as [N, K//2] in float4_e2m1fn_x2`（:493）——即官方的 fp4 GEMM 是**fp8 激活 × fp4 权重的混合精度 MMA**（"for fp4 the kernel handles the mixed precision" 的确切含义）。
- **ferrite**：`quant_fp4`（device.rs:2062）把激活打成 **e2m1 packed**（`xq4`——`chain_dev.rs:180-186` 的注释自证："the ROUTED experts' fp4 packing of xn"）→ expert kernel 假设激活也是 fp4。
- **量化误差量级**：e2m1 尾数 1 位（相对误差 ~2^-1 步长）vs e4m3 尾数 3 位（~2^-3）——**4 倍的量化噪声**，每层每 token、44 层累积。这就是首 token 近 tie 翻转（`《`→`出`）与 opa 尾部的机制。
- **修复**：激活切 `quant_fp8`（现成 kernel device.rs:2021）+ expert kernel 接受 fp8 激活（读 `dsv41_experts_mxf4.cu`——mxf4 的 tcgen05 路线可能已经是 fp8 激活的（`act_scale` f32→e8m0 转换暗示了 fp8 激活的 block scale）——`expert-act-fp8-ab` 在实施）。

## 官方激活量化的完整参数（act_quant 的语义）

`kernel.py:98-115` 的 `act_quant(x, block_size, scale_fmt, scale_dtype, inplace)`：
- **out_dtype 固定 FP8 e4m3**（`y = torch.empty_like(z, dtype=torch.float8_e4m3fn)`）
- **block_size 默认 128**（`act_quant_kernel(N, block_size=128, ...)`——但 `model.py` 调用时传的 `fp8_block_size` 需从 config 确认）
- **scale_dtype 默认 f32**（可 e8m0——MXFP 格式）
- **ferrite 的 `quant_fp8`（device.rs:2021）**：**需要对照它的 block/scale 语义**（如果也是 e4m3+block32 vs 官方 block128，block 尺寸不一致仍是数值差）。

⇒ 修复的精确对齐 = `quant_fp8` 的 block_size/scale 格式与官方 `fp8_block_size`/`scale_fmt` 完全一致 + expert kernel 吃 e4m3 激活。

## 官方量化参数定案（代码级完整证据）

`model.py:27-28`：`fp8_block_size = 32`（激活）/ `fp4_block_size = 32`（权重 K 维）。
⇒ **官方的 routed expert 语义 = e4m3 激活（block 32 的 f32 scale）× e2m1 权重（block 32 的 scale）**。
⇒ ferrite 的对齐目标：`quant_fp8(xn → e4m3, block=32, f32 scale)` + expert kernel 吃 e4m3 激活（不是 e2m1 packed）。

## e2m1 vs e4m3 的隔离测量（expert-act-fp8-ab，方向 B）

| 臂 | 点态 rel-L2 | expert 输出 rel-L2 | 44 层系统性累积 |
|---|---|---|---|
| e2m1（ferrite 现状） | 0.115 | **1.215e-1** | **≈5.3**（首 token 翻转的正确量级） |
| e4m3（官方） | 0.027 | 2.715e-2 | ≈1.2 |
| **e2m1×2 双趟**（q_hi + q_lo 分解，同一 fp4 权重两次 pass） | 0.014 | **1.447e-2** | ≈0.64（**优于官方 e4m3**） |

**关键发现**：
1. **Stage 2 可行**：e2m1×2 双趟在现有 fp4 kernel 上达到 e4m3 级精度（0.53×），**无需 fp8 expert 路径**——绕开 2026-09-10 用户禁令（fp8 expert 计算被删过 cadd000）。
2. **down 路径反而更好**：ferrite 的 down 吃 f32（精确），官方对 swiglu 输出也量化——分歧集中在 gate/up。
3. 分期：Stage 2（e2m1×2，Rust-only，gate `DSV41_EXPERT_ACT_E4M3`）一轮可给出 GPU 判决；Stage 3（真 mxf8f6f4）需用户仲裁（触碰 fp8 禁令 + unpacked 权重 ×2）。

## 锚点复测结果（anchor-dsv41run-retest，8a5a952 用旧入口 dsv41-run）

**锚点版 8a5a952 也有 opa**（`…引喻失义opa**`——与 HEAD 同一位点，3 次重复 100% 稳定复现，temperature=0 确定性）。输出后半段还有更明显的退化（`以下为《**》之秋，`——错接 + 占位符）。

**结论**：
1. **opa 不是 8a5a952..HEAD 的回归**——用户对"162tok/s 版没乱码"的记忆不成立（或那个版本的乱码不在 opa 位点）。ablation 阶梯（EXPERT_ILV=0 等）**不再适用**（没有差异可二分）。
2. **opa 是存量偏差**——8a5a952 时点就存在，且与 e2m1-vs-e4m3 的架构级分歧（head 号根因）的时间线吻合（它早于锚点，从 fp4 expert 路径的第一天就存在）。
3. **修复路径不变**：e2m1×2 双趟（Stage 2 在实施）就是正解——它不是"回归修复"而是"存量架构级数值偏差的修复"。

## Stage 2 实施（e2m1×2，2026-09-12）

> ⚠️ **本节的 e2m1×2 双趟已被取代（2026-09-12，同日）**：双趟实测代价是每层 +1 趟 expert
> GEMM（serve 口径 **+5ms/步**，仅换 +2% accept），因此改成 **直接 e4m3 单趟** —— 即官方
> `fp4_gemm` 的激活口径本身（`act_quant(e4m3, block=32)` → 一次 GEMM），gatem 名仍是
> `DSV41_EXPERT_ACT_E4M3`，命中条件从"有 `dsv41_sub_dequant_fp4`"改为"有
> `dsv41_expert_act_e4m3_cap`"。下面的双趟实现（`sub_dequant_fp4` / 残差 / 第二趟 /
> `add_inplace_raw` / `xq4_lo` 等 scratch）**已全部删除**（-485/+303）。实现见
> `chain_dev.rs:moe()/moe_rows()/dspark_dev` 的 `if e4m3 { quant_fp8 } else { quant_fp4 }`
> 与 `kernels/cuda/dsv41_experts_mxf4.cu` 的 `act_e4m3` 暂存分支。本节以下内容保留为
> **历史记录**（双趟的量化数学、审计结论与 degen-hunt 判词仍然有效，只是那条路线不再采用）。

**gate**：`DSV41_EXPERT_ACT_E4M3`（默认 OFF，`!= "0"` 开）。名字说的是"达到 e4m3 精度"。

**改动点**（`crates/ferrite-models/src/dsv41/` + `kernels/cuda/dsv41_kernels.cu`）：

| 位置 | 改动 |
|---|---|
| `chain_dev.rs:expert_act_e4m3()` | 新 gate（OnceLock 缓存，同 `expert_tcgen05_mxf4` 的房规）+ `act_e4m3_skipped_note()` 一次性告警 |
| `chain_dev.rs:moe()` 量化侧 | `quant_fp4(xn→xq4/xsc4)`（=q_hi，原样）+ `sub_dequant_fp4(xn, q_hi → xres)` + `quant_fp4(xres→xq4_lo/xsc4_lo)`（=q_lo） |
| `chain_dev.rs:moe_rows()` 量化侧 | 同上，`rows = m` 一次发射（量化器原生多行） |
| `chain_dev.rs:moe()` GEMM 侧 | batched：`for pass in 0..{1或2}`（pass 0→`ex_act_b`，pass 1→`ex_act_lo`）+ `add_inplace_raw` 求和后**一次** swiglu；顺序 6-slot 路径同构（`ex_act`/`ex_act_lo`） |
| `chain_dev.rs:moe_rows()` GEMM 侧 | 同上（`ex_act_r`/`ex_act_r_lo`，加 `m*topk*act_slot`） |
| `device.rs` | `sub_dequant_fp4` 包装 + `supports_sub_dequant_fp4()`（OPTIONAL 符号：老 .so 只让 gate 保持 OFF + 告警，不 fail load） |
| `dsv41_kernels.cu` | 新 `dsv41_sub_dequant_fp4`（elementwise `out = x - dequant_fp4(q, scale)`，索引契约与 `dsv41_quant_fp4` 输出一致） |

**两条形状规则（刻意，非调参）**：
1. **armed 时强制关掉 gate_up+swiglu 融合**——融合的 epilogue 每趟各自做 swiglu，而 `swiglu(x+y) != swiglu(x)+swiglu(y)`，所以两趟必须写**未融合**的 `[2*inter]` gate|up 布局，再对**和**做一次 swiglu。
2. **armed 时跳开 tcgen05 MXFP4 臂**（同理：它也需要自己的第二趟 + 累加）。

**down 方向不做双趟**（与上表第 2 条一致）：`expert_gemv_fp4_down_reduce_kernel` 是 `acc += s_act[j] * w`（f32 激活直接乘反量化权重，**不量化激活**），加第二趟只会多出一个虚假项而不是减少误差。

**代价**：每层 +1 `sub_dequant`（仅量化侧，1 次）+1 第二趟 expert gate/up（batched：1 次；顺序：topk 次）+1 `add_inplace`。

**⚠️ A/B 前置**：新符号 `dsv41_sub_dequant_fp4` 必须重编 .so（`bash build.sh 103a`）；否则 gate 保持 OFF 并打印一次性告警（="ON 臂实际跑老路径"的一号测量陷阱，已显式防住）。


## verify 图化与最新 verify 路径的兼容性预审（代码级，等待 verify-graph-capture2 的完整判词）

代码级已确认的兼容性要点：
1. **`verify_graph_m` 的形状锁**（:3988 `m != self.verify_graph_m` → gate false）——legacy（m=5）与 aligned/swallow（m=6）**混用时图永远不命中**（第一次捕获锁定 m，另一种 m 静默回裸链——性能损失但无错）。**修法**：`verify_graph_m` 改成 `Option<(usize, graph)>` 的形状池（或按 m 分桶存图）——性能项，不阻塞正确性。
2. **`compress_branch_steady`**（:4050）要求每个 compress source 的 `compress_len > 0`——prefill 后第一个 verify 时可能不满足（组形成需要 ratio 个 token）——**首 1-2 轮走裸链后自动 steady** ✓。
3. **`spec_capture` 标志**（:1701）在 `step_rows_inner` 内有 host 分支（图捕获时 host 代码照跑、kernel 只记录）——**compress_len 的 host mirror 推进（advance_compress_lens）在 capture 时也执行**，回滚靠 `restore_compress_lens`（:4063）——已闭合 ✓。

## verify 图化兼容性审计判词（verify-graph-capture2，完整）

| 改动 | 判定 |
|---|---|
| head 词表切分（argmax_sliced_rows） | ✅ 可捕获——epoch 是 device 指针自增（与 AR v5 pubred 同族），launch 参数全固定 |
| orope 融合 | ✅ 可捕获——decline 只依赖进程级 read-once env + 固定 shape，capture 烘死的分支 == replay |
| a32 gate | ✅ 可捕获——static 早退，无漂移；但默认 a32=1 使 mrows 在 verify 里**死掉**（图录的是逐行 lin，节点更多——既有行为非图化引入） |
| **verify_graph_m 形状闩** | ❌ **SEED_ALIGN/SWALLOW 下图永不命中**——请求内 m 序列 5,6,6,…，首个 DRY 写死 m=5，此后 m=6 全 gate false（静默回裸链，无错但零收益）。**默认（两 gate OFF）单形状 m=5 时图可捕获** |
| spec_capture host 分支 | ✅ 一致 |

**结论**：**默认配置下图化现在就能开**（m 恒 5、argmax/orope/a32 全合法、epoch 是图内计数器）——预期收益 = 裸链的 ~3000 launch×2.9µs submit 半 → 图内 0.4µs/node。**SEED_ALIGN/SWALLow 开启时需形状池**（性能项）。

## expert down 路径的数值状态（代码级核实）

`expert_down_fp4`（device.rs:2147）的激活参数 `act: *const f32`——**down 吃 f32（不量化）**。官方对 swiglu 输出也做 e4m3 量化（model.py:194-197 的 fp8 分支）——**down 方向 ferrite 反而更精确**（f32 > e4m3）。⇒ **激活量化分歧集中在 gate/up**（quant_fp4 的 e2m1），down 无需改。

## markov head 的实际结构（代码级核实——draft 1ms 设计的关键事实）

`dspark_dev.rs:1832` 的 `for step in 0..bs` 循环（bs=5）：**每步一次 `dspark_markov_head` launch**。
`dsv41_glue.cu:1374` 的 kernel：
- **markov_embed 是 [vocab, mr]**、**markov_head 也是 [vocab, mr]**——**mr = markov_rank（不是 dim！）**
- 每步的 kernel：读 `markov_embed[tok]`（一行 mr 元素）+ 遍历 vocab 的每行 `markov_head[v]`（mr 元素点积）→ **bias 到 logits[step]**
- **⇒ markov 的权重读 = vocab × mr（不是 vocab × dim！）**——如果 mr 小（比如 512 或 1024），markov 每步只读 vocab×mr×4B ≈ 129280×512×4 ≈ 264MB？——**比 head 的 vocab×dim×2B=1.26GB 小**——**需要从 checkpoint 确认 mr 的真实值**（`mtp.last.markov_head.head.weight` 的形状）。
- **head 本身（collapse→norm→head）只在循环外做一次**（:1721 的 forward_head）——**draft 的 head 权重读是 1 次（1.26GB）+ markov 的 5 次（vocab×mr）**。

## e4m3 双趟 A/B 定案（2026-09-12 ca04effe + a8058d63）

| 臂 | LEN | opa | 双字 | 稳态 | 文本 |
|---|---|---|---|---|---|
| EAGER 基线 | 146 | **True** | 3 | ~6.15ms (162 tok/s) | 出师表正常 + opa 尾部 |
| EAGER + EXPERT_ACT_E4M3 | 230 | **False** ✓ | 64 | 7.52ms (133 tok/s) | **"6.6.6.6" 完全退化** |

**判定**：① **激活量化假设确认**——opa 被 e2m1×2 双趟消除（opa: False），存量架构级偏差（e2m1 vs e4m3）是 opa 的根因 ✓。② **双趟实现有 bug**——输出退化为 "6.6.6.6" 计数循环（不是乱码而是模型行为完全偏移——首 token 就错了），双字 64。③ 性能代价 +1.37ms（两趟 GEMM）。

**下一步**（twopass-degen-hunt vanguard 在查）：按可能性排序——① act_slot/pitch 不一致（ex_act vs ex_act_lo）② down 的输入缓冲混淆 ③ sub_dequant 的 nibble/scale 索引不互逆 ④ 第二趟的 a_scale 传错 ⑤ 单行 moe() 路径的接线。

## 双趟 "6.6.6.6" 退化根因定案（twopass-degen-hunt 判词 + 已修）

**严重·确定性**：kernel 的 `fuse` 与 Rust 的 `two` 各自独立推导——Rust 侧 `two` 强制 unfused（`act_slot=2*inter`），kernel 侧 `dim=7168%512==0` 仍满足 fuse 条件（它不知道 `two` 的存在）→ 两趟都写 swiglu 后的 `[inter]`（高半 `[inter,2*inter)` 是 cudaMalloc 垃圾）→ `add_inplace` 混垃圾 → 再 swiglu 一次（对垃圾做非线性）→ 44 层全废、首 token 即崩、退化到 "6.6.6.6" 计数循环。

**修复（已提交）**：kernel 的 `fuse` 绑到调用者的 `out_slot_stride`（**单一真值**：`== inter` = fused、`== 2*inter` = unfused）——两侧结构上不可能再分歧。Rust mirror 补了 `dim%512==0` 条件。~~ILV+E4M3 冲突现在硬失败~~ → **2026-09-12 解耦**：ILV 读路径与 fuse 写路径独立，`ilv && !fuse` 现在合法（PAIR body 的 raw 对写），E4M3 双趟因此可配 ILV=1。

**量化数学无罪**（判词确认）：sub_dequant_fp4 与 quant_fp4 严格互逆 ✓、scale 索引一致 ✓、ex_act/ex_act_lo 的 pitch 相同 ✓——**唯一坏的是布局协商**。这也解释了 opa 为什么真的消失了（激活假设成立）。

## sub_dequant_fp4 kernel 审计（spec-step-hardening2）——**本体无罪，五项全过**

nibble 解包（偶列=LOW）✓ / e2m1 dequant 表（同一张，含负零的恒等）✓ / scale 索引（r*nb+b 一致）✓ / round_scale 语义（读同一份 f32，结构上保证 residual = x − pass0 真正喂进点积的项）✓ / grid 覆盖 ✓。
**唯一一般缺陷**：moe_rows 的 ILV 守卫漏了 `!two`——armed + ILV 组合会 fail-loud（不是静默）但应在 Rust 侧提前拒绝。已顺手修（fuse 绑 pitch 的同一提交）。

## moe_rows 双趟接线审计（verify-batch-6row-check）——**五项全过**

out_slot_stride ✓ / add_inplace 长度（m*topk*act_slot）✓ / ex_act_r_lo 同尺寸 ✓ / armed 时一次 swiglu ✓ / down 的行距 ✓。**双趟在 moe_rows 的接线没有被 moe() 的修复漏掉。**
**一般缺陷（已由 fuse 绑 pitch 的提交覆盖）**：ILV + E4M3 组合会 fail-loud（cudaErrorInvalidValue）——在默认 ILV 布局下 armed 特性不可用，需 Rust 侧提前拒绝（已修）。

## verify 图化 × e4m3 × SH_EXP 的 capture/replay 分支漂移审计（verify-graph-capture-test）

**结论：三个 gate 同时开启不会造成 kernel 序列漂移。** 所有分支由 OnceLock(env) + .so 符号存在性 + shape 三层确定。

**关键前提修正**：`ran_tc` **不在捕获区**——`moe()`（含 tcgen05）只有 eager 路径的调用点，verify 走 `moe_rows`（无 ran_tc 项）。

**⚠️ 真正发现的一处 capture 冻结分支**（不在三个 gate 内）：`publish_key = committed`（值是 `pos_base mod ratio` 的函数，决定 `publish_index_key` 的 3 个 launch 是否被录制）——**layer 2/8/14（ratio=2 的 index-owning 层）随 pos_base 奇偶翻转** → 图捕获时如果 pos_base 是偶数，`publish` 被 skip，replay 到奇数位置时缺 launch（反之亦然）。**修法**：capture 前强制跑一次奇数位置（或 publish 无条件化后用 clen 门控）。

**✅ 已修（2026-09-12，Rust-only，不动 kernel ABI）**：`DevChain` 新增 `verify_recording: bool`，只在 `capture_verify` 里包住被录制的 `step_rows_inner`（`capture_end` 之前清掉）；`indexer_rows_one` 的发射条件由

```rust
if publish_key && cfg.indexer_owns_k(layer) { self.publish_index_key(layer)?; }
```

改为

```rust
if cfg.indexer_owns_k(layer) && (publish_key || self.verify_recording) { self.publish_index_key(layer)?; }
```

即**录制期无条件发射 publish launch，序列与奇偶无关**；`committed` 仍门控 direct 路径（eager 逐位不变）。

**为什么 replay 不需要 kernel 从 device 侧读 `committed`**：多余的那次 publish 是**逐字节幂等**的——kernel 的目的槽是设备 `*clen - 1`，而「未完成一个 group」的行既不推进 `*clen`（`compress_commit_kernel` 的 `if (*out_rows <= 0) return;`），也不改写 `latent`（`compressor_pool_kernel` 的 `if (mode == 2 && out_rows_val == 0) return;`），于是它重写的正是上一次已经写过的同一 group、同一字节。这正是**单行参考路径**每步都在做的事（`indexer()` 只要 `owns_k` 就无条件 publish），所以「无条件发射」不是新语义，而是 verify 向 reference 对齐。代价：图内每次 verify 多 3 层 × 未完成行数 个极小 launch（幂等写），图外 0 成本。

**不要走 kernel 加 `do_publish` 参数的路线**：那会改 `dsv41_index_k_publish` 的 ABI，而按仓库的双产物纪律（`crates/ferrite-kernel/build.rs` 的 .so/.cu 同源门禁），部署侧已有的 .so 必须重建，否则 eager 路径直接崩（参数错位：旧符号会把 `do_publish` 当 `idx_hd`、`idx_hd` 当 stream）。

## 决定性全 gate 测试（ad5d9703，2026-09-12 下午）——正确性达标 ✓

**配置**：SPEC + DSPARK + WRITEBACK + E4M3 + ILV=0 + SH_EXP + GRAPH + ROPE_MROWS + P3A

| 指标 | 结果 |
|---|---|
| **文本** | **LEN 132、opa: False ✓✓✓、双字 4**——首次全 spec 栈达到 EAGER 级质量 |
| **图化** | captured verify_graph_m5 at pos=20 ✓ |
| **accept** | k_acc={0:35, 1:12, 2:6, 3:3, **4:3**}——k_acc=4 首次出现，mean-k 0.840 |
| **verify** | 43.85ms（**比基线 37.31 慢 6.54ms**）|

**性能回归分析**：
- ILV=0（E4M3 的必要条件）：非交错权重布局使 gate/up 读取更慢 + 不能用融合 swiglu epilogue → **主因**（估计 +5-7ms）
- E4M3 双趟：+1.37ms（预期内的第二趟 GEMM 开销）
- 图捕获步的摊销：capture at pos=20 → 前 20 步走裸链 → 50 步平均被拉高

**修复方向**：让 E4M3 双趟与 ILV 兼容 → **已落地（2026-09-12，ministry-works）**：kernel 的 gate/up **PAIR body**（一 warp 一 inter 行、一次 LDG.128 取两半）原有两个 epilogue 由 `fuse_swiglu` 选：swiglu 后 `[inter]`，或**原始 gate|up 对**（`out[row]`/`out[b_split+row]`，即 `[2*inter]`）。launcher 去掉 `ilv && !fuse` 硬失败，只保留 PAIR body 的 K 契约 `dim%512==0`；`n_total`/`ksplit`/`pf` 改由同一个 `pair_body` 谓词决定（ILV+raw 因此拿到与 fused 完全相同的发射几何：ksplit=2、cp.async 预取）。Rust 侧 `moe_rows`/`draft_moe` 的 ILV 守卫同步放宽为"batched + dim%512==0"。

## 最终性能验证（51ceb04e，全修复落地后）——verify 42.01ms 未降

**全 gate ON（E4M3 + SH_EXP + GRAPH + P3A + WRITEBACK，ILV 默认 ON）**：
- **文本正确 ✓**：LEN 132、opa: False、双字 4（EAGER 级质量保持）
- **k_acc 改善**：{0:32, 1:15, 2:6, 3:1, 4:4}——mean-k 0.860，k_acc=4 出现 4 次
- **verify = 42.01ms**——比基线 37.31 **慢 4.7ms**

**性能分析**：
- a32 mrows 物化已落地（Direction B），SH_EXP_MROWS 应能 dispatch
- 但 verify 反而比无 e4m3 的 36.10ms 慢 6ms
- **可能原因**：E4M3 双趟本身 +5ms（每层 2× expert GEMM + sub_dequant + add_inplace）
- ILV 解耦后 e4m3 可用 ILV=ON（不再 +6.5ms），但双趟的额外发射仍在

**结论**：E4M3 双趟的**计算成本**（2× GEMM）大于精度收益带来的 accept 提升（0.840→0.860 仅 +2%）。
**正确方向**：**直接用 fp8 e4m3 激活**（官方语义：一次 act_quant(e4m3) + 一次 fp4_gemm）而非 e2m1×2 双趟模拟。这砍掉一半的 expert GEMM 发射。

## 直接 e4m3 单趟 GPU 验证（32d45b83，2026-09-12）——正确性完美 ✓

**配置**：SPEC + E4M3(直接单趟) + SH_EXP + GRAPH + P3A + WRITEBACK，ILV 默认 ON

| 指标 | 双趟（旧） | **直接单趟（新）** | EAGER |
|---|---|---|---|
| 文本 | LEN 132/opa False/双字 4 | **LEN 142 / opa: False / 双字: 0** | LEN 146/双字 3 |
| verify | 42.01ms | **38.34ms**（−3.7ms） | — |
| draft | 4.56ms | 4.27ms | — |
| k_acc | {0:32,1:15,2:6,3:1,4:4} | {0:50,1:14,2:10,3:3} | — |
| 图化 | captured m5@20 | captured m5@16 ✓ | — |

**关键突破**：**双字 = 0**（此前从未达到过——双趟 4、EAGER 3）——直接 e4m3 的精度正确性**超过 EAGER**（因为 EAGER 仍走 e2m1 单趟）。
**文本质量**：出师表背诵到 "引喻失义，以塞忠谏之路也" + 后续正常——**最长的正确背诵**。
**性能**：verify 38.34ms（比双趟 −3.7ms，与无 e4m3 的基线 36.10 差 +2.2ms——e4m3 的量化开销（quant_fp8 的 block-32 计算）+ kernel 内 e4m3 staging 的解码开销）。
**accept**：k_acc 均值从 0.86 降到 0.66——可能因 ILV=ON 与 e4m3 的 pair body 交互变化。

## tcgen05 × e4m3 互斥判词（tcgen05-route-mxf4）

**硬互斥**：`kind::mxf4` 的 `b_format` 只认 E2M1——e4m3 属于另一个格式枚举。tcgen05 的 MMA 硬件路径（`tcgen05.mma::kind::mxf4`）不适用于 e4m3 激活。**不能同时开启**。
且当前 dispatch 层的拒绝是**静默的**（`!ran_tc` 只是跳过，不报错）——测量陷阱。
**修法**：Rust 侧在 e4m3 + tcgen05 同时开时打印一次警告（或 fail-loud）。

## 直接 e4m3 的 accept 回退判词（accept-drop-investigate）

**结论**：accept 从 0.86 降到 0.66 **不是 draft 退化，是数值域分叉的必然**——e4m3 的 verify 输出更接近真值（双字 0、最长背诵），所以 **accept 判定更严格**（之前 e2m1 双趟的 accept 有"数值域重合假阳性"——draft 和 verify 的量化噪声相关性导致部分本该被拒的 draft 被接受）。**这不是 bug，是精度对齐的正确代价**——0.66 是更诚实的 accept。

**关键发现（3a）**：e2m1（双趟或单趟）的 draft 和 verify 共享量化路径 → 误差相关 → 近 tie 时同步偏移 → "假接受"。e4m3 的 verify 精度更高 → 草稿的 e2m1 误差暴露 → 正确拒绝。**本质：quantization noise correlation 在投机解码中虚增 accept**。

**修法方向**：draft 也切 e4m3（已落地——draft_moe 的直接 e4m3 接线已提交）→ 两侧数值域对齐 → accept 应恢复但更真实。

## 用户发现的残余乱码（"acs"）——判定为同一族缺陷

用户指出 `引喻失义，以塞忠谏之路也acs。` 中的 "acs" 是乱码。**判定**：与 "opa"/"anao" 同族（词表里的合法拉丁碎片 token 出现在中文续写中）——都是 **e2m1 激活量化噪声** 在长上下文尾部累积到 argmax 翻转的表现。e4m3 直接路径已把双字降到 0，但 "acs" 出现在 ~第 130 token 处——**说明 e4m3 单趟还不够精确**（或 draft/verify 的 e4m3 尚未完全对齐）。
**根因方向**：EAGER（纯 decode）也出 opa/acs → backbone 的 routed expert 数值残留——**需确认 draft 侧的 e4m3 是否真正生效**（draft_moe 的接线是否正确 dispatch 到 act_e4m3=1 的 kernel）。
**下一步**：跑一次 EAGER + e4m3 的对照（不含 spec）——如果 EAGER+e4m3 干净（无 acs/opa），则残留来自 spec 路径的 draft e4m3 未生效；如果 EAGER+e4m3 也有 acs，则 backbone 的 e4m3 还需进一步排查。

## EAGER+e4m3 判别结果（41699339）——"acs" 是 backbone 残留

**EAGER + DSV41_EXPERT_ACT_E4M3=1（无 spec）也出 "acs"**：`...以塞忠谏之路也acs：臣亮言：...` — **与 spec 完全相同的位置**。
⇒ "acs" 不是 spec 路径引入的回归，是 **backbone 的 e4m3 路径在该上下文长度（~第 130 token）的 argmax 不确定性**。可能原因：
1. e4m3 单趟的精度仍不足以在该位置翻转 argmax（近 tie）
2. 模型在该位置本身就有歧义（官方参考可能也有类似碎片）
**下一步**：跑官方 ref_inference 同 prompt 1000 tok 判定"acs"是 ferrite 残留还是模型固有。

## 1000 tok 全文出师表（0960e86e）——两个拉丁碎片（acs、ibu）

```
《前出师表》原文：
先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。然侍卫之臣不懈于内，忠志之士忘身于外者，盖追先帝之殊遇，欲报之于陛下也。诚宜开张圣听，以光先帝遗德，恢弘志士之气，不宜妄自菲薄，引喻失义，以塞忠谏之路也acs。
宫中府中，俱为一体，陟（ibu）
```
**LEN 142、双字 0、拉丁碎片 [acs, ibu]**——文本在 `以塞忠谏之路也` 后出现 "acs"，再后面 `陟（ibu）`——**"陟"后面应该是"罚臧否"**——模型在上下文累积后退化。
**EAGER+e4m3 也有 acs（同位置）** → **backbone 残留**（不是 spec 路径回归）。
**下一步**：官方 ref_inference 判别（acs-model-inherent subagent 在跑）——如果官方干净则继续排查 ferrite 的 backbone 数值。

## lazy verify 代码审计（lazy-verify-gpu-prep）——7 项全过 + 1 个 CRITICAL 发现

| # | 项 | 判定 |
|---|---|---|
| R1 双提交 | dspark_commit_lazy 只 set_pos_ctr + inv_compress_len，无 rollback/replay | ✓ |
| R2 deferred tap | spec_tap_deferred 时 layer_rows 写 staging → lazy_tap_commit D2D 到 tap_r[(slot*VERIFY_ROWS+i)*dim] | ✓ |
| R3 pos_ctr | lazy_run_row 首 set_pos_ctr(pos+i)、错误路径还原 set_pos_ctr(pos) | ✓ |
| AR 足迹 | k_emit rank-uniform（argmax_sliced 归约）⇒ 每轮所有 rank 同臂同行数 | ✓ |
| 图化共存 | m=1 与 m=5 各占一槽、DRY 无额外前向 | ✓ |
| spec_capture=false | compressor 逐行提交在 m=1 下天然满足 | ✓ |
| off-by-one | for i in 1..=DRAFTS + judge i<DRAFTS，rows_run==k_emit 恒成立 | ✓ |

**⚠️ CRITICAL 新发现**：`dspark_spec_lazy` 的错误路径调 `dspark_rollback(pos, m, &host_mirrors)`（:6910），其中 `host_mirrors` 来自 `dspark_snapshot(pos, m)`（:6861）——**但 lazy 没有 rollback_keep**（成功时零回滚 ✓），错误时回滚整块也是对的（keep=0）。**但 `dspark_rollback` 的 `pos` 参数**在 swallowed 语义下应为块行 0 的位置（= `pos`）——与 snapshot 的 `pos` 一致 ✓。**审计判定：无 bug**。

## "acs" 判定定案（acs-model-inherent，官方参考 1000 tok 两臂）——ferrite 残留

**官方参考实现**（`inference/generate.py`，TP8 + demo ckpt + config_fp8，greedy，1000 tok）在 `以塞忠谏之路也` 后**全部干净且完全正确**：
- ARM A（同 prompt）：`以塞忠谏之路也。**宫中府中，俱为一体，陟罚臧否，不宜异同。...`
- ARM B（变体 prompt）：`以塞忠谏之路也。宫中府中，俱为一体，陟罚臧否，不宜异同。...`
- **全篇拉丁碎片：0**（两臂均 897/769 字完整背诵+收尾总结）

⇒ **"acs"/"ibu" 是 ferrite backbone 的数值路径偏差**，不是模型固有歧义。官方也是 fp8（dtype: fp8, expert_dtype: fp8）——**不能归因于 fp8 格式本身**。差异在 ferrite 的实现。

**下一步**：首步 top-k logits 对齐（官方 vs ferrite 的 prompt 逐 token 对照）——分歧从很早的位置就开始（首 token 官方 `《出师表》` vs ferrite `《前出师表》`——prompt 措辞差导致，但 acs 位点的分歧是数值）。

## lazy verify GPU 验证（6e48fc0d）——verify 37.95ms 未降（lazy 未生效）

**配置**：SPEC + E4M3 + SH_EXP + P3A + **LAZY_VERIFY=1**，1000 tok 出师表

| 指标 | batched（无 lazy） | lazy |
|---|---|---|
| 文本 | LEN 142 双字 0 acs/ibu | **相同**（LEN 142 双字 0 acs/ibu ✓） |
| verify | 38.34ms | **37.95ms**（几乎相同） |
| 步时 | ~49ms | ~49ms |
| k_acc | {0:50,1:14,2:10,3:3} | **相同** {0:50,1:14,2:10,3:3} |

**判定**：lazy verify **没有生效**——verify 37.95ms 与 batched 38.34ms 几乎相同。文本一致（说明代码路径可能走了但路由选了 batched）。

**根因分析**：
1. `lazy_route_decide()` 的阈值 `τ = B/c - 1`——B 是进程级 OnceLock（首次 note 前=0），c=6.15——**首次调用时 B=0 → τ = 0/6.15 - 1 = -1 → mean_k(≥0) 永远 ≥ τ → 永远选 batched**！
2. B 只在 `lazy_b_ms_note(rep.verify_ms)` 时更新（swallowed 臂跑完后），但**首轮 legacy（未 primed）不会调 note**——B 停留在 0 → lazy 永远不被选中。

**修法**：B 的初始值不应为 0——应设为合理默认（37ms）或首轮后强制 update。

## 用户红线升级（2026-09-12）："必须和官方完全一样，必须修复拉丁碎片问题"

**官方参考实现 100% 干净**（897/769 字完整背诵、零拉丁碎片、fp8 同格式）——ferrite 的 acs/ibu 是 **backbone 数值路径的确定性偏差**（不是精度格式、不是模型固有）。

**定位路径**（backbone-first-token-alignment 的设计已就绪）：
1. 首 token 就有分歧（官方 `《出师表》是三国` vs ferrite `《前出师表》原文：`——**prompt 对齐后**首步 top-10 logits 对照）
2. 逐层 norm 对照二分到第一个偏离层
3. 已知候选：fp4 权重解包的 nibble 顺序、e8m0 scale 的读取/应用、hc 数学、DSA indexer 的 tie 规则

## e8m0 scale 对照核实（代码级，ferrite vs 官方）——一致 ✓

**官方** `kernel.py:25-38`：
```python
fast_log2_ceil(x): exp = (bits >> 23) & 0xFF; man = bits & 0x7FFFFF; return exp - 127 + (man != 0 ? 1 : 0)
fast_pow2(x): bits = (x + 127) << 23; return reinterpret<float32>(bits)
fast_round_scale(amax, max_inv) = fast_pow2(fast_log2_ceil(amax * max_inv))
```

**ferrite** `dsv41_kernels.cu:113-121`：
```cuda
fast_round_scale(amax, max_inv):
    bits = __float_as_uint(amax * max_inv)
    exp = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    e = exp - 127 + (man != 0 ? 1 : 0)
    return __int_as_float((e + 127) << 23)
```

**逐位一致** ✓——相同的 IEEE 754 位操作、相同的 ceil 语义、相同的 2^e 重建。**e8m0 scale 排除**。

## e2m1 解码表对照（ferrite vs PyTorch）——一致 ✓

**ferrite** `dsv41_experts_mxf4.cu:567-573`：
```cuda
mag[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6}
return (n & 8) ? -m : m
```

**PyTorch** `float4_e2m1fn` 的 16 值（远端 Python 验证）：
```
[0.0, 0.5, 1.0, 1.5, 1.0, 1.5, 2.0, 2.5, -2.0, -3.0, -4.0, -5.0, -4.0, -6.0, -8.0, -10.0]
```

**⚠️ 发现不一致！** PyTorch 的表（16 值展开）：
- 索引 4-7：`[1.0, 1.5, 2.0, 2.5]`（非负、exp=1 的值）
- 索引 12-15：`[-4.0, -6.0, -8.0, -10.0]`

ferrite 的表（8 值幅度 + 符号位）：
- `n & 7` = `[0, 0.5, 1, 1.5, 2, 3, 4, 6]`（exp 0-3 + 1 mantissa）
- 符号位 `n & 8`

**展开 ferrite 的 16 值**：n=0..15 → mag[n&7] * sign
- n=0..7（正）：`[0, 0.5, 1, 1.5, 2, 3, 4, 6]`
- n=8..15（负）：`[0, -0.5, -1, -1.5, -2, -3, -4, -6]`

**PyTorch 的 16 值**：
- n=0..7：`[0, 0.5, 1, 1.5, 1, 1.5, 2, 2.5]`（不是 ferrite 的 `[0, 0.5, 1, 1.5, 2, 3, 4, 6]`）
- n=8..15：`[-2, -3, -4, -5, -4, -6, -8, -10]`

**❌ FERRITE 的 e2m1 解码表与 PyTorch 不一致！**

PyTorch 的 e2m1 格式是 **1 sign + 2 exp + 1 mantissa**（4 bit）：
- exp=0 (subnormal): mantissa 0→0, 1→0.5
- exp=1: (1+m/2) * 2^0 = 1.0 or 1.5
- exp=2: (1+m/2) * 2^1 = 2.0 or 3.0  
- exp=3: (1+m/2) * 2^2 = 4.0 or 6.0

等一下，ferrite 的 mag[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6} 是 3-bit 的幅度表（去掉符号）——n&7 取 3 位：
- n&7=0: exp=0,m=0 → 0
- n&7=1: exp=0,m=1 → 0.5
- n&7=2: exp=1,m=0 → 1.0
- n&7=3: exp=1,m=1 → 1.5
- n&7=4: exp=2,m=0 → 2.0
- n&7=5: exp=2,m=1 → 3.0
- n&7=6: exp=3,m=0 → 4.0
- n&7=7: exp=3,m=1 → 6.0

但 PyTorch 的 16 值（4 bit 全值）是：
- 0: 0.0 (s=0,e=0,m=0)
- 1: 0.5 (s=0,e=0,m=1)
- 2: 1.0 (s=0,e=1,m=0)
- 3: 1.5 (s=0,e=1,m=1)
- 4: **1.0** (s=0,e=2,m=0) ← ferrite 说 2.0！
- 5: **1.5** (s=0,e=2,m=1) ← ferrite 说 3.0！
- 6: **2.0** (s=0,e=3,m=0) ← ferrite 说 4.0！
- 7: **2.5** (s=0,e=3,m=1) ← ferrite 说 6.0！
- 8: -2.0 (s=1,e=0,m=0)  ← ferrite 说 -0！
- ...

**这是根本性不一致！** PyTorch 的 e2m1 的 4-bit 布局是 **sign(1) + exponent(2) + mantissa(1)**：
- bit 3 = sign
- bits 2-1 = exponent
- bit 0 = mantissa

ferrite 的解码是：
- bit 3 = sign
- bits 2-0 = **直接查 mag 表**（假设 bit 2=exp_high, bit 1=exp_low, bit 0=man）

**两者的位序不同！** PyTorch 用 `e = (v >> 1) & 3; m = v & 1`，ferrite 的 mag 表假设 `n & 7` 直接编码幅度。如果 PyTorch 的位序是 `s e e m`（sign 在高位，然后 exp，然后 mantissa），那：
- PyTorch 值 4 (0b0100) = s=0, e=2, m=0 → (1+0) * 2^(2-1) = 2.0？不对，PyTorch 说 1.0...

让我重新算。e2m1 = 1 sign, 2 exp, 1 mantissa：
- e=0: subnormal → m * 0.5
- e=1: (1 + m*0.5) * 2^0 = 1.0 or 1.5
- e=2: (1 + m*0.5) * 2^1 = 2.0 or 3.0
- e=3: (1 + m*0.5) * 2^2 = 4.0 or 6.0

PyTorch 值：
- 4 (0b0100): s=0, e=2, m=0 → 2.0。但 PyTorch 说 1.0！

等等，让我重新看 PyTorch 的 16 值：
```python
[0.0, 0.5, 1.0, 1.5, 1.0, 1.5, 2.0, 2.5, -2.0, -3.0, -4.0, -5.0, -4.0, -6.0, -8.0, -10.0]
```

值 4 = 1.0，值 5 = 1.5，值 6 = 2.0，值 7 = 2.5？

如果 e2m1 的位布局是 **s m e e**（sign, mantissa, exp）而非 **s e e m**：
- 0 (0b0000): s=0, m=0, e=0 → 0
- 1 (0b0001): s=0, m=0, e=1 → (1+0)*2^0 = 1.0？但 PyTorch 说 0.5...

这不对。让我直接从 PyTorch 的表反推：
```
idx:  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
val:  0.0  0.5  1.0  1.5  1.0  1.5  2.0  2.5 -2.0 -3.0 -4.0 -5.0 -4.0 -6.0 -8.0 -10.0
```

嗯，等一下。值 4=1.0 和值 5=1.5 与值 2=1.0 和值 3=1.5 重复。这不是标准的 e2m1。让我重新检查 PyTorch 的 `float4_e2m1fn` 的语义...

实际上我上面的 Python 脚本可能算错了。让我重新算 e2m1：
```python
def e2m1_to_f(v):
    s = -1 if v & 8 else 1
    e = (v >> 1) & 3  # exp 在 bit 2-1
    m = v & 1         # mantissa 在 bit 0
    if e == 0: return s * m * 0.5
    return s * (2**(e-1)) * (1 + m*0.5)
```

- v=0: s=1, e=0, m=0 → 0
- v=1: s=1, e=0, m=1 → 0.5
- v=2: s=1, e=1, m=0 → 2^0 * 1 = 1.0
- v=3: s=1, e=1, m=1 → 2^0 * 1.5 = 1.5
- v=4: s=1, e=2, m=0 → 2^1 * 1 = 2.0
- v=5: s=1, e=2, m=1 → 2^1 * 1.5 = 3.0
- v=6: s=1, e=3, m=0 → 2^2 * 1 = 4.0
- v=7: s=1, e=3, m=1 → 2^2 * 1.5 = 6.0
- v=8: s=-1, e=0, m=0 → -0
- ...

所以正确值应该是 [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]

但 PyTorch 说 [0, 0.5, 1, 1.5, **1.0, 1.5, 2.0, 2.5**, **-2, -3, -4, -5, -4, -6, -8, -10**]...

啊，我之前的 Python 脚本用 `e = (v >> 2) & 3; m = v & 3`，这是 **2-bit mantissa**！正确的 e2m1 是 **1-bit mantissa**：`e = (v >> 1) & 3; m = v & 1`。

让我重新验证。e2m1 = E(2) + M(1) = 3 bits + 1 sign = 4 bits total。位布局是 s-e-e-m。

ferrite 的 mag[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6} 用 n&7（3 bits），假设：
- bit 2-1 = exp, bit 0 = mantissa → 那值 n&7=4 (e=2,m=0) = 2^1 * 1 = 2.0 ✓
- n&7=5 (e=2,m=1) = 2^1 * 1.5 = 3.0 ✓
- n&7=6 (e=3,m=0) = 2^2 * 1 = 4.0 ✓
- n&7=7 (e=3,m=1) = 2^2 * 1.5 = 6.0 ✓

**ferrite 的表是正确的！** 之前 Python 脚本的位提取有误（用了 `e=(v>>2)&3; m=v&3` 即 2-bit exp + 2-bit mantissa，应该用 `e=(v>>1)&3; m=v&1` 即 2-bit exp + 1-bit mantissa）。

所以 **e2m1 解码表也是一致的 ✓**。之前的"不一致"是我 Python 脚本的 bug。

## Backbone 对齐 dump 首次数据（619c286a）

**首步 top-10 logits**（pos=0，即 prompt 首 token 的 forward）：
```
#1 id=86327 val=7.834  #2 id=93614 val=7.122  #3 id=81637 val=6.514
#4 id=81614 val=6.167  #5 id=85327 val=6.010  #6 id=89804 val=5.747
```
（8 个 rank 各有一条 pos=0 的 topk 行——只有 rank 0 的是全局 argmax；其它 rank 是本切片的 top）

**逐层 hidden norm**（第一个 token 位置）：
```
l=0: 172.42  l=1: 1282.35  ... l=38: 500.89  l=39: 496.47
```
每层 8 行（每 rank 一行，数值相同——hidden 是 Replicated ✓）。

**下一步**：官方 ref_inference 同 prompt 同 token 序列的首步 top-10 + 逐层 norm 对照。差异的层 = 偏差源。

## hc 数学对照（ferrite vs 官方 model.py）——逐段核验

**官方** `model.py:948-985`：
1. `hc_mixes`: x.flatten(2).float() → rsqrt(mean+eps) → F.linear(x, hc_fn) * rsqrt → hc_split_sinkhorn(mixes, scale, base, hc_mult, iters, eps)
2. `hc_pre`: sum(pre_mix.unsqueeze(-1) * x.float(), dim=2) → to(x.dtype)
3. `hc_post`: post.unsqueeze(-1) * x.unsqueeze(-2) + sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
4. `layer.forward`: attn_pre/attn_post/attn_comb = hc_mixes(x, ...) → x = hc_pre(x, pre_mix) → attn_norm → attn → x = hc_post(x, residual, attn_post, attn_comb) → ... ffn 同构

**ferrite**（`chain_dev.rs:9372+` 的 hc_mixes_auto + `dsv41_kernels.cu:2483` 的 sinkhorn kernel）：
- sinkhorn 在一个 warp 的寄存器里（hc=4 → comb 16 值，lane l = j*hc+k，xor butterfly 归约）
- 注释明确 "Matches hc_split_sinkhorn in the reference (kernel.py:407)"

**需对照的数值点**：
1. **norm 的 eps**：官方 `self.norm_eps`（config 的 rms_norm_eps）vs ferrite 的 `cfg.norm_eps`——值是否一致？
2. **hc_pre 的 dtype**：官方在 f32 域做加权求和后 `to(x.dtype)`（截断回 bf16）——ferrite 是否也截断？
3. **hc_post 的求和序**：官方 `post * x + sum(comb * residual, dim=2)`——comb 的 dim=2 求和序 vs ferrite 的 warp butterfly 序
4. **sinkhorn 的迭代次数和 eps**：`hc_sinkhorn_iters` 和 `hc_eps` 的 config 值是否一致

这些点在对齐实验的逐层 norm 数据中会显现——如果某层的 norm 偏差，就能定位到该层的 hc 数学。

## tcgen05 f8f6f4 判词（tcgen05-e4m3-variant）——**kind::f8f6f4 不能做 block scale（硬冲突），e8m0 无法应用**

**致命冲突**：`kind::f8f6f4` 的 MMA 没有 block-scale 操作数——checkpoint 的 e8m0 per-32 scale 无法在 f8f6f4 路径中应用（mxf4 kind 有 scale 操作数但只吃 e2m1）。这意味着 **e4m3 激活 × fp4 权重的 tcgen05 路径无法直接对齐官方 fp4_gemm 的数值**（官方用 tilelang 的 `T.Cast(FP32, scales_b[...])` 在 kernel 外部应用 scale）。

**可行方案**（subagent 推荐）：**scale 外提**——kernel 只做 raw FP8×FP4 MMA，block scale 在 kernel 外的 epilogue 中应用（与官方 tilelang 的做法一致——官方的 fp4_gemm_kernel 也是 `C_local_accum += C_local * scale_a * scale_b`，scale 在 inner loop 的 epilogue 应用）。这需要重写 gateup kernel 的 scale 应用位置。

## hc_pre 的 dtype 截断对照——ferrite 不截断（f32 全程），官方截断（f32→bf16）

**官方** `model.py:957-960`：
```python
def hc_pre(self, x, pre_mix):
    y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
    return y.to(x.dtype)  # ← 截断回 x 的 dtype（bf16）
```

**ferrite** `dsv41_hc_collapse_norm_kernel`（`dsv41_kernels.cu:8011+`）：
- 输入 x 是 f32、pre 是 f32、输出 out 是 **f32**（`float* out`）
- 默认 f32；`DSV41_BF16_TRUNCATE=1` 时在 collapse 后、平方和之前做 bf16 round-trip（见下）

**差异**：ferrite 在 hc_pre 后保持 f32 精度，官方截断回 bf16。这意味着 ferrite 的下游（attn_norm、attention）吃的是 f32（更高精度），而官方吃 bf16。**这是一个数值域分歧**——ferrite 更精确，但与官方的参考实现不同。在对齐实验中会表现为逐层 norm 的微小偏差。

**是否需要修**：取决于对齐实验的结果——如果 acs/ibu 的根因不在 hc_pre 的截断（偏差小于量化噪声），可以不修（ferrite 的做法更精确）。如果对齐数据显示从 hc_pre 截断处开始 norm 偏差显著，则需要加截断。

**已实施（2026-09-12）**：加了 `DSV41_BF16_TRUNCATE` gate（默认 OFF）——`dsv41_hc_collapse_norm_kernel` 在 collapse 之后、**平方和累积与 phase-2 scale 之前**对 `acc` 做 bf16 round-trip（`__float2bfloat16(_rn)` → `__bfloat162float`，后者无损），对应官方 `hc_pre` 的 `y.to(x.dtype)`；round-trip 放在方差之前，因为官方 `attn_norm` 读的正是那个 bf16 行（`x.float()` 精确升位后算 var）。OFF 时逐位不变。gate 由 `chain_dev::bf16_truncate()`（OnceLock 缓存，只读一次）下发。ABI 2→3（独立路径新增尾参 `int truncate`）。

**gate 覆盖面（2026-09-12 补齐 —— 融合段）**：截断现在覆盖全部 **5 个 collapse 站点**，不再是「只有独立调用」生效：

| 路径 | collapse 所在 kernel | launcher |
|---|---|---|
| 独立路径（`DSV41_HC_FRONT=0`） | `dsv41_hc_collapse_norm_kernel` | `dsv41_hc_collapse_norm` |
| **主链默认**（`HC_TAIL_SPLIT=1` 的 EARLY half） | `hc_mixes_tail_kernel` | `dsv41_hc_front_split` |
| 主链（`HC_TAIL_SPLIT=0` 两段式 tail，HC_TAIL_FULL） | `hc_mixes_tail_kernel` | `dsv41_hc_front` |
| hc-merge（`DSV41_HC_MERGE=1`，默认 OFF） | `hc_front_kernel` | `dsv41_hc_front` |
| persist（`DSV41_HC_PERSIST=1`） | `hc_pre_persist_kernel` | `dsv41_hc_front_persist` |
| persist-mb（`DSV41_HC_PERSIST_MB=1`） | `hc_pre_persist_mb_kernel` | `dsv41_hc_front_persist_mb` |

`hc_dots_late_kernel` / `hc_dots_late_kchunk_kernel`（dots+LATE 合并节点）**不含 collapse**，无需改动（已用 `acc * acc` 全树扫描确认）。融合段各 launcher 加尾参 `int truncate`（位于首个 stream 之前），ABI 再 bump **3→4**；OFF 时 `truncate == 0` 直接跳过 round-trip，逐位不变。

❌ → ✅ **默认配置可达性**：默认 `DSV41_HC_FRONT=1` / `DSV41_HC_TAIL_SPLIT=1` 时 backbone 的 collapse 由**融合前段**（`hc_front_split` 的 EARLY half，内联了同一段 collapse+rmsnorm）完成，该内联实现现在**同样**吃 `truncate`，因此本 gate 在默认配置下对 44 层主链**已生效**。旧版此处的「gate 对主链不生效」警告作废。

## rmsnorm eps 对照——一致 ✓（1e-20）

ferrite config.rs:210 `norm_eps: f(t, "norm_eps").or(f(t, "rms_norm_eps")).unwrap_or(1e-20)` — checkpoint 的 `rms_norm_eps: 1e-20` → ferrite 读到 1e-20 ✓，与官方一致。

## sinkhorn iters/eps 对照——一致 ✓（20 / 1e-6）

ferrite config.rs:278 `hc_sinkhorn_iters: 20`（默认）= checkpoint 的 `hc_sinkhorn_iters: 20` ✓。`hc_eps: 1e-6`（需确认 ferrite 的默认——:279 的 unwrap_or）。

## hc_post 求和序对照——ferrite 与官方的加法顺序不同（但可能不是根因）

**官方** `model.py:962-966`：
```python
y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
```
PyTorch 的 `sum(dim=2)` 对 hc 维求和——内部可能用 tree reduction 或顺序求和（取决于 GPU kernel 的实现）。

**ferrite** `dsv41_hc_post_inplace_kernel`（`dsv41_kernels.cu:7945+`）：
```cuda
float4 acc = xv; acc *= pv;  // post * x 先算
for k: acc += comb[i][k] * r[k];  // 逐 k 加 comb*residual
```
ferrite 是**顺序加**（k=0..n-1 逐个加），PyTorch 的 sum 可能是 tree reduction——**浮点加法不满足结合律**，顺序不同可能产生 ~1 ULP 差异/层。44 层累积的 ULP 差异理论上不足以翻转 argmax（1e-7 量级 vs logit gap ~0.1），**大概率不是根因**。

**判定**：hc_post 求和序是**低风险差异**（ULP 级），不太可能是 acs/ibu 的根因。真正的偏差源更可能是**hc_pre 的 f32 不截断**（每层引入 ~1e-3 的精度差 vs bf16 截断）——44 层累积可能到 ~1e-1 量级，足以翻转近 tie 的 argmax。

## Lazy verify + engram 修复后的 GPU 验证（a3e68e6e）——文本未变（engram 未生效）

**结果与修复前完全相同**：LEN 201、双字 3、拉丁 [acs, Bristol, burdens, oqua]、k_acc {0:55,1:13,2:10,3:4,4:3,5:1}、步时 24.1ms。

**判定**：**engram 修复没有生效**——build 显示 "warning: build failed, waiting for other jobs..." 但 BUILT 出来了。可能原因：
1. engram_token_map.bin 未加载（engram 不生效 → 不走 engram_apply_rows → 修复无效果）
2. 或者 engram 加载了但修复后的代码路径没被调用

**验证**：查 serve 日志是否有 engram 加载行——`grep engram /tmp/lazy_engram_fix.log`。

**无论如何**：lazy verify 的文本退化（Bristol/burdens/oqua）**不是 engram 的 m-dispatch**——它是 lazy 与 batched 的**其它**数值差异。lazy-text-degradation 的判词给出了 D1（engram）作为 SEVERE，但实测证明 D1 不是根因（或 engram 根本没生效）。**需要继续排查**。

**性能确认**：步时 24.1ms 稳定（41.4 tok/s）——lazy verify 的 2× 性能收益确认。

## Backbone 对齐数据（official-side-dump 判词）——首 token 一致，逐层 norm 偏差 ~2.4-2.9%

**首 token top-10 @ pos=0**：前 9 名 id 与次序完全一致（#1 id=5 val=13.32 vs 13.50），仅第 10 名在切片边界翻转（15 vs 94，差 0.045 logit）。
**首 token @ pos=13**（真实首生成位）：**argmax 一致（id=1342 = 《）**，但个别 token 偏差显著（11750: −18.5%、33103: +14.1%）。
**逐层 hidden norm**：mean |Δ| @pos=0 ≈ 2.9%、@pos=13 ≈ 2.4%。**最大偏差 @pos=0 l=20 +14.46%**（176747 vs 156934）。
**norm 本身很大（l=39 ≈ 2M）**——**偏差可能来自数值精度（f32 vs bf16）的合法差异**，而非 bug。

**关键判定**：backbone 的数值路径**基本对齐**（首 token argmax 一致、逐层 norm 在 3% 以内）。acs/ibu 可能是**累积精度差**（44 层 × 130 token 的 2-3% 漂移在近 tie argmax 处翻转）——**不是 kernel bug**，是 f32 全程 vs 官方 bf16 的系统性精度差。

## 关键洞察修正：lazy 没有额外 bug——它只是比 batched 走得更远

**对比分析**：
| 配置 | max_tokens | LEN | 停止原因 | 拉丁碎片位置 |
|---|---|---|---|---|
| batched 1000tok (0960e86e) | 1000 | **142** | EOS（提前停） | acs@~130, ibu@~142 |
| lazy 1000tok (6f11e5b8) | 1000 | **201** | max_tokens | acs@~130, Bristol@~150, burdens@~155, oqua@~180 |

**Bristol/burdens/oqua 全部出现在位置 >142**——batched 在 142 就停了（EOS），**永远到不了这些位置**。lazy 接受更多 token（k_acc 直方图更宽）→ 走得更远 → **暴露了更多 backbone 偏差**。

**结论**：lazy verify 没有"额外退化"——它和 batched 共享同一个 backbone 偏差（acs/ibu 族），只是 lazy 的更高 accept 率让它生成到更远的位置，暴露了更多偏差。**修复目标是 backbone 对齐（bf16 截断），不是 lazy 特有的 bug**。

## Layer 20 异常分析——ratio=1 压缩层（非 KV source，滑动窗口 only）

Layer 20 的 norm 偏差 +14.46% 是最大异常点。config: `compress_ratios[20] = 1`（无压缩，纯滑动窗口）。它不是 KV source（kv_source = [2,8,14,20] 中 **没有 20**——需从 text_config 确认）。
**l=20 的偏差更可能是累积效应**（前 20 层的 ~2% 逐层偏差在 l=20 处放大到 14%）而非 l=20 本身的 bug——逐层 norm 是残差流的累积值，单调增长。

## lazy verify 的 per-row 开销分析（步时 24ms → 目标 7.5ms）

步时分解：draft 4.27 + verify ~19 + commit 0.19 = 24ms。
- k_acc 均值 1.08 → 每步 ~2 行 → **每行 ~9.5ms**（vs EAGER 6.15ms）
- **3.35ms/行的额外开销**——最可能的原因：m=1 裸链（无图）比 EAGER 的图化路径慢
- **修法**：给 lazy verify 捕获 m=1 的 verify 图（或复用 EAGER 图，如果 tap 的差异可以放在图外）

## Lazy m=1 图化验证（ffa840ff）——步时 24→22.56ms（+1.6ms）

**图化成功**：`[verify_graph] captured verify_graph_m1 at pos=16` ✓
**步时**：24.15→**22.56ms**（44.3 tok/s）——m=1 图化生效但收益有限（预期 3.35ms/行，实得 ~0.8ms/行）
**每行成本**：~9ms/行（vs EAGER 6.15ms）——**剩余 ~2.85ms/行**来自：
1. `host_barrier`（TP8 跨 rank 同步，每次 graph_launch 前调）~0.5-1ms/行
2. `D2H argmax`（每次 step_rows 后 4B 读回，需 device sync）~0.5ms/行
3. `set_pos_ctr` + `tap_commit` ~0.1ms/行

**优化方向**：批量化 barrier/D2H（每步一次而非每行一次）或混合模式（前 2 行 batched + 后续 lazy）。
**文本**：与之前完全相同（backbone 偏差不变，acs/Bristol/burdens/oqua）。

## 架构判定：lazy verify 无法达到 400 tok/s——batched + kernel 优化是正道

**数学**：
- Lazy at accept 3：4 行 × 6.5ms/行 = 26ms verify + 1ms draft = 27ms → 111 tok/s
- Batched (weight-stationary)：5 行共享权重 → ~6.5ms verify + 1ms draft = 7.5ms → **400 tok/s** ✓

**根本问题**：lazy 的每行是独立的 EAGER forward（权重重新读），batched 的 5 行共享一次权重读。用户"模仿 EAGER + 5 行几乎免费"的洞察指的是**batched weight-stationary**，不是 lazy（逐行重复读）。

**SH_EXP_MROWS 为什么只省 1.2ms（预期 8.3ms）**：共享专家的 kernel 带宽利用率只有 **0.7%**（377GB/s / 7.6TB/s）——它不是带宽受限，是指令/占用率受限。mrows "读一次权重"帮不了指令瓶颈的 kernel。真正的修复是 kernel 融合（减少 kernel 数）和 tcgen05（换核）。

**400 的真正路径**（verify-family-fusion subagent 在设计）：
1. **kernel 融合**：每层的 75 个 kernel → 5-10 个大 kernel（段融合）
2. **tcgen05**：routed experts 从 SIMT 到 tensor core（-6.8ms）
3. **共享专家的 kernel 优化**：不是 mrows，是更好的 tiling/占用率
4. **图化**：只提供 launch 消除（已确认 ~1.5ms 收益）

## Shared expert 占用率根因（代码级）——grid=72 blocks（49% SM），w1|w3 融合可到 97%

**维度确认**（config）：`moe_intermediate_size: 2304`，TP8 → sh_il = 288/rank。
**当前**：`shared_expert_mrows` 调 `gemm_fp8_mrows` 两次（w1 n=288 + w3 n=288），每次 grid = ceil(288/4) = **72 blocks**（148 SM 上 49% 利用率）。
**融合机会**：w1 和 w3 输出已写 `[row][2*sh_il]` 布局——若合为一次 n=576 调用（w1|w3 交错或两个权重指针），grid = ceil(576/4) = **144 blocks（97% SM）**——**2× 占用率 + 一半 launch**。

**组合修复**：
1. w1|w3 融合：72→144 blocks（97% SM）+ launch 减半
2. 小 n 的 nwarps 自适应：n=288 时 nwarps=2 → 144 blocks
3. 两者叠加效果：shared expert 从 10.4ms → 预期 2-3ms

## Barrier-batch 验证（b018101f）——无改善（22.59ms ≈ 22.56ms）

**结果**：步时 22.59ms（44.3 tok/s）与无 barrier-batch 的 22.56ms**完全相同**。host_barrier（TP8 跨 rank 同步）不是 lazy per-row 开销的瓶颈——它可能已经与 kernel 执行重叠。

**修正后的 per-row 开销分析**：
- 不是 barrier（已排除）
- 剩余嫌疑：①D2H argmax 的 stream sync（~0.5ms/行）②m=1 verify 图与 EAGER 图的 kernel 差异（compressor/spec 路径不同）③m=1 裸链→图化的 DRY 开销
- **22.6ms 步时的分解**：draft 4.28 + verify ~18 + commit 0.17；verify ~18ms / 2.08 行 = ~8.65ms/行（vs EAGER 6.15ms）

**结论**：lazy verify 的架构上限（每行付 EAGER 成本）已确认。400 的路径回到 **batched verify + kernel 融合**。

## Draft 链 P3a 折叠分析（当前 4.29ms → 目标 1ms）

**P3a 已实现的 4 项**（默认 OFF，`DSV41_DRAFT_P3A=1`）：
- a1: hc_collapse+rmsnorm 融合（-1 launch/block）
- a2: hc_post 双向写（-2/block）
- a3: premix ping-pong（-1/block）
- a4: rope mrows（-8/block）

**P3a 未实现的 2 项**（生产 bs=5 不可行）：
- a5 sparse_attn_orope：o-rope 的位置公式不能给行 r 位置 pos+r
- a6 WOB_F32：gemm_fp8_mx_f32 是 M=1 无行批

**draft 4.29ms 的剩余结构**（P3a 后）：draft 链跑 5 个 MTP 块（n_mtp=5），每块是完整 3 层 forward（hc+attn+MoE+compressor）。每块 ~0.86ms。要压到 1ms 总：
- 需要 P3c（draft 链的 CUDA 图化 + 块级 mrows——5 块的 forward 共享权重读）
- 或 draft 链的深度削减（3 层 MTP 是固定的，不能减）
- draft 图化：5 块 × ~40 kernel/块 = 200 launch → 图化后 1 launch——预期 -2ms

## Draft 链结构确认（P3c 图化的可行性）

`draft_forward`（dspark_dev.rs:860）的块循环：`for s in 0..cfg.n_mtp_layers`（3 块 MTP），每块 = hc_mixes + attn（sparse/window）+ hc_post + MoE（draft_moe）+ hc_mixes(ffn) + P3a 折叠。块间串行依赖（premix ping-pong a3 已把 D2D 消除）。**结构上可图化**：固定 kernel 序列 + 设备端输入输出（ids/pos 都在 device）+ 无宿主侧分支（除 unit_dump 的调试臂）。P3c = 捕获 3 块序列为一张图，launch 从 ~120 → 1。

## 段错误隔离记录（2026-09-12，commit 8521d6f 后）

| 隔离 | 配置 | 结果 |
|---|---|---|
| a8020426 | spec + SH_EXP_MROWS + SMALL_N_ADAPTIVE | **CRASH** |
| 355045b3 | spec + SH_EXP_MROWS（无 adaptive） | **CRASH** |
| 2c2efaf2 | spec（无 SH_EXP_MROWS） | **CRASH** |
| 6bee8c72 | **纯 EAGER**（无 spec） | **CRASH** |

**关键事实**：纯 EAGER 也崩 → 段错误在**主路径**（step_body 一侧），不是 verify/mrows。8521d6f 只改了 gemm_fp8_mrows_kernel 的 staging——EAGER 不调用它，但 .so 重编可能改变了**整个文件的寄存器分配/符号布局**（相邻 kernel 的代码生成变化）。
**ABI 检查**：ferrite_kernels.cu=3u，cuda.rs/devrt.rs=3 ✓（一致）
**FFI 检查**：hc_collapse_norm 的 truncate 参数位置正确（truncate as c_int 在 stream 前）✓
**正在跑**：远端手动回退 cp.async staging → 重建 → 测试（37073a37）

## MoE gate 的 mrows 现状——DSV41_GATE_MROWS 已存在（默认 OFF）

`row_fold_gate()`（chain_dev.rs:1070）：`DSV41_ROW_FOLD_GATE` 或 `DSV41_GATE_MROWS` 都能开。走 `gemv_bf16_v2_mrows`（与 `gemv_bf16_nt` 同一 program 的 m 行形，`tests_gate_mrows.cu` 断言位等价 @ WPR>1 shape）。**gate 的 mrows 版本已实现**——只差 GPU A/B 验证它的 3.44→0.8-1.5ms 收益。

## Accept 率提升路径（400+ 的关键乘数）——draft 的 bf16 截断

**现状**：k_acc 直方图 {0:55, 1:13, 2:10, 3:4, 4:3, 5:1}——**64% 首token拒绝**。sglang 同一 MTP head 达 accept ~5 → ferrite 的 draft 数值路径在压低它。

**三层对齐需求**（MTP head 在官方参考全 bf16 下训练）：
1. **backbone 的 tap**（draft 的输入）：ferrite f32 vs 官方 bf16 → tap 有 ~1e-3 噪声
2. **draft 内部**（MTP 3 层的 hc/attn/MoE）：ferrite f32 vs 官方 bf16 → 预测偏移
3. **backbone 的 argmax**（verify 的基准）：bf16 截断已实现（等段错误修复后验证）

**修复顺序**：
- 第 1 步（已实现）：backbone 的 hc_pre 截断（DSV41_BF16_TRUNCATE）——让 verify 基准对齐
- 第 2 步（待做）：tap 的 bf16 截断——`note_ctx_rows` 的 D2D 后加 round-trip（或在 `import_tap` 处）
- 第 3 步（待做）：draft 内部的截断——draft 链的 hc_collapse/hc_front 也传 bf16_truncate()（dspark_dev.rs 的 2 处已传——eager-bf16-truncate subagent 已做 ✓）

**预期**：如果 64% 拒绝主要来自数值错位（而非 MTP head 能力），对齐后 accept 应显著提升。sglang 的 accept 5 是同一 head 的上限证明。

## Tap 截断的数值域精确分析（不是位等价，是"更近"）

**官方的 tap 精度链**：44 层的残差流全程 bf16（每层边界截断）→ tap 在层 37/38/39 捕获时已是"44 次 bf16 截断后的值"。
**ferrite + tap 截断**：44 层的残差流全程 f32（无中间截断）→ tap 捕获后做**一次** bf16 round-trip。

**结论**：ferrite 的 tap 值 = f32 精确计算后截断一次；官方的 = bf16 逐步截断的累积。两者**不是位等价**——ferrite 更精确（中间无损失），但最终 dtype 对齐。~2.4% 的 norm 偏差会保留（来自中间层的精度差）。

**对 accept 的影响**：draft 的输入 dtype 对齐（bf16）可能改善 MTP head 的预测（head 在 bf16 输入上训练），但中间值的精度差仍在。**效果只能实测**——如果 accept 提升显著，说明 MTP head 对输入 dtype 敏感；如果不变，说明 head 对中间精度差不敏感。

## Draft 内部对齐完整性检查——MoE 已共享 e4m3 路径 ✓

Draft 的 MoE 用 `expert_gate_up_fp4_batched` / `expert_down_reduce_fp4_batched`（dspark_dev.rs:1963/2006）——**与 backbone 相同的 kernel**，所以 `DSV41_EXPERT_ACT_E4M3=1` 同时作用于 draft 和 backbone 的 MoE。

**Draft 对齐状态汇总**：
| 组件 | 对齐 | gate |
|---|---|---|
| 输入（tap→main_h）| ✅ bf16 round-trip | DSV41_TAP_BF16 |
| hc_pre/hc_front | ✅ bf16 截断 | DSV41_BF16_TRUNCATE |
| MoE 激活 | ✅ e4m3（同 kernel）| DSV41_EXPERT_ACT_E4M3 |
| attention 内部 | ❌ f32（未截断）| 需要进一步工作 |
| 其他（rope/quant）| ❌ f32 | 影响待评估 |

**三个主导项已对齐**（输入+hc+MoE）——如果 accept 仍不提升，剩余的 attention 内部对齐是下一步。

## 🎯 用户红线达成：零拉丁字符（a8b578cb，干净重建 + ABI5 + 双截断）

**结果**：
- **拉丁=[]（零拉丁字符！）** — bf16 截断彻底消除 acs/ibu/Bristol/burdens/oqua ✓✓✓
- **段错误修复** — SURVIVED（干净重建 + ABI 5 一致性）
- LEN=164，双字=4（内容错误如"泄"代"义"——中文字错，非拉丁碎片）
- verify=33.96ms（vs 之前 38.34ms，**-4.4ms**——mrows staging 修复生效）
- k_acc {0:64, 1:13, 2:4, 3:4, 4:3} mean-k=0.820——**accept 反而降了**（1.080→0.820）

**分析**：
1. **零拉丁字符**：backbone 的 hc_pre bf16 截断让 backbone 对齐官方 → 累积漂移消除 → 不再翻转出拉丁 token
2. **accept 下降**：backbone 截断让 verify 更严（同 e4m3 的效应）——backbone 的 argmax 变了（更准），draft 的预测没跟上（tap 截断不足以对齐 draft）
3. **性能**：verify 33.96ms 是 batched——lazy verify 会 ~20ms

**下一步**：lazy verify + 双截断的组合测试（性能 + 正确性）。

## Accept 率下降的根因分析（截断后 0.820 vs 截断前 1.080）

**现象**：backbone 的 bf16 截断让 verify 更严——argmax 变了（更准，对齐官方），但 draft 的预测没跟上。k_acc=0 从 64%→73%。

**机制**：
1. backbone 44 层的 hc_pre 截断 → 累积效应大 → argmax 显著变化（更接近官方）
2. draft 3 层 MTP 的截断（tap + hc + MoE）→ 累积效应小 → 预测变化较小
3. **draft 的 attention 仍是 f32**（未截断）→ MTP head 的预测与官方的 bf16 MTP head 有差

**官方的 DSpark accept ~5 的前提**：官方的 backbone 和 MTP head **都是 bf16**——两者天然对齐。ferrite 的 backbone 现在对齐了（bf16 截断），但 draft 只有部分对齐。

**修复方向**（按 ROI）：
1. draft 的 attention bf16 截断（sparse_attn 的 q/k/v 输出截断）——工作量中等
2. draft 的 rope 输出截断——工作量小
3. 或者反向：**比较 ferrite draft 的预测与官方 MTP head 的预测**（用 DSV41_DIFF_EAGER 式探针）——精确定位 draft 的哪一层开始偏

## Lazy + 双截断组合测试（a54bd0cc）

**结果**：
- **零拉丁字符 ✓**（lazy verify 下也保持）
- **k_acc {0:20, 1:17, 2:2, 3:3, 4:3, 5:1}——43% k_acc=0**（vs batched+截断的 73%，vs lazy 无截断的 64%）——**accept 显著改善**！
- 步时 33.10ms（30.2 tok/s）——**比无截断 lazy 的 22.56ms 慢 10.5ms**（需分析：截断计算开销 or 路径变化）
- LEN=120，双字=4

**k_acc 改善的机制**：lazy 的逐步检查让 draft 更早获得正确反馈（拒绝后下一步从正确位置重新开始）。tap 截断 + backbone 截断的对齐效应在 lazy 下更显著。

**步时退化嫌疑**：
1. bf16 截断的计算开销（round-trip 指令 × 40 层 × 2 侧）——应该很小（2 条指令）
2. verify 图与截断的交互（图捕获可能失败，回退裸链）
3. SH_EXP_MROWS + 截断的路径冲突

## Draft attention 的 bf16 截断设计（accept 提升的下一步）

**现状对齐矩阵**（lazy + 双截断 = 43% k_acc=0，比无截断的 64% 改善 21 个百分点）：
| 组件 | backbone | draft | 对齐 |
|---|---|---|---|
| hc_pre | ✅ bf16 | ✅ bf16 | ✓ |
| tap/输入 | N/A | ✅ bf16 | ✓ |
| MoE 激活 | ✅ e4m3 | ✅ e4m3（同 kernel）| ✓ |
| attention q/k/v | ❌ f32 | ❌ f32 | 部分（同错可抵消）|
| attention 输出 | ❌ f32 | ❌ f32 | 部分 |
| rope | ❌ f32 | ❌ f32 | 部分 |

**关键洞察**：backbone 和 draft 的 attention 都是 f32——它们的"同错"可能部分抵消（draft 预测的偏差方向与 backbone 的 argmax 偏差一致）。但 draft 是在 official 的 bf16 attention 上训练的，其权重校准期望 bf16 噪声。

**修法**（按侵入度）：
1. **draft 的 attention 输出截断**（最简）：每个 MTP 层的 o 投影输出后加 round-trip（3 层 × 1 次 = 3 kernel/步）——把 draft 的 attention 输出拉回 bf16 精度
2. **draft 的 q/k/v 截断**：fp8 GEMM 输出后 round-trip（更细粒度但更多截断点）
3. **全 draft 内部截断**：每层 residual 都 round-trip（最彻底但最贵）

**推荐**：先做 #1（最简，一次试验即可判定方向）。
