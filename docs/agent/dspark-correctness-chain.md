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
