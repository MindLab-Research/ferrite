# spec 步跨步不变量清单与断言层设计

> 刑部（spec-step-hardening）· 2026-09-12 · 审计判词（主 agent 代落盘）
> 范围：`DevChain::{dspark_spec_step, dspark_spec_aligned, dspark_spec_swallowed,
> dspark_commit, dspark_rollback_keep, dspark_snapshot, step_rows, step_rows_inner,
> compress_replay, compress_row, compress_proj_rows, advance_compress_lens}`
> + `DsparkDev::{import_tap, draft_forward, note_ctx_rows, seed_window, ensure_pos_dev}`
> + `serve.rs` 的 `DecodeRun` 驱动循环。
> 目标：把"跨步不变量被静默破坏"这一族**一次性封死**。

## 0. 结论摘要

本会话已确认的 8 个根因**全部**落在同一族：**"上一轮的输出是下一轮的输入"的某个连接点，被一侧独立重算/偏移后无人校验**。共性：
1. 破坏后**不报错**——只是一步或几步之后的 token 变错，或者 accept 率缓慢塌陷；
2. 两侧**各自持有同一个量的副本**（device 计数器 vs host mirror，生产者行距 vs 消费者行距，caller 假定的 pos_base vs device 读到的 pos_base），其中一个改了，另一个不动；
3. 深埋在**行距/索引算术**里，类型系统看不见。

断言层设计原则：**每一对副本都强制对齐，且在破坏发生的那一步就报错**（一步必爆）。

**现状统计**：穷举 **29 条**跨步不变量：
- 零成本可加（纯 host 比较）：13 条
- 1 次小 D2H（4B~160B）：10 条
- 毒化自检（每请求首轮一次）：4 条
- 结构性/静态（编译期或构造性）：2 条

## 1. 设备缓冲清单（"谁能被子孙读到"）

| # | 状态 | 形状 | 谁写 | 谁跨步读 |
|---|---|---|---|---|
| D1 | `s.pos_ctr` | `[1] i32` | step_body argmax / dspark_commit→set_pos_ctr | 所有 kernel 的 pos；下一轮 driver |
| D2 | `s.ids` | `[1] i32` | step_body argmax | 下一轮 step_body 的 embed |
| D3 | `s.clen` | `[n_layers] i32` | compress_commit_on / compress_replay | indexer_topk 的 n_pos、sparse_attn 的 n |
| D4 | `layers[l].ring` | `[win+max_comp, hd]` | ring_append / compress_commit_on | 注意力读；snapshot/rollback 覆盖 |
| D5 | `layers[l].state_kv/state_score` | `[ratio, hd]` | compressor_pool_on | 下一轮 pool；snapshot/rollback |
| D6 | `layers[l].latent` | `[hd]` | compressor_pool_on (mode 2) | publish_index_key（跨步） |
| D7 | `layers[l].out_rows` | `[1] i32` | compressor_pool_on | 同一调用内 commit |
| D8 | `layers[l].index_k` | `[max_comp, index_hd]` | publish_index_key（槽 *clen-1） | indexer_topk |
| D9 | `layers[l].idxs` | `[win+topk] i32` | window_idxs/indexer_topk 每步重写 | 仅同一步 |
| D10 | `s.dspark_tap` | `[slots, dim]` | layer() hook / carry_kept_tap | 下一轮 import_tap |
| D11 | `s.dspark_tap_r` | `[slots, VERIFY_ROWS, dim]` | layer_rows hook（spec_capture） | note_ctx_rows / carry_kept_tap |
| D12 | `s.spec_snap_kvp/_scp` | `[n_layers, VERIFY_ROWS, hd]` | compress_proj_rows | compress_replay（同一步内 commit 阶段） |
| D13 | `s.dspark_snap_{ring,state,latent,clen,out_rows}` | 按层 | dspark_snapshot | dspark_rollback_keep |
| D14 | `s.pos_rows` | `[VERIFY_ROWS] i32` | step_rows；compress_replay 又重写 | 各 kernel 位置指针 |
| D15 | `s.ids_r` | `[m] i32` | step_rows 每次上传 | step_rows_inner 的 embed |
| D16 | `s.argmax_r` | `[m] i32` | verify head / argmax_sliced_rows | step_rows 的 D2H |
| D17 | `eng_dev.cache` | `[max_seq] i64` | engram_hash_step | 跨步 n-gram |
| D18 | `s.eng_ids_r` | `[m, …] i64` | engram_hash_step | engram_apply_rows（同一步） |

## 2. Host 状态清单

| # | 状态 | 含义 |
|---|---|---|
| H1 | `LayerCache.compress_len` | `s.clen[l]` 的 host 镜像 |
| H2 | `DsparkDev.pos_dev` | 设备 RoPE 基的 host 影子 |
| H3 | `DsparkDev.n_win_cached` | idxs 当前几何的有效窗长 |
| H4 | `DevChain.spec_primed` | 引导轮标志（AR 足迹对齐） |

## 3. 29 条不变量（按族分组）

### 族 A：位置/计数器（8 条）
A1. `s.ids == emitted.last()`（每臂结尾）——**本会话根因 #8**
A2. `pos_ctr == p`（driver vs device）——debug_assert 已有但 release 编译掉
A3. `pos_rows[r] == pos_base + r`（构造性，但 pos_base 来源无断言）
A4. `DsparkDev.pos_dev == pos_ctr`（RoPE 基的影子）
A5. `verify_out.len() == m`
A6. `argmax_r[0] < vocab`（首行抽查）
A7. `emitted[i]` 的位置严格 `pos+1+i`
A8. `set_pos_ctr(pos_base + keep)` 与 `p += emitted.len()` 恒等（构造性已证）

### 族 B：行距/布局（6 条）
B1. quant 的源行距 == 缓冲真实 pitch——**本会话根因 #6**
B2. quant 的目行距 == 下游消费方假设
B3. `tap_r` 的 slot 行距 == VERIFY_ROWS——**本会话根因 #7**
B4. `logits_r` 的行距 == vocab 或 seg（按 head_slice）
B5. `ex_act_b` 的 slot 行距 == act_slot
B6. `pos_rows` 被 compress_replay 重写后的行距

### 族 C：压缩器/注意力状态（7 条）
C1. `compress_len[l]（host）== *clen[l]（device）`（回滚后）
C2. `compress_replay(pos_base, keep)` 的 pos_base/keep 与块基址一致
C3. `dspark_snapshot` 覆盖的层集合 == `dspark_rollback_keep` 恢复的层集合
C4. `state_kv/state_score` 的回滚值 == snapshot 值（逐层首元素抽查）
C5. `index_k` 的槽 `*clen-1` 在回滚后不被跨步读到
C6. `latent` 的跨步 publish（mode 2）与 snapshot 的时序
C7. `out_rows` 的回滚

### 族 D：draft/tap（5 条）
D1. `dspark_tap` 的 slot 语义（层 37/38/39 的输出）
D2. `seed_window(pos)` 的 pos 与 tap 的实际位置——**本会话判词（seed↔tap 差一位）**
D3. `note_ctx_rows` 的 m/keep 与 verify 块的对应
D4. `carry_kept_tap` 的行号 == keep-1
D5. `draft_forward` 的 pos==0 早退的 AR 足迹对齐

### 族 E：engram/杂项（3 条）
E1. `engram cache` 的跨步 n-gram 键（回滚不覆盖——判词：方向性安全）
E2. `spec_capture` 标志的一致性（host 分支）
E3. `argmax_sliced_rows` 的 epoch 消耗 == 1/调用

## 4. 断言层实施（spec-inv-assert-impl 正在做）

gate `DSV41_INV_CHECK=1`（默认 OFF）——优先族 A（8 条，全部 4B D2H）+ 族 B 的 B1/B3（静态 debug_assert）。
