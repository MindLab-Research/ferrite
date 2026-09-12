# EAGER vs verify：attention 之外的 per-row 剩余差距（2026-09-12）

> 口径：`chain_dev.rs`，默认 gate，`world == 1`，verify 块 `m = 6`（`VERIFY_ROWS = 6`）。
> 只读分析，未改任何代码。EAGER 侧 = `layer()` / `attention()` / `moe()`（每步 1 行）；
> verify 侧 = `layer_rows()` / `attention_rows()` / `moe_rows()`（一块 m 行）。
> **"发数" = 主机侧 kernel launch 数**（fork/join 事件与 stream 切换不计）。

---

## 0. 结论先说

1. **总量上 verify 每行已经比 EAGER 便宜**（~23 发/行 vs EAGER ~31 发/行，见 §6），
   所以"模仿 eager"的剩余 ROI **不在总量，而在"哪些段仍随 m 线性增长"**。
2. 每层 137 发里有 **114 发（83%）是 per-row 线性段**：
   indexer 42、shared expert 30、compressor proj 12、compressor pool+commit 12、
   ring+window 12、MoE gate 6。
3. 这 114 发里 **有 53 发可以用「EAGER 已有实现」直接消掉**（2 段是纯 gate 翻转，
   2 段是同 kernel 传 `pos_rows + r`，2 段是已实现的 mrows 孪生），剩余 2 段需要新 kernel。
4. **L4-6 的 fork/join 与 shared-expert mrows 家族互斥**（代码级互斥，见 §5），
   规划时不能同时开。

---

## 1. 逐段差距表（每层，m = 6）

| 段 | EAGER（1 行） | verify（R2 后，1 块 6 行） | per-row 差距 | 可否用 EAGER 的实现消除 |
|---|---|---|---|---|
| **hc front ×2** | 2（`hc_front_split` ×2） | 6（`hc_mixes` + `hc_collapse` + `rmsnorm` ×2） | 块级 +4（**+0.67/行**） | ✅ **纯 gate**：`HC_FRONT_ROWS=1` → `hc_mixes_auto`(`rows=m`) 走 EAGER 的 `hc_front_split`，2 发 |
| q/kv 投影 | 2（`lin2` + `lin_rope_norm`） | 2 ✅ R2 | 0 | 已完成 |
| qr 归一 / q rope | 0（融进 `lin_rope_norm`） | 0 ✅ R2b | 0 | 已完成 |
| kv norm+rope | 1（`rmsnorm_rope_on`） | 2（`norm_rows_on` + `apply_rope_on`） | 块级 +1 | ⚠️ EAGER 的两发融成 1 发；verify 的 `rows` 形式是 2 发（可做 `rmsnorm_rope_mrows`，收益仅 1 发/层） |
| **ring append + window** | **1**（`ring_win_fuse_ph`） | **12**（`ring_append` + `window_idxs`，2/行） | **+1/行 = +6/层 = +240/步** | ✅✅ **同一 kernel，零新代码**：`dsv41_ring_win_fuse` 收 `pos_ctr: *const c_int`，verify 传 `pos_rows + r` 即是 EAGER 的 m=1 调用（append 与 idxs 同位置指针，位级一致） |
| comp_placeholder | 0（融进 `ring_win_fuse_ph`） | 0~m（仅"owner 非 index 非 compress-source"层） | 生产层为 0 | 生产层（2/8/14/20）都是 index_source → 不触发 |
| **compressor 投影** | **2/行** | **2/行 × 6 = 12** | 0（每行同位） | ❌ 无 `rows` 形式（`lin_f32` 无 mrows 孪生；`b6-mrows-f32-design` B4/B6 是候选）—— **唯一"EAGER 也每行付"的段** |
| **compressor pool+commit** | **1**（`compressor_fused`，state+pool+commit 融 1 发） | **12**（`compressor_pool_on` + `compress_commit_on`，2/行） | **+1/行 = +6/层** | ✅ `COMPRESSOR_MROWS=1`（**已实现，默认 OFF**）→ 1 发；注意 EAGER 的 `compressor_fused` 本身拒绝 `seqlen != 1`，不能直接复用 |
| **indexer** | **7**（publish 4 + q 1 + w 1 + topk 1） | **42**（7/行） | 0/行 | ✅ `INDEXER_MROWS=1`（**已实现，默认 OFF**）→ 4 + 5m = 34（比 EAGER 还少） |
| sparse attn + o-rope + o-quant | 1（`sparse_attn_orope`） | 6（1/行） | 0 | 已完成（`VERIFY_OROPE` 默认 ON） |
| wo_a + wo_b | 2（每步） | 2（块级 `wo_a_grouped_fp8` + `proj_mrows`） | − | verify **更好**（块级摊销） |
| **AR ×2** | 2（含 `hcpost` fold） | 2（plain） | 0 | — |
| **hc_post ×2** | **0**（融进 AR 的 `hcpost` epilogue） | **2**（`hc_post_rows` ×2） | 块级 +2 | ✅ **纯 gate**：`VERIFY_AR_FOLD=1` → `ar_hc_post_fold_rows`（**已实现，默认 OFF**） |
| **MoE gate+route** | **1**（`gemv_bf16_route`，`ROUTE_FUSE` 默认 ON） | **6**（`gemv_bf16` ×6）+ 1 `route_topk` | **+1/行 = +6/层** | ⚠️ 无 mrows route-fused kernel。`ROW_FOLD_GATE=1`（默认 OFF）只把 6 → 1（`gemv_bf16_v2_mrows`），仍比 EAGER 多 1 发 `route_topk` |
| MoE routed（quant+gateup+down） | 3 | 3（rows 原生） | 0 | 平手 |
| **MoE shared expert** | **5**（含 join add） | **30**（5/行） | 0/行 | ✅ `SH_EXP_MROWS=1` → 6 发（**已实现，默认 OFF**）；另有 `SH_PAIR_M` / `SH_EXP_FUSED` 两个更激进的孪生 |
| **head** | 1 `gemv_bf16` + 1 `argmax_sliced`（每步） | 6 `gemv_bf16` + 1 `argmax_sliced_rows` | 步级 +5（**不随层增长**） | ✅ `VERIFY_HEAD_MROWS=1`（**已实现，默认 OFF**）→ 1 发 v1-order 折叠（位级一致） |
| **每层合计** | **~31**（每行） | **~137**（每块 6 行）= **22.8/行** | — | — |

---

## 2. 属于"attention"的剩余项（R2 未覆盖）

R2 只覆盖了投影族（`lin2` + `lin_rope_norm`）。attention 内部仍是 per-row 的还有：

| 项 | EAGER 发数 | verify 发数 | 性质 |
|---|---|---|---|
| ring append + window_idxs | 1（融合） | 2/行 | **同 kernel 可复用** |
| compressor 投影 | 2 | 2/行 | 每行同位（EAGER 也是 2）|
| compressor pool+commit | 1（融合） | 2/行 | 需 mrows 孪生 |
| indexer publish | 4 | 4/行 | 每行必须（每个 commit 组要自己的 key）|
| indexer q/w front | 2 | 2/行 | 需 mrows 孪生（已实现）|
| indexer topk | 1 | 1/行 | 必须每行（读行自己的 `*clen`）|
| o-rope + o-quant | 0（`sparse_attn_orope` 内融） | 0 | 已完成 |

**attention 段唯一"零成本复用"项 = ring+window**：EAGER 用 `ring_win_fuse` 把
"append + window_idxs" 融成 1 发（`attention()`:13831-13870）。verify 在 per-row
interleave 里分两次调用（`attention_rows()`:9969-9981），因为 B2 的**块级**融合
（`verify_ring_win`）在 swap/ring 翻越时是错的（审计缺陷 #2）——但**逐行**调用
EAGER 这个 kernel 没有那个问题：append 与 idxs 对同一行、同一 `pos_rows + r`，
正是 EAGER 的 m=1 语义。

---

## 3. 缺口排序（ROI，按每层可省发数）

| 排名 | 段 | 每层可省 | 每步（40 层） | 落地方式 | 风险 |
|---|---|---|---|---|---|
| 1 | shared expert | −24 | −960 | `SH_EXP_MROWS=1`（已实现） | 与 `VERIFY_FORK` 互斥（已接线）；需 `sh_il%32==0` |
| 2 | compressor pool+commit | −11 | −440 | `COMPRESSOR_MROWS=1`（已实现） | 与 `ATTN_MROWS` 互斥（fence 已写）；需 per-row 快照 |
| 3 | indexer front | −8 | −320 | `INDEXER_MROWS=1`（已实现） | 读侧 `*clen` 已改成 per-row 快照 |
| 4 | ring+window | −6 | −240 | **复用 EAGER 的 `ring_win_fuse`（无新 kernel）** | 低：逐行语义 = EAGER |
| 5 | MoE gate | −4 | −160 | `ROW_FOLD_GATE=1` → 6→1 发（仍多 1 `route_topk`） | 低 |
| 6 | hc front | −4 | −160 | `HC_FRONT_ROWS=1`（已接线） | 中：verify 侧未有 A/B 记录 |
| 7 | hc_post fold | −2 | −80 | `VERIFY_AR_FOLD=1`（已实现） | 中：v5 pubred epilogue 形状 |
| 8 | head | −5/步 | −5/步 | `VERIFY_HEAD_MROWS=1`（已实现） | 低（v1 位级一致）|

合计：每层 137 → **78 发（−43%）**，即 ~3140 → ~2580 发/步（按 40 层）。
全部落地后 verify 每行 ≈ 13 发 vs EAGER 31 发。

**剩下两个"真需要新 kernel"的段**：
- compressor 投影的 mrows 形式（2m → 2）—— EAGER 也是 2，所以这是"verify 比 EAGER 贵"的唯一实质项，+10 发/块/层；
- MoE 的 gate+route mrows 融合（6+1 → 1）—— EAGER 有（`gemv_bf16_route`），mrows 版没有。

---

## 4. gate 默认值一览（本次分析依据）

**EAGER 侧默认 ON**：`FUSE_B1`、`FUSE_C`、`HCPOST_EPI`、`DUAL_CHAIN`、`COMPRESS_SIDE`、
`COMPRESS_FUSE`、`MOE_DUAL`、`MOE_BATCH`、`ROUTE_FUSE`、`GATEUP_FUSE`、`DOWN_FUSE`、
`SWIGLU_Q`、`SH_EXP_MX2`、`NR_FUSE`、`QR_EPI`、`NORM_FUSE`、`ROPE_FUSE`、`SPARSE_OROPE`、
`OROPE_Q`、`RING_WIN_FUSE`、`COMP_PLACEHOLDER_FUSE`、`HC_TAIL_SPLIT`、`ADD_EPI`、`HEAD_SLICE`。
默认 OFF：`MIX_GATE`、`MOE_EPI_ADD`、`IDX_FUSE`、`WO_QUANT_FUSE`、`AR_STORE_FUSE`、`HC_PERSIST*`。

**verify 侧默认 OFF（"已接线未测"）**：`VERIFY_FORK`、`VERIFY_AR_FOLD`、`COMPRESSOR_MROWS`、
`SH_EXP_MROWS`、`SH_PAIR_M`、`SH_EXP_FUSED`、`ATTN_MROWS`、`INDEXER_MROWS`、`ROW_FOLD_GATE`、
`ROW_FOLD_ROPE`、`VERIFY_ROPE_MROWS`、`VERIFY_HEAD_MROWS`、`VERIFY_HEAD_FOLD`、`HC_FRONT_ROWS`。
默认 ON（verify 侧）：`VERIFY_OROPE`、`VERIFY_HEAD_SLICED`。
R2：`ATTN_LIN_FUSE` 默认 OFF（`=1` 开）、`INDEXER_QR_RAW` 默认 ON（R2 下）。

---

## 5. L4-6 fork/join 的接线状态与冲突

`verify_fork()`（`chain_dev.rs:1163`，默认 **OFF**）已接线三处 seam：
1. attention q/kv 双链（`side_stream2`，`attention_rows()`:9592-9596 / :9788-9790）；
2. compressor 投影第三流（`side_stream3`，:9611-9623 / :9839-9840）；
3. MoE routed vs shared（`side_stream2`，`moe_rows()`:11701-11713 / :12271-12278）。

**代码级互斥（规划必读）**：`moe_rows()` 的 `dual` 谓词里显式排除
`sh_exp_mrows() || sh_pair_m() || sh_exp_fused()`（:11706-11708），
反向 `sh_mrows_done` 在 `dual` 时强制 `false`（:12130-12134）——
**两条优化族不能同时开**。同理 `ATTN_MROWS` 在 `mrows_own_owner`（compressor 已 hoist）
时 decline（:9925-9936）。所以 §3 的 1/2/5 项要分组合 A/B，不是全开。

---

## 6. 每行口径的对照（为什么总量不是问题）

| | EAGER | verify (m=6) |
|---|---|---|
| 每层发数 | 31（= 1 行的成本） | 137（= 6 行的成本） |
| 每行发数 | 31 | 22.8 |
| 块级摊销项 | — | hc 6、wo 2、AR 2、MoE 非 shared 11 → 21 发摊到 6 行 |
| 每行线性项 | 全部 | 114 发（indexer 42 + shared 30 + comp 24 + ring 12 + gate 6） |

结论：verify 的架构优势来自"块级摊销"，劣势来自"per-row 家族没有 mrows 孪生"。
"模仿 EAGER" 的正确含义 = **把每个 per-row 家族换成它的 mrows 孪生（或 EAGER 的
单行融合 kernel 逐行调用）**，而不是把整条链退回单行。
