# EAGER 融合算子 → verify 路径的迁移方案

> 工部（ministry-works）· 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 任务（用户明确要求）：**「参考 eager 实现的融合算子，必须性能很高」**——把 EAGER 路径（`layer()`，m=1 单行 decode）
> 已经上线的高性能融合，迁移到 verify 路径（`layer_rows()`，m=5 多行块）。
>
> 代码基线：`crates/ferrite-models/src/dsv41/chain_dev.rs`（`layer()`@12008 / `layer_rows()`@8558 /
> `attention_rows()`@8846 / `moe_rows()`@10816 / `hc_mixes_auto()`@11634 / `shared_expert_mrows()`@11422）、
> `crates/ferrite-models/src/dsv41/device.rs`、`kernels/cuda/*.cu`。
> 性能基线：verify(m=5) = **37.31 ms / 6224 launches**（`DSV41_TIMING`）；EAGER 主链步 = **6.15 ms**。
> 输入文档：`verify-family-fusion.md`、`verify-architecture-floor.md`、`verify-operator-optimization-list.md`、
> `verify-ms-breakdown.md`、`dspark-verify-perf-plan.md`、`final-400-config.md`。
>
> **本文档只做迁移规划，不含实施**。所有 gate 一律「先 parity、后 A/B、默认 OFF」。

---

## 0. 判决（先读这六条）

1. **EAGER 与 verify 不是「同一套 fusion，一个开一个关」——存在一个结构性的 `m=1 → m=5` 缺口。**
   EAGER 的全部融合核都写在 **单行程序**上（`gemm_fp8_mx_rope`、`lin2`、`rmsnorm_rope`、`gemv_bf16_route`）；
   而 verify 折叠 5 行时走的是**另一条核**（`gemm_fp8_mrows`、`proj_mrows`）。两者**K 序不同**，
   不能直接互换（这也是 `draft-verify-program-audit` 发现的 accept 天花板根因之一）。
   ⇒ 迁移任务分成两类：**(A) 已有 m 行核、只差 flag 的**（便宜）与 **(B) 需要新写 m 行融合核的**（贵）。

2. **8 项里，6 项已在树里、默认 OFF（Category A）**：
   `hc_collapse_norm(A1)` / `hc_post_inplace_rows(A1)` / `hc_front_split(A2)` / `VERIFY_AR_FOLD` /
   `compressor_fused_mrows` / `apply_rope_mrows`。**`sparse_attn_orope` 已经默认 ON**（`VERIFY_OROPE`）。
   ⇒ **最大的一笔钱不需要写一行 kernel，只需要「先修 truncate 坑、再翻 flag」**。

3. **`hc_front_split`（HC_FRONT+TAIL_SPLIT）是这批的第一个大项**：verify 的 hc 链现在是
   **10 发/层**（2 block × 5 发），EAGER 的是 **~2 发/层**（front 1 + post 0~1）。
   迁移路径 **A2 `HC_FRONT_ROWS=1`** 已实现且 `rows`-agnostic；**唯一前置是 A1 的 `truncate=false` 修复**——
   **该修复已落地**（`collapse_norm_rows`@8438 传 `false`，2026-09-12）。⇒ **A1+A2 可以立刻 A/B**。

4. **Category B 里唯一真正缺的 m 行融合是 `gemm_fp8_mx_rope → verify`。**
   EAGER 把 **wq_b 投影 + rope** 融成一发（`lin_rope` / `lin_rope_norm`）；verify 走
   `proj_mrows(wq_b)` + `apply_rope_mrows` **两发**。没有 `gemm_fp8_mrows_rope` 这个符号。
   ⇒ **需要新 kernel**（把 `gemm_fp8_mrows` 的 epilogue 加一个 rope 相位，逐位等价按 C1–C6 论证）。

5. **收益的第一性来源是「少发 kernel」**：verify 6224 发 ≈ 50% 提交 + 49% 每发最小执行
   （`verify-ms-breakdown §2`）。每少一发 ≈ 省 `2.9+3.3 = 6.2µs`。Category A 全部兑现 roughly
   **−4200 发 ≈ −14~18 ms**（口径见 §4，含 flag-flip 的实测兑现度风险 R6）。

6. **风险排序：数值 > 死锁 > 收益高估。**
   `HC_VERIFY_FUSE` 之前正是因为 **BF16_TRUNCATE 第一次漏进 verify** 而破坏零拉丁基线（现已修）；
   `sh_pair` 是 **grid barrier** 核（grid > co_res 会死锁）；`gemm_fp8_mx_rope` 的 m 行版
   会引入**新的 K 序**（是数值改动，不是 launch 改动）。三者都必须 A/B。

---

## 1. EAGER `layer()` 的融合算子全清单（代码级）

> 来源：`chain_dev.rs` 的 `layer()` / `attention()` / `moe()` / `compress_on()` / `step_body()`。
> 「默认」列 = 代码里 `unwrap_or(...)` 的实值（不是注释声称的值——本仓有注释漂移的历史）。

### 1.1 attention 半链

| # | EAGER 融合 | gate（默认） | 折叠了什么 | kernel / 落点 |
|---|---|---|---|---|
| E1 | `lin2`（wq_a + wkv 同激活） | `PROJ_FUSE` (ON) | quant1 一次 + 一个 `gemm_fp8_mx2` | `gemm_fp8_mx2`@4054 |
| E2 | `lin_rope_norm`（rmsnorm 前置 + wq_b + rope 尾） | `NORM_FUSE` (ON) | rmsnorm_q + quant1 + gemv + rope → 1 发 | `gemm_fp8_mx_rope_norm`@4191 |
| E3 | `lin_rope`（wq_b + rope 尾） | `ROPE_FUSE` (ON) | gemv + rope → 1 发 | `gemm_fp8_mx_rope`@4094 |
| E4 | `rmsnorm_rope`（kv norm + kv rope） | `NR_FUSE` (ON) | 2 发 → 1 发 | `rmsnorm_rope`@2809 |
| E5 | `ring_win_fuse_ph`（ring append + window idxs + placeholder） | `RING_WIN_FUSE` (ON) / `COMP_PH_FUSE` (ON) | 2~3 发 → 1 发 | `dsv41_ring_win_fuse_ph`@1046 |
| E6 | `sparse_attn_orope`（o-rope + fp8 emit 折进 attn 尾） | `SPARSE_OROPE` (ON) | attn + rope + quant → 1 发 | `dsv41_sparse_attn_orope`@7522 |
| E7 | `gemm_fp8_mx_q`（wo_a epilogue 出 fp8） | `WO_QUANT_FUSE` (**OFF**) | 省 quant1(s.wo) | `gemm_fp8_mx_q`@13181 |
| E8 | `gemm_fp8_mx_f32`（wo_b 直读 f32） | `WOB_F32` (ON) | 省 quant1(s.wo) | `dsv41_gemm_fp8_mx_f32` |
| E9 | `wo_pair`（wo_a→wo_b 单发 grid-sync） | `WO_PAIR` (**OFF**) | 2 发 → 1 发 | `dsv41_gemm_fp8_wo_pair`@6360 |
| E10 | `hc_front_split`（hc 前端的 EARLY/LATE 侧流拆分） | `HC_TAIL_SPLIT` (ON) | 见 1.3 | `dsv41_hc_front_split`@9898 |
| E11 | `hc_collapse_norm`（collapse + norm） | `FUSE_B1` (ON) | 2 发 → 1 发 | `dsv41_hc_collapse_norm` |
| E12 | `hc_post_inplace`（post + 免 memcpy） | `FUSE_C` (ON) | 2 发 → 1 发（去掉 h2 D2D） | `dsv41_hc_post_inplace` |
| E13 | `ar_hc_post_fold`（hc_post 折进 AR pubred 尾） | `HCPOST_EPI` (ON) + `FUSE_C` | 2 发 → 1 发 | `ferrite_p2p_ar_v5_hcpost` |

### 1.2 MoE 半链

| # | EAGER 融合 | gate（默认） | 折叠了什么 | kernel |
|---|---|---|---|---|
| E14 | `gemm_bf16_fp8x2`（gate + 共享 w1/w3 同发） | `MIX_GATE` (**OFF**) | 3 发 → 1 发 | `dsv41_gemm_bf16_fp8x2` |
| E15 | `gemv_bf16_route`（gate GEMV 尾跑 route_topk） | `ROUTE_FUSE` (ON) | 2 发 → 1 发 | `ferrite_gemv_bf16_v2_route`@3386 |
| E16 | `gateup_fuse`（expert gate/up + swiglu 折进 epilogue） | `GATEUP_FUSE` (ON) | 3 发 → 1 发 | `expert_gate_up_fp4_batched` |
| E17 | `down_fuse`（expert down + reduce 单发） | `DOWN_FUSE` (ON) | 2 发 → 1 发 | `expert_down_reduce_fp4_batched` |
| E18 | `add_epi`（共享专家 merge 折进 AR store 尾） | `ADD_EPI` (ON) | 省 `ferrite_add` | `all_reduce_inplace_add` |
| E19 | `sh_exp_mx2`（共享 w1\|w3 单发） | `SH_EXP_MX2` (ON) | 2 发 → 1 发 | `gemm_fp8_mx2` |
| E20 | `swiglu_q`（swiglu + fp8 emit 单发） | `SWIGLU_Q` (ON) | 2 发 → 1 发 | `dsv41_swiglu_limit_q` |
| E21 | `moe_epi_add`（共享 w2 + add 折进 epilogue） | `MOE_EPI_ADD` (**OFF**) | 2 发 → 1 发 | `dsv41_gemm_fp8_mx_add` |
| E22 | `sh_pair`（共享三段一体 + grid barrier） | `SH_PAIR` (**OFF**) | 3 发 → 1 发 | `dsv41_gemm_fp8_sh_pair`@6737 |
| E23 | `compress_fuse`（compressor state+pool+commit 单发） | `COMPRESS_FUSE` (ON) | 3 发 → 1 发 | `dsv41_compressor_fused`@7759 |

### 1.3 侧流重叠（不减 launch 数，减关键路径）

| # | EAGER | gate | 作用 |
|---|---|---|---|
| E24 | `dual_chain`（kv 链整体挪到 side stream 2） | `DUAL_CHAIN` (ON) | kv norm+rope 与 q 链重叠 |
| E25 | `compress_side`（compressor 4 发挪到 side stream 3） | `COMPRESS_SIDE` (ON) | 与 q/kv 链重叠 |
| E26 | `moe_dual`（共享专家半链挪到 side stream 2） | `MOE_DUAL` (ON) | 与 routed 链重叠 |

**EAGER 的融合哲学（一句话）**：**把「相邻、同输入、无跨块依赖」的 kernel 折成一发，
再把「有跨块依赖但可并行」的整段挪到 side stream 被主链遮蔽。**

---

## 2. verify `layer_rows()` 的现状（代码级）

> 来源：`layer_rows()`@8558 / `attention_rows()`@8846 / `moe_rows()`@10816 /
> `collapse_norm_rows()`@8438 / `hc_post_rows()`@8510 / `shared_expert_mrows()`@11422。

### 2.1 hc 链：**10 发/层**（EAGER ~2 发）

`layer_rows` 的每个 block（attn + ffn）跑的是**原始分离链**（`HC_VERIFY_FUSE` **默认 OFF**）：

```
每 block 5 发：hc_mixes(1) + hc_collapse(1) + norm_rows(1) + hc_post(1) + memcpy_d2d(1)
每层 2 block = 10 发  ⇒ 40 层 = 400 发/步  ⇒ 2.96ms @ 53 GB/s（全表最低带宽）
```

树里**已有的** m 行融合（都在 `FUSE_B1`/`FUSE_C` 语义下，但被 `HC_VERIFY_FUSE` 双重把关）：

| 融合 | 落点 | 现状 |
|---|---|---|
| A1-a `hc_collapse_norm(rows=m)` | `collapse_norm_rows`@8457 | 默认 OFF；**`truncate=false` 修复已落地**（2026-09-12） |
| A1-b `hc_post_inplace_rows(rows=m)` | `hc_post_rows`@8519 | 默认 OFF；需 `.so` 有 `supports_hc_post_inplace_rows` |
| A2 `hc_mixes_auto(rows=m)`（= `hc_front_split`） | `layer_rows`@8630 | 默认 OFF（`HC_FRONT_ROWS`）；`hc_front_split` 本身 rows-agnostic |
| AR fold `all_reduce_inplace_hcpost_rows` | `ar_hc_post_fold_rows`@4461 | 默认 OFF（`VERIFY_AR_FOLD`）；**依赖 A1**（`hc_verify_fuse()`） |

**关键历史**：`hc_verify_fuse` 默认被改成 OFF 的原因是——它第一次让 `DSV41_BF16_TRUNCATE`
作用于 verify（`hc_collapse_norm` 是唯一携带该 gate 的形态），叠加 P0 系列后破坏零拉丁基线
（bisect 到 `050c7fd`）。**修复 = verify 调用点硬编码 `truncate=false`**，已落地。⇒ A1 现在可以重新 A/B。

### 2.2 attention：**逐行循环**（EAGER 是单行融合）

`attention_rows` 的算子选择：

| 环节 | EAGER | verify 现状 | 差距 |
|---|---|---|---|
| wq_a + wkv | `lin2`（1 发） | `quant_rows` + `proj_mrows`×2（3 发） | **无 mrows2**（B） |
| q norm | `lin_rope_norm` 前置 | `norm_rows`（1 发，已 m 行） | 仅差「是否折进 wq_b」 |
| wq_b + rope | `lin_rope`（1 发） | `proj_mrows` + `apply_rope`（默认逐行；`VERIFY_ROPE_MROWS` 可 mrows） | **无 mrows_rope**（B） |
| kv norm + rope | `rmsnorm_rope`（1 发） | `norm_rows` + `apply_rope(rows=m)`（2 发） | **无 mrows rmsnorm_rope**（B） |
| ring + window | `ring_win_fuse_ph`（1 发/行） | `ring_append` + `window_idxs`（**2 发/行**） | **已有核未接线**（A） |
| sparse attn + o-rope + fp8 | `sparse_attn_orope` | `sparse_attn_orope` **逐行**（`VERIFY_OROPE` 默认 ON） | 已对齐；只差 b·m 单发（A，生产 decline） |
| wo_a | `gemm_fp8_mx` per group | `wo_a_grouped_fp8`（1 发，m 行） | verify 侧更好 |
| wo_a 的 fp8 emit | `gemm_fp8_mx_q` epilogue | `quant_fp8` 逐行 | 差 5 发（B） |
| wo_b | `gemm_fp8_mx_f32`（直读 f32） | `quant_fp8` 逐行 + `proj_mrows` | **无 mrows_f32**（B） |

**注意 `VERIFY_OROPE` 已默认 ON**——即 **o-rope+quant 融合已经在 verify 用上了**（逐行调用，
但 kernel 是同一个 `sparse_attn_orope`）。用户清单里的第 2 项**已完成**。

### 2.3 MoE / head / compressor

| 环节 | EAGER | verify 现状 | 差距 |
|---|---|---|---|
| gate GEMV | `gemv_bf16_route`（gate+route 1 发） | 逐行 `gemv_bf16` + `route_topk`（6 发） | `GATE_MROWS` (A) + **无 mrows route**（B） |
| gate + 共享 w1/w3 | `gemm_bf16_fp8x2`（1 发） | 无（gate 与共享分开） | MIX_GATE 无 m 行版 |
| 共享专家 | 逐 token（m=1） | 逐行 5 发 ×5 行 = 25 发/层 | `SH_EXP_MROWS`(A) / `SH_EXP_FUSED`(A) |
| routed gate/up | `gateup_fuse` | `expert_gate_up_fp4_batched`（m 行） | ✅ 已对齐 |
| routed down | `down_fuse` | `expert_down_reduce_fp4_batched`（m 行） | ✅ 已对齐 |
| 共享 merge | `add_epi`（折进 AR） | `add_inplace_raw`（1 发）+ 独立 AR | **无 add-in-AR**（B，小） |
| indexer | 单行 | `indexer_rows_one` 逐行（~5 发/行） | `INDEXER_MROWS`(A) front |
| compressor | `compress_fuse`（1 发） | `compress_proj_rows` 逐行 lin_f32×2 + `compress_row` 逐行 pool+commit | `COMPRESSOR_MROWS`(A) |
| head | 单行 `gemv_bf16` | 逐行 `gemv_bf16`（v1 序） | `VERIFY_HEAD_MROWS`(A)（v1 折，位等价） |
| 侧流重叠 | dual_chain / moe_dual / compress_side | **无** | 全新（C，难） |

---

## 3. 迁移可行性：逐融合分类

### 3.1 Category A — 代码已在树里，只差 flag（**零 kernel 工作**）

| 融合 | gate | verify 调用点 | 前置条件 | 位等价？ |
|---|---|---|---|---|
| **hc_collapse_norm** | `DSV41_HC_VERIFY_FUSE=1` | `collapse_norm_rows`@8457 | `FUSE_B1`(ON) + `truncate=false`（已修） | ✅ 逐语句 superset |
| **hc_post_inplace_rows** | 同上 | `hc_post_rows`@8519 | `.so` 有 `hc_post_inplace_rows` | ✅ 逐线程列独立 |
| **hc_front_split（A2）** | `DSV41_HC_FRONT_ROWS=1` | `layer_rows`@8630 | `HC_TAIL_SPLIT`(ON) + `.so` 有 `hc_front_split` | ✅ rows-agnostic |
| **AR fold** | `DSV41_VERIFY_AR_FOLD=1` | `ar_hc_post_fold_rows`@4461 | **`HC_VERIFY_FUSE=1`** + `fuse_c` + `.so` 符号 | ✅ 同 `__fmaf_rn` 序 |
| **sparse_attn_orope** | `DSV41_VERIFY_OROPE` (已 ON) | `attention_rows`@9329 | — | ❌ 与 triple 非位等价（设计上对齐 EAGER） |
| **indexer front mrows** | `DSV41_INDEXER_MROWS=1` | `indexer_rows_m`@9144 | front 位等价已论证 | ✅ |
| **compressor mrows** | `DSV41_COMPRESSOR_MROWS=1` | `compress_rows_fused`@9110 | `seqlen==1` 逐位复现 | ✅ |
| **gate mrows** | `DSV41_GATE_MROWS=1` | `row_fold_gate`@10852 | v2 序位等价（`tests_gate_mrows.cu`） | ✅ |
| **共享专家 mrows** | `DSV41_SH_EXP_MROWS=1` | `shared_expert_mrows`@11503 | 4 步位等价 | ✅ |
| **共享专家三段一体（M=1）** | `DSV41_SH_EXP_FUSED=1` + `DSV41_SH_PAIR=1` | `shared_expert_mrows`@11444 | `.so` 有 `sh_pair`；**grid barrier 死锁风险** | ✅（声称） |
| **head v1 mrows** | `DSV41_VERIFY_HEAD_MROWS=1` | `step_rows_inner`@5773 | v1 序位等价 | ✅ |
| **q rope mrows** | `DSV41_VERIFY_ROPE_MROWS=1` | `attention_rows`@8986 | 行独立 | ✅ |
| **attention b·m 单发** | `DSV41_ATTN_MROWS=1` | `attention_rows`@9164 | **生产恒 decline**（见 §5-R4） | ✅（门开时） |

### 3.2 Category B — 需要**新 kernel**（EAGER 有、verify 无 m 行态）

| # | 缺的核 | 对应 EAGER | verify 现在的发数 | 新核设计 | 位等价论证 |
|---|---|---|---|---|---|
| B1 | `dsv41_gemm_fp8_mrows_rope` | `gemm_fp8_mx_rope`(E3) | `proj_mrows` + `apply_rope_mrows` = 2 发/层 | `gemm_fp8_mrows` 的 epilogue 加 rope 相位（每行自己的 pos） | 需新写 C1–C6（rope 只动尾部 rd 维） |
| B2 | `dsv41_gemm_fp8_mrows_norm_rope` | `lin_rope_norm`(E2) | `norm_rows` + `proj_mrows` + `apply_rope` = 3 发/层 | 同上 + prologue 做 rmsnorm | 同上（prologue = `rmsnorm_q` 逐字） |
| B3 | `dsv41_gemm_fp8_mrows2` | `lin2`(E1) | `quant_rows` + `proj_mrows`×2 = 3 发/层 | `gemm_fp8_mrows` 双 family（wq_a + wkv 同激活） | `mx2` 契约的 m 行推广 |
| B4 | `dsv41_rmsnorm_rope_mrows` | `rmsnorm_rope`(E4) | `norm_rows` + `apply_rope(m)` = 2 发/层 | `rmsnorm_rope` 加 rows 维 | 逐行独立 |
| B5 | `ferrite_gemv_bf16_v2_mrows_route` | `gemv_bf16_route`(E15) | `gemv_bf16_v2_mrows` + `route_topk` = 2 发/层 | mrows gate 的 last-block 选举跑 route | route epilogue 的 m 行推广 |
| B6 | `dsv41_gemm_fp8_mrows_f32`（wo_b） | `gemm_fp8_mx_f32`(E8) | `quant_fp8` 逐行 + `proj_mrows` = 6 发/层 | mrows GEMV 直读 f32 激活 | 略（非位等价，更准） |
| B7 | AR + shared merge（add_epi m 行） | `add_epi`(E18) | `add_inplace_raw` + AR = 2 发/层 | `all_reduce_inplace_add` 的 m 行版 | 逐行独立 |

### 3.3 Category C — 侧流重叠（不减 launch，减关键路径）

EAGER 的 `dual_chain` / `compress_side` / `moe_dual` 在 verify **完全没有对应**。
verify 的 `layer_rows` 全程单流（`step_rows_sync` 里无 fork/join）。这是**最大的架构级缺口**，
但也是**风险最高**（涉及 `.so` 的 side-stream 原语、m 行块的 stream 顺序、CUDA graph 捕获）。
**建议放在 Category A/B 全部 A/B 之后**，单独立项。

---

## 4. 预期性能：逐融合的 launch 与 ms

### 4.1 口径

- **每发成本**：verify 6224 发 ≈ 50% 提交（2.9µs）+ 49% 每发最小执行（3.3µs）
  ⇒ **省一发 ≈ 省 6.2µs**（`verify-ms-breakdown §2`）。保守起见下表用 **3.3µs**（只算执行半，
  与 `verify-operator-optimization-list` 的口径一致）。
- **ms 节省分两栏**：`launch 账`（可确定，纯计数）与 `ms 账`（设计口径，**未实测**）。
- verify 基线：**37.31 ms / 6224 发 / 14.09 GB**（m=5）。逐族账见 §4.3。

### 4.2 逐融合

| 融合 | 现状发/层 | 迁移后发/层 | 省发/步（40 层） | launch 账 | ms 账（设计） | 状态 |
|---|---:|---:|---:|---:|---:|---|
| **A1** hc_collapse_norm + hc_post_inplace | 10 | 6 | **−160** | −0.53ms | −1.3~1.7 | 代码就位 |
| **A2** hc_front_split | 6 | 4 | **−80** | −0.26ms | （含上） | 代码就位 |
| **AR fold** | 4 | 2 | **−80** | −0.26ms | −0.24 | 代码就位（依赖 A1） |
| ring_win_fuse 接线（C1） | 2/行×5=10 | 1/行×5=5 | **−200** | −0.66ms | −0.5~0.8 | 核已在，需接线 |
| **共享专家 mrows** | 25 | 5 | **−800** | −2.64ms | −3~5 | 代码就位 |
| **gate mrows** | 5 | 1 | **−160** | −0.53ms | −2.0 | 代码就位 |
| **indexer front mrows** | ~5/行×8 层 | ~1/行 | **−128** | −0.42ms | −1.0~1.5 | 代码就位 |
| **compressor mrows** | 4/源层×5 | 1 | **−48** | −0.16ms | −0.2~0.3 | 代码就位 |
| **head v1 mrows** | 5 | 1 | **−4**（head 1 次/步） | −0.01ms | −0.7~0.9 | 代码就位 |
| **q rope mrows** | 5 | 1 | **−160** | −0.53ms | −0.1~0.2 | 代码就位 |
| **sh_exp_fused（M=1）** | 25 | 7 | **−720** | −2.38ms | −1.8 | 代码就位（grid barrier） |
| **A 类小计** | | | **−2540** | **−8.4ms** | **−11~15** | |
| B1 `mrows_rope`（wq_b） | 2 | 1 | **−40** | −0.13ms | −0.4~0.7 | **新核** |
| B2 `mrows_norm_rope` | 3 | 1 | **−80** | −0.26ms | −0.8~1.2 | **新核**（依赖 B1） |
| B3 `mrows2`（wq_a+wkv） | 3 | 2 | **−40** | −0.13ms | −0.3~0.6 | **新核** |
| B4 `rmsnorm_rope_mrows` | 2 | 1 | **−40** | −0.13ms | −0.2~0.4 | **新核** |
| B5 `mrows_route`（gate） | 2 | 1 | **−40** | −0.13ms | −0.3~0.6 | **新核**（依赖 gate mrows） |
| B6 `mrows_f32`（wo_b） | 6 | 1 | **−200** | −0.66ms | −0.6~1.0 | **新核** |
| B7 AR+merge | 2 | 1 | **−40** | −0.13ms | −0.2~0.4 | **新核** |
| **B 类小计** | | | **−480** | **−1.6ms** | **−2.8~4.9** | |
| **A+B 合计** | | | **−3020** | **−10.0ms** | **−14~20** | |

**读法**：
- **A 类是「零 kernel、纯 flag/接线」**：口径 **−8.4ms（launch 账）~ −15ms（ms 账）**。
- **B 类每条要 1–3 人日**，合计再 **−1.6ms（launch）~ −5ms（ms）**。
- 与 `verify-architecture-floor` 的 L1→L3 台阶吻合：**37.3 → ~22ms（保守）/ ~11ms（目标）**。

### 4.3 与已有账本的对账（避免重复计数）

`verify-ms-breakdown §1` 的 6224 发逐族：

| 族 | ms | 发数 | 本方案动的 |
|---|---:|---:|---|
| routed experts | 8.30 | 400 | 不在范围（tcgen05 换核） |
| shared expert | 10.40 | 1000 | **SH_EXP_MROWS/FUSED（−800）** |
| head | 1.12 | 10 | **HEAD_MROWS（−8，但字节 5×→1×）** |
| 投影族 | 3.70 | 2000 | **B1/B2/B3（rope/norm 折进投影）** |
| attention | 2.80 | 880 | **ring_win_fuse（−200）+ ATTN_MROWS（blocked）** |
| hc 链 | 2.96 | 400 | **A1+A2+AR fold（−320）** |
| MoE gate | 3.44 | 200 | **GATE_MROWS（−160）+ B5（−40）** |
| indexer | 2.50 | 230 | **INDEXER_MROWS（−128）** |
| AR v5 | 1.40 | 240 | **B7（−80）** |
| compressor | 0.55 | 80 | **COMPRESSOR_MROWS（−48）** |
| engram | 0.42 | 24 | — |
| norm/quant | 0.20 | 680 | **rope/quant 折进上列各族** |

⇒ 投影族的 2000 发里，本方案通过 B1/B2 折掉 wq_b 的 rope/norm 相位；
但**「层内 m 合并」**（每层 4 投影 → 1~2 发）是另一项独立工作（`verify-family-fusion` W6）。

---

## 5. 正确的 gate 组合（verify 级融合的完整启用方案）

### 5.1 启用顺序（4 波）

**Wave 0 · 前置（无条件）**
```
# 只验证、不改行为：
DSV41_HC_VERIFY_FUSE=0   # 保持基线，先跑一次确认 6224 发 / 37.31ms
```

**Wave 1 · hc 链（最大单块，依赖链清晰）**
```
DSV41_HC_VERIFY_FUSE=1    # A1: collapse_norm + post_inplace_rows
DSV41_HC_FRONT_ROWS=1     # A2: hc_front_split 到 verify
DSV41_VERIFY_AR_FOLD=1    # AR fold（必须 A1 先 ON）
# 隐含保留：FUSE_B1=1(默认) FUSE_C=1(默认) HC_TAIL_SPLIT=1(默认)
# 验收：零拉丁 + faults=0 + hc 链 400→80 发
```

**Wave 2 · 逐族 mrows（互相独立，可单独 A/B）**
```
DSV41_SH_EXP_MROWS=1      # 共享专家 25→5 发/层
DSV41_GATE_MROWS=1        # gate 5→1 发/层
DSV41_INDEXER_MROWS=1     # indexer front
DSV41_COMPRESSOR_MROWS=1  # compressor
DSV41_VERIFY_HEAD_MROWS=1 # head v1 折（⚠️ 不与 SWALLOW_STEP 同开）
DSV41_VERIFY_ROPE_MROWS=1 # q rope
# 验收：逐族 launch 计数 + 四段文本
```

**Wave 3 · 三段一体 + ring/win 接线**
```
DSV41_SH_EXP_FUSED=1 DSV41_SH_PAIR=1   # 共享三段一体（grid barrier，单独 A/B）
# 新接线（不是 flag）：attention_rows 用 ring_win_fuse 替 ring_append+window_idxs
```

**Wave 4 · 新 kernel（B 类，逐个 parity 后翻）**
```
B1 mrows_rope  → 新 gate DSV41_VERIFY_PROJ_ROPE_MROWS=1
B2 mrows_norm_rope → 依赖 B1
B3 mrows2      → DSV41_VERIFY_PROJ_MROWS2=1
B4 rmsnorm_rope_mrows → DSV41_VERIFY_NR_MROWS=1
B5 mrows_route → DSV41_VERIFY_GATE_ROUTE=1
B6 mrows_f32   → DSV41_VERIFY_WOB_MROWS_F32=1
B7 AR+merge    → DSV41_VERIFY_ADD_EPI=1
```

### 5.2 互斥 / 依赖矩阵（**硬约束**）

| 组合 | 关系 | 原因 |
|---|---|---|
| `VERIFY_AR_FOLD=1` × `HC_VERIFY_FUSE=0` | **无效** | `ar_hc_post_fold_rows` 自身 gate 含 `hc_verify_fuse()` |
| `HC_FRONT_ROWS=1` × `FUSE_B1=0` | 半失效 | front 的 EARLY 半即 collapse，`FUSE_B1=0` 时 `norm_w` 为 null → decline |
| `SH_EXP_FUSED` × `MIX_GATE` | **互斥** | `sh_pair` 的 phase 1 要读自己的 `xq`；`sh_via_mixed` 会破坏 |
| `ATTN_MROWS` × `COMPRESSOR_MROWS` | **互斥** | 见 §5.3 |
| `VERIFY_HEAD_MROWS` × `SWALLOW_STEP` | **危险** | m=6 与 `argmax_rows` 死锁（`192ae83`） |
| `HC_FRONT_ROWS` × `HC_DL_SIDE` | 需核验 | 两条都动 `hc_front_split` 的 side stream |

### 5.3 关键：`COMPRESSOR_MROWS` 与 `ATTN_MROWS` 的取舍

两者**同时开会拒绝**（`attention_rows`@9165 的 decline）：
`COMPRESSOR_MROWS` 把 compressor 提前 hoist 成块级一发，live counter 变块末值，
而 `ATTN_MROWS` 需要逐行 clen snapshot（设备侧）。**只能选一个**：

- **选 `COMPRESSOR_MROWS`**：compressor −48 发，但 attention 保持逐行（880 发）。
- **选 `ATTN_MROWS`**：attention 880→~160 发（**收益更大**），但 compressor 保持逐行。
- **生产推荐 `ATTN_MROWS=0`（因为它在 `world>1` 和 `pos+m-1>=win` 时恒 decline）**，
  即**先吃 compressor**；attention 的 b·m 要等 `row_pitch` 尾参 + ring/window 合核（§6-R4）。

---

## 6. 风险与 A/B

| # | 风险 | 触发 | 影响 | 应对 |
|---|---|---|---|---|
| **R1** | **BF16_TRUNCATE 漏进 verify** | `HC_VERIFY_FUSE=1` | 破坏零拉丁基线（历史事故 `050c7fd`） | **已修**：`collapse_norm_rows` / `hc_mixes_auto` 的 verify 调用点硬编码 `truncate=false`；A/B 时**同时确认 `DSV41_BF16_TRUNCATE` 的两种取值** |
| **R2** | **grid barrier 死锁** | `SH_EXP_FUSED=1` + `SH_PAIR=1` | launcher 的 grid > co_res 时挂死 | 沿用 `dsv41_gemm_fp8_sh_pair` 的 `co_res_cached` 硬 cap；先跑 M=1 parity 再 A/B |
| **R3** | **数值不等价（新 K 序）** | B1/B2/B3/B6 | 文本变化 | **B 类每个都要 py 级 parity case**（`tests_dsv41_*.cu` 扩 m 行）；`gemm_fp8_mx_rope` 的 mrows 版是**新程序**，不能声称位等价 |
| **R4** | **b·m 的因果序**（defect #1/#2） | `ATTN_MROWS=1` + `world>1` 或环回绕 | **静默错答案** | 保持 `world==1` + `pos+m-1<win` 的 decline；长上下文/TP8 要 `row_pitch` 尾参 + ring/window 块内 r 升序合核（**另立项**） |
| **R5** | **per-row clen 快照缺失** | `COMPRESSOR_MROWS` + 逐行 reader | 读到块末计数器（未来） | 已由 `clen_rows_r` / `latent_rows_r` 快照解决；A/B 时验 `idx_lens` |
| **R6** | **收益高估** | 所有 ms 账 | 设计口径 ≠ 实测（历史「全 mrows 开只 −1.21ms」） | **先做一次 nsys**（按 kernel 名聚合，数 4 个数）；launch 账（−3020 发）比 ms 账可靠；以 **launch 计数**为首要判据 |
| **R7** | **gate 误接 / `.so` 不同源** | 全部 | 两臂都测旧路径（本仓 #1 陷阱） | 每个新核 `supports_*` 符号探测 + 一次性 warning；`.so` 与 Rust 同源重建（`build.sh` BUILD_ID） |
| **R8** | **head mrows × argmax 死锁** | `VERIFY_HEAD_MROWS=1` + `SWALLOW_STEP` | `argmax_rows` 与 rows=6 | 两者不同开（`192ae83`） |

### 6.1 每波统一验收判据

1. **逐位**：m 行融合核的 row r 输出 == M=1 核的 row r 输出（byte-exact，B 类做不到的必须显式标「数值改动」）；
2. **launch 数**：nsys 计数下降到设计值；
3. **文本**：四段（Paris/Tokyo/1+1/静夜思）正文逐字正确 + `faults=0`；
4. **A/B**：同会话、同二进制、背靠背（`scripts/dsv41_serve_ab.sh`），报 p50 + faults；
5. **回退**：任一 gate OFF / 老 `.so` ⇒ 走既有路径，输出与今天逐位相同。

---

## 7. 影响范围与工作量

| 文件 | 改动 |
|---|---|
| `kernels/cuda/dsv41_kernels.cu` | 新 `dsv41_gemm_fp8_mrows_rope` / `_norm_rope` / `_fp8_mrows2` / `rmsnorm_rope_mrows` / `gemm_fp8_mrows_f32` |
| `kernels/cuda/ferrite_kernels.cu` | `ferrite_gemv_bf16_v2_mrows_route` |
| `kernels/cuda/dsv41_glue.cu` | AR add 的 m 行 epilogue（B7） |
| `crates/ferrite-models/src/dsv41/device.rs` | 7 个新 launcher + `supports_*` 探测 |
| `crates/ferrite-models/src/dsv41/chain_dev.rs` | `attention_rows` 的 ring_win_fuse 接线；B 类调用点；新 gate |
| `crates/ferrite-dsv41/tests/` | 每核的 m 行 parity case |

**工作量**：
- Wave 1（hc 链）：**0.5 人日**（纯 flag A/B）+ 0.5 人日验零拉丁
- Wave 2（6 个 mrows flag）：**1 人日**（纯 A/B）
- Wave 3（SH_PAIR + ring/win 接线）：**1.5 人日**
- Wave 4（7 个新核）：**8–12 人日**（B1/B2 各 2，B3/B4/B5/B6/B7 各 1）
- **合计 ≈ 11–15 人日**；Wave 1+2（≈ −8.4ms launch 账）**1.5 人日内可拿**。

---

## 8. 关键架构决策（供后续引用）

1. **EAGER 的融合是 m=1 的；verify 的折叠是 m 行的**——两者共享的是**融合意图**，不是 kernel。
   直接「照搬 EAGER 调用」在 verify 会退化成逐行 m 发（更慢）；正确做法是**给 m 行核加融合相位**。
2. **最大的钱在「已有 m 行核 + flag OFF」**（Category A：hc 链、shared、gate、indexer、compressor、head、rope）。
   这批**零 kernel 工作**，应先全部 A/B 吃掉，再谈新核。
3. **`sparse_attn_orope` 已迁移完成**（`VERIFY_OROPE` 默认 ON）——不要重复投资。
4. **ring_win_fuse 是「核已有、verify 未接线」的漏网之鱼**——`attention_rows` 每行还是 2 发
   （`ring_append` + `window_idxs`），换成 `ring_win_fuse` 每行 1 发，**−200 发/步、零新核**。
   这是本方案发现的**最低成本的单点收益**。
5. **gate 组合有三条硬互斥**（AR-fold↔A1、SH_PAIR↔MIX_GATE、ATTN_MROWS↔COMPRESSOR_MROWS），
   启用顺序必须按 §5.2 的依赖矩阵走，否则「gate ON 但什么都没变」。
6. **收益判据以 launch 计数为准**：ms 账是设计口径（本仓历史上 flag 兑现度接近 0）；
   launch 计数是可 nsys 验证的硬账。

---

*工部 · 只读分析 + 本文档（唯一产出），未执行任何 GPU 命令、未改动任何源码。*
*所有 ms/launch 数均标了来源（实测/代码计数/设计推算）；**未实测项集中在 §4.2 的 ms 账与 §6-R6**。*
*代码引用基于工作树 HEAD `69d3f43` + 未提交的 `verify-operator-optimization-list.md`；行号以函数名为准。*
