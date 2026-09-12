# draft 链（4.9ms）的 launch 与带宽账本

> 口径：400 tok/s 要求 **verify ≤5.5ms + draft ≤1ms + 主链吞掉（−6.15ms）**。
> 现状实测：`[dspark] steps=100 mean-k=0.520 tok/step=1.520 draft=4.99ms verify=38.85ms commit=0.19ms`
> （`dspark_verify.rs:540` 的日志口径，即 `DSV41_TIMING` 的 `draft=` 字段）。
> 方法：只读代码逐条计数（launch 数=代码精确）+ 从 shape 推字节（精确）+ 用仓库内已实测的
> 每核/每 launch 成本换算。**本机无 GPU（无 nvidia-smi / B300），所有「ms」均为推算**；
> 凡无仓库实测支撑的拆分已在 §6 标为「待验证」。
> 生产配置：dim 5120 / hd 512 / nh 64 / q_lora 1280 / o_lora 1024 / o_groups 8（hpg 8，ol_total 8192）
> / window 128 / hc 4 / vocab 129280 / moe_inter_dim 2304 / MTP：128 routed + 1 shared, topk 3
> / dspark_markov_rank 256 / n_target 3 / n_mtp_layers 3 / bs=dspark_block_size 5 / TP8。

---

## 0. 结论（TL;DR）

1. **launch 数 = 289–292（≈290）/draft_forward**，平均 **16.8µs/launch**（4.9ms ÷ 290）。
   仓库实测流式提交只有 **2.904µs/launch**（`graph_bench`），所以 4.9ms 里只有 ~0.85ms 是**发射开销**，
   剩下 ~3.9ms 是 **290 个串行小核各自的执行/依赖延迟**（每个 ~13µs，链完全串行：block s+1 依赖 block s）。
   ⇒ **draft 不是带宽瓶颈，也不是发射瓶颈，是「串行小核条数」瓶颈。**
2. **权重/激活字节 ≈ 2.6 GB/步**（代码精确）。其中 **head 1357MB + markov_head 662MB = 2019MB = 76%**。
   按 7.5TB/s 峰值算地板 ~0.35ms，按 GEMV 实际有效带宽（3–5TB/s）算 **0.5–0.9ms** —
   ⇒ **字节地板本身就已经压在 1ms 门口**，光靠「减字节」到不了 1ms，必须同时砍 launch 条数。
3. **launch 分布极不均**：MoE 占 150/290 = **52%**，其中**共享专家一家 78 条（27%）**，
   且它是**逐行**（5 行）跑同一份 4.4MB 权重 ⇒ 权重被读 5 遍。这是全表最肥、最机械的一刀。
4. 三个疑点的答案：head 是**字节大头但不是时间大头**（~0.3ms）；markov 的**逐步重读是真的**（5×132MB）；
   draft 的 attention **无法复用 verify 的投影**（`mtp.{s}.*` 与 `layers.{l}.*` 是不同权重）；
   3 个 block **可以图化**，但先要解掉 4 个 host 依赖。

---

## 【分析范围】

| 文件 | 取用内容 |
|---|---|
| `crates/ferrite-models/src/dsv41/dspark_dev.rs` | `draft_forward`(668) / `draft_attention`(1028) / `draft_moe`(1315) / `draft_head`(1722) / `seed_window` / `rope_at` / `ensure_idxs`（launch 计数依据） |
| `crates/ferrite-models/src/dsv41/device.rs` | launcher 签名（每次调用实际发几个 kernel） |
| `crates/ferrite-models/src/dsv41/weights.rs` | MTP 权重 shape / dtype / Shard（字节账本依据） |
| `crates/ferrite-models/src/dsv41/config.rs` | 生产几何（dim/vocab/inter/topk/mr…） |
| `kernels/cuda/dsv41_kernels.cu` `dsv41_glue.cu` | `dsv41_sparse_attn`（split+merge 两发）、`dsv41_dspark_markov_head`（逐步全词表扫描）、`head_gemv_bf16_mrows` |
| `docs/agent/dspark-verify-perf-plan.md` | §4 的前序分解与仓库实测基准（298µs 全词表 head、2.904µs/launch、0.411µs/node） |

---

## 1. launch 账本（289–292 条）

**统计口径**：只数 `draft_forward` 内的 kernel 发射。`Ok(false)` 回退臂（wo_a 未融合 = 40 条/block、
MoE 未批化 = 3×bs×topk 条/block）**不计入基线**（生产走 fused/batched）；`mrows` 门默认 OFF 的
**按 OFF 计**（即 per-row），因为「默认 OFF 是 house rule」。

### 1.1 一次性（13 条 + 2 次阻塞 H2D + 1 次阻塞 D2H）

| 调用点 | 行 | 次数 | 备注 |
|---|---|---|---|
| `upload_bytes_at(pos_base)` | 764 | 1–2 | **阻塞 H2D**（`ensure_pos_dev`；seed(pos-1)/queries(pos) 两值 ⇒ 2 次） |
| `upload_bytes_at(ids)` | 772 | 1 | **阻塞 H2D**（`(bs+1)*4` B） |
| `quant1(main_h)` | 642 | 1 | |
| `gemm_fp8_mx(main_proj)` | 643 | 1 | [5120, 15360] fp8 |
| `rmsnorm(main_norm)` | 654 | 1 | |
| `embed_expand_dev` | 781 | 1 | |
| `memcpy_d2d(pre_in←premix_init)` | 807 | 1 | |
| `hc_collapse`（head 前） | 1739 | 1 | |
| `rmsnorm(dspark_norm)` | 1748 | 1 | |
| `head_gemv_bf16_mrows` | 1793 | 1 | **已多行**（读 1 次全词表） |
| `dspark_markov_head` | 1833 | **5** | 逐 step，**串行依赖**（step s+1 读 `ids[s+1]`） |
| **小计** | | **13** (+2 H2D) | |

图外的 `drafts()`（`dspark_dev.rs:580`）另有 **1 次阻塞 D2H**（20B），但**阻塞整条链**的尾延迟，不计入 kernel 数。

### 1.2 每 block（92 条 × 3 block = 276 条）

#### (a) hc 链 + 残差 + memcpy —— 10 条/block → 30

| 调用点 | 行 | 次/block |
|---|---|---|
| `hc_mixes(attn)` | 1009 (via 816) | 1 |
| `hc_collapse` | 845 | 1 |
| `rmsnorm(attn_norm, rows=bs)` | 871 | 1 |
| `hc_post(attn)` | 891 | 1 |
| `memcpy_d2d(h ← h_out)` | 901 | 1 |
| `hc_mixes(ffn)` | 1009 (via 908) | 1 |
| `hc_collapse_norm(ffn)` | 917 | 1 |
| `hc_post(ffn)` | 936 | 1 |
| `memcpy_d2d(h ← h_out)` | 946 | 1 |
| `memcpy_d2d(pre_in ← pre_ffn)` | 956 | 1 |

#### (b) attention（`draft_attention`）—— 32 条/block → 96

| 调用点 | 行 | 次/block |
|---|---|---|
| `quant1(main_x)`（seed_window） | 1866 | 1 |
| `gemm_fp8_mx(wkv)`（seed） | 1867 | 1 |
| `rmsnorm(mk)`（seed） | 1878 | 1 |
| `apply_rope(mk)`（seed） | 1886 | 1 |
| `memcpy_d2d(→ ring slot)` | 1895 | 1 |
| `quant1(xn)` | 1061 | 1 |
| `gemm_fp8_mx(wq_a)` | 1062 | 1 |
| `rmsnorm(qr, rows=bs)` | 1073 | 1 |
| `quant1(qr)` | 1081 | 1 |
| `gemm_fp8_mx(wq_b)` | 1083 | 1 |
| `apply_rope(q)` **×bs** | 1097 (`rope_queries`) | **5** |
| `quant1(xn)` | 1102 | 1 |
| `gemm_fp8_mx(wkv)` | 1103 | 1 |
| `rmsnorm(kv, rows=bs)` | 1114 | 1 |
| `apply_rope(kv, rows=bs)` | 1125 | 1 |
| `memcpy_d2d(window → all_kv)` | 1156 / 1161+1168 | 1–2（wrap 时 2） |
| `memcpy_d2d(kv → all_kv)` | 1175 | 1 |
| `sparse_attn` = **split + merge 两发** | 1190 | **2** |
| `apply_rope_inv(o)` **×bs** | 1206 (`rope_queries_inv`) | **5** |
| `quant1(o)` | 1233 | 1 |
| `wo_a_grouped_fp8`（已融合） | 1237 | 1 |
| `quant1(wo)` | 1284 | 1 |
| `gemm_fp8_mx(wo_b)` | 1285 | 1 |

#### (c) MoE（`draft_moe`）—— 50 条/block → **150（占全链 52%）**

| 调用点 | 行 | 次/block | 门 |
|---|---|---|---|
| `gemv_bf16(gate)` **×bs** | 1335 | **5** | 无（逐行，M=1 bf16） |
| `route_topk` | 1344 | 1 | |
| `quant_fp4` | 1364 | 1 | |
| `expert_gate_up_fp4_batched` **×bs** | 1488 | **5** | `DSV41_DRAFT_MOE_MROWS` **默认 OFF** |
| `swiglu_limit_batched` **×bs** | 1519 | **5** | 同上（fused 时 0） |
| `expert_down_reduce_fp4_batched` **×bs** | 1528 | **5** | 同上 |
| `quant1(xn)`（共享专家） | 1633 | 1 | |
| 共享 `gemm_fp8_mx(w1)` **×bs** | 1641 | **5** | 无（逐行，**未 port MROWS**） |
| 共享 `gemm_fp8_mx(w3)` **×bs** | 1652 | **5** | 同上 |
| 共享 `swiglu_limit` **×bs** | 1663 | **5** | 同上 |
| 共享 `quant1(ex_act)` **×bs** | 1669 | **5** | 同上 |
| 共享 `gemm_fp8_mx(w2)` **×bs** | 1672 | **5** | 同上 |
| `add_inplace` | 1684 | 1 | |
| `all_reduce_inplace`（AR v5 = **1 kernel**） | 1711 | 1 | TP8 |

> **共享专家一家 26 条/block × 3 = 78 条 = 全链 27%**，且是逐行 ⇒ 权重读 5 遍。全表最肥的一刀。

### 1.3 汇总

| 段 | 条/block | ×3 | 占比 |
|---|---|---|---|
| MoE（routed 15 + 共享 26 + route/quant/add/AR 9） | 50 | **150** | **52%** |
| attention | 32 | **96** | 33% |
| hc + 残差 + memcpy | 10 | **30** | 10% |
| 一次性（含 markov 5） | — | **13** | 5% |
| **合计** | **92** | **289–292** | 100% |

**每 launch 平均 4.9ms / 290 = 16.8µs**；其中实测流式提交地板 2.904µs ⇒
**发射开销 ~0.85ms（17%），核执行/依赖延迟 ~4.05ms（83%）。**

---

## 2. 带宽账本（≈2.6 GB/步）

dtype 依 `weights.rs`：投影 = FP8（1B/elem + 4B/32 scale）、专家 = FP4（0.5B/elem）、
head = BF16、markov 对 = F32（`dev` 以 `as_f32()` 寻址）。激活全是 `bs×dim×4 = 100KB` 量级，忽略。

| 权重 | shape（全局/本地） | dtype | 单份字节 | 读次数 | 合计 |
|---|---|---|---|---|---|
| **head.weight** | [129280, 5120] Replicated | BF16 | **1357 MB** | **1**（mrows 后） | **1357 MB** |
| **markov_head.head.weight** | [129280, 256] Replicated | F32 | 132.4 MB | **5（每 step 全词表扫）** | **662 MB** |
| markov_head.embed.weight | [129280, 256] | F32 | 132.4 MB | 1 **行**（`er = embed[tok]`） | ~0 |
| main_proj | [5120, 15360] | FP8 | 78.6 MB | 1 | 78.6 MB |
| wq_b ×3 | [32768, 1280] | FP8 | 41.9 MB | 1/block | 125.8 MB |
| wo_b ×3 | [5120, 8192] | FP8 | 41.9 MB | 1/block | 125.8 MB |
| wo_a ×3 | [8192, 4096] | FP8 | 33.6 MB | 1/block（grouped） | 100.7 MB |
| wq_a ×3 | [1280, 5120] | FP8 | 6.55 MB | 1/block | 19.7 MB |
| wkv ×3 | [512, 5120] | FP8 | 2.62 MB | **2/block**（seed + kv） | 15.7 MB |
| gate ×3 | [128, 5120] | BF16 | 1.31 MB | 5 发/block **同权重** | 3.9 MB |
| routed experts ×3 | topk3 × [320/288, ⌈2560/…⌉] | FP4 | ~2.21 MB/专家/rank | **5 行各自 topk，各读 3 专家** | **~99 MB** |
| shared expert ×3 | w1/w3/w2 各 [288, 5120] | FP8 | 4.42 MB/block/rank | **5（逐行）** | **66.3 MB** |
| **合计** | | | | | **≈2655 MB** |

**关键读数：**
- **head + markov = 2019MB = 76%**。其余全部权重加起来只有 636MB。
- **markov 的重读是真的**：`dsv41_dspark_markov_head` 的 `for (v = ...; v < vocab; v += gridDim.x*nwarp)` 对
  **每个 step 完整扫 `markov_head[v*mr .. +mr]`**，5 个 step 串行且 `er = embed[ids[step]]` 每步都变
  ⇒ **无法缓存、无法跨步复用**，662MB 是硬读。
- **路由专家的多行化不减字节**（各行的 top-3 专家不同）：99MB 是 5 个 token 的真实工作量，
  `rows=bs` 只省 launch（15→3/block），不省字节（与 verify 计划 §2.5 的结论一致）。
- **共享专家逐行 = 权重读 5 遍**：66.3MB 里只有 13.3MB 是必要的，**53MB 是纯重复**。

**带宽地板**：2.655GB @ 7.5TB/s 峰值 = **0.35ms**；按 GEMV/专家的实际有效带宽（3–5TB/s）
= **0.53–0.89ms**。⇒ **即使把 launch 全砍掉，字节本身就在 1ms 上下**。这条决定了下限。

---

## 3. 可砍项（按 预期 ms × 把握 排序）

| # | 项 | 现状 | 优化后 | 预期收益 | 把握 | 依据 |
|---|---|---|---|---|---|---|
| 1 | **共享专家多行**（port `DSV41_SH_EXP_MROWS`，verify 侧已实现） | 26 条/block，66.3MB | 6 条/block，13.3MB | **−0.5 ~ −0.8ms** | **高**（verify 有现成 kernel + 门禁纪律） | 78→18 launch；53MB/步重复读 |
| 2 | **draft 图化**（`DSV41_DRAFT_GRAPH`） | 290 条裸流 | 1 次 replay | **−0.4 ~ −0.7ms** | 中（需先解 4 个 host 依赖） | 2.904µs→0.411µs/node（实测）；verify 图是现成配方 |
| 3 | **路由专家多行**（`DSV41_DRAFT_MOE_MROWS=1`，**代码已就位，待 A/B**） | 15 条/block | 3 条/block | **−0.2 ~ −0.35ms** | **高**（只差 A/B 门禁） | 45→9 launch；字节不变 |
| 4 | **rope 多行**（`rope_queries`/`_inv`：rows=bs·nh, step=1） | 10 条/block | 2 条/block | **−0.15 ~ −0.25ms** | 高（`apply_rope` 已支持 rows+step） | 30→6 launch；同 K 序，无重结合 |
| 5 | **`sparse_attn_orope`**（P1 kernel 已在 `device.rs:2203`） | 8 条/block（sparse 2 + rope_inv 5 + quant 1） | 1 条/block | **−0.2ms** | 中（需 bit-identity A/B） | 24→3 launch |
| 6 | **`hc_post_inplace` + 去 memcpy**（`device.rs:4374` 已有） | 3 memcpy/block | 0 | **−0.1 ~ −0.2ms** | 高 | 9 memcpy→0；混音缓冲 ping-pong 可去掉 `pre_in←pre_ffn` |
| 7 | **gate 多行**（bf16 GEMV，`head_gemv_bf16_mrows` 已是有先例 | 5 条/block | 1 条/block | −0.05 ~ −0.1ms | 中（n=128 太小，收益存疑） | 15→3 launch；1.31MB 权重少读 4 遍 |
| 8 | **head 词表切分**（1.36GB→170MB @ TP8） | 1 条，1357MB | 1 条，170MB | **−0.15 ~ −0.25ms** | 中（需跨 rank argmax） | 省 1.19GB；代价 5 行 × 跨 rank top-1 |
| 9 | **markov 词表切分 + 跨 rank argmax** | 5 条，662MB | 5 条，83MB | **−0.1 ~ −0.2ms** | 中-高（5 次串行交换） | 省 578MB；但 5 步串行 ⇒ 交换延迟吃掉一半 |
| 10 | markov 权重降精度（F32→BF16） | 662MB | 331MB | −0.05ms | 低（argmax 近邻风险） | 若 loss 可接受；**待验证** |
| 11 | head 与 verify 的 head 融合 | 2 次读 1.36GB | 1 次 | **0（不可行）** | — | verify 的 head 输入依赖 draft 的输出，**跨相位，不能同 launch** |

### 不该做的（避免过度优化）
- **别动 `memcpy_d2d(window→all_kv)` 的「按位置序」两段拷贝**（`win_rows` 的 `s0`）：
  代码注释明确这是 ring 环绕后的**正确性**要求（按 slot 序会重复/驱逐一行候选）。
- **别为省字节去改 expert 的 FP4**：99MB 是 5 个 token 的真实 top-3 工作量。
- **别指望「减字节」单独达标**：字节地板 0.35–0.89ms 已经贴着 1ms，必须与减 launch 同时做。

---

## 4. 三条路径（4.9ms → 1ms）

> 排序 = 可行性；三条**必须全做**才压到 ~1ms（单独任一条都不够）。

### Path 1 —— 机械搬运 verify 的既有优化（低风险，先做）
把 verify 侧已经落地/已验证的 kernel 搬到 draft：**共享专家 mrows**（#1）+ **路由专家 mrows A/B**（#3）
+ **rope 多行**（#4）+ **`sparse_attn_orope`**（#5）+ **`hc_post_inplace`/去 memcpy**（#6）+ gate 多行（#7）。

- **launch：290 → ~150**；字节：2655 → 2602MB（只省共享专家的 53MB）。
- **预期：4.9 → 3.2–3.7ms**（按每砍一条 launch 省 ~10–15µs 的边际延迟算，扣掉合并后大核自身的时间）。
- 成本 **低**／风险 **低**（全是已有 kernel + 每项单独 A/B 门）。
- **注意**：这一步**动不了** head+markov 的 2.0GB（76% 字节），所以时间降不下来多少 —— 它买的是「条数」。

### Path 2 —— draft 自己的 CUDA graph（最高杠杆，中风险）
3 个 block 的形状**完全固定**（bs=5 / 3 block / win 常量），device pos counter 已落地（半程），
`DSV41_VERIFY_GRAPH` 是**现成的捕获配方**（DRY → rollback → CAPTURE → REPLAY）。但要先解 **4 个 host 依赖**：

| # | host 依赖 | 位置 | 修法 |
|---|---|---|---|
| D1 | `upload_bytes_at(ids)` 每步阻塞 H2D | 772 | 改「图外 H2D 到 capture 录下的同一地址」（verify 图的 `ids_r` 纪律） |
| D2 | `seed_window` 的 `slot = pos % win` **主机算地址** | 1892 | 换 `Device::ring_append`（device counter 派生 slot；代码注释已点名） |
| D3 | `memcpy_d2d(window→all_kv)` 的 **size/branch 主机算**（`n_win`/`s0`，wrap 时 2 段） | 1152–1174 | 换一个「定长 win 行 + device 派生 n_win」的 copy kernel |
| D4 | `ensure_idxs` 的 H2D（host 分支） | 2085 | 用 device-side idxs 生成 kernel，或把捕获推迟到 `n_win` 稳定（pos ≥ win）之后 |

- **预期：−0.4 ~ −0.7ms**（290 条 × (2.904−0.411)µs ≈ 0.73ms；Path 1 后 150 条 ⇒ ~0.4ms）。
- 成本 **中**／风险 **中**（捕获期分配、per-request drop、AR epoch —— mega-graph 的坑已有记录）。
- **诚实提醒**：图只吃掉**节点间空隙**，不改核执行的 memory latency；
  「每 launch 16.8µs vs 提交地板 2.9µs」说明大头是**执行/依赖延迟**，所以图的上限没想象中大。

### Path 3 —— head + markov 的词表切分（字节杠杆，中-高风险）
唯一能砍掉那 76% 字节的路：head 1357→170MB、markov 662→83MB（TP8 各切 1/8）。
代价是**跨 rank argmax**：draft 必须在每个 rank 产出**全局相同**的 top-1（markov 是词表级采样），
所以 head 的 5 行与 markov 的 5 步各需一次 8-rank 的 top-1 交换。
- **预期：−0.25 ~ −0.45ms**（字节 2.0GB→0.25GB，但要付 10 次串行交换的延迟）。
- 成本 **中-高**／风险 **中-高**（跨 rank 协议 + 近邻 argmax 数值域，`argmax_pub` 曾与 AR 暂存冲突死锁过）。

### 合计落点

| 路径 | 累计 draft | 关键动作 |
|---|---|---|
| 现状 | **4.90ms** | 290 launch / 2.66GB |
| + Path 1 | **3.2–3.7ms** | launch 290→150 |
| + Path 2 | **2.6–3.2ms** | 150 launch 图化 |
| + Path 3 | **1.0–1.5ms** | 字节 2.6GB→0.9GB |

⇒ **三条全做落在 1.0–1.5ms；要稳进 ≤1ms，Path 2 必须拿到上界、Path 3 必须兑现。**
边际达标（与 verify 的 ≤5.5ms 同理），**没有单点决胜项**——这是本账本最重要的结论。

---

## 5. 给尚书省的三句话

1. **先做 Path 1 的共享专家 mrows**（#1）：78 条 launch + 53MB 重复读，**全表最高性价比**，
   且 verify 侧已实现的 `sh_exp_mrows` 就是模板，只差搬到 draft。
2. **Path 2（draft 图）是达标的关键路径**，但现在还不合法：4 个 host 依赖（D1–D4）必须先解，
   其中 **D2（`seed_window` 的主机 slot）** 是最硬的一处，`ring_append` 是现成解。
3. **Path 3 是唯一碰得到那 2.0GB（76%）字节的路**，但它引入跨 rank argmax —— 建议**先做 Path 1+2，
   再看实测是否已 <1.5ms 决定是否上 Path 3**，避免为一个 0.3ms 引入死锁级风险。

---

## 6. 待验证（本机无 GPU，未经本次实测）

1. **4.9ms 的三向拆分**（发射 ~0.85ms / 字节地板 ~0.5–0.9ms / 执行-依赖延迟 ~3.2ms）是**推算**；
   总时 4.9ms 与逐条 launch 计数（290）是精确的，但「每 launch 16.8µs 里多少是发射、多少是执行」
   需一次 nsys 实测（按 `DSV41_TIMING` + `nsys profile`；draft 段只有 ~290 节点，好定位）。
2. **Path 1 的边际收益**（−1.2 ~ −1.7ms）依赖「每砍一条 launch 省 10–15µs」这个假设；
   需 A/B 实测（共享专家 mrows 单开、路由专家 mrows 单开，各测 `draft=` 字段）。
3. **Path 2 图化的真实收益**：`graph_bench` 的 2.493µs/node 是**空核**测得；draft 的核平均 ~13µs，
   重叠率未知 ⇒ −0.4 ~ −0.7ms 是**上界估计**。
4. **markov 的 F32 读**：本报告按 `as_f32()` 推断 markov 对在 device 上是 F32（132.4MB/个）；
   若 checkpoint 是 BF16 且 kernel 走 `__half2` 解码，则字节减半、该项收益减半 —— 需查
   `load_tensor` 的 dtype 处理与 `.so` 里的符号签名确认。
5. **Path 3 的跨 rank argmax 延迟**（10 次交换，估 0.05–0.1ms）未实测。

---

*户部 · 基于 2026-09-12 仓库状态（HEAD `3128025` dspark wave2）。*
*所有「ms」为推算值；launch 计数与字节数为代码精确值。*
