# accept 1.214 → 3 的差距分析（根因 · 路径排序 · 策略）

> 工部 · 2026-09-12 · **只读分析，未改动任何源码、未执行 GPU 命令；本文件为唯一产出**
> 输入：`accept-first-strategy.md`（本仓已有一版结论）、`draft-accept-boost.md`、`dspark-correctness-chain.md`、
> `final-400-config.md`、`swallow-step-400-necessity.md`；全部代码事实给出 `文件:行号`。
> 结论与 `accept-first-strategy.md` 的 §1.3 有**实质分歧**（见 §2.4），请以本文件的臂对照表为判定入口。

---

## 0. TL;DR（六条）

1. **口径先钉死**：`accept = mean-k = 每步被接受的 draft 数`；`tok/step = mean-k + 1`。
   400 = `(1+mean-k) × 1000/step_ms` ⇒ **accept 3 @10ms / accept 2.2 @8ms**。三张表的数字
   （1.022 / 1.214 / 0.898）**未标注 arm 与直方图**，跨 arm 比较存在整体错 1 格的风险（§1）。

2. **【根因 A · 最高置信】legacy 臂的 `seed` 内容比它的相位新一档。**
   `draft_forward(token, pos)` 的 tap 取自**本步** `step_dev`（内容 = 位置 `pos` 的隐藏），
   而 seed 落在相位 `pos-1`（`dspark_dev.rs:1779`），块行在 `pos+r`（`1861`，默认 `kv_pos=pos`），
   accept 链以 `pos+1` 为判据 —— **几何整体对齐官方 `start_pos = pos-1`，只有 seed 的内容超前 1**。
   代码自证：`carry_kept_tap` 的注释写明 seed 要的是「**one before the swallow round's own counter**」
   （`chain_dev.rs:7326-7327`），而 SWALLOW 臂正是靠**上一轮携带的 tap** 来满足它。

3. **【根因 B · 最高置信】`DSV41_SEED_POS` 是「第三个、内部不自洽的混合臂」，不是官方臂。**
   它只搬了 **seed 相位**（`1779`）与 **kv 基址**（`1861`），三处没搬：
   **query/o 的 RoPE 基址**（`1823` / `1961` 仍是 `pos`）、**块 row-0 的 token**（仍是 anchor 的 `token`，
   官方要求是下一位的 `next`）、**accept 链**（drafts 变成对 `pos+2..` 的提案，判据却仍是 `pos+1` 的 `next`）。
   所以 **0.898 不能用来否定「seed 相位」假说** —— 它否定的只是一个半成品臂。

4. **【修正既有判词】`SEED_POS` 补 2 行 ≠ 官方。**
   `accept-first-strategy.md` §1.3 主张"把 `1823`/`1961` 的 `pos` 换成 `kv_pos` 即得官方臂"。
   补齐后仍有 **row-0 token** 与 **accept 链**两处错位（§2.4）。真正等于官方的是 **`DSV41_SEED_ALIGN`**
   （调用约定 `draft_forward(next, pos+1)`，`chain_dev.rs:7269`），或保持 legacy 几何的
   **`SWALLOW_STEP`**（tap 传递）。**两者的 accept 都从未测过。**

5. **【sglang 不是算法差异】** 同一结构：一块并行 draft（噪声行）+ 顺序 markov（`run_markov_block`，
   每 step `base_logits + w2(w1[prev])` → **argmax**，`greedy_step_sampler`）；贪心下你我都是 argmax。
   4.1× 的原始差里，**块长占了大部分**：sglang `gamma=7`/verify 8 行（`dspark_config.py:17`）vs 我们 bs=5。
   换算成 per-token 命中率：**p ≈ 0.86（sglang）vs 0.56（ferrite）≈ 1.5×**。
   真正的差异是「**几何正确 + draft/verify 同源数值**」，不是采样或链深。

6. **上限**：均匀 p 下 `mean-k = Σ_{k=1..5} p^k`；p=0.83→3.0，p=0.86→**3.16**（复刻 sglang 的 p），p=0.95→4.3。
   对确定性文本（出师表），瓶颈**不在 MTP head 容量**（同一 head 在 sglang 达 0.86），**在对齐**。

---

## 1. 口径（不钉死则 1.6× 是假缺口）

| 量 | 定义 | 代码出处 |
|---|---|---|
| `k_acc` | 本步被接受的 draft 数（0..=5） | `chain_dev.rs:152` |
| `mean-k` | `Σ k_acc / steps` | `serve.rs:587` |
| `tok/step` | `(Σ k_acc + steps)/steps = mean-k + 1` | `serve.rs:588` |

已测三组（`dspark-correctness-chain.md:1334-1343`）：

| 组合 | 数字 | 判定 |
|---|---|---|
| 基线（无杠杆） | 1.022 | 基准 |
| **P0-3+P1-5** | **1.214** | 当前最佳 |
| +SEED_POS+DRAFT_ATTN+TAP_BF16 | 0.898 | 退化（**3 gate 捆绑，不可归因**） |

**两个必须补的口径**：
- **arm 名**：legacy / lazy / aligned / swallowed 的块布局不同 ⇒ `k_acc` 的物理含义差一格
  （`chain_dev.rs:1711-1718` 的三行表）。1.214 是哪个 arm 未记录。
- **k_acc 直方图**：`draft-accept-boost.md §0.6` 已发现"文档 mean-k 0.833 vs 直方图 0.677，差恰好 36/232"。
  直方图决定"缺的是首 token 还是尾部"，直接决定下面的排序。

---

## 2. 差距的根因（三层，按置信度）

### 2.1 官方几何（先把参照物钉死）

`ref_inference/model.py` 的 decode 步（`i = start_pos`）：

| 成分 | 官方 | 行号 |
|---|---|---|
| tap | 层 **INPUT** 的 `h.mean(hc)`（`main_hiddens.append` 在 `layer()` **之前**） | `1264-1267` |
| seed（main KV） | `main_x=main_norm(main_proj(tap))` → `wkv`，RoPE **`freqs_cis[i]`**，槽 `i%win` | `1039-1042`, `1065` |
| 块行（q 与 kv 同一基址） | `freqs_cis[i+1 : i+1+bs]` ⇒ 行 r 在 **`i+1+r`** | `1055`, `1059/1061/1068` |
| row-0 的 token | `output_ids`（**下一位的 token**，不是 anchor 自己） | `1131-1133` |
| 候选集 | `arange(min(win, i+1)) ++ win+arange(bs)` | `1021-1029` |

**关键**：官方把「tap 的位置」与「块的行起点」分开了 —— **seed 在 `i`，块在 `i+1`**。
ferrite 的 `pos` 是"当前步正在处理的 token 的位置"（`emitted[i]` 在 `pos+1+i`，`chain_dev.rs`）。

### 2.2 四个臂的对照表（本文件的判定入口）

`T` = 本步 tap（内容位置 `pos`）；`T'` = 上一轮携带的 tap（内容位置 `pos-1`）。

| 臂 | 调用 | seed 相位/内容 | 块 kv | q/o | row-0 token | accept 链 | 自洽 | =官方？ |
|---|---|---|---|---|---|---|---|---|
| **legacy（默认，1.214）** | `f(token,pos)` | `pos-1` / **`T@pos`** ✗ | `pos+r` | `pos+r` | `token@pos` | drafts[j] vs `next`/verify[j-1] | ✅ | **内容超前 1** |
| **SEED_POS（0.898）** | `f(token,pos)` | `pos` / `T@pos` ✓ | `pos+1+r` | **`pos+r`** ✗ | **`token@pos`** ✗ | **仍是 `pos+1` 判据** ✗ | ❌ 三处 | ❌ 半成品 |
| **SEED_ALIGN** | `f(next,pos+1)` | `pos` / `T@pos` ✓ | `pos+1+r` | `pos+1+r` ✓ | `next@pos+1` ✓ | 索引对齐（6 行块） | ✅ | **✅** |
| **SWALLOW** | `f(token,pos)` + `T'` | `pos-1` / **`T'@pos-1`** ✓ | `pos+r` | `pos+r` | `token@pos` | 索引对齐（6 行块） | ✅ | **✅** |

代码出处：`seed_pos`/`kv_pos` `dspark_dev.rs:1779/1861`；`rope_queries` `1823`、`rope_queries_inv` `1961`、
行位置 `3226-3234`（`pos + r`，无 `+1`）；`win_rows` `3463-3477`；SEED_ALIGN 调用 `chain_dev.rs:7269`；
SWALLOW 的 tap 携带 `chain_dev.rs:7480-7484` + `carry_kept_tap` `7999-8020`。

### 2.3 为什么 legacy 会停在 1.214（根因 A）

把 legacy 映射到官方：**官方 `start_pos = pos-1`**（因为 row-0 的 token 在官方是"下一位"，
而 ferrite 传的是 anchor 自己 ⇒ `pos = i+1`）。于是：

- seed 应装 **`h[pos-1]`**（内容）+ 相位 `pos-1`；ferrite 装的是 `h[pos]` ⇒ **整条 ring 的内容统一超前 1 格**
  （每一步 `seed_window(s, p-1)` 写的是 `tap@p`，`dspark_dev.rs:1779-1780`）。
- 代码自己承认这一点：`carry_kept_tap` 的调用注释 ——
  "`k_acc - 1` sits at `pos + k_acc`, i.e. **one before the swallow round's own counter — the position its seed needs**"
  （`chain_dev.rs:7325-7327`）。
- 现象吻合：k_acc 直方图 `{0:55,1:13,…}` ⇒ **64% 步首 token 就被拒**。MTP head 的 row-0 输出
  恰恰是"用 (target hidden, token) 预测下一个 token"——hidden 与 token 的配对差一位，
  head 的输入分布就错位，首 token 命中率被压到 ~0.36（另一处记录为 0.60，见 `chain_dev.rs:7055` 的注释，
  两个数本身说明口径/arm 未钉死）。
- **为什么拖到 accept ~1.2 才暴露**：ring 是"整体超前 1"的刚性平移，块内相对距离没变，
  所以 attention 不是随机而是系统性偏移 ⇒ 文本仍正确（draft 不影响 committed 文本），accept 只是被压。

### 2.4 为什么 SEED_POS 退化（根因 B）——并对既有判词提出修正

`DSV41_SEED_POS` 的设计意图是"把 seed 挪到 `pos`、window 配套、块 kv 挪到 `pos+1`"
（`dspark_dev.rs:241-282` 的三站点注释）。已搬了 2 个站点（+3 站点的 window/win_rows），**漏了 3 个**：

1. **q/o 的 RoPE 基址**：`rope_queries(q, pos)`（`1823`）、`rope_queries_inv(o, pos)`（`1961`）
   仍在 `pos`，而 kv 在 `pos+1+r` ⇒ **query 比自己的 KV 早一个位置**（读"未来"的 key）。
   官方是 q/kv/o **共用同一个 `freqs_cis` 切片**（`model.py:1055 → 1059/1061/1068`）。
2. **块 row-0 的 token**：仍是 `token`（`ids[0]=t0`，`dspark_dev.rs:1152`），而"块在 `pos+1`"要求它是 `next`。
3. **accept 链**：`drafts()` 取的是 markov 采样结果（row r → `ids[r+1]`），
   row r 在 `pos+1+r` ⇒ `drafts[0]` 是对 **`pos+2`** 的提案，而 `dspark_spec_step` 仍拿它比 `next`（`pos+1` 的 token）。

**对 `accept-first-strategy.md` §1.3 的修正**：该文主张"补 §1 的 2 行即得官方臂（表格里标 ✅/✅）"。
按上表，补齐 q/o 后仍有 (2)(3) 两处错位 ⇒ **补齐后仍是错臂**，一次 GPU A/B 会得到"又是退化"的结论，
从而误判"相位不是主因"。**正确的判定实验是 SEED_ALIGN（`draft_forward(next,pos+1)`）与 SWALLOW，
而不是继续修 SEED_POS。**

> 另注：SEED_ALIGN 的 `win_rows` 用 `min(win-1, ·)`（`3465-3467`），官方是 `min(win, start_pos+1)`；
> 即"整窗口少一行"（最老的一行被丢掉）。这一行也值得在同一次 A/B 里核对（1 行改动）。

### 2.5 第二层：draft 与 verify 的数值域（中置信）

已确认的事实链（决定"单侧对齐 = 降 accept"）：
`BF16_TRUNCATE`（只挪 verify）→ accept 1.080→0.820；`+DRAFT_BF16_DOMAIN`（draft 的 head/MoE 同挪）→ **1.214**。

| 量 | 官方域 | draft 现状 | verify 现状 | gate | 判定 |
|---|---|---|---|---|---|
| tap → main_h | bf16 | f32（`dspark_dev.rs:871` round-trip） | f32 | `TAP_BF16` | **未隔离测** |
| MoE 激活 | e4m3 | **同 kernel / 同 gate**（`dspark_dev.rs:2241`） | 同 | `EXPERT_ACT_E4M3` | ✅ 已闭环（F1 的代码事实已不成立） |
| head `normed` / MoE `xn` | bf16 | f32（`2946` / `2144` round-trip） | — | `DRAFT_BF16_DOMAIN` | ✅ 已测（+19% 的来源之一） |
| attention `o` / `wo` | bf16 | f32（`2006`/`2071`/`2093`） | — | `DRAFT_BF16_DOMAIN` / `DRAFT_ATTN_BF16` | ⚠️ 后者的**独立效果未测** |
| **KV** | `act_quant(e4m3, block=32)`（`model.py:1042/1062`） | **无 act_quant，全 f32** | backbone ring 也是 f32（`chain_dev.rs:62`） | 无 | ⚠️ 同向偏离 ⇒ 只在"成对"时有意义 |

**结论**：`0.898` 那一行是"1 个错臂 + 2 个未隔离 gate"的捆绑，**它给出的"保持 OFF"结论没有实验支撑**。

### 2.6 第三层：结构（低置信，长尾）

draft 的 attention 是手写独立路径（ring + `sparse_attn` + 5 个 kernel 体，`dspark_dev.rs` 3600 行），
与 verify 的 DSA 路径（`chain_dev.rs` 14k 行）**不是同一段代码**。即使相位与域都对齐，
仍有 ~1e-3 级路径差异。sglang 把这条差异用"同一个 backend"消灭了（`dspark_config.py:30`），
ferrite 没有。这是"到 0.86 的最后一段"最可能的剩余阻力，但**不是当前 1.214→3 的主因**。

---

## 3. sglang 的 accept ~5 是怎么来的（回答方向 2）

**不是算法差异**，逐项核对（`sglang-pr-mig/python/sglang/.../speculative/dspark_components/`）：

| 维度 | sglang | ferrite | 差异？ |
|---|---|---|---|
| draft 链深 | **一块并行**：`query_token_num` 行（行 0 = anchor，其余 mask），一次 forward | 同（`ids=[t0,noise×4]`） | ❌ |
| 序列化部分 | markov：`for step: step_logits = base_logits[:,step] + w2(w1[prev]); prev = argmax` | 同（`dspark_markov_head_kernel`） | ❌ |
| 采样（贪心） | `greedy_step_sampler = torch.argmax` | argmax（float→key 单调映射） | ❌ |
| 采样（非贪心） | Gumbel（`SampleStepTokens`，`greedy_mask = top_k<=1`） | 只做贪心 | 无关（当前是贪心） |
| 块长 | `gamma = num_draft_tokens-1 = 7`，verify 8 行 | `bs=5`，verify 5/6 行 | **✅ 主因之一** |
| draft/verify 数值 | 同一 backend / 同一量化配置 / 同一 KV 池 | 两条手写路径 | **✅ 主因之一** |
| draft 的 KV 来源 | `TargetHiddenKvInjector` 把 target hidden 写进**真实 KV 池**（融合 norm+rope），几何天然是官方那套 | 自建 f32 ring | ✅ 几何项 |
| 调度 | confidence head + planner（动态 gamma） | 无 | 提 tok/s，不提 accept |

数值化（`accept-first-strategy.md §2.1` 已算过，此处复核并保留）：

| 实现 | 块长 | tok/step | mean accepted | per-token p |
|---|---|---|---|---|
| sglang | 7 | ~4.99 | ~3.99 | **≈0.86** |
| ferrite | 5 | 2.214 | 1.214 | **≈0.56** |

⇒ **原始比 4.1× 里约 1.6× 是块长口径**；真正要补的是 per-token 的 0.56→0.86。

---

## 4. 理论上限（回答方向 3）

模型：`mean-k = Σ_{k=1..5} Π_{j≤k} p_j`（p 为逐位命中率；接受链在首个失败处 break）。

| p | mean-k | tok/step | @10ms tok/s | @12ms | @7.5ms（地板） |
|---|---|---|---|---|---|
| 0.56（现状） | 1.21 | 2.21 | 221 | 184 | 295 |
| 0.75 | ~2.4 | 3.4 | 340 | 283 | 453 |
| **0.83** | **~3.0** | 4.0 | **400 ✓** | 333 | 533 |
| **0.86（sglang 级）** | **~3.16** | 4.16 | 416 | 347 | 555 |
| 0.95 | ~4.3 | 5.3 | 530 | 442 | 707 |

**上限的答案**：
- **绝对上限 = 5**（5 个 draft 全中；需要 p→1，不可达）。
- **现实上限 ≈ 3.16**：把 per-token 命中率复刻到 sglang 的 0.86，而 sglang 在 `gamma=7` 上就是这个水平
  —— 这是"同一 MTP head"的**外部可达性证明**。
- "MTP 3 层近似 44 层"**不是瓶颈的证明**：head 的输入是**目标层的真实 hidden**（不是自己跑 44 层），
  所以它做的是"用 (hidden@i, token@i+1) 预测 i+2"这件被专门训练过的近邻任务；
  0.56 与 0.86 的差距只能来自**输入配对/相位**，不能来自"层数少"。
- 因此"p=0.83 ⇒ accept 3"与用户的校准（"预测 5 个能对 3 个"）在数值上完全一致 ——
  **目标应改写为 p: 0.56 → 0.83（mean-k 1.214 → 3.0）**。

---

## 5. 剩余提升路径（排序，含成本/风险/证伪条件）

| # | 动作 | 预期 accept | 成本 | 风险 | 证伪条件 |
|---|---|---|---|---|---|
| **S0** | **口径钉死**：同一 arm 重测 `基线`/`P0-3+P1-5`，打印 arm 名 + **完整 k_acc 直方图** | 校准 ±20% | 0.2 人日 | 无 | — |
| **S1a** | **`DSV41_SEED_ALIGN=1` 的 accept A/B**（现成调用约定 `draft_forward(next,pos+1)`，官方几何；顺带核对 `win_rows` 的 `win-1`） | **1.4~2.5** | 0.3 人日 | 低（需 `spec_primed` 引导；与 SWALLOW 互斥） | ≤1.214 ⇒ 根因 A 降级 |
| **S1b** | **`DSV41_SWALLOW_STEP=1` 的 accept A/B**（tap 传递 = 根因 A 的另一半；同时 −4.55ms） | **1.4~2.5** | 0.3 人日 | 中（新 arm，需 6 行 verify 形状池） | ≤1.214 ⇒ 同上 |
| **S2** | **单变量隔离** `TAP_BF16` / `DRAFT_ATTN_BF16`（在 S1 的最好基线上各一轮） | ±0.1~0.2 each | 0.3 人日 | 低 | 皆 ≤0 ⇒ dtype 假设降级 |
| **S3** | **逐 stage 探针**：`unit_dump` 从"整块"扩到"逐 stage"（定位第一个 rel-L2>1e-2 的 stage） | 诊断 | 1~2 人日 | 低 | — |
| **S4** | **成对域对齐**：draft 的 KV `act_quant` ↔ backbone ring 同步量化；或 attention 内部 bf16 ↔ verify 同量 | +0.2~0.6 | 1~2 人日 | 中 | 成对后无提升 ⇒ 转 S5 |
| **S5** | **draft/verify 共用 attention kernel**（结构层消灭长尾） | +0.3~0.8 | 3~5 人日 | 高 | — |

不投（明确负面清单）：
- **继续修 SEED_POS**：三处不自洽（§2.4）⇒ 预期仍是退化。
- **单独给 draft 加 `act_quant`**：draft 与 backbone ring 同向偏离（f32）⇒ 单侧 = 去相关 = 降 accept。
- **e2m1→e4m3 当新杠杆**：draft 已与 backbone 同源（`dspark_dev.rs:2241`）。
- **伪问题**：draft 的 engram 注入（target 层 37/38/39 **不是** engram 层——engram 在 1/14；
  MTP 层（40/41/42）也没有 engram）、hc sinkhorn（20 已同官方 `config.rs:278`）、temperature（贪心）。

累计路径（保守/乐观）：`1.214 → S1a/S1b → 1.4/2.5 → S2 → 1.5/2.7 → S4 → 1.7/3.3 → S5 → 2.0/3.5`。

---

## 6. 策略建议：「攻 accept」还是「攻步时」

**数学（`tok/s = (1+mean-k) × 1000/step_ms`）**：

| 组合 | tok/s | 400 需要 |
|---|---|---|
| accept 1.214 + 步时 7.5ms（**架构地板**） | **295** | ✗ 不可达 |
| accept 1.214 + 步时 10ms | 221 | ✗ |
| accept 3.0 + 步时 10ms | **400** | ✓ |
| accept 3.16（上限）+ 步时 10ms | 416 | ✓ |
| accept 2.2 + 步时 8ms | 400 | ✓（要触地板） |
| accept 3.16 + 步时 33ms（当前 lazy） | 126 | ✗ |

**判定**：
1. **纯攻步时不可达**：accept 1.214 即使把步时压到架构地板 7.5ms 也只有 **295 tok/s**（差 1.36×）。
   所以"接受 accept ~1.2，转攻步时"这条路线在数学上**已出局**（除非把地板证伪）。
2. **纯攻 accept 也不够**：accept 3.16 在 33ms 步时下只有 126 tok/s。
3. **必须两条腿，但排序上 accept 优先**：
   - accept 的**边际收益高**：每 +1 mean-k ≈ +83 tok/s（@12ms）/ +100（@10ms）；
   - accept 的**修补成本极低**：S1a/S1b 都是**现成 gate 的一次 A/B**（0.6 人日，零代码风险），
     而步时侧（batched + swallow + mrows + tcgen05）已在推进且**已触地板**（7.4~9.2ms）；
   - **SWALLOW 是唯一的"一举两得"项**：它是步时项（−4.55ms）**同时**是根因 A 的修复项（tap 传递）⇒
     应把它与 batched-400 的步时测试**合并成同一次 A/B**，一次拿到 accept 与 step 两个结果。
4. **落点预测**：accept 修到 2.5~3.2 + 步时 10~12ms ⇒ **290~390 tok/s**；400 需要
   `accept ≥ 3.0 且步时 ≤ 10ms`（或 `accept ≥ 2.2 且步时 ≤ 8ms`）。
   **400 的现实性完全取决于 S1 一次 A/B 的结果** —— 因此建议：
   > **先花 0.6 人日跑 S0+S1a+S1b，再决定是否投 S4/S5 与步时第二轮。**

---

## 7. 必须钉死的未知（否则上面的区间会偏）

| # | 未知 | 影响 | 取证 |
|---|---|---|---|
| U1 | `1.214` / `1.022` / `0.898` 分别是哪个 **arm** | 跨 arm 比较差一格 token，全部结论要重算 | S0 打印 arm 名 |
| U2 | **k_acc 直方图**的真实分布（首 token 拒绝率） | 决定"修配对"还是"修尾部" | S0 |
| U3 | SEED_ALIGN 的 `win_rows` 是否应含 `min(win,·)` 全窗口 | ±1 行 KV 的候选集 | S1a 同轮核对 |
| U4 | draft 的 KV 与 backbone ring 的 f32/e4m3 差（"成对"的边界在哪） | S4 的成败 | S3 探针 |

---

## 8. 一句话交付

> **差距不在 MTP head 的层数，也不在算法（sglang 与我们同构，块长差占了 4.1× 中的 1.6×）。**
> **差距在"tap 的内容与 draft 几何的配对"：legacy 臂把 `h[pos]` 放进了 `pos-1` 的相位
> （整条 ring 统一超前一格，对应 64% 首 token 拒绝）；`SEED_POS` 是一个三处不自洽的混合臂，
> 它的 0.898 不能用来否定相位假说。**
> **等于官方几何的两个臂（`SEED_ALIGN` / `SWALLOW_STEP`）都已实现，accept 却从未测过。**
> **先测这两个（0.6 人日、零代码风险）；`SWALLOW` 同时兑现 −4.55ms ⇒ 是 accept 与步时的交汇点。**
> **accept 1.214 下即使步时触地板也只有 295 tok/s ⇒ 400 必须 accept ≥ 3.0（p ≥ 0.83）与步时 ≤ 10ms 同时成立。**

---

*工部 · 只读分析，未执行 GPU 命令、未改动任何源码；本文件为唯一产出。*
*所有代码事实给出 `文件:行号`；与 `accept-first-strategy.md §1.3` 的分歧已显式标出（§2.4）。*
