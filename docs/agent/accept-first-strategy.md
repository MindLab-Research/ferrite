# accept 优先策略的执行计划（accept 1.214 → 3）

> 工部 · 2026-09-12 · **只读分析，未改动任何源码、未执行 GPU 命令；本文件为唯一产出**
> 缘起：`dspark-correctness-chain.md §1454`（arch-floor-insights）判决——
> `accept 停 1.2，任何 verify 优化都改变不了量级。第一优先级是 accept，不是 verify。`
> 校准：accept 1.214 → 需 5.54ms/步（低于 L5 地板 8~9ms ⇒ 物理不可达）；accept 3 → 10.0ms（L5 刚好）。
> 本文件回答：**为什么停在 1.214、sglang 凭什么 ~5、以及 1.214 → 3 的具体路径（每步预期 + 成本）**。

---

## 0. TL;DR（四条，按重要性）

1. **【代码级·高置信】`DSV41_SEED_POS` 臂存在 q/kv RoPE 相位不一致——这是 P0-1 co-fix 漏掉的第四个站点。**
   `win_rows` 与 `kv_pos` 被改成官方语义（`pos+1`），但 `rope_queries` / `rope_queries_inv` 仍用旧基址 `pos`。
   ⇒ SEED_POS 臂下 **draft 的 kv 行在 `pos+1+r`、q/o 行在 `pos+r`，相差整整一个位置**。
   官方参考用**同一个** `freqs_cis` 切片同时驱动 q/kv/o（`model.py:1055→1059/1061/1068`）。
   SEED_POS 的两次退化（079ffbaf 单开、10ba0e73 组合）由此得到单一解释。**修复 = 2 行。**

2. **【口径·高置信】`0.898` 那一行不是"追加杠杆退化"的证据，而是一个被污染的组合。**
   SEED_POS 的 q/kv 错位是一阶破坏，它把 `DRAFT_ATTN_BF16` / `TAP_BF16` 的信号淹掉了。
   这两个 gate 在修好的基线上**从未被单独 A/B 过**——现在的"应保持 OFF"结论无实验支撑。

3. **【结构性·高置信】accept 是 draft↔verify 的**一致性**度量，不是绝对正确性度量。**
   证据链：`BF16_TRUNCATE`（只把 verify 侧对齐官方）把 accept 从 1.080 **打到** 0.820（§964）；
   `+DRAFT_BF16_DOMAIN`（把 draft 侧也拉过去）回到 1.214（§1297）。
   ⇒ **单侧对齐 = 去相关 = accept 下降**。这一条决定后面每一步的排序方式（必须成对移动）。

4. **【口径·重要】"5 vs 1.2" 的原始比是 4.1×，换算成 per-token 命中率只有 ~1.5×。**
   见 §2.1：sglang gamma=7（p≈0.86）vs ferrite bs=5（p≈0.56）。
   在 bs=5 上复刻 sglang 的 p，**上限正好是 mean-k ≈ 3.16** —— 与用户校准的"上限 ~3"吻合。
   ⇒ 目标应写成 **p: 0.56 → 0.86（mean-k 1.214 → 3.16）**，而不是"1.2 → 5"。

---

## 1. 为什么 accept 只到 1.214

### 1.1 先立口径：accept 到底是什么

spec 一步 = draft 提 5 个 token → verify（m 行批前向）逐行判 → 接受前缀。
`drafts[j]` 被 `verify_out[j-1]`（或 `next`）判，**verdict 是两个数值路径 argmax 的相等性**：

```
accept ≈ P( argmax(draft路径(x)) == argmax(verify路径(x)) )
```

不是 `P(draft == truth)`。所以：

| 现象 | 机制 |
|---|---|
| `BF16_TRUNCATE=1`（verify 对齐官方）→ accept 1.080→0.820 | verify 挪了，draft 没挪 ⇒ 去相关 |
| `+DSV41_DRAFT_BF16_DOMAIN`（draft 的 head/MoE 也挪）→ 1.214 | 两侧同步 ⇒ 重新相关，且高于基线 |
| `+SEED_POS+DRAFT_ATTN+TAP_BF16` → 0.898 | **draft 又单侧挪了三次**（且 SEED_POS 本身破功，见 §1.3） |

**推论（本计划的总原则）**：任何"把 draft 对齐官方"的动作，必须问一句
"**verify 侧对应量在不在同一个域**"。不在，就先别做，或者成对做。

### 1.2 三层对齐的进度表（对照 `§933-946` 与代码实测）

| 组件 | 官方域 | ferrite draft 现状 | verify 现状 | gate | 判定 |
|---|---|---|---|---|---|
| tap → main_h | bf16 | f32（`dspark_dev.rs:871` 有 round-trip） | f32（层输出 f32） | `TAP_BF16` | 单侧对齐 ⇒ 风险项 |
| hc_pre/hc_front | bf16 | 截断（同 verify kernel） | 截断（`BF16_TRUNCATE`） | `BF16_TRUNCATE` | ✅ 已同步 |
| MoE 激活 | e4m3 | 同 kernel（`EXPERT_ACT_E4M3` 双刃） | 同 kernel | `EXPERT_ACT_E4M3` | ✅ 同源 |
| head `normed` | bf16 | f32（`2946` round-trip） | — | `DRAFT_BF16_DOMAIN` | ✅ 已测 |
| MoE `xn` | bf16 | f32（`2144` round-trip） | f32（backbone `xn`） | 同上 | ⚠️ 单侧 |
| attention `o` / `wo` | bf16 | f32（`2006`/`2071` / `2093`） | — | 同上 / `DRAFT_ATTN_BF16` | ⚠️ 未隔离测 |
| **attention 的 q / kv** | **bf16 + kv 走 `act_quant`** | **全 f32，无 act_quant** | backbone ring 也是 f32（`chain_dev.rs:62`） | 无 | ❌ **未做** |
| attention 累加域 | fp32 | fp32 | fp32 | — | ✅ |
| RoPE 相位 | `start_pos+1+r` | 见 §1.3 | — | `SEED_POS` | ❌ **破功** |

### 1.3 剩余差距 #1：SEED_POS 的 q/kv 相位不一致（**两行修复**）

**代码事实**（`crates/ferrite-models/src/dsv41/dspark_dev.rs`）：

```rust
1779:  let seed_pos = if seed_pos_fix() { pos } else { pos - 1 };
1780:  self.seed_window(s, seed_pos, slot_dev)?;          // seed 行
1823:  self.rope_queries(self.q.ptr as *mut f32, pos)?;   // ← q 用 pos
1861:  let kv_pos = if seed_pos_fix() { pos + 1 } else { pos };
1862:  self.rope_at(self.kv.ptr ..., kv_pos as i32, false)?; // ← kv 用 kv_pos
1961:  self.rope_queries_inv(self.o.ptr as *mut f32, pos)?;  // ← o 用 pos
3226-3234: rope_queries 内层 → 行 r 的位置 = pos + r          // ← 同样用 pos
```

**官方参考**（`ref_inference/model.py`）只有一个相位切片：

```
1055:  freqs_cis = self.freqs_cis[start_pos + seqlen : start_pos + seqlen + block_size]
1059:  apply_rotary_emb(q[..., -rd:], freqs_cis)      // q   ← 同一个 freqs_cis
1061:  apply_rotary_emb(kv[..., -rd:], freqs_cis)     // kv  ← 同一个 freqs_cis
1068:  apply_rotary_emb(o[..., -rd:], freqs_cis, True)// o   ← 同一个 freqs_cis
```

**结论**：`q`、`kv`、`o` 必须共享同一基址（`start_pos + 1 + r`）。
`SEED_POS` 臂把 kv 基址改成了 `pos+1`，却把 q/o 留在 `pos`。

| 臂 | seed | kv 行 | q/o 行 | 内部自洽 | vs 官方 |
|---|---|---|---|---|---|
| 默认 | `pos-1` ✗ | `pos+r` ✗ | `pos+r` ✗ | ✅ | 整体偏 1 |
| SEED_ALIGN | `pos-1` ✓ | `pos+r` | `pos+r` | ✅ | ✅（该臂 `pos` 已是 anchor+1） |
| **SEED_POS（现状）** | `pos` ✓ | **`pos+1+r`** | **`pos+r`** | ❌ **差 1** | 半对 |
| SEED_POS（补齐后） | `pos` ✓ | `pos+1+r` ✓ | `pos+1+r` ✓ | ✅ | ✅ |

**这解释了两次退化**：079ffbaf（只挪 seed，窗口还没配套）与 10ba0e73（挪了 seed+kv+window，q/o 没挪）。
两次都是"draft 的 attention 行列错位"——一阶破坏，足以压掉任何 dtype 对齐的收益。

**修复**：把 `1823` / `1961` 的 `pos` 换成 `kv_pos`：
在默认臂下 `kv_pos == pos` ⇒ **逐位等价，零风险**；在 SEED_POS 臂下补齐官方语义。

> 顺带证伪一条：**P3-9 的 oracle 修复（`dspark.rs` 的 `start_pos+seqlen+r`）对 serve 无因果**。
> 它是 host oracle，`dspark_parity` 用 device 自比（`§1107` 自述），而且 oracle 自己也**没有** kv 的 `act_quant`（见 §1.4）。
> 它不需要 GPU 验证——需要的是把它的**结论**落到 device 侧，也就是本节这两行。

### 1.4 剩余差距 #2：draft 的 KV 少了官方的 `act_quant`

**官方**（`model.py:1042` / `1062`）：

```
main_kv = self.kv_norm(self.wkv(main_x)); apply_rotary_emb(...)
act_quant(main_kv, fp8_block_size=32, scale_fmt="ue8m0", scale_dtype=e8m0, inplace=True)
```

`act_quant(inplace=True)` 的语义（`kernel.py:41-95`，逐行读过）：
`out_dtype = in_dtype if inplace`，kernel 写的是
`Cast(bf16, Cast(f32, Cast(fp8, clamp(x/s, ±448))) * s)` ——
**把值吸附到 e4m3 网格（block=32、power-of-2 scale），仍以 bf16 存储**。是量化，不是 dtype 转换。

**ferrite**：`seed_window`（`3154-3208`）与 `draft_attention`（`1825-1869`）产出的 `mk`/`kv` 全程 f32，
**没有任何 act_quant 站点**；`sparse_attn` 收的也是 `*const f32`（`device.rs:2484`）。
**且 host oracle 同样缺**（`dspark.rs:113-121`）⇒ parity 测试永远查不到这个缺口。

**但这一项必须谨慎**（§1.1 的原则）：`chain_dev.rs:62` 明写
`window ring, [window, head_dim] f32（the release stores it fp8; the ring is f32 in this increment）`
—— **backbone 的 ring 也是 f32**。draft 与 verify 在这里同向偏离 ⇒ 误差相关 ⇒ 这可能正是 1.2 还能站住的原因。
所以：**单独给 draft 加 act_quant 很可能降 accept**；正确形态是 draft KV **与** backbone ring 同时量化。
⇒ 排在 §3 的 S4，且必须先有 §3-S3 的定位证据。

### 1.5 已被排除/降级的嫌疑

| 项 | 判定 | 依据 |
|---|---|---|
| P0-2 temperature | 非阻塞 | `§1083`；checkpoint 无生成配置，两侧都 argmax |
| P0-4 wo_a 格式 | 证伪 | `§1090`；checkpoint 的 wo_a 确为 fp8，ferrite 处理正确 |
| `audit-ffi-args`（quant1 只量化一行） | 已修 | `1787`/`1828` 注释（D1 fix） |
| markov / confidence / 3 个 mtp block | 非瓶颈 | `draft-accept-boost §12`：与官方逐字同构 |
| verify 剩 1 mismatch（0.4%） | 非 accept 杠杆 | `draft-accept-boost §F4` |
| `DSV41_DIFF_EAGER` | **现有探针不覆盖 draft** | 它只比 spec vs eager 的 token（`chain_dev.rs:3423-3516`），不定位 draft 哪一层开始偏 |

---

## 2. sglang accept ~5 的机制

### 2.1 先把口径换算掉（一半的"4× 缺口"是块长）

- sglang：`DEFAULT_DSPARK_GAMMA = 7`（`dspark_config.py:17`），gamma+1 = 8 token/步。
  `383.7 tok/s × 13ms = 4.99 tok/step`（`verify-architecture-floor §7`）⇒ 含 anchor 共 ~5。
- ferrite：`config.rs:594 assert_eq!(c.dspark_block_size, 5)`，mean-k 1.214（accepted drafts）⇒ 2.214 tok/step。

按几何分布反解 per-token 命中率 p（截断在块长处）：

| 实现 | 块长 | tok/step | mean accepted | p |
|---|---|---|---|---|
| sglang | 7 draft | 4.99 | ~3.99 | **≈0.86** |
| ferrite | 5 draft | 2.21 | 1.214 | **≈0.56** |

**⇒ 原始比 4.1×（5 / 1.214），per-token 比只有 ~1.5×（0.86 / 0.56）。**
**⇒ ferrite 在 bs=5 上的天花板（复刻 sglang 的 p）≈ mean-k 3.16。** 这正是"上限 ~3"的出处。

### 2.2 sglang 的结构优势：draft 与 verify **共用同一套 kernel**

sglang 的 draft 走 SGLang 自己的模型后端（`dsv4` attention backend + fp8 权重 + bf16 激活 + fp8 KV cache，
`dspark_config.py:30 DSV4_DRAFT_ATTENTION_BACKEND = "dsv4"`）。
⇒ draft 与 verify 的**量化路径、attention backend、MoE、KV 域全部同源** ⇒ 数值误差 100% 相关 ⇒ accept 高。

ferrite 的 draft 是**手写独立路径**（`dspark_dev.rs`，175KB / 3600 行），
与 `chain_dev.rs`（14k 行）的 verify 各走各的 f32/quant 组合 ⇒ 误差去相关 ⇒ accept 塌。

**这解释了同一个 MTP head 的两个数字，也定义了本计划的目标形态：**
> 不是"把 draft 改成官方"，而是**"让 draft 的数值路径与 ferrite 自己的 verify 对齐"**。
> 官方只是这两条路径的公共参照物——当 verify 已经对齐官方时，两者等价。

### 2.3 我们缺的三件事（按结构层次）

1. **相位层**：draft 的 q/kv/o/seed 必须内部自洽且 = 官方（§1.3，2 行）。
2. **域层**：draft 的 attention 内部（q/kv/attn 输入输出）与 verify 对应量同域。
   ferrite draft 全 f32；verify 的对应量在 e4m3(KV 权重量化)/bf16 之间摇摆。§3-S3 先定位，再成对移动。
3. **结构层**：draft 的 attention 是"ring + sparse_attn + 5 个 kernel 体"，与 verify 的 DSA 路径**不是同一段代码**。
   这是长尾：即使域和相位都对齐，仍有 ~1e-3 级别的路径差异。**sglang 用同一个 backend 消灭了它，ferrite 没有。**

---

## 3. 执行计划（按 ROI 排序）

前提：每条都用**同一 arm、同一公式**测量，且守住红线
（零拉丁 + 出师表逐字 + `DSV41_DIFF_EAGER` 的 `[diff]` 一致）。

| # | 动作 | 预期 accept | 成本 | 风险 | 证伪条件 |
|---|---|---|---|---|---|
| **S0** | **口径钉死**：同一 arm 上重测 `基线` / `P0-3+P1-5` / `懒+批` 三组，记录 arm 名 + k_acc 直方图 | 校准 ±20% | 0.5 人日 | 无 | — |
| **S1** | **SEED_POS 的 q/o 基址补齐**（`1823`/`1961` 用 `kv_pos`），只开 `DSV41_SEED_POS=1` | **1.4 ~ 1.8** | **0.5 人日** | **低**（默认臂逐位等价） | 若仍 ≤1.214 ⇒ 相位不是主因，转 S3 |
| **S2** | **单变量 A/B**：在 S1 的修好基线上，分别单开 `TAP_BF16` / `DRAFT_ATTN_BF16`（各 1 轮） | ±0.1~0.2 each | 0.5 人日 | 低 | 若两项皆 ≤0 ⇒ "dtype 对齐"假设降级，全部预算转 S3/S5 |
| **S3** | **draft 侧逐层探针**：把 `unit_dump`（`unit_dump.rs` 已存在 + `DSV41_DSPARK_UNIT_INJECT`）从"整块对照"扩到"逐 stage 对照"，定位第一个 rel-L2 > 1e-2 的 stage | 诊断（不直接提分） | 2 人日 | 低 | — |
| **S4** | 按 S3 结果做**成对**对齐（候选：draft KV 的 act_quant + backbone ring 的 act_quant 同步；或 attn 内部 bf16 + verify 同量同步） | **+0.2 ~ 0.6** | 1~2 人日 | 中 | 若成对后仍无提升 ⇒ 结构层（§2.3-3）是主因 |
| **S5** | **draft 全链路 bf16 域**（q/kv/o/attn 累加之外的每一处）+ 与 verify 侧同步 | **+0.3 ~ 0.8** | 3~5 人日 | 高（触及 150+ kernel 的域约定） | — |
| **S6** | 若 S1~S5 后仍 < 2.5：转向**结构性怀疑**——tap 内容（层号/`mean(hc)`/位置）、markov 5 步的输入、confidence 门控 | — | 3+ 人日 | 高 | — |

### 3.1 累计路径（保守 / 乐观）

```
1.214
 └─ S1 (0.5d)  → 1.40 / 1.80
     └─ S2 (0.5d)  → 1.45 / 1.95        （±0.1~0.2，用来给人/给钱做判据）
         └─ S4 (1-2d) → 1.65 / 2.40
             └─ S5 (3-5d) → 1.95 / 3.10   ← 复刻 sglang 的 p ⇒ 上限 3.16
```

**S1 是唯一的"零风险高杠杆"项**：2 行代码、默认臂位等价、直接检验 arch-floor-insights 的"是数值 bug"判决。
**建议先跑 S0+S1+S2 三个（合计 1.5 人日）再决定 S4/S5 是否值得投。**

### 3.2 accept 提升的吞吐换算（步时 35ms / 10ms 两档）

| accept（mean-k） | tok/step | @35ms | @10ms |
|---|---|---|---|
| 1.214（今天） | 2.21 | 63 tok/s | — |
| 2.0 | 3.00 | 86 | — |
| **3.16（上限）** | **4.16** | 119 | **416 tok/s ✓** |

⇒ **accept 单独补到位只能到 ~120 tok/s；400 仍需步时 ≤10ms。**
arch-floor-insights 的判决据此应改写成两句：
`accept 优先（因为它是乘数且成本低）；但 accept 到 3.16 也只是把 400 从"绝无可能"变成"刚好可能"。`

---

## 4. 必须钉死的三个未知（否则上面区间会偏）

| # | 未知 | 影响 | 取证方式 |
|---|---|---|---|
| U1 | **1.214 是哪个 arm 的数字**（lazy / batched / aligned 的 accept 公式不同，见 `chain_dev.rs:7192-7214`） | 所有跨 arm 比较可能整体错 1 个 token 的位置 | S0：同一 arm 重测 + 打印 arm 名与 k_acc 直方图 |
| U2 | **draft 的 p 到底是多少**（现口径 mean-k 1.022 vs 文档 0.833，差 36/232；`draft-accept-boost §6`） | 目标值会偏 | S0：k_acc 直方图的完整分布 |
| U3 | **verify 侧每个量与官方域的差**（这决定"单侧 vs 成对"） | S4/S5 的成败 | S3 的逐 stage 探针 + backbone 侧同位置的 dump |

---

## 5. 一句话交付

> **accept 停在 1.214 的直接原因找到了：`DSV41_SEED_POS` 只搬了 seed / window / kv 三个站点，
> 漏了 q 与 o 的 RoPE 基址——draft 的 attention 在"官方语义"臂上 q 与 kv 相差一个位置。**
> **修它只要 2 行，且默认臂逐位等价。**
> 其后按"**成对对齐**"（draft 与 verify 同域）推进：1.214 → 1.8（0.5+0.5 人日）→ 2.4（+2 人日）→ 3.16（+5 人日，bs=5 的绝对上限）。
> **accept 不是能力上限问题——是相位 1 处 + 域 N 处，且域必须两侧同动。**

---

*工部 · 只读分析，未执行 GPU 命令、未改动任何源码；本文件为唯一产出。*
*所有代码事实均给出 `文件:行号`；所有推断项明确标注为推断（§1.4 / §3 区间）。*
