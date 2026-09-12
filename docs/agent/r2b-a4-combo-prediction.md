# R2 + R2b + A4（+SH_PAIR）组合栈的吞吐预测 与 到 400 的差距

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 代码基线：HEAD `1b40889`（`crates/ferrite-models/src/dsv41/chain_dev.rs`、`kernels/cuda/ferrite_kernels.cu`，逐条 `file:line` 核对）。
> 输入：`dspark-correctness-chain.md`（R2 验证节 + nsys 族表）· `projection-family-optimization.md`（R2 的 7→2 发）·
> `ar-further-optimization.md`（A4/惊群）· `lazy-verify-optimization-path.md`（k_emit 税）· `400-fastest-path-roadmap.md`（口径）·
> `batched-400-v2-remaining-roi.md` · `swallow-step-400-necessity.md`。
> **本机无 GPU ⇒ 所有 ms 标了来源（实测 / 账本推算 / 结构推算）。**

---

## 0. 判决（先读六条 —— 其中三条修正任务前提）

1. **组合栈的落点 ≈ 85 tok/s（区间 84~87），不是 90~95。** 相对 R2 的 82.9 只有 **+2~4%**。
   理由见 §2~§5：R2b 的 launch 账被高估了 ~2×（indexer 只在 **8 层**，不是 40 层），
   A4 只能收「惊群份额」，而 22.5µs/AR 的主体是**对端到达延迟（数据依赖）**，不是轮询开销。

2. **🔴 任务前提「R2 的 82.9 tok/s 对应步时 ~29ms」不成立（口径混用）。**
   计数任务的实测是 `completion=130 / 1568ms`。29ms/步只有在 `k_emit≈2.4`（accept≈1.4）时才成立；
   若 accept=5（`k_emit=6`），步时是 **~71ms**。**「82.9 tok/s ↔ 29ms」这两个数不能同时为真**，
   这与仓库自己反复纠正的「22.56ms 是 accept≈1.1 的步时」是同一类错误（`400-fastest-path-roadmap §1`）。
   ⇒ 本文件**不用绝对步时反推**，改用 §3 的「实测 delta → µs/launch → launch 增量」自洽账。

3. **🔴 R2b 的真实节省是 56 发/行，不是 120 发/行。**
   - `norm_rows`（1 发/层）确实在**全 40 层**被跳过 ⇒ −40 发/行；
   - 但 indexer 的 `quant+proj+rope → lin_rope_norm`（−2 发/层）只发生在 **8 个 `index_source` 层**
     （`index_layers = [2,8,14,20,24,28,32,36]`，`dsv41_flash.json`） ⇒ **−16 发/行**。
   - 合计 **−56 发/行**（不是 3×40=120）。任务把 indexer 当成了 40 层全有。

4. **🔴 `SH_PAIR_M=1` 实测是中性偏负**（78.8 → 78.1，−0.9%，k_acc 不变）。
   组合栈里它**不应计为收益**；它是「verified-correct 的免费件」，不是加速件。

5. **A4 的正确期望是 −0.2~−1.0 ms/步（大概率 ~−0.5）**，且它的**真正价值是把 A1 探针变成可信实验**——
   只有 A1 的 `[ar-probe]` 分布才能把 22.5µs 拆成「真等待 / 真搬运 / nsys 伪影」。
   A4 本身不动那条数据依赖，因此**不会兑现 20.8% 里的主要部分**。

6. **400 @ accept 5 在 lazy 下数学不可能；batched（SWALLOW）是唯一路径。**（§6 论证）
   lazy 的地板 **41.4ms（145 tok/s，即使 c_row 达到 EAGER 的 6.15ms）**；400 要求步时 ≤15ms ⇒
   `c_row ≤ 1.75ms/行` = 比 EAGER 的 m=1 GEMV 快 **3.5×** —— 在「每行一遍权重 + 每行一套 launch」的
   结构下不可能。SWALLOW(m=6) 把「一次权重读 + 一次 per-step 族」摊到 6 行 ⇒ ~12.5ms ⇒ 480 tok/s。

---

## 1. 计数任务的实测口径（尽量钉死）

| 臂 | 实测 | 来源 |
|---|---|---|
| Wave 1 基线 | `78.8 tok/s`（144 tok / 1827ms；另一次 120 tok / 46×33.1ms） | `dspark-correctness-chain` |
| + SH_PAIR_M=1 | `78.1 tok/s`（k_acc 逐位不变） | 同上 |
| + R2 ATTN_LIN_FUSE | `82.9 tok/s`（**completion=130 / 1568ms**；k_acc ≈ 5） | 同上 |

两个不能同时为真的数：
- `130 tok / 1568ms` ⇒ **步时 71ms**（若 `k_emit=6`，即 accept 5）或 **34ms**（若 `k_emit=2.8`）。
- 「29ms」既不属于前者也不属于后者。

⇒ **唯一稳的量是「同任务、同 binary 的相对位移」**：R2 = `78.1 → 82.9` = **时间 ×0.942，即 −5.8%，−96ms/请求**。
本文件所有增量都乘在这个相对基线上，不做绝对步时反推。

---

## 2. R2 的 launch 账（反过来标定 µs/launch）

### 2.1 R2 省的 launch 数（代码核对）

`attention_rows(m=1)`（`chain_dev.rs:9409-9538`）：

| | 之前（mrows 化） | R2 之后 |
|---|---|---|
| wq_a + wkv | `quant_rows(xn)` 1 + `proj_mrows(wq_a)` 1 + `proj_mrows(wkv)` 1 = **3** | `lin2` **1**（EAGER `:4221`，含 quant） |
| q norm + wq_b + rope | `norm_rows(qr)` 1 + `quant_rows(qr)` 1 + `proj_mrows(wq_b)` 1 + `apply_rope_mrows` 1 = **4** | `lin_rope_norm` **1**（`:4359`，norm+fp8+gemv+rope） |
| 补偿 norm | — | `norm_rows`（**R2 新加**，因为 `lin_rope_norm` 把 `qr_r` 留成 raw，indexer 的 q 半要读归一化 qr）**1** |
| **小计** | **7** | **3** |

⇒ **R2 净省 4 发/层**（仓库文档写「7→~4」= 省 3；差 1 是补偿 norm 算不算的口径，本文取 4，并在 §3 给区间）。

### 2.2 由实测 delta 反解 µs/launch

```
Δtime(R2) = 1568 × (82.9/78.1 − 1) = 96ms
launches saved = 4 发/层 × 40 层 × rows ≈ 160 × 130 = 20,800 发
⇒ µs/launch ≈ 96,000µs / 20,800 = 4.6µs      （若 R2 只省 3/层则 6.2µs）
```

**4.6µs/launch 与仓库的核算自洽**：GPU 侧 ramp/drain ~3.3µs + 被隐藏的 submit ~2.9µs
（`batched-400-v2-remaining-roi §5.1`），lazy 无图路径里每行还有 blocking H2D/D2H
（`chain_dev.rs:8168/8182`）把 launch 串起来 ⇒ 有效 ~4~6µs 是合理的。

> ⚠️ 任务用的 **2µs/launch 是「纯 submit」，不是 lazy 路径的有效成本**。它是本账里最大的单个偏差。

---

## 3. R2b 的预期节省

### 3.1 launch 账（代码核对）

| 项 | 层数 | 发数/行 | 依据 |
|---|---:|---:|---|
| 跳过补偿 `norm_rows`（`chain_dev.rs:9530`，`qr_raw_r` 置位即跳过） | **40**（`lin_fuse` 在 m=1 的每一层都取，`:9415`） | −40 | 全层 |
| indexer q 半 `quant+proj+rope → lin_rope_norm`（`:10503`、`:10695`） | **8**（`index_source_layer_ids`） | −16 | 每层 −2 |
| **合计** | | **−56/行** | |

**注意**：`indexer_rows_one`（per-row select，m=1 lazy 走这条，`INDEXER_MROWS` 默认 OFF）与
`indexer_front_rows`（block-wide hoist）**都被 R2b 改了**，但两者都只在 `is_idx_src` 层生效 ⇒ 8 层。

### 3.2 预期节省

```
R2b = 56 发/行 × 130 行 × µs
µs=4.6 ⇒ −33ms  → 1568→1535ms → 84.7 tok/s（+2.2% over R2）
µs=6.2 ⇒ −45ms  → 1523ms      → 85.4 tok/s
µs=2.0 ⇒ −15ms  → 1553ms      → 83.7 tok/s   （任务口径的下限）
```

⇒ **R2b ≈ −15 ~ −45ms/请求 ≈ −0.7 ~ −2.0 ms/步（@22 步）**。
量级与任务的「−1.4ms/步」接近，但**结构不同**：任务按 120 发/行 × 2µs，本文按 56 发/行 × 4.6µs。
两者在总量上偶然抵近，但任务口径在**只开 indexer 的层数**与**launch 有效成本**上各错一次、方向相反。

> **附注（R2b 的额外小件）**：`lin_rope_norm` 同时消掉了 indexer 的 `quant_rows` 中间写回与
> `proj_mrows` 的 mrows 实例，可能略带 RAM/L2 收益；量小，未计入。

---

## 4. A4 的预期节省

### 4.1 A4 治什么（代码）

`ar5_wait_round`（`ferrite_kernels.cu:8964`）：
- **OFF 臂**：`if (threadIdx.x < world)`、**无 `blockIdx` 守卫** ⇒ n=5120 时 `ceil(5120/64/4)=20` 块
  × 8 线程 = **160 个轮询者**打同一组 8 个 stamp 字（v5 融合 publish/reduce 时丢失的 v3 性质）。
- **ON 臂**：`blockIdx.x==0` 轮询 → `__threadfence()` + 写 `epoch[1]`；其余 19 块只等**一个本地字**
  ⇒ **160 → 8 个轮询者**，并带回一个 broadcast hop。

### 4.2 能收多少

22.5µs/AR 的「非工作量」三候选（`ar-further-optimization §3.1`）：
1. **peer stamp 等待（对端到达延迟）** —— 这是**数据依赖**（最慢 peer 何时把 payload fence 完才写 stamp），
   A4 **不动它**（block 0 仍直接等同样的 stamp）；
2. **nsys 自旋放大**（~300×）—— 与 arm 无关；
3. **L2 争用**（160 poller × 100ns 打 8 条 L2 行）—— **A4 只治这一项**。

⇒ A4 的上界 = 第 3 项的份额。文档给的保守值 **−1~3µs/AR**；再乘 rows-AR 次数：
- `k_emit=6` ⇒ `80 处/层·行 × 6 = 480 AR/步` ⇒ **−0.5 ~ −1.4 ms/步**（上界）；
- 但第 3 项是否真有 1~3µs 是未知数（需 A1 探针），且 A4 的 broadcast 引入 ~0.3~1µs 串行 hop 抵消一部分。

**本文取 −0.2 ~ −1.0 ms/步（中枢 ~−0.5）**，即 **−4 ~ −22ms/请求**。

### 4.3 A4 的真正价值（不是 ms）

A4 让 **A1 探针在两个臂测同一 site（block 0）**，且把「惊群」这个混杂因子从探针里拿掉 ⇒
A1 的 `[ar-probe] avg_spin` 才第一次能读成「对端到达延迟」。**先有 A1 的分布，才谈得上 A2（图化）
和后续 AR 项**。所以 A4 是**诊断前置件**，不是一个独立的吞吐项。

---

## 5. 组合预测

### 5.1 主表（锚定在 R2 的 82.9）

| 项 | 请求级 Δms | tok/s | 依据强度 |
|---|---:|:---:|---|
| R2（实测） | — | **82.9** | ✅ 实测 |
| + SH_PAIR_M=1 | ~0（−0.9%，噪声） | 82.9 | ✅ 实测为中性 |
| + R2b | −15 ~ −45 | 83.7 ~ 85.4 | 结构推算（launch 账 × µs/launch） |
| + A4 | −4 ~ −22 | 84.4 ~ 86.2 | 设计口径（上界） |
| **组合** | **−19 ~ −67** | **≈ 84 ~ 87** | 中枢 **~85.6** |

### 5.2 为什么不是 90~95（两种独立证法）

1. **用任务自己的 launch 数也到不了**：任务口径 R2b=720 发/步、A4=−0.5ms/步，
   在 **29ms 步时**下合计 −1.9ms = **−6.5%** ⇒ 82.9 → **~88.5**，仍 < 90。
2. **用实测锚定的 µs**：要到 92 tok/s 需 `−155ms` ⇒ 按 4.6µs/发需要 **~34,000 发/行** ——
   相当于把当前 verify 的**全部** launch 数（~2,800 发/行）再砍一个数量级。launch 修剪已到边际。

⇒ **90~95 是口径叠加造成的乐观**（把 R2b 的层数 ×40、把 µs 取 2、把步时取 29ms，三处同时偏乐观）。

### 5.3 判据（上机时怎么读，避免 self-deception）

- **不要只看 tok/s**：同会话 `R2`（只开 `ATTN_LIN_FUSE`）↔ `R2+R2b`（+`INDEXER_QR_RAW`）背靠背
  交错 A/B；R2b 的预期位移只有 **+1~2%**，**小于任务的 +2~4%**，必须用**多样本中位**判。
- **要有 launch 证据**：`[dsv41]`/`[verify_graph]` 的 figure 节点数，或 nsys 按 kernel 名数
  `rmsnorm_rows`/`quant_rows`/`apply_rope_mrows` 的实例数应各降 ~40 行、`idx` 侧降 ~16。
  **只看 tok/s 会分不清「R2b 生效」与「gate 没生效」（本项目 #1 陷阱）**。
- **A4 必须先开 A1 探针**：`DSV41_AR_PROBE=1`，比较 `SINGLE_POLL=0/1` 的 `avg_spin`；
  若 `avg_spin` 不变 ⇒ A4 无收益，**照文档约定不留**（< 1% 就不进默认值）。
- **正确性红线**：R2b 动了 `qr` 的 raw/norm 语义 ⇒ 必跑 `dspark_parity` 行级（`verify_bad==0`）+ 零拉丁 + `k_acc` 逐位不变。

---

## 6. 到 400 的剩余差距 与 batched 的必要性

### 6.1 lazy 的结构性天花板

```
lazy 步时 = k_emit × c_row + draft + commit           k_emit = 1 + mean_k
```
- accept=5 ⇒ `k_emit=6`；实测 `c_row = 8.18ms/行 @ accept≈1.2`（`lazy-verify-optimization-path §1.1`）。
- 即使把 `c_row` 压到 EAGER 的 **6.15ms**（hc 融合 + sync 收敛 + SH_PAIR + tcgen05 全部兑现）：
  `6 × 6.15 + 4.3 + 0.2 = 41.4ms` ⇒ **145 tok/s**（这是 lazy 的数学地板）。
- 400 要求步时 ≤15ms ⇒ `c_row ≤ (15 − 4.5)/6 = 1.75ms/行` = **EAGER 的 1/3.5**。
  在「一行一遍权重读 + 一行一套 per-row launch」的 lazy 结构下这一条不成立。
- 组合栈（R2b+A4）只动 **launch 的常数项**（−2ms/步量级），**不动 `k_emit × c_row` 这个乘积**
  ⇒ 对 400 的贡献 ~0。

### 6.2 lazy 为什么贵（不是核慢，是重付）

- **per-step 族被 ×k_emit 重付**：hc 链（400 发/块，53 GB/s ⇒ **纯 launch-bound**）与 AR v5（80 轮/步）
  在 lazy 下每步付 `k_emit` 次。`k_emit=6` 时这两族的重付 ≈ **+10ms/步**（相对 batched 只付 1 次）。
- **per-row 族的权重读 ×k_emit**：投影族 ~4.4ms/步是 **17× 带宽地板**（instruction-bound），
  但 lazy 连那 1× 的带宽都不共享 —— 逐行各把 23.84MB/层读一遍。
- launcher 侧：verify 6224 发/步 ⇒ GPU 侧 ramp/drain ~20ms（`batched-400-v2-remaining-roi §5.1`）。
  **削 launch 的唯一杠杆是族级融合/按行摊薄 —— 二者都要 `m>1`。**

### 6.3 batched（SWALLOW m=6）为什么是唯一路径

| | lazy（m=1/行，k_emit=6） | batched（SWALLOW，恒 6 行） |
|---|---|---|
| per-step 族（hc/AR） | `6×` ⇒ 亏 ~10ms | **1×** |
| per-row 族（proj/shared/routed/indexer/attn…） | `6×` | **6 行共享一次权重读 + 一次 launch** |
| 主链 forward | 每行一遍（6.15ms×6） | **anchor 折进 verify 行 0**（吞掉一次独立主链步，−4.55ms） |
| 形状 | m=1（mrows 全部失效） | m=6（mrows 族 + SH_PAIR M≥2 phase-1 并行度全部生效） |
| 步时 | 6×c_row + 4.5 ≈ **41~53ms** | forward 7~8 + 4.3 + 0.2 ≈ **12.5ms** |
| 吞吐 | **145 ~ 112 tok/s** | **~480 tok/s**（诚实 60% 兑现 ≈ 15~17ms ⇒ 350~400） |

**必要性论证（三句）**：
1. **400 要求把「每行一遍权重 + 每行一套 launch + 每步一套 per-step 族」三者都变成「每 6 行一次」。**
   lazy 的结构**定义上**做不到（它逐行早退：第 r+1 行的输入是第 r 行的 argmax）。
2. **只有 batched 能同时摊薄这三样**，因为它的 6 行是一次 m=6 forward（权重与 per-step 族各付一次），
   而早退在前 2~3 行内命中时，**多余的 speculative 行成本 < 省下的 per-step 族重付**（τ≈3.5，
   accept 5 > τ ⇒ batched 赢）。
3. **R2/R2b/A4 是「lazy 内部的 launch 修剪」**：它们把 82.9 推到 ~85（+2~4%），
   但它们**不改变 400 依赖的那个乘积结构** —— 所以对 400 的贡献约等于 0。
   400 的关键路径是 **Plan B（unanimity-or-direct）把 SWALLOW + 图的 `ar5-hang` 解掉**，
   而不是继续在 lazy 上削 launch。

---

## 7. 诚实校准

1. **µs/launch（4.6~6.2）是从 R2 的单个实测 delta 反解的**（96ms / 15,600~20,800 发），
   误差源：R2 真实省的是 3 还是 4 发/层的口径差、130 行是否等于 rows、
   以及非 launch 项（`lin2`/`lin_rope_norm` 的核内工作）也在 delta 里。**绝对量给区间**。
2. **`rows ≈ tokens ≈ 130`** 是近似（每行 emit ~1 token），量级正确；不影响相对结论。
3. **A4 的 −0.2~−1.0ms 是上界**：它治的 L2 争用份额未实测（这正是 A1 探针要回答的）。
   若探针显示 `avg_spin` 主要由对端到达支配，A4 的真实收益会落到 **~0~−0.3ms**。
4. **SH_PAIR_M=1 的 −0.9%** 在 ±1% 噪声内，本文按「中性」计，不宣称负收益。
5. **本机无 GPU**：除 R2 的 82.9 与基线 78.8/78.1 外，全部增量是结构/账本推算，未实测。
6. **未改任何源码**；本文件是唯一产出。

---

## 8. 建议的最小验证（2 次 GPU 会话）

| # | 会话 | 内容 | 判据 |
|---|---|---|---|
| **P1** | GPU(1) | 同会话交错 A/B：`ATTN_LIN_FUSE=1`（R2）↔ `+INDEXER_QR_RAW=1`（R2b）；每臂 ≥5 样本取中位 | tok/s 位移 **+1~2%**；nsys 里 `norm_rows`/`quant_rows`/`apply_rope_mrows` 各 −40 行、idx 侧 −16；零拉丁 + `verify_bad==0` + k_acc 逐位不变 |
| **P2** | GPU(1) | `AR_SINGLE_POLL=0/1` × `AR_PROBE=1`（4 臂，交错） | `avg_spin` 位移；若 <1% 收益 ⇒ 不进默认值；同时把 22.5µs 拆成「等待/搬运」 |

**铁律**：同一远端同时只有一个测试驱动；每门读回 `/proc/<pid>/environ`；
**不要用 `6/0.02256×k_emit` 之类的口径式反推**做 A/B 判据（本文件的 §0-2 就是被这个坑过的证据）。

---

*工部 · 只读分析 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*所有 ms 标来源（实测 / 账本推算 / 结构推算）；口径冲突处已显式标注（§0-2、§0-3、§0-4）。*
