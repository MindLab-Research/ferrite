# batched 400 v2 之后的剩余步时优化清单（ROI 排序）

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 代码基线：HEAD `65f092e`（`crates/ferrite-models/src/dsv41/chain_dev.rs` /
> `crates/ferrite-models/src/dsv41/tp.rs` / `kernels/cuda/ferrite_kernels.cu`）。
> 任务：`SWALLOW_STEP` + 全 mrows 落地后，**若步时 >10ms 还剩哪些优化**。
> 输入：`swallow-step-400-necessity.md` · `verify-family-fusion.md` · `verify-ms-breakdown.md`
> （含其 §修正） · `verify-marginal-cost.md` · `hc-chain-bandwidth-analysis.md` · `final-400-battle.md`。

---

## 0. 判决（先读五条）

1. **五项里有两项的前提在 HEAD 已经过时/已经作废**：
   - **AR v5 已经是 2-kernel，不是 3-kernel**（`publish`/`stamp` 早在 2026-09-10 就融进了
     `p2p_ar_pubred_v5_kernel`）。文档里的「3 核 / 240 发」是旧口径。⇒ 「能否 2-kernel」= **已经是**；
     真正的剩余是「能否 1-kernel」（store 折进 producer 尾）。
   - **verify 图化不是性能杠杆**：同会话 A/B 实测只 **−1.5ms**（预期 −24ms），
     `{SH_EXP_MROWS+VERIFY_GRAPH+VERIFY_ROPE_MROWS+DRAFT_P3A}` 全开只 **−1.21ms**。
     CUDA async launch 已让 submit 与执行重叠 ⇒ 图化只剩「正确性/顺序工具」的价值。
2. **hc 链是最高 ROI 的剩余项**（−1.3~1.7ms，成本 0.5~1 人日）：
   三个折核件**都已在树里**，且 A1-a 的 `truncate=false` 修复已落；卡点只剩
   **A2 臂把 `bf16_truncate()` 带进了 verify**（与 A1 当年打破零拉丁是同一个坑）。
3. **attention 的 scratch ring 快照，「拷贝成本」不是瓶颈**（~10MB/步 ≈ 1.4µs），
   真正的成本是**注意力核的复合视图重构**（3–4 人日、高风险、正落在 defect #1/#2 最密的代码里）。
4. **indexer 的 front 半已经实现（`DSV41_INDEXER_MROWS`）**，select 半被三个结构性约束挡住
   （scan bound / runtime out_stride / clen 因果），其中前两个是**小改**、第三个与 attention 同源。
5. **排序（ROI 降序）**：hc(A2 truncate 修复) → indexer(front 已就绪 + select 补 stride) →
   attention(scratch ring) → AR(store 折 1 发) → verify graph(≈0，不列入收益)。

---

## 1. AR v5（文档口径 1.40ms）——前提已过时，实际只剩 ≤0.24ms

### 1.1 HEAD 的真实形态（代码为准）

| 事实 | 证据 |
|---|---|
| `ferrite_p2p_ar_v5` = **store + pubred 两发** | `ferrite_kernels.cu:8987-9019`（`p2p_ar_store_v5_kernel` `:8853` + `p2p_ar_pubred_v5_kernel` `:8895`） |
| `publish`/`stamp` **已融进 pubred**（不是独立核） | `:8902-8907`「the publish used to be its own 1-block kernel and the reduce another launch — **3 kernels per AR**」；`:8915-8922` 在 pubred 内 `atomicExch_system` + `*epoch = e+1` |
| `advance` 也是 pubred 内的一行（block0/thread0） | `:8921-8922`，无独立 launch |
| verify 的两处 AR 都走 `all_reduce_inplace`（=2 发） | `chain_dev.rs:9630`（注意力 `wo_out_r`）、`:11287`（MoE `moe_out_r`） |
| `end_round()` 对 v5 是 no-op（无 host barrier、无核） | `tp.rs:734-740` |

⇒ HEAD 下 verify 的 AR = **80 次 × 2 核 = 160 发**，不是 240。
（`verify-ms-breakdown.md §1` 第 13 行 / `verify-family-fusion.md` 全景修正 / `dsv41_verify_perf_plan §1`
的「80 次 × 3 核 = 240」是 2026-09-10 融合之前的记账，**已漂移**。上机时用 `DSV41_VERIFY_GRAPH`
的图节点数或 nsys 直接数一次，别按 240 编预算。）

### 1.2 唯一的真实剩余：store 折进 producer 尾（2→1 发）

- **已存在但默认关**：decode 的 MoE AR 有 `gemm_fp8_mx_ar`（store 折进 GEMM 尾）+ `all_reduce_inplace_pubred_only`
  （`chain_dev.rs:13146/13180`），但被 `DSV41_AR_STORE_FUSE` 关掉，且**默认 OFF**——理由是
  `:13027-13031`「round 19 showed the fused path breaks the four texts even with GATEUP/DOWN_FUSE off」
  （与 GATEUP_FUSE 数值 bug 耦合，未解）。
- **verify 侧还没接线**：verify 的注意力 producer 是 `gemm_fp8_mx_or_swap`（`chain_dev.rs:9596`），
  它**没有 `_ar` 变体**，所以 verify 的 store 无法折进 producer。MoE 侧同理（`moe_rows` 走逐行 GEMM）。
- **另有一条便宜的路**：`DSV41_VERIFY_AR_FOLD`（默认 OFF，`chain_dev.rs:11814`）把
  `hc_post_inplace_rows` 折进 pubred 尾 = A1 链 240 发里省 **80 发**（≈−0.24ms）；但它**依赖 A1
  （`hc_verify_fuse()`），A1 现在是 OFF** ⇒ 先解 §3 的 truncate 坑，这条才可开。

### 1.3 结论

| 方案 | 预期 | 成本 | 风险 |
|---|---|---|---|
| 折 verify 注意力 AR 的 store 进 producer 尾（新 `_ar` 变体 + m 行） | **−0.2ms**（80 发 × ~2.5µs 的 store ramp/drain） | 1.5–2 人日 | **高**（AR_STORE_FUSE 的数值破功未解，且 verify 是正确性红线路径） |
| 折 `VERIFY_AR_FOLD`（依赖 A1） | −0.24ms | 0.5 人日 | 中（m 行 epilogue 跨 TU，需 parity） |

**AR 不是步时 >10ms 的解**：即使全折，也就 −0.4ms 上限，且它是**协议地板**（store 等 peer 的延迟）。
「3-kernel → 2-kernel」这一问无需再花机时——**代码里已经是 2**。

---

## 2. attention 2.8ms 的 scratch ring 快照：拷贝不是成本，核重构才是

### 2.1 约束回顾（为什么 m 行批不了）

- 行内因果链是 `read(r) → append(r) → read(r+1) → append(r+1)`，**`sparse_attn(r)` 必须早于
  `append(r+1)`**（`chain_dev.rs:7938+` 的注释写死「THE ORDER IS THE WHOLE POINT」）。
- `DSV41_ATTN_MROWS` 的 `b·m` 单发只在 **`world == 1 && pos + m - 1 < win`** 时取
  （`chain_dev.rs:9096-9123`）。`win = window_size = 128`（`configs/dsv41_flash.json:59`）⇒
  只有 `pos < ~123` 安全，**长上下文恒 decline**（环已回绕，`window_idxs` 的 `idx > start_pos`
  过滤永不触发，行 `0..m-2` 读到块自己的未来行 = audit defect #2）。

### 2.2 「scratch ring 快照」的真实成本（用真实 shape 算）

| 项 | 数值 | 说明 |
|---|---|---|
| ring 尺寸 | `(win=128 + max_comp) × hd=512 × 4B` ≈ **256KB/H 层** | `chain_dev.rs:2891`（ring 只在 KV-owner 层分配） |
| 块前快照 | D2D copy 256KB ⇒ **0.037µs @7TB/s** + 1 launch/owner | `kv_snap_ring` **已分配**（`chain_dev.rs:3100`，`40×128×512×4 = 10.5MB`） |
| 全步快照 | ~30 owner × 256KB ≈ 7.7MB ⇒ **~1.1µs** + ~30 launch ≈ 0.1ms | 带宽利用率仅 4.9%，**拷贝量根本不是约束** |
| 需要被「读回」的旧内容 | 每行 r 只需 r 个被覆盖 slot（Σr = m(m-1)/2 = 15 slot × 512 float） | 逻辑上极小 |

⇒ **`verify-family-fusion.md` §W2 的「copy 成本 ~window×dim×2 bytes × m 行，可能得不偿失」是误判**：
在 7TB/s 下这是微秒级。**真正的成本全在核侧**——每个行 r 必须读一个**复合视图**
（snapshot 里被本块覆盖的 r 个 slot + live ring 其余部分），这要求：

1. `window_idxs` / `sparse_attn{,_warp,_pf,_split,_orope}` / `merge` 加**第二 base 指针 + 逐行碰撞计数**
   （`dsv41_kernels.cu` 的 5 个 kernel 体，**已因 ABI 5 的 `clen_rows`/`idx_stride` 改过一轮**）；
2. 或改成「块内 r 升序的 ring/window 合核」——但**它不解决 `attn(r)` 早于 `append(r+1)` 的依赖**，
   仍只能逐行串行，省不掉边界成本；
3. 还欠 `row_pitch` 尾参（`world>1` 时行距是 `nh*hd`，kernel 用 `h*d`）——见 `verify-family-fusion.md` §W2 补正 3。

### 2.3 结论

| 方案 | 预期 | 成本 | 风险 |
|---|---|---|---|
| scratch ring 快照 + 复合视图核重构 | **−1.3ms**（2.8 → ~1.5） | **3–4 人日** | **高**：正落在 defect #1/#2 最密的代码；需新 parity + 长上下文 A/B |
| 只做 `row_pitch` + 保持短上下文门槛 | ≈0（受益面仍 <128 token） | 0.5 人日 | 低 |

**建议：不在「测试即将运行」的窗口内动它。** ROI 低于 hc（§3），风险高于 indexer（§4）。

---

## 3. hc 链 2.96ms —— 最高 ROI 的剩余项（A2 的 bf16_truncate 修复）

### 3.1 现状：A1 被默认关掉，verify 回到原始 10 发链

- `DSV41_HC_VERIFY_FUSE` **默认 OFF**（`chain_dev.rs:11791-11795` 的 `.map(|v| v == "1").unwrap_or(false)`，
  是个**反向默认**）；理由（`:11782-11790`）：A1 让 verify 第一次吃到
  `hc_collapse_norm(truncate=...)` + `hc_post_inplace_rows`，**GPU A/B（9b55ea04）显示它打破零拉丁**，
  与 P0 系列 commit 叠加时尤其明显；`HC_VERIFY_FUSE=0` 恢复零拉丁。
- ⇒ 当前 verify 走的是**原始 10 发/层**：`hc_mixes` + `hc_collapse` + `norm_rows` + `hc_post` + `memcpy_d2d`，×2 侧
  （`layer_rows`，`chain_dev.rs:8500-8700`）。400 发/步 = 2.96ms。

### 3.2 A1-a 的修复已经在了，A2 的坑还在

| 半 | 代码 | truncate | 状态 |
|---|---|---|---|
| **A1-a**（`hc_collapse_norm` rows=m） | `collapse_norm_rows`（`chain_dev.rs:8380-8420`） | **显式 `false`**（2026-09-12 修复，注释「truncate = false on purpose」） | ✅ 已修，**不会**再带 BF16_TRUNCATE 进 verify |
| **A1-b**（`hc_post_inplace_rows` rows=m） | `hc_post_rows`（`chain_dev.rs:8452`） | 无 truncate 参数 | ✅ 与 truncate 无关 |
| **A2**（`hc_front_rows` → `hc_mixes_auto` → `hc_front_split`） | `hc_mixes_auto`（`chain_dev.rs:11556-11714`，调用点 `:11625`） | **传 `bf16_truncate()`（活 gate！）** | ❌ **同一个坑：A2 会把 BF16_TRUNCATE 重新带进 verify** |

⇒ **A2 的修复 = 把 `hc_mixes_auto` 里 verify 调用点的 `bf16_truncate()` 改成 `false`**
（或加一个「verify 形态」标志，单行 `layer()` 保持读 gate）。这是与 A1-a 完全同构的一行修复，
但**必须在 A2 上机 A/B 前做**，否则会把「A2 的语义问题」和「truncate 破零拉丁」混在一起，
重演 A1 那次的误判。

### 3.3 预期（沿用 hc-chain-bandwidth-analysis §5/§7.3）

| 形态 | launch/步 | ms/步 | 依赖 |
|---|---:|---:|---|
| 现状（A1/A2 全 OFF） | 400 | **2.96** | — |
| A1（`HC_VERIFY_FUSE=1`，truncate 已修） | 240 | ~2.3 | A1-a/b 已就绪 |
| A1 + A2（`HC_FRONT_ROWS=1`，**truncate 改成 false**） | 240（主径遮住 dots/LATE） | **~1.3–1.7** | 需 `row_pitch` 无关（hc 侧已是 rows=m 原生） |
| + `VERIFY_AR_FOLD=1` | 160 | ~1.3 | 依赖 A1 |

**净收 −1.3~−1.7ms，成本 0.5–1 人日**（一行 truncate 修复 + 重跑 A/B 确认零拉丁 + 看
`[dsv41]` 的 launch 计数）。**这是全清单里 ROI 最高的一项。**

⚠️ 一个已知的技术债：`DSV41_HC_VERIFY_FUSE` 的**反向默认**（`v == "1"` 才开）与项目其余 gate 的
`v != "0"` 默认不一致，A/B 脚本容易踩空。落地时建议连同默认值语义一起复核（`hc-chain-bandwidth-analysis.md §7.2`
写的是 ON，代码是 OFF——以代码为准）。

---

## 4. indexer select 半（80 发/步）——front 已就绪，select 缺 stride

### 4.1 拆分

- 家族现状（`indexer_rows_one`，`chain_dev.rs:9893`）：每行 5 发 =
  `publish`(1) + `lin`(2，quant1+gemm) + `apply_rope`(1) + `lin_bf16`(1) + `indexer_topk`(1)；
  8 个 index-source 层 × 5 行 ≈ **230 发/步**。
- **select 半 = `publish` + `indexer_topk`** = 2/行 × 5 行 × 8 层 = **80 发/步**（与任务口径一致）。
- **front 半已经实现**：`DSV41_INDEXER_MROWS=1`（默认 OFF）→ `indexer_front_rows`（`chain_dev.rs:9790`）
  把 `idx_wq_b` 投影+rope 与 `idx_weights` 折成 **4 发/层**（省 16/层的 front launch）。

### 4.2 select 为什么没融（三个结构性约束，`chain_dev.rs:9740-9768`）

1. **scan bound**：`indexer_topk_kernel` 收 `n_pos` 后又用 `*lens`（=行 0 的计数）覆盖它
   （`dsv41_kernels.cu:2854`）⇒ 逐行 `lens[m]` 各异时，行 1..m-1 会被钳到行 0 的计数、丢掉最新组。
   需要 `n_pos = max(lens)`。
2. **输出行距**：kernel 写 `out[row*cols + i]`，`cols = min(topk, n_pos)` 是**运行期值**，
   而 `idxs_r` 的行距是**固定** `win + index_topk`。⇒ m 行单发需要一个显式 **`out_stride`** 参数。
3. **因果**：逐行计数只在**所有行 commit 之后**才存在；融 select 会连带搬动 `sparse_attn`
   （正是 §2 的 attention 家族重构）。

### 4.3 结论

| 子项 | 预期 | 成本 | 风险 |
|---|---|---|---|
| 开 `DSV41_INDEXER_MROWS`（front，**代码已就位**） | −1.0~−1.5ms（整族 2.50→~1.0，含边界） | **0（只差默认值/A-B）** | 低（逐位等价已论证） |
| select 补 `out_stride` + `n_pos=max(lens)`（2 行 C + 1 行 launcher） | −0.1~−0.2ms（80→~44 发） | 1 人日 | 中（需 parity） |
| select 的 clen 因果（设备侧 `clen_rows` 快照） | 与 attention 半共享 | 见 §2 | 高 |

**注**：`indexer_rows_one` 已经带了 `mrows_clen: Option<ptr>` 参数（COMPRESSOR-MROWS 的 hoist 用，
`chain_dev.rs:9888-9892`）——**indexer 侧的 clen 设备快照机制已部分存在**，比 attention 侧容易。
但 select 半要真正收益，仍要先解 attention 的因果前置。

**建议：先只开 front（零成本），select 的 stride 改动作低优先级。**

---

## 5. verify 的 6224 launch → 图化后 submit：**已实测，答案是否**

### 5.1 实测结论（同会话 A/B，`verify-ms-breakdown.md §修正` + `final-400-battle.md`）

```
裸链 verify              = 37.31ms
{SH_EXP_MROWS, VERIFY_GRAPH, VERIFY_ROPE_MROWS, DRAFT_P3A} 全开 = 36.10ms   ⇒ 仅 −1.21ms
图化：pos=20 捕获成功、30/50 步 replay；replay 35.5ms vs 裸链 37ms          ⇒ 仅 −1.5ms
```

⇒ **CUDA async launch 已经让 CPU submit 与 GPU 执行重叠**。2.9µs/launch 的提交成本被 GPU 工作在
时间上隐藏了（CPU 跑到前面去了）。图化**不消除** GPU 侧的 per-kernel ramp/drain（~3.3µs/发，
6224 发 ≈ 20ms 的「执行半」）。⇒ 「50% submit + 50% exec」的分解**已被推翻**。

### 5.2 因此

- **图化的定位 = 正确性/顺序工具**（保证 launch 顺序、让 replay 走同一条序列、去掉 host 分支），
  **不是性能工具**。不要为它编 −8ms 的预算。
- **真正削 launch 成本的是「少发核」**（族级融合），融合把 N 从 6224 砍到 ~1200-1400 时，
  那 20.5ms 的 GPU 侧边界成本才降到 ~5ms——**这是唯一不依赖核效率假设的收益**。
- **与 SWALLOW_STEP 的交互**：swallow 让形状从 m=5 变 **m=6**，`VERIFY_ROWS=6` 恰好是分配上限，
  per-shape 槽（`VERIFY_GRAPH_SLOTS=3`，`chain_dev.rs:107/5192`）自动支持两形状。
  A/B 时必须确认日志出现 **`[verify_graph] captured verify_graph_m6`**（否则 6 行块退回 direct launch，
  swallow 的 −4.5ms 会缩水）。**本轮不要同时开 `LAZY_VERIFY`**（会占第三槽、且走 m=1 行循环）。

---

## 6. 按 ROI 排序的剩余优化清单（交付物）

| 排名 | 优化 | 预期 ms | 实施成本 | 风险 | 前置/备注 |
|---|---|---:|---|---|---|
| **1** | **hc A2 的 `bf16_truncate` 修复 → 开 `HC_FRONT_ROWS`（+ 复发 A1）** | **−1.3 ~ −1.7** | **0.5–1 人日** | **中** | 一行改 `hc_mixes_auto` 的 verify 调用点传 `false`；重跑 A/B 确认零拉丁；顺带复核 `HC_VERIFY_FUSE` 的反向默认 |
| **2** | **indexer front（`DSV41_INDEXER_MROWS=1`）+ select 补 `out_stride`/`n_pos=max(lens)`** | **−1.0 ~ −1.6** | 0（front）/ 1 人日（select） | 低-中 | front 代码已就位、逐位等价已论证；select 真收益仍需 attention 的 clen 因果 |
| **3** | **attention `b·m` 的 scratch ring 快照 + 复合视图核重构** | **−1.3**（2.8→1.5） | 3–4 人日 | **高** | 拷贝成本可忽略（~1.4µs/步）；成本全在 5 个 kernel 体 + merge + `row_pitch`；defect #1/#2 高发区 |
| **4** | **AR store 折进 verify producer 尾（2→1 发）** | **−0.2** | 1.5–2 人日 | **高** | `AR_STORE_FUSE` 数值破功未解；verify 是正确性红线路径。**注意：AR 已是 2-kernel，不是 3** |
| **5** | **verify 图化** | **≈0（已入账 −1.5）** | — | — | **不是性能项**；只作顺序/正确性工具；用图节点数复核 AR 记账 |
| 并行项 | `VERIFY_AR_FOLD`（折 hc_post 进 pubred） | −0.24 | 0.5 人日 | 中 | 依赖 #1（A1 重新 ON） |
| 并行项 | `row_pitch` 尾参（ATTN_MROWS 的 TP>1 使能） | ≈0（仅短上下文） | 0.5 人日 | 低 | 单独无肉，属 #3 的零件 |
| 大象 | **routed experts tcgen05 e4m3（N=8 容 m=6）** | **−5 ~ −6.8** | 4–5 人日 | 高（e4m3 parity 前置） | 不在本 5 项内，但是**唯一能越 3.4× 地板的动作**；与 e4m3 激活互斥需先解 |

### 6.1 一句话建议

> **测试窗口内可低风险拿到的是 #1（hc）+ #2 的 front 半 ≈ −2.3 ~ −3.3ms**，
> 且两者都是「代码已就位、只差一行 truncate 修复 + 默认值」。#3/#4 是**架构级重构/未解数值坑**，
> 应排在 tcgen05 之后。**图化（#5）不要计入剩余收益**——实测已经证明它只有 −1.5ms。

### 6.2 上机前必须钉死的三个未知（否则清单会偏）

1. **AR 的真实核数**：HEAD 代码 = 2 发/AR；文档 = 3 核/240 发。用图节点数或 nsys 数一次
   （`DSV41_AR_V5` 的 device 侧自旋在 nsys 下是病态，改用 `cuda_gpu_trace`/图节点数）。
2. **A1 打破零拉丁的真实机理**：是 truncate，还是 `hc_post_inplace_rows` 的跨 TU？
   若是后者，A2 的 truncate 修复不够（但这与 `collapse_norm_rows` 的修复注释矛盾，倾向 truncate）。
3. **`DSV41_INDEXER_MROWS` 开后的真实 launch 数**：front 是 4 发/层（省 16），
   但 select 仍 2/行 ⇒ 家族落在 ~80+32=112 发/步，别按「230→40」的乐观值编预算。

---

*工部 · 只读分析 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*所有 ms/launch 数均标注来源；代码行号以 HEAD `65f092e` 为准，读代码时以函数名为准。*
