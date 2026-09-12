# verify 的「5 行边际成本」：物理来源拆解（户部 · 资源与性能）

> **核心问题**：EAGER 单行 = 6.15ms，verify 5 行 = 38.34ms = **6.23×**。
> 若 5 行共享权重（mrows 兑现），边际应 ≈ 计算增量 0.5~1ms（总 ~1.2× = 7ms）。
> **为什么是 6.2×？那 23ms 在哪？**
>
> 方法：**只读代码**（默认值 + 调用点逐条核对）+ 仓库内实测/账本数字核算。
> 本机无 GPU，未跑 nsys；凡未经实测的项在 §7 列明。HEAD `b40345a`。
> 户部 · 2026-09-12

---

## 0. TL;DR —— 三句话

1. **38.34ms = 6.15ms × 5.03（字节倍数）× 1.24（效率损失）。**
   主导项是**字节倍数 5×**，不是"kernel 变慢了"。
2. **"5 行共享权重"只在字节域部分成立**：11 个族里**只有 3 个真的折到 1×**
   （投影族 ✅、hc ✅、engram ✅）；**其余 8 个仍按行读/按行发**——
   其中 3 个是 `flag 默认 OFF`（共享专家 / gate / head），
   4 个根本没有 mrows 路径（attn / indexer / compressor / ring-window），
   routed 是设计上不可压的 5×。
3. **即使 flag 全开，也拿不到 1×**：这一族 kernel 跑在 HBM 峰值的 4.9%
   （**issue-bound，不是带宽 bound**）。mrows 折叠的是**权重解码**，
   折叠不掉**激活解码 + FMA**——后两者仍 ∝ m（指令数下限 0.66×，不是 0.2×）。

**⇒ 每多一行 verify，成本 8.05ms = 1.31 × 一个完整 EAGER 步。**
verify(5 行) = 38.34ms **> 5 × EAGER = 30.75ms**：
**当前的"批处理"比跑 5 次独立单行 forward 还慢 25%。**

---

## 1. 三个基准（口径先钉死）

| # | 量 | 值 | 来源 |
|---|---|---|---|
| B1 | EAGER 单行（m=1，40 层，TP8，整步图） | **6.15ms** / 162.6 tok/s | `STATUS.md:7901/7919` |
| B2 | verify 5 行（直接 e4m3 后） | **38.34ms** | `dspark-correctness-chain.md:409`（32d45b83） |
| B3 | verify 逐族 ms（m=5） | routed 8.3 / shared 10.4 / head 1.12 / 投影 3.7 / attn 2.8 / hc 2.96 / gate 3.44 / indexer 2.5 / 其它 1.79 | `verify-ms-breakdown.md §1`（自校准 +1.3%） |

**B1 的逐族口径（EAGER nsys，`STATUS.md:7919`）**：
`gemv 2.7ms（44%）· expert 2.0ms（32%）· hc/AR/misc 1.45ms（24%）`。
⇒ **EAGER 的 routed ≈ 2.0ms；非路由 ≈ 4.15ms。**

**B2/B3 的口径警告（必须先读）**：
B3 的逐族 ms 来自 `verify-ms-breakdown.md §1` 的账本，该账本按
**未切分的 head**（5 × 1323.8MB = 6619MB，占 14.09GB 的 **47%**）计算；
而 `VERIFY_HEAD_SLICED` 现在**默认 ON**（`chain_dev.rs:1105`）。
切分后 m=5 的总字节是 **8.30GB**（不是 14.09GB），head 从 1.12ms → ~0.14ms。
**本文件 §2/§3 的逐族判定不依赖这个口径**（判定基于"每族 1× 还是 5×"，是代码性质），
但 §3 的 ms 绝对值沿用账本口径以便与仓库既有文档对账。

**三个自洽性检查**（本文件模型的自校准）：
```
时间比  38.34 / 6.15                        = 6.23
字节比  14092.5MB / 2802.1MB（见 §2 表）      = 5.03
效率比  (14092.5/38.34) / (2802.1/6.15)
        = 367.6 GB/s / 455.6 GB/s            = 0.807  ⇒ 1.24× 损失
6.23 ≈ 5.03 × 1.24                            ✓ 自洽
```

---

## 2. 逐族 1×/5× 判定（代码为证）

EAGER m=1 字节按同一 shape 规则重算（`weights.rs::tensor_specs × local_shape`，TP8），
与 B3 账本逐项对齐；右侧是**代码侧判定**。

| 族 | EAGER m=1 MB | verify m=5 MB | × | 判定 | 代码依据（file:line） |
|---|---:|---:|---:|---|---|
| **routed experts** | 626.7 | 3133.4 | **5.0** | 🔴 **天然 5×，不可压** | `moe_rows` 单发但 `grid.z = rows`，30 vs 6 个 assignment → 字节真的 ×5（`chain_dev.rs:8245-8296`） |
| **shared expert** | 177.1 | 885.3 | **5.0** | 🔴 **应 1×，flag 默认 OFF** | `sh_exp_mrows()` → `DSV41_SH_EXP_MROWS ... unwrap_or(false)`（`:952-954`）；折叠块已写（`:8543`）但默认不进 |
| **投影族** | 955.1 | 955.1 | **1.0** ✅ | 🟢 **唯一真正兑现的 mrows** | `proj_mrows` + `quant_rows`（`:6990/2941`）；a32 decline 已在 `4c98b30` 移除（`dsv41_kernels.cu:5084-5119`） |
| **head**（sliced） | 165.5 | 827.5 | **5.0** | 🟠 应 1×，**折叠因数值错误禁用** | `verify_head_fold()` 默认 false（`:1111-1117`）；注释自陈 FOLD=1 时 echo 33% |
| **attention KV** | 52.4 | 262.1 | **5.0** | 🟠 可 1×（launcher 已支持 `b·m`） | `for r in 0..m { sparse_attn_orope(...) }`（`:7805+`）；`kAttnMaxBM=8`（`dsv41_kernels.cu:1449`）——**能力在、调用没走** |
| **hc 链** | 157.3 | 157.3 | **1.0** ✅ | 🟡 **已 1×**，但核只跑 53GB/s | `hc_mixes(..., m as i32, ...)` 原生 rows（`:7376`） |
| **MoE gate** | 157.3 | 786.4 | **5.0** | 🟠 **应 1×，flag 默认 OFF** | `row_fold_gate()` → `unwrap_or(false)`（`:918-921`）；mrows 核已存在 |
| **indexer** | 87.0 | 435.2 | **5.0** | 🟠 **逐行，无 mrows 路径** | 行循环内 `indexer_rows_one(layer, r, ...)`（`:7862`） |
| **compressor** | 73.4 | 367.0 | **5.0** | 🟠 **逐行，无 mrows 路径** | 行循环内 `compress_row(layer, r, pos_base)`（`:7841`） |
| **engram** | 315.0 | 315.0 | **1.0** ✅ | 🟢 已 rows=m | `advance_compress_lens` 注释 + `engram_apply_rows` |
| **激活 / 状态** | 35.3 | 176.6 | **5.0** | ⚪ 小 | `h_r/x_r/xn_r/logits_r` 均 `[VERIFY_ROWS, ...]` |
| **合计** | **2802.1** | **14092.5** | **5.03** | | 与 B3 账本逐项一致 ✓ |

**统计**：11 族中 **8 族是 5×**，3 族是 1×。
**5× 的 8 族再分三类**：
- **不可压（1 族）**：routed experts —— 路由是 MoE 的定义。
- **flag 默认 OFF / 未接线（3 族）**：shared（10.4ms）、gate（3.44ms）、head（1.12ms）
  —— **代码已写、逐位等价已论证，只差默认值**。
- **无 mrows 路径（4 族）**：attn（2.8ms）、indexer（2.5ms）、compressor（0.55ms）、ring/window
  —— **要么 launcher 已支持（attn 的 `b·m`）、要么需要新的合并核**。

---

## 3. 「23ms 差距」归因表

**差距定义**：实测 38.34 − 理想 15（= routed 8.3 + 非路由折叠后 ~7）= **23.3ms**。

| # | 项 | 现状 ms | 折叠后应到 | **缺口 ms** | 占比 | 机理 | 来源 |
|---|---|---:|---:|---:|---:|---|---|
| 1 | **shared expert** | 10.40 | 2.10 | **+8.30** | **36%** | 5× 重读（885→177MB）**且**核只跑 85GB/s | 账本 §2 + `SH_EXP_MROWS` 默认 OFF |
| 2 | **MoE gate** | 3.44 | 0.69 | **+2.75** | 12% | 5× 重读（786→157MB） | `ROW_FOLD_GATE` 默认 OFF |
| 3 | **投影族** | 3.70 | 1.00 | **+2.70** | 12% | **mrows 已生效**；缺口是 15.5µs/发固定项 × ~200 发 | 账本 §4.2（`已折叠，肉在固定项`） |
| 4 | **indexer** | 2.50 | 0.40 | **+2.10** | 9% | 逐行 ×5 + `indexer_topk` 单核地板（21GB/s） | `:7862` |
| 5 | **e4m3 直接路径** | +2.24 | 0 | **+2.24** | 10% | `quant_fp8(block=32)` + kernel 内 e4m3 解码 | `dspark-correctness-chain.md:416`（36.10 → 38.34） |
| 6 | **hc 链** | 2.96 | 1.00 | **+1.96** | 8% | **已是 rows=m**；缺口是核效率（**53GB/s = 峰值 0.07%**） | 账本 §2（全表最低带宽） |
| 7 | **attention** | 2.80 | 1.10 | **+1.70** | 7% | 逐行发起（`b·m` 已支持未用） | `:7805` |
| 8 | **head** | 1.12 | 0.19 | **+0.93** | 4% | fold 因数值错误禁用（sliced 已抵掉大部分） | `:1111` |
| 9 | 其它（compressor/AR/norm/engram） | 2.34 | 1.79 | **+0.55** | 2% | AR 是协议地板，其余逐行 | 账本 §2 |
| | **合计** | | | **+22.7** | **≈23 ✓** | | |

**读法**：
- **只有 20%（≈6.3ms）属于"routed experts 的 5× 字节"**——即任务描述的"天然不可压"部分。
  它在事实 #3 里被当作主因，但按 ms 算**它是小头**（8.3ms 里的 6.3ms 增量）。
- **80%（≈17.5ms）来自"本应 1× 却仍 5×"的族**：
  shared(8.3) + gate(2.75) + indexer(2.1) + attn(1.7) + head(0.93) + compressor(0.35) = **16.1ms**，
  正是 §2 表里 flag OFF / 未接线的 6 个族。
- **另有 ~4.7ms 是"折了也不快"**：投影 (2.7) + hc (1.96)——**字节已是 1×，时间却没到 1×**。
  这是 §4 物理机理的实证点。
- **2.24ms 是 e4m3 的精度代价**（换来"双字 = 0"，非优化项）。

---

## 4. 物理来源：为什么折叠也拿不到 1×（三层机理）

### 机理 1 —— 这些 kernel 根本不是带宽 bound ⇒ 少读字节 = 少花时间（×0）

| 族 | 字节 | 实测 ms | 达成带宽 | 占 HBM 峰值 |
|---|---:|---:|---:|---:|
| head（sliced / unsliced） | 6619MB | 1.12 | **5.9 TB/s** | **77%** ✅ |
| routed experts | 3133MB | 8.30 | 378 GB/s | 4.9% |
| 投影族 | 955MB | 3.70 | 258 GB/s | 3.4% |
| gate | 786MB | 3.44 | 229 GB/s | 3.0% |
| indexer | 435MB | 2.50 | 174 GB/s | 2.3% |
| attention | 262MB | 2.80 | 94 GB/s | 1.2% |
| shared expert | 885MB | 10.40 | **85 GB/s** | **1.1%** |
| hc | 157MB | 2.96 | **53 GB/s** | **0.7%** |

**⇒ 除 head 外，全表跑在峰值的 0.7%~4.9%。**
**M=1 解码的 GEMV/glue 族的真正约束是 issue/occupancy（`STATUS:7870` "SIMT gemv 是 compute-bound，L2 无关"），
不是 HBM 字节。** 因此 **mrows「权重读一次」的收益上限 = 该族字节时间的占比 ≈ 5%**，
而不是字节模型的 80%。**这是 23ms 差距最大的单个物理来源。**

> 反证：head 是唯一跑满带宽的族（5.9TB/s），而它只占 1.12ms/38ms = 3%。
> 凡是"字节有意义"的族都不重要；凡是要紧的族字节都无意义。

### 机理 2 —— mrows 是 weight-stationary，不是 row-parallel ⇒ FMA/激活解码仍 ∝ m

逐字读 `gemm_fp8_mrows_kernel`（`kernels/cuda/dsv41_kernels.cu:4958-5082`）：

```cpp
for (int kb = 0; kb < nb_k; ++kb) {              // k/32 = 160 次
    const float wv = s_lut[row_s[j]] * sb;       // 权重解码：1 次（C4 明确 hoisted）
    for (int r = 0; r < M; ++r) {                // M 次
        const float av = s_lut[s_a[r*k + j]] * s_as[r*nb_k + (j>>5)];  // 激活解码：M 次
        acc[r] += av * wv;                       // FMA：M 次
    }
}
```

- **折叠掉的是权重解码/加载**（每 `(row, kb)` 1 次，而不是 M 次）。
- **折叠不掉的是激活解码 + FMA**（各 M 次）。
- **指令数下限**：`(1 w-decode + 1 w-scale-mul) + M×(1 a-decode + 1 a-scale-mul + 1 FMA)`
  = `2 + 3M`，对逐行的 `M × (2 + 3) = 5M`：
  ```
  M=5:  (2+15)/(25) = 17/25 = 0.68×     ← 最好情况
  M→∞:  3M/5M = 0.60×
  ```
  **⇒ 完美折叠的物理下限是 0.6~0.68×，永远不可能是 1/M = 0.2×。**

- **二次惩罚**：smem 随 M 增长（`nwarps*k + 256*4 + M*nb_k*4 + M*k`），
  寄存器累加器 `acc[M]` 也随 M 增长 ⇒ **占用率随 M 下降**。
  这解释了为什么实测投影族只到 **3.70ms**（= EAGER 投影 ~2.6ms 的 1.42×），
  而不是"5 行只贵 10%"。

### 机理 3 —— per-row 家族还有 m× 的 launch 与 m× 的串行链

- **8 个 5× 族里有 4 个是显式 `for r in 0..m`**（attn / indexer / compressor / ring-window），
  它们的 launch 数真的 ×5。每个 launch 的 GPU 侧固定成本（dispatch + tail/drain）
  按仓库审计是 **~3.3µs**（`dspark-perf-400-plan.md §六`）。
- **attention 尤其结构性**：`append(r) → window(r) → compress(r) → select(r) → sparse_attn(r)`
  必须严格按行序（因果性，`chain_dev.rs:7736-7743` 把这条写成"THE ORDER IS THE WHOLE POINT"）。
  ⇒ 这 5 行是**串行链**，没有跨行并行可藏延迟；mrows 只能合并 launch，不能并行这个依赖链。
- **注意**：launch 的 **CPU submit** 已被图化 A/B 证明不是瓶颈（全 gate + 图化只 −1.21ms，
  `verify-ms-breakdown.md §修正`）——但 **GPU 侧的 per-kernel drain/tail 仍在**，
  且它随 kernel **数**（∝ m）增长，不随字节。

---

## 5. 对四个候选解释的裁决

| 候选 | 裁决 | 证据 |
|---|---|---|
| **a. mrows 未真正 dispatch** | ⚠️ **成立，且是 36.10ms 那次的直接原因**（H1） | `28515b6`（10:08）在 `dsv41_gemm_fp8_mrows` 入口加了 `if (g_gemv_a32) return 2;`（a32 默认 1）⇒ **SH_EXP_MROWS 与 proj mrows 同时被静默打回逐行**；`4c98b30`（11:42）用 a32 寄存器物化移除该 decline（`dsv41_kernels.cu:5084-5119` 注释自陈）。**但**：修复后的 38.34 仍没兑现 −8.3ms（36.10 + 2.24(e4m3) = 38.34）⇒ 修好 dispatch 也没换来时间 ⇒ 指向 H2。 |
| **b. kernel 本身不是权重驻留（折叠不省时间）** | ✅ **成立，物理主因** | 机理 1（达成带宽 0.7~4.9%，字节非约束）+ 机理 2（FMA/激活解码 ∝ m，下限 0.68×）。实测佐证：投影族 mrows **已生效**却只 1.42× EAGER；hc **已是 rows=m** 却跑 53GB/s。 |
| **c. launch 开销（3000 发 × 3µs = 9ms）** | ❌ **作为主因被否决** | 图化 A/B：`{SH_EXP, GRAPH, ROPE, P3A}` 全开只 **−1.21ms**（预期 −24ms）。CUDA async launch 已让 submit 与执行重叠；**38ms 几乎全是 GPU 执行时间**。 |
| **d. compressor/indexer 逐行** | ✅ **成立，但是小头** | indexer 2.5 + compressor 0.55 ≈ 3.05ms（§3 表第 4/9 行），占 23ms 差距的 ~13%。 |

**综合判词**：
> **23ms 差距 ≈ (a) 配置层 ≈16ms（flag 默认 OFF + 未接线的 6 个族）
> + (b) 物理层 ≈4.7ms（折了也不快的投影/hc）
> + (d) 逐行小项 ≈3ms
> − 重叠 ≈1ms。**
> **(a) 是"一行 flag / 一次接线"就能拿回的部分；(b) 才是真墙**——
> 它说明即使把 §2 的 8 个 5× 族全部折到 1×，落点也只是
> **38.34 − 16.1 ≈ 22ms（3.6× EAGER），不是 7.4ms（1.2× EAGER）。**

---

## 6. 修复路径

### 6.0 前置（阻塞其他一切）：一次 nsys 判别 H1/H2

```
DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_TIMING=1 \
  DSV41_SH_EXP_MROWS=1 DSV41_ROW_FOLD_GATE=1 DSV41_VERIFY_ROPE_MROWS=1 \
  + nsys profile serve，按 kernel 名聚合 verify 段
看 4 个数：gemm_fp8_mrows / gemv_bf16_v2_mrows / shared-expert 段 / expert_gate_up_fp4_batched(rows)
判据：
  launch 数 1200→240 但 verify_ms 只降 ~1ms  ⇒ H2（机理 1/2 成立，折叠无肉）
  launch 数 1200→240 且 verify_ms 降 ≥6ms    ⇒ H1（a32 是唯一阻塞，已修）
  launch 数不变                                ⇒ 仍未 dispatch，查剩余 decline 路径
```

### 6.1 动作表（按 收益 × 把握）

| # | 动作 | 落点 | 预期 | 前置 | 性质 |
|---|---|---|---|---|---|
| **1** | `DSV41_SH_EXP_MROWS=1` 默认翻转（A/B 后） | `chain_dev.rs:952` | **−8.3ms** | #0 | **最大单项**；代码已在、逐调用点等价已论证 |
| **2** | `DSV41_ROW_FOLD_GATE=1` 默认翻转 | `chain_dev.rs:918` | **−2.75ms** | #0 | 核已存在（`gemv_bf16_v2_mrows`） |
| **3** | indexer / compressor 多行化 | `:8259` / `:7806` | **−2.4ms** | — | 需新 mrows 入口（`indexer_topk` 的 stride 是运行期值） |
| **4** | `sparse_attn` 的 `b·m` 单发（grid 已支持） | `:7805`（launcher `kAttnMaxBM=8`） | **−1.7ms** | 行本地 `clen` 快照 | 注意：**合并的是 launch，不是行序**——内核内按 r 升序循环 |
| **5** | hc 链占用率（53GB/s） | `dsv41_glue.cu` `hc_mixes` | −1.0~−1.9ms | — | 纯 occupancy/kernel 效率 |
| **6** | head fold（K 序 parity 后） | `:1111` | −0.9ms | 数值 parity | 低优先（sliced 已抵掉大半） |
| **7** | **routed experts 换核**（tcgen05 mxf4 / TMA） | `dsv41_experts_mxf4.cu` | **−6.8ms** | ⚠️ **与 e4m3 互斥** | **唯一攻击"5% 峰值带宽"这个真墙的动作** |

**落点预测**：
```
38.34ms
 −8.3  (#1 shared)
 −2.75 (#2 gate)
 −2.4  (#3 indexer/compressor)
 −1.7  (#4 attn b·m)
 −1.9  (#5 hc)
 ≈ 21.2ms   ← 若 H1（折叠真的有效）
```
**21.2ms = 3.4× EAGER，仍远不是 1.2×（7.4ms）。**
剩下的 3.4× 全部落在 §4 机理 1/2：**投影 3.7 + routed 8.3 + hc 2.96 + 其它 ≈ 17ms
是"字节已 1× 或不可压、但核只跑 5% 峰值"的地板。**
⇒ **要越过它，只能换计算范式（#7 tcgen05 把 30 个 per-slot 小 GEMV 变成几个 M=128 的 GEMM），
不能靠 flag。**

### 6.2 ⚠️ 关键风险：若 #0 判为 H2，则 #1~#6 的预期全部作废

`36.10ms（全 gate，a32 仍 decline）→ 38.34ms（a32 已修 + e4m3）`：
**+2.24ms 恰好等于 e4m3 的代价 ⇒ a32 修复（mrows 真正上线）贡献 ≈ 0。**
这是 H2 的现场指纹。若 #0 确认 H2：
- **不要**按 §6.1 的逐项预期推进（会得到 ~0 收益，重演"v17→v21 四变体全中性"）；
- 把资源全部转到 **#7（换核，攻带宽/占用）** 与 **减少 kernel 数（融合，攻 per-launch 地板）**；
- 并复核 EAGER 的 6.15ms 是否也可被同一机理压低（它同样跑在 5% 峰值——
  说明"per-row 成本"本身有巨大的核效率空间）。

---

## 7. 待验证（无实测支撑）

| # | 断言 | 影响 | 建议 |
|---|---|---|---|
| M1 | 38.34ms 那次 `proj_mrows` / `SH_EXP_MROWS` 是否真的 dispatch | 决定 §3 的归因结构（H1 还是 H2） | **#0 的 nsys**（本文件最高优先级） |
| M2 | 账本 14.09GB 含**未切分 head**；切分后应为 8.30GB | 若按 8.30GB，效率损失从 1.24× 升到 2.1×（381→216 GB/s）⇒ **机理 1 的权重更大** | 确认 38.34 那次 `VERIFY_HEAD_SLICED` 的实际取值 |
| M3 | `gemm_fp8_mrows` 对投影族的实测时间（3.70ms 是账本推算） | 决定机理 2 的折扣系数（预测 0.68×，实测 1.42× ⇒ 需解释） | nsys 单核 `gemm_fp8_mrows` 的 dram__bytes 与 duration |
| M4 | `EAGER 6.15ms` 的逐族分解只有 3 个 bucket（gemv/expert/hc+AR+misc），无逐族值 | §3 表左列（EAGER 逐族）用了字节×达成带宽的推算 | EAGER 同口径 nsys |
| M5 | shared expert 的 17.3µs/发（隔离微基准）在 serve 是否适用 | 决定 §6.1 #1 的 −8.3ms 是否可信 | `DSV41_SH_EXP_MROWS` 的 serve A/B |
| M6 | attention per-row 的 2.8ms 是 split+merge 还是单块 | 决定 #4 的收益 | nsys 数 `sparse_attn_split/merge` |

---

## 8. 一句话交付

> **verify 的 5 行边际 = 8.05ms/行 = 1.31 × 一个完整 EAGER 步。**
> 这不是"routed experts 的 5× 字节"造成的（那只是 23ms 差距的 20%），
> 而是 **8/11 个族根本没有折到 1×**（3 个 flag 默认 OFF、4 个无 mrows 路径、1 个设计不可压），
> **叠加"折了也不省时间"的物理事实**——
> 这一族 kernel 跑在 HBM 峰值的 **0.7~4.9%**，字节从来不是约束，
> 而 `gemm_fp8_mrows` 折叠的只有权重解码，激活解码 + FMA 仍 ∝ m（指令数下限 0.68×）。
> **先做 #0（nsys 判 H1/H2），再决定是把 flag 翻一遍（H1，≈−17ms 到 21ms）
> 还是直接换核（H2，才有机会越过 3.4× 的地板）。**

---

*户部 · 只读分析，未执行任何 GPU 命令、未改动任何代码（本文件为唯一产出）。*
*本轮对代码的动作：读 `chain_dev.rs` / `dsv41_kernels.cu` / `STATUS.md` / git log 追时间线。*
*所有 ms/字节均标注了来源；推算项与口径冲突已在 §1/§7 显式列明。*
