# verify 的算子级优化清单（ROI 排序）

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 任务：基于 arch-floor（L0=37.31ms → L3=20-22ms → L5=8-9ms）与 `verify-family-fusion` 设计，
> 列出**具体的算子优化项**（用户强调「算子优化也很重要，一起做」）。
> 代码基线：工作树 HEAD `69d3f43`（`crates/ferrite-models/src/dsv41/chain_dev.rs` /
> `crates/ferrite-models/src/dsv41/device.rs` / `kernels/cuda/*.cu` / `kernels/cuda/build.sh`）。
> 输入：`verify-architecture-floor.md`（工部，14:59）· `verify-family-fusion.md`（中书省，14:16）·
> `verify-ms-breakdown.md`（户部，含 §修正）· `batched-400-v2-remaining-roi.md`（工部，14:50）·
> `tcgen05-e4m3-grouped-expectation.md`（工部，15:23）· `NEXT-SESSION-HANDOVER.md`。

---

## 0. 判决（先读这五条）

1. **「算子优化」在 verify 里只有两条物理路径**，别把它们混成一锅：
   - **(A) 少发 kernel / 多发 warp**（族级融合、mrows、行进 grid）——削的是 **per-kernel 固定成本 + 低占用**；
   - **(B) 换核效率**（tcgen05 删 L1TEX μop）——削的是 **instruction-bound** 的每发执行时间。
   `arch-floor §5.2` 的判词：**mrows（今天所有 flag 的那一类）不动这两堵墙中的任何一堵**（它加寄存器累加器、不加 warp）。
   这解释了为什么「全 mrows 开」实测只有 **−1.21ms**（`verify-ms-breakdown §修正`）。

2. **翻 flag 的实测兑现度 ≈ 0，但「代码已就位、只差 A/B」的项仍然值钱**——因为它们便宜。
   本清单把「设计口径的预期收益」与「实测证据」**分列两栏**，不拿设计数字当承诺。

3. **最大的确定性单项是 SH_PAIR（shared expert 三段一体）**，因为它是**唯一**把「三段 → 一核」
   的骨架写好了的族（`dsv41_gemm_fp8_sh_pair`，`dsv41_kernels.cu:6512`，已接线到
   `shared_expert_mrows`，`chain_dev.rs:11444`）。**但它当前是 M=1**（逐行发），
   **不是**设计里的「1 launch/层」。

4. **routed experts（8.30ms）是唯一没有机械解的族**：字节不可压（30 assign 选 29 唯一专家），
   只能换核。**且 tcgen05 的 −6.8ms 是「gate/up + down 全换」的口径**——
   当前 e4m3 grouped 臂**只换 gate/up**（无 tcgen05 down kernel），所以本 arm 的**收益上限只有一半**（−1.0 ~ −3.8ms）。

5. **验收基线仍是「四段文本 + faults=0 + 同会话背靠背 A/B」**（`NEXT-SESSION-HANDOVER §2`）。
   算子优化全部走「**先 parity、后 A/B、默认 OFF**」；任何一个 gate 的 OFF 路径必须逐位回退。

---

## 1. verify 的算子分解（m=5，37.31ms，6224 launches）

来源：`verify-ms-breakdown §1`（字节账本 × 每核单价）+ `routed-expert-residual §2.3`（gateup/down 拆分）。

| # | 族 | ms | launches | 达成 BW | 5× 的根因（代码） | 算子优化落点 |
|---|---|---:|---:|---:|---|---|
| 1 | **shared expert** | **10.40** | 1000 | 85 GB/s | `moe_rows` 逐行 5 发（`quant1→gemm_fp8_mx2→swiglu→quant1→gemm_fp8_mx_add`） | **SH_PAIR 三段一体**（已接线，M=1） |
| 2 | **routed experts** | **8.30** | 400 | 378 GB/s | SIMT fp4 LUT gather；`grid.z=rows` 已折但字节 ×5 | **tcgen05 e4m3 grouped**（仅 gate/up） |
| — | └ gate/up | 4.82 | — | — | 200 行 × 24.1µs | ↑ |
| — | └ down | 3.48 | — | — | 200 行 × 17.4µs | **无 tcgen05 down kernel** |
| 3 | **投影族** | 3.70 | 2000 | 258 GB/s | 每层 4 投影 × 5 行；`proj_mrows` 已存在但**未层内合并** | 层内 m 合并（W6，未做） |
| 4 | **MoE gate** | 3.44 | 200 | 229 GB/s | 逐行 `gemv_bf16`；`ROW_FOLD_GATE` 默认 OFF | **GATE_MROWS**（已就位） |
| 5 | **hc 链** | 2.96 | 400 | **53 GB/s** | 10 发/层；A1/A2 全 OFF | **A1+A2**（truncate 已修） |
| 6 | **attention** | 2.80 | 880 | 94 GB/s | 行循环 `ring_append+window_idxs+sparse_attn_orope` | **ATTN_MROWS**（生产 decline） |
| 7 | **indexer** | 2.50 | 230 | 174 GB/s | `indexer_rows_one` 逐行 5 发 | **INDEXER_MROWS front**（已就位） |
| 8 | **all-reduce v5** | 1.40 | 160(2发) | — | 协议地板（80 次/步） | `VERIFY_AR_FOLD`（依赖 A1） |
| 9 | **head** | 1.12 | 10 | **739 GB/s** | 逐行 `gemv_bf16`（v1 序） | **HEAD v1 mrows**（已就位） |
| 10 | **compressor** | 0.55 | 80 | 667 GB/s | 逐行 `lin_f32`×2 + pool/commit | **COMPRESSOR_MROWS**（已就位） |
| 11 | **engram** | 0.42 | 24 | — | 小项 | — |
| 12 | **norm+cast+quant** | 0.20 | 680 | — | 逐行 rope/quant（launch 折进别族） | `src_stride` 尾参（W7） |
| | **合计** | **37.79** | **6224** | 381 GB/s | 模型自校准 +1.3% | |

> **口径提醒**：37.31ms 几乎全是 **GPU kernel 执行时间**（图化只 −1.5ms，证明 CPU submit 已被
> async launch 隐藏）。所以「算子优化」= 压 **GPU 侧**时间，不是压 submit。

---

## 2. 按 ROI 排序的算子优化清单（交付物）

**ROI 定义 = 预期节省 ms ÷ 实施成本**。表内「预期 ms」标 (实测)=已上机证据、(设计)=文档口径未实测。

| 排名 | 算子优化 | 当前 ms | 优化后 ms | 节省 | 成本 | **实施状态** | **下一步** |
|---|---|---:|---:|---:|---|---|---|
| **1** | **GATE_MROWS**（MoE gate 5 发 → 1 发） | 3.44 | **0.7**（设计） | **−2.75** | **0** | ✅ 已实现 + parity（`tests_gate_mrows.cu`）；`DSV41_GATE_MROWS`/`DSV41_ROW_FOLD_GATE` **默认 OFF**（`chain_dev.rs:1225`） | **上机 A/B（零成本）**：开 flag → 看 `verify_ms` + 四段文本 |
| **2** | **A1+A2 hc 折核**（10 发/层 → 部分/1 发） | 2.96 | **1.3~1.7** | **−1.3~−1.7** | **0.5–1 人日** | ✅ 代码全在：A1 `HC_VERIFY_FUSE`（**反向默认** `v=="1"`）+ A2 `HC_FRONT_ROWS`（默认 OFF）；**A2 的 `bf16_truncate` 坑已在 `6f5de2b` 修**（verify 调用点传 `false`，`chain_dev.rs:8656`）；A1-a 的 `collapse_norm_rows` 也已 `false` | **上机 A/B**：先 A1（`HC_VERIFY_FUSE=1`）确认零拉丁 → 再 A2（`HC_FRONT_ROWS=1`）；顺带复核 A1 的反向默认 |
| **3** | **INDEXER_MROWS（front 半）** | 2.50 | **1.0~1.5**（设计） | **−1.0~−1.5** | **0** | ✅ 已实现（`indexer_front_rows`，`chain_dev.rs:9838`）；`DSV41_INDEXER_MROWS` 默认 OFF；**逐位等价已论证**；`indexer_rows_one` 已带 `mrows_clen: Option<ptr>` | **上机 A/B（零成本）**：开 flag；用 `vg`/launch 计数复核（select 半仍 2/行，别按 230→40 编预算） |
| **4** | **HEAD v1 mrows** | 1.12 | **0.25~0.40** | **−0.7~−0.9** | **0** | ✅ 已实现（`dsv41_gemv_bf16_v1_mrows`，`dsv41_glue.cu:440`）；`DSV41_VERIFY_HEAD_MROWS` 默认 OFF；`verify_head_mrows_note` 有一次性日志 | **上机 A/B**；⚠️ **不要与 SWALLOW_STEP(m=6) 同开**（`argmax_rows` 与 rows=6 死锁，见 `192ae83`） |
| **5** | **COMPRESSOR_MROWS** | 0.55 | **0.25~0.35** | **−0.2~−0.3** | **0** | ✅ 已实现（`dsv41_compressor_fused_mrows`，`dsv41_kernels.cu:7784`）；`DSV41_COMPRESSOR_MROWS` 默认 OFF；`seqlen==1` 逐位复现单行核 | **上机 A/B**（收益小，随同 #1~#4 一起测） |
| **6** | **SH_PAIR 三段一体（M=1 版）** | 10.40 | **~8.6** | **−1.8** | **0（已接线）** | ✅ 已接线：`sh_exp_fused()` gates（`DSV41_SH_EXP_FUSED=1` **AND** `DSV41_SH_PAIR=1`，均默认 OFF，`chain_dev.rs:1366/1329`）+ `supports_sh_pair()`；**已调用 `shared_expert_mrows`（`:11444`）**。⚠️ **M=1**：每行 1 发，25→7 launches/层（不是 1） | **上机 A/B 先验 M=1 版**（它是 `sh_exp_mrows` 的**首选替换**，失败回退 mrows）；同时确认 `dsv41_gemm_fp8_sh_pair` 在 `.so` 里 |
| **6b** | **└ SH_PAIR template\<M\>（三段一核）** | （内含） | **2.5~5.5** | **−4.9~−7.9** | **3.5 人日** | ❌ **未做**（`gemm_fp8_sh_pair_kernel` 只支持单激活行；M 维/`co_res` cap 需新写） | 若 #6 A/B 通过且证明「fused grid 比 mrows 快」，再做 `template<M>`；**这是 shared expert 的真正 5×→1×** |
| **7** | **tcgen05 e4m3 grouped（routed gate/up）** | 4.82（gateup） | **1.0~2.8** | **−1.0~−3.8** | **4–5 人日 + GPU parity** | ⚠️ **骨架已编入 .so**（`build.sh:101/113`）+ `ld_uint4_a16` 对齐修复已落（`5332129`）；但**从未有过 GPU parity**（smoke 存活但空输出/错位）。5-gate 链：`E4M3 + TCGEN05_E4M3 + EXPERT_GROUPED + GATEUP_FUSE=0 + ILV=0` | **重跑 smoke（对齐修复后）→ e4m3 臂 GPU parity → 再谈收益**；down 仍是 SIMT（无 down 核） |
| **8** | **ATTN_MROWS（b·m 单发）** | 2.80 | **1.0~1.5** | **−1.3~−1.8** | **3–4 人日** | ⚠️ ABI 5 已实现 5 个 kernel 体（`clen_rows`/`idx_stride`/`row_step`），但**生产恒 decline**：`world>1`（缺 `row_pitch`）或 `pos+m-1>=win`（环回绕 = defect #2） | **不在测试窗口动**。前置：① `row_pitch` 尾参（TP>1）② ring/window 块内 r 升序合核或块前 ring 快照 |
| **9** | **VERIFY_AR_FOLD**（hc_post 折进 pubred 尾） | 1.40 | 1.16 | **−0.24** | **0.5 人日** | ✅ 已实现（`DSV41_VERIFY_AR_FOLD` 默认 OFF，`chain_dev.rs:11893`）；**依赖 A1**（`hc_verify_fuse()`） | **排在 #2 之后**（A1 重新 ON 才可开）；AR 已是 2-kernel 不是 3 |

**一句话取舍**：**#1~#5 是「零成本 flag + A/B」的一整批**（合起来设计口径 **−6~−7ms**），
**#6 是唯一「代码已就位、结构更高」的单项**，#7 是唯一能越「5% 峰值」这堵墙的换核，
#8/#9 是高风险/低收益的尾项。

---

## 3. 三类算子的性质区分（为什么不能按「省 ms」直接排）

| 类别 | 削什么物理量 | 覆盖族 | 代表项 | 实测证据 |
|---|---|---|---|---|
| **(A) 融合 / 多发 warp** | per-kernel 固定成本 + 在飞 warp 数 | shared/gate/attention/indexer/compressor/hc | SH_PAIR、GATE_MROWS、INDEXER、COMPRESSOR、A1/A2 | 全 mrows 开 **−1.21ms**（说明「只折字节」不行，要真融合） |
| **(B) 换核（删 μop）** | L1TEX μop 吞吐 | routed experts | tcgen05 e4m3 grouped | 未兑现（GPU parity 未过） |
| **(C) 纯接线（省 launch，不改核）** | launch 数 / 边界 | head/indexer front/norm/quant | HEAD v1 mrows、INDEXER front | 理论 −1~−2ms，**未单独 A/B** |

**关键机理（`arch-floor §5.1/§5.2`）**：
```
per (row, kb):  1 w-decode + 1 w-scale-mul + M × (1 a-decode + 1 a-scale-mul + 1 FMA)
逐行: M × 5 指令       折叠后: 2 + 3M       ⇒ M=5: 17/25 = 0.68×（物理下限，拿不到 1/M）
```
⇒ **mrows 的指令模型天花板就是 0.68×**，且它**不加 warp**（不攻击占用墙）。
**真正的 5×→1× 只有「M 行进 grid / block」+「阶段走 smem」的族级融合**（family-fusion §2.2）。

---

## 4. 与 arch-floor 台阶的对应（每个算子优化值多少台阶）

| 台阶 | 变化 | ms | 本清单里的算子 | 备注 |
|---|---|---:|---|---|
| **L0** | 今天 | 37.31 | — | 实测 |
| **L1** | 只翻 flag（全 mrows 族 + hc A1/A2 + graph） | 31~33 | #1 GATE_MROWS + #3 INDEXER + #2 hc + #6 SH_PAIR + #4/#5 | 实测子集 = −1.21ms；**L1 的收益最不确定** |
| **L2** | + tcgen05（routed 换核） | 25~26 | #7 tcgen05 e4m3 grouped | 阻塞 = 4 gate 联合 + e4m3 parity；**只 gate/up 则只有一半收益** |
| **L3** | + 族级融合（N÷5） | 20~22 | #6b SH_PAIR template\<M\> + 投影层内合并 + attention b·m | **真地板**（保留今天核效率） |
| **L4** | + 占用/MLP 修复（行进 grid、K-split、TMA 深度） | 11~14 | 需新写（非本清单现有项） | v19/v21/v24 证明「只动一个因子」无效 |
| **L5** | + kernel 内 cp.async 流水 + 满 wave | 8~9 | 需第二轮 kernel 重写 | 400 的算术地板；**未开工** |

**算子清单的定位**：把 L0→L1 的**便宜部分**吃干净（#1~#6，零到低风险），
并**启动** L2（#7 的 parity 是唯一开关）与 L3 的**最大单项**（#6b）。

---

## 5. 诚实校准（必须写在账上的三件事）

1. **「全 mrows 开」的实测 −1.21ms 是本清单最大的反向证据。**
   #1/#3/#4/#5 的「预期 ms」全部来自**设计口径**，它们**与 SH_EXP_MROWS 的两次零收益同源**
   （instruction-bound，不是 launch-bound）。⇒ **#1 的 −2.75ms 是假设，不是承诺**；
   必须用一次 nsys「按 kernel 名聚合」钉死（`verify-ms-breakdown §9-U1`）。

2. **SH_PAIR 的 M=1 版可能不比 mrows 快**：它的 launch 数（7/层）**比 mrows 路径（6/层）还多 1 发**。
   它的真正价值是**去掉 `sh_act_r` 的 global 往返 + swiglu 的独立 launch**（阶段融合），
   但那一点在 m=1 下被逐行发核稀释。⇒ **M=1 A/B 若中性，不要下结论「融合无效」——
   要直接做 `template<M>`**（那才是 family-fusion §3.1 的设计点）。

3. **tcgen05 的 −6.8ms 不能记在本 arm 头上**：本 arm 只换 gate/up（4.82ms），
   down（3.48ms）仍是 SIMT。**修正后的预期 = −1.0 ~ −3.8ms**（乐观/现实），
   且**数值红线（拉丁/双字）必须重验**——新 GEMM 引擎的舍入与 SIMT 不同，**不是逐位相等**。

---

## 6. 建议执行顺序（本会话可做）

```
第 1 波（零成本，全部上机 A/B）：
  GATE_MROWS + INDEXER_MROWS + HEAD_MROWS + COMPRESSOR_MROWS + NORM_MROWS + VERIFY_ROPE_MROWS
  → 一次 nsys 按 kernel 名聚合，数 launch 数 + 每发 µs
第 2 波（低成本，hc）：
  HC_VERIFY_FUSE=1（验零拉丁）→ HC_FRONT_ROWS=1 → 看 [dsv41] launch 计数
第 3 波（SH_PAIR）：
  DSV41_SH_PAIR=1 + DSV41_SH_EXP_FUSED=1（M=1 A/B）→ 通过则排 template<M>（3.5 人日）
第 4 波（tcgen05）：
  重跑 smoke（对齐修复后）→ e4m3 臂 GPU parity → 5-gate 链
```

**第 1 波的三个铁律**（否则数据无效）：
- 每次 A/B **只动一个 gate**（`ROW_FOLD_GATE` 反向默认 `v != "0"`，易踩空）；
- 判据 = **四段文本逐字 + faults=0 + 同会话背靠背 p50**（`scripts/dsv41_serve_ab.sh`）；
- 缺 nsys 证据时**不要对某个 mrows gate 下「declined」结论**（5 个 gate 静默 `return Ok(false)`，
  只有 HEAD/ATTN 有 one-shot 日志，`batched-400-v2-prediction §3.2`）。

---

*工部 · 只读分析 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*所有 ms/launch 均标注来源；「预期 ms」未标 (实测) 的均为设计口径，已在 §5 显式降级。*
*代码行号以 HEAD `69d3f43` 为准，读代码时以函数名为准。*
