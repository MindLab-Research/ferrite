# MTP verify 摊薄模型 — 正确口径（用户裁决 2026-09-13，取代此前所有 400 路线口径）

> 本文档是 **400 冲刺的唯一权威性能模型**。此前基于"verify 6 行成本 ≈ 4-5× eager"建立的全部
> 理论（15ms 阶梯 / accept-5 口径 / L4/L5 25-40 人日唯一路径 / lazy 上限 145）**全部作废**，见 §4。

## 0. 一句话

**MTP 最大的好处就是 verify 批量并发（m 行）比 eager 单行根本慢不了多少。**
单并发目标是 **step time ~8ms + accept length 2-3 ⇒ 375-500 tok/s**。

## 1. 正确的性能模型

- **单并发 decode 是 memory-bound**：eager 单步 6.33ms（实测）的主体是**权重读取带宽**，不是算力。
- **GPU 算力严重 underutilized**（batch=1 时 SM 大量空闲）。
- **verify(m 行) 共享同一份权重读**：6 行在同一份权重上并发，权重只读一次。
  ⇒ **verify(m) ≈ eager(1) + ε**，其中 ε = m 倍激活量的计算（相对权重带宽是小头）
  + 少量真正 per-row 的不可避免计算（如 MoE 路由后的 per-(row,expert) 对）。
- **合理落点**：draft（3 行小头模型，~1.5-2ms）+ verify（6 行 ≈ 6.5-7ms，即 eager+ε）
  + commit（~0.5ms）⇒ **总 step ≈ 8-9ms**。
- **tok/s = (acc + 1) × 1000 / step_ms**：acc 2~3 × step 8ms ⇒ **375~500 tok/s**。这就是 400 的路径。

## 2. 实测病灶（当前最大问题）

A0 会话实测（SWALLOW 63.8 栈）：`draft=3.87ms verify=28.17ms commit=0.47ms`。
**verify(6 行) = 28.17ms = eager(1 行) 6.33ms 的 4.45×——这不是物理极限，是实现未摊薄的病。**

摊薄应该发生而没有发生 ⇒ verify 内部存在**按行重复付出**的成本。排查方法（唯一正确方法）：
**逐 kernel 对比 eager(1 行) vs verify(6 行)——凡是耗时 ~6×（未摊薄）的就是病灶；正常摊薄后应该 ~1×。**

按嫌疑排序：
1. **MoE per-row 路由展开**：6 行各自 topk 选 expert ⇒ 6×topk 个 (row,expert) 对；不同行选不同
   expert ⇒ expert 权重按行重复读。MoE 的 grouped GEMM（tcgen05 e4x grouped，正在根修
   misaligned = w2 SF pitch 10B）是解这个的载体之一。
2. **per-row kernel 未进 m=6 块**：凡是"每行一发"的 kernel（40 层 × 6 行的 launch）都该进
   mrows（M=6 共享 staging）；对照 63.8 栈已开的 mrows gate 清单找漏网。
3. **图 launch 结构**：verify 一步应该是 1 张 m=6 的大图，不是 6 张单行图拼接。
4. **attention 侧 per-row 计算**（scores/dots/indexer/compressor）是否真正 m=6 批量化。
5. **draft 链**（3.87ms/3 行）：draft 图化覆盖率与 P3A 折叠完整性。

## 3. 400 的正确账（取代所有旧阶梯）

| 项 | 当前 | 目标 | 差距 |
|---|---|---|---|
| verify(6 行) | 28.17ms | **≈ 6.5-7ms（eager+ε）** | −21ms（摊薄修复） |
| draft(3 行) | 3.87ms | ~1.5-2ms | −2ms |
| commit | 0.47ms | ~0.5ms | — |
| **step** | **32.5ms** | **~8-9ms** | **−24ms** |
| acc | 2-3（用户校准） | 2-3 | — |
| **tok/s** | **~70（32.5ms, acc2.5）** | **375-500** | — |

**主战场 = 把 verify 从 4.45× 压回 ~1.05×**。这是实现问题（哪些 kernel 没共享权重读/没批量化），
**不是**"25-40 人日的 kernel 全面重写"——旧的 L4/L5 估时建立在错误前提上，作废重估。

## 4. 作废结论清单（错误理论，勿再引用）

| # | 错误表述 | 原出处 | 为什么错 | 正确版本 |
|---|---|---|---|---|
| 1 | "batched 每步代价是固定 C(6)，6 行永远跑满 ⇒ `tok/s = k_emit/C(6)`，C(6)≈28ms 是结构性代价" | batched-fragility-structural-analysis §1 | C(6) **应该 ≈ C(1)**（权重读共享）；28ms 的量级是实现未摊薄的病，不是结构常数 | `tok/s = (acc+1)/step_ms`，step_ms 目标 8ms；accept 仍是乘数这一点保留 |
| 2 | "verify 每行 4.7ms vs eager 6.33ms——mrows 共享收益 26%" | 本 session 多处 | 26% 摊薄远低于预期；正确预期是 verify 整体 ≈ eager 单行（每行边际 ~1.05ms） | 4.45× 是病灶指标，见 §2 |
| 3 | "400 = accept 5（H 区口径）+ 步时 ≤15ms" | swallow-400-final-roadmap §5、accept 报告矩阵 | 把 acc 5（只有计数 H 区前 61 行出现）当必要条件；正确口径是 acc 2-3 + 8ms | acc 2-3 + step 8ms ⇒ 375-500 |
| 4 | "400 唯一路径 = L4/L5 全面 kernel 重写，25-40 人日" | swallow-400-final-roadmap §6、AGENTS.md | 建立在"28ms 是物理事实"上；实际是未摊薄 bug | 主战场 = 摊薄修复（逐 kernel 找 6× 项），人日重估 |
| 5 | "lazy 上限 ~145 tok/s" | 多份文档 | lazy 的逐行 verify 同样可以 batched 摊薄；该上限基于逐行 6.15ms 的未摊薄假设 | 摊薄后 lazy/batched 路径都重估 |
| 6 | "2.5ms/行是硬仗"（verify 行成本 4.7→2.5） | verify-step-breakdown 任务书 | 框架错误：不是把每行成本从 4.7 磨到 2.5，而是把整块从 28.17 压到 ~7 | 见 §3 |

**保留有效的结论**（这些不受影响）：
- tcgen05 根因定谳（w2 SF pitch 10B，8/8 ranks）与根修（已实施待 GPU 复验）——真 bug；
- A0 探针判决（AR 稳态 spin 3.4µs，nsys 78µs 是自旋放大假象）——测量结论本身成立；
- 范式革命（模型自然退化、EAGER 对照判据）；
- fold_r 6×/A1a 8×/R1 7× 退化与 FORBIDDEN 清单（实证）；
- 1b act_cp16 逐位等价（实测）与激活回执；
- batched 脆弱性判词中"accept 是乘数、量级判据"等结构性观察。

## 5. 下一步（按此执行，勿回旧路线）

1. **逐 kernel 未摊薄排查**（最高优先）：nsys 或 kernel 级计时，eager(1) vs verify(6) 同 kernel
   耗时对比表——目标：找出所有 ~6× 项。这是 GPU 实验不是文档工作。
2. 每个病灶项：确认 m=6 批量化路径（mrows / grouped GEMM / 图合并），修复后逐 kernel 复测倍数。
3. draft 链（3.87→~1.5-2ms）与 verify 并行推进。
4. 每修复一批：同会话背靠背吞吐（step_ms 是第一指标，acc 只看是否被数值问题打坏）。
