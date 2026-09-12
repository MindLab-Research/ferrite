# DSpark 性能路径：从 62ms/步到 ≤11ms/步（400 tok/s 的完整账本）

**日期**: 2026-09-12（Wave 2 性能阶段）
**当前**（batch verify 版）: 主链 step_dev 6.15ms + draft 5.85ms + verify(5 行) 38.68ms + commit 0.19ms ≈ **62ms/步 @ accept 1.12 → 18 tok/s**

## 一、收益来源的量化（为什么 verify 现在这么慢）

**核心事实**: verify 5 行 = 38.68ms = **7.74ms/行**，而单行图化的主链步 = **6.15ms/行**。
→ **batch 的权重复用收益完全没有兑现**：`step_rows` 的投影/MoE/head 都还是"逐行 5 次调用"（每次读一遍 40 层权重），加上**裸链无图**（~4000 launch/步 × ~5µs ≈ 20ms 纯发射开销）。
→ **这是 400 tok/s 的全部机会所在**：同一份权重被读 5 次是纯浪费。

**目标分解**（accept ≥4.5 时 400 tok/s 需步时 ≤11ms）:

| 项 | 现在 | 目标 | 手段 |
|---|---|---|---|
| verify 5 行 | 38.68 | **≤8** | ①图化（发射→0）②多行投影/head/MoE（权重读 1 次） |
| draft | 5.85 | **≤3** | P1/P3（head 多行 ✓、wo_a 融合、device pos counter） |
| 主链步 | 6.15 | **0（吞掉）** | verify 6 行 [anchor,d1..d5]，anchor 行的 forward 同时产出 tap |
| commit | 0.19 | 0.19 | — |
| **合计** | **62** | **~11** | **× 4.5 accept = ~410 tok/s** |

## 二、阶段 1：verify 的多行化 + 图化（38.68 → ≤8ms）

**边界划分（与正确性修复协调）**:
- **保持逐行**（单行调用形状 = parity 保障，launch 开销可忽略）: `sparse_attn`（b=1,m=1）、`indexer_topk`、`compress_rows` 的 pool/commit、clen 相关读取——**interleave-rows-fix 正在做这件事**
- **多行化**（权重读 1 次，数值域安全）:
  1. **投影**（q/kv/wq_b/wo 的 gemm）: 多行 GEMV 模式（P1 的 `head_gemv_bf16_mrows` 已验证逐位一致：C1-C5——同 K 序、独立累加器、无跨行重结合）
  2. **head**: `head_gemv_bf16_mrows`（已写、已接线 chain_dev.rs:3115——确认 .so 符号存在）
  3. **MoE expert**: `expert_gate_up_fp4_batched` 的 `rows=m`（行独立累加）+ down_reduce 的 rows=m
  4. **norm/hc 类**: `rmsnorm`/`hc_collapse`/`hc_post` 的多行版（rmsnorm-multrow subagent 正在回退逐行绕过）
- **图化**: `DSV41_VERIFY_GRAPH=1`（已实现，默认 OFF）——DRY→capture→replay，输入缓冲（ids_r/pos_rows）在图外刷新

## 三、阶段 2：吞掉主链步（6.15 → 0）

**原理**（spec 的循环本质）: 上一轮 verify 的最后一行的 argmax（bonus）就是本轮的 anchor——**它从不需要一个独立的主链步去 forward**。

**新流程**（每轮）:
```
输入: anchor（上轮 bonus 的 token）、hidden（上轮 verify 的 anchor 行 tap）、pos
1. draft(anchor, pos) → d1..d5          # 与现在相同
2. verify 6 行 [anchor, d1..d5] @ pos..pos+5, ONE batched forward
   - anchor 行的 tap → 本轮的 draft 输入（下一轮用）
   - verify_out[0] = anchor 行的 argmax = pos+1 的预测（对 d1 的验证）
   - verify_out[r] = d_r 行的 argmax = pos+1+r 的预测
3. accept: drafts[j] vs verify_out[j]（j=0 起，anchor 行的 argmax 就是 d1 的对照）
   k = 最长匹配数；emitted = verify_out[0..k-1]（k 个 token，含 bonus）
4. bonus = verify_out[k-1]；pos += k
首轮: prefill 后跑一次普通主链步（提供第一个 anchor + tap）
```

**与当前实现的差异**:
- 当前: 主链 step_dev 提供 next + tap；verify 5 行 [d1..d5]（anchor 行的 KV 由主链步 append）
- 新: verify 6 行含 anchor（它的 forward 提供自己的 KV + tap）；主链步完全省去
- **收益**: 省 6.15ms/步；verify 多 1 行（多行化后 ≈ +1.6ms）→ **净省 ~4.5ms/步**
- **代价/风险**: ①首轮的引导逻辑（prefill 后的一次主链步）②tap 从 verify 的行收集（`dspark_tap_r` 已存在——取 anchor 行的 slot）③accept 链的下标（verify_out[0] 起 vs 现在 verify_out[j-1] 起）④interleave 修复后 verify 的行形状变化

## 四、阶段 3：accept 的进一步提升（可选）

- **confidence-gated verify**（官方 DeepSpec 的 per-request ragged verify）: 用 confidence head 的分数决定 verify 行数——高置信的 block 只 verify 前几行（省计算），低置信的全 verify
- **更长 draft**（block_size 5 → 固定 5 ✓ checkpoint 约定）——不能改
- **draft 的接受率**（accept 依赖 draft 质量——MoE ILV 修复 + interleave 修复后重测）

## 五、验收与止损

- 每阶段单独测（同会话背靠背 A/B）: `FERRITE_*` gate 的开关对照
- **止损门**: 阶段 1 的 verify 若 >12ms（多行化 + 图化后）→ 检查 kernel 的行独立性（C1-C5）
- **正确性红线**: 每阶段的文本（出师表逐字）+ `dspark_parity` 的 verify 行级对照（`verify_bad == 0`）
