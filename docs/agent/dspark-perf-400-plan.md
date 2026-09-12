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
  - **启用前置**（`verify_graph_gate`，chain_dev.rs:2901+）：`!eng_host()` && `!stats_dbg()` && `!phase_dbg()` && `compress_branch_steady()`（每个 compress source 都已提交过至少一组——稳态）&& `supports_dspark_snapshot()`（P0 kernel 在 .so 里）&& `supports_memset_async()` && **`(comm.is_none() || ar_v5())`**——**TP8 下必须让 AR 走 v5**（`DSV41_AR_V5`/`FERRITE_P2P_AR5` 的 opt-in）否则 gate 恒 false，图化静默不启用（"测了没变化"的陷阱）
  - A/B 口径：`verify_ms` 的对照（plan §3.3/§4）

## 三、阶段 2：吞掉主链步（6.15ms → 0）

**原理**（spec 的循环本质）: 上一轮 verify 的最后一行的 argmax（bonus）就是本轮的 anchor——**它从不需要一个独立的主链步去 forward**。

**新流程**（每轮）:
```
首轮（spec_primed == false，serve 的 prefill 之后第一次）:
  旧路径不变: step_dev(anchor, pos) → next(=pos+1 的 token) + tap
  然后 draft(next, pos+1) → d1..d5; verify 5 行 [d1..d5] @ pos+1..pos+5; accept(现状)
  结束把 spec_primed = true

后续轮（吞掉主链步）:
  输入: token = 上轮的 bonus（已在上一轮的 verify 里被 forward——KV 已 append）
        pos   = token 的位置（== pos_ctr）
        tap   = 上一轮 verify 的 anchor 行（行 0）的层输出 → 已拷进 dspark_tap
  1. draft_forward(token, pos + 1) → d1..d5        # 与现在完全相同的调用
  2. verify 6 行 [token, d1..d5] @ pos..pos+5（m = DSPARK_DRAFTS + 1）, ONE batched forward
     · verify_out[0] = token 行的 argmax = pos+1 的预测 —— 这就是本轮「本来由主链步提供的 next」
     · verify_out[j] = d_j 行的 argmax = pos+1+j 的预测
  3. accept: k_acc = 1; while k_acc <= DSPARK_DRAFTS && drafts[k_acc-1] == verify_out[k_acc-1] { k_acc += 1 }
     （k_acc ∈ 1..=6 —— 至少 1，因为 verify_out[0] 永远有效）
  4. commit: keep = k_acc（保留的行 = 0..k_acc-1，即 pos..pos+k_acc-1）
     —— dspark_commit 的 keep 语义在旧版是 k_acc（保留 0..keep），新版 keep = k_acc 的含义是「保留的行数」
     —— 需要给 rollback_keep/compress_replay 显式核对「保留行数」而非「最后一行的下标」（旧调用的 keep=k_acc 保留 0..k_acc 行 = 同一件事，但新版的行偏移是 pos 起而非 pos+1 起）
  5. emitted = verify_out[..k_acc]（k_acc 个 token，最后一个是 bonus）
  6. tap 的跨轮传递: verify 的 anchor 行（行 0）的层输出 → dspark_tap
     —— dspark_tap_r 的布局是 [slot][row][dim]（slot stride = VERIFY_ROWS*dim）
     —— 3 次 D2D（每 slot 拷 dim）：dspark_tap_r + slot*VERIFY_ROWS*dim + 0*dim → dspark_tap + slot*dim
  7. 返回: next = verify_out[0], k_acc, emitted, pos_ctr = pos + k_acc
```

**与当前实现的差异**:
- 当前: 主链 step_dev 提供 next + tap；verify 5 行 [d1..d5]（anchor 行的 KV 由主链步 append）
- 新: verify 6 行含 anchor（它的 forward 提供自己的 KV + tap）；主链步完全省去
- **收益**: 省 6.15ms/步；verify 多 1 行（多行化后 ≈ +1.6ms）→ **净省 ~4.5ms/步**
- **代价/风险**:
  1. 首轮的引导逻辑（`spec_primed` 字段 + reset 清零）
  2. tap 从 verify 的行收集（3 次 D2D，每轮一次）
  3. accept 链的下标（verify_out[0] 起 vs 现在 verify_out[j-1] 起）
  4. commit 的 keep 语义与行偏移（pos 起，不是 pos+1——**必须仔细核对 `dspark_commit`/`dspark_rollback_keep`/`compress_replay` 的行基址假设**）
  5. **snapshot 的 m = 6**（VERIFY_ROWS=8 ✓ 够）
  6. **draft 窗口的 seed**：旧版 seed_window 到 pos（t0 的位置）；新版 anchor 在 pos（已被 verify forward）——**seed 的位置语义要重新对齐**（anchor 自己的 KV 由 verify 的 anchor 行 append——**窗口的 seed 可能要移到上一轮**）

**实施顺序**（依赖）: 必须在阶段 1（多行化/图化）与 interleave 修复稳定之后（同一 chain_dev.rs，多 subagent 并发编辑）——实施前先确认工作树干净、无并行编辑。

## 四、阶段 3：accept 的进一步提升（可选）

- **confidence-gated verify**（官方 DeepSpec 的 per-request ragged verify）: 用 confidence head 的分数决定 verify 行数——高置信的 block 只 verify 前几行（省计算），低置信的全 verify
- **更长 draft**（block_size 5 → 固定 5 ✓ checkpoint 约定）——不能改
- **draft 的接受率**（accept 依赖 draft 质量——MoE ILV 修复 + interleave 修复后重测）

## 五、验收与止损

- 每阶段单独测（同会话背靠背 A/B）: `FERRITE_*` gate 的开关对照
- **止损门**: 阶段 1 的 verify 若 >12ms（多行化 + 图化后）→ 检查 kernel 的行独立性（C1-C5）
- **正确性红线**: 每阶段的文本（出师表逐字）+ `dspark_parity` 的 verify 行级对照（`verify_bad == 0`）
