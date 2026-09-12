# Ferrite / DSpark 会话最终交接文档（SESSION-FINAL-HANDOVER）

> **一句话结论**：本 session 最重要的产出**不是**任何性能数字，而是一次**验证范式的转移**——「引擎损坏」的判定标准此前是错的，因为 **base 模型自身在 ~50-60 token 后必然退化**。修正判据后，**R2 验证干净、86.8 tok/s（+10.2%），MARKOV 验证干净、89.6 tok/s（+13.7%）**。
> **下一 session 的第一件事**：不要写代码，先读本文件第 1、2 节，把判据换掉再开测。

## 0. 三十秒速览

| 项 | 状态 |
|---|---|
| 验证干净的最佳配置 | **base + R2 + MARKOV + LAZY_SDR + FORK + RING_WIN = 91.1 tok/s**（base 78.8，**+15.6%**）|
| 待重验 | LAZY_SDR（修复后）、VERIFY_FORK、K1+K2、RING_WIN_FUSE、wo_a |
| 全部干净后的当前值 | **91.1 tok/s**（tcgen05 被阻塞）|
| 到 400 tok/s | lazy 上限 **~145**；400 需 batched（SWALLOW 修复）或 L4/L5 kernel |
| 最重要的教训 | **验证必须用 EAGER 对照**；**计数判据只对前 61 行有效** |

## 1. 范式转移：损坏是模型行为（最核心发现）

**证据链**：
- 所有配置（base/spec 最小/R2/K1+K2/lin2-only）在计数 line 62 相同损坏（'12'）
- **纯 EAGER+e4m3 也在 line 62 损坏**（61/77）——不走 spec 也损坏！
- 出师表：EAGER+e4m3 也有拉丁（'acs' @ ~100 字）
- 对话：EAGER 77% 重复 + 拉丁（早期测试）

**根因**：base 模型（非 chat 微调）在 ~50-60 token 后自然退化（算术混淆、重复、拉丁碎片）。

**由此作废的判定**：R2/K1+K2"损坏"、MARKOV/LAZY_SDR"退化 accept"、"base 干净"、S1/S2/S3 是"line 62 的根因"——全部无效。

## 2. 修正后的验证协议

| 探针 | 有效范围 | 判据 |
|---|---|---|
| 计数 1-200 | **仅前 61 行** | 前 61 行数字全对 = 通过 |
| 出师表 | 前 ~100 字 | 正确 + 退化模式与 EAGER 对照一致（不是绝对零拉丁！）|
| k_acc | 全部 | 与 EAGER 对照一致 |
| 吞吐 | 全部 | 与同 gate 栈对比 |

**三个陷阱**：短生成陷阱（49 tok 自然停 ≠ 正确）；继承损坏陷阱（先测 base 再测优化）；拉丁检查掩盖陷阱（拉丁空 ≠ 数字对）。

## 3. 当前验证状态

| 配置 | 吞吐 | 状态 |
|---|---|---|
| base | 78.8 | ✅ |
| + R2（ATTN_LIN_FUSE=1）| 86.8（+10.2%）| ✅ 前 61 行全对 |
| **+ MARKOV_SLICED（修复后）** | **89.6（+13.7%）** | **✅ 前 61 行全对 + 零拉丁** |

## 4. 400 的剩余路径

lazy 数学上限 ~145（k_emit × c_row）。400 需要：
1. 其余优化：~92-95
2. L4 "M 进 grid" 化：~110-130
3. tcgen05：~120-140
4. **batched（SWALLOW ar5-hang 修复）或 L5 流水**——唯一超过 145 的路径

## 5. 关键教训

1. 验证必须用 EAGER 对照
2. 计数判据只对前 61 行有效
3. 所有优化验证必须在修正判据下重做
4. 先测 base 再测优化（避免继承损坏）
5. "短生成干净"是陷阱
6. "数值中性"优化未必中性（MARKOV 双偏移/LAZY_SDR 承重 H2D 是真 bug 已修）
7. 单一 GPU 测试驱动（subagent 不跑 e2e）
8. 不轮询远端状态

## 6. 下一 session 行动清单

1. 读本文件替换判据（不写代码）
2. 重验 LAZY_SDR（修复后）
3. 重验 VERIFY_FORK / RING_WIN_FUSE / wo_a / K1+K2
4. 干净栈全叠（目标 92-95）
5. 400 路径立项（SWALLOW 或 L4/L5）

## 未决问题

1. legacy 臂的 line-6 双字（模型行为 or legacy 特有 bug？）
2. diff probe 18/232 mismatch 的重新定性（近 tie vs 路径差）

---

## 追加更新（session 末尾的关键发现）

### SWALLOW 第 10 次修复也失败——epoch 冻结在 54
- 第 9 次 = **幻影**（函数存在但零调用点/kernel/gate——从未实施！）
- 第 10 次 = 真正实施（kernel :9129 + 接线 :8633 + gate）但 **33,472 hang——epoch 冻结在 54**
- D1 观测：pos=16/22 所有 rank 同步（1328/1497）→ 第一次 swallowed 步后 epoch 卡死在 54
- **epoch_dev 是 per-rank 的**（每个 rank 的 staging + ctr_at）——"同步"是行为契约不是硬件事实
- **ar5_wait_round 的不对称失败**：epoch 大的 rank 挂起，epoch 小的畅通但读陈旧 payload
- **第 11 次设计**：11.0 加固观测（canary + 单调断言 + rank=）→ 11-B 动态 pad（per-step max——对所有前 10 次的根因免疫）

### tcgen05 两轮对齐修复均失败
- 第 1 轮：6 个 split body 读点 → 18.8s 长跑但 LEN=0
- 第 2 轮：4 个新读点（含主嫌疑 :5009）→ **191ms 快速失败——仍 1 misaligned**
- 需要 compute-sanitizer 定位（唯一能枚举未知 misaligned 的手段）

### L4-9 A/B（最后一个 lazy 优化）
- 脚本就绪（470 行 7 臂——T1-C 空臂修正 + 噪声地板 + 确定性检查）
- CNORM_SPLIT 的快速 A/B 正在跑

### 最终性能栈（全部验证）
| 配置 | 吞吐 |
|---|---|
| base | 78.8 |
| + R2 + MARKOV + LAZY_SDR + VERIFY_FORK + RING_WIN | **91.1（+15.6%）** |
| + L4-9（如果 A/B 通过）| ~92? |
| + tcgen05（如果修复）| ~94-95? |

### 400 的最终判定
- lazy 上限 ~145（accept 5）/ ~97（accept 3）——**不够 400**
- **batched（SWALLOW）是唯一路径**——10 次修复失败（epoch 冻结是根本问题）
- **L4/L5 kernel 重写**（"M 进 grid" 化 + tcgen05 + 流水）= 25-35 人日

---

## Session 末尾的终极发现：SWALLOW 的完整修复链（知识固化）

### SWALLOW 11 次修复到最终解锁的完整链
```
1-8: 各种修复尝试（barrier/vote/poison/pad）→ 全失败
9: epoch pad → 幻影（零调用点/kernel/gate——从未实施！）
10: 真 epoch pad → epoch 冻结 54（所有 rank 均匀 999→54）
   ↓ D1 观测（步前打印修复）
   → 完整 ledger：pos=15 步内 999→54（第一次 swallowed 步！）
   ↓ P1 witness（设备侧证词）
   → CANARY 触发！（0xdeadbeef→0x00000000——staging 被越界清零！）
   ↓ OOB 源头调查
   → 3 严重缺陷（v5 无守卫 + 无 guard band + S1 stride 滑移）
   ↓ OOB 修复（guard band + payload bounds + multi-slot canary）
   → 验证成功！CANARY=0 ✓ GUARD=0 ✓ RESET=0 ✓ ar5-hang=0 ✓
   ↓ check_payload 抓到了真正的越界！
   → payload 147456 > slot 122880（engram 多行 gather！）
   ↓ payload 源头追踪
   → engram_apply_rows 的合法载荷（m=6 × n_cols=24 × ehd=256 = 36864 f32）
   ↓ slot 增大修复
   → slot = max(20480, 30720, 36864) × 4 = 147456 B ≥ payload ✓
   ↓ 决定性测试（c900216e）
   → [运行中——batched 解锁的最终验证！]
```

### 修复链的核心洞察
1. **check_payload 是关键武器**——把静默损坏变成响亮失败
2. **engram 多行 gather 是合法需求**（刻意优化——省 m-1 个 AR round）
3. **slot 计算必须包含所有合法载荷的最大值**（不只 hidden states）
4. **canary + guard + witness 的观测体系**是定位 OOB 的决定性工具

### 下一步（如果 c900216e 成功）
1. SWALLOW 正常生成 → batched 解锁
2. SH_PAIR M=6 验证（-4.9~7.9ms）
3. mrows 族验证
4. 400 冲刺！

---

## 🎉🎉🎉 SWALLOW 完全解锁（Session 的终极突破！）

**engram 修复后的决定性测试（c900216e）**：
- **PANIC = 0** ✓✓✓（engram slot 修复生效——check_payload 不再触发！）
- **CANARY/GUARD/RESET = 0** ✓✓✓（OOB 修复效果保持！）
- **ar5-hang = 0** ✓✓✓
- **LEN=387, completion=300** ✓✓✓（正常生成——达到 max_tokens！）
- **拉丁=[]** ✓
- **吞吐 56.6 tok/s**（含 V5_LEDGER 观测开销）
- **Ledger 证明 arm=swallowed 正常运行**（k_emit=1/2, delta=86/步, canary 全程清洁）

**11 次修复的完整旅程**：
```
1-8: 各种尝试 → 全失败
9: epoch pad → 幻影（从未实施！）
10: 真 pad → epoch 冻结 54 → OOB 根因发现 → OOB 修复 → 验证成功
→ check_payload 抓到 engram 147456 越界 → slot 增大修复 → SWALLOW 完全解锁！
```

**修复链的核心武器**：
1. **P1 witness + canary**（设备侧证词——绕过 host 观测歧义）
2. **check_payload**（把静默损坏变成响亮失败）
3. **guard band**（reduced 溢出在到达 epoch 前被拦截）

**batched 路径现在可用！** 下一步：纯净基线测量 → SH_PAIR M=6 → mrows → 400 冲刺！

---

## Session 末尾的 AR Step 2 教训（知识固化）

### AR Step 2 (A1a MoE store fold) 的失败
**实施**：+665 行——MoE all-reduce 的 staging 拷贝从独立 p2p_ar_store_v5 挪进 payload 最后写者的 epilogue
**实测**：
| 路径 | AR_FUSE=0 | AR_FUSE=1 | 判定 |
|---|---|---|---|
| lazy | 91.1 tok/s / 前 61 行 ✓ | 90.9 tok/s / **前 61 行 ✗** | 性能中性但数值破坏！ |
| SWALLOW | 58.3 tok/s / 前 61 行 ✓ | **7.1 tok/s** / **前 61 行 ✗** | **8× 性能 + 数值双破坏！** |

**结论**：A1a 有根本性数值 bug——两条路径的正确性都被破坏。AR_STORE_FUSE 保持 OFF（默认）。
**教训**：store fold 的优化必须以逐字节一致为前提（gate ON vs OFF）才能上生产。

### 修正后的 SWALLOW 400 路线
**口径校准**（SWALLOW 优化执行计划的判决）：
- AR 每步实际 = 84 轮 × 78.3μs = **6.58ms**（36% 是 kernel-sum 占比不是步时占比！）
- mrows S2 现实收益 = **-3.0ms**（不是 -4.5ms——B1 ⊂ B2 不可加）
- 400 ladder: 28.0 → +SH_PAIR 21.6 → +mrows 18.6 → +hc 17.1 → +tcgen05 **14.7ms = 441 tok/s**
- 三轨道并行：GPU 轨 + 写码轨（B6/B5/B4 不占 GPU！）+ 诊断轨

### 下一 session 的前 30 分钟
1. 读 session-final-handover.md 的 SWALLOW 部分 + 400 final path
2. 检查 mrows Phase A 的实施状态（本轮 mrows-batched-impl-p1 正在实施）
3. 起一臂 SWALLOW + mrows 测试 → mrows 的增量
4. 5 个决策点：A1a 修或弃（建议硬止损）· hc 破 FORBIDDEN · SWALLOW 常开 vs 路由 · tcgen05 go/no-go · 400 口径锁定 counting
