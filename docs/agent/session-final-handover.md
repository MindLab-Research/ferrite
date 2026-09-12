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
