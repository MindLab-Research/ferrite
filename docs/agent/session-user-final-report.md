# Ferrite / DSpark 会话最终报告（面向决策）

## 一、一句话结论
**本 session 最大的产出不是性能数字，而是一次验证范式的更正**：过去判定的「引擎损坏」绝大多数是 **base 模型自身的自然退化**。修正判据后，干净栈从 **78.8 → 91.1 tok/s（+15.6%）**。

## 二、需要拍板的三件事
1. **红线重新定义**：绝对零拉丁在 >60 token 生成下不可达成（EAGER 对照也违反——模型行为）。选项：接受模型行为 / 换 chat 微调模型 / 限制测试长度
2. **400 的路径**：lazy 上限 ~145（accept 5）/~97（accept 3）——不够 400。batched（SWALLOW）是唯一路径——9 次修复失败，第 10 次方向分析中
3. **下一步优先级**：tcgen05 重测（对齐修复——如果成功 +3-5%）vs SWALLOW 第 10 次修复 vs L4/L5 kernel 工作

## 三、性能成果（全部修正判据下验证）
| 配置 | 吞吐 | 增量 |
|---|---|---|
| base | 78.8 | — |
| + R2（ATTN_LIN_FUSE）| 86.8 | +10.2% |
| + MARKOV_SLICED | 89.6 | +3.2% |
| + VERIFY_FORK | 90.8 | +1.5% |
| **+ RING_WIN_FUSE** | **91.1** | **+0.3%（总 +15.6%）** |

## 四、范式转移（最重要的发现）
- Base 模型在 ~50-60 token 后自然退化：计数 line 62 "重置到 12"、出师表 ~100 字拉丁、对话 77% 重复
- **EAGER+e4m3 对照确认**——纯 EAGER（无 spec）也损坏！
- **之前所有"优化损坏"判定作废**——R2/K1/K2/MARKOV/LAZY_SDR 的"损坏"都是模型行为

## 五、9 个真 bug 修复
MARKOV 双偏移、LAZY_SDR 承重 H2D、S1/D1 DIRECT 双计、D2 池饥饿、S3 路由锁定、A4 单块轮询、indexer_topk 烧入 n_pos、K2 竞态、K1 decline 路径

## 六、400 的诚实评估
- 当前 91.1 + tcgen05（如果修复）≈ 94-95
- lazy 上限 ~145（需要 L4/L5 全面化——15-25 人日）
- **400 必须走 batched**——SWALLOW 9 次修复失败（ar5-hang 多根因）

---

## Session 最终完整总结（追加——所有最终数据）

### 一、性能最终成绩
| 配置 | 吞吐 | 验证 |
|---|---|---|
| base | 78.8 | ✓ 前 61 行 |
| **最终干净栈** | **91.1-91.4（+15.6%）** | **✓ 前 61 行 + 零拉丁** |

**最终干净栈的 gate**：R2(ATTN_LIN_FUSE) + MARKOV_SLICED + LAZY_SDR + VERIFY_FORK + RING_WIN_FUSE + base

### 二、SWALLOW 的 11 次修复历程
| # | 尝试 | 结果 |
|---|---|---|
| 1-8 | 各种（barrier/vote/poison 等）| 全失败 |
| 9 | epoch pad（**幻影**——从未实施！）| ❌ |
| 10 | 真 epoch pad（常数 81）| ❌ epoch 冻结 54 |
| **11** | **动态 pad（RankMax）** | **✓ 0 hang！但 epoch 仍 54 + EOS 提前** |

### 三、tcgen05 的 2 轮修复
- 第 1 轮：6 个读点 byte-fallback → 部分生效
- 第 2 轮：4 个新读点 → **仍 1 misaligned**（TMA bulk 的 16B 硬对齐无法 fallback！）

### 四、最关键的 3 个发现
1. **范式转移**：所有"损坏"是模型行为（base 模型 ~50-60 token 后自然退化——EAGER 对照确认）
2. **SWALLOW 的 epoch 54 谜**：epoch 从 1328 均匀降到 54（不是写入而是某种恢复/切换机制）
3. **动态 pad 消除 hang**：0 ar5-hang（10 次失败后首次！）但计算仍被破坏

### 五、用户需要决策的 3 件事
1. **红线重定义**：零拉丁在 >60 tok 不可达成（模型行为）——接受/换模型/限长度？
2. **400 的路径**：SWALLOW 修复（epoch 54 谜待解）vs L4/L5 kernel 重写（25-35 人日）
3. **下一步优先级**：epoch 54 调查 vs tcgen05 修复 vs L4/L5 启动

### 六、下一 session 的启动清单
1. 读 session-final-handover.md + 400-FINAL-PATH 文档
2. 检查 epoch54-source-and-stop 的调查结果（如果完成）
3. 决定：继续 SWALLOW 调查 or 启动 L4/L5
