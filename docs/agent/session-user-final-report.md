# Ferrite / DSpark 会话最终报告（面向决策）— 2026-09-12 深夜收官版

## 一、一句话结论
**本 session 最大的产出不是性能数字，而是一次验证范式的更正 + SWALLOW 的完全解锁**：过去判定的「引擎损坏」绝大多数是 **base 模型自身的自然退化**（EAGER 对照确认）；修正判据后 lazy 干净栈 **91.1 tok/s（+15.6%）**，SWALLOW（400 唯一路径）经 **11 次修复后完全解锁**，batched 最优 **63.8 tok/s（+9.4%）**。

## 二、性能最终成绩（双路径定型）

| 路径 | 吞吐 | 配置要点 | 判定 |
|---|---|---|---|
| **lazy m=1 干净栈** | **91.1（+15.6%）** | R2+MARKOV+VERIFY_FORK+RING_WIN+LAZY_SDR 等 23 gate | 全局最佳；天花板 ~145@accept5 |
| **SWALLOW m=6 + mrows b2+b3** | **63.8（+9.4%）** | 同栈换 SWALLOW_STEP + ATTN_MROWS_ROPE_NORM + ATTN_MROWS2 | batched 最优；**400 唯一路径** |
| SWALLOW 基线 | 56.7 | 无 mrows | — |
| SWALLOW 全 gate | 58.3 | — | — |

lazy 增量分解：R2(+10.2%) + MARKOV_SLICED(+3.2%) + VERIFY_FORK(+1.5%) + RING_WIN(+0.3%)。

## 三、范式转移（最重要的发现，勿重做）
- Base 模型在 ~50-60 token 后自然退化：计数 line 62 "重置到 12"、出师表 ~100 字拉丁、对话 77% 重复——**EAGER+e4m3 对照同点损坏**。
- 验证标准 v2：计数只对前 61 行有效；拉丁判定需 EAGER 对照；**gate ON vs OFF 逐字节一致**才是 kernel 改动的正确性判据（A1a 教训）。
- 之前 session 所有"R2/K1/K2 损坏"判定作废（假阳性）。

## 四、SWALLOW 11 次修复链（终局：完全解锁 ✓）
1-8 各类失败 → 9 幻影（epoch_pad 无调用点）→ 10 epoch 冻结 54 → **P1 witness：CANARY 0xdeadbeef→0x00000000（OOB 写清零 staging！）** → OOB 修复（guard band+bounds+multi-slot canary）验证 ✓ → check_payload panic 暴露真实源：**engram gather 载荷 147456B > slot 122880B** → engram slot 增大 → **SWALLOW 完全解锁**（0 hang/0 panic/300 token 正常生成/出师表零拉丁/全 gate 58.3）。
- 附带发现：**V5_LEDGER 观测本身破坏生成**（10 个 D2H 同步点）——生产配置必须无观测。

## 五、AR 优化线（已全部关闭，勿重开）
| 项 | 判决 | 证据 |
|---|---|---|
| R1 (SINGLE_POLL) | **无效** | SWALLOW 7× 退化（10.7）；lazy 中性（90.0） |
| R2 (attn+MoE 轮合并) | **拓扑不可能** | 依赖链：轮 B payload 在轮 A 结果传播前不存在（attn AR→hc_post→ffn前端→MoE→MoE AR），相邻 AR 无独立对；两轮复用同一 `s.o` 撞别名；三处独立背书。且 -3.3ms 预算未验证（78.3µs/轮 = nsys 读数有 ~300× 自旋放大嫌疑；账本 17.3µs → 40 轮实际 0.69ms） |
| R3 (PDL) | 低价值 | 只救 0.17-0.4ms（≤1.5%） |
| A1a (AR_STORE_FUSE) | **永久 OFF** | SWALLOW 8×（7.1）+ lazy 输出损坏；MoE 载体生产不可达 |

AR 是 SWALLOW #1 开销（36.0%，6.58ms/步 = 84 轮）和 lazy #1（27.1%）——但四条优化路全被判死。**400 的 AR 缺口需 L4/L5 级别的重写，不是 knob。**

## 六、三份判决（本 session 收官的结构性判决，前两份判词 + 两项 GPU 实验定谳）
1. **batched 脆弱性结构分析**（`docs/agent/batched-fragility-structural-analysis.md`）：batched m=6 把 accept 变成吞吐乘数（`tok/s = k_emit/C(6)`）⇒ 任何打坏 accept 的 knob 都是 6-8× 灾难；lazy 坐在每个 knob 的"零"角上所以扛造。**量级判据：观测倍数 ≈k_emit/≈m → 先查 accept；倍数 1.x → 才查协议。** 对 batched 有效的只有"一阶折叠"（mrows b2+b3 已兑现）；二阶 knob 全部 FORBIDDEN。
2. **tcgen05 "rank 7" 叙事作废 + 根因定谳**（`docs/agent/tcgen05-rank7-verdict.md` §10）：均匀分片在数学上产生不出 {7} 不对齐集合；`serve.rs:250` 只保留第一个 Err ⇒ "总是 rank 7"是上报竞态产物（历史 5/6/7 都出现过）。**第 5 轮判定实验（观测修复合入后）确凿定谳**：`[tp] step failed on 8/8 ranks` 全部同文本 cuda error 716 + ALIGN_AUDIT 唯一 violation = **w2 SF 行 pitch 10 字节**（rank 对称布局缺陷，row1&15=10 ⇒ 每行偏离 16B 网格）。根修 = SF 行 stride 与逻辑 k/32 解耦（行间 padding + 内核索引参数化），属 L4-3/L4-4 收尾。
3. **A0 探针判决：AR 已无肉**（`docs/agent/swallow-ar-first-step-design.md` §7）：site 分流落地后 SWALLOW 63.8 栈实测——**稳态 avg_spin = 6179 cyc ≈ 3.4µs/轮 ≪ 17.3µs 判读线**。nsys 账本（78.3µs/轮、6.58ms/步、36% kernel-sum）是**自旋放大假象**；真值 AR ≈ 0.3-0.6ms/步 ≈ 28ms 步时的 **~2%**。AR 四分支 T1-T3（负载均衡/协议/向量化）全部失去靶子——**SWALLOW 28ms 的肉在真实计算 kernel（MoE 17.4%/投影 15.1%/hc_dots）= L4/L5 的对象**。

## 七、400 的诚实评估（⛔ 2026-09-13 终局订正：本节口径作废）
> **⛔ 终局订正**：本节"15ms@accept5 / L4/L5 25-40 人日 / lazy 上限 145"全部建立在"verify(6行)=28ms 是结构性代价"的**错误前提**上，作废。正确口径（`mtp-verify-amortization-model.md`）：**verify(m) ≈ eager(1)+ε（权重读共享）⇒ 400 = step ~8ms + acc 2-3（375-500 tok/s）**；28.17ms = eager 4.45× 是**实现未摊薄的病**（逐 kernel 找 ~6× 项：MoE per-row 路由展开 / per-row kernel 未进 m=6 块 / 图 launch 结构）。以下仅存档。
- 400 ladder（需全部优化兑现）：AR 28→21.6→18.6→17.1→14.7ms = **441 tok/s @ accept 5**（AR 每步实际 6.58ms 不是 10.1ms）
- lazy 上限 ~145；**400 必须走 batched 且需 L4/L5 全面重写（25-40 人日）**——仅计数口径成立（accept 5）
- 60% 兑现 → ~300 tok/s
- **结构性风险**：batched 下一个错误优化吞掉 6×（判词一）⇒ 正确性门禁必须逐字节一致

## 八、需要拍板的三件事
1. **红线重定义**：绝对零拉丁在 >60 token 不可达（模型行为）。接受模型行为 / 换 chat 微调 / 限制测试长度？
2. **400 投入决策**：L4/L5 25-40 人日（唯一剩余路径）vs 接受 lazy ~145 vs **路由形态**（SWALLOW 常开 + lazy⇄batched 按任务路由，`batched 更好 ⟺ mean_k > B/c−1 ≈ 3.55`）——产品决策
3. **tcgen05 第 5 轮**：观测修复（进行中）合入后跑判定实验（8 行 vs 1 行 Err → 路径 A/B/C），成本 1 个 GPU session——建议做（若 SF 行 pitch 根修成立 +3-5% 且两条路径共享）

## 九、下一 session 启动清单
1. 读 `session-final-handover.md` + `swallow-400-final-roadmap.md`（198 行）+ 本报告
2. 检查 tcgen05-observation-fix 的实施（serve.rs Err 收集等 5 项）→ 主 agent GPU 跑第 5 轮判定实验
3. 检查 lazy-l45 的 A/B 设计（`docs/agent/lazy-l45-next-ab-design.md`）→ 决定 1b/B6/B5/B4 的 lazy A/B 跑批
4. mrows Phase A 状态检查 → 起一臂测试
5. 5 个决策：A1a 修或弃（已判弃）/ hc arm / 路由 / tcgen05 / 400 口径

## 十、FORBIDDEN 清单（生产禁用，永久）
`MROWS_FOLD_R`（6×）· `AR_STORE_FUSE`（8×）· `AR_SINGLE_POLL`（SWALLOW 7×）· `V5_LEDGER`（破坏生成）· `VERIFY_FORK` 仅 lazy 用 · `batched_400_v2.sh` 的 B400_MROWS_A arm（有 rsync 副作用）
