# batched (SWALLOW m=6) 优化脆弱性结构分析 — 判词

> 来源：ministry-justice / batched-fragility-rootcause（2026-09-12，代码基线只读审查）。
> 审查对象：为什么同一族优化在 lazy m=1 全兑现（+15.6%）、在 batched m=6 上反复 6-8× 灾难。
> 本文档取代此前散落的"SWALLOW 脆弱"直觉描述，给出**结构性机制**与**量级判据**。

## 0. 一句话判决

**batched m=6 把「优化」从"改一条独立流水线"变成了"改一台已经在最优点上的机器"，并把「accept」从模型指标变成吞吐的乘数。** 因此：

- 任何改变 **m=6 共享结构** 的 knob，其代价项 ∝ m（lazy 下恒为 0）→ 灾难；
- 任何导致**数值**走偏的 knob，其后果被 accept 链放大 **÷k_emit（≈6~8×）** → 灾难；
- lazy m=1 恰好坐在**每个 knob 的"零"角**上——knob 要么 no-op，要么可控扰动。

## 1. 结构性根因：m=6 是"一个共享体的四个优化旋钮"

`dspark_spec_swallowed`（`chain_dev.rs:8887-8941`）vs `dspark_spec_lazy`（`:9347-`）：

| | lazy m=1 | SWALLOW m=6 |
|---|---|---|
| step_rows 执行体 | `step_rows_sync(&rows_in[i..=i])` 1 行一次，循环 k_emit 次（`:9232`） | `step_rows(&[anchor,d1..d5])` 6 行一次无条件（`:8896,8940`） |
| 每步 kernel 链 | k_emit × 40 层（独立、可错峰） | 1 × 40 层，m=6 同发 |
| 每步 AR 轮数 | k_emit×2×40 ≈ 480 轮，每轮 n=5120 | 2×40 ≈ 84 轮，每轮 n=6·5120=30720（`chain_dev.rs:11831` `fb(m*dim)`） |
| 每步代价 | ∝ k_emit（早退 `:9278-9282`） | **固定 C(6)**（6 行永远跑满） |
| 每步产出 | k_emit token | k_emit token |

四个旋钮（同时是收益源和脆弱源）：

1. **共享权重行** `gemm_fp8_mrows_kernel<M>`（`dsv41_kernels.cu:5334-5551`）：并行度只来自 n；M 进 `float acc[M]` + per-block M 行激活 staging。wkv shape smem=56064B 中 m·k=30720B 占 55%，占用压到 4 blocks/SM。smem/占用/寄存器预算由 M 决定，已配平。
2. **共享 AR 轮**：84 轮每轮是 6 行块的全局屏障。lazy 把同样数据摊成 480 轮可错峰。
3. **共享 accept 判定**（**放大器，本判词最重要发现**）：`spec_accept(&drafts, &rows, true)`（`chain_dev.rs:8961`）整块一次判定，而每步代价固定 C(6)：
   ```
   batched:  tok/s = k_emit / C(6)   ∝ k_emit   ← accept 是乘数
   lazy:     tok/s ≈ 1/c_row          ← accept 近似无关
   ```
   ⇒ 任何把 accept 打坏的数值改动，在 batched 直接吞吐 ÷k_emit(5~6) + 同步/回退开销 → 7~8×；在 lazy 只是那一行提前早退。
4. **共享 launch/占用预算**：1b/a32/act_cp16 等"装载方式"knob——batched 搬运已被 M-fold 配平 → 中性（B5/B4/1b 实测 −2.3% 噪声内）。

## 2. 缺陷清单（已定谳）

| 项 | 倍数 | 机制 | 代价项口径 | 判决 |
|---|---|---|---|---|
| **fold_r** | 6.2×（63.8→10.3） | `ng=ceil(M/fold_r)`，每 ng-block 重新 staging 同一行权重（`dsv41_kernels.cu:5449-5452`）；历史 auto 规则 `n≤1024→fold_r=1` 让 wkv(n=512)/w1,w3(n=288) 的 ng=6 | 权重流量 ×ng ∝ m | **永久 FORBIDDEN**；auto 已改保守 return m |
| **A1a AR_STORE_FUSE** | 8.2×（58.3→7.1） | store 折进 producer epilogue，正确性依赖 epilogue 把 partial 写进 peer staging；epilogue decline/走错臂 → store 不发生 → AR 读陈旧 staging → verify argmax 与 draft 不对齐 → k_emit 崩 1 而 C(6) 不变 | 数值错 × accept 乘数 | **永久 OFF**（且 MoE 载体生产不可达，收益上限 −0.08~0.16ms/步） |
| **R1 AR_SINGLE_POLL** | 7×（SWALLOW 10.7） | OFF 臂 120 blocks 全 poll（960 poller/轮）；ON 臂 block0 轮询→syncthreads→threadfence→写**单字无双缓冲** epoch[1]，其余 119 块等本地字。两跳发布在 6× 宽 grid 上时序脆弱（round e 慢块与 round e+1 的 block0 竞争同一字） | **代码上界只解释 ≤0.2ms/步 ⇒ 7× 落在 accept 崩塌 regime** | **高风险，判不投**；再议前提 = A0 探针 avg_spin 位移 + gate ON/OFF 逐字节一致 |
| **B5/B4/1b** | 中性（−2.3%） | 只改装载方式/fold 粒度，M-fold 已摊销 | — | 不加（B4 另有依赖 VERIFY_FORK 未单独验证） |

## 3. 为什么 lazy m=1 扛造（三个结构性理由）

1. **lazy 坐在每个旋钮的"零"角**：fold_r 的 `ng=ceil(1/fold_r)=1` 恒成立 ⇒ 结构上无法生效（"lazy 不适用"不是没测，是不可能有害）；M-fold 无共享 ⇒ 无权重重读可发生；act_cp16 的 m=1 staging 本来就小 ⇒ 无占用可救。
2. **爆炸半径**：lazy 逐行早退（`chain_dev.rs:9278-9324`），一行数值偏只杀那行 draft；batched 一次判定 6 行。
3. **收益形态**：lazy 加性（同族优化 +15.6%），batched 乘性且被 accept 调制（+9.4%，且剩下的余量恰好是"代价 ∝ m"档）。

> 统一表述：lazy 的优化 = 在 6 条独立流水线上各砍一刀，每刀只影响自己；batched 的优化 = 在一台 m=6 大机器上动一颗螺丝，动错一个 6 行同时停。

## 4. 什么"能"对 batched 有效 — mrows b2+b3 口径

唯一兑现的 +9.4%（63.8 tok/s）= **一阶折叠**：把**因 m>1 才存在**的逐行 launch 合成 m 行一发（`apply_rope_mrows`/`gemm_fp8_mrows` per-row 独立累加器链，数值逐位等价）。特征：不新增共享、代价项不随 m 增长、恰好是 SWALLOW 还没吃的肥肉。B5/B4/1b 同方向但余量已被 b2/b3 和 m=6 本身吃掉 ⇒ 中性。

## 5. 对 400 路径的启示（5 条）

1. **batched 的风险不是"不够快"，是"一个错误优化吞掉 6×"**。`tok/s = k_emit/C(6)` ⇒ 吞吐正是 accept 的读数。⇒ batched 路径正确性门禁必须升级为 **gate ON vs OFF 逐字节一致**，不接受"吞吐没掉所以没事"。
2. **杠杆只有两个：accept 与固定代价 C(6)**。但 AR 轮数合并（84→44）**拓扑不可能**（ar-r2-merge-impl-prep 判决：轮 B payload 在轮 A 结果传播前不存在——attn AR → hc_post → ffn 前端 → MoE → MoE AR 依赖链上相邻 AR 无任何独立对，跨层也不行；两轮复用同一 `s.o` 撞别名；`ar-l4l5:150/157`、`ar-further-optimization:159` 三处独立背书）。且 **-3.3ms 预算未验证**：78.3µs/轮是 nsys 读数（对 v5 自旋有 ~300× 放大嫌疑），账本 17.3µs → 砍 40 轮实际只有 0.69ms；A0 探针（device `clock64()`，capture-safe，禁与 nsys 同跑）是 AR 任何后续工作的**第一步**，需先补 SWALLOW site 分流小件（0.5 人日）。**R1 类扰动承载正确性协议的方向也错误**（收益上界 0.1~0.2ms/步，风险整块 accept 崩塌，本判词 §2 判不投）。若 A0 证实"等待是真的"，靶子是 **A2c rank 负载均衡**与**臂选择**（22.5µs/AR 的主体是"对端到达延迟"——合并轮不减到达延迟反而 payload 翻倍）。
3. **所有二阶 knob（fold_r/STORE_FUSE/SINGLE_POLL/act_cp16/a32 变体）在 batched 默认 OFF + FORBIDDEN**。共同特征：改动落在 m=6 共享体上，而不是落在因 m>1 才冗余的地方。
4. **tcgen05 misaligned 同源**：也是改 m=6 共享体的搬运器（TMA bulk + 16B 硬对齐）；batched 下对齐约束比 lazy 更紧（payload 大 6×、切分更多）。⚠️ 但 rank 7 叙事已被数学否证，见 `tcgen05-rank7-verdict.md`。
5. **路由而非二选一**：`batched 更好 ⟺ mean_k > B/c − 1`（B≈28ms 时阈值 3.55）。SWALLOW 常开 + lazy⇄batched 按任务路由（带 Schmitt 滞回）：计数型（accept 5）batched 划算；出师表/对话 batched 净亏。产品决策，需仲裁。

## 6. 模式分析（可复用判据）

- **代价票面记账**：本项目对"收益票面"记账充分、对"代价票面"记账不足。4 个退化项（fold_r/1b/B5/B4）都是代价项 ∝ m 而票面按单行算。⇒ **新 knob 的 A/B 票面必须显式写"代价项是否随 m 缩放"**。
- **量级纪律（最有用判据）**：观测倍数 ≈k_emit 或 ≈m ⇒ 先查 accept 再查流量；倍数 1.x ⇒ 才查协议/等待。R1 的 7× 落在 accept regime 而非等待 regime（代码把等待差上界压在 ≤0.2ms/步）——这条判据省 GPU 会话。
- **幻影门仍活跃**：`batched_400_v2.sh` GATES 不含 SH_PAIR_M；"lazy 中性 vs batched 7×"目前非同会话同 harness ⇒ 结论入库前必须 `/proc/<pid>/environ` 实读 + `nm -D` + nsys name 三证。
