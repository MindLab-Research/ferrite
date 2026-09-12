# 从 ~85 到 400 的最终前沿分析（两堵墙 · 三条路径 · 一个最高 ROI）

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 代码基线：工作树 HEAD `f88c7c1`（含 K1 decline-path 修复）。
> 输入（现场核对）：`lazy-verify-95ms-gap-analysis.md` · `lazy-verify-optimization-path.md` ·
> `swallow-unlocked-next-plan.md` · `plan-b-swallow-readiness.md` · `final-400-battle.md` ·
> `l4-l5-kernel-path.md` · `sh-pair-template-m-design.md` · `dspark-sglang-real-data.md` ·
> `verify-specific-fusion-kernel-design.md` · `ar-further-optimization.md` · `verify-architecture-floor.md` ·
> `perf-roadmap.md`（CUTLASS 级 MMA gemv 节）· `dspark-correctness-chain.md`（本 session 日志）。
> **本机无 GPU ⇒ 所有 ms 标了来源（实测 / 账本推算 / 设计口径）。**

---

## 0. 判决（先读七条）

1. **lazy 的天花板不是 145，是 EAGER 的 162.6 tok/s。**
   `tok/s = k_emit / (k_emit·c_row + d + c)`，当 `k_emit → ∞` 时收敛到 `1/c_row`。
   c_row = 6.15（= EAGER 本身，EAGER 也是 m=1 逐 token）⇒ **lazy 的数学极限 = EAGER 的速率**。
   145 只是 accept 5（k_emit=6）时的取值，不是另一个天花板。**两个数都 < 400 ⇒ lazy 是局部最优，不是 400 的路径。**

2. **400 必须同时跨过两堵墙，缺一不可**：
   - **墙 1 —— batched（m ≥ 2 的行间权重共享）**：把 `k_emit · c_row` 的乘积关系变成近平台关系（§3.1/§3.5）。
   - **墙 2 —— SIMT → MMA（去掉 instruction-bound）**：把每元素 ~2–3 条 warp 指令换成每 MMA 一条（§3.2）。
   只跨墙 1 ⇒ sglang 实测的 verify 水平 7.3ms 仍拿不到（ferrite 的 SIMT 地板 ≈15.5ms）；
   只跨墙 2 ⇒ m=1 的 MMA 是 1/16 的 tile 浪费，收益被吃掉（§3.2 末）。

3. **"batched 被 ar5-hang 阻塞"这句话必须精确为"batched + CUDA graph 被阻塞"。**
   SWALLOW **nograph**（`SWALLOW_STEP=1` + `VERIFY_GRAPH=0`）已实测 **0 ar5-hang + 零拉丁 ✓**
   （`dspark-correctness-chain` 的 SWALLOW 突破节）。hang 只出现在图开启后，
   三个**结构同源**的臂分歧源：verify graph 四臂（唯一奇点 = Capture 的 2 次 barrier）、
   draft graph 四臂（`plan-b-swallow-readiness §3-H1`，Plan B 结构上覆盖不到）、
   `spec_primed` 分叉（§3-H4，第 9 次修复已提交 `8907e9b`，未验证）。
   ⇒ **batched 这条臂今天就是可用的**，只是拿不到图那 −1.5ms。

4. **两堵墙的"已验证先例"都在仓库里**——400 不是"没有路"，是"把已验证的两条路铺满全族"：
   - 墙 1 的先例：`gemm_fp8_sh_exp_pair_kernel<M>` 的 **M 进 grid**（`sh-pair-template-m-design.md`）——
     仓库里唯一被设计论证过的"真共享"机制；且 SH_PAIR M=6 的 parity **已确认为虚警（kernel 无 bug，是测试哨兵问题，`d05bfde`）⇒ 该路径当前是解锁的，只差 e2e A/B**。
   - 墙 2 的先例：**CUTLASS 级 fp8 MMA gemv**（`perf-roadmap.md` 2026-09-09 节）——
     微基准 **5.2×（q_a 1536×4096）/ 8.6×（lm_head 19360×4096）**，与 fp64 参考**逐位一致 maxrel=0**，
     serve 端 **19.20 → 17.44ms（e2e −9.2%）**，再加 fast quant → **15.07ms**。
     **这是本仓库自己量过的、位级一致的 MMA 收益**，不是设计口径。

5. **accept 是门槛，不是并列轴**（`dspark-sglang-real-data §4.2`）：
   τ = 2.214 时，**即使步时压到 sglang 的实测 verify 下界 7.3ms，也只有 303 tok/s**。
   400 在 accept 1.214 下**物理不可达**。而 GH200 的独立铁证是：
   sglang 一个 **silent shared-expert loader/mapping gap** 就把 accept 从 ~5 打到 **1.3–1.8**
   ——**正好落在 ferrite 的 1.214 量级**。⇒ 这是本项目**期望值最高的一次只读+少量 GPU 的调查**。

6. **"instruction-bound" 的机理本轮仍未终局**（三条候选机制都在账上，§3.1）：
   M1 warp-per-row（M 只加每 warp 链长）、M2 μop/指令（0.7–4.9% 峰值带宽）、
   M3 per-kernel 固定延迟（6232 发 × 3.3µs ≈ 20.6ms = 37ms 的 55%，图化只 −1.5ms）。
   **三条的共同解药是同一条：更少、更大的 kernel + tensor core。**
   所以路径分析对"M1/M2/M3 哪条为真"是**稳健的**——但**预算的量级**取决于哪条为真，
   必须先做 §4-Step 1 的那一次 nsys 校准。

7. **最高 ROI 的下一步不是修 ar5-hang，也不是继续打磨 lazy**，而是：
   **在 batched nograph（m=6）上做一次口径校准 + 零代码 mrows/SH_PAIR A/B**
   （2–3 人日、3–4 次 GPU，潜在 −9~13ms ⇒ 步时 ~19ms ⇒ **counting 316 tok/s**）。
   同时并行开 **accept 审计**（3–8 人日）作为高方差大注。
   理由：这是**唯一同时满足三个条件的动作**——(a) 臂今天就能跑（不需要先解 hang），
   (b) 目标形状是 m≥2（mrows/SH_PAIR/MMA 三者的共同前提），(c) 它同时把 §0-6 的三条机制裁决掉。

---

## 1. 口径钉死（数字地基，全部标来源）

> ⚠️ 本项目的 #1 陷阱是"不同臂 / 不同 m / 不同计时器混在一个乘式里"。下表每行都钉死臂 + m + 计时器。

| # | 量 | 值 | 臂/口径 | 来源 |
|---|---|---|---|---|
| D1 | EAGER `c_row`（m=1 完整 40 层 forward） | **6.15 ms** → 162.6 tok/s | EAGER，m=1 | 任务给定 + `final-400-battle §1.1 F5` |
| D2 | lazy verify 步时（内部口径） | **22.56 ms** | lazy，m=1/行，accept 1.214 | `dspark-correctness-chain` 实测（24.15 图化后 −1.6） |
| D3 | lazy 的 `k_emit` | **2.214**（mean-k 1.214 + bonus） | lazy，出师表/对话 | `dspark-sglang-real-data §2.2` |
| D4 | lazy 的 `c_row` | **8.18 ms**（= 18.11 / 2.214） | 反推 | `lazy-verify-95ms-gap-analysis` |
| D5 | batched verify（m=5） | **37.31 ms** | batched，m=5，**Wave 1 前** | `verify-ms-breakdown` nsys（唯一纯 GPU 口径） |
| D6 | batched verify（m=5）达成的带宽 | **381 GB/s = 峰值 5.0%**（14.09 GB / 37.31 ms） | batched | `final-400-battle §3.1` |
| D7 | 折叠后的真字节 | **5.66 GB**（@7TB/s = 0.81ms = 37ms 的 2%） | batched，dedup 后 | `final-400-battle §3.1` |
| D8 | 单发 kernel 的固定执行时间 | **~3.3 µs**（6232 发 ≈ 20.6ms = 37ms 的 55%） | batched | `final-400-battle §3.2` |
| D9 | 图化对 verify 的收益 | **−1.5 ms**（37.31 → 35.5，预期 −24） | batched | `verify-ms-breakdown §修正` 同会话 A/B |
| D10 | mrows 四件套（SH_EXP+GRAPH+ROPE+P3A） | **−1.21 ms**（预期 −24） | batched | `verify-ms-breakdown §修正` |
| D11 | SH_PAIR 的 target saving | **−4.9 ~ −7.9 ms**（10.4 → 3~5.4） | batched m≥2 | `sh-pair-template-m-design §1.1` 的 launch 账 |
| D12 | B300 HBM 峰值 | **7.6–7.7 TB/s** | — | `perf-roadmap`（"7.6TB/s 下仅 0.27ms"） |
| D13 | sglang（同模型同硬件 B300 TP8 B=1） | verify **7.3ms 实测**；step **13.03ms**（推导）；accept **~5**；**383.7 tok/s** | sglang | `dspark-sglang-real-data §1.1`（一手博客） |
| D14 | sglang 的 kernel 数 / MFU / per-row | **~15–20 kernel/层**（ferrite ~156 = 10×）/ tensor-core MFU **30–50%**（ferrite 0.7–4.9% = 10×）/ **1.8ms/行**（ferrite 7.5ms = 4×） | 两边同口径 | `dspark-correctness-chain:1473` |
| D15 | ferrite 的 accept（任务依赖） | counting **~4.8–5.0** / 出师表 **1.214** / 对话 **0.96** | lazy | `dspark-correctness-chain` |
| D16 | 400 的乘积约束 | `τ/step ≥ 0.4 tok/ms`；accept 5 ⇒ step ≤ **15ms**；accept 3 ⇒ ≤ **10ms**；accept 1.214 ⇒ ≤ **5.54ms** | — | `dspark-sglang-real-data §4.1` |

**关于"当前基线 78.8"的口径警告（必须写在账上）**：
`78.8 = 144 tokens / 1827 ms 端到端`（含 prefill ~1000ms）。**它不是纯生成速率**——
同一次 run 的生成段是 `~827ms / 25 步 ≈ 33ms/步`（≈175 tok/s 生成口径，`dspark-correctness-chain:2744`），
而内部口径的 lazy 步时是 22.56ms（98 tok/s 生成）。三个数在三把尺子上。
⇒ **本文一律用"生成速率"口径**：`tok/s = k_emit / step`。**"85→400"在这种口径下应是 98→400 = 4.1×**，
不是 4.7×。这个差不是小事——它等于把 prefill 的 ~1s 摊进了分母。

---

## 2. lazy 的真实上限（为什么 145 是结构性的，不是调参不够）

### 2.1 单调的代数

```
tok/s = k_emit / (k_emit · c_row + d + c)          d = draft（~1.6–3.6ms/步）, c = commit（~0.2）
```

| 情境 | c_row | k_emit | step | tok/s |
|---|---:|---:|---:|---:|
| S0 lazy 现状（对话/出师表） | 8.18 | 2.214 | **22.56** | **98** |
| S0' lazy 现状（counting） | 8.18 | 6.0 | 52.9 | 113 |
| S2 lazy + 零代码（c_row→6.15） | **6.15** | 2.214 | **17.4** | **127** |
| **S2'' lazy 天花板（accept 5）** | **6.15** | **6.0** | **40.7** | **145** |
| **lazy 的渐近上限（k_emit→∞）** | **6.15** | ∞ | — | **162.6 = EAGER 本身** |

⇒ **lazy 是把 batched 的 m 行拆成 m 次 m=1 forward 的臂。它在数学上不可能超过 EAGER。**
145 只是"k_emit=6 时离渐近线还差 10%"的取值。

### 2.2 85 → 145 的具体路径（c_row 8.18 → 6.15 + accept↑）

| # | 项 | 改动 | 预期 | 成本 | 风险 / 前置 |
|---|---|---|---:|---|---|
| **L1** | K1+K2（R2 同程序替代） | 已实现（`ATTN_MROWS2` / `ATTN_MROWS_ROPE_NORM`） | c_row −5~6% | 0（验证中） | 双任务矩阵（计数数字顺序 + 出师表零拉丁） |
| **L2** | `DSV41_LAZY_SDR=1` | **0 代码**（已实现，默认 OFF） | −0.7 ms/步 | 0.5 人日 A/B | 低；`rows_run == k_emit` 不变 |
| **L3** | hc A1/A2（`HC_VERIFY_FUSE` + `HC_FRONT_ROWS`） | **0 代码**（truncate 坑已修 `6f5de2b`；A2 的 `bf16_truncate` 需先改 false） | −2.9~3.8 ms/步 | 0.5~1 人日 | 中；**一次只动一个 gate + 读 kernel 名** |
| **L4** | AR 等待压缩（A2 让 lazy per-row 走图 / A3 SDR / A4 单块 poll / B1 `rows_add`） | A4/B1 是新代码 | −1.5~3.0 ms/步（**上界**，实测未知） | 2~4 人日 + 1 GPU | 中高；**必须先做 wait 探针（0.5 人日）量化"真等待 vs 探针伪影"** |
| **L5** | argmax D2H 尾部批量 | 小改 | −0.3 ms/步 | 1 人日 | 低；**早退依赖不可去**，只能批量尾部行 |
| **L6** | indexer front mrows / VERIFY_ROPE/HEAD_MROWS | 0 代码 | **lazy 下 = 0**（m=1 ⇒ 折核≡逐行） | — | 只在 batched 下存在，**不得计入 lazy** |

**lazy 侧的判决**：
- `c_row → 6.15` 是**可达的**（L1+L2+L3 即可，成本 ≤2 人日、全部低风险）。
- 但 **accept 只在 counting 上到 5**；出师表 1.214 / 对话 0.96 ⇒ lazy 的现实峰值 = **127（正文）/ 145（counting）**。
- ⇒ **lazy 不是 400 的路径。只做 L1~L3（零/低成本），L4~L5 只有在"batched 仍被阻塞且 needs 一个止血"时才投。**

---

## 3. 145 → 400：两堵墙 + 三条路径

### 3.1 墙 1 —— batched 的权重共享为什么没生效（实测证据链）

**实测事实**：`verify(m=5) = 37.31ms` vs `EAGER(m=1) = 6.15ms` ⇒ **6.06× 线性**。
如果行间权重共享生效，m=5 应该 ≈ 6.15 + 4×激活边际（≈ +1.6ms）≈ 7.8ms。**差 4.8×。**

**三条候选机制（各自都有实测背书，且共同指向同一解药）**：

| # | 机制 | 证据 | 解药 |
|---|---|---|---|
| **M1** | **warp-per-row**：`gemm_fp8_mrows` 每个 warp 拥一行输出、权重行进 SMEM/寄存器，M 只增加每 warp 的 FMA 链长 | `sh-pair-template-m-design §1.1`（`row = blockIdx.x*32 + warp`）+ `§3.3` 的指令模型 `4 + 4·fold_r` ⇒ fold_r=1 时权重解码只占一半指令 | **权重解码必须摊到更多元素上**（K-split + 更大 tile）或**换 MMA** |
| **M2** | **instruction/μop-bound**：kernel 只跑 **0.7–4.9% 峰值带宽** ⇒ 时间不由字节决定 ⇒ 省字节（mrows）≈ 0 | `SH_EXP_MROWS` **两次零收益**；四件套全开 **−1.21ms vs 预期 −24ms**；v17→v21 四变体全中性 | **换核**（tensor core 每指令处理多元素） |
| **M3** | **per-kernel 固定延迟**：`6232 发 × 3.3µs = 20.6ms`（37ms 的 55%）；**图化只 −1.5ms** ⇒ submit 半已被 async launch 隐藏 | `final-400-battle §3.2` + F2（图化实测） | **少发核 + 每核更大**（族级融合）；这才是图化的**唯一合法价值** |

**⇒ 关键结论（对路径选择稳健）**：M1/M2/M3 谁是主因不影响"做什么"，只影响"能拿多少"：
**三者都要求「更少、更大的 kernel，且每个 kernel 内部用 MMA 处理更多元素」**——
这正好等于 **swallow（m=6）+ mrows/SH_PAIR（少发核）+ tcgen05/MMA（换核）** 三件套。

**⇒ 但预算量级必须实测**：M2 为真 ⇒ mrows 全部作废（省字节无用）；
M3 为真 ⇒ mrows/SH_PAIR 的 launch 账**足额兑现**（−4.5~7.9ms）。
这个分叉值 **±10ms**，是 §4-Step 1 那一次 nsys 的唯一目的。

### 3.2 墙 2 —— SIMT → MMA（唯一的 instruction-bound 根治）

**仓库内已验的 MMA 收益（不是设计口径）**：`perf-roadmap.md` 2026-09-09 节

| shape | SIMT `gemv_fp8_v2` | **MMA `gemv_fp8_mma_b16`** | 加速 |
|---|---|---:|---:|
| q_a 1536×4096 | 0.031 ms | **0.006 ms** | **5.2×** |
| lm_head 19360×4096 | 0.353 ms | **0.041 ms** | **8.6×** |

- 与 fp64 参考**逐位一致 maxrel=0**；serve **19.20 → 17.44ms（−9.2%）**；+fast quant → **15.07ms**。
- 做法：M=16 token × N=8 输出行/warp × warp 级 K-split；3 段 cp.async 流水；`ldmatrix.x4`；权重按原生 e4m3 读（1 字节）。
- **该手段当时只用到 `matmul_dev`（B=16 的通用引擎），dsv41 的单序列链没有吃它。**
  ⇒ **dsv41 的投影族 / head / attn 全部还是 SIMT `gemm_fp8_mrows`。**

**dsv41 侧的 MMA 现状盘点**：

| 族 | 现状核 | MMA 形态 | 状态 |
|---|---|---|---|
| 投影族（wq_a/wkv/wq_b/wo）、head | `gemm_fp8_mrows_kernel<M>`（SIMT） | **无** | ❌ 未换核；CUTLASS 级手段已在树里（另一引擎） |
| routed experts（gate/up） | `tc5::mxf4` / `tc5::e4x`（tensor core 骨架） | 有骨架 | ⚠️ `e4m3 × tcgen05` **互斥**（`chain_dev.rs:687`）；e4x **从未上机**（2 个 `[OPEN]`） |
| routed experts（**down**） | `expert_gemv_fp4_down_reduce_kernel`（SIMT，3.48ms） | **不存在** | ❌ 从零写（L4-4） |
| shared expert | `gemm_fp8_sh_exp_pair_kernel<M>`（SIMT，M 进 grid） | 无 | ❌ 未换核 |

**m=1 的 MMA 陷阱（关键）**：MMA 的 M 维最小 16/64，m=1 时 **15/16 的 tile 是浪费**。
⇒ **lazy（m=1）即使在投影族换 MMA 也拿不到 3.2 的 5–8×**（那 5–8× 的微基准是 M=16）。
**这就是"墙 1 与墙 2 必须同时跨"的严格理由**：MMA 需要 batched 提供 M。

### 3.3 三条路径（+ 一条并行轴）

#### Path B —— batched 优先（修 ar5-hang **或** 直接走 nograph）

| 步骤 | 改动 | 预期 | 成本 | 风险 |
|---|---|---|---:|---|
| **B0** | **batched + SWALLOW nograph**（今天就能跑） | 基线（serve ~40ms，**待 nsys 定标**） | 0 | 低（已验 0 hang + 零拉丁） |
| **B1** | m=6 mrows 族逐个 A/B（`GATE_MROWS` → `INDEXER_MROWS` → `VERIFY_ROPE_MROWS` → `VERIFY_HEAD_MROWS` 最后单独） | **−4.5 ~ −5.8ms**（设计口径，兑现率未知） | **0 代码**，3~4 GPU | 中（`SH_EXP_MROWS` 零收益是先例） |
| **B2** | SH_PAIR `template<M=6>` e2e A/B（**parity 已确认虚警**） | **−4.9 ~ −7.9ms** | 0 代码 + 1~2 GPU | 中（需看 kernel 名确认 M=6 变体真的跑） |
| **B3** | tcgen05（gate/up，**先澄清 e4m3 互斥**） | −1.0 ~ −3.8ms | 4~5 人日 + 3~4 GPU | 高（e4x 未上机；e4m3 冲突） |
| **B4** | 图化解锁（ar5-hang 系统性修复，§3.4） | −1.5ms（图本身）+ **AR 等待上界 −3.9ms** | 5~10 人日 + 6~12 GPU | **高（0/8 历史）** |

**B 的落点（60% 兑现率，counting 口径）**：step ≈ **19ms → 316 tok/s**（`swallow-unlocked-next-plan §6` 的自校准）。
足额兑现才到 **~15ms → 400**。**B 是唯一能在 3–5 人日内拿到 −9~13ms 的路径。**

#### Path C —— "M 进 grid"全族化 + L4/L5（kernel 级重写）

- **内容**（`l4-l5-kernel-path.md`）：L4（占用/MLP，16.5~21.5 人日）+ L5（流水/满 wave，9~13 人日）= **25.5~34.5 人日**，预期 **−7.5 ~ −16.4ms**。
- **核心串行主干**：`tcgen05 gate/up → L4-3 K-split → L4-4 down 新核 → L5-3 e4x kRing`（11~15 人日）。
- **诚实校准**：**L4/L5 的全部 ms 预期在仓库里零实测背书**；反向证据三条（v17→v21 四变体全中性、`SH_EXP_MROWS` 两次零收益、四件套 −1.21 vs 预期 −24）。
- **判词**：**C 是"跨过 400 之后"的层**（把中等 accept 也拉进 400），**不是通往 400 的最快路**。
- 但 C 的 L4-3/L4-4（tcgen05 K-split + down 换核）**是墙 2 在 MoE 上的具体落点**，与 Path B 的 B3 是同一件事——**应合并投资，不重复记账**。

#### Path D —— library kernel 替换（cuBLAS / CUTLASS / DeepGEMM）

- **现状**：仓库已绑 cuBLAS（`gemm_f32`/`gemm_bf16`），`cublas_m1()` 存在但**默认 OFF**（"只有一个 handle、绑定主流派、不能发侧流"）。
- **适配性问题**：
  - **routed MoE 是 per-token 散射**（30 个 assignment → 29 个唯一专家，去重仅 3%）⇒ cuBLAS 的 dense GEMM 不适合，需要 **grouped GEMM**（CUTLASS grouped / DeepGEMM 型）。
  - **ABI 约束**：per-slot 间接寻址 + CUDA-graph 安全（`ids` 作为设备数组入参）——现有 tcgen05 骨架已满足（18-param ABI），library 版本要重新满足。
- **正确性风险（本项目特有）**：**本仓的红线是"逐位等价"**（NCCL 升序、FMA 收缩被功率二量化放大到 1 ULP 就能翻 token）。
  cuBLAS/CUTLASS 的累加顺序与 ferrite 的契约不同 ⇒ **任何 library kernel 都要重做逐位 parity，或接受"近边界翻转"**。
  这正是 `perf-roadmap` 里 bf16 tensor-core gemv 5 次迭代失败、以及 R2 损坏的同一类坑。
- **成本/风险**：投影族/head 10~15 人日（集成 + parity）；MoE grouped 15~20 人日。**风险中高、收益与 Path C 重叠**。
- **判词**：**D 是 C 的一种"少写 kernel"的替代实现，不构成独立路径。** 唯一值得单列的是
  **"复用 dsv41 之外那套已验证的 CUTLASS 级 fp8 MMA gemv"**——它已在本仓量到 5.2–8.6× 且位级一致，**迁移成本可能低于新写**。

#### Path E（并行轴）—— accept 审计

- **门槛性质**：τ=2.214 ⇒ **步时无限好也只有 303 tok/s**（§0-5）。
- **最高价值的线索**：GH200 上 sglang 的 **silent shared-expert loader/mapping gap** 把 accept 从 ~5 打到 1.3–1.8（`dspark-sglang-real-data §4.4-4`）。
  **ferrite 的 1.214 落在同一区间**。且 ferrite 自己的 `draft-numerical-audit` 已找到 **5 个严重缺陷**。
- **改动**：以"同构排查"方式审计 draft/loader/shared-expert/tap 路径是否存在 silent mapping 缺口；复查 5 个已知 draft 缺陷的修复状态。
- **成本**：3~8 人日 + 3~5 GPU。**风险：未知（可能只是工作负载/模型行为，不是 bug）**——但**期望值最高的一注**（accept 1.214→3+ 等于 1.4× 吞吐，且是 400 的必要条件）。

### 3.4 ar5-hang 的修复方向（系统性轮次分析，不是第 9 个点修）

**已知的三个臂分歧源（结构同源）**：

| 源 | 会合数表 | 覆盖状态 |
|---|---|---|
| **verify graph 四臂** | Direct(1) / Dry(1) / Replay(1) / **Capture(2)** —— 唯一奇点 | Plan B（`RankVote`）已实现，**从未在 `VERIFY_GRAPH=1` 下上机**；且 `VERIFY_GRAPH=0` 时是 **strict no-op**（`chain_dev.rs:5807` 早退） |
| **draft graph 四臂** | direct(0) / dry(0) / replay(1) / **capture(2)** | **Plan B 结构上覆盖不到**（H1）；per-rank 输入：`graph_failed` latch + `self.unit` |
| **`spec_primed` 分叉** | legacy 臂每步 AR 足迹 ~161 round，swallowed 臂 ~81 ⇒ 差 ~80 | H4；第 9 次修复已提交（`spec_primed_unanimous` + `serve.rs` 毒化），**未验证** |

**正确的修法（一条不变式，不是三个补丁）**：

> **不变式**：任何 **per-rank 决策**，若会改变"本步的 collective 轮数"（`host_barrier` 次数 / AR round 数），
> 必须**先经 `RankVote` 全 unanimity**（任何 rank 不同意 ⇒ **全体**走保守臂），然后才允许执行。

- **把这条不变式同时铺到三处**：`verify_graph_gate`（已做）、`draft_graph_arm`（同构扩展，低成本）、
  `spec_primed`（已做）。**三处统一为一个"collective footprint"投票**。
- **加一个轮数校验（新，且这是真正把"8 次失败"变成"可诊断"的关键）**：
  每步末把每 rank 的 AR round 计数做一次 device-side checksum 交换，不等即打
  `[round-mismatch] step=N rank=R my=.. peer=..`。
  把"500 万次自旋后才发现的 hang"变成**当步即报 + 给出分歧起点**。
  （`plan-b-swallow-readiness §3` 的判定树是事后考古；这一步是把它变成在线断言。）
- **成本**：5~10 人日 + 6~12 GPU。**风险高（历史 0/8）**。
- **收益的正确读法**：**图的收益只有 −1.5ms**（已实测）；真正的价值是 **rank drift → 0 ⇒ AR 的 peer-stamp 等待下降**
  （上界 3.9ms/step，`ar-further-optimization §4-A`）。**但这只是上界，实测未知。**
  ⇒ **先做 0.5 人日的 wait 探针（A1/D3）**，若等待 <1ms 则**不投 B4**，把预算给 B1/B2/E。

### 3.5 到达 400 的乘法表（generation 口径，全部标来源）

| 阶段 | 改动 | step | k_emit | tok/s | 依据强度 |
|---|---|---:|---:|---:|---|
| **S0** | lazy 现状（干净基线） | 22.56 | 2.214 | **98** | 实测 |
| **S1** | + K1+K2 + 干净栈（wo_a/RING_WIN/FORK/MARKOV/SDR） | ~20.6 | 2.214 | ~107 | 账本（78.8→86-88 e2e ≈ ×1.09-1.11） |
| **S2** | + lazy 零代码（hc + SDR + AR 折叠，c_row→6.15） | 17.4 | 2.214 | **127** | 账本 |
| **S2″** | **lazy 天花板（counting，accept 5）** | 40.7 | 6.0 | **145** | 代数（§2.1） |
| **S3** | batched nograph + swallow + mrows 族 + SH_PAIR（**60% 兑现**） | ~19 | 6.0 | **316** | 设计口径 × 历史兑现率 |
| **S4** | + tcgen05（gate/up + down） | ~16.5 | 6.0 | ~364 | 修正口径（**非 −6.8：down 无核**） |
| **S5** | + MMA 全族化 / L5 流水 | ~13 | 6.0 | **~460 ✓** | 设计口径，**仓内零实测** |

**判决**：
1. **400 的临界点落在 S4 → S5 之间（15ms）**，且**S3/S4/S5 三项必须同时足额**——仓史上从未有哪一轮把设计口径全兑现。
2. **步时侧不是唯一门槛**：accept 1.214 × 步时 5.54ms = 400，**低于 L5 地板 8–9ms** ⇒ 出师表/对话在两堵墙全跨完后仍不可达。
   **400 是 counting-口径目标**（`swallow-unlocked-next-plan §6-2`）。
3. **最短可行集** = `swallow(nograph 即可) + mrows 族 + SH_PAIR<M> + tcgen05 + MMA/融合`。

---

## 4. 推荐：最高 ROI 的下一步（含止损门）

> 排序原则：**先做"臂今天就能跑 + 零/低成本 + 结果会改写预算"的**，再做"高方差大注"，最后做"25 人日级的重写"。

### Step 1（1~2 人日，1 GPU）—— batched nograph 的**口径校准 nsys**（全计划性价比最高的一次会话）

- **配置**：`SWALLOW_STEP=1` + `VERIFY_GRAPH=0` + `DRAFT_GRAPH=0`（避开 hang），counting prompt（高 accept = 满 6 行）。
- **读法（`nsys-wave1-analysis-framework §1.2/§1.4`）**：带 `--capture-range` 切出 decode 净窗，**按 kernel 名聚合**，
  `<M>` 模板参数就是运行期行数指纹（`gemm_fp8_mrows_kernel<6>` vs `<1>`）。
- **要拿到的三个数**：
  1. **真实族表**（routed / shared / attn / hc / gate / indexer / AR / head 的 ms + 每核 launch 数）；
  2. **每核的 wave 数 + 达成带宽**（→ 直接裁决 M1 vs M2 vs M3）；
  3. **SH_PAIR / mrows 变体是否真的被 dispatch**（本项目 #1 陷阱："gate 设了但没生效"）。
- **判据/分支**：达成带宽 ≪ 峰值且 wave 数足 ⇒ **M2（μop）** ⇒ mrows 无望，全力 MMA；
  launch 数高、单核 µs 级、wave 数低 ⇒ **M3（固定延迟）** ⇒ mrows/SH_PAIR 的 launch 账足额，优先 B1/B2。
- **不做这一步的后果**：S3~S5 的全部预算 ±10ms 漂移，且会重演"mrows 为什么没兑现"的误判。

### Step 2（2~3 人日，3~4 GPU）—— batched nograph 的**零代码 A/B**（潜在 −9~13ms）

- 顺序：`GATE_MROWS` → `INDEXER_MROWS`（front）→ `VERIFY_ROPE_MROWS` → `VERIFY_HEAD_MROWS`（**最后单独**，它是历史 hang 的那个组合）→ `SH_PAIR_M=6`。
- **每格判据（缺一不可）**：计数 1-200 数字顺序正确 + 出师表零拉丁 + `k_acc` 逐位不变 + **看 kernel 名确认变体真的跑了**。
- **止损门**：任一 gate 的 `verify_ms` 位移 < 预期的 40% ⇒ **停**，转 Step 4/tcgen05；**不要在同一轮叠 gate 找感觉**。

### Step 3（并行，3~8 人日，3~5 GPU）—— **accept 审计**（高方差大注）

- 按 GH200 的 loader-gap 同构案例，审计 `draft_forward` / `draft_moe` / shared-expert remap / tap / markov 的
  **silent mapping 缺口**；复查 `draft-numerical-audit` 的 5 个缺陷。
- **判据**：counting ≠ 出师表/对话的 accept 差是否可被"某个 map 没生效"解释。
- **期望**：accept 1.214 → 2~3 即等于 **1.4~2.4× 吞吐**，且是 400 的必要条件（§0-5）。

### Step 4（有条件，5~10 人日，6~12 GPU）—— ar5-hang 系统性修复

- **前置门**：Step 1 必须证明"AR 等待 ≥2ms/step"（否则花的 5~10 人日只换 −1.5ms 的图）。
- 内容：§3.4 的**一条不变式（铺三处）+ 轮数校验**。**不要做第 9 个点修。**

### Step 5（15~25 人日）—— 墙 2 的正面攻坚：MMA 全族化

- 优先复用**本仓已验证的 CUTLASS 级 fp8 MMA gemv**（5.2–8.6×、位级一致、e2e −9.2%）迁移到 dsv41 的投影族 + head；
- MoE：`tc5::mxf4` K-split（L4-3）+ **down 新核**（L4-4）；**先澄清 `e4m3 × tcgen05` 互斥**（R1）。
- **单层微基准门（不达标即止损）**：gateup 22.2µs / down 17.2µs。

### 止损门汇总

| 风险 | 触发信号 | 止损动作 |
|---|---|---|
| kfill/mrows 假兑现 | 任一 mrows gate 位移 < 预期 40% | 停 mrows 线，转 MMA |
| SH_PAIR M=6 仍 parity 失败 | 逐字节 diff 不一致 | 冻结 `SH_PAIR_M` 默认 OFF，−5~8ms 从阶梯划掉 |
| ar5-hang 再失败 | 任一格 hang / `[round-mismatch]` 持续 | 保留 **nograph** 作生产形态（已可用），不进 Step 4 |
| accept 审计无果 | 找不到 mapping 缺口 + 5 个缺陷均已修 | accept 视为**模型/工作负载上限**，400 只在 counting 口径追求 |
| MMA 重写中性 | L4-3/L4-4 单项 < −1ms | 停止变体矩阵（勿重演 v17→v21），改走 library 迁移 |

---

## 5. 诚实校准（必须写在账上）

1. **本文的一切都建立在"内部口径"上**（22.56ms / 6.15ms / 37.31ms）。serve 墙钟（25–65ms、78.8 e2e）
   **不可作为步时基线**（用户已两次判为不可信；含 curl/SSE/admission/tail）。
2. **"instruction-bound" 的三条机制（M1/M2/M3）尚未终局**。它们的**解药相同**，但**预算量级差 ±10ms**。
   这是 Step 1 存在的唯一理由。
3. **SH_PAIR parity 的"虚警"结论来自 `d05bfde`（测试哨兵 0x5A 可产出 + 双重计数）**——
   它证明的是"kernel 没有那个 bug"，**不等于"SH_PAIR 的 −4.9~7.9ms 会兑现"**（后者仍是设计口径）。
4. **CUTLASS 级 MMA gemv 的 5.2–8.6× 是在 `matmul_dev`（B=16）上量的，不是 dsv41 单序列链。**
   迁移到 dsv41 的收益**未测**；且 m=1（lazy）下 tile 浪费会吃掉大部分收益。
5. **L4/L5 的 25.5~34.5 人日与 −7.5~−16.4ms 全是设计口径，仓内零实测背书。**
   按历史 60% 兑现率，S3~S5 的现实落点是 **~18–20ms ⇒ 300–340 tok/s**，不是 400。
6. **本机无 GPU**：除标"实测"的行外，一切 ms 都是账本推算或设计口径。
7. **"当前基线 78.8"是 end-to-end（含 prefill）口径**；本文一律用生成速率口径（S0 = 98）。
   任务书的"85→400"若按 e2e 读是 4.7×，按生成读是 4.1×——**这个差别必须显式承认**，否则预算会偏 ~20%。

---

## 附：一句话总结

> **lazy 的天花板是 EAGER 自己（162.6 tok/s，accept 5 时 145）；从 85 到 400 必须同时跨"batched（m≥2 行间共享）"
> 与"SIMT→MMA（去掉 instruction-bound）"两堵墙。两堵墙的先例都在本仓、都已验证
> （SH_PAIR 的 M 进 grid；CUTLASS 级 fp8 MMA gemv 的 5.2–8.6×、位级一致）。batched 不是不可用，
> 只是"batched+graph"被 ar5-hang 阻塞——nograph 已 0 hang + 零拉丁。**
> **最高 ROI 的下一步：先在 batched nograph（m=6）上做一次口径校准 nsys + 零代码 mrows/SH_PAIR A/B
> （2–3 人日、3–4 次 GPU，潜在 −9~13ms ⇒ 316 tok/s），并行开 accept 审计这一高方差大注；
> ar5-hang 的系统性修复只在探针证明"AR 等待 ≥2ms/step"后才投。400 是 counting-口径目标。**
