# SWALLOW 解锁后的优化路径 —— 基线测量 → SH_PAIR M=6 → mrows → 400

> 工部 · 2026-09-12 · **只读勘察 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 输入（现场核对）：`swallow-unlocked-400-final-path.md` · `swallow-unlocked-t1t4-test-design.md` ·
> `swallow-unlock-400-sprint.md` · `batched-400-v2-remaining-roi.md` · `400-fastest-path-roadmap.md` ·
> `swallow-contingency-plan.md` · `dspark-correctness-chain.md`（尾部）· `nsys-clean-stack-91-design.md`。
> 读码：`chain_dev.rs` / `tp.rs` / `dsv41_kernels.cu` / `devrt.rs` / `device.rs` /
> `scripts/{batched_400_v2.sh,sh_pair_ab.sh}`。代码基线 HEAD `d9261bb`。
> **口径纪律**：每条 ms 标来源（**实测** / **launch 账** / **设计** / **代数**）。

---

## 0. 判决（先读八条）

1. **❗ `56.6 tok/s` 不是步时退化，是 accept 口径。** `tok/s = k_emit / 步时`。
   56.6 @ `k_emit≈2.2`（accept≈1.2，出师表口径）⇒ **步时（含观测）≈38.9ms；去观测 ≈31ms = 设计 S0**。
   **步时没有退化。** 要复现 S0 的 `~194 tok/s`，必须在**计数 prompt（accept 5）**上测：
   同一 31ms 步时给 `6/0.031 = 194`。任务里「56.6 远低于 200」是**口径混用**（出师表的 tok/step 配了计数的预期）。
   【代数，见 §1】

2. **S0 已被本次运行间接证实 ≈31ms**（不是从 56.6 派生的，两者是不同口径）。区间 29–34ms。
   ⇒ **Step 1 的实测是把 31ms 钉到 ±1ms**——因为 ±3ms 恰好是 400 ✓/✗ 的分界（§5）。
   【代数 + 设计锚点】

3. **观测开销的机理是「同步点」，不是「4 字节」。** `download_u32`（`devrt.rs:1302`）是**同步 pageable
   `cudaMemcpy` D2H** ⇒ 每次调用 = **一次全流同步**（GPU 排空 + CPU 追赶），代价远超 4B 传输本身。
   而 **V5_LEDGER 每次 probe 读 `4 个 canary + 1 epoch = 5 个 D2H`**（`chain_dev.rs:9666-9695`），
   `pre + note` 两次 = **10 个同步点/步**——**canary 从 1→4 槽的 OOB 修复把 ledger 的 D2H 翻了 4 倍**。
   ⇒ 吞吐轮必须 `V5_LEDGER=0`；且**值得单测「能否退掉 `DYNAMIC_PAD`」**（再多 1 D2H + 1 RankMax/步）。
   【读码，`tp.rs:333` `V5_LEDGER_CANARY_OFFS: [usize; 4]`】

4. **`SH_PAIR M=6` 的 gate 只需 `DSV41_SH_PAIR_M=1`**（+ `.so` 符号 `dsv41_gemm_fp8_sh_exp_fused`）。
   它是 `shared_expert_mrows` 里**第一个** dispatch 的分支（`chain_dev.rs:13526`），**不**需要
   `DSV41_SH_EXP_FUSED` / `DSV41_SH_PAIR`——那两个是**下面**旧 M=1 臂的门（`:13589`）。
   ⚠️ **`t1t4` 文档的 P6 把 M-row 臂与旧 M=1 臂混为一谈**（声称要 `SH_EXP_FUSED+SH_PAIR`）——**以代码为准**，
   否则会设两个无用门、并误判「幻影门」。同名陷阱同 `swallow-unlock-400-sprint §B1 的 gate 写法`（正确）。
   【读码】

5. **`SH_PAIR` 的 `−4.9~7.9ms` 是 launch 账，项目自己的实测先例与之冲突。**
   同会话实测：图化只 `−1.5ms`、`{SH_EXP_MROWS+GRAPH+ROPE+P3A}` 全开只 `−1.21ms`
   （`batched-400-v2-remaining-roi §5.1`）——**「减 launch 数」在本仓几乎不买账**（CPU submit 已被
   async launch 与 GPU 重叠隐藏）。**但 M=6 的特殊性在于它同时把 phase-1 grid 从 9 块（M=1）提到 54 块**
   （`ceil(n1/32)·M`，`dsv41_kernels.cu:7984`）——这是**并行度**增益，不是 launch 增益 ⇒ 有可能真赚。
   ⇒ **必须 A/B 定谳，不得按 −7.9 编 400 的预算。** 【launch 账 vs 实测先例】

6. **400 的代数（不可协商）**：`400 tok/s ⇒ 步时 ≤ 2.5·k_emit ms`。
   `accept 5 (k_emit=6) ⇒ ≤15.0ms`；`accept 3 (k_emit=4) ⇒ ≤10.0ms`；
   `accept 1.214（出师表）⇒ ≤5.54ms`（**物理不可达**，低于 L5 floor 8–9ms）。
   设计全兑现 `S5 = 14.5ms = 414 tok/s ⇒ 400 ✓`；但从**实测 S0=31** 起，设计增量 Σ=16.5 ⇒ 14.5。
   **S0 的 ±3ms 决定 400 ✓/✗。** 【代数】

7. **ROI 排序（按收益·概率 ÷ 成本·风险，不按任务给的收益序）**：
   ```
   R1 SH_PAIR M=6（最大单项、零代码、arm 已编译）
     R2 hc 链（最便宜的真收益：≤1 人日，一行 truncate 修复）
       R3 mrows 族（零代码，但兑现存疑——SH_EXP_MROWS 两测零收益）
         R4 B6（需实现，小）
           R5 tcgen05（go/no-go，当前 blocked）
             R6 L4（条件触发、16~21 人日、仓内零实测背书）
   ```
   详见 §4。

8. **先做的不是优化，是「钉死 S0 + 量清观测税」。** 因为①S0 的 ±3ms 是 400 分界；②`DYNAMIC_PAD` 的
   1 D2H + 1 RankMax 可能是白丢的（P0.5-a）；③若不分离观测，后面每一格 A/B 的 `verify_ms` 位移都会被
   观测噪声吃掉（本轮 `56.6` 正是被观测污染的样本）。【工程纪律】

---

## 1. 口径校正：`56.6` 到底是什么（任务 Q1）

### 1.1 代数（三次反推同一个数）

```
观测值        tok/s = completion_tokens / e2e = 300 / 5.30 = 56.6        （实测）
恒等式        tok/s = k_emit / step_ms × 1000
ledger 样本   k_emit ∈ {1, 2}  ⇒ accept ≈ 1（出师表/对话口径，不是计数的 5.0）
反推步时      step_with_obs = k_emit_avg / 56.6 × 1000
              k_emit_avg = 2.2  ⇒ 38.9ms   （2.2 = 出师表 accept 1.214 + 1）
去观测        38.9 − (5~10) = 28.9~33.9ms
任务给的估值  ~65~75 tok/s ⇒ step = 2.2/70 = 31.4ms   ← 与设计 S0 (31ms) 吻合
```
⇒ **三者自洽于 `step_without_obs ≈ 31ms`。** 也就是说：

### 1.2 为什么 `6 / 0.031 = 194` 与 `56.6` 不矛盾

**SWALLOW 的 verify 块恒 m=6 行**（`VERIFY_ROWS=6`，`chain_dev.rs:84`），**与 accept 无关**
（`400-fastest-path-roadmap §1.3`：batched 基线「与 accept 无关」）。⇒ **步时与 accept 无关。**

| 口径 | accept | k_emit | 步时 | tok/s | 对照 |
|---|---:|---:|---:|---:|---|
| 出师表（本次运行） | 1.214 | 2.214 | 31ms | **71** | ≈ task 的 65–75 ✓ |
| 计数 | 5.0 | 6.0 | 31ms | **194** | = 设计 S0 ✓ |

> **一句话**：`56.6` 与 `200` 是**同一台机器、同一 31ms 步时、两个不同的 accept 口径**。
> 「56.6 远低于 S0 的 ~200」**不是性能问题**——本次跑的是低 accept prompt。
> **要在计数 prompt 上重跑，才知道 S0 的真票面。**

### 1.3 观测开销的实数化（为什么是 5–10ms 而不是 10µs）

读码结果（**这是本文件最有行动价值的一条**）：

| 观测 | 每次 probe 的 D2H | 每步 probe | **同步点/步** | 证据 |
|---|---:|---:|---:|---|
| `V5_LEDGER` | **5**（4 canary + 1 epoch） | 2（pre + note） | **10** | `chain_dev.rs:9666`（canary 循环 ×4）、`:9689`（epoch）、`:9712/:9721`（pre/note） |
| `SWALLOW_DYNAMIC_PAD` | 1（epoch） | 1 | **1**（+1 RankMax 会合） | `chain_dev.rs:9852-9855` |
| `pos_ctr` / `inv_ids` 等 | 1 | 1–2 | 1–2 | `:9888` |

`download_u32`（`devrt.rs:1302`）：
```rust
let mut b = [0u8; 4];
unsafe { (self.cudart.memcpy)(b.as_mut_ptr() as *mut c_void, ptr, 4, CUDA_MEMCPY_D2H) };
```
**同步、pageable 栈缓冲的 `cudaMemcpy`** ⇒ 每次调用**排空整条流**（GPU 必须把所有在跑的核做完，
host 才能取回那 4 字节）。**代价不在传输（4B），而在「失去 CPU/GPU 重叠」**：默认路径下 host 领先 GPU
把下一步 ~6000 发 launch 提前排好，同步点一插，这一步的重叠全丢。

> ⇒ **10 个同步点/步解释 5–10ms 完全成立。**
> ⇒ **`canary 1→4 槽`（OOB 修复）把 ledger 的 D2H 从 1/probe 翻到 5/probe**——任何「ledger 只贵一点点」
> 的旧口径**已过时**。吞吐轮 `V5_LEDGER=0` 是硬纪律（`batched_400_v2.sh:157-167` 已经这么写）。

---

## 2. Step 1 —— 基线测量（**先于一切**）

> 目的：把 S0 钉到 ±1ms，并**分离**两个观测税（V5_LEDGER / DYNAMIC_PAD）。
> 姿态：一臂一进程（`OnceLock` 每进程读一次）；**一 prompt 一 serve**（`[dspark] steps=` 累加器跨请求不清零，
> `sh_pair_ab.sh` 口径 2）；交错 A B C A B C 抵消热漂。

### 2.1 三臂矩阵（同一 binary + 同一 `.so`）

| 臂 | `V5_LEDGER` | `DYNAMIC_PAD` | 读什么 | 回答 |
|---|---|---|---|---|
| **A**（复现） | 1 | 1 | `[v5-ledger]` 行 + 步时 | 复现 `56.6` / `79ms?` 观测污染 |
| **B**（真基线） | **0** | 1 | 步时（唯一可信） | **S0 = 无观测步时** |
| **C**（Pad 税务） | **0** | **0** | 步时 + `ar5-hang` 计数 | **DYNAMIC_PAD 能否退**（`P0.5-a`） |

gate 串（逐字，基于 `batched_400_v2.sh:145-156`，**已按 §0-4 修正**）：
```bash
COMMON="DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 \
        DSV41_EXPERT_ACT_E4M3=1 DSV41_BF16_TRUNCATE=1 \
        DSV41_SH_EXP_MROWS=1 DSV41_MROWS_SMALL_N_ADAPTIVE=1 \
        DSV41_GATE_MROWS=1 DSV41_VERIFY_HEAD_MROWS=1 \
        DSV41_INDEXER_MROWS=1 DSV41_NORM_MROWS=1 DSV41_COMPRESSOR_MROWS=1 \
        DSV41_DRAFT_GRAPH=1 DSV41_DRAFT_P3A=1 DSV41_VERIFY_GRAPH=1 \
        DSV41_SWALLOW_STEP=1 \
        DSV41_TIMING=1 DSV41_DSPARK_DEBUG=1"
A="$COMMON DSV41_V5_LEDGER=1 DSV41_SWALLOW_DYNAMIC_PAD=1"
B="$COMMON                   DSV41_SWALLOW_DYNAMIC_PAD=1"
C="$COMMON"
# 禁止：DSV41_LAZY_VERIFY / DSV41_HC_VERIFY_FUSE / DSV41_HC_FRONT_ROWS（batched_400_v2.sh:175）
# 幻影：DSV41_OOB_GUARD（树中无此 env）/ DSV41_SWALLOW_EPOCH_PAD（与 DYNAMIC_PAD 叠加 = over-pad）
```

> ⚠️ **`batched_400_v2.sh:154` 现在写的是 `SWALLOW_EPOCH_PAD=1`**（常数 pad），而脚本注释（`:163-164`）
> 说 pad 是「the fix」。**这与 `t1t4` 文档 P4 的结论（常数 pad + 动态 pad = over-pad）冲突**。
> ⇒ Step 1 三臂**逐字手写**，不直接跑脚本；脚本正式化（`EPOCH_PAD → DYNAMIC_PAD`）是**独立交付项**。

### 2.2 两 prompt（口径分离）

| prompt | accept | 用途 |
|---|---|---|
| 计数 `请从 1 数到 200…` | ~5.0 | **400 的分母**（k_emit=6） |
| 出师表 | ~1.21 | 复现 `56.6`、零拉丁红线 |

`MAXTOK=1000`（计数到 200 需要 ~40 步 @ k_emit 6，够看稳态）。

### 2.3 判据（合取）

| # | 判据 | 通过线 | 失败指向 |
|---|---|---|---|
| S0-1 | 稳态步时 `steady_median`（丢弃前 10 步） | **落 29–34ms** | >34 ⇒ 真退化，先归因再优化；<27 ⇒ 质疑测量 |
| S0-2 | `k_emit` 直方图（计数 prompt） | **mode = 6**（accept 5） | <6 ⇒ accept 支线未解（`R10`），400 分母作废 |
| S0-3 | 观测税 = A − B | **记录绝对 ms**（预期 5–10ms） | 若 >12ms ⇒ 查是否 `pre`+`note` 双 probe 都开 |
| S0-4 | pad 税 = B − C | **记录**；C 臂 `ar5-hang == 0` | C 有 hang ⇒ pad 不能退，回 B |
| S0-5 | `[verify_graph] captured verify_graph_m6` | **必须出现** | 缺 ⇒ 6 行块退直发，SWALLOW 缩水（≤−4.5ms） |
| S0-6 | 红线 | 计数数字顺序 + 前 61 行 + `k_acc` 逐位 + 零额外拉丁 | 见 `swallow-contingency-plan §4.4` |
| S0-7 | 稳定性 | **3× 独立 run 全 0 `ar5-hang`**，其中 1 次长跑跨历史 hang 步数 | 任一 hang ⇒ 假解锁，回 nograph |

**通过线**：`S0-1 ∧ S0-2 ∧ S0-5 ∧ S0-6 ∧ S0-7`。任一缺证据 ⇒ **不得下结论**（缺证据 exit 2）。

### 2.4 为什么这一步不能跳

- 400 的代数对 S0 的**±3ms 敏感**（§5）；
- `S0-3/S0-4` 是两个「白拿」的机会（如果 `DYNAMIC_PAD` 能退，等于白赚一条）；
- 若不分离观测，后面每一格（SH_PAIR/mrows/…）的 `verify_ms` 位移都会被 5–10ms 的观测抖动吃掉。

---

## 3. Step 2 —— SH_PAIR M=6 验证设计（任务 Q2）

### 3.1 机制（读码，非推断）

| 项 | 落点 | 证据 |
|---|---|---|
| gate | `DSV41_SH_PAIR_M=1`（**唯一必需**） | `chain_dev.rs:1468` `sh_pair_m()`；`:13526` dispatch FIRST |
| fold knob | `DSV41_SH_PAIR_M_FOLD ∈ 1..=6`（runtime 参数，不重编） | `chain_dev.rs:1483` `sh_pair_m_fold()` |
| 符号前置 | `supports_sh_exp_fused()` → `dsv41_gemm_fp8_sh_exp_fused` | `:13527`；launcher `dsv41_kernels.cu:7944` |
| kernel | `gemm_fp8_sh_exp_pair_kernel<M>` | `dsv41_kernels.cu:7617` |
| 形状门 | `m ≤ 6`、`sh_il%32==0`、`dim%32==0`、`m ≤ 8` | `:13528-13532`（镜像 kernel 特化） |
| launch 形态 | 三段一核：`quant_rows` + **ONE** `sh_exp_fused<6>` | `:13536-13575` |

### 3.2 三证上场（缺一 = 幻影门，报 ABORT 不报失败）

```bash
# 1) .so 真带符号（本机 kernels/cuda/*.so 不存在，必须在远端取）
ssh $NODE 'nm -D ~/ferrite/kernels/cuda/libferrite_kernels.so | grep -c dsv41_gemm_fp8_sh_exp_fused'  # ≥1
# 2) env 实读（防「设了没生效」——本仓 #1 陷阱）
ssh $NODE "tr '\0' '\n' < /proc/\$(pgrep -x ferrite-serve|head -1)/environ | grep -E 'SH_PAIR' | sort"
# 3) 上场证据（nsys kernel 名，唯一能证 template 实例真跑）
#    nsys 出现 gemm_fp8_sh_exp_pair_kernel<6>（不是 <1>，也不是旧臂 gemm_fp8_sh_pair_kernel）
```

> ⚠️ **M 特化各自 `cudaFuncSetAttribute`**（`dsv41_kernels.cu:7994-8013` 宏已覆盖 1..=8）——历史上漏设
> `<m>` 导致 `cudaErrorInvalidValue`。**launcher 返回 2 = decline**（形态不满足/无 resident config），
> 此时 Rust 侧**无声**回落逐行链（`shared_expert_mrows` 的 `return Ok(false)` 语义）——**只能靠 nsys 取证**。

### 3.3 A/B 矩阵（复用 `scripts/sh_pair_ab.sh` 骨架，四臂一 prompt）

| arm | `SH_PAIR_M` | `SH_PAIR_M_FOLD` | 读作 |
|---|---|---|---|
| base | OFF | — | 逐行链（参照） |
| m1f1 | 1 | 1（设计 §3.3 winner） | `template<6>`，phase-1 grid=54 块 |
| m1f2 | 1 | 2 | phase-1 折 2 行/块 |
| m1f6 | 1 | 6 | 纯 M-fold（SM 覆盖最差，grid=9） |

- **交错 A B A B**（base↔m1f1）抵消热漂；**每臂一进程**（`OnceLock`）。
- **判据四腿**：① `steady_median` 相对 base **≥ −2ms**（保守门：设计大头 −4.9~7.9 是 launch 账，
  §0-5 说不一定兑现）；② 红线（前 100 字零拉丁 + `先帝创业未半` + 无双字）；③ `k_acc` 序列逐位不变
  （bit-identical by construction，kernel C1–C8 契约）；④ 三证上场。
- **止损**：位移 < −0.8ms（< 设计 40%）⇒ 与 `SH_EXP_MROWS` 两次零收益同判——**instruction-bound**，
  立刻转 mrows 族，不投 L4-2 variant 矩阵。

> **对任务表述的修正**：任务把 SH_PAIR M=6 的收益记作 `−4.9~7.9ms`（**launch 账**）。
> 横向校准：nsys 实测 SH_PAIR 族 **~8–9%**（`nsys-clean-stack-91 §行 239`），在 31ms 步上 = **2.5–2.8ms**。
> ⇒ **现实预期区间 = `−0.8 ~ −2.8ms`（保守/中性），`−4.9~7.9` 是上限不是中位。** 争点是 M=6 的
> **并行度增益**（9→54 块）能否把它从「launch 账」变成「真 GPU 时间」——这正是 A/B 要回答的。

---

## 4. ROI 排序（任务 Q3）

> ROI := 兑现 ms × 兑现概率 ÷ (人日 × 风险)。「兑现概率」取本仓实测先例的折算，不取设计口径。

| 排名 | 优化 | 预期 ms（设计 / **校准**） | 成本 | 风险 | 兑现概率依据 | 前置 |
|---:|---|---|---:|---|---|---|
| **R1** | **SH_PAIR M=6** | −4.9~7.9 / **−0.8~2.8** | **0 代码**（gate + `.so`） | 中（符号/parity/形态 decline） | M=6 是唯一把 M 进 grid ⇒ 并行度真增益；但 launch 账先例偏负 | 无（`t1t4` P6 的双门说法是错的，见 §0-4） |
| **R2** | **hc 链**（A1+A2） | −1.3~1.7 / **−1.3~1.7** | **≤1 人日**（一行 truncate） | 中（历史破零拉丁） | 折核件全在树；A1-a 已修，A2 同构待修 | 先把 `hc_mixes_auto` verify 调用点（`chain_dev.rs:11625`）的 `bf16_truncate()` 改 `false` |
| **R3** | **mrows 族**（5 gate 逐个） | −4.5~5.8 / **−0~2** | **0 代码** | 低（逐位等价已论证） | `SH_EXP_MROWS` 两次零收益 + `{四件套}=−1.21ms` ⇒ **instruction-bound 先例** | 一 gate 一轮，`HEAD_MROWS` 最后单独上（历史 ar5-hang） |
| **R4** | **B6**（`dsv41_gemm_fp8_mrows_f32`） | −0.66~1.5 / **−0.66~1.5** | **0.5 人日 + 实现** | 中（需 parity） | 第一性计数（−200~240 发/步），但同为 launch 账 | 可与 R1/R3 的 GPU A/B **并行写码**（不占 GPU） |
| **R5** | **tcgen05**（grouped gate/up） | −1.0~3.8 / **?** | 会话成本 | **高（blocked）** | 2 轮对齐修复失败（TMA bulk 16B 硬对齐无 fallback） | go/no-go：符号预检 → 冒烟 LEN=0/misaligned 即关 |
| **R6** | **L4**（占用/MLP） | −5~8 / **?** | **16~21 人日** | **极高（零背书）** | v17→v21 四变体全中性（「只动一个因子无效」） | 仅当 S5 实测 >15ms 且 accept ≥3 |

### 4.1 推荐执行序（关键路径）

```
Step 1  三臂基线（A/B/C）× 2 prompt × 3 run      ← 唯一真门，不测则后面每格都污染
  ↓
R1      SH_PAIR M=6 A/B（base×m1f1×m1f2×m1f6）
  ↓
R2      hc 链（先落一行 truncate 修复 → A/B）
  ↓
R3      mrows 族（逐个，止损 40%）
  ↓
R4      B6（写码与 R1/R3 的 GPU A/B 并行）
  ↓
R5      tcgen05 go/no-go（独立会话）
  ↓
[R6 L4 条件触发]
```

**两条与任务不同的排序理由**：
1. **R2（hc）排在 R3（mrows）之前**——尽管 mrows 是「零代码」，但它的**兑现概率被两次零收益先例严重打折**，
   而 hc 是「一行修复 + 树内折核件齐备 + 有带宽分析背书」，**期望值更稳**。
2. **R3 单变量、逐 gate**——`batched_400_v2.sh` 默认把 6 个 mrows gate **一起开**，失去归因能力；
   必须拆成单变量 A/B（`swallow-unlock-400-sprint §B2`）。

---

## 5. 400 可达性数学（任务 Q4）

### 5.1 阶梯（**两条起点口径并列**）

`tok/s = 6 / step_ms × 1000`（accept 5，k_emit=6）。设计增量取 mid（来源：`swallow-unlocked-400-final-path §3.1`）。

| 阶段 | 增量 | **S0=31ms（设计）** | tok/s | **S0=34ms（实测上界）** | tok/s |
|---|---:|---:|---:|---:|---:|
| **S0** 解锁基线（无 SH_PAIR） | — | **31.0** | **194** | **34.0** | **176** |
| S1 +SH_PAIR M=6 | −6.4 | 24.6 | 244 | 27.6 | 217 |
| S2 +mrows 族 | −5.15 | 19.5 | 308 | 22.5 | 267 |
| S3 +hc 链 | −1.5 | 18.0 | 333 | 21.0 | 286 |
| S4 +B6 | −1.08 | 16.9 | 355 | 19.9 | 302 |
| **S5 +tcgen05** | −2.4 | **14.5** | **414 ✓** | **17.5** | **343 ✗** |
| S6 +L4 | −6.5 | 8.0 | 750 | 11.0 | 545 |

### 5.2 三个硬约束

```
① 400 @ accept 5  ⇒ 步时 ≤ 15.0ms
   · S0=31 + 设计 Σ(16.5) = 14.5ms ⇒ 414 ✓（需 ≈97% 兑现，仓史无先例）
   · S0=34 + 设计 Σ(16.5) = 17.5ms ⇒ 343 ✗（差 14%）
   ⇒ S0 的 3ms 差值 = 400 的 ✓/✗ 分界
② 400 @ accept 3  ⇒ 步时 ≤ 10.0ms
   · 全足额 14.5ms ⇒ 276 ✗（差 31%）；必须叠 L4（8.0ms ⇒ 500）
   ⇒ accept 3 的 400 属于 L4 之后
③ 400 @ accept 1.214（出师表）⇒ 步时 ≤ 5.54ms  ⇒ 物理不可达（< L5 floor 8–9ms）
```

### 5.3 兑现率门槛（代数）

```
所需兑现率 = (S0 − 15.0) / Σ_design(S1..S5) = (31 − 15) / 16.5 = 97.0%
60% 兑现（历史值）= 31 − 9.9 = 21.1ms ⇒ 284 tok/s ⇒ ✗（差 29%）
```
**且反向证据**：nsys AR 实测 **27.1%** > 设计 17–20%（`swallow-unlocked-400-final-path §0-3`）⇒
通信被低估 35–40% ⇒ **实际兑现率很可能 <60%**。

### 5.4 判定（三句）

1. **400 @ accept 5**：**条件可达**（14.5ms/414），但要求 S1–S5 **≈97% 全额兑现**——仓史无先例；
   现实落点更可能 **~21ms / ~284 tok/s（差 29%）**。
2. **400 @ accept 3**：**本路线不可达**（全足额仅 276），必须叠 L4（16~21 人日、零背书）。
3. **400 @ 出师表**：**物理不可达**。
   ⇒ **400 是「counting 口径（accept≥5）+ 全足额」的目标。**

### 5.5 与 lazy 的对比（路由，不是性能）

`lazy 更好 ⟺ mean_k < B/c − 1`（`c = 6.15 ms/row`）：

| B（batched 步时） | 阈值 mean_k | 计数 5.0 | 出师表 1.214 | 对话 0.96 |
|---:|---:|---|---|---|
| **31ms**（当前 S0） | **4.04** | batched ✓ | lazy ✓ | lazy ✓ |
| 21ms（S2 后） | 2.41 | batched ✓ | lazy ✓ | lazy ✓ |
| **15ms**（S5 全兑现） | **1.44** | batched ✓ | lazy（勉强） | lazy ✓ |
| 10ms（S6 L4） | 0.63 | batched ✓ | batched ✓ | batched ✓ |

⇒ **当前形态只有计数型走 batched 划算**；优化步时本身在扩大 batched 的适用面。
**SWALLOW 不能当全局默认**——正确形态是「SWALLOW 常开 + lazy⇄batched 按任务路由」（产品决策，提请仲裁）。

---

## 6. 测试矩阵（一次 GPU 会话，背靠背交错）

```bash
# ── Step 1：基线三臂 × 2 prompt（6 serve）
# ── Step 2：SH_PAIR 四臂 × 1 prompt（4 serve，复用 sh_pair_ab.sh）
# ── Step 3：hc 一行修复后的 A/B（2 serve）
# 每臂必录（缺一不能下结论）：
#   1) /proc/<pid>/environ 逐门读回
#   2) nm -D $SO 符号存在性
#   3) [dspark] steps= 的 verify/draft/commit 中位
#   4) nsys 按 kernel 名聚合（sh_exp_pair_kernel<6> vs 逐行）
#   5) 计数数字顺序 + 前 61 行 + k_acc 逐位 + 零额外拉丁
#   6) [verify_graph] captured verify_graph_m6
```

### 6.1 每条结论的「最小证据集」

| 结论 | 最小证据 |
|---|---|
| S0 = X ms | 3× run 的 `steady_median` + `ar5-hang=0` + 计数 k_emit mode=6 |
| 观测税 = Y ms | A vs B 同 prompt 同会话 |
| pad 可退 | C 臂 `ar5-hang=0` 且红线绿 |
| SH_PAIR 上场 | `nm -D` 符号 + `/proc/environ` + nsys `...<6>` |
| SH_PAIR 有效 | `steady_median` 位移 ≥2ms（否则 instruction-bound） |

---

## 7. 陷阱清单（本路径专属，全部有源码/文档理由）

1. **`download_u32` 是同步拷贝**（`devrt.rs:1302`）⇒ 每个观测 probe 都是流水线同步点；
   **`V5_LEDGER` = 10 点/步**（4 canary + 1 epoch，×2）⇒ 吞吐轮必须 `V5_LEDGER=0`。
2. **`canary 1→4 槽`把 ledger 成本翻了 4 倍**（`tp.rs:333`）——旧「ledger 只贵一点点」口径作废。
3. **`SH_PAIR M=6` 只需 `DSV41_SH_PAIR_M=1`**（`chain_dev.rs:13526`）；
   `t1t4` P6 的「需 `SH_EXP_FUSED`+`SH_PAIR`」是把它和旧 M=1 臂（`:13589`）混了——设了会得幻影门。
4. **`batched_400_v2.sh:154` 仍是 `SWALLOW_EPOCH_PAD=1`**（常数 pad），与 `t1t4` P4 的
   「常数 pad + 动态 pad = over-pad」冲突 ⇒ Step 1 手写 gate 串，不跑脚本。
5. **`sh_pair_m` 的 decline 是无声的**（返回 `Ok(false)` 回落逐行）⇒ 只能靠 nsys 取证，
   **缺 nsys 证据不得下「已上场」结论**。
6. **SH_PAIR 的 `−4.9~7.9` 是 launch 账**；nsys 实测该族仅 ~8–9%（≈2.5–2.8ms）⇒
   预期按 `−0.8~2.8ms` 编，别按 7.9。
7. **`VERIFY_HEAD_MROWS` 与 m=6 历史出过 ar5-hang** ⇒ 最后单独上（`batched_400_v2.sh:149` 里它默认开着，
   这是历史组合，A/B 时须先关掉再逐个开）。
8. **`[verify_graph] captured verify_graph_m6` 必须出现**——否则 6 行块退直发，SWALLOW 缩水，
   SH_PAIR 的 A/B 前提不成立。
9. **口径三件套**：每次比较必须标 `arm + m + timer`（serve 墙钟 / `[dspark] steps=` / nsys），
   禁止跨会话跨栈比（本轮 `56.6` 就是一次口径混用的产物）。
10. **accept 是门槛不是可选项**：τ < 2.80 时 400 物理不可达（现出师表 τ=1.214）。

---

## 8. 交付清单（供尚书省分派）

| # | 项 | 内容 | 优先级 | 风险 |
|---|---|---|---|---|
| 1 | **Step 1** | 三臂基线（A/B/C）× 2 prompt × 3 run，钉死 S0 + 观测税 + pad 税 | **P0** | 无 |
| 2 | **Step 2** | SH_PAIR M=6 四臂 A/B（三证上场 + 四腿判据） | **P0** | 中（符号/形态 decline） |
| 3 | Step 3 | hc 链：先落一行 truncate 修复（`chain_dev.rs:11625`），再 A/B | P1 | 中（零拉丁） |
| 4 | Step 4 | mrows 族逐个 A/B（拆开脚本的默认并联） | P1 | 低 |
| 5 | Step 5 | B6 实现（可与 1/2 并行写码） | P2 | 中（parity） |
| 6 | Step 6 | tcgen05 go/no-go | P2 | 高 |
| 7 | 脚本 | `batched_400_v2.sh` 正式化：`EPOCH_PAD → DYNAMIC_PAD` + FORBIDDEN 增列「EPOCH+DYNAMIC 同设」 | P1 | 低（需重跑基线） |
| 8 | 条件 | L4（仅当 S5 >15ms 且 accept ≥3） | P3 | 极高（16~21 人日） |

**不做**：
- 不在 Step 1 前叠任何优化 gate（失去 S0 的归因能力）。
- 不在吞吐测量里开 `V5_LEDGER`（10 个同步点/步）。
- 不把 `SH_PAIR_M` 与其它 gate 同轮（它是唯一真收益点，须单变量）。
- 不为 mrows 族「一起开」编预算（会被 instruction-bound 吃掉）。

---

## 9. 一页纸结论

1. **`56.6 tok/s` 是口径，不是退化**：`k_emit≈2.2`（出师表）配 31ms 步时。**计数口径下同一 31ms 给 194 tok/s**
   ——**S0 已被本次运行间接证实 ≈31ms**。任务是口径混用。
2. **观测开销 = 同步点**：`download_u32` 是同步拷贝；`V5_LEDGER` 因 canary 1→4 槽已涨到 **10 个 D2H/步**。
   ⇒ 基线测量必须三臂分离（观测 / pad / 纯净）。
3. **SH_PAIR M=6 只需 `DSV41_SH_PAIR_M=1`**（`t1t4` P6 的双门说法是错的）；收益按 **−0.8~2.8ms** 编
   （nsys 实测族占比 ~8–9%），A/B 定谳——争点是 phase-1 grid 9→54 块的并行度增益。
4. **ROI 序**：SH_PAIR M=6 → hc 链 → mrows 族 → B6 → tcgen05 → L4。
5. **400 可达性**：@accept 5 **条件可达**（14.5ms/414，需 97% 兑现，无先例；现实 ~21ms/284）；
   @accept 3 **需 L4**；@出师表 **物理不可达**。
   **S0 的 ±3ms 就是 400 的 ✓/✗ 分界 ⇒ 第一优先是钉死 S0，不是上优化。**

---

*工部 · 只读勘察 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*所有 ms 标来源（实测 / launch 账 / 设计 / 代数）；行号以 HEAD `d9261bb` 为准；与任务前提冲突处已显式给出依据与 file:line。*
