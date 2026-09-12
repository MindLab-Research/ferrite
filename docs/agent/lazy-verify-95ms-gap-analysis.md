# lazy verify 剩余 9.5ms 的具体来源（工部 · kernel 视角）

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 代码基线：HEAD `3d9709b`（`crates/ferrite-models/src/dsv41/chain_dev.rs`，逐条 `file:line` 核对）。
> 输入账本：`lazy-verify-optimization-path.md` · `verify-ms-breakdown.md` · `verify-calc-floor.md` ·
> `verify-architecture-floor.md` · `sh-pair-template-m-design.md` · `tcgen05-e4m3-grouped-expectation.md` ·
> `routed-expert-residual.md` · `nsys-wave1-analysis-framework.md` · `dspark-correctness-chain.md` ·
> `draft-graph-lazy-interaction.md`。
> **本机无 GPU ⇒ 所有 ms 标了来源（实测 / 账本推算 / 设计口径）。**

---

## 0. 判决（先读六条）

1. **🔴 「33ms 的 lazy verify」这一行必须先证实真伪——它很可能是 batched，或含 prefill 污染。**
   仓库内部口径的 lazy 步时是 **22.56ms**（verify 18.11 + draft 4.28 + commit 0.17，`dspark-correctness-chain` 实测）。
   33ms 出现在三处：① `dspark-correctness-chain` 的 serve 侧「生成阶段 ~24 步 × ~33ms」
   （**serve 墙钟，含 curl/SSE/admission/queue/tail**，`nsys-wave1-analysis-framework §0-1`）；
   ② batched 臂 ~37~39ms 的四舍五入；③ `nsys_wave1.sh` 无 `--capture-range`，CSV 里
   **load + prefill + decode 三段同符号合并**（`§1.3`）。**没有一条是「lazy 纯 GPU 执行时间」。**
   ⇒ **第一件事：读 `[dspark] route=` 行 + nsys 窗口切分，确认 lazy 真被选中、且 33ms 是 decode 净窗。**
   后者与 22.56 差 ~10ms，**这 10ms 可能本身就是那「9.5ms」的记账误差**（见 §3.0）。

2. **表里 8 个族的 ms 是 m=5 口径，lazy 是 m=1 逐行——不可直接相加。**
   `shared 10.4 / routed 8.3 / attention 2.8 / head 1.12` 都是 m=5 账本值
   （`verify-ms-breakdown §1`：1000 / 400 / 880 / 10 发）。lazy 每行一个 m=1 forward，
   **launch 数按 `k_emit=2.214` 重标定**，不是沿用 m=5 的数。最典型：`shared expert` 在 m=1 下
   是 **5 发/层 = 200 发/行 ≈ 2.08ms/行**（不是 10.4；`lazy-verify-optimization-path §0-4`）。

3. **🔴 「✓ mrows」这一列在 lazy 下全是空账。** `GATE_MROWS / INDEXER_MROWS / VERIFY_HEAD_MROWS /
   ATTN_MROWS / ROPE_MROWS` 的语义是「把 m 行折成 1 发」；**m=1 时 rows=1，mrows ≡ 逐行**
   （`swallow-unlocked-next-plan §3.1`，commit `33af60c` 已显式更正）。
   把 gate/indexer/head 标成「✓」会让 9.5ms 的来源被系统性低估。

4. **2.03ms/行残差的三个候选里，两个已被仓库排除。**
   `图 replay overhead ≈ 0`（m=1 图化实测 **−1.6ms**，24.15→22.56，replay 比裸链**更快**）；
   `host_barrier ≈ 0`（barrier-batch A/B `b018101f` 无变化）。**只剩 host round-trip + 路径差异**（§2）。

5. **lazy 的 per-row 地板 = EAGER 的 6.15ms/行，不是 5ms。**
   EAGER 本身也是 m=1 逐 token（`dspark-correctness-chain`「EAGER 是 m=1 逐 token 的 step_dev 循环」）。
   ⇒ `2.214 × 6.15 + 3.6 + 0.2 = 17.4ms` 是 **lazy 架构下的步时地板**。
   **15ms 在这条地板之下 ⇒ 必须（a）accept↑ 减行数，或（b）把 c_row 压到 6.15 以下（换核）。**
   这正是任务「关键问题」里那 17.3ms 的含义——**它不是接近 15，它是 lazy 的硬地板**。

6. **9.5ms 的诚实归属（本文件结论，详见 §3/§4）：**
   ```
   4.5ms  ← per-row 残差（c_row 8.18 → 6.15）：host round-trip + verify 路径的 spec 开销
   2.3ms  ← 打破 EAGER 地板：tcgen05(gate/up) −2.0 + SH_PAIR M=1 −1.2（后者的 phase-1 收益在 lazy 不成立）
   1.0ms  ← routed down 无 tcgen05 + e4x 120× 过量（本 arm 拿不到，需 swapAB+kRing 重写）
   1.0ms  ← per-step 族（hc+AR）在 lazy 下的 ×k_emit 重付（结构税，无机械解）
   0.7ms  ← draft（P3A+MARKOV，flag 就位）/ 其余地板抹平
   ```
   **没有一项是「翻 flag」能拿的；2.3+1.0 是换核，4.5 是「让 verify 路径对齐 EAGER」。**

---

## 1. 33ms 分解的逐项重标定（回答任务 1/2/3/4）

### 1.1 shared expert —— 「9.2ms」在 lazy 下不成立

| 口径 | 发数 | ms | 来源 |
|---|---:|---:|---|
| m=5 账本 | 25 发/层 × 40 = **1000** | **10.40** | `verify-ms-breakdown §1`（85 GB/s） |
| **lazy m=1 折算** | **5 发/层 × 40 = 200/行** | **≈ 2.08/行 → 4.6/步** | `lazy-verify-optimization-path §0-4` 的 m=1 折算 |

**⇒ 任务表里的 `shared expert ~10.4` 放在 lazy 表里是错口径。** 只有一种情况例外：
若「33ms」是 batched 臂（路由 bug 或 serve 假值），10.4 才是对的——**这正是 §0-1 要先用读数裁决的**。

**SH_PAIR M=1 为什么只省 1.2ms：**
- `gemm_fp8_sh_pair_kernel`（`dsv41_kernels.cu:6511`）的 grid 按 `max(n1,n2)` 定尺（= 160 block），
  而 **phase 1 的映射 `row = blockIdx.x*32 + warp` 在 `n1=sh_il=288` 时只有 9/160 个 block 在干活，
  151 个 block 在 grid barrier 上空转**（`sh-pair-template-m-design §0-2/§1.1`）。
- M=1 时这条并行度缺陷**不会被修**（`template<M>` 的修复是 `M × ceil(n1/32)` 个 block，`§2.3/§3.3`）。
- M=1 的真实收益只有：**5 发 → 2 发**、去掉 `sh_act_r` 的 global 往返、swiglu 独立 launch 消失。
  ⇒ 1.1~1.3ms，与实测口径一致。
- **`template<M≥2>` 的 −4.9~7.9ms 对 lazy 完全不可用**（lazy 恒 m=1）。
  ⇒ **shared expert 的最大杠杆是 batched-only**——这是任务表用「SH_PAIR M=1（−1.2ms）」掩盖掉的结构性缺口。

### 1.2 routed experts —— 「−2ms」已是现实值，−6.8ms 从来不在这条臂上

- `tcgen05-e4m3-grouped-expectation §0/§5.1`：**−6.8ms 是 swapAB 全 routed（gate/up＋down）的目标数**；
  本 arm 是 `tc5::e4x` **dense masked tile**（只覆盖 gate/up），与那个设计**不是同一个 kernel**。
- 实测拆分（`routed-expert-residual §2.3`，m=5）：**gateup 4.82 / down 3.48**。
  **down 没有 tcgen05 版本**（`tcgen05_bench.sh` 头注）⇒ 3.48ms 原样保留。
  ⇒ 本 arm 收益上限 = 4.82ms，**修正落点 −1.0~−2.8ms**。
- e4x 的三个成本结构（`§5.2`）：**M=128 固定 ⇒ ~120× 张量核过量**（每 expert 只 1~3 行，
  却跑满 128 行 MMA）；**无 cp.async / TMA / 流水**（单缓冲 + 每 atom 一次 commit/wait）；
  **每 32 块一次 TMEM 回读 + 寄存器折叠**。
- ⇒ **任务里「tcgen05 −2ms」已经吃到现实上限**；剩下的 ~4.3ms（down 的 3.48 + e4x 过量）**不是调参能拿的**，
  要 `expert-tcgen05-plan.md` 的 **swapAB + kRing=8 TMA ring**（研究级 1~3 周 GPU 迭代，
  前置门：单层 microbench 打 22.2µs(gateup)/17.2µs(down)）。

### 1.3 attention —— 因果序问题在 lazy 下**根本不存在**

- 缺陷 #2（`verify-architecture-floor §4.2`）：verify 块内是 `read(r)→append(r)→read(r+1)→append(r+1)`，
  `window=128` 且长上下文恒回绕 ⇒ 行 `0..m-2` 会读到**块自己的未来行**。
  `DSV41_ATTN_MROWS` 因此在 `world>1 || pos+m-1 ≥ win` 时**全部 decline**——生产里永不生效。
- scratch ring 方案：**`clen_rows[m]` 设备快照**（部分已在：`indexer_rows_one` 已带 `mrows_clen: Option<ptr>`）
  + ring/window 的 block 内 **r 升序合核**（或块前 ring 快照）；`kv_snap_ring` 已分配（10.5MB）。
- **但对 lazy 无价值**：lazy 恒 m=1 ⇒ 无「未来行」⇒ 因果缺陷不触发，且 **mrows ≡ 逐行**。
  attention 在 lazy 下的真实成本 ≈ `2.80/5 × 2.214 ≈ 1.2ms/步`，已接近其 launcher 地板。
  ⇒ **attention 不是 9.5ms 的来源**，把它列成「未优化」会误导预算。

### 1.4 head —— 同样是 lazy 恒 0 的 mrows

- `VERIFY_HEAD_MROWS`（`dsv41_gemv_bf16_v1_mrows`，`chain_dev.rs:6138/6187`）折的是 m 行；**m=1 ⇒ 逐行**。
- head 是**全表唯一跑满带宽的族**（5.9 TB/s = 77% 峰值，`verify-architecture-floor §2.1`），
  `verify-calc-floor §4.1` 判「无肉」。
- lazy 下 head ≈ `1.12/5 × 2.214 ≈ 0.5ms/步` ≈ 地板。
- **per-row 优化（任务问的）没有意义**：它已经在带宽地板上；真正的 head 收益（切分 −1.3ms）**已实施**
  （`DSV41_VERIFY_HEAD_SLICED` 默认 ON），折叠（再 6×）因 K 序 parity **不做**。
  ⇒ **head 不是来源。**

---

## 2. 🔴 关键问题：`c_row = 8.18` vs EAGER `6.15` —— +2.03ms/行从哪来

### 2.1 先把三个候选钉死（`lazy-verify-optimization-path §1.2` + `dspark-correctness-chain`）

| 候选 | 判定 | 证据 |
|---|---|---|
| **图 replay overhead** | **≈ 0，净负** | m=1 图化 **−1.6ms**（24.15→22.56）。replay 的 submit 被 CUDA async launch 隐藏（`swallow-nograph §2`）；每行只摊一次性 DRY+CAPTURE。**不是残差。** |
| **host_barrier** | **≈ 0** | barrier-batch A/B（`b018101f`）：22.59 ≈ 22.56，无变化。**不是残差。** |
| **argmax D2H** | **✅ 有，~0.5ms/行** | `step_rows_sync` 结尾把 `argmax_r` 阻塞 D2H 回读——**这正是早退的依赖**（`chain_dev.rs:5329-5337` 的 `m*4` 字节契约）。`k_emit × 0.5 ≈ 1.1ms/步`。 |

### 2.2 host round-trip 的剩余两项（已可消）

| 项 | 现状 | 代码事实 |
|---|---|---|
| `set_pos_ctr` H2D | **4B blocking H2D/行** | `fn set_pos_ctr` = `ul_i32(self.s.pos_ctr.ptr, &[pos])`（`:8682`） |
| `pos_ctr` read-back D2H | **每行一次 full device sync** | `step_rows_sync` 的 `download_u32(pos_ctr)`（`:5350`）——**lazy 已用 `pos_base_hint` 消除**（`:5348`） |
| `lazy_tap_commit` | **3× D2D/行** | `DSPARK_TAP_SLOTS=3` 次 `memcpy_d2d`（`:8137-8143`） |

**✅ 这三项的解已经写在代码里了：`DSV41_LAZY_SDR`**（`fn lazy_sdr`，`:2356`，**默认 OFF**）：
- `lazy_run_row` 跳过 per-row `set_pos_ctr`（`:8182`）；
- `dspark_spec_lazy` 只在块首写一次 counter（`:8364`）；
- `lazy_tap_commit` 用**一次 `cudaMemcpy2DAsync`** 取代 3 次 D2D（`:8124-8135`）。
⇒ **这一项是「已实现、只差 A/B」的零代码收益**（账本 −0.7ms，与 L2 同源：`lazy-verify-optimization-path §L2`）。

### 2.3 残差的大头：verify 路径 vs EAGER 路径的**逐行核集差异**

`c_row verify 8.18` 与 `c_row EAGER 6.15` 都是 m=1。**两者不是同一个 per-row 核集**：

| 差异 | 量级 | 依据 |
|---|---:|---|
| **hc 链**：verify 走 raw chain（10 发/层），EAGER 走融合（4~6 发/层） | **+0.8~1.5/行** | `verify-eager-fusion-migration §2.1`；`HC_VERIFY_FUSE/HC_FRONT_ROWS` 默认 OFF（`chain_dev.rs:12239/12293`） |
| **verify 独有的 spec 路径**：compressor pool/commit 逐行、tap staging、`engram` 逐行 | **+0.2~0.5/行** | `verify-ms-breakdown §1` 的「其余」 |
| **host round-trip**（§2.1+§2.2） | **+0.5~0.6/行** | 同上 |
| **合计** | **≈ +1.5~2.6/行** | 与残差 **+2.03**（区间 1.4~2.4）一致 |

> ⚠️ **口径提醒（`§6-2` 的同一警告）**：hc 那一项与 §3 的「per-step 族 ×k_emit」是**同一现象的两个切法**，
> **不得相加**。前者按「每行比 EAGER 贵多少」记，后者按「每步重付几遍」记。

### 2.4 如果残差消除：`c_row=6.15` ⇒ 17.4ms（**这是地板，不是接近 15**）

```
2.214 × 6.15（EAGER c_row）+ 3.6（draft）+ 0.2（commit） = 17.42 ms
```
任务说「17.3ms 接近 15ms」——**方向对，但性质要摆正**：17.4ms 是**在 lazy 架构下、把 spec 开销完全抹平
后能达到的最低步时**（因为 EAGER 本身就是 m=1 逐 token，6.15ms/行是 m=1 forward 的地板）。
**要进 15ms，必须让 c_row < 5.05ms（= (15−3.8)/2.214）——即比 EAGER 还快 18%。**
在 SIMT GEMV 的核效率下，这只能靠**换核**（tcgen05）+ **真正的族级融合**，不是抹残差。

---

## 3. 9.5ms 的来源分解

### 3.0 先扣掉「记账差」——这部分可能不是 kernel 问题

| 差 | 量 | 说明 |
|---|---:|---|
| serve 33ms vs 内部 22.56ms | **~10ms** | serve 墙钟含 curl/SSE/admission/queue/tail；**不含在模型预算里**（`nsys-wave1 §0-1`） |
| 「生成阶段 ~24 步 × ~33ms」与 lazy 内部口径 | ~10ms | 若路由选了 batched（τ/B 初始化 bug，`dspark-correctness-chain:497`），33ms 是 **batched** 的数 |

⇒ **在把 33ms 当作 lazy 基线之前，必须先排掉这 ~10ms 的记账/选臂误差**——否则「9.5ms」里一半是幻觉。

### 3.1 分解表（按「24.5 → 15」）

| # | 来源 | ms | 性质 | 能否在 lazy 下拿 | 依据强度 |
|---|---|---:|---|---|---|
| **1** | **per-row 残差**（c_row 8.18→6.15）：argmax D2H + counter H2D + tap D2D + hc 未融 + spec 路径 | **4.5** | host round-trip + 路径对齐 | 部分：SDR（−0.7）零代码；hc 融合（−1.3~1.7/块 ×k_emit）；argmax D2H **去不掉**（早退依赖） | 实测锚 + 账本 |
| **2** | **打破 EAGER 地板**（c_row < 6.15）：tcgen05(gate/up) + SH_PAIR M=1 | **2.3** | 换核 | ⚠️ SH_PAIR M=1 的 −1.2 里含「已实现」；tcgen05 −2.0 已在 24.5 里 | 设计口径 |
| **3** | **routed 的锁死半边**：down 无 tcgen05（3.48）+ e4x 120× 过量 | **1.0** | 换核（研究级） | ❌ 需 swapAB+kRing 重写（1~3 周） | 代码级 |
| **4** | **per-step 族 ×k_emit**：hc 400 发/行 + AR 80 轮/行，lazy 逐行重付 | **1.0** | 结构税 | ❌ 无机械解（除非 accept↑ 减行 / 回 batched） | 实测锚 + 推论 |
| **5** | **draft**：P3A + MARKOV_SLICED（+ 长任务 DRAFT_GRAPH） | **0.7** | flag | ✅ 已就位，A/B 即可（长任务才兑现 DRAFT_GRAPH） | 账本 |
| | **合计** | **9.5** | | | |

> **重叠警告**：#1 的 hc 项与 #4 是同一现象的两种记账，**本表已按「#1 = 每行对齐 EAGER 的差；#4 = 每步重付的遍数」分开**，
> 相加前需确认不重复（`lazy-verify-optimization-path §6-2`）。

### 3.2 与任务表的对账（哪些项被高估/低估）

| 任务表项 | 我的修正 | 差 |
|---|---|---|
| shared expert 9.2ms | **lazy 下 ≈ 4.6ms**（m=1 口径）；9.2 是 m=5 | 高估 ~4.6ms |
| routed 6.3ms | 6.3 可信；但**能再收的只有 down 半边 + e4x** | 一致 |
| attention 2.8ms | **lazy 下 ≈ 1.2ms**，且因果序修复对 lazy 无价值 | 高估 ~1.6ms |
| head 1.1ms | **lazy 下 ≈ 0.5ms**，已带宽地板 | 高估 ~0.6ms |
| gate/indexer/hc/AR「✓」 | **lazy 下 mrows ≡ 逐行**；「✓」是空账 | 误导 |

⇒ **任务表把大量 m=5 口径混进 lazy 表**，使「9.5ms 从哪来」这个问题本身偏了 ~6ms。
**先把表按 m=1 重标定，9.5ms 才会收敛到上面 §3.1 的 5 项。**

---

## 4. 优化方案 + 实施成本（逐项）

| # | 项 | 改动 | 预期 | 成本 | 风险/前置 |
|---|---|---|---|---|---|
| **A** | **口径校准（必做，先做）** | 无代码：确认 `[dspark] route=`、nsys 窗口切分、按 kernel 名聚合 lazy decode 段 | 把「9.5ms」变成实测 | 1 GPU 会话 | 无；**不做则后面全部重复记账**（`§1.3` 的 R6 陷阱） |
| **B** | **`DSV41_LAZY_SDR=1`** | **0 代码**（已实现，`:2356/:8124/:8182/:8364`） | −0.7ms/步 | 0.5 人日 A/B | 判据：`rows_run == k_emit` 不变 + `DSV41_INV_CHECK=1` 全绿 |
| **C** | **hc 融合（`HC_VERIFY_FUSE` + `HC_FRONT_ROWS` + `VERIFY_AR_FOLD`）** | **0 代码**（env） | −1.3~1.7/块 **×2.214 = −2.9~3.8/步** | 0.5~1 人日 A/B | truncate 坑已修（`6f5de2b`）；**一次只动一个 gate**，读 kernel 名确认生效 |
| **D** | **argmax D2H 压缩** | 代码：把「早退依赖」改成**行 0 的 argmax 单独 D2H + 后续行批量回读**（或 `cudaStreamQuery` 轮询） | 0（诚实：早退依赖不可去）**~ −0.3ms**（尾部行可批量） | 1~2 人日 | ⚠️ 不能买 speculative row（`§L2` 的论证：+1.79 行/步 = +10ms） |
| **E** | **SH_PAIR M=1** | **0 代码**（`DSV41_SH_PAIR_M=1`，parity 已修 `3d9709b`） | −1.1~1.3/步 | 0.5 人日 | **lazy 下无 phase-1 收益**（`§1.1`）；A/B 中性≠融合无效 |
| **F** | **tcgen05 e4m3 grouped（gate/up）** | 5-gate 链 + **GPU parity** | −1.0~2.0/步 | **4~5 人日 + parity 会话** | e4m3 臂**从未上机**；`down` 无核；回退时因 `GATEUP_FUSE=0` **+0.4ms 反而更慢** |
| **G** | **draft P3A + MARKOV_SLICED** | **0 代码**（flag） | −0.7~0.8/步 | 0.5 人日 | 已就位 |
| **H** | **DRAFT_GRAPH（L6）** | 0 代码 + **barrier 对称化**（`draft-graph-lazy-interaction §4`，per-rank latch 的 epoch 错位） | 长任务 −3.3/轮；**短探针 ~0** | 1 人日 + 1 GPU | `pos≥win` 覆盖率：短跑 ~11~36%；**上机前必须补对称化** |
| **I** | **tcgen05 down（swapAB+kRing）** | 新核（研究级） | −1.5~2.0/步（down 3.48 → ~1.5） | **1~3 周 GPU 迭代** | 前置门：单层 microbench 22.2/17.2µs，不达标止损 |
| **J** | **attention scratch ring（clen_rows + r 升序合核）** | 新代码 | **lazy 下 0**（m=1 无因果缺陷）；**batched 下 −1.3/步** | 3~4 人日 | 只在打算回 batched 时值 |

**ROI 排序**：**B + C + G（零代码，合计 −4.3~5.3ms）→ E（零代码）→ A（校准）→ F（换核）→ H → I/J。**

---

## 5. 诚实校准（必须写在账上）

1. **「33ms lazy verify」未经证实**。三处线索都指向它不是 lazy 纯执行时间（serve 墙钟 / batched / prefill 污染）。
   本文件的一切增量都以「内部口径 lazy 22.56ms」为锚；**A 会话是唯一能把它变实测的动作**。
2. **表里 5 个族的 ms 是 m=5 口径**，lazy 是 m=1。**直接用这张表算 9.5ms 会偏 ~6ms**（§3.2）。
3. **§2.3 的 hc 项与 §3.1 的 #4 是同一现象的两个切法，不得相加**（沿用 `lazy-verify-optimization-path §6-2`）。
4. **F（tcgen05）从未上机**，且本 arm 的收益上限被 down 锁死；**13.6~15.3ms 是条件落点**，
   无 F 的诚实落点是 **17.4ms**（= lazy 的 m=1 地板）。
5. **本项目的 #1 测量陷阱是「gate 设了但没生效」**（5 个 mrows gate 静默 `return Ok(false)`、
   `SH_EXP_FUSED` 需 `.so` 符号、`HC_VERIFY_FUSE` 反向默认）。每个 gate 翻转必须**读回 + 单独 commit
   + 看 kernel 名**，而不是只看 `verify=`（`final-400-config §6-R9`）。
6. **本机无 GPU**，所有 ms 标了来源，未实测项集中在 D4/D9 与 §3.0 的记账差。

---

## 附：一句话总结

> **9.5ms 里，~6ms 是「用 m=5 口径的表去算 m=1 的 lazy」造成的记账偏差（shared/attention/head/mrows 四类），
> 真正的 kernel 缺口是 ~4.5ms 的 per-row 残差（argmax D2H 不可去 + hc 未融 + spec 路径）
> 加 ~2.3ms 的「打破 EAGER 地板」（tcgen05 gate/up + SH_PAIR M=1，后者在 lazy 下无 phase-1 收益）
> 加 ~1ms 的 per-step 族 ×k_emit 结构税。**
> **先做 A（校准口径）+ B/C/G（零代码 flag，−4.3~5.3ms），
> 再谈 F（tcgen05，唯一换核项）——在 lazy 的 m=1 架构下，17.4ms 是地板，15ms 需要换核或 accept↑。**

---

*工部 · 只读分析 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*代码行号以 HEAD `3d9709b` 为准，读代码时以函数名为准。*
