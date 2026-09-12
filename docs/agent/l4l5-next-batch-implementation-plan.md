# L4/L5 下一批 kernel 实施计划 —— 从「1a/1b 已落树」到「400 缺口 −7~9ms」

> 工部（ministry-works）· 2026-09-12 · **只读勘察 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码、未改任何 gate 默认。
> 任务：1a（`fold_r`）/ 1b（activation `cp.async16`）已实施待 GPU 验证 ⇒ 规划 **L4/L5 的下一批工作**。
> 代码基线：工作树 HEAD `b3c85db`（`git log` 现场核对）；行号一律以**函数名/符号**为准（本仓有行号漂移史）。
> 输入文档：`l4-mgrid-first-step-design.md` · `l4-occupancy-mlp-design.md` · `l4-l5-kernel-path.md` ·
> `b6-mrows-f32-design.md`（§9 实施记录）· `b5-b4-b6-gpu-verification-design.md` · `l49-ab-test-design.md` ·
> `swallow-unlocked-next-plan.md` · `ar-l4l5-optimization-design.md` · `400-final-frontier-analysis.md`。
> **口径纪律**：每条 ms 标来源（**实测** / **账本** / **代数** / **设计**），本机无 GPU、无 nvcc。

---

## 0. 结论先行（六条，前两条纠正任务前提）

1. **「按 ROI 排序」在 L4/L5 上只能当「先拿白捡的钱」用，不能当施工顺序用。**
   本批次的**真实排序轴是「有没有实测背书」**：本批里有 **5 项（1a/1b/B6/B5/B4）已经全部落树、零代码、只差一次同会话 A/B**
   ——它们是本次唯一能**当场把设计口径换成实测数字**的项；而 L4-3/L4-4/L5-3（账本最大，−3.5~7.0ms）**零实测、要从零写核**，
   排在它们后面。**先清算已落树的，再开新核的**（`l4-occupancy §4.1` 的「ROI 用于拿白捡的钱，波次排序用于关键路径」）。

2. **❗ L4-7 的「dots grid」半边**已经在 m=6 下兑现了 —— **不要再把它写进本批的票面**。
   现场读码：`hc_mix_dots_kernel` / `hc_dots_late_kernel` 的 launcher 是 **`<<<dim3(mix, rows), ...>>>`**
   （`dsv41_kernels.cu` 的 `hc_dots_late` launcher / `hc_mix_dots` 的 `blockIdx.x=m, blockIdx.y=r`）
   ⇒ **m=6 时 (24,6)=144 块 = 97% SM**（任务表已确认）。
   `l4-occupancy §2.2 L4-7` 的「dots 网格从 5 块变 mix × rows」是 **m=1（24 块）口径**——在 batched m=6 下**它已经是 144**。
   ⇒ **L4-7 在本批的剩余增量 = 只有「侧流遮蔽 sinkhorn」那半边 ≈ −0.6ms/步**（不是 −1.3~−1.7，那含 A1/A2 的 L1 欠账，两处不得相加）。

3. **本批的 ROI 冠军是「AR 的 R1」——但它不是 L4/L5 的活。**
   AR 占 **36% / ~6.6ms**（nsys 排名 1）；R1（`DSV41_AR_SINGLE_POLL`，A4 单块轮询）**代码已在树内、默认 OFF、零风险**。
   它是 400 缺口的**头号**来源，但账归 **协议层（AR 工单）**，本文件只把它列为**关键路径前置**，不重复计 L4/L5 的票面。

4. **1a/1b 的预期不变：−0.5 ~ −1.0ms（batched 口径，设计）**，1b 把握最高（姊妹核同款修复已在树），1a 需扫格定符号。
   两者**都逐位等价、都可单变量 A/B**；1a 有个硬耦合——**必须与 `DSV41_SH_PAIR_M=1` 同臂测**，否则 `sh w1/w3` 的 mrows 靶子还在，混淆归因。

5. **B6 是本批「已落树但最重」的一项**：`dsv41_gemm_fp8_mrows_f32`（kernel+launcher+FFI+两处接线+parity 套件+B400 arm）
   已全部落树、`cargo check --workspace` EXIT=0，**CUDA 侧从未编译过**（本机无 nvcc/`.so`）。
   **它不是位等价项**（跳过 fp8 量化往返，行部分和略更准）⇒ 验收走**红线 + `DSV41_DIFF_EAGER`**，不走 memcmp。
   预期 **−0.66~−1.5ms**（launch 账：verify −200~−240 发/步 × 3.3/6.2µs）；**首要判据是 launch 计数，不是 ms**。

6. **400 的缺口账**：`AR R1+R2 = −4~6ms → 22-24ms（250-270 tok/s @ accept 5）`；到 400（= 15ms @ k_emit=6）**还差 −7~9ms**。
   本批 5 项已落树项 + 4 项设计完成项 + 3 项新核项，**设计口径合计 −7.4 ~ −15.5ms**；
   按仓史 **60% 兑现率** ⇒ **−4.4 ~ −9.3ms** ⇒ **落在缺口下沿**。**结论：本批必须足额兑现，才有 400；否则落点 300-340 tok/s。**

---

## 1. 账本校准（先把本批要动的量钉在 nsys 表上）

**SWALLOW nsys 数据（batched m=6，步时 ~28ms）——逐行标「本批谁动它」**：

| 排名 | Time% | 步时 | 族 | **本批动作** | 本批预期 |
|---|---:|---:|---|---|---:|
| 1 | **36.0%** | ~6.6ms | AR（84 轮 × 78.3µs） | ❌ 非本批（协议层）：R1 在树（`DSV41_AR_SINGLE_POLL`）、R2 需设计 | — |
| 2 | 17.4% | ~4.9ms | MoE interleave（tcgen05） | **L4-3 K-split**（+L5-1 PDEPTH 同批） | −1.0~3.0 |
| 3 | **15.1%** | ~4.2ms | gemv 投影（mrows b2+b3） | **1a/1b（已落树）** + L4-1 crossover + L4-8 P2/P3 + L5-2 | −0.7~1.6 |
| 4 | 6.7% | ~1.9ms | hc_dots | **L4-8 KCHUNK**（grid 已 144，见 §0-2） | −0.2~0.3 |
| 5+6 | 8.7% | ~2.4ms | expert_gemv_fp4 | **L4-4 down 换核** + L5-4 | −2.0~2.5 |
| 其他 | 16.1% | ~4.5ms | B6/B5/B4 · L4-7 · L4-9 · L4-6 | **B6/B5/B4（已落树）** + L4-7 侧流 + L4-9 + L4-6 | −1.6~3.3 |

> **读表纪律（沿 `swallow-nsys-batched-analysis-framework §4`）**：nsys 下 **v5 的 publish 自旋被追踪放大**
> （实测 240s/69 步）⇒ **只读 launch 计数与 GridX 分布，不读绝对 ms**。上表「步时」是**账本推算**（Time% × 28ms），
> 不是直接读出的 ms；每一项的收益**必须由同会话 A/B 的 `steady_median` 落定**。

---

## 2. 下一批清单（按「能否当场兑现」分四波，非按设计 ms 单调排）

### W-N1 —— 已落树、零代码、只差一次 GPU 会话（**本批的最高优先**）

| # | 项 | 落点 / gate | 预期（口径） | 把握 | 硬前置 |
|---|---|---|---:|---|---|
| **N1-1** | **1b** 激活 `cp.async16` | `DSV41_MROWS_ACT_CPASYNC`（默认 OFF） | **−0.3 ~ −0.5**（指令账） | **高** | `DSV41_SH_PAIR_M=1` **同臂** |
| **N1-2** | **1a** `fold_r` | `DSV41_MROWS_FOLD_R`（0=auto） | −0.2 ~ −0.5（占用账，需扫格） | 中低 | 同上 |
| **N1-3** | **B6** wo_b m-rows f32 | `DSV41_VERIFY_WOB_MROWS_F32`（默认 OFF） | **−0.66 ~ −1.5**（launch 账 200~240 发/步） | 中 | N1-2 定稿形状 |
| **N1-4** | **B5** gate m-rows + route | `DSV41_GATE_MROWS_ROUTE`（默认 OFF，核在树） | −0.13（launch 账） | 中低 | N1-2 |
| **N1-5** | **B4** kv 半链 rmsnorm+rope | `DSV41_RMSNORM_ROPE_MROWS`（默认 OFF，核在树） | −0.13（launch 账） | 中低 | 必须整块同开（否则 kv 半链回转主流） |
| | **小计** | | **−1.4 ~ −2.8ms** | | |

**为什么这波第一**：五项**全部已落树**（`b6-mrows-f32-design §9` 记录 B6 六件落树；B5/B4 的核 + `B400_B5/B400_B4` arm 已在
`scripts/batched_400_v2.sh`），**零 kernel 开发**，一次 GPU 会话（~1 人日）就能把 5 个设计口径换成实测。

**这一波特有的三个坑**（都在文档里有先例）：
* **`DSV41_MROWS_FOLD_R` / `DSV41_MROWS_ACT_CPASYNC` 是 `.cu` 侧的 `getenv`**（`mrows_act_cpasync_host()`
  + `dsv41_mrows_fold_r_for`），**不在 Rust 侧** ⇒ `/proc/<pid>/environ` 读不到这**两个**（能读到 Rust gate）。
  **判活的唯一硬证据是 nsys 的 `GridX` 分布**（`fold_r=M` ⇒ GridX∈{144,128,320,512,640}；`fold_r=1` ⇒ ×6）。
* **1a 的 nsys 判据失效点**：kernel 名不变（同一个 `gemm_fp8_mrows_kernel<6>`）⇒ **按 kernel 名数 launch 的框架在这里无效**，
  必须改用 `cuda_gpu_trace` 的 `GridX` 列。
* **B6 非位等价** ⇒ 红线（计数数字顺序 + 出师表零拉丁）+ `DSV41_DIFF_EAGER` mismatch **不增**；
  **首要判据 = `dsv41_quant_fp8`(wo_b) 调用数归零 + `gemm_fp8_mrows_f32_kernel` 计数 ≈ 40+3/步**。

### W-N2 —— 设计完成、纯接线 / 翻 gate（无需新核）

| # | 项 | 落点 / gate | 预期（口径） | 把握 | 硬前置 |
|---|---|---|---:|---|---|
| **N2-1** | **L4-7**（剩余半边）hc 侧流遮蔽 sinkhorn | `hc_mixes_auto` + `dl` 流（A2 接线已在树） | **−0.6**（per-step 设计） | 中高 | `DSV41_HC_FRONT_ROWS`（**默认已 ON**）+ `hf` 侧流原语 |
| **N2-2** | **L4-8** hc_dots KCHUNK | `DSV41_HC_DL_KCHUNK`（默认 OFF，核在树） | −0.2 ~ −0.3（hc） | 中 | KCHUNK 对齐约束（启动器自带） |
| **N2-3** | **L4-9** CNORM/NORM dim-split A/B | `DSV41_CNORM_SPLIT` / `DSV41_NORM_SPLIT`（默认 OFF，核在树） | −0.1 ~ −0.3 | 低 | ⚠️ **`NORM_SPLIT` 单独设是空臂**（需 `DSV41_NORM_MROWS=1` 真对照 T1-C0） |
| **N2-4** | **L4-1** mrows crossover | `kMrowsSmallN2: 512→640`（吃 wkv n=512） | −0.2 ~ −0.6 | 中低 | **L3（SH_PAIR_M）落地后趋 0**（shared n=288 被接管） |
| | **小计** | | **−1.1 ~ −1.8ms** | | |

**N2-1 的诚实边界**：`l4-occupancy §2.2` 写 L4-7 = −1.3~−1.7ms，**那是 A1+A2 的总账（归 L1）**；
**L4 自己的增量 = 侧流遮蔽 + dots 网格 = −0.6ms**。本波**只取 −0.6**（§0-2 已证 dots 网格已在 m=6 兑现）。

### W-N3 —— kernel 重写（**L4 的绝对量大头、唯一关键路径**）

| # | 项 | 落点 | 预期（设计） | 人日 | 把握 | 硬前置 |
|---|---|---|---:|---:|---|---|
| **N3-1** | **L4-3** tcgen05 gate/up **K-split**（第三 grid 维 + 升序 reduce） | `tc5::mxf4::expert_tcgen05_gateup_mxf4_kernel` / `m4_launch_gateup`（源码 3 处 `[K-SPLIT TODO]`） | **−1.0 ~ −3.0** | 3~4 | 中 | **L2 tcgen05 落地** + U2 微基准门（gateup 22.2µs，不达标即止损） |
| **N3-2** | **L4-4** tcgen05 **down 换核**（swapAB，从零写） | **新核** `tc5::down::*`（不存在；对照 `expert_gemv_fp4_down_reduce_kernel`） | **−2.0 ~ −2.5** | 3~4 | 中低 | N3-1 形状定稿 |
| **N3-3** | **L5-3** e4x K-chunk TMA ring | `tc5::e4x::*`（抄 `tc5::mxf4` 的 kRing=8） | −0.5 ~ −1.5 | 3~4 | 中 | N3-2 |
| | **小计** | | **−3.5 ~ −7.0ms** | 9~12 | | |

**关键路径（唯一串行主干）**：`L2 tcgen05 gate/up → N3-1 K-split → N3-2 down → N3-3 e4x kRing`。
**这一波是本批唯一能越过「5% 峰值」那堵墙的动作**（`arch-floor §5.2`：先删 μop，再上 warp）。

**三条止损线**（写死，不许事后放宽）：
* **U2 单层微基准不达标 ⇒ 该项不进集成**（`tcgen05-e4m3-grouped-expectation §0`）。
* **K-split 的升序 reduce 若无法逐位 ⇒ 走容忍度 A/B（四段文本 + `faults=0`）**，且必须**同一轮只翻这一个变量**。
* **L4-5（e4x M=128 过量）不在本波**：先用 N3-1/N3-2 的**实测**决定是否值得投（`l4-occupancy §2.1` 已判低 ROI）。

### W-N4 —— MLP / 流水 / 满 wave（N3 定稿后收尾）

| # | 项 | 落点 | 预期（设计） | 人日 | 前置 |
|---|---|---|---:|---:|---|
| **N4-1** | **L4-6** verify 多流发射（`dual_chain` / `compress_side` / `moe_dual`） | `chain_dev.rs::layer_rows`（无新核；原语在 `device.rs`） | −0.5 ~ −1.5 | 3~4 | **N2-1**（共用 devrt 原语）+ CUDA graph event 配对 |
| **N4-2** | **L5-1** gateup PDEPTH 2~3 | `expert_gemv_fp4_batched_kernel` | −0.3 ~ −0.6 | 2 | **N3-1 同批**（否则 occupancy 被吃） |
| **N4-3** | **L5-2** gemv/mrows 双缓冲 | `gemm_fp8_mrows_kernel<M>` / `gemm_fp8_gemv_kernel` | −0.3 ~ −0.8 | 2~3 | **N1-2 定稿**（residency vs overlap 陷阱） |
| **N4-4** | **L5-4** down 40-reg + attn 预取加深 | `expert_gemv_fp4_down_reduce_kernel` 等 | −0.3 ~ −0.6 | 1.5~2 | N3-2 |
| **N4-5** | **L4-2** SH_PAIR phase-1 K-split | `gemm_fp8_sh_exp_pair_kernel<M>`（**非逐位**） | −0.3 ~ −0.8 | 2 | L3 的 SH_PAIR 落地 |
| **N4-6** | **L5-5/6** 满 wave 收尾 | 全族 grid 调 148 整数倍 | −0.3 ~ −1.0 | 2 | N3/N4 全部 |
| | **小计** | | **−2.0 ~ −5.3ms** | 13~17 | |

---

## 3. 依赖图（硬约束，照抄并更新 `l4-occupancy §4.2`）

```
[已落树]  1a / 1b / B6 / B5 / B4 ──────────────→ W-N1（一次 A/B，零代码）
                 │（1a 定稿 wkv/wq_a 的 fold_r 与 nwarps）
                 ├──→ N2-4 L4-1 crossover
                 └──→ N4-3 L5-2 双缓冲

[已落树]  HC_FRONT_ROWS(=ON) / hf 侧流原语 ──┬──→ N2-1 L4-7 侧流遮蔽
                                              └──→ N4-1 L4-6 多流发射

[已落树]  hc_dots 144 块 / KCHUNK 核 ────────→ N2-2 L4-8
[已落树]  CNORM/NORM split 核 ───────────────→ N2-3 L4-9（需 NORM_MROWS 真对照）

[L2]  tcgen05 gate/up 落地 ──→ N3-1 K-split ─┬──→ N3-2 down 换核 ──┬──→ N3-3 e4x kRing
                                             │                      └──→ N4-4 down 40-reg
                                             └──→ N4-2 L5-1 PDEPTH（同批）

[AR 工单] R1（在树）→ R2（设计）──── 关键路径前置（进 400 的 15ms 必须先有 AR 的 22-24ms）
```

**三条不得违反的顺序**：
1. **N1 必须在 N3 之前**：N1 零成本且能给 N3 的形状定稿（L4-3 的 grid 要用 1a 后的 `fold_r` 语义做对照）。
2. **N3-1 与 N4-2 同批**：`l4-l5-kernel-path §2.1 L5-1` 明确「必须与 K-split 同时做，否则 occupancy 被吃掉」。
3. **L4-1 排在 L3 之后或同臂**：shared expert n=288 被 SH_PAIR 接管后，L4-1 只剩 wkv。

---

## 4. 400 缺口账（本批能否补上）

**起点**（SWALLOW nsys）：**步时 ~28ms / 214 tok/s**（@ k_emit=6，accept 5）。

| 阶段 | 动作 | Δ | 步时 | tok/s | 口径 |
|---|---|---:|---:|---:|---|
| — | AR R1+R2 | −4 ~ −6 | **22 ~ 24** | **250 ~ 270** | 账本（AR 工单） |
| **W-N1** | 1a/1b + B6 + B5 + B4 | −1.4 ~ −2.8 | 19.2 ~ 22.6 | 265 ~ 312 | 设计（launch/指令/占用账） |
| **W-N2** | L4-7 + L4-8 + L4-9 + L4-1 | −1.1 ~ −1.8 | 17.4 ~ 21.5 | 279 ~ 345 | 设计 |
| **W-N3** | tcgen05 K-split + down + e4x ring | −3.5 ~ −7.0 | **10.4 ~ 18.0** | **333 ~ 577** | 设计（**零实测**） |
| **W-N4** | MLP + 流水 + 满 wave | −2.0 ~ −5.3 | 5.1 ~ 16.0 | 375 ~ 1176 | 设计（**零实测**） |

**判决（诚实）**：
1. **400 的临界点落在 W-N3 的中段**（15ms）：W-N3 足额 ⇒ 越过 400；W-N3 只有一半 ⇒ 落回 333-400 的门口。
2. **W-N1/W-N2 是「必须拿到」的钱**（零/低成本 + 有代码先例）⇒ **现实落点 280-345 tok/s**。
3. **W-N3/W-N4 的 −5.5~−12.3ms 仓内零实测背书**，且反向证据权重不低
   （`{SH_EXP_MROWS, VERIFY_GRAPH, VERIFY_ROPE_MROWS, DRAFT_P3A}` 全开只 −1.21ms 预期 −24；v17→v21 四变体全中性）
   ⇒ **按 60% 兑现率：本批合计 −4.4~−9.3ms ⇒ 落点 300-340 tok/s**，与 `swallow-unlocked-next-plan §6` 的结论一致。
4. **不要用「×k_emit(2.214)」把 lazy 的折算值入预算**（`l4-occupancy §5.2`）——×k_emit 是「为什么 L4 要排在 lazy 上」的**理由**，不是收益承诺。

---

## 5. 验证协议（本批共用一套，逐波可裁剪）

### 5.1 本地硬门禁（0 GPU，每次改动后）
```bash
cargo check --workspace                                   # 项目硬门禁
cd kernels/cuda && bash build.sh 103a                     # .cu 变了 ⇒ 必须先于 cargo build（AGENTS.md 纪律）
# 双产物同源（本仓 #1 陷阱 = 两臂都跑旧路径）
cat kernels/cuda/.build_id  # 与二进制内嵌 id 一致，否则进程拒启
```

### 5.2 每项的三段判据（缺一不算完成）

| 段 | 判据 | 备注 |
|---|---|---|
| **① 活性** | nsys `cuda_gpu_kern_sum` 里该 kernel/GridX 出现，且**计数符合预期** | B6：`quant_fp8`(wo_b) 归零；1a：`GridX = nt·fold_r`；L4-9：split kernel 计数 > 0 |
| **② 正确性** | 逐位项 ⇒ `raw u32` memcmp；非逐位项（B6/L4-2/L4-3/L4-4/L4-9）⇒ **红线**：计数前 61 行 + 出师表零拉丁 + `faults=0` + `ar5-hang=0` | B6 另加 `DSV41_DIFF_EAGER` mismatch **不增** |
| **③ 收益** | **同会话背靠背 A/B** 的 `steady_median` 位移（`STEADY_SKIP=20`），**一 gate 一变一 commit** | nsys 不读 ms；`/proc/<pid>/environ` 实读证明变量进了进程 |

### 5.3 本批特有的四个坑（都在文档里有先例，必须写进每次测试的 checklist）
1. **`.cu` 侧 gate 不进 `/proc/environ`**（1a/1b 是 `getenv` in `.cu`）⇒ 用 nsys `GridX` 判活，别用 envchk 判死。
2. **1a 必须与 `DSV41_SH_PAIR_M=1` 同臂**，否则 shared expert 的 mrows 靶子还在。
3. **L4-9 的 `DSV41_NORM_SPLIT=1` 单独设 = 空臂**（`l49-ab-test-design §0-C1`）⇒ 必须有 `DSV41_NORM_MROWS=1` 的真对照 T1-C0。
4. **gate 翻转单独 commit + 读回**（`dspark-correctness-chain` 的 R6 陷阱：gate 设了但没生效）。

### 5.4 每项的止损线

| 项 | 止损 |
|---|---|
| 1a（`fold_r`） | `wkv` 上中性或负 ⇒ 只留 1b，1a 记「中性」封存 |
| 1b（act cp.async16） | 中性 ⇒ 说明该核不是 staging 受限 ⇒ **重估 instruction-bound 归因**（上报） |
| B6 | `\|Δsteady_median\| < 0.8ms` 且 launch 计数已兑现 ⇒ 判 instruction-bound，记录并转下一项 |
| L4-7 | 侧流开了但 sinkhorn 仍暴露 ⇒ 检查 event 配对是否真的重合（不是「没接上」而是「没重叠」） |
| L4-3/L4-4 | U2 微基准不达标 / parity 无法收敛 ⇒ **不进集成**，不投变体矩阵（勿重演 v17→v21） |

---

## 6. 逐项交付物清单（本批）

| 波 | 项 | 交付物 / 改动落点 |
|---|---|---|
| N1 | 1b | `DSV41_MROWS_ACT_CPASYNC`（**已落树**）+ `kernels/cuda/tests_dsv41_gemm_mrows.cu` 的 parity 扩轴（**若未加**） |
| N1 | 1a | `fold_r` + `dsv41_mrows_fold_r_for`（**已落树**）+ parity 的 `fold_r` 轴 |
| N1 | B6 | **已落树**（§9 六件）+ 本机未编译的 CUDA 侧验收留给远端 |
| N1 | B5/B4 | **已落树**（核 + `B400_B5/B400_B4` arm）；本批只做 A/B |
| N2 | L4-7 | `chain_dev.rs` 的 `dl` 侧流接线（`hc_mixes_auto` 的 sinkhorn 遮蔽）+ 本文件 |
| N2 | L4-8 | `DSV41_HC_DL_KCHUNK=1` A/B（核在树） |
| N2 | L4-9 | `l49-ab-test-design.md` 的 7 臂矩阵执行 + 结果记录 |
| N3 | L4-3 | `dsv41_experts_mxf4.cu` 的第三 grid 维 + 升序 reduce + parity |
| N3 | L4-4 | **新核** `tc5::down::*` + fused asc-slot reduce |
| N3 | L5-3 | `tc5::e4x` 的 kRing TMA + mbarrier |
| N4 | L4-6 | `chain_dev.rs::layer_rows` 三条 fork/join |
| N4 | L5-1/2/4/5/6 | 逐 kernel 的 PDEPTH/双缓冲/reg 压/wave 收尾 |
| 全体 | — | 每项 A/B 结果回写 `docs/agent/`（含实测 `steady_median` 与 nsys 计数），供下一批重排 ROI |

---

## 7. 诚实校准（必须写在账上）

1. **本批 11 项里，只有 W-N1 的 5 项有代码先例（已落树）**；W-N2 的 3 项有骨架；W-N3/W-N4 的 6 项**零实测背书**。
   任何把 W-N3/W-N4 的 ms 当承诺的排期，都会重演 `swallow-unlocked-next-plan §0-C1/C2` 的「计时器混用」事故。
2. **nsys 的绝对 ms 不可信**（v5 publish 自旋放大 300×）⇒ 本文件所有「步时」列都是 Time%×28ms 的**账本推算**，
   必须由同会话 A/B 的 `steady_median` 落定。
3. **1a/1b 的设计预期（−0.5~1.0ms）与反向证据并存**：`SH_EXP_MROWS` 两次实测零收益。
   1b 是「隔壁修过这边漏了」的高把握项；1a 是外推，符号必须由 A/B 定。
4. **本批不含 AR**（36% 的绝对量最大项）：AR 是 400 的**前置**，不是 L4/L5 的票面。本文件把它列在依赖图的头部，
   引用时必须声明两处口径不得相加。
5. **L4-7 的票面从 −1.3~−1.7 修正为 −0.6**（dots 网格已在 m=6 兑现）：与 `l4-occupancy §2.2` 的差异是**口径差**
   （m=1 vs m=6），不是新证据。

---

*工部 · 只读勘察 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码、未改任何 gate 默认。*
*所有 kernel 名以 `__global__` / `extern "C"` 符号为准；所有 ms 标来源（实测 / 账本 / 代数 / 设计）。*
*代码基线 HEAD `b3c85db`；`file:line` 现场核对，读码时仍以函数名为准。*
