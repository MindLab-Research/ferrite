# SGLang 16 步阶梯 → ferrite 门级映射与执行顺序

> 口径：他们 873.6 tok/s 是 **4×GB300 / attention TP4 + MoE TP4 / BS=1 / 模拟 acc 5.5**；目标 = **真实 acc 2.2 下 ~450 tok/s**。
> 算式：`tok/s = tok/step ÷ step`，`tok/step = mean-k + 1 ≈ 3.24` ⇒ **450 tok/s ⇔ step ≈ 7.2ms**（当前 32.5ms，需 4.5×）。

## 执行顺序（三阶，逐条一次一个变量）

| 阶 | 动作 | 预期 | 备注 |
|---|---|---|---|
| **①** | **`DSV41_VERIFY_GRAPH=1`（+ `DSV41_AR_V5=0`）** | verify 28.17 → **~5ms** ⇒ step ≈ 9.5ms ⇒ **~340 tok/s** | 最大且最确定的单项：~7000 次流式发射 ×2.9µs = 20.3ms ⇒ 图内 ~0.4µs/node ≈ 2.8ms |
| **②** | 已有门/已有实现批量转正：`MOE_DOWN_BS`（接线→开）、`MOE_TILELANG`（非捕获臂）+`MOE_BF16_DEQUANT`、MROWS 家族、`P3_MEGAKERNEL`、`HC_VERIFY_FUSE/FRONT_ROWS/AR_FOLD`、`SH_PAIR_M`、**`VERIFY_FORK`**、**`DRAFT_GRAPH`**、**`ATTN_PROJ_ALIGN`** | 设计合计 **−6~−9ms** ⇒ step ≈ 7~8ms ⇒ **400~460 tok/s** ✅ | ⚠️ 每条一次一个变量；`GRAPH_STEP` 臂**必带 `AR_V5=0`**；lazy INIT 臂先预初始化 |
| **③** | MoE down blockscaled（kernel 已在 `.so`，只差 Rust 接线）+ shared expert 单核（照抄他们 PR 39296/39313 的 SF 布局与"**只乘一次**"契约） | **−1~−2ms** ⇒ step ≈ 7.2ms ⇒ **450 落定** | **同时**把五道精度门转正（红线），否则"达标"也是错值 |

## 关键门与位置（当前 HEAD）

| 门 | 位置 | 默认 | 说明 |
|---|---|---|---|
| `DSV41_VERIFY_GRAPH` | `chain_dev.rs:4430` | **OFF** | verify 图化（**已确认 engaged**：`[verify_graph] captured verify_graph_m5 at pos=22` ✓） |
| `DSV41_AR_V5` | — | ON | **开整步图时必须 `=0`**，否则 all-reduce v5 路径挂死 ✗ |
| `DSV41_DRAFT_GRAPH` | `dspark_dev.rs:249` | OFF | draft 3.87ms 的最大杠杆 |
| `DSV41_ATTN_PROJ_ALIGN` | `dspark_dev.rs:159` | OFF | **accept 杠杆**；代码自述"NUMERICAL fix, not perf"（draft 四投影对齐官方 `F.linear`） |
| `DSV41_VERIFY_FORK` | `chain_dev.rs:2734` | OFF | verify 侧双流重叠（博客第 8 步 plain 已兑现 **+23%**，verify 侧未接线） |
| MROWS 家族 | `:4887 / :3201 / :3094 / :2590 / :8071` | 全 OFF | 多行合并；**实测全开仅 −1.21ms**（§5.1） |
| `DSV41_SH_PAIR_M` / `SH_EXP_MROWS` | `:3449 / :3257` | OFF | shared expert 多行；**逐位相同 ✓**，但 `SH_EXP_MROWS` 实测 0 收益 |
| `DSV41_MOE_TILELANG` | `chain_dev.rs:787` | OFF | **实测 70.0µs/层 = SIMT 250µs 的 28%**；但**与 `VERIFY_GRAPH` 互斥**（host `moe_align` D2H 不能进捕获） |
| `DSV41_EXPERT_TCGEN05_E4M3` | `chain_dev.rs` | OFF | grouped tcgen05 e4m3（**同官方 fp4_gemm 格式**）⇒ **实测未上场**：dense 臂要求 `m%128==0`（verify m=6 不可能）；grouped masked 被 `GATEUP_FUSE` 拦住；且 `GATEUP_FUSE=0` 反而 **52→60ms** ✗ |

## ⚠️ 两条硬约束（踩过的）

1. **`MOE_TILELANG` 与 `VERIFY_GRAPH` 互斥**（host `moe_align` 的 D2H 不能进图捕获，`:780-783`）。
2. **手写 fp4 BS 臂（`MOE_TILELANG_BS`/`MOE_BS_HANDWRITTEN`）当前会毁模型** ✗（1..100 全乱码）⇒ **一切测量必须它 OFF**。

## 精度（用户红线）

五道门**全部默认 OFF、出货脚本一个都没开** ⇒ 默认臂下**比官方"更精确" 4 处、乘序错 1 处** ✗：
A2 `WINDOW_KV_QUANT`(`:2320`)、A3 `COMPRESS_LATENT_QUANT`(`:1471`)、A4 `INDEXER_FP4_RT`、`ROUTED_DOWN_QUANT`(`:1395`)、I3 `ATTN_P_BF16`。
加上已接线的 `DSV41_ACTQ_FLOOR`（把量化下限放到 **amax**，官方 `kernel.py:76` 位置 ✓）。
**转正流程**：`~/promote_precision.sh "<GATE=1> [DBG=1]"`（DBG 五点回读逐元素差 0 → `wq_check.py` 文本红线 → 快速臂无回归）✓，**逐项、一次一个变量** ✓。
⚠️ 我们 8 个 `quant_fp8` 调用点**全部 `round_scale=true`** ⇒ 标度恒为 2 的幂 ✓ ⇒ **不存在** SGLang PR 39289 那种"非 2 的幂标度 + UE8M0 位重解释"的静默错值 ✓（唯一退化角落：全 0 block ⇒ `fmaxf(s,1e-30)` 得非幂次，`ACTQ_FLOOR=1` 即闭环）。

## ⚠️ 读数口径（`verify-accept-audit` 纠正，必读——我曾读错）

1. **`step pos=N: X ms (Y tok/s)` 里的 `Y` 恒等于 `1/X`** ✗（`serve.rs:796` 打印的是
   `1.0/dt`，即"步/秒"；非 spec 单步解码下恰好等于 token/s，**spec 下不是**）⇒
   `Y × X ≡ 1.0` 对任何步都成立，**零信息量** ✗。**不要用它推断 accept** ✗。
2. **真正的 tok/步在另一行**：`[dspark] steps=… mean-k=… tok/step=…`（`serve.rs:681-692`）✓，
   其中 **`tok/step = mean-k + 1`**（`emitted.len() = k_acc + 1`，`chain_dev.rs:11925-11927`）✓。
3. **位置增量法**：`DecodeRun` 里 `p += step_len`（`serve.rs:796`），而 `step_time` 打的是**步前**的 `p`
   ⇒ **相邻 `step pos=` 的差值 = 上一步的 emitted 长度** ✓ ⇒ 多 token 步直接可见 ✓。
4. **已记录的 accept（从旧日志）**：参考基线 `ab_dac`/`ab_old` = **mean-k 2.240** ✓（= 用户口径的 2.2 ✓）；
   `ab_m2/m4/rwf/s1on` = 1.310；`ab_hzone` 0.485；`ab_best2` 0.660；
   **`ab_tl2` 0.005 / `ab_mma1` 0.015 = 已判死的"单侧换程序"臂** ✗
   （`docs/agent/proj-mma-verdict.md` 原文：verify 侧单独换程序 ⇒ **mean-k 2.240 → 0.020，accept 崩 99%**）✗。
   ⇒ **accept 崩的根因是"换成了非逐位等价的第三个程序"，不是"m 行结构"** ✓。
5. `DSV41_DIFF_EAGER` 只验 **verify 的 emitted 值**（它们全部来自 `verify_out`/`next`）⇒ **不验 draft** ✗；
   若 accept 问题出在 draft 侧，这个工具是空转的 ✓（历史 55 行里 54 行 `first_mismatch=none` ✓）。

## ⚠️⚠️ 决定性对照（2026-09-13 19:10，重排优先级）

| 配置（其余同：`SPEC=1 DSPARK=1 AR_V5=0`、BS 臂 OFF） | step | 步数 |
|---|---|---|
| `VERIFY_GRAPH=1 GRAPH_STEP=1` | 51 ms | **只 6 步就停** ✗ |
| **两个图都关**（= 文档 32.5ms 的口径） | **61.5 ms** ✗ | 43+ 步 ✓ |
| 文档基线 | 32.5 ms | — |

**结论（两条都很重要）**：
1. **图化确实有用**（51 < 61.5 ✓），但图臂会**在 ~6 步后停住** ✗（日志普查：图 engage 的臂 6 步、无图臂 148 步）
   ⇒ 那个"停住"是**独立问题**（可能与我们之前修过的 `.shared::cluster` / 图槽位有关），**必须先修掉才能用图** ✓。
2. **无论开不开图，我们都远高于文档的 32.5ms（~1.9×）** ✗ ⇒ **⇒ 这个差距不在图化**，
   而在**别处（MoE / 结构）** ✓ ⇒ 与账本一致（verify 里 MoE 的 routed 8.3ms + shared 10.4ms 是最大块）✓
   ⇒ **⇒ 4.5× 只能靠"换核"，而不是调参** ✓。

**⇒ 因此优先级：大块头（MoE 换核：grouped tcgen05 e4m3 = 与官方 fp4_gemm 同格式）> 图化修停住 > 其余门。**

## 📊 开关层已穷尽的实证清单（2026-09-13 19:12，全部实测）

| 杠杆 | 实测 step | 备注 |
|---|---|---|
| 图开（`VERIFY_GRAPH=1 GRAPH_STEP=1`） | **51 ms** | 但**只 6 步就停** ✗ ⇒ 独立待修 |
| **图关（与文档 32.5ms 同口径）** | **61.5 ms** | 43+ 步 ✓ |
| `DRAFT_GRAPH=1` | 51.7 ms | **0 变化** ✗ |
| `VERIFY_FORK=1`（图关） | **59.0 ms** | vs 61.5 ⇒ **+4%** ✓（博客的 +23% 是他们 plain 路） |
| `GATEUP_FUSE=0` | 60 ms | **反向** ✗（只拆融合、未换来 MMA） |
| `EXERT_TCGEN05_E4M3=1 EXERT_GROUPED=1` | 52 ms | **MMA 未上场** ✗ |
| `MOE_TILELANG_BS=1` | — | **毁模型** ✗（1..100 全乱码） |
| 文档基线 | 32.5 ms | — |

**⇒ 结论：开关层找不到那 ~1.9×（更不要说 4.5×）** ✗。差距在**结构**：账本说 verify 37.3ms 里
**MoE（routed 8.3 + shared 10.4）占一半** ✓ ⇒ **只能靠换核** ✓：
1. **grouped tcgen05 e4m3**（与官方 fp4_gemm **同格式** ✓）—— 前置链已逐行读清
   （`moe_route_grouped` 5 门 + `moe_experts_grouped_gate_up` 7 检 ✓；容量常量 `grp_n_experts=max(n_routed, dspark)=384`、
   `grp_topk_max=6`、`m=6≤VERIFY_ROWS=6` **均通过** ✓ ⇒ **拦路点在"符号在但 kernel 未真跑"（门 #4/#5）** ✗）
   ⇒ **待办：在 `dsv41_route_group` / `dsv41_route_gather_rows` 的返回值处加一次性出声**（门 #4/#5 的 `built`/`gathered` ✗）。
2. **shared expert 单核**（照抄 SGLang PR 39296/39313 的 SF 布局 + "只乘一次"契约 ✓）。
3. 图化**先修"6 步停住"**（不然图只能用在短段 ✓）。

