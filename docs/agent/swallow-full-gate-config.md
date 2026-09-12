# SWALLOW 全 gate 配置（每个 gate 的 SWALLOW 适用性判定）

> 工部 · 2026-09-12 · **只读勘察 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 基线：工作树 HEAD `1a41467`（**注意：`chain_dev.rs` / `device.rs` / `tp.rs` 有未提交的并行改动**，
> 行号可能漂移；结论均以**函数名 + gate 表达式**为准，不依赖行号）。
> 读码：`crates/ferrite-models/src/dsv41/{chain_dev,dspark_dev,config}.rs`、
> `scripts/batched_400_v2.sh`、`docs/agent/{nsys-clean-stack-91-design,accept-1214-to-2-3-path,verify-specific-fusion-kernel-design,swallow-unlocked-shpair-m6-throughput-plan}.md`。

---

## 0. 先读六条（含三条**修正任务前提**，必须上报尚书省）

1. **❗「TAP_INPUT / DRAFT_BF16_DOMAIN 是 SWALLOW 缺的 accept 杠杆」——成立，但少了一个前置。**
   `TAP_INPUT` 的 tap hook **整段被 `cfg.dspark_armed()` 包着**（`layer()` 与 `layer_rows()` 两处），
   而 `dspark_armed()` 要求 **`DSV41_DSPARK=1`**。任务给的 SWALLOW 配置清单里**没有 `DSV41_DSPARK`**，
   也没有 `DSV41_SIDS_WRITEBACK`——**只加 TAP_INPUT 而不加 DSPARK，等于没加**（tap 根本不写；
   且 `dspark_spec_step` 第一件事就是 `if !cfg.dspark_armed() { return Err(...) }`，直接报错）。
   `nsys-clean-stack-91-design.md:29` 早就记过同一件事（"用户给的栈清单漏了两个权威 base env 里有的 gate"）。

2. **❗「R2 让 lazy 栈 +15.6%」里的 `R2`（`DSV41_ATTN_LIN_FUSE`）在 SWALLOW 下是空门（不可移植）。**
   `attention_rows` 里 `lin2_gate = lin_gate.lin2() && m == 1`、`rope_norm_gate = lin_gate.rope_norm() && m == 1`
   ——**`m == 1` 专属**。SWALLOW 的 verify 恒 `m = VERIFY_ROWS = 6`。
   ⇒ `+15.6%` 里只有 **MARKOV(+3.2%) + FORK(+1.5%) + RING_WIN(+0.3%) ≈ +5%** 可移植；
   **R2 的 +10.2% 不会出现在 SWALLOW 上**。SWALLOW 的 R2 对应物是 **K1/K2**
   （`DSV41_ATTN_MROWS2` / `DSV41_ATTN_MROWS_ROPE_NORM`，`verify-specific-fusion-kernel-design.md` 的
   "R2 替代方案"，`m > 1` 专用），两者 **DEFAULT OFF / 从未在 SWALLOW 上实测**——不要按 R2 的票面编预算。

3. **❗`DSV41_LAZY_SDR` 在 SWALLOW 下是空门，且 `DSV41_LAZY_VERIFY` 必须为 0。**
   `lazy_sdr()` 只出现在 lazy 的逐行路径（`lazy_tap_commit` / lazy round），其前置
   `spec_tap_deferred` **只由 lazy 臂置位**、`reset()` 清掉；SWALLOW 路径上恒为 `false`。
   且 §0-4 的臂分派里 `lazy_verify()` 分支**排在 `swallow_step()` 之后但会覆盖它**（见 §2），
   `batched_400_v2.sh` 的 `FORBIDDEN` 把 `DSV41_LAZY_VERIFY` 列第一：
   lazy 的 v5 footprint 是 `3 + 81*k_emit`（k_emit 相关），**常数 epoch pad 无法拉平 ⇒ 永久 epoch rift / 挂死**。

4. **臂分派优先级（`dspark_spec_step`，决定"哪些 gate 不能同时开"）**：
   ```
   (1) swallow_step() && spec_primed_unanimous() && spec_primed  → dspark_spec_swallowed  ★ 目标臂
   (2) lazy_verify() && spec_primed                              → lazy | swallowed（路由）
   (3) seed_align() && spec_primed                               → dspark_spec_aligned
   (4) 以上都不满足                                              → legacy 5 行臂（+ step_dev）
   ```
   ⇒ SWALLOW 生产配置必须 **`LAZY_VERIFY=0` + `SEED_ALIGN=0`**（首选臂是 (1)，但 (2) 会抢，(3) 是另一个块布局）。

5. **`batched_400_v2.sh` 的 `FORBIDDEN` 与代码当前状态**不一致，**必须先重验再决定**：
   脚本禁 `DSV41_HC_VERIFY_FUSE` / `DSV41_HC_FRONT_ROWS`（理由：把 `BF16_TRUNCATE` 带进 verify，
   破坏零拉丁 = 050c7fd）；但代码里这个洞**已在 2026-09-12 被堵**：
   * A1-a：`collapse_norm_rows` → `hc_collapse_norm(..., /*truncate=*/ false)`（硬编码 false，注释明说）；
   * A2：`layer_rows` 的两处 `hc_mixes_auto(..., truncate = false)` 同样硬编码 false。
   ⇒ 代码注释称"此后 `HC_VERIFY_FUSE=1` 可以重新武装（不再触碰 verify 数值）"，**脚本注释仍是旧的**。
   结论：这两个门**可以进配置，但必须走红线复验**（零拉丁 + 计数前 61 行），不能凭脚本注释判死。

6. **`DSV41_VERIFY_AR_FOLD` 单选无效**：`ar_hc_post_fold_rows` 的 decline 条件含
   `!Self::hc_verify_fuse()`。⇒ **`VERIFY_AR_FOLD=1` 而 `HC_VERIFY_FUSE=0` 时它整个是死的**。

---

## 1. SWALLOW 路径的判据（"是否生效"的定义）

SWALLOW = `step_rows(m = 6)`（**没有** `step_dev`，从第 2 轮起），逐层走
`layer_rows()`（不是 `layer()`）+ `moe_rows()` + `compressor_mrows()`。

由此得两条机械判据：

| 判据 | 含义 |
|---|---|
| **M 判据** | 门的读点落在 `layer_rows` / `attention_rows` / `moe_rows` / `compressor_mrows` / `step_rows` 里 ⇒ SWALLOW 生效 |
| **m=1 判据** | 门表达式含 `m == 1`、或其函数只被 `layer()`（单行）/ lazy 逐行路径调用 ⇒ SWALLOW **空门** |

**`m == 1` 专属的门**（在 SWALLOW 下不生效）：
`DSV41_ATTN_LIN_FUSE`(R2)、`DSV41_INDEXER_QR_RAW`(R2b)、`DSV41_LAZY_SDR`、`DSV41_HC_FRONT` 的单行臂，以及 **`DSV41_BF16_TRUNCATE`**（见 §3.2）。

---

## 2. 臂分派与 epoch 安全（不能同时开的门）

| 门 | 处置 | 依据 |
|---|---|---|
| `DSV41_LAZY_VERIFY` | **必须 0** | 与 SWALLOW 同开会抢臂；`3 + 81*k_emit` footprint 无法用常数 pad 拉平 ⇒ epoch rift |
| `DSV41_SEED_ALIGN` | **必须 0** | 另一个 6 行块布局（`dspark_spec_aligned`），A/B 会串味 |
| `DSV41_SWALLOW_EPOCH_PAD` | **必须 1** | 第 9 个修复本体（补 81 个空 round，让各臂 rounds/step 相等）；脚本称"NOT optional" |
| `DSV41_SWALLOW_DYNAMIC_PAD` | 可选（更强） | 11-B per-step epoch consensus：不管"为什么"落后都能拉平（含 head 几何 ±1 / capture 臂） |
| `DSV41_VERIFY_GRAPH` | 与 SWALLOW 有**暖机耦合** | `SWALLOW_GRAPH_WARMUP_BLOCKS = 3`：`swallow_step()` 时前 3 个 block 不给图（DRY→CAPTURE→REPLAY 的 ar5-hang 修复） |

---

## 3. 逐 gate 适用性判定

### 3.1 SWALLOW 核心（必需）

| gate | SWALLOW | 依据 |
|---|---|---|
| `DSV41_SPEC=1` | ✅ 必需 | 进 spec 路径 |
| `DSV41_DSPARK=1` | ✅ **必需（任务清单缺！）** | `cfg.dspark_armed()` 是整个 tap hook 的前置；`dspark_spec_step` 无它直接 `Err` |
| `DSV41_SIDS_WRITEBACK=1` | ✅ **必需（任务清单缺！）** | `s.ids` 回写；缺它下一轮 embed 的 token 会滞后 `k_acc` 位（计数任务自锁重复） |
| `DSV41_SWALLOW_STEP=1` | ✅ 核心 | 臂选择 |
| `DSV41_SWALLOW_EPOCH_PAD=1` | ✅ **必需（任务清单缺！）** | §2 |
| `DSV41_VERIFY_GRAPH=1` | ✅ 生效（M 判据） | `step_rows` 图化，含暖机窗口 |
| `DSV41_TIMING=1` | ✅ 诊断（只影响输出） | `[dspark] steps=` 行 |
| `DSV41_DSPARK_DEBUG=1` | 诊断（可关） | 逐轮 trace |
| `DSV41_V5_LEDGER=1` | 观测税：吞吐轮**必须 0** | 每次 probe 5 个 D2H（4 canary + epoch），pre+note = 10 个同步点/步 ⇒ 5–10ms |

### 3.2 正确性组

| gate | SWALLOW | 依据 |
|---|---|---|
| `DSV41_EXPERT_ACT_E4M3=1` | ✅ 生效（M 判据） | `moe_rows`（+ 单行 + draft 各一处）都读它 |
| `DSV41_BF16_TRUNCATE=1` | ⚠️ **m=1 专属 ⇒ SWALLOW 下只作用于 bootstrap 轮** | `layer_rows` 两处硬编码 `truncate = false`；只有 `layer()`（第 1 轮的 `step_dev`）读门 |

> **这条要如实上报**：`BF16_TRUNCATE` 是脚本里的"零拉丁红线（non-negotiable）"，但在 SWALLOW 的**稳态**
> 它一次都不生效——SWALLOW 从第 2 轮起没有单行 forward。它的实际作用面 = **第 1 轮 bootstrap**。
> 也就是说：**SWALLOW 的零拉丁红线不能靠 `BF16_TRUNCATE` 保证**，必须靠 m 行链自身的 f32 语义 + 红线复验。

### 3.3 accept 杠杆（本次关键新增）

| gate | SWALLOW | 依据 |
|---|---|---|
| `DSV41_TAP_INPUT=1` | ✅ **生效！** | `layer_rows`：(spec_capture ‖ spec_tap_deferred) && tap_input() → 把 tap 从"层输出"挪到"层输入"（reference `model.py:1261-1267`）；SWALLOW 的 `spec_capture=true` 包住 `step_rows` ⇒ 写入 `dspark_tap_r` 的每行；再由 `carry_kept_tap` 取 `k_emit-1` 行交给下一轮 draft。**注意 `layer()` 也有一份同门 hook（bootstrap 轮）** |
| `DSV41_DRAFT_BF16_DOMAIN=1` | ✅ **生效**（draft 专属，臂无关） | `dspark_dev` 四处：head `normed`(C#8)、MoE `xn`(C#9)、attn-out / o_lora(p04-woa-format) |

**TAP_INPUT 在 SWALLOW 下的语义（回答任务问题 2）**：
* 是，**swallowed 臂确实使用 tap_input**——两条 hook（`layer_rows` 顶部的 INPUT 采集、底部的 OUTPUT 采集）
  **互相排斥**（`if tap_input()` / `if !tap_input()`），保证只有一份被写。
* **SWALLOW 特有的位置语义**：swallowed 臂的 draft 用的是**上一轮**的 tap（本轮 anchor 的 forward 是块的第 0 行，
  发生在 draft 之后），`carry_kept_tap` 交的是**最后一个已提交行**（`k_emit-1`）的隐藏。
  这是设计内的**位置**差异（文档已写死），与 `TAP_INPUT` 无关；
  `TAP_INPUT` 只改**采集点（层输入 vs 层输出）**，不动这个位置。
  ⇒ 两者叠加才是"官方几何 + 官方采集点"。
* **实测票面**：`accept-1214-to-2-3-path.md §4` —— `TAP_INPUT` 是唯一被证实的 accept 杠杆 **+19%**。

### 3.4 e2e 优化组（lazy 91.1 栈）——**适用性必须逐个重判**

| gate（lazy 栈名） | SWALLOW | 依据 |
|---|---|---|
| `DSV41_MARKOV_SLICED`（MARKOV） | ✅ 生效 | draft head 几何（`markov_head_geom`，draft 路径，臂无关） |
| `DSV41_VERIFY_FORK`（FORK） | ✅ 生效（M 判据） | `attention_rows` 的 attn dual chain + `moe_rows` 的 MoE dual chain；**但** MoE 半边在 `sh_exp_mrows()/sh_pair_m()/sh_exp_fused()` 任一为真时被排除 ⇒ 与 SH_PAIR_M 同开时**只剩 attention 半边** |
| `DSV41_RING_WIN_FUSE`（RING_WIN） | ✅ 生效（M 判据） | R3 `verify_ring_win_fuse`，`attention_rows` 里 `m` 行的 ring append + per-row window（默认 OFF，这是 R3 的 A/B 面；B2 单行面默认 ON） |
| `DSV41_ATTN_LIN_FUSE`（R2） | ❌ **空门** | `m == 1` 专属（§0-2） |
| `DSV41_LAZY_SDR`（SDR） | ❌ **空门** | lazy 逐行专属（§0-3） |
| — | 🆕 SWALLOW 的 R2 对应物 = `DSV41_ATTN_MROWS2`(K1) + `DSV41_ATTN_MROWS_ROPE_NORM`(K2) | 均为 `m > 1` 的 verify 专用融合；默认 OFF / 未实测 |

### 3.5 Wave 1（m 行族）——**M 判据全过**

| gate | SWALLOW | 依据 / 备注 |
|---|---|---|
| `DSV41_SH_EXP_MROWS=1` | ✅ | `shared_expert_mrows`（m 行） |
| `DSV41_SH_PAIR_M=1` | ✅ | M 行 shared expert（`m <= 8` ⇒ m=6 OK）；**只需这一个门**（`.so` 符号 `dsv41_gemm_fp8_sh_exp_fused`）；**不需要** `SH_EXP_FUSED`/`SH_PAIR`（那是下面旧 M=1 臂的门） |
| `DSV41_HC_VERIFY_FUSE=1` | ✅ 代码已放行 / ⚠️ 脚本仍禁 | A1：`collapse_norm_rows` 硬编码 `truncate=false` ⇒ 洞已堵；**须红线复验** |
| `DSV41_HC_FRONT_ROWS=1` | ✅ 代码已放行 / ⚠️ 脚本仍禁 | A2：`layer_rows` 两处 `hc_mixes_auto` 硬编码 `truncate=false`；**须红线复验** |
| `DSV41_VERIFY_AR_FOLD=1` | ⚠️ **依赖 HC_VERIFY_FUSE** | `ar_hc_post_fold_rows`：`!verify_ar_fold() ‖ !fuse_c() ‖ !hc_verify_fuse()` ⇒ decline。单独开 = 死门 |
| `DSV41_GATE_MROWS=1` | ✅ | `row_fold_gate`（= `DSV41_ROW_FOLD_GATE` 别名），`moe_rows` 的多行 gate GEMV（`m <= 8`） |
| `DSV41_INDEXER_MROWS=1` | ✅ | `attention_rows` 的 indexer front |
| `DSV41_COMPRESSOR_MROWS=1` | ✅ | verify block 的 compressor 合并 |
| `DSV41_VERIFY_HEAD_MROWS=1` | ✅（**任务清单缺，权威脚本有**） | verify 的 sliced head GEMV 行折叠 |
| `DSV41_NORM_MROWS=1` | ✅（**任务清单缺**） | verify block 的 rmsnorm 多行 |
| `DSV41_MROWS_SMALL_N_ADAPTIVE=1` | ✅（**任务清单缺**，kernel 侧 getenv） | `dsv41_kernels.cu:3825`；靶子是 shared expert n=288 / wkv n=512 —— **SH_PAIR_M 落地后该靶子基本消失**，可留作 A/B |

### 3.6 draft 组

| gate | SWALLOW | 依据 |
|---|---|---|
| `DSV41_DRAFT_P3A=1` | ✅ | draft 链的 P3a 折叠（臂无关） |
| `DSV41_DRAFT_GRAPH=1` | ✅（任务清单缺） | draft 的 CUDA 图（权威脚本有；与 lazy 有交互，对 SWALLOW 无） |

---

## 4. 全 gate SWALLOW 配置（可直接 source）

```bash
# ============================================================================
# SWALLOW 全 gate 配置 v1  (2026-09-12, baseline 1a41467)
# 用法：source 后再启 serve。行尾 ★ = 任务清单缺、但必需；❌ = 空门（保留仅为 A/B，勿计入收益）
# ============================================================================

## --- A. 核心（必需）---
DSV41_SPEC=1
DSV41_DSPARK=1              # ★ 必需：武装 tap hook；缺它 dspark_spec_step 直接 Err
DSV41_SIDS_WRITEBACK=1      # ★ 必需：s.ids 回写（滞后 embed 的自锁修复）
DSV41_SWALLOW_STEP=1        # 核心臂
DSV41_SWALLOW_EPOCH_PAD=1   # ★ 必需：第 9 个修复（81 空 round 拉平 epoch）
# DSV41_SWALLOW_DYNAMIC_PAD=1  # 可选更强（11-B per-step epoch consensus）
DSV41_VERIFY_GRAPH=1        # verify 图化（含 SWALLOW 3-block 暖机）
DSV41_TIMING=1              # 诊断（测量必需）
# DSV41_DSPARK_DEBUG=1      # 诊断（逐轮 trace，吞吐轮建议关）
# DSV41_V5_LEDGER=1         # 观测税：吞吐轮必须 0（10 个同步点/步）

## --- B. 正确性 ---
DSV41_EXPERT_ACT_E4M3=1
DSV41_BF16_TRUNCATE=1       # ⚠️ m=1 专属：SWALLOW 下只作用于第 1 轮 bootstrap

## --- C. accept 杠杆（本次关键新增）---
DSV41_TAP_INPUT=1           # ✅ 生效（layer_rows + layer 两条 hook 互斥）；实测 +19% accept
DSV41_DRAFT_BF16_DOMAIN=1   # ✅ 生效（draft，臂无关）

## --- D. e2e 优化（lazy 91.1 栈，逐个重判后）---
DSV41_MARKOV_SLICED=1       # ✅ 生效
DSV41_VERIFY_FORK=1         # ✅ 生效（attention 半边；MoE 半边被 SH_PAIR_M 排除）
DSV41_RING_WIN_FUSE=1       # ✅ 生效（R3 verify ring+win，m 行）
DSV41_ATTN_LIN_FUSE=1       # ❌ 空门（m==1 专属）——保留仅为 A/B，勿计 +10.2%
# DSV41_LAZY_SDR=1          # ❌ 空门 + 禁止与 SWALLOW 同开
# --- SWALLOW 的 R2 对应物（未实测，A/B-first，勿入 v1 预算）---
# DSV41_ATTN_MROWS2=1            # K1
# DSV41_ATTN_MROWS_ROPE_NORM=1   # K2

## --- E. Wave 1（m 行族，M 判据全过）---
DSV41_SH_EXP_MROWS=1
DSV41_SH_PAIR_M=1           # M=6 shared expert；不需要 SH_EXP_FUSED/SH_PAIR
DSV41_HC_VERIFY_FUSE=1      # ⚠️ 脚本禁；代码已堵 truncate 洞 → 须红线复验
DSV41_HC_FRONT_ROWS=1       # ⚠️ 同上
DSV41_VERIFY_AR_FOLD=1      # ⚠️ 依赖 HC_VERIFY_FUSE=1（否则死门）
DSV41_GATE_MROWS=1
DSV41_INDEXER_MROWS=1
DSV41_COMPRESSOR_MROWS=1
DSV41_VERIFY_HEAD_MROWS=1   # ★ 权威脚本有
DSV41_NORM_MROWS=1          # ★ 权威脚本有
DSV41_MROWS_SMALL_N_ADAPTIVE=1  # ★ 权威脚本有（SH_PAIR_M 后收益变小）

## --- F. draft ---
DSV41_DRAFT_P3A=1
DSV41_DRAFT_GRAPH=1         # ★ 权威脚本有

## --- G. 必须保持 OFF ---
# DSV41_LAZY_VERIFY=1       # 禁止：抢臂 + k_emit 相关 footprint ⇒ epoch rift
# DSV41_SEED_ALIGN=1        # 禁止：另一个 6 行块布局
```

### 4.1 最小可跑集（若只想要 accept 修复，不动 Wave 1）

```bash
DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 DSV41_TIMING=1 \
DSV41_SWALLOW_STEP=1 DSV41_SWALLOW_EPOCH_PAD=1 DSV41_VERIFY_GRAPH=1 \
DSV41_EXPERT_ACT_E4M3=1 DSV41_BF16_TRUNCATE=1 \
DSV41_TAP_INPUT=1 DSV41_DRAFT_BF16_DOMAIN=1 \
DSV41_DRAFT_P3A=1 DSV41_DRAFT_GRAPH=1
```
（= 权威 `batched_400_v2.sh` 矩阵 **+ TAP_INPUT + DRAFT_BF16_DOMAIN**，即任务要害。
`V5_LEDGER` 吞吐轮给 0。）

### 4.2 环境侧（非 gate，但同一次启动必须齐）

`FERRITE_*=1` 族 + `NCCL_NVLS_ENABLE=0`（该节点必带）+ `CUDA_VISIBLE_DEVICES` + `DSV41_KERNELS=<.so>`
+ `DSV41_MODEL_DIR`；**双产物同源**（`.cu` 变了先 `build.sh 103a` 再 `cargo build`）。

---

## 5. 收益口径校正（别再拿 91.1 的票面套 SWALLOW）

| 项 | lazy 91.1 栈 | SWALLOW 可移植部分 |
|---|---|---|
| base | — | — |
| R2 `ATTN_LIN_FUSE` | +10.2% | **0（m=1 空门）** |
| MARKOV `MARKOV_SLICED` | +3.2% | +3.2% ✅ |
| VERIFY_FORK | +1.5% | +1.5% ✅（MoE 半边被 SH_PAIR_M 去掉，票面可能略降） |
| RING_WIN | +0.3% | +0.3% ✅ |
| **合计** | **+15.6%** | **≈ +5%（不含 R2）** |

⇒ 任务书里"e2e 优化 gates（R2/MARKOV/FORK/RING_WIN = +15.6%）"**不能整包搬到 SWALLOW**：
**+10.2% 的 R2 不在其中**。若要把这 10% 找回来，路径是 K1/K2（`ATTN_MROWS2` + `ATTN_MROWS_ROPE_NORM`），
**需要一次独立的 A/B + parity**（`kernels/cuda/tests_dsv41_r2_parity.cu` 是现成骨架）。

**accept 侧的真正大头**仍是 `TAP_INPUT`（+19%，几何项）+ `DRAFT_BF16_DOMAIN`（数值项），
且 `56.6 tok/s` 的口径问题（`swallow-unlocked-shpair-m6-throughput-plan.md §1`）说明：
**必须在计数 prompt（accept 5）上测**才能看到 SWALLOW 的真票面（≈194 tok/s @ 31ms 步时）。

---

## 6. 待尚书省裁决的三件事

1. **`HC_VERIFY_FUSE` / `HC_FRONT_ROWS` 是否解禁**：代码洞已堵（硬编码 `truncate=false`），
   脚本仍禁。建议走一次**红线复验**（零拉丁 + 计数前 61 行 + EAGER 并列），过则解禁并**同步改脚本的
   `FORBIDDEN`/注释**（否则下一次运行会被脚本自己拦下）。
2. **是否投入 K1+K2** 作为 SWALLOW 的 R2 替代（这是唯一能找回那 ~10% 的路径，但零实测背书）。
3. **`VERIFY_AR_FOLD` 的依赖**是否需要显式断言（当前 `HC_VERIFY_FUSE=0` 时静默死门，
   建议加一行启动告警，避免"设了但没生效"的幻影门）。

---

## 7. 一句话交付

> **SWALLOW 全 gate = 权威 `batched_400_v2.sh` 矩阵 + `DSPARK`/`SIDS_WRITEBACK`/`SWALLOW_EPOCH_PAD`（补必需）
> + `TAP_INPUT`/`DRAFT_BF16_DOMAIN`（accept 要害，+19%/+数值）
> + `MARKOV_SLICED`/`VERIFY_FORK`/`RING_WIN_FUSE`/`SH_PAIR_M`/`VERIFY_HEAD_MROWS`/`NORM_MROWS`/`DRAFT_GRAPH`（可移植的 e2e）
> + `HC_VERIFY_FUSE`/`HC_FRONT_ROWS`/`VERIFY_AR_FOLD`（须红线复验）
> − `ATTN_LIN_FUSE`/`LAZY_SDR`/`LAZY_VERIFY`/`SEED_ALIGN`（m=1 空门或抢臂）。**
