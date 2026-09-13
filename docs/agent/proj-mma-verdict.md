# PROJ-MMA 判决 — **判死（REJECT）**，第二轮 scratch widening 不做

> 上游：`docs/agent/tensorcore-proj-design.md`（§2 数值论证 / §3 实施框架 / §4 收益估算 / §7 并入时机）、
> `docs/agent/proj-mma-gpu-verification.md`（第一轮 ks=1 手册）。
> 第一轮接线（gate OFF 默认）：commit `33b75a7` 之前一轮，载体
> `kernels/cuda/dsv41_proj_mma_skel.cu` + `device.rs`（14 参 ABI）+ `chain_dev.rs`（① MMA > ② MPAR > ③ legacy）。
> 本文是**判决**，不是实测报告：本轮（第二轮）**零代码改动**，未跑 GPU，未做 e2e。
> 唯一的第一轮 GPU datum（mean-k 0.020）由尚书省转达，出处标注在 §1。
> 日期：2026-09-13。

---

## 0. 一句话

**判停。** 这条路的**唯一**收益是**估算**的 −3.5ms；它的**唯一可行形态 (b′)** 要求把 mma 变成
fp8 投影族的**唯一程序** ⇒ 整栈数值基线（刚落地、被 ledger 全部在飞项当作参考线的
**mean-k 2.240**）作废并需全面重校准；而它的**非 (b′) 形态**（只换 verify 侧）已实测
**mean-k 2.240 → 0.020（accept 崩 99%）**，且设计 §2.2 已**证明**"与 SIMT 逐位"不可达 ⇒
**不存在"既保住 accept 参考、又拿到 tensor-core 摊薄"的形态**。

**第二轮 scratch widening 因此不做**（`[ks][M][n]` partial + `ctr` 的 Rust sizing 不落地）。
它**不是**因为工程难度判死——工程上是便宜的（§3.2：scratch ~400 KB、最坏 +15% 流量、一天量级）；
判死的是**性价比**：+14% 的**估算**收益，换**整栈数值程序重置** + S1 级 accept 再调查。

---

## 1. 第一轮 GPU datum，以及它**证明**什么、**不证明**什么

### 1.1 datum（尚书省转达）

| 项 | 值 |
|---|---|
| 回执 | `[proj-mma] ARMED m=5 n=5120 k=1024 ks=1`（wo_b 形状，320 warp） |
| 门禁一 step_ms | 未构成结论（ks=1 网格远小于 ks 规则值，手册 §2.3 已预告） |
| **门禁二 mean-k** | **0.020**（clean 基线 **2.240**，接受带 2–3） |
| 判读（手册 §6.1 判据） | **掉出接受带 ⇒ 直接弃臂，不解释** |

## 1.2 严格读法（别过度归因，也别不足归因）

这**只**证明：**verify 侧单独换程序 ⇒ accept 崩**。这正是设计 §2.3 判的
「非逐位改动**充分危险**」的实测形态，也是 §2.4.2 备选表里"风险类型更差"的那一格
（verify 与 eager 变成两个不同程序 = **WOB 形状**）。

它**不**证明：

- **不**证明 mma kernel 有实现 bug——第一轮**从未跑过**手册 §7 的**内部契约**
  （`row r of mma<M> ≡ row r of mma<1>`，逐位），那是唯一可做的 byte-compare，也是
  "程序自洽"的唯一证据。第一轮唯一跑的是 e2e 臂。
- **不**证明 (b′) 形态也一定崩（(b′) 让 draft/verify 同源，accept 有可能回来）。

**量级值得记一笔（但不是判据）**：WOB 先例是 1.34 → 0.75–0.92（**−35%**），
本轮是 2.240 → 0.020（**−99%**）。差异提示数值偏差可能**大于几 ULP**
（设计 §2.1 的 D1 只该给 ~1e-7）。要分清"near-tie 敏感"与"实现偏差"，只需一次 micro bench
（§7 内部契约 + 与 SIMT 的**容差**分布）。**本轮不做**——理由见 §5：两种结果都**不改变判决**。

## 1.3 为什么"不需要"区分上面两者

判死建立在**两个独立**的事实上，任何单一事实被推翻都不影响结论：

1. **(a) 形态必崩**（跨程序）——已实测，且 §2.3 的读法说明它无法用静态论证排除；
2. **(b′) 形态的成本**（§2）与"−3.5ms **估算**、**零**性能实测"（§3.1）严重不成比例。

---

## 2. (b′) 完整版的精确内容与代价

### 2.1 (b′) 的定义（设计 §2.4.1，逐字）

> 让 mma 程序成为投影族的**唯一**程序——m=1（eager/draft）与 m=2..8（verify）**都走同一个
> `gemm_fp8_mrows_mma_kernel<M>`**。于是"verify row r ≡ eager row r"逐位成立（D 的第 r 列
> 只依赖 B 的第 r 列与 A）。**唯一前提：ks 必须是 (n,k) 的函数，绝不能是 m 的函数**。

**它能给什么 / 不能给什么**：

- ✅ **mma 内部**逐位一致（`mma<M>` 的 row r ≡ `mma<1>` 的 row r）——这是它值钱的地方；
- ❌ **与旧 SIMT 程序**的差**仍在**（§2.1 D1/D2/D3）⇒ 它是"**整栈换程序**"，**不是**恢复原数值。
  故 EAGER 对照与文本质量**都会变**——这是"换数值程序"的全面重校准，不是优化。

### 2.2 工作量清单（每条都对应树内的一个具体面）

| # | 项 | 事实 / 出处 |
|---|---|---|
| W1 | **m=1 侧改路由**：现在 m=1 的投影不是"一个 kernel"，是一个**家族** | `device.rs` 的 fp8 gemm 入口共 16 个，m=1 侧约 13 个（`gemm_fp8_mx` / `mx2` / `mx_rope` / `mx2_rope` / `mx_add` / `mx_f32` / `mx_rope_norm` / `swapab` / `wo_pair` / `sh_pair` / `sh_exp_fused` ……）。mma 要成为**唯一**程序 ⇒ 每个 m=1 调用形状要么有 mma 形态，要么退回**未融合**序列（丢掉 `proj_fuse` / `wo_quant_fuse` / K2 融合省下的 launch）。调用点：`gemm_fp8_mx_or_swap` **7 处**，每处每层一次 |
| W2 | **`DSV41_SWAPAB` 不能当 (b′) 的另一半** | 它是**第三个程序**：自己的 `kSwapabKSplit=8` + `n<1664` decline。第一轮 AP′ 因此**不是** (b′)（手册 §6.2 自陈） |
| W3 | **ks 规则必须全局固定**（§2.4.1 唯一前提） | m=1 侧若无 scratch ⇒ ks 只能 1，而 verify 侧 ks=2/4/8/32 ⇒ **结合律不同 ⇒ 直接违背 (b′)**。**这正是第二轮 scratch widening 存在的唯一理由**——也就是说：**第二轮不是可选项，是 (b′) 的前置条件**。反之，不做 (b′) ⇒ 第二轮无意义 |
| W4 | **验收面重置**：`mean-k = 2.240` 是 S1 fix 刚落地、且 ledger 里**四个在飞项**（5a L2 broadcast −4~6ms / tcgen05-716 unlock −2ms / draft parity suite / proj-mma −3.5ms）**都对着它定量**的参考线 | `verify-amortization-lesion-audit.md` §10.10（commit `6937d77`）；AGENTS/ledger 状态行。换程序 ⇒ 这条线**与全部 accept 数据作废**，且不是"重跑一次"，是"哪条退化边界被移动"的**再调查**——**S1 级别的排查成本** |
| W5 | **质量面**：draft/EAGER 文本会变 | 需要 `swapab_parity.rs` 式的**文本指纹**基线重建（四 prompt 输出 / 数字位置 / DIFF_EAGER） |

### 2.3 (b′) **也**不能统一整条判定路径（诚实边界）

判定路径上不只有 fp8 投影：**MoE routed experts 是 fp4/mxf4**（另一套程序，`dsv41_experts_mxf4.cu`），
**head/logits** 也不在 fp8 投影族内。所以 (b′) 只把"fp8 投影族"统一，剩下的是
"**未知混合**（哪些同源、哪些不同源）"——accept 结果**仍只能双门禁实测**。

⇒ **(b′) 买的是"一个 maybe"，不是"一个证明"。** 而它的价格是 §2.2 的 W1–W5。

---

## 3. 收益 vs 成本

### 3.1 收益（**全部是估算**，无实测）

| 项 | 值 | 出处 |
|---|---|---|
| 票面 | verify **24.5 → 21.0ms**（**−3.5ms**），区间 −2.5…−4.5 | 设计 §4（**估算**：E1 M 折叠变免费 −2.5 / E2 单行成本坍缩 −0.3 / E3 (b′) 下 m=1 同源 −0.7） |
| ledger 口径 | step ~28.6ms @ acc 2.24 ⇒ ~104 tok/s；−3.5ms ⇒ **~118 tok/s（+14%）** | AGENTS/ledger 状态行 |
| **性能实测证据** | **零** | ks=1 形态网格仅 320 warp（wo_b）/ 更少（小 n），**不构成**性能结论（手册 §2.3） |

+14% 的**估算**收益**是重大**的——所以判死必须靠成本，不能靠"收益小"。

### 3.2 成本（分两类，别混）

**(i) 工程成本：便宜，不是判死理由。**
第二轮 scratch widening 的实测账（本轮勘察所得，供未来复活时直接用）：

| 形状 | n × k | tiles | ks（自动规则） | kc | scratch `[ks][M][n]` f32（M=6） |
|---|---|---:|---:|---:|---:|
| `wo_b` | 5120 × 1024 | 320 | 2–4 | 512 | **245–491 KB** |
| `wq_b` | 4096 × 1280 | 256 | 4 | 320 | 393 KB |
| `wq_a` | 1280 × 5120 | 80 | 8 | 640 | 245 KB |
| **`wkv`** | **512 × 5120** | **32** | **32** | **160** | **393 KB** |
| `sh w1/w3` | 288 × 5120 | 18 | 32 | 160 | 221 KB |

- scratch 上界 **< 0.5 MB**（可 pool / lazy grow，`Device` 级）。
- ks>1 的**结构性代价**：partial 写 + 归约读 = `ks·M·n·4B`/call（wkv 393 KB vs 权重 2.62 MB ⇒
  **+15%**；wo_b 245 KB vs 5.24 MB ⇒ +4.7%），外加每 tile 的到达票 `atomicAdd`。
  ⇒ 这条路**自己付回一部分** E1 的摊薄收益（设计 §4 的 E1 是最值钱的那一项）。
- 附一条**本轮发现**（未来复活要绕）：Rust 侧**没有** SM 计数查询
  （`grep MultiProcessorCount|cudaDeviceGetAttribute` 在 `crates/**/*.rs` **零命中**）。
  而 ks 规则吃 `sms`。所以"复制公式"**不是免费**的——必须**导出** C 侧规则
  （`extern "C" dsv41_proj_mma_ks_for(n,k)`，内部带 `proj_mma_sm_count()` 与 `DSV41_PROJ_MMA_KS` 覆盖）
  才能在 Rust 侧读到**同一个** ks，避免 sizing 与 launcher 解析出的 ks 漂移（漂移 = 越界）。

**(ii) 判死的那一项成本：数值基线重置。** §2.2 的 W1–W5：
m=1 全家族改路由（或丢融合）+ 全局 ks 固定 + `mean-k 2.240` 参考线与全部 accept 数据作废 +
文本指纹基线重建 + S1 级再调查。**这是"换程序"的价格，不是"优化"的价格。**

---

## 4. 判停 + 树内动作清单

### 4.1 判决

- **PROJ_MMA 判死（REJECT）**，加入 rejection list（与 MPAR ×2 / wo_a nwarps / p3lite+ALIGN /
  GROUPED-without-TCGEN05 / COMP·ENGRAM 同列）。
- **第二轮 scratch widening：不做**（`[ks][M][n]` partial + `ctr` 的 Rust sizing **不落地**，
  C 侧 `dsv41_proj_mma_skel.cu` 的 ks>1 分支保持在**已写好但不被喂**的状态）。

### 4.2 保留什么（零运行时成本）

| 保留 | 理由 |
|---|---|
| 第一轮接线（Rust 三分支 + C 入口 + TU 在 `build.sh`） | gate **默认 OFF**：`proj_mma()` 恒 false ⇒ `proj_mows` 走 `dsv41_gemm_fp8_mrows`；即便 arm，C 侧 `g_proj_mma=0` 也是**立即 return 2** ⇒ **逐字节 legacy、零运行时成本**。拒收路线在树内的惯例就是"kernel 留存 + gate OFF"（MPAR 同例） |
| `partial = null` / `ctr = null` 的传参形态 | 它**正是"ks>1 自动 decline"的安全阀**（C 侧 `if (ks>1 && (partial==nullptr||ctr==nullptr)) return 2`）。**不要**"顺手清理"——去掉它 = 引入越界风险而零收益 |
| `DSV41_PROJ_MMA` / `DSV41_PROJ_MMA_KS` | 降级为**诊断门**（micro bench 直调仍可用），**不再**作为性能臂 |
| 第一轮手册 `proj-mma-gpu-verification.md` | 保留为证据链（它的 §2.3 / §6.2 / §11 已预告本轮结论） |

### 4.3 本轮**零代码改动**

`device.rs` / `chain_dev.rs` / `dsv41_kernels.cu` / `build.sh` **一行未动**，
工作树停在 `33b75a7`（干净）。理由是 4.1 的"不做"：第二轮的 C 侧语义已在骨架里写全，
Rust 侧的价值**完全**依附于 (b′)，而 (b′) 已判死 ⇒ 落地即是**死代码 + 新越界面**。

---

## 5. 复活条件（什么会改变判决）

按杠杆从大到小：

1. **验收口径不再依赖"draft 与 verify 同源"**（例如非投机/纯吞吐 workload，或 verification
   换成容忍数值差的统计判据）。这一条一成立，(a) 形态立刻可用（scratch 加宽是**已设计好**的）。
2. **出现逐位可达的 tensor-core 投影路径**。注意 §2.2 证的是 **fp8 `mma.m16n8k32`** 的块内
   求和顺序不可指定；若未来换到"可指定结合顺序"的指令族或全精度累加路径，前提变了要重判。
   （**bf16 投影不算**：那是另一条量化路线，不在本设计内。）
3. **(b′) 的工程面被别的原因顺带铺平**（例如 m=1 侧因其他理由已经全族走 mma）——
   那时边际成本才可能掉到"值得一试"。
4. **低成本诊断（可做但不必做）**：micro bench 跑手册 §7 的**内部契约**
   （`mma<M>` row r ≡ `mma<1>` row r，逐位）+ 与 SIMT 的**容差分布**。
   结果二选一：容差 ~1e-7 而 accept 崩 99% ⇒ 崩的机制比 near-tie 更结构性；
   容差 ≫1e-7 ⇒ kernel 侧有实现偏差。**两种都不改变判停**（(a) 必崩、(b′) 成本照旧），
   所以只在"要把第一轮的 −99% 归因清楚"时才花这一刀。

---

## 6. 证据链

| 文件 | 关系 |
|---|---|
| `docs/agent/tensorcore-proj-design.md` | §2.2 逐位不可能（证明）、§2.4.1 (b′) 定义、§2.4.2 代价、§3.4 scratch、§4 −3.5ms 估算 |
| `docs/agent/proj-mma-gpu-verification.md` | 第一轮手册：§2.2 回执纪律、§2.3 为何 ks=1、§5/§6 双门禁、§6.2 AP′ ≠ (b′)、§7 内部契约（未跑）、§11 范围外 |
| `kernels/cuda/dsv41_proj_mma_skel.cu` | C 入口：`proj_mma_ks_for(n,k,sms,override)`（只吃 n,k）、`ks>1 && null ⇒ return 2`、`partial[ks][M][n]` 槽位、`ctr[n/16]` 自复位 |
| `crates/ferrite-models/src/dsv41/device.rs:694` | 14 参 ABI（含 partial/ctr/pmma_n），第一轮传 null |
| `crates/ferrite-models/src/dsv41/chain_dev.rs:5405` | `proj_mrows` 三分支 ① MMA > ② MPAR > ③ legacy |
| `docs/agent/verify-amortization-lesion-audit.md` §10.10 | `mean-k 2.240` 参考线（S1 fix 后，commit `6937d77`） |

---

*判决：REJECT。保留 gate-OFF 接线（零成本），不做 scratch widening，PROJ_MMA 入 rejection list。*
