# verify（5/6 行批处理 forward）计算下限账本（户部 · 资源与性能）

> 目标口径：400 tok/s ⇒ accept 3 ⇒ 步时 ≤7.5ms ⇒ **verify ≤5.5ms**。现状 **37ms**（`DSV41_TIMING`，m=5）。
> 方法：**只读代码 + shape 精确推导**（`config.rs::production()` 的真实参数 + `weights.rs::local_shape` 的 TP8 切片规则
> + `chain_dev.rs` 的逐调用点）+ 仓库内**实测每核单价**（`STATUS.md`）折算。
> 本机无 GPU ⇒ 不跑 nsys；所有时间数标了来源（`实测` / `推算`）。凡未实测支撑的项在 §7 列「待验证」。
>
> 户部 · 2026-09-12 · base commit `bacbfd4`

---

## 【分析范围】

| 文件 | 用途 |
|---|---|
| `crates/ferrite-models/configs/dsv41_flash.json` | production 真实参数（本账本第一手数据） |
| `crates/ferrite-models/src/dsv41/config.rs` | `production()` / `compress_ratios` / `moe_config` / dtype |
| `crates/ferrite-models/src/dsv41/weights.rs` | 张量表 + `Shard` 规则 + `local_shape`（每卡切片）+ `K_ATOM=64` |
| `crates/ferrite-models/src/dsv41/chain_dev.rs` | `step_rows` / `layer_rows` / `attention_rows` / `moe_rows` / `indexer_rows_one` / `compress_proj_rows` / `engram_apply_rows` + 各 env 默认值 |
| `crates/ferrite-models/src/dsv41/load.rs` | `KEEP_BF16` 与「bf16→f32 加宽」规则（决定实际 dtype） |
| `kernels/cuda/dsv41_kernels.cu` | `sparse_attn` 的 split+merge 默认、`gemm_fp8_*` |
| `crates/ferrite-dsv41/STATUS.md` | 实测单价（唯一的时间来源） |

---

## 1. 口径（从 config 读出，非记忆）

```
n_layers 40 · dim 5120 · n_heads 64 · head_dim 512 · rope_head_dim 64
q_lora 1280 · o_lora 1024 · o_groups 8 (hpg=8) · hc_mult 4
window 128 · index_topk 512 · index_n_heads 32 · index_head_dim 128
n_routed 384 · topk 6 · n_shared 1 · moe_inter 2304 · vocab 129280
compress_ratios = [0,0] + [2]*18 + [1]*20   → 38 层有 compressed ring，2 层 window-only
kv_source_* = {2,8,14,20} · index_source = {2,8,14,20,24,28,32,36} · engram = {1,14}
dtype: dense = fp8(e4m3, 32x32 ue8m0) · experts = fp4(e2m1, 2 值/字节) · head/embed/indexer = bf16 · compressor = f32
TP8 (world=8): 每卡 nlh = 8 头 · nlg = 1 组 · experts 按 inter 切 (无 EP，每卡都留 384 个专家) · 共享专家 SHARED_TP 默认 ON
K_ATOM = 64  ⇒ inter_local = padded(2304/8 = 288) = 320      ← 每专家字节比 288 的写法多 11%
```

**关键 dtype 事实（`load.rs`）**：`KEEP_BF16` 只有 4 个名字（`ffn.gate`、`indexer.wq_b/wk/weights_proj`）；
`head.weight` 按名字特判保留 bf16；**其余所有 bf16 权重在加载期无损加宽为 f32** —— 所以 compressor 的
`comp_wkv/comp_wgate`（512×5120）在显存里是 **f32 = 10.49MB/个**，不是 2.6MB。

---

## 2. 账本总表（m = 5 行，每卡/每 verify 步）

字节来源：`weights.rs::tensor_specs` × `local_shape`，fp8 = 1B/元素 + scale(o·k/1024)，
fp4 = 0.5B/元素 + scale(o·k/32)，bf16 = 2B，f32 = 4B。
「现状 MB」= 按**代码今天的调用形态**实际流过的字节；「折叠 MB」= 权重每层只读一次的极限。

| # | 项 | shape 推导（每卡） | 现状 MB | 折叠 MB | ms@7TB/s<br>(折叠后) | 现状 ms | 达成带宽 | 现状/实现下限 | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **MoE 路由专家**<br>gate+up+down | 40L × (5×6=30 assign) × **2.6112MB**<br>w1=w3=320×2560B+320×160B scale=0.781MB<br>w2=5120×160B+5120×10B=0.870MB | **3133.4** | 3133.4 | **0.448** | **8.30** | 378 GB/s | 1.0× | ⚠️ 字节不可压 |
| 2 | **MoE 共享专家**<br>w1/w3/w2 | 40L × 3 × (288×5120B fp8)=1.4746MB<br>→ 1× = 177.1MB；`SH_EXP_MROWS` 默认 **OFF** ⇒ ×5 | 885.3 | **177.1** | **0.025** | **10.40** | **85 GB/s** | **5.0×** | 🔴 **最大肉** |
| 3 | **投影族**<br>wq_a/wkv/wq_b/wo_a/wo_b | 40L × 23.878MB（wq_a 6.554 + wkv 2.621 + wq_b 5.243 + wo_a 4.194 + wo_b 5.243，均已含 scale）<br>TP8 后：wq_a/wkv 复制、wq_b/8、wo_a/8、wo_b/8 | 955.1 | 955.1 | 0.136 | 3.70 | 258 GB/s | 3.7× | 🟡 已折叠，肉在固定项 |
| 4 | compressor 投影<br>(4 源层) | 3×(wkv+wgate f32 512×5120×4B=10.49MB×2) + 1×(wkv)=10.49MB = 73.4MB；`compress_proj_rows` 逐行 | 367.0 | 73.4 | 0.010 | 0.55 | 667 GB/s | 2.8× | ⚪ 小肉 |
| 5 | **indexer 权重**<br>(8 层) | 8×(wq_b 4096×1280×2B=10.486MB + wp 32×5120×2B=0.328MB) + 4×(wk 128×512×2B=0.131MB) = 87.0MB；`indexer_rows_one` 逐行 | 435.2 | 87.0 | 0.012 | 2.50 | 174 GB/s | 6.3× | 🟠 有肉 |
| 6 | **attention KV 读**<br>(DSA ring, f32) | 每行每层 = (win + min(clen,512)) 行 × 512 × 4B；P≥2k 后 topk 封顶 ⇒ 2L×128 + 38L×640 = 24576 行/行·全层<br>+ index_k 读 5×8×min(clen,512)×128×4B = 10.5MB | 262.1 | 262.1 | 0.037 | ~2.80 | ~94 GB/s | ~4× | 🟠 有肉 |
| 7 | **head**<br>[vocab,dim] bf16 | 129280×5120×2B = **1323.8MB**（`Shard::Replicated`，每卡全量）<br>`VERIFY_HEAD_FOLD` 默认 **0** ⇒ 逐行 ×5 | **6619.1** | **992**<br>(仅切分，见 §补充 ⑤) | 0.024 | ~1.12 | — | **6.7×** | 🟢 **切分已实施**；折叠（165.5MB，再 6×）待 K 序 parity |
| 8 | engram (2 层) | 2×(wkv 25600×6144 fp8 = 157.3MB + scale) + gather 24×256B×5行 | 315.0 | 315.0 | 0.045 | 0.42 | 750 GB/s | 2.0× | ⚪ 无肉 |
| 9 | **MoE gate**<br>(router) bf16 | 40L × 384×5120×2B = 3.93MB/L；`ROW_FOLD_GATE` 默认 **OFF** ⇒ 逐行 ×5 | 786.4 | 157.3 | 0.022 | **3.44** | 229 GB/s | **5.0×** | 🟠 **有肉** |
| 10 | hc 权重 f32 | 40L × 2 × (24×20480×4B = 1.966MB) = 157.3MB（`hc_mixes` 已带 rows=m） | 157.3 | 157.3 | 0.022 | 2.96 | **53 GB/s** | 3.0× | 🟡 有肉 |
| 11 | 激活/状态 | h_r(m×4×5120×4B) 8 遍 + xn/xq/ex_act/moe_out + logits_r(2.6MB 写+读) | 176.6 | 176.6 | 0.025 | 0.20 | 885 GB/s | 1.1× | ⚪ 无肉 |
| 12 | all-reduce<br>(attn + MoE) | 80 次 × 3 核（store/reduce/stamp），payload 小 | — | — | — | 1.40 | — | — | ⚪ 协议地板 |
| | **合计** | | **14.09 GB** | **5.66 GB** | **0.809** | **39.6** | 381 GB/s | | |

**模型自校准**：上表「现状 ms」合计 **39.6ms**，实测 **37ms**（+7%）⇒ 分解可信（各项内部时间取自
`STATUS.md` 的实测每核单价，不是本机实测）。

### m=6（spec 步）的系数
逐行项 ×1.2；专家 = 40×36×2.6112MB = **3.76GB**（+0.63GB）。⇒ m=6 的字节 ≈ 16.1GB，模型 ms ≈ 44ms。

---

## 3. 三层下限：账本告诉你真正的约束是什么

| 层 | 定义 | 数值 | 与 5.5ms 的关系 |
|---|---|---|---|
| **T1 硬件带宽下限** | 折叠后字节 ÷ 7TB/s | **0.809 ms**（现状流量 14.09GB ÷ 7TB/s = 2.01ms） | **5.5ms 的 15%** ⇒ **字节根本不是约束** |
| **T2 同效带宽下限** | 折叠后字节 ÷ **今天达成的** 381 GB/s | **14.9 ms** | 目标的 **2.7×** ⇒ 只折叠权重 = 37→15ms，仍差 3 倍 |
| **T3 达标所需带宽** | 5.66GB ÷ 5.5ms | 需 **1029 GB/s**（今天的 2.7×） | 专家的 3.133GB 单独就需 **570 GB/s** |

**核心结论**：verify 的瓶颈**不是字节数，是「达成的有效带宽」**。
14.09GB / 37ms = **381 GB/s = HBM 峰值(7.672TB/s) 的 5.0%**。
这与主链的独立测量完全一致（`STATUS:7870` gemv 有效带宽 373GB/s = 峰值 4.9%）——
**verify 与主链在同一堵墙上：小 N、低占用的 GEMV 族只跑到峰值的 5%。**

一个更强的旁证：**head 是唯一跑到带宽的项**（4.44 TB/s，N=129280 的大 GEMV），
而它恰好是字节数最大、也最容易被砍 8~25× 的项。

---

## 4. 逐项差距分析

### 4.1 在下限 2× 以内（无肉，别再动）
| 项 | 现状/下限 | 理由 |
|---|---|---|
| all-reduce（1.40ms） | 1.0× | NVLink 协议地板（17.3µs/次已实测为地板），只能减**次数** |
| 激活/状态（0.20ms） | 1.1× | 已被各核融合吃掉 |
| **路由专家（8.30ms）** | 1.0× | **字节不可压**（30 个 assignment 选 29 个唯一专家，去重只省 3%）且该核已在其有效带宽上。**肉只在「核效率」，不在「调度」** |
| engram（0.42ms） | 2.0× | 2 层，已 rows=m |

### 4.2 超下限最多的项（有肉）
| 排名 | 项 | 现状 → 下限 | 差距 | 差距的**性质** |
|---|---|---|---|---|
| 1 | **共享专家** | 10.40 → 2.10 | **5.0×** | **纯重复读**：同一专家 ×5 行读了 5 遍（885MB vs 177MB），且该核本身只有 85 GB/s（实测 72-85） |
| 2 | **head** | 1.49 → 0.05 | **30×** | 逐行 ×5 遍 + **未切分**（每卡 1.32GB 全量）；切分后只需 165MB |
| 3 | **indexer** | 2.50 → 0.40 | **6.3×** | 逐行 lin（×5）+ `indexer_topk` **50.1µs/次（21 GB/s）** 的单核地板 |
| 4 | **MoE gate** | 3.44 → 0.69 | **5.0×** | 逐行 ×5（`ROW_FOLD_GATE` 默认 OFF，已有 mrows 核） |
| 5 | attention KV / sparse | ~2.80 → 1.10 | ~2.5× | 逐行发起（`b*m` 已支持！grid 是 `(b*m, h)`）+ 8-block 低占用 |
| 6 | hc 链 | 2.96 → 1.00 | 3.0× | 10 发/层（已 rows=m），**字节 157MB 却跑 53 GB/s** ⇒ 纯 launch/占用 |
| 7 | 投影族 | 3.70 → 1.00 | 3.7× | 已折叠（权重读一次），肉在 **15.5µs/发的固定项 × 200 发** |
| 8 | compressor | 0.55 → 0.20 | 2.8× | 36 发 f32 小 GEMM，字节 73MB 本可忽略 |

---

## 5. 结论：5.5ms 可达性的硬门槛

1. **字节不是门槛**：折叠后 5.66GB，@7TB/s = 0.81ms（目标的 15%）。
2. **全折叠 + 保现核 ≈ 16.6ms**（§2 表「实现下限」列合计）——其中
   **路由专家 8.30 + 协议/launch 地板（AR 1.40 + hc 1.00 + 激活 0.18）** 就占了 10.9ms，**已是 5.5ms 的 2 倍**。
3. ⇒ **5.5ms 必须同时满足两条**（缺一不可）：
   - **(a) 折叠到「权重每层读一次」**（14.09GB → 5.66GB，−60% 流量）；
   - **(b) 有效带宽从 381 GB/s 提到 ≥1029 GB/s**（2.7×），其中**专家核必须 ≥570 GB/s**（今天 286-443 GB/s）。
4. 换句话说：**只做多行化（launch/流量削减）到不了 5.5ms（落点 ~16ms）；只做核效率不折叠也到不了（落点 ~26ms）。**
   这与「launch 削减收益有限」的判断一致，但把「瓶颈在带宽/计算」进一步收敛为：
   **瓶颈是「小 N GEMV 族的达成带宽」**，而这一族在 verify 里占 12.09GB / 14.09GB = **86% 的流量**。

---

## 6. 三个最有肉的优化方向（按 预期ms × 把握 排序）

| # | 方向 | 落点（代码级） | 预期 | 把握 | 加权 |
|---|---|---|---|---|---|
| **①** | **共享专家多行化**（`DSV41_SH_EXP_MROWS=1`，**代码已提交、默认 OFF**）<br>`shared_expert_mrows`：`quant_rows` + 两次 `gemm_fp8_mrows`(w1/w3, out_stride=2·sh_il) + `swiglu_limit_q(rows=m)` + `gemm_fp8_mrows`(w2) + 一次 `add_inplace` | `chain_dev.rs:7444` / `:7551` | **10.40 → 2.10ms（−8.3ms）**<br>字节 885→177MB（−0.89GB/步 已由 commit 93c4439 计算） | **高**：代码在、bit-identity 逐调用点已论证、只差默认值 + 一次 A/B | **7.5** |
| **②** | **MoE gate 行折叠**（`DSV41_ROW_FOLD_GATE=1`）+ **indexer 多行**<br>`gemv_bf16_mrows(nrows=m)` 已是现成核（`ferrite_gemv_bf16_nt`，与 v2 逐位同序）；`indexer_rows_one` 的 `lin` 改 `proj_mrows` | `chain_dev.rs:7188` / `:6826` | **3.44 → 0.69ms（−2.75ms）**<br>+ indexer −0.6ms | **高**（gate，flag 已存在）/<br>中（indexer） | **2.3** |
| **③** | **sparse_attn 的 `b·m` 单发**<br>launcher 已支持 `grid=(b*m, h)` 且 split 默认 ON；把行循环折成 `b=1, m=5` 一发（split+merge 各一）⇒ 400 核 → 80 核；**前置**：行内 `clen_rows_r` 快照（因为每行的 `*clen` 不同——这正是 AGENTS.md 里 B1 的因果修复点） | `chain_dev.rs:6482` / C launcher `dsv41_sparse_attn` | **~2.80 → 1.10ms（−1.7~2.1ms）** | 中（依赖 B1 的行本地 clen） | **2.2** |
| 4 | 路由专家的**核效率**（cp.async/TMA/tcgen05；现状 80% issue 停等在 LUT smem gather） | `dsv41_experts_mxf4.cu` | **8.30 → 1.5~2.5ms（−5.8~6.8ms）** | **低-中**（历史 5 次尝试失败，M=128 mxf4 实测 16.8GB/s） | 2.1 |
| 5 | ~~**head 折叠 + 词表切分**（`VERIFY_HEAD_FOLD=1` + 多行跨卡 argmax，TODO#3）~~<br>→ **已实施：词表切分**（`DSV41_VERIFY_HEAD_SLICED=1` 默认 ON，逐行 `gemv_bf16(n=seg)` + `dsv41_argmax_sliced_rows` 一次 v5 round 批 m 行）；**折叠仍未做**（mrows 是 v2 核、与 eager 的 v1 核 K 序有差 ⇒ 数值改动，`FOLD` 保持默认 OFF） | `chain_dev.rs::verify_head_geom` / `kernels/cuda/dsv41_kernels.cu::dsv41_argmax_sliced_rows` | **1.49 → ~0.19ms（−1.3ms）**；折叠的额外 6×（→0.05ms）**未拿** | **切分：高**（每行与 eager 同一 v1 核，逐位同源；跨卡一次 round）<br>折叠：低（需先证 K 序 parity） | — |
| 6 | hc 链融合（`hc_mixes` 尾部 / collapse / post 并核） | `dsv41_glue.cu` | **2.96 → 1.00ms** | 中 | 1.0 |

**补充说明（诚实排序）**：若按「把握×收益」严格排，`⑤ head 折叠+切分` 的加权（1.4）低于 ①②③，
但它是**唯一「改两个 env + 一个已存在的 argmax」就能拿 −1.4ms** 的项（单位工作量收益最高）；
`④ 专家核效率` 是**结构性前提**——不做它，①②③⑥ 全做完也只是 39.6 → ~18ms，仍到不了 5.5ms。

---

## 7. 待验证（无实测支撑，必须 nsys 复核）

| # | 断言 | 为什么需要验证 |
|---|---|---|
| V1 | 投影族现状 = 3.70ms（mrows 核 `15.5µs 固定项 + bytes/1.5TB/s`） | 「15.5µs 固定项」是 **M=1** GEMV 的实测拟合；`gemm_fp8_mrows`（M=5 weight-stationary）的固定项未实测 ⇒ 真值区间 **1.6~3.7ms** |
| V2 | 路由专家现状 = 8.30ms | 用的是 rows=1 时的 443/286 GB/s；`grid.z=rows=5` 后 CTA 数 ×5，**达成带宽可能上移**（若到 1.3TB/s ⇒ 2.4ms）。这是全表**最大不确定项**（±6ms） |
| V3 | attention = ~2.80ms | `sparse_attn_split + merge`（split_c=4）在 verify 的每行耗时未实测；10.5µs 是旧单块 `sparse_attn_pf` 的单价 |
| V4 | 共享专家现状 = 10.40ms | 用的是隔离微基准 `fp8 sharedexp n=256 = 17.3µs`；当前 sh_il=288、且 `SH_EXP_MX2` 默认 ON（w1/w3 合一次）⇒ 实际 2 次 GEMM/行而非 3 次 |
| V5 | 激活/状态 176.6MB | 估算（h_r 按 8 遍计），非精确计数 |
| V6 | 专家 per-rank 字节用 `padded(288)=320` | 若某路径按 288 计，每专家 2.36MB（−10%），专家项 3.13→2.82GB |

**建议的一次性验证实验**（一条 nsys，不用改码）：
`DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_TIMING=1` + nsys profile，按 kernel 名聚合 verify 段，
重点看 4 个数：`expert_gateup/down`（V2）、`gemm_fp8_mrows`（V1）、`sparse_attn_split+merge`（V3）、
`gemv_bf16_v2`（gate，验证 200 发假设）。这 4 个数落定后，本账本的 ±7% 模型即可收敛到 ±2%。

---

*户部 · 只读分析，未执行任何 GPU 命令、未改动任何代码（本文件为唯一产出）。*

---

# 更新（第二轮）— head 切分 / o-rope 融合 / a32 回退落地后

> 户部 · 2026-09-12 · HEAD `53bae56`（a32 gate `28515b6` 10:08 · orope `8f18105` 10:18 · head slice `109f87c` 10:24）
> 方法同第一轮：只读代码 + shape 精确推导 + 仓库内实测单价折算。**本机无 GPU，未跑 nsys。**
> 本轮第三项（a32）是**唯一一个让账本变差**的变化，所以先给判定。

## 0. 一句话结论

**新预期基线 ≈ 40 ~ 45ms（m=6），不是 37ms 以下。**
head 的 −1.3~−1.5ms **被 a32 回退吃掉了**——而且不止：回退同时把 `proj_mrows` 的
**weight-stationary 收益**（`AGENTS.md:94` 自己记的「多行化的真实收益 = 权重读一次（~3ms）」）还了回去。
关键判定（§3）：**即使 verify 图化，a32 回退仍净亏 ~3.9ms > head 的 −1.3ms**。
到 5.5ms 的最新缺口：**~34.7 ~ 39.1ms（6.3~8.1×）**。

---

## 1. 三个已落地变化：代码级核对（先用代码钉住口径）

| 变化 | 落点 | 代码事实 | 口径修正 |
|---|---|---|---|
| **head 词表切分** | `chain_dev.rs::verify_head_geom` + `DSV41_VERIFY_HEAD_SLICED`（**默认 ON**，`verify_head_sliced()→unwrap_or(true)`）｜`device.rs:2622 argmax_sliced_rows` | 每 rank 只投 `seg = vocab/world = 129280/8 = 16160` 行 → **165.5MB/rank/行**；每行仍是**同一个 v1 核**（`gemv_bf16`，n=seg=16160 仍 >2048 所以不走 v2）⇒ 与 eager 逐位同源；argmax 是 `dsv41_argmax_sliced_rows` **一次 v5 round 批 m 行**（不是 m 次 `argmax_sliced`） | ⚠️ **上一版 6619MB 是 m=5（5×1323.8），本轮 992MB 是 m=6（6×165.5）——两者不可直接相减。** 同 m=6 归一：**7943 → 993MB（−6.95GB，−87.5%）** |
| **o-rope 融合（P1v）** | `8f18105`｜`DSV41_VERIFY_OROPE`（默认 ON）｜`sparse_attn_orope` 调用点在**行循环内**（`chain_dev.rs:6820`） | 每行每层：`sparse_attn` + `apply_rope` + `quant_fp8` 三个 launch → **1 个**（融合核的 phase 2 旋转 + phase 3 发射 fp8）⇒ **−2 launch/行/层** | 任务口径的「−80 launch/步」是 **eager（m=1）** 的数；verify 是 **−2×m×40 = −480/步（m=6）**。ms 上很小（~0.2ms），**但它是正确性对齐**（两臂同核），不是优化项 |
| **a32 gate** | `28515b6`：`dsv41_kernels.cu:4985 if (g_gemv_a32) return 2;`（`g_gemv_a32` 默认 **true**）｜`chain_dev.rs::proj_mrows` 把 `Ok(false)` 当「未执行」 | **decline 在 C 端，Rust 侧 `mrows` 变量看不到它**——见 §2 | 见下 |

---

## 2. Q3 的精确核算：decline 之后走的是什么循环

### 2.1 调用侧（这是判定成立的关键）

```rust
// chain_dev.rs:6497  —— 只看 .so 符号 / swapab / 行数，**不看 a32**
let mrows = self.dev.supports_gemm_fp8_mrows() && !Self::swapab() && m <= VERIFY_ROWS;
let took_akv = if mrows {
    self.quant_rows(xn_r, m, dim)?;           // ① 先发射 staging
    let ok_a  = self.proj_mrows(wq_a, ...)?;  // ② C 端 return 2 ⇒ Ok(false)
    let ok_kv = self.proj_mrows(wkv,  ...)?;
    ok_a && ok_kv                             // ③ false
} else { false };
if !took_akv { for r in 0..m { self.lin(wq_a); self.lin(wkv); } }  // ④ 落到逐行
```

`lin()` = **`quant1` + `gemm_fp8_mx(m=1)`**（`chain_dev.rs:2625`）——即 **M=1 SIMT GEMV**，也就是 EAGER 跑的那个程序。
**要点：① 是纯粹浪费——staging launch 已经发射，decline 后又把同样的量化按行重做一遍。**
`quant1` 的 T1/T2 省略在这里**不生效**（它们按指针门控 `s.xn.ptr` / `s.qr.ptr`，而 verify 的源是 `xn_r + r*dim`，不是同一个 buffer）。

### 2.2 launch 账（每层 / m=6）

| 站点 | mrows 路径 | a32 decline 后 | Δ |
|---|---|---|---|
| `wq_a`+`wkv` | `quant_rows`×1 + gemm×2 = **3** | `quant_rows`×1(废) + 6×(`quant1`+gemm)×2 = **25** | **+22** |
| `wq_b` | `quant_rows`×1 + gemm×1 = **2** | `quant_rows`×1(废) + 6×(`quant1`+gemm) = **13** | **+11** |
| `wo_a`（grouped） | 1（`wo_a_grouped_fp8` **自读 a32 门**，不受影响） | 1 | **0** |
| `wo_b` | pack×6 + gemm×1 = **7** | pack×6(废) + 6×(`quant1`+gemm) = **18** | **+11** |
| **合计** | **13** | **57** | **+44/层 = +1760/步** |

拆开看这 +1760：
- **gemm：+20/层 = +800/步** ← 这一项正好等于 commit 说的「+800 launches/step」——**原话只算了 gemm**；
- **`quant1`：+16/层 = +640/步**（每个 `lin` 自带一次量化）；
- **纯废 launch：+8/层 = +320/步**（wq_a/wkv、wq_b 的 `quant_rows` + wo_b 的 pack 循环，decline 后仍先发射）。

m=5 口径（上一版账本的行数）：**+36/层 = +1440/步**，其中 gemm +640。

⇒ **任务里「800 × 3µs = 2.4ms」低估了 ~2.2×**（真实增量 ~1760 launch），但——**3µs 本身也不是正确的单价**，见下。

### 2.3 「每 launch 小」是不是真的？——不是，代价有两层

**(a) launch 层**（仓库实测地板，来源 `graph_bench` / verify 审计）：

| 口径 | 单价 | 1760 launch | 只算 gemm 800 |
|---|---|---|---|
| 图内 dispatch（`DSV41_VERIFY_GRAPH=1`） | 0.411µs/node | **0.72ms** | 0.33ms |
| 流式 host submit（**今天的默认，verify 未图化**） | 2.904µs/launch | **5.11ms** | 2.32ms |
| + 小核最小执行（verify 自己的口径 3.3µs） | 6.2µs | 10.9ms | 4.96ms |

**(b) 权重重读层（主项，且任务描述里没提）**：
`gemm_fp8_mrows` 是 **weight-stationary**——每个 warp 把**一行权重 stage 进 smem**（`row_s[warp*k] ← w[row*k]`），
然后对 **M 行激活**复用（`for r in 0..M: acc[r] += av*wv`，`dsv41_kernels.cu:4936`）。
逐行 `gemm_fp8_mx(m=1)` 则是**每行把整块权重重读一遍** ⇒ 4 个受影响的投影权重被读 **m 次**：

```
每层受影响权重 = wq_a 6.554 + wkv 2.621 + wq_b 5.243 + wo_b 5.243 = 19.661 MB
× 40 层 × (m−1) = m=6: +3.93 GB/步   (m=5: +3.14 GB/步)
```

**旁证（同一仓库自证）**：`AGENTS.md:94` 写「verify 37ms：**多行化的真实收益 = 权重读一次（~3ms）**」——
3.14GB @ ~1TB/s ≈ 3.1ms，与 3.9GB @ 1TB/s ≈ 3.9ms 同量级。**这就是 a32 回退还给回去的东西。**

---

## 3. 关键判定：会不会吃掉 head 的 −1.3ms？

**会，而且不够。** 按最保守（对 a32 最有利）的假设——verify **已经图化**：

| 分项 | 值 |
|---|---|
| a32 回退 · launch（图化 0.411µs × 1760） | **+0.72 ms** |
| a32 回退 · 权重重读（+3.93GB @ ~1TB/s） | **+3.9 ms** |
| **回退合计** | **+4.6 ms** |
| head 切分收益（m=6：6×298µs → 6×48µs） | **−1.50 ms** |
| o-rope 融合（同核对齐的副产品） | **−0.20 ms** |
| **三变化净效应** | **+2.9 ms** |

⇒ 非图化（今天的默认）时把 launch 换成 5.11ms：**回退合计 +9.0ms，净效应 +7.3ms。**

**结论**：`−1.3ms` 的 head 收益在**两种情形下都被吃光**（最乐观情形下也净亏 ~2.9ms）。
`+800 launch` 之所以看起来"小"，是因为只数了 gemm 一项、且用了 3µs 这个纯发射单价——
**漏掉了 (i) 640 个 `quant1` + 320 个纯废 launch，(ii) weight-stationary 回退带来的 3.9GB 重读。**

### 3.1 一个必须先钉死的前置不确定项（否则上面的判定会翻转）

**37.31ms 那次测量（09:5x）时，mrows 到底有没有生效？**

| 证据方向 | 内容 |
|---|---|
| **支持「生效」**（⇒ 回退 = 新增 +4.6~9ms） | `AGENTS.md:94` 明写「verify 37ms：多行化的真实收益 = 权重读一次（~3ms）」⇒ 37ms 已含 mrows；根因 #6（`quant_rows` 行距 bug）的指纹（行 0 恒对、r≥1 全错）**只能在 mrows 分支出现** |
| **支持「未生效」**（⇒ 回退 = 0，head 的 −1.3ms 保住） | verify 审计（`86b1608`, 09:30）把 verify **38.5ms** 记成 **6232 launch**（其中投影 2000 = **逐行口径**），并把 mrows 列为"产出的削减 2000→~400" |

**一次 A/B 就能定**（正是 a32 提交自己点名的 verification posture）：

```
同一 serve 跑 DSV41_GEMV_A32=1 vs =0，比 [dspark] 的 verify_ms
  A32=0 ⇒ mrows 生效（多行）；A32=1 ⇒ 逐行
判据：Δverify_ms ≥ 1.3ms ⇒ head 收益被吃；≈0 ⇒ 37.31ms 本就是逐行，a32 gate 零变化
```

---

## 4. 更新后的账本

### 4.1 变化项（m=6，每卡/每步）

| 项 | 上一版 | 本轮 | 字节变化 | ms 变化 |
|---|---|---|---|---|
| **head**（词表切分） | 7943MB / 1.79ms（未切） | **993MB / 0.29ms** | **−6.95 GB** | **−1.50 ms** |
| **o-rope**（P1v 融合） | per-row rope+quant（2×m×40 launch） | 1 launch/行/层 | ~0 | **−0.20 ms**（+ −480 launch/步） |
| **投影族**（a32 回退） | 955MB / 3.70ms（mrows） | **4886MB / 8.3~12.7ms** | **+3.93 GB** | **+4.6 ~ +9.0 ms** |
| 三项净 | | | **−3.02 GB** | **+2.9 ~ +7.3 ms** |
| **共享专家（flag OFF）** | 885MB / 10.40ms（m=5） | 1062MB / **12.48ms**（m=6） | — | 仍在桌上，**最大项** |

### 4.2 新的预期基线（锚定测量值，不重算全表）

```
37.31ms（实测，m≈5，mrows ON，未切 head，无 verify-orope）
 −1.50    head 词表切分（m=6 归一：6×298µs → 6×48µs）
 −0.20    o-rope 融合（同核对齐）
 +4.6~9.0 a32 回退（图化 0.72 + 权重 3.9 ／ 非图 5.11 + 权重 3.9）
 ─────────
 = 40.2 ~ 44.6 ms   ← 新的预期基线（m=6；若 §3.1 判为"未生效"则 35.6ms）
```

**流量侧的现状（m=6）**：16.1GB（上一版 m=6 归一）− 6.95（head）+ 3.93（a32）= **≈13.1 GB/步**
⇒ 达成带宽 ≈ 13.1GB / 40ms ≈ **327 GB/s**（比第一轮的 381 更低）。
**注意方向：head 切分省的是"已经跑到 4.44TB/s 的字节"，a32 回退加的是"每层只跑到 ~1TB/s 的权重重读流量"——
所以字节净减 3GB，时间反而净增 ~3ms。**

---

## 5. 到 5.5ms 的最新路径（按 把握 × 收益 排序）

| # | 方向 | 落点 | 预期 | 把握 | 加权 |
|---|---|---|---|---|---|
| **1** | **共享专家 mrows**（`DSV41_SH_EXP_MROWS=1`，代码已就位 93c4439，**默认 OFF**）<br>`shared_expert_mrows`：`quant_rows` + `gemm_fp8_mrows`×2(w1/w3) + `swiglu_limit_q(rows=m)` + `gemm_fp8_mrows`(w2) + `add_inplace` | `chain_dev.rs:8018/8026/8038/8068` | **12.48 → 2.52ms（−9.96ms）**；字节 1062→212MB | **高**（只差默认值 + 一次 A/B） | **9.0** |
| **2** | **verify 图化**（`DSV41_VERIFY_GRAPH=1`，A/B 脚本现成 `scripts/verify_graph_ab.sh`） | `verify_graph_gate` | launch submit 2.904→**0.411µs/node** ⇒（3000~4600 launch）**−3~5ms**；**并把 a32 回退的 launch 代价从 5.11 压到 0.72ms** | 中高（图捕获条件已就位；`eng_host/stats_dbg/phase_dbg` 会拒绝） | **3.0** |
| **3** | **MoE gate 行折叠**（`DSV41_ROW_FOLD_GATE=1`）+ **indexer 多行**<br>复用现成 `gemv_bf16_mrows` / `proj_mrows` | `chain_dev.rs:7188` / `:6826` | **4.13 → 0.83ms（−3.3）** + indexer **3.00 → 2.30（−0.7）** | 高（gate flag 已存在）/ 中 | **2.6** |
| **4** | **a32 Direction B**：给 mrows 核补 a32（物化 `s_af`）变体，让两个臂再次同表达式 | `dsv41_kernels.cu:4985`（删掉 decline） | **投影 8.3~12.7 → 3.7ms（拿回 4.6~9.0ms）** | **中**（是 §3 的止损项，非新增收益） | **2.3** |
| 5 | 路由专家**核效率**（`dsv41_experts_mxf4.cu`：cp.async/TMA/tcgen05；现状 80% issue 停等在 LUT smem gather） | 同 | **9.96 → 2~3ms（−7~8ms）** | **低-中**（历史 5 次失败；M=128 mxf4 实测 16.8GB/s） | 2.0 |
| 6 | hc 链融合 + compressor 多行 + engram | `dsv41_glue.cu` | 3.55 → 1.5（−2.0）+ 0.66→0.4 | 中 | 1.2 |
| 7 | head **折叠**（K 序 parity 后再谈，`VERIFY_HEAD_FOLD` 保持 OFF） | `head_gemv_bf16_mrows` | 0.29 → 0.05（−0.24） | 低 | 0.3 |

**1~4 全做完**：40.2~44.6 → **~18~20ms**。**仍然差 3.5×**——这与第一轮的结论一致：
**字节/launch 都不是门槛，门槛是「小 N GEMV 族的达成带宽」**（第一轮 T1/T2/T3 不变）：

| 层 | 定义 | 数值 | 与 5.5ms |
|---|---|---|---|
| T1 | 折叠后字节 ÷ 7TB/s | 5.66GB ÷ 7TB/s = **0.81 ms** | 目标的 15%（**a32 回退不改变折叠目标，只抬高现状流量**） |
| T2 | 折叠后字节 ÷ 今天达成的 327~381 GB/s | **14.9 ~ 17.3 ms** | 目标的 **2.7~3.1×** |
| T3 | 5.66GB ÷ 5.5ms | 需 **≥1029 GB/s**（今天的 2.7~3.1×） | 专家 3.76GB 单独就要 **684 GB/s** |

⇒ **a32 回退把 T2 又推远了 ~0.4ms，并且把"多行化"这条已经到手的路退回去了一半。**
最新路径的两条硬腿没变：**(a) 折叠到权重每层读一次；(b) 有效带宽 ≥1029 GB/s（专家核 ≥684 GB/s）。**

---

## 6. 待验证（第二轮新增；前三项任一落定都能把上面的区间收窄一半）

| # | 断言 | 为什么必须验证 | 建议实验 |
|---|---|---|---|
| **V7** | 37.31ms 基线**已含 mrows** | 决定 a32 回退是 **+4.6~9ms** 还是 **0**（§3.1 两张证据冲突） | 同 serve `DSV41_GEMV_A32=1` vs `=0`，比 `verify_ms` |
| **V8** | 逐行 `gemm_fp8_mx` 的权重重读**未被 L2 吸收**（⇒ ~1TB/s 有效带宽） | 若被 L2 吸收，权重项 <<3.9ms，§3 的判定会**翻转成"没有吃掉"** | nsys 看 `gemm_fp8_gemv` 的 dram__bytes 是否 ~m× 权重；或 A32=0/1 的 Δ |
| **V9** | 「launch 单价」用哪个：0.411 / 2.904 / 6.2µs | 同一 1760 launch 的估价在 **0.72 ~ 10.9ms** 之间（15×） | 图化 A/B：`verify_ms(graph)` 与裸流的差 |
| **V10** | `argmax_sliced_rows` 真的把 cross-rank argmax 压成 **1 次 v5 round** | 若退化成 m 次，head 的 −1.5ms 会被协议地板吃掉 | 数 `[dspark]` 段内的 v5 round 次数（或 nsys 看 stamp/poll 核） |
| V1–V6 | 同第一轮（mrows 核固定项未实测；路由专家 ±6ms；attention 每行单价；共享专家 17.3µs 的适用性；激活字节；`padded(288)=320`） | **V2（专家 ±6ms）仍是全表最大不确定项** | 一条 nsys 按 kernel 名聚合 verify 段 |

---

*户部 · 只读分析，未执行任何 GPU 命令、未改动任何代码（本文件为唯一产出）。*
*本轮对代码的唯一动作：读 `chain_dev.rs` / `device.rs` / `dsv41_kernels.cu` + `git log -S` 追时间线。*
