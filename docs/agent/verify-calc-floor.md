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
| 7 | **head**<br>[vocab,dim] bf16 | 129280×5120×2B = **1323.8MB**（`Shard::Replicated`，每卡全量）<br>`VERIFY_HEAD_FOLD` 默认 **0** ⇒ 逐行 ×5 | **6619.1** | 165.5<br>(折叠+切分) | 0.024 | **1.49** | 4.44 TB/s | **30×** | 🟠 **最便宜的肉** |
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
| 5 | **head 折叠 + 词表切分**（`VERIFY_HEAD_FOLD=1` + 多行跨卡 argmax，TODO#3） | `chain_dev.rs:4094` / `:970` | **1.49 → 0.05ms（−1.44ms）** | **最高**（两个部件都已实现/已实测 48.5µs） | 1.4 |
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
