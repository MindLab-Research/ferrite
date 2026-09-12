# nsys per-kernel 分析框架（Wave-1 verify 剩余瓶颈识别）

> 任务：为即将跑起来的 `scripts/nsys_wave1.sh` 设计**读表口径 + kernel↔族映射 + 判定标准**。
> 方式：**只读**——读 `scripts/nsys_wave1.sh`、`crates/ferrite-models/src/dsv41/chain_dev.rs`、
> `kernels/cuda/*.cu` 与仓库账本（`verify-architecture-floor` / `dsv41-nsys-v14-plan` /
> `verify-ms-breakdown` / `dsv41-kernel-inventory-v3`）。未执行 GPU 命令、未改动任何源码。
> 基线：`DSV41_TIMING` verify(m=5) = **37.31ms / 6224 launches / 14.09GB**（`verify-architecture-floor §1`）。
> 交付前请以本文件为唯一读表依据；所有 kernel 名均以 **CSV 现场字符串**为准（见 §1.3）。

---

## 0. 判决（先读五条）

1. **`[dsv41] step pos=N: X.XXms` 是假的**（serve 侧墙钟，含 curl/SSE/admission/queue/tail 五项非模型开销）。
   `[dspark] steps=` 里的 `verify=` 也只半真（含 host barrier + D2H sync）。
   **唯一纯模型执行时间 = nsys per-kernel GPU 时间。** 这是整个框架的前提。
2. **Wave 1 组合（脚本 §GATES）不含 SH_PAIR**：`DSV41_SH_PAIR` / `DSV41_SH_EXP_MROWS` / `DSV41_SH_PAIR_M`
   **都不在 gate 列表里**（`chain_dev.rs:1259/1331/1392` 三个 gate 默认 OFF）。
   ⇒ **shared expert 在本轮 profile 里"仍然 ~10ms"是预期结果，不是失败。** 它是 Wave 2 的头号目标。
3. **Wave 1 只动三件事**：hc 链（`HC_VERIFY_FUSE`+`HC_FRONT_ROWS`+`BF16_TRUNCATE`）、
   mrows 族（`GATE_MROWS`+`INDEXER_MROWS`+`COMPRESSOR_MROWS`）、verify 图化+AR 折叠。
   **routed experts（~8.3ms）、attention（~2.8ms）、shared expert（~10.4ms）三项 Wave 1 一律不碰。**
4. **因此"剩余瓶颈"的预期排序**（L0 账本，`verify-architecture-floor §2.3`）：
   `shared expert 10.40` ≈ `routed 8.30` ≫ `投影 3.70` > `gate 3.44` > `hc 2.96` >
   `attention 2.80` > `indexer 2.50` > `head 1.12`。
   **Wave 1 后前两项合计仍占 ~50%** —— 这就是本框架要钉死的结论。
5. **框架的产出是"三张表差分"**，不是单一 CSV：`cuda_gpu_kern_sum`（谁最贵）+ `cuda_gpu_trace`（每发实测 µs /
   时间窗口）+ `cuda_api_sum`（launch 提交 vs 执行）。只读 sum 会把 prefill 混进 decode。

---

## 1. 数据源与口径（怎么读表）

### 1.1 三张 report 的分工

| report | 给什么 | 在本框架里的用途 |
|---|---|---|
| `cuda_gpu_kern_sum` | 每个 kernel：**Total / Instances / Avg / Med / Min / Max**（按名字聚合） | **谁最贵**（占比 + 实例数）。脚本已自动跑（`nsys_wave1.sh:203`） |
| `cuda_gpu_trace` | 每个 kernel **实例**：Start / Duration / Grid / Block / Stream | **时间窗口切分**（只取 decode 段）+ 每发实测 µs + launch gap |
| `cuda_api_sum` | 每个 CUDA API 调用：Total / Instances / Avg | **launch 提交 vs 执行**：把 `cudaLaunchKernel` 的 host 时间与 GPU 时间对照，判"submit bound vs exec bound" |

```bash
NSYS=/usr/local/cuda-13.2/bin/nsys
REP=/tmp/wave1_nsys.nsys-rep
$NSYS stats --report cuda_gpu_kern_sum  --format csv "$REP" > /tmp/kern_sum.csv
$NSYS stats --report cuda_gpu_trace    --format csv "$REP" > /tmp/kern_trace.csv
$NSYS stats --report cuda_api_sum      --format csv "$REP" > /tmp/api_sum.csv
```

### 1.2 CSV 列序（**Name 是最后一个字段**）

```
cuda_gpu_kern_sum: Time(%), Total Time(ns), Instances, Avg(ns), Med(ns), Min(ns), Max(ns), StdDev(ns), Name
cuda_api_sum:      Time(%), Total Time(ns), Num Calls, Avg(ns), Min(ns), Max(ns), StdDev(ns), Name
```

⚠️ kernel 名含**空格、逗号、`<`/`>`**（template 实例，如 `gemm_fp8_mrows_kernel<5>`）。
**禁止用 `awk` 按列切；必须用 python `csv` + `r[-1]` 取 Name**（`dsv41-nsys-v14-plan §2.2`，`nsys_wave1.sh:201-202` 已按此实现）。

### 1.3 ⚠️ 单次采集的三个口径陷阱（本框架的**核心风险**）

1. **`nsys_wave1.sh` 只跑一次、无 `--capture-range`**（头部 §CAPTURE RANGE 说明：DSV41 serve 没有
   profiler_stop hook，只能整进程抓 + SIGINT 收尾）。⇒ **CSV 里混着 4 段**：
   权重加载（load kernels）+ **prefill** + decode + 收尾。
   **`sum` 表按名字聚合 = decode 与 prefill 同符号合并**，`Med` 是两者混合population（`§2.2` 的"Med 含 prefill"坑）。
2. **脚本的 `ms/stp/w` 归一化（`nsys_wave1.sh:236`）用 `STEPS = log 里 "[dsv41] step pos=" 的行数`**。
   这只是**粗略 proxy**：分子里仍含 load+prefill 的 kernel 时间，分母用 decode 步数 ⇒ **系统性高估**。
   **decode 净值必须用 §1.4 的窗口切分或两次采集差分。**
3. **AR 行是 host-barrier 伪影**：`AR_SAFE=(DSV41_AR_V5=0 DSV41_GRAPH_STEP=0)` 把 AR 钉到 host barrier
   （`nsys_wave1.sh:105-109`），所以 `ar_stamp_kernel` / `ar_store_kernel` / `ar_reduce_kernel` / `ar_mark_kernel`
   出现的是**伪路径**。生产 v5 的绝对值测不到（自旋在 nsys 下 ~300× 病态）。
   **AR 行不参与占比排名**；取生产口径 **~0.65ms**（`dsv41-nsys-v14-plan §3 #5`）。
   同样 `dsv41_hc_post_inplace_kernel` 是 `AR_V5=0` 的副产物（生产 `HCPOST_EPI=ON` 已折进 AR epilogue）。

### 1.4 decode-only 净值：两种正确口径

**口径 A（推荐，无需重跑）——`cuda_gpu_trace` 时间窗口切分：**

`cuda_gpu_trace` 给每个实例的 `Start (ns)`。decode 段 = 从第一个 `[dsv41] step` 对应的 step 序列开始，
到停止前。实操上取**最后一次大 kernel（prefill 的 `gemm_fp8_*` / attention 大 grid）之后**的窗口：

```python
import csv, collections
# trace CSV: Start(ns), Duration(ns), ..., GridX.., Name(末列)
rows = []
for r in csv.reader(open('/tmp/kern_trace.csv')):
    if len(r) < 6 or not r[0].replace('.','',1).isdigit():   # skip header/notes
        continue
    try: start, dur = float(r[0]), float(r[1])
    except ValueError: continue
    rows.append((start, dur, r[-1].strip()))
t0 = float(input('decode window start ns = '))   # 人工确认 prefill 结束点
t1 = max(s for s,_,_ in rows)
agg = collections.defaultdict(lambda: [0, 0])
for s, d, n in rows:
    if s < t0: continue
    agg[n][0] += 1; agg[n][1] += d
tot = sum(v[1] for v in agg.values())
STEPS = int(input('decode steps = ')); WORLD = 8
for n,(c,d) in sorted(agg.items(), key=lambda x:-x[1][1]):
    print(f"{d/tot*100:6.1f}% calls={c:6d} {d/1e6/STEPS/WORLD:8.3f} ms/stp/w  {n[:58]}")
```

**口径 B（更干净，需重跑两次）——差分法（`dsv41-nsys-v14-plan §2.1`）：**
`MAXTOK=1` 与 `MAXTOK=30` 各采一次，`decode = many − one`，除以 `steps=N-1` 与 `world=TP`。
差分把 prefill 的放大项消掉。**这是 `dsv41_profile.sh` 的权威口径。**

> **对本轮的建议**：既然 `nsys_wave1.sh` 固定 `MAXTOK=20`，先跑它，用**口径 A** 出 decode 表；
> 若 sum 表与窗口表结论冲突（尤其 shared expert / routed experts），再补一次 `MAXTOK=1` 走口径 B 差分。

---

## 2. kernel → 族 映射表（从 CSV 名字反查族）

**通用规则**：nsys `Name` = **demangle 后的 `__global__` 函数名**（含 template 参数）。
先按**子串**归类（`gemm_fp8_` / `sparse_attn` / `hc_` / `expert_` / `compressor` / `indexer_`），
模板实例 `kernel<M>` 里的 `M` **就是 verify 行数**（Wave 1 的 mrows 是 m=5；SWALLOW 后是 m=6）。

| 族 | CSV 里的 kernel 名（子串匹配） | 定义位置 / 入口符号 | L0 实测 ms | Wave 1 是否动 |
|---|---|---|---:|---|
| **投影** | `gemm_fp8_mrows_kernel<M>`（多行，`M=1..8`） | `dsv41_kernels.cu:5210` / `dsv41_gemm_fp8_mrows` | 3.70 | ✅ mrows（隐含在 verify 路径） |
| | `gemm_fp8_kernel`（M=1 GEMV 旧路径） | `dsv41_kernels.cu:267` | | |
| | `gemm_fp8_swapab_kernel`、`gemm_fp8_wo_pair_kernel` | `dsv41_kernels.cu`（`dsv41_gemm_fp8_swapab` / `_wo_pair`） | | |
| | `wo_a_grouped_gemv_kernel` | `dsv41_kernels.cu:5537` | | |
| | `dsv41_gemm_fp8_mx*` 融合变体（ROPE/norm/add epilogue） | 入口 `dsv41_gemm_fp8_mx/_rope/_rope_norm/_add/_f32/_mx2`；**__global__ 名从 CSV 现场确认** | | |
| **shared expert** | `gemm_fp8_sh_pair_kernel`（**M=1 旧臂**） | `dsv41_kernels.cu:6513` / `dsv41_gemm_fp8_sh_pair` | **10.40** | ❌ **未启用**（SH_PAIR 不在 GATES） |
| | `gemm_fp8_sh_exp_pair_kernel<M>`（**新 template 臂**） | `dsv41_kernels.cu:6933` / `dsv41_gemm_fp8_sh_exp_fused` | | ❌（Wave 2） |
| **routed experts** | `expert_gemv_fp4_batched_kernel`（gate/up+swiglu） | `dsv41_experts_mxf4.cu:1229` | **8.30**(族) | ❌ |
| | `expert_gemv_fp4_down_reduce_kernel` | `dsv41_experts_mxf4.cu:2107` | | ❌ |
| | `expert_gemv_fp4_kernel`（逐行旧臂）、`interleave_gateup_fp4_kernel` | `dsv41_experts_mxf4.cu:642 / 2434` | | ❌ |
| | `expert_tcgen05_gateup_kernel` / `_mxf4_kernel` / `_e4_kernel` | `dsv41_experts_mxf4.cu` | | ❌（L2，未开工） |
| | `swiglu_limit_batched_kernel` / `swiglu_limit_kernel` | `dsv41_glue.cu:1129 / 173` | | ❌ |
| | `moe_down_reduce_kernel`、`moe_route_kernel` | `dsv41_experts_mxf4.cu:2020` / `dsv41_kernels.cu:2562` | | ❌ |
| **attention** | `sparse_attn_split_kernel` | `dsv41_kernels.cu:1506` | **2.80**(族) | ❌ |
| | `sparse_attn_merge_kernel` | `dsv41_kernels.cu:1772` | | ❌（**hot-kernel-restore 后仍存在**，见 §5.2） |
| | `sparse_attn_orope_kernel` / `sparse_attn_kernel` / `sparse_attn_pf_kernel` / `sparse_attn_warp_kernel` | `dsv41_kernels.cu:2057 / 940 / 1133 / 1044` | | ❌ |
| **hc 链** | `hc_collapse_kernel`（旧两段之一） | `dsv41_glue.cu:304` | **2.96**(族) | ✅ `HC_VERIFY_FUSE` |
| | `dsv41_hc_collapse_norm_kernel`（**A1 融合**） | `dsv41_kernels.cu:9055` | | ✅ |
| | `hc_mixes_kernel` / `hc_mixes_rows_kernel` / `hc_mixes_ss_kernel` / `hc_mixes_post_kernel` | `dsv41_kernels.cu:2398/8729/8707/8751` | | ✅ `HC_FRONT_ROWS`→`hc_mixes_auto` |
| | `hc_mix_dots_kernel`、`hc_dots_late_kernel(_kchunk)`、`hc_mixes_tail_kernel`、`hc_front_kernel` | `dsv41_kernels.cu:9147/9969/10117/9220/9405` | | ✅ |
| | `hc_pre_persist_kernel` / `hc_pre_persist_mb_kernel` | `dsv41_kernels.cu:9644 / 10751` | | ✅ |
| | `dsv41_hc_post_inplace_kernel`（**AR_V5=0 伪影**）、`_rows_kernel` | `dsv41_kernels.cu:8902 / 8975` | | 伪影，勿计入 |
| **gate / route** | `gemv_bf16_v2_kernel`（**默认 gate**，含 route 融合 epilogue） | `ferrite_kernels.cu:3217` | **3.44** | ✅ `GATE_MROWS` 走 `gemv_bf16_nt_kernel` |
| | `gemv_bf16_nt_kernel`（**mrows 契约：per-row 累加器**） | `ferrite_kernels.cu:3437` | | ✅ 期望出现 |
| | `gemv_bf16_kernel`（head / idx_weights） | `ferrite_kernels.cu:3015` | | |
| | `gemv_bf16_v1_mrows_kernel`、`head_gemv_bf16_mrows_kernel` | `ferrite_kernels.cu` / `dsv41_glue.cu:561` | | |
| **indexer** | `indexer_topk_kernel`、`indexer_score_kernel(_v2)`、`index_k_publish_kernel` | `dsv41_kernels.cu:2874/2678/2770/` | **2.50** | ✅ `INDEXER_MROWS`（front 折进 `indexer_rows_one`） |
| | `argmax_xchg_v5_rows_kernel` / `dsv41_argmax_sliced_rows` | `dsv41_kernels.cu` | | ✅ |
| **compressor** | `compressor_fused_kernel`（**已融合**） | `dsv41_kernels.cu:3238` | <0.03 | ✅ `COMPRESSOR_MROWS`→`compressor_fused_mrows_kernel` |
| | `compressor_fused_mrows_kernel` | `dsv41_kernels.cu:3378` | | ✅ 期望出现 |
| | `compressor_state_kernel` / `compressor_pool_kernel` / `compress_commit_kernel` | `dsv41_kernels.cu:3057/3100/` | | 应**消失**（`COMPRESS_FUSE`） |
| **head** | `head_gemv_bf16_mrows_kernel`、`gemv_bf16_kernel`（head 形状） | `dsv41_glue.cu:561` | 1.12 | ❌ |
| **norm/rope** | `rmsnorm_rope_kernel`、`rmsnorm_q_kernel`、`dsv41_rmsnorm_rows_kernel` | `dsv41_kernels.cu` | (含"其余"~0.30) | ❌ |
| | `apply_rope_kernel` / `apply_rope_mrows_kernel` / `rope_precompute_kernel` | `dsv41_kernels.cu` | | |
| **AR（伪）** | `ar_stamp_kernel` / `ar_store_kernel` / `ar_reduce_kernel` / `ar_mark_kernel` | — | 伪 ~0.65(生产) | ❌ 勿排名 |
| **engram / dspark** | `engram_*`、`dspark_markov_head(_sliced)_kernel`、`dspark_ring/comp_*` | `dsv41_kernels.cu` | — | ❌ |

> **模板展开的核对**：`gemm_fp8_mrows_kernel<5>` 与 `gemm_fp8_sh_exp_pair_kernel<5>` 的 `<5>` 就是 m。
> 若 CSV 出现 `gemm_fp8_mrows_kernel<1>` 且 calls 远多于预期 ⇒ mrows 没吃到（decline 回落到 M=1 循环）。
> 若 shared expert 的行名是 `gemm_fp8_sh_pair_kernel`（无 `<M>`）⇒ **SH_PAIR_M 未生效**（本轮本就未启用）。

---

## 3. 四个分析维度（怎么算）

### 维度 1：占比排名（谁最贵）
`sum` 表按 `Total Time` 降序 → 每族占比。**但先剔除 prefill/load 与 AR 伪影**（§1.3）。

### 维度 2：launch 开销 vs 执行时间
用 `sum` 表的 `Instances` 与 `Avg(ns)`，或用 `trace` 表逐实例：

```
每发全价 t_call = Avg (ns)
launch 数 N_call = Instances
核内下限 t_min  ≈ max(bytes/BW, μop/issue)   （见 verify-architecture-floor §2.3 模型）
可回收量 ≈ Σ_f N_f × (t_f − t_f^min)
```
`verify-architecture-floor §1`：实测 37.31ms / 6224 = **5.99µs/发**；其中**只有 ~1.4µs 是"发核"**，
其余 ~4.6µs 是核自己的 μop + 延迟暴露。⇒ **"少发核"上界 ≈ 6224×1.4µs ≈ 8.7ms，不是 20.5ms。**
本维度就是验证这条：**看 `Avg` 是否远大于 `1.4µs`（是 ⇒ exec/延迟 bound，不是 submit bound）。**

### 维度 3：kernel 间 gap（launch 提交的间隙）
用 `trace` 表把同一 stream 的实例按 `Start` 排序，算相邻 `gap = next.Start − (cur.Start+cur.Duration)`：

```python
inst = sorted([(s,d,n) for s,d,n in rows if s>=t0], key=lambda x:x[0])
gaps = [(inst[i+1][0]-(inst[i][0]+inst[i][1]), inst[i][2]) for i in range(len(inst)-1)]
# 汇报 p50/p90 gap 与 gap 最大的 top-10 前驱 kernel
```
- gap ≈ 0 ⇒ 排队满（GPU 忙）。
- gap 大且前驱是 tiny kernel ⇒ **launch 提交/依赖链 ramp 暴露**（图化只证明 CPU submit 重叠，不证明 GPU 侧重叠，
  `verify-architecture-floor §4.3`）。
- **Wave 1 的 `VERIFY_GRAPH=1` 应让 gap 分布整体左移**；若没左移，图没吃到（查 `.so` / gate 是否被 env 覆盖）。

### 维度 4：同族多符号的"新 vs 旧"并存（gate 是否真生效）
**Wave 1 的每个 gate 都有"新旧两臂"。判定生效 = 新符号出现/加权，旧符号消失/降权。**
这是本框架最硬的自检维度（见 §4）。

---

## 4. 判定标准（每族预期 + 通过判据）

**基准**：L0 = 37.31ms（`verify-architecture-floor §1`）；Wave 1 = L1/L2 之间，
子集实测 `{SH_EXP,GRAPH,ROPE,P3A} = −1.21ms`（§3 表 L1 行）。
⇒ **Wave 1 后总量预期 ~31~33ms**（不是 20~22；20~22 是 L3，需族级融合，Wave 1 没做）。

| 族 | Wave 1 预期 | 若不符的解读 |
|---|---|---|
| **hc 链** | **2.96 → ~1.3~1.7ms**；`dsv41_hc_collapse_norm_kernel` 出现、`hc_collapse_kernel` 消失（A1 生效）；`hc_mixes_rows/ss` 出现替代 `hc_mixes_kernel`（A2 生效） | 若 hc 仍 ~2.96 ⇒ `HC_VERIFY_FUSE=1` 或 `BF16_TRUNCATE=1` 没吃到（`.so` 无 `hc_collapse_norm` 符号，或 gate 被 env 覆盖）；**A2 的 `bf16_truncate=false` 一行未修**（§8 排名 4） |
| **gate** | 3.44 → 降（计划口径 −2.75，未验）；`gemv_bf16_nt_kernel` 出现、`gemv_bf16_v2_kernel` 的 MoE-router calls（~40/layer）降权 | 若 `gemv_bf16_v2_kernel` calls 不变 ⇒ `GATE_MROWS` decline（查 `gemv_bf16_v2_wanted` + `.so` 有 `dsv41_gemv_bf16_v2_mrows`） |
| **indexer** | 2.50 → −1.0~1.5；front 折进 rows 形态 | 若 `indexer_topk_kernel` calls 不变 ⇒ `INDEXER_MROWS` 的 select 半缺 `out_stride` / `n_pos=max(lens)` |
| **compressor** | <0.03ms，**量级不变**；`compressor_fused_mrows_kernel` 出现、`compressor_state/pool/commit_kernel` 消失 | 若旧三核仍在 ⇒ `COMPRESSOR_MROWS` 或 `COMPRESS_FUSE` 未吃到 |
| **shared expert** | **仍 ~10.4ms（预期！）** — SH_PAIR/SH_EXP_MROWS/SH_PAIR_M **不在 GATES** | 若显著 <10 ↓ 说明 Wave 1 意外改变了它，需查是否有别的手臂；若 >10 ↑ 查是否 5× 重读加剧 |
| **routed experts** | **仍 ~8.3ms（预期！）** — tcgen05 未开工 | 若明显变化 ⇒ 检查 `DSV41_SKIP_EXPERTS` / `EXPERT_FP4_MODE` 是否被外部 env 污染 |
| **attention** | **仍 ~2.8ms（预期！）** — 未碰 | `ATTN_MROWS` 未在 GATES；若 `sparse_attn_split/merge` calls ≠ 40/layer 需查 |
| **投影** | mrows 生效：`gemm_fp8_mrows_kernel<5>` 为主，`gemm_fp8_kernel`（M=1）calls 大幅降 | 若 `<1>` 实例多 ⇒ mrows decline 回 M=1 |
| **总量** | **31~33ms**（vs L0 37.31） | >34 ⇒ `LAZY_VERIFY` 只跑 2 行/`VERIFY_GRAPH` 没生效；<30 ⇒ 有意外收益，逐 gate 二分归因 |

### 4.1 ⚠️ 对任务前提里四条"预期结果解读"的修正

| 任务前提 | 修正 |
|---|---|
| "如果 shared expert 仍 ~10ms：**SH_PAIR 未生效或 instruction-bound**" | **部分错**：Wave 1 **没启用 SH_PAIR**，所以"仍 ~10ms"是**预期基线**，不能判 SH_PAIR 失败。SH_PAIR 的 parity 与增益要在**另开 gate 的 A/B** 里测（`sh-pair-template-m-design.md`）。instruction-bound 是**原因**（§5.1 μop 墙），不是判据。 |
| "如果 hc 链 ~1ms：A1+A2 融合生效" | ✅ 方向对。判据用 **符号替换**（`hc_collapse_norm` 出现 + `hc_collapse` 消失），比 ms 更硬。 |
| "如果 attention ~3ms：最大剩余项" | ⚠️ attention 2.80 是**第三档**；真正最大剩余项是 **shared expert 10.40 + routed 8.30**。attention 只占 ~7.5%。 |
| "如果 MoE gate ~1ms：GATE_MROWS 生效" | ✅ 但要按**符号**判（`gemv_bf16_nt_kernel` 出现），且 gate 族 L0 是 **3.44ms**（不是 1ms）；降到 ~1ms 是乐观端。 |

---

## 5. 已知陷阱与误读清单（采集/读表前必读）

1. **`Med` 含 prefill**（`§2.2`）：decode 与 prefill 实例数差异大的 kernel（`gemm_fp8_*`、`sparse_attn_*`）`Med` 偏高。
   **decode 结论用 §1.4 的窗口/差分，`Med` 只做交叉核对。**
2. **`sparse_attn_merge_kernel` 仍在**（`dsv41-nsys-v14-plan` 头部 hot-kernel-restore 注）：`f6d2dde` 的
   sparse-merge 选举折叠**已整体回退删除**，所以 **40 次/步的 merge launch 仍存在**。
   v14 计划里"merge 行从 CSV 消失"的判据**作废**；以"无 fold 的两 launch 形态"为基线。
3. **`hc_mixes_tail_kernel` / `gemv_bf16_v2_kernel` 是聚合行**：同符号不同 `mode`/grid，`sum` 表按名字合并
   （`§5.3 / §5.8`，160 次一行 / 混合 population）。**"EARLY 恢复 1.7µs"无法从 sum 直读**，需 `trace` 按 grid/block 拆。
4. **`gemv_bf16_v2_kernel` ≠ `gemv_bf16_fp8x2_kernel`**（`§5.8` 澄清）：后者由 `DSV41_MIX_GATE` 门控，**默认 OFF，HEAD 上 0 调用**。
   不要把 sum 表里 `gemv_bf16_v2_kernel` 的 48 次当成混核。
5. **AR 行是 host-barrier 伪影**（`§5.1`）：不参与排名；取生产 0.65ms。
6. **`dsv41_hc_post_inplace_kernel` 是 `AR_V5=0` 副产物**：生产 `HCPOST_EPI=ON` 已折进 AR epilogue。
7. **空 CSV 静默打 0.0ms**：脚本对 `rows<5` 硬失败（`nsys_wave1.sh:205`）；若非空但全 0，先看 `$LOG` 里 binary 是否真跑了。
8. **名字含空格/逗号** → 一律 `--format csv` + python（`nsys_wave1.sh:201-202` 已实现）。

---

## 6. 一次性交付命令（profile 落地后照抄）

```bash
# 0) 采集（用户已批准的命令；约 5min，DUR=300 硬上限）
scripts/nsys_wave1.sh                     # → /tmp/wave1_nsys.{nsys-rep,csv,log}

# 1) 三张表
NSYS=/usr/local/cuda-13.2/bin/nsys; REP=/tmp/wave1_nsys.nsys-rep
$NSYS stats --report cuda_gpu_kern_sum --format csv "$REP" > /tmp/kern_sum.csv
$NSYS stats --report cuda_gpu_trace   --format csv "$REP" > /tmp/kern_trace.csv
$NSYS stats --report cuda_api_sum     --format csv "$REP" > /tmp/api_sum.csv

# 2) sum 表 → 族占比（脚本已内置前 20 名，这里做"族聚合"）
python3 - <<'PY'
import csv,re,collections
fam=[('投影',r'gemm_fp8_(mrows|swapab|wo_pair|mx)'),('shared expert',r'sh_(pair|exp)'),
     ('routed experts',r'expert_|interleave_gateup|swiglu_limit|moe_(route|down)'),
     ('attention',r'sparse_attn'),('hc',r'hc_'),('gate',r'gemv_bf16'),
     ('indexer',r'indexer_|argmax'),('compressor',r'compressor_|compress_'),
     ('head',r'head_gemv'),('AR伪影',r'\bar_')]
rows=[]
for r in csv.reader(open('/tmp/kern_sum.csv')):
    if len(r)<6: continue
    try: total,inst=float(r[1]),int(r[2])
    except ValueError: continue
    rows.append((total,inst,r[-1].strip()))
tot=sum(t for t,_,_ in rows) or 1
agg=collections.defaultdict(lambda:[0,0,0])
for t,i,n in rows:
    for f,p in fam:
        if re.search(p,n,re.I) or (f=='投影' and n.startswith('gemm_fp8')):
            agg[f][0]+=t; agg[f][1]+=i; agg[f][2]+=1; break
    else: agg['其他'][0]+=t; agg['其他'][1]+=i; agg['其他'][2]+=1
print(f"{'族':<16}{'share':>7}{'calls':>8}{'kernels':>8}")
for f in sorted(agg,key=lambda k:-agg[k][0]):
    t,i,k=agg[f]; print(f"{f:<16}{t/tot*100:6.1f}%{i:8d}{k:8d}")
print(f"total GPU kernel time = {tot/1e6:.1f} ms  (含 prefill/load，须用 §1.4 口径 A 修正)")
PY
```
**然后**：对 `shared expert` / `routed experts` / `attention` 三大项，用 §1.4 口径 A 在 `cuda_gpu_trace` 上做
decode 窗口重算，得到**真净 ms/步/world**，与 §4 表逐行回填。

---

## 7. 交付一句话

> **读表口径**：`kern_sum`（谁最贵）+ `kern_trace`（窗口切分 + 每发实测 + gap）+ `api_sum`（submit vs exec）三表联动；
> **但单次采集的 CSV 含 load+prefill，必须先按 §1.4 口径 A/B 剔掉**，否则占比与 ms/步系统性偏高。
> **族映射**：模板名 `kernel<M>` 的 `M` 就是 verify 行数——这是判断 mrows/SH_PAIR 是否生效的硬判据。
> **Wave 1 后剩余瓶颈的预期结论**：**shared expert ~10.4ms + routed experts ~8.3ms ≈ 剩余的一半**（Wave 1 一律没碰），
> 其后 attention ~2.8ms；hc（~1.3）与 gate/indexer 是被 Wave 1 吃掉的部分。
> ⇒ **Wave 2 的头号目标是 shared expert（SH_PAIR_M）与 routed experts（tcgen05），不是 attention。**
> 框架的最终用途 = 钉死 `verify-architecture-floor §9` 的 **U1（per-kernel boundary 可回收量）** 与 **U2（routed ±6ms）**。

*工部 · 只读分析，未执行 GPU 命令、未改动任何源码；本文件为唯一产出。*
*所有 kernel 名均标注定义位置（file:line）；CSV 现场字符串为准。*
