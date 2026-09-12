# 91.1 干净栈的 nsys per-kernel profiling 设计（为下一 session 的 L4/L5 提供准确基线）

> 户部 · 2026-09-12 · **只读勘察 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 现场核对（工作树 HEAD，`git log` 见 §1.4）：`crates/ferrite-models/src/dsv41/{chain_dev.rs,tp.rs}`、
> `kernels/cuda/{dsv41_kernels.cu,dsv41_glue.cu,dsv41_experts_mxf4.cu}`、
> `scripts/{nsys_wave1.sh,dsv41_profile.sh,l49_ab.sh,sh_pair_ab.sh,verify_graph_ab.sh}`、
> `docs/agent/{nsys-wave1-analysis-framework.md,dspark-correctness-chain.md,l4-occupancy-mlp-design.md,l4-l5-kernel-path.md,batch-reverification-plan.md,l49-ab-test-design.md}`。
> **本机无 GPU（`nvidia-smi` 无输出）⇒ 本文件所有分布均为设计口径，标了来源，未实测处显式标注。**

---

## 0. 判决（先读六条，前两条推翻直觉）

1. **nsys pass 不能跑"图开的 91.1 栈"——必须跑"AR-safe 的等价栈"，而后者会自动让 `VERIFY_GRAPH` decline。**
   - nsys 下设备侧 AR v5 的 publish 自旋被逐节点追踪放大 **~300×**（实测 240s / 69 步），所以必须 pin
     `DSV41_AR_V5=0 DSV41_GRAPH_STEP=0`（`nsys_wave1.sh:105-109`、`dsv41-nsys-v14-plan §2.1`）。
   - `ar_v5() = GRAPH_STEP ∥ AR_V5`（`tp.rs:1027-1047`）⇒ 两腿全 0 ⇒ **`ar_v5()==false`**。
   - 而 verify graph 的 per-chain 前置**包含"device-side AR"**（`chain_dev.rs:2152-2155`）⇒
     **`verify_graph_gate` 会 decline，verify 回落到 direct launches**（`chain_dev.rs:6313-6316` 的
     `pos_base<1`/无图分支之外，`capture_verify` 不再被调用）。
   - **这不是缺陷，是设计上唯一可读的口径**：`verify-ms-breakdown §修正` 已证"算子执行时间不因图而变，
     图只 −1.5ms 的 submit 半"。**decline 之后 verify 的每个 kernel 才出现在 `cuda_gpu_kern_sum` 里。**

2. **`cuda_gpu_kern_sum` 看不见图 replay 的 kernel——这是历史方案的最大坑。**
   `dspark-correctness-chain:2971-2973` 白纸黑字："nsys 的 kern_sum 只显示非图 kernel；图 replay 的
   kernel（verify 的主要部分）不在统计中。"⇒ 如果照抄 91.1 栈的 env（`VERIFY_GRAPH=1`）却不同时 pin AR-safe，
   会得到一个**只统计到 draft+AR、verify 近乎为空**的假表。§0-1 的 decline 正好把它变成 direct，规避此坑。

3. **用户给的栈清单漏了两个权威 base env 里有的 gate**：`DSV41_DSPARK=1`、`DSV41_SIDS_WRITEBACK=1`。
   权威 91.1 栈（`l49_ab.sh:184-189` 的 `BASE_ENV`，设计 §2 / batch plan §4 verbatim）里两者都在
   ——`SPEC` 只 arm spec 模式，**DSpark 的 real-commit 路径（verify gates 生存的载体）由 `DSPARK` 开**，
   `nsys_wave1.sh:89` 的头注也写明"without them the verify gates are inert"。**命令必须补上，否则 verify 退化成普通 step。**

4. **三张 report 缺一不可，单看 sum 会把 prefill/load 混进 decode。**
   `kern_sum`（谁最贵）+ `kern_trace`（窗口切分 + 每发 µs + gap）+ `api_sum`（submit vs exec）。
   计数任务只有 ~20 个 spec step，prefill/load 占比不小，**decode 净值必须做时间窗口切分**（§3.2 口径 A）。
   kernel 名含空格/逗号/模板尖括号 ⇒ **必须 `--format csv` + python `r[-1]` 取 Name，禁止 awk 切列**（`nsys_wave1.sh:201-202`）。

5. **本次 profile 要回答的三个问题**（也是它的唯一产出）：
   (a) 91.1 栈的**真实族占比**（旧基线 20.8/19.4/9.7/15-20/12 是"无图 + SH_PAIR_M=1"，缺 R2/MARKOV/FORK/RING_WIN）；
   (b) 新 gate 是否**按设计生效**（判据是"新符号出现 / 旧符号消失"，比 ms 更硬，§4）；
   (c) 下一 session 的 **L4/L5 最大项**（从 (a) 的排序推导，§5）。

6. **本文件不是"再跑一次 nsys_wave1.sh"**。现脚本的 `GATES`（`:91-103`）只有 Wave 1 十项，
   **缺** `ATTN_LIN_FUSE / MARKOV_SLICED / LAZY_SDR / VERIFY_FORK / RING_WIN_FUSE / SH_EXP_MROWS /
   SH_PAIR_M / TAP_INPUT / DRAFT_BF16_DOMAIN / DRAFT_P3A / EXPERT_ACT_E4M3`。
   必须换 GATES（§2.1），否则跑的是 base 不是 91.1。

---

## 1. 前置：环境与时效

### 1.1 工具与产物
| 项 | 值 | 来源 |
|---|---|---|
| nsys | `/usr/local/cuda-13.2/bin/nsys` | `nsys_wave1.sh:74` |
| 追踪 | `--trace=cuda,nvtx --sample=none`（+ `--cuda-graph-trace=node` 兜底） | `dsv41-nsys-v14-plan §2.1` |
| 二进制 | `./target/release/ferrite-serve`（`--model dsv41 --serve --tp 8`） | `nsys_wave1.sh:80` |
| `.so` | `kernels/cuda/libferrite_kernels.so`（`build.sh 103a`） | `nsys_wave1.sh:76` |
| 模型 | `/opt/dlami/nvme/models/DeepSeek-V4.1-Flash` | `nsys_wave1.sh:75` |
| 硬上限 | `DUR=300s`（用户 5 分钟限）——靠 SIGINT nsys 收尾 | `nsys_wave1.sh:150-151` |

### 1.2 计数任务（用户指定：1-100，500 max tokens，快速）
- prompt（逐字，与 `verify_graph_ab.sh:92` 一致）：
  `请从 1 数到 100，每个数字单独占一行，只输出数字本身，不要任何解释。`
- `temperature=0`，`max_tokens=500`，`stream=false`。
- 预期 accept≈5（`400-final-frontier-analysis D15`：counting accept ~4.8-5.0）⇒ `k_emit≈6` ⇒
  **约 100/6 ≈ 17-20 个 spec step**。步数少、prefill/load 占比高 ⇒ §3.2 必须切窗口。

### 1.3 AR-safe（"nccl 模式"的等价物）
- DSV41 **没有 NCCL AR**：all-reduce 是 `Collective::all_reduce_inplace`（`tp.rs:597`），
  注释自陈 "peer copies plus a local reduction — no NCCL"（`tp.rs:10`）。`FERRITE_P2P` 是 GLM 路径的，
  DSV41 不读（`nsys_wave1.sh:14-44`）。
- 所以"nccl 模式"落为 **host-barrier AR**：`DSV41_AR_V5=0 DSV41_GRAPH_STEP=0`（两腿缺一不可）。
  外加 `NCCL_NVLS_ENABLE=0` 廉价护栏、`env -u FERRITE_P2P`（`var_os` 是 presence-check，
  设 `=0` 反而**开** p2p——`nsys_wave1.sh:30-32`）。

### 1.4 时效（执行前必做）
```bash
git log --oneline -10          # HEAD 会话中仍在新提交（91.1/6ff8058/L4-9 都在最近 10 分钟内）
cat kernels/cuda/.build_id     # 与二进制内嵌 id 必须一致，否则进程拒启
```

---

## 2. nsys 命令

### 2.1 GATES（91.1 栈 + AR-safe pins）——替换 `nsys_wave1.sh` 的 `GATES`
```bash
GATES=(
  # DSpark real-commit 路径（verify gates 的载体，缺则 inert）
  DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 DSV41_TIMING=1
  # 专家激活精度 + 截断
  DSV41_EXPERT_ACT_E4M3=1 DSV41_BF16_TRUNCATE=1
  # lazy + verify 图（图在本 pass 会自动 decline，见 §0-1）
  DSV41_LAZY_VERIFY=1 DSV41_VERIFY_GRAPH=1
  # 共享专家
  DSV41_SH_EXP_MROWS=1 DSV41_SH_PAIR_M=1
  # 本轮四件新优化
  DSV41_ATTN_LIN_FUSE=1      # R2
  DSV41_MARKOV_SLICED=1      # MARKOV
  DSV41_LAZY_SDR=1           # SDR
  DSV41_VERIFY_FORK=1        # FORK
  DSV41_RING_WIN_FUSE=1      # RING_WIN
  # Wave 1（hc 融合 + mrows 族 + AR 折叠）
  DSV41_HC_VERIFY_FUSE=1 DSV41_HC_FRONT_ROWS=1 DSV41_VERIFY_AR_FOLD=1
  DSV41_GATE_MROWS=1 DSV41_INDEXER_MROWS=1 DSV41_COMPRESSOR_MROWS=1
  # 其余（与权威 base env 逐字一致）
  DSV41_TAP_INPUT=1 DSV41_DRAFT_BF16_DOMAIN=1 DSV41_DRAFT_P3A=1
)
AR_SAFE=(DSV41_AR_V5=0 DSV41_GRAPH_STEP=0)     # host-barrier AR（nsys 必需）
NCCL_SAFE=(NCCL_NVLS_ENABLE=0)                 # 廉价护栏
```

### 2.2 采集（一条命令）
```bash
NSYS=/usr/local/cuda-13.2/bin/nsys
OUT=/tmp/clean91_nsys
DUR=300 ; MAXTOK=500 ; PORT=8698
PROMPT='请从 1 数到 100，每个数字单独占一行，只输出数字本身，不要任何解释。'

env -u FERRITE_P2P "${GATES[@]}" "${AR_SAFE[@]}" "${NCCL_SAFE[@]}" \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    DSV41_MODEL_DIR=/opt/dlami/nvme/models/DeepSeek-V4.1-Flash \
    DSV41_KERNELS=$PWD/kernels/cuda/libferrite_kernels.so \
  "$NSYS" profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none \
    --output="$OUT" --force-overwrite=true \
    ./target/release/ferrite-serve --model dsv41 --serve --tp 8 \
      --model-dir /opt/dlami/nvme/models/DeepSeek-V4.1-Flash --port "$PORT" \
  >"$OUT.log" 2>&1 &
NSYS_PID=$!
# 硬上限：到点 SIGINT nsys（这才是 report 落盘的唯一方式——DSV41 serve 没有 profiler_stop hook，
# 绝不能用 --capture-range=cudaProfilerApi，见 nsys_wave1.sh:46-54）
( sleep "$DUR"; kill -INT "$NSYS_PID" 2>/dev/null ) &

# 等 "serving"，再驱动 ONE 请求（weight load 发生在首个请求之后，超时给足）
until grep -q "serving" "$OUT.log"; do sleep 2; done
curl -s -m 260 -X POST "http://localhost:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"dsv41\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAXTOK,\"temperature\":0}" \
  >"$OUT.reply.json"

kill -INT "$NSYS_PID"; sleep 15; kill -9 "$NSYS_PID" 2>/dev/null
pkill -9 -x ferrite-serve 2>/dev/null
```

### 2.3 三张表
```bash
$NSYS stats --report cuda_gpu_kern_sum --format csv "$OUT.nsys-rep" > /tmp/kern_sum.csv
$NSYS stats --report cuda_gpu_trace   --format csv "$OUT.nsys-rep" > /tmp/kern_trace.csv
$NSYS stats --report cuda_api_sum     --format csv "$OUT.nsys-rep" > /tmp/api_sum.csv
```

### 2.4 活性自检（先于读表，防止把假表当真）
```bash
grep -c "\[dsv41\] step pos=" "$OUT.log"      # >0 才有 decode 步
grep -c "\[dspark\] steps="  "$OUT.log"       # >0 才有 verify/draft/commit 分解
grep    "\[verify_graph\]"   "$OUT.log"       # 期望「无 captured 行」= 图已 decline（§0-1 预期）
wc -l /tmp/kern_sum.csv                        # <5 行 = 没抓到（nsys_wave1.sh:205 同款硬失败）
grep -cE "gemm_fp8|expert_|hc_" /tmp/kern_sum.csv   # >0 = verify 的 kernel 真的可见（§0-2 的坑）
```
> **若 `[verify_graph] captured …` 行存在** ⇒ 图没 decline ⇒ verify kernel 会被 kern_sum 漏掉 ⇒
> 本 pass 作废，改跑 `DSV41_VERIFY_GRAPH=0`（纯 direct）再读；`--cuda-graph-trace=node` 只是兜底。

---

## 3. 读表口径

### 3.1 kernel → 族 映射（增补 91.1 新符号）
在 `nsys-wave1-analysis-framework §2` 基表上追加/替换：

| 族 | 91.1 栈里的 kernel 名（子串） | 新 gate 带来的变化 |
|---|---|---|
| **投影** | `gemm_fp8_mrows_kernel<M>`（M=行数）；**R2 后新增** `gemm_fp8_kernel`（=`lin2`/mx2，wq_a+wkv 融合）、`gemm_fp8_mx_rope_norm`（=`lin_rope_norm`，norm+wq_b+rope 一发） | `mrows<1>` 实例应**大降**（verify m=1 的 7 发 → 2 发）；`wo_a_grouped_gemv_kernel` 不受 R2 影响 |
| **shared expert** | `gemm_fp8_sh_exp_pair_kernel<1>`（SH_PAIR_M）；`gemm_fp8_sh_pair_kernel`（旧臂，应消失） | `SH_PAIR_M=1` 生效判据 = 前者出现 |
| **routed experts** | `expert_gemv_fp4_batched_kernel`、`expert_gemv_fp4_down_reduce_kernel`、`interleave_gateup_fp4_kernel`、`swiglu_limit_*` | `EXPERT_ACT_E4M3` 改激活解码（1 字节/值），**核不变**；tcgen05 未开 |
| **AR** | `p2p_ar_*_v5*`（**本 pass 应近乎为 0**——v5 被 pin 关）；host-barrier 走 `ar_store/ar_reduce`；`dsv41_hc_post_inplace_kernel` 是 `AR_V5=0` 伪影 | `VERIFY_AR_FOLD` 生效判据 = AR 相关实例数下降 |
| **hc** | `hc_mixes_rows_kernel`/`hc_mixes_ss_kernel`（替代 `hc_mixes_kernel`）；`dsv41_hc_collapse_norm_kernel`（替代 `hc_collapse_kernel`）；`hc_dots_late_kernel`、`hc_mixes_tail_kernel`、`hc_mix_dots_kernel` | `HC_VERIFY_FUSE`/`HC_FRONT_ROWS` 生效判据 = 符号替换 |
| **draft** | **MARKOV 后新增** `dspark_markov_head_sliced_kernel`（替代 `dspark_markov_head_kernel`）+ `argmax_xchg_v5_kernel`（每步 5 轮） | MARKOV 生效判据 = sliced 符号出现 + 旧符号消失 |
| **ring/window** | **RING_WIN 后新增** `ring_win_fused_kernel`（替代 `ring_append_kernel` + `window_idxs_kernel`） | 生效判据 = fused 出现、两个旧核消失 |
| **gate/indexer/compressor** | `gemv_bf16_nt_kernel` / `indexer_rows_one` / `compressor_fused_mrows_kernel` | ⚠️ **lazy m=1 下三个 mrows gate 恒为 0**（折核≡逐行，`swallow-unlocked-next-plan §3.2`）⇒ 本 pass **不应期待**它们出现，出现反而是信号 |
| **norm/rope** | `rmsnorm_rope_kernel`、`apply_rope_kernel`、`dsv41_rmsnorm_rows_kernel`、`dsv41_hc_collapse_norm_kernel` | — |

### 3.2 decode 净值（口径 A：`kern_trace` 时间窗口切分）
`kern_sum` 含 load+prefill+decode 三段（单次采集、无 capture-range）。计数任务步数少，
**必须**用 `cuda_gpu_trace` 的 `Start(ns)` 切出 decode 窗口（取最后一次大 kernel 之后），再按 kernel 聚合：

```python
import csv, collections
rows=[]
for r in csv.reader(open('/tmp/kern_trace.csv')):
    if len(r)<6 or not r[0].replace('.','',1).isdigit(): continue
    try: s,d=float(r[0]),float(r[1])
    except ValueError: continue
    rows.append((s,d,r[-1].strip()))          # Name 在末列
t0=float(input('decode window start ns = '))   # 人工确认 prefill 结束点（看第一个 [dsv41] step 前的最后一个大 kernel）
STEPS=int(input('decode steps = ')); WORLD=8
agg=collections.defaultdict(lambda:[0,0])
for s,d,n in rows:
    if s<t0: continue
    agg[n][0]+=1; agg[n][1]+=d
tot=sum(v[1] for v in agg.values()) or 1
for n,(c,d) in sorted(agg.items(),key=lambda x:-x[1][1])[:25]:
    print(f"{d/tot*100:6.1f}% calls={c:6d} {d/1e6/STEPS/WORLD:8.3f} ms/stp/w  {n[:58]}")
```
**口径 B（更干净，需重跑一次）**：`MAXTOK=1` 与 `MAXTOK=500` 各采一次，`decode = many − one`。
差分把 prefill 的放大项消掉——`dsv41_profile.sh` 的权威口径。
**AR 行不参与排名**：本 pass 是 `AR_V5=0`，host-barrier 的 AR 实例是**伪路径**（`nsys-wave1-analysis-framework §1.3`），
取其**生产 v5 口径 ~0.66ms/步**（`dsv41-kernel-inventory-v3 §0`）另账。

### 3.3 族聚合（csv+python，禁止 awk）
复用 `nsys-wave1-analysis-framework §6` 的族聚合脚本，把族名表换成 §3.1。

### 3.4 四个分析维度
1. **占比排名**（剔除 prefill/load 与 AR 伪影后，谁最贵）；
2. **launch vs 执行**：看 `Avg(ns)` 是否 ≫ 1.4µs ⇒ exec/延迟 bound（`verify-architecture-floor §1`）；
3. **kernel 间 gap**（`kern_trace` 相邻 Start 差）——**VERIFY_FORK 的唯一可测证据**：fork 把三条
   侧链（q/kv、compressor、routed/shared）分发到 3 条流，kernel 名不变，**只有 span 重叠变化**；
   判据 = gap 分布左移 + 同层 kernel 的 span 区间出现跨流重叠（`dsv41-layer-fusion §570-578` 的
   `--cuda-graph-trace=node` 同款做法）；
4. **新旧符号并存**（§3.1 的"生效判据"）——gate 是否真吃到的**硬证据**（比 ms 硬，本项目 #1 陷阱是"gate 没进进程"）。

---

## 4. 预期新分布（设计口径，待本 pass 实测）

**对比基线**（`dspark-correctness-chain:2977-3013`，**无图**、Wave1 + SH_PAIR_M=1 + lazy，78.8 tok/s 栈）：
AR 三件套 **20.8%** / 投影(mrows+wo_a) **19.4%** / SH_PAIR **9.7%** / MoE 路由专家 **15-20%** /
hc 链 **12%** / draft **~19%**（另一节口径）。

**91.1 栈相对它的净变化**（逐 gate 推）：

| 族 | 旧基线 | **预期新** | 变化源（与旧基线的差异） |
|---|---:|---:|---|
| **AR** | 20.8% | **~17-20%** | `VERIFY_AR_FOLD` 把 hc_post 折进 AR store 尾（−88 发/步）；但 AR 本身是协议地板，**占比降幅有限** |
| **MoE routed experts** | 15-20% | **~18-23%** | 自身几乎不变（`EXPERT_ACT_E4M3` 只改解码精度），**占比被动上升**（其他族下降） |
| **投影** | 19.4% | **~11-14%** | **R2（`ATTN_LIN_FUSE`）= 本轮最大单项**：verify m=1 从 7 发/层 → 2 发/层（`chain_dev.rs:2259-2268`）；`mrows<1>` 实例数大降 |
| **hc 链** | 12% | **~8-11%** | `HC_VERIFY_FUSE`+`HC_FRONT_ROWS` 折核（10 发/层 → 4-6 发/层） |
| **SH_PAIR** | 9.7% | **~8-9%** | 中性（旧基线已含 SH_PAIR_M=1） |
| **draft** | ~19% | **~12-15%** | **MARKOV_SLICED**：markov head 扫 126 MiB → 15.8 MiB（40×），5 扫 4 次 L2 命中 |
| **ring/window** | (含在其它) | **小幅降** | RING_WIN：`ring_append`+`window_idxs` 两发 → `ring_win_fused` 一发 |
| **compressor/indexer/gate** | 小 | **不变** | ⚠️ 三个 mrows gate 在 lazy m=1 下**恒为 0** |

> **三条诚实边界**：
> ① 全部为**设计口径**——R2/MARKOV/FORK/RING_WIN 的 ms 收益仓内**没有单项实测**（只有 e2e 91.1 的合成账）；
> ② 占比是**相对值**，某个族"占比上升"可能只是别的族变快（MoE 就是这种）；
> ③ **FORK 不改变任何 kernel 名或数量**，它只在 `kern_trace` 的 span/gap 上可见（§3.4-3）——
>    若 sum 表看起来"没变化"，**不能判 FORK 无效**。

---

## 5. 下一 session 的 L4/L5 目标（从 §4 分布推导）

**推导规则**：L4/L5 只动 **kernel 内部**（占用/MLP/流水），所以靶子 = 分布里**占比大 + 且是 kernel 效率问题**
（不是字节问题、不是协议地板）的族。按 §4 预期分布排序：

| 优先 | 族（预期占比） | 命中 L4/L5 项 | 为什么是它 | 预期（设计口径） | 前置/止损 |
|---|---|---|---:|---|---|
| **1** | **MoE routed experts**（~18-23%） | **L4-3**（`tc5::mxf4` gate/up K-split）+ **L4-4**（`tc5::down` 新核）+ **L5-1**（gateup PDEPTH）+ **L5-3**（`tc5::e4x` kRing） | 分布第一且是**纯核效率**（SIMT，K3 只有 30 CTA/15 SM，`l4-occupancy K3`）；L4-3+L4-4 = L4 收益的 **~50-60%** | −3.0~−5.5ms | **tcgen05 未落地**（`6ff8058` misaligned 失败）⇒ L4-3/L4-4 属**修复后**工作，非本 pass 可动 |
| **2** | **投影族**（~11-14%） | **L4-1**（`gemm_fp8_mrows` nwarps/crossover）+ **L5-2**（gemv 双缓冲） | R2 已把**发数**降过一轮，剩下的肉在**固定项/占用**（`mrows<1>` 的每发 ≥固定延迟） | −0.5~−1.4ms | L4-1 与 L3（SH_PAIR）在半 expert 上互斥；**注意 lazy 下 ×k_emit** |
| **3** | **hc 链**（~8-11%） | **L4-7**（hc 侧流 + dots 网格，**0 代码**）+ **L4-8**（`HC_DL_KCHUNK`）+ **L4-9**（`collapse_norm/rmsnorm` dim-split） | m=1 下 `hc_mixes`/`collapse_norm` **只有 1 CTA、1 SM**（`l4-occupancy K6/K9`）——**占用最刺眼**，且 L4-7 是**最高 ROI** | −1.3~−1.7ms（L4-7 已含在栈中，增量 −0.6ms） | L4-9 已实施、gate 默认 OFF ⇒ **本 pass 后可立刻做它的 A/B**（`l49-ab-test-design` 现成） |
| **4** | **SH_PAIR**（~8-9%） | **L4-2**（phase-1 K-split） | bit-identical 并行度上限 = `ceil(n1/32)·M`；**m=1 时只有 9 个 block**（`l4-occupancy K2`） | −0.3~−0.8ms | 非逐位；m=1 的收益本就小（`lazy-verify §0-4`） |
| **5** | **AR**（~17-20%） | ⚠️ **不是 L4/L5 目标** | AR 是**协议地板**（17.3µs/轮），减次数的唯一路是 accept↑——那是模型/工作负载问题，不是 kernel 重写 | — | `l4-occupancy §2.3`：AR 明确列入"不做" |
| **6** | **draft**（~12-15%） | **L5**（draft 图化，长任务）+ P3A a1-a4 | 短任务（计数 100）图化收益 <1%；**MARKOV 已吃过一轮** | 长任务显著 | `dspark-correctness-chain:2929-2944`：短任务收益可忽略 |

**关键路径（唯一串行主干，与 `l4-l5-kernel-path §3.1` 一致）**：
`tcgen05 gate/up 修复落地 → L4-3 K-split → L4-4 down 换核 → L5-3 e4x kRing`（~11-15 人日）。
**L4-7/L4-1/L4-8/L4-9 可并行**，且**不依赖 tcgen05**。

**本 pass 的直接产出用于 L4/L5 的三件事**（`l4-l5-kernel-path §5-4` 的 U1）：
1. 每族 **CTA 数 / 148**（SM 覆盖率）与**尾波占比** ⇒ L4-1/L4-2/L4-9 的唯一输入；
2. `hc_dots_late` / `gemm_fp8_mrows` 的**每发 µs + warp 数** ⇒ L4-8/L5-2 的基线；
3. AR 与 hc 的 **span 重叠**（`kern_trace`）⇒ 验证 L4-7 侧流遮蔽是否真在起作用。

---

## 6. 风险与止损

| 风险 | 触发信号 | 止损 |
|---|---|---|
| **verify kernel 被图漏掉**（§0-2） | `[verify_graph] captured` 行存在 / kern_sum 里无 `gemm_fp8_*` | 改跑 `DSV41_VERIFY_GRAPH=0` 重采 |
| **AR v5 自旋把 pass 拖死** | 请求 >260s 无返回 / `.log` mtime 停滞 | 确认 `AR_V5=0`+`GRAPH_STEP=0` 两腿都在（`/proc/<pid>/environ` 实读） |
| **gate 没进进程**（本项目 #1 陷阱） | 符号替换判据（§3.4-4）不成立 | `tr '\0' '\n' < /proc/<pid>/environ \| grep DSV41_ \| sort` 读回 |
| **prefill/load 混入** | sum 表总值 ≫ STEPS×步时 | 必须做 §3.2 窗口切分（或口径 B 差分） |
| **占比被"别的族变快"误导** | 某族占比上升但绝对值未测 | 同时报 **绝对 ms/步/world**，不只报 %（§4-②） |
| **把 FORK 判成无效**（§4-③） | sum 表无变化 | 必须看 `kern_trace` 的 span/gap |

---

*户部 · 只读勘察 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*所有分布为设计口径；修正处（用户栈清单缺 DSPARK/SIDS_WRITEBACK、verify graph 须 decline、kern_sum 漏图 kernel）已显式标注依据与 file:line。*
