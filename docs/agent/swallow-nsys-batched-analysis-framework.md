# SWALLOW (batched) 的 nsys per-kernel 分析框架（batched 分布 + 与 lazy 的对比表）

> 户部 · 2026-09-12 · **只读勘察 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 现场核对（工作树 HEAD `969850c`）：`crates/ferrite-models/src/dsv41/{chain_dev.rs,tp.rs,config.rs}`、
> `kernels/cuda/{dsv41_kernels.cu,dsv41_experts_mxf4.cu,dsv41_glue.cu}`、
> `scripts/{batched_400_v2.sh,nsys_wave1.sh,sh_pair_ab.sh}`、
> `docs/agent/{nsys-clean-stack-91-design.md,nsys-wave1-analysis-framework.md,swallow-unlocked-next-plan.md,
> swallow-unlocked-shpair-m6-throughput-plan.md,swallow-unlocked-400-final-path.md,batch-reverification-plan.md,
> dsv41-kernel-inventory-v3.md,lazy-batched-gate.md}`。
> **本机无 GPU、`kernels/cuda/*.so` 不存在（`ls` No such file）⇒ 本文件所有分布均为「设计口径 / 预期方向」，凡未实测处一律显式标注。**

---

## 0. 判决（先读七条，前三条修正任务前提）

1. **❗ `DSV41_SH_PAIR_M` 不在 `scripts/batched_400_v2.sh` 的 GATES 里（现场核对 `:145-156`）。**
   任务前提「SH_PAIR M=6 已激活（`DSV41_SH_PAIR_M=1` 在 gates 里）」与**仓库现状不符**：
   当前 batched 矩阵是 `SPEC/DSPARK/SIDS_WRITEBACK/EXPERT_ACT_E4M3/BF16_TRUNCATE/SH_EXP_MROWS/MROWS_SMALL_N_ADAPTIVE/
   GATE/VERIFY_HEAD/INDEXER/NORM/COMPRESSOR_MROWS/DRAFT_GRAPH/DRAFT_P3A/VERIFY_GRAPH/SWALLOW_STEP/SWALLOW_EPOCH_PAD/
   V5_LEDGER/TIMING/DSPARK_DEBUG`，**没有 `SH_PAIR_M`**（`swallow-unlocked-400-final-path §1` 也这么写）。
   ⇒ **若照脚本裸跑，测到的是「无 SH_PAIR」的 batched 栈**——「SH_PAIR M=6 的效果」这一问会得到假答案（幻影门）。
   **nsys 的 gate 串必须显式加 `DSV41_SH_PAIR_M=1`**（§1.2）。

2. **❗ `DSV41_SWALLOW_STEP=1` 与 `DSV41_LAZY_VERIFY=1` 互斥（脚本 `FORBIDDEN:175`）。**
   batched 的 nsys **绝不能带 `LAZY_VERIFY`**；否则测的是 lazy 臂（m=1），与「batched 分布」这一目标无关。
   同理禁止 `HC_VERIFY_FUSE` / `HC_FRONT_ROWS`（它们在 batched 矩阵里被显式排除，`:175`；`HC_FRONT_ROWS` 是 lazy/91.1 栈的 L4-7 项）。

3. **❗ 任务给的「AR 27.1% #1」与生产 AR 口径（0.66ms ≈ 2%）不能用同一把尺子读。**
   nsys 的 clean-stack 口径必须 pin `AR_V5=0 DSV41_GRAPH_STEP=0`（见 §0-4），此臂下 AR 走 **host-barrier 伪路径**
   （`ar_store/ar_reduce/ar_stamp`），其 GPU 时间**不代表生产**（`nsys-wave1-analysis-framework §1.3`：
   生产 v5 AR = **0.66ms/步**，`dsv41-kernel-inventory-v3 §0-口径1`）。
   ⇒ **27.1% 要么是伪路径放大、要么来自另一个（非 AR-safe）栈——两者都必须在读表前钉死**（§3.3 的 A0 自检）。
   **batched vs lazy 的 AR 对比只能比「轮数/步」，不能比 ms。** 且「每次更大」这一说法**不成立**：
   AR 是 per-layer 的 `dim` 宽 reduce，**与 m 无关**——正确的差异是「每步轮数」（§2.3）。

4. **nsys 必须跑「AR-safe 的等价栈」，代价是 `VERIFY_GRAPH` 会 decline，verify 回落直发。**
   `ar_v5() = GRAPH_STEP ∥ AR_V5`（`tp.rs:1027-1047`），两腿全 0 ⇒ `ar_v5()==false`；
   而 `verify_arm_local` 的 armed 条件含 `comm.is_none() || ar_v5()`（`chain_dev.rs:6371`）⇒ **图不 arm，verify 走 Direct**。
   **这不是缺陷**：`cuda_gpu_kern_sum` 看不见图 replay 的 kernel（`nsys-clean-stack-91 §0-2`），
   decline 成直发后**每个 kernel 才出现在 sum 表里**。SWALLOW 的 `m=6` 块在直发臂下**仍然成形**
   （`swallow_step()` 与 `ar_v5()` 无关，`chain_dev.rs:2697`）⇒ batched 的 name 判据仍然有效。
   ⚠️ **推论**：nsys 下的**绝对 ms ≠ 生产 S0（31ms）**——图只省 submit 半（−1.5ms），且 SWALLOW 的
   `SWALLOW_GRAPH_WARMUP_BLOCKS=3`（`chain_dev.rs:2730`）在 nsys 下恒走 direct。**本框架只读「分布/占比/计数」，不读绝对 ms。**

5. **`kern_sum` 会混进 load + prefill + decode 三段（脚本无 `--capture-range`，整进程抓 + SIGINT 收尾）。**
   batched 计数任务步数更少（accept 5 ⇒ ~17-20 步/100 token），**prefill/load 占比更大** ⇒ 必须做 decode 窗口切分（§3.2）。
   同时 `Med` 是混合 population，**decode 结论只能用窗口/差分**。

6. **三个「mrows 折核」族在 lazy（m=1）下恒为 0；batched（m=6）下才第一次存在（这是 SWALLOW nsys 的核心里程碑）。**
   `GATE_MROWS / INDEXER_MROWS / COMPRESSOR_MROWS / VERIFY_ROPE_MROWS / VERIFY_HEAD_MROWS / NORM_MROWS`
   全是「m 行折成 1 发」，lazy 每行恒 m=1 ⇒ 折核退化成自身 ⇒ 零节省（`swallow-unlocked-next-plan §C3`）。
   ⇒ **batched nsys 的第一产出：这批 `<6>` 模板实例是否真的出现**（硬证据，比 ms 硬）。

7. **本次 profile 要回答的四个问题**（也是唯一产出）：
   (a) SWALLOW batched 的**真实族分布**（与 lazy 的 27.1/19/18.5/7.8 逐族对照）；
   (b) **SH_PAIR M=6 是否上场 + 占比变动**（判据：`gemm_fp8_sh_exp_pair_kernel<6>` 出现、旧 M=1 臂消失）；
   (c) **AR 的轮数/步**（batched 应 ≈ 1/k_emit 行 × lazy）；
   (d) **400 的可削减预算**（从 (a) 的排序推导 31ms→15ms 的 −16ms 从哪来，§6）。

---

## 1. 采集（命令 + gate 串 + AR-safe 处理）

### 1.1 工具与产物
| 项 | 值 | 来源 |
|---|---|---|
| nsys | `/usr/local/cuda-13.2/bin/nsys` | `nsys_wave1.sh:74` |
| 追踪 | `--trace=cuda,nvtx --cuda-graph-trace=node --sample=none` | `nsys-clean-stack-91 §2.2` |
| 二进制 | `./target/release/ferrite-serve --model dsv41 --serve --tp 8` | `nsys_wave1.sh:80` |
| `.so` | `kernels/cuda/libferrite_kernels.so`（`build.sh 103a`；**本机不存在，须远端构建**） | `swallow-unlocked-400-final-path §0-4` |
| 模型 | `/opt/dlami/nvme/models/DeepSeek-V4.1-Flash` | `nsys_wave1.sh:75` |
| 硬上限 | `DUR=300s`（用户 5 分钟限）——到点 SIGINT nsys | `nsys_wave1.sh:150-151` |

### 1.2 GATES（batched/SWALLOW 栈 + SH_PAIR_M + AR-safe pins）——**逐字手写，不裸跑脚本**
```bash
# 基座 = scripts/batched_400_v2.sh:145-156 的 GATES，两处修正：
#   (i)  ★ 补 DSV41_SH_PAIR_M=1（脚本里没有，见 §0-1）
#   (ii) 去 DSV41_V5_LEDGER（观测税，吞吐/分布轮不要；§1.4）
GATES=(
  DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1
  DSV41_EXPERT_ACT_E4M3=1 DSV41_BF16_TRUNCATE=1
  DSV41_SH_EXP_MROWS=1 DSV41_MROWS_SMALL_N_ADAPTIVE=1
  DSV41_GATE_MROWS=1 DSV41_VERIFY_HEAD_MROWS=1
  DSV41_INDEXER_MROWS=1 DSV41_NORM_MROWS=1 DSV41_COMPRESSOR_MROWS=1
  DSV41_DRAFT_GRAPH=1 DSV41_DRAFT_P3A=1
  DSV41_VERIFY_GRAPH=1
  DSV41_SWALLOW_STEP=1
  DSV41_SH_PAIR_M=1              # ★ 本次新增（否则 SH_PAIR 一问作废）
  DSV41_TIMING=1 DSV41_DSPARK_DEBUG=1
)
AR_SAFE=(DSV41_AR_V5=0 DSV41_GRAPH_STEP=0)   # host-barrier AR（nsys 必需，§0-4）
NCCL_SAFE=(NCCL_NVLS_ENABLE=0)
# 禁止：DSV41_LAZY_VERIFY / DSV41_HC_VERIFY_FUSE / DSV41_HC_FRONT_ROWS   （§0-2）
# 幻影：DSV41_OOB_GUARD（树中无此 env）/ DSV41_SWALLOW_EPOCH_PAD 与 DYNAMIC_PAD 同设（over-pad）
```

### 1.3 采集（一条命令）
```bash
NSYS=/usr/local/cuda-13.2/bin/nsys
OUT=/tmp/swallow_batched_nsys
DUR=300 ; PORT=8698
PROMPT='请从 1 数到 200，每个数字单独占一行，只输出数字本身，不要任何解释。'   # accept≈5 ⇒ k_emit≈6
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
( sleep "$DUR"; kill -INT "$NSYS_PID" 2>/dev/null ) &
until grep -q "serving" "$OUT.log"; do sleep 2; done
curl -s -m 260 -X POST "http://localhost:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"dsv41\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":300,\"temperature\":0}" \
  >"$OUT.reply.json"
kill -INT "$NSYS_PID"; sleep 15; kill -9 "$NSYS_PID" 2>/dev/null
pkill -9 -x ferrite-serve 2>/dev/null
```
> **⚠️ lazy 的对照口径**：任务给的 lazy 数（27.1/19/18.5/7.8）来自**另一个臂**（`DSV41_LAZY_VERIFY=1`，
> **不含** `SWALLOW_STEP`）。**两臂必须同 prompt 同 harness 各采一次**，跨栈比 ms 无效（`swallow-unlocked-shpair-m6 §7-9`）。

### 1.4 三张表
```bash
$NSYS stats --report cuda_gpu_kern_sum --format csv "$OUT.nsys-rep" > /tmp/batched_kern_sum.csv
$NSYS stats --report cuda_gpu_trace   --format csv "$OUT.nsys-rep" > /tmp/batched_kern_trace.csv
$NSYS stats --report cuda_api_sum     --format csv "$OUT.nsys-rep" > /tmp/batched_api_sum.csv
```
> `DSV41_V5_LEDGER` 是**观测臂**（10 个同步点/步，canary 4 槽已把 D2H 翻 4×，
> `swallow-unlocked-shpair-m6 §1.3`）：**分布/吞吐轮必须关**，只在单独一轮读 `[v5-ledger]` 行。

### 1.5 活性自检（先于读表，防假表）
```bash
grep -c "\[dsv41\] step pos=" "$OUT.log"          # >0 才有 decode 步
grep -c "\[dspark\] steps="  "$OUT.log"           # >0 才有 verify/draft/commit 分解
grep    "\[verify_graph\]"   "$OUT.log"           # AR-safe 下期望「无 captured 行」（§0-4 预期）
grep -c "ar5-hang"           "$OUT.log"           # 必须 0
wc -l /tmp/batched_kern_sum.csv                    # <5 行 = 没抓到，硬失败
grep -c "gemm_fp8_mrows_kernel<6>\|sh_exp_pair_kernel<6>" /tmp/batched_kern_sum.csv  # >0 = batched m=6 真的上了
```

---

## 2. kernel → 族 映射表（batched m=6 专表）

**通用规则**：nsys `Name` = demangle 后的 `__global__` 函数名（含 template 参数）；
先按子串归类，**模板 `<M>` 里的 M 就是 batched 的行数（本 pass = 6）**；**必须 `--format csv` + python `r[-1]`，禁止 awk 切列**（名字含空格/逗号/尖括号）。

| 族 | batched CSV 里的 kernel 名（子串） | batched 专属 / 与 lazy 的 name 差异 | lazy 侧对应 |
|---|---|---|---|
| **AR（伪）** | `ar_store_kernel` / `ar_reduce_kernel` / `ar_stamp_kernel` / `ar_mark_kernel` / `p2p_ar_*` / `dsv41_hc_post_inplace_kernel` | AR-safe 伪路径；**只数「轮数/步」，不排名**（§3.3） | 同（伪） |
| **共享专家** | **`gemm_fp8_sh_exp_pair_kernel<6>`**（SH_PAIR M=6，本次头号判据） | 旧 M=1 臂应**消失**：`gemm_fp8_sh_pair_kernel`；逐行臂 `gemm_fp8_kernel` | `gemm_fp8_sh_pair_kernel`（M=1）/ `gemm_fp8_sh_exp_pair_kernel<1>` |
| **共享专家 gate** | `gemv_bf16_kernel`（共享 gate + lm_head + engram 的**混合行**） | 与 lazy 同符号；**calls 应 ÷k_emit**（batched 每层 1 发） | 同 |
| **routed experts** | `expert_gemv_fp4_batched_kernel`（gate_up+swiglu）、`expert_gemv_fp4_down_reduce_kernel`、`interleave_gateup_fp4_kernel`、`swiglu_limit_batched_kernel`、`moe_*` | 核不变；**每发吃 m=6 行** ⇒ calls ÷（每步行数/k_emit）但**字节数不变**⇒ 占比可能被动上升 | `expert_gemv_fp4_kernel`（逐行）/ `interleave_gateup_fp4_kernel` |
| **投影 / gemv** | `gemm_fp8_mrows_kernel<6>`、`gemm_fp8_kernel`、`gemm_fp8_swapab_kernel`、`gemm_fp8_wo_pair_kernel`、`wo_a_grouped_gemv_kernel`、`dsv41_gemm_fp8_mx*` | **`<6>` 是本 pass 的硬判据**；`<1>` 实例多 ⇒ mrows 没吃到 | `gemm_fp8_mrows_kernel<1>`（lazy 恒 1）/ `gemm_fp8_gemv_kernel` |
| **gate/route** | `gemv_bf16_nt_kernel`（GATE_MROWS 契约：per-row 累加器）、`gemv_bf16_v2_kernel`（默认）、`moe_route_kernel`/`route_topk_kernel` | lazy 下 `nt` **恒为 0**；batched 应**首次出现** | `gemv_bf16_v2_kernel` |
| **indexer** | `indexer_topk_batched_kernel` / `indexer_score_kernel(_v2)` / `index_k_publish_kernel` / `argmax_xchg_v5_rows_kernel` | INDEXER_MROWS（front 折进 rows）**只在 m>1 存在** | `indexer_topk_kernel` |
| **compressor** | `compressor_fused_mrows_kernel` | lazy 恒为 0；batched 应出现 | `compressor_fused_kernel` |
| **hc 链** | `hc_mixes_rows_kernel` / `hc_mixes_ss_kernel` / `hc_mixes_tail_kernel` / `hc_dots_late_kernel(_kchunk)` / `hc_mix_dots_kernel` / `dsv41_hc_collapse_norm_kernel` / `hc_collapse_kernel` | batched 折 6 行；`hc_*_rows` 与 lazy 同名但 grid/m 不同 | 同（m=1） |
| **head** | `head_gemv_bf16_mrows_kernel`（VERIFY_HEAD_MROWS） | lazy 恒为 0（每行单发） | `gemv_bf16_kernel`（head 形状）/ `head_gemv_bf16_mrows_kernel<1>` |
| **norm/rope** | `rmsnorm_rope_kernel` / `apply_rope_kernel` / `apply_rope_mrows_kernel` / `dsv41_rmsnorm_rows_kernel` | VERIFY_ROPE_MROWS 折核 | 同 |
| **attention** | `sparse_attn_split_kernel` / `sparse_attn_merge_kernel` / `sparse_attn_pf_kernel` / `sparse_attn_orope_kernel` | 未在本轮 gate 里（`ATTN_MROWS` 不在 GATES） | 同 |
| **draft/dspark** | `dspark_markov_head(_sliced)_kernel` / `engram_*` / `dspark_ring/comp_*` | 与 lazy 同 | 同 |

> **生效判据（比 ms 硬，本项目 #1 陷阱是「gate 没进进程」）**：见 §4 每族的「新符号出现 / 旧符号消失」。

---

## 3. 读表口径

### 3.1 CSV 列序（Name 是最后一个字段）
```
cuda_gpu_kern_sum: Time(%), Total Time(ns), Instances, Avg(ns), Med(ns), Min(ns), Max(ns), StdDev(ns), Name
cuda_gpu_trace:    Start(ns), Duration(ns), ..., GridX/Y/Z, BlockX/Y/Z, Stream, Name
cuda_api_sum:      Time(%), Total Time(ns), Num Calls, Avg(ns), Min(ns), Max(ns), StdDev(ns), Name
```
> kernel 名含空格/逗号/`<`/`>` → `python csv` + `r[-1]`。

### 3.2 decode 窗口切分（口径 A，无需重跑）
`sum` 含 load+prefill+decode。batched 计数任务步数少 ⇒ **必须切窗口**：取**最后一次大 kernel（prefill 的 `gemm_fp8_*` / attention 大 grid）之后**为 decode 起点。

```python
import csv, collections
rows=[]
for r in csv.reader(open('/tmp/batched_kern_trace.csv')):
    if len(r)<6 or not r[0].replace('.','',1).isdigit(): continue
    try: s,d=float(r[0]),float(r[1])
    except ValueError: continue
    rows.append((s,d,r[-1].strip()))
t0=float(input('decode window start ns = '))     # 人工确认 prefill 结束点
STEPS=int(input('decode steps = ')); WORLD=8
agg=collections.defaultdict(lambda:[0,0])
for s,d,n in rows:
    if s<t0: continue
    agg[n][0]+=1; agg[n][1]+=d
tot=sum(v[1] for v in agg.values()) or 1
for n,(c,d) in sorted(agg.items(),key=lambda x:-x[1][1])[:30]:
    print(f"{d/tot*100:6.1f}% calls={c:6d} {d/1e6/STEPS/WORLD:8.3f} ms/stp/w  {n[:58]}")
```
**口径 B（更干净）**：`MAXTOK=1` 与 `MAXTOK=300` 各采一次，`decode = many − one`（`dsv41_profile.sh` 权威口径）。

### 3.3 AR 专用口径（三条，缺一会把伪影读成头号瓶颈）
- **A0（自检）**：`grep -c 'ar_store_kernel\|ar_reduce_kernel'` > 0 ⇒ 走的是 host-barrier 伪路径
  ⇒ **AR 行只报「轮数/步」，不能报 ms/占比**。若 sum 表 AR 占比 ≈27%，**这仍是伪影放大**，不是生产。
- **A1（生产口径换算）**：AR 的 ms 取 **0.66ms/步**（v5，`dsv41-kernel-inventory-v3 §0-口径1`），
  与伪路径值分开列账（两栏：「伪路径实测%」+「生产口径 ms」）。
- **A2（轮数对比，唯一可比的量）**：`p2p_ar_*` 的 **Instances / 步**。
  · lazy（m=1 逐行）：每层每行的 verify 都有一轮 ⇒ **轮数 ≈ 40层 × 行数/步**；
  · batched（m=6）：**轮数 ≈ 40层 × 1** ⇒ 应 ≈ `1/行数 × lazy`。
  **协议地板 17.3µs/轮与 m 无关**（`nsys-clean-stack-91 §5`）⇒ AR 的收益 = **减轮数**，不是减体积。
  ⚠️ 任务前提「AR 每次更大」**不成立**，已修正（§2.3/§0-3）。

### 3.4 族聚合
复用 `nsys-wave1-analysis-framework §6` 的族聚合脚本，把族名表换成 §2。

---

## 4. lazy vs SWALLOW：逐族对比表（预期方向 + 机理 + 硬判据）

**基准**：lazy 干净栈（任务给定，同一 harness）：`AR 27.1% / interleave_gateup 19% / gemv 18.5% / hc_dots 7.8%`
（余 ~27% 为共享专家、attention、indexer、compressor、head、draft、norm/rope）。
lazy 步时 ≈33ms、SWALLOW 步时 ≈31ms（工作口径）。

换算到 ms（33ms 基准，**仅供对照，非实测**）：AR ≈8.9 / routed ≈6.3 / gemv ≈6.1 / hc_dots ≈2.6 / 其余 ≈9.1。

| 族 | lazy（m=1 per-row）占比 | **SWALLOW（m=6 batched）预期方向** | 机理 | **硬判据（新符号↑ / 旧符号↓）** | 兑现概率 |
|---|---|---|---|---|---|
| **AR** | 27.1%（#1，**伪路径**） | **轮数 ÷行数 ⇒ 占比大降**；协议地板仍在 | 每步轮数 40×行 → 40×1；per-round 17.3µs 不变 | `p2p_ar_*` Instances/步 ÷行数；伪路径 ms 不排名 | 高（结构性，必发生） |
| **routed experts** | `interleave_gateup` 19% | **占比被动上升或持平**（字节不变、calls ÷） | 同核，每发 6 行；K3 SIMT 核效率不变 | `interleave_gateup_fp4_kernel` calls ÷；`expert_gemv_fp4_batched_kernel` 出现 | 高（但**无净收益**，除非 tcgen05） |
| **共享专家** | ~8-9%（族） | **大幅下降**（若 M=6 真上场） | 逐行链（quant+w1w3+swiglu+barrier+w2+add）×6 → **1 发**；phase-1 grid 9→54 块 | **`gemm_fp8_sh_exp_pair_kernel<6>`↑**、`gemm_fp8_sh_pair_kernel`↓、Rust 无声 decline 只能靠 nsys 取证 | 中（launch 账先例偏负——`SH_EXP_MROWS` 两测零收益） |
| **投影 / gemv** | `gemv 18.5%` | **下降**（mrows 折核首次生效） | `gemm_fp8_mrows_kernel<1>`×行 → `<6>` 一发 | **`gemm_fp8_mrows_kernel<6>`↑**、`<1>` 实例↓ | 中（R2 已在 lazy 吃过一轮；batched 的 `m=6` 是新变量） |
| **gate/route** | 计入 gemv | **新符号出现**（lazy 恒 0） | `gemv_bf16_nt_kernel` per-row 累加器契约 | **`gemv_bf16_nt_kernel` 出现**（lazy 为 0） | 中（兑现率存疑，`§4 C3`） |
| **indexer** | 小 | **新符号出现** | `indexer_front_rows` 折进 `indexer_rows_one` | `indexer_topk_batched_kernel` 出现 | 中 |
| **compressor** | <0.03ms | **新符号出现** | `compressor_fused_mrows_kernel` | 该符号出现（lazy 为 0） | 高（量级不变，判据看符号） |
| **head** | 1.12ms | **下降**（VERIFY_HEAD_MROWS） | 6 行 → 1 发（`chain_dev.rs:5935` sliced-head fold） | `head_gemv_bf16_mrows_kernel` calls ↓；`<1>` 消失 | 中（**历史 ar5-hang 组合，最后单独上**） |
| **norm/rope** | ~0.3ms | 微降 | VERIFY_ROPE_MROWS 折核 | `apply_rope_mrows_kernel` 形态出现 | 高（小） |
| **hc 链** | `hc_dots 7.8%` | **下降**（batched 折行） | per-row → batched rows | `hc_*_rows` 的 grid/m 变化 | 中 |
| **attention** | ~2.8ms | 基本不变（未碰） | — | `sparse_attn_*` calls ≈ 40/层 | — |
| **draft** | ~19%（另一节口径） | 基本不变 | — | `dspark_markov_head_sliced_kernel` | — |

> **三条诚实边界**：
> ① 全部为**设计口径/预期方向**——R2/mrows/SH_PAIR 的 ms 收益仓内**没有 batched 单项实测**；
> ② 占比是**相对值**——某族「占比上升」可能只是别的族变快（routed 就是这种）；
> ③ **SH_PAIR M=6 的收益上限 −4.9~7.9ms 是 launch 账**；nsys 实测该族仅 ~8-9%，
>   按 `−0.8~2.8ms` 编预期（`swallow-unlocked-shpair-m6 §3.3`），**不得按 7.9 编预算**。

---

## 5. 四个分析维度（核算方法）

1. **占比排名（谁最贵）** — `sum` 按 `Total Time` 降序 → 族占比；**先剔 prefill/load 与 AR 伪影**（§3.2/§3.3）。
2. **launch vs 执行** — `Avg(ns)` 是否 ≫ 1.4µs（发核下限）。实测 37.31ms/6224 发 = **5.99µs/发**，
   其中只有 ~1.4µs 是「发核」⇒ **「少发核」上界 ≈ 6224×1.4µs ≈ 8.7ms**（`verify-architecture-floor §1`）。
   batched 的 calls 应整体低于 lazy；**若 calls 降了而 ms 不降 ⇒ exec/延迟 bound，launch 账不买账**（本仓常态）。
3. **kernel 间 gap** — `trace` 相邻 Start 差；gap≈0 ⇒ GPU 忙；gap 大且前驱是 tiny kernel ⇒ 提交/依赖链 ramp。
   **batched 的 gap 分布应整体左移**（calls 少）；不左移 ⇒ 图没吃到（但 AR-safe 下图本就 decline，故本维度只作参考）。
4. **新旧符号并存** — 每族的「生效判据」（§4）。**这是最硬的自检**：gate 没进进程是本项目 #1 陷阱
   （`/proc/<pid>/environ` 实读 + `nm -D` 符号 + nsys name 三证）。

---

## 6. 优化目标推导（31ms → 15ms，−16ms 从哪来）

### 6.1 硬代数
```
400 tok/s ⇒ 步时 ≤ 2.5·k_emit ms     (k_emit = 1 + accept)
accept 5 → k_emit 6 → 步时 ≤ 15.0ms
accept 4 → k_emit 5 → ≤ 12.5ms
accept 3 → k_emit 4 → ≤ 10.0ms
accept 1.214（出师表）→ ≤ 5.54ms  ← 低于 L5 floor 8-9ms，物理不可达
```
**SWALLOW 现状 ~31ms ⇒ 需削减 ≥16ms**（`swallow-unlocked-400-final-path §0-6`）。

### 6.2 可削减预算（从 §4 的 SWALLOW 分布推导的**决策规则**，非固定答案）
```
可削减预算 B_opt = 步时 − AR协议地板(生产 0.66ms) − 已最优核（attention 等未碰族）
```
**推导步骤**（读表后照此填）：
1. 从 §4 表取 SWALLOW 实测占比 → 排序，标出「大项且是核效率问题（非协议地板/非字节问题）」的族；
2. 对每个候选族，查 `l4-l5-kernel-path §1/§2` 拿对应的 L4/L5 项与设计增量；
3. **累加 mid 值**，与 16ms 比：
   · 若 `Σ_candidate < 16ms` ⇒ **400 在「纯 kernel 优化」下不可达**（需 accept↑ 或架构级重写）；
   · 若 `Σ_candidate ≥ 16ms` ⇒ 400 条件可达，但**需按本仓兑现率（历史 60%）打折重算**。

### 6.3 当前候选池（设计口径，待 SWALLOW 实测重排）
| 优先 | 族（lazy 占比） | 对应项 | 设计增量(mid) | 前置/止损 |
|---|---|---:|---|---|
| 1 | **routed experts**（19%） | ① tcgen05 gate/up（`tc5::e4x`）+ ② L4-3 K-split + L4-4 down 换核 | −2.4（①）/ −3.0~5.5（①②合） | tcgen05 已 2 轮修复失败（misaligned，`6ff8058`）⇒ **先修后测** |
| 2 | **投影 / gemv**（18.5%） | L4-1 `mrows` nwarps/crossover + L5-2 gemv 双缓冲 | −0.5~1.4 | L4-1 与 L3（SH_PAIR）在半 expert 上互斥；batched 下 ×k_emit 消失 ⇒ 预期比 lazy 好 |
| 3 | **SH_PAIR / 共享专家**（~8-9%） | SH_PAIR M=6（**本 pass 已在 gate**） | −0.8~2.8（校准） | 见 §4；**这是唯一「已编译、零代码」项** |
| 4 | **hc 链**（hc_dots 7.8%） | HC A1/A2 + L4-7 侧流 + L4-9 dim-split | −1.3~1.7 | A2 的 `bf16_truncate=false` 一行未修 |
| 5 | **mrows 族**（lazy 恒 0） | GATE/INDEXER/ROPE/NORM/COMPRESSOR/HEAD mrows | −4.5~5.8（设计）/ −0~2（实测先例） | 一 gate 一轮；HEAD_MROWS 最后单独上（历史 ar5-hang） |
| 6 | **B6** | `dsv41_gemm_fp8_mrows_f32`（B 类公共祖先） | −0.66~1.5 | 需实现（0.5 人日） |
| 7 | **L4 占用/MLP**（条件） | hc 侧流 + adaptive + tcgen05 K-split + down 换核 | −5~8 | 16~21 人日、**仓内零实测背书**；仅当 S5 >15ms 且 accept≥3 |
| — | **AR** | ⚠️ **不是 kernel 目标** | — | 协议地板（17.3µs/轮）；唯一路 = accept↑（模型/工作负载） |

**关键路径**：`tcgen05 修复 → L4-3 → L4-4`；`SH_PAIR M=6 / mrows / hc / B6` 可并行且不依赖 tcgen05。

### 6.4 判决规则（本框架的最终产出）
- **SWALLOW 的主要削减必须来自「routed + 投影 + hc」三族**（若实测显示它们在 batched 下仍占 ~45%）。
- **AR 若实测占比 >20% 但为伪路径 ⇒ 必须换算生产口径后再判**；若换算后 AR ≤3%，**不得把它当头号目标**。
- **SH_PAIR M=6 的兑现门槛**：nsys 出现 `<6>` **且**该族占比相对 lazy 降 ≥20%（否则 = instruction-bound，止损转 mrows）。
- **若 Σ_candidate < 16ms ⇒ 明确结论「400 不可达于纯 kernel 路线」**（这是本框架最有价值的负结果）。

---

## 7. 陷阱清单（batched 专属，全部有源码/文档理由）

1. **gate 串裸抄脚本 ⇒ 漏 `SH_PAIR_M`**（§0-1）⇒ SH_PAIR 一问作废，且误判「M=6 无效」。**必须手写加门 + 三证**。
2. **误带 `LAZY_VERIFY` 或 `HC_*`** ⇒ 测到 lazy 臂（互斥，脚本 `FORBIDDEN:175`）。
3. **把 AR 伪路径 ms 当生产**（§3.3 A0）⇒ AR 被读成 27% 头号瓶颈；生产口径是 0.66ms。
4. **`kern_sum` 漏图 kernel**（§0-4）：AR-safe 图 decline 恰好规避；但**若日志出现 `[verify_graph] captured`** ⇒ 图没 decline ⇒ sum 表漏 verify kernel ⇒ **本 pass 作废**，改 `DSV41_VERIFY_GRAPH=0` 重跑。
5. **`Med` 含 prefill**：decode 结论只用 §3.2 窗口/差分。
6. **SH_PAIR 的 decline 是无声的**（`shared_expert_mrows` 返回 `Ok(false)` 回落逐行）⇒ 缺 nsys `<6>` 证据不得下「已上场」结论。
7. **M 特化各自 `cudaFuncSetAttribute`**：历史漏设 `<m>` 导致 `cudaInvalidValue`（`dsv41_kernels.cu:7994-8013` 宏已覆盖 1..=8）。
8. **名字含空格/逗号** → 一律 `--format csv` + python（禁 awk）。
9. **口径三件套**：每次比较必须标 `arm + m + timer`（serve 墙钟 / `[dspark] steps=` / nsys），**禁跨栈跨会话比**。
10. **`V5_LEDGER` 是观测臂**（10 个同步点/步）⇒ 分布轮必须关（§1.4）。
11. **nsys 下绝对 ms ≠ 生产 S0**（AR-safe + SWALLOW warmup 恒 direct + 图 decline）⇒ **只读分布/计数/占比**。
12. **`expert_gemv_fp4_*` 在 lazy 与 batched 同名**：必须靠 **grid/行数**（`trace` 表）区分，不能只看 sum 表名。

---

## 8. 判据矩阵（PASS / 失败指向）

| # | 判据 | 通过线 | 失败指向 |
|---|---|---|---|
| B0 | `/proc/<pid>/environ` 含 `SH_PAIR_M=1` 且不含 `LAZY_VERIFY` | 逐门读回一致 | 幻影门/错臂 |
| B1 | `nm -D $SO \| grep -c dsv41_gemm_fp8_sh_exp_fused` | ≥1 | `.so` 未带符号 ⇒ 远端重建 |
| B2 | sum 表出现 `gemm_fp8_sh_exp_pair_kernel<6>` | 出现（非 `<1>`、非旧臂） | SH_PAIR decline 回逐行 |
| B3 | sum 表出现 `gemm_fp8_mrows_kernel<6>` / `gemv_bf16_nt_kernel` / `compressor_fused_mrows_kernel` | 出现 | mrows 族未生效（lazy 残留） |
| B4 | 日志无 `[verify_graph] captured`（AR-safe 预期） | 无 captured 行 | 图未 decline ⇒ 本 pass 作废 |
| B5 | `p2p_ar_*` Instances/步 | ≈ 40（每层 1），< lazy 的 ÷行数 | AR 仍逐行 ⇒ 未走 batched |
| B6 | decode 窗口内族分布（§3.2） | 与 lazy 逐族对照，写清「升/降/中性」 | — |
| B7 | `ar5-hang` 计数 | 0 | 未解锁 ⇒ 回 nograph |
| B8 | `Σ_candidate vs 16ms` | 明确写「可达/不可达」 | — |

---

## 9. 一页纸结论

1. **先修 gate 串**：脚本 GATES **缺 `SH_PAIR_M`**（§0-1）+ AR-safe pins + 去 `V5_LEDGER`；**禁止 `LAZY_VERIFY`**。裸跑脚本会得到「无 SH_PAIR」的假 batched 分布。
2. **nsys 只读分布/计数**：AR-safe 下 `VERIFY_GRAPH` decline、SWALLOW warmup 恒 direct ⇒ **绝对 ms ≠ 生产 31ms**（§0-4/§7-11）。
3. **AR 用「轮数/步」比，不用 ms**：任务前提「AR 每次更大」不成立（AR 与 m 无关）；27.1% 若为伪路径，换算生产口径仅 0.66ms（§3.3/§0-3）。
4. **头号判据是 m=6 符号**：`<6>` 出现 = batched 折核真上场；`<1>` 仍在 = 退直发/未生效（§4/§8）。
5. **优化目标**：400 需 −16ms。候选池设计 Σ≈16.5ms，但本仓历史兑现率 60% ⇒ 现实 ~21.6ms/278 tok/s。**若 SWALLOW 实测的 `Σ_candidate < 16ms`，则「纯 kernel 路线的 400 不可达」是本 pass 最有价值的结论**（§6）。

---

*户部 · 只读勘察 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*所有分布为设计口径；未实测处（`kernels/cuda/*.so` 本机不存在）显式标注；修正处（脚本缺 SH_PAIR_M、AR 与 m 无关、AR-safe 下绝对 ms 不可读）已给出 file:line 依据。*
