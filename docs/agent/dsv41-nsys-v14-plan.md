# DSV4.1 v14 最终 nsys 分解方案（会话终态）

**一句话**：v14 = HEAD（`175462d` fold 代码清理）+ `f6d2dde`（sparse-merge 选举折叠，默认 ON）+ `595437d`

> ❌ **2026-09-12 hot-kernel-restore 生效后本计划部分作废**：`f6d2dde`（sparse-merge 选举折叠）已整体回退删除——
> v14 实测它「真中性」（=0/=1 均 6.61ms），但其代码存在性（split kernel +14 运行时参数 + 选举块）给 40-launch/step
> 的热点 kernel 带来 +0.34ms。因此 `sparse_attn_merge_kernel` 的 launch（40 次）**仍存在**，本文件里
> 「merge 行从 CSV 消失」的成功判据不再适用；下采 v14 应以 **无 fold 的两 launch 形态**为基线。
（COMPRESS_FUSE 3→1，默认 ON）。预期步时 **~6.15ms ≈ 162 tok/s**。本文给出**采集命令、口径、v9 基线、
预期 vs 实测对照表模板**，供上机后逐行回填。

> ⚠️ **工作树注意**：当前工作树有未提交的 **RW_FOLD**（ring_win 全折叠进 `rmsnorm_rope_kernel`，
> `chain_dev.rs` / `device.rs` / `dsv41_kernels.cu`）。构建 v14 前先确认要不要带上它——它会改变
> `rmsnorm_rope_kernel` 的签名与节点数，**口径会多一项**。若只想验证 fold 清理 + sparse-merge，
> 建议 `git stash` 后再采集，保持单一变量。

---

## 1. 采集命令（图关 + host-barrier 模式）

```bash
bash scripts/dsv41_profile.sh 30 /tmp/dsv41-prof-v14
```

脚本（`scripts/dsv41_profile.sh:67-89`）内部固定：

| 项 | 值 | 为什么 |
|---|---|---|
| `DSV41_AR_V5=0` | host-barrier AR | 设备侧 v5 的 publish 自旋在 nsys 逐节点追踪下放大 ~300×（实测 240s / 69 步） |
| `DSV41_GRAPH_STEP=0` | 关整步图 | 整步图 ~400 节点，逐节点追踪极贵；**每 kernel 成本在两种模式下相同**（`DSV41_TOKTRACE` 逐位验证） |
| `--trace=cuda --cuda-graph-trace=node --sample=none` | — | 只抓 CUDA |
| 两次采集 | `prof 1 one` + `prof 30 many` | 差分消掉 prefill 的放大项 |

⚠️ **这两项 pin 是 load-bearing**：图开时 `ar_v5()` 被强制为真（`tp.rs:701-702` `graph || env`），
host-barrier 是 host 行为、图里抓不到。**不要去掉**。

**输出**：`/tmp/dsv41-prof-v14/{one,many}.{csv,nsys-rep,log}`。

---

## 2. 提取 kernel 中位数

### 2.1 脚本自带的 decode-only 差分表（权威的 ms/步）

脚本最后直接打印（`dsv41_profile.sh:91-137`）：`decode-only net GPU time`、`share / calls/stp / us/call / ms/step / kernel`。
**这是 decode 净成本的唯一正确口径**（`many − one`，除以 `steps = N-1` 与 `world = 8`）。

### 2.2 从 CSV 提取每核中位数（Med 列）

CSV 列固定：`Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name`
（`STATUS.md:3570` 实证；**Name 是最后一个字段**，kernel 名含空格/逗号，禁止用 awk，用 python csv）。

```bash
python3 - <<'PY'
import csv
def load(p):
    d = {}
    for r in csv.reader(open(p)):
        if len(r) < 6: continue
        try: med = float(r[4]); inst = int(r[2])
        except ValueError: continue
        d[r[-1].strip()] = (inst, med)          # Name 取末字段
    return d
one  = load('/tmp/dsv41-prof-v14/one.csv')       # 1-token：prefill + 1 步
many = load('/tmp/dsv41-prof-v14/many.csv')      # 30-token
N, world = 30, 8
print(f"{'calls/stp':>9} {'med µs':>8} {'name':<60}")
for k, (i2, m2) in sorted(many.items(), key=lambda x: -x[1][1]):
    i1, _ = one.get(k, (0, 0.0))
    di = i2 - i1
    if di <= 0: continue
    print(f"{di/(N-1)/world:9.1f} {m2/1000:8.1f} {k[:60]}")
PY
```

> ⚠️ **Med 列的口径**：nsys 的 `Med (ns)` 是该 kernel **在整个 run 内所有实例**的中位数，**含 prefill 实例**。
> 对 decode 与 prefill 行数差异大的 kernel（如 `gemm_fp8_gemv`、`sparse_attn_*`）它会偏高。
> **decode 净成本一律以 §2.1 的差分为准**；§2.2 的 Med 只用于"这个核对不对得上"的交叉验证。
> 脚本注释（`dsv41_profile.sh:4-8`）明确记过这个坑：`hc_mixes "49% of the step"` 就是平均口径的 artifact。

---

## 3. v9 基线（6.23ms，生产 v5 AR 口径）

来源：`STATUS.md:7273-7279`（v9 汇总）+ `STATUS.md:6920-6935`（v8 精确表，v9 只改了标注项）。

| # | kernel | 次/步 | med µs | ms/步 | 备注 |
|---|---|---|---|---|---|
| 1 | `gemm_fp8_gemv_kernel` | 246 | ~9.5–10.0 | ~2.3–2.4 | 246 次是最大项（35–38%），指令级地板 |
| 2 | `expert_gemv_fp4_batched_kernel`（gateup+swiglu） | 40 | **23.9** | 0.96 | v9 数值 |
| 3 | `expert_gemv_fp4_down_reduce_kernel` | 40 | **17.4** | 0.70 | down fix 后（回归期 23.8） |
| 4 | `hc_mixes_tail_kernel`（EARLY+LATE **聚合**） | 160 | ~7.1 | 1.13 | 80 EARLY + 80 LATE，同一符号聚合 |
| 5 | **AR v5**（store+pubred） | ~246 | — | **~0.65** | profile 里是 host-barrier 伪影，取生产口径 0.65 |
| 6 | `hc_mix_dots_kernel` | 80 | 7.0 | 0.56 | 侧流，藏在主流窗口下 |
| 7 | `sparse_attn_split_kernel` | 40 | ~7.7 | ~0.31 | — |
| 8 | **`sparse_attn_merge_kernel`** | **40** | **5.1** | **~0.20** | ← v14 消失 |
| 9 | `gemv_bf16_v2_kernel`（gate+route 融合） | 48 | 9.2 | 0.44 | — |
| 10 | `rmsnorm_rope_kernel`（NR_FUSE） | 40 | 2.6 | 0.10 | — |
| 11 | `quant_fp4_fused_kernel` | 40 | 1.8 | 0.07 | ← v14 不变（fold 已删，恢复独立） |
| 12 | `compressor_state_kernel` | 3 | 1.7 | 0.005 | ← v14 变为 fused |
| 13 | `compressor_pool_kernel` | 4 | 5.4 | 0.021 | ← v14 变为 fused |
| 14 | `compress_commit_kernel` | 4 | 2.1 | 0.009 | ← v14 变为 fused |
| 15 | `dsv41_hc_post_inplace_kernel` | 80 | 1.9 | 0.15 | **profile artifact**（生产已折进 AR pubred epilogue，`HCPOST_EPI` 默认 ON） |
| 16 | `argmax_kernel` + `argmax_xchg_v5` | 1 | ~59 | 0.06 | 跨 rank 交换 |
| — | 其余（rmsnorm_q / apply_rope / engram / add / window_idxs / ring_append / …） | — | — | ~0.30 | 见 kernel-inventory-v3 §1 |

**v9 合计 ≈ 6.23ms（生产 v5 AR 口径）。**

---

## 4. 预期变化（v14 vs v9）与实测对照表模板

### 4.1 逐项预期

| 变化 | 机制 | 预期 Δ |
|---|---|---|
| **`sparse_attn_merge_kernel` 消失**（40 次 → 0） | `DSV41_SPARSE_MERGE_FOLD`（默认 ON，`f6d2dde`）：split 每组 `(b·m,h)` 的 C 个 chunk block 用 per-group ticket `g_attn_ticket[row][hh]`（`dsv41_kernels.cu:943`）选举最后完成者，由它就地跑 `sparse_attn_merge_body`（`:956`，与 merged kernel 共用同一份体）。merge 只占 8/148 SM ≈5%，5.1µs 大半是固定开销 → 省 40 个 launch/节点 | **−5.1µs × 40 = −0.20ms**（split 侧增加 merge body，净 ≈ −0.15~−0.20ms） |
| **`quant_fp4_fused_kernel` 不变**（40 次） | `175462d` 删除了 quant-fold（含其缺陷 gate）后，fp4 直出回到独立 launch，与 v9 一致 | **0** |
| **`sparse_attn_split_kernel` 次数不变**（40） | merge 折进 split 的 winner block，kernel 数不变，Med 可能微升（多了 merge 尾部） | Med 微升，calls 不变 |
| **compressor 3→1** | `DSV41_COMPRESS_FUSE`（默认 ON，`595437d`）：`compressor_state`+`compressor_pool`+`compress_commit` 三段在 decode 上是单 block 严格链，合成 `compressor_fused_kernel`（`dsv41_kernels.cu:2786`），逐字节不变 | 3 行消失、1 行出现；**ms 量级不变（<0.03ms）**，节点 −2/kv-source 层 |
| **`hc_mixes_tail` 的 fp4 块消失** | fold 清理把 `hc_mixes_tail_kernel` 的 `xq4/xsc4` 参数删除（`STATUS.md:7457` 计划的清理，`175462d` 已执行）。该 fp4 块只活在 fold 实验期（v12/v13） | **EARLY 恢复 ~1.7µs**（回到 `806ec7a` 记录的基线 1.7µs；fold 期被 inflate） |

### 4.2 对照表模板（上机后回填"实测"两列）

| kernel | 次/步(v9) | med µs(v9) | ms/步(v9) | 预期 v14 变化 | 实测 calls | 实测 med µs | 实测 ms/步 | 判定 |
|---|---|---|---|---|---|---|---|---|
| `sparse_attn_merge_kernel` | 40 | 5.1 | 0.20 | **归零**（折进 split） | _ | _ | _ | _ |
| `sparse_attn_split_kernel` | 40 | ~7.7 | ~0.31 | calls 不变，med 微升 | _ | _ | _ | _ |
| `compressor_fused_kernel` | — | — | — | **新出现**（~3–4 次） | _ | _ | _ | _ |
| `compressor_state_kernel` | 3 | 1.7 | 0.005 | **归零** | _ | _ | _ | _ |
| `compressor_pool_kernel` | 4 | 5.4 | 0.021 | **归零** | _ | _ | _ | _ |
| `compress_commit_kernel` | 4 | 2.1 | 0.009 | **归零** | _ | _ | _ | _ |
| `quant_fp4_fused_kernel` | 40 | 1.8 | 0.07 | **不变** | _ | _ | _ | _ |
| `hc_mixes_tail_kernel`（聚合） | 160 | ~7.1 | 1.13 | EARLY 侧恢复 ~1.7µs（聚合 Med 可能下移） | _ | _ | _ | _ |
| `gemm_fp8_gemv_kernel` | 246 | ~9.5 | ~2.3 | 不变 | _ | _ | _ | _ |
| `expert_gemv_fp4_batched_kernel` | 40 | 23.9 | 0.96 | 不变 | _ | _ | _ | _ |
| `expert_gemv_fp4_down_reduce_kernel` | 40 | 17.4 | 0.70 | 不变 | _ | _ | _ | _ |
| `hc_mix_dots_kernel` | 80 | 7.0 | 0.56 | 不变 | _ | _ | _ | _ |
| `gemv_bf16_v2_kernel` | 48 | 9.2 | 0.44 | 不变 | _ | _ | _ | _ |
| `rmsnorm_rope_kernel` | 40 | 2.6 | 0.10 | 不变（除非带 RW_FOLD） | _ | _ | _ | _ |
| **脚本合计（decode-only net）** | — | — | **6.23** | **~6.15** | — | — | _ | _ |

### 4.3 判定标准

- **sparse-merge 成功**：`sparse_attn_merge_kernel` 行**从 CSV 中消失**（或 calls=0），
  且四段文本全对（bit-exact 的依据是两条路径都跑 `sparse_attn_merge_body`，
  `tests_dsv41_sparse_pfsplit.cu` 的 `REF=<label>` 臂可对拍 fold ON/OFF）。
- **compress-fuse 成功**：`compressor_fused_kernel` 出现、`compress_commit_kernel` 行消失
  （注意 `compress_commit_kernel` 仍存在于 `dsv41_glue.cu:658`，只是 fused 分支不再调它）。
- **总量**：脚本打印的 `decode-only net GPU time` 应落在 **6.10–6.20ms**。若 >6.25ms，回看
  `sparse_attn_split_kernel` 的 Med 是否被 merge 尾部推高（预期 +0.5~2µs/call，不该是 +5µs）。

---

## 5. 口径陷阱（采集前必读）

1. **AR 行是 host-barrier 伪影**：`ar_reduce/ar_store/ar_stamp` 是 `AR_V5=0` 路径的产物。
   生产 v5 绝对值**测不到**（自旋在 nsys 下 300× 病态）。本方案沿用约定取 **0.65ms**，
   并靠 serve 实测反证（v9 的 0.65 与 serve 吻合）。
2. **Med 含 prefill**：见 §2.2。decode 结论用 §2.1 差分。
3. **`hc_mixes_tail` 是一个聚合行**：EARLY 与 LATE 是同符号不同 `mode` 参数，
   `cuda_gpu_kern_sum` 按名字聚合 → 160 次一行。**"EARLY 恢复 1.7µs"无法直接从 CSV 读出**，
   需要 `cuda_gpu_trace` 或按 grid/block 拆实例才能分离。把它当"聚合 Med 是否下移"的弱信号。
4. **`dsv41_hc_post_inplace_kernel` 是 profile artifact**：生产默认 `HCPOST_EPI=ON` 已折进 AR pubred，
   profile 里还出现是 `AR_V5=0` 关掉了 v5 路径的副产物。
5. **`sparse_attn_split_kernel` 的 winner 分支**：fold 后由最后一个 chunk block 跑 merge body，
   winner 读完最后一块后 `atomicExch` 清零 ticket（graph replay 复用）。若 Med 显著上升，
   先查是不是 C（`DSV41_SPARSE_SPLIT` 默认 4）与 `nwarp` 的匹配（merge epilogue 只折 4 warp，
   `sh_acc[4][512]` 有静默截断）。
6. **空 CSV 会静默打印 0.0ms**：脚本对 `rows < 5` 硬失败（`dsv41_profile.sh:80-85`），
   但若 CSV 非空却全是 0 行，先看 `*.log` 里 binary 是否真的跑了。
7. **名字含空格/逗号** → 一律 `--format csv` + python，禁用 table+awk。
8. **`gemv_bf16_v2_kernel` ≠ `gemv_bf16_fp8x2_kernel`**（2026-09-12 澄清，防误读）：
   `ferrite_kernels.cu:2880` 的 `gemv_bf16_v2_kernel` 是**纯 bf16** 的 gate GEMV v2，
   含 route 融合 epilogue（`gv2_route_epilogue`，`ferrite_kernels.cu:2788`），
   入口 `ferrite_gemv_bf16_v2`（:2955）/ `ferrite_gemv_bf16_v2_route`（:2990）。
   它**不含任何 fp8**。`gemv_bf16_fp8x2_kernel`（`dsv41_kernels.cu:5954`，入口
   `dsv41_gemm_bf16_fp8x2`）才是 bf16 gate + fp8 共享专家 w1/w3 的混合核，由
   `DSV41_MIX_GATE` 门控——**默认 OFF**（`chain_dev.rs:540-543`，round-25 定案 9.38 vs 9.71ms），
   在 HEAD 上**调用 0 次**。
   §3 表 #9 的 48 次 = **40 次 MoE router gate（n=384, k=5120, WPR=8, 384 blocks,
   route 融合） + 8 次 idx_weights（`chain_dev.rs:3758` `lin_bf16` → `dev.gemv_bf16`
   → `gemv_bf16_v2_wanted(32)` 命中，n=32, k=5120）**。两者同符号却不同 grid 形状 ⇒
   这一行的 Med 是**混合 population**（同 §5.3 的 `hc_mixes_tail` 陷阱）。

---

## 6. 失败/回退预案

| 现象 | 首查 |
|---|---|
| `sparse_attn_merge_kernel` 仍在 CSV | `DSV41_SPARSE_MERGE_FOLD` 是否被 env 覆盖为 0；split 分支是否命中（`split_c>0 && b*m<=8 && h<=64`）|
| `compressor_fused_kernel` 未出现 | `compress_fuse()` 的 shape gate：仅 decode（`b==seqlen==1 && ratio>1 && pos>0`）且 `.so` 带 `dsv41_compressor_fused` 符号 |
| 总时 >6.3ms | 先回退到 `DSV41_SPARSE_MERGE_FOLD=0` / `DSV41_COMPRESS_FUSE=0` 单变量二分 |
| 文本错乱 | 跑 `tests_dsv41_sparse_pfsplit.cu` 的 `REF` 对拍（fold 的位级一致性判据） |
