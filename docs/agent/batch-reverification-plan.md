# 剩余优化的批量重验计划（BATCH-REVERIFICATION-PLAN）

> 工部 · 2026-09-12 · **只读勘察 + 本文件（唯一产出）**。未执行任何 GPU 命令、未改动任何源码。
> 基线：工作树 HEAD `2330f53`（会话进行中仍在上游提交——见 §0.2 的时效警告）。
> 现场核对：`crates/ferrite-models/src/dsv41/{chain_dev.rs,device.rs,load.rs,weights.rs}`、
> `kernels/cuda/{dsv41_kernels.cu,dsv41_experts_mxf4.cu,dsv41_glue.cu,build.sh}`、
> `scripts/{tcgen05_smoke.sh,sh_pair_ab.sh,nsys_wave1.sh,batched_400_v2.sh}`、
> `docs/agent/{dspark-correctness-chain.md,l4-occupancy-mlp-design.md,tcgen05-retest-after-guardfix.md,session-final-handover.md}`。
> **所有行号均对当前工作树现场核对；推算项已标注口径。**

---

## 0. TL;DR —— 先纠正 5 个前提（任务表已过期）

任务表把 5 项都列为"待重验"，但按当前工作树，**其中 3 项已有定论、1 项已在栈中、只有 2 项真正需要新 GPU**。

| 任务表项 | 声称状态 | **当前事实（HEAD 2330f53 现场核对）** | 结论 |
|---|---|---|---|
| **RING_WIN_FUSE** | 剩余，+0.5% | **✅ 已重验成功**（commit `ea1a87d`）：91.1 tok/s，前 61 行正确 + 零拉丁 | **删除——不需要测** |
| **L4-7（hc 侧流）** | 剩余，+0.5-1%，"需确认是否已含" | **已含**：`DSV41_HC_FRONT_ROWS=1` 就在干净栈 base env 里（`scripts/sh_pair_ab.sh:193`、`scripts/nsys_wave1.sh:94`、最终栈清单 `dspark-correctness-chain.md:4745+`） | **已在栈中——不再是独立项** |
| **wo_a cp.async16** | 剩余，+1%，"已在 .so" | **已内置 `.so`，无 gate**（commit `68cc55e`，`dsv41_kernels.cu:6208-6244`，`dsv41_cp_async16` 12 处） | **无法 env A/B——只能 nsys 确认或双 `.so` 对照** |
| **tcgen05** | 剩余，+3-5%，"需测" | **❌ 已测且失败**（commit `6ff8058`）：`sync: misaligned address`（rank 7），LEN=0，208ms 快速失败；**split body 的对齐 bug** | **不是"重验"，是"修复→重测"** |
| **L4-9 collapse_norm** | 剩余，+1%，"需实施" | **已实施**（commit `2f4fc62`，`device.rs:6461-6490` + `dsv41_kernels.cu:9905-10050`），gate `DSV41_CNORM_SPLIT` / `DSV41_NORM_SPLIT` 默认 OFF | **唯一真正需要 A/B 的项** |

**一句话**：本批的**真正可执行新 GPU 工作 = 2 个 Tier**——**Tier 1（低风险）**：L4-9 的 `CNORM_SPLIT`/`NORM_SPLIT` A/B（预期 +1%，非逐位，需数值+k_acc 判据）；**Tier 2（高风险）**：tcgen05 的 split-body 对齐修复 + 重测（预期 +3-5%，但要先修 bug）。

### 0.1 高效性的三条硬约束（决定"批量"的形态）

1. **一次重建，多 arm 共享**——`.cu` 变了才需要 `build.sh`；env gate 的 A/B 只重启 serve。L4-9 的 kernel **已编入默认 `.so`**（`build.sh` 无门控），所以 Tier 1 只需**一次双产物重建**即可跑全部 arm。
2. **一 serve 一 prompt**——`[dspark]` 累加器是 process-level（`batched_400_v2.sh:20-22`），两 prompt 混平均。arm 之间必须 `pkill -9 -x ferrite-serve`。
3. **arm 完整性用 `/proc/<pid>/environ` 实读**——本项目的 #1 测量偏差陷阱是"gate 没进进程"（`sh_pair_ab.sh` 的 `envchk`、`tcgen05_smoke.sh:289` 已实现）。

### 0.2 时效警告

本文件写于 HEAD `2330f53`。会话进行中上游**仍在新提交**（`ea1a87d` 91.1、`6ff8058` tcgen05 失败、`2330f53` SWALLOW epoch pad 实施都在最近 10 分钟内）。**执行前必须重新 `git log --oneline -10` 核对**——L4-9 / tcgen05 的任何一项可能已被别的 arm 抢先测掉。

---

## 1. 判据（v2 协议，全部 arm 共用）

> 本 session 最重要的范式转移：**计数 line-62 "重置到 12"、出师表 ~100 字拉丁是 base 模型的自然退化，不是引擎 bug**（EAGER+e4m3 对照确认 61/77）。绝对零拉丁在 >60 token 生成下**不可达成**。详见 `dspark-correctness-chain.md` 的 EAGER 对照节 + `r2-reverification-test-design.md` §1。

| 探针 | 口径 | 通过判据 | 来源 |
|---|---|---|---|
| **P1 计数** | prompt `请从1数到200，每个数字单独一行。`，`temperature=0`，`max_tokens=1000` | **仅前 61 行** `lines[0..61] == ["1".."61"]`；`first_bad` 必须 **≥** control 的 `first_bad`（不得提前） | 修正判据 |
| **P2 出师表** | `请完整背诵《出师表》全文，从先帝创业未半而中道崩殂开始。`，`max_tokens=1000` | 前 100 字逐字正确；**退化模式与同会话 EAGER 对照一致**（不要求绝对零拉丁） | v2 协议 |
| **P3 k_acc** | `[dsv41] step pos=` 的 delta 直方图 | 与 control 一致（**不得退化**——MARKOV/LAZY_SDR 的教训：数值中性优化也可能是承重件） | 教训 4 |
| **P4 吞吐** | end-to-end tok/s（含 prefill）**或** `[dspark] steps=` 的 verify_ms | 方向与预期一致；**只与同会话 control 比** | 口径纪律 |
| **P5 hang** | 全程 | **0 ar5-hang** | — |
| **P6 数值** | 仅非逐位 arm（L4-9、tcgen05） | 前 10 字与 control 同（MMA/f32 结合序不同 → 只比前缀，**不比全字节**） | `tcgen05_smoke.sh` §3 |

**EAGER 对照（P2 的前提）**：每批**首轮**必须跑一次纯 EAGER（`DSV41_SPEC=0 DSV41_DSPARK=0`）作为"模型退化基线"，否则 P2 的"一致"无参照。

---

## 2. 批量执行总图

```
Phase 0  无 GPU（一次，~5 min）
  P0.1  git log --oneline -10            → 核对时效（§0.2）
  P0.2  build.sh 103a + cargo build --release  → 双产物同源
  P0.3  nm -D 符号 precheck              → 需要的符号都在 .so
  └─ 通过 ↓
Phase 1  GPU（1 次 serve，~3 min）
  P1.1  启动 91.1 栈 + EAGER 对照（两个 arm）
  P1.2  /proc/<pid>/environ 实读 → 确认"已含项"真的在、未含项真的不在
  └─ 通过 ↓
Phase 2  GPU（Tier 1：L4-9，3 arms，~9 min）     ← 本批唯一的低风险新测
Phase 3  GPU（Tier 2：tcgen05，先修复后 5 arms，~30 min + 修复时间）
Phase 4  离线（wo_a 的 nsys 确认 / 双 .so 对照）
```

**为什么这个顺序**：Tier 1 的 kernel 已在默认 `.so` 里（一次重建后即可），且 gate 默认 OFF、decline 安全，**失败代价最低**；tcgen05 需要先改 `.cu`（重建 + 修复迭代），放在后面不阻塞 Tier 1 的产出。

---

## 3. Phase 0 —— 双产物重建 + 符号审计（无 GPU）

```bash
cd ~/ferrite
bash kernels/cuda/build.sh 103a
touch crates/ferrite-kernel/build.rs && cargo build --release
cat kernels/cuda/.build_id        # 与二进制内嵌的 id 必须一致（否则进程拒绝启动）
```

**符号审计**（Tier 1 + Tier 2 需要，一次做完）：

```bash
SO=$HOME/ferrite/kernels/cuda/libferrite_kernels.so
# Tier 1（L4-9，两个都要）
nm -D --defined-only $SO | grep -E 'dsv41_hc_collapse_norm_split|dsv41_rmsnorm_rows_split'
# 已含项（确认在，不用测）
nm -D --defined-only $SO | grep -E 'dsv41_ring_win_fuse$'
# Tier 2（tcgen05 五符号）
nm -D --defined-only $SO | grep -E 'dsv41_expert_act_e4m3_cap|dsv41_expert_gemm_e4m3_grouped|dsv41_route_group|dsv41_route_gather_rows|dsv41_route_scatter_rows'
```

**预期**：L4-9 两符号在（`build.sh` 无门控，默认编入）；tcgen05 五符号在（`DSV41_BUILD_TCGEN05_E4M3` 默认 ON，`build.sh:113`）。任一缺失 ⇒ **该 Tier 停**，先补 `.so`。

---

## 4. Phase 1 —— arm 完整性审计（1 次 serve，回答"L4-7 是否已含"）

> 这一步成本极低但价值最高：**把"已在栈中"和"待测"钉死**，避免重复测已完成的项（RING_WIN 就是被前提表错误列为"剩余"的典型）。

```bash
# 启动 91.1 栈（= 当前最终干净栈；L4-7 已含）
cd ~/ferrite && pkill -9 -x ferrite-serve; sleep 6
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_NVLS_ENABLE=0 \
  LD_LIBRARY_PATH=$HOME/ferrite/kernels/cuda \
  DSV41_KERNELS=$HOME/ferrite/kernels/cuda/libferrite_kernels.so \
  DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 DSV41_TIMING=1 \
  DSV41_EXPERT_ACT_E4M3=1 DSV41_BF16_TRUNCATE=1 \
  DSV41_LAZY_VERIFY=1 DSV41_VERIFY_GRAPH=1 \
  DSV41_SH_EXP_MROWS=1 DSV41_SH_PAIR_M=1 \
  DSV41_ATTN_LIN_FUSE=1 DSV41_MARKOV_SLICED=1 DSV41_LAZY_SDR=1 \
  DSV41_VERIFY_FORK=1 DSV41_RING_WIN_FUSE=1 \
  DSV41_HC_VERIFY_FUSE=1 DSV41_HC_FRONT_ROWS=1 DSV41_VERIFY_AR_FOLD=1 \
  DSV41_GATE_MROWS=1 DSV41_INDEXER_MROWS=1 DSV41_COMPRESSOR_MROWS=1 \
  DSV41_TAP_INPUT=1 DSV41_DRAFT_BF16_DOMAIN=1 DSV41_DRAFT_P3A=1 \
  ./target/release/ferrite-serve --model dsv41 --serve --tp 8 \
  --model-dir /opt/dlami/nvme/models/DeepSeek-V4.1-Flash --port 8699 \
  > ~/batch_p1.log 2>&1 &
# 等 "chain ready, serving"
tr '\0' '\n' < /proc/$(pgrep -x ferrite-serve | head -1)/environ | grep -E '^DSV41_' | sort
```

**审计清单（逐项打勾）**：

| 检查 | 期望 | 不符的后果 |
|---|---|---|
| `DSV41_HC_FRONT_ROWS=1` 在 | ✅ L4-7 已含 | 若缺 ⇒ L4-7 确实是剩余项，需补测 |
| `DSV41_RING_WIN_FUSE=1` 在 | ✅ 已在 | — |
| `DSV41_CNORM_SPLIT` / `DSV41_NORM_SPLIT` **不在** | ✅ L4-9 是 OFF（本批要 A/B 的对照） | 若在 ⇒ 先关掉再测 |
| `DSV41_EXPERT_TCGEN05_E4M3` / `DSV41_EXPERT_GROUPED` / `DSV41_GATEUP_FUSE=0` / `DSV41_EXPERT_ILV=0` **不在** | ✅ tcgen05 未武装 | — |
| `DSV41_ATTN_LIN_FUSE=1` 且 `DSV41_SH_EXP_MROWS=1` 且 `DSV41_SH_PAIR_M=1` | ✅ | — |

**P2 的 EAGER 对照**（同会话）：`pkill` 后以 `DSV41_SPEC=0 DSV41_DSPARK=0` + 计数/出师表跑一轮，记录 `first_bad` 与拉丁出现位置——作为 Tier 1 的 P2 参照。

---

## 5. Phase 2 —— Tier 1：L4-9（CNORM/NORM dim-split）A/B

### 5.1 为什么是唯一真正的低风险新测

- **已实施**：`dsv41_hc_collapse_norm_split` / `dsv41_rmsnorm_rows_split`（`dsv41_kernels.cu:9905/10000`），Rust 落点 `Device::hc_collapse_norm`（`device.rs:5980`）+ `rmsnorm_rows`。
- **默认 OFF 且 OFF 逐位等价**：`cnorm_split_wanted()` / `norm_split_wanted()`（`device.rs:6461/6472`）默认 false；OFF 分支发原 launch 原参数（`device.rs:5994-5999`），**bit-for-bit 今日路径**。
- **数值契约**：`dim` 切 `nchunks` 段，两阶段归约（per-chunk partial → 末块 fold），**非逐位**（f32 结合序变）→ **必须上 P6**。
- **PRECONDITION 已确认安全**（`dsv41_kernels.cu:9858-9865`）：`[row]` scratch 是 device 全局，同一 Device 的 main stream 一次只发一发 split；`rmsnorm_rows_on`（VERIFY_FORK 的 kv 侧流）**故意不路由**到 split（`device.rs:6468-6471`）⇒ **不与 VERIFY_FORK 侧流竞态**。

### 5.2 臂矩阵（3 arms，逐 arm 重启 serve）

| # | arm | extra env | 目的 | 预期 |
|---|---|---|---|---|
| **T1-A** | control | （空） | 对照（= 91.1 栈），本批同会话基线 | 91.1 |
| **T1-B** | +CNORM | `DSV41_CNORM_SPLIT=1` | hc_collapse_norm 的 dim-split（m=1 的 1-CTA→nc-CTA） | +0.5% |
| **T1-C** | +NORM | `DSV41_NORM_SPLIT=1` | rmsnorm_rows 的 dim-split | +0.5% |
| **T1-D** | +两者（可选） | 两个都 `=1` | 组合（单项安全 ≠ 组合安全） | +1% |

**执行**：复用 `scripts/sh_pair_ab.sh` 的 `launch` / `gen` / `envchk`（改 `BASE_ENV` 为 §4 的 91.1 栈，`arm_extra` 换成上表）——**不新写脚本**。

**每一 arm 的通过清单**（`docs/agent/dspark-correctness-chain.md:3804` 的五条 + P6）：
1. 计数前 61 行正确（P1）
2. 出师表退化模式与 EAGER 对照一致（P2）
3. k_acc 序列不退化（P3）
4. 吞吐 ≥ control（P4）
5. 0 ar5-hang（P5）
6. **首 10 字同 control**（P6，非逐位 arm 必须）

**NC sweep（可选，第二波）**：`DSV41_NORM_SPLIT_NC` 覆盖 chunk 数（默认 `(dim+1023)/1024` = 5）。若 T1-B/C 有效但小于预期，再 sweep `NC ∈ {2,4,8}`——**运行期参，不重编**。

### 5.3 风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| **非逐位** | f32 结合序变，理论上最坏 1 ULP/行 × 40 层 | OFF 逐位等价 ⇒ 只可能是 split 版引入；P6 首 10 字 + P2 退化模式一致即可放行 |
| **device 全局 scratch 串扰** | `g_nsplit_*_ss[row][ck]` 是 `__device__` 全局；两发同 row 的 split 会 cross-talk | PRECONDITION 已由"main-stream-only + 不路由 `_on`"保证（`dsv41_kernels.cu:9858`）；**仍要在 arm 里确认没有其它 `_on` 调用点** |
| **收益落在噪声内** | 预期 −0.1~−0.3ms（`l4-occupancy-mlp-design.md` §2.1 把握"低"）| 同会话背靠背跑 T1-A/B/C/D 四次取中位；单 arm >1% 才值得进栈 |
| **`rows` cap** | `DSV41_NORM_SPLIT_MAXR=256`，lazy m=1 远低于此 | 无 |

---

## 6. Phase 3 —— Tier 2：tcgen05（修复 → 重测，不是"重验"）

### 6.1 现状：已失败，且失败点已被精确定位

- **实测**（commit `6ff8058`）：`sync: misaligned address`（rank 7），LEN=0，208ms 快速失败。
- **根因判词**（`tcgen05-retest-after-guardfix.md` §3.2 + `6ff8058`）：
  - `ld_uint2_a8` 守卫修的是 **pair body**（`pair_body = ((fuse_swiglu != 0) || ILV) && (b_split > 0)`，`dsv41_experts_mxf4.cu:1405`）里的 `uint2` 读（`:1677/:1706/:1708/:1754`）。
  - 而 tcgen05 冒烟臂是 `GATEUP_FUSE=0` + `ILV=0` ⇒ **`pair_body = false`** ⇒ 走 **split body**（`:1873/:1874` `uint32`、`:1943/:1944` `uint16`、`:1946` `float4`）——**守卫零覆盖**。
  - **baseline 从不跑 split body**（默认 `GATEUP_FUSE=1` ⇒ `pair_body=true`）⇒ split body 的 8/4/16-byte 读**首次在真实形状上执行**，对齐 bug 一直潜伏。
- **任务表的三个问题，先给答复**：
  - **Q1（GATEUP_FUSE=0 是否与现有栈冲突）**：冲突是**真实的、且是 load-time 的**。`GATEUP_FUSE=0` 经 `ilv_ok()`（`load.rs:767-778`，合取项 `cd::gateup_fuse()`）**强制 `ilv=false`** ⇒ ROUTED expert 的 w1/w3 从"交错单区"退回"6 个独立平面"。这**不是纯 runtime flag**：它改权重布局，**必须重启 serve**（不能热切）。**副作用**：走 batched expert 的**所有**路径都失去 ILV 的 LDG.128 合并读（实测 ILV 仅 −0.09ms，`tcgen05-retest §2`，所以代价小，但非零）。
  - **Q2（ILV=0 是否影响 SH_PAIR_M）**：**不影响**。ILV 作用于 **routed** expert 的 `w1/w3`（fp4）；SH_PAIR_M 作用于 **shared** expert 的 `shared_w1/w3/w2`（fp8）——**不同 tensor、不同 shard 规则**（`shared_expert_tp()`，`weights.rs:471`）。ILV=0 只改 routed 池的 view 布局，shared expert 链路看不见。唯一共同点是"都要 load-time"。
  - **Q3（RING_WIN_FUSE 与 VERIFY_FORK 的交互）**：**已由实测回答——不冲突**（`ea1a87d`：两 gate 共存，91.1 tok/s，前 61 行 + 零拉丁）。代码上：R3 的融合 launch 在 `attention_rows` 的 **per-row interleave 内**、**VERIFY_FORK 的 join 之后**，且用**主 stream**（`device.rs:4956` 的 `self.stream`）——ring_append 是 `kv_r` 的第一个消费者，join 恰在其前（`chain_dev.rs:1120`），顺序天然正确。

### 6.2 修复前置（必须先做，不能直接重跑）

**嫌疑点（按 `tcgen05-retest-after-guardfix.md` §5/§8 排序）**：

1. **split body 的 byte-wise 读**：`:1873/:1874`（`uint32`，`bp = brow + (g<<8) + (lane<<3)` ⇒ 理论 8B 对齐，需实测验证 `brow` 的基址）、`:1943/:1944`（`uint16`）、`:1946`（`float4`，`s_act + j`，需要 16B 对齐——`extern __shared__ float s_act[]` 的动态基址对齐需确认）。
2. **`e4m3_gemm_grouped_kernel`（tc5::e4x）本身**——"sync: misaligned address"也可能是 **mbarrier/TMEM** 相关（tcgen05 的 mbarrier + smem 需要 8/16B 对齐），而非 SIMT 读。
3. **两条 `[OPEN]` 解码猜测**（dense `idesc` format code / fp4 packed-vs-unpacked）——这两个会导致**静默错值**，不是 misaligned；但修完对齐后必须继续防。

**诊断纪律（照 `tcgen05-retest-after-guardfix.md` §4.3 执行）**：
```
export CUDA_LAUNCH_BLOCKING=1         # 首轮必须：让 fault 归到真正 fault 的 launch
export DSV41_VERIFY_GRAPH=0 DSV41_GRAPH_MOE=0   # 显式关图，去掉 capture 变量
# 首轮建议额外：compute-sanitizer --tool memcheck --launch-timeout 120 ...
```
**注意**：`CUDA_LAUNCH_BLOCKING` / sanitizer 与计时**不同轮**（污染 ms 读数）。

### 6.3 修复后的重测矩阵（5 arms，照 §4.3 的 R0/R1a/R1b/R1c/R2）

| # | env | 跑哪条 body | 目的 |
|---|---|---|---|
| **R0** | 无 tcgen05 门 | pair body（生产默认） | 引擎自证（不 PASS 则全部归因作废） |
| **R1a** | `GATEUP_FUSE=0 ILV=0` | SIMT **split body**（4B 读） | **本次 misaligned 的直接嫌疑**；修好应 PASS |
| **R1b** | `GATEUP_FUSE=1 ILV=0` | pair body（uint2 读） | 守卫受益者，回归确认 |
| **R1c** | `GATEUP_FUSE=1 ILV=1` | pair body（uint4 读） | 生产默认回归确认 |
| **R2** | R1a + `TCGEN05_E4M3=1 EXPERT_GROUPED=1 MOE_BATCH=1` + nsys | grouped tcgen05 | **目标**；正证据 = `e4m3_gemm_grouped_kernel` 计数 > 0 |

**R2 的 PASS 门槛（缺一 = 空洞）**：
1. **L2/L3 正证据**：`nsys stats --report cuda_gpu_kern_sum` 里 `e4m3_gemm_grouped_kernel` **调用数 > 0**（成功是静默的，`dsv41_experts_mxf4.cu:6120` 零打印）。
2. 无 4 条 decline 告警（`tcgen05_smoke.sh:314` 的 exact substring）。
3. 无 fault（`/health` OK + 日志无 `illegal|fault|CUDA error|panic|abort`）。
4. 首 10 字与 R0 一致（MMA 与 SIMT 的 f32 求和次序不同，**只比前缀**）。

**复用现成资产**：`scripts/tcgen05_smoke.sh` **已实现 R0/R1a/R1b/R1c/R2 的 stage 划分 + 符号 precheck + decline 计数 + 拉丁/双字检测**（HEAD 的 `ARM_GATES`/`CTRL_GATES`，`:114-118`）——修复后**直接跑它**，不新写。

### 6.4 风险

| 风险 | 说明 |
|---|---|
| **修一个暴露另一个** | split body 的对齐点有 3 处（uint32/uint16/float4）+ tcgen05 侧 mbarrier/TMEM，可能逐个暴露（SWALLOW 的 8 次修复教训）|
| **GATEUP_FUSE=0 的 load-time 耦合** | 它顺手关掉 ILV（−0.09ms），所以 R2 的收益 = tcgen05 增益 **减去** ILV 的损失；口径上不能把 −0.09ms 算进 tcgen05 |
| **静默错值** | 两条 `[OPEN]`（dense idesc / fp4 packed-vs-unpacked）⇒ 即使不崩也可能错值；P6 + P1 必须过 |
| **与 batched 路径的交互** | tcgen05 grouped 在 `moe_rows`（多行 verify），但当前栈是 **lazy（m=1）**；lazy 走 `step_rows(m=1)` → `moe_rows(m=1)`，仍会到 grouped。**但收益是按 MoE 三件套 38.1% 的 tensor-core 化估的，m=1 下 tile 利用率低**（`l4-occupancy-mlp-design.md` L4-5：1-CTA f8f6f4 的 M 硬件固定 128，m=1 时 ~120× 过量 MMA）⇒ **lazy 下 tcgen05 的收益可能显著低于 +3-5%**（那是 batched 的口径）|

---

## 7. Phase 4 —— wo_a / 已含项的收尾

| 项 | 状态 | 收尾动作 |
|---|---|---|
| **wo_a cp.async16** | 已内置 `.so`（`68cc55e`），**无 gate** | env 无法 A/B。二选一：① **nsys 确认** `wo_a_grouped_gemv_kernel` 的 staging 走 cp.async（kernel 时长/指令数对照）；② **双 `.so` 对照**（`68cc55e^` 编一个旧 `.so` vs HEAD 的）+ 同二进制——成本高，**建议只在怀疑时做** |
| **L4-7（HC_FRONT_ROWS）** | 已在 base | Phase 1 的 `/proc environ` 确认即可；若要量化它的**增量**，需临时 `DSV41_HC_FRONT_ROWS=0` 跑一轮 A/B（可选） |
| **RING_WIN_FUSE** | 已验（91.1） | 无 |
| **其余**（K1/K2 作 R2 备选） | 已验（88.7，比 R2 慢） | 无 |

---

## 8. 汇总：本批的"高效"在哪

1. **砍掉 3 个伪待测项**（RING_WIN 已验、L4-7 已含、wo_a 无 gate）——任务表的 5 项缩到 **2 个真 Tier**。
2. **一次双产物重建**服务全部 Tier 1 arm（L4-9 kernel 默认编入 `.so`）。
3. **复用现成脚本**：`sh_pair_ab.sh`（Tier 1 的 launch/gen/envchk）、`tcgen05_smoke.sh`（Tier 2 的 R0-R2）——**不新写批量脚本**。
4. **Tier 排序按"失败代价"**：L4-9（decline 安全、OFF 逐位、预期小）先跑，不阻塞；tcgen05（要改 `.cu`）后跑。
5. **每 arm 用 `/proc environ` 实读 + 同会话 control**，把"gate 没进进程"和"跨会话漂移"两个已知偏差源都掐掉。

**总 GPU 预算**：Phase 0（0 GPU）→ Phase 1（1 serve）→ Tier 1（3~4 serve）→ Tier 2（5 serve，仅在修复后）。**若 Tier 1 判为噪声内（<1%），可直接跳过其入栈，转 Tier 2。**

---

## 9. 待上报/批准项

1. **tcgen05 的 split-body 对齐修复**是**独立一件工程**（改 `dsv41_experts_mxf4.cu` 的读路径或加守卫），需批准后再动；不要与 Tier 1 混在一个 verdict 里。
2. **若 R1a 复现 misaligned** ⇒ split body 的对齐修复成立；若 R1a PASS 而 R2 崩 ⇒ 归因到 tcgen05 kernel 自身（mbarrier/TMEM），另立修复项。
3. **wo_a 的双 `.so` 对照**（若要做）需要临时编一个 `68cc55e^` 的 `.so`——违反"严禁组合不同版本"的日常纪律，**仅限一次性对照**，需批准。

---

*工部 · 只读勘察 + 本文件（唯一产出）；未执行任何 GPU 命令、未改动任何源码。*
*前提修正（RING_WIN 已验 / L4-7 已含 / wo_a 无 gate / tcgen05 已失败 / L4-9 已实施）均对 HEAD `2330f53` 现场核对；所有 `file:line` 已复核。*
