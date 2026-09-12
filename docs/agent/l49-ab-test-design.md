# L4-9（CNORM/NORM dim-split）A/B 测试详细设计

> 工部 · 2026-09-12 · **只读勘察 + 本文件（唯一产出）**。未执行任何 GPU 命令、未改动任何源码。
> 现场核对（HEAD `0691d2a`）：`crates/ferrite-models/src/dsv41/{device.rs,chain_dev.rs,dspark_dev.rs}`、
> `kernels/cuda/dsv41_kernels.cu`、`scripts/{sh_pair_ab.sh,tcgen05_smoke.sh,nsys_wave1.sh}`、
> `docs/agent/{batch-reverification-plan.md,r2-reverification-test-design.md,dspark-correctness-chain.md}`。
> 所有 `file:line` 对工作树复核。

---

## 0. TL;DR —— 先纠正 batch plan Phase 2 的 3 处，否则会得出假结论

**一句话**：L4-9 确实是唯一真需 A/B 的剩余项，但 batch plan 的 4 臂矩阵**有一个空臂、缺一个真对照、缺一个噪声地板**。修正后的矩阵是 **7 个 serve**（含 2 个必要的对照/重复臂），判据从 batch plan 的 6 条扩到 **8 条**（新增活性 P7 + 确定性 P8）。

| # | batch plan Phase 2 的原设计 | **现场核对后的修正** | 后果（若照原设计跑） |
|---|---|---|---|
| **C1** | **T1-C** = `DSV41_NORM_SPLIT=1` | **空臂**。`DSV41_NORM_SPLIT` 只 gate `Device::rmsnorm_rows`（`device.rs:5816`），而它的**唯一**调用点是 `ChainDev::norm_rows`（`chain_dev.rs:4407`），该分支要求 `norm_mrows()` = `DSV41_NORM_MROWS`（`chain_dev.rs:1273`，**默认 OFF**）。91.1 栈（batch plan §4）与 `sh_pair_ab.sh:BASE_ENV` **都没有** `DSV41_NORM_MROWS` | 会跑出一个 **0 效果**，误判"L4-9 的 norm 半边无用"，**而 kernel 从没跑过** |
| **C2** | 未给 T1-C 的真对照 | 新增 **T1-C0** = `DSV41_NORM_MROWS=1`（**单独**）。C 必须与 C0 比（单变量），不能与 A 比 | 与 A 比会把 `NORM_MROWS` 入口本身的效果混进 `NORM_SPLIT` 的账 |
| **C3** | 未给噪声地板；"同会话背靠背跑四次取中位" | 新增 **T1-A2** = control 重复（同 env、第二次）。预期各 +0.5%，**没有噪声地板这条数字无法判定** | 无法区分"真 +0.4%"与"噪声 0.4%" |
| 附 | 未给确定性判据 | 新增 **T1-D2** = T1-D 重跑 → 逐字节自比对。kernel 的 phase-2 fold 顺序 **FIXED**（`dsv41_kernels.cu:9959/10030` 的 `for q in 0..nchunks`）⇒ split **确定性**——这条一次就能排除 batch plan §5.3 列为"风险"的 device-global scratch 串扰 | 串扰只能靠"看起来没事"放行 |

---

## 1. 事实清单（现场核对，执行前须再 `git log` 核对时效）

### 1.1 gate 与代码落点

| 项 | 事实 | 出处 |
|---|---|---|
| 精确 gate 名 | `DSV41_CNORM_SPLIT` / `DSV41_NORM_SPLIT`（**默认 OFF**，读到 `!= "0"` 即 ON） | `device.rs:6461` / `:6472` |
| chunk 数覆盖 | `DSV41_NORM_SPLIT_NC`（运行期参，默认 `(dim+1023)/1024`，`clamp(1,16)`） | `device.rs:6482-6491` |
| 形状 guard（Rust 先 decline） | `rows>0 && rows<=256 && nc>=2 && nc<=16 && dim>=nc` | `device.rs:6496` |
| C 侧 sentinel | `DSV41_NORM_SPLIT_DECLINE = 0x7FFF`（故意避开 `cudaErrorInvalidValue=1`） | `dsv41_kernels.cu:9877` / `device.rs:6456` |
| CNORM 落点 | `Device::hc_collapse_norm`（`device.rs:5964`，split 分支 `:5980`） | — |
| NORM 落点 | `Device::rmsnorm_rows`（`device.rs:5803`，split 分支 `:5816`） | — |
| OFF 逐位等价 | OFF 分支发**原 launch 原参数**（`device.rs:5994-5999`）；stale `.so` / decline 同样回退 | — |
| 符号 | `dsv41_hc_collapse_norm_split` / `dsv41_rmsnorm_rows_split`；**默认编入 `.so`**（`build.sh` 无门控） | `dsv41_kernels.cu:9979/10042` |

### 1.2 生产形状（决定 NC 默认值）

- DSV41：`dim = 5120`、`hc_mult = 4`（`config.rs:277`；`hc-chain-bandwidth-analysis.md:42`）
- ⇒ 默认 `NC = (5120+1023)/1024 = 5`，**每 chunk 恰 1024**（= 原 kernel 的 blockDim），无残余尾巴。
- lazy 栈的行数：`rows = 1`（`dspark_spec_lazy` `chain_dev.rs:9029` 走 `lazy_run_row` → `step_rows_sync(1 row)`，`chain_dev.rs:8891`）。
- 原 kernel 在 `rows=1` 时 **grid=1**（`hc_collapse_norm` 的 grid(rows)）⇒ 1 CTA/1 SM —— **L4-9 要修的正是这个**；split 后 grid = `(5, 1)` = 5 CTA。

### 1.3 T1-B 是**真臂**（逐调用点核对 CNORM 的热度）

`hc_collapse_norm` 的调用点：

| 调用点 | rows | 门槛 | 91.1 栈是否在跑 |
|---|---|---|---|
| `layer()` m=1 主链（`chain_dev.rs:13838` / `:13948`） | 1 | `fuse_b1()` = `DSV41_FUSE_B1`（**默认 ON**） | ✅ |
| `step_body` 尾部（`chain_dev.rs:5559`） | 1 | `fuse_b1()` | ✅ |
| `collapse_norm_rows`（`chain_dev.rs:9665`，verify m-row） | m(=1 lazy) | `fuse_b1() && hc_verify_fuse()`（91.1 栈显式 `DSV41_HC_VERIFY_FUSE=1`） | ✅ |
| draft 侧（`dspark_dev.rs:1386` / `:1496`） | bs | `DSV41_DRAFT_P3A=1`（91.1 栈已开） | ✅ |

⇒ lazy 栈里 `hc_collapse_norm(rows=1)` **每步每层都跑**。T1-B 命中的是热路径，不是边角。

### 1.4 T1-C 的空臂机理（C1 的证据链）

`rmsnorm_rows` 的唯一调用点 = `norm_rows`（`chain_dev.rs:4407`，`if norm_mrows() && ...`）。
`norm_rows` 的调用点：`:6735`（verify 块末 norm）、`:9687`（`collapse_norm_rows` 的非 fuse 回退）、`:10360`、`:11413` —— 全部需要 `DSV41_NORM_MROWS=1`。
`sh_pair_ab.sh:BASE_ENV`、batch plan §4 的 91.1 栈、`docs/agent/dspark-correctness-chain.md:2152` 的对照清单里，`DSV41_NORM_MROWS` **都不在**（只在 `batched_400_v2.sh:139` / `full_stack_test.sh:11` 里）。
⇒ **`DSV41_NORM_SPLIT=1` 单独设，等于什么都没改。**

### 1.5 非逐位但**确定性**（决定判据形态）

- **非逐位**：phase 1 per-chunk 用**同一** shfl 树（`nsplit_ss_fold`，`dsv41_kernels.cu:9891`），phase 2 把 nchunks 个 partial **按 chunk 升序**相加；与单块版本的括号化不同 ⇒ `inv` 差 ~1 ULP/行 ⇒ 该行全部输出元素差 ~1 ULP。沿线 argmax 传播。
- **确定性**：fold 顺序 **FIXED**（`for (int q = 0; q < nchunks; ++q)`，`:9959` / `:10030`），ticket 自复位（`:9965` / `:10031`）⇒ **同一 arm 重跑必给逐字节相同输出**。这是 P8 的依据。
- **竞态防线**：`rmsnorm_rows_on`（VERIFY_FORK 的侧流 kv norm）**故意不路由**到 split（`device.rs:6468-6471`）；`hc_collapse_norm` 无 `_on` 孪生 ⇒ 每 Device 同时最多一发 split（`dsv41_kernels.cu:9858-9865` 的 PRECONDITION）。

---

## 2. 臂矩阵（7 serve，严格串行）

**BASE_ENV = 91.1 栈**（batch plan §4，逐字）：

```
NCCL_NVLS_ENABLE=0  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 DSV41_TIMING=1
DSV41_EXPERT_ACT_E4M3=1 DSV41_BF16_TRUNCATE=1
DSV41_LAZY_VERIFY=1 DSV41_VERIFY_GRAPH=1
DSV41_SH_EXP_MROWS=1 DSV41_SH_PAIR_M=1
DSV41_ATTN_LIN_FUSE=1 DSV41_MARKOV_SLICED=1 DSV41_LAZY_SDR=1
DSV41_VERIFY_FORK=1 DSV41_RING_WIN_FUSE=1
DSV41_HC_VERIFY_FUSE=1 DSV41_HC_FRONT_ROWS=1 DSV41_VERIFY_AR_FOLD=1
DSV41_GATE_MROWS=1 DSV41_INDEXER_MROWS=1 DSV41_COMPRESSOR_MROWS=1
DSV41_TAP_INPUT=1 DSV41_DRAFT_BF16_DOMAIN=1 DSV41_DRAFT_P3A=1
```

| # | arm | extra env（**唯一变量**） | 对照 | 目的 | 预期 |
|---|---|---|---|---|---|
| **T1-A** | control | （空） | — | 91.1 栈基线 | 91.1 tok/s |
| **T1-A2** | control×2 | （空，**重复**） | — | **噪声地板 σ** | 与 A 差 = σ |
| **T1-B** | +CNORM | `DSV41_CNORM_SPLIT=1` | T1-A | collapse_norm 切分 | +0.5% |
| **T1-C0** | +NORM_ENTRY | `DSV41_NORM_MROWS=1` | T1-A | rmsnorm_rows 的**真对照**（入口本身） | ~0（bit-exact 口径） |
| **T1-C** | +NORM | `DSV41_NORM_MROWS=1 DSV41_NORM_SPLIT=1` | **T1-C0** | rmsnorm_rows 的 dim-split | +0.5% |
| **T1-D** | +both | `DSV41_CNORM_SPLIT=1 DSV41_NORM_MROWS=1 DSV41_NORM_SPLIT=1` | T1-C0 / T1-B | 组合（单项安全 ≠ 组合安全） | +1% |
| **T1-D2** | T1-D×2 | = T1-D，**重复** | — | **确定性 P8** | 与 D 逐字节同 |

> **口径纪律**：一 serve 一 prompt（`[dspark]` 累加器是 process-level）；arm 间 `pkill -9 -x ferrite-serve`（或 `POST /shutdown`）；每臂 `/proc/<pid>/environ` 实读证明变量真的进了进程。
> **可选第二波**：若 B/C 有正收益但小于预期，sweep `DSV41_NORM_SPLIT_NC ∈ {2,4,8}`（运行期参，不重编）——**每换一个 NC 输出都会变**（仍确定性），P1/P6 须重跑。

---

## 3. 精确测试命令

### Phase 0（无 GPU，~5 min，一次）

```bash
cd ~/ferrite
git log --oneline -10                       # 时效核对（L4-9/tcgen05 可能已被别的 arm 抢先）
bash kernels/cuda/build.sh 103a             # .so
touch crates/ferrite-kernel/build.rs && cargo build --release   # 二进制（双产物同源）
cat kernels/cuda/.build_id                  # 与二进制内嵌 id 一致，否则进程拒启

SO=$HOME/ferrite/kernels/cuda/libferrite_kernels.so
nm -D --defined-only $SO | grep -E 'dsv41_hc_collapse_norm_split|dsv41_rmsnorm_rows_split'
# 期望：两符号都在。缺任一 ⇒ Tier 1 停（构建问题，不是 kernel 问题）。
```

### Phase 1（1 serve，~3 min）——arm 完整性 + EAGER 对照

```bash
# 启动 91.1 栈
pkill -9 -x ferrite-serve; sleep 6
cd ~/ferrite && nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_NVLS_ENABLE=0 \
  LD_LIBRARY_PATH=$HOME/ferrite/kernels/cuda \
  DSV41_KERNELS=$HOME/ferrite/kernels/cuda/libferrite_kernels.so \
  <BASE_ENV 逐字> \
  ./target/release/ferrite-serve --model dsv41 --serve --tp 8 \
  --model-dir /opt/dlami/nvme/models/DeepSeek-V4.1-Flash --port 8699 > ~/l49_p1.log 2>&1 &
# 等 "chain ready, serving"
tr '\0' '\n' < /proc/$(pgrep -x ferrite-serve | head -1)/environ | grep -E '^DSV41_' | sort
```

**审计清单**：

| 检查 | 期望 | 不符的后果 |
|---|---|---|
| `DSV41_CNORM_SPLIT` **不在** | ✅ 对照 | 若在 ⇒ 先去掉 |
| `DSV41_NORM_SPLIT` **不在** | ✅ 对照 | 同上 |
| `DSV41_NORM_MROWS` **不在** | ✅ 对照 | 同上 |
| `DSV41_HC_VERIFY_FUSE=1` 在 | ✅ CNORM 走 fuse 路径（`collapse_norm_rows`） | 若缺 ⇒ 改由 `layer()` m=1 路径承担，仍活 |
| `DSV41_FUSE_B1` 未显式 `=0` | ✅ CNORM 活 | 若 `=0` ⇒ **T1-B 变空臂** |

**P2 的 EAGER 参照**（同会话）：`pkill` 后 `DSV41_SPEC=0 DSV41_DSPARK=0`（其余 BASE_ENV 保留）跑计数 + 出师表 → 记录 `first_bad` 与拉丁出现位置，作为 P2 的"模型退化基线"。

### Phase 2（7 serve）——复用 `sh_pair_ab.sh` 骨架

```bash
# 不新写脚本：把 sh_pair_ab.sh 的三处改成本设计（其余 launch/gen/envchk/teardown/metrics 原样）
#   BASE_ENV   ← §2 的 91.1 栈
#   ARMS       ← (A A2 B C0 C D D2)
#   arm_extra  ← §2 表的 extra env
#   envchk 断言：control 臂 MUST NOT carry CNORM_SPLIT/NORM_SPLIT/NORM_MROWS；
#                B 臂 MUST carry CNORM_SPLIT=1 且 MUST NOT carry NORM_*；
#                C0 臂 MUST carry NORM_MROWS=1 且 MUST NOT carry NORM_SPLIT；
#                C 臂 MUST carry 两者；D/D2 臂 MUST carry 三者。
#   PROMPT_DI  ← "请从 1 数到 1000，每个数字单独占一行，只输出数字本身，不要任何解释。"
#                （sh_pair_ab.sh:145；或本设计的短版 "请从1数到200，每个数字单独一行。"）
#   每臂收尾 POST /shutdown（不是 kill -INT），再 pkill -9 -x ferrite-serve 兜底。
```

**每条 arm 录 3 个产物**：计数文本、出师表文本、`[dsv41] step pos=` / `[dspark] steps=` 日志行。

**P7 活性（每臂可选，但强烈建议至少在 B/C/D 各做一次）**：

```bash
# 复用 nsys_wave1.sh 的 cuda_gpu_kern_sum；只需确认调用数 > 0
nsys stats --report cuda_gpu_kern_sum --format csv /tmp/l49.nsys-rep \
  | grep -E 'hc_collapse_norm_split|rmsnorm_rows_split'
# 期望：B 的 hc_collapse_norm_split_kernel 计数 > 0；C 的 rmsnorm_rows_split_kernel 计数 > 0。
# 这是"arm 真的跑了"的**唯一**硬证据（非逐位但可能输出相同 ⇒ 文本无法证活）。
```

---

## 4. 判据清单（8 条，前 5 条来自 v2 协议，后 3 条为本设计新增）

| 探针 | 口径 | 通过判据 |
|---|---|---|
| **P1 计数** | prompt `请从1数到200，每个数字单独一行。`，`temperature=0`，`max_tokens=1000`，`stream=false`。按行 `strip()` 取非空行 | **前 61 行** `lines[0..61] == ["1".."61"]`；**`first_bad(arm) ≥ max(61, first_bad(control))`**（必须 ≥ 61，且不得早于 control） |
| **P2 出师表** | `请完整背诵《出师表》全文，从先帝创业未半而中道崩殂开始。`，`max_tokens=1000` | **不引入 EXTRA 拉丁**：`latin(arm) ≤ latin(control)` 且 `latin_samples` 无新词（v2 协议：>60 tok 绝对零拉丁不可达成，判据是"不新增"）；前 100 字逐字正确且含 `先帝创业未半` |
| **P3 k_acc** | `[dsv41] step pos=` 的 delta − 1 | 前 10 步序列 md5 与对照同；`|Δmean_k| < 0.05` |
| **P4 吞吐** | `steady_mean`（`[dsv41] step pos=` per-round ms，`STEADY_SKIP=20`）+ `verify_ms`（`[dspark] steps=`） | 只与**同会话**对照比；见 §5 |
| **P5 hang** | 全程 | **0** 次 `ar5-hang`，`/health` 不超时，无 fault |
| **P6 内容**（非逐位 arm 必须） | **首 10 字与同会话 control 逐字相同**（对计数 + 出师表各一次） | `leading_chars ≥ 10` 为 PASS；`0 < leading_chars < 10` 为 WARN（人眼看）；`= 0` 为 FAIL |
| **P7 活性**（新增） | nsys `cuda_gpu_kern_sum` 里 split kernel 调用数 | `> 0`（否则该臂作废，不是"零收益"） |
| **P8 确定性**（新增） | T1-D 与 T1-D2 的计数/出师表输出 md5 | **逐字节相同**（不一致 ⇒ 撞上 device-global scratch 串扰/竞态 ⇒ FAIL） |

### P6 的具体方法（复用现成实现，不重写）

`scripts/tcgen05_smoke.sh:449-476` 已实现 leading-char agreement（`TOKEN_MATCH_CHARS=${TOKEN_MATCH_CHARS:-10}`）。直接复用其 PY 段：对 `(control.txt, arm.txt)` 逐字符比对到首个不同，输出 `prefix_chars` + 两侧上下文。

```bash
python3 - "$CTRL.txt" "$ARM.txt" 10 <<'PY'
import sys
base, arm, need = open(sys.argv[1]).read(), open(sys.argv[2]).read(), int(sys.argv[3])
n = min(len(base), len(arm)); i = 0
while i < n and base[i] == arm[i]: i += 1
print("prefix_chars=%d  identical=%s  base[%d]=%r  arm[%d]=%r"
      % (i, base == arm, i, base[max(0,i-6):i+8], i, arm[max(0,i-6):i+8]))
print("P6: PASS" if i >= need else ("P6: WARN" if i > 0 else "P6: FAIL"))
PY
```

> ⚠️ **为什么不能只查拉丁**：dim-split 的偏差是**数值**的（数字变成另一个数字），字符集上全是 ASCII 数字——拉丁检查会**完全漏掉**。P6 是本设计的核心内容判据，P1 是它的结构化版本。

---

## 5. 决策规则（入栈 / 跳过）

### 5.1 前置门（任一不过，该臂作废，不得进入收益判定）

1. Phase 0 两符号在 + 双产物同源；
2. `/proc environ` 证明该臂的**唯一变量**真的进了进程，且 control 臂不含任何 split/entry gate；
3. **P7 活性**：该臂的 split kernel 调用数 > 0。

### 5.2 每臂的红线（任一 FAIL ⇒ 该臂 OFF，不改默认）

- P1 `first_bad < 61` ⇒ FAIL
- P2 `latin(arm) > latin(control)` 或出现新拉丁样本 ⇒ FAIL
- P5 任何 hang/fault ⇒ FAIL
- P6 `prefix_chars == 0` ⇒ FAIL
- P3 k_acc 漂移 ⇒ FAIL

**P1 的边界读法（非逐位 arm 特有）**：
- `first_bad ≥ 61` 且 `≥ first_bad(control)` ⇒ ✅
- `first_bad ∈ [50, 61)` **且 P8 确定性通过**（重跑一致）⇒ 记为 **WARN**，人工对文本判"是模型放大还是引擎新损坏"；只有确认放大才放行
- `first_bad < 50` 或 **P8 不一致** ⇒ FAIL

### 5.3 收益判定（只有红线全过的臂才判）

先算**噪声地板**（来自 T1-A vs T1-A2）：

```
σ = |steady_mean(T1-A) − steady_mean(T1-A2)| / steady_mean(T1-A)     (百分比)
```

对每个通过红线的 split 臂 X（B / C / D）：

| Δ% = (steady_mean(对照) − steady_mean(X)) / steady_mean(对照) | 判定 | 动作 |
|---|---|---|
| **σ ≥ 1%** | **噪声地板过高** | **直接跳过入栈**（batch plan §8 的出口）——本批边际收益 +0.5% 不可能在这种噪声下被判定 |
| Δ% ≤ −0.3% | 明确负收益 | **OFF**：记录"dim-split 在 m=1 下不划算" |
| −0.3% < Δ% ≤ max(1σ, 0.3%) | 中性 / 噪声内 | **跳过**：gate 保持默认 OFF（不入栈） |
| Δ% > max(1σ, 0.3%) | 真收益 | **入栈**：env-gate 保留，把该 gate 写进 91.1 栈脚本 + 文档，默认 ON |

> 对照的选择：**T1-B vs T1-A**；**T1-C vs T1-C0**；**T1-D vs T1-C0**（若 C0 与 A 已证 ~0，也可 vs T1-A，两数并列给出）。

### 5.4 组合与独立性

- **T1-D 只在 T1-B 与 T1-C 都通过红线时才判**；D 单独 FAIL ⇒ **组合不安全**，只入单独通过的臂。
- T1-C0 自身若红线不过 ⇒ `NORM_MROWS` 引入新问题，**先查 NORM_MROWS**，不要把账算到 L4-9 的 `NORM_SPLIT` 头上（并据此上报）。
- 两个 gate **分别独立判定**：可以"CNORM_SPLIT 入栈、NORM_SPLIT 跳过"。

### 5.5 出口

- **T1-B 与 T1-C 皆跳过** ⇒ **L4-9 判为噪声内/无收益**，gate 全部保持默认 OFF，本项关闭（不再重复测）。
- **至少一个入栈** ⇒ 更新栈脚本 + AGENTS.md 的一行状态 + 本文件的"实测"回填。
- **任一 FAIL** ⇒ 回工部定位（P6 的 `prefix_chars` 给出是"首字即错"还是"后段放大"），**不移动默认**。

---

## 6. 复用清单（不新写脚本）

| 需要 | 复用 |
|---|---|
| serve 启动 / 健康轮询 / `/proc environ` / teardown / metrics 解析 | `scripts/sh_pair_ab.sh` 的 `run_case`（`:535-616`）——只改 `BASE_ENV` / `ARMS` / `arm_extra` / `PROMPT_DI` |
| P6 leading-char agreement | `scripts/tcgen05_smoke.sh` 的 PY 段（`:449-476`），`TOKEN_MATCH_CHARS=10` |
| P7 活性（kernel 调用数） | `scripts/nsys_wave1.sh` 的 `cuda_gpu_kern_sum`（`:195-233`） |
| P1/P2/P3/P4 指标 | `sh_pair_ab.sh` 的 `metrics_of`（steps/mean_k/kacc_md5/steady_mean/verify_ms/latin/md5） |
| 计数 / 出师表 prompt | `sh_pair_ab.sh:145` 的 `PROMPT_DI`（逐字一致，保证与历史对照可比） |

---

## 7. 风险与陷阱（本设计显式规避的）

1. **空臂（最高危）**：`NORM_SPLIT` 单独设 = no-op（§1.4）。执行前必须用 P7 证明 kernel 真的跑了。
2. **噪声 > 信号**：预期各 +0.5%（batch plan §5.3 把握"低"）。**T1-A2 噪声地板是必要组成，不是可选**。
3. **非逐位 ≠ 损坏**：不要用"整段文本逐字相同"当判据。用 **P6 首 10 字 + P8 确定性**。
4. **只查拉丁会假阴性**：数值偏差全是 ASCII 数字，拉丁检查漏掉（§4 的 P6 方法）。
5. **gate 泄漏**：必须 `/proc environ` 实读；control 臂含任何 split gate ⇒ ABORT 该臂，不并入平均。
6. **NC sweep 会改输出**：每换 NC 都要重跑 P1/P6（仍确定性，可与先前的 NC 结果并列但不互比）。
7. **别重跑已验证的 baseline**：T1-A **是** A/B 的对照臂，不是重验 78.8；必须与本轮背靠背（跨会话数字不可比）。
8. **图/时间污染**：P7 的 nsys 与 P4 的计时**不同轮**（nsys 会污染 ms 读数）。

---

## 8. 交付物清单（执行时）

- [ ] Phase 0 的 `nm` 两符号 + `.build_id`
- [ ] Phase 1 的 `/proc environ` 快照 + EAGER 对照的 `first_bad`/拉丁
- [ ] 7 臂 × `/proc environ` 快照（每臂一份）
- [ ] 7 臂 × P1 计数全文 + `first_bad`/`ok` 表
- [ ] 7 臂 × P2 出师表拉丁计数 + 前 100 字 + md5
- [ ] 对照臂 × P3 k_acc 序列（md5）+ 均值
- [ ] 7 臂 × P4 `steady_mean` / `verify_ms` / `tok/s`
- [ ] B/C/D 的 **P6** `prefix_chars` + 两侧上下文
- [ ] B/C/D 的 **P7** nsys kernel 计数
- [ ] D 与 D2 的 **P8** md5 自比对
- [ ] §5 决策表逐行判定 + 结论（入栈 / 跳过 / OFF + 根因方向）

---

*工部 · 只读勘察 + 本文件（唯一产出）；未执行任何 GPU 命令、未改动任何源码。*
*关键修正（T1-C 空臂 / T1-C0 真对照 / T1-A2 噪声地板 / P7 活性 / P8 确定性）均对 HEAD `0691d2a` 现场核对；所有 `file:line` 已复核。*
