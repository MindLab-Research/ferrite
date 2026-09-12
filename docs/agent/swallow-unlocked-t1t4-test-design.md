# SWALLOW 完全解锁后的第一次完整测试设计 —— T0–T4（完整命令 + 判据 + 预期）

> 工部 · 2026-09-12 · **只读勘察 + 设计；未改动任何源码、未执行 GPU 命令。**
> 输入（现场核对）：`crates/ferrite-dsv41/src/serve.rs`、`crates/ferrite-dsv41/src/bin/dsv41-run.rs`、
> `crates/ferrite-models/src/dsv41/{tp.rs,chain_dev.rs,config.rs}`、`scripts/batched_400_v2.sh`、
> `docs/agent/{oob-fix-verification-test-design.md,oob-fix-result-analysis-framework.md,swallow-1token-fix-design.md,swallow-result-action-plan.md,dspark-correctness-chain.md}`。
> 代码基线：HEAD `249c1b2`（`serve: THE ENGRAM GATHER SLOT FIX`）。行号以该树为准；无法从源码定论的推断标 `[未验证]` + 证伪条件。
> **本文件的职责**：设计「engram slot 修复落地后」的**第一次**完整 SWALLOW 测试（T1–T4），给出每步命令、判据、预期与判读树。它**续接** `oob-fix-verification-test-design.md`（T0–T4 骨架）与 `oob-fix-result-analysis-framework.md`（A–H / R0–R7），并**修正**两文档中已被本次修复推翻的前提。

---

## 0. 六条前提校正（不先钉死，整支测试会测错东西）

前两份文档写作于 OOB 修复验证期（`f7cc53b`/`8365345`）。此后发生了 **engram slot 修复**（`249c1b2`），
它改变了「第一次 swallowed 步能跑到哪里」。以下六条是**现场 grep 出来的事实**，其中 P2/P3/P5 与前两份文档的当前文字冲突，**以本节为准**。

### P1 —— 修复形态 = **只改 `.rs`**，双产物只重编二进制

```bash
git show --stat 249c1b2 | tail -3      # 只有 docs/…，实现落在 serve.rs + dsv41-run.rs（.rs）
grep -n "eng_rows\|eng_cols" crates/ferrite-dsv41/src/serve.rs crates/ferrite-dsv41/src/bin/dsv41-run.rs
# serve.rs:410-413   dsv41-run.rs:362-366
```

修复内容（两处 `Collective::new` 生产构造点**各一份**）：

```rust
let eng_cols = cfg.engram_max_ngram_size.saturating_sub(1) * cfg.engram_n_heads;   // (4-1)*8 = 24
let eng_rows = VERIFY_ROWS * eng_cols * cfg.engram_head_dim;                        // 6*24*256 = 36864 f32
let ar_bytes = (hc_dim.max(VERIFY_ROWS * cfg.dim).max(eng_rows)) * 4;              // 36864*4 = 147456 B
```

⇒ **T0 只 `cargo build --release`（不跑 `kernels/cuda/build.sh`）**。理由：kernel 侧 stride 是**运行期参数**
（`tp.rs:581/676/904/918/960/1003` 全是 `self.bytes / 4`），slot 变宽不需要重编 `.cu`。
**但仍要跑 §2.1 的同源证明**——除非拿到的二进制与 `.so` 的 `.build_id` 对得上，否则「只改 .rs」是**假设**，不是事实。

### P2 —— **上一次的绿只覆盖到 engram AR 为止**（这条最重要）

上一次「OOB 修复验证」（`a418f8c`：CANARY=0 / GUARD=0 / RESET=0 / ar5-hang=0）跑的那次请求，
**在第一次 swallowed 步的 engram AR 处 panic 了**（`check_payload`：`payload 147456 > slot 122880`）。
`check_payload` 是 `assert!`（`tp.rs:596-603`），它把 rank pool 线程 **panic 掉**——

⇒ **第一次 swallowed 步只执行到 engram AR 就死了**。它**之后**的一切
（`dspark_commit` → `carry_kept_tap` → `note_ctx_rows` → `inv_ids` → 后续层 / 后续步）
**在「已验证」的那次运行里从未执行过**。

**这决定了 T1 的定位**：修复后步骤会**穿过**原来 panic 的那堵墙，进入一段**零既往证据的新可达区**
（正是 `swallow-1token-fix-design.md §1.5 的新增面` + `§2.A 的 E1–E6`）。
因此 T1 的主判据不只是「旧的四条还绿」，还要有**一条结构性证据证明步骤确实越过了旧墙**（见 §3.2 C0）。

### P3 —— 1-token 的机制随修复变了（**测量通道随之改变**）

- **修复前**：engram AR panic → `[dsv41] rank … spec step err … POISONING`（`serve.rs:624`）或直接 panic 文本
  `collective payload 147456 > slot 122880`（`tp.rs:599`）。
- **修复后若仍有别的 panic**：rank pool 是一个线程（`serve.rs:182` `"dsv41-tp-pool"`），panic 会**杀掉整个 pool 线程**
  → `res_tx` 发送端被 drop → `recv_timeout` 立即返回 **Disconnected** → `"tp pool: a rank did not answer (wedged in a collective?)"`
  （`serve.rs:257`）→ `SingleFlight::fail` retire → `driver` 报 `Length` + 1 token（`swallow-1token-fix-design.md §1.1-1.3`）。

⇒ **T1 判「无 panic」必须同时 grep 三条通道**（缺一会漏）：
`collective payload`（panic 文本）/ `did not answer`（pool 线程亡）/ `spec step err … POISONING`（arm `Err` 冒泡）。
**「LEN=1」本身不是根因，只是这三条中任一条的下游签名。**

### P4 —— gate 串必须**显式写**，不能直接跑 `batched_400_v2.sh`

脚本 `GATES` 当前仍是 `DSV41_SWALLOW_EPOCH_PAD=1`（`scripts/batched_400_v2.sh:154`），**不是** `DYNAMIC_PAD`。
而 `oob-fix-verification-test-design.md §0.2` 已证：**常数 pad + 动态 pad = over-pad ⇒ 触发新的 ar5-hang**。
⇒ T1–T4 的 gate 串**逐字手写**（§3.1），**不设** `DSV41_SWALLOW_EPOCH_PAD`；
`batched_400_v2.sh` 的正式化（`EPOCH_PAD → DYNAMIC_PAD`）是**独立交付项**（§8-1），**不在本次 T1 前做**（避免同时动测试与实现）。

### P5 —— T3 的 baseline 是 **lazy 干净栈的 91.1**，而当前 batched 票面是 **≈50–65 tok/s**（**别按「>91.1」判红**）

- `AGENTS.md:90`：91.1 tok/s = base + R2 + MARKOV + VERIFY_FORK + RING_WIN（**lazy 栈**）。
- `swallow-result-action-plan.md §0/§2.3`：本轮 **batched**（SWALLOW + m=6）票面
  `verify ≈33–36ms`、`步时 ≈37–40ms`、`mean_k ≈1.02–1.2`、`tok/s ≈50–65`；
  **lazy 今天更快（22.56ms vs batched ~39ms），但那是 accept 低到 lazy 只跑 2 行的假象**。

⇒ **T3 的主判据不是「tok/s > 91.1」**（那是跨栈比，苹果对橘子）。主判据是
**同栈、同会话、只换一个 arm 的 step-time / verify_ms 对比**，且必须**在 matched accept 下比**（§5）。
`> 91.1` 只作**参考陈述**，不作通过线。

### P6 —— `SH_PAIR_M` 是**编译期模板** + 有 Wave-1 冲突，T1 主体**不默认带它**

- `DSV41_SH_PAIR_M=1` 走 `template<M>`（`chain_dev.rs:1468` `fn sh_pair_m()`，读 env 一次缓存），要求 **`DSV41_SH_EXP_FUSED=1` 且 `DSV41_SH_PAIR=1`**
  且 `.so` 真的带 `dsv41_gemm_fp8_sh_pair` 符号（`supports_sh_pair()`），三者缺一则 gate 空转（幻影门）。
- `dspark-correctness-chain.md:2160`：**`VERIFY_HEAD_MROWS` 与 SWALLOW m=6 历史上出过 ar5-hang**，
  Wave 1 明确「先不开 `VERIFY_HEAD_MROWS`」。

⇒ **T1 主体 = SWALLOW + DYNAMIC_PAD + V5_LEDGER + 全 mrows（除 `VERIFY_HEAD_MROWS`）+ 图化 + 红线门**；
`SH_PAIR_M` 作**独立 arm S**（§3.4），在 T1 全绿后单跑，**且必须先用 `.so` 符号 + `[sh_pair]` 日志确认它真上场**。

### P7 —— RESET 在**当前代码**上是「活判据」，不是「已保证为 0」

`chain_dev.rs:9722-9730`：probe 对 pre+note **合流**做同-rank 下降断言；
`21e8a8a`（祖先）明确：「若 OOB 仍在，**pos=15 的 note 会打 RESET**（54 < 999）」。
⇒ 修复后步骤能走到 pos=15 的 note，**这恰好是 RESET 唯一能现形的地方**。
「RESET=0」是**要重新赢得的结论**，不是继承的结论——**把它当主判据，不当背景**。

### P8 —— 新的 slot 与 engram 载荷**恰好贴合（==）**，是「零余量边界」

新 slot = `max(20480, 30720, 36864) × 4 = 147456`，engram 载荷 = `6×24×256×4 = 147456`。
`check_payload` 用 `len <= self.bytes` ⇒ **相等成立**，但**余量为 0**。
⇒ 任何几何变动（`VERIFY_ROWS 6→7`、`engram_max_ngram_size 4→5`、`engram_n_heads 8→9`）都会立刻再触发 panic。
**T0.2 必须做一个「载荷预算推导」断言**，把这条边界钉进测试（§2.2）。

---

## 1. 测试矩阵总览

| 测试 | 配置（相对 T1 主臂） | 主判据（全过才 ✓） | 预期 |
|---|---|---|---|
| **T0** | 前置门：`cargo check` + `cargo build`（.rs-only）+ 同源证明 + 载荷预算断言 + **可失败对照** | §2 | 编译/同源/边界/可证伪四关 |
| **T1** | SWALLOW + DYNAMIC_PAD + V5_LEDGER + 全 mrows(-VERIFY_HEAD_MROWS) + 图化 + 计数 1→50 | §3.2 的 C0–C8 | **穿越旧墙 + OOB 效果保持** |
| **T2** | T1 完全同臂，只换 prompt = 计数 1→200 | 前 61 行精确 + 不早停 | SWALLOW 正常生成 |
| **T3** | T1 主臂 **去掉** `V5_LEDGER`（观测关）+ 出师表计时 | step-time/verify_ms 同栈对比 + matched accept | batched weight-sharing 的步时 |
| **T4** | T1 主臂 + 出师表 | 前 ~100 字零拉丁 + 前缀 + 无双字 | 红线 |
| **S** | T1 主臂 + `SH_PAIR_M=1`（+`SH_EXP_FUSED/SH_PAIR`） | §3.4 上场确认 + 红线 | 独立 arm（可选） |

**一次 serve 一条 gate 串**；串不许中途加、不许临场 export（`batched_400_v2.sh:183` 的纪律）。
T2/T3/T4 **背靠背**用同一二进制、同一 `.so`（T3 可直接复用 T4 的出师表 run，见 §5）。

---

## 2. T0 —— 前置门（编译 + 同源 + 边界 + 可证伪）

### 2.0 本地硬门禁（0 GPU，先跑）

```bash
cd /home/smith/src/ferrite && cargo check --workspace; echo "EXIT=$?"      # 必须 0
cargo test -p ferrite-types spec_step 2>&1 | tail -3                        # accept 边界（若无 GPU 用 --no-run）
```

### 2.1 双产物：**只 cargo build**（.rs-only），但同源证明照做

```bash
# 远端只编二进制（不跑 build.sh）——engram 修复是 .rs
ssh -o BatchMode=yes ubuntu@43.202.208.136 'cd ~/ferrite && git fetch -q origin && git reset -q --hard origin/main && \
  source ~/.cargo/env && cargo build --release'

# 同源证明：二进制必须内嵌当前 .so 的 .build_id（防「.so 是旧的」这类不可见不同源）
ssh ubuntu@43.202.208.136 'ID=$(cat ~/ferrite/kernels/cuda/.build_id); echo "build_id=$ID"; \
  echo "embed=$(strings ~/ferrite/target/release/ferrite-serve | grep -cF -- "$ID")  (必须 >=1)"; \
  md5sum ~/ferrite/kernels/cuda/libferrite_kernels.so; \
  ls -l --time-style=+%s ~/ferrite/kernels/cuda/libferrite_kernels.so ~/ferrite/target/release/ferrite-serve'
```

**判据**：`embed >= 1`；记录 `.so` 与二进制的 md5/mtime。
> ⚠️ 若实施者**顺手**改了 `.cu`（例如为解决 engram 而改 gather kernel），则 §2.1 **必须**加 `cd kernels/cuda && bash build.sh 103a`。
> 判据只看一件事：**`.cu` 变 ⟺ `.so` 必须重编**。`build.sh` 末句无 skeleton flag 时会 rc=1 却已成功——**成功判据是日志里 `built …libferrite_kernels.so for sm_103a`**，不是 rc（`batched_400_v2.sh:253-262`）。

### 2.2 载荷预算断言（**P8 的边界钉死**，0 GPU）

```bash
python3 - <<'PY'
import json, glob, re
# 从 config 取几何（engram_max_ngram_size=4, engram_n_heads=8, engram_head_dim=256, dim=5120）
cfg = json.load(open(glob.glob('/opt/dlami/nvme/models/DeepSeek-V4.1-Flash/config.json')[0]))
g = lambda k, d=None: cfg.get(k, d)
dim = g('hidden_size') or g('dim')
VERIFY_ROWS = 6                                   # chain_dev.rs:84
hc_dim = g('hc_mult', 1) * dim                    # serve.rs:394
ng, nh, ehd = g('engram_max_ngram_size'), g('engram_n_heads'), g('engram_head_dim')
n_cols = (ng - 1) * nh
pay = {'hc': hc_dim, 'verify': VERIFY_ROWS * dim, 'engram': VERIFY_ROWS * n_cols * ehd}
slot_f = max(pay.values())
print(f"payloads(f32) = {pay}")
print(f"slot = {slot_f*4} B   (payload engram = {pay['engram']*4} B)")
print("BOUNDARY", "OK(exact)" if slot_f == pay['engram'] else f"NOTE: engram is NOT the max — recompute")
print("FITS", "OK" if pay['engram'] <= slot_f else "FAIL")
PY
```

**判据**：`slot == 147456`、`engram payload == 147456`、`FITS OK`。
> 若这条**对不上**（config 与 `serve.rs` 的推导不一致）⇒ **先停**，说明 slot 计算与 config 脱钩，T1 必然再 panic。

### 2.3 可失败对照（**必须能红**）—— 本次用 **E1** 做「harness 能看见 fault」的证明

OOB 修复**无 gate**（`oob-fix-result-analysis-framework.md §0-C1`：`grep -rn OOB_GUARD crates kernels` = 0 行），
所以旧文档 §2.3 的「`OOB_GUARD=0` 负对照」**不可做**；而「归档 pre-fix 日志」只证明史上红过，**弱证伪**。

**本次有一条更强的、在树内可红的对照**：`E1`（`swallow-1token-fix-design.md §2.A-E1`）——
`DSV41_INV_CHECK=1` 且 `DSV41_SIDS_WRITEBACK=0` ⇒ swallowed 臂 `inv_ids` **必然**返回 `Err`
⇒ 请求走 §P3 的同一条 fault 路 ⇒ `LEN=1` + `spec step err … POISONING`。

```bash
# 与 T1 主臂逐字相同，只把 DSV41_SIDS_WRITEBACK=1 改成 0（其余不动），MAXTOK=64
grep -c 'spec step err'   $NEG_LOG     # 必须 > 0
grep -c 'arm=swallowed'   $NEG_LOG     # 期望：有 pre 无 note（结构性证据）
```

**判据**：负对照**必须红**（出现 `spec step err` / `did not answer`）。
若**不复现** ⇒ **测试 harness 看不见 fault** ⇒ 先修 harness 再跑 T1（铁律：不会失败的测试 = 零信息）。
> 若实施者认为改 `SIDS_WRITEBACK` 有副作用（影响真实路径），退回到「用归档 pre-fix 日志做弱证伪」并在报告里**显式标注弱证伪**。

---

## 3. T1 —— SWALLOW 基础验证（主臂，计数 1→50）

### 3.0 T1 定位（P2 的落点）

T1 要证 **两件事同时成立**：
1. **穿越旧墙**：第一次 swallowed 步**跑完**（不再在 engram AR 处 panic）——**新增判据 C0**；
2. **OOB 效果保持**：旧文档的主判据 A–D（canary / guard / RESET / epoch / ar5-hang）**依旧全绿**。

前者是本次修复的**靶子**，后者是**回归哨兵**。**缺 C0 则 T1 无意义**（只是重跑了一次旧的绿）。

### 3.1 主臂 gate 串（逐字，一次 serve 一条）

```bash
GATES="DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 \
DSV41_EXPERT_ACT_E4M3=1 DSV41_BF16_TRUNCATE=1 \
DSV41_SH_EXP_MROWS=1 DSV41_MROWS_SMALL_N_ADAPTIVE=1 \
DSV41_GATE_MROWS=1 DSV41_INDEXER_MROWS=1 DSV41_NORM_MROWS=1 \
DSV41_COMPRESSOR_MROWS=1 \
DSV41_DRAFT_GRAPH=1 DSV41_DRAFT_P3A=1 DSV41_VERIFY_GRAPH=1 \
DSV41_SWALLOW_STEP=1 DSV41_SWALLOW_DYNAMIC_PAD=1 \
DSV41_V5_LEDGER=1 \
DSV41_TIMING=1 DSV41_DSPARK_DEBUG=1 DSV41_INV_CHECK=1"

# 禁止项（任一出现 ⇒ 该臂 ABORT）
FORBIDDEN="DSV41_LAZY_VERIFY DSV41_HC_VERIFY_FUSE DSV41_HC_FRONT_ROWS"
# 幻影/冲突门（设了没用或有反效果 ⇒ 同样 ABORT）：
PHANTOM="DSV41_OOB_GUARD DSV41_SWALLOW_EPOCH_PAD DSV41_VERIFY_HEAD_MROWS"
```

逐条理由：
- **不设 `DSV41_SWALLOW_EPOCH_PAD`**（P4：与 DYNAMIC_PAD 叠加 = over-pad）。
- **不设 `DSV41_VERIFY_HEAD_MROWS`**（P6：Wave-1 与 m=6 冲突，历史上 ar5-hang）。
- **不设 `DSV41_OOB_GUARD`**（P1/C1：树里无此 env，设了 = 幻影门）。
- **`DSV41_INV_CHECK=1`**：它是本设计的**附加**不变量门（`chain_dev.rs:2019/2045/9825`）；注意与 `SIDS_WRITEBACK=1` **同开**（P/`swallow-1token-fix-design.md §2.A-E1`：两者必须同生共死）。
- **`DSV41_DSPARK_DEBUG=1`**：`emitted={:?}`（`serve.rs:640`）是「k_emit 是否随 accept 变化」的廉价证据（§3.2 C6）。
- **`SH_PAIR_M` 不入本串**（P6，见 §3.4）。

### 3.2 T1 判据（逐条，全部可执行）

日志：`LOG=/tmp/swallow_t1/run.log`。先跑统一 parser（§4），再逐条核对。

#### C0（**新增**，主判据）—— 步骤穿越旧墙：swallowed arm 返回了 `Ok`

```bash
grep -cE '\[v5-ledger\] pos=[0-9]+ rank=[0-9]+ epoch=[0-9]+ canary=0x[0-9a-f]+ arm=swallowed k_emit=' "$LOG"   # 必须 > 0
```
**为什么这是 C0**：note 行只在 `dspark_spec_swallowed` **返回 `Ok` 之后**才打印（`chain_dev.rs:8186`，前面那句是 `?` 冒泡）。
上一次运行**没有**这行（panic 在 note 之前）。**这行出现 = 结构性证明「engram AR 那堵墙被穿过了」**。
> 若这行**缺** ⇒ **不是**「OOB 回归」，而是「步骤仍在别处 fault」⇒ 走 §6 判读树，**别**把它当 A/B/C 失败。

#### C1 —— 无 panic（**新增三通道**，P3）
```bash
grep -c 'collective payload' "$LOG"        # 必须 == 0   （tp.rs:599 panic 文本）
grep -c 'did not answer'     "$LOG"        # 必须 == 0   （serve.rs:257，pool 线程亡）
grep -c 'spec step err'      "$LOG"        # 必须 == 0   （serve.rs:624，arm Err 冒泡）
```

#### C2 —— canary 不被动（**主判据 A**；值集合 ⊆ `{expected}`，**不硬编码** 0xdeadbeef，C4）
```bash
grep -c '\[v5-ledger-CANARY\]' "$LOG"                          # 必须 == 0
grep -o 'canary=0x[0-9a-f]\+' "$LOG" | sort -u                 # 必须 ⊆ {expected}
grep -o 'expected=0x[0-9a-f]\+' "$LOG" | sort -u               # 出现即已违反（这些行本身是 C2 的否定）
```

#### C3 —— guard 不被动（**主判据 B**，新增核心）
```bash
grep -c '\[v5-ledger-GUARD\]' "$LOG"                                       # 必须 == 0
grep -o '\[v5-ledger-GUARD\][^\n]*value=0x[0-9a-f]\+' "$LOG" | sort -u     # 必须空/⊆{expected}
```
> guard 与 canary **共用同一 magic**（`tp.rs:356`）⇒ 区域靠 `word=`/`slot=` 字段区分，**不靠值**。

#### C4 —— RESET=0（**主判据 C**，P7 的落点）
```bash
grep -c '\[v5-ledger-RESET\]' "$LOG"          # 必须 == 0
```
> 这是**修复后步骤第一次能走到 pos=15 的 note**——RESET 唯一能现形的地方。**这行一旦出现，就是真·字被写小。**

#### C5 —— epoch 单调且不跌破 999（**主判据 D**）
§4 parser 输出须 `MONOTONIC OK` 且 `NO_DROP_BELOW_999 OK`。
> **`NO_DROP_BELOW_999 SKIP` 不是 ✓**：它是「这一跑太短，epoch 没到 999」。
> 计数 1→50 的短任务**很可能 SKIP**（这正是题目要的「快速验证」的代价）。
> **处置**：T1 若 SKIP，**不阻塞**，但**必须**由 T4 的长跑（出师表，≥1000 tok）补 `NO_DROP_BELOW_999` 证据。
> 在报告里写成「C5: T1 SKIP / T4 OK」，**不得**写成「C5 ✓」。

#### C6 —— SWALLOW 真上场 + k_emit ≥ 1（**主判据 E/H**）
```bash
grep -c 'arm=swallowed' "$LOG"                                  # 必须 > 0
grep -oE 'k_emit=[0-9]+' "$LOG" | sort -u                       # 必须 ⊆ 非零；出现 k_emit=0 ⇒ E6 下溢
grep -oE 'emitted=\[[0-9, ]*\]' "$LOG" | head                    # emitted 长度随 accept 变化（DSPARK_DEBUG）
grep -c '\[verify_graph\] captured' "$LOG"                      # 期望 > 0；no 要查 gate，不阻塞
```

#### C7 —— ar5-hang = 0（**主判据 F**；分两类计数，别合并）
```bash
grep -cE '\[ar5-hang\] rank=[0-9]+ peer=[0-9]+ need=[0-9]+ cur=[0-9]+ rows=' "$LOG"   # argmax_rows
grep -cE '\[ar5-hang\] rank=[0-9]+ site=[0-9]+' "$LOG"                                # pubred/bcast
```
两者**都**必须 0（watchdog 差 10×，合并会掩盖一类）。

#### C8 —— 生成长度 > 1（**1-token 症状消失**）
```bash
python3 - "$LOG" <<'PY'
import re, sys
log = open(sys.argv[1], errors="ignore").read()
print("v5-ledger-pre lines =", len(re.findall(r'\[v5-ledger-pre\]', log)))
print("v5-ledger note lines =", len(re.findall(r'\[v5-ledger\] pos=', log)))
PY
grep -oE '"finish_reason":"[a-z]+"' $RESP | sort -u
```
**判据**：请求返回 `completion > 1`；`finish_reason` 为 `stop`（自然 EOS）或 `length`（**且** `completion ≈ max_tokens`）。
**不允许**：`completion == 1` + `length`（= P3 的 fault 签名）。

### 3.3 T1 通过线（合取）

`C0>0` ∧ `C1==0(三通道)` ∧ `C2==0` ∧ `C3==0` ∧ `C4==0` ∧ `C5 MONOTONIC OK` ∧ `C6>0 且 k_emit≥1` ∧ `C7==0` ∧ `C8 正常`。
**任一项无证据（如日志缺该行）⇒ 不得下「通过」结论（铁律：缺证据 exit 2）。**

### 3.4 可选 arm S —— `SH_PAIR_M=1`（**必须三证上场，否则 = 幻影门**）

T1 全绿后**单跑**（`scripts/sh_pair_ab.sh` 的骨架可复用）：

```bash
# S 臂 = T1 主臂 + 下面三门（三者缺一，gate 空转）
S_GATES="$GATES DSV41_SH_EXP_FUSED=1 DSV41_SH_PAIR=1 DSV41_SH_PAIR_M=1"
# 1) .so 必须真带符号
ssh $NODE 'nm -D ~/ferrite/kernels/cuda/libferrite_kernels.so | grep -c dsv41_gemm_fp8_sh_pair'   # 必须 >= 1
# 2) 实读 env 证明三门活着
ssh $NODE "tr '\0' '\n' < /proc/\$(pgrep -x ferrite-serve|head -1)/environ | grep -E 'SH_' | sort"
# 3) 运行日志出现 SH_PAIR M 臂的“上场”证据（chain_dev.rs:13511 的 M-row arm 分支）
grep -nE 'sh_pair|sh_exp|M-row' "$SLOG" | head
```
**判据**：符号存在 ∧ env 活着 ∧ 日志有上场证据；三项齐了才读红线（前 100 字零拉丁）。
**任一缺** ⇒ **该 arm ABORT**，报「gate 未上场」，**不报失败**。

---

## 4. 统一 parser（一次摘全，C2–C8 共用）

```bash
cat > /tmp/swallow_analyze.py <<'PY'
import re, sys, collections
log = open(sys.argv[1], errors="ignore").read()

# C2/C3 —— canary / guard
can  = re.findall(r"\[v5-ledger-CANARY\][^\n]*", log)
grd  = re.findall(r"\[v5-ledger-GUARD\][^\n]*", log)
cset = set(re.findall(r"canary=(0x[0-9a-f]+)", log))
gset = set(re.findall(r"\[v5-ledger-GUARD\][^\n]*?value=(0x[0-9a-f]+)", log))
eset = set(re.findall(r"expected=(0x[0-9a-f]+)", log))
print(f"C2/canary lines={len(can)} values={sorted(cset)} expected={sorted(eset)}")
print(f"C3/guard  lines={len(grd)} values={sorted(gset)}")
for l in can[:3]: print("   CANARY:", l.strip())
for l in grd[:3]: print("   GUARD :", l.strip())

# C4 —— RESET
reset = re.findall(r"\[v5-ledger-RESET\][^\n]*", log)
print(f"C4/RESET lines={len(reset)}")
for l in reset[:3]: print("   RESET :", l.strip())

# C5 —— 每 rank epoch 序列（pre + note 合流）
pat = re.compile(r"\[v5-ledger(?:-pre)?\] pos=(\d+) rank=(\d+) epoch=(\d+)")
seq = collections.defaultdict(list)
for m in pat.finditer(log):
    seq[int(m.group(2))].append((int(m.group(1)), int(m.group(3))))
bad = [(r, seq[r][i-1], seq[r][i]) for r in seq for i in range(1, len(seq[r])) if seq[r][i][1] < seq[r][i-1][1]]
flat = [e for rows in seq.values() for _, e in rows]
first999 = next((i for i, e in enumerate(flat) if e >= 999), None)
print(f"C5/ranks={sorted(seq)}  lines={len(flat)}")
print("  MONOTONIC        :", "OK" if not bad else f"FAIL {bad[:3]}")
print("  NO_DROP_BELOW_999:", "SKIP (跑太短 — 由 T4 补，不是 ✓)" if first999 is None else
      ("OK" if not [e for e in flat[first999:] if e < 999] else f"FAIL {[e for e in flat[first999:] if e<999][:5]}"))

# C1 —— panic 三通道
print("C1/panic-channel payload=", len(re.findall(r'collective payload \d+ > slot \d+', log)),
      " did_not_answer=", len(re.findall(r'did not answer', log)),
      " spec_step_err=", len(re.findall(r'spec step err', log)))

# C6 —— 上场
ke = re.findall(r'k_emit=(\d+)', log)
print("C6/arm=swallowed lines=", len(re.findall(r'arm=swallowed', log)),
      " k_emit set=", sorted(set(map(int, ke))) if ke else "[]")

# C7 —— hang 分类
hr = len(re.findall(r"\[ar5-hang\] rank=\d+ peer=\d+ need=\d+ cur=\d+ rows=", log))
hs = len(re.findall(r"\[ar5-hang\] rank=\d+ site=\d+", log))
print(f"C7/ar5-hang total={hr+hs} (argmax_rows={hr} pubred/bcast={hs})")
PY
python3 /tmp/swallow_analyze.py "$LOG"
```

---

## 5. T2 / T3 / T4

### 5.1 T2 —— 计数前 61 行（SWALLOW 正常生成）

**配置**：与 T1 **完全相同的 gate 串、同一二进制/`.so`**，只换 prompt。`temperature=0, max_tokens=1000`。
Prompt：`请从 1 数到 200，每个数字单独一行。`

```bash
python3 - /tmp/swallow_t2/resp.json <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))["choices"][0]["message"]["content"]
lines = [l.strip() for l in c.splitlines() if l.strip()]
ok = 0
for i, l in enumerate(lines):
    if l == str(i+1): ok += 1
    else: break
print(f"total_lines={len(lines)} ok_lines={ok} first_bad={ok+1}")
print("P1_FIRST61", "OK" if ok >= 61 else f"FAIL (ok={ok})")
print("NOT_EARLY_EOS", "OK" if len(c) > 60 else f"FAIL (chars={len(c)})")
PY
```

| 判据 | 通过 | 失败指向 |
|---|---|---|
| 前 61 行 | `ok >= 61` | `<61` ⇒ 与**同会话 EAGER 对照**再判（模型疲劳 vs 引擎） |
| 不早停 | `chars > 60` | `≈6 token` ⇒ 仍 fault（回 T1 的 C0/C1） |
| 损坏点（辅助） | `first_bad == 62`（与 EAGER 同模式） | `<61` ⇒ 新损坏点 |

> **第 62 行起不判**（`r2-reverification-test-design.md §1`：EAGER 无 spec 亦在第 62 行损坏）。

### 5.2 T3 —— 吞吐（batched weight-sharing 的步时）

**配置**：T1 主臂 **去掉 `DSV41_V5_LEDGER`**（P：每步一次 4B D2H，计量跑必须关；`batched_400_v2.sh:157-167` 的纪律）。
prompt = 出师表，`max_tokens=1000`。**可与 T4 共用同一次 serve**（脚本已对每步计时）。

```bash
# 观测关：只去掉 V5_LEDGER，其余逐字同 T1
T3_GATES="$(echo "$GATES" | sed 's/DSV41_V5_LEDGER=1//')"
# 一次 serve + 出师表（复用 §3.1 的启动骨架），收集日志
grep -cE '\[ar5-hang' "$LOG"                                     # 必须 0
grep -oE '\[dsv41\] step pos=[0-9]+ .*ms' "$LOG" | head          # 步时序列
grep -oE 'verify_ms=[0-9.]+|draft_ms=[0-9.]+|commit_ms=[0-9.]+' "$LOG" | head
grep -oE '"completion_tokens":[0-9]+' $RESP ; grep -oE '"e2e[a-z_]*":[0-9.]+' "$LOG"
```

**口径**（AGENTS.md）：`tok/s = completion_tokens / e2e_seconds`（end-to-end，不用段平均）。

| 判据（主） | 通过 | 备注 |
|---|---|---|
| **同栈 step-time 对比** | batched 的 `steady_median` 相对**同会话同栈**基线下降 | 只换一个 arm，禁跨栈 |
| **matched accept** | 两臂 `mean_k` 同量级（accept-N 的步时不可配 accept-M） | `mean_k≈1.02–1.2` 是本轮票面 |
| **verify_ms** | ≈33–36ms（无 tcgen05 的 batched 地板） | 低于 30ms 先**质疑测量** |
| **ar5-hang** | == 0 | 动态 pad 效果保持 |
| **红线** | 前 100 字零拉丁 + `先帝创业未半` | 见 T4 |
| **参考（非判据）** | tok/s 票面 ≈50–65；`>91.1` 只作陈述 | **P5：91.1 是 lazy 栈，不是同栈基线** |

> ⚠️ **P5 红线**：不得因 `tok/s < 91.1` 判 T3 ✗。正确结论形如
> 「batched 票面 55 tok/s（accept≈1.1，步时 38ms）；lazy 栈 91.1 因 accept≈1 只跑 2 行，不可直接配」。
> `batched` 反超 lazy 的条件是 `accept ≥ 2.5`（`swallow-result-action-plan.md §2.3`）——**那是 accept 战役，不是本次测试的判据**。

### 5.3 T4 —— 出师表零拉丁（红线，**范围修正：只判前 ~100 字**）

```bash
python3 - /tmp/swallow_t4/resp.json <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))["choices"][0]["message"]["content"]
head = "".join(c.split())[:100]
latin = [ch for ch in head if ("a" <= ch <= "z" or "A" <= ch <= "Z")]
dbl = sum(1 for i in range(1, len(head)) if head[i] == head[i-1])
print("head_chars=", len(head))
print("HEAD_ZERO_LATIN", "OK" if not latin else f"FAIL {latin[:5]}")
print("HEAD_NO_DBL",     "OK" if dbl == 0 else f"FAIL dbl={dbl}")
print("KAISHEN_PREFIX",  "OK" if head.startswith("先帝创业未半") else f"FAIL head={head[:12]!r}")
PY
```

| 判据 | 通过 | 失败 |
|---|---|---|
| 前 ~100 字零拉丁 | ASCII 字母数 == 0 | 出现即红线 ✗ |
| 开头 `先帝创业未半` | 逐字正确 | 前缀错 ✗ |
| 无相邻双字 | `dbl == 0` | `>0` ✗ |

> T4 的**附加价值**（P7/C5）：出师表 `max_tokens=1000` 是唯一能保证 epoch 到 999 的长跑 ⇒
> **`NO_DROP_BELOW_999` 的补证在 T4**（T1 大概率 SKIP）。若 T4 里 C5 仍 SKIP，则 epoch 从未到 999，报告须写明。

---

## 6. 判读树（T1 一次定向到唯一分支；续接 R0–R7）

```
T0 过 → 门核对（env 逐字 vs §3.1 + FORBIDDEN/PHANTOM 均不在）
│  失败 ⇒ 该臂作废，重跑，不判读
▼
C0 swallowed-note > 0 ?
├─ 否 ──▶ 步骤仍死在某处（不是 OOB 回归）：
│         ├─ C1 payload/did_not_answer/spec_err 任一 >0 ⇒ 按通道定位：
│         │     · collective payload  ⇒ 新越界入口（再算 slot 预算，回 §2.2）
│         │     · spec step err       ⇒ arm Err ⇒ 对号入座 E1–E6
│         │     · did not answer      ⇒ 非 ar5-hang 的 hang ⇒ 加 phase 打印（B3）定位
│         └─ 三者皆 0 且 note 缺      ⇒ ledger 零行 ⇒ 查 gate / 首步 hang（旧 R7）
│
└─ 是 ──▶ C2/C3/C4 任一 >0 ?
          ├─ 是 ⇒ 越界仍在：C3(guard) 命中⇒读 word=/value= 反推 reduced 方向写入者；
          │        C2(canary) 命中⇒读 slot=(0/1/2/3)→off=8/16/32/48，越宽越远；
          │        C4(RESET) 仅命中⇒真·字被写小（复查修复是否覆盖该写者）
          └─ 否 ⇒ C5/C7 检查
                   ├─ C7 ar5-hang >0 ⇒ pad 未收敛（查 arm=swallowed 计数 + 动态 consensus）
                   └─ C7==0 ⇒ 看 C8
                              ├─ C8 completion==1 + length ⇒ 仍有 fault（回 C1 通道）
                              └─ C8 正常 ⇒ ✅ T1 通过 → T2 → T3 → T4
```

---

## 7. 陷阱清单（本次专属，全部有源码理由）

1. **C0（swallowed note）不能漏**（P2）：没有它，T1 只是「重跑了一次旧的绿」——旧绿只覆盖到 engram AR。
2. **「无 panic」要 grep 三条通道**（P3）：`collective payload` / `did not answer` / `spec step err`；只查第一条会漏「arm Err」与「pool 线程亡」。
3. **判据不硬编码 `0xdeadbeef`**（C4 in framework）：用 `expected=`；guard 与 canary 同 magic ⇒ 靠 `word=`/`slot=` 区分。
4. **`NO_DROP_BELOW_999 SKIP` 不是 ✓**（framework §6-2）：短任务（计数 1→50）大概率 SKIP ⇒ 由 T4 补证，报告如实标注。
5. **T3 的 91.1 是 lazy 栈**（P5）：跨栈比会得出错误的「batched 不如 lazy」；主判据是**同栈 step-time + matched accept**。
6. **`SH_PAIR_M` 三证上场**（P6）：符号 + env + 日志证据；缺一即「幻影门」，报 ABORT 不报失败。
7. **slot 与载荷恰好相等（余量为 0）**（P8）：任何几何变动（VERIFY_ROWS/ngram/n_heads）会立刻再 panic ⇒ §2.2 的预算断言必须进 T0。
8. **`.cu` 若被顺手改动 ⇒ 必须重编 `.so`**（P1）：成功判据是 `built …libferrite_kernels.so for sm_103a`，不是 `build.sh` 的 rc。
9. **`inv_ids` 与 `sids_writeback` 必须同开/同关**（E1）：T1 主臂两者同开；负对照只关 `SIDS_WRITEBACK`。

---

## 8. 交付清单（供尚书省分派）

| # | 项 | 内容 | 优先级 | 风险 |
|---|---|---|---|---|
| 1 | T0.0 | `cargo check --workspace` + `cargo test -p ferrite-types spec_step` | P0 | 无 |
| 2 | T0.1 | `cargo build --release`（.rs-only）+ 同源证明（`.build_id` embed）+ md5 | P0 | 无（若 `.cu` 变则加 `build.sh`） |
| 3 | T0.2 | 载荷预算断言（slot==147456==engram payload） | P0 | 无 |
| 4 | T0.3 | 可失败对照（`SIDS_WRITEBACK=0` + `INV_CHECK=1` ⇒ 必红） | P0 | 低 |
| 5 | T1 | 主臂 gate 串（§3.1，**无** EPOCH_PAD/VERIFY_HEAD_MROWS/OOB_GUARD）+ C0–C8 | P0 | 无 |
| 6 | T2 | 计数 1→200，前 61 行 + 不早停 | P0 | 无 |
| 7 | T3 | 去 `V5_LEDGER` + 出师表，同栈 step-time/verify_ms 对比 | P1 | 无 |
| 8 | T4 | 出师表前 ~100 字零拉丁 + 补 `NO_DROP_BELOW_999` 证据 | P0 | 无 |
| 9 | `scripts/batched_400_v2.sh` | 正式化：`SWALLOW_EPOCH_PAD=1` → `SWALLOW_DYNAMIC_PAD=1`；`FORBIDDEN` 增列「同时设 EPOCH_PAD+DYNAMIC_PAD」 | P1 | 低（需重跑 T3/T4 基线） |
| 10 | arm S | `SH_PAIR_M` 独立臂（三证上场）| P2 | 中（编译期模板 + `.so` 依赖）|

**不做**：
- 不在 T1 前改任何 kernel 语义；不新增 gate（新增 gate = 新增一处「以为在跑其实没跑」）。
- 不把 `*epoch = e+k` 改成 `atomicMax`（会掩盖清零，`epoch54-final-fix-path.md §7`）。
- 不在 T3 之前把 `V5_LEDGER` 打开测吞吐（每步 D2H 会污染计量）。

---

## 9. 一页纸结论

1. **本次修复（`249c1b2`）只改 `.rs`** ⇒ 只 `cargo build`；但同源证明照做（`.build_id` embed）。
2. **T1 的靶子是「穿越旧墙」**（C0：`arm=swallowed` 的 **note 行**必须出现）——上一次的绿只覆盖到 engram AR，之后全是**零证据新可达区**（E1–E6）。
3. **OOB 效果保持**是回归哨兵：canary/guard/RESET 三条**都是活判据**，RESET 因步骤终于能走到 pos=15 的 note 而**重新有了现形的机会**（P7）。
4. **「无 panic」= 三通道 grep**：`collective payload` / `did not answer` / `spec step err`；`LEN=1` 只是它们的下游。
5. **T3 的 baseline 是 lazy 的 91.1，但 batched 票面 ≈50–65**（P5）——主判据是**同栈 step-time + matched accept**，不是绝对 tok/s。
6. **可证伪性用 `SIDS_WRITEBACK=0 + INV_CHECK=1` 构造必红对照**（旧文档的 `OOB_GUARD` 负对照不存在）；不自证「测试能失败」的绿 = 零信息。

---

*工部 · 只读勘察 + 设计；唯一产出为本文件。未改动任何源码、未执行 GPU 命令。*
*所有行号以 HEAD `249c1b2` 为准；无法从源码定论的推断均显式标注 `[未验证]` 并给出证伪条件。*
