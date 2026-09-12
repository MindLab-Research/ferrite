# R2 重验测试设计（ATTN_LIN_FUSE 的 clean-stack 重判）

> 工部 · 2026-09-12 · **只读勘察 + 设计；未改动任何源码，未执行 GPU 命令。**
> 输入：`crates/ferrite-models/src/dsv41/chain_dev.rs`（`attn_lin_fuse` / `AttnLinFuse` / `attention_rows`）、
> `kernels/cuda/tests_dsv41_r2_parity.cu`（现成的 R2 字节 parity 测试）、
> `scripts/dsv41_serve_ab.sh` / `scripts/sh_pair_ab.sh` / `scripts/verify_mrows.sh`（现成的探针/判据骨架）、
> 远端 `/tmp/{eager_e4m3_out.json, lazy_best_count_out.json, r2_out.json, r2ba4_out.json, lin2_count_out.json}`
> （本会话在跑的计数对照产出，均已读）。
> **一句话**：所有在测臂（EAGER / base / R2 / lin2）的计数损坏都从**第 62 行**开始且模式相同
> （`...60,61,12,13,14,15...`）——这是**模型自然行为**（探针跑到 ~60 token 后的"数数疲劳"），
> 不是 R2 的 bug。R2 的重验判据必须**只判前 ~60 行 + 出师表零拉丁**，并加一条 EAGER 并列对照。

---

## 0. TL;DR（四条）

1. **证据已足以翻案**：`/tmp/eager_e4m3_out.json`（18:49，纯 EAGER+e4m3，无任何 spec/lazy/graph/R2）
   输出 `1..61` 然后 `12,13,14,15,16,17,18,29,20,...`——**与 base/R2/lin2 的损坏点（第 62 行）和模式完全一致**。
   ⇒ 第 62 行的"重置到 12"**与 R2 无关**，是 base 模型在长计数上的自然混淆。
2. **因此 R2 的"损坏"判定（62-65 行错）是假阳性**——它继承的是模型行为，不是 R2 自己引入的。
   R2 的 +6%（78.8→82.9）**必须先按干净判据重判，再决定是否采信**。
3. **重验判据三段**：① 计数**前 61 行**（`1..61` 精确）② 出师表 1000 tok **零拉丁** + 前 100 字内容正确
   ③ **EAGER 并列对照**（同一会话、同一 prompt、同一并发下跑纯 EAGER，作为"模型行为基线"）。
   任何一段不过 = 判 R2 ✗；三段全过 = 判 R2 干净。
4. **现成的最强判据不是文本而是 `tests_dsv41_r2_parity.cu`**：它对真实形状做 **memcmp 位级**比对
   （`lin2+lin_rope_norm` vs 分离链）。**位等 ⇒ 数学生证，文本探针沦为辅助**；位不等 ⇒ 定位到
   具体 launch（A==C 证明 prologue、C!=B 证明 mrows/rope 侧），直接给出根因方向。**必须先跑它。**

---

## 1. 判定链（重验的前置门）

```
[Gate 0] EAGER+e4m3 计数对照
   ├─ EAGER 也 ~52-62 损坏（且模式同 base） ──► [模型行为] ──► 走 §2 重验 R2（本设计的默认分支）
   └─ EAGER 干净（1..200 全对）        ──► [spec 路径 bug] ──► R2 仍是嫌疑，走 §6 的 B 分支（不翻案）

[Gate 0.5] 若 Gate 0 = 模型行为，但仍想 100% 排除"base 引擎共有 bug"：
   跑 ref_inference/（PyTorch 参考）或已知好端点，同 prompt 数数 ──► 参考也 ~60 损坏 = 铁证
```

**Gate 0 现状（只读，已完成的证据）**：

| 臂 | 产出文件（远端 /tmp）| 数字正确 | 首个错误行 | 损坏模式 |
|---|---|---|---|---|
| **EAGER+e4m3**（无 spec）| `eager_e4m3_out.json` 18:49 | 61/77 | **第 62 行** | `12,13,14,15,16,17,18,29,20,...` |
| base（lazy+SH_PAIR+Wave1）| `lazy_best_count_out.json` 16:58 | 61/72 | **第 62 行** | `12,13,14,15,28,29,30,31,32,33,44` |
| **R2（=1，全开）** | `r2_out.json` 17:31 | 61/65 | **第 62 行** | `12,13,14,15`（更早停）|
| R2b+A4 | `r2ba4_out.json` 17:36 | 61/65 | **第 62 行** | 同 R2 |
| **lin2（=2，仅一半）** | `lin2_count_out.json` 18:11 | 61/72 | **第 62 行** | 与 base 逐字相同 |

> **关键读法**：五个臂的**首个错误行全部 = 62**，且第 62 行的错值**全部是 `12`**（"重置到 12"）。
> EAGER 无 spec 也如此 ⇒ 这是模型的输出，不是引擎的。**Gate 0 = 模型行为，通过。**

---

## 2. 重验配置（arms）

重验要在**同一会话、同一二进制+.so、背靠背**跑四臂；唯一变量是 `DSV41_ATTN_LIN_FUSE`。

### 2.1 固定 base env（四臂共用）

基线与 78.8 干净基线一致（lazy + SH_PAIR_M=1 + Wave 1），**S1/S3 修复默认已 ON**（代码默认）：
`DSV41_LAZY_ROUTE_LOCK` 默认 `!=0` ⇒ ON（S3 锁定）；DIRECT 双计在 `chain_dev.rs:6032` 的 `if !mirror_advanced`
分支已修（S1）。**重验不需要显式设置 S1/S3**——只要用 HEAD 的默认值，并在 arm 完整性检查里确认它们没被显式关掉。

```bash
NCCL_NVLS_ENABLE=0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# —— 与 scripts/sh_pair_ab.sh 的 BASE_ENV 同源（Wave 1 + lazy + mrows + e4m3）——
DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1
DSV41_HC_VERIFY_FUSE=1 DSV41_HC_FRONT_ROWS=1 DSV41_VERIFY_AR_FOLD=1
DSV41_GATE_MROWS=1 DSV41_INDEXER_MROWS=1 DSV41_COMPRESSOR_MROWS=1
DSV41_LAZY_VERIFY=1 DSV41_VERIFY_GRAPH=1
DSV41_BF16_TRUNCATE=1
DSV41_EXPERT_ACT_E4M3=1 DSV41_DRAFT_P3A=1 DSV41_TAP_INPUT=1 DSV41_DRAFT_BF16_DOMAIN=1
DSV41_SH_PAIR_M=1                    # 干净 78.8 基线的一部分
DSV41_TIMING=1                       # 步时/dspark steps 行必需
# 不设：DSV41_SH_EXP_MROWS（base 必须是 per-row 链）、DSV41_SWALLOW_STEP（lazy 已隐含）
```

### 2.2 四臂（严格串行，单变量）

| # | arm | extra env | 目的 | 期望 |
|---|---|---|---|---|
| A | **EAGER** | `DSV41_SPEC=0 DSV41_DSPARK=0`（其余同上，去掉 spec 相关）| **模型行为基线**（本轮新测，替换 18:49 的临时探针）| 计数第 62 行损坏 |
| B | **base** | （空）| R2 的对照 | 计数第 62 行损坏 |
| C | **R2=1** | `DSV41_ATTN_LIN_FUSE=1` | 被测臂（全开）| 前 61 行对 + 零拉丁 |
| D | **R2=2 / =3**（可选 bisect）| `DSV41_ATTN_LIN_FUSE=2` / `=3` | 若 C 有异，一眼定位是哪半边 | 各自与 B 同 |

> **口径纪律**（AGENTS.md）：arm 完整性用 `/proc/<pid>/environ` 实读（`sh_pair_ab.sh` 的 `run_case` 已实现），
> 确认 `DSV41_ATTN_LIN_FUSE` 的**值**真的进了 serve；C 臂必须 `=1`，B/A 臂必须**不含**该变量。
> 任何一臂的 gate 泄漏 ⇒ 该臂 ABORT，不并入平均。

---

## 3. 探针设计（每条探针都有独立判据）

### P1 · 计数（主判据：前 61 行）

- **Prompt**（三臂逐字相同，取自 `scripts/sh_pair_ab.sh`，与历史对照一致）：
  `请从1数到200，每个数字单独一行。`（`temperature=0`, `max_tokens=1000`, `stream=false`）
- **解析**：按行 `strip()`，取非空行；第 `i` 行应 == `str(i+1)`（i 从 0 起）。
- **主判据（红线）**：**前 61 行全部正确**（`lines[0..61] == ["1".."61"]`）。
  - ⚠️ **不要用"前 60"**：实测损坏点在第 **62** 行（`1..61` 正确），用 60 会丢 1 行裕度。
  - ⚠️ **第 62 行起不判**——那是模型行为（见 §1）。
- **辅助指标**：`first_bad = 首个 != str(i+1) 的行号`；`ok = 正确行数`。
  重验里 `first_bad` 是**关键交叉检验量**：R2 的 `first_bad` 必须 **≥** base 的 `first_bad`
  （即 R2 不得把损坏点**提前**）。若 R2 的 `first_bad < 61` ⇒ **R2 自己的新损坏点** ⇒ ✗。

### P2 · 出师表（主判据：零拉丁 + 内容）

- **Prompt**：`请完整背诵《出师表》全文，从先帝创业未半而中道崩殂开始。`（`temperature=0`, `max_tokens=1000`）
- **判据**：
  1. **零拉丁**（红线，与计数同级）：答案中 ASCII 字母数 == 0（`sh_pair_ab.sh` 的 `latin` 指标）。
  2. 前 100 字（去空白）以 `先帝创业未半` 开头且**逐字正确**到至少 100 字。
  3. 无相邻双字（`dbl == 0`，出师表 canary；**不适用于计数 prompt**）。
- ⚠️ **假阴性风险**（历史教训）：出师表**自然停止**可能在损坏点之前（例如只生成 93 tok）。
  ⇒ 出师表**只作红线/内容判据，不作 accept/损坏点判据**；损坏点一律以 **P1 计数**为准。

### P3 · k_acc 序列（accept 不变性）

- 从 `[dsv41] step pos=<p>` 行的位置差解析：`k_acc = Δpos - 1`（`k_emit = k_acc+1`）。
- **判据**：R2 与 base **前 10 步**（进入模型损坏区之前）的 `k_acc` 序列**逐项相同**（或 `md5` 相同）；
  均值差 `|Δmean-k| < 0.05`（`sh_pair_ab.sh` 的 `EPS`）。
- 理由：R2 只是把投影族 7 发降到 2 发，**不应改变 accept**；若 k_acc 变了 ⇒ 程序差异已在起作用 ⇒ 查 `VERIFY_HEAD_FOLD/SLICED` 是否被 arm 顺带改动。

### P4 · 吞吐（+6% 是否复现）

- **口径**（AGENTS.md）：吞吐 = **end-to-end**（`completion_tokens / e2e_seconds`）；步时用 `[dsv41] step pos`/`[dspark] steps` 行，**不用段平均**。
- 用 P1 的同一计数请求计时（真实生成，不是压测）。
- **判据**：R2 相对 base 的 `tok/s` ≥ **+3%**（+6% 的宽松下限，留噪声空间）；
  同时 `steady_median`（steady step wall，`sh_pair_ab.sh` 口径，`STEADY_SKIP=20`）R2 应低于 base。
- ⚠️ **accept 必须可比**：只有 P3 的 k_acc 相同，P4 的 tok/s 才能配（accept-N 的步时不能配 accept-M 的 tok/step）。

### P5 · 位级 parity（**最强判据，先跑**）

- 现成资产：`kernels/cuda/tests_dsv41_r2_parity.cu`（真实形状，memcmp 位级）。
- 构建（单 TU，本地可编，无需 GPU）：
  ```bash
  nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 \
       -o /tmp/t_r2parity kernels/cuda/tests_dsv41_r2_parity.cu
  # 节点上 CUDA 来自 pip wheel 时加 -I/-L（见该文件头注释）
  ```
- 运行：`CUDA_VISIBLE_DEVICES=7 /tmp/t_r2parity`（峰值 ~16 MB）；`--quick` 快速版。
- **判据**：
  - **exit 0（全部位等）** ⇒ R2 数学上 = 分离链 ⇒ 文本探针只需**确认不回归**，R2 **可直接采信**。
  - **exit 1（有 diff）** ⇒ 看报告：`A == C`（fused prologue = rmsnorm_q + mx_rope）且 `C != B`
    ⇒ 差异在 **mrows/apply_rope_mrows** 侧（不是 R2 的 norm 复用）；反之则在 R2 的 prologue 侧。
    这直接给出**根因方向**，比任何文本探针都快。
- **控制项（D1-D4）必须先全等**：控制项不等 = 环境/harness 坏，不是 R2。

---

## 4. 判据汇总（决策表）

| 判据 | 通过（R2 干净 ✓） | 失败（R2 的 bug ✗）|
|---|---|---|
| **P5 位级 parity** | 全部 bit-identical（exit 0） | 有 diff（且控制项全等）|
| **P1 计数前 61 行** | `1..61` 精确正确 | 前 61 行内任何错行 |
| **P1 损坏点交叉** | `first_bad(R2) ≥ first_bad(base)` | `first_bad(R2) < 61`（新损坏点）|
| **P2 出师表** | 零拉丁 + 前 100 字正确 | 出现任何 ASCII 字母 |
| **P3 k_acc** | 与 base 前 10 步逐项同 | 序列/均值漂移 |
| **P4 吞吐** | ≥ base +3%，steady wall 更低 | 无提升或更慢 |

**判定逻辑**：
- 决策表**任一行 ✗** ⇒ R2 有**该行指向的**问题，回工部定位（P5 报告给方向）。
- 全部 ✓（或 P5 位等 + P1/P2/P3 ✓）⇒ **R2 干净，+6% 采信**，进入 R2 的采信栈。

### 反例的读法（务必区分"R2 的 bug" vs "模型行为"）
1. **只有第 62 行之后错，且与 base/EAGER 同模式** ⇒ **不是 R2 的 bug**（模型行为），判 ✓。
2. **前 61 行内出错** ⇒ R2 引入新损坏点，判 ✗。
3. **出师表出现拉丁** ⇒ 红线，判 ✗（历史 R2 的原始症状就是这个，必须零容忍）。
4. **k_acc 变了** ⇒ 判 ✗（R2 不该动 accept），先查 arm 是否顺带改了 head 相关 gate。

---

## 5. 预期结果

若"第 62 行损坏 = 模型行为"成立（Gate 0 已支持）：

| 指标 | base | R2=1（预期）| 说明 |
|---|---|---|---|
| P1 首个错误行 | 62 | **62** | 相同 ⇒ R2 未提前损坏点 |
| P1 前 61 行 | ✓ | **✓** | 主判据 |
| P2 零拉丁 | ✓ | **✓** | 红线 |
| P2 前 100 字 | ✓ | **✓** | 内容 |
| P3 k_acc（前 10 步）| 全 5（或与 base 同）| **同 base** | accept 不变 |
| P4 tok/s | ~78-79 | **~82-85** | **+6% 复现 ⇒ 采信** |
| P5 parity | — | **exit 0** | 位等 ⇒ 生证 |

**结论出口**：
- **P5 位等 + P1/P2/P3 ✓** ⇒ R2 是"被误判的最大单项优化"（verify m=1 从 7 发降到 2 发），
  干净栈速度 **78.8 → ~83-85 tok/s**，可直接采信或与 K1/K2 对比择一。
- **P1 前 61 行 ✗ 或 P2 有拉丁** ⇒ R2 确有独立 bug，与 base 的模型行为是**两回事**，维持 OFF，
  按 P5 的方向（prologue 侧 / mrows 侧）修。

---

## 6. 执行顺序（单轮制、背靠背）

> 纪律：严格串行（serves 不能重叠，`sh_pair_ab.sh` 已用 flock 防重叠）；每臂 `POST /shutdown` 收尾；
> 跨版本比较**同会话背靠背**；禁止前台 sleep（用健康轮询）。

```
0. 双产物重编（若 HEAD 有 .cu 改动）：kernels/cuda/build.sh 103a → touch build.rs → cargo build --release
1. P5：nvcc 编 tests_dsv41_r2_parity.cu → CUDA_VISIBLE_DEVICES=7 跑 → 记 exit + 报告   ← 先做，最快定性
2. A(EAGER)  →  P1+P2  →  shutdown     ┐
3. B(base)   →  P1+P2+P3+P4 → shutdown │ 同一会话、背靠背、同一 prompt/并发/温度
4. C(R2=1)   →  P1+P2+P3+P4 → shutdown │
5. (可选) D(R2=2 / =3) → P1+P2 → shutdown ┘
6. 汇总 §4 决策表；P1 的 first_bad 三臂并列；P4 的 tok/s base vs R2 并列
```

**每臂收尾自检**（复制 `sh_pair_ab.sh` 的 run_case 检查）：
`/proc/<pid>/environ` 里 `DSV41_ATTN_LIN_FUSE` 的值正确 / 无 `DSV41_SH_EXP_MROWS` / 无 `DSV41_SWALLOW_STEP` /
`DSV41_LAZY_VERIFY=1` 存在；`faults == 0`（grep illegal/fault）；健康轮询不超时。

---

## 7. 风险与陷阱（本设计显式规避的）

1. **判据边界错位**：损坏点实测 = **第 62 行**（不是 60，也不是 52）。用 60 丢裕度；用 52 会把模型行为误判成 bug
   （`eager_count_out.json` 的旧样本停在 52，是**另一条模型轨迹**，不是新损坏点）。**以 P1 的 `first_bad` 实测为准。**
2. **出师表假阴性**：出师表自然停止 < 损坏点时会"看起来干净"。**不能用出师表判损坏点**，只判红线/内容。
3. **拉丁检查单独不够**：必须加上计数数字顺序（AGENTS.md 红线纪律）。本设计 P1+P2 双红线。
4. **arm 泄漏**：`ATTN_LIN_FUSE` 若是 set 在 shell 而非 serve 进程，单变量失效 ⇒ 必须 `/proc/environ` 实读。
5. **位等 vs 文本**：文本全对**不能**证明位等（历史教训："构造性等价论证不够，必须真实形状逐字节 diff"）。
   所以 **P5 先跑**；P5 位等后文本探针才降为辅助。
6. **口径混用**：P4 只在 P3 的 k_acc 相同时才可比（accept-N ≠ accept-M 的 tok/step）。
7. **别重跑已验证的 baseline**：78.8 已多轮验证；本设计里 base 臂**不是**重验 baseline，而是 A/B 的对照臂，
   必须与本轮 R2 背靠背（跨会话数字不可比）。

---

## 8. 交付物清单（执行时）

- [ ] P5 `t_r2parity` 的完整 stdout（exit code + 各 tag 的 bad_elems/max_ulp）
- [ ] A/B/C 三臂 × P1 的完整答案文本 + `first_bad`/`ok` 表
- [ ] A/B/C 三臂 × P2 的拉丁计数 + 前 100 字 + md5
- [ ] B/C × P3 的 k_acc 序列（md5）+ 均值
- [ ] B/C × P4 的 e2e tok/s + steady_median
- [ ] `/proc/environ` 快照（三臂各一份）
- [ ] §4 决策表逐行判定 + 结论（R2 采信 / 不采信 + 根因方向）

---

*工部 · 只读勘察 + 本文件（唯一产出）；未改动任何源码、未执行 GPU 命令。*
*所有"实测"数字均标注远端文件与时间戳；口径冲突处已显式标注。*
