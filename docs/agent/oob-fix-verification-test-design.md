# OOB 修复后的验证测试设计 —— T0–T4（完整命令 + 判据 + 预期）

> 工部 · 2026-09-12 · **只读勘察 + 设计；未改动任何源码、未执行 GPU 命令。**
> 输入：`crates/ferrite-models/src/dsv41/{tp.rs,chain_dev.rs,device.rs}`、
> `kernels/cuda/{ferrite_kernels.cu,dsv41_kernels.cu}`、
> `scripts/batched_400_v2.sh`（重建-同源-测量一体化）、
> `docs/agent/{epoch54-final-fix-path.md,epoch54-witness-fix-templates.md,swallow-fix11-design.md}`、
> `dspark-correctness-chain.md:5679-5725`（决定性 witness 测试的原始读数）。
> 代码基线：工作树 HEAD `f7cc53b`（`main`，工作区干净）。行号以该树为准；无法从源码定论的推断标 `[未验证]` + 证伪条件。

---

## 0. 三条前提校正（不先读会重犯「修一个误读」的病）

### 0.1 「OOB 清零 staging」的源码级形状，决定了测试要抓的是什么

决定性读数（`dspark-correctness-chain.md:5683-5686`，HEAD 之前的 `2e42ebb7`）：

```
[v5-ledger-pre] pos=15 rank=0 epoch=999  canary=0xdeadbeef arm=pre          ← 正常
[v5-ledger-CANARY] pos=15 rank=0 arm=swallowed canary=0x00000000 expected=0xdeadbeef  ← 🚨 清零
[v5-ledger-RESET]  pos=15 rank=0 arm=swallowed prev=999 cur=54 drop=945     ← 降级被抓住
[v5-ledger]        pos=15 rank=0 epoch=54 canary=0x00000000 arm=swallowed
```

三个**互锁**的事实必须被测试**同时**盯住，缺一个就分辨不出「修好了」与「症状换了」：

| # | 事实 | 源码位置 | 含义 |
|---|---|---|---|
| **F-canary** | magic 写在 `staging + ctr_at + 8`，**全树无 kernel 写它** | `tp.rs:313/318/361-367`、`canary_dev()` `tp.rs:463-467` | 它变了 = 有越界写踩进了 epoch 尾部（world A3） |
| **F-epoch** | epoch 在 `staging + ctr_at`；写者全是 `e+k`（k≥1） | `tp.rs:455`；`ferrite_kernels.cu:9067/9141`、`dsv41_kernels.cu:8472/8538` | **单调不可降**；出现 54 = 有字被清后重数 |
| **F-RESET** | probe 对**同 rank**的 epoch 做下降断言 | `chain_dev.rs:9661-9670` | `prev=999 cur=54` 必打一行——**这是判据，不是假设** |

⇒ **测试的主判据不是「文本好不好」，而是「canary 是否回到 magic、epoch 是否仍单调、RESET 是否为 0」**。文本探针（T2/T4）只作辅助与红线。

### 0.2 pad 的选择：**动态 pad ON、常数 pad OFF**（否则验证的是另一件事）

`swallow_dynamic_pad()`（`chain_dev.rs:1970`）= 11-B 的 per-step epoch consensus，读本 rank epoch → 取 world MAX → 把落后者 pad 到 max（`:9804-9820`）。**它严格支配常数 pad**（`swallow_epoch_pad()` = 写死 `2*n_layers+1 = 81`，`:1913`，`chain_dev.rs:1925`）。

**⇒ T1–T4 全部用 `DSV41_SWALLOW_DYNAMIC_PAD=1`，且不设 `DSV41_SWALLOW_EPOCH_PAD`。** 两个一起开 = 先补 81 再补 max−me ⇒ **over-pad**（本 rank 反而领先）⇒ 触发新的 `[ar5-hang]`。这不是「更保险」，是把验证对象换成了「双 pad 的 bug」。

> `batched_400_v2.sh` 的 `GATES` 里目前写死的是 `DSV41_SWALLOW_EPOCH_PAD=1`（const `:154`）。**这也是本设计交付给实施者的一个改动项（见 §9 第 8 行）**——不是手改，是把新矩阵正式化。

### 0.3 测试必须**能失败**（可证伪性），否则 T1 的「绿」没有信息量

项目的上一次教训是「10 次修复全失败，其中一次是幻影（零调用点）」。**一个永远为真的断言 = 零信息。** 所以：

- **首选**：OOB 修复若可 gate（推荐 `DSV41_OOB_GUARD`，默认 **ON**，`=0` 复现 pre-fix），则 T0.3 的负对照是「同一 gate 串 + `OOB_GUARD=0`」⇒ **必须复现 canary=0 + RESET≠0**。复现不了 ⇒ 测试没在测你修的东西 ⇒ 先修测试。
- **备选**：修复若为直接修（无 gate），则用**归档的 pre-fix 日志**（`2e42ebb7`）作负对照，把它的 `canary/RESET` 行 md5 记下来。**但这弱一档**——它只能证明「历史上失败过」，不能证明「当前测试能抓住回归」。

> 结硬寨：**T1 的绿，必须与 T0.3 的红成对出现才有意义。** 只报绿不报红 = 第 9 种幻影的同族病。

---

## 1. 测试矩阵总览

| 测试 | 配置（相对 T1 主臂） | 判据（全过才算 ✓） | 预期 |
|---|---|---|---|
| **T0** | 前置门：`cargo check` + 双产物重编（若 `.cu` 变）+ 负对照 | 见 §2 | 编译/同源/可证伪三关 |
| **T1** | SWALLOW + DYNAMIC_PAD + V5_LEDGER + V5_WITNESS + OOB 修复 | canary 恒 `0xdeadbeef`；RESET=0；epoch 单调；`ar5-hang`=0；witness W1/W2/W5 成立 | **OOB 修复生效** |
| **T2** | T1 主臂 + 计数 prompt（1→200） | 前 61 行 `1..61` 精确；答案不早停（>60 tok） | SWALLOW 正常生成 |
| **T3** | T1 主臂但 `V5_LEDGER=0 V5_WITNESS=0`（观测关）+ 出师表计时 | e2e tok/s > 91.1；steady_median 下降；mean-k 可比；图 captured | batched weight-sharing 的吞吐 |
| **T4** | T1 主臂 + 出师表 | 前 ~100 字零拉丁 + `先帝创业未半` + 无相邻双字 | 红线 |

**「组合 gate」= T1 主臂。** 逐字串见 §3.0；**一次 serve 一条串**，串不许中途加、不许临场 export。

---

## 2. T0 —— 前置门（编译先行 + 双产物 + 可证伪性）

### 2.0 本地硬门禁（0 GPU）

```bash
cd /home/smith/src/ferrite && cargo check --workspace; echo "EXIT=$?"
# EXIT 必须 = 0。任何 warning-as-error 级的门禁在此拦下。
```

### 2.1 双产物重编 —— **OOB 修复若改了 `.cu`（staging 布局 / AR kernel / memset）必做**

判据只看一件事：**改了 `.cu` ⟺ `.so` 必须重编 ⟺ 二进制必须重编**（`build.rs` 的 `.build_id` 门禁会拒绝不同源的组合；md5 记指纹）。

```bash
# 本地推 main 后，远端一次做完两产物（顺序不可颠倒）
git push origin main
ssh -o BatchMode=yes ubuntu@43.202.208.136 'cd ~/ferrite && git fetch -q origin && git reset -q --hard origin/main && \
  cd kernels/cuda && bash build.sh 103a && cd ~/ferrite && source ~/.cargo/env && cargo build --release'

# 同源证明（不是假设）：二进制必须内嵌 .so 的 .build_id；顺手记 md5
ssh ubuntu@43.202.208.136 'ID=$(cat ~/ferrite/kernels/cuda/.build_id); \
  echo "build_id=$ID"; \
  echo "embed=$(strings ~/ferrite/target/release/ferrite-serve | grep -cF -- "$ID")  (必须 >=1)"; \
  md5sum ~/ferrite/kernels/cuda/libferrite_kernels.so'
```

> ⚠️ `build.sh` 末句是 `[ ${#SKELETON_FLAGS[@]} -gt 0 ] && echo …`，无 skeleton flag 时它会以 rc=1 退出**却已成功构建**。**成功判据是日志里的 `built …libferrite_kernels.so for sm_103a`**，不是 rc（`batched_400_v2.sh:253-262` 的注释就是为此写的）。
> 若 OOB 修复**只改 `.rs`**（例如只在 `tp.rs` 补 `assert!(len <= self.bytes)`、或只加 ledger/witness 读取），仍建议重编二进制；`.so` 时间戳只需 ≤ 二进制。

### 2.2 新增/改动的 kernel 单测（若修复落在 `.cu`）

```bash
# 全 workspace 单测（本地无 GPU 者用 --no-run 先过编译；GPU 机跑真值）
cargo test --workspace --no-run 2>&1 | tail -5
# AR/collective 隔离微基准（OOB 修复若动 AR kernel，这是最快回归）
DSV41_KERNELS=$PWD/kernels/cuda/libferrite_kernels.so \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 cargo test --release -p ferrite-dsv41 \
  --test ar_micro -- --nocapture
# 可选：把 alternated store-grid（AR stamp fold 的单调 arrival base）也压一遍
AR_MICRO_WORLD=8 AR_MICRO_ROUNDS=32 AR_MICRO_ALT_N=512 \
  DSV41_KERNELS=$PWD/kernels/cuda/libferrite_kernels.so cargo test --release -p ferrite-dsv41 \
  --test ar_micro -- --nocapture
```
**判据**：`ar_micro` 每轮逐元素 = host 参考（各 rank buffer =世界求和），0 failure。

### 2.3 可证伪性负对照（**必须做**）

```bash
# 与 T1 主臂**逐字相同**的 gate 串，只多一个 OOB_GUARD=0（若修复可 gate）
# 预期：canary=0x00000000 + [v5-ledger-RESET] 出现 + epoch 掉到 O(10~100)
grep -c '\[v5-ledger-CANARY\]' $NEG_LOG     # 必须 > 0
grep -c '\[v5-ledger-RESET\]'  $NEG_LOG     # 必须 > 0
```
**判据**：负对照**必须复现**。若修复不可 gate ⇒ 记录归档 pre-fix 日志 `2e42ebb7` 的三行读数与 md5，并显式标注「本条为弱证伪」。

---

## 3. T1 —— OOB 修复核验（主臂）

### 3.0 主臂 gate 串（逐字，一次 serve 一条）

```bash
GATES="DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 \
DSV41_EXPERT_ACT_E4M3=1 DSV41_BF16_TRUNCATE=1 \
DSV41_SH_EXP_MROWS=1 DSV41_MROWS_SMALL_N_ADAPTIVE=1 \
DSV41_GATE_MROWS=1 DSV41_VERIFY_HEAD_MROWS=1 \
DSV41_INDEXER_MROWS=1 DSV41_NORM_MROWS=1 DSV41_COMPRESSOR_MROWS=1 \
DSV41_DRAFT_GRAPH=1 DSV41_DRAFT_P3A=1 DSV41_VERIFY_GRAPH=1 \
DSV41_SWALLOW_STEP=1 DSV41_SWALLOW_DYNAMIC_PAD=1 \
DSV41_V5_LEDGER=1 DSV41_V5_WITNESS=1 \
DSV41_OOB_GUARD=1 \
DSV41_TIMING=1 DSV41_DSPARK_DEBUG=1 DSV41_INV_CHECK=1"

# 三个禁止项（任一出现 ⇒ 该臂 ABORT）：惰性/融合/A2 臂会换掉被测路径
FORBIDDEN="DSV41_LAZY_VERIFY DSV41_HC_VERIFY_FUSE DSV41_HC_FRONT_ROWS"
```

要点（每条都有源码理由）：
- **不设** `DSV41_SWALLOW_EPOCH_PAD`（§0.2：常数 pad + 动态 pad = over-pad）。
- `DSV41_V5_WITNESS=1` 只有在 OOB 修复确实落了 witness 时才加；**没落就别设**（设了没读 = 第 9 种「以为在跑其实没跑」）。gate 名以实施为准。
- `DSV41_OOB_GUARD=1` 为「gate 或直接修」的占位：**直接修则删这一行**，负对照改走 §2.3 备选。
- `DSV41_INV_CHECK=1` 是本设计对既有原语的复用（跨步不变量断言，`chain_dev.rs:2019/2045/9825`）；它报 `[inv-fail]` 是**附加**证据，不作主判据。

### 3.1 启动 + 发请求（远端，取自 `batched_400_v2.sh` 的 `run_case` 骨架）

```bash
NODE=ubuntu@43.202.208.136
PORT=8691
LOG=/tmp/oob_t1/run.log
# 短 prompt 就让第一次 swallowed 步发生（pre-fix 的爆点就在 pos=15），不必跑满
PROMPT='请完整背诵《出师表》全文，从先帝创业未半而中道崩殂开始。'
MAXTOK=200

ssh -o BatchMode=yes $NODE "cd ~/ferrite && pkill -9 -x ferrite-serve 2>/dev/null; sleep 8; \
  nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    LD_LIBRARY_PATH=\$HOME/ferrite/kernels/cuda \
    DSV41_KERNELS=\$HOME/ferrite/kernels/cuda/libferrite_kernels.so \
    $GATES ./target/release/ferrite-serve --model dsv41 --serve --tp 8 \
    --model-dir /opt/dlami/nvme/models/DeepSeek-V4.1-Flash --port $PORT > $LOG 2>&1 &"

# health（最多 5 min）
ssh $NODE "for i in \$(seq 1 60); do curl -s --noproxy '*' -m 2 http://localhost:$PORT/health >/dev/null && exit 0; sleep 5; done; exit 1"

# 实读 env（证明 gate 串活着、FORBIDDEN 没溜进来）
ssh $NODE "tr '\0' '\n' < /proc/\$(pgrep -x ferrite-serve | head -1)/environ | grep -E '^DSV41_' | sort"

# 发一次请求
BODY=$(printf '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"stream":false,"temperature":0}' "$PROMPT" "$MAXTOK")
ssh $NODE "cd ~/ferrite && curl -s --noproxy '*' -m 1800 http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '$BODY'" > /tmp/oob_t1/resp.json

# 收尾：POST /shutdown（不是 kill -INT）
ssh $NODE "curl -s --noproxy '*' -m 5 -X POST http://localhost:$PORT/shutdown"
ssh $NODE "cat $LOG" > /tmp/oob_t1/run.log
```

### 3.2 T1 判据（逐条）

```bash
LOG=/tmp/oob_t1/run.log
```

**(1) canary 不被清零 —— 主判据 A**

```bash
grep -c '\[v5-ledger-CANARY\]' "$LOG"        # 必须 == 0
grep -o 'canary=0x[0-9a-f]\+' "$LOG" | sort -u
#   必须只出现 canary=0xdeadbeef（OOB 修复若改了布局，则以 ledger 的 expected= 为准，见下）
```
> ⚠️ 若 OOB 修复**改了 staging 布局**（如 `ctr_at` 与 `reduced` 间插 guard、或 canary 扩成 4 字），`V5_LEDGER_CANARY` / `V5_LEDGER_CANARY_OFF` 会跟着改（`tp.rs:313/318/361-367`）。**判据必须写成「日志里 canary 值集合 == {expected}」而不是硬编码 0xdeadbeef**，否则会把「布局改对」误判成「回归」。摘 expected 的办法：
```bash
grep -o 'expected=0x[0-9a-f]\+' "$LOG" | sort -u     # 出现即已失败（这些行本身就是 (1) 的违反）
```

**(2) RESET 不触发 —— 主判据 B**

```bash
grep -c '\[v5-ledger-RESET\]' "$LOG"          # 必须 == 0
```
> pre-fix 这里应该是 `prev=999 cur=54 drop=945`（`dspark-correctness-chain.md:5685`）。修好后**一行都不该有**。

**(3) epoch 从 999 继续递增、不到 54 —— 主判据 C**

```bash
python3 - "$LOG" <<'PY'
import re, sys, collections
log = open(sys.argv[1], errors="ignore").read()
# 同时匹配 pre 行与 note 行（两者格式见 chain_dev.rs:9676-9680 / :9701-9705）
pat = re.compile(r"\[v5-ledger(?:-pre)?\] pos=(\d+) rank=(\d+) epoch=(\d+) canary=(0x[0-9a-f]+)")
seq = collections.defaultdict(list)   # rank -> [(pos, epoch)]
for m in pat.finditer(log):
    seq[int(m.group(2))].append((int(m.group(1)), int(m.group(3))))
bad = []
for rank, rows in seq.items():
    for i in range(1, len(rows)):
        if rows[i][1] < rows[i-1][1]:      # 同 rank 下降 = 违反
            bad.append((rank, rows[i-1], rows[i]))
print(f"ranks_seen={sorted(seq)}  lines={sum(len(v) for v in seq.values())}")
print("MONOTONIC", "OK" if not bad else f"FAIL {bad[:3]}")
# 关键单体：出现过 999 之后，任何 epoch 不得 < 999（54 就是违反）
flat = [e for rows in seq.values() for _, e in rows]
first999 = next((i for i, e in enumerate(flat) if e >= 999), None)
if first999 is not None:
    viol = [e for e in flat[first999:] if e < 999]
    print("NO_DROP_BELOW_999", "OK" if not viol else f"FAIL {viol[:5]}")
else:
    print("NO_DROP_BELOW_999 SKIP (never reached 999; 增大 MAXTOK/换长 prompt)")
PY
```
**预期**：`MONOTONIC OK` + `NO_DROP_BELOW_999 OK`；pos=15 那条 `arm=swallowed` 的 epoch **≥ 999**（pre-fix 是 54）。
> 若 `NO_DROP_BELOW_999 SKIP` ⇒ 这一跑太短，第一次 swallowed 步还没把 epoch 推到 999。**把 MAXTOK 提到 ≥ 1000**，或换成 T4 的出师表长跑，别把 SKIP 当 ✓。

**(4) `ar5-hang` = 0 —— 主判据 D（动态 pad 的效果必须保持）**

```bash
grep -c '\[ar5-hang\]' "$LOG"                  # 必须 == 0
# 分开计数（第 9 次设计 §2.3 的 10× watchdog 差）：argmax_rows 带 rows=，pubred 不带
grep -cE '\[ar5-hang\] rank=[0-9]+ peer=[0-9]+ need=[0-9]+ cur=[0-9]+ rows=' "$LOG"   # argmax_rows
grep -cE '\[ar5-hang\] rank=[0-9]+ site=[0-9]+' "$LOG"                                # pubred/bcast
```
（格式源：`dsv41_kernels.cu:8556` 带 `rows=`；`ferrite_kernels.cu:9045/9048` 带 `site=`。）

**(5) SWALLOW 真的在跑（防「gate 活着但没上场」）**

```bash
grep -c 'arm=swallowed' "$LOG"                 # 必须 > 0（batched 子臂至少跑过一轮）
grep -c '\[verify_graph\] captured' "$LOG"     # 期望 > 0（若 gate 生效；见 batched_400_v2.sh 的三态判读）
grep -oE '\[draft_graph\] captured the draft chain at pos=[0-9]+' "$LOG"
```

### 3.3 W5 witness 不变量（**witness 落了才跑；这是「不能把修复后的行为误读为新 OOB」的关键**）

witness 行格式（`epoch54-witness-fix-templates.md §0`）：
```
[v5-witness] pos=<n> rank=<r> writer=<id> e_read=<R> e_wrote=<W> clk=<c>
```

**先验「映射正确」再读结论**——这是 W5 的前置：`writer=<id>` 的 id→kernel 映射必须以**当前树**为准校对，不能照抄旧文档。

```bash
# 1) 枚举当前树所有写 *epoch 的 kernel（判据：只有这些能出现在 witness 里）
grep -n '\*epoch\s*=' kernels/cuda/*.cu

# 2) 摘出 witness 实际出现的 writer id 集合
grep -oE 'writer=[0-9]+' "$LOG" | sort -u

# 3) 逐 id 核 k，并核 W1（写入恒 >= 读值 + 1）与 W5（读了旧字/别字）
python3 - "$LOG" <<'PY'
import re, sys, collections
log = open(sys.argv[1], errors="ignore").read()
pat = re.compile(r"\[v5-witness\] pos=(\d+) rank=(\d+) writer=(\d+) e_read=(\d+) e_wrote=(\d+) clk=(\d+)")
w1_bad, w5_bad = [], []
ids = set()
maxw = collections.defaultdict(int)      # rank -> 已见最大 e_wrote
n = 0
for m in pat.finditer(log):
    n += 1
    pos, rank, wid, er, ew = (int(m.group(i)) for i in range(1, 6))
    ids.add(wid)
    if ew < er + 1:                       # W1: 任何写者写出的值 >= 它读到的 + 1
        w1_bad.append((pos, rank, wid, er, ew))
    if er < maxw[rank]:                   # W5: 不许读到比已见最大值更旧的字
        w5_bad.append((pos, rank, wid, er, maxw[rank]))
    maxw[rank] = max(maxw[rank], ew)
print(f"witness lines={n}  ranks={sorted(maxw)}  writer_ids={sorted(ids)}")
print("W1 e_wrote>=e_read+1  :", "OK" if not w1_bad else f"FAIL {w1_bad[:3]}")
print("W5 e_read>=max_e_wrote:", "OK" if not w5_bad else f"FAIL {w5_bad[:3]}")
# id->kernel 的 k 校核（对照 §3.3 表；pad=id2 的 k=pad 需单独从对照表取，不在此自动判）
PY
```

| 不变量 | 判据 | 违反 ⇒ 结论 |
|---|---|---|
| **W1** | 每条 `e_wrote == e_read + k`（k≥1） | 记录/算术错（结果 A-2） |
| **W5** | 任意行 `e_read ≥ 本 rank 之前已见的最大 e_wrote` | 读了旧字/别字（世界 C′）——**这正是「把修复后的行为误读为新 OOB」的防线** |
| **W3** | 同 pos 同 writer 跨 rank 的 `e_wrote` 逐字相等 | 跨 rank rift |

> **映射自查**（本设计的强制步骤）：若 `writer=<id>` 出现了一个 §3.3 表里没有的 id，或某 id 的 `e_wrote ≠ e_read + k`，**先怀疑 witness 的 id 映射/字段打包，再怀疑 kernel**（`epoch54-witness-fix-templates.md §2.A-2` 的判例）。**在映射自证之前，witness 的一切结论作废**。

### 3.4 T1 通过线（合取）

`(1) 0 行 CANARY` **且** `(2) 0 行 RESET` **且** `(3) MONOTONIC OK 且 NO_DROP_BELOW_999 OK` **且** `(4) ar5-hang==0` **且** `(5) arm=swallowed>0` **且** `(§3.3 W1/W5 OK，若 witness 在跑)`。

---

## 4. T2 —— 计数前 61 行（SWALLOW 正常生成）

**配置**：与 T1 **完全相同的 serve / gate 串**（同一二进制、同一 `.so`、背靠背），只换 prompt。

**Prompt**（`temperature=0, max_tokens=1000, stream=false`）：
```
请从 1 数到 200，每个数字单独一行。
```

**解析与判据**（复用 `l49_ab.sh:479` 的 `first_bad` 口径）：

```bash
python3 - /tmp/oob_t2/resp.json <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))["choices"][0]["message"]["content"]
lines = [l.strip() for l in c.splitlines() if l.strip()]
ok = 0
for i, l in enumerate(lines):
    if l == str(i+1):
        ok += 1
    else:
        break
first_bad = ok + 1
print(f"total_lines={len(lines)}  ok_lines={ok}  first_bad={first_bad}")
print("P1_FIRST61", "OK" if ok >= 61 else f"FAIL (ok={ok})")
# SWALLOW 正常生成：不再 6 token 后 EOS
print("NOT_EARLY_EOS", "OK" if len(c) > 60 else f"FAIL (chars={len(c)}, 疑似早停)")
PY
```

| 判据 | 通过 | 失败 |
|---|---|---|
| **前 61 行** | `ok >= 61`（`1..61` 精确） | `ok < 61` |
| **正常生成** | 答案 chars > 60（远大于 pre-fix 的 6-token EOS） | `chars ≈ 6 token` ⇒ 仍早停 ⇒ ✗ |
| 损坏点交叉（辅助） | `first_bad == 62`（与 EAGER 同模式 = 模型自然行为） | `first_bad < 61` ⇒ 新损坏点 ⇒ ✗（与 EAGER 对照定位） |

> ⚠️ **第 62 行起不判**——`r2-reverification-test-design.md §1` 已证 EAGER 无 spec 也在第 62 行损坏（模型「数数疲劳」）。**前 61 行是红线，不是前 60。**
> 若 `first_bad < 61`：必须与**同会话 EAGER 对照**（去 `DSV41_SPEC/DSPARK`）并列再判，区分「引擎新损坏」与「模型行为」。

---

## 5. T3 —— 吞吐（batched weight-sharing 的收益）

**配置**：T1 主臂 **去掉两个观测门**（`V5_LEDGER=0 V5_WITNESS=0`）——账本每步一次 4B D2H，**计量跑必须关**（`batched_400_v2.sh:157-167` 的 `B400_V5_LEDGER=0` 口径）。prompt = 出师表，`max_tokens=1000`。

**口径**（AGENTS.md）：`tok/s = completion_tokens / e2e_seconds`（end-to-end，不用段平均）；步时用 `[dsv41] step pos=` 与 `[dspark] steps=`。

**测法与解析**：用 `batched_400_v2.sh` 的 `metrics_of`（`scripts/batched_400_v2.sh:340-528`）即可，它已产出 `steady_mean/median/min/p10`、`kacc_mean`、`verify_ms`、图三态、`latin/dbl`。**但需先完成 §9-8 的 gate 更新**（把 `SWALLOW_EPOCH_PAD=1` 换成 `SWALLOW_DYNAMIC_PAD=1`，让脚本测的是修复后的矩阵）。

```bash
B400_V5_LEDGER=0 bash scripts/batched_400_v2.sh
# 产出：/tmp/batched_400_v2/run.{log,dspark,metrics,txt,env,resp.json}
grep -E '^(steady_median|steady_mean|kacc_mean|tok_step|verify_ms|chars|vg_engaged|vg_shapes)=' \
  /tmp/batched_400_v2/run.metrics
grep -c '\[ar5-hang' /tmp/batched_400_v2/run.log
```

| 判据 | 通过 | 备注 |
|---|---|---|
| **e2e tok/s** | **> 91.1**（lazy clean-stack 基线，`AGENTS.md:90`） | 同会话背靠背比；跨会话不复用 |
| **steady_median** | **< lazy 基线的 steady_median** | 步墙是 per-step 真值 |
| **mean-k 可比** | accept 与基线同量级（否则 tok/s 不可配）| accept-N 的步时不能配 accept-M |
| **图 engaged** | `vg_engaged=captured`（或 `failed` 显式报告，不作阻塞） | `no` 要查 gate |
| **ar5-hang** | `== 0` | 动态 pad 的效果保持（同 T1-D） |
| **红线** | `latin=0 dbl=0 has_kaishen=yes`（`batched_400_v2.sh` 的 rc）| 见 T4 的范围修正 |

> ⚠️ 基线口径：91.1 = base + R2 + MARKOV + VERIFY_FORK + RING_WIN（lazy 栈）。lazy 上限 ~145（accept 5）/ ~97（accept 3）。**若 T3 的 tok/s 落在 91.1~97 且 accept≈3，要把「是否真的超过 lazy 上限」分开陈述**，别把「超过 91.1 的一条线」当成「batched 赢 lazy」。

---

## 6. T4 —— 出师表零拉丁（红线）

**配置**：T1 主臂 + 出师表 prompt（`max_tokens=1000`）。可与 T3 共用一次 serve 的日志（T3 已跑出师表）。

**判据（**范围修正**：只判前 ~100 字）**：

```bash
python3 - /tmp/oob_t4/resp.json <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))["choices"][0]["message"]["content"]
head = "".join(c.split())[:100]                  # 去空白后取前 100 字
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
| **前 ~100 字零拉丁** | ASCII 字母数 == 0 | 出现任何拉丁字符 ⇒ 红线 ✗ |
| **`先帝创业未半` 开头** | 逐字正确 | 前缀错 ⇒ ✗ |
| **无相邻双字** | `dbl == 0` | > 0 ⇒ ✗ |

> **范围修正的源码/历史依据**：AGENTS.md 记「出师表零拉丁在 >60 tok 生成下不可达成（模型行为）」——`batched_400_v2.sh` 对**整篇**判 `latin==0` 会把模型自然行为算成失败。**本测试把红线收敛到「前 ~100 字」**（`r2-reverification-test-design.md §3 P2` 同款范围）。若整篇 `latin>0` 但前 100 字零拉丁，判 **T4 ✓**，并在报告里注明「latin 出现在第 N 字后（模型行为）」。

---

## 7. 判据汇总 + 判读树

### 7.1 汇总表

| 测试 | 主判据 | 通过线 | 失败指向 |
|---|---|---|---|
| **T0.0** | `cargo check --workspace` | EXIT=0 | 编译门 |
| **T0.1** | 同源证明 + md5 | `embed>=1` | 双产物纪律 |
| **T0.3** | 负对照复现 | CANARY>0 且 RESET>0 | 测试不能失败 ⇒ 先修测试 |
| **T1** | canary / RESET / epoch / hang / witness | 见 §3.4 | §7.2 |
| **T2** | 前 61 行 + 不早停 | `ok>=61` 且 chars>60 | 引擎损坏 或 模型行为（EAGER 对照） |
| **T3** | e2e tok/s | > 91.1 且 steady_median 下降 | 没提速 ⇒ 查 pad/图 |
| **T4** | 前 100 字 | 零拉丁 + 前缀 + 无双字 | 红线 ✗ |

### 7.2 T1 失败时的判读树（一次定向到唯一分支）

```
T1 跑完，读 [v5-ledger*] 与 [v5-witness]
│
├─ CANARY=0 仍出现（canary != expected）
│   ├─ 有 witness 且某写者 e_read ≤ 53  ──► 世界 A2′/C′：字被清/读了别字 ⇒ 查 epoch_ptr 同一性
│   └─ witness 全正常、ledger 读到坏值   ──► 世界 D：读序/观测错位 ⇒ ledger 改 stream-ordered D2H
│
├─ CANARY 保持 magic 但有 [v5-ledger-RESET]
│   ──► 真·字被写小（A）⇒ 复查 OOB 修复是否覆盖了该写者；补 reduced↔ctr_at 的 guard
│
├─ CANARY/RESET 全绿，但 [ar5-hang] > 0
│   ──► pad 不对称未收敛 ⇒ 查动态 consensus 是否真在 head-of-step 跑（arm=swallowed 计数）
│
└─ 全绿但 epoch 停在某个值不再推进（冻结 ≠ 降级）
    ──► 不是本次 OOB 修复的靶子 ⇒ 退回 swallow-fix11 §2 的「冻结 laggard」分支
```

> **核心纪律**：`epoch 54 → EOS 提前` 曾是假设（`epoch54-final-fix-path.md §0 M2`），**只有 OOB 修复后 EOS 恢复正常，才证明这条因果成立**。T2 的「chars>60」是它的直接证伪点。

---

## 8. 判据的「不能误读」清单（三处易错，全部有源码根据）

1. **canary 偏移可能随修复改变** —— 判据写「日志值集合 == expected」，不写死 `0xdeadbeef`；否则把「布局改对」误判为回归（`tp.rs:313/318`）。
2. **witness writer id 必须先自证映射** —— 出现未知 id 或 `e_wrote ≠ e_read+k`，先查 witness 打包，不先查 kernel（`epoch54-witness-fix-templates.md §2.A-2`）。**这是「不把修复后的行为误读为新 OOB」的防线（W5）。**
3. **`ar5-hang` 要分 `rows=` 与不带 `rows=` 两类** —— argmax_rows 与 pubred 的 watchdog 差 10×，合并计数会掩盖其中一类（`chain_dev.rs:1865-1866`）。

---

## 9. 交付清单（供尚书省分派；OOB 修复落地的同时）

| # | 项 | 内容 | 优先级 | 风险 |
|---|---|---|---|---|
| 1 | T0.0 | `cargo check --workspace` EXIT=0 | P0 | 无 |
| 2 | T0.1 | `.cu` 变则 `build.sh 103a` + `cargo build --release` + 同源证明 + md5 | P0（双产物纪律）| 无 |
| 3 | T0.3 | 负对照（`OOB_GUARD=0` 或归档 pre-fix 日志）**必须复现** | P0（可证伪性）| 无 |
| 4 | T1 | 主臂 gate 串 + 五判据 + W1/W5 witness 校验 | P0 | 无 |
| 5 | T2 | 计数 1→200，前 61 行 + 不早停 | P0 | 无 |
| 6 | T3 | 观测关（`V5_LEDGER=0 V5_WITNESS=0`）+ 出师表计时，tok/s > 91.1 | P1 | 无 |
| 7 | T4 | 出师表前 ~100 字零拉丁（范围修正版）| P0（红线）| 无 |
| 8 | `scripts/batched_400_v2.sh` | **正式化新矩阵**：`SWALLOW_EPOCH_PAD=1` → `SWALLOW_DYNAMIC_PAD=1`（§0.2）；若 OOB 修复有 gate，一并入 `GATES`；`FORBIDDEN` 增列「同时设 `EPOCH_PAD` 与 `DYNAMIC_PAD`」这种 over-pad 组合 | P1 | 低（改 gate 串，需重跑 T3/T4 基线）|
| 9 | `crates/ferrite-models/src/dsv41/tp.rs` | 若修复侧要补：v5 路径 `assert!(len <= self.bytes)`（`publish()` 只在 v2 有，`tp.rs:770`）| P2 | 低 |

**不做**：
- 不把 `*epoch = e+k` 改成 `atomicMax`（`atomicMax` 会**掩盖清零**——把旧高值粘住，54 再也观察不到，症状伪装成「修好」，`epoch54-final-fix-path.md §7` 明确反对）。**先上观测拿到事实，再决定语义。**
- 不新增 gate（新增 gate = 新增一处「以为在跑其实没跑」）。
- 不在 T1 之前改任何 kernel 语义。

---

## 10. 一句话总结

OOB 修复后要证的**不是「输出变好了」**，而是三个设备侧事实**同时**成立：**canary 回到 magic（没被越界踩）、epoch 同 rank 单调不回退（没被清零重数）、RESET 恒 0**；再用 T2（前 61 行 + 不早停）证明「epoch 54 ⇒ EOS 提前」这条因果确实断开了，T3 证明动态 pad 没有把 batched 的收益吃掉，T4 守红线。**而这一切的前提，是 T0.3 的负对照必须先红**——一个不会失败的测试，绿得再整齐也等于零信息。

---

*工部 · 只读勘察 + 设计；唯一产出为本文件。未改动任何源码、未执行 GPU 命令。*
*所有行号以工作树 HEAD `f7cc53b` 为准；无法从源码定论的推断均显式标注 `[未验证]` 并给出证伪条件。*
