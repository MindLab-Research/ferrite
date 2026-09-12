# OOB 修复后的 SWALLOW 测试结果分析框架（每种结果的判定 + 下一步行动）

> 工部 · 2026-09-12 · **只读勘察 + 设计；未改动任何源码、未执行 GPU 命令。**
> 输入（现场核对）：`crates/ferrite-models/src/dsv41/tp.rs`、`crates/ferrite-models/src/dsv41/chain_dev.rs`、
> `crates/ferrite-dsv41/src/serve.rs`、`kernels/cuda/ferrite_kernels.cu`、`scripts/batched_400_v2.sh`、
> `docs/agent/{oob-fix-verification-test-design.md,epoch54-alternate-fix-design.md,dspark-correctness-chain.md,swallow-result-action-plan.md}`。
> 代码基线：工作树 HEAD `8365345`（`main`，OOB 修复提交）。
> 本框架的职责：**接收测试日志 → 判「修复成功 / 症状换形 / 独立 bug」→ 给出唯一下一步**，不重复测试设计（执行命令见 `oob-fix-verification-test-design.md`）。

---

## 0. 四条前提校正（不先钉死，判读表会整体读错）

> 这四条是**现场 grep 出来的事实**，其中 C1/C2 与 `oob-fix-verification-test-design.md` 的当前文字冲突——
> 该文档写作时把这两项当"占位待实施"，实施已给出答案。**以本节的 grep 为准。**

### C1 —— OOB 修复是**直接修**，`DSV41_OOB_GUARD` gate **不存在**

```bash
grep -rn 'OOB_GUARD' --include=*.rs --include=*.cu crates kernels   # = 0 行
# 命中只出现在 docs/agent/oob-fix-verification-test-design.md（设计占位文字）
```

Plan A 的 guard/4-canary 是 `Collective::new` 里**无条件**的布局（`tp.rs:384/405-430`），Plan B 的
`check_payload` 是**无条件** `assert!`（`tp.rs:596`）——两者都没有 env 开关。

⇒ **结论（三条连锁）**：
1. 测试设计 §3.0 主臂 gate 串里的 `DSV41_OOB_GUARD=1` **必须删**。留着 = 第 9 种「以为在跑其实没跑」的幻影门（一个没被任何代码读取的 env）。
2. 负对照（§2.3 首选）**无法做** ⇒ 只能走 §2.3 **备选**：用归档的 pre-fix 日志 `2e42ebb7` 的三行读数作红线基准，**并显式标注「弱证伪」**（只证明史上失败过，不证明当前测试能抓回归）。
3. ⟹ **T1 的「绿」信息量下调一档**。补强办法：把 `[v5-ledger-GUARD]` / `[v5-ledger-CANARY]` 的**出现判据本身**当成半可证伪项——它们能在当前 binary 上确实打出（`chain_dev.rs:9648/9678` 有打印点），所以「0 行」不是恒真；只是无法在当前 binary 上把"应该红"跑出来。**这一点必须在报告里写明，不得省略。**

### C2 —— V5 **WITNESS 没有落**，§3.3 的 W1/W3/W5 检查全部 N/A

```bash
grep -rn 'v5-witness\|v5_witness' --include=*.rs crates   # = 0 行
```

witness 的模板文档（`epoch54-witness-fix-templates.md`）在，**实现不在**。OOB 修复提交（`8365345`）只含 Plan A + Plan B + Ledger，无 witness。

⇒ **结论**：
1. gate 串里**不许**设 `DSV41_V5_WITNESS=1`（设了没读 = 同一族幻影病）。
2. 测试设计 §3.3 整节 **N/A**，不进 T1 通过线。
3. **反向哨兵**：若日志里**真**出现 `[v5-witness]` 行 ⟹ 跑的 binary 不是当前树 ⟹ **停下来核对产物同源**（`.build_id` / md5），不要开始分析。

### C3 —— `check_payload` 是 `assert!`（panic/abort），**不是** `Err`

任务判读表把它写成「check_payload 触发（Err）」。实际是**硬 panic**：

```rust
// tp.rs:596-603
fn check_payload(&self, len: usize) {
    assert!(len <= self.bytes,
        "collective payload {len} > slot {} (a v5 AR payload must fit the staging slot \
         — see docs/agent/epoch54-final-fix-path.md §5-A2′)", self.bytes);
}
```

⇒ 触发即 **rank 进程 panic / abort**（不是优雅返回、不是 hang、不写任何 `[v5-ledger-*]`）。它的可观测是
**panic 文本**里 `collective payload` 关键字 + backtrace，二不是 ledger 行。判读必须走**另一条 grep 通道**（见 §2 主判据 G）。
> ⚠️ 若 serve 用 `catch_unwind`/线程隔离兜住 panic，则表现为**该请求失败 / 该 rank 退出**而非整进程死——两种都算"Plan B 命中"，都要摘入口名。
> ⚠️ gate 位置按入口不同：多数入口在函数**首行无条件**断言；`all_reduce_inplace` 夹在 `if ar_v5()` 内（`tp.rs:957-958`）。所以"没触发"在有 v5 的臂上才算证据。

### C4 —— 布局改了 ⇒ 偏移全变 ⇒ **判据一律用日志里的 `expected=`，不硬编码 `0xdeadbeef`**

修复把 `ctr_at` 上移了 `V5_LEDGER_GUARD_BYTES = 8`（`tp.rs:384`），canary 从 1 字扩成 4 字
（`V5_LEDGER_CANARY_OFFS = [8,16,32,48]`，`tp.rs:333`）。当前 magic 值**仍是** `0xDEADBEEF`，且 guard 与
canary **共用同一 magic**（`V5_LEDGER_GUARD = V5_LEDGER_CANARY`，`tp.rs:356`）。

⇒ 判据写成 **「日志里 canary 值集合 ⊆ {expected}」**；guard 与 canary 的**区域区分靠 `word=`/`slot=` 字段**，不靠值（两者值相同！）。
> 若后续再改 magic，硬编码 `0xdeadbeef` 的检查会把"布局改对"误判成"回归"——这正是设计文档 §3.2(1) 的警告，此处执行它。

---

## 1. 修复引入的信号清单（测试要盯的行）

| # | 行前缀 | 完整格式 | 源码 | 修复角色 |
|---|---|---|---|---|
| S1 | `[v5-ledger-stream]` | `rank={r} stream={:?} world={w}` | `chain_dev.rs:9625` | 一次性；证 ledger 与 v5 kernel 同 stream（world A2 的廉价证伪） |
| S2 | **`[v5-ledger-GUARD]`** | `pos={p} rank={r} arm={a} word={w} value={v:#010x} expected={:#010x} — the 'reduced' array was overrun into ctr_at's guard (tp.rs layout)` | `chain_dev.rs:9648` | **Plan A 新增**：`reduced`→epoch 的 8B guard 被踩 |
| S3 | `[v5-ledger-CANARY]` | `pos={p} rank={r} arm={a} slot={s} off={o} canary={v:#010x} expected={:#010x}` | `chain_dev.rs:9678` | **扩展**：1→4 槽；`slot/off` 点名哪一槽 |
| S4 | `[v5-ledger-RESET]` | `pos={p} rank={r} arm={a} prev={q} cur={e} drop={d}` | `chain_dev.rs:9702` | epoch 同 rank 下降（per-rank） |
| S5 | `[v5-ledger-pre]` | `pos={p} rank={r} epoch={e} canary={c:#010x} arm=pre` | `chain_dev.rs:9715` | 步骤 STARTING epoch（hang 时也能归因） |
| S6 | `[v5-ledger]`（note） | `pos={p} rank={r} epoch={e} canary={c:#010x} arm={a} k_emit={k} delta={d}` | `chain_dev.rs:9739` | 步骤 POST epoch + 增量 |
| S7 | **panic `collective payload {len} > slot {bytes}`** | 进程 stderr / `<req> 500` | `tp.rs:599` | **Plan B 新增**：v5 AR 载荷越槽 |
| S8 | `[ar5-hang]` | argmax_rows: `rank=… peer=… need=… cur=… rows=`；pubred/bcast: `rank=… site=…` | `dsv41_kernels.cu:8556`；`ferrite_kernels.cu:9045/9048` | epoch rift（回归哨兵） |

> **区域语义（`tp.rs` 布局，修复后）**：`[parity0][parity1][stamps: world*4][reduced: world*4]` → **`[guard: 8B = 2 words]`** → `ctr_at` = `{epoch(+0), A4 flag(+4)}` → 尾 64B 内 `{canary@+8, +16, +32, +48}`。
> guard 在 `reduced` 与 epoch **之间**；canary 在 epoch **之后**。**打 guard = 来自 reduced 方向的越界；打 canary = 已越过 guard/epoch 的越界。**

---

## 2. 主判据（A–H，落成命令；**全过才叫修复成功**）

```bash
LOG=/tmp/oob_t1/run.log
```

| 判据 | 命令 | 通过线 |
|---|---|---|
| **A** canary 不被动 | `grep -c '\[v5-ledger-CANARY\]' "$LOG"` <br> `grep -o 'canary=0x[0-9a-f]\+' "$LOG" \| sort -u` | `== 0`；值集合 ⊆ `{expected}`（**不硬编码**，见 C4） |
| **B** guard 不被动（**新增核心**） | `grep -c '\[v5-ledger-GUARD\]' "$LOG"` <br> `grep -o 'value=0x[0-9a-f]\+' "$LOG" \| sort -u` | `== 0`；`value` 集合 ⊆ `{expected}` |
| **C** 不降级 | `grep -c '\[v5-ledger-RESET\]' "$LOG"` | `== 0` |
| **D** epoch 单调 | §4 parser：`MONOTONIC OK` 且 `NO_DROP_BELOW_999 OK` | 两者 OK（**SKIP 不当 ✓**，见 §6） |
| **E** SWALLOW 正常生成 | `grep -c 'arm=swallowed' "$LOG"`；max_tokens 计数 / 生成 token 数 | `arm=swallowed > 0` **且** 生成 `> 60` token（**不再 6-token EOS**） |
| **F** 不 hang | `grep -c '\[ar5-hang\]' "$LOG"`（可再按 argmax_rows / pubred 分列） | `== 0` |
| **G** Plan B 未触发 | `grep -c 'collective payload' "$LOG"` | `== 0` |
| **H** gate 真上场 | `grep -c 'arm=swallowed' "$LOG"` + §5 门核对 | `> 0` 且 env 逐门相符 |

**通过 = A∧B∧C∧D∧E∧F∧G∧H。** 缺任一项证据 **不得**下"成功"结论（项目铁律：缺证据 exit 2）。
**且**：报告必须附 §0-C1 的「弱证伪」声明——负对照不可做，绿的强度已标注。

---

## 3. 判读顺序（**先环境，再主判，后症状**）

```
§5 门核对(E_env)  ──失败──▶ 停：读数与配置不符，作废
      │OK
      ▼
A/B/C 越界三判据 ──任一命中──▶ R1 / R2（越界仍在）
      │全 0
      ▼
D epoch 单调    ──失败/降级──▶ R3（无越界的 epoch rift）
      │OK
      ▼
G Plan B panic  ──命中──▶ R4（入口越界）
      │无
      ▼
F ar5-hang      ──命中──▶ R6（hang 独立源）
      │0
      ▼
E SWALLOW 生成  ──仍 6 token──▶ R5（EOS 独立问题）
      │>60 tok
      ▼
H 上场确认      ──未上场──▶ 停：gate 活着没上场
      │OK
      ▼
   ✅ R0 全绿：OOB 修复成功 → SWALLOW 解锁
（全程 canonical 信号为 0：ledger 零行 ⇒ R7）
```

---

## 4. 统一 parser（一次摘全，判据 A–D 共用）

```bash
cat > /tmp/oob_analyze.py <<'PY'
import re, sys, collections
log = open(sys.argv[1], errors="ignore").read()

# ---- 门（E_env）：证明 T1 主臂真在跑，且 FORBIDDEN 没溜进来 —— 由 §5 单独做 ----

# ---- A/B：canary 与 guard 的值集合 + 是否点名 ----
canary_lines = re.findall(r"\[v5-ledger-CANARY\][^\n]*", log)
guard_lines  = re.findall(r"\[v5-ledger-GUARD\][^\n]*", log)
cset = set(re.findall(r"canary=(0x[0-9a-f]+)", log))
gset = set(re.findall(r"\[v5-ledger-GUARD\][^\n]*?value=(0x[0-9a-f]+)", log))
eset = set(re.findall(r"expected=(0x[0-9a-f]+)", log))
print(f"A/canary lines={len(canary_lines)}  values={sorted(cset)}  expected={sorted(eset)}")
print(f"B/guard  lines={len(guard_lines)}  values={sorted(gset)}")
for l in canary_lines[:4]: print("   CANARY:", l.strip())
for l in guard_lines[:4]:  print("   GUARD :", l.strip())

# ---- C：RESET ----
reset = re.findall(r"\[v5-ledger-RESET\][^\n]*", log)
print(f"C/RESET  lines={len(reset)}")
for l in reset[:4]: print("   RESET :", l.strip())

# ---- D：每 rank epoch 序列（pre + note 合流），单调 + 不跌破首个 >=999 ----
pat = re.compile(r"\[v5-ledger(?:-pre)?\] pos=(\d+) rank=(\d+) epoch=(\d+)")
seq = collections.defaultdict(list)
for m in pat.finditer(log):
    seq[int(m.group(2))].append((int(m.group(1)), int(m.group(3))))
bad = [(r, seq[r][i-1], seq[r][i]) for r in seq for i in range(1, len(seq[r])) if seq[r][i][1] < seq[r][i-1][1]]
print(f"D/ranks={sorted(seq)}  epochs_lines={sum(len(v) for v in seq.values())}")
print("  MONOTONIC        :", "OK" if not bad else f"FAIL {bad[:3]}")
flat = [e for rows in seq.values() for _, e in rows]
first999 = next((i for i, e in enumerate(flat) if e >= 999), None)
if first999 is None:
    print("  NO_DROP_BELOW_999: SKIP (never reached 999 — raise MAXTOK/prompt, NOT a pass)")
else:
    viol = [e for e in flat[first999:] if e < 999]
    print("  NO_DROP_BELOW_999:", "OK" if not viol else f"FAIL {viol[:5]}")

# ---- F：ar5-hang 分类 ----
hang_rows  = len(re.findall(r"\[ar5-hang\] rank=\d+ peer=\d+ need=\d+ cur=\d+ rows=", log))
hang_site  = len(re.findall(r"\[ar5-hang\] rank=\d+ site=\d+", log))
print(f"F/ar5-hang total={hang_rows+hang_site} (argmax_rows={hang_rows} pubred/bcast={hang_site})")

# ---- G：Plan B panic ----
pl = re.findall(r"collective payload \d+ > slot \d+", log)
print(f"G/check_payload panics={len(pl)}")
for l in pl[:4]: print("   PANIC :", l)

# ---- E/H：swallow 真上场 ----
print(f"E+H/arm=swallowed lines={len(re.findall(r'arm=swallowed', log))}  "
      f"verify_graph captured={len(re.findall(r'\[verify_graph\] captured', log))}")
PY
python3 /tmp/oob_analyze.py "$LOG"
```

---

## 5. 门核对（E_env，先做；不做则下面的读数全部作废）

```bash
# 实读 env，证明 gate 串活着、且 C1/C2 的两个幻影门 + FORBIDDEN 都不在
ssh $NODE "tr '\0' '\n' < /proc/\$(pgrep -x ferrite-serve | head -1)/environ | grep -E '^DSV41_' | sort"
```
通过线：
- **含** `DSV41_SWALLOW_STEP=1 DSV41_SWALLOW_DYNAMIC_PAD=1 DSV41_V5_LEDGER=1`（动态 pad 开、常数 pad **不**开）；
- **不含** `DSV41_SWALLOW_EPOCH_PAD`（否则 over-pad ⇒ 换成了"双 pad bug"，见设计 §0.2）；
- **不含** `DSV41_LAZY_VERIFY` / `DSV41_HC_VERIFY_FUSE` / `DSV41_HC_FRONT_ROWS`（FORBIDDEN）；
- **不含** `DSV41_OOB_GUARD` / `DSV41_V5_WITNESS`（C1/C2：当前树不读这两个 env）；
- 缺任一项 ⇒ 该臂作废，重跑，**不要**开始判读。

---

## 6. 结果判定表（逐行 = 一种观测 → 判定 → 唯一下一步）

> 观测组合互斥且**穷尽**（含"ledger 零行"这一兜底）。命令见 §4 输出字段。

| 编号 | 观测（§4 字段） | 判定 | 下一步行动 | 优先级 |
|---|---|---|---|---|
| **R0** | `A=0 B=0 C=0` ∧ `MONOTONIC/NO_DROP_OK` ∧ `E: >60tok` ∧ `F=0 G=0 H>0` | ✅ **OOB 修复成功**（且 EOS 消失） | **SWALLOW 解锁 → batched 优化 → 400 冲刺**（`swallow-result-action-plan.md` §2.2 档 B 起）。附 C1 弱证伪声明 | — |
| **R1** | `B>0`（guard 打行，`value≠expected`） | 越界**仍在**，但 `reduced`→epoch 的 guard **接住了**（Plan A 生效、Plan B 没覆盖该入口） | 读 `word=`（0 或 1）与 `value=`：把 value 当"越界落点的原始数据"反推 **是哪个 `reduced`/stamp 写者越过尾部**（对齐 `world*4` 边界算第几个 writer）。把该写者的 `len` 与 `stride` 对 6 入口表核 | **P0** |
| **R2** | `A>0`（`B==0` 或 `B>0`）canary 打行 | 越界**越过 guard/epoch** 直接踩尾（可能是**非 reduced 方向**：stamp / ready_row / 宽写） | 读 `slot=`（`0/1/2/3` → `off=8/16/32/48`）：slot 越大 = 写越宽/越远。`B==0` 且 `A>0` = guard 没挡住 ⇒ 越界**不从 reduced 来**，改查 stamp/ready_row 方向 | **P0** |
| **R3** | `C>0` ∧ `A=0` ∧ `B=0` | epoch 降级但**无任何越界痕迹** = **world A2（stale write，非 OOB）** | 独立 bug 线：查 stream（`[v5-ledger-stream]` 与 kernel stream 是否同一）/ pad / arm 相位；走 `epoch-54-alternate-fix-design.md` 的 A2 分支，**不要**再改 OOB | **P1** |
| **R4** | `G>0`（`collective payload …`） | Plan B **命中**：某 v5 入口 `len > bytes` | `grep 'collective payload'`；带 backtrace/入口名跑，用 §1 的 6 入口表定位：<br>`all_reduce_inplace_pubred_only` `tp.rs:611` · `all_reduce_inplace_hcpost` `:653` · `all_reduce_inplace_add` `:715` · `all_reduce_inplace_hcpost_add` `:753` · `all_reduce_inplace_hcpost_rows` `:818` · `all_reduce_inplace` `:946`。<br>**那个入口的 `len > bytes` 就是越界源** | **P0** |
| **R5** | `A=B=C=0` ∧ `D OK` ∧ `F=0 G=0` ∧ `E: 仍 6 token` | OOB **修了**，EOS 问题**独立存在** | 调查 EOS 的独立来源（与越界无关）：draft/verify 域漂移、`spec_primed`/`carry_kept_tap` 下标、EOS logit。冻结 OOB 线，开独立 issue | **P1** |
| **R6** | `F>0`（`[ar5-hang]`） | OOB **不是** hang 的唯一原因 | 按 §4 分列 `argmax_rows` vs `pubred/bcast`：前者查 `rows=` 形状，后者查 `site=`。对齐 `spec_primed_unanimous`（arm 相位）+ 动态 pad（`chain_dev.rs:9766+`）。**与 OOB 解耦**分析 | **P1** |
| **R7** | §4 全部计数 `=0`（**ledger 零行**） | 不是 R0–R6 任一条：**ledger 从未产出** | 二选一先证：(a) gate 没开/没生效 → 重跑 §5；(b) 进程 hang 在**第一步**（`ar5-hang` 的"ledger 打 0 行"签名，`chain_dev.rs:9580`）→ 转 R6 且用 `[v5-ledger-pre]` 的缺失确认。**不得**把"零输出"当"无越界" | **P0** |

**速读版**（对应任务表）：
- canary 保持 + RESET=0 + SWALLOW 正常 → **R0 成功** → 冲刺；
- canary 清零 + GUARD 行 → **R1**（看 GUARD 值 → 越界源）；
- canary 保持 + RESET=0 + SWALLOW 仍 6 token → **R5**（EOS 独立）；
- check_payload 触发 → **R4**（哪个入口 = 越界源）；
- 仍 ar5-hang → **R6**（hang 独立源）；
- **补充分支**：guard 单独命中（R2 的对偶）→ **R1**；ledger 零行 → **R7**。

---

## 7. 陷阱与边界（这些是"看似成功/看似失败"的误判点）

1. **判据不硬编码 `0xdeadbeef`**（C4）：用 `expected=`。guard 与 canary 同 magic ⇒ 只能用 `word=`/`slot=` 区分区域。
2. **`NO_DROP_BELOW_999 SKIP` 不是 ✓**：跑太短、epoch 没到 999。**把 MAXTOK 提到 ≥1000**或换长 prompt 重跑，别当通过（设计 §3.2(3)）。
3. **`delta` 是 `wrapping_sub`**：设备侧 `e+1u` 语义；出现大 delta 先怀疑 wrap/相位，别直接判 RESET。
4. **RESET 是 per-rank、pre+note 合流**：`v5_ledger_probe` 同时喂两者 ⇒ pre→note 的下降**已在 probe 内**报出（`chain_dev.rs:9722-9730`），`note` 不另建 seen 表。**不要**另写一个 seen 表（那是死代码，也会与 probe 打架）。
5. **ledger 读失败 ≠ 没跑**：S1–S6 是 **OBSERVATION**，D2H 失败只打 `[v5-ledger] … read failed` 并返回 `None`，步骤继续（`chain_dev.rs:9657/9670/9692`）。所以"某行缺"可能是读失败，要 grep `read failed` 排除。
6. **Plan B 无 gate ⇒ 负对照弱**（C1）：报告必须显式标注"绿色为弱证伪下的通过"，不可写成与有 gate 时同等强度。
7. **`[v5-witness]` 出现 = 产物不同源**（C2）：不是好消息，先核对 `.build_id`/md5，停分析。
8. **`check_payload` 多数入口无条件**（C3）：所以"未触发"只有在**该臂确实走了 v5 AR**（`arm=swallowed>0` 或 ledger 有行）时才算证据。

---

## 8. 交付给实施者的两处改动（本框架的 action item，非源码）

| # | 改动 | 位置 | 理由 |
|---|---|---|---|
| 1 | **删**主臂 gate 串里的 `DSV41_OOB_GUARD=1` | `oob-fix-verification-test-design.md:139`（及 §3.0 正文 :149） | C1：树里无此 env，留着 = 幻影门 |
| 2 | **删**主臂 gate 串里的 `DSV41_V5_WITNESS=1`（并标 §3.3 N/A） | 同文档 §3.0 / §3.3 | C2：witness 未实现，设了没读 = 幻影病 |

> 这两项**不改源码**（`只读 + 设计`），只改测试配置与判读口径。实施者照此更新后，§5 的门核对才与 gate 串自洽。

---

## 9. 一页纸结论

- **先证伪、再判绿**：负对照不可做（C1）⇒ 报告须标"弱证伪"；`[v5-ledger-GUARD]`/`[v5-ledger-CANARY]` 的**0 行**是当前 binary 上可失败的量（打印点在 `chain_dev.rs:9648/9678`），不是恒真。
- **三判据是主判**：`A(canary=0)` ∧ `B(guard=0)` ∧ `C(RESET=0)`；`D(epoch 单调)` 定"降级还是越界"；`G(payload panic)` 定"越界源入口"；`F(ar5-hang)` 与 `E(>60tok)` 定"独立问题"。
- **唯一下一步由 R0–R7 决定**：R0 → 冲刺；R1/R2/R4 → 追越界源；R3/R5/R6 → 各开独立线；R7 → 先证 gate/hang。
- **判读顺序**：门核对 → A/B/C → D → G → F → E → H（缺证据 exit 2）。
