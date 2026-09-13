# R0 / R1 最小验证手册 —— acc-improve-path 判决的前两步（零风险诊断）

> 本文件是 `accept-ceiling-analysis.md` §5.2 / §6 的 **S0（R0）** 与 **P3（R1）** 的落地手册：
> 只写代码与命令，**不含任何远端 GPU 操作**。GPU 命令集中在 §6，由主 agent 串行执行。
>
> 代码入口：`crates/ferrite-models/src/dsv41/acc_hist.rs`（新模块）、
> `chain_dev.rs`（四臂挂钩 + oracle probe）、`dspark_dev.rs`（`import_tap_row`）。

---

## 1. 为什么需要这两步（判决书前提）

当前票面是**一个数**：`[dspark] steps=… mean-k=… tok/step=…`。而 accept 链是**前缀链**
（`ferrite_types::spec_accept`，`while matched < lim && drafts[matched] == judges[matched]`），
所以每个 round 只有两种命运：

* **首链接断**：`drafts[0] != judges[0]` ⇒ `k_acc = 0`（一次只吐 1 个 token）；
* **走到尾部**：`k_acc -> DSPARK_DRAFTS`。

两个判决书给出的票面：

| 量 | 值 | 来源 |
|---|---|---|
| 首 token 单点率 `p1` | ≈ **0.43** | `accept-ceiling-analysis.md` §1.3 |
| 尾部条件率 `q` | ≈ **0.82** | 同上（已贴 SGLang 的 0.86） |
| 当前最优栈 `mean-k` | **2.240** | `verify-amortization-lesion-audit.md:186`（SWALLOW + TAP_INPUT） |

**核心矛盾**：旧直方图的**均值上界是 1.93**，与 2.240 **数学互斥** ⇒ 当前栈的直方图
**从未被测过**。在拿到同臂直方图之前，任何"改进 accept"的方案都在和鬼影打架——因为
`mean-k` 相同、形状不同（首链接断 vs 尾部早停）对应完全不同的修法。

* **R0** = 校准：把当前栈的直方图与 `p1/p_j` 摆出来，同时验证 `2.240` 这个数自己站得住。
* **R1** = 判别：给 draft head 喂**主链自己在该位置的 hidden**，看它能不能跟主链的贪心。
  这是分开「几何/输入错」与「head 能力不够」的**唯一决定性**实验（`accept-ceiling-analysis.md` §5.2）。

---

## 2. 代码改动清单（本次）

| 文件 | 改动 |
|---|---|
| `crates/ferrite-models/src/dsv41/acc_hist.rs` | **新增**。`DSV41_ACC_HISTOGRAM` / `DSV41_ORACLE_TAP` 两个 `OnceLock` 门；逐 step 打印、直方图 + `p1/p_j/tail_q` 汇总、oracle 计数与汇总；2 个单元测试（p 阶梯定义 + 分箱 clamp） |
| `crates/ferrite-models/src/dsv41/mod.rs` | 注册 `pub mod acc_hist;` |
| `chain_dev.rs` | **R0**：四个臂的 `k_acc` 计算处各加一行 `acc_hist::note(...)`（`legacy` / `aligned` / `swallowed` / `lazy`）。**R1**：`oracle_tap()` 门（`DSV41_ORACLE_TAP=1`）+ `DevChain::oracle_tap_probe`，在 `dspark_spec_swallowed` 的 **post-verify / pre-accept** 处调用 |
| `dspark_dev.rs` | `DsparkDev::import_tap_row(tap_r, rows, row)`：把 `dspark_tap_r` 的**某一行**（`[slot][rows][dim]` 布局）gather 进 `main_h`，不做投影/种窗（`draft_forward` 自己做） |
| `ferrite-dsv41/src/serve.rs` | 请求结束（`DecodeRun` 收尾）与进程收尾各打一次 `print_summary("serve" / "serve-shutdown")`（rank 0） |
| `ferrite-dsv41/src/bin/dsv41-run.rs` | one-shot 收尾打一次 `print_summary("oneshot")` |
| `AGENTS.md` | 两张门表加 `DSV41_ACC_HISTOGRAM` / `DSV41_ORACLE_TAP` 两行 |

**默认行为零变化**：两个门都默认 OFF（`OnceLock` 缓存，per-step 无 `getenv`）。
OFF 时每个挂钩是 1 个 cached bool 分支 + 1 次 early return，**不动任何数值路径**。

---

## 3. R0：同臂直方图 + p1/p_j 分解

### 3.1 门与输出

```bash
DSV41_ACC_HISTOGRAM=1   # 默认 OFF
```

**逐 step 行**（每个 spec round 一行，四条臂共用格式）：

```text
[acc-hist] step=37 arm=swallowed pos=412 k_acc=0 first_match=false p1_running=0.4324
```

* `step`：本进程内的 step 序号（`AtomicU64`，不是 `pos`）。
* `arm`：本 round 真正跑的臂（`legacy` = bootstrap 轮，`swallowed` = 稳态轮；
  `aligned`/`lazy` 只在对应门开时出现）。**这一列是"同臂"口径的证据**——
  若稳态轮里混进大量 `legacy`，说明 `spec_primed` 没立住，直方图不是 SWALLOW 的。
* `first_match`：`drafts[0]` 是否等于该 round 的**首判词**（锚行 argmax；legacy 臂是 `next`）。
  它与 `k_acc >= 1` 是同一个事件，打印出来是为了**独立交叉验证** accept 记账。
* `p1_running`：到本步为止的**在线** `p1`（只看 `k_acc>0` 的比例）。

**收尾汇总行**（请求结束 / 进程收尾 / one-shot 结束各一次）：

```text
[acc-hist-summary] tag=serve steps=197 mean-k=2.2411 tok/step=3.2411 hist={0:41 1:33 2:34 3:38 4:26 5:25} p1=0.7919 first_match=0.7919 p_j={ 0.7922 0.8025 0.7532 0.7308} tail_q=0.7697 arms={swallowed:196 legacy:1} oracle: steps=197 hit=171 miss=26 rate=0.8680
```

（数字是**格式示意**，不是测量值。）

### 3.2 统计量的定义（照抄自 `acc_hist.rs` 的模块文档）

`hist[k]` = `k_acc = k` 的步数（`k = 0..=DSPARK_DRAFTS = 5`），`N = Σ hist[k]`：

```text
p1   = (N - hist[0]) / N                        首链接率
p_j  = Σ_{k>=j} hist[k] / Σ_{k>=j-1} hist[k]    第 j 链接的**条件**率（j = 2..=5）
mean-k = Σ k·hist[k] / N                        与 [dspark] 行的 mean-k 同口径
tail_q = mean(p_2, p_3, p_4, p_5)               "尾部"的平均条件率
```

注意 `p_j` 是**条件**率（给定链走到第 j-1 链接）——这正是判决书里 `p1 ≈ 0.43` 与
`q ≈ 0.82` 的对照口径。`first_match` 与 `p1` 必须相等（见 §3.1）；不等 ⇒ 臂的 accept
记账本身有问题，先查这个再说别的。

### 3.3 判据（R0 的三件事）

1. **`mean-k` 必须复现 2.240 ± 噪声。** 对不上 ⇒ 直方图所属的栈 ≠ 判决书的栈（先查门表与
   `arms={}`），**不要**拿它做后续推理。这一步就是校准的意义。
2. **`p1` 与 `q` 的复现**：判决书给 `p1≈0.43 / q≈0.82`；若实测 `p1` 明显不同，说明
   "首 token 单点"这一诊断需要重新表述（直方图会直接告诉我们断点在哪）。
3. **形状**：`hist[0]` 与 `hist[5]` 的双峰性。`hist[0]` 大 ⇒ 首链接主导 ⇒ **R1 是对的实验**；
   若 `hist[0]` 小、而 `hist[1..4]` 厚 ⇒ 断点在链中段，`p1` 不是主要损失项，R1 的判据要换成
   "第 j 链接的 oracle 率"（同一套设施，见 §4.4 的扩展）。

### 3.4 零风险论证

* 只在 `k_acc` **已经算出来之后**读它，不参与任何决策、不写设备内存、不发 collective。
* 逐 step 行是一次 `eprintln!`（rank 0 才有意义；其余 rank 的行是重复的，分析时只取 rank 0）。
* 不影响 `[dspark]` 计时行：挂钩处无同步、无 D2H。

---

## 4. R1：oracle tap 对照

### 4.1 问题

SWALLOW 臂里，draft 在**锚行 forward 之前**跑，所以它拿不到 `pos` 处的 hidden，只能吃
`carry_kept_tap` 递过来的**上一轮最后一条 KEPT 行**——位置 `pos - 1`（`carry_kept_tap` 取
`keep - 1`）。也就是说：**draft 被问的是 `pos`，喂它的却是 `pos-1` 的 hidden。**

同一个 `p1 ≈ 0.43` 有两种解释：

| 假设 | 含义 | 若成立，后续投什么 |
|---|---|---|
| **H2 几何/输入** | head 本来能跟，只是 tap 相位/层/块几何错 | `SEED_ALIGN` 一类几何修复**有效** |
| **H1 能力** | 即便喂对 hidden，3 层 head 也复现不了主链的近 tie | 几何修复**无效**，只能加深 draft/换头/降块长 |

### 4.2 探针设计（`DSV41_ORACLE_TAP=1`）

**做法**：在本轮 verify 块刚跑完、**accept 之前**，把 `dspark_tap_r` 的**第 0 行**
gather 进 `main_h`，然后用**同一个 `(token, pos)`** 再跑一次 `draft_forward`，
比较新的 `drafts[0]` 与 `rows[0]`。

**为什么第 0 行就是"主链自己的 hidden"**：SWALLOW 块是 `[token, d1..d5]` 落在
`pos..pos+5`，**第 0 行就是 `token` 在 `pos` 处的 forward**，其 hidden 正是 draft 被问的
那个位置的主链 hidden（`TAP_INPUT=1` 时是层的 attention 输入、默认是层输出——与线上 tap
同一个采集点）。`rows[0]` 就是主链在 `pos+1` 的贪心 argmax（= `next`），而 `drafts[0]`
propose 的正是 `pos+1`。所以：

```text
oracle rate = P( drafts[0] == rows[0] | draft 吃的是 pos 处的主链 hidden )
```

**逐 step 行**：

```text
[acc-oracle] pos=412 draft_top=12345 main_top=12345 hit=true
```

**汇总**并入 `[acc-hist-summary]` 的 `oracle: steps=… hit=… miss=… rate=…`。

### 4.3 判据（判决书的映射）

| oracle rate | 判定 | 后续 |
|---|---|---|
| **≥ 0.85** | **H2（几何/输入）** | head 没问题 ⇒ 线上 `p1≈0.43` 是 tap 相位/几何错的 ⇒ **`SEED_ALIGN` 类修复值得投**（P2/P4/P5 排序不变） |
| **≈ 0.43**（无提升） | **H1（能力）** | 喂对输入也跟不上 ⇒ 几何修复**不解决**首链接 ⇒ 转 P6（加深 draft / 换头 / 降块长） |
| 高，但**只在主链 top1-top2 gap 小处翻转** | 不可约熵 | 天花板由负载定；改不动 |
| 高，但**主链 gap 大处也翻** | 输入配对系统偏 | tap 内容/层/位置整体错 ⇒ 干净的相位检验 |

### 4.4 扩展（可选，同一设施）

* **逐链接**：`import_tap_row(tap_r, VERIFY_ROWS, j)` 给的是第 j 行的主链 hidden，配
  `rows[j]` 就是第 j 链接的 oracle。当前只做 `j = 0`（首链接是判决书的要害）。
* **静默面**：R1 只报"命中与否"。若要 top-1/top-2 gap 分布，把 `DsparkDev` 的 logits 行
  用现成的 `dump_unit("logits_row0", …)` 拉出来即可（`DSV41_DSPARK_UNIT_DUMP` 设施）。

### 4.5 零风险论证（为什么探针不动任何状态）

探针位于 **post-verify / pre-accept**，此时：

1. **commit 还没发生**，`pos_ctr` 仍在 `pos` ⇒ `DSV41_DRAFT_GRAPH` 的 replay 定位到**同一
   个 ring 槽**（`slot_dev` 由设备计数器导出），不会写错槽。
2. **accept 用不到 draft 的任何输出**：`k_emit` 只用 `drafts`（本轮 verify 前已 D2H）与
   `rows`；commit 只用 `dspark_tap_r` 与 `host_mirrors` —— 三者 draft forward 都不写。
3. **draft 自己的状态幂等**：`seed_window(s, pos)` 只是 `(s, pos)` 的函数，重跑写同一槽。
4. **`main_h` 会被覆盖**，但下一轮的 `import_tap` 在任何人读它之前用 carried tap 重写。
5. **失败只记日志、不传播**（`vrow0_step` 的纪律）：本轮块合法、commit 必须发生，探针不许
   对一个已决的 step 报错。

**代价**：每轮多一次 draft forward（~4-9 ms）。这是**定位工具，不是性能模式**——
A/B 计时轮里不要开。

**已知边界**：只接在 **SWALLOW 臂**（当前最优栈的臂）。`lazy` 臂有 host_barrier + 行循环的
锁步纪律，额外一次 draft 可能打破 lockstep；`legacy`/`aligned` 臂本身就拿的是同位 tap
（`step_dev` 先跑），oracle 对照在那里没有信息量。这两类臂**只挂 R0 直方图**。

---

## 5. 交叉验证（不花 GPU 的检查）

* `first_match == p1`（同一次运行的 `[acc-hist-summary]`）：不等 ⇒ accept 记账有问题。
* `mean-k`（`[acc-hist-summary]`）vs `[dspark] steps=… mean-k=…`（`DSV41_TIMING=1`）：两者
  同口径，应当一致；不一致 ⇒ 直方图的 step 集合与计时行的 step 集合不同（例如有的步走了
  shadow 分支）。
* `Σ hist[k] == steps`。
* `cargo test -p ferrite-models --lib acc_hist`（p 阶梯定义 + 分箱 clamp）。

---

## 6. GPU 验证命令清单（主 agent 串行执行；本手册不含任何远端操作）

> **纪律**：一次一个 serve；一 serve 一 prompt（`[dspark]` / `[acc-hist]` 累加器是**进程级**的）；
> 同一会话背靠背比较；收尾一律 `POST /shutdown`。
>
> **执行位置**：serve 绑的是**节点上的** 8320 端口，所以下面每一段都是**一条 `ssh` 命令**
> 把整段脚本喂给节点执行（`bash -s`），本地不留半截状态。`~/.cargo/env`、`$HOME/ferrite`、
> `NCCL_NVLS_ENABLE=0` 都是节点侧的既有前提（见 `AGENTS.md`）。

### 6.0 部署（双产物同源，**必做**）

```bash
# 本地 → 远端
git push origin main
ssh -o BatchMode=yes ubuntu@43.202.208.136 'cd ~/ferrite && git fetch -q origin && \
  git reset -q --hard origin/main && cd kernels/cuda && bash build.sh 103a && \
  cd ~/ferrite && source ~/.cargo/env && cargo build --release'
```

启动前确认 **binary 内嵌 build id == `.so` 的 `.build_id`**（不一致 ⇒ 数字无意义，
`batched_400_v2.sh` 的门禁逻辑照抄）。

### 6.1 R0 —— 同臂直方图（SWALLOW + TAP_INPUT + 2.24 栈）

一条 ssh 跑完「起 → 等就绪 → 一次请求 → 红线 → shutdown → 取数」。

```bash
ssh -o BatchMode=yes ubuntu@43.202.208.136 'bash -s' <<'REMOTE'
set -u
cd ~/ferrite || exit 9
LOG=/tmp/r0_hist.log; OUT=/tmp/r0_hist.json; PORT=8320
pkill -9 -x ferrite-serve 2>/dev/null; sleep 4

setsid env \
  NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 DSV41_TIMING=1 \
  DSV41_SWALLOW_STEP=1 DSV41_SWALLOW_EPOCH_PAD=1 DSV41_VERIFY_GRAPH=1 \
  DSV41_EXPERT_ACT_E4M3=1 DSV41_BF16_TRUNCATE=1 \
  DSV41_TAP_INPUT=1 DSV41_DRAFT_BF16_DOMAIN=1 \
  DSV41_DRAFT_P3A=1 DSV41_DRAFT_GRAPH=1 \
  DSV41_ACC_HISTOGRAM=1 \
  LD_LIBRARY_PATH=$HOME/ferrite/kernels/cuda \
  timeout 900 ./target/release/ferrite-serve --model dsv41 --serve --tp 8 \
    --model-dir /opt/dlami/nvme/models/DeepSeek-V4.1-Flash --port $PORT \
  > "$LOG" 2>&1 < /dev/null &

# 就绪轮询（在节点上，最多 ~8 min）
READY=0
for i in $(seq 1 80); do
  grep -q "chain ready, serving" "$LOG" && { READY=1; echo "READY ~$((i*6))s"; break; }
  grep -qi "build-id mismatch" "$LOG" && { echo "!!!!! BUILD-ID MISMATCH !!!!!"; break; }
  pgrep -x ferrite-serve >/dev/null || { echo "!!!!! serve died before ready"; break; }
  sleep 6
done
[ "$READY" = 1 ] || { echo "=== NOT READY, tail ==="; tail -30 "$LOG"; exit 2; }

# 单次请求：出师表 max_tokens=300（与判决书同负载、同文本）
curl -s --noproxy "*" -m 300 http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"请完整背诵《出师表》全文。"}],"max_tokens":300,"stream":false}' \
  > "$OUT"

# 收尾（R0 的汇总在这里打；不要 kill -INT）
curl -s --noproxy "*" -m 10 -X POST http://localhost:$PORT/shutdown >/dev/null
sleep 3; pkill -9 -x ferrite-serve 2>/dev/null

echo "=== 红线（不能重复、不能乱码） ==="
python3 - "$OUT" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))['choices'][0]['message']['content']
bad=[i for i in range(1,len(c)) if c[i]==c[i-1] and not c[i].isspace()]
print('LEN',len(c),'双字',len(bad)); print('HEAD',''.join(c[:160].split()))
PY
echo "=== ★ 直方图汇总 ==="; grep "\[acc-hist-summary\]" "$LOG"
echo "=== 交叉验证：mean-k 计时行 ==="; grep "dspark\] steps=" "$LOG" | tail -2
echo "=== arms 分布 / 逐 step 行数 ==="; grep -c "\[acc-hist\]" "$LOG"
REMOTE
```

**读什么**：`[acc-hist-summary]` 的 `hist={…}`、`p1`、`p_j`、`tail_q`、`arms={}`，
并按 §3.3 判。

### 6.2 R1 —— oracle tap 对照

与 6.1 **同一个栈、同一个 prompt**，只把 `DSV41_ACC_HISTOGRAM=1` 加/换成
`DSV41_ORACLE_TAP=1`（两个都开也可以，日志同时给直方图与 oracle）。

```bash
ssh -o BatchMode=yes ubuntu@43.202.208.136 'bash -s' <<'REMOTE'
set -u
cd ~/ferrite || exit 9
LOG=/tmp/r1_oracle.log; OUT=/tmp/r1_oracle.json; PORT=8320
pkill -9 -x ferrite-serve 2>/dev/null; sleep 4

setsid env \
  NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 DSV41_TIMING=1 \
  DSV41_SWALLOW_STEP=1 DSV41_SWALLOW_EPOCH_PAD=1 DSV41_VERIFY_GRAPH=1 \
  DSV41_EXPERT_ACT_E4M3=1 DSV41_BF16_TRUNCATE=1 \
  DSV41_TAP_INPUT=1 DSV41_DRAFT_BF16_DOMAIN=1 \
  DSV41_DRAFT_P3A=1 DSV41_DRAFT_GRAPH=1 \
  DSV41_ACC_HISTOGRAM=1 DSV41_ORACLE_TAP=1 \
  LD_LIBRARY_PATH=$HOME/ferrite/kernels/cuda \
  timeout 900 ./target/release/ferrite-serve --model dsv41 --serve --tp 8 \
    --model-dir /opt/dlami/nvme/models/DeepSeek-V4.1-Flash --port $PORT \
  > "$LOG" 2>&1 < /dev/null &

READY=0
for i in $(seq 1 80); do
  grep -q "chain ready, serving" "$LOG" && { READY=1; break; }
  pgrep -x ferrite-serve >/dev/null || { echo "!!!!! serve died"; break; }
  sleep 6
done
[ "$READY" = 1 ] || { tail -30 "$LOG"; exit 2; }

curl -s --noproxy "*" -m 300 http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"请完整背诵《出师表》全文。"}],"max_tokens":300,"stream":false}' \
  > "$OUT"

curl -s --noproxy "*" -m 10 -X POST http://localhost:$PORT/shutdown >/dev/null
sleep 3; pkill -9 -x ferrite-serve 2>/dev/null

echo "=== ★ oracle rate（判据） ==="; grep "\[acc-hist-summary\]" "$LOG"
echo "=== oracle 明细（前 20 行） ==="; grep "\[acc-oracle\]" "$LOG" | head -20
echo "=== hit/miss ==="; grep -c "hit=true" "$LOG"; grep -c "hit=false" "$LOG"
echo "=== 探针失败（应为 0） ==="; grep -c "oracle tap probe" "$LOG"
echo "=== 红线 ==="
python3 - "$OUT" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))['choices'][0]['message']['content']
bad=[i for i in range(1,len(c)) if c[i]==c[i-1] and not c[i].isspace()]
print('LEN',len(c),'双字',len(bad))
PY
REMOTE
```

**读什么**：`oracle: … rate=`。按 §4.3 的阈值映射判 H2 / H1。

### 6.3 阴性对照（可选，确认探针本身可靠）

R1 的 probe 若把 gather 行换成 `k_emit-1`（即 `carry_kept_tap` 那一行），
`oracle rate` 应等于同期 `[acc-hist]` 的 `p1`——**这是"探针没有把输入搞坏"的阴性对照**。
当前代码只做 row 0（oracle 臂）；该对照需要一行改动
（`import_tap_row(..., 0)` → `import_tap_row(..., k_emit - 1)`），留给需要时再做。
`k_emit` 在 `dspark_spec_swallowed` 的作用域内就是 `k_acc + 1`。

### 6.4 计时轮纪律

`DSV41_ORACLE_TAP=1` 会给每轮加一次 draft forward ⇒ **禁止**在计时/吞吐 A/B 里开。
R0 的 `DSV41_ACC_HISTOGRAM=1` 只是一次 `eprintln!`，影响在噪声以下，但仍建议诊断轮与
计时轮分开跑。

---

## 7. 决策树

```text
R0：mean-k ≈ 2.240 且 arms 主要是 swallowed？
 ├─ 否 ⇒ 栈不对（门表/priming），先修口径，不要往下走
 └─ 是 ⇒ 看 hist[0]
      ├─ hist[0] 厚（首链接主导）
      │    └─ R1 oracle rate
      │         ├─ ≥0.85 ⇒ H2 几何派 ⇒ 投 SEED_ALIGN 类几何修复（P2/P4/P5）
      │         └─ ≈p1   ⇒ H1 能力派 ⇒ 转 P6（加深 draft / 换头 / 降块长）
      └─ hist[0] 薄（断点在中段）
           ⇒ "首 token 单点"表述需重写；把 R1 扩成逐链接（§4.4）
```

---

## 8. 已知限制 / 待办

1. R1 只接 SWALLOW 臂（理由见 §4.5 末）；`lazy` 臂的版本需要先解决它的锁步纪律。
2. R1 只报 `j = 0` 的命中率，不给 logits gap 分布（§4.4 的扩展留着）。
3. `DSV41_ORACLE_TAP=1` 与 `DSV41_DSPARK_UNIT_DUMP=1` 同开时，unit dump 可能 armed 在
   **探针那次 forward** 上（它是"第一个 `pos > 0` 的 forward"）——要拿 golden 就分开跑。
4. 直方图是**进程级累计**（请求之间不 reset）：一 serve 一 prompt 才能把一次运行的数当
   "一个样本"读（与 `batched_400_v2.sh` 的口径一致）。
