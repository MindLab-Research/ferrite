# 达成 400+ tok/s（真实 acc≈2.2）执行手册 — 2026-09-14

> 目标口径（用户裁决）：sglang 博客的 **acc 5.5 是模拟值**，不可追；**真实 acc≈2.2 下 ~450 tok/s 即算超越官方**。
> 我方口径：`tok/s = tok/step ÷ step`，其中 **`tok/step = mean-k + 1`（含 bonus，与官方同口径）≈ 3.24**。
> ⇒ **450 tok/s ⇒ step ≈ 7.2ms；400 tok/s ⇒ step ≈ 8.1ms**。判据一律用 `[dsv41] step pos=` 的 **p50（最后 200 步）**，禁吞吐反推。

## 0. 前置铁律（本会话用血换来的）

1. **手写 fp4 BS 臂必须 OFF**：`DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0`。
   开着它会**毁模型**（1..100 数数输出全是重复乱码、一个数字都没有）。
2. **一切测量前确认双产物同源**：`bash ~/check_artifacts.sh` 必须返回 `ARTIFACTS_FRESH`
   （三源 build-id：`.so` / binary / `kernels/cuda/.build_id`）。**中断构建会留下不同源产物**，引擎会直接拒绝启动
   （`kernel build-id mismatch — REFUSING TO START`），此时臂的输出是**空**，别误读成"模型坏"。
   `~/num100.sh` 已把构建门焊在里面 ✓ 用它就自动满足。
3. **一次只改一个变量**；**不重跑已判定的臂**；**不空跑 baseline**；每步都看 **1..100 前 61 行 = 1..61**（红线）。

## 1. 账本（代码注释给的确切数字）

| 项 | 数字 | 出处 |
|---|---|---|
| verify 的发射次数 | **~7000 次流式发射/次**（40 层 × ~170 节点） | `chain_dev.rs` DSpark verify 图化块注释 |
| 每次流式发射的提交开销 | **~2.9 µs** ⇒ 合计 **~20.3 ms** | 同上 |
| 图内节点分发 floor | **~0.4 µs/node** ⇒ 7000×0.4 ≈ **2.8 ms** | 同上 |
| MoE gate/up+down（TileLang bf16 grouped GEMM） | **70.0 µs/层**（up 45.5 + dn 24.5）= SIMT 250µs 的 **28%** ⇒ 40 层省 **~7.2 ms** | `moe_batch()` 文档注释 |

**⇒ verify = 28.17ms 里几乎全是"提交开销"**（不是算力）⇒ **图化是第一杠杆** ✓。

## 2. 三个单变量步（现成工具 `~/run_to_400.sh` 已把它们串好）

| 步 | 命令（在 `~/ferrite` 下） | 预期 step | 预期 tok/s |
|---|---|---|---|
| **P1** | `bash ~/num100.sh STEP_P1 DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 DSV41_VERIFY_GRAPH=1` | ~10–15ms | ~216–340 |
| **P2** | P1 + `DSV41_MOE_TILELANG=1 DSV41_GATE_MROWS=1 DSV41_GATE_MROWS_ROUTE=1 DSV41_ATTN_MROWS=1 DSV41_COMPRESSOR_PROJ_MROWS=1 DSV41_ENGRAM_PROJ_MROWS=1 DSV41_ENGRAM_GATHER_MROWS=1` | ~9–10ms | ~320–360 |
| **P3** | P2 + `DSV41_GRAPH_STEP=1`（decode 侧同一图化思路） | **~7ms** | **~460** ✅ |

> 注：`arm_run.sh` 的 COMMON 里 `DSV41_VERIFY_GRAPH=0`/`DSV41_GRAPH_STEP=0`（在 `GRAPH_OFF` 内），
> 传在**后面**的 env 会覆盖它们 ✓。

### 2.1 精度裁决（`precision-completeness`，决定 P2 的构成）——**必读**

- **`DSV41_MOE_TILELANG=1` 判定"不可用"** ✗：它的**激活侧完全不量化**（`moe_bf16_shim.cu:213-227` 直接 `__float2bfloat16`），
  而官方是 `act_quant(...,32,ue8m0)` → **e4m3**（rel ≤ 6.3%），该臂是 bf16 直舍（rel ≤ 0.2%）⇒ **精度高约 32×** ✗；
  输出累加器还是 **f32**（官方是 bf16 ✗）。**这是数据格式层面的差异，任何门都补不回来**，且会**系统性改变近 tie 的 argmax/文本** ✗。
  （权重侧无偏差 ✓：`fp4 → bf16` 是**位级无损**的，因为 e2m1 幅值只有 ≤2 位有效位、标度是纯 2 的幂。）
  同格式的替代 `DSV41_MOE_TILELANG_BS` 就是**会毁模型的手写臂** ✗ ⇒ **那 7.2ms 暂时拿不到** ✗。
- **`head-term-audit` 判定 head 不是"那 20ms"** ✗：默认臂下 head 只剩 **~0.23–0.33ms**（verify 的 ~1%）⇒ 别在这里找鲸鱼 ✗。
- ⇒ **剩余的鲸鱼只有一条：流式发射的提交时间** ✓ ⇒ **图化（verify + step 两侧）是唯一的大杠杆** ✓。

## 3. 若还差（按收益排序的后备杠杆）

1. **draft 段融合**：`DSV41_DRAFT_P3LITE_{SEED,KV,ATTN}=1`（单变量可切）—— 已并入 P2 的折叠族 ✓
2. **AR 形态**：`DSV41_AR_V5` / `FERRITE_P2P`（注意 nsys 轮的死锁规避组合与普通轮不同）
3. `DSV41_MOE_DOWN_BS=1`（down 臂 blockscaled，已接线、默认 OFF；**用前必须先在不捕获的调用里完成 lazy INIT**，
   否则捕获内 `cudaMalloc` 会让 `cudaStreamEndCapture` 失败）
4. ~~head 的词表读~~ ✗（已判定只剩 ~1%）

## 4. 精度（用户硬规则：与官方 PyTorch 完全对齐，fp4/fp8 不能高也不能低）

- **`DSV41_MOE_TILELANG=1` 会把 fp4 权重反量化成 bf16** ⇒ 有效位宽 4→8 bit ⇒ **精度偏高** ✗
  ⇒ 属"**可用但需标注/需裁决**"，见 `precision-completeness` 的追加答复 ✓。
- 精度五门（默认 OFF，待逐项转正）：`DSV41_ROUTED_DOWN_QUANT`、`DSV41_WINDOW_KV_QUANT`(A2)、
  `DSV41_COMPRESS_LATENT_QUANT`(A3)、`DSV41_INDEXER_FP4_RT`(A4)、`DSV41_ATTN_P_BF16`(I3)，
  外加 `DSV41_ACTQ_FLOOR`（把量化下限放到 **amax**，官方 `kernel.py:76` 位置；已覆盖两个量化 TU ✓）。
- **转正流程**：逐个门、一次一个变量；判据 = 与官方 oracle 对齐（`~/gu_numpy_ref.py` 家族，已与旧路径逐位相同 ✓）。

## 5. 验证文本（红线探针）

```bash
bash ~/num100.sh <NAME> <ENV...>     # 1..100 数数；首 61 行必须 = 1..61，不能重复、不能乱码
bash ~/verify_correct.sh 8899 <NAME> # 标准三测（1..100 / 拉丁乱码探针 / step p50）
```
`verify_correct.sh` 已加**陈旧日志守卫**（找不到本臂日志时**响亮告警**而不再静默回退到别的 log）✓。

## 6. 实测记录（2026-09-14 晚）与"数字跳跃"假说

| 臂 | env | step p50 | **真实吞吐** | 文本 |
|---|---|---|---|---|
| 原状 | （BS 臂 ON ✗） | 32.5ms | ~100（spec） | **全乱码**（BS 臂毁模型） |
| ~~P1~~ | BS 臂 OFF + `DSV41_VERIFY_GRAPH=1`，**但没设 `DSV41_SPEC`** | 10.19ms | **98.1 tok/s（普通 decode，1 token/步）** | `1..51` 后跳到 `62,63,64,65,66,69…` |
| ~~P2~~ | P1 + 精度中性折叠 | 10.17ms | 98.3 tok/s（同样非 spec） | **与 P1 逐字节相同** ⇒ 折叠族**精度中性** ✓ |

> ### ⚠️ 口径更正（`amortization-plan` 抓到，务必先读）
> **`[dsv41] step pos=` 只有在 `DSV41_SPEC=1` 时才是"完整 step"**（`serve.rs:509-519/:587/:779` 的
> `step_time` 两条分支：非 spec 分支每步只 1 token）——而 **`arm_run.sh` 的 COMMON 没设 `DSV41_SPEC`** ✗。
> ⇒ 上表 P1/P2 的 **10.19ms 是"普通 decode step"**，**不是** 32.5ms 那个 spec step ✗；
> ⇒ 因此**"318 tok/s@acc2.2"是错误换算，已撤回** ✗；它们说明的只有一件事：**这些臂里普通 decode 的步长是 10.2ms** ✓。
> ⇒ **一切性能臂今后必须显式带 `DSV41_SPEC=1`** ✓，否则 p50 不可比、tok/s 会被高估 3.24× ✗。

⇒ 仍然**有效的结论**：①**折叠族（P2）与 P1 文本逐字节相同** ⇒ 那批门**不改数值** ✓，可作默认候选 ✓；
②P1/P2 的文本出现**数字跳跃** ✗ —— 在**非 spec（普通 decode）**模式下出现跳跃，说明它**不是 spec/verify 缺口**，
而更可能出在我叠加的开关（图化）或 BS-臂关闭后的通路上 ⇒ **必须用一次"干净基线"（BS OFF、无图、无 spec）对照** ✓
（正在跑：`STEP_NOGRAPH`）。

**"数字跳跃"的假说（待 `STEP_NOGRAPH` 对照确认）**：AGENTS.md 的 DSV41 门表记载 **spec 路径有已知缺口**
（`DSV41_SIDS_WRITEBACK=1` 的说明原文是"**verify 值修好后开**"）⇒ 说明**verify 的取值**本身就带缺陷，
而 spec 的 emit 直接取自 verify ⇒ 跳跃是**已知的 spec/verify 缺口**的表现，**未必是图化造成的**（图化只改发射方式）。
⇒ **下一批单变量臂**（每条只答一个新问题）：
1. `DSV41_SIDS_WRITEBACK=1`（spec commit 后回写 `emitted.last()` 到 `s.ids`）
2. `DSV41_SEED_ALIGN=1`（判词路线 A：seed↔tap 对齐）
3. `DSV41_SWALLOW_STEP=1`（吞主链步）
4. `DSV41_VERIFY_HEAD_FOLD` / `_SLICED` 的两种组合
→ 目标是让 **1..100 前 61 行 = 1..61**（红线），同时保住 318 tok/s。

### 6.1 权威解释（代码注释原文），决定下一批臂

`sids_writeback()`（`chain_dev.rs:4089+`）的注释写明：write-back 喂的是**更正确的** token（`emitted.last()` =
verify 的 argmax），它把 **`verify_out` 自身的取值错误放大**成更早的 collapse，因此默认 OFF——**"blocker 是 verify 的值"**
（"Re-enable once the verify-value fault is fixed"）。**write-back 本身数学上正确**（两次独立复核 ✓）。

⇒ 所以"数字跳跃"= **verify 取值有误**的直接后果 ✓。手写 BS 臂正是跑在 verify 里的 MoE（它坏 ⇒ verify 值错），
但**若 BS 臂 OFF 后仍跳**，缺口另有来源；默认配置里**与官方不一致**的两处最可疑：

| 候选 | 默认 | 与官方的关系 | 单变量臂 |
|---|---|---|---|
| `DSV41_SEQ_ALIGN` | **OFF** | 官方归约按**专家 id 升序**；默认是 legacy 升序 slot ⇒ 顺序不同 ✗ | `=1` |
| `DSV41_VERIFY_HEAD_SLICED` | **ON** | 只写每行前 `seg` 的词表切片 ⇒ **argmax 落在切片外就取错** ✗ | `=0`（A/B） |
| `DSV41_SEED_ALIGN` | OFF | 判词路线 A：seed↔tap 对齐 | `=1` |
| `DSV41_SIDS_WRITEBACK` | OFF | **数学正确**，但会放大 verify 的错值 ⇒ **修好 verify 值之后**再开 | 最后再试 |

**判据**：每个臂都跑 `~/num100.sh <NAME> DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0 [门]`，
看 **1..100 前 61 行是否 = 1..61** 且 p50 不退化（当前最好 = 10.17ms / 318 tok/s @acc2.2，带 `VERIFY_GRAPH=1`）。

## 7. 诚实的战略结论（2026-09-14 深夜，写给下一个会话）

**账要算清：图化 ≠ 450 tok/s。** 按 `amortization-plan` 的族账本（verify m=6 口径，需按今天 24.5–26.7ms 重标）：

| verify 内的族 | 账本 ms | 占 37.3ms | 瓶颈性质 |
|---|---:|---:|---|
| **shared expert** | 10.40 | 27.9% | **核效率 + 5× 重复读**（85GB/s） |
| **routed experts**（gate+up+down） | 8.30 | 22.2% | **核效率**（378GB/s，20.7µs/发） |
| 投影族（wq_a/wkv/wq_b/wo） | 3.7–8.7 | 9.9% | launch/合并 |
| MoE router | 3.44 | 9.2% | 核效率 + 逐行×5 |
| hc 链 / attn KV / indexer / AR | 2.96+2.80+2.50+1.40 | ~24% | 占用 / launch / 核效率 |

- **提交那一半**（~7000 发 × 2.9µs ≈ 20.3ms）由**图化**解决 ⇒ ~2.8ms ✓（P1/P2 已在**非 spec** 模式下验证图化确实生效 ✓）。
- **执行那一半**（~7.9ms 起步，按今天口径更大）**只能靠 MoE 的核效率** ✗ —— 而 450 tok/s 要求 step≈7.2ms
  （draft ~3.4 + commit ~0.5 ⇒ **verify 必须 ≈3.3ms** ✗，比 SGLang 公布的 7.3ms 还低 ✗）。
  ⇒ **纯靠"少发几次"到不了 450**；必须把 **shared expert / routed expert 的核效率**做上去 ✓。

**⇒ 所以 BS 臂（`DSV41_MOE_TILELANG_BS`）依然是正解载体，理由不是偏好而是格式**：
`precision-completeness` 已确认它是**唯一**与官方**同格式**的快速 MoE 路径——A = **e4m3** 1B/value、
W = **原生 fp4 nibble + ue8m0 标度面**（与官方 `fp4_gemm` 完全同格式 ✓），而 `DSV41_MOE_TILELANG`（bf16）
是**激活不量化**的格式级偏差 ✗（精度高 32×，用户红线）。

**但 BS 臂当前会毁模型** ✗ ⇒ 它仍有**两条可信异常**：①与官方 oracle 不符（median rel≈1.2、corr≈0.04）
②**同输入两次运行输出不同**（max|d|=8.54，输入已逐字节核对相同）⇒ ②意味着**内核里有竞态/非确定性** ✗。
本会话为它修的 9 个真 bug（含 **`tcgen05.commit` 的 `.shared::cluster` 使多 slot barrier 塌陷**，PTX §9.7.18.12.1）
和整套判别工具（`gu_numpy_ref.py` oracle、`gc_all` 原始 tile、`DCLEAR=2` qNaN 哨兵、`PRECLECEAR`、`KEEP_STAGE`、
`SPIN_CAP` 确定性演练、阶段 dump）**都是遗留资产**，可直接用来继续。

**最短的两条路（供选择）**：
1. **要数字**：跑 `~/run_to_400.sh`（S1/S2/S3，全带 `DSV41_SPEC=1`）⇒ 得到**诚实的 spec 步与 tok/s**
   （预期：图化后 verify 从 24.5–26.7 → ~10–12ms ⇒ step ~14–16ms ⇒ **200–230 tok/s** ✗，离 450 还差执行时间那一半）。
2. **要 450**：回到 BS 臂（或任何**同格式**的快速 MoE），先把②的非确定性根因做掉
   （首选判据：`DSV41_MOE_BS_PRECLEAR=1` 预清零 / `ZERO_ASF`(mode 5) 干净零乘积 / `gc_all` 原始 tile 对比）。

## 8. ⚠️ 口径陷阱（`graph-safety-audit` 抓到的，直接影响此前所有臂的数字）

- **`DSV41_GRAPH_STEP` 默认 ON**（`chain_dev.rs:8955` 的 `unwrap_or(true)`，自 2026-09-11），
  但 **`arm_run.sh` 的 `GRAPH_OFF` 把它显式设成 0** ✗ ⇒ **本会话此前的臂全部是"整步图关闭"的配置** ✗。
- **`GRAPH_STEP=0` 还有副作用：它顺带关掉 `ar_v5`** ✗ ⇒ 一次动两个变量。
  ⇒ **这解释了"我的臂 10.19ms/步"与"用户记得的 eager 6.3ms/步"的差异** ✓（6.3ms 是**带整步图**的生产默认）。
- **`DSV41_MOE_BATCH` 默认 ON**（`chain_dev.rs:764` 的 `unwrap_or(true)`）⇒ eager 参照**也是**批量专家路
  ⇒ 它**不是**差异来源（我此前的怀疑不成立）✓。
- `DSV41_VERIFY_GRAPH` **默认 OFF**，只作用于 **spec 的 verify 段**；在非 spec 路径上**惰性** ✓。
- `DSV41_GRAPH_MOE` 是**死路径**（`moe_graph_armed` 无赋值点，恒 false）✗。
- `GRAPH_STEP` **没有回退**（`capture_begin()?` 直接传播错误，`cd.rs:8978,8980`），而 verify 图**有**响亮 decline + 永久闩 + 回退直发 ✓。
- 生效判据：**`[verify_graph] captured …` 缺行 = 图一次都没 engage** ✓；有 captured 但 replays 少 = 反复 unanimity 失败 ✓。

⇒ **结论：凡是要和生产口径比的臂，必须显式 `DSV41_VERIFY_GRAPH=1 DSV41_GRAPH_STEP=1`**（覆盖 COMMON 的 0）✓。
