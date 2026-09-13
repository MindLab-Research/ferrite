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

## 9. ⚠️⚠️ 最重要的一条：**eager ≠ verify**，只有 verify 的数才算数

用户 2026-09-14 的纠正，必须放在最前面：

- **MTP/spec 的 verify 走的是 `chain_dev.rs::moe_rows`（以及 `layer_rows` 系列）**；
  **eager/decode 走的是 `chain_dev.rs::moe`（以及 `layer`）** —— **两套独立实现** ✗。
- ⇒ **在 eager 路上量到的任何改进（图化、折叠、步长）对 MTP 的 tok/s 都没有意义** ✗。
  （本会话此前的 P1/P2 就是犯了这个错：`DSV41_SPEC` 未设 ⇒ 量的是单行 decode 步 ✗。）
- ⇒ **唯一有意义的判据**：`DSV41_SPEC=1` 下的 `[dsv41] step pos=` p50 ✓（`~/run_to_400.sh` 的每条臂都带它 ✓）。
- ⇒ 而 **verify 的慢是"发射次数"性质**：~7000 次/verify（40 层 × ~170 节点）× ~2.9µs submit ≈ **20.3ms**
  ⇒ 这就是为什么 **图化（`DSV41_VERIFY_GRAPH=1`）是针对 MTP 的头号杠杆** ✓，也是为什么
  **`moe_rows` 里逐行发射（"one activation row per launch" + `for r in 0..m` 的 gate GEMV）必须合并** ✓。

**MTP 专项杠杆清单（都在 verify 路上，按预期收益）**：
| # | 杠杆 | env | 针对 verify 的什么 |
|---|---|---|---|
| 1 | **verify 图化** | `DSV41_VERIFY_GRAPH=1` | ~7000 次发射的提交时间（20.3ms → ~2.8ms） |
| 2 | **m 行合并（MROWS 家族）** | `DSV41_GATE_MROWS=1 DSV41_GATE_MROWS_ROUTE=1 DSV41_ATTN_MROWS=1 DSV41_COMPRESSOR_PROJ_MROWS=1 DSV41_ENGRAM_{PROJ,GATHER}_MROWS=1` | `moe_rows` 的逐行发射 |
| 3 | **同格式 grouped MoE** | `DSV41_EXPERT_GROUPED=1 DSV41_EXPERT_TCGEN05_E4M3=1` | 6 行共享专家权重的重复读（shared 10.4ms + routed 8.3ms） |
| 4 | draft 图化 | `DSV41_DRAFT_GRAPH=1` | 草稿段的发射 |
| 5 | 整步图 | `DSV41_GRAPH_STEP=1`（默认 ON，但 runner 的 `GRAPH_OFF` 会关掉 ⇒ 必须显式覆盖） | decode 侧单行步 |

## 10. 「数字跳跃」的层级判决（`skip-provenance`）与下一步

- 观测来源锁定：`/v1/chat/completions` **无 `stream` 字段** ⇒ 走**非流式**分支 ⇒ 文本是
  **`decode(all_ids)` 一次性解码**的产物（`api.rs:239-245`）⇒ **不是**增量反分词吞字 ✗。
- `temperature=0` **是纯 argmax，全仓库无采样代码**（`temperature` 在 HTTP 层只被反序列化、从不读取）✓。
- **该观测根本没走 spec**（未设 `DSV41_SPEC`）⇒ 所以跳跃**与 spec emit 无关** ✓。
- ⇒ **判决：最可能的层 = 模型前向数值 / argmax** ⇒ 即在**我那个 eager 配置**下，本该"永远对"的路也跳了 ✗。
- ⇒ 两个可能：**(a) 我的配置与"old 路"不同**；(b) **当前树里 eager 也坏了（回归）**。
  而审计指出的**确切差别**是：**`arm_run.sh` 的 `GRAPH_OFF` 关掉了默认 ON 的 `GRAPH_STEP`，并顺带关掉 `ar_v5`** ✗
  ⇒ **正在跑的 P0（两图显式 ON，`ar_v5` 回正常）就是判定它的臂** ✓（P0 文本若 1..61 干净 ⇒ 跳跃是 `GRAPH_OFF` 造成的 ✓）。

**定位 verify 取值缺陷的原生工具**（用户明确只有 verify 路对 MTP 有意义）：
- `DSV41_DIFF_EAGER=1`：逐轮重放并报**第一个 mismatch 的 index 与绝对位置**（AGENTS 原文称"定位利器"）✓
- `DSV41_TOKTRACE=1`：非 spec 路径每步打 `[toktr] ds= pos= tok=`（**直接看 argmax 写下的 id**，零改动）✓
- 二者已封装进 `scripts/campaign/verify_value_probe.sh`（带 `DSV41_SPEC=1`）✓

## 11. 🎯 两个决定"一切数字"的武装门（2026-09-14 深夜四）

**实测证据**：生产式配置（BS 臂 OFF + `DSV41_SPEC=1` + 两图 ON + `AR_V5=0`，但**没**带 `DSV41_DSPARK`）
跑出 `step pos=22: 50.55ms (19.8 tok/s)` ⇒ **换算 = 1.0 token/步** ✗ ⇒ 即 **draft/verify 未真正武装**，
每步仍付 verify 的代价却只吐 1 个 token —— 这是**最坏形态** ✗。

**代码依据**（`config.rs:464-472`）：
```rust
pub fn dspark_armed(&self) -> bool {
    if !self.dspark_enabled() { return false; }
    *ARMED.get_or_init(|| std::env::var("DSV41_DSPARK").map(|v| v != "0").unwrap_or(false))
}
```
⇒ **真正的 spec 武装 = `DSV41_SPEC=1` **+** `DSV41_DSPARK=1`** ✓（AGENTS 的表里两者并列 ✓）。
`serve.rs:473` 那句 `[dspark] shadow mode armed … all effects rolled back` 是**过时措辞** ✗
（它只在 `dspark_armed()` 为真时打印 ⇒ 它是**武装成功**的标志，不是"回滚"✗）。

⇒ **所以任何用于 MTP/tok-s 判定的臂，环境必须是**：
```bash
DSV41_SPEC=1 DSV41_DSPARK=1 \
DSV41_VERIFY_GRAPH=1 DSV41_GRAPH_STEP=1 DSV41_AR_V5=0 \
DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0
```
（前两者决定 spec 是否真跑；后三者决定图与 AR 形态；最后两者是那条"永远对"的正确路径。）
**判定口径**：`tok/步 = 打印的 tok/s × step_ms / 1000` 必须 ≈ `mean-k + 1`（≈3.2）；若 ≈1.0 ⇒ **没武装** ✗。

## 12. 🔴 当前真正的拦路缺陷：`44 / 77 / 1010` 的**周期-3 数字重复**（用户判断被证实）

**原文（`~/armrun_VDIFF.txt` 的 OUT 行，`DSV41_DIFF_EAGER=1 DSV41_TOKTRACE=1` 那次）**：
```
1 2 3 44 5 6 77 8 9 1010        ← 4→"44"、7→"77"、10→"1010"
```
- **重复的编号是 4、7、10 ⇒ 步长 3 ⇒ 周期-3 模式** ✓（不是随机错字 ✗）。
- **同一次运行里 `[diff] first_mismatch=none` 全场** ⇒ **m 行 verify 与单行 eager 逐 token 一致** ✓
  ⇒ 也就是说 **两条路都错、且错得一样** ✗ ⇒ **缺陷在"两条路共用的那段前向"**，不在 spec/verify 取值 ✓。
- 因此 `sids_writeback()` 注释里说的 "the blocker is verify's values" **不是当前这个现象的原因** ✗（verify 与 eager 一致 ✓）。
- AGENTS.md 记录的参照是 **"数字任务…EAGER 对照完美 1..100"** ✓ ⇒ 所以这是**树内/eager 路的回归或某个 flag 造成的** ✓。

**判据与下一步（已备脚本）**：
1. `~/eager_min.sh`（**用户建议的最小 eager 臂**：`SPEC=0 DSPARK=0`、BS 臂 OFF、无图、把 COMMON 压到最小）
   ⇒ 文本**完美** ⇒ 是某个 flag ✗（逐项二分）；文本**仍重复** ⇒ 共用前向在树内坏了 ✗。
2. `~/run_to_400.sh`（`EXP_ORACLE` 臂先跑）⇒ BS 臂 OFF 下 gate/up dump vs 已与官方逐位相同的 `gu_numpy_ref.py`
   ⇒ 一致 ⇒ 缺陷在 attention/head；不一致 ⇒ SIMT MoE 路也有份。
3. 拿到 `cat -A ~/num100_last.txt` 的**字节级**文本（`num100.sh` 已加落盘 ✓）⇒ 区分"模型真重复" vs "反分词/拼接"。

## 13. 「成串丢号」的新假设：**位置/记账比实际吐出的 token 走得快**（2026-09-14 深夜五）

**字节级事实**（`~/num100_last.txt`，纯单行 eager、`DSV41_SPEC=0`、BS 臂 OFF）：
`1..51, 62..66, 69..77, 88..95, 100` ⇒ 缺失 `[52,61]`(10) `[67,68]`(2) `[78,87]`(10) `[96,99]`(4)。
- 逐行检查：**没有**"一行两个数字" ⇒ 不是合并造成的；整数 token 数 = 74 = 行数 ⇒ 确实是**丢了 26 个数** ✓。
- `[diff] first_mismatch=none`（m 行 verify 与单行 eager 逐 token 一致）⇒ **不在 spec 层** ✓。

**假设**：模型不是"丢了上下文"，而是**"它以为已经写过 52..61"** ⇒ 即**位置计数 / KV 记账推进得比实际吐出的 token 多** ✗ ⇒
表现正是"**跳过一段再对上**" ✓（若是普通的数值翻车，应当出现**孤立**错号，而不是成串跳过 ✓）。

**为什么这个假设可判**（两条互斥的观测，一次臂即可区分）：
1. **`DSV41_TOKTRACE=1`**：非 spec 路径每步打印 `[toktr] ds= pos= tok=`（读的是设备上 argmax 真正写下的 id）
   ⇒ 若**id 流里就少了 52..61** ⇒ 模型侧（位置/记账）✗；若 **id 流里有** ⇒ 反分词/拼接侧 ✗。
2. **`pos` 的推进序列**：若某几步 `pos` 一次 +11（而只吐 1 个 token）⇒ 直接证实记账超前 ✓。
3. **对照**：官方 `ref_inference` 在同样输入下的位置推进（`model.py` 的 `pos_ctr` 语义）是否每 token +1。

**先前的相关否决/存疑**（勿重复）：
- A2（`DSV41_WINDOW_KV_QUANT=1`）**实测不可用**：打开后服务**不再应答**（请求全部 `http=000`）✗ ⇒ 精度五门尚未转正，不能再当实验门用。
- `gateup.f32` 的 oracle 对比**不成立**：eager 路的 dump 点在 **swiglu 之后**（`chain_dev.rs` 的 swiglu 在 24078、dump 在 24244）⇒ 拿它比 oracle 的 **raw gate|up** 必然 corr≈0 ✗。
- `moe_out.f32` 的整块对比**被 oracle 自己警告**：我们的 `moe_out_r` 是**本 rank 的 K-部分和**且**含 shared expert**，官方值需要 8 卡部分和 ⇒ 不是同口径 ✗。

## 14. 🎯 官方 API 作为"行为参照"（用户提供，2026-09-14 深夜六）——**缺口是我们的 bug**

```bash
curl -sS --noproxy '*' https://mint-alpha.macaron.xin/v1/chat/completions \
  -H 'Authorization: Bearer <KEY>' -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-flash","messages":[{"role":"user","content":"请从1数到100，每个数字单独一行"}],
       "max_tokens":400,"temperature":0,"stream":false}'
```
**实测（同一条 prompt）**：
| 对象 | 结果 |
|---|---|
| **官方 deepseek-flash** | `numbers=100  gaps=[]  1..100 齐全` ✓（`usage: completion=304, reasoning=104`） |
| 我们（最小 eager） | `numbers=75   gaps=[(51,60),(66,69),(73,84),(87,89),(95,100)]` ✗ |

⇒ **⇒ 官方零缺口 ⇒ 缺口确定是*我们*的缺陷** ✗（不是"模型本来就会跳" ✗）。
⇒ 现在有**两个官方参照**：**CPU oracle**（`gu_numpy_ref.py` / `moe_block_ref.py`，判定**数值**是否与官方逐位一致）
＋ **官方 API**（判定**行为**：文本是否重复/乱码/丢号）✓。两者互不替代。

**已确认的缺陷特征**：①最小 eager（SPEC/DSPARK/E4M3/BF16/ILV/两图全关 + BS 臂 OFF）**照样丢号** ⇒ 不是 flag ✗；
②**两次运行缺口不同**（`[52,61]…` vs `[51,60]…`）⇒ **非确定 ⇒ 竞态** ✗；③`pos` 每步 +1 ✓ ⇒ 不是位置记账 ✗。
⇒ 按下一条（§15）用项目自己的二分纪律定位引入它的 commit。

## 15. 环翻卷边界（window=128）与首缺口的定量核对

`kv-index-semantics` 的硬边界：**KV 环 window = 128**，`oldest = start_pos % window + 1`
（`glue.cu:1181-1196`）⇒ `start_pos` 从 127 到 128 时 `oldest` 由 `128` 跳到 `1` ✓（**翻卷点**）。

**用真实 prompt 长度核对**（注意：serve 的 chat template 会加前缀，**官方 API 报 `prompt_tokens=35`**，
而不是裸 prompt 的 11 ✓）：
- 若 prompt = 35、每数字 ~2 token ⇒ 翻卷（绝对 `pos=128`）发生在**生成第 (128−35)/2 ≈ 46 个数字**附近；
- 实测**首个缺口 = 数字 51** ✗ ⇒ 相差约 5 个数字（约 10 token）⇒ **同一量级、方向一致，但不精确** ✓
  ⇒ 所以这条边界**很可能但对不上位**，必须由**实测**裁决（正在跑：`N=30/40` 期望干净、`N=80` 期望缺口 ✓）。
- 我此前的估算（"pos=128 恰好落在第 52 个"）用的是 prompt=11 ✗ ⇒ **已改正为 prompt≈35** ✓。

**判据**：`~/numN.sh`（参数化 N ✓，同一 runner 与原始文本落盘 ✓）——
`N=30/40`（pos_max ≈ 35+80=115 < 128）**干净** ⇒ 边界坐实 ✓；**照样有缺口** ⇒ 边界不是病因 ✗，
改走确定性二分 `~/gap_bisect.sh`（已备，单次运行即可判 ✓）。

## 16. ⚠️ 「陈旧产物」类陷阱——第三次发作，且它可能**颠倒**了整条回归结论

**这次的形态**：被停掉的 bisect 只做 `git checkout <commit> -- kernels crates`（**部分检出** ✗）。
把它恢复成 `origin/main` 之后，`~/ensure_built.sh` 依据自己的 **`.cu` 内容哈希戳**判断"内核源码未变 ⇒
跳过 `build.sh`" ✓，但 **`.so` 仍是那个中间提交编出来的** ✗ ⇒ `cargo build` 直接失败：
`error: failed to run custom build command for ferrite-kernel`（`CARGO_RC=101`）——
**`build.rs` 门禁发现不同源并拒绝** ✓（门禁是对的；`ensure_built.sh` 的判断是错的 ✗）。

**前两次同族**：① 我误加 `touch` 强制全量重编；② `build.sh` 曾静默跳过变了的 shim；③ 中断构建留下
"只重编了 `.so`、cargo 没跑" ⇒ 引擎 `kernel build-id mismatch — REFUSING TO START`。

**⇒ 关键含义（可能颠倒结论）**：如果**今天那几次"对不上金标准"的臂**用到了**不同源**的产物 ✗，
那么"gate|up 回归"就**不是代码回归**，而是**测量被污染** ✗✗ —— 用户问的"之前正常是不是幻觉"，
答案很可能是**反过来**：**今天显得"坏"才是幻觉** ✓。

**⇒ 根治（三条，缺一不可）**：
1. `ensure_built.sh` 在**跳过** `build.sh` 之前，必须跑 `check_artifacts.sh`（**三源 build-id 比对**：
   `.so` / 二进制 / `kernels/cuda/.build_id`）⇒ 不一致就**强制** `build.sh`（而不是只信自己的戳 ✗）。
2. 任何 `git checkout <rev> -- kernels crates` 之后**强制** `build.sh`（bisect 脚本里做 ✓）。
3. `build.sh` 判断单个 TU 是否需重编时，一律用**内容哈希**，不用 mtime。

**⇒ 判据（廉价）**：故意把 `.so` 换成旧的 ⇒ `ensure_built.sh` 必须**响亮失败并重编**，而不是报 "unchanged, skipping"。

## 17. 🎯 两簇证据：金标准是"孤例"——支持用户的"幻觉"判断

把 `/tmp` 里**全部** 3840-float 的 `gate|up` 类产物两两比对（矩阵，见 §17 脚本），得到**两个簇**：

| 簇 | 成员（时间） | 含义 |
|---|---|---|
| **A（= 官方 oracle）** | `gu_in_GD4_OLD`(**17:09 金标准**)、`gu_bs_new`(17:14)、`gu_numpy_old`(17:10)、`gu_numpy_GD4OLD`(18:01)、`gu_bs_row0`(17:17)、`gu_km_range`(17:22)、`oracle_chk/oracle.f32`(**今天 18:31 的 oracle 重算**) | **oracle 对这批输入稳定 ✓**；17:09 时**旧路输出 == oracle** ✓ |
| **B** | `gu_old.f32`(**16:45**)、`oracle_chk/eager/gateup.f32`(**今天 18:31 我们自己的 dump**) | **我们旧路自 16:45 起输出未变** ✓ |

**⇒ A ≠ B（corr −0.009）**，而**金标准是 17:09 之前唯一落在 A 簇的产物** ✗ ⇒
**单调回归解释不了"比它早的(16:45)与比它晚的(今天)都属于 B"** ✗ ⇒ 更像是 **17:09 那次用了一个"瞬时不同"的树/二进制**（当时战役正在频繁改文件 ⇒ 脏树 ✓）⇒
**⇒ 用户"之前正常是不是幻觉"的判断很可能成立** ✓。

**金标准自身的 ENV（其臂日志逐字）**：
```
[GD4_OLD] ENV: DSV41_GATEUP_DUMP=/tmp/gu_in_GD4_OLD DSV41_MOE_BS_SFDUMP=/tmp/sfd_GD4_OLD \
            DSV41_MOE_TILELANG_BS=0 DSV41_MOE_BS_HANDWRITTEN=0
[GD4_OLD] OUT: '1\n2\n3\n4\n5\n6\n7\n8\n9\n10'   ← 用的是 arm_run.sh 的"数到10"短 prompt
```
⇒ 配置 = `arm_run.sh` 的 COMMON 默认 + BS 臂 OFF ✓（`arm_run.sh` mtime 15:56 < 17:09 ⇒ **无漂移** ✓）。
（`batch4_fix.sh` 开头会重建双产物 ✓ 并打印 `.so` 的 md5 ✓ —— 但那次批次的**标准输出没有落盘** ✗，
所以无法直接比对二进制；只能靠**干净的端点复现**来裁决 ✓。）

**⇒ 裁决脚本 `~/golden_endpoint.sh`**（已入库）：
① HEAD 上清树 + **同源重编** + 用金标准原样配置跑臂 + 与金标准比对（应落 B 簇）；
② 检出 `ac69b054`（金标准 mtime 17:09:22 之前的最新提交，17:07:29）⇒ **强制重编**（部分检出后戳不可信 ✗）⇒ 同样比对。
**两者都不落 A 簇 ⇒ 金标准不可复现 ⇒ 无代码回归（幻觉成立）** ✓；**若 `ac69b054` 落 A 簇 ⇒ 回归真实** ⇒ 再按"只碰过 4 个非 BS 文件的 9 个提交"收窄二分 ✓。

## 18. 🎯 MMA（tcgen05）为什么"没跑"——三条硬事实与正路（用户红线："必须 mma"）

**实测**：`DSV41_EXPERT_TCGEN05_E4M3=1 DSV41_EXPERT_GROUPED=1 DSV41_MOE_BATCH=1 DSV41_EXPERT_ILV=0`
下单步仍是 **~52ms**（与 SIMT 完全一致）。臂自己的警告逐字解释了原因（**不是 kernel 坏，是没上场** ✗）：

1. **dense 版 tc5::e4x 启动器要求 `m % 128 == 0`、`dim % 64 == 0`、`2*inter % 64 == 0`** ✗
   —— verify 是 **m=6** ⇒ **算术上永远不被接受** ✓（与 kernel 正确性无关）。
2. **为不规则行数设计的 GROUPED masked MMA 臂"没有拿到 stage"** ✗，原因写得很清楚：
   > `DSV41_EXPERT_GROUPED` is set, but the routed MoE still runs the proven per-(row, slot) launches:
   > gate/up is on the **FUSED swiglu shape** (`DSV41_GATEUP_FUSE` + fp4 mode 2 + `dim % 512 == 0`)
   ⇒ **被 fused 形状抢先** ✓；而 `:17506/:17551` 的注释说明 **e4x epilogue 只 CLAMP、从不 fuse** ⇒
   两者**互斥** ⇒ **关掉 `DSV41_GATEUP_FUSE`（默认 ON）才能让 grouped MMA 上场** ✓。
3. **dense 单行臂另有 err 716 故障**（`docs/agent/tcgen05-716-e4m3-confound-verdict.md`）⇒ 单行路不可靠 ✓。

**⇒ 正路（已封装成 `~/mma_take_over.sh`）**：`DSV41_GATEUP_FUSE=0` + grouped e4m3 武装，
且**必须先确认 `declined / did not take the stage` 警告消失**，再相信任何计时 ✓。
（`DSV41_GATEUP_FUSE` 默认 ON、`unwrap_or(true)`，`chain_dev.rs:4510-4512`；使用点 `18817 / 23943 / 24209 / dspark 2931`。）

