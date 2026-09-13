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

| 臂 | env | step p50 | @acc2.2 | 文本 |
|---|---|---|---|---|
| 原状 | （BS 臂 ON ✗） | 32.5ms | ~100 | **全乱码**（BS 臂毁模型） |
| **P1** | BS 臂 OFF + `DSV41_VERIFY_GRAPH=1` | **10.19ms** | **318 tok/s** | `1..51` 后跳到 `62,63,64,65,66,69…` |
| **P2** | P1 + 精度中性折叠（GATE_MROWS[_ROUTE]/ATTN/COMPRESSOR/ENGRAM/P3LITE） | **10.17ms** | **318.6** | **与 P1 逐字节相同** ⇒ 折叠族**精度中性** ✓ |

⇒ **图化确认真实生效**（32.5→10.2ms = 3.2×，且 p10/p90 只有 ±0.1ms）。**P2 与 P1 文本逐字节一致** ⇒ 那批折叠门不改数值 ✓（可作为默认候选）。

**"数字跳跃"的假说（待 `STEP_NOGRAPH` 对照确认）**：AGENTS.md 的 DSV41 门表记载 **spec 路径有已知缺口**
（`DSV41_SIDS_WRITEBACK=1` 的说明原文是"**verify 值修好后开**"）⇒ 说明**verify 的取值**本身就带缺陷，
而 spec 的 emit 直接取自 verify ⇒ 跳跃是**已知的 spec/verify 缺口**的表现，**未必是图化造成的**（图化只改发射方式）。
⇒ **下一批单变量臂**（每条只答一个新问题）：
1. `DSV41_SIDS_WRITEBACK=1`（spec commit 后回写 `emitted.last()` 到 `s.ids`）
2. `DSV41_SEED_ALIGN=1`（判词路线 A：seed↔tap 对齐）
3. `DSV41_SWALLOW_STEP=1`（吞主链步）
4. `DSV41_VERIFY_HEAD_FOLD` / `_SLICED` 的两种组合
→ 目标是让 **1..100 前 61 行 = 1..61**（红线），同时保住 318 tok/s。
