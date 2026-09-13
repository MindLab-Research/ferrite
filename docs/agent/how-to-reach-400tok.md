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

## 3. 若还差（按收益排序的后备杠杆）

1. **head/logits 的词表读**（`DSV41_VERIFY_HEAD_SLICED` 默认 ON；关掉会回到每行读全量 **1262MB**）
   ⇒ 见 `head-term-audit` 的字节/launch 表；这是 verify 里可能的**最后一条大鲸鱼** ✓
2. **draft 段融合**：`DSV41_DRAFT_P3LITE_{SEED,KV,ATTN}=1`（单变量可切）
3. **AR 形态**：`DSV41_AR_V5` / `FERRITE_P2P`（注意 nsys 轮的死锁规避组合与普通轮不同）
4. `DSV41_MOE_DOWN_BS=1`（down 臂 blockscaled，已接线、默认 OFF；**用前必须先在不捕获的调用里完成 lazy INIT**，
   否则捕获内 `cudaMalloc` 会让 `cudaStreamEndCapture` 失败）

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
