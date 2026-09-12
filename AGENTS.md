# ferrite — Agent Working Guide

Rust-native inference engine for **GLM-5.3-Flash**（hybrid GatedDeltaNet linear attention + DSA sparse attention + MoE）与 **DeepSeek-V4.1-Flash**（DSV41），single-node TP over CUDA graphs。
Read `README.md` for the design contract; this file is the operational guide: build/test loop, every runtime flag, demo configs, and the profiling workflow that actually works on this hardware.

## Repo layout (hot paths)

```
crates/ferrite-exec/src/tp.rs        TP cluster + mega-graph chain + MTP step (mtp_step, mega_chain_dev)
crates/ferrite-kernel/src/cuda.rs    CudaBackend: FFI, graphs, GDN/DSA/MoE device chains, MtpState
crates/ferrite-serve/src/main.rs     GLM binary: load checkpoint → prefill → decode loop (one-shot + --serve)
crates/ferrite-http/src/             THE shared serve stack: api.rs (axum routes + SSE), driver.rs, engine.rs, single_flight.rs, serve.rs, tokenizer.rs
crates/ferrite-dsv41/src/bin/dsv41-run.rs  DSV41 runner（one-shot + --serve = TP rank pool behind StepEngine）
crates/ferrite-models/src/dsv41/     DSV41 的核心（chain_dev.rs ~9800 行 / dspark_dev.rs / serve 接线）
kernels/cuda/*.cu                    ALL device kernels (sm_103a)；build.sh → libferrite_kernels.so
docs/agent/*.md                      战役计划与知识文档（详细内容写这里，不放本文件）
```

## Build / deploy / test loop

Local (no GPU needed): `cargo check --workspace`（硬门禁）。Remote: `ssh -o BatchMode=yes ubuntu@43.202.208.136`，repo `~/ferrite`，模型 `/opt/dlami/nvme/models/{GLM-5.3-Flash,DeepSeek-V4.1-Flash}`，GPU 0-7。

```bash
# local → remote（远端 origin 只跟 main；push 后远端必须双产物重编）:
git push origin main
ssh ubuntu@43.202.208.136 'cd ~/ferrite && git fetch -q origin && git reset -q --hard origin/main && \
  cd kernels/cuda && bash build.sh 103a && cd ~/ferrite && source ~/.cargo/env && cargo build --release'
```

`bash build.sh 103a` = sm_103a (B300)（必带；README 的 100a 已过期）。
**双产物纪律**：`.cu` 变了必须先 `build.sh` 再 `cargo build`（build.rs 门禁会拒绝不同源的组合）；md5 记 `.so` 指纹。

## 硬性禁令（用户明令）

1. **禁止 `git revert`**（退化一律 env-gate 默认关，代码保留）；禁止 rsync 同步代码（用 git）。
2. **每次切版本后双产物重编**（`.so` + 二进制同源）。
3. **禁止硬件复位类 API**；严禁重跑已验证的 baseline；禁止无意义的验证跑测。
4. **严禁 MTP/投机解码的旧禁令已解除**（2026-09-12 用户指令：dspark block-5 单并发 ≥400 tok/s）。
5. **严禁前台 sleep**（后台 watchdog/task_wait）；测试单轮制；跨版本比较同会话背靠背。
6. **AGENTS.md 保持简洁**（用户 2026-09-12：详细内容写 docs/agent/，本文件不放长篇会话记录）。

## 测试纪律

- 正确性红线（用户）：**不能重复、不能乱码**——两者同级。
- 每次改动**人眼看文本**（出师表逐字/数字任务数数）；`DSV41_DIFF_EAGER=1` 的 `[diff]` 行是定位利器（anchor 应每轮与 eager 一致；mismatch 的 index 直接指向 verify 的第几行）。
- 判定 kernel 改动用**数值等价**（隔离微基准/逐位比对），文本只作辅助。
- serve 收尾一律 `POST /shutdown`（不是 kill -INT）。

## GLM 环境flag（标准 MTP 运行）

```bash
NCCL_NVLS_ENABLE=0 \
FERRITE_MEGA=1 FERRITE_NCCL=1 FERRITE_P2P=1 FERRITE_WORKER_POOL=1 \
FERRITE_LAYER_DEV=1 FERRITE_GDN_DEV=1 FERRITE_MOE_DEV=1 FERRITE_DSA_DEV=1 FERRITE_HEAD_DEV=1 \
FERRITE_MTP=1 CUDA_VISIBLE_DEVICES=4,5,6,7 \
LD_LIBRARY_PATH=$HOME/ferrite/kernels/cuda ./target/release/ferrite-serve --backend cuda --tp 4 \
  --model-dir /opt/dlami/nvme/models/GLM-5.3-Flash --lib kernels/cuda/libferrite_kernels.so ...
```

`NCCL_NVLS_ENABLE=0` 在该节点必带（否则 NCCL 静默回退 host AR，~2.4x 慢）。其余 flag 见 `docs/agent/`（perf 系列）。

## DSV41 dspark（当前主战场）的关键 env

| env | 作用 | 默认 |
|---|---|---|
| `DSV41_SPEC=1` | 真 spec 模式（draft+verify+commit） | OFF |
| `DSV41_DSPARK=1` | armed（tap hook 等） | OFF |
| `DSV41_DSPARK_DEBUG=1` | 逐轮 `[dspark-dbg]` trace | OFF |
| `DSV41_DIFF_EAGER=1` | spec vs eager 逐位对照 probe | OFF |
| `DSV41_VERIFY_HEAD_FOLD=0` | verify 的 head 走 per-row v1（folded 有 K 序差） | **OFF（=per-row）** |
| `DSV41_VERIFY_HEAD_SLICED=0` | verify 的 head 关词表切分（回到每行读全量 1262MB） | **ON（切分 + 1 个 v5 round）** |
| `DSV41_SIDS_WRITEBACK=1` | spec commit 后回写 emitted.last() 到 s.ids | OFF（verify 值修好后开） |
| `DSV41_SWALLOW_STEP=1` | 吞主链步（6 行块） | OFF |
| `DSV41_SEED_ALIGN=1` | 判词路线 A（seed↔tap 对齐） | OFF |
| `DSV41_VERIFY_GRAPH=1` | verify 的 CUDA 图化 | OFF |
| `DSV41_TIMING=1` | `[dspark] steps=` 计时行 | OFF |

## 诊断模式（下次直接用）

- **数字任务数数**（"请从 1 数到 100，每个数字单独一行"）——**对重复/错位最灵敏的探针**（EAGER 对照完美 1..100）。
- **出师表背诵**（长上下文 + 背诵）——对累积型污染灵敏（EAGER 对照 LEN 146）。
- **`DSV41_DIFF_EAGER=1`**——每轮重放 emitted.len() 个单行 forward（同前缀 KV），报第一个 mismatch 的 index 与绝对位置。
- serve 卡住/日志 mtime 停滞 = 挂了（查 `stat -c %y` + pgrep，别等）。
- 加载错防线（三道 runtime + 三道编译期 + git hooks）见 `docs/agent/` 的加载防线文档。

## 当前状态与下一步（2026-09-12，详见 docs/agent/）

**正确性（用户红线：不能重复不能乱码）**：
- 8 个根因全部修复并验证（详见 `docs/agent/dspark-correctness-chain.md`）
- 最后一个根因：**routed expert 激活量化 e2m1 vs 官方 e4m3**（4×噪声差、44 层累积=opa 的来源）——e2m1×2 双趟修复（`DSV41_EXPERT_ACT_E4M3=1`）消除 opa ✓；布局契约 bug（fuse 绑 pitch）已修；ILV+E4M3 冲突有 Rust 守卫。
- spec 步跨步不变量断言层（`DSV41_INV_CHECK=1`，8 条优先项）。
- diff probe（`DSV41_DIFF_EAGER=1`）：mismatch 从首轮即错→18→1（o-rope 对齐后）。

**性能（400 tok/s 口径：accept 3 ⇒ 步时 ≤7.5ms）**：
- 基线 48ms/步（verify 37.31 + draft 4.37 + 主链 6.15 + commit 0.33）@ accept 0.83
- **accept 是真正的乘数**（0.83→3 = 3.6× 缺口 vs 性能 1.6× 缺口）
- 已实现的 gate（全部默认 OFF，等 A/B）：SH_EXP_MROWS(−8.3ms)、VERIFY_GRAPH(−15ms submit)、DRAFT_P3A(−37 launch)、MARKOV_SLICED(−1ms draft)、VERIFY_ROPE_MROWS(−0.6ms)、SWALLOW_STEP(−6.15ms 主链)、tcgen05 mxf4(−6.8ms routed experts)
- **最终判定**（`docs/agent/final-400-config.md`）：全部落地后步时 ~12ms，accept 3 → 250 tok/s；**7.5ms 触架构地板**（weight-stationary 下限 7.4-9.2ms）——400 需要 accept ≥4 或 verify ≤5.5ms（需 tcgen05 + 融合全部兑现）
- verify 的 ms 分解（`docs/agent/verify-ms-breakdown.md`）：50% launch submit + 49% in-kernel 残差 + 5% 带宽——先 A（合并/mrows）后 B（图化）
- routed experts 残差分析（`docs/agent/routed-expert-residual.md`）：377GB/s 的成因 = 每 fp4 值 2.5 条 L1TEX 解码指令 → tcgen05 是唯一根治路径（−6.8ms）

**测试纪律**：单一 GPU 测试驱动 = 主 agent（subagent 禁止 e2e）；禁止轮询远端状态（等通知）；禁止跑已知结果的测试。

**性能与知识文档**：`docs/agent/dspark-perf-400-plan.md`（400 的账本与融合清单）、`docs/agent/unified-engine-battle-plan.md`（战役计划）、`docs/agent/wave4-prefill-plan.md`（1M）、`docs/agent/wave5-multiseq-plan.md`（多并发）、`docs/agent/mtp-unification-analysis.md`（MTP 统一）、`docs/agent/dspark-swallow-step-diff.md`（吞主链步）。会话级的详细记录归 `/tmp/AGENTS_session_archive.md`（本地）与 git 历史。
