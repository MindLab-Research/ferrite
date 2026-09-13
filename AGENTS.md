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

## 当前状态与下一步（2026-09-12 session 深夜收官——SWALLOW 解锁 + A1a 教训）

**Session 成果（783+ commits，107 知识文件）**：
- **范式转移**：所有"损坏"判定是模型行为（EAGER 对照确认）。验证协议 v2：计数只对前 61 行有效；退化与 EAGER 一致 = 干净。
- **lazy 干净栈：91.1 tok/s（+15.6%）**：R2(+10.2%) + MARKOV(+3.2%) + VERIFY_FORK(+1.5%) + RING_WIN(+0.3%)
- **🎉 SWALLOW 完全解锁！**（11 次修复：OOB 根因（staging 被越界清零→canary 抓到！）→ OOB 修复（guard band + bounds check）→ engram slot 修复（147456 ≥ payload）→ **300 token 正常生成 + 出师表红线通过（零拉丁 ✓）+ 全 gate 58.3 tok/s**）
- **AR Step 2 (A1a) 教训**：+665 行 store fold 在两条路径破坏数值（lazy 90.9/输出退化 + SWALLOW 7.1/8× 退化）——**AR_STORE_FUSE 永久 OFF**
- **400 路线（⛔ 终局订正 2026-09-13，旧口径作废）**：**正确模型 = MTP verify 摊薄**（`docs/agent/mtp-verify-amortization-model.md`，必读）：单并发 decode 是 memory-bound，**verify(m 行) 应 ≈ eager(1 行)+ε**（权重读共享）⇒ **400 = step ~8ms + acc 2-3**（375-500 tok/s）。实测 verify=28.17ms = eager 6.33ms 的 **4.45× = 实现未摊薄的病**（不是物理极限）——主战场 = **逐 kernel 对比 eager(1) vs verify(6)，找出所有 ~6× 未摊薄项**（MoE per-row 路由展开 / per-row kernel 未进 m=6 块 / 图 launch 结构 / attention per-row 计算）并批量化修复。~~"L4/L5 25-40 人日唯一路径"~~ 作废重估。A0 判决（AR 无肉，nsys 78µs 是自旋假象）保留有效；**任何提案引用 nsys 的 AR µs 数必须先过 A0 探针复测**
- **tcgen05 根因已定谳（第 5 轮判定实验）**（`docs/agent/tcgen05-rank7-verdict.md §10`）：**8/8 ranks 同文本 cuda error 716**（"只有 rank 7"= serve.rs 上报竞态，正式作废）+ ALIGN_AUDIT 唯一 violation = **w2 SF 行 pitch 10B**（rank 对称）⇒ 根修 = SF 行 stride 与逻辑 k/32 解耦（行间 padding + 内核索引参数化，inter/world 需被 512 整除），属 L4-3/L4-4 收尾

**400 的诚实判定**：lazy 上限 ~145；SWALLOW 需要全部优化兑现（mrows + hc/B6 + tcgen05 + AR 重设计 + L4/L5）；60% 兑现 → ~300 tok/s；**先钉死 S0 步时（28ms）是 G2 的全部意义**。

**验证纪律（v2）**：EAGER 对照必须；前 61 行判据；三探针（数字+拉丁+k_acc）缺一不可；**gate ON vs OFF 逐字节一致**（A1a 的教训）。

**下一 session 前 30 分钟**：读 handover 的 SWALLOW 部分 → mrows Phase A 状态检查 → 起一臂测试 → 5 个决策（A1a 修或弃 / hc arm / 路由 / tcgen05 / 400 口径）。
