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
| `DSV41_ACC_HISTOGRAM=1` | R0：逐 step `[acc-hist]` 行 + 收尾 `[acc-hist-summary]`（k_acc 直方图 + p1/p_j 分解；`docs/agent/r0-r1-accept-diagnosis-manual.md`） | OFF |
| `DSV41_ORACLE_TAP=1` | R1：oracle tap 对照（draft 改喂主链同位 hidden，比 `drafts[0]` vs `rows[0]`，收尾报 rate） | OFF |

**2026-09-13 摊薄修复批新增 gate**（详见 `docs/agent/verify-amortization-lesion-audit.md` §9 + 各设计文档；全部默认 OFF、逐位论证、双门禁验证中）：

| gate | 作用 | 前置 |
|---|---|---|
| `DSV41_ATTN_MROWS=1` | sparse attn m=6 批量（TP8 row_pitch 已修，world!=1 decline 已除） | 无（新符号 `_rp` 自动回落） |
| `DSV41_COMPRESSOR_PROJ_MROWS=1` | compressor 投影 ONE launch（gemv_f32_v2_mrows） | `DSV41_GEMV_F32_V2` 默认路径 |
| `DSV41_ENGRAM_PROJ_MROWS=1` | engram 投影 mrows 折行 | 无 |
| `DSV41_ENGRAM_GATHER_MROWS=1` | engram gather rows 批量（id_stride） | 无 |
| `DSV41_MROWS_MPAR=N/auto` | gemm_fp8_mrows 的 warp 级 M 并行（rpb=每块输出行数） | 无（OFF=逐字节旧程序） |
| `DSV41_EXPERT_GROUPED_DOWN=1` | MoE down 的 expert 并集去重 | 需 `DSV41_EXPERT_GROUPED=1` 同臂 |
| `DSV41_DRAFT_P3LITE_{SEED,KV,ATTN}=1` | draft 段融合三开关（单变量可切） | 无 |
| `DSV41_SF_STRIDE_PAD=0` | w2 SF 根修逃生门（默认 ON） | — |
| `DSV41_AR_PROBE=1` + site 分流 | AR device 探针（attn/moe 分桶符号 `ferrite_p2p_ar_v5_attn/_moe`） | 无 |

## 性能基准与"击败 sglang"（2026-09-14 用户指令，必读）

**参考**：`https://www.sglang.io/blog/deepseek-v4.1-flash-kernel-optimization`
（同族模型 DSV4.1-Flash 的 kernel 优化，**35.2 → 873.6 tok/s**，4×GB300、BS=1、attention TP4 + MoE TP4、
random 4k/1k、**模拟 accept 5.5**）。**详情与 16 步阶梯见 `docs/agent/moe-bs-crash-investigation.md` §97/§98。**

**三条必须记住的**：
1. **可比性折算**：他们 4 卡、我们单机 8 卡 ⇒ 只比**每 GPU**。plain decode 他们 **50.8**、我们 **14.0**（落后 3.6×）；
   我们的 400 目标仅为其全优化态（218.4/GPU）的 **23%** ⇒ **400 是必经里程碑，不是终点**。
2. **他们的第一条教训与我们同源**："确认一个 GEMM 实际 dispatch 到哪个 kernel，通常比调 tile 更值钱"
   —— 他们只修 **FP8 量化块/scale 布局**就从 35→118（**3.3×**）。
3. **测量口径必须固定 accept**（他们都用模拟 5.5；我们当前实测 ~2.24）⇒ 用 `~/bench_protocol.sh`
   按同口径测（random 4k/1k、固定输出 1024、BS=1、greedy），并**同轮背靠背 A/B、一次一个变量**。

### 真正的病灶：verify 的 **4.45× 未摊薄**（裁决 2026-09-13，勿再犯）
- **正确模型**（`docs/agent/mtp-verify-amortization-model.md`）：单并发 decode 是 memory-bound，
  **verify(m 行) 应 ≈ eager(1 行) + ε**（权重读被多行共享）。
- **实测**：`verify = 28.17 ms` = `eager(1) = 6.33 ms` 的 **4.45×** ⇒ 这是**实现未摊薄的病** ✗，不是物理极限。
- **缺口分解**（据已归档分析）：MoE 的 **per-row 36-sweep FMA 恒等式**（≈6×、SIMT issue-bound）
  + **M-in-register GEMM**（≈3.8× 指令）+ **~54 次约束发射/层**。
- **修复载体**：MoE 那部分 ← **BS 臂（tcgen05 fp4 blockscaled，`DSV41_MOE_TILELANG_BS` + `DSV41_MOE_BS_HANDWRITTEN`）**；
  GEMM 那部分 ← MROWS 家族（已落地）；发射那部分 ← 融合/重叠门。
- **目标量级**：verify 压到 ~6.3 ms 量级 ⇒ 配合 draft 3.87 + commit 0.47 ⇒ step ≈ 10.6 ms
  ⇒ `tok/step 3.24 / 10.6 ms` ≈ **306 tok/s**；再往下压 step（或按 §106 的 tok/step 口径）才到 450 ✓。
- ⚠️ **禁止**用 `[dspark] steps=` 的 `verify=` 累积均值做性能判断（含 prefill 污染，会得到"verify > step"这种不可能的结论）。

## 当前回归状态（2026-09-14 深夜，读这条就够）

- **症状**：serve 能启动/加载/武装 BS 臂，但**首个请求时卡死** —— 日志刷 `[ar5-hang] … TIMEOUT -> PARK`
  且**整轮 0 个 `[dsv41] step pos`**；`curl` 拿不到响应（`http=000`）。**用户锚点："之前从来没卡过"** ⇒ 是回归。
- **已排除**（有证据）：**环境**（把 COMMON 恢复成改动前的形态、单臂对照 `S1` **照样卡** ✗）；
  代码合并里的**惰性项**（SEQ_ALIGN / cpasync / down 两个新 TU 全默认 OFF 或未接线）；
  **生产路径的 ABI 穿线**（`seq_align` 落位正确 ✓）；新 shim 的**加载期副作用**（无 ✓）。
- **两大嫌疑（均已处置）**：
  1. **`DSV41_MOE_BS_NUMCHECK` 探针**：它在 decode 路径做 **D2H + 流同步**，而多 rank **lockstep**
     ⇒ 一个 rank 同步、其余等 ⇒ 死锁。**时间线吻合**：早轮能跑出 48 step 是因为该探针**不可达**（`goto` 跳过 + M1 回归），
     我修好 M1/M2 之后它才激活 ⇒ 随之开始卡。**已改为 opt-in** ✓（从 COMMON 移除）。
  2. **有界等待**（唯一"总是生效"的语义改动）：**已门控回默认 OFF**，恢复项目历史上从未卡过的原无界等待 ✓。
- **配置保真**：`arm_run` 的 COMMON 现已与**三个最老备份逐字一致**（只少了那个探针）✓。
- **判据**：跑 `~/stage_probe.sh`（分阶段：起服务 → /health → 单请求 → 状态）或
  `~/staged_verify.sh probe`；**看到 `ar5-hang` 刷屏 + 0 step 就先怀疑回读类探针**，
  并看 `~/arm_summary.sh <arm>` 的 `counters:` 行（它会直接点出 HANG SIGNATURE）。
- **若仍卡** ⇒ 按 `~/bisect_probe.sh <commit>` 二分（候选：`7e54fd25` = 合并波之前的代码状态）。

## 回归定位纪律（2026-09-14 事故复盘，必守）

1. **"以前从来不这样"是最值钱的定位信息**：把搜索空间从"所有可能"缩到"**最近改了什么**"，
   再按"**是否总是生效**"一刀切开（默认 OFF 的改动是惰性的，先排除）。
2. **对照实验必须先做**：一次"剥离可疑 env 的单臂"几分钟就能**证伪整层**（本次据此排除了环境 ✗），
   远胜于在错误的层里反复猜 —— **先证伪整层，再进入下一层**。
3. **诊断探针一律 opt-in，绝不默认开启** ✗：`DSV41_MOE_BS_NUMCHECK` 这类探针会在**多 rank lockstep**
   的解码路径里做 **D2H 拷贝 + 流同步** ⇒ 一个 rank 卡同步、其余 rank 在 all-reduce 里等它 ⇒
   **整集群死锁**，症状是 **`[ar5-hang]` 刷屏 + 整轮 0 个 step**（看起来像"模型/内核卡住" ✗）。
   ⇒ 需要数值判据时，**显式传该 env 跑单臂诊断**；性能臂/正常臂**不得**带它。
   ⇒ 见到"0 个 step + ar5-hang 刷屏"，**先怀疑探针**（或作 D2H/同步的其它钩子），再怀疑模型/kernel。
4. **改动必须门控**：任何**总是生效**的语义改动都是回归的最高嫌疑（本次有界等待已在 §117 门控回默认 OFF，
   恢复原无界等待；代码保留不删）。
5. **回归定位要用可观测证据**（`ps` / 日志 mtime / `grep -c "step pos"` / `ar5-hang` 计数），
   **而不是再跑一整轮**；长轮（`verify_all.sh`）之前先用**分阶段**（`~/staged_verify.sh <stage>`）逐段确认。

## 合并纪律（2026-09-14 事故复盘，必守）

1. **合并后必须用项目自身的构建验收**（`bash build.sh 103a`），**不能只靠单文件编译**：
   本次 `glue_e2m1_encode` 重复定义让**完整构建失败**，而单文件检查报 RC=0（标志/上下文不同）。
2. **看到 `build failed`，那一轮的任何 e2e 结果都不能用**（二进制陈旧 ⇒ 跑的不是你以为的代码，
   与"测量偏置陷阱"同类）。
3. **构建命令必须传播真实退出码**：`bash build.sh 103a | tail -1 && …` 的退出码是 `tail` 的 **0**
   ⇒ 构建失败也会"看起来成功"（本次 T3 回合就是这么跑了一轮**陈旧二进制**）。
   正确写法：`set -o pipefail`（或 `; echo BUILD_RC=${PIPESTATUS[0]}`），并**显式检查**后再跑臂。
4. **多 subagent 汇入同一大文件时，新增辅助函数必须带唯一前缀**（如 `a2_`/`a3_`/`i3_`），
   否则极易同名重复定义——给 subagent 的 brief 里要写明这一条。
4. 合并 subagent 的 worktree 改动：**导出未提交 diff**（排除 docs）再 `git apply`；
   冲突时用 `patch -p1 -F3`；并**三项确认**（新符号计数 / `diff --stat` 与报告一致 / 我此前的修复仍在），
   且确认**无 `.rej`/`.orig` 残留**。`git merge <worktree-branch>` 无效（改动通常未提交）。

## 测量与工具纪律（用户裁决 2026-09-13，防遗忘）

1. **step time 必须真实测量且看 p50**：`[dsv41] step pos=` 行的 **p50**（中位数，最后 200 步——用户裁决 2026-09-13）；**禁止吞吐反推**（受 prefill/accept 污染）。**⚠️ `[dspark] steps=` 的 draft/verify/commit 分解是累积均值，不是 per-step 真实值——verify 可以比 step 还长（含 prefill 污染），不能用它做性能判断（用户裁决 2026-09-14：verify > step 一定是错的）**。**每次 nsys 分析完必须给用户当前的时间 breakdown 表**（各项 ms/步 + 占比 + 修复载体 + 修复后目标——用户裁决 2026-09-13）。
2. **NCU 只跑特定 kernel 的 micro bench**（tests_*.cu 二进制），**不能 e2e**；**必须 sudo + 绝对路径**（`sudo /usr/local/cuda-13.2/bin/ncu -c N -o <rep> -f <test_bin>`——无 sudo 时 report 静默不写，三次失败实证；sudo 后 5.4MB rep 秒级生成）。**subagent 编译纪律**：默认只做**单文件 compile-only**（`nvcc -c shim.cu -o /tmp/x.o`）；**禁止 subagent 跑完整 `build.sh 103a`**（8-TU 全量 ~3.5 分钟 + 多 subagent 同时编译争抢 CPU——完整产物重编是主 agent 的统一职责；唯一例外：接线验证可用 `/tmp` 隔离副本跑一次全链确认 SRCS/头路径）。**subagent GPU 纪律（用户裁决 2026-09-13，铁律）**：**禁止 subagent 在远端机器上做一切 GPU 操作**——包括 micro bench、tilelang/python 运行、AOT 生成（占 GPU）、任何 GPU 占用——**GPU 测量是主 agent 的专属职责**（多 subagent 的 bench 一起跑会争抢 GPU 互相污染数据）。subagent 的远端活动仅限：nvcc compile-only（CPU）+ 文件读写。GPU 需求一律写成"生成命令清单/验证手册"留给主 agent 执行。
3. **nsys 多跑**（per-kernel 时间唯一来源）。**成功配方（v6，实测验证 116MB rep）**：对常驻 serve 用 `nsys profile --trace=cuda --sample=none --duration=<秒> --kill=SIGTERM -o <rep>`（vLLM/SGLang 社区标准）——duration 到时 nsys 自动停采集并杀 serve → finalize → rep 落盘；脚本侧再 `POST /shutdown` 双保险 + 轮询 rep 文件出现（finalize 大 rep 要几分钟，116MB 正常）。**无 sudo 可行**（--trace=cuda --sample=none 免 perf 权限；wave1 与 v6 双重实证）。**禁用外部 SIGINT**（多进程架构下单播 INT 破坏 finalize，四连败教训）；清理进程用 `pkill -x nsys`/`pkill -x ferrite-serve`（`pkill -f nsys` 会匹配 ssh 自身命令行自杀）。**死锁规避必带**：`DSV41_AR_V5=0 DSV41_GRAPH_STEP=0` + `env -u FERRITE_P2P` + `NCCL_NVLS_ENABLE=0`；nsys 轮只看 kernel 相对倍数（AR 形态已变），吞吐数字必须来自非 nsys 轮。**提交纪律**：工作树有 peer subagent 运行时**禁止 `git add -A`**（半成品会被扫入 HEAD 导致远端编译失败——dsv41_kv_win_fetch 未定义事件实证）——只 add 自己确认过的文件。
4. **所有 e2e 必须 background 模式**（serve 启动的 ssh 会挂住前台）。
5. **MTP 性能模型（用户裁决，勿再犯）**：单并发 decode 是 memory-bound ⇒ **verify(m 行) ≈ eager(1 行)+ε**；**400 = step ~8ms + acc length 2-3**。任何"verify 4-5× eager 是结构性代价"的理论都是错的（详见 `docs/agent/mtp-verify-amortization-model.md` + `verify-amortization-lesion-audit.md`）。

## 诊断模式（下次直接用）

- **数字任务数数**（"请从 1 数到 100，每个数字单独一行"）——**对重复/错位最灵敏的探针**（EAGER 对照完美 1..100）。
- **出师表背诵**（长上下文 + 背诵）——对累积型污染灵敏（EAGER 对照 LEN 146）。
- **`DSV41_DIFF_EAGER=1`**——每轮重放 emitted.len() 个单行 forward（同前缀 KV），报第一个 mismatch 的 index 与绝对位置。
- serve 卡住/日志 mtime 停滞 = 挂了（查 `stat -c %y` + pgrep，别等）。
- 加载错防线（三道 runtime + 三道编译期 + git hooks）见 `docs/agent/` 的加载防线文档。

## 当前状态与下一步（2026-09-14 深夜——harness 九项拦截已全清；首次可信验证轮在跑；精度五门待逐项转正）

**详细记录一律在 `docs/agent/moe-bs-crash-investigation.md`（§10–§22），本节只放结论与指针。**

- **手写 kernel 的 6 个真 bug 已修**：idesc `b_sf_id` 漏 `<<4`、smem 布局与 descriptor 的 swizzle 声明不匹配、
  K-block 递进单位误判 8×、缺 `fence.proxy.async`、缺 `tcgen05.fence::before/after_thread_sync`（3 处）。
- **🏁 判定标准（用户裁决，§99/§106）**：SGLang 的 **acc 5.5 是模拟值**（其 server 配置钉死它来测 kernel 速度），
  **不作为我们要追的水平**；**我们在真实 acc ≈ 2.2 下达到 ~450 tok/s 即算击败 sglang**。
  ⚠️ **口径差 1**：他们的 "accept length" **含 bonus token**，我方 `mean-k` **不含** ⇒
  **同口径我方 `tok/step = mean-k + 1 = 3.24`**；由 `tok/s = tok/step ÷ step` ⇒ **450 tok/s 需要 step ≈ 7.2 ms**
  ⇒ **这是一道纯 step 削减题**（当前 spec step ≈ 32.5ms = draft 3.87 + verify 28.17 + commit 0.47）。
  真正的病灶见下（verify 的 **4.45× 未摊薄**）。
- **🧰 harness 自身曾拦路九次（§111 全清单，务必先读再动 harness）**：其中三处是**我改脚本时自己造**的
  （`sed "$d"` 双引号展开致死、`GRAPH_OFF` 多行拼接被拆断、**花括号污染 python 引号**导致 OUT 行从未生成）。
  ⇒ **铁律：改完 harness 必须用"真实最小用例"跑一遍，`bash -n` 不够**（查不出 `set -u` 未定义变量与语义破坏）。
  现 `arm_run.sh` 的 OUT 行**已用真实执行验证可解析** ✓。
- **🔬 首次可信验证轮在跑**：`~/arm_run.sh`（图门全关 + 焊好探针路径）与 `~/arm_run_fast.sh`（图 ON，性能专用）
  分工见 §42/§109；`~/verify_all.sh` 一条命令跑完"重编 → BS 臂文本判据 → 五道精度门 DBG → cp.async 门 → spec 两侧门 → 博客口径基准"。
- **🎯 fp4 语义定谳（§47，硬件实测 relerr=0）**：fp4 操作数是 **packed（2 元素/字节）但放在 16 B 容器里、
  硬件只读每槽前 8 B**（TMA dtype `16U4_ALIGN16B` = 16 个 4-bit 元素 = 8 B 数据/16 B 容器）⇒ 每行 footprint 128 B、
  每 stage 16384 B。写公式 `hw_pack_sw128()`（§47）；**描述符/递进/idesc 保持官方原值不变**（`lbo=1,sbo=64,layout=2`、`ki*32 B`）。
  曾被误判为"unpacked"（§43）——那是 container footprint 与容器内数据的混淆。**默认已转正**（`DSV41_MOE_BS_UNPACKED=1` 可回退对照）。
- **🎯 两个接线缺陷已修（§62/§65）**：① **swapAB epilogue 输出打包错位**（修复前 640 个输出里 **512 个位置错**，
  纯算术核验修复后 640/640 正确）；② **`HANDWRITTEN` 分支的 `goto` 跳过 `[NC]` 探针**（已在 `goto` 前接上探针）。
- **⚠️ 一个我自己造成的回归已回退（§65 M1）**：为定位 `[NC]` 加的 `[NC-TRACE]` 仪器化把独立 `return X;`
  变成"打印块 + 无条件 return" ⇒ **9 处守卫失效、BS 臂整体死掉、e2e 静默回落老路径** ⇒
  **F/P/U/E 各回合的文本判据全部作废**（那些"文本更接近计数"的结论建立在错误前提上）。
  **教训：仪器化只能"加"，绝不能重写控制流；改完必须验证语义（编译通过 ≠ 行为不变）。**
- **精度（用户硬性要求："不能高也不能低"）——已实现 5 道门，全部默认 OFF、待逐项转正**：
  | 门控 | 语义 |
  |---|---|
  | `DSV41_ROUTED_DOWN_QUANT` | 路由权重时机 + routed-down 输入量化（§14/§21） |
  | `DSV41_WINDOW_KV_QUANT` (A2) | 窗口 KV 的 fp8(block32, e8m0) 就地往返 |
  | `DSV41_COMPRESS_LATENT_QUANT` (A3) | 压缩 latent 的 fp4(**block16** + **e4m3 非幂次**标度) 往返 |
  | `DSV41_INDEXER_FP4_RT` (A4) | indexer q/k 的 fp4(block32, 幂次) 往返 |
  | `DSV41_ATTN_P_BF16` (I3) | attention PV 的概率操作数 bf16 舍入 |
  **单测 116 passed / 0 failed**（对照独立 Python 参考）；**OFF 路径逐门核实**（四门"不调用"型、I3 内核侧参数型）；
  **无半挂**（A2 覆盖融合/非融合、A4 覆盖 q/k、A3 两个调用点）。
  **⚠️ 出货脚本尚未开启任一门**（§63/§64）⇒ 默认臂下仍不对齐。
  **转正流程**：`~/promote_precision.sh "<GATE=1> [DBG=1]"`（DBG 五点回读逐元素差 0 → `~/wq_check.py` 文本红线
  → 快速臂无回归），**逐项、一次一个变量**；四处改动的阶段与 MoE 解耦（§74）⇒ 不被 BS 臂阻塞。
- **累加序审计（§77）**：17 项顺序型差异中 **10 项 CPU 即可判定无害**（官方每个算子末端都有 bf16/fp8 量化边界
  吸收 ≤1e-6 的 f32 序差 ⇒ 每行约 1.3 个元素偏 1 个量化 ulp）；**4 项建议 GPU 抽检**；
  **3 项必须 GPU 量化**，而**全部风险的唯一落点是 head logits**（f32、无量化边界、末端直接 argmax）。
- **旧结论已作废（保留供参考）**：早期判定的"硬件把 fp4 按 PACKED 读"曾一度被 §43 推翻、后由 §47 **最终调和**"

  其中 **3 项是"缺失量化"**（A2 窗口 KV / A3 压缩 latent（fp4 block16 + **e4m3 非幂次标度**）/ A4 indexer q/k）。
  已开三条实现线；另有 1 项**已有补丁**（路由权重时机 + routed-down 输入量化，门控 `DSV41_ROUTED_DOWN_QUANT`）。
  **⚠️ 出货脚本没开该门**（§63/§64）⇒ 默认臂下仍不对齐。**转正流程已脚本化**：
  `~/promote_precision.sh "<GATE=1> [DBG=1]"`（DBG 五点回读逐元素差 0 → `~/wq_check.py` 文本红线 → 快速臂无回归），**逐项、一次一个变量**。
- **⚠️ 旧结论已作废**：曾判定"硬件把 fp4 按 PACKED 读"（旧 §29）——**这是错的**，见 §43。
  我那个冲激探针的"0x02 只剩一半"极可能是**探针自己只写了 64 B/行**造成的假象。
- **🎯 现在的权威事实（§43）**：另一个 subagent 用官方入口 `T.tcgen05_gemm_blockscaled()` 在本机跑出了
  **通过 float64 金标准**的最小参考（**两种朝向都 PASS**，`rel≈1.3e-07`）。其生成码给出的权威约定是
  **unpacked fp4 smem（1 值/字节，TMA 把全局 packed 展开）+ SW128 + descriptor (lbo=1,sbo=64,layout=2) +
  K 递进 `ki*32` + idesc `144708608|(ki<<29)|(ki<<4)`** —— **与我们现有 SW128 路径逐项一致**。
  生成码并**明确警告**"packed `float4_e2m1fn` 能编译能跑但静默错值" ⇒ `DSV41_MOE_BS_PACKED` **保持 OFF**。
- **🎯 §44 逐项核对结论**：布局公式 / descriptor / idesc / K 递进 / `enable_d` 时机（只在最首个子 MMA 清零）/
  SF 投递与转置顺序 / **SF 字节序（LSB = 组内最低 K-block）** / SF 词组 group-major / 粒度 32 —— **全部与官方 PASS 参考一致**。
  ⇒ **参数面已排除**，缺陷在**我们自己的实现细节**里；且官方 a8b4/a4b8 **两朝向都 PASS** ⇒ 我们的朝向依赖
  必然来自**我们自己**的 swapAB 实现（两朝向都错 ⇒ 先在**共用部分**找）。
- **在途线索**：subagent `bs-packed-geometry` 已改派为"拿官方 PASS 参考当 oracle，同输入逐元素对拍"，
  重点查：①A/B tile 的内容装配（激活行/ W 行距 2560 / `w_stride`）②**SF 内容与行的对应**
  （尤其 **W3 的 SF 是否落在 B 行 64..127 上**，以及 K 组 `k` 与逐迭代重写 tile 是否同步）③朝向/M-N 角色。
- **⚠️ `DSV41_MOE_BS_SCALEVEC1X` 也须保持 OFF**（官方生成码**没有** `.scale_vec::1X` 后缀）。
- **头号假设 = SF 字节序**（硬件可能 MSB-first：byte j 携带 K-block `3-j`；我们两侧都按 LSB-first 打包）。
  门控 `DSV41_MOE_BS_SFREV=1` 用一行 idesc 改动（`sf_id = 3-ki`）同时修正权重侧与激活侧。
- **门控一览**（全默认 OFF）：`DSV41_MOE_BS_{HANDWRITTEN,CANON,SCALEVEC1X,SWAPAB,SFREV,NUMCHECK}`。
- **诊断纪律（踩过的坑）**：
  1. `NUMCHECK` 需 **`DSV41_GRAPH_STEP=0`** 才触发（whole-step graph 捕获期 shim 一律 decline）；
  2. **16 组合微基准的 FAIL 不作数**（换布局结果逐位相同 = 物理上不可能）；
  3. in-tree **PH0 探针在本机跑不过它自己的金标准**（"已验证原语"的前提需重验）；
  4. 编译检查要编 **shim**（`moe_bs_handwritten.cu` 被 `moe_bs_shim.cu` `#include`，单独编必假报错）。
- **精度对齐（用户硬性要求）已审计，两处待修**（见 §14/§21，**等 BS 臂正确后一次改一个变量**）：
  ① 路由权重我方在 w2 epilogue 乘、官方在 w2 **之前**乘；② routed 的 down 输入我方是 **f32**、官方是 **e4m3(block32) 量化**
  （⇒ 我方精度偏高）。激活量化本身与官方**逐字节一致**。
- **现成远端工具**：`~/arm_run.sh <name> [ENV...]`（单臂 e2e + 开关回执 + [NC]）、
  `~/sfrev_round.sh`（SFREV 三臂决定性实验）、`~/next_round.sh`（微基准 + swapAB 矩阵）、
  `~/verify_correct.sh <port> <label>`（1..100 前 61 行 + 拉丁探针 + step p50）。
