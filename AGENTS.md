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

## 当前状态与下一步（2026-09-14——手写 fp4 MoE BS 臂：🎯 根因已定谳 = fp4 操作数必须按 PACKED staging）

**详细记录一律在 `docs/agent/moe-bs-crash-investigation.md`（§10–§22），本节只放结论与指针。**

- **手写 kernel 的 6 个真 bug 已修**：idesc `b_sf_id` 漏 `<<4`、smem 布局与 descriptor 的 swizzle 声明不匹配、
  K-block 递进单位误判 8×、缺 `fence.proxy.async`、缺 `tcgen05.fence::before/after_thread_sync`（3 处）。
- **🎯 根因（§29，实证非推导）**：**硬件把 fp4(E2M1) 操作数按 PACKED（2 元素/字节，低 nibble=偶 k）读，
  而我们按 unpacked（1 元素/字节）写** ⇒ 所有奇数 K 元素恒为 0、K 覆盖腰斩。铁证：B 字节全 `0x22`（两 nibble 都=1.0）
  时稠密 parity **relerr=0 精确 PASS**；全 `0x02` 时 D 恰好是金标准的 **1/2**。官方 W 子 tile 8192 B
  = 128 行 × 64 B 亦独立佐证 packed。**其它全部已验证正确**（36/36 冲激命中、A 侧字节级一致、descriptor lbo/sbo/K 递进、
  idesc、M/N 方向、TMEM 读回、MMA 确实发射——含 TMEM 毒化自证）。
  ⇒ 修法 = 把 fp4 操作数按 packed staging（64 B/行 for K=128）+ 相应改 descriptor/K-block 递进。
- **⚠️ 因此之前一切基于"全错文本"的结论都是在 B 打包错的条件下得到的**：朝向（SWAPAB）与 SF 字节序（SFREV）
  都必须**修好打包后用受控实验重新判定**（§27 给出了 `~/orient_controlled.sh`，且必须以 `[NC]` 数值为主判据）。
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
