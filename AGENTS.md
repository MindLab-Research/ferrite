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

1. **step time 必须真实测量且看 p50**：`[dspark] steps=` 的 draft/verify/commit 分解（累计均值）+ per-step `[dsv41] step pos=` 行的 **p50**（中位数，最后 200 步——用户裁决 2026-09-13）；**禁止吞吐反推**（受 prefill/accept 污染）。**每次 nsys 分析完必须给用户当前的时间 breakdown 表**（各项 ms/步 + 占比 + 修复载体 + 修复后目标——用户裁决 2026-09-13）。
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

## 当前状态与下一步（2026-09-14 中午——fp4 MoE crash 三路 runtime audit 完毕 + DIAG 在测）

**crash 排查完整记录**（illegal memory access in MMA kernel）：

**已否定假设**（三路 audit subagent + 手工验证）：
1. ✅relinquish_alloc_permit（修复 d25d5fe 但非根因——JIT 无它也不 crash）
2. ✅idesc ki-bits（illegal-instr-3 权威位表：[4,6)=b_sf_id 非 k_size，sf_id 选择是故意的）
3. ✅fast math（NO_FAST_MATH 也 crash——只改变错误类型 illegal instruction↔illegal memory access）
4. ✅w3 指针（w3-ptr-audit：w1=pool+0, w3=pool+870400, w_stride=2641920 全部正确）
5. ✅Eid 值域（eid-init-audit：每 rank 全部 384 expert TP-split by inter 非 EP，值域 [0,384) 全合法）
6. ✅descriptor 参数（desc-param-diff：6 个 descriptor 除有意 stride 差异外全部匹配；dtype 14=16U4_ALIGN16B 验证一致）
7. ✅内存边界（W1 TMA reach 1,012,674,560 < pool 1,014,497,280 ✓；shared memory 全在 kSmem 内；TMEM 160<512）
8. ✅JIT vs AOT 源码一致（diff 仅差手加的 relinquish asm）；模板一致；dtype 枚举一致

**关键事实**：
- JIT（纯布局 gstride=819200 + host descriptor）**不 crash 但读错数据**
- AOT（块布局 gstride=2641920 + shim descriptor）**crash（illegal memory access）**
- MMA skip 测试确认 crash 在 MMA kernel 内
- 8/8 ranks 全部 crash

**Runtime 诊断在途**：
- Eid DIAG（一次性打印实际值）+ SYNC-DIAG（per-kernel sync 隔离 gather/MMA/scatter）
- compute-sanitizer 脚本已部署（~/sanitizer_run.sh）——如果 DIAG 不能定位
- 诊断决策树脚本已部署（~/diag_chain.sh）

**其他交付**（全部已提交）：
- C5 shared expert TileLang fp8 MMA（gen_sh_exp_aot.py 430行 + sh_exp_shim.cu 303行 + Rust 接线）
- P5 MTile grid-stride（激活 staging 一次共享，DSV41_MTILE_GRIDSTRIDE gate）
- H1 draft hc front 别名（DSV41_DRAFT_HC_FRONT）
- D3 KV block 32（官方 block_size=32，window-KV 当前 raw f32）
- build.sh 选择性 fast-math（tilelang_gen 无 fast math，其余保持——防 err 900 capture crash）
- push400_test.sh 已更新含 SH_EXP_TILELANG + MTILE_GRIDSTRIDE_T=8

**当前账**：step ≈28.6ms @ acc 2.24 ⇒ ~104 tok/s。
**TileLang 预期（全接线后）**：投影 7.4→1.5ms + MoE 10→2.8ms ⇒ verify ~10-12ms ⇒ step ~14-16ms ⇒ **~200-230 tok/s**。
**在途**：第六次重编（63a0f948：xsc 算术修复 + moe_bs 守卫 + 全部接线）→ DUAL_ARTIFACT_OK 后 T 臂双挂重跑（`~/tl_dual_test.sh` 已部署：GEMM_TILELANG + GEMM_TILELANG_EAGER + 计数 first-51 + 拉丁 + dspark 分解）。

**已判死（勿重试）**：MPAR（两败）、⑤a L2 直读（四档负）、proj-mma（acc 崩 0.02）、p3lite+ALIGN（acc −0.22 + l4 parity FAIL）、GROUPED 布局（+16ms）、g1 union（+0.46ms）、launch 税、复制 eager 重写、M-tile 参数调优（BN 钳 4）、bf16 dequant 显存（+105GiB/rank）。

**T 臂乱码关键教训**（`docs/agent/tl-garbage-verdict.md`）：
1. **`.max()` 的域比存在本身重要**——放在 `/32` 前面是 no-op，移到后面才是真 floor。
2. **parity 微基准的输入必须与生产布局同构**——hash 输入验证的是"自洽"而非"同构"。
3. **"两个谓词是一对"**——A 有 m-predicate 而 ASC 没有，靠 reduce 兜底是隐性耦合。
4. **m=1 基线对 stride 类缺陷免疫**——out_stride/a_stride/scale pitch 错误只有多行才暴露。
5. **半挂配置是设计内非法**——接线契约明文规定双侧同换（eager + verify 同挂）。

- **acc 2-3 达标 ✓✓**：S1 tap 越界根因修复（`hc_collapse` per-row pre 契约 vs m-row hook 单行 4-float 越界——Fix A `dspark_pre_mean_r` 复制零成本等价，commit 6f6f513）→ **mean-k 1.34→2.240**（超 lazy 2.120；归因闭环 fix-off=1.38）；TAP_PARITY H 区全 IDENTICAL + COMP_PARITY 19/0（S2 干净）——双嫌疑闭环。
- **正确性红线通过**：出师表拉丁 = EAGER 对照同现（模型行为）；DIFF_EAGER 48/48 none；计数 first-51 OK 全臂。

**当前账（step ≈28.6ms @ acc 2.24 ⇒ ~104 tok/s）**：verify 24.5（已含 hc −3.4 + ATTN_MROWS −2.98 + SH/COMP/INDEXER 融合）+ draft 3.8 + commit 0.5。
**在途修复（4 subagent）**：proj-mma 接线（−3.5ms 中央）+ ⑤a L2 广播（−4~6ms，逐位安全）+ tcgen05 gate/up 716（MoE grouped 唯一解锁，−2ms）+ draft parity 套件。
**已否决（勿重试）**：MPAR warp-并行（二连败）、wo_a nwarps（8 已最优）、p3lite+ALIGN（acc −0.22）、GROUPED 无 TCGEN05（+10.3ms）、COMP/ENGRAM（票面高估 10×）。
**方法论**：AR_SAFE nsys 表对生产无效；票面必须 nsys 时间占比；mrows 零摊销（⑤a/⑤b 是真解）；env 回读断言必须；step p50 口径 + 出师表红线 + [sh-gate] 回执。


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
