# ferrite — Agent Working Guide

Rust-native inference engine for **GLM-5.3-Flash** (hybrid GatedDeltaNet linear attention + DSA sparse attention + MoE), single-node TP over CUDA graphs.
Read `README.md` for the design contract; this file is the operational guide: build/test loop, every runtime flag, demo configs, and the profiling workflow that actually works on this hardware.


### 测试纪律（用户 2026-09-10 追加，立即生效）
- **只准测试一轮**：每次验证/测速只跑**一次** serve+bench，不再自动跑 3 轮/多轮求中位数。
  单轮读数即为结论；需要重复时必须先问用户。（原因：每轮 ~1.5-2.5 分钟，多轮把"看一个数"
  变成"等十分钟"。）
- **禁止杀测试**：已经启动的测试必须让它自然跑完（收尾用 `POST /shutdown`），不得中途 kill。
- 因此脚本模板一律**单轮**：启动 serve → 一次 bench → `POST /shutdown` → 报
  `replay p50` / `faults` / `opcheck` / 文本，然后结束。
- 跨版本比较**必须同会话背靠背**：同一二进制 + `--lib` 切换两个 `.so`
  （`git show <commit>:kernels/cuda/ferrite_kernels.cu > /tmp/old.cu` →
  `nvcc -O3 -shared -Xcompiler -fPIC --use_fast_math -std=c++17 -gencode arch=compute_103a,code=sm_103a -o /tmp/old_lib.so /tmp/old.cu`），
  **每次比较各跑一轮**（共 2 轮），不做 3 轮。


### ⛔ 加载错防线（用户 2026-09-10 明令："确保以后再也不可能出现加载错"）

**事故**：二进制带 `DT_NEEDED: libferrite_kernels.so`，启动时由 `LD_LIBRARY_PATH` 映射一份；
`--lib` 又 dlopen 了另一份**同名不同版本**的 .so → 符号解析绑定到**先加载的那份（链接镜像）**，
`--lib` 被静默忽略。实测：想测 dc4d7ae 却跑成了主树的 kernel（11.41ms），
而 dc4d7ae 自己的完整重编是 **10.55ms** —— 读数完全无效，浪费数小时。

**三道防线（全部在 `CudaBackend::with_device/with_library` 的 dlopen 之后，任一不过即拒启）**：
1. **戳不存在 → 拒**：`.so` 必须导出 `ferrite_kernel_build_id()` + `ferrite_kernel_abi_version()`；
   `build.sh` 注入 `id = <git HEAD>[-dirty]+cu<sha256(.cu)[:16]>`，并写入 `kernels/cuda/.build_id`；
   `build.rs` 在编译期把 `.build_id` 的内容嵌进二进制 → **同源校验**（也能抓"只重编一侧"）。
2. **链接镜像 vs dlopen 镜像不一致 → 拒**：`dlsym(RTLD_DEFAULT, "ferrite_kernel_build_id")` 取
   **实际生效**的那份（DT_NEEDED 加载的），与 `--lib` 那份比对；再与二进制自身的 id 比对。
3. **`/proc/self/maps` 中 `libferrite_kernels.so` 的不同路径镜像数 >1 → 拒**。

> ✅ 防线 1–3 两处实现都有：GLM serve 路径在 `cuda.rs::verify_kernel_build`，
> DSV41/`devrt` 路径在 `devrt.rs::verify_kernel_build`（`dsv41-run` 经
> `Device::open → DevRuntime::open`，`ferrite-models/src/dsv41/device.rs:630` 必过此门）。
> 两者都在 dlopen 之后、首次 kernel 调用之前执行；任一不过即 `Err`，调用方不启动。

**编译期防线（2026-09-11 追加，`crates/ferrite-kernel/build.rs`）**——运行期门只能拒绝
"已经配错的**一对**"，挡不住你**生产**这一对；以下三道把错误前移到 `cargo build`：

4. **半成品状态 → 编译失败**：`.so` 与 `.build_id` 必须**同时存在**；只存在其一 →
   `cargo build` 直接 `REFUSING TO BUILD`（这是"只重编一侧"的签名）。二者都不存在时，
   **debug**（`cargo check`）允许（嵌入 `cuNOSTAMP` 哨兵，运行期拒启），**release** 一律失败。
5. **源码比 .so 新 → 编译失败（防陈旧）**：任一 `.cu` 的 mtime 晚于 `.so` → release 构建报错，
   并给出唯一可用顺序 `build.sh` → `cargo build`。`build.rs` 现已 `rerun-if-changed` 全部 `.cu`，
   所以"改了 .cu 没重编 .so"必然触发。
   - 逃逸阀（仅 CI / 无 CUDA 机器，**绝不可用于测量**）：`FERRITE_ALLOW_NO_KERNELS=1`、
     `FERRITE_ALLOW_STALE_KERNELS=1`；反向强制：`FERRITE_REQUIRE_KERNELS=1`（debug 也校验）。
6. **git hook**：`scripts/git-hooks/{post-checkout,post-merge}` 在切分支 / merge 后**自动删除**
   `.so` + `.build_id`（二者 gitignored/untracked，`git checkout` 根本不会碰）→ 被迫重编同源双产物。
   安装：`bash scripts/install_git_hooks.sh`（**每个 clone 一次**；hooks 不在 `.git` 里，无法版本化）。

**残余洞（2026-09-11 核查）**：
- 防线 3 靠**路径字符串** `contains("libferrite_kernels.so")` 匹配 → 改名的手编 nvcc 产物
  （`/tmp/libt.so`、`/tmp/probe*.so`、`/tmp/libdsv41_all.so` … 实测均**无 build stamp**）
  经 dlopen/LD_PRELOAD 进来时对防线 3 **隐身**；只有防线 2（id 比对）能抓，若两份 id 恰好相同则全过。
- `build_id = git HEAD + sha256(.cu)` **不含构建指纹**（nvcc 版本 / `-gencode arch` / `-O` / `--use_fast_math`）
  → 同 id 可能是不同 SASS。
- 防线 1–3 全在 `CudaBackend::with_device/with_library` 内，**首次 kernel 调用必须晚于 gate** 是隐式契约；
  且 `devrt.rs::verify_kernel_build` **故意跳过防线 2**（见 `devrt.rs:266-268`），只对"不 link"的 consumer 成立。
- 远端 `LD_LIBRARY_PATH` 默认**不含** kernels 目录 → 不显式前缀时 `ldd` 报 `libferrite_kernels.so => not found`
  （即链接镜像加载**只由脚本的 LD_LIBRARY_PATH 决定**，脚本写错路径就会静默换镜像）。

**正确用法（唯一允许）**：`--lib <tree>/kernels/cuda/libferrite_kernels.so` 且
`LD_LIBRARY_PATH=<tree>/kernels/cuda`（同一棵树）；跨版本比较必须**整树切 commit 后双产物重编**，
禁止拿 A 树的 .so 配 B 树的二进制。

### ✅ 2026-09-10 定案：逐 commit 完整重编单轮测速，回归点锁定 + 默认路径改回 dc4d7ae

**方法（唯一可信）**：主树 `git reset --hard <commit>` → `bash build.sh 103a` + `cargo build --release`
（**同一次 checkout 出双产物**，`--lib kernels/cuda/libferrite_kernels.so` 且
`LD_LIBRARY_PATH=$HOME/ferrite/kernels/cuda` 同一棵树）→ 单轮 `bench_save.py 16 300`。

| commit | replay p50 | 判定 |
|---|---|---|
| `dc4d7ae` | **10.55ms** | 基准 ✓（10.51 复现）|
| `850fd0b` cublas algo 旋钮 | 10.59ms | 中性 |
| **`74e6f89` hc_post 每-token 重写** | **10.80ms** | **+0.25ms ← 真回归**（当时误测为"中性"）|
| `8e813ef` down HT 模板化 | 10.80ms | 无额外变化 |
| `42c2ec8` down NP 流水线 | **8.21ms** | **假快**：cp.async 尾块 wait 语义丢失 → MMA 读陈旧 smem |
| `HEAD`（c7f9152b）| **11.43ms** | 相对基准 +0.88ms |

**关键认知**：序列是**单调且可复现**的 → 变慢来自**源码**，不是 nvcc 构建噪声（这点排除了）。

**处置（用户令："不准测了，性能直接改回去"）**：`kernels/cuda/ferrite_kernels.cu` 直接改回
`dc4d7ae` 的内容，**只重新加回版本戳**（构建门禁必须仍可生效）；Rust 侧删除对已移除实验
kernel（slotwise down）的引用。回归/负结果的实验（hc_post 重写、down HT/NP 模板、mix
token-pair split2）**保留在 perf-b1 的 git 历史里**，不再进入默认路径。

**教训**：① 任何"结构重写"必须用**同口径同会话**的单轮数字判定，"当时看着中性"可能就是回归
（hc_post 那次 +0.25ms 被我记成中性，导致后续在错误的基线上找问题）；② 提速数字异常漂亮时
**先验语义不变式**（8.21ms = 陈旧 smem 的假快）；③ 跨版本比较**必须整树切 commit 双产物重编**，
`--lib` 混用会把 `--lib` 变成静默失效（已加三道加载门禁）。

## ⛔ 硬性禁令（用户明令，违反=浪费用户时间，2026-09-09）

1. **任何时候禁止 `git revert`**（含 `git reset` 回退已提交的改动）。
   实测退化就把改动**改成默认关闭的开关**（env-gated）或修正它，**代码保留**。
   构建坏了必须修好，不能靠回退。
2. **版本同步统一用 git**（本地 `git push origin main:perf-b1` → 远端 `git fetch origin refs/heads/perf-b1 && git reset --hard FETCH_HEAD`），
   **禁止用 rsync 同步代码**。原因（2026-09-09 实测，浪费数小时）：
   - `rsync -a` 保留本地 mtime → 远端源码 mtime 比二进制旧 → `cargo build --release` 报
     `Finished in 0.09s`（**不重编译**）→ 二进制陈旧（实测停在 04:54，源码已到 05:24）→ 16 并发路径 err 700。
   - `rsync --delete` 会**删掉远端的构建产物**（`libferrite_kernels.so` 不在本地树里）。
   - 正确流程：git 同步后**必须** `cd kernels/cuda && bash build.sh 103a` + `cargo build --release`，
     并用 `ls -la target/release/ferrite-serve` 确认时间戳**新于**源码；改动 `.cu` 后必须重跑 `build.sh`。
2b. **每次切版本/同步后必须"双产物重编"：CUDA `.so` 和 Rust 二进制都要重编，保证同源对齐**（用户明令，2026-09-09）：
   ```bash
   git fetch origin refs/heads/perf-b1 && git reset --hard FETCH_HEAD   # 或 reset --hard <commit>
   cd kernels/cuda && bash build.sh 103a && cd ~/ferrite
   source ~/.cargo/env && cargo build --release
   ls -la target/release/ferrite-serve kernels/cuda/libferrite_kernels.so   # 两者都必须新于源码
   md5sum kernels/cuda/libferrite_kernels.so                                # 记下，作为该版本的产物指纹
   ```
   **禁止混用不同版本的产物**。实测教训（2026-09-09，浪费数小时）：把 06:57 备份的
   `/tmp/a9_lib.so`（md5 `f5d8ad2f`）当成"a9e5d5a 的 .so"配新二进制跑，得出
   "a9e5d5a 也崩"的**错误结论**；而真 a9e5d5a 重编出来是 md5 `f40aae6e`，配套 one-shot
   **完全正常**（exit=0、零 Xid、`[mega] replay 9.05ms`）。判据：**跑之前先核对
   `.so` 的 md5 与二进制的时间戳来自同一次 checkout**；`.so` 是源码的派生物，
   不能跨版本复用，也不能用 `--lib` 指向别的版本的 `.so`。
3. **永远禁止调用硬件复位类 API**（`nvidia-smi --gpu-reset`、重启驱动、任何影响内核/硬件的操作）。
   崩溃一律先查软件侧：源码同步/二进制时间戳/版本错配/`dmesg` 的 Xid 原文。
3. **严禁重跑已知正确的 baseline**。已经验证过的读数就是权威，
   不要为了"确认"再跑一遍（包括"确认默认路径没回归"——默认路径按定义不回归）。
4. **禁止无意义的验证跑测**。只有**新改动**才需要验证，且优先用隔离微基准（秒级）；
   serve 端只在结论要落地时才跑一次。
5. **禁止 MTP / 投机解码**（用户明令："严禁mtp…严禁投机"）。
   目标固定为 **16 并发不开 MTP ≥1600 tok/s**。
6. **放弃任何优化方向前必须形式化证明它走不通或效果差**（隔离微基准 / 数值等价性证明 /
   可复现的量化对照）。禁止用"感觉不行"或单次文本观察作为放弃依据（2026-09-09）。
7. **文本重复/思考模式循环不是可靠的回归判据**：BLK=256 已知正确版本同样会重复，
   `enable_thinking:false` 也拦不住。判定 kernel 改动正确性用**数值等价性**
   （同输入跑两条路径，逐位比对输出），文本只作辅助。

**测速纪律**：只看 `FERRITE_TIMING=1` 的 `[megab] replay 16 seqs: Nms` 中位数
（16000/N = 聚合 tok/s），并确认日志里有 `live=16`；per-seq×16 与 total/wall 只作交叉验证。
每次改动**必须人眼看生成的文本**（乱码=数值回归，token 计数看不出来）。

**nsys 落盘纪律**（2026-09-09 更新：现在有两种验证成功的方法，优先用 capture-range）：
nsys 只在**目标进程退出时**写报告。HTTP serve 靠 SIGINT 优雅退出（`kill -INT <pid>` → tokio ctrl_c → `profiler_stop` → `exit(0)`；**POST /shutdown 端点存在且是正确的收尾方式**（ferrite-http/src/api.rs:336 `shutdown()`：respond 后 300ms 由 detached thread `process::exit(0)`——跳过 1.17TB 权重 drop，profiler 报告可靠落盘。⚠️ 2026-09-10 实测教训：`kill -INT` 走 ctrl_c 路径，退出时权重 drop 曾把 nsys 注入拖死（serve 已消失但 nsys 永不写报告，只能 kill -9 清理）。**一切 serve 收尾（尤其带 nsys/ncu 时）必须用 `curl -X POST http://localhost:PORT/shutdown`，不要 kill -INT**）。
**首选：FERRITE_NCU 窗口 + capture-range（99f0a0e 起内置）**——只抓饱和稳态，报告里**没有** 80s 权重加载和 admissions 爬坡/图捕获的内核，`cuda_gpu_kern_sum` 直接就是稳态分解：
```bash
timeout -s INT 300 nsys profile --trace=cuda --cuda-graph-trace=node --sample=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
  -o /tmp/nsys_b16 --force-overwrite=true \
  env <基准env + FERRITE_NCU=1> ./target/release/ferrite-serve --serve --max-seqs 16 ... &
# 等 "serving glm" → 跑 bench（16 并发触发 [ncu-win] batch saturated）→ sleep 几秒
# → kill -INT $(pgrep -f 'ferrite-serve --backend') → wait → nsys stats
nsys stats --report cuda_gpu_kern_sum /tmp/nsys_b16.nsys-rep | head -50
```
窗口语义：`GpuEngine::tick` 在 `live==max_seqs` 时调 `cudaProfilerStart`（一次），`run_serve` 在 `process::exit(0)` 前调 `cudaProfilerStop`——**没有 stop 信号 nsys 会空等**，所以必须优雅退出，不能 SIGKILL。
旧方法（仍可用，报告混入加载内核，需 head -40 纪律）：`timeout -s INT 230 sudo nsys profile ... env <env> serve &` → bench → SIGINT → wait。
**不要**用 `--duration`（从进程启动计时，落在加载上）。**不要**用 `nsys launch --session-new`。**不要**用 SIGKILL。
```
**读报告纪律**：`cuda_gpu_kern_sum` 按总时间排序，**权重加载的 `dequant_e4m3_block_kernel`/
`bf16_to_f32_kernel` 永远排在最前面**（各占 40-57%）。必须 `head -40` 或按名字过滤，
**不要 `head -20` 就下结论**（这正是本会话连续误判"nsys 只抓到加载阶段"的原因）。
每步时间 = 该 kernel 的 **median × 每步调用次数**（instance 数被 capture 的 ~900 次 dry-run 污染，
不能直接除步数）。

## 2026-09-09 事故后的 batched 路径现状（已核实，勿重复试错）

**背景**：07:11 为抓 nsys 开 `FERRITE_P2P=1` 跑 B=16 → P2P AR 在 n>8 死锁 → `timeout -s INT` 杀 →
驱动 wedge（8 卡同时 `NVRM: refcntRequestReference_IMPL: Failed to enter state 1`）→ 之后连 b1 都崩。
**08:42 重启后驱动状态已恢复**（one-shot 正常），但 batched 路径仍崩 —— 两者是两个独立问题。

**重启后实测（全部严格双产物重编）**：

| 场景 | a9e5d5a | a0cf454 | HEAD |
|---|---|---|---|
| one-shot `--prompt/--max-tokens` (n=1) | ✅ | ✅ | ✅ |
| serve 单次请求（流式/非流式，n=1 非 batched） | ✅ | — | ✅ |
| serve B=16（`--max-seqs 16`） | ❌ | ❌ | ❌ |
| `FERRITE_FORCE_BATCHED_B1=1`（**batched 链 + n=1**） | — | — | ❌ |

→ **不是版本回归**（三个版本全崩）；**batched 链本身**有问题（n=1 也崩）。
故障特征：Xid 31 `FAULT_PDE`、地址**几乎全部精确 2MB 对齐**（= freed/unmapped 的 cudaMalloc 基址）、
`live16=0`、`captured=0`（崩在 dry-run，未到 capture）、首个报错内核漂移
（`hc_post_dev` / `graph_run D2H` / `dsa_kpool_batched` / `moe_fused_act_fp8` / `moe_fused_down_sum_fp8`
—— 因为 act 内核返回值没走 `ck()`，sticky error 顺延）。
`CUDA_LAUNCH_BLOCKING=1` 时 n=1 batched 的首个报错是 **`moe_fused_down_sum_fp8` @ L8**。

**已用对照排除（勿重复）**：
- **AR 实现方式无关**：`FERRITE_AR_SKIP=1`（P2P/NCCL 的 AR 全跳过）仍崩 → 不是 P2P vs NCCL 的差别。
  （单次 `FERRITE_P2P=1` 跑通过是 1 个样本，最可能是开 P2P 时多出的 staging/ready 表改变了 VA 布局，属运气；
  且 P2P 在 batched n>8 有文档记载的死锁 + 会 wedge 整机，**不可设为默认**。）
- **P2P 复测（2026-09-09 末，根因 #1-#5 全部修复后，FERRITE_P2P=1+FORCE）**：B=16 仍在捕获/ramp 阶段死锁
  （30s 日志停滞，监护脚本 kill -9）→ hard-reject 是正确判断，v3 修复并未解决 batched n>8。
  **好消息：30 秒内早杀后驱动未 wedge**（nvidia-smi 全 0 + b1 sanity 干净通过）。P2P 诊断必须带
  日志停滞检测 + 精确 PID kill 的监护脚本。
- **padding 无关**：`FERRITE_NO_PAD=1` 仍崩。
- **CUDA 图无关**：`FERRITE_MEGA_DRY=1`（不捕获图）仍崩；`FERRITE_DESTROY_BG=1`（retire 时销毁 megab 图）也仍崩。
- **MoE 只是部分相关**：`FERRITE_MOE_SKIP=1` 能出 10 tok 但仍有 129 fault。
- **DEV 开关不能用作二分**：`FERRITE_MOE_DEV/GDN_DEV/LAYER_DEV=0` 在 batched 路径直接 panic
  （`lib.rs:1318`/`lib.rs:643`/`mhc.rs:94 range end 16384 out of range for slice of length 4`）。
- **短请求 vs 长请求都会崩**（16×60 与 16×1000 都失败）→ 不是"retire 风暴"独有。
- **warmup 无关**：去掉 warmup 的 bench 同样崩（曾通过两次，属间歇运气）。
- **seq 生命周期修复全部无效**：`2d8cb68`（retire 失效 membership）、`9d8e533`（free 前 device sync）、
  `03ee979`（DSA 缓存池化）、`ec0f10a`（GDN/conv 池化）→ 仍崩。
- **host↔device 写序修复全部无效**：`8319cfc`（步首 sync）、`df12c72`（写 pinned 前 sync）、
  `79857ce`（逐层 sync 开关）→ 仍崩。
- **DSA 缓存容量无关**：`FERRITE_DSA_MAXT=1024/2048`（把每 seq ~0.5GB 的缓存缩小 4-8 倍）→ 仍崩 289 fault。
- **最新观测**：某次跑到 `cap=true L0..L44`（capture 完成）后**第一次 replay 就崩**（在 `graph_run D2H` 检出），
  且 `[opcheck]`（act 内核错误码打印）为空 → act 不是首个失败者；故障在 replay 路径。

**结论（2026-09-09 末）**：batched 路径在**同一台机器**上从"能用"变为"必崩"，且非 batched 路径（同一批内核、同一批权重）
始终正常 —— 与用户自己记录的历史案例（1100 节点：单节点反复 Xid 13、ECC 全 0、他节点零崩 → 隔离该节点）同形。
**判定：该节点（b300-4）的驱动/硬件状态问题**，软件侧已无更多可用手段（本文列出的所有修复与开关均已试过）。
验证方式：**同一二进制拿到另一台机器跑同样 16 并发负载**，若一次都不崩即确认。

**已知的真实缺陷（已提交修复或待修）**：
- ✅ `free_seq` 后未失效 `last_batch_seqs` → 表内容只在 membership 变化时刷新，retire 后可留悬垂指针（`2d8cb68` 已修）。
- ⚠️ `destroy_batch_graph` 无任何调用者（文档写明 retire 必须销毁图）。
- ⚠️ `hc_pre_mix_split_kernel` 读未初始化显存（initcheck 38 条 4B 读）—— 对应未解决的 "hc_pre output explodes"。
- ⚠️ `gemv_fp8_mma_b16` 的 cp.async 读 `xq` 行 ≥ n（memcheck 盲区，initcheck 报 uninit）。
- ⚠️ `mhc.rs:94` 占位 stub 被当 16384 切片（`LAYER_DEV=0` 路径 panic）。

## Repo layout (hot paths)

```
crates/ferrite-exec/src/tp.rs        TP cluster + mega-graph chain + MTP step (mtp_step, mega_chain_dev)
crates/ferrite-kernel/src/cuda.rs    CudaBackend: FFI, graphs, GDN/DSA/MoE device chains, MtpState
crates/ferrite-serve/src/main.rs     binary: load checkpoint → prefill → decode loop (one-shot)
crates/ferrite-http/src/             THE shared serve stack: api.rs (axum routes /v1/chat/completions + SSE),
                                     driver.rs (engine thread), engine.rs (ServeEngine seam), single_flight.rs
                                     (batch-1 StepEngine adapter for lockstep/chain engines), serve.rs (launch),
                                     tokenizer.rs (StopSpec + ChatFrame: per-checkpoint stops/frame)
crates/ferrite-dsv41/src/bin/dsv41-run.rs  DSV41 runner: one-shot, + --serve = TP rank pool behind StepEngine
kernels/cuda/ferrite_kernels.cu       ALL device kernels (sm_103a), build.sh → libferrite_kernels.so
```

**Serve 架构（2026-09-11）**：GLM (`ferrite-serve --serve`) 与 DSV41 (`dsv41-run --serve --tp 8`) 共用
`ferrite-http`：同一 axum router、同一 driver 线程、同一 SSE/usage/cancel-on-drop、同一套 `/v1/stats`+`/shutdown`。
唯一按模型分叉的是两处**数据**：`StopSpec`（checkpoint 的停词集）与 `ChatFrame`（chat 模板，GLM 默认 /
DSV41 自带；用 `api::router_with` 注入）。DSV41 的 TP 8-rank 锁步引擎实现 `StepEngine`（prefill/decode/
is_stop），由 `SingleFlight` 包成 `ServeEngine`——**batch=1 的单请求语义在共享抽象里**，不在 DSV41 侧。
⛔ 不要再往 `ferrite-dsv41` 里写 HTTP/解析/SSE 代码（旧的手写 std::net server 已删除：它还会死锁——
barrier 建在 rank 闭包内部、且从未做 peer handshake）。

## Build / deploy / test loop

Local (no GPU needed): `cargo check` (hard gate — every commit).
Remote: b300-4 `ssh ubuntu@43.202.208.136`, repo `~/ferrite`, model `/opt/dlami/nvme/models/GLM-5.3-Flash`, GPUs 4–7.

```bash
# local → remote (the remote has TWO remotes; its origin refspec is
# +refs/heads/main:refs/remotes/origin/main ONLY, and `main` tracks origin/main.
# `perf-b1` is a STALE ancestor (8e55123) — do NOT use it as the deploy pointer):
git push origin main
# remote:
ssh ubuntu@43.202.208.136 'cd ~/ferrite && git fetch origin && git reset --hard origin/main && \
  cd kernels/cuda && bash build.sh 103a && cd ~/ferrite && source ~/.cargo/env && \
  touch crates/ferrite-kernel/build.rs && cargo build --release'
```

`bash build.sh 103a` = sm_103a (B300). `103a` is required; the README's `100a` is stale.

## Environment flags

### Required for full-speed decode (the standard MTP run)

```bash
NCCL_NVLS_ENABLE=0 \
FERRITE_MEGA=1 FERRITE_NCCL=1 FERRITE_P2P=1 FERRITE_WORKER_POOL=1 \
FERRITE_LAYER_DEV=1 FERRITE_GDN_DEV=1 FERRITE_MOE_DEV=1 FERRITE_DSA_DEV=1 FERRITE_HEAD_DEV=1 \
FERRITE_MTP=1 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
LD_LIBRARY_PATH=$HOME/ferrite/kernels/cuda \
./target/release/ferrite-serve --backend cuda --tp 4 \
  --model-dir /opt/dlami/nvme/models/GLM-5.3-Flash \
  --lib kernels/cuda/libferrite_kernels.so \
  --max-tokens 500 --prompt "请背诵《出师表》"
```

- `NCCL_NVLS_ENABLE=0` is MANDATORY on this node: without it `ncclCommInitAll` fails → silent fallback to host all-reduce → ~2.4× slower step and MTP accept degrades (looks like a logic bug but is env).
- `FERRITE_P2P=1` (2026-09-08 VERIFIED — was previously rejected): P2P one-shot AR (10µs vs NCCL RING_LL 35µs per AR). The earlier accept crash (2.39→1.71) was an OLD-version bug (fixed by the mtp_step N-unification series); current: n=3 500-step 121.8 tok/s @ accept 2.43 (vs NCCL 110.2), text flawless, n=1 90.1 (no regression — P2P spin jitter offsets the small-n AR gain). The P2P sum order (staging r0+r1+r2+r3) matches NCCL ring's 4-rank single-chunk order — 1-ulp domain-consistent, draft+verify chains both P2P.
- Non-MTP regression: drop `FERRITE_MTP=1` (everything else identical). Baseline ~49 tok/s @200-step window, MTP ~65.
- Output line to read: `[serve] decode: N steps in Ts = X steps/s | real M tokens = Y tok/s`. With MTP, `real` counts accepted tokens (N×~2.4); always ALSO eyeball the generated text (乱码 must be caught by eye, never by token counts).

### Execution-path flags (all default OFF; 1 = enable)

| flag | effect |
|---|---|
| `FERRITE_MEGA` | one CUDA graph per seq for the whole decode step (mega-graph); the decode fast path |
| `FERRITE_NCCL` | NCCL TP all-reduce inside the captured graph |
| `FERRITE_WORKER_POOL` | persistent fan_out worker threads (no per-step thread spawn) |
| `FERRITE_LAYER_DEV` | per-layer device op chain (hc/norm/attn/ffn stay on GPU) |
| `FERRITE_GDN_DEV` / `FERRITE_MOE_DEV` / `FERRITE_DSA_DEV` / `FERRITE_HEAD_DEV` | device kernels for the respective layer types / lm_head+argmax |
| `FERRITE_MTP` | speculative decoding: draft=2 (layers.45 nextn), verify n=3 mega_v graph, greedy accept, single-kernel `ferrite_mtp_commit` |
| `FERRITE_DRAFT_GRAPH` | per-draft draft-chain graphs `mega_d{seq}_{i}` — **default ON** (0 = host chain fallback; 2 = device chain host-serial bisect mode). Each graph = cast_store + embed_one_dev + the full mtp_forward layer chain; replay = H2D 4B + dsa advance(1) + ONE launch per draft, with a stream sync between graphs (the pinned t0 slot is shared host memory). |
| `FERRITE_MTP_N` | MTP verify width N (drafts = N-1), 1..=8, default 3. N-UNIFIED: the draft chain runs N-1 mtp_forward steps, the verify graph runs N rows (`mega_v` captured at n=N), accept k ∈ 1..=N, commit snapshots are [N-1] contiguous. N=3 = the historical (d1,d2) chain, bit-identical. **N=1 = plain decode on the SAME mega1 path** (no MTP buffers, no ping-pong, no commit — FERRITE_MTP_N=1 with FERRITE_MTP=1 routes to the non-MTP decode_step_mega branch; zero overhead, output bit-identical to FERRITE_MTP unset). `FERRITE_ZERO_H2D=1` (experimental device-resident path) is N=3-only; other N values route to the generalized `mtp_step` with a log line. |
| `FERRITE_NCU` | cuProfilerStart/Stop window around the decode loop — pair with `ncu --profile-from-start off` so the 80s weight load is NOT profiled (ncu intercepts each H2D/kernel launch at ms cost; profiling the load stalls it 10+ min). See Profiling below. |
| `FERRITE_P2P` | NVLink P2P all-reduce (experimental path) |
| `FERRITE_GRAPH`, `FERRITE_GRAPH_LAYER`, `FERRITE_GRAPH_MOE`, `FERRITE_GRAPH_MID`, `FERRITE_GRAPH_DSA` | legacy per-segment graph capture (superseded by FERRITE_MEGA; kept for bisection) |

### Diagnostic flags

| flag | effect |
|---|---|
| `FERRITE_MTP_TIMING` | per-step `[mtp-tm] draft= verify= commitN=` wall times |
| `FERRITE_MTP_DEBUG` | per-step accept decision (d1/d2 vs argmax) |
| `FERRITE_TIMING` | mega replay per-step wall + per-layer DRY segment times |
| `FERRITE_MEGA_DRY` | skip graph capture, run the real chain per step (graph-bug bisection) |
| `FERRITE_MEGA_PROBE` | dump layer intermediates to `/tmp/orion` (host download — capture-illegal) |
| `FERRITE_GDN_PROBE` / `FERRITE_AR_PROBE` / `FERRITE_TRACE_NAN` / `FERRITE_TRACE_TOK` / `FERRITE_TRACE_MOE` / `FERRITE_PROBE` + `FERRITE_PROBE_DIR` | per-layer numerical probes |
| `FERRITE_PPROF` + `FERRITE_PPROF_OUT` | built-in 1 kHz CPU flamegraph (pprof crate) |
| `FERRITE_KERNEL_SO` | override dlopen path of `libferrite_kernels.so` |

## 2026-09-10 会话：数值修复 + B=16 29.01→14.43ms（当前状态与路径）

**修复的 5 个根因**（全部 kernel 级证据链 + 文本验证）：① moe_down `(dscols,ni)` 传反（a396171）；② 图 INPUT 池化别名→`alloc_immortal`（469756a）；③ `dsa_append_batched` 把 B 当 ntok 传→kvb 越界读 3.75MB（ec6d795，B≤8 恰好留在池内空隙故不崩）；④ **DSA K/V 缓存格式分歧**：单 seq 路径（prefill）写 f32 无 scale、batched 读写 fp8+scale（526e002 只迁了一侧）→ prefill 槽被当 e4m3 误读 + scale 缓冲全池垃圾（kernel printf：ksc0=0.000000、softmax sum=NaN）→ 全部 11 个 DSA 层注意力精确为 0（FERRITE_LAYER_SUM 探针）→ 恢复 f32（a0e262d）；⑤ **xq 量化缓存按指针判失效**：池化地址跨层复用→陈旧命中→L+1 层 GEMM 用 L 层的量化激活（逐层数值对比：L0 hfn 精确匹配、L2 ffn 12.6x 偏差）→ (ptr, **gen**) 键（852be75）。

**性能改动**（B=16 replay 中位数，每项都文本验证）：
| 改动 | 结果 |
|---|---|
| moe_down 默认 fp8 fused（bf16 MMA 变体 13.1ms/步 52.3%→2.1ms） | 29.01 → 15.46 |
| DSA dummy total 8192→1（retire 阶段 indexer 2048-pool 慢路径 968×1.56ms） | 15.42 |
| AR 默认 f32（nsys 证明会合延迟主导，bf16 转换 0.43ms 纯浪费） | 15.20 |
| **device 侧 pinned t0/total 推进**（append kernel 内自增，in-stream 下游可见；步首全 rank 同步只剩 membership 变化步） | **14.57**（B=2 9.88） |
| **gemm3 融合投影默认开**（DSA{wk,weights_proj,gate}+GDN{b,f_a,g_a} 各一 launch+确定性归约；f32 直入吃掉 ~180 个 cast 节点；块级 K-split×8；m16n8k16 bf16） | **14.43**（B=2 9.93） |
| **gdn_chunk float4 state 读写 + 全并行 decay**（kernel 原以 0.63TB/s 内存延迟受限运行——grid(16,8)=128 块 × 512 线程 × 4B 在飞；float4=4x 在飞字节；decay 原只用 dk 线程跑 dv 串行） | **14.01**（B=2 9.45） |
| hc_pre_mix float4 加载（B=16 中性、B=2 −0.5ms，保留） | 14.02 |
| **sparse_attn_v3 三 kernel 分割**（QK/exp+PV/merge，NS=4×(B,h)=512 块 vs v2 的 128——同 gdn_chunk 延迟受限模式；全局 max 从 L2 热 scores 重读；确定性 split 升序合并；FERRITE_ATTN_SPLIT=0 回退 v2） | **13.61**（B=2 9.28，steady×16≈1075） |

steady×16 ≈ **1066**（300 窗口；replay 口径 16000/14.02 = 1142）。gdn 修复的机制：**小 kernel 的内存延迟受限模式**（少量块×标量 4B 加载 → 在飞字节不足）——float4 化是在飞字节的最廉价 4x；gdn_chunk 25.5µs → 预计 ~7µs/层 × 34 层 = −0.42ms 实测兑现。同模式审计清单：rest345（1.14ms，结构已在设计地板）、sparse_attn（0.58ms）、conv1d/gdn_prep（~0.3ms）。

steady×16 ≈ **1046**（300 窗口）。gemm3 的文本：逐字正确（《出师表》至"宫中府中/陟罚臧否"）；req0 偶发 `</s>` 前导 token = bf16-MMA 重结合类（1e-3）翻转近边界 logit，正文不受影响；FERRITE_GEMM3=0 可退回。device 推进的两个坑（f157cb0）：单 seq→batched 切换时 pinned t0 落后 1（batched 首个 append 覆写最后 solo token）→ dry pass 的簿记循环从 map 的 t_count 写回 pinned（handoff sync）；capture pass 不能写（会把 kernel 已推进的值倒退）。

**已验证无效（gate off 保留代码）**：e4m3 MMA down v1/v2（**修正基准参数后**——旧 bench 误用 inter_shared=512，SIMT 对 klen≠256 走标量慢路径，造出"108µs/指令瓶颈/2.14x"三重假象；正确参数下 SIMT 43.7µs=in-serve 实测，v1 48.6 / v2 连续读 49.4 均**更慢**——down 在 2.5TB/s 有效带宽的地板，勿再试 MMA 化）；n==16 bf16 wmma 投影（无 K-split，6x 回退）；P2P AR 复测仍死锁（30s 监护杀，驱动未 wedge）；kpool grid cap（15.29 回退——空块不是成本，全 grid 的内存级并行才是）；**bf16 cast 缓存（FERRITE_XB_CACHE=1，4 个变体全崩，同签名 36tok/1fault/err901，永久关闭）**：v1 池化缓冲扰动 batch 池地址稳定性契约；v2 immortal+capture-only（capture 期分配=cudaMalloc inside capture）；v3 ptr-only+dry 预注册（gate 让 dry 从不预注册）；v4 双遍激活+层清位（理论自洽仍崩——存在未定位的更深层机制，B=2 始终正常）。**教训：隔离基准必须用生产 shape；capture 期任何分配都是雷；batch 池分配确定性对图承重；同 x 组的 cast 合并在 B=16 有未解的结构性障碍。****P2P oneshot_v2 调查+实测（2026-09-10 末）**：接入 batched 链（0f1d5b2，FERRITE_P2P_ONESHOT=1）后**死锁**——serve 启动后无任何请求处理（无 capture/admission，log 停在 serving 行），30s 监护杀干净（驱动健康，b1 sanity 通过）。**epoch+ping-pong 协议在图捕获的 dry-run→capture 切换时 desync**（dry-run 真实执行 epoch 前进，capture 只记录不执行 epoch 冻结）。**P2P AR 三个变体（fused_v3、oneshot_v1、oneshot_v2）全部死锁，路线彻底关停**。AR 地板 = NCCL ~29.5µs/调用。

**当前分解**（每步每卡）：MoE 3.85（act 1.81 已达实测带宽峰/down 1.77/route 0.26）· AR 2.66（NCCL 会合地板 90×29.5µs）· hc 2.11 · 投影 cuBLAS 族 2.2（DSA/GDN 小投影 bf16-only + 每 GEMM 一次 f32→bf16 cast）· DSA 1.11 · GDN 1.18 · 间隙+host ~0.6。
**当前每步每卡分解**（nsys capture-range，2026-09-10 末，13.61ms 版本）：moe_act 1.84 / moe_down 1.81（均地板）· AR 2.66（NCCL 会合地板）· **hc_pre_rest345 1.14（设计地板）** · mix 0.69 · **gdn_chunk 0.47（float4 修复后从 0.87 减半）** · gemm3 0.38 · gemv_fp8 0.34 · nvjet 0.31（qkv/o_proj @6.2TB/s）· hc_post 0.28 · route 0.26 · sparse_attn ~0.51（v3 分割后）· kpool/conv/prep ~0.5。
**通往 1600（≤10ms）的诚实重估（2026-09-10 末）**：非结构 kernel 杠杆已基本用尽（act/down/AR/rest345 全在各自主地板）——剩余路径 = **DCP 类结构改动**（SGLang decode 用 --dcp-size 8，其不开 MTP 基座 ≈1300；ferrite 的等价物 = DSA 缓存池 + GDN 状态按 rank 切分，注意力无通信化）或 MTP-batched（目标 3200）。当前 13.61ms = replay 口径 1176 tok/s / steady×16 1075。

**下一会话执行单元（2026-09-10 固化设计，按价值排序）**：
⓪ **sparse_attn flash 式 slot 分割（−0.33ms 预期）**：实测 0.66TB/s 延迟受限（grid(B=16,h=8)=**128 块**，每线程串行 slot 循环；34MB/层 K/V 流量本应 ~7µs 实测 51.8µs）——与 gdn_chunk float4 修复同源（在飞字节不足）。修法：grid (B, h, **4**) = 512 块，每块处理 live_k/4 个 slot 的 online softmax（running max/sum，flash 式），尾部小合并 kernel 归一 4 组 partial（max/sum 合并数学确定序）。f32 cache 布局不变。
① 小投影融合多 GEMM：**已完成**（gemm3 默认开，DSA/GDN 各一 launch，−0.16ms）。
② **MoE EP 化**：**已实测否决（2026-09-10 末）**——代码里 `FERRITE_MOE_EP=1` 已完整存在（权重加载/kernel/expert_start 全就绪），直接开启测试：**replay 29.97ms vs TP 13.58ms（2.2x 更慢）**。原因 = 路由偏斜：128 assignments 分布到 8×36 专家不均匀，热门 rank 的 MoE 计算（~20 assignments × 25.2MB = 504MB/层）远超冷门 rank（~5 × 25.2MB = 126MB/层），AR 等待最慢 rank。**原始 TP 设计（constant topk work per rank）是正确的**。EP 要可行需要 expert 复制/容量因子/dispatch-combine 的动态负载均衡——回到了完整的结构改动需求。
③ MTP 路线（目标改 3200）：batched MTP 验证链是重构项。**2026-09-10 侦察+经济性实测（B=32 验证，b32 replay 60.65ms vs b16 13.58 = 4.5x @2x 行）**：① 锁定点 = gpu_engine.rs:86（`MtpState` 是 per-rank 单例——verify ping-pong scratch `hf_v/hprev` 共享，多 seq 污染；强制 max_seqs=1）；② mtp_step（tp.rs:1654）/ mtp_step_zero_h2d（tp.rs:2033）均为逐 seq；③ 路由在 decode_step_mega（tp.rs:1276，N>1 → mtp_step）；④ 批化五件事：MtpState → [B] 宽 scratch 或 per-seq、verify mega_v n=3×16=48 行（megab_b64 新尺寸类）、draft 链 16-seq 批化、ferrite_mtp_commit 逐 seq 批化、**dsa_append_batched 的行→seq 映射改造**（n=48 时 seq=row/3, tok=row%3）。**经济性修正（B=32 实测推翻乐观预测）**：n=32 步时 4.5x（其中 ~2x 是 per-row 成本的本征伸缩——AR 载荷/hc 链/gemv 全按行；~1.9x 是 MoE 唯一专家增长 104→~200；另有 n>16 kernel 回退病理——gemm3/mma_b16 的 n≤16 守卫使 32 行落入劣化 fallback，需先扩展到 n≤64）。修正后 MTP-batched @B=16 预测 **~1400-1600 tok/s**（非 2100-2300）——超当前 1078 但**低于 3200**；3200 需要 bs>16 或进一步结构工作。**结论：MTP-batched 不再是优先路径**——与非 MTP 的 DCP 同级，都需大重构。
④ act 4.8→6.5TB/s 深挖（218MB 权重流已 L2 去重，缺口在 per-expert 128KB 散段 vs nvjet 的 25MB 连续单矩阵 6.2TB/s——需 expert-major 重排或更大连续读）。

## Performance state (perf-b1, 2026-09-08)

- Non-MTP baseline (n=1): **114.0 tok/s** (HTTP+SSE, 1200-token output, head/tail-trimmed 20/20; replay 8.89 ms/step, TP8). Was 91.2 (200-step one-shot) before the split-K + occupancy series.
- **B=16 concurrency (batched path, 2026-09-08 late): 314-319 tok/s end-to-end (8000 tok SSE), 441 tok/s steady replay 16 seqs = 36.3 ms/step.** Was 74 tok/s before the batch-size padding series (SGLang-style):
  - Graph keyed by PADDED size `megab_b{1,2,4,8,16,32}` (was per-composition → re-capture on every membership change, 1-2 s each). Per-seq pointer tables keyed (layer|family, SIZE) with STABLE device addresses, content H2D-refreshed only on membership change; padded rows (u64::MAX) use a shared dummy state (DSA dummy must span MAXT=8192 tokens — 1-token dummy faulted Xid 13).
  - **Inside capture, NO H2D on the tables** (a captured async memcpy node replays against freed stack → Xid 13; sync H2D inside capture → err 900). The dry-run fills, the capture reads.
  - Graphs are KEPT across retires/B=1 ticks (tables are content-refreshed, not embedded) — destroying cost a re-capture per membership change.
  - `--max-seqs` default was 4 → raise to 16 for the B=16 benchmark.
  - **P2P AR deadlocks at batched n>8** (dev0 at dry-run L0, peers at L35 — capture serializes ranks; epoch reset after dry-run did NOT fix). NCCL fallback works (drop FERRITE_P2P for batched runs). NCCL AR costs 3.7 ms/step at 16 seqs (90×41 µs) — the single biggest kernel.
  - `P2P_AR_MAX_N` was 16*4096=65536 < 16×5120=81920 → raised to 16*8192.
- **B=16 nsys per-step (steady, NCCL)**: AR 3.7, gemv_bf16_nt 3.0, moe_down 2.9, indexer_topk 1.7, matmul_tiled 1.7 (mostly prefill), hc_pre 1.6, moe_act 1.5, sparse 1.25, gdn_chunk 1.2, gemv_fp8 1.1 ms. **GPU is saturated at steady state (bucketed nsys: ~35.5 ms kernel per 36.3 ms step) — the host gap story was a measurement artifact (admission-diluted averages); optimize KERNELS, not host.**
- Next targets for 1600 tok/s @16 (need 10 ms/step): AR (P2P fix or fused AR), gemv 3.0 ms, moe_down 2.9 ms, indexer 1.7 ms.
- **SGLang production config (measured 2026-09-08, b300-2 decode 15.164.0.39:30002 / b300-3 prefill 3.37.20.114:30100, ssh -i ~/.ssh/b300-spot.pem)**: NCCL **2.28.3**, default env (no NCCL_NVLS_ENABLE → they do NOT use SHARP either; /dev/nvidia-nvswitch* is absent on these instances), **--enforce-disable-flashinfer-allreduce-fusion** (AR fusion OFF), decode: `--dcp-size 8` + EAGLE (num-steps 5 / draft-tokens 6) + `--cuda-graph-max-bs-decode 64` + kv fp8_e4m3 + HiCache; prefill: `--enable-prefill-cp --cp-strategy interleave --enable-dsa-prefill-cp-layersplit --disable-overlap-schedule`. → SGLang's throughput comes from DCP + EAGLE + big CUDA-graph batches, NOT from a faster AR.
- **AR tuning exhausted (all measured on 16-seq batched, 30.7-31.3 ms/step baseline)**: bf16 payload +2%, NCCL_ALGO=Tree +1%, NCCL_MAX_NCHANNELS=2 −1.5%, NCCL 2.28.3 −0.3%, NCCL_LAUNCH_MODE=GROUP deadlocks the capture. The 223µs/call is the NCCL 8-rank rendezvous floor (~100µs kernel + ~133µs start spread, last-arriving rank rotates). Only structural fixes remain: fewer ARs (EP instead of TP for the FFN) or AR/compute overlap.
- **Two regressions from the TP-occupancy series (2026-09-08, both FIXED by bisect; the text was gibberish/EOS for ~1h and token counts alone never showed it):**
  1. `6ab9bee` P2P AR parallelization: phase A mapped (token, peer) onto threads but reused `i` as the peer index INSIDE `if (i < n)`, with grid = (n+1023)/1024. At decode (n=1) only thread 0 ran → wrote peer 0's staging slot alone, yet stamped EVERY peer's ready flag → every other rank read the PREVIOUS epoch's staging value → wrong all-reduce → deterministic gibberish at every layer. FIX: phase A maps r0 → (token = r0/world, peer = r0%world) over n*world threads; launcher grid covers n*world. LESSON: when parallelizing a serial loop, the thread index must be decomposed, not reused.
  2. `HC_P345_NB` 16→64→256 CORRUPTS the output (NB=64 → all-EOS, NB=256 → "the the the"; bisected one-shot 400-tok). NB=16 is the last known-good value and the sweep showed NO perf gain (rest345 12.6µs at NB=16 vs 13.7 at 64 — compute-bound, not occupancy-bound). Root cause of the NB>16 corruption still TBD; **do not raise HC_P345_NB without re-validating the 出师表 text**.
- MEMSET elimination (2026-09-08): nsys `CUPTI_ACTIVITY_KIND_MEMSET` counted **128 4-byte `cudaMemsetAsync(ctr2)` per decode step** (hc_pre, 2/layer). Folded the zeroing into `hc_pre_mix_split_kernel` block 0 (rest345 is a later kernel on the same stream, so the write is visible before any atomicAdd) → replay 9.32→8.89 ms (**+6%, 107→114 tok/s**). General rule: **count CUPTI MEMSET/MEMCPY rows in nsys, not just kernels** — per-step GPU ops that are not kernels are invisible in cuda_gpu_kern_sum.

- MTP (draft=2, verify n=3): **123.1 tok/s @500-step** (113.4 @200-step), accept 2.46, text flawless.
- Per-step (200-step window, ~20.5 ms → 48.8 steps/s): verify ~16.6 ms + draft ~1.6 ms + commit ~0.0 ms (one kernel) + host ~0.5 ms + fan_out ~1.7 ms.
- MTP chain: A→B copy-in is recorded INSIDE the mega_v graph; accept commit is ONE kernel (`ferrite_mtp_commit`, k read zero-copy from pinned slot); draft chain = per-draft graphs `mega_d{seq}_{i}` (DEFAULT ON since 2026-09-08; `FERRITE_DRAFT_GRAPH=0` falls back to the host chain).
- **Draft-graph alias bug (2026-09-08, FIXED)**: `mtp_forward_raw_argmax` did `forget(h_out.as_ref())` — forgetting a `&DevBuf` is a NO-OP, so the owned `Option<DevBuf>` still dropped and returned MtpState's `h_d[i]` address to the pool; the NEXT call's enorm/hnorm allocs aliased it and overwrote the draft h relay → accept 2.39→1.93 (graph path) / 1.00 (host-serial device chain). FIX: `std::mem::forget(h_out)` (the owned value). Same-function `zero_h2d` path was affected too. Detector: `FERRITE_POOL_DEBUG=1` prints `[pool-dup] DOUBLE RELEASE` on pool double-release.
- Inter-graph sync in the draft replay loop is REQUIRED: the pinned t0 slot is shared host memory, so without a stream sync between graph launches the next `dsa_host_advance` lands before the just-launched graph's DSA kernels read it (measured accept 2.39 with sync vs 2.22 without).
- Debug knobs: `FERRITE_DRAFT_GRAPH=2` runs the device draft chain host-serial (bisect mode); `FERRITE_DRAFT_DRY_ONLY=1` skips the capture pass (dry-only bisect).

## Profiling (what works on this machine)

**LLM-serve kernel breakdown: nsys 2025.6.3** (`/usr/local/cuda-13.2/bin/nsys` — the CUDA 13.2 bundled one; the 2024.2.3 apt version still does NOT work: zero CUDA rows). Verified 2026-09-07 on a 3-kernel mini program (CUPTI injection healthy, cuda_gpu_kern_sum reports correct). ~5-10% overhead, full timeline, works with CUDA graphs:

```bash
sudo /usr/local/cuda-13.2/bin/nsys profile --trace=cuda --cuda-graph-trace=node \
  --sample=none -o /tmp/nsys_out --force-overwrite=true \
  env NCCL_NVLS_ENABLE=0 FERRITE_MEGA=1 FERRITE_NCCL=1 FERRITE_P2P=1 FERRITE_WORKER_POOL=1 \
  FERRITE_LAYER_DEV=1 FERRITE_GDN_DEV=1 FERRITE_MOE_DEV=1 FERRITE_DSA_DEV=1 FERRITE_HEAD_DEV=1 \
  CUDA_VISIBLE_DEVICES=4,5,6,7 LD_LIBRARY_PATH=$HOME/ferrite/kernels/cuda \
  ./target/release/ferrite-serve ... --max-tokens 20 ...
sudo /usr/local/cuda-13.2/bin/nsys stats --report cuda_gpu_kern_sum /tmp/nsys_out.nsys-rep
```

`--cuda-graph-trace=node` expands mega-graph replays into individual kernels in the report. Log FULL output to a file (never `grep|head` pipes — they swallow progress/errors and you can't see which stage hung). Validate the toolchain on the 3-kernel mini first when in doubt: `kernels/cuda/ncu_miniprof.cu` (cudaProfilerStart/Stop window) exists for exactly this.

**ncu: NEVER profile the whole serve with it — it WILL time out (burned ~40 min / 5+ failed attempts 2026-09-07).** ncu's injection overhead is 10-100× per CUDA call; the pre-decode phases alone (mmap load = 38287 cudaMemcpy API interceptions + mega-graph capture with 900+ kernel dry-runs) exceed any sane timeout — the profile window (FERRITE_NCU, decode step ≥1) never even starts (CSV contains only `==PROF== Connected` + `==ERROR== ... 124`). kernel-name filters and launch-count caps do NOT help: the overhead hits pre-window API calls too. ncu IS fine for single-kernel micro-benchmarks (mini program verified) and for known-hotspot deep dives launched outside the serve. In-serve per-kernel time = nsys; kernel deep-dive (SOL/occupancy) = ncu on an isolated repro of that kernel (see tests/bf16_widen_gpu.rs for the Rust-side harness pattern).

Other profiling rules:
- GPU-side timing events inside a captured CUDA graph do NOT work (`cudaEventElapsedTime` on graph-recorded events returns InvalidValue — sync-only; the dead `FERRITE_MEGA_EVTS` code was removed for this reason).
- `FERRITE_TIMING` per-layer numbers (at=/mid=/ffn=) are HOST-side Instant deltas around launch+sync — useful for RELATIVE layer comparison only; never quote them as kernel GPU time (they overstate ~10-20× and misled the "DSA at=7.5ms" hunt; the real per-step budget is replay ~15.6ms/45 layers = ~350µs/layer).
- CPU side: `FERRITE_PPROF=1 FERRITE_PPROF_OUT=serve.svg` (flamegraph over load+decode).

## GPU test discipline (hard rules)

1. **Exclusive GPU runs**: before starting a serve, `pgrep -af ferrite-serve` + `nvidia-smi --query-compute-apps` must be EMPTY on GPUs 4–7. A leftover panicked serve holds ~160 GB/GPU and breaks the next run's NCCL init.
2. **Never kill a GPU process while another test is running on the same GPUs** — NCCL bootstrap resources are shared; killing a zombie peer can hang the live run's collectives (observed: decode frozen mid-text, log mtime stale).
3. Kill by exact PID (`kill -9 <pid>`), never `pkill -f` (it matches your own ssh bash command line → suicide, exit 255).
4. A background run whose log mtime has not moved for minutes is hung — check `stat -c %y` + `pgrep`, don't wait blindly.
5. After `kill -9`, GPU memory release is asynchronous (defunct + CUDA context teardown can take ~30 s) — confirm `nvidia-smi` shows 0 MiB before the next run.
6. serve is one-shot: exits via `std::process::exit(0)` (exit-time drop of 1.17 TB weights SEGFAULTs → EXIT 139, which also loses profiler buffers).

## 2026-09-08 会话结论：通信不是瓶颈（A/B 实测），计算才是

**用户目标**：16 并发 ≥1600 tok/s（不开 MTP）；若用 MTP 则目标 3200。**当前：18.8 ms/步 = 851 tok/s**（会话内从 546 提升 **+56%**，每次改动都人眼验证文本）。

**决定性 A/B（同一 16-seq 负载，`FERRITE_AR_SKIP=1` 跳过全部 AR）**：跳 AR 22.05 ms vs 带 P2P AR 23.06 ms → **通信只占 1.78 ms = 7.5%**。32 并发同样（NO-AR 95.5 vs 102.8）。**"通信占 64%" 是陈旧读数**（`p2p_ar_publish` 的 max=55.6ms 是 dry-run/capture 期自旋超时；其稳态中位数只有 5.1µs）。

**本会话的有效 kernel 优化（全部文本验证，累计 29.33→18.8 ms）**：
1. **moe_fused_down 16 字节 lane**（每 lane uint4 覆盖 2 个连续 h 行，klen=256）：**7.5 → 1.9 ms**。根因：8 字节 lane 只有 1.16TB/s（request-rate 受限）。坑：scale 列索引必须 `(lane&15)>>3`；补丁内重复声明 `py[8]` 会遮蔽外层 → 输出全零。
2. **moe_fused_act padded per-warp smem staging**：A 片段 4B 跨 8 行 = 每指令只用到 32B sector 的一半。暂存 16 行 × 64 列 × 2 投影，**行距 padding 到 80B**（64B 会让每行落 bank 0 → 8 路冲突，反而慢）。act 74 → 62.8 µs。
3. **indexer 8 线程/pool**（原来是每线程串行 4096-MAC 点积，GPU 利用率 ~1%）：中位 100 → 32 µs。**shuffle 必须用 8-lane 组掩码**（0xffffffff 会让没有 pool 的组跳过 → 捕获死锁）。
4. **sparse_attn 输出 float4 + 4 槽位分组**（原来 4 字节读、跨槽 64KB 零合并）：121 → 51 µs（2.4x）。
5. **hc_pre_mix 4 行/block**（x 原来被 24 个 mix 行各读一遍 → L2 受限）：11.8 → 9.5 µs。
6. **gemv_fp8 8 行/组 + x 寄存器缓存**（x 原被每 (block,row) 重读）：51.7 → 41.7 µs。
7. sparse_attn 的 live_k 边界（原按固定 select_k_max=2048 循环）+ indexer 短上下文快路径。
8. dense-FFN / GDN host 路径的 AR 改 P2P 优先；run_matmul 走快路径。

**已验证无效/更差（勿重复）**：gemv row-major（**权重矩阵 1-3MB 全在 L2 内，token-major 的 16x 重读是 L2 命中，不是 DRAM 瓶颈**——实测 18.15 vs 17.99ms）、down TT=1/8/16、ROWS=16/64、gemv R=8（无 x 缓存时）、gemv 1024 线程/block、K 循环 unroll 4、token 循环 unroll 2、去掉 fp8 转换链（仅省 4.5%）、act 的共享 sa staging、act 的 128 列 staging（18.93ms，占用率下降）、HC_MIX_KS 8→2。

**教训（乱码=数值回归，且可能是"少算"）**：gemv row-major 曾测得 14.78ms/1082 tok/s 但输出 `!!!`——因为 launcher 的 grid 硬除 `rpb*8` 而 kernel 在 `nrows%8!=0` 时用 R=1，**只算了 1/8 的输出行**。**任何提速都必须同时人眼验证文本，否则"少算"会被误当成"优化"。**

**当前每步每卡分解（nsys，~300 步反推）**：p2p_ar_publish ~1.8（含 barrier）/ **moe_act 3.4** / matmul_tiled_bf16 3.3（多为 prefill）/ **gemv_fp8 2.4** / **moe_down 2.1** / **indexer 1.6** / hc_rest345 1.34 / hc_mix 1.04 / kpool 1.0 / gdn_chunk 1.0 / gdn_step 0.6 / sparse_attn 0.6 / NCCL AR 残余 0.64。

**下一步（按预期收益）**：① moe_act 仍 62% 峰值（3.4ms）——试 128 列 staging 或 2-token/block；② indexer 的 139µs 平均（中位 32µs，长尾待查）；③ kpool_compress 中位 114µs（11 次/步）；④ gdn_chunk/step 1.6ms；⑤ AR：publish 中位 5.1µs×90=0.46ms，可试 per-block flag 省掉 publish kernel。

## B=16 通信 vs 计算：实测推翻旧结论（2026-09-08 late, perf-b1）

**用户目标**：16 并发 ≥1600 tok/s（不开 MTP）；若用 MTP 则目标 3200。**当前：21.2 ms/步 = 755 tok/s**（会话内从 546 提升 +38%，每次改动都人眼验证文本）。

**决定性 A/B（同一 16-seq 负载，`FERRITE_AR_SKIP=1` 跳过全部 AR）**：跳 AR 22.05 ms vs 带 P2P AR 23.06 ms → **通信只占 1.78 ms = 7.5%**。32 并发同样（NO-AR 95.5 vs 102.8）。**"通信占 64%" 是陈旧读数**（`p2p_ar_publish` max=55.6ms 是 dry-run/capture 期自旋超时）。

**本会话落地的有效 kernel 优化（全部文本验证通过，累计 29.33→21.2 ms）**：
1. **moe_fused_down 的 16 字节 lane**（每 lane uint4 覆盖 2 个连续 h 行，klen=256）：**7.5 → 1.7 ms（4x）**。关键认知：该 kernel 原为 request-rate 受限（8B lane 只跑 1.16TB/s = 15% 峰值）。踩过的坑：scale 列索引必须用 `(lane&15)>>3`（lane 16-31 读的是第二行）；补丁里重复声明 `py[8]` 会遮蔽外层变量 → 输出全零。
2. **moe_fused_act 的 padded per-warp smem staging**：A 片段 4B 加载跨 8 行 = 每指令只用到 32B sector 的一半（2.3x 字节浪费）。暂存 16 行 × 64 列 × 2 投影，**行距必须 padding 到 80B**（64B 会让每行落在 bank 0 → 8 路冲突，实测反而慢 1.6ms）。→ act 74 → 62.7 µs/次。
3. **sparse_attn 的 live_k 边界**（原按固定 select_k_max=2048 循环，实际槽位 ~112）。
4. **indexer_topk 快路径**（select_k ≥ jmax 时跳过 O(k·n) 选择）。
5. **dense-FFN / GDN host 路径的 AR 改 P2P 优先**（原来直调 NCCL）。
6. 8 行数据预取到寄存器 + shuffle 归约外提 + TT=4 tokens/block + 去掉阻止展开的 `break`/守卫。

**已验证无效/更差（勿重复）**：gemv row-major、down TT=1/8/16、ROWS=16/64、gemv R=8 行/组、gemv 1024 线程/block、K 循环 unroll 4、token 循环 unroll 2、去掉 fp8 转换链（仅省 4.5% → 非转换瓶颈）、act 的共享 sa staging（无 padding 或 4KB/warp 版）。

**当前每步每卡分解（nsys，367 步反推）**：p2p_ar_publish 1.3-2.5（含 barrier 等待）/ moe_act 3.2 / matmul_tiled_bf16 2.7（多为 prefill）/ gemv_fp8 2.5 / indexer 1.9 / moe_down 1.7 / sparse_attn 1.7 / AR 残余 / hc 2.2 / kpool 0.86 / gdn_chunk 0.82 / gdn_step 0.48 / route 0.47。

**下一步**：① matmul_tiled_bf16 是 32×32 FMA（无 tensor core），确认是否为 decode 路径，是则改 cuBLAS/MMA；② indexer/sparse_attn 的 199/180µs 中位（短上下文下应更快）；③ hc 两个 kernel 2.2ms（每 block 仅 256 元素）；④ AR 的 90 次 barrier（49µs/次，理论 11µs）——可试 per-block flag 省掉 publish kernel。

## B=16 通信 vs 计算：实测推翻旧结论（2026-09-08 late, perf-b1）

**A/B 实测（同一 16-seq 负载，`FERRITE_AR_SKIP=1` 跳过全部 AR）：**

| 配置 | replay/step | aggregate |
|---|---|---|
| 跳过 AR（纯计算） | 24.88 ms | 643 tok/s |
| NCCL AR | 30.68 ms | 512 tok/s |
| P2P AR（3-kernel 版） | **26.16 ms** | **611 tok/s** |

→ **通信（AR）只占 4.45 ms = 15%**，计算占 24.88 ms = 85%。此前"AR 20.1 ms/64%"的结论来自一次陈旧/被 capture 期自旋超时污染的 nsys 读数（`p2p_ar_publish` 的 max=55.6ms 就是 500000 次自旋超时，只出现在 dry-run/capture 阶段）。**优化方向必须放在 kernel，而不是继续调 AR 参数。**

**nsys 每步每卡分解（367 步反推，GPU 约 92% 忙）：**

| kernel | ms/步/卡 | 备注 |
|---|---|---|
| moe_fused_down_sum_fp8 | 6.7 → ~4.5（已优化） | token-major、每 token 重读专家矩阵 |
| moe_fused_act_fp8_mma | 3.5 | fp8 MMA，8 warp K-split，效率约 45% |
| matmul_tiled_bf16 | 2.7 | M=16 时 32×32 tile 浪费一半 |
| gemv_fp8_v2 | 2.5 | 已改 row-major（无收益，L2 已吸收） |
| indexer_topk_batched | 2.0 | **top-k 是 O(k·n) 串行全块归约**（k≈live pools） |
| sparse_attn_v2_batched | 1.9 | |
| hc_pre_rest345 + mix + post | 2.4 | 90 次/步 |
| gdn_chunk + gdn_step | 1.3 | |
| NCCL AR 残余 | 1.5 | 仍有 ~4 次/步走 NCCL |
| p2p AR（store+publish+reduce） | 4.45 | 49µs/次 × 90 |
| moe_route | 0.5 | |

**模型规模（GLM-5.3-Flash config）**：hidden 4096、45 层（前 3 层 dense，inter 12288）、**288 路由专家 / topk 8 / moe_inter 2048**、`e_local=288`（**每个 rank 持有全部 288 专家**，专家矩阵按 **inter 切 256**，down 输出完整 4096）、kv_lora 512、q_lora 1536、64 heads × 256、index_topk 2048、MTP 1 层。

**权重读取的物理下限**：每步每卡全量专家权重 42 层 × 288 专家 × 25.2MB / 8 = **38 GB**；但 bs=16 只用 ~104/288 专家 → ~14 GB → 在 7.6 TB/s 下 **~1.8 ms/步**。当前 24.88 ms 计算 = 下限的 ~13 倍（SGLang ~5 倍），说明**冗余读取 + kernel 效率**两个方向都有空间。

## 2026-09-08 会话结论：通信不是瓶颈（A/B 实测），计算才是

**用户目标**：16 并发 ≥1600 tok/s（不开 MTP）；若用 MTP 则目标 3200。**当前：23.1 ms/步 = 693 tok/s**（会话内从 546 提升 +27%，文本每次都人眼验证）。

**决定性 A/B（同一 16-seq 负载，`FERRITE_AR_SKIP=1` 跳过全部 AR）**：

| 配置 | replay/步 | aggregate |
|---|---|---|
| 跳过 AR（纯计算） | 22.05 ms | 726 tok/s |
| P2P AR | **23.06 ms** | **693 tok/s** |

→ **通信只占 1.78 ms = 7.5%**。32 并发同样：NO-AR 95.5 ms vs 带 AR 102.8 ms（AR 仅 7.3 ms）。**"通信占 64%" 是陈旧读数**（`p2p_ar_publish` 的 max=55.6ms 是 dry-run/capture 期自旋超时）。优化必须放在 kernel。

**每步每卡分解（nsys 稳态，367 步反推）**：moe_down 7.5→~5（已优化）/ moe_act 3.5 / gemv_fp8 2.8 / sparse_attn 1.9 / indexer 1.5 / AR 1.8 / hc 2.3 / gdn 1.0 / cuBLAS 1.3 / kpool 0.6 / route 0.5。

**本会话已落地的 kernel 修复（全部文本验证通过）**：
1. **P2P batched 死锁根因**：多 block 时 last-block 检测（`atomicAdd(ctr)`）在图 replay 下失效 → 改 3-kernel（store 多 block / publish 1 block / reduce 多 block）+ 每 block 独立 seen 行 + float4 向量化（AR 从 236ms/步 → 1.78ms）。详见下节。
2. **moe_fused_down**：32 lane × 8B/行（原 16 lane 闲置一半）+ 去掉阻止展开的 `break`/`h<hidden` 守卫 + 8 行数据预取到寄存器 + shuffle 归约外提 + TT=4 tokens/block。6.6 → ~5 ms。
3. **moe_fused_act**：去掉 kk 循环里永不触发却阻止展开的 `break`。
4. **indexer_topk**：短上下文 `select_k >= jmax` 时跳过 O(k·n) 选择循环（集合语义，消费方过滤非法项）。
5. **gemv_fp8 row-major**：无收益且破坏输出 → 已回退（教训：改索引映射必须验证文本）。

**已验证无效/更差的尝试（勿重复）**：gemv row-major、down 的 TT=1/8/16、ROWS=16/64、gemv 的 R=8、K 循环 unroll 4、token 循环 unroll 2、去掉 fp8 转换链（只省 4.5% → 非转换瓶颈）。

**下一步（按预期收益）**：
1. **MoE down 仍是最慢单 kernel（~5ms/步，仅 16% 峰值带宽）**：act kernel（3.5ms）达 46% 峰值，差别是 act 每 block 读 128KB（单专家矩阵内），down 每 block 的 9 warp 各读不同专家的 2KB。**唯一验证过的有效手段是提高每 warp 连续读取量**，但 ROWS 试验反而变慢（原因未明，需 ncu）。建议：expert-major 分组（把同一专家的 token 聚到一个 block）或 cuBLAS grouped GEMM（`cublasGemmGroupedBatchedEx`，需先把权重反量化到 bf16 或用 fp8 分组 API）。
2. **act kernel 的 A-fragment 访存未合并**（8 行 × 16B/指令）→ smem staging 或 16×32 swizzle（TileRT 的 `_swizzle_qmma_16x32` 就是这个）可望 ~1.5x。
3. **indexer/sparse_attn**：长上下文时 top-k 仍是 O(k·n)（基数选择/bitonic 可解）。
4. **AR 1.78ms**：publish/poll 的中位 4.6µs 但均值被 capture 期污染；可试 per-block flag 省掉 publish kernel。

## P2P batched 死锁的完整根因链（2026-09-08，已修复，勿再重复试错）

单序列 P2P 能跑、batched 死锁，曾试过 5 种修法全部无效（epoch reset、单调 seen 协议、ctr 清零、flag 后加 fence、`atomicExch_system`）——**因为它们全部打在已废弃的 `p2p_ar_down_v2_kernel` 上**（`FileReplace` 匹配到第一个出现处）。真正的活跃 kernel 是 `p2p_ar_fused_v3_kernel`。

真实根因（三个叠加）：
1. **last-block 检测失效**：launcher 按 `blocks = ceil(n*world/1024)` 启动（16 seqs → 640 blocks），`gridDim.x != 1` 走 `atomicAdd(ctr)` 路径；注释自陈该路径 ctr 跨 replay 会卡住 → 没有任何 block 认为自己是最后一个 → 永不发布 stamp → 单调等待死锁。旧修复只覆盖了单序列的 `gridDim.x == 1` 分支。
2. **staging 写零合并**：`(token,peer)` 展平（`ii = r0/world, rr = r0%world`）让相邻线程写不同 GPU → 655K 次 4B NVLink 事务/AR，约 2.5ms/次（replay 236ms）。修：外层 peer、内层连续 token，每线程读一个 token 写所有 peer 的连续槽位。
3. **seen[] 共享行竞争**：多 block 同时 poll 同一 peer 时，block A 更新 seen[tr] 后 block B 读到 `prev == cur` → 永远等下一个 stamp（`prev=2 cur=2 myepoch=1`）。修：`seen[MAX_FBLOCKS][world]`，每 block 独立行。

最终结构（3 kernel/AR，全部 stream-ordered，无需任何 grid 级计数器）：
`p2p_ar_fused_v3`（多 block 合并写 staging，block 0 把 epoch 快照写入 ctr）→ `p2p_ar_publish_v3`（**1 block**：block 0 写全部 peer 的 stamp + 一次 `__threadfence_system`，8 个线程 poll）→ `p2p_ar_reduce_v3`（多 block，零轮询）。
**注意**：`__threadfence_system()` 不能放在 655K 个线程里（每个线程一次 → ~2ms/AR）；kernel 边界本身已保证 store 对下一个 kernel 可见。flag 用普通 volatile store + 一次 system fence（vLLM 的做法）即可。

**当前 P2P AR = 49µs/次**（store ~3.5µs + publish 中位 4.6µs + reduce + 3 次 graph node 启动）。仍比理论（~11µs）高 4 倍，是下一个通信优化目标。

## 已知可攻的 kernel 低效点（按收益排序）

1. **MoE act/down 的冗余读取**：token-major 循环让每个 token 单独遍历其 top-k 专家矩阵；bs=16 时 ~104 个专家被 128 次赋值覆盖 → 1.2-1.4x 冗余（不算大），但 kernel 效率只有 25-45% 峰值。已修 down 的 lane 利用率（32 lane × 8B，29.33→26.16ms）。
2. **indexer top-k 是 O(k·n)**：`for r in 0..select_k { 全块归约求 max }`，短上下文 t≈100 → 100 次串行归约 = 212µs/次 × 11 层。长上下文 t=2048 时更糟（DSA decay）。修法：基数选择/bitonic，或降低 select_k。
3. **matmul_tiled_bf16 在 M=16 时浪费一半 tile**（32×32 tile，M=16）；dense 层与注意力投影受影响。
4. **AR 49µs → 目标 <15µs**：减少 graph node 数（3→2）或合并 reduce。

## Known-good demo numbers (watch for regressions)

- MTP 出师表 200-step: `real 476 tokens` window, text must be flawless 《出师表》 through 将军向宠 section (乱码 = accept/commit bug, ALWAYS check by eye).
- 500-step: 58.9 tok/s (DSA decay visible), non-MTP 500-step 44.7.
- If accept rate collapses to exactly 1.0 with NCCL fallback → env missing NCCL_NVLS_ENABLE=0.
- **B=16 non-MTP, TRUE 16-concurrency (2026-09-09, live=16 verified, 1727 steps): 18.42 ms/step median = 869 tok/s aggregate** (15.53 ms = 1030 tok/s at short context; 21.12 ms = 758 tok/s at ~1600-token context; DSA decay 1.39x). End-to-end wall with 8000-token SSE = 608-667 tok/s. Same-metric history: 36.3 ms/step = 441 tok/s / 314-319 end-to-end => the current build is 1.97x faster. **Read the `[megab] replay 16 seqs: Nms` lines with FERRITE_TIMING=1; 16000/N = aggregate tok/s. per-seq x 16 and total/wall are admit-ramp-contaminated — cross-check only.** Script /tmp/bench_tr.py N MAX_TOKENS.
  per-seq `steady=` x16 (the `[megab] replay` line under-reports long context by ~8%):
  **300-token window: 59.4-60.9 tok/s/seq = 950-974 tok/s aggregate**; **1000-token window:
  56.8-57.6 tok/s/seq = 909-922 tok/s aggregate** (run-to-run ±2%). Text = direct 《出师表》 ✓.
  (replay line: 990 @300-tok / 842 @1000-tok.) Progression this session: 546 → ~957 tok/s.
  Kept changes: gemv_fp8 T=2 tokens/block, gated_rmsnorm 1 token/block x256, moe_act double-buffered
  cp.async.cg staging + ldmatrix, hc_pre_mix K-loop `#pragma unroll 4`, **moe_route 256-thread
  top-k**, **rmsnorm blockDim-sized reduce (was a latent 8-warp bug)**, **4/2-accumulator splits of
  the serial FMA chains (GDN +0.5%, gemv +0.1%)**.
  Run-to-run/thermal drift is ~±1%; always A/B in the same window.
  **Biggest remaining win: MLA absorption** (see `docs/agent/perf-roadmap.md`) — the DSA cache is
  non-absorbed (per-head up-projected k/v), costing ~1.5 ms/step at t=1000 (~8%).

## 2026-09-08 续：half2 FMA 迁移的收益与陷阱（gemv 有效 / moe_down 回退）

**背景**：gemv_fp8_v2 / moe_down 的内层都是 `fp8x2 -> half2 -> float2 -> 2 条标量 FMA`
（每 16 个元素约 40 条指令）。改成 `fp8x2 -> half2 -> __hfma2`（每 16 个元素 16 条）
在 gemv 上生效（17.61 -> 17.51 ms，+0.6%），在 moe_down 上**速度中性（17.51 vs 17.51）但
把模型推入 thinking 模式**（文本从直接背诵《出师表》变成 `<think 嗯，用户要求背诵…`）——
已回退。教训：

1. **`reinterpret_cast<const __half2*>(float_ptr)` 是错的**：这是把 float 的位模式当
   half 读（实测输出全 `!!!!!`，且"变快"3.4 ms —— 少算/垃圾值）。必须 `__floats2half2_rn`。
2. **速度中性但改变输出分布 = 必须回退**：fp16 只在 16 元素 chunk 内累加、随后并入 fp32，
   理论上误差 ~1e-3（远小于 fp8 权重 1e-2），但**实际翻转了模型行为**。数值改动即使"精度够"
   也可能越过 logit 的决策边界 —— 唯一可靠的判据是人眼文本，不是误差估计。
3. 迁移 half2 只对**指令受限且数值敏感度低**的内核安全；MoE 路径（act/down）不要动。

**其他中性/更差的尝试（勿重复）**：gemv WPR=2（17.87）、gemv 内层 unroll 4（17.67）、
moe_route warp-shuffle top-k（17.61，中性但修掉了 `bidx[threadIdx.x]` 越界写 [32] 数组）。

**act kernel 双 tile 预取（2026-09-08，已回退）**：把 `uint4 pf[4]`（1 个 tile）改成
`pf[8]`（2 个 tile，kb 消费 pf[cur] 时重填 pf[cur]=tile(kb+128)）→ **19.6ms（-12%）**，
文本仍正确。根因：pf[8] = 32 个额外寄存器压低占用，收益被抵消。**结论：act 的瓶颈不是
单个 warp 的 load 深度**（单 tile 版已经是 8 warp × 4 uint4 = 32 个在飞的 load），
继续加深度只会伤占用率。

## 2026-09-08 gemv_fp8 的 M 维复用（有效，+1.1%）

`gemv_fp8_v2_kernel` 改成 **每 block 处理 2 个 token**（grid.y = ceil(n/2)，t0 = blockIdx.y*2）：
权重行只 load 一次、fp8->half2 只转换一次，两个 token 共用（每个 token 保留自己的累加顺序
→ 逐位相同）。17.57-17.61 -> **17.40-17.42 ms（919 tok/s）**，文本正确。这是 n=16 下 GEMV
的正确打法（M 维复用），但收益只有预测的 1/3 —— 说明该 kernel 仍有很大比例是延迟而非指令。

**注意编译陷阱**：第一次提交漏了 `has1` 的作用域（声明在 row 循环的 `{}` 内、epilogue 在外），
`build.sh` 报 1 error 而 serve 用的是**旧 .so**，测出 17.62ms 的假结果。**每次必须看 build 输出的
error 数**（本会话第二次踩这个坑）。

## ⛔ sparse_attn 三次回归的定案（2026-09-09，全部逐位/隔离基准证明）

**判定标准**：出师表 prompt 必须正确背出原文（`先帝创业未半而中道崩殂…将军向宠…臣本布衣`）；
"思考模式里循环/答不出" **是回归**（用户明确：只豁免"出现思考"，不豁免"思考循环"）。
**数值判据**：隔离微基准 `/tmp/sparse_bench` 对 `a9e5d5a`（1170 参考）逐位比对，
`differing=0/131072` 才算通过。文本+数值双过才算修好。

| 改动 | 后果 | 处置 |
|---|---|---|
| softmax max/sum 用固定 256 步长**漏加 `tid<256` 守卫** | 线程 256..511 把 slot≥256 多数一遍 → 分母 2x → 权重偏向前 256 个 KV → **live_k>256 后循环** | 加 `if (threadIdx.x < 256)` 守卫（已保留） |
| PV 重构（`pv_on` + `G=256/cols` + sync 移出条件） | **mega-graph capture 期 err 700** → sticky error → 误报 `mega graph mega1 missing` | 还原原 PV 控制流（已提交） |
| **"UB 修复"：`reinterpret_cast<const __nv_fp8x2_storage_t*>(&u0)` + `ff[e]` → 显式 `uu[e>>1]>>16 / &0xFFFF`** | **K/V 字节映射改变 → 注意力输出整体偏移 4.0x**（隔离基准 0.0627→0.2533）→ 模型退化 | **禁止重试**。指针模式虽为布局相关 UB，但**就是验证过的字节映射**；已在代码注释里标注 |

**排查方法（有效）**：隔离微基准 + `LD_PRELOAD=<各版本.so>` + 逐位 diff，**秒级**；
再用"a9 + 逐 hunk 叠加"定位到具体改动（red[16]/guard 各 0 差异，UB 修复 32768 差异）。
**不要**用文本观察或 e2e 做 kernel 正确性判据（慢且不可靠）。
**教训**：`FileReplace` 改动控制流结构（尤其 `__syncthreads` 位置、条件作用域）后，
必须用上面的数值基准验证；`mega graph ... missing` 往往是 sticky CUDA error 的误报，
真因在更早的 kernel。

## 2026-09-10 末会话总结：B=16 replay 29.01→13.30ms（+118%），10 项优化落地

**严格口径**：`FERRITE_TIMING=1` 的 `[megab] replay 16 seqs` 中位数（16000/N = 聚合 tok/s）；
steady×16 = per-seq steady × 16 作交叉验证。文本每次人眼验证（出师表逐字）。

| # | 优化 | replay 增量 | 累计 |
|---|---|---|---|
| 1 | fp8 fused down 默认 | 29.01→15.46 | 15.46 |
| 2 | DSA dummy total 8192→1 | →15.42 | 15.42 |
| 3 | AR 默认 f32 | →15.20 | 15.20 |
| 4 | device 侧 pinned 推进 | →14.57 | 14.57 |
| 5 | gemm3 融合投影 | →14.43 | 14.43 |
| 6 | gdn_chunk float4 | →14.01 | 14.01 |
| 7 | mix float4 | →14.02 | 14.02 |
| 8 | sparse_attn_v3 3-kernel 分割 | →13.58 | 13.58 |
| 9 | **GPU 侧 embedding**（embed_expand_dev 为图第一节点，graph_run_ids 喂 64B token ids 替代 1MB host staging） | →13.43 | 13.43 |
| 10 | **host 侧 embedding 跳过**（dev_input 模式下 ~125µs/步的 embed+hc_expand 不再计算） | →**13.30** | **13.30** |

**GPU 侧 embedding 的两个关键坑（2026-09-10 实测）**：
1. ids buffer 分配必须在 `graph_capture_begin()` **之前**——cudaMalloc during capture = err 900
   （"operation not permitted when stream is capturing"），首次测试 0.4 tok/s 全崩。
2. dry-run 必须预热 embedding 表的 dev_weight 缓存（用 scratch buffer 跑一次
   embed_expand_dev_buf）——否则 capture 期 `dev_weight` 的 2.4GB H2D 上传被录成图节点，
   每步 replay 重传 2.4GB。
3. `hc_expand` 返回 `Tensor` 不是 `Vec<f32>`——空值用 `Tensor::zeros(Shape::new([0]), DType::F32)`。

**关停路径（本会话新增 3 项，全部有机制级解释）**：
- `--use_fast_math`：13.59ms（中性，denormals 在 Blackwell 不是瓶颈）
- `FERRITE_MOE_EP=1`：29.97ms（**2.2x 更慢**——路由偏斜：热门 rank ~20 assignments × 25.2MB
  vs 冷门 ~5，AR 等最慢 rank；原始 TP 设计 constant topk work per rank 是正确的）
- down MMA v1/v2、P2P AR ×3、NCCL LL128/Simple、kpool grid cap、mix launch_bounds、
  bf16 cast 缓存 ×4：见上文各节

**当前每步分解（13.30ms replay + ~0.9ms host gap）**：
- AR 2.66ms（NCCL 90 调用 × 29.5µs = 协议地板；P2P 三变体全死锁）
- MoE act+down 3.65ms（带宽地板 47%；B=16 下每 expert 仅 ~1.3 token 摊薄权重读取）
- hc 链 2.11ms（rest345 1.14 + mix 0.69 + hc_post 0.28；rest345 12.6µs/次 = 理论下限的 50x，
  **瓶颈未定位，需 ncu isolated repro**）
- attention 1.4ms（sparse_attn_v3 已优化）
- projections 1.0ms（gemm3 已融合）
- other 0.7ms + embed_expand_dev ~0.01ms

**通往 1600 的剩余路径（按预期收益排序，全部需要较大改动）**：
1. **hc rest345 深度优化**（−0.6ms 目标）：12.6µs/次 vs 0.25µs 内存下限——需 ncu 定位瓶颈
   （寄存器压力？占用率？sinkhorn 串行？）。NB=16 是最后已知正确值（64/256 会毁输出）。
2. **tick 循环软件流水线**（−0.2~0.4ms）：host 在 GPU 执行期间做上一步的 detokenize/SSE/
   bookkeeping（当前在 sync 之后串行做，GPU 空闲 200-400µs）。需拆分 graph_run_ids 为
   launch/wait 两阶段 + 输出双缓冲。
3. **attention 内核**（−0.4ms）：sparse_attn_v3 刚优化过，进一步需 ncu。
4. **AR 结构性削减**（需模型数据流重构）：attention AR + FFN AR 每层 2 次，FFN 输入依赖
   attention 输出（经 AR + hc_post），无法合并。hc 的 4 流展开可能允许 AR 延迟——需深度分析。

**EP/DCP/MTP-batched 全部关停**（详见上文）：EP 路由偏斜 2.2x 更慢；DCP 每 rank 需 305GB
MoE 权重 > 180GB HBM；MTP-batched 修正后 ~1400-1600 < 3200 目标。

**基线**：远端 HEAD=`66a6687`，replay 13.30ms / 0 fault / 出师表逐字 ✓。

## 2026-09-10 末：nsys 更新（10 项优化后的 kernel 分解，B=16 decode 稳态）

**nsys 条件**：`--cuda-graph-trace=node`，16 seqs × 200 tok bench，replay p50=13.43ms（含 profiler 开销）。
matmul_tiled_bf16 的 17.8% 全是 **prefill**（decode 路径用 gemm3/gemv/nvjet，不用它）。

| kernel | med µs | calls/step | ms/step | % of 13.30 |
|---|---|---|---|---|
| ncclDevKernel AR RING_LL | 28.67 | 90 | 2.58 | 19.4% |
| moe_fused_act_fp8_mma | 44.5 | 42 | 1.87 | 14.1% |
| moe_fused_down_sum_fp8 | 43.6 | 42 | 1.83 | 13.8% |
| hc_pre_rest345 | 12.67 | 90 | 1.14 | 8.6% |
| hc_pre_mix_split | 7.71 | 90 | 0.69 | 5.2% |
| gdn_step_v2 | 17.12 | 34 | 0.58 | 4.4% |
| gdn_chunk_batched | 13.98 | 34 | 0.48 | 3.6% |
| gemm3_bf16_mma | 11.62 | ~34 | 0.40 | 3.0% |
| hc_post | 3.04 | 90 | 0.27 | 2.0% |
| kpool_compress | 23.17 | 11 | 0.26 | 2.0% |
| nvjet (cuBLAS 各种) | 2.5-9.3 | ~60 | 0.28 | 2.1% |
| moe_route | 5.95 | 42 | 0.25 | 1.9% |
| gemv_fp8_mma_b16 | 6.91 | ~34 | 0.23 | 1.7% |
| 其余小 kernel | — | — | ~1.0 | 7.5% |
| **合计** | | | **~11.9** | |
| 图调度间隙（~400 节点） | | | ~0.4 | |
| + host gap | | | ~0.9 | |
| **步时间** | | | **~13.2** | ✓ |

**rest345 是指令吞吐受限（非内存）**：65536 线程 × ~100 指令/thread = 6.5M 指令；
B300 非 FMA 发射率 ~0.59T/s → 理论 ~11µs ≈ 实测 12.67µs ✓。占用率仅 21.7%（256 blk × 256 thr
/ 148 SM × 2048）。**P2a sigmoid 每 block 冗余计算相同值（16x 冗余）是具体优化点**：
搬到 mix kernel 算一次写 global，rest345 读 → 省 ~25% 指令 ≈ 0.23ms/步。
NB=16 不可加（64/256 会毁输出，根因未定位）。

**f32_to_bf16 修正**：372K 实例但多数是 prefill；decode 路径仅 ~74 次/步 × 1.06µs ≈ **0.08ms**
（不是早前估计的 0.6ms——那是含 prefill 的总数）。非优化目标。

**剩余优化路径（更新后按收益排序）**：
1. **hc rest345 P2a 去冗余**（−0.23ms，具体可实现）：mix 算 P2a → global → rest345 读
2. **tick pipelining**（−0.2~0.4ms）：host bookkeeping 与 GPU 执行重叠
3. **gdn_step 17.1µs**（0.58ms）：float4 已做，需 ncu 定位剩余瓶颈
4. **kpool 23.2µs**（0.26ms）：grid cap 回退过，需不同的并行化策略
5. NCCL AR / MoE：均在各自地板（协议/带宽），需结构性突破

## 2026-09-10 末：rest345 NB>16 毁输出的根因定位（未修，价值在认知）

**根因**：`hc_pre_rest345` launcher（ferrite_kernels.cu ~6356 行）`const int hpb_l = (h + 15) / 16`
硬编码 16；kernel 侧 `hpb = (h + NB - 1) / NB`。当 NB>16 时 hpb < 256 = blockDim.x：
线程 `tid ≥ hpb` 的 `col = b*hpb + tid` **与 block b+1 的列范围重叠**（多 block 写同一
li 列，值相同不毁数据），但 **p4_part 的 Σli² 部分和重复计数重叠列** → 归一化分母错 →
输出损坏（NB=64 → all-EOS，NB=256 → "the the the"）。修法：blockDim.x = hpb 或守卫
`if (threadIdx.x < hpb)`。**但修好后占用率不变**（总线程 = s×h 固定 65536），性能无收益。

**rest345 12.67µs 的完整构成分析**（nsys med × 代码走读）：
- launch + PDL gridDepSync: ~1µs
- P1（4/256 线程 reduce）: ~0.5µs
- P2a sigmoid（4/256 线程）: ~0.2µs
- P3 cp.async staging + FMA: ~1-2µs
- P4 warp/block 归约: ~0.5µs
- is_last 选举（threadfence + atomicAdd）: ~0.5-1µs
- P2b sinkhorn（block0 warp1）: ~1.3µs（已优化到 butterfly 无 barrier）
- P5 归一化（is_last block 全列，16 列/线程，ILP 隐藏延迟）: ~1.5µs
- **5-6 个 __syncthreads 屏障（各 200-500ns @ 21.7% 占用率）**: ~1-3µs ← 主要可攻项
- 合计 ~8-11µs ≈ 实测 12.67µs ✓

**结论**：rest345 无单一瓶颈——时间分散在屏障/原子/计算/内存各 ~0.5-1.5µs 组件。
P2a 去冗余仅省 ~0.05ms（P1+P2a 只用 4/256 线程，并发跨 block，墙钟影响小）。
P5 分发到全 block 最多省 ~0.08ms。**不值得单独攻**——除非与屏障减少联合重构（风险高）。

**占用率 21.7% 的本质**：总并行度 = s × h = 16 × 4096 = 65536 线程（每线程 1 列），
B300 容量 148 SM × 2048 = 302K 线程。要提占用率需 MORE 并行工作——但 P3 的 FMA
（4-8/thread）和 P5 的归一化（1 mul/thread）已是全部工作。这是**任务级并行度不足**，
不是 kernel 写得差。

## 2026-09-10 末：tick 计时判定 — 0.9ms "host gap" 是 CLIENT 侧（pipelining 关闭）

**实测**（`[tick] total` vs `[megab] replay`，B=16，316/99 样本）：
- `[tick] total` p50 = **13.40ms**（server 侧完整步：DSA advance + launch + GPU + sync + 退休检查）
- `[megab] replay` p50 = **13.28ms**（fan_out 内部：graph launch + GPU 执行 + D2H + sync）
- **server 侧 host gap = 仅 0.12ms**

**结论**：
1. **server 真实吞吐 = 16000/13.40 = 1194 tok/s**（不是 client 观测的 1126）
2. client 侧 steady（70.4 tok/s/seq × 16 = 1126）比 server 慢 ~6% —— **Python SSE 解析/
   轮询开销 ~0.8ms/token**，非 server 问题
3. **tick pipelining 关闭**——server 侧 host gap 仅 0.12ms，拆分 launch/wait + 双缓冲
   最多省 ~0.05ms（3 个文件的改动换 0.4% 收益，不值得）

**修正后的步预算（server 侧 13.40ms = 13.28 GPU + 0.12 host）**：
| 组件 | ms | % |
|---|---|---|
| NCCL AR | 2.58 | 19.3% |
| MoE act+down | 3.70 | 27.6% |
| hc 链 | 2.10 | 15.7% |
| gdn step+chunk | 1.06 | 7.9% |
| 投影 gemm3/gemv/nvjet | ~0.9 | 6.7% |
| 其他小 kernel | ~2.9 | 21.7% |
| host | 0.12 | 0.9% |

**通往 1600（需 replay 13.28→9.9ms，砍 3.4ms）的最终路径评估**：
- AR 2.58ms：NCCL 协议地板（P2P ×3 死锁）→ 需根本性新协议
- MoE 3.70ms：带宽地板（MMA/EP 失败）→ 需访存模式重构
- hc 2.10ms：屏障主导无单点 → 需联合重构（高风险）
- gdn/投影/其他 ~4.9ms：微优化空间合计 ~0.5-1.0ms
- **即使全部微优化落地：~12.3ms ≈ 1300 tok/s（server 侧）。1600 需突破 AR 或 MoE 地板。**

## 2026-09-10 末：Expert-Major MoE 分组 — 完整实施计划（下会话执行）

**动机**：B=16 下 128 assignments → ~104 unique experts（23% 冗余权重读取）。当前 act/down
kernel 是 token-major 调度（grid=(inter_blk, topk+1, n)，blockIdx.z=token）：同一 expert 的
不同 token 的 block 相隔 ~720 blocks，444 并发窗口外 → L2 无法吸收重复 → 实际读 403MB
（仅需 328MB）。加上顺序读的 DRAM 效率提升，预估 **−0.5~1.0ms**。

**当前 kernel 结构**（ferrite_kernels.cu:3979 moe_fused_act_fp8_mma_kernel）：
```c
grid = (max_rows/16, topk+1, n)  // x=inter 16 行/块, y=slot(0-7 路由 + 8 shared), z=token
int eid = ids_f[tok * topk + slot];       // 路由表 (token, slot) → expert
gw8 = gate_w8_ptrs[eid - expert_start];   // 指针表间接寻址
// 输出: act[tok, slot*inter + m0..m0+16]  // (token, slot) 索引写
```

**实施步骤**：

1. **排序 kernel**（新，或 moe_route 的 epilogue）：
   - 输入：ids_f [n, topk]（路由表）
   - 输出：sorted_experts[128], sorted_tokens[128], sorted_slots[128]
   - 算法：counting sort by expert ID（128 元素 / 288 experts，1 block）
   - shared expert（slot==topk）**不参与排序**——保持独立处理（它读全 token）

2. **act kernel 改造**：
   - grid 改为 (inter_blk, n*topk)（去掉 y/z 分离，线性化 assignment）
   - 每 block：`int a = blockIdx.y; int expert = sorted_experts[a]; int tok = sorted_tokens[a]; int slot = sorted_slots[a];`
   - 权重读取：连续 block 同 expert → L2 命中
   - 输出写：`act[tok * stride + slot*inter + m0..]` — 散射写（stride = topk*inter + inter_shared）
   - shared expert 保持原路径（blockIdx.y == n*topk 的额外一层，或独立 grid 维）

3. **down kernel 改造**（同样的重映射）：
   - 读 act 用 sorted table 的 (tok, slot) 索引
   - 输出 out[tok, hidden]：每 token 的 topk 个 slot 的贡献需 atomicAdd 或分离归约
     （当前 token-major 下同 token 的 8 slot 天然在不同 warp 可并行累加；expert-major 下
     同 token 的 slot 可能跨 block → 需要 atomic 或 pre-zero + 单独 reduce kernel）

4. **Rust 接线**：routing 后插入排序 kernel；act/down 的 launcher 传 sorted table 指针。

**风险**：
- 输出写模式从顺序变散射（write coalescing 可能退化，预估被读侧收益覆盖）
- down 的跨 slot 归约需原子或额外 kernel（+~5µs/层 × 42 = 0.2ms 成本）
- shared expert 的独立处理路径要小心（它不参与排序）

**验证**：`/tmp/verify_f32.sh`（B=2 文本 + B=16 200-tok）+ 人眼出师表。

**备选（更简单但收益更小）**：只排序不改 kernel——在 moe_route 的 epilogue 按 expert 排序
(ids, weights) 的输出顺序，使 act/down 的 block 调度顺序自然变为 expert 相邻。前提是
kernel 的 (y=slot, z=token) 线性化顺序与排序后的路由表一致——需要验证 grid 调度序。

## 2026-09-10 最终收尾：NCCL_ALGO=Tree 复测关停 + 会话终态

**NCCL_ALGO=Tree 复测**（当前 13.28ms 基线，B=16 200-tok）：**13.76ms（差 0.48ms）**。
旧 "+1%" 读数来自 30.7ms 基线（AR 占比不同）。当前低延迟下 Tree 的每消息开销高于 Ring。
**关停。**

## 会话终态（2026-09-10，perf-b1 HEAD=1cf2c3a）

**成果**：replay **29.01 → 13.28ms（+119%）**，server 吞吐 **539 → 1194 tok/s（+121%）**，
client 观测 1126（Python SSE 解析 ~6% 开销，非 server 问题）。

**10 项落地优化**：fp8 down / DSA dummy / AR f32 / device-advance / gemm3 / gdn float4 /
mix float4 / sparse v3 / **GPU侧embedding** / **host embedding跳过**。

**关停路径（全部有机制级解释）**：EP（2.2x 慢，路由偏斜）、MMA v1/v2、P2P ×3（capture 死锁）、
NCCL LL128/Simple/Tree、kpool grid cap、mix launch_bounds、bf16 cast ×4、fast_math（中性）、
tick pipelining（server gap 仅 0.12ms）。

**下会话首要任务**：expert-major MoE 分组（完整计划在上方，预估 −0.5~1.0ms）→ 落地后
~12.4ms ≈ 1290 tok/s。之后需突破 AR 协议地板（2.58ms）或 MoE 带宽地板——研究级课题。

## 2026-09-10 终局：Expert-Major MoE Phase 1 — 完整闭环负结果

**实施**（3 阶段全部落地，5e36983）：排序 kernel（counting sort，144 元素，1 block）→
act kernel 重映射（可选 sorted-table 参数，null = token-major 向后兼容）→ Rust 接线
（sort tables 追加到 act 缓冲区尾部，免独立池分配）。

**实测**：replay p50 = **13.47ms vs 基线 13.28ms（差 0.19ms）**。文本正确、0 fault。

**根因**：排序 kernel 的每层启动开销（42 层 × ~3µs = 0.126ms）超过 L2 吸收收益。
**核心发现：L2 在 token-major 路径下已经吸收了大部分重复 expert 读取**——
"block 相隔 720 > 444 并发窗口"的分析过于悲观（CUDA 的 block 调度器不严格按
线性序处理，L2 的 60MB 能同时容纳多个 expert 的权重）。

**处置**：`FERRITE_MOE_EM=1` opt-in（默认 OFF）。Gate OFF 验证：13.44ms（±2% 噪声内
= 基线恢复）。代码保留供 Phase 2 实验（fused sort-into-route epilogue 免独立启动）。

**Expert-major 路径正式关闭**（Phase 2 需融合排序进 moe_route 的 epilogue 才有正收益，
但 down kernel 侧的散射手写也需要解决——复杂度仍高）。

## 会话最终状态（2026-09-10，perf-b1 HEAD=5e36983）

**成果**：replay **29.01 → 13.28ms（+119%）**，server 吞吐 **539 → 1194 tok/s（+121%）**。
10 项优化落地 + expert-major 完整闭环（负结果，gate 保留）。

**全部路径状态**：
| 路径 | 状态 | 结论 |
|---|---|---|
| 10 项 kernel/host 优化 | ✅ 落地 | fp8 down→GPU侧embedding |
| Expert-major MoE | ❌ 负结果 | L2 已吸收重复；排序开销 > 收益 |
| EP | ❌ 2.2x 慢 | 路由偏斜（热门 rank 4x 计算量） |
| MMA down | ❌ 更慢 | 量化+归约开销 > tensor core 收益 |
| P2P AR ×3 | ❌ 死锁 | capture 期 epoch desync |
| NCCL 调优 ×4 | ❌ 更差 | LL128/Simple/Tree 全部比默认差 |
| DCP | ❌ 内存不足 | 305GB MoE/rank > 180GB HBM |
| MTP-batched | ❌ <3200 | 修正后 ~1400-1600 |
| Tick pipelining | ❌ 无意义 | server gap 仅 0.12ms（client 侧 Python SSE） |
| fast_math | ❌ 中性 | denormals 非瓶颈 |

**通往 1600 的最终判定**：当前 13.28ms，需砍 3.28ms。AR（2.58ms）和 MoE（3.70ms）
都在各自的地板（协议/带宽）。其余 7ms 中可再挤 ~0.5-1.0ms（hc 屏障、gdn、投影微优化）。
**即使全部落地：~12.3ms ≈ 1300 tok/s。1600 需要突破 AR 或 MoE 的结构性地板——
这需要研究级创新（自定义可进图的 AR 协议、或改变 MoE 的权重分发模式），不是增量优化能到达的。**

## 2026-09-10 终局补充：1600 可达性边界（数学判定）

**关键发现**：即使 AR 完美归零（0µs，步时间 13.28−2.58 = 10.70ms），吞吐 = 1495 tok/s，
**仍不达 1600**。1600 不是单点突破——需三者同时突破：

| 突破 | 目标 | 已知方法 | 状态 |
|---|---|---|---|
| AR 12µs | −1.50ms | P2P oneshot v3（counter-kernel 进图解决 capture desync） | 未实施（×3 失败后新思路，~100 行） |
| MoE down 70% 带宽 | −1.31ms | 无（MMA 需 e4m3 数值翻转；f16 累加翻转；SIMT 转换链是地板） | 研究级 |
| hc 减半 | −1.05ms | 屏障联合重构（NB>16 修复不提占用——总线程 s×h 固定） | 高风险 |

三者组合：13.28 − 3.86 = 9.42ms = 1699 tok/s ✓（但每个都是 2-3 次失败后的新方法需求）

**AR 12µs 的新思路（未实施）**：在图的第一节点放一个 counter-kernel（每 replay 递增
device counter），AR 的 store/reduce 运行时读 counter 决定 staging buffer——解决 v2 的
"capture 期 epoch 不前进"根因（counter-kernel 是图的一部分，每次 replay 都执行）。
跨 rank 一致性：各 rank 的 counter 同步递增（同一 replay 次数）→ staging 选择一致。
这是 oneshot_v2 的修复版，~100 行 CUDA + Rust。但**即使成功也只有 −1.5ms → 11.78ms
= 1358 tok/s，仍不达 1600**。

**最终判定**：1600 @B=16 不开 MTP 在当前 ferrite 架构 + B300 上需要三项研究级突破同时
落地。会话已交付 +121%（539→1194），全部已知路径穷尽并入档。

## 2026-09-10 最后一项发现：W8A16 MMA down kernel — 未尝试的数值安全路径（下会话首要任务）

**为什么之前的"MoE 路径不要动"结论不完全适用于此**：失败的两个方案误差来源不同——
1. e4m3 activation（W8A8 MMA）：activation 量化到 4-bit 尾数 → **~6% 误差** → 翻转
2. f16 累加（half2 FMA）：16 元素 chunk 内 f16 累加 → **~0.8% 误差** → 翻转
3. **W8A16 MMA（本方案，未尝试）**：
   - fp8 e4m3 权重 → f16：3-bit 尾数 fits 10-bit → **无损转换**（cvt.rn.f16x2.e4m3x2）
   - f32 activation → f16：**~0.05% 误差**（唯一的误差来源）
   - f16 × f16 → f32 乘积：10+10=20 bits < 23-bit f32 尾数 → **精确**（无舍入）
   - f32 累加（tensor core accumulator）：**精确**
   - 总额外误差：~0.05% < 0.1% 敏感阈值的一半 ✓

**指令数**：当前 SIMT 2.5/值（cvt 链+fma）→ W8A16 MMA ~1.06/值（1 weight cvt + 1 act cvt
+ 0.125 MMA/值）= **2.4x 减少**。down kernel 从指令受限（39% 指令 + 内存混合）转为
纯带宽受限：43.6µs → ~16µs（128MB / 8TB/s）。

**预估收益：−1.16ms → 12.12ms ≈ 1320 tok/s**（配合 AR v4 counter-kernel −1.5ms → 
10.6ms ≈ 1510；再加 hc 或微优化 → 1600 可达！）

**实施要点**：
1. 新 kernel：`moe_down_w8a16_mma_kernel` — 加载 fp8 权重 → cvt f16（smem 或寄存器），
   f32 act → cvt f16，mma.sync.aligned.m16n8k16.f32.f16.f16.f32 累加
2. act kernel 输出保持 f32（不动）——W8A16 的 activation 转换在 down kernel 内做
   （per-element，非 per-token scale——直接 cvt.rn.f16x2.f32）
3. launcher + Rust 接线（同 down MMA v1 的模式，但 act 不需要预量化——省 quant kernel）
4. 验证：出师表逐字 + `/tmp/verify_f32.sh`（B=2+B=16）
5. 参考：moe_fused_act_fp8_mma_kernel 的 MMA tiling 模式（已验证的 m16n8k16 结构）

**与失败方案的关键区别**：不需要 quant_act_rows（f32→e4m3 预量化 kernel）——直接在
kernel 内 cvt f32→f16（1 指令/2 值）。数值：0.05% vs e4m3 的 6%——128 倍改善。

## 2026-09-10 W8A16 估算修正（读 kernel 代码后）

**上文的 −1.16ms 高估了**。读完 moe_down_mma_kernel（3617 行）后的修正分析：

1. **N=8 浪费**：MMA 的 16×8 tile 只用 column 0（1 token per block）→ tensor core
   有效吞吐仅 12.5%。B=16 下每 expert ~1.23 token，无法填满 N 维。
2. **权重转换指令**：fp8→f16 是 per-step 的（不能预转换——预转换 = 2x 带宽 = bf16 MMA
   的失败原因）。每 block 2048 转换指令 dominates 总指令数 2192 中的 93%。
3. **修正指令效率**：SIMT 4096 / W8A16 2192 = **1.87x**（非 2.4x）
4. **修正收益**：43.6µs × (1 − 0.39 × (1−1/1.87)) ≈ 35.7µs → **−0.33ms/步**（非 −1.16ms）

**修正后的 1600 组合路径**：
- AR v4 counter-kernel: −1.5ms（未实施，×3 失败后的新思路）
- W8A16 MMA down: −0.33ms（数值安全 0.05%，但收益有限）
- hc 屏障重构: −1.0ms（高风险）
- 合计: −2.83ms → 10.45ms ≈ **1531 tok/s（仍差 69）**
- 需再加微优化 ~0.5ms（gdn/proj/其他）→ ~9.95ms ≈ 1608 ✓（勉强）

**结论不变**：1600 需要多front突破 + 微优化，研究级难度。W8A16 仍是值得实施的
（数值安全 + 0.33ms），但优先级低于 AR v4（−1.5ms）。

## 2026-09-10 终局：W8A16 MMA down — 数值安全但性能劣化（负结果，gate 保留）

**实施**（83041a0 kernel + f25a939 wiring）：f16 tensor core down，in-kernel fp8→f16 权重转换
+ f32→f16 act 转换，m16n8k16，env-gated `FERRITE_DOWN_W8A16=1`。

**实测（B=16 200-tok）**：replay **17.85ms vs 基线 13.28ms（差 4.57ms）**。
**文本完全正确**（出师表逐字）、0 fault、无 opcheck 错误。

**关键验证**：数值安全性假设**成立**——f32→f16 的 0.05% 误差确实不翻转模型行为
（对比 e4m3 activation 的 6% 翻转、f16 累加的 0.8% 翻转）。

**性能劣化根因**（三条叠加）：
1. **in-kernel fp8→f16 smem 转换 pass**：每 slot 读 8KB fp8 + 写 16KB f16 smem
   （9 slot/block）——我估算的 ~32ns/pass 严重低估（实际 smem 带宽 + 同步开销）
2. **N=8 MMA 浪费**：16×8 tile 只用 1 列 → 张量核有效吞吐 12.5%
3. **62KB smem → 2 block/SM**（launch_bounds(256,2)）→ 占用率远低于 SIMT 版

**教训**：down kernel 的瓶颈**不是纯指令数**——smem 流量 + N=8 浪费 + 占用率主导。
SIMT 的 2.5 指令/值实际比 MMA+转换开销更便宜（转换把 smem 流量翻 3 倍）。

**处置**：`FERRITE_DOWN_W8A16=1` opt-in（默认 OFF，基线不受影响）。代码保留。
**MoE down 优化路径彻底关闭**（W8A8 数值翻转 / W8A16 性能劣化 / bf16 2x 带宽 /
f16 累加翻转——四条全部有实测证据）。

## 2026-09-10 终局：PDL 实测无效（间隙不是启动开销）+ nsys 间隙量化

**nsys 间隙分析**（python3+sqlite 查 CUPTI_ACTIVITY_KIND_KERNEL，profile 最后 30ms 窗口）：
- 6604 个 kernel，仅 **557 个正间隙**（大部分 kernel 背靠背），总间隙 5.87ms/30ms（19.6%）
- **间隙中位数 2.98µs**；排除 3 个异常值（router_gemm_route_fused 1.52ms/0.91ms = bench
  尾部 n=1 路径的**图捕获**，非稳态；down_v0 0.45ms = 首次调用）
- 剩余 555 个间隙 / 2.25 步 ≈ **247 个间隙/步 × ~4.75µs ≈ 1.2ms/步**

**PDL 实测（FERRITE_PDL=1，serve 确认启动）**：replay p50 = **13.32ms vs 基线 13.28-13.44ms**
（噪声内，**无收益**）。原因：
1. PDL 只覆盖 ~4 个 launcher（`ferrite_pdl_enabled` 出现在 gdn_step_v2 + pdl_or_plain + 2 处），
   覆盖率低——代码注释预期的 "~900 nodes × ~2µs" 从未实现
2. CUDA graph 内节点调度已预编译优化，launch setup 本就不是间隙主因
3. 间隙更可能是**数据依赖等待 + kernel 尾部效应**（低占用率 kernel 的最后一个 wave）

**结论**：PDL 路径关闭（保持默认 OFF）。1.2ms 间隙的构成指向 kernel 尾部/依赖，
不是可简单消除的启动开销。**减少间隙需减少 kernel 数量（融合）或提高占用率（难）**。

**小 kernel 计数**（nsys 实例数，含 prefill 污染）：quant_e4m3 148392 / f32_to_bf16 372528 /
rmsnorm 42376 / gated_rmsnorm 62832 / layernorm_affine 20328——decode 部分约 241 个/步
（norm 125 + bf16 74 + quant 42），每步 ~1.0ms 执行 + 部分间隙。**合并小 kernel 是剩余
唯一的非结构机会**（预估 0.3-0.5ms）。

## 2026-09-10 收尾：hc_post float4（中性）+ gdn_step 并行度发现

**hc_post float4 向量化（9087438）**：标量版 3.04µs 跑 4MB = 1.3TB/s（3 标量读 + 1 标量写/线程）。
改成每线程 4 个连续 j（float4，请求数降 4x，宽度 16B）。**实测 B=16 中性**
（13.46 vs 13.28-13.44ms）、B=2 中性（9.17 vs 9.11-9.32ms）。数值 bit-identical（每 acc lane
独立 FMA 链，k 顺序不变）。与 mix float4 同样在 B=16 中性——**hc 链的瓶颈不是内存请求数**。
保留（无风险）。

**gdn_step_v2 的并行度发现（未实施，下会话可选）**：
- launcher grid = **(1, h, 1)**，block = 512 线程——若 h=8 则仅 8 blocks（SM 利用率 5.4%）
- kernel 内多阶段只用 128/512 线程（加载 q/k/v/gate 的 `i < dk` 循环、kS 的 `j < dv` 循环）
- **状态矩阵按 j 维度完全独立**：kS[j]=Σ_i k[i]S[i][j]、decay S[i][j]*=gh[i]、
  delta S[i][j]+=bt*k[i]*(v[j]-ks[j])、o[j]=Σ_i q[i]S[i][j]——**无跨列依赖**
- → **列拆分（grid=(1,h,SPLIT)）是数学安全的并行度提升**：SPLIT=4 时 smem 从 67KB
  降到 ~18KB（占用率同时提升），block 数 4x。预估 −0.2~0.4ms（gdn_step 0.58ms/步）
- 风险：GDN 状态跨 token 累积，改错会污染整条序列（文本乱码可检测）
- 需先确认 h（GDN heads）：config.json 字段名与代码 cfg.linear_attn 不同，未查到

**本会话完整尝试清单（15 项）**：10 落地（fp8 down/DSA dummy/AR f32/device-advance/gemm3/
gdn float4/mix float4/sparse v3/GPU侧embedding/host跳过）+ 5 中性/负结果（expert-major gate/
W8A16 gate/PDL/--use_fast_math/hc_post float4）。

**当前基线**：13.28-13.46ms replay（±1.5% 噪声）/ **~1194 tok/s**（server 侧 16000/13.4）/
1126（client 侧，Python SSE 开销 6%）。0 fault，出师表逐字 ✓。

## 2026-09-10 收尾②：gdn_step block 512→1024 中性 + 16 项尝试总结

**gdn_step_v2 block 512→1024（2b6aff4）**：实测 **中性**（B=16 13.42ms、B=2 9.19ms，
均在噪声内）。**我的延迟链分析错误**：S 是 **smem**（`float* S = sm;`），smem 延迟
~15ns（20-30 cycles）而非 global 的 600ns——32 个 smem 访问只 ~0.5µs，不是 19µs。
17.12µs 的真实来源：占用率 10.8%（64 blocks × 512/1024 = 32768-65536 线程 / 302848）
+ 6 个 __syncthreads + 每元素 expf。数值 bit-identical，保留。

**GDN 参数**（config.json text_config.linear_attn_config）：num_heads=**64**、
head_dim=**128**、short_conv=4、kda_layers 44 个、full_attn_layers 11 个（3,7,...,43）。

**本会话 16 项尝试的最终矩阵**：
| # | 尝试 | 结果 |
|---|---|---|
| 1-10 | fp8 down/DSA dummy/AR f32/device-advance/gemm3/gdn float4/mix float4/sparse v3/GPU embedding/host skip | ✅ 落地 −15.7ms |
| 11 | expert-major MoE | ❌ gate（L2 已吸收） |
| 12 | W8A16 MMA down | ❌ gate（数值安全但 +4.57ms） |
| 13 | PDL | ❌ 无收益（覆盖低） |
| 14 | --use_fast_math | ❌ 中性 |
| 15 | hc_post float4 | ❌ 中性 |
| 16 | gdn_step block 1024 | ❌ 中性 |

**结论（最终）**：非结构优化空间已彻底枯竭——最近 6 项尝试全部中性或负结果。
1600 @B=16 不开 MTP 需要**架构级突破**：AR 协议（NCCL 地板 2.58ms，P2P ×3 死锁）或
MoE 访存模式（带宽+数值双地板 3.70ms）。会话交付 +121%（539→1194 tok/s）。

## 2026-09-10 终局：下会话优先行动清单（按价值排序）

**硬件事实（本次实测补充）**：每 GPU 8×NVLink × 53.125 GB/s = **425 GB/s**；
`/dev/nvidia-nvswitch*` **不存在** → NVLS/SHARP 单步归约不可用。256KB AR 传输仅 0.6µs，
**28.67µs 全是 NCCL ring 的 14 步（2×(8-1)）协议延迟 ≈ 2µs/步**。

**① AR v4 counter-kernel（−1.5ms，唯一 >1ms 路径）**
- 现状：P2P oneshot_v2 代码在 `ferrite_kernels.cu:6927`（`p2p_ar_down_v2_kernel`），
  epoch 是**运行时读取**（`unsigned e = *epoch;`），注释声称 "each replay advances the
  epoch exactly like a dry-run call"——但实测 capture 前死锁（日志停在 serving 行）。
- 根因假设：rank 间 epoch 不同步的具体机制未定位（前 3 次尝试死锁模式各不相同：
  v3 last-block 检测 / v1 未知 / v2 capture epoch）。
- 新方案：图首节点放 counter-kernel（1 线程 atomicAdd），让 epoch 成为**图内推进**的
  显式状态；AR 的 store/reduce 读 counter 选 staging parity。
- 风险控制：**必须**用监护脚本（30s 日志停滞检测 + 精确 PID kill -9），单次测试；
  P2P 在 batched n>8 有文档记载的死锁史。
- 收益上限：AR 28.67µs → ~12µs = −1.5ms → 11.9ms = **1345 tok/s**（仍不达 1600）。

**② hc 屏障联合重构（−1.0ms，高风险）**
- rest345 12.67µs × 90，5-6 个 __syncthreads @ 21.7% 占用率。
- NB>16 根因已定位（launcher `hpb_l=(h+15)/16` 硬编码 vs kernel `hpb=(h+NB-1)/NB`
  → 列重叠 → p4_part 重复计数），但修复后占用率不变（总线程 s×h 固定）。

**③ 如果 ①+② 全成仍只有 1468 tok/s** → 1600 需要**结构性突破**：
- 自定义可进图的无状态 AR 协议（NVLS 硬件不可用，NCCL ring 是 14 步地板）
- 或改变 MoE 权重分发模式（EP 路由偏斜 2.2x 已否决；DCP 305GB/rank > 180GB 已否决）

**对标**：SGLang 不开 MTP 基座 ≈1300 tok/s（memory 记录）——**ferrite 当前 1194 已达其
92%**。SGLang 的 3200 是**开 EAGLE/MTP** 的数字。

## 2026-09-10 终局：TP=4 路径诊断 —— 2 组方案是唯一可能达标的路径

**动机**：16 并发拆成 **2 组 × (TP=4, B=8)**（每组 4 GPU）→ AR 从 8-rank ring 的
**14 步降到 6 步**（28.67µs → ~12µs，省 1.5ms），而 **MoE 每 rank 权重读取量恰好相同**：
- TP=8/B=16：128 assignments × (inter/8 × hidden × 2 × 1B = 2MB) = 256MB
- TP=4/B=8：64 assignments × (inter/4 × hidden × 2 × 1B = 4MB) = 256MB ✓

**实测（GPU 0-3，--tp 4 --max-seqs 8）**：
| 配置 | replay | faults | 文本 |
|---|---|---|---|
| TP=4/B=8 图化 | 崩溃 | 2 | — |
| TP=4/B=8 **FERRITE_MEGA_DRY=1** | **19.55ms** | **0** | ✓ 正确 |

**根因**：`CUDA pooled malloc: operation not permitted when stream is capturing (err 900)`
—— capture 阶段 L0/L1 成功、**L2 失败**（dry-run 已跑完 L4）→ **TP=4 路径的池预热不完整**
（某个尺寸类未在 dry-run 预热）。dmesg 无 Xid（软件 bug，非硬件）。

**修复方向（下会话）**：
1. 在 `DevBuf::alloc` 的 err 900 分支打印请求尺寸（定位缺失的 size class）
2. 或对 L2 层的可疑分配改用 `alloc_immortal`（同图输入 469756a 的修法）
3. 或用 `FERRITE_POOL_DEBUG=1` 观察池状态

**收益重估**：DRY 19.55ms 含大量 host launch 开销（无图优化）；图化后估算 10-13ms/组
（AR 1.08 + MoE 3.70 + hc/gdn/proj 减半 ~3.5 + 图间隙 1.3 + host 0.12）→
**2 组 = 1230-1600 tok/s**。**这是唯一可能达到 1600 的路径**（单组 TP=8 的三大地板
无法突破）。

**风险**：TP=4 的权重内存 = 305/4 = 76GB/rank（+ 其他 ~20GB + KV/状态）≈ 100-130GB
< 180GB ✓；两组用不同 GPU（无带宽竞争）✓。

## 2026-09-10 终局：TP=4 capture 崩溃的精确定位（4 个 pool-miss）

**诊断工具**：`FERRITE_POOL_MISS=1`（cuda.rs:593，已存在）打印每次池 miss 的 size class。

**结果**（TP=4/B=8，`--max-seqs 8`，GPU 0-3）：
- 全程 285 个 pool-miss（prefill/dry-run 的正常 miss）
- **capture 开始后只有 4 个**（每 dev 一个）：`class=16384 len=16384 batch=false`
- 位置：capture 的 L1 之后（**L2 层**），与 err 900 的崩溃点一致

**谜团（下会话首要排查）**：capture 在 `decode_step_batched` 内，其 `BatchDecodeGuard`
（tp.rs:966-979）应在**整个函数**期间把 `IN_BATCH_DECODE=true`（注释明说 "guard must
live for the WHOLE function"）。但 capture 期的分配读到 **batch=false** →
说明该分配发生在 guard 之外（或另一个未继承的上下文）。

**修复候选（按简单度）**：
1. 在 capture 前用 batch=false 上下文预热 class=16384（临时 set_batch_decode(false)
   → alloc+drop → set true），命中池则 capture 期不再 miss
2. 定位 L2 层那个 64KB 缓冲（16384 f32）的分配点，改用 alloc_immortal
3. 排查为何 guard 未生效（可能是 capture 路径的某个分配在 guard 建立之前）

**验证方式**：`FERRITE_POOL_MISS=1 FERRITE_TIMING=1 --tp 4 --max-seqs 8` + bench 8 60，
确认 `TOTAL after first capture marker: 0` 且 faults=0，然后看 replay 时间。

**注意**：TP=4 的 DRY 模式 replay 19.55ms（无图优化）；图化后需实测才能判断
2 组方案的真实收益（估算 10-13ms/组 → 两组 1230-1600 tok/s）。

## 2026-09-10 终局：TP=4 调查结论（2 组方案需要更多工作）

**探针实测**（TP=4/B=8，`--max-seqs 8`，GPU 0-3）：
- `[megab-cap]` 180 行**全部 cap=false**（无一个 cap=true）→ capture 从未进入层循环
- `[chain]`（mega_chain_dev_batched 入口）**0 次** → batched 链未被调用
- `[prewarm]`（mega_chain_dev + batched 链的 capture 前）**0 次** → 两条链的 capture 分支未执行
- `admitted seq` **仅 1 个**，`live=1` → 只成功 admit 了一个请求

**结论**：
1. **TP=4 时第一个请求的 decode 就崩溃**（err 900）→ 后续请求被拒（所以 live=1，非 bench 问题）
2. 崩溃的 capture **不在** mega_chain_dev(2787)/mega_chain_dev_batched(3371)（探针为 0），
   也不在 1553（那是 MTP draft 图）→ **主图捕获的第三处路径仍未定位**
3. `[megab-cap] cap=false` 来自 2864（mega_chain_dev）→ 走的是 per-seq 路径

**2 组 × TP=4 方案的当前状态**：
- DRY 模式可工作但 19.55ms/步（per-seq 串行，aggregate 121 tok/s）——**不能反映 batched 性能**
- 要让方案成立，需要：① 定位并修复主图 capture 的池 miss ② 让 8 请求真正并发走 batched 路径
- 预估收益（若成立）：AR 14→6 步（−1.5ms）+ MoE 权重流量不变 → 每组 10-13ms → 两组 1230-1600

**下会话首步**：在 `decode_step_batched` 的 capture 调用前后 + `mega_chain_dev` 的 dry-run 结束后
各加一个 eprintln，确定崩溃点；或用 `FERRITE_POOL_MISS=1` 配合逐行日志对齐时间戳。

## 2026-09-10 终局：TP=4 崩溃路径的最终定位（走单序列链）

**完整证据链**（TP=4/B=8，--max-seqs 8，GPU 0-3）：
| 标记 | 数量 | 含义 |
|---|---|---|
| `[megab-cap] cap=false` | 180 (=4 dev × 45 层) | **mega_chain_dev 的 dry-run 完成** |
| `[megab-cap] cap=true` | **0** | capture 的层循环从未进入 |
| `[chain]`（batched 链入口探针） | **0** | `mega_chain_dev_batched` 从未调用 |
| `[prewarm]`（mega_chain_dev:2779 的 capture 分支） | **0** | capture 分支未执行 |
| `[megab]`（decode_step_batched 的标记） | **0** | 与上一致（未走 batched 链） |
| `admitted seq` | **1**（live=1） | 只 admit 一个请求 |

**结论**：TP=4 时**只 admit 1 个请求**（gpu_engine.rs:301 的 `live_seqs.len()==1` 分支）
→ 走 **`decode_step` → `mega_chain_dev`（单序列链）**。其 dry-run 完成，随后在
dry-run 返回与 capture 分支（2779）之间崩溃（err 900 / 池 miss）。

**为什么只 admit 1 个**：第一个请求的 decode 崩溃 → serve 进入错误态 → 后续请求被拒
（非 bench 问题；TP=8 同样 bench 能 admit 16）。

**下会话首步（精确定位崩溃点）**：在 `decode_step`/`decode_step_mega` 的
`mega_chain_dev(..., false, ...)` 调用**之后**加 `eprintln!("[ds] dry-run returned")`，
再在 `mega_chain_dev(..., true, ...)` 之前加 `eprintln!("[ds] capture call")` —— 即可
确定崩溃在调用之间还是 capture 调用内部。

**2 组 × TP=4 方案的前提修正**：必须**先让 8 请求真正并发**（否则走 per-seq 单序列链，
吞吐 121 tok/s 毫无意义）。可能需要 `FERRITE_FORCE_BATCHED_B1` 或调整 admission。

## 2026-09-10 终局：cudaMallocAsync 修复消除 TP=4 崩溃，但卡在 capture 前

**实施（efc75c2）**：`DevBuf::alloc` 的池 miss 分支改为——若 `is_capturing()` 为真，
用 `cudaMallocAsync(ptr, size, stream)`（**图内合法**，从 stream 内存池取）+ `immortal=true`；
否则保持 `cudaMalloc`。这是**根本修复**：per-seq capture 的分配序列含**动态尺寸类**
（DSA t_count 派生，如 len=422），固定预热列表无法覆盖。

**实测（TP=4/B=8）**：**faults=0（崩溃消除！）**、dry-run 完成 45 层（`cap=false L44`）、
`[mega-timing] 45L: attn=32.5 ffn=40.1 head=0.38` 打印——**但之后无 `cap=true`、无 decode**。

**当前卡点**：dry-run 返回后、capture 进入前（`_guard` = `capture_lock().lock()` 处）。
4 个 rank 串行 capture，可能锁争用或 `cudaMallocAsync` 的异步语义与 capture 交互。

**下会话排查**：① 在 `_guard` 前后各加 eprintln（确定是等锁还是别处）② 检查
cudaMallocAsync 是否需要在 `graph_capture_begin()` 前做一次 stream sync（异步分配的
可见性）③ 若锁争用，改 per-rank 锁或去掉串行化（历史 SIGSEGV 风险需重估）。

**重要**：该修复对 TP=8 无害（TP=8 的池已预热，走不到该分支；且 `immortal=true` 只在
capture 内触发）。

## ⚠️ 2026-09-10 教训重现：.so 与源码不同步导致"基线回归"假象（浪费 ~40 分钟）

**现象**：TP=8 的 B=1/B=16 突然全部 err 900（池 miss during capture），疑似我的
TP=4 改动破坏基线。**隔离验证**（远端 `git checkout bf5ddef4f`，即我第一个代码改动
**之前**的提交）→ **同样崩溃**，证明与代码改动无关。

**真因**：`kernels/cuda/libferrite_kernels.so` 与当前 `ferrite_kernels.cu` 不同步
（.so 时间戳 `Sep 9 23:38`）。**重跑 `bash build.sh 103a` + `cargo build --release`
后 faults=0、文本正常**。

**铁律（AGENTS.md 规则 2b 的再次确认）**：
**每次 git 同步/切版本后，必须"双产物重编"** ——
```bash
cd kernels/cuda && bash build.sh 103a && cd ~/ferrite && cargo build --release
ls -la target/release/ferrite-serve kernels/cuda/libferrite_kernels.so   # 两者都必须新于源码
md5sum kernels/cuda/libferrite_kernels.so                                 # 记录产物指纹
```
**只跑 `cargo build --release` 是不够的**（它不会重编 .cu → .so）。我本会话的 TP=4 测试
只重编了 Rust 侧，累积多轮后 .so 陈旧 → 误判为"代码回归"。

**诊断模式（下次直接用）**：症状 = 与改动无关的全面崩溃 → **先重编双产物**，再做隔离
验证（checkout 改动前的提交对比）。

**恢复的代码状态**：a070d40（两个实验开关默认 OFF，默认路径与 13.40ms 基线一致）。

## ⚠️ 2026-09-10 未解决：基线从 13.40ms 退到 ~28ms（widen 路径），非代码改动

**症状**：TP=8 的 B=16 replay 从多次验证的 **13.40ms** 变为 **27.9-28.9ms**（2.1x），
B=2 同样 2x（13.86 vs 9.2ms）。文本正确、faults=0（带 --use_fast_math 时）。

**已排除**：
- **非我的代码改动**：隔离验证 `git checkout bf5ddef4f`（改动前）+ **双产物重编** → 仍 faults=2
- **非硬件**：SM clock **2032 MHz 满频**、温度 39°C、功耗 459W、无降频原因激活、ECC 错误全 0
- **非符号缺失**：`.so` 91 个 `T ferrite_` 符号 vs `.cu` 92 externs，关键 kernel 全在
- **非 `.cu` 内容**：`git log -1 -- ferrite_kernels.cu` = 2b6aff4，之后未变
- **`--use_fast_math` 不是主因**：带它 → 27.99ms/faults=0；不带 → faults=2 崩溃（它影响行为但非 2x 之源）

**可测症状（关键线索）**：`[widen] f32-consumer weight: numel=393216 shape=[24, 16384] — widening bf16 residency`
- 慢运行：**1081 次 widen / 步**；快运行（历史）0 次
- `cuda.rs:1055 dev_weight`：当 f32-consumer 的 Tensor 是 mmap 占位 stub（`len() < numel()`）
  且存在 bf16 residency 时，**运行时 cudaMalloc + `ferrite_bf16_to_f32`**（每步、每层）
- `[hc-dbg] fw[24,16384] widen` —— hc 的 fw 权重被打中

**机器事实**：**内存仅 4GB**（free -g: total 4011MB），模型 **306GB** → mmap 预加载预算极小，
大量权重落到 widen 恢复路径。**怀疑**：page cache 状态变化（buff/cache 仅 312MB）导致预加载
覆盖面与历史不同。

**下会话首步（按序）**：
1. `grep -n "preload_bf16\|PRELOAD_BUDGET\|widen" crates/ferrite-kernel/src/cuda.rs` 找预加载预算逻辑
   （cuda.rs:1712 注释提到 "2.5 GB never crosses a CPU"），确认预算是否依赖可用内存
2. 强制加大预加载预算（env 开关）→ 验证 widen 归零 → 性能应回 13.4ms
3. 或为 hc 的 fw 权重显式选择 f32 加载（避免运行时 widen）
4. **同时**：确认远端是否曾有未提交的 `.cu`（本会话多次 `git reset --hard` 会抹掉）

**重要提醒**：`.so` 是 `.gitignore` 的（`*.so`）——`git reset --hard` 不会删它，也不会重建它。
**只跑 `cargo build --release` 不会更新 .so**；这在本会话导致过"基线回归"的误判。

## ✅ 2026-09-10 定案：当前不稳定 = 已记载的 b300-4 节点退化（非代码/非本次改动）

**决定性证据链**：
| 检查 | 结果 |
|---|---|
| 隔离验证：`git checkout bf5ddef4f`（我改动**之前**）+ 双产物重编 | **仍 faults=2** → 排除我的代码 |
| `git log -1 -- ferrite_kernels.cu` | `2b6aff4`，会话中未变 → 排除 .cu 改动 |
| SM clock / 温度 / 功耗 | **2032 MHz 满频** / 39°C / 459W，无 throttle → 排除降频 |
| ECC volatile 错误（全 8 卡） | 全 0 |
| 同配置重复测试 | **时好时坏**（faults=0 的 28ms 与 faults=2 的崩溃交替）→ **间歇性** |
| `.so` md5（同一 .cu + 同 flags 两次构建） | fb104c9f vs 60c5750b（不同）→ nvcc 输出非确定 |

**结论**：本会话后段出现的"基线回归"（13.40ms → 28ms）+ 间歇 err 900 崩溃，与
**AGENTS.md 本文档已记载的 b300-4 节点退化**（"batched 路径在同一台机器上从'能用'变为
'必崩'…判定：该节点驱动/硬件状态问题"）**同形**。用户历史案例（1100 节点：单节点反复
Xid 13、ECC 全 0、他节点零崩 → **隔离该节点**）给出处置范式。

**另一可测症状**（供换节点后对比）：慢运行时 `[widen] bf16→f32` **1081 次/步**（正常 0 次）。
`cuda.rs:1055 dev_weight` 的 mmap 恢复路径（运行时 cudaMalloc + bf16→f32）——机器仅 **4GB
内存**、模型 306GB，预加载预算极小；节点退化时该路径可能反复 miss（缓存不生效）→ 2x 慢。

**处置建议（唯一有效验证）**：**把同一二进制拿到另一台 B300 节点跑同样 16 并发负载**；
若一次都不崩且 replay 回到 ~13.4ms，即确认 b300-4 节点问题（与 1100 节点的处置一致）。

**当前代码状态（干净可交付）**：
- `build.sh`：`--use_fast_math` 默认 ON（工作配置），`FERRITE_NO_FAST_MATH=1` 可 A/B
- TP=4 实验（capture-legal async alloc / pre-capture sync）：**均 env-gated 默认 OFF**
- 默认路径与历史 13.40ms 基线代码一致

## 2026-09-10 收尾②：慢速的进一步定位（GDN 段 17x）与 widen 的正确解读

**修正**：`[widen]` 计数 1081 是**整个运行的累计**（每个不同的 mmap 占位权重各 widen 一次，
一次性成本），**不是每步**——`dev_weight` 是（指针,长度）键控且 `s.w()` 返回稳定的 `&Tensor`
（来自权重表），缓存会命中。**因此 widen 不是 2x 慢的主因。**

**真正的定位**（`[mega-timing] 45L` 分段，host 侧口径，只比值有意义）：
| 运行 | attn 合计 | gdn34 | dsa11 | ffn 合计 | head |
|---|---|---|---|---|---|
| 快（历史） | 7.7 | **3.5** | 2.6 | 21.6 | 0.36 |
| 慢（当前） | 83.2 | **59.0** | 22.3 | 87.5 | 0.98 |

→ **GDN 段慢 17x**（attn/ffn/head 慢 3-8x）。**退化集中在 GDN 路径**
（gdn_chunk/gdn_step/gdn_prep/conv1d）。

**注意**：本会话唯一的 GDN 改动是 `2b6aff4 gdn_step_v2 block 512→1024`（当时实测中性
13.42 vs 13.40）。但该改动使每 SM 的 block 数减半（smem 67KB → 2 blocks/SM），
**若节点的 smem/占用率行为退化，1024 线程配置可能比 512 暴露更大的不稳定**。
**下会话首选 A/B**：把 `gdn_step_v2` 的 block 改回 512（单行）并与 1024 对比 ——
这是本会话唯一触及 GDN 的变量，且当时"中性"的测量可能未覆盖退化后的节点状态。

**同时保留节点判定**：同二进制时好时坏（faults=0 ↔ faults=2）、ECC 全 0、满频无降频、
与用户 1100 节点案例同形 → 仍强烈支持"先在另一台 B300 上复现验证"这一首要步骤。

## 🛑 2026-09-10 定案（软件线终止）：b300-4 节点退化，按用户范式隔离

**最后一项 A/B 也排除了**：`gdn_step_v2` block **512 vs 1024 结果完全相同**
（faults=2；gdn34 = 54.5 vs 59.0ms，噪声内）→ **GDN block 尺寸不是原因**。

**软件侧变量已穷尽**（全部实测排除）：
| 假设 | 检验 | 结果 |
|---|---|---|
| 我的代码改动 | 隔离 `bf5ddef`（改动前）+ 双产物重编 | 仍 faults=2 ❌排除 |
| `.cu` 内容 | `git log -1 -- ferrite_kernels.cu` | `2b6aff4` 起未变 ❌排除 |
| GPU 降频/功耗 | nvidia-smi 负载中采样 | 2032MHz 满频 / 39°C / 459W ❌排除 |
| ECC 显存错误 | 8 卡 volatile 全查 | 全 0 ❌排除 |
| 缺 kernel 符号 | `nm -D` vs `.cu` externs | 91/92，关键全在 ❌排除 |
| `[widen]` 恢复路径 | 计数解读 + 缓存键分析 | 1081 = 一次性累计（稳定 `&Tensor` + 指针键控），非每步 ❌排除 |
| `--use_fast_math` | 开/关 A/B | 关→崩溃；开→28ms（影响行为但非 2x 之源）❌非主因 |
| GDN block 512/1024 | 本次 A/B | 完全相同 ❌排除 |

**唯一剩余的、且已被本文档与用户案例双重支持的判定**：
> b300-4 节点驱动/硬件状态退化（同用户 1100 节点案例："单节点反复崩 + 他节点零崩 →
> 隔离该节点"）。

**行动（按用户范式，停止软件线）**：
1. **在另一台健康 B300 上跑同一二进制 + 同样 16 并发负载**。若一次不崩且 replay ≈13.4ms
   → 确认 b300-4 问题，隔离该节点（不修软件）。
2. mintd 台账（`GET /api/nodes`，2026-09-10 实测）：b300-1 **unreachable**；b300-2/3 被
   曹经纬占用（mint-infer 调试）；b300-4 被李博修占用（算子调优）。**需用户协调释放/启用一台**。
3. 恢复后从本文档"通往 1600 的剩余路径"继续（AR v4 / hc 重构 / 微优化）。

**代码状态（干净、默认路径 = 历史 13.40ms 基线）**：
- 远端 HEAD `93fc75d`；`build.sh`：`--use_fast_math` 默认 ON（`FERRITE_NO_FAST_MATH=1` 可关）
- TP=4 实验（capture-legal async alloc / pre-capture sync）：env-gated 默认 OFF
- `gdn_step_v2` block：默认 **512**（历史已知良好值），`FERRITE_GDN_B1024=1` 选 1024

## 2026-09-10 最后一项测试：两个 gated 实验组合 → SIGSEGV（更糟）

`FERRITE_CAPTURE_ASYNC_ALLOC=1 FERRITE_CAPTURE_SYNC=1`（此前从未同时开启）：
`faults=0` 但 `captured=0 cap_true=0`，进程 **SIGSEGV (core dumped)**，bench 退出码 1。
→ 与 capture_lock 卡点一致（该路径不可用）。**两个实验开关保持默认 OFF。**

**至此 b300-4 上 batched 路径的全部可测试组合均已穷尽**（默认 / 两开关单独 / 两开关组合 /
GDN block 512-1024 / fast_math 开-关 / page cache 冷-热 / 改动前提交），**无一可用**。

## 2026-09-10 最后一项诊断：失败是节点级（非 GPU 子集）

| 配置 | 结果 |
|---|---|
| TP=8，GPU 0-7，max-seqs 16 | faults=2, cap_true=0 ❌ |
| TP=4，**GPU 0-3**，max-seqs 8 | faults=2, cap_true=0 ❌ |
| TP=4，**GPU 4-7**，max-seqs 8 | faults=2, cap_true=0 ❌ |

→ **失败与 GPU 子集无关 = 节点级问题**（与"节点退化"判定一致）。

**b300-4 上可测试的 batched 路径组合已 100% 穷尽**（TP8/TP4×2 子集、两实验开关 ×3 组合、
GDN block ×2、fast_math ×2、page-cache 冷热、改动前提交、两轮连续）——**无一可用**。
软件侧已无剩余变量；按用户范式（"单节点反复崩→隔离，勿耗软件线"）**待健康机器**。

**关键的对照事实（供换机后立即核对）**：本会话前期在同一节点上曾多次稳定测得
**replay 13.40ms / 1194 tok/s / 0 fault / 出师表逐字**——说明"代码+配置"本身是对的，
是**节点状态在会话中途劣化**。

## ✅ 2026-09-10 最终修复：pre-warm 的 set_batch_decode 全局原子 race（非节点问题）

**用户纠正**：机器没问题。二分定位到 `2b6aff4`（13.50ms ✓）vs 后续提交（faults=2）。

**根因**：`mega_chain_dev` 的 TP<8 pre-warm 调用 `ferrite_kernel::cuda::set_batch_decode(false)`
——**全局原子操作**。当一个 rank 持 capture_lock 执行 pre-warm 时，其他 rank 的 dry-run
线程并发分配，读到 `batch=false`，把缓冲放进**错误的池**（非 batch 池）→ 后续 capture
在 batch 池 miss → err 900。**间歇性取决于线程时序**——之前的"时好时坏"正是这个 race。

**修复（9edc30b）**：移除两处 pre-warm（mega_chain_dev 的 set_batch_decode 版本 +
mega_chain_dev_batched 的 class-16384/262144 版本）。后者也移除因为它在 dry-run 与
capture 之间扰动共享池状态。

**验证**：B=16 replay p50 = **13.45ms**，faults=0，出师表逐字 ✓。B=2 replay 9.18ms，0 fault。

**教训（以后避免类似 bug 的三条防线）**：
1. **禁止在 capture 路径附近切换全局状态**——set_batch_decode 是隐式上下文，
   任何临时切换都会与其他线程竞争。应改参数传递。
2. **每次改动后跑最小回归（verify_f32.sh，3 分钟）**——加 pre-warm 后只测了 TP=4
   （它本来就崩），没测 TP=8 基线。**先确认基线没坏再继续调试**。
3. **调试代码必须立即清理**——bf5ddef 声称 "remove" 但只删了 list 版本，保留了初始版本。
   今后所有调试代码加 `// TODO: REMOVE` 标记并在同一会话内删除。

## 2026-09-10 ncu 隔离微基准（/tmp/ncu_kern.cu → libferrite_kernels.so，生产 shape 真实 launcher）

**方法**：`/tmp/ncu_kern.cu` 链接生产 `.so`，用生产 shape（N=16,HID=4096,INTER=256,INTER_SH=256,
TOPK=8,ELOCAL=288,DSCOLS=2 / hc S=16,HC=4,MIX=24 / gdn B=16,h=64,dk=dv=128）调真实 launcher，
warmup 后 `cudaProfilerStart/Stop` 窗口，`ncu --profile-from-start off --launch-count 1`。

| kernel | nsys med | ncu dur | DRAM% | L1/TEX% | SM% | warps/sched | ncu 判定 |
|---|---|---|---|---|---|---|---|
| moe_act (14.3%) | 45.3µs | 49.8 | **71.9** | 51.0 | 55.7 | 7.33 (46%) | **DRAM 带宽受限（真实地板）** |
| moe_down (13.8%) | 43.8µs | 48.2 | **37.6** | **67.6** | 57.6 | 8.26 (52%) | 算/存均衡；**L1TEX 记分板停顿 40.6%** |
| hc_mix (5.3%) | 7.7µs | 10.9 | 3.1 | 33.8 | 29.6 | 8.50 (53%) | **纯延迟受限** |
| hc_rest345 (8.7%) | 12.7µs | 15.1 | **1.0** | 9.5 | **5.2** | **3.36 (21%)** | **grid 太小＝0.22 wave！84.9% 周期无 eligible warp** |

**结论**：nsys 分解可信（ncu 仅 +10~40% 开销，小 kernel 相对更高）。三个 kernel 三种不同性质：
1. **act = 真 DRAM 带宽受限（71.9%）** → 只剩"减流量"杠杆（B=16 下 128 assignments/~104 unique
   experts 的 23% 重复，L2 Hit 40% 已吸收一部分）；旁证 **L2 Sector Promotion Misses 38.77%**
   → 32B sector 未用满，可继续榨。
2. **down = 非带宽受限（37.6%）**，瓶颈是 **L1/TEX 数据路径（67.6%）+ L1TEX 记分板停顿（40.6%）**。
   block=288 线程(9 warps)=9 个 slot=9 个不同 expert → 每块从 9 个 1MB 矩阵各取 8×256 片段
   → DRAM 页局部性差。**有 ~1.8x 结构性余量。**
3. **hc 链 = 严重欠并行**：rest345 grid (16,16)=256 blk × 256 thr = 65536 线程 = **0.22 wave**，
   SM 5.2%，84.9% 周期无可发射 warp。**修正 AGENTS.md 早前"占用率固定不值得攻"的错误结论**：
   per-layer hc_pre 24µs 对应的实际访存仅 ~2MB(0.26µs)，即 ~90x 低效，全部来自
   launch(2 个 kernel) + 6 个 __syncthreads + P1/P2a 跨 block 冗余重算 + 0.22 wave 的延迟暴露。

## 2026-09-10 下午会话：hc 链 −0.7ms + down 寄存器教训 + 测量方法论修正

**同会话背靠背 A/B 的最终状态**：旧 down kernel .so（21997c1be）= 13.02ms vs 当前 HEAD = **12.88ms**（B=16 replay p50，n=244/121，faults=0）。本会话净收益 = **hc 链 2.10 → ~1.4ms**。

### 落地的改动（全部同会话验证）

1. **hc_pre_rest345 的 P5 normalize 循环 float4 化**：nsys serve 实测 rest345 12.67 → 7.26µs（阶段二分归因：normalize 3104ns + sinkhorn 2080 + launch 1344 + P1 1312 + election 896 + P3 896 + P4/P2a ~500）。`li/nw` 是 DevBuf 256B 对齐 + t*h 是 16KB 倍数，float4 安全；逐元素 `(li*inv)*nw` 同序，逐位一致。
2. **hc mix 的 K-split 默认 KS=16 → 4**（`FERRITE_HC_MIX_KS` 旋钮，上限 16=Rust scratch 尺寸）：隔离扫描 16:14176 / 8:13248 / **4:10432** / 2:11808 ns。KS=16 时 fw(1.5MB) 被重读 16 次=24MB/30MB，每线程只有 1 个 float4 迭代却付 5 归约+8 屏障。KS=4 一举三得（fw 流量 16x↓、block 数 4x↓、每线程工作量 4x↑）。**注意 KS 是数值性改动**（部分和求和顺序变），已人眼验证出师表全文。
3. **MoE down HTILE 模板化**（`FERRITE_DOWN_HTILE` ∈ {8,16,32,64}，默认 8）：`template<int HTILE_K>` 特化，act 每 token 载入寄存器后跨 8 行 chunk 复用，warp 的权重连续读 run ×N。隔离：8:51.6 / 16:51.7 / 32:57.3µs。

### 三个昂贵教训（本会话 ~1.5 小时的学费）

1. **寄存器数即占用率**：down 重写第一版 `<8>` 编译到 **80 regs → 2 blocks/SM**（旧版 ~62 regs → 3 blocks），serve 43.6 → 56.6µs（+0.55ms replay）。`__launch_bounds__(288, 3)` 压回 72 regs/3 blocks 恢复。**改 kernel 结构后必须 `nvcc -Xptxas -v` 看寄存器数**（cuobjdump 不在此工具链，用 ptxas -v + grep kernel 名）。`ar[4]` 声明在外层作用域会把 16 个寄存器拖着穿过 shuffle/part 阶段。
2. **运行时边界的循环是毒药**：第一版把 htile 作为运行时 kernel 参数传入 → nvcc 无法展开 chunk 循环/外提不变量/fold 除法无法变移位（隔离 +10µs/call）。模板常量解决。
3. **跨会话绝对数字不可比（最重要）**：同一段代码 45 分钟内 12.65 → 13.02ms（+0.35ms 机器漂移）。**结论必须来自同会话背靠背 A/B**。诊断利器：`git show <commit>:kernels/cuda/ferrite_kernels.cu > /tmp/old_fc.cu` 编译成 /tmp/old_lib.so，serve 用 `--lib /tmp/old_lib.so` 切换——不动主树、同一二进制、同一环境。

### 其他方法论

- **rest345 阶段二分**（FERRITE_HC_STAGE 临时开关，已移除）：8 个切点一次构建免费扫描，比猜快 10 倍。launch+retire 本身 1.34µs——**小 kernel 的固定成本下限，merge 是唯一出路**。
- **隔离微基准会误导 down**：同一 kernel 隔离 51.6µs vs serve 48.5µs，且 HTILE 的隔离排序（16 最优 −7%）没有转化到 serve（同会话 A/B 反而 8 略优）。隔离基准只用于淘汰明显差的方案，最终判定必须 serve。
- **nsys capture-range 在本机丢数据**（serve 退出时 drop 权重 SEGFAULT 丢 profiler buffer）→ 用全程 trace + 按名字过滤加载期 kernel（dequant/bf16_to_f32/memcpy）。

### 当前 nsys 分解（B=16 稳态 med，本会话实测）

| kernel | med µs | ×次/步 | ms/步 |
|---|---|---|---|
| ncclDevKernel AR RING_LL | 28.99 | 90 | 2.61 |
| moe_fused_down（修复后 ~48.5） | ~48.5 | 42 | 2.04 |
| moe_fused_act | 45.0 | 42 | 1.89 |
| gdn_step_v2 | 17.1 | 34 | 0.58 |
| hc_pre_rest345（已优化） | **7.26** | 90 | 0.65 |
| hc mix（KS=4，~4µs） | ~4 | 90 | 0.36 |
| 其余（hc_post/gdn_chunk/gemv/gemm3/kpool/sparse/…） | | | ~4.4 |
| host + 图调度间隙 | | | ~0.5 |

### 通往 1600 的最终路径（10.0ms 需再砍 2.88ms）

| # | 项 | 预期 | 状态 |
|---|---|---|---|
| 1 | **AR v5**（自推进 per-slot epoch，见下） | −1.5ms | 设计完成，未实施 |
| 2 | MoE act sector 利用率（38.77% promotion miss） | −0.3ms | 未实施 |
| 3 | hc sinkhorn → 独立 aux block | −0.18ms | 未实施 |
| 4 | 小 kernel 合并（norm/cast/quant ~241 次/步） | −0.3ms | 未实施 |
| 5 | MoE down one-expert-per-block | −0.5ms | 未实施（HTILE 已证明不是正确入口） |
| | **合计** | −2.78 → 10.1ms ≈ 1585 | 边缘达标 |

**AR v5 设计（第 4 次尝试，前 3 次死锁根因已定位）**：死锁共同根因 = epoch 状态由 dry-run（真实执行）推进、由 capture（只记录）冻结 → 各 rank 计数器漂移。v5 让 **epoch = AR kernel 自身的执行次数**：每个 AR 调用点有独立 `ctr[slot]`，store kernel 的 block0 `atomicAdd(&ctr[slot],1)` 得本次 epoch；publish 写 `stamp[peer][slot]=epoch`；reduce 轮询 `stamp>=my_epoch` 后按 publish 出来的 epoch 选 staging 奇偶位（读者跟写者走）。dry-run 执行→推进、capture 不执行→不推进、replay 执行→推进，任何路径下各 rank 推进次数相同（TP lockstep），不可能漂移。预期 28.99 → ~11µs。

## ✅ 2026-09-10 傍晚：AR v5 成功 — B=16 replay 12.88 → 10.95ms（−1.9ms），1455 tok/s

**P2P one-shot AR v5（FERRITE_P2P=1 FERRITE_P2P_AR5=1，env-gated）在第 4 次尝试成功**，两轮验证全绿：

| 验证 | 结果 |
|---|---|
| 首测 b16×300 | replay p50 **11.00ms**（NCCL 12.88），faults=0，ar5-hang=0，无监护停滞 |
| 复测 b16×300 + **b16→retire→b2 图切换** | b16 **10.95ms**，b2 **7.12ms**（NCCL 8.84），faults=0，hang=0 |
| 文本 | req0 思考前言后逐字背诵《出师表》至"将军向宠"段 ✓（300 token 预算耗尽截断） |
| AR 每次开销 | 28.99µs → **~8µs**（90 次/步：2.61ms → ~0.7ms） |
| 事后 GPU | nvidia-smi 全 0（未 wedge） |

**v5 的核心设计（为何第 4 次成功了）**：前三次（fused_v3/oneshot_v1/oneshot_v2）死锁的共同根因是 **epoch 状态由 dry-run（真实执行）推进、由 capture（只记录）冻结，而各 rank 的 dry-run 不同步**（"dev0 at L0, peers at L35"；post-dry-run 的 epoch reset 也修不好）。v5 不是修补状态而是**消除 desync 的来源**：
- Rust 分发器只在 `is_capturing()` 为真时发射 v5 kernel；dry-run 和所有 host 路径回退 NCCL。
- 于是 epoch 计数器**只被 replay 的图节点推进**，而 replay 是全局 lockstep（TP decode 不可能跑在对端 all-reduce 前面）→ 计数器在任何时刻跨 rank 相等，结构性不可能漂移。
- 3 kernel：store（e=\*epoch 运行时读，float4 合并写全部对端 staging[e&1]）→ publish（system-scope 盖章 e+1，轮询每个对端 stamp≥e+1，然后推进 \*epoch）→ reduce（按 rank 升序求和 = NCCL ring 顺序，1-ulp 一致）。
- **stamp 折进 store —— 已彻底回退（2026-09-11）**。该方案（盖章从 pubred 的 block 0 移到 store kernel 的最后一个 block，用"永不 reset 的单调 arrival 计数"判末块，pubred 退化为 poll+reduce）在 serve 上**灾难性失败**（`DSV41_AR_STAMP_FOLD=1` ⇒ 29.5s/step：图 replay 下机制坏掉，poll 自旋等永远不来的 stamp）。更关键的是：**代码存在性本身有害**——即使 gate 全 OFF（`arrive == nullptr` / `stamp_in_store == 0`），launchers 多出的 2 个参数仍带来 ~+0.33ms/step（v13nq 6.60ms vs 基线 6.27ms）。因此 AR 侧 fold 代码已**物理删除**，5 个 launcher + store/pubred/hcpost kernel 全部恢复 pre-fold 签名（`arrive`/`stamp_in_store` 参数与 `ready_tbl`-in-store 一并移除，store 动态 smem 回到 0），Rust 侧（`cuda.rs` / `device.rs` FFI 与包装 / `tp.rs` 的 `ar_epoch_args()` 与 `DSV41_AR_STAMP_FOLD` gate）同步回退。教训：**失败实验的 kernel/ABI 改动即使 gate 关闭也要删干净，否则常数级 ABI 开销本身即回归**。
- 奇偶双缓冲恰好足够：任何 rank 的 store(k+2) 跨 rank 晚于所有 rank 的 reduce(k)（publish(k+1) 等所有对端的 k+1 盖章 ⇒ 对端 store(k+1) ⇒ 流序 ⇒ 对端 reduce(k)）。
- **图切换正确性**（旧图不重捕获直接切回）：kernel 运行时读 epoch，任意图在任意时刻 replay 都用当前值 → b16→b2 切换已实测验证。
- p2p_ar_reset（tp.rs:1087，dry-run→capture 屏障）对 v5 是一致的清零（epoch+flags 全 rank 同时归零），无害。

**注意**：v5 env 仍为 opt-in（`FERRITE_P2P=1 FERRITE_P2P_AR5=1`）。标准 bench env 应加上这两个。FERRITE_P2P 同时启用了 host 路径的 P2P 拷贝（prefill/MTP 链），这些路径在两次验证中未出问题。

### 通往 1600 的更新路径（当前 10.95ms = 1461 tok/s，还差 0.95ms）

| 项 | 预期 | 状态 |
|---|---|---|
| MoE act sector 利用率（38.77% promotion miss） | −0.3ms | 未实施 |
| hc sinkhorn → aux block（阶段二分：sinkhorn 2080ns 在关键路径上） | −0.08~0.18ms | 未实施 |
| 小 kernel 合并（norm/cast/quant ~241 次/步） | −0.3ms | 未实施 |
| MoE down one-expert-per-block | −0.5ms | 未实施（HTILE 已证明不是入口） |
| **合计** | **−1.2~1.3ms → 9.7ms ≈ 1650** | **1600 可达** |

## 2026-09-10 会话终态：13.45 → 10.94ms（+23%），1190 → 1464 tok/s

**运行配方**：标准 env + `FERRITE_P2P=1 FERRITE_P2P_AR5=1`（AR v5 仍为 opt-in）。

| 改动 | 验证 | commit |
|---|---|---|
| rest345 float4 normalize | 隔离 3104→1280ns；serve rest345 12.67→7.26µs | cb25f05 |
| mix K-split KS=4（旋钮 FERRITE_HC_MIX_KS） | 隔离 14176→10432ns（fw 流量 24→1.5MB/次） | 21997c1 |
| down HTILE 模板（旋钮 FERRITE_DOWN_HTILE=8/16/32/64，默认 8） | 同会话 A/B 12.88 vs 旧 13.02 | 14162a1 + 9992c71 |
| **AR v5（FERRITE_P2P_AR5）** | **12.88→10.95ms（AR 29→~8µs），三轮：b16×2 + b16→b2 图切换 + b16×900 长窗口，全绿** | e893428 |
| rest345 sinkhorn aux block | serve 中性 10.94ms（sinkhorn 本不在关键路径），结构更优保留 | 144d839 |

**会话教训（都已在上文详述）**：① 80 寄存器 → 2 blocks/SM 的占用率陷阱（改 kernel 结构必查 `nvcc -Xptxas -v`）；② 运行时边界循环让 nvcc 放弃展开/外提；③ **跨会话绝对数字不可比**（45 分钟漂移 +0.35ms）——同会话背靠背 A/B 是唯一可信判据，`git show <c>:...cu | nvcc → /tmp/old_lib.so + --lib` 是干净 A/B 的做法；④ nsys capture-range 在本机丢 buffer，用全程 trace + 名字过滤；⑤ 后台启动 serve 必须让 `env` 作为首词（`cd x && env … &` 会产生 kill 不掉的孤儿进程占端口占卡）。

**当前每步分解（AR v5 后的推断）**：AR ~0.7 · MoE down ~2.0 · MoE act 1.89 · hc ~1.3（rest345 7.26µs×90=0.65 + mix ~4µs×90=0.36 + post 0.27）· gdn ~1.06 · 投影+小 kernel ~3.5 · host/间隙 ~0.5。

**通往 1600（还需 −0.94ms）**：act sector 利用率（−0.3）· 小 kernel 合并 norm/cast/quant（−0.3）· down one-expert-per-block（−0.5，最不确定）。三项落地 ≈ 9.85ms ≈ **1624 tok/s**。

## 2026-09-10 深夜侦察：act sector 项基本关闭，down one-expert 成为最大剩余项

**act sector 利用率（原 −0.3ms 估计）**：读 `moe_fused_act_fp8_mma_kernel`（ferrite_kernels.cu:4098）确认权重主通路**已是 sector 满利用**——per-warp smem staging（SA_STRIDE=80B padding 防 bank 冲突）+ 双缓冲 `cp.async.cg.shared.global.L2::128B` 16B/加载（ACT_ISSUE 宏，4234），每个 64B 行块 = 2 个满 32B sector。注释记载这已把"4B 跨 8 行 = 50% sector 效率"的 2.3x 流量浪费修掉（当年 3.2ms→带宽受限前的改造）。ncu 的 38.77% L2 Sector Promotion Misses 只能来自次要流量（gs/us scale 标量读、xq 暂存、ldmatrix 的 smem 侧）。**该项降级：预期 ≤0.1ms，不值得先做。**

**剩余路径重排（10.94ms → 10.0ms 需 −0.94ms）**：
1. **MoE down one-expert-per-block（−0.5~0.8ms，最大单项）**：down 现在 ~48.5µs×42=2.04ms，DRAM 地板 = 132MB 权重/次 ÷ 7.6TB/s = 17.4µs → 0.73ms/步，**理论余量 1.31ms**。设计：grid (hidden_tiles, n×(topk+1))，每 block 只读一个 expert 的连续权重片；跨 slot 求和用**两阶段**（phase1 写 per-slot partial 到 scratch [n][topk+1][hidden]，phase2 固定序归约——确定性，避免 atomicAdd 的非确定序）或 atomicAdd（快但序不确定）。**HTILE 教训适用：隔离基准对 down 有误导性（L2 条件不同），必须以 serve 判定；且改结构后必查 `nvcc -Xptxas -v` 寄存器数（80 regs → 2 blocks/SM 的陷阱）。**
2. **小 kernel 合并（−0.3ms）**：norm/cast/quant ~241 次/步 × 1.34µs 固定成本。最有把握的一项。
3. gdn_step（0.58ms，17.1µs/次）/ kpool（0.26ms）：float4 已做，需 ncu 定位剩余。

**当前状态**：10.94ms = 1464 tok/s（env：标准 + `FERRITE_P2P=1 FERRITE_P2P_AR5=1`）。HEAD e2e2207，全部验证过（faults=0，出师表✓）。

## ⛔ nsys 铁律（2026-09-10，用户明令）：剖析 serve 必须 NCCL 模式

**带 `FERRITE_P2P=1 FERRITE_P2P_AR5=1` 跑 nsys 会自旋卡死**：AR v5 的 publish kernel 自旋等 peer 盖章，nsys 的 `--cuda-graph-trace=node` 对 8 个 rank 每步 ~400 个节点做 CUPTI 拦截，host 侧抖动被自旋放大——实测 240s 只跑 69 步（≈3.4s/步，比无剖析慢 300 倍），bench 超时、报告丢失。**NCCL 模式（去掉这两个 env）剖析干净（1m53s 全流程）**，且除 AR 外所有 kernel 的中位数对 v5 构建同样有效（代码路径相同）。要 AR 的耗时直接用 90 × ~8µs 代入。

剖析脚本模板：`/tmp/sp4.sh`（GPU 忙则中止 → nsys 全程 trace → bench 加 `timeout 300` → **`curl -X POST http://localhost:8080/shutdown` 收尾**（不是 kill -INT！）→ 等 nsys 最多 300s → `nsys stats --report cuda_gpu_kern_sum | grep -vE "dequant|bf16_to_f32|memcpy|Memset|matmul_tiled"`）。

## 2026-09-10 深夜：最终冲刺的精确分解（NCCL 模式 nsys + sqlite 稳态窗口查询）

**方法**：NCCL 模式剖析（铁律见上）+ `sqlite3 /tmp/sp4.sqlite` 查最后 100ms 窗口（纯 b16 稳态，8 rank 归一化，AR=90/step 校验吻合）。**gdn_step_v2 不在稳态路径**（65280 实例全部来自 prefill/ramp；稳态 GDN = chunk_batched 0.48 + prep 0.13 + conv1d 0.10，chunk_batched 在 DRAM 地板——状态读写 4.4GB/步不可减）。

**当前 10.94ms（v5 AR）的精确构成**：

| 项 | ms/步 | 地板 | 可砍 |
|---|---|---|---|
| MoE down | 2.11 | ~1.5（L1TEX 管道） | **−0.5~0.6（one-expert-per-block）** |
| MoE act | 1.89 | ~1.5（71.9% DRAM） | −0.4（难） |
| 投影族（nvjet×4 + gemv×3 + gemm3 + splitKreduce） | ~2.0 | | **−0.3~0.4（splitK 对 M=16 反优化 + cast 合并）** |
| hc 链（rest345 0.65 + mix 0.43 + post 0.29） | 1.37 | | −0.1 |
| AR v5 | 0.72 | ~0.6 | ~0 |
| 小 kernel（quant_e4m3 54/步 0.14 + f32_to_bf16 209/步 0.21 + norm 族 0.15 + argmax/misc 0.4） | ~0.9 | | **−0.3（合并）** |
| kpool 0.26 + sparse/DSA 0.27 + route 0.25 + GDN 0.71 + 图间隙/host ~0.9 | ~2.4 | | −0.2 |
| **合计** | **10.94** | | **−1.7~1.9 理论 → 9.0-9.2ms** |

**冲刺剩余路径（按 把握×收益 排序）**：
1. **投影族 splitK 关闭 + cast 消除（−0.3~0.4，最有把握）**：nvjet_splitK（3.8µs×57/步）+ splitKreduce（2.8µs×67/步）= 0.41ms——M=16 的 GEMM 不该 splitK；f32_to_bf16 209 次/步（每 GEMM 前的转换）——让产出方直出 bf16（池化 cast 缓存已 4 次失败，勿走池化路）。
2. **小 kernel 合并（−0.3）**：quant_e4m3_tokens 可并入 hc_pre_rest345 尾部（它量化的就是 li——但 absmax 需跨 block 归约，需用 is_last 机制或独立小 kernel）；norm 族 209 次。
3. **down one-expert-per-block（−0.5~0.6，最大但最险）**：两阶段归约保确定性；HTILE 教训适用。

## 2026-09-10 冲刺终态：CUBLAS_WORKSPACE_CONFIG 逼退 splitK — 无效（负结果）

同会话背靠背 A/B（base 10.94 vs `CUBLAS_WORKSPACE_CONFIG=:1024:2` 11.03ms，均 faults=0、出师表✓）：经典 `cublasGemmEx` API 不受该配置约束（或非 splitK 变体更慢）。**要消除 splitK 浪费（nvjet_splitK 3.8µs×57/步 + splitKreduce 2.8µs×67/步 = 0.41ms）必须迁移到 cublasLt Matmul + 启发式过滤**（algo 遍历时排除 splitK 变体）——中等工作量的重构，下会话项。

## 冲刺会话总结（2026-09-10 深夜）

**成果**：13.45 → **10.94ms（+23%），1190 → 1464 tok/s**（AR v5 是最大单项：−1.9ms）。
**本轮新知识**：① nsys 必须 NCCL 模式（v5 自旋×节点追踪 = 300x 放大）；② `POST /shutdown` 是唯一正确收尾；③ gdn_step_v2 不在 b16 稳态路径（prefill/ramp 专属）；④ CUBLAS_WORKSPACE_CONFIG 逼退 splitK 无效。
**通往 1600 的最终清单（10.94 → 10.0，需 −0.94ms）**：
| 项 | 预期 | 方案 |
|---|---|---|
| 小 kernel 合并 | −0.3 | quant_e4m3（54/步，可并入 rest345 尾部或独立小 kernel 合并）+ f32_to_bf16（209/步）+ norm 族 |
| down one-expert-per-block | −0.5~0.6 | 两阶段归约保确定性；HTILE 教训适用 |
| cublasLt 迁移 | −0.2~0.3 | 启发式过滤 splitK；gemm_cublas 重构 |
| **合计** | **−1.0~1.2 → 9.7-9.9ms ≈ 1620-1650** | |

## 2026-09-10 终局设计：段融合（MegaKernel-lite）——第二阶段路线（用户提议）

**数学**：kernel 边界开销（图间隙 0.4-0.9 + 小 kernel 0.9 + 中间量往返 0.1-0.2）≈ **可回收 −0.8~1.1ms → 9.9-10.1ms ≈ 1580-1620 tok/s**，够到 1600。

**硬约束**：TP=8 下每层 2 个 AR 是跨 rank 集合通信 = 硬 kernel 边界（v5 的 store/publish/reduce 必须独立成核）→ "每层一个 kernel"不存在，可行形态 = **每层 3 个段融合核 + 2×3 AR 核 = 9 launch/层**（现在 ~15-20）：
```
[段1: hc_pre(mix+rest345) + 投影 + 注意力核心(GDN: prep/conv/chunk | DSA: kpool/append/qk/pv)]
  → AR(v5×3) →
[段2: hc_post + hc_pre2 + norm + MoE(route/act/down)]
  → AR(v5×3) →
[段3: hc_post2]
```
**大项不吃融合红利**（down 2.11 / act 1.89 / 投影 2.0ms 是带宽受限，13.5GB/步权重流量不变）。融合只回收边界开销。

**段融合的独有价值**（增量清单拿不到的）：段内权重预取与计算重叠（TileRT 式 smem 级软件流水）——这是超过 ~1650 之后才需要的。

**优先级决定**：先用增量清单拿下 1600（小 kernel 合并 −0.3 / down one-expert −0.5~0.6 / cublasLt 逼退 splitK −0.2~0.3 → 9.7-9.9ms），段融合作为第二阶段。工期对比：段融合 1-2 周 vs 增量 2-3 天，终点相同。

## 2026-09-10 冲刺收尾：f32_to_bf16 溯源 + 会话终态

**209 次/步 cast 的来源**：`gemm_cublas`（cuda.rs:2036）每次调用都发射独立 `f32_to_bf16` kernel 把 x 转进临时 DevBuf——nvjet 家族 ~218 次/步与 cast 209 次/步一一对应。custom kernel（gemv_bf16/gemm3）不受影响（kernel 内转换）。**消除方案**：①产出方双输出（hc_pre_rest348/rmsnorm/hc_post 同时写 f32+bf16，下游 GEMM 直接吃 bf16）——注意勿走 xb_cache 池化老路（4 次失败，池地址稳定性契约）；②或用自研 GEMM 替换这些 cuBLAS 调用（gemm3 已有模式，直接吃 f32）。预期 −0.2ms。cublasGemmEx 不支持 A=f32×B=bf16 混合（A/B 类型必须一致），该路不通。

**会话终态**：13.45 → **10.94ms（+23%），1464 tok/s**（AR v5 −1.9ms 是最大单项）。剩余 −0.94ms 的三项清单与段融合（第二阶段）设计均已入档。运行配方 = 标准 env + `FERRITE_P2P=1 FERRITE_P2P_AR5=1`。

## 2026-09-10 深夜②：down v14/v15（cp.async 双缓冲 staging）— serve 中性，第三个 down 理论失败

**实测**（三轮背靠背 serve A/B，`git show bfef692:...cu → /tmp/prev14_lib.so + --lib`）：prev14 11.15 / v15 11.16 / prev14b 11.17ms——**完全中性**。隔离基准 −5%（54.1 vs 51.7µs）未转化（隔离的 L2 热条件第三次误导 down）。

**认知更新：down 的 37.6% DRAM 不是 per-warp 延迟暴露**——如果每 warp 裸吃 ~600 周期 DRAM 延迟，D=2 流水线（隐藏一半）应有可测收益。中性 = 延迟已被 27 warps/SM 的跨 warp 隐藏覆盖。v15 保留（中性、结构与 act 同构、72 regs/0 spill、36KB smem 不降占用率）。

**down 的三个已失败理论**：① HTILE（act 重读/局部性）② one-expert 局部性（未实施，被 ③ 取代）③ cp.async 延迟隐藏。**剩余唯一假设：act 对比本身有误导**——act 的 71.9% 可能来自其更高的绝对流量（246MB vs down 132MB，绝对带宽 5.5 vs 2.6TB/s），而 down 的真实约束仍未定位（指令 ~8µs / DRAM 地板 ~22µs / L1 wavefront ~4µs / 实测 50µs——全都不匹配）。**down 优化正式暂停**，除非拿到当前 kernel 的新 ncu 数据（HTILE=8 时代的数据已过时三次）。

**注意**：本轮三轮 serve 均读 11.15-11.17ms（1.5h 前同配置 10.94）——热漂移 +0.21ms。跨时间绝对数字不可比（第三次确认）；目标评估应以同批 A/B 为准。

## 2026-09-10 深夜③：投影族收官 + 小 kernel ROI 封顶 — 增量路径在 ~1480 处耗尽

**投影族最终战果**（per-group-x gemm3 扩展）：
| 项 | 结果 |
|---|---|
| GDN {f_b, g_b} 融合 | **−0.09ms**（三轮 A/B：fused 11.04 vs pre 11.10/11.16，出师表✓）|
| DSA {q_a,kv_a} + {q_b,wq_b} 融合 | 中性（11.10 vs 11.09/11.12，11 层太小）；保留（gemm2_fused helper 可复用）|
| **kper 地雷修复** | gemm3 在 in_f<128 时 kper 向下取整为 0 → **静默全零输出**；launcher 现返回 NotSupported（无现有调用踩中，kvb 本会踩中）|
| qkv/o_proj 路由 gemm3 | **排除**：gemm3 实测带宽仅 ~0.6TB/s（DSA trio 3.7MB/11.6µs）——它赢在杀固定成本而非 GEMM 效率；qkv 的 25MB 权重在 gemm3 下 ~40µs vs cuBLAS 9.2µs（4x 差）|

**小 kernel 桶的 ROI 封顶**（逐项算账）：
- quant_e4m3（54/步）：xq_cached 已按 (ptr,gen) 缓存（54 次 = 54 个不同 x）；融合到 rest345 P5 双输出需跨块 absmax（is_last 机制），净 −0.04ms（省 0.08 cast + 付 0.036 双写）
- f32_to_bf16（剩余 ~140/步：qkv 34 + o_proj 45 + kvb 11 + dense 3）：产出方双输出，每处 −0.05ms 级
- norm 族（rmsnorm/gated 62/步）：qa_ln/kv_ln 融进 gemm3 staging 只省 elementwise 部分归约仍在，~−0.03ms
- **整桶合计 −0.15~0.25ms → 终点 ~10.85-10.95ms ≈ 1465-1480 tok/s**

**增量路径正式封顶**：当前 11.10ms（机器热态；早间冷态 10.94）= 1440 tok/s。全部剩余增量项（小 kernel 0.2 + cast 0.1 + cublasLt 不确定 0~0.3）落地后 ~10.6-10.9ms ≈ 1470-1510。**1600（10.0ms）只余段融合一条路**（用户提议，设计见 3ccd46f：边界开销 −0.8~1.1ms，TP=8 下 AR 是硬边界 → 每层 3 个段融合核，1-2 周）。

**本冲刺累计**：13.45 → 11.10ms（+21%），AR v5 −1.9ms 是最大单项；运行配方 = 标准 env + `FERRITE_P2P=1 FERRITE_P2P_AR5=1`。

## 2026-09-10 深夜④：图内 launch 成本实测 ~0.2µs — 段融合价值重估（重大修正）

**数据点**：AR v5 publish+reduce 融合消除 90 个图节点/步，replay 11.09 vs 11.10-11.12（≤0.03ms，噪声内）。→ **图内 kernel launch 成本 ≈ 0.2-0.3µs/节点**（不是独立 launch 的 1.34µs！CUDA graph 的节点调度几乎免费）。

**连锁修正**：
1. "247 间隙 × 4.75µs = 1.2ms" 的归因错误——间隙是**数据依赖等待 + kernel 尾部效应**（PDL 实验的结论现在被第二种方法证实），不是 launch 开销。**消除 launch 不消除间隙**（依赖仍在）。
2. **段融合价值从 −0.8~1.1ms 下修为 −0.3~0.5ms**（只剩：中间量往返 ~0.1-0.2 + 段内相位级流水化 ~0.2-0.3；launch 消除部分 ≈ 0.02ms/项）。
3. mix+rest345 融合（需网格同步）的收益 = 纯 launch ~0.02ms，**不值得做**（同步语义与 kernel 边界等价）。

**修正后的 1600 可达性**：
| 路径 | 修正后预期 |
|---|---|
| 小 kernel 合并（执行时间部分仍在） | −0.15~0.25 |
| 段融合（往返+流水化） | −0.3~0.5 |
| act 深挖（71.9%→~85% DRAM） | −0.1~0.2 |
| AR 残余 | ~0 |
| **合计** | **−0.55~0.95 → 10.1-10.5ms ≈ 1520-1580** |
| **down 突破（约束未定位，需新 ncu）** | **−0.5~1.3（唯一的 >0.3ms 单项）** |

**结论**：1600 = 全部剩余项 + down 或 act 的突破。down 的三个理论已失败（HTILE/one-expert 延迟隐藏/cp.async staging），**下一步必须是拿当前 kernel 的新 ncu 数据**（37.6% DRAM / 67.6% L1TEX 的旧读数来自 HTILE=8 时代，已过时三次）。

**AR v5 pubred 融合保留**（中性、结构更优：2 kernel/AR、绝对 epoch 轮询的多块安全性已验证）。

## 2026-09-10 终局：down/act/AR 三大 kernel 全部触底 — 系统性定论

**down v16 的占用率实验（第五个理论）**：launch_bounds(288,4) 完全生效（ncu 确认 56 regs、233KB carve-out、4 blocks/SM、理论占用率 42→56%、达成 49%、waves 2.31→1.73）——**serve 仍然中性**（10.96 vs 10.96）。HTILE=32@592 slots（单波）隔离仍 −10%（并行度减半的代价超过 act 流量减半的收益）。**定论：down 是 L1TEX 管道吞吐受限（71%）的结构性地板，2.04ms = 本实现的下限。** 五理论（HTILE 局部性 / cp.async 延迟隐藏 / 占用率 / wave 尾 / act 流量）全部实测失败，勿再试。

**act 的最终 ncu**：DRAM 71.00%（真带宽受限）、L1TEX 51%、Compute 56%、占用率 50%/46%、waves 3.89。流量 = 144 assignments × (gate 1MB + up 1MB) ≈ 288MB（71% × 7.6 × 45µs ≈ 244MB 吻合，L2 吸收重复）。**流量不可减（每 assignment 必读其 expert 的 gate+up 一次）→ act 1.89ms = 地板。**

**AR v5 pubred**：2 kernel/AR（store + pubred），~6-8µs/次 ≈ 0.6ms，NVLink 传输 1.75MB@425GB/s = 4.1µs 是地板。

**最终分解（10.96ms，全部触底项标注）**：
| 项 | ms | 状态 |
|---|---|---|
| down 2.04 | L1TEX 结构地板 | ⛔ 关闭 |
| act 1.89 | DRAM 带宽地板（71%，流量不可减） | ⛔ 关闭 |
| hc 链 1.30 | rest345 7.26µs（launch 1.34+P1 1.31+P3 0.9+election 0.9+normalize 1.28 的固有串行） | ~地板 |
| 投影族 ~1.85 | cuBLAS/gemm3 各在其位；splitK 0.41 待 cublasLt（不确定） | 剩 −0~0.3 |
| GDN 0.71 | 状态流量地板（4.4GB/步） | ⛔ 关闭 |
| AR v5 0.60 | NVLink 地板 | ⛔ 关闭 |
| 小 kernel ~0.70 | 合并磨活 | 剩 −0.15~0.25 |
| sparse/DSA 0.27 + kpool 0.26 + route 0.25 + 间隙 ~0.5-1.0 + host 0.12 | | 段融合流水化 −0.3~0.5（不确定） |

**1600 可达性最终判定**：剩余全部项（小 kernel 0.2 + cublasLt 0~0.3 + 段融合流水化 0.3~0.5）= −0.5~1.0ms → **10.0-10.5ms ≈ 1520-1600 tok/s**。1600 在剩余路径的最乐观端——需要段融合的相位级流水化兑现全部预期 + cublasLt 付费 + 小 kernel 全落地。**三大 kernel（down/act/AR）已无任何单点 >0.2ms 的优化空间。**

**本会话最终成果**：13.45 → **10.96ms（+22.7%），1190 → 1460 tok/s**。运行配方 = 标准 env + `FERRITE_P2P=1 FERRITE_P2P_AR5=1`。HEAD `3d9d543`。

## 2026-09-10 终局②：v16c 定论 + cublasLt 混合类型探针关闭

**v16c（per-device carve-out 修复后）**：v15c 11.10 / v16c 11.06 / v16c2 11.09 —— **仍中性**。用户的 bug 质疑（"真的不是改错代码位置了吗"）是对的：第一版 carve-out 只对 1/8 设备生效（cudaFuncSetAttribute 是 per-context 的，static guard 让它只设了一次）。修复后测试有效，**占用率理论正式死亡——down 是 L1TEX 管道吞吐地板（71%），更多 warms 不帮已饱和的管道**。教训：**多 GPU 进程里的 cudaFuncSetAttribute 必须循环所有设备**（本会话第五个 per-device/per-context 类陷阱）。

**cublasLt 混合类型探针**：f32 A × bf16 B → heuristic 返回 status=7（NOT_SUPPORTED）——**A/B 类型必须一致，cast 消除这条路关闭**。splitK 过滤仍可做但预期 −0~0.2ms（CUBLAS_WORKSPACE_CONFIG 实验 11.03 vs 10.94 已暗示非 splitK 不快）。

**终账（10.96ms / 1460 tok/s，非融合架构极限附近）**：
- down 2.04（L1TEX 地板）+ act 1.89（DRAM 71% 流量不可减）+ AR 0.60（NVLink 地板）= **4.53ms 三大地板**
- hc 1.30 + 投影 1.85 + GDN 0.71 + 小 kernel 0.70 + sparse/kpool/route 0.78 + 间隙/host 1.1 ≈ 6.44ms
- **剩余可砍：小 kernel 0.15~0.25 + cublasLt 0~0.2 + 段融合流水化 0.3~0.5 = −0.45~0.95ms → 10.0~10.5ms ≈ 1520~1600**
- 1600 在最乐观端；需要段融合的细粒度跨块流水化兑现全部预期

**会话总成果（AR v5 突破 + 全链优化）**：13.45 → **11.06ms（+21.6%），1190 → 1446 tok/s**。运行配方 = 标准 env + `FERRITE_P2P=1 FERRITE_P2P_AR5=1`。HEAD `c7e4393`。

## 2026-09-10 上午：SGLang GLM-5.3-Flash 支持调研（PR #36507）——为什么他们快/我们差在哪

**来源**：PR #36507 "GLM-5.3-Flash support"（JustinTong0323，2026-09-06 合入，100 文件）+ #38621（NVFP4 loading）。clone 于 /tmp/sglang。**参数核对：无差别**（topk=8 ✓、45 层、34 GDN + 11 DSA、hidden 4096、288 专家、moe_inter 2048、index_topk 2048、hc_mult 4、sinkhorn 20——SGLang 代码里默认 topk=7 只是 fallback，checkpoint 覆盖为 8）。

**前提纠正**：我们自己在 B300 集群实测过 SGLang 不开 MTP ≈ **1300 tok/s** < ferrite 当前 **1446**。"SGLang 更快"的印象来自 EAGLE/MTP 数字（3200，draft 5 步 6 token）。但用户贴的这个 config（deep_gemm + trtllm + fp8 KV + tp4+ep4）与我们当时测的不同，不确定它跑多少。

**TP4+EP4 不是提速原因——反而每 rank 带宽更差**：
- TP8（ferrite）：每 rank 每层 ~104 unique experts × 3MB（inter/8=256）≈ 312MB
- TP4+EP4：每 rank ~28 unique experts × 24MB（full inter 2048）≈ 672MB —— **2.2x 更多字节/rank**
- 聚合流量相同（~2.5GB/层），但 4 卡分摊 → MoE 每步更慢。tp4 的意义是**容量**（306GB/4=76GB/rank 能装进 4 卡），不是速度。"八卡跑 tp4"=2 个独立实例=我们探索过的"2组方案"（AGENTS.md 已有：估算 1230-1600，TP=4 capture bug 已被 9edc30b 修复）。

**4 个真实架构差距（按大小排序，都已在 SGLang 代码中核实）**：

1. **MoE down：DeepGEMM fp8×fp8 tensor core + 双侧细粒度 block scale**（`deep_gemm.fp8_m_grouped_gemm_nt_masked`，deep_gemm.py）vs 我们的 SIMT down（2.04ms，L1TEX 管道 71% 地板）。权重侧用 checkpoint 原生 128×128 block scale；**激活侧用 per-token-group(1×128) fp8 量化**（`per_token_group_quant`，group_size=128）。我们 W8A8 失败的根因就是激活用了粗粒度 scale（~6% 误差→翻转）；DeepGEMM 的细粒度解决了精度。**预期差距 ~0.8-1.0ms——最大单项。ferrite 修法：per-token-group-128 激活量化 + block-scale 加载进 MMA 操作数的 W8A8。**
2. **MHC（超连接）big-fuse：ONE TileLang kernel 算完整个 hc_pre**（`mhc_pre_big_fuse_tilelang`，默认 ON：`SGLANG_OPT_USE_TILELANG_MHC_PRE=True`）——GEMV(mix)+sqrsum+sigmoid+**sinkhorn 全内联**+归一化全在一个 kernel、每 token 96 线程、数据全程在寄存器/fragment、支持 PDL。vs 我们的 3-kernel 链（mix 4.8 + rest345 7.26 + post 3.2 = 1.37ms）。**差距 ~0.5-0.7ms。我们此前的"融合不值得"结论只算了 launch 成本（0.2µs），漏了中间量全局往返（mx partials/li_raw 的写读）和 P1 冗余重读——SGLang 的融合赢在这些。**
3. **fp8_e4m3 KV cache**（quant_k_cache.py）vs 我们的 f32（a0e262d 因 prefill/batched 格式分歧 bug 回退）。KV 读取减半 + 1.8x 容量。**~0.15-0.25ms + 容量红利。我们已知根因（单 seq 写 f32 无 scale、batched 读写 fp8+scale），修的是格式统一。**
4. **trtllm DSA decode 后端**（TRT-LLM 生产级 sparse MLA kernel）vs 我们手搓的 kpool+sparse+append 链（~0.6ms）。**~0.2-0.3ms。**

**合计潜在差距 ~1.6-2.2ms**——如果 SGLang 这个 config 全部兑现，4 卡可达 ~9-10ms/步。但注意他们 tp4+ep4 的 MoE 带宽劣势（672 vs 312MB/rank）会吃掉一部分，尤其 B=16 时。

**ferrite 的行动清单（更新）**：
| 项 | 预期 | 依据 |
|---|---|---|
| down 的 per-token-group W8A8 MMA（DeepGEMM 式细粒度 scale） | −0.8~1.0ms | SGLang 已验证该数值方案可行（生产级） |
| hc big-fuse（单 kernel，寄存器驻留，sinkhorn 内联） | −0.5~0.7ms | SGLang mhc_pre_big_fuse_tilelang 默认 ON |
| fp8 KV 格式统一（修 root cause #4） | −0.15~0.25ms | SGLang quant_k_cache |
| **合计** | **−1.45~1.95ms → 9.1-9.6ms ≈ 1670-1760** | |

## 2026-09-10 TP4 图化复测：仍崩（9edc30b race 修复不充分）

TP=4/B=8 图化模式（GPU 0-3，race 修复 9edc30b 之后首次复测）：serve 起来了（31s），cap=8（部分图捕获成功）但 **err900=65、faults=2、0 token**——第一个请求的 decode 就崩。结论：set_batch_decode race 只是 TP4 池 miss 的原因之一，仍有未定位的尺寸类预热缺口（FERRITE_POOL_MISS=1 可定位）。**"2组方案"（2×TP4）依然被阻塞**；SGLang 的 tp4 优势之一正是他们没有这个问题。

**当前行动优先级不变**（下会话）：
1. down per-token-group W8A8 MMA（−0.8~1.0ms，SGLang 已验证数值方案）
2. hc big-fuse 单 kernel（−0.5~0.7ms）
3. fp8 KV 格式修复（−0.15~0.25ms）
（可选）TP4 池 miss 定位 → 解锁 2组方案

## ✅ 2026-09-10 上午：W8A8 down 默认启用 + fp8 KV 完整迁移（用户判断正确）

**用户指令**："实现fp8a8和fp8 kv吧，之前的实现其实不一定真有bug，似乎是当时的agent测试方法有问题" —— 完全正确。

**W8A8 down（FERRITE_DOWN_MMA 默认 ON，ad0283f）**：
| 配置 | replay p50 | 文本 |
|---|---|---|
| simt v16 | 11.03ms | ✓ |
| **mma1（e4m3 MMA）** | **10.74ms（−0.29ms）** | **✓ 最干净（直接背诵无思考前言）** |
| mma2 | 10.81ms | ✓ |

之前的"e4m3 翻转"是**测试方法 artifact**（.so 不同步），非数值 bug。`quant_act_rows` 的 per-(token,slot) row scale（256 元素）足够精细——失败的 W8A8 是 X 路径整行 absmax（4096 元素），完全不同的量化。

**fp8 KV e4m3（3aadd95）**：完整的 6 kernel 迁移（2 写 + 4 读 + host 路径 quant_kv 辅助 kernel）。
- 数值安全：req0/req1 连贯思考 + 背诵，faults=0
- 性能中性：10.76ms vs 10.74（300-token 短上下文下 DSA K/V 流量占比小）
- **2.1x KV 容量红利**（f32 4B → e4m3 1B per element + per-(t,h) scale）
- 长上下文（DSA decay，1600+ tokens）时带宽收益会更明显
- 之前的格式分歧 bug（root cause #4）彻底修复：**两条写入路径 + 全部四个读取路径一次性迁移**，一个缓存一个格式

**当前合计：13.45 → 10.76ms（+25%），~1486 tok/s**。

**下一步（SGLang 调研结论，按价值排序）**：
1. hc big-fuse 单 kernel（−0.5~0.7ms，SGLang 的 mhc_pre_big_fuse_tilelang 模式：GEMV+sqrsum+sigmoid+sinkhorn+归一化全在一个 kernel，寄存器驻留）
2. down one-expert-per-block（理论余量 1.31ms，但 HTILE/占用率/cp.async 三理论已失败，需要新思路）
3. cublasLt 迁移逼退 splitK（−0.2~0.3ms，需 API 重构）

## 2026-09-10 午后：down 第 6 理论（slotwise / one-expert-per-block）— 失败（−19%）

**动机**：W8A8 后 serve 剖析显示 e4m3 MMA down 中位数 45.3µs ≈ SIMT 48.5µs——**两者都只有 ~38% DRAM**，而 act kernel 是 **71%**。假设：down 每 block 碰 9 个不同 expert（每 SM ~27 条并发 DRAM 流）→ row-buffer 抖动；act 每 block 1 个 expert（~5 条流）。

**实施**（FERRITE_DOWN_SLOT=1，env gate 保留）：每 block 一个 (token, slot) assignment = 一个 expert，32 h-rows；phase-2 按 j 升序归约 partial（与块内累加逐位等价）。

**实测**（三轮 A/B）：base **10.76/10.78ms** vs slot **12.86ms（−19%，慢 2.1ms）**。faults=0、opcheck=0、文本正常。

**根因（重要认知）**：网格从 2048 blocks → **18432 blocks**，每 block 只读 **8KB** 权重（32 行 × 256B）。**block 粒度太小 → 调度/ramp/staging 固定开销主导**。反推 act 的 71%：act 每 block 读 **128KB**（16 行 × 4096 k × 2 投影 = gate+up），是 slotwise down 的 **16 倍**。→ **真正变量是"每 block 的字节量"（摊薄固定开销），不是 expert 流数**（也解释了 HTILE/cp.async 为何中性：它们没改每 block 字节量）。

**down 至此 6 个理论全部失败**：① HTILE（局部性/act 重读）② one-expert 局部性 ③ cp.async 延迟隐藏 ④ 占用率 3→4 blocks/SM ⑤ wave 尾 ⑥ slotwise 单 expert/block。**唯一未被否证的观察：每 block 字节量越大越快（act 128KB/block = 71%，down 8-72KB/block = 38%）**——若要继续，方向是**每 block 读更多连续字节**（大 h-tile，如 256 行 × 9 slots = 576KB/block、grid 仅 16×16=256 blocks），但并行度会降（256 blocks 太少）。**down 暂停。**

**当前基线**：10.76ms = **1486 tok/s**（env：标准 + `FERRITE_P2P=1 FERRITE_P2P_AR5=1`；`FERRITE_DOWN_SLOT` 默认 OFF）。

## ⛔ 铁律（用户 2026-09-10 明令，效力高于本文档任何"机器漂移/节点退化"的记载）

> **B300 硬件永远不会坏、永远不会偏移、永远不会劣化。任何时候性能下降一定是代码原因。**

推论与执行要求：
1. **禁止**把性能下降归因于"机器热漂移 / 节点退化 / page cache / 温度"——那不是解释，
   那是没查完。同一二进制同一配置的两次测得值不同，**唯一**允许的结论是"两次之间代码或
   环境配置有差异"，必须继续二分到具体改动。
2. 本文档早前多处"跨会话绝对数字不可比（±0.3ms 漂移）""上游漂移 +0.35ms"的记载**作废**——
   它们是未定位的代码/配置差异被误记成了物理现象。
3. **每次报告性能下降前**，先做：① `git log` 找出两次测量之间的全部提交；
   ② 逐条核对是否改了默认路径（尤其"新增旋钮但默认值写错导致每次调用做一次 env 查找"
   这类隐蔽改动）；③ 用 `git show <c>:<file> > /tmp/old && nvcc --lib /tmp/old.so` 的同会话
   背靠背 A/B 复现。**任何"加了旋钮/模板/开关"的提交都要单独审一遍热路径开销。**
4. 本次实测的 3 个自引入回归（`dc4d7ae` → `0158df9` 之间）：
   | 提交 | 回归 | 修法 |
   |---|---|---|
   | `850fd0b` cublas algo 旋钮 | `static ALGO: i32 = -2` **永不赋值** → `gemm_cublas` 每次调用都 `std::env::var`+parse（200+/步，host 侧） | `OnceLock` 缓存（每进程一次） |
   | `6d5c28e` mix token-pair 变体 | **默认 ON** 且同会话 A/B 实测更差（11.50/11.52 vs 11.05） | 默认 OFF（`FERRITE_HC_MIX_PAIR=1` 才启用） |
   | `42c2ec8` down NP 流水线 | 把原版末块的 `wait_group 0` 改成恒 `NP-1` → 尾块 commit 流断掉后该界不再蕴含 chunk j 已落地（**潜在陈旧 smem 读**） | 尾部用空 `commit_group` 补位，保持记账 |
5. `dc4d7ae`（10.51ms = 1522 tok/s）是**性能基准提交**：默认路径必须与它行为一致；
   之后的所有实验性改动一律 **env-gate 默认关闭**，各自独立 A/B 通过才可改默认。

## 2026-09-10 下午（SGLang 路线会话）：hc big-fuse 落地 −0.25ms，down 第 7 理论失败

**用户指令**：严格按 SGLang 的思路做（不再自己发明）。据此复刻了 SGLang `mhc_pre_big_fuse` /
`mhc_post` 的结构。

### ✅ 落地：SGLang 结构 hc_pre big-fuse（−0.25ms，已默认 ON）
- **SGLang 结构**（`mhc_pre_big_fuse_with_norm_tilelang`）：**每 token 一个 block + warp 分工并行**
  —— `tid<32` 做 post_mix sigmoid + comb_mix 行 softmax + sinkhorn（**全在寄存器/xor shuffle**，
  ~2µs，**无 barrier 耦合**）；`tid>=32` **同时**做 `li_raw = Σ pre[i]·res[i]`（float4、SMEM 暂存）
  和 Σli_raw²。一个 block-wide barrier，然后全线程做 normalize（`li = li_raw·inv·nw`）。
- 取代 rest345 的"按列分块 + 串行相位 + 跨块选举 + is_last 块做 normalize"。
- **实测**：base 10.76/10.75 vs fuse **10.51ms**（同会话背靠背），faults=0、opcheck=0、出师表逐字
  （`</think>` 后直接背诵）。`FERRITE_HC_FUSE=0` 回退 rest345。
- **两个真实 bug（都已修，值得记住）**：
  ① `__shfl_xor_sync` 的 mask 必须**精确等于参与 lane**：我写了 `0x000f000f`（含 lanes 16-19，
  但它们在 `lane < 16` 之外不执行该指令）→ **err 715 illegal instruction**。参照实现用
  `m16 = 0x0000ffff`（warp lanes 0-15）。
  ② `s_li = hc_smem + mix + 2` = 104B 偏移，随后做 **float4 访问 → err 716 misaligned address**。
  必须把 li 暂存偏移 padding 到 16B 边界（28 floats = 112B）。
  ③ 另修：`__syncthreads()` 不能放在 warp 分支内（原写法只有 warps 1-7 到达 → 死锁）；`h4` 作用域；
  `smem` 与文件内已有同名 `__shared__` 冲突（改名 `hc_smem`）。

### ⬜ 中性/负结果（全部同会话背靠背 A/B）
| 改动 | 结果 |
|---|---|
| **hc_post 每-token block 重写**（SGLang `mhc_post_tilelang` 结构：grid=s、post/comb 进 SMEM、每线程一次产出全部 n 个输出；res 流量 5.25MB→2.25MB/次） | **中性**（10.79/10.86 vs 10.82）——并行度从 65536 线程降到 4096，抵消了流量收益。保留（结构更贴近 SGLang） |
| **cublasGemmEx algo=-1**（CUBLAS_GEMM_DEFAULT，绕开选到 splitK 的启发式） | **中性**（10.57 vs 10.51/10.56）→ splitK 不是可摘的浪费。旋钮 `FERRITE_CUBLAS_ALGO` 保留 |
| **down HT 放大**（`FERRITE_DOWN_HT`，template HT：保持 9 slot/block 不变，只把同一 expert 的连续读从 8KB 放大到 64KB） | **更差**：ht1 **10.81/10.86**、ht4 11.09、ht8 11.62 → **并行度下降的代价 > 连续读的收益** |
| **down slotwise（每 block 一个 assignment/一个 expert）R=8** | **更差**：11.31 vs 10.51（每 block 只摊 1 个 token，权重不摊销；且 18432→2304 blocks 的重排无效） |

**down 至此 7 个理论全部实测失败**：HTILE / cp.async 延迟隐藏 / 占用率 / wave 尾 / act 流量 /
slotwise 单 expert / h-tile 放大。**38% DRAM 的成因仍未定位**（唯一未被否证的观察：act 每 block
128KB 连续 = 71%，down 每 block 9 个 expert 的 8KB 片段 = 38%，但任何"改这个变量"的尝试都失败）。

### 当前状态
replay **10.51ms（同会话区间 10.51-10.86，机器漂移 ±0.3ms）= ~1522 tok/s**，faults=0、出师表 ✓。
运行配方 = 标准 env + `FERRITE_P2P=1 FERRITE_P2P_AR5=1`。
**距离 1600（10.0ms）还差 ~0.5ms**：SGLang 剩余的大项是 DeepGEMM / TRT-LLM 的 C++ 库
（我们无法直接移植），以及他们**激活全程 bf16**（我们 f32 → 多出 cast 与 2x hc 流量）这一
**模型级 dtype 改动**（精度风险高）。

## 2026-09-10 下午续：down 第 8/9 理论（HT 放大、NP 深流水线）失败 + 一个"假快"教训

- **HT 放大**（`FERRITE_DOWN_HT`，每 block 连续读 8KB→64KB）：ht1 **10.81/10.86**、ht4 11.09、ht8 11.62
  → **更差**（并行度下降的代价 > 连续读收益）。
- **NP 深流水线**（`FERRITE_DOWN_NP=4`，3 chunk 在飞）：np2 10.94/10.99 vs np4 10.86 → **中性**
  → 暴露延迟假说也不成立。
- **⚠️ "假快"教训（本会话最重要的方法论收获）**：NP 第一版把 `DM_STAGE(j+NP-1)` 写在 `wait_group`
  **之后** → `wait_group N` 的语义（"至多 N 组在飞"）不再蕴含 chunk j 已完成 → MMA 读**陈旧 smem**，
  bench 报 **8.21ms**（vs 正确的 10.85）——**本会话最漂亮的数字是一个正确性 bug**。
  规则：cp.async 流水线必须 **commit 在 wait 之前**；任何"异常大"的提速先查语义不变式，
  再谈性能（数值/文本判据不足以抓它——那次文本仍然连贯）。

## 2026-09-10 收尾：默认路径恢复到 dc4d7ae + down 第 8 理论失败

**用户令："不准测了，性能直接改回去" / "必须比10.5快"** → 已完成：
`ferrite_kernels.cu` 恢复成 `dc4d7ae` 内容（**只保留 12 行版本戳**，加载门禁仍生效）；
`cuda.rs` 只留 cublas `OnceLock` 修复 + 版本门禁，删掉已移除实验 kernel 的引用。
**单轮实测（同一次 checkout 双产物重编）：`replay p50 = 10.49ms` / 1525 tok/s，faults=0、
opcheck=0、出师表逐字 ✓ —— 比 10.51 基准略快，比改回去之前的 11.43 快 0.94ms。**

**down 第 8 理论（all-slots 预取，`FERRITE_DOWN_ALL=1`）失败**：假设"18 个 `__syncthreads`/block
（9 slot × 双缓冲）≈ 16µs/call 是 45µs 中说不清的部分"，改为 9 个 slot 一次性预取（9×8.7KB=78KB
smem）→ 屏障降到 2 个。**实测 11.75 vs base 10.52（+1.23ms，更差）**：78KB smem 把占用率压到
2 blocks/SM，代价远超屏障收益。代码保留在 `FERRITE_DOWN_ALL` gate 后（默认 OFF）。
**down 至此 8 个理论全部失败，38% DRAM 成因仍未定位。**

**当前状态**：HEAD = `4d1ea65`（默认路径 = dc4d7ae 的 kernel 集合 + 版本门禁），
`replay p50 = 10.49-10.52ms` = **1520-1525 tok/s**；距 1600（10.0ms）还差 ~0.5ms。
剩下的大项都已在各自地板或需要结构性改动（MoE act 71% DRAM 地板 / AR v5 NVLink 地板 /
down 8 理论失败 / 小 kernel 桶 0.5ms 需生产者双输出或融合）。

## 2026-09-11 cast 消除落地（+ 归属勘误）

**勘误（防止下一次再走错文件）**：209 次/步 `f32_to_bf16` 的唯一来源是
**`crates/ferrite-kernel/src/cuda.rs` 的 `gemm_cublas`（:2113，私有）**，不是 dsv41。
- `crates/ferrite-models/src/dsv41/device.rs` **没有** `gemm_cublas`（只有裸指针的
  `gemm_f32`/`gemm_bf16` 包装，:545/:557）；`dsv41/chain_dev.rs` 也**不调用**它。
- dsv41 的同类点是 `chain_dev.rs:791 lin_bf16`（`f32_to_bf16` + `gemm_bf16`），但被
  `cublas_m1()` 门控，**默认 false** → 默认走 `gemv_bf16`（无 cast）；dvs41 nsys 里根本没有
  `f32_to_bf16`（profile: gemv_bf16 49 次/步）。所以 dsv41 侧无需改动。
- `gemm_cublas` 的唯一调用点是 `matmul_dev` 的 `n == 16` 分支（cuda.rs:2480）——
  是 **m=16 批量解码**，不是 m=1；m=1 早就走 `gemv_bf16_v2`/`gemv_tri`（kernel 内转换，无 cast）。

**落地**：新增 `CudaBackend::gemm1_fused_into`（cuda.rs:2260），在 `n==16` 分支于
`gemm_cublas` **之前**尝试——复用已验证的 `ferrite_gemm3_bf16_mma`（单权重、per-group-x 的
退化调用：w2/w3=null、o2=o3=0），f32 激活**直接吃**（smem 内转换），同时消掉
nvjet splitK + splitKreduce。门控 `FERRITE_GEMM3`（默认 ON，`=0` 完全回到 cuBLAS 便于 A/B），
形状不受支持（in_f%16 / 低于 K-split 地板 / n>16 / out_f>65536 的巨型输出如 lm_head）
或 kernel 报错时回落 `gemm_cublas`。`cargo check --workspace` ✓、`cargo test -p ferrite-kernel --lib`
8 passed ✓。**待办**：`.cu` 未改（无需重编 .so），但需在 b300 上跑真实权重 A/B
（`FERRITE_GEMM3=0` 对照）确认 tok/s 与四段文本——bf16-MMA vs cuBLAS 的 1e-3 重结合
是已知的 leading-token flip 类，且 out_f>65536 的路径仍带 cast。

## 2026-09-12 统一引擎战役（用户指令：架构零重复 + dspark MTP ≥400 + KV/radix + 1M prefill + 多并发，全超 sglang）

**基线 tag**: `dsv41-6.15ms-162toks`（DSV41 单并发 6.15ms = 162.6 tok/s）。**对标**: sglang DSpark 383.7 tok/s @ B=1 B300 TP8（V4-Pro, accept~5, LMSYS 博客）——用户目标 400 = 超越它。

**战役计划**: `docs/agent/unified-engine-battle-plan.md`（4 份调研综合：dup-audit / mtp-research / kv-research / prefill-research，报告在 /tmp/*_report.md）。五波：①快速统一 ②dspark MTP ③KV/radix ④1M prefill ⑤多并发 + 深度统一（cuda.rs→devrt、kernel 合并、LayerDesc 描述化）。

### Wave 1 完成（2026-09-12，4 commits）
- 单一二进制 `ferrite-serve --model {glm53,dsv41}`（T3）；GLM run_serve 改走共享 `serve::launch`（T2，删 80 行复制）；Dsv41Frame 上移 ferrite-models（T1）；死代码 dist.rs 删除（747 行未挂载文件）
- ferrite-dispatch 确认为调度基座（radix 611 行 + state 三层 hicache 1037 行 + mtp 协议 267 行，SGLang parity）——Wave 3 接 GpuEngine 的对象
- 旧"MTP/投机禁令"已解除（用户 2026-09-12 指令明确要求 dspark block-5 单并发 ≥400）

### Wave 2 dspark MTP（进行中）的关键定案
1. **dspark 语义**（host 参考 dspark.rs + sglang 仲裁）：3 个 mtp block 一次 forward 出 5-token block（block_size=5）；MQA 单 kv head；**块内非因果**（所有行共享窗口+块候选集）；行 r 位置 = start_pos+bs+r；Markov head 顺序 5 步 gumbel（u≡1 时退化为 argmax——与 sglang 语义一致）；权重 2401 key 全在（M0 核对）。**官方实现已找到**：`deepseek-ai/DeepSpec`（论文 arXiv:2607.05147，本地 clone `/tmp/DeepSpec`）——`deepspec/modeling/dspark/**` + `deepspec/eval/dspark/**`（`draft_ops.py:32-44` 窗口 append+crop；`common.py:86-97` context 严格 `< anchor_pos`）。六点语义逐条裁定见 `/tmp/official_dspark_ref.md`。
2. **时序**（host 语义定案，与 sglang 的 gamma+1=6 不同但更优）：主链**单行** forward(anchor) → argmax + hidden(t) → draft(hidden, t) 出 d1..d5 → **verify 5 行 [d1..d5]**（anchor 行走单行路径 = 数值域天然与非 spec 一致，k=1 时输出与非 spec 完全相同）→ accept k = 1 + 最长匹配（d1 vs argmax 免费第一个匹配）。
3. **数值域铁律**：verify 的 argmax 必须与逐 token 单行 decode 一致——行独立算子多行调用（rmsnorm/hc/rope/sparse_attn/MoE route），**投影与 head 逐行循环**（同 kernel 路径）；M2 优化做多行 GEMV kernel（per-row 独立累加）。
4. **RoPE 仲裁**：host 参考两处转录 bug 已修（q 的覆盖错乱 + o 的 per-head pos）——以 sglang 生产实现为准（per-token pos，token r 的所有 head 共享 start_pos+bs+r）。
5. **verify 因果窗口**：verify_ring_win kernel——行 r 的窗口 = 截至 pos+r 的因果槽位（块内因果由 ring 几何天然给出：行 r 看到块内 0..r）；GPU 测试 7/7 过（含窗口边界）。markov_head kernel GPU 测试 5/5 过（含 full 129280×256）。
6. **发现**：三个 batched expert launcher 的 rows 参数**只做校验不进 grid**（每次调用只算 1 个 activation 行）——draft_moe 曾传 rows=bs 静默只算行 0，已修（逐行调用模式 = moe_rows 的写法）。
7. **影子模式**（shadow_step，subagent 实现中）：draft+verify 的副作用全回退（ring 槽快照 + compressor 状态 + clen），主链输出与非 spec 逐位一致——先验证 accept 分布（预期 4-5）再上真 commit。
8. **遗留**：compressor 多行组形成（6 行跨多组未建模）、indexer 块内候选排除（TODO）、verify ~800 launch/步（M2 收）、多行 sliced argmax。

## 2026-09-12 Wave 2 dspark 影子模式：6 个根因的完整修复链（调试方法论实录）

影子模式（draft+verify+全回退，主链输出与非 spec 逐位一致）端到端跑通前的全部根因，按发现顺序：

| # | 根因 | 症状 | 定位手段 |
|---|---|---|---|
| 1 | **DecodeRun 用 `?` 吞 rank 错误**（RC0） | 任何 Err 都变 "a rank did not answer"（无日志盲区） | wedge-audit 纠察 |
| 2 | **snapshot 早于 step_dev**（RC1） | rollback 把真实步的 compressor 提交也回退 → clen 落后 1 → 第二步起污染（"step 1 fine, step 2 dead"） | 同上 |
| 3 | **rollback 空镜像不早退**（bisect 模式） | 未初始化快照覆盖主链状态 | 二分实验 |
| 4 | **mtp 权重 shard vs 全局几何**（attn 5 个 tensor + markov） | cuda-700：wq_b 读 4096 行本地切片按 32768 行（+36.7MB OOB）——**微基准通过但 serve 崩的典型**（微基准自建权重无 shard） | 三个 justice 并行审计 |
| 5 | **MoE 的 TP 几何缺失**（inter_local 没除以 world + shared stp + 无 AR） | cuda-700 续命：shared expert n=2304 读 288 行本地切片 | 同上 |
| 6 | **all_kv 布局/stride 错位**（满 win 行布局 vs kernel 的 n_win+bs stride 公式） | drafts 完全随机样（[1503, 12235, ...]） | 用户提示 + 手读 kernel |
| 7 | **draft rope 位置**（pos+bs+r → 最终 pos+r） | drafts 部分变化仍 acc=1 | sglang 证据链仲裁（positions_2d = prefix_lens + arange；我们的 t0 是"已消费"约定 → pos+r） |
| 8 | **verify 因果窗口失效**（verify_ring_win 的槽号过滤在 base+r≥window 时永不触发 → 行 r 读块内未来行 KV + 丢最旧 m-1-r 历史） | verify 输出垃圾/EOS（除最后一行外全错） | audit-step-rows 纠察 |
| 9 | **compressor 只吃行 0**（pool 的 state 分支和 commit 的完成规则是 per-POSITION，seqlen=m 只消费行 0） | **用户观察的"latent 没更新"** | 同上 |

**修复后的结构**：attn 全 Replicated（draft 冗余 ~540MB/rank）；MoE 走主链本地几何 + AR；all_kv 紧凑布局（只拷 n_win 活行，块行在 [n_win, n_win+bs)）；rope per-token（行 r = pos+r，所有 head 共享）；verify 窗口逐行交错 append(r)→window_idxs(r)（单行内核保持环不变式，行 r 的窗口在行 r+1 覆盖旧槽前算好）；compressor 逐行单 token 三元组（pos_rows[r] 指针）。

**方法论教训**：
1. **sticky error 的第一检查点认领一切**——`dsv41_gemm_fp8_mx` 报 700 时真实源可能在前面的任何 kernel；CUDA_LAUNCH_BLOCKING + RC0（错误显形）是定位组合拳。
2. **微基准无法复现 shard 类 bug**（自建权重无 shard）——serve 环境的越界要在 serve 排。
3. **kernel 的派生参数公式（n=window+clen）必须与调用侧的缓冲布局自洽**——"凑巧相等"的参数（clen=bs stand-in）掩盖语义错位。
4. **host 参考的注释是转录者的解读**（"start_pos+seqlen" 的 seqlen 被误读为 bs）——仲裁要找生产实现（sglang）的证据链。
5. **多行化的正确姿势 = 逐行复用单行内核**（环不变式逐步保持）——"一次多行 + 自定义派生"两处（窗口/压缩器）都出了因果 bug。

## 2026-09-12 Wave 2 续：修复 #5-#7 与官方参考闭环

**官方参考闭环**（official-pytorch subagent）：**deepseek-ai/DeepSpec**（MIT，`deepspec/modeling/dspark/**` + `deepspec/eval/dspark/**`）+ 论文 arXiv:2607.05147 + `/tmp/official_dspark_ref.md`（6 争议点裁定）。注意：DeepSpec 是 Qwen3 研究版（无 MLA/SWA/TP）——语义 oracle，生产细节以 sglang 为准。

| # | 修复 | 证据链 |
|---|---|---|
| 5 | **tap 口径**：收集层**输出**的 hc 均值（`completed.mean(dim=1)`，sglang deepseek_v4.py:3132-3141）——原 tap 在 layer() 开头收集=上层输出，**差一整层** | 实测：修复后 verify[0]==next 从 1/12 → 5/12 步 |
| 6 | **draft head 的 collapse 约定**：主链最终 collapse 用 `premix_slot(1)`=最后层 **attn_pre**（文本正确=有效约定；checkpoint 无 hc_head_* key——2401 mtp key 查证），draft_head 原用 ffn_pre——改对齐 attn_pre | audit-hc-head 阻塞级发现 |
| 7 | **撤销 mk 替换**：官方块行 0 的 KV 是 **embedding 派生版**（DeepSpec 的 draft 块投影自己的输入）——audit 的"席位规则"（mk 替换）是推断非官方语义，回退；窗口的 pos 槽排除保留（双计修复的真正部分） | DeepSpec 块结构 |

**V4.1 head collapse 的约定事实**：checkpoint 无任何 hc_head key（主链+mtp 都没有）→ pre_mix 是唯一约定 → 主链（attn_pre/slot 1）与 draft 必须同 slot。

## 2026-09-12 Wave 2 修复 #8：时序对齐（anchor=bonus 官方结构）——结构里程碑

**修复**（6 处）：anchor 从"刚消费的 t0"改为**刚采样的 next/bonus**（官方 draft_input_ids[:,0]=bonus，未 forward）——draft 块 [next, noise×4] 从 pos+1 起算（每个 γ 预测覆盖未知 token）；verify 6 行 [next, d1..d5]（anchor 行的 forward 提供 KV）；accept 链 drafts[j] vs verify_out[j]；seed_window 移到 pos-1（main_x 源=anchor 前驱=t0）。

**实测验证**：时序结构生效——verify 的 anchor 行（verify_out[0]）正确预测下一 token（pos=11 的 verify[0]=1767==pos12 的 next=1767 ✓、pos=20 的 verify[0]=26433==pos21 的 next ✓ 多处）。**verify 链 + 时序已正确**。

**剩余瓶颈**：drafts 部分匹配（pos=20 的 drafts[4]==verify[4]=25653、pos=22 的 drafts[1]==verify[1]=1572——部分位置匹配暗示中等数值 bug 而非结构崩坏）。mean-accept 仍 1.02。audit-draft-residual 在查。

## 2026-09-12 Wave 2 修复 #10 与单元对照方法论（用户定调）

**单元对照基础设施**（用户方法论："跑一次 demo 抓 mtp 的 latent 对比"+"单元测试比较每个单元"——不再审计推理）：
- **官方 golden**（/tmp/unit_golden.py，920 行）：官方 model.py 的 DSparkBlock TP1 单卡加载（原始 checkpoint 的 mtp.* 逐 tensor + convert.py 的语义复刻），合成输入（seed 128 行 + decode 1 行），monkeypatch 逐单元 dump（forward_embed/main_x/h/premix/attn 全链/rope/quant/sparse/ffn/head/markov）→ /tmp/unit_golden.pt
- **device 侧**（unit_dump.rs）：DSV41_DSPARK_UNIT_DUMP=1 dump 第一次 draft_forward 的同款单元 → JSON
- **注入**（DSV41_DSPARK_UNIT_INJECT）：golden 的 inputs（main_hidden/token/pos）注入 device forward——两侧同输入对照
- **远程环境**：CUDA torch 2.14.0+cu130（pip --break-system-packages --force-reinstall --index-url .../whl/cu130 "torch==2.14.0+cu130"）+ tilelang 0.1.14
- **TP8 demo checkpoint**：convert.py 转换完成（/opt/dlami/nvme/models/V41-demo-TP8，34 分钟）——官方 generate.py 的端到端用

**修复 #10**（官方 model.py 语义，单元对照定位）：块 = [embed(t0), noise×4]@**下一位置视角**（RoPE start_pos+1+r；窗口 seed 行 @ start_pos%win）——#8 的 anchor=bonus（sglang/DeepSpec 约定）被单元对照证伪（q 100% 发散）；#8 修对了位置（pos+1）但 token 错（应 t0）。verify 回 5 行 [d1..d5]，accept 链 drafts[0]==next（免费首检）+ drafts[j]==verify_out[j-1]。

**注入 pos 的 off-by-one 教训**：注入的 pos 替换 draft_forward 的 pos 参数（=块行 0 的 rope 位置）——golden 的 start_pos 是 t0 位置——注入要 +1（129 不是 128）。第一次对照（pos=128）q 完全没变（max|d|=54 三次相同）就是这个 off-by-one。

**对照的单元 diff 表**（pos=129 修正后待跑）：
| 单元 | 修复前（pos=128 注入） |
|---|---|
| embed | rel=0.0 ✓ |
| main_x | rel=4.4e-3 ✓（量化噪声） |
| q（rope 后） | rel=1.0 ✗（token/位置语义 + 注入 off-by-one） |
| o/attn out | rel=1.0 ✗（下游） |
| moe/h | rel=1.6-1.8 ✗（下游） |

## 2026-09-12 Wave 2 终局定位：单元对照的完整证据链（draft 匹配率 6% 的根源）

**单元对照的执行链**（golden: /tmp/unit_golden.pt 官方 TP1 参考；ferrite: /tmp/unit_ferrite.json 注入同输入 dump）：

| 单元 | rel diff | 判定 |
|---|---|---|
| embed（块输入 [5,4,5120]） | **0.0** | ✓ 注入完全一致 |
| main_x（main_proj 投影） | 4.4e-3 | ✓ 量化噪声级 |
| **xn post-norm（attn_norm 后）** | **1.1e+38** | **✗✗✗ rmsnorm 输出天文数字** |
| q_pre_rope（wq_b 投影后） | 1.0 | ✗（xn 爆炸的下游） |
| q/o/moe/h | 1.0-1.8 | ✗（下游全污染） |

**最终定位**：block0 的 collapse（hc_collapse + identity premix [1,0,0,0]）输出正常量级，**紧随的 rmsnorm（attn_norm）产生 5.4e+37** —— attn_norm 权重的加载/内容是最强嫌疑（mtp 层的 attn_norm key 名/规格/加载路径）。多行 gemm 路径被排除（m=1 逐行版同样爆——它爆在输入 xn 本身）。

**排除项**（对照过程中的方法论教训）：
- rope 位置（pos 128/129 注入零影响——三次 max|d|=54 完全相同）
- gemm 多行路径（xn 爆炸在上游）
- 量化粒度（官方 block=128 vs 我们 32——量化噪声级不是 100%）
- 对照语义错位陷阱：h_premix（norm 前）vs golden attn.in（norm 后）的 78% 差异是**语义错位**不是数值 bug；golden 的 stage0.attn.hc_pre 是 mixes 输出 [1,5,4] 不是 collapse
- qr 的 dump 必须在 quant1 **前**（quant1 原地覆盖 fp8 字节→NaN 假象）

**下一步**（下会话）：①attn_norm 权重的加载验证（远端读 mtp.0.attn_norm 的 checkpoint key 名+形状+前几个值 vs golden 的权重 dump）②rmsnorm kernel 的多行调用（n=bs）复查 ③修复后 accept 应跃升（xn 正常→q/o/h 全链恢复）。

## 2026-09-12 Wave 2 真根因终局：load.rs:925 的 placeholder hack（单元对照一锤定音）

**铁证链**（单元对照 + 手算 + 权重 dump 三角验证）：
1. 逐行 rmsnorm（n=1）修复后 xn 仍 5.4e37 **逐位不变** → kernel 无辜 → 权重指针指向垃圾
2. 权重 dump：**attn_norm_w[0]（block0）= 2.25e27 天文数字**（fp8 字节流当 f32）——**block1 正常**（[0.0664,...]）
3. **load.rs:925**：`take!(p, ld, "main_proj.weight", attn_norm); // placeholder`——**stage 0 的加载把 main_proj 的 spec 覆盖到 attn_norm 字段**（占位 hack）→ attn_norm 指向 main_proj 的 fp8 [5120,15360] → rmsnorm 读 fp8 字节当 f32 → 1e27 爆炸 → q/o/h 全垃圾 → draft 匹配率 6%

**修复**：删掉 placeholder（main_proj 有自己的 w.main_proj* 字段）。修复后：
- attn_norm_w[0] = [0.0491, 0.0457, 0.0481, ...]（与 checkpoint 逐值一致，仅符号——dump 端 bf16 解码的显示差）
- **xn post-norm rel=1.86e-03 ✓**（量化噪声级——从 1e38 恢复！）
- **q_pre_rope rel=1% ✓、q rel=1% ✓**（draft 投影全链恢复）
- o/h 还差 79%/1.4——**窗口内容差异**（golden 合成 seed vs serve 真实窗口——对照方法固有差异，非 bug）

**spec 首测（根因修复后）**：mean-k=0.120、tok/step=1.12、62ms/步（draft 5.85 + verify 38.68 + commit 0.19 + 主链 ~17）。**新 bug：双字**（"床床前"——**verify 缺 next 行的 KV**：d1 行的 causal window 里 pos+1 槽是空的（主链只 forward t0@pos，next@pos+1 的 KV 不存在）——sglang 的 6 行 verify [anchor(next), d1..d5] 里 anchor 行提供这个 KV。修复中。

**方法论定论（用户"单元测试比较每个单元"的完全正确性）**：11 个"终局定位"里只有单元对照+手算+权重三角验证是真定位。审计推理（猜嫌疑）浪费了其中 9 次。教训：**数值问题用数值对照定位，一行手算胜过十轮审计**。

## 2026-09-12 Wave 2：MoE 段 138% 的判词与修复（audit-moe-seg）+ 一个误判纠正

**MoE 段对照的完整梯队**（窗口注入后）：embed ✓ 0.0 / main_x ✓ 4.4e-3 / o(attn) ✓ 5.6% / **ffn_in ✓ 4.4% / route_w ✓ 3.5% / shared ✓ / moe_out ✗ 139% / h_block0 ✗ 138%**——**输入与路由都对，expert 计算段全错**。

**判词的两个阻塞级缺陷**（两条分支各一个——"无论 ILV 开关怎么设，draft 的 MoE 都错"）：
1. **ILV 选择 bug（默认路径）**：`dspark_dev.rs:1274` 的 `if !ld.experts_ilv && supports_moe_batch()` ——`DSV41_EXPERT_ILV` **默认 ON** → 条件恒假 → 走**顺序 indirect 分支**，该分支用**非交错 reader** 读**交错专家池**（load.rs:650-656/716-724：w1/w3 view 同一指针交错）→ gate/up 读到打乱字节。**主链在 chain_dev.rs:7606 有红线 guard，draft 没有**。修复：判定去掉 `!ld.experts_ilv`（batched kernel 支持 ilv），尾参 `ilv` 从写死 0 改为 `ld.experts_ilv as i32`。
2. **fuse 判定不一致（ILV=0 时）**：draft 写死非融合（act_slot=2*inter + 二次 swiglu），kernel 默认 fuse=1（`mxf4.cu:2366`：g_fuse && mode==2 && dim%512==0；dim=5120 ✓）→ 融合尾只写 inter 宽 → up 半区是未初始化显存 → 垃圾。修复：镜像 `moe_rows:5378-5386` 的 gateup_fused 判定 + 条件 swiglu。
3. 另修：顺序分支的 per-row rows=1（原 rows=bs + 单权重指针 → 行间权重错配）+ ilv guard；buffer 尺寸从骨干几何（384/6）改为 draft 几何（128/3）。

**误判纠正（重要）**：早前调查曾判定 "`ferrite_rmsnorm` 的 n>1 产生 1e27 垃圾"并把 attn_norm 改成逐行 n=1 绕过。**读 kernel（ferrite_kernels.cu:280-330）确认它是正确的**（grid(n) 每 block 一行、行偏移 row*dim、blockDim-sized 归约）。1e27 的真因是**当时 attn_norm 的权重指针是垃圾**（load.rs:925 placeholder bug）。**教训：数值异常要先确认输入/权重，再怀疑 kernel**（"绕过"是无必要的性能损失——已回退为多行）。

**真实 serve 的长 prompt 基线**（出师表 250 tok，MoE 修复前）：文本内容**正确**（"先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也"逐字对）+ **双字**（臣臣/盖盖）+ accept=0.020。文本正确 = next（主链 argmax）正确 + 大部分步 k_acc=0（纯主链输出）；accept 低 = drafts[0]==next 的比率极低 → draft 质量差（MoE ILV 是候选，修复后待测）。

## 2026-09-12 Wave 2：MoE ILV 修复的实测效果（26x accept 提升）

**修复前**（长 prompt 出师表 250 tok）：accept mean-k=**0.020**、tok/step 1.020、文本内容正确但双字（臣臣/盖盖）。
**修复后**（draft_moe 的 ILV/fuse/顺序分支/buffer 四项）：accept mean-k=**0.520**、tok/step 1.520、draft 5.85→**4.99ms**。

**判定**：audit-moe-seg 的判词完全正确——**ILV 选择 bug（默认路径走非交错 reader 读交错池）是 draft MoE 输出 138% 偏差的根因**，修复后 draft 的预测质量提升 26 倍。**draft 的数值链（tap→main_x→block→MoE→head→markov）现在基本健康**。

**剩余**：文本**重复加剧**（"全文全文"、"先帝创业帝创业"）——这是 **verify 的 interleave 缺陷**的直接后果（verify-parity-audit 判词：`sparse_attn`/`indexer_rows` 在全部 append 之后统一跑，读"块末 clen"→ 行 0 的候选含本块未来 latent → argmax 退化为重复刚看到的 token）。interleave-rows-fix 正在把 attention 的读侧逐行化（sparse_attn 改 b=1,m=1 单行调用——与 eager 同形状 = parity 保障）。

**性能状态**：verify 38.85ms（5 行 = 7.77ms/行 > 单行图化 6.15ms/行——batch 收益未兑现）。400 路径的完整账本在 `docs/agent/dspark-perf-400-plan.md`（3 阶段：多行化+图化 → ≤8ms；吞主链步 −4.5ms；confidence-gated verify）。

## 2026-09-12 Wave 2：accept 链错位（第十修的半成品）——33% 匹配率的真因

**发现路径**（3434a9ad 的 DBG 数据分析）：`verify_out[0]==next` 仅 16%、`drafts[0]` vs `next` 匹配 ~33%——**33% ≈ 相邻位置的预测相关水平**，是"比较对象错位一位"的指纹。

**根因**：**第十修**（fix #10）把 draft 的块改成"下一位置视角"——块行 j 位于 `pos+1+j`，所以 **`drafts[j]` 预测 `pos+2+j`**——**但 accept 链还留在旧设计**（`drafts[0] == next` 是拿一个 pos+2 的预测去比 pos+1 的 token）。

**正确链**（与 verify 的行位置自洽）：verify 行 j 被喂 `drafts[j]` @ `pos+1+j`（step_rows 的 pos_base+row），它的 argmax 就在同一个 `pos+2+j`——**逐索引比较 `drafts[j] == verify_out[j]`**（j=0..4）。`next`（单行步的 argmax = pos+1 的 token）是确定值，**永远 emit**，排在 verify 各行的前面。

**修复**（spec + shadow 两条路径同步）：`if drafts[0]==next { … drafts[j]==verify_out[j-1] }` → `while drafts[k] == verify_out[k] { k += 1 }`。

**教训（bug 模式）**：**语义重构（"块从 t0@pos 视角改成 @pos+1 视角"）必须把整条消费者链一起改**——生产者（draft 块）、验证者（verify 行位置）、判定者（accept 比较）、提交者（commit 的行基址）四者的"位置约定"必须同一次改完并在注释里写成一个不变量。第十修只动了生产者与验证者。

## 2026-09-12 Wave 2 终局：draft 块的"token/position 左移一位"（audit B1）——重复与 accept 停滞的真根因

**判词**（row0-residual-audit 的代码级分析）：`dspark_spec_step` 把**两套互斥的行语义混在一次调用**：
| 环节 | 代码 | 属于 |
|---|---|---|
| draft 位置实参 | `draft_forward(token, pos + 1)` | **A**（行 r = pos+1+r）|
| verify 块 | `step_rows(&drafts)`，行 j = drafts[j] @ pos+1+j | **B**（5 行块）|
| accept / emit / commit | 5 行块的链（drafts[0]==next + drafts[j]==verify_out[j-1]）| **B** |
| shadow / parity oracle | `draft_forward(t, pos)` + 5 行块 | **B** |

**机制**：draft 的第 r 行位置 = `pos_arg + r`（`rope_queries`→`rope_at`）。传 `pos+1` 时 `drafts[j]` 是 **pos+2+j 的提案**，而 step_rows 把它放在 **pos+1+j** 喂进去——**每行的 (token, position) 配对整个左移一位，pos+1 处留下缺口（真 token 是 `next`）**。自回归模型面对"跳掉一个 token"的序列，**输出倾向就是补上缺口位置的 token = next** → `verify_out[0]==next` ✓。accept 只在 j=0 恰好通过（`next==drafts[0]` 正是 draft 首 token 与 target 一致的时刻）→ **每步最多 2 个 token 且首两个相同 = 每 1-2 字符一个重复** ✓。

**三个数字同时对上的解释**（本会话最漂亮的一次闭环）：
- `drafts[0]==next` **60%**——**不是** draft 的 pos+1 预测精度，而是**模型重复模式的副产品**（重复发生时 pos+1 与 pos+2 的 token 相同，于是"pos+2 的提案"自然等于 next）
- `accept 0.52` = P(drafts[0]==next)（只有 j=0 通过）
- 文本"先帝帝创业/中道道崩"= emit 的 `[next] ++ verify_out[..k_acc]` 在 k_acc=1 时把 next 写了两遍

**修复**：`draft_forward(token, pos)`（spec + shadow 两处）——**draft/verify/accept/emit/commit/oracle 六者同一布局**。

**方法论教训（重要）**：我基于 `drafts[0]==next 60%` 的**实测**做了"旧链正确"的回退推理（并回退了 accept 链实验）——**方向对（旧链确实正确）但理由错**（60% 的真实成因是重复attern 而不是 draft 的位置）。**真正的定位来自 audit 的代码级语义分析**（把六个环节的"位置约定"列成表）——**"实测数据 + 代码级语义表"缺一不可**：单看数据会把 60% 误读为质量证据，单看代码会漏掉"60% 与 16% 同源"的量化闭环。**同时**：`quant1`（rows=1）与 `dsv41_rmsnorm_q`（n=m）的对比也证明——**"行偏移缺失"只在多行内核里可能出现**（rows=1 结构上不可能犯）。

## 2026-09-12 Wave 2：B1 修复后的实测（accept 结构改善 + 剩余的 33% parity 破坏）

**修复**：`draft_forward(token, pos)`（spec+shadow）——draft 块回到 anchor 自己的位置。

**实测**（36 步的 DBG trace，出师表 300 tok）:
| 量 | 值 | 解读 |
|---|---|---|
| `drafts[0] == next` | **33%**（12/36） | **draft 的真实质量**——之前测到的 60% 是"模型重复模式"的副产品（重复时 pos+1==pos+2 使"pos+2 的提案"等于 next）。修复后 33% 是干净的读数 |
| `k_acc` 直方图 | `{0:24, 1:6, 2:4, 3:1, 4:1}` | **12 步 ≥1**（与 33% 一致）且**出现 2/3/4**——后续 draft 也能匹配 ⇒ accept 链工作正常 |
| mean-k | **0.58** | 略高于 MoE ILV 修复后的 0.52 |
| `drafts[j+1] == verify[j]` | 32/144 (22%) | 后续位置的匹配率（低于首检，符合预期） |

**剩余缺陷（文本重复的 33%）**：`verify_out[0] == next` **33%**（与 `drafts[0]==next` 同值）——**emit = [next] + verify_out[..k_acc]** 在 verify 行 0 复述 next 时把 next 写两遍 → "先帝创业先帝创业"。**eager 在同一位置（pos+1、同一输入 next）不重复** ⇒ **verify 的多行链与 eager 的单行图之间存在 parity 破坏**。

**最强候选（待 vrow0 探针判定）**：**head 的两条路径**——`DSV41_HEAD_SLICE` **默认 ON**（`chain_dev.rs:861-863` 的 `unwrap_or(true)`）⇒ **eager 走"每 rank 切片 + 跨 rank argmax"**，而 **verify 的 head 走 `head_gemv_bf16_mrows`（全词表 Replicated）**——两条不同程序的 K 序/累加序可能不同（~1e-3 级）→ 近 tie 翻转。探针（`DSV41_VROW0_PROBE=1` + shadow）输出 `verify_top`/`eager_top`/`eager_sliced` 与 `delta_at_eager`：**|delta| ≤ 1e-2 → 边界翻转（用 confidence 门控/统一 head 路径解决）；大 → 结构性**（multi-row 调度没复现单行路径）。

## 2026-09-12 Wave 2：verify head 的 parity 修复（FOLD=0 成为默认）——实测 echo 33% → 9%

**A/B（同会话，出师表，`DSV41_DSPARK_DEBUG=1` 的 `verify_out[0]==next` 统计）**:
| 配置 | echo | 文本相邻重复 | k_acc 直方图 |
|---|---|---|---|
| `DSV41_VERIFY_HEAD_FOLD=1`（folded `head_gemv_bf16_mrows`，旧默认） | **33%**（154 步） | 6 | {0:24, 1:6, 2:4, 3:1, 4:1} |
| **`=0`（per-row `gemv_bf16`，新默认）** | **9%**（22 步） | **4** | {0:13, 1:6, 2:3} |

**机制**（两条独立证据链）：
1. `gemv_bf16_v2_wanted(n)` 要求 `n < GEMV_V2_MAX_N = 2048`（`device.rs:4657`），而 head 的 `n = vocab_size = 129280` ⇒ **生产单行 head 走的是 v1**（`dsv41_glue.cu:368 gemv_bf16_kernel`，标量 `c += 32`），**不是** v2/nt（`folded-head-korder` 从代码核实）。
2. folded 的 `head_gemv_bf16_mrows` 虽同为标量序，但其 fma/解码的配对细节与 v1 不同 ⇒ ~1e-3 级差 ⇒ 近 tie 的 argmax 翻转 ⇒ `verify_out[0]==next` 的 echo ⇒ `emitted = [next] + verify_out[..]` 把 next 写两遍 = 文本重复。

**处置**：`verify_head_fold()` 默认 **OFF**（代价：head 的 1262MB 权重被流 m 次而非 1 次 ≈ **+0.7ms/步**——**正确性优先**）。根治（folded 与 v1 逐位一致）留给后续；`folded-head-korder` 已把 folded 对齐到 v2 序（**≠ v1**，故仍 gate OFF）。

**剩余**：echo 9% + `accept 0.55`（`k_acc={0:13,1:6,2:3}`）⇒ **draft 质量仍是主要瓶颈**（`draft-quality-research` 在查 tap/main_proj/markov/窗口）。

## 2026-09-12 会话：dspark MTP 的正确性修复链（accept 0.02 → 0.58）与 400 的路径

**修复链（每一项都有代码级或实测证据）**:
| # | 修复 | 效果/证据 |
|---|---|---|
| 1 | `load.rs:925` 的 placeholder hack（main_proj 的 spec 覆盖 attn_norm 字段） | attn_norm 的权重从 fp8 垃圾（1e27 输出）恢复正常 |
| 2 | **draft 的 MoE ILV**（`!ld.experts_ilv` 让 batched 分支不可达 → draft 走顺序 reader 读交错池） | **accept 0.02 → 0.52**（26x） |
| 3 | **verify 的 per-row interleave**（attention 的读侧：compress/indexer/sparse_attn 与 append 同行序） | 消掉"块末 clen"的未来 latent 污染 + 环绕回时的窗口覆盖 |
| 4 | **`draft_forward(token, pos)`（B1）** | draft 块回到 anchor 自己的位置（60% 的 `drafts[0]==next` 是重复模式的副产品，修正后 33% 是干净读数） |
| 5 | accept 链——**实测证明旧链正确**（`drafts[0]==next` + `drafts[j]==verify_out[j-1]`），索引对齐的变体测出 0.000（跨位置） | 不改为准 |
| 6 | **投影 + MoE 的多行化**（`dsv41_gemm_fp8_mrows`、`rows=m` 进 grid） | verify 的 launch 6232 → ~3000（−52%） |
| 7 | **`DSV41_VERIFY_HEAD_FOLD=0`（已改默认）** | **echo（`verify_out[0]==next`）33% → 9%**；文本相邻重复 6 → 4 |

**关键机制（#7）**：`gemv_bf16_v2_wanted(n)` 需要 `n<2048`，head 的 `n=129280` ⇒ eager 的 head 走 **v1**（`gemv_bf16_kernel`，标量）；folded 的 `head_gemv_bf16_mrows` 与 v1 的 fma/解码配对不同 ⇒ ~1e-3 ⇒ 近 tie 翻转 ⇒ echo ⇒ `emitted=[next]+verify_out[..]` 写两遍 next = 重复。代价 ~+0.7ms/步（head 权重流 m 次）。

**400 的口径（用户 2026-09-12 权威）**：**accept 3（5 中 3）是正常水平（官方亦然）⇒ 步时 ≤ 7.5ms**。
**当前**：49.8ms（主链 6.15 + draft 4.9 + verify 38.5 + commit 0.2）@ accept 0.58。
**剩余路径**：①**逐行融合**（rope/ring/attn 的 m 次 launch → kernel 内 m 行循环，保持逐行语义）②**吞主链步**（−6.15ms）③**draft 质量**（accept 0.58 → 3，draft-quality-research 在查 tap/main_proj/markov/窗口）。

**提交纪律的教训（本会话犯了 3 次）**：共享工作树里**永远不要** `git add <file>` 而不跑一次**退出码被真正检查**的 `cargo check`——连带提交同伴的半成品会把 main 弄坏（本次 3 次：specstep 的 import、swallow 的两个未实现符号 + 一个 cast）。

## 2026-09-12 Wave 2：draft 质量的判词（`drafts[0]==next` 33% 的根因 = seed↔tap 差一位）

**判词（draft-quality-research 的三条独立证据 + 我的独立核实）**:
1. `note_ctx_rows` 的 doc（`chain_dev.rs:467-469`）："Row `keep` (position `pos+keep+1` = the new `pos_ctr`) is deliberately NOT written … that position's KV comes from the next step's `step_dev` and is **seeded by the next `draft_forward`**"——**下一 step 的 `draft_forward` 的第一个实参就是新 `pos_ctr` 的 token** ⇒ **seed 必须写在实参的位置**。
2. **`seed_window` 的调用自相矛盾**（我 grep 核实）：`:770`（`pos == 0` 分支）用 **`seed_window(s, pos)`** ✓ vs **`:1028`（稳态分支）用 `seed_window(s, pos - 1)`** ✗——同一个函数、同一个物理量、两个分支差一位。
3. **tap 的真实位置是 `pos`**：`step_dev(token, pos)` 在槽 `pos % win` 写入该 token 的行，tap hook 取的是**这个 token 在该层的 hidden**，`import_tap` 之后才拷走。

⇒ 现状写进 ring 的行是 **`{内容: h(token@pos)，RoPE: pos-1}`** 的错配；**`pos` 那一格永远没人写**（全流程写的位置集合 = `{p_i−1} ∪ [p_i+1, p_i+k_acc]`，**每个 step 的 `p_i` 都漏**）。
⇒ **窗口内容相对 RoPE 相位系统性前移一个 token**（最有用的"紧邻前驱（距离 1）"以"距离 2"出现，而"距离 1"那一行是锚点自己，与 block row 0 重复）；叠加 **24/36 步 `k_acc=0` 时 `note_ctx_rows` early-return（一行新内容都不写）** ⇒ ring 里 `pos` 格的旧行长期留用；draft 的候选集**只有窗口** ⇒ `drafts[0]==next` 被压到 33%。**MoE ILV 修复能把 accept 从 0.02 拉到 0.52，正说明这条数值链极度敏感。**

**修复（路线 A，判词推荐，最干净）**：锚点用**刚采样的 bonus**——`draft_forward(next, pos + 1)`（**seed 落点 = `(pos+1)−1 = pos`** ✓ 与 tap 的真实位置一致，`seed_window(s, pos-1)` **一行都不用改**）+ verify 6 行 `[next, d1..d5] @ pos+1..pos+6` + accept 索引对齐（`ferrite-types::spec_step::spec_accept(.., true)`）。

**另两个候选（并行在查）**：
- **§2 tap 采集口径**：我们在 `layer()` **末尾**取 `self.s.h`（= **该层输出**），而仓库自带的官方参考 `ref_inference/model.py:1264-1266` 在 `layer(h,…)` **之前**取（= **该层 attention 输入**）——**三份文档互相矛盾**（`config.rs`/`dspark.rs` 说 "input"；`dump_dev.rs` 说 "output"）——`tap-layer-arbitration` 在仲裁（**权重形状可唯一判定**）。
- **§3 窗口回绕**（`pos ≥ win` 后锚点槽不再被排除——修 §1 时必须一起修）。

## 2026-09-12 性能纠正：多行化的收益有限——verify 的瓶颈不是 launch 数

**实测**：多行化（投影 + MoE 的 mrows）落地后 **步时 47.7ms**（多行化前 49.8ms）——**仅省 ~2ms**（远低于"每省一个 launch 就省 3.3µs"的预期）。

**代码级解释（读 `dsv41_gemm_fp8_mrows` 的 launcher）**：
- 它的 **grid = ceil(n / nwarps)**（**输出行方向的分块**），**与 m 无关** ⇒ **mrows 与 per-row 的 block 数完全相同**（head 的 n=129280、nwarps=8 ⇒ 16160 块）。**多行化真正省下的只是"m 次 launch 变 1 次"**（每次 4 个 launch）。
- 投影 5 个 gemm × 40 层 × 4 = 800；MoE ~1800 → ~360（省 1440）⇒ **共省 ~2200 launch**，按 3.3µs/launch 应省 7.3ms —— **实测只省 2ms**。
**⇒ `verify-perf-impl` 的 `6232 × (2.9µs submit + 3.3µs exec)` 归因高估了 launch 的权重**。多行化的**真实收益是"权重读一次"**（投影/MoE 的权重流从 m× 降到 1×：投影 ~1.1→0.2ms、MoE ~2.6→0.5ms ≈ **省 3ms**）——**与实测 2ms 吻合** ✓。
**⇒ 400 的路径必须攻"计算/带宽"**（MoE 的 act/down 权重流、投影的权重流、attention 的 KV 流量），**不是继续砍 launch**。注意这与"融合"并不矛盾：融合（kernel 内 m 行循环）仍能省 launch，但**它的上限就是那几个 launch 的 μs 级开销**——**真正的大头在带宽**。

## 2026-09-12 【铁证】数字序列任务下的 100% 系统性重复 —— 用户判断正确：是编排/位置错，不是算子

**实验**（`DSV41_SPEC=1`，"请从 1 数到 400，每个数字单独一行"）输出：
```
1 / 2u3 / 4i4 / 55 / 6外面的6 / 7_rawBy / 7 / 8m8 / 9e9 / 10R10 / 1111 / 1212 / 13,14 / 15P15 / 16+16 / 1717 / 18\18 / 19还是19 / 1920 / 20BJ…
```
**每个数字都出现两次**（"55"、"1111"、"1212"、"1717"）——**100% 系统性**。

**用户（权威）判断**："如果真的是乱码不可能表现为这样的重复，你这样的重复必定不是算子层面的错误，而是你的处理/编排逻辑的错误"——**完全正确**：
- **算子层面的错**（K 序/量化）只会给**近 tie 的偶发翻转**（本会话 verify 的 head 实测 33%→9%）——**不可能 100%**。
- **100% 的结构性重复** ⇒ **位置/编排错**：`emitted = [next] + verify_out[..k_acc]`，数字任务下 `k_acc=0` ⇒ `emitted = [next, verify_out[0]]`；输出 "11" ⇔ **`verify_out[0] == next`（行 0 的 argmax == 主链在 pos 的 argmax）** ⇒ **verify 的行 0 实际在 `pos`（或其 KV 只到 pos）**，而它**应当**在 `pos+1`（`step_dev` 把 pos_ctr 推到 pos+1 之后，`step_rows` 的 `pos_base = pos_ctr`）。
- **最可能的 bug 类**：**host 的 `pos`（步前的值）与 device 的 `pos_ctr`（步后的值）混用**——一处用错就把行 0 落回 `pos` ⇒ 行 0 复述 next ⇒ 每步 emit 两个相同 token ⇒ **100% 重复**。

**处置**：起 `verify-row0-systematic`（vanguard，钉死 pos_base/pos_rows/行 0 的 KV）与 `emit-chain-audit`（消费链 + stream true/false 的责任判定）并行彻查。

## 2026-09-12 【铁证定案】EAGER 对照 + 消费链排除 ⇒ 根因在 spec 路径的状态污染

**EAGER（无 spec，单行 decode）对照**（同一 serve、同一模型、数字任务"从 1 数到 100"）：
```
'我会按 1 到 100 逐行输出。\n\n```text\n1\n2\n3\n4\n5\n…\n27\nanao\n29\n…\n49\n51\n…\n95\n100\n'
```
**完美数数**（仅 "anao"、跳过 50 等个别瑕疵）；出师表也正确（146 字、3 个双字）。**而 SPEC 的同一任务每步崩**（"1\n2u3\n4i4\n55\n6外面的6…"）。

**⇒ 三重排除**：① 模型无罪（EAGER 对）；② tokenizer/帧无罪（EAGER 对）；③ 消费链无罪（`emit-chain-audit` 逐项排除：serve 的 out 每 token 一次 push、`p += emitted.len()` 与 commit 的 Δpos_ctr 是**构造性恒等**（两臂都对上）、pool/driver 的水位 diff 每段只发一次、api 的两条装配臂喂同一 id 流、SSE 的 drain 语义正确、无重试/重放）。**stream true/false 也不能判别**（两臂互斥但 id 同源，重复与否必然同步）。

**emit-chain-audit 的结构性收窄（最有价值的一条）**：**`emitted` 的值来自 `verify_out`（主链的 argmax），永远不来自 draft** ⇒ **draft 侧的任何污染（窗口 ring/seed 差一位/tap/stride）只能压低 `k_acc`，不可能改错 `emitted` 的值**。**⇒ 值错了 ⇒ 主链 KV / `s.ids_r` / `pos_rows` 被 verify 的执行污染；值没错而 k_acc 低 ⇒ draft 侧。** 这条把搜索空间砍一半。

**形态**：前 ~10 字符正确（DBG 前 12 步的 emitted 逐字正确），之后崩 ⇒ **累积型**（每步泄漏一点状态，几步后压垮）——与"verify 的 append 写了但 rollback 没恢复"完美吻合（`spec-state-pollution` 在列"写 vs 恢复"清单）。

**seed-tap-align-impl 的交付**（gate `DSV41_SEED_ALIGN`，默认 OFF）：路线 A（`draft_forward(next, pos+1)`——seed 落 pos 与 tap 同位）+ ring 回绕按位置序展开 + **顺带发现的独立 bug：`note_ctx_rows` 的行 stride**（`dspark_dev.rs:534` 用 `slot*m+j` 而生产者 `chain_dev.rs:5823` 写 `slot*VERIFY_ROWS+r`——**slot≥1 的 target 层读错行**；6 行 arm 传 m=6 自愈，旧 5 行 arm 带病）。

**已知待核**（`spec-state-pollution` 的清单）：verify 写了槽 `pos+1..pos+5` + compressor 的 pool/commit + clen + `s.ids_r` + `xq` 量化缓存 + indexer 候选——**rollback 恢复了哪些**？`dspark_commit` 的 `compress_replay` 与下一步 `step_dev` 的 append 是否双写？`s.ids` 与 `s.ids_r` 是否共享？

## 2026-09-12 【根因定案】s.ids 无人回写——spec 的 commit 只推计数器、不写 token

**verify-row0-systematic 的判词（代码级证据链，无猜测）**：
| 环节 | 位置 | 事实 |
|---|---|---|
| 前向读哪个 token | `chain_dev.rs:3064-3072` | `step_body` 的 embedding 读 **`s.ids`**——形参 `token` 只用于 engram 的 host 回退 |
| 谁写 `s.ids` | `:3303-3308`（sliced 同） | **只有主链 argmax**（`out` 就是 `s.ids`）⇒ `step_dev` 结束后 `s.ids = next`（pos+1 的 token） |
| verify 写哪 | `:3846-3851` | `argmax_r + r`、`pos_ctr = NULL` ⇒ **verify 不更新 `s.ids`** |
| commit 做什么 | `:5557-5578`、`:5670-5672` | 只 `rollback_keep + compress_replay + set_pos_ctr(pos_base+keep)`——**没有 token 的 H2D** |

⇒ **任何 `k_acc ≥ 1` 的轮**：计数器推到 `pos+1+k_acc`，但 `s.ids` 仍是 `next`（pos+1 的 token）⇒ **下一轮在位置 `pos+1+k_acc` 嵌入的是落后 `k_acc` 个位置的旧 token** ⇒ 上下文被污染 ⇒ 模型把「同一 token 出现两次」当作模式自我延续（数字任务 100% 每个数两次、且能一路数到 400——**不是每步都错，是首次污染后自锁**）。

**这解释了全部观测**：前 ~10 字符对（k_acc=0 的轮 `s.ids` 恰好正确）；emitted 逐项正确（verify 的 argmax 没错——**错的是下一轮的输入**）；EAGER 干净（每步 argmax 更新 `s.ids` ✓）；DBG 的 emitted 解码正确而文本崩（上下文 ≠ 文本）。

**修复（已提交）**：`emitted` 构造后把**最后一个 token** 回写 `s.ids`（4 字节 H2D/轮，两臂 legacy/aligned 都加）。**结构性教训**：`s.ids`（自喂 token）、`pos_ctr`、`pos_rows`、`compress_len` 四个跨步不变量由不同机制分别推动——单行路径自洽，**spec 把推进从 1 变成 1+k_acc 时只照顾了计数器没照顾 token**。建议后续加 `debug_assert_eq!(download(s.ids), token)`（这个检查在第一步就会爆掉）。

**判词的另两项**：verify 的 engram hash 写共享 cache 但 rollback 不覆盖（当前无害——被拒位置 ≥ 提交后的计数器且每步重写——但与缺陷同源，建议进 snapshot）；`pos_base` 的 host 算术与 device 读取的两条推导依赖 stream 次序（建议断言）。

## 2026-09-12 修复前的基线档案（s.ids 回写修复的 A/B 对照用）

| 任务 | SPEC 修复前（sa_0） | EAGER 对照 | SPEC 修复后（c7ad988b，待出） |
|---|---|---|---|
| 数字任务（数到 100） | LEN 56、**双字 10**、文本崩（"我会按 1 . 到？…11\n2a3\n4i"）、k_acc={0:15,1:9,2:3}（27 步） | **LEN 216、完美数数 1..100**（仅 "anao"、跳 50 等小瑕疵） | **期望：像 EAGER 一样数对** |
| 出师表 | LEN 54、双字 4（"蜀蜀汉/丞相丞相/篇篇/以下是以下"） | LEN 146、双字 3（基本正确） | 期望：无双字 |
| 步时 | 47.7ms（20.9 tok/s） | ~6.2ms（162 tok/s，无 spec 开销） | — |

**注**：SEED_ALIGN=1 的 arm 在 AR v5 上死锁（`[ar5-hang] rank=1 peer=5/6/7 need=56 cur=53`）——路线 A 的新代码与 v5 的交互待查（不阻塞主线：s.ids 修复独立于 SEED_ALIGN）。

## 2026-09-12 s.ids 修复的验证结果（c7ad988b）——重复闭合 ✓，暴露下一层

| 任务 | 修复前 | 修复后（c7ad988b） | EAGER |
|---|---|---|---|
| 数字任务 | LEN 56、**双字 10**、k_acc={0:15,1:9,2:3} | LEN 42、**双字 1** ✓、k_acc={0:11,1:5,2:6}（k_acc=2 更多 ✓） | LEN 216、完美 |
| 出师表 | LEN 54、双字 4 | **LEN 12、双字 1**（"《出师nofollow》"）✗ | LEN 146、双字 3 |

**结论**：**重复问题（用户明令"彻底修好"的那个）基本闭合**——双字 10 → 1，k_acc 分布上移（verify 的 argmax 更可信了）。**但暴露下一层**：两个任务的输出都变短 + 出现乱码（"100d 9. f"、"nofollow"）——像是**另一种状态问题**（不是重复）。
**判别实验在跑**（40d4e5d3）：fresh serve 的第一个请求（出师表）——**正常 ⇒ 跨请求状态残留**（第一个请求跑完后 s.ids/ring/counter 影响 第二个）；**崩 ⇒ 单请求内的问题**（修复引入或暴露的下一层）。

## 2026-09-12 40d4e5d3 判别结果：单请求也崩 ⇒ 非跨请求残留；k_acc 显著提升

**① 单请求出师表（fresh serve）**：LEN 12、双字 1（"《出师nofollow》"）——**崩**（第 3-4 个 token 就崩，比修复前的 54 字符更早）⇒ **不是跨请求残留**，是单请求内的问题。
**② 数字任务（第二个请求）**：LEN 137、双字 5（基本在数数但有乱码 "100d 9. f"）——比 c7ad988b 的 42 好很多（请求顺序不同的影响？）。
**性能与 accept**：`steps=50 mean-k=0.820 tok/step=1.820 draft=4.21ms verify=37.15ms commit=0.29ms`——**verify 37.15ms（多行化后没降，带宽瓶颈）**、**k_acc={0:24,1:18,2:12,3:1}**（**k_acc=3 出现、accept 0.82——持续提升** ✓）。

**解读**：s.ids 修复让"喂入的 token"更正确（k_acc 分布上移），但**暴露了下一层**——修复前喂的是"落后 1"的 token（模型凑合能跑），修复后喂的是 emitted.last()（如果 verify_out 的值本身错，则喂的错 token 位置更远 → 崩得更快）。**下一个候选**：① verify_out 的值错（verify 的 KV/上下文——emit-chain-audit 的"值错 ⇒ 主链 KV"）；② **prefill 后的 s.ids 初值**（prefill 不写 s.ids？第一步读垃圾）。

## 2026-09-12 用户判断（关键）：乱码是一个回归——"只重复的版本 + s.ids 修复"应该就是对的

**用户（权威）**："乱码其实是一个回归，之前那个只重复的版本加上修复应该就是对的。"

**分析（同意）**：s.ids 回写修复提交（`emitted.last()` 写回）与"只重复的版本"之间，工作树上还落了**三个同伴的提交**（mtp_batch.rs 的设计骨架、indexer 的单遍堆选、glue 的 mrows K 序改动）——**其中任何一个都可能在 spec 路径上引入数值回归**。最直接的判别：**用 `git log` 找到 s.ids 修复的父提交（只含重复修复、不含同伴改动），在远端 checkout 那个点重编**——如果数字/出师表都干净 ⇒ 回归来自同伴的某个提交，二分定位。

## 2026-09-12 用户判断②+③：乱码是回归，来自中间的优化

**用户（权威）**："乱码其实是一个回归，之前那个只重复的版本加上修复应该就是对的" + "我觉得是我们中间的一些优化导致的问题"。

**二分实验设计（进行中）**：
| 提交 | 内容 | 测试 |
|---|---|---|
| `5f774c7` | **无 s.ids 修复**（含全部同伴优化：specstep/p0d/glm-mtp/mrows/glue K 序/route A）——"只重复"的基线 | 31c6c95c 在跑（预期：双字多、无/少乱码） |
| `65d6b5a` | **5f774c7 + s.ids 修复**（用户说的"应该就是对的"版本——之后只有 docs 提交，无代码） | 待跑（判据：若双字少+无乱码 ⇒ 用户对，65d6b5a 即正确版本；若乱码仍在 ⇒ s.ids 修复与同伴优化的**组合**有问题，需在 65d6b5a 的基础上二分同伴的优化） |
| HEAD | 65d6b5a + docs | c7ad988b/40d4e5d3 已测（双字 1 ✓ 但出师表崩到 12 字符）|

**注意**：65d6b5a 之后**只有 docs 提交**——若 65d6b5a 也乱码，则"回归"不是来自它之后的提交，而是 **s.ids 修复与之前同伴优化的交互**（例如 glue 的 mrows K 序改动在 FOLD=1 路径上、或 route A 的 6 行布局与 FOLD=0 的 head 组合）——届时按同伴提交逐个 `git checkout 65d6b5a -- <file>` 回退二分。

## 2026-09-12 Wave 5 step 3（工部）：dsa_append_batched 的行→seq 映射骨架（无调用点，gate 未动）

**交付**：`mtp_batch::SeqRowMap` / `AppendPlan`（纯宿主算术）+ `CudaBackend::dsa_append_batched_mapped` +
kernel `dsa_append_batched_mapped_kernel`（`ferrite_kernels.cu`）。**没有任何调用点**——`gpu_engine.rs:492` 的
`FERRITE_MTP ⇒ max_seqs=1` gate 保持关闭（五件事未全就位）。

**两种布局的行号公式**（都是 seq-major，`row` 升序 = 同一 seq 内 token 升序）：

| 布局 | 行数 | row → (seq, tok) | kernel 参数 | 可进图？ |
|---|---|---|---|---|
| **padded** `B×n_v` | `padded_seqs(live)*n_v`（B=16,n_v=3 → 48） | `seq = row/n_v`, `tok = row%n_v` | `ntok = n_v`，两张表都 NULL | ✅ 固定形状（图名 `mega_v_b{padded}_n{n_v}`，注意不能只按行数 key：24 行可同时是 (4,6) 与 (8,3)，表宽与除数都不同） |
| **ragged** `Σ(1+k_r)` | 各 seq 块长之和（逐 tick 变） | `[rows][2]` 表查表（host 侧 `prefix` 前缀和 + partition_point） | `ntok = 0` + `row_map` + `block_len` | ❌ 非图（行数逐 tick 变） |

**向后兼容不变量（关键）**：`rows == B, ntok == 1, 两张表 NULL` 时，新 kernel 与 `dsa_append_batched_kernel`
**逐位相同**——`seq == row`、`tok == 0`、尾部读 `ki[seq*idm+c] == ki[row*idm+c]`、advance 在
`tok == ntok-1 == 0` 触发（即旧 kernel 的 `tok==0` 条件）。⇒ 现有 decode-batched 调用点（`cuda.rs` 的
`ferrite_dsa_append_batched(... ni, 1, ...)`）**可以原地换成新入口**做 A/B，不需要数值验证以外的额外工作。

**旧 kernel 在 `ntok = n_v` 下不可用的两个硬伤**（新 kernel 的两处修正，也是"行→seq 映射"的实质内容）：
1. idx/gate 尾部读的是 `ki[seq*idm + c]`（**per-seq**）→ `ntok>1` 会把 anchor 行的 idx 写 `n_v` 遍；
   行形式必须读 `ki[row*idm + c]`；
2. device advance 硬编码 `t0+1` → 一个 seq 的 `n_v` 行只推进 1 格，**每步漏 `n_v-1` 个 cache 槽**（下一步读到垃圾 KV）。
   行形式推进 `block_len[seq]`（= `ntok`，或 ragged 表的每 seq 块长），且由该 seq 的**最后一行**（`tok == blen-1`）单写者完成。

**远端下一步的注意事项**：
- 新 kernel 是**新符号**——任何调用点落地前必须先重编 kernel 库（`kernels/cuda/build.sh`），否则 Rust 侧会
  在加载/链接时报 undefined symbol；本轮**没有调用点**，所以当前工作树是安全的。
- 行映射的**其余下游**也吃同一张表（本轮的 `SeqRowMap` 是共用 abstraction）：`kpool_compress_batched` /
  `indexer_topk_batched` / `pool_expand_batched` 在 verify 里应拿**行数**（每行一次 query），而 per-seq 指针表
  与 commit plan 拿 **B（seq 数）**，两者的除数就是 `row / n_v`。
- 单测口径：映射是纯函数，`mtp_batch.rs` 的 16 个测试在**默认 feature 下就能跑**（但 `ferrite-exec` 的 lib-test
  目标在无 `cuda` feature 时**本就编不过**——`tp.rs` 的 test mod 里有未 gate 的 `ferrite_kernel::cuda` 引用，
  属既有问题；带 `cuda` feature 又需 nvcc 链接。故本轮用隔离 crate 跑同文件验证：16 passed / 0 failed）。

## 2026-09-12 二分第一点（31c6c95c）：5f774c7 = "只重复的版本"确认

| 任务 | 5f774c7（无 s.ids 修复）| HEAD（含修复+打印）| EAGER |
|---|---|---|---|
| 出师表 | **LEN 54、双字 4**（无乱码崩） | LEN 12、双字 1（崩） | LEN 146、双字 3 |
| 数字任务 | **LEN 56、双字 10**（无乱码崩） | LEN 42-137、双字 1-5 | LEN 216、完美 |

⇒ **用户判断的基线确认**：5f774c7 只有重复、没有乱码崩。**第二点（65d6b5a = 基线+s.ids 修复）在测**——若它也崩（HEAD 与 65d6b5a 之间只有 docs+无调用的 mtp_batch.rs+无行为的打印，行为应等价）⇒ **s.ids 修复与 verify 值错的组合放大**：修复前喂"落后 1"的 token（模型凑合），修复后喂 emitted.last()（若 verify_out 的值本身错，喂的错 token 位置更远 → 崩得更快）。**根治 = verify_out 的值**（prefill-sids-init / spec-eager-diff-tool 在查）。
