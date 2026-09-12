# tcgen05 残余 misaligned — compute-sanitizer 定位方案

> 工部 · 2026-09-12 · **只读调查 + 本文件（唯一产出）**。未执行任何 GPU 命令、未改动任何源码。
> 现场核对：`kernels/cuda/dsv41_experts_mxf4.cu`（HEAD 工作树）、`kernels/cuda/build.sh`、
> `crates/ferrite-dsv41/src/{serve.rs,bin/dsv41-run.rs}`、`crates/ferrite-exec/src/tp.rs`、
> `crates/ferrite-models/src/dsv41/{load.rs,device.rs,chain_dev.rs}`、
> `scripts/{tcgen05_smoke.sh,tcgen05_bench.sh,dsv41_tcgen05_mxf4_verify.sh}`。
> 所有行号对当前工作树现场核对；推算项标注口径。

---

## 0. 结论摘要（先看这个）

1. **TP8 是「单进程、8 线程」不是「8 进程」**（`serve.rs:182` 一个 owner 线程 → `tp::run_ranks(world, …)` →
   `ferrite-exec/src/tp.rs:4087` 的 `std::thread::scope`，每 rank 一线程 + 自己的 `cudaSetDevice(rank)`）。
   ⇒ **一次 `compute-sanitizer` 调用即可覆盖全部 8 个 rank**，无需 8 份进程；报告里的 device id 就是 rank
   （`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` 下逻辑设备 == 物理 GPU == rank）。
2. **byte-fallback 有两类读点永远救不了**，这是「两轮修复后仍 misaligned」最可能的结构性原因：
   - `cp.async.bulk.shared::cluster.global`（TMA/G2S，`e4_bulk_g2s` `:4789`）——**global 源地址与 smem 目的
     都硬要求 16B 对齐**，无法像 `ld_uint4_a16` 那样拆成 16 个字节；
   - `cp.async.bulk.prefetch.L2.global`（`dsv41_w2_pf_bulk` `:3000`）——**srcMem 硬要求 16B 对齐**，
     同理不可 byte 化。
   ⇒ 第 1/2 轮修的 `ld_uint4_a16/ld_uint2_a8` 系都是**同步 LDG**；如果真凶在 bulk 路径，byte-fallback
   在原理上就盖不住，必须换成「基址/步长对齐」的修法。
3. **launcher 有 host 侧 `al16` 门**（`tc5::e4` `:5218-5222`；grouped `:5917-5922`）：平面基址/步长不对齐会
   **返回 `cudaErrorInvalidValue`（拒绝发射）而不是 fault**。⇒ 若 fault 真发生在某个 tcgen05 kernel 内，
   说明**对齐问题不在被检查的基址上，而在未被检查的内部偏移**（`row*kbytes`、`pk0`、`rr*nsf`、
   `e*stride` 之外的项）。这条把「嫌疑面」从「基址」缩到「内部偏移」。
4. **单 GPU（TP1）能否复现，判据是「故障类别」**：形状/步长类（rank 无关）能复现；TP-shard seam 类只在
   TP>1 出现。⇒ 方案里**同时**跑 TP1（便宜、快、可迭代）与 TP8（定版），并把「TP1 复现 / 不复现」当作
   免费的分类证据。
5. **`--launch-timeout` 容易误用**：它是「等待应用发起 CUDA 活动 / attach thunk 的时限」，
   **不是** kernel 超时。ferrite-serve 是「先绑监听、首个请求才 `ensure_loaded()`」，加载期很长，写成
   `--launch-timeout 120` 有假超时风险 ⇒ 首轮用 `--launch-timeout 0`（不限）。
6. **最便宜的「正证据」路径不是 serve，是独立 harness**（`kernels/cuda/tests_tcgen05_*.cu`）：不加载权重、
   秒级起 kernel、可直接喂「故意不对齐的 view」把故障稳定复现，再上 sanitizer。serve 只是**最终定版**。

---

## 1. 事实核对（哪条代码在跑、守没守住）

### 1.1 归因：三个 tcgen05 kernel + 一个 SIMT 回退

| kernel | 行号 | 什么时候跑 | 本次故障相关性 |
|---|---|---|---|
| `expert_tcgen05_gateup_e4_kernel` | `:4888` | **prefill / eager decode 的单行 `moe()`**（swapAB e4m3） | ⭐ **191ms 快速失败的头号候选** |
| `e4m3_gemm_kernel` | `:5517` | 多行 dense tile | `e4x_tile=false` **硬编码**（`chain_dev.rs:11523`）⇒ 本路径**永远不发** |
| `e4m3_gemm_grouped_kernel` | `:5904` | **spec verify 的多行 `moe_rows()`** | 只在 `DSV41_SPEC/DSPARK` 武装时到 |
| `expert_gemv_fp4_batched_kernel` | `:1400` 域 | 上面任一门关掉时的 SIMT 回退 | 第 1/2 轮守卫的宿主 |

**关键推论（回答「prefill 的哪个 kernel」）**：prefill 是「**每个 prompt token 一次 forward**」的单行路径
（`serve.rs` 的 `RankCmd::Prefill` 注释：「consume the prompt, one forward per prompt token」），
它走的是**单行 `moe()`** ⇒ 命中 `expert_tcgen05_gateup_e4_kernel`（swapAB），**不是** grouped kernel。
所以「191ms 快速失败」如果发生在 prefill，**头号嫌疑是 swapAB e4 kernel**；grouped kernel 是 verify 路径，
要等 spec/dspark 武装后的多行步才到。

### 1.2 第 1/2 轮修的是什么（覆盖边界，必须写进证据）

- `ed720d6` 加了 `ld_uint2_a8`（`:170`），打在 **pair body** 的直读点（`:1722/:1727/:1735/:1738/:1741/
  :1753/:1756/:1758`）。
- `pair_body = ((fuse_swiglu != 0) || ILV) && (b_split > 0)`（`:1405`）。
  **冒烟臂 `GATEUP_FUSE=0 + ILV=0 ⇒ pair_body == false`** ⇒ 走 **split body**，而这些直读点**不被执行**。
- 第 2 轮补了 split body 的 4/2 字节兄弟（`ld_uint32_a4` / `ld_uint16_a2` 同类，见 `:198-232` 的注释块）
  + `:5009` 一带的 `ld_uint4_a16`（SF prologue 的 scale 读，`tc5::e4`）。
- **修正一个前提**：`ILV=0` ⇒ `ilv_ok()==false`（`load.rs:767-778`，把 `gateup_fuse()` 作为合取项）
  ⇒ **`interleave_gateup_fp4_kernel`（`:2479`）在冒烟臂里根本不会被调用**。所以「load-time interleave
  也可能 misaligned」这条在 `ILV=0` 下**不成立**；若实测确实发生，说明**跑的那一轮 env 不是 ILV=0**
  ⇒ **必须先 dump `/proc/<pid>/environ` 把实际门钉死**（见 §2.1）。

### 1.3 未被任何 byte-fallback 覆盖的读点（真正的残余嫌疑）

| 读点 | 行号 | 对齐要求 | byte-fallback 能救？ |
|---|---|---|---|
| `e4_bulk_g2s`（A 权重行，源 = `w1p/w3p`） | `:4977` | **16B（硬）** | ❌ 不能 |
| `e4_bulk_g2s`（B 激活，源 = `act`） | `:4986` | **16B（硬）** | ❌ 不能 |
| `dsv41_w2_pf_bulk`（L2 prefetch） | `:3002` | **16B（硬）** | ❌ 不能 |
| SF prologue 的 scale 读（`w1sp/w3sp`） | `:5024/:5032` | 16B（LDG） | ✅ 已救（第 2 轮） |
| SIMT pair/split body 直读 | `:1722…:1758` | 4/8/16B | ✅ 已救（第 1/2 轮） |
| `tcgen05` MMA 的 **smem descriptor** | `e4x_make_desc` | smem 16B | ❌ 不可 byte 化（**另一类故障**） |
| TMEM 读 `tc_ld_x16` | `:5090` 一带 | TMEM 语义 | ❌ 非 global fault |

⇒ **「两轮 byte-fallback 没解决」的最佳解释**：真凶落在 ❌ 那一组，而这一组**不是字节可拆的**。

---

## 2. Phase 0 —— 无 GPU 的先决条件（一个 GPU 都不该花在这）

### 2.1 把「到底跑了什么」钉死（必做，否则归因无效）

```bash
# 1) 上一次失败轮的真实 env（不是 shell 的 env）
ssh ubuntu@<node> "tr '\0' '\n' < /proc/\$(pgrep -x ferrite-serve | head -1)/environ \
  | grep -E '^DSV41_|^CUDA_' | sort"        # 注意：进程已崩则先复现一轮再抓

# 2) 崩溃时刻的日志：先看 load 边界，再看 fault
ssh ubuntu@<node> "grep -nE 'tp pool ready|misaligned|illegal|fault|CUDA error|716|panic' ~/<本案日志>"
```

**判定**：
- 出现 `[dsv41] tp pool ready: 8 ranks loaded`（`serve.rs:228`）**之前**就崩 ⇒ **load 期 kernel**
  （若 env 里 `DSV41_EXPERT_ILV=1`，头号是 `interleave_gateup_fp4_kernel`；若 `ILV=0`，转查
  `load_tensor`/DMA 路径，不在本 doc 范围）。
- 在它**之后**、首个 prefill 内崩 ⇒ **`expert_tcgen05_gateup_e4_kernel`**（§1.1）。
- 时间戳对齐：191ms 这个数**相对什么**？（进程启动？首个请求？）——把口径写进报告，否则无法解释
  「第 1 轮 18.8s / 第 2 轮 191ms」的差异。**这一条比任何 kernel 名都重要。**

### 2.2 让 sanitizer 能出「kernel 名 + 指令 + 行号」

compute-sanitizer 读 **debug info** 才能给源码行；`build.sh:129` 现在的 flag **没有 `-lineinfo`**。

```bash
# 改 build.sh（或临时加一个环境变量口子）给 nvcc 加 -lineinfo：
#   "$NVCC" -O3 -shared -Xcompiler -fPIC $FAST_MATH_FLAG -lineinfo -std=c++17 ...
# ⚠️ -lineinfo 不改变 codegen；但会改 .cu 内容/产物 ⇒ BUILD_ID 会变 ⇒ 必须双产物一起重编：
cd ~/ferrite
bash kernels/cuda/build.sh 103a                       # 重编 .so（写新 .build_id）
touch crates/ferrite-kernel/build.rs && cargo build --release   # Rust 侧读新 .build_id
```

> 是否需要 `-G`？**不要**。`-G`（device debug）会关掉大量优化、改 codegen、拖慢 100x+，并且可能**掩盖**
> tcgen05/PTX inline asm 的时序。`-lineinfo` 足够给「文件:行 + SASS 指令」。
> ⚠️ `-lineinfo` 是**源码改动**（build.sh 一行）——按工部纪律，**待批准**（见 §8）。

### 2.3 其它先决条件

```bash
which compute-sanitizer || ls /usr/local/cuda/bin/compute-sanitizer   # 必须存在
compute-sanitizer --version && nvcc --version                          # 版本需支持 sm_103a（CUDA ≥ 12.8 系）
bash scripts/tcgen05_smoke.sh --dry-run      # 五符号齐 + .build_id 双产物一致（无 GPU 也能跑）
```

> **sm_103a 支持是硬前提**：compute-sanitizer 的版本低于 GPU 架构时会直接拒绝 attach 或给出误导性
> 报告。若节点 toolkit 太旧，**先在 1 张 GPU 上用独立 harness（§4）验证 sanitizer 能正常报错**，
> 再上 serve。

---

## 3. Phase 1 —— sanitizer 命令设计（分层，逐层加约束）

### 3.1 统一环境头（所有轮共用）

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=$HOME/ferrite/kernels/cuda
export DSV41_KERNELS=$HOME/ferrite/kernels/cuda/libferrite_kernels.so
# 首轮必须：把异步 fault 归到真正 fault 的那次 launch
export CUDA_LAUNCH_BLOCKING=1
# 显式关图：去掉「只记录不执行 / 静默降级」的执行路径变量
export DSV41_VERIFY_GRAPH=0 DSV41_GRAPH_MOE=0
export DSV41_TIMING=1
# arm 门（与 tcgen05_smoke.sh 一致；注意两个 starts_with('1') 的门不能写 =true）
export DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_EXPERT_ACT_E4M3=1 DSV41_MOE_BATCH=1
export DSV41_EXPERT_TCGEN05_E4M3=1 DSV41_EXPERT_GROUPED=1
export DSV41_GATEUP_FUSE=0 DSV41_EXPERT_ILV=0
```

### 3.2 S0 —— 全 rank、单进程、首触（推荐第一条）

```bash
compute-sanitizer \
  --tool memcheck \
  --launch-timeout 0 \
  --destroy-on-device-error kernel \
  --show-backtrace device \
  --print-limit 0 \
  --error-exitcode 86 \
  --log-file /tmp/san_s0.txt \
  ./target/release/ferrite-serve --model dsv41 --serve --tp 8 \
    --model-dir /opt/dlami/nvme/models/DeepSeek-V4.1-Flash --port 8699
```

要点逐条：

| 参数 | 取值与理由 |
|---|---|
| `--tool memcheck` | 默认工具；只它报 **Misaligned Address / err 716**。`racecheck/initcheck/synccheck` 与本题无关，别开（更慢、更多假阳性）。 |
| `--launch-timeout 0` | **不限**。它是「等应用发起 CUDA 活动的时限」，不是 kernel 超时；serve 先绑监听、首个请求才加载，写 120 有假超时风险（会打印 `No attachable process found … timed-out`，见 NVIDIA 论坛案例）。 |
| `--destroy-on-device-error kernel` | **枚举模式**：只销毁出错 kernel 的状态、**保留 context**，让后续 kernel 继续跑并各自报错。**默认 `context` 会在第一个 fault 后毁掉 context**，你只会看到「1 misaligned」——这正是本轮要打破的。**首轮先跑默认 `context` 拿一条干净报告，再跑 `kernel` 枚举全部。** |
| `--show-backtrace device` | 要 **device 侧**调用栈（带 `-lineinfo` 才有 `文件:行`）。`yes` 也行但混入 host 帧。 |
| `--print-limit 0` | 关掉打印上限（不确定取 0 还是极大值时，先 `compute-sanitizer --help | grep -i print-limit` 现场核对）。 |
| `--log-file` | 报告落盘；**必须**，因为 serve 前台日志与 sanitizer 报告会交错。 |
| `--error-exitcode 86` | 让「有错」在脚本里可判（serve 正常时不会退出，靠请求驱动）。 |

**驱动一次请求**（sanitizer 只是 instrument，不会自己去发 HTTP）：

```bash
# 另一个终端 / 后台：等 /health → 发「你好」max_tokens=20（抓 crash）
curl -s localhost:8699/health
curl -s localhost:8699/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":20}' | head -c 400
```

**观察窗口**：sanitizer 下整体**慢 10–100x**。191ms 的故障若在 prefill，sanitizer 下可能变成数秒~数十秒；
若在 load 期，会先经历一次极慢的权重加载（可能数十分钟）。⇒ **S0 之前先做 §4 的独立 harness，能省掉
一次全量加载**。

### 3.3 S1 —— 单 GPU 隔离（便宜、可迭代）

```bash
# 只留一张卡；TP 仍为 8 会让 7 个 rank 找不到设备（cudaSetDevice 越界）——
# 所以单卡必须配 --tp 1（世界大小 = 设备数），且只能测「shape/stride 类」故障（见 §5 分类）
export CUDA_VISIBLE_DEVICES=7
compute-sanitizer --tool memcheck --launch-timeout 0 \
  --destroy-on-device-error kernel --show-backtrace device \
  --log-file /tmp/san_s1.txt \
  ./target/release/ferrite-serve --model dsv41 --serve --tp 1 \
    --model-dir /opt/dlami/nvme/models/DeepSeek-V4.1-Flash --port 8698
```

> ⚠️ **`--tp 1` 与 8 张卡同时可见是两个不同实验**：`CUDA_VISIBLE_DEVICES=7` 把逻辑设备数压成 1，
> `--tp 8` 会直接在建 rank 池时失败。要么「8 卡 + tp 8」（S0），要么「1 卡 + tp 1」（S1）。
> 不要写「8 卡可见 + CUDA_VISIBLE_DEVICES=7 只跑 rank 7」——那不是隔离，只是把 rank 0..6 掐死。

### 3.4 S2 —— 只 instrument 一次 kernel 的事件（可选，降噪）

若 S0/S1 报告被无关噪声淹没，用 `--kernel-name` / `--launch-skip` + `--launch-count`（按 `--help` 现场核对
拼写）把窗口收到首次 prefill 的 MoE 段。**不建议首轮就用**——先看全量报告，别提前假设是哪个 kernel。

---

## 4. Phase 1' —— 独立 harness（强烈推荐，最便宜的「正证据」）

serve 的两个成本（加载 + 全栈噪声）在定位阶段都是浪费。仓库已有现成 harness：

```bash
ls kernels/cuda/tests_tcgen05_*.cu
#   tests_tcgen05_mxf4.cu            (tc5::mxf4 家族)
#   tests_tcgen05_mxf4_gateup.cu     (swapAB mxf4，5 个 parity case)
#   tests_tcgen05_mxf8f6f4_1x.cu     (e4m3/f8f6f4 家族，75KB，最全)
#   tests_dsv41_r2_parity.cu         (R2 parity 全家桶)
```

```bash
nvcc -O3 -lineinfo -gencode arch=compute_103a,code=sm_103a -o /tmp/t_e4 tests_tcgen05_mxf8f6f4_1x.cu
compute-sanitizer --tool memcheck --launch-timeout 0 --show-backtrace device \
  --destroy-on-device-error kernel --log-file /tmp/san_harness.txt \
  env CUDA_VISIBLE_DEVICES=7 DSV41_EXPERT_TCGEN05_E4M3=1 /tmp/t_e4
```

**为什么它比 serve 强**：
1. **秒级**起 kernel，不用加载 40GB 权重；一轮迭代从「半小时」降到「几秒」。
2. **可控输入**：可以**故意**把一个 view 做成 `base+4`（模拟 TP-shard seam）或让 `nsf`/`kbytes` 非 16 倍数，
   把「哪一类偏移会 fault」逐项证伪/证实——这是 serve 做不到的。
3. **隔离**：一个 kernel、一个设备、无 collectives、无图捕获，报告干净。
4. `dsv41_tcgen05_mxf4_verify.sh` 第 3 步已经把「一个 case 一个进程」的隔离纪律写好了
   （脚本 `:267` 的注释：device-side fault 会毒化整个 context ⇒ 同 run 后续 case 报伪 rc=46）。

> **代价**：harness 里的 view 是我们自己构造的，可能与 Rust 侧 `DevBuf::view` 的真实偏移**不同**
> ⇒ harness PASS ≠ serve PASS。所以 harness 用来**缩小嫌疑**，serve（S0）用来**定版**。

---

## 5. 关键问题逐条答复

### Q1. TP8 的 8 个 rank，sanitizer 能跟踪吗？需要单 rank 测试吗？

- **能，且是一次调用。** `TpRankPool::new`（`serve.rs:151-192`）只 spawn **一个** `dsv41-tp-pool` 线程
  （`:182-192`），内部 `tp::run_ranks(world, …)` 用 `std::thread::scope`（`ferrite-exec/src/tp.rs:4087`）
  为每个 rank 起一个线程，每个 rank 在自己的线程里 `cudaSetDevice(rank)`。**8 rank = 1 进程 = 8 线程
  = 8 个 CUDA context** ⇒ compute-sanitizer attach 一次就全覆盖。
- **rank ↔ 报告里的 device id**：逻辑设备号 = `CUDA_VISIBLE_DEVICES` 里的**位置**。`0,1,2,3,4,5,6,7`
  时逻辑 == 物理 == rank。看到 `Device 5` 就是 **rank 5**。**但**：fault 是 sticky 的
  （`devrt.rs:1066` 的注释：「misaligned address … (rank-local)」），sanitizer 报的可能是**首个** fault；
  要枚举所有 rank，用 `--destroy-on-device-error kernel`（见 §3.2）。
- **何时才需要「单 rank」**：只有当你要排除「rank 间 collective 干扰」时——用 **TP1**（§3.3），
  而不是「8 卡只跑 1 rank」。

### Q2. 单 GPU（TP1）能否复现？TP-shard 的 view 对齐问题是否只在多 GPU 下？

按故障成因分两类，**TP1 的结果本身就是分类证据**：

| 类 | 成因 | TP1 复现？ | 典型点 |
|---|---|---|---|
| **A｜形状/步长类** | kernel 内部偏移不是 granule 的整数倍（rank 无关） | ✅ 会 | `rr*nsf`（scale 行字节数 `nk_blk=k/32`）、`pk0`、`row*kbytes`、`g*(kPackK/2)` |
| **B｜TP-shard seam 类** | `w1_base + e*w1_stride` / `b_base + e*b_stride` 的 view 基址落在非 16B 边界 | ❌ **只在 TP>1** | `load.rs` 的 per-rank shard 切分；`device.rs` 的 `DevBuf::view` |
| **C｜布局类** | `ILV` / `fuse` 决定的 body 选择（A/B 之外的第三条轴） | ✅ 会（TP1 也能跑 ILV=1） | `pair_body`（`:1405`） |

**但有一条重要的反证**：`tc5::e4` launcher（`:5218-5222`）与 grouped launcher（`:5917-5922`）都**检查了**
平面的 `al16` 与 `str16`。⇒ **类 B 会在 host 侧被拒（`cudaErrorInvalidValue`），表现为「arm 静默回退」，
而不是 device fault**。所以：
- 若 S1（TP1）**复现** fault ⇒ 类 A/C，且**与 shard 无关**——最可能的结构性原因在**未检查的内部偏移**
  或 **bulk copy**（§1.3）。
- 若 S1 **不复现**、S0（TP8）复现 ⇒ 要么是类 B（但被 launcher 挡掉，应表现为回退）、要么是
  **多 rank 并发/collective 的次生现象** ⇒ 这时 sanitizer 的 device id 是唯一能分辨 rank 的证据。

**结论**：**两个都跑**。TP1 便宜（1/8 权重、无 collective）先跑；TP8 定版。

### Q3. 「191ms 快速失败」到底卡在哪个 kernel？

三步把它钉死，**不要靠猜**：

1. **口径**：确认 191ms 是「进程启动 → 崩」还是「首个请求 → 崩」。前者把范围拉到 load 期，后者锁在 prefill。
2. **load 边界**：`grep 'tp pool ready'`（`serve.rs:228`）出现与否（§2.1）。
3. **正证据**（三选一，按成本）：
   - `CUDA_LAUNCH_BLOCKING=1`：让 err 716 归到**真正 fault 的那次 launch**（异步 fault 默认 sticky，
     不在 sync 点不报——这是「错误离真凶很远」的根因）。
   - **nsys**：`nsys profile -t cuda --stats=false -o /tmp/tc5_san …` + `nsys stats --report cuda_gpu_kern_sum`。
     数 `e4m3_gemm_grouped_kernel` / `expert_tcgen05_gateup_e4_kernel` / `expert_gemv_fp4_batched_kernel`
     的调用数——**谁 > 0 谁就是嫌疑人**（这也是 `tcgen05-retest-after-guardfix.md` §4.4 方案 A 的金标准）。
     ⚠️ 计时别与 profiling 同轮。
   - **sanitizer 报告本身**（带 `-lineinfo` 就有 kernel 名 + 文件:行 + SASS 指令）。

---

## 6. 报告解读（格式 → 归因）

memcheck 的 misaligned 报告形如（字段口径）：

```
========= Invalid __global__ write of size 8 bytes
=========     at 0x7f...  in dsv41_experts_mxf4.cu:1727:expert_gemv_fp4_batched_kernel(...)
=========     by thread (0,0,0) in block (12,0,0)
=========     Address 0x7f8a2c004004 is misaligned (actual: 4, expected: 8)
=========     and is located in the global memory region
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     ...
=========     Device: 5
```

逐字段读法：

| 字段 | 含义 | 归因动作 |
|---|---|---|
| `Invalid __global__/__shared__ read/write` | 哪一类访问 | `__shared__` ⇒ 是 smem/descriptor 类（另一条修法）；`__global__` ⇒ 权重/激活读 |
| `in <file>:<line>:<kernel>(…)` | **kernel 名 + 源码行**（要 `-lineinfo`）+ 参数（可分辨模板/形状） | **这就是要找的「kernel 名」**；对照 §1.3 表定位到具体读点 |
| `at 0x…` | faulting **PC/SASS 地址** | `cuobjdump -sass libferrite_kernels.so` / `nvdisasm` 反查是哪条指令（`LDG.E.128` / `CPASYNC…BULK` / `MMA`） |
| `thread (x,y,z) in block (X,Y,Z)` | 线程/块 | 与 `lane`、`row`、`group` 推导对得上 → 确认是哪条循环的哪个偏移 |
| `Address 0x… is misaligned (actual: A, expected: E)` | **出错地址 + 实际/要求对齐** | `actual` 就是那个「非对齐偏置」的指纹：`actual=4 expected=8` ⇒ 4 字节偏；`expected=16, actual=8` ⇒ 16B 要求被 8B 偏破 |
| `Device: N` | 哪个设备（= rank） | 判断「全体 rank 一致」还是「单 rank 特例」（rank-specific ⇒ 类 B 的 shard seam） |
| `Saved host backtrace …` | host 侧调用栈 | 反查是 prefill 还是 verify 路径（哪个 Rust 调用点） |

**配套反汇编**（拿到指令级正证据）：

```bash
cuobjdump -sass $HOME/ferrite/kernels/cuda/libferrite_kernels.so | \
  awk '/Function : .*expert_tcgen05_gateup_e4_kernel/,/^\s*$/' | grep -nE 'LDG|CPASYNC|LDS|MMA' | head -40
# 行号对齐：cuobjdump 会带 .loc；用 -lineinfo 后 nvdisasm --print-line-info 更直观
nvdisasm --print-line-info --print-code <cubin>
```

**memcheck 的盲点（必须知道，否则会「跑了 sanitizer 还是 0 报错」）**：
- **异步 bulk 拷贝**（`cp.async.bulk*` / TMA tensor descriptors）**不保证**被 memcheck 的
  Misaligned 检查覆盖；`tcgen05` 的 **TMEM** 访问同样不在其 global 模型里。
- ⇒ 如果 S0 报 **0 errors 但进程仍崩**，**不要**下「对齐没问题」的结论：转 `CUDA_LAUNCH_BLOCKING=1`
  + nsys 的 kernel 名，并用 `cuobjdump -sass` 看该 kernel 的 bulk/MMA 指令——**这是 §1.3 里 ❌ 那组
  的专用取证路径**。

---

## 7. 归因矩阵（跑完必做）

| sanitizer / 观测 | 解释 | 动作 |
|---|---|---|
| 报告指名 **`expert_tcgen05_gateup_e4_kernel`** + `__global__` + `expected 16` | prefill swapAB 臂的内部偏移或 bulk 源 | 按 `:4977/:4986`（bulk）vs `:5024`（LDG）分流；**bulk ⇒ 改基址/步长对齐，不是 byte-fallback** |
| 报告指名 **`e4m3_gemm_grouped_kernel`** | verify grouped 臂 | 按 `:6006`（`ld_uint4_a16` 已守）vs 其 bulk 路径分流 |
| 报告指名 **`expert_gemv_fp4_batched_kernel`** | SIMT 回退（**说明 arm 没生效**） | 先查 decline 告警 / env / 五符号；**别**把它记在 tcgen05 账上 |
| 报告指名 **`interleave_gateup_fp4_kernel`** | 只在 `ILV=1` 可能 | 立刻回查 `/proc/environ`：**说明跑的不是 ILV=0 臂**，前面所有归因要重做 |
| 报 `__shared__` misaligned / descriptor | smem/MMA descriptor 类 | 另一条修法（smem 布局的 16B 对齐），与 global byte-fallback 无关 |
| **0 errors 但进程仍崩** | memcheck 盲区（bulk/TMA/TMEM） | 转 `CUDA_LAUNCH_BLOCKING` + nsys + `cuobjdump -sass`（§6 盲点） |
| 报 **多个** device 同一个地址偏置 | 全体 rank 同构 ⇒ 类 A/C | 形状/步长类，TP1 也能复现 ⇒ 在 harness 里修/验 |
| **只在某一个 device** | rank-specific ⇒ 类 B（shard seam） | 查该 rank 的 `w1/w3` view 基址；修法是**对齐 view**，不是 byte-fallback |

---

## 8. 需要批准 / 上报的「方案外」项

1. **`build.sh` 加 `-lineinfo`**（1 行，源码改动）——否则 sanitizer 报告**没有 `文件:行`**，
   「kernel 名 + 指令 + 地址」里会缺最关键的一维。**请批准**（或者给 build.sh 加一个
   `FERRITE_LINEINFO=1` 的 env 口子，避免污染正常构建）。
2. **独立 harness 的 `-lineinfo` 编译**（`tests_tcgen05_*.cu`）——不改仓库，只是编译命令，属建议项。
3. 若结论指向 **bulk copy（`cp.async.bulk`）**：修法是**对齐 view/步长**（可能需要改
   `load.rs` 的 shard 切分或 `device.rs` 的 `DevBuffer::view`），**不是**再加一个 byte-fallback；
   这是一个**独立的工程设计**，应单独立项、单独评审（工部不自行改方案）。
4. `--launch-timeout` / `--destroy-on-device-error` 的**取值**（`0` vs `kernel`）——本 doc 给的默认值
   基于官方文档语义；上机前请用 `compute-sanitizer --help` **现场核对**（版本间 flag 曾变动）。

---

## 9. 一页执行顺序

```
Phase 0（无 GPU）
  0a 抓上一轮的 /proc/environ + 崩溃日志 → 确认「ILV/fuse 实际取值」+「崩在 load 还是 prefill」
  0b 确认 compute-sanitizer 存在且支持 sm_103a
  0c 【待批准】build.sh 加 -lineinfo → 重编 .so + cargo build --release（双产物一致）
  0d bash scripts/tcgen05_smoke.sh --dry-run   → 五符号齐

Phase 1'（1 GPU，最便宜，先做）
  1a nvcc -lineinfo 编 tests_tcgen05_mxf8f6f4_1x.cu（e4m3）与 tests_tcgen05_mxf4_gateup.cu（swapAB）
  1b compute-sanitizer --tool memcheck --destroy-on-device-error kernel --show-backtrace device
     → 拿 kernel 名 + 文件:行 + 指令 + (actual,expected)
  1c 在 harness 里构造「故意非对齐的 view」逐项证伪：base+4 / nsf 非 16 倍数 / kbytes 非 16 倍数
     → 判定类 A / 类 B / bulk 路径
Phase 1（TP1，便宜迭代）
  2  CUDA_VISIBLE_DEVICES=7 + --tp 1 + sanitizer
     →「复现」= 类 A/C（shape/stride，与 shard 无关）
     →「不复现」= 类 B / 多 rank 次生（必须 TP8）
Phase 1（TP8，定版）
  3  S0 命令（§3.2）→ 定版报告；枚举模式跑一遍拿到全部 rank 的错
Phase 2（只有定位完成才做）
  4  按 §7 矩阵落到具体读点 → 若是 bulk 路径，转「对齐 view/步长」的独立设计（上报尚书省）
```

**一句话**：**先别上 serve。**「191ms 快速失败」先被 `-lineinfo` 的 sanitizer 报告变成
`<kernel 名>:<行>:<SASS 指令> @ <地址> (actual=A, expected=E), Device=N` 这一串正证据；
`sanitizer 报 0 但进程仍崩` 就说明真凶在 **bulk/TMA 盲区**——而 byte-fallback 恰恰对那一类**原理上无效**，
这才是「两轮修复没解决」的答案。

---

*工部 · 只读调查 + 本文件（唯一产出）；未执行任何 GPU 命令、未改动任何源码。*
*TP8 = 单进程 8 线程、launcher 的 `al16` 门、`ILV=0 ⇒ 不调 interleave kernel`、两条 bulk 路径的
16B 硬要求，均已对工作树 HEAD 现场核对；`-lineinfo` 改动与「对齐 view」的修法明确标注为「待批准」。*
