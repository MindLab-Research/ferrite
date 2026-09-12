# AR L4/L5 第一步的实施：A2b（超时语义）+ A0（探针）

> 工部 · 2026-09-12 · 实施 + `cargo check`（本机无 GPU / 无 nvcc，**未执行任何 GPU 命令**）。
> 设计：`docs/agent/ar-l4l5-optimization-design.md` §4-A2b / §4-A0 / §5（执行顺序第 1、2 项）。
> 唯一改动文件：`kernels/cuda/ferrite_kernels.cu`（**Rust 侧零改动**，见 §3）。

---

## 1. A2b — 超时不再被当作"成功"（安全项，无条件生效）

### 改前的代码事实（设计 §4-A2b）

`ar5_wait_round` 的三处等待，超时（`spins > 5,000,000` ≈ 0.5s）时都 `printf("[ar5-hang]…")` 然后 **`break`**，
随后**照常推进**：

| 臂 | 超时后的行为 |
|---|---|
| OFF（每块都轮询） | `break` → 调用者的 `__syncthreads()` → **直接 reduce**，读对端从未 publish 的 staging |
| A4 block 0 | `break` → `__syncthreads()` → 仍写 `epoch[1] = e+1`（**把超时当成功广播**）→ 其它块照常 reduce |
| A4 其它块 | `break` → 直接 reduce（读到的是 block 0 没发过的广播字） |

⇒ 不是"卡住"，是**静默错数**，与 SWALLOW 臂"6 token 后 EOS（epoch 54）"的观测链一致。

### 改后

新增 `ar5_timeout(bcast, site, my_rank, peer, need, cur, trap)`：**超时的等待者绝不 publish、
绝不推进 epoch、绝不进入 reduce**，先报错再二选一：

| 模式 | 触发 | 行为 | 用途 |
|---|---|---|---|
| **PARK（默认）** | `DSV41_AR_TIMEOUT_TRAP` 未设/0 | `for (;;) __nanosleep(1000000u)` | 内核永不完成 ⇒ stream 不推进 ⇒ 不可能吐出基于坏轮次的 token；由 `serve.rs STEP_TIMEOUT` 报"a rank did not answer" |
| **TRAP** | `DSV41_AR_TIMEOUT_TRAP=1` | `asm volatile("trap;")` | 下一次 CUDA API 立刻返回硬错误，比 1800s 的 pool watchdog 快，且不会被误读成"慢" |

诊断行（保留旧 tag，grep 不变；新增 rank/site/spins/mode，单进程最多 8 行避免 160 行洪泛）：

```
[ar5-hang] rank=3 site=1 peer=5 need=1329 cur=1328 spins>5000000 TIMEOUT -> PARK
[ar5-hang-bcast] rank=3 site=0 need=1329 cur=1328 spins>5000000 TIMEOUT -> PARK
```

**注意（对 SWALLOW 实验的语义变化）**：改前 epoch 裂开时进程会吐错 token；改后会 **wedge（默认）
或 crash（TRAP=1）**。这是设计要的方向（"响亮失败"），但做 A/B 时请显式选一个：
`DSV41_AR_TIMEOUT_TRAP=1` 通常更省一个测量周期。

**常规路径开销 = 0**：`spins` 分支本来就在，只换了分支体；热路径（`__nanosleep` + volatile load +
比较 + `++spins`）指令一字未动。

---

## 2. A0 — 探针：site 标签 + 每 rank + stamp/spin/epi 三段

### 2.1 输出格式

`DSV41_AR_PROBE=1`，**不要在 nsys 下跑**（追踪会把自旋放大），每 512 轮打一行（设备 `printf` → stdout）：

```
[ar-probe] rank=0 site=1 n=512 avg_spin=31000 cyc max=88000 cyc avg_stamp=1200 cyc avg_epi=4200 cyc max_epi=9000 cyc
```

单位 = SM 周期（B300 ~1.8GHz ⇒ 1μs ≈ 1800 周期）。**对照值：账本口径 17.3μs/轮 ≈ 31k 周期。**

### 2.2 三个字段回答三个问题（判据见设计 §4-A0）

| 字段 | 问题 | 判据 |
|---|---|---|
| `avg_spin` | nsys 的 53.4μs 是真的吗？ | ≫31k ⇒ 追踪放大为主 ⇒ 转投 MoE/投影；≈31k ⇒ 走 A2；≪31k ⇒ AR 已无肉 |
| `rank` | 是"最后一个到达者"造成的吗？ | 某 rank 的 `avg_spin` 系统性为 0 ⇒ A2c（负重平衡）；所有 rank 都大 ⇒ 同步开销本身（A2d/A1） |
| `stamp`/`spin`/`epi` | 一轮里多少是**工作** | `epi` ≈ 工作地板（设计 §2 推算 4–6μs ≈ 7–11k 周期）；`stamp` = entry→poll 开始（8 次远程 atomic + barrier + epoch 写） |

### 2.3 site 的定义与**已知局限**

site 是**入口身份**（编译期常量，由各 `extern "C"` launcher 传给 kernel）：

| site | 入口 | 对应 Rust 调用路径 |
|---|---|---|
| 0 `MOE` | `ferrite_p2p_ar_v5_hcpost_add` / `_add` | `moe_reduce` 的 MoE all-reduce（**图外**，40 轮/步） |
| 1 `ATTN` | `ferrite_p2p_ar_v5_hcpost` / `ferrite_p2p_ar_pubred_v5` | `layer()` 的 attention all-reduce（**图内**，40 轮/步） |
| 2 `VERIFY` | `ferrite_p2p_ar_v5_hcpost_rows` | verify（m 行，lazy 下每行一轮） |
| 3 `OTHER` | `ferrite_p2p_ar_v5` | 非折叠的通用路径 |

**为什么不用"每次调用多传一个 `site` 参数"**：那要改 6 个 `extern "C"` 签名；一旦加载到旧 `.so`，
Rust 会把 `site` 当成 `cudaStream_t` 传进去（静默灾难）。入口身份是**零 ABI 风险**的等价信息。

**局限（必须知道）**：`_hcpost` 这一个入口同时服务 attention 折叠与 MoE 折叠（后者只在 .so 缺
`_hcpost_add` 时发生）。默认干净栈走 `_hcpost_add`，所以不歧义；若某次运行看到 `site=1` 的例数
≈ 2×层数，说明落在该 fallback 上，需先让 biased 入口可用再分开读。

---

## 3. Rust 侧与 gate 语义

- **Rust 零改动**：`probe` / `site` / `trap` 都由 .so 内部决定（`site` 是 launcher 编译期常量，
  `trap` 读 `DSV41_AR_TIMEOUT_TRAP`），Rust 的 6 个调用点与 device.rs 的函数签名**不变**。
- **三个 gate 都是"进程读一次"**（`ferrite_ar_single_poll` / `ferrite_ar_probe` /
  `ferrite_ar_timeout_trap`）：它们选的是 **kernel 臂**，必须在 capture 与同一 graph node 的每次
  replay 之间保持一致（capture 里逐次 `getenv` 是会被烘进图的隐患）。
- **gate OFF 零开销**：`probe=0` 时所有 `clock64()` / 计数器 / `if (probe) __syncthreads()` 全部跳过。
  代价只有：2 个新增 kernel 参数（`site`、`trap`，各 1 寄存器）+ 16B 静态 shared。热路径指令不变。
- **`__shared__ unsigned long long s_probe[2]`**：`[0]` = spin（block 0 的 8 个轮询者 atomicMax），
  `[1]` = stamp 段（block 0 / thread 0 写）。`epi` 是线程 0 的局部量，不占 shared。
  `t_entry`（kernel 入口 `clock64()`）与 T0（poll 开始）**是同一个线程**（block 0 / thread 0），
  所以两个时钟差没有跨 SM 偏斜问题。

---

## 4. 落地要求（**别忘**）

1. `.cu` 改了 ⇒ 必须按**双产物纪律**重建：`cd kernels/cuda && bash build.sh 103a` 再 `cargo build --release`
   （`build.rs` 的同源门禁 + 运行期 `ferrite_kernel_build_id` 会拒绝错配的组合）。
2. A0 的运行：`DSV41_AR_PROBE=1` + 生产 gate 集，直接跑 `./target/release/ferrite-serve`，读 stdout。
   **默认与 nsys 互斥**：nsys 逐节点追踪会把 publish 自旋放大（实测 240s/69 步），
   在 nsys 下读 `avg_spin` 得到的是伪影。
3. `DSV41_AR_TIMEOUT_TRAP=1` 只影响"超时之后怎么死"，不影响任何正常轮次的路径与数值。

---

## 5. 本机验证边界（诚实声明）

- 本机无 nvcc、无 GPU：**`.cu` 只做了人工审校**（6 个 launch 点的实参顺序逐一对照 kernel 形参、
  括号配平、无 `s_spin` 残留、`site`/`trap` 传参链完整），**编译在远程 B300 节点进行**。
- `cargo check --workspace` EXIT=0（Rust 侧未改动，故恒过）。
- **未做**：A1a（store 折进 producer）、A1b（PDL）、A2c/A2d、B（臂选择）——按设计 §5 属第 3 步及以后。
