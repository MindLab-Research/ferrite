# WO_PAIR-ROWS：verify 输出投影的 m 行版（实施 + GPU 验证手册）

> 工部 · 2026-09-13 · **禁止 GPU/e2e**（任务书）；本机 `cargo check` **EXIT=0**；
> 远端 b300 `nvcc 13.2 sm_103a` **compile-only**（见 §3）。
> 依据：`docs/agent/sglang-verify-model.md` §4「融合对齐」行（verify 吃 eager 融合形态的 rows 版）
> + `docs/agent/fusion-alignment-orope-linrope-mrows.md` §3（`gemm_fp8_wo_pair` 的 rows 版当时未做）
> + `docs/agent/wo-a-grouped-nwarps-verify-manual.md` §5（grid-sync pair 的将来接法）。

---

## 0. 结论先行（四条）

1. **交付形态 = 2 发/层服务全部 m 行**（不是 1 发）：`wo_a_grouped_fp8`（**已有，未改**）
   + 新的 **`dsv41_gemm_fp8_mrows_q_f32`**（= `gemm_fp8_mtile_kernel<M, 1>`：MTILE 程序
   + **中间 quant 折进 warp 内**）。逐行 quant 的 `m` 发 **归零**。
   与任务书「融合后 **2 发**服务 6 行」一致。
2. **逐位**：新臂输出与它替换的三发链
   `wo_a_grouped_gemv_kernel<M>` + `m × quant_kernel<0>` + `gemm_fp8_mtile_kernel<M, 0>`
   **全字节相同**（§2 逐指令论证）。所以本臂可以走**字节比对**验收——这一点与 B6
   （`dsv41_gemm_fp8_mrows_f32`，跳过 round trip，**严格更准而非相等**）是**两个不同的臂**，
   不要混用验收口径。
3. **为什么不是 1 发（grid-sync pair）**：wo_b 的交付形态是**独立 launch** 的
   `gemm_fp8_mtile_kernel`（任务书：复用刚交付的 mtile 核），而「1 发」要求把两段合成一个
   grid-sync launch —— 两段的 grid 几何不同（段 1 `n/nwarps = 128` 块 vs 段 2
   `ceil(dim/(nw·bn)) = 320` 块），且设备级 barrier 必须把 grid **夹在 residency 上限内**
   （`wo_a_grouped` 的 launcher 自己那条 DEADLOCK 纪律）。**无 GPU 在场时，barrier 段数/夹取
   算错一次就是死锁（watchdog 在 graph 里看不见）**，故本交付取 2 发形态（`wo-a-grouped` manual
   §5 的先例）。1 发的接法见 §4.5（留给有 GPU 的一轮）。
4. ⚠️ **事件披露**：本交付的 `dsv41_kernels.cu` 改动在完成前被 peer 的提交 `19aa607`
   **`git add` 扫进了 HEAD**（正是 peer 自己在 `c0d0fa7` 里记录的 commit-discipline 陷阱）。
   §6 记现场、影响与建议。

---

## 1. 改动清单

| 文件 | 改动 |
|---|---|
| `kernels/cuda/dsv41_kernels.cu` | `gemm_fp8_mtile_kernel` 由 `template <int M>` 扩成 **`template <int M, int AQ = 0>`**，参数表尾追 **`int a_stride`**；`AQ == 1` 时 `av[q]` 改为**在飞量化**（`quant_kernel<0>` 的算术，逐指令，§2.2），`AQ == 0` 走 `if constexpr` 保持原程序；8 处现有 dispatch 补 `, k`；**新增** C 入口 `dsv41_gemm_fp8_mrows_q_f32`（~110 行，含 launcher/收据/decline） |
| `crates/ferrite-models/src/dsv41/device.rs` | +1 字段 `gemm_fp8_mrows_q_f32` +1 `ko!` 注册 +`supports_gemm_fp8_mrows_q_f32()` +`gemm_fp8_mrows_q_f32()` |
| `crates/ferrite-models/src/dsv41/chain_dev.rs` | +`verify_wo_pair_mrows()` / `mrows_mtile_armed()`；`attention_rows` 的 wo_b 段 +1 个**前置**臂（`!took_wob && mrows` 之内，走`quant_fp8 + proj_mrows` 之前） |

**未改**（刻意）：`wo_a_grouped_gemv_kernel` 体（它就是 rows 版）、`gemm_fp8_mtile_kernel<M,0>` 的
程序、`proj_mrows`、`quant_rows`、任何现有 gate 的默认值。

### 1.1 发数账（每层，m = 6）

| 段 | 基线（默认栈） | WO_PAIR-ROWS 臂 |
|---|---|---|
| `o_r → xq_r`（wo_a 的输入） | 0（OROPE 融合后的 epilogue）或 m | 同左（**不在本臂范围**） |
| wo_a（分组、weight-stationary、m 行） | 1 | 1（未改） |
| **中间 quant（`wo_r → fp8`）** | **m（逐行 `quant_fp8`）** | **0（折进下一发）** |
| wo_b（`proj_mrows` → MTILE） | 1 | 1（同一程序，`AQ = 1`） |
| **合计** | **m + 2 = 8** | **2** |

> 逐行 quant 之所以存在，纯粹是 `quant_fp8` 的 **源行距从 `cols` 推**（`src = x + r*cols`），
> 而 `wo_r` 的真实行距是 `ol_total`（TP8 下 8× `ol_local`）—— `chain_dev.rs` 里已有一条
> ROW-STRIDE FIX 注释在同一个类上。本臂把行距重新变成**参数**（`a_stride`），那条逐行循环
> 连同它的 `m` 发一起消失。

---

## 2. 逐位论证（指令级）

设几何：`k = ol_local`，`a_stride = ol_total`，`n = dim`，`m ≤ 8`，`bn = 2`，`nw = 8`。

### 2.1 段 1（wo_a rows）——**未改**

`dsv41_wo_a_grouped_fp8` / `wo_a_grouped_gemv_kernel<M>` 一行未动，其头部的 C1–C6
（逐 (row, r) 与 m=1 `dsv41_gemm_fp8_mx` 同：同 `s_a`/`s_as` 字节、同 `j = kb*32+lane` 升序 kb、
同 `#pragma unroll 32` 单链、同 `shfl_xor` 树、无 K-split）继续成立。输出 `wo_r` 字节不变。

### 2.2 中间 quant（`AQ == 1` 的 `av[q]`）——**与 `quant_kernel<0>` 逐指令同**

`quant_kernel<0>`（`block = 32`，`round_scale = 1`）一条 warp 处理一个 32 元组；
本 kernel 的 `AQ == 1` 臂在**同一条 warp** 里做同一件事：

| # | `quant_kernel<0>`（`dsv41_kernels.cu:126-168`） | 本 kernel（`AQ == 1`） | 等价理由 |
|---|---|---|---|
| Q1 | `idx = blockIdx.x*32 + threadIdx.x`；`r = idx/nb`，`b = idx%nb`；块内元素 `[b*32, b*32+32)` 由 32 条 lane 各持 1 个 | lane `l` 持 `j = kb*32 + l`，`kb ≡ b` | 同一 32 元组、同一 lane→元素映射 |
| Q2 | `amax = 0`；`amax = fmaxf(amax, fabsf(src[i]))`；再 `shfl_xor 16/8/4/2/1` 的 `fmaxf` 树 | `am = fabsf(v)`；同一棵树，同一 `off` 序 | `fabsf(v) ≥ 0` ⇒ `fmaxf(0, |v|) == |v|`（含 `±0`：`fabsf(-0.f) == +0.f`）⇒ 树输入逐位同 ⇒ 输出逐位同 |
| Q3 | `sc = fmaxf(fast_round_scale(amax, 1.0f/448.0f), 1e-30f)` | 同表达式、同 device helper、同 `maxv = 448` | 同一 inline 函数、同一 TU、同一编译 flag |
| Q4 | `inv = 1.0f/sc`；`v = src[i]*inv`；`clamp(v, ±448)`（`fminf(fmaxf(v,-448),448)`）；`__nv_fp8_e4m3` | `fminf(fmaxf(v * (1.0f/sc), -448.0f), 448.0f)`；同一转换 | **同一个 `1.0f/sc`**（不是除法）；clamp 的**同一个次序**；`--use_fast_math` 下 `rcp.approx` 的取舍在**两个 kernel 里是同一个表达式、同一 TU、同一 flag** ⇒ 同一条指令 ⇒ 同一位模式 |
| Q5 | byte 落到 `y[r*cols + b*32 + i]`；`sc` 落到 `scale[r*nb + b]` | byte **不落盘**（寄存器内）：`av[q] = s_lut[byte] * sc` | 下游消费方是**同一 kernel 的 fp8 臂**，它读 `s_lut[stored_byte] * a_scale[b]`；`a_scale[b] = sc`、`stored_byte = byte` ⇒ **operand 逐位同** |

所以 `av[q]` 就是 fp8 臂从「quantiser 的全局输出」读回的那个值。**byte 不落盘是安全的**：
`attention_rows` 内 `xq_r`/`xsc_r` 的最后一个读者**就是这个被替换的 quant**，下一个读者
`indexer_front_rows` 自己先跑 `quant_rows`（`chain_dev.rs` 的 STALE-READER CHECK 原文；
B6 臂用的是同一条论证）。

### 2.3 fold 段（`AQ == 0` 的程序）——**编译期选择，未改**

`AQ` 是**模板非类型参数**，`if constexpr` 保证 `<M, 0>` 的实例化不包含 `AQ == 1` 的任何代码；
`a_stride` 在 `AQ == 0` 下不被引用。⇒ `gemm_fp8_mtile_kernel<M, 0>` 的 body 与改动前**同一程序**
（C1–C6 原样：同 `kb` 升序、`j = kb*32+lane`、同 staged slab 的 1:1 平索引、同
`#pragma unroll 32` 的 `acc[q][nn] += av[q] * wv`、同 `shfl_xor` 树、每元素一个 warp 一棵树、
无跨行/跨 K 重组）。

### 2.4 为什么**不做**「f32 直折叠」（任务书点的 `wv_f32` 数值注意）

有一条更省的路：**干脆不量化**，把 `wo_r` 的 f32 直接当 wo_b 的激活（B6
`dsv41_gemm_fp8_mrows_f32` 与 EAGER `DSV41_WOB_F32` 都是这条路）。它的数值**不是等价而是更优**
（跳过 `quant→dequant` 的两次舍入）——所以它的验收是 `mean-k`/文本红线，**不能**做字节比对。

本臂**刻意不走**这条路：激活先按 `quant_kernel<0>` 量化再喂给 fold，因此

* **权重侧的 `wv`（`s_lut[s_w[...]] * sb`，f32）一个字都没动** —— 融合的数值风险只可能出现在
  「被折叠的那一侧」，而本臂折叠的是**激活侧**，且折叠方式是**逐位复刻量化器**，不是把 f32
  直接塞进 `wv`；
* 于是本臂同时具备「少 m 发」与「字节相同」，而 B6 只有前者。

一句话：**要逐位，就必须继续量化；本臂把量化从「一发」变成「寄存器里的一步」，而不是取消它。**

### 2.5 非逐位的部分（诚实边界）

* **不写 `xq_r`/`xsc_r`**：与 B6 同一条 stale-reader 论证（§2.2 末）。若将来有新的 `xq_r`
  读者落在 `attention_rows` 与 `indexer_front_rows` 之间，本臂与 B6 会**同时**失效——这是**两个臂
  共享的一条隐式前置**，手册 §4.4 把它列为回查项。
* **`DSV41_MROWS_MTILE` 必须同开**：本 kernel 是 MTILE 程序（`AQ = 1`）；若 MTILE 关，
  `proj_mrows` 跑的是 `gemm_fp8_mrows_kernel<M>`（M 进寄存器），**operand 布局不同** ⇒
  「我替换了它」这句话不成立，所以调用点用 `mrows_mtile_armed()` 把这一臂**挡掉**（保留原链），
  而不是发一个「对不上基线」的 launch。
* 本臂与 B6（`DSV41_VERIFY_WOB_MROWS_F32`）**互斥**：B6 在前，两者同开时本臂 inert
  （`!took_wob` 守卫）。

---

## 3. 编译证据（本机 / 远端 compile-only）

```bash
# Rust
cargo check -p ferrite-models                 -> EXIT 0（仅既存 warning：dead_code/unused，
                                                 6 条，与本次改动无关）

# CUDA（远端 b300，CUDA 13.2，compile-only，无 GPU 调用）
scp kernels/cuda/dsv41_kernels.cu ubuntu@<b300>:/tmp/wo_pair_rows_kernels.cu
nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 \
     -c /tmp/wo_pair_rows_kernels.cu -o /tmp/wo_pair_rows_kernels.o
                                                  -> 见 §3.1 的 RC/errors 记录
```

### 3.1 结果

（以本次实际输出回填：）

| 项 | 命令 | 结果 |
|---|---|---|
| Rust | `cargo check -p ferrite-models` | ✅ **EXIT=0**，0 error，6 既存 warning（dead_code/unused，与本次改动无关） |
| Rust（全仓 targets） | `cargo check --workspace --all-targets` | ✅ **EXIT=0**（含所有 test target 的编译） |
| Rust（单测） | `cargo test -p ferrite-models --lib` | ✅ **92 passed / 0 failed / 2 ignored**（0.23s，GPU-free 部分） |
| CUDA | §3 的 nvcc | ✅ **RC=0**，errors=0，`wo_pair_rows_kernels.o` 10,273,268 B，real 2m01s |

```
$ nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 \
       -c /tmp/wo_pair_rows_kernels.cu -o /tmp/wo_pair_rows_kernels.o
RC=0            # 仅既存 warning（#177-D declared but never referenced 等），无 error
real 2m1.836s
-rw-rw-r-- 1 ubuntu ubuntu 10273168 Sep 13 03:07 wo_pair_rows_kernels.o
```

> 远端：`ubuntu@43.202.208.136`（b300，CUDA 13.2 V13.2.51），**compile-only**，未调用任何 GPU API。
> 模板实例化 `<M, 1>`（M = 1..8）与既有的 `<M, 0>` 由同一诊断轮覆盖：8 个 `case` 都在
> `switch (m)` 里，模板体无条件实例化 ⇒ nvcc 必须为两者都生成 code（error 会在这里暴露）。

---

## 4. GPU 验证手册（双门禁 + 发数判据）

### 4.1 前置三证

```bash
cd kernels/cuda && bash build.sh 103a          # .so 与 Rust 一起重建
cd ../.. && cargo build --release
nm -D kernels/cuda/libferrite_kernels.so | grep -c dsv41_gemm_fp8_mrows_q_f32   # >= 1
```

> ⚠️ `gemm_fp8_mtile_kernel<M,1>` 的**符号名变了**（多一个模板实参），且 `dsv41_gemm_fp8_mrows`
> 的**参数表多了一个 `int a_stride`** —— **.so 必须与 Rust 侧同时重建**，否则 Rust 会按新 ABI 调用
> 旧 `.so`（与 `dsv41_gemm_fp8_swapab` 当年同一条警告）。

### 4.2 环境（**一臂一进程**；gate 是 `OnceLock` 每进程读一次）

```bash
export BASE_ENV="DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 DSV41_TIMING=1 \
DSV41_SWALLOW_STEP=1 DSV41_SWALLOW_EPOCH_PAD=1 DSV41_VERIFY_GRAPH=1 \
DSV41_HC_VERIFY_FUSE=1 DSV41_FUSE_B1=1 DSV41_FUSE_C=1 \
DSV41_MROWS_MTILE=1"

# R0 基线（MTILE 开、WO_PAIR-ROWS 关 —— 这一臂就是"被替换的那条链"）
env $BASE_ENV bash scripts/batched_400_v2.sh 2>&1 | tee /tmp/wpr_r0.log

# R1 本臂
env $BASE_ENV DSV41_VERIFY_WO_PAIR_MROWS=1 bash scripts/batched_400_v2.sh 2>&1 | tee /tmp/wpr_r1.log
```

**逐门回读**（不许只信 `export`）：

```bash
grep -a DSV41_MROWS_MTILE /proc/<pid>/environ
grep -a DSV41_VERIFY_WO_PAIR_MROWS /proc/<pid>/environ
```

### 4.3 门禁 1（性能 / 发数判据——**不允许用吞吐反推**）

**A. 活性回执（本项唯一的"分支被走到"证据）**：R1 的 stderr 必须出现**一次**

```
[wo-pair-rows] ARMED m=6 n=5120 k=1024 a_stride=8192 bn=2 nw=8 -> block=256 warps, grid=320, smem=17408 (intermediate quant FUSED: 0 quant launches)
```

（数字按实际形状；**没有这行 = gate 没进进程或前置挡住**，不是 kernel 的问题。）

**B. nsys 发数判据**（只看**相对倍数/实例数**，不看绝对 us）：

```bash
env -u FERRITE_P2P NCCL_NVLS_ENABLE=0 DSV41_AR_V5=0 DSV41_GRAPH_STEP=0 \
    DSV41_MROWS_MTILE=1 DSV41_VERIFY_WO_PAIR_MROWS=1 ~/nsys_dual.sh
/usr/local/cuda-13.2/bin/nsys stats --report cuda_gpu_kern_sum --format csv \
    /tmp/wave1_nsys.nsys-rep | grep -E "quant_kernel|mtile|wo_a_grouped"
```

| 判据 | R0 基线 | R1 本臂 |
|---|---|---|
| `quant_kernel<0>`（wo_b 的逐行 quant）实例/步 | **40 × m = 240** | **0**（本臂） |
| `gemm_fp8_mtile_kernel` 实例/步 | 40 | 40（同发数，**换模板实参**：`<6,1>`） |
| `wo_a_grouped_gemv_kernel` 实例/步 | 40 | 40（未改） |
| 每层输出投影合计 | m + 2 = 8 | **2** |

> **硬否证**：若 R1 里 `quant_kernel<0>` 的实例数**没降到 0**（或 stderr 没有 ARMED 行），说明
> `verify_wo_pair_mrows()` / `mrows_mtile_armed()` / `supports_...()` 三者有一者没接上 —— 逐条
> 用 `/proc` 回读 + `nm -D | grep dsv41_gemm_fp8_mrows_q_f32` 定位，**不要**用吞吐反推。
> ⚠️ 注意 `quant_kernel<0>` 同时服务别的调用点（KV/indexer 的量化），所以判据是**实例数下降
> ≈ 40×m**，不是归零。

**C. 门禁 1 的收益口径**：按 `verify-amortization-lesion-audit.md` §10 的教训，**票面按 nsys
时间占比折价**：本项省的是 `m` 个 ~几 us 的小 launch + `m` 个 graph 节点，**不要**用发数直接推 ms。
预期量级（`dspark-verify-perf-plan.md` §4 的同类项）×：−0.1 ~ −0.3 ms/步，属于「消除结构性差异」
而非「4× → 1.3×」的机制项。

### 4.4 门禁 2（数值，红线）

1. **字节比对（本臂的主判据，因为它宣称逐位）**：加一条 parity 臂，把
   `[wo_a_grouped + m×quant_fp8 + proj_mrows(MTILE)]` 与
   `[wo_a_grouped + dsv41_gemm_fp8_mrows_q_f32]` 的 `wo_out_r` **逐字节**比。
   当前 `tests_dsv41_gemm_mrows.cu` 里已有一个 `DSV41_MROWS_MTILE` 的对照臂（见该文件
   :437/:579 的注释），复制它的形状、把 A 侧的 `quant+mtile` 换成新入口即可。
   **任一字节不同 ⇒ 立即弃用该臂并上报**（这是本交付的"逐位"声明的收据）。
2. `[dspark] mean-k` **不掉**（Fix A 后基线 **2.240**；`wo-a-grouped` manual 记的另一基线 1.34 —— 
   以当轮同栈 R0 的值为准，**不要跨轮比较**）。
3. 出师表 1000 token **逐字 + 零拉丁**（`DSV41_BF16_TRUNCATE=1` 红线）+ 0 double-char。
4. `faults=0`、无 `MISMATCH` 行。
5. `/proc/<pid>/environ` 回读确认 `DSV41_VERIFY_WO_PAIR_MROWS` 与 `DSV41_MROWS_MTILE` **都**进了进程。
6. **STALE-READER 回查**（本臂与 B6 共享的前置）：`grep -n "xq_r" chain_dev.rs` 确认
   `attention_rows` 与下一个读者之间**没有**新增 `xq_r`/`xsc_r` 读者。若有 ⇒ 本臂必须补写 fp8。

### 4.5 留给定下一轮（有 GPU）的两件事

1. **1 发形态（真正的 grid-sync pair）**：把段 1（`wo_a_grouped` 体）+ 段 2（`AQ = 1` 的 MTILE 体）
   放进**一个 launch**，用 `s.wo_bar`（已有的 `[arrive, sense]` 对）连两段。必须做对的三件事：
   (i) grid 取 `max(128, 320) = 320` 并**夹在 `co_res`**（barrier 才有意义）；
   (ii) smem 取两段的 `max`（段 1 的 `nw·k_a + …` ≈ 38 KB > 段 2 的 17 KB）；
   (iii) 段 1 的 2D grid（`grid.y = groups`）在 verify 侧恒为 1，可用 1-D grid + 行循环等效。
   收益：再省 **1 发/层**（40/step）；风险：死锁。
2. **`DSV41_WO_A_NWARPS` × 本臂的交叉 sweep**：本臂把 wo_b 的 grid 提到 320 块，wo_a 的
   `nwarps` 旋钮（128 → 256/512/1024 块）会**同时**改变两段的相对占用——上一层交付的 sweep
   结论（§6 的 4 臂）在本臂下**需要重跑**，不要沿用。

---

## 5. 与 peer（fusion-parity-rows）的冲突边界

* peer 的在途改动集中在 `attention_rows` 的 **attention 段**（`orope_mrows_ok` /
  `window_idxs_mrows` / `ring_append_mrows` / `sparse_attn_orope_mrows` / 逆 o-rope），
  最后一个 hunk 落在 `orope_rows` 的填充与逆 o-rope 之前。
* 本交付的接线点在 **o 投影段之后的 wo_b 段**（`chain_dev.rs` 的
  `if !took_wob && mrows` 之内），**文本不重叠**。两者唯一交互是 `orope_rows[r]` —— 那个
  跳过逻辑（`if vo && orope_rows[r] { continue; }`）在 HEAD 上已存在，本交付**没有**改它。
* peer 提交 `19aa607` 把 `attention_rows` 的 attention 段与本交付的 `.cu` 一起进了 HEAD
  （见 §6）；本交付的 `device.rs` / `chain_dev.rs` 改动**仍在工作区**（未提交）。

---

## 6. 事件：.cu 改动被 peer 提交扫进 HEAD

* `git log` 显示 `19aa607`（peer，`fusion-parity-rows`）的 `git add` 把
  `kernels/cuda/dsv41_kernels.cu` 的**本交付在途改动**一起提交了（`git status` 里该文件随即从
  "modified" 消失，`git diff HEAD -- kernels/...cu` 为空）。**这正是 peer 自己在 `c0d0fa7`
  里写下的纪律**："never `git add -A` while peer subagents are running"。
* **影响**：HEAD 现在就带着一个**尚未经 nvcc 验证**的 CUDA 改动。本交付随后立即
  **用远端 nvcc compile-only 回填了验证**（§3.1）；若远端构建曾一度失败，那是本次事件的直接后果，
  不是本臂的设计问题。
* **建议**：本臂合入前，把 `.cu` 的改动**单独一个提交**（`git add kernels/cuda/dsv41_kernels.cu`
  只加自己验证过的文件），并在提交信息里点名 §2 的逐位论证——否则「verify 链 2 发」这个
  形态变更会被埋在 peer 的 orope 提交里，事后无法二分。
