# `wo_a_grouped_gemv_kernel` 占用率旋钮（`DSV41_WO_A_NWARPS`）实施 + GPU 验证手册

> 工部 · 2026-09-13 · **未执行 GPU/e2e**（任务禁止）。本机 `cargo check` **EXIT=0**；
> 远端 nvcc 13.2 `sm_103a` **compile-only**（`dsv41_kernels.cu`）**EXIT=0 / errors=0**。
> 依据：proj-head-lesion 判决（verify 侧 2.49ms/步，v6 nsys 4.5%，latency/占用率病）。
> 改动**仅 1 文件**：`kernels/cuda/dsv41_kernels.cu`（launcher `nwarps` + 两处注释）。

---

## 0. 结论先行（三条）

1. **已实施**：`dsv41_wo_a_grouped_fp8` launcher 的 `nwarps` 由硬编码 8 改为 **env 可调**
   （`DSV41_WO_A_NWARPS=N`，默认 8 = 现状；N 夹到 `[1,8]`，只能**减小**）。
   kernel 体**一行未动** ⇒ **逐位等价**（§2 论证），改动面 = 「grid/block 形状」这一个自由度。
2. **`WO_PAIR` 的 verify 接线：不做（保守）。** 现有 `gemm_fp8_wo_pair_kernel` 是**严格 m=1**
   （phase 1 只 stage **一条** block-wide 激活行，`dsv41_kernels.cu:7950` 自己把 `nlg!=1` 之外的
   形状列为 NOT COVERED），且 phase 2 是 **f32 激活**形态——verify 现状是 **fp8 量化**。
   接 verify（rows=m）需要**新写一个 m-rows pair kernel** + 改数值形态，**在无 GPU 验证的前提下
   不可交付**。任务本项显式允许"先不做"。理由与将来接法见 §5。
3. **预期收益有上限，且 sweep 非单调**——`nwarps` **不改变在飞 warp 总数**（恒 = n，§3）。
   它改的是 (i) 网格是否盖满 148 SM、(ii) blocks/SM、(iii) 激活行重读冗余。三项反向，
   **sweep 才是答案**；本手册给出判据与止损（§4、§6）。

---

## 1. 改动（唯一实施项 + 两处注释）

### 1.1 launcher：`nwarps` env-gate

`kernels/cuda/dsv41_kernels.cu:7331`（`dsv41_wo_a_grouped_fp8` 内）：

```c
static const int wo_a_nwarps_env = [] {          // 一次读；本 launcher ~40 发/步，getenv 不能进热路径
    const char* v = getenv("DSV41_WO_A_NWARPS");
    if (v == nullptr || v[0] == '\0') return 8;
    const int x = atoi(v);
    return (x >= 1 && x <= 8) ? x : 8;           // 越界/非法 -> 8（现状）
}();
int nwarps = (n >= 8) ? 8 : n;
if (n % nwarps) return 2;                        // ← 原守卫，一字未改
if (wo_a_nwarps_env < nwarps) {                  // 只可能"减小"
    int nw = wo_a_nwarps_env;
    while (nw > 1 && (n % nw)) nw >>= 1;         // N 不整除 n -> 降到最大 2 的幂因子
    nwarps = nw;
}
```

其后的 `smem` / `grid` / `block` / 八个特化**原样使用**这个 `nwarps`（`:7377-7390`）。

**默认路径的 accept/decline 集完全不变**：`env` 未设 → `wo_a_nwarps_env == 8` →
`if (8 < nwarps)` 恒假 → 与改前**逐指令相同**。

### 1.2 注释对齐（无功能）

- 内核头 `:7173` 的 `__launch_bounds__(256)` 论证：`nwarps` 现由 env 只可能减小，`block ≤ 256` 仍成立。
- launcher `:7332` 增「NWARPS IS THE OCCUPANCY KNOB」段，记结论与形状算术。

---

## 2. 逐位等价论证（为什么改 `nwarps` 不改任何一个 bit）

kernel 体对 `nwarps` 的**全部依赖**只有三处（读码复核）：

| 依赖点 | 位置 | 用法 |
|---|---|---|
| `nwarps = (blockDim.x + 31) >> 5` | `:7195` | 仅用于 smem 槽位偏移（`s_lut/s_as/s_a` 基址）与行映射 |
| `row = blockIdx.x * nwarps + warp` | `:7198` | 本 warp 认领的输出行 |
| smem 布局 `s_w` 的 `nwarps*k` 跨度 | `:7208` | 每 warp 独占 `[warp*k, (warp+1)*k)` |

输出 (g, row, r) 的**计算**与之无关：

- **同字节**：本 warp 的权重行恒为 `wg + row*k`（`row_s` 恒为 `s_w + warp*k`），激活行恒为
  `a + r*a_stride + g*k`；staging 是**纯拷贝**（权重 cp.async16 / 激活标量字节拷，均无算术）。
- **同走序**：`#pragma unroll 32` 的 `kb = 0..nb_k-1` 升序、`j = kb*32 + lane`（C1）。
- **同表达式**：`acc += (s_lut[s_a[j]] * s_as[j>>5]) * (s_lut[row_s[j]] * sb)`（C2/C6，单条串行链，
  **无 K-split、无 split accumulator**）。
- **同归约**：`shfl_xor` 树只在 **32 lane 内**（`off = 16..1`），与 blockDim **无关**（C3）。
- **同写出**：`og[r*out_stride + row]`，`og`/`out_stride` 不含 `nwarps`。

**覆盖性**：`grid.x = n / nwarps` 且 launcher 保证 `n % nwarps == 0` ⇒ `[0, n)` 每行**恰好**被一个
warp 认领一次。**不同的只是**：这 n 行由「128 个 8-warp block」还是「1024 个 1-warp block」承担 ——
而一个 row 的点积**不依赖谁拥有它**（该 kernel 自身的注释即以此为据）。

**结论**：对任意 `nwarps ∈ {1,2,4,8}`（含 env 降幂后的值），输出张量**逐位相同**。
`__syncthreads()` 的作用域随 block 变小而变小，但该 barrier 只做"staging 发布"，**无跨行数据共享**，
故作用域变化不可观测。这与 `__launch_bounds__` 的 pin（仅寄存器预算）同理，不动数值。

---

## 3. 占用率算术（为什么 `nwarps` 是对的数，且为什么它不是免费的）

**形状**（verify 侧，从判决 + 符号名反推）：`n = o_lora_rank = 1024`，`k = hpg*hd = 4096`，
`a_stride = nh*hd`，`groups(=grid.y) = 1`（`nlg = groups/world = 1`），权重 `n*k = 4 MiB`。
`M` = template 实参（判决给 `<5>` ⇒ M=5）。

`smem(nwarps) = nwarps*4096 + 1024(256 f32 LUT) + (4096/32)*4(激活 scale) + 4096(激活行)`
            `= 4096*nwarps + 5632` 字节。

| `DSV41_WO_A_NWARPS` | `block` | `smem` B | blocks/SM (≤48KB) | `grid.x = 1024/nw` | **grid vs 148 SM** | 在飞 warp 上限 |
|---|---|---|---|---|---|---|
| **8（base）** | 256 | 38400 | **1** | **128** | **128 < 148 ⇒ 20 SM 空转** | 128×8 = 1024 |
| 4 | 128 | 22016 | 2 | 256 | 256 > 148 ✓ | min(256, 296)=256×4 = 1024 |
| 2 | 64 | 13824 | 3 | 512 | 512 > 148 ✓ | min(512, 444) ⇒ 444×2 = 888 + 68 排队 |
| 1 | 32 | 9728 | 5 | 1024 | 1024 > 148 ✓ | min(1024, 740) ⇒ 740 + 284 排队 |

**关键不变式（与 mpar 判决的 (a) 同源）**：**在飞 warp 总数 = `grid.x * nwarps = n = 1024`，与 `nwarps` 无关。**
所以 `nwarps` **不是**"增加并行度"的旋钮，它只重排这 1024 个 warp 的落点。真正的可回收量是：

- **base 的 20 个空 SM**（128/148 = 86.5% SM 覆盖）——这是 base 唯一确实吃亏的地方；
- **每 r 的 2 次 `__syncthreads()`**（M×2 = 10 次/block）：base 要在 **8 个 warp** 间取最大偏斜，
  `nwarps=1` 时 barrier 退化为单 warp 同步（≈免费）。这是**结构性 latency** 的直接来源。

---

## 4. 反向效应：sweep **非单调**的两个理由（诚实风险）

**① 激活行 staging 的 per-thread 串行链变长。** `s_a` 的 staging 是
`for (i = threadIdx.x; i < k; i += blockDim.x) s_a[i] = ar[i];`（标量字节拷，**未 cpasync 化**），
每线程迭代数 = `k / blockDim.x`：

| nwarps | blockDim.x | 每线程 `s_a` 装载次数 |
|---|---|---|
| 8 | 256 | 16 |
| 4 | 128 | 32 |
| 2 | 64 | 64 |
| **1** | **32** | **128** |

这条链**完全暴露在 `__syncthreads()` 之前**（staging → barrier → fold），且每 r 重来一次。
`nwarps=1` 时它比 base 长 **8×** —— 这**可能吃掉**上面两项收益，甚至净亏。

**② 激活行重读冗余随 `grid.x` 线性放大。** 每 block 每 r 重读 `k + k/8 = 4608 B` 激活
（本 block 内所有 warp 共享一份，但 **不同 block 之间不复用**）：

| nwarps | 激活字节/发（`grid.x × M × 4608`，M=5） | 权重字节/发 | 合计 |
|---|---|---|---|
| 8 | 128×5×4608 = **2.95 MB** | 4.19 MB | 7.1 MB |
| 4 | 256×5×4608 = **5.90 MB** | 4.19 MB | 10.1 MB |
| 2 | **11.8 MB** | 4.19 MB | 16.0 MB |
| 1 | **23.6 MB** | 4.19 MB | 27.8 MB |

激活工作集只有 ~23 KB（**L2 常驻**），所以这多是 L2 带宽而非 DRAM —— 但 `nwarps=1` 时
**激活流量反超权重流量 5.6×**，一旦 L2 侧成为瓶颈就是纯亏。叠加 §3 的**波次量化**
（nwarps=2 有 68 个 block 排到第二波，nwarps=1 有 284 个），**预期最优点在 4 或 2，1 可能回退。**

> **给 sweep 的止损**：若 8→4→2→1 的 kernel 时间**单调上升**（4 就是最优点）或**全段中性**，
> 则说明"2.49ms 的病"不在 `nwarps` 这一个自由度上 —— **立即停止变体矩阵**，
> 转 §4.1 的两个同族杠杆（staging 形态 / barrier 结构），不要再扫 `nwarps`。

### 4.1 同族杠杆（**不在本批次，别顺手做**）

1. **`s_a` 的标量 staging → `dsv41_cp_async16`**（权重侧 R1 已做，激活侧仍是字节拷）：
   纯拷贝 ⇒ 逐位等价，且**直接压掉 §4① 的串行链**（16 B/lane/issue，与权重侧同规则）。
   这是"隔壁修过这边漏了"的最强候选。
2. **每 r 的 2 次 `__syncthreads()`**：`s_a/s_as` 是**同一份激活行**被 nwarps 个 warp 共享；
   若要动 barrier，需先论证"行内多缓冲/双 r 流水"，**触及结构 ⇒ 需要独立 parity 门禁**，不在本次范围。

---

## 5. `WO_PAIR` 的 verify 接线：**不做（保守）**——理由与将来接法

**现有内核的形状是硬约束，不是调参项**（读码）：

| 约束 | 位置 | 对 verify 的含义 |
|---|---|---|
| phase 1 **只 stage 一条** block-wide 激活行（`a` 是 `[ka]` 一行） | `:7986-8025` | verify 需要 **M=5 行** ⇒ **不可用** |
| `NOT COVERED: nlg != 1`（phase 1 的单行假设） | `:7949-7951` | verify 的 `nlg` 恰为 1，但**行数**仍卡死 |
| phase 2 = **f32 激活**形态（`s_af` 预解码） | `:7919-7921` | verify 现状是 **fp8 量化** ⇒ **数值口径不同** |
| launcher 只收 `na/ka/nb/kb` 的 **m=1** 行形状（无 `rows`、无 `a_stride`、无 `out_stride`） | `:8145-8164` | verify 的 `a_stride/out_stride` **超出 ABI** |
| 设备级 barrier 需 `rows` 行全在各 block 的 `mid` 里 | `:7931-7937` | 多行版 smem 会**炸 48KB**（M 条激活） |

⇒ 接 verify = **新写一个 m-rows pair kernel**（多行 staging + 两次 phase 的 `a_stride/out_stride` ABI
+ smem 重算 + 死锁上限重算），**并且**改数值形态（phase 2 从 fp8 变 f32，或保留 fp8 但重写 pair 的
phase 2 —— 后者等于把"pair 省的那次 launch"又赚回来一点点）。

**并且它与本项收益不叠加**：pair 省的是 **1 次 launch + 1 个 graph node**（verify 侧 wo_b 是
`proj_mrows`，**不是** m=1 gemv），而本项治的是 **occupancy**。两者是**不同轴**，
**在拿到 §6 的 nsys 回执（wo_a kernel 时间是否真的降）之前，没有理由投入 pair 新核。**

**若将来要做**（留给下一轮，需 GPU 在场）：
1. 先跑完 §6 的 8/4/2/1 sweep，拿到 `wo_a_grouped_gemv_kernel` 的**per-instance us**与
   `grid.x` 回执；若 kernel 时间已降到 < ~20us/发（≈ 4.5% → <1.5%），**pair 的 ROI 自然为负，不做**。
2. 若 kernel 时间仍在且 launch 间隙是瓶颈，则**新写 `wo_a_grouped_gemv_pair_mrows_kernel`**：
   phase 1 = 本 kernel 的 m-rows body（**直接复用**，逐位等价），phase 2 = verify 的 `proj_mrows` body，
   用 `s.wo_bar`（已有的 `[arrive,sense]` 对）连接；**必须**先过 `*_parity.rs` 逐位门禁 +
   EAGER 对照（**标注** phase 2 若走 f32 则**非逐位**，按 §6 门禁 2 的 `mean-k` 红线判）。

---

## 6. GPU 验证手册（双门禁 + nsys 判据）

### 6.1 前置（三证）

```bash
cd kernels/cuda && bash build.sh 103a                 # .so 与 Rust 一起重建
cd ../.. && cargo build --release
nm -D kernels/cuda/libferrite_kernels.so | grep -c dsv41_wo_a_grouped_fp8   # >= 1
```

> ⚠️ **env 必须进到 serve 进程**：本项是**函数内 `static const`（只读一次）**，
> 且**不改 kernel 名、不改发数** ⇒ `/proc/<pid>/environ` 只能证明"变量进了进程"，
> **不能证明分支被走到**。活性回执请看 **nsys 的 `Grid X` 列**（见 6.4）——这是本项唯一的活性证据。

### 6.2 A/B sweep（**一臂一进程**，计数 200 tok，读 `[dspark] steps=50`）

```bash
export BASE_ENV="DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 DSV41_TIMING=1 \
DSV41_SWALLOW_STEP=1 DSV41_SWALLOW_EPOCH_PAD=1 DSV41_VERIFY_GRAPH=1 \
DSV41_HC_VERIFY_FUSE=1 DSV41_FUSE_B1=1 DSV41_FUSE_C=1"

for NW in 8 4 2 1; do
  env $BASE_ENV DSV41_WO_A_NWARPS=$NW bash scripts/batched_400_v2.sh 2>&1 | tee /tmp/woa_nw$NW.log
done
```

- `NW=8` 即 base（与**不设**该 env 等值，可留一臂 `unset` 做交叉核对）。

**门禁 1（性能）**：`[dspark]` 分解的 **`verify=` 中位数**，各臂相对 `NW=8` **下降**；
本项是 2.49ms 中的一支，**目标 −0.3 ~ −0.8ms**（覆盖 20 个空 SM + barrier 偏斜）。
**门禁 2（数值，红线）**：**`mean-k` 不掉**（A0 基线 **1.34**；掉了立即弃用该臂）；
另加三段文本红线：**零拉丁 / 0 double-char / 「先帝创业未半」**。
本项**论证上逐位等价**，门禁 2 **应当**自动通过 —— 但**照办测量**，不接受"论证过所以不测"。

### 6.3 最快的一枪：**同进程形状回执**（不需要 e2e）

若只想确认"env 生效 + 形状按预期变"，先跑一次 nsys 即可（§6.4），
**不必**先花 4 个 200-tok 进程。

### 6.4 nsys 判据（**本项的活性回执**）

```bash
DUR=60 MAXTOK=20 DSV41_WO_A_NWARPS=4 bash scripts/nsys_wave1.sh
/usr/local/cuda-13.2/bin/nsys stats --report cuda_gpu_kern_sum --format csv /tmp/wave1_nsys.nsys-rep | \
  grep -E "wo_a_grouped_gemv_kernel"
```

| 判据 | base (NW=8) | NW=4 | NW=2 | NW=1 |
|---|---|---|---|---|
| **实例数/步** | 40 (+3 draft) | 40 | 40 | 40 |
| **`Grid X`** | **128** | **256** | 512 | 1024 |
| `Grid Y` | 1 (verify) / 8 (draft) | 同 | 同 | 同 |
| `Block` | 256 | 128 | 64 | 32 |
| **`Total Time`（聚合）** | 2.49ms/步 | **↓ 为验收** | ↓ | ↓ |

- **活性**：`Grid X` 必须**按上表变化**；不变 ⇒ env 没进到进程（查 `run.env` /
  `/proc/<pid>/environ` 回读），**不是** kernel 的问题。
- **验收**：`wo_a_grouped_gemv_kernel` 的 **聚合 GPU 时间下降**（发数不变，所以**判据是时间不是实例数**）；
  同时看 **`Total Time / Instances`（per-instance us）** 是否降 —— 若聚合降而 per-instance 平，
  说明只是 scheduling 改善。
- **回归守卫**：draft 侧（`Grid Y = 8`，3 发/步）的 per-instance 时间**不得上升**——
  env 对**同一个 launcher 的两个调用点**同时生效，draft 的 `grid = (1024/nw)×8` 也在变。

### 6.5 止损（写死，避免变体矩阵）

1. **4 臂全中性**（`verify=` 中位数差 < 0.1ms）⇒ `nwarps` **不是**该 2.49ms 的病根
   ⇒ **停**，转 §4.1 的 `s_a` cpasync（最强候选）。
2. **单调上升**（4 优于 2 优于 1）⇒ 取 **4** 为生产值，并**上报**"§4① 的标量 staging 链是主因"，
   下一步做 §4.1-1。
3. **门禁 2 任意臂掉 `mean-k`** ⇒ **立即弃用该臂**（数值红线），上报尚书省。

---

## 7. 本轮编译验证（无 GPU 工作）

| 项 | 命令 | 结果 |
|---|---|---|
| Rust 侧 | `cargo check` | ✅ **EXIT=0**（仅既有 warning：dead_code / unreachable_code） |
| CUDA 侧 | 远端 `nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 -Xptxas -v -c dsv41_kernels.cu` | ✅ **EXIT=0 / errors=0** |
| 特化实例 | 同上 | ✅ `wo_a_grouped_gemv_kernel<1..8>` **8/8** 全部生成 |
| 寄存器 | `-Xptxas -v` | `Used 32 registers, used 1 barriers`（8 个实例**一致**）⇒ 占用率**受 smem 限制，非寄存器** |

> 32 regs / 1 barrier 是 §3 的直接支撑：占用率瓶颈**确定**在 smem（`4096*nwarps + 5632`），
> 故 `nwarps` 正是唯一能移动 blocks/SM 的自由度。

---

*工部 · 改动面：`kernels/cuda/dsv41_kernels.cu`（launcher `nwarps` env-gate + 2 处注释）；
`chain_dev.rs` / `experts_mxf4.cu` 未动（peer 区，mpar-rework / tcgen05-716 在改）。*
*生产路径默认（env 未设）与改前**逐指令相同**。*
