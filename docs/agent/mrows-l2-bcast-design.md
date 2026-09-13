# gemm_fp8_mrows 的 ⑤a L2-无-smem 广播 — 设计、逐位论证与验证手册

> 载体：`kernels/cuda/dsv41_kernels.cu` 的 `gemm_fp8_mrows_l2_kernel<M>` + `dsv41_gemm_fp8_mrows` launcher。
> Gate：`DSV41_MROWS_L2BCAST=rpb`（**默认 OFF**，unset/`0` = 逐字节回到 M-in-register 程序）。
> 上游设计：`tensorcore-proj-design.md` §5.1（deliverable ⑤a）。
> 病灶：`verify-amortization-lesion-audit.md` §2（fold_r 6× 退化）+ §10.9（MPAR 二连败）。
> 日期：2026-09-13。**GPU 未验证**——本文件是设计与验证手册，不是实测结论。

---

## §0 一句话

`fold_r` 把 M 折进 grid 时，每个 M-组 block **重新** `cp.async16` stage 同一份权重到自己的
private smem（= 权重 DRAM 往返 ×ng）⇒ 63.8 → 10.3 tok/s。⑤a 保留「M 进 grid」这一并行度
来源，但**删掉 smem staging**：每 warp 直接 `__ldg` 从 global 读自己那一行的 fp8 字节
（L1 → L2）。同组的 `ng` 个 block 读同一片权重地址，**第一次 HBM、其余 L2 命中**——于是
「1× 指令（每元素与 m=1 同构）× 6× 并行 + 零 prologue + 逐位」，恰好把 MPAR 失败的两个
成本项（1.59× 指令 + per-block prologue）同时去掉。

---

## §1 为什么是 ⑤a（两个前车之鉴）

| 臂 | 机制 | 实测 | 死因 |
|---|---|---|---|
| `fold_r`（M→grid） | 每 M-组 block 重新 `cp.async16` stage 同一权重行到 **private smem** | 63.8 → 10.3 tok/s（6× 恶化） | **权重 DRAM 往返 ×ng**（重复的是 prologue，不是字节）；audit §2 |
| MPAR（M→warp） | 1 warp / (输出行, 激活行) 对，权重 slab 每 block stage 一次 | rpb=1 +0.52ms，auto 累计 +1.28ms | **LUT build + slab staging 复制 ×grid** + **每元素 1.59× 指令**；mrows-mpar-design §2.3 / audit §10.9 |

**⑤a 的取舍表**：

| 维度 | fold_r（败） | MPAR（败） | ⑤a |
|---|---|---|---|
| 权重 staging | private smem slab ×ng | slab 每 block 一次（×grid） | **无**（直读 L1/L2） |
| LUT | smem 表 ×ng | smem 表 ×grid | **无**（`e4m3_to_f` 内联） |
| 权重 DRAM 字节 | ×ng | 1×（但 prologue 暴露 ×grid） | **1×**（同组 block 同波 → L2 命中） |
| 每元素指令 | 现状（M 折叠，1× 并行） | **1.59×** | **≈ m=1 的 1×** |
| 并行度（在飞 warp） | n·M（但被 DRAM 吃掉） | n·M | **n·M** |
| 数值 | 逐位 | 逐位 | **逐位**（§3） |

---

## §2 设计

### 2.1 kernel（`gemm_fp8_mrows_l2_kernel<M>`，dsv41_kernels.cu:5957）

```
grid : nt * ng        nt = ceil(n / nwarps)   （输出行 tile，来自 n）
                      ng = ceil(m / fold_r)   （激活行组，fold_r = 每块激活行数）
block: nwarps 个 warp（= dsv41_mrows_warps_for(n) * 32 线程，与 legacy 同几何）
warp : 负责 ONE 输出行 row = it*nwarps + warp，对 rn <= fold_r 个激活行各持 acc[q]
       无 prologue、无 barrier、无 smem
consume : for kb (== #pragma unroll 32):
              sb = ue8m0_to_f(__ldg(wsr[kb]))
              j  = kb*32 + lane
              wv = e4m3_to_f(__ldg(wr[j])) * sb              ← 直读 L1/L2，无 smem
              for q in 0..rn-1:
                  av = e4m3_to_f(__ldg(a[(r0+q)*k + j])) * __ldg(a_scale[(r0+q)*nb_k + kb])
                  acc[q] += av * wv                          ← 与 m=1 同序的单链
epilogue: 每 q 一棵 shfl_xor(16,8,4,2,1) → lane 0 写 out[(r0+q)*out_stride + row]
```

**与 legacy kernel 的差 = 两处 smem 槽被删、读数改指向 staging 拷贝的源地址**：
`dyn smem = 0` ⇒ 无 `__syncthreads`、无 per-M `cudaFuncSetAttribute`（legacy 的 smem ceiling
traps 在 ⑤a 根本不存在）。`row >= n` 时 warp-uniform 提前 return（无 barrier ⇒ 安全；
`shfl_xor` 全 mask 仍有效，因为 `row` 只依赖 warp index）。

### 2.2 grid 排布 = L2 论证本身

```
g  = blockIdx.x % ng      ← 激活行组「变化最快」
it = blockIdx.x / ng      ← 输出行 tile
```

同一个 `it`（同一片权重 tile `w[it*nwarps*k .. +nwarps*k)`）由 `ng` 个 block 读——让这 `ng`
个 block 的 `blockIdx` **相邻**，它们就落在同一波 → 同一地址的读合并成 **1 次 DRAM fetch +
(ng-1) 次 L2 命中**。反过来（`it` 变化最快）会把共享者散布到 `nt`（wo_b 达 640）之外，
L2 大概率仍能服务但没有局部性保证。这是本 kernel 唯一「非逐字照抄 legacy」的结构决定，
且与数值无关（哪个 block 算哪行不进 parity 论证）。

### 2.3 env gate + launcher 分派

| 取值 | 语义 |
|---|---|
| unset / `0` | **OFF**（M-in-register 程序，逐字节） |
| `=1` | `fold_r = 1`，**每行一个 block**（最大并行：~m × 在飞 warp） |
| `=2` / `=3` | 每块 2 / 3 个激活行（块少而宽） |
| `> m` | clamp 到 `m`（退化为 legacy 几何，但无 staging） |

launcher（`dsv41_gemm_fp8_mrows`）：在 `mpar` 声明后插入 ⑤a 分支，条件 `l2rpb > 0 && fold_r == m`，
**先于 MPAR**；两者同时 armed 时 ⑤a 胜出，回执显式标注 `[DSV41_MROWS_MPAR also armed -> shadowed]`
（两个 alternative program 不得同时跑；防「armed but inert」幻影 gate）。

`fold_r == m` 的约束同 MPAR：⑤a 自己带 M 的 grid 维，legacy `fold_r` 若同时 > 1 会
把两个 M 折叠复合（一片权重 tile 被读 `ng_l2 · ng_legacy` 次）。

**活性回执**（每进程一行，首次 ARMED launch）：
```
[mrows-l2] ARMED m=.. n=.. k=.. fold_r=.. -> ng=.. x nt=.. = .. blocks x .. threads, smem=0 (...)
```

---

## §3 逐位等价论证（deliverable ①a）

契约：`out[r][row]`（r = 激活行，row = 输出行）与 m=1 程序解码同一行**逐位相同**——
沿用 legacy `gemm_fp8_mrows_kernel<M>` header 的 C1–C6（该 kernel 已证明 == m=1 decode）。

| 契约 | legacy `gemm_fp8_mrows_kernel<M>` | ⑤a | 为什么相同 |
|---|---|---|---|
| C1 K 走序 | `kb` 升序，`j = kb*32 + lane`，`#pragma unroll 32` | 同 | 逐字保留（launcher 的 `mode >= 3` 前置不变） |
| C2 操作数/字节 | `s_lut[s_w[warp*k+j]]*sb`；`s_lut[s_a[q*k+j]]*s_as[...]` | `e4m3_to_f(__ldg(wr[j]))*sb`；`e4m3_to_f(__ldg(a[(r0+q)*k+j]))*__ldg(a_scale[...])` | `s_lut[b] ≡ e4m3_to_f(b)`（**表就是用它建的**）；`s_w`/`s_a` 是 `w`/`a` 的**纯拷贝**（拷贝宽度不可观测，legacy header 原话）；`s_as`/`wsr` 同源 |
| C2b a32 两臂 | a32=1 材料化 `af[q]`；a32=0 内联 | 单内联形 | 两臂是**同一 FMUL 的同一值**（legacy header：「Both arms are the same expression」）；故 ⑤a 对 `DSV41_GEMV_A32` 两个取值都逐位相等，**不重读该 gate**（同 MPAR） |
| C3 归约 | `shfl_xor` off=16,8,4,2,1，每 (warp,行) 一棵 | 同 | 逐字保留，每元素恰由 1 个 warp 算 |
| C4/C5 无跨行/跨 K 重组 | `acc[q]` 独立链；无 K-split | 同 | `wv` 的 hoist 只是复用同一 (row,kb) 值，不动结合律 |
| C6 累加式 | `acc[q] += av * wv`，`#pragma unroll 32` | 同 | 同一源表达式；拆 acc / 多路 unroll 会破坏（build.sh 记录过 ~1 ULP drift） |

**唯一变的是「哪个 warp 算哪个元素」与「字节从哪来」**——行独立（legacy header：
「the block geometry does not enter the parity argument」），而字节就是同一个字节。
故 `fold_r` 的任意取值（含 =1）都逐位等价于 m=1 程序。

---

## §4 L2 命中分析（deliverable ①b）

**流量账**（m = 6，`fold_r = 1` ⇒ ng = 6；权重按 `n*k` B，激活按 `m*k` B）：

| 投影 | n × k | nwarps | nt | ng | grid | 权重字节 | 权重 DRAM | 权重 L2 读 | 激活 L2 读 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `wq_a` | 1280×5120 | 4 | 320 | 6 | 1920 | 6.55 MB | **6.55 MB (1×)** | 39.3 MB (6×) | nt·m·k = 9.8 MB |
| `wkv` | 512×5120 | 4 | 128 | 6 | 768 | 2.62 MB | **2.62 MB (1×)** | 15.7 MB (6×) | 3.9 MB |
| `wq_b` | 4096×1280 | 8 | 512 | 6 | 3072 | 5.24 MB | **5.24 MB (1×)** | 31.4 MB (6×) | 3.9 MB |
| `wo_b` | 5120×1024 | 8 | 640 | 6 | 3840 | 5.24 MB | **5.24 MB (1×)** | 31.4 MB (6×) | 3.1 MB |
| `sh w1/w3` | 288×5120 | 2 | 144 | 6 | 864 | 1.47 MB | **1.47 MB (1×)** | 8.8 MB (6×) | 4.4 MB |

- **为什么 DRAM 是 1×**：B300 L2 ≈ 126 MB，最大的权重也才 6.55 MB——整个权重矩阵常驻
  L2绰绰有余。同 `it` 的 6 个 block 同波（`g` 变化最快），首批读 miss、其余命中。
- **激活为何不是新增放大**：legacy（fold_r=m）每 block 读 m 个激活行、共 `nt·m·k`；⑤a
  （fold_r=1）每 block 读 1 个激活行、共 `ng·nt·k = m·nt·k` ——**完全相同**。块内 `nwarps`
  个 warp 读同一激活行由 L1 吸收（不是新增 L2 流量）。
- **代价就是 L2 读 ×ng**——这正是验收判据：**DRAM ~1× / L2 读 ~ng×**（§6.3）。

**风险（tensorcore §5.1 已点名）**：网格 ×ng 倍块数（wo_b 640 → 3840）——块调度/波次的
固定开销可能重新吃掉收益（MPAR 的 L-A 教训）。故首轮必须做 **rpb sweep + nsys 波次判读**，
先看符号再看幅度。次风险：同组 block 同波 ⇒ 首读**同时 miss**（6 个 block 同时等 1 次
HBM fill）；换 `it` 快的排布可错开，但牺牲局部性——留作旋钮，不在首轮。

---

## §5 编译验证（deliverable ②）

| 目标 | 结果 |
|---|---|
| `cargo check --workspace --all-targets` | ✅ **EXIT=0**（仅既存 warning，`ferrite-serve` unreachable_code / `ar_micro` unused_variables——与本次无关） |
| 远端 `nvcc -c dsv41_kernels.cu`（sm_103a） | ✅ **NVCC_EXIT=0 / 0 errors**；8 个 `gemm_fp8_mrows_l2_kernel<1..8>` 全部 **`used 0 barriers`**（证实无 smem / 无 `__syncthreads`） |
| 远端 `nvcc -o t_gemm_mrows tests_dsv41_gemm_mrows.cu` | ✅ **EXIT=0 / 0 errors**（唯一 warning 是 MPAR kernel 既有的 `nwarp` 未引用 :5704，非本改动） |

**ptxas 寄存器/spill（sm_103a, -O3）**：

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| regs | 32 | 40 | 32 | 38 | 40 | 47 | 40 | 48 |
| spill | 0 | 0 | 0 | 0 | 0 | 0 | **28B st / 40B ld** | 0 |
| barriers | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

M=7 有 24B stack / 28B spill stores——`acc[7]` + 运行期 `rn` break 的小codegen 伪影。
**生产形状 m = 1 / 6 全部 0 spill**；M=7 只在 dispatch `m=1..8` 的罕见形状出现，且 legacy
`gemm_fp8_mrows_kernel` 在 M=7 也有同类现象（同一 `acc[M]`+`rn` 结构）。若后续在意，可在
M=7 走 fold_r=m 退化形态（无 staging，同样逐位）。

**改动清单（本任务只动一个文件）**：

| 文件 | 位置 | 内容 |
|---|---|---|
| `kernels/cuda/dsv41_kernels.cu` | :5884–6040（新区域） | `gemm_fp8_mrows_l2_kernel<M>` + `g_mrows_l2bcast` + `dsv41_mrows_l2bcast_for` |
| 同上 | :6122–6170（launcher 内） | `dsv41_gemm_fp8_mrows` 的 ⑤a 分支 + 活性回执（**在 `mpar` 声明后、MPAR 分支前**） |

**未触碰**：`chain_dev.rs` / `device.rs` / `dspark_dev.rs` / `build.sh` / `tests_*.cu`
（peer `proj-mma-wiring` 的改动区）。`git diff --name-only` = 仅 `kernels/cuda/dsv41_kernels.cu`。

**复现命令**：
```bash
nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 -Xptxas -v \
     -c kernels/cuda/dsv41_kernels.cu -o /tmp/l2_ubuntu/dsv41_kernels.o
```
（`.cu` 不在 cargo 编译图内——运行时 dlopen `libferrite_kernels.so`——故 Rust 侧不受影响。）

---

## §6 GPU 验证手册 — deliverable ③

> 前置：本任务**禁止 GPU/e2e**。以下由主 agent 在空闲 GPU 上执行。

### 6.1 构建（compile-only 同一机）

```bash
nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 \
     -o /tmp/l2_$USER/t_gemm_mrows kernels/cuda/tests_dsv41_gemm_mrows.cu
```

### 6.2 逐位等价验收（**先于性能**，micro bench）

`g_mrows_l2bcast` 是**文件级 static（load 时读 getenv）**，同进程无法 sweep ⇒ **一个 rpb 一个进程**：

```bash
for rpb in 1 2 3 6; do
  echo "== DSV41_MROWS_L2BCAST=$rpb =="
  DSV41_MROWS_L2BCAST=$rpb /tmp/l2_$USER/t_gemm_mrows --quick
done
DSV41_MROWS_L2BCAST=1 /tmp/l2_$USER/t_gemm_mrows          # 全形状
```

**判据**：每个值下 `RESULT: all checks passed`（m 行 launch 与 m 个 m=1 launch **逐位**相等，
NaN sentinel 覆盖无空洞）。任一位差 ⇒ 停，回报（不要拿性能数）。
每个 rpb 应打印 `[mrows-l2] ARMED ...` 一行——**没打印说明没走 ⑤a**，那次运行不算数
（可能被 legacy 更早的 decline 拦下：`mode < 3` / `NO_GEMV_FP8` / 形状）。

> 可选 follow-up：仿 `mr_mpar_contract()` 加一个 `mr_l2_contract()`（pin
> `dsv41_mrows_l2bcast_for(m) in [1,m]`）。本任务只改 `dsv41_kernels.cu`，未加。

### 6.3 性能验收（micro bench，nsys / CUDA event）

同形状、同 m 下比较（unset = legacy 基线）：

```bash
/tmp/l2_$USER/t_gemm_mrows --quick                              # legacy（M-in-register）
DSV41_MROWS_L2BCAST=1 /tmp/l2_$USER/t_gemm_mrows --quick        # ⑤a
```

**判据优先级**：先看**符号**（`t(m, ⑤a) / t(m, legacy)` 是否 < 1.0，目标 ~1/4–1/5 的单行
gemv 步进），再看幅度。**若更慢** ⇒ latency/调度成本占优，回报，不要硬调参数掩盖。

**rpb sweep**：`rpb ∈ {1, 2, 3, 6}`。`=1` 并行最大但块最多；`=6` 是最少块（无 staging 的 legacy 几何）。

### 6.4 nsys / ncu 判据（本路线的核心读数）

**nsys（波次/块数）**：确认 grid = `nt × ng` 且同 `it` 的 `ng` 个 block 落在同一波
（`launch__waves_per_multiprocessor`、GridX）。wo_b 640 → 3840 块、wkv 128 → 768 块。

**ncu（DRAM 1× / L2 ng×）**：

```bash
ncu --kernel-name regex:gemm_fp8_mrows -m \
    dram__bytes_read.sum,\
    lts__t_sectors_srcunit_tex_op_read.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    sm__throughput.avg.pct_of_peak_sustained_elapsed \
    /tmp/l2_$USER/t_gemm_mrows --quick
```

| 读数 | legacy 基线 | ⑤a 期望 | 说明 |
|---|---|---|---|
| `dram__bytes_read.sum`（权重占比） | 1× | **≈ 1×**（≤1.2×） | L2 接住了同组重读；**>1.5× ⇒ L2 排布失败，换 `it` 快排布** |
| `lts__t_sectors_..._read`（L2 读） | 1× | **≈ ng = 6×** | ×6 是本路线的**预期代价**，不是 bug |
| `sm__warps_active.pct` | ~13% | **显著上升** | 在飞 warp n·M；若仍 <25% ⇒ 块数不够 / 调度瓶颈 |
| `sm__throughput.pct` | 12.8–13% | 上升 | 从 latency-bound 向 throughput 移动 |

### 6.5 e2e 双门禁（仅当 micro bench 符号转正）

每个优化臂必须同时报告 **`step_ms`**（`[dspark]` 分解）**AND** **`mean-k`**：

- `mean-k`：⑤a **逐位** ⇒ 必须与当前栈（Fix A ON，§10.10 的 2.240）**在噪声内相同**——
  掉了 = 数值回归，立即弃用该 gate（红线，不是 byte-compare）。
- `step_ms`：目标把投影族（gemv 15.4% + mrows 15.2% ≈ 21% 时间）摊薄到 ~1/4–1/5。
- **不许只报步时**（accept 可能被牺牲）；**不许只报 acc**（可能零收益）。

### 6.6 回滚

```bash
unset DSV41_MROWS_L2BCAST     # 立即回到 M-in-register 程序（逐字节）
```

---

## §7 待定 / 风险

1. **符号未定**：⑤a 的赌注是「删掉 prologue + 1× 指令」足以把 fold_r 的 6× 退化翻正。
   若 `ng` 倍块数的调度/波次开销 + 同波首读 miss 仍吃掉收益 ⇒ 回报，转 ⑤b（cluster DSMEM）。
2. **同波首读 miss**：§2.2 的 `g`-快排布让 6 个 block 同时等同一次 HBM fill。若 ncu 显示
   DRAM 显著 >1×，改 `it`-快排布（错开），代价是局部性——需实测二选一。
3. **激活读的指令数**：每 warp 每 kb 一次 1-byte LDG（激活）+ 一次（权重）。若 L1 请求带宽
   成为新瓶颈（`l1tex__...mem_global_op_ld` 撞顶），follow-up：块内把激活行 stage 一次到
   smem（但这会重新引入小 prologue，需权衡）或改 `nwarps`。
4. **`fold_r = m` 的退化形态**：网格回到 `nt`、无 staging——也可作为「smem staging 到底值多少」
   的干净 A/B 点（同一排布，有/无 staging）。
5. **未做**：⑤b（cluster DSMEM 权重广播）——按 tensorcore §5.3，等 ⑤a 的结论出来再谈。
