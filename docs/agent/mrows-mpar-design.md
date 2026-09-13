# gemm_fp8_mrows 的 M-真并行（MPAR）— 根因解剖、设计与验证

> 载体：`kernels/cuda/dsv41_kernels.cu` 的 `gemm_fp8_mrows_mp_kernel<M>` + `dsv41_gemm_fp8_mrows` launcher。
> Gate：`DSV41_MROWS_MPAR`（**默认 OFF**，unset/`0` = 逐字节回到 M-in-register 程序）。
> 权威模型：`mtp-verify-amortization-model.md`（verify(m) ≈ eager(1)+ε，400 = step ~8ms + acc 2-3）。
> 病灶清单：`verify-amortization-lesion-audit.md` §2（根层：`gemm_fp8_mrows_kernel<M>` 的 M 进寄存器串列）。
> 日期：2026-09-13。**GPU 未验证**——本文件是设计与验证手册，不是实测结论。

---

## §0 一句话

`gemm_fp8_mrows_kernel<M>` 把 M 藏在**单个 warp 的寄存器**里（`float acc[M]`），并行度只来自 n
⇒ verify 的 m=6 退化为 latency-bound（实测 verify=eager 的 4.45×）。MPAR 把 **M 变成一个 warp 轴**：
一个 warp 算一个 (输出行, 激活行) 对，`rpb` 个输出行的权重行只在 smem **stage 一次**，
M 个 warp 共享它。在飞 warp 从 n 变成 n·M，权重流量仍是 1×，每行 fma 链逐位不变。

---

## §1 根因解剖（为什么 m=6 是 4.45×）

### 1.1 现有程序（`gemm_fp8_mrows_kernel<M>`，:5334）

```
grid : nt = ceil(n / nwarps) 块          ← 并行度只来自 n，与 M 无关
block: nwarps 个 warp
warp : 负责 ONE 输出行 row，对 M 个激活行各持一条 acc[q]（float acc[M]）
       prologue: 每 warp stage 自己那一行权重（cp.async16）；block 级 stage M 行激活 s_a[m*k]
       consume : for kb: wv = s_lut[s_w[j]]*sb
                        for q: af[q] = s_lut[s_a[q*k+j]]*s_as[q*nb_k+(kb)]
                               acc[q] += af[q]*wv
       epilogue: 每 q 一棵 shfl_xor 树 → out[q*out_stride + row]
```

生产形状（batched m=6，`l4-mgrid-first-step-design.md` §1.2）：

| 投影 | n × k | `nwarps` | 块数 | SM 覆盖 | smem(M=6) |
|---|---|---:|---:|---:|---:|
| `wq_a` | 1280 × 5120 | 4 | 320 | 216% | 56 064 B |
| **`wkv`** | **512 × 5120** | **4** | **128** | **86%** | **56 064 B** |
| `wq_b` | 4096 × 1280 | 8 | 512 | 346% | 19 904 B |
| `wo_b` | 5120 × 1024 | 8 | 640 | 432% | 16 128 B |
| `sh w1/w3` | 288 × 5120 | 2 | 144 | 97% | 45 824 B |

### 1.2 三个可分的病灶假说

| # | 假说 | 机制 | 预期 NCU 签名 |
|---|---|---|---|
| **H1** | **串行 acc** | 每 warp 的 `acc[M]` 把 M 行折叠成一条 warp 内的依赖链；M 倍指令 + 有限 ILP | `stall_wait`（定长依赖）高、IPC 低、FMA 管线未饱和 |
| **H2** | **smem staging / LDS 吞吐** | prologue（权重 cp.async16 + 256 项 LUT + M 行激活 stage）与 consume 的 gather LDS 是主成本 | `l1tex__data_pipe_lsu_wavefronts_mem_shared` 高、`stall_short_scoreboard` 高、bank conflict 高 |
| **H3** | **占用低（在飞 warp 不够）** | 128~320 块 / 148 SM、每块只 2~8 warp ⇒ 每 SM 4~16 warp（6~25%），无法掩盖访存延迟 | `sm__warps_active.avg.pct_of_peak` 低（<25%）、`stall_long_scoreboard` 高、块数 < SM 数 |

**关键结构性事实**：H1/H3 都预测「时间 ~M×」（加 M 倍工作、却无新增并行度来吸收），H2 预测
「时间 ~M× 且 LSU/smem 通道饱和」。所以**单看倍数不能分辨**，必须看**哪条管线饱和 + 占用**。
判读表见 §4。

### 1.3 fold_r 的反证（已经实测，锁定 H2 的存在）

`fold_r`（M 进 grid）实测把 63.8 → 10.3 tok/s（6× 恶化）——因为每个 M-组的 block **重新**
cp.async16 stage 同一份权重行 ⇒ 权重 HBM 流量 ×ng。这同时证明：

- **权重 staging 的 HBM 字节是真实成本**（不能无限复制）；
- **M 进 grid 这条路的死相**——所以任何「M 进 grid」的方案都必须先解决「权重只读一次」，
  否则就是 fold_r 的复刻。

### 1.4 设计死锁（审计判词）

> M-in-register（串行，H1） **vs** M-in-grid（重读权重，H2/fold_r）——两条路都通向 4.45×。
> 需要**第三条路：M 真并行 + 权重只 stage 一次**。

---

## §2 候选方向评估

| # | 候选 | 机制 | 权重流量 | M 并行 | 判决 |
|---|---|---|---|---|---|
| 1 | **Warp 级 M 分配**（本方案） | block 内每 warp 管一个 (输出行, 激活行) 对；`rpb` 行权重一次性 stage 到 smem，M 个 warp 广播读 | **1×** | **M×（n→n·M warp）** | ✅ **选定** |
| 2 | Block 级 M 分割 + L2 复用 | M 进 grid，权重不经 smem 直接 L2 读 | 1×（若 L2 常驻） | M×（块数 ×M） | ⚠️ 备选；L1 命中无保证、LDG 延迟高，先做 NCU 判读 |
| 3 | 两阶段（权重→smem 一次 + 全 M 共享） | 即**现状**：weight 本就 stage 一次，全 M 从 smem 算 | 1× | **1×** | ❌ 就是现状，串行 acc 未解 |
| 4 | Tensor Core（mma，B300 w8a8） | 用 mma 指令做 m=6 的 tile GEMM | — | — | 🔶 独立机会：数值路径变化，需 EAGER 逐位对照；不在本次 |

### 2.1 为什么选方向 1

1. **它在结构上同时满足两条硬约束**：权重只 stage 一次（不像 fold_r / 方向 2），M 又真并行
   （不像现状）。这正是 §1.4 的「第三条路」。
2. **它保住了逐位等价**（§3）：M 变成 warp 轴，每个 (行, 激活行) 仍由**恰好一个 warp** 以
   原有的 `kb` 升序、`acc += av*wv` 单链、每 warp 一棵 `shfl_xor` 树计算 ⇒ 与 m=1 程序逐位相同。
3. **它直接攻 H3**：在飞 warp 从 n 变成 n·M（verify 处 6×），block 由「每 SM 4~16 warp」
   变成「每 SM ~18~30 warp」。
4. **它把 H2 的代价限定在可控范围**：smem 反而更小（不再 stage M 行激活 s_a），代价只有
   「每行的权重 decode（`s_lut[rs[j]]*sb`）被 M 个 warp 各做一次」——见 §2.3 的计数表，
   总指令 ≈ 现状的 1.6×，摊到 6× 的 warp 上。

### 2.2 选定设计（`gemm_fp8_mrows_mp_kernel<M>`，:5616）

```
grid : ceil(n / rpb)                          ← 输出行按 rpb 分组（不再有 fold_r 维）
block: rpb * M 个 warp（<= 1024 线程）
warp : warp 编号 w = rr + rpb * g
       rr = w % rpb  → 本 block 的第 rr 个输出行 row = row0 + rr
       g  = w / rpb  → 激活行 g
       ── 同一行的 M 个 warp（g=0..M-1）读同一片 s_w + rr*k
prologue: 整块协作 cp.async16 stage rpb 行权重（= rpb*k 字节）到 s_w；
          build 256 项 LUT；retire → __syncthreads()（跨 warp 发布）
consume : rs = s_w + rr*k ; ar = a + g*k ; asr = a_scale + g*nb_k ; wsr = w_scale + (row>>5)*nb_k
          for kb: sb = ue8m0_to_f(wsr[kb]) ; j = kb*32+lane
                  wv = s_lut[rs[j]] * sb
                  av = s_lut[ar[j]] * asr[kb]      ← 激活直接从 global(L1) 读，不 stage
                  acc += av * wv                   ← 单条标量链
epilogue: 一棵 shfl_xor 树（off = 16,8,4,2,1）→ lane 0 写 out[g*out_stride + row]
```

**smem 布局**：`s_w`（rpb*k 字节）| `s_lut`（256 f32）。
**没有激活 slab**：每个激活行被恰好一个 warp 读（同 block 内同 g 的 rpb 个 warp 共享 L1 副本），
block 级 stage 只会把 M 比例的 prologue 重新引入，与设计目标相反。

**launcher 选路**（`dsv41_gemm_fp8_mrows`，:5798）：

```
mpar = dsv41_mrows_mpar_for(m)           // 0 = OFF
if (mpar > 0 && fold_r == m) {           // 两个 M 折叠不能同时开
    launch gemm_fp8_mrows_mp_kernel<m>  with  rpb = mpar,  block = mpar*m warps
    return
}
... 原路径（M-in-register），逐字节不变
```

`fold_r == m` 的约束：MPAR 把 M 放在 **warp 布局**里，grid 不能再带 M（否则两个独立 M 折叠）。

### 2.3 指令计数表（每 warp 每 kb；`wkv` 形状 n=512, k=5120, M=6）

| 类别 | 现状 `mrows<M>`（每 warp 1 输出行 × M 激活行） | MPAR（每 warp 1 个 (行,激活) 对） |
|---|---:|---:|
| LDS `s_w` | 1 | 1（`rs`） |
| LDS `s_lut`（权重 decode） | 1 | 1 |
| LDS `s_a` | **M** | 0（改 LDG） |
| LDS `s_lut`（激活 decode） | **M** | 1 |
| LDS `s_as` | **M** | 0（改 LDG） |
| **LDS 小计** | **2 + 3M = 20** | **3** |
| LDG `wsr` | 1 | 1 |
| LDG `ar` / `asr` | 0 | 2（L1 命中） |
| **LDG 小计** | **1** | **3** |
| FMUL（`wv`、`av`） / FMA | 1 + 2M = 13 | 3 |
| **每 warp 每 kb 合计** | **34** | **9** |
| **warp 总数** | n | n·M = 6n |
| **每 kb 全 grid 合计** | **34n** | **54n** |
| **每输出元素** | 34/M ≈ **5.67** | **9** |

**读表**：
- MPAR 的总指令 ≈ 现状的 **1.59×**（LSU 分量 ≈ 1.7×，FP 分量 ≈ 1.38×）——即 kernel header 里
  「~1.7×」的出处（那是 LSU 口径）。
- 代价是**权重 decode 被 M 个 warp 各做一次**（LDS `s_lut` 的 1 → 每行 M 次），
  **换回**的是「激活 stage 的 M 条 LDS」变成「M 个 warp 各 1 条 LDG（L1）」，以及**最重要的**：
  并行度 n → nM。
- 每元素 9 条指令 == **m=1 程序每元素的 9 条**（现状 M=1 时：5 LDS + 1 LDG + 3 FP = 9）。
  所以 MPAR 是「把 m=1 的每元素成本复制到 M 个 warp 上」，不是「加 M 倍成本」。
- **符号未知的是「1.59× 指令」vs「6× 并行度」**：现状若确为 latency-bound（H1/H3），MPAR 赢；
  若现状其实 issue/吞吐-bound（H2 主导），MPAR 输。**这正是 §4 微基准要定的符号。**

### 2.4 `rpb`（每块输出行数）是唯一的调参轴

`rpb*m <= 32`（1024 线程 / 32）。`DSV41_MROWS_MPAR=N`：

| 取值 | 语义 |
|---|---|
| unset / `0` | **OFF**（现状程序，逐字节） |
| `auto`（或负值） | `rpb = 1024/(32m)`（最宽块）：M=1→32, 2→16, 3→10, 4→8, 5→6, 6→5, 7→4, 8→4 |
| `=N` | `rpb = clamp(N, 1, 1024/(32m))`，**显式 sweep 用** |

`rpb` **不改变总流量**（权重 stage = grid·rpb·k = n·k；激活读 = grid·rpb·M·k = n·M·k），
只改变「块数 vs 块宽」的分布：

- `rpb` 大 → 块少而宽（prologue/LUT 复制少），但 grid 可能 < 148 SM（`wkv` n=512, auto→103 块）。
- `rpb` 小 → 块多而窄（分布均匀），但 LUT build 复制多。

> ⚠️ **给 GPU 首跑的提示**：`auto`（最宽块）在**小 n**（`wkv` n=512、`sh w1/w3` n=288）会让
> grid（103 / 58）**低于 SM 数 148**，可能留下 30~60% 的 SM 空转。首轮 sweep 必须包含
> `rpb ∈ {1, 2, 3, auto}` 直接看分布的影响，别把 `auto` 当定论。

---

## §3 数值红线（逐位等价论证）

**契约**：MPAR 的 `out[g][row]` 与 m=1 解码该行**逐位相同**（沿用现状 kernel header 的 C1–C6）。
MPAR 与 `gemm_fp8_mrows_kernel<M>` **是同一个程序的重排**，只换了「谁算哪个元素」：

| 契约 | 现状 | MPAR | 为什么相同 |
|---|---|---|---|
| C1 K 走序 | `kb` 升序，`j = kb*32 + lane` | 同 | 逐字保留 |
| C2 操作数/字节 | `s_w[rr*k+j]`、`s_a[q*k+j]`、`s_as[..]`、`wsr[..]` | `s_w[rr*k+j]`、`a[g*k+j]`、`a_scale[g*nb_k+kb]`、`w_scale[..]` | 激活 slot 是**纯拷贝**（`s_a[q*k+i]=a[...]`），拷贝宽度不可观测；`asr[kb] ≡ s_as[q*nb_k+(j>>5)]`（`j>>5==kb`） |
| C3 归约 | 每 q 一棵 `shfl_xor` off=16,8,4,2,1 | 每 (warp) 一棵，同 off 序列 | 每元素恰好由 1 个 warp 计算、1 棵树 |
| C4/C5 无跨行/跨 K 重组 | `acc[q]` 独立链 | 标量 `acc` 独立链 | 一个 warp 一个元素 |
| C6 累加式 | `acc[q] += af[q]*wv`，`#pragma unroll 32` | `acc += av*wv`，`#pragma unroll 32` | 同一表达式（见下） |
| a32 两臂 | a32=1 材料化 `af[q]`，a32=0 内联 | 内联 | 两臂是同一 FMUL 的同一值（header 的「bit-identical by construction」），故 MPAR 单形式对两个 gate 取值都逐位相等；**MPAR 不重读 `DSV41_GEMV_A32`** |

**唯一被允许变化的东西是「哪个 warp 算哪个元素」**（行独立 ⇒ 不进入 parity 论证，现状 header 原话：
「the block geometry does not enter the parity argument — rows are independent」）。所以：
**M 分配改变后，每行的 fma 链顺序、每行的归约树逐行保持。**

---

## §4 NCU 判读表（三假说怎么区分）— deliverable ③

**纪律**：**只跑 micro bench**（`tests_dsv41_gemm_mrows.cu` 二进制），**不能 e2e**
（`verify-amortization-lesion-audit.md` §5.3）。同形状跑 m=1 与 m=6 两轮，逐指标对照。

### 4.1 分工判据（单轮 m=6 就能定性质）

| 假说 | 主判据指标 | 读数模式（m=6） | 反证 |
|---|---|---|---|
| **H3 占用低** | `sm__warps_active.avg.pct_of_peak_sustained_active`；`launch__waves_per_multiprocessor`；`launch__occupancy_limit_*` | **< 25%**；waves < 1（`wkv` 128 块 < 148 SM） | 若 m=1 与 m=6 的 `warps_active` **相同且都 <25%** ⇒ H3 是**发射属性**（grid 不变），非 M 属性 ⇒ MPAR 的正靶 |
| **H1 串行 acc** | `smsp__warp_issue_stalled_wait_per_warp_active`（定长依赖）；`smsp__inst_executed.avg.per_cycle_active`（IPC）；`sm__pipe_fma_cycles_active` | `stall_wait` **高** + IPC **低** + FMA 管线**未饱和** | 若 IPC 随 M 基本不变（0.5~1 区间）⇒ 不是纯依赖链 |
| **H2 smem/staging** | `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`；`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`；`smsp__warp_issue_stalled_short_scoreboard_per_warp_active` | LSU 波前 / bank conflict **高**，随 M 上升；`short_scoreboard` 高 | 若 `smem wavefronts` 在 m=6 只 ~M× 于 m=1 的**同一** grid，而时间也 ~M× ⇒ H2 主导（加的工作就是 smem） |

**一图流**：
```
看 sm__warps_active.pct  ── <25%? ── 是 ──► H3 参与（MPAR 正靶）
                            │否
                            ▼
看哪条管线饱和 ── FMA pipe 饱和 / stall=wait ──► H1（串行链）
                 │
                 └── LSU(smem) wavefront/bank-conflict 饱和 / stall=short_sb ─► H2（staging）
```

### 4.2 定符号实验（决定 MPAR 的 sign）

MPAR 的赌注：**m=6 现状是 latency-bound（H1/H3）**，所以「1.59× 指令 × 摊到 6× warp」净赢。
若现状其实 **issue-bound（H2 主导）**，MPAR 会 ~1.6× 变慢（header 已如实警告）。

| 测量 | latency-bound（MPAR 预期赢） | issue-bound（MPAR 预期输） |
|---|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed`（m=6 现状） | 低（<50%） | **高（>80%）** |
| `smsp__inst_executed.avg.per_cycle_active`（IPC） | 低（现状浪费） | 高（已满发） |
| `sm__warps_active.pct`（m=6 现状） | <25% | 任意，但管线饱和 |
| MPAR 实测 step / kernel 时间 | 下降（目标 1.0~1.2× m=1） | 上升 ~1.6× |

### 4.3 直接判读表（`ncu --set full` 或指定 metrics）

```
ncu --target-processes all --launch-count 20 --kernel-name regex:gemm_fp8_mrows \
    -m sm__warps_active.avg.pct_of_peak_sustained_active,\
       smsp__inst_executed.avg.per_cycle_active,\
       sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
       l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
       l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
       smsp__warp_issue_stalled_wait_per_warp_active.pct,\
       smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
       smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
       launch__waves_per_multiprocessor \
    ./t_gemm_mrows
```

（若指标名在 NCU 版本上不同，用 `--metrics regex:stalled`、`--section SchedulerStats`
`--section WarpStateStats` `--section Occupancy` `--section MemoryWorkloadAnalysis` 取得等价的
分组；关键是**占用 / IPC / 管线饱和 / stall 类型**四组。）

---

## §5 实施（deliverable ②）

### 5.1 改动清单（本任务，kernel 侧）

| 文件 | 改动 |
|---|---|
| `kernels/cuda/dsv41_kernels.cu` | ① 新增 `gemm_fp8_mrows_mp_kernel<M>`（:5616，MPAR 程序）；② 新增 gate `g_mrows_mpar` + `dsv41_mrows_mpar_for(m)`（:5704）；③ launcher `dsv41_gemm_fp8_mrows` 加 MPAR 选路分支（:5798）+ `[mrows-mpar] ARMED` 活性回执 + 每 M 特化的 smem 属性 |
| `kernels/cuda/tests_dsv41_gemm_mrows.cu` | 新增 MPAR contract pin + 头注释（同进程无法 sweep，逐值重跑） |
| `docs/agent/mrows-mpar-design.md` | 本文件 |

**默认 OFF**：`DSV41_MROWS_MPAR` 未设 ⇒ `dsv41_mrows_mpar_for` 返回 0 ⇒ 走原路径，逐字节不变。

### 5.2 编译验证

- `cargo check --workspace --all-targets` → **EXIT=0**（Rust 侧不受影响；只有 warning，无 error）。
- 远端 `nvcc` compile-only（`-gencode arch=compute_100a,code=sm_100a -O3 --use_fast_math -std=c++17`，
  远端 CUDA 13.2，`ubuntu@43.202.208.136`，私有目录 `/tmp/mpar_ubuntu`，**无 GPU**）：

  | 目标 | 结果 |
  |---|---|
  | `nvcc -c kernels/cuda/dsv41_kernels.cu` | **EXIT=0**（仅既有 warning：`k1max` / `e2m1_to_f` 未引用，非本次改动） |
  | `nvcc -o t_gemm_mrows tests_dsv41_gemm_mrows.cu` | **EXIT=0**（含新增 MPAR contract pin） |

  命令（可复现）：
  ```bash
  nvcc -gencode arch=compute_100a,code=sm_100a -O3 --use_fast_math -std=c++17 \
       -c kernels/cuda/dsv41_kernels.cu -o /tmp/mpar_$USER/dsv41_kernels.o
  nvcc -gencode arch=compute_100a,code=sm_100a -O3 --use_fast_math -std=c++17 \
       -o /tmp/mpar_$USER/t_gemm_mrows kernels/cuda/tests_dsv41_gemm_mrows.cu
  ```

> ⚠️ **`chain_dev.rs` / `device.rs` / `dspark_dev.rs` 不在本任务范围内**（任务书明列禁止区，peer 改动区）。
> 工作树里这些文件存在**并发 peer**（`dsv41/deepseek-v41-flash-support`）正在改动的 Rust 代码 +
> 前一个（已死）subagent 留下的 180 行 WIP（`DSV41_ENGRAM_PROJ_MROWS` / `DSV41_COMPRESSOR_PROJ_MROWS`
> gate，peer 已补齐 `device.rs` 的 `dsv41_gemv_f32_mrows` 绑定）。**本任务全程未写这三个文件**
> （`git diff --name-only`：本任务的写入只有 `kernels/cuda/dsv41_kernels.cu`（前序 MPAR 代码）+
> `kernels/cuda/tests_dsv41_gemm_mrows.cu` + 本文件）。
>
> 中途曾观察到 `cargo check` EXIT=101：6 个错误**全部**是 peer 改到一半的 `sparse_attn` /
> `sparse_attn_orope` 参数数不匹配（`chain_dev.rs` / `device.rs:3207` / `dspark_dev.rs`），
> 与 mrows 完全不相交；peer 收敛后重跑即 **EXIT=0**。`.cu` 文件不在 cargo 编译图内
> （运行时 dlopen `libferrite_kernels.so`），其校验走远端 `nvcc`。

### 5.3 活性回执（防幻影 gate）

ARBOM 时第一次 ARMED launch 打印一行：
```
[mrows-mpar] ARMED m=.. n=.. k=.. rpb=.. -> block=.. warps, grid=.., smem=..
```
gate 未设时**不打印**。若 `.env` 里 armed 而日志无此行 ⇒ launcher 更早就 decline 了
（mode<3 / NO_GEMV_FP8 / 形状拒绝 / `fold_r != m`）。

---

## §6 GPU 验证手册 — deliverable ④

> 前置：本任务**禁止 GPU**。以下由主 agent 在有空闲 GPU 的机器执行。
> 远端（`43.202.208.136`）已有 `/tmp/mrows_bench`（`tests_dsv41_gemm_mrows.cu` 编出的可执行）。

### 6.1 构建（远端，compile-only 也在同一机）

```bash
# 每次改内核后重建（无 GPU 也可编译）：
nvcc -gencode arch=compute_100a,code=sm_100a -O3 --use_fast_math -std=c++17 \
     -o /tmp/mpar_$USER/t_gemm_mrows kernels/cuda/tests_dsv41_gemm_mrows.cu
```

### 6.2 逐位等价验收（**先于性能**）

`dsv41_mrows_mp_kernel` 的 `g_mrows_mpar` 是**文件级 static（load 时读 getenv）**，
同进程无法 sweep ⇒ **一个 rpb 一个进程**：

```bash
for rpb in 1 2 3 5; do
  echo "== DSV41_MROWS_MPAR=$rpb =="
  DSV41_MROWS_MPAR=$rpb /tmp/mpar_$USER/t_gemm_mrows --quick
done
DSV41_MROWS_MPAR=auto /tmp/mpar_$USER/t_gemm_mrows --quick   # 全形状
```

**验收判据**：每个值下 `RESULT: all checks passed`（m 行 launch 与 m 个 m=1 launch **逐位**相等，
sentinel 覆盖无空洞）。任一位差 ⇒ 停，回报（不要拿性能数）。

- 形状覆盖：`--quick` 覆盖生产 6 形状（`wq_a`/`wkv`/`wq_b`/`wo_b`/`sh`/`m=1`）；
  全量再补 `dispatch/m=1..8`、`n%nwarps`、`tiny`。
- 每个 rpb 应打印 `[mrows-mpar] ARMED ...` 一行——**没打印说明没走 MPAR**，那次运行不算数。

### 6.3 性能验收（micro bench，nsys 或 CUDA event）

在**同一形状**上比较（`m=1` 是基准）：

```bash
# 现状（M-in-register）
          /tmp/mpar_$USER/t_gemm_mrows --quick
# MPAR
DSV41_MROWS_MPAR=auto /tmp/mpar_$USER/t_gemm_mrows --quick
```

- **目标**：`t(m=6, MPAR) / t(m=1) ≈ 1.0~1.2`（现状是 2~4×）。
- **rpb sweep**：`rpb ∈ {1,2,3,auto}`——`auto` 在 `wkv`/`sh` 上 grid < 148 SM，
  很可能不是最优（§2.4）。
- **判据优先级**：先看符号（是否 < 现状的 1.0×），再看幅度。**若 MPAR 更慢** ⇒ §4.2 的
  issue-bound 分支成立，回报主 agent，不要硬调参数掩盖。

### 6.4 NCU 深层判读（micro bench only）

见 §4.3 的命令。**不要**对 e2e serve 跑 NCU（审计 §5.3）。

### 6.5 回滚

```bash
unset DSV41_MROWS_MPAR     # 立即回到 M-in-register 程序（逐字节）
```

---

## §7 待定 / 风险

1. **符号未定**：MPAR 是 1.59× 指令换 6× 并行度。§4.2 的 NCU 判读定符号。若现状实为
   issue-bound，则需转向**方向 2（L2 复用）**或**降低每元素指令**（例如把 decode 后的 f32
   权重 stage 进 smem 一次，rpb=1 时 4·k·4B = 20KB 可行）——列为 follow-up。
2. **`rpb=auto` 的 SM 覆盖**：小 n 时 grid < 148（§2.4）。首轮必须 sweep。
3. **方向 4（tensor core）**：m=6 fp8 mma 是独立机会，但它换数值路径，须 EAGER 逐位对照
   （不在本次）。
4. **`chain_dev.rs` 的 180 行 WIP**：见 §5.2 注，非本任务。
