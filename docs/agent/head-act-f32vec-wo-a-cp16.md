# head-act-f32vec + wo-a-cp16：两个逐位安全小票面的打包交付（2026-09-13）

> 出处：`verify-amortization-lesion-audit.md` §10.2（proj-head 判决 + wo-a-opt 备注 3）。
> 范围：**只改两个 .cu**（`dsv41_glue.cu` 小项①、`dsv41_kernels.cu` 小项②），两个独立 env 门禁，默认 OFF。
> 纪律：compile-only（本机 `cargo check` + 远端 b300 `nvcc -c`）；**本交付不含任何 GPU/e2e 数字**，GPU 手册见 §4。

---

## 0. 落地位置修正（与任务书的一处出入，先报）

任务书写「小项① kernel：`dsv41_kernels.cu` 搜 gemv_bf16_v1_mrows」。实际：

| 符号 | 文件 | 行（HEAD 4593c8e） |
|---|---|---|
| `gemv_bf16_v1_mrows_kernel<M>` + launcher `dsv41_gemv_bf16_v1_mrows` | **`dsv41_glue.cu`** | kernel :440 / launcher :1252 |
| `wo_a_grouped_gemv_kernel<M>` + launcher `dsv41_wo_a_grouped_fp8` | `dsv41_kernels.cu` | kernel :7436 / launcher :7548 |

⇒ 小项① 的改动落在 `dsv41_glue.cu`（不是 `dsv41_kernels.cu`）。**冲突防护仍然成立**：peer `a4-inv-bugfix` 改的是 `dsv41_kernels.cu:2053 apply_rope_mrows_kernel`（我的②在 :7436+，不同区域），peer `tcgen05-716` 在 `dsv41_experts_mxf4.cu`，**无人碰 `dsv41_glue.cu` 的 gemv 区**。开工 `git status` = clean（peer 改动未落盘）。

---

## 1. 小项 ①：`DSV41_HEAD_ACT_F32VEC`（默认 OFF）

### 1.1 病灶与方案

`gemv_bf16_v1_mrows_kernel` 把 m 行激活折进一次权重扫描（消掉 m 次权重重读），但**激活仍是每行独立读**：每个输出行的 warp 都要走完 M 行 × k 列的 f32。激活 4B/元素 vs 权重 2B/元素 ⇒ 每输出行的激活流是权重流的 `2*M` 倍。draft head 形状 (m=6, k=5120) 下，一个 8-warp block 搬 **960 KB 激活 vs 80 KB 权重**——摊薄省下的权重字节，在激活侧又付回去了（这就是「摊薄失效」的字节账）。

**方案 = 激活读路径 smem staging（只改读法，不改数值）**：block 把 M 行激活的 **K-chunk（KT=1024）纯拷贝**进 shared memory 一次，`__syncthreads()` 后每个 warp 从 smem 读自己的输出行。每 (block, chunk) 的激活搬运量从 `nwarps*M*kt` 降到 `M*kt` ⇒ 全局/L2 激活流量 **÷8**，被替换掉的 per-lane 读（原为 L1 常驻但被权重流冲刷）变成 LDS。

### 1.2 逐位论证（读法不改数值）

| 契约 | 不变性 |
|---|---|
| lane→c 元素序列 | legacy `for (c = lane; c < k; c += 32)`；新版 `for (c = base+lane; c < base+kt; c += 32)`，`base` 以 KT=1024（32 的倍数）**升序**步进 ⇒ 每个 lane 看到的 c **集合与顺序逐字相同** |
| `wv = __bfloat162float(wr[c])` | 同位置同字节，仍然 hoist 出 r 循环 |
| `acc[r] += wv * x[r*k+c]` | 同一条单链 FMA，每行一个独立累加器，跨 r 无任何合并（C4 保持） |
| 规约 | 同 `__shfl_xor_sync(0xFFFFFFFFu, a, off)`、同 off=16,8,4,2,1 树 |
| 写回 | 同 `if (lane == 0) out[r*n + row] = a`（无 bias，与 legacy 一致） |
| row→warp / grid | 非契约（行独立，kernel 头已声明）；新版复刻 legacy 的 `row = blockIdx.x*nwarp + wid; row += gridDim.x*nwarp` 映射 |

**唯一改动 = 同一批 f32 字节从哪块内存读出**：staging 是 `sx[r][i] = x[r*k + base + i]` 的**纯拷贝**（零算术），consume 只读 staging 槽位 ⇒ 任何一行输出的每一个 bit 都不动。

> ⚠️ 明确排除的伪方案：**float4 版 consume body**（每 lane 消费连续 4 个 c）会改变 lane→c 归属 ⇒ 改变每个部分和，那正是 `gemv_bf16_nt_kernel` 的 v2 程序（`head_gemv_bf16_mrows` 曾以 ~1e-3 偏差、33% 回声失败于 verify，b8b67c0）。本小项的向量化**只限拷贝宽度**（float4 load/store 同一批字节，16B 未对齐时自动退回标量）。

### 1.3 形状 / 死锁防护

- staging 是 **block 级**，故所有 `__syncthreads()` 必须被 block 内每个线程到达：行下标与 trip count 全部由 **block 一致量**（`row0`、`stride`、`iters`）算出；`row < n` 谓词**只**包住 consume 与 store，**不包任何 barrier**（最后一 block 的不活跃 warp 照样走满 barrier，不会死锁）。
- smem = `M*1024*4` ≤ **32 KB**（M≤8）< 默认 48 KB 动态上限 ⇒ **不需要 `cudaFuncSetAttribute` 舞步**，也不掉占用率。
- **为什么不 stage 整块 M×k slab**：那是 120 KB @ (6,5120) ⇒ 1 block/SM，占用率塌方；同族 `gemv_bf16_kernel` 的注释里记着「per-block 激活 staging 实测 2.3× step-time 回归」（2026-09-11 revert）。K-chunk 版把 smem 压到 32 KB 以内，代价是每 pass 10 个 barrier（5 chunk × 2）。**这是工部的取舍判断，请尚书省知悉**：若更想要「一次 stage 整块」的形态，KT 改为 k 只剩 1 行改动。

### 1.4 回执

```
[head-act-f32vec] ARMED m=.. n=.. k=.. blocks=.. x 256 threads, KT=1024, smem=.. B -> activation read path = smem staging (staged once per block per K-chunk, shared by all 8 warps x M rows; float4 copy where 16B-aligned, scalar otherwise)
[head-act-f32vec] ARMED but DECLINED m=.. k=..: smem=.. B > 48KB -> the legacy read path ...
```
每进程一行；门禁未设（出厂默认）时**什么都不打印**。加上新 kernel 自带符号名（`gemv_bf16_v1_mrows_act_kernel<M>`），nsys 的 kernel-name 本身就是「这臂跑没跑」的硬证据（不依赖 env 回读）。

---

## 2. 小项 ②：`DSV41_WO_A_CPASYNC`（默认 OFF）

### 2.1 病灶与方案

`wo_a_grouped_gemv_kernel` 的**权重**行早已是 `dsv41_cp_async16`（16B/lane/issue，R1 2026-09-12），而**激活**侧 `s_a` 还是逐字节标量：

```c
for (int i = threadIdx.x; i < k; i += blockDim.x) s_a[i] = ar[i];   // 1 B/thread/issue
```

k=4096 时 16 个 issue（256 线程），权重侧 512 B/warp-issue ⇒ **同一批字节 16 倍指令**。改成同一规则（`dsv41_cp_async16` + `cp.async.commit_group`）即可。

### 2.2 逐位论证（字节搬运，不改数值）

- staging 是**纯拷贝**：同一个全局地址的同一批字节 → 同一个 `s_a` 槽位（`s_a[i] = ar[i]` 无任何算术）；
- consume 路径**一字未动**：`kb` 升序、`j = kb*32 + lane`、`acc += (s_lut[s_a[j]] * s_as[j>>5]) * (s_lut[row_s[j]] * sb)`、`#pragma unroll 32` 串行链、`shfl_xor` 树、`og[...] = acc + (bias ? bias[row] : 0.f)` 全同；
- ⇒ 与 m=1 `dsv41_gemm_fp8_mx` 的 C1–C6 逐位等价（kernel 头）继续成立。

### 2.3 对齐 / 同步

- 对齐：launcher 已拒 `k & 31` 与 `a_stride & 31` ⇒ 每行偏移 `r*a_stride + g*k` 是 32 B 的倍数 ⇒ 16B 对齐；`s_a = s_as + nb_k`，`nb_k = k/32` floats = k/8 B（4 的倍数）⇒ 16B 对齐。两侧仍用 `dsv41_f4_ok` **运行时复检**，不齐则退回标量循环（err-716 陷阱的既有护栏）。
- 同步：`s_a` 是 **block 级**槽位（每个 warp 的 consume 读别的 warp 搬的字节）⇒ cp.async 组必须在**发布 barrier 之前** retire：
  `issue → commit → wait_all → __syncthreads() → consume`（`gemm_fp8_mrows_kernel` 的 a16 先例明文记着「wait 放在 barrier 之后是 data race」）。
- k%16 tail 与权重侧同一写法保留（launcher 全部拒 `k & 31`，属不可达防御）。

### 2.4 回执

```
[wo-a-cp16] ARMED rows=.. groups=.. n=.. k=.. nwarps=.. smem=.. -> activation staging = cp.async16 (16B per lane per issue)
```
（同一行的另一分支：16B guard 判定不齐时打印 `scalar loop (... the arm is INERT here ...)`，把「armed 但空转」的幻影门堵死。）

---

## 3. compile-only 证据

| 项 | 命令 | 结果 |
|---|---|---|
| 本机 cargo check | `cargo check` | ✅ EXIT=0（仅 `ferrite-serve` 既有 warning ×1，与本次无关） |
| 远端 nvcc（glue） | `nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 -c dsv41_glue.cu` | ✅ 见 §3.1 |
| 远端 nvcc（kernels） | 同上 `dsv41_kernels.cu` | ✅ 见 §3.1 |

### 3.1 远端 nvcc 结果（b300, CUDA 13.2, `ubuntu@43.202.208.136`, compile-only 无 GPU）

> 见本文件末尾「实测粘贴」段（交付时由工部填入本次 rc + ptxas 行）。

---

## 4. GPU 验证手册（**双门禁小票面**，禁止 e2e 的纪律除外——本手册就是给拥有者跑 e2e 的）

### 4.0 前置（先钉住「这臂真的跑了」）

两条 gate 都有**自己的 kernel 符号名**（① `gemv_bf16_v1_mrows_act_kernel<M>` / ② 同符号、只多一个 cp.async 段），因此：

1. **env 回读**（§10.1 纪律，幻影门温床）：`grep -a DSV41_HEAD_ACT_F32VEC /proc/<pid>/environ`、`grep -a DSV41_WO_A_CPASYNC /proc/<pid>/environ` 必须命中；
2. **回执行**必须在 stderr 出现：`[head-act-f32vec] ARMED ...` / `[wo-a-cp16] ARMED ...`；
3. nsys 表里必须出现 ① 的新 kernel 名（② 靠回执 + 时间）。

### 4.1 逐位门（**红线：mean-k 不变**）

| 步 | 臂 | 期望 |
|---|---|---|
| G0 | 基线（两 gate 全 OFF） | `[dspark] mean-k` = 基准值（当前 SWALLOW 栈 2.240）；计数 first-51 OK |
| G1 | 只开 `DSV41_HEAD_ACT_F32VEC=1` | **mean-k 与 G0 完全相等**（逐位等价 ⇒ argmax 分布不动）；in-process 逐位比对见 4.3 |
| G2 | 只开 `DSV41_WO_A_CPASYNC=1` | 同上 |
| G3 | 双开 | 同上 |

判据：mean-k **不是「接近」而是「相等」**（这两项是纯拷贝/读法改动，任何 delta 都说明实现有 bug，不是噪声）。辅以 §10.5 的护栏：`DSV41_TAP_PARITY=1` / `DSV41_COMP_PARITY=1` 在独占进程各跑一次，必须仍全 IDENTICAL。

### 4.2 时间门（kernel 相对时间 + 步时）

1. **per-kernel 时间**（nsys 相对倍数，按 §5 纪律）：`DSV41_AR_V5=0 DSV41_GRAPH_STEP=0`、`env -u FERRITE_P2P`、`NCCL_NVLS_ENABLE=0`、5 分钟 SIGINT 硬帽；看 `gemv_bf16_v1_mrows*`（①）与 `wo_a_grouped_gemv*`（②）两行 self time。
   - ① 目标：**同形状同 m 的 per-launch 时间下降**（staging 摊销 = 8× 少读激活）；若上涨 ⇒ L2 已在替 smem 做事，把 KT 调大（减少 barrier）或直接弃用。
   - ② 目标：**时间持平或下降**（纯指令数优化，无流量变化；55.7µs 的 wo_a 里 staging 只占小头，票面本来就是小）。
2. **步时**（非 nsys 的 e2e serve，看 `[dspark] steps=` 的 draft/verify/commit 分解）：① 只影响 draft+verify 的 head 调用（`gemv_bf16_v1_mrows` 352 发/轮），② 只影响 draft 的 wo_a（40 发/轮）。
3. **AR_SAFE vs 生产**：§10.2 已判「AR_SAFE 下 head 读全量 1.323GB（8× 放大）」。① 的收益按**生产口径（v5 ON、词表切片 16160 行）**判；AR_SAFE 轮的 head 数字只作倍数参考。

### 4.3 in-process 逐位护栏（新增，强烈建议）

`tests_dsv41_draft_parity.cu` 已有 `DRAFT_HEAD_FOLD v1` 臂（:886，`dsv41_gemv_bf16_v1_mrows` vs 逐行 `dsv41_gemv_bf16` 的 in-process 双跑比对）。**本次新增两个臂**（同一模式）：

- `HEAD_ACT_F32VEC`：同一 (w, x, m, n, k) 跑 `DSV41_HEAD_ACT_F32VEC=0` 与 `=1` 两次，`to_bits` 逐位比 ⇒ 期望 `maxdiff=0.000e+00`；
- `WO_A_CPASYNC`：同一 (a, a_scale, w, w_scale, g, rows) 跑 `DSV41_WO_A_CPASYNC=0/1` 两次，同上。

> ⚠️ 现成测试二进制不含这两臂（本次禁止 GPU，未改测试文件以避免与 peer 抢 `tests_*`）。**落到 GPU 时请以 tester 的版本优先**（协作规则）。

### 4.4 弃用条件（写清楚，免得下次又「armed 空转」）

- ① 若 per-launch 时间不降或 mean-k 有任何 delta ⇒ 关闭 gate，留存 `[head-act-f32vec]` 的 nsys 行作为「L2 已足够」的证据；
- ② 若时间不降 ⇒ 关闭 gate（预期本臂是小票面，不值得占门禁位）。

---

## 5. 风险与坦白（工部的判断，非方案结论）

1. **① 的收益上界受权重流支配**：AR_SAFE 的 1.36ms/发 ≈ 1.323GB 权重 / 973GB/s，即**该 kernel 基本是权重 DRAM 带宽绑定**；激活侧 1.94GB 是 **L2 常驻**（工作集 120KB），时间占比可能只有 10–15%。而 staging 的代价是每 pass 10 个 barrier（≈15% issue 开销）。⇒ ① **可能是 wash 甚至小负**，这也是它必须默认 OFF 的原因。真正想再拿这块肉，方向是把**权重**流再摊薄（tensor-core/持久化），不是激活读法。
2. **② 的票面本来就小**（16× 指令削减，但 staging 在 55.7µs 里是小头）；它的价值是「同一 kernel 内权重/激活 staging 规则**对称**」，消掉一处 16× 不对称。
3. 两项都**未动 Rust 侧、未动 ABI、未加新符号依赖**（新 kernel 仅本 TU 内模板实例，launcher 只在本 TU 内 dispatch）⇒ 陈旧 `.so` 与新旧二进制互不干扰。

---

## 实测粘贴（compile-only）

_（见交付回复 §编译状态）_
