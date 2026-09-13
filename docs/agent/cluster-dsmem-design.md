# ⑤b — cluster (CGA) + DSMEM 权重广播：mrows 摊薄的硬件正解

> 交付物：**⑤b**（`docs/agent/tensorcore-proj-design.md` §5.2 认领的备选路线三选一）。
> 上一级：`mrows-mpar-design.md` §1.4「第三条路：M 真并行 + 权重只 stage 一次」。
> 姊妹路线：⑤a L2 广播（`docs/agent/mrows-l2-bcast-design.md`，peer `l2-broadcast-amort` 在实施）。
> 日期：2026-09-13。**本文件是预研 + 设计，不含实测**；⑤b **不实施**，等 ⑤a 的实测裁决。
> 状态：compile-only 已过（CUDA 13.2 / sm_103a，见 §6）。

---

## §0 一句话

`gemm_fp8_mrows_kernel<M>` 把 M 藏在**一个 warp 的寄存器**里；MPAR 把 M 摊成**块内 warp 轴**但败在
**prologue 被 `grid` 次复制**（`mrows-mpar-design.md` §2.3 / audit §10.9）。⑤b 把 M 放到 **thread-block
cluster 的 block-rank 轴**上，用硬件 **DSMEM（distributed shared memory）** 让同 cluster 的 M 个 block
**读同一片 smem**——权重 HBM 流量 1×、smem staging 1 次、M 真并行、**逐位**。这是唯一同时满足三条硬约束的
**硬件**原语路线（MPAR 用 warp 布局去近似它，cluster 直接给）。

**它不新增任何基建到生产路径**：本仓库此前**没有任何 cluster 代码**（`grep -rn` 无 `__cluster_dims__` /
`map_shared_rank`）。⑤b 是一个**新范式**，因此只做设计 + 编译验证，作为 ⑤a 的保险路线。

---

## §1 映射设计（kernel 结构）

### 1.1 问题的结构（为什么 cluster 的轴必须是「激活行」）

权重 `w[row][:]` **只依赖输出行 `row`，不依赖激活行**。M 个激活行共享同一行权重——这是 mrows 摊销的
全部来源。所以要让一个 cluster 的多个 block **读同一片权重 smem**，它们的 `row` 必须相同、激活行不同：

> **cluster 的 block-rank == 激活行轴 `g`。** cluster 内 M 个 block 服务**同一组输出行**、**M 个不同激活行**。

这正面回答了任务里的抉择：

- ❌ 「每 cluster 6 block，每 block 1 行」——若「行」指**输出行**，6 个 block 是 6 个**不同**输出行 ⇒ 读的是
  **不同权重**，DSMEM 共享**没有对象**。（这是 fold_r 的排布，不是 ⑤b。）
- ✅ 「每 block 1 行」应读作「每 block 一个 **(输出行组, 激活行)** 对」；rank 轴 = 激活行。

### 1.2 选定布局（Layout A：rank 0 全量 stage）

```
grid  : M * ceil(n / rpb)  blocks     ← 1D；blockIdx.x = row-group * M + rank
block : rpb warps（rpb*32 线程）      ← warp w 负责输出行 row0 + w
rank  : g = blockIdx.x % M            ← 本 block 服务的激活行
row0  : rid * rpb,  rid = blockIdx.x / M
```

- **rank 0** 把 `rpb` 行权重（`rpb*k` 字节）`cp.async16` stage 到**自己的 smem**；
- **所有 rank** 用 `cluster.map_shared_rank(s_w, 0)` 拿到 rank 0 的 smem 句柄，**读同一片字节**；
- `cluster.sync()` 发布（rank 0 先 retire 自己的 `cp.async` 再进 barrier，见 §3.2）；
- LUT **每 block 本地建**（256 f32，纯函数；见 §3.3）。

**与 MPAR 的唯一结构差异**：MPAR 一个 block = `rpb*M` 个 warp（M 在 warp 轴），⑤b 一个 block = `rpb` 个
warp（M 在 cluster 轴）。**这一个小差异就是 ⑤b 的全部价值来源**，见 §1.3。

### 1.3 为什么这个差异能解 MPAR 的死因（**核心论证**）

MPAR 的 `rpb` 上限被 **block 线程数**顶死：block = `rpb*M` warps ≤ 1024 线程 ⇒

```
MPAR :  rpb_max = floor(1024 / (32*M)) = floor(32 / M)      M=6 → rpb_max = 5
⑤b   :  rpb_max = floor(1024 / 32)      = 32                M=6 → rpb_max = 32
```

prologue 的**复制份数 = 承载 prologue 的 block 数 = ceil(n / rpb)**（①⑤a/MPAR 的 `grid`）。
于是同样的权重字节（`ceil(n/rpb)*rpb*k ≈ n*k = 1×`）被切进 **`rpb` 越大越少** 的 prologue 轮次里：

| 形状 | n | k | MPAR rpb_max | MPAR prologue 轮次 | ⑤b rpb | ⑤b prologue 轮次 | 轮次 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `wq_a` | 1280 | 5120 | 5 | 256 | 16 | 80 | 3.2× |
| `wkv` | 512 | 5120 | 5 | 103 | 16 | 32 | **3.2×** |
| `wq_b` | 4096 | 1280 | 5 | 820 | 16 | 256 | 3.2× |
| `wo_b` | 5120 | 1024 | 5 | 1024 | 32 | 160 | **6.4×** |
| `sh` | 288 | 5120 | 5 | 58 | 4 | 72 | —（见下） |

> **这就是 ⑤b 相对 MPAR 的增量**：`rpb` 不再被 `M` 除，prologue 的**DRAM 往返次数**随之降
> `rpb_⑤b / rpb_MPAR` 倍。MPAR 实测的败因**恰是** prologue 的暴露（audit：issue→build→wait 顺序导致
> 每 block 一次 DRAM 往返在关键路径上，`rpb=1` 时 grid 达 `n`）——⑤b 用 cluster 把「同一片 slab 的 M 次
> prologue」合并成 **1 次/ cluster**。

**⚠️ 诚实边界**：⑤b 不降低**权重字节**（MPAR 已是 1×），它降低的是 **prologue 轮次**（=暴露的 DRAM 延迟）
与 **LUT 复制份数**。`sh`（n=288）两档都低于 148 SM，收益取决于 §1.4 的覆盖规则，不保证。

### 1.4 覆盖（occupancy）规则 —— `rpb` 不能一味放大

⑤b 的 `grid` 是 `M * ceil(n/rpb)` **块**，但一个 **cluster 占 M 个 SM**（同 GPC 内 M 个 block 必须共驻）。
要喂满 148 SM，需要 `ceil(n/rpb) ≥ 148 / M ≈ 25` 个 cluster 并发：

```
rpb_cover = clamp( floor(n * M / SM_count), 1, 32 )        // SM_count ≈ 148
```

| 形状 | n | rpb_cover | clusters | ⑤b grid | 每 block smem(rpb*k) | smem 档 |
|---|---:|---:|---:|---:|---:|---|
| `wq_b` | 4096 | 32(cap) | 128 | 768 | 40 KB | 5 block/SM |
| `wo_b` | 5120 | 32(cap) | 160 | 960 | 32 KB | 6 block/SM |
| `wq_a` | 1280 | 32(cap) | 40 | 240 | 160 KB | **1 block/SM ⚠️** |
| `wkv` | 512 | 20 | 26 | 156 | 100 KB | 2 block/SM |
| `sh` | 288 | 11 | 27 | 162 | 55 KB | 4 block/SM |

**大 `rpb` 撞 smem 天花板**：`rpb_max=32` 在 `k=5120` 的 `wq_a` 上要 `160 KB/block`，逼近 B300 的
~227 KB/SM 上限（⇒ 1 block/SM，占用塌）。所以 `rpb` 的真实上限是 **`min(32, smem_ceiling / k)`**，
`k=5120` 时约 40 行余量、`k=1024` 时可取满 32。**这是 ⑤b 的第一号调参旋钮。**

### 1.5 Layout B（分布式切片）—— 避免 uniform smem 浪费

**关键约束（CUDA 语义）**：dynamic smem 大小是 **launch 级 uniform**——一个 kernel 启动里**所有 block
拿到同样大小的 dynamic smem**。Layout A 里只有 rank 0 用到 `rpb*k`，但 rank 1..M-1 也被**强制分配**同样
大小 ⇒ 有效 smem 浪费 `(M-1)/M`。

**Layout B**：把 slab 按**行**切成 M 片，rank r stage 第 r 片到**自己的** smem：

```
rows_per_rank = ceil(rpb / M)
rank r  : stage rows [r*rows_per_rank, (r+1)*rows_per_rank)  到 s_w[0 .. rows_per_rank*k)
所有 rank: 读 rows_per_rank 片，每片经 map_shared_rank(s_w_base_of[src], src)
          warp 的 rr 落在 src = rr / rows_per_rank
```

- 每 block smem = `ceil(rpb/M)*k`（Layout A 的 **1/M**）；
- staging **在 cluster 内并行**（M 个 block 同时 cp.async，而非 rank 0 独扛 `rpb*k`）；
- 远端读比例相同（每条权重字节被 M 个 rank 各读一次，其中 1/M 本地、`(M-1)/M` 远端）。

**代价**：每 warp 需对「本行所属的源 rank」做一次 `map_shared_rank`（M 个基址表，可 hoist 到循环外）。
**建议**：首版用 **Layout A**（简单、易验证逐位），`rpb` 取小避 smem 天花板；smem 成为瓶颈时再上 B。

---

## §2 与逐位等价（bit-identity）的兼容论证

⑤b **只改权重的读路径**（本地 smem → 远端 smem），**不改** K walk、操作数、累加链、归约树。
`mrows-mpar-design.md` 的 C1–C6 逐条照搬，逐位性由「读的 byte 同源同值」保证：

| 约束 | ⑤b 是否保持 | 论证 |
|---|---|---|
| **C1** 同 K walk | ✅ | `kb` 升 0..nb_k-1，`j = kb*32 + lane`，**一字未动** |
| **C2** 同操作数同 byte | ✅ | `rs[j]` 是 `w[row*k + j]` 经 cp.async **纯拷贝**进的 smem；`map_shared_rank(s_w,0)` **解引用同一物理 smem 行**。拷贝的宽度/位置不可观测（C2 原文），远端读与本地读**返回同一 byte** |
| **C3** 同归约 | ✅ | `shfl_xor` 树 `off=16,8,4,2,1` **逐字不动**，每 (warp,row) 一次 |
| **C4/C5** 无跨行重组 / 无 K-split | ✅ | 无 `acc[M]`，无跨 rank 求和；每个 `(row, act)` 由**恰好一个 warp** 算 |
| **C6** 单链 `acc += av*wv` | ✅ | `#pragma unroll 32` 源码形态不动；`av`/`wv` 表达式不动 |
| LUT | ✅ | 本地建，256 项来自**同一个** `e4m3_to_f`（逐位与旧 build 相同） |

> **一句话**：逐位安全的依据是「**换地址不换值**」。远端 smem 与本地 smem 是同一份物理字节的两个
> 视图；`mapa` 只是把「读哪个 SM 的 smem」编进地址，**不产生任何算术**。因此 ⑤b 的输出与 MPAR 与
> legacy 与 m=1 **四者逐位相同**。

**唯一的硬约束**：`cluster.sync()` 之前，rank 0 的 `cp.async` 必须 **retire**（`cp_wait_all` +
`__syncthreads`），否则远端读可能读到**未落地的 slab**——这是**正确性**要求，不是数值要求。见 §3.2。

---

## §3 DSMEM / cluster API 用法（本次 compile-only 已钉死）

### 3.1 API 清单

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template <int M>
__global__ void __cluster_dims__(M, 1, 1)          // ① 编译期 cluster 维度
cluster_mrows_kernel(/* 同 MPAR 的 ABI */) {
    extern __shared__ uint8_t smem[];              // ② 每 block 的 smem（launch 级 uniform 大小）
    cg::cluster_group cluster = cg::this_cluster();
    const unsigned rank = cluster.block_rank();    // ③ == 激活行 g

    uint8_t* s_w = smem;
    float*   s_lut = reinterpret_cast<float*>(s_w + (size_t)rpb * k);

    if (rank == 0) {                               // ④ 只 rank 0 stage 权重 slab
        /* cp.async16 整片 rpb*k；然后 */
        __pipeline_commit(); __pipeline_wait_prior(0);
        __syncthreads();                           //    retire，再发布
    }
    cluster.sync();                                // ⑤ 发布 rank 0 的 smem 给全 cluster

    uint8_t* s_w_remote = (uint8_t*)cluster.map_shared_rank(s_w, 0);   // ⑥ DSMEM 句柄
    /* consume：rs = s_w_remote + warp*k；表达式与 MPAR 逐字相同 */

    cluster.barrier_arrive();                      // ⑦ 分裂 barrier（可 arrive 早 / wait 晚）
    cluster.barrier_wait();
}
```

### 3.2 cluster 语义要点（正确性红线）

1. **动态 smem 是 launch 级 uniform**（§1.5）——不能「rank 0 要 160KB、其余要 1KB」。这决定了布局选型。
2. **`cp.async` 必须在 `cluster.sync()` 前 retire**：`cluster.sync()` 只保证「所有 block 到达 barrier」，
   它**不会**替你等 `cp.async` 落地。rank 0 必须 `cp_wait_all()` + `__syncthreads()` 后再进 barrier。
   （这是 ⑤b 版本的「issue→cover→wait→barrier」重排，对应 MPAR 的 prologue 修复。）
3. **`__cluster_dims__` 要求 grid 是 cluster 维度的整数倍**——`grid = M * ceil(n/rpb)` **按构造**满足。
4. **cluster 的 block 必须同 GPC 共驻**，占用受 `cudaOccupancyMaxActiveClusters` 约束；cluster ≤ 8 可移植，
   > 8 需 `cudaFuncAttributeNonPortableClusterSizeAllowed`（M=6 落在可移植区间，无需该属性）。

### 3.3 为什么 LUT 本地建（而不复用 rank 0 的）

LUT 是**纯函数**（`lut[b] = e4m3_to_f(b)`），与 map_shared_rank 无关。若全 cluster 读 rank 0 的 LUT，
则**每一次 per-element decode**（`s_lut[rs[j]]`，每 warp `nb_k` 次）都变成**远端 load**——把本该廉价的
查表放大成 cluster 网络流量。**本地建**（256 项 / block，一次性的几百条 store）**逐位相同**且把
decode 留在 LDS 快速路径。`tensorcore-proj-design.md` §5.2 写的「rank 0 建 LUT」是**LUT 复制份数**最小化，
不是**读延迟**最优——两者在 LUT 这个尺度上本地建完胜。**采本地建。**

### 3.4 另一种启动面：runtime cluster 属性

`__cluster_dims__` 把 cluster 维度烧进每个实例化（M ∈ 1..8 各一份，和 MPAR 的 per-M 模板一致，可接受）。
若想让**同一个实例化**在 launch 时决定 cluster 维度（M 还是扫参旋钮时有用）：

```cuda
cudaLaunchConfig_t cfg = {};
cudaLaunchAttribute a[1];
a[0].id = cudaLaunchAttributeClusterDimension;
a[0].val.clusterDim = {M, 1, 1};
cfg.gridDim = M * grid_per_cluster; cfg.blockDim = rpb*32; cfg.attrs = a; cfg.numAttrs = 1;
cudaLaunchKernelEx(&cfg, cluster_mrows_kernel<M>, /* args */);
```

两条路 **落到同一条 PTX 路径**（本次两条都编译通过，见 §6）。**建议**：生产选 `__cluster_dims__`（少一次
driver 往返 / 与现有 per-M 模板同构）。

---

## §4 延迟 / 带宽账（**预研估算，待 micro-bench 判读**）

### 4.1 单次访问延迟（数量级）

| 路径 | 延迟（cycles，数量级） | 备注 |
|---|---:|---|
| LDS（本地 smem） | ~20–30 | 基线 |
| **LDSM / DSMEM（`mapa` 远端 smem）** | **~30–60** | 跨 SM 走 cluster 网络；任务给的 ~30 vs ~20 是**乐观下界**，文档按 **+10–30 cyc** 计 |
| L2 hit | ~200–270 | Blackwell |
| DRAM | ~400–600 | |

**关键**：consume 循环每次迭代 1 条 `s_w` 读 + 1 条 `s_lut` 读 + 1 条 activation LDG，`#pragma unroll 32`
提供 ILP。**远端 LDS 的额外 ~10–30 cyc 在 32× 展开下大概率被覆盖**（吞吐受限而非延迟受限）——
**这是 ⑤b 必须用 micro-bench 判读的第一件事**（若远端 LDS 的吞吐只有本地的一半，则这条路的读路径本身就是
瓶颈）。

### 4.2 每 launch 的字节流量（M=6，`rpb` 覆盖后）

| 路线 | 权重来源 | 每 launch 读字节 | 后端 |
|---|---|---|---|
| legacy / MPAR | 本地 smem 每 warp | `n*M*k`（全本地 LDS） | LDS |
| **⑤a** L2 广播 | L1/L2 直读 | `n*M*k`（全 L2 请求） | L2 (~8 TB/s) |
| **⑤b** DSMEM | rank 0 本地 + 其余远端 | 本地 `n*k` + **远端 `(M-1)*n*k`** | cluster 网络 |

- **⑤b 的远端字节 = `(M-1)*n*k`**（M-1 个非 staging rank 各读整片 `n*k`）。相对 ⑤a 的 `M*n*k` 只降到
  `(M-1)/M` **份数**，但 ⑤b 的**后端是 cluster 网络（smem 级，~20 TB/s 量级）而非 L2（~8 TB/s）**——
  **带宽比是 ⑤b 的护城河**（任务给的 ~20 TB/s vs ~8 TB/s）。
- **⑤b 相对 MPAR 的代价**：MPAR 的 `n*M*k` 全是**本地** LDS，⑤b 有 `(M-1)/M` 变**远端**（每 byte 慢
  ~10–30 cyc）。**⑤b 押的是「prologue 轮次 ↓3–6× 抵得上远端读开销」**——MPAR 输在 prologue（延迟暴露），
  不在读路径，这个押注是**有结构依据**的。

### 4.3 净账（一句话）

> **⑤b = MPAR 的读路径（改远端）+ prologue ÷(rpb 比)**。若 prologue 是 MPAR 的瓶颈（audit 指向此），
> 且远端 LDS 吞吐/延迟在 unroll 32 下被覆盖，则 ⑤b **转正**；否则退回 ⑤a。

---

## §5 ⑤a vs ⑤b 决策矩阵（**待 ⑤a 实测填表**）

| 维度 | ⑤a L2 广播（在途） | ⑤b cluster DSMEM |
|---|---|---|
| 权重 DRAM 字节 | 1× | 1× |
| 权重读请求量 | `M×`（全走 L1/L2） | 本地 1× + 远端 `(M-1)×`（走 cluster 网络） |
| 读后端带宽 | L2 ~8 TB/s（共享） | smem/cluster 网络 ~20 TB/s 量级 |
| prologue | **无**（零 smem、零 cp.async、零 LUT） | 1 次/cluster（`ceil(n/rpb)` 次，比 MPAR ↓3–6×） |
| smem | **0** | `rpb*k`（uniform，可能撞 227KB 天花板） |
| 逐位 | ✅ | ✅ |
| 代码/基建复杂度 | 低（删 smem slot，改直读） | **高**（新范式：cluster 布局 + barrier + DSMEM + 占用调优） |
| 块数 | `M*ceil(n/nwarps)` | `M*ceil(n/rpb)`（同量级） |
| 主要风险 | 块调度开销；L2 请求 `M×` 打满 | DSMEM 延迟/带宽；cluster 占用；uniform smem 浪费 |
| **选它的条件** | L2 命中率够 + 延迟可接受（⑤a 实测**不输**） | **⑤a 实测输**（L2 请求 `M×` 打满或延迟暴露）**且** DSMEM 带宽优势成立 |

### 5.1 判据（decision rule）

```
if  ⑤a(实测, 所有形状) >= 0            → 用 ⑤a（零新范式，收工）
elif MPAR 的病灶被 NCU 证实 = prologue (DRAM 延迟暴露)
     and ⑤b micro-bench: 远端 LDS 吞吐 >= 本地 LDS 的 ~60%
                                        → 上 ⑤b（Layout A，小 rpb）
else                                     → 回 mma 主路 or 维持 legacy
```

### 5.2 与主路（§1–§3 mma swapAB）的关系

⑤a/⑤b **不与 mma 竞争**：两者**逐位安全**，是 mma「数值双门禁不过」时的**退路**，也可作独立臂并行验证
（`tensorcore-proj-design.md` §5.3 原文）。⑤b 的**唯一**独立价值 = 它同时满足三约束，是 mrows 摊薄的
**硬件正解**；⑤a 是它的**软件近似**（L2 代替 DSMEM）。

---

## §6 compile-only 验证结果（本次已跑，CUDA 13.2 / sm_103a）

**载体**：`kernels/cuda/tests_cluster_dsmem.cu`（新文件，**不进 build.sh**；带 host harness，`-c` 下由
`FERRITE_CLUSTER_COMPILE_ONLY` 之外的分支隔离，纯编译干净）。**不碰** `dsv41_kernels.cu`（peer ⑤a 在改）。

| 检查 | 命令 | 结果 |
|---|---|---|
| compile-only | `nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 -c` | ✅ EXIT=0 |
| full link（含 host harness：`cudaLaunchKernelEx` + `cudaOccupancyMaxActiveClusters`） | 同上 `-o` 出二进制 | ✅ EXIT=0，1.1 MB 二进制 |
| **PTX 指令落地** | `-ptx` 后 `grep` | ✅ 命中 10 处 |

**PTX 证据**（确认真生成了硬件原语，而非静默降级）：

```
mov.u32   %r1, %cluster_ctarank;              ← cluster rank 读取
barrier.cluster.arrive;                       ← cluster barrier
barrier.cluster.wait;
mapa.shared::cluster.u32  %r5, %r4, 0;        ← DSMEM 地址映射（rank 0 的 smem）
```

**结论**：
1. `__cluster_dims__(M,1,1)` **带模板参数**在 CUDA 13.2/sm_103a 上编译通过（任务 §1 的语法问号解除）；
2. `map_shared_rank` / `cluster.sync` / `barrier_arrive/wait` 全部落地为 `mapa` + `barrier.cluster`；
3. **runtime cluster 属性**（`cudaLaunchAttributeClusterDimension`）与 `cudaOccupancyMaxActiveClusters`
   均链接通过（两条启动面都可选）。
4. **未做 GPU/e2e**（本任务禁止）——占用/延迟/带宽账待 ⑤a 之后另起 micro-bench。

---

## §7 明确不实施（本任务边界）

- **不实施**：⑤b 不进 `build.sh`，不加 runtime gate，不改 `dsv41_kernels.cu`（⑤a peer 在改）。
- **不抢跑**：⑤b 的增量价值**完全取决于 ⑤a 的实测结论**——⑤a 够用时 ⑤b 无独立价值（§5）。
- **本文件 + `tests_cluster_dsmem.cu`** 是 ⑤b 的全部产出：**设计 + 编译验证**，作为 ⑤a 输时的保险路线。

## §8 风险 / 待定（进 GPU 轮前必须回答）

1. **远端 LDS 吞吐**（头号）：DSMEM 的稳态带宽是本地 smem 的几成？>60% 则 ⑤b 成立。（micro-bench：一个
   循环读 `map_shared_rank` 的 smem，对比本地 LDS，量 cycles/byte。）
2. **cluster 占用**：`cudaOccupancyMaxActiveClusters` 在 `rpb*k=100–160KB` 时给出几个？若 < `148/M`，
   cluster 串行化，⑤b 直接输。
3. **uniform smem 浪费**：Layout A 的 `(M-1)/M` 浪费能否被 `rpb` 调小吸收；否则转 Layout B（§1.5）。
4. **barrier 开销**：`cluster.sync()` 的跨 SM 同步相对 `__syncthreads()` 的额外成本（每 cluster 一次，
   应可忽略，但需确认）。
5. **GPC 拓扑**：148 SM / GPC 数决定 M=6 cluster 能否共驻；若 GPC 只 4 SM，M=6 直接不可行（待查）。

---

## 附：与三份上游文档的锚点

| 上游 | 位置 | ⑤b 在此的落点 |
|---|---|---|
| `tensorcore-proj-design.md` | §5.2 | 本文件 = 该节的完整实现设计 |
| `tensorcore-proj-design.md` | §5.3 | 排序表「⚠️ ⑤a 结论出来后再说」→ §5 决策矩阵 |
| `mrows-mpar-design.md` | §1.4 | 「第三条路」= 本文件 §1（cluster 的 block-rank 轴） |
| `mrows-mpar-design.md` | §2.3 / §4 | 1.59× 指令天花板仍在（⑤b 不消）；prologue 修复 = §1.3 |
| `mrows-l2-bcast-design.md`（⑤a） | — | §5 决策矩阵的「实测」列待其填 |
