# grouped / permuted routing 布局设计（工部 · 工程实现）

> 目标：给 expert-CENTRIC 的 tcgen05 e4m3 专家 GEMM
> （`dsv41_expert_gemm_e4m3_ext`）补上它缺的前置条件——**grouped（permuted）路由布局**。
> 当前 `moe_rows` 的路由是逐 `(row, slot)` 的 `route_idx_r[m][topk]`，而该 kernel 一次
> launch 只认 **一个 expert + 一整块 dense 行**，直接喂逐 (row,slot) 表 = **静默错值**。
>
> 本文档 = 布局契约 + kernel 清单 + 等价性论证 + **尚未闭环的部分（必读 §6）**。
>
> 工部 · 2026-09-12 · 只写代码 + `cargo check`（本机无 GPU，未跑任何 GPU 验证）

---

## 0. 一句话结论（先看这个）

**布局 + 搬运三件套已完整落地并接线（env-gated、默认 OFF）；但 grouped GEMM 本身还没到能发的形状，
所以 `DSV41_EXPERT_GROUPED=1` 目前是「数值 no-op」——老实说出来了（一次性告警）。**

原因不是实现没做完，而是**形状不匹配**，见 §6。要真收 -6.8ms，缺的是
**group-indexed masked kernel**（DeepGEMM 的 `m_grouped_gemm_nt_masked` 那种
"grid 铺 m-tile + group 表查 expert + 越界行 mask"），不是本文档交付的布局。

---

## 1. 问题陈述

| 侧 | 事实 | 出处 |
|---|---|---|
| e4x kernel | expert-CENTRIC：`grid = (n_total/kNTile, rows/kMTile)`，B 侧全靠 `base + ids[slot] * stride` 派生 ⇒ **一次 launch 一个 expert 覆盖所有行**；`out` 是 dense `[rows, n_total]` | `kernels/cuda/dsv41_experts_mxf4.cu:5392`（kernel）、`:5652`（launcher）、`:5673`（grid） |
| e4x 形状契约 | `k % 64 == 0`、**`rows % 128 == 0`**、`n_total % 64 == 0`，且 a/b/b_hi 16B 对齐 | `e4x_launch_gemm` 的 `if (k % kAtomK != 0 \|\| rows % kMTile != 0 \|\| n_total % kNTile != 0)`；`kMTile=128`、`kNTile=64`、`kAtomK=64`（`:105-107`） |
| 路由 | `route_idx_r[m][topk]`，逐 (row,slot) 给 expert id；`route_topk` 原生有 `rows` 维 | `chain_dev.rs`（`moe_rows` 的 gate+route 段） |
| 现状 | `moe_rows` 里 `e4x_tile = false`，e4x 臂拒绝下发、回落到 SIMT batched 路径（**不错值，只是没有 tcgen05**） | `chain_dev.rs` 的 `e4x_armed/e4x_shape/e4x_tile` 三条件 + `tcgen05_e4m3_ext_skipped_note` |

**缺口**：把 `(row, slot)` 表变成 **expert-major 的连续行块**，并让激活 / 输出可以按这个排列搬进搬出。
这就是本文档交付的东西。

---

## 2. 布局契约（唯一权威定义）

设在某层某 step：`m` = 本 block 行数、`topk` = 每行激活专家数、`n_routed` = 路由专家总数。
记 **assignment** = 一个 `(row, slot)` 对，扁平下标 **`i = r * topk + t`**，总数 **`n_assign = m * topk`**。

### 2.1 grouped 顺序 `g`

`g ∈ [0, n_assign)`，排序键 = **(expert 升序, 同一 expert 内 `i` 升序)**。

两个关键性质：

1. **expert e 的赋值占据连续区间 `[starts[e], starts[e] + counts[e])`** ⇒ 一个 expert 的操作数
   就是一块 contiguus 的 dense 行块（正是 expert-centric kernel 要的）。
2. **顺序是 `route_idx_r` 的纯函数**：由单线程串行扫描生成，**没有 atomic、不依赖 block 调度** ⇒
   同一张路由表重放（CUDA graph replay / 重 capture）得到**逐位相同的排列**，不引入不确定性。

### 2.2 输出数组

| 数组 | 形状 | 含义 |
|---|---|---|
| `counts` | `[n_routed]` i32 | 每个 expert 分到的赋值数（= 行数） |
| `starts` | `[n_routed + 1]` i32 | `counts` 的**排他前缀和**；`starts[e]` = expert e 的第一个 grouped 位置，`starts[n_routed]` = 可路由赋值总数（表合法时 == `n_assign`） |
| `expert_rows` | `[n_routed * m_cap]` i32 | expert e 的**第 j 个 grouped 行的源 row r**，位于 `e * m_cap + j`（`m_cap` 固定 stride，方便"整块当 `e*m_cap` 取"） |
| `expert_slots` | `[n_routed * m_cap]` i32 | 同上，源 **slot t** |
| `perm_map` | `[n_assign]` i32 | **原始 → grouped**：`perm_map[i] = g`；`ids[i]` 越界时 `-1` |
| `gather_src` | `[n_assign]` i32 | **grouped → 原始**：`gather_src[g] = i`；无赋值落到该位置时 `-1` |
| `active` / `n_active` | `[n_routed]` i32 / `[1]` i32 | **非空 expert 的升序紧凑表**，让 host 只迭代 ~`topk*m` 个活 expert，而不是 384 个 |

**为什么两个方向的映射都要**：gather 走 grouped→原始，scatter 走 原始→grouped；在热 kernel 里临时反查
等于每个元素一次 O(n_routed) 扫描。

**为什么 `perm_map[i] < 0` 被显式写成 poison**：越界 id（路由表损坏）不能留下"上一 step 的旧值"——
`gather_src = -1` 让 gather **写 0 行**，`perm_map = -1` 让 scatter **写 0 行**，于是损坏条目最多贡献 0，
不可能把另一个 step 的激活搬进活 slot。

### 2.3 `m_cap` 的选取

`m_cap` 是 `expert_rows`/`expert_slots` 的 per-expert stride。真实上界是 `m`（`route_topk` 每选中一个
expert 就把它从选择分数里标 `-INFINITY` ⇒ 一行内 topk 个 id 互不相同 ⇒ `counts[e] <= m`）。
本实现取**最坏情况 `VERIFY_ROWS * topk`**（384 expert 时 55 KB/数组），这样即使路由表被破坏成重复 id，
kernel 也只是写满而不会截断/越界；配 `j < m_cap` 的 belt-and-braces 保护。

容量由**单一来源**决定，避免"分配处 / 启动处两份常量漂移"：

```rust
grp_n_experts(cfg)  = max(n_routed_experts, dspark_n_routed_experts, 1)   // = 384
grp_topk_max(cfg)   = max(n_activated_experts, dspark_n_activated_experts, 1) // = 6
grp_m_cap(cfg)      = VERIFY_ROWS * grp_topk_max(cfg)                     // = 36
```

`moe_route_grouped` 在下发前**再验一次**运行时形状（`n_routed`/`topk`/`m`/`row_bytes`）落在这组容量内，
超出就**拒绝并告警**（不是静默越界写）。

---

## 3. kernel 清单

三个新符号，全在 `kernels/cuda/dsv41_route.cu`（匿名 namespace 内的 kernel + 文件末尾 `extern "C"` 入口）：

| 符号 | 作用 | 形状 / 契约 |
|---|---|---|
| `dsv41_route_group` | 建布局：histogram → 前缀和 → 串行确定赋值 → 两个方向的映射 + active 表 | `<1 block, 256 thr, 2*n_experts*4 B smem>`；串行 `n_assign` 次（36） |
| `dsv41_route_gather_rows` | 激活 gather：`[m][row_bytes]` → `[n_assign][row_bytes]`，源行 = `gather_src[g] / topk` | `<n_assign blocks, 256 thr>`；q 与 sc 一次 launch 都搬；`row_bytes % 16 == 0` 走 `uint4`，否则逐字节 |
| `dsv41_route_scatter_rows` | 输出 scatter：`[n_assign][n]` → `[m*topk][n]`，由 `perm_map` 驱动（每个目标行从已知源行拷） | `<n_assign blocks, 256 thr>`；`n % 4 == 0` 走 `float4` |
| `dsv41_route_group_smem` | 给 host 量 smem 的纯函数（避免 Rust 侧复制公式） | `2 * n_experts * 4` |

⚠️ **命名冲突（已规避）**：`dsv41_gather_rows` / `dsv41_scatter_rows` **已被 attention 的 KV gather/scatter 占用**
（`dsv41_glue.cu:1157`，签名完全不同）。新入口一律带 `route_` 中缀 —— 同名 `extern "C"` 是链接错误，
而复用一个符号会把错误的 ABI 交给调用方。

### 3.1 gather 的一个关键细节

`gather_src[g] = i = r*topk + t`，**但源行是 `r = i / topk`，不是 `i`**：
激活是按 **row** 量化/存储的（`xq4_r[r*dim]`），一行的 topk 个 slot 读的是**同一份字节**。
所以 grouped buffer 有 `m * topk` 行而源只有 `m` 行 —— 这份"复制"正是让**一个 expert 的操作数连续**的代价，
也是 dense launch 能成立的原因。

---

## 4. 接线（host 侧）

| 位置 | 改动 |
|---|---|
| `device.rs` `struct Kernels` | `route_group` / `route_group_gather` / `route_group_scatter` 三个 **`ko!` 可选符号**（stale `.so` → `None` → 回落） |
| `device.rs` 包装层 | `route_group(...)` / `route_gather_rows(...)` / `route_scatter_rows(...)` 返回 `Result<bool>`（`false` = 符号缺失）；`supports_route_group()` = **三个符号齐备才 true**（只查一个 = 只会建表不能搬字节，会把"armed 却测旧路径"做成默认） |
| `chain_dev.rs` `struct Scratch` | `grp_counts/grp_starts/grp_rows/grp_slots/grp_perm/grp_src/grp_active/grp_nactive/grp_xq/grp_xsc`，**无条件分配**（~184 KB ~ 静态分配图优先于省字节） |
| `chain_dev.rs` env 门 | `expert_grouped()` ← `DSV41_EXPERT_GROUPED`，`starts_with('1')`（严格 `1` 前缀，默认 OFF，`OnceLock` 缓存一次 —— 每次 getenv 是 capture hazard） |
| `chain_dev.rs` | `moe_route_grouped(m, topk, n_routed, row_bytes)`：建表 + gather；`Ok(true)` = 布局在 `s.grp_*` 里已就绪 |
| `moe_rows` 调用点 | 放在 **`quant_fp8`/`quant_fp4` 之后、gate/up launch 之前** —— 这是"e4m3 字节已经在 `xq4_r`、且还没被任何后续 pass 覆盖"的唯一时点 |

`row_bytes` 由激活格式决定：e4m3 臂 = `dim`（一值一字节），fp4 臂 = `dim/2`；scales 恒为每 32 值一个 f32。

### 4.1 三种"decline"都要说话

`moe_route_grouped` 的契约是 **`Ok(false)` = 回落，永不 fail step**（和本仓库其它可选臂一致）。三种 decline
各有一条**一次性**告警，理由是本仓库第一号陷阱是「armed 的 gate 实际在测旧路径」：

1. `.so` 没有 `dsv41_route_*`（stale build）；
2. 运行时形状超出 `grp_*` 容量 / `row_bytes % 16 != 0`（防越界写）；
3. **布局建成了、但 per-expert dense GEMM 还没发**（= §6 的形状问题）。

---

## 5. 等价性论证（为什么这套搬运不改数值）

1. **gather / scatter 是纯数据搬运，零算术、零重量化**：gather 是 `uint4`/`float4`/逐字节拷贝，
   scatter 同理。`grp_xq + starts[e] * row_bytes` 起的那一块，**逐字节等于**逐 (row,slot) 路径原本要喂给
   expert `e` 的那些激活行。scatter 反向同理。
2. **布局本身不参与数值**：`counts`/`starts`/`perm_map`/`gather_src` 只决定"哪些行共享一次 launch"，
   不进任何乘加。
3. **GEMM 侧若换成 grouped 调用，逐行计算不变**：同一个 `e4m3_gemm_kernel`、同一条 K 序
   （`k0` 升序的 stage 循环 + stage 内 `atom` 升序 + scale fold 在 tmem 读出后一次乘）、同一 epilogue
   （`epi_mode` / `limit` / `row_weight` 逐字继承，见 `:5615-5639`）。**行与行之间不共享输出元素、
   不共享累加器、不共享 smem staging**（行只通过 `m_base` 平移基址）⇒ grouped 调用里第 x 行的结果 == 逐行
   调用里同一行的结果。
4. **可测的判据**：既然 1+2 是纯置换，可以直接比对 ——
   把 grouped 路径的 `grp_xq` 按 `perm_map` 逆搬回 `[m][topk][dim]`，应与 `xq4_r` 的逐 (row,slot) 展开
   **逐字节相同**（`memcmp`）。scatter 同构。这条不需要 GPU 上的数值容差。

---

## 6. ⚠️ 尚未闭环：为什么 grouped 布局本身不足以解锁 -6.8ms

**这是本文档最重要的部分，也是必须回指尚书省的选择。**

任务书里的形状是 `m×topk` 个 assignment、每 expert 一次 dense GEMM。但本引擎的实际形状是：

| 量 | 值 | 出处 |
|---|---|---|
| `m` | `<= VERIFY_ROWS = 6` | `chain_dev.rs:84`；`moe_rows(layer, ld, m)` 的唯一调用者 `layer_rows` 就在 verify 路径 |
| `topk` | 6 | `config.rs::production_shapes()`（`n_activated_experts == 6`） |
| `n_routed` | 384 | 同上 |
| `n_assign = m*topk` | **<= 36** | 推导 |
| `counts[e]`（真实路由，每行 topk 互异） | **1 ~ 3** | `counts[e] <= m = 6`，且 36 个赋值摊到 <= 36 个活 expert |
| e4x launcher 要求 | **`rows % 128 == 0`** | §1 |

于是 per-expert dense launch = 把 **1~3 行真实数据 pad 到 128 行**，128/1.5 ≈ **85× 的 tensor-core 空转**。
即使 tensor core 比 SIMT 快 10~20×，净账也是**负的**。所以：

- "布局 + gather/scatter" 是**必要**前置，但**不充分**；
- 真解是 **group-indexed masked kernel**：一次 launch，grid 铺满 `(n_tiles, total_m_tiles)`，
  每个 m-tile 用 group 表（`starts`/`counts`，或 DeepGEMM 的 `m_indices`）**查出自己的 expert**，
  并把 `row >= counts[group]` 的行 mask 掉 —— 这样 36 行真实数据只付 36 行的 MMA，且只付一次 launch。
- `starts`/`counts`/`grp_rows`/`grp_slots` 这几个数组**正是那个 kernel 需要的输入**，所以本次交付没有白做。

**给尚书省的决策点**：是否开一个"group-indexed masked e4x kernel"的任务？
（这是新 kernel，不是布局改造；`e4x_launch_gemm` 的 `rows % 128 == 0` 契约要不要放宽/新增一个 masked
launcher，属于方案决策，工部不自行改。）

---

## 7. 验证计划（GPU 侧，本次未跑）

| 步骤 | 判据 |
|---|---|
| A. 布局自洽 | 小 shape（如 `m=4, topk=3, n_routed=8`）host 侧复算：`Σcounts == n_assign`、`starts` 单调、`perm_map`/`gather_src` 互为逆 |
| B. 搬运逐位 | §5.4 的 `memcmp`：gather→`perm_map` 逆搬运 == 原始 `[m][topk][dim]` 展开，**0 字节差** |
| C. 幂等 | 同一路由表连跑两次 `route_group`，两次 `perm_map` **逐位相同**（证明无调度依赖） |
| D. 边界 | 人为把 `ids[i]` 置越界：`perm_map/gather_src` 该位置 = `-1`，gather/scatter 写 0，**不读到旧 step 数据** |
| E. 端到端 | `DSV41_EXPERT_GROUPED=1` vs `=0`：**文本逐字一致** + `DSV41_DIFF_EAGER=1` 的 `[diff]` 行一致（当前预期：完全一致，因为 GEMM 未换） |

⚠️ `.cu` 变了 ⇒ 远端**双产物重编**（`build.sh 103a` 后 `cargo build --release`）；`dsv41_route.cu` 在
`build.sh` 的 `SRCS` 里并计入 `CU_HASH`，所以 BUILD_ID 会变，`.so` 与二进制必须同源。

---

## 8. 改动清单

| 文件 | 改动 |
|---|---|
| `kernels/cuda/dsv41_route.cu` | +`route_group_kernel` / `route_group_gather_kernel` / `route_group_scatter_kernel`，+4 个 `extern "C"` 入口（`dsv41_route_group`、`dsv41_route_group_smem`、`dsv41_route_gather_rows`、`dsv41_route_scatter_rows`） |
| `crates/ferrite-models/src/dsv41/device.rs` | `Kernels` +3 可选符号；+4 个包装方法（`route_group` / `route_gather_rows` / `route_scatter_rows` / `supports_route_group`） |
| `crates/ferrite-models/src/dsv41/chain_dev.rs` | +`expert_grouped()` 门 + `expert_grouped_skipped_note()`；+`grp_*` 容量三函数；`Scratch` +10 缓冲区（含分配）；+`DevChain::moe_route_grouped`；`moe_rows` 调用点 |

`cargo check --workspace`：**EXIT=0**。
