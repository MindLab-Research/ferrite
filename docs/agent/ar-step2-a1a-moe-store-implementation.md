# AR L4/L5 第二步的实施：A1a —— store 折进 producer（MoE 侧）

> 工部 · 2026-09-12 · 实施 + `cargo check`（本机无 GPU / 无 nvcc，**未执行任何 GPU 命令**）。
> 设计：`docs/agent/ar-l4l5-optimization-design.md` §4-A1a / §5（执行顺序第 3 项），
> `docs/agent/stage-b-execution.md` §1 第 2 项。
> 上一步：`docs/agent/ar-step1-a2b-a0-implementation.md`（A2b + A0）。

---

## 1. Step 2 是哪一项

设计 §5 的执行顺序：

```
1) A2b 超时语义        ← Step 1 已实施
2) A0  探针            ← Step 1 已实施
3) A1a store 折进 producer   ← 本步
4) A1b PDL
5) B  臂选择
```

**A1a = 每轮的 2 发（store + publish/reduce）→ 1 发**：把 AR 的 staging 拷贝从独立的
`p2p_ar_store_v5_kernel` 挪进 payload **最后写者**的 epilogue，AR 只剩 publish+reduce。

**现状（读码复核）**：**attn 侧已在树内**（`dsv41_gemm_fp8_gemv_kernel` 的 AR 参数 +
`Device::gemm_fp8_mx_ar` + `Collective::all_reduce_inplace_pubred_only` +
`ferrite_p2p_ar_pubred_v5`，gate `chain_dev.rs` 的 `ar_store_fuse()` = `DSV41_AR_STORE_FUSE`，默认 OFF）。
**MoE 侧是本步的缺口**——即本文件实施的内容。

---

## 2. 改动（6 个文件、gated OFF）

### 2.1 `kernels/cuda/ferrite_kernels.cu`

1. **AR store 助手**（`p2p_ar_v5_slot_base` / `p2p_ar_v5_store_elem`）：与
   `p2p_ar_store_v5_kernel` 的寻址表达式逐字一致（同一 slot 公式
   `((e&1)*world + my_rank)*stride + i`）。长注释写清载体的三条硬要求——
   **last writer** / **全覆盖且不越界**（`slot+stride` 是下一个 rank 的 payload）/ **同流且在 publish 之前（无 PDL）**。
2. **`add_kernel` 增 5 个可选尾参**（默认 `nullptr/0`）+ 新入口 **`ferrite_add_store`**：
   开/关跑**同一份编译产物**，`x + y` 的位不可能因两条路径而异。
3. **`ferrite_p2p_ar_pubred_v5_moe`**：与 `ferrite_p2p_ar_pubred_v5` 逐字同体，只把 A0 探针的
   site 标成 `AR5_SITE_MOE`（用新符号而不是加参数，避免改 `device.rs` 按名绑定的 ABI）。
4. **`ferrite_p2p_ar_pubred_v5_hcpost`**：`ferrite_p2p_ar_v5_hcpost` 的**去掉 store** 版
   （publish + reduce + hc-post 折叠）。**同时覆盖 `_hcpost_add`（ADD_EPI）**：折叠的残差只改
   "publish 什么"，store 被载体接管后这就是载体的事，hc-post 半边消费的是 `out`。

### 2.2 `kernels/cuda/dsv41_experts_mxf4.cu`

**`moe_down_reduce_kernel` 增 5 个可选尾参 + 新入口 `dsv41_moe_down_reduce_st`**：
store 拷贝加在 `acc` **定稿之后**，所以"升序 slot 求和"这条数值契约（注释里的
`((0+p0)+p1)+…`）一字未动；`dsv41_moe_down_reduce` 不传参 ⇒ 两模式同源。

### 2.3 `crates/ferrite-models/src/dsv41/device.rs`

- 新符号装载 + `Kernels` 字段：`add_inplace_ar`(`ferrite_add_store`)、
  `moe_down_reduce_ar`(`dsv41_moe_down_reduce_st`)、`p2p_ar_pubred_v5_moe`、`p2p_ar_pubred_v5_hcpost`。
- 包装：`add_inplace_ar(...) -> Result<bool>`、`moe_down_reduce_ar(...) -> Result<bool>`
  （**`.so` 缺符号 ⇒ `Ok(false)`，一个字都没发**，调用者回退到原路径 ⇒ store 绝不会静默丢失）、
  `p2p_ar_pubred_v5_moe(...)`、`p2p_ar_pubred_v5_hcpost(...) -> Result<bool>`。
- `supports_ar_store_carriers()` / `supports_ar_pubred_moe()`。

### 2.4 `crates/ferrite-models/src/dsv41/tp.rs`

- `all_reduce_inplace_pubred_only_moe(buf, len)`：同 attn 版，site 标 MoE。
- `all_reduce_inplace_hcpost_pubred_only(buf, len, res, post, comb, hc_n, hc_h) -> Result<bool>`：
  同 `all_reduce_inplace_hcpost` 的形状门（`hc_n∈[1,8]`、`hc_h%4==0`、`len/4==hc_h`），只少 store。

### 2.5 `crates/ferrite-models/src/dsv41/chain_dev.rs`

- **`ar_store_fuse_moe(dev, comm)`**（新门）：`DSV41_AR_STORE_FUSE`（默认 OFF，与 attn 侧同一个开关）
  **且** 符号齐备 **且** AR v5。门 OFF 时短路在第一个 `OnceLock` 读，其余判断不执行。
- **`moe()` → `Result<bool>`**（返回值 = 本 rank 是否接管了 store）：
  - `shared_here = sh_w.is_some() && sh_w2_ok`（与共享专家块的进入条件逐字对应）；
  - 走 `!shared_here`（本 rank 不算共享专家）⇒ **routed batched down-reduce 是 `s.o` 的最后写者**，
    用 `moe_down_reduce_ar` 携带；顺序专家路（逐 slot 累加）**不携带**（最后写者是最后一个 slot 的 launch）。
  - `shared_here` 且**独立的**共享专家 merge 真的跑（`!add_epi_ready()`，含 `dual` join 那一支）⇒
    **`add_inplace` 是最后写者**，用 `add_inplace_ar` 携带。
  - **ADD_EPI / A5(`moe_epi_add`) 下不携带**：合并后的字节只存在于 AR（或 w2 GEMV epilogue）内部，
    没有可挂载的 producer ⇒ 该 rank 保持今天的两发 AR。这是本步唯一"少做"的地方，是刻意的保守。
- **`moe_reduce(layer, carried)`**：`carried` 时只跑 publish+reduce
  （先试 hc-post 折叠版 `ar_hc_post_fold_after_store`，decline 则
  `all_reduce_inplace_pubred_only_moe`），带 `debug_assert!(add_in.is_none())`；
  否则**原路径一字未动**。
- **`ar_hc_post_fold_after_store`**：`ar_hc_post_fold` 的 store-less 孪生，保留同一
  `hc_tail_join()`（tail split 的副作用顺序必须一致）。
- **调用点**：`let ar_carry = ar_store_fuse_moe(...); let moe_carried = if 图 { 回放 ⇒ ar_carry … }`。
  决策是进程常量，所以捕获进图的载体在回放时依然成立；store 在段内、publish+reduce 在段外，
  与今天"独立 store kernel 在段外"的流序完全一致。

---

## 3. 正确性论证（三段，与 attn 侧同一套）

1. **只搬不求和**：载体写进 peer 槽的字节 == 它写进本地 `out` 的字节（同一个寄存器值）。
2. **同 kernel**：`add_kernel` / `moe_down_reduce_kernel` 只是加了默认尾参，开/关是**同一份编译产物**；
   载体把拷贝放在 `acc`/`v` 定稿之后，求和链与 slot 顺序未动。
3. **顺序不变**：store 只是**更早**发生（从自己的 launch 挪进 producer），仍在本 rank 上一次
   publish 之后 ⇒ 奇偶双缓冲的跨 rank 证明不变。epoch 仍是**内核运行期**读，图回放安全。

---

## 4. gate 与零开销

- `DSV41_AR_STORE_FUSE` 默认 **OFF** ⇒ `ar_carry` 恒 false ⇒ 两个载体一次都不发，
  `moe_reduce` 走原路径。热路径只多一个 bool 判断/层（门 OFF 时短路在不碰 `comm`/`uses_v5`）。
- 内核侧：载体入口只在显式传 store 参数时走 store 分支；默认调用点传 `nullptr`，
  多一个分支 + 2~5 个寄存器参数（与 Step 1 的 A0 同量级）。

## 5. 验收（本机 = 只有 `cargo check`）

- `cargo check --workspace` **EXIT=0**。
- `cargo test --workspace`：`ferrite-models` 有 **1 个既有失败**
  （`dsv41::weights::tests::shard_factor_is_consistent_with_local_shape`，72 vs 96）；
  已用 `git stash` 在**改动前**复现 ⇒ **与本次改动无关**。`ferrite-kernel` 的 GPU 测试
  在本机**链接失败**（无 CUDA 运行时），同为环境问题。
- `.cu` 只做人工审校（括号配平、实参顺序逐一对形参、无跨 TU 残留）。

## 6. 有 GPU 时按序做（**未做**）

1. `cd kernels/cuda && bash build.sh 103a`（本机无 nvcc）——双产物纪律。
2. **位级两道**：①新 `.so` 且 `DSV41_AR_STORE_FUSE=0` vs 改动前的 `.so` ⇒ token 流必须逐字节一致
   （测 `add_kernel`/`moe_down_reduce_kernel` 的**重编译漂移**）；②新 `.so` 上
   `DSV41_AR_STORE_FUSE=1` vs `=0` ⇒ 必须一致。
3. **节点数**：`p2p_ar_store_v5_kernel` 的 launch 计数下降（MoE site：不跑共享专家的 rank 每层 −1）。
4. **性能**：步时对比（设计口径 −1~2μs/轮）；单点回退用 `DSV41_AR_STORE_FUSE=0`。
5. **捕获**：`DSV41_GRAPH_STEP` + `DSV41_GRAPH_MOE` 下多轮连续请求（载体落在捕获段内）。
6. **A0 探针**：`[ar-probe]` 里 site=0（MoE）与 site=1（ATTN）应各占总轮数的一半。

## 7. 本步**未**做

- **engram AR**（2 节点/步）与 **attn 侧的复测**（树内已就位，属 Step 1 之前的遗留项）。
- **ADD_EPI / A5 覆盖的 rank**：需要给 `add_kernel`/w2 GEMV 载体加 `bias`（作为 store 的 ADD_EPI 等价位），
  收益是 rank 0 也省一发；风险与工作量都更大，留给后续。
- **A1b PDL**、**B 臂选择**：按设计 §5 属第 4、5 步。
