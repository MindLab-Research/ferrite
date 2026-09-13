# W2-MROWS 的 TP8 支持（`DSV41_ATTN_MROWS` 在 `world > 1` 下生效）

> 病灶审计 #2 的「快赢臂最大项」：sparse attention 的 m=6 批量 launch 在 TP8 下被
> `attn_mrows_decline("world != 1: ...")` 一次性挡掉，票面 ~3-4ms（verify 28.17ms 的
> 主力摊薄项之一）。本文记录**实施完成后的 ABI / 索引 / 等价性 / fence 处理**。

## 0. 背景：为什么会 decline

| 项 | 值 |
|---|---|
| 调用形态 | `b=1, m=m(=6), grid=(b*m, h)`, `h = nlh = nh/world` |
| kernel 旧索引 | `q/out + ((row*h + hh)*d)` → 行距 `h*d = nlh*hd` |
| 调用方行布局 | `q_r`/`o_r` 的**真实行距是 `nh*hd`**（未切分的 head 轴）；per-row 调用传的是 `q_r + r*nh*hd` |
| 结论 | `world == 1` 时 `nlh == nh` 两者相等；`world > 1` 差 `world` 倍 → 必须显式 `row_pitch`，旧 ABI 没有 |

**关键区分（实施中核实）**：`q_r`/`o_r` 是**全宽** `[rows, nh, hd]`，行距 `nh*hd`；
而 `xq_r`/`xsc_r` 是本 rank 的**紧凑块** `[rows, nlh, hd]`，行距 `nlh*hd`。
所以 `row_pitch` 只作用于 q/out，**不能**作用于 fp8 发射（见 §2.2）。

## 1. ABI：加符号，不改 ABI（A1a 先例）

旧符号签名**一字未动**，新增两个 `extern "C"` 入口：

```
dsv41_sparse_attn_rp        (..., clen_rows, idx_stride, row_pitch, stream)
dsv41_sparse_attn_orope_rp  (..., clen_rows, idx_stride, row_step, row_pitch, stream)
```

实现方式是**共享 impl + 薄转发**（`dsv41_sparse_attn_impl` / `dsv41_sparse_attn_orope_impl`
带 `row_pitch`；旧符号转发 `0`）。因此：

- 只带旧 `.so` 的加载方：行为**逐位不变**（旧符号通往同一内核，`row_pitch=0`）。
- Rust 侧新增 `sparse_attn_rp` / `sparse_attn_orope_rp` 两个 `Option` 字段（`ko!`），
  另有 `Device::supports_sparse_attn_rp()` 能力查询；**缺失即 decline**，回落 per-row。
- Rust 包装器**保留原签名**（`sparse_attn` / `sparse_attn_orope` 薄转发到 `*_pitched`），
  所以 `dspark_dev.rs`、eager 臂、per-row 循环的调用点**一行未改**。

## 2. kernel 索引改动与逐位等价论证

### 2.1 改动

六个 kernel（`sparse_attn` / `sparse_attn_warp` / `sparse_attn_pf` / `sparse_attn_split`
/ `sparse_attn_merge` / `sparse_attn_orope`）各加尾参 `int row_pitch`，并在函数头计算：

```c
const size_t rp = row_pitch ? (size_t)row_pitch : (size_t)h * d;
```

所有 q/out 行寻址由 `((size_t)row * h + hh) * d` 改为：

```c
q   + (size_t)row * rp + (size_t)hh * d
out + (size_t)row * rp + (size_t)hh * d
```

（`row = bb*m + mm` 的展平形式在五个 kernel 内原样使用；merge 用 `row = blockIdx.x`。）

### 2.2 刻意**不改**的地方

| 位置 | 索引 | 原因 |
|---|---|---|
| `sparse_attn_orope_kernel` phase 3 | `xbase = ((bb*m+mm)*h + hh)*d` | xq/xsc 是紧凑块，行距 `nlh*hd = h*d`，per-row 调用传 `xq_r + r*nlh*hd`，本来就是对的 |
| `sparse_attn_merge_kernel` 发射段 | `xbase = ((row*h + hh)*d)` | 同上（key-split 默认臂：merge 承担 rope+fp8 尾段） |
| `sparse_attn_split_kernel` 输出 | `g_attn_part[row][hh][...]` | 按**行号**索引的全局 scratch，不是指针，与 pitch 无关 |
| `idxs` | `idxs + row * (idx_stride ? idx_stride : topk)` | 沿用 W2 既有的 `idx_stride = ist` 契约 |

### 2.3 逐位等价论证

- **`world == 1`**：调用方传 `row_pitch = 0` → `rp = (size_t)h*d = nlh*hd`。
  `(size_t)row * h * d + (size_t)hh * d` 与旧式 `((size_t)row*h + hh) * d` 是同一整数
  表达式（分配律，`size_t` 无溢出差异，`hh < h`、`row < b*m` 均为小量）→ **逐位一致**。
- **`world > 1`**：`rp = nh*hd`。row `mm` 的绝对地址 = `base + mm*nh*hd + hh*hd`，
  与 per-row 调用 `base + mm*nh*hd` 再让 kernel 算 `hh*hd` **完全相同**。
  per-row 路径是既有正确路径，故「每行与 per-row 老路径逐位」由构造保证
  （行独立性即设计契约：每行只读自己的 q、自己的 idxs 行、共享的 kv/sink 快照）。
- **fp8 发射**：两条路径的 xq/xsc 偏移本就相同（§2.2），不引入新差异。

## 3. COMPRESSOR-MROWS fence：**修掉，不再是 decline**

旧 fence 原文说：m-row attention 用 `clen_rows[mm]` 作每行上界，而这一臂是**从 host
mirror 手搓一个 host 数组再 `as_ptr()` 传进 device kernel** ——
「a separate, pre-existing defect of the ATTN_MROWS arm」；叠上 compressor hoist
（live counter 也是 block-final）就「no snapshot that a device kernel can use」。

**修正**：`clen_rows` 完全改走 **device 内存**，host 数组删除（`clen_rows[r] = ...` 的
staging 也一并去掉，行循环只剩 per-row launch）：

| 场景 | 传给 m-row launch 的 `clen_rows` | 等价于 per-row 路径读到的值 |
|---|---|---|
| `mrows_own_owner`（本层或其 owner 的 compressor 被 hoist） | `clen_rows_r + owner*VERIFY_ROWS`（**device**，hoist 内核写的「row r commit 之后」的计数） | per-row 循环传的 `clen_row = clen_rows_r + owner*VERIFY_ROWS + r`，同一个值 |
| 其余（本 block 不推进 counter：consumer 的 owner 早已跑完整个 block；或本层无 commit） | `nullptr` → kernel 读 `*clen`（= `clen_owner`，owner 的 live counter，block 内恒定） | per-row 调用传的 `clen_owner`，同一个值 |

**唯一保留的 decline**（比原来窄得多）：

```
is_comp_src && comp_proj && !mrows_own_owner
```

即「本层 compressor 走了 per-row 路径（commit 逐行推进 live counter），而 device 上没有
中间值的快照」。此时 m-row launch 只能读到一个 block-final 计数，会给 row `0..m-2`
它们自己的未来 → **拒绝，回落 per-row**。

触发条件（实测口径）：`DSV41_COMPRESSOR_MROWS` gate 关闭、`.so` 无
`dsv41_compressor_fused_mrows`、`pos_base <= 0`（**prefill 后第一步**）、`m < 2`、
`ratio <= 1`、或该层没有 `comp_wkv`/`comp_norm`。63.8 生产栈开着 `COMPRESSOR_MROWS=1`，
稳态（`pos_base > 0`、`ratio > 1`）下 hoist 生效 → **不再 decline**。

## 4. 验收状态

- `cargo check --workspace --all-targets` → **EXIT=0**。
- 远端 nvcc compile-only（`ubuntu@43.202.208.136`, `arch=compute_103a`）→ 见任务报告。
- 本机 `cargo test -p ferrite-dsv41` 的 `ar_hcpost_parity` 失败是**环境性**的
  （`dlopen(libcudart.so) failed`，本机无 CUDA），与本改动无关。

## 5. GPU 验证手册（A/B：`DSV41_ATTN_MROWS=1` vs `0`）

前提：TP8（`world = 8`），`COMPRESSOR_MROWS=1` 同栈。**e2e 一律后台模式**。

### 5.1 先看 decline 是否消失

```bash
# serve 启动后抓 stderr
grep -c "DSV41_ATTN_MROWS=1 declined" serve.log          # 期望：稳态不再增长
grep    "DSV41_ATTN_MROWS=1 declined" serve.log | sort -u
```

判据：稳态步（`pos_base + m - 1 < window`，即未进长上下文）应**没有新的 decline 行**；
若仍出现，只可能是 §3 收窄后的那一条（per-row commit 无快照），或 `pos + m - 1 >= window`
（ring 已翻转，设计上不覆盖）。**若出现「world > 1 and the .so has no
`dsv41_sparse_attn_rp`」→ `.so` 没重编**，先重建再测。

### 5.2 步时/吞吐（第一指标）

```bash
# 非 nsys 轮，看真实分解（禁止吞吐反推）
DSV41_ATTN_MROWS=1 bash scripts/batched_400_v2.sh
# 对照臂
DSV41_ATTN_MROWS=0 bash scripts/batched_400_v2.sh
```

看 `[dspark] steps=` 的 `verify` 分解：期望 `verify` **下降 ~3-4ms**（票面），
`step_ms` 同步下降；`acc` 只看是否被数值问题打坏（不受本改动影响）。

### 5.3 数值红线（逐位）

1. **`world == 1` 逐位**：同一 seed/输入，`DSV41_ATTN_MROWS=1` vs `0`，
   前 61 行（H 区计数口径）**逐字节一致**。
2. **`world > 1` 每行与 per-row 逐位**：临时把 `attn_row_pitch` 强制为 0
   （即走旧 `h*d` 路径）会显式 decline——不要用它做对照；正确对照是
   `DSV41_ATTN_MROWS=0`（per-row 老路径）与 `=1` 在**同一 TP8 栈**下比较：
   每个 verify 行的 logits/argmax 必须一致（行独立性 ⇒ 允许跨行乱序，
   但同一行的数值必须逐位）。
3. **kv/sink 快照**：两臂共用同一 ring 镜像（gate 的 ring 未翻转前置），
   故 attention 输入本身相同，差异只应来自 launch 结构。

### 5.4 回滚

`DSV41_ATTN_MROWS=0` 即回到 per-row（默认）。`.so` 缺 `dsv41_sparse_attn_rp`
时 `world > 1` 自动 decline，与改动前行为一致。
