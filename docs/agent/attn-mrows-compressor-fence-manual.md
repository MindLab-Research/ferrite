# ATTN_MROWS 的 compressor-snapshot fence 修复（`DSV41_ATTN_MROWS` 的最后一条 decline）

> 病灶：`attn_mrows_decline("this block's compressor commits per row and left no
> device snapshot: an m-row launch would bound every row with the block-final
> counter")`。本文记录**根因（file:line + counter 形态）**、**修复 diff 与其逐位
> 论证**、**编译结果**，以及 **GPU 验证手册**。

---

## 1. 根因

### 1.1 fence 的判定链（修复前）

| 位置 | 内容 |
|---|---|
| `crates/ferrite-models/src/dsv41/chain_dev.rs:11653` | `let mrows_attn =` — `DSV41_ATTN_MROWS` 的 gate（`attn_mrows()`，默认 OFF） |
| `chain_dev.rs:11601` | `let mrows_own_owner = if is_comp_src { mrows_compress } else { owner != layer && mrows_compress_ok(owner, ...) }` — 「本层（或其 owner）的 compressor 是否被 hoist」 |
| `chain_dev.rs:11578`（修复前） | `let mrows_compress = if is_comp_src && comp_proj { self.compress_rows_fused(layer, m, pos_base, &mut comp_len_rows)? } else { false }` — COMPRESSOR-MROWS 的 hoist 入口 |
| `chain_dev.rs:13045` | `mrows_compress_ok` — hoist 的静态前置条件 |
| **`chain_dev.rs:11666`（修复前）** | **fence：`else if is_comp_src && comp_proj && !mrows_own_owner { attn_mrows_decline(...); false }`** |

即：**本层是 compressor 的 source、projection 跑了、但 hoist 没接住** → decline。

### 1.2 为什么 hoist 会「没接住」——触发面就是 `ratio <= 1` 的 kv source

`mrows_compress_ok`（`chain_dev.rs:13045`）的完整判定：

```
dev_ok   = self.dev.supports_compressor_fused_mrows()      // .so 有符号
shape_ok = m >= 2 && m <= VERIFY_ROWS && pos_base > 0
ratio    = cfg.compress_ratio(layer)
w_ok     = comp_wkv.is_some() && comp_norm.is_some()
ok       = dev_ok && shape_ok && ratio > 1 && w_ok          // :13058
```

生产配置（`crates/ferrite-models/configs/dsv41_reference_flat.json`）：

```
compress_ratios   = [0,0, 2 x18, 1 x20, 0,0,0]
kv_source_layers  = [2, 8, 14, 20]
  → layer 2  ratio 2   (kv source)  → hoist 接住 ✓
  → layer 8  ratio 2   (kv source)  → hoist 接住 ✓
  → layer 14 ratio 2   (kv source)  → hoist 接住 ✓
  → layer 20 ratio 1   (kv source)  → ratio > 1 为假 → hoist 拒绝 ✗
```

`ratio == 1` 的层没有 `comp_wgate`（`compress_proj_rows` 的 `:12894`「ratio == 1:
no gate, so the pooled value IS the projection」），但它的 **pool + commit 半边照样
每行跑一次**：`compress_proj_rows` 仍返回 `Ok(true)`（只要 `comp_wkv`/`comp_norm`
在），所以 `comp_proj == true`；`compress_row` 的规则 `(pos + r + 1) % 1 == 0`
**每一行都成立**，于是 `clen` 每行 +1。

→ `is_comp_src && comp_proj && !mrows_own_owner` 恒真，**layer 20 每步都 decline**
（`pos_base <= 0` 的首步同理，那时 hoist 也被 `shape_ok` 拒）。

### 1.3 counter 形态（为什么「没有 device 快照」是准确的描述）

| 对象 | 形态 |
|---|---|
| **live counter** | `s.clen` = **`[n_layers]` i32**（`chain_dev.rs:280` / 分配 `:4127`）——**每层一个标量**，`compress_commit_kernel` 用 `*clen = len + 1` 原地推进 |
| **per-row 快照** | `s.clen_rows_r` = **`[n_layers, VERIFY_ROWS]` i32**（`chain_dev.rs:476` / 分配 `:4225`）——只有 hoisted 的 `compressor_fused_mrows_kernel` 才写它（`dsv41_kernels.cu:3555`：`if (clen_rows != nullptr && tid == 0) clen_rows[r] = *clen;`） |
| **读侧** | m-row 的 sparse attn 读 `const int cl = clen_rows ? clen_rows[mm] : *clen;`（`dsv41_kernels.cu:982` / `:1082` / `:1185` / `:1551` / `:2124`） |

所以：per-row 路径推进的是**每层一个的标量**，中间值**没有第二份存放点**；块级
launch 在循环之后发出，那时 `*clen` 已是 **block-final**。fence 的判决是正确的
（m-row launch 会把 row `0..m-2` 用它们自己的未来做上界 = audit defect #1），
**缺的不是判决，是快照本身**。

---

## 2. 修复

### 2.1 改动（**只动 `chain_dev.rs`，kernel 一行未改**）

| 位置 | 改动 |
|---|---|
| `chain_dev.rs:11710` | 新增 `let attn_own_snapshot = mrows_attn && is_comp_src && comp_proj && !mrows_compress;` — 「本层的 compressor 走 per-row 路径，而块级 launch 会被发出」 |
| `chain_dev.rs:11727` | `attn_clen = if mrows_own_owner \|\| attn_own_snapshot { clen_rows_r + owner*VERIFY_ROWS } else { null }` |
| `chain_dev.rs:11808` | per-row commit 之后，若 `attn_own_snapshot`：`memcpy_d2d(clen_rows_r + owner*VERIFY_ROWS + r, clen + owner, 4)` |
| `chain_dev.rs:11666`（修复前） | **fence 整条删除**；gate 现在只剩 `world > 1` 无 `_rp` 符号、与 `pos + m - 1 >= win`（ring 翻转）两条 |

**关键选择——为什么用 `memcpy_d2d` 而不是 host 值 upload**：快照必须由 **device
读 device** 产生。host 侧确实知道每行的值（`compress_rows_fused:13152` 的同一
确定性规则），但 H2D 上传会把值 **烘进 graph**——正是本仓库反复记录的
"captured but not updated" 一族 bug（`dsv41_glue.cu:906` 的 ring slot 注释、
`chain_dev.rs:6160` 的 16-byte D2D 注释）。`memcpy_d2d`（`devrt.rs:1303`）走
`cudaMemcpyAsync`、在**本 stream** 上，仓库把「可被 graph 捕获」明确写在它的注释里
（`devrt.rs:1304`、`chain_dev.rs:6160`），所以 replay 时读到的是**当次的** counter。

### 2.2 逐位论证

修复**不改任何 kernel、不改任何计算**，只把「行 r 的上界」这件事从
「不存在的快照」变成「存在的快照」：

1. **snapshot 的值 = per-row 读到的值**。per-row 路径里 row r 的读者（indexer /
   `comp_placeholder`）在 `*clen` 上读到的是 **row r 的 commit 之后、row r+1 的
   commit 之前** 的计数（同一 stream、按序发出）。新增的 D2D 拷贝紧接
   `compress_row(layer, r, pos_base)` 之后发出，读的是**同一个 device 标量**，
   按 stream 序在前述读者与块级 launch 之间 → **值恒等**。
   非提交行（`(pos+1) % ratio != 0`）上 counter 不变，hoisted kernel 同样为这种行
   记录「不变的值」（`dsv41_kernels.cu:3555` 在 `if (out_rows_val > 0)` 之外），两侧
   逐行一致。
2. **kernel 侧无分支差异**。kernel 只做 `cl = clen_rows ? clen_rows[mm] : *clen`；
   现在两条臂都交出**同值**的 `clen_rows[mm]`，`n`/`topk`/`idx_stride` 与循环体内
   的求和顺序一字不变 → 输出逐位一致（行独立性是既有设计契约）。
3. **per-row 读者未受影响**。`mrows_own_owner` 仍是「hoist 是否接住」，所以
   `clen_row`（`chain_dev.rs:11824`）在 per-row 臂上仍为 `None` → per-row 的
   indexer / `comp_placeholder` / `sparse_attn` 继续读 live counter，
   **这一臂的 launch 序列与修复前逐条相同**。
4. **修复前该臂根本不会发出块级 launch**（fence 拦掉了），所以不存在「行为回归」：
   新解锁的块级 launch 用的是**正确的** per-row 上界。
5. **代价**：每个 `attn_own_snapshot` 的层每步多 `m` 次 4-byte async D2D
   （生产配置下就是 layer 20，`m = 6` → 6 次）。不改 kernel、不加 launch 类型，
   对 kernel 时间不构成影响。

### 2.3 与 `mrows_compress_ok` 的关系（**故意不动**）

`ratio > 1` 是 hoist 的**语义**前置（`ratio == 1` 无 gate/无 pooling 状态，
`compressor_fused_mrows_kernel` 未在该 shape 上验证），本次**不去放宽它**——
放宽会改变 layer 20 的 kernel 选择（per-row pool+commit 对 → 一个 fused launch），
越出「只解锁、不换 kernel」的红线。修法 (a)（per-row 侧补快照）对
**所有** hoist 接不住的成因（gate OFF / 缺符号 / `pos_base <= 0` / `ratio <= 1` /
缺权重）一视同仁，是覆盖面更大的那一种。

---

## 3. 编译结果

| 项 | 命令 | 结果 |
|---|---|---|
| Rust 类型检查 | `cargo check --workspace --all-targets` | ✅ **EXIT=0** |
| crate 单测 | `cargo test -p ferrite-models --lib` | ✅ **92 passed / 0 failed / 2 ignored** |
| 远端 nvcc（compile-only, `ubuntu@43.202.208.136`, `sm_103a`） | `nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 -c` on `dsv41_kernels.cu` + `dsv41_glue.cu` | ✅ **rc=0 / errors=0**（仅既有的 unused-variable 警告） |

未改动任何 `.cu`，故 nvcc 结果与 HEAD 相同——上面两条是确认「工作树仍干净可编」。

> 本地 `cargo test -p ferrite-dsv41` / `--workspace --lib` 的失败是**环境性**的
> （`dlopen(libcudart.so) failed`、`ferrite-exec` 缺 `cuda`/`nccl` feature），与本次
> 改动无关；`-p ferrite-exec` 在 HEAD 上同样失败（已用 baseline 对照确认）。

---

## 4. GPU 验证手册（`DSV41_ATTN_MROWS=1` vs `0`）

前提：TP8（`world = 8`），同栈 `DSV41_COMPRESSOR_MROWS=1`。e2e 一律后台模式。

### 4.1 门禁一：decline 行消失

```bash
grep -c "DSV41_ATTN_MROWS=1 declined" serve.log      # 稳态不再增长
grep    "DSV41_ATTN_MROWS=1 declined" serve.log | sort -u
```

判据：

- **不**应再出现 `"this block's compressor commits per row and left no device
  snapshot"` ——该字符串在本次改动后**已从源码中删除**（`grep -c` 应为 0）。
- 仍可能出现且**属设计**：`"pos + m - 1 >= window: the ring has turned over"`
  （`POS_BASE + 6 - 1 >= 128` 之后，即长上下文的绝大部分步）。
- 若出现 `"world > 1 and the .so has no dsv41_sparse_attn_rp"` → `.so` 没重编，
  先重建再测。
- 交叉确认 layer 20 的 hoist 收据：`grep "COMPRESSOR_MROWS" serve.log` 应有一行
  `DECLINED: ratio <= 1 (no compression state) (layer 20 ...)`——这正是被 fence 挡住
  的那一层，`ATTN_MROWS` 的 decline 消失而 `COMPRESSOR_MROWS` 的 decline 保留，
  是本次修法正确的指纹。

### 4.2 门禁二：步时（第一指标）

```bash
DSV41_ATTN_MROWS=1 bash scripts/batched_400_v2.sh
DSV41_ATTN_MROWS=0 bash scripts/batched_400_v2.sh
```

看 `[dspark] steps=` 的 `verify` 分解。**期望方向**：`verify` 相对 `=0` 臂继续下降
（layer 20 的 6 条 per-row sparse attn → 1 条块级 launch），`step_ms` 同步。
注意本票只解锁**一层**（layer 20）的块级 launch，量级远小于 ATTN_MROWS 全臂的
−2.98ms；不要把整臂的差值都归给本改动。若差值 ≈ 0 且 4.1 的 decline 确实消失，
说明该层的块级 launch 本身已不是瓶颈（与「mrows amortization 为零」的既有结论
一致），此时以 4.1 + 4.3 为准，不要用吞吐反推。

### 4.3 门禁三：数值（逐位）

1. **同栈 A/B**：`DSV41_ATTN_MROWS=1` vs `=0`（TP8、`COMPRESSOR_MROWS=1` 不变），
   逐行 logits/argmax 必须一致。行独立性 ⇒ 允许跨行乱序，但**同一行必须逐位**。
2. **快照本身的对照（推荐加做）**：临时让 layer 20 也走 hoist 的读侧口径——
   即在 `DSV41_COMPRESSOR_MROWS=1` 且**把 `compress_ratios[20]` 视作 2 不成立**
   的前提下，唯一可对照的是「per-row 读 live counter」（`=0` 臂）。两臂的
   `clen_rows_r[20][r]` 应与 per-row 臂逐行读到的 live counter 相同：
   `DSV41_HC_DEBUG`-类一次性 note 或 `[dsv41]` 调试打印（如需可在下一轮临时加）。
   **判据**：`clen_rows_r[20][0..m]` 严格非降、末值 == 块末 `*clen`。
3. **kv/sink 快照**：两臂共用同一 ring 镜像（gate 的 ring 未翻转前置），attention
   输入本身相同，差异只应来自 launch 结构。
4. **graph 臂**：本改动引入 `memcpy_d2d`（`cudaMemcpyAsync` D2D）。仓库已把该原语
   标注为 graph-capturable（`chain_dev.rs:6160`），但**首次上机请确认**
   `DSV41_VERIFY_GRAPH=1` 仍能 capture（日志里不应出现
   `[verify_graph] ... capture FAILED`）。若驱动拒绝该节点，`step_rows_sync` 的
   capture 失败路径会 latch 到 DIRECT launch（`chain_dev.rs:6698-6730`），
   属**优雅降级**而非报错——但那就说明本修法需要换成「commit kernel 侧写快照」。

### 4.4 回滚

- 关 `DSV41_ATTN_MROWS=0` → 回到 per-row（默认）。
- 关 `DSV41_COMPRESSOR_MROWS=0` → 所有 compress-source 层都走本次新增的 per-row
  快照路径（成本线性增长到每层 `m` 次 4-byte D2D），**语义仍正确**——这也是本修法
  覆盖 gate 关闭场景的证据。
