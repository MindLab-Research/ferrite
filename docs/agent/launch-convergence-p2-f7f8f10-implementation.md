# launch-convergence P2: F7 / F10 / F8 实现记录

> 工部 · 2026-09-13 · **纯代码，未执行任何 GPU 命令**。验证方式：`cargo check --workspace` EXIT=0
> + `cargo test -p ferrite-models --lib`（97 passed）。
> 三个 gate 全部 **DEFAULT OFF**，旧路径逐 launch 不变；未重编 `.so`，因此三条臂在现网 `.so` 上自动落回旧路径
> （每条臂都额外要求一个**本 build 新引入的符号**，见下 §4）。

---

## 0. 与任务书不一致的三处（须尚书省裁决）

任务书的三条修法里，只有 F7 与代码现状完全吻合，另两条与代码冲突，实现时做了等价/更强的处理：

1. **F10「改为 1 次块级 D2D（m × clen_size 一趟）」在本代码上是不可实现的。**
   被替换的是 `chain_dev.rs::attention_rows` 的 `attn_own_snapshot` 臂：每行一次 `memcpy_d2d`，
   **源是同一个标量**（`s.clen[layer]` 的**活计数器**），而计数器在行与行之间**会推进**
   （`ratio` 行提交一个 group），所以 6 行的快照是 6 个**不同**的值。一次 `memcpy_d2d(dst, src, m*4)`
   要求源侧有 m 个不同的值，源侧只有一个 —— 无论 `spitch=0` 的 2D 形式还是 1D 形式都构造不出来。
   ⇒ 实现改为**值等价且更强**的形态：快照由 **commit 内核自身**写入
   （`dsv41_compress_commit_rows`，多一个 `clen_row_out` 尾参），
   6 发 D2D → **0 发**（比"1 发"还少一发），且写入线程就是 bump 计数器的那一个线程、读回同一地址，
   值与"紧跟其后发 D2D"逐行相同（含**不提交 group 的行**：计数不变，快照即该不变值）。
   为此 kernel 的早退 `if (*out_rows <= 0) return;` 改为 `if (... > 0) { ... }`（本 kernel 无任何
   `__syncthreads`，跳过分支的线程不会与 barrier 相关 —— 原注释就是这么写的）。

2. **F8「改为 b=1, m=rows（骨架 grid(m,b) 已支持）」的"已支持"只对 grid 成立。**
   本仓自己已在 `indexer_rows_m` 的 header（`WHY THE SELECT HALF IS NOT FUSED HERE`）里写明三个 blocker，
   实测确认：① 内核用 `n_pos = *lens` 覆盖上限，即 **row 0 的计数**，而 verify 各行的计数随 `r` **递增**，
   行 1..m-1 会被静默夹到 row 0；② 内核写 `out[row * cols + i]`，`cols = min(topk, cl)` 是**运行期值**，
   而 `idxs_r` 的行距是固定 `offset + index_topk`；③ 放置点（见 §3）。
   ⇒ 内核改为：上限取 **各行 `lens` 的最大值**（m=1 时恒等）、本行 `cl = lens[mm]`、
   **`cols` 按本行**（`min(topk, cl)`，否则 rank 语义会漂）、输出行距 `pitch = out_stride`（0 = 旧的 `cols`），
   stage B 的 chunk 扫描上界由 `n_pos` 改 `cl`（m=1 恒等）。全部改动在 `b*m == 1` 下**逐位等价**。

3. **F7「3 调用点」：本 HEAD 只有 2 个真需要（第 3 个不存在）。**
   两处 `for r in 0..m { quant_fp8(...) }` 的源行距 ≠ `cols`：
   `o_r`（行距 `nh*hd` vs `cols = nlh*hd`）与 `wo_r`（行距 `ol_total` vs `cols = ol_local`）。
   计划里可能算作第 3 处的 `quant_rows` 助手，其 4 个活调用点（`xn_r`/`qr_r`）的源行距**恒等于 `cols`**
   （`dim` / `ql`），在那里折叠只会把 1 发换成 1 发、零收益，故未改（不引入死代码）。

---

## 1. 三个 fix 的改动与 gate

| fix | gate（DEFAULT OFF） | 改动文件 | 语义 |
|---|---|---|---|
| F7 | `DSV41_F7_QUANT_STRIDE=1` | `dsv41_kernels.cu`（`quant_kernel` + 2 个入口）、`device.rs`、`chain_dev.rs` ×2 调用点 | 源行距显式化，`m` 发 → **1 发**（×2 处/层） |
| F10 | `DSV41_F10_CLEN_BLOCK=1` | `dsv41_glue.cu`（`compress_commit_kernel` + 新入口）、`device.rs`、`chain_dev.rs` | 快照由 commit 内核写，**6 发 → 0 发**/层 |
| F8 | `DSV41_F8_TOPK_ROWS=1` | `dsv41_kernels.cu`（stage A ×2 + `indexer_topk_kernel` + 新入口）、`device.rs`、`chain_dev.rs` | 选择行化，**6 发 → 1 发**/层 |

### F7 细节

* `quant_kernel<FP4>` 增 `int src_stride`：`ss = src_stride > 0 ? src_stride : cols`，
  `src = x + r*ss + b*block`。**目标行距恒为 `cols`**（各调用点目标都是紧凑的），
  算术（amax / scale / byte）一字未动 ⇒ `src_stride == cols` 时逐位等于旧内核。
* `dsv41_quant_fp8` 追加**尾参** `int src_stride = 0`（放在 stream **之前**，与本文件 `dsv41_gemm_fp8_mx`
  的尾部默认参风格一致），`src_stride != 0 && src_stride < cols` 直接 `cudaErrorInvalidValue`
  （短行距只会来自笔误，静默重叠读是宁可拒绝的）。
* 调用点 1（wo_b 输入）：`rows = m, cols = ol_local, src_stride = ol_total`；
  调用点 2（wo_a 输入）：`rows = m, cols = nlh*hd, src_stride = nh*hd`。
  调用点 2 只在**整块融合状态一致**时折叠：`all_o_fused`（旧循环本来是空转 ⇒ 折叠也必须空转）或
  `!some_o_fused`（旧循环是整块 ⇒ 1 发替代）；**混合块保持逐行**（融合行已由 fused kernel 写过同字节，
  覆盖无害，但保持旧路径更保守）。
* 期望收益：两处各 `m-1` 发（m=6 ⇒ 5 发）× 40 层 ≈ **−0.28ms**（任务书口径）。

### F10 细节

* `compress_commit_kernel(..., int* clen_row_out = nullptr)`；`clen_row_out` 由
  `threadIdx.x == 0` 在（可能的）`*clen = len + 1` 之后写 `*clen_row_out = *clen`。
* 新入口 `dsv41_compress_commit_rows(latent, cos, sin, ring, out_rows, clen, clen_row_out, hd,
  rope_dim, half, window, ratio, s)`；旧入口一字未改（不传该参）。
* `chain_dev.rs::compress_row` 增 `clen_row_out` 形参（唯一调用点已更新）；
  `attention_rows` 里仅当 `F10 && supports_compress_commit_rows() && attn_own_snapshot` 时传非空，
  并跳过对应的 `memcpy_d2d`。
* 期望收益：`attn_own_snapshot` 配置下 **6 发 → 0 发**/层 ≈ **−0.34ms**（任务书口径；本实现 ≤ 它）。

### F8 细节

* stage A（v1/v2）与 stage B 的 `n_pos` 上限：`lens != nullptr` 时取 `max(lens[0..m))`（m=1 恒等）。
* `indexer_topk_kernel`：`cl = lens[mm]`（夹在上限内）、`cols = min(topk, cl)`（本行）、
  `pitch = out_stride > 0 ? out_stride : cols`、写循环用 `pitch`、chunk 扫描上界用 `cl`。
* 新入口 `dsv41_indexer_topk_rows(..., offset, out_stride, ...)`；`uses_candidates != 0` 与
  `0 < out_stride < topk` 直接拒绝（掩码行距用的是发射上限，per-row `lens` 下语义会错；
  本调用点恒传 `nullptr/false`）。旧入口 `dsv41_indexer_topk` 一字未改。
* 放置（任务书未提，但这是正确性前提）：per-row **key publish 留在循环内**（顺序不能动，
  否则后面的行的 key 还没写就被前面行的选择读到），只有 **select 半**搬到循环之后、
  两条块级注意力臂之前。折叠仅当
  `is_idx_src && m > 1 && (mrows_attn || orope_mrows_ok) && (mrows_own_owner || attn_own_snapshot)`：
  * 前者保证循环内**没有任何东西读 `idxs_r`**（每行 attn 已被跳过）；
  * 后者保证每行的界是**设备快照**（`clen_rows_r`），行 `mm` 的界因此还是它自己读取点的值
    （后续行的 publish 写的是**严格更晚**的 slot：`*clen - 1`，而 `clen` 单调递增）。
* `indexer_rows_one` 增 `select: bool`：`false` 时 key publish 与 q/w 半照旧执行，直接返回。
* 输出地址与逐行调用**逐地址一致**：`out = idxs_r + offset`，行距 `ist`（`offset + index_topk`）。
* 期望收益：**6 发 → 1 发**/层 ≈ **−0.34ms**（任务书口径）。

---

## 2. 编译 / 测试

```
cargo check --workspace                     → EXIT=0
cargo test -p ferrite-models --lib          → 97 passed; 0 failed
```

`cargo test --workspace --no-run` 中 `gpu_smoke` / `ferrite-exec` 的**链接**失败是环境性的
（本机无 CUDA runtime：`undefined symbol: cudaGetErrorString / ferrite_graph_begin|end|instantiate`），
与本改动无关。**`.cu` 改动本机无法编译验证（无 nvcc）**，已逐行人工核对；
上机前建议先 `bash kernels/cuda/build.sh` 确认 nvcc 通过。

## 3. A/B 建议顺序（门都默认 OFF，需逐条单开）

1. F7（改动最小、风险最低）：`DSV41_F7_QUANT_STRIDE=1` + 重编 `.so`；
   红线段：verify 的 wo_a/wo_b 数值 —— 期望**逐位**等于关闭时（F7 只改寻址，不改值）。
2. F10：`DSV41_F10_CLEN_BLOCK=1`（需 `ATTN_MROWS` 或 `OROPE_MROWS` 才有量）；
   红线段：块级 attn 的每行 `topk` 界 —— 期望与逐行 D2D 时逐位相同。
3. F8：`DSV41_F8_TOPK_ROWS=1`（同上）；红线段：`idxs_r` 压缩半区逐 int 相同，
   以及 `sparse_attn` 读到的 selection 相同。
4. 三者叠加后再看总账（任务书 −1.5ms 口径里，F7 是最大且最稳的一项）。
