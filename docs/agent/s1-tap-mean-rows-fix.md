# S1 主修：tap 块写的 per-row `pre` 越界 —— E3 护栏 + 等价化修复（2026-09-13）

> 票面：0.78 accept 损失的头号嫌疑（SWALLOW 1.34 vs lazy 2.12）——SWALLOW 的 m=6 块在 tap
> （draft 的输入 `main_h`）上与 lazy 的 m=1 不等价。
> 工作目录 HEAD：`b6b69c4`；**只改 `crates/ferrite-models/src/dsv41/chain_dev.rs`（1 文件），.cu 未动**。
> 约束：禁 GPU/e2e；验证级 = `cargo check --workspace --all-targets` + 远端 nvcc compile-only。
>
> ⚠️ 工作树共享：本轮 `chain_dev.rs` 同时承载 **peer 的 S2（`DSV41_COMP_PARITY` /
> `CompParityState` / `comp_parity` / sh-gate 启动打印）**改动。本文只记录 **工部（S1/E3）**
> 的改动，peer 部分不在交付范围。

---

## 0. 结论先说（一条，且与 E2/E5 判决一致）

**S1 的根因不是 hc 前端 kernel 的选择，而是 tap 块写这一行本身：`hc_collapse` 的 `pre` 是
PER-ROW-STRIDED（`pre[rows][hc]`），而 m-row tap hook 传的是单行 `dspark_pre_mean`（`hc` ＝ 4
个 float）。`rows = m` 时 `r >= 1` 的每一行都读越界。**

三条证据：

1. **kernel 契约**（`kernels/cuda/dsv41_glue.cu:304-316`）：
   ```c
   const int r = (int)(t / (size_t)dim);
   const float* pre_r = pre + (size_t)r * hc;          // ← 逐行 stride
   const float* x_r   = x + (size_t)r * hc * dim + c;
   for (int i = 0; i < hc; i++) acc = fmaf(pre_r[i], x_r[(size_t)i * dim], acc);
   ```
2. **契约测试**（`kernels/cuda/tests_dsv41_glue.cu:462-482`）：`run_hc_collapse_case(rows, hc, dim)`
   分配的是 `pre[rows][hc]`，CPU 参考写成 `pre[r*hc + i]`，并用
   `dsv41_hc_collapse(dx, dpre, dout, rows, hc, dim, 0)` 对拍。⇒ `pre` 的**契约就是逐行**。
3. **调用端**（`chain_dev.rs` 两个 m-row tap hook）：`rows = m` 却传
   `dspark_pre_mean`——`Scratch` 里它是 `dev.alloc(fb(hc).max(8))` ＝ **4 个 float**。
   `hc_collapse_kernel` 在 `r >= 1` 时读 `pre + r*hc`，即第 16 字节之后 —— **越界**。

**与 E2 的吻合**：E2 三臂（`HC_VERIFY_FUSE=0 / HC_FRONT_ROWS=0 / VERIFY_AR_FOLD=0`）各自关掉后
mean-k 均 1.380 不变 ⇒ hc 前端 kernel 的选择不是来源。**本 bug 非门控**（三个 env 一个都不读），
所以关三开门当然不变 —— E2 的结果不是排除，而是**指向 tap hook 本身**。

**与 E5 的吻合**：E5 显示同 pos 的 drafts 第一个预测就分叉（pos=52: lazy `[201,511,...]` vs
SWALLOW `[426,397,...]`），SWALLOW 的 tap 让 draft 变差。本 bug 直接污染
`note_ctx_rows` / `carry_kept_tap` 消费的 `tap_r` 行 `1..k_emit`（行 0 恰好正确，因为
`pre + 0`）——正是「lazy 数字位预测对、SWALLOW 错」的形态。

**为什么 hc_collapse「行独立」的既有论证会漏掉它**：`layer_rows` 的注释
（`chain_dev.rs:10965-10973`）写了「`out[t] = Σ_i pre[r*hc + i] * x[...]`，无跨行项 ⇒ m-block
与 m 次单行调用逐位相同」。**它引用了 kernel 的逐行 `pre` 索引，却传了单行 `pre`**——m 次单行
调用每次都读 `pre + 0`，m-block 读 `pre + r*hc`，两者只有在 `pre` 是 `[m, hc]` 全相等时才等
价。论证的反例就是这一行。

---

## 1. E3 护栏（`DSV41_TAP_PARITY=1`，默认 OFF）——首要交付

**位置**：`chain_dev.rs` `dspark_dump_step` 之后（新 `tap_parity_probe`），调用点在
`dspark_spec_swallowed` 的 6-row `step_rows` 之后、accept 之前（post-block / pre-commit）。

**做什么**：把 SWALLOW 块写的 `dspark_tap_r`（`[DSPARK_TAP_SLOTS, VERIFY_ROWS, dim]` f32）
D2H；再把同一个块按 **lazy 的程序**（`k_emit`... 实测用全 `m` 行，见下）逐行重放一遍；
两份逐 **f32 位**（`to_bits()`，不合并 ±0 / NaN）比对，报**第一个分歧 `(slot, layer, row, elem)`**。

**rollback 纪律**（照 `diff_eager_probe` :4747-4789 的 take/put-back）：

| 步 | 动作 | 说明 |
|---|---|---|
| 1 take | D2H `dspark_tap_r`；存 `verify_blocks/captures/replays` | m-block 的 tap 完好，调用者快照仍有效 |
| 2 undo | `dspark_rollback(pos, m, host_mirrors)` | 回到块前状态，reference 从同一起点跑 |
| 3 reference | `for i in 0..m`：`set_pos_ctr(pos+i)` → `spec_capture=false; spec_tap_deferred=true` → `step_rows_sync(&rows_in[i..=i], skip_barrier=true, Some(pos+i))` → `spec_tap_deferred=false` → `lazy_tap_commit(i)` | **lazy 的逐行程序**；`skip_barrier` 与 lazy 循环一致（每 rank 跑同一序列） |
| 4 take | D2H reference 的 `dspark_tap_r` | |
| 5 put-back | `dspark_rollback` → `set_pos_ctr(pos)` → 重跑**原 6-row 块**（确定性）→ 还原三个 graph 计数 | **无条件 put-back**；重跑让调用者的 commit 看到块 1 自己的状态（argmax_r / ring / compressor carry） |
| 6 compare | 逐 slot/row/elem `to_bits()` 比对 | 第一个分歧即报；否则 `IDENTICAL` |

要点：
- **全 rank 跑 1-5**（它们是 `step_rows` 调用 = v5 AR 轮次，rank-0-only 会错相位）；只有**打印**是 rank 0。
- 一次内部错误（重放/下载失败）在 put-back **之后**才以 `Err` 返回——put-back 无条件（同 `diff_eager_probe`）。
- 成本（仅 gate ON）：整块被 **多前向两次**（一次逐行做 reference，一次 6-row 复原）。**这是诊断臂，单独进程跑，不进 A/B 计时臂。**

**调用点代码**：
```rust
// ---- 3b. E3 (DSV41_TAP_PARITY=1) / S1 fix (DSV41_TAP_STRICT_ROWS=1) ----
if Self::tap_parity() || Self::tap_strict_rows() {
    self.tap_parity_probe(pos, m, &rows_in, &host_mirrors, Self::tap_strict_rows())?;
}
```

---

## 2. S1 分叉点代码分析（为什么 tap 会不同）

`layer_rows` 的 tap hook 与 lazy 的路径对比（`DSV41_TAP_INPUT` 默认 OFF，故走 end-of-block hook）：

| | SWALLOW（`spec_capture=true`） | lazy（`spec_tap_deferred=true`） |
|---|---|---|
| dst | `dspark_tap_r + slot*VERIFY_ROWS*dim` | `dspark_tap + slot*dim`（staging） |
| rows | **`m`（6）** | **`1`** |
| pre | `dspark_pre_mean`（**4 float**） | `dspark_pre_mean`（4 float） |
| 后续 | 无（块写直达 `tap_r`） | `lazy_tap_commit(i)` 把 staging 拷到 `tap_r[(slot*VERIFY_ROWS+i)*dim]` |

⇒ **唯一的内容分叉就是 `rows`**：块写让 kernel 用 `pre + r*hc` 索引一个只有 `hc` 个元素的缓冲。

**逐项排除其它候选**（都逐位等价，或至少不是 rows=m 独有）：
- `hc_collapse_kernel` 本身行独立（无跨行项）——排除。
- `hc_collapse_norm`（`dsv41_kernels.cu:11083`）`grid=rows, blockDim=1024`、collapse/norm 语句与
  `hc_collapse`+`norm_rows` 逐句对齐；`hc_post_inplace_rows`（:10930）与单行
  `dsv41_hc_post_inplace_kernel` 同 TU、同 `__fmaf_rn` 升 k 链——逐位等价。
- AR fold（`ferrite_kernels.cu:9702` `ar5_hc_post_col4`）用 `__fmul_rn`+`__fmaf_rn` 显式钉死，
  且是默认 OFF 的独立臂——不是本 bug。
- 「布局/顺序」差异：块写与逐行 commit 的**目标偏移**是同一个公式 `(slot*VERIFY_ROWS+r)*dim`，
  顺序也都被 `lazy_tap_commit` 对齐——**不是布局问题，是内容问题（`pre` stride）**。

---

## 3. S1 修复（等价化）——Fix A，默认 ON；Fix B 为保守备胎

### Fix A（`DSV41_TAP_MEAN_ROWS`，默认 **ON**；`=0` ＝ 历史臂）

**改动**：新增 `Scratch::dspark_pre_mean_r: DevBuf` ＝ `fb(VERIFY_ROWS * hc)`（`[VERIFY_ROWS][hc]`
全 `1/hc`），`reset()` 里填满；m-row hook 通过 `tap_pre(rows)` 选 `pre`：

```rust
fn tap_pre(&self, rows: usize) -> *const f32 {
    if rows > 1 && Self::tap_mean_rows() {
        self.s.dspark_pre_mean_r.as_f32()
    } else {
        self.s.dspark_pre_mean.as_f32()   // 历史单行指针，rows == 1 逐位不变
    }
}
```

两个 m-row hook（tap_input hook + end-of-block hook）由 `self.s.dspark_pre_mean.as_f32()`
改为 `self.tap_pre(rows)`。`layer()` 的两处**单行**（`rows = 1`）hook 不动 —— 它们永远读 `pre + 0`。

**数值等价论证（bit-exact by construction）**：

对块的第 `r` 行、列 `c`：
- 块写：`acc_block(r,c) = Σ_i fmaf(pre_r[i], h_r_block[r*hc*dim + i*dim + c], acc)`，`pre_r[i] = 1/hc`。
- lazy 第 `i=r` 行：`acc_lazy = Σ_i fmaf(pre[i], h_r_single[i*dim + c], acc)`，`pre[i] = 1/hc`，
  随后 `lazy_tap_commit(r)` 写到 `tap_r[(slot*VERIFY_ROWS + r)*dim + c]`。
- **项与项、序与序、操作数与操作数完全相同**（同一 `fmaf` 链、升序 `i`）⇒ 只要
  `h_r_block` 的第 `r` 行与 lazy 单行的 `h_r` 逐位相同，两者就 **bit 相同**；
  目标偏移也是同一公式。∎

`h_r_block` 行 `r` ↔ lazy 单行的 `h_r` 的逐位相同，是代码库已文档化的
mrows 家族不变式（并将由 E3 在字节级直接证伪/证实）。

**代价**：一个 `96 B` 的缓冲 + `reset` 一次 24-float H2D。稳态零成本。**decode 路径逐位不变**
（`rows == 1` 时 `tap_pre` 返回历史指针）。`=0` 保留 OOB 读，供同二进制 A/B。

### Fix B（`DSV41_TAP_STRICT_ROWS=1`，默认 OFF）——保守备胎

`tap_parity_probe(..., adopt = true)`：在 put-back 之后把 reference（逐行 tap）**H2D 覆盖回
`dspark_tap_r`**，让 draft 直接消费 lazy 的块。**绕过块写分叉**，代价是 verify 前向约 ×2。
用于「即使 Fix A 论证有漏，也能靠 per-row 生成把 draft 输入扳回来」——预期 mean-k 1.34 → 2.1+。

---

## 4. 本轮验证结果

| 项 | 结果 |
|---|---|
| `cargo check --workspace --all-targets` | ✅ **EXIT=0**（仅既有 warning；与 peer 的 S2 改动共存） |
| `cargo test -p ferrite-models --lib` | ✅ **92 passed / 0 failed / 2 ignored** |
| 远端 nvcc compile-only（sm_103a，6 TU） | ✅ **6/6 `rc=0 errors=0`**（`dsv41_kernels.cu` / `dsv41_glue.cu` / `ferrite_kernels.cu` / `dsv41_route.cu` / `dsv41_vision.cu` / `dsv41_experts_mxf4.cu`） |
| `cargo test -p ferrite-dsv41 --test ar_hcpost_parity` | ⚠️ 本机无 `libcudart.so` ⇒ bind 处 panic（**既有环境限制，非本次改动**；见 `hc-front-rows-break-fix.md` §3） |
| GPU / e2e | 未跑（任务禁止） |

---

## 5. GPU 验证手册

**前置（三证）**：`cd kernels/cuda && bash build.sh 103a` + `cargo build --release` +
符号三证（`nm -D libferrite_kernels.so | grep -c dsv41_hc_front_split` ≥ 1，build-id / ABI 与二进制一致）。
**一臂一进程**，计数 200 tok。

### 5.1 E3 定位分歧点（先跑，证明机制）

```bash
export BASE_ENV="DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 DSV41_TIMING=1 \
DSV41_SWALLOW_STEP=1 DSV41_SWALLOW_EPOCH_PAD=1 DSV41_VERIFY_GRAPH=1 \
DSV41_HC_VERIFY_FUSE=1 DSV41_FUSE_B1=1 DSV41_FUSE_C=1"

# arm bug : 历史臂（单行 pre，越界）
env $BASE_ENV DSV41_TAP_MEAN_ROWS=0 DSV41_TAP_PARITY=1 bash scripts/batched_400_v2.sh 2>&1 | tee /tmp/s1_bug.log
# arm fix : 修复臂（per-row 均值）
env $BASE_ENV DSV41_TAP_MEAN_ROWS=1 DSV41_TAP_PARITY=1 bash scripts/batched_400_v2.sh 2>&1 | tee /tmp/s1_fix.log
```

**判据**：
- `arm bug` 应出现 `[tap-parity] MISMATCH pos=... slot=... layer=Some(37|38|39) row>=1 elem=...`
  ⇒ 第一处分歧**定位到 (layer, row, elem)**，且 `row >= 1`（行 0 必 IDENTICAL——`pre + 0`）是
  本 bug 的**指纹**。
- `arm fix` 应出现 `[tap-parity] IDENTICAL ...`（第一条即足；全 IDENTICAL 更佳）。
- 若 `arm fix` 仍 MISMATCH：说明 `h_r` 本身在 m-row 与 m=1 间有 ULP 漂移（E3 会给出新的
  `(layer,row,elem)`），此时改用 Fix B。

### 5.2 accept / 吞吐 A/B

```bash
# 基线（历史臂）与修复臂，其他 env 同上，去掉 DSV41_TAP_PARITY
env $BASE_ENV DSV41_TAP_MEAN_ROWS=0 bash scripts/batched_400_v2.sh 2>&1 | tee /tmp/s1_ab_base.log
env $BASE_ENV DSV41_TAP_MEAN_ROWS=1 bash scripts/batched_400_v2.sh 2>&1 | tee /tmp/s1_ab_fix.log
# 保守备胎（可选，单变量）：tap 走逐行
env $BASE_ENV DSV41_TAP_MEAN_ROWS=1 DSV41_TAP_STRICT_ROWS=1 bash scripts/batched_400_v2.sh 2>&1 | tee /tmp/s1_ab_strict.log
```

**门禁**：
1. **数值**：修复臂 `mean-k` 相对基线**上升**（目标 **1.34 → 2.1+**；A0 基线 1.34，lazy 2.12）。
2. **文本红线**：零拉丁 / 0 double-char / 「先帝创业未半」三段文本照旧。
3. **性能**：Fix A 的 `step_ms` 与基线**持平**（稳态零成本；只有 `reset` 多一次 24-float H2D）。
   Fix B 预期 +一次 6-row 前向 ≈ +1.6ms/轮——只作机制验证。

**环境哨兵**（env 回读纪律，必需）：`/proc/<serve pid>/environ` 里回读
`DSV41_TAP_MEAN_ROWS`；`[sh-gate] startup` 行确认 env 进了 serve 进程。

### 5.3 回归守卫

- `cargo test --release -p ferrite-dsv41 --test hc_post_rows_parity`（A1-b）真机跑通。
- 本改动不碰 A1 / hc 三开门；`DSV41_TAP_MEAN_ROWS=0` 应复现 HEAD 的 accept 数字（同二进制 A/B 的自证）。

---

*工部 · 改动仅 `chain_dev.rs`（E3 `tap_parity_probe` + Fix A `dspark_pre_mean_r`/`tap_pre` +
Fix B `DSV41_TAP_STRICT_ROWS` + 两个 m-row hook 的 `pre`）；.cu 未动。*
