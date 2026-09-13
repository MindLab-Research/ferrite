# hc 前端在 verify（rows=m）的断裂根因与修复（2026-09-13）

> 票面：−3.0~3.6ms/步（病灶排位 #2，nsys：`hc_mixes_kernel` 30272 发 / 30272×51.4µs）
> 工作目录 HEAD：`5fa0efd`；本次只改 `chain_dev.rs`（86 行，1 文件）。
> 约束：禁 GPU/e2e；验证级 = `cargo check --workspace --all-targets` + 远端 nvcc compile-only。

---

## 0. 结论先说（三条，第 1 条推翻本票的既定假设）

1. **`hc_front_split` 链对 `rows = m` 是完备的 —— 「它对 rows>1 有硬检查/形状假设」的假设被证伪。**
   逐条核过（`kernels/cuda/dsv41_kernels.cu`）：

   | 位置 | 检查/索引 | 与 rows 相关？ |
   |---|---|---|
   | `dsv41_hc_front_split` :12598 | x/hc_fn/hc_scale/hc_base/pre/post/comb 非空 | 否 |
   | :12606 | side / fork_ev / join_ev 非空 | 否（`supports_hc_tail_split()` 已前置判过） |
   | :12608 | `rows <= 0 \|\| hc <= 0 \|\| dim <= 0` | 否（rows=6 过） |
   | :12609 | `!g_hc_front` | 否 |
   | :12610 | `rows > DSV41_HC_SPREAD_MAXR`（=**2048**，:10567） | 否（6 ≪ 2048） |
   | :12611 | `(w_norm==null) != (pre_collapse==null)` | 否（两者都非空） |
   | :12614 | `w_norm == null \|\| out == null` | 否（都非空） |
   | :12616 | `mix > 64`（hc=4 ⇒ mix=24，`configs/dsv41_flash.json`/`config.rs:277`） | 否 |

   三个 kernel 的行基址也全部原生：
   `hc_mix_dots_kernel`（:11285 `r = blockIdx.y`、:11292 `x + r*hc_dim`、:11326 `g_hc_part[r][m][*]`）、
   `hc_mixes_tail_kernel`（:11365 `r = blockIdx.x`、:11373 `xr = x + r*hc_dim`、:11405 `pre[r*hc+j]`、:11445 `comb[r*hc*hc+jk]`、:11453 EARLY 用 `out + r*dim`）、
   `hc_dots_late_kernel`（:12110 `r = blockIdx.y`、:12126/12189/12224 行基址、:12167 逐行选举 `g_hc_dl_done[r]`）。
   ⇒ **decline 与 rows 无关。**

2. **真正的断裂在决策层：`hc_mixes_auto` 的 `if / else if / else` 是「选一支」，不是「链」。**
   `chain_dev.rs:14509` 起：
   ```
   if  tail_split && supports && !norm_w.is_null()   → hc_front_split     … rows 无关 ✔
   else if rows == 1 && persist_mb …                 → hc_front_persist_mb … rows=6 死
   else if rows == 1 && persist …                    → hc_front_persist    … rows=6 死
   else                                              → hc_front            … 只在上面全不中时
   if fused { return Ok(true) }  →  raw hc_mixes（:14627）
   ```
   ⇒ 对 `rows = 6`，**唯一的融合入口是 tail split**。而 split 一旦「被选中但返回 decline」，
   Rust 的 `if/else` 不会继续走 `else` 链（`hc_front`），而是**直接落到 raw `hc_mixes`**
   —— 也就是 `:14504-14508` 注释里声称的 `(persist_mb -> persist -> two-launch -> hc_mixes), each step silent`
   **这条链在实现里从来不存在**。rows=6 时两个 persist 臂被 `rows == 1` 判死，
   于是 **rows 完备、且不需要 side stream / event 的 two-launch `hc_front` 变成了不可达**。
   decline 的映射（`device.rs:7187`）：`rc == 1 → Ok(false)`（1 == `cudaErrorInvalidValue`）。

3. **split 为什么会 decline？两个候选，输出上不可区分（这本身就是缺陷）；修复让两者都无害：**
   - **(i) `hc_front_rows()` 为 false** ⇒ `layer_rows` 的 `else`（:10670 / :10743）直接发 raw。
     `DSV41_HC_FRONT_ROWS` **默认 OFF**（:14804），而 nsys 表所剖的 **A0 基线栈**正是默认配置。
     这条能完全解释 80 发/步 raw，**且不需要任何代码断裂**。注意：A0 的
     「手动 SWALLOW + HC gates ON、输出正确（前 61 行零拉丁）」**不构成 split 跑过的证据** ——
     所有臂逐位等价，输出正确是三种臂的共同结果（这正是「gate ON 却走 raw」长期无法判读的原因）。
   - **(ii) split 被选中但 decline**：`dsv41_hc_front_split` 的**预期 decline 与真实 CUDA 错误都返回 1**
     （round-42 的 `1 == cudaErrorInvalidValue` 碰撞；house rule 已是「decline == 2」，
     仅 hc front 两个入口未改）。树内已记录的两次真实触发：b300-4 wedge 期
     「`hc_front_split` 的 3 处事件调用失败」（`STATUS.md:6387`）、以及 ABI 错位（:7353，cuda error 709）。

---

## 1. 修复

`crates/ferrite-models/src/dsv41/chain_dev.rs`（唯一改动文件）：

1. **把 fallback 真正串起来**（`hc_mixes_auto`，:14547-14598）：
   split 被选中但 decline 时，**先退 `hc_front`（two-launch）**，只有 `hc_front` 也 decline 才落到 raw。
   新增 `hc_front_note("hc_front_split declined -> hc_front (two-launch)", rows)`。
2. **决策点一次性诊断**（`DSV41_HC_DEBUG=1`，默认关）：`hc_front_debug()` / `hc_front_note()`（:14819-14840），
   按 `(note, rows)` 去重打印。三处埋点：
   - `layer_rows` 的 attn / ffn `else`（:10667 / :10741）：`hc_front_rows() off -> raw hc_mixes (gate)`；
   - `hc_mixes_auto` split decline（:14576）。
   下一次 GPU 轮**一行日志即可把 (i)/(ii) 判死**。
3. 修正 `hc_mixes_auto` 头部注释（:14504-14518），把「链」的承诺写成代码事实。

### 数值等价论证（bit-exact by construction）

- split 与 `hc_front` 跑的是**同一组 kernel、同一批操作数、同一批语句**：EARLY = `hc_mixes_tail_kernel(HC_TAIL_EARLY)`
  （collapse+rmsnorm+T1 fp8）、dots = `hc_mix_dots_kernel` / `hc_dots_late_kernel`、LATE = `hc_mixes_tail_kernel(HC_TAIL_LATE)`；
  差别只有 fork/join 事件与流（`dsv41_kernels.cu:12581-12584` 已写明「Bit-identical to dsv41_hc_front: the same
  statements execute with the same operands on whichever stream」）。
- `HC_TAIL_FULL`（`hc_front` 用）就是 EARLY+LATE 的同一条内核、同一顺序；split 只是把它们拆到两条流上。
- rows=m 的 fp8 契约不变：`layer_rows` 传 `xq = xsc = null`（tail 的 T1 是单行布局，:11467-11470 注释），
  `attention_rows` 仍用自己的 `quant_rows`。⇒ 与 raw 路径的 fp8 归属一致。

### 未做（留给尚书省决定，理由：最小改动 + peer 安全）

- 把 hc front 两个入口的**预期 decline 从 1 改成 2**（house rule），Rust 只把 2 当 decline、1 当硬错误。
  这会改 ABI 语义（需同步 `devrt.rs` 的 `EXPECTED_ABI`，不在本任务声明的区域内），
  且 decode 热路径也用 `hc_front`，故不单方面改。
- 翻转 `hc_front_rows()` 默认（docs 里已列为「唯一剩余动作」）。本次保持 OFF + 诊断可见。

---

## 2. GPU A/B 手册（双门禁 + nsys 判据）

前置（三证）：`cd kernels/cuda && bash build.sh 103a` + `cargo build --release` + 符号三证
（`nm -D libferrite_kernels.so | grep -c dsv41_hc_front_split` ≥ 1，且 build-id / ABI 与二进制一致）。

**一臂一进程，计数 200 tok，读 `[dspark] steps=50`。**

```bash
# A/B 两臂（同二进制，只差 A2 开关 + 本修复的诊断）
# arm base : 生产基线（A2 默认 OFF）
# arm a2   : A2 ON —— 本修复的目标臂
export BASE_ENV="DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 DSV41_TIMING=1 \
DSV41_SWALLOW_STEP=1 DSV41_SWALLOW_EPOCH_PAD=1 DSV41_VERIFY_GRAPH=1 \
DSV41_HC_VERIFY_FUSE=1 DSV41_FUSE_B1=1 DSV41_FUSE_C=1"
env $BASE_ENV DSV41_HC_FRONT_ROWS=1 DSV41_HC_DEBUG=1 bash scripts/batched_400_v2.sh 2>&1 | tee /tmp/a2.log
env $BASE_ENV DSV41_HC_FRONT_ROWS=0 bash scripts/batched_400_v2.sh 2>&1 | tee /tmp/base.log
```

**门禁 1（性能）**：`[dspark]` 分解的 `verify=` 中位数，a2 相对 base **≥ −3.0ms/步**。
**门禁 2（数值）**：`mean-k` **不掉**（A0 基线 **1.34**）；掉了立即弃用该臂（红线）。
另加零拉丁 / 0 double-char / 「先帝创业未半」三段文本红线。

**诊断判据（`DSV41_HC_DEBUG=1` 的 `[hc-front]` 行）**：
- 出现 `hc_front_rows() off -> raw hc_mixes (gate)` ⇒ 主因 (i)：env 没进到 serve 进程（查 `run.env` / `/proc/<pid>/environ` 回读）。
- 出现 `hc_front_split declined -> hc_front (two-launch)` ⇒ 主因 (ii)：split 运行期 decline；此时**本修复已把它降级成 two-launch**，
  仍应看到下面的 nsys 判据成立，且应单独排查事件/流失败（`[hc_tail] fork/join events unavailable`）。
- 两条都不出现 ⇒ split 真的跑了（修复无效也无妨，验收本就该过）。

**nsys 判据（swallow A0 栈同法）**：
```bash
DUR=60 MAXTOK=20 DSV41_HC_FRONT_ROWS=1 DSV41_HC_DEBUG=1 bash scripts/nsys_wave1.sh
/usr/local/cuda-13.2/bin/nsys stats --report cuda_gpu_kern_sum --format csv /tmp/wave1_nsys.nsys-rep | \
  grep -E "hc_mixes_kernel|hc_mix_dots_kernel|hc_mixes_tail_kernel|hc_dots_late_kernel"
```
- **`hc_mixes_kernel` 实例 → 0**（验收）
- **`hc_mix_dots_kernel` → 80/步 × 352**（验收；two-launch 与 split 都发它）
- 附带：`hc_mixes_tail_kernel` ≈ 80/步 × 352（或 `hc_dots_late_kernel` 同量，取决于 `DSV41_HC_DL_MERGE`）

**回归守卫**：`cargo test --release -p ferrite-dsv41 --test hc_post_rows_parity`（A1-b）在真机跑通；
本修复不碰 A1，但 A1/A2 同开时两者共用 `hc_post_rows` 的 `hc_tail_join`，属同一条链。

---

## 3. 本轮验证结果

| 项 | 结果 |
|---|---|
| `cargo check --workspace --all-targets` | ✅ EXIT=0（仅既有 warning） |
| 远端 nvcc compile-only（sm_103a，6 TU） | ✅ 6/6 `errors=0`（`dsv41_kernels.cu` / `ferrite_kernels.cu` / `dsv41_glue.cu` / `dsv41_route.cu` / `dsv41_vision.cu` / `dsv41_experts_mxf4.cu`） |
| `cargo test -p ferrite-dsv41` | ⚠️ 本机无 `libcudart.so` ⇒ GPU-gated 测试在 **bind 处 panic**（既有环境限制，非本次改动；`ar_hcpost_parity.rs:62`） |
| GPU / e2e | 未跑（任务禁止） |

*工部 · 改动仅 `chain_dev.rs` 1 文件 82+/4-；.cu 未动。*
