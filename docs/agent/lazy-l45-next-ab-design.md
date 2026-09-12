# lazy 路径的 L4/L5 下一批 —— L4-7 结案 + 1a/1b/B6/B5/B4 的 lazy A/B 设计

> 工部（ministry-works）· 2026-09-12 · 基线 HEAD `a5b94c8`（工作树现场核对）。
> 输入：`l4l5-next-batch-implementation-plan.md` §2 W-N1/W-N2 · `l4-occupancy-mlp-design.md` §2.2 · `l4-l5-kernel-path.md` §3 ·
> `lazy-verify-optimization-path.md` §「mrows 族在 lazy 下恒为 0」 · `batch-reverification-plan.md` §4 · `lazy_graph_ab.sh` 头注。
> 口径纪律：每条 ms 标来源（**实测** / **账本** / **代数** / **设计**）；行号一律以**函数名/符号**为准（本仓有行号漂移史）。

---

## 0. 结论先行（六条，前三条纠正任务前提）

1. **❗ L4-7 不是「设计完成未实施」——它已经全量落树，并且已经在 lazy 的干净栈 env 里。**
   三条证据链（§1）：`.cu` 的 `dsv41_hc_front_split(..., side_dl)` + `DSV41_HC_DL_SIDE` 第 4 侧流已把 **dots+LATE 移到独立流**（这正是「侧流遮蔽 sinkhorn」半边）；
   dots 网格 launcher 已是 `dim3(mix, rows)`（网格半边在 m=6 下早已兑现，计划 §0-2 自己已确认）；
   Rust 侧 `supports_hc_dl_side()` **默认 ON**、`hc_mixes_auto` 已接线，且 `DSV41_HC_FRONT_ROWS=1` **同时出现在三处权威栈 env** 里。
   ⇒ **本项的「下一批」票面已经是 0**，不应再排进任何批次。**剩余的可动项只有一个：把 gate 默认从 OFF 翻 ON（§1.4），而这必须先有一次 A/B。**

2. **❗ 1a（`fold_r`）在 lazy 上可证是 no-op —— 不要投这一臂。**
   `dsv41_mrows_fold_r_for(n, m)` 把 `fold_r` clamp 进 `[1, m]`；lazy 的 verify 块是 **`m = 1`**（`lazy_run_row` → `step_rows_sync(1 行)` → `layer_rows(m=1)`），
   于是 `fold_r ≡ 1 ⇒ ng = ceil(1/1) = 1 ⇒ grid = nt`，**与 gate OFF 逐字相同**。本轮已在真机把这个解析表钉住（§2 的实测输出）。
   ⇒ 计划 W-N1 里「1a 需扫格定符号」在 lazy 上**没有可扫的格**。

3. **1b 是本批唯一「已实测位等价 + 真省工作量」的 lazy 候选，建议首投。**
   本轮已在 B300 真机跑通 parity：**`DSV41_MROWS_ACT_CPASYNC=0/1` 在全部生产形状（含 m=1 的 lazy 块）与 m=1 单行参考逐位相同**（§5.1 实测输出）。
   与 B5/B4/B6 不同，1b 省的是**内核内的指令数**（不是发射），所以在 lazy 的图化世界里仍然全部有效（§3）。

4. **B5/B4/B6 在 lazy 上仍然「活」，但计划给的 ms 数一个都不能迁移过来。**
   `m=1` 时 `mrows = supports_gemm_fp8_mrows() && !swapab() && m <= VERIFY_ROWS` **为真**，所以 mrows 内核确实被派发（§4.1）——三项都会落到 `m=1` 的分支上。
   但计划里的 −0.13 / −0.66 / −0.79 ms 是 **batched m=6、40 发/步** 的口径；lazy 每步约 **40 × k_emit(≈2.214) ≈ 88.6 个层行**，且这些层行**已在图里**（§3）⇒ 收益的量纲完全不同。

5. **lazy 改变了发射账：图化把 B5/B4/B6 的「省发射」大半吃掉，把 1b/B6 的「省工作量」留下。**
   `VERIFY_GRAPH=1` + `VERIFY_GRAPH_SLOTS = 3` 让 m=1 行拥有自己的图槽（`lazy_graph_ab.sh` 记录的 23885a7：裸链 ~9.5 ms/行 → 图 ~6.15 ms/行）。
   ⇒ 在图内**没有 submit 成本**，器件的「少一次发射」只剩「少一个图节点的执行开销」；而删掉一个内核的**工作量**（B6 的 quant）、删掉内核内的**指令**（1b）则全额保留。

6. **本轮实际改动（§5.3）**：`kernels/cuda/tests_dsv41_gemm_mrows.cu` 补 **1b/1a 的 parity 轴**（计划 §6 N1 交付物「parity 扩轴（若未加）」），
   并修掉同一文件里一个**预存**的覆盖计数缺陷（`wq_b` 的 gap 区被算成「未写」，套件恒 EXIT=1）。`cargo check --workspace` EXIT=0（§5.4）。

---

## 1. L4-7 结案：为什么「实施」不成立

### 1.1 `.cu` 侧：侧流原语已在，且已经比设计更细
`dsv41_kernels.cu` 的 `extern "C" int dsv41_hc_front_split(...)` 签名末尾是 **`cudaStream_t side_dl`**，函数体里：

* `(0)` `cudaEventRecord(fork_ev, main)` → `cudaStreamWaitEvent(side, fork_ev)`：EARLY 半（collapse+rmsnorm+T1 fp8）钉在侧流；
* `(0b)` **同一个 record 的第二个 waiter** → `cudaStreamWaitEvent(dl, fork_ev)`，其中 `dl = side_dl ? side_dl : side`：
  **dots + LATE（ss/sigmoid/sinkhorn/comb）离开 EARLY 之后，走自己的第四流** —— 这就是 L4-7 的「侧流遮蔽 sinkhorn」半边；
* 头注写死了它的流序契约与位等价论证（EARLY / dots / LATE 三者在语句与内存上都互斥），
  以及 `join_ev` 仍会被 `hc_tail_join` 消费（main 只在 hc_post 前等一次）。

`DSV41_HC_DL_KCHUNK`（L4-8 的 K-chunk 版本，`hc_dots_late_kchunk_kernel`）也已在同一文件内，默认 OFF。

### 1.2 launcher 侧：dots 网格半边早已是 `(mix, rows)`
```
hc_mix_dots_kernel<<<dim3((unsigned)mix, (unsigned)rows), (unsigned)g_hc_dots_t, smem, s>>>
```
（`dsv41_kernels.cu` 的 `dsv41_hc_front` 两支 + `hc_front_split` 共用同一形式）。
`mix = hc * (2 + hc)`。⇒ `l4-occupancy §2.2` 的「dots 网格从 5 块变 `mix × rows`」是 **m=1 口径**；
在 batched m=6 下它**本来就是 `mix × 6`**（计划 §0-2 已确认，所以票面从 −1.3~−1.7 修到 −0.6）。

### 1.3 Rust 侧：默认 ON，且 lazy 的块正是 `layer_rows(1)`，走的就是这条线
| 环节 | 落点 | 状态 |
|---|---|---|
| 第四侧流 | `devrt.rs::side_stream4()`（`DSV41_DL_PRIO` 控制优先级，默认 default） | 已创建（`create_side_stream`，失败则打印告警并回落） |
| gate | `device.rs::supports_hc_dl_side()` = `env DSV41_HC_DL_SIDE != "0"`（**`unwrap_or(true)` ⇒ 默认 ON**）&& `side_stream4` 非空 | 已接线 |
| 传参 | `device.rs::hc_front_split()` 末位实参 `if supports_hc_dl_side() { side_stream4() } else { null }` | 已接线 |
| 调用者 | `chain_dev.rs::hc_mixes_auto()` 首个分支 `hc_tail_split() && supports_hc_tail_split() && !norm_w.is_null()` → `dev.hc_front_split(...)` | 已接线 |
| lazy 的行 | `lazy_run_row` → `step_rows_sync(&rows[i..=i])` → `step_rows_inner(m=1)` → `layer_rows(1, …)` → `hc_mixes_auto`（受 `DSV41_HC_FRONT_ROWS`） | 已接线 |

### 1.4 唯一的剩余可动项：gate 默认仍是 OFF（这是风险，不是收益）
`chain_dev.rs::hc_front_rows()` = `DSV41_HC_FRONT_ROWS`，**默认 OFF**（`unwrap_or(false)`）。
而三处权威栈 env 都显式设成 1：

* `scripts/sh_pair_ab.sh` 的 `BASE_ENV`
* `scripts/l49_ab.sh:184-189` 的 `BASE_ENV`（91.1 栈的 verbatim 来源）
* `scripts/nsys_wave1.sh` 的 `GATES`

⇒ **任何不带 env 的 lazy 复现会静默丢掉 L4-7**（`hc_mixes_auto` 回落 `hc_front`，dots 仍在 main 上、sinkhorn 不被遮蔽）。
按仓规（`dspark-correctness-chain` R6：gate 翻转单独 commit + 读回），**翻默认前必须先有一次同会话 A/B**：
基线（`HC_FRONT_ROWS=1`）vs `HC_FRONT_ROWS=0`，判据见 §5.2 的四段。这也是 `batch-reverification-plan.md §4 Phase 1`（arm 完整性审计）的实测化。

> **诚实边界**：L4-7 的增量是本批里**唯一有实测背书**的一项（91.1 栈本身就含它），所以「它的增量是多少」只能靠 A/B 反推，
> 而 A/B 的差值会与 `hc_verify_fuse`（A1）耦合——两者必须分开翻，否则差值不可归因。

---

## 2. 1a（`fold_r`）在 lazy 上：可证 no-op，真机已钉

`dsv41_kernels.cu::dsv41_mrows_fold_r_for(int n, int m)` 的全部分支：

| gate 取值 | 返回 |
|---|---|
| unset / `0` | `m`（ng = 1） |
| `auto`（`-1`） | `m`（2026-09-12 修订：auto **不再折叠**） |
| 正数 `v` | `clamp(v, 1, m)` |

于是 **m = 1 ⇒ 无论 gate 怎么设，`fold_r = 1`、`ng = ceil(1/1) = 1`、`grid = nt`** ——
与 gate OFF 的 `fold_r = 1`（因为 `m = 1`）**逐字相同**，`s_a` 尺寸、`r0/rn`、`row` 的上层全都不动。

**真机实测（本轮，`tests_dsv41_gemm_mrows.cu` 的 `mr_fold_r_contract()`，B300 GPU7）**：

```
[fold_r] DSV41_MROWS_FOLD_R=<unset> -> resolved (fold_r, ng) per production shape:
         wkv            n=512   m=5 -> fold_r=5 ng=1
         wq_a           n=1280  m=5 -> fold_r=5 ng=1
         wq_a           n=1280  m=6 -> fold_r=6 ng=1
         wq_b           n=2048  m=6 -> fold_r=6 ng=1
         wo_b           n=5120  m=5 -> fold_r=5 ng=1
         sh_w1/w3       n=288   m=6 -> fold_r=6 ng=1
         lazy/m=1       n=512   m=1 -> fold_r=1 ng=1     <-- lazy 的块
         lazy/m=1/wq_a  n=1280  m=1 -> fold_r=1 ng=1     <-- lazy 的块
[fold_r] OK  (resolution inside [1,m]; identity held for OFF/auto/0)
```

**判决**：1a 在 lazy 上**零作用**，A/B 臂是纯浪费。
（顺带解释了一件事：SWALLOW 上 `fold_r=auto` 曾 6× 崩，而 lazy 从不受影响——因为 lazy 的 m 恒为 1，`auto` 的旧规则 `n<=1024→1` 与 clamp 结果碰巧一致。）

---

## 3. lazy 的真实经济：图化把「省发射」大半吃掉

| 事实 | 落点 / 来源 | 后果 |
|---|---|---|
| lazy 每行 = 一次 `step_rows_sync(1)` → 40 次 `layer_rows(m=1)` | `lazy_run_row` | 层行数 = 40 × k_emit |
| k_emit ≈ **2.214** | `l4-occupancy §5.2`（计划 §4 明确「不要用 ×k_emit 入预算」——这里是**理由**不是承诺） | ≈ **88.6 层行/步** |
| `VERIFY_GRAPH=1` + `VERIFY_GRAPH_SLOTS = 3` 给 m=1 行自己的图槽 | `chain_dev.rs` 常量 + `lazy_graph_ab.sh`（23885a7：裸链 ~9.5 → 图 ~6.15 ms/行） | 行内 ~96 个发射**在图内，无 submit 成本** |
| 91.1 栈含 `DSV41_VERIFY_GRAPH=1` | `l49_ab.sh:184-189` | 上述成立 |

**推论（本文件最重要的一条）**：

* **B5 / B4 省的是「一次发射/层行」** ⇒ 在图内只剩「少一个图节点的执行开销」，残值可能落进噪声；
* **B6 省的是「一个 quant 内核的工作量 + 一次发射」** ⇒ 工作量那半全额保留，发射那半被图吃掉；
* **1b 省的是「内核内 ~77 warp-指令/块」** ⇒ 与图化正交，**全部保留**；
* 因此 **lazy 下的期望排序是 1b ≳ B6 > B5 ≈ B4**，与 batched 的（B6 > B5 ≈ B4 ≈ 1b）**不同**。

> **反证也在文档里**：`lazy-verify-optimization-path.md` 第 46-47 行已判「mrows 族在 lazy 下恒为 0，不要投」。
> 那条判断对 **`GATE_MROWS`/`INDEXER_MROWS` 这类「把 m 行折成 1 发」** 成立（m=1 时 1 发 ≡ 逐行）；
> **但它不覆盖 B5/B4**——B5/B4 是「把**两个**程序并成**一个**」（gate+route / norm+rope），在 m=1 下**仍然少一次发射**，与 m 无关。这是本文件对该结论的边界修正。

---

## 4. 四项在 lazy 上的活性与判据

### 4.1 「m=1 真的会走 mrows 内核」——活性前提先钉死
```
chain_dev.rs:  let mrows = self.dev.supports_gemm_fp8_mrows() && !Self::swapab() && m <= VERIFY_ROWS;
```
`dsv41_kernels.cu::dsv41_gemm_fp8_mrows` 的形状门是 `m ∈ [1, 8]`、`k % 32 == 0`、`out_stride >= n`、`g_gemv_fp8_mode >= 3`、无 `DSV41_NO_GEMV_FP8`。
⇒ **m = 1 是被接受的一种 M 特化**（`gemm_fp8_mrows_kernel<1>`，`FERRITE_SET_MROWS_SMEM(1)` 单独设过属性），
所以 1a/1b 所在的 kernel 在 lazy 上**每层行都跑**（`DSV41_SWAPAB` 默认 OFF，故 `!swapab()` 为真）。
⇒ 三项（1b/B6/B5）与 B4 的 m=1 分支**都不是空臂**。

### 4.2 逐项表

| 项 | gate | 默认 | m=1 是否活 | 活性判据（nsys **计数**，不读 ms） | 正确性判据 | lazy 预期（方向） |
|---|---|---|---|---|---|---|
| **1b** | `DSV41_MROWS_ACT_CPASYNC` | OFF（**`.cu` 侧 `getenv`，每发读**） | ✅ `gemm_fp8_mrows_kernel<1>` | 内核名不变 ⇒ 只能靠 **GridX 分布 + 位等价单测**；本轮的 `tests_dsv41_gemm_mrows` 已把位等价钉住 | **逐位**（纯 copy 替换，**本轮真机已验，含 m=1**） | **正**（省 ~77 warp-指令/块 × 88.6 块/步） |
| **B6** | `DSV41_VERIFY_WOB_MROWS_F32` | OFF（Rust gate，需 `.so` 符号） | ✅ `wob_mrows_f32() && mrows && supports_*` | `gemm_fp8_mrows_f32` 计数 ≈ 层行数；**`quant_fp8`(wo_b) 调用数归零** | **非逐位**（跳 fp8 往返，更准）⇒ 红线 + `DSV41_DIFF_EAGER` mismatch **不增** | 正（删一个内核的工作量；图内仍有效） |
| **B5** | `DSV41_GATE_MROWS_ROUTE` | OFF（还需 `row_fold_gate()`；**栈里 `DSV41_GATE_MROWS=1` 已满足**） | ✅ | `ferrite_gemv_bf16_v2_mrows_route` 计数 ≈ 层行数；`route_topk` 计数归零 | 逐位（同 GEMV 程序，只是选举位置变） | 弱正（图内只剩节点开销） |
| **B4** | `DSV41_RMSNORM_ROPE_MROWS` | OFF（**必须与 `DSV41_VERIFY_FORK` 同臂**；栈里已 ON） | ✅ 但 `norm_rows_on`+`apply_rope_on` 走 `kv_stream` | `dsv41_rmsnorm_rope_mrows` 计数 ≈ 层行数；`dsv41_rmsnorm_rows`(kv) + `apply_rope`(kv) 各归零 | 逐位（同程序 + 一个 barrier） | 弱正（同上，且**流必须留在 `kv_stream`**） |

### 4.3 两个必须先过的硬前置（否则臂读成「无效果」）

1. **`.so` 符号检查**（三项 Rust gate 都是 `ko!()` 探测，符号缺失 ⇒ `Ok(false)` **静默回落**）：
   ```bash
   nm -D kernels/cuda/libferrite_kernels.so | grep -cE \
     'dsv41_gemm_fp8_mrows_f32|ferrite_gemv_bf16_v2_mrows_route|dsv41_rmsnorm_rope_mrows'   # 期望 3
   ```
   1b 没有独立符号（它是 `gemm_fp8_mrows` 的参数），只能靠 §4.2 的 `GridX` + 位等价单测。
2. **`.cu` 变了 ⇒ 先 `bash kernels/cuda/build.sh 103a`，再看 `kernels/cuda/.build_id` 与二进制内嵌 id 一致**（仓规：两臂同源，否则「两臂都跑旧路径」）。

---

## 5. 协议与实施记录

### 5.1 本轮已完成的实测（真机，B300 GPU7，`tests_dsv41_gemm_mrows`）

1b 的 parity（`mr_case_cp16_axis`：同一进程内先 unset 再 `=1`，两臂都对 **同一份 m=1 单行参考** 做 bit-memcmp）：

```
[wq_a/m=6+bias/cp16=0] OK ... (7680 elems bit-identical)
[wq_a/m=6+bias/cp16=1] OK ... (7680 elems bit-identical)
[wkv/m=5/cp16=0]       OK ... (2560 elems bit-identical)
[wkv/m=5/cp16=1]       OK ... (2560 elems bit-identical)
[wo_b/m=5/cp16=0]      OK ... (25600 elems bit-identical)
[wo_b/m=5/cp16=1]      OK ... (25600 elems bit-identical)
[m=1/cp16=0]           OK ... (64 elems bit-identical)
[m=1/cp16=1]           OK ... (64 elems bit-identical)
[lazy/m=1/wkv/cp16=0]  OK ... (512 elems bit-identical)     <-- lazy 的块
[lazy/m=1/wkv/cp16=1]  OK ... (512 elems bit-identical)     <-- lazy 的块
...
RESULT: all checks passed
```

⇒ **1b 的「逐位」判据从设计口径升级为实测**；lazy 的 m=1 块也在覆盖内。

### 5.2 lazy A/B 的四段判据（缺一不算完成）

| 段 | 判据 |
|---|---|
| ① 活性 | nsys `cuda_gpu_kern_sum` 里出现 §4.2 指定的计数关系；**B6 的 `quant_fp8`(wo_b) 归零**；1b 用 `GridX` |
| ② 正确性 | 逐位项（1b/B5/B4）⇒ 见 §5.1 的套件 + 端到端 `raw u32` memcmp；B6 ⇒ 红线（计数前 61 行 + 出师表零拉丁 + `faults=0` + `ar5-hang=0`）+ `DSV41_DIFF_EAGER` mismatch 不增 |
| ③ 收益 | **同会话背靠背** A/B 的 `steady_median`（`STEADY_SKIP=20`），**一 gate 一变一 commit**；nsys 不读 ms |
| ④ 生效证明 | `tr '\0' '\n' < /proc/<pid>/environ \| grep DSV41_ \| sort` 读回（R6 陷阱） |

> **1b 的 ④ 例外**：`DSV41_MROWS_ACT_CPASYNC` 是 `.cu` 侧 `getenv`，**不进 `/proc/environ` 判死的名单要反过来看**——它其实在 environ 里（进程级 env），
> 但**看到 env 不等于 kernel 里生效**（`dsv41_f4_ok` 对齐守卫可能在某个形状上退回标量）。⇒ 1b 的生效证明**只能靠 §5.1 的单测 + GridX**，不能靠 envchk。

**止损线**（写死，不许事后放宽）：

| 项 | 止损 |
|---|---|
| 1b | 中性 ⇒ 说明该核在 lazy 的 m=1 形状上不是 staging 受限（m=1 的 staging 只有 1 行）⇒ 记「中性」并转 B6，**不投变体矩阵** |
| B6 | `\|Δsteady_median\| < 0.8ms` 且 `quant_fp8` 计数已兑现 ⇒ 判图内节点开销不可见，记录后转下一项 |
| B5/B4 | 任意一项中性 ⇒ **两项一起封存**（同源：都是「两发并一发」的发射账，图化后同命运） |
| L4-7 翻默认 | A/B 差值 < 噪声 σ ⇒ 默认**保持 OFF**，只在栈 env 里显式设（不动默认 = 不引入不可归因的变化） |

### 5.3 本轮代码改动

| 文件 | 改动 | 为什么 |
|---|---|---|
| `kernels/cuda/tests_dsv41_gemm_mrows.cu` | +`mr_case_cp16_axis()` / +`mr_fold_r_contract()` / +main 调用点 / 头注两节 | 计划 §6 N1 交付物「parity 扩轴（若未加）」：原文件**零** `fold_r`/`act_cp16` 覆盖、零 `setenv` |
| 同上 | **修预存缺陷**：覆盖计数把「故意不写的 gap 区」算成未写 | `wq_b`（`out_stride=4096 > n=2048`）恒报 `12288 = 6*2048` 未写 ⇒ 套件恒 `EXIT=1`，会把真失败藏在已知红臂后面 |

### 5.4 本地硬门禁

```bash
cargo check --workspace          # EXIT=0（本轮已跑）
# .cu 变了 ⇒ 远端编译（本机无 nvcc）：
bash scripts/dsv41_compile_check.sh
# 或直接：nvcc ... tests_dsv41_gemm_mrows.cu（会 #include dsv41_kernels.cu）
```

---

## 6. 建议的施工顺序（lazy，按「有没有实测背书」排，不按设计 ms 排）

| # | 项 | 成本 | 依据 | 预期 |
|---|---|---|---|---|
| 1 | **1b** `DSV41_MROWS_ACT_CPASYNC=1` | 零代码 | **本轮已实测位等价**（含 m=1） | 正但小；唯一「省工作量且与图正交」的项 |
| 2 | **B6** `DSV41_VERIFY_WOB_MROWS_F32=1` | 零代码（先验符号） | 设计 §9 六件已落树 | 正；且把 verify 的 wo_b **对齐到 EAGER 的 `wob_f32`**（一致性收益） |
| 3 | **B5 + B4** | 零代码 | 核在树 | 弱正；图内只剩节点开销，两项同命运一起判 |
| 4 | **L4-8** `DSV41_HC_DL_KCHUNK=1` | 零代码 | 核在树（计划 N2-2） | 可与上面搭同一次会话 |
| 5 | **L4-7 翻默认**（`HC_FRONT_ROWS` OFF→ON） | 1 行 + A/B | §1.4 | 只有 A/B 差值 > σ 才翻 |
| — | ~~1a~~ | — | §2 可证 no-op | **不投** |
| — | ~~L4-7 实施~~ | — | §1 已全量在树 + 已在栈 | **不投**（票面已为 0） |

> **诚实校准**：本文件给出的所有收益排序都是**方向**，不是承诺。lazy 的层行在图里，任何「少一次发射」的项都必须由同会话 A/B 的 `steady_median` 落定；
> 计划 §4 的 400 缺口账**不能**用 ×k_emit 把 batched 的 ms 折算进 lazy（`l4-occupancy §5.2` 已判）。

---

*工部 · 代码改动集中在 `tests_dsv41_gemm_mrows.cu`（parity 轴 + 一个预存测试缺陷）；未改任何 gate 默认、未改任何生产核。*
*真机证据：B300 GPU7，`CUDA_VISIBLE_DEVICES=7 /tmp/t_gemm_mrows` → `RESULT: all checks passed`。*
*所有 kernel 名以 `__global__` / `extern "C"` 符号为准；所有 ms 标来源（实测 / 账本 / 代数 / 设计）。*
