# mrows 族在 SWALLOW batched 模式下的实施设计（S2）

> 工部 · 2026-09-12 · **只读勘察 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 输入（现场核对）：`swallow-unlocked-shpair-m6-throughput-plan.md` · `swallow-full-gate-config.md` ·
> `swallow-nsys-batched-analysis-framework.md` · `verify-eager-fusion-migration.md` ·
> `b6-mrows-f32-design.md` · `400-fastest-path-roadmap.md`。
> 读码：`crates/ferrite-models/src/dsv41/{chain_dev,device,config}.rs`、
> `kernels/cuda/{dsv41_kernels.cu,dsv41_glue.cu,ferrite_kernels.cu}`、`scripts/batched_400_v2.sh`。
> 基线 HEAD `ba88df1`。**口径纪律**：每条 ms 标来源（**实测** / **launch 账** / **设计** / **代数**）。

---

## 0. 先读六条（其中三条**修正任务前提**，必须上报）

1. **❗「mrows 族是 S2、价值 −4.5ms」——成立，但只有 3/6 项是「零代码接线」。**
   任务表把 B1–B6 并列成「mrows 项」，但按代码事实它们是两类：

   | 类 | 项 | kernel 是否存在 | 实施成本 |
   |---|---|---|---|
   | **接线（零 kernel）** | B1 / B2 / B3 | **已存在** | 加一个 env gate |
   | **新 kernel（设计已备）** | B4 / B5 / B6 | **不存在** | 写核 + parity + 接线 |

   ⇒ **S2 不是一次「翻转 6 个 gate」的动作，是「3 个 A/B + 3 个新核」。** 编预算时必须分开，
   否则会重演 `swallow-unlock-400-sprint §B1` 的「幻影门」误判。
   【读码：§2 的符号表】

2. **❗ B1 ⊂ B2 —— 两者在同一个调用点（wq_b）互斥，收益**不可相加**。**
   B2 = K2 = `gemm_fp8_mrows_rope_norm`（**norm + quant + wq_b + rope 四合一**），
   B1 = 同一条链上少折一个 norm（`proj_mrows + apply_rope_mrows`）。
   **K2 上场后，B1 的调用点被整段吃掉 ⇒ B1 的增量 = 0。** 任务给的
   `B2(-0.8~1.2) + B1(-0.4~0.7)` 若直接相加 = 重复计账。
   【读码：`chain_dev.rs:10790-10823`（K2）、`:10864-10922`（B1 的两个 launch）、`:10810` `q_norm_fused` 短路】

3. **❗ 6 个 mrows gate 里的 6 个**已经**在权威配置里（`GATE_MROWS`/`INDEXER_MROWS`/`NORM_MROWS`/
   `COMPRESSOR_MROWS`/`SH_EXP_MROWS`/`VERIFY_HEAD_MROWS`）——它们**不计入 S2 的增量**。**
   `scripts/batched_400_v2.sh:145-156` 的 `GATES` 已含这 6 个。58.3 tok/s 的基线**已经带着它们**。
   ⇒ S2 的账只能来自**本配置里还没有的**那几条：`VERIFY_ROPE_MROWS` / `ATTN_MROWS_ROPE_NORM` /
   `ATTN_MROWS2`（零代码）+ B4/B5/B6（新核）。
   【读码：脚本 GATES 逐字 vs `chain_dev.rs` 的 gate 函数表】

4. **B6 是唯一「非位等价」项**（`dsv41_gemm_fp8_mrows_f32`）——它让 wo_b 直读 f32 激活，
   跳过 fp8 往返，**比现状更准**，与本仓已有的 `DSV41_WOB_F32`（E8，`gemm_fp8_mx_f32`）
   同一语义。⇒ 它的 gate **不能**和「bit-identical by construction」的 B1–B5 共用验收口径：
   它必须走**红线复验**（计数数字顺序 + 出师表零拉丁），不是逐位比对。
   【读码：`chain_dev.rs:4885-4899`（E8 的注释）、`b6-mrows-f32-design.md §3.1/:195`】

5. **B4 的靶子已被 NORM_MROWS 削掉一半，B5 的靶子已被 GATE_MROWS 削掉一半。**
   * kv 链现状 = `norm_rows_on(kv_r)`（1 发，**已经是 m 行**）+ `apply_rope_on(kv_r, rows=m, step=1)`
     （1 发，**已经是 m 行单发**）⇒ B4 只剩「把这两发并成一发」= **−1 发/层**。
   * MoE 现状 = `gemv_bf16_v2_mrows`（GATE_MROWS，1 发）+ `route_topk(rows=m)`（1 发，**已经是 m 行**）
     ⇒ B5 只剩「gate 的 last-block 选举跑 route」= **−1 发/层**。
   ⇒ B4/B5 都是 **−40 发/步**的小项，**不要按 lazy 口径的「逐行 × m」去编它们的票面**。
   【读码：`chain_dev.rs:10945-10968`、`:12963-12995`】

6. **mrows 族的上限是 nsys 的「gemv 投影」族 = 15.1% / ~4.2ms**，不是加法账。
   launch 账（§3）给 `−2.8ms（3.3µs/发）~ −5.2ms（6.2µs/发）`，**恰好盖住 4.2ms 这个天花板**——
   说明这套账**自洽但不留余量**：任何一项的 decline（静默回落）都会把对应份额直接吃掉。
   ⇒ **每一项都必须有 nsys 的 kernel 名证据**（§5），否则不能计入兑现。
   【实测：`swallow-nsys-batched-analysis-framework §6.3` 表 7 项 vs 本文件 launch 账】

---

## 1. batched 路径的「mrows 判据」（与 lazy m=1 的本质差异）

### 1.1 机械判据

SWALLOW 从第 2 轮起走 `step_rows(m=6)` → `layer_rows` → `attention_rows` / `moe_rows` /
`compressor_mrows`；**没有** `layer()`（单行）和 lazy 逐行路径。

| 判据 | 含义 |
|---|---|
| **M 判据** | gate 的读点落在 `attention_rows` / `moe_rows` / `step_rows` 内 ⇒ SWALLOW 生效 |
| **m=1 判据** | gate 表达式含 `m == 1`、或只被 `layer()` / lazy 调用 ⇒ SWALLOW **空门** |

### 1.2 为什么 m=6 才有收益（任务是**对的**）

`gemm_fp8_mrows_kernel<M>` 是**权重驻留（weight-stationary）**形态：一个 block 内多个 warp
各自持有一段权重行，`M` 只决定「每个 warp 为几行服务」。
* **lazy m=1**：M 唯一地加每 warp 的工作量，权重只读一遍本来就是必然 ⇒ **mrows 无收益**（任务前提 ✓）。
* **SWALLOW m=6**：`quant_rows` 把激活块量化**一次**，6 行共享同一份 fp8 字节；权重读一遍为 6 行服务
  ⇒ **权重带宽 ÷6、launch ÷6**（`quant_rows` 的那一发也省了）⇒ **真实收益**。

**并且**：`mrows` 这个布尔量在 batched 路径里**根本没有 gate**：

```rust
// chain_dev.rs:10588
let mrows = self.dev.supports_gemm_fp8_mrows() && !Self::swapab() && m <= VERIFY_ROWS;
```

⇒ 只要 `.so` 带符号且 `m<=6`，`proj_mrows`（wq_a/wkv/wq_b/wo_b）**就已经在跑**。
这解释了为什么 B1/B3 的「mrows 化」不需要动 kernel——**它们早就是 mrows 了，缺的只是旁支的融合**
（rope / 两 family 合一 / f32 直读）。这是本设计最关键的一条：**先看「哪些已经跑了」，再谈接线。**

---

## 2. 逐项：现有符号 vs 缺的符号（`nm -D` 口径）

| # | 目标核（`b6-mrows-f32-design`/`verify-eager-fusion-migration §3.2` 命名） | 树内状态 | 证据 |
|---|---|---|---|
| B1 | `dsv41_gemm_fp8_mrows_rope`（wq_b + rope） | **无**（但**不需要**：见 §0-2） | `grep` 无匹配 |
| B1′ | `dsv41_gemm_fp8_mrows`（wq_b 本体）+ `dsv41_apply_rope_mrows` | **有** | `dsv41_kernels.cu:5387`、`:9182` |
| B2 | `dsv41_gemm_fp8_mrows_rope_norm`（=K2，norm+quant+wq_b+rope） | **有** | `dsv41_kernels.cu:6006` |
| B3 | `dsv41_gemm_fp8_mrows2`（=K1，wq_a + wkv 双 family） | **有** | `dsv41_kernels.cu:5650` |
| B4 | `dsv41_rmsnorm_rope_mrows`（kv norm + rope） | **无** | `grep` 无匹配 |
| B5 | `ferrite_gemv_bf16_v2_mrows_route`（gate + route 单发） | **无** | `grep` 无匹配（只有 `ferrite_gemv_bf16_v2_route`，m=1 版） |
| B6 | `dsv41_gemm_fp8_mrows_f32`（wo_b 直读 f32） | **无**（设计已备，`b6-mrows-f32-design.md`） | `grep` 无匹配 |
| — | `dsv41_rmsnorm_rows`（NORM_MROWS 用） | **有** | `dsv41_kernels.cu:9373` |
| — | `ferrite_gemv_bf16_v2_mrows`（GATE_MROWS 用） | **有** | `ferrite_kernels.cu:3355` |

**Rust 侧 probe 方法**（`device.rs`）：
* 已有：`supports_gemm_fp8_mrows()`@4413、`supports_gemm_fp8_mrows_rope_norm()`@4505、
  `supports_gemm_fp8_mrows2()`@4583。
* 待加：`supports_gemm_fp8_mrows_f32()`（B6）、`supports_rmsnorm_rope_mrows()`（B4）、
  `supports_gemv_bf16_v2_mrows_route()`（B5）。

---

## 3. 每项：接线点 + 预期 + 验证

> 层数 `n_layers = 40`（`config.rs:171` `unwrap_or(40)`）；`VERIFY_ROWS = 6`（`chain_dev.rs:84`）。
> 每发成本双口径：**3.3µs**（execution 半，保守）/ **6.2µs**（launch 全账，`verify-ms-breakdown §2`）。
> 「省发/步」= 省发/层 × 40 层。

---

### B2 —— `mrows_rope_norm`（K2）· **零代码，ROI 第一**

| 项 | 内容 |
|---|---|
| **gate** | `DSV41_ATTN_MROWS_ROPE_NORM=1`（**严格 `== "1"`**，`chain_dev.rs:2432`；默认 OFF） |
| **接线点** | `attention_rows` @`chain_dev.rs:10790`：`k2_took = if attn_mrows_rope_norm() && mrows && self.dev.supports_gemm_fp8_mrows_rope_norm()` → `Self::mrows_rope_norm(...)`@10792 |
| **替换掉什么** | 现状 q 链尾 = `norm_rows(qr_r)`(1) + `quant_rows(qr_r)`(1) + `proj_mrows(wq_b)`(1) + q rope 逐行(**6**) = **9 发/层** → K2 **1 发** + 强制保留的 `norm_rows` **1 发** = **2 发/层** ⇒ **−7 发/层 = −280 发/步** |
| **⚠️ 陷阱（读码确认）** | K2 的 `qr_norm_out` **必须传 null**（`mrows_rope_norm`@4719-4727 的 FIX）：kernel 的 prologue 在每个 block 都跑，`Some(qr_r)` 会让 128 个 resident block 互相 RAW/WAR，且写量 128× ⇒ **9.2 tok/s cliff**。写回交给后续 `norm_rows`（`:10843-10858` 的 FIX 明确「K2 也要跑 norm_rows」）。A/B 时若发现 K2 反而变慢，先查这里有没有被改回 `Some` |
| **预期** | **launch 账 −0.92ms @3.3µs / −1.74ms @6.2µs**（含 q rope 的 6 发）；设计口径 −0.8~1.2ms ✓ 一致 |
| **位等价** | ✅ by construction：kernel header 声明每 segment 逐字复刻 `dsv41_rmsnorm_rows_kernel` / `quant_kernel<0>` / `gemm_fp8_mrows_kernel<M>` / `apply_rope_mrows_kernel`；`pos_rows` 是显式 device 数组（无 `pos_ctr`） |
| **验证** | ① `nm -D $SO \| grep -c dsv41_gemm_fp8_mrows_rope_norm` ≥1；② nsys sum 表出现 **`dsv41_gemm_fp8_mrows_rope_norm` 的 kernel 名**（这是唯一能证 template/launcher 真跑的证据——decline 是**静默**的 `Ok(false)`）；③ `grep -c "proj_mrows"` 的 fp8 GEMV Instances/步 从 40 掉到 0（wq_b 那一路）；④ `k_acc` 逐位不变 + 计数数字顺序 + 前 61 行 |

---

### B6 —— `dsv41_gemm_fp8_mrows_f32`（wo_b 直读 f32）· **需新核，ROI 第二**

| 项 | 内容 |
|---|---|
| **gate** | `DSV41_VERIFY_WOB_MROWS_F32=1`（默认 OFF；照 `b6-mrows-f32-design §5.1`，**不另起名**） |
| **接线点** | `attention_rows` @`chain_dev.rs:11616`：现 `let took_wob = if mrows { for r in 0..m { quant_fp8(...) } ; proj_mrows(wo_b) }`；B6 插到**最前**：`if Self::wob_mrows_f32() && supports_gemm_fp8_mrows_f32() && mrows { gemm_fp8_mrows_f32(wo_r, wo_b, wo_b_scale, null, wo_out_r, m, dim, ol_local, /*a_stride=*/ol_total, dim) }`（照设计 §5.2 逐字） |
| **替换掉什么** | 现状 wo_b = 逐行 `quant_fp8`(**6**) + `proj_mrows`(1) = **7 发/层** → **1 发/层** ⇒ **−6 发/层 = −240 发/步** |
| **⚠️ ABI 扩展（必须）** | 新增 `a_stride` 参数：`wo_r` 的**真实行距是 `ol_total`**（8× `ol_local` under TP8），而 `k = ol_local`。现状靠调用方**逐行打包**兜（`:11617-11631` 的 "ROW STRIDE FIX"）；B6 把这个错配升成显式参数并**写进 launcher decline 表**（`a_stride >= k`）⇒ 一次性消灭 `dspark-correctness-chain` 根因 F1/F2 的复发面 |
| **预期** | launch 账 **−0.79ms @3.3µs / −1.49ms @6.2µs**；设计口径 −0.66~1.5ms ✓ 一致。**这是 B 类里省发最多的一项** |
| **位等价** | ❌ **非位等价**（跳过 fp8 往返，严格更准），与 E8 `DSV41_WOB_F32` 同语义。⇒ 验收走**红线**，不走 memcmp |
| **验证** | ① `nm -D` 符号；② nsys 出现 `gemm_fp8_mrows_f32_kernel<6>` 且 `quant_fp8` 的 wo_b 调用数**归零**（设计 §6 的「gate 不变性」判据：`dsv41_quant_fp8` 计数 = 0）；③ 红线：计数数字顺序 + 出师表零拉丁 + `k_acc` 直方图 mode 不降；④ 新 `tests_dsv41_gemm_mrows_f32.cu` 的 raw-u32 parity vs `gemm_fp8_mx_f32`（设计 §6 六臂） |
| **陈旧读者检查（接线红线）** | 现状 `mrows` 臂把 wo_b 的 fp8 激活写进 `xq_r`/`xsc_r`；B6 不再写。**已核**（设计 §5.2）：`attention_rows` 内 `xq_r` 的最后一次使用就是那次 quant；后续 `moe_rows` 的共享专家**自己先 `quant_rows` 再读** ⇒ 跳过安全。**实现时用 `grep -n xq_r` 复核一遍，不凭文档** |

---

### B1 —— q rope 折叠（`VERIFY_ROPE_MROWS`）· **零代码，但被 B2 覆盖**

| 项 | 内容 |
|---|---|
| **gate** | `DSV41_VERIFY_ROPE_MROWS=1`（**或** `DSV41_ROW_FOLD_ROPE=1`；`verify_rope_mrows()`@1207 OR 两者） |
| **接线点** | `attention_rows` @`chain_dev.rs:10908-10922`：`q_roped = q_norm_fused \|\| (verify_rope_mrows() && apply_rope_mrows(q_r, ..., m, nlh, nh*hd, hd, rd, half, pos_rows, false))` |
| **替换掉什么** | q rope 逐行 **6 发/层** → **1 发/层** ⇒ **−5 发/层 = −200 发/步** |
| **⚠️ 与 B2 互斥（本文件 §0-2）** | `q_roped = q_norm_fused \|\| ...`：**B2 上场时 `q_norm_fused = true` ⇒ 本行整段短路，B1 的增量 = 0**。⇒ B1 的正确定位是 **B2 的 fallback / 分步回退**，不是独立收益项 |
| **预期** | 单独上：launch 账 **−0.66ms @3.3µs / −1.24ms @6.2µs**；与 B2 同开：**0** |
| **位等价** | ✅ 行独立（`apply_rope_mrows_kernel`@`dsv41_kernels.cu:2005`，`t = pos_rows[r]`，与 `apply_rope` 的 `off=r, step=0` 同值） |
| **验证** | ① nsys 出现 `apply_rope_mrows_kernel`（且 q 侧 `apply_rope_kernel` Instances/步 从 240 掉到 0）；② 与 B2 **分两轮**跑（一 gate 一 serve，`OnceLock`），否则无法归因 |

---

### B5 —— `ferrite_gemv_bf16_v2_mrows_route`（gate + route 单发）· **需新核**

| 项 | 内容 |
|---|---|
| **gate** | `DSV41_GATE_MROWS_ROUTE=1`（**新 gate**，默认 OFF；沿用 `DSV41_MIX_GATE`/`DSV41_ROW_FOLD_GATE` 的命名法） |
| **接线点** | `moe_rows` @`chain_dev.rs:12963-12995`：现状 `gate_folded = row_fold_gate() && gemv_bf16_v2_mrows(...)`（1 发）+ `route_topk(rows=m)`（1 发）；B5 把 route 的 **last-block 选举**折进 mrows gate（镜像 EAGER 的 `ferrite_gemv_bf16_v2_route`@`device.rs:4604`） |
| **替换掉什么** | **2 发/层 → 1 发/层** ⇒ **−1 发/层 = −40 发/步** |
| **仿写骨架** | EAGER 的 `ferrite_gemv_bf16_v2_route` 是现成的 m=1 版（`chain_dev.rs:4587-4620` 注释把「last-block election on `ctr`」讲清了）；B5 = 把该 election 的 `route_topk` body 换成 **`rows` 维**（`route_topk` 已经有 `rows` 参数，所以 body 不用改，只改 election 的行循环） |
| **预期** | launch 账 **−0.13ms @3.3µs / −0.25ms @6.2µs**（**小项**，§0-5：GATE_MROWS 已吃掉 m×gemv） |
| **位等价** | ✅（声称）：route epilogue 的 m 行推广；`ctr` 须 4B zeroed ONCE（照 EAGER 的纪律，图 replay 才干净） |
| **验证** | ① nsys 出现 `ferrite_gemv_bf16_v2_mrows_route`（单核）且 `route_topk` 计数/步 从 40 掉到 0；② `k_acc` 逐位；③ 计数顺序 + 前 61 行 |

---

### B3 —— `mrows2`（K1，wq_a + wkv 双 family 单发）· **零代码**

| 项 | 内容 |
|---|---|
| **gate** | `DSV41_ATTN_MROWS2=1`（**严格 `== "1"`**，`chain_dev.rs:2365`；默认 OFF） |
| **接线点** | `attention_rows` @`chain_dev.rs:10595-10648`：`mrows2 = attn_mrows2() && supports_gemm_fp8_mrows2() && !swapab() && m<=VERIFY_ROWS` → `Self::proj_mrows2(...)`@10632 |
| **替换掉什么** | 现状 `quant_rows`(1，与 mrows 臂共享) + `proj_mrows`×2(2) = **3 发/层** → **2 发/层** ⇒ **−1 发/层 = −40 发/步**（**只省两个 family 合成一发**；staging 那发两臂共用，不省） |
| **预期** | launch 账 **−0.13ms @3.3µs / −0.25ms @6.2µs**（小项） |
| **位等价** | ✅：`gemm_fp8_mrows2` = `gemm_fp8_mrows_kernel<M>` + family 折叠；row `row` of family `f` 逐位等于被替换的那一发 |
| **⚠️ 与 R2 同开时** | `lin2_gate`（R2 单行臂）在 `m>1` 恒 false ⇒ 无冲突。但 **K1 排在 `lin2` 之前**（`:10592` 注释），两者同开 K1 赢 |
| **验证** | ① nsys 出现 `dsv41_gemm_fp8_mrows2` 且 fp8 GEMV 的 Instances/步 从 80（wq_a+wkv 各 40）掉到 40；② `k_acc` 逐位 |

---

### B4 —— `dsv41_rmsnorm_rope_mrows`（kv norm + rope）· **需新核，ROI 最后**

| 项 | 内容 |
|---|---|
| **gate** | `DSV41_RMSNORM_ROPE_MROWS=1`（新 gate，默认 OFF） |
| **接线点** | `attention_rows` @`chain_dev.rs:10945-10968`：现状 `norm_rows_on(kv_r, ..., m, hd, eps, kv_stream)`(1) + `apply_rope_on(kv_r, ..., rows=m, step=1, ..., kv_stream)`(1) → 合 1 发 |
| **替换掉什么** | **2 发/层 → 1 发/层** ⇒ **−1 发/层 = −40 发/步** |
| **预期** | launch 账 **−0.13ms @3.3µs / −0.25ms @6.2µs**（小项；设计要求更保守） |
| **⚠️ VERIFY_FORK 交互** | 这两发现在是 `norm_rows_on` / `apply_rope_on`，**带显式 stream**——B4 的新核必须接 `kv_stream`（`VERIFY_FORK` 的 kv 半链），否则会**静默地把 kv 半链拖回主流**，抵消 FORK 的收益。核的 ABI 必须是 `..._on(..., cudaStream_t)` |
| **位等价** | ✅ 行独立（每行只 norm+rotate 自己的 `hd`） |
| **验证** | ① nsys 出现新核名；② `apply_rope_kernel` 的 kv 侧 Instances/步 从 40 掉到 0；③ **A/B 必须与 `VERIFY_FORK` 同臂**（FORK=1 时测，否则掩盖 stream 回归） |

---

## 4. 实施顺序与 ROI（对任务序的**两点修正**）

任务序：`B2 > B6 > B1 > B5 > B3 > B4`。按「兑现 ms × 兑现概率 ÷（人日 × 风险）」重排：

| 序 | 项 | 代码 | 省发/步 | launch 账 @3.3µs / @6.2µs | 兑现概率依据 | 一句理由 |
|---:|---|---|---:|---|---|---|
| **1** | **B2**（K2） | **0** | −280 | **−0.92 / −1.74ms** | 中：位等价已论证；K1/K2 是「已编译待验」 | 最大单项 + 零代码 ⇒ 无条件先做 |
| **2** | **B6** | 新核 0.5 人日 | −240 | **−0.79 / −1.49ms** | 中：非位等价，需红线 | 省发第二多；**写码可与 1/3 的 GPU A/B 并行，不占 GPU** |
| **3** | **B5** | 新核 | −40 | −0.13 / −0.25ms | 中：有 m=1 版可仿 | 小项，但**同一趟 GPU 会话可顺手测** |
| **4** | **B3**（K1） | **0** | −40 | −0.13 / −0.25ms | 中：位等价 | **零代码 ⇒ 应排在 B5 之前**（任务把它排在 B5 后，按成本应前移） |
| **5** | **B1** | **0** | −200（**B2 后 = 0**） | −0.66 / −1.24ms（B2 前） | 低：被 B2 覆盖 | 定位 = B2 的 fallback 与分步回退；**不与 B2 同轮测** |
| **6** | **B4** | 新核 | −40 | −0.13 / −0.25ms | 低：靶子被 NORM_MROWS 削半 + FORK stream 风险 | 性价比最低，收尾 |

**修正 1**：**B1 与 B2 必须分轮、（更正确地说）B2 一旦上场 B1 不再有独立收益**（§0-2）。
若把 B1 当成 B2 之外的 −0.4~0.7ms 编进预算，就是重复计账。
**修正 2**：**B3（零代码）应排在 B5（新核）之前**——同样的 −40 发/步，前者成本 ≈0。

**族合计（不重复计账）**：
```
Σ = B2 + B6 + B5 + B3 + B4        （B1 被 B2 覆盖）
  = −280 −240 −40 −40 −40 = −640 发/步
  = −2.11ms @3.3µs  ／  −3.97ms @6.2µs
```
对照任务给的 `S2 = −4.5ms`（设计口径）与吞吐计划的 `−5.15ms（mid）`：
* @6.2µs 全账 `−3.97ms` ≈ 设计 mid 的 **77%**；
* @3.3µs 保守 `−2.11ms` ≈ 设计 mid 的 **41%**——**与本仓历史兑现率 60% 一致偏低**。
⇒ **结论：S2 的现实票面约 −2.1 ~ −4.0ms，中位 −3.0ms**（不是 −4.5）。且天花板是 nsys 的
「gemv 投影」族 **4.2ms**（§0-6）——**账与天花板重合，无余量**。

---

## 5. 验证矩阵（一次 GPU 会话，背靠背交错）

**通用纪律**（照 `batched_400_v2.sh` 与 `swallow-nsys-batched-analysis-framework §7`）：
1. **一 gate 一 serve**（`OnceLock` 每进程只读一次；多 gate 同开 ⇒ 失去归因）。
2. **一 prompt 一 serve**（`[dspark] steps=` 累加器跨请求不清零）。
3. **交错 A B A B**（抵消热漂）。
4. **`V5_LEDGER=0`**（吞吐轮；10 个 D2H 同步点/步会吃掉全部位移）。
5. **禁** `LAZY_VERIFY` / `SEED_ALIGN` / `HC_*`（抢臂/串味，`FORBIDDEN:175`）。

**每臂必录（缺一不能下结论）**：
| # | 证据 | 命令 / 判据 |
|---|---|---|
| V1 | **env 实读**（防「设了没生效」——本仓 #1 陷阱） | `tr '\0' '\n' < /proc/$(pgrep -x ferrite-serve)/environ \| grep -E '<GATE>'` |
| V2 | **符号存在性** | `nm -D $SO \| grep -c <symbol>`（B2/B3 已有；B4/B5/B6 需先重建） |
| V3 | **上场证据（唯一硬证）** | nsys sum 表：`gemm_fp8_mrows_rope_norm`(B2) / `gemm_fp8_mrows_f32_kernel<6>`(B6) / `ferrite_gemv_bf16_v2_mrows_route`(B5) / `gemm_fp8_mrows2`(B3) / `apply_rope_mrows_kernel`(B1) / 新核(B4)。**decline 是静默的 `Ok(false)`，没有核名就不能说「已上场」** |
| V4 | **步时** | `[dspark] steps=` 的 `verify_ms` / `steady_median`（丢前 10 步；skip=20） |
| V5 | **正确性** | 计数：数字顺序 + 前 61 行；出师表：**零拉丁** + `先帝创业未半` + 无双字 |
| V6 | **逐位**（B1/B2/B3/B4/B5） | `k_acc` 序列**逐位不变**（bit-identical by construction） |
| V7 | **红线**（B6 专用） | 不比对逐位；查 `quant_fp8` 的 wo_b 调用数归零 + 红线 + `k_acc` mode 不降 |

**判据（每项 |Δsteady_median| 的双门）**：
* **保守门 ≥ −0.8ms**：`swallow-unlocked-shpair-m6-throughput-plan §3.3` 的止损线
  （< 设计 40% ⇒ 判 instruction-bound，立刻转下一项）。理由：`SH_EXP_MROWS` **两次零收益**先例。
* **期望窗**：`|Δ| ∈ [launch 账 × 60%, launch 账]` 视为「正常兑现」；
  `Δ > 0`（变慢）⇒ 先查 **§3 各项的「⚠️ 陷阱」**（K2 的 `qr_norm_out` / B6 的 `a_stride` /
  B4 的 `kv_stream`），不要先怀疑测量。

---

## 6. 与 S2 预算的对账（一句话）

> **mrows 族的 S2 = 3 个零代码接线（B1/B2/B3）+ 3 个新核（B4/B5/B6）；
> B1 ⊂ B2（不可相加），B4/B5 的靶子已被 NORM_MROWS/GATE_MROWS 削半（各 −40 发/步）。
> 不重复计账的合计 = −640 发/步 = −2.1ms(保守) ~ −4.0ms(全账)，中位 ≈ −3.0ms；
> 天花板是 nsys「gemv 投影」族的 4.2ms。⇒ S2 的现实票面是「−3ms 量级」，
> 而 400 需要 Σ(S1..S5) 的 97% 兑现（`swallow-unlocked-shpair-m6-throughput-plan §5.3`）——
> S2 单靠自己不够，必须与 SH_PAIR M=6 / hc 链并联，且每一项都要有 kernel 名证据。**

---

## 附：任务前提修正汇总（报尚书省）

| # | 任务表述 | 代码事实 | 依据 |
|---|---|---|---|
| 1 | 「mrows 清单 B1–B6」= 6 个并列的 mrows 项 | B1/B2/B3 是**零代码接线**（kernel 已在树）；B4/B5/B6 是**新核**；且 B1 ⊆ B2 | §2 符号表、§3 |
| 2 | 「B2(-0.8~1.2) + B1(-0.4~0.7)」可加 | **不可加**：K2 上场后 `q_norm_fused` 短路 B1 的整个调用点 | `chain_dev.rs:10810`、`:10908` |
| 3 | 「B1 mrows_rope (wq_b) 在 SWALLOW 有效」 | wq_b 的 `proj_mrows` **本来就在跑**（`mrows` 无 gate）；B1 的增量只剩 **q rope 折叠** | `chain_dev.rs:10588` |
| 4 | 「B5 mrows_route −0.3~0.6ms / B4 −0.2~0.4ms」 | 各只剩 **−1 发/层 = −40 发/步 = −0.13ms@3.3µs**（GATE_MROWS/NORM_MROWS 已吃掉 m× 部分） | `chain_dev.rs:10945-10995`、`:12963-12995` |
| 5 | 「实施序 B2>B6>B1>B5>B3>B4」 | 按成本应把 **B3（零代码）前移到 B5（新核）之前**；B1 应标注为「B2 的 fallback，不与 B2 同轮」 | §4 |
| 6 | 6 个 mrows gate 里的 `SH_EXP_MROWS`/`GATE_MROWS`/`INDEXER_MROWS`/`NORM_MROWS`/`COMPRESSOR_MROWS`/`VERIFY_HEAD_MROWS` | **已在权威配置里**，58.3 tok/s 基线已带 ⇒ **不计入 S2 增量** | `batched_400_v2.sh:145-156` |

---

## 7. 工部实施记录（2026-09-12 · Phase A 第一批）

**结论：Phase A 的 3 个「零代码接线」在 HEAD 已全部落树、已连在 batched 路径上 ⇒ 引擎侧无需新增代码。**
勘察逐点核对设计 §3 的接线点（行号见下），`cargo check --workspace` EXIT=0。

| 项 | gate（默认 OFF） | 接线点（均在 `attention_rows` = batched 路径） | kernel / launcher | 落树 |
|---|---|---|---|---|
| B2 | `DSV41_ATTN_MROWS_ROPE_NORM` | `chain_dev.rs:10790-10809`（`k2_took`）→ `Self::mrows_rope_norm`@4696 | `dsv41_gemm_fp8_mrows_rope_norm`@`dsv41_kernels.cu:6006` | `97ce72e` |
| B3 | `DSV41_ATTN_MROWS2` | `chain_dev.rs:10595-10648`（`mrows2`/`m2_ok`）→ `Self::proj_mrows2`@4769 | `dsv41_gemm_fp8_mrows2`@`:5650` | `196a7d6` + `a04ab3f` |
| B1 | `DSV41_VERIFY_ROPE_MROWS` | `chain_dev.rs:10908-10922`（`q_roped`） | `dsv41_apply_rope_mrows`@`:9182` | `626251e` / `75c1c15` |

两点与设计一致的**代码事实复核**（不是文档转述）：
* K2 的 `qr_norm_out` 确为 `null`（`chain_dev.rs:4729`），且后续 `norm_rows` 在 `k2_took` 时仍会跑（`qr_raw_r = q_norm_fused && !k2_took && …`，`:10836`）——即 §3 的陷阱已被 FIX 覆盖；
* 三个 gate 的读点全部落在 `step_rows`→`layer_rows`→`attention_rows`（`:6042`/`:10261`/`:10388`/`:10558`）内，满足 §1.1 的 M 判据。

**真正缺失的是「把 gate 送到节点上」这一段接线**：serve 经 `rssh`（`ssh NODE "<quoted>"`）拉起，其 env 只由 `$GATES_ONELINE` 构成 ⇒ 本机 `export` 到不了节点，而 decline 是静默的（`Ok(false)`），正是 Y1/V1 说的「设了没生效」。
⇒ 已在 `scripts/batched_400_v2.sh` 加 opt-in arm（默认 unset ⇒ GATES 逐字不变）：
`B400_MROWS_A=b2|b3|b1`，一次只允许一个 gate（B1 与 B2 必须分 serve 跑，§0-2）。arm 的 gate 进入 `$GATES_ONELINE`，因此 `-- gates:` banner、`$LOGDIR/run.env` 的 `/proc/<pid>/environ` 实读、FORBIDDEN 校验都自动覆盖它。

*工部 · 本段为实施记录；phase A 无 GPU 命令执行，验收只到 `cargo check` + `bash -n`。*

---

## 8. 工部实施记录（2026-09-12 · Phase B 第二批：B5 + B4）

**结论：本设计 §3 的 B5 / B4 两个「新核」已按 B6 的实施模式全部落树 —— kernel + launcher + device.rs FFI/probe + `chain_dev.rs` gate/接线 + parity suite + 脚本 opt-in arm；两个 gate 默认 OFF，`cargo check --workspace` EXIT=0。**

实施模式完全照 B6（§0-1 / `b6-mrows-f32-design.md`）：**不换程序、只搬代码** —— 新核的每一段决定结果的表达式都逐字复制自它在用的现有 kernel，逐位等价由构造保证（parity suite 在 GPU 上钉死）。

### B5 —— `ferrite_gemv_bf16_v2_mrows_route`（gate GEMV + route 单发）

| 项 | 内容 |
|---|---|
| **gate** | `DSV41_GATE_MROWS_ROUTE=1`（严格 `== "1"`，默认 OFF）**且** `row_fold_gate()`（`DSV41_GATE_MROWS`/`DSV41_ROW_FOLD_GATE`）——融合就是多行程序本身，没有逐行形态可折 |
| **kernel** | `ferrite_kernels.cu`：把 `Gv2RouteEpi` 加 `nrows` 字段、把 `gv2_route_epilogue` 的「rows == 1」硬编码改成 `for (rr < epi.nrows)` 行循环（**同一份 `route_topk` body，v2 入口传 `nrows = 1` ⇒ 行为逐字不变**），给 `gemv_bf16_nt_kernel<NT, WPR>` 加 `rte` 形参（null ctr ⇒ 无 barrier/atomic/smem，逐字不变），新增 C 入口 `ferrite_gemv_bf16_v2_mrows_route`（`nrows == 1` 转发到 M=1 的 `ferrite_gemv_bf16_v2_route`） |
| **程序复用（关键）** | B5 实例化的是**同一个** `gemv_bf16_nt_kernel<NT, WPR>`（`NT×WPR` 双重 dispatch 与 `ferrite_gemv_bf16_nt` 逐字相同，只有 epilogue 不同）⇒ 满足该入口自己的「**NO SECOND TRANSCRIPTION**」铁律，不产生第二份 K 序定义 |
| **接线点** | `moe_rows` @`chain_dev.rs`（`gate_folded` 块）：`row_fold_gate() && gate_mrows_route()` → 融合入口；返回 `Ok(false)`（stale `.so`/shape/smem）⇒ **回落到 `gemv_bf16_v2_mrows` + `route_topk` 两发**（即参考） |
| **新计数器** | `Scratch::route_ctr_r`（4B，**与 EAGER 的 `route_ctr` 分开**：两个 gate 可同开，一个 counter 服务两个 election 会双双损坏），build 时 zero **一次**，kernel 自己复位（graph replay 干净） |
| **替换掉什么** | **2 发/层 → 1 发/层** ⇒ **−1 发/层 = −40 发/步**（设计 §0-5：`GATE_MROWS` 已吃掉 m×gemv） |
| **位等价** | ✅ by construction（GEMV 半 = 同一程序 + epilogue 在其 block 的 `y[]` store 之后；route 半 = `route_topk_kernel` 的 body 逐行，MAX 选择 + 确定性 tie-break ⇒ reduce 树形状不影响结果） |
| **parity** | `kernels/cuda/tests_mrows_route.cu`：融合 vs `gemv_bf16_v2_mrows` + `dsv41_route_topk`，**scores / weights / indices 三个缓冲区整块 raw-bit 比对** + qNaN sentinel 覆盖；m=2/5/6、score_func=2/0、topk=6/8、WPR=4/8、k 非 `32*8*WPR` 倍数 |

### B4 —— `dsv41_rmsnorm_rope_mrows`（kv norm + rope 单发）

| 项 | 内容 |
|---|---|
| **gate** | `DSV41_RMSNORM_ROPE_MROWS=1`（严格 `== "1"`，默认 OFF） |
| **kernel** | `dsv41_kernels.cu`（紧接 `dsv41_rmsnorm_rows`）：`dsv41_rmsnorm_rope_mrows_kernel` = **phase 1 `dsv41_rmsnorm_rows_kernel` 的 body 逐字**（含 1024 线程的 `red[32]` 跨 warp 折叠 —— blockDim 是逐位等价的必要条据）+ **phase 2 `apply_rope_kernel` 的 trailing `2*half` 旋转逐字**（`t = pos_rows[row]`）；ABI 带 **stream** |
| **接线点** | `attention_rows` @`chain_dev.rs`（kv 半链，`norm_rows_on` + `apply_rope_on` 处）：`rmsnorm_rope_mrows()` → 融合入口，`Ok(false)` ⇒ 原两发（`norm_rows_on` + `apply_rope_on`）逐字运行 |
| **⚠️ stream 红线** | 两发都是 `*_on`（`VERIFY_FORK` 的 `kv_stream` 侧链）；融合入口把 stream 作为**参数**（Rust `Device::rmsnorm_rope_mrows(..., s)`），硬件码里写死主流会把 kv 半链悄悄拖回主流、抵消 FORK 收益。**A/B 必须与 `DSV41_VERIFY_FORK` 同臂**（设计 §3 的陷阱已被 ABI 覆盖） |
| **替换掉什么** | **2 发/层 → 1 发/层** ⇒ **−1 发/层 = −40 发/步**（设计 §0-5：`NORM_MROWS` 已吃掉 m×norm；kv rope 本来就是 m 行单发） |
| **位等价** | ✅ by construction：phase 1 行独立 + blockDim 相同；phase 2 无归约、每元素恰好写一次（线程映射不影响结果）；**唯一新增是一条相位间 `__syncthreads()`** —— 只定内存序，不动数值 |
| **parity** | `kernels/cuda/tests_rmsnorm_rope_mrows.cu`：融合 vs `dsv41_rmsnorm_rows` + `dsv41_apply_rope`，整块 raw-bit + qNaN sentinel；非 0 `pos_base`（钉死「用 `pos_rows[row]` 而不是 `row`」）、`rope_off = dim − 2*half`（钉死「旋转尾段而不是头段」）、`inverse=1`（生产不走的那个 arm）、三种 head 宽度 |

### 脚本 opt-in arm（把 gate 送到节点上）

与 §7 同因（serve 经 `rssh`，env 只由 `$GATES_ONELINE` 构成）：`scripts/batched_400_v2.sh` 加两个 arm，**默认 unset ⇒ `GATES_ONELINE` 逐字不变**：

* `B400_B5=1` → `DSV41_GATE_MROWS=1 DSV41_GATE_MROWS_ROUTE=1`
* `B400_B4=1` → `DSV41_RMSNORM_ROPE_MROWS=1`

### 本次验收（无 GPU / 无 nvcc ⇒ 只到源码级）

| 判据 | 结果 |
|---|---|
| `cargo check --workspace` | ✅ **EXIT=0** |
| `bash -n scripts/batched_400_v2.sh` | ✅ OK |
| gate OFF 默认行为不变 | ✅ B5：`row_fold_gate()` false ⇒ `route_fused=false` ∧ `gate_folded=false` ⇒ **逐行 gemv + route_topk 原样**；B4：`kv_fused=false` ⇒ **`norm_rows_on` + `apply_rope_on` 原样**（两处 fallback 就是参考实现本身） |
| 符号/接线完整性 | ✅ `nm -D` 待 GPU 节点重建后核（V2）；两条新符号已进 `kernels` 表与 `ko!` |

**留给 GPU 会话的三件事**（设计 §5 的判据，本次无 GPU 不能做）：
1. 重建 `.so`（`bash build.sh 103a`）后 `nm -D` 核两条新符号 —— **decline 是静默的**，没有符号就等于 arm 空转；
2. nsys sum 表出现 `ferrite_gemv_bf16_v2_mrows_route` / `dsv41_rmsnorm_rope_mrows` 的 **kernel 名**（这是唯一「真上场」的硬证），且 `route_topk` / kv 侧 `apply_rope_kernel` 的 Instances/步 各从 40 掉到 0；
3. `k_acc` 逐位 + 计数数字顺序 + 出师表零拉丁（两项都声称位等价，所以走 V6 而不是红线）；B4 必须与 `DSV41_VERIFY_FORK` 同臂。

*工部 · 本段为实施记录；本批无 GPU 命令执行，验收只到 `cargo check` + `bash -n`。*

---

*工部 · 只读勘察 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*所有 ms 标来源（实测 / launch 账 / 设计 / 代数）；行号以 HEAD `ba88df1` 为准；*
*与任务前提冲突处已显式给出依据与 file:line。*
