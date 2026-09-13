# hc 链的「m 行批量化」审计 —— 生产形态判定 + rows-化判定（判决：**不需要 rows 化**，缺的是 dispatch 不是 shape）

> 工部 · 2026-09-13 · 只读勘察（+ 一条**上报尚书省**的潜在事故）。基线 HEAD `96aea15`（工作树有 3 个已改文件：
> `chain_dev.rs` / `dspark_dev.rs` / `dsv41_kernels.cu`，均非本次触碰）。**未跑 GPU/e2e**；远端 `nvcc compile-only` 已跑。
> 依据：`docs/agent/sglang-verify-model.md`、`verify-amortization-lesion-audit.md` §10.1/§10.2/§10.3、`g3-per-row-batched-audit.md`。

**本次回答的四问**：①生产形态判定（raw 30272 的来源 file:line）②rows 化 diff（**空**，逐位论证）③cargo check + nvcc ④GPU 验证手册。

---

## 0. 三条结论（先看）

1. **hc 链的每一个 kernel 都已经是 rows-native**（`hc_mixes_kernel` grid=rows；`hc_mix_dots_kernel` / `hc_dots_late_kernel`
   grid=(mix, rows)；`hc_mixes_tail_kernel` grid=rows；`hc_front_kernel` grid=(mix+1, rows)；`hc_post_inplace_rows` grid=(x, rows)）。
   `rows = m` 的一次调用**就是**整块 m 行 —— 本任务书假设的「按行组织、需要 m 行批量化」在 hc 族上**不成立**。
   缺的那一半是 **dispatch**（哪条路径被发射），不是 **shape**（kernel 认不认 m）。
2. **raw `hc_mixes_kernel` 的 86 发/步 = 80（verify，仅当 `DSV41_HC_FRONT_ROWS` 未进进程时）+ 6（draft，无条件）**。
   80 那半是 v6 nsys 脚本的 env 丢失假象（§10.1 已记）；**6 那半是生产真实的 raw**，与 AR_SAFE 无关
   —— 而且**开了 HC_FRONT_ROWS 也消不掉**（draft 不经过 `hc_mixes_auto`）。
3. ⚠️ **上报**：`hc_front_kernel`（`DSV41_HC_MERGE=1`）在 `rows = m` 下有**共驻性隐患**（grid=(mix+1, rows)=150 > 148 SM，
   每块 160 KiB smem ⇒ 1 块/SM；tail 块的 ticket spin 会与它等待的 dot 块抢 SM），而 kernel 头注释仍写
   「`rows == 1 at every current call site`」——**该注释已过期**（`hc_mixes_auto` 的 else 分支在 rows=m 时就走它）。
   两个门都 opt-in（`DSV41_HC_MERGE=1` + `HC_FRONT_ROWS=1`）故未爆；**未改代码，交尚书省裁决**。

---

## 1. 生产形态判定（交付 ①）

### 1.1 hc 的调用图（file:line 全链路）

| 路径 | 入口 | 发射点 | 门 |
|---|---|---|---|
| **verify**（m 行） | `chain_dev.rs:11786` / `:11864`（`layer_rows` 的 attn/ffn 两处） | `if Self::hc_front_rows()` → `hc_mixes_auto`；**else → `self.dev.hc_mixes(...)`**（`:11816-11829` / `:11890-11903`） | `DSV41_HC_FRONT_ROWS`，**默认 OFF**（`chain_dev.rs:16249-16252`，`unwrap_or(false)`） |
| **draft**（bs 行） | `dspark_dev.rs:1502` / `:1668`（`draft_body` 每 block 两处） | `if draft_p3lite().hc_front` → `hc_front_fused`；**else → `DsparkDev::hc_mixes`**（`:1907-1943`）→ **`self.dev.hc_mixes(...)` 直调**（`:1929`） | `DSV41_P3LITE_HC_FRONT` / `DSV41_DRAFT_P3LITE_A_SEG`（`dspark_dev.rs:579-582`），默认走 master `DSV41_DRAFT_P3LITE`（默认 OFF） |
| **单行 decode**（anchor 步） | `chain_dev.rs:16385` / `:16505`（`layer()`） | **无条件**走 `hc_mixes_auto`（rows=1） | — |

`hc_mixes_auto`（`chain_dev.rs:15927`）的四臂：`hc_front_split`（`:15990`）→ 降级 `hc_front`（`:16020`）→ `rows==1` 的 `persist_mb`/`persist` → **else 落到 raw `hc_mixes`**（`:16115`）。

### 1.2 86 发/步的精确分解

```
86 = 40 层 × 2（attn+ffn，layer_rows）        ← verify：仅当 HC_FRONT_ROWS 未进进程时才是 raw
   +  3 块 × 2（attn+ffn，draft_body）        ← draft：无条件 raw（除非 P3LITE hc_front 开）
```
`n_mtp_layers = 3`（`config.rs:520/:612` 断言），`n_layers = 40` ⇒ **80 + 6 = 86**，与 nsys 的 30272/352(步) 逐位吻合。

- **80 那半**：§10.1 已定谳 —— v6 nsys 脚本（`env -u ... nsys profile ...`）env 传递丢失，`DSV41_HC_FRONT_ROWS` 没进进程
  ⇒ `layer_rows` 走 else 分支的 raw `hc_mixes`（`chain_dev.rs:11816`），**一次调用一个 launch**（见 §2：grid=rows，6 块）。
  best 系列的 `/proc/<pid>/environ` 回读里 `HC_FRONT_ROWS=1` 在册 ⇒ **生产不是这个口径**。
- **6 那半**：`DsparkDev::hc_mixes` 是**直调 `dev.hc_mixes`**（`dspark_dev.rs:1929`），不走 `hc_mixes_auto`，
  所以 `DSV41_HC_FRONT_ROWS` 与它无关，`hc_tail_split` 也与它无关 —— **开了三开门也照发 6 发/步**。
  这是本次审计里唯一**真实存在**的 raw hc 发射。
  → 消除它只有一条路：开 `DSV41_P3LITE_HC_FRONT=1` / `DSV41_DRAFT_P3LITE_A_SEG=1`（走 `draft_hc_front`，
  `dsv41_kernels.cu:12151`），那是 draft 侧的另一条融合臂（自带 parity 前置），不属本任务。

### 1.3 三开门下的 verify 形态

`hc_mixes_auto` 的分派条件与命中（rows = m = 6，生产栈）：

| 臂 | 条件 | 生产（三开门） | 落到 |
|---|---|---|---|
| `hc_front_split` | `hc_tail_split()`（默认 ON, `:16339-16342`）+ `supports_hc_tail_split()`（符号 + `side_stream`/`fork_event`/`join_event`, `device.rs:7511-7516`）+ `norm_w != null`（FUSE_B1 默认 ON ⇒ `layer_rows` 传 attn_norm/ffn_norm） | ✅ 命中 | `dsv41_hc_front_split`（`dsv41_kernels.cu:13776`）→ `hc_mixes_tail(EARLY)` + `hc_dots_late`（`:13932`）/ 或 DL_MERGE=0 时的 dots+tail 两发 |
| ↓ 若 split decline | 形状/符号/流不满足 | — | `Self::hc_front_note("…declined -> hc_front (two-launch)")` + `hc_front`（`:13622` 两发）—— **仍是融合链，不是 raw** |
| `persist_mb` / `persist` | 都要 `rows == 1` | ❌（rows=6）跳过 | — |
| else → `hc_front` | — | — | 两发融合 |
| **raw `hc_mixes`** | `fused == false` 时唯一出口 | **不可达**（split 或 hc_front 必有一个返回 true） | — |

**⇒ 结论：`HC_FRONT_ROWS=1` 的 verify 不发 raw `hc_mixes_kernel`。** §10.3 的「融合 ARMED 后 raw 30272 是 AR_SAFE 假象」
在此被**代码级确认**：raw 的出口在 `hc_mixes_auto` 的最后一跳（`chain_dev.rs:16115`），而 split/`hc_front` 在
生产形状下都不会 decline（`g_hc_front` 默认 ON、`mix = hc*(2+hc) = 24 ≤ 64`、`rows = 6 ≤ 2048`）。
**verify 侧的真实生产发射 = 每层 per front：`hc_mixes_tail_kernel`(EARLY, grid=rows) + `hc_dots_late_kernel`(grid=(mix,rows))**。

### 1.4 `hc_dots_late` 的「112 发/步」对不上（诚实标注）

`hc_dots_late_kernel` 全树**只有一个发射点**（`dsv41_kernels.cu:13932`，在 `dsv41_hc_front_split` 内；
唯一调用者是 `chain_dev.rs:15990` 的 `hc_mixes_auto`）。所以每步发数 = 「走 split 的 `hc_mixes_auto` 调用数」：

- 只有 m 行 verify 走：**80**（40 层 ×2）
- 加上单行 anchor 链（`layer()` 无条件走 `hc_mixes_auto`，split 默认 ON）：**+80 = 160**

**112 与两者都不吻合**。而 `dspark-correctness-chain.md:2986` 记的是 `hc_dots_late_kernel` **总 8417 发**
（14.9µs/发）；8417/112 ≈ 75 步、8417/86 ≠ 352 —— 即任务书里「×86/步」与「×112/步」**不是同一个 step 基数**。
→ **该发数需在修正 env（`/proc` 回读断言）后重数**；本审计只给出「唯一起点 + 每调用恰好 1 发」这一确定事实。

---

## 2. rows 化判定：**不需要**（交付 ②，逐位论证）

### 2.1 全链 kernel 的 rows 支持（签名级证据）

| kernel | file:line | row 维 | 逐行寻址 | launcher grid |
|---|---|---|---|---|
| `hc_mixes_kernel` | `dsv41_kernels.cu:2579` | `const int r = blockIdx.x;`（`:2585`） | `x + r*hc_dim`、`pre + r*hc`、`comb + r*hc*hc` | `:11701` `<<<rows, mix*32, smem>>>` |
| `hc_mix_dots_kernel` | `:12470` | `m = blockIdx.x; r = blockIdx.y`（`:12472-12473`） | `g_hc_part[r][m][…]` | `:13692`/`:13943` `<<<dim3(mix, rows), …>>>` |
| `hc_mixes_tail_kernel` | `:12543` | `const int r = blockIdx.x;`（`:12553`） | `g_hc_part[r][…]`、`pre + r*hc` | `:13696`/`:13859`/`:13956` `<<<rows, …>>>` |
| `hc_dots_late_kernel` | `:13292` | `m = blockIdx.x; r = blockIdx.y`（`:13297-13298`） | `x + r*hc_dim`、`g_hc_part[r][m][…]`、`g_hc_dl_done[r]` | `:13932` `<<<dim3(mix, rows), …>>>` |
| `hc_front_kernel` | `:12728` | `r = blockIdx.y`（`:12738`） | `g_hc_ticket[r]`、`g_hc_part[r][…]` | `:13668` `<<<dim3(mix+1, rows), …>>>` |
| `hc_pre_persist_kernel` | `:12967` | `r = blockIdx.x`（`:12977`） | `g_hc_part[r][…]` | `:14018` `<<<rows, …>>>` |
| `hc_pre_persist_mb_kernel` | `:14074` | `r = blockIdx.y`（`:14086`） | `g_hc_part[r][…]`、`g_hc_mb_done[r]` | `:14312` `<<<dim3(mix*split+1, rows), …>>>` |
| `dsv41_draft_hc_front_kernel` | `:11988` | `r = blockIdx.x`（`:11999`） | `pre + r*hc`、`comb + r*hc*hc` | `:12167` `<<<rows, 1024, …>>>` |
| `hc_post_inplace_rows_kernel` | `:11816` | `row = blockIdx.y`（`:11821`） | `res + row*n*h`、`post + row*n`、`comb + row*n*n` | `:11865` `<<<dim3(…, rows), 256>>>` |
| `hc_collapse_norm_kernel` | `:11896` | `row = blockIdx.x`（`:11901`） | `pre + row*hc`、`x + row*hc*dim` | `:11937` `<<<rows, 1024>>>` |
| `ferrite_hc_post`（staging 形） | `ferrite_kernels.cu:2679` | `t = idx/(n*h4)`（`:2701`） | `post[t*n+i]`、`res+(t*n+k)*h` | `:2721` grid=`total/256`（**m 已在线性 index 里**） |

**每一条的 row 维都只把基址平移 `r × <pitch>`**（`r*hc_dim` / `r*hc` / `r*hc*hc` / `g_hc_part[r][…]`），
没有任何跨行归约、跨行原子或跨行共享槽。唯一「跨块」的状态是 `g_hc_part[r][…]` / `g_hc_dl_done[r]` / `g_hc_ticket[r]`
—— **全部按 r 索引**，即每行一份、互不相交。

**⇒ 逐位论证**：把 `rows = 1` 的调用改成 `rows = m`，只有两个变化 ——（a）grid 的 row 维从 1 变 m；
（b）kernel 内 `r` 的取值域从 `{0}` 变 `{0..m-1}`。对任意固定 `r`，块内语句序列、lane 分组、归约树、
FMA 链、寄存器累加器**逐字不变**（所有行相关量都只经 `r` 进基址）。因此「rows=m 调用的第 r 行」==
「rows=1 调用（`x` 指向 `x + r*hc_dim`）」的**同一段程序**⇒ **逐位相同**。
这与树内既有断言一致（`hc_collapse` 的 row-independence：`dsv41_glue.cu:301-317`；`hc_mix_dots` 的
「same lane assignment, bit-identical」：`:12488-12503`）。

### 2.2 因此本任务的交付是「空 diff」

- **无 kernel 需要 rows 化**（§2.1）。
- **无 wiring 需要加**：`layer_rows` 已经用 `rows = m` 调 `hc_mixes_auto`（`chain_dev.rs:11786`，参数 `m` 在第 21 个位置），
  `hc_mixes_tail_kernel` 的 ss 分组（`c2 = lane + m*32, stride mix*32`）与 `hc_dots_late_kernel` 的
  `g_hc_dl_done[r]` 选举**都已经是 per-row 语义**（`dsv41_kernels.cu:13346/13355/13414`）。
- **票面的真正来源是 fusion 不是 rows**：raw（1 launch，6 块 = 6 SM）`51.4µs` vs 融合
  （EARLY 1.7µs + dots_late 14.9µs，152 块 ≈ 148 SM）—— **3.1×** 来自**并行度从 6 SM → 148 SM**，
  即「spread」而不是「按行批量化」。raw 已经「批量化」了，它只是**块太少**。

---

## 3. ⚠️ 上报尚书省：`hc_front_kernel` 在 rows=m 的共驻性隐患（未改）

**事实**（`dsv41_kernels.cu:12705-12925`）：

- `DSV41_HC_MERGE=1` 时 `dsv41_hc_front`（`:13660`）发射 `hc_front_kernel`，**grid = (mix+1, rows)**（`:13668`），
  block = 1024，dynamic smem = `2*hc_dim*4` = **163840 B = 160 KiB** ⇒ **每 SM 1 块**（opt-in 上界 ~227 KiB）。
- kernel 头注释写「B300 keeps ~296 blocks resident against the `2*(mix+1) = 50` this needs, so the spin always
  resolves. **`rows == 1` at every current call site.**」（`:12725-12727`）。
- **该注释已过期**：`hc_mixes_auto` 的 else 分支（`chain_dev.rs:16090`）在 `rows = m` 时会走 `hc_front`
  —— 即在 **`HC_FRONT_ROWS=1` + `DSV41_HC_TAIL_SPLIT=0`（或 split decline）** 的组合下，`rows = m = 6`
  ⇒ **grid = (25, 6) = 150 块 > 148 SM**，而 1 块/SM ⇒ **2 块排在队尾**。
- 若排队的 2 块里有某行的 **dot 块**，而该行的 **tail 块**已在跑并 spin `g_hc_ticket[r] < mix`
  ⇒ 该 tail 等它、它等 SM —— **~5 s watchdog**（`:12847`）才 break，然后 tail **带着未发布的 part 继续跑**
  ⇒ **静默错答**（不是 hang）。

**为什么现在没爆**：两个门都 opt-in（`DSV41_HC_MERGE=1` 默认 OFF，`HC_FRONT_ROWS` 默认 OFF），
且 nsys/best 系列都用默认 `DSV41_HC_MERGE`。**但这是一个「开了就错」的雷**。

**建议（三选一，交尚书省）**：
1. 在 `dsv41_hc_front` 的 merge 分支加 `rows == 1` 前置（与 `hc_mixes_auto` 的两个 persist 臂同款），decline 到两发；
2. 或把注释改成事实（`rows ≤ 1` 是 merge 臂的**前置条件**，不是「当前恰好」）；
3. 或把 `hc_mixes_auto` 的 else 分支对 `rows > 1` 直接走两发 `hc_front`（跳过 merge）—— 与 ①等价，但落在 Rust 侧。
> 本工部按「不自行修改方案未涉及的部分」纪律**未动代码**。

---

## 4. 编译/测试证据（交付 ③）

| 检查 | 命令 | 结果 |
|---|---|---|
| 工作区全 target | `cargo check --workspace --all-targets` | **EXIT=0**（`Finished dev profile`；warning 全部既存：`ferrite-models` 6 条、`ar_micro.rs` unused `peers`、`ferrite-serve/main.rs:93` unreachable） |
| 远端 nvcc（hc 两文件） | `nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 -c` on `ubuntu@43.202.208.136`（CUDA 13.2, B300 SXM6） | `dsv41_kernels.cu` **exit=0** / `dsv41_glue.cu` **exit=0**，无 error |

> ⚠️ `scripts/dsv41_compile_check.sh` 的**第一步**（`cargo test -p ferrite-dsv41`）在本机**必失败**：
> `ar_hcpost_parity` 的 `dlopen(libcudart.so)` 失败（本机无 CUDA runtime）——**环境性、与本改动无关**（本任务零代码改动）。
> 因此 nvcc 两步是直接在远端跑的同款命令，不是脚本输出。

---

## 5. GPU 验证手册（交付 ④）

> 双门禁：每臂同时报 `step_ms`（`[dspark] steps=` 分解）**AND** `mean-k`（Fix-A 基线 2.240）。
> **一臂一进程**；`/proc/<pid>/environ | grep DSV41_` 逐门回读（§10.1 幻影门纪律）——**这是本任务的关键**：
> §1.2 的 80 发假象就是**没有回读**造成的。

### 5.1 零代码改动，本任务只出「口径修正 + 判据」

| 臂 | env | 期望 | 判据（nsys，AR_SAFE） |
|---|---|---|---|
| **H0** 生产栈（三开门 + `HC_FRONT_ROWS=1`） | 最优栈原样 | verify 侧 `hc_mixes_kernel` **= 0 发/步** | nsys kernel 表里 `hc_mixes_kernel` 的步均发数应为 **6**（只剩 draft），`hc_dots_late_kernel` = 走 split 的调用数（若单行 anchor 链也在跑则为 160/步） |
| **H1** raw 反证臂 | `DSV41_HC_FRONT_ROWS=0` | verify 侧 raw 复活 **80 发/步** | `hc_mixes_kernel` 步均应 = **86**；**这不是回归**，是 A/B 的反证腿 |
| **H2** 生产真实残量 | H0 + `+DSV41_P3LITE_HC_FRONT=1`（或 `DSV41_DRAFT_P3LITE_A_SEG=1`） | draft 的 6 发 → `dsv41_draft_hc_front` | `hc_mixes_kernel` → **0 发/步**；同时必须过 draft parity（`draft-p3lite-segment-fusion.md` 的 R1/R2/R3） |

### 5.2 证据采集（禁止吞吐反推）

```bash
# 每臂：一进程 + env 回读断言（缺一不可）
env DSV41_HC_FRONT_ROWS=1 DSV41_HC_VERIFY_FUSE=1 DSV41_VERIFY_AR_FOLD=1 \
    DSV41_HC_DEBUG=1 ... bash scripts/batched_400_v2.sh
grep -c DSV41_HC_FRONT_ROWS /proc/$(pgrep -f ferrite-serve | head -1)/environ   # 必须 = 1

# nsys：只看 kernel 相对发数（死锁规避 AR_SAFE）；AR_SAFE 口径的绝对值只作倍数用
env -u FERRITE_P2P NCCL_NVLS_ENABLE=0 DSV41_AR_V5=0 DSV41_GRAPH_STEP=0 ~/nsys_dual.sh
# 判据（发数，不是时间）：
#   H0: hc_mixes_kernel ≈ 6×steps        ；hc_dots_late_kernel ≈ 80×steps（或 160×steps）
#   H1: hc_mixes_kernel ≈ 86×steps       （80 verify + 6 draft）
#   H2: hc_mixes_kernel == 0             ；多出 dsv41_draft_hc_front_kernel ≈ 6×steps
```
> `DSV41_HC_DEBUG=1` 让每个 `(note, rows)` 只打一行（`chain_dev.rs:16269-16285`）；
> **`[hc-front]` 无 note** ⇒ split 成功（§10.3 的判读法）；出现
> `hc_front_split declined -> hc_front (two-launch)` ⇒ 流/事件没建起来，**仍不是 raw**，但要查原因。
> **`hc_front_rows() off -> raw hc_mixes (gate)`** 这行一出现，就说明 H0 的 env 又丢了。

### 5.3 红线（任一破即弃该臂）

1. 出师表 1000 token **逐字 + 零拉丁**（`BF16_TRUNCATE=1` 红线）；
2. `[dspark] mean-k` 保持 **2.240±噪声**（掉出 ⇒ 数值回归）；
3. `faults=0`、无 `MISMATCH`；
4. `/proc/<pid>/environ` 回读确认 `DSV41_HC_FRONT_ROWS` / `DSV41_HC_VERIFY_FUSE` / `DSV41_VERIFY_AR_FOLD` 真进进程；
5. H2 额外红线：draft parity 全过（`DSV41_DSPARK_UNIT_DUMP` / `dspark_parity`），否则「消掉 6 发」的收益不能收。

### 5.4 预期判读（票面校准）

- **H0 vs H1 的 raw 差**：80 发 × 51.4µs = **4.11ms/步**（AR_SAFE 口径）—— **这是 verify 的 AR_SAFE 假象部分**，
  生产栈（`HC_FRONT_ROWS=1`）里**本来就不存在**；不要把它算进「还欠的优化量」。
- **生产真实的 raw 残量**：draft 6 发 × 51.4µs = **0.31ms/步**（H2 的票面，需过 parity 才能收）。
- **`hc_dots_late`**：若生产是「m 行 verify + 单行 anchor」两条链 ⇒ ~160 发/步；它的 14.9µs 是 **spread 形态的
  单发耗时**（152 块已铺满 148 SM），**再 rows 化没有可打的空间** —— 真剩余杠杆是 `DSV41_HC_DL_KCHUNK`
  （`:13265`，默认 OFF，`NEXT-SESSION-HANDOVER.md:112` 记 v22 略负）与 `DSV41_HC_DL_SIDE`（第四条流）。

---

## 6. 战略提醒（给主 agent）

- **本项（nsys #4 的 hc 族）在「m 行批量化」这个维度上已无肉**：hc 全链 rows-native（§2），
  `51.4µs → 7.4µs` 的 3.1× 已经在 `hc_mixes_auto` 的 fusion 里，差的是 **dispatch 覆盖率**不是 shape。
- 想再动 hc，只有三格：
  ①**draft 的 6 发 raw**（H2，0.31ms，需 draft parity）；
  ②**单行 anchor 链**是否真的每步都在跑（若在，那是 80 发 `hc_dots_late` 的第二条来源，属 §1.4 的待重数项）；
  ③`hc_front_kernel` 的共驻性修复（§3，防雷，不是提速）。
- **40 层串行链长**（`vg-shape` 判死的那条）仍然是 verify floor 的主因 —— hc 族不是那条路的解。
