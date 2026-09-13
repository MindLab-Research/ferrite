# 融合对齐 —— verify 吃 eager 的融合形态（rows 版）

> 工部 · 2026-09-13 · SGLang 抄作业路线第 4 项（`sglang-verify-model.md` §4「融合对齐」）。
> 基线 HEAD `908a824`（+ 工作树内 peer 的 `dsv41_experts_mxf4.cu` 改动，未触碰）。
> **未跑 GPU/e2e**；远端 `nvcc compile-only` + `cargo check` 已跑。
> 依据：`docs/agent/sglang-verify-model.md` §3/§4、`docs/agent/g3-per-row-batched-audit.md` §0.1/§1、
> `docs/agent/verify-amortization-lesion-audit.md` §10（per-layer-census 的判决）。

---

## 0. 一句话结论（含一条**对任务书前提的纠偏**）

1. **任务 1 的真正障碍不是「kernel 没有 rows 维」，而是「块级 attention 在稳态不成立」**。
   `sparse_attn_orope_kernel` / `sparse_attn_split_kernel` **本来就收 `b, m` + `clen_rows` +
   `idx_stride` + `row_step` + `row_pitch`**（W2-MROWS-TP 全 ABI 齐备），verify 也已经有
   `sparse_attn_orope_pitched` 的调用点（`chain_dev.rs` 的 `mrows_attn` 块）——只是它挂在
   `ATTN_MROWS` 门下，而 `ATTN_MROWS` 的硬前置 `pos_base + m - 1 < win`（`:12489`）**在稳态恒
   decline**（g3-audit §0.1 已判）。所以「再写一个 rows kernel」解决不了稳态问题。
2. **稳态下块级 attention 必然读错**：ring 是 block 内 m 行**原地 append** 的滑动窗口；ring 一旦
   翻转（`pos_base + m - 1 >= win`），后行的 append 会覆盖前行窗口仍在枚举的槽 —— 这就是
   audit defect #2（也正是 `verify_ring_win` 被 revert 的原因）。**任何「先 append 全部 m 行、
   再整体 attention」的写法在稳态都是静默损坏**。
3. **本交付给出的真解**：**把 append 延后 + 按位置解析窗口槽**（`dsv41_kv_win_fetch`）。
   - 块级 attention 跑的时候 ring 里**只有块前历史**（append 还没发生），所以没有任何行能覆盖别的行的槽；
   - 每个窗口槽 `idx` 用**行自己的位置**反解出它该持有的位置 `p = pos_r - ((pos_r - idx) mod win)`；
     `p >= pos_base`（本块的行）就读 `s.kv_r`（本块自己的 KV 行），否则读 ring。
   - 两条分支返回的都是 per-row 路径当场 append 后读到的**同一批字节** ⇒ **逐位等价，且不需要
     no-turnover 前置** ⇒ **稳态可用**。
4. **任务 2**：`lin_rope_norm` 的 rows 版**已经存在** = `dsv41_gemm_fp8_mrows_rope_norm`（K2，
   `dsv41_kernels.cu:7412` 的 `gemm_fp8_mrows_rope_norm_kernel<M>`），本次只做**接线**（新 gate
   `DSV41_VERIFY_LINROPE_MROWS=1`）。⚠️ **但它的 parity 收据是缺的**（`sglang-verify-model.md` §5：
   l4/K2 FAIL `6400/6400`），所以本 gate **默认 OFF 且不做任何「逐位」宣称** —— 见 §3。

---

## 1. 交付 ①：`sparse_attn_orope` 的 rows 版（`DSV41_VERIFY_OROPE_MROWS=1`）

### 1.1 改动清单

| 文件 | 改动 |
|---|---|
| `kernels/cuda/dsv41_kernels.cu` | +`dsv41_kv_win_fetch`（device helper，位置解析的窗口读取）+ 2 个 ring reader 的 6+6 个读取点改用它；`sparse_attn_split_kernel` 收 `base`/`kv_rows`；`sparse_attn_orope_kernel` 收 `kv_rows`；`_impl` 收 `kv_rows`；+`dsv41_sparse_attn_orope_mrows`、+`window_idxs_mrows_kernel`/`dsv41_window_idxs_mrows`、+`ring_append_mrows_kernel`/`dsv41_ring_append_mrows` |
| `crates/ferrite-models/src/dsv41/device.rs` | +3 个符号字段/注册 + `sparse_attn_orope_mrows` / `window_idxs_mrows` / `ring_append_mrows` / `supports_sparse_attn_orope_mrows` |
| `crates/ferrite-models/src/dsv41/chain_dev.rs` | +gate `verify_orope_mrows()` / `sparse_orope_env_ok()`；`attention_rows` 里 `orope_mrows_ok` 谓词、窗口索引 hoist、逐行 attention/append 跳过点、块级融合 launch + 延后 append |

### 1.2 KV 窗口的**行独立性**论证（任务书要点）

**先说结论：m 行的窗口**不是**天然独立的 —— 它们共享同一个 ring，而 ring 的槽是「位置 mod win」的
原地覆盖。** 具体地，设 ring 满（`pos_base >= win`）：

- 行 r 的窗口 = 位置 `[pos_r-win+1, pos_r]`，共 win 个位置；因为 `win` 个连续位置对 `mod win` 恰好
  各占一个槽，窗口枚举到的是**全部 win 个槽**。
- 本块的行 `pos_base..pos_base+m-1` 写槽 `(pos_base+j) mod win`，与历史槽**重叠**。
- 于是行 r 需要的槽 `s` 上，后行 j>r 的 append 会覆盖掉行 r 要的历史位置 `p_r(s) = p' - win`，
  而且 `kv[p_r(s)]` 已经不在任何地方（ring 被覆盖、`kv_r` 只存本块 m 行）—— **不可恢复**。

所以「参考 `_pitched` 的 row_pitch 方案做行独立」在**读取寻址**层面做到了，但**内容层面**不成立。
唯一安全的做法是让「行 r 读 ring 的时刻」等于「per-row 路径中行 r 读 ring 的时刻」，即：

- **append 延后**：ring 在整块 attention 期间只含块前历史（= per-row 路径里「行 r 未 append 前行
  r+1..m-1 也还没 append」的那个状态）；
- **本块位置改由 `kv_rows` 供给**：`p >= pos_base` 时读 `s.kv_r[(p-pos_base)*d + c]` —— 正是
  per-row 路径刚 append 进去的那份字节。

这两条合起来，使「行 r 的 KV 窗口」逐字节等于 per-row 路径的同一个窗口，**且与 m/ring 翻转无关**。

### 1.3 位置反解为什么是唯一解（`p = pos_r - ((pos_r - idx) mod win)`）

`window_idxs_kernel` 的 decode 分支输出的是**槽号**（`idx ∈ [0,win)`），不是位置。给定行 r 的
`pos_r` 与槽号 `idx`，窗口内满足 `p ≡ idx (mod win)` 的位置**唯一**（窗口长度恰为 win），就是
`p = pos_r - ((pos_r - idx) mod win)`（C 里先 `%` 再补正到非负）。验证（`win=128, pos_r=200`）：

| 索引项 | idx | p |
|---|---|---|
| `c=0` | `oldest = 73` | `200-((200-73)%128)=200-127=73` ✓ |
| `c=127` | `72` | `200-((200-72)%128)=200-0=200` ✓ |

而 `p < pos_base` 分支读到的 ring 槽一定就是 `kv[p]` 本身：per-row 路径里 ring 槽 `idx` 持有
「≤ (当时最大已 append 位置 = pos_base-1) 且 ≡ idx」的最大位置 `p'`；因为 `p < pos_base`、`p ≡ idx`，
且 `p` 是 ≤ pos_r 里 ≡ idx 的**最大**者，故 `p' = p`。**两条分支是同一批字节。**

### 1.4 指令级逐位论证（P3B/l4 教训：落到 fma 链 / 累积器 / launch 边界）

**a) fma 链与累积器：零改动。** attention 主体（`qv` staging、`my_acc[i] = my_acc[i]*corr + e*kb[i]`、
`my_smax/my_se` 更新、`__shfl_xor_sync` 归约、`sh_smax/sh_se/sh_acc` 的 warp 合并、`phase 2` 逆 rope、
`phase 3` fp8 发射）**逐行 copy 未动**，唯一改的是 `kb0/1/2[i]` 的**取数地址**：
`kv[kBase + idx*d + c]` → `dsv41_kv_win_fetch(...)`。取值相同 ⇒ 后续所有 fma 的输入相同 ⇒ 累积器
逐次加法相同 ⇒ 输出位相同。

**b) 归约树宽度：未动。** `blockDim = 128`（4 warp）与 per-row 路径一致；`nwarp = blockDim.x>>5`
决定 `sh_acc[4][512]` 的合并与 `wsc[w]` 权重，改成别的宽度都会改求和顺序 —— 本 kernel **没有**改。

**c) launch 边界 / grid 形状：与 per-row 等价。** 融合版 grid = `(b*m, h)`，每个 block 仍是
「一行 × 一个 head」，与 per-row 的 `b=m=1` 单发**同一 program**（同一 `blockIdx` 语义、
同一 `hh` 循环）。split arm 同理：grid = `(split_c, b*m, h)` + merge `(b*m, h)`，与 per-row 的
split arm 同形。

**d) 位置源一致。** rope 位置 `tt = *base*mul + off + hh*step + mm*row_step`，本臂传
`mul=1, off=0, step=0, row_step=1` ⇒ `tt = *base + mm`；per-row 路径传 `off=r` ⇒ `tt = *base + r`。
两者都读**同一个 live device counter**（`pos_ctr`），且 `*base == pos_base`（per-row 路径本身成立
的前提）。子替换用的 `pos_r` 也用同一个 `*base + mm`，**不与 rope 位置分叉**（K2 教训）。

**e) 窗口索引 hoist：字面相同。** `window_idxs_mrows_kernel` 的 decode 分支是
`window_idxs_kernel` 的**逐句 copy**（同 `oldest = (start_pos%window)+1`、同 wrap、同
`idx > start_pos → -1`、同 `start_pos == 0` 特例），行 r 用 `pos_rows[r]`、写
`idxs + r*idx_stride`（`idx_stride = ist = win + index_topk`，正是 per-row 的 `idxs_r + r*ist`）。
⇒ 与 m 次单行 `window_idxs` **逐字节相同**。

**f) append 延后：只改「读写相对顺序」，不改最终 ring 状态。** `ring_append_mrows_kernel` 写
`ring[slot(pos_rows[r])*hd + i] = kv_rows[r*hd+i]`，与 per-row 的 `ring_append` / `ring_win_fuse`
的 append 半**同址同值**（`slot = pos mod window`）。唯一的差别是它排在整块 attention 之后：
attention 读的是「只有历史」的 ring（由 (1.3) 的替换补上本块行），写完之后 ring 的状态与 per-row
接口完全一致 ⇒ **对下一 step / rollback 的可见状态逐字节不变**。

**g) 图捕获期分支一致。** 谓词 `orope_mrows_ok` 在 capture 期求值一次（`OnceLock` 缓存 env + 符号
探针 + 形状），与 per-row 路径的 gate 读法一致；kernel 内的 `kv_rows != nullptr` 是**参数**不是
env，图重放时行为不变。`pos_base / pos_r` 全部来自 device counter，不烧进图。

### 1.5 发数账（m=6，每层）

| | 关（现状） | 开 |
|---|---|---|
| ring append + window | 6（`ring_win_fuse`，或 `RING_WIN_FUSE` 未开时 12） | 1（`window_idxs_mrows`）+ 1（`ring_append_mrows`，延后） |
| sparse attn + o-rope + o-quant | 6（逐行 `sparse_attn_orope`） | 1（`sparse_attn_orope_mrows`） |
| **合计** | **12（或 18）** | **3** |

⇒ 每层省 **9~15 发**（40 层/step ⇒ 360~600 发/step）。**并行度不降**（grid 仍是 `(m,h)`，不是
串行单块）—— 这是本设计相对「per-head 串行」方案的关键优势。

### 1.6 与 `ATTN_MROWS` / `RING_WIN_MROWS` 的互斥（代码级）

- `orope_mrows_ok` 显式要求 `!mrows_attn`；`rw_mrows` 显式要求 `!orope_mrows_ok`（本臂包含
  「窗口 hoist + append 延后」，与 RING_WIN_MROWS 重复，故让本臂优先）。
- 本臂要求 `owns_kv && !ring_owner_shared()`：延后 append 只在「本块是该 ring 的唯一写者」时成立。
  `DSV41_RING_OWNER=1`（跨层共享 ring，consumer 读 owner 的 ring）下，owner 的延后 append 会让
  随后运行的 consumer 读到本块的未来行 —— 那正是 per-row interleave 存在的意义，因此**该组合下
  本臂主动 decline**（consumer/owner 共享场景保留逐行路径）。

---

## 2. 交付 ④：`lin_rope_norm` 的 rows 版（`DSV41_VERIFY_LINROPE_MROWS=1`）

**已存在的 kernel**：`dsv41_gemm_fp8_mrows_rope_norm`（`dsv41_kernels.cu:7614` 的 launcher +
`gemm_fp8_mrows_rope_norm_kernel<M>` :7412），四段分别镜像
`dsv41_rmsnorm_rows_kernel`(1024) / `quant_kernel<0>`(block 32) / `gemm_fp8_mrows_kernel<M>`
的升序 `kb` 链 / `apply_rope_mrows_kernel`。**本次只做接线**：

- 新 gate `verify_linrope_mrows()`（`chain_dev.rs:2970`，strict `=="1"`，默认 OFF）；
- `attention_rows` 的 `k2_took` 谓词从 `attn_mrows_rope_norm() && mrows && ...` 放宽为
  `(attn_mrows_rope_norm() || verify_linrope_mrows()) && mrows && supports_gemm_fp8_mrows_rope_norm()`。
- 走到的路径与 draft 的 K2 臂**完全同一条**（`mrows_rope_norm` → `qr_norm_out = null`；后续
  `norm_rows` 仍然运行，为 indexer 的 q half 物化归一化行；`qr_raw_r` 逻辑不变）。
- 发数：`norm_rows + quant_rows + proj_mrows + apply_rope_mrows` 四发 → **1 发**（每层省 3）。

**⚠️ 逐位论证的诚实边界（P3B/l4 教训的正身）**：

- 本 kernel 的「逐位」是**文档声称**，`tests_dsv41_draft_parity.cu` 的 l4/K2 臂记录为
  **FAIL 6400/6400**（`sglang-verify-model.md` §5），`6400 = m*k = 5×1280` 正是 **`qr_norm` 那半**
  （即 segment 1 = NORM），**不是** rope 半 —— 说明分叉在 segment 1，不在发射几何。
- 头号候选（**假设，不是收据**）：l4 臂用 `qr_norm_out == qr_raw`（同一指针）调用，而 kernel 把
  `qr_raw` 与 `qr_norm_out` **都声明为 `__restrict__`** ⇒ 严格别名 UB；**出厂调用方传 `null`**，
  不走这条。若真因如此，parity 的 FAIL 只属于那条臂的用法，不代表出厂路径不等价。
- 因此本 gate **默认 OFF**，且**不带任何「逐位」宣称**：先在 l4 臂上把 `qr_norm_out` 改成 `null`
  （= 出厂形状）重跑，OK 之后才允许把 `DSV41_VERIFY_LINROPE_MROWS=1` 进栈。若重跑仍 FAIL，
  分叉就在四段本身，需要按 segment 1/2/3 逐段二分（指令级：`__fmaf_rn` vs 隐式收缩、
  `s_red` 跨 warp 折叠、`s_rows` 的 rope 交换）。

---

## 3. 交付 ②③（低优先，未做，说明理由）

- **`gemm_fp8_wo_pair` 的 rows 版**：任务书已标「低优先，时间允许再做」。勘察结论：verify 侧的
  wo_a/wo_b **已经是块级摊销**（`wo_a_grouped_fp8` 单发 + `proj_mrows`），`eager-verify-per-row-gap.md`
  §1 记 verify 在这两段**比 eager 更省**；本项 ROI 最低，故未动。
- **压缩器 / indexer 的 per-row 族**：不在本任务范围（g3-audit §3 已判 `compress_rows_fused`
  的 grid=1 是**正确性约束**，跨行耦合，不可 grid 化）。

---

## 4. 编译/测试证据（交付 ③）

```
cargo check -p ferrite-models                 -> EXIT 0（7 warning，既存风格；含一条
                                                 `verify_orope_mrows` 未用 → 已修：谓词漏用，已补）
cargo check --workspace --all-targets         -> EXIT 0（无 error）
远端 nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 -c dsv41_kernels.cu
                                              -> 见 §6 记录（compile-only）
```

⚠️ **已知坑（本次踩到并已修）**：`dsv41_kv_win_fetch` 必须定义在 `sparse_attn_split_kernel`
**之前**（两个 ring reader 都用它），首轮 nvcc 报了 6 处 `identifier undefined`；已把定义上移，并在
`sparse_attn_orope_kernel` 处留注释指针。

---

## 5. GPU 验证手册（交付 ④，双门禁 + 发数判据）

> **纪律**（`verify-amortization-lesion-audit.md` §5）：每臂**一进程**（`OnceLock` 每进程读一次
> env）；`/proc/<pid>/environ | grep DSV41_` **逐门回读**；吞吐/步时只认非 nsys e2e 的
> `[dspark] steps=` 分解；per-kernel 用 nsys 且只看相对倍数；**一臂同时报 `step_ms` AND `mean-k`**。

### 5.1 本轮三臂（同栈、交错 A/B/A/B）

| 臂 | env 增量 | 期望 | 判据 |
|---|---|---|---|
| **R1** orope rows（冷启动 + 稳态都生效） | `DSV41_VERIFY_OROPE_MROWS=1` | 每层 12→3 发；**稳态也生效**（与 ATTN_MROWS 的本质区别） | nsys 见 `sparse_attn_orope_mrows` / `window_idxs_mrows` / `ring_append_mrows`；逐行 `sparse_attn_orope` 实例 → 0；逐行 `ring_win_fuse`/`ring_append` 实例 → 0 |
| **R2** linrope rows（**需先补 parity 收据**） | `+DSV41_VERIFY_LINROPE_MROWS=1` | 每层 4→1 发；quad 里 `norm_rows+quant_rows+proj_mrows+apply_rope` 四族实例 → 0，改见 `gemm_fp8_mrows_rope_norm_kernel<6>` | 同上；`qr_norm`/`q` byte 对比见 5.3 |
| **R0** 基线（同栈，无新 gate） | — | 现状 | 发数与 R1/R2 对照 |

### 5.2 发数判据（不允许用吞吐反推）

```bash
# 同栈基线（R0）
DSV41_TIMING=1 DSV41_DSPARK_DEBUG=1 bash scripts/batched_400_v2.sh
# R1
DSV41_VERIFY_OROPE_MROWS=1        bash scripts/batched_400_v2.sh
# R2（parity 收据补齐后才允许）
DSV41_VERIFY_LINROPE_MROWS=1      bash scripts/batched_400_v2.sh
# nsys 轮（只看 kernel 相对倍数）
env -u FERRITE_P2P NCCL_NVLS_ENABLE=0 DSV41_AR_V5=0 DSV41_GRAPH_STEP=0 ~/nsys_dual.sh
```

- **R1 的关键否证**：若 nsys 里仍见**逐行** `sparse_attn_orope` / `ring_append` 实例，说明
  `orope_mrows_ok` 被某个前置挡住（`owns_kv` / `ring_owner_shared` / `ATTN_SEQ` / 符号缺 /
  `mrows_attn` 同开）—— 逐条 `/proc` 回读 + `nm -D libferrite_kernels.so | grep _mrows` 定位。
- **R1 的正确性红线**：**稳态（`pos_base + m - 1 >= win`）下必须仍然走 rows 臂**。若在稳态看到
  rows 臂消失 ⇒ 前置写错（本臂**不该**有 turnover decline）；反之若 rows 臂在稳态产生
  `MISMATCH` ⇒ 替换的 position 代数错了（回查 §1.3）。

### 5.3 逐位红线（任一破即弃该臂）

1. 出师表 1000 token **逐字 + 零拉丁**（`DSV41_BF16_TRUNCATE=1` 红线）；
2. `[dspark] mean-k` 逐位/Z_不变（Fix A 后基线 **2.240**；掉出 ⇒ 数值回归）；
3. `faults=0`、无 `MISMATCH` 行；
4. `/proc/<pid>/environ` 回读确认新 gate 真进进程；
5. **R1 专属**：`DSV41_VERIFY_OROPE_MROWS=1` + `DSV41_ATTN_MROWS=1` 同开时，`ATTN_MROWS` 应被
   本臂压过（本臂优先）；若两者都想验，分两次单开。
6. **R2 专属**：`DSV41_VERIFY_LINROPE_MROWS=1` **必须先有 l4 臂（`qr_norm_out=null`）OK 的收据**；
   没有收据不得进栈（本 gate 不带逐位承诺）。

### 5.4 预期判读与战略提醒

- R1：票面（每层 −9 发 × 40 = −360 发/step）。按 `g3-audit` §10 的教训「票面必须按 nsys 时间占比
  折价」，**verify 的真实收益需 per-kernel 时间表验证**，不要用发数直接推 ms。
- R2：票面每层 −3 发，但**先解决 parity 收据**再谈收益。
- **战略提醒**：本项属「融合对齐」（路线图第 4 项，⭐×4），真正的 4×→1.3× 机制仍在 **G2（M 进
  GEMM tile）**；本项是消除「verify 用未融合 pair 形式」这一**结构性差异**，与 eager 对齐形态。

---

## 6. 附：变更清单

| 文件 | 改动 |
|---|---|
| `kernels/cuda/dsv41_kernels.cu` | `sparse_attn_split_kernel` +`base`/`kv_rows`；`sparse_attn_orope_kernel` +`kv_rows`；`_impl` +`kv_rows`（含 `b!=1` decline）+2 处 split 调用点；+`dsv41_sparse_attn_orope_mrows`；+`window_idxs_mrows_kernel`/`dsv41_window_idxs_mrows`；+`ring_append_mrows_kernel`/`dsv41_ring_append_mrows`；+`dsv41_kv_win_fetch`；12 个 ring 读取点改为位置解析 |
| `crates/ferrite-models/src/dsv41/device.rs` | +3 字段/注册 + 4 方法 |
| `crates/ferrite-models/src/dsv41/chain_dev.rs` | +`verify_orope_mrows()` / `sparse_orope_env_ok()` / `verify_linrope_mrows()`；`attention_rows` 的 `orope_mrows_ok` + 窗口 hoist + 2 处跳过点 + 融合 launch + 延后 append；`attn_own_snapshot` 并入本臂；`rw_mrows` 让位；`k2_took` 放宽 |

**未改**（按任务书优先级）：`gemm_fp8_wo_pair` rows 版（ROI 最低）、`compress_rows_fused`（正确性约束）。
