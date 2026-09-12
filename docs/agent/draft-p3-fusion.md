# draft 链 P3 —— 段级融合的实施方案（可实施级）

> 上游：`docs/agent/draft-1ms-design.md` §3 P3（本文档是它的细化，不改变它的定位与结论）。
> 账本：`docs/agent/draft-perf-ledger.md`（289–292 launch / ~2.66 GB / 4.9ms，本机无 GPU，ms 为推算）。
> 段内核先例：`docs/agent/dsv41-persistent-arch.md`（L1 段级 + P1c/P1d 实测）、`docs/agent/dsv41-layer-fusion.md`（tile 对齐 + 三条硬性律）。
> 仓库状态 HEAD `d0d475a`。生产几何：dim 5120 / hc 4 / nh 64 / hd 512 / window 128 / vocab 129280 / mr 256 /
> n_target 3 / n_mtp_layers 3 / bs 5 / TP8 / MoE 128 routed + 1 shared, topk 3。
> **本文档只做规划，不含实施。**所有判定带 `file:line`；所有 ms 为推算，标「(推)」。

---

## 0. 结论（TL;DR）

1. **每 mtp block 的 92 条 launch 里，真正"不可融合"的只有 4 类**：`hc_mixes` 的跨块归约（1 条）、
   `sparse_attn` 的 key-split（2 条）、`apply_rope` 的逐行形态（15 条，但可多行化 → 3 条）、
   以及 **block 末尾的 MoE all-reduce（1 条，物理边界）**。其余 ~73 条都是**同一相位链上的相邻小核**——
   这是 P3 可做的全部空间。
2. **一个必须先纠正的前提：draft 的 attention 是 REPLICATED 的，draft 每 block 只有 1 个 AR，不是 2 个。**
   `dspark_dev.rs:1347-1358` 明文：draft 的 attention 每个 rank 都持有整份 `wq_b`/`wo_a`/`head` 且映射**全局**
   head/group/vocab 几何；只有 MoE 是 TP 切的。⇒ 每 step 的 AR = **3 个**（每 block 一个，`dspark_dev.rs:1745-1760`），
   不是 `dsv41-persistent-arch.md §1` 对主链算的 2/层。**P3 的段边界因此比主链更自由**：
   段 A 与段 B 之间没有物理屏障，唯一屏障在 MoE 之后。
3. **"每段一核"不是无条件成立的**。段内相位链有 3 类**跨块耦合**：① 行归约（rmsnorm / hc_collapse /
   hc_mixes 的 ss 与 24 路 dots）；② 以 K 为输入的 GEMM（输出列所有权 ≠ 输入行所有权）；
   ③ sparse_attn 的 key-split。**任何一类都不能靠 `__syncthreads()` 解决**（它只同步一个 block）。
   本节给出两条形态：
   - **形态 I（无栅栏，P3a/P3b）**：只在"所有权一致"的相位之间融合 + 搬运已有融合核。落到 ~**55–65** launch/步。
   - **形态 II（段内核 = 相位机 + 软件网格栅栏，P3c）**：需要 **co-resident grid**（grid ≤ 常驻容量）
     与自复位栅栏。落到 ~**19–23** launch/步，即设计目标的 18–20。
4. **P1c/P1d 的两次实测（`dsv41-persistent-arch.md §6`）已经把形态 II 的三个坑钉死**，方案必须逐条回避：
   ① 不许用"最后 publish 的块跑 tail"的**选举**形态（P1d +3.3ms）；
   ② 不许让**尾块单独长自旋**（hc-merge +3.2ms）；
   ③ 不许 >1 波（regs → blocks/SM → wave）。形态 II 的栅栏必须是**全块到达**型。
5. **数值域只有一条红线**：融合核与基线核若不在**同一个编译单元**，FMA 收缩逐位不一致 ⇒ 1 ULP ⇒
   确定性文本变化（`hc_post_parity.rs` 实测 2.98e-8 = 2^-25）。仓库已有正解（`dsv41_kernels.cu:7942-7947`
   的 `dsv41_hc_collapse_norm_kernel` 逐句 pin `fmaf` + `shfl_down` 树 + 升序跨 warp 和）——**照抄，不要另创**。

---

## 1. 现状分析：draft_forward 的完整 kernel 序列

### 1.1 全局序列（`dspark_dev.rs:712-1028`）

| 阶段 | 位置 | launch |
|---|---|---|
| prologue：`quant1(main_h)` / `gemm_fp8_mx(main_proj)` / `rmsnorm(main_norm)` | `:686/:687/:698`（`project_main_x`） | 3 |
| `embed_expand_dev` | `:825` | 1 |
| `memcpy_d2d(pre_in ← premix_init)` | `:851` | 1 |
| **3 × mtp block** | `:856-1002` | **276** |
| head：`hc_collapse` / `rmsnorm(dspark_norm)` / `head_gemv_bf16_mrows` | `:1832/:1841/:1913` | 3 |
| markov：`dspark_markov_head` ×5（**严格逐步串行**） | `:1957-2032` | 5 |
| 阻塞 H2D：`pos_base`（1–2）+ `ids`（1） | `:808/:815` | （不计核数） |
| **合计** | | **289–292** |

### 1.2 每 block 92 条：按 P3 段分组（`s = 0..2`）

#### 【段 A】hc 前端 + attention —— 35 条（`draft_forward:860-927` + `draft_attention:1072-1341`）

| # | 调用 | 行 | 次 | 融合判定 |
|---|---|---|---|---|
| A1 | `hc_mixes(attn)` | `:860`（`hc_mixes` fn `:1032`） | 1 | **必须独立**：24 路 dot 每条都是整条 `hc_dim=20480` 的归约；sinkhorn 必须留在**单个 warp 的寄存器**里（`dsv41_kernels.cu:7828-7832` 实测：搬出去答案就变） |
| A2 | `hc_collapse` | `:889` | 1 | **可融合**（行局部） |
| A3 | `rmsnorm(attn_norm, rows=bs)` | `:915` | 1 | **可与 A2 融合 —— `hc_collapse_norm` 已存在**（`dsv41_kernels.cu:7948/7983`，FFN 侧在用） |
| A4 | `quant1(main_x)` | `:2050` | 1 | 可融合（行局部）/ 可被 `gemm_fp8_mx_f32` 折掉 |
| A5 | `gemm_fp8_mx(wkv)`（seed） | `:2051` | 1 | 输出列所有权 |
| A6 | `rmsnorm(mk)` | `:2062` | 1 | 行局部 |
| A7 | `apply_rope(mk)` | `:2070` | 1 | 行局部 |
| A8 | `memcpy_d2d(→ ring slot)` | `:2079` | 1 | **可消除**（`ring_append`，`device.rs:3780`） |
| A9 | `quant1(xn)` | `:1105` | 1 | 可折（见 A10） |
| A10 | `gemm_fp8_mx(wq_a)` | `:1106` | 1 | 输出列所有权 |
| A11 | `rmsnorm(q_norm)` | `:1117` | 1 | 行局部 |
| A12 | `quant1(qr)` | `:1125` | 1 | 可折（见 A13） |
| A13 | `gemm_fp8_mx(wq_b)` | `:1127` | 1 | 输出列所有权 |
| A14 | `apply_rope(q)` **×bs** | `:1141`（`rope_queries:2092`） | **5** | **可多行合 1**（`apply_rope` 已支持 rows+step） |
| A15 | `quant1(xn)` | `:1146` | 1 | 可折 |
| A16 | `gemm_fp8_mx(wkv)` | `:1147` | 1 | 输出列所有权 |
| A17 | `rmsnorm(kv)` | `:1158` | 1 | 行局部 |
| A18 | `apply_rope(kv, rows=bs)` | `:1169` | 1 | 行局部 |
| A19 | `memcpy_d2d(window → all_kv)` | `:1196-1217` | 1–2 | **可消除**（定长 win 行 + device 派生 `n_win`） |
| A20 | `memcpy_d2d(kv → all_kv)` | `:1219` | 1 | 同上 |
| A21 | `sparse_attn`（split+merge **两发**） | `:1234`（launcher `dsv41_kernels.cu:7068-7130`） | **2** | **必须独立**（key-split 跨块）+ 可合 1（`sparse_attn_orope`，`dsv41_kernels.cu:7162`） |
| A22 | `apply_rope_inv(o)` **×bs** | `:1250`（`rope_queries_inv:2109`） | **5** | 可多行 1；且可折进 A21（`try_orope`） |
| A23 | `quant1(o)` | `:1277` | 1 | 折进 A21/A22 的 epilogue |
| A24 | `wo_a_grouped_fp8` | `:1281` | 1 | 输出列所有权（已 weight-stationary，1 发覆盖 8 group） |
| A25 | `quant1(wo)` | `:1328` | 1 | 可被 `gemm_fp8_mx_f32` / `WOB_F32` 折掉 |
| A26 | `gemm_fp8_mx(wo_b)` | `:1329` | 1 | 输出列所有权 |

**段 A 小计 = 35**（A19 取 1）。

#### 【段 B】attn 残差 + ffn 前端 + MoE —— 54 条（`draft_forward:935-1001` + `draft_moe:1359-1761`）

| # | 调用 | 行 | 次 | 融合判定 |
|---|---|---|---|---|
| B1 | `hc_post(attn)` | `:935` | 1 | 行局部，可与 B2 融合（`hc_post_inplace`，`dsv41_kernels.cu:7882/7919`） |
| B2 | `memcpy_d2d(h ← h_out)` | `:945` | 1 | **可消除**（in-place） |
| B3 | `hc_mixes(ffn)` | `:952` | 1 | **必须独立**（同 A1） |
| B4 | `hc_collapse_norm(ffn)` | `:961` | 1 | 行局部（已是融合核） |
| B5 | `gemv_bf16(gate)` **×bs** | `:1378-1386` | **5** | **可多行合 1**（`head_gemv_bf16_mrows` 同族先例） |
| B6 | `route_topk` | `:1388` | 1 | 可与 B5 融合（`DSV41_ROUTE_FUSE` 的 epilogue 形态，主链已验证） |
| B7 | `quant_fp4` | `:1408` | 1 | 行局部 |
| B8 | `expert_gate_up_fp4_batched` **×bs** | `:1532` | **5** | **可 mrows 合 1**（门已在 `:1505/:1522`，默认 OFF） |
| B9 | `swiglu_limit_batched` **×bs** | `:1563` | **5** | `gateup_fused` 时是 0；否则可 mrows 1 |
| B10 | `expert_down_reduce_fp4_batched` **×bs** | `:1572` | **5** | **可 mrows 合 1**（升序 slot 契约必须保留） |
| B11 | `quant1(xn)`（共享） | `:1677` | 1 | 可折 |
| B12 | `gemm_fp8_mx(w1)` **×bs** | `:1685` | **5** | **可 mrows 合 1**（验证侧 `shared_expert_rows` 是模板，`chain_dev.rs:8136-8226`） |
| B13 | `gemm_fp8_mx(w3)` **×bs** | `:1696` | **5** | 同上 |
| B14 | `swiglu_limit` **×bs** | `:1707` | **5** | 可 `swiglu_limit_q`（swiglu+quant 一发） |
| B15 | `quant1(ex_act)` **×bs** | `:1713` | **5** | 被 B14 的 `_q` 变体折掉 |
| B16 | `gemm_fp8_mx(w2)` **×bs** | `:1716` | **5** | **可 mrows 合 1** ⚠️ `n` 必须是 `sh_il` 不是 `inter_local`（`:1681-1684` 的 8× 越界历史 bug） |
| B17 | `add_inplace` | `:1728` | 1 | 可被 `gemm_fp8_mx_add`（A5 形态）折进 B16 的 epilogue |
| B18 | **`all_reduce_inplace`（AR，v5 = store + pubred **两发**）** | `:1745-1760` | **1 调用 / 2 核** | **物理边界，绝不进核内** |

**段 B 小计 = 54**（含 AR 的 2 个物理核，账本口径记 1 条 launch）。

#### 【段 C】ffn 残差 —— 3 条（`draft_forward:980-1001`）

| # | 调用 | 行 | 次 | 融合判定 |
|---|---|---|---|---|
| C1 | `hc_post(ffn)` | `:980` | 1 | 行局部 |
| C2 | `memcpy_d2d(h ← h_out)` | `:990` | 1 | **可消除** |
| C3 | `memcpy_d2d(pre_in ← pre_ffn)` | `:1000` | 1 | **可消除**（premix ping-pong：块序是 `attn_pre → ffn_pre → 下一块 collapse`，把 `pre_in` 改成"上一块的 `pre_ffn` 指针"即可，零拷贝） |

**段 C 小计 = 3。**

**校验**：35 + 54 + 3 = **92/block** ×3 = 276，+ 一次性 13 = **289** ✓（与账本 §1.3 一致）。

### 1.3 三段与太子描述的对照

| 太子描述 | 本文档段名 | 内容 |
|---|---|---|
| ① [hc+norm+投影] | 段 A 的前半 + 段 B 的后半 | `hc_mixes/hc_collapse/rmsnorm` + 全部投影 GEMM |
| ② [attention+MoE] | 段 A 的后半 + 段 B 的前半 | `sparse_attn/wo_*` + `gate/route/experts/shared` |
| ③ [head 输出收集] | **一次性段 D**（不在 block 内） | `hc_collapse + rmsnorm + head_gemv`（`:1832-1921`）+ markov ×5 |

**为什么"attention+MoE"不能各自成段、而必须"投影+attn"同段**：AR 只在 MoE 之后（§0.2）。
若把 attn 单独成段，则 attn 段的输出 `o` 要落 global 再被段 B 读，多一次 L2 往返而零收益；
反之把 `wo_b` 留在段 A 末尾，段的**最后一个写者**正好是 AR 的 store 可折入点（`dsv41-persistent-arch.md §2` 安全级）。

### 1.4 必须独立的 4 类（不可融合清单）

| 类 | 实例 | 不可融合的物理原因 |
|---|---|---|
| **跨块归约** | `hc_mixes`（A1/B3） | ss 与 24 路 dots 都是整条 `hc_dim=20480` 的归约；`sinkhorn` 必须单 warp（`dsv41_kernels.cu:7828-7832`） |
| **key-split** | `sparse_attn` 的 split（A21） | block (ck, row, h) 只拥有 `topk*ck/C .. topk*(ck+1)/C` 的 key，merge 必须等全部分片（`dsv41_kernels.cu:1485-1486`） |
| **逐步串行** | `dspark_markov_head` ×5（head 段） | step s+1 的输入 `ids[s+1]` 就是 step s 的 argmax（`dsv41_glue.cu:1388/1492`）⇒ **5 步绝不合并**（`draft-1ms-design.md §1.3(a)`） |
| **跨 rank 物理边界** | MoE 的 `all_reduce_inplace`（B18） | v5 = store + pubred 两发；**hc-merge 已证**：单核 + ticket 自旋 = **+3.2ms**；段边界必须锚在这里 |

---

## 2. 数值域风险（段内融合）

### 2.1 FMA 收缩 —— 1 ULP 的确定性塌陷（最高优先级）

**机理**：`--use_fast_math` 下 nvcc 自行决定 `a*b+c` 是否收缩成 `fma`。两个 kernel 若分处不同编译单元
（`dsv41_kernels.cu` / `ferrite_kernels.cu` / `dsv41_glue.cu` / `experts_mxf4.cu` / `route.cu`），会**独立**决定，
结果差 1 ULP。

**仓库实测证据**：
- `hc_post_parity.rs`：2.98e-8 = 2^-25（`dsv41-persistent-arch.md §5`）。
- `dsv41_kernels.cu:7942-7947` 的注释原文：*"two kernels in different translation units already disagreed
  about contraction once and the result was a one-ulp shift that flipped preambles"* —— 这就是
  `dsv41_hc_collapse_norm_kernel` 把 `fmaf` + `shfl_down` 树 + 升序跨 warp 和**逐句 pin 死**的原因。
- P1b 的跨 CU epilogue 用显式 `__fmul_rn`/`__fmaf_rn` 才过关（`dsv41-persistent-arch.md §6` P1b 行）。

**对 P3 的硬约束**：
1. **段内核与它替代的每一个基线核必须在同一个 `.cu`**。落点：段 A/B/C 的核放 `kernels/cuda/dsv41_kernels.cu`
   ——它与 `hc_mixes` / `hc_collapse` / `hc_collapse_norm` / `hc_post_inplace` / `sparse_attn` / `apply_rope`
   同 TU。**但 AR（`ferrite_kernels.cu`）与 experts（`experts_mxf4.cu`）不同 TU** ⇒
   凡是跨 TU 的算式，**显式 `__fmaf_rn`/`__fmul_rn`/`__fadd_rn`**（P1b 的 epilogue 是范本）。
2. 段内核里**每个相位都配一个 `*_parity.rs` 逐位门禁**（`crates/ferrite-dsv41/tests/`，现有 8 个同族测试：
   `hc_parity.rs` / `hc_post_parity.rs` / `hc_persist_parity.rs` / `hc_dl_merge_parity.rs` / `ar_hcpost_parity.rs` /
   `attn_parity.rs` / `swapab_parity.rs` + `dspark_parity.rs`）。**先过 parity，再上 serve A/B。**
3. **不设"容差门禁"**。draft 的 token 是 129280 路近邻 argmax，1 ULP 就能翻 top-1（`draft-1ms-design.md §2.1`
   的 bf16-cast 讨论同源）。

### 2.2 行独立性（m = bs = 5）

draft 与主链最大的不同：**一次 launch 里有 5 行**。融合核必须保证**行间零共享**：

- **正例（照抄）**：`head_gemv_bf16_mrows` 的 C4「per-row independent accumulators with no cross-row
  recombination」（`dspark_dev.rs:1866-1878` 注释）；`expert_*_fp4_batched` 的 `blockIdx.z = arow`，
  「ROW INDEPENDENCE block」（`dspark_dev.rs:1488-1493`）；`rmsnorm` 的 n>1 路径（`grid(n)` 一 block 一行，
  `dspark_dev.rs:904-914` 记录了当年把 1e27 误归因于它的历史）。
- **反例（禁止）**：任何把 5 行摊进同一个归约树的写法（例如为了凑满 1024 线程把 5 行拼成一条 25600 长的
  归约）。行 r 的 `s2`/`ss`/`acc` 必须**只**由行 r 的线程持有。
- **融合核的相位划分建议**：元素级相位（collapse/post/quant/add）可以按 (row, column-tile) 二维切；
  归约级相位（rmsnorm/norm）**必须一行一树**。二者之间靠 **grid 栅栏**（形态 II）或**拆成两个 launch**（形态 I）。

### 2.3 归约顺序（逐指令照抄）

`dsv41-persistent-arch.md §5` 的三条硬性律，落到 draft 的段内核：

| 归约 | 基线顺序 | 禁止的改动 |
|---|---|---|
| `rmsnorm` | `shfl_down` 树 + **升序**跨 warp 部分和（`ferrite_kernels.cu:278-321` 语义） | 换成 `shfl_xor` / `atomicAdd` / 两级树重排 |
| `hc_collapse` | over `hc`（4 项）**升序** | 向量化后的重结合 |
| `hc_post` | over `hc` 升序，显式 `__fmaf_rn`（`dsv41_kernels.cu:7913`） | 换 `fmaf` 依赖编译器 |
| MoE down-reduce | **slot 升序**累加（`expert_down_reduce_fp4_batched`） | 并行 slot 归约 |
| MoE AR | **rank 升序**（v5 已保证 1-ulp 一致） | 改 store/pubred 的次序 |
| hc 的 K-split | **split = 1**（`dsv41-layer-fusion.md §1`） | `split>1` = 非逐位（P1d 实测的文本差异即由此而来） |

### 2.4 其它两个易踩的数值坑

- **`gateup_fused` 的三方镜像**：`dspark_dev.rs:1446-1449` 必须在段内核里**逐字复现**
  （`g_fuse && g_expert_fp4_mode==2 && dim%512==0`），否则 `act_slot` 差一个 `inter` ⇒ 静默乱码（Defect 2）。
- **`sh_il` vs `inter_local`**：共享专家的 w1/w3 的 `n` 必须是 `sh_il`（`:1681-1684` 记录 8× 越界的真实 bug）。
  段内核把共享专家折进去时，这个断言要**显式写进核内 `assert` 或 host 侧 `if`**。

---

## 3. 分段方案（每段一核的输入/输出/形状）

### 3.0 先说清"每段一核"的两个形态

相位链的三种跨块耦合（§0.3）决定了两条互斥的实现路径：

| | **形态 I（无栅栏）** | **形态 II（段内核 = 相位机 + 网格栅栏）** |
|---|---|---|
| 栅栏 | 无；相位之间只能是 **launch 边界** | **软件网格栅栏**：全块 `atomicAdd` 到达 + 自旋到计数满 + 自复位 |
| 前置条件 | 相位链里每一段相邻对**所有权一致**，或**拆 launch** | **grid ≤ 常驻容量**（co-resident，否则死锁）+ 栅栏自复位（图 replay 安全） |
| 段 A | 拆成 A-1（hc 链）+ A-2（投影链）+ A-3（attn 核）… ≈ 6–8 launch | 1 launch |
| 风险 | 低（全是已验证 kernel 的组合） | **高**（P1c/P1d 两次实测都回归） |
| 落点 | ~55–65 launch/步 | **~19–23 launch/步** |

**本文档把形态 I 定为 P3a/P3b 的交付，形态 II 定为 P3c，且形态 II 默认 OFF、单独 A/B。**

### 3.1 段 A —— `dsv41_draft_seg_a`

**输入**（全部在 kernel 启动前已由前驱写就，故块间只读，无需栅栏）：
| 缓冲 | 形状 | 来源 |
|---|---|---|
| `h` | `[bs, hc, dim] f32`（= 102400 f32 = 400 KB） | 上一块的 `hc_post`（段 C）或 `premix_init` |
| `pre_in` | `[bs, hc] f32` | 上一块的 `pre_ffn`（ping-pong，零拷贝） |
| `main_x` | `[dim] f32` | prologue |
| `window[s]` | `[win, hd] f32` | `note_ctx_rows` / `seed_window` / 上一块 |
| `idxs` | `[bs, win+bs] i32` | `ensure_idxs`（`n_win` 稳定后不再变） |
| 权重 | `wq_a/wq_b/wkv/wo_a/wo_b` + `q_norm/kv_norm/attn_norm` + `attn_sink` | `DsparkDev::w.mtp[s]` |

**输出**：
| 缓冲 | 形状 | 语义 |
|---|---|---|
| `o` | `[bs, nh*hd] = [5, 4096] f32` | `wo_b` 的输出（段 B 的 `hc_post` 消费者） |
| `pre_attn` | `[bs, hc]` | attn 的 mixes pre（**段 B 的 `hc_collapse_norm(ffn)` 要用它，不是 `pre_ffn`** —— `dspark_dev.rs:959-963`） |
| `post/comb` | `[bs,hc]` / `[bs,hc,hc]` | attn 的 hc_post 系数（段 B 的 B1 用） |
| `xq/xsc` | 可复用 scratch | 段内量化缓存 |
| `window[s]` / `all_kv` | 原地更新 | ring slot + 候选 KV |

**grid/block（形态 II）**：
- `G = 148 × k`（k = 每 SM 常驻块数，**由 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 实测决定**，
  目标 `G ≥ 296`），`block = 256`。**先用 P1d 的复核手段验 occupancy**（`dsv41-persistent-arch.md §6`：
  「regs 才是限制，smem 不是」）。
- 每个相位用 `for (i = bid; i < n_phase; i += gridDim.x)` 网格跨步；元素级相位走 (row, col-tile) 二维展平
  （bs=5 × dim=5120 ⇒ 25600 个元素，G=296 × 256 = 75776 线程 ⇒ 0.34 元素/线程 ⇒ 每线程 ~3–4 元素）。

**相位链（形态 II，13 个相位 / 12 个栅栏）**：
```
P1  hc_mixes(attn)                 ← 独立 launch（跨块归约，见 §1.4）
   ── launch 边界 ──
P2  hc_collapse + rmsnorm(attn_norm)   （= hc_collapse_norm 的 attn 变体；行局部，可栅栏内合并）
P3  quant(xn) → xq/xsc
P4  gemm wq_a  [5,1280,5120]
P5  rmsnorm(q_norm) + quant(qr)
P6  gemm wq_b  [5,4096,1280]
P7  rope(q)（rows=bs*nh, step 语义）  ‖ seed_window(wkv/rmsnorm/rope/ring)
P8  kv 链：quant+gemm wkv + rmsnorm + rope
P9  all_kv 组装（定长 win 行 + device 派生 n_win，替代 3 次 memcpy_d2d）
P10 sparse_attn（split）→ merge(+o-rope+quant epilogue)
P11 wo_a_grouped（读段内直出的 xq）
P12 gemm wo_b（+ 可并 A25 的 quant，走 f32 激活）
P13 [可选] AR#0 的 store 折入（draft 的 attn 无 AR，此位留给主链同构性检查，**draft 侧不需要**）
```
⚠️ **P1（`hc_mixes`）不进相位机**：它的 24 路 dots + sinkhorn 是"单 warp 串行链 + 跨块归约"，
P1d 已实测把这类结构塞进多块相位机 = **+3.3ms**。它保持独立 launch（**1 条**，不是 35 条）。

**段 A 目标时长（推）**：12 相位 × ~4µs 真实工作 + 11 栅栏 × ~2.5µs ≈ **75–90µs**（vs 现状 35 × 16.8 ≈ 590µs (推)）。

### 3.2 段 B —— `dsv41_draft_seg_b`

**输入**：`o`（段 A）、`h`、`post/comb`（段 A 的 attn 系数）、`pre_attn`、`xn` scratch、`gate_w/gate_bias`、
`experts[*]`（fp4 交错池）、`shared_w1/w2/w3`、`route_*`。
**输出**：`moe_out` = **partial 和**（`[bs, dim] f32`，每 rank 只是 `1/world`）—— **它必须走出核、被 AR 归约**。
**grid/block**：同段 A 的 `G`/`256`。

**相位链**：
```
Q1  hc_post(attn)  → h    （行局部；in-place，替代 B1+B2）
Q2  hc_mixes(ffn)       ← 独立 launch（同 A1 的理由）
   ── launch 边界 ──
Q3  hc_collapse_norm(ffn)（已是融合核）
Q4  gate(gemv bf16, mrows) + route_topk（fused epilogue 形态）
Q5  quant_fp4
Q6  experts: gate_up(mrows) [+ swiglu 或 fused] 
Q7  experts: down_reduce(mrows)  —— **slot 升序**（数值契约）
Q8  shared: w1/w3(mrows) → swiglu_limit_q → w2(mrows) + add_inplace（A5 形态并入 epilogue）
   ── launch 边界 ──
Q9  AR（v5：store + pubred，**核外**）
```
**段 B 目标时长（推）**：~8 相位 × ~5µs + 7 栅栏 × 2.5µs + AR 两发 ~10–25µs ≈ **80–100µs**。

### 3.3 段 C —— `dsv41_draft_seg_c`（可整段消失）

**输入**：AR 之后的 `moe_out`（full block）、`h`、`post/comb`（ffn 的）。
**输出**：`h`（下一块读）。

**两条实现**：
- **C-a（推荐，最省）**：把 `hc_post(ffn)` 折进 **AR 的 pubred epilogue** —— 仓库里**已有同族先例**：
  `p2p_ar_pubred_v5_hcpost_kernel`（`ferrite_kernels.cu:9075-9130`，`ar5_hc_post_col4/col1` 在 `:9019/:9053`），
  主链用它把 `hc_post` 折进 AR（`DSV41_HCPOST_EPI`）。**draft 侧需要一个新的变体**：draft 的 AR 域是
  `n = bs*dim = 25600`（`dspark_dev.rs:1755`），而 `hc_post` 的域是 `hc*dim = 20480` **per row × bs 行**，
  两者形状不同 ⇒ 新 kernel `p2p_ar_pubred_v5_hcpost_rows`（grid 覆盖 `n4 = 6400`，每线程 4 列 × 全部 hc 行，
  行维 `blockIdx.y ∈ [0, bs)`）。⇒ **段 C 完全消失，省 3 launch/block**。
  ⚠️ 跨 TU：`hc_post` 的算式在 `dsv41_kernels.cu:7882`，AR 在 `ferrite_kernels.cu` ⇒
  **必须显式 `__fmul_rn`/`__fmaf_rn`**（照抄 `ar5_hc_post_col4` 的写法），并配 `ar_hcpost_parity.rs` 同族门禁。
- **C-b（保守）**：`dsv41_hc_post_inplace`（已有）+ premix ping-pong（去掉 C2/C3 两个 memcpy）。
  ⇒ 1 launch/block。

### 3.4 一次性段（段 D）

| 项 | 现状 launch | P3 后 |
|---|---|---|
| `project_main_x`（quant+gemm+rmsnorm） | 3 | **1**（`gemm_fp8_mx_f32` 折 quant；或小相位机） |
| `embed_expand_dev` | 1 | 1（可与 prologue 同段） |
| head：`hc_collapse` + `rmsnorm` + `head_gemv` | 3 | **1**（`hc_collapse_norm` + head） |
| markov ×5 | 5 | **5（不动）** —— 严格逐步串行，是 P2 的领域不是 P3 的 |

### 3.5 P3 之后的总账

| | launch/步 |
|---|---|
| 现状 | 289–292 |
| P0–P2 后 | ~130 |
| **P3 形态 I** | **~55–65** |
| **P3 形态 II（P3c 全开）** | 3 block × (A 1 + B 1 + AR 2 + C 0) + 段 D ~3 + markov 5 ≈ **19–21** ✓ |

---

## 4. AR 的处理（本设计的硬边界）

1. **draft 每 block 只有 1 个 AR**（MoE 之后，`dspark_dev.rs:1745-1760`）。段 A 与段 B 之间**无屏障**，
   段 B 与段 C 之间**必有屏障** —— 这就是三段划分的由来，也是 `draft-1ms-design.md §3` 把 AR 写在段 B 末尾的原因。
2. **AR 的两个核（store + pubred）的参数化**：`draft` 用 `Collective::all_reduce_inplace`（Rust 侧一行），
   C 侧 `ferrite_ar_v5`（`ferrite_kernels.cu:8930-8943`）发 `p2p_ar_store_v5_kernel`（`dim3(blocks, world)`）
   + `p2p_ar_pubred_v5_kernel`。**store 的 `epoch` 是设备端读的**（`:4046` 注释），所以图 replay 下是对的。
3. **安全级融合（P3c 可选）**：把 **store 折进段 B 的最后一个写者**（`add_inplace` / `moe_down_reduce` 的 epilogue，
   主链对 wo_b 的 `DSV41_AR_STORE_FUSE` 是现成先例），pubred 仍独立 launch，其自旋窗口用 PDL 与段 C 重叠。
   ⇒ 段 B 的"末写者"同时是 AR store，省 1 个节点。
4. **绝不做的三件事**（都有实测回归背书）：
   - ❌ **pubred 进核内 / 尾块自旋**（hc-merge 单核 + ticket = **+3.2ms**）。
   - ❌ **store+stamp 合并**（`DSV41_AR_STAMP_FOLD=1`：serve 实测 **29.5s/step 灾难**，代码已物理删除，
     见 `dsv41-persistent-arch.md §6` P5 行）。
   - ❌ **`argmax_pub` 与 AR 共用暂存**（账本 §3 注：曾死锁；`44f4956` 记录过 watchdog 静默超时 →
     归约出陈旧半量 → **发出一个看似合理的错 token**）。P3 若让段内核内自带栅栏，**必须复核栅栏的
     暂存槽与 v5 的 staging 完全不相交**。
5. **栅栏与 AR 的 epoch 交互（形态 II 的隐藏雷）**：段内核内的软件栅栏若要自复位，必须像
   `gv2_route_epilogue`（`ferrite_kernels.cu:2749-2798`，注释记 "graph-safe"）与 `mk_ctr`
   （`dsv41_glue.cu:1493-1495`）那样**从设备端计数派生**，不能依赖 host 复位 —— 否则第二次 replay 会死等。
   **这是 P3c 的第一个验收测试。**

---

## 5. 实施分期

### P3a —— 最融合友好的相位链（hc 链 + 零风险融合）【工作量 S｜风险 低】

**目标**：把"所有权一致、且有现成核"的相邻对全部合掉。**不动任何算术**。

| # | 动作 | 位置 | 省 |
|---|---|---|---|
| a1 | attn 侧的 `hc_collapse` + `rmsnorm(attn_norm)` → **`dsv41_hc_collapse_norm`**（`:7983`，FFN 侧已在用，签名完全匹配 `(h, pre_in, attn_norm, xn, bs, hc, dim, eps)`） | `dspark_dev.rs:889-922` | 1/block |
| a2 | `hc_post(attn/ffn)` + `memcpy_d2d(h ← h_out)` → **`dsv41_hc_post_inplace`**（`:7919`） | `:935-949` / `:980-994` | 2/block |
| a3 | `memcpy_d2d(pre_in ← pre_ffn)` → **premix ping-pong 指针** | `:1000` | 1/block |
| a4 | `rope_queries` / `rope_queries_inv` → **rows = bs*nh, step 语义的 1 发** | `:2092/:2109`（`apply_rope` 已支持） | 8/block |
| a5 | `sparse_attn` + `apply_rope_inv` + `quant1(o)` → **`sparse_attn_orope`**（`:7162`，已有） | `:1234/:1250/:1277` | 6/block |
| a6 | `quant1(wo)` → **`gemm_fp8_mx_f32` / `WOB_F32`**（已有） | `:1328` | 1/block |

**落点（推）**：289 → **~245** launch/步；draft 4.37 → **~3.9ms (推)**。
**验收**：每一项单独 env 门、默认 OFF；`hc_post_parity.rs` / `attn_parity.rs` / `dspark_parity.rs` 逐位绿；
同二进制背靠背 A/B + 五段全文。

### P3b —— 段内核的外围（mrows + epilogue 折叠）【工作量 M｜风险 低-中】

**目标**：把每 block 的 54 条 MoE + 35 条 attn 压到"段内核可以一次吃下"的规模。

| # | 动作 | 位置 | 省 |
|---|---|---|---|
| b1 | 共享专家 **mrows**（`shared_expert_rows` 模板，`chain_dev.rs:8136-8226`）—— 26 → 6 | `dspark_dev.rs:1676-1727` | 20/block |
| b2 | routed 专家 **mrows A/B**（门已就位 `:1505/:1522`，只差实测）—— 15 → 3 | `:1525-1588` | 12/block |
| b3 | gate **mrows** + `route_topk` epilogue 折叠 —— 6 → 1 | `:1378-1400` | 5/block |
| b4 | `add_inplace` → `gemm_fp8_mx_add`（A5 形态） | `:1728` | 1/block |
| b5 | `quant1(xn/qr/wo)` → 各 GEMM 的 f32 直读 | `:1105/:1125/:1277` | 3/block |

**落点（推）**：~245 → **~130**；draft ~2.6–3.0ms (推)。
**这一步之后，每 block 剩 44 条、每 step 剩 ~130 条 —— 正是 `draft-1ms-design.md` 的 P0 落点。**

### P3c —— 段内核（形态 II：相位机 + 网格栅栏）【工作量 L｜风险 高】

**这是唯一能穿过 1ms 的一步。**

| 子步 | 内容 | 前置 |
|---|---|---|
| **c0** | **栅栏原语**：`dsv41_grid_barrier`（全块 `atomicAdd` + 自旋到满 + 自复位）。先写一个 micro-test：co-residency + 跨 replay 自复位 + 10⁴ 次迭代无死锁 | 独立 PR，先于一切 |
| **c1** | **段 C 消失**：`p2p_ar_pubred_v5_hcpost_rows`（§3.3 C-a） | c0 不需要；独立可做 |
| **c2** | **段 A 相位机**（§3.1 的 13 相位） | c0；occupancy 实测（`cudaOccupancyMaxActiveBlocksPerMultiprocessor`） |
| **c3** | **段 B 相位机**（§3.2 的 9 相位 + store 折入末写者） | c0 + c1 |
| **c4** | 段 D 折叠（prologue + head） | c2 |

**段内核的机械约束（写进核头注释，像 `dsv41_kernels.cu:7942-7947` 那样）**：
1. 核内**禁止** `cudaMalloc` / `cudaStreamSynchronize` / host 交互（图捕获合法性）。
2. 每个相位结束 `__syncthreads()`；**相位之间若跨块读写，必须插 `dsv41_grid_barrier`**。
3. **禁止**"最后到达的块跑 tail"的选举形态（P1d +3.3ms）。
4. **禁止**任何单块长自旋（hc-merge +3.2ms）。
5. `grid ≤ 148 × blocks_per_SM`，**在 host 侧 assert**（P1d 的 1.3 波就是这么坏的）。
6. `hc_mixes` **不进相位机**（`split` 语义 + 单 warp sinkhorn）。
7. 段内核与基线核**同 TU**；跨 TU 算式显式 `__fmaf_rn`。

**回退链**：`DSV41_DRAFT_SEG_A/B/C` 三个 env，**默认 OFF**；`OFF` 时逐 kernel 路径（P3b 的 ~130 条）必须逐位不变。

---

## 6. 影响范围

- **修改文件**
  - `crates/ferrite-models/src/dsv41/dspark_dev.rs`（段调用点替换、ping-pong premix、mrows 接线）
  - `kernels/cuda/dsv41_kernels.cu`（**段 A/B/C 内核的家**；`hc_collapse_norm` attn 变体）
  - `kernels/cuda/ferrite_kernels.cu`（`p2p_ar_pubred_v5_hcpost_rows`；AR store 折入）
  - `crates/ferrite-models/src/dsv41/device.rs`（段内核 launcher + 符号探测 `supports_*`）
  - `crates/ferrite-dsv41/tests/`（新增 `draft_seg_{a,b,c}_parity.rs`、`grid_barrier_parity.rs`）
- **影响模块**：draft forward；MTP exec 图（`tp.rs` 的 `FERRITE_DRAFT_GRAPH`，P3 后节点数暴降但形状不变）；
  P2 的 markov 切分面（**段 D 的 markov 段依赖 P2** —— `draft-1ms-design.md §4` 的「P2 在 P3 里是前置件」）
- **兼容性**：**无 API breaking change**。全部新门默认 OFF；`world==1` / 老 `.so`（符号探测）/ `hc != 4` /
  `bs > 8` 一律回退现路径。
  ⚠️ **但**：P3c 会改变每 step 的 kernel 拓扑，**依赖 nsys 逐核计数的脚本需同步**；
  unit dump（`DSV41_DSPARK_UNIT_DUMP`）的 per-unit 名字会变 —— **段内核必须保留同样的 dump 点**，
  否则 `dspark-correctness-chain.md` 的逐单元对比链断掉。

## 7. 风险评估

| 风险 | 概率 | 应对 |
|---|---|---|
| **形态 II 的网格栅栏死锁**（grid > 常驻容量 / 相位不齐） | 高 | c0 独立 micro-test：co-residency + 跨 replay 自复位 + 10⁴ 迭代；host 侧 `assert(grid ≤ 148*blocks_per_SM)` |
| **重蹈 P1d（+3.3ms）** | 中-高 | 核头注释里明文禁止选举/尾块自旋；相位机**只做全块到达型栅栏**；regs 用 `-Xptxas -v` + occupancy API 实测 |
| **重蹈 hc-merge（+3.2ms）** | 中 | 段内藏自旋**只允许段首全块栅栏**；AR 永远独立 launch / PDL 重叠（§4.4） |
| **跨 TU 1 ULP 翻转 token** | 中-高 | 同 TU 优先；跨 TU 显式 `__fmaf_rn`；每个段核配 `*_parity.rs`；**先过 parity 再 serve A/B** |
| **行间共享被引入**（为凑满线程把 5 行拼一树） | 中 | 相位划分按 §2.2；parity 测试**必须开 bs=5**（以及 bs=1 的退化用例） |
| **`gateup_fused` / `sh_il` 三方镜像在段内核里走偏** | 中 | 核内显式 assert；parity 覆盖交错池 + `gateup_fused` on/off 两种 |
| **收益不达标**（1.0–1.4ms 是推算，本机无 GPU） | 中 | **先做一次 nsys**（draft 段仅 ~290 节点，好定位）拆开「发射 / 字节 / 依赖延迟」三向，再定 P3c 投产；P3a/P3b 不依赖这个数 |
| **P2 未落地时段 D 的 head 段无意义** | 低 | 段 A/B/C 与 P2 正交，可先行；段 D 排在 P2 之后 |

## 8. 建议分工

- **工部**：P3a 全部 + P3b 的 b1–b5（段内核之前的机械折叠）+ P3c 的 c0（栅栏原语）+ c1–c4（段内核实现）。
- **户部**：① 一次 **nsys** 实测（draft 段）填平账本 §6.1 的三向拆分；② **occupancy 实测**
  （`cudaOccupancyMaxActiveBlocksPerMultiprocessor`，P3c 的 co-residency 前提）；③ P3a 每项的 A/B
  （`DSV41_TIMING` 的 `draft=` 字段）；④ 段内核的字节/时长地板复算。
- **刑部**：每个段核的**逐位 parity**（`draft_seg_{a,b,c}_parity.rs`）+ 边界用例（bs=1 / world=1 / hc=5 越界 /
  `gateup_fused` on-off / 段内核 launch 失败的回退路径 = 必须逐位 == P3b 路径）。
- **兵部**：① **网格栅栏的故障审查**（死锁 / 跨 replay 自复位 / 栅栏暂存与 v5 staging 不相交）；
  ② AR 与段内核 epoch 的交接；③ 段内核 launch 失败时**不得静默降级**（`chain_dev.rs:3964-3979` 的
  "print, do not silently degrade" 纪律）。
- **礼部**：落地后更新 `draft-perf-ledger.md`（§1 的 launch 表）+ `draft-1ms-design.md` §3 P3
  + `dsv41-persistent-arch.md`（补"draft 每 block 只有 1 个 AR"这条与主链的差异）+ 本文档。
- **吏部**：段内核的核头注释规范（**必须写清相位链、栅栏位置、数值契约三条**，对齐
  `dsv41_kernels.cu:7942-7947` 的现存范式）+ `#[allow(clippy::too_many_arguments)]` 的 launcher 风格一致性。

---

## 9. 需要太子/门下补充或裁定的信息

1. **`bs > 8` 的上界**：`head_gemv_bf16_mrows` 的 dispatch 是 1..=8（`device.rs:3330-3332`）。
   段内核的 block 数 = `G`，与 bs 无关；但**行维相位**（rmsnorm 的一行一树）在 `bs` 增长时
   会重新压低并行度。**需要确认生产的 bs 上界**（现在是 5）。
2. **P2 的落地时序**：段 D 的 head 段与 P2 的切分几何耦合（`draft-1ms-design.md §4`）。
   建议 **P3a/P3b 立即开工、P3c 等 P2 + nsys 实测之后**。
3. **形态 I 是否可接受为最终交付**：若 nsys 实测显示"每 launch 的 16.8µs 里发射只占 2.9µs、执行/依赖
   占 ~13µs"这个拆分成立，则形态 I（~55–65 launch）大概率停在 **1.8–2.3ms**，**过不了 1ms 门槛** ⇒
   必须上形态 II。**这个判断依赖那次 nsys，建议由户部优先给出。**

---

*中书省 · 基于 2026-09-12 仓库状态（HEAD `d0d475a`）。launch 计数为代码精确值；ms 为边际估算，待 nsys 实测。*
*本文档只做规划，不含实施；所有新增门默认 OFF，先 parity 后 A/B。*
