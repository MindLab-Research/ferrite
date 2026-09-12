# L5 — draft P3A + MARKOV_SLICED 的 A/B 验证准备

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 代码基线：当前 HEAD。逐条 `file:line` 核对过。
> 上游：`lazy-verify-optimization-path.md §3-L5` · `final-400-config.md §2.2/§3.2` · `draft-1ms-design.md`。
> ⚠️ 本机无 GPU ⇒ 所有 ms 均为账本/设计口径，**显式标注来源**。

---

## 0. 判决（先读五条）

1. **`DSV41_MARKOV_SLICED` 不是一个「markov 折叠」，它同时切了两样东西**：
   **head GEMV 的行距 + markov bias 的行距**，且两者必须共享同一个 `(seg, base)`
   （`dspark_dev.rs:3108-3130`）。名字只点了 markov，收益账是按 **head+markov 两笔**记的。
2. **−1.00ms 这个数在 `draft-1ms-design.md` 自己的字节算术里不成立**。该文档 §2.3 明算：
   每 rank 合计省 ~1.7 GB，按 4.5 TB/s ≈ **−0.38ms**，并自己写了「**P2 的上限就是 ~0.4ms**」。
   再扣掉切片引入的 5 轮 v5 往返（协议地板 17.3µs/轮，`lazy-verify-optimization-path §2.1` = ~87µs），
   **诚实期望是 −0.3 ~ −0.5ms，不是 −1.0ms**。⇒ A/B 的判读区间必须事先钉死，否则会把成功读成失败。
3. **P3A 的 4 项在 HEAD 的默认值仍是 OFF**（`dspark_dev.rs:365` `unwrap_or(false)`）。
   「Wave 1 已开启」是靠 **serve 环境变量**（`scripts/full_stack_test.sh:14`、`s2_ab_matrix.sh:137`、
   `lazy_graph_ab.sh:137`、`nsys_wave1.sh`），不是代码默认。
   ⇒ **A/B 第一件事是读回运行进程的 env 证明 P3A 真的在**（本项目 #1 测量陷阱）。
4. **「draft 当前 4.28ms」这个基线很可能不含 P3A**。`lazy-verify-optimization-path §1.1` 的表里
   明写 `draft 4.28 实测 … P3a/markov 未开`。若 Wave 1 真的开了 P3A，draft 应已 ~4.0ms。
   ⇒ **base 臂必须实测，不得假定**；否则 P3A 的 −0.30 会与 MARKOV_SLICED 的增量重复计账。
5. **`MARKOV_SLICED` 与 lazy verify 正交，且它的收益不会被 `×k_emit` 稀释**（§5）。
   这是它与 mrows 族的本质区别：mrows 是「把 m 行折成一发」（m=1 时恒为 0），
   MARKOV_SLICED 是「词表切分 + 固定 5 步循环」，draft 每步只跑一次。

---

## 1. MARKOV_SLICED 的机制

### 1.1 它切的是什么

| 项 | 全词表臂 | 切片臂 | 依据 |
|---|---|---|---|
| `markov_head` `[vocab, mr] f32` | **整表 126.2 MiB / rank / 步** | 本 rank 的 `[base, base+seg)` = **15.78 MiB** | `dspark_dev.rs:3185-3186` |
| `head.weight` `[vocab, dim] bf16` | **整表 1262.5 MiB / rank**（Replicated） | 同分区的 **157.8 MiB** | `dspark_dev.rs:3128-3133`、`weights.rs:90` |
| `logits` 行距 | `vocab` = 129280 | **`seg` = 16160** | `dspark_dev.rs:3129` |
| `markov_embed` | 整表（er 是**全局** token 的行） | **不切** | `glue.cu:1674-1678` |
| 每步跨 rank 往返 | 0 | **+1 轮 v5**（packed key 折叠） | `dspark_dev.rs:3205-3237` |

几何：`vocab=129280`、`world=8` ⇒ `seg = 16160`、`base = rank*16160`（整除 ✓）。
生产参数 `mr = dspark_markov_rank = 256`（`config.rs:116/306`）。

**「切分」不是省字节技巧，是唯一物理使能项**：126.2 MiB 既放不进片上
（148 SM × 227 KB ≈ 33.6 MB）也放不进 L2（60 MB），而 markov 的 5 步严格顺序、
`er` 每步都变（无跨步复用）⇒ 全词表臂的 5 次扫描是 **5 次强制 HBM 通过**。
切 1/8 后 15.78 MiB **进得了 L2** ⇒ 5 次扫描里 4 次是 L2 命中
（`glue.cu:1638-1647`、`draft-1ms-design.md §1.2`）。

### 1.2 三个部件（缺一即回退全词表）

1. **几何判定（唯一权威）** — `markov_head_geom()`（`dspark_dev.rs:2984-3009`）
   返回 `Some((seg, base))`。先过全部回退条件，再谈布局：

   | 条件 | 来源 |
   |---|---|
   | `DSV41_MARKOV_SLICED` 开（默认 OFF） | `dspark_dev.rs:490-493` |
   | `world > 1` | 单 rank 无可切 |
   | `vocab % world == 0` | 16160 × 8 = 129280 ✓ |
   | `head.weight` 是 **BF16**（切过的 head GEMV 是 bf16-only 核） | `dspark_dev.rs:2990-2995` |
   | `.so` 同时有 **两个** 新符号 | `supports_dspark_markov_head_sliced` / `supports_argmax_key_pub` |
   | collective 是 **v5** 且 `8 <= c.bytes`（一个 key 放得进 v5 槽） | `dspark_dev.rs:3000-3004` |

   ⚠️ 符号检查**在布局之前**：陈旧的 `.so` 会让全词表行距继续生效，
   而不是留下一个「半切分的 `logits`」被回退臂误读。

2. **切片核 + 折叠核**
   - `dspark_markov_head_sliced_kernel`（`glue.cu:1689-1805`）：
     **与全词表核逐指令同程序**（`glue.cu:1711-1737` vs `1518-1546`）——
     同 lane→mr 的 float4 fma 链、同 `__shfl_xor` 树、同 partial/ctr 最后一块选举。
     两处差异：(a) `vocab → n`、`v` 是**本地**行；(b) packed key 的低 32 位是
     `0xFFFFFFFF - (idx_off + v)`，即**携带全局 index**。
     **它不写 `ids[step+1]`**，只把本片的 winner 写进 `local_key[0]`
     （`glue.cu:1799`）——刻意不落一个「看起来合理的本地 token」（`44f4956` 的失败模式）。
   - `dsv41_argmax_key_pub`（`dsv41_kernels.cu:7939-7948`）→ `argmax_xchg_v5_kernel`：
     **恰好一轮 v5**（发布 key → stamp → `*epoch` +1 → 轮询 → 取 max → `id = ~low32`），
     不复做本地归约（markov 核已在寄存器里有答案，`dsv41_kernels.cu:7918-7924`）。
   - C 入口 `dsv41_dspark_markov_head_sliced`（`glue.cu:1817-1833`）：
     `markov_head` 传的是**已偏移**的指针；`n` 是切片宽度。

3. **调用点** — `draft_head()`（`dspark_dev.rs:3128-3255`）：
   `geom = markov_head_geom()` → `lg_pitch = seg`、`head_ptr = head + base*dim*2`
   → 一次多行 head GEMV（`head_gemv_bf16_v1_mrows`）
   → `for step in 0..bs`：`markov_head_sliced` + `argmax_key_pub`

### 1.3 epoch 足迹 —— 与文档注释的一处不一致（需实测裁）

`glue.cu:1680-1688` 与 `dspark_dev.rs:472-482` 都写：
**「3 blocks × 5 = 15 extra v5 rounds per `draft_forward`，外加同样的 3 轮 MoE AR」**。

但 HEAD 的代码路径是：`draft_body` 里 `n_mtp` 块循环**之后**只调 **一次** `draft_head()`
（`dspark_dev.rs:1557`），`draft_head` 内 `for step in 0..bs`（`:3180`）每步 **1 次** `argmax_key_pub`
（`:3215`）。markov head 只加载最后一块（`load.rs:945-957`），`self.w.markov_head` 是**单个** `Option`。
⇒ **实际是 5 轮 / draft_forward，不是 15 轮**。

| | 轮数 | 代价 @17.3µs/轮 |
|---|---:|---:|
| 代码路径（HEAD） | **5** | **~87 µs** |
| 代码注释声称 | 15 | ~260 µs |

⇒ **A/B 时用 nsys 数 `argmax_xchg_v5_kernel` 的发射次数**（每步应为 5）——这同时验证了
「注释对不上代码」这条，也给出了 v5 开销的真实值（设计文档 §2.3 估的是 25~50µs，偏乐观）。

**死锁安全性**：draft 在 **每一个 rank** 上以**相同发射序列**运行（权重 replicated + 每块一次 MoE AR）
⇒ 足迹**对称**。这是 v5 不悬挂的结构性理由；`markov_sliced` 的注释明确要求
**gate 必须是 all-ranks gate**——子集开这个 gate 正好破坏该不变量（`dspark_dev.rs:481-482`）。

### 1.4 与 DRAFT_P3A 的关系：**正交，互不依赖**

|  | DRAFT_P3A | MARKOV_SLICED |
|---|---|---|
| 性质 | launch 数折叠（机械） | 词表并行重写（几何 + 通讯） |
| 落点 | **块循环 + rope**（`dspark_dev.rs:1301-1553`、`3363-3377`） | **`draft_head`**（`:3128-3255`） |
| 改字节？ | 否（同核同指令） | 是（per-rank 字节 ÷8） |
| 加 v5 轮？ | 否 | 是（+5/步） |
| gate | `DSV41_DRAFT_P3A`（+4 个 per-item） | `DSV41_MARKOV_SLICED` |
| 共享状态 | `h/pre_in/pre_ffn/post/comb/xn` 等 scratch | `logits/mk_key/mk_partial/mk_ctr/markov_head` |

无共享 buffer、无调用序耦合 ⇒ **可独立 A/B，也可叠加**。

---

## 2. draft P3A 的四项折叠：当前状态

`draft_p3a()`（`dspark_dev.rs:362-377`）：

```rust
let master = env("DSV41_DRAFT_P3A") != "0"          // 未设 => false  (默认 OFF)
item = |name| env(name) 有值 ? (val != "0") : master // 未设 => 跟随 master
DraftP3a { collapse_norm: DSV41_P3A_COLLAPSE_NORM,
           hcpost_swap:   DSV41_P3A_HCPOST_SWAP,
           premix_pp:     DSV41_P3A_PREMIX_PP,
           rope_mrows:    DSV41_P3A_ROPE_MROWS }
```

| 项 | override | 折叠 | 省发 |
|---|---|---|---|
| a1 | `DSV41_P3A_COLLAPSE_NORM` | attn `hc_collapse` + `rmsnorm(attn_norm)` → `dsv41_hc_collapse_norm` | 1/块 |
| a2 | `DSV41_P3A_HCPOST_SWAP` | 两次 `hc_post` 互写对方 buffer，退掉 2 次 `memcpy_d2d(h <- h_out)` | 2/块 |
| a3 | `DSV41_P3A_PREMIX_PP` | premix ping-pong，退掉 `memcpy(pre_in <- pre_ffn)` + 循环前 `premix_init` 拷贝 | 1/块 + 1/步 |
| a4 | `DSV41_P3A_ROPE_MROWS` | `bs` 行 query（及 inverse-query）rope → 一发 `dsv41_apply_rope_mrows` | 8/块 |

- **per-item override 一旦 SET 就压过 master**（`=0` 单独关一项，便于隔离 A/B）。
- 缺失两项（未实现，**不要去找**）：a5 `sparse_attn_orope`（o-rope epilogue 的 position 表达式
  给不出 `pos+r`）、a6 `WOB_F32`（`dsv41_gemm_fp8_mx_f32` 是无行批的 M=1 GEMV）。
- 归属收益：`final-400-config §2.2` 记 **−0.30ms（draft）**。

### 2.1 ⚠️ P3A 的「已开启」是 env 级的 —— A/B 必须读回

HEAD 的**代码默认仍是 OFF**。已开启的证据全部在脚本里：

```
scripts/full_stack_test.sh:14      DSV41_DRAFT_P3A=1
scripts/s2_ab_matrix.sh:137        DSV41_DRAFT_P3A=1   (canonical lazy arm)
scripts/lazy_graph_ab.sh:137       DSV41_DRAFT_P3A=1
scripts/batched_400_v2.sh:141      DSV41_DRAFT_GRAPH=1 DSV41_DRAFT_P3A=1   (batched 臂，禁 lazy)
scripts/nsys_wave1.sh:99           DSV41_LAZY_VERIFY=1 (Wave1 profile 臂)
```

⇒ A/B 前/中**必须** `tr '\0' '\n' < /proc/$PID/environ | grep DSV41_` 读回（§3.1 判据 #2）。

**另一个盲点**：`DraftP3a` 的解析结果**没有任何打印**（不像 `verify_head_mrows_note` 有 note_once）。
a1~a4 全跟随 master 时「4 项全开」是**推断**，不是证据。
低成本补强（A/B 准备的一部分，1 处 4 行）：在 `draft_p3a()` 的 `get_or_init` 里加一条一次性
`eprintln!("[draft] P3A collapse_norm={} hcpost_swap={} premix_pp={} rope_mrows={}", ...)`，
落在既有的「print, do not silently degrade」纪律内。

---

## 3. A/B 设计

### 3.1 矩阵（同二进制、背靠背、交错）

两个 gate 都是 `OnceLock` **每进程读一次** ⇒ 一臂一进程。**交错跑**（A B A B A B）抵消时钟/热漂。

```bash
# 两臂共同（Wave-1 lazy 链 + P3A 恒开）
COMMON="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1
        DSV41_TIMING=1 DSV41_DSPARK_DEBUG=1
        DSV41_LAZY_VERIFY=1 DSV41_VERIFY_GRAPH=1
        DSV41_EXPERT_ACT_E4M3=1 DSV41_SH_EXP_MROWS=1
        DSV41_DRAFT_P3A=1"                 # ← 两臂都开，作为 base

# 臂 A（base）:  marker, 期望 draft ≈ 4.0ms（若 P3A 已兑现 −0.30）
# 臂 B（+slice）: DSV41_MARKOV_SLICED=1   期望 draft ≈ 3.5~3.7ms
```

**明确排除的项**（会混淆 −0.3ms 量级）：

| 排除 | 理由 |
|---|---|
| `DSV41_DRAFT_GRAPH=1`（P3c） | 独立杠杆（`draft-graph-p3c.md`），且需 `pos>=win` + `ar_v5`；应单独 A/B，不与 P2 混 |
| `DSV41_DRAFT_HEAD_FOLD=0` | 默认 ON（v1-order fold，与 verify 同程序）。toggle 它会引入 v2-fold 的数值差（`dspark_dev.rs:73-98`） |
| `DSV41_DRAFT_MOE_MROWS` / `P3B` | 各自独立 A/B：`final-400-config §3.2` 记 −0.2~0.35、−0.5~0.8，属补刀 |
| `DSV41_SEED_POS` / `SEED_ALIGN` | 会改 draft 的 position 语义（`dspark_dev.rs:274-306`） |
| mrows 族（`GATE_MROWS`/`INDEXER_MROWS`/…） | m=1 下恒为 0（`lazy-verify-optimization-path §3.2`） |

✅ **建议**：若 P3A 的基线尚未实测（§0-4），把矩阵扩成三臂：
`P3A=0` → `P3A=1` → `P3A=1 + MARKOV_SLICED=1`。**第一段顺带回答「4.28 是否含 P3A」**，
成本只是多一个进程。

### 3.2 readout（每臂都要）

| # | 项 | 命令/位置 | 判据 |
|---|---|---|---|
| 1 | **主判据** `draft=` | `[dspark] steps=N … draft=X.XXms verify=… commit=…`（`dspark_verify.rs:100-112`；打印点 `chain_dev.rs:7148/7387/7640/7857/8205`） | 中位 `draft=` 的 A→B 差 |
| 2 | **gate 真在** | `tr '\0' '\n' < /proc/$PID/environ \| grep DSV41_` | `DSV41_MARKOV_SLICED=1` 与 `DSV41_DRAFT_P3A=1` 同时可见 |
| 3 | **符号真在**（跑之前） | `nm -D --defined-only kernels/cuda/libdsv41.so \| grep -E 'dsv41_(dspark_markov_head_sliced\|argmax_key_pub)'` | **两个都必须在**；缺一个 ⇒ gate 返回 `None` ⇒ A/B 测的是全词表臂（本项目 #1 陷阱） |
| 4 | **核真在跑**（nsys） | `nsys profile … scripts/nsys_wave1.sh` | `dspark_markov_head_sliced_kernel` 与 `argmax_xchg_v5_kernel` 各 **5 次/步**（§1.3） |
| 5 | **文本逐字** | 出师表 300 tok × N≥3，md5 + `双字` 计数 | A/B md5 相同、`双字=0`、无拉丁残片 |
| 6 | **predict 质量** | `[spec e2e] mean-k=`（`dspark_verify.rs:110`） | mean-k 落在 A 臂的噪声带内（数值域若被动，这里先红） |
| 7 | **无死锁** | `grep -c 'ar5-hang'`；超时内跑完 | =0 且正常退出（⚠️ 见 §4-h） |
| 8 | **无故障** | `[spec e2e] … faults=` | `faults=0` |
| 9 | 行级 parity（可选，若跑 `dspark_parity`） | `DSV41_INV_CHECK=1` | `verify_bad == 0` |

### 3.3 期望值与判读区间（**先钉死，再看数**）

| 来源 | 值 |
|---|---:|
| head 切片省字节（1104.7 MiB/rank @ 4.5TB/s） | −245 µs |
| markov 切片省字节（~615 MiB/rank @ 4.5TB/s） | −137 µs |
| 加：5 轮 v5 往返（5 × 17.3 µs） | **+87 µs** |
| **净（字节账）** | **≈ −295 µs** |
| `draft-1ms-design §2.3` 自述（P2 上限） | −330 µs |
| `final-400-config §2.2` 记 | **−1.00 ms** ← 无出处 |
| `lazy-verify-optimization-path §3-L5` 记（P3A+slice 合） | −0.70 ~ −0.80 ms |

⇒ **判据**：
- **PASS**：`draft=` 降幅 ≥ **0.20ms** 且 §3.2 #5/#6/#7/#8 全绿。
- **中性（不是失败）**：降幅 0.05~0.20ms。这个量级与「字节账 −0.3 被 v5 往返吃掉一半」一致；
  若如此，**不要下「切分无效」的结论**——它的物理价值（L2 驻留）本来就在 P3（段级融合）里兑现
  （`draft-1ms-design §1.2/§4`：切分是 P3 的**前置件**，head 段的 158 MiB 几何）。
- **NEG**：降幅 < 0.05ms 或 mean-k 移动 ⇒ 先查 #3/#4（gate/so 是否真生效），再查 §4。
- **异常**：降幅 > 1.5ms ⇒ 怀疑混淆项（`DRAFT_GRAPH` 漏进 env）或计时口径变了。

---

## 4. 数值域风险评估（draft 的预测质量）

**结论：三项子变换各自是逐位恒等，合起来是「全词表 argmax 的精确重排」。
但 head GEMV 的切片是唯一沾数值的面，必须靠 A/B 文本门确认。**

| # | 面 | 风险 | 依据 / 处置 |
|---|---|---|---|
| a | **head GEMV 的行距从 129280 → 16160** | ⚠️ **最需要盯的一项** | `head_gemv_bf16_v1_mrows` 每行 `acc[r]` 是**独立升序链**，注释明写「行值不依赖 row→warp 映射」（`glue.cu:585-587`）。grid 随 n 变（`ceil(n/8)` cap 4096）只改行的**分派**，不改行的**求和** ⇒ 逐位不变。**旁证**：verify 侧的同一刀默认已 ON（`VERIFY_HEAD_SLICED` `unwrap_or(true)`，`chain_dev.rs:5860-5907`），且 draft 走的正是 verify 的同款 v1 程序（`dspark_dev.rs:73-98` 记录了 v2-fold 被判为数值变化的历史）。 |
| b | markov bias | 低 | `lrow[v] += <wr,er>`：同 float4 fma 链、同 `shfl_xor` 树（`glue.cu:1711-1737` vs `1518-1546`）。逐行恒等。 |
| c | argmax / tie-break | 低（**但这是正确性的命门**） | key = `(monotone(value)<<32) \| (0xFFFFFFFF-(idx_off+v))`，全局取 max ⇒ 等于全词表 argmax，含「值同则取最小全局 index」。全词表核用的是同一个 `0xFFFFFFFF - v`（`:1543`）⇒ tie 规则一致。 |
| d | **head 与 markov 的分区必须同构** | ⚠️ 静默错误面 | 若两者用不同的 `(seg, base)`，`logits[v]` 会被 `markov_head[v+k]` bias，**仍然产出一个看似合理的 token**（「pitch is not in the type」，`dspark_dev.rs:3110-3115`）。代码把决策收敛到 `markov_head_geom()` 一处，但**运行时没有断言**（两侧类型不同，`debug_assert_eq!` 写不出来）⇒ **A/B 的文本门就是这道守卫**。 |
| e | `idx_off + v` 溢出 | 无 | 最大 113120 + 16159 = 129279 < 2³¹。 |
| f | `-0.0` 符号 | 无 | 两个核共用 `dspark_markov_f2key`（`:1479-1482`）。 |
| g | `mk_partial/mk_ctr` 容量 | 无 | seg=16160 ⇒ blocks = ⌈16160/64⌉ = **253** ≤ `MARKOV_MAX_BLOCKS=2048`（`dspark_dev.rs:46-48`）；无需扩容。 |
| h | **单 key 折叠核无 watchdog 打印** | ⚠️ 观测缺口 | `argmax_xchg_v5_kernel` 轮询 5s 后**静默 break**（`dsv41_kernels.cu:7789-7796`），随后从**陈旧 parity 半量**里取 max ⇒ 发出一个「看似合理但是错的」token——正是 `44f4956` 的失败模式。多行 twin 有 `[ar5-hang]` printf（`:7871-7873`），单 key 版**没有**。⇒ A/B 准备的一部分：把那 3 行 printf 补到单 key 核（1 处改动），否则一次悬挂在日志里只表现为「文本变了」。 |
| i | rank 非对称 | ⚠️ 死锁面 | 只要有一个 rank 走全词表（.so 不同 / comm 非 v5），轮数就错位。实践中 8 rank 同二进制同 staging；**A/B 必须查每 rank 日志的 `faults`/超时**。 |

**若 mean-k 或文本 md5 变了**：按 (d) → (a) 的顺序查。
先确认两处 `(seg, base)` 同源（读 `markov_head_geom` 的唯一调用点），
再用 `DSV41_DRAFT_HEAD_FOLD=0` 把 head 打回逐行——若差异消失，锁定 (a)；
若仍在，锁定 (b)/(c)（但 (b)/(c) 按上表应为恒等 ⇒ 那说明是 (d)）。

---

## 5. 与 lazy verify 的兼容性（「m=1」下的有效性）

**问：MARKOV_SLICED 在 m=1 下是否有效？答：有效，且不被稀释。** 三条依据：

1. **lazy verify 不改 draft 的行数。** lazy 改的是 *verify* 逐行 `step_rows(m=1)`；
   draft 仍然每步调 **一次** `draft_forward(token, pos)`、固定 `bs=5`
   （`chain_dev.rs:8203/7855/7638`），markov 循环是 `for step in 0..bs`。
   ⇒ 「m=1」作用在 verify 侧的 per-row 族，作用不到 draft 的 markov。
2. **MARKOV_SLICED 不是行折叠。** mrows 族（`GATE_MROWS`/`INDEXER_MROWS`/`VERIFY_HEAD_MROWS`/…）
   是「把 m 行折成 1 发」⇒ m=1 时 rows=1，与逐行等价，收益恒 0（`lazy-verify-optimization-path §3.2`）。
   MARKOV_SLICED 是「词表切 1/8 + 固定 5 步」⇒ **与 m 无关**。
3. **它的 v5 往返是每步常数，不是 `×k_emit`。** lazy 的 per-step 族（hc 链 / AR）被 `×k_emit`
   重付，是 lazy 的隐藏税；draft 每步只跑一次 ⇒ 切分的 +5 轮是 **常数**，
   **不会被 `k_emit` 放大**。这正好让 L5 成为 lazy 下「干净」的一项。

**要求（均需在 A/B 中保持恒定，使所有 rank 取同一臂）**：
- `DSV41_LAZY_VERIFY=1` 在 A/B 全程不动（路由由 `lazy_route_decide` 按运行期量选臂，
  环境固定 ⇒ 各 rank 的臂选择一致）。
- 不引入 `DRAFT_GRAPH`：`draft_graph_arm` 要求 `pos >= window_size` + `ar_v5`
  （`draft-graph-p3c.md §4`）；且捕获会把 markov+sliced 折叠**录进图**。
  这不是不能做（v5 是设备侧协议，可录；`verify_head_geom` 的「capture-independent」纪律
  `chain_dev.rs:5862-5874` 给了模板），但属于 **P3c 的独立 A/B**，不进本轮的 −0.3ms 账。
- ⚠️ 若日后开 P3c：`markov_head_geom()` 在 DRY / CAPTURE / REPLAY 三处**必须返回同一结果**
  （它是 env + `.so` 驱动且 `OnceLock` 缓存 ⇒ 天然满足，但要写进记录）。

---

## 6. 最小执行清单（给下一轮会话）

**本地（无 GPU）**
1. `cargo check --workspace`
2. `nm -D --defined-only kernels/cuda/libdsv41.so | grep -E 'dsv41_(dspark_markov_head_sliced|argmax_key_pub)'` → **两个符号都在**
3. （准备项，1~2 处小改，落「print, do not silently degrade」纪律）
   - `draft_p3a()` 加一次性 P3A 决议打印（§2.1）
   - `markov_head_geom()` 加一次性 `seg/base/world` 打印（可选，与上同类）
   - 单 key 折叠核补 `[ar5-hang]` printf（§4-h）

**GPU（一次会话，交错 3~6 轮）**
4. `ab_markov_slice.sh`（由 `ab_shmrows.sh` 克隆：`launch` / `gen` / `gens` / `envchk`，增 `sochk`）
   - tag `base`：COMMON
   - tag `mslice`：COMMON + `DSV41_MARKOV_SLICED=1`
   - （若基线未知）tag `nop3a`：COMMON − `DSV41_DRAFT_P3A=1`
5. 每轮取 §3.2 的 #1/#2/#5/#6/#7/#8；任一轮加一次 nsys 取 #4。

**成文**
6. 把实测 `draft=` 增量、nsys 的 5 vs 15 轮裁断、v5 往返真实开销写回
   `lazy-verify-optimization-path.md §3-L5` 与 `final-400-config.md §2.2/§3.2`
   （后者现在的 `−1.00ms` 需要被实测定值取代或标注为乐观口径）。

---

## 附：一句话总结

**`MARKOV_SLICED` 是把 draft 的 `markov_head`（126.2 MiB）与 `head.weight`（1262.5 MiB）按词表
切到 8 个 rank（各 15.78 / 157.8 MiB），用一轮 v5 packed-key 折叠把本片 winner 合回全局 token——
它的物理意义是「让 5 次 markov 扫描中的 4 次命中 L2」，因此是 P3 段级融合的前置件，而不是一个
独立的 −1ms。
诚实期望 **−0.3 ~ −0.5ms（被 5 轮 v5 往返吃掉一部分）**；数值上三项子变换各自逐位恒等，
唯一需要 A/B 确认的是 head GEMV 的行距变化；与 lazy verify 正交、不被 `×k_emit` 稀释。**

---

*工部 · 只读分析 + 本文件（唯一产出），未执行 GPU 命令、未改动任何源码。*
*所有 ms 均标来源（账面换算 / 设计口径 / 实测锚点）；本机无 GPU ⇒ 新增项均待 V3 会话实测。*
