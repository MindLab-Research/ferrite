# Wave 5 实施文档 — 多并发（MTP × batched，padded 与 ragged）

> 基线：`/home/smith/src/ferrite` @ `a8336e1`（Wave 3 step 1，GLM parked-seq prefix cache 已落地）· 2026-09-12
> 只读侦察，未改任何代码。本文件所有 `file:line` 均已读码核对；推断处标 ⚠️。
> 上游：`docs/agent/unified-engine-battle-plan.md`（Wave 2/3 状态）、`docs/agent/dspark-perf-400-plan.md`、`docs/agent/wave4-prefill-plan.md`、`docs/agent/perf-roadmap.md:1664-1675`（MTP 禁令与解除）

---

## 0. 一句话结论

**GLM 的非-MTP batched 解码已经是成熟品**（`decode_step_batched` + 按 padded size 的图池 + per-seq pointer tables + 共享 dummy state）；Wave 5 的**全部新增工作量**在三处：

1. **GLM MTP × batched**：把 `MtpState`（per-rank 单例）拆成 per-seq，并把 verify 图从 `n=FERRITE_MTP_N` 扩到 `B×n` 行（新尺寸类 `megab_b{3B}`）；
2. **DSV41 batched**：`TpRankPool` 的 batch-1 lockstep 升级为 padded-B 的批处理（复用 GLM 的图池 + 指针表模式，`step_rows` 已有 m 行机器可复用）；
3. **MTP × batched 的 ragged verify**：每 seq 的 draft 块独立 → 行数 = Σ(1+k_i)（ragged）或 B×(k+1)（padded），二选一（§3 给了权衡）。

**风险最高的是 DSV41 的显存**：per-seq ring/index_k 在 `DSV41_MAX_POS=64k` 下实测 **≈5.2 GB/序列/rank**（§2.5 实测账），B=8 就是 ~42 GB——这决定 DSV41 的 batched 必须先做 **per-seq 状态共享/页化**，否则 B 根本抬不起来。

---

## 1. 现状矩阵（四组合）

| 组合 | 状态 | 证据 | 缺的机制 |
|---|---|---|---|
| **GLM batched（无 MTP）** | ✅ **成熟** | `ferrite-exec/src/tp.rs:896` `decode_step_batched`；图池 `tp.rs:1003-1009`；pad `tp.rs:1010-1011`；`gpu_engine.rs:980-1014` 调度；per-seq 表 `cuda.rs:3432`(gdn) / `cuda.rs:3512`(dsa)；dummy `cuda.rs:3467-3482`, `cuda.rs:3630` | 无（B>16 的 kernel 快路径退化是已知项，见 §5） |
| **GLM + MTP × batched** | ❌ **互斥** | `gpu_engine.rs:634-639` **强制** `max_seqs=1`（"MtpState is a per-rank singleton"）；`gpu_engine.rs:939-979` MTP 走 per-seq round-robin；`tp.rs:3244` `mega_chain_dev_batched` **无 VerifyIO 参数**（注释 `tp.rs:3240-3241` "Non-MTP path"） | ①`MtpState`→`[B]` 宽 / per-seq（`cuda.rs:763`）②verify 图 `mega_v` 的 `B×n` 行 + 新尺寸类 ③draft 链的 B 行批化（`tp.rs:1684` `mtp_step` 逐 seq）④`ferrite_mtp_commit` 的逐 seq 批化（`cuda.rs:345`）⑤`dsa_append_batched` 的 row→seq 映射（`cuda.rs:4683`，n=3B 时 seq=row/3、tok=row%3） |
| **DSV41 batched（无 dspark）** | ❌ **batch-1 排队** | `ferrite-dsv41/src/serve.rs:120` `TpRankPool`（注释 "all executing the same request in lockstep"）；`serve.rs:46` `RankCmd::{Prefill,Decode,DecodeRun}` 全单请求；`serve.rs:229` `broadcast` 收全部 rank 回执 = lockstep；`serve.rs:265` `impl StepEngine`；`serve.rs:1040` `build_serve_engine` 包成 `SingleFlight<TpRankPool>`（`ferrite-http/src/single_flight.rs:101`）；**`--max-seqs` 对 dsv41 完全忽略**（`main.rs:64,187` 只传给 GLM 路径） | ①`StepEngine`→`ServeEngine`（B 行）②per-seq ring/clen/index_k（§2.2）③RankCmd 的批协议 ④AR staging 随 B 扩容 |
| **DSV41 + dspark × batched** | ❌ **batch-1** | `chain_dev.rs:4682` `dspark_spec_step`：verify 块固定 5 行、**单个 pos_base**（`chain_dev.rs:3220-3221`）；`chain_dev.rs:4463` `dspark_shadow_step` 同理；`dspark_dev.rs:75` `DsparkDev` 的 window ring **per MTP block**（`dspark_dev.rs:118-119`），非 per-seq | ①per-seq 的 verify 块（ragged，§3）②per-seq draft state（`DsparkDev` 需 `B` 份或 per-seq 切片）③per-seq `note_ctx_rows` 的 ring 基址 |

**一句话**：GLM 的 batched 是"照抄对象"，DSV41 的 `step_rows` 是"半个现成"（m≤6 行机器已存在，但绑定单序列），MTP×batched 是"真正的空地"。

---

## 2. DSV41 的 batched 路线

### 2.0 关键前提：DSV41 已有 m 行机器，但绑定单序列

`chain_dev.rs` 的 `step_rows`（`chain_dev.rs:3205`）→ `step_rows_inner`（`chain_dev.rs:3466`）→ `layer_rows`（`chain_dev.rs:4983`）→ `attention_rows`（`chain_dev.rs:5128`）/ `moe_rows`（`chain_dev.rs:5934`）**已经实现了 m 行（m ≤ `VERIFY_ROWS=6`，`chain_dev.rs:84`）的批 forward**：

- MoE：`expert_gate_up_fp4_batched` / `expert_down_fp4_batched` / `swiglu_limit_batched` 已是 **rows=m** 的原生多行 launch（`chain_dev.rs:6092,6130,6154`；kernel `dsv41_experts_mxf4.cu:2402`）。
- head：`head_gemv_bf16_mrows`（`device.rs:2941`，kernel 1..=8 `dsv41_glue.cu:1143-1150`）。
- hc/norm/route：`hc_mixes`/`hc_collapse`/`hc_post`/`rmsnorm`/`route_topk` 都有 rows 维。
- **q/kv/o 投影与 sparse_attn 仍逐行**（`chain_dev.rs:5152-5193,5293-5385`，注释 `chain_dev.rs:5120-5127`：这是为了"每行与单行路径逐位一致"的 parity 保障）。

**关键差异（单序列 → 多序列）**：

| 维度 | 现在（单序列 m 行） | 多序列 B 行需要 |
|---|---|---|
| 位置 | `pos_rows[r] = pos_base + r`（**连续**，`chain_dev.rs:3220-3221`）| 每行的 seq 各自的位置 → **per-row pos 表**（`pos_rows` 已是 device 数组，`chain_dev.rs:382`，只需换成 B 个不同的值）✅ 结构现成 |
| KV ring | `self.layers[owner].ring`（**单份**，`chain_dev.rs:5269` `ring_ptr`）| **per-seq ring 指针表**（照抄 GLM 的 `gdn_state_tables`/`dsa_ptr_tables`，`cuda.rs:3432/3512`） |
| clen | `self.s.clen[owner]`（单份，`chain_dev.rs:5275`, `chain_dev.rs:216`）| **per-seq clen 表** `[B][n_layers]` |
| index_k | `self.layers[key_owner].index_k`（`chain_dev.rs:5842`）| per-seq index_k 指针表 |
| compressor state | `layers[l].state_kv/state_score/latent/out_rows`（`chain_dev.rs:63-75`）| per-seq |
| AR payload | `m*dim`（staging `serve.rs:392` 按 `VERIFY_ROWS*dim`）| `B*m*dim`（**必须扩容 staging**，见 §2.4） |
| pos_ctr | 单个 device 计数器（`chain_dev.rs:207`）| per-seq 计数器（或 B 行 pos 表 + 单计数器只做簿记） |

**结论**：DSV41 batched **不是从零写 kernel**，而是**把"per-序列状态"变成"per-行状态"**——这正是 GLM 在 `decode_step_batched` 里做的事（`tp.rs:1010-1011` pad + `tp.rs:1174-1206` 刷新 per-size 指针表）。

### 2.1 要改的文件/函数清单

**A. `crates/ferrite-dsv41/src/serve.rs`（rank 命令协议 → 批协议）**

| 项 | 现状 | 改法 |
|---|---|---|
| `RankCmd`（`serve.rs:46`） | `Prefill(Vec<u32>)` / `Decode{token,pos}` / `DecodeRun{token,pos,n}` | 新增 `DecodeBatch { rows: Vec<RowStep> }`（`RowStep{seq, token, pos}`）；`Prefill`/`DecodeRun` 保留为单请求回退 |
| `TpRankPool`（`serve.rs:120`） | `lookahead: VecDeque<u32>`、`res: Receiver<(rank, Vec<u32>)>` | 保留；`res` 的 payload 从 `Vec<u32>` 变 `Vec<(seq, u32)>` 或保持 rank0 返回 B 个 argmax（rank 间一致） |
| `broadcast`（`serve.rs:229`） | 收全部 rank 回执 | 不变（lockstep 不变，batch 是"同一命令跑 B 行"） |
| `impl StepEngine`（`serve.rs:265`） | `prefill`/`decode` 单请求 | **改为 `impl ServeEngine`**（`ferrite-http/src/engine.rs:174`），或新增 `BatchStepEngine` trait（`submit`/`tick`/`output`/`cancel`/`deregister`/`status`/`live_rows`） |
| `build_serve_engine`（`serve.rs:1040`） | 返回 `SingleFlight<TpRankPool>` | 改返回 `TpRankPool`（直接 `ServeEngine`），或 `BatchedShim<TpRankPool>`；`--max-seqs` 从此生效（`main.rs:64` 已解析，只需接入） |
| `pool_rank_body`（`serve.rs:349`） | `rank_loop` 的 lockstep 循环 | 新增 `DecodeBatch` 分支：对每个 live seq 调 `chain.step_rows`/新的 batched step；dspark 分支见 §3 |

⚠️ **`TpRankPool::new` 的 rank 线程只建一个 `DevChain`**（`serve.rs:401` `let mut chain = DevChain::new(...)`）。批处理要么（a）**一个 chain + per-seq 状态切片**（推荐：内存可控，见 §2.5），要么（b）**B 个 chain**（内存 = B × 5.2GB，不可行）。

**B. `crates/ferrite-models/src/dsv41/chain_dev.rs`（per-seq 状态）**

| 项 | 现状（单序列） | 改法 |
|---|---|---|
| `LayerCache`（`chain_dev.rs:60`） | `ring`（`chain_dev.rs:64`, alloc `chain_dev.rs:1518`）、`idxs`、`state_kv/state_score/kvp/scp/latent/out_rows`、`index_k`（`chain_dev.rs:75`） | 新增 `LayerCacheBatch`：把 `ring`/`index_k`/`state_*` 从 `DevBuf` 变 **`[B]` 指针表**（`DevBuf` 数组或一张 device 指针表，照 GLM `cuda.rs:3432` 的 `gdn_state_tables` 模式） |
| `Scratch.clen`（`chain_dev.rs:216`, alloc `chain_dev.rs:1561`） | `[n_layers] i32` | `[B][n_layers] i32` + 每行的 base 偏移 |
| `pos_ctr`（`chain_dev.rs:207`） | 单个 device 计数器 | per-seq `[B]` 计数器；或 B 行 `pos_rows`（`chain_dev.rs:382`）填 B 个不同的值，`pos_ctr` 只做行内簿记 |
| `step_rows` / `step_rows_inner`（`chain_dev.rs:3205/3466`） | `pos_base` 标量 + `pos_rows[r]=pos_base+r` | **新函数 `step_rows_batch(rows: &[RowStep])`**：`pos_rows[r] = rows[r].pos`，`seq_of_row[r]` 映射表；其余逻辑保持 |
| `attention_rows`（`chain_dev.rs:5128`） | `ring_ptr` 单份（`chain_dev.rs:5269`）、`clen_owner` 单份（`chain_dev.rs:5275`） | 循环内按 `r` 取 **per-seq ring/clen**（`ring_ptr[r]`/`clen_ptr[seq_of_row[r] * n_layers + owner]`）；`indexer_rows_one`（`chain_dev.rs:5763`）同样按行取 `index_k`/`idx_lens` |
| `compress_row`（`chain_dev.rs:5959`）/ `indexer_rows_one` | `self.layers[layer]` 单份 | 按行取 per-seq 切片 |
| `moe_rows`（`chain_dev.rs:5934`） | 已 rows=m | **不变**（唯一需要确认的是 `ex_act_r` 等 `_r` scratch 的行容量，`chain_dev.rs:1636-1651`，当前按 `VERIFY_ROWS=6` 分配 → 需按 `B` 扩容，见 §4） |

**C. `crates/ferrite-models/src/dsv41/tp.rs`（AR）**

- `Collective::all_reduce_inplace`（`tp.rs:528`）的 `len` 参数：多行时 = `m*dim`，多序列时 = `B*m*dim`（payload 随 B 线性增长）。
- **staging 必须扩容**：`serve.rs:392` `ar_bytes = (hc_dim.max(VERIFY_ROWS*cfg.dim))*4` → batched 需 `B*VERIFY_ROWS*dim*4`（⚠️ 这就是 `serve.rs:386-391` 注释里"垫错会静默 desync 成 wedge"的同一失败模式，**必须先改这里**）。
- `ar_v5()`（`tp.rs:690`）是图捕获的前置（`serve.rs:3366` 的 gate）；batched 下 keep v5。

**D. kernel 侧（`kernels/cuda/dsv41_*.cu`）**

- **`dsv41_sparse_attn` 已是 `(b,m)` 形状**（`dsv41_kernels.cu:6777` grid `(b*m, h)`）→ 多行/多序列直接可用，**但注意它的 `clen` 是单指针**（`dsv41_kernels.cu:6779`）→ 多序列必须逐行传不同的 clen（当前 `attention_rows` 就是逐行传，`chain_dev.rs:5370-5384`）✅ 无需改 kernel。
- `dsv41_indexer_topk`：**硬上限 `b*m ≤ kIdxMaxRows=8` 就报错**（`dsv41_kernels.cu:2574, 6932`）。B>8（或多行 m 叠加）**必须抬高 `kIdxMaxRows`**，而它连带 `g_idx_score[kIdxMaxRows][kIdxMaxPos]`（`dsv41_kernels.cu:2576`）膨胀：8×65538×4 B = 2 MB 现在放 `__device__`；抬到 32×65538×4 = 8.4 MB 可接受，抬到 512 就是 128 MB ❌。⇒ **B≤16 的路线必须先重构 `g_idx_score` 的尺寸策略**（见 `wave4-prefill-plan.md:172` 的同一条警告：把 score 收进 `kIndexerChunk` 内联循环，删掉全局数组）。
- `head_gemv_bf16_mrows` 的 `1..=8` 分派（`dsv41_glue.cu:1138`）→ B>8 会 `Ok(false)` 回退逐行（慢但正确）；B≤8 直接可用。

### 2.2 数据流：一次 batched tick（DSV41）

```text
serve driver (B live seqs)
   │  RankCmd::DecodeBatch { rows: [(seq0,tok0,pos0), (seq1,tok1,pos1), ...] }
   ▼
TpRankPool::broadcast  ──▶  8 rank 线程（lockstep 不变）
                              │
                              ▼
                        DevChain::step_rows_batch(rows)      # 新
                          ├─ embed_expand_dev(rows)          # rows=B
                          ├─ for layer: layer_rows_batch(layer, B)
                          │     ├─ hc/rmsnorm rows=B          （现成）
                          │     ├─ attention_rows: 逐行，但 ring/clen 从 per-seq 表取
                          │     ├─ AR: payload B*dim          （staging 需扩容）
                          │     └─ moe_rows(rows=B)           （现成，scratch 需扩容）
                          └─ head_gemv_bf16_mrows(rows=B)     （B≤8 现成）
                              argmax per row → B 个 token
   ◀── rank0 返回 B 个 token（B 个 seq 各推进 1）
```

### 2.3 DSV41 prefill

`prefill_chain`（`serve.rs:701`）逐 token forward、`Prefill(Vec<u32>)` 单请求（`serve.rs:503`）。Wave 5 **不要求** batched prefill；但要注意 `RankCmd::Prefill` 与 `DecodeBatch` 的**互斥**（prefill 一进来就重置 chain，`serve.rs:269` `lookahead.clear()`）。短期保持"prefill 串行插入、decode 批处理"的 piggyback 语义即可（对应 `ferrite-dispatch/src/batch.rs:470-530` 的 `PrefillMode::Piggyback`）。

### 2.4 AR staging 扩容（**必做前置**）

```rust
// serve.rs:392 现状
let ar_bytes = (hc_dim.max(crate::chain_dev::VERIFY_ROWS * cfg.dim)) * 4;
// 改为
let max_rows = max_batch_size * crate::chain_dev::VERIFY_ROWS;   // ragged 时是上界
let ar_bytes = (hc_dim.max(max_rows * cfg.dim)) * 4;
```
⚠️ `Collective::new` 的 `bytes` 决定 staging `[2][world][bytes]`（`tp.rs:127-128`）与 `slot_stride_elems`（`tp.rs:258`）；**slot 越界写会污染邻 rank 的 parity 半区并 desync 成 wedge**（`serve.rs:386-391` 的实测教训）。

### 2.5 显存账（DSV41，**决定 B 上限**）

读码实测（`chain_dev.rs:1494-1518` 的 alloc，`DSV41_MAX_POS` 默认 65536）：

| ratio | ring `(window+max_comp)*hd*4` | index_k `max_comp*index_hd*4` |
|---|---|---|
| 1 | **134.5 MB** | 33.6 MB |
| 2 | 67.4 MB | 16.8 MB |

生产配置 `compress_ratios [0,0,2×18,1×20,0,0,0]`（`config.rs:14`）：40 层合计 **≈5.21 GB/序列/rank**（ratio=0 层按 `max(1)` 分配，见 `chain_dev.rs:1504`）。

⇒ **B=8 需 ~42 GB/rank**（还没算权重）。B300 180 GB 单卡放得下，但和权重挤在一起会紧。
⇒ **Wave 5 的 DSV41 batched 必须同时做 per-seq 状态的"页化/共享"**，否则 B 抬不上去。可选：
1. **ring 按 max_pos 页化**：ring 只在 `pos < max_pos` 范围内分配（当前就是满配）；改成"按请求实际 max_new 分配"（`submit` 时已知 `prompt+max_new`，`gpu_engine.rs:179` GLM 已有此模式）。
2. **index_k 与 ring 的压缩半区共享**（consumer 层本就读 owner 的，`chain_dev.rs:5320-5322`）——已经共享了 ring，但 `index_k` 每层独立 alloc（`chain_dev.rs:1521`），可只给 `is_index_source` 的层分配。
3. **B 的分档**：B∈{1,2,4} 先落地（内存 ×4 = ~21 GB 可承受），B≥8 等页化。

---

## 3. MTP × batched 的 ragged verify

### 3.0 问题定义

每个 seq 独立跑 draft → 得到自己的 draft 块 `d_r = [d_1..d_{k_r}]`，长度 `k_r` 可能不同（DSV41 dspark 固定 5；GLM MTP `FERRITE_MTP_N` 默认 3，drafts = n-1）。verify 要把 B 个 seq 的 verify 行拼进**一次** batched forward。

两种拼法：

### 3.1 方案 A — Padded `B × (k+1)`（推荐起步）

```text
行布局: [ seq0: a0 d1 d2 d3 ] [ seq1: a1 d1 d2 d3 ] ...   # 每 seq 固定 k+1 行，不足补 PAD
行数  : B * (k+1)          （DSV41: B*6 = 6B；GLM: B*3 = 3B）
row→seq: seq = row / (k+1)   （常量除法，kernel 里可编译期算）
pos[row]: seq 自身 pos + (row % (k+1))
```

| 维度 | 评估 |
|---|---|
| **空间** | 固定 `6B` 行缓存；DSV41 的 `_r` scratch 从 `VERIFY_ROWS=6` 扩到 `B*6`（`chain_dev.rs:1609-1651`）；B=16 → 96 行 |
| **时间** | 浪费 = `B*(k+1) - Σ(1+k_r)` 行。**但实际上短块只在尾部出现**（prefill 后第一轮 anchor 行、max_new 截断），稳态下每 seq 都满 5 drafts → 浪费 ≈ 0 |
| **实现** | 简单：`row→seq = row/(k+1)`、`pos = pos_base[seq] + row%(k+1)`、attention mask 用 PAD 行的 `pos = -1` 或 `clen=1` 使其自注意力（dummy 策略见 §4） |
| **排序坑** | `attention_rows` 的**逐行 interleave 顺序**必须保持（`chain_dev.rs:5242-5262` 的 audit 缺陷 #1/#2：block-wide 读会把"行 r 的 future commit"算进 row r 的 clen）→ padded 布局下行的顺序仍是 seq-major，**同 seq 内 row 递增即天然正确** ✅ |
| **AR payload** | `B*(k+1)*dim`（比 ragged 大，但 staging 已按 `max_rows` 分配，§2.4） |

### 3.2 方案 B — Ragged `Σ(1+k_r)`（后续优化）

```text
行布局: [ seq0: a0 d1 d2 ] [ seq1: a1 d1 d2 d3 d4 ] ...    # 恰好 Σ 行
行数  : Σ_r (1+k_r)
row→seq: 需要一张 row→seq 映射表（device i32 数组，每 tick 由 host 写）
```

| 维度 | 评估 |
|---|---|
| **空间** | 省 `B*(k+1) - Σ` 行（稳态下 ≈0；尾部长块差异大时省得多） |
| **时间** | 无浪费行；但**每行要查 `seq_of_row[row]`**（一次 global load，相对 40 层的权重流量可忽略）；**attention 需要行级 mask**（同一 batched launch 里不同行落在不同 seq 的 KV 上——`sparse_attn` 逐行调用时天然隔离，`chain_dev.rs:5370-5384`，所以**只要逐行传对的 ring/clen 就行，不需要 mask**）✅ |
| **实现** | 比 A 多一张映射表 + 行的可变 pitch；`row→seq` 的常量除法变查表；pos 表本来就是 per-row device 数组（`chain_dev.rs:382`）→ 与 A 相同 |
| **额外收益** | 与 **GLM MTP 的 accept 计数**对齐：GLM 的 `k_acc` 逐 seq 不同（`tp.rs:1993` commit），ragged 能让"本轮 verify 的行数"直接等于"诚实需要的行数" |
| **风险** | 行数**逐 tick 变化** → CUDA 图要么按行数分档（`1..=max_rows` 每档一张图，图数量爆炸），要么**不进图**（DSV41 的 `DSV41_VERIFY_GRAPH` 本就默认 OFF，`chain_dev.rs:3314`）。若要求图化，ragged 与"固定形状图池"天然冲突 ⇒ **ragged 只适合非图化路径** |

### 3.3 结论：A 起步，B 作为"图化关闭"时的优化

- **先做 A（padded B×6 / B×3）**：固定形状 → 可进图池（`megab_b{6B}` / `megab_b{3B}`），与 GLM 现有 `[1,2,4,8,16,32]` 图池同构，**只需在图池 ladder 里加 48/96 档**（`tp.rs:1021` + `gpu_engine.rs:1006`）。
- **B 只在两个条件同时满足时才做**：(a) 尾部短块造成的浪费实测 >10%；(b) 该路径不需要 CUDA 图（DSV41 dspark 的 verify 当前就非图化）。
- **判断依据**：`dspark-perf-400-plan.md` 已把 verify 目标定成"6 行一次 forward"（吞掉主链步，`dspark-perf-400-plan.md` §三），稳态下块长固定 ⇒ **A 的浪费 ≈ 0，B 的收益 ≈ 0**。A 就够了。

### 3.4 GLM MTP × batched 的具体行数与图名

GLM 的 verify 是 `FERRITE_MTP_N` 行（默认 3 = `[t_last, d1, d2]`，`tp.rs:1464-1480`）。B 个 seq 拼起来：

- **padded**：行数 = `B * n_v`；B=16, n_v=3 → **48 行**。
- **现有图池只到 32**（`tp.rs:1021`）⇒ **必须新增尺寸类**（32 → 64，或直接 `48` 精确档；`tp.rs:1003-1008` 的 `find(|&s| s >= n)` 需要 48/64 在表里）。
- 图名：`megab_b{48}`（沿用 `format!("megab_b{size}")`，`tp.rs:1009`/`gpu_engine.rs:1011`）。
- **MtpState 需要 per-seq**：`mtp_setup_bufs`（`tp.rs:2556`）现在分配**一份**共享 scratch（`cuda.rs:763` 的 `MtpState`）。B 行 verify 需要每个 (seq, GDN layer) 一组 `(conv_a, gdn_a, conv_b, gdn_b, conv_snaps, gdn_snaps)`（`tp.rs:2570-2581`）→ **要么 `[B]` 扩容，要么 per-seq 指针表**（照抄 `gdn_state_tables`）。
- **draft 链**：`mtp_step`（`tp.rs:1684`）现在对单 seq 跑 nd 步 draft graph（`mega_d{seq}_{i}`）。B 行批化 = 每 seq 的 draft 图 `mega_d{seq}_{i}` 已是 per-seq 的（`tp.rs:1739`）→ **天然可分**，但要共享的 `MtpState` 缓冲（`emb_devs`/`h_d`/`d_argmax_dev`，`tp.rs:2637-2648`）拆 per-seq。
- **commit**：`ferrite_mtp_commit`（`cuda.rs:345`）的 plan 是 per-layer 6 指针表（`tp.rs:2596-2621`）→ 需要 `[B]` 行版本（每行自己的 A 侧 state 基址）。

---

## 4. pad 策略

### 4.1 GLM 已有（照抄即可）

| 机制 | 位置 | 说明 |
|---|---|---|
| **padded size 图池** | `tp.rs:1003-1009` / `gpu_engine.rs:1006` | `[1,2,4,8,16,32]` 找 `≥n` 的最小档；一张图服务该档所有成员组合 |
| **u64::MAX 哨兵** | `tp.rs:1011` `pseqs.resize(size, u64::MAX)` | 补齐的行用 `u64::MAX` 标记"非真实 seq" |
| **shared dummy state（GDN）** | `cuda.rs:3467-3482` | pad 行的 conv/gdn 指针指向 `(u64::MAX, layer)` 的共享 dummy；输出丢弃 |
| **dummy DSA cache** | `cuda.rs:3562-3572`, `cuda.rs:3630` | pad 行指 `dsa_dummy(family,...)`：**MAXT=8192 的 `k_nope/v/k_idx/k_gate`**（与真实 cache 同 token 维，否则 Xid 13 越界），但 `pinned_total=1`（`cuda.rs:3670`）——让 kernel 走**快路径**（total=1 → indexer 的 `select_k >= jmax` 快路径、sparse_attn 1 slot）。曾经 `total=8192` 时每退休一个 seq 加一个 dummy，每个 DSA 层的 indexer 走满 2048-pool 慢路，实测 1.56ms/launch（`cuda.rs:3657-3662`）|
| **表内容按需刷新** | `tp.rs:1174-1206` | 只在 membership 变化时刷 per-size 指针表（每步刷 ~2ms 主机时间） |
| **图不销毁** | `gpu_engine.rs:694-733` | retire 后保留 `megab_b{size}` 图（指针表内容是刷新的，不内嵌 seq 指针） |

### 4.2 Wave 5 需要新增的 pad 规则

| 场景 | 规则 |
|---|---|
| **DSV41 batched 的 pad 行** | 需要一个 **per-seq 的 dummy ring / dummy clen**：pad 行的 `ring_ptr` → 共享 dummy ring（`window+max_comp` 大小，零初始化即可，因为输出丢弃）；`clen` 恒为 1（同 GLM 的 `total=1`，走快路径）；`pos_rows[row] = 0`（避免读越界） |
| **DSV41 indexer 的 `b*m` 上限** | pad 行也计入 `b*m`！`kIdxMaxRows=8`（`dsv41_kernels.cu:2574`）意味着 **B + pad 后 ≤8**。B=16 必须先做 §2.1D 的 `g_idx_score` 重构 |
| **GLM MTP verify 的 pad 行** | verify 的 `[B][n_v]` 输入里，pad 行填 `PAD_TOKEN`（`bucket.rs:42` = 0）；DSA 的 `dsa_append_batched` 的 `ntok` **必须为 1**（`cuda.rs:4683` 的 ROOT CAUSE #3 注释：grid=(B,ntok) 读 `kvb+(seq+tok)*row`，ntok>1 就跨 B 读越界）——**MTP verify 的 3B 行必须走新的 row→(seq,tok) 映射**（seq=row/n_v, tok=row%n_v），不能复用 ntok 参数 |
| **MTP commit 的 pad 行** | `ferrite_mtp_commit`（`cuda.rs:345`）的 plan 表要为 pad 行指向 dummy（A 侧 state 是 dummy，写回丢弃）；`k_pin` 的 pad 行填 0（accept=0） |
| **DSV41 dspark 的 pad 行** | draft 块长度不一的 seq（尾轮）→ padded 布局下补 `PAD_TOKEN`，pos 填 `≤0` 使 RoPE 数学安全（`apply_rope` 的 `off` 为负会算错，需查 `dsv41_kernels.cu` 的 rope kernel 是否 clamp ⚠️ 待验证） |

---

## 5. 单流不掉速的机制（用户验收点）

用户的验收：**"多并发下单流不严重下滑"**。会随 B 增长的项（逐个列 + 对策）：

### 5.1 逐项清单

| # | 随 B 增长的项 | 证据 | 增长形态 | 对策 |
|---|---|---|---|---|
| 1 | **AR 载荷** | GLM: `nccl.all_reduce_f32(partial, ..., n*hidden)`（`tp.rs:3113,3637`；`p2p` `tp.rs:4544,4796`）；DSV41: payload `m*dim`（`tp.rs:528`）+ staging `serve.rs:392` | **线性** B×hidden（GLM hidden=5120 → B=16 是 320 KB/AR；P2P_AR_MAX_N=16×8192 `tp.rs:737` 正是为此） | AR 是 **NCCL ring / P2P v5** 的固定会合延迟 + 线性载荷；对策：①P2P v2/v5（`tp.rs:737` 注释：35µs→10µs）②**AR 与下一层计算 overlap**（`perf-roadmap.md:1675` 的"AR/compute overlap"，B 越大越值得） |
| 2 | **MoE 的唯一专家数** | GLM 实测：B=16 唯一专家 ~104/288，B=32 → ~200（主 agent 侦察） | **次线性**（B 增大 → 覆盖更多专家 → 权重流量上升） | 这是 MTP-batched @B=32 实测只有 4.5x（而非 2x 本征伸缩）的主因之一。对策：①**expert-major 分组**（`perf-roadmap.md:1676`）②B=16 是甜蜜点（唯一专家饱和前） |
| 3 | **attention 的 KV 流量** | GLM DSA：`sparse_attn_v2` 按 (B, topk) 读；DSV41：`sparse_attn` grid `(b*m, h)` | **线性** B×t | 单流成本**不变**（每 seq 读自己的 KV）；关键是不能让 B 行的 KV 读取**串行化**——`sparse_attn` 已是 `(b*m, h)` 并行 grid ✅ |
| 4 | **hc 链** | `moe_layer_dev`/`hc_mixes` 的 rows 维（`chain_dev.rs:4994`，`cuda.rs:5328`） | **线性**（B 次，但每行独立） | 已批化（rows=B 一次 launch），成本是 B 倍算术但**一次 launch**（发射开销不随 B 线性） |
| 5 | **小 kernel 的 launch 数** | GLM batched 链 ~700 节点/步（`perf-roadmap.md:1674`）；DSV41 的 q/kv/o 投影逐行（`chain_dev.rs:5152-5193`） | **线性**（若不批化） | ①GLM：已进图（replay 零发射）②DSV41：`step_rows_inner` 的逐行投影要改成 rows=B 的 GEMV（注意**数值域**：`chain_dev.rs:3156` 注释明确"rows 路径 pin 了已验证的 GEMV 分支"） |
| 6 | **per-seq 状态指针表的刷新** | `tp.rs:1174-1206`（仅 membership 变化时） | **O(B)** 但只在成员变化时 | 已有对策（缓存 `last_batch_seqs`）；DSV41 照抄 |
| 7 | **图的 capture 抖动** | `tp.rs:996-998`（composition-keyed 图 → 16 请求流进来 8 次 capture → 74 tok/s） | 只在**未预热**的档位 | 已对策：按 padded size 预热（`gpu_engine.rs:1001-1011`）；Wave 5 加 48/96 档后要**在启动时预热**（`gpu_engine.rs:220` 的 `FERRITE_NCU` 窗口已经是"batch 饱和才开捕获"的现成钩子） |

### 5.2 量化预期（GLM，主 agent 侦察）

- B=32 实测 4.5x（~2x 本征伸缩 + MoE 唯一专家 104→200 + **n>16 的 kernel 回退**）。
- n>16 回退的证据：`cuda.rs:2425,2494`（`n >= 2 && n <= 16` 的 fp8 MMA 快路径）、`cuda.rs:2262,3970`（`n > 16` 拒绝 gemm3/bf16-GEMM 快路径）、`cuda.rs:2494`（`n <= 16 && small_n_rows` 的 GEMV 快路径）。⇒ **B>16 掉进 tiled GEMM，128 行 tile 严重浪费（`cuda.rs:2501-2505` 实测 n=4 batched 105ms vs n=1 16ms）**。
- 修正后 MTP-batched @B=16 预测 **~1400-1600 tok/s**（超当前但低于 3200）。
- ⇒ **对策**：把 `small_n_rows` 的 GEMV 路径扩到 `n≤32`（或在 B∈{17..32} 时显式走 GEMV），是提 B 上限最直接的一刀。

### 5.3 单流（B=1）的"不掉速"保证

关键：**B=1 必须走单流路径，不能因 batched 代码存在而降级**。GLM 已有：
- `gpu_engine.rs:988-999`：`live_seqs.len()==1` 且 `FERRITE_FORCE_BATCHED_B1` 未设 → 走 `decode_step`（per-seq mega GEMV）。实测 batched B=1 图比 mega B=1 **慢 1.9x**（`gpu_engine.rs:990-994`：17.95ms vs 9.55ms）。
- `cuda.rs:4334-4338` / `cuda.rs:4536-4540`：`gdn_layer_dev_batched(n==1)` / `dsa_layer_dev_batched(n==1)` 转发回单行路径。
- **Wave 5 必须保持这条**：DSV41 batched 落地后，B=1 也要走 `step_rows(m=5)` 或 `step_dev`，不能强制走 batched 图。

---

## 6. 实施顺序 + 验收

### 6.1 依赖图

```text
P0  基线固化（无：先量 B=1 单流 tok/s，作为"不掉速"的对照）
 │
 ├─▶ P1 GLM MTP×batched（独立，价值最高：B=16 单流不掉的 MTP）
 │      P1a MtpState → per-seq       ──▶ P1b verify 图 B×n 行 + 新尺寸类
 │      P1c draft 链批化             ──▶ P1d commit 批化
 │      P1e dsa_append_batched 的 row→(seq,tok) 映射
 │
 ├─▶ P2 DSV41 batched（独立）
 │      P2a AR staging 扩容（serve.rs:392）★前置
 │      P2b per-seq ring/clen/index_k 指针表
 │      P2c step_rows_batch + RankCmd::DecodeBatch
 │      P2d impl ServeEngine（替 SingleFlight）
 │      P2e kIdxMaxRows 重构（B>8 才需要）
 │
 └─▶ P3 ragged verify（P1/P2 稳定后，按 §3.3 的判据决定做不做）
```

### 6.2 每步的 A/B 方法

| 步 | 改动 | A/B 口径 | 通过标准 |
|---|---|---|---|
| **P0** | 无 | `FERRITE_TIMING=1` 记 B=1 的 `[megab] replay` / `[dsv41] step pos=` 单流 tok/s | 存档作对照 |
| **P1a** | `MtpState` per-seq（`tp.rs:2556`） | `FERRITE_MTP=1 --max-seqs=1` 的四段文本 + 逐位 token 序列 vs 现状 | **文本不变、逐位一致**（MTP 的 iron law，`ferrite-dispatch/src/mtp.rs:47-57`） |
| **P1b** | verify 图 `B×3` 行 + ladder 加 48/64 | B=2 的 verify 单测：两 seq 的 token 序列 = 分别单跑拼接；`FERRITE_LAYER_SUM=1`（`tp.rs:3391`，N 个相同 prompt 必须逐位一致） | 逐位一致 + `[megab]` 无 Xid 31 |
| **P1c/d** | draft 链 + commit 批化 | B=2 的 accept 率 vs 单流（应相同，因为每行数值域不变） | **accept 率不降**（drop >0.05 即回归，`mtp.rs:47-57`） |
| **P1e** | `dsa_append_batched` row→(seq,tok) | B=2 MTP 的 DSA cache 内容 vs 单流逐 seq 跑 | cache 逐字节一致 |
| **P1 总验收** | | B∈{1,2,4,8,16} 的聚合 tok/s + **B=1 单流 tok/s**（与 P0 对照） | 聚合随 B 单调升；**B=1 单流 ≥ P0×0.98**；B=16 单流 ≥ B=1 的 1/16 × 0.7（"不严重下滑"） |
| **P2a** | staging 扩容 | B=1 的四段文本不变（扩容不该影响数值） | 文本不变 |
| **P2b** | per-seq 指针表 | B=1 与单流逐位一致（`step_rows_batch` 在 B=1 时退化为 `step_rows`） | 逐位一致 |
| **P2c/d** | `DecodeBatch` + `ServeEngine` | B=2 并发两请求 vs 串行两请求的**输出文本** + 各自的 TTFT/TPOT | 文本一致；聚合 tok/s ≥ 串行的 1.5x（初版目标，不追 2x） |
| **P2e** | `g_idx_score` 重构 | B=8/16 的 `indexer_topk` 不再返回 `cudaErrorInvalidValue`（`dsv41_kernels.cu:6932`） | B=16 无错；与 B≤8 的文本一致 |
| **P3** | ragged | 若做：对比 padded 的 verify_ms 与 scratch 峰值 | 浪费 >10% 才做 |

### 6.3 每个里程碑的"停机"条件

- **P1 若 verify 图 `B×3` 行捕获即崩（Xid 31）** ⇒ 回退到"per-seq 图 + 逐 seq replay"（B 个 `mega_v{seq}`，无 batched，但 `max_seqs` 可开 >1）——**这是 P1 的降级路径，不是失败**。
- **P2 若 per-seq 显存 >60 GB** ⇒ B 限 4，先做 §2.5 的页化。

---

## 7. 需要太子补充的信息

1. **用户对 B 的目标档位**：GLM 要 B=16？B=32？（§5.2 显示 B>16 会掉进 tiled GEMM）。DSV41 的 B 上限受显存约束（§2.5），是 B=4 还是 B=8？
2. **DSV41 是否要求 batched prefill**？（当前 `prefill_chain` 逐 token，`serve.rs:701`；§2.3 假设不做）
3. **ragged verify 是否必须图化**？（决定 §3.3 做 A 还是 B）
4. **`DSV41_MAX_POS` 的生产值**：默认 64k（`chain_dev.rs:1500`），若生产用 1M 则 per-seq 显存 ×8（§2.5 的页化从"可选"变"必须"）。

---

## 8. 建议分工

- **吏部（代码质量）**：`TpRankPool`/`DevChain` 的 per-seq 状态改造前的接口审查——`StepEngine`→`ServeEngine` 的 trait 变更（`single_flight.rs:51` / `engine.rs:174`）涉及面广，先审命名与生命周期。
- **户部（资源/性能）**：§2.5 的显存账 + §5.1 的逐项成本；产出"B 上限 vs 显存"表，作为 P2 的准入条件（`Collective` staging 与 DSV41 per-seq 状态的 sizing）。
- **兵部（安全）**：AR staging 越界（`serve.rs:392`）、`dsa_append_batched` 的 `ntok` 越界（`cuda.rs:4683`）、`u64::MAX` 哨兵与 `free_seq` 的 UAF（`gpu_engine.rs:694-733` 的"证书不销毁"依赖指针表内容刷新）——这三处都是"静默 desync 成 wedge"的历史坑。
- **刑部（bug 审查）**：`attention_rows` 的逐行 interleave 顺序（`chain_dev.rs:5242-5262`）在 padded/ragged 布局下的正确性；pad 行的 pos 边界（§4.2 最后一行）。
- **礼部（文档）**：更新 `perf-roadmap.md:1664-1675` 的 MTP 禁令记录（已于 2026-09-12 解除）+ `unified-engine-battle-plan.md` 的 Wave 5 状态。
- **工部（实现）**：P1a→P2d 的代码落地。**P1 与 P2 可并行**（不同 crate），但 P1e 与 P2b 都动 `cuda.rs` 的指针表机制 → **需先约定接口**（`gdn_state_tables`/`dsa_ptr_tables` 的 DSV41 对应物）。

---

## 附录 A：关键 `file:line` 索引

**GLM batched**
- `crates/ferrite-exec/src/tp.rs:896` `decode_step_batched`
- `crates/ferrite-exec/src/tp.rs:1003-1011` padded size + u64::MAX 哨兵
- `crates/ferrite-exec/src/tp.rs:1174-1206` per-size 指针表刷新
- `crates/ferrite-exec/src/tp.rs:3244` `mega_chain_dev_batched`（**无 VerifyIO**）
- `crates/ferrite-kernel/src/cuda.rs:3432` `gdn_state_tables` / `cuda.rs:3512` `dsa_ptr_tables`
- `crates/ferrite-kernel/src/cuda.rs:3630` `dsa_dummy`（MAXT=8192, total=1）
- `crates/ferrite-serve/src/gpu_engine.rs:980-1014` batched 调度；`gpu_engine.rs:634-639` MTP 强制 max_seqs=1
- `crates/ferrite-dispatch/src/bucket.rs:38` `BUCKET_LADDER`；`bucket.rs:42` `PAD_TOKEN`

**GLM MTP**
- `crates/ferrite-exec/src/tp.rs:1684` `mtp_step`；`tp.rs:2556` `mtp_setup_bufs`；`tp.rs:2649` `MtpState`
- `crates/ferrite-kernel/src/cuda.rs:763` `MtpState` 定义（per-rank 单例）
- `crates/ferrite-kernel/src/cuda.rs:345` `ferrite_mtp_commit`
- `crates/ferrite-dispatch/src/mtp.rs:1-57` 协议与 iron law；`mtp.rs:65-67` `DRAFT_DEPTH/VERIFY_WIDTH`

**DSV41**
- `crates/ferrite-dsv41/src/serve.rs:46` `RankCmd`；`serve.rs:120` `TpRankPool`；`serve.rs:229` `broadcast`；`serve.rs:265` `impl StepEngine`；`serve.rs:392` AR staging；`serve.rs:1040` `build_serve_engine`
- `crates/ferrite-models/src/dsv41/chain_dev.rs:3205` `step_rows`；`:3466` `step_rows_inner`；`:4983` `layer_rows`；`:5128` `attention_rows`；`:5934` `moe_rows`；`:4682` `dspark_spec_step`；`:84` `VERIFY_ROWS`
- `crates/ferrite-models/src/dsv41/chain_dev.rs:1494-1521` per-seq alloc（ring/index_k）；`:1504` `max(1)` 的 ratio=0 处理
- `crates/ferrite-models/src/dsv41/dspark_dev.rs:75` `DsparkDev`；`:65` `DSPARK_DRAFTS=5`
- `kernels/cuda/dsv41_kernels.cu:2574` `kIdxMaxRows=8`；`:6932` 硬报错；`:6777` `dsv41_sparse_attn` 的 `(b,m)`
- `kernels/cuda/dsv41_glue.cu:1138` head mrows 的 `m>8` 拒绝

**共享**
- `crates/ferrite-http/src/single_flight.rs:51` `StepEngine`；`:101` `SingleFlight`
- `crates/ferrite-http/src/engine.rs:174` `ServeEngine`
- `crates/ferrite-dispatch/src/batch.rs:109` `SchedConfig`；`batch.rs:470` `plan_into`
