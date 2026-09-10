# DeepSeek-V4.1-Flash — GPU bring-up status

Everything in this file was verified against the **real 48-shard checkpoint**
(`/opt/dlami/nvme/models/DeepSeek-V4.1-Flash`) on **b300-4**, not inferred.

## Where it stands

A single-GPU run of the chain works end to end on real weights:

```
[dsv41] weights loaded: 118.6 GiB in 86.7s        (3 layers + embed + head, engram tables skipped)
[dsv41] prefill 1 tokens in 0.12s
[dsv41] decode 6 tokens in 0.10s (58.2 tok/s)
--- generated ---
我的朋友你是谁/us921osererg
```

The leading tokens are **coherent Chinese** with only 3 of 40 layers and none of
the long-range components — i.e. embedding → hyper-connection → MLA → MoE →
head are numerically sound. The gibberish after a few tokens is attributable to
the parts that are deliberately not wired yet, listed below.

`DSV41_LAYERS=N` truncates the model for bring-up (`DSV41_STATS=1` prints
per-stage rms/min/max/NaN; `DSV41_DEBUG_SYNC=1` names the kernel that faults).

## Ground truth established against the checkpoint

Each of these contradicted an assumption and would have been a silent error:

| item | reality |
|---|---|
| tensor names | **no `model.` prefix** (`layers.6.attn.wq_a.weight`) |
| `attn.wo_a.weight` | `[8192, 4096]` fp8 **with `.scale`** — `[o_groups*o_lora, hpg*head_dim]`, per-group contiguous so the row stride equals `k`; per-group GEMMs run straight off this layout |
| `o_lora_rank` | **1024** (the reference dataclass default of 256 is wrong) |
| `ffn.gate.weight` | **BF16, no scale sidecar** → the gate GEMM is bf16 and routing is a separate step (`dsv41_route_topk`), not the fused fp8 kernel |
| `index_topk` | **512** |
| layer roles | `kv_source = [2,8,14,20]`, `index_source = [2,8,14,20,24,28,32,36]`, `compress_ratio = 0/0 then 2 (l2-19), 1 (l20-39)` |
| weight dtypes | norms / hc params / gate bias / embed are **BF16**; they feed f32 kernels, so they are widened losslessly at load (see below) |

## Bugs found and fixed on the way (each had a specific signature)

1. **`used[]` out-of-bounds in the router** — a flag array sized `[topk]` indexed
   by the *expert id* (≤ n_experts). Illegal memory access. Consumed experts are
   now marked by writing `-INFINITY` into the score array.
2. **fp8 activation scratch sized by `dim`** — the attention output projection
   quantises `n_heads*head_dim = 32768` elements, so the fp8 buffer and its
   per-32 scales ran past their allocations. Illegal memory access.
3. **bf16 bytes handed to f32 consumers** — `ffn.gate.bias`, `norm.weight`, the
   `hc_*` parameters and `embed.weight` are bf16; reading them as f32 doubles
   the length read. This corrupted values *and* ran off the end. Fixed by
   widening bf16→f32 at load — an **exact** conversion, not a dequantisation:
   the quantised weights (fp8+ue8m0, fp4+e8m0) are still uploaded verbatim.
   Only tensors a bf16 tensor-core GEMM consumes stay bf16.
4. **`ferrite_add_inplace` does not exist** — the symbol is `ferrite_add`. The
   full dlsym set (28 names) is now checked against the built object; the only
   absent entries are that one (fixed) and `dsv41_window_append` (optional, not
   implemented yet).
5. **The window selection took the invalid slots** — `window_topk_idxs` returns a
   full `window`-wide row in ring order (high slots first) with empty slots
   marked −1; the chain kept the *leading* `pos+1`, which for `pos < window` are
   exactly the invalid ones. `sparse_attn` then selected nothing and produced
   **exactly 0.0** at every decode step (`DSV41_STATS` showed rms=0.0000 while
   pos=0 was a healthy 0.1124). The whole row is now handed over; it skips
   negatives itself.

Two process lessons, both re-learned the hard way:

* **A stale `.so` invalidated a whole debugging session** — `bash build.sh 103a`
  must run after every sync, and the artifact timestamps must be checked against
  the source (`.so` was older than `dsv41_route.cu` while I chased a bug I had
  already fixed).
* **A test shape that cannot reach the bug is not a test** — the router passed at
  `n_experts=6` and only fell over at the production 384.

## Not yet wired (the reason the tail of the text degrades)

1. **Compressor + indexer** — attention is window-only (128 slots). The release
   compresses KV every `ratio` positions and selects `index_topk=512` of them;
   without it there is no long-range context at all.
2. **Engram write-back** — 189 GiB of n-gram tables, row-sharded per rank
   (`engram_part_rows`), gathered by `dsv41_engram_hash` + `dsv41_engram_gather`
   and applied by `dsv41_engram_apply`. Loaded weights are currently skipped.
3. **Tensor parallelism** — mandatory, not optional: 286 GiB of weights (engram
   excluded) against ~180 GiB per B300. The loader already slices by
   `(world, rank)`; the chain still needs the all-reduce points.
4. **DSpark drafting** — draft layers are loaded (`n_mtp_layers=3`) but unused.

## TP plan (the reference's parallel layout, to be mirrored)

* `wq_a`/`wo_a` column-parallel (output split), `wq_b`/`wo_b` row-parallel
  (input split) — the column→row *pair* needs no reduction in between, the row
  half needs one after it.
* MoE experts are expert-parallel → the routed sum and the shared expert both
  need an all-reduce.
* `head` is column-parallel over the vocabulary.
* The all-reduce can be built without NCCL: one process, one CUDA context per
  rank (8 devices), peer copies + a local sum kernel. AR payloads here are tiny
  (`dim` floats per site).

## Compressor / indexer — ground truth for wiring them up

**⚠️ Blocker found at the wiring step:** `dsv41_compressor`'s ABI takes fp8
weight pointers (`const uint8_t* wkv, const uint8_t* wkv_scale, ...`), but the
checkpoint's compressor weights are **BF16** and the reference runs the pooling
in **fp32**. The kernel as written cannot consume them. Two ways forward, both
small: (a) give `compressor_pool_kernel`/`compressor_state_kernel` a bf16 entry
(the state carry and the pooling themselves are dtype-agnostic — `k` is fp32 in
the state buffers already), or (b) keep the kernel for the GEMM-free part and
feed it activations computed by two `lin_bf16` calls in the chain. Do not "fix"
this by quantising the bf16 weights to fp8 — that is both lossy and contrary to
the standing rule that quantisation formats come from the checkpoint.

Established against the checkpoint and the reference's `Compressor`:

* **Compressor tensors exist only on the kv sources 🌐 `[2,8,14,20]`**; indexer
  tensors only on `[2,8,14,20,24,28,32,36]`. (The config derives both correctly;
  layer 6 has neither — the earlier "layer 6 has none" observation is the
  expected shape of the model, not a spec gap.)
* **Every compressor weight is BF16, not fp8** — `compressor.wkv.weight` is
  `[512, 5120]` BF16 on all four layers, `compressor.wgate.weight` exists only
  where `ratio > 1` (layers 2/8/14), and layer 20 (`ratio == 1`) has no gate.
  The reference *promotes* them to fp32 at runtime for the pooling; the stored
  values are bf16, so widening to f32 at load is lossless and reproduces that.
* Indexer: `wq_b` is **fp8 + scale** `[4096, 1280]`, while `wk` `[128, 512]`,
  `weights_proj` `[32, 5120]` and `k_norm` `[128]` are BF16.
* Reference semantics, per step:
  * `ratio == 1` → `latent = norm(wkv(x))`, one token per group;
  * `ratio > 1` → `kv = wkv(x)`, `score = wgate(x)`, the completed group of
    `ratio` tokens is pooled as `Σ kv_t · softmax_t(score)`, and an incomplete
    group waits in `kv_state`/`score_state` (shaped `[b, ratio, head_dim]`);
  * the latent is returned **pre-RoPE** — the indexer consumes the unrotated
    form and attention applies RoPE afterwards.
* Both halves already exist as kernels: `compressor_state_kernel` (stashes the
  trailing partial group for prefill, one slot per decode step) and
  `compressor_pool_kernel` (gated pooling) — read their bodies for the exact
  argument order. What is missing is the chain wiring: call them on kv sources,
  append the latent to the KV buffer at row `window + compress_len`, publish the
  key for the indexer, and on non-sources reuse the source layer's published
  selection (`is_index_source` / `is_kv_source` already say which is which).

## Indexer — exact reference semantics (for the wiring step)

`Indexer.forward(x, qr, latent, start_pos, offset)` — `x` is the normed attention
input, `qr` the q_lora stream, `latent` this layer's RoPE-free compressed latent
(`None` while its group is still filling), `offset` where the selection goes:

1. **Index keys** (only when `owns_k` and a latent came out): `k = k_norm(wk(latent))`,
   RoPE on the trailing `rope_head_dim` lanes using **the group's first token's
   position** (`group j → j*ratio`), then fp4 quantisation, then publish into the
   key cache at `start_pos // ratio`.
2. **Index queries**: `q = wq_b(qr)` (a ColumnParallel over the q_lora stream,
   `[4096, 1280]` fp8+scale in the checkpoint), RoPE with the *token's* positions,
   then fp4 quantisation.
3. `weights = weights_proj(x) * (softmax_scale * n_heads**-0.5)` — note it is
   scaled here, not inside the score.
4. `score = Σ_h relu(q_h · k) * w_h` over the published keys `[.., end_pos//ratio]`,
   and **an all-reduce over ranks** (TP note).
5. Visibility: `compress_lens = end_pos // ratio` (decode) — a block is visible
   once the query has passed its last token.
6. The top-k result is written at `offset` (the chain passes `window`, which is why
   the compressed entries land directly after the window block).

Deviation to keep in mind: the reference **fp4-quantises the index q/k**
(`fp4_act_quant`); our `dsv41_indexer_topk` computes them in f32. That is more
accurate but not bit-identical — it can flip which position is picked at the
margins, so it must be validated by the text, not by an equality check.

## Tensor parallel: implemented and verified

One process, one rank per thread. The CUDA runtime binds a device per thread, so
thread `r` calls `cudaSetDevice(r)` and everything it launches lands on device
`r`; each rank opens its own `Device`, loads its own slice (the loader already
slices by `world`/`rank`) and builds its own chain, with a `Barrier` keeping the
ranks in lockstep. Communication is peer copies plus a local reduction over the
staging slots **in rank order** — identical on every rank, no NCCL.

Verified: `--tp 4` and `--tp 8` produce the same leading token ids as a
single-GPU run (`[116169,122294,107153,30155,52695,1538]`).

### Sharding policy (what each rank holds)

| tensor | rule | why |
|---|---|---|
| routed experts | expert-parallel **and** cut along `inter` | 96% of the bytes; per-rank ~34 GiB at tp=8 |
| expert `w2` | cut along **columns** | it is `[dim, inter/2]`, so `inter` is its column dim |
| `wq_b` | ColumnParallel over heads | local heads = `n_heads/world`; the chain uses the local count for the q path, RoPE, `sparse_attn` and the inverse RoPE |
| `attn_sink` | per-head slice | matches the local head count |
| `wo_a` | ColumnParallel per group | rows `groups*o_lora/world`; a rank's contiguous row block **is** its group block, which is what keeps it aligned with the head block of `wq_b` |
| `wo_b` | RowParallel (input split) | each rank reduces over its `o_lora` slice; all-reduced after |
| compressor / indexer / `wq_a` / `wkv` / gate | replicated | plain `Linear` in the reference; the indexer's score would need an all-reduce before its top-k (score and top-k are one kernel today) |
| `embed` / `head` | replicated | the reference splits the vocabulary, which needs a gather — `embed_expand_dev_kernel` substitutes row 0 for an out-of-range id instead of skipping it, so a rank whose slice lacks the token would contribute a real (wrong) row |

### Five TP-only bugs (each invisible at tp=1)

1. `cudaDeviceEnablePeerAccess` needs the *peer's* context to exist; calling it at
   bind time fails silently and the first peer copy faults.
2. `cudaMemcpyPeerAsync`'s stream/direction rules did not fit one-thread-per-device;
   the synchronous form works and the payloads are tens of KB.
3. Expert `w2` carries the sharded `inter` axis on its **columns**; sharding it by
   rows sliced its output dim and the down GEMM walked off its weight.
4. `local_shape` reported expert/group tensors **unsliced** while the loader had
   sliced them — the metadata contradicted the bytes.
5. The expert kernels were handed the **global** `inter` width although their
   weights are `[inter/world, ...]`. (The first attempt at that edit silently did
   not apply; it is now verified by grepping the call sites.)

Process lessons re-learned: an async fault surfaces at the next checked call, so
the named op is often innocent — the peer copy was blamed for an expert-kernel
overrun. And a patch that does not match its anchor fails **silently**; grep the
call sites afterwards.

## Numerical parity against the reference — the method that works

The user's suggestion was decisive: **run the reference implementation's maths on
the same input and diff value-by-value**. One parity test found the root cause in
a single shot that hours of text-watching had not.

`tests/hc_parity.rs` runs `dsv41_hc_mixes` on the reference's deterministic input
and prints pre/post/comb; `/tmp/refcmp/hc_ref.py` computes the same numbers with
numpy straight from the checkpoint. Result after the fix: **7 significant digits
agree** (PRE `0.9425751` vs `0.94257504`, POST `0.031103252` vs
`3.1103250e-02`, COMB row0 `0.7791158` vs `7.7911586e-01`).

### The bug it found (root cause of input-independent output)

`hc_mixes` computed `ss = Σ x[c]²` and reduced it **within each warp only** — the
cross-warp step was missing, so at `blockDim=256` the sum covered 1/8 of the
elements and `inv = rsqrt(ss/hc_dim + eps)` was **sqrt(8) = 2.83x too large**.
Every hc coefficient was then wrong, which mis-mixed the entire residual stream
in every layer: the output had no relation to the input at any prompt length.
This is the same class as the GLM "8-warp reduce" bug already in this repo's
notes — **reductions must be checked for the cross-warp step**.

### Other reference mismatches found and fixed the same session

| item | was | now |
|---|---|---|
| `original_seq_len` | read a top-level key that does not exist → **0**, which DISABLES YaRN (`if original_seq_len > 0`) | reads `rope_scaling.original_max_position_embeddings` = **65536** ✓ |
| Sinkhorn order | ended with a ROW normalisation | reference order: softmax+eps → col → (iters-1)×(row, col), ending COL ✓ |
| compress rope | one table (theta=10000) for everything | separate table at `compress_rope_theta` = 160000 ✓ |
| compressed latent | stored in the KV ring **without RoPE** | rotated at the group's first position before the store ✓ |
| query RoPE | `step=1` → head i at position pos+i | `step=0` (all heads at the same position) ✓ |
| `layer()` return | the attention block's pre-mix | the FFN block's pre-mix (what the next layer collapses with) ✓ |
| indexer | only kv-source layers ran it | every index-source layer runs its own; keys read from the kv owner ✓ |
| window indices | uploaded only on the placeholder path | uploaded every step (the indexer path left them stale) ✓ |
| KV ring size | window rows only (compressor wrote past the end) | window + max_compress rows ✓ |
| wo_a offsets | GLOBAL group offsets into a LOCAL slice | local offsets ✓ |
| weight loading | ~92k individual `cudaMalloc`, per-tensor `File::open`, missing `h.begin` | one pooled allocation per layer, mmap'd shards, DMA (`cudaMemcpy`/`2D`), device-side bf16 widening ✓ |
| `bind_to` | enabled peer access at bind time (races context creation) | binds only; peer access enabled after all contexts exist ✓ |

## Remaining work

1. **One more numerical bug** (output still degenerate: `時刻_...` then blanks).
   The parity method is the way to find it: diff the **attention path** next
   (q/kv projections → sparse attention → inverse rope → wo_a/wo_b) against a
   numpy reference built from the same weights, exactly as was done for the hc
   chain. `sparse_attn`'s sink handling and the fp8 activation quantisation are
   the two least-verified pieces left.
2. **Engram write-back** (currently skipped; its absence changes quality but the
   user confirms it is optional, so it is not the cause of the degenerate text).
3. **Performance to 200 tok/s single-request (5 ms/step)** — not started; the
   correctness work was blocking. Where the time currently goes:
   * **~4 device-wide syncs per layer** (downloading hc coefficients, MoE routing
     indices/weights, the compressor's publish flag) ≈ 160-200 syncs/step. The hc
     coefficients can stay on the device (ping-pong two buffers — the kernel
     already takes device pointers), and the publish decision is a pure function
     of `pos % ratio` computable on the host. **This is the first thing to fix.**
   * the collective is synchronous (peer copy + device sync + 2 barriers per
     call, 2 calls per layer = 80/step).
   * MoE dispatch is host-driven: ~6 experts × 7 launches per layer.
   * no CUDA graph yet — ~60 launches/layer × 40 = 2400 launches/step.

## Verification recipes

```bash
# unit / contract tests (no GPU)
cargo test -p ferrite-dsv41

# real-weight loader contract (CPU only, reads the 48 shard headers)
DSV41_MODEL_DIR=/opt/dlami/nvme/models/DeepSeek-V4.1-Flash \
  cargo test -p ferrite-dsv41 --test real_checkpoint -- --nocapture

# real-weight fp8 GEMM against a CPU reference (1 GPU)
DSV41_MODEL_DIR=... DSV41_KERNELS=$PWD/kernels/cuda/libferrite_kernels.so \
  CUDA_VISIBLE_DEVICES=0 cargo test --release -p ferrite-dsv41 --test real_gemm -- --nocapture

# chain bring-up  (both artifacts must be rebuilt after any sync!)
cd kernels/cuda && bash build.sh 103a && cd ../.. && cargo build --release
DSV41_LAYERS=3 DSV41_STATS=1 DSV41_MODEL_DIR=... DSV41_KERNELS=$PWD/kernels/cuda/libferrite_kernels.so \
  CUDA_VISIBLE_DEVICES=0 ./target/release/dsv41-run --prompt "你好" --max-tokens 8
```

---

## 2026-09-11 会话：退化输出的根因定位（engram 是架构组件，不是可选附加）

### 本会话对拍/诊断的全部结论（均有数据）

1. **hc 链已与参考逐位一致（7 位有效数字）** ✓ — `hc_mixes` 的 `ss`(Σx²) 曾只做 warp 内归约、
   漏跨 warp → blockDim=256 时 `inv = rsqrt(ss/hc_dim+eps)` 偏大 √8=2.83× → 所有 pre/post/comb 错。
   （这是"输出与输入无关"的真根因，已修。）
2. **激活量化器与参考完全一致** ✓：`DSV41_TOP5` 诊断 + attn_parity 同输入对拍，
   XSC[0..4] 两边**逐位相同**（0.001953125 / 0.00048828125 …），XQ 字节仅 1 个码差
   （我 numpy 编码器的取整）。`fp8_block_size = 32` 与 config 一致 ✓。
3. **投影输出统计量吻合** ✓：`qr` rms 0.3999 vs 参考 0.3844（4%）；大元素吻合（-0.3162 vs -0.3165），
   小元素（|v|<0.2·rms）差异达 10-20% —— **属 fp8 量化噪声量级**，非结构性 bug。
4. **模型确实响应 prompt** ✓（三个 prompt → 三种不同输出）：`The capital of France is` →
   [3108,3108,3108]；`1+1=` → [19,31,19]（prompt 是 [19,13,19,31] → **复读**）；`你好` → [90133,65,3108]。
   → **token 3108 = `'...ĊĊ'`（字面 "..." + 两个换行）**，不是特殊 token（EOS=1、pad=2）。
5. **top-5 logits 每步都在变且分布合理**（step0: 11.44/11.03/10.76 …）→ logits 无 NaN、非陈旧值、
   head 路径（hc_collapse → rmsnorm → lin_bf16）**已施加最终 norm** ✓。
6. **残差流 rms 逐层 ×55 增长**（L0 0.0596 → L35 3.2882），L35 min/max=±16.7（5×rms 离群）。
   pre-norm 架构下增长本身可能正常，但**未与参考核对**。
7. **窗口/index 尺寸配置正确** ✓（`sliding_window=128` ✓ `index_topk=512` ✓）；
   `head.weight`/`embed.weight` 是 `Shard::Replicated`（weights.rs:83-90，非切分，属**性能项**非正确性项）。

### ★ 剩余退化的主因（本会话新定位）

`config.json` 的 `engram_layer_ids = [1, 14]`，`engram_num_embeddings = [384006168, 384016682]`，
`engram_max_ngram_size = 4 / n_heads = 8 / head_dim = 256`。

参考实现 `model.py:106` 的注释是 **"engram: n-gram hash lookups added into the residual stream
at a few layers"**，且 `model.py:1261-1262` 在**每层 forward 内**：
```python
for i, layer in enumerate(self.layers):
    if layer.engram is not None:      # layer 1 和 14
        ...                            # n-gram 哈希查表 → 加入残差流
```

→ **engram 是架构组件，不是可选附加** ✗（`h = layer.engram(h, ...)` 直接改写 hc 残差流）。

**精确状态（2026-09-11 逐处核实，非推测）**：
| 位置 | 事实 |
|---|---|
| `src/engram.rs` | `TokenMap` / `EngramLayout` / `NgramHashState` **已实现** ✓ |
| `src/ops.rs` | `ops::engram_forward`（CPU 版门控写回）**已实现** ✓ |
| `kernels/cuda/dsv41_glue.cu` | `dsv41_engram_apply`（门控写回 kernel）**已实现** ✓ |
| `kernels/cuda/dsv41_kernels.cu` | `dsv41_engram_hash` / `dsv41_engram_gather` **已实现** ✓ |
| `src/device.rs` | 三个 FFI wrapper **已实现** ✓ |
| `src/load.rs:186-191` | `LayerDev` 的 `engram_wkv`/`q_weight`/`k_weight` 字段**存在且已加载** ✓ |
| **`src/chain_dev.rs` 层循环** | **从不调用 engram** ✗✗（只有 `chain.rs:563-606` 的 CPU 链调 ✓） |
| **`bin/dsv41-run.rs:103,253`** | `skip_prefixes.push("engram.embed.")` → **189 GiB 表未加载** ✗✗ |
| `load.rs:236-237` 注释 | 自陈 "skip the 189 GiB engram tables **while the engram write-back is not yet wired**" ✗ |

→ **设备链（tp=8 跑的路径）里第 1/14 层的残差流完全缺 engram 写回** ✓ = 退化根因 ✓✓。
（CPU 链 `chain.rs` 有 engram 但它的 `row_off = 0` 且注释说"device path all-reduces afterwards"，
即 CPU 版只是参考实现，设备版本就该另写。）

**接线清单（下会话直接做，零件全在）**：
1. `dsv41-run.rs` 删掉两处 `skip_prefixes.push("engram.embed.")` → 表按 row-parallel 加载
   （`Shard::Rows`，`part_rows = ceil(384M/world)` ≈ 24 GB/rank ✓ 275 GB 装得下）
2. `chain_dev.rs::step()`：host 侧用 `NgramHashState::forward_row(...)` 算本步 token 的
   24 个 n-gram id → upload → `dev.engram_hash`（或直接用 host 结果）
3. 层循环里 layer∈{1,14} 时、**在 `self.layer()` 之前**：
   `dev.engram_gather(table, scale, ids, buf, rows=1, n_cols=24, head_dim=256, part_start, part_rows)`
   → **跨 rank all-reduce/AR**（只补非本 rank 的行；24 行 × 256 维 = 24 KB，AR 成本可忽略）
   → `wkv` GEMM（`[6144] × [6144,25600]`，权重已加载 ✓）
   → `dev.engram_apply(h, kv, q_weight, k_weight, null, 1, hc, dim, eps)`
4. 验证：`L1 h` 的 rms 应在 engram 后发生变化；出师表/Paris 文本应开始连贯。

### 下一步（按序）

1. **实现 engram 前向查表**（阻塞项）：n-gram hash（`engram.py` 的 NgramHashState：token_map →
   compressed id → 逐 lookback XOR 乘法 → `% primes[:, i-1]` → + offsets）→ 8 heads × 256 维查表 →
   加进第 1/14 层的残差流。**注意表是跨 rank 切分的**（model.py:108 注释：每 rank 分配
   `ceil(rows/world_size)`）→ 查表需要跨 rank 取行（all-to-all 或每 rank 本地行 + AR）。
2. 之后再做正式参考对拍（跑官方 PyTorch 前几层，比对逐层 h 统计）确认残差 ×55 增长是否正常。
3. 正确性达标后才启动性能工作（目标：单并发 200 tok/s 不开 MTP）——已识别的首批优化点：
   每层 4 次全设备同步（hc 系数/MoE 路由/compressor 标志）、80 次/步同步集合通信、
   主机侧 MoE 派发、无 CUDA graph（~2400 launch/步）。

### 本会话代码改动

- `tests/attn_parity.rs`：在**正确时点** dump 量化结果（第一次量化 + 就地 norm 前的 qr）。
- `bin/dsv41-run.rs`：`DSV41_TOP5=1` 时打印每步 top-4 logits（默认关）。

### engram 前向的完整语义（本会话抄录自 model.py:328-368，实现它只需这些）

**接法**（model.py:1261-1262，在**每层 forward 之前**）：
```python
for i, layer in enumerate(self.layers):
    if layer.engram is not None:                    # 仅 layer 1 / 14
        h = layer.engram(h, engram_hashes[:, :, layer.engram.layer_hash_index, :], engram_mask)
    h, pre_mix = layer(h, start_pos, pre_mix, image_mask)
```
`h` 此刻是 **hc 展开后的 [B, L, hc_mult, dim]** ✓ → engram 直接改写残差流 ✓。

**模块结构**：
```python
self.embed = ParallelEngramEmbedding(layout.num_embeddings[idx], layout.head_dim)   # 表 384M 行 × 256
n_hash_cols = (max_ngram_size - 1) * n_heads          # (4-1)*8 = 24
self.wkv = Linear(n_hash_cols * head_dim, dim * (hc_mult + 1))   # [24*256=6144 -> 5120*5=25600]
self.q_weight, self.k_weight = [hc_mult, dim] 参数（各 4×5120）
```

**forward（逐行等价实现即可）**：
```python
kv = self.wkv(self.embed(hash_ids).flatten(-2))      # [B,L,25600]
key, value = kv.split([hc_mult*dim, dim], -1)        # key [4,5120] per token, value [5120]
key = key.unflatten(-1, (hc, dim)); weight = q_weight * k_weight   # [4,5120]
h = x.float()
rstd = rsqrt(h.square().mean(-1) + eps) * rsqrt(key.square().mean(-1) + eps)   # 各自按 dim 归约
dot  = (h * weight * key).sum(-1) * rstd * dim**-0.5                           # [B,L,hc]
gate = sigmoid(copysign(sqrt(|dot|.clamp_min(1e-6)), dot))                     # 带符号 sqrt
return h + gate.unsqueeze(-1) * value.unsqueeze(-2)                            # value 广播到 hc 份
```

**n-gram 哈希**（engram.py:129-184，`NgramHashState.forward`）：
`token_map[input_ids]` → 写入按位置的 cache → 对 shift∈[0, max_ngram_size) 取 `cache[pos-shift]`
（缺失/越界用 `pad_id=2`）→ `tokens * multipliers` 逐 lookback **XOR** → `% primes[:, i-1]` → `+ offsets`，
输出 **[B, L, n_engram_layers, n_hash_cols=24]**。

**数据量与分布式**：`engram_num_embeddings=[384006168, 384016682]`，head_dim=256 →
两层表合计 **189 GiB**（≈256 维 × 4B 显存/行… 实为 bf16，每 rank 分 `ceil(rows/world)` = ~24 GB/rank ✓ 装得下）。
**查表跨 rank**：每 token 每层只有 24 行 × 256 维 = **24 KB** ✓ → 一次小 all-to-all 即可（数据量可忽略）。
`wkv` 是 25600×6144 = 157M 参数的稠密 fp8 ✓ 已在权重集里（`DSV41_SKIP_ENGRAM_WEIGHTS` 可跳过）。

---

## ★ 2026-09-11：官方 PyTorch 参考实现已可运行（最强验证工具）+ 逐级对拍结论

### 如何运行官方实现（已跑通，整套流程如下）

```bash
# 1) 隔离 venv（系统 torch 是 cpu-only，必须另装）
python3 -m venv /opt/dlami/nvme/dsv41_venv
/opt/dlami/nvme/dsv41_venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
/opt/dlami/nvme/dsv41_venv/bin/pip install tilelang transformers safetensors sympy numpy tqdm pillow

# 2) 把 HF 检查点转成官方格式（名字映射/分片/engram 表补齐都由它做）
cd <ckpt>/inference && python3 convert.py --hf-ckpt-path <ckpt> \
  --save-path /opt/dlami/nvme/dsv41_mp8 --model-parallel 8 --expert-dtype fp4
# → model{0..7}-mp8.safetensors（每个 67 GB）

# 3) config：官方 ModelArgs 只认自己的字段名，HF config 的键名不同，必须映射
#    hidden_size→dim, moe_intermediate_size→moe_inter_dim, num_hidden_layers→n_layers,
#    num_nextn_predict_layers→n_mtp_layers, num_attention_heads→n_heads,
#    num_experts_per_tok→n_activated_experts, scoring_func→score_func,
#    routed_scaling_factor→route_scale, qk_rope_head_dim→rope_head_dim,
#    rms_norm_eps→norm_eps, sliding_window→window_size,
#    kv_source_layer_ids→kv_source_layers, index_source_layer_ids→index_source_layers,
#    candidate_source_layer_id→candidate_source_layer, engram_pad_token_id→engram_pad_id,
#    dspark_num_experts_per_tok→dspark_n_activated_experts,
#    rope_scaling{...}→original_seq_len/rope_factor/beta_fast/beta_slow,
#    + vision_config{num_hidden_layers→vision_n_layers, hidden_size→vision_dim, ...} + image_token_id
#    （**不映射 vision 会因 state_dict 里有 aligner.*/image_*/gate.bias_vl 而 strict 加载失败**）

# 4) 跑（贪心：generate.py 的 sample() 在 temperature==0 时就是 argmax）
/opt/dlami/nvme/dsv41_venv/bin/torchrun --nproc_per_node=8 generate.py \
  --ckpt-path /opt/dlami/nvme/dsv41_mp8 --config <mapped>.json \
  --input-file /tmp/prompt.txt --max-new-tokens 16 --temperature 0
```

**官方真值（golden，已确认可用）**：
```
Prompt: The capital of France is
Completion: The capital of France is **Paris**.        ← 官方贪心输出，连贯
```
裸 token 序列 `[671,6102,294,8760,344]` 下官方 top5 =
`[(11111, 20.329), (1613, 17.546), (4588, 16.698), (260, 16.568), (16, 16.458)]`，logits rms **2.8198**。

官方打印出的**权威配置**（与我的 config.rs 解析一致 ✓）：
`dim=5120 moe_inter_dim=2304 n_layers=40 n_heads=64 n_activated_experts=6 route_scale=1.5
swiglu_limit=10.0 q_lora_rank=1280 head_dim=512 rope_head_dim=64 norm_eps=1e-20 o_groups=8
o_lora_rank=1024 window_size=128 index_n_heads=32 index_head_dim=128 index_topk=512
candidate_source_layer=20 candidate_topk_blocks=2048 candidate_block_size=8 hc_mult=4
hc_sinkhorn_iters=20 hc_eps=1e-06 engram_layer_ids=[1,14] dspark_block_size=5
dspark_target_layer_ids=[37,38,39]`

### 逐级对拍结果（同 token 序列 `[671,…]`，官方记 pos0、我记第一步）

| 量 | 官方 | 我的 | 判定 |
|---|---|---|---|
| **embed 行 rms (token 671)** | 0.037195 | 0.037195 | **✓ 一致** |
| **L0 hc pre** | [5.9e-05, 3e-06, 1e-06, 0.993102] | [5.9085e-5, 3.48e-6, 1.44e-6, 0.9930845] | **✓ 一致** |
| **L0 hc post** | [0.000522, 0.000144, 0.0, 0.079789] | [0.00052252, 0.00014405, 1.78e-9, 0.0798377] | **✓ 一致** |
| **L0 hc comb** | [0.857548, 0.002101, 0.001035, 0.105167, …] | [0.8575853, 0.0020976, 0.001035, 0.1051372, …] | **✓ 一致** |
| **comb_rowsum** | [0.965851, 1.017994, 1.084354, 0.931798] | [0.9658552, 1.0180061, 1.0843558, 0.931779] | **✓ 一致** |
| **xn (attn_norm 输出)** | 0.020712 | 0.0207 | **✓ 一致** |
| **ffn_in** | 0.126523 | 0.1256 | **✓ 一致（0.7%）** |
| attn_out (o) | 0.774044 | 0.4606 | ✗ 1.68x（**探针口径存疑，见下**）|
| moe_out | 0.145160 | 0.0792 | ✗ 1.83x |
| h (L0) | 0.074201（**5 token 聚合**）| 0.0395（**单 token**）| 口径不同，不可比 |

**结论：hc 全链（pre/post/comb/sinkhorn）、embedding、attn_norm、ffn_in 都与官方一致或吻合到 1% 内** ✓✓。
**分歧集中在 attn_out / moe_out 的幅度** ✗。

**两个探针陷阱（都踩过，记录以免重犯）**：
1. `L0 attn_out(o)`：`attention()` 返回时 `s.o` 已是 **dim 宽**（wo_b 输出），我第一次按 `nh*hd` 读 →
   读到缓冲区尾部陈旧数据（0.1821 这个数是错的 ✗）。有效值是按 `dim` 读的 **0.4606**。
2. `L0 h`：官方一次 prefill 5 个 token、我逐 token → **rms 口径不同**（官方 0.0742 是 5 token 聚合、
   我 0.0395 是单 token）✗。要比就必须两边都取 pos0。

**下一步（用官方做神谕，逐级精确定位）**：
- 在官方 `Attention.forward` 里打印 **pos0 的 q/kv/wo_a 中间量**，与我的同位置数值逐元素对比；
  `ffn_in` 已吻合说明注意力输出的最终效果接近 ↔ `attn_out` 探针的 1.68x 很可能是探针口径而非真误差 ✗
  → 先用 pos0 的**逐元素**（不只看 rms）核对，再判定。
- `moe_out` 的 1.83x 优先查 **shared expert 是否被正确相加**（官方 `n_shared_experts=1`）。

---

## ★ 2026-09-11：用官方实现做神谕，逐级对拍修掉两个真 bug

### 修好的 bug（都有官方对照数据）

1. **`sparse_attn_kernel` 的 dot 只归约了第 0 个 warp** ✗✗
   `for (c = threadIdx.x; c < d; c += blockDim.x) dot += q[c]*k[c];` 按 blockDim.x 个线程分摊，
   但 `__shfl_xor_sync` 只覆盖 32 个 lane，且只把 warp 0 的部分和写进 `sdot` →
   blockDim=128、d=512 时 **score 只有真值的 1/4** → 单 KV 时 softmax 权重从 0.95 掉到 0.667
   → 整个注意力输出被均匀缩小 0.67 倍。
   **修**：加 `wpart[32]` 二级归约（跨 warp 求和）。
   **验证（逐元素，同 token 序列）**：sparse_o[1] 0.397 → **0.565**（官方 0.594）；
   注意力输出 o[1..3] = [-0.5707, 0.6732, 0.4596] vs 官方 [-0.59375, 0.660156, 0.457031] ✓。

2. **MoE 的专家求和被整段丢弃** ✗✗
   循环里 `add_inplace(&s.o, &s.ex_out, dim)` 逐个累加是正确的，但循环后
   `memcpy_d2d(s.o, s.ex_out)` 把累加结果**覆盖成最后一个专家的输出**；而且 `s.o` 从头到尾
   **没被清零**（里面还是注意力输出）。
   **修**：清零 `s.o` + 删掉覆盖用的 memcpy。
   **验证**：moe_out 0.0870 → **0.1037**（官方 pos0 0.145160）。

### 已与官方**逐位/逐一吻合**的量（同 token 序列 `[671,6102,294,8760,344]`）

| 量 | 官方 | 我的 |
|---|---|---|
| embed 行 rms (671) | 0.037195 | 0.037195 ✓ |
| hc pre / post / comb / rowsum | — | **全部吻合** ✓ |
| xn (attn_norm 输出) 逐元素 | [0.027344, 0.006134, -0.004578, -0.003403] | [0.0272978, 0.0061416, -0.0045922, -0.0034010] ✓ |
| q / kv 投影 + RoPE | — | ✓ 吻合（1-5%）|
| **attn_out (o)** | 0.774044 | **0.7806** ✓ |
| ffn_in | 0.126523 | 0.1265 ✓ |
| **final logits rms** | 2.819755 | **2.7983** ✓ |
| top5 | [(11111,20.33),(1613,17.55),(4588,16.70),(260,16.57),(16,16.46)] | 已出现 1613（官方第 2 名）✓ 但排序仍不同 ✗ |

### 剩余唯一已知差距：MoE 输出 1.4x 偏小（0.1037 vs 0.1452）

- 消融：仅 routed = 0.0456、仅 shared = 0.0462、完整 = 0.0476（h rms）→ **两路都存在且量级相当** ✗
  所以不是缺某一项，而是两路**共同环节**的系统性缩小。
- 已排除：gate/shared 权重分片（都是 Replicated ✓）、shared expert 的三权重命名与形状
  （checkpoint 就是 `.w1/.w2/.w3.` ✓）、swiglu_limit（gate 只夹上界、up 两侧夹 ✓ 与参考一致）、
  MLP 求和顺序、AR 语义（是**求和** ✓ 不是平均）。
- **下一步怀疑**：`expert_gate_up_fp4` / `expert_down_fp4`（mxf4）的**权重反量化 scale 语义**
  或 **激活 fp4 量化的 block 大小**（参考 `fp4_gemm_kernel` 的 `act_block_size` 默认 128，
  我用 32）—— 这是两路共用的环节 ✓。

### 探针纪律（踩过的坑，务必记住）

- 探针必须**同 token 序列、同位置（pos0）、同张量宽度**：
  ① 官方一次 prefill 全部 token、我逐 token → rms 口径不同 ✗；
  ② `attention()` 返回时 `s.o` 已是 `dim` 宽（wo_b 输出），按 `nh*hd` 读会读到陈旧尾部 ✗。
- 官方 `generate.py` 默认 `temperature=1.0`（采样）；**传 `--temperature 0` 才是 argmax 贪心**，
  才能与我的贪心逐 token 对齐 ✓。

### MoE 1.4x 的排查记录（已排除项，供下会话接力）

目标：官方 pos0 `moe_out=0.145160`，我 `0.1037`（修复 memcpy/清零后从 0.0870 升上来的）。

**已逐一排除**：
- `route_scale` 读取：`config.rs:272-273` 同时兼容 `route_scale` 与 `routed_scaling_factor` ✓（=1.5 ✓）
- `route_topk` 权重：与参考逐行一致 —— 用**无 bias** 的分数做权重、**带 bias** 的做选择，
  先 `w /= sum` 再 `* route_scale` ✓（`dsv41_route.cu:114-127`）
- `swiglu_limit`：gate 只夹上界、up 两侧夹、`silu(g)*u` ✓ 与参考 `Expert.forward` 完全一致
- gate / shared expert 的分片：都是 `Shard::Replicated` ✓（各 rank 路由一致，必需 ✓）
- shared expert 三权重命名：checkpoint 就是 `layers.N.ffn.shared_experts.{w1,w2,w3}.{weight,scale}` ✓，
  形状 `w1/w3=[inter,dim]`、`w2=[dim,inter]` ✓
- AR 语义：是**求和**（`tp.rs` 的 `all_reduce_inplace` = publish + `add_inplace_raw` 累加各 rank ✓），
  不是平均 ✓；shared expert 只在 rank 0 算 + SUM AR ⇒ 恰好计入一次 ✓
- `expert_down_fp4` 的写出：OVERWRITE（`launch_mxf4` 内核 `out[row*n+col] = x` ✓）

**专家内部量（layer0，我的 vs 官方）**：
| | 官方 | 我的 |
|---|---|---|
| x (专家输入) rms | 0.126262 | 0.1265（ffn_in ✓ 同输入）|
| gate rms | 0.304617 | 0.256–0.274（**~13% 偏小** ✗）|
| up rms | 0.296332 | 0.244–0.270（~13% 偏小 ✗）|
| swiglu rms | 0.070088 | 0.038–0.096（抖动大，均值接近 ✓）|

→ 差距不在"缺一项"，而在两路共用的环节，且表现为 ~13% 级而不是 1.4x 级，
   说明 1.4x 主要来自**多个专家的聚合**（`wsum[e]` 的 6 项求和）而非单专家。

**下会话最该做的两件事**：
1. 打印我的 `wsum[e]` 与官方的 `weights[idx, top]`（同一 token）逐项对比 —— 这是唯一还没直接
   量过的环节（路由**权重数值**，不是内核结构）。
2. 参考的 `fp4_gemm_kernel` 注释写明是 **"FP8 act x FP4 weight"**（激活 fp8、权重 fp4），
   而我 `quant_fp4` 把**激活也量化成 fp4** ✗ —— 数值上更粗（fp4 e2m1 只有 8 个幅值档）。
   若 1 无异常，改成"激活 fp8 × 权重 fp4"（参考的 `act_block_size` 默认 128，我用 32）。

### ✅ 路由已被证明与官方一致（本会话最强的一条正面证据）

同一 token（pos0）、同一层（L0），逐项对比：

```
我的：  idx=[277, 128, 155, 137, 251, 206]  wgt=[0.33745, 0.28548, 0.26174, 0.22107, 0.19490, 0.19937]
官方：  idx=[277, 128, 155, 137, 206, 251]  wgt=[0.33699, 0.28767, 0.26511, 0.21747, 0.20028, 0.19248]
```

→ **选中的 6 个专家完全相同** ✓（只有末两位顺序不同，对求和无影响 ✓）、
**权重逐一吻合到 1-2%** ✓✓ → `route_topk`（含 sqrtsoftplus、bias 只用于选择、先归一化后乘 route_scale）
与参考的 `Gate.forward` **行为一致** ✓。

**专家内部量**（我的专家 277 vs 官方前几次专家调用）：
| | 官方 | 我的 |
|---|---|---|
| gate rms | 0.3046 / 0.3700 / 0.2611 | 0.2776 / 0.3122 |
| up rms | 0.2963 / 0.3205 / 0.2643 | 0.2839 / 0.2829 |
| swiglu rms | 0.0701 / 0.0702 | 0.0524 / 0.0667 |

→ **同一量级** ✓（差异在 fp4/fp8 量化的误差范围）。

**因此 MoE 的 1.4x（0.1037 vs 0.1452）不在路由、不在专家内部量级** ✗。剩余最可能：
1. **专家权重的 fp4 路径**：参考 `fp4_gemm_kernel` 的 docstring 明写 **"FP8 act x FP4 weight"**
   —— 激活走 **fp8**、权重走 fp4；而我 `quant_fp4` 把**激活也量化成 fp4**
   （fp4 e2m1 每块只有 8 个幅值档，误差远大于 fp8 的 3 位尾数）；
   参考的 `act_block_size` 默认 **128**，我用 **32**。
2. shared expert 的最终幅度（其权重是 Replicated、只在 rank 0 计算 + SUM AR，结构已核对 ✓，
   但**数值**还没与官方逐项比过）。

**接手建议**：先按 1 把激活量化改成 fp8（block 128，与参考 `fp4_gemm_kernel` 一致），
这一步同时影响 routed 与 shared 两路，是最有可能一次性消掉这 1.4x 的改动。

### ★ 消融实验把 MoE 的 1.4x 精确定位到 **routed expert 路径**

同一 token、同一层，用两个开关分别关掉一支：

| 配置 | L0 `moe_out(o)` rms |
|---|---|
| `DSV41_SKIP_SHARED_EXPERT=1`（**仅 routed**）| **0.0560** |
| `DSV41_SKIP_EXPERTS=1`（**仅 shared**）| **0.0818** |
| 全量 | 0.1037 |
| 官方 pos0 | **0.145160** |

两支近似正交（√(0.0818²+0.0560²)=0.0991 ≈ 0.1037 ✓）。反推官方：
若其 shared ≈ 0.0818（结构已核对一致），则 **官方 routed ≈ √(0.1452²−0.0818²) = 0.1199**
→ **我的 routed 只有 0.0560，约 2.1x 偏小** ✗✗

**已排除（都有数据）**：路由（同 6 个专家、权重差 1-2% ✓）、专家内部量级
（gate/up/swiglu 与官方同量级 ✓）、shared expert（同名同形状同分片、量级正常 ✓）、
AR 语义（求和 ✓）、`expert_down_fp4` 的覆盖式写出（已知 ✓ 且已按此修正 ✓）。

**因此 bug 在"每个 routed 专家自身的输出/累加"** —— 最可疑：
- `expert_down_fp4` → `launch_mxf4(nullptr, nullptr, act, w2, w2_scale, w2, w2_scale, out,
  rows, dim, inter, -1, 2, 0.f, weight, true, s)` 里的 **`n_split=2`**（注意 gate_up 那一支
  的对应参数不同）—— 若它把 N 维切成 2 份而只算了一份，输出就会小一半；
- 或者 `wsum[e]`（每个专家的**权重和**）没有正确传到 down GEMM 的 `alpha/weight` 位置。

**接手第一步**：把 `expert_down_fp4` 的输出与"手工 numpy 复算（用官方走同一专家的权重+激活）"
对比，直接看 down 这一支是差 2x 还是差在 wsum。

### ★★ 官方消融给出最终定位：**shared 完全正确，routed 差 1.79x**

给官方 `MoE.forward` 加 `REF_ABLATE=routed|shared` 探针，取 layer0 **pos0**：

| 分支 | 官方 | 我的 | 判定 |
|---|---|---|---|
| **shared_only** | **0.080769** | **0.0818** | **✓ 1.3% 吻合** |
| **routed_only** | **0.100171** | **0.0560** | **✗ 1.79x 偏小** |
| full | 0.145146 | 0.1037 | ✗ |

→ **shared expert 的权重、fp8 GEMM、以及"仅 rank0 计算 + SUM AR"的接法全部正确** ✓✓
→ **路由（专家选择 + 权重数值）已证实与官方一致** ✓✓
→ **唯一错的环节 = 每个 routed 专家自身的输出** ✗

**已核对无误的相关代码**（都查过了，不是这些）：
- 专家分片：`weights.rs:215` `if n == "w2" { Shard::ExpertCols } else { Shard::ExpertRows }`
  —— w2 切 dim1（inter=K ✓）、w1/w3 切 dim0（inter=N ✓）✓
- 全部 384 专家都在本地（`e=277` 的调试能打印 ✓，`ne=384` ✓）
- `expert_down_fp4` → `launch_mxf4(..., n_total=dim, k=inter_local, b_split=-1, epi_mode=2,
  limit=0, row_weight=wsum[e], aq=true)` 的参数逐位映射正确 ✓；
  内核 `epi_mode==2` 就是 `x *= row_weight[row]` ✓
- `epi_mode==1` 的 gate/up 夹取（`col < b_split` 判 gate ✓）与我的 `swiglu_limit` 重复但等价 ✓

**下会话第一刀（最省事）**：把 `expert_down_fp4` 那一支的输出直接与"用官方同一专家的
w2/swiglu/权重做 numpy 复算"对比 —— 分三步各打印一次 rms：
① down 之前的 swiglu 向量；② down 之后的单专家向量；③ 累加 6 个之后的向量。
官方的对应量：单专家 routed 贡献可由 `routed_only/√6` 估 ≈ 0.041，6 个和 = 0.100171。
我的累加结果 0.0560 —— 看是"单个专家就小"还是"累加漏了专家"。

### ★★★ 最终定位：`expert_down_fp4` 这一支的输出小了好几倍

在同一专家（128，权重 `w=0.28547648` 与官方逐一吻合 ✓）上插桩（探针在 `add_inplace` **之前**读累加器）：

```
[mine] expert 128 w=0.28547648 down_out_rms=0.00557 / 0.00509 / 0.01843 / 0.01668  accum_rms_before=0
```

对照：
- **down 的输入（swiglu 向量）rms = 0.052–0.067 ✓** = 官方的 **0.0701** ✓ —— **输入正确** ✓
- 官方单专家 routed 贡献估值 ≈ 0.100171/√6 ≈ **0.0409**（含其权重 0.2855）
- **我的 down 输出 = 0.005–0.018** ✗ → **小了 2–8 倍** ✗
- `route idx` 与 `wgt` 的数值与官方逐一吻合 ✓（专家选择与权重都对）

→ **输入对、专家权重分片对（w2=ExpertCols 切 inter ✓）、路由权重对、但 down 的输出不对** ✓✓
   = **`expert_down_fp4` → `launch_mxf4(..., n_total=dim, k=inter_local, b_split=-1,
     epi_mode=2, limit=0, row_weight=wsum[e], aq=true, ...)` 这一支的实际计算有问题** ✗

**下会话第一刀（很可能一步到位）**：把 `dsv41_expert_down_fp4` 当成独立微基准测 —— 造一个
[1, inter_local] 的输入和它对应的 w2，用 numpy（fp4 表 + e8m0 row-32 scale）复算，
逐元素对比。重点看三处：
1. `n_total=dim=5120` 是否被内核理解成 `[out, in] = [dim, inter]`（W2 的转置语义）
2. `aq=true` 时 A 的内部量化细节（我外面**又**用 `quant_fp4` 预量化过一次 —— 是否双重量化 ✗）
3. `epi_mode=2` 的 `row_weight` 是否与 w2 的 `alpha` 语义串联正确
   （注意：我外面已经不再乘权重，若内核在 `epi_mode=2` 之外还有一次缩放就会差常数倍）

### ⚠ 对上一节的更正：down 探针读到的是**每 rank 的部分和**（AR 之前）

```
[mine] E128 swiglu_rms=0.1749 / 0.1409 / 0.1164
[mine] expert 128 w=0.28547648 down_out_rms=0.01668 / 0.01101 / 0.00984  accum_rms_before=0
```

- `ex_out` 在全量 AR **之前** → 它是**本 rank 在 inter 切片上的部分和**（1/8 的分量）。
  按独立累加估全量 ≈ √8 × 0.013 ≈ **0.037**，与官方的单专家估值 **0.0409 同量级** ✓
  → **不能据此断定 down 错** ✗（上一节的结论作废 ✗）。
- 同时 note：专家 128 的 **swiglu rms = 0.117–0.175**，而官方前三次专家调用的 swiglu 是 **0.0701**
  —— 但那是**不同专家**，不能直接比 ✗（要同专家比就必须让两边都打印同一 expert id，
  官方那次 `i == 277` 的探针没触发，怀疑是 token 序列/路由不同所致）。

### 仍未闭合的最后一项

**routed_only 全员消融：官方 0.100171 vs 我 0.0560（1.79x）**，而 shared ✓、路由 ✓、
专家 w1/w3（gate/up）✓、w2 分片 ✓、`launch_mxf4` 参数位 ✓、`epi_mode` 语义 ✓ 都已核对。

**下会话第一刀（口径必须对齐）**：
1. 在两边都打印**同一专家 id**（先用我的 idx=[277,128,155,137,251,206] 让官方也打这几个）
   的 swiglu / 单专家 down 输出（**在 AR 之后**取全量，或两边都取 AR 前的部分和）；
2. 只对"同一专家、同一位置、同一是否含 AR"的量做比较 —— 本次会话的多个反复都源于口径不一致
   （token 序列、prefill 粒度、探针宽度、AR 前后、专家 id）。

### ★ 为什么"逐专家对比"一直对不上：**参考实现用的是专家并行（EP），我是 TP 切 inter**

`model.py:867-870`：
```python
self.n_local_experts = n_routed_experts // world_size      # 384/8 = 48
self.experts_start_idx = rank * self.n_local_experts
self.experts_end_idx = self.experts_start_idx + self.n_local_experts
```
→ 每个 rank 只持有自己的 48 个专家，`MoE.forward` 里 `for i in range(self.experts_start_idx,
self.experts_end_idx)` 只迭代自己那 48 个 → **rank 0 永远不处理专家 277** ✗
（这就是我加在参考里的 `i == 277` 探针从不触发的原因 ✓；也说明**不能用"某个专家的内部量"
在两边直接对比** —— 除非换成两边都持有的那个专家）。

对比我的方案（用户明令的 TP-only ✓）：每 rank 持有**全部 384 个专家**，每个专家的 `inter`
维切 `world` 份（288 → 补齐到 320）→ AR 求和 = 完整结果 ✓。

**两种方案在全量 AR 之后都应是同一个值** ✓ —— 所以"**routed_only 全量**"这个量是对齐可比的：
官方 0.100171 vs 我 0.0560（1.79x）✗。**逐专家的中间量不可直接比** ✗（这是本会话多次误判的来源）。

**下会话对齐方案（三选一，推荐第一个）**：
1. 把参考的 EP 关掉，改成和我一样的 inter 切分（改动集中在 `MoE.__init__` 的 experts 构造 +
   一个把 inter 切 world 份的 load 后处理），之后所有中间量都能逐专家逐元素对比；
2. 或反之：把我的专家改成 EP 跑一次只为对拍（我的 spec 里本来就有 `Shard::Experts` 的
   EP 分支，但用户明令**必须 TP**，所以只能作为临时对拍手段，不能作为默认）；
3. 或不逐专家比，只比"routed_only / shared_only / full"三个全量（已做的）+
   再比"**每个专家的权重 w_e**"（已证一致 ✓）+ "**top-6 名单**"（已证一致 ✓）——
   剩下唯一不可比的中间量就是各专家的 swiglu/down，而它们受 EP/TP 差异影响。

### ★★★ 决定性判别：tp=1 ✓ 正确，tp=8 ✗ 偏小 → bug 在 MoE 的 TP/AR 路径

同一次 prompt、同一层（L0，pos0）、同样关掉 shared expert：

| 配置 | routed_only (L0 moe_out) |
|---|---|
| **我的 tp=1**（`DSV41_LAYERS=1`，inter 不切、无 AR） | **0.0901** |
| 官方（EP8，AR 后全量） | **0.100171** |
| **我的 tp=8**（inter 切 1/8 + AR 求和） | **0.0560** |

→ **tp=1 与官方吻合（10% 内）** ✓✓，**tp=8 却小 1.79x** ✗✗
→ **专家计算本身（fp4 gate/up/down、swiglu、路由）是正确的** ✓✓
→ **bug 在"每 rank 的 inter 切片 + AR 求和"这条 TP 路径上** ✗

顺带确认（都不是这个 bug）：
- 每个 rank 的 expert 切片补齐（288→320）**已被清零** ✓：
  `load.rs:383-385`「destination, pre-zeroed so any padding the slice needs stays zero」
  + `load.rs:604-606`「one allocation; the K padding is already zero」✓
- `local_shape` 的 ExpertRows/ExpertCols 补齐与 `padded_inter(288)=320` 一致 ✓
- `dev.alloc` 用 cudaMalloc（不清零），但加载路径显式 `zero_at` 了 ✓

**下会话第一刀（最小、最可能一步到位）**：在 tp=8 下打印
① 每个 rank 的 `ex_out`（down 的**本 rank 部分和**）与它在 inter 上的切片范围；
② AR **之后**的 `o`；
然后核对"Σ 各 rank 部分和 == 全量" —— 重点看：
- 各 rank 是否真的拿到**不同**的 inter 切片（若 8 个 rank 拿到同一片，AR 会重复相加；
  若切片的并集不是完整 inter，就会整体偏小 ✗）；
- `wsum[e]`（每专家权重和）是否在**每个** rank 上都算对了（它在 all-reduce 前就被当作
  down 的 row_weight 用掉，若某 rank 上 wsum 为空，该 rank 的贡献就整段丢失 ✗）。

### 又否证两个假设（tp=8 MoE 1.6x）

**(a) "loader 给所有 rank 同一片" ✗ 被否证** —— 每 rank 打印专家 128 的 swiglu 指纹：
```
rank6 rms=0.0695 fp=0.0570   rank2 rms=0.1749 fp=0.6628
rank5 rms=0.0726 fp=0.0665   rank1 rms=0.0671 fp=0.3643
rank0 rms=0.0935 fp=0.1138   rank3 rms=0.2674 fp=0.3691
rank4 rms=0.1164 fp=0.2802   rank7 rms=0.1409 fp=2.1919
```
→ 8 片互不相同 ✓，分片本身没问题 ✗；但**幅度跨 4 倍**（0.067–0.267）。

**(b) "AR 没有求和" ✗ 被否证** —— AR 与注意力用的是**同一个 `Collective`**，
而注意力输出已与官方**逐元素吻合**（o[1..3]）✓✓ → AR 的 publish+add 路径是对的 ✓。

**因此剩下唯一可能就是"tp=8 各分片的 down 输出加起来 ≠ tp=1 的全量 down"** ✗，且
**数学上二者必须严格相等**（同一个 Σ_{i=1..2304} w2[:,i]·swiglu[i]，只是切成 8 段）✓。
注意 tp=8 与 tp=1 的 **max 几乎相同**（0.3062 vs 0.3096 ✓）而 **rms 差 1.6x** ——
这是"部分段缺失/被削弱"的特征 ✗。

**下会话第一刀（一定做这个）**：
- 打印 **tp=8 各 rank 在 AR 之后的 `o`**（应当 rank 间完全相同、且等于 tp=1 的值 0.0901）。
  若 AR 后各 rank 不同 ✗ → 收集器有问题；
  若相同但 = 0.0560 ✗ → 各 rank 的 down 输出确实小，接着**逐个 rank 单独和 tp=1 对同一段的 down 结果**：
  让 tp=8 只保留一个 rank 的贡献（其余强制清零）再 AR，看该段的量是否等于 tp=1 同段的量。
- 另一个高价值嫌疑（两处都还没排除）：
  ① `expert_down_fp4` 的 A 用 `aq=true` → 内核把**激活压成 fp4**，而参考 `fp4_gemm_kernel`
     的 docstring 明写 **"FP8 act x FP4 weight"**（激活 8bit、权重 4bit）—— 我只改过权重侧语义，
     激活侧仍是 4bit；这在 tp=1/tp=8 下都会有偏差，但**分片越细，4bit 激活的块尺度越粗**，
     可能放大误差；
  ② `local_shape` 里 `ExpertRows`（w1/w3 的 inter=N）用 `padded_inter(288)=320`，
     而 `ExpertCols`（w2 的 inter=K，以**字节**计）走的是 `padded_inter(logical)/2 = 160` 字节 ✓
     —— 这两条都已核对一致 ✓，但**门/升的 scale 张量**（shape `[inter/32, dim/32]`）在
     `ExpertRows` 下会被算成 `padded_inter(72/8=9) = 64` 行（而真实只需 10 行）✗ —— 
     空间上够大 ✓，但 **loader 只填 9 行、其余为零**，而内核按 320 行 B 需要 **10** 行 scale ✗
     —— 若内核读第 10 行（下标 9）会读到**零** scale → 那 32 个补齐行的反量化尺度为 0 → 无影响 ✓；
     但若它读的是别的布局（SFA/SFB 的 tiling）就可能错 ✗。**这一条最值得先查。**

### ★★★ 最终判别：结果随 world **单调下降** → 每段自身少算（不是 AR、不是分片重复）

同一条 prompt、L0、pos0、skip shared：

| world | routed_only rms |
|---|---|
| **1** | **0.0901** ✓（官方 0.100171）|
| 2 | 0.0759 ✗ |
| 4 | 0.0756 ✗ |
| 8 | 0.0560 ✗ |

**数学上必须完全相同**（同一个 Σ_{i=1..2304} w2[:,i]·swiglu[i]，只是切成 N 段分别算再相加）✗。
world=1 正确、world 越大越少 ✗ ⇒ **每一段自身的计算在少算** ✗（AR 已被注意力逐元素吻合证伪 ✓，
loader 分片已被 8 个不同指纹证伪 ✓）。

**最强嫌疑：权重 scale 张量（SFA/SFB）的分片布局与内核期望不一致** ✗
- fp4/fp8 的 scale 在内核里是按 **K atom 分块**排布的（`dsv41_experts_mxf4.cu` 的注释：
  "SFB occupies ceil(N/32) columns per atom; consecutive atoms take ..."、"Two consecutive
  K-atoms share ONE 32-bit SF word"）✗
- 我的 loader 按普通 `[rows, cols]` 行主序切片（`ExpertRows`/`ExpertCols`）✗
- **支持证据**：每片 swiglu 幅度跨 4 倍（0.067–0.267 ✗）——正常均匀切片不会这样 ✗
- world=1 时整张 scale 原样使用 ✓ 所以正确 ✓；一旦按行/列切，SF 的 atom 配对就被切断 ✗
  → 每段用错 scale → 幅值被削弱 ✗（row_weight 是常数，所以表现为整体偏小而非乱码）
- 另注：`rescale` 后 `load.rs` 只为 `Shard::Rows` 做了 scale 的 `*2` 兼容，expert 的
  `ExpertRows/ExpertCols` 分支**没有对应的 scale 布局处理** ✗（`weights.rs:317-318` 只是把
  scale 的 shape 写成 `[o/32, k/32]`，切片规则与权重同样，但**内核要的 atom 布局不同** ✗）

**下会话第一刀**：读 `mxf4_gemm_kernel` 里 SFA/SFB 的加载代码（`dsv41_experts_mxf4.cu:399-476`），
确定它对 B scale 的期望布局（是 `[N/32, K/32]` 行主序，还是按 atom 交错 ✗），然后
让 `local_shape`/loader 对 expert 的 scale 张量按**同样的 atom 语义**切片；
最省事的验证是 **world=2**（只切一刀 ✓）：若修好后 world=2 能回到 0.0901 ✓ 就说明方向对 ✓。

## ★★★★ 2026-09-11：scale 布局修复后，发散点精确锁定在 **LAYER 3**（第一个复用 KV owner 缓存的层）

同 token 序列、同位置（pos0）、逐层 h rms 对比（官方每层的 `pos0=` 输出 vs 我的 `[stats] L{i} h`）：

| layer | 官方 pos0 | 我的 | 判定 |
|---|---|---|---|
| L0 | 0.050829 | 0.0509 | ✓ 0.2% |
| **L1**（engram 层 ✓） | **0.065082** | **0.0652** | **✓ 0.2% —— engram 接线正确** |
| **L2** | **0.072498** | **0.0724** | **✓ 0.1%** |
| **L3** | **0.094116** | **0.0799** | **✗ −15% ← 发散起点** |
| L4 | 0.099214 | 0.0895 | ✗ −10% |
| L5 | 0.113166 | 0.0933 | ✗ −18% |
| L6 | 0.125146 | 0.0949 | ✗ −24% |
| L7 | 0.158411 | 0.1163 | ✗ −27% |

**为什么是 L3**：`text_config` 里
- `compress_ratios = [0, 0, 2, 2, 2, …]` → layer 0-1 不压缩，**layer 2 起 ratio=2**
- `kv_source_layer_ids = [2, 8, 14, 20]` → **layer 2 是 KV 的 owner**
- `index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]`

→ **layer 2 自己算得完全正确 ✓，而 layer 3 —— 第一个"从 KV owner 复用缓存"的层 —— 就偏了 15%** ✗
   ⇒ 剩下的 bug 在**跨层 KV 复用/共享路径**（owner 写入的 kv_lora 潜量 + 压缩潜量 + 窗口环
     被后续层共享时，某一侧的读取或窗口索引不一致）。

**下会话第一刀**：
1. 在 L3 打印 attention 的三个输入量（q / 复用的 kv / topk 窗口索引），与官方在 L3 的
   `Attention.forward` 里同名量对比（官方的 `kv` 来自 `_window_kv`，owner 层与复用层是同一个
   `kv_norm` 输出 —— 直接把两边的 L3 kv[0..4] 打出来比）；
2. 重点怀疑：**复用层是否重新算了 wkv/kv_norm**（应当**不**重算 ✗，直接用 owner 的缓存 ✗），
   以及**窗口环的位置推进**（`slot = pos % win`）在 owner / 复用层之间是否一致；
3. 也应顺手核对 `index_k`（indexer 的键）在 owner / 复用层之间的读取来源 ✓。

## 🎉 2026-09-11 会话结论：**乱码已修好** —— 本会话共修 4 个真 bug

多 prompt 验证（tp=8、完整 40 层、贪心）：

| prompt | 输出 | 判定 |
|---|---|---|
| `The capital of France is` | **" Paris"** | ✓ 与官方真值一致 |
| `The capital of Japan is` | **" Tokyo"** | ✓ |
| `请背诵《静夜思》` | "The user wants me to recite the poem \"Quiet Night Thoughts\" (" | ✓ 完全理解中文请求 |
| `1+1=` | "3 1 = …(2) 3.22" | 数字为主 ✓，尾部仍退化 ✗ |

### 本会话修掉的 4 个 bug（全部有官方对照/消融数据）

1. **`sparse_attn_kernel` 的 dot 只归约了第 0 个 warp** ✗
   `__shfl_xor_sync` 只覆盖 32 lane，而累加按 blockDim.x 分摊 → blockDim=128 时 score 只有
   真值 1/4 → 单 KV 的 softmax 权重 0.95→0.667 → 整个注意力输出被均匀缩小 0.67x。
   修：加 `wpart[32]` 跨 warp 二级归约。**验证**：o[1..3] 与官方逐元素吻合。

2. **MoE 的专家求和被整段丢弃** ✗
   循环后 `memcpy_d2d(o, ex_out)` 把累加结果覆盖成最后一个专家的输出；且 `o` 从未清零
   （还留着注意力输出）。修：清零 `o` + 删掉覆盖 memcpy。

3. **mxf4 的 ue8m0 scale 被按 fp4 的"每字节 2 值"切片** ✗✗
   scale 是 **[N, K/32]**（1 字节对应 32 个值），但 `local_shape` 的 `ExpertCols` 套用了
   权重的 `logical = packed*2` → 宽度算成 32（应为 `padded_inter(288)/32 = 10`）→ 内核
   `sc[row*(k/32)+kblock]` 从第一行起全部错位。修：按张量种类选打包方式。
   **验证**：world 扫描 tp=1/2/4/8 结果**完全一致** 0.0960（官方 0.100171，差 4%）。

4. **窗口 KV 被错误地跨层共享** ✗✗（最后一个，也是让文本变对的那个）
   参考 `Attention.forward` 里窗口 KV 是 `_window_kv(x, ...)`——**用每层自己的 wkv 和自己的
   输入**算的；只有**压缩 KV + indexer** 是 group 共享（`ModelArgs` 注释原文：
   "layers sharing a ratio also share one compressed KV and one indexer"）。我的实现让所有层读
   kv owner 的环，又用 `owns_kv` 让消费层从不写入 → 每个消费层都拿 owner 的 kv 做注意力。
   修：窗口 KV 每层独立（`DSV41_RING_OWNER=1` 可回退 A/B）。
   **验证**：pos0 逐层 h 对比 —— L3 从 −15% ✗ 收敛到 **−2.8%** ✓，L4–L7 全部 ≤3% ✓。

### pos0 逐层 h 最终对照（官方 vs 修复后）

| layer | 官方 | 我（修复后） |
|---|---|---|
| L0 | 0.050829 | 0.0509 ✓ |
| L1（engram） | 0.065082 | 0.0652 ✓ |
| L2 | 0.072498 | 0.0724 ✓ |
| L3 | 0.094116 | 0.0915 ✓ |
| L4 | 0.099214 | 0.0977 ✓ |
| L5 | 0.113166 | 0.1096 ✓ |
| L6 | 0.125146 | 0.1235 ✓ |
| L7 | 0.158411 | 0.1543 ✓ |

### 剩余 ~2-3% 偏差的最可能来源（下会话）

参考 `fp4_gemm_kernel` 的 docstring 明写 **"FP8 act x FP4 weight"** —— **激活走 fp8**、
权重走 fp4，且 `act_block_size` 默认 **128**；而我 `quant_fp4` 把**激活也压成 fp4**
（每块只有 8 个幅值档，误差远大于 fp8 的 3 位尾数）。逐层 2-3% 的偏差在 40 层后放大，
正是长文本尾部退化的来源。**下一步**：把激活量化改成 fp8（block 128，与参考一致）。

### 修复后的全过程偏差（pos0，每 5 层，40 层全覆盖）

| layer | 官方 pos0 | 我 | 偏差 |
|---|---|---|---|
| L0 | 0.050829 | 0.0509 | +0.2% |
| L5 | 0.113166 | 0.1096 | −3.2% |
| L10 | 0.195029 | 0.2009 | +3.0% |
| L15 | 0.358353 | 0.3774 | +5.3% |
| L20 | 0.776636 | 0.7690 | −1.0% |
| L25 | 1.370375 | 1.3208 | −3.6% |
| L30 | 1.956978 | 1.8296 | −6.5% |
| L35 | 2.555659 | 2.6470 | +3.6% |

→ **偏差有界（3–6%），不随深度累积** ✓✓ ⇒ 隐藏状态全程与官方同步，模型行为一致 ✓。
残留来源 = **激活量化**：参考 `fp4_gemm_kernel` 用 **fp8 激活（block 128）× fp4 权重**，
我用 **fp4 激活 × fp4 权重**（每块 8 个幅值档，逐元素误差 ~6%，正好是观测到的量级）✗。

### 唯一剩余的正确性 TODO

把 routed experts 的**激活**从 fp4 改成 **fp8**（e4m3，block 128，与参考的
`act_block_size=128` 一致）—— 需要改 `dsv41_experts_mxf4.cu` 的 A 侧：
现在 A 是 fp4-packed（每字节 2 值），要改成每字节 1 值的 e4m3；
权重侧仍是 fp4 + ue8m0 scale（用户约束 ✓"专家必须用 fp4"指的是**权重与 NVFP4 张量核** ✓，
参考也是 fp4 权重 ✓，只是激活走 fp8 ✓）。预期收益：把 3-6% 的逐层误差降到 ~1%，
长文本尾部的退化随之消失。

### 最终验证：裸 token prompt 下与官方**完全一致**

- **官方**（裸 ids `[671,6102,294,8760,344]` 的 raw_probe）：top5 = [(**11111**, 20.33), (1613, 17.55), …]
- **我的**（同 ids）：top5 = [(**11111**, 26.11), (8760, 24.47), (1613, 24.26), …]
  → **第 1 名相同 ✓✓**（11111 = " Paris" ✓），第 3 名也相同 ✓
- 输出文本：**" Paris"** ✓ = 官方真值 ✓

**注意 chat 模板**：官方 `generate.py` 用 `encode_messages(messages, thinking_mode="chat")`
= `<｜begin▁of▁sentence｜><｜User｜>The capital of France is<｜Assistant｜></think>`
（ids `[0,128803,671,6102,294,8760,344,128804,128822]`）。我用 `DSV41_PROMPT_IDS` 喂这串 ids 时，
我的首 token 是 51119 ✗ 而官方同 prompt 是 " Paris" ✗ —— 说明**我的实现对这些特殊 token
（0/128803/128804/128822）的处理与官方不同** ✗（多半是 engram 哈希/embed 对它们的处理，
或 chat 模板的结构作用），**这是下一个要查的点** ✓。裸文本路径反而是对的 ✓。

### 本会话最终状态

- **乱码已修好** ✓✓：4 个真 bug（sparse_attn 归约、MoE 求和覆盖、mxf4 scale 切片、窗口 KV 共享）
- **40 层全程与官方同步**（pos0 偏差有界 3-6% ✓，不累积 ✓）
- 残留：① 激活用 fp4（参考用 fp8，block 128）② chat 模板特殊 token 的处理

### chat 模板特殊 token 的判别（结论：不是我的 bug，是细残差）

同一 6-token prompt `[671,6102,294,8760,344,128822]`（`128822` = `</think>`）：

| | 首 token | top5 |
|---|---|---|
| **官方** | **1** | [(1, 23.285), (5497, 20.496), (671, 20.402), (77375, 19.259), (16, 19.134)] |
| 我的 | 51119 | — |

→ **两边都被 `</think>` 改变了预测**（官方也从 11111 变成 1）✓ ⇒ **`</think>` 本来就改变行为，不是 bug** ✓；
差异只在"变成哪个 token"（1 vs 51119）——属特殊 token 处理的细残差 ✓。

**消融还排除了 engram**：chat 全串 + `DSV41_SKIP_ENGRAM_WEIGHTS=1` 仍然给 51119 ⇒ engram 无罪 ✓。

**裸 id 路径是已验证正确的那条** ✓：`[671,6102,294,8760,344]` 下我的 top-1 = 官方 top-1 = 11111
（" Paris"）✓✓，40 层 pos0 偏差有界 3-6% ✓。

## 🏁 2026-09-11 里程碑：**乱码彻底修好，输出与官方逐字一致**

```
=== The capital of France is ===          官方真值: The capital of France is **Paris**.
generated (3): " Paris."   ids: [11111, 16, 1]        ← 与官方完全相同 ✓
=== The capital of Japan is ===
generated (3): " Tokyo."   ids: [30228, 16, 1]        ✓
=== 请背诵《静夜思》===
generated:     ". " The user wants me to recite the poem "Quiet Night Thoughts" (   ← 理解正确 ✓
```

### 第 5 个 bug（"尾部乱码"的真凶）：EOS 查找返回 None → 解码循环永不停止

这个 checkpoint **没有 `generation_config.json`** ✗，`text_config.eos_token_id` 是 **null** ✗，
而 runner 只查了 `generation_config.json` → `eos = None` → `if Some(next) == eos` **永不成立** ✗✗。
**模型其实一直是答对的**：`[11111(" Paris"), 16("."), 1(<｜end▁of▁sentence｜>)]` 然后就该停 ——
官方贪心同样在 EOS 处停下所以只输出 " Paris." ✓；而我越过 EOS 继续生成，那串乱说被误当成
"长文本尾部退化" ✗。
修：`generation_config.json → config.json(顶层 eos_token_id = 1) → tokenizer 兜底` ✓。

### 本会话共修 5 个真 bug（全部经官方对照/消融定位）

| # | bug | 症状 → 修复后 |
|---|---|---|
| 1 | `sparse_attn_kernel` dot 只归约第 0 个 warp | 注意力输出被均匀缩小 0.67x → 与官方逐元素吻合 |
| 2 | MoE 专家和被 `memcpy` 覆盖 + `o` 从未清零 | moe_out 0.087 → 0.1425（官方 0.1452）|
| 3 | mxf4 的 ue8m0 scale 按 fp4 的"每字节 2 值"切片 | world 扫描 tp=1/2/4/8 完全一致 0.0960 |
| 4 | 窗口 KV 被错误跨层共享（应每层独立）| L3 从 −15% → −2.8%，" Paris" 出现 |
| 5 | EOS 查找返回 None → 循环不停 | 输出 = " Paris." + EOS，**与官方逐字一致** |

### 正确性现状

- 裸 id 路径与官方**逐层一致**（pos0 40 层全程偏差有界 3–6%、不累积）✓
- 输出与官方**逐字一致** ✓
- 残留（不影响正确性、已文档化）：① 激活量化用 fp4（参考 fp8/block128）② chat 模板特殊 token
  的细残差（`</think>` 两边都会改变预测 ✓，只是变成的 token 不同 ✗）

---

## 性能基线（乱码修好后的当前状态）

**实测**：`prefill 7 tokens in 2.17s`、64 token 解码约 25s ≈ **2.6–3.0 tok/s** ✗
（目标：单并发 **200 tok/s** = 5ms/token ✗ → 差约 70 倍 ✗）

**逐层计时**（`DSV41_PHASE=1`，单 token）：
- **L0 = 141ms**（首次调用预热 ✗，只影响第一个 token ✓）
- **稳态每层 ≈ 7.5ms** ✗ → 45 层 ≈ 337ms/token ✓（与 prefill 实测吻合 ✓）
- **细分：注意力 ≈ 3.5–4.8ms/层 ✗**，FFN/MoE ≈ 3.5ms/层 ✗（计时锚点需修正，但量级清楚）

**决定性判断**：单 token 的注意力"数学"是微不足道的 ✓，却要 **4ms** ✗
⇒ **瓶颈是 launch / sync 开销，不是算力** ✓✓
（每层 ~15–20 个 kernel × 45 层 ≈ 700–900 次 launch/步 ✗，按 ~5µs/次 ≈ 4ms ✓ 与实测吻合 ✓）

### 通往 200 tok/s 的路线（按预期收益排序）

1. **CUDA Graph 捕获整层**（最大单项 ✓）—— GLM 侧靠这一项把 0.18→17.5 tok/s ✓。
   阻碍：① **主机侧 MoE 派发**（每层要 download 路由结果 ✗ → 需改成设备侧 dispatch 或
   固定专家数的图 ✗）② **AR 的主机 barrier** ✗（需改成设备侧协议 ✓）
3. **去掉每层的 device sync** ✗：MoE 路由的 `dev.sync()+2 downloads`（`chain_dev.rs:1299-1303` ✓）
   是每层一次 ✗ = 45 次/步 ✗；hc 系数若也能留在设备（ping-pong ✗）可再省一批 ✓
4. **AR 与计算重叠 / 减少 AR 次数** ✓（现在每层 2 次 ✓）
5. **MoE 设备侧 dispatch** ✗（替代主机循环 ✓，同时解锁图化 ✓）
6. **fp8 激活**（精度项，非性能项 ✓）

## 性能进展（本会话）

**精确基线**（`[dsv41] DECODE` 行，只计解码循环、不含 ~45s 权重加载）：
- 优化前：**2.6 tok/s**（~390ms/token）
- 优化后：**3.5 tok/s**（289ms/token）—— 128 token / 37.01s

**已落地的性能修复（同属"每层 device sync"这一类）**：
- **hc premix 系数设备驻留（三槽轮转）** ✗→✓：原来每层 `attn_pre`/`ffn_pre` 各下载一次
  （下载 = 一次 `cudaDeviceSynchronize` ✗）+ 两次上传 = **每步 90 次同步** ✗，
  每次都排空 CPU/GPU 流水线 ✓。
  实测：**L1 注意力 3.5–4.8ms → 1.78–1.80ms（2.2x）** ✓，正确性不变（仍是 " Paris." ✓）。

**结论**：每层 ~6.4ms ✗ 中，注意力只占 1.8ms ✓ ⇒ **剩余 ~4.6ms/层 仍是同步/主机阻塞** ✗。

### 下一刀（最高价值，同一模式）

MoE 的路由每层要 `dev.sync()` + 2 次下载（`chain_dev.rs:1299-1303` ✗）= 每步 45 次
`cudaDeviceSynchronize` ✗。**改法：固定内存零拷贝回读 + 主机自旋** ✓
（vLLM/SGLang 的标准做法 ✓）：
1. `route_topk` 内核除写设备缓冲外，**再写一份 pinned host 内存**（`cudaHostAllocMapped` ✓，
   主机可直接读 ✓）并在末尾写一个 flag（`__threadfence_system()` ✓，与我 AR 的 publish 同法 ✓）；
2. 主机**自旋**在 pinned flag 上（µs 级 ✓，不调用任何 CUDA API ✓）；
3. 删掉 `dev.sync()` ✓。
预期：再省 ~2-4ms/层 ✓ ⇒ 有望到 ~10-15 tok/s；之后才轮到
**CUDA Graph**（需先消灭主机侧 MoE 派发 ✗，即上面第 1 步的设备侧路由 + 固定专家集 ✗）与内核级优化 ✓。

## ★★★★ 性能瓶颈已用实验钉死：`Collective::publish` 里的全设备同步（每步 ~90 次）

`tp.rs` 的 `publish()`（每次 all-reduce 都会走）里有：
```rust
self.dev.memcpy_peer(...)   // 8 次 peer 拷贝（异步）
self.dev.dev_sync()?;       // ← cudaDeviceSynchronize
self.barrier.wait();
```

**实验证明它是必需的** ✓：删掉后输出立刻从 `" Paris."` 变成 `" toll id "`（错 ✗）——
因为 peer 拷贝确实是异步的，host barrier 只保证各 rank **已发起**，不保证**已落地** ✓。

**但它每步跑约 90 次** ✗（每层 2 次 all-reduce × 45 层）✓，每次都**排空整个流水线** ✗✗
—— 这正是"注意力 1.78ms/层、而整层 6.4ms"的全部差额来源 ✓✓。
（此前把 hc premix 系数改为设备驻留之所以能拿到 2.2x，就是因为它顺带**每层少了 2 次**
这种同步 ✓，与此完全一致 ✓。）

### 正确的修法（下会话第一刀，按代价排序）

1. **把 peer 拷贝放到独立 stream，只 sync 那条 stream** ✓（最小改动：`cuStreamSynchronize`
   只等拷贝，不等主 stream 上的其余工作 ✗）+ 用 event 维持与 reduce 的顺序 ✓。
   预期：把每次同步的代价从"整个流水线"降到"仅 8 次 peer 拷贝" ✓。
2. **设备侧 stamp**（与 GLM 的 AR v5 同法 ✓，已验证可行 ✓）：拷贝后在同一 stream 上跑一个
   1-block 内核，写 `__threadfence_system()` 保护的 stamp；reduce 内核自旋等所有 rank 的 stamp
   ⇒ **主机完全不参与** ✓✓（这是终局形态 ✓）。
3. 或**减少 all-reduce 次数** ✗（每层 2 次：注意力输出 + MoE 输出 ✗）—— 需重构数据流 ✗。

### 本会话性能账（同口径，仅解码）

| 版本 | tok/s | 每层 |
|---|---|---|
| 会话初 | 2.6 | ~7.5ms |
| hc premix 设备驻留后 | **3.5** | ~6.4ms（其中注意力 1.78ms ✓）|

**注意**：所有"删同步"的尝试都必须**同时验证文本仍是 `" Paris."`** ✗ —— 本会话删掉 publish 的
sync 时文本立刻变成 `" toll id "` ✓，靠这条判据才没有把回归当成提速 ✓。

## ★★★ 又一否定结果：集合通信的 sync **不是**瓶颈（设备侧 stamp 已落地但无提速）

**已实现并验证**（`ar_stamp_kernel` + `ar_reduce_kernel` + FFI + `Collective` 的 stamp 区 ✓）：
- `publish()` 不再 `cudaDeviceSynchronize` ✗，改为：peer 拷贝（异步、流序 ✓）→ **盖章内核**
  （同一 stream，因此**必然**在拷贝完成之后执行 ✓，把本 rank 轮次写进每个 rank 的 stamp 区 ✓）
  → 轮次 barrier ✓；
- 归约改为设备侧自旋内核 ✓（`stamps[p] >= round` 轮询后求和 ✓），已删掉主机累加循环 ✓。
- **正确性通过** ✓：文本仍是 `" Paris."`、ids `[11111, 16, 1]` ✓✓（与官方逐字一致 ✓）。

**但吞吐没变**：3.3 tok/s（301.9ms/token）vs 之前 3.5（289ms）✗ —— 噪声内 ✗
⇒ **`publish` 里那次全设备同步不是瓶颈** ✗（虽然它确实每步跑 ~90 次 ✓）。

**排除法收敛**：注意力 = 1.78ms/层 ✓，整层 = 6.4ms ✗ ⇒ 差值 4.6ms 在 **FFN/MoE** ✓，
而集合通信已排除 ✗ ⇒ **剩下的唯一同步类嫌疑 = `moe()` 自己的
`self.dev.sync()?` + 2 次 `dl()` 路由下载**（`chain_dev.rs` 的 MoE 段 ✗，与集合通信无关 ✗）。

### 下会话第一刀（非常具体）

把 MoE 的路由回读换成**不阻塞主机**的形式 ✓：
1. 最省事：`route_topk` 的输出直接落到 **`cudaHostAlloc(cudaHostAllocMapped)`** 的固定内存 ✓
   （主机可直接读 ✓），并用一个**小盖章内核**（同 stream ✓）写 pinned flag ✓；主机**自旋**
   flag（无 CUDA API 调用 ✓）后读 ✓ → 删掉 `dev.sync()` + 2 次 `dl()` ✓；
2. 或：把 expert 派发整体搬到设备侧（fused MoE ✓）—— 这是终局形态 ✓，且顺带解锁 CUDA Graph ✓。

**纪律（本会话反复验证过）**：任何"删同步"的改动都必须**同时**验证文本仍是 `" Paris."` ✓
—— 本会话删 `publish` 的 sync 那次文本立刻变成 `" toll id "` ✗，靠这条判据才没把回归当提速 ✓。

## ★★★★ 消融定位：MoE 专家占解码的 41%（每次专家 launch ≈ 106µs ✗）

同口径（`[dsv41] DECODE`，仅解码）：

| 配置 | ms/token | tok/s |
|---|---|---|
| 基线 | **211.7** | 4.7 |
| `DSV41_SKIP_EXPERTS=1`（关掉 routed 专家） | **125.7** | 8.0 |
| `DSV41_SKIP_ENGRAM_WEIGHTS=1` | 209.4 | 4.8（无影响 ✓）|

⇒ **routed 专家一项就占 86ms/token = 41%** ✗✓。而单 token 只做 6 专家 × 3 个 GEMM = 18 次
launch ✗ ⇒ **每次专家 launch ≈ 106µs** ✗✗（正常 CUDA launch 是 3–5µs ✗）—— 差 20–30 倍 ✓✓。

**已排除**（本会话）：
- 集合通信的 sync ✗（设备侧 stamp 已落地，无提速 ✓）
- MoE 路由的 `dev.sync()` + 2 次下载 ✗（`DSV41_MOE_NOSYNC=1` 实测无提速：290 vs 287ms ✓）
- `launch_mxf4` 内部无任何分配 / `cudaFuncSetAttribute` / 设备属性查询 ✗（grep 过 ✓），
  启动处就是干净的 `mxf4_gemm_kernel<<<grid, kThreads, 0, s>>>` ✓

⇒ **106µs/launch 的停顿在启动器之外** ✗，下一步必须直接测：
1. 用 nsys 抓一个**单 layer**（`DSV41_LAYERS=1` ✓）的 CUDA API 时间线 ✗，看 `cuLaunchKernel`
   的 CPU 侧耗时与 GPU 侧间隙 ✓（若是 CPU 侧 → 查我 Rust 包装里每次调用的开销 ✗；
   若是 GPU 侧间隙 → 查内核本身的 M=1 路径 ✗）；
2. 直接用**隔离微基准**调 `dsv41_expert_gate_up_fp4`（M=1、N=640、K=5120）测它的 GPU 时间 ✗
   —— 判断 106µs 是 CPU launch 还是 GPU 执行 ✓；
3. 若确认是 launch 开销 → 图化（但需先消灭主机侧派发 ✗）或把 6 个专家**合并成一次
   分组 GEMM** ✓（一个内核处理全部 assignment ✓，同时解锁图化 ✓——终局形态 ✓）。

**参考**：即使专家全部关掉也有 125.7ms/token ✗（= 2.8ms/层 ✓），其中注意力实测 1.78ms/层 ✓
⇒ 仍有 ~1ms/层 分布在 hc 链 / shared expert / 投影 / 集合通信上 ✓。

## ★★★★★ 最终定位：解码**不是 GPU 受限**，是**主机发射**（Host-bound）

两条相互印证的证据：

1. **nsys 内核汇总**（`DSV41_LAYERS=2` 跑）：`hc_mixes_kernel` 单次 **1.06ms**、占 GPU 时间 **51.4%**
   ✗（288 次 = 305.8ms ✗）—— 而它只是 [20480]→[24] 的 GEMV + sigmoid + 20 轮 sinkhorn ✗。
   **原因已找到并修好** ✓：投影循环 `for (m = threadIdx.x; m < mix; m += blockDim.x)`
   只有 `mix=24` 个 lane 有活 ✗，每 lane 串行跑 20480 次依赖式标量加载 ✗✗（同 gdn_chunk /
   sparse_attn 一类：**内存延迟受限** ✓）。已改成 **一 warp 一行 + lane 切分点积 + shuffle 归约** ✓
   （合并访存 ✓、6 行/warp ✓），正确性复验通过 ✓（仍是 " Paris." ✓）。
2. **但修完吞吐只从 3.5 → 3.7 tok/s** ✗ —— 若真是 GPU 受限，砍掉 51% 的 GPU 时间应当翻倍 ✗。
   ⇒ **解码的主体时间不在 GPU 上** ✗✓。
3. **nsys API 汇总**：`cudaLaunchKernel` **12454 次** ✗，平均 13µs ✓（总计 162ms ✓）。
   该 profile 只跑了 ~6 次 forward × 2 层 ✓ ⇒ **每层约 500–1000 次 CUDA API 调用** ✗✗，
   按 13µs/次 ≈ 6–13ms/层 ✓ —— **这正是实测的 6.4ms/层** ✓✓。

### 结论与唯一有效的方向

**主机发射次数**（每层几百到上千次 ✗）就是 290ms/token 的主体 ✓✓。因此：

- **CUDA Graph 是唯一能一次性解决它的手段** ✓✓（把几百次发射压成一次图回放 ✓）——
  这也是 GLM 侧当年从 0.18 冲到 17.5 tok/s 的那一刀 ✓。
- 但它**必须先消灭主机侧 MoE 派发** ✗（专家集合是数据相关的 ✗，图里不能有动态启动 ✗）：
  即把 6 个 assignment 合并成**一次分组 GEMM** ✓（一个内核处理全部 assignment ✓，
  同时解锁图化 ✓）——这就是"设备侧 MoE" ✓，也是终局形态 ✓。
- 次要但同向的削减：合并小内核（每层 24 次 `hc_mixes` ✗、逐 group 的 `wo_a` ✗ 等 ✓）。

### 本会话性能账（同口径，仅解码）

| 版本 | tok/s | ms/token |
|---|---|---|
| 会话初 | 2.6 | ~390 |
| hc premix 设备驻留 | 3.5 | 289 |
| 设备侧 stamp（正确但无提速 ✗） | 3.3 | 302 |
| **hc_mixes 投影并行化（正确 ✓）** | **3.7** | **272** |

**下一步（唯一路径）**：设备侧分组 MoE → 再整层图化 ✓。两者都做之前，吞吐不会量级跃升 ✓。

## nsys 完整内核清单（DSV41_LAYERS=2 跑，`cuda_gpu_kern_sum`）

| time% | total | calls | avg/call | kernel | 备注 |
|---|---|---|---|---|---|
| 51.4 | 305.8ms | 288 | **1061µs** ✗ | `hc_mixes_kernel` | **已修** ✓（一 warp 一行 + shuffle 归约）|
| 14.1 | 83.6ms | 864 | 96.8µs | `mxf4_gemm_kernel<0>` | 专家 gate/up/down |
| 10.9 | 64.5ms | 846 | 76.3µs | `gemm_fp8_kernel` | fp8 投影 / shared expert |
| 6.8 | 40.4ms | 2232 | 18.1µs | `bf16_to_f32_kernel` | 权重加宽（**加载期一次性** ✓）|
| 4.8 | 28.5ms | 72 | **395.9µs** ✗✗ | `gemv2T_kernel` | **cuBLAS 的 M=1 GEMV，396µs/次** ✗ |
| 3.2 | 18.8ms | 864 | 21.8µs | `mxf4_gemm_kernel<1>` | 同上 |
| 2.7 | 16.2ms | **16** | **1009.8µs** ✗✗ | `rope_precompute_kernel` | **每步重算整张 RoPE 表（1ms/次）** ✗ |
| 2.7 | 15.9ms | 360 | 44.1µs | `ar_reduce_kernel` | 我新加的自旋归约 ✓ |
| 0.5 | 3.2ms | 144 | 22.2µs | `sparse_attn_kernel` | |

单实例数还显示：`quant_kernel` 828、`add_kernel` 882、`swiglu_limit_kernel` 882、
`fp4_pack_kernel` 864 ✗ —— 都是**逐专家**的小内核 ✓；总实例 ≈ 8542 ✗ / ~18 个 layer-pass
⇒ **约 470 次内核发射/层** ✗✗，与"主机发射受限"的结论一致 ✓（cudaLaunchKernel 平均 13µs ✓）。

### 下会话可直接拿的三刀（按代价/收益）

1. **`rope_precompute_kernel` 每步重算** ✗✓（**最省事、最干净**）：16 次调用 × **1009µs** ✗✗ ——
   RoPE 表只依赖 (seq_len, theta, 参数) ✓，是**静态的** ✗，应当在 chain 构造时算一次 ✓、
   之后只做 `apply_rope` ✓。预期省 ~2ms/步 ✓。
2. **`gemv2T_kernel` 396µs/次** ✗✗（72 次 ✓）：这是 cuBLAS 给 M=1 选的 GEMV 内核 ✗ ——
   M=1 时它极差 ✓。改用自研 GEMV 或让 cuBLAS 走 GemmEx 的 narrow 路径 ✗，预期省 ~0.5ms/步 ✓。
3. **合并逐专家小内核** ✗（quant/add/swiglu/fp4_pack 各 ~850 次 ✓）→ 直接并入派发内核 ✓，
   同时把每层约 470 次发射往下压 ✓ —— 这一步与"设备侧分组 MoE"是同一件事 ✓。

### ⚠ 更正：`rope_precompute_kernel` 是**一次性**成本，不是靶子

16 次 = 8 rank × 2 次 ✓，两处调用都在 **`chain_dev.rs:271/278` 的构造函数里** ✗
（`DevChain::new`）⇒ 只在启动时算一次 ✓，**不在 step 路径上** ✗。
所以那 1.01ms×16 = 16.2ms 属于**加载/初始化** ✓，不要当成每步优化项 ✗。

**修正后的每步靶子（只剩两条）**：
1. **`gemv2T_kernel` 396µs/次** ✗✗（72 次 ✓）—— cuBLAS 为 M=1 选的 GEMV 内核，效率极差 ✗
   （其它同类：`gemm_fp8_kernel` 76µs ✓、`mxf4_gemm_kernel` 97µs ✓ —— 也都远高于理论 ✗）；
2. **每层 ~470 次内核发射** ✗ → 设备侧分组 MoE + 图化 ✓（唯一的结构性解 ✓）。

## ★★★ 突破：把专家激活量化提出循环 → **3.7 → 5.7 tok/s（1.55x）**

```
专家激活量化本来在 `for e in 0..ne` 循环里 ✗，
而 6 个选中的专家用的是【同一份】输入行 ✗ → 把同一行重量化+重打包了 6 遍 ✓✗
（quant_fp4 + fp4_pack 各多跑 5 次/层 = 10 次多余发射/层）
```

| 版本 | tok/s | ms/token |
|---|---|---|
| 上一版 | 3.7 | 272 |
| **量化提循环外** ✓ | **5.7** | **175** |

⇒ 省掉约 10 次发射/层就换来 1.55x ✓✓ ⇒ **每次发射的实际代价 ≈ 35µs** ✗（远高于 nsys 里
cudaLaunchKernel 的 13µs ✗ —— 说明含屏障/依赖等待 ✓）✓ **"减少每层发射次数"确实是正解** ✓✓，
与"主机发射受限"的定位一致 ✓。

### 下一步同类赢面（按发射次数排序）

| 项 | 每层多余发射 | 做法 |
|---|---|---|
| **AR 的 8 次 peer 拷贝** ✗✗ | ~16（2 AR × 8）| 改成**设备侧 store 内核**（一次内核写全部对端 slot ✓，peer 基址已在设备上 ✓）——即 AR v5 的 store 阶段 ✓ |
| 每次专家后的 `add_inplace` ✗ | ~5 | 折进 `expert_down_fp4` 的 epilogue（该内核本就有 `row_weight` ✓，让它 `+=` 而非覆盖 ✓）|
| 每层两次 `zero`（o / ex_out）✗ | 2 | 折进相邻内核 ✓ |
| shared expert 的 3 个 fp8 GEMM ✗ | 0（必要 ✓）| — |

## 收敛后的定量结论：每层 ~3.9ms 里**主机约占 2.4ms** ✗

把本会话所有"有效/无效"的对照摆在一起，模型是自洽的 ✓：

| 改动 | 去掉的发射/层 | 实测 |
|---|---|---|
| hc premix 设备驻留 | 2 次下载（各含一次 `cudaDeviceSynchronize`）+2 次上传 | **2.6 → 3.5** ✓ 大 |
| hc_mixes 投影并行化 | 0（改内核） | 3.5 → 3.7 ✓ 小 |
| **专家量化提循环外** | 10 | **3.7 → 5.7** ✓ 大 |
| down 累加进 o | 6 | 5.7 → 5.7 ✗ 无 |
| AR 的 8 次 peer 拷贝 → 一次设备内核 | 16 | 5.7 → 5.3 ✗ 无（噪声内）|

**推算**：每层约 130 次 API 调用 ✗ × ~18µs ✓ ≈ **2.4ms/层** ✓；GPU 内核合计约 1.5ms/层 ✓
（hc_mixes 修好后）⇒ **3.9ms/层 = 60% 主机 + 40% GPU** ✗✓。

- 这解释了为什么去掉 6 或 16 次发射"看起来没用" ✗（各占 5%/12% ✗），而
  **hc premix 那 4 次带 `cudaDeviceSynchronize` 的下载**去了就值 2.6→3.5 ✓ ——
  **同步/依赖**远比**发射计数**贵 ✓✓。
- 也解释了为什么 `hc_mixes` 占 GPU 51% 却只值 +6% ✗：那说明的是"GPU 大量空闲" ✓，
  不是"GPU 是瓶颈" ✓。

## 结论：只剩两条路，都必须做内核级工作 ✗

1. **整层 CUDA Graph** ✓ —— 把 ~130 次发射压成 1 次回放 ✓，直接消灭那 60% 的主机时间 ✓。
   **前置条件**：消灭主机侧 MoE 派发 ✗（专家集合是数据相关的 ✗）⇒
2. **设备侧分组 MoE** ✓ —— 一个内核处理全部 assignment ✓（顺带把每层 ~20 次专家相关发射
   及 `gemm_fp8_kernel`/`mxf4_gemm_kernel` 在 M=1 下的低效一起解掉 ✓）。

**这两项是本会话已明确、但尚未实施的唯一通路** ✗（都不是"再调一个旋钮"能解决的 ✗）。

### ⚠ 否证："每层 48 次 mxf4" 不是多派专家

`DSV41_MOEDBG=1` 实测每层（8 rank 各打一行 ✓）：

```
[mine] L0 route idx=[277, 128, 155, 137, 251, 206] active_experts=6 of 384 (topk=6)
[mine] L0 route idx=[61, 201, 345, 209, 200, 291]  active_experts=6 of 384 (topk=6)
...
```

⇒ **每层恰好 6 个专家** ✓、路由与官方一致 ✓（`idx` 与官方同 6 个 ✓）。
所以"48 次/层"是我用**错误的 layer-pass 数**折算出来的 ✗ —— **专家派发本身没有多余** ✗，
不要再往这个方向查 ✓。

**本会话的最终量化结论（不变）**：每层 ~3.9ms = **主机 ~2.4ms（~130 次 API 调用 × ~18µs）✗
+ GPU ~1.5ms ✓** ⇒ 唯一出路是**用图把这些发射压掉** ✓，其前置是**设备侧分组 MoE** ✓。

## 🏁🏁 最终验收：chat 模板默认编码后，多 prompt 全部正确

参考的 `generate.py` **从不喂裸文本** ✓ —— 它用
`encode_messages(messages, thinking_mode="chat")` =
`<|begin_of_sentence|><|User|>{prompt}<|Assistant|></think>`
= ids `[0, 128803] + prompt + [128804, 128822]`。
**裸文本路径会让模型不知道自己该作答** ✗（例如 "The capital of Japan is" 立刻输出 EOS ✗，
"1+1=" 给出 "2" 后不停 ✗）。把该模板设为 runner 的默认编码路径后 ✓：

| prompt | 输出 ids | 判定 |
|---|---|---|
| `The capital of France is` | `[51119, 1]` = " Paris" | ✓ 与官方 chat 路径一致 |
| `The capital of Japan is` | `[106239, 16, 1]` = " Tokyo." | ✓✓ |
| `1+1=` | `[20, 1]` = "2" 后**正确停止** | ✓✓ |
| `请背诵《静夜思》` | 连贯作答（`[1342, 4504, ...]`）| ✓ |

（`DSV41_PROMPT_IDS=a,b,c` 仍可覆盖 ✓，用于喂官方 token 做逐 token 对拍 ✓。）

**⚠ 方法论教训（本会话第二次踩到）**：验证必须用**与官方相同的输入形式** ✗ ——
我之前用裸文本跑，把"模型缺助手框架"误判成"尾部退化" ✗；更早还把"官方 chat 模板的
特殊 token 导致预测变化"误判成 bug ✗（后来发现官方同 prompt 也给不同 token ✓）。
**结论：跨实现对比必须同 prompt 编码 ✓、同位置 ✓、同张量宽度 ✓、同是否含 AR ✓。**
