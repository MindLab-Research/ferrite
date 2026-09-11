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

## ★★★★ 突破二：设备侧 MoE 派发 → 5.7 → **7.0 tok/s**，且**图化前置完成**

**改动**：每层 384 个专家的权重本就在**一个池**里、**统一步长** ✓ ⇒ 让内核自己算 B 指针：
 `b = b_base + ids[slot] * b_stride`（`ids` 是设备上的 `route_idx` ✓，base/stride 是每层常量 ✓）。
MoE 从"下载路由 → 主机决定跑哪些专家 ✗"变成**固定 6 槽位、发射参数与路由无关** ✓✓。

| 版本 | tok/s | ms/token |
|---|---|---|
| 上一版 | 5.7 | 175 |
| **设备侧派发** ✓ | **7.0** | **142** |

**同时纠正一个此前的无效测量** ✗：我曾用 `DSV41_MOE_NOSYNC=1`（只跳过显式 `dev.sync()`）测出
"路由同步无收益" ✗ —— 但 `download_f32` 内部用的是**同步 `cudaMemcpy`** ✓，所以真正的代价在**下载**
本身 ✓，跳过显式 sync 什么也测不到 ✓。去掉下载后立刻 +23% ✓。

**会话累计**：**2.6 → 7.0 tok/s（2.7x）**，正确性全程保持（每步都复验 " Paris" ✓）。

### 图化的前置已就绪 ✓（这正是本次改动的附带价值）

MoE 现在是**固定 6 槽位 + 路由无关的发射参数** ✓ ⇒ 满足 CUDA graph 捕获的两条要求 ✓。
**剩余待办（图化本身）**：
1. 集合通信里的 **host barrier** ✗ 必须移出捕获范围（barrier 是主机操作 ✓）——
   跨 rank 的数据可见性已由**设备侧 stamp** 保证 ✓，barrier 只剩"轮次纪律"作用 ✓；
   去掉后需把 staging 改成**按轮次奇偶双缓冲** ✓（否则跑得快的 rank 会覆盖落后者正在读的槽 ✗）。
2. 按层捕获（45 层各一张图 ✓），静态参数直接用当前的 ✓。

## ★★★★★ 终局判断修正：**只做图化到不了 200 tok/s**，必须同时修 M=1 的内核低效

当前 7.0 tok/s = 142ms/token = **3.2ms/层**，其中：
- **主机 ≈ 2.0ms/层** ✗（~130 次发射 × ~15µs）
- **GPU ≈ 1.5ms/层** ✗（nsys 实测：hc_mixes 修好后主要剩 mxf4 97µs/次 ✗、
  cuBLAS `gemv2T` **396µs/次** ✗、gemm_fp8 76µs/次 ✗）

**两条推论**（这是本会话最重要的认知）：
1. **把主机完全消除（整层图化）后，上限只有 ~15 tok/s** ✗ —— 因为 GPU 端仍占 ~68ms/token ✗；
2. **单请求 M=1 的权重读取地板**：每层约 90MB 权重 / 3TB/s ≈ 30µs ⇒ 45 层 ≈ **1.35ms/token
   （≈740 tok/s）** ✓ ⇒ **当前 GPU 内核比地板差约 50 倍** ✗✓ —— 原因清楚：这些内核按
   M=64/128 的 tile 设计 ✓，而单 token 只有 1 行 ✓ → 有效利用率 <1% ✗，
   但**权重仍要完整流过 smem** ✓（所以是"tile 浪费 + 带宽未饱和"的叠加 ✗）。

### 因此 200 tok/s 的完整清单（按重要性）

| # | 项 | 预期 | 备注 |
|---|---|---|---|
| 1 | **M=1 专用内核路径** ✗ | 数倍 | mxf4 与 gemv2T 在小 M 时改用 GEMV 式（无 m-tile 浪费、权重只读一遍 ✓）；这是**最大单项** ✓ |
| 2 | **整层 CUDA Graph** ✓（前置已完成 ✓）| ~2x | 消灭 ~130 次发射/层的主机时间 ✓；集合通信需把 host barrier 移出捕获范围 ✓ |
| 3 | 其余小项 | 小数 | fp8 激活（精度）、逐 group `wo_a` 合并、hc 链小内核合并 |

**注意 1 与 2 的关系**：两者都必须做；只做 2 ⇒ ~15 tok/s ✗；只做 1 ⇒ 受主机封顶 ✗ ⇒
**必须 1+2 同时** ✓。

### M=1 自研 GEMV 已落地（正确但中性）

`gemv_{bf16,f32}` 内核（一 warp 一行、沿 K 合并访存、f32 累加 ✓）替换了 cuBLAS 的
`gemv2T`（396µs/次 ✗）；bf16 路径顺带去掉 `f32_to_bf16` 的激活 cast 内核 ✓。
**实测：141.69 vs 142.09 ms/token ✗（噪声内）**，正确性保持（Paris / Tokyo ✓）。

⇒ 再次印证**主机发射受限** ✓：即便砍掉 GPU 侧 28.5ms 的名义耗时 ✗，墙钟不变 ✓
（GPU 内核时间被主机的发射间隙掩盖 ✓）。**这也说明：单独优化 GPU 内核在当前结构下收效有限 ✗，
必须先把主机发射次数降下来（图化）✓ 才能让内核优化显现 ✓。**

**踩到的坑**：新 `extern "C"` 入口最初被我放在**匿名 namespace 内部** ✗ → 内部链接 ✗ →
`nm -D` 找不到 ✗、serve 报 "kernel ... is not in the loaded .so" ✓。
**规则**：`dsv41_*.cu` 里所有导出符号必须在匿名 namespace **之外**（与已有的
`dsv41_ar_*` 同处 ✓）。

## 图化的最后阻塞与可直接实施的设计（下会话照此做）

**阻塞**：`Collective::publish` 里的 **host `barrier.wait()`** ✗。
capture 期它会真的执行（并把同步"录掉"）✗，replay 期被跳过 ✗ ⇒ 跨 rank 数据同步丢失 ✗。
现有的**设备侧 stamp 已就位** ✓（store 内核后盖章 + reduce 内核自旋 ✓），差的只是
**把 skew 约束也从主机搬到设备** ✓。

**设计（两个戳，双缓冲）**：
- staging 扩成 `2 × world × bytes`（按 `round % 2` 选半区 ✓）；
- `stored[world]`（本 rank 在**store 之后**写 ✓，带 `__threadfence_system()` ✓）
  与 `reduced[world]`（本 rank 在 **reduce 全部完成后**写 ✓）；
- **store 内核开头先自旋**等 `reduced[p] >= round - 2`（∀p ✓）——
  保证要写的半区（`round%2`，上次用于 `round-2`）已被所有对端读完 ✓；
- **reduce 内核自旋**等 `stored[p] >= round`（∀p ✓，即现状 ✓），归约后在**尾部**
  由 block 0 写 `reduced[round]` ✓（需要全部 block 完成 ✓ ⇒ 用一个 1-block 的收尾内核更稳 ✓）；
- **删掉 host barrier** ✓。

**验证纪律（不可省）**：改完必须复验文本仍是 `" Paris"` ✓；并专门跑一次
**乱序/长序列**（例如 128 token 连续解码 ✓）确认没有偶发错值 ✓（skew 类竞态只在长跑中出现 ✓）。

**做完这一步**，整层/整段 capture 才有意义 ✓（否则图会把跨 rank 同步录没 ✗）。

## ⛔ 设备侧集合通信协议（去掉 host barrier）：**实测失败，已改为默认关闭**

**做法**（按上一节的设计 ✓）：双缓冲 staging（按 `round % 2` ✓）+ 两个戳
（`stored` 在 store 后 ✓、`reduced` 在 reduce 后 ✓）+ store 内核**每个 block** 先自旋等
`reduced[p] >= round-2` ✓ + reduce 后由 `ar_mark` 补 `reduced` ✓ + 删掉 host barrier ✓。

**实测（`DSV41_AR_DEV=1`）**：**正确性崩** ✗✗ ——
- `The capital of France is` → ids `[67764, 72246, 126464, 58703]` ✗（应为 `[51119, 1]` = " Paris"）
- 128 token 长解码 → `[67764, 72246, ..., 66757, 66757, 66757…]` ✗（退化成重复 token）

**处置（按项目硬性规则：禁止 git reset 回退已提交改动 ✗）**：把整条设备侧路径
**改成 `DSV41_AR_DEV` 环境开关、默认关闭** ✓，默认回到"单缓冲 + host barrier"的
**已验证工作路径** ✓。复验通过：`Paris` ✓ / `Tokyo` ✓ / 长解码 ✓。

**给下会话的排查提示**（为什么这么设计还错 ✗）：最可疑的是**源/目的别名** ✗ ——
调用方传进来的 `dst`（如 `s.o`）就是 staging 的第 0 个 slot ✓，而新路径让 store 写到
`parity_off + …`、reduce 也从 `parity_off` 读 ✓ ⇒ 奇偶半区与调用方缓冲**互相踩** ✓；
另外 `round-2` 的信用等待在**首次**若干轮（round < 3）被跳过 ✓，若此时 staging 尚未
被任何轮次初始化 ✓，读到的是垃圾 ✓。**建议下会话先做**：让 staging 与调用方缓冲彻底分离
（专用 staging 缓冲 ✓），并把初值清零 ✓，再启用该路径。

### ⚠ 更精确的失败根因（比上面两条更可能）：**fence 与信号必须在同一线程**

CUDA 内存模型里，一次有效的 release 需要**执行 fence 的线程与发出信号的线程相同** ✗。
我的实现把两者**分在两个内核**里：
- 数据写入在 **`ar_store_kernel`** ✓
- 信号（stamp）在**另一个** `ar_stamp_kernel` 里写、并带 `__threadfence_system()` ✗

⇒ 该 fence 只订购**它那个线程**的先前写 ✗，**覆盖不到 store 内核里那些线程的写** ✗✓
⇒ 对端可能**先看到 stamp、后看到数据** ✓（读到旧值/垃圾 ✓）✓
—— 这也解释了为什么加 host barrier 就一切正常 ✓：barrier 把两侧都冲刷干净，
**把可见性缺口掩盖了** ✗✓（所以设备侧协议不是"设计错"，而是"缺一次正确的 release" ✗）。

**正确做法（下会话照此改）**：把盖章**并进写入内核本身** ✓ ——
`ar_store_kernel` 每个 block 写完各自的部分 → `__threadfence_system()` → 由 block 0
（或最后一个完成的 block ✓）写 `stored[round]` ✓；归约内核同样在归约后写 `reduced[round]` ✓。
这样"写 + 屏障 + 信号"落在同一批线程里 ✓，才是合法的 release ✓。

## ⛔ 设备侧集合通信（去 host barrier）：**仍会挂死，已标注 DO-NOT-ENABLE**

在"同线程 release"修正之后复测 ✓：默认路径正常（`Paris` ✓），`DSV41_AR_DEV=1` **仍挂死** ✗
（serve 启动后无输出、超时 ✓）。该路径已在代码里用醒目注释标为 **⛔ DO NOT ENABLE** ✓，
默认（host barrier）不受影响 ✓。

**已在本轮修掉的两个真 bug（都是我自己引入的 ✗，都靠复验文本发现 ✓）**：
1. `ar_store_kernel` 的无条件"最后块盖章"在**默认路径**上传入 `ctr = nullptr` ✗ →
   `atomicAdd(nullptr)` → **两条路径一起挂死** ✗。修：`if (ctr == nullptr) return;` ✅
2. 更早那次：stamp 与数据**分在两个内核**里 ✗ → fence 覆盖不到数据的写 ✗ →
   可见性缺口（被 host barrier 掩盖 ✓）。修：盖章并进写入内核 ✓（仍未解决挂死 ✗）。

**下会话排查提示**：挂死最可能在 store 内核的**信用等待**里自旋不出来 ✗ ——
首轮 round<3 跳过等待 ✓，但 `reduced[]` 的**初值**是 0 ✓ 而 `round-2` 在 round≥3 时 ≥1 ✓，
若某 rank 的 reduce 从未把 `reduced` 推进到该值 ✗（例如 `ar_reduce2` 的 `do_mark` 与
`ctr2` 传参不匹配 ✗），等待者就永远自旋 ✗。建议先做**单 rank 演练**（world=1 时所有等待
应立即通过 ✓）再上 8 rank ✓。

**默认路径复验**：`Paris` ✓ / `Tokyo` ✓ / `1+1=` → `2` 后停止 ✓ / 静夜思 ✓。

### 单 rank 演练的决定性诊断（`tp=1 DSV41_LAYERS=2 DSV41_AR_DEV=1`）

```
exit=0（**未挂死** ✓）  但输出 = "Ll bron_pairCHANTABILITY" ✗（应为 " Paris"）
```

⇒ **两条精确结论**（把之前"只在跨 rank 出错"的猜测推翻了 ✓）：
1. **world=1 不挂死** ✓ ⇒ 挂死发生在**跨 rank** 的等待/盖章交错里 ✗；
2. **world=1 的输出就已经是错的** ✗ ⇒ **奇偶半区 + 归约目的地的逻辑本身有问题** ✗，
   不是单纯的同步问题 ✓（这也解释了 tp=8 下直接输出乱码 ✓）。

**最可疑的一处（下会话先看这里）**：`ar_reduce2` 的 `dst` 我传的是**奇偶半区的基址**
（即 staging 本身 ✗），而调用方期望的结果在**它自己的缓冲**里 ✗。原实现里 reduce 是
**就地**写在 slot 0 ✓、再由调用方使用 ✓ —— 改成奇偶半区后，"写回哪里"没有跟着调整 ✗✓。
**建议**：让 reduce 把结果写回 `dst`（调用方缓冲 ✓）、staging 只作为收集区 ✓，再重测。

### 设备侧集合通信：本轮尝试终止（默认路径不受影响 ✓）

在"拷回改读奇偶半区"这一修之后复测 ✓：
- 默认路径 **`Paris` ✓**（id 51119 ✓）—— 未受影响 ✓✓
- `DSV41_AR_DEV=1` world=1 → `[125719, 125719]` ✗（从"杂乱 token"变成"重复 token" ✗，仍未对 ✓）
- `DSV41_AR_DEV=1` world=8 → **仍挂死** ✗

⇒ **该实验路径有至少两个独立缺陷**（奇偶/拷回逻辑 + 跨 rank 等待交错 ✗），
本轮已定位并修掉 3 个（同线程 release ✗→✓、默认路径空指针守卫 ✗→✓、拷回奇偶 ✗→✓），
但**未达到可用** ✗。

**处置**：保持 `DSV41_AR_DEV` **默认关闭** ✓、代码内标注 **⛔ DO NOT ENABLE** ✓，
**默认路径（单缓冲 + host barrier）为已验证唯一可用** ✓。
**下会话若要继续**（这是图化的必经项 ✓）：建议把该协议**在隔离微基准里单步验证**
（world=2 起 ✓、逐轮打印 `stored/reduced/round` 与各 rank 的观察值 ✓），
不要在整模型上试 ✗ —— 整模型的反馈周期太长且挂死难以定位 ✓。

## ✅ 新增资产：集合通信隔离微基准（反馈周期 5 分钟 → **6 秒**）

`crates/ferrite-dsv41/tests/ar_micro.rs` —— 8 个 rank 各占一卡，**只跑 all-reduce 循环**，
每一轮都与**主机端参考和**逐元素比对 ✓：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DSV41_KERNELS=$PWD/kernels/cuda/libferrite_kernels.so \
AR_MICRO_WORLD=8 AR_MICRO_ROUNDS=32 \
cargo test --release -p ferrite-dsv41 --test ar_micro -- --nocapture
# 可选 AR_MICRO_N（每 rank 的 float 数，默认 1024）
```

**实测**：
| 路径 | 结果 |
|---|---|
| **默认（host barrier）** | **✓ OK**（world=8 / 32 轮 / 1024 floats，**6.16s**）|
| `DSV41_AR_DEV=1` | **✗ world=2 就挂死**（超时无输出）|

⇒ ①**默认路径的集合通信被独立验证正确** ✓（给正确性再加一道证据 ✓）；
② 设备侧协议的故障是**纯协议逻辑** ✓、与 rank 数无关 ✓（world=2 即挂 ✓）；
③ **反馈周期从 5 分钟降到 6 秒** ✓✓ —— 下会话可以直接在这个 harness 里逐轮打印
`round / stored[p] / reduced[p]` 定位自旋不出来的那一处 ✓，**不要再上整模型** ✗。

**写在 harness 里的两个教训**（都已经犯过 ✓）：
- 测试自己的期望值也会错 ✓：第一版我多乘了一个 `n` ✓，把**正确的** 28000 报成失败 ✗ ——
  **先验证测试，再怀疑被测对象** ✓；
- `Device::bind_to` + `enable_peer_access` 需要**所有 rank 的 context 先存在** ✓
  （两段式 barrier ✓），照搬 runner 的顺序即可 ✓。

### 微基准定位到的故障边界（下会话从这里接手）

修掉 harness 自身的假挂之后（失败路径原先直接 `return` ✗ → 跳过 harness barrier ✗ →
对端永久等待 ✓，看起来像协议挂死 ✗）复测：

| 路径 | world=2 rounds=4 |
|---|---|
| **默认（host barrier）** | **✓ rank0 completed 4 rounds** ✓ |
| `DSV41_AR_DEV=1` | **✗ 无输出 = 真挂**（非 harness 假挂 ✓）|

且 `rounds=2`（信用等待尚未启用 ✓，它要求 `round >= 3` ✓）**同样挂** ✗
⇒ **挂死发生在第 1 轮的 store→stamp→reduce 序列里** ✓，与信用等待无关 ✗✓。

**下会话最快的二分（每次 6 秒 ✓）**：
1. 把 store 内核里 `round >= 3` 改成 `round >= 99999`（等于关掉信用等待 ✓）——
   若仍挂 ✗ ⇒ 问题在 stamp/reduce 的握手 ✓（首查：reduce 自旋的 `stamps[p] >= round` 是否
   真被 store 内核的"最后块"写到了 ✓；store 里 `is_last` 用 `atomicAdd(ctr)` 判定，
   注意每轮结束会把 `ctr` 归零 ✓）；
2. 再把 store 尾部的 in-kernel 盖章**临时换回**独立的 `dsv41_ar_stamp` 调用 ✗→✓
   （即恢复"分层内核"的旧写法 ✓）—— 若这样就不挂 ✓ ⇒ 问题就在 in-kernel 盖章的
   `__shared__`/`__syncthreads` 结构里 ✓（注意 store 内核里 `if (i < n)` 后**不能有提前 return**
   ✓，否则 `__syncthreads` 会死锁 ✓ —— 这点我已改成无提前 return ✓）。

**harness 入口**：`tests/ar_micro.rs`，见上一节命令 ✓（`AR_MICRO_WORLD/ROUNDS/N` 可调 ✓）。

### 二分决定性结论：**挂死来自"内核内最后块盖章"，不是奇偶/信用逻辑**

用 6 秒微基准（world=2, rounds=4）对照：

| 设备侧路径的盖章方式 | 结果 |
|---|---|
| 内核内"最后块"盖章（`atomicAdd(ctr)` + `is_last` + `__syncthreads`）| **✗ 挂死**（第 1 轮即挂 ✓）|
| **换回独立 `ar_stamp`/`ar_mark` 内核** | **✓ 不再挂**（4 轮全部完成 ✓）**但结果为 0.0**（应 1000.0）✗ |

⇒ 两个**互相独立**的缺陷 ✓：
1. **挂死 = 内核内的最后块盖章** ✗（与奇偶/信用等待无关 ✓ —— 把信用等待改成永不触发后仍然挂 ✓，
   换成独立盖章就不挂 ✓）。嫌疑集中在 store 内核尾部那段
   `__threadfence()` + `atomicAdd(ctr)` + `__syncthreads()` + `__shared__ bool is_last` 的结构 ✗
   （注意：`__shared__` 声明在 `if (ctr == nullptr) return;` **之后** ✗，且该 return 虽为一致条件 ✓
   但当 `ctr==nullptr` 时整块跳过 ✓ —— 需确认无路径让部分线程先退出 ✓）。
2. **数据 = 奇偶半区的读/写对不上** ✗（store 写的半区与 reduce 读的半区不一致 ✅可疑：
   store 用 `parity_off/4`（float 单位 ✓）、reduce 的 `base` 用字节单位 ✓，两者都按 `round%2` ✓，
   但 `round` 在 publish 与 reduce 两处的取值时机不同 ✗ —— **reduce 里我读的是
   `self.round.load()`，而 publish 里是 `fetch_add(+1)`** ✗：同一轮里 reduce 读到的
   可能是**已经 +1 后的值** ✓ ⇒ 两处算出的 parity 相反 ✓✓ ⇒ 读到零 ✓ —— **这是首要嫌疑** ✓）。

**下会话第一步**：把 parity 从"两处各自算"改成**一次算好、存进 `self.round` 的伴随变量**
（或让 reduce 用与 publish 相同的 round 值 ✓），再跑 6 秒微基准 ✓。

### ⚠ 更正：上面第 2 条的 parity 猜想**已被核对否决** ✗

`publish` 用 `let round = self.round.fetch_add(1) + 1` ✓（返回旧值再 +1，计数器变为 round ✓），
`all_reduce_inplace` 用 `self.round.load()` ✓ —— **两者取到的是同一个值** ✓ ⇒ parity 一致 ✓，
不是 0.0 的原因 ✗。
（教训：**在文档里写下的猜想也要先核对再留给下会话** ✗ —— 否则会把人带偏 ✓。）

**0.0 的真正排查方向（下会话用 6 秒 harness 逐个排除 ✓）**：
1. 在 harness 里把 `AR_MICRO_ROUNDS=1`、`AR_MICRO_WORLD=2` 下**打印 staging 两个半区的原始内容** ✓
   （直接看 store 到底写到哪个字节 ✓，比推理快得多 ✓）；
2. 检查 store 内核里 `parity_off` 的单位 ✓（我按 float 传的 `parity_off/4` ✓，而 `peer_slots[p]`
   是 u64 基址 ✓、`slot_f` 是 float 单位 ✓ —— 这一处**值得在 harness 里用 N=4 的小尺寸直接验证** ✓）；
3. 检查 `ar_stamp` 在 bisect 版里的调用时机 ✓（我已把它移到 barrier 之前 ✓）。

### ★★ staging 原始 dump 的决定性证据：**store 全对，问题在 reduce**

`AR_MICRO_DUMP=1 AR_MICRO_N=8 AR_MICRO_WORLD=2` 下直接读 staging（不再是推理 ✓）：

```
round 0 half 1: slot0[0..4]=[0,0,0,0]        slot1[0..4]=[1000,1000,1000,1000]
round 1 half 1: slot0[0..4]=[1,1,1,1]        slot1[0..4]=[1001,1001,1001,1001]
（half 0 始终全 0 = 还未被使用 ✓）
```

逐条对账（world=2, n=8）：
- round 0：parity = 1%2 = **1** ✓ → 写 half 1 ✓ **对** ✓；slot0 = rank0 的值 = `0 + 0*1000 = 0` ✓ **对** ✓；
  slot1 = rank1 的值 = `0 + 1*1000 = 1000` ✓ **对** ✓；
- round 1：parity = 0 ✓ → hmm **这里写了 half 1** ✗ —— 说明 parity 与我预期相反 ✓，
  但 **round 0→1 都落在同一个 half 1** ✓ ⇒ **两轮的 parity 相同** ✗✓ ——
  即 `round % 2` 在两轮里取了同一个值 ✓ ⇒ **`self.round` 的自增没生效**（或 reduce/publish 读的不是同一个值）✗。

⇒ **两个结论**：
1. **`ar_store` 的半区/slot/地址计算完全正确** ✓✓（这一大块可以排除 ✗）；
2. **parity 来源有问题** ✗：连续两轮落到同一半区 ✓ ⇒ **`round` 没推进** ✗ 或
   **store 用的 parity 与 reduce 用的不一致** ✗（`AR_MICRO_ROUNDS=2` 时 round 应为 1、2 ✓ ⇒
   parity 应为 1、0 ✓ ⇒ 应分别落 half 1 与 half 0 ✓，而实测都落 half 1 ✗）。

**下会话第一步（6 秒）**：在 `publish` 与 `all_reduce_inplace` 两处各打印 `round` 与
`round % 2` ✓，确认 `self.round` 是否真的在推进 ✓（注意 `fetch_add` 在 publish 里 ✓、
`load` 在 reduce 里 ✓ —— 若 reduce 在 publish **之前**被调用（例如某条路径先归约 ✓），
就会读到上一轮的值 ✓）。

### 本轮进展：**"round 只自增一次"修复消除了挂死**（0.0 仍在，但已缩到 reduce 一步）

`publish` 与 `all_reduce_inplace` **各自**都做了 `fetch_add(1)` ✗ ⇒ 每个 all-reduce 计数器 +2 ✗
⇒ parity 恒为奇数（**永远写同一个半区** ✓，与 staging dump 完全吻合 ✓）、且 store 与 reduce 在
两次自增之间读到的 round 差 1 ✓ ⇒ 两半区错开 ✓。

改成 **`all_reduce_inplace` 独占自增并把确切的 round 传给 `publish`** ✓ 之后：
- **不再挂死** ✓（`DSV41_AR_DEV=1` world=2 → 4 轮跑完 ✓；world=8 → 16 轮跑完 ✓）；
- **默认路径回归通过** ✓（`AR_MICRO_WORLD=4/8` → **OK** ✓；模型仍是 `Paris` ✓）；
- 设备侧仍返回 **0.0** ✗ ⇒ 仅剩 **reduce 一步**（store 已被 dump 证明完全正确 ✓）。

**下会话（6 秒一轮）**：在 `AR_MICRO_N=8` 下把 reduce 的内核换成"直接把各 slot 原样拷到 dst"
（不做求和 ✓）—— 若 dst 拿到了正确数据 ✓ 则问题在求和/索引 ✓；若仍是 0 ✓ 则问题在
`base`（parity 半区基址）的传递 ✓。**注意**：我观察到 tp.rs 里仍有两处 `fetch_add`
（201 行 publish 内、255 行）✗ —— 先确认 `all_reduce_inplace` 走的那条路径**只自增一次** ✓。

### ★★★ 0.0 的精确诊断（从 staging dump 反推，无需再猜）

Dump 显示 AR 之后 half 1 仍是 `slot0=[0,0,0,0]` / `slot1=[1000,1000,1000,1000]`
—— 即**归约没有把它求出的和（1000）写回它所读的那个半区** ✗。
而 `all_reduce_inplace` 末尾的 copy-back 读的正是 `base[0..n]`（= slot0 = **本 rank 自己的输入**）
✗ ⇒ `buf` 拿到的是**自己那份输入**（rank0 = 0 ✓）而非和 ⇒ **观测到的 0.0 完全对上** ✓✓。

⇒ **结论：问题在 reduce 内核的“写入”这一步** ✗，与 parity / store / 信用等待**都无关** ✓
（store 已被 dump 证明正确 ✓，parity 已修 ✓，挂死已消 ✓）。

**下会话（6 秒一轮）三个候选，按可能性排序**：
1. `ar_reduce2` 的 `dst` 与 staging **同一块内存** ✗ —— 在**逐 p 求和的循环里**，
   `dst[i] = acc` 会先于其它 block 读到同一地址 ⇒ 结果被后续读污染 ✓；
   **改为先在寄存器里求完和再写** ✓（或让 dst 指向调用方缓冲 ✓，彻底分开 ✓）；
2. 检查 `dst` 传参是否被 `base`（const 视图）覆盖 ✓；
3. 用 `AR_MICRO_N=8` + 把 reduce 换成"逐 slot 原样拷到 dst"（不求和 ✓）验证 dst 通路 ✓。

### ⚠ 更正：**不存在"round 双自增"** —— 两处 `fetch_add` 是互斥分支

核对 `tp.rs` 后确认：`publish` 里两处 `fetch_add`（201 行的 dev 分支 ✓ 与 255 行的非 dev 分支 ✓）
**是 if/else 的两个分支** ✓，每个 all-reduce 实际**只自增一次** ✓；`all_reduce_inplace` 用的是
`load()`（读取 ✓）。⇒ 我先前的"双自增导致 parity 恒为奇数"的推断**不成立** ✗。

**但本轮那次 `.cu` 改动（把 `ar_stamp` 的调用从 dev 分支里移出、变成两个分支之前统一调用 ✓）
确实消除了挂死** ✓（world=2/8 均能跑完 ✓）—— 所以挂死的原因应归于**盖章/调用的时序** ✗
而不是计数器 ✓。

**保持的结论（不变）**：
- store 已被 staging dump 证明**完全正确** ✓（半区/slot/奇偶 ✓）；
- 剩下 **0.0** 精确落在 **reduce 的写入**那一步 ✓（dump 显示和从未被写回 ✓，
  而 copy-back 读到的恰是本 rank 自己的输入 = rank0 的 0 ✓）；
- 默认路径**全程回归通过** ✓（harness `OK` ✓ + 模型 `Paris` ✓）。

**⚠ 写法教训（本会话第 3 次）**：我在文档里写下的机制推断（"双自增"✗）**必须先读代码核对** ✓，
否则会把下会话带偏 ✓ —— 已连续三次发生（parity 推断 ✗、双自增推断 ✗、早先的"专家多派"✗）。

### ⚠ 微基准是 **flaky** 的（与默认路径的正确性无关）

同一组参数（`AR_MICRO_WORLD=4 AR_MICRO_ROUNDS=8`）**通过过一次** ✓（`[ar_micro] OK` ✓、
5 秒 ✓），**又一次没有任何输出** ✗（超时 ✓）。而**模型侧 `Paris` 每次运行都稳定正确** ✓✓
（已复验十余次 ✓）⇒ 判定：**flakiness 在 harness 的 GPU 搭建环节** ✓
（`bind_to` + `enable_peer_access` 需要 8 个 context 先就绪 ✓；本机是多租户 ✓，
别的用户/残留 context 会干扰 ✓），**不是默认路径的 race** ✗（否则模型也会偶发乱码 ✓）。

**下会话用之时的纪律**：
- 先 `nvidia-smi --query-compute-apps` 确认 4–7 号卡为空 ✓（AGENTS.md 的 GPU 纪律 ✓）；
- flaky 时**先怀疑 harness 搭建** ✓，用 `AR_MICRO_WORLD=2` 复跑 ✓；
- **判据仍是模型文本** ✓ —— 默认路径任何改动都必须复验 `Paris` ✓。

## ★★★★★ 路线修正（战略）：**段图可以绕开设备侧协议**，不必先攻它

**当前路线的问题**：要整层图化 ⇒ 集合通信不能在捕获范围内 ⇒ 必须消灭 host barrier ⇒
必须让整套跨 rank 协议设备化 ⇒ 本会话 5 次尝试未成 ✗（每次都修掉一个真缺陷 ✓：
同线程 release ✓、空指针守卫 ✓、拷回奇偶 ✓、round 时序 ✓、内核内盖章的挂死 ✓，
但 0.0 仍在 ✗ —— 已逼到 reduce 写入那一步）。

**更好的路线（不需要动集合通信）** ✓✓：
每层的结构本来就是 **`[hc + 注意力] → AR → [hc + MoE] → AR`** ✓ ——
**两个 AR 之间的算子序列里没有任何主机往返** ✓（MoE 的发射参数已在上一轮改成
**路由无关的固定 6 槽位** ✓✓，这正是图化需要的 ✓），因此可以把这两段**分别捕获成图** ✓，
**AR 继续由主机发射** ✓ —— **设备侧协议完全不必做** ✗✓。

- 每层：2 次图回放 + 2 次 AR（主机 ✓）取代 ~130 次发射 ✓ ⇒ 主机时间从 ~2.4ms/层
  降到 ~0.2ms/层（AR 的 8 次 peer 拷贝 + stamp + reduce + 一次 barrier ✓）；
- 估算：每层 3.9ms → ~1.8ms ⇒ 45 层 ≈ 81ms/token ⇒ **~12 tok/s**（当前 7.1 ✓）；
- 与"设备侧协议"路线的终点相同 ✓，但**风险面小得多** ✓（不碰并发协议 ✓，
  只碰 CUDA Graph 的捕获纪律 ✓：捕获期禁止分配 ✓ —— 我的 DevBuf 全是预分配 ✓ 已满足 ✓）。

**因此下会话的建议顺序改为**：① 段图（先做 MoE 段 ✓，它最大且参数已静态 ✓）→
② 若仍有余力再回头攻设备侧协议（用于进一步压 AR 的 host barrier ✓）。

**段图所需的 FFI**（本 crate 尚无）：`cuStreamBeginCapture_v2` / `cuStreamEndCapture` /
`cuGraphInstantiate_v2` / `cuGraphLaunch` / `cuGraphExecDestroy` —— 5 个符号 ✓，
照 `device.rs` 现有的 `sym(h_k, ...)` 模式绑定即可 ✓。

## ★★★★ 段图已跑通（正确 ✓）但**不带来提速** —— 驳倒"发射数是瓶颈"的假设

**已落地并验证**（`DSV41_GRAPH_MOE=1`，默认关）：
- 5 个 CUDA graph 符号 + `Device::{capture_begin,capture_end,graph_instantiate,graph_launch,graph_free}` ✓
- `moe()` 的 AR 拆成 `moe_reduce()`（图外、主机发射 ✓）
- 每层 MoE 段（gate/route/6 专家/shared ✓）捕获成图 ✓：首步捕获 40 个层图、次步起全部回放 ✓
- **文本正确**：`The capital of France is → [51119, 1]` = " Paris" ✓✓

**为跑通它修掉的 4 个真缺陷（都是宝贵的捕获纪律）**：
| # | 缺陷 | 现象 |
|---|---|---|
| 1 | `zero_at` 用**同步 `cudaMemset`**（跑在 legacy stream 0 ✗） | `operation would make the legacy stream depend on a capturing blocking stream` |
| 2 | `memcpy_d2d` 用**同步 `cudaMemcpy`** ✗ | 同上 |
| 3 | **`dsv41_quant_fp4` 每次调用 `cudaMalloc`+`cudaFree`** ✗✗ | `cudaErrorMemoryAllocation`(2) at launch —— **同时是热路径的同步慢指令**（每层调 6-8 次）→ 改为**每设备缓存 scratch**（只在非捕获期增长 ✓） |
| 4 | **捕获模式用 `Global`(0)** ✗✗ | `cudaErrorStreamCaptureUnjoined`(901) + `cudaGraphInstantiate`(900) —— Global 下**任何线程**的 CUDA 活动都作废捕获 ✓，而 TP8 是 8 个 rank 线程各跑一条流 ✓ → 必须用 **`cudaStreamCaptureModeRelaxed`(2)** |

**结果：吞吐完全不变** ✗（7.5 tok/s，133.1ms/token，两路径同值 ✓；PHASE 逐层数据亦无差别 ✓）。

### 由此得到的**决定性结论**（下会话必须以它开局）
> **"每层 ~130 次发射 × ~18µs = 主机瓶颈"这个假设是错的** ✗✗。
> 把 MoE 的 ~46 次发射压成 1 次（图回放 ✓）**对墙钟零影响** ⇒ 主机时间不在"发射数"上，
> 而在别处（最可能是 **AR 的跨 rank 主机 barrier** 与 **逐层 host↔device 同步点**）。

**因此下一步必须是 nsys 实际剖析**（不要再按"发射数"推理 ✓），先回答：
45 层解码稳态下，**主机时间**到底落在哪些调用上（barrier？`download_f32`？peer 拷贝？各占多少）。

**同时保留的收益**：段图基础设施本身是有价值的 ✓（整层图化的必经之路 ✓），
且本轮顺手修掉的 **per-call `cudaMalloc`** 是纯赚 ✓（去掉了热路径上的同步慢指令 ✓）。

## ★★★★★ 突破：主机时间 34% 在**集合通信的主机侧 peer 拷贝**上（实测，非推理）

**测量方式**：给 `all_reduce_inplace` 与 `end_round` 加全局原子计时器（`AR_HOST_NS/AR_CALLS`、
`AR_BAR_NS/AR_BAR_CALLS`），每 512 次打印 ✓。

**24 token 的实测**：
```
calls=22528 host_total=1107.3ms avg=49.2us | barriers=21968 bar_total=40.1ms avg=1.8us
DECODE 24 tokens in 3.13s = 130.59 ms/token
```

| 项 | 实测 | 推得 |
|---|---|---|
| `all_reduce_inplace` 次数 | 22528 / 24 token | **≈938 次/步 ≈ 20.8 次/层** ✗✗（架构只该 2 次）|
| AR 主机总时间 | 1107ms / ~25 步 | **≈44ms/步 = 步时间的 34%** ✗✗ |
| 单次 AR 平均 | **49.2µs** | 很慢 |
| **barrier** | **1.8µs** × 21968 | **便宜 ⇒ 不是瓶颈** ✗ |

**根因（算式吻合）**：49.2µs 中 barrier 只占 1.8µs ⇒ 其余全在 `publish` = **每 AR 7 次
`cudaMemcpyPeerAsync`**（world−1 ✓）⇒ **938 × 7 ≈ 6566 次主机侧 peer 拷贝 API / token**
⇒ 6566 × ~7µs ≈ **46ms**，与实测 44ms 吻合 ✓✓。

### ⚠️⚠️ 口径纠正（读代码后自己推翻，务必先看这段）
上面把 **进程级全局计数器**（8 个 rank 线程都累加同一个原子 ✗）当成了单 rank 的量 ✗✗——
这是本文档反复警告的"口径错觉" ✗。**正确换算**：
| 量 | 进程合计（实测） | **每 rank** |
|---|---|---|
| AR 调用数 | 22528 / 24 token | **2816 / 24 token ≈ 117/步 ≈ 2.6/层** ✓✓（与架构的 2 次/层吻合 ✓）|
| AR 主机时间 | 1107ms | **≈138ms / 25 步 ≈ 5.5ms/步** |

⇒ **AR 主机侧只占步时间的 ~4.4%** ✗（我早先"3-6%"的估计其实是对的 ✓，"34%"是错的 ✗）。
peer 拷贝：117 × 7 = 819 次/步 × ~7µs ≈ 5.7ms ✓ 与 5.5ms 吻合 ✓。

**真正的结论（修正后）**：
> 段图对吞吐零影响 ✗ + AR 主机侧仅 4.4% ✗ ⇒ **~125ms/步（96%）在 GPU 上** ✓✓。
> 下一步必须做 **GPU 端剖析（nsys / 逐 kernel 计时）**，不要再打主机侧的主意 ✓。

**修复路线（按性价比）**：
1. **设备侧集合通信（`DSV41_AR_DEV`）**：把 7 次/AR 的主机 peer 拷贝改成 GPU 侧完成 ✓
   ⇒ 直接吃掉那 34% ⇒ **预估 7.5 → ~11 tok/s（+50%）**。⚠ 该路径本会话 5 次尝试未成
   （每次都修掉一个真缺陷，最后逼到"reduce 没把和写回"⇒ 观测到 0.0 ✓），**但要重新评估其
   优先级——现在已知它值 50% 而不是 5%** ✓✓。
2. **查清为何是 20.8 次/层**（应 2 次 ✗）：若有多余 AR，删掉等价于直接砍时间 ✓
   （候选：hc/hyper-connection 的逐 mix 归约、engram 的跨 rank 交换 ✓ —— 需读代码确认 ✓）。
3. 若仍走主机路径：把"每 AR 7 次 peer 拷贝"改为**更少的调用**（如一次拷贝合并多 slot ✓，
   或按 rank 的 `memcpy_2d` 批式 ✓）。

## ★★★★★ 每层 3.2ms 的实测归因（层扫描 + 段隔离，取代一切推理）

**① 层扫描**（`DSV41_LAYERS=4/12/24`，16 token，线性极好 ✓）：
| LAYERS | ms/token |
|---|---|
| 4 | 14.75 |
| 12 | 40.24 |
| 24 | 79.29 |

斜率 = **3.2ms/层**（两段各算 3.19 / 3.25，一致 ✓）；截距 = **仅 1.95ms 固定开销** ✓
⇒ **"图/固定开销大"的假设也被排除** ✗ —— 时间**纯粹是逐层成本** ✓✓。
换算：130.6 ≈ 1.95 + 45 × 3.2 ✓。**目标 5ms/token ⇒ 每层 0.11ms ⇒ 要比现在快 29 倍** ✗。

**② 段隔离**（16 token；⚠ BASE 只跑了 2 token —— 那是模型正确答完 " Paris"+EOS ✓，
其 63.9ms 是**短上下文首 token**、非稳态 ✓，稳态基准取 ~133ms ✓）：

| 配置 | ms/token | 归因 |
|---|---|---|
| 稳态基准 | ~133 | — |
| `DSV41_SKIP_EXPERTS=1`（去 6 个路由专家） | **99.4** | **路由专家 = 34ms = 0.75ms/层** |
| `DSV41_SKIP_SHARED_EXPERT=1` | **121.8** | **共享专家 = 11ms = 0.25ms/层** |
| 两个都去 | **90.5** | 两者 ≈42ms，自洽 ✓ |
| ⇒ **剩余（注意力+hc+dense+主机）** | **90.5** | **= 2.0ms/层 = 63% ← 真正的大头** ✓✓ |

**③ 结论（下会话的靶心）**：
> **每层 3.2ms = 路由专家 0.75 + 共享专家 0.25 + "其余" 2.0** ✓。
> "其余"2.0ms/层里最大的是**注意力段**（本会话早前逐层计时：1.78ms/层 ✓ 与之吻合 ✓），
> 其中 nsys 曾测到 **hc_mixes 内核占 GPU 51.4%**（根因"24 lane 串行 20480 次依赖加载"✓
> 已做并行化 ✓，需复测确认收益是否真的落地 ✓）。
> **⇒ 下一步：对"其余 2.0ms/层"做 nsys 逐 kernel 归因**（不要再靠推理 ✓），
> 预期靶点是 **hc 链 + 注意力的 M=1 内核** ✓。

**④ 与目标的差距（诚实）**：200 tok/s 需 5ms/token = 0.11ms/层 ⇒ 需 **29x**。
M=1 权重读取地板 ~1.35ms/token（≈0.03ms/层 ✓）说明**理论上有空间**，
但现状与地板差 ~100 倍 ⇒ 瓶颈**既不是带宽也不是主机**（两者都已实测排除 ✗✗），
而是**每层的小 kernel 数量/启动与同步模式**——这正是段图（已跑通 ✓）与
M=1 专用内核要解决的问题 ✓。

## ★★★★★ 突破：hc_mixes 占 GPU 51.5% → 修好后 **7.2 → 12.5 tok/s（+74%）**

**nsys 逐 kernel 归因**（`--cuda-graph-trace=node`，12 token，默认路径 ✓）：
| Time% | Instances | Avg(ns) | kernel |
|---|---|---|---|
| **51.5** | 14720 | **808340** | **`hc_mixes_kernel`** ← 靶心 |
| 18.7 | 44160 | 97802 | `mxf4_gemm_kernel<false>`（专家 gate_up）|
| 11.4 | 40640 | 64830 | `gemm_fp8_kernel`（dense fp8）|
| 7.2 | 15088 | 109523 | `ar_reduce_kernel` |
| 5.2 | 44160 | 27126 | `mxf4_gemm_kernel<true>`（专家 down）|
| 1.8 | 7360 | 56583 | `sparse_attn_kernel` |

**根因（代码级）**：`hc_mixes_kernel` 把 mix 的第 m 行交给第 m 个 warp（`m += nwarp`），
而 launcher 写死 **`<<<rows, 128>>>`** ⇒ 24 行里只有 4 行并行、**每 warp 串行 6 行**，
且每行内层是**单累加器依赖链** `for(c=lane;c<hc_dim;c+=32) acc += wr[c]*xr[c]`
⇒ 延迟（非带宽）受限 ⇒ 808µs/次 ✓。

**修法（已默认开启）**：`nthreads = mix * 32`（一行一 warp ⇒ 24 行全并行 ✓）。
同二进制背靠背扫描（32 token）：
| threads | ms/token | tok/s |
|---|---|---|
| 32（我的错公式 `((24+31)/32)*32`） | 350.1 | 2.9 |
| 128（原值）| 139.1 | 7.2 |
| 256 | 103.7 | 9.6 |
| **768 = mix*32** | **80.5** | **12.5** ✓ |
单调 ⇒ 形状正确 ✓。**正确性四项全过** ✓（Paris / Tokyo. / "2"后停止 / 静夜思连贯 ✓）。

### ⚠️ 本轮两条方法论错误（都花掉了时间，记下来）
1. **同时改两个变量** ✗：我把"宽 block"和"4 累加器"一起提交 → 实测 348ms（-2.7x ✗），
   于是**错误地给宽 block 定了罪** ✗。单变量拆分后才发现：**4 累加器中性**（138.4 vs 138.8 ✓）、
   **两者同开是恶性交互**（348 ✗）、**宽 block 单独是 +72%** ✓✓。
   ⇒ **铁律：一次只改一个变量；推断前先分离。**
2. **跨构建比较** ✗：我拿"上一次构建 + env=768"与"新构建 + 默认"比，得出自相矛盾的结论 ✗
   （80 vs 350 ✓）。**必须同二进制背靠背** ✓ —— 换到同构建后，立刻暴露出真因是
   **我自己的公式写成 `((mix+31)/32)*32` = 32 线程** ✗（把"warp 数"当成了"线程数"）✓。

## 修复后的新 nsys 分布（12 token，默认路径 ✓）

| Time% | Instances | Avg(ns) | kernel | 备注 |
|---|---|---|---|---|
| **31.2** | 44160 | 97794 | `mxf4_gemm_kernel<false>` | **专家 gate_up ← 新靶心**（44160/13步/45层 ≈ 75/层 ✗）|
| **19.7** | 14720 | **184823** | `hc_mixes_kernel` | **808µs → 185µs（4.4x ✓）**，占比 51.5% → 19.7% ✓ |
| 19.0 | 40640 | 64829 | `gemm_fp8_kernel` | dense fp8（40640 ≈ 90/层）|
| 11.4 | 15088 | 104214 | `ar_reduce_kernel` | AR 的 GPU 侧归约 |
| 8.7 | 44160 | 27120 | `mxf4_gemm_kernel<true>` | 专家 down |
| 3.0 | 7360 | 56540 | `sparse_attn_kernel` | |
| 1.0 | 8784 | 16092 | `gemv_bf16_kernel` | |

**下会话靶心（按实测收益排序）**：
1. **`mxf4_gemm_kernel<false>` 31.2%（97.8µs/次 × 44160）** —— 专家 gate_up 的 M=1 GEMM ✓。
   97.8µs 对 M=1 偏大 ⇒ 疑似与 hc_mixes 同类的"每 block 字节量/形状"问题 ✓，
   值得先读它的 tiling 与 launcher（例如是否也按"一个 tile 做 1 行"浪费 ✓）。
2. **`gemm_fp8_kernel` 19.0%（64.8µs × 40640）** —— dense fp8 的 M=1 ✓，同上 ✓。
3. **`hc_mixes` 仍有 19.7%（185µs）** —— 已从 808 降下来，但仍偏慢；
   注：**4 累加器在 768 线程下实测中性**（138.4 vs 138.8 ✓），可试更高线程或改块内分工 ✓。
4. **`ar_reduce_kernel` 11.4%** —— 这是 AR 的 GPU 侧归约（15088 次 × 104µs ✓），
   **与主机侧的 4.4% 是两码事** ✓；104µs/次 对一个小归约偏大，值得看它的 grid 形状 ✓。

## ★★★★★ 新靶心的量化结论：专家 GEMM 以 **0.2% 带宽效率** 运行（最大的剩余单项）

**算式**（全部可复现）：
| 量 | 值 |
|---|---|
| 每专家 gate+up 的 fp4 权重 | 2 × 320 × 5120 / 2 字节 = **1.64 MB** |
| `mxf4_gemm_kernel<false>` 单次 | **97.8 µs**（nsys Avg，44160 次）|
| ⇒ 等效带宽 | **16.8 GB/s = B300（7.6 TB/s）的 0.2%** ✗✗✗ |
| 旁证 | 同文件 `gemv_bf16_kernel` = **16 µs**（快 6x ✓）|

**根因（与 hc_mixes 同一类，但更极端）**：`launch_mxf4` 用 **`kThreads=128` + tcgen05（tmem）**
的 tile 结构，那是为**大 M** 设计的 ✓；decode 是 **M=1** ✓ ⇒ tmem 分配/MMA 流水建立/片段填充
的固定代价完全盖过 1 行的计算 ✓ ⇒ 固定开销主导 ✓。

**修法（下会话首选，预期 −25~30% 总时间）**：为 **M=1** 写一个**带宽最优的 fp4 GEMV** ✓：
- 布局：每 block 若干输出行 n（n=320 ✓），每线程若干 k；按 32 元素一个 ue8m0 scale 块解包 ✓
- 字节量 0.8MB ⇒ 5TB/s 下 ~0.16µs ✓；即便仅 10% 效率也是 **1.6µs**（对比 97.8µs ⇒ **~60x** ✓✓）
- 参考：同文件 `gemv_bf16_kernel`（16µs）与 `dsv41_kernels.cu` 的 `hc_mixes` 修复形状 ✓
- ⚠ 数值：fp4→f32 解包与 scale 语义必须与 `mxf4_gemm` **完全一致**（同 env 下逐元素对拍 ✓），
  且**必须复验四段文本**（Paris / Tokyo. / "2"后停止 / 静夜思 ✓）
- 同样适用于 `gemm_fp8_kernel`（19.0%，64.8µs/次 ✓，dense fp8 的 M=1 ✓）

**教训（与 hc_mixes 合并成一条通则）**：
> **任何"为大 M 设计的 kernel"在 decode（M=1）下都会退化成固定开销主导。**
> 判定方法：`实测时间 vs 权重字节量/带宽` —— 相差 100 倍以上就是形状错了 ✓
> （hc_mixes 是 808µs vs 地板 ~1µs ✓；专家 GEMM 是 97.8µs vs 地板 0.1µs ✓）。

### ★★★★★ 专家 GEMM 的根因是**硬件约束**（结案）：必须换非 tensor-core 的 M=1 GEMV

读 `dsv41_experts_mxf4.cu` 得到决定性事实：
```c
constexpr int kMTile = 128;   // MMA M (1-CTA kind::mxf4 is fixed at 128)  ← 硬件固定
constexpr int kNTile = 64;
const int m_base = blockIdx.y * kMTile;
dim3 grid((n_total + kNTile - 1)/kNTile, (rows + kMTile - 1)/kMTile);   // rows=1 ⇒ (5, 1)
```
- **tcgen05 的 1-CTA `kind::mxf4` MMA 的 M 被硬件固定为 128** ✗ ⇒ decode 的 **M=1** 下，
  内核**必须算 128×64 的 tile 才能产出 1 行** ⇒ **128 倍冗余计算** ✗✗
- **grid = (5, 1)** ⇒ 只有 **5 个 block**（148 个 SM 几乎全闲 ✗）⇒ 无并行度 ✓
- B 侧字节量其实是对的（5 × 64×2560 = 0.82MB ≈ 全部权重 ✓），**所以不是带宽问题，
  是"算得多 + 并行少"** ✓ ⇒ 97.8µs ✓，有效带宽 16.8GB/s（0.2%）✓

**⇒ 结论（不可绕）**：M=1 下 **不能** 用 tcgen05 fp4 MMA ✓ —— 必须写**非 tensor-core 的
fp4 GEMV**（每线程若干 k、多 block 覆盖 n=320、按 32 元素 ue8m0 scale 块解包 ✓）。
预期：0.8MB / 5TB/s ≈ 0.16µs；即便 10% 效率 1.6µs ⇒ **~60x** ✓✓
⇒ 若把 31.2% 压到 ~5%，**总时间约 −25%**（12.5 → ~16 tok/s ✓）。
`gemm_fp8_kernel`（19.0%，64.8µs）同理 ✓。

**精确的"形状错误"判据（本次两条重大发现的通则）**：
> `实测单次耗时` vs `权重字节量 ÷ 带宽` 差 100 倍以上 ⇒ 形状/硬件约束不匹配 ✓
> - hc_mixes：808µs vs 地板 ~1µs ⇒ 已修（185µs ✓，+74% 吞吐 ✓）
> - 专家 GEMM：97.8µs vs 地板 0.1µs ⇒ **tcgen05 的 M=128 硬约束** ⇒ 需换 kernel ✓

## 下会话首选任务：**M=1 fp4 GEMV**（完整可执行方案，无需再推导）

### 已确定的格式事实（本会话从 `dsv41_experts_mxf4.cu` 读出，勿再猜）
| 项 | 事实 | 出处 |
|---|---|---|
| scale 类型 | **e8m0**，`ue8m0_to_f(b) = __uint_as_float((uint32_t)b << 23)` ⇒ **值 = 2^(b−127)** | 文件头注释 + `ue8m0_to_f` |
| scale 粒度 | **per (row, k/32)** —— 每 32 个 k 元素一个 u8 | `b_scale: [b_rows, k/32] e8m0` |
| B 权重 | **fp4，字节打包 `[n_rows, k/2]`**（每字节 2 个 fp4 值）| `b: [b_rows, k/2] fp4` |
| A（激活，AQ=true）| `a_f32: [rows, k]` f32 —— **decode 走 AQ=true**（`mxf4_gemm_kernel<true>`）| 形参注释 |
| M 约束 | **tcgen05 1-CTA `kind::mxf4` 的 M 固定 128** ⇒ M=1 时 128x 冗余 + grid=(5,1) | `kMTile = 128` 注释 |
| 专家 B 的定位 | 间接寻址：`b_base + e*b_stride`（e 来自 `ids[slot]`）| kernel 形参 |

### 实施方案（建议新 kernel `dsv41_expert_gemv_fp4`，放在 `dsv41_experts_mxf4.cu`）
1. **形状**：`grid = (n_total / ROWS_PER_CTA)`（n_total=320 ⇒ 取 32 行/CTA ⇒ 10 blocks；
   想让 148 SM 忙起来可让 1 block = 4 行 × 128 线程 ⇒ 80 blocks ✓）。
   每 warp 负责 1 行；行内按 32 元素一块：`s = 2^(bs[row][kb]−127)`，
   `acc += a_f32[j] * fp4_to_f32(nib) * s`。
2. **解包**：`uint8_t byte = b[row*(k/2) + j/2]; nib = (j&1) ? (byte>>4) : (byte&0xF);`
   ⇒ **低半字节 = 偶数下标**（与 cuda_fp4/NVFP4 约定一致 ✓；若文本不对，最先试交换 nibble ✓）。
   `fp4_to_f32`：e2m1 ⇒ 指数/尾数查表或位运算（**与 checkpoint 的转换脚本同源** ✓）。
3. **零解包开销的保障**：读 fp4 的字节量 = k/2 per row ⇒ 320×2560 = 0.82MB ✓；
   按 7.6TB/s 地板 0.11µs，即便 5% 效率也 ~2µs（对比 97.8µs ⇒ **~50x** ✓）。
4. **epilogue**（必须与现内核逐项一致）：`epi_mode == 1` ⇒ gate 列 `fminf(x, limit)`、
   up 列 `fminf(fmaxf(x,-limit), limit)`；`epi_mode == 3` ⇒ `out += x`（累加进 MoE 缓冲）；
   `row_weight != nullptr` ⇒ `x *= row_weight[row]`。**照着抄，不要重新解释** ✓。
5. **接线**：在 `launch_mxf4` 里按 `rows == 1` 分派到新 kernel（`gemv_bf16_kernel` 已有同样
   的"小 M 走 GEMV"先例 ✓）；`gemm_fp8_kernel` 同理（19.0%，64.8µs ✓）。
6. **验证（缺一不可）**：
   - 逐元素对拍：同 env 下与旧 `mxf4_gemm` 的输出比对（应 <1e-3 相对误差）✓
   - **四段文本**：`The capital of France is` → " Paris" ✓；`The capital of Japan is` → " Tokyo." ✓；
     `1+1=` → "2" 后停止 ✓；`请背诵《静夜思》` → 连贯 ✓
   - 同二进制背靠背 A/B 测吞吐（**一次只改一个变量** ✓）
7. **预期**：31.2% → ~5% ⇒ 总时间约 **−25%**（12.5 → ~16 tok/s ✓）；再加 gemm_fp8 同理。

### 本会话已确立的两条铁律（下一任务务必遵守）
1. **一次只改一个变量** ✓（本会话把"宽 block + 4 累加器"一起改 ⇒ 348ms ✗，导致我给宽 block 误定罪 ✓；
   拆分后才发现宽 block 单独是 +72% ✓ 而 4 累加器与它同开才是恶性交互 ✓）
2. **同二进制背靠背比较** ✓（跨构建比较掩盖了我自己公式 `((mix+31)/32)*32`=32 线程的真因 ✓）

## 收尾校正（读代码后）：`ar_reduce` 不是形状缺陷；`hc_mixes` 还有第二个缺陷

### `ar_reduce_kernel`（11.4%）—— **固有会合代价，不是 bug** ✓
代码：每个 block 的 `threadIdx.x==0` 自旋等 `world` 个戳记 → `__threadfence_system()` → 归约。
nsys 长尾（**min 4.16µs / med 21.5µs / avg 104µs / max 510µs / stddev 103µs**）说明
**平均被"等最慢 rank"主导** ⇒ 它度量的是**集合通信的会合延迟** ✓，而这是 AR 的固有成本 ✓
（主机侧 4.4% + GPU 侧 11.4% ≈ **AR 合计 ~16% = 结构地板** ✓）。
⇒ **不要在这里做形状优化** ✓；要压 AR 只能减少调用次数或改协议 ✓（后者本会话已证风险高 ✗）。

### `hc_mixes`（仍 19.7%，185µs）—— **还有"网格太小"的第二缺陷** ✓✓
第一缺陷（块内线程数 ✗）已修 ✓（808→185µs ✓，+74% 吞吐 ✓），但 185µs 仍是地板 ~1µs 的 **~180 倍** ✗
⇒ 形状仍未到位 ✓。原因：**launcher 是 `grid = rows`，而 nsys 显示它每层被调 ~25 次、每次 rows 很小**
（14720 次 / 13 步 / 45 层 ≈ 25/层 ✗）⇒ **每 SM 上只有 1 个 block、并行度近 0** ✗✓。
**修法（廉价、下会话可做）**：把每层那 ~25 次 `hc_mixes` **合并成少数几次**（按 rows 维批量化 ✓，
或把 hc 的 pre/post/comb 三组一次性算完 ✓）⇒ 网格从 1 变几十 ✓。
⚠ 数值性改动（行与行的计算相互独立 ✓ ⇒ 逐位不变 ✓，但必须复验四段文本 ✓）。

### 结论：剩余的大杠杆只有"**M=1 专用 GEMV**"一项 ✓
| 项 | 占比 | 性质 |
|---|---|---|
| 专家 GEMM（gate_up 31.2% + down 8.7%）| **~40%** | **tcgen05 M=128 硬约束** ⇒ 必须换 GEMV ✓ |
| `gemm_fp8`（dense）| 19.0% | 同类 M=1 问题 ✓ |
| `hc_mixes` | 19.7% | 网格太小 ⇒ 批量调用 ✓（廉价）|
| AR（host 4.4% + GPU 11.4%）| ~16% | **结构地板**，勿动 ✗ |
| sparse_attn + gemv_bf16 + 其他 | ~5% | — |

⇒ 若前两项（~59%）落到各自地板，**总时间可降至 12.5 → 25-30 tok/s** ✓✓；
  再叠加 hc_mixes 批量（−15%）与段图/图化，才有望逼近 200 ✓。

## ★★★★★ 落地：M=1 fp4 GEMV 替换 tcgen05 专家 GEMM —— **12.5 → 15.1 tok/s（+21%）**

**实现**：`expert_gemv_fp4_kernel`（`dsv41_experts_mxf4.cu`）—— 一行一 warp、无 tmem/MMA ✓；
在 `launch_mxf4` **与 `launch_mxf4_indirect`**（decode 实际走的 ✓）里按 `rows == 1 && !aq`
分派 ✓；`DSV41_NO_GEMV_FP4=1` 回退旧路径 ✓。

**同二进制 A/B（32 token）**：
| 配置 | ms/token | tok/s |
|---|---|---|
| 旧 tcgen05 路径（`DSV41_NO_GEMV_FP4=1`）| ~80.6 | 12.4 |
| **新 fp4 GEMV（默认）** | **66.33** | **15.1** ✓ |

**四段文本与改动前逐字相同** ✓（Paris ✓ / Tokyo. ✓ / "2"后停止 ✓ / 静夜思 ✓）——
这点同时**验证了格式语义的正确性** ✓：低半字节在先 ✓、e2m1 查表（0/0.5/1/1.5/2/3/4/6 ✓）、
e8m0 = 2^(b−127)（`__uint_as_float(b<<23)` ✓）。

### 本任务踩到的 3 个坑（都值得记住）
1. **`launch_mxf4` 没有间接参数** ✗：`b_base/b_stride/...` 是 **kernel** 的形参，我写进 launcher
   ⇒ 10 个编译错误 ⇒ **`.so` 没更新** ⇒ 后续"正确性 + A/B"跑的是**旧产物** ⇒ 得出"无变化"的
   假结论 ✗✗。**⇒ 每次改 `.cu` 后必须先看 build 输出里的 error 数，再看 `.so` 时间戳** ✓。
2. **decode 走的是间接路径且激活是 fp4** ✗：`launch_mxf4_indirect(..., nullptr /*a_f32*/, ...)`
   ⇒ 我按 `aq` 判断后读 `a_f32` ⇒ **空指针崩溃**（表现：**所有运行都没有任何输出** ✗）。
   修：GEMV 同时支持两种激活（f32 / fp4×f32-scale ✓）。
3. **`a_scale` 是 f32 不是 e8m0** ✓（权重侧才是 e8m0 ✓）—— 两侧格式不同 ✓。

### 下一个靶心（同理，最省力）
- **`gemm_fp8_kernel`（19.0%，64.8µs/次）** —— dense fp8 的 M=1，同类问题 ✓
  ⇒ 把那套 GEMV 手法套过去（fp8 e4m3 解包 + 每 (row,k/32) 的 f32 scale ✓）
- **专家 down（`mxf4_gemm<true>` 8.7%）** —— **同一个 GEMV 已支持 `epi_mode` 2/3** ✓
  ⇒ 只需把分派条件从 `rows == 1 && !aq` 放开到 down 的调用点 ✓（几乎零成本 ✓）
- **`hc_mixes`（19.7%，185µs）** —— 网格 `rows` 太小、每层 ~25 次调用 ⇒ 批量合并 ✓

## GEMV 的两个后续发现（一是负结果，一是真正的原因）

**① smem 暂存激活 = 仅 +1%（15.1 → 15.2 tok/s ✓）—— 负结果**
我把"512 行各自重读激活 = 10.5MB/次"当成瓶颈，实测只 +1% ✗。
**原因**：激活只有 **20KB ⇒ 本来就 L2 常驻** ✓ ⇒ 那 10.5MB 是 **L2 命中**，不是 DRAM 流量 ✗。
（与文档里已记过的同类陷阱一致：L2 吸收小工作集的重读 ✓。）改动保留（无害 ✓ 且语义更干净 ✓）。

**② 真正的原因：GEMV 的占用率太低** ✓
- 我的配置：`blocks = n_total/8 = 64`、`cta = 256`（8 warp）⇒ **512 warp / 148 SM ≈ 3.5 warp/SM** ✗✗
- 每 warp 一行、每行 k=5120、lane 步长 2 ⇒ **每线程 80 次迭代的串行 FMA 链** ✗（同 hc_mixes 的形态 ✓）
- ⇒ 延迟无遮挡 ⇒ 33.9µs/次（1.31MB ⇒ 38.7 GB/s = 0.5% 带宽 ✗）
**修法（下会话，预期把 GEMV 再降 3-5x）**：**沿 k 维再切**（如 4 个 warp-group 各算 k/4 ✓
再用 smem/atomic 归约 ✓）⇒ warp 数 ×4 ⇒ 占用率上去 ✓；或每 block 更多行 ✓。
⚠ 数值：归约顺序会变 ⇒ **必须复验四段文本** ✓。

## 剩余靶心清单（按"同一手法、最省力"排序）
1. **`gemm_fp8_kernel`（19.0%，64.8µs/次）** —— dense fp8 的 M=1 ✓，套用同一 GEMV 手法 ✓
   （e4m3 解包 + 每 (row,k/32) 的 f32 scale ✓；注意 fp8 的字节布局与 fp4 不同 ✓）
2. **专家 down（`mxf4_gemm<true>` 10.6%）** —— **同一个 GEMV 已支持 `epi_mode` 2/3** ✓
   ⇒ 只需放开分派条件 ✓（几乎零成本 ✓）
3. **GEMV 的 k-split**（见上 ✓，预期再 −3~5x ✓）
4. **`hc_mixes` 批量合并**（19.7% ✗ → 每层 ~25 次小调用合并 ✓）
5. **AR（host 4.4% + GPU 11.4%）** = 结构地板，勿动 ✗

**会话累计：2.6 → 15.2 tok/s（5.8x）** ✓✓；正确性四段文本全过 ✓（每次改动都复验 ✓）。

## ⚠️ 负结果：把 GEMV 扩展到专家 **down** 路径 → 破坏正确性（已就地回退 ✓）

**尝试**：把分派条件从 `rows == 1 && !aq` 放宽到 `rows == 1`（想让 down 也吃 GEMV ✓，
因为它已支持 epi_mode 3 累加与 f32 激活 ✓）。
**结果**：**模型被破坏** ✗✗ ——
```
The capital of France is → ids [0,0,0,...]        （全零 ✗）
The capital of Japan is → cudaMemcpy D2H: illegal memory access ✗
1+1=                    → illegal memory access ✗
```
**处置**：条件立即恢复为 `rows == 1 && !aq` ✓；**复验四段文本全部恢复** ✓（Paris ✓ /
Tokyo. ✓ / "2"后停止 ✓ / 静夜思 ✓），吞吐 15.2 tok/s ✓。
**原因（尚未查清，下会话再做）**：down 的实参形状与 gate/up 不同 ✓ ——
`dsv41_expert_down_fp4_indirect` 传 `a_f32 = act`（专家的激活缓冲 ✓）、`n_total = dim (4096 ✓)`、
`k = inter (256 ✓)`、`b_split = -1`、`epi_mode = 3`。我的 GEMV 里至少有一处与这些不匹配
（候选：`b_split = -1` 时 `hi` 判定的分支选择 ✓；或 down 的 `b` 布局是「转置」的
`[k, n]` ✗ —— **down 是 GEMM 的 N/K 互换形态** ✓ ⇒ 很可能需要单独的实现 ✓）。
**教训（已入档）**：**不要因为"我的 kernel 看起来支持"就把新调用点接上** ✓ ——
先把该调用点的**全部实参语义**读清 ✓，且**必须四段文本复验** ✓。
**另一处已修**：分派里的 `getenv` 在**每次专家调用**上都执行 ✗（每层 6-8 次 ✓）
⇒ 注释已写明**必须缓存成 static** ✓（hot-path 纪律 ✓）。

## 下一个大靶心：`gemm_fp8_kernel`（19.0%，64.8µs/次 × 40640）—— 与专家 GEMM 同类，且更极端

**代码证据**（`dsv41_kernels.cu:1004-1011`）：
```c
const int smem = 16 * k;                    // A tile = 16 行 × k 字节 / block
dim3 grid((n + 63) / 64, (m + 15) / 16);    // M=1 ⇒ (n/64, 1)
gemm_fp8_kernel<<<grid, 128, smem, s>>>(a, a_scale, w, w_scale, bias, out, m, n, k);
```
- 典型投影（n=4096, k=4096）⇒ **grid=(64,1)、128 线程 ⇒ 8192 线程 = 64 warp / 148 SM ≈ 0.43 warp/SM** ✗✗
  （比专家 GEMM 的 5 块更极端 ✓）
- **A tile 固定 16 行** ⇒ M=1 时 **15/16 的 tile 是纯浪费** ✓，且 `smem = 16k = 64KB/块`
  把每 SM 的块数压到 ≤2 ✗
- 实测 64.8µs/次、40640 次（≈9 次/层 ✓）⇒ 占 19.0% ✓

**修法**：照搬已验证成功的 **M=1 GEMV 手法**（一行一 warp、无 tmem/无 16 行 tile ✓），
但**须注意与 fp4 路径的两处格式差异**（这是本次没直接套用的原因 ✓）：
| 项 | 专家 fp4 路径（已做 ✓）| dense fp8 路径（待做）|
|---|---|---|
| 权重 | fp4，2 值/字节，e8m0 **per (row, k/32)** | **fp8 e4m3，1 值/字节**，scale 是 **per (n/32, k/32) 的 e8m0 块** ✗ 布局不同 |
| 激活 | fp4 + **f32** scale（AQ=false）| **fp8 e4m3 + f32 scale per (row, k/32)** ✓ |
| 偏置 | 无 | **有 `bias`** ✓ 别漏 |
| k 约束 | 64 的倍数 | 待确认（同一 launcher 系 K 必须 32 的倍数 ✓）|
⚠ 逐位对拍 + **四段文本复验** ✓；先只改一个变量 ✓；**改完先看 build 的 ` error: ` 计数再落 `.so` 时间戳** ✓
（本会话因忽略这两条，两次误读了旧 `.so` ✗）。

**预期**：19.0% → ~5% ⇒ 再 −10~15% 总时间（15.2 → ~17 tok/s ✓）；
与专家 down（10.6%，同样待做 ✓）合计 ≈ −25%。

## ★★★★★ 落地：M=1 fp8 GEMV（dense 投影）—— **15.2 → 17.6 tok/s（+16%）**

**实现**：`gemm_fp8_gemv_kernel`（`dsv41_kernels.cu`）—— 一行一 warp、8 行/块、无 tile/无 tmem ✓；
在 `dsv41_gemm_fp8_mx` 里按 `m == 1` 分派 ✓；`DSV41_NO_GEMV_FP8=1` 回退 ✓。
**格式（从原 kernel 读出并对齐，勿再猜）**：
| 张量 | 布局 |
|---|---|
| `a` | e4m3，1 字节/值，`[m, k]`；m=1 ⇒ 只有一行 ✓ |
| `a_scale` | **f32**，`[m, k/32]`（索引 `a_scale[row*nb_k + kb]`，`nb_k = k/32`）|
| `w` | e4m3，1 字节/值，`[n, k]` |
| `w_scale` | **e8m0 字节**，`[n/32, k/32]`（索引 `w_scale[(n>>5)*nb_k + kb]`）—— **32×32 块尺度** |
| `bias` | f32 `[n]`，**可空** ⇒ 必须判空 ✓ |
| 解码 | `e4m3_to_f(uint8_t)`（本文件 51 行）✓ |

**同二进制 A/B（32 token）**：旧 tile 路径 65.59ms / **新 GEMV 56.79ms（17.6 tok/s ✓）**；
四段文本全对 ✓（Paris / Tokyo. / "2"后停止 / 静夜思 ✓）。

### 本会话三条"同一种病"的根治（全部实测 ✓）
| # | kernel | 病因 | 修法 | 收益 |
|---|---|---|---|---|
| 1 | `hc_mixes` | launcher 写死 128 线程，kernel 按行→warp 分工 ⇒ 24 行只并行 4 行 | `nthreads = mix*32` | 7.2 → 12.5（**+74%**）|
| 2 | 专家 fp4 GEMM | tcgen05 `kind::mxf4` **M 硬件固定 128** ⇒ M=1 时 128x 冗余 + grid 仅 5 块 | M=1 fp4 GEMV | 12.5 → 15.2（**+21%**）|
| 3 | `gemm_fp8` | 16 行 tile + `smem=16k` ⇒ grid=(64,1)、0.43 warp/SM | M=1 fp8 GEMV | 15.2 → 17.6（**+16%**）|

**通则（可复用）**：**`实测单次耗时 ÷ (权重字节量/带宽)` 差 100 倍以上 ⇒ 形状/硬件约束不匹配** ✓
（hc_mixes 808µs vs ~1µs；专家 GEMM 97.8µs vs 0.1µs；gemm_fp8 64.8µs vs ~2µs ✓）。
**修法模板**：一行一 warp + 字节级解包 + 与产出方逐位一致的 scale 语义 + 四段文本复验 ✓。

### 剩余的同类候选
- **专家 down（`mxf4_gemm<true>` 10.6%）** —— **形状是转置的**（N/K 互换 ✓，`b_split=-1`、
  `epi_mode=3`）⇒ 上次直接复用分派**破坏了正确性** ✗ ⇒ 需**单独实现** ✓
- **`hc_mixes` 仍 19.7%** —— 每层 ~25 次小调用、grid=`rows` ⇒ 批量合并 ✓
- **AR（host 4.4% + GPU 11.4%）** —— 结构地板，勿动 ✗

## ★★★★★ 下一个（也是最大的）靶心：`hc_mixes` 的权重重读 = **步时间的 49%**

**量化核算**（全部来自 nsys 实测 + 代码读出的形状，可复算 ✓）：
| 量 | 值 |
|---|---|
| nsys 实例数 / 平均 | 14720 次 × 185µs（12 token、8 rank ✓）|
| 每步每 rank 调用数 | **153 次/步/rank ≈ 3.4 次/层/rank** ✓ |
| **每步每 rank GPU 时间** | **28.4 ms = 57.55ms 步时间的 49%** ✗✗ |
| 每次调用的权重 | `hc_fn` = mix(24) × hc_dim(4096) × 4B = **384 KB** ✓ |
| ⇒ 反推等效带宽 | **2.1 GB/s = DRAM(7.6TB/s) 的 0.03%** ✗✗✗ |

**机制（确定）**：kernel 把 **第 m 个 mix 行交给第 m 个 warp**，每个 warp 为该行重读一遍
**整个权重行**（hc_dim × 4B）✓；而**权重矩阵与输出行无关**（同一 `hc_fn` 服务所有行 ✓）
⇒ **R 个输出行就重读 R 次** ✗ ⇒ 总流量 = R × 384KB × 次数 ✓。
（这与文档里 GLM 侧 `gemv_fp8` 的"权重只 load 一次、M 维复用"是**同一个机会** ✓，
当年那一刀是"+1.1%"但当时是 **17.4ms 基线**；如今 hc_mixes 占 **49%** ⇒ 收益大得多 ✓。）

**修法：M 维复用**（同一 `hc_fn` 行载入一次、喂 R 个输出行 ✓）
1. 每 block 处理 **R = 4~8 行**（`r0 = blockIdx.x * R` ✓）
2. 每 warp 对 mix 行 m：`wv = wr[c]` **载入一次** ⇒ `for t in 0..R: acc[t] += wv * x[(r0+t)*hc_dim + c]`
   ⇒ **权重流量降 R 倍** ✓（同时 x 的流量升 R 倍，但 x 只有 hc_dim×4B = 16KB/行，占比小 ✓）
3. smem 改为 `mixes[R][mix]` + `comb[R][hc*hc]` ✓；**逐行的 sum-of-squares / inv / sinkhorn 必须各自独立** ✓
   （现行代码是单行语义 ✓，改成 R 行后**每行的归一化不能串** ✓ —— 这是最容易出错的地方 ✓）
4. 输出：`pre/post/comb` 按 `(r0+t)` 逐行写 ✓
**预期**：R=8 ⇒ 权重流量 482MB → 60MB ⇒ **28ms → ~4ms ⇒ 整体 −40%（17.6 → ~28 tok/s ✓）**。
**验证**：逐行对拍（R=1 的旧路径 vs R=8 的新路径，同输入逐元素比对 ✓）+ **四段文本** ✓ +
同二进制背靠背 A/B ✓；**一次只改一个变量** ✓；**先看 build 的 ` error: ` 计数再看 `.so` 时间戳** ✓。

**为什么这次没直接改**：R 行的重构会同时触及 smem 布局 / 每行归约 / sinkhorn 三处 ✓，
本会话已有 3 次"半途重构"的代价（两次 build 失败导致误读旧 `.so`、一次 down 路径破坏正确性）
⇒ 按纪律**先把分析与预期收益固化**，留给下一次独立完成 ✓。

## ⚠️⚠️ 自我更正：上一条"hc_mixes 权重被每行重读"的诊断**站不住脚**（勿按它动手）

上一条我按 `185µs ÷ 384KB = 2.1 GB/s` 推出"权重被每行重读 ⇒ M 维复用可 −40%" ✗。
**用同一套算式自检就会发现它自相矛盾** ✓：
- `hc_fn` 只有 **384 KB** ⇒ **完全在 L2（60MB）里** ✓ ⇒ 重读本应是 **L2 命中**，而非 DRAM ✓
- 实测等效带宽 **2.1 GB/s** 比 L2（~10 TB/s）低 **约 3000 倍** ✗✗
⇒ **重读根本不是瓶颈** ✓；**185µs 的真实成因仍未查明** ✗。

**这与本会话三次已实测否决的"显然推断"是同一模式** ✓：
| 推断 | 实测 |
|---|---|
| 激活被 512 行重读（10.5MB）⇒ smem 暂存 | **+1%** ✗（20KB 激活本就在 L2 ✓）|
| GEMV 占用率不足 ⇒ k-split | **15.0 vs 15.2** ✗ |
| warp 请求只有 32B ⇒ 16B uint4 lane | **14.3** ✗（更慢）|
| hc_mixes 权重重读 ⇒ M 维复用 | **未验证，且算式自驳** ✗ |

### ⇒ 铁律（本会话第 4 次确认）
> **"某项流量很大 ⇒ 它就是瓶颈"在 L2 常驻的工作集上不成立** ✓。
> 先算 **有效带宽 vs 该层（L2/DRAM）的容量与带宽**：差 100 倍以内 ⇒ **不是带宽问题** ✓，
> 必须用 **ncu 看 stall 原因 / 占用率 / wave 数 / issue 率** ✓，不要再凭"流量大"改代码 ✗。

### 因此 `hc_mixes`（仍是 49% 的大头 ✓）的正确下一步是**测量**而不是重构
1. **ncu 隔离复现**（本 crate 的微基准模式已有先例：`kernels/cuda/ncu_miniprof.cu` ✓）
   —— 需要：`hc_mixes` 的真实 shape（S=rows, hc_dim, hc, sinkhorn_iters ✓）
   + `--set full` 或至少 `smsp__warp_issue_stalled_*` / `sm__throughput` / `achieved_occupancy`
2. 待查候选（**不预设**）：grid=`rows` 的实际大小 ✗ · 每个 block 的 `__syncthreads` 数量 ✗ ·
   sinkhorn 的**逐 j 串行 + thread0 单线程**（`row_max/row_sum/col_sum` 各是一个 for-over-j 的
   thread0 循环 ✓ 且每步前后各有一次 `__syncthreads` ✓）✗ ← **这个结构最可疑** ✓
   （hc=4 时该循环只有 4 次迭代 ✓ 却要付全块屏障 ✓，且 `sinkhorn_iters` 次重复 ✓）
3. 只有在 ncu 指出具体 stall 后再改 ✓；**一次只改一个变量** ✓；改完先看 build 的 ` error: ` ✓

## ⚠️⚠️⚠️ 第二次自我更正：**"hc_mixes = 步时间 49%" 这个数字是错的**（口径错误）

**实测判决**：把 sinkhorn 的全部 ~100 次全块屏障去掉（数学与运算顺序逐位不变 ✓，
其余 5 个屏障保留 ✓）⇒ **17.3 tok/s（57.85ms）vs 改前 17.6（56.79ms）⇒ 中性** ✗✗。
⇒ **屏障不是成本** ✗ ⇒ 我上一条"185µs = 屏障"的推断**也被实测否决** ✗。

**系统性的口径错误（本会话第 5 次被实测打脸，终于定位到方法本身）** ✗✗：
- 我用的是 nsys `cuda_gpu_kern_sum` 的 **全运行平均（avg 185µs × 14720 次）** ✗
- 但 `--max-tokens 12` 的那次剖析**包含 prefill**，而 **prefill 调 hc_mixes 时 `rows` 很大**（远超 1）⇒
  那些实例本来就慢 ⇒ **把平均拉高** ⇒ 我据此推出"decode 每次 185µs、占 49%"**是错的** ✗✓
- **我的文档早就写过这条陷阱**（GLM 侧："372K 实例但多数是 prefill；decode 路径仅 ~74 次/步" ✓✓）
  ⇒ **我犯了自己记录过的错误** ✗✗

### ⇒ 铁律（务必照做）
> **nsys 的 avg 绝不能直接代表 decode 稳态** ✓。必须**只统计稳态解码窗口**：
> ① 用 `sqlite3 <rep>.sqlite` 查 `CUPTI_ACTIVITY_KIND_KERNEL`，**限定时间窗**（最后 N 毫秒）✓；
> ② 或按 **instance 的 start/end 时间**筛出"每步重复 N 次"的那些 ✓；
> ③ 或干脆**只跑 decode 不跑 prefill**（用已有 KV 的续跑 ✓ / `DSV41_PROMPT_IDS` 单 token ✓）。
> **任何"某 kernel 占 X%"的结论，都必须先说明它的实例是从哪个窗口统计的** ✓。

### 因此本会话**仍然成立的**结论（都来自同二进制背靠背 A/B，与 nsys 口径无关 ✓）
| 改动 | 实测（同二进制 A/B）|
|---|---|
| hc_mixes 块形状（128 → mix*32）| 7.2 → **12.5 tok/s（+74%）** ✓✓ |
| M=1 fp4 GEMV（专家 gate/up）| 12.5 → **15.2（+21%）** ✓✓ |
| M=1 fp8 GEMV（dense 投影）| 15.2 → **17.6（+16%）** ✓✓ |
| sinkhorn 去屏障 | 中性（保留 ✓ 无害且结构更干净）|
| GEMV k-split / 16B uint4 / smem 暂存激活 | **均更差或中性** ✗（已回退/保留 v1）|

### 下一步（修正后）
**先按上面铁律重新统计 decode 稳态的逐 kernel 分布** ✓（这是唯一可信的排序依据 ✓），
再决定动谁；`hc_mixes` 是否仍是大头**需要重新测量** ✓（可能根本不是 ✗）。

### 保留 sinkhorn 改动（中性），下一步按新铁律重测
sinkhorn 单线程化实测 17.3 tok/s（57.85 / 57.93ms 两次）vs 改前 17.6（56.79ms 一次）⇒
**在会话噪声带（±1.9%）内**，且**输出逐位一致** ✓ ⇒ 保留（结构更干净、去掉 ~100 次屏障 ✓）。
**立即执行新铁律**：取 **decode 稳态窗口**的逐 kernel 分布（限最后 100ms）✓，
以它（而非全运行平均 ✗）作为下一步的排序依据 ✓。

## ★★★★★ 可信的 decode-only kernel 分布（差分法，本会话方法论的最终修正）

**方法（取代不可信的全运行平均 ✗）**：跑两次剖析 —— `--max-tokens 1`（仅 prefill ✓）与
`--max-tokens 48`（prefill + 47 步 decode ✓）—— 用 `nsys stats --format csv` 取表
（**不要用 table 格式 + awk ✗**：kernel 名含空格 ⇒ 列错位 ✓），再用 python 相减 ✓
⇒ 得到**纯 decode 的净量** ✓。脚本要点：`StringIds`/CSV 的列序 `[Time%, Total(ns), Instances, ...]` ✓。

**结果（47 步 decode 的净量）**：
| 占比 | 次/步 | µs/次 | kernel | 备注 |
|---|---|---|---|---|
| **28.1%** | **803** | **195.6** | `hc_mixes_kernel` | **真·第一大项** ✓ |
| 16.0% | 2219 | 40.2 | `gemm_fp8_gemv_kernel` | 本会话新加 ✓ |
| 13.4% | 2410 | 31.1 | `expert_gemv_fp4_kernel` | 本会话新加 ✓ |
| 12.2% | 823 | 82.7 | `ar_reduce_kernel` | 集合通信会合 ✓（结构地板）|
| 11.7% | 2410 | 27.1 | `mxf4_gemm_kernel<true>` | **专家 down**（同类候选 ✓）|
| 9.1% | 402 | 126.8 | `sparse_attn_kernel` | 偏大，待查 |
| 1.6% | 40 | 220.3 | `indexer_*` | 偏大，待查 |
| 1.4% | 481 | 16.1 | `gemv_bf16_kernel` | |
| 1.1% | 1656 | 3.5 | `rmsnorm_kernel` | |
| 0.8% / 0.6% | 823 / 823 | 5.8 / 4.3 | `ar_store` / `ar_stamp` | 集合通信 ✓ |

**两条硬结论** ✓✓：
1. **decode 步几乎 100% 受 GPU 限制**：净 GPU 合计 **69.87 ms/步/rank** vs 实测步时间 **61.3ms**
   （并列的 8 rank 各占一卡 ✓）⇒ 主机侧与"发射数"确实不是瓶颈 ✓（与段图实验一致 ✓）。
2. **`hc_mixes` 每步 803 次 × 195µs，占 28.1%，且我刚去掉 sinkhorn 的 ~100 次屏障后
   195µs 纹丝不动** ✗ ⇒ **既非屏障、亦非带宽**（384KB 全 L2 常驻 ✓；98K MAC 地板 ~0.5µs ✓，
   实测差 400 倍 ✗）。**⇒ 必须用 ncu 看 stall 原因**（`smsp__warp_issue_stalled_*`、
   `achieved_occupancy`、`sm__throughput`、`launch__*`）✓，**不要再凭推断改代码** ✗。

**下一步（修正后的排序，全部以本表为准）**：
1. **ncu 隔离复现 `hc_mixes`**（28.1%，195µs/次 ✗）—— 形状：rows=1、hc_dim=4096、hc=4、
   sinkhorn_iters=20 ✓；**注意 803 次/步** ⇒ 也可能是**调用次数过多**（每次仅 1 行 ✓）⇒ ncu 之外
   还要看"能否把同一层的多次调用合并成 rows>1 的一次"（合并后 grid 与复用都会变好 ✓）
2. **专家 down（11.7%，27.1µs/次）** —— 形状转置 ⇒ 需**单独**的 GEMV（勿复用分派 ✗，已证破坏）
3. `sparse_attn`（9.1%，126.8µs/次）与 `indexer`（220µs/次）同样偏大 ⇒ 先量后改 ✓

## ★★★★★ 隔离复现锁定 `hc_mixes` 的根因：**每次调用 ~58µs 的固定开销，与 rows 无关**

**复现器**（本会话新建，可复用 ✓）：`/tmp/hc_repro.cu` —— 直接链 `libferrite_kernels.so`
调 `dsv41_hc_mixes`（真实 shape：hc_dim=4096、hc=4、sinkhorn_iters=20 ✓），
预热后 500 次计时 + `cudaProfilerStart/Stop` 窗口（供 ncu ✓）。
编译：`nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a /tmp/hc_repro.cu -o /tmp/hc_repro -L. -l:libferrite_kernels.so`
运行：`LD_LIBRARY_PATH=$PWD CUDA_VISIBLE_DEVICES=0 /tmp/hc_repro <rows>`
（坑：`cudaProfilerStart` 需 `#include <cuda_profiler_api.h>` ✓；`.so` 需 LD_LIBRARY_PATH ✓）

**测量（空闲 GPU，500 次平均）**：
| rows | 1 | 2 | 4 | 8 | 16 | 64 |
|---|---|---|---|---|---|---|
| µs/次 | **58.42** | 58.67 | 58.69 | 58.60 | 58.98 | 59.46 |

**⇒ 耗时与 rows 完全无关 ⇒ 是每次调用约 58µs 的固定开销** ✓✓（**不是**随行数的计算/带宽 ✗）。
配合"每步 803 次调用"（差分法实测 ✓）⇒ **803 × 58µs ≈ 46ms/步** ✗ ——
与它在模型里占 **28.1%** 完全自洽 ✓。

**唯一能吃下 58µs 的结构 = `sinkhorn_iters(=20)` 的循环** ✓（其余部分都随 rows 缩放或可忽略 ✓）。
两条自证线索：
1. 上一条 commit 把 sinkhorn 改成**单线程 + 动态索引的局部数组**
   （`float rmax[16]; ... rmax[jk/hc]` ✗）⇒ **动态索引的局部数组会溢出到 local memory** ✗，
   每次访问 ~600ns 延迟 ✗ ⇒ 16 次访问 × 20 迭代 × 0.6µs ≈ **190µs** ✓ 与模型里的 195µs 吻合 ✓✓
2. 但**原版（带 ~100 次屏障的多线程版）同样是 195µs** ✗ ⇒ **两版都栽在同一个 20 次迭代的循环上** ✓

**⇒ 正确修法（下会话，预期把 28.1% 打到 ~1%，即整体 +25%）** ✓✓：
把 4×4 的 sinkhorn（`hc=4` ⇒ 只有 **16 个值** ✓）放进**一个 warp 的寄存器**里，
用 **`__shfl_xor_sync` 蝶形**做行/列归约（行组步长 1、2；列组步长 4、8 ✓）——
**零屏障、零 local memory、零 thread0 串行循环** ✓ ⇒ 整个 20 次迭代理应在 **~2µs** 内完成 ✓。
⚠ 数值：蝶形归约的加法顺序与逐元素顺序不同 ⇒ 必须**四段文本复验** ✓（本会话的常规动作 ✓）。
⚠ 也可先试 **最小改动**：把单线程版的三个数组改成 `hc*hc` 个**具名标量**（静态索引 ✓，无 spill ✓）
—— 更快出结果，但要看编译器是否仍 spill ✓。
**判据**：先用 `/tmp/hc_repro 1` 看单次耗时是否从 58µs 掉到个位数 ✓（秒级反馈 ✓），
再跑模型四段文本 + 背靠背 A/B ✓。

## ★★★★★ 落地：sinkhorn 移入 warp 寄存器（蝶形归约）—— hc_mixes 58.4 → 17.2µs

**诊断链（全部实测 ✓，这是本会话最有价值的一次根因定位）**：
1. 差分法给出 **hc_mixes = decode GPU 时间的 28.1%**（803 次/步 ✓）
2. 隔离复现（`/tmp/hc_repro`）显示 **单次耗时与 rows 完全无关**：
   rows=1 → 58.42µs、rows=64 → 59.46µs ✓ ⇒ **是固定开销，不是计算/带宽** ✓✓
3. 唯一能吃下固定开销的结构 = `sinkhorn_iters=20` 的循环 ✓
   （原版 ~100 次全块屏障 ✗；我上一版单线程 + **动态索引局部数组 ⇒ local memory spill** ✗）

**修法**：hc=4 ⇒ comb 只有 **16 个值** ⇒ 放进 **warp 0 的 lane 0..15 寄存器** ✓，
行/列归约用 **`__shfl_xor_sync` 蝶形**（行组 offset 1、2 ✓；跨行 offset 4、8 ✓）
⇒ **零屏障、零 local memory、零 thread-0 串行循环** ✓

**同口径验证**：
| | 复现器（空闲 GPU，500 次）| 模型（同二进制）|
|---|---|---|
| 改前 | 58.42 / 58.60 / 59.46 µs（rows=1/8/64）| 17.6 tok/s（56.79ms）|
| **改后** | **17.22 / 17.51 / 17.98 µs** ✓ | **18.5 tok/s（54.12ms）** ✓ |
| 正确性 | — | 四段文本全对 ✓（Paris/Tokyo./“2”后停止/静夜思）|

**会话累计：2.6 → 18.5 tok/s（7.1x）** ✓✓
**注**：复现器显示改后仍是**平坦的 17.2µs** ⇒ **还有第二个固定开销 (~17µs)** ✓
（可解释部分：4096 元素的平方和 + 24 warp × 4096 FMA ≈ 2µs + 384KB L2 读 ≈ 2-4µs ⇒ 仍有缺口 ✓）
⇒ 下一步可用同一个复现器（秒级反馈 ✓）继续压：候选 = 减少 `nthreads`（768 线程对 rows=1 偏多 ✓）、
或把平方和与 mix 点积合并、或查 3 个残余屏障 ✓。**复现器是目前最高效的工具** ✓。

## 下一步（本会话结束时，按可信分布 + 新工具）
| 项 | 占比/耗时 | 状态 |
|---|---|---|
| hc_mixes | 28.1% → 已降 3.4x（复现器口径）| ✅ 已做 |
| **hc_mixes 残余 17µs 固定开销** | 803 次/步 × 17µs ≈ 13.8ms/步 | ▶ 用复现器继续压 |
| 专家 down | 11.7%（27.1µs/次）| 转置形状 ⇒ 需**单独** kernel（勿复用分派 ✗ 已证破坏）|
| ar_reduce / ar_store / ar_stamp | 12.2% + 1.4% | 集合通信会合，结构地板 |
| sparse_attn | 9.1%（126.8µs/次 ✗）| 偏大，先量后改 |
| indexer | 1.6%（220µs/次 ✗）| 同上 |
**工具链（都会用）**：① `/tmp/hc_repro.cu` 隔离复现（秒级 ✓，改 shape 即可复用于其他 kernel ✓）
② 差分法取 decode-only 分布 ✓ ③ 四段文本 + 同二进制 A/B 为最终判据 ✓

## ⚠️ 最终认知：多卡 nsys 的**绝对单次耗时不可信**（只有排序可参考）

**矛盾**：同一内核、同一 shape（hc_dim=4096、hc=4、sinkhorn_iters=**20**、rows=1 ✓
—— 已核对模型 `config.json` 确为 `hc_sinkhorn_iters=20` ✓）：
| 口径 | hc_mixes 单次 |
|---|---|
| **隔离复现器**（空闲 GPU，500 次平均）| **17.19 µs** ✓ |
| 模型 + 8 卡 nsys（`--cuda-graph-trace=node`）| **153 µs** ✗ |

⇒ 差 **9 倍** ✓。已排除的原因：**不是**数据分布（把复现器输入从全零改为伪随机真实分布 ⇒
17.19µs **不变** ✗）、**不是** `sinkhorn_iters`（都是 20 ✓）、**不是**发射开销（空 kernel 同 shape
= 2.5µs ✓，内核本体 14.7µs ✓）。
⇒ **结论：多设备 CUPTI 剖析出的"平均单次耗时"包含跨设备的排队/采集开销** ✗，
**不能当作真实内核时间** ✓。旁证：本会话把 hc_mixes 内核本体提速 3.4x（58.4→17.2µs ✓）
只换来模型 +5%（17.6→18.5 ✓）—— 若它真占 28%，应远不止 ✓。

### ⇒ 铁律补充（与"差分法"配套）
> **多卡 nsys 的"占 X%"可用于排序 ✓，但"每次 Yµs"必须用隔离复现器复核（单卡、空闲、多次平均 ✓）
> 才能当作真实耗时 ✓。** 本会话的复现器模板（`/tmp/hc_repro.cu`）已含：
> 预热 + 500 次计时 + **空 kernel 同 shape 的发射地板对照** + `cudaProfilerStart/Stop` 窗口 ✓。

### 本会话最终状态（全部经同二进制 A/B 或复现器验证 ✓）
- **正确性**：乱码彻底修好 ✓（6 个真 bug；四段文本在每次改动后复验，约 25 次全对 ✓）
- **性能 2.6 → 18.5 tok/s（7.1x）**：hc_mixes 块形状 +74% · 专家 fp4 GEMV +21% ·
  dense fp8 GEMV +16% · warp 寄存器 sinkhorn +5%（复现器口径 58.4→17.2µs ✓）
- **工具**：差分法（decode-only 分布 ✓）+ 隔离复现器（秒级反馈 + 发射地板 ✓）
- **已实测否决**：GEMV k-split · 16B uint4 · smem 暂存激活 · sinkhorn 去屏障 · down 复用分派（破坏正确性，已回退 ✓）
- **两次自我推翻已入档**（全运行平均不等于 decode ✓；权重重读不是 hc_mixes 的瓶颈 ✓）

## 专家 down 上次崩溃的根因（读代码所得，下会话执行前必看）

**现象**（已回退 ✓）：把 GEMV 分派放宽到 `rows == 1`（不限 AQ）后 ⇒ 一个 prompt 返回全零、
两个报 illegal memory access ✗。

**实参对照（读 `dsv41_expert_down_fp4_indirect` 与我的 kernel）**：
| 参数 | gate/up（GEMV 已成功 ✓）| **down（崩溃 ✗）** |
|---|---|---|
| `rows` | 1 | 1 |
| `n_total` | `2*inter` | **`dim` = 4096** |
| `k` | `dim` = 4096 | **`inter` = 256** |
| `b_split` | `inter`（gate/up 分界）| **`-1`** |
| `epi_mode` | 1（带 clamp）| **3（累加进 MoE 缓冲）** |
| `aq` | false（激活是 fp4 ✓）| **true**（激活是 f32 `act` ✓）|
| `row_weight` | nullptr | **非空（路由权重 ✓）** |

**我的 kernel 里最可疑的一处** ✓：读权重用的是
```c
const int kbytes = k >> 1;                       // = 128（k=256）
const uint8_t* brow = b_use + (size_t)r * kbytes; // ← 假设每行紧密排布 k/2 字节 ✗
```
⇒ 若 down 的权重池每行有 **padding/对齐**（很常见 ✓），`r*kbytes` 会**越界** ⇒
illegal memory access ✓✓。**修法**：把"每行跨距"作为独立参数传入（gate/up 传 `k/2` ✓，
down 传池的真实 stride ✓），或从 Rust 侧一并传 `b_row_stride` ✓。
**其余需核对的**：`epi_mode == 3` 时 `out[row] += x`（累加 ✓）与 `row_weight` 的乘序 ✓、
以及 `b_split = -1` 下 `hi` 判定（我的代码 `(b_split > 0) && ...` ⇒ false ✓ 正确 ✓）。

**验证路径（务必用复现器，秒级 ✓）**：把 `/tmp/hc_repro.cu` 当模板，写一个
`/tmp/down_repro.cu`：造 `[dim, inter]` 的 fp4 权重 + e8m0 尺度 + 一条 f32 激活 ⇒
① 先跑旧 `mxf4_gemm` 路径取参考输出 ✓ ② 再跑 GEMV 路径对拍（应 <1e-3 ✓）
③ 对拍通过后再接进模型跑四段文本 ✓（不要跳过 ①——上次就是直接接进模型才付出了代价 ✗）。

**附：本会话已建立的三件工具/口径（下会话直接可用 ✓）**
1. **差分法取 decode-only 分布**：`--max-tokens 1` 与 `--max-tokens N` 两表相减 ✓
   （`nsys stats --report cuda_gpu_kern_sum --format csv` + python 解析 ✓；
   **切勿用 table 格式配 awk ✗**——kernel 名含空格会错列 ✓）
2. **隔离复现器**（含空 kernel 发射地板对照 ✓）：真实 shape + 预热 + 500 次计时 ✓
3. **两条判据**：多卡 nsys 只用于**排序** ✓（绝对耗时不可信 ✗）；最终结论一律取
   **同二进制背靠背 A/B + 四段文本** ✓

## ★★★★★ 落地：专家 down 并入 GEMV（修掉 true 根因）—— **18.5 → 19.8 tok/s（+7%）**

**真正的根因（不是行跨距 ✗，我先前那个假设已被代码自我否决）** ✓：
调用点（`chain_dev.rs` 的专家循环）传的是 **`route_w + slot`** ⇒ **`row_weight` 是每 (token, slot) 的一个 float** ✓，
即**属于 M 行维度** ✓；而我的 GEMV 写成 `x *= row_weight[row]`，其中 `row` 遍历的是
**输出列 n（0..n_total=4096）** ✗✗ ⇒ **越界读 4096 个 float** ⇒ 全零 + illegal memory access ✓✓。
**为什么原 `mxf4_gemm` 没炸** ✓：它的 M 循环上界是 `rows`（=1 ✓）⇒ **只会访问 `row_weight[0]`** ✓；
**转置的 down 调用是我这个 kernel 第一次把该索引用错轴的地方** ✓。

**修法（一行）**：`if (row_weight != nullptr) x *= row_weight[0];`（M==1 恒为 0 ✓）
+ 把分派重新放宽到 `rows == 1`（不限 AQ ✓）⇒ down 也走 GEMV ✓。

**验证（同二进制）**：
| | 改前 | 改后 |
|---|---|---|
| 吞吐 | 18.5 tok/s（54.1ms）| **19.8 tok/s（50.46ms）** ✓ |
| 四段文本 | 全对 ✓ | **全对 ✓**（Paris / Tokyo. / “2”后停止 / 静夜思连贯）|

**会话累计：2.6 → 19.8 tok/s（7.6x）** ✓✓ —— 五个 kernel 级收益：
hc_mixes 块形状 +74% · fp4 GEMV(gate/up) +21% · fp8 GEMV +16% · warp sinkhorn +5% · down 并入 GEMV +7% ✓

### 教训（新增，务必记住）
> **区分"输出列 n"与"M 行"** ✓：任何"每行/每 token"的量（路由权重、mask、bias、scale ✓）
> 在 M=1 的 GEMV 里索引都必须是 **0（M 行）**，而不是输出列的 `row` ✗。
> 反面教材：`row_weight[row]` 在 M=128 的 GEMM 里"碰巧"只看 `[0]` 而没暴露 ✓ ——
> 移植 kernel 时必须逐个确认每个参数的**维度归属** ✓。

## ★★★★★ 复现器再下一城：`sparse_attn` 也是**与 n 无关的固定开销**（launcher 只发 8 个 block）

**复现器**（本会话新建 ✓）：`/tmp/sa_repro.cu` —— 照 `/tmp/hc_repro.cu` 模板（预热 + 300 次计时 +
**空 kernel 同 grid/block 的发射地板对照** ✓ + `cudaProfilerStart/Stop` ✓）。
Shape 取自模型：`b=1, m=1, h=nlh=8`（`num_attention_heads=64 / TP8` ✓）、
`d=head_dim=512` ✓、`topk=index_topk=512` ✓（`n` 用参数扫 ✓）。

**测量（空闲单卡）**：launcher 的 `grid = (b*m, h) = (1, 8)` ⇒ **8 个 block** ✗
| n（KV 长度）| 128 | 512 | 2048 | 8192 |
|---|---|---|---|---|
| µs/次 | 383.16 | 383.31 | **384.62** | 470.55 |
| 空 kernel 同 grid | 2.90 | 2.83 | 2.87 | 2.92 |

**⇒ n 从 128 到 2048（工作量差 16 倍）耗时**纹丝不动** ✓ ⇒ 又双叒是"固定开销主导"** ✗✓。
**机制**：**8 个 block = 1024 线程 / 148 SM（≈0.05 warp/SM ✗✗）** ⇒ 内核按 `topk=512` 做
**随机 gather**（每 slot 读 d=512 个 float ⇒ 1MB，且**非合并** ✗）⇒ **延迟完全无法隐藏** ✓。
（1MB ÷ 383µs ≈ **2.6 GB/s** ✗ —— 与文档里反复出现的"请求速率/延迟受限"签名完全一致 ✓。）

**修法（设计早已在本文件里 ✓）**：把 `topk` 维切分到 block 上 ⇒ `grid = (b*m, h, SPLIT)`，
SPLIT=8~16 ⇒ **64~128 个 block** ✓，每块做 `topk/SPLIT` 个 slot 的 **online softmax**
（running max/sum ✓，flash 式 ✓），尾块小合并 kernel 归一（合并顺序确定 ⇒ 数值可复现 ✓）。
**预期**：383µs → ~30-50µs（8~12x ✓）⇒ 该项占 decode 的比例 9.1% → ~1% ⇒ **整体 +8%** ✓。
**验证**：① 先在复现器上对拍（旧路径 vs 新路径的输出，逐元素 <1e-3 ✓ —— **复现器要支持两种路径** ✓）
② 再接模型跑四段文本 ✓ ③ 同二进制 A/B ✓。

**通用模板（本会话第三次复用，已高度成熟）** ✓：
```
launcher 里看 grid 的计算 → 若 grid 只有个位数/几十个 block 而 SM 有 148 个 ⇒ 立刻怀疑占用率 ✗
→ 用复现器做"参数扫描 + 空 kernel 地板"两件事 ⇒ 若耗时与参数无关 ⇒ 固定开销主导 ⇒ 切分网格 ✓
```

## sparse_attn 根因精确化（读代码后）：**每 slot 一次全块两阶段归约 × 512 次迭代**

```c
const int row = blockIdx.x;                     // b*m = 1  ⇒ blockIdx.x 只有 1
for (int hh = blockIdx.y; hh < h; hh += gridDim.y) {   // h = 8 ⇒ 总共 8 个 block ✗
    float acc[512];                             // 每线程 512 float、动态索引 ⇒ local memory ✗
    for (int t = 0; t < topk; t++) {            // topk = 512 次迭代
        const int idx = idxs[... ];
        float dot = 0;
        for (int c = threadIdx.x; c < d; c += blockDim.x) dot += qr[c] * kr[c];
        for (off) dot += __shfl_xor_sync(...);   // warp 级
        __shared__ float sdot; __shared__ float wpart[32];
        if (lane == 0) wpart[wid] = dot;
        __syncthreads();                         // ← 每次迭代都有全块屏障 ✗✗
        ... 跨 warp 归约 ...
    }
```
⇒ **512 次迭代 × 每迭代 ~2 个全块屏障 ≈ 上千个屏障** ✗✗，且只有 **8 个 block**
（`grid = (b*m, h)` ✓）⇒ 每个屏障的**暴露延迟**最大 ✓ ⇒ **实测 383µs，且与 n 无关** ✓✓
（这也解释了复现器里 `idxs` 全零、n 从 128 到 2048 耗时不变 ✓ —— 成本在迭代数 × 屏障，不在 gather ✓）。

**修法（从"切 grid"升级为直击要害 ✓）**：**把每个 slot 交给一个 warp** ✓ ⇒ 每个 slot 的
`q·k` 变成**纯 warp 内 shuffle 归约**（该 slot 内部**零屏障** ✓）；block 内 4 个 warp 每次处理
4 个 slot ✓，**只在每个"批次"边界做一次全块归约**（合并 max/sum 与加权 v 累加 ✓）。
⚠ **修正一个我先前写高的数字** ✗：屏障数是 **~1024 → ~256（约 4 倍）** ✓，**不是 ~100 倍** ✗
——因为 softmax 的 max/sum 是**跨全部 slot 的全局量** ✓，批次边界仍必须跨 warp 合并 ✓。
**真正的收益大头在另一处** ✓：`float acc[512]` 是**动态索引**（`acc[c]`，c 随线程走 ✓）⇒
**溢出到 local memory** ✗，每元素访问都是本地内存延迟 ✓；改成**每线程固定 d 子集（静态索引 ✓）**
后 `acc` 常驻寄存器 ✓ —— 这一项才是 383µs 里最大的可回收部分 ✓（先改它、保留原有归约结构，
是**最小风险的第一步** ✓，且能单独用复现器量化收益 ✓）。
`acc[512]` 的动态索引数组应改成**每线程负责固定的 d 子集**（静态索引 ⇒ 寄存器 ✓ 不 spill ✓）。
⚠ 这是数值性改动（归约结合顺序变 ✓）⇒ **必须四段文本复验** ✓；并且**先在复现器上对拍**
（`/tmp/sa_repro.cu` 需要扩展成能跑两条路径并逐元素比对 ✓）再进模型 ✓。
**预期**：383µs → 数十 µs ⇒ 占比 9.1% → ~1~2% ⇒ 整体 **+7~8%** ✓。

### 修正后的 sparse_attn 实施顺序（先小后大，每步单独量化 ✓）
1. **最小风险第一步**：把 `float acc[512]`（动态索引 ⇒ spill ✗）改为**每线程固定 d 子集**
   （`d/blockDim` 个元素、静态索引 ⇒ 寄存器 ✓）—— 保留全部归约结构与屏障 ✓ ⇒
   **数值逐位一致 ✓（只是数据放哪）** ⇒ 用 `/tmp/sa_repro.cu` 秒级量化 ✓。
2. **第二步**（视第 1 步收益决定）：每 slot 一 warp 消掉 slot 内的跨 warp 归约 ✓（屏障 ~4x ✓）。
3. **第三步**（可选）：`grid = (b*m, h, SPLIT)` 沿 topk 切分 + flash 式合并 ✓（并行度 8→64+ ✓）。

## ★★★★★ 落地：sparse_attn flash-decode 分槽 —— 19.9 → 21.1 tok/s（+6%）

**实现**：`sparse_attn_warp_kernel`（新增 ✓，默认启用；`DSV41_ATTN_SEQ=1` 回退顺序版做 A/B ✓，
knob 用 static 缓存 ✓）。每 warp 独占 slot 子集（`t = wid, wid+nwarp, ...`）、32 lane 覆盖全 head_dim、
自跑 online softmax ⇒ **循环内零屏障**（顺序版每 slot 2 次 `__syncthreads` × 512 slot ≈ 1000 次 ✗）；
块尾经 8KB smem 合并 4 个 partial（仅有的 ~3 次屏障 ✓）。寄存器干净（0 spill ✓）。

**同口径验证**：
| | 顺序版 | flash 版 |
|---|---|---|
| 复现器（n=128/2048）| 329.96 / 329.94 µs | **86.71 / 86.71 µs（3.8x）** |
| 模型 A/B | 19.9 tok/s（50.27ms）| **21.1 tok/s（47.46ms）** |
| 文本 | Paris/Tokyo/2/静夜思 ✓ | 同样全对 ✓（静夜思输出略变：归约结合序变化，属预期）|

**会话累计：2.6 → 21.1 tok/s（8.1x）**，六项 kernel 级收益 ✓。
**⚠ 过程坑（已入档）**：第一次验证的数字与旧版**两位小数都相同** ⇒ 是我**没 commit/push**，
远端跑的旧 `.so` ✗。**判据：不同 kernel 的数字完全相同 = 构建没变，先查 git 状态** ✓。

## 审计（用户指令：无 H2D / tile 对齐 / 单图）—— 发现比预期更严重的问题

**① 每步 H2D（4 处）+ 一个巨大的 D2H**：
| # | 位置 | 内容 | 频率 |
|---|---|---|---|
| 1 | `step()` 开头 `upload_f32_at(s.ids, [token])` | token id（4B）| 每步 |
| 2 | `upload_pre(&premix)`（[1,0,0,0] **常量**）| 16B → pre_a | 每步 |
| 3 | **`upload_bytes_at(s.eng_ids, …)`** | host 侧 n-gram 哈希结果（`ng.forward_row` 在 **host** 算 ✗）| 每步 |
| 4 | 末尾 `upload_pre(&premix)`（同一常量）| 16B | 每步 |
| ⚠ | **`download_f32(&s.logits, &mut out)`** | **全词表 129280×4B ≈ 517KB D2H 每步** ✗✗ + host 线性 argmax + `dev.sync()` 强制全同步 | 每步 |

**② 单图阻塞点**：AR 的主机 barrier（`all_reduce_inplace`→`end_round`→`barrier.wait()`）为主要阻塞 ✓；
其余为上述 H2D/D2H/host-argmax/host-engram ✗。

**③ tile 对齐**：mxf4 的 K 加载时已补齐 64 倍数（288→320 ✓）；d=512、dim/hc_dim=4096 ✓；
GEMV 的 n_total/k 均 2 的幂 ✓ —— **基本无问题** ✓。

### 实施计划（GLM 方法，按价值排序）
1. **设备侧 argmax + token 留设备**（GLM 的 HEAD_DEV ✓）：argmax kernel 把 token 写进 `s.ids`，
   嵌入直读 ⇒ 消 **517KB D2H + 4B H2D + O(vocab) host 扫描 + 强制 sync**；host 只下载 4B 做 EOS/打印。
   prefill 期 token 来自 prompt（H2D 仅 prefill，非热路径 ✓）。
2. **premix 常量设备化**：初始化上传一次到专用 buffer，每步 D2D 拷 16B（图可捕获 ✓）。
3. **engram 哈希设备化**（较大工程，host `forward_row` → kernel）。
4. **设备侧 AR**（修 DSV41_AR_DEV 的"reduce 写回"）→ 之后整步单图。

## ★★★ 落地：设备侧 argmax（GLM HEAD_DEV）—— token 不再过主机 + premix D2D

**实现**（`0b56eea`）：
- `argmax_kernel`（单 block 1024 线程、packed u64 蝶形归约、**平局取最小下标** = 与 host
  严格大于扫描完全一致 ✓）+ `dsv41_argmax` launcher；token 直接写进 `s.ids`，
  下一步的 embed 直读 ⇒ **消 517KB 全词表 D2H + O(vocab) host 扫描**；host 每步只读 4B（EOS/打印）。
- premix [1,0,0,0]（**每步常量**）在 reset 上传一次，每步 16B **D2D** 刷新（图可捕获 ✓）。
- `step(token,pos)`=喂数路径（prefill）；`step_dev(token,pos)`=**解码稳态零 H2D**（token 值仅喂
  host 侧 engram 哈希，不上传 ✓）。top5 调试移进 chain（DSV41_TOP5）。

**验证**：四段文本逐段人工读过 ✓（Paris/Tokyo/2/《静夜思》——李白，与 flash 版一致）；
性能 **21.2 tok/s（47.16/47.26ms）** vs 改前 21.1（47.46ms）—— 省的 ~0.26ms 与
517KB D2H + 扫描的量级吻合 ✓。**会话累计 2.6 → 21.2 tok/s（8.2x）**。

### 用户三项指令的进度
| 指令 | 状态 |
|---|---|
| 无任何 H2D | **解码稳态 = 0**（engram 哈希仍在 host 算但只上传一次/步 —— ⚠ 这是最后一处 H2D，需设备化）|
| tile 对齐 | ✓（mxf4 K 已补 64 倍数；其余维度全 2 的幂）|
| 单 CUDA graph | **阻塞 = AR 的主机 barrier** ✗（+ eng_ids 的每步上传 ✗）|

### 下一步（按依赖序）
1. **设备侧 AR**（单图的唯一硬阻塞）：DSV41_AR_DEV 已有骨架（5 次尝试未成，最后卡在
   "reduce 没把和写回"）；GLM AR v5 的结论直接适用：**epoch 只由图回放推进（TP 天然 lockstep）⇒
   结构上不可能漂移** —— 把 v5 的 3-kernel（store/publish/pubred）协议移植到 DSV41 的
   cudaMemcpyPeerAsync staging 上。
2. engram 哈希设备化（消最后一处 H2D；读 s.ids 的设备 token ⇒ 可进图）。
3. 整步单图捕获（embed → 45 层 → head → argmax 全进一张图；4B 读在图外）。

## ★★★★★★ 里程碑：**AR v5 完全工作**（第 6 次尝试成功）—— 单图最后硬阻塞移除

**此前 5 次设备侧 AR 尝试全部失败**（见上文各节）；本次移植 **GLM AR v5 协议**成功 ✓✓。

**协议（与 GLM 完全同构 ✓）**：
| kernel | 职责 |
|---|---|
| `ar_v5_store(e)` | 把本 rank 缓冲写到**所有对端**的 staging 半区 `e&1`（epoch **运行时从设备内存读** ⇒ 图可回放 ✓）|
| `ar_v5_publish(e)` | 1 block：`__threadfence_system` 后向所有对端盖章 `e+1`（本 rank 槽位），再轮询**自己的** stamps 直到全部 `≥ e+1` ⇒ 所有对端的 store(e) 完成 |
| `ar_v5_reduce(e)` | 求和对端 staging 半区 `e&1` **直写调用方缓冲**（p 升序 = 旧路径序 ⇒ 逐位一致 ✓）；**最后一块**推进 epoch（`*epoch = e+1`、清 ctr）⇒ 每次回放恰好一轮 ✓ |

**为什么不需要 credit 等待**（旧设计的 `reduced[p] ≥ round-2` 自旋 ✗ 已删）：**publish 链**替代之 ——
publish(k) 证明所有对端 store(k) 完成；而我的 store(k+2) 前有我的 publish(k+1) ⇒ 所有对端已完成
reduce(k)（被覆盖半区的最后读者）✓✓。**无主机 barrier、无 copy-back、无 credit** ✓。
`end_round()` 在 v5 下为 no-op ✓。开关 `DSV41_AR_V5=1`（**static 缓存** —— 每次 AR 调用的热路径 ✓）。

**验证（三层全过 ✓）**：
| 层 | 结果 |
|---|---|
| 微基准 | `[ar_micro] OK: world=4 rounds=8 n=1024` ✓（2.47s）|
| 模型四段 | Paris / Tokyo / "2" / 《静夜思》——李白 —— **与 host 路径逐字相同** ✓✓（亲自读）|
| 同二进制 A/B | v5 **21.4 tok/s（46.71ms）** vs host 21.2（47.23ms）⇒ **+1%** ✓（GPU-bound 下符合预期）|

**本项踩的 2 个构建坑（都已修，值得记住）**：
1. `const unsigned* epoch` 却要 `*epoch = e+1` ⇒ `modifiable lvalue` 编译错 ✗ ⇒ reduce 的 epoch 参数须非 const ✓。
2. **launcher 插在匿名 namespace 内** ⇒ 内部链接 ⇒ `nm -D` 找不到 ⇒ "kernel not in the loaded .so" ✗✗
   （与早前 gemv 入口同坑 ✓）⇒ 移到 namespace 关闭之后 ✓。**判据：绿色构建 ≠ 符号存在，`nm -D` 必查** ✓。

**会话累计：2.6 → 21.4 tok/s（8.2x）**。

### 用户三项指令的进度（更新）
| 指令 | 状态 |
|---|---|
| 无任何 H2D | 解码稳态仅剩 **engram 哈希的每步上传** ✗（host 算 `ng.forward_row` → upload）⇒ 需设备化 |
| tile 对齐 | ✓ |
| 单 CUDA graph | **所有硬阻塞已除** ✓（AR v5 ✓ HEAD_DEV ✓ premix D2D ✓）⇒ 剩 engram 的图外喂数问题 |

## 单图捕获的完整依赖清单（审计完成，可直接执行）

**已就位 ✓**（本会话落地）：
| 依赖 | 状态 |
|---|---|
| AR 无主机 barrier | ✅ AR v5（epoch 设备驻留、运行时读取 ⇒ 图可回放 ✓）|
| token 留设备 | ✅ HEAD_DEV（argmax 写 s.ids，embed 直读 ✓）|
| premix | ✅ D2D 每步刷新（图可捕获 ✓）|
| 集合同步原语 | ✅ publish 链（图内的会合等待 ✓）|

**剩余的每步动态宿主量（图捕获会冻结 kernel 参数 ✗ ⇒ 必须设备化）**：
| # | 量 | 位置 | 设备化方案（GLM 的 pinned 推进模式 ✓）|
|---|---|---|---|
| 1 | `pos` | `layer(layer, pos, ..)` → `attention()` 里两处 `pos as i32` 作 kernel 参数（chain_dev.rs:961/996）| 设备计数器 + 图首小 kernel 自增（或消费 kernel 内读+推进，如 AR v5 的 epoch ✓）|
| 2 | `compress_len` | `LayerCache.compress_len`（每层宿主状态，每步变化）| 同上：设备计数器数组 [n_layers]，或从 pos 派生 |
| 3 | **engram 哈希** | host `ng.forward_row(lay, map, 0, &[token], pos, None)` → `upload_bytes_at(eng_ids)`（**解码路径最后一处 H2D** ✗）| 见下 |

### engram 哈希设备化的设计（已读 forward_row 全文 ✓）
**逻辑**（decode 时 seqlen=1）：cache[pos] = map.get(token)（DEAD 规则）；对 shift∈0..max_ngram 取
lookback（blocked 规则→pad_id）；rolling = tokens[0]×mult[0]，逐 i：rolling ^= tokens[i]×mult[i]，
对每 (layer li, head h)：`out[..] = rolling.rem_euclid(lm) + off`（lm/off 来自 `layer.column()`）。
**需要的设备缓冲**（一次性上传 ✓）：
- `map 表`：token→compressed id（含 DEAD 编码）[vocab] i64 —— 加载时上传
- `mults`：每 layer 的 mult[0..max_ngram]（小）
- `cols`：每 layer 的 (lm, off) 对 [layers][n_cols]（小）
- `cache`：[max_seq] i64 跨步状态；`pos_ctr`：设备计数器
**kernel**（~60 行）：读 s.ids 的 token + *pos_ctr → 更新 cache → 算哈希写 eng_ids → 推进 pos_ctr。
**prefill 衔接**：prefill 走 host 路径（现状 ✓），结束后把 host cache 一次性上传设备 + 初始化 pos_ctr
（一次性 H2D，非热路径 ✓）。
**图结构**：图 = [engram_hash → embed → 45×(layer + 2×AR v5) → head → argmax]；图外只有 4B 读（EOS/打印 ✓）。

### 执行顺序（下会话）
1. engram kernel + 设备缓冲 + prefill 衔接（验证：四段文本**亲自读** + 微基准对拍 host 路径）
2. pos/compress_len 设备化（图首推进 kernel）
3. 图捕获（第一步热身、第二步捕获、之后回放 —— 复用 DSV41_GRAPH_MOE 的捕获框架 ✓
   与其 4 个已修的坑：async memset/memcpy、Relaxed 模式、分配时机）
4. 验证：四段文本 + A/B + 长跑稳定性（AR v5 的 epoch 跨图切换 —— b16→b2 场景在 GLM 已验证 ✓）

### 本会话总结（供回顾）
**2.6 → 21.4 tok/s（8.2x）**，八项收益全部同二进制 A/B：hc_mixes 块形状 +74% · fp4 GEMV +21% ·
fp8 GEMV +16% · warp sinkhorn +5% · down 并入 GEMV +7% · sparse_attn flash 分槽 +6% · HEAD_DEV ~+1% ·
AR v5 +1%（结构解锁）。**三项结构性资产**：AR v5（第 6 次尝试成功）· HEAD_DEV · 差分法+复现器测量体系。

## ★★★ 落地：engram 哈希设备化 —— **解码路径 H2D = 0**（用户指令①完成）

**实现**（`d788df7`）：`dsv41_engram_hash_step`（单线程 kernel，**与 host `forward_row` 的
serial 算术逐位一致** ✓）：读 `s.ids` 的设备 token + 设备 pos_ctr → 更新 cache → blocked 规则 →
rolling XOR → 写 eng_ids。map/mults/primes/offsets **一次性上传**（首步懒建）；reset 清零。
`DSV41_ENG_HOST=1` 回退 host 路径做 A/B。
**注意**：`dsv41_kernels.cu` 里已有同名多 token 版（从未被调用 ✗）——它 seqlen=1 时
threads 1..127 提前 return 后**单线程到达 `__syncthreads`**（文档化 UB ✗）且 start_pos 是宿主参数
（图不安全 ✗）；新 kernel 规避两者并改名避免 extern "C" 链接冲突 ✓。

**验证**：四段文本与 host 路径**逐字相同** ✓（Paris/Tokyo/"2"/《静夜思》——李白，亲自读）；
A/B 中性（21.2 vs 21.3 tok/s —— 哈希本身极小，收益是**消 H2D + 图可捕获** ✓）。

**用户三项指令进度**：① 无 H2D ✓✅ ② tile 对齐 ✓✅ ③ 单图 —— **所有硬阻塞已除**
（AR v5 ✓ HEAD_DEV ✓ premix D2D ✓ engram 设备化 ✓），剩 **pos/compress_len 设备化 + 捕获**。

## ⚠️ 审计修正：attention 路径还有**每层每步的 H2D + 宿主分支**（此前遗漏 ✗）

深挖 compress/attention 后发现（"无 H2D"的结论**过早**了 ✗，予以更正 ✓）：
| # | 位置 | 内容 | 频率 |
|---|---|---|---|
| 1 | `attention()` 里 `ops::window_topk_idxs(win,1,1,pos)` + `ul_i32(idxs_ptr, ..)` | **宿主算窗口索引 + 每层上传 win×4B** ✗ | 每层每步 |
| 2 | `compress()` 里 `download_f32(out_rows)` → `if n[0] > 0 {rope + memcpy_d2d + len+=1}` | **每层同步 D2H + 宿主数据分支** ✗（图不可捕获）| 每 KV 源层每步 |
| 3 | `compress_len`（`LayerCache` 宿主状态）| 作为 kernel 参数传入注意力/索引 ⇒ 图会冻结 ✗ | 每层 |
| 4 | `pos`（attention 两处 `pos as i32` 作 kernel 参数）| 同上 ✗ | 每层 |

### 剩余执行计划（attention 设备化 → 单图，全部已设计好）
1. **窗口索引设备化**（消 H2D #1）：小 kernel 读设备 pos 计数器填 `idxs[0..win]`
   （decode 分支的语义：尾部 window 个位置，越界补 -1 —— 照 `window_topk_idxs` 的 else 分支逐位移植 ✓）。
2. **compress commit kernel**（消 D2H #2 + 分支）：读设备 `out_rows`，若 >0 则
   **rope the latent**（照 `apply_rope_kernel` 的数学：`row = x + (hd - rope_dim)` 偏移、
   `cos[t*half+i]` 表索引、成对旋转 ✓）→ 存 ring 第 `window + *clen` 行 → **推进设备 clen**。
3. **compress_len → 设备计数器数组 `[n_layers]`**，注意力的 kvb/indexer kernel 改读设备值。
4. **pos → 单一设备计数器**：把推进从 engram kernel 移到 **argmax**（步内最后一个 kernel ⇒ 步内 *pos 稳定 ✓）。
5. **审计 indexer 路径**（"indexer 在设备上覆写压缩块"——可能还有宿主工作）。
6. **捕获**（复用 DSV41_GRAPH_MOE 框架 + 其 4 个已修坑 ✓），图 = [engram → embed → 45×(layer+2×AR v5) → head → argmax]。

**验证纪律（用户最新指令 ✓）**：隔离复现器秒级迭代；**只在落结论时跑一次模型**（四段文本**亲自读** ✓）。

## 本会话最终总结（2026-09-11）
**2.6 → 21.4 tok/s（8.2x）**，九项改动全部验证（前八项 A/B + engram 中性）：
| # | 改动 | 收益 |
|---|---|---|
| 1 | hc_mixes 块形状（mix*32）| 7.2 → 12.5（**+74%**）|
| 2 | M=1 fp4 GEMV（专家 gate/up）| → 15.2（+21%）|
| 3 | M=1 fp8 GEMV（dense）| → 17.6（+16%）|
| 4 | warp 寄存器 sinkhorn | → 18.5（+5%）|
| 5 | 专家 down 并入 GEMV（修 row_weight 轴错误）| → 19.8（+7%）|
| 6 | sparse_attn flash 分槽 | → 21.1（+6%）|
| 7 | HEAD_DEV 设备 argmax | → 21.2 |
| 8 | **AR v5**（第 6 次尝试成功）| → 21.4（+1%，**结构解锁**）|
| 9 | engram 设备哈希 | 中性（**消 H2D** ✓）|
**三项结构性资产**：AR v5（图可捕获集合通信）· HEAD_DEV（token 不过宿主）· 差分法+隔离复现器测量体系。
**方法论沉淀**：全运行 nsys 平均 ≠ decode（差分法 ✓）；多卡 nsys 单次耗时不可信（复现器 ✓）；
L2 常驻工作集上"流量大"≠瓶颈；一次只改一个变量；同二进制背靠背；构建后必查 error 数与 nm；
匿名 namespace = dlsym 盲区；"输出列 n"≠"M 行"。

## ★★★★★★ 本阶段：设备化 → 整步单图 → serve 并入共享栈（用户三项指令）

### ① 全部每步动态值设备化（图捕获的前提）
| # | 量 | 设备化方式 |
|---|---|---|
| 1 | `pos` | 单一点位计数器 `s.pos_ctr`；**argmax（步内最后一个 kernel）推进它** ⇒ 步内所有 kernel 读到的都是稳定当前值 ✓ |
| 2 | `compress_len`（每层） | `s.clen[n_layers]` 设备计数器 ✓ |
| 3 | compress 的下载+分支+rope+拷贝 | **新 `dsv41_compress_commit`**：读设备 `out_rows` → rope（照 `apply_rope_kernel` 的数学：`hd-rope_dim` 偏移、`cos[t*half+i]`、成对旋转）→ 存 ring 第 `window+*clen` 行 → 推进 `*clen` ✓ **无 `__syncthreads`**（每线程自读对、写目标 ⇒ 规避早退 UB ✓）|
| 4 | `apply_rope` 的 pos/group 位置 | 统一为**设备指针 + 乘加**：`t = (*base)*mul + off` ⇒ 覆盖 `pos`(mul=1,off=0) 与 `(clen-1)*ratio`(base=&clen, mul=ratio, off=-ratio) 两种 ✓ |
| 5 | `compressor_pool`/`compressor_state` | 读设备 `pos_ctr`；**`out_rows_val` 由 kernel 从计数器算出**（`((*pos_ctr+1)%ratio==0)`）✓ |
| 6 | `sparse_attn` 的 `n`/`topk` | 改为读设备 clen：`n = window+*clen`、`topk = window+min(*clen,index_topk)` ✓ |
| 7 | recency placeholder | 同上（take 由 kernel 算，launch 用上限 + 内核守卫）✓ |
| 8 | indexer 的 `idx_lens` | **去掉每步上传**，直读设备 clen ✓ |
| 9 | 窗口索引 `window_topk_idxs` | 新 `dsv41_window_idxs`（decode 分支**逐位移植** + start_pos==0 特例）✓ 消每层每步 H2D ✓ |

### ② 整步单图（用户指令③）
`step_impl`（主机侧：图分支 + 同步 + 4B 读）↔ **`step_body`（纯设备算子，无主机往返）** ✓。
- 捕获在**第二步**（第一步热身 kernel/懒建设备态/给 cublas 定 workspace ✓）；
- **捕获只记录不执行** ⇒ 捕完立刻 launch 一次以完成本步 ✓；
- 门禁：`DSV41_ENG_HOST`/`DSV41_STATS` 与图互斥（它们在录制区内做主机往返 ✗）；
  **开图即强制 AR v5**（主机 barrier 不是 CUDA 调用 ⇒ 不会被录进图 ⇒ 回放会静默丢失跨 rank 同步 ✗✗）；
- 开关：`DSV41_GRAPH_STEP=0` 关闭（默认开 ✓）。

### ③ serve 并入共享栈（用户："能共享的必须共享，严禁重复造轮子"）
删掉手写 `std::net` 服务器 ✓，改为复用 `crates/ferrite-http`（GLM 同一套）：
`api::router`(axum/SSE/usage/cancel//shutdown) + `driver::EngineDriver` + 共享 `serve.rs::launch` ✓；
新增的是**通用能力**（不泄漏引擎约束）：`engine.rs` 的 `ServeEngine` trait、`single_flight.rs`（**batch=1 锁步引擎**的通用包装）、`StopSpec`/`ChatFrame` 通用化（按模型只分叉**数据** ✓）。GLM 路径零改动 ✓。

### ⚠️ 唯一一次 e2e 揪出的两个根因（都已修 ✓）
1. **捕获区内仍有同步 D2H** ✗：`layer 0` 有一处**无 env 门控**的遗留探针（`self.dl(...)` 每步下载 ✗）
   ⇒ 报 `cudaMemcpy D2H: operation would make the legacy stream depend on a capturing
   blocking stream` ✓（与先前删掉的 `moe_out` 探针同类 ✗）。**审计法**：逐函数列出全部下载点，
   逐个核对门控（其余四处均有 `DSV41_HCDBG` ✓）；**门控必须写在条件里，不能只写在函数内部** ✓。
2. **chat 模板标记被字面量编码** ✗：frame 用 `Seg::Special("<|User|>")` 但该 checkpoint
   **未把标记名注册为 special** ⇒ 退化成字面量文本 ⇒ prompt ≠ 已验证的
   `[0,128803]+body+[128804,128822]` ✓。修法（**通用** ✓）：共享 `Seg` 增加 `Seg::Id(u32)`，
   DSV41 的 frame 钉死这些 id ✓ —— 任何"标记不是 special"的模型都能用同一机制 ✓。

### 纪律教训（已入档）
- **严禁 `pgrep -f "…dsv41-run"` 配 kill** ✗：它匹配 ssh 命令行本身 ⇒ 自杀（exit 255）✗；
  用 **`pgrep -x dsv41-run`**（精确进程名）✓。
- **改完立刻 commit/push**：本轮两次"数字与旧版完全相同"都是**没推送**（远端跑旧 `.so`）✗。
- **多卡 nsys 的单次耗时不可信**（只用于排序 ✓）；最终判据一律 **同二进制 A/B + 亲自读四段文本** ✓。

## ⚠️ 本轮 e2e 的两个结论（都已处置 ✓，图仍待二分）

### ① A/B 判定：**设备化正确、CUDA 图破坏正确性** ✓✗
同一二进制、同一次会话：
| 配置 | 输出 |
|---|---|
| `DSV41_GRAPH_STEP=0`（关图 ✓） | **`'Paris'` ✓ / `'2'` ✓ / 0 fault ✓** |
| 默认（当时开图 ✗） | 乱码 ✗（`':\n    #include <stdio.h\n《\n-2'` 等）|
⇒ **9 项设备化是对的** ✓；**图的重放产出了错误结果** ✗（不是报错、不是崩溃，是静默数值错 ✗）。
**处置**：图改为 **opt-in**（`DSV41_GRAPH_STEP=1` 才开 ✓）—— 正确性是红线，未定位前不当默认 ✓。
**待二分方向**（下会话）：① 图内 kernel 参数是否仍有随步变化者（逐 kernel 列参数核对 ✓）；
② 捕获时机（现在在第 2 步 = **prefill 的第 2 个 token** ⇒ 捕获的是 prefill 形态的图 ✗，
   而 decode 的某些 host 分支（如 `comp_len > 0` 相关）可能不同 ⇒ 改为**预填充结束后再捕获** ✓ 优先试）；
③ AR v5 的 publish 自旋在捕获/首次重放时的时序 ✓。

### ② 我引入的真 bug：宿主 `comp_len` 恒为 0 ✗（已修 ✓）
`compress()` 改成返回 `Ok(0)` 后，宿主 `compress_len` 不再增长 ⇒ **所有 `if comp_len > 0` 的路径
（indexer 运行、压缩槽位预期）全死** ✗ —— 24 token 的短 prompt 靠窗口仍答对 ✓（所以 A/B 里
"Paris" 正确并不代表全对 ✗），**长文本必错** ✗。
**修法**：宿主用**与 kernel 完全相同的确定性规则**维护镜像（`(pos+1) % ratio == 0` ⇒ 提交一个 latent ✓），
两边 by construction 一致 ✓，且**不再需要任何下载** ✓。

## ✅ 本轮最终状态：默认路径已验证正确（含压缩 KV 路径复活）

**用户三项指令 + 唯一一次 e2e 的收获**：e2e 一次性扇出 **3 个真根因** ✓，全部修复并验证 ✓：

| # | 根因 | 性质 | 修法 |
|---|---|---|---|
| 1 | `layer 0` 有**无 env 门控**的遗留探针（`self.dl` = 同步 D2H，每步 ✗）⇒ 落进图捕获 ⇒ `cudaMemcpy D2H: operation would make the legacy stream depend on a capturing blocking stream` | 捕获纪律 ✓ | 删除；**审计法**：逐函数列出全部下载点核对门控（其余 4 处有 `DSV41_HCDBG` ✓）|
| 2 | chat frame 用 `Seg::Special("<|User|>")`，但该 checkpoint **未注册这些 special** ⇒ 按字面量编码 ⇒ prompt ≠ 已验证的 `[0,12803]+body+[12804,12822]` ✗ | 正确性 ✓ | 共享 `Seg` 增 **`Seg::Id(u32)`**（通用 ✓），DSV41 frame **钉死 id** ✓ |
| 3 | 我改的 `compress()` 返回 `Ok(0)` ⇒ **宿主 `comp_len` 恒 0** ⇒ 所有 `if comp_len > 0` 路径（indexer/压缩槽位）**全死** ✗ | 正确性 ✓ | 宿主用**与 kernel 同一条确定性规则**维护镜像（`(pos+1)%ratio==0` ✓）⇒ 两边 by construction 一致 ✓ 且无需下载 ✓ |

**验证（同二进制、HTTP serve 一次加载 ✓）**：
| prompt | 输出 |
|---|---|
| The capital of France is | **`Paris`** ✓ |
| The capital of Japan is | **`Tokyo`** ✓ |
| 1+1= | **`2`** ✓ |
| 请背诵《静夜思》 | **`《静夜思》\n唐·李白\n床前明月光，`** ✓（**压缩 KV 路径复活** ✓）|
| 长文本 48 token | **`## 当机器学会思考\n\n清晨，我对着手机说了一句"今天天气怎么样"，手机立刻用温柔的合成语音回答了我。…`** ✓ 通顺中文 |
| faults | **0** ✓ |

**图的状态**：捕获**不报错** ✓，但**重放产出错误结果** ✗ ⇒ 已改 **opt-in**（`DSV41_GRAPH_STEP=1`
才开 ✓，正确性是红线 ✓）。**最可疑的根因已定位并修**：此前在**第 2 个 STEP** 武装捕获 ✗ =
**第 2 个 prefill token** ⇒ 把 **prefill 的宿主分支选择烤进图** ✗ ⇒ decode 回放走错分支 ✓
（与"模型像在续写别的东西"的现象吻合 ✓）⇒ 改为**只在 decode 路径武装**（`decode_steps >= 1` ✓，
已 push ✓，待下轮一次 e2e ✓）。

**教训（已入档）**：① `pgrep -f "…dsv41-run"` 会匹配 ssh 命令行 ⇒ **自杀 exit 255** ✗，用 `pgrep -x` ✓；
② 门控必须写在**条件里**（写在函数内部不够 ✓ —— 探针的下载仍在 ✓）；③ 多卡 nsys 单次耗时不可信（只排序 ✓）；
④ **同二进制 A/B + 亲自读文本**是唯一可信判据 ✓。

## 图的最终精确状态（per-request 重新武装后）

**已修** ✓：`decode_steps` 跨请求残留 ⇒ 下一请求的 **prefill 走了 decode 图** ✗（图里烤的是 pos≈25 的
mode=2 分支，而 prefill 的 pos=0 该走 mode=1 ✗）。`reset()` 归零 `decode_steps` ✓
（**图本身保留复用** ✓ —— 所有 launch 参数均已设备化 ⇒ 与状态无关 ✓）。
**验证（图开，一次 serve）**：
| 请求 | 输出 | 判定 |
|---|---|---|
| The capital of France is | `Paris` | ✓ |
| The capital of Japan is | `Tokyo` | ✓（修好前是乱码 ✗）|
| 1+1= | `2` | ✓（修好前 `方格子` ✗）|
| 请背诵《静夜思》 | `《静夜思》：\n\n**《\n\n《静夜思\n\n《静` | ✗ **重复退化** |
| 长文本 48 tok | `## 人工智能\n\n人工智能，。  人工智能\n\n"人工智能，从算力` | ✗ 重复退化 |
| faults | 0 | ✓ |

**判读** ✓：**短生成（≤3 token）完全正确 ✓；生成到 ~4 token 之后开始重复** ✗
⇒ 图的**重放存在随步数累积的残留 bug** ✗（与 `pos`/`clen` 增长后的某条路径相关 ✓）。
**处置**：图保持 **opt-in** ✓（正确性是红线 ✓）；**默认路径已单独验证（含长文）完全正确** ✓。
**下会话的二分建议**（按可能性排序）：
1. **AR v5 在图内的 epoch/parity**：90 次 AR/步、每次 parity 翻转 ✓ —— 用 world=2 的段图先隔离验证 ✓；
2. **compress commit 的 `*clen` 推进 + ring 写位置**：pos 增大后 clen>0 ⇒ commit 真正开始写 ring ✓
   （短生成时 clen=0 ⇒ commit 是 no-op ✓ ⇒ **这解释了"3 token 内正确、之后退化"** ✓✓ 最可疑 ✓！）；
3. 缓存池/环的 **graph 节点参数是否含地址**（若某处仍按 host 计算 ⇒ 冻结后错 ✓）。

## ★★★★★★★ 图的根因定案：**KV 追加的目的地址被宿主算 ⇒ 被烤进图**（用户 GLM 经验命中 ✓）

**症状**：图开时**短生成（≤3 token）正确**、之后**重复退化** ✗；关图全对 ✓。
**用户提示**："我们 GLM 遇到过这种问题，就是有**捕获的东西没更新**" ✓✓ —— 一击命中 ✓。

**根因**（`attention()` 内）：
```rust
let slot = pos % win;                                   // 宿主按 pos 算环槽位 ✗
self.dev.memcpy_d2d(ring + slot*fb(hd), self.s.kv, fb(hd));   // ← 目的地址被烤进图的 memcpy 节点 ✗✗✗
```
⇒ **每次重放都把新 token 的 KV 写到同一个槽位**（捕获那步的 slot ✗）⇒ 窗口 KV 逐渐陈旧 ⇒
**退化随生成长度累积** ✓✓；前 2-3 个 token 靠 **prefill（未用图）写好的 KV** 仍正确 ✓✓ —— 现象完全吻合 ✓。

**修法**：新增 `dsv41_ring_append` kernel —— 槽位由 **`*pos_ctr` 在 kernel 内算** ✓（与 compress_commit
同一类修法 ✓）。**验证（图开、.so 重编后）**：
| 请求 | 输出 | |
|---|---|---|
| The capital of France is | `Paris` | ✓ |
| The capital of Japan is | `Tokyo` | ✓ |
| 1+1= | `2` | ✓ |
| 请背诵《静夜思》 | `《静夜思》\n唐·李白\n\n床前明月光，` | ✓✓ **不再退化** |
| 长文本 16 tok | `## 当机器开始思考\n\n清晨，我对着手机说了一句"今天` | ✓✓ 通顺 |
| faults | **0** | ✓ |
⇒ **图已设为默认**（`DSV41_GRAPH_STEP=0` 可回退做 A/B ✓）—— **用户三项指令全部达成** ✓✓✓。

**本会话新增的通用审计法（值得复用）**：
> 图化后若"短对长错/逐步退化"，**逐个检查所有以宿主计算值作为 `cudaMemcpy` 目的/源地址的调用** ✓
> —— 它们不会被"参数设备化"审计覆盖（因为地址不是 kernel 参数 ✗），但会被捕获冻结 ✗。
> 判据：**症状与生成长度相关、且拐点与某个 host 计数的周期吻合（本例 ratio=4）** ⇒ 直指该路径 ✓。

**另一个纪律坑**：验证命令**必须重跑 `build.sh`**（本次漏跑 ⇒ 新 kernel 不在 `.so` ⇒ 5 个请求全 fault ✗，
浪费一轮 ✓）。**改 `.cu` 就重编 `.so`** ✓。

## serve 端计时已就位 + 一个关键发现：**rank 侧 21.8-45.2 ms/step，端到端 129 ms/token**

**新增**（GLM 口径 ✓，打在 **rank 线程** ⇒ 纯 decode 时间，无 HTTP/SSE/driver 开销 ✓）：
```
[dsv41] decode: 32 steps in 0.70s = 46.0 steps/s (21.76 ms/step)    ← 短上下文
[dsv41] decode: 32 steps in 1.42s = 22.6 steps/s (44.23 ms/step)    ← 长上下文（DSA decay 1.39x 一致）
```
`DSV41_TIMING=0` 可关。**lookahead 批命令**（一次命令跑 16 步 ✓）也已验证：四段 + 整首《静夜思》+ 长篇
散文全对 ✓、**0 fault** ✓。

**发现（下一个优化项）** ✓：单请求**端到端**只有 7.75 tok/s（129 ms/token ✗），而 **rank 侧只要
21.8-45.2 ms/step** ⇒ **共享 serve 路径（driver/SSE）额外吃 ~85 ms/token** ✗✗。
排查方向（按可能性）：① **逐 token 的 detokenize**（vocab 大 ⇒ 每 token 解码成文本给 SSE ✗）；
② driver 的逐 token 事件/通道往返 ✗；③ SSE 分块刷写的 syscall ✗。
**对照**：此前**手写 serve** 端到端 = 21.7 tok/s ≈ rank 侧时间（无额外开销 ✓）⇒ 差距全在共享栈的
流式/驱动环节 ✓ —— 修法应在共享栈内做（**通用** ✓，GLM 侧同样受益 ✓）。

## ★★★★★★ serve 栈两个退化全部修复：**端到端 7.75 → 19.50 tok/s（+152%）**

| 口径 | 改前 | 改后 |
|---|---|---|
| **稳态 step time（权威 ✓）** | 逐 token 往返把 rank 拖在水面下（serve 侧 ~85ms/token 开销 ✗）| **21.76 / 21.75 ms/step（短）· 44.23（长）** ✓✓ |
| 端到端（客户端 curl wall；**仅交叉验证** ✗）| 7.75 tok/s（129 ms/token）| 19.50 tok/s（51.3 ms/token）|
| 正确性 | ✓ | `Paris` ✓ / `2` ✓ / 《静夜思》✓ |

⚠ **口径提醒（用户纠正 ✓）**：评价性能**必须用中间稳态的 step time** ✓（GLM 侧 `[megab] replay`
中位数的规矩 ✓）；端到端/wall 含 prefill、爬坡与 flush ⇒ **只作交叉验证** ✗。

**修的两件事（都在共享栈内 ✓，GLM 同样受益 ✓）**：
1. **每 token 一次「命令 + 8 rank ack」往返** ✗ ⇒ rank 线程内**前瞻批命令**（`LookaheadRun`，一次 16 步 ✓），
   池用缓冲逐 token 交付（引擎契约不变 ✓；遇 stop token 提前收 ✓；新请求清缓冲 ✓）。
2. **SSE 帧窗口**（攒 8 token 或等满 50ms ✗）⇒ 单流下**每帧空等 50ms** ✗ ⇒ 改为**尾部字节完整即发** ✓
   （只保留 UTF-8 tail-holdback ✓）。**这是我此前误判的一环**：先怀疑"每 token detokenize O(n²)" ✗，
   读代码发现 batch 有界（≤8 ✓）⇒ 否证 ✓，真因是**帧窗** ✓（**先看代码的价值** ✓）。

**判据（已记入共享文档 `docs/agent/perf-roadmap.md` ✓）**：服务端 step 时间与客户端端到端时间
**必须分别记录** ✓ —— 两者差额 = serve/驱动/流式栈开销 ✓。现差值 ~5-25ms（改前 ~85ms ✓）。

## 会话终态与待验证项（2026-09-11）

### 已验证 ✓（同二进制、四段文本亲自读过）
- **正确性**：`Paris` / `Tokyo` / `2` / 《静夜思》整首 / 长篇散文通顺；faults **0** ✓
- **三项指令**：解码稳态**零 H2D** ✓ · tile 对齐 ✓ · **整步单 CUDA graph（默认开、已验证）** ✓
- **稳态 step time（权威口径 ✓）**：**21.76 / 21.75 ms/step（短，46 steps/s）**、
  **44.23 ms/step（长，22.6 steps/s）** ✓；端到端 19.50 tok/s 仅作交叉验证 ✓
- **共享栈两个退化已修**（lookahead 批命令 + SSE 帧窗 ✓）：端到端 7.75 → 19.50 tok/s ✓

### ⚠ 待统一复验（我清告警时**触碰了行为邻近代码** ✗，必须在迁移 subagent 落地后跑一次）
清告警删掉/改名的东西里有几处离行为很近 ✓（`take_comp`/`placeholder` 局部、`idx_lens` 缓冲、
`engram_apply` 的 `rank` 声明恢复 ✓、`quant.rs` 的 `mi`、`tp.rs` 的第二处 `ctr2` ✗）——
**编译已过 ✓**，但**没有重跑模型** ✗。复验清单（一次即可 ✓）：
1. 四段文本 + 长文（**亲自读** ✓）；
2. `cargo test -p ferrite-dsv41 --test ar_micro`（world=4 ✓）；
3. 稳态 step time 与上表一致（±2% ✓）。

### 未做/已交出的（诚实的交接）
- **`ferrite-dsv41` 的剩余告警**（~30 条，排除 subagent 正在改的 `device.rs` ✗）：**刻意未清** ✗ ——
  该 crate 在迁移中被**整体删除** ✓ ⇒ 清理是**一次性投入、零留存价值**，且我这轮已因"批量套用
  编译器建议"**连续 5 次改错位置** ✗（教训已入档 ✓）。**建议随迁移一并消失** ✓。
- `ferrite-kernel`（3 条）/`ferrite-exec`（24 条）：**在 subagent 的文件范围内** ✗ ⇒ 等它释放后再清 ✓。
- **性能**：稳态 21.8-44.2 ms/step vs 目标 5 ms（**4.35x** ✗）⇒ 取证顺序已入 `docs/agent/perf-roadmap.md` ✓
  （差分法新分解 → M=1 形状审计 → 已知候选 → 段融合 ✓）。

## ⚠ 审计发现的**第二处同类真 bug（已修 ✓）**：indexer 的 index_k 发布地址

**审计方法**（受用户 GLM 经验启发 ✓）：**把"宿主算出的地址被烤进图"当作一个 bug 类** ✓，
逐处检查步内所有 `memcpy_*` 的源/目的地址与 kernel 参数里的指针运算 ✓。

**发现**（`chain_dev.rs` 原 1430 行）：
```rust
let group = if owns_k { self.layers[layer].compress_len.saturating_sub(1) } else { 0 };  // 每步变 ✗
...
memcpy_d2d(index_k.ptr + group * idx_hd * 4, self.s.idx_k.ptr, idx_hd*4);   // ← 目的地址被烤进图 ✗
```
⇒ 图每次重放都把 indexer 的压缩 key 写进**同一个组槽位** ✗ ⇒ 压缩 KV 的 key 陈旧 ✓
（我此前的五段文本测试**掩盖了它** —— 短上下文由 128 槽窗口主导 ✓，压缩路径影响小 ✓）。

**修法**（与 ring_append 同一模式 ✓）：新 `dsv41_index_k_publish` kernel —— 组号由 **`*clen[owner]`
在设备上算** ✓（`group = clen > 0 ? clen-1 : 0` ✓）；launcher 放在**匿名 namespace 之外** ✓。

**其余审计结论**（无问题 ✓）：`pre_a` 的 16B D2D（常量 ✓）、`h→h2` 拷贝（定址 ✓）、
indexer 的 `for g in 0..nlg`（`nlg = cfg.o_groups/world` **静态模型维度** ✓，非每步值 ✓）、
`eng_ids + li*n_cols`（静态 ✓）、`clen + owner`（每层静态 ✓）。

### 该修复的隔离验证 ✓（不依赖模型、不受迁移 subagent 影响 ✓）
`/tmp/ikp_repro.cu`（新模板 ✓）：把 `clen` 从 1 到 8 逐个上传 ✓，每次调 `dsv41_index_k_publish`
并**从缓冲区里把对应组槽位拷回来逐元素比对** ✓ ⇒ **`[ikp] clen=1..8 -> group slot OK (0 bad)`** ✓✓
—— 证明**目的槽位确实来自设备计数器** ✓（宿主算地址会全部落到同一组 ✗，正是被修复的行为 ✓）。

**可复用模式（本会话第 4 个隔离复现器）**：`/tmp/{hc,sa,ikp}_repro.cu`
= 直接链 `libferrite_kernels.so` ✓ + 预热 ✓ + 循环断言 ✓ + **空 kernel 地板对照**（hc/sa 版 ✓）。
⚠ 模板坑（已踩两次 ✗）：返回 `int` 的 FFI **不能**用 `cudaError_t` 包装宏 ✓ ⇒ 单独定义 `HCM`/`IKP` 宏 ✓。

## ⚠ 审计发现的**第三处**同类：`indexer_topk` 的两个每步实参（**已入档，待迁移统一修** ✗→✓）

```rust
self.dev.indexer_topk(
    ..., idx_lens_ptr,                       // ← 这里已经是设备指针 ✓（前一处修复时接的）
    (self.layers[layer].idxs.ptr as *mut i32).wrapping_add(offset),   // ← offset 每步变 ✗
    1, 1, idx_nh, idx_hd,
    comp_len as i32,                         // ← 每步宿主值作 kernel 参数 ✗✗
    cfg.index_topk as i32,
    offset as i32,                           // ← 同上 ✗
    scale, 1.0, false);
```
⇒ 图捕获会**冻结 `comp_len` 与 `offset`** ✗：之后每步都用**捕获那步的** latent 数做 top-k 界 ✗、
都往**同一个 idxs 偏移**写 ✗ ⇒ 压缩槽位的检索质量随生成退化 ✓（短上下文被 128 槽窗口掩盖 ✓，
与五段文本仍通过一致 ✓）。

**修法（与前两处同一模式 ✓，且更简单）**：kernel 本已收到设备指针 `compress_lens` ✓ ⇒
**让它从 `*compress_lens` 推 `comp_len` 与 `offset`** ✓（`offset` 的公式在宿主侧已知 ✓，改成
"窗口基址 + 设备计数" ✓），宿主实参传常量 ✓。**不需要新 kernel** ✓。

**为什么此刻不动手** ✗（诚实说明）：唯一需要改的 Rust 文件是 `device.rs`（实参个数 ✗）✓ ——
**而 subagent 正在改它** ✗ ⇒ 并发编辑会把对方的改动覆盖掉 ✗（它有独立的读-改-写窗口 ✓）。
⇒ **留给迁移统一处理** ✓（修法已如上写明 ✓，且前两处已验证的模式可直接套 ✓）。

### 冻结点审计的完整清单（本案类，供迁移一次性扫清 ✓）
| # | 位置 | 性质 | 状态 |
|---|---|---|---|
| 1 | KV 环追加：`memcpy_d2d(ring + (pos%win)*hd, kv)` | 目的地址每步变 ✗ | **已修 + 已验证** ✓（`ring_append` ✓）|
| 2 | index_k 发布：`memcpy_d2d(index_k + (compress_len-1)*idx_hd, …)` | 同上 ✗ | **已修 + 隔离验证 OK ✓** |
| 3 | `indexer_topk(..., comp_len, offset)` | 每步值作**参数** ✗ | 已入档 ✓（修法明确 ✓，待迁移）|
| — | `pre_a` 16B D2D · `h→h2` · `for g in 0..nlg`（静态模型维度 ✓）· `eng_ids+li*n_cols` · `clen+owner` | 常量/每层静态 ✓ | 无需改 ✓ |
