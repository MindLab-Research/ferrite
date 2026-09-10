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
