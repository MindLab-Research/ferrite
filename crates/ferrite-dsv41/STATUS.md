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
