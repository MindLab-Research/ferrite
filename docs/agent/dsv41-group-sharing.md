# DSV41: the KV / selection GROUP-SHARING contract (and the three bugs it hid)

Written 2026-09-14 after the "count 1..100" bisection. Everything here is
measured against `/opt/dlami/nvme/dsv41_ref/model.py` and the reference config.

## The reference's structure (config + model.py)

`config_text.json`:

```
window_size          = 128
index_topk           = 512
compress_ratios      = [0,0, 2 x18 (L2..L19), 1 x20 (L20..L39), 0,0,0]
kv_source_layers     = [2, 8, 14, 20]
index_source_layers  = [2, 8, 14, 20, 24, 28, 32, 36]
candidate_source_layer = 20,  candidate_topk_blocks = 2048, candidate_block_size = 8
```

Three DIFFERENT things are shared across a layer group, each from a different
owner:

| thing | owner | reference site |
|---|---|---|
| sliding-window KV | **nobody** — every layer computes its own from its own `wkv` | `Attention._window_kv` (model.py:700-720), called for every layer |
| compressed KV (`compress_kv`) | the **kv source** of the group (2/8/14/20) | published at model.py:746-748, read back at :763 `return shared_attn.compress_kv[:bsz,:compress_len]` |
| selection (`topk_idxs`) | the **most recent index source** (2/8/14/20/24/28/32/36) | model.py:725-726 `if not self.is_index_source: return shared_attn.topk_idxs`; written at :736 |

and the two KV sources are concatenated per layer before ONE attention call:

```python
kv, topk_idxs = self._window_kv(...)                 # this layer's own window
if self.compress_ratio:
    compress_kv, compress_idxs = self._compress_kv(...)
    kv = torch.cat([kv, compress_kv], dim=1)         # model.py:779-787
    topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)
o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
```

`compress_len = (start_pos + seqlen) // ratio` (model.py:744-746) — derived from
the position, not from a stored counter.

**The two halves of the model differ**: for the ratio-2 group `kv_source ==
index_source == {2,8,14}`, so "kv owner" and "most recent index source" coincide;
for the ratio-1 group `kv_source = {20}` while `index_source =
{20,24,28,32,36}` — so layers 25..27 must read their **compressed KV from 20**
but their **selection from 24** (and 29..31 from 28, ...). Any code that conflates
the two roles is right in the first half and wrong in the second.

## Bug 1 — a consumer never saw the compressed rows (fixed, 05bfd7bf)

`compressor_*` runs only for a `kv_source` (`chain_dev.rs` `compress_on`, called
under `cfg.is_kv_source(layer)`), and it writes **its own layer's** ring
(`cache.ring.ptr` where `cache = &self.layers[layer]`). Nothing ever writes a
consumer's ring. On the read side `owner = if ring_owner_shared() { kv_owner(layer) }
else { layer }` and `DSV41_RING_OWNER` defaults **false**, so a consumer read its
own (empty) `[win, win+clen)` — and `clen[layer]` for a consumer is never advanced
either, so `sparse_attn` received `clen = 0` and attended **window-only**.

Measured (layer 3, pos 116, `DSV41_GT_LAYER=3`): ours
`kind34 = [win=128, index_topk=512, scale=0.044194, clen=0]` versus the reference's
`sparse_attn` call with `kv = (1, 186, 512)` = 128 window + 58 compressed.

Fix: `kv_src = kv_owner(layer)` for the ring/clen, and `DSV41_RING_COMP_COPY`
(default ON) copies the source's `[win, win+clen)` block into the consumer's ring
(the reference's `cat`).

## Bug 2 — a consumer used the wrong selection (fixed, 05bfd7bf)

`idxs_ptr` pointed at the kv owner's (or, with the per-layer ring, at the
consumer's OWN) buffer, and the branch chain keyed the "inherit" case on
`!owns_kv` — which is never true when the ring is per-layer. Every consumer
therefore ran the **recency placeholder** into its own buffer instead of reading
the index source's `topk_idxs`.

Fix: `sel_src = if is_index_source(layer) { layer } else { index_source_for(layer)
.unwrap_or(kv_src) }` (`Dsv41Config::index_source_for` = the most recent index
source at or before the layer), `idxs_ptr = layers[sel_src].idxs`, the inherit
branch keys on `sel_src != layer`, and `ph_ok` gained `sel_src == layer` so the
fused placeholder can never clobber the shared selection.

## Bug 3 — `MOE_DUAL` forked while the routed path was the sequential one (fixed)

The `dual` predicate claimed to mirror `batched` but omitted `!expert_act_e4m3()`
and `!expert_cublas()`. Both force the routed chain onto the **sequential** loop,
which writes `s.ex_act[0 .. 2*320]` on the main stream — exactly the region the
shared half's `gemm_fp8_mx2_on` writes on the side stream. A real data race at
layer 0 / pos 0: the shared term measured 0.175694 (correct, reference 0.175733)
when the race lost and 0.148034 when it won; adding any D2H probe in between
flipped it. Mirror the predicate EXACTLY.

## Not a bug: engram (exonerated)

`engram_apply` matches `Engram.forward` term for term; the earlier "the hash ids
differ" verdict was an artifact of comparing a **pos-0 reference dump** against
our record that had been overwritten by the last step (`std::fs::write` in a
per-step probe writes the LAST call). With the dump keyed by `(step, pos, token)`
the ids are identical at every position (23/23) and the gathered rows are
bit-exact (`max diff = 0`); only the `wkv` output differs, by 1.5e-3, which is
the deliberate f32-direct read versus the reference's bf16→fp8 round trip.

## Instrumentation added for this (all default OFF)

| flag | meaning |
|---|---|
| `DSV41_GT_XDUMP=<path>` | op-level dumps, `[u64 step][u64 kind][f32 x n]` |
| `DSV41_GT_XDUMP_POS=<pos>` | restrict the dump to ONE step. **Mandatory** for a 120-step run: an unrestricted dump writes ~10 MB/step (the per-layer hc kinds) and ~2000 blocking D2H copies per step |
| `DSV41_GT_LAYER=<n>` | the layer the attention-op kinds attach to (0 = default). Layer 0/1 have `compress_ratio == 0`, so the compressed path can only be seen from layer 2 |
| `DSV41_GT_HDUMP`, `DSV41_GT_LOGITS_DUMP` | per-layer residual h / per-step logits |
| `DSV41_ENG_DUMP=<path>` | engram internals per position: ids, pre-engram h, gathered rows, `wkv` output |
| `DSV41_L0DBG` | the layer-0/pos-0 segment anchors the reference prints itself |
| kind 31 / 32 / 34 / **35** | ring rows `[0,32)`, the idxs row, `[win, index_topk, scale, clen]`, and **ring rows `[win, win+32)` = the compressed latents** |

Reference-side counterpart: `ref_attn2.py` patches the module-level `sparse_attn`
and dumps `(q, kv, idxs)` for a chosen `(layer, pos)` — the strict split of "our
inputs differ" versus "our operator differs".

## The rope table is PER-LAYER (found 2026-09-14; this was the real "KV/rope" bug)

The reference builds ONE `freqs_cis` per layer (`Attention.__init__`,
model.py:685-698), forked on `compress_ratio`:

```python
if self.compress_ratio:
    original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta  # 65536, 160000
else:
    # disable YaRN and use base rope_theta in pure sliding-window attention
    original_seq_len, rope_theta = 0, args.rope_theta                               # 0, 10000
freqs_cis = precompute_freqs_cis(rope_head_dim, max_seq_len, original_seq_len, rope_theta,
                                 factor, beta_fast, beta_slow)
```

and that ONE table is used by FOUR rope sites:

| site | reference line |
|---|---|
| the q rope | `apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])` (:551) |
| the **window-KV** rope | `_window_kv(x, freqs_cis, start_pos)` (:700) → `apply_rotary_emb(kv[..., -rd:], freqs_cis)` (:706) |
| the indexer's q/k rope | `self.indexer.freqs_cis = self.freqs_cis` (:733-734) |
| the compressor's latent rope | (:754-756) |

So layers 0 and 1 (`compress_ratio == 0`) rotate with theta 10000 and NO YaRN,
while **every layer 2..39 rotates with theta 160000 plus the YaRN blend** — for
q, the window KV, the indexer and the latent alike. We applied the main table to
q and the window KV for EVERY layer (the latent and the indexer were already on
the compression table), i.e. 38 of 40 layers were rotated with a 16x-wrong theta
and no YaRN.

**Measured with the dumps already in hand (layer 2, pos 116, no GPU needed)**:

```
q (post-rope)    rel = 6.87e-01   with MATCHING rms (ours 1.640 vs ref 1.631)
   -> a PHASE error, not a magnitude error: the rope-table signature.
q rope lanes 448:452  ours [-3.45312, -0.91016,  0.25195, -0.49414]
                      ref  [-3.39062, -0.92578, -0.05518, -0.42383]
window-KV rows   rel = 6.4e-02 / 9.2e-02 / 8.1e-02 / 1.2e-01
   -> small on purpose: the rope only touches the trailing 64 of 512 lanes.
```

**Fix (5cc28b09)**: `comp_rope` (a `Cell<bool>`, set in `attention()` from
`compress_ratio(layer) > 0` **before any rope runs**) plus `rope_cos()` /
`rope_sin()` accessors that all nine shared rope call sites now use — the three
helpers (`lin_rope`, `lin_rope_norm`, `lin2_rope`, i.e. all three q-rope paths),
`attention()`'s explicit q / window-KV / o-rope sites, and the kv-rope fallbacks.
The indexer's `lin_rope` fallback therefore lands on the compression table by
construction, which is what `indexer.freqs_cis = self.freqs_cis` means.

## Next fix: the compressed latent is missing its fp4 round-trip

The reference quantises each compressed latent **after the rope and before the
cache write** (`model.py:758-761`):

```python
apply_rotary_emb(latent[..., -rope_head_dim:], freqs)
# Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0.
fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
self.compress_kv_cache[:bsz, start_pos // ratio : ...] = latent
```

and the two scale derivations are DIFFERENT (`kernel.py:129-184`):

```python
fp4_max = 6.0
if scale_dtype == FP8:                  # compressed KV: block 16, E4M3 scale
    amax = max(amax, 6 * 2**-9)
    s    = float8_e4m3(amax / 6)        # NOT a power of two
else:                                   # indexer: block 32, E8M0 (power-of-two)
    amax = max(amax, 6 * 2**-126)
    s    = fast_round_scale(amax, 1/fp4_max)
# inplace writes the DEQUANTISED value back:
Y = cast_e2m1(clamp(x / s, -6, 6)) * s
```

`dsv41_idx_fp4_rt` (`dsv41_kernels.cu:3419`, block 32 + `fast_round_scale(amax, 1/6)`)
is exactly the **indexer** arm; it is called only on `idx_k`/`idx_q`
(`chain_dev.rs:4819`, `:4919`). The **compressed latent gets no round-trip at
all** — `compress_commit_kernel` (`dsv41_glue.cu:1053-1079`) ropes and stores, and
its verbatim copy inside `compressor_fused_kernel`'s stage 3
(`dsv41_kernels.cu:3385-3408`) does the same. Both bodies must change together
(the file says so explicitly).

Implementation sketch: inside the store loop, after the rope, each warp's 32
lanes cover 32 consecutive columns = exactly two 16-element blocks, so a
half-warp `__shfl_xor_sync` tree (masks 8/4/2/1, **exact masks only** — the
`__shfl_xor_sync` mask-mismatch trap this project already hit once) gives the
block amax; then `s = (float)__nv_fp8_e4m3(amax / 6.f)` (with `amax` floored at
`6 * 2^-9`), `d = clamp(x/s, +-6)`, the e2m1 nearest (the `mags[8]` table
`idx_fp4_rt_kernel` uses, first-on-ties), and the write-back is `d * s`.

**Measured (2026-09-14, layer 2, pos 116, before the fix)** — `dsv41_gt_layer=2`
`gt_xdump_pos=116`, our kind 35 (ring rows `[128,160)`) versus the reference's
`sparse_attn` `kv` rows `[128,160)` (ref_attn2 kind 11 = `[window(128) |
compress(clen)]`, all 186 rows dumped):

```
our kind34 = [win=128, index_topk=512, scale=0.044194, clen=58]   # clen=58 = the reference's 58 compressed rows
row 128: ours rms=0.19792  ref rms=0.19641  maxdiff=0.106
row 129: ours rms=0.38329  ref rms=0.38181  maxdiff=0.148
row 130: ours rms=0.38635  ref rms=0.38707  maxdiff=0.149
row 131: ours rms=0.39126  ref rms=0.39009  maxdiff=0.157
over 32 rows x 512: maxabs=0.208, rel=1.02e-01
```

⇒ the rows ARE the same latents (per-row rms matches to 4 digits) and the 10.2%
is the fp4 signature (e2m1 has a 1-bit mantissa, ~25% per-element worst case,
~10% RMS) — i.e. the missing round-trip is the whole of this difference.

⚠️ **Parser trap that produced a bogus 146% first**: the reference's `kv` is a
flat `[186 * 512]` array, so the compressed rows are
`kv.reshape(-1, 512)[128:]` — a ROW slice. Slicing the flat vector
(`kv[128:]`) starts 128 *elements* into row 0 and compares the window block
against our compressed rows, which reports a meaningless ~146% (and a
`reshape` failure, since 95104 is not a multiple of 512). Always reshape before
slicing when the reference dump is flat. (`compkvcmp.py` carried this bug for
one round; it now reshapes first.)

**Measured AFTER the fix (same layer/pos, same command):**

| | rel | maxabs |
|---|---|---|
| before | 1.02e-01 | 0.208 |
| after (block-16 E4M3 RT in place) | 8.02e-02 | 0.312 |

Only 10.2% -> 8.0%, and the reason is a SECOND missing boundary: the reference
feeds the round-trip a value that is **already bf16** (`Compressor.forward`
returns `self.norm(kv.to(dtype))`, dtype = bf16), while our
`compressor_pool_kernel` writes the norm result as raw f32. e2m1 has a 1-bit
mantissa, so a 0.4% input difference flips near-boundary elements by a whole
step (1.0 -> 1.5, 25%) and the RMS difference survives. **Next: round the pooled
latent to bf16 before the rope/RT** (the `norm(kv.to(dtype))` boundary).

**And the round-trip is NOT what breaks the token count** — measured with the
same binary and prompt:

```
before: 1..60 correct, then (61,31),(62,32)                       -> 62 numbers
after : 1..60 correct, then (61,31),(62,32),(63,33),(64,44),(65,45) -> 65 numbers
```

The correct prefix and the first wrong number are UNCHANGED (still the 61st),
so the compressed latents' 8-10% error is not the cause at that step — which
fits the geometry: at pos ~135 the sliding window still covers all but the first
seven tokens, so the compressed rows are a small correction there. Look
elsewhere (the window KV's own `win_kv_quant_rt`, or that step's hc/MoE
precision).

## Known remaining difference (inert at short context, matters at long)

Our `indexer()` ropes the indexer's k and q with the **main** rope tables
(`self.cos/self.sin`, theta=10000, no YaRN), while the reference uses the layer's
`freqs_cis`, which for a `compress_ratio > 0` layer is the **compression** rope
(theta=160000, YaRN, model.py:680-698 → `self.indexer.freqs_cis = self.freqs_cis`).
It is inert while `clen <= index_topk`: then the top-k selects **every** candidate,
so the score ORDER cannot change the selected SET (which is why layer 2's idxs
matched the reference 186/186 at pos 116 with `clen = 58 < 512`). It becomes a real
difference once `clen > index_topk`, i.e. past pos ≈ `index_topk * ratio`.
