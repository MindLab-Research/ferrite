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

## Known remaining difference (inert at short context, matters at long)

Our `indexer()` ropes the indexer's k and q with the **main** rope tables
(`self.cos/self.sin`, theta=10000, no YaRN), while the reference uses the layer's
`freqs_cis`, which for a `compress_ratio > 0` layer is the **compression** rope
(theta=160000, YaRN, model.py:680-698 → `self.indexer.freqs_cis = self.freqs_cis`).
It is inert while `clen <= index_topk`: then the top-k selects **every** candidate,
so the score ORDER cannot change the selected SET (which is why layer 2's idxs
matched the reference 186/186 at pos 116 with `clen = 58 < 512`). It becomes a real
difference once `clen > index_topk`, i.e. past pos ≈ `index_topk * ratio`.
