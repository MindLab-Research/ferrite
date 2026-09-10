# DeepSeek-V4.1-Flash — performance model

All numbers are derived from the released checkpoint's safetensors headers and
the config, not from a run. They exist to fix the optimisation order: on this
model the arithmetic is cheap and the **weight bandwidth** is everything.

## Weight footprint (475.2 GiB total, 48 shards)

| component | size | dtype | note |
|---|---|---|---|
| routed experts (384 x 40 layers x 3 matrices) | **275.7 GiB** | fp4 e2m1, I8-packed | the dominant term |
| engram n-gram tables (layers 1, 14; ~384M rows each) | **189.1 GiB** | fp8 e4m3 + e8m0 | read only by 24 row-gathers per token |
| dense attention / norms / embeddings | 8.9 GiB | fp8 e4m3, 32x32 ue8m0 | |
| vision tower | 0.9 GiB | bf16 | off the text path |
| MTP / DSpark | 0.7 GiB | fp8 | |

Per expert: `w1 [2304, 5120]` fp4 = 5.9 MB, and the same for `w2`/`w3` → 17.7 MB
per expert. At TP8 each rank holds 48 of the 384 experts → 850 MB per layer,
34.5 GiB for the model. Adding the engram shard (23.6 GiB) and the dense weights
(1.1 GiB) puts a rank at **~60 GB**, comfortably inside a 288 GB B300 (TP4 would
be ~120 GB and also fit).

## Per-step traffic (the number that sets the step time)

Batch 16, top-k 6 → 96 assignments over ~85 distinct experts per layer (the
duplicate rate is only 1.13x, so there is almost no reuse to exploit).

* per rank per layer: ~10.6 distinct experts x 17.7 MB ≈ **188 MB**
* x 40 layers ≈ **7.5 GB per rank per step**
* at ~7.6 TB/s that is **≈1 ms/step just for expert weights**

That is the floor for the MoE, and it is why **fp4 must be consumed natively**:
feeding the same weights as bf16 (4x the bytes) costs ~4 ms/step, and even the
lossless fp4→e4m3 re-encode (2x) costs ~2 ms/step. The engram's contribution is
tiny by comparison (~314 MB of `wkv` + 24 gathered rows per token), and the
attention is cheap because there is **one KV head** with a 128-slot window plus
512 compressed positions. There is no linear attention (no GDN) in this model
at all: the compute pressure is far lower than GLM-5.3-Flash's, which had the
GDN recurrence on top.

## What that implies for the kernel portfolio

1. **fp4 expert GEMM is the whole ball game.** Native MXFP4 on sm_103a means
   `tcgen05.mma ... kind::mxf4` (the checkpoint's e8m0 / k-block-32 scales are
   exactly the hardware MX layout) with tensor-memory accumulators. See
   `crates/ferrite-dsv41/src/kernels.rs` for the probe result and why the
   `mma.sync` fp4 forms do not exist on this part.
2. **Dense fp8 GEMMs** use the proven
   `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` with the ue8m0 32x32
   scales applied per k-block in the epilogue.
3. **Engram** is a latency problem, not a bandwidth one: 24 row-gathers of 256
   bytes per token per engram layer, from a row-sharded table, followed by an
   all-reduce. Keep the gathers in flight (one descriptor per row), do not
   serialise them.
4. **The indexer / candidate path** adds small kernels per index-source layer
   (8 of them) plus a top-512 selection; fuse the score, the candidate mask and
   the top-k into one pass.
5. **Cross-layer sharing is free bandwidth-wise but constrains scheduling**: the
   4 `kv_source` layers must publish before their consumers read, which is
   compatible with the in-order layer walk but forbids layer-level parallelism
   across a source boundary.

## What is *not* worth optimising first

* The attention math: one KV head, a 128-slot window and 512 compressed
  positions make it a small fraction of the step.
* The vision tower: off the text path; it only matters for multimodal requests.
* The 8.9 GiB of dense weights: ~1.1 GiB per rank read once per step.

## Targets to keep honest

* Non-MTP, batch 16: the MoE floor above (~1 ms) plus attention/hc/engram →
  a step in the low single-digit ms is the realistic target for a *correct*
  implementation; anything in the tens of ms means a dequantising or
  non-tensor-core path slipped in.
* The DSpark draft costs one extra block forward (block size 5) per step; its
  value depends on the acceptance rate, which is a modelling question, not a
  kernel one.
