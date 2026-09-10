# ferrite-dsv41 — DeepSeek-V4.1-Flash support

This crate is the DeepSeek-V4.1-Flash model: config, quantisation primitives,
engram hashing, every operator (CPU golden + CUDA ABI), the checkpoint
mapping/sharding and the model chain. It is deliberately self-contained so the
GLM-5.3-Flash code paths are untouched; the kernels are linked into the same
`libferrite_kernels.so` (see `kernels/cuda/build.sh`).

## Performance contract (non-negotiable)

1. **No arbitrary dequantisation.** A weight is never unpacked into a bf16/f32
   buffer and fed to a bf16 GEMM. fp8/fp4 weights stay packed in device memory
   for the whole run; `quant.rs`'s dequant helpers exist only for checkpoint
   verification and for the CPU golden path.
2. **Every large matmul is a tensor-core MMA over the native format**, and a
   quantised operand is never computed in a *different* precision:
   * routed experts are fp4 (NVFP4 / MXFP4) and run on **fp4 tensor cores** —
     `tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X` with the
     checkpoint's own e8m0 / k-block-32 scales, which is exactly the hardware MX
     layout. **Computing them through fp8 is forbidden** (user directive
     2026-09-10): it would double the expert weight bytes, and the experts are
     58% of the checkpoint and the dominant term of a decode step. The ABI
     therefore has no fp8 expert entry point at all, and a test enforces it.
   * dense weights are fp8 e4m3 *in the checkpoint* and stay fp8 —
     `mma.sync.m16n8k32.f32.e4m3.e4m3.f32` (that is their native format; the
     reference computes them the same way in `fp8_gemm_kernel`).
   * runtime activations stay in their own formats (window KV fp8/128,
     compressed KV fp4/16, indexer q,k fp4/32; the expert inputs are fp4).
3. **Block scales are applied in the epilogue**, per k-block, with a separate
   accumulator — exactly the reference `fp8_gemm_kernel` scheme
   (`acc += dot(a_k, b_k) * scale_a[row, kblk] * scale_b[nblk, kblk]`), never by
   pre-multiplying the operands.

The ABI and the semantics of every kernel are in [`kernels`](src/kernels.rs).

## What the model is, structurally

40 backbone layers + 3 DSpark draft layers (`mtp.*`); dim 5120, 64 heads,
head_dim 512, **one KV head**, q_lora 1280, o_lora 1024 with `o_groups` 8
(block-diagonal low-rank output), window 128, 384 routed + 1 shared experts
(top-k 6, sqrtsoftplus routing with a selection-only bias, route scale 1.5),
hyper-connections (hc 4, sinkhorn 20 — *identical* to GLM-5.3-Flash), engram
n-gram tables at layers 1 and 14, vision tower, fp8 dense weights at 32x32
ue8m0 blocks and **fp4 experts**.

The awkward parts, in the order they bite:

| mechanism | where | why it is awkward |
|---|---|---|
| **engram** | layers 1, 14 | ~384M-row fp8 tables (189 GiB of the 475 GiB checkpoint), n-gram hash lookups, row-sharded so every lookup needs an all-reduce |
| **layered KV compression + cross-layer sharing** | `compress_ratios`, `kv_source_layers` | only 4 layers own/publish a compressed KV; the other ratio>0 layers read that same cache, so the layer order is a hard dependency |
| **two-level indexer** | `index_source_layers`, `candidate_source_layer` | 8 layers run their own fp4 indexer; layer 20 additionally publishes block candidates consumed by 21..39 |
| **window ring** | every layer | the cache is a `window_size`-slot ring; the index rows are ring-rotated and `-1`-marked |
| **hyper-connection ordering** | every block | each sub-block's coefficients are consumed by the *next* one (`Block.forward`), see `chain.rs` |
| **DSpark** | `mtp.*` | 5-token block draft with its own attention (window + draft block), a Markov head that biases logits **sequentially**, and a confidence head |
| **fp4 experts** | all MoE layers | 275 GiB of the checkpoint; dequantising them would quadruple the per-step weight traffic |

## Layout

```
src/config.rs    released config (HF + flat layouts), layer roles, derived helpers
src/quant.rs     e2m1 / e4m3 / ue8m0 codecs, block scales, the reference bit tricks,
                 the fp4->e4m3 cast (loader/verification only)
src/engram.rs    bucket primes (verified against the reference), hash multipliers,
                 the compressed token map and the rolling-XOR hash state
src/ops.rs       CPU golden implementations of every new operator
src/kernels.rs   the CUDA ABI + the semantics of each entry point
src/chain.rs     the model chain: layer order, hc threading, MLA, MoE, top-level
                 forward and forward_spec
src/dspark.rs    the draft stage: block attention, Markov head, confidence head
src/weights.rs   checkpoint tensor map, sharding rules, safetensors header check
src/vision.rs    vision tower (ViT32 + aligner + image span embeddings)
kernels/cuda/dsv41_kernels.cu   native-format tensor-core kernels
kernels/cuda/dsv41_vision.cu    vision kernels (bf16, native bf16 MMA)
```

## Reference material

The reference implementation ships with the checkpoint and is the source of
truth for every semantic decision here:

```
/opt/dlami/nvme/models/DeepSeek-V4.1-Flash/inference/{model.py,kernel.py,engram.py,convert.py}
/opt/dlami/nvme/models/DeepSeek-V4.1-Flash/encoding/   (prompt format; not model code)
/opt/dlami/nvme/models/DeepSeek-V4.1-Flash/evaluation/ (DeepSWE agent harness; not model code)
```

## Verification status

* `cargo test -p ferrite-dsv41` — config, quantiser round-trips, engram hash
  (primes against the reference's golden values, hand-computed rolling XOR),
  every operator's CPU golden behaviour, the weight map (shapes checked against
  the released safetensors header), DSpark index generation.
* `kernels/cuda/*.cu` — compile-checked with `nvcc -arch=sm_103a` (no GPU run).
* **Not yet done**: on-device numerics against the reference model, and the
  end-to-end chain run. Those need the B300 and are the next step.

## Open items to verify on hardware

1. ~~Which fp4 MMA form exists on `sm_103a`~~ — **settled by probe** (see
   `kernels.rs`): the only warp-level MMA that compiles is the fp8
   `m16n8k32.f32.e4m3.e4m3.f32`; **every fp4 `mma.sync` form is rejected**
   ("Instruction 'mma with FP6/FP4 floating point type' not supported on
   .target 'sm_103a'", and `kind::mxf4` is an "Illegal modifier" for `mma`).
   fp4 on this part is reachable only through
   `tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X` (5th-gen
   tensor cores: tmem accumulator, smem operand + scale descriptors, mbarrier
   completion). `kind::mxf4`'s scale type is fixed to **ue8m0**, which is
   exactly the checkpoint's expert scale format -- no conversion needed.
2. The engram gather path in `chain.rs::forward` is stubbed (`// NOTE:`): the
   table lookup itself is implemented (kernels + `weights`), but the chain's
   per-layer wiring needs the real table buffers to be threaded through.
3. Engram hash multipliers are stored as constants with provenance (see
   `engram.rs`): reproducing numpy's SeedSequence/PCG64 from scratch is not
   worth the risk when the values are model constants.
4. Vision: the tower is wired but not part of the text-only critical path.
