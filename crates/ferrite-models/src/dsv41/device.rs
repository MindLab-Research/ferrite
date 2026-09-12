//! Device layer for DeepSeek-V4.1-Flash.
//!
//! The generic device primitives — dlopen of libcudart/libcublas and of the
//! kernel `.so`, the stream, byte-granular allocation, H2D/D2H/D2D copies,
//! peer access and the CUDA-graph capture primitives — live in the SHARED
//! [`ferrite_kernel::devrt`] layer. This file keeps only what is
//! model-specific: the DeepSeek kernel ABI table and its typed launch wrappers.
//!
//! A handful of GLM kernels are still reused **read-only** where the geometry
//! is identical: the hyper-connection chain (`ferrite_hc_pre` / `ferrite_hc_post`
//! — hc_mult 4 / sinkhorn 20 / eps 1e-6 is the same configuration), `rmsnorm`,
//! the embedding expander, `ferrite_add` and the f32<->bf16 casts. Reuse means
//! calling the existing symbol; no GLM source is touched.

use std::ffi::{c_int, c_long, c_uint, c_void};

use ferrite_types::{FerriteError, Result};

pub use ferrite_kernel::devrt::{CuStream, DevBuf, DevRuntime};

/// Resolve a REQUIRED kernel symbol in the loaded `.so`.
macro_rules! km {
    ($rt:expr, $name:expr) => {
        unsafe { std::mem::transmute_copy(&$rt.kernel_sym($name)?) }
    };
}
/// Resolve an OPTIONAL kernel symbol (staged bring-up: absent = None).
macro_rules! ko {
    ($rt:expr, $name:expr) => {
        $rt.kernel_sym_opt($name)
            .map(|p| unsafe { std::mem::transmute_copy(&p) })
    };
}

/// The DeepSeek kernels (this crate's own) plus the read-only GLM reuse set.
struct Kernels {
    // ---- dsv41 (this crate) ----
    gemm_fp8_mx: unsafe extern "C" fn(
        *const u8, *const f32, *const u8, *const u8, *const f32, *mut f32,
        c_int, c_int, c_int, CuStream,
        // AR v5 store fusion (attn wo_b, M=1 only): a non-null staging table makes
        // the epilogue also store the row partial into every peer's slot. Null/0
        // keeps the old behaviour.
        *const *mut f32, *const c_uint, c_int, c_int, c_int,
        // B1 (quantised row compression, M=1 only): non-null `xq`/`xsc` make the
        // epilogue ALSO emit the fp8 e4m3 bytes of `out` and their per-32-block
        // scales. The launcher then needs n % 32 == 0 and raises the block to 32
        // warps; `xq` must not alias `a`. Null keeps the old path bit for bit.
        *mut u8, *mut f32,
    ) -> c_int,
    /// Two same-activation fp8 projections in ONE gemv launch: rows below n1 map
    /// to the first family, the rest to the second, both sharing the staged
    /// activation. Returns cudaErrorInvalidValue when the shape does not fit, so
    /// the caller falls back to two gemm_fp8_mx calls.
    gemm_fp8_mx2: unsafe extern "C" fn(
        *const u8, *const f32,
        *const u8, *const u8, *const f32, *mut f32, c_int,
        *const u8, *const u8, *const f32, *mut f32, c_int,
        c_int, CuStream,
    ) -> c_int,
    /// RoPE fusion: the M=1 GEMV whose epilogue also rotates the trailing
    /// `rope_rd` lanes of each head (`dsv41_apply_rope`'s rotation, term for
    /// term). Optional: a stale `.so` has no entry and the caller keeps the
    /// (gemm_fp8_mx, apply_rope) pair. Returns 2 when the shape cannot take it
    /// (never 1, which is cudaErrorInvalidValue).
    gemm_fp8_mx_rope: Option<
        // ABI: the stream is the LAST parameter, matching the kernel's
        // `dsv41_gemm_fp8_mx_rope(..., cudaStream_t s)` exactly. It must NOT sit
        // after (n, k) the way `gemm_fp8_mx` does - that symbol has optional
        // trailing args, this one does not, and a wrong position silently
        // shifted every rope arg so the kernel read `rope_rd == 0`, declined,
        // and the caller fell back to the standalone apply_rope.
        unsafe extern "C" fn(
            *const u8, *const f32, *const u8, *const u8, *const f32, *mut f32,
            c_int, c_int,
            *const f32, *const f32, *const c_int, c_int, c_int, c_int, c_int, c_int, c_int,
            CuStream,
        ) -> c_int,
    >,
    /// RoPE fusion for the two-family GEMV: family 1 rotates with `rope_hd1`,
    /// family 2 with `rope_hd2` (the wq_b / idx_wq_b pair). Optional, like
    /// `gemm_fp8_mx_rope`.
    gemm_fp8_mx2_rope: Option<
        // ABI: stream LAST, as in `dsv41_gemm_fp8_mx2_rope(..., cudaStream_t s)`
        // (see the note on gemm_fp8_mx_rope - same mis-ordered-stream bug).
        unsafe extern "C" fn(
            *const u8, *const f32,
            *const u8, *const u8, *const f32, *mut f32, c_int,
            *const u8, *const u8, *const f32, *mut f32, c_int,
            c_int,
            *const f32, *const f32, *const c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int,
            CuStream,
        ) -> c_int,
    >,
    /// NORM_FUSE: the M=1 rope GEMV whose PROLOGUE produces the fp8 activation it
    /// consumes. Instead of reading a pre-quantised (`a`, `a_scale`) pair it takes
    /// the RAW f32 row plus the norm weight and eps, and reduces / normalises /
    /// encodes it in shared memory with `rmsnorm_q_kernel`'s arithmetic, term for
    /// term. The standalone `rmsnorm_q` launch between `qr`'s producer and the
    /// wq_b gemv therefore disappears (one launch + one graph node per attention).
    /// Optional: a stale `.so` has no entry and the caller keeps the pair.
    /// ABI: stream LAST, as in `dsv41_gemm_fp8_mx_rope_norm(..., cudaStream_t s)`.
    gemm_fp8_mx_rope_norm: Option<
        unsafe extern "C" fn(
            // qr_raw (f32 row), qr_w (norm weight), qr_eps
            *const f32, *const f32, f32,
            // w, w_scale, bias, out, n, k
            *const u8, *const u8, *const f32, *mut f32, c_int, c_int,
            // rope_cos, rope_sin, rope_base, mul, off, step, inverse, rd, hd
            *const f32, *const f32, *const c_int, c_int, c_int, c_int, c_int, c_int, c_int,
            CuStream,
        ) -> c_int,
    >,
    /// A5: the same M=1 w2 GEMV with the trailing `ferrite_add` folded into its
    /// epilogue (`out += w @ a`). A separate symbol, so a stale `.so` simply has
    /// no entry and the caller keeps the gemm_fp8_mx + add_inplace pair. Returns
    /// 2 when the shape cannot use the GEMV (never 1 — cudaErrorInvalidValue).
    gemm_fp8_mx_add: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *const u8, *const u8, *const f32, *mut f32,
            c_int, c_int, c_int, CuStream,
        ) -> c_int,
    >,
    /// wo_b f32-activation GEMV: the M=1 GEMV that reads the RAW f32 activation
    /// (`dsv41_gemm_fp8_mx_f32`) instead of an fp8 (`a`, `a_scale`) pair, so the
    /// `quant1(s.wo)` launch between wo_a and wo_b disappears. A separate symbol,
    /// so a stale `.so` simply has no entry and the caller keeps the
    /// (quant1, gemm_fp8_mx) pair. Returns 2 when the shape/mode cannot use it
    /// (never 1 — cudaErrorInvalidValue, the round-42 collision).
    /// ABI: stream LAST — `dsv41_gemm_fp8_mx_f32(a_f32, w, w_scale, bias, out, n,
    /// k, s)` (this symbol has no C++ default tail args, so it does NOT follow
    /// `gemm_fp8_mx`'s "stream after the shape" layout).
    gemm_fp8_mx_f32: Option<
        unsafe extern "C" fn(
            *const f32, *const u8, *const u8, *const f32, *mut f32, c_int, c_int, CuStream,
        ) -> c_int,
    >,
    /// swapAB M=1 GEMV (`dsv41_gemm_fp8_swapab`): the M=1 decode as a tensor-core
    /// GEMM with the WEIGHT on M (16 output rows) and the ACTIVATION on B's
    /// column 0. It replaces the SIMT `gemm_fp8_gemv_kernel` for the plain fp8
    /// path -- no a32 materialisation, no LUT decode, no FFMA consume chain.
    ///
    /// NOT bit-identical to the SIMT gemv (the tensor core sums raw fp8 products
    /// and scales per k block, where the SIMT kernel does per-element
    /// `(a*sa)*(w*sb)` and reduces by shuffles). Same scheme as the m>1 dense MMA
    /// path; parity is judged by text/fingerprint, not bit equality.
    ///
    /// `partial` / `ctr` are the LAST-BLOCK REDUCTION scratch for the ks > 1
    /// (K-split) path: `partial` is `ks * n` f32 slots (one per (K partition,
    /// output row)) and `ctr` is one u32 arrival ticket per 16-row tile
    /// (`n / 16`). The kernel writes `out` itself through the elected block, so it
    /// no longer memsets `out` and no longer needs `out` pre-zeroed; the caller
    /// only has to ensure `ctr` is ZERO before the first call (the kernel
    /// self-resets it, so a captured graph replays clean). Both may be null when
    /// the shape reduces to ks == 1 (that path is a plain store).
    ///
    /// Optional: a stale `.so` has no entry and the caller keeps the SIMT gemv.
    /// Returns 2 when the shape cannot take it (n % 16 != 0 or k % 32 != 0; never
    /// 1 -- cudaErrorInvalidValue, the round-42 collision).
    /// ABI: stream LAST, like `dsv41_gemm_fp8_mx_f32`.
    gemm_fp8_swapab: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *const u8, *const u8, *const f32, *mut f32, c_int, c_int,
            *mut f32, *mut c_uint, CuStream,
        ) -> c_int,
    >,
    /// chain-pair-grid-sync: the wo_a -> wo_b pair as ONE grid-sync launch
    /// (`dsv41_gemm_fp8_wo_pair`). Phase 1 is the wo_a fp8 GEMV (mode 4 + a32),
    /// phase 2 the wo_b f32-activation GEMV, joined by a sense-reversing
    /// device-wide barrier instead of a stream edge -- so the two GEMVs of every
    /// layer's output projection become one launch (40 saved per step).
    ///
    /// `bar` is a persistent 2 x u32 device buffer, `[arrive, sense]`, zeroed
    /// once at init; the kernel self-resets it every launch, so a captured graph
    /// replays correctly. A separate symbol, so a stale `.so` simply has no entry
    /// and the caller keeps the two launches. Returns 2 when the shape/arm cannot
    /// use it (never 1 — cudaErrorInvalidValue, the round-42 collision).
    /// ABI: stream LAST, like `dsv41_gemm_fp8_mx_f32`.
    gemm_fp8_wo_pair: Option<
        unsafe extern "C" fn(
            // phase 1: fp8 activation, wo_a weights, k-blocks
            *const u8, *const f32, *const u8, *const u8, *const f32, c_int, c_int,
            // phase 2: wo_b weights, shape
            *const u8, *const u8, *const f32, c_int, c_int,
            // intermediate f32 row / final f32 row / barrier [arrive, sense]
            *mut f32, *mut f32, *mut c_uint,
            CuStream,
        ) -> c_int,
    >,
    /// chain-pair-batch 链2: the shared expert's (w1w3 -> swiglu -> w2) chain as
    /// ONE grid-sync launch (`dsv41_gemm_fp8_sh_pair`). Phase 1 computes the
    /// gate/up PAIR per warp (both weight rows over the same fp8 activation,
    /// k = dim) and applies the swiglu epilogue, emitting the f32 activation AND
    /// the fp8 pair; a sense-reversing device-wide barrier joins it to phase 2,
    /// the w2 M=1 GEMV (k = the shared expert's local inter = phase 1's row
    /// count). Three launches (w1w3, swiglu, w2) become one.
    ///
    /// The barrier state is the kernel's OWN module-level `[arrive, sense]` pair
    /// (`g_sh_arrive`/`g_sh_sense`), deliberately separate from `wo_bar`; it is
    /// zero-initialised by the module loader and self-resetting per launch, so no
    /// device buffer travels through this ABI.
    ///
    /// `aq`/`aqsc` is the phase-1 fp8 output and MUST be disjoint from `a`/`a_scale`
    /// (phase 1 reads the `xn` quant while it writes the swiglu pair; one buffer
    /// would be a cross-block race inside a single grid-sync launch).
    ///
    /// A separate symbol, so a stale `.so` simply has no entry and the caller
    /// keeps the three launches. Returns 2 when the shape/arm cannot use it
    /// (never 1 — cudaErrorInvalidValue, the round-42 collision).
    /// ABI: stream LAST.
    gemm_fp8_sh_pair: Option<
        unsafe extern "C" fn(
            // phase 1: fp8 activation (k = dim), w1/w3 weights + scales, limit, n1/k1
            *const u8, *const f32, *const u8, *const u8, *const u8, *const u8, f32, c_int, c_int,
            // phase 1 outputs: f32 activation, fp8 pair, per-32-block scale
            *mut f32, *mut u8, *mut f32,
            // phase 2: w2 weights + scale, row count, out
            *const u8, *const u8, c_int, *mut f32,
            CuStream,
        ) -> c_int,
    >,
    quant_fp8: unsafe extern "C" fn(
        *const f32, *mut u8, *mut f32, c_int, c_int, c_int, c_int, CuStream,
    ) -> c_int,
    quant_fp4: unsafe extern "C" fn(
        *const f32, *mut u8, *mut f32, c_int, c_int, c_int, c_int, CuStream,
    ) -> c_int,
    expert_gate_up_fp4: unsafe extern "C" fn(
        *const u8, *const f32, *const u8, *const u8, *const u8, *const u8, *mut f32,
        c_int, c_int, c_int, f32, CuStream,
    ) -> c_int,
    expert_down_fp4: unsafe extern "C" fn(
        *const f32, *const u8, *const u8, *const f32, *mut f32,
        c_int, c_int, c_int, CuStream,
    ) -> c_int,
    engram_hash: unsafe extern "C" fn(
        *const i32, *mut i64, *const i64, *const i64, *const i64, *const i32, *const u8,
        *mut i64, c_int, c_int, c_int, c_int, c_int, c_int, c_int, i64, CuStream,
    ) -> c_int,
    engram_gather: unsafe extern "C" fn(
        *const u8, *const u8, *const i64, *mut f32,
        c_int, c_int, c_int, i64, i64, CuStream,
    ) -> c_int,
    sparse_attn: unsafe extern "C" fn(
        *const f32, *const f32, *const f32, *const i32, *mut f32,
        c_int, c_int, c_int, c_int, *const c_int, c_int, c_int, f32, CuStream,
    ) -> c_int,
    // P1 (DSV41_SPARSE_OROPE): sparse attention + inverse o-rope + fp8 emission
    // in ONE launch, replacing the `sparse_attn` + `apply_rope_q` (+ `quant1`)
    // triple. Same block geometry as `sparse_attn_pf_kernel` (grid=(b*m,h),
    // block=128, one head's full d row per block); the o row is normalised into
    // shared memory and the rope + fp8 passes run in the same launch, so the
    // emitted bytes are `dsv41_quant_fp8(roped o)`'s. Optional: an older .so
    // without it keeps the three-launch path. Returns 0 on success, 1/2/3 when
    // the shape declines (caller then runs the fallback).
    sparse_attn_orope: Option<unsafe extern "C" fn(
        *const f32, *const f32, *const f32, *const i32, *mut f32,
        c_int, c_int, c_int, c_int, *const c_int, c_int, c_int, f32,
        *const f32, *const f32, *const c_int, c_int, c_int, c_int, c_int, c_int, c_int,
        *mut u8, *mut f32, CuStream,
    ) -> c_int>,
    indexer_topk: unsafe extern "C" fn(
        *const f32, *const f32, *const f32, *const u8, *const i32, *mut i32,
        c_int, c_int, c_int, c_int, c_int, c_int, c_int, f32, f32, c_int, CuStream,
    ) -> c_int,
    candidate_blocks: unsafe extern "C" fn(
        *const f32, *const i32, *mut u8, c_int, c_int, c_int, c_int, CuStream,
    ) -> c_int,
    compressor: unsafe extern "C" fn(
        *const f32, *const u8, *const u8, *const u8, *const u8, *const f32,
        *mut f32, *mut f32, *mut f32, *mut i32,
        c_int, c_int, c_int, c_int, c_int, c_int, *const c_int, f32, CuStream,
    ) -> c_int,
    rope_precompute: unsafe extern "C" fn(
        *mut f32, *mut f32, c_int, c_int, c_int, f32, f32, f32, f32, CuStream,
    ) -> c_int,
    apply_rope: unsafe extern "C" fn(
        *mut f32, *const f32, *const f32, c_int, c_int, c_int, c_int, *const c_int, c_int, c_int,
        c_int, c_int, CuStream,
    ) -> c_int,
    // rmsnorm + rope on one row in a single launch (the kv chain's adjacent
    // pair). Optional: an older .so without it falls back to the two launches.
    rmsnorm_rope: Option<unsafe extern "C" fn(
        *const f32, *const f32, *mut f32, *const f32, *const f32, c_int, c_int, c_int, c_int,
        *const c_int, c_int, c_int, c_int, c_int, f32, CuStream,
    ) -> c_int>,
    // T2: rmsnorm that ALSO emits the fp8 (byte + per-32-block scale) of its own
    // normalised output, so the next fp8 activation consumer of that row (the
    // qr -> wq_b projection) skips its quant launch. Optional: an older .so
    // without it falls back to (rmsnorm, quant_fp8). Returns 0 on success, 1
    // when the shape cannot use the fused emission (caller then runs rmsnorm).
    rmsnorm_q: Option<unsafe extern "C" fn(
        *const f32, *const f32, *mut f32, c_int, c_int, f32, *mut u8, *mut f32, CuStream,
    ) -> c_int>,
    // bf16 gate + fp8 shared expert in ONE launch (same activation). Optional:
    // falls back to the separate launches.
    gemm_bf16_fp8x2: Option<unsafe extern "C" fn(
        *const c_void, *const f32, *mut f32, c_int, *const u8, *const f32, *const u8, *const u8,
        *mut f32, c_int, *const u8, *const u8, *mut f32, *const f32, c_int, CuStream,
    ) -> c_int>,
    // Cross-rank argmax over a vocabulary-sliced lm_head (DSV41_HEAD_SLICE).
    argmax_sliced: Option<unsafe extern "C" fn(
        *const f32, c_int, c_int, *mut c_int, *mut u64, *mut c_int, *const *mut u64,
        *const *mut u32, *mut c_uint, *mut u64, *const c_uint, c_int, c_int, c_long, CuStream,
    ) -> c_int>,
    hc_mixes: unsafe extern "C" fn(
        *const f32, *const f32, *const f32, *const f32, *mut f32, *mut f32, *mut f32,
        c_int, c_int, c_int, c_int, f32, CuStream,
    ) -> c_int,
    moe_route: unsafe extern "C" fn(
        *const f32, *const u8, *const u8, *const f32, *mut f32, *mut i32, *mut i32,
        c_int, c_int, c_int, c_int, f32, c_int, f32, c_int, CuStream,
    ) -> c_int,
    add_inplace: Option<unsafe extern "C" fn(*const f32, *const f32, *mut f32, c_int, CuStream) -> c_int>,
    hc_collapse: Option<unsafe extern "C" fn(*const f32, *const f32, *mut f32, c_int, c_int, c_int, CuStream) -> c_int>,
    ar_stamp: Option<unsafe extern "C" fn(*const u64, c_int, c_int, c_uint, CuStream) -> c_int>,
    ar_store: Option<
        unsafe extern "C" fn(*const u64, c_int, c_int, *const f32, i64, i64, CuStream) -> c_int,
    >,
    ar_store2: Option<
        unsafe extern "C" fn(
            *const u64, c_int, c_int, *const f32, i64, i64, i64,
            *const c_uint, c_uint, *const u64, *mut c_uint, CuStream,
        ) -> c_int,
    >,
    ar_reduce2: Option<
        unsafe extern "C" fn(
            *mut f32, *const f32, i64, i64, c_int, *const c_uint, c_uint,
            *const u64, c_int, *mut c_uint, c_int, CuStream,
        ) -> c_int,
    >,
    ar_mark: Option<unsafe extern "C" fn(*const u64, c_int, c_int, c_uint, CuStream) -> c_int>,
    gemv_bf16: Option<unsafe extern "C" fn(*const c_void, *const f32, *mut f32, c_int, c_int, CuStream) -> c_int>,
    // v2 (vectorized uint4 + K-split) M=1 GEMV, from ferrite_kernels.cu. Optional:
    // an older .so without the symbol keeps the v1 kernel above. Note the ABI
    // differs from `dsv41_gemv_bf16`: (x, w, bias, out, in_f=k, out_f=n, nrows, s).
    gemv_bf16_v2: Option<
        unsafe extern "C" fn(
            *const f32, *const c_void, *const f32, *mut f32, c_int, c_int, c_int, CuStream,
        ) -> c_int,
    >,
    // The route-fused twin of the above: the same GEMV launch with the MoE
    // route (`dsv41_route_topk`'s body) run by its LAST block. ABI adds the
    // route outputs + bias/scale/score_func and the per-call counter `ctr`
    // (4B device memory, zeroed once at allocation — the kernel self-resets it
    // in place, so a captured graph replays correctly). Optional: an older .so
    // without the symbol keeps the two-launch gate + route_topk pair.
    gemv_bf16_v2_route: Option<
        unsafe extern "C" fn(
            *const f32, *const c_void, *const f32, *mut f32, c_int, c_int, c_int, *mut f32,
            *mut c_int, *const f32, c_int, c_int, f32, c_int, *mut c_uint, CuStream,
        ) -> c_int,
    >,
    gemv_f32: Option<unsafe extern "C" fn(*const f32, *const f32, *mut f32, c_int, c_int, CuStream) -> c_int>,
    // v2 (vectorized float4 + K-split) f32 M=1 GEMV, from dsv41_glue.cu.
    // Optional: an older .so without the symbol keeps the v1 kernel above.
    // Same ABI as v1: (w, x, out, n=out_f, k=in_f, s).
    gemv_f32_v2: Option<unsafe extern "C" fn(*const f32, *const f32, *mut f32, c_int, c_int, CuStream) -> c_int>,
    argmax: Option<unsafe extern "C" fn(*const f32, *mut c_int, c_int, *mut c_int, CuStream) -> c_int>,
    window_idxs: Option<unsafe extern "C" fn(*mut i32, *const c_int, c_int, CuStream) -> c_int>,
    comp_placeholder:
        Option<unsafe extern "C" fn(*mut i32, *const c_int, c_int, c_int, CuStream) -> c_int>,
    ring_append:
        Option<unsafe extern "C" fn(*mut f32, *const f32, *const c_int, c_int, c_int, CuStream) -> c_int>,
    // B2: apply_rope whose epilogue ALSO emits the fp8 (byte + per-32-block
    // scale) of the whole roped region, so the consumer's `quant_fp8(o)` launch
    // (40/step) disappears. Optional: falls back to (apply_rope, quant_fp8).
    // Returns 1 when the shape cannot take the fused emission.
    apply_rope_q: Option<unsafe extern "C" fn(
        *mut f32, *const f32, *const f32, c_int, c_int, c_int, c_int, *const c_int, c_int, c_int,
        c_int, c_int, *mut u8, *mut f32, CuStream,
    ) -> c_int>,
    // B2: one launch for the `ring_append` + `window_idxs` pair (two adjacent,
    // mutually independent one-block kernels). `ring == null` skips the append
    // half but still writes the indices, so the standalone `window_idxs` launch
    // disappears for every layer. Optional: falls back to the two.
    ring_win_fuse: Option<unsafe extern "C" fn(
        *mut f32, *const f32, *const c_int, c_int, c_int, *mut i32, CuStream,
    ) -> c_int>,
    // B3: the same launch with the `comp_placeholder` recency block folded in
    // (`idxs[window, window + take)`). `clen == null` disables that half and is
    // then byte-identical to `ring_win_fuse`. Optional: an older .so without the
    // symbol keeps the `ring_win_fuse` + `comp_placeholder` pair.
    ring_win_fuse_ph: Option<unsafe extern "C" fn(
        *mut f32, *const f32, *const c_int, c_int, c_int, *mut i32, *const c_int, c_int, CuStream,
    ) -> c_int>,
    index_k_publish:
        Option<unsafe extern "C" fn(*mut f32, *const f32, *const c_int, c_int, CuStream) -> c_int>,
    compress_commit: Option<
        unsafe extern "C" fn(
            *const f32,
            *const f32,
            *const f32,
            *mut f32,
            *const c_int,
            *mut c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            CuStream,
        ) -> c_int,
    >,
    engram_hash_step: Option<
        unsafe extern "C" fn(
            *const i64,
            *mut i64,
            *const i64,
            *const u64,
            *const u64,
            *mut i64,
            *const i32,
            *const i32,
            i64,
            c_int,
            c_int,
            c_int,
            i64,
            CuStream,
        ) -> c_int,
    >,
    expert_gate_up_fp4_indirect: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *mut f32, c_int, c_int, c_int, f32,
            *const u8, i64, *const u8, i64, *const u8, i64, *const u8, i64,
            *const c_int, c_int, CuStream,
        ) -> c_int,
    >,
    expert_down_fp4_indirect: Option<
        unsafe extern "C" fn(
            *const f32, *mut f32, c_int, c_int, c_int, *const f32,
            *const u8, i64, *const u8, i64, *const c_int, c_int, CuStream,
        ) -> c_int,
    >,
    // Batched MoE experts (DSV41_MOE_BATCH, default OFF): one launch per
    // (layer, direction) instead of one per (layer, slot). See the launcher
    // comments in dsv41_experts_mxf4.cu for the numerics contract.
    expert_gate_up_fp4_batched: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *mut f32, i64, c_int, c_int, c_int, f32, c_int,
            *const u8, i64, *const u8, i64, *const u8, i64, *const u8, i64,
            *const c_int, c_int, CuStream,
        ) -> c_int,
    >,
    /// tcgen05 MXFP4 gate/up, Phase-1 skeleton (`DSV41_EXPERT_TCGEN05_MXF4`,
    /// default OFF). OPTIONAL on purpose: `build.sh` defines no
    /// `DSV41_TCGEN05_GATEUP_MXF4_SKELETON`, so a stock `.so` has no such
    /// symbol and the probe below keeps the proven GEMV path. The ABI note in
    /// kernels.rs lists the three shape asymmetries (contiguous gate|up pool,
    /// e8m0 activation scales, no per-slot expert id) that the Phase-2
    /// indirect launcher has to close before `moe()` can dispatch to it.
    expert_tcgen05_gate_up_mxf4: Option<
        unsafe extern "C" fn(
            *const u8, *const u8, *const u8, *const u8, *mut f32, i64, c_int, c_int, f32, c_int,
            CuStream,
        ) -> c_int,
    >,
    /// Load-time gate/up interleave (DSV41_EXPERT_ILV): rewrites an expert's
    /// w1/w3 blocks into one 8-byte-granule-interleaved region so the fused
    /// gate/up GEMV fetches both with one LDG.128. Optional: an .so without it
    /// leaves DSV41_EXPERT_ILV inert and the plain layout (and its fallbacks)
    /// stay in force.
    interleave_gateup_fp4:
        Option<unsafe extern "C" fn(*const u8, *const u8, *mut u8, i64, CuStream) -> c_int>,
    expert_down_fp4_batched: Option<
        unsafe extern "C" fn(
            *const f32, i64, *mut f32, i64, c_int, c_int, c_int, *const f32, i64, c_int,
            *const u8, i64, *const u8, i64, *const c_int, CuStream,
        ) -> c_int,
    >,
    /// Fused down + reduce (DSV41_DOWN_FUSE, default ON): ONE launch computes
    /// every slot's fp4 down GEMV and sums the per-slot contributions in
    /// ascending slot order, writing straight into `out` (overwrite). Replaces
    /// the (expert_down_fp4_batched, moe_down_reduce) pair; both of those stay
    /// for the DSV41_DOWN_FUSE=0 fallback.
    expert_down_reduce_fp4_batched: Option<
        unsafe extern "C" fn(
            *const f32, i64, *mut f32, c_int, c_int, c_int, *const f32, i64, c_int,
            *const u8, i64, *const u8, i64, *const c_int, CuStream,
        ) -> c_int,
    >,
    moe_down_reduce: Option<unsafe extern "C" fn(*const f32, *mut f32, c_int, c_int, CuStream) -> c_int>,
    /// w2 L2 prewarm (DSV41_W2_PREWARM): `cp.async.bulk.prefetch.L2.global` over
    /// every slot's w2 + scale rows, launched between the gate/up and the down
    /// launch. Writes nothing; returns 0 always (best effort).
    w2_l2_prewarm: Option<
        unsafe extern "C" fn(*const u8, i64, *const u8, i64, *const c_int, c_int, i64, i64, CuStream)
            -> c_int,
    >,
    swiglu_limit_batched:
        Option<unsafe extern "C" fn(*mut f32, c_int, c_int, f32, i64, c_int, CuStream) -> c_int>,
    ar_reduce: Option<
        unsafe extern "C" fn(*mut f32, *const f32, i64, i64, c_int, *const c_uint, c_uint, CuStream) -> c_int,
    >,
    compressor_pool: Option<
        unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32, *mut f32, *mut f32, *mut c_int, c_int, c_int, c_int, c_int, c_int, *const c_int, f32, CuStream) -> c_int,
    >,
    /// COMPRESS_FUSE: the decode compressor's three 1-block stages (state carry,
    /// gated pool + RMSNorm, rope + ring commit) as ONE launch. Declines the
    /// non-decode shapes; optional, so an older .so without the symbol keeps the
    /// three-launch path.
    compressor_fused: Option<
        unsafe extern "C" fn(
            *const f32,
            *const f32,
            *const f32,
            *mut f32,
            *mut f32,
            *mut f32,
            *mut c_int,
            *const f32,
            *const f32,
            *mut f32,
            *mut c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            *const c_int,
            f32,
            CuStream,
        ) -> c_int,
    >,
    route_topk: Option<
        unsafe extern "C" fn(*const f32, *const f32, *mut f32, *mut c_int, *mut c_int, c_int, c_int, c_int, c_int, f32, c_int, CuStream) -> c_int,
    >,
    engram_apply: Option<
        unsafe extern "C" fn(*mut f32, *const f32, *const f32, *const f32, *const u8, c_int, c_int, c_int, f32, CuStream) -> c_int,
    >,
    swiglu_limit: Option<unsafe extern "C" fn(*mut f32, c_int, c_int, f32, CuStream) -> c_int>,
    /// A4: swiglu + the fp8 pair the following GEMV consumes, in one launch.
    /// Returns 1 when the inter % 32 warp alignment cannot be met, so the caller
    /// keeps the swiglu_limit + quant1 pair.
    swiglu_limit_q:
        Option<unsafe extern "C" fn(*mut f32, c_int, c_int, f32, *mut u8, *mut f32, CuStream) -> c_int>,
    gather_rows: Option<unsafe extern "C" fn(*const f32, *const i32, *mut f32, c_int, c_int, CuStream) -> c_int>,
    scatter_add_rows: Option<
        unsafe extern "C" fn(*const f32, *const i32, *const f32, *mut f32, c_int, c_int, CuStream) -> c_int,
    >,
    window_append: Option<unsafe extern "C" fn(
        *const f32, *mut u8, *mut f32, c_int, c_int, c_int, c_int, CuStream,
    ) -> c_int>,
    // ---- GLM kernels reused read-only (identical geometry) ----
    rmsnorm: unsafe extern "C" fn(*const f32, *const f32, *mut f32, c_int, c_int, f32, CuStream) -> c_int,
    hc_pre: unsafe extern "C" fn(
        *const f32, *const f32, *const f32, *const f32, *mut f32, *mut f32, *mut f32,
        c_int, c_int, c_int, c_int, f32, f32, c_int, CuStream,
    ) -> c_int,
    hc_post: unsafe extern "C" fn(
        *const f32, *const f32, *const f32, *const f32, *mut f32,
        c_int, c_int, c_int, CuStream,
    ) -> c_int,
    /// Segment C fused: the hyper-connection post-mix evaluated in place on the
    /// residual stream (res is both the residual input and the destination), which
    /// removes the h2 staging buffer and its device-to-device copy.
    hc_post_inplace: unsafe extern "C" fn(
        *mut f32, *const f32, *const f32, *const f32,
        c_int, c_int, CuStream,
    ) -> c_int,
    /// Segment B cluster 1: hc_collapse + rmsnorm(ffn_norm) in one kernel.
    hc_collapse_norm: unsafe extern "C" fn(
        *mut f32, *const f32, *const f32, *mut f32,
        c_int, c_int, c_int, f32, CuStream,
    ) -> c_int,
    /// hc_mixes spread over one block per projection row with cp.async staging,
    /// plus the sum-of-squares/sigmoid/sinkhorn tail in one trailing kernel.
    /// Returns an error when DSV41_HC_FRONT is off, so the caller keeps hc_mixes.
    hc_front: unsafe extern "C" fn(
        *const f32, *const f32, *const f32, *const f32,
        *const f32, *const f32,
        *mut f32, *mut f32, *mut f32, *mut f32,
        c_int, c_int, c_int, c_int, f32, f32, *mut u8, *mut f32, CuStream,
    ) -> c_int,
    /// Stage-C persistent prototype: the whole hc front end in ONE block as a
    /// `__syncthreads` phase machine (no ticket, no spin). Same ABI as
    /// `hc_front`. Optional so a stale `.so` simply falls back to the two-launch
    /// path; the caller gates it with `DSV41_HC_PERSIST=1` (default OFF).
    hc_front_persist: Option<
        unsafe extern "C" fn(
            *const f32, *const f32, *const f32, *const f32,
            *const f32, *const f32,
            *mut f32, *mut f32, *mut f32, *mut f32,
            c_int, c_int, c_int, c_int, f32, f32, *mut u8, *mut f32, CuStream,
        ) -> c_int,
    >,
    /// Stage-C persistent MULTI-BLOCK form: the same front end as
    /// `hc_front_persist`, but the dots spread over `mix * split` blocks (one per
    /// projection row and K chunk), the collapse on its own parallel block, and
    /// the tail elected to whichever dot block finishes last — no ticket, no
    /// spin. Same ABI and fallback contract. Bit-exact at split = 1; a tolerance
    /// gate is required for split > 1 (see the kernel comment). Gated behind
    /// `DSV41_HC_PERSIST_MB=1` (default OFF).
    hc_front_persist_mb: Option<
        unsafe extern "C" fn(
            *const f32, *const f32, *const f32, *const f32,
            *const f32, *const f32,
            *mut f32, *mut f32, *mut f32, *mut f32,
            c_int, c_int, c_int, c_int, f32, f32, *mut u8, *mut f32, CuStream,
        ) -> c_int,
    >,
    /// hc TAIL SPLIT (`DSV41_HC_TAIL_SPLIT`, default ON): same front end as
    /// `hc_front`, but the DOTS and the LATE half (ss/sigmoid/sinkhorn/comb)
    /// leave `main` and run in that order on the side stream, between `fork_ev`
    /// and `join_ev`. The EARLY half (collapse/rmsnorm/fp8) stays on `main`: its
    /// only consumer is the projection group right after the call, so stream
    /// order orders it on both sides — an EARLY-on-side variant needed an extra
    /// in_ev/early_ev pair for no main-stream win and was reverted (2026-09-11).
    /// `main` therefore never waits the dots (4.9 us) or LATE (10.7 us) here; the
    /// caller waits `join_ev` before the hc_post that consumes `comb`.
    /// `in_ev`/`early_ev` remain in the ABI as dead slots (the caller passes
    /// null). Optional: a stale `.so` falls back to the single-launch `hc_front`.
    ///
    /// LAST parameter (`side_dl`) is the `DSV41_HC_DL_SIDE` stream: the
    /// dots+LATE half is issued there, beside EARLY on `side`, instead of behind
    /// it. `null` = collapse onto `side` (the pre-split scheduling, and the
    /// degrade path when the runtime could not create a fourth stream).
    hc_front_split: Option<
        unsafe extern "C" fn(
            *const f32, *const f32, *const f32, *const f32,
            *const f32, *const f32,
            *mut f32, *mut f32, *mut f32, *mut f32,
            c_int, c_int, c_int, c_int, f32, f32, *mut u8, *mut f32,
            CuStream, CuStream, *mut c_void, *mut c_void, *mut c_void, *mut c_void,
            CuStream,
        ) -> c_int,
    >,
    embed_expand_dev: unsafe extern "C" fn(
        *const c_void, *const c_int, *mut f32, c_int, c_int, c_int, c_int, CuStream,
    ) -> c_int,
    f32_to_bf16: unsafe extern "C" fn(*const f32, *mut c_void, c_long, CuStream) -> c_int,
    bf16_to_f32: unsafe extern "C" fn(*const c_void, *mut c_void, c_long, CuStream) -> c_int,
    /// The SHARED all-reduce v5 entry (ferrite_kernels.cu) — one protocol for
    /// both models. DSV41's staging already IS the shared layout (parity
    /// halves `[2][world][stride]`, a `[world]` ready row, a device epoch),
    /// so this is a straight symbol reuse with its own tables passed in.
    p2p_ar_v5: Option<
        unsafe extern "C" fn(
            *const f32,
            *const *mut f32,
            *const *mut u32,
            *mut c_uint,
            *const f32,
            *const c_uint,
            *mut f32,
            c_int,
            c_int,
            c_int,
            c_int,
            CuStream,
        ) -> c_int,
    >,
    /// AR v5 publish+reduce WITHOUT the store: the store half was fused into the
    /// producer kernel's epilogue (`dsv41_gemm_fp8_mx`'s staging args), so this
    /// entry skips `p2p_ar_store_v5_kernel` and only polls/reduces. Same shapes as
    /// `ferrite_p2p_ar_v5` minus `partial` and `staging_tbl` (the caller is not
    /// storing from here).
    p2p_ar_pubred_v5: Option<
        unsafe extern "C" fn(
            *const *mut u32,
            *mut c_uint,
            *const f32,
            *const c_uint,
            *mut f32,
            c_int,
            c_int,
            c_int,
            c_int,
            CuStream,
        ) -> c_int,
    >,
    /// AR v5 with the segment-C `hc_post_inplace` folded into the pubred
    /// epilogue (`ferrite_p2p_ar_v5_hcpost`): same shapes as `p2p_ar_v5` plus the
    /// residual stream `hc_res` (`[hc_n][hc_h]`) and the hyper-connection
    /// `post`/`comb`, written straight back onto `hc_res`. Saves the standalone
    /// `dsv41_hc_post_inplace` launch at each DSV41 AR (2 per layer).
    p2p_ar_v5_hcpost: Option<
        unsafe extern "C" fn(
            *const f32,
            *const *mut f32,
            *const *mut u32,
            *mut c_uint,
            *const f32,
            *const c_uint,
            *mut f32,
            c_int,
            c_int,
            c_int,
            c_int,
            *mut f32,
            *const f32,
            *const f32,
            c_int,
            c_int,
            CuStream,
        ) -> c_int,
    >,
    /// AR v5 with the elementwise residual `add` folded into the store epilogue
    /// (`ferrite_p2p_ar_v5_add`): same shapes as `p2p_ar_v5` plus a `bias`
    /// pointer published as `partial[i] + bias[i]`. Replaces the standalone
    /// `ferrite_add` launch immediately before the MoE all-reduce (ADD_EPI).
    p2p_ar_v5_add: Option<
        unsafe extern "C" fn(
            *const f32,
            *const f32,
            *const *mut f32,
            *const *mut u32,
            *mut c_uint,
            *const f32,
            *const c_uint,
            *mut f32,
            c_int,
            c_int,
            c_int,
            c_int,
            CuStream,
        ) -> c_int,
    >,
    /// `ferrite_p2p_ar_v5_hcpost` + the ADD_EPI residual (same shapes as
    /// `p2p_ar_v5_hcpost`, `bias` inserted right after `partial`).
    p2p_ar_v5_hcpost_add: Option<
        unsafe extern "C" fn(
            *const f32,
            *const f32,
            *const *mut f32,
            *const *mut u32,
            *mut c_uint,
            *const f32,
            *const c_uint,
            *mut f32,
            c_int,
            c_int,
            c_int,
            c_int,
            *mut f32,
            *const f32,
            *const f32,
            c_int,
            c_int,
            CuStream,
        ) -> c_int,
    >,
}

// ------------------------------------------------------------------- Device

/// The DeepSeek device handle: the shared runtime plus this model's kernel
/// table. Every primitive below forwards to [`DevRuntime`]; only the kernel
/// launches are model-specific.
pub struct Device {
    rt: DevRuntime,
    kernels: Kernels,
    /// Cached stream handle — the kernel wrappers submit here
    /// (identical to `rt.stream()`).
    stream: CuStream,
    /// Set when `hc_front_split` issued a tail LATE half whose `join_event` the
    /// main stream has not yet waited. `hc_tail_join` consumes it. Interior
    /// mutability because every kernel wrapper takes `&self`.
    hc_split_armed: std::cell::Cell<bool>,
}

impl Device {
    /// `kernel_so` is the path to `libferrite_kernels.so` (the dsv41 kernels
    /// are linked into the same object as GLM's).
    pub fn open(kernel_so: &str) -> Result<Device> {
        let rt = DevRuntime::open(kernel_so)?;
        let stream = rt.stream();
        let kernels = Kernels {
            gemm_fp8_mx: km!(rt, "dsv41_gemm_fp8_mx"),
            gemm_fp8_mx2: km!(rt, "dsv41_gemm_fp8_mx2"),
            gemm_fp8_mx_rope: ko!(rt, "dsv41_gemm_fp8_mx_rope"),
            gemm_fp8_mx2_rope: ko!(rt, "dsv41_gemm_fp8_mx2_rope"),
            gemm_fp8_mx_rope_norm: ko!(rt, "dsv41_gemm_fp8_mx_rope_norm"),
            gemm_fp8_mx_add: ko!(rt, "dsv41_gemm_fp8_mx_add"),
            gemm_fp8_mx_f32: ko!(rt, "dsv41_gemm_fp8_mx_f32"),
            gemm_fp8_swapab: ko!(rt, "dsv41_gemm_fp8_swapab"),
            gemm_fp8_wo_pair: ko!(rt, "dsv41_gemm_fp8_wo_pair"),
            gemm_fp8_sh_pair: ko!(rt, "dsv41_gemm_fp8_sh_pair"),
            quant_fp8: km!(rt, "dsv41_quant_fp8"),
            quant_fp4: km!(rt, "dsv41_quant_fp4"),
            expert_gate_up_fp4: km!(rt, "dsv41_expert_gate_up_fp4"),
            expert_down_fp4: km!(rt, "dsv41_expert_down_fp4"),
            engram_hash: km!(rt, "dsv41_engram_hash"),
            engram_gather: km!(rt, "dsv41_engram_gather"),
            sparse_attn: km!(rt, "dsv41_sparse_attn"),
            sparse_attn_orope: ko!(rt, "dsv41_sparse_attn_orope"),
            indexer_topk: km!(rt, "dsv41_indexer_topk"),
            candidate_blocks: km!(rt, "dsv41_candidate_blocks"),
            compressor: km!(rt, "dsv41_compressor"),
            rope_precompute: km!(rt, "dsv41_rope_precompute"),
            apply_rope: km!(rt, "dsv41_apply_rope"),
            rmsnorm_rope: ko!(rt, "dsv41_rmsnorm_rope"),
            rmsnorm_q: ko!(rt, "dsv41_rmsnorm_q"),
            gemm_bf16_fp8x2: ko!(rt, "dsv41_gemm_bf16_fp8x2"),
            argmax_sliced: ko!(rt, "dsv41_argmax_sliced"),
            hc_mixes: km!(rt, "dsv41_hc_mixes"),
            moe_route: km!(rt, "dsv41_moe_route"),
            add_inplace: ko!(rt, "ferrite_add"),
            hc_collapse: ko!(rt, "dsv41_hc_collapse"),
            ar_stamp: ko!(rt, "dsv41_ar_stamp"),
            ar_store: ko!(rt, "dsv41_ar_store"),
            ar_store2: ko!(rt, "dsv41_ar_store2"),
            ar_reduce2: ko!(rt, "dsv41_ar_reduce2"),
            ar_mark: ko!(rt, "dsv41_ar_mark"),
            gemv_bf16: ko!(rt, "dsv41_gemv_bf16"),
            gemv_bf16_v2: ko!(rt, "ferrite_gemv_bf16_v2"),
            gemv_bf16_v2_route: ko!(rt, "ferrite_gemv_bf16_v2_route"),
            gemv_f32: ko!(rt, "dsv41_gemv_f32"),
            gemv_f32_v2: ko!(rt, "dsv41_gemv_f32_v2"),
            argmax: ko!(rt, "dsv41_argmax"),
            engram_hash_step: ko!(rt, "dsv41_engram_hash_step"),
            window_idxs: ko!(rt, "dsv41_window_idxs"),
            comp_placeholder: ko!(rt, "dsv41_comp_placeholder"),
            compress_commit: ko!(rt, "dsv41_compress_commit"),
            ring_append: ko!(rt, "dsv41_ring_append"),
            apply_rope_q: ko!(rt, "dsv41_apply_rope_q"),
            ring_win_fuse: ko!(rt, "dsv41_ring_win_fuse"),
            ring_win_fuse_ph: ko!(rt, "dsv41_ring_win_fuse_ph"),
            index_k_publish: ko!(rt, "dsv41_index_k_publish"),
            expert_gate_up_fp4_indirect: ko!(rt, "dsv41_expert_gate_up_fp4_indirect"),
            expert_down_fp4_indirect: ko!(rt, "dsv41_expert_down_fp4_indirect"),
            expert_gate_up_fp4_batched: ko!(rt, "dsv41_expert_gate_up_fp4_batched"),
            expert_tcgen05_gate_up_mxf4: ko!(rt, "dsv41_expert_tcgen05_gate_up_mxf4"),
            interleave_gateup_fp4: ko!(rt, "dsv41_interleave_gateup_fp4"),
            expert_down_fp4_batched: ko!(rt, "dsv41_expert_down_fp4_batched"),
            moe_down_reduce: ko!(rt, "dsv41_moe_down_reduce"),
            expert_down_reduce_fp4_batched: ko!(rt, "dsv41_expert_down_reduce_fp4_batched"),
            w2_l2_prewarm: ko!(rt, "dsv41_w2_l2_prewarm"),
            swiglu_limit_batched: ko!(rt, "dsv41_swiglu_limit_batched"),
            ar_reduce: ko!(rt, "dsv41_ar_reduce"),
            route_topk: ko!(rt, "dsv41_route_topk"),
            compressor_pool: ko!(rt, "dsv41_compressor_pool"),
            compressor_fused: ko!(rt, "dsv41_compressor_fused"),
            engram_apply: ko!(rt, "dsv41_engram_apply"),
            swiglu_limit: ko!(rt, "dsv41_swiglu_limit"),
            swiglu_limit_q: ko!(rt, "dsv41_swiglu_limit_q"),
            gather_rows: ko!(rt, "dsv41_gather_rows"),
            scatter_add_rows: ko!(rt, "dsv41_scatter_add_rows"),
            window_append: ko!(rt, "dsv41_window_append"),
            rmsnorm: km!(rt, "ferrite_rmsnorm"),
            hc_pre: km!(rt, "ferrite_hc_pre"),
            hc_post: km!(rt, "ferrite_hc_post"),
            hc_post_inplace: km!(rt, "dsv41_hc_post_inplace"),
            hc_collapse_norm: km!(rt, "dsv41_hc_collapse_norm"),
            hc_front: km!(rt, "dsv41_hc_front"),
            hc_front_persist: ko!(rt, "dsv41_hc_front_persist"),
            hc_front_persist_mb: ko!(rt, "dsv41_hc_front_persist_mb"),
            hc_front_split: ko!(rt, "dsv41_hc_front_split"),
            embed_expand_dev: km!(rt, "ferrite_embed_expand_dev"),
            f32_to_bf16: km!(rt, "ferrite_f32_to_bf16"),
            bf16_to_f32: km!(rt, "ferrite_bf16_to_f32"),
            p2p_ar_v5: ko!(rt, "ferrite_p2p_ar_v5"),
            p2p_ar_pubred_v5: ko!(rt, "ferrite_p2p_ar_pubred_v5"),
            p2p_ar_v5_hcpost: ko!(rt, "ferrite_p2p_ar_v5_hcpost"),
            p2p_ar_v5_add: ko!(rt, "ferrite_p2p_ar_v5_add"),
            p2p_ar_v5_hcpost_add: ko!(rt, "ferrite_p2p_ar_v5_hcpost_add"),
        };
        Ok(Device { rt, kernels, stream, hc_split_armed: std::cell::Cell::new(false) })
    }

    // ------------------------------------------- shared device primitives
    // Every method below forwards to the shared `ferrite_kernel::devrt`
    // runtime; the model layer owns no cudart/cublas binding of its own.

    pub fn stream(&self) -> CuStream {
        self.stream
    }

    /// True when the runtime owns the SECOND side stream and its fork/join
    /// events. A cudart without the event primitives, or a failed stream create,
    /// reports false and the caller keeps the serial kv chain.
    pub fn supports_dual_chain(&self) -> bool {
        !self.rt.side_stream2().is_null()
            && !self.rt.fork2_event().is_null()
            && !self.rt.join2_event().is_null()
    }

    /// The second side stream (the dual chain's kv half). Only meaningful when
    /// [`Self::supports_dual_chain`] is true.
    pub fn side_stream2(&self) -> CuStream {
        self.rt.side_stream2()
    }

    /// Attention dual chain — fork. Record the fork event on the MAIN stream
    /// (the side stream may not start before the fork point's work is done) and
    /// make `side_stream2` wait it. Legal inside a capture: both ops become
    /// graph edges.
    pub fn dual_chain_fork(&self) -> Result<()> {
        self.rt.record_event(self.rt.fork2_event(), self.stream)?;
        self.rt
            .stream_wait_event(self.rt.side_stream2(), self.rt.fork2_event())
    }

    /// Attention dual chain — join. Record the join event on `side_stream2` and
    /// make the MAIN stream wait it. Must run after the kv half is fully issued
    /// and before its first consumer (ring append / sparse_attn). Legal inside a
    /// capture.
    pub fn dual_chain_join(&self) -> Result<()> {
        self.rt
            .record_event(self.rt.join2_event(), self.rt.side_stream2())?;
        self.rt
            .stream_wait_event(self.stream, self.rt.join2_event())
    }

    /// True when the runtime owns the THIRD side stream and its fork/join
    /// events. A cudart without the event primitives, or a failed stream create,
    /// reports false and the caller keeps the serial compressor.
    pub fn supports_compress_side(&self) -> bool {
        !self.rt.side_stream3().is_null()
            && !self.rt.fork3_event().is_null()
            && !self.rt.join3_event().is_null()
    }

    /// The third side stream (the compressor's four launches). Only meaningful
    /// when [`Self::supports_compress_side`] is true.
    pub fn side_stream3(&self) -> CuStream {
        self.rt.side_stream3()
    }

    /// True when the loaded .so carries COMPRESS_FUSE (`dsv41_compressor_fused`).
    /// Optional: a stale .so reports false and the compressor keeps the three
    /// separate launches (state + pool + commit), which are bit-identical.
    pub fn supports_compress_fuse(&self) -> bool {
        self.kernels.compressor_fused.is_some()
    }

    /// Compressor side stream — fork. Record the fork event on the MAIN stream
    /// and make `side_stream3` wait it. Legal inside a capture: both ops become
    /// graph edges.
    pub fn compress_side_fork(&self) -> Result<()> {
        self.rt.record_event(self.rt.fork3_event(), self.stream)?;
        self.rt
            .stream_wait_event(self.rt.side_stream3(), self.rt.fork3_event())
    }

    /// Compressor side stream — join. Record the join event on `side_stream3`
    /// and make the MAIN stream wait it. Must run after the compressor is fully
    /// issued and before its first consumer: the `indexer`'s `latent` read (a
    /// kv-source index layer reads `self.layers[layer].latent`, written by
    /// `compressor_pool`) and `sparse_attn`'s compressed ring rows. Legal inside
    /// a capture.
    pub fn compress_side_join(&self) -> Result<()> {
        self.rt
            .record_event(self.rt.join3_event(), self.rt.side_stream3())?;
        self.rt
            .stream_wait_event(self.stream, self.rt.join3_event())
    }

    pub fn device_id(&self) -> i32 {
        self.rt.device_id()
    }

    pub fn device_count(&self) -> i32 {
        self.rt.device_count()
    }

    pub fn bind_to(gpu: i32) -> Result<()> {
        DevRuntime::bind_to(gpu)
    }

    pub fn enable_peer_access(&self) -> Result<usize> {
        self.rt.enable_peer_access()
    }

    pub fn memcpy_peer(
        &self,
        dst_dev: i32,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
    ) -> Result<()> {
        self.rt.memcpy_peer(dst_dev, dst, src, bytes)
    }

    pub fn sync(&self) -> Result<()> {
        self.rt.sync()
    }

    pub fn dev_sync(&self) -> Result<()> {
        self.rt.dev_sync()
    }

    pub fn alloc(&self, bytes: usize) -> Result<DevBuf> {
        self.rt.alloc(bytes)
    }

    pub fn mem_free(&self) -> usize {
        self.rt.mem_free()
    }

    pub fn view(ptr: *mut c_void, bytes: usize) -> DevBuf {
        DevBuf::view(ptr, bytes)
    }

    pub fn upload(&self, bytes: &[u8]) -> Result<DevBuf> {
        self.rt.upload(bytes)
    }

    pub fn upload_f32(&self, v: &[f32]) -> Result<DevBuf> {
        self.rt.upload_f32(v)
    }

    pub fn upload_f32_at(&self, dst: *mut c_void, off: usize, v: &[f32]) -> Result<()> {
        self.rt.upload_f32_at(dst, off, v)
    }

    pub fn download_f32(&self, src: &DevBuf, out: &mut [f32]) -> Result<()> {
        self.rt.download_f32(src, out)
    }

    pub fn upload_bytes_at(&self, dst: &DevBuf, bytes: &[u8]) -> Result<()> {
        self.rt.upload_bytes_at(dst, bytes)
    }

    pub fn download_u8(&self, src: &DevBuf, out: &mut [u8]) -> Result<()> {
        self.rt.download_u8(src, out)
    }

    pub fn upload_from(&self, dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<()> {
        self.rt.upload_from(dst, src, bytes)
    }

    pub fn upload_from_2d(
        &self,
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
    ) -> Result<()> {
        self.rt.upload_from_2d(dst, dpitch, src, spitch, width, height)
    }

    pub fn zero_at(&self, ptr: *mut c_void, bytes: usize) -> Result<()> {
        self.rt.zero_at(ptr, bytes)
    }

    pub fn zero(&self, b: &DevBuf) -> Result<()> {
        self.rt.zero(b)
    }

    /// [`Self::zero`] issued on `s` instead of the main stream. Only the
    /// compressor's `ratio == 1` no-gate path uses it (`DSV41_COMPRESS_SIDE`).
    pub fn zero_on(&self, b: &DevBuf, s: CuStream) -> Result<()> {
        self.rt.zero_at_on(b.ptr, b.bytes, s)
    }

    pub fn memcpy_d2d(&self, dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<()> {
        self.rt.memcpy_d2d(dst, src, bytes)
    }

    pub fn download_u32(&self, ptr: *const c_void) -> Result<u32> {
        self.rt.download_u32(ptr)
    }

    pub fn free(&self, b: &DevBuf) {
        self.rt.free(b)
    }

    pub fn gemm_f32(
        &self,
        x: *const c_void,
        w: *const c_void,
        out: *mut f32,
        rows: i32,
        n_out: i32,
        k: i32,
    ) -> Result<()> {
        self.rt.gemm_f32(x, w, out, rows, n_out, k)
    }

    pub fn gemm_bf16(
        &self,
        x: *const c_void,
        w: *const c_void,
        out: *mut f32,
        rows: i32,
        n_out: i32,
        k: i32,
    ) -> Result<()> {
        self.rt.gemm_bf16(x, w, out, rows, n_out, k)
    }

    pub fn capture_begin(&self) -> Result<()> {
        self.rt.capture_begin()
    }

    pub fn capture_end(&self) -> Result<*mut c_void> {
        self.rt.capture_end()
    }

    pub fn graph_instantiate(&self, g: *mut c_void) -> Result<*mut c_void> {
        self.rt.graph_instantiate(g)
    }

    pub fn graph_launch(&self, e: *mut c_void) -> Result<()> {
        self.rt.graph_launch(e)
    }

    pub fn graph_free(&self, g: *mut c_void, e: *mut c_void) -> Result<()> {
        self.rt.graph_free(g, e)
    }

    /// Launch-status check, forwarded to the shared runtime (honours
    /// FERRITE_DEBUG_SYNC / DSV41_DEBUG_SYNC).
    fn kerr(&self, rc: c_int, what: &str) -> Result<()> {
        self.rt.kerr(rc, what)
    }


    pub fn add_inplace(&self, dst: &DevBuf, src: &DevBuf, n: i64) -> Result<()> {
        let f = self.need(self.kernels.add_inplace, "ferrite_add")?;
        let rc = unsafe {
            f(dst.ptr as *const f32, src.ptr as *const f32, dst.ptr as *mut f32, n as c_int, self.stream)
        };
        self.kerr(rc, "ferrite_add")
    }

    /// Device-wide synchronisation (used by the all-reduce, which must know
    /// that its peer copies have landed before summing them).
    pub fn add_inplace_raw(&self, dst: *mut c_void, src: *const c_void, n: i64) -> Result<()> {
        let f = self.need(self.kernels.add_inplace, "ferrite_add")?;
        let rc = unsafe {
            f(dst as *const f32, src as *const f32, dst as *mut f32, n as c_int, self.stream)
        };
        self.kerr(rc, "ferrite_add")
    }

    fn need<T: Copy>(&self, f: Option<T>, name: &str) -> Result<T> {
        f.ok_or_else(|| {
            FerriteError::Config(format!(
                "kernel {name} is not in the loaded .so — rebuild kernels/cuda (bash build.sh 103a)"
            ))
        })
    }

    /// True when the loaded .so carries the whole DSV41_MOE_BATCH kernel set.
    /// The batched dispatch is optional, so a stale .so falls back to the
    /// sequential expert loop instead of failing the step.
    pub fn supports_moe_batch(&self) -> bool {
        self.kernels.expert_gate_up_fp4_batched.is_some()
            && self.kernels.expert_down_fp4_batched.is_some()
            && self.kernels.moe_down_reduce.is_some()
            && self.kernels.swiglu_limit_batched.is_some()
    }

    /// True when the loaded .so carries the load-time gate/up interleave entry
    /// point (`dsv41_interleave_gateup_fp4`, ABI 2). Without it DSV41_EXPERT_ILV
    /// stays inert and the pools keep the plain w1/w3 layout.
    pub fn supports_expert_ilv(&self) -> bool {
        self.kernels.interleave_gateup_fp4.is_some()
    }

    /// True when the loaded .so carries the fused down+reduce entry point
    /// (`dsv41_expert_down_reduce_fp4_batched`). A stale .so leaves
    /// DSV41_DOWN_FUSE inert and the (batched down, moe_down_reduce) pair runs.
    pub fn supports_down_fuse(&self) -> bool {
        self.kernels.expert_down_reduce_fp4_batched.is_some()
    }

    /// True when the loaded .so carries the w2 L2 prewarm entry point
    /// (`dsv41_w2_l2_prewarm`). A stale .so keeps DSV41_W2_PREWARM inert and the
    /// down GEMV streams w2 from HBM exactly as before.
    pub fn supports_w2_prewarm(&self) -> bool {
        self.kernels.w2_l2_prewarm.is_some()
    }

    /// True when the loaded .so carries the ADD_EPI (residual-in-store) all-reduce
    /// entry (`ferrite_p2p_ar_v5_add`). A stale .so leaves DSV41_ADD_EPI inert and
    /// the standalone `add_inplace` + `all_reduce_inplace` pair runs.
    pub fn supports_ar_add(&self) -> bool {
        self.kernels.p2p_ar_v5_add.is_some()
    }

    /// Same probe for the hc-post fold's ADD_EPI variant
    /// (`ferrite_p2p_ar_v5_hcpost_add`) — the one the default MoE AR uses.
    pub fn supports_ar_hcpost_add(&self) -> bool {
        self.kernels.p2p_ar_v5_hcpost_add.is_some()
    }

    pub fn gemm_fp8_mx(
        &self,
        a: *const u8,
        a_scale: *const f32,
        w: *const u8,
        w_scale: *const u8,
        bias: *const f32,
        out: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        self.gemm_fp8_mx_on(a, a_scale, w, w_scale, bias, out, m, n, k, self.stream)
    }

    /// [`Self::gemm_fp8_mx`] issued on `s` instead of the main stream. The MoE
    /// dual chain's shared half (`DSV41_MOE_DUAL`) sends its w1/w3 and w2 GEMVs
    /// here so they run beside the routed experts' fp4 chain.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_mx_on(
        &self,
        a: *const u8,
        a_scale: *const f32,
        w: *const u8,
        w_scale: *const u8,
        bias: *const f32,
        out: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        s: CuStream,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.gemm_fp8_mx)(
                a, a_scale, w, w_scale, bias, out, m, n, k, s,
                std::ptr::null(), std::ptr::null(), 0, 0, 0,
                std::ptr::null_mut(), std::ptr::null_mut(),
            )
        };
        self.kerr(rc, "dsv41_gemm_fp8_mx")
    }

    /// B1: the M=1 GEMV whose epilogue ALSO emits the fp8 row compression of
    /// `out` — the e4m3 bytes plus the per-32-block f32 scales, with
    /// `quant_kernel<0>`'s arithmetic term for term, so the consumer's
    /// `quant1(out)` launch disappears. The byte pair is bit-identical to that
    /// launch, but `xq` must NOT alias `a`/`a_scale`: the quantised input is
    /// still being staged by blocks that start late.
    ///
    /// Requires `n % 32 == 0`: the kernel gives the block 32 warps so that its
    /// 32 consecutive rows ARE one quant block (a gemv warp produces one row, so
    /// the amax of a block spans 32 warps — unlike the hc-tail/swiglu producers
    /// where a warp's 32 lanes are the 32 elements).
    ///
    /// Ok(false) means the shape/`mode` cannot take the fused path (the kernel
    /// returns `cudaErrorInvalidValue`); the caller then runs the plain
    /// `gemm_fp8_mx` followed by its own `quant1`, which is bit-identical.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_mx_q(
        &self,
        a: *const u8,
        a_scale: *const f32,
        w: *const u8,
        w_scale: *const u8,
        bias: *const f32,
        out: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        xq: *mut u8,
        xsc: *mut f32,
    ) -> Result<bool> {
        let rc = unsafe {
            (self.kernels.gemm_fp8_mx)(
                a, a_scale, w, w_scale, bias, out, m, n, k, self.stream,
                std::ptr::null(), std::ptr::null(), 0, 0, 0, xq, xsc,
            )
        };
        // ⚠️ LEGACY decline contract (rounds 37-41), deliberately left as-is:
        // `rc == 1` here means BOTH "the fused path declined" AND the real
        // `cudaErrorInvalidValue`. `dsv41_gemm_fp8_mx` also returns 1 from its
        // SetAttribute/launch paths WITHOUT clearing the sticky flag, so a real
        // failure is read as a graceful fallback and the sticky error survives -
        // exactly the crash class r42 fixed for the gemv launchers. Only
        // reachable with DSV41_WO_QUANT_FUSE=1 (default OFF). Migrate this symbol
        // to the r42/r43 convention (decline == 2) before flipping it ON.
        if rc == 1 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_mx")?;
        Ok(true)
    }

    /// `wo_b` GEMV with the AR v5 store FUSED into the epilogue: the row partial
    /// goes both to `out` and (for every peer) to the staging slot of this round,
    /// so the following AR is publish+reduce only (`p2p_ar_pubred_v5`). M=1 only;
    /// `stride` is the per-slot FLOAT element count (`bytes/4`), the same unit the
    /// store kernel uses. The AR must follow on the same stream with no other
    /// all-reduce in between (both this store and the pubred read `*epoch`).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_mx_ar(
        &self,
        a: *const u8,
        a_scale: *const f32,
        w: *const u8,
        w_scale: *const u8,
        out: *mut f32,
        n: i32,
        k: i32,
        staging_tbl: *const *mut f32,
        epoch: *const c_uint,
        world: i32,
        my_rank: i32,
        stride: i32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.gemm_fp8_mx)(
                a, a_scale, w, w_scale, std::ptr::null(), out, 1, n, k, self.stream,
                staging_tbl, epoch, world, my_rank, stride,
                std::ptr::null_mut(), std::ptr::null_mut(),
            )
        };
        self.kerr(rc, "dsv41_gemm_fp8_mx")
    }

    /// Two same-activation fp8 projections in one gemv launch (rows below n1 map
    /// to the first family). Returns Ok(false) when the fused kernel declines
    /// the shape, so the caller runs the two separate calls; any other error
    /// propagates.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_mx2(
        &self,
        a: *const u8,
        a_scale: *const f32,
        w1: *const u8,
        w1_scale: *const u8,
        bias1: *const f32,
        out1: *mut f32,
        n1: i32,
        w2: *const u8,
        w2_scale: *const u8,
        bias2: *const f32,
        out2: *mut f32,
        n2: i32,
        k: i32,
    ) -> Result<bool> {
        self.gemm_fp8_mx2_on(
            a, a_scale, w1, w1_scale, bias1, out1, n1, w2, w2_scale, bias2, out2, n2, k,
            self.stream,
        )
    }

    /// [`Self::gemm_fp8_mx2`] issued on `s` instead of the main stream (the MoE
    /// dual chain's shared-expert w1/w3 pair).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_mx2_on(
        &self,
        a: *const u8,
        a_scale: *const f32,
        w1: *const u8,
        w1_scale: *const u8,
        bias1: *const f32,
        out1: *mut f32,
        n1: i32,
        w2: *const u8,
        w2_scale: *const u8,
        bias2: *const f32,
        out2: *mut f32,
        n2: i32,
        k: i32,
        s: CuStream,
    ) -> Result<bool> {
        let rc = unsafe {
            (self.kernels.gemm_fp8_mx2)(
                a,
                a_scale,
                w1,
                w1_scale,
                bias1,
                out1,
                n1,
                w2,
                w2_scale,
                bias2,
                out2,
                n2,
                k,
                s,
            )
        };
        if rc == 1 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_mx2")?;
        Ok(true)
    }

    /// True when the loaded .so carries BOTH rope-fused GEMV entry points. A
    /// stale .so leaves DSV41_ROPE_FUSE inert and the standalone apply_rope runs.
    pub fn supports_rope_fuse(&self) -> bool {
        self.kernels.gemm_fp8_mx_rope.is_some() && self.kernels.gemm_fp8_mx2_rope.is_some()
    }

    /// RoPE fusion (q rope): the M=1 GEMV whose epilogue also rotates the trailing
    /// `rope_rd` lanes of each of `rope_hd`-wide head, with `apply_rope_kernel`'s
    /// rotation expression verbatim, so the result is bit-identical to the
    /// standalone launch. Ok(false) => the caller runs `gemm_fp8_mx` + apply_rope.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_mx_rope(
        &self,
        a: *const u8,
        a_scale: *const f32,
        w: *const u8,
        w_scale: *const u8,
        bias: *const f32,
        out: *mut f32,
        n: i32,
        k: i32,
        rope_cos: *const f32,
        rope_sin: *const f32,
        rope_base: *const c_int,
        rope_mul: i32,
        rope_off: i32,
        rope_step: i32,
        rope_inverse: bool,
        rope_rd: i32,
        rope_hd: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.gemm_fp8_mx_rope else {
            return Ok(false);
        };
        let rc = unsafe {
            f(
                a,
                a_scale,
                w,
                w_scale,
                bias,
                out,
                n,
                k,
                rope_cos,
                rope_sin,
                rope_base,
                rope_mul,
                rope_off,
                rope_step,
                rope_inverse as i32,
                rope_rd,
                rope_hd,
                self.stream,
            )
        };
        // r43 decline-code fix: the launcher's shape decline is 2 (never 1, which
        // is cudaErrorInvalidValue and would make a real launch failure look like
        // a graceful fallback).
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_mx_rope")?;
        Ok(true)
    }

    /// RoPE fusion for the two-family GEMV (wq_b / idx_wq_b): family 1's head
    /// width is `rope_hd1`, family 2's is `rope_hd2`; both rotate the same
    /// `rope_rd` trailing lanes with the same cos/sin and position counter.
    /// Ok(false) => the caller runs `gemm_fp8_mx2` + both apply_rope launches.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_mx2_rope(
        &self,
        a: *const u8,
        a_scale: *const f32,
        w1: *const u8,
        w1_scale: *const u8,
        bias1: *const f32,
        out1: *mut f32,
        n1: i32,
        w2: *const u8,
        w2_scale: *const u8,
        bias2: *const f32,
        out2: *mut f32,
        n2: i32,
        k: i32,
        rope_cos: *const f32,
        rope_sin: *const f32,
        rope_base: *const c_int,
        rope_mul: i32,
        rope_off: i32,
        rope_step: i32,
        rope_inverse: bool,
        rope_rd: i32,
        rope_hd1: i32,
        rope_hd2: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.gemm_fp8_mx2_rope else {
            return Ok(false);
        };
        let rc = unsafe {
            f(
                a,
                a_scale,
                w1,
                w1_scale,
                bias1,
                out1,
                n1,
                w2,
                w2_scale,
                bias2,
                out2,
                n2,
                k,
                rope_cos,
                rope_sin,
                rope_base,
                rope_mul,
                rope_off,
                rope_step,
                rope_inverse as i32,
                rope_rd,
                rope_hd1,
                rope_hd2,
                self.stream,
            )
        };
        // r43 decline-code fix: shape decline is 2 (see gemm_fp8_mx_rope).
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_mx2_rope")?;
        Ok(true)
    }

    /// NORM_FUSE: true when the loaded .so carries the norm-fused rope GEMV
    /// (`dsv41_gemm_fp8_mx_rope_norm`). A stale .so leaves DSV41_NORM_FUSE inert
    /// and the (rmsnorm_q, gemm_fp8_mx_rope) pair runs.
    pub fn supports_gemm_fp8_norm(&self) -> bool {
        self.kernels.gemm_fp8_mx_rope_norm.is_some()
    }

    /// Producer/consumer fusion of `rmsnorm_q` into the M=1 rope GEMV: the kernel
    /// PROLOGUE computes the RMSNorm of the raw f32 row `qr_raw` (k elements,
    /// weight `qr_w`, eps `qr_eps`) and the fp8 e4m3 encoding of the result --
    /// bytes plus per-32-block scales -- with `rmsnorm_q_kernel`'s arithmetic,
    /// term for term, straight into shared memory. The standalone `rmsnorm_q`
    /// launch and the `xq`/`xsc` hand-off disappear; the gemv's own dot and rope
    /// epilogue are untouched, so `out` is bit-identical to
    /// (`rmsnorm_q`, `gemm_fp8_mx_rope`).
    ///
    /// `qr_raw` is the row the caller would have handed to `rmsnorm_q`, and it
    /// must stay valid for the whole launch - on this path it is NOT normalised in
    /// place. The shape gates are the rope launcher's exactly (plus a forced
    /// 32-warp block, the reference's 1024-thread reduction tree); Ok(false) on
    /// any decline, so the caller keeps the old pair.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_mx_rope_norm(
        &self,
        qr_raw: *const f32,
        qr_w: *const f32,
        qr_eps: f32,
        w: *const u8,
        w_scale: *const u8,
        bias: *const f32,
        out: *mut f32,
        n: i32,
        k: i32,
        rope_cos: *const f32,
        rope_sin: *const f32,
        rope_base: *const c_int,
        rope_mul: i32,
        rope_off: i32,
        rope_step: i32,
        rope_inverse: bool,
        rope_rd: i32,
        rope_hd: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.gemm_fp8_mx_rope_norm else {
            return Ok(false);
        };
        let rc = unsafe {
            f(
                qr_raw,
                qr_w,
                qr_eps,
                w,
                w_scale,
                bias,
                out,
                n,
                k,
                rope_cos,
                rope_sin,
                rope_base,
                rope_mul,
                rope_off,
                rope_step,
                rope_inverse as i32,
                rope_rd,
                rope_hd,
                self.stream,
            )
        };
        // Round-42 fix: the C side's shape-decline is now 2 (never 1, which
        // collides with cudaErrorInvalidValue and made a REAL launch failure
        // look like a graceful decline while the sticky error propagated).
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_mx_rope_norm")?;
        Ok(true)
    }

    /// A5: true when the loaded .so carries the fused-epilogue w2 GEMV
    /// (`dsv41_gemm_fp8_mx_add`). A stale .so leaves DSV41_MOE_EPI_ADD inert and
    /// the (gemm_fp8_mx, add_inplace) pair runs.
    pub fn supports_gemm_fp8_add(&self) -> bool {
        self.kernels.gemm_fp8_mx_add.is_some()
    }

    /// A5: the M=1 w2 GEMV with the trailing add_inplace folded in
    /// (`out += w @ a`). Ok(false) => the caller runs gemm_fp8_mx into a scratch
    /// row and then add_inplace, exactly as before.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_mx_add(
        &self,
        a: *const u8,
        a_scale: *const f32,
        w: *const u8,
        w_scale: *const u8,
        bias: *const f32,
        out: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<bool> {
        self.gemm_fp8_mx_add_on(a, a_scale, w, w_scale, bias, out, m, n, k, self.stream)
    }

    /// [`Self::gemm_fp8_mx_add`] issued on `s` instead of the main stream. Not
    /// used by the MoE dual chain (its w2 must land in a DISJOINT scratch while
    /// the routed down-reduce owns `s.o`), but kept beside its siblings so the
    /// two paths cannot drift.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_mx_add_on(
        &self,
        a: *const u8,
        a_scale: *const f32,
        w: *const u8,
        w_scale: *const u8,
        bias: *const f32,
        out: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        s: CuStream,
    ) -> Result<bool> {
        let f = self.need(self.kernels.gemm_fp8_mx_add, "dsv41_gemm_fp8_mx_add")?;
        let rc = unsafe { f(a, a_scale, w, w_scale, bias, out, m, n, k, s) };
        // r43 decline-code fix: shape decline is 2 (see gemm_fp8_mx_rope).
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_mx_add")?;
        Ok(true)
    }

    /// wo_b: true when the loaded .so carries the f32-activation GEMV
    /// (`dsv41_gemm_fp8_mx_f32`). A stale .so leaves DSV41_WOB_F32 inert and the
    /// (quant1, gemm_fp8_mx) pair runs.
    pub fn supports_gemm_fp8_f32(&self) -> bool {
        self.kernels.gemm_fp8_mx_f32.is_some()
    }

    /// chain-pair-grid-sync: true when the loaded .so carries the fused
    /// wo_a -> wo_b grid-sync pair kernel (`dsv41_gemm_fp8_wo_pair`). A stale
    /// .so leaves DSV41_WO_PAIR inert and the (wo_a, wo_b) two-launch path runs.
    pub fn supports_wo_pair(&self) -> bool {
        self.kernels.gemm_fp8_wo_pair.is_some()
    }

    /// chain-pair-grid-sync: the wo_a -> wo_b pair as ONE launch, joined by a
    /// device-wide sense-reversing barrier instead of a stream edge. `bar` is the
    /// caller's persistent `[arrive, sense]` u32 pair (zeroed once); the kernel
    /// self-resets it every launch, so a captured graph replays correctly.
    /// Ok(false) => the shape/arm cannot use it and the caller keeps the two
    /// calls. Nothing about the numerics changes: each row is still one warp in
    /// the same lane order, so both outputs are bit-identical.
    /// ABI: stream LAST (this symbol has no C++ default tail args).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_wo_pair(
        &self,
        a: *const u8,
        a_scale: *const f32,
        wa: *const u8,
        wa_scale: *const u8,
        wa_bias: *const f32,
        na: i32,
        ka: i32,
        wb: *const u8,
        wb_scale: *const u8,
        wb_bias: *const f32,
        nb: i32,
        kb: i32,
        mid: *mut f32,
        out: *mut f32,
        bar: *mut u32,
    ) -> Result<bool> {
        let f = self.need(self.kernels.gemm_fp8_wo_pair, "dsv41_gemm_fp8_wo_pair")?;
        let rc = unsafe {
            f(
                a, a_scale, wa, wa_scale, wa_bias, na, ka, wb, wb_scale, wb_bias, nb, kb, mid,
                out, bar as *mut c_uint, self.stream,
            )
        };
        // A shape/arm decline is 2 (1 collides with cudaErrorInvalidValue).
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_wo_pair")?;
        Ok(true)
    }

    /// chain-pair-batch 链2: true when the loaded .so carries the fused shared
    /// expert chain (`dsv41_gemm_fp8_sh_pair`). A stale .so leaves DSV41_SH_PAIR
    /// inert and the (w1w3, swiglu, w2) three-launch chain runs.
    pub fn supports_sh_pair(&self) -> bool {
        self.kernels.gemm_fp8_sh_pair.is_some()
    }

    /// chain-pair-batch 链2: the shared expert's (w1w3 -> swiglu -> w2) chain as
    /// ONE launch, joined by the kernel's own device-wide sense-reversing barrier
    /// instead of two stream edges. Nothing about the numerics changes: phase 1
    /// is gemm_fp8_mx2's two rows walked by ONE warp with swiglu_limit_q's
    /// epilogue, phase 2 is the standalone w2 GEMV, so `act`, `aq`/`aqsc` and
    /// `out` are bit-identical to the three launches this replaces.
    ///
    /// Ok(false) => the shape/arm cannot use it and the caller keeps the three
    /// calls. `aq`/`aqsc` must be DISJOINT from `a`/`a_scale` (see the FFI note).
    /// ABI: stream LAST (this symbol has no C++ default tail args).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_sh_pair(
        &self,
        a: *const u8,
        a_scale: *const f32,
        wg: *const u8,
        wg_scale: *const u8,
        wu: *const u8,
        wu_scale: *const u8,
        limit: f32,
        n1: i32,
        k1: i32,
        act: *mut f32,
        aq: *mut u8,
        aqsc: *mut f32,
        w2: *const u8,
        w2_scale: *const u8,
        n2: i32,
        out: *mut f32,
        s: CuStream,
    ) -> Result<bool> {
        let f = self.need(self.kernels.gemm_fp8_sh_pair, "dsv41_gemm_fp8_sh_pair")?;
        let rc = unsafe {
            f(
                a, a_scale, wg, wg_scale, wu, wu_scale, limit, n1, k1, act, aq, aqsc, w2, w2_scale,
                n2, out, s,
            )
        };
        // A shape/arm decline is 2 (1 collides with cudaErrorInvalidValue).
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_sh_pair")?;
        Ok(true)
    }

    /// wo_b: the M=1 GEMV that reads the RAW f32 activation, so the consumer's
    /// `quant1(wo)` launch disappears. NOT bit-identical to the fp8 path — it
    /// skips the quantise->dequantise round trip and is strictly more accurate.
    /// Ok(false) => the caller runs `quant1(wo)` + the plain `gemm_fp8_mx`.
    /// ABI: stream LAST (this symbol has no C++ default tail args).
    pub fn gemm_fp8_mx_f32(
        &self,
        a_f32: *const f32,
        w: *const u8,
        w_scale: *const u8,
        bias: *const f32,
        out: *mut f32,
        n: i32,
        k: i32,
    ) -> Result<bool> {
        let f = self.need(self.kernels.gemm_fp8_mx_f32, "dsv41_gemm_fp8_mx_f32")?;
        let rc = unsafe { f(a_f32, w, w_scale, bias, out, n, k, self.stream) };
        // Round-42 fix: decline is 2 now (1 collides with cudaErrorInvalidValue).
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_mx_f32")?;
        Ok(true)
    }

    /// swapAB M=1 GEMV: `out[n] = a[1,k] @ w[n,k]^T` with the weight as the MMA's
    /// M (16 output rows) and the activation as B's column 0. The tensor-core
    /// replacement for the SIMT `gemm_fp8_gemv_kernel` on the plain fp8 path.
    ///
    /// `Ok(true)` = it ran; `Ok(false)` = the shape declined (n % 16 != 0 or
    /// k % 32 != 0) or the .so has no `dsv41_gemm_fp8_swapab`, in which case the
    /// caller keeps `gemm_fp8_mx`. NOT bit-identical to the SIMT gemv.
    ///
    /// `partial` / `ctr` are the K-split last-block-reduction scratch (see the
    /// kernel table above): `ks * n` f32 + one u32 per 16-row tile, `ctr` zeroed
    /// once by the caller. Only touched when the shape actually splits K.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_swapab(
        &self,
        a: *const u8,
        a_scale: *const f32,
        w: *const u8,
        w_scale: *const u8,
        bias: *const f32,
        out: *mut f32,
        n: i32,
        k: i32,
        partial: *mut f32,
        ctr: *mut c_uint,
    ) -> Result<bool> {
        let Some(f) = self.kernels.gemm_fp8_swapab else {
            return Ok(false);
        };
        let rc = unsafe {
            f(a, a_scale, w, w_scale, bias, out, n, k, partial, ctr, self.stream)
        };
        // 2 is the graceful shape decline (1 collides with cudaErrorInvalidValue).
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_swapab")?;
        Ok(true)
    }

    /// True when the loaded .so carries the swapAB M=1 GEMV
    /// (`dsv41_gemm_fp8_swapab`). A stale .so leaves DSV41_SWAPAB inert and the
    /// SIMT gemv runs.
    pub fn supports_gemm_fp8_swapab(&self) -> bool {
        self.kernels.gemm_fp8_swapab.is_some()
    }

    pub fn quant_fp8(
        &self,
        x: *const f32,
        y: *mut u8,
        scale: *mut f32,
        rows: i32,
        cols: i32,
        block: i32,
        round_scale: bool,
    ) -> Result<()> {
        self.quant_fp8_on(x, y, scale, rows, cols, block, round_scale, self.stream)
    }

    /// [`Self::quant_fp8`] issued on `s` instead of the main stream (the MoE
    /// dual chain's shared half quantises `xn` beside the routed fp4 chain).
    pub fn quant_fp8_on(
        &self,
        x: *const f32,
        y: *mut u8,
        scale: *mut f32,
        rows: i32,
        cols: i32,
        block: i32,
        round_scale: bool,
        s: CuStream,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.quant_fp8)(
                x,
                y,
                scale,
                rows,
                cols,
                block,
                round_scale as i32,
                s,
            )
        };
        self.kerr(rc, "dsv41_quant_fp8")
    }

    pub fn quant_fp4(
        &self,
        x: *const f32,
        y: *mut u8,
        scale: *mut f32,
        rows: i32,
        cols: i32,
        block: i32,
        round_scale: bool,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.quant_fp4)(
                x,
                y,
                scale,
                rows,
                cols,
                block,
                round_scale as i32,
                self.stream,
            )
        };
        self.kerr(rc, "dsv41_quant_fp4")
    }

    /// Native fp4 experts (tcgen05 MXFP4): gate and up in one pass.
    #[allow(clippy::too_many_arguments)]
    pub fn expert_gate_up_fp4(
        &self,
        a: *const u8,
        a_scale: *const f32,
        w1: *const u8,
        w1_scale: *const u8,
        w3: *const u8,
        w3_scale: *const u8,
        out: *mut f32,
        rows: i32,
        dim: i32,
        inter: i32,
        limit: f32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.expert_gate_up_fp4)(
                a, a_scale, w1, w1_scale, w3, w3_scale, out, rows, dim, inter, limit, self.stream,
            )
        };
        self.kerr(rc, "dsv41_expert_gate_up_fp4")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn expert_down_fp4(
        &self,
        act: *const f32,
        w2: *const u8,
        w2_scale: *const u8,
        weight: *const f32,
        out: *mut f32,
        rows: i32,
        dim: i32,
        inter: i32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.expert_down_fp4)(act, w2, w2_scale, weight, out, rows, dim, inter, self.stream)
        };
        self.kerr(rc, "dsv41_expert_down_fp4")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn engram_hash(
        &self,
        token_map: *const i32,
        cache: *mut i64,
        primes: *const i64,
        offsets: *const i64,
        multipliers: *const i64,
        input_ids: *const i32,
        mask: *const u8,
        out: *mut i64,
        batch_row: i32,
        seqlen: i32,
        max_seq: i32,
        start_pos: i32,
        n_layers: i32,
        max_ngram: i32,
        n_heads: i32,
        pad_id: i64,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.engram_hash)(
                token_map, cache, primes, offsets, multipliers, input_ids, mask, out, batch_row,
                seqlen, max_seq, start_pos, n_layers, max_ngram, n_heads, pad_id, self.stream,
            )
        };
        self.kerr(rc, "dsv41_engram_hash")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn engram_gather(
        &self,
        table: *const u8,
        table_scale: *const u8,
        hash_ids: *const i64,
        out: *mut f32,
        rows: i32,
        n_cols: i32,
        head_dim: i32,
        part_start: i64,
        part_rows: i64,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.engram_gather)(
                table, table_scale, hash_ids, out, rows, n_cols, head_dim, part_start, part_rows,
                self.stream,
            )
        };
        self.kerr(rc, "dsv41_engram_gather")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attn(
        &self,
        q: *const f32,
        kv: *const f32,
        sink: *const f32,
        idxs: *const i32,
        out: *mut f32,
        b: i32,
        m: i32,
        h: i32,
        d: i32,
        clen: *const c_int,
        window: i32,
        index_topk: i32,
        scale: f32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.sparse_attn)(
                q, kv, sink, idxs, out, b, m, h, d, clen, window, index_topk, scale, self.stream,
            )
        };
        self.kerr(rc, "dsv41_sparse_attn")
    }

    /// P1 (DSV41_SPARSE_OROPE): sparse attention + inverse o-rope + fp8 emission
    /// of the roped output, in ONE launch. Replaces the
    /// `sparse_attn` + `apply_rope_q` (+ `quant1`) triple the caller would
    /// otherwise run; the three-launch sequence is bit-identical, so a decline
    /// (or a missing symbol) is always safe to fall back on.
    ///
    /// `Ok(false)` means "fall back" (symbol absent, or the C launcher's shape
    /// decline returned 1/2/3). `Ok(true)` means this launch produced both the
    /// roped `out` row and the fp8 `xq`/`xsc` of it.
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attn_orope(
        &self,
        q: *const f32,
        kv: *const f32,
        sink: *const f32,
        idxs: *const i32,
        out: *mut f32,
        b: i32,
        m: i32,
        h: i32,
        d: i32,
        clen: *const c_int,
        window: i32,
        index_topk: i32,
        scale: f32,
        cos: *const f32,
        sin: *const f32,
        base: *const c_int,
        rope_rd: i32,
        half: i32,
        mul: i32,
        off: i32,
        step: i32,
        inverse: bool,
        xq: *mut u8,
        xsc: *mut f32,
    ) -> Result<bool> {
        let f = match self.kernels.sparse_attn_orope {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe {
            f(q, kv, sink, idxs, out, b, m, h, d, clen, window, index_topk, scale, cos, sin, base,
              rope_rd, half, mul, off, step, inverse as i32, xq, xsc, self.stream)
        };
        // 1/2/3 are the decline sentinels (see the C launcher); anything else is
        // a real launch error.
        if (1..=3).contains(&rc) {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_sparse_attn_orope")?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn indexer_topk(
        &self,
        q: *const f32,
        index_k: *const f32,
        weights: *const f32,
        candidates: *const u8,
        compress_lens: *const i32,
        out: *mut i32,
        b: i32,
        m: i32,
        nh: i32,
        hd: i32,
        n_pos: i32,
        topk: i32,
        offset: i32,
        softmax_scale: f32,
        head_scale: f32,
        uses_candidates: bool,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.indexer_topk)(
                q, index_k, weights, candidates, compress_lens, out, b, m, nh, hd, n_pos, topk,
                offset, softmax_scale, head_scale, uses_candidates as i32, self.stream,
            )
        };
        self.kerr(rc, "dsv41_indexer_topk")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn candidate_blocks(
        &self,
        logits: *const f32,
        compress_lens: *const i32,
        mask: *mut u8,
        rows: i32,
        n_pos: i32,
        topk_blocks: i32,
        block_size: i32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.candidate_blocks)(
                logits, compress_lens, mask, rows, n_pos, topk_blocks, block_size, self.stream,
            )
        };
        self.kerr(rc, "dsv41_candidate_blocks")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compressor(
        &self,
        x: *const f32,
        wkv: *const u8,
        wkv_scale: *const u8,
        wgate: *const u8,
        wgate_scale: *const u8,
        norm_w: *const f32,
        state_kv: *mut f32,
        state_score: *mut f32,
        latents: *mut f32,
        out_rows: *mut i32,
        b: i32,
        seqlen: i32,
        dim: i32,
        head_dim: i32,
        ratio: i32,
        start_pos: i32,
        pos_ctr: *const c_int,
        eps: f32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.compressor)(
                x, wkv, wkv_scale, wgate, wgate_scale, norm_w, state_kv, state_score, latents,
                out_rows, b, seqlen, dim, head_dim, ratio, start_pos, pos_ctr, eps, self.stream,
            )
        };
        self.kerr(rc, "dsv41_compressor")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_precompute(
        &self,
        cos: *mut f32,
        sin: *mut f32,
        dim: i32,
        seqlen: i32,
        original_seq_len: i32,
        base: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.rope_precompute)(
                cos, sin, dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow,
                self.stream,
            )
        };
        self.kerr(rc, "dsv41_rope_precompute")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_rope(
        &self,
        x: *mut f32,
        cos: *const f32,
        sin: *const f32,
        rows: i32,
        row_len: i32,
        dim: i32,
        half: i32,
        base: *const c_int,
        mul: i32,
        off: i32,
        step: i32,
        inverse: bool,
    ) -> Result<()> {
        self.apply_rope_on(
            x, cos, sin, rows, row_len, dim, half, base, mul, off, step, inverse, self.stream,
        )
    }

    /// [`Self::apply_rope`] issued on `s` instead of the main stream. Used by
    /// the attention dual chain (`DSV41_DUAL_CHAIN`), which runs the kv half —
    /// kv norm + rope — on the runtime's second side stream under the q chain.
    /// Kernel and operands are unchanged, so the result is bit-identical.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_rope_on(
        &self,
        x: *mut f32,
        cos: *const f32,
        sin: *const f32,
        rows: i32,
        row_len: i32,
        dim: i32,
        half: i32,
        base: *const c_int,
        mul: i32,
        off: i32,
        step: i32,
        inverse: bool,
        s: CuStream,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.apply_rope)(
                x, cos, sin, rows, row_len, dim, half, base, mul, off, step, inverse as i32, s,
            )
        };
        self.kerr(rc, "dsv41_apply_rope")
    }

    /// rmsnorm + rope on one row in a single launch; `Ok(false)` when the
    /// loaded .so predates the kernel, so the caller runs the two launches.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_rope(
        &self,
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        cos: *const f32,
        sin: *const f32,
        n: i32,
        dim: i32,
        rope_len: i32,
        half: i32,
        base: *const c_int,
        mul: i32,
        off: i32,
        step: i32,
        inverse: bool,
        eps: f32,
    ) -> Result<bool> {
        self.rmsnorm_rope_on(
            x, w, out, cos, sin, n, dim, rope_len, half, base, mul, off, step, inverse, eps,
            self.stream,
        )
    }

    /// [`Self::rmsnorm_rope`] issued on `s` instead of the main stream. This is
    /// the kv half of the attention dual chain (`DSV41_DUAL_CHAIN`): it reads
    /// `s.kv` (written by `lin2`, i.e. before the fork) and writes it back, so it
    /// shares no buffer with the q chain and stays bit-identical.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_rope_on(
        &self,
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        cos: *const f32,
        sin: *const f32,
        n: i32,
        dim: i32,
        rope_len: i32,
        half: i32,
        base: *const c_int,
        mul: i32,
        off: i32,
        step: i32,
        inverse: bool,
        eps: f32,
        s: CuStream,
    ) -> Result<bool> {
        let f = match self.kernels.rmsnorm_rope {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe {
            f(x, w, out, cos, sin, n, dim, rope_len, half, base, mul, off, step, inverse as i32,
              eps, s)
        };
        self.kerr(rc, "dsv41_rmsnorm_rope")?;
        Ok(true)
    }

    /// bf16 gate + fp8 shared expert in one launch (same activation); `Ok(false)`
    /// when the loaded .so predates the kernel, so the caller runs the separate
    /// launches instead.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_bf16_fp8x2(
        &self,
        wb: *const c_void,
        biasb: *const f32,
        outb: *mut f32,
        nb: i32,
        a: *const u8,
        a_scale: *const f32,
        wf1: *const u8,
        ws1: *const u8,
        outf1: *mut f32,
        nf: i32,
        wf2: *const u8,
        ws2: *const u8,
        outf2: *mut f32,
        x: *const f32,
        k: i32,
    ) -> Result<bool> {
        let f = match self.kernels.gemm_bf16_fp8x2 {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe {
            f(wb, biasb, outb, nb, a, a_scale, wf1, ws1, outf1, nf, wf2, ws2, outf2, x, k,
              self.stream)
        };
        // ⚠️ LEGACY decline contract (rounds 37-41), left as-is: `rc == 1` is
        // both "declined" and `cudaErrorInvalidValue`, and the launcher's
        // SetAttribute failure also returns 1 without clearing the sticky flag -
        // the r42 crash class. Fallback-safe, but a genuine launch failure is
        // reported as a decline. Migrate to decline == 2 if re-touched.
        if rc == 1 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_bf16_fp8x2")?;
        Ok(true)
    }

    /// Cross-rank argmax over a vocabulary-sliced lm_head: local reduce, one
    /// published u64 per rank, a final cross-rank pick. `Ok(false)` when the
    /// loaded .so predates the kernel.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn argmax_sliced(
        &self,
        v: *const f32,
        n: i32,
        idx_off: i32,
        out: *mut c_int,
        packed: *mut u64,
        pos_ctr: *mut c_int,
        peer_staging: *const *mut u64,
        ready_tbl: *const *mut u32,
        epoch: *mut c_uint,
        staging_local: *mut u64,
        ready_local: *const c_uint,
        world: i32,
        rank: i32,
        stride_bytes: i64,
    ) -> Result<bool> {
        let f = match self.kernels.argmax_sliced {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe {
            f(v, n, idx_off, out, packed, pos_ctr, peer_staging, ready_tbl, epoch, staging_local,
              ready_local, world, rank, stride_bytes, self.stream)
        };
        if rc == 1 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_argmax_sliced")?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_mixes(
        &self,
        x: *const f32,
        hc_fn: *const f32,
        hc_scale: *const f32,
        hc_base: *const f32,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        rows: i32,
        hc_dim: i32,
        hc: i32,
        sinkhorn_iters: i32,
        eps: f32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.hc_mixes)(
                x, hc_fn, hc_scale, hc_base, pre, post, comb, rows, hc_dim, hc, sinkhorn_iters,
                eps, self.stream,
            )
        };
        self.kerr(rc, "dsv41_hc_mixes")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_route(
        &self,
        x: *const f32,
        gate_w: *const u8,
        gate_w_scale: *const u8,
        gate_bias: *const f32,
        weights: *mut f32,
        indices: *mut i32,
        hist: *mut i32,
        rows: i32,
        dim: i32,
        n_experts: i32,
        topk: i32,
        gate_temp: f32,
        norm_topk_prob: bool,
        route_scale: f32,
        score_func: i32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.moe_route)(
                x, gate_w, gate_w_scale, gate_bias, weights, indices, hist, rows, dim, n_experts,
                topk, gate_temp, norm_topk_prob as i32, route_scale, score_func, self.stream,
            )
        };
        self.kerr(rc, "dsv41_moe_route")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn window_append(
        &self,
        kv: *const f32,
        cache: *mut u8,
        cache_scale: *mut f32,
        rows: i32,
        head_dim: i32,
        window: i32,
        start_pos: i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.window_append, "dsv41_window_append")?;
        let rc = unsafe { f(kv, cache, cache_scale, rows, head_dim, window, start_pos, self.stream) };
        self.kerr(rc, "dsv41_window_append")
    }

    /// MoE routing from pre-computed (bf16-gate) scores.
    #[allow(clippy::too_many_arguments)]
    pub fn route_topk(
        &self,
        scores: *const f32,
        bias: *const f32,
        weights: *mut f32,
        indices: *mut i32,
        hist: *mut i32,
        rows: i32,
        n_experts: i32,
        topk: i32,
        norm_topk_prob: bool,
        route_scale: f32,
        score_func: i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.route_topk, "dsv41_route_topk")?;
        let rc = unsafe {
            f(
                scores, bias, weights, indices, hist, rows, n_experts, topk,
                norm_topk_prob as i32, route_scale, score_func, self.stream,
            )
        };
        self.kerr(rc, "dsv41_route_topk")
    }

    /// Compressor pooling half (bf16 projections are done by the caller).
    #[allow(clippy::too_many_arguments)]
    pub fn compressor_pool(
        &self,
        kvp: *const f32,
        scp: *const f32,
        norm_w: *const f32,
        state_kv: *mut f32,
        state_score: *mut f32,
        latents: *mut f32,
        out_rows: *mut i32,
        b: i32,
        seqlen: i32,
        head_dim: i32,
        ratio: i32,
        start_pos: i32,
        pos_ctr: *const c_int,
        eps: f32,
    ) -> Result<()> {
        self.compressor_pool_on(
            kvp, scp, norm_w, state_kv, state_score, latents, out_rows, b, seqlen, head_dim,
            ratio, start_pos, pos_ctr, eps, self.stream,
        )
    }

    /// [`Self::compressor_pool`] issued on `s` instead of the main stream. The
    /// compressor's pooling half runs on the third side stream under
    /// `DSV41_COMPRESS_SIDE`; kernel and operands are unchanged, so the result
    /// is bit-identical.
    #[allow(clippy::too_many_arguments)]
    pub fn compressor_pool_on(
        &self,
        kvp: *const f32,
        scp: *const f32,
        norm_w: *const f32,
        state_kv: *mut f32,
        state_score: *mut f32,
        latents: *mut f32,
        out_rows: *mut i32,
        b: i32,
        seqlen: i32,
        head_dim: i32,
        ratio: i32,
        start_pos: i32,
        pos_ctr: *const c_int,
        eps: f32,
        s: CuStream,
    ) -> Result<()> {
        let f = self.need(self.kernels.compressor_pool, "dsv41_compressor_pool")?;
        let rc = unsafe {
            f(
                kvp, scp, norm_w, state_kv, state_score, latents, out_rows, b, seqlen, head_dim,
                ratio, start_pos, pos_ctr, eps, s,
            )
        };
        self.kerr(rc, "dsv41_compressor_pool")
    }

    /// COMPRESS_FUSE: the decode compressor's state carry + gated pool/RMSNorm +
    /// rope/ring commit as ONE 1-block launch, issued on `s` (the third side
    /// stream under `DSV41_COMPRESS_SIDE`, like the pair it replaces). DECODE
    /// ONLY — `b == seqlen == 1` and `ratio > 1`; the caller gates the shape and
    /// the `DSV41_COMPRESS_FUSE` flag, and an older .so without the symbol falls
    /// back to `compressor_pool_on` + `compress_commit_on` (see
    /// [`Self::supports_compress_fuse`]). Kernel and operands are unchanged, so
    /// the bytes are bit-identical - only two launch boundaries disappear.
    #[allow(clippy::too_many_arguments)]
    pub fn compressor_fused_on(
        &self,
        kvp: *const f32,
        scp: *const f32,
        norm_w: *const f32,
        state_kv: *mut f32,
        state_score: *mut f32,
        latent: *mut f32,
        out_rows: *mut i32,
        cos: *const f32,
        sin: *const f32,
        ring: *mut f32,
        clen: *mut std::os::raw::c_int,
        b: i32,
        seqlen: i32,
        head_dim: i32,
        ratio: i32,
        rope_dim: i32,
        half: i32,
        window: i32,
        pos_ctr: *const c_int,
        eps: f32,
        s: CuStream,
    ) -> Result<()> {
        let f = self.need(self.kernels.compressor_fused, "dsv41_compressor_fused")?;
        let rc = unsafe {
            f(
                kvp, scp, norm_w, state_kv, state_score, latent, out_rows, cos, sin, ring, clen, b,
                seqlen, head_dim, ratio, rope_dim, half, window, pos_ctr, eps, s,
            )
        };
        self.kerr(rc, "dsv41_compressor_fused")
    }

    pub fn gather_rows(
        &self,
        src: *const f32,
        idx: *const i32,
        out: *mut f32,
        n: i32,
        dim: i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.gather_rows, "dsv41_gather_rows")?;
        let rc = unsafe { f(src, idx, out, n, dim, self.stream) };
        self.kerr(rc, "dsv41_gather_rows")
    }

    pub fn scatter_add_rows(
        &self,
        src: *const f32,
        idx: *const i32,
        weight: *const f32,
        dst: *mut f32,
        n: i32,
        dim: i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.scatter_add_rows, "dsv41_scatter_add_rows")?;
        let rc = unsafe { f(src, idx, weight, dst, n, dim, self.stream) };
        self.kerr(rc, "dsv41_scatter_add_rows")
    }

    pub fn swiglu_limit(&self, gate_up: *mut f32, rows: i32, inter: i32, limit: f32) -> Result<()> {
        self.swiglu_limit_on(gate_up, rows, inter, limit, self.stream)
    }

    /// [`Self::swiglu_limit`] issued on `s` instead of the main stream (the MoE
    /// dual chain's shared half, on its `SWIGLU_Q`-off fallback).
    pub fn swiglu_limit_on(
        &self,
        gate_up: *mut f32,
        rows: i32,
        inter: i32,
        limit: f32,
        s: CuStream,
    ) -> Result<()> {
        let f = self.need(self.kernels.swiglu_limit, "dsv41_swiglu_limit")?;
        let rc = unsafe { f(gate_up, rows, inter, limit, s) };
        self.kerr(rc, "dsv41_swiglu_limit")
    }

    /// A4: true when the loaded .so carries the swiglu+fp8 fused epilogue. A
    /// stale .so leaves DSV41_SWIGLU_Q inert and the (swiglu_limit, quant1) pair
    /// runs.
    pub fn supports_swiglu_q(&self) -> bool {
        self.kernels.swiglu_limit_q.is_some()
    }

    /// A4: swiglu + clamp + the fp8 pair the next GEMV reads, in one launch.
    /// Ok(false) => the caller runs swiglu_limit and then quant1, as before.
    pub fn swiglu_limit_q(
        &self,
        gate_up: *mut f32,
        rows: i32,
        inter: i32,
        limit: f32,
        xq: *mut u8,
        xsc: *mut f32,
    ) -> Result<bool> {
        self.swiglu_limit_q_on(gate_up, rows, inter, limit, xq, xsc, self.stream)
    }

    /// [`Self::swiglu_limit_q`] issued on `s` instead of the main stream (the
    /// MoE dual chain's shared half).
    pub fn swiglu_limit_q_on(
        &self,
        gate_up: *mut f32,
        rows: i32,
        inter: i32,
        limit: f32,
        xq: *mut u8,
        xsc: *mut f32,
        s: CuStream,
    ) -> Result<bool> {
        let f = self.need(self.kernels.swiglu_limit_q, "dsv41_swiglu_limit_q")?;
        let rc = unsafe { f(gate_up, rows, inter, limit, xq, xsc, s) };
        if rc == 1 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_swiglu_limit_q")?;
        Ok(true)
    }

    pub fn hc_collapse(
        &self,
        x: *const f32,
        pre: *const f32,
        out: *mut f32,
        rows: i32,
        hc: i32,
        dim: i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.hc_collapse, "dsv41_hc_collapse")?;
        let rc = unsafe { f(x, pre, out, rows, hc, dim, self.stream) };
        self.kerr(rc, "dsv41_hc_collapse")
    }

    /// Stamp `round` into every rank's stamp array (including our own) after the
    /// data peer copies have completed on this stream.
    pub fn ar_stamp(&self, peer_stamps: *const u64, world: i32, rank: i32, round: u32) -> Result<()> {
        let f = self.need(self.kernels.ar_stamp, "dsv41_ar_stamp")?;
        let rc = unsafe { f(peer_stamps, world, rank, round, self.stream) };
        self.kerr(rc, "dsv41_ar_stamp")
    }

    /// Begin capturing work queued on this device's stream.
    pub fn ar_store2(
        &self,
        peer_slots: *const u64,
        world: i32,
        rank: i32,
        src: *const f32,
        n: i64,
        slot_f: i64,
        parity_off: i64,
        reduced: *const c_uint,
        round: u32,
        peer_stamps: *const u64,
        ctr: *mut c_uint,
    ) -> Result<()> {
        let f = self.need(self.kernels.ar_store2, "dsv41_ar_store2")?;
        let rc = unsafe {
            f(peer_slots, world, rank, src, n, slot_f, parity_off, reduced, round, peer_stamps, ctr,
              self.stream)
        };
        self.kerr(rc, "dsv41_ar_store2")
    }

    /// Reduce with the same-threads release: the kernel that reads the slots also
    /// announces `reduced` once every block is done.
    #[allow(clippy::too_many_arguments)]
    pub fn ar_reduce2(
        &self,
        dst: *mut f32,
        staging: *const f32,
        n: i64,
        slot_f: i64,
        world: i32,
        stamps: *const c_uint,
        round: u32,
        peer_reduced: *const u64,
        rank: i32,
        ctr2: *mut c_uint,
        do_mark: i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.ar_reduce2, "dsv41_ar_reduce2")?;
        let rc = unsafe {
            f(dst, staging, n, slot_f, world, stamps, round, peer_reduced, rank, ctr2, do_mark,
              self.stream)
        };
        self.kerr(rc, "dsv41_ar_reduce2")
    }

    /// Announce that this rank has finished reducing `round`.
    pub fn ar_mark(&self, peer_reduced: *const u64, world: i32, rank: i32, round: u32) -> Result<()> {
        let f = self.need(self.kernels.ar_mark, "dsv41_ar_mark")?;
        let rc = unsafe { f(peer_reduced, world, rank, round, self.stream) };
        self.kerr(rc, "dsv41_ar_mark")
    }

    /// Lean M=1 GEMV over bf16 weights with an f32 activation (out[n] = W[n,k]·x).
    /// Replaces cuBLAS's gemv2T path, which ran at ~40 GFLOP/s for a single row.
    ///
    /// Small-n dispatch: the v1 kernel gives one warp per output row at 8 rows /
    /// block, so a latency-bound small matrix gets only n/8 blocks (the MoE gate
    /// at n=384 -> 48 of 148 SMs, 12.5% occupancy, scalar 160-iteration K loop,
    /// measured 17.2us). `gemv_bf16_v2` vectorizes (uint4 = 8 bf16/load) and
    /// K-splits across WPR warps per row, so the gate becomes 384 blocks x 8
    /// warps. Only shapes below `GEMV_V2_MAX_N` are diverted: the lm_head
    /// (n = vocab / per-rank slice, >= 16k rows) keeps the v1 path, whose
    /// accumulation order is bit-exact with the pre-v2 baseline. v2's K-slice
    /// partials change f32 summation order by ~1e-6 (documented in
    /// ferrite_kernels.cu), which is far below any routing/top-k threshold.
    /// `DSV41_GEMV_V2=0` pins v1 (A/B escape hatch).
    pub fn gemv_bf16(&self, w: *const c_void, x: *const f32, out: *mut f32, n: i32, k: i32) -> Result<()> {
        if gemv_bf16_v2_wanted(n) {
            if let Some(f) = self.kernels.gemv_bf16_v2 {
                // ABI differs from v1: (x, w, bias, out, in_f=k, out_f=n, nrows=1).
                let rc = unsafe { f(x, w, std::ptr::null(), out, k, n, 1, self.stream) };
                return self.kerr(rc, "ferrite_gemv_bf16_v2");
            }
        }
        let f = self.need(self.kernels.gemv_bf16, "dsv41_gemv_bf16")?;
        let rc = unsafe { f(w, x, out, n, k, self.stream) };
        self.kerr(rc, "dsv41_gemv_bf16")
    }

    /// Fused M=1 gate GEMV + MoE route: ONE launch where `gemv_bf16_command` +
    /// `route_topk` used to be two. `ferrite_gemv_bf16_v2_route` runs the same
    /// GEMV (same WPR shape, same accumulation -> the scores are bit-identical
    /// to `gemv_bf16`) and its LAST block — the "last block" election on
    /// `ctr`, cf. `router_gemm_route_fused_kernel` — then runs
    /// `dsv41_route_topk`'s body over the finished score row, bit-exactly.
    ///
    /// Removes 40 launches/step (the audit's 5.2us x 40 = 0.21ms + 40 graph
    /// nodes). `ctr` must be 4B of device memory zeroed ONCE at allocation:
    /// the kernel resets it in place before returning, which is what makes a
    /// captured graph replay correct without a per-call memset.
    ///
    /// Returns Ok(false) when the fused symbol or shape is unavailable — the
    /// CALLER must then run the gate and `route_topk` as two launches. The
    /// preconditions mirror `ferrite_gemv_bf16_v2`'s v1 fallback: `nrows == 1`
    /// and `k % 8 == 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_bf16_route(
        &self,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        n: i32,
        k: i32,
        weights: *mut f32,
        indices: *mut i32,
        route_bias: *const f32,
        topk: i32,
        norm_topk_prob: bool,
        route_scale: f32,
        score_func: i32,
        ctr: *mut c_uint,
    ) -> Result<bool> {
        if !gemv_bf16_v2_wanted(n) || (k & 7) != 0 || n <= 0 || topk <= 0 || topk > n {
            return Ok(false);
        }
        let Some(f) = self.kernels.gemv_bf16_v2_route else {
            return Ok(false); // older .so: keep the two-launch pair
        };
        // ABI: (x, w, bias, out, in_f=k, out_f=n, nrows=1, route out/bias/params, ctr, s).
        let rc = unsafe {
            f(
                x,
                w,
                std::ptr::null(),
                out,
                k,
                n,
                1,
                weights,
                indices,
                route_bias,
                topk,
                norm_topk_prob as i32,
                route_scale,
                score_func,
                ctr,
                self.stream,
            )
        };
        self.kerr(rc, "ferrite_gemv_bf16_v2_route")?;
        Ok(true)
    }

    /// Same, f32 weights.
    /// The decode-step n-gram hash on the device (removes the last per-step H2D
    /// and makes the step graph-capturable). Single thread, bit-identical to the
    /// host reference.
    #[allow(clippy::too_many_arguments)]
    pub fn engram_hash_step(
        &self,
        map: *const i64,
        cache: *mut i64,
        mults: *const i64,
        lms: *const u64,
        offs: *const u64,
        eng_ids: *mut i64,
        token: *const i32,
        pos_ctr: *const i32,
        map_len: i64,
        n_layers: c_int,
        max_ngram: c_int,
        n_heads: c_int,
        pad_id: i64,
    ) -> Result<()> {
        let f = self.need(self.kernels.engram_hash_step, "dsv41_engram_hash_step")?;
        let rc = unsafe {
            f(map, cache, mults, lms, offs, eng_ids, token, pos_ctr, map_len, n_layers, max_ngram,
              n_heads, pad_id, self.stream)
        };
        self.kerr(rc, "dsv41_engram_hash_step")
    }

    /// The SHARED all-reduce v5 entry (`ferrite_p2p_ar_v5`, ferrite_kernels.cu):
    /// one call = store + publish/fused-reduce. DSV41 passes exactly the shapes
    /// GLM does — pointer tables of the peers' staging bases (`staging_tbl`) and
    /// ready rows (`ready_tbl`), this rank's own `staging_local` base and
    /// `ready_local` row, and a DEVICE `epoch` the kernels read at runtime (which
    /// is what makes a captured graph replay). `stride` is the per-slot element
    /// count; `out` may alias `partial` (the store kernel has fully consumed it
    /// at the kernel boundary).
    #[allow(clippy::too_many_arguments)]
    pub fn p2p_ar_v5(
        &self,
        partial: *const f32,
        staging_tbl: *const *mut f32,
        ready_tbl: *const *mut u32,
        epoch: *mut c_uint,
        staging_local: *const f32,
        ready_local: *const c_uint,
        out: *mut f32,
        n: c_int,
        world: c_int,
        my_rank: c_int,
        stride: c_int,
    ) -> Result<()> {
        let f = self.need(self.kernels.p2p_ar_v5, "ferrite_p2p_ar_v5")?;
        let rc = unsafe {
            f(partial, staging_tbl, ready_tbl, epoch, staging_local, ready_local, out, n,
              world, my_rank, stride, self.stream)
        };
        self.kerr(rc, "ferrite_p2p_ar_v5")
    }

    /// The publish+reduce half of AR v5, with NO store (`ferrite_p2p_ar_pubred_v5`).
    /// Used after a producer kernel fused the staging store into its epilogue (see
    /// `gemm_fp8_mx_ar`). The caller must have launched that producer on the same
    /// stream with no intervening all-reduce.
    #[allow(clippy::too_many_arguments)]
    pub fn p2p_ar_pubred_v5(
        &self,
        ready_tbl: *const *mut u32,
        epoch: *mut c_uint,
        staging_local: *const f32,
        ready_local: *const c_uint,
        out: *mut f32,
        n: c_int,
        world: c_int,
        my_rank: c_int,
        stride: c_int,
    ) -> Result<()> {
        let f = self.need(self.kernels.p2p_ar_pubred_v5, "ferrite_p2p_ar_pubred_v5")?;
        let rc = unsafe {
            f(ready_tbl, epoch, staging_local, ready_local, out, n, world, my_rank, stride,
              self.stream)
        };
        self.kerr(rc, "ferrite_p2p_ar_pubred_v5")
    }

    /// AR v5 with the segment-C `hc_post_inplace` folded into the pubred
    /// epilogue (`ferrite_p2p_ar_v5_hcpost`). Same shapes as [`Self::p2p_ar_v5`]
    /// plus the residual stream `hc_res` (`[hc_n][hc_h]`, row stride `hc_h`) and
    /// the hyper-connection `hc_post`/`hc_comb`, written straight back onto
    /// `hc_res` in the same ascending-k order the standalone kernel uses. `n`
    /// (the payload element count) must equal `hc_h`.
    ///
    /// `Ok(false)` when the loaded .so predates the entry (or the launcher
    /// rejects the shape, returns 1), so the caller runs the plain
    /// `all_reduce_inplace` + `hc_post_inplace` pair instead.
    #[allow(clippy::too_many_arguments)]
    pub fn p2p_ar_v5_hcpost(
        &self,
        partial: *const f32,
        staging_tbl: *const *mut f32,
        ready_tbl: *const *mut u32,
        epoch: *mut c_uint,
        staging_local: *const f32,
        ready_local: *const c_uint,
        out: *mut f32,
        n: c_int,
        world: c_int,
        my_rank: c_int,
        stride: c_int,
        hc_res: *mut f32,
        hc_post: *const f32,
        hc_comb: *const f32,
        hc_n: c_int,
        hc_h: c_int,
    ) -> Result<bool> {
        let f = match self.kernels.p2p_ar_v5_hcpost {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe {
            f(partial, staging_tbl, ready_tbl, epoch, staging_local, ready_local, out, n,
              world, my_rank, stride, hc_res, hc_post, hc_comb, hc_n, hc_h,
              self.stream)
        };
        // 1 == the launcher declined the shape (see `ferrite_p2p_ar_v5_hcpost`);
        // the caller then runs the unfused pair as before.
        if rc == 1 {
            return Ok(false);
        }
        self.kerr(rc, "ferrite_p2p_ar_v5_hcpost")?;
        Ok(true)
    }

    /// `ferrite_p2p_ar_v5_add` — AR v5 with the elementwise residual folded into
    /// the store epilogue. `bias` is published as `partial[i] + bias[i]` (see the
    /// kernel note); the reduce is the unchanged v5 one, so the result is
    /// bit-identical to `add_inplace` + `p2p_ar_v5`. `Ok(false)` on a stale .so
    /// (no symbol), so the caller keeps the standalone add.
    #[allow(clippy::too_many_arguments)]
    pub fn p2p_ar_v5_add(
        &self,
        partial: *const f32,
        bias: *const f32,
        staging_tbl: *const *mut f32,
        ready_tbl: *const *mut u32,
        epoch: *mut c_uint,
        staging_local: *const f32,
        ready_local: *const c_uint,
        out: *mut f32,
        n: c_int,
        world: c_int,
        my_rank: c_int,
        stride: c_int,
    ) -> Result<bool> {
        let f = match self.kernels.p2p_ar_v5_add {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe {
            f(partial, bias, staging_tbl, ready_tbl, epoch, staging_local, ready_local, out,
              n, world, my_rank, stride, self.stream)
        };
        if rc == 1 {
            return Ok(false);
        }
        self.kerr(rc, "ferrite_p2p_ar_v5_add")?;
        Ok(true)
    }

    /// `ferrite_p2p_ar_v5_hcpost_add` — the hc-post fold AND the ADD_EPI
    /// residual in one launch. `Ok(false)` on a stale .so or a declined shape;
    /// the caller then falls back to the unfused path.
    #[allow(clippy::too_many_arguments)]
    pub fn p2p_ar_v5_hcpost_add(
        &self,
        partial: *const f32,
        bias: *const f32,
        staging_tbl: *const *mut f32,
        ready_tbl: *const *mut u32,
        epoch: *mut c_uint,
        staging_local: *const f32,
        ready_local: *const c_uint,
        out: *mut f32,
        n: c_int,
        world: c_int,
        my_rank: c_int,
        stride: c_int,
        hc_res: *mut f32,
        hc_post: *const f32,
        hc_comb: *const f32,
        hc_n: c_int,
        hc_h: c_int,
    ) -> Result<bool> {
        let f = match self.kernels.p2p_ar_v5_hcpost_add {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe {
            f(partial, bias, staging_tbl, ready_tbl, epoch, staging_local, ready_local, out,
              n, world, my_rank, stride, hc_res, hc_post, hc_comb, hc_n, hc_h,
              self.stream)
        };
        if rc == 1 {
            return Ok(false);
        }
        self.kerr(rc, "ferrite_p2p_ar_v5_hcpost_add")?;
        Ok(true)
    }

    /// Stable argmax (ties -> lowest index); writes the winning index as i32 and
    /// advances the device position counter (the argmax is the step's last
    /// kernel, so the counter is stable during the step).
    pub fn argmax(&self, v: *const f32, out: *mut c_int, n: i32, pos_ctr: *mut c_int) -> Result<()> {
        let f = self.need(self.kernels.argmax, "dsv41_argmax")?;
        let rc = unsafe { f(v, out, n, pos_ctr, self.stream) };
        self.kerr(rc, "dsv41_argmax")
    }

    /// The decode-step window indices (the trailing window in ring-slot order),
    /// read from the device position counter - replaces the host computation
    /// and its per-layer H2D upload.
    pub fn window_idxs(&self, idxs: *mut i32, pos_ctr: *const c_int, window: i32) -> Result<()> {
        let f = self.need(self.kernels.window_idxs, "dsv41_window_idxs")?;
        let rc = unsafe { f(idxs, pos_ctr, window, self.stream) };
        self.kerr(rc, "dsv41_window_idxs")
    }

    /// Publish the roped index key into the owner's group slot, with the slot
    /// derived from the DEVICE latent counter (a host-computed destination would
    /// be frozen by a graph capture - the same class as the ring append).
    pub fn index_k_publish(
        &self,
        dst_base: *mut f32,
        src: *const f32,
        clen: *const c_int,
        idx_hd: i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.index_k_publish, "dsv41_index_k_publish")?;
        let rc = unsafe { f(dst_base, src, clen, idx_hd, self.stream) };
        self.kerr(rc, "dsv41_index_k_publish")
    }

    /// Append the KV row into the window ring at the DEVICE-derived slot. This
    /// replaces a cudaMemcpy whose destination address was host-computed and
    /// therefore frozen by a graph capture (every replay wrote the same slot).
    pub fn ring_append(
        &self,
        ring: *mut f32,
        kv: *const f32,
        pos_ctr: *const c_int,
        window: i32,
        hd: i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.ring_append, "dsv41_ring_append")?;
        let rc = unsafe { f(ring, kv, pos_ctr, window, hd, self.stream) };
        self.kerr(rc, "dsv41_ring_append")
    }

    /// B2: apply_rope whose epilogue emits the fp8 of the whole roped region
    /// (byte + per-32-block scale), the exact pair `quant_fp8(o)` would have
    /// produced. `Ok(false)` means the .so predates the symbol or the shape
    /// declined - the caller then runs the (apply_rope, quant1) pair.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_rope_q(
        &self,
        x: *mut f32,
        cos: *const f32,
        sin: *const f32,
        rows: i32,
        row_len: i32,
        dim: i32,
        half: i32,
        base: *const c_int,
        mul: i32,
        off: i32,
        step: i32,
        inverse: bool,
        xq: *mut u8,
        xsc: *mut f32,
    ) -> Result<bool> {
        let f = match self.kernels.apply_rope_q {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe {
            f(x, cos, sin, rows, row_len, dim, half, base, mul, off, step, inverse as i32, xq, xsc,
              self.stream)
        };
        // 1 == shape declined (caller falls back); any other non-zero is a real
        // launch error.
        if rc == 1 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_apply_rope_q")?;
        Ok(true)
    }

    /// B2: one launch for the ring append and the window indices. `ring` may be
    /// null (a consumer layer that does not own its store) - the indices half
    /// still runs, which is what removes the standalone `window_idxs` launch for
    /// every layer. `Ok(false)` means the .so lacks the symbol and the caller
    /// must run `ring_append` + `window_idxs` as before.
    pub fn ring_win_fuse(
        &self,
        ring: *mut f32,
        kv: *const f32,
        pos_ctr: *const c_int,
        window: i32,
        hd: i32,
        idxs: *mut i32,
    ) -> Result<bool> {
        let f = match self.kernels.ring_win_fuse {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe { f(ring, kv, pos_ctr, window, hd, idxs, self.stream) };
        self.kerr(rc, "dsv41_ring_win_fuse")?;
        Ok(true)
    }

    /// B3: [`Self::ring_win_fuse`] whose epilogue ALSO writes the
    /// `comp_placeholder` recency block (`idxs[window + j] = window + *clen -
    /// take + j`, `take = min(*clen, index_topk)`) for `j < take` - one launch
    /// and one graph node fewer per layer. The two `idxs` blocks are disjoint and
    /// the bound is read from the DEVICE counter by both kernels, so the fused
    /// launch is bit-identical to the pair it replaces.
    ///
    /// `clen == null` (with `index_topk == 0`) keeps the placeholder half off,
    /// which reproduces `ring_win_fuse` byte for byte - the caller uses that for a
    /// layer whose [window, ..) block belongs to someone else (an index-source
    /// layer's indexer, or a compress source whose counter this step's compressor
    /// is still advancing). `Ok(false)` means the .so lacks
    /// `dsv41_ring_win_fuse_ph`; the caller then runs the `ring_win_fuse` +
    /// `comp_placeholder` pair as before.
    #[allow(clippy::too_many_arguments)]
    pub fn ring_win_fuse_ph(
        &self,
        ring: *mut f32,
        kv: *const f32,
        pos_ctr: *const c_int,
        window: i32,
        hd: i32,
        idxs: *mut i32,
        clen: *const c_int,
        index_topk: i32,
    ) -> Result<bool> {
        let f = match self.kernels.ring_win_fuse_ph {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe { f(ring, kv, pos_ctr, window, hd, idxs, clen, index_topk, self.stream) };
        self.kerr(rc, "dsv41_ring_win_fuse_ph")?;
        Ok(true)
    }

    /// The fused compressor commit: reads `out_rows` on the device, ropes the
    /// latent at (*clen) * ratio, stores it into the ring at row window + *clen
    /// and advances the counter. Replaces the host's download + branch + rope +
    /// copy, which was a sync D2H per layer per step and uncapturable.
    #[allow(clippy::too_many_arguments)]
    pub fn compress_commit(
        &self,
        latent: *const f32,
        cos: *const f32,
        sin: *const f32,
        ring: *mut f32,
        out_rows: *const c_int,
        clen: *mut c_int,
        hd: i32,
        rope_dim: i32,
        half: i32,
        window: i32,
        ratio: i32,
    ) -> Result<()> {
        self.compress_commit_on(
            latent, cos, sin, ring, out_rows, clen, hd, rope_dim, half, window, ratio,
            self.stream,
        )
    }

    /// [`Self::compress_commit`] issued on `s` instead of the main stream. The
    /// commit (rope + ring store + counter bump) runs on the third side stream
    /// under `DSV41_COMPRESS_SIDE`; kernel and operands are unchanged, so the
    /// result is bit-identical.
    #[allow(clippy::too_many_arguments)]
    pub fn compress_commit_on(
        &self,
        latent: *const f32,
        cos: *const f32,
        sin: *const f32,
        ring: *mut f32,
        out_rows: *const c_int,
        clen: *mut c_int,
        hd: i32,
        rope_dim: i32,
        half: i32,
        window: i32,
        ratio: i32,
        s: CuStream,
    ) -> Result<()> {
        let f = self.need(self.kernels.compress_commit, "dsv41_compress_commit")?;
        let rc = unsafe {
            f(latent, cos, sin, ring, out_rows, clen, hd, rope_dim, half, window, ratio, s)
        };
        self.kerr(rc, "dsv41_compress_commit")
    }

    /// The recency placeholder for the compressed rows (the no-indexer safety
    /// net): idxs[win + j] = win + clen - take + j - replaces an upload.
    pub fn comp_placeholder(
        &self,
        idxs: *mut i32,
        clen: *const c_int,
        window: i32,
        index_topk: i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.comp_placeholder, "dsv41_comp_placeholder")?;
        let rc = unsafe { f(idxs, clen, window, index_topk, self.stream) };
        self.kerr(rc, "dsv41_comp_placeholder")
    }

    /// f32-weight M=1 GEMV. Only user: the compressor's kvp/scp projections
    /// (`chain_dev.rs::lin_f32`, n = head_dim = 128, k = dim = 5120).
    ///
    /// Small-n shapes dispatch to the vectorized + K-split v2 kernel (see
    /// `dsv41_gemv_f32_v2` in dsv41_glue.cu): v1 launches 16 blocks / 128
    /// warps (11% of the SMs) with 160 serial 4B loads per warp, i.e. 48x
    /// above the 0.34us HBM floor for this 2.6MB weight. v2 splits K across
    /// WPR warps per row (smem fold) and uses float4 loads → 1024 warps.
    /// Per-element rounding is pinned to v1's fused FFMA via __fmaf_rn, so
    /// only the K-slice fold order changes (~1e-6 f32). Larger f32 GEMVs
    /// (none exist today) keep v1's bit-exact path. `DSV41_GEMV_F32_V2=0`
    /// pins v1 (A/B escape hatch).
    pub fn gemv_f32(&self, w: *const f32, x: *const f32, out: *mut f32, n: i32, k: i32) -> Result<()> {
        self.gemv_f32_on(w, x, out, n, k, self.stream)
    }

    /// [`Self::gemv_f32`] issued on `s` instead of the main stream. The
    /// compressor's kvp/scp projections (`DSV41_COMPRESS_SIDE`) run the f32
    /// GEMV on the third side stream; kernel and operands are unchanged, so the
    /// result is bit-identical.
    pub fn gemv_f32_on(
        &self,
        w: *const f32,
        x: *const f32,
        out: *mut f32,
        n: i32,
        k: i32,
        s: CuStream,
    ) -> Result<()> {
        if gemv_f32_v2_wanted(n) {
            if let Some(f) = self.kernels.gemv_f32_v2 {
                let rc = unsafe { f(w, x, out, n, k, s) };
                return self.kerr(rc, "dsv41_gemv_f32_v2");
            }
        }
        let f = self.need(self.kernels.gemv_f32, "dsv41_gemv_f32")?;
        let rc = unsafe { f(w, x, out, n, k, s) };
        self.kerr(rc, "dsv41_gemv_f32")
    }

    /// Indirect expert gate/up: the weights come from the per-layer pools plus the
    /// device-side expert id, so the launch arguments do not depend on the routing.
    #[allow(clippy::too_many_arguments)]
    pub fn expert_gate_up_fp4_indirect(
        &self,
        a: *const u8,
        a_scale: *const f32,
        out: *mut f32,
        rows: i32,
        dim: i32,
        inter: i32,
        limit: f32,
        w1_base: *const u8,
        w1_stride: i64,
        w1s_base: *const u8,
        w1s_stride: i64,
        w3_base: *const u8,
        w3_stride: i64,
        w3s_base: *const u8,
        w3s_stride: i64,
        ids: *const i32,
        slot: i32,
    ) -> Result<()> {
        let f = self.need(
            self.kernels.expert_gate_up_fp4_indirect,
            "dsv41_expert_gate_up_fp4_indirect",
        )?;
        let rc = unsafe {
            f(
                a, a_scale, out, rows, dim, inter, limit, w1_base, w1_stride, w1s_base, w1s_stride,
                w3_base, w3_stride, w3s_base, w3s_stride, ids, slot, self.stream,
            )
        };
        self.kerr(rc, "dsv41_expert_gate_up_fp4_indirect")
    }

    /// Indirect expert down; accumulates into `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn expert_down_fp4_indirect(
        &self,
        act: *const f32,
        out: *mut f32,
        rows: i32,
        dim: i32,
        inter: i32,
        row_weight: *const f32,
        w2_base: *const u8,
        w2_stride: i64,
        w2s_base: *const u8,
        w2s_stride: i64,
        ids: *const i32,
        slot: i32,
    ) -> Result<()> {
        let f = self.need(
            self.kernels.expert_down_fp4_indirect,
            "dsv41_expert_down_fp4_indirect",
        )?;
        let rc = unsafe {
            f(
                act, out, rows, dim, inter, row_weight, w2_base, w2_stride, w2s_base, w2s_stride,
                ids, slot, self.stream,
            )
        };
        self.kerr(rc, "dsv41_expert_down_fp4_indirect")
    }

    /// Batched indirect expert gate/up: ONE launch covers all `slots` top-k
    /// slots (grid.y = slot). `out` holds `slots` consecutive [2*inter] blocks,
    /// `out_slot_stride` floats apart; the slots never share a written element.
    /// DSV41_MOE_BATCH only — see `moe_batch()` in chain_dev.rs.
    #[allow(clippy::too_many_arguments)]
    pub fn expert_gate_up_fp4_batched(
        &self,
        a: *const u8,
        a_scale: *const f32,
        out: *mut f32,
        out_slot_stride: i64,
        rows: i32,
        dim: i32,
        inter: i32,
        limit: f32,
        slots: i32,
        w1_base: *const u8,
        w1_stride: i64,
        w1s_base: *const u8,
        w1s_stride: i64,
        w3_base: *const u8,
        w3_stride: i64,
        w3s_base: *const u8,
        w3s_stride: i64,
        ids: *const i32,
        ilv: i32,
    ) -> Result<()> {
        let f = self.need(
            self.kernels.expert_gate_up_fp4_batched,
            "dsv41_expert_gate_up_fp4_batched",
        )?;
        let rc = unsafe {
            f(
                a, a_scale, out, out_slot_stride, rows, dim, inter, limit, slots, w1_base,
                w1_stride, w1s_base, w1s_stride, w3_base, w3_stride, w3s_base, w3s_stride, ids, ilv,
                self.stream,
            )
        };
        self.kerr(rc, "dsv41_expert_gate_up_fp4_batched")
    }

    /// tcgen05 MXFP4 gate/up — the Phase-1 `kind::mxf4` swapAB kernel
    /// (`DSV41_TCGEN05_GATEUP_MXF4`, default OFF, read once inside the `.so`).
    ///
    /// `w`/`w_scale` are ONE expert's contiguous `[2*inter, dim/2]` gate|up pool
    /// and its `[2*inter, dim/32]` e8m0 scales (rows `[0, inter)` = gate, then
    /// up); `act`/`act_scale` are the shared quantised row (packed e2m1 and
    /// `[dim/32]` e8m0 BYTES — see the kernels.rs ABI note); `out` holds `slots`
    /// `[2*inter]` blocks, `out_slot_stride` floats apart, with the `limit`
    /// clamp applied in the epilogue (`split == inter`).
    ///
    /// Returns `Ok(false)` when the entry did not run — the `.so` gate is OFF,
    /// or the launcher rejected the shape and swallowed the error — so a caller
    /// can keep the proven GEMV path. Anything else is an error.
    ///
    /// ⚠️ The routed MoE cannot use this yet: it feeds four separate pools
    /// (w1/w3 + scales) selected by device-side ids and f32 activation scales,
    /// while this launcher takes one direct contiguous pool and e8m0 bytes.
    /// Until the Phase-2 indirect launcher lands, this entry serves the parity
    /// harness / microbench only (see `supports_expert_tcgen05_mxf4`).
    #[allow(clippy::too_many_arguments)]
    pub fn expert_tcgen05_gate_up_mxf4(
        &self,
        w: *const u8,
        w_scale: *const u8,
        act: *const u8,
        act_scale: *const u8,
        out: *mut f32,
        out_slot_stride: i64,
        inter: i32,
        dim: i32,
        limit: f32,
        slots: i32,
    ) -> Result<bool> {
        let f = self.need(
            self.kernels.expert_tcgen05_gate_up_mxf4,
            "dsv41_expert_tcgen05_gate_up_mxf4",
        )?;
        let rc = unsafe {
            f(
                w, w_scale, act, act_scale, out, out_slot_stride, inter, dim, limit, slots,
                self.stream,
            )
        };
        if rc == 0 {
            return Ok(false); // gate OFF, or the launcher declined the shape
        }
        self.kerr(rc, "dsv41_expert_tcgen05_gate_up_mxf4")?;
        Ok(true)
    }

    /// True when the loaded `.so` carries the tcgen05 MXFP4 gate/up entry point
    /// (`dsv41_expert_tcgen05_gate_up_mxf4`). The stock `build.sh` defines no
    /// `DSV41_TCGEN05_GATEUP_MXF4_SKELETON`, so this is false there and the
    /// proven GEMV/GEMM path stays in force — the same staged-bring-up contract
    /// as `supports_moe_batch` / `supports_down_fuse`.
    pub fn supports_expert_tcgen05_mxf4(&self) -> bool {
        self.kernels.expert_tcgen05_gate_up_mxf4.is_some()
    }

    /// Interleave one expert's gate/up fp4 blocks on the device (DSV41_EXPERT_ILV,
    /// load-time only): `dst[16i..16i+8) = g[8i..8i+8)`, `dst[16i+8..16i+16) =
    /// u[8i..8i+8)` for `bytes/8` granules. A pure permutation — bit-identical
    /// weights, half the load instructions in the fused gate/up GEMV.
    pub fn interleave_gateup_fp4(
        &self,
        g: *const u8,
        u: *const u8,
        dst: *mut u8,
        bytes: i64,
    ) -> Result<()> {
        let f = self.need(
            self.kernels.interleave_gateup_fp4,
            "dsv41_interleave_gateup_fp4",
        )?;
        let rc = unsafe { f(g, u, dst, bytes, self.stream) };
        self.kerr(rc, "dsv41_interleave_gateup_fp4")
    }

    /// Batched indirect expert down: ONE launch covers all `slots` slots and
    /// WRITES the [slots][dim] scratch (no accumulation across slots). The
    /// routing weight is PER SLOT (read at `row_weight[slot * rw_stride]`).
    /// `act_base` holds `slots` [2*inter] slices `act_stride` floats apart.
    #[allow(clippy::too_many_arguments)]
    pub fn expert_down_fp4_batched(
        &self,
        act_base: *const f32,
        act_stride: i64,
        out: *mut f32,
        out_slot_stride: i64,
        rows: i32,
        dim: i32,
        inter: i32,
        row_weight: *const f32,
        rw_stride: i64,
        slots: i32,
        w2_base: *const u8,
        w2_stride: i64,
        w2s_base: *const u8,
        w2s_stride: i64,
        ids: *const i32,
    ) -> Result<()> {
        let f = self.need(
            self.kernels.expert_down_fp4_batched,
            "dsv41_expert_down_fp4_batched",
        )?;
        let rc = unsafe {
            f(
                act_base, act_stride, out, out_slot_stride, rows, dim, inter, row_weight,
                rw_stride, slots, w2_base, w2_stride, w2s_base, w2s_stride, ids, self.stream,
            )
        };
        self.kerr(rc, "dsv41_expert_down_fp4_batched")
    }

    /// Fixed-order sum of the batched down scratch into `out` (see the kernel
    /// comment: the ascending-slot order is the numerical contract).
    pub fn moe_down_reduce(&self, part: *const f32, out: *mut f32, n: i32, slots: i32) -> Result<()> {
        let f = self.need(self.kernels.moe_down_reduce, "dsv41_moe_down_reduce")?;
        let rc = unsafe { f(part, out, n, slots, self.stream) };
        self.kerr(rc, "dsv41_moe_down_reduce")
    }

    /// Fused down + reduce (DSV41_DOWN_FUSE, default ON): ONE launch computes
    /// every slot's fp4 down GEMV and sums the per-slot contributions in
    /// ASCENDING slot order straight into `out` (overwrite). Bit-identical to
    /// `expert_down_fp4_batched` + `moe_down_reduce`; `act_base` holds `slots`
    /// slices `act_stride` floats apart and only the swiglu half (the first
    /// `inter` floats) of each is read.
    #[allow(clippy::too_many_arguments)]
    pub fn expert_down_reduce_fp4_batched(
        &self,
        act_base: *const f32,
        act_stride: i64,
        out: *mut f32,
        rows: i32,
        dim: i32,
        inter: i32,
        row_weight: *const f32,
        rw_stride: i64,
        slots: i32,
        w2_base: *const u8,
        w2_stride: i64,
        w2s_base: *const u8,
        w2s_stride: i64,
        ids: *const i32,
    ) -> Result<()> {
        let f = self.need(
            self.kernels.expert_down_reduce_fp4_batched,
            "dsv41_expert_down_reduce_fp4_batched",
        )?;
        let rc = unsafe {
            f(
                act_base, act_stride, out, rows, dim, inter, row_weight, rw_stride, slots, w2_base,
                w2_stride, w2s_base, w2s_stride, ids, self.stream,
            )
        };
        self.kerr(rc, "dsv41_expert_down_reduce_fp4_batched")
    }

    /// w2 L2 PREWARM (DSV41_W2_PREWARM, default OFF, `=1` enables): warm every slot's w2 rows
    /// (`sel_bytes` = dim * (inter/2)) plus their e8m0 scale rows
    /// (`sc_bytes` = dim * (inter/32)) into L2 so the down GEMV that follows is
    /// an L2 hit instead of a ~600 ns HBM round trip. The kernel is
    /// fire-and-forget (writes nothing, returns 0), so this never fails a step
    /// and the caller can issue it unconditionally.
    ///
    /// MUST be issued on the same stream immediately AFTER the batched gate/up
    /// launch and BEFORE the down launch: the region it warms is exactly what
    /// the down launch reads (`w2_base + ids[slot]*w2_stride + row*(inter/2)`,
    /// rows 0..dim-1 contiguous), and the gate/up tail is the window where the
    /// memory system is otherwise idle. Issue it earlier and the gate/up pass's
    /// own w1/w3 stream evicts what it warmed.
    #[allow(clippy::too_many_arguments)]
    pub fn w2_l2_prewarm(
        &self,
        w2_base: *const u8,
        w2_stride: i64,
        w2s_base: *const u8,
        w2s_stride: i64,
        ids: *const i32,
        slots: i32,
        sel_bytes: i64,
        sc_bytes: i64,
    ) -> Result<()> {
        let Some(f) = self.kernels.w2_l2_prewarm else {
            return Ok(());   // stale .so: no prewarm, old timing
        };
        // Return value is always 0 and the entry point swallows its own launch
        // error (best effort: the down launch that follows is the correctness
        // path). Not routed through `kerr` on purpose - a hint must never fail
        // the step.
        let _ = unsafe {
            f(
                w2_base, w2_stride, w2s_base, w2s_stride, ids, slots, sel_bytes, sc_bytes,
                self.stream,
            )
        };
        Ok(())
    }

    /// Batched swiglu: grid.y = slot over `slots` consecutive [2*inter] blocks.
    pub fn swiglu_limit_batched(
        &self,
        gate_up: *mut f32,
        rows: i32,
        inter: i32,
        limit: f32,
        slot_stride: i64,
        slots: i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.swiglu_limit_batched, "dsv41_swiglu_limit_batched")?;
        let rc = unsafe { f(gate_up, rows, inter, limit, slot_stride, slots, self.stream) };
        self.kerr(rc, "dsv41_swiglu_limit_batched")
    }
    /// Publish `src` into every rank's staging slot for this rank, from the device.
    pub fn ar_store(
        &self,
        peer_slots: *const u64,
        world: i32,
        rank: i32,
        src: *const f32,
        n: i64,
        slot_f: i64,
    ) -> Result<()> {
        let f = self.need(self.kernels.ar_store, "dsv41_ar_store")?;
        let rc = unsafe { f(peer_slots, world, rank, src, n, slot_f, self.stream) };
        self.kerr(rc, "dsv41_ar_store")
    }

    /// Sum the `world` staging slots into `dst`, spinning on the local stamps
    /// until every peer has published `round` — the host never waits.
    pub fn ar_reduce(
        &self,
        dst: *mut f32,
        staging: *const f32,
        n: i64,
        slot_f: i64,
        world: i32,
        stamps: *const c_uint,
        round: u32,
    ) -> Result<()> {
        let f = self.need(self.kernels.ar_reduce, "dsv41_ar_reduce")?;
        let rc = unsafe { f(dst, staging, n, slot_f, world, stamps, round, self.stream) };
        self.kerr(rc, "dsv41_ar_reduce")
    }

    pub fn engram_apply(
        &self,
        x: *mut f32,
        kv: *const f32,
        q_weight: *const f32,
        k_weight: *const f32,
        token_mask: *const u8,
        rows: i32,
        hc: i32,
        dim: i32,
        eps: f32,
    ) -> Result<()> {
        let f = self.need(self.kernels.engram_apply, "dsv41_engram_apply")?;
        let rc = unsafe { f(x, kv, q_weight, k_weight, token_mask, rows, hc, dim, eps, self.stream) };
        self.kerr(rc, "dsv41_engram_apply")
    }

    pub fn rmsnorm(
        &self,
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        n: i32,
        dim: i32,
        eps: f32,
    ) -> Result<()> {
        self.rmsnorm_on(x, w, out, n, dim, eps, self.stream)
    }

    /// [`Self::rmsnorm`] issued on `s` instead of the main stream. Only the
    /// attention dual chain's kv fallback (`DSV41_DUAL_CHAIN` with `NR_FUSE`
    /// off, or an `.so` without `dsv41_rmsnorm_rope`) uses it; every other
    /// caller stays on the main stream.
    pub fn rmsnorm_on(
        &self,
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        n: i32,
        dim: i32,
        eps: f32,
        s: CuStream,
    ) -> Result<()> {
        let rc = unsafe { (self.kernels.rmsnorm)(x, w, out, n, dim, eps, s) };
        self.kerr(rc, "ferrite_rmsnorm")
    }

    /// T2: `rmsnorm` + the fp8 quantisation of its OWN normalised output in one
    /// launch. The caller then skips the `quant_fp8` it would have run on the
    /// same row. `Ok(false)` when the loaded .so predates the kernel, so the
    /// caller runs the two launches instead.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_q(
        &self,
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        n: i32,
        dim: i32,
        eps: f32,
        xq: *mut u8,
        xsc: *mut f32,
    ) -> Result<bool> {
        let f = match self.kernels.rmsnorm_q {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe { f(x, w, out, n, dim, eps, xq, xsc, self.stream) };
        // 1 == the launcher declined the shape (the warp-shuffle amax needs
        // whole warps); the caller then runs rmsnorm + quant_fp8 as before.
        if rc == 1 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_rmsnorm_q")?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_pre(
        &self,
        res: *const f32,
        fw: *const f32,
        scale: *const f32,
        base: *const f32,
        li: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        s: i32,
        n: i32,
        h: i32,
        mix: i32,
        rms_eps: f32,
        hc_eps: f32,
        iters: i32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.hc_pre)(
                res, fw, scale, base, li, post, comb, s, n, h, mix, rms_eps, hc_eps, iters,
                self.stream,
            )
        };
        self.kerr(rc, "ferrite_hc_pre")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_post(
        &self,
        x: *const f32,
        res: *const f32,
        post: *const f32,
        comb: *const f32,
        out: *mut f32,
        s: i32,
        n: i32,
        h: i32,
    ) -> Result<()> {
        let rc = unsafe { (self.kernels.hc_post)(x, res, post, comb, out, s, n, h, self.stream) };
        self.kerr(rc, "ferrite_hc_post")
    }

    /// Fused segment C: hc_post written straight back onto the residual stream.
    /// Bit-identical to `hc_post` + the h2 copy (the accumulation keeps the same
    /// ascending-k order), but without the staging buffer or the copy.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_post_inplace(
        &self,
        res: *mut f32,
        x: *const f32,
        post: *const f32,
        comb: *const f32,
        n: i32,
        h: i32,
    ) -> Result<()> {
        let rc = unsafe { (self.kernels.hc_post_inplace)(res, x, post, comb, n, h, self.stream) };
        self.kerr(rc, "dsv41_hc_post_inplace")
    }

    /// Fused segment B cluster 1: collapse the hyper-connection rows and
    /// normalise, writing `out` directly (the intermediate `x` is not needed).
    #[allow(clippy::too_many_arguments)]
    pub fn hc_collapse_norm(
        &self,
        x: *mut f32,
        pre: *const f32,
        w: *const f32,
        out: *mut f32,
        rows: i32,
        hc: i32,
        dim: i32,
        eps: f32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.hc_collapse_norm)(x, pre, w, out, rows, hc, dim, eps, self.stream)
        };
        self.kerr(rc, "dsv41_hc_collapse_norm")
    }

    /// Fused, spread hc front end: the mixes dot products on one block per
    /// projection row (cp.async staged) and the sum-of-squares / sigmoid /
    /// sinkhorn tail in one trailing kernel. Returns Ok(false) and does nothing
    /// when the gate is off, so the caller falls back to hc_mixes.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_front(
        &self,
        x: *const f32,
        hc_fn: *const f32,
        hc_scale: *const f32,
        hc_base: *const f32,
        w_norm: *const f32,
        pre_collapse: *const f32,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        out: *mut f32,
        rows: i32,
        hc: i32,
        dim: i32,
        sinkhorn_iters: i32,
        eps: f32,
        eps_norm: f32,
        xq: *mut u8,
        xsc: *mut f32,
    ) -> Result<bool> {
        let rc = unsafe {
            (self.kernels.hc_front)(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                w_norm,
                pre_collapse,
                pre,
                post,
                comb,
                out,
                rows,
                hc,
                dim,
                sinkhorn_iters,
                eps,
                eps_norm,
                xq,
                xsc,
                self.stream,
            )
        };
        // The kernel reports InvalidValue when the gate is off or the row count is
        // beyond the spread tables; both mean "use hc_mixes instead", not an error.
        if rc == 1 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_hc_front")?;
        Ok(true)
    }

    /// True when the loaded `.so` carries the tail-split entry
    /// (`dsv41_hc_front_split`) AND the runtime owns a side stream + the events
    /// the split actually records/waits. A stale `.so`, or a cudart without the
    /// event primitives, reports false and the caller keeps the single-launch
    /// `hc_front`.
    ///
    /// `in_event`/`early_event` are deliberately NOT gated on: since EARLY runs
    /// on main (2026-09-11) they are dead ABI slots, so requiring them would
    /// needlessly disable the split when only those two failed to create.
    pub fn supports_hc_tail_split(&self) -> bool {
        self.kernels.hc_front_split.is_some()
            && !self.rt.side_stream().is_null()
            && !self.rt.fork_event().is_null()
            && !self.rt.join_event().is_null()
    }

    /// True when the runtime owns a FOURTH side stream, so the tail split's
    /// dots+LATE half can be issued BESIDE EARLY instead of queued behind it
    /// (`DSV41_HC_DL_SIDE`, default ON). Purely additive: when false the
    /// launcher receives a null `side_dl` and keeps the pre-split order, which is
    /// bit-identical (same statements, same operands, same stream as before).
    ///
    /// No `.so` gate: the split's ABI grew one trailing stream parameter, and the
    /// build's version stamp (`verify_kernel_build`) already refuses a mismatched
    /// pair, so the symbol's presence is not evidence about its arity.
    pub fn supports_hc_dl_side(&self) -> bool {
        static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *F.get_or_init(|| {
            std::env::var("DSV41_HC_DL_SIDE")
                .map(|v| v != "0")
                .unwrap_or(true)
        }) && !self.rt.side_stream4().is_null()
    }

    /// hc tail split front end (`DSV41_HC_TAIL_SPLIT`): the EARLY tail half
    /// (collapse + rmsnorm + T1 fp8, writing `out`/`xq`/`xsc`), the dots, and the
    /// LATE half (ss + sigmoid + sinkhorn + comb, writing `pre`/`post`/`comb`) all
    /// run on the side stream; `main` waits only the EARLY half. Returns Ok(true)
    /// when it ran — the caller MUST then call [`Self::hc_tail_join`] before the
    /// hc_post that reads `comb`. Ok(false) means "fall back to hc_front /
    /// hc_mixes".
    ///
    /// Same ABI as `hc_front` plus (side stream, fork event, join event), which
    /// the C++ launcher records/waits internally. The split is bit-identical to
    /// `hc_front`: both halves execute the same statements with the same operands.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_front_split(
        &self,
        x: *const f32,
        hc_fn: *const f32,
        hc_scale: *const f32,
        hc_base: *const f32,
        w_norm: *const f32,
        pre_collapse: *const f32,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        out: *mut f32,
        rows: i32,
        hc: i32,
        dim: i32,
        sinkhorn_iters: i32,
        eps: f32,
        eps_norm: f32,
        xq: *mut u8,
        xsc: *mut f32,
    ) -> Result<bool> {
        let f = self.need(self.kernels.hc_front_split, "dsv41_hc_front_split")?;
        let rc = unsafe {
            f(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                w_norm,
                pre_collapse,
                pre,
                post,
                comb,
                out,
                rows,
                hc,
                dim,
                sinkhorn_iters,
                eps,
                eps_norm,
                xq,
                xsc,
                self.stream,
                self.rt.side_stream(),
                // Dead ABI slots (EARLY-on-main): the launcher neither validates
                // nor records/waits them. See `supports_hc_tail_split`.
                std::ptr::null_mut(),
                self.rt.fork_event(),
                std::ptr::null_mut(),
                self.rt.join_event(),
                // `DSV41_HC_DL_SIDE`: the dots+LATE half goes on its own stream
                // (null = it stays behind EARLY on `side`, the pre-split order).
                if self.supports_hc_dl_side() {
                    self.rt.side_stream4()
                } else {
                    std::ptr::null_mut()
                },
            )
        };
        // InvalidValue (1) = gate off / no collapse half / shape outside the
        // spread tables → the caller keeps the two-launch path.
        if rc == 1 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_hc_front_split")?;
        self.hc_split_armed.set(true);
        Ok(true)
    }

    /// Join the side stream issued by [`Self::hc_front_split`] back onto the main
    /// stream. Must run BEFORE the hc_post that consumes `comb`. A no-op when no
    /// split is pending, so both hc_post sites can call it unconditionally — a
    /// wait on an event that was never recorded is an illegal capture op, which
    /// is exactly what the `armed` flag prevents.
    pub fn hc_tail_join(&self) -> Result<()> {
        if self.hc_split_armed.replace(false) {
            self.rt.stream_wait_event(self.stream, self.rt.join_event())?;
        }
        Ok(())
    }

    /// True when the loaded `.so` carries the Stage-C persistent prototype
    /// (`dsv41_hc_front_persist`). A stale `.so` reports false and the caller
    /// keeps the two-launch `hc_front` path.
    pub fn supports_hc_persist(&self) -> bool {
        self.kernels.hc_front_persist.is_some()
    }

    /// Stage-C persistent prototype: the whole hc front end in ONE block as a
    /// `__syncthreads` phase machine (no ticket, no spin). Same arguments and
    /// same fallback contract as [`Self::hc_front`] — Ok(false) means "the
    /// kernel refused this shape, use hc_mixes instead".
    ///
    /// ⚠️ This preserves bit-exactness by construction (see the kernel comment)
    /// but is EXPECTED to be slower than the two-launch dots until the segment
    /// kernels land: the 24 projection rows now share ONE SM. Gated behind
    /// `DSV41_HC_PERSIST=1` (default OFF) for exactly that reason.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_front_persist(
        &self,
        x: *const f32,
        hc_fn: *const f32,
        hc_scale: *const f32,
        hc_base: *const f32,
        w_norm: *const f32,
        pre_collapse: *const f32,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        out: *mut f32,
        rows: i32,
        hc: i32,
        dim: i32,
        sinkhorn_iters: i32,
        eps: f32,
        eps_norm: f32,
        xq: *mut u8,
        xsc: *mut f32,
    ) -> Result<bool> {
        let f = self.need(self.kernels.hc_front_persist, "dsv41_hc_front_persist")?;
        let rc = unsafe {
            f(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                w_norm,
                pre_collapse,
                pre,
                post,
                comb,
                out,
                rows,
                hc,
                dim,
                sinkhorn_iters,
                eps,
                eps_norm,
                xq,
                xsc,
                self.stream,
            )
        };
        // Same contract as hc_front: InvalidValue means "use hc_mixes instead".
        if rc == 1 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_hc_front_persist")?;
        Ok(true)
    }

    /// True when the loaded `.so` carries the multi-block Stage-C prototype
    /// (`dsv41_hc_front_persist_mb`). A stale `.so` reports false and the caller
    /// keeps the single-block / two-launch path.
    pub fn supports_hc_persist_mb(&self) -> bool {
        self.kernels.hc_front_persist_mb.is_some()
    }

    /// Stage-C persistent MULTI-BLOCK form: the hc front end still in ONE launch,
    /// but the dots spread over `mix * split` blocks (one per projection row and
    /// K chunk) with the collapse on its own parallel block and the tail elected
    /// to the last-finishing dot block — no ticket, no spin. Same arguments and
    /// same fallback contract as [`Self::hc_front`].
    ///
    /// ⚠️ Bit-exact with the two-launch path at split = 1 only; split > 1 splits
    /// the dot reduction and is deterministic but not bit-identical, so it needs
    /// a tolerance gate. Gated behind `DSV41_HC_PERSIST_MB=1` (default OFF).
    #[allow(clippy::too_many_arguments)]
    pub fn hc_front_persist_mb(
        &self,
        x: *const f32,
        hc_fn: *const f32,
        hc_scale: *const f32,
        hc_base: *const f32,
        w_norm: *const f32,
        pre_collapse: *const f32,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        out: *mut f32,
        rows: i32,
        hc: i32,
        dim: i32,
        sinkhorn_iters: i32,
        eps: f32,
        eps_norm: f32,
        xq: *mut u8,
        xsc: *mut f32,
    ) -> Result<bool> {
        let f = self.need(self.kernels.hc_front_persist_mb, "dsv41_hc_front_persist_mb")?;
        let rc = unsafe {
            f(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                w_norm,
                pre_collapse,
                pre,
                post,
                comb,
                out,
                rows,
                hc,
                dim,
                sinkhorn_iters,
                eps,
                eps_norm,
                xq,
                xsc,
                self.stream,
            )
        };
        // Same contract as hc_front: InvalidValue means "use hc_mixes instead".
        if rc == 1 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_hc_front_persist_mb")?;
        Ok(true)
    }

    /// Embedding gather + hc expansion, straight onto the residual stream.
    pub fn embed_expand_dev(
        &self,
        table: *const c_void,
        ids_dev: *const c_int,
        out: *mut f32,
        n: i32,
        hidden: i32,
        mult: i32,
        vocab: i32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.embed_expand_dev)(table, ids_dev, out, n, hidden, mult, vocab, self.stream)
        };
        self.kerr(rc, "ferrite_embed_expand_dev")
    }

    /// bf16 -> f32 on the device (GLM's kernel). Used to widen weights without
    /// staging them through host memory.
    pub fn bf16_to_f32(&self, src: *const c_void, dst: *mut c_void, n: i64) -> Result<()> {
        let rc = unsafe { (self.kernels.bf16_to_f32)(src, dst, n, self.stream) };
        self.kerr(rc, "ferrite_bf16_to_f32")?;
        // load-time op: synchronise so a fault inside the kernel is attributed
        // here rather than to whatever call happens to run next
        self.dev_sync().map_err(|e| {
            FerriteError::Config(format!("ferrite_bf16_to_f32 (n={n}): {e}"))
        })
    }

    pub fn f32_to_bf16(&self, src: *const f32, dst: *mut c_void, n: i64) -> Result<()> {
        let rc = unsafe { (self.kernels.f32_to_bf16)(src, dst, n, self.stream) };
        self.kerr(rc, "ferrite_f32_to_bf16")
    }

    /// The gate_up+swiglu fusion needs the batched gate_up symbol with the
    /// fused signature (the trailing `fuse_swiglu` parameter). A stale .so
    /// without it makes `supports_gateup_fuse` false and the caller runs the
    /// unfused 2*inter + separate swiglu path.
    pub fn supports_gateup_fuse(&self) -> bool {
        // The signature check is the build-id gate: if the .so was built from
        // the same checkout as this binary, the fused parameter is there.
        // (The kernel entry point itself is unchanged; only the trailing
        // parameter was added, and old .so entries silently ignore it via
        // the default `0` from the launcher's `g_fuse` fallback.)
        self.kernels.expert_gate_up_fp4_batched.is_some()
    }
}

/// Row count below which the M=1 bf16 GEMV diverts to the vectorized + K-split
/// v2 kernel. Above this the shape already streams weights efficiently on v1,
/// and the lm_head (n = vocab, or its per-rank slice) must stay on v1's
/// bit-exact path. The small-n users are the MoE gate (384), the indexer's
/// wk (index_head_dim) and wp (index_n_heads) projections.
const GEMV_V2_MAX_N: i32 = 2048;

/// `DSV41_GEMV_V2=0` pins every M=1 bf16 GEMV to v1 (A/B / bisect escape
/// hatch). Default ON: the .so exposes `ferrite_gemv_bf16_v2`.
fn gemv_bf16_v2_wanted(n: i32) -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    n > 0
        && n < GEMV_V2_MAX_N
        && *F.get_or_init(|| std::env::var("DSV41_GEMV_V2").map(|v| v != "0").unwrap_or(true))
}

/// Row count below which the M=1 f32 GEMV diverts to the vectorized + K-split
/// v2 kernel. The only f32 GEMV user is the compressor's kvp/scp projection
/// (n = head_dim = 128) — the bound is generous so any future small f32 shape
/// benefits, while large ones keep v1's bit-exact order.
const GEMV_F32_V2_MAX_N: i32 = 2048;

/// `DSV41_GEMV_F32_V2=0` pins every M=1 f32 GEMV to v1 (A/B escape hatch).
/// Default ON: the .so exposes `dsv41_gemv_f32_v2`.
fn gemv_f32_v2_wanted(n: i32) -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    n > 0
        && n < GEMV_F32_V2_MAX_N
        && *F.get_or_init(|| std::env::var("DSV41_GEMV_F32_V2").map(|v| v != "0").unwrap_or(true))
}
