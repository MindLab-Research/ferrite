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
        c_int, c_int, c_int, c_int, *const c_int, c_int, c_int, f32,
        // W2-MROWS (DSV41_ATTN_MROWS, ABI 5): the m-row call's inputs. `clen_rows`
        // is the per-row compressor-length snapshot (`nullptr` = read the single
        // scalar counter, which is what every pre-W2 caller wants and what keeps
        // the per-row result byte-identical); `idx_stride` is the `idxs` row pitch
        // (0 = the historical `topk` pitch). See `dsv41_sparse_attn`.
        *const c_int, c_int, CuStream,
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
        *mut u8, *mut f32,
        // W2-MROWS (DSV41_ATTN_MROWS, ABI 5): `clen_rows`/`idx_stride` as in
        // `sparse_attn`; `row_step` makes the rope position affine in the row
        // (`tt = base*mul + off + hh*step + mm*row_step`, 0 = the old formula), so
        // an m-row launch ropes row `mm` at `pos_base + mm`.
        *const c_int, c_int, c_int, CuStream,
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
    // ROW-FOLD (DSV41_ROW_FOLD_ROPE): the m-verify-row form of `apply_rope`
    // above — one launch, the kernel loops r ascending, row r roped at
    // `pos_rows[r]`. Optional: an .so without the symbol keeps the per-row
    // launches (the fallback is bit-identical by construction).
    apply_rope_mrows: Option<unsafe extern "C" fn(
        *mut f32, *const f32, *const f32, c_int, c_int, c_int, c_int, c_int, c_int, *const c_int,
        c_int, CuStream,
    ) -> c_int>,
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
    /// The dsv41-side multi-row rmsnorm (`rows` norm rows of `dim`, row stride
    /// `dim`, the shared weight row `w[dim]`), the `DSV41_NORM_MROWS` arm's
    /// kernel. Bit-identical to `rmsnorm`'s per-row program at the same blockDim
    /// (see `dsv41_rmsnorm_rows_kernel`). Optional: an `.so` without the symbol
    /// (or with the gate off) keeps the pre-existing `rmsnorm` call.
    rmsnorm_rows: Option<unsafe extern "C" fn(
        *const f32, *const f32, *mut f32, c_int, c_int, f32, CuStream,
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
    // Multi-row twin of `argmax_sliced`, for the DSpark verify's vocabulary-sliced
    // head (DSV41_VERIFY_HEAD_SLICED): every row reduces its own slice and the
    // whole block exchanges in ONE v5 epoch round, so the verify's head pays the
    // same cross-rank cost the single-row eager head does instead of one round
    // per row. Optional: an .so without it keeps the verify's per-row
    // FULL-vocabulary head (see `verify_head_geom`).
    argmax_sliced_rows: Option<unsafe extern "C" fn(
        *const f32, c_int, c_int, c_int, *mut c_int, *mut u64, *mut c_int, *const *mut u64,
        *const *mut u32, *mut c_uint, *mut u64, *const c_uint, c_int, c_int, c_int, c_long,
        CuStream,
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
    // ROW-FOLD (`DSV41_ROW_FOLD_GATE`): the m-activation-row form of the v2 GEMV
    // above (`ferrite_gemv_bf16_nt`, `gemv_bf16_nt_kernel<NT, WPR>`) — same WPR
    // heuristic, same K-slice walk and smem fold, one accumulator per token.
    // Optional: an .so without it keeps the per-row `gemv_bf16` loop.
    gemv_bf16_nt: Option<
        unsafe extern "C" fn(
            *const f32, *const c_void, *const f32, *mut f32, c_int, c_int, c_int, CuStream,
        ) -> c_int,
    >,
    // The MoE gate's multi-row entry (`DSV41_GATE_MROWS`,
    // `ferrite_gemv_bf16_v2_mrows`): the SAME multi-row v2 program as
    // `gemv_bf16_nt` above (one definition in the .so — the fold runs the
    // program it claims parity with), under the gate's own name and its 1..=8
    // bound. Optional: an .so without it falls back to `gemv_bf16_nt`, then to
    // the per-row `gemv_bf16` loop. ABI is `gemv_bf16_nt`'s verbatim.
    gemv_bf16_v2_mrows: Option<
        unsafe extern "C" fn(
            *const f32, *const c_void, *const f32, *mut f32, c_int, c_int, c_int, CuStream,
        ) -> c_int,
    >,
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
    // Multi-row head GEMV, from dsv41_glue.cu (`dsv41_head_gemv_bf16_mrows`).
    // ONE pass over the [n, k] bf16 weight for all `m` activation rows instead
    // of one pass per row: the verify block's head. `x` is [m, k] f32, `out` is
    // [m, n] f32. Optional: a stale .so without the symbol keeps the per-row
    // loop, and m outside 1..=8 is refused (cudaErrorInvalidValue) for the same
    // reason. Row r is bit-identical to the single-row `gemv_bf16` launch of
    // row r (the kernel header carries the C1-C5 argument).
    // ABI: (w, x, out, m=rows, n, k, s).
    head_gemv_bf16_mrows: Option<
        unsafe extern "C" fn(*const c_void, *const f32, *mut f32, c_int, c_int, c_int, CuStream) -> c_int,
    >,
    // V1-ORDER multi-row GEMV, from dsv41_glue.cu (`dsv41_gemv_bf16_v1_mrows`):
    // the SAME v1 program `dsv41_gemv_bf16` runs, with `m` activation rows
    // folded into one weight pass. This is the fold the verify's SLICED head
    // actually needs — `head_gemv_bf16_mrows` above is the v2 (`gemv_bf16_nt`,
    // WPR == 1) program, a NUMERICAL change against the v1 head that measured as
    // one (`verify_head_fold`). Row r here is bit-identical to the per-row
    // `gemv_bf16` launch of row r, because the head's shape stays on v1
    // (`gemv_bf16_v2_wanted(n)` needs `n < 2048`; the head's `n` is the
    // vocabulary / its slice). `x` is [m, k] f32, `out` is [m, n] f32. Optional:
    // a stale .so without the symbol keeps the per-row loop, and m outside 1..=8
    // is refused. NOTE v1's body is scalar, so — unlike `head_gemv_bf16_mrows` —
    // there is no `k % 8 == 0` bound.
    // ABI: `head_gemv_bf16_mrows`'s verbatim — (w, x, out, m=rows, n, k, s).
    gemv_bf16_v1_mrows: Option<
        unsafe extern "C" fn(*const c_void, *const f32, *mut f32, c_int, c_int, c_int, CuStream) -> c_int,
    >,
    /// Tap bf16 round-trip (DSV41_TAP_BF16): in-place elementwise
    /// `x[i] = bf16_to_f32(f32_to_bf16(x[i]))` for the dspark draft's main_h
    /// buffer, aligning the MTP head's input with the official model's bf16
    /// hidden states. Optional so a stale `.so` simply declines.
    bf16_roundtrip: Option<unsafe extern "C" fn(*mut f32, i64, CuStream) -> c_int>,
    // Grouped low-rank output projection, from dsv41_kernels.cu
    // (`dsv41_wo_a_grouped_fp8`). The draft's block-diagonal `wo_a` in ONE
    // launch per (group tile x MTP block) instead of one m=1 gemv per
    // (group, row): the weight rows are staged once and every activation row
    // folds against them. Optional — a stale .so without the symbol keeps the
    // per-(group, row) loop, and the C entry returns 2 (declined, never
    // cudaErrorInvalidValue) for a shape/mode it cannot take.
    // Row of the multi-row launch is bit-identical to the m=1 launch of the
    // same (group, row); the kernel header carries the argument.
    // ABI: (a, a_scale, w, w_scale, bias, out, groups, rows, n, k, a_stride,
    //       out_stride, s).
    wo_a_grouped_fp8: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *const u8, *const u8, *const f32, *mut f32,
            c_int, c_int, c_int, c_int, c_int, c_int, CuStream,
        ) -> c_int,
    >,
    // Dense multi-row fp8 GEMV, from dsv41_kernels.cu (`dsv41_gemm_fp8_mrows`).
    // The weight-stationary form of `dsv41_gemm_fp8_mx`'s **m == 1** program: one
    // block owns `nwarps` output rows, stages those weight rows ONCE, and folds
    // all `m` activation rows against them. This is what lets the verify's
    // `attention_rows` project wq_a / wkv / wq_b / wo_b (and, via
    // `wo_a_grouped_fp8`, wo_a) with ONE launch per projection instead of one per
    // row, without touching the numerical domain: `dsv41_gemm_fp8_mx` at m > 1
    // runs `gemm_fp8_kernel`'s 16-row TILE, a DIFFERENT program, so passing
    // `m = m` to the existing symbol would silently swap the summation — this
    // kernel reproduces the m == 1 consume expression, k-walk and shuffle tree
    // per row (the kernel header carries the C1-C6 argument).
    //
    // `a` is [m, k] fp8 (row r at +r*k), `a_scale` [m, k/32] f32, `w`/`w_scale`
    // as in `gemm_fp8_mx`, `out` f32 with row r's element `row` at
    // `+r*out_stride + row` — `out_stride` is a SEPARATE parameter because the
    // wq_b call site writes `nlh*head_dim` of an `nh*head_dim` row.
    //
    // Optional: a stale .so without the symbol keeps the per-row loop, and the C
    // entry returns 2 (declined, never cudaErrorInvalidValue) for `m` outside
    // 1..=8, a `k` that is not a multiple of 32, `out_stride < n`, or a run
    // configured for the reordering `DSV41_GEMV_FP8_MODE` 0/1 arms.
    // ABI: (a, a_scale, w, w_scale, bias, out, m, n, k, out_stride, s).
    gemm_fp8_mrows: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *const u8, *const u8, *const f32, *mut f32,
            c_int, c_int, c_int, c_int, CuStream,
        ) -> c_int,
    >,
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
    // DSpark verify: the m-row block append + per-row CAUSAL window indices in
    // one launch (the multi-row twin of `ring_win_fuse`). Row r's window ends
    // at *pos_ctr + r, so the intra-block causal order falls out of the ring
    // geometry. `idxs` is `[m, window]`.
    verify_ring_win: Option<unsafe extern "C" fn(
        *mut f32,
        *const f32,
        *const c_int,
        c_int,
        c_int,
        c_int,
        *mut i32,
        CuStream,
    ) -> c_int>,
    // DSpark verify snapshot/rollback (P0): ONE launch per layer per direction
    // replaces the per-slot `cudaMemcpyAsync` loop. Same bytes, device-side slot
    // arithmetic, so the pair is CUDA-graph capturable. Optional: a stale .so
    // has no entries and the caller keeps the memcpy path.
    dspark_ring_save: Option<
        unsafe extern "C" fn(*mut f32, *const f32, c_int, c_int, c_int, c_int, CuStream) -> c_int,
    >,
    dspark_ring_restore: Option<
        unsafe extern "C" fn(
            *mut f32,
            *const f32,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            CuStream,
        ) -> c_int,
    >,
    #[allow(clippy::type_complexity)]
    dspark_comp_save: Option<
        unsafe extern "C" fn(
            *const f32,
            *const f32,
            *const f32,
            *const c_int,
            *const c_int,
            *mut f32,
            *mut f32,
            *mut c_int,
            *mut c_int,
            c_int,
            c_int,
            c_int,
            CuStream,
        ) -> c_int,
    >,
    #[allow(clippy::type_complexity)]
    dspark_comp_restore: Option<
        unsafe extern "C" fn(
            *mut f32,
            *mut f32,
            *mut f32,
            *mut c_int,
            *mut c_int,
            *const f32,
            *const f32,
            *const c_int,
            *const c_int,
            c_int,
            c_int,
            c_int,
            CuStream,
        ) -> c_int,
    >,
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
            *const c_int, c_int, c_int, CuStream,
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
            *const c_int, c_int, c_int, CuStream,
        ) -> c_int,
    >,
    /// Capability marker for the DIRECT e4m3 activation path
    /// (`DSV41_EXPERT_ACT_E4M3`). `act_e4m3` is a trailing argument of the
    /// launchers, and a stale .so would ignore it and decode e4m3 bytes as fp4
    /// nibbles — a silent wrong answer. OPTIONAL: only the direct-e4m3 build
    /// exports this, so a stale .so keeps the gate OFF (reported once).
    expert_act_e4m3_cap: Option<unsafe extern "C" fn() -> c_int>,
    /// tcgen05 MXFP4 gate/up, Phase-1 skeleton (`DSV41_EXPERT_TCGEN05_MXF4`,
    /// default OFF). OPTIONAL on purpose: `build.sh` defines no
    /// `DSV41_TCGEN05_GATEUP_MXF4_SKELETON`, so a stock `.so` has no such
    /// symbol and the probe below keeps the proven GEMV path. The ABI (18
    /// params) takes the loader's four weight planes as base/stride pairs plus
    /// a device-side `ids[slot]`, so `moe()` can dispatch to it directly; see
    /// the ABI note in kernels.rs.
    expert_tcgen05_gate_up_mxf4: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *mut f32, i64, c_int, c_int, f32, c_int, *const u8, i64,
            *const u8, i64, *const u8, i64, *const u8, i64, *const c_int, CuStream,
        ) -> c_int,
    >,
    /// tcgen05 **e4m3-activation** gate/up (`DSV41_EXPERT_TCGEN05_E4M3`, default
    /// OFF). The `tc5::e4` sibling of the arm above: same swapAB mapping and the
    /// same 18-parameter ABI, but the activation is the OFFICIAL e4m3 form
    /// (`act_quant(e4m3, block=32)`, ONE byte per value) and the MMA is
    /// `kind::mxf8f6f4 ... scale_vec::1X`. Its own symbol on purpose — the two
    /// arms differ ONLY in the activation's byte layout, which no argument can
    /// express, and a stale `.so` decoding e4m3 bytes as fp4 nibbles would be a
    /// silent wrong answer rather than a failure. OPTIONAL on the same terms as
    /// `expert_tcgen05_gate_up_mxf4` (`build.sh` compiles it in by default, opt
    /// out with `DSV41_BUILD_TCGEN05_E4M3=0`).
    expert_tcgen05_gate_up_e4m3: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *mut f32, i64, c_int, c_int, f32, c_int, *const u8, i64,
            *const u8, i64, *const u8, i64, *const u8, i64, *const c_int, CuStream,
        ) -> c_int,
    >,
    /// tcgen05 **e4m3-activation dense-tile** gate/up/down (`tc5::e4x`,
    /// `DSV41_EXPERT_TCGEN05_E4M3` — the SAME runtime gate as the swapAB e4m3
    /// arm above, because the two share one activation FORMAT and differ only
    /// in launch shape). OPTIONAL on exactly the same terms (`build.sh`
    /// compiles it in with the same skeleton flag, opt out with
    /// `DSV41_BUILD_TCGEN05_E4M3=0`).
    ///
    /// The `kind::f8f6f4` MMA cannot carry a block-scale operand (`ptxas`
    /// rejects `.kind::f8f6f4` + `.block_scale`), so this arm runs the MMA
    /// RAW and applies the per-32-block scales in the accumulation step
    /// (`C_local_accum += C_local * sa * sb`) — the official tilelang inner
    /// loop's "scale 外提" form. It is the e4m3 twin of `mxf4_gemm_kernel`:
    /// **M = the activation rows** masked to 128 (grid.y tiles them), N = the
    /// output columns, instead of the swapAB arm's M = 2*inter / N = 8.
    ///
    /// ABI (25 params). Shapes are DENSE: `a` is `[rows, k]` e4m3 (**one byte
    /// per value**), `a_scale` is `[rows, k/32]` **f32** (the same per-row
    /// layout `dsv41_quant_fp8` writes — the kernel converts to e8m0), `b` is
    /// the PACKED fp4 weight `[b_rows, k/2]` with its e8m0 `[b_rows, k/32]`
    /// scales, and `out` is `[rows, n_total]` (row pitch `n_total`). `b_hi`/
    /// `b_hi_scale` is the SECOND pool with `b_split` = the FIRST pool's row
    /// count (`n < b_split` reads `b`, the rest `b_hi` at `row = n - b_split`)
    /// — the launcher's way of expressing gate|up. `epi_mode` 1 clamps with
    /// `limit` (gate/up), 2 multiplies by `row_weight[row]` (down), 3 adds
    /// into `out`. `ids != nullptr` derives every weight plane from
    /// `base + ids[slot] * stride`: **ONE expert per launch**.
    ///
    /// ⚠️ The launcher refuses (nonzero rc) unless `k % 64 == 0`,
    /// `rows % 128 == 0`, `n_total % 64 == 0` and `a`/`b`/`b_hi` are 16-byte
    /// aligned — a caller must pre-check the contract.
    expert_gemm_e4m3_ext: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *const u8, *const u8, *const u8, *const u8, *mut f32, c_int,
            c_int, c_int, c_int, c_int, f32, *const f32, *const u8, i64, *const u8, i64,
            *const u8, i64, *const u8, i64, *const c_int, c_int, CuStream,
        ) -> c_int,
    >,
    /// GROUP-INDEXED MASKED sibling of `expert_gemm_e4m3_ext` — DeepGEMM's
    /// `m_grouped_gemm_nt_masked`. Same MMA/format/K-fold as the dense-tile arm,
    /// but the expert of a CTA comes from the GROUPED routing layout
    /// (`active[blockIdx.z]` + `counts`/`starts`), the A rows are the grouped
    /// activation rows (`starts[e] + m_off`), and the rows past `counts[e]` are
    /// MASKED (zero-fill in, no store out). That removes the dense arm's
    /// `rows % 128 == 0` requirement, which is what makes the tcgen05 e4m3 arm
    /// reachable at decode shapes (`m <= 6`, `topk = 6`, O(1) rows per expert).
    ///
    /// This ONE entry point needs BOTH runtime gates (`DSV41_EXPERT_TCGEN05_E4M3`
    /// and `DSV41_EXPERT_GROUPED`, read once inside the `.so`); rc == 0 means
    /// "nothing ran", the caller's fallback contract. A nonzero rc is a REJECTED
    /// contract (see [`Self::expert_gemm_e4m3_grouped`]), not a missing feature.
    expert_gemm_e4m3_grouped: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *mut f32, *const c_int, *const c_int, *const c_int,
            *const c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, f32, *const u8, i64,
            *const u8, i64, *const u8, i64, *const u8, i64, CuStream,
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
    /// COMPRESSOR-MROWS (`DSV41_COMPRESSOR_MROWS=1`, default OFF): the verify
    /// block's `seqlen = m` rows of the decode compressor in ONE launch, rows
    /// ascending. Optional, so an .so without the symbol (or the gate off) keeps
    /// the per-row path verbatim.
    compressor_fused_mrows: Option<
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
            *mut c_int,
            *mut f32,
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
    /// GROUPED (permuted) routing, `DSV41_EXPERT_GROUPED`. The expert-centric
    /// routed GEMMs need one expert's rows contiguous; `route_idx_r[m][topk]` is
    /// per-(row, slot). `dsv41_route_group` builds the layout (counts / starts /
    /// per-expert row+slot lists / both directions of the permutation) and the
    /// two movers apply it to the activation and to the expert output. All three
    /// are OPTIONAL: a stale `.so` has no entry and the caller keeps the
    /// per-(row, slot) path, exactly like the other staged-bring-up arms.
    route_group: Option<
        unsafe extern "C" fn(*const c_int, *mut c_int, *mut c_int, *mut c_int, *mut c_int, *mut c_int, *mut c_int, *mut c_int, *mut c_int, c_int, c_int, c_int, c_int, CuStream) -> c_int,
    >,
    route_group_gather: Option<
        unsafe extern "C" fn(*const u8, *const f32, *mut u8, *mut f32, *const c_int, c_int, c_int, c_int, c_int, CuStream) -> c_int,
    >,
    route_group_scatter: Option<
        unsafe extern "C" fn(*const f32, *mut f32, *const c_int, c_int, c_int, CuStream) -> c_int,
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
    /// DSpark draft head: one sequential step of the block sampler (`glue.cu`).
    dspark_markov_head: Option<
        unsafe extern "C" fn(
            *mut f32, *const f32, *const f32, *const f32, *const f32, *mut i32, *mut f32,
            c_int, c_int, c_int, c_int, *mut u64, *mut c_uint, CuStream,
        ) -> c_int,
    >,
    /// Vocabulary-sliced twin of `dspark_markov_head` (DSV41_MARKOV_SLICED): one
    /// rank walks its own `n`-wide row slice of `markov_head` (the caller passes
    /// the slice base into the REPLICATED tensor) and publishes the slice's
    /// packed winner into `local_key[0]` instead of writing `ids[step + 1]`. The
    /// caller then folds the ranks with `argmax_key_pub`. A SEPARATE symbol on
    /// purpose: the old entry keeps its arity, so a stale .so cannot be called
    /// through a mismatched signature. Optional — `None` keeps the
    /// full-vocabulary Markov loop (see `DsparkDev::markov_head_geom`).
    dspark_markov_head_sliced: Option<
        unsafe extern "C" fn(
            *mut f32, *const f32, *const f32, *const f32, *const f32, *mut i32, *mut f32,
            c_int, c_int, c_int, c_int, c_int, *mut u64, *mut u64, *mut c_uint, CuStream,
        ) -> c_int,
    >,
    /// Cross-rank fold of an ALREADY-PACKED argmax key: one v5 epoch round,
    /// nothing else (the local reduce belongs to the producer's kernel). The
    /// draft's sliced Markov loop needs it because its five steps are strictly
    /// sequential and cannot be batched into one multi-row round. Optional — see
    /// `Self::supports_argmax_key_pub`.
    argmax_key_pub: Option<
        unsafe extern "C" fn(
            *const u64, *mut c_int, *const *mut u64, *const *mut u32, *mut c_uint, *mut u64,
            *const c_uint, c_int, c_int, c_long, CuStream,
        ) -> c_int,
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
    /// The MULTI-ROW form of `hc_post_inplace`: the same fold, but every row
    /// reads its own `post`/`comb` slice (`+ r*hc`, `+ r*hc*hc`) and writes its
    /// own `hc*dim` residual block. That is the shape `layer_rows()` (verify)
    /// needs; the single-row entry above indexes row 0 of all three and so
    /// cannot serve a block. Kept as a SEPARATE symbol so the single-row decode
    /// path's codegen stays byte-for-byte untouched and a stale `.so` simply
    /// resolves it to None — the caller then keeps the `hc_post` + copy pair.
    /// Optional; bit-identical to that pair for any `rows`.
    hc_post_inplace_rows: Option<
        unsafe extern "C" fn(
            *mut f32, *const f32, *const f32, *const f32,
            c_int, c_int, c_int, CuStream,
        ) -> c_int,
    >,
    /// Segment B cluster 1: hc_collapse + rmsnorm(ffn_norm) in one kernel.
    /// `truncate` is trailing (before the stream): round the collapsed row to
    /// bf16, `DSV41_BF16_TRUNCATE`, default OFF.
    hc_collapse_norm: unsafe extern "C" fn(
        *mut f32, *const f32, *const f32, *mut f32,
        c_int, c_int, c_int, f32, c_int, CuStream,
    ) -> c_int,
    /// hc_mixes spread over one block per projection row with cp.async staging,
    /// plus the sum-of-squares/sigmoid/sinkhorn tail in one trailing kernel.
    /// Returns an error when DSV41_HC_FRONT is off, so the caller keeps hc_mixes.
    hc_front: unsafe extern "C" fn(
        *const f32, *const f32, *const f32, *const f32,
        *const f32, *const f32,
        *mut f32, *mut f32, *mut f32, *mut f32,
        c_int, c_int, c_int, c_int, f32, f32, *mut u8, *mut f32, c_int, CuStream,
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
            c_int, c_int, c_int, c_int, f32, f32, *mut u8, *mut f32, c_int, CuStream,
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
            c_int, c_int, c_int, c_int, f32, f32, *mut u8, *mut f32, c_int, CuStream,
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
            c_int, c_int, c_int, c_int, f32, f32, *mut u8, *mut f32, c_int,
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
    /// The MULTI-ROW form of `p2p_ar_v5_hcpost` (`ferrite_p2p_ar_v5_hcpost_rows`):
    /// the AR payload is `hc_rows * hc_h` floats with row stride `hc_h`, and
    /// `hc_res` / `hc_post` / `hc_comb` are the per-row slices the m-row
    /// `hc_mixes` writes (`res + r*hc_n*hc_h`, `post + r*hc_n`,
    /// `comb + r*hc_n*hc_n`). One launch replaces the verify's
    /// `all_reduce_inplace` + `hc_post_inplace_rows` pair (2 launches -> 1 at
    /// each of the 2 AR sites per layer). A stale `.so` reports `Ok(false)` and
    /// the caller keeps the pair.
    p2p_ar_v5_hcpost_rows: Option<
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
            c_int,
            CuStream,
        ) -> c_int,
    >,
}


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
            apply_rope_mrows: ko!(rt, "dsv41_apply_rope_mrows"),
            rmsnorm_rope: ko!(rt, "dsv41_rmsnorm_rope"),
            rmsnorm_q: ko!(rt, "dsv41_rmsnorm_q"),
            rmsnorm_rows: ko!(rt, "dsv41_rmsnorm_rows"),
            gemm_bf16_fp8x2: ko!(rt, "dsv41_gemm_bf16_fp8x2"),
            argmax_sliced: ko!(rt, "dsv41_argmax_sliced"),
            argmax_sliced_rows: ko!(rt, "dsv41_argmax_sliced_rows"),
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
            gemv_bf16_nt: ko!(rt, "ferrite_gemv_bf16_nt"),
            gemv_bf16_v2_mrows: ko!(rt, "ferrite_gemv_bf16_v2_mrows"),
            gemv_bf16_v2_route: ko!(rt, "ferrite_gemv_bf16_v2_route"),
            gemv_f32: ko!(rt, "dsv41_gemv_f32"),
            gemv_f32_v2: ko!(rt, "dsv41_gemv_f32_v2"),
            head_gemv_bf16_mrows: ko!(rt, "dsv41_head_gemv_bf16_mrows"),
            gemv_bf16_v1_mrows: ko!(rt, "dsv41_gemv_bf16_v1_mrows"),
            bf16_roundtrip: ko!(rt, "dsv41_bf16_roundtrip"),
            wo_a_grouped_fp8: ko!(rt, "dsv41_wo_a_grouped_fp8"),
            gemm_fp8_mrows: ko!(rt, "dsv41_gemm_fp8_mrows"),
            argmax: ko!(rt, "dsv41_argmax"),
            engram_hash_step: ko!(rt, "dsv41_engram_hash_step"),
            window_idxs: ko!(rt, "dsv41_window_idxs"),
            comp_placeholder: ko!(rt, "dsv41_comp_placeholder"),
            compress_commit: ko!(rt, "dsv41_compress_commit"),
            ring_append: ko!(rt, "dsv41_ring_append"),
            apply_rope_q: ko!(rt, "dsv41_apply_rope_q"),
            ring_win_fuse: ko!(rt, "dsv41_ring_win_fuse"),
            ring_win_fuse_ph: ko!(rt, "dsv41_ring_win_fuse_ph"),
            verify_ring_win: ko!(rt, "dsv41_verify_ring_win"),
            dspark_ring_save: ko!(rt, "dsv41_dspark_ring_save"),
            dspark_ring_restore: ko!(rt, "dsv41_dspark_ring_restore"),
            dspark_comp_save: ko!(rt, "dsv41_dspark_comp_save"),
            dspark_comp_restore: ko!(rt, "dsv41_dspark_comp_restore"),
            index_k_publish: ko!(rt, "dsv41_index_k_publish"),
            expert_gate_up_fp4_indirect: ko!(rt, "dsv41_expert_gate_up_fp4_indirect"),
            expert_down_fp4_indirect: ko!(rt, "dsv41_expert_down_fp4_indirect"),
            expert_gate_up_fp4_batched: ko!(rt, "dsv41_expert_gate_up_fp4_batched"),
            expert_act_e4m3_cap: ko!(rt, "dsv41_expert_act_e4m3_cap"),
            expert_tcgen05_gate_up_mxf4: ko!(rt, "dsv41_expert_tcgen05_gate_up_mxf4"),
            expert_tcgen05_gate_up_e4m3: ko!(rt, "dsv41_expert_tcgen05_gate_up_e4m3"),
            expert_gemm_e4m3_ext: ko!(rt, "dsv41_expert_gemm_e4m3_ext"),
            expert_gemm_e4m3_grouped: ko!(rt, "dsv41_expert_gemm_e4m3_grouped"),
            interleave_gateup_fp4: ko!(rt, "dsv41_interleave_gateup_fp4"),
            expert_down_fp4_batched: ko!(rt, "dsv41_expert_down_fp4_batched"),
            moe_down_reduce: ko!(rt, "dsv41_moe_down_reduce"),
            expert_down_reduce_fp4_batched: ko!(rt, "dsv41_expert_down_reduce_fp4_batched"),
            w2_l2_prewarm: ko!(rt, "dsv41_w2_l2_prewarm"),
            swiglu_limit_batched: ko!(rt, "dsv41_swiglu_limit_batched"),
            ar_reduce: ko!(rt, "dsv41_ar_reduce"),
            route_topk: ko!(rt, "dsv41_route_topk"),
            route_group: ko!(rt, "dsv41_route_group"),
            route_group_gather: ko!(rt, "dsv41_route_gather_rows"),
            route_group_scatter: ko!(rt, "dsv41_route_scatter_rows"),
            compressor_pool: ko!(rt, "dsv41_compressor_pool"),
            compressor_fused: ko!(rt, "dsv41_compressor_fused"),
            compressor_fused_mrows: ko!(rt, "dsv41_compressor_fused_mrows"),
            engram_apply: ko!(rt, "dsv41_engram_apply"),
            swiglu_limit: ko!(rt, "dsv41_swiglu_limit"),
            swiglu_limit_q: ko!(rt, "dsv41_swiglu_limit_q"),
            gather_rows: ko!(rt, "dsv41_gather_rows"),
            scatter_add_rows: ko!(rt, "dsv41_scatter_add_rows"),
            dspark_markov_head: ko!(rt, "dsv41_dspark_markov_head"),
            dspark_markov_head_sliced: ko!(rt, "dsv41_dspark_markov_head_sliced"),
            argmax_key_pub: ko!(rt, "dsv41_argmax_key_pub"),
            window_append: ko!(rt, "dsv41_window_append"),
            rmsnorm: km!(rt, "ferrite_rmsnorm"),
            hc_pre: km!(rt, "ferrite_hc_pre"),
            hc_post: km!(rt, "ferrite_hc_post"),
            hc_post_inplace: km!(rt, "dsv41_hc_post_inplace"),
            hc_post_inplace_rows: ko!(rt, "dsv41_hc_post_inplace_rows"),
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
            p2p_ar_v5_hcpost_rows: ko!(rt, "ferrite_p2p_ar_v5_hcpost_rows"),
        };
        Ok(Device { rt, kernels, stream, hc_split_armed: std::cell::Cell::new(false) })
    }

    // ------------------------------------------- shared device primitives
    // Every method below forwards to the shared `ferrite_kernel::devrt`
    // runtime; the model layer owns no cudart/cublas binding of its own.

    pub fn stream(&self) -> CuStream {
        self.stream
    }

    /// Tap bf16 round-trip (DSV41_TAP_BF16): in-place elementwise
    /// `x[i] = bf16_to_f32(f32_to_bf16(x[i]))`. Returns Ok(false) when the
    /// symbol is absent (stale .so) so the caller can treat it as a no-op.
    pub fn bf16_roundtrip(&self, x: *mut f32, n: i64) -> Result<bool> {
        let Some(f) = self.kernels.bf16_roundtrip else {
            return Ok(false);
        };
        let rc = unsafe { f(x, n, self.stream) };
        self.kerr(rc, "dsv41_bf16_roundtrip")?;
        Ok(true)
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

    /// True when the loaded .so carries COMPRESSOR-MROWS
    /// (`dsv41_compressor_fused_mrows`). Optional: a stale .so reports false and
    /// the caller keeps the per-row pool+commit pair, which is the bit-identical
    /// fallback.
    pub fn supports_compressor_fused_mrows(&self) -> bool {
        self.kernels.compressor_fused_mrows.is_some()
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

    /// True while this device's stream is inside a stream capture. The model
    /// layer's view of the one state the v5 epoch protocol may run in — see
    /// [`Self::argmax_sliced_rows`] for the rule and why it exists.
    pub fn capturing(&self) -> bool {
        self.rt.capturing()
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

    /// True when the loaded .so carries the P0 DSpark snapshot/rollback kernel
    /// set. `dspark_snapshot`/`dspark_rollback_keep` fall back to their per-slot
    /// `cudaMemcpyAsync` loop when a stale .so lacks these, so the pair is a
    /// pure emission change (identical bytes either way).
    pub fn supports_dspark_snapshot(&self) -> bool {
        self.kernels.dspark_ring_save.is_some()
            && self.kernels.dspark_ring_restore.is_some()
            && self.kernels.dspark_comp_save.is_some()
            && self.kernels.dspark_comp_restore.is_some()
    }

    /// True when `zero_at`/`zero_at_on` will NOT fall back to the synchronous
    /// `cudaMemset` on the legacy stream. The verify graph's `compress_proj_rows`
    /// zeroes `scp_r` on the `ratio == 1` no-gate path, so a capture needs this.
    pub fn supports_memset_async(&self) -> bool {
        self.rt.has_async_memset()
    }

    /// True when the loaded .so carries the DEVICE-derived window-ring append
    /// (`dsv41_ring_append`, `dsv41_glue.cu`). The draft's graph gate needs it:
    /// `seed_window`'s destination used to be the HOST-computed
    /// `window + (pos % win)*hd`, and a captured `cudaMemcpyAsync` bakes that
    /// address, so every replay would have appended to the SAME slot. The kernel
    /// derives the slot from a device counter instead, which is what makes the
    /// append replay-safe. A stale .so therefore keeps the draft on its direct
    /// launches (gate refuses).
    pub fn supports_ring_append(&self) -> bool {
        self.kernels.ring_append.is_some()
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

    /// True when the loaded `.so` carries the MULTI-ROW hc-post fold
    /// (`ferrite_p2p_ar_v5_hcpost_rows`) — the verify path's AR fold
    /// (`chain_dev::ChainDev::ar_hc_post_fold_rows`). A stale `.so` reports false
    /// and the caller keeps the `all_reduce_inplace` + `hc_post_inplace_rows`
    /// pair, so the switch is free to make.
    pub fn supports_ar_hcpost_rows(&self) -> bool {
        self.kernels.p2p_ar_v5_hcpost_rows.is_some()
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
        // W2-MROWS (DSV41_ATTN_MROWS): `b*m > 1` needs BOTH the per-row counter
        // snapshot and the `idxs` row pitch. `(null, 0)` is the per-row/shape-P0
        // call: every row reads the single scalar counter and the `topk` pitch,
        // byte-for-byte identical to the pre-W2 launcher.
        clen_rows: *const c_int,
        idx_stride: i32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.sparse_attn)(
                q, kv, sink, idxs, out, b, m, h, d, clen, window, index_topk, scale, clen_rows,
                idx_stride, self.stream,
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
        // W2-MROWS (DSV41_ATTN_MROWS): `clen_rows`/`idx_stride` as in
        // [`Self::sparse_attn`]; `row_step` adds `mm * row_step` to the rope
        // position (`0` = the per-row call's `off = r`, `step = 0` spelling).
        clen_rows: *const c_int,
        idx_stride: i32,
        row_step: i32,
    ) -> Result<bool> {
        let f = match self.kernels.sparse_attn_orope {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe {
            f(q, kv, sink, idxs, out, b, m, h, d, clen, window, index_topk, scale, cos, sin, base,
              rope_rd, half, mul, off, step, inverse as i32, xq, xsc, clen_rows, idx_stride,
              row_step, self.stream)
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

    /// ROW-FOLD (`DSV41_ROW_FOLD_ROPE`): [`Self::apply_rope`] for all `m` verify
    /// rows of `x` in ONE launch (`x` = the row-0 base, rows live `row_stride`
    /// apart, each `rows` head rows of `row_len`), row r roped at `pos_rows[r]`.
    ///
    /// Bit-identical to the `m` per-row launches it replaces: the kernel body is
    /// `apply_rope_kernel`'s, the per-row position is the same integer (the
    /// device array `attention_rows` already uploads — `pos_rows[r] == pos_base +
    /// r`, exactly what the `off = r, step = 0` form computed off the counter),
    /// and the rows touch disjoint memory with no cross-row reduction. See the
    /// kernel header in `dsv41_kernels.cu` for the instruction-level argument.
    ///
    /// `Ok(false)` = NOT performed, keep the per-row loop: the loaded .so
    /// predates the symbol, or the shape is degenerate.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_rope_mrows(
        &self,
        x: *mut f32,
        cos: *const f32,
        sin: *const f32,
        m: i32,
        rows: i32,
        row_stride: i32,
        row_len: i32,
        dim: i32,
        half: i32,
        pos_rows: *const c_int,
        inverse: bool,
    ) -> Result<bool> {
        let Some(f) = self.kernels.apply_rope_mrows else {
            return Ok(false);
        };
        if m <= 0 || rows <= 0 || row_len <= 0 || half <= 0 {
            return Ok(false);
        }
        let rc = unsafe {
            f(
                x,
                cos,
                sin,
                m,
                rows,
                row_stride,
                row_len,
                dim,
                half,
                pos_rows,
                inverse as i32,
                self.stream,
            )
        };
        self.kerr(rc, "dsv41_apply_rope_mrows")?;
        Ok(true)
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

    /// Multi-row twin of [`Self::argmax_sliced`], for the DSpark verify's
    /// vocabulary-sliced head: one local reduce per row over that row's own
    /// `n`-wide slice, then ONE v5 epoch round for the whole block.
    ///
    /// `row_stride` is the logits buffer's row pitch in F32 ELEMENTS (`seg` for
    /// the verify's sliced rows — the head writes them `seg` apart), `idx_off`
    /// the slice's base in the GLOBAL vocabulary, so `out[i]` is a global index
    /// either way. `pos_ctr` may be NULL: the verify must not advance it, and
    /// only the epoch advance is unconditional.
    ///
    /// `Ok(false)` when the loaded .so predates the symbol, or the kernel
    /// DECLINED (the block's keys do not fit the v5 slot). The caller gates the
    /// layout on [`Self::supports_argmax_sliced_rows`] and on the slot bound, so
    /// a decline here is an inconsistency it must not swallow — see
    /// `verify_head_geom`.
    ///
    /// # The epoch advance is UNCONDITIONAL
    ///
    /// This entry is ONE round of the shared v5 epoch sequence: the kernel lands
    /// the block's keys in every peer's CURRENT parity slot, stamps, advances
    /// `*epoch` (`*epoch = e + 1`), then polls every peer for the same round.
    /// `epoch` is the SAME device counter the MoE all-reduce advances
    /// (`c.epoch_dev()`), so the two MUST advance together per step. The AR keeps
    /// no capturing guard on this code path either: DSV41's MoE all-reduce goes
    /// through `all_reduce_inplace` → `p2p_ar_v5` (`tp.rs`), which advances one
    /// round wherever it runs, capture or not. (`p2p_ar_v2`'s
    /// `if !is_capturing() { return Ok(false) }` is a different arm, one DSV41
    /// does not take.) A guard here would drop this side's round while the AR
    /// kept its own — one round of permanent drift per step, after which every
    /// poll waits for a stamp the peers publish at a different round (the ar5
    /// disaster, gap 22). Hence: no guard, and `Ok(false)` only for the two arms
    /// above.
    #[allow(clippy::too_many_arguments)]
    pub fn argmax_sliced_rows(
        &self,
        v: *const f32,
        n: i32,
        idx_off: i32,
        row_stride: i32,
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
        rows: i32,
        stride_bytes: i64,
    ) -> Result<bool> {
        let f = match self.kernels.argmax_sliced_rows {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe {
            f(v, n, idx_off, row_stride, out, packed, pos_ctr, peer_staging, ready_tbl, epoch,
              staging_local, ready_local, world, rank, rows, stride_bytes, self.stream)
        };
        if rc == 1 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_argmax_sliced_rows")?;
        Ok(true)
    }

    /// Is the multi-row cross-rank argmax in this .so? The verify's head slicing
    /// is gated on it BEFORE any layout decision is taken, so a stale .so keeps
    /// `logits_r` at the full-vocabulary row pitch instead of leaving a
    /// half-sliced buffer for the fallback to misread.
    pub fn supports_argmax_sliced_rows(&self) -> bool {
        self.kernels.argmax_sliced_rows.is_some()
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

    // ---- GROUPED (permuted) routing (DSV41_EXPERT_GROUPED) ------------------
    //
    // The routed expert GEMMs are expert-CENTRIC: one launch covers ONE expert
    // over a DENSE `[rows, k]` activation block. `route_idx_r[m][topk]` is
    // per-(row, slot), so the grouped form has to be built before such a launch
    // can be issued at all. `route_group` builds it; the two movers apply it.
    // See `kernels/cuda/dsv41_route.cu` for the layout contract and
    // docs/agent/grouped-routing-design.md for the design.

    /// Build the grouped routing layout from `route_idx_r` (`ids[m][topk]`).
    /// Outputs (all may be null when the caller wants a subset):
    /// `counts[n_experts]`, `starts[n_experts + 1]` (exclusive prefix sum),
    /// `expert_rows`/`expert_slots` (per expert, `m_cap` stride),
    /// `perm_map[n_assign]` (original -> grouped), `gather_src[n_assign]`
    /// (grouped -> original), `active[..]` + `n_active` (compact non-empty
    /// experts). Returns `Ok(false)` when the loaded `.so` has no entry point.
    #[allow(clippy::too_many_arguments)]
    pub fn route_group(
        &self,
        ids: *const i32,
        counts: *mut i32,
        starts: *mut i32,
        expert_rows: *mut i32,
        expert_slots: *mut i32,
        perm_map: *mut i32,
        gather_src: *mut i32,
        active: *mut i32,
        n_active: *mut i32,
        m: i32,
        topk: i32,
        n_experts: i32,
        m_cap: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.route_group else {
            return Ok(false);
        };
        let rc = unsafe {
            f(
                ids, counts, starts, expert_rows, expert_slots, perm_map, gather_src, active,
                n_active, m, topk, n_experts, m_cap, self.stream,
            )
        };
        self.kerr(rc, "dsv41_route_group")?;
        Ok(true)
    }

    /// Gather the activation rows into the grouped order. `src_q`/`src_sc` are
    /// the dense per-row activation (`[m][row_bytes]`, `[m][sc_row_bytes]`),
    /// `dst_q`/`dst_sc` the `[n_assign][...]` grouped twin. A grouped position
    /// whose `gather_src` is `-1` is written as zeros. Either buffer pair may be
    /// null. Returns `Ok(false)` when the `.so` has no entry point.
    #[allow(clippy::too_many_arguments)]
    pub fn route_gather_rows(
        &self,
        src_q: *const u8,
        src_sc: *const f32,
        dst_q: *mut u8,
        dst_sc: *mut f32,
        gather_src: *const i32,
        n_assign: i32,
        topk: i32,
        row_bytes: i32,
        sc_row_bytes: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.route_group_gather else {
            return Ok(false);
        };
        let rc = unsafe {
            f(
                src_q, src_sc, dst_q, dst_sc, gather_src, n_assign, topk, row_bytes, sc_row_bytes,
                self.stream,
            )
        };
        self.kerr(rc, "dsv41_route_gather_rows")?;
        Ok(true)
    }

    /// Scatter the grouped expert output (row pitch `n`) back to the
    /// per-(row, slot) layout `[m * topk][n]`, driven by `perm_map`
    /// (original -> grouped). `perm_map[i] < 0` writes zeros. Returns `Ok(false)`
    /// when the `.so` has no entry point.
    pub fn route_scatter_rows(
        &self,
        src: *const f32,
        dst: *mut f32,
        perm_map: *const i32,
        n_assign: i32,
        n: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.route_group_scatter else {
            return Ok(false);
        };
        let rc = unsafe { f(src, dst, perm_map, n_assign, n, self.stream) };
        self.kerr(rc, "dsv41_route_scatter_rows")?;
        Ok(true)
    }

    /// True when the loaded `.so` carries the whole grouped-routing set
    /// (`dsv41_route_group` + `dsv41_route_gather_rows` +
    /// `dsv41_route_scatter_rows`). Probed as a SET on purpose: a `.so` carrying
    /// only the layout builder could not move a single activation, and a caller
    /// that armed `DSV41_EXPERT_GROUPED` would then measure the ungrouped path
    /// (the project's #1 measurement-bias trap). `supports_moe_batch`'s
    /// convention.
    pub fn supports_route_group(&self) -> bool {
        self.kernels.route_group.is_some()
            && self.kernels.route_group_gather.is_some()
            && self.kernels.route_group_scatter.is_some()
    }
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

    /// COMPRESSOR-MROWS: the verify block's `seqlen = m` rows of the decode
    /// compressor as ONE 1-block launch (rows ascending — see
    /// [`Self::compressor_fused_mrows_on`]'s kernel header). `b == 1`, `seqlen >=
    /// 1`, `ratio > 1` — the caller gates the shape and the
    /// `DSV41_COMPRESSOR_MROWS` flag, and an older .so without the symbol falls
    /// back to the per-row `compressor_pool_on` + `compress_commit_on` pair (see
    /// [`Self::supports_compressor_fused_mrows`]).
    ///
    /// `clen_rows` / `latent_rows` are the read side's per-row snapshots and may
    /// be NULL; a hoisting caller MUST pass them (see `chain_dev::
    /// compress_rows_fused`), because the arms that read the live counter or the
    /// shared `latent` after the launch would otherwise see the block's END state.
    #[allow(clippy::too_many_arguments)]
    pub fn compressor_fused_mrows_on(
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
        clen_rows: *mut std::os::raw::c_int,
        latent_rows: *mut f32,
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
        let f = self.need(
            self.kernels.compressor_fused_mrows,
            "dsv41_compressor_fused_mrows",
        )?;
        let rc = unsafe {
            f(
                kvp, scp, norm_w, state_kv, state_score, latent, out_rows, cos, sin, ring, clen,
                clen_rows, latent_rows, b, seqlen, head_dim, ratio, rope_dim, half, window,
                pos_ctr, eps, s,
            )
        };
        self.kerr(rc, "dsv41_compressor_fused_mrows")
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

    /// DSpark draft head: ONE step of the sequential block sampler
    /// (`dspark.rs::forward_head`'s Markov loop). See the kernel comment in
    /// `dsv41_glue.cu` for the sampling numerics (at the reference's constant
    /// `u == 1` the Gumbel-max is exactly the stable argmax of the biased row,
    /// so a single vocab pass is enough) and for the `partial`/`ctr` contract.
    /// The caller issues this once per draft row, in order.
    #[allow(clippy::too_many_arguments)]
    pub fn dspark_markov_head(
        &self,
        logits: *mut f32,
        h: *const f32,
        markov_embed: *const f32,
        markov_head: *const f32,
        confidence_proj: *const f32,
        ids: *mut i32,
        confidence: *mut f32,
        dim: i32,
        vocab: i32,
        markov_rank: i32,
        step: i32,
        partial: *mut u64,
        ctr: *mut u32,
    ) -> Result<()> {
        let f = self.need(self.kernels.dspark_markov_head, "dsv41_dspark_markov_head")?;
        let rc = unsafe {
            f(
                logits, h, markov_embed, markov_head, confidence_proj, ids, confidence, dim, vocab,
                markov_rank, step, partial, ctr, self.stream,
            )
        };
        self.kerr(rc, "dsv41_dspark_markov_head")
    }

    /// Vocabulary-sliced Markov step (DSV41_MARKOV_SLICED): the same program as
    /// [`Self::dspark_markov_head`] restricted to this rank's `[idx_off,
    /// idx_off + n)` rows of the replicated `markov_head` (the caller passes that
    /// slice's base pointer). It biases `logits`' `step` row (pitch `n`) and
    /// publishes the slice's packed winner — key = (value key | ~(idx_off + v)),
    /// i.e. carrying the GLOBAL index so the fold's winner and tie rule are the
    /// full-vocabulary argmax's — into `local_key[0]`. It does NOT write
    /// `ids[step + 1]`: that is [`Self::argmax_key_pub`]'s job, and leaving a
    /// local answer behind would be a plausible-looking wrong token.
    ///
    /// `markov_embed` is the FULL [vocab, mr] tensor (the step's token is global);
    /// only `markov_head` is sliced.
    #[allow(clippy::too_many_arguments)]
    pub fn dspark_markov_head_sliced(
        &self,
        logits: *mut f32,
        h: *const f32,
        markov_embed: *const f32,
        markov_head: *const f32,
        confidence_proj: *const f32,
        ids: *mut i32,
        confidence: *mut f32,
        dim: i32,
        n: i32,
        markov_rank: i32,
        step: i32,
        idx_off: i32,
        local_key: *mut u64,
        partial: *mut u64,
        ctr: *mut u32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.dspark_markov_head_sliced else {
            return Ok(false);
        };
        let rc = unsafe {
            f(
                logits, h, markov_embed, markov_head, confidence_proj, ids, confidence, dim, n,
                markov_rank, step, idx_off, local_key, partial, ctr, self.stream,
            )
        };
        self.kerr(rc, "dsv41_dspark_markov_head_sliced")?;
        Ok(true)
    }

    /// Fold the ranks' already-packed argmax keys into `out` (one global index)
    /// as ONE v5 epoch round. `Ok(false)` when the loaded .so predates the symbol
    /// or the key cannot fit the v5 slot (the kernel's DECLINE sentinel) — the
    /// caller gates the geometry on [`Self::supports_argmax_key_pub`] and on the
    /// slot size, so a decline here must be treated as an inconsistency, not
    /// swallowed into a fallback that has already written a sliced layout.
    ///
    /// `pos_ctr` is deliberately not exposed: the draft owns no device position
    /// counter, and a Markov step must never advance one.
    #[allow(clippy::too_many_arguments)]
    pub fn argmax_key_pub(
        &self,
        packed: *const u64,
        out: *mut c_int,
        peer_staging: *const *mut u64,
        ready_tbl: *const *mut u32,
        epoch: *mut c_uint,
        staging_local: *mut u64,
        ready_local: *const c_uint,
        world: i32,
        rank: i32,
        stride_bytes: i64,
    ) -> Result<bool> {
        let Some(f) = self.kernels.argmax_key_pub else {
            return Ok(false);
        };
        let rc = unsafe {
            f(
                packed, out, peer_staging, ready_tbl, epoch, staging_local, ready_local, world,
                rank, stride_bytes, self.stream,
            )
        };
        if rc == 1 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_argmax_key_pub")?;
        Ok(true)
    }

    /// Is the vocabulary-sliced Markov step in this .so? The draft gates the
    /// slice geometry on it BEFORE the head GEMV takes an offset, so a stale .so
    /// keeps the full-vocabulary `logits` pitch instead of leaving a half-sliced
    /// buffer for the fallback to misread.
    pub fn supports_dspark_markov_head_sliced(&self) -> bool {
        self.kernels.dspark_markov_head_sliced.is_some()
    }

    /// Is the packed-key cross-rank fold in this .so? Both halves of the sliced
    /// Markov path must be present; the kernel alone would leave `ids[step + 1]`
    /// as this rank's LOCAL winner, which is a wrong token that looks plausible.
    pub fn supports_argmax_key_pub(&self) -> bool {
        self.kernels.argmax_key_pub.is_some()
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

    /// ROW-FOLD (`DSV41_ROW_FOLD_GATE`): the MoE gate's `m` activation rows in
    /// ONE multi-row GEMV — `x` = the `[rows, k]` block, `out` = `[rows, n]`,
    /// `w` = the same bf16 `[n, k]` gate weight.
    ///
    /// THE PROGRAM THIS MATCHES is [`Self::gemv_bf16`]'s for the gate's shape:
    /// `n = n_routed < GEMV_V2_MAX_N`, so the per-row call takes
    /// `ferrite_gemv_bf16_v2` (`gemv_bf16_v2_kernel<WPR>`, WPR = `gv2_wpr(n)`),
    /// and `ferrite_gemv_bf16_nt(..., nrows = m)` is that program's m-token form
    /// (`gemv_bf16_nt_kernel<NT, WPR>`): the SAME `gv2_wpr` heuristic, the same
    /// `kper`/`k0`/`k1` K-slice per warp, the same uint4 weight load + four
    /// `__bfloat1622float2` decodes + two 4-term FMA groups, the same
    /// `__shfl_down_sync` tree, the same smem partial fold
    /// (`sum += part[t][(warp/WPR)*WPR + j]`, j ascending) — with one independent
    /// accumulator per token and no cross-token recombination. Row `r` is
    /// therefore bit-identical to the single-row `gemv_bf16` of row `r` it
    /// replaces (the `tests_dsv41_head_mrows` suite asserts exactly this
    /// nt-vs-m×v2 identity; `tests_gate_mrows.cu` re-asserts it at the gate's own
    /// WPR > 1 shape, which is where the smem fold — not the head's WPR == 1
    /// path — is what has to match).
    ///
    /// ⚠️ NOT `head_gemv_bf16_mrows`: that kernel is the WPR == 1 program only
    /// (`out_f >= 16384`), which is the head's shape, not the gate's. Reusing it
    /// here would silently change the K-split and with it the summation order.
    ///
    /// `Ok(false)` = NOT performed, keep the per-row loop: the loaded .so
    /// predates the symbol, the per-row path would NOT take v2 (so there is no
    /// parity to claim), or the shape is outside `ferrite_gemv_bf16_nt`'s
    /// dispatch (`k % 8 != 0`, `rows` not in 2..=8/12/16).
    pub fn gemv_bf16_mrows(
        &self,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        rows: i32,
        n: i32,
        k: i32,
    ) -> Result<bool> {
        // Only fold when `gemv_bf16` would take v2: v1 is a different
        // accumulation order and the nt kernel does not reproduce it.
        if !gemv_bf16_v2_wanted(n) || self.kernels.gemv_bf16_v2.is_none() {
            return Ok(false);
        }
        let Some(f) = self.kernels.gemv_bf16_nt else {
            return Ok(false);
        };
        if !(2..=8).contains(&rows) || (k & 7) != 0 {
            return Ok(false);
        }
        // `FERRITE_GEMV_SKIP` is a timing-only ablation INSIDE the nt entry: it
        // would make this launch a no-op while the per-row path computed the
        // scores. Treat it as "do not fold".
        if gemv_bf16_nt_skip() {
            return Ok(false);
        }
        let rc = unsafe { f(x, w, std::ptr::null(), out, k, n, rows, self.stream) };
        self.kerr(rc, "ferrite_gemv_bf16_nt")?;
        Ok(true)
    }

    /// The MoE gate's multi-row GEMV (`DSV41_GATE_MROWS`) — the plan-named entry
    /// (`ferrite_gemv_bf16_v2_mrows`, C side: `ferrite_kernels.cu`, next to the
    /// v2 launcher).
    ///
    /// It runs the SAME program [`Self::gemv_bf16_mrows`] runs, and that method's
    /// contract applies verbatim: `n = n_routed = 384 < GEMV_V2_MAX_N`, so the
    /// per-row call takes `ferrite_gemv_bf16_v2` (`gemv_bf16_v2_kernel<WPR>`,
    /// `WPR = gv2_wpr(384) = 8`, `rpb = 1`), and the multi-row form is
    /// `gemv_bf16_nt_kernel<NT, 8>` — the same `kper`/`k0`/`k1` K-slice per warp,
    /// the same uint4 weight load + four `__bfloat1622float2` decodes + two
    /// 4-term FMA groups, the same `__shfl_down_sync` tree, the same per-token
    /// smem fold (`sum += part[t][(warp/WPR)*WPR + j]`, j ascending). Row r is
    /// BIT-IDENTICAL to the single-row `gemv_bf16` of row r (`tests_gate_mrows.cu`
    /// asserts it at the gate's own WPR = 8/4 shapes), which is what makes this a
    /// pure byte/launch win: the gate weight is streamed ONCE for all `m` rows
    /// instead of `m` times, and `m` launches become one.
    ///
    /// ⚠️ NOT `gemv_bf16_mrows`' v1 order: `head_gemv_bf16_mrows` is the
    /// `WPR == 1` program only (the head's shape, `out_f >= 16384`). The gate at
    /// WPR = 8 has a cross-warp K-split whose smem fold is part of the
    /// accumulation order — using the head's kernel here would silently change it
    /// (the FOLD regression, commit b8b67c0).
    ///
    /// `Ok(false)` = NOT performed, keep the per-row loop: the .so predates BOTH
    /// multi-row symbols, the per-row path would NOT take v2 (v1 is a different
    /// accumulation order, so there is no parity to claim), `rows` is outside
    /// 1..=8, or `k % 8 != 0` (the same bound the C entry declines on).
    pub fn gemv_bf16_v2_mrows(
        &self,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        rows: i32,
        n: i32,
        k: i32,
    ) -> Result<bool> {
        // Only fold when `gemv_bf16` would take v2 (see `gemv_bf16_mrows`).
        if !gemv_bf16_v2_wanted(n) || self.kernels.gemv_bf16_v2.is_none() {
            return Ok(false);
        }
        if !(1..=GEMV_V2_MROWS_MAX).contains(&rows) || (k & 7) != 0 {
            return Ok(false);
        }
        // `FERRITE_GEMV_SKIP` is a timing-only ablation INSIDE the multi-row
        // entry: treating it as "do not fold" keeps the ablation honest.
        if gemv_bf16_nt_skip() {
            return Ok(false);
        }
        // The gate's own symbol first; `ferrite_gemv_bf16_nt` — the SAME program
        // under its historical name — is the fallback for a .so built before
        // this entry landed. (The C entry handles `rows == 1` itself; the nt
        // dispatch starts at 2.)
        if let Some(f) = self.kernels.gemv_bf16_v2_mrows {
            let rc = unsafe { f(x, w, std::ptr::null(), out, k, n, rows, self.stream) };
            self.kerr(rc, "ferrite_gemv_bf16_v2_mrows")?;
            return Ok(true);
        }
        if rows < 2 {
            return Ok(false);
        }
        let Some(f) = self.kernels.gemv_bf16_nt else {
            return Ok(false);
        };
        let rc = unsafe { f(x, w, std::ptr::null(), out, k, n, rows, self.stream) };
        self.kerr(rc, "ferrite_gemv_bf16_nt")?;
        Ok(true)
    }

    /// Multi-row twin of `gemv_bf16` for the DSpark verify's head: ONE launch
    /// streams the `[n, k]` bf16 weight ONCE and folds all `rows` activation
    /// rows against it (`x` = `[rows, k]` f32, `out` = `[rows, n]` f32), where
    /// the per-row loop paid one full pass per row. `head.weight` is
    /// `[129280, 5120]` bf16 = 1262 MB REPLICATED on every rank, so six rows
    /// streamed it six times — 6 x 298us = 1.79ms, the verify's largest single
    /// term after the projection family (dspark-verify-perf-plan.md §1.2).
    ///
    /// Row r of the multi-row launch is BIT-IDENTICAL to the production
    /// single-row head GEMV of row r: `gemv_bf16_nt_kernel`'s per-token body at
    /// WPR == 1 (`ferrite_kernels.cu`), which is the same body
    /// `gemv_bf16_v2_kernel` runs and the program the head's shape
    /// (`out_f >= 16384`) gets from `gv2_wpr`. The kernel's header carries the
    /// argument (same lane->k mapping, same transcribed 8-element FMA groups,
    /// same `__shfl_down_sync` tree, per-row independent accumulators, no
    /// cross-row recombination, no K-split).
    ///
    /// ⚠️ DOMAIN OF THAT PARITY: (a) `out_f >= 16384`, i.e. WPR == 1 — below it
    /// the production GEMV K-splits a row across WPR warps and folds the partials
    /// in smem, a different program; (b) `k % 8 == 0`, which this launcher
    /// enforces below by declining. The verify's head (`n = 129280`) satisfies
    /// both. Do not reuse this kernel outside that domain and expect a bit-exact
    /// A/B against the `gemv_bf16` it would then be replacing.
    ///
    /// `Ok(false)` means NOT performed — keep the per-row loop: either the
    /// loaded .so predates the symbol, or `rows` is outside the kernel's 1..=8
    /// dispatch set, or `k` is not a multiple of 8 (the same bound the C entry
    /// enforces).
    pub fn head_gemv_bf16_mrows(
        &self,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        rows: i32,
        n: i32,
        k: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.head_gemv_bf16_mrows else {
            return Ok(false);
        };
        if !(1..=8).contains(&rows) || (k & 7) != 0 {
            return Ok(false);
        }
        let rc = unsafe { f(w, x, out, rows, n, k, self.stream) };
        self.kerr(rc, "dsv41_head_gemv_bf16_mrows")?;
        Ok(true)
    }

    /// V1-ORDER multi-row twin of [`Self::gemv_bf16`] for the DSpark verify's
    /// SLICED head: ONE launch streams the `[n, k]` bf16 weight ONCE and folds
    /// all `rows` activation rows into it, where the per-row loop paid `m`
    /// passes over the slice (`seg * k * 2` = 158 MB at world = 8; the head is
    /// 1262 MB REPLICATED, `dspark-verify-perf-plan.md` §1.2). `x` is [rows, k]
    /// f32 (`xn_r`), `out` is [rows, n] f32 (`logits_r`, the slice pitch).
    ///
    /// Row r is BIT-IDENTICAL to the single-row `gemv_bf16` launch of row r,
    /// which for the head's shape is v1: `gemv_bf16_v2_wanted(n)` diverts only
    /// `n < 2048`, and the head's `n` is the vocabulary (or its per-rank slice,
    /// 16160 at world = 8), so the per-row program is `dsv41_gemv_bf16` /
    /// `gemv_bf16_kernel`. The kernel header (dsv41_glue.cu) carries the
    /// transcription argument: same `c = lane; c += 32` scan, one independent
    /// accumulator per row, same `__shfl_xor_sync` tree, no K-split.
    ///
    /// ⚠️ WHY THIS EXISTS NEXT TO [`Self::head_gemv_bf16_mrows`]. That entry is
    /// the v2 (`gemv_bf16_nt`, WPR == 1) program, so folding the v1 head with it
    /// is a NUMERICAL change, and it measured as one (`verify_head_fold`: 33%
    /// echo). It is the fold to use only when the per-row program IS v2 — i.e.
    /// for `n < 2048`. The two entries have IDENTICAL ABIs, so a caller must
    /// know which program it is claiming parity with; the verify's head needs
    /// this one (v1).
    ///
    /// `Ok(false)` means NOT performed — keep the per-row loop: either the
    /// loaded .so predates the symbol or `rows` is outside the kernel's 1..=8
    /// dispatch set (the C entry refuses `m > 8` rather than reading garbage).
    /// No `k` bound: v1's body is scalar, so any `k` is in its domain.
    pub fn head_gemv_bf16_v1_mrows(
        &self,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        rows: i32,
        n: i32,
        k: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.gemv_bf16_v1_mrows else {
            return Ok(false);
        };
        if !(1..=8).contains(&rows) {
            return Ok(false);
        }
        let rc = unsafe { f(w, x, out, rows, n, k, self.stream) };
        self.kerr(rc, "dsv41_gemv_bf16_v1_mrows")?;
        Ok(true)
    }

    /// The draft's block-diagonal `wo_a` as ONE weight-stationary launch.
    ///
    /// `a` is the quantised attention output `[rows][a_stride/4 f32]` (fp8 bytes,
    /// group `g`'s k-element segment at `+g*k`), `w` the [n, k] fp8 weight whose
    /// group `g` row block starts at `+g*n*k`, `out` the f32 `[rows][out_stride]`
    /// result whose group `g` columns start at `+g*n`. The C entry lays the group
    /// out along `grid.y`, so all `groups` groups ride in one launch.
    ///
    /// Every (row, r) of this launch is BIT-IDENTICAL to the m=1 `gemm_fp8_mx`
    /// of the same (group, r) the caller would otherwise issue: same ascending
    /// kb / lane->c walk, same staged bytes, same serial `acc` chain with the same
    /// `shfl_xor` tree, no cross-row recombination (the kernel header carries the
    /// C1-C6 argument). What changes is the traffic: the group's weight block is
    /// read ONCE for all `rows` rows instead of once per row, and `groups * rows`
    /// launches collapse into one.
    ///
    /// `Ok(false)` means NOT performed — keep the per-(group, row) loop: either
    /// the loaded .so predates the symbol or the C entry declined the shape/mode
    /// (it returns 2, not cudaErrorInvalidValue, so a decline can never be read as
    /// a launch failure).
    #[allow(clippy::too_many_arguments)]
    pub fn wo_a_grouped_fp8(
        &self,
        a: *const u8,
        a_scale: *const f32,
        w: *const u8,
        w_scale: *const u8,
        bias: *const f32,
        out: *mut f32,
        groups: i32,
        rows: i32,
        n: i32,
        k: i32,
        a_stride: i32,
        out_stride: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.wo_a_grouped_fp8 else {
            return Ok(false);
        };
        let rc = unsafe {
            f(
                a, a_scale, w, w_scale, bias, out, groups, rows, n, k, a_stride, out_stride,
                self.stream,
            )
        };
        // 2 = the kernel's "declined" (shape/mode), the caller falls back.
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_wo_a_grouped_fp8")?;
        Ok(true)
    }

    /// Dense multi-row fp8 GEMV: `out[r, :] = a[r, :] @ w^T + bias` for every
    /// `r < rows`, with the weight rows staged ONCE for the whole batch.
    ///
    /// The weight-stationary form of [`Self::gemm_fp8_mx`]'s **m == 1** program.
    /// That distinction is the whole reason this entry point exists:
    /// `dsv41_gemm_fp8_mx` dispatches on `m` between two different programs
    /// (SIMT warp-per-row GEMV at `m == 1`, the 16-row tile MMA at `m > 1`), so
    /// passing `m = rows` to it would silently change the summation — the
    /// verify's iron rule is "row r of an m-row launch == the m=1 decode of row
    /// r", bit for bit. This kernel reproduces the `m == 1` consume expression,
    /// its ascending-kb walk and its `shfl_xor` tree per row (the kernel header
    /// in `dsv41_kernels.cu` carries the C1-C6 argument).
    ///
    /// `a` is `[rows, k]` fp8 e4m3 (row `r` at `+r*k`), `a_scale` `[rows, k/32]`
    /// f32, `out` f32 with row `r`'s element `row` at `+r*out_stride + row`.
    /// `out_stride` is its own parameter and NOT implied by `n`: the wq_b call
    /// site writes `nlh*head_dim` elements of an `nh*head_dim` row.
    ///
    /// `Ok(false)` means NOT performed — keep the per-row loop: either the loaded
    /// .so predates the symbol, or the C entry declined (it returns 2, never
    /// cudaErrorInvalidValue, so a decline can never read as a launch failure).
    /// `rows` outside 1..=8 is refused here as well, the same bound the C entry
    /// enforces and the template dispatch covers.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_mrows(
        &self,
        a: *const u8,
        a_scale: *const f32,
        w: *const u8,
        w_scale: *const u8,
        bias: *const f32,
        out: *mut f32,
        rows: i32,
        n: i32,
        k: i32,
        out_stride: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.gemm_fp8_mrows else {
            return Ok(false);
        };
        if !(1..=8).contains(&rows) {
            return Ok(false);
        }
        let rc = unsafe {
            f(a, a_scale, w, w_scale, bias, out, rows, n, k, out_stride, self.stream)
        };
        // 2 = the kernel's "declined" (shape/mode), the caller falls back.
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_mrows")?;
        Ok(true)
    }

    /// True when the loaded .so carries the multi-row fp8 GEMV
    /// (`dsv41_gemm_fp8_mrows`). A stale .so leaves the projection sites on their
    /// per-row loop, which is the bit-exact reference they were verified against.
    pub fn supports_gemm_fp8_mrows(&self) -> bool {
        self.kernels.gemm_fp8_mrows.is_some()
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

    /// The MULTI-ROW hc-post fold (`ferrite_p2p_ar_v5_hcpost_rows`): AR v5 with
    /// `hc_post_inplace_rows` in the pubred epilogue. `hc_rows` is the number of
    /// rows in the payload and `hc_h` their stride (the residual's column count);
    /// `n` must be `hc_rows * hc_h`. `Ok(false)` when the `.so` predates the entry
    /// (or the launcher rejects the shape, returns 1), so the caller runs the
    /// plain `all_reduce_inplace` + `hc_post_inplace_rows` pair instead.
    ///
    /// The caller MUST NOT have issued any other all-reduce since its store/
    /// producer (the v5 epoch contract), exactly as [`Self::p2p_ar_v5_hcpost`].
    #[allow(clippy::too_many_arguments)]
    pub fn p2p_ar_v5_hcpost_rows(
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
        hc_rows: c_int,
    ) -> Result<bool> {
        let f = match self.kernels.p2p_ar_v5_hcpost_rows {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe {
            f(partial, staging_tbl, ready_tbl, epoch, staging_local, ready_local, out, n,
              world, my_rank, stride, hc_res, hc_post, hc_comb, hc_n, hc_h, hc_rows,
              self.stream)
        };
        // 1 == the launcher declined the shape (see
        // `ferrite_p2p_ar_v5_hcpost_rows`); the caller then runs the unfused pair.
        if rc == 1 {
            return Ok(false);
        }
        self.kerr(rc, "ferrite_p2p_ar_v5_hcpost_rows")?;
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

    /// DSpark verify: append the m-row block's kv rows into their ring slots
    /// and emit the per-row CAUSAL window indices `[m, window]` (row r's window
    /// ends at `*pos_ctr + r`, which makes the intra-block causal order a
    /// consequence of the ring geometry). One launch for both halves.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_ring_win(
        &self,
        ring: *mut f32,
        kv: *const f32,
        pos_ctr: *const c_int,
        window: i32,
        hd: i32,
        m: i32,
        idxs: *mut i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.verify_ring_win, "dsv41_verify_ring_win")?;
        let rc = unsafe { f(ring, kv, pos_ctr, window, hd, m, idxs, self.stream) };
        self.kerr(rc, "dsv41_verify_ring_win")
    }

    /// DSpark snapshot (P0): save ONE layer's `m` ring slots in a single launch.
    /// `snap` is that layer's `[m][head_dim]` slice and `pos_base` is `pos + 1`
    /// (the first row's slot base), so the kernel computes the same
    /// `(pos_base + j) % window` the host loop used. Falls back to the memcpy
    /// path when the loaded `.so` predates the kernel.
    pub fn dspark_ring_save(
        &self,
        snap: *mut f32,
        ring: *const f32,
        pos_base: i32,
        window: i32,
        hd: i32,
        m: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.dspark_ring_save else {
            return Ok(false);
        };
        let rc = unsafe { f(snap, ring, pos_base, window, hd, m, self.stream) };
        self.kerr(rc, "dsv41_dspark_ring_save")?;
        Ok(true)
    }

    /// DSpark rollback (P0): restore ONE layer's ring slots from the snapshot,
    /// keeping rows `0..keep` (`keep >= m` is a no-op). See
    /// [`Self::dspark_ring_save`] for the fallback contract.
    #[allow(clippy::too_many_arguments)]
    pub fn dspark_ring_restore(
        &self,
        ring: *mut f32,
        snap: *const f32,
        pos_base: i32,
        window: i32,
        hd: i32,
        m: i32,
        keep: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.dspark_ring_restore else {
            return Ok(false);
        };
        let rc = unsafe { f(ring, snap, pos_base, window, hd, m, keep, self.stream) };
        self.kerr(rc, "dsv41_dspark_ring_restore")?;
        Ok(true)
    }

    /// DSpark snapshot (P0): save ONE compress-source layer's carry -- both
    /// `state_kv`/`state_score` segments, `latent`, and the `clen`/`out_rows`
    /// counters -- in a single launch. `snap_state` is the layer's
    /// `2 * max_ratio * head_dim` slice, `snap_latent` its `head_dim` slice.
    #[allow(clippy::too_many_arguments)]
    pub fn dspark_comp_save(
        &self,
        state_kv: *const f32,
        state_score: *const f32,
        latent: *const f32,
        clen: *const c_int,
        out_rows: *const c_int,
        snap_state: *mut f32,
        snap_latent: *mut f32,
        snap_clen: *mut c_int,
        snap_out_rows: *mut c_int,
        ratio: i32,
        max_ratio: i32,
        hd: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.dspark_comp_save else {
            return Ok(false);
        };
        let rc = unsafe {
            f(
                state_kv, state_score, latent, clen, out_rows, snap_state, snap_latent, snap_clen,
                snap_out_rows, ratio, max_ratio, hd, self.stream,
            )
        };
        self.kerr(rc, "dsv41_dspark_comp_save")?;
        Ok(true)
    }

    /// DSpark rollback (P0): restore ONE compress-source layer's carry. The
    /// whole carry is restored regardless of a keep prefix (the compressor
    /// cannot be rewound row by row). See [`Self::dspark_ring_save`] for the
    /// fallback contract.
    #[allow(clippy::too_many_arguments)]
    pub fn dspark_comp_restore(
        &self,
        state_kv: *mut f32,
        state_score: *mut f32,
        latent: *mut f32,
        clen: *mut c_int,
        out_rows: *mut c_int,
        snap_state: *const f32,
        snap_latent: *const f32,
        snap_clen: *const c_int,
        snap_out_rows: *const c_int,
        ratio: i32,
        max_ratio: i32,
        hd: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.dspark_comp_restore else {
            return Ok(false);
        };
        let rc = unsafe {
            f(
                state_kv, state_score, latent, clen, out_rows, snap_state, snap_latent, snap_clen,
                snap_out_rows, ratio, max_ratio, hd, self.stream,
            )
        };
        self.kerr(rc, "dsv41_dspark_comp_restore")?;
        Ok(true)
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
    /// `act_e4m3` (trailing): `a` holds e4m3 bytes (1/value) + `a_scale` f32 per 32
    /// instead of the packed e2m1 nibbles (`DSV41_EXPERT_ACT_E4M3`).
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
        act_e4m3: i32,
    ) -> Result<()> {
        let f = self.need(
            self.kernels.expert_gate_up_fp4_indirect,
            "dsv41_expert_gate_up_fp4_indirect",
        )?;
        let rc = unsafe {
            f(
                a, a_scale, out, rows, dim, inter, limit, w1_base, w1_stride, w1s_base, w1s_stride,
                w3_base, w3_stride, w3s_base, w3s_stride, ids, slot, act_e4m3, self.stream,
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
    /// `act_e4m3` (trailing): `a` holds e4m3 bytes (1/value, row pitch `dim`)
    /// + `a_scale` f32 per 32, instead of packed e2m1 (`DSV41_EXPERT_ACT_E4M3`).
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
        act_e4m3: i32,
    ) -> Result<()> {
        let f = self.need(
            self.kernels.expert_gate_up_fp4_batched,
            "dsv41_expert_gate_up_fp4_batched",
        )?;
        let rc = unsafe {
            f(
                a, a_scale, out, out_slot_stride, rows, dim, inter, limit, slots, w1_base,
                w1_stride, w1s_base, w1s_stride, w3_base, w3_stride, w3s_base, w3s_stride, ids, ilv,
                act_e4m3, self.stream,
            )
        };
        self.kerr(rc, "dsv41_expert_gate_up_fp4_batched")
    }

    /// True when the loaded .so carries the DIRECT e4m3 activation path
    /// (`dsv41_expert_act_e4m3_cap`). A stale .so has no such symbol, and
    /// `DSV41_EXPERT_ACT_E4M3` must then stay OFF (reported once) instead of
    /// feeding e4m3 bytes to a kernel that decodes them as fp4 nibbles — a
    /// silent wrong answer, not a failure.
    pub fn supports_expert_act_e4m3(&self) -> bool {
        self.kernels.expert_act_e4m3_cap.is_some()
    }

    /// tcgen05 MXFP4 gate/up — the Phase-1 `kind::mxf4` swapAB kernel
    /// (`DSV41_TCGEN05_GATEUP_MXF4`, default OFF, read once inside the `.so`).
    ///
    /// The four `*_base`/`*_stride` pairs are the loader's actual weight planes
    /// (`w1` gate / `w3` up and their e8m0 scale planes, `load.rs:644`) and
    /// `ids[slot]` selects one expert per `grid.y` slot on the device — pass
    /// `nullptr` to make the bases BE the direct pointers (one expert).
    /// `act`/`act_scale` are the shared quantised row: packed e2m1 and its
    /// `[dim/32]` **f32 power-of-two** scales (the quantiser's native output;
    /// the kernel converts to e8m0 — see the kernels.rs ABI note). `out` holds
    /// `slots` `[2*inter]` blocks, `out_slot_stride` floats apart, with the
    /// `limit` clamp applied in the epilogue (`split = inter`).
    ///
    /// Returns `Ok(false)` when the `.so` gate is OFF (rc == 0, nothing ran) so
    /// a caller can keep the proven GEMV path. ⚠️ A REJECTED shape/alignment
    /// returns a nonzero code and IS an error: the caller must pre-check the
    /// contract (`dim % 128 == 0`, `2*inter % 128 == 0`, 16-byte aligned bases
    /// and strides) or it will fail the step instead of falling back.
    #[allow(clippy::too_many_arguments)]
    pub fn expert_tcgen05_gate_up_mxf4(
        &self,
        act: *const u8,
        act_scale: *const f32,
        out: *mut f32,
        out_slot_stride: i64,
        inter: i32,
        dim: i32,
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
    ) -> Result<bool> {
        let f = self.need(
            self.kernels.expert_tcgen05_gate_up_mxf4,
            "dsv41_expert_tcgen05_gate_up_mxf4",
        )?;
        let rc = unsafe {
            f(
                act, act_scale, out, out_slot_stride, inter, dim, limit, slots, w1_base, w1_stride,
                w1s_base, w1s_stride, w3_base, w3_stride, w3s_base, w3s_stride, ids, self.stream,
            )
        };
        if rc == 0 {
            return Ok(false); // gate OFF: nothing ran, the caller falls back
        }
        self.kerr(rc, "dsv41_expert_tcgen05_gate_up_mxf4")?;
        Ok(true)
    }

    /// tcgen05 **e4m3-activation** gate/up — the `tc5::e4` arm
    /// (`DSV41_EXPERT_TCGEN05_E4M3`, default OFF, read once inside the `.so`).
    ///
    /// Byte-for-byte the same ABI and the same arguments as
    /// [`Self::expert_tcgen05_gate_up_mxf4`]; only the MEANING of `act` changes:
    /// here it is the OFFICIAL e4m3 activation (`act_quant(e4m3, block=32)`),
    /// i.e. **`dim` bytes per row, one byte per value**, not `dim/2` packed
    /// bytes. `act_scale` is the same `[dim/32]` f32 power-of-two vector (the
    /// kernel converts to e8m0 either way). `out`/`out_slot_stride`/`limit`/
    /// `split = inter` are identical to the e2m1 arm, including the UNFUSED
    /// `[2*inter]` gate|up layout (the epilogue only clamps).
    ///
    /// ⚠️ The e4m3 shape is at least as strict as the e2m1 one: `dim % 128 == 0`,
    /// `2*inter % 128 == 0`, 16-byte aligned bases/strides **and** a 16-byte
    /// aligned `act` (the activation row is addressed directly in 16-byte
    /// chunks). A rejected shape returns a nonzero code and IS an error.
    #[allow(clippy::too_many_arguments)]
    pub fn expert_tcgen05_gate_up_e4m3(
        &self,
        act: *const u8,
        act_scale: *const f32,
        out: *mut f32,
        out_slot_stride: i64,
        inter: i32,
        dim: i32,
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
    ) -> Result<bool> {
        let f = self.need(
            self.kernels.expert_tcgen05_gate_up_e4m3,
            "dsv41_expert_tcgen05_gate_up_e4m3",
        )?;
        let rc = unsafe {
            f(
                act, act_scale, out, out_slot_stride, inter, dim, limit, slots, w1_base, w1_stride,
                w1s_base, w1s_stride, w3_base, w3_stride, w3s_base, w3s_stride, ids, self.stream,
            )
        };
        if rc == 0 {
            return Ok(false); // gate OFF: nothing ran, the caller falls back
        }
        self.kerr(rc, "dsv41_expert_tcgen05_gate_up_e4m3")?;
        Ok(true)
    }

    /// True when the loaded `.so` carries the tcgen05 e4m3-activation gate/up
    /// entry point (`dsv41_expert_tcgen05_gate_up_e4m3`, compiled in by
    /// `build.sh` unless `DSV41_BUILD_TCGEN05_E4M3=0` is exported). The runtime
    /// gate `DSV41_EXPERT_TCGEN05_E4M3` is still default OFF, so a present symbol
    /// changes no behaviour on its own.
    pub fn supports_expert_tcgen05_e4m3(&self) -> bool {
        self.kernels.expert_tcgen05_gate_up_e4m3.is_some()
    }

    /// tcgen05 **e4m3-activation dense M=128 tile** expert GEMM — the
    /// `tc5::e4x` arm (`DSV41_EXPERT_TCGEN05_E4M3`, default OFF, read once
    /// inside the `.so`; the SAME variable the swapAB e4m3 arm above reads,
    /// since the two differ only in launch shape).
    ///
    /// Unlike the swapAB arm this one is a plain dense tile GEMM, so the
    /// arguments are shapes and pointers rather than a slot pitch:
    ///   `a`/`a_scale` : the DENSE activation `[rows, k]` **e4m3, one byte per
    ///                   value** (exactly what `quant_fp8` writes) and its
    ///                   `[rows, k/32]` **f32** per-row power-of-two scales;
    ///   `b`/`b_scale` : the PACKED fp4 weight `[b_rows, k/2]` + e8m0
    ///                   `[b_rows, k/32]` (the loader's planes verbatim);
    ///   `b_hi`/`b_hi_scale` + `b_split` : the SECOND pool (up for gate/up,
    ///                   `b_split = inter`; pass `null` and a negative
    ///                   `b_split` for the single-pool down form). The kernel
    ///                   reads `b` for `n < b_split` and `b_hi` at
    ///                   `row = n - b_split` otherwise;
    ///   `out` : `[rows, n_total]`, row pitch `n_total` (DENSE — not the
    ///                   `[row][slot][act_slot]` pitch the batched GEMV writes);
    ///   `epi_mode`/`limit`/`row_weight` : 1 = gate/up clamp, 2 = down times
    ///                   `row_weight[row]`, 3 = accumulate into `out`.
    /// `ids != nullptr` derives every weight plane from `base + ids[slot] *
    /// stride`, i.e. **ONE expert for the whole launch**.
    ///
    /// Returns `Ok(false)` when the `.so` gate is OFF (rc == 0, nothing ran) so
    /// a caller can keep the proven path. ⚠️ A REJECTED shape/alignment returns
    /// a nonzero code and IS an error: the caller must pre-check the contract
    /// (`k % 64 == 0`, `rows % 128 == 0`, `n_total % 64 == 0`, 16-byte aligned
    /// `a`/`b`/`b_hi` and strides) or the step fails instead of falling back.
    #[allow(clippy::too_many_arguments)]
    pub fn expert_gemm_e4m3_ext(
        &self,
        a: *const u8,
        a_scale: *const f32,
        b: *const u8,
        b_scale: *const u8,
        b_hi: *const u8,
        b_hi_scale: *const u8,
        out: *mut f32,
        rows: i32,
        n_total: i32,
        k: i32,
        b_split: i32,
        epi_mode: i32,
        limit: f32,
        row_weight: *const f32,
        b_base: *const u8,
        b_stride: i64,
        bs_base: *const u8,
        bs_stride: i64,
        bh_base: *const u8,
        bh_stride: i64,
        bhs_base: *const u8,
        bhs_stride: i64,
        ids: *const i32,
        slot: i32,
    ) -> Result<bool> {
        let f = self.need(
            self.kernels.expert_gemm_e4m3_ext,
            "dsv41_expert_gemm_e4m3_ext",
        )?;
        let rc = unsafe {
            f(
                a, a_scale, b, b_scale, b_hi, b_hi_scale, out, rows, n_total, k, b_split, epi_mode,
                limit, row_weight, b_base, b_stride, bs_base, bs_stride, bh_base, bh_stride,
                bhs_base, bhs_stride, ids, slot, self.stream,
            )
        };
        if rc == 0 {
            return Ok(false); // gate OFF: nothing ran, the caller falls back
        }
        self.kerr(rc, "dsv41_expert_gemm_e4m3_ext")?;
        Ok(true)
    }

    /// True when the loaded `.so` carries the tcgen05 e4m3-activation
    /// DENSE-TILE expert GEMM (`dsv41_expert_gemm_e4m3_ext`, `tc5::e4x`,
    /// compiled in by `build.sh` unless `DSV41_BUILD_TCGEN05_E4M3=0`). A
    /// separate probe from [`Self::supports_expert_tcgen05_e4m3`] on purpose:
    /// the two arms ship under one skeleton flag but are independent symbols,
    /// so a `.so` carrying only the swapAB entry point must leave this arm on
    /// the fallback instead of failing the step. Its runtime gate
    /// (`DSV41_EXPERT_TCGEN05_E4M3`) is still default OFF, so a present symbol
    /// changes no behaviour on its own.
    pub fn supports_expert_gemm_e4m3_ext(&self) -> bool {
        self.kernels.expert_gemm_e4m3_ext.is_some()
    }

    /// tcgen05 **e4m3-activation GROUP-INDEXED MASKED** expert GEMM — the
    /// `m_grouped_gemm_nt_masked` form the grouped routed path needs. Same MMAs,
    /// same K-fold and same formats as [`Self::expert_gemm_e4m3_ext`]; what
    /// changes is WHERE a CTA's operands come from:
    ///
    ///   `a`/`a_scale` : the GROUPED activation `[n_assign, k]` (e4m3, one byte
    ///                   per value) + `[n_assign, k/32]` f32 scales — the gathered
    ///                   rows of `dsv41_route_gather_rows`;
    ///   `out`         : `[n_assign, n_total]`, GROUPED row order (the scatter
    ///                   `dsv41_route_scatter_rows` moves it to the
    ///                   `[m*topk][n]` per-(row, slot) layout);
    ///   `active`/`n_active`, `counts`, `starts` : the group table of
    ///                   `dsv41_route_group`. CTA (x = N tile, y = m-tile inside
    ///                   one expert, z = cursor) reads `e = active[z]`,
    ///                   `row_base = starts[e] + y*128`, and masks the rows past
    ///                   `counts[e]`;
    ///   `m_cap`       : the caller's per-expert row capacity (`grp_m_cap`), the
    ///                   bound the grid's m-tile count is derived from;
    ///   `n_assign`    : `m * topk`, the grid's z bound (>= `n_active` always);
    ///   `b*`          : the four weight planes with their expert strides
    ///                   (identical to the dense arm's indirect form).
    ///
    /// `epi_mode` is 0 (plain store) or **1** (gate/up clamp) only: modes 2/3 are
    /// per-(row, slot) concepts and the launcher REFUSES them rather than index
    /// the wrong row — the grouped pipeline applies `route_w_r` after the
    /// scatter.
    ///
    /// Returns `Ok(false)` when the `.so` gates are OFF (rc == 0, nothing ran),
    /// so the caller keeps the proven per-(row, slot) launches. ⚠️ A REJECTED
    /// shape returns a nonzero code and IS an error: the caller must pre-check
    /// the contract (`k % 64 == 0`, `n_total % 64 == 0`, 16-byte aligned
    /// operand bases and B strides).
    #[allow(clippy::too_many_arguments)]
    pub fn expert_gemm_e4m3_grouped(
        &self,
        a: *const u8,
        a_scale: *const f32,
        out: *mut f32,
        active: *const i32,
        n_active: *const i32,
        counts: *const i32,
        starts: *const i32,
        n_experts: i32,
        n_assign: i32,
        m_cap: i32,
        n_total: i32,
        k: i32,
        b_split: i32,
        epi_mode: i32,
        limit: f32,
        b_base: *const u8,
        b_stride: i64,
        bs_base: *const u8,
        bs_stride: i64,
        bh_base: *const u8,
        bh_stride: i64,
        bhs_base: *const u8,
        bhs_stride: i64,
    ) -> Result<bool> {
        let f = self.need(
            self.kernels.expert_gemm_e4m3_grouped,
            "dsv41_expert_gemm_e4m3_grouped",
        )?;
        let rc = unsafe {
            f(
                a, a_scale, out, active, n_active, counts, starts, n_experts, n_assign, m_cap,
                n_total, k, b_split, epi_mode, limit, b_base, b_stride, bs_base, bs_stride,
                bh_base, bh_stride, bhs_base, bhs_stride, self.stream,
            )
        };
        if rc == 0 {
            return Ok(false); // either gate OFF: nothing ran, the caller falls back
        }
        self.kerr(rc, "dsv41_expert_gemm_e4m3_grouped")?;
        Ok(true)
    }

    /// True when the loaded `.so` carries the group-indexed masked e4m3 GEMM
    /// (`dsv41_expert_gemm_e4m3_grouped`) — a separate symbol from the dense
    /// arm's on purpose (same staged-bring-up rule as
    /// [`Self::supports_expert_gemm_e4m3_ext`]): a `.so` built before this arm
    /// existed must leave the grouped path on its fallback instead of failing the
    /// step. Its runtime gates (`DSV41_EXPERT_TCGEN05_E4M3` **and**
    /// `DSV41_EXPERT_GROUPED`) are both default OFF, so a present symbol changes
    /// no behaviour on its own.
    pub fn supports_expert_gemm_e4m3_grouped(&self) -> bool {
        self.kernels.expert_gemm_e4m3_grouped.is_some()
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

    /// Multi-row rmsnorm for the verify block (`DSV41_NORM_MROWS`): `rows` norm
    /// rows of `dim`, row stride `dim`, the shared weight row `w[dim]`, in place
    /// when `out == x`.
    ///
    /// This is the dsv41-side twin of [`Self::rmsnorm`]'s per-row program at the
    /// same blockDim (see `dsv41_rmsnorm_rows_kernel`), so a `rows = m` call
    /// replaces the block's norm launch and each row lands bit just as it would
    /// have through the per-row entry. `Ok(false)` when the loaded `.so` predates
    /// the symbol (or when the caller's gate is off), and the caller then keeps
    /// the launch it already made - strictly additive, no new failure mode.
    pub fn rmsnorm_rows(
        &self,
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        rows: i32,
        dim: i32,
        eps: f32,
    ) -> Result<bool> {
        let f = match self.kernels.rmsnorm_rows {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe { f(x, w, out, rows, dim, eps, self.stream) };
        self.kerr(rc, "dsv41_rmsnorm_rows")?;
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

    /// True when the loaded `.so` carries the MULTI-ROW `hc_post_inplace`
    /// (`dsv41_hc_post_inplace_rows`). A stale `.so` reports false and the caller
    /// keeps the `hc_post` + copy pair, so the switch is free to make.
    pub fn supports_hc_post_inplace_rows(&self) -> bool {
        self.kernels.hc_post_inplace_rows.is_some()
    }

    /// The multi-row form of [`Self::hc_post_inplace`]: `res` is `[rows, n, h]`,
    /// `x` is `[rows, h]` and the coefficients are the m-row slices `post` =
    /// `[rows, n]`, `comb` = `[rows, n*n]`. Bit-identical to `hc_post` (with the
    /// same per-row strides) + the copy back — see the kernel header.
    ///
    /// `Ok(false)` when the `.so` has no such entry: the caller must keep the
    /// staging pair rather than launch a kernel that is not there.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_post_inplace_rows(
        &self,
        res: *mut f32,
        x: *const f32,
        post: *const f32,
        comb: *const f32,
        rows: i32,
        n: i32,
        h: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.hc_post_inplace_rows else {
            return Ok(false);
        };
        let rc = unsafe { f(res, x, post, comb, rows, n, h, self.stream) };
        self.kerr(rc, "dsv41_hc_post_inplace_rows")?;
        Ok(true)
    }

    /// Fused segment B cluster 1: collapse the hyper-connection rows and
    /// normalise, writing `out` directly (the intermediate `x` is not needed).
    ///
    /// `truncate` (`DSV41_BF16_TRUNCATE`, default OFF): round the collapsed row
    /// back to bf16 before the norm, reproducing the official `hc_pre`'s
    /// `y.to(x.dtype)` (`ref_inference/model.py:957-960`). The gate is read once
    /// by `chain_dev::bf16_truncate`.
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
        truncate: bool,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.hc_collapse_norm)(
                x, pre, w, out, rows, hc, dim, eps, truncate as c_int, self.stream,
            )
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
        truncate: bool,
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
                truncate as c_int,
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
        truncate: bool,
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
                truncate as c_int,
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
        truncate: bool,
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
                truncate as c_int,
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
        truncate: bool,
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
                truncate as c_int,
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

/// Activation-row bound of the multi-row v2 GEMV (`DSV41_GATE_MROWS` /
/// `ferrite_gemv_bf16_v2_mrows`). The C entry declines above it
/// (`cudaErrorInvalidValue`), so the Rust wrapper refuses the fold first and
/// keeps the per-row loop — a 9-row fold would be a shape bug, not a slow path.
/// The gate's production shapes (m = 5/6 verify rows) are well inside it, and
/// `VERIFY_ROWS` is the buffer bound the fold reads `s.xn_r` within.
const GEMV_V2_MROWS_MAX: i32 = 8;

/// `FERRITE_GEMV_SKIP=1` is a timing-only ablation INSIDE `ferrite_gemv_bf16_nt`
/// (its entry returns `cudaSuccess` without launching). The row fold routes the
/// MoE gate through that entry, so it must not engage while the ablation is on
/// (the per-row `gemv_bf16` path has no such skip). Read once, like the other
/// hot-path gates.
fn gemv_bf16_nt_skip() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("FERRITE_GEMV_SKIP").is_ok())
}

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
