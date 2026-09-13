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

/// Which A0 probe bucket a v5 all-reduce reports itself in — the Rust side of
/// `ferrite_p2p_ar_v5{,_attn,_moe}` (ferrite_kernels.cu, see the A0 note there).
///
/// The label is PURELY diagnostic: it selects a counter slot inside
/// `ar5_probe_report` for the `[ar-probe]` line. Nothing about the store, the
/// polls, the epoch advance or the reduce depends on it, which is exactly why a
/// caller may select a bucket without a numerical gate. `Other` is the plain
/// entry — the lumped bucket everything the split cannot tell apart lands in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArV5Site {
    Other,
    Attn,
    Moe,
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
    /// M-ROW SH_PAIR: the M=1 symbol above with the verify block's `m` activation
    /// rows folded into ONE launch (`gemm_fp8_sh_exp_pair_kernel<M>`). Phase 1
    /// walks the (i-tile, activation-row) split of the grid -- `fold_r` activation
    /// rows per block, 1 by default -- and phase 2 carries `M` accumulators over
    /// the same w2 row, so the whole block costs 2 launches per layer instead of
    /// `m`+2.
    ///
    /// Bit-identity to the per-row chain is the kernel's C1-C8 contract; the
    /// bodies are the M=1 kernel's, transcribed term for term.
    ///
    /// `aq`/`aqsc` is phase 1's fp8 output and MUST be disjoint from `a`/`a_scale`
    /// (same cross-block race as the M=1 symbol). `act` may be NULL (it is a
    /// by-product in this arm; a null pointer drops the stores). `epi_add != 0`
    /// folds `ferrite_add(out, phase-2)` into phase 2's store as a
    /// read-modify-write of `out`.
    ///
    /// A separate symbol, so a stale `.so` simply has no entry and the caller
    /// keeps its per-row loop. Returns 2 when the shape/arm cannot use it (never
    /// 1 -- cudaErrorInvalidValue, the round-42 collision).
    /// ABI: stream LAST.
    gemm_fp8_sh_exp_fused: Option<
        unsafe extern "C" fn(
            // phase 1: fp8 activation (k = dim) + scale, w1/w3 weights + scales,
            // limit, n1/k1, the activation-row count and the fold knob
            *const u8, *const f32, *const u8, *const u8, *const u8, *const u8, f32, c_int, c_int,
            c_int, c_int,
            // phase-1 by-product f32 rows (nullable) + stride
            *mut f32, c_int,
            // phase 1 output: fp8 pair + per-32-block scales + their strides
            *mut u8, *mut f32, c_int, c_int,
            // phase 2: w2 weights + scale, row count, out stride, epi_add, out
            *const u8, *const u8, c_int, c_int, c_int, *mut f32,
            CuStream,
        ) -> c_int,
    >,
    // TILELANG shared expert (`dsv41_sh_exp_tilelang`, from
    // `kernels/cuda/tilelang_gen/sh_exp_shim.cu`, generated by
    // `kernels/tilelang/gen_sh_exp_aot.py`; design `docs/agent/c5-sh-exp-tilelang-design.md`):
    // the whole shared-expert chain (w1w3 -> swiglu -> fp8 emit -> w2) as TWO TileLang
    // fp8 MMA launches, so the M activation rows SHARE one weight read (the SIMT arm
    // `dsv41_gemm_fp8_sh_exp_fused` re-reads w1/w3 m times and serialises its two
    // phases with a grid barrier).
    //
    // ABI: the SAME list as `gemm_fp8_sh_exp_fused` PLUS one `w2sc_pitch` int (the
    // generated kernel bakes w2.scale's UNPADDED 9 B row pitch, so the shim must see
    // the real pitch at run time — a mismatch is a silent wrong-row read, the
    // `tcgen05-rank7-verdict.md §10` class of failure). `fold_r` is an ABI placeholder
    // (the M fold is built into the mma tile, independent of m); `act` must be null.
    //
    // (b') CONTRACT — DOUBLE SWAP: the generated program puts M inside the mma tile, so
    // m=1 and m<=8 run the SAME program and row r of an M-row launch equals row r of the
    // M=1 launch OF THIS PROGRAM. It is NOT bit-identical to the SIMT arm, so
    // `DSV41_SH_EXP_TILELANG` must be taken on BOTH sides (eager and verify) or neither —
    // one-sided is exactly the `DSV41_PROJ_MMA` death.
    sh_exp_tilelang: Option<
        unsafe extern "C" fn(
            // L1: fp8 activation (k = dim) + scale, w1/w3 + scales, limit, n1/k1/m
            *const u8, *const f32, *const u8, *const u8, *const u8, *const u8, f32, c_int, c_int,
            c_int,
            // ABI placeholder + the L1 by-product slot (must be null / 0)
            c_int, *mut f32, c_int,
            // L1 output (L2 input): fp8 pair + per-32 scales + their row pitches
            *mut u8, *mut f32, c_int, c_int,
            // L2: w2 + scale, n2, the UNPADDED w2.scale row pitch, out stride, epi_add, out
            *const u8, *const u8, c_int, c_int, c_int, c_int, *mut f32,
            CuStream,
        ) -> c_int,
    >,
    // The two PHASES of the same program as separate entries (the shim stores them as
    // `dsv41_sh_exp_tilelang_gu` / `_dn`): 户部's L1/L2 microbench and a Rust-side
    // per-phase decline attribution. Same frozen geometry, same rc contract.
    sh_exp_tilelang_gu: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *const u8, *const u8, *const u8, *const u8, f32, c_int, c_int,
            c_int, *mut u8, *mut f32, c_int, c_int, CuStream,
        ) -> c_int,
    >,
    sh_exp_tilelang_dn: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *const u8, *const u8, c_int, c_int, c_int, c_int, c_int, c_int,
            c_int, *mut f32, c_int, CuStream,
        ) -> c_int,
    >,
    quant_fp8: unsafe extern "C" fn(
        *const f32, *mut u8, *mut f32, c_int, c_int, c_int, c_int, CuStream,
    ) -> c_int,
    // F7 (`dsv41_quant_fp8_stride`, from dsv41_kernels.cu): the STRIDED spelling of
    // the quantiser above — the same `quant_kernel<0>`, with the SOURCE row pitch
    // passed in instead of being derived from `cols`.
    //
    // WHY IT EXISTS. Every multi-row call site in the verify feeds a source whose
    // rows are NOT `cols` apart (`o_r` is `nh*hd` per row while the rank's slice is
    // `nlh*hd`; `wo_r` is `ol_total` while the contraction is `ol_local`), so the
    // block call had to be spelled as a per-row LOOP of `m` single-row launches —
    // the "verify-value-hunt" row-stride fix, which is correct but costs `m`
    // launches per site per layer. With the pitch explicit, each loop collapses to
    // ONE launch while every row keeps the bytes its own `rows = 1` launch wrote
    // (`r` only picks base pointers; the row's block amax is its own reduction).
    //
    // `src_stride == 0` is the legacy "== cols" spelling, and the C entry rejects
    // `0 < src_stride < cols` (overlapping rows) instead of reading row 0's tail.
    //
    // Optional: a stale .so without the symbol keeps the per-row loop verbatim,
    // which is the bit-exact reference the fold is judged against.
    // ABI: (x, y, scale, rows, cols, block, round_scale, src_stride, s).
    quant_fp8_stride: Option<
        unsafe extern "C" fn(
            *const f32, *mut u8, *mut f32, c_int, c_int, c_int, c_int, c_int, CuStream,
        ) -> c_int,
    >,
    quant_fp4: unsafe extern "C" fn(
        *const f32, *mut u8, *mut f32, c_int, c_int, c_int, c_int, CuStream,
    ) -> c_int,
    // A2 (`DSV41_WINDOW_KV_QUANT`): the sliding-window KV's IN-PLACE fp8 e4m3
    // round trip -- the reference's `act_quant(..., inplace=True)` on the
    // post-RoPE row (`ref_inference/model.py:705`). Quantise the row in blocks
    // of `block` and write the DEQUANTISED value back, so the ring stores the
    // f32 number the official stores. The element type does not change.
    //
    // Return codes: 0 = ran; 1 = the SHAPE was refused (the row is untouched);
    // 2 = no round trip applies. Optional: a stale `.so` without the symbol
    // keeps the row unquantised, announced by a one-shot notice -- an A/B arm of
    // "WINDOW_KV_QUANT" under that state would measure the OLD path.
    win_kv_quant_rt: Option<
        unsafe extern "C" fn(*mut f32, c_int, c_int, *mut f32, CuStream) -> c_int,
    >,
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
    // ENGRAM-GATHER-MROWS, from dsv41_kernels.cu (`dsv41_engram_gather_rows`):
    // the `rows`-fold of `engram_gather` above, for the verify's multi-row
    // engram write-back (`chain_dev.rs::engram_apply_rows` issued ONE gather per
    // row). `id_stride` is the per-TOKEN pitch of `hash_ids` in elements —
    // `eng_ids_r` is `[row][engram layer][n_cols]`, so one engram layer's rows
    // are `n_eng * n_cols` apart; the single-row entry passes `n_cols`, i.e. it
    // keeps the contiguous historical addressing exactly.
    //
    // Optional (a stale .so keeps the per-row loop, the bit-exact reference).
    // ABI: (table, table_scale, hash_ids, out, rows, n_cols, head_dim, id_stride,
    //       part_start, part_rows, s) -> 0 launched, 2 DECLINED.
    engram_gather_rows: Option<
        unsafe extern "C" fn(
            *const u8, *const u8, *const i64, *mut f32,
            c_int, c_int, c_int, c_int, i64, i64, CuStream,
        ) -> c_int,
    >,
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
    // W2-MROWS-TP (`DSV41_ATTN_MROWS` at `world > 1`): the same two launches with
    // an explicit `q`/`out` ROW PITCH in elements. Add-symbol-no-ABI-change: the
    // symbols above are untouched (they forward pitch 0), so an `.so` that predates
    // the TP arm still loads and behaves exactly as before — these are `Option`,
    // and their absence simply declines the `world > 1` batch (the old behaviour).
    // `xq`/`xsc` are NOT affected by the pitch (see the C launchers).
    sparse_attn_rp: Option<unsafe extern "C" fn(
        *const f32, *const f32, *const f32, *const i32, *mut f32,
        c_int, c_int, c_int, c_int, *const c_int, c_int, c_int, f32,
        *const c_int, c_int, c_int, CuStream,
    ) -> c_int>,
    sparse_attn_orope_rp: Option<unsafe extern "C" fn(
        *const f32, *const f32, *const f32, *const i32, *mut f32,
        c_int, c_int, c_int, c_int, *const c_int, c_int, c_int, f32,
        *const f32, *const f32, *const c_int, c_int, c_int, c_int, c_int, c_int, c_int,
        *mut u8, *mut f32,
        *const c_int, c_int, c_int, c_int, CuStream,
    ) -> c_int>,
    indexer_topk: unsafe extern "C" fn(
        *const f32, *const f32, *const f32, *const u8, *const i32, *mut i32,
        c_int, c_int, c_int, c_int, c_int, c_int, c_int, f32, f32, c_int, CuStream,
    ) -> c_int,
    // F8 (`dsv41_indexer_topk_rows`, from dsv41_kernels.cu): the indexer SELECTION
    // for a whole block in ONE launch — `b = 1, m = rows`, a PER-ROW `compress_lens`
    // array and an explicit output row pitch.
    //
    // WHY IT EXISTS. The grid has always been `dim3(m, b)` and each row of it is an
    // independent selection, so `m` per-row calls carry no information an `m`-row
    // call lacks — but the old entry could not EXPRESS two things a `b*m > 1` call
    // needs (and `chain_dev.rs`'s `indexer_rows_m` header spells out the same three
    // blockers): the launch ceiling had to be the MAXIMUM of the per-row counts
    // (the kernels took `*lens`, i.e. row 0's, and the verify's counts ascend with
    // `r`, so rows 1..m-1 lost their newest groups), and the output row pitch had to
    // be the caller's (`idxs_r`: `offset + index_topk`) rather than the kernel's
    // `cols = min(topk, cl)`, a RUNTIME value. Both are now explicit and both are
    // no-ops at `b*m == 1`, which is why `dsv41_indexer_topk` is untouched.
    //
    // The folded call site (`indexer_rows_one` at `m > 1`, hoisted to
    // `indexer_select_rows`) is valid only where the per-row attention is NOT taken
    // (nothing reads `idxs_r` inside the block loop) and every row's bound is a
    // device snapshot (`clen_rows_r`) — see the Rust-side gate.
    //
    // Optional: a stale .so without the symbol keeps the per-row `b = m = 1` call,
    // which is the bit-exact reference (each row's launch already carried its own
    // `*lens`).
    // ABI: (q, index_k, weights, candidates, compress_lens, out, b, m, nh, hd, n_pos,
    //       topk, offset, out_stride, softmax_scale, head_scale, uses_candidates, s).
    indexer_topk_rows: Option<
        unsafe extern "C" fn(
            *const f32, *const f32, *const f32, *const u8, *const i32, *mut i32,
            c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, f32, f32, c_int, CuStream,
        ) -> c_int,
    >,
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
    /// L4-9: the dim-split twin of [`Self::rmsnorm_rows`]
    /// (`dsv41_rmsnorm_rows_split_kernel`), `DSV41_NORM_SPLIT` (default OFF).
    /// `grid = (nchunks, rows)` — the `dim` axis is cut into `nchunks`
    /// contiguous segments so a `rows = 1` call spreads over `nchunks` SMs
    /// instead of one. Optional: an `.so` without the symbol, or a gate that is
    /// off, keeps the original `dsv41_rmsnorm_rows` launch unchanged.
    rmsnorm_rows_split: Option<unsafe extern "C" fn(
        *const f32, *const f32, *mut f32, c_int, c_int, f32, c_int, CuStream,
    ) -> c_int>,
    // B4 (`DSV41_RMSNORM_ROPE_MROWS`, default OFF, from dsv41_kernels.cu): the
    // verify block's kv half — `norm_rows(kv_r)` + `apply_rope(kv_r)` — in ONE
    // launch. Phase 1 is `dsv41_rmsnorm_rows_kernel`'s body verbatim (same 1024
    // threads per row, so the cross-warp fold is the same sum in the same
    // order); phase 2 is `apply_rope_kernel`'s trailing-`2*half` rotation at
    // `pos_rows[r]` (== `pos_base + r`, the position the `off = 0, step = 1`
    // form computed). The only addition is a barrier between the phases, which
    // orders memory and moves no value ⇒ bit-identical to the two launches it
    // replaces. It takes the STREAM explicitly: the kv half is the
    // `DSV41_VERIFY_FORK` side chain, and hard-coding the main stream would drag
    // it back off the side stream.
    //
    // Optional: a stale .so without the symbol keeps `norm_rows + apply_rope`,
    // whose numerics are the reference. The C entry returns 2 (declined) for a
    // shape outside its domain (rope region past the row, non-positive sizes);
    // this wrapper pre-checks the same set and returns Ok(false).
    // ABI: (x, w, out, rows, dim, eps, cos, sin, rope_off, half, pos_rows,
    //       inverse, s).
    rmsnorm_rope_mrows: Option<unsafe extern "C" fn(
        *const f32, *const f32, *mut f32, c_int, c_int, f32, *const f32, *const f32, c_int, c_int,
        *const c_int, c_int, CuStream,
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
    // D2 EPOCH PAD (`DSV41_SWALLOW_EPOCH_PAD`), `dsv41_v5_epoch_pad_kernel`:
    // advance this rank's v5 epoch by `pad` EMPTY rounds in ONE launch, stamping
    // this rank's slot in every peer's ready row (and the A4 broadcast word) with
    // the final value `e + pad`, so the swallowed arm's per-step round footprint
    // equals the legacy/aligned arm's. Optional: an .so predating the symbol
    // keeps the swallowed arm's smaller footprint and the pad is skipped — the
    // caller reports that once rather than padding silently. See
    // docs/agent/swallow-fix9-round-ledger-design.md §3.2.
    v5_epoch_pad: Option<unsafe extern "C" fn(
        *const *mut u32, *mut c_uint, c_int, c_int, c_uint, CuStream,
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
    /// `ferrite_add_store` (A1a): the same elementwise add, whose epilogue ALSO
    /// copies `z` into every peer's staging slot — i.e. the launch carries the
    /// following all-reduce's store. Used only where the standalone add is the
    /// payload's LAST writer (see `Device::add_inplace_ar`).
    add_inplace_ar: Option<
        unsafe extern "C" fn(
            *const f32,
            *const f32,
            *mut f32,
            c_int,
            *const *mut f32,
            *const c_uint,
            c_int,
            c_int,
            c_int,
            CuStream,
        ) -> c_int,
    >,
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
    // B5 (`DSV41_GATE_MROWS_ROUTE`, default OFF, from ferrite_kernels.cu): the
    // multi-row MoE gate GEMV (`ferrite_gemv_bf16_v2_mrows`) with the route
    // election folded in — ONE launch where the gate + `route_topk` used to be
    // two. It instantiates the SAME `gemv_bf16_nt_kernel<NT, WPR>` program, so
    // the GEMV half is bit-identical to `gemv_bf16_v2_mrows`; the elected block
    // then runs `dsv41_route_topk`'s body over the whole [rows, n_experts] score
    // block, row for row. `nrows == 1` forwards to `gemv_bf16_v2_route` (the
    // M=1/ROUTE_FUSE program), so the two entries agree at the boundary.
    //
    // Optional: a stale .so without the symbol keeps the two-launch pair (the
    // gate GEMV + `route_topk`), whose numerics are the reference. The C entry
    // declines (cudaErrorNotSupported) for `rows` outside 1..=8, `k % 8 != 0`
    // (v1 is a different accumulation order), `topk` outside [1, n], a null
    // route output, or route smem above 48 KB; this wrapper pre-checks the same
    // set and returns Ok(false) so the caller falls back.
    // ABI: (x, w, bias, out, in_f, out_f, nrows, weights, indices, route_bias,
    //       topk, norm_topk_prob, route_scale, score_func, ctr, s).
    gemv_bf16_v2_mrows_route: Option<
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
    // TILELANG head bf16 GEMM, from `kernels/cuda/tilelang_gen/head_bf16_shim.cu`
    // (`dsv41_head_bf16_tilelang`; prototype: docs/agent/tilelang-attn-head.md).
    // The K-split + M-pad-16 mma program of the head: M rides INSIDE the mma
    // tile, so m=1 and m=6 issue the same number of mma instructions — the
    // prototype measured M6/M1 = 1.00 against `gemv_bf16_v1_mrows`'s 3.23-3.81,
    // and 2.93x on the m=6 absolute (slice 125.2 -> 42.2us).
    //
    // ⚠️ A/B ARM, NOT A PRODUCTION REPLACEMENT (default OFF): the head's real
    // program is "bf16 weight x f32 activation" (f32 FMA), and tensor-core bf16
    // mma has no bf16 x f32 form, so the shim CASTS the activation to bf16 on
    // its host side. The head feeds an argmax, and that cast's ~2.3e-2 max_rel
    // (prototype §4) is far above the ~1e-3 near-tie flip threshold
    // (`draft-head-fold-v2-argmax-verdict.md`, 33% echo), so this arm OWES a
    // numeric debt that only the verify's acc gate (mean-k / Z_) can repay. It
    // exists to measure the M-in-tile win, not to be switched on in production.
    //
    // ABI is `dsv41_gemv_bf16_v1_mrows`'s verbatim — (w, x, out, m=rows, n, k, s)
    // — so the Rust arm is that entry's drop-in: `w` is the [n, k] bf16 head
    // weight (or its slice), `x` the [m, k] f32 activation, `out` the [m, n] f32
    // logits (its row stride MUST be n). SHAPE-SPECIFIC: the frozen geometry
    // (n=16160=kTLN, k=5120) is baked into the generated grid, so the C entry
    // DECLINES (returns 2, never cudaErrorInvalidValue) for every other shape
    // (n != 16160 / k != 5120 / m outside 1..=8 / a base that is not 16B-aligned
    // / an INIT failure / a call inside a capture before INIT has completed).
    // Optional: a stale .so without the symbol keeps the per-row loop.
    head_bf16_tilelang: Option<
        unsafe extern "C" fn(*const c_void, *const f32, *mut f32, c_int, c_int, c_int, CuStream) -> c_int,
    >,
    /// Tap bf16 round-trip (DSV41_TAP_BF16): in-place elementwise
    /// `x[i] = bf16_to_f32(f32_to_bf16(x[i]))` for the dspark draft's main_h
    /// buffer, aligning the MTP head's input with the official model's bf16
    /// hidden states. Optional so a stale `.so` simply declines.
    bf16_roundtrip: Option<unsafe extern "C" fn(*mut f32, i64, CuStream) -> c_int>,
    /// A3 (DSV41_COMPRESS_LATENT_QUANT): the compressed-KV latent's fp4 roundtrip
    /// (`dsv41_compress_ring_quant`, dsv41_glue.cu) — per-16-block amax, an E4M3
    /// (NOT power-of-two) scale, e2m1 codes, dequantised in place. Optional so a
    /// stale `.so` declines to the bf16-only boundary with one note instead of
    /// launching nothing at all. ABI: (ring, out_rows, clen, window, hd,
    /// bf16_first, row_quant, dbg, s).
    compress_ring_quant: Option<
        unsafe extern "C" fn(
            *mut f32,
            *const c_int,
            *const c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            *mut f32,
            CuStream,
        ) -> c_int,
    >,
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
    // PROJ-MMA (`dsv41_gemm_fp8_mrows_mma`, from the dedicated PROJ-MMA TU
    // `kernels/cuda/dsv41_proj_mma_skel.cu`): the TENSOR-CORE form of the
    // multi-row fp8 GEMV above — `gemm_fp8_mrows_mma_kernel<M>`, ONE
    // `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` retiring
    // 16x8x32 = 4096 MACs, with the WEIGHT on the MMA's M (16 output channels
    // per tile) and the ACTIVATION on its N (8 rows). The mapping is the M<=8
    // generalisation of `dsv41_gemm_fp8_swapab` (M=1 decode on the tensor core),
    // i.e. the same axis choice the tree already ships. See
    // docs/agent/tensorcore-proj-design.md §3 (implementation framework) §7
    // (merge timing).
    //
    // ⚠️ NUMERICS: this program is NOT bit-identical to `gemm_fp8_mrows` — the
    // k-block scale is applied to the 32-element SUM instead of per element, and
    // the tensor core's intra-block summation order is hardware-defined (§2 of
    // the design proves the SIMT parity impossible, not merely unproven). What
    // this program DOES guarantee, bit for bit, is
    //     row r of an M-row launch == row r of the M=1 launch of THIS program
    // because column r of the mma's D is a function of column r of B and A
    // alone (same program at every m ⇒ (b′) program-consistent parity). The arm
    // is therefore judged by the dual gate (step_ms AND mean-k), never by a byte
    // compare against the SIMT path.
    //
    // ROUND 1 (`ks == 1`) needs NO scratch, so the wrapper below passes
    // `partial`/`ctr` null: the C entry only proceeds when its resolved K-split
    // is 1 (any ks > 1 hits its own scratch guard and returns 2 = declined).
    // Consequence: the arm needs `DSV41_PROJ_MMA_KS=1` until the `[ks][M][n]`
    // partial scratch is sized on this side. `pmma_n` follows the launcher's
    // contract (the caller's promise that `partial` covers `n` rows) and takes
    // the tightest truthful value with no scratch: `n` itself.
    //
    // Optional: a stale .so without the symbol keeps the `gemm_fp8_mrows` path,
    // and the C entry returns 2 (declined, never cudaErrorInvalidValue) for `m`
    // outside 1..=8, `n` not a multiple of 16, `k` not a multiple of 32,
    // `out_stride < n`, a non-16B-aligned activation/weight base, or a K-split
    // that resolves above 1 with no scratch pointer.
    // ABI: (a, a_scale, w, w_scale, bias, out, m, n, k, out_stride, partial,
    //       ctr, pmma_n, s).
    gemm_fp8_mrows_mma: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *const u8, *const u8, *const f32, *mut f32,
            c_int, c_int, c_int, c_int, *mut f32, *mut c_uint, c_int, CuStream,
        ) -> c_int,
    >,
    // TILELANG (`dsv41_gemm_fp8_tilelang_wkv`, from the generated TU
    // `kernels/cuda/tilelang_gen/wkv_shim.cu`): the FIRST TileLang-generated
    // projection kernel — the wkv shape (n=512, k=5120) of the fp8 projection GEMM,
    // emitted by tilelang 0.1.14 and frozen into the tree (route (a) source
    // vendoring; see docs/agent/tilelang-integration-design.md and
    // tilelang_gen/PROVENANCE.md).
    //
    // WHY IT IS A PROGRAM SWAP AND NOT A KNOB, and why it is the ONLY projection
    // arm that is safe to compare across m: the TileLang program puts M INTO the
    // mma tile, so m=1 and m=6 issue the SAME number of mma instructions (the
    // prototype measured M6/M1 = 1.00 on all four shapes, against the SIMT mrows
    // family's 5.73/5.26/3.11/3.04x). It is NOT bit-identical to `gemm_fp8_mrows`
    // (the k-block scale lands on the 32-element SUM, the tensor core's intra-block
    // summation order is hardware-defined — §2 of tensorcore-proj-design.md proves
    // SIMT parity unreachable). What it DOES guarantee per the same argument is
    //     row r of an M-row launch == row r of the M=1 launch OF THIS PROGRAM
    // ⇒ the arm must be taken on BOTH sides (eager and verify) or on neither:
    // `DSV41_GEMM_TILELANG` is DEFINED as that double swap. Taking it on the verify
    // side alone is exactly the `DSV41_PROJ_MMA` death (mean-k 2.240 -> 0.020),
    // which is why `proj_mrows` checks this arm FIRST and the eager `lin` carries
    // the same arm.
    //
    // FIRST PHASE = ONE SHAPE. The generated geometry (n/k/bN/ks) is baked into the
    // grid and index arithmetic, so the C entry DECLINES (returns 2, never
    // cudaErrorInvalidValue) for every other shape: `m` outside 1..=8, `n != 512`,
    // `k != 5120`, `out_stride != n`, a non-null `bias` (the generated kernel has no
    // bias path), a base that is not 16B-aligned, or an INIT failure (the [ks][16][n]
    // partial scratch could not be allocated — e.g. the first call landed inside
    // capture). A decline leaves the caller on its existing kernel.
    // ABI: (a, a_scale, w, w_scale, bias, out, m, n, k, out_stride, s) — the SAME
    // list as `gemm_fp8_mrows`, so a caller swaps the program without touching its
    // activation staging.
    gemm_fp8_tilelang_wkv: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *const u8, *const u8, *const f32, *mut f32,
            c_int, c_int, c_int, c_int, CuStream,
        ) -> c_int,
    >,
    // TILELANG PHASE 2 — the remaining three DENSE projection shapes of the fp8
    // projection family, from `kernels/cuda/tilelang_gen/{wq_a,wq_b,wo_b}_shim.cu`
    // (generated by `kernels/tilelang/gen_proj_shapes_aot.py`, frozen dumps in
    // `tilelang_gen/*_tl.cu`; see PROVENANCE.md §8).
    //
    // Same program as `gemm_fp8_tilelang_wkv` (M rides INSIDE the mma tile, so
    // m=1 and m<=6 issue the same number of mma instructions), same double-swap
    // contract and same `DSV41_GEMM_TILELANG` gate — only the frozen geometry
    // differs. Each entry is SHAPE-SPECIFIC (the generated grid / index arithmetic
    // bakes n, k, k-split and the output row stride `OS`), so its C entry DECLINES
    // (returns 2, never cudaErrorInvalidValue) for every shape but its own; the
    // caller tries the shapes in turn and keeps its existing kernel on decline.
    //
    // ⚠️ `out_stride` (OS) is per shape, taken from the REAL call site, and it is
    // NOT always `n`:
    //   * wq_a: OS = q_lora_rank = n = 1280            (verify `proj_mrows` + eager `lin`)
    //   * wq_b: OS = nh*head_dim = 32768 ≠ n = 4096    (ColumnParallel: the rank writes
    //            the leading nlh*hd of a full nh*hd row). The indexer site
    //            (`idx_wq_b`, out_stride == n) therefore DECLINES here.
    //   * wo_b: OS = dim = n = 5120                    (verify `proj_mrows` + eager `lin`)
    // ABI = the SAME list as `gemm_fp8_tilelang_wkv` / `gemm_fp8_mrows`:
    // (a, a_scale, w, w_scale, bias, out, m, n, k, out_stride, s).
    gemm_fp8_tilelang_wq_a: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *const u8, *const u8, *const f32, *mut f32,
            c_int, c_int, c_int, c_int, CuStream,
        ) -> c_int,
    >,
    gemm_fp8_tilelang_wq_b: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *const u8, *const u8, *const f32, *mut f32,
            c_int, c_int, c_int, c_int, CuStream,
        ) -> c_int,
    >,
    gemm_fp8_tilelang_wo_b: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *const u8, *const u8, *const f32, *mut f32,
            c_int, c_int, c_int, c_int, CuStream,
        ) -> c_int,
    >,
    // TILELANG PHASE 2 — the GROUPED wo_a shape, from
    // `kernels/cuda/tilelang_gen/wo_a_shim.cu`
    // (`dsv41_gemm_fp8_tilelang_wo_a`). It is the TileLang form of
    // `wo_a_grouped_fp8` (the same call site's arguments), so its ABI is that
    // entry's, NOT the dense one: the group dimension rides grid.y and the two
    // compiled variants are (G=1, a_stride=4096) — verify@TP8, where a_stride
    // degenerates to k — and (G=8, a_stride=32768) — the TP1 form. The C entry
    // picks the variant on (groups, a_stride) and DECLINES (2) on anything else.
    //   * out_stride (OS) = ol_total = groups*o_lora_rank = 8192, NOT G*n: wo_a is
    //     ColumnParallel, so the rank's nlg groups land at column offset g*n inside
    //     a full-width row.
    //   * quantisation is OUTSIDE the shim (the call site already hands it fp8 +
    //     per-32 f32 scale), which is what makes this the lowest-integration-cost
    //     shape (tilelang-proj-phase2.md §6.1).
    //   * `bias` is null at this site (`:8177`); a non-null bias declines.
    // ABI: (a, a_scale, w, w_scale, bias, out, groups, rows, n, k, a_stride,
    //       out_stride, s) — the SAME list as `wo_a_grouped_fp8`.
    gemm_fp8_tilelang_wo_a: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *const u8, *const u8, *const f32, *mut f32,
            c_int, c_int, c_int, c_int, c_int, c_int, CuStream,
        ) -> c_int,
    >,
    // K2, from dsv41_kernels.cu (`dsv41_gemm_fp8_mrows_rope_norm`): the verify's
    // q-chain tail folded into ONE launch — rmsnorm + fp8 quantisation + the
    // wq_b multi-row fp8 GEMV + RoPE, each segment reproduced from the kernel the
    // verify path runs TODAY (`dsv41_rmsnorm_rows_kernel`, `quant_kernel<0>`,
    // `gemm_fp8_mrows_kernel<M>`, `apply_rope_mrows_kernel`), so the fused launch
    // is the same program as the four-launch sequence it replaces.
    //
    // It is the K2 replacement for R2's `lin_rope_norm`: NO EAGER kernel program
    // is involved (R2 reused the `gemm_fp8_gemv_kernel` family, a different
    // program than the `mrows` chain) and the rope position is an explicit
    // `pos_rows[r]` DEVICE ARRAY — the kernel contains no `pos_ctr`, no
    // `mul`/`off`/`step` (R2's second difference).
    //
    // `qr_norm_out` is where the normalised rows land; the shipped caller passes
    // `s.qr_r` (in place), which is what keeps the indexer's q half reading the
    // same bytes it reads today with no RAW flag and no compensating launch.
    // `nullptr` is legal (the row stays raw and the caller owes the norm).
    //
    // Optional: a stale `.so` without the symbol keeps the four-launch sequence,
    // and the C entry returns 2 (declined, never cudaErrorInvalidValue) for `m`
    // outside 1..=8, a `k`/`n`/`rope_hd` that is not a multiple of 32, an odd or
    // out-of-range `rope_rd`, `out_stride < n`, or a run configured for the
    // reordering `DSV41_GEMV_FP8_MODE` 0/1 arms.
    // ABI: (qr_raw, qr_w, qr_eps, qr_norm_out, w, w_scale, bias, out, m, n, k,
    //       out_stride, cos, sin, pos_rows, rope_rd, rope_hd, inverse, s).
    gemm_fp8_mrows_rope_norm: Option<
        unsafe extern "C" fn(
            *const f32, *const f32, f32, *mut f32, *const u8, *const u8, *const f32, *mut f32,
            c_int, c_int, c_int, c_int, *const f32, *const f32, *const c_int, c_int, c_int,
            c_int, CuStream,
        ) -> c_int,
    >,
    // K1 (`dsv41_gemm_fp8_mrows2`, from dsv41_kernels.cu): the TWO-FAMILY form of
    // the multi-row fp8 GEMV above -- the verify's wq_a + wkv pair in ONE launch,
    // since both project the SAME quantised activation row with the same `k`.
    // The warp's output row selects its family (`row < n1`), so the two
    // projections share one grid, one block prologue and one graph node; the
    // code is `gemm_fp8_mrows_kernel<M>` with four `fam1 ? x1 : x2` base
    // selections, so row `row` is bit-identical to the same row of the two
    // separate `dsv41_gemm_fp8_mrows` launches it replaces (the kernel header
    // carries C1-C6 and the fuse argument).
    //
    // TWO output strides (one per family), not one: the wq_a row strides by `ql`
    // while the wkv row strides by `hd`, and those differ (1280 vs 512). That is
    // the single ABI deviation from the design doc's §3.2 signature, which
    // carries one `out_stride`.
    //
    // Optional: a stale .so without the symbol keeps the two `gemm_fp8_mrows`
    // launches, and the C entry returns 2 (declined, never
    // cudaErrorInvalidValue) for `m` outside 1..=8, `k` not a multiple of 32,
    // `out_stride_f < n_f`, a null pointer, or a run configured for the
    // reordering `DSV41_GEMV_FP8_MODE` 0/1 arms / `DSV41_NO_GEMV_FP8`.
    // ABI: (a, a_scale, w1, w1_scale, bias1, out1, n1, out_stride1,
    //       w2, w2_scale, bias2, out2, n2, out_stride2, m, k, s).
    gemm_fp8_mrows2: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *const u8, *const u8, *const f32, *mut f32,
            c_int, c_int, *const u8, *const u8, *const f32, *mut f32, c_int, c_int,
            c_int, c_int, CuStream,
        ) -> c_int,
    >,
    // B6 (`dsv41_gemm_fp8_mrows_f32`, from dsv41_kernels.cu): the MULTI-ROW form
    // of the f32-activation GEMV (`dsv41_gemm_fp8_mx_f32`) — fp8 e4m3 weights x
    // RAW f32 activations. It folds the verify's wo_b from `m x quant_fp8 +
    // 1 x proj_mrows` into ONE launch per layer, and it is the same program as
    // the M=1 decode of the same (row, r): row r of this launch is BIT-IDENTICAL
    // to the M=1 `dsv41_gemm_fp8_mx_f32` call of row r (EAGER's `DSV41_WOB_F32`
    // path), since the f32 domain's "materialisation" is the identity
    // (`s_af[i] = a_f32[i]`, a pure copy).
    //
    // NOT bit-identical to the OLD verify path (the `quant_fp8 + proj_mrows`
    // pair): it skips the quantise -> dequantise round trip, so the activation
    // keeps its full f32 mantissa and the row partials are slightly MORE
    // accurate. That is the `DSV41_WOB_F32` / E8 lever, and it is why the
    // acceptance is the red line (counting order + zero Latin) rather than a
    // memcmp against the old bytes.
    //
    // `a_stride` — the activation row PITCH in f32 elements — is an explicit
    // parameter, not a caller promise: verify's `wo_r` is [m, ol_total] while
    // k = ol_local (8x apart under TP8), so the pitch can NOT be derived from
    // `k`. The C entry declines `a_stride < k` instead of reading row 0's tail
    // (the verify-value-hunt root cause F1/F2).
    //
    // Optional: a stale .so without the symbol keeps the
    // `quant_fp8 + proj_mrows` pair, and the C entry returns 2 (declined, never
    // cudaErrorInvalidValue) for `m` outside 1..=8, `k` not a multiple of 32,
    // `a_stride < k`, `out_stride < n`, a null pointer, or a run configured for
    // the reordering `DSV41_GEMV_FP8_MODE` 0/1 arms / `DSV41_NO_GEMV_FP8`.
    // ABI: (a_f32, w, w_scale, bias, out, m, n, k, a_stride, out_stride, s).
    gemm_fp8_mrows_f32: Option<
        unsafe extern "C" fn(
            *const f32, *const u8, *const u8, *const f32, *mut f32,
            c_int, c_int, c_int, c_int, c_int, CuStream,
        ) -> c_int,
    >,
    // WO_PAIR-ROWS (`dsv41_gemm_fp8_mrows_q_f32`, from dsv41_kernels.cu): the
    // verify block's `wo_a_grouped -> quant -> wo_b` chain as TWO launches, with
    // the INTERMEDIATE QUANT folded into the second one. The kernel is
    // `gemm_fp8_mtile_kernel<M, 1>` -- the just-delivered MTILE program with the
    // activation operand re-derived in-warp (`quant_kernel<0>`'s arithmetic for
    // this warp's own 32-element scale block) instead of read from the quantiser's
    // global output.
    //
    // Numerics: `out` is BIT-IDENTICAL to `wo_a_grouped_gemv_kernel<M>` +
    // `m x quant_kernel<0>` + `gemm_fp8_mtile_kernel<M, 0>` (see the C entry's
    // header: the quant is `quant_kernel<0>` term for term, the fold is the
    // shipped MTILE program). Unlike B6
    // ([`Self::gemm_fp8_mrows_f32`]), this arm CAN be judged by a byte comparison.
    // What it saves is the `m` per-row quant launches the f32-source pitch forced
    // (the `quant_fp8` ABI has no `a_stride`).
    //
    // Neither the fp8 nor its scales are materialised (the byte lives in a
    // register), which is safe exactly where B6 is safe: within `attention_rows`
    // the last reader of `xq_r`/`xsc_r` IS the quant this replaces, and the next
    // reader (`indexer_front_rows`) re-quantises first.
    //
    // Optional: a stale .so without the symbol keeps the `quant_fp8 + proj_mrows`
    // pair, and the C entry returns 2 (declined, never cudaErrorInvalidValue) for
    // `m` outside 1..=8, `k` not a multiple of 32 (the quant's scale block), a
    // null pointer, `a_stride < k`, `out_stride < n`, an smem over the device
    // ceiling, or a run configured for the reordering `DSV41_GEMV_FP8_MODE` 0/1
    // arms / `DSV41_NO_GEMV_FP8`.
    // ABI: (a_f32, a_stride, w, w_scale, bias, out, m, n, k, out_stride, s).
    gemm_fp8_mrows_q_f32: Option<
        unsafe extern "C" fn(
            *const f32, c_int, *const u8, *const u8, *const f32, *mut f32,
            c_int, c_int, c_int, c_int, CuStream,
        ) -> c_int,
    >,
    // v2 (vectorized float4 + K-split) f32 M=1 GEMV, from dsv41_glue.cu.
    // Optional: an older .so without the symbol keeps the v1 kernel above.
    // Same ABI as v1: (w, x, out, n=out_f, k=in_f, s).
    gemv_f32_v2: Option<unsafe extern "C" fn(*const f32, *const f32, *mut f32, c_int, c_int, CuStream) -> c_int>,
    // COMPRESSOR-PROJ-MROWS, from dsv41_kernels.cu (`dsv41_gemv_f32_mrows`): the
    // MULTI-ROW form of the v2 f32 GEMV above — `m` activation rows folded into
    // ONE pass over the weight, one INDEPENDENT accumulator per row. It is a
    // transcription of `gemv_f32_v2_kernel<WPR>` (same `kper` slice arithmetic,
    // same `float4` walk, same `__fmaf_rn` chain, same `__shfl_down_sync` tree,
    // same cross-slice fold), and the ONLY change the fold makes is hoisting the
    // weight `float4` out of the row loop — the identical bytes, reused, which
    // cannot change any row's value. So each row is bit-identical to that row's
    // own `m = 1` launch, which is what makes this the fix for the verify's
    // per-row compressor projections (`chain_dev.rs::compress_proj_rows`: two
    // `lin_f32_on` calls per row per compress-source layer at m = 6) rather than
    // a different program.
    //
    // The entry DECLINES (returns 2, never cudaErrorInvalidValue) when the
    // per-row reference would not be v2 (that gate off, `n >= 2048`), when
    // `m` is outside 1..=8, when `k % 4 != 0` or a pointer is null.
    //
    // Optional: a stale .so without the symbol keeps the per-row loop, which is
    // the bit-exact reference the kernel was transcribed from.
    // ABI: (w, x, out, m, n, k, s).
    gemv_f32_mrows: Option<
        unsafe extern "C" fn(*const f32, *const f32, *mut f32, c_int, c_int, c_int, CuStream) -> c_int,
    >,
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
    // RING-WIN-MROWS (`DSV41_RING_WIN_MROWS=1`): the whole verify block's append +
    // per-row causal window indices in ONE launch - the rows form of
    // `ring_win_fuse` with an explicit `idxs` row stride (`ist = window +
    // index_topk`, so each row's bytes land where the per-row call wrote them).
    // Caller enforces `*pos_ctr + m - 1 < window` (no ring turnover). Optional:
    // an older .so without the symbol keeps the per-row path.
    ring_win_fuse_mrows: Option<unsafe extern "C" fn(
        *mut f32,
        *const f32,
        *const c_int,
        c_int,
        c_int,
        c_int,
        *mut i32,
        c_int,
        CuStream,
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
    // SPARSE-ATTN-ROPE-MROWS (`DSV41_VERIFY_OROPE_MROWS=1`): the verify block's
    // ONE fused sparse-attention + inverse o-rope + fp8 launch, valid in the
    // STEADY STATE (the `kv_rows` substitution reads the block's own KV rows for
    // every window slot whose position falls inside the block, so the ring's
    // turns over do not matter). The caller MUST follow it with
    // `ring_append_mrows`. Optional: a stale .so leaves the caller on the per-row
    // `sparse_attn_orope` sequence.
    sparse_attn_orope_mrows: Option<unsafe extern "C" fn(
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *const i32,
        *mut f32,
        c_int,
        c_int,
        c_int,
        c_int,
        *const c_int,
        c_int,
        c_int,
        f32,
        *const f32,
        *const f32,
        *const c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        *mut u8,
        *mut f32,
        *const c_int,
        c_int,
        c_int,
        c_int,
        CuStream,
    ) -> c_int>,
    // SPARSE-ATTN-ROPE-MROWS support: the block's window indices, ONE launch for
    // all m rows (`window_idxs`'s decode branch per row, `idxs` rows pitched at
    // `idx_stride = window + index_topk`). Optional: falls back to the per-row
    // `window_idxs` / `ring_win_fuse` calls.
    window_idxs_mrows:
        Option<unsafe extern "C" fn(*mut i32, *const c_int, c_int, c_int, c_int, CuStream) -> c_int>,
    // SPARSE-ATTN-ROPE-MROWS support: the block's ring APPEND, run AFTER the
    // fused attention (the deferral is what makes the batch correct). Optional:
    // falls back to the per-row `ring_append` / `ring_win_fuse` calls.
    ring_append_mrows: Option<
        unsafe extern "C" fn(*mut f32, *const f32, *const c_int, c_int, c_int, c_int, CuStream) -> c_int,
    >,
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
    // F10 (`dsv41_compress_commit_rows`, from dsv41_glue.cu): the commit launch
    // with the row's COUNTER SNAPSHOT written in-kernel.
    //
    // WHY IT EXISTS. The verify's block-wide attention arms hand the m-row launch a
    // per-row bound (`clen_rows_r[owner][r]` = the counter after row r's own
    // commit) while the compressor ran PER ROW, so row r's copy has to be taken
    // between the two. Reading the live counter is the only device-side source, and
    // it is a SCALAR that advances: one `memcpy_d2d` per row (6 launches + 6 graph
    // nodes per layer) is the minimum a caller-side copy can do, because a single
    // D2D of `m * 4` bytes cannot carry `m` DISTINCT values. Taking the snapshot
    // inside the commit instead costs nothing — it is one store on the thread that
    // already bumped the counter — and the values are the same row for row,
    // including rows that complete no group (the counter is simply unchanged).
    //
    // Optional: a stale .so without the symbol keeps the per-row `memcpy_d2d` loop,
    // which is the bit-exact reference. The entry is a separate symbol (rather than a
    // trailing default parameter on `dsv41_compress_commit`) precisely because a
    // stale symbol would accept an extra argument, ignore it, and leave
    // `clen_rows_r` holding the PREVIOUS step's counters — silently wrong bounds for
    // a whole block.
    // ABI: (latent, cos, sin, ring, out_rows, clen, clen_row_out, hd, rope_dim,
    //       half, window, ratio, s).
    compress_commit_rows: Option<
        unsafe extern "C" fn(
            *const f32,
            *const f32,
            *const f32,
            *mut f32,
            *const c_int,
            *mut c_int,
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
            *const c_int, c_int, c_int, c_int, CuStream,
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
            *const u8, i64, *const u8, i64, *const c_int, c_int, CuStream,
        ) -> c_int,
    >,
    /// TILELANG MoE grouped GEMM — up (gate‖up) arm (`DSV41_MOE_TILELANG`, default
    /// OFF). Replaces the routed gate/up launch with the TileLang bf16 grouped
    /// MMA (`kernels/cuda/tilelang_gen/moe_bf16_shim.cu`): one dense operand block
    /// per expert segment instead of a per-(row, slot) GEMV sweep, GPU-measured at
    /// 45.5µs/layer vs the 250µs SIMT baseline (docs/agent/tilelang-moe-grouped.md
    /// §3.3). It consumes the **bf16 copy** of the expert weights that the load-time
    /// dequant (`DSV41_MOE_BF16_DEQUANT` / [`Self::moe_fp4_to_bf16`]) produces, the
    /// host-computed moe_align tables and the f32 activations. See the shim's header
    /// for the full ABI/rc contract (`0` fired / `2` DECLINED / else cuda error).
    moe_tilelang_gate_up_bf16: Option<
        unsafe extern "C" fn(
            *const f32,
            *mut f32,
            *const c_void,
            *const c_int,
            *const c_int,
            *const c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            CuStream,
        ) -> c_int,
    >,
    /// TILELANG MoE grouped GEMM — down arm (`DSV41_MOE_TILELANG`, the same runtime
    /// gate as the up arm). Writes the per-slot down partials
    /// (`24.5µs/layer`, grouped), which the existing `moe_down_reduce` still sums in
    /// the fixed ascending-slot order.
    moe_tilelang_down_bf16: Option<
        unsafe extern "C" fn(
            *const f32,
            *mut f32,
            *const c_void,
            *const c_int,
            *const c_int,
            *const c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            CuStream,
        ) -> c_int,
    >,
    /// Load-time fp4(e2m1 + ue8m0) -> bf16 dequant (`DSV41_MOE_BF16_DEQUANT`,
    /// default OFF). The bf16 arm's PRECONDITION: the grouped kernel streams bf16
    /// weights, so the expert fp4 pool has to be expanded ONCE at load time (4x the
    /// fp4 bytes resident — the memory decision is the owner's, see
    /// tilelang_gen/PROVENANCE.md §8). `k` must be a multiple of 32 (the ue8m0 block).
    moe_fp4_to_bf16: Option<
        unsafe extern "C" fn(*const c_void, *const c_void, *mut c_void, c_int, c_int, c_int, c_int, CuStream)
            -> c_int,
    >,
    /// DEVICE-SIDE `moe_align` (`dsv41_moe_align_from_group`,
    /// `dsv41_moe_align.cu`): projects the TileLang arm's `order`/`counts`/`eid`
    /// tables out of `dsv41_route_group`'s output ON THE DEVICE.
    ///
    /// This is the half that lets the TileLang MoE arm enter the verify graph:
    /// the host `moe_align` needs a D2H read of `route_idx_r` (a full sync and an
    /// illegal capture op), which is exactly why the arm used to be eager-only.
    /// Args (all DEVICE pointers, caller runs `route_group` on the same stream
    /// first): `(active, n_active, counts_by_e, starts, gather_src, order,
    /// counts_seg, eid, nseg_out, seg_cap, bm, stream)`. The output semantics are
    /// bit-for-bit `moe_align_host` (see the kernel's header comment).
    moe_align_from_group: Option<
        unsafe extern "C" fn(*const c_int, *const c_int, *const c_int, *const c_int, *const c_int, *mut c_int, *mut c_int, *mut c_int, *mut c_int, c_int, c_int, CuStream) -> c_int,
    >,
    /// The TileLang up arm's DEVICE-TABLE twin
    /// (`dsv41_moe_tilelang_gate_up_bf16_dev`, `tilelang_gen/moe_bf16_shim.cu`).
    /// Same three launches as `moe_tilelang_gate_up_bf16`, but `eid`/`order`/
    /// `counts`/`nseg` are DEVICE buffers (`nseg` is a pointer, not a value), so
    /// there is no H2D upload — i.e. the whole sequence is legal inside a
    /// CUDA-graph capture. The host-table entry is kept as the A/B baseline.
    moe_tilelang_gate_up_bf16_dev: Option<
        unsafe extern "C" fn(*const f32, *mut f32, *const c_void, *const c_int, *const c_int, *const c_int, *const c_int, c_int, c_int, c_int, c_int, CuStream) -> c_int,
    >,
    /// The TileLang down arm's device-table twin (see the up field above; the
    /// extra `c_int` before the stream is `act_pitch`).
    moe_tilelang_down_bf16_dev: Option<
        unsafe extern "C" fn(*const f32, *mut f32, *const c_void, *const c_int, *const c_int, *const c_int, *const c_int, c_int, c_int, c_int, c_int, c_int, CuStream) -> c_int,
    >,
    /// TILELANG MoE block-scaled arm — the **native fp4 weights** up (gate‖up) grouped
    /// GEMM (`DSV41_MOE_TILELANG_BS`, default OFF; mutually exclusive with
    /// `DSV41_MOE_TILELANG`). Unlike the bf16 arm this one reads the loader's fp4
    /// pool and its e8m0 scales **as-is** (zero dequant, zero bf16 copy), which is
    /// the whole point: it does not need `DSV41_MOE_BF16_DEQUANT` and its
    /// `+105 GiB/rank` bf16 mirror. Args:
    /// `(xq4, xsc4, out, w1, w3, sfw1, sfw3, eid, order, counts, nseg, w_stride,
    ///   rows, dim, inter, topk, stream)` — ⚠️ `xq4` is the **e4m3** activation
    /// (`[rows][dim] /* NOT [rows*topk]: activations are quantised per row */`, ONE byte per value — the official DeepSeek-V4.1
    /// `act_quant(fp8_block_size=32)` form the D2 fix adopted; it was packed e2m1
    /// `[dim/2]` before 2026-09-13) and `xsc4` its `[rows*topk][dim/32]` f32
    /// scales. `w1`/`w3` the expert pool's gate/up planes (still packed fp4),
    /// `sfw1`/`sfw3` the LOAD-TIME group-major packed scale words (see
    /// [`Self::moe_bs_pack_wsf`]). The activation's byte layout is versioned by
    /// the `dsv41_moe_bs_act_e4m3_cap` symbol (see
    /// [`Self::supports_moe_bs_act_e4m3`]) because the C ABI shape did not change.
    /// `w_stride` is the **measured per-expert block stride** in bytes (the pool
    /// packs each expert as one 128 B aligned six-plane block, so consecutive
    /// experts' `w1` are a whole block apart — NOT `NP*K/2`); it becomes the
    /// `gstride[1]` of the W1/W3 TMA descriptors.
    moe_tilelang_gate_up_bs: Option<
        unsafe extern "C" fn(
            *const u8,
            *const f32,
            *mut f32,
            *const c_void,
            *const c_void,
            *const c_void,
            *const c_void,
            *const c_int,
            *const c_int,
            *const c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            i64,
            CuStream,
        ) -> c_int,
    >,
    /// The block-scaled arm's DEVICE-TABLE twin
    /// (`dsv41_moe_tilelang_gate_up_bs_dev`, `tilelang_gen/moe_bs_shim.cu`).
    /// Same three launches (gather → block-scaled grouped MMA → scatter) and the
    /// same TMA-descriptor construction as [`Self::moe_tilelang_gate_up_bs`], but
    /// `eid`/`order`/`counts` are the DEVICE segment tables (`tl_eid`/`tl_order`/
    /// `tl_counts` filled by [`Self::moe_align_from_group`]) and `nseg` is a
    /// DEVICE **pointer** (`tl_nseg`, not a value). No H2D upload and no D2H read
    /// ⇒ the whole sequence is legal inside a CUDA-graph capture, which is what
    /// lets the native-fp4 arm enter the verify graph. The host-table entry is
    /// kept as the A/B baseline. Same `rc` contract (2 = DECLINED).
    ///
    /// `w_stride` is the measured per-expert **block stride** in bytes (see
    /// [`Self::moe_tilelang_gate_up_bs`]); it is the `gstride[1]` of both W1/W3
    /// TMA descriptors. It is required (not defaulted): the pool layout is the
    /// loader's private per-layer quantity, so only the caller can measure it.
    moe_tilelang_gate_up_bs_dev: Option<
        unsafe extern "C" fn(
            *const u8,
            *const f32,
            *mut f32,
            *const c_void,
            *const c_void,
            *const c_void,
            *const c_void,
            *const c_int,
            *const c_int,
            *const c_int,
            *const c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            i64,
            CuStream,
        ) -> c_int,
    >,
    /// Load-time repack for the block-scaled arm: one expert's e8m0 plane,
    /// row-major `[rows, k/32]` u8 → **group-major packed** `[words*rows]` u32
    /// (`word[g*rows + row] = u32(src[row*(k/32) + g*4 .. +4])`), which is the
    /// layout `T.tcgen05_gemm_blockscaled`'s SF operand wants. A pure byte
    /// permutation of already-ue8m0 data ⇒ bit-exact and lossless. Called
    /// `2 * n_routed` times per layer at load (see `load.rs`).
    moe_bs_pack_wsf:
        Option<unsafe extern "C" fn(*const c_void, *mut c_void, c_int, c_int, CuStream) -> c_int>,
    /// Capability marker for the block-scaled arm's **e4m3 A operand**
    /// (`dsv41_moe_bs_act_e4m3_cap`, `tilelang_gen/moe_bs_shim.cu` §8). The D2 fix
    /// (2026-09-13) changed `xq4`'s MEANING from packed fp4 nibbles (`dim/2` B/row)
    /// to e4m3 (`dim` B/row) **without changing its C ABI shape**, so a stale `.so`
    /// would silently decode 5120 B rows as 2560 B of fp4 — a wrong answer, not a
    /// fault. OPTIONAL on the same terms as `expert_act_e4m3_cap`: only the
    /// e4m3-activation build exports it, so a stale `.so` keeps the arm OFF
    /// (reported once) instead of feeding e4m3 bytes to an fp4 kernel.
    moe_bs_act_e4m3_cap: Option<unsafe extern "C" fn() -> c_int>,
    moe_down_reduce: Option<unsafe extern "C" fn(*const f32, *mut f32, c_int, c_int, CuStream) -> c_int>,
    /// `dsv41_moe_down_reduce_seq` (DSV41_SEQ_ALIGN #5): the same fixed-order sum,
    /// plus the ascending-EXPERT-ID slot permutation when `seq_align != 0`.
    /// A SEPARATE SYMBOL on purpose: a `.so` built before this change has only the
    /// 5-argument `dsv41_moe_down_reduce`, and the ABI would silently swallow the
    /// extra arguments — an armed gate would then measure the OLD order and say
    /// nothing (this project's #1 trap). `None` => the gate is inert and the
    /// caller keeps the legacy order, loudly.
    moe_down_reduce_seq: Option<
        unsafe extern "C" fn(*const f32, *mut f32, c_int, c_int, *const c_int, c_int, CuStream) -> c_int,
    >,
    /// `dsv41_moe_down_reduce_st` (A1a): the same fixed-order sum, whose epilogue
    /// also copies `out` into every peer's staging slot (carries the following
    /// all-reduce's store). Valid only where this sum is `s.o`'s LAST writer.
    moe_down_reduce_ar: Option<
        unsafe extern "C" fn(
            *const f32,
            *mut f32,
            c_int,
            c_int,
            *const *mut f32,
            *const c_uint,
            c_int,
            c_int,
            c_int,
            CuStream,
        ) -> c_int,
    >,
    /// w2 L2 prewarm (DSV41_W2_PREWARM): `cp.async.bulk.prefetch.L2.global` over
    /// every slot's w2 + scale rows, launched between the gate/up and the down
    /// launch. Writes nothing; returns 0 always (best effort).
    w2_l2_prewarm: Option<
        unsafe extern "C" fn(*const u8, i64, *const u8, i64, *const c_int, c_int, i64, i64, CuStream)
            -> c_int,
    >,
    swiglu_limit_batched:
        Option<unsafe extern "C" fn(*mut f32, c_int, c_int, f32, i64, c_int, c_int, CuStream) -> c_int>,
    /// ROUTED DOWN PREP (DSV41_ROUTED_DOWN_QUANT, default OFF): the official
    /// routed expert's activation pipeline in ONE in-place launch (route weight
    /// -> bf16 -> block-32 e4m3 quant+dequant), so the down GEMV consumes the
    /// operand the reference's `fp4_gemm` consumes. Optional: an older .so
    /// without it leaves the gate inert (with one notice).
    routed_down_prep: Option<
        unsafe extern "C" fn(*mut f32, *const f32, c_int, c_int, c_int, c_int, *mut f32, CuStream)
            -> c_int,
    >,
    /// A4 (`DSV41_INDEXER_FP4_RT`, default OFF): the indexer's q/k **fp4 round
    /// trip**, in place — the reference's `fp4_act_quant(x, 32, True)` with the
    /// default e8m0 power-of-two scale (`ref_inference/model.py:546` for k,
    /// `:552` for q; `kernel.py:126-183`). Block-32 amax -> `2^ceil(log2(amax/6))`
    /// -> e2m1 code -> the DEQUANTISED value written back. Optional: an older
    /// `.so` without the symbol leaves the gate inert (with a one-shot notice).
    /// ABI: (x, rows, cols, io, tag, dbg, stream).
    indexer_fp4_rt: Option<
        unsafe extern "C" fn(
            *mut f32,
            c_int,
            c_int,
            c_int,
            *const std::os::raw::c_char,
            *mut f32,
            CuStream,
        ) -> c_int,
    >,
    /// I3 (DSV41_ATTN_P_BF16_DBG): copy the attention PV probe buffer
    /// (`g_attn_p_dbg`, filled by the sparse-attention kernels when that env is
    /// set) to the host. An ADDED symbol: a `.so` without it simply cannot run
    /// the probe, and the gate itself never needs it.
    attn_p_dbg_read: Option<unsafe extern "C" fn(*mut f32, c_int, CuStream) -> c_int>,
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
    swiglu_limit: Option<unsafe extern "C" fn(*mut f32, c_int, c_int, f32, c_int, CuStream) -> c_int>,
    /// A4: swiglu + the fp8 pair the following GEMV consumes, in one launch.
    /// Returns 1 when the inter % 32 warp alignment cannot be met, so the caller
    /// keeps the swiglu_limit + quant1 pair.
    swiglu_limit_q:
        Option<unsafe extern "C" fn(*mut f32, c_int, c_int, f32, c_int, *mut u8, *mut f32, CuStream) -> c_int>,
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
    /// P3-lite segment A (draft): the whole hyper-connection front end --
    /// `hc_mixes` AND the collapse+rmsnorm -- as ONE launch, one block per draft
    /// row. Optional: a stale `.so` (or a reachable decline) leaves the caller on
    /// the `hc_mixes` + `hc_collapse_norm` pair. Returns 2 on a decline, 0 on a
    /// launch; the caller reads 2 as "use the pair".
    ///
    /// ABI: (x, hc_fn, hc_scale, hc_base, pre, post, comb, pre_collapse, w_norm,
    ///       out, rows, hc, dim, sinkhorn_iters, eps, eps_norm, truncate, stream)
    /// -- `pre_collapse` is the INCOMING premix slot the collapse reads, which is
    /// NOT the `pre` the mixes write (the draft walks the premix slots).
    draft_hc_front: Option<
        unsafe extern "C" fn(
            *const f32, *const f32, *const f32, *const f32,
            *mut f32, *mut f32, *mut f32,
            *const f32, *const f32, *mut f32,
            c_int, c_int, c_int, c_int, f32, f32, c_int, CuStream,
        ) -> c_int,
    >,
    /// L4-9: the dim-split twin of [`Self::hc_collapse_norm`]
    /// (`dsv41_hc_collapse_norm_split_kernel`), `DSV41_CNORM_SPLIT` (default
    /// OFF). Same arguments plus a trailing `nchunks` before the stream, and
    /// `grid = (nchunks, rows)` instead of `grid = rows` — the fix for the
    /// m = 1 one-CTA case. Optional: an `.so` without the symbol, or a gate
    /// that is off, keeps the fused single-launch version unchanged.
    hc_collapse_norm_split: Option<unsafe extern "C" fn(
        *mut f32, *const f32, *const f32, *mut f32,
        c_int, c_int, c_int, f32, c_int, c_int, CuStream,
    ) -> c_int>,
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
    /// P3 MEGAKERNEL (docs/agent/p3-megakernel-verify-design.md §2.1; gate
    /// `DSV41_P3_MEGAKERNEL=1`, default OFF): the verify block's hc front end —
    /// the dots spread over `mix*split` blocks, the collapse/rmsnorm on its own
    /// PARALLEL block, and the tail (ss/sigmoid/sinkhorn/comb) ELECTED to the
    /// last-finishing dot block — in ONE launch, PLUS the row-based T1 fp8 emit
    /// (`xq[r*pitch + c]`, `xsc[r*pitch/32 + c/32]`). Same shape as
    /// `hc_front_persist_mb`; the row base in the emit is what lets a multi-row
    /// block drop its trailing `quant_rows(xn)` launch. Bit-exact at `split == 1`
    /// (the `< 1 ulp` caveat for `split > 1` is `hc_front_persist_mb`'s). Returns
    /// false on any decline so the caller keeps the older chain.
    verify_hc_front_prefused: Option<
        unsafe extern "C" fn(
            *const f32, *const f32, *const f32, *const f32,
            *const f32, *const f32,
            *mut f32, *mut f32, *mut f32, *mut f32,
            *mut u8, *mut f32, c_int,
            c_int, c_int, c_int, c_int, f32, f32, c_int, CuStream,
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
    /// `ferrite_p2p_ar_v5_attn` / `ferrite_p2p_ar_v5_moe` (A0 site split):
    /// byte-for-byte `p2p_ar_v5` with the probe's site label set to the attention
    /// / MoE half, so SWALLOW's two verify all-reduces (`chain_dev.rs`
    /// `layer_rows` / `moe_rows`, which run through `all_reduce_inplace`) stop
    /// sharing the lumped `AR5_SITE_OTHER` bucket. Optional: an `.so` predating
    /// them falls back to `p2p_ar_v5` and the probe just reports `Other`.
    p2p_ar_v5_attn: Option<
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
    /// See [`Kernels::p2p_ar_v5_attn`].
    p2p_ar_v5_moe: Option<
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
    /// `ferrite_p2p_ar_pubred_v5_moe` (A1a): byte-for-byte `p2p_ar_pubred_v5`
    /// with the A0 probe's site label set to the MoE site — the MoE AR runs the
    /// same publish+reduce once a producer carried the store.
    p2p_ar_pubred_v5_moe: Option<
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
    /// `ferrite_p2p_ar_pubred_v5_hcpost` (A1a): the publish+reduce AND the
    /// hc-post fold with NO store — the payload was already carried by a
    /// producer's epilogue. Covers the ADD_EPI (`_hcpost_add`) path too, because
    /// the folded residual only changes what is PUBLISHED (now the carrier's
    /// business); the hc-post half consumes `out`.
    p2p_ar_pubred_v5_hcpost: Option<
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
            gemm_fp8_sh_exp_fused: ko!(rt, "dsv41_gemm_fp8_sh_exp_fused"),
            sh_exp_tilelang: ko!(rt, "dsv41_sh_exp_tilelang"),
            sh_exp_tilelang_gu: ko!(rt, "dsv41_sh_exp_tilelang_gu"),
            sh_exp_tilelang_dn: ko!(rt, "dsv41_sh_exp_tilelang_dn"),
            quant_fp8: km!(rt, "dsv41_quant_fp8"),
            quant_fp8_stride: ko!(rt, "dsv41_quant_fp8_stride"),
            quant_fp4: km!(rt, "dsv41_quant_fp4"),
            win_kv_quant_rt: ko!(rt, "dsv41_win_kv_quant_rt"),
            expert_gate_up_fp4: km!(rt, "dsv41_expert_gate_up_fp4"),
            expert_down_fp4: km!(rt, "dsv41_expert_down_fp4"),
            engram_hash: km!(rt, "dsv41_engram_hash"),
            engram_gather: km!(rt, "dsv41_engram_gather"),
            engram_gather_rows: ko!(rt, "dsv41_engram_gather_rows"),
            sparse_attn: km!(rt, "dsv41_sparse_attn"),
            sparse_attn_orope: ko!(rt, "dsv41_sparse_attn_orope"),
            sparse_attn_rp: ko!(rt, "dsv41_sparse_attn_rp"),
            sparse_attn_orope_rp: ko!(rt, "dsv41_sparse_attn_orope_rp"),
            indexer_topk: km!(rt, "dsv41_indexer_topk"),
            indexer_topk_rows: ko!(rt, "dsv41_indexer_topk_rows"),
            candidate_blocks: km!(rt, "dsv41_candidate_blocks"),
            compressor: km!(rt, "dsv41_compressor"),
            rope_precompute: km!(rt, "dsv41_rope_precompute"),
            apply_rope: km!(rt, "dsv41_apply_rope"),
            apply_rope_mrows: ko!(rt, "dsv41_apply_rope_mrows"),
            rmsnorm_rope: ko!(rt, "dsv41_rmsnorm_rope"),
            rmsnorm_q: ko!(rt, "dsv41_rmsnorm_q"),
            rmsnorm_rows: ko!(rt, "dsv41_rmsnorm_rows"),
            rmsnorm_rows_split: ko!(rt, "dsv41_rmsnorm_rows_split"),
            rmsnorm_rope_mrows: ko!(rt, "dsv41_rmsnorm_rope_mrows"),
            gemm_bf16_fp8x2: ko!(rt, "dsv41_gemm_bf16_fp8x2"),
            argmax_sliced: ko!(rt, "dsv41_argmax_sliced"),
            argmax_sliced_rows: ko!(rt, "dsv41_argmax_sliced_rows"),
            v5_epoch_pad: ko!(rt, "dsv41_v5_epoch_pad"),
            hc_mixes: km!(rt, "dsv41_hc_mixes"),
            moe_route: km!(rt, "dsv41_moe_route"),
            add_inplace: ko!(rt, "ferrite_add"),
            add_inplace_ar: ko!(rt, "ferrite_add_store"),
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
            gemv_bf16_v2_mrows_route: ko!(rt, "ferrite_gemv_bf16_v2_mrows_route"),
            gemv_f32: ko!(rt, "dsv41_gemv_f32"),
            gemv_f32_v2: ko!(rt, "dsv41_gemv_f32_v2"),
            gemv_f32_mrows: ko!(rt, "dsv41_gemv_f32_mrows"),
            head_gemv_bf16_mrows: ko!(rt, "dsv41_head_gemv_bf16_mrows"),
            gemv_bf16_v1_mrows: ko!(rt, "dsv41_gemv_bf16_v1_mrows"),
            head_bf16_tilelang: ko!(rt, "dsv41_head_bf16_tilelang"),
            bf16_roundtrip: ko!(rt, "dsv41_bf16_roundtrip"),
            compress_ring_quant: ko!(rt, "dsv41_compress_ring_quant"),
            wo_a_grouped_fp8: ko!(rt, "dsv41_wo_a_grouped_fp8"),
            gemm_fp8_mrows: ko!(rt, "dsv41_gemm_fp8_mrows"),
            gemm_fp8_mrows_mma: ko!(rt, "dsv41_gemm_fp8_mrows_mma"),
            gemm_fp8_tilelang_wkv: ko!(rt, "dsv41_gemm_fp8_tilelang_wkv"),
            gemm_fp8_tilelang_wq_a: ko!(rt, "dsv41_gemm_fp8_tilelang_wq_a"),
            gemm_fp8_tilelang_wq_b: ko!(rt, "dsv41_gemm_fp8_tilelang_wq_b"),
            gemm_fp8_tilelang_wo_b: ko!(rt, "dsv41_gemm_fp8_tilelang_wo_b"),
            gemm_fp8_tilelang_wo_a: ko!(rt, "dsv41_gemm_fp8_tilelang_wo_a"),
            gemm_fp8_mrows_rope_norm: ko!(rt, "dsv41_gemm_fp8_mrows_rope_norm"),
            gemm_fp8_mrows2: ko!(rt, "dsv41_gemm_fp8_mrows2"),
            gemm_fp8_mrows_f32: ko!(rt, "dsv41_gemm_fp8_mrows_f32"),
            gemm_fp8_mrows_q_f32: ko!(rt, "dsv41_gemm_fp8_mrows_q_f32"),
            argmax: ko!(rt, "dsv41_argmax"),
            engram_hash_step: ko!(rt, "dsv41_engram_hash_step"),
            window_idxs: ko!(rt, "dsv41_window_idxs"),
            comp_placeholder: ko!(rt, "dsv41_comp_placeholder"),
            compress_commit: ko!(rt, "dsv41_compress_commit"),
            compress_commit_rows: ko!(rt, "dsv41_compress_commit_rows"),
            ring_append: ko!(rt, "dsv41_ring_append"),
            apply_rope_q: ko!(rt, "dsv41_apply_rope_q"),
            ring_win_fuse: ko!(rt, "dsv41_ring_win_fuse"),
            ring_win_fuse_ph: ko!(rt, "dsv41_ring_win_fuse_ph"),
            ring_win_fuse_mrows: ko!(rt, "dsv41_ring_win_fuse_mrows"),
            verify_ring_win: ko!(rt, "dsv41_verify_ring_win"),
            sparse_attn_orope_mrows: ko!(rt, "dsv41_sparse_attn_orope_mrows"),
            window_idxs_mrows: ko!(rt, "dsv41_window_idxs_mrows"),
            ring_append_mrows: ko!(rt, "dsv41_ring_append_mrows"),
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
            moe_down_reduce_seq: ko!(rt, "dsv41_moe_down_reduce_seq"),
            moe_down_reduce_ar: ko!(rt, "dsv41_moe_down_reduce_st"),
            expert_down_reduce_fp4_batched: ko!(rt, "dsv41_expert_down_reduce_fp4_batched"),
            moe_tilelang_gate_up_bf16: ko!(rt, "dsv41_moe_tilelang_gate_up_bf16"),
            moe_tilelang_down_bf16: ko!(rt, "dsv41_moe_tilelang_down_bf16"),
            moe_tilelang_gate_up_bf16_dev: ko!(rt, "dsv41_moe_tilelang_gate_up_bf16_dev"),
            moe_tilelang_down_bf16_dev: ko!(rt, "dsv41_moe_tilelang_down_bf16_dev"),
            moe_align_from_group: ko!(rt, "dsv41_moe_align_from_group"),
            moe_fp4_to_bf16: ko!(rt, "dsv41_moe_fp4_to_bf16"),
            moe_tilelang_gate_up_bs: ko!(rt, "dsv41_moe_tilelang_gate_up_bs"),
            moe_tilelang_gate_up_bs_dev: ko!(rt, "dsv41_moe_tilelang_gate_up_bs_dev"),
            moe_bs_pack_wsf: ko!(rt, "dsv41_moe_bs_pack_wsf"),
            moe_bs_act_e4m3_cap: ko!(rt, "dsv41_moe_bs_act_e4m3_cap"),
            w2_l2_prewarm: ko!(rt, "dsv41_w2_l2_prewarm"),
            attn_p_dbg_read: ko!(rt, "dsv41_attn_p_dbg_read"),
            swiglu_limit_batched: ko!(rt, "dsv41_swiglu_limit_batched"),
            routed_down_prep: ko!(rt, "dsv41_routed_down_prep"),
            indexer_fp4_rt: ko!(rt, "dsv41_indexer_fp4_rt"),
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
            draft_hc_front: ko!(rt, "dsv41_draft_hc_front"),
            hc_collapse_norm_split: ko!(rt, "dsv41_hc_collapse_norm_split"),
            hc_front: km!(rt, "dsv41_hc_front"),
            hc_front_persist: ko!(rt, "dsv41_hc_front_persist"),
            hc_front_persist_mb: ko!(rt, "dsv41_hc_front_persist_mb"),
            verify_hc_front_prefused: ko!(rt, "dsv41_verify_hc_front_prefused"),
            hc_front_split: ko!(rt, "dsv41_hc_front_split"),
            embed_expand_dev: km!(rt, "ferrite_embed_expand_dev"),
            f32_to_bf16: km!(rt, "ferrite_f32_to_bf16"),
            bf16_to_f32: km!(rt, "ferrite_bf16_to_f32"),
            p2p_ar_v5: ko!(rt, "ferrite_p2p_ar_v5"),
            p2p_ar_v5_attn: ko!(rt, "ferrite_p2p_ar_v5_attn"),
            p2p_ar_v5_moe: ko!(rt, "ferrite_p2p_ar_v5_moe"),
            p2p_ar_pubred_v5: ko!(rt, "ferrite_p2p_ar_pubred_v5"),
            p2p_ar_pubred_v5_moe: ko!(rt, "ferrite_p2p_ar_pubred_v5_moe"),
            p2p_ar_v5_hcpost: ko!(rt, "ferrite_p2p_ar_v5_hcpost"),
            p2p_ar_v5_add: ko!(rt, "ferrite_p2p_ar_v5_add"),
            p2p_ar_v5_hcpost_add: ko!(rt, "ferrite_p2p_ar_v5_hcpost_add"),
            p2p_ar_pubred_v5_hcpost: ko!(rt, "ferrite_p2p_ar_pubred_v5_hcpost"),
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

    /// [`Self::bf16_roundtrip`] on an explicit stream. `DSV41_BF16_TRUNCATE`'s
    /// boundaries sit inside chains that ride the side streams (`dual_chain`
    /// forks the kv norm+rope onto stream 2, `COMPRESS_SIDE` puts the
    /// compressor's four launches on stream 3, `MOE_DUAL` puts the shared
    /// expert's on stream 2), and a round trip issued on the MAIN stream while
    /// its producer is still running on a side stream would race instead of
    /// ordering. Same symbol, same kernel, same `Ok(false)`-when-stale
    /// contract; only the stream moves.
    pub fn bf16_roundtrip_on(&self, x: *mut f32, n: i64, s: CuStream) -> Result<bool> {
        let Some(f) = self.kernels.bf16_roundtrip else {
            return Ok(false);
        };
        let rc = unsafe { f(x, n, s) };
        self.kerr(rc, "dsv41_bf16_roundtrip")?;
        Ok(true)
    }

    /// A3 (`DSV41_COMPRESS_LATENT_QUANT`): the fp4 roundtrip of the row the
    /// compressed-KV commit just wrote. MUST be issued on the SAME stream as the
    /// commit launch — it reads the device counter the commit bumped, and reads
    /// and rewrites the ring row the commit produced, so a different stream would
    /// both race and pick the wrong slot.
    ///
    /// `row_quant == 0` is a no-op INSIDE the kernel (one predicate per thread),
    /// so the gate can be read on the host without a second code path; a stale
    /// `.so` returns Ok(false) and the caller keeps the bf16-only boundary with
    /// one note (house rule: an armed-but-inert gate announces itself).
    pub fn compress_ring_quant_on(
        &self,
        ring: *mut f32,
        out_rows: *const c_int,
        clen: *const c_int,
        window: i32,
        hd: i32,
        bf16_first: i32,
        row_quant: i32,
        dbg: *mut f32,
        s: CuStream,
    ) -> Result<bool> {
        let Some(f) = self.kernels.compress_ring_quant else {
            return Ok(false);
        };
        let rc = unsafe {
            f(
                ring,
                out_rows,
                clen,
                window as c_int,
                hd as c_int,
                bf16_first as c_int,
                row_quant as c_int,
                dbg,
                s,
            )
        };
        self.kerr(rc, "dsv41_compress_ring_quant")?;
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

    /// Strided D2D copy — `height` rows of `width` bytes, `dpitch`/`spitch` apart.
    /// See [`ferrite_kernel::devrt::DevRuntime::memcpy_d2d_2d`].
    pub fn memcpy_d2d_2d(
        &self,
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
    ) -> Result<()> {
        self.rt.memcpy_d2d_2d(dst, dpitch, src, spitch, width, height)
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

    /// [`Self::add_inplace`] whose epilogue ALSO copies `dst` into every peer's
    /// staging slot (A1a, `ferrite_add_store`) — i.e. the launch carries the
    /// following all-reduce's store, so the caller must then run the
    /// publish+reduce half instead of the full all-reduce.
    ///
    /// `Ok(false)` when the loaded .so has no `ferrite_add_store` (stale build):
    /// NOTHING was launched and the caller must fall back to `add_inplace` + the
    /// ordinary all-reduce, so the store can never be dropped silently.
    #[allow(clippy::too_many_arguments)]
    pub fn add_inplace_ar(
        &self,
        dst: &DevBuf,
        src: &DevBuf,
        n: i64,
        staging_tbl: *const *mut f32,
        epoch: *const c_uint,
        world: i32,
        my_rank: i32,
        stride: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.add_inplace_ar else {
            return Ok(false);
        };
        let rc = unsafe {
            f(
                dst.ptr as *const f32,
                src.ptr as *const f32,
                dst.ptr as *mut f32,
                n as c_int,
                staging_tbl,
                epoch,
                world,
                my_rank,
                stride,
                self.stream,
            )
        };
        self.kerr(rc, "ferrite_add_store")?;
        Ok(true)
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

    /// True when the loaded .so carries the DSV41_SEQ_ALIGN (#5) scratch-reduce
    /// entry point (`dsv41_moe_down_reduce_seq`). Absent => the gate's non-fused
    /// down path keeps the ascending-slot order, and the caller says so once.
    pub fn supports_moe_down_reduce_seq(&self) -> bool {
        self.kernels.moe_down_reduce_seq.is_some()
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

    /// True when the loaded .so carries BOTH MoE store carriers of A1a
    /// (`dsv41_moe_down_reduce_st` / `ferrite_add_store`). Without them
    /// `DSV41_AR_STORE_FUSE` stays inert for the MoE site and every AR keeps its
    /// own `p2p_ar_store_v5_kernel` launch.
    pub fn supports_ar_store_carriers(&self) -> bool {
        self.kernels.moe_down_reduce_ar.is_some() && self.kernels.add_inplace_ar.is_some()
    }

    /// True when the loaded .so carries the MoE publish+reduce halves of A1a
    /// (`ferrite_p2p_ar_pubred_v5_moe` for the plain AR,
    /// `ferrite_p2p_ar_pubred_v5_hcpost` for the folded one). A stale .so
    /// reports false and the MoE site stays on the full all-reduce.
    pub fn supports_ar_pubred_moe(&self) -> bool {
        self.kernels.p2p_ar_pubred_v5_moe.is_some()
            && self.kernels.p2p_ar_pubred_v5_hcpost.is_some()
    }

    /// True when the loaded `.so` carries the MULTI-ROW hc-post fold
    /// (`ferrite_p2p_ar_v5_hcpost_rows`) — the verify path's AR fold
    /// (`chain_dev::ChainDev::ar_hc_post_fold_rows`). A stale `.so` reports false
    /// and the caller keeps the `all_reduce_inplace` + `hc_post_inplace_rows`
    /// pair, so the switch is free to make.
    pub fn supports_ar_hcpost_rows(&self) -> bool {
        self.kernels.p2p_ar_v5_hcpost_rows.is_some()
    }

    /// True when the loaded `.so` carries BOTH A0 site-labelled v5 entries
    /// (`ferrite_p2p_ar_v5_attn` / `ferrite_p2p_ar_v5_moe`, the probe's site
    /// split). This is the self-check the probe runbook's `nm -D` step stands for:
    /// without them every SWALLOW verify all-reduce lands in the lumped
    /// `AR5_SITE_OTHER` bucket and the `[ar-probe]` line cannot say which half of
    /// the step is waiting. The call path itself falls back to the plain entry per
    /// call ([`Self::p2p_ar_v5_site`]), so a stale `.so` never breaks the AR.
    pub fn supports_ar_v5_site(&self) -> bool {
        self.kernels.p2p_ar_v5_attn.is_some() && self.kernels.p2p_ar_v5_moe.is_some()
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

    /// True when the loaded .so carries the M-ROW shared-expert chain
    /// (`dsv41_gemm_fp8_sh_exp_fused`, the `template<M>` form of
    /// `dsv41_gemm_fp8_sh_pair`). A stale .so leaves `DSV41_SH_PAIR_M` inert and
    /// the caller keeps its per-row chain (the M=1 arm, then SH_EXP_MROWS, then
    /// the per-row loop).
    pub fn supports_sh_exp_fused(&self) -> bool {
        self.kernels.gemm_fp8_sh_exp_fused.is_some()
    }

    /// The M-row shared expert (`w1w3 -> swiglu -> fp8 emit | barrier | w2`) as
    /// ONE launch for the whole verify block, with the optional `epi_add`.
    ///
    /// `fold_r` (1..=rows) is the A/B knob: how many activation rows one block
    /// owns in phase 1. 1 keeps the M=1 kernel's exact consume path (`s_af[j]`)
    /// and the design's §3.3 parallel axis; >1 trades SM coverage for fewer
    /// instructions. It is a RUNTIME argument, so the A/B does not recompile.
    ///
    /// `act` may be null (a by-product in this arm). `aq`/`aqsc` must be
    /// DISJOINT from `a`/`a_scale`, and their strides are the caller's layout
    /// (`aq_stride` must be a multiple of 16 -- phase 2 reads the rows as
    /// uint4). `epi_add != 0` makes phase 2 read-modify-write `out`, folding the
    /// caller's separate `ferrite_add(out, sh_out)` away.
    ///
    /// Ok(false) => the shape/arm cannot use it and the caller keeps its chain.
    /// ABI: stream LAST (this symbol has no C++ default tail args).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_sh_exp_fused(
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
        rows: i32,
        fold_r: i32,
        act: *mut f32,
        act_stride: i32,
        aq: *mut u8,
        aqsc: *mut f32,
        aq_stride: i32,
        aqsc_stride: i32,
        w2: *const u8,
        w2_scale: *const u8,
        n2: i32,
        out_stride: i32,
        epi_add: i32,
        out: *mut f32,
        s: CuStream,
    ) -> Result<bool> {
        let Some(f) = self.kernels.gemm_fp8_sh_exp_fused else {
            return Ok(false);
        };
        if !(1..=8).contains(&rows) {
            return Ok(false);
        }
        let rc = unsafe {
            f(
                a, a_scale, wg, wg_scale, wu, wu_scale, limit, n1, k1, rows, fold_r, act,
                act_stride, aq, aqsc, aq_stride, aqsc_stride, w2, w2_scale, n2, out_stride,
                epi_add, out, s,
            )
        };
        // A shape/arm decline is 2 (1 collides with cudaErrorInvalidValue).
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_sh_exp_fused")?;
        Ok(true)
    }

    /// TILELANG shared expert (`dsv41_sh_exp_tilelang`, from
    /// `kernels/cuda/tilelang_gen/sh_exp_shim.cu`) — the whole
    /// (w1w3 -> swiglu -> fp8 emit -> w2) chain as TWO TileLang fp8 MMA launches, so
    /// the M activation rows SHARE one weight read and the SIMT arm's grid barrier
    /// disappears. Design: `docs/agent/c5-sh-exp-tilelang-design.md`.
    ///
    /// ABI = [`Self::gemm_fp8_sh_exp_fused`]'s list PLUS `w2sc_pitch`: the generated
    /// kernel bakes `w2.scale`'s **UNPADDED 9 B row pitch** (`[160, 9]`), and a pool
    /// whose pitch later changes would make it read the wrong row SILENTLY — so the
    /// shim takes the pitch as a run-time argument and DECLINES on any other value.
    /// `fold_r` is an ABI placeholder (the M fold lives inside the mma tile); `act`
    /// must be null (the generated kernel has no `act` path).
    ///
    /// ⚠️ (b') DOUBLE SWAP: this program puts M inside the mma tile, so m=1 and m<=8
    /// run the SAME program (row r of an M-row launch == row r of the M=1 launch OF
    /// THIS PROGRAM), but it is NOT bit-identical to the SIMT arm. Take it on BOTH
    /// sides (eager + verify) or neither — one-sided is the `DSV41_PROJ_MMA` death.
    ///
    /// `Ok(false)` => the shape/arm cannot use it (or a stale `.so` has no symbol) and
    /// the caller keeps its existing chain. Stream LAST.
    #[allow(clippy::too_many_arguments)]
    pub fn sh_exp_tilelang(
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
        rows: i32,
        fold_r: i32,
        act: *mut f32,
        act_stride: i32,
        aq: *mut u8,
        aqsc: *mut f32,
        aq_stride: i32,
        aqsc_stride: i32,
        w2: *const u8,
        w2_scale: *const u8,
        n2: i32,
        w2sc_pitch: i32,
        out_stride: i32,
        epi_add: i32,
        out: *mut f32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.sh_exp_tilelang else {
            return Ok(false);
        };
        if !(1..=8).contains(&rows) {
            return Ok(false);
        }
        let rc = unsafe {
            f(
                a, a_scale, wg, wg_scale, wu, wu_scale, limit, n1, k1, rows, fold_r, act,
                act_stride, aq, aqsc, aq_stride, aqsc_stride, w2, w2_scale, n2, w2sc_pitch,
                out_stride, epi_add, out, self.stream,
            )
        };
        // A shape/pitch/alignment decline is 2 (1 collides with cudaErrorInvalidValue).
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_sh_exp_tilelang")?;
        Ok(true)
    }

    /// True when the loaded .so carries the TileLang shared-expert entry
    /// (`dsv41_sh_exp_tilelang`, compiled in from `kernels/cuda/tilelang_gen/` — see
    /// `kernels/cuda/build.sh`). A stale .so that predates the TU leaves
    /// `DSV41_SH_EXP_TILELANG` inert and the caller on its existing chain.
    pub fn supports_sh_exp_tilelang(&self) -> bool {
        self.kernels.sh_exp_tilelang.is_some()
    }

    /// L1 only (`dsv41_sh_exp_tilelang_gu`): gate+up -> swiglu+limit -> fp8 quant.
    /// Same program as [`Self::sh_exp_tilelang`]'s first launch, exposed separately so
    /// 户部's microbench can time L1 and L2 independently (design §建议分工) and so a
    /// decline can be attributed to a phase. The arm itself uses the combined entry.
    #[allow(clippy::too_many_arguments)]
    pub fn sh_exp_tilelang_gu(
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
        rows: i32,
        aq: *mut u8,
        aqsc: *mut f32,
        aq_stride: i32,
        aqsc_stride: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.sh_exp_tilelang_gu else {
            return Ok(false);
        };
        if !(1..=8).contains(&rows) {
            return Ok(false);
        }
        let rc = unsafe {
            f(
                a, a_scale, wg, wg_scale, wu, wu_scale, limit, n1, k1, rows, aq, aqsc, aq_stride,
                aqsc_stride, self.stream,
            )
        };
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_sh_exp_tilelang_gu")?;
        Ok(true)
    }

    /// L2 only (`dsv41_sh_exp_tilelang_dn`): the w2 down GEMM, with `epi_add != 0`
    /// folding the caller's `add_inplace_raw(out, sh_out)` into its store. Same frozen
    /// geometry and same `w2sc_pitch` gate as the combined entry (see above).
    #[allow(clippy::too_many_arguments)]
    pub fn sh_exp_tilelang_dn(
        &self,
        aq: *const u8,
        aqsc: *const f32,
        w2: *const u8,
        w2_scale: *const u8,
        n2: i32,
        k2: i32,
        w2sc_pitch: i32,
        aq_stride: i32,
        aqsc_stride: i32,
        out_stride: i32,
        epi_add: i32,
        out: *mut f32,
        rows: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.sh_exp_tilelang_dn else {
            return Ok(false);
        };
        if !(1..=8).contains(&rows) {
            return Ok(false);
        }
        let rc = unsafe {
            f(
                aq, aqsc, w2, w2_scale, n2, k2, w2sc_pitch, aq_stride, aqsc_stride, out_stride,
                epi_add, out, rows, self.stream,
            )
        };
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_sh_exp_tilelang_dn")?;
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

    /// F7 (`DSV41_F7_QUANT_STRIDE`): is the STRIDED quantiser in this `.so`?
    ///
    /// The folded call sites all keep the per-row loop when this is false — a
    /// stale `.so` resolves the plain `dsv41_quant_fp8` but would silently ignore a
    /// pitch passed to it, which is exactly the misread the arm removes.
    pub fn supports_quant_fp8_stride(&self) -> bool {
        self.kernels.quant_fp8_stride.is_some()
    }

    /// [`Self::quant_fp8`] with the SOURCE row pitch passed explicitly
    /// (`src_stride == 0` ⇒ `== cols`, the legacy spelling).
    ///
    /// The DESTINATION is always compact at `cols` (that is the layout every
    /// caller's `xq_r`/`xsc_r` staging promises), so the argument moves nothing but
    /// the source reads: a `rows = m` call with `src_stride = S` writes exactly the
    /// bytes `m` single-row calls at `x + r*S`, `y + r*cols`, `scale + r*(cols/32)`
    /// wrote (the row index only picks base pointers, and the per-32-block amax is
    /// the row's own reduction).
    ///
    /// Fails loudly on a `.so` without the symbol rather than quietly taking the
    /// `cols` pitch: the caller's gate tests
    /// [`Self::supports_quant_fp8_stride`] first.
    pub fn quant_fp8_strided(
        &self,
        x: *const f32,
        y: *mut u8,
        scale: *mut f32,
        rows: i32,
        cols: i32,
        block: i32,
        round_scale: bool,
        src_stride: i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.quant_fp8_stride, "dsv41_quant_fp8_stride")?;
        let rc = unsafe {
            f(
                x,
                y,
                scale,
                rows,
                cols,
                block,
                round_scale as i32,
                src_stride,
                self.stream,
            )
        };
        self.kerr(rc, "dsv41_quant_fp8_stride")
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

    /// Is the A2 round trip in this `.so`? (See [`Self::win_kv_quant_rt_on`].)
    pub fn supports_win_kv_quant_rt(&self) -> bool {
        self.kernels.win_kv_quant_rt.is_some()
    }

    /// A2 (`DSV41_WINDOW_KV_QUANT`): the sliding-window KV's IN-PLACE fp8 e4m3
    /// round trip on `kv[0, cols)`, issued on `s`.
    ///
    /// This is the reference's `act_quant(kv, 32, "ue8m0", e8m0fnu, True)`
    /// (`ref_inference/model.py:705`) on the post-RoPE row, whose `inplace=True`
    /// arm (`kernel.py:83-88`) stores the DEQUANTISED value: the row becomes
    /// `e4m3_decode(e4m3_encode(clamp(v / s, ±448))) * s` with
    /// `s = 2^ceil(log2(max(amax, 1e-4) / 448))`. The buffer keeps its f32
    /// element type -- only the VALUE moves onto the official grid.
    ///
    /// Returns `Ok(true)` when the round trip ran. `Ok(false)` means the arm
    /// declined and **the row is untouched**: either the loaded `.so` has no
    /// symbol (a one-shot notice says so -- otherwise an A/B arm would silently
    /// measure the OLD path) or the launcher refused the shape (`cols % block`,
    /// or a block the launch geometry cannot express). Never a partial snap.
    pub fn win_kv_quant_rt_on(
        &self,
        kv: *mut f32,
        cols: i32,
        block: i32,
        dbg: Option<*mut f32>,
        s: CuStream,
    ) -> Result<bool> {
        let Some(f) = self.kernels.win_kv_quant_rt else {
            return Ok(false);
        };
        if kv.is_null() || cols <= 0 || block <= 0 {
            return Ok(false);
        }
        let dbg = dbg.unwrap_or(std::ptr::null_mut());
        let rc = unsafe { f(kv, cols, block, dbg, s) };
        if rc == 1 || rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_win_kv_quant_rt")?;
        Ok(true)
    }

    /// [`Self::win_kv_quant_rt_on`] on the main stream.
    pub fn win_kv_quant_rt(
        &self,
        kv: *mut f32,
        cols: i32,
        block: i32,
        dbg: Option<*mut f32>,
    ) -> Result<bool> {
        self.win_kv_quant_rt_on(kv, cols, block, dbg, self.stream)
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

    /// ENGRAM-GATHER-MROWS (`DSV41_ENGRAM_GATHER_MROWS`): the multi-row engram
    /// gather — `rows` tokens' `n_cols` hash rows each, gathered in ONE launch.
    /// `id_stride` is the per-TOKEN pitch of `hash_ids` (in elements); the
    /// caller passes the engram-layer stride `n_eng * n_cols` that its
    /// `[row][engram layer][col]` id layout implies. Per element this is the
    /// same id, the same table lookup and the same `out` slot as the per-row
    /// `engram_gather` calls it replaces, so the rows are bit-identical.
    ///
    /// `Ok(true)` = launched. `Ok(false)` = NOT performed (a stale `.so` without
    /// the symbol, `rows` outside `1..=8`, `id_stride < n_cols`, a bad shape);
    /// the caller keeps its per-row loop, which is the bit-exact reference.
    /// Never an error, exactly as [`Self::gemm_fp8_mrows`] documents.
    #[allow(clippy::too_many_arguments)]
    pub fn engram_gather_rows(
        &self,
        table: *const u8,
        table_scale: *const u8,
        hash_ids: *const i64,
        out: *mut f32,
        rows: i32,
        n_cols: i32,
        head_dim: i32,
        id_stride: i32,
        part_start: i64,
        part_rows: i64,
    ) -> Result<bool> {
        let Some(f) = self.kernels.engram_gather_rows else {
            return Ok(false);
        };
        if !(1..=8).contains(&rows) || id_stride < n_cols {
            return Ok(false);
        }
        let rc = unsafe {
            f(
                table, table_scale, hash_ids, out, rows, n_cols, head_dim, id_stride, part_start,
                part_rows, self.stream,
            )
        };
        // 2 = the entry's "declined" (shape), the caller falls back.
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_engram_gather_rows")?;
        Ok(true)
    }

    /// W2-MROWS-TP: the pre-TP entry, kept with its EXACT signature so every
    /// existing caller (the per-row loop here, the eager arm, `dspark_dev.rs`) is
    /// untouched. `row_pitch = 0` means `h*d`, the only spelling there was.
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
        clen_rows: *const c_int,
        idx_stride: i32,
    ) -> Result<()> {
        self.sparse_attn_pitched(q, kv, sink, idxs, out, b, m, h, d, clen, window, index_topk,
                                 scale, clen_rows, idx_stride, 0)
    }

    /// W2-MROWS-TP: [`Self::sparse_attn`] with an explicit `q`/`out` row pitch.
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attn_pitched(
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
        // W2-MROWS-TP: the `q`/`out` row pitch in elements (`0` = `h*d`, the
        // pre-TP default, and the only value a `world == 1` call passes). Non-zero
        // is the `world > 1` batch and dispatches to `dsv41_sparse_attn_rp`; the
        // caller gates that arm on [`Self::supports_sparse_attn_rp`], so a missing
        // symbol here is a caller bug and reports as one rather than silently
        // launching the `h*d` kernel.
        row_pitch: i32,
    ) -> Result<()> {
        if row_pitch != 0 {
            let Some(f) = self.kernels.sparse_attn_rp else {
                return Err(FerriteError::Config(
                    "dsv41_sparse_attn_rp missing: the TP row-pitch arm needs it".into(),
                ));
            };
            let rc = unsafe {
                f(q, kv, sink, idxs, out, b, m, h, d, clen, window, index_topk, scale, clen_rows,
                  idx_stride, row_pitch, self.stream)
            };
            return self.kerr(rc, "dsv41_sparse_attn_rp");
        }
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
    /// W2-MROWS-TP: the pre-TP entry (see [`Self::sparse_attn`]) — same exact
    /// signature and `row_pitch = 0`.
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
        clen_rows: *const c_int,
        idx_stride: i32,
        row_step: i32,
    ) -> Result<bool> {
        self.sparse_attn_orope_pitched(q, kv, sink, idxs, out, b, m, h, d, clen, window, index_topk,
                                       scale, cos, sin, base, rope_rd, half, mul, off, step,
                                       inverse, xq, xsc, clen_rows, idx_stride, row_step, 0)
    }

    /// W2-MROWS-TP: [`Self::sparse_attn_orope`] with an explicit `q`/`out` row pitch.
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attn_orope_pitched(
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
        // W2-MROWS-TP: the `q`/`out` row pitch as in [`Self::sparse_attn`].
        // Non-zero dispatches to `dsv41_sparse_attn_orope_rp`; an `.so` without
        // that symbol simply declines (`Ok(false)`), which the caller's m-row arm
        // answers by falling back to the plain `sparse_attn` entry.
        row_pitch: i32,
    ) -> Result<bool> {
        // The two symbols have DIFFERENT signatures (the `_rp` one carries the
        // pitch), so they cannot share a `match`; dispatch on the pitch first.
        let (rc, what) = if row_pitch != 0 {
            let Some(f) = self.kernels.sparse_attn_orope_rp else {
                return Ok(false);
            };
            (
                unsafe {
                    f(q, kv, sink, idxs, out, b, m, h, d, clen, window, index_topk, scale, cos,
                      sin, base, rope_rd, half, mul, off, step, inverse as i32, xq, xsc,
                      clen_rows, idx_stride, row_step, row_pitch, self.stream)
                },
                "dsv41_sparse_attn_orope_rp",
            )
        } else {
            let Some(f) = self.kernels.sparse_attn_orope else {
                return Ok(false);
            };
            (
                unsafe {
                    f(q, kv, sink, idxs, out, b, m, h, d, clen, window, index_topk, scale, cos,
                      sin, base, rope_rd, half, mul, off, step, inverse as i32, xq, xsc,
                      clen_rows, idx_stride, row_step, self.stream)
                },
                "dsv41_sparse_attn_orope",
            )
        };
        // 1/2/3 are the decline sentinels (see the C launcher); anything else is
        // a real launch error.
        if (1..=3).contains(&rc) {
            return Ok(false);
        }
        self.kerr(rc, what)?;
        Ok(true)
    }

    /// W2-MROWS-TP: whether the loaded `.so` carries `dsv41_sparse_attn_rp`, the
    /// explicit-row-pitch entry an `m`-row batch needs at `world > 1`. The
    /// attention gate asks this and declines the arm when it is absent, so an
    /// `.so` that predates the TP work keeps the per-row launch sequence
    /// (the pre-TP behaviour) instead of failing.
    pub fn supports_sparse_attn_rp(&self) -> bool {
        self.kernels.sparse_attn_rp.is_some()
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

    /// F8 (`DSV41_F8_TOPK_ROWS`): is the block-wide selection in this `.so`?
    ///
    /// The arm must test this — the folded call hands the entry a per-row `lens`
    /// array and an explicit output pitch, and a stale `.so` would take neither.
    pub fn supports_indexer_topk_rows(&self) -> bool {
        self.kernels.indexer_topk_rows.is_some()
    }

    /// [`Self::indexer_topk`] for a WHOLE block: `b = 1, m = rows`, a per-row
    /// `compress_lens` array, and `out_stride` as the output row pitch.
    ///
    /// Every row of the launch is the independent selection its own `m = 1` call
    /// was: `lens[mm]` bounds row `mm`, the kernel's ceiling is the MAXIMUM of the
    /// rows (so no row is clamped to row 0's count), and row `mm`'s `min(topk, cl)`
    /// picks land at `out + mm * out_stride + i` — the address the per-row call
    /// passed as its `out` base (see the C entry's header for why both had to
    /// become explicit parameters).
    ///
    /// `uses_candidates` must be false (the mask's row pitch is the launch ceiling);
    /// the C entry declines it rather than guessing.
    pub fn indexer_topk_rows(
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
        out_stride: i32,
        softmax_scale: f32,
        head_scale: f32,
        uses_candidates: bool,
    ) -> Result<()> {
        let f = self.need(self.kernels.indexer_topk_rows, "dsv41_indexer_topk_rows")?;
        let rc = unsafe {
            f(
                q,
                index_k,
                weights,
                candidates,
                compress_lens,
                out,
                b,
                m,
                nh,
                hd,
                n_pos,
                topk,
                offset,
                out_stride,
                softmax_scale,
                head_scale,
                uses_candidates as i32,
                self.stream,
            )
        };
        self.kerr(rc, "dsv41_indexer_topk_rows")
    }
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

    /// D2 EPOCH PAD (`DSV41_SWALLOW_EPOCH_PAD`) — ONE launch that advances this
    /// rank's v5 epoch by `pad` EMPTY rounds, so the arm that DROPS `step_dev`
    /// (the swallowed arm) still pays the same number of v5 rounds per step as
    /// the arm that runs it (legacy/aligned): `legacy == aligned == 165`,
    /// `swallowed == 84`, and the missing `81 = 2 * n_layers + 1` is exactly what
    /// this call adds back. See
    /// `docs/agent/swallow-fix9-round-ledger-design.md` §3.2 and the kernel's
    /// own note (`dsv41_v5_epoch_pad_kernel`).
    ///
    /// `ready_tbl` is the `[world]` peer ready-row array (`peer_stamps_u32()`);
    /// the kernel stamps `ready_tbl[r][rank] = e + pad` for every peer `r` and
    /// advances the A4 broadcast word `epoch + 1` with the same value. No staging
    /// is written, because a padded round has no payload and no reader.
    ///
    /// `Ok(false)` when the loaded `.so` predates the symbol. This is the ONE
    /// silent-skip arm and the caller reports it once (`ar5-hang`'s history is a
    /// string of fixes that were believed to run and did not) — every other
    /// non-zero return is a REAL launch error and is answered as one. Note the
    /// `rc == 1` "declined" convention the argmax entries use is deliberately NOT
    /// reused here: this entry has no decline condition, and swallowing a genuine
    /// error as "declined" would rebuild exactly the phantom-fix failure mode.
    pub fn v5_epoch_pad(
        &self,
        ready_tbl: *const *mut u32,
        epoch: *mut c_uint,
        world: i32,
        rank: i32,
        pad: u32,
    ) -> Result<bool> {
        let f = match self.kernels.v5_epoch_pad {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe { f(ready_tbl, epoch, world, rank, pad, self.stream) };
        self.kerr(rc, "dsv41_v5_epoch_pad")?;
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

    /// DEVICE-SIDE `moe_align` (`dsv41_moe_align_from_group`): project the
    /// TileLang MoE arm's three segment tables (`order` / `counts_seg` / `eid`)
    /// out of [`Self::route_group`]'s device output.
    ///
    /// Every pointer is a DEVICE buffer and `route_group` must have run on the
    /// same stream first; `n_active` is `grp_nactive` (the `[1]` i32 live length),
    /// `counts_by_e`/`starts`/`gather_src` are the group tables, and `nseg_out`
    /// receives `min(*n_active, seg_cap)`. The tables are **bit-for-bit
    /// `moe_align_host`** (the kernel's header comment carries the equivalence
    /// argument), which is what makes this a drop-in replacement for the host
    /// version — minus the D2H read that made the arm eager-only.
    ///
    /// Returns `Ok(false)` on `rc == 2` (declined: a contract break in
    /// `seg_cap`/`bm`), an `Err` on any other non-zero rc — the shim rc contract.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_align_from_group(
        &self,
        active: *const i32,
        n_active: *const i32,
        counts_by_e: *const i32,
        starts: *const i32,
        gather_src: *const i32,
        order: *mut i32,
        counts_seg: *mut i32,
        eid: *mut i32,
        nseg_out: *mut i32,
        seg_cap: i32,
        bm: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.moe_align_from_group else {
            return Ok(false);
        };
        let rc = unsafe {
            f(
                active, n_active, counts_by_e, starts, gather_src, order, counts_seg, eid,
                nseg_out, seg_cap, bm, self.stream,
            )
        };
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_moe_align_from_group")?;
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

    /// DSV41_SEQ_ALIGN (#5/#13) as a kernel-argument value, 0 or 1. The gate
    /// itself caches the env lookup (one `getenv` per process, never per call —
    /// a per-call getenv is a CUDA-graph capture hazard and a hot-path slip).
    /// Read here so every swiglu/down launch carries the SAME flag without
    /// touching each call site.
    fn seq_align_i32(&self) -> c_int {
        if crate::dsv41::chain_dev::seq_align() {
            1
        } else {
            0
        }
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
        let rc = unsafe { f(gate_up, rows, inter, limit, self.seq_align_i32(), s) };
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
        let rc = unsafe { f(gate_up, rows, inter, limit, self.seq_align_i32(), xq, xsc, s) };
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

    /// TILELANG head bf16 GEMM (`DSV41_HEAD_TILELANG`, default OFF; see
    /// `head_tilelang()` in chain_dev.rs) — the K-split + M-pad-16 mma program
    /// from `kernels/cuda/tilelang_gen/head_bf16_shim.cu`, prototype in
    /// docs/agent/tilelang-attn-head.md.
    ///
    /// Deliberately the SAME argument list as [`Self::head_gemv_bf16_v1_mrows`] —
    /// `(w, x, out, m=rows, n, k)` — so the arm is that entry's drop-in: `w` is
    /// the `[n, k]` bf16 head weight (or its per-rank slice), `x` the `[m, k]`
    /// f32 activation (`xn_r`), `out` the `[m, n]` f32 logits (`logits_r`, whose
    /// row stride MUST be `n`).
    ///
    /// ⚠️ A/B ARM (NOT a production replacement). The head's real program is
    /// "bf16 weight x f32 activation" (f32 FMA); tensor-core bf16 mma has no
    /// such form, so the shim CASTS the activation to bf16 — a ~2.3e-2 max_rel
    /// change that is far above the argmax near-tie flip threshold (see the
    /// `Kernels::head_bf16_tilelang` field). Its purpose is to MEASURE the
    /// M-in-tile win (prototype: M6/M1 = 1.00 vs the v1 fold's 3.23-3.81, and
    /// 2.93x at m=6), not to be switched on in production.
    ///
    /// `Ok(false)` means NOT performed — keep the per-row / v1-fold path: either
    /// the loaded `.so` has no `dsv41_head_bf16_tilelang` (a stale build, or the
    /// frozen dumps were not AOT-generated yet — see
    /// `kernels/tilelang/gen_head_aot.py`), or the C entry DECLINED (it returns
    /// 2, never cudaErrorInvalidValue, so a decline can never read as a launch
    /// failure). The C decline set is: `n != 16160` / `k != 5120` / `rows`
    /// outside 1..=8 / a base that is not 16B-aligned / an INIT failure (the
    /// resident scratch could not be allocated) / a call inside a CUDA-graph
    /// capture BEFORE the one-time INIT has completed (the shim's capture guard,
    /// the verify-graph-tl-audit P0-2 red line). Every other non-zero rc is a
    /// real cuda error.
    pub fn head_bf16_tilelang(
        &self,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        rows: i32,
        n: i32,
        k: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.head_bf16_tilelang else {
            return Ok(false);
        };
        if !(1..=8).contains(&rows) {
            return Ok(false);
        }
        let rc = unsafe { f(w, x, out, rows, n, k, self.stream) };
        // 2 = the shim's "declined" (shape / alignment / INIT / capture), the
        // caller falls back. Every other non-zero rc is a real cuda error.
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_head_bf16_tilelang")?;
        Ok(true)
    }

    /// True when the loaded `.so` carries the TileLang head entry
    /// (`dsv41_head_bf16_tilelang`). A `.so` built before the AOT dumps existed
    /// (or on which `gen_head_aot.py` has not been run — the shim's
    /// `__has_include` guard then compiles an empty TU) has no such symbol, so
    /// `DSV41_HEAD_TILELANG` must stay OFF and be reported instead of silently
    /// measuring the OLD path (the project's #1 measurement-bias trap).
    pub fn supports_head_tilelang(&self) -> bool {
        self.kernels.head_bf16_tilelang.is_some()
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

    /// PROJ-MMA: the TENSOR-CORE program for the same call shape as
    /// [`Self::gemm_fp8_mrows`] — `dsv41_gemm_fp8_mrows_mma` /
    /// `gemm_fp8_mrows_mma_kernel<M>` (see the `Kernels::gemm_fp8_mrows_mma`
    /// field for the mapping and the numerics contract, and
    /// docs/agent/tensorcore-proj-design.md §3).
    ///
    /// Deliberately the SAME argument list as `gemm_fp8_mrows`, so a caller can
    /// swap the program without touching its activation staging: `a` [rows, k]
    /// fp8 e4m3, `a_scale` [rows, k/32] f32, `out` f32 with row `r` at
    /// `+r*out_stride` (`out_stride` a separate parameter, for the wq_b site
    /// whose row is `nh*head_dim` wide while this rank writes `nlh*head_dim`).
    ///
    /// The K-split scratch (`partial` [ks][M][n], `ctr` [n/16]) is NOT wired on
    /// this side yet — round 1 is `ks == 1`, which needs none — so this wrapper
    /// passes both null and `pmma_n = n`. A K-split the C entry resolves above 1
    /// then DECLINES (returns 2) instead of touching scratch, and the caller
    /// keeps its `gemm_fp8_mrows` readout. Arming therefore requires
    /// `DSV41_PROJ_MMA=1` AND a ks that resolves to 1 (`DSV41_PROJ_MMA_KS=1`)
    /// until the `[ks][M][n]` sizing lands.
    ///
    /// `Ok(false)` means NOT performed — either the loaded .so predates the
    /// symbol (PROJ-MMA TU absent from the build) or the C entry declined (it
    /// returns 2, never cudaErrorInvalidValue, so a decline can never read as a
    /// launch failure).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_mrows_mma(
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
        let Some(f) = self.kernels.gemm_fp8_mrows_mma else {
            return Ok(false);
        };
        if !(1..=8).contains(&rows) {
            return Ok(false);
        }
        let rc = unsafe {
            f(
                a,
                a_scale,
                w,
                w_scale,
                bias,
                out,
                rows,
                n,
                k,
                out_stride,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                n, // pmma_n: no scratch claim — the C entry only proceeds at ks == 1
                self.stream,
            )
        };
        // 2 = the kernel's "declined" (shape / mode / scratch), the caller falls back.
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_mrows_mma")?;
        Ok(true)
    }

    /// True when the loaded .so carries the PROJ-MMA entry
    /// (`dsv41_gemm_fp8_mrows_mma`, compiled in by default from the PROJ-MMA TU
    /// — see `kernels/cuda/build.sh`). A stale .so that predates the TU leaves
    /// the projection sites on `gemm_fp8_mrows` (MPAR > legacy), which is the
    /// program they were verified against.
    pub fn supports_gemm_fp8_mrows_mma(&self) -> bool {
        self.kernels.gemm_fp8_mrows_mma.is_some()
    }

    /// TILELANG: the generated wkv-shape fp8 projection GEMM
    /// (`dsv41_gemm_fp8_tilelang_wkv`, from `kernels/cuda/tilelang_gen/wkv_shim.cu`) —
    /// the multi-row GEMM whose M rides inside the mma tile, so m=1 and m=6 cost the
    /// same. See the `Kernels::gemm_fp8_tilelang_wkv` field for the mapping, the
    /// (b′) program-consistent parity contract and the double-swap requirement.
    ///
    /// Deliberately the SAME argument list as `gemm_fp8_mrows`, so a caller swaps
    /// the program without touching its activation staging: `a` [rows, k] fp8 e4m3,
    /// `a_scale` [rows, k/32] f32, `out` f32 with row `r` at `+r*out_stride`.
    ///
    /// FIRST PHASE = THE wkv SHAPE ONLY (n=512, k=5120). The generated geometry is
    /// baked into the grid/index arithmetic, so the C entry declines every other
    /// shape; `Ok(false)` here therefore means exactly "this projection is not the
    /// generated shape" for the wq_a / wq_b / wo_b sites, and "NOT performed" for
    /// wkv. Either way the caller keeps its existing kernel.
    ///
    /// `Ok(false)` also covers a stale `.so` without the symbol (the TILELANG TU
    /// absent from the build) and the C entry's declines (it returns 2, never
    /// cudaErrorInvalidValue, so a decline can never read as a launch failure).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_tilelang_wkv(
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
        let Some(f) = self.kernels.gemm_fp8_tilelang_wkv else {
            return Ok(false);
        };
        if !(1..=8).contains(&rows) {
            return Ok(false);
        }
        let rc = unsafe {
            f(
                a,
                a_scale,
                w,
                w_scale,
                bias,
                out,
                rows,
                n,
                k,
                out_stride,
                self.stream,
            )
        };
        // 2 = the shim's "declined" (shape / bias / alignment / INIT failure), the
        // caller falls back. Every other non-zero rc is a real cuda error.
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_tilelang_wkv")?;
        Ok(true)
    }

    /// True when the loaded .so carries the TileLang wkv entry
    /// (`dsv41_gemm_fp8_tilelang_wkv`, compiled in from `kernels/cuda/tilelang_gen/`
    /// — see `kernels/cuda/build.sh`). A stale .so that predates the TU leaves the
    /// projection sites on their existing kernels.
    pub fn supports_gemm_fp8_tilelang_wkv(&self) -> bool {
        self.kernels.gemm_fp8_tilelang_wkv.is_some()
    }

    /// Helper: issue one of the PHASE-2 dense TileLang entries (`symbol` labels the
    /// entry for the error path). Same contract and argument list as
    /// [`Self::gemm_fp8_tilelang_wkv`] — the three new shapes differ only in the
    /// frozen geometry baked into their own dumps, so the wrapper body is shared
    /// and the per-shape entry point just picks the `Option<fn>`. `Ok(false)` means
    /// "declined / not performed" (stale `.so` or the C shape gate), never a
    /// failure.
    #[allow(clippy::too_many_arguments)]
    fn gemm_fp8_tilelang_dense_2(
        &self,
        f: Option<
            unsafe extern "C" fn(
                *const u8,
                *const f32,
                *const u8,
                *const u8,
                *const f32,
                *mut f32,
                c_int,
                c_int,
                c_int,
                c_int,
                CuStream,
            ) -> c_int,
        >,
        symbol: &str,
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
        let Some(f) = f else {
            return Ok(false);
        };
        if !(1..=8).contains(&rows) {
            return Ok(false);
        }
        let rc = unsafe {
            f(
                a,
                a_scale,
                w,
                w_scale,
                bias,
                out,
                rows,
                n,
                k,
                out_stride,
                self.stream,
            )
        };
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, symbol)?;
        Ok(true)
    }

    /// TILELANG (phase 2): the wq_a shape (`n=1280, k=5120`) of the mma-tile fp8
    /// projection GEMM, from `kernels/cuda/tilelang_gen/wq_a_shim.cu`. The
    /// generated `out` row stride (`OS`) is `q_lora_rank == n`, so the C entry
    /// declines any other `out_stride`. See [`Self::gemm_fp8_tilelang_wkv`] for the
    /// double-swap contract; this wrapper only differs by symbol.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_tilelang_wq_a(
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
        self.gemm_fp8_tilelang_dense_2(
            self.kernels.gemm_fp8_tilelang_wq_a,
            "dsv41_gemm_fp8_tilelang_wq_a",
            a,
            a_scale,
            w,
            w_scale,
            bias,
            out,
            rows,
            n,
            k,
            out_stride,
        )
    }

    /// TILELANG (phase 2): the wq_b shape (`n=4096, k=1280`), from
    /// `kernels/cuda/tilelang_gen/wq_b_shim.cu`. ⚠️ Its generated `OS` is
    /// `nh*head_dim = 32768`, NOT `n = nlh*head_dim = 4096` (ColumnParallel: the
    /// rank fills only the leading `nlh*hd` of a full row), so the C entry declines
    /// the indexer site (`idx_wq_b`, `out_stride == n`). See
    /// [`Self::gemm_fp8_tilelang_wkv`] for the double-swap contract.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_tilelang_wq_b(
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
        self.gemm_fp8_tilelang_dense_2(
            self.kernels.gemm_fp8_tilelang_wq_b,
            "dsv41_gemm_fp8_tilelang_wq_b",
            a,
            a_scale,
            w,
            w_scale,
            bias,
            out,
            rows,
            n,
            k,
            out_stride,
        )
    }

    /// TILELANG (phase 2): the wo_b shape (`n=5120, k=1024`), from
    /// `kernels/cuda/tilelang_gen/wo_b_shim.cu`. Its generated `OS` is `dim == n`.
    /// See [`Self::gemm_fp8_tilelang_wkv`] for the double-swap contract.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_tilelang_wo_b(
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
        self.gemm_fp8_tilelang_dense_2(
            self.kernels.gemm_fp8_tilelang_wo_b,
            "dsv41_gemm_fp8_tilelang_wo_b",
            a,
            a_scale,
            w,
            w_scale,
            bias,
            out,
            rows,
            n,
            k,
            out_stride,
        )
    }

    /// TILELANG (phase 2): the GROUPED wo_a shape, from
    /// `kernels/cuda/tilelang_gen/wo_a_shim.cu`. Same argument list as
    /// [`Self::wo_a_grouped_fp8`], because it IS that call site's program in
    /// TileLang form (quantisation stays outside, on the caller's side).
    ///
    /// The C entry accepts only the two frozen variants — `(groups == 1,
    /// a_stride == 4096)` (verify@TP8, where `a_stride` degenerates to `k`) and
    /// `(groups == 8, a_stride == 32768)` (the TP1 form) — with
    /// `n == 1024, k == 4096, out_stride == 8192`, and declines (2) everything
    /// else, including a non-null `bias`. `Ok(false)` therefore means "keep the
    /// per-(group, row) loop", never a performance claim and never a failure.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_tilelang_wo_a(
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
        let Some(f) = self.kernels.gemm_fp8_tilelang_wo_a else {
            return Ok(false);
        };
        if !(1..=8).contains(&rows) {
            return Ok(false);
        }
        let rc = unsafe {
            f(
                a,
                a_scale,
                w,
                w_scale,
                bias,
                out,
                groups,
                rows,
                n,
                k,
                a_stride,
                out_stride,
                self.stream,
            )
        };
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_tilelang_wo_a")?;
        Ok(true)
    }

    /// True when the loaded .so carries the TileLang wq_a entry (phase 2 dense).
    pub fn supports_gemm_fp8_tilelang_wq_a(&self) -> bool {
        self.kernels.gemm_fp8_tilelang_wq_a.is_some()
    }

    /// True when the loaded .so carries the TileLang wq_b entry (phase 2 dense).
    pub fn supports_gemm_fp8_tilelang_wq_b(&self) -> bool {
        self.kernels.gemm_fp8_tilelang_wq_b.is_some()
    }

    /// True when the loaded .so carries the TileLang wo_b entry (phase 2 dense).
    pub fn supports_gemm_fp8_tilelang_wo_b(&self) -> bool {
        self.kernels.gemm_fp8_tilelang_wo_b.is_some()
    }

    /// True when the loaded .so carries the TileLang grouped wo_a entry (phase 2).
    pub fn supports_gemm_fp8_tilelang_wo_a(&self) -> bool {
        self.kernels.gemm_fp8_tilelang_wo_a.is_some()
    }

    /// K2: the verify's q-chain tail in ONE launch — rmsnorm(`qr_raw`) + its fp8
    /// quantisation + the wq_b multi-row fp8 GEMV + the RoPE of the result.
    ///
    /// Each of the four segments is reproduced from the kernel the verify path
    /// runs TODAY there (`dsv41_rmsnorm_rows_kernel`, `quant_kernel<0>`,
    /// `gemm_fp8_mrows_kernel<M>`, `apply_rope_mrows_kernel`), so the fused
    /// launch is the same program as the four-launch sequence, per element (the
    /// kernel header in `dsv41_kernels.cu` carries the per-segment argument).
    /// This is deliberately NOT `lin_rope_norm`'s family: R2's corruption came
    /// from re-pointing the `mrows` chain at EAGER `m = 1` programs and from
    /// reading the position out of `*pos_ctr`; here no EAGER program takes part
    /// and `pos_rows` is an explicit device array (no `pos_ctr`/`mul`/`off`/`step`
    /// exists in the kernel).
    ///
    /// `qr_norm_out` receives the normalised rows and is where the caller decides
    /// the coupling: `Some(qr_r)` (the shipped call) leaves `qr_r` normalised —
    /// byte for byte what `norm_rows` left there — so the indexer's q half needs
    /// no RAW flag; `None` leaves `qr_raw` untouched and the caller owes the norm.
    ///
    /// `Ok(false)` means NOT performed — keep `norm_rows + quant_rows +
    /// proj_mrows + apply_rope_mrows`: either the loaded .so predates the symbol,
    /// or the C entry declined (it returns 2, never cudaErrorInvalidValue, so a
    /// decline can never read as a launch failure). The `1..=8` bound and the
    /// `m <= VERIFY_ROWS` staging bound are re-checked here, exactly as in
    /// [`Self::gemm_fp8_mrows`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_mrows_rope_norm(
        &self,
        qr_raw: *const f32,
        qr_w: *const f32,
        qr_eps: f32,
        qr_norm_out: *mut f32,
        w: *const u8,
        w_scale: *const u8,
        bias: *const f32,
        out: *mut f32,
        rows: i32,
        n: i32,
        k: i32,
        out_stride: i32,
        rope_cos: *const f32,
        rope_sin: *const f32,
        pos_rows: *const c_int,
        rope_rd: i32,
        rope_hd: i32,
        rope_inverse: bool,
    ) -> Result<bool> {
        let Some(f) = self.kernels.gemm_fp8_mrows_rope_norm else {
            return Ok(false);
        };
        if !(1..=8).contains(&rows) {
            return Ok(false);
        }
        let rc = unsafe {
            f(
                qr_raw,
                qr_w,
                qr_eps,
                qr_norm_out,
                w,
                w_scale,
                bias,
                out,
                rows,
                n,
                k,
                out_stride,
                rope_cos,
                rope_sin,
                pos_rows,
                rope_rd,
                rope_hd,
                rope_inverse as i32,
                self.stream,
            )
        };
        // 2 = the kernel's "declined" (shape/mode), the caller falls back.
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_mrows_rope_norm")?;
        Ok(true)
    }

    /// True when the loaded .so carries K2
    /// (`dsv41_gemm_fp8_mrows_rope_norm`). A stale .so leaves
    /// `DSV41_ATTN_MROWS_ROPE_NORM` inert and the four-launch sequence runs,
    /// which is the bit-exact reference K2 was written against.
    pub fn supports_gemm_fp8_mrows_rope_norm(&self) -> bool {
        self.kernels.gemm_fp8_mrows_rope_norm.is_some()
    }

    /// K1: the two-family form of [`Self::gemm_fp8_mrows`] — two projections of
    /// the SAME activation row (the verify's wq_a + wkv) in ONE launch. Row `row`
    /// of family `f` is bit-identical to row `row` of the separate
    /// `gemm_fp8_mrows` launch for that family (the kernel header carries the
    /// C1-C6 argument and the warp-level family split).
    ///
    /// TWO output strides: the wq_a output row strides by `ql`, the wkv row by
    /// `hd`, and those differ — a single stride cannot address both buffers (the
    /// one ABI deviation from the design doc's §3.2 signature).
    ///
    /// `Ok(false)` means NOT performed — the caller runs its two
    /// `gemm_fp8_mrows` launches: either the loaded .so predates the symbol, or
    /// the C entry declined (it returns 2, never cudaErrorInvalidValue, so a
    /// decline can never read as a launch failure). `rows` outside 1..=8 is
    /// refused here as well, the same bound the template dispatch covers.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_mrows2(
        &self,
        a: *const u8,
        a_scale: *const f32,
        w1: *const u8,
        w1_scale: *const u8,
        out1: *mut f32,
        n1: i32,
        out_stride1: i32,
        w2: *const u8,
        w2_scale: *const u8,
        out2: *mut f32,
        n2: i32,
        out_stride2: i32,
        rows: i32,
        k: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.gemm_fp8_mrows2 else {
            return Ok(false);
        };
        if !(1..=8).contains(&rows) {
            return Ok(false);
        }
        // `bias` is null for both families (the projection sites have none); it is
        // a parameter only so the epilogue's `acc + (bias ? bias[rrow] : 0.f)` is
        // the very same expression the single-family kernel emits.
        let rc = unsafe {
            f(
                a,
                a_scale,
                w1,
                w1_scale,
                std::ptr::null(),
                out1,
                n1,
                out_stride1,
                w2,
                w2_scale,
                std::ptr::null(),
                out2,
                n2,
                out_stride2,
                rows,
                k,
                self.stream,
            )
        };
        // 2 = the kernel's "declined" (shape/mode), the caller falls back.
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_mrows2")?;
        Ok(true)
    }

    /// True when the loaded .so carries K1 (`dsv41_gemm_fp8_mrows2`). A stale
    /// .so leaves the wq_a / wkv pair on its two `proj_mrows` launches, which are
    /// the bit-exact reference K1 is verified against.
    pub fn supports_gemm_fp8_mrows2(&self) -> bool {
        self.kernels.gemm_fp8_mrows2.is_some()
    }

    /// B6: the MULTI-ROW form of the f32-activation GEMV — fp8 e4m3 weights x
    /// RAW f32 activations, so the verify's wo_b no longer needs the `m x
    /// quant_fp8` round trip in front of its `proj_mrows`.
    ///
    /// Row r of this launch is BIT-IDENTICAL to the M=1
    /// [`Self::gemm_fp8_mx_f32`] call of row r — the f32 domain's
    /// "materialisation" is the identity (`s_af[i] = a_f32[i]`, a pure copy),
    /// and this kernel emits the same `acc += a_f32[j] * (s_lut[w[j]] * sb)`
    /// serial chain, the same `kb`-ascending `j = kb*32 + lane` walk and the
    /// same `shfl_xor` tree. It is NOT bit-identical to the OLD verify path it
    /// replaces (`m x quant_fp8 + proj_mrows`): skipping the fp8 round trip is
    /// strictly more accurate (`DSV41_WOB_F32`'s lever), so the acceptance is
    /// the red line, not a memcmp.
    ///
    /// `a_stride` is the activation row PITCH in f32 elements — an explicit
    /// contract, because the verify's `wo_r` is [m, ol_total] while k =
    /// ol_local (8x apart under TP8) and the pitch therefore cannot be derived
    /// from `k`. The C entry declines `a_stride < k` rather than reading row 0's
    /// tail (verify-value-hunt root cause F1/F2).
    ///
    /// `Ok(false)` means NOT performed — the caller keeps its
    /// `quant_fp8 + proj_mrows` block and then its per-row loop: either the
    /// loaded .so predates the symbol, or the C entry declined (it returns 2,
    /// never cudaErrorInvalidValue, so a decline can never read as a launch
    /// failure). `rows` outside 1..=8 is refused here as well, the bound the
    /// template dispatch covers.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_mrows_f32(
        &self,
        a_f32: *const f32,
        w: *const u8,
        w_scale: *const u8,
        bias: *const f32,
        out: *mut f32,
        rows: i32,
        n: i32,
        k: i32,
        a_stride: i32,
        out_stride: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.gemm_fp8_mrows_f32 else {
            return Ok(false);
        };
        if !(1..=8).contains(&rows) {
            return Ok(false);
        }
        let rc = unsafe {
            f(
                a_f32,
                w,
                w_scale,
                bias,
                out,
                rows,
                n,
                k,
                a_stride,
                out_stride,
                self.stream,
            )
        };
        // 2 = the kernel's "declined" (shape/mode), the caller falls back.
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_mrows_f32")?;
        Ok(true)
    }

    /// True when the loaded .so carries B6 (`dsv41_gemm_fp8_mrows_f32`). A stale
    /// .so leaves `DSV41_VERIFY_WOB_MROWS_F32` inert and the verify's wo_b on the
    /// `quant_fp8 + proj_mrows` pair, which is the reference B6 was written
    /// against.
    pub fn supports_gemm_fp8_mrows_f32(&self) -> bool {
        self.kernels.gemm_fp8_mrows_f32.is_some()
    }

    /// WO_PAIR-ROWS: the verify's `wo_a_grouped -> quant -> wo_b` chain with the
    /// INTERMEDIATE QUANT folded into the wo_b launch (`gemm_fp8_mtile_kernel<M,1>`
    /// -- see the `Kernels::gemm_fp8_mrows_q_f32` field for the numerics contract).
    /// `a_f32` is the RAW `wo_r` matrix (`rows` x `a_stride` ELEMENTS, the `k`
    /// contracted columns at the start of each row); `out` is the wo_b result.
    ///
    /// `Ok(false)` means NOT performed -- keep the `m x quant_fp8 + proj_mrows`
    /// block: either the loaded .so predates the symbol, or the C entry declined
    /// (it returns 2, never cudaErrorInvalidValue, so a decline can never read as a
    /// launch failure). `rows` outside 1..=8 is refused here as well, the bound the
    /// template dispatch covers.
    ///
    /// ABI: the `gemm_fp8_mrows_f32` argument order with `a_stride` moved next to
    /// `a` (the activation's own pitch belongs with the activation).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_mrows_q_f32(
        &self,
        a_f32: *const f32,
        a_stride: i32,
        w: *const u8,
        w_scale: *const u8,
        bias: *const f32,
        out: *mut f32,
        rows: i32,
        n: i32,
        k: i32,
        out_stride: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.gemm_fp8_mrows_q_f32 else {
            return Ok(false);
        };
        if !(1..=8).contains(&rows) {
            return Ok(false);
        }
        let rc = unsafe {
            f(
                a_f32,
                a_stride,
                w,
                w_scale,
                bias,
                out,
                rows,
                n,
                k,
                out_stride,
                self.stream,
            )
        };
        // 2 = the kernel's "declined" (shape/mode), the caller falls back.
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_mrows_q_f32")?;
        Ok(true)
    }

    /// True when the loaded .so carries WO_PAIR-ROWS
    /// (`dsv41_gemm_fp8_mrows_q_f32`). A stale .so leaves
    /// `DSV41_VERIFY_WO_PAIR_MROWS` inert and the verify's wo_b on the
    /// `m x quant_fp8 + proj_mrows` block, which is the reference WO_PAIR-ROWS was
    /// written against.
    pub fn supports_gemm_fp8_mrows_q_f32(&self) -> bool {
        self.kernels.gemm_fp8_mrows_q_f32.is_some()
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

    /// B5 (DSV41_GATE_MROWS_ROUTE, default OFF): [`Self::gemv_bf16_v2_mrows`]
    /// with the MoE route folded into the GEMV's last block — ONE launch where
    /// the gate GEMV + `route_topk` used to be two (`ferrite_gemv_bf16_v2_mrows_route`).
    ///
    /// The GEMV half is the SAME `gemv_bf16_nt_kernel<NT, WPR>` program
    /// `gemv_bf16_v2_mrows` names (the .so holds one definition — see that
    /// entry's "NO SECOND TRANSCRIPTION" rule), so every gate score is
    /// bit-identical to the two-launch path; the elected block then runs
    /// `dsv41_route_topk`'s body row for row over the [rows, n] score block, so
    /// `weights`/`indices` are bit-identical too.
    ///
    /// `Ok(false)` = NOT performed, keep `gemv_bf16_v2_mrows` + `route_topk`:
    /// the .so predates the symbol, the per-row path would not take v2
    /// (`gemv_bf16_v2_wanted(n)`, `k % 8 != 0` — v1 is a DIFFERENT accumulation
    /// order, so there is no parity to claim), `rows` is outside 1..=8, `topk`
    /// is outside [1, n], or the route scratch would exceed 48 KB. The checks
    /// mirror the C entry's decline set exactly, so an armed gate never silently
    /// measures the old path with this returning `Ok(true)`.
    ///
    /// `ctr` is `s.route_ctr` (4B, zeroed ONCE at allocation): the kernel
    /// resets it in place before its grid ends, which is what makes a captured
    /// graph replay correct without a per-call memset.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_bf16_v2_mrows_route(
        &self,
        w: *const c_void,
        x: *const f32,
        out: *mut f32,
        rows: i32,
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
        // The same preconditions `gemv_bf16_v2_mrows` folds on, plus the route's
        // own shape/smem bounds — all of them also declined by the C entry, so
        // this never has to classify a cudaError_t.
        if !gemv_bf16_v2_wanted(n) || self.kernels.gemv_bf16_v2.is_none() {
            return Ok(false);
        }
        if !(1..=GEMV_V2_MROWS_MAX).contains(&rows) || (k & 7) != 0 {
            return Ok(false);
        }
        if n <= 0 || topk <= 0 || topk > n || ctr.is_null() {
            return Ok(false);
        }
        // Route scratch: [n] act + [n] selection scores + [topk] picks.
        if (n as usize) * 2 * std::mem::size_of::<f32>() + (topk as usize) * std::mem::size_of::<i32>()
            > 48 * 1024
        {
            return Ok(false);
        }
        // `FERRITE_GEMV_SKIP` is a timing-only ablation INSIDE the entry:
        // treating it as "do not fold" keeps the ablation honest.
        if gemv_bf16_nt_skip() {
            return Ok(false);
        }
        let Some(f) = self.kernels.gemv_bf16_v2_mrows_route else {
            return Ok(false); // older .so: keep the two-launch pair
        };
        // ABI: (x, w, bias, out, in_f=k, out_f=n, nrows=rows, route out/bias/params, ctr, s).
        let rc = unsafe {
            f(
                x,
                w,
                std::ptr::null(),
                out,
                k,
                n,
                rows,
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
        self.kerr(rc, "ferrite_gemv_bf16_v2_mrows_route")?;
        Ok(true)
    }

    /// True when the loaded .so carries B5
    /// (`ferrite_gemv_bf16_v2_mrows_route`). A stale .so leaves
    /// `DSV41_GATE_MROWS_ROUTE` inert and the verify's routing on the
    /// `gemv_bf16_v2_mrows` + `route_topk` pair, which is the reference B5 was
    /// written against.
    pub fn supports_gemv_bf16_v2_mrows_route(&self) -> bool {
        self.kernels.gemv_bf16_v2_mrows_route.is_some()
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

    /// [`Self::p2p_ar_v5`] with the A0 probe's site bucket selected by `site`
    /// (`ferrite_p2p_ar_v5_attn` / `_moe`, A0 site split). Same call, same work,
    /// same bits: the label only picks the `[ar-probe]` counter slot, which is
    /// why a caller may select a bucket without a numerical gate.
    ///
    /// A `.so` that predates the labelled entry falls back to the plain one (the
    /// all-reduce still runs; the probe just reports it under `Other`), so the
    /// switch is free both ways.
    #[allow(clippy::too_many_arguments)]
    pub fn p2p_ar_v5_site(
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
        site: ArV5Site,
    ) -> Result<()> {
        let (labelled, name) = match site {
            ArV5Site::Attn => (self.kernels.p2p_ar_v5_attn, "ferrite_p2p_ar_v5_attn"),
            ArV5Site::Moe => (self.kernels.p2p_ar_v5_moe, "ferrite_p2p_ar_v5_moe"),
            ArV5Site::Other => (None, "ferrite_p2p_ar_v5"),
        };
        let (f, name) = match labelled {
            Some(f) => (f, name),
            None => (
                self.need(self.kernels.p2p_ar_v5, "ferrite_p2p_ar_v5")?,
                "ferrite_p2p_ar_v5",
            ),
        };
        let rc = unsafe {
            f(partial, staging_tbl, ready_tbl, epoch, staging_local, ready_local, out, n,
              world, my_rank, stride, self.stream)
        };
        self.kerr(rc, name)
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

    /// [`Self::p2p_ar_pubred_v5`] with the A0 probe's site label set to the MoE
    /// site (`ferrite_p2p_ar_pubred_v5_moe`, A1a). Identical work, so the MoE
    /// half of the 80 rounds/step stays distinguishable from the attention half
    /// in the `[ar-probe]` lines.
    #[allow(clippy::too_many_arguments)]
    pub fn p2p_ar_pubred_v5_moe(
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
        let f = self.need(self.kernels.p2p_ar_pubred_v5_moe, "ferrite_p2p_ar_pubred_v5_moe")?;
        let rc = unsafe {
            f(ready_tbl, epoch, staging_local, ready_local, out, n, world, my_rank, stride,
              self.stream)
        };
        self.kerr(rc, "ferrite_p2p_ar_pubred_v5_moe")
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

    /// `ferrite_p2p_ar_pubred_v5_hcpost` (A1a): the publish+reduce AND the
    /// hc-post fold, with NO store — the payload was already carried by a
    /// producer's epilogue (see [`Self::moe_down_reduce_ar`] /
    /// [`Self::add_inplace_ar`]). Covers the ADD_EPI path too: the folded
    /// residual changes only what is PUBLISHED, so it is the carrier's business.
    ///
    /// `Ok(false)` on a stale .so or a declared shape mismatch (the launcher
    /// returns `cudaErrorInvalidValue`), so the caller falls back to the full
    /// `all_reduce_inplace_hcpost*` and the producer's store is simply redone.
    #[allow(clippy::too_many_arguments)]
    pub fn p2p_ar_pubred_v5_hcpost(
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
        hc_res: *mut f32,
        hc_post: *const f32,
        hc_comb: *const f32,
        hc_n: c_int,
        hc_h: c_int,
    ) -> Result<bool> {
        let f = match self.kernels.p2p_ar_pubred_v5_hcpost {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe {
            f(ready_tbl, epoch, staging_local, ready_local, out, n, world, my_rank, stride,
              hc_res, hc_post, hc_comb, hc_n, hc_h, self.stream)
        };
        self.kerr(rc, "ferrite_p2p_ar_pubred_v5_hcpost")?;
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

    /// RING-WIN-MROWS (`DSV41_RING_WIN_MROWS=1`): one launch for the whole verify
    /// block's ring append + per-row causal window indices - the rows form of
    /// [`Self::ring_win_fuse`] with an explicit `idxs` row stride. `pos_ctr` is
    /// the `pos_rows` array (row r's start position is `*pos_ctr + r`), `kv` is
    /// the contiguous `[m, hd]` block, and row r's indices land at
    /// `idxs + r * idx_stride` - the exact base the per-row call used
    /// (`idxs_r + r*ist`), so row r's bytes are identical to it.
    ///
    /// ⚠️ The CALLER must guarantee `*pos_ctr + m - 1 < window` (no ring
    /// turnover) - see the kernel header. `Ok(false)` means the `.so` lacks the
    /// symbol and the caller keeps the per-row path.
    #[allow(clippy::too_many_arguments)]
    pub fn ring_win_fuse_mrows(
        &self,
        ring: *mut f32,
        kv: *const f32,
        pos_ctr: *const c_int,
        window: i32,
        hd: i32,
        m: i32,
        idxs: *mut i32,
        idx_stride: i32,
    ) -> Result<bool> {
        let f = match self.kernels.ring_win_fuse_mrows {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe { f(ring, kv, pos_ctr, window, hd, m, idxs, idx_stride, self.stream) };
        self.kerr(rc, "dsv41_ring_win_fuse_mrows")?;
        Ok(true)
    }

    // -----------------------------------------------------------------------
    // SPARSE-ATTN-ROPE-MROWS (`DSV41_VERIFY_OROPE_MROWS=1`): the verify block's
    // fused attention + o-rope + fp8, correct in the steady state.
    // -----------------------------------------------------------------------

    /// Whether the loaded `.so` carries `dsv41_sparse_attn_orope_mrows` (and the
    /// two support entries). A stale `.so` answers `false` and the caller keeps
    /// the per-row `sparse_attn_orope` sequence, byte for byte.
    pub fn supports_sparse_attn_orope_mrows(&self) -> bool {
        self.kernels.sparse_attn_orope_mrows.is_some()
            && self.kernels.window_idxs_mrows.is_some()
            && self.kernels.ring_append_mrows.is_some()
    }

    /// SPARSE-ATTN-ROPE-MROWS: `n` fused m-row sparse-attention launches with the
    /// block's OWN KV rows (`kv_rows`, `[m, d]` at pitch `d`) substituted for
    /// every window slot whose position falls inside the block. `ring` is the
    /// attention's KV source (the owner's cache); the block's appends are DEFERRED
    /// and must be issued by [`Self::ring_append_mrows`] after this launch.
    ///
    /// `Ok(false)` = fall back (symbol absent, or the C launcher's shape /
    /// `b != 1` decline returned 2/3).
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attn_orope_mrows(
        &self,
        q: *const f32,
        kv: *const f32,
        kv_rows: *const f32,
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
        clen_rows: *const c_int,
        idx_stride: i32,
        row_step: i32,
        row_pitch: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.sparse_attn_orope_mrows else {
            return Ok(false);
        };
        let rc = unsafe {
            f(q, kv, kv_rows, sink, idxs, out, b, m, h, d, clen, window, index_topk, scale, cos,
              sin, base, rope_rd, half, mul, off, step, inverse as i32, xq, xsc, clen_rows,
              idx_stride, row_step, row_pitch, self.stream)
        };
        if (2..=3).contains(&rc) {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_sparse_attn_orope_mrows")?;
        Ok(true)
    }

    /// SPARSE-ATTN-ROPE-MROWS support: the block's `m` window-index rows in ONE
    /// launch, `window_idxs`'s decode branch per row (row r's position is
    /// `pos_rows[r]`, its output row at `idxs + r * idx_stride`). Byte-identical
    /// to `m` single-row `window_idxs` calls. `Ok(false)` = keep the per-row path.
    pub fn window_idxs_mrows(
        &self,
        idxs: *mut i32,
        pos_rows: *const c_int,
        window: i32,
        m: i32,
        idx_stride: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.window_idxs_mrows else {
            return Ok(false);
        };
        let rc = unsafe { f(idxs, pos_rows, window, m, idx_stride, self.stream) };
        // SENTINEL FIX (orope-hang-debug): rc==1 is a SHAPE-family decline from
        // the C entry — the per-row wrapper treats 1..=3 as Ok(false) (silent
        // fallback); this mrows wrapper must agree or the same shape that the
        // old arm tolerated becomes a hard abort here.
        if (1..=3).contains(&rc) {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_window_idxs_mrows")?;
        Ok(true)
    }

    /// SPARSE-ATTN-ROPE-MROWS support: the block's ring append, ONE launch for
    /// all `m` rows (`ring[slot(pos_rows[r])] = kv_rows[r]`). Issued AFTER the
    /// fused attention, which is what makes the batch correct. `Ok(false)` =
    /// keep the per-row append.
    pub fn ring_append_mrows(
        &self,
        ring: *mut f32,
        kv_rows: *const f32,
        pos_rows: *const c_int,
        window: i32,
        hd: i32,
        m: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.ring_append_mrows else {
            return Ok(false);
        };
        let rc = unsafe { f(ring, kv_rows, pos_rows, window, hd, m, self.stream) };
        // SENTINEL FIX (orope-hang-debug): same decline contract as above.
        if (1..=3).contains(&rc) {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_ring_append_mrows")?;
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

    /// F10 (`DSV41_F10_CLEN_BLOCK`): is the in-kernel counter snapshot in this `.so`?
    ///
    /// The arm MUST test this: the twin is a separate symbol exactly because a stale
    /// `dsv41_compress_commit` would accept (and ignore) the extra argument, leaving
    /// `clen_rows_r` at the previous step's counters — a wrong per-row bound for the
    /// whole block rather than a missing optimisation.
    pub fn supports_compress_commit_rows(&self) -> bool {
        self.kernels.compress_commit_rows.is_some()
    }

    /// [`Self::compress_commit_on`] with the row's counter snapshot written by the
    /// kernel (`clen_row_out`; null ⇒ this is the entry above, launch for launch).
    ///
    /// The store is made by the thread that already bumps `*clen` and reads the same
    /// location back, so it is ordered behind the commit and carries exactly the
    /// value a `memcpy_d2d` issued after this launch would have copied — including on
    /// a row that completes no group, where the counter is unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn compress_commit_rows_on(
        &self,
        latent: *const f32,
        cos: *const f32,
        sin: *const f32,
        ring: *mut f32,
        out_rows: *const c_int,
        clen: *mut c_int,
        clen_row_out: *mut c_int,
        hd: i32,
        rope_dim: i32,
        half: i32,
        window: i32,
        ratio: i32,
        s: CuStream,
    ) -> Result<()> {
        let f = self.need(self.kernels.compress_commit_rows, "dsv41_compress_commit_rows")?;
        let rc = unsafe {
            f(
                latent,
                cos,
                sin,
                ring,
                out_rows,
                clen,
                clen_row_out,
                hd,
                rope_dim,
                half,
                window,
                ratio,
                s,
            )
        };
        self.kerr(rc, "dsv41_compress_commit_rows")
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

    /// COMPRESSOR-PROJ-MROWS (`DSV41_COMPRESSOR_PROJ_MROWS`): the multi-row f32
    /// GEMV — `m` activation rows of `x` [`[m, k]`, row r at `+r*k`] projected
    /// against the ONE f32 weight `w` [`[n, k]`] into `out` [`[m, n]`, row r at
    /// `+r*n`], in ONE launch that streams the weight ONCE. This is the fold of
    /// the verify's per-row `lin_f32_on` loop (`chain_dev.rs::compress_proj_rows`:
    /// `comp_wkv` + `comp_wgate`, 2 launches per row per compress-source layer).
    ///
    /// `Ok(true)` = launched. `Ok(false)` = NOT performed (a stale `.so` without
    /// the symbol, `m` outside `1..=8`, or the C entry's decline — the per-row
    /// reference not being v2, `n >= 2048`, `k % 4 != 0`); the caller keeps its
    /// per-row loop, which is the bit-exact reference this kernel was transcribed
    /// from. Never an error: a decline is a property of the shape/mode, exactly
    /// as [`Self::gemm_fp8_mrows`] documents.
    ///
    /// Stream-taking on purpose, like [`Self::gemv_f32_on`]: the compressor's
    /// projections run on the side stream `DSV41_COMPRESS_SIDE` selects.
    pub fn gemv_f32_mrows(
        &self,
        w: *const f32,
        x: *const f32,
        out: *mut f32,
        rows: i32,
        n: i32,
        k: i32,
        s: CuStream,
    ) -> Result<bool> {
        let Some(f) = self.kernels.gemv_f32_mrows else {
            return Ok(false);
        };
        if !(1..=8).contains(&rows) {
            return Ok(false);
        }
        let rc = unsafe { f(w, x, out, rows, n, k, s) };
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemv_f32_mrows")?;
        Ok(true)
    }

    /// True when the loaded `.so` carries the multi-row f32 GEMV
    /// (`dsv41_gemv_f32_mrows`). A stale `.so` leaves the compressor's
    /// projections on their per-row loop, which is the bit-exact reference they
    /// were verified against.
    pub fn supports_gemv_f32_mrows(&self) -> bool {
        self.kernels.gemv_f32_mrows.is_some()
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
                act_e4m3, self.seq_align_i32(), self.stream,
            )
        };
        self.kerr(rc, "dsv41_expert_gate_up_fp4_batched")
    }

    /// TILELANG MoE grouped GEMM — up (gate‖up), the bf16 arm
    /// (`DSV41_MOE_TILELANG`, default OFF; see `moe_tilelang()` in chain_dev.rs).
    ///
    /// Fires the shim's three-launch sequence (gather f32->bf16 → grouped MMA →
    /// scatter) and writes the **RAW gate‖up** `[rows][topk][2*inter]` layout into
    /// `out` (row pitch `topk*2*inter`), exactly the layout
    /// `dsv41_expert_gate_up_fp4_batched` writes when swiglu is NOT fused — so the
    /// existing separate swiglu pass still applies. `w_up` is the **bf16 copy**
    /// `[E, 2*inter, dim]` produced by the load-time dequant
    /// (`DSV41_MOE_BF16_DEQUANT`), `eid`/`order`/`counts` are the HOST moe_align
    /// tables (see `moe_align_host` in chain_dev.rs), `ids` is not passed: the
    /// segment expert table `eid` already IS the router's output, re-derived
    /// host-side from `route_idx_r`.
    ///
    /// Returns `Ok(false)` on `rc == 2` (DECLINED: shape/mode not accepted) so the
    /// caller keeps the proven SIMT launch, and an `Err` on any other non-zero rc
    /// (the shim's own rc contract: `2` is the only fallback).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_tilelang_gate_up_bf16(
        &self,
        act: *const f32,
        out: *mut f32,
        w_up: *const c_void,
        eid: *const c_int,
        order: *const c_int,
        counts: *const c_int,
        nseg: i32,
        rows: i32,
        dim: i32,
        inter: i32,
        topk: i32,
    ) -> Result<bool> {
        let f = self.need(
            self.kernels.moe_tilelang_gate_up_bf16,
            "dsv41_moe_tilelang_gate_up_bf16",
        )?;
        let rc = unsafe {
            f(act, out, w_up, eid, order, counts, nseg, rows, dim, inter, topk, self.stream)
        };
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_moe_tilelang_gate_up_bf16")?;
        Ok(true)
    }

    /// TILELANG MoE grouped GEMM — down, the bf16 arm (`DSV41_MOE_TILELANG`).
    ///
    /// `act` is the swiglu'd `[rows*topk][inter]` activation (the existing
    /// `dsv41_swiglu_limit_batched` output), `w_dn` the bf16 copy
    /// `[E, dim, inter]`, and `out` the per-slot partials `[rows][dim] /* NOT [rows*topk]: activations are quantised per row */` — the
    /// same buffer `dsv41_expert_down_fp4_batched` writes, so the existing
    /// fixed-order `moe_down_reduce` sum is unchanged. `act_pitch` is the SOURCE
    /// slot stride in floats: the swiglu pass writes in place, so with the unfused
    /// layout the live `inter` values of each assignment sit `2*inter` apart (the
    /// fused layout uses `inter`). Same `rc` contract as the up arm.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_tilelang_down_bf16(
        &self,
        act: *const f32,
        out: *mut f32,
        w_dn: *const c_void,
        eid: *const c_int,
        order: *const c_int,
        counts: *const c_int,
        nseg: i32,
        rows: i32,
        dim: i32,
        inter: i32,
        topk: i32,
        act_pitch: i32,
    ) -> Result<bool> {
        let f = self.need(
            self.kernels.moe_tilelang_down_bf16,
            "dsv41_moe_tilelang_down_bf16",
        )?;
        let rc = unsafe {
            f(
                act, out, w_dn, eid, order, counts, nseg, rows, dim, inter, topk, act_pitch,
                self.stream,
            )
        };
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_moe_tilelang_down_bf16")?;
        Ok(true)
    }

    /// TILELANG MoE grouped GEMM — up, the **DEVICE-TABLE** bf16 arm
    /// (`dsv41_moe_tilelang_gate_up_bf16_dev`, same `DSV41_MOE_TILELANG` gate).
    ///
    /// The twin of [`Self::moe_tilelang_gate_up_bf16`] that can run INSIDE a
    /// CUDA-graph capture: `eid`/`order`/`counts` are DEVICE buffers (the
    /// `tl_eid`/`tl_order`/`tl_counts` scratch filled by
    /// [`Self::moe_align_from_group`]) and `nseg` is a DEVICE pointer
    /// (`tl_nseg`) instead of a value, so the shim performs no H2D upload and no
    /// D2H read at all. Everything else — the frozen shape gate, the `grid.y =
    /// SEG_CAP` launches and the raw gate‖up output layout — is identical, which
    /// is what makes the host-table entry a valid A/B baseline.
    ///
    /// `Ok(false)` on `rc == 2` (declined: shape/alignment, or INIT not done);
    /// `Err` on any other rc.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_tilelang_gate_up_bf16_dev(
        &self,
        act: *const f32,
        out: *mut f32,
        w_up: *const c_void,
        eid: *const c_int,
        order: *const c_int,
        counts: *const c_int,
        nseg: *const c_int,
        rows: i32,
        dim: i32,
        inter: i32,
        topk: i32,
    ) -> Result<bool> {
        let f = self.need(
            self.kernels.moe_tilelang_gate_up_bf16_dev,
            "dsv41_moe_tilelang_gate_up_bf16_dev",
        )?;
        let rc = unsafe {
            f(act, out, w_up, eid, order, counts, nseg, rows, dim, inter, topk, self.stream)
        };
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_moe_tilelang_gate_up_bf16_dev")?;
        Ok(true)
    }

    /// TILELANG MoE grouped GEMM — down, the device-table bf16 arm
    /// (`dsv41_moe_tilelang_down_bf16_dev`). Same contract as the up twin above;
    /// `act_pitch` is the raw `2*inter` slot stride the up arm's RAW gate‖up
    /// output leaves in `ex_act_r`/`ex_act_b`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_tilelang_down_bf16_dev(
        &self,
        act: *const f32,
        out: *mut f32,
        w_dn: *const c_void,
        eid: *const c_int,
        order: *const c_int,
        counts: *const c_int,
        nseg: *const c_int,
        rows: i32,
        dim: i32,
        inter: i32,
        topk: i32,
        act_pitch: i32,
    ) -> Result<bool> {
        let f = self.need(
            self.kernels.moe_tilelang_down_bf16_dev,
            "dsv41_moe_tilelang_down_bf16_dev",
        )?;
        let rc = unsafe {
            f(
                act, out, w_dn, eid, order, counts, nseg, rows, dim, inter, topk, act_pitch,
                self.stream,
            )
        };
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_moe_tilelang_down_bf16_dev")?;
        Ok(true)
    }

    /// Load-time fp4(e2m1 + ue8m0) → bf16 expand (`DSV41_MOE_BF16_DEQUANT`).
    /// `wq` is `[n, k/2]` packed e2m1 (low nibble = even k), `ws` is `[n, k/32]`
    /// ue8m0 (`scale = 2^(byte-127)`) and `out_bf16` is `[n, k]` bf16.
    ///
    /// `wq_pitch` / `ws_pitch` are the PHYSICAL row pitches in bytes (`<= 0` = the
    /// natural `k/2` / `k/32`). They must be given explicitly: with
    /// `DSV41_SF_STRIDE_PAD` (default ON) the loader pads every weight plane's row
    /// pitch to a 16-byte multiple, so `w2.scale`'s rows are 16 B — not `k/32` — apart
    /// and the natural stride would read the WRONG scales (a silent wrong answer).
    /// `k % 32 == 0` and 16-byte-aligned bases are required (rc 2 otherwise).
    pub fn moe_fp4_to_bf16(
        &self,
        wq: *const c_void,
        ws: *const c_void,
        out_bf16: *mut c_void,
        n: i32,
        k: i32,
        wq_pitch: i32,
        ws_pitch: i32,
    ) -> Result<bool> {
        let f = self.need(self.kernels.moe_fp4_to_bf16, "dsv41_moe_fp4_to_bf16")?;
        let rc = unsafe { f(wq, ws, out_bf16, n, k, wq_pitch, ws_pitch, self.stream) };
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_moe_fp4_to_bf16")?;
        Ok(true)
    }

    /// TILELANG MoE block-scaled arm — up (gate‖up), the **native fp4 weights** path
    /// (`DSV41_MOE_TILELANG_BS`; mutually exclusive with `DSV41_MOE_TILELANG`).
    ///
    /// Fires the shim's three-launch sequence (gather the e4m3 activation + pack its
    /// SF → block-scaled grouped MMA → scatter) and writes the **RAW gate‖up**
    /// `[rows][topk][2*inter]` layout into `out`, exactly the layout
    /// `dsv41_expert_gate_up_fp4_batched` writes when swiglu is NOT fused, so the
    /// existing separate swiglu pass still applies.
    ///
    /// `xq4`/`xsc4` are the **already-quantised** routed activations (`[rows*topk]`
    /// slots of `dim` **e4m3** bytes — one byte per value, the official
    /// `act_quant(fp8_block_size=32)` form — and `dim/32` f32 scales; the same
    /// buffers the proven batched launch reads when `DSV41_EXPERT_ACT_E4M3=1`),
    /// `w1`/`w3` are the expert pool's gate/up planes (still packed fp4) and
    /// `sfw1`/`sfw3` the load-time packed group-major scale words. No bf16 copy is
    /// involved, so `DSV41_MOE_BF16_DEQUANT` is NOT a precondition. The activation
    /// format is versioned by `dsv41_moe_bs_act_e4m3_cap` — see
    /// [`Self::supports_moe_bs_act_e4m3`].
    ///
    /// Returns `Ok(false)` on `rc == 2` (DECLINED) so the caller keeps the proven
    /// SIMT launch, and an `Err` on any other non-zero rc.
    ///
    /// `w_stride` is the per-expert **block stride** in bytes (the pool packs each
    /// expert as one 128 B aligned six-plane block), measured by the caller from
    /// the expert pool's pointers. It is the `gstride[1]` of the W1/W3 TMA
    /// descriptors; deriving it from `NP*K/2` was the 2026-09-13 bug.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_tilelang_gate_up_bs(
        &self,
        xq4: *const u8,
        xsc4: *const f32,
        out: *mut f32,
        w1: *const c_void,
        w3: *const c_void,
        sfw1: *const c_void,
        sfw3: *const c_void,
        eid: *const c_int,
        order: *const c_int,
        counts: *const c_int,
        nseg: i32,
        w_stride: u64,
        rows: i32,
        dim: i32,
        inter: i32,
        topk: i32,
    ) -> Result<bool> {
        let f = self.need(
            self.kernels.moe_tilelang_gate_up_bs,
            "dsv41_moe_tilelang_gate_up_bs",
        )?;
        let rc = unsafe {
            f(
                xq4, xsc4, out, w1, w3, sfw1, sfw3, eid, order, counts, nseg, rows,
                dim, inter, topk, w_stride as i64, self.stream,
            )
        };
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_moe_tilelang_gate_up_bs")?;
        Ok(true)
    }

    /// TILELANG MoE block-scaled arm — up (gate‖up), the **DEVICE-TABLE** twin
    /// (`dsv41_moe_tilelang_gate_up_bs_dev`, same `DSV41_MOE_TILELANG_BS` gate).
    ///
    /// The twin of [`Self::moe_tilelang_gate_up_bs`] that can run INSIDE a
    /// CUDA-graph capture: `eid`/`order`/`counts` are DEVICE buffers (the
    /// `tl_eid`/`tl_order`/`tl_counts` scratch filled by
    /// [`Self::moe_align_from_group`]) and `nseg` is a DEVICE pointer
    /// (`tl_nseg`) instead of a value, so the shim performs no H2D upload and no
    /// D2H read at all — the host `moe_align` round trip is what made the arm
    /// eager-only. Everything else — the frozen shape gate, the TMA-descriptor
    /// construction (W/SFW maps cached per expert-pool base), the `grid.y =
    /// SEG_CAP` launches and the raw gate‖up output layout — is identical, which
    /// is what makes the host-table entry a valid A/B baseline.
    ///
    /// `Ok(false)` on `rc == 2` (declined: shape/alignment, TMA init failure, or
    /// INIT not done); `Err` on any other rc.
    ///
    /// The operand contract is identical, including the activation format
    /// (`xq4` = `[rows][dim] /* NOT [rows*topk]: activations are quantised per row */` u8 **e4m3**, one byte per value) — see
    /// [`Self::moe_tilelang_gate_up_bs`].
    ///
    /// `w_stride` is the measured per-expert **block stride** in bytes — the
    /// `gstride[1]` of the W1/W3 TMA descriptors, same contract as
    /// [`Self::moe_tilelang_gate_up_bs`].
    #[allow(clippy::too_many_arguments)]
    pub fn moe_tilelang_gate_up_bs_dev(
        &self,
        xq4: *const u8,
        xsc4: *const f32,
        out: *mut f32,
        w1: *const c_void,
        w3: *const c_void,
        sfw1: *const c_void,
        sfw3: *const c_void,
        eid: *const c_int,
        order: *const c_int,
        counts: *const c_int,
        nseg: *const c_int,
        w_stride: u64,
        rows: i32,
        dim: i32,
        inter: i32,
        topk: i32,
    ) -> Result<bool> {
        let f = self.need(
            self.kernels.moe_tilelang_gate_up_bs_dev,
            "dsv41_moe_tilelang_gate_up_bs_dev",
        )?;
        let rc = unsafe {
            f(
                xq4, xsc4, out, w1, w3, sfw1, sfw3, eid, order, counts, nseg, rows,
                dim, inter, topk, w_stride as i64, self.stream,
            )
        };
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_moe_tilelang_gate_up_bs_dev")?;
        Ok(true)
    }

    /// Load-time ue8m0 repack for the block-scaled arm (one expert's plane per call).
    /// `src` is `[rows, k/32]` row-major u8, `dst` the `[words*rows]` u32 group-major
    /// pool. A byte permutation of already-e8m0 data ⇒ bit-exact; `k % 128 == 0` and
    /// 16-byte-aligned bases are required (rc 2 otherwise).
    pub fn moe_bs_pack_wsf(
        &self,
        src: *const c_void,
        dst: *mut c_void,
        rows: i32,
        k: i32,
    ) -> Result<bool> {
        let f = self.need(self.kernels.moe_bs_pack_wsf, "dsv41_moe_bs_pack_wsf")?;
        let rc = unsafe { f(src, dst, rows, k, self.stream) };
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_moe_bs_pack_wsf")?;
        Ok(true)
    }

    /// True when the `.so` carries BOTH TileLang MoE grouped-GEMM arms. A stock
    /// `.so` (or a stale one from before the tilelang_gen MoE shim) has neither, so
    /// `DSV41_MOE_TILELANG` must stay OFF and be reported — otherwise an armed gate
    /// would silently measure the OLD path (the project's #1 measurement-bias trap).
    pub fn supports_moe_tilelang(&self) -> bool {
        self.kernels.moe_tilelang_gate_up_bf16.is_some()
            && self.kernels.moe_tilelang_down_bf16.is_some()
    }

    /// True when the loaded `.so` carries the whole DEVICE-TABLE MoE path: the
    /// device-side `moe_align` (`dsv41_moe_align_from_group`) **and** both
    /// `*_dev` shim entries. Probed as a SET on purpose: arming `DSV41_MOE_TILELANG`
    /// on a `.so` that has only part of it would fall back to the host tables (or
    /// to the SIMT path) while the operator believes the graph arm ran — the
    /// project's #1 measurement-bias trap. `supports_route_group` covers the
    /// `dsv41_route_group` half of the chain and is checked separately by the
    /// caller; both probes are required for the in-graph arm.
    pub fn supports_moe_align_from_group(&self) -> bool {
        self.kernels.moe_align_from_group.is_some()
            && self.kernels.moe_tilelang_gate_up_bf16_dev.is_some()
            && self.kernels.moe_tilelang_down_bf16_dev.is_some()
    }

    /// True when the `.so` carries the block-scaled (native fp4 weights) MoE arm,
    /// its load-time SF repack **and** the e4m3-activation capability marker
    /// ([`Self::supports_moe_bs_act_e4m3`]). `DSV41_MOE_TILELANG_BS` must stay OFF
    /// (and be reported) without all three — the same build-vs-runtime split as
    /// every other arm here. The capability marker is part of the SET on purpose:
    /// the arm's `xq4` argument changed meaning (packed fp4 → e4m3) without
    /// changing its ABI shape, so a `.so` predating the D2 fix would read e4m3
    /// bytes as fp4 nibbles — a SILENT wrong answer. Probing it here keeps the arm
    /// off instead of measuring garbage.
    pub fn supports_moe_tilelang_bs(&self) -> bool {
        self.kernels.moe_tilelang_gate_up_bs.is_some()
            && self.kernels.moe_bs_pack_wsf.is_some()
            && self.kernels.moe_bs_act_e4m3_cap.is_some()
    }

    /// True when the block-scaled shim's A operand is the **e4m3** activation
    /// (`dsv41_moe_bs_act_e4m3_cap`, `tilelang_gen/moe_bs_shim.cu` §8).
    ///
    /// The D2 fix (2026-09-13) changed `xq4`'s semantics from packed fp4 nibbles
    /// (`dim/2` B/row) to e4m3 (`dim` B/row) while leaving the C ABI shape (a
    /// `const uint8_t*` in the same position) untouched, so a stale `.so` would
    /// silently decode 5120 B rows as 2560 B of fp4. This probe is the same
    /// device-vs-runtime gate as [`Self::supports_expert_act_e4m3`]; it is exposed
    /// separately so an arm that declines can name WHICH half is missing. The set
    /// probe [`Self::supports_moe_tilelang_bs`] includes it too.
    pub fn supports_moe_bs_act_e4m3(&self) -> bool {
        self.kernels.moe_bs_act_e4m3_cap.is_some()
    }

    /// True when the block-scaled arm's **device-table** set is present: the
    /// device-side `moe_align` (`dsv41_moe_align_from_group`) **and** the
    /// `*_dev` shim entry. Probed as a SET for the same reason as
    /// [`Self::supports_moe_align_from_group`]: a `.so` carrying only part of it
    /// would fall back to the host tables (or the SIMT launch) while the operator
    /// believes the in-graph arm ran — the project's #1 measurement-bias trap.
    /// `supports_moe_tilelang_bs` covers the host-shim / load-time-pack half and is
    /// checked separately by the caller; both probes are required for the graph arm.
    pub fn supports_moe_tilelang_bs_dev(&self) -> bool {
        self.kernels.moe_align_from_group.is_some()
            && self.kernels.moe_tilelang_gate_up_bs_dev.is_some()
    }

    /// True when the `.so` carries the load-time fp4→bf16 dequant
    /// (`DSV41_MOE_BF16_DEQUANT` must then stay OFF and be reported).
    pub fn supports_moe_bf16_dequant(&self) -> bool {
        self.kernels.moe_fp4_to_bf16.is_some()
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
        // ★ `rc == 0` means EXACTLY ONE thing: the `.so`'s own gate was OFF
        // (nothing was launched), and the caller keeps the proven GEMV path.
        // Every other outcome is a real error and goes through `kerr`, including
        // a REJECTED ARGUMENT LIST — `m4_launch_gateup` returns
        // `cudaErrorInvalidValue` for a shape OR a 16B-alignment violation, and
        // prints an `[align] tc5::mxf4: …` line to stderr before it does. The two
        // used to be indistinguishable downstream (both surfaced as
        // `!ran_tc` → fallback), which is the project's #1 measurement trap: a
        // layout accident read as "the arm never engaged" instead of "the arm
        // refused". If a round shows the arm not running, check stderr for an
        // `[align]` line BEFORE concluding the gate is off.
        if rc == 0 {
            return Ok(false); // .so gate OFF: nothing ran, the caller falls back
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
    /// comment: the ascending-slot order is the numerical contract). The
    /// DSV41_SEQ_ALIGN (#5) variant with the official's order is
    /// [`Self::moe_down_reduce_seq`].
    pub fn moe_down_reduce(&self, part: *const f32, out: *mut f32, n: i32, slots: i32) -> Result<()> {
        let f = self.need(self.kernels.moe_down_reduce, "dsv41_moe_down_reduce")?;
        let rc = unsafe { f(part, out, n, slots, self.stream) };
        self.kerr(rc, "dsv41_moe_down_reduce")
    }

    /// Fixed-order sum of the batched down scratch into `out` (see the kernel
    /// comment: the ascending-slot order is the numerical contract).
    ///
    /// DSV41_SEQ_ALIGN (#5): when `seq_align != 0` the slot order is REPLACED by
    /// the ascending-EXPERT-ID permutation (the official's accumulation order,
    /// `ref_inference/model.py:895-900`) using `ids` = this row's `[slots]` router
    /// output. `seq_align == 0` runs the legacy kernel/order.
    ///
    /// `Ok(false)` when the gate is armed but the loaded `.so` has no
    /// `dsv41_moe_down_reduce_seq` (stale build): NOTHING was launched, so the
    /// caller must fall back to [`Self::moe_down_reduce`] and say so once — an
    /// armed-but-inert gate must never be silent.
    pub fn moe_down_reduce_seq(
        &self,
        part: *const f32,
        out: *mut f32,
        n: i32,
        slots: i32,
        ids: *const i32,
        seq_align: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.moe_down_reduce_seq else {
            return Ok(false);
        };
        let rc = unsafe { f(part, out, n, slots, ids, seq_align, self.stream) };
        self.kerr(rc, "dsv41_moe_down_reduce_seq")?;
        Ok(true)
    }

    /// [`Self::moe_down_reduce`] whose epilogue ALSO copies `out` into every
    /// peer's staging slot (A1a, `dsv41_moe_down_reduce_st`) — valid only where
    /// this sum is `s.o`'s LAST writer (the rank does NOT run the shared expert).
    /// The caller must then run the AR's publish+reduce half, not the full
    /// all-reduce.
    ///
    /// `Ok(false)` when the loaded .so has no `dsv41_moe_down_reduce_st` (stale
    /// build): NOTHING was launched and the caller must fall back, so the store
    /// can never be dropped silently.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_down_reduce_ar(
        &self,
        part: *const f32,
        out: *mut f32,
        n: i32,
        slots: i32,
        staging_tbl: *const *mut f32,
        epoch: *const c_uint,
        world: i32,
        my_rank: i32,
        stride: i32,
    ) -> Result<bool> {
        let Some(f) = self.kernels.moe_down_reduce_ar else {
            return Ok(false);
        };
        let rc = unsafe {
            f(
                part, out, n, slots, staging_tbl, epoch, world, my_rank, stride, self.stream,
            )
        };
        self.kerr(rc, "dsv41_moe_down_reduce_st")?;
        Ok(true)
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
        seq_align: i32,
    ) -> Result<()> {
        let f = self.need(
            self.kernels.expert_down_reduce_fp4_batched,
            "dsv41_expert_down_reduce_fp4_batched",
        )?;
        let rc = unsafe {
            f(
                act_base, act_stride, out, rows, dim, inter, row_weight, rw_stride, slots, w2_base,
                w2_stride, w2s_base, w2s_stride, ids, seq_align, self.stream,
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
        let rc =
            unsafe { f(gate_up, rows, inter, limit, slot_stride, slots, self.seq_align_i32(), self.stream) };
        self.kerr(rc, "dsv41_swiglu_limit_batched")
    }

    /// ROUTED DOWN PREP (DSV41_ROUTED_DOWN_QUANT, default OFF): the official
    /// routed expert's activation pipeline, in place, in ONE launch — route
    /// weight, then the `x.to(bf16)` boundary, then the block-32 e4m3
    /// quantise/dequantise act_quant round trip whose f32 result is what the
    /// official `fp4_gemm` multiplies. `pitch` is the slot pitch the gate/up arm
    /// wrote (`act_slot`: `inter` fused, `2*inter` raw) and `route_w` is read at
    /// the flat `(row*slots + slot)` index, exactly like the down launcher's
    /// `rw_stride == 1` addressing.
    ///
    /// ⚠️ The caller must then pass `row_weight = nullptr` to the down launch:
    /// this call has already applied the route weight.
    ///
    /// `dbg` is None in production. Some(ptr) enables the 5x32 probe dump
    /// (DSV41_ROUTED_DOWN_QUANT_DBG) — the kernel's output is identical either way.
    pub fn routed_down_prep(
        &self,
        act: *mut f32,
        route_w: *const f32,
        rows: i32,
        slots: i32,
        pitch: i32,
        inter: i32,
        dbg: Option<*mut f32>,
    ) -> Result<()> {
        let f = self.need(self.kernels.routed_down_prep, "dsv41_routed_down_prep")?;
        let d = dbg.unwrap_or(std::ptr::null_mut());
        let rc = unsafe {
            f(act, route_w, rows, slots, pitch, inter, d, self.stream)
        };
        self.kerr(rc, "dsv41_routed_down_prep")
    }

    /// I3 (DSV41_ATTN_P_BF16_DBG): copy the sparse attention's PV probe buffer to
    /// the host. The buffer is filled by whichever sparse-attention kernel the
    /// launcher selected, when `DSV41_ATTN_P_BF16_DBG=1` armed it (the flag is
    /// read in the launcher, not here — this method only reads back). Errors with
    /// the missing-symbol reason when the loaded `.so` predates I3.
    pub fn attn_p_dbg_read(&self, host: *mut f32, n: i32) -> Result<()> {
        let f = self.need(self.kernels.attn_p_dbg_read, "dsv41_attn_p_dbg_read")?;
        let rc = unsafe { f(host, n, self.stream) };
        self.kerr(rc, "dsv41_attn_p_dbg_read")?;
        // The copy is issued on the compute stream; the caller's `host` buffer is
        // ordinary memory, so the stream must drain before it can be read. Doing it
        // here keeps the one-shot probe's contract in one place.
        self.sync()
    }

    /// Whether the loaded `.so` carries the I3 probe readback entry. The gate
    /// `DSV41_ATTN_P_BF16` lives entirely in the kernel launchers and does NOT
    /// need this; only the `_DBG` readback does.
    pub fn supports_attn_p_dbg_read(&self) -> bool {
        self.kernels.attn_p_dbg_read.is_some()
    }

    /// True when the loaded .so carries the routed-down prep entry point
    /// (`dsv41_routed_down_prep`). A stale .so leaves DSV41_ROUTED_DOWN_QUANT
    /// inert — with a one-shot notice from the chain, never silently.
    pub fn supports_routed_down_prep(&self) -> bool {
        self.kernels.routed_down_prep.is_some()
    }

    /// A4: the indexer's in-place fp4 round trip for ONE row-block of `rows` rows
    /// (`rows` == 1 on the k path, `index_n_heads` on the q path) — the
    /// reference's `fp4_act_quant(x, 32, True)` with the default e8m0 scale
    /// (`model.py:546`/`:552`, `kernel.py:126-183`). `cols` (`index_head_dim`)
    /// must be a multiple of 32.
    ///
    /// `io` == 1 is the faithful arm (the reference reads and writes BF16:
    /// `in_dtype` default `kernel.py:128`, `out_dtype = in_dtype` `:136`);
    /// `io` == 0 skips both bf16 boundaries and is strictly more precise, so it
    /// is an A/B arm only.
    ///
    /// `dbg` is None in production. Some(ptr) enables the 224-float probe dump
    /// (DSV41_INDEXER_FP4_RT_DBG) — the round trip is identical either way. The
    /// probe does a D2H readback, so it MUST NOT be armed inside a capture.
    pub fn indexer_fp4_rt(
        &self,
        x: *mut f32,
        rows: i32,
        cols: i32,
        io: i32,
        tag: *const std::os::raw::c_char,
        dbg: Option<*mut f32>,
    ) -> Result<()> {
        let f = self.need(self.kernels.indexer_fp4_rt, "dsv41_indexer_fp4_rt")?;
        let d = dbg.unwrap_or(std::ptr::null_mut());
        let rc = unsafe { f(x, rows, cols, io, tag, d, self.stream) };
        self.kerr(rc, "dsv41_indexer_fp4_rt")
    }

    /// True when the loaded .so carries the indexer fp4 round-trip entry point
    /// (`dsv41_indexer_fp4_rt`). A stale .so leaves DSV41_INDEXER_FP4_RT inert —
    /// with a one-shot notice from the chain, never silently.
    pub fn supports_indexer_fp4_rt(&self) -> bool {
        self.kernels.indexer_fp4_rt.is_some()
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
        // L4-9 (DSV41_NORM_SPLIT, default OFF): the dim-split twin. Only a shape
        // the C launcher accepts is issued, and a decline falls through to the
        // original launch below — so OFF (or a stale `.so`, or a decline) is the
        // pre-existing call, bit for bit.
        if norm_split_wanted() {
            if let Some(fs) = self.kernels.rmsnorm_rows_split {
                let nc = norm_split_chunks(dim);
                if norm_split_fits(rows, dim, nc) {
                    let rc = unsafe { fs(x, w, out, rows, dim, eps, nc, self.stream) };
                    if rc != NORM_SPLIT_DECLINE {
                        self.kerr(rc, "dsv41_rmsnorm_rows_split")?;
                        return Ok(true);
                    }
                }
            }
        }
        let f = match self.kernels.rmsnorm_rows {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe { f(x, w, out, rows, dim, eps, self.stream) };
        self.kerr(rc, "dsv41_rmsnorm_rows")?;
        Ok(true)
    }

    /// [`Self::rmsnorm_rows`] issued on `s` instead of the main stream. Only the
    /// verify block's attention dual chain (`DSV41_VERIFY_FORK`) uses it: its kv
    /// half (norm + rope) rides the second side stream under the q chain, and the
    /// split is bit-identical exactly because the SAME `dsv41_rmsnorm_rows`
    /// launch is issued, only on another stream. Same `Ok(false)` contract (a
    /// stale `.so`/a declined shape) as [`Self::rmsnorm_rows`], so the caller
    /// falls back to the stream-parameterised plain rmsnorm either way.
    pub fn rmsnorm_rows_on(
        &self,
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        rows: i32,
        dim: i32,
        eps: f32,
        s: CuStream,
    ) -> Result<bool> {
        let f = match self.kernels.rmsnorm_rows {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe { f(x, w, out, rows, dim, eps, s) };
        self.kerr(rc, "dsv41_rmsnorm_rows")?;
        Ok(true)
    }

    /// B4 (DSV41_RMSNORM_ROPE_MROWS, default OFF): [`Self::rmsnorm_rows_on`]'s
    /// norm AND the trailing-`2*half` RoPE of the same rows in ONE launch
    /// (`dsv41_rmsnorm_rope_mrows`) — where the kv half issued two.
    ///
    /// Phase 1 is `dsv41_rmsnorm_rows_kernel`'s body verbatim at the same
    /// blockDim (1024), so the normalized rows are bit-identical; phase 2 is
    /// `apply_rope_kernel`'s rotation of the columns `[rope_off, rope_off+2*half)`
    /// at `pos_rows[row]` — the identical integer the two-launch form computed
    /// as `pos_base + row` — with no reduction, so it cannot move a value. The
    /// only addition is a barrier between the phases (memory order, not math).
    ///
    /// The stream is a PARAMETER, not `self.stream`: the kv half is the
    /// `DSV41_VERIFY_FORK` side chain, and the two launches this replaces are the
    /// `*_on` entries. Hard-coding the main stream would drag the kv chain back
    /// off the side stream and eat the fork's gain.
    ///
    /// `Ok(false)` = NOT performed, keep `rmsnorm_rows_on` + `apply_rope_on`:
    /// the .so predates the symbol, or the shape is outside the fusion's domain
    /// (`rows <= 0`, `dim <= 0`, `half <= 0`, `rope_off < 0`,
    /// `rope_off + 2*half > dim`). The C entry declines the same set with 2, so
    /// an armed gate never silently measures the old path while returning
    /// `Ok(true)`.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_rope_mrows(
        &self,
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        rows: i32,
        dim: i32,
        eps: f32,
        cos: *const f32,
        sin: *const f32,
        rope_off: i32,
        half: i32,
        pos_rows: *const c_int,
        inverse: bool,
        s: CuStream,
    ) -> Result<bool> {
        let Some(f) = self.kernels.rmsnorm_rope_mrows else {
            return Ok(false); // older .so: keep the two-launch kv pair
        };
        if rows <= 0 || dim <= 0 || half <= 0 || rope_off < 0 || rope_off + 2 * half > dim {
            return Ok(false);
        }
        let rc = unsafe {
            f(
                x,
                w,
                out,
                rows,
                dim,
                eps,
                cos,
                sin,
                rope_off,
                half,
                pos_rows,
                inverse as i32,
                s,
            )
        };
        // 2 = the kernel's "declined" (shape), the caller falls back.
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_rmsnorm_rope_mrows")?;
        Ok(true)
    }

    /// True when the loaded .so carries B4 (`dsv41_rmsnorm_rope_mrows`). A stale
    /// .so leaves `DSV41_RMSNORM_ROPE_MROWS` inert and the verify's kv half on
    /// `norm_rows + apply_rope`, which is the reference B4 was written against.
    pub fn supports_rmsnorm_rope_mrows(&self) -> bool {
        self.kernels.rmsnorm_rope_mrows.is_some()
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
        // L4-9 (DSV41_CNORM_SPLIT, default OFF): the dim-split twin — the fix
        // for the m = 1 one-CTA case. Issued only for a shape the C launcher
        // accepts; a decline falls through to the original launch below, so OFF
        // (or a stale `.so`, or a decline) is the pre-existing call, bit for bit.
        if cnorm_split_wanted() {
            if let Some(fs) = self.kernels.hc_collapse_norm_split {
                let nc = norm_split_chunks(dim);
                if norm_split_fits(rows, dim, nc) {
                    let rc = unsafe {
                        fs(x, pre, w, out, rows, hc, dim, eps, truncate as c_int, nc, self.stream)
                    };
                    if rc != NORM_SPLIT_DECLINE {
                        self.kerr(rc, "dsv41_hc_collapse_norm_split")?;
                        return Ok(());
                    }
                }
            }
        }
        let rc = unsafe {
            (self.kernels.hc_collapse_norm)(
                x, pre, w, out, rows, hc, dim, eps, truncate as c_int, self.stream,
            )
        };
        self.kerr(rc, "dsv41_hc_collapse_norm")
    }

    /// P3-lite segment A (draft): the hyper-connection front end in ONE launch.
    ///
    /// Phase 1 is `hc_mixes_kernel`'s program at its OWN logical width (`mix *
    /// 32` threads: the ss walk strides by that literal and the cross-warp fold
    /// sums `mix` partials), phase 2 is `dsv41_hc_collapse_norm_kernel`'s body at
    /// its own 1024. The collapse reads `pre_collapse` -- the caller's INCOMING
    /// premix slot, never the `pre` phase 1 just wrote -- so the two phases share
    /// no buffer and `__syncthreads()` between them only orders the launches it
    /// replaces. See the kernel header for the full argument.
    ///
    /// `Ok(false)` = no symbol in the loaded `.so`, or a decline (an `hc` whose
    /// reference launch width is not `mix * 32`, the spread variant, or a
    /// `DSV41_HC_MIXES_THREADS` override) -- the caller then runs the
    /// `hc_mixes` + `hc_collapse_norm` pair, which is the reference.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_hc_front(
        &self,
        x: *const f32,
        hc_fn: *const f32,
        hc_scale: *const f32,
        hc_base: *const f32,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        pre_collapse: *const f32,
        w_norm: *const f32,
        out: *mut f32,
        rows: i32,
        hc: i32,
        dim: i32,
        sinkhorn_iters: i32,
        eps: f32,
        eps_norm: f32,
        truncate: bool,
    ) -> Result<bool> {
        let Some(f) = self.kernels.draft_hc_front else {
            return Ok(false);
        };
        let rc = unsafe {
            f(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                pre,
                post,
                comb,
                pre_collapse,
                w_norm,
                out,
                rows,
                hc,
                dim,
                sinkhorn_iters,
                eps,
                eps_norm,
                truncate as c_int,
                self.stream,
            )
        };
        if rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_draft_hc_front")?;
        Ok(true)
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

    /// True when the loaded `.so` carries the P3 megakernel entry
    /// (`dsv41_verify_hc_front_prefused`). A stale `.so` reports false and the
    /// caller keeps the split / two-launch front end plus its `quant_rows`.
    pub fn supports_verify_hc_front_prefused(&self) -> bool {
        self.kernels.verify_hc_front_prefused.is_some()
    }

    /// P3 MEGAKERNEL (`DSV41_P3_MEGAKERNEL=1`, default OFF; design
    /// `docs/agent/p3-megakernel-verify-design.md` §2.1): [`Self::hc_front_persist_mb`]'s
    /// one-launch phase structure — dots spread over `mix*split` blocks, collapse
    /// on its own parallel block, tail elected to the last-finishing dot block —
    /// plus the ROW-BASED T1 fp8 emit.
    ///
    /// `xq`/`xsc` are the caller's `[rows, xq_pitch]` (fp8 e4m3 bytes) and
    /// `[rows, xq_pitch/32]` (f32) staging in the layout `quant_rows` writes and
    /// `gemm_fp8_mrows_kernel` reads back (`a + r*k`, `a_scale + r*(k/32)`), so
    /// the same `xq_pitch = cols` the caller would have passed to `quant_rows`
    /// makes the emitted pair byte-identical to that launch. A null pair (or
    /// `xq_pitch <= 0`) disables the emit — the older kernel's path exactly.
    /// The kernel is only entered when `xq != nullptr`, so a caller that wants
    /// the collapse WITHOUT the fp8 can pass nulls and use
    /// [`Self::hc_front_persist_mb`] semantics.
    ///
    /// Returns `Ok(false)` on ANY decline (stale `.so`, a shape the C entry does
    /// not specialise — `rows > DSV41_HC_SPREAD_MAXR`, `dim % 32 != 0` for an
    /// emit request, an unusable chunk — or a `cudaFuncSetAttribute` refusal):
    /// the caller then runs its older chain, which is the bit-exact reference at
    /// `split == 1`. ⚠️ `split > 1` (`DSV41_P3_MK_SPLIT`) is deterministic but
    /// NOT bit-exact (the K partials recombine in `ck` order); `split == 1` is
    /// the default and the parity target.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_hc_front_prefused(
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
        xq: *mut u8,
        xsc: *mut f32,
        xq_pitch: i32,
        rows: i32,
        hc: i32,
        dim: i32,
        sinkhorn_iters: i32,
        eps: f32,
        eps_norm: f32,
        truncate: bool,
    ) -> Result<bool> {
        let f = self.need(
            self.kernels.verify_hc_front_prefused,
            "dsv41_verify_hc_front_prefused",
        )?;
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
                xq,
                xsc,
                xq_pitch,
                rows,
                hc,
                dim,
                sinkhorn_iters,
                eps,
                eps_norm,
                truncate as c_int,
                self.stream,
            )
        };
        // The C entry's decline contract: 1 (InvalidValue) and 2 (shape) both
        // mean "fall back", exactly as the sibling `hc_front*` entries do.
        if rc == 1 || rc == 2 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_verify_hc_front_prefused")?;
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

// ---------------------------------------------------------------- L4-9 split
//
// `dsv41_hc_collapse_norm` / `dsv41_rmsnorm_rows` launch `grid = rows`, so at
// m = 1 they run as ONE block on ONE SM. The split entries cut the `dim` axis
// into `nchunks` contiguous segments (`grid = (nchunks, rows)`) and do the row
// reduction in two stages (per-chunk partial -> elected-last-block fold), which
// is the fix for that worst case. Both gates are default OFF: the OFF arm issues
// the original launch with the original arguments, i.e. bit-for-bit today's
// path, and a stale `.so` without the new symbols reports `None` and falls back
// the same way.

/// Caps of the split kernels (`DSV41_NORM_SPLIT_MAXR` / `_MAXC` in
/// `dsv41_kernels.cu`). The C launcher declines above them via
/// `DSV41_NORM_SPLIT_DECLINE`, and this side declines FIRST so a shape the split
/// cannot take never reaches the entry (a decline is the same fall-back as a
/// missing symbol). Keep the two in step.
const NORM_SPLIT_MAXR: i32 = 256;
const NORM_SPLIT_MAXC: i32 = 16;

/// The split launchers' decline sentinel (`DSV41_NORM_SPLIT_DECLINE` in
/// `dsv41_kernels.cu`). Deliberately NOT `1`: that is `cudaErrorInvalidValue`,
/// which the legacy `dsv41_rmsnorm_q` uses as its decline and which the caller
/// therefore cannot distinguish from a real failure. Only this exact code means
/// "keep the original launch"; every other non-zero code is an error.
const NORM_SPLIT_DECLINE: c_int = 0x7FFF;

/// `DSV41_CNORM_SPLIT=1` routes [`Device::hc_collapse_norm`] through the
/// dim-split entry when the shape fits. Default OFF (numerically equivalent, not
/// bit-identical — see the kernel header).
fn cnorm_split_wanted() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_CNORM_SPLIT").map(|v| v != "0").unwrap_or(false))
}

/// `DSV41_NORM_SPLIT=1` routes [`Device::rmsnorm_rows`] through the dim-split
/// entry when the shape fits. Default OFF, same contract as
/// [`cnorm_split_wanted`]. [`Device::rmsnorm_rows_on`] (the verify attention
/// fork's side-stream kv norm) is deliberately NOT routed: its `[row]` slots
/// would be shared with a concurrent main-stream split launch. See the kernel
/// header's PRECONDITION note.
fn norm_split_wanted() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_NORM_SPLIT").map(|v| v != "0").unwrap_or(false))
}

/// The `dim` cut for the split entries: one chunk per 1024 elements (the
/// blockDim both originals already use, so each chunk is one full-width pass of
/// the same tree), clamped to `[1, NORM_SPLIT_MAXC]`. `DSV41_NORM_SPLIT_NC`
/// overrides it so the chunk count can be swept without a rebuild; a value the
/// kernel cannot take declines there and the caller keeps the original launch.
fn norm_split_chunks(dim: i32) -> i32 {
    static NC: std::sync::OnceLock<Option<i32>> = std::sync::OnceLock::new();
    let want = *NC.get_or_init(|| {
        std::env::var("DSV41_NORM_SPLIT_NC")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
    });
    let nc = want.unwrap_or_else(|| (dim + 1023) / 1024);
    nc.clamp(1, NORM_SPLIT_MAXC)
}

/// `true` when the split entry should be issued for this `(rows, dim)`: the gate
/// is on, the `.so` exports the symbol, and the shape is one the C side accepts
/// (`rows <= NORM_SPLIT_MAXR`, at least two chunks, every chunk non-empty).
fn norm_split_fits(rows: i32, dim: i32, nc: i32) -> bool {
    rows > 0 && rows <= NORM_SPLIT_MAXR && nc >= 2 && nc <= NORM_SPLIT_MAXC && dim >= nc
}
