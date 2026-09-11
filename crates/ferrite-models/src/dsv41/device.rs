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
    /// (gemm_fp8_mx, apply_rope) pair. Returns 1 when the shape cannot take it.
    gemm_fp8_mx_rope: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *const u8, *const u8, *const f32, *mut f32,
            c_int, c_int, CuStream,
            *const f32, *const f32, *const c_int, c_int, c_int, c_int, c_int, c_int, c_int,
        ) -> c_int,
    >,
    /// RoPE fusion for the two-family GEMV: family 1 rotates with `rope_hd1`,
    /// family 2 with `rope_hd2` (the wq_b / idx_wq_b pair). Optional, like
    /// `gemm_fp8_mx_rope`.
    gemm_fp8_mx2_rope: Option<
        unsafe extern "C" fn(
            *const u8, *const f32,
            *const u8, *const u8, *const f32, *mut f32, c_int,
            *const u8, *const u8, *const f32, *mut f32, c_int,
            c_int, CuStream,
            *const f32, *const f32, *const c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int,
        ) -> c_int,
    >,
    /// A5: the same M=1 w2 GEMV with the trailing `ferrite_add` folded into its
    /// epilogue (`out += w @ a`). A separate symbol, so a stale `.so` simply has
    /// no entry and the caller keeps the gemm_fp8_mx + add_inplace pair. Returns
    /// 1 when the shape cannot use the GEMV.
    gemm_fp8_mx_add: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *const u8, *const u8, *const f32, *mut f32,
            c_int, c_int, c_int, CuStream,
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
    gemv_f32: Option<unsafe extern "C" fn(*const f32, *const f32, *mut f32, c_int, c_int, CuStream) -> c_int>,
    argmax: Option<unsafe extern "C" fn(*const f32, *mut c_int, c_int, *mut c_int, CuStream) -> c_int>,
    window_idxs: Option<unsafe extern "C" fn(*mut i32, *const c_int, c_int, CuStream) -> c_int>,
    comp_placeholder:
        Option<unsafe extern "C" fn(*mut i32, *const c_int, c_int, c_int, CuStream) -> c_int>,
    ring_append:
        Option<unsafe extern "C" fn(*mut f32, *const f32, *const c_int, c_int, c_int, CuStream) -> c_int>,
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
            *const c_int, CuStream,
        ) -> c_int,
    >,
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
    swiglu_limit_batched:
        Option<unsafe extern "C" fn(*mut f32, c_int, c_int, f32, i64, c_int, CuStream) -> c_int>,
    ar_reduce: Option<
        unsafe extern "C" fn(*mut f32, *const f32, i64, i64, c_int, *const c_uint, c_uint, CuStream) -> c_int,
    >,
    compressor_pool: Option<
        unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32, *mut f32, *mut f32, *mut c_int, c_int, c_int, c_int, c_int, c_int, *const c_int, f32, CuStream) -> c_int,
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
            gemm_fp8_mx_add: ko!(rt, "dsv41_gemm_fp8_mx_add"),
            quant_fp8: km!(rt, "dsv41_quant_fp8"),
            quant_fp4: km!(rt, "dsv41_quant_fp4"),
            expert_gate_up_fp4: km!(rt, "dsv41_expert_gate_up_fp4"),
            expert_down_fp4: km!(rt, "dsv41_expert_down_fp4"),
            engram_hash: km!(rt, "dsv41_engram_hash"),
            engram_gather: km!(rt, "dsv41_engram_gather"),
            sparse_attn: km!(rt, "dsv41_sparse_attn"),
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
            gemv_f32: ko!(rt, "dsv41_gemv_f32"),
            argmax: ko!(rt, "dsv41_argmax"),
            engram_hash_step: ko!(rt, "dsv41_engram_hash_step"),
            window_idxs: ko!(rt, "dsv41_window_idxs"),
            comp_placeholder: ko!(rt, "dsv41_comp_placeholder"),
            compress_commit: ko!(rt, "dsv41_compress_commit"),
            ring_append: ko!(rt, "dsv41_ring_append"),
            index_k_publish: ko!(rt, "dsv41_index_k_publish"),
            expert_gate_up_fp4_indirect: ko!(rt, "dsv41_expert_gate_up_fp4_indirect"),
            expert_down_fp4_indirect: ko!(rt, "dsv41_expert_down_fp4_indirect"),
            expert_gate_up_fp4_batched: ko!(rt, "dsv41_expert_gate_up_fp4_batched"),
            expert_down_fp4_batched: ko!(rt, "dsv41_expert_down_fp4_batched"),
            moe_down_reduce: ko!(rt, "dsv41_moe_down_reduce"),
            expert_down_reduce_fp4_batched: ko!(rt, "dsv41_expert_down_reduce_fp4_batched"),
            swiglu_limit_batched: ko!(rt, "dsv41_swiglu_limit_batched"),
            ar_reduce: ko!(rt, "dsv41_ar_reduce"),
            route_topk: ko!(rt, "dsv41_route_topk"),
            compressor_pool: ko!(rt, "dsv41_compressor_pool"),
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
            embed_expand_dev: km!(rt, "ferrite_embed_expand_dev"),
            f32_to_bf16: km!(rt, "ferrite_f32_to_bf16"),
            bf16_to_f32: km!(rt, "ferrite_bf16_to_f32"),
            p2p_ar_v5: ko!(rt, "ferrite_p2p_ar_v5"),
            p2p_ar_pubred_v5: ko!(rt, "ferrite_p2p_ar_pubred_v5"),
            p2p_ar_v5_hcpost: ko!(rt, "ferrite_p2p_ar_v5_hcpost"),
        };
        Ok(Device { rt, kernels, stream })
    }

    // ------------------------------------------- shared device primitives
    // Every method below forwards to the shared `ferrite_kernel::devrt`
    // runtime; the model layer owns no cudart/cublas binding of its own.

    pub fn stream(&self) -> CuStream {
        self.stream
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

    /// True when the loaded .so carries the fused down+reduce entry point
    /// (`dsv41_expert_down_reduce_fp4_batched`). A stale .so leaves
    /// DSV41_DOWN_FUSE inert and the (batched down, moe_down_reduce) pair runs.
    pub fn supports_down_fuse(&self) -> bool {
        self.kernels.expert_down_reduce_fp4_batched.is_some()
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
        let rc = unsafe {
            (self.kernels.gemm_fp8_mx)(
                a, a_scale, w, w_scale, bias, out, m, n, k, self.stream,
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
                self.stream,
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
                self.stream,
                rope_cos,
                rope_sin,
                rope_base,
                rope_mul,
                rope_off,
                rope_step,
                rope_inverse as i32,
                rope_rd,
                rope_hd,
            )
        };
        if rc == 1 {
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
                self.stream,
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
            )
        };
        if rc == 1 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_mx2_rope")?;
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
        let f = self.need(self.kernels.gemm_fp8_mx_add, "dsv41_gemm_fp8_mx_add")?;
        let rc =
            unsafe { f(a, a_scale, w, w_scale, bias, out, m, n, k, self.stream) };
        if rc == 1 {
            return Ok(false);
        }
        self.kerr(rc, "dsv41_gemm_fp8_mx_add")?;
        Ok(true)
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
        let rc = unsafe {
            (self.kernels.quant_fp8)(
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
        let rc = unsafe {
            (self.kernels.apply_rope)(
                x, cos, sin, rows, row_len, dim, half, base, mul, off, step, inverse as i32,
                self.stream,
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
        let f = match self.kernels.rmsnorm_rope {
            Some(f) => f,
            None => return Ok(false),
        };
        let rc = unsafe {
            f(x, w, out, cos, sin, n, dim, rope_len, half, base, mul, off, step, inverse as i32,
              eps, self.stream)
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
        let f = self.need(self.kernels.compressor_pool, "dsv41_compressor_pool")?;
        let rc = unsafe {
            f(
                kvp, scp, norm_w, state_kv, state_score, latents, out_rows, b, seqlen, head_dim,
                ratio, start_pos, pos_ctr, eps, self.stream,
            )
        };
        self.kerr(rc, "dsv41_compressor_pool")
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
        let f = self.need(self.kernels.swiglu_limit, "dsv41_swiglu_limit")?;
        let rc = unsafe { f(gate_up, rows, inter, limit, self.stream) };
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
        let f = self.need(self.kernels.swiglu_limit_q, "dsv41_swiglu_limit_q")?;
        let rc = unsafe { f(gate_up, rows, inter, limit, xq, xsc, self.stream) };
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
            f(partial, staging_tbl, ready_tbl, epoch, staging_local, ready_local, out, n, world,
              my_rank, stride, self.stream)
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
            f(partial, staging_tbl, ready_tbl, epoch, staging_local, ready_local, out, n, world,
              my_rank, stride, hc_res, hc_post, hc_comb, hc_n, hc_h, self.stream)
        };
        // 1 == the launcher declined the shape (see `ferrite_p2p_ar_v5_hcpost`);
        // the caller then runs the unfused pair as before.
        if rc == 1 {
            return Ok(false);
        }
        self.kerr(rc, "ferrite_p2p_ar_v5_hcpost")?;
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
        let f = self.need(self.kernels.compress_commit, "dsv41_compress_commit")?;
        let rc = unsafe {
            f(latent, cos, sin, ring, out_rows, clen, hd, rope_dim, half, window, ratio,
              self.stream)
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

    /// The single 4-byte host read per decode step (EOS check + printing). The
    /// token itself stays on the device; only this value crosses back.
    pub fn gemv_f32(&self, w: *const f32, x: *const f32, out: *mut f32, n: i32, k: i32) -> Result<()> {
        let f = self.need(self.kernels.gemv_f32, "dsv41_gemv_f32")?;
        let rc = unsafe { f(w, x, out, n, k, self.stream) };
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
    ) -> Result<()> {
        let f = self.need(
            self.kernels.expert_gate_up_fp4_batched,
            "dsv41_expert_gate_up_fp4_batched",
        )?;
        let rc = unsafe {
            f(
                a, a_scale, out, out_slot_stride, rows, dim, inter, limit, slots, w1_base,
                w1_stride, w1s_base, w1s_stride, w3_base, w3_stride, w3s_base, w3s_stride, ids,
                self.stream,
            )
        };
        self.kerr(rc, "dsv41_expert_gate_up_fp4_batched")
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
        let rc = unsafe { (self.kernels.rmsnorm)(x, w, out, n, dim, eps, self.stream) };
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
