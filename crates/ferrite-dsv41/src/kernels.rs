//! Kernel ABI for DeepSeek-V4.1-Flash.
//!
//! `kernels/cuda/dsv41_kernels.cu` implements exactly these `extern "C"`
//! symbols; the model chain calls them through these declarations. The ABI is
//! the contract between the Rust chain and the CUDA kernels, so the semantics
//! of every entry point are spelled out here (and mirrored by the CPU golden
//! implementations in [`crate::ops`]).
//!
//! # Performance contract (non-negotiable)
//!
//! * No weight is ever dequantised into a bf16/f32 buffer. fp8/fp4 weights stay
//!   packed in device memory for the whole run.
//! * Every large matmul is a tensor-core MMA over the native format.
//!
//! # Verified instruction availability on sm_103a (B300, CUDA 13.2, ptxas)
//!
//! | form | status |
//! |---|---|
//! | `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` | **works** — dense path |
//! | `mma.sync...kind::f8f6f4.f32.e2m1.e2m1.f32` | **REJECTED**: `Instruction 'mma with FP6/FP4 ...' not supported on .target 'sm_103a'` |
//! | any other `mma.sync` fp4/e2m1 form | rejected the same way |
//! | `tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X` | the only fp4 entry point on Blackwell |
//!
//! The probe was run with an explicit
//! `-gencode arch=compute_103a,code=sm_103a` and the generated PTX confirmed
//! `.target sm_103a`, so this is not a missing-`a`-suffix artefact. There is no
//! warp-level fp4 MMA on this part: fp4 lives on the 5th-generation tensor
//! cores (`tcgen05`), i.e. tensor-memory accumulators, shared-memory operand
//! descriptors, block-scale descriptors and `tcgen05.commit` + mbarrier
//! completion (CCCL ships wrappers in
//! `cuda/__ptx/instructions/generated/tcgen05_mma.h`).
//!
//! The expert kernels therefore have two implementations behind one ABI:
//!
//! 1. **primary** — `tcgen05.mma ... kind::mxf4` with the checkpoint's own
//!    e8m0 / k-block-32 scales, which is exactly the hardware MXFP4 layout;
//! 2. **fallback** — a *lossless* fp4 -> e4m3 re-encode at load time (the
//!    reference's own `convert.py::cast_e2m1fn_to_e4m3fn`, exact because an
//!    e2m1 value times a power-of-two offset stays representable in e4m3) fed
//!    to the proven fp8 `m16n8k32` MMA.
//!
//! Neither path dequantises to bf16/f32. The fallback doubles the expert
//! weight bytes (1 vs 0.5 per parameter), which is why the tcgen05 form is the
//! performance target.
//! * Block scales (ue8m0) are applied per k-block in the epilogue with a
//!   separate accumulator, exactly like the reference `fp8_gemm_kernel`:
//!   `acc += dot(a_k, b_k) * scale_a[row, kblk] * scale_b[nblk, kblk]`.
//! * Runtime activations stay fp8/fp4 (window KV fp8/128, compressed KV fp4/16,
//!   indexer q,k fp4/32).

#![allow(non_snake_case)]

use std::ffi::c_void;

pub type CuStream = *mut c_void;

extern "C" {
    // ---------------------------------------------------------------- dense fp8
    /// `out[m, n] = a[m, k] @ w[n, k]^T`, a fp8 e4m3 with per-(row, k/32) ue8m0
    /// scales and `w` fp8 e4m3 with per-(n/32, k/32) ue8m0 scales.
    /// `bias` may be null. Uses m16n8k32 fp8 MMA with the epilogue scheme above.
    pub fn dsv41_gemm_fp8_mx(
        a: *const u8,
        a_scale: *const u8,
        w: *const u8,
        w_scale: *const u8,
        bias: *const f32,
        out: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        stream: CuStream,
    ) -> i32;

    /// Activation quantisation, fp8 e4m3, `block` elements per scale.
    /// `round_scale` selects the power-of-two (ue8m0) scale.
    pub fn dsv41_quant_fp8(
        x: *const f32,
        y: *mut u8,
        scale: *mut f32,
        rows: i32,
        cols: i32,
        block: i32,
        round_scale: i32,
        stream: CuStream,
    ) -> i32;

    /// Activation quantisation, fp4 e2m1 (I8-packed output, 2 per byte).
    pub fn dsv41_quant_fp4(
        x: *const f32,
        y: *mut u8,
        scale: *mut f32,
        rows: i32,
        cols: i32,
        block: i32,
        round_scale: i32,
        stream: CuStream,
    ) -> i32;

    // --------------------------------------------------------------- experts fp4
    /// Fused gate+up expert projection: `[rows, dim] x W1/W3[inter, dim] -> [rows, 2*inter]`
    /// (gate first, then up). Weights are fp4 I8-packed with per-row-per-32
    /// ue8m0 scales; activations are fp4-quantised on the fly; the MMA is the
    /// native e2m1 one. `probs` (optional) scales each row.
    pub fn dsv41_expert_gate_up_fp4(
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
        stream: CuStream,
    ) -> i32;

    /// Down projection of the experts: `[rows, inter] x W2[dim, inter] -> [rows, dim]`,
    /// accumulating the per-row routing weight.
    pub fn dsv41_expert_down_fp4(
        act: *const f32,
        w2: *const u8,
        w2_scale: *const u8,
        weight: *const f32,
        out: *mut f32,
        rows: i32,
        dim: i32,
        inter: i32,
        stream: CuStream,
    ) -> i32;

    // ----------------------------------------------------------------- engram
    /// Gather `n_cols` table rows per token, dequantise (fp8 + ue8m0 row-32
    /// scales) and write `[rows, n_cols * head_dim]` bf16/f32 rows.
    pub fn dsv41_engram_gather(
        table: *const u8,
        table_scale: *const u8,
        hash_ids: *const i64,
        out: *mut f32,
        rows: i32,
        n_cols: i32,
        head_dim: i32,
        part_start: i64,
        part_rows: i64,
        stream: CuStream,
    ) -> i32;

    /// n-gram hash ids for one batch row (`NgramHashState.forward`).
    pub fn dsv41_engram_hash(
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
        stream: CuStream,
    ) -> i32;

    // ----------------------------------------------------- sparse attention
    /// fp8 window-KV append into the ring buffer (`slot = pos % window`).
    pub fn dsv41_window_append(
        kv: *const f32,
        cache: *mut u8,
        cache_scale: *mut f32,
        rows: i32,
        head_dim: i32,
        window: i32,
        start_pos: i32,
        stream: CuStream,
    ) -> i32;

    /// The sparse attention itself: gather `topk` positions out of `kv`
    /// (window + compressed concatenated along the position axis), online
    /// softmax, then the sink in the denominator.
    pub fn dsv41_sparse_attn(
        q: *const f32,
        kv: *const f32,
        sink: *const f32,
        idxs: *const i32,
        out: *mut f32,
        b: i32,
        m: i32,
        h: i32,
        d: i32,
        n: i32,
        topk: i32,
        scale: f32,
        stream: CuStream,
    ) -> i32;

    /// The indexer side attention: rectified scores -> weights_proj -> top-k ->
    /// position-sorted indices with `offset` and -1 markers.
    pub fn dsv41_indexer_topk(
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
        uses_candidates: i32,
        stream: CuStream,
    ) -> i32;

    /// Level one of the indexer: mark the `topk_blocks` best blocks
    /// (the newest, partially filled block is pinned).
    pub fn dsv41_candidate_blocks(
        logits: *const f32,
        compress_lens: *const i32,
        mask: *mut u8,
        rows: i32,
        n_pos: i32,
        topk_blocks: i32,
        block_size: i32,
        stream: CuStream,
    ) -> i32;

    /// `Compressor.forward`: pool `ratio` tokens into one latent with a softmax
    /// gate (ratio 1 = plain projection). Returns `out_rows` (0 = no group
    /// completed on this step, nothing written).
    pub fn dsv41_compressor(
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
        eps: f32,
        stream: CuStream,
    ) -> i32;

    /// YaRN frequency table (dual theta: the compressed path rotates at its own
    /// theta because one latent stands for `ratio` tokens).
    pub fn dsv41_rope_precompute(
        cos: *mut f32,
        sin: *mut f32,
        dim: i32,
        seqlen: i32,
        original_seq_len: i32,
        base: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        stream: CuStream,
    ) -> i32;

    /// Rotary embedding over the trailing `dim` lanes; `inverse` negates.
    pub fn dsv41_apply_rope(
        x: *mut f32,
        cos: *const f32,
        sin: *const f32,
        rows: i32,
        row_len: i32,
        dim: i32,
        half: i32,
        pos0: i32,
        step: i32,
        inverse: i32,
        stream: CuStream,
    ) -> i32;

    /// `hc_split_sinkhorn`: mixes -> pre/post/comb plus the Sinkhorn iteration
    /// (identical geometry to GLM-5.3-Flash, hc 4 / 20 iterations / eps 1e-6).
    pub fn dsv41_hc_mixes(
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
        stream: CuStream,
    ) -> i32;

    /// MoE routing: sqrtsoftplus scores, selection bias, top-k, normalisation,
    /// route scale. `hist[i]` receives the token count of expert i (for the
    /// grouped expert dispatch).
    pub fn dsv41_moe_route(
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
        norm_topk_prob: i32,
        route_scale: f32,
        score_func: i32,
        stream: CuStream,
    ) -> i32;
}

/// Convenience: the reference's `score_func` selector.
pub mod score_func {
    pub const SOFTMAX: i32 = 0;
    pub const SIGMOID: i32 = 1;
    pub const SQRTSOFTPLUS: i32 = 2;
}
