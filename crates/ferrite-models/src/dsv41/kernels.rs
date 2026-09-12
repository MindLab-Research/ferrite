//! Kernel ABI for DeepSeek-V4.1-Flash.
//!
//! `kernels/cuda/dsv41_kernels.cu` implements exactly these `extern "C"`
//! symbols; the model chain calls them through these declarations. The ABI is
//! the contract between the Rust chain and the CUDA kernels, so the semantics
//! of every entry point are spelled out here (and mirrored by the CPU golden
//! implementations in [`crate::dsv41::ops`]).
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
//! The routed **experts are fp4 and must be computed on fp4 tensor cores**
//! (NVFP4 / MXFP4 on Blackwell) — `tcgen05.mma ... kind::mxf4` with the
//! checkpoint's own e8m0 / k-block-32 scales, which is exactly the hardware MX
//! layout. Routing fp4 data through an fp8 GEMM is explicitly forbidden (it
//! would double the expert weight bytes on the dominant term of the step), so
//! this ABI deliberately has **no fp8 expert entry point**.
//!
//! The dense weights, by contrast, *are* fp8 e4m3 in the checkpoint and stay
//! fp8 — that is their native format, and the reference computes them the same
//! way (its `fp8_gemm_kernel`).
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
    ///
    /// The trailing group is the AR v5 store fusion (`staging_tbl`/`epoch`/`world`/
    /// `my_rank`/`stride`, M=1 only) followed by B1's fp8 row compression
    /// (`xq`/`xsc`, M=1 only, `n % 32 == 0`, `xq` must not alias `a`). All of them
    /// default to null/0, i.e. the plain GEMM, and `Device` (device.rs) is the
    /// binding the engine actually calls.
    pub fn dsv41_gemm_fp8_mx(
        a: *const u8,
        // a_scale: f32 power-of-two activation scales, one per (row, k/32) --
        // the reference's `scales_a` type. (w_scale below is ue8m0 BYTES.)
        a_scale: *const f32,
        w: *const u8,
        w_scale: *const u8,
        bias: *const f32,
        out: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        stream: CuStream,
    ) -> i32;

    /// A5: the M=1 w2 GEMV with the caller's trailing `add_inplace` folded into
    /// the epilogue (`out += a @ w^T`). Returns 1 when the shape cannot use the
    /// GEMV, so the caller keeps the gemm_fp8_mx + add_inplace pair.
    pub fn dsv41_gemm_fp8_mx_add(
        a: *const u8,
        a_scale: *const f32,
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

    /// tcgen05 MXFP4 gate/up (Phase-1 skeleton, `DSV41_TCGEN05_GATEUP_MXF4`):
    /// swapAB `kind::mxf4.block_scale.scale_vec::2X`, BOTH operands packed
    /// e2m1, TMA ring `kRing=8`, `M=128` tile (= `dsv41_experts_mxf4.cu`'s
    /// `tc5::mxf4::expert_tcgen05_gateup_mxf4_kernel`).
    ///
    /// ⚠️ **Not in the stock build**: `build.sh` defines no
    /// `DSV41_TCGEN05_GATEUP_MXF4_SKELETON`, so a normal `.so` has no such
    /// symbol — `Device::supports_expert_tcgen05_mxf4()` is the probe and the
    /// proven GEMV/GEMM path stays in force.
    ///
    /// **ABI (18 params, 2026-09-12 gap resolution; was 11).** The three shape
    /// asymmetries vs `dsv41_expert_gate_up_fp4_batched` are CLOSED:
    ///   1. weight layout — the kernel takes the loader's TWO separate pools
    ///      (`w1` gate, `w3` up) as `*_base`/`*_stride` pairs and applies the
    ///      row split `r < inter ? w1[r] : w3[r - inter]` per weight row
    ///      (mirrors `mxf4_gemm_kernel`'s `b`/`b_hi` + `b_split`);
    ///   2. activation scale — `act_scale` is the quantiser's natural
    ///      `[dim/32]` **f32 power-of-two** array; the kernel converts it with
    ///      its own `f_pow2_to_ue8m0` (lossless for `round_scale=true`), so
    ///      `dsv41_quant_fp4`'s ABI and the whole SIMT path stay untouched;
    ///   3. expert indirection — `ids[slot]` selects the expert per `grid.y`
    ///      slot ON THE DEVICE, so the routing never reaches the host and the
    ///      launch arguments stay static (CUDA-graph safe).
    /// There is deliberately no separate direct-pointer parameter pair: with
    /// `ids == nullptr` the four bases ARE the direct pointers, so one form
    /// serves both the parity harness and the serve dispatch.
    ///
    /// Returns 0 when the gate is OFF (read once inside the `.so`, process
    /// level). A REJECTED shape/strides returns a nonzero CUDA error code
    /// (so a misaligned pool fails loudly instead of silently misplacing
    /// bytes); the launcher checks `dim % 128 == 0`, `dim % 64 == 0`,
    /// `dim <= 5120`, `rows = 2*inter`, `rows % 128 == 0` (⇒ `inter % 64 == 0`,
    /// `split = inter % 32 == 0`) and the 16-byte alignment of all four bases
    /// and all four strides.
    #[allow(clippy::too_many_arguments)]
    pub fn dsv41_expert_tcgen05_gate_up_mxf4(
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

    // NOTE (user directive, 2026-09-10): the routed experts are fp4 in the
    // checkpoint and MUST be computed with fp4 tensor cores (NVFP4 / MXFP4 on
    // Blackwell). Computing them through fp8 is forbidden -- there is
    // deliberately NO fp8 expert entry point in this ABI.

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

    /// COMPRESS_FUSE: `dsv41_compressor_pool` + `dsv41_compress_commit` (and the
    /// state carry) as ONE 1-block launch, for the DECODE shape only
    /// (`b == seqlen == 1`, `ratio > 1`, `start_pos > 0`). The caller
    /// (`chain_dev::compress_on`) enforces that shape gate plus
    /// `DSV41_COMPRESS_FUSE`; a non-decode shape returns `cudaErrorInvalidValue`.
    /// `out_rows` is a DEVICE pointer, and the `start_pos == 0` / `ratio == 1`
    /// cases keep the three-launch path (their state mapping differs).
    #[allow(clippy::too_many_arguments)]
    pub fn dsv41_compressor_fused(
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
        clen: *mut i32,
        b: i32,
        seqlen: i32,
        head_dim: i32,
        ratio: i32,
        rope_dim: i32,
        half: i32,
        window: i32,
        pos_ctr: *const i32,
        eps: f32,
        stream: CuStream,
    ) -> i32;

    /// COMPRESSOR-MROWS (`DSV41_COMPRESSOR_MROWS=1`, default OFF): the verify
    /// block's `seqlen = m` rows of the decode compressor in ONE 1-block launch —
    /// `dsv41_compressor_fused`'s three stages, rows ASCENDING inside the kernel
    /// (the state carry's slot is position-derived, so rows r and r+2 share a slot
    /// at ratio 2, and the commit's `*clen` is the cross-row shared state).
    /// SHAPE: `b == 1`, `seqlen >= 1`, `ratio > 1`; anything else returns
    /// `cudaErrorInvalidValue`. The caller (`chain_dev::compress_rows_fused`)
    /// enforces the shape gate plus the env flag.
    ///
    /// `clen_rows` / `latent_rows` are the READ SIDE's per-row snapshot outputs and
    /// may be NULL (see the kernel header): a caller that hoists the whole block's
    /// compressor must give its per-row readers `clen_rows[r]` (the counter after
    /// row r's commit) and its per-row publishes `latent_rows + r*hd` (row r's
    /// pooled row) instead of the block-final live counter / shared `latent`.
    #[allow(clippy::too_many_arguments)]
    pub fn dsv41_compressor_fused_mrows(
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
        clen: *mut i32,
        clen_rows: *mut i32,
        latent_rows: *mut f32,
        b: i32,
        seqlen: i32,
        head_dim: i32,
        ratio: i32,
        rope_dim: i32,
        half: i32,
        window: i32,
        pos_ctr: *const i32,
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
    // ------------------------------------------------------------- glue ops
    /// Engram write-back: `x[i,j,:] += gate * value[j,:]` where the gate is the
    /// reference's normalised-dot sigmoid (signed sqrt + sigmoid) of the stream
    /// against the key, and `kv` holds `[rows, hc*dim + dim]` (the wkv output
    /// over the gathered rows). `token_mask` may be null (text-only batches).
    pub fn dsv41_engram_apply(
        x: *mut f32,
        kv: *const f32,
        q_weight: *const f32,
        k_weight: *const f32,
        token_mask: *const u8,
        rows: i32,
        hc: i32,
        dim: i32,
        eps: f32,
        stream: CuStream,
    ) -> i32;

    /// Compressor, pooling half only: the projections run on the bf16
    /// tensor-core path (the checkpoint stores them bf16), so this entry takes
    /// their outputs and does the state carry, the gated pool, the RMSNorm and
    /// the out_rows decision. `out_rows` is a DEVICE pointer.
    #[allow(clippy::too_many_arguments)]
    pub fn dsv41_compressor_pool(
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
        eps: f32,
        stream: CuStream,
    ) -> i32;

    /// MoE routing from pre-computed gate scores (the checkpoint's gate is
    /// bf16, so the gate GEMM is a separate bf16 GEMM). Selection uses
    /// `act(score) + bias`; the weights come from the unbiased `act(score)`,
    /// normalised and scaled. `hist` may be null.
    pub fn dsv41_route_topk(
        scores: *const f32,
        bias: *const f32,
        weights: *mut f32,
        indices: *mut i32,
        hist: *mut i32,
        rows: i32,
        n_experts: i32,
        topk: i32,
        norm_topk_prob: i32,
        route_scale: f32,
        score_func: i32,
        stream: CuStream,
    ) -> i32;

    /// SwiGLU with the training clamps: `out[i] = silu(min(gate, limit)) *
    /// clamp(up, -limit, limit)`, applied in place over `[rows, inter]` for each
    /// half of the fused gate_up buffer.
    pub fn dsv41_swiglu_limit(
        gate_up: *mut f32,
        rows: i32,
        inter: i32,
        limit: f32,
        stream: CuStream,
    ) -> i32;

    /// A4: the same SwiGLU+clamp, plus the fp8 e4m3 pair the following GEMV
    /// consumes (one ue8m0-equivalent f32 scale per 32-wide block). Returns 1
    /// when `inter % 32 != 0`, so the caller keeps dsv41_swiglu_limit + quant.
    pub fn dsv41_swiglu_limit_q(
        gate_up: *mut f32,
        rows: i32,
        inter: i32,
        limit: f32,
        xq: *mut u8,
        xsc: *mut f32,
        stream: CuStream,
    ) -> i32;

    // ------------------------------------------------- batched MoE experts fp4
    // DSV41_MOE_BATCH (default OFF): ONE launch per (layer, direction) instead
    // of one per (layer, top-k slot). Grid.y is the slot; every block derives
    // its expert from `ids[blockIdx.y]`. Per-slot outputs are DISJOINT, so a
    // batched result is bit-identical to the sequential per-slot loop (same
    // experts, same per-row K dot order).
    /// Batched fp4 gate/up: `out` holds `slots` [2*inter] blocks, `out_slot_stride`
    /// floats apart. `a`/`a_scale` is the ONE shared quantised activation row.
    ///
    /// `ilv` (ABI 2): the w1/w3 pools are interleaved at an 8-byte granule
    /// (DSV41_EXPERT_ILV), so the fused gate/up body derives the up bytes from
    /// the gate pointer and reads both with one LDG.128. Only valid together
    /// with the fused layout; the launcher rejects `ilv` + unfused.
    #[allow(clippy::too_many_arguments)]
    pub fn dsv41_expert_gate_up_fp4_batched(
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
        stream: CuStream,
    ) -> i32;

    /// Load-time gate/up interleave (DSV41_EXPERT_ILV): a pure 8-byte-granule
    /// permutation of `g`/`u` into `dst`, `bytes` per side. Bit-identical data,
    /// halved load instructions in the fused gate/up GEMV.
    pub fn dsv41_interleave_gateup_fp4(
        g: *const u8,
        u: *const u8,
        dst: *mut u8,
        bytes: i64,
        stream: CuStream,
    ) -> i32;

    /// Batched fp4 down: WRITES the [slots][dim] scratch (no accumulation), with
    /// the routing weight PER SLOT at `row_weight[slot * rw_stride]`.
    #[allow(clippy::too_many_arguments)]
    pub fn dsv41_expert_down_fp4_batched(
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
        stream: CuStream,
    ) -> i32;

    /// `out[i] = sum over slots in ASCENDING slot order` — the same order the
    /// sequential `out[i] += x` loop used, hence bit-identical (fp addition is
    /// not associative).
    pub fn dsv41_moe_down_reduce(
        part: *const f32,
        out: *mut f32,
        n: i32,
        slots: i32,
        stream: CuStream,
    ) -> i32;

    /// down + reduce FUSED (DSV41_DOWN_FUSE, default ON): ONE launch computes
    /// every slot's fp4 down GEMV and sums the per-slot contributions in
    /// ASCENDING slot order into `out` (OVERWRITE). Bit-identical to
    /// `dsv41_expert_down_fp4_batched` + `dsv41_moe_down_reduce`, which stay as
    /// the DSV41_DOWN_FUSE=0 fallback. `act_base` holds `slots` slices
    /// `act_stride` floats apart and only the first `inter` floats of each (the
    /// swiglu half) are read.
    #[allow(clippy::too_many_arguments)]
    pub fn dsv41_expert_down_reduce_fp4_batched(
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
        stream: CuStream,
    ) -> i32;

    /// w2 L2 PREWARM (DSV41_W2_PREWARM, default OFF, `=1` enables): pull every slot's w2 rows
    /// (`dim * (inter/2)` bytes) plus their e8m0 scale rows (`dim * (inter/32)`)
    /// into L2 with `cp.async.bulk.prefetch.L2.global`, so the down GEMV that
    /// follows answers from L2 instead of HBM. FIRE-AND-FORGET: writes nothing,
    /// reads only `ids` (a router output, not the gate/up launch's output) and
    /// the weight pools, and ALWAYS returns 0 - the bytes it warms are exactly
    /// the bytes the down launch reads, so it is bit-exact by construction.
    /// Launch on the same stream, immediately after the gate/up launch: its
    /// grid rides the gate/up ramp-down under PDL.
    #[allow(clippy::too_many_arguments)]
    pub fn dsv41_w2_l2_prewarm(
        w2_base: *const u8,
        w2_stride: i64,
        w2s_base: *const u8,
        w2s_stride: i64,
        ids: *const i32,
        slots: i32,
        sel_bytes: i64,
        sc_bytes: i64,
        stream: CuStream,
    ) -> i32;

    /// Batched SwiGLU: grid.y = slot over `slots` consecutive [2*inter] blocks.
    pub fn dsv41_swiglu_limit_batched(
        gate_up: *mut f32,
        rows: i32,
        inter: i32,
        limit: f32,
        slot_stride: i64,
        slots: i32,
        stream: CuStream,
    ) -> i32;

    /// Gather `n` rows of `dim` floats by index (`src` rows may repeat).
    pub fn dsv41_gather_rows(
        src: *const f32,
        idx: *const i32,
        out: *mut f32,
        n: i32,
        dim: i32,
        stream: CuStream,
    ) -> i32;

    /// Accumulate `src[i, :] * weight[i]` into `dst[idx[i], :]`.
    pub fn dsv41_scatter_add_rows(
        src: *const f32,
        idx: *const i32,
        weight: *const f32,
        dst: *mut f32,
        n: i32,
        dim: i32,
        stream: CuStream,
    ) -> i32;

    /// DSpark draft head, ONE sequential step of `dspark.rs::forward_head`:
    /// bias `logits[step, :]` in place with `<markov_head[v], markov_embed[ids[step]]>`,
    /// sample `ids[step + 1]` from the biased row (a stable argmax - see the
    /// kernel comment for why this is bit-identical to `ops::gumbel_argmax` at
    /// the reference's constant `u == 1`), and score `confidence[step]`.
    ///
    /// `partial` is `[gridDim.x]` u64 scratch (size it to `MAX_BLOCKS` = 2048)
    /// and `ctr` is ONE u32 that must be zero at allocation: the kernel's last
    /// elected block resets it before returning, which is what makes the next
    /// step's launch (and a captured graph's replay) start clean.
    #[allow(clippy::too_many_arguments)]
    pub fn dsv41_dspark_markov_head(
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
        stream: CuStream,
    ) -> i32;

    // ------------------------------------------------ dspark snapshot/rollback
    //
    // The verify block's save/restore pair (dspark-verify-perf-plan P0). They
    // replace a host loop of small `cudaMemcpyAsync` calls -- one per ring slot
    // per layer per direction, ~520 stream submissions -- with ONE launch per
    // layer per direction. The slot arithmetic stays exactly
    // `(pos_base + j) % window` with `pos_base == pos + 1`; it is merely
    // evaluated on the device now, which is what makes the pair capturable.
    //
    // Both halves are pure element moves (one thread per destination element),
    // so the bytes are bit-identical to the memcpy sequence they replace.
    //
    /// Save `m` ring slots: `snap[j*hd + i] = ring[((pos_base+j) % win)*hd + i]`.
    pub fn dsv41_dspark_ring_save(
        snap: *mut f32,
        ring: *const f32,
        pos_base: i32,
        window: i32,
        hd: i32,
        m: i32,
        stream: CuStream,
    ) -> i32;

    /// The inverse of [`Self::dsv41_dspark_ring_save`], with a KEEP prefix:
    /// rows `0..keep` stay in the ring, only rows `keep..m` are restored.
    /// `keep >= m` is a successful no-op.
    pub fn dsv41_dspark_ring_restore(
        ring: *mut f32,
        snap: *const f32,
        pos_base: i32,
        window: i32,
        hd: i32,
        m: i32,
        keep: i32,
        stream: CuStream,
    ) -> i32;

    /// Save ONE compress-source layer's carry in a single launch: `state_kv`
    /// and `state_score` (each `ratio * hd` at their own base in the layer's
    /// `2 * max_ratio * hd` slice) plus `latent` (`hd`) plus the 4-byte device
    /// counters `clen` and `out_rows`.
    #[allow(clippy::too_many_arguments)]
    pub fn dsv41_dspark_comp_save(
        state_kv: *const f32,
        state_score: *const f32,
        latent: *const f32,
        clen: *const i32,
        out_rows: *const i32,
        snap_state: *mut f32,
        snap_latent: *mut f32,
        snap_clen: *mut i32,
        snap_out_rows: *mut i32,
        ratio: i32,
        max_ratio: i32,
        hd: i32,
        stream: CuStream,
    ) -> i32;

    /// The inverse of [`Self::dsv41_dspark_comp_save`].
    #[allow(clippy::too_many_arguments)]
    pub fn dsv41_dspark_comp_restore(
        state_kv: *mut f32,
        state_score: *mut f32,
        latent: *mut f32,
        clen: *mut i32,
        out_rows: *mut i32,
        snap_state: *const f32,
        snap_latent: *const f32,
        snap_clen: *const i32,
        snap_out_rows: *const i32,
        ratio: i32,
        max_ratio: i32,
        hd: i32,
        stream: CuStream,
    ) -> i32;
}

/// Convenience: the reference's `score_func` selector.
pub mod score_func {
    pub const SOFTMAX: i32 = 0;
    pub const SIGMOID: i32 = 1;
    pub const SQRTSOFTPLUS: i32 = 2;

}
