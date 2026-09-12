//! GPU forward chain for DeepSeek-V4.1-Flash.
//!
//! Config-driven throughout: layer roles (`is_kv_source` / `is_index_source`),
//! per-layer expert counts, compression ratios, the engram layer set and the
//! draft layers all come from [`Dsv41Config`]; nothing here hard-codes the
//! production geometry.
//!
//! # Scope of this file, stated plainly
//! Implemented and exercised end to end on the GPU:
//!   * embedding + hyper-connection expansion,
//!   * the hyper-connection chain (`hc_mixes` -> `hc_collapse` -> block ->
//!     `hc_post`), with the reference's pre-mix threading (block N's attention
//!     mix collapses block N's FFN, the initial mix is the one-hot [1,0,0,0]),
//!   * the MLA window path: latent q/kv projections (fp8 tensor cores), q/kv
//!     RMSNorm, RoPE, the window ring, sparse attention over the selected
//!     positions, inverse RoPE and the block-diagonal grouped output projection,
//!   * MoE: bf16 gate GEMM, `noaux_tc` routing, per-expert MXFP4 tensor-core
//!     GEMMs and the fp8 shared expert.
//!
//! **Not in this file yet** (each is a separate increment, and none of them is
//! silently skipped — `RunOpts` reports which are active):
//!   * the compressor + indexer paths (attention is window-only here),
//!   * the engram write-back,
//!   * DSpark drafting.
//!
//! One token per `step`: the KV ring is per-sequence, so a batched prefill
//! would need per-row ring state. The kernels are all batched-ready; this
//! restriction is in the driver, not the maths.

use std::ffi::c_void;

use ferrite_types::{FerriteError, Result};

use std::sync::Arc;

use crate::dsv41::config::{Dsv41Config, KvMode};
use crate::dsv41::tp::Collective;
use crate::dsv41::device::{CuStream, DevBuf, Device};
use crate::dsv41::dspark_dev::{DsparkDev, DSPARK_DRAFTS, DSPARK_TAP_SLOTS};
use crate::dsv41::load::{Dsv41DevWeights, LayerDev};

/// Runtime switches for isolating a stage during bring-up.
#[derive(Debug, Clone, Default)]
pub struct RunOpts {
    pub skip_experts: bool,
    pub skip_shared_expert: bool,
}

impl RunOpts {
    pub fn from_env() -> Self {
        let b = |k: &str| std::env::var(k).map(|v| v != "0").unwrap_or(false);
        RunOpts {
            skip_experts: b("DSV41_SKIP_EXPERTS"),
            skip_shared_expert: b("DSV41_SKIP_SHARED_EXPERT"),
        }
    }
}

/// Per-layer state that persists across steps.
struct LayerCache {
    /// window ring, `[window, head_dim]` f32 (the release stores it fp8; the
    /// ring is f32 in this increment — see the file header)
    ring: DevBuf,
    /// the selection for the current token: `[window + index_topk]` i32, -1 unused
    idxs: DevBuf,
    /// compressor carry state, `[ratio, head_dim]`, and the projections' outputs
    state_kv: DevBuf,
    state_score: DevBuf,
    kvp: DevBuf,
    scp: DevBuf,
    latent: DevBuf,
    out_rows: DevBuf,
    /// compressed rows published so far in this sequence
    compress_len: usize,
    /// pre-RoPE keys of those rows, `[max_compress, index_head_dim]`
    index_k: DevBuf,
}

/// The DSpark verify block's row count: `step_rows` accepts up to this many
/// tokens and every `_r` buffer below is sized for it. The production block is
/// `dspark_block_size` (6); the buffers are sized once here so a smaller `m`
/// (a short block at the end of a request) simply uses a prefix of them.
pub const VERIFY_ROWS: usize = 6;

/// What one shadow-mode DSpark step observed. See
/// [`DevChain::dspark_shadow_step`] for the orchestration: the step runs the
/// draft and a real verify block, then puts the main chain back exactly where
/// the single-row path left it, so nothing here is committed — the report is a
/// measurement of what the speculative path WOULD have emitted.
#[derive(Debug, Clone, Copy)]
pub struct DsparkShadowReport {
    /// The step's real output: `step_dev`'s argmax, i.e. exactly the token the
    /// chain emits with speculation disabled.
    pub next: u32,
    /// What the draft proposed for positions `pos + 1 ..= pos + DSPARK_DRAFTS`.
    pub drafts: [u32; DSPARK_DRAFTS],
    /// The verify block's per-row argmax: row `j` (position `pos + 1 + j`, fed
    /// `drafts[j]`) predicted `verify_out[j]` for `pos + 2 + j`.
    pub verify_out: [u32; DSPARK_DRAFTS],
    /// `k` (1..=DSPARK_DRAFTS + 1): the number of tokens the speculative path
    /// WOULD have emitted from this step — the accepted draft prefix plus one
    /// bonus token. `1` means no draft survived, i.e. `next` alone.
    pub accepted: usize,
    /// The draft phase (`import_tap` + `draft_forward` + the drafts D2H).
    pub draft_ms: f32,
    /// The verify phase (`step_rows`, one m-row forward).
    pub verify_ms: f32,
}

struct Scratch {
    h: DevBuf,     // [hc*dim]
    h2: DevBuf,    // [hc*dim] (hc_post lands here, then copies back)
    x: DevBuf,     // [dim]
    xn: DevBuf,    // [dim]
    xq: DevBuf,    // [dim] fp8 e4m3
    xsc: DevBuf,   // [dim/32 + 8] f32 scales
    /// T2 (MoE side): the ROUTED experts' fp4 packing of `xn`. Deliberately
    /// DISJOINT from `xq`/`xsc`: when it shared those buffers the pack clobbered
    /// the fp8 the hc tail had just emitted for `xn` (T1), so the shared expert
    /// below paid a second `quant1(xn)`. With its own scratch that fp8 survives
    /// and the shared expert's quant1 hits the still-set flag.
    xq4: DevBuf,   // [dim] fp4 nibbles (dim/2 bytes used)
    xsc4: DevBuf,  // [dim/32 + 8] f32 scales
    /// T1: set right after the hc tail emitted the fp8 quantisation of `xn`
    /// alongside its f32 write-back; the NEXT quant1 whose source is `xn` skips
    /// its launch (pointer-gated in quant1) and clears the flag. Any other
    /// quant1 (qr, o, ex_act, engram rows) is untouched - different source.
    /// Cell because quant1 takes &self (the whole lin/lin2 chain does).
    xq_of_xn_valid: std::cell::Cell<bool>,
    /// T2 (attention side): set when `rmsnorm(qr)` emitted the fp8 of its own
    /// normalised output through `ferrite_rmsnorm_q`; the NEXT quant1(qr) - the
    /// wq_b (and, under IDX_FUSE, idx_wq_b) projection's - skips its launch and
    /// clears the flag, exactly like `xq_of_xn_valid`. Consume-once is required
    /// here: `s.xq` is rewritten by the o/wo quantisations between the wq_b and
    /// the indexer's idx_wq_b, so only the FIRST qr consumer can be spared.
    xq_of_qr_valid: std::cell::Cell<bool>,
    /// NORM_FUSE: set by `attention()` when the wq_b rope GEMV took the fused
    /// path, i.e. `qr` is still the RAW wq_a output (no `rmsnorm_q` ran, so its
    /// normalisation happens inside each consumer's gemv prologue). `indexer()`
    /// reads it to send idx_wq_b down the same fused launch instead of
    /// `quant1(qr)` over the - now raw - buffer; it clears the flag when it
    /// consumes it. Written on EVERY `attention()` path, so a layer that never
    /// reaches the indexer cannot leak a stale value into the next step.
    qr_raw: std::cell::Cell<bool>,
    /// L2+L3: set by `attention()` when the wq_b + idx_wq_b pair was computed
    /// in ONE mx2 launch (`DSV41_IDX_FUSE`), so `indexer()` skips its own
    /// `lin(idx_wq_b)`. Cleared after the indexer consumes it (and whenever the
    /// fused launch declines the shape). Cell because the lin/lin2 chain takes
    /// &self while the indexer takes &mut self.
    idx_q_ready: std::cell::Cell<bool>,
    /// DSV41_ROPE_FUSE, attention side: set alongside `idx_q_ready` when the
    /// wq_b + idx_wq_b mx2 launch ALSO rotated `s.idx_q` in its epilogue, so the
    /// indexer skips its standalone `apply_rope(s.idx_q)`. Meaningful only while
    /// `idx_q_ready` is set (the two are written together on every path).
    idx_q_rope: std::cell::Cell<bool>,
    /// MoE scatter destination (written by the dispatch kernels; the host never
    /// reads it back, which is why rustc flags it).
    #[allow(dead_code)]
    pre: DevBuf,   // [hc]
    post: DevBuf,  // [hc]
    comb: DevBuf,  // [hc*hc]
    qr: DevBuf,    // [q_lora]
    q: DevBuf,     // [nh*head_dim]
    kv: DevBuf,    // [head_dim]
    o: DevBuf,     // [nh*head_dim]
    wo: DevBuf,    // [o_lora]
    logits: DevBuf,
    ids: DevBuf,   // [1] i32
    /// The device position counter: the argmax (the step's last kernel)
    /// advances it, so every kernel during the step reads a stable current pos.
    pos_ctr: DevBuf, // [1] i32
    /// Vocabulary-sliced lm_head only: the rank-local argmax publishes its
    /// packed (key | ~global index) comparison key here, and `argmax_pub_kernel`
    /// copies it into every peer's staging slot. Without it the sliced path
    /// would hand a NULL pointer to a kernel that dereferences it.
    argmax_packed: DevBuf, // [1] u64
    /// Per-layer compressed-KV counters, advanced by the compressor's commit
    /// kernel on the device (the host used to track compress_len and download
    /// `out_rows` to decide). Consumers read this instead of a launch argument.
    clen: DevBuf, // [n_layers] i32
    // MoE
    scores: DevBuf,    // [n_experts] f32
    route_idx: DevBuf, // [topk] i32
    route_w: DevBuf,   // [topk] f32
    /// [1] u32 — the fused gate GEMV's last-block election counter
    /// (`ferrite_gemv_bf16_v2_route`, DSV41_ROUTE_FUSE). Zeroed ONCE at build:
    /// the elected block resets it in place before its grid ends, so a
    /// captured-graph replay never needs a host memset here. A run that dies
    /// mid-grid leaves it non-zero, which would silently disable the election —
    /// DSV41_ROUTE_FUSE=0 is the recovery.
    route_ctr: DevBuf,
    ex_in: DevBuf,     // [dim]
    ex_act: DevBuf,    // [2*inter]
    ex_out: DevBuf,    // [dim]
    /// DSV41_MOE_BATCH only: the per-slot gate/up outputs, `[topk][2*inter]`.
    /// The sequential loop reused ONE ex_act per slot (overwrite); the batched
    /// gate/up writes every slot in one launch, so it needs disjoint slices.
    ex_act_b: DevBuf,
    /// DSV41_MOE_BATCH only: the per-slot down scratch, `[topk][dim]`. The
    /// batched down WRITES here (no cross-slot accumulation) and a fixed-order
    /// reduction sums the slots into `o` in the sequential order.
    ex_down_b: DevBuf,
    hist: DevBuf,      // [n_experts] i32
    idx_q: DevBuf,     // [index_n_heads * index_head_dim]
    idx_k: DevBuf,     // [index_head_dim]
    idx_w: DevBuf,     // [index_n_heads]
    // bf16 staging for the cuBLAS path
    bf16: DevBuf,
    // hc premix coefficients, kept DEVICE-resident and rotated between layers.
    // They used to round-trip through the host every layer (one download each for
    // attn_pre and ffn_pre, plus two uploads), and a download is a full device
    // sync — 45 layers x 2 syncs per step, each one draining the CPU/GPU
    // pipeline, which is where the ~7.5ms/layer went. Three slots, no aliasing.
    pre_a: DevBuf,
    /// The constant incoming premix [1,0,0,0], uploaded ONCE at reset; each step
    /// copies it into slot 0 with a 16-byte D2D (graph-capturable) instead of an H2D.
    premix_const: DevBuf,
    // ---- DSpark target-hidden tap (speculative decode) ----
    /// Per-copy mean of the target layers' attention inputs, `[3, dim]`
    /// (`dspark_target_slot(l)` = 0/1/2). Filled by a one-block `hc_collapse`
    /// hook in `layer()` whenever dspark is armed — fixed buffers, so the hook
    /// is graph-capturable with the rest of the step.
    dspark_tap: DevBuf,
    /// `[hc]` of `1/hc` — the mean weights for the tap's collapse.
    dspark_pre_mean: DevBuf,
    pre_b: DevBuf,
    pre_c: DevBuf,
    // ---- engram (n-gram memory write-back into the hc residual stream) ----
    /// hash ids for one token: `[n_engram_layers * n_hash_cols]` i64
    eng_ids: DevBuf,
    /// gathered table rows: `[n_hash_cols * engram_head_dim]` f32 (0 for rows
    /// another rank owns, so the collective sums one real row per column)
    eng_rows: DevBuf,
    /// the wkv projection's output: `[hc*dim + dim]` f32 (key then value)
    eng_kv: DevBuf,
    /// fp8 activation + per-32 scale for the wkv GEMM over `eng_rows`
    eng_xq: DevBuf,
    eng_xsc: DevBuf,
    // ---- B1: wo_a's quantised row compression ----
    /// The fp8 e4m3 bytes the wo_a gemv epilogue emits for `s.wo` (gated by
    /// DSV41_WO_QUANT_FUSE), which is what wo_b then consumes. It needs its OWN
    /// buffer rather than sharing `xq`/`xsc`: the wo_a gemv READS `xq` (the
    /// quantised attention output) as its input, and the blocks that stage that
    /// input late would read bytes the epilogue had already overwritten.
    wo_q: DevBuf,   // [n_groups*o_lora] fp8 bytes
    wo_qsc: DevBuf, // [n_groups*o_lora/32 + 8] f32
    // ---- chain-pair-grid-sync: wo_a -> wo_b grid-sync barrier state ----
    /// `[arrive, sense]` u32 pair for `gemm_fp8_wo_pair_kernel`'s device-wide
    /// barrier. Persistent and zeroed ONCE at init: the kernel resets `arrive` and
    /// flips `sense` on every launch, so a captured graph replays correctly and
    /// nothing re-initialises it per step. Must NOT be shared with any other
    /// barrier — two concurrent grids on one pair corrupt each other's arrive
    /// count (the wo pair runs on the main stream, serialised, which is what makes
    /// one pair sufficient).
    wo_bar: DevBuf, // [2] u32
    // ---- chain-pair-batch 链2: shared expert w1w3 -> swiglu -> w2 scratch ----
    /// The swiglu epilogue's fp8 pair (`dsv41_gemm_fp8_sh_pair`'s phase-1
    /// output / phase-2 activation). It needs its OWN buffers rather than
    /// `xq`/`xsc`: phase 1 READS `xq` (the quantised `xn`) as its activation
    /// while its epilogue WRITES these bytes, and inside one grid-sync launch
    /// there is no barrier between the read and the write — a block that emitted
    /// early would clobber bytes a later block still has to stage (the exact
    /// hazard `wo_q`/`wo_qsc` document for the wo pair).
    sh_q: DevBuf,   // [inter] fp8 bytes
    sh_qsc: DevBuf, // [inter/32 + 8] f32
    // ---- swapAB K-split last-block reduction scratch ----
    /// `[kSwapabKSplit][n]` f32 partial slots for `dsv41_gemm_fp8_swapab`'s ks > 1
    /// path: partition `kp` writes its 16-row tile's sums at `partial[kp*n + r]`
    /// and the LAST partition of the tile (per-tile ticket, `swapab_ctr`) reduces
    /// them into `out`. This is what replaced the launcher's `cudaMemsetAsync(out)`
    /// + `atomicAdd(&out[r])` pair — one fewer graph node per call and a FIXED
    /// reduction order (bit-deterministic) instead of an atomic race.
    ///
    /// Sized by the largest `n` ANY swapAB call site can pass (`swapab_n`), times
    /// the compile-time max K split (8 = `DSV41_SWAPAB_KSPLIT` in
    /// dsv41_kernels.cu). One buffer serves every call site: the calls are
    /// serialised on the main stream, and PDL's `cudaGridDependencySynchronize()`
    /// keeps a successor grid's slot writes behind this grid's end.
    swapab_part: DevBuf,
    /// One u32 arrival ticket per 16-row tile (`n/16` entries) for the same
    /// reduction. Must be ZERO before the first swapAB call; the kernel resets
    /// each entry after the elected block consumes it, so it is never re-zeroed
    /// per step (doing so would race the kernel's own reset — the `wo_bar`
    /// discipline).
    swapab_ctr: DevBuf,
    /// The `n` bound `swapab_part`/`swapab_ctr` were sized for. A call with a
    /// larger `n` is not routed to swapAB (the guard in `gemm_fp8_mx_or_swap`),
    /// so the scratch can never be overrun by a config the allocation missed.
    swapab_n: usize,
    // ==================== DSpark verify: the m-row buffers ====================
    //
    // `step_rows` runs a whole verify block (`m <= VERIFY_ROWS` rows) through the
    // same layer stack `step_body` runs one row of, so every per-row buffer above
    // needs an m-row twin here. They are deliberately SEPARATE from the single-row
    // ones: `step_rows` must not perturb the state a decode step leaves behind
    // (the whole point is that both paths keep working), and several of the
    // single-row buffers (`s.q`, `s.o`, `s.kv`, `s.logits`, ...) are far too small
    // for m rows anyway.
    //
    // Layout convention: row-major with the row stride taken from the GEOMETRIC
    // width each buffer was sized for (e.g. `nh*hd` for q/o, `dim` for x/xn).
    // Only the places that hand a WHOLE multi-row region to one launch (the AR
    // payloads, the hc kernels, `embed_expand_dev`, the head/gate loops written
    // per row) rely on contiguity, and each such call site says so.
    h_r: DevBuf,       // [m, hc*dim] residual stream (embed + hc expansion)
    h2_r: DevBuf,      // [m, hc*dim] hc_post staging (hc_post is not in-place for rows > 1)
    x_r: DevBuf,       // [m, dim] collapse output
    xn_r: DevBuf,      // [m, dim] normalised block input
    q_r: DevBuf,       // [m, nh*hd] wq_b output, row base at `r*nh*hd`
    kv_r: DevBuf,      // [m, hd] window KV
    qr_r: DevBuf,      // [m, q_lora] wq_a output / its normalisation
    o_r: DevBuf,       // [m, nh*hd] sparse attention output (after the inverse o-rope)
    wo_r: DevBuf,      // [m, n_groups*o_lora] wo_a output
    wo_out_r: DevBuf,  // [m, dim] attention block output (wo_b + the wo AR)
    moe_out_r: DevBuf, // [m, dim] MoE block output (routed + shared expert)
    ids_r: DevBuf,     // [m] i32 token ids
    argmax_r: DevBuf,  // [m] i32 per-row argmax (the head's output)
    logits_r: DevBuf,  // [m, vocab]
    /// [m, hc] the CONSTANT incoming premix, `[1,0,0,0]` per row (uploaded once per
    /// `step_rows`, the m-row twin of `premix_const`).
    premix_r: DevBuf,
    /// [m, hc] slot 1: this layer's attn_pre — the FFN collapse's premix, and after
    /// the last layer the final collapse's premix (the single-row `pre_b`).
    pre_r: DevBuf,
    /// [m, hc] slot 2: this layer's ffn_pre — the NEXT layer's incoming premix (the
    /// single-row `pre_c`).
    pre2_r: DevBuf,
    post_r: DevBuf,    // [m, hc] hc_post gate
    comb_r: DevBuf,    // [m, hc*hc] hc_post mix
    /// [m] i32 the row positions `pos_base + r`. The verify forward never advances
    /// the device counter, so every kernel that takes a position POINTER (the
    /// engram hash, `ring_append`, `window_idxs`) is pointed at this table instead
    /// of at `pos_ctr`.
    pos_rows: DevBuf,
    /// [m, window + index_topk] i32 the per-layer selection `sparse_attn` reads,
    /// row stride `window + index_topk` (the indexer's picks follow each row's
    /// window block at `+window`).
    idxs_r: DevBuf,
    /// [m, window] i32 the RAW `verify_ring_win` output. The kernel writes its
    /// index half with a row stride of `window` (it only ever wrote one row
    /// before), so it cannot fill `idxs_r`'s wider rows directly; each row's
    /// window block is copied from here into `idxs_r` (see `attention_rows`).
    idxs_win_r: DevBuf,
    idx_q_r: DevBuf,   // [m, index_n_heads*index_head_dim]
    idx_w_r: DevBuf,   // [m, index_n_heads]
    // ---- compressor, m rows ----
    kvp_r: DevBuf,     // [m, hd] the compressor's kv projection
    scp_r: DevBuf,     // [m, hd] the compressor's gate/score projection
    // ---- MoE, m rows ----
    scores_r: DevBuf,   // [m, n_experts] the bf16 gate's output
    route_idx_r: DevBuf, // [m, topk] i32
    route_w_r: DevBuf,   // [m, topk] f32
    xq4_r: DevBuf,       // [m, dim] fp4 nibbles (dim/2 bytes per row)
    xsc4_r: DevBuf,      // [m, dim/32 + 8] f32 scales
    /// [m, topk, 2*inter] routed gate|up output. The batched expert launchers
    /// process ONE activation row per call (their `rows` argument is validated but
    /// never enters the grid — see `moe_rows`), so this is a per-row slot block:
    /// row `r`'s slot `t` starts at `r*(topk*act_slot) + t*act_slot`.
    ex_act_r: DevBuf,
    /// [m, topk, dim] the DOWN_FUSE=0 fallback's per-slot scratch.
    ex_down_r: DevBuf,
    /// [m, 2*inter] the shared expert's gate|up row (one row at a time).
    sh_act_r: DevBuf,
    // ---- engram, m rows ----
    eng_ids_r: DevBuf,  // [m, n_engram_layers * n_cols] i64
    eng_rows_r: DevBuf, // [m, n_cols * engram_head_dim] f32
    eng_kv_r: DevBuf,   // [m, (hc+1)*dim] f32
    eng_xq_r: DevBuf,   // [m, n_cols*ehd] fp8 bytes
    eng_xsc_r: DevBuf,  // [m, n_cols*ehd/32 + 8] f32
    // ==================== DSpark shadow mode: the verify write-set save ======
    //
    // `dspark_shadow_step` runs a draft + a whole verify block once per step and
    // then puts the MAIN chain back exactly where the single-row path would have
    // left it (the verify's only lasting effect is the returned report). What the
    // block is about to write is copied in here first and copied back after.
    //
    // Layout: indexed BY LAYER — a layer that is neither a ring owner nor a
    // compress source never touches its slice. Both sides of the step recompute
    // the same owner/source predicates, so one index space means the save and the
    // restore cannot disagree about a slot. See `DevChain::dspark_snapshot` for
    // the inventory and the reasoning behind each member.
    /// `[n_layers][VERIFY_ROWS][head_dim]` f32 — the window-ring slots the block
    /// appends to, `(pos + 1 + j) % window` for row `j`.
    dspark_snap_ring: DevBuf,
    /// `[n_layers][2][max_ratio * head_dim]` f32 — the compressor carry:
    /// `state_kv` then `state_score`, each layer's own `ratio * head_dim` floats
    /// at its own base (the buffers are max-sized because `ratio` is per layer).
    dspark_snap_state: DevBuf,
    /// `[n_layers][head_dim]` f32 — the compressor's pooled latent row. NOT a
    /// per-step scratch: it is an index-key INPUT on later steps (see
    /// `dspark_snapshot`), so leaving the verify's latents behind would corrupt
    /// the main chain's published keys.
    dspark_snap_latent: DevBuf,
    /// `[n_layers]` i32 — the DEVICE compressed-row counters (`s.clen`).
    dspark_snap_clen: DevBuf,
    /// `[n_layers]` i32 — the compressor's `out_rows` decision.
    dspark_snap_out_rows: DevBuf,
}

/// Device-resident state for the engram n-gram hash: the compressed-token map,
/// the per-layer multipliers and the (prime, offset) columns, all uploaded ONCE,
/// plus the cross-step token cache and the position counter (zeroed at reset).
struct EngDev {
    map: DevBuf,    // [vocab] i64
    cache: DevBuf,  // [max_seq] i64
    mults: DevBuf,  // [n_layers * 4] i64
    lms: DevBuf,    // [n_layers * n_cols] u64
    offs: DevBuf,   // [n_layers * n_cols] u64
    max_seq: usize,
}

/// DSV41_ENG_HOST=1 keeps the host hash + per-step upload (the A/B fallback).
fn eng_host() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_ENG_HOST").map(|v| v != "0").unwrap_or(false))
}

/// DSV41_MOE_BATCH=1 collapses the routed-expert fp4 GEMV family from one
/// launch per (layer, top-k slot) to one launch per (layer, direction), which
/// is the MoE family's real lever (the per-call launch floor is ~3.05 us and
/// the inner loops are measured-exhausted — see docs/agent/perf-roadmap.md).
///
/// DEFAULT OFF: the batched path is a separate kernel set and the sequential
/// path is the live verified one. Read ONCE and cached (the house rule from
/// dsv41_glue.cu's g_hc_spread and eng_host() above — a per-call getenv is a
/// hot-path slip), and `"0"` means OFF even though it is "set".
/// `pub(crate)`: the loader's `ilv_ok` reads these same gates to decide the
/// routed experts' weight layout, so the two sides can never drift.
pub(crate) fn moe_batch() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_MOE_BATCH").map(|v| v != "0").unwrap_or(true))
}

/// Mirrors dsv41_experts_mxf4.cu's dispatch test
/// (`rows == 1 && getenv("DSV41_NO_GEMV_FP4") == nullptr`, a BARE getenv: any
/// value at all, even "0", sends the rows==1 call through the tcgen05 GEMM).
/// That GEMM reads the plain w1/w3 layout, so the loader's `ilv_ok` must see
/// this knob before it interleaves the pools.
pub(crate) fn no_gemv_fp4() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var_os("DSV41_NO_GEMV_FP4").is_some())
}

/// DSV41_EXPERT_TCGEN05_MXF4=1 arms the tcgen05 MXFP4 swapAB gate/up
/// (`tc5::mxf4::expert_tcgen05_gateup_mxf4_kernel`, `dsv41_experts_mxf4.cu:4007`).
/// Mirror of the launcher's own gate test (`e[0] == '1'`, a strict "1..." prefix,
/// NOT the usual `!= "0"`: the default is OFF and stays OFF if the value is
/// anything else). Read ONCE and cached: the `.so` reads the same variable once
/// per process, so a per-call getenv here could only ever add a hot-path slip and
/// a capture hazard (plan §5), never a different decision.
///
/// ⚠️ The gate is armed by the `.so`, which loads with a
/// `DSV41_TCGEN05_GATEUP_MXF4_SKELETON` build only — the stock build has no such
/// symbol and `Device::supports_expert_tcgen05_mxf4()` is false.
pub(crate) fn expert_tcgen05_mxf4() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("DSV41_EXPERT_TCGEN05_MXF4").map(|v| v.starts_with('1')).unwrap_or(false)
    })
}

/// One-shot notice for an ARMED-but-undispatchable mxf4 gate. The `.so` reads the
/// same env var itself, so an operator who exports `DSV41_EXPERT_TCGEN05_MXF4=1`
/// believes the step now runs the tcgen05 gate/up — while `moe()` keeps issuing
/// the proven GEMV (see the call site: no symbol in the .so, the interleaved
/// weight layout, or no batched path). A silent no-op there is the project's #1
/// measurement-bias trap (an "ON" arm that measures the OLD path), so it is said
/// out loud once.
fn tcgen05_mxf4_skipped_note(reason: &str, so_has_symbol: bool) {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        eprintln!(
            "warning: DSV41_EXPERT_TCGEN05_MXF4 is set, but the routed MoE still dispatches the \
             gate/up to the proven GEMV/GEMM: {reason}. Any A/B run with this gate ON measures \
             the OLD path. (symbol present in .so: {so_has_symbol})"
        );
    });
}

/// DSV41_DOWN_FUSE=0 reverts the batched down direction to the two-launch
/// (expert_down_fp4_batched + moe_down_reduce) pair. DEFAULT ON
/// (`.unwrap_or(true)`: f3b1be1 had flipped it OFF after the round-18 corruption,
/// then it was flipped back ON — the .cu header note calls the old "OFF" text
/// stale, and the nsys v3 profile sees the fused kernel 40x/step); the
/// fused kernel produces the same bits in one launch (see
/// expert_gemv_fp4_down_reduce_kernel in dsv41_experts_mxf4.cu for the contract:
/// ascending serial slot loop, verbatim K loop, explicitly rounded per-slot
/// product). Read ONCE and cached like the other gates, and `"0"` means OFF even
/// though it is "set".
fn down_fuse() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_DOWN_FUSE").map(|v| v != "0").unwrap_or(true))
}

/// Mirrors the CUDA launcher's `g_expert_fp4_mode` (dsv41_experts_mxf4.cu:694):
/// unset => 2 (the shared-lut + split-accumulator path), else the parsed value
/// (0 scalar / 1 vectorised, kept for bisection). `atoi` semantics on a
/// malformed value => 0.
///
/// This matters because the batched gate/up launcher only fuses gate_up+swiglu
/// when `g_fuse && g_expert_fp4_mode == 2 && dim % 512 == 0`
/// (dsv41_experts_mxf4.cu:1268). The host must test the SAME mode before it
/// assumes the fused inter-width layout: at mode 0/1 the kernel writes the
/// full 2*inter (gate|up), so a host that still believes "fused" advances
/// act_slot by `inter` and skips the separate swiglu launch -> silent data
/// misalignment, not a perf difference. Read ONCE and cached like the other
/// gates (per-call getenv is a hot-path slip).
pub(crate) fn expert_fp4_mode() -> i32 {
    static M: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        std::env::var("DSV41_EXPERT_FP4_MODE")
            .map(|v| v.parse::<i32>().unwrap_or(0))
            .unwrap_or(2)
    })
}

/// DSV41_NR_FUSE=0 reverts the kv chain to the two-launch rmsnorm + rope pair.
fn nr_fuse() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_NR_FUSE").map(|v| v != "0").unwrap_or(true))
}

/// DUAL_CHAIN (DSV41_DUAL_CHAIN, default ON): the kv half of `attention()` (kv
/// norm + rope) is issued on the runtime's SECOND side stream so it overlaps the
/// q chain (rmsnorm_q/NORM_FUSE + wq_b + rope) that stays on the main stream.
/// The two chains touch disjoint buffers (`s.kv` vs `s.qr`/`s.q`/`s.xq`) and meet
/// only at the kv chain's first consumer, so the split is bit-identical — the
/// kernels and their operands are untouched, only the stream they are issued on.
/// "0" is the A/B arm (serial).
///
/// Two shape/runtime gates, both silent:
/// - `kv_early` (the fused `lin2` took): the unfused fallback's `wkv` lin would
///   quantise into the SHARED `s.xq`/`s.xsc`, which the q chain also writes, so
///   overlapping them there would race on the activation buffer.
/// - `Device::supports_dual_chain()`: the runtime must own the second side
///   stream and both disable-timing fork/join events (see `devrt.rs`).
fn dual_chain() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_DUAL_CHAIN").map(|v| v != "0").unwrap_or(true))
}

/// COMPRESS_SIDE (DSV41_COMPRESS_SIDE, default ON): a kv-source layer's four
/// compressor launches (`lin_f32` kvp/scp + `compressor_pool` + `compress_commit`,
/// ~30us) are issued on the runtime's THIRD side stream so they overlap the q
/// chain (~13.5us) and the kv chain (which rides `side_stream2`) that own the
/// main stream in the same window. The compressor reads `s.xn` + the position
/// counter + this layer's own state, and writes only layer-private buffers
/// (`kvp`/`scp`/`state_kv`/`state_score`/`latent`/`out_rows` + the ring's
/// COMPRESSED rows + this layer's device counter) — disjoint from both the q
/// chain's `qr`/`q`/`xq`/`xsc` and the kv chain's `s.kv` — so the split is
/// bit-identical: the kernels and their operands are untouched, only the stream
/// they are issued on. "0" is the A/B arm (serial).
///
/// Gates (each silent, falling back to the serial compressor):
/// - `cublas_m1()`: the cuBLAS-M1 path binds one handle to the MAIN stream, so
///   an f32 projection issued on the side stream would go to the wrong stream.
///   Only an A/B knob (default OFF).
/// - `Device::supports_compress_side()`: the runtime must own the third side
///   stream and both disable-timing fork/join events (see `devrt.rs`).
/// The per-layer shape gates (ratio > 0, kv source, `comp_wkv`/`comp_norm`
/// present) live at the call site, where `compress()`'s own early return has
/// the same conditions.
fn compress_side() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_COMPRESS_SIDE").map(|v| v != "0").unwrap_or(true))
}

/// COMPRESS_FUSE (DSV41_COMPRESS_FUSE, default ON): the decode compressor's
/// three 1-block launches — the state carry (`compressor_state`), the gated pool
/// + RMSNorm (`compressor_pool`) and the rope/ring commit (`compress_commit`) —
/// collapse into ONE `dsv41_compressor_fused`. They are a strict chain on this
/// layer's own private buffers (state_kv/state_score -> latent/out_rows ->
/// ring/clen), each already a single block in decode, so the fusion changes
/// neither the decomposition nor a single byte: it only removes two launch
/// boundaries and two graph nodes per kv-source layer (3 of them at ratio 2).
/// `=0` reverts to the three-launch sequence, which is bit-identical.
///
/// DECODE ONLY: the fused kernel carries the `start_pos > 0` state mapping and
/// the `mode 2` pool, so `ratio == 1` and `pos == 0` keep the old path (see the
/// shape gate in `compress_on`).
fn compress_fuse() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_COMPRESS_FUSE").map(|v| v != "0").unwrap_or(true))
}

/// MOE_DUAL (DSV41_MOE_DUAL, default ON): the MoE's SHARED expert half is issued
/// on the runtime's second side stream so it overlaps the ROUTED experts' fp4
/// chain on the main stream. The two halves of `moe()` read `xn` through
/// DISJOINT quantisations (routed: `quant_fp4` -> `s.xq4`/`s.xsc4`; shared:
/// `quant1` -> `s.xq`/`s.xsc`) and write DISJOINT buffers, so the split is
/// bit-identical — the kernels and their operands are untouched, only the
/// stream they are issued on. The shared half's w2 lands in `s.ex_out` (it may
/// NOT use `s.o`, which the routed down-reduce is writing concurrently) and the
/// join then runs the SAME `add_inplace(&s.o, &s.ex_out)` the serial non-fused
/// path already ran, in the same order.
///
/// Gated OFF (serial) when any of these holds, each a genuine data race or a
/// missing runtime capability:
/// - `MIX_GATE` (`DSV41_MIX_GATE`): its fused gate+w1+w3 launch makes the
///   shared half a consumer of the gate and writes `s.xq` on the MAIN stream,
///   which the shared half's own `quant1(xn)` would then race.
/// - the routed path is not the BATCHED one: the sequential loop reuses
///   `s.ex_act`, which the shared half writes.
/// - no shared expert on this rank (`sh_w` None), or `SKIP_EXPERTS`.
/// - `Device::supports_dual_chain()`: the runtime must own the second side
///   stream and both disable-timing fork/join events (see `devrt.rs`).
/// "0" is the A/B arm (serial).
fn moe_dual() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_MOE_DUAL").map(|v| v != "0").unwrap_or(true))
}

/// DSV41_ROPE_FUSE=0 reverts the q rope (and, under IDX_FUSE, the idx_q rope) to
/// the standalone `apply_rope` launch. DEFAULT ON. The fused epilogue performs
/// `apply_rope_kernel`'s rotation expression verbatim on the same `v = acc + bias`
/// f32 the rope kernel would have read back, so the result is bit-identical; the
/// kernel declines (and the caller falls back) on any shape it cannot take. Read
/// ONCE and cached like the other gates - a per-call getenv is the hot-path slip
/// they all avoid (this branch runs 40x/step, inside graph capture).
fn rope_fuse() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_ROPE_FUSE").map(|v| v != "0").unwrap_or(true))
}

/// DSV41_SH_EXP_MX2=0 reverts the shared expert's gate/up to two launches.
fn sh_exp_mx2() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_SH_EXP_MX2").map(|v| v != "0").unwrap_or(true))
}

/// A5: DSV41_MOE_EPI_ADD=0 reverts the shared expert's w2 to the
/// (gemm_fp8_mx, ferrite_add) pair. DEFAULT OFF (parked with the other
/// round-18-era fusion gates; the A5 epilogue itself is bit-identical, see the
/// call site's note -- DSV41_MOE_EPI_ADD=1 re-enables). Read ONCE and cached
/// like the other gates — a per-call getenv is exactly the hot-path slip they
/// avoid (this branch runs 40x/step, inside graph capture).
fn moe_epi_add() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_MOE_EPI_ADD").map(|v| v != "0").unwrap_or(false))
}

/// A4: DSV41_SWIGLU_Q=0 reverts the shared expert's swiglu to the
/// (swiglu_limit, quant1) pair. DEFAULT ON: the fused epilogue is bit-identical
/// to that pair, on all three products --
///   * the f32 write-back is the SAME register value `v` (never a global
///     re-read), so the f32 row is unchanged;
///   * the amax is an fmaxf tree over the SAME 32 values: the launcher declines
///     unless `inter % 32 == 0` and every warp owns exactly one 32-element
///     scale block (t counts blocks, `i = (b << 5) | lane`), so no cross-warp
///     reduction is needed -- this is precisely the shape quant_kernel<0>'s
///     block=32 reduce covers (see the wo_b1 lesson in STATUS.md);
///   * the scale/byte arithmetic is quant_kernel<0>'s term for term
///     (fmaxf(fast_round_scale(amax, 1/448), 1e-30) -> clamp +-448 ->
///     __nv_fp8_e4m3), one `1.0f / sc` per element exactly as the quant kernel
///     computes its `inv`.
/// Evidence: kernels/cuda/tests_dsv41_glue.cu `swiglu_q` case asserts xq / xsc /
/// the f32 row are all bit-exact against `dsv41_swiglu_limit` + the real
/// `dsv41_quant_fp8`(block=32, round_scale=1) -- the test the round-18 gating
/// never ran.
///
/// ⚠️ The "round-18 numerical bug" this gate was parked on is a MISATTRIBUTION:
/// the A4 code first appears in f3b1be1, whose own message reports the round-18
/// run, and `f3b1be1^` contains no `swiglu_limit_q` / `act_q` at all -- A4 was
/// not in the binary that produced the garbage texts. That failure's root cause
/// was the `.cu`/Rust DEFAULT SPLIT of the gateup+swiglu fusion (STATUS.md round
/// 18-21, `.cu:1362` g_fuse vs the Rust gate); A4/A5 were then swept OFF with it
/// (f6a1c08) "for the safe baseline". DSV41_SWIGLU_Q=0 still reverts.
/// Read ONCE and cached like the other gates.
fn swiglu_q() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_SWIGLU_Q").map(|v| v != "0").unwrap_or(true))
}

/// chain-pair-batch 链2 (DSV41_SH_PAIR, default OFF): the shared expert's
/// (w1w3 -> swiglu -> w2) chain as ONE grid-sync launch
/// (`dsv41_gemm_fp8_sh_pair`) instead of three. Phase 1 walks BOTH of a row's
/// weight rows (w1 = gate, w3 = up) in one warp over the same fp8 activation and
/// applies the swiglu epilogue in-register (so the separate swiglu launch
/// disappears), a sense-reversing device-wide barrier joins it to phase 2, and
/// phase 2 is the standalone w2 GEMV. Every output is bit-identical to the three
/// calls this replaces; what disappears is two launches + two graph nodes per
/// layer (80/step).
///
/// Default OFF because it is a NEW kernel whose only verified property so far is
/// the decline path (`gate.shape_ok()`); "=1" is the bring-up arm and the text
/// check is what flips it, like every other gate here.
///
/// Shape gates, all mirrored by the kernel's own specialisation:
///   * `dim % 32 == 0` and `sh_il % 32 == 0`: phase 1's k-blocks and BOTH phases'
///     32-element scale rows. `sh_il % 32 == 0` ALSO makes one block own whole
///     32-row scale blocks (the fp8 epilogue's amax is a single block tree) and
///     keeps phase 2's cp.async16 weight rows 16-byte aligned.
///   * a real w2 (the shared half needs it to run at all).
///   * no MIX_GATE (`sh_via_mixed`): the mixed launch produces w1/w3 from its own
///     `s.xq` write, so phase 1 would consume the wrong activation.
fn sh_pair() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_SH_PAIR").map(|v| v == "1").unwrap_or(false))
}

/// ADD_EPI (DSV41_ADD_EPI, default ON): fold the shared expert's merge
/// `s.o += s.ex_out` into the MoE all-reduce's STORE epilogue instead of running
/// the standalone `ferrite_add` kernel between them.
///
/// The AR's store already publishes `s.o` to every peer's staging slot and the
/// reduced sum is what `hc_post` consumes, so publishing `s.o[i] + s.ex_out[i]`
/// is exactly the pair's value: same operands, same ascending-rank reduce ⇒
/// BIT-IDENTICAL, one launch (and one graph node) shorter, and — unlike folding
/// into a producer — it needs NO reorder of the MoE chain (so no change to the
/// PDL adjacency of the expert launches) and no change to the `dual` fork.
///
/// Falls back to the standalone `add_inplace` when the loaded .so has no
/// `ferrite_p2p_ar_v5_add` / `..._hcpost_add` symbol (old build), when the
/// protocol is not AR v5, or when no shared expert ran this layer.
fn add_epi() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_ADD_EPI").map(|v| v != "0").unwrap_or(true))
}

/// DSV41_MIX_GATE=0 keeps the MoE gate and the shared expert as two launches.
/// DEFAULT OFF (round 25: 9.38 vs 9.71ms - the mixed kernel's fp8 branch,
/// even WITH the LUT+a32 port, is a net loss against the separate path
/// where the shared expert's mx2 already has the full optimization set).
fn mix_gate_shared() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_MIX_GATE").map(|v| v != "0").unwrap_or(false))
}

/// DSV41_ROUTE_FUSE=0 reverts the gate GEMV + route_topk pair to two launches.
/// DEFAULT ON: `ferrite_gemv_bf16_v2_route` runs the route in the gate GEMV's
/// last block (40 launches/step and 40 graph nodes saved, and the route is
/// bit-exact — see the kernel's epilogue note). Only the plain bf16 gate path
/// can fuse; the MIX_GATE (fp8x2) and cuBLAS-M1 gate paths keep the separate
/// route_topk, and so does an .so without the symbol (checked per call, cheap).
fn route_fuse() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_ROUTE_FUSE").map(|v| v != "0").unwrap_or(true))
}

/// DSV41_IDX_FUSE=0 reverts the wq_b + idx_wq_b pair to two launches.
///
/// L2+L3 fusion: `wq_b` and the indexer's `idx_wq_b` read the SAME `qr` buffer
/// (rmsnorm writes it in place; neither touches it in between), both have
/// k = q_lora_rank, and the mx2 contract is bit-identical to the two singles
/// (each row is still one warp in the same lane order). Only the index-source
/// layers carry `idx_wq_b`; every other layer falls through to the single
/// `lin(wq_b)`. Read ONCE and cached (the layer loop runs 40x/step, inside the
/// graph-capture region) — per-call getenv is the hot-path slip the other gates
/// already avoid.
fn idx_fuse() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_IDX_FUSE").map(|v| v != "0").unwrap_or(false))
}

/// DSV41_HEAD_SLICE enables the vocabulary-sliced lm_head: each rank projects
/// only its 1/world slice (8x less head weight traffic per step — the full
/// 129280-row head measured 298us against 48us for one rank's slice) and a
/// cross-rank argmax exchange - ONE round of the shared v5 epoch sequence -
/// picks the winner with the same lowest-index tie rule.
///
/// DEFAULT ON (verified 2026-09-11: 10.86 -> 10.51ms, all four prompts
/// verbatim-correct, zero faults). DSV41_HEAD_SLICE=0 restores the full head.
fn head_slice() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_HEAD_SLICE").map(|v| v != "0").unwrap_or(true))
}

// ---------------------------------------------------------------------------
// Diagnostic / A-B gates that live INSIDE the per-layer hot path (the layer
// loop runs 40x per step, inside the CUDA-graph capture region). A per-call
// `std::env::var` there takes the environment lock and may allocate a String —
// pure host cost that inflates graph construction and replay latency, exactly
// the hot-path slip the other gates above already avoid. Read ONCE and cache,
// same house rule as eng_host()/fuse_c().
// ---------------------------------------------------------------------------

/// DSV41_PHASE=1 prints per-layer segment timings.
fn phase_dbg() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_PHASE").map(|v| v != "0").unwrap_or(false))
}

/// DSV41_HCDBG=1 dumps L0 activation magnitudes for the hyper-connection debug.
fn hc_dbg() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_HCDBG").map(|v| v != "0").unwrap_or(false))
}

/// DSV41_STATS=1 enables the per-stage magnitude probes (and disables the step graph).
fn stats_dbg() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_STATS").map(|v| v != "0").unwrap_or(false))
}

/// DSV41_STATS_EVERY (default 5): probe only every Nth layer.
fn stats_every() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("DSV41_STATS_EVERY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5)
    })
}

/// DSV41_MOEDBG=1 dumps the router's selection per layer.
fn moe_dbg() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_MOEDBG").map(|v| v != "0").unwrap_or(false))
}

/// DSV41_RING_OWNER=1 restores the old shared-ring window behaviour for A/B.
fn ring_owner_shared() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_RING_OWNER").map(|v| v != "0").unwrap_or(false))
}

/// DSV41_AR_STORE_FUSE=1 re-enables the wo_b all-reduce store epilogue.
fn ar_store_fuse() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_AR_STORE_FUSE").map(|v| v != "0").unwrap_or(false))
}

/// DSV41_GATEUP_FUSE (default ON): gate/up fusion in the batched MoE path.
pub(crate) fn gateup_fuse() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_GATEUP_FUSE").map(|v| v != "0").unwrap_or(true))
}

/// T2: DSV41_QR_EPI=0 reverts the qr rmsnorm's fused fp8 emission (the wq_b
/// projection then quantises `qr` with its own dsv41_quant_fp8 launch, as
/// before). DEFAULT ON - the emitted pair is bit-identical to that launch by
/// construction (same absmax over the same 32-element block, same
/// fast_round_scale, same clamp + __nv_fp8_e4m3 round), and an .so without
/// `dsv41_rmsnorm_q` falls back on its own. Cached like the other gates: this
/// site is inside the 40x/step layer loop.
fn qr_epi() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_QR_EPI").map(|v| v != "0").unwrap_or(true))
}

/// NORM_FUSE (DSV41_NORM_FUSE, default ON): the wq_b (and, on the indexer side,
/// idx_wq_b) M=1 rope GEMV computes the RMSNorm + fp8 encoding of its own `qr`
/// input in its PROLOGUE, so the standalone `rmsnorm_q` launch - 40 per step,
/// 0.13 ms measured with nsys v5 - and its graph node disappear. Same
/// producer/consumer shape as the hc-tail / swiglu / B1 epilogues, but on the
/// consumer's side: the activation is produced where it is consumed, so nothing
/// is written to global memory or read back.
///
/// Bit-identical by construction: the prologue's cross-warp reduction tree, its
/// blockDim-strided element loop, the per-32-block amax shuffle and the
/// `fast_round_scale` + clamp + `__nv_fp8_e4m3` round are
/// `rmsnorm_q_kernel`'s, term for term, and the launcher forces the same
/// 1024-thread block. The gemv then reads the same bytes out of shared memory,
/// so its dot - and therefore its output - is unchanged.
///
/// `=0` reverts to the (rmsnorm_q, gemm_fp8_mx_rope) pair; an .so without
/// `dsv41_gemm_fp8_mx_rope_norm` falls back on its own. Cached like the other
/// gates: this branch runs 40x/step inside the layer loop.
///
/// ⚠️ The fused path leaves `qr` UNNORMALISED, so every later reader of `qr`
/// must use the same fused launch. `attention()` records that in `s.qr_raw`, and
/// `indexer()` sends its idx_wq_b down the fused launch when it is set. That is
/// also why the fusion is refused while `IDX_FUSE` is on: the two-family launch
/// shares one `xq` between wq_b and idx_wq_b and cannot leave `qr` raw.
fn norm_fuse() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_NORM_FUSE").map(|v| v != "0").unwrap_or(true))
}

/// B2 (DSV41_OROPE_Q, default ON): the inverse o-rope's epilogue emits the fp8
/// of the whole roped region, so the `quant1(s.o)` launch right before wo_a
/// (40/step) disappears. The emitted pair is `dsv41_quant_fp8(o)`'s, term for
/// term (same 32-element block, same fast_round_scale, same clamp + e4m3), so
/// the model output is unchanged; "0" reverts to rope + quant1. An .so without
/// `dsv41_apply_rope_q`, or a declining shape, falls back on its own.
fn orope_q() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_OROPE_Q").map(|v| v != "0").unwrap_or(true))
}

/// P1 (DSV41_SPARSE_OROPE, default ON): folds the inverse o-rope AND the fp8
/// emission of the roped attention output INTO the sparse-attention launch.
/// `sparse_attn_pf_kernel` and the o-rope call shape of `apply_rope_kernel` own
/// the same block (grid=(b*m,h), block=128, one head's full d row), so this is a
/// geometry-preserving fusion: the o row is normalised into shared memory, then
/// the rope + fp8 passes run in the same launch, emitting exactly the bytes the
/// `quant1(s.o)` launch would have. It removes TWO launches per layer (40 each
/// per step): the standalone `apply_rope_q` and the `quant1` that followed it.
///
/// "0" reverts to `sparse_attn` + `apply_rope_q` + `quant1`. An .so without
/// `dsv41_sparse_attn_orope`, or a declining shape (the plain call would not
/// have picked `sparse_attn_pf_kernel`, or hd % 32 != 0), falls back on its own
/// — the fallback is bit-identical by construction.
fn sparse_orope() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_SPARSE_OROPE").map(|v| v != "0").unwrap_or(true))
}

/// B2 (DSV41_RING_WIN_FUSE, default ON): one launch does the window ring append
/// and the window indices (two adjacent, mutually independent one-block
/// kernels), saving 40 launches/step. "0" reverts; an .so without
/// `dsv41_ring_win_fuse` falls back on its own.
fn ring_win_fuse() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_RING_WIN_FUSE").map(|v| v != "0").unwrap_or(true))
}

/// B3 (DSV41_COMP_PLACEHOLDER_FUSE, default ON): the recency-placeholder launch
/// (30/step, ~1.0us each) writes `idxs[win, win+take)` into the SAME buffer the
/// `ring_win_fuse` epilogue already writes `idxs[0, win)` into, and `sparse_attn`
/// is the only reader of both. Nothing writes between the two points except the
/// indexer (which owns [win, ..) itself and therefore takes the fused launch
/// WITHOUT the placeholder half), so the placeholder folds into the ring_win
/// launch and its launch + graph node disappear. The bound stays derived from the
/// DEVICE counter, so a captured graph replays correctly. "0" reverts; an .so
/// without `dsv41_ring_win_fuse_ph` falls back on its own.
fn comp_ph_fuse() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("DSV41_COMP_PLACEHOLDER_FUSE")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// DSV41_CUBLAS_M1=1 routes the M=1 f32/bf16 linears through cuBLAS.
fn cublas_m1() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_CUBLAS_M1").map(|v| v != "0").unwrap_or(false))
}

fn build_eng_dev(
    dev: &Device,
    lay: &crate::dsv41::engram::EngramLayout,
    map: &crate::dsv41::engram::TokenMap,
    max_seq: usize,
) -> Result<EngDev> {
    let n_cols = lay.n_hash_cols();
    let nl = lay.layers.len();
    let d_map = dev.alloc(map.map.len() * 8)?;
    dev.upload_bytes_at(
        &d_map,
        unsafe { std::slice::from_raw_parts(map.map.as_ptr() as *const u8, map.map.len() * 8) },
    )?;
    let mut mults: Vec<i64> = Vec::with_capacity(nl * 4);
    for li in 0..nl {
        mults.extend_from_slice(lay.multipliers(li));
    }
    let mut lms: Vec<u64> = Vec::with_capacity(nl * n_cols);
    let mut offs: Vec<u64> = Vec::with_capacity(nl * n_cols);
    for li in 0..nl {
        for c in 0..n_cols {
            let (lm, off) = lay.layers[li].column(c);
            lms.push(lm);
            offs.push(off);
        }
    }
    let up = |v: &[u64]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>() };
    let d_mults = dev.alloc(mults.len() * 8)?;
    dev.upload_bytes_at(
        &d_mults,
        unsafe { std::slice::from_raw_parts(mults.as_ptr() as *const u8, mults.len() * 8) },
    )?;
    let d_lms = dev.alloc(lms.len() * 8)?;
    dev.upload_bytes_at(&d_lms, &up(&lms))?;
    let d_offs = dev.alloc(offs.len() * 8)?;
    dev.upload_bytes_at(&d_offs, &up(&offs))?;
    let d_cache = dev.alloc(max_seq * 8)?;
    dev.zero_at(d_cache.ptr, max_seq * 8)?;
    Ok(EngDev {
        map: d_map,
        cache: d_cache,
        mults: d_mults,
        lms: d_lms,
        offs: d_offs,
        max_seq,
    })
}

pub struct DevChain<'a> {
    pub dev: &'a Device,
    pub cfg: &'a Dsv41Config,
    pub w: &'a Dsv41DevWeights,
    pub opts: RunOpts,
    /// Tensor-parallel collective. `None` runs the model on one device; when
    /// present, the row-parallel sites reduce across ranks.
    pub comm: Option<Arc<Collective>>,
    layers: Vec<LayerCache>,
    s: Scratch,
    cos: DevBuf,
    sin: DevBuf,
    /// the compressor's KV uses a DIFFERENT rope theta (160000 vs 10000);
    /// without separate tables every rope call uses the main theta
    cos_comp: DevBuf,
    sin_comp: DevBuf,
    // ---- engram ----
    /// Per-layer MoE-segment graphs (only with DSV41_GRAPH_MOE=1). The segment is
    /// everything up to the all-reduce; the AR stays host-issued.
    moe_graph: Vec<Option<*mut std::ffi::c_void>>,
    /// Armed from the second step on, once every kernel is warm.
    moe_graph_armed: bool,
    /// How many steps this chain has run (the first one warms the kernels).
    step_count: u32,
    /// Diagnostics for the MoE segment graphs.
    moe_graph_captures: u32,
    moe_graph_replays: u32,
    /// n-gram hash state (host side; the token cache spans prefill + decode)
    /// The whole-step CUDA graph (captured on the first DECODE step; see step_impl).
    step_graph: Option<*mut std::ffi::c_void>,
    /// Decode-path steps only: the capture must NOT happen during prefill, because
    /// the host's launch decisions (which branches, which kernel args) can differ
    /// between prefill and decode and a capture freezes them.
    decode_steps: u32,
    /// Device-side engram hash state (built lazily on the first step).
    eng_dev: Option<EngDev>,
    ngram: Option<crate::dsv41::engram::NgramHashState>,
    eng_layout: Option<crate::dsv41::engram::EngramLayout>,
    eng_map: Option<crate::dsv41::engram::TokenMap>,
    /// ADD_EPI (see [`DevChain::add_epi`]): `moe()` defers the shared-expert
    /// merge `s.o += s.ex_out` into the following MoE all-reduce's store
    /// epilogue and records the bias here, indexed by layer; `moe_reduce(layer)`
    /// READS it. `None` means the standalone `add_inplace` already ran (or no
    /// shared expert on this rank).
    ///
    /// Per-layer and never consumed on purpose: the whole step is captured into
    /// ONE CUDA graph, so every host decision here — including this field — is
    /// evaluated once at capture and the AR's arguments are baked into the graph.
    /// A `take()` would drop the bias for every replay, and a single shared field
    /// would let one layer's capture clobber another's under a per-layer MoE
    /// graph (`DSV41_GRAPH_MOE`).
    moe_add_in: Vec<Option<*const f32>>,
}

fn fb(n: usize) -> usize {
    n * 4
}

impl<'a> DevChain<'a> {
    pub fn new(
        dev: &'a Device,
        cfg: &'a Dsv41Config,
        w: &'a Dsv41DevWeights,
        opts: RunOpts,
        map: Option<crate::dsv41::engram::TokenMap>,
    ) -> Result<Self> {
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let hd = cfg.head_dim;
        let nh = cfg.n_heads;
        let inter = cfg.moe_inter_dim;
        let n_exp = cfg.n_routed_experts.max(1);
        let topk = cfg.n_activated_experts.max(1);
        let ql = cfg.q_lora_rank;
        // engram sizing: (max_ngram_size - 1) * n_heads hash columns, one row
        // of `engram_head_dim` each, for the layers the config lists (1 and 14).
        let n_eng_layers = cfg.engram_layer_ids.len().max(1);
        let eng_cols =
            cfg.engram_max_ngram_size.saturating_sub(1).max(1) * cfg.engram_n_heads.max(1);
        let ehd = cfg.engram_head_dim.max(1);
        let bf16_cap = dim.max(ql).max(nh * hd).max(cfg.vocab_size);
        let bf16 = dev.alloc(bf16_cap * 2)?;
        // swapAB last-block-reduction scratch bound: the largest `n` ANY
        // `gemm_fp8_mx_or_swap` call site can pass. That is exactly the set `xq`
        // is sized from (attention out = nh*hd, the wo_a output, `dim`, `inter`),
        // plus the engram kv row `(hc+1)*dim` and the indexer's q. The partial
        // buffer is `8 * swapab_n` floats (8 = the compile-time max
        // `DSV41_SWAPAB_KSPLIT`), the ticket array one u32 per 16-row tile.
        let swapab_n = (hc + 1)
            .saturating_mul(dim)
            .max(nh * hd)
            .max(cfg.index_n_heads.max(1) * cfg.index_head_dim.max(1))
            .max(cfg.n_groups_o_lora())
            .max(inter)
            .max(dim)
            .max(16);

        // The model's max_position_embeddings (1M) sizes NOTHING at runtime:
        // index_k alone would be 256-512 MiB per layer (~16.5 GiB over 43
        // layers), and on a 4 GB host the driver's per-allocation bookkeeping
        // for that many mappings is what actually dies (a 256 MiB cudaMalloc
        // 'fails' with 182 GB free). Caches are sized for the positions this
        // run can actually reach — DSV41_MAX_POS, default 64k.
        let max_pos = cfg
            .max_seq_len
            .min(
                std::env::var("DSV41_MAX_POS")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(65536),
            );
        let mut layers = Vec::with_capacity(cfg.n_layers + cfg.n_mtp_layers);
        for l in 0..cfg.n_layers + cfg.n_mtp_layers {
            let ratio = cfg.compress_ratio(l).max(1);
            let max_comp = max_pos / ratio + 2;
            // No indexer bound constant lives here any more: the kernel walks the
            // compressed rows in fixed-size chunks, so its shared memory depends on
            // the chunk size and index_topk only - never on this per-step count. That
            // removes the whole "`*lens` past the cap silently loses candidates"
            // envelope the constant carried.
            // The KV buffer holds the window ring FOLLOWED by the compressed
            // latents: rows [0, window) are the ring, [window, window+max_comp)
            // are the compressor's output. The earlier allocation was window
            // rows only, so the compressor's memcpy_d2d wrote past the end —
            // silent corruption at 3 layers (adjacent allocation masked it),
            // SIGSEGV in the driver at 24+.
            layers.push(LayerCache {
                ring: dev.alloc(fb((cfg.window_size + max_comp) * hd))?,
                idxs: dev.alloc(fb(cfg.window_size + cfg.index_topk + 8).max(4))?,
                state_kv: dev.alloc(fb(ratio * hd))?,
                state_score: dev.alloc(fb(ratio * hd))?,
                kvp: dev.alloc(fb(hd))?,
                scp: dev.alloc(fb(hd))?,
                latent: dev.alloc(fb(hd))?,
                out_rows: dev.alloc(4)?,
                compress_len: 0,
                index_k: dev.alloc(fb(max_comp * cfg.index_head_dim.max(1)))?,
            });
        }

        let s = Scratch {
            h: dev.alloc(fb(hc * dim))?,
            h2: dev.alloc(fb(hc * dim))?,
            x: dev.alloc(fb(dim))?,
            xn: dev.alloc(fb(dim))?,
            // sized for the LARGEST activation quantised anywhere in the chain
            // (the attention output has n_heads*head_dim elements, well past
            // `dim` — sizing these by `dim` overflowed on the output projection)
            xq: dev.alloc(dim.max(nh * hd).max(cfg.o_lora_rank).max(inter))?,
            xsc: dev.alloc(fb(dim.max(nh * hd).max(cfg.o_lora_rank).max(inter) / 32 + 8))?,
            // fp4 packs 2 values per byte, so one row needs dim/2 bytes at most
            xq4: dev.alloc(dim.max(8))?,
            xsc4: dev.alloc(fb(dim / 32 + 8))?,
            xq_of_xn_valid: std::cell::Cell::new(false),
            xq_of_qr_valid: std::cell::Cell::new(false),
            qr_raw: std::cell::Cell::new(false),
            idx_q_ready: std::cell::Cell::new(false),
            idx_q_rope: std::cell::Cell::new(false),
            pre: dev.alloc(fb(hc))?,
            post: dev.alloc(fb(hc))?,
            comb: dev.alloc(fb(hc * hc))?,
            qr: dev.alloc(fb(ql))?,
            q: dev.alloc(fb(nh * hd))?,
            kv: dev.alloc(fb(hd))?,
            o: dev.alloc(fb(nh * hd))?,
            wo: dev.alloc(fb(cfg.n_groups_o_lora()))?,
            logits: dev.alloc(fb(cfg.vocab_size))?,
            ids: dev.alloc(4)?,
            pos_ctr: dev.alloc(4)?,
            argmax_packed: dev.alloc(8)?,
            clen: dev.alloc(cfg.n_layers * 4)?,
            scores: dev.alloc(fb(n_exp))?,
            route_idx: dev.alloc(fb(topk).max(4))?,
            route_w: dev.alloc(fb(topk).max(4))?,
            route_ctr: dev.alloc(4)?,
            ex_in: dev.alloc(fb(dim))?,
            ex_act: dev.alloc(fb(2 * inter.max(dim)))?,
            ex_out: dev.alloc(fb(dim))?,
            // DSV41_MOE_BATCH scratch (allocated unconditionally: it is a few
            // hundred KB and keeps the allocation graph static). Sized by the
            // FULL `inter`, which is >= the padded local width the kernels use.
            ex_act_b: dev.alloc(fb(topk.max(1) * 2 * inter))?,
            ex_down_b: dev.alloc(fb(topk.max(1) * dim))?,
            hist: dev.alloc(fb(n_exp))?,
            idx_q: dev.alloc(fb(cfg.index_n_heads.max(1) * cfg.index_head_dim.max(1)))?,
            idx_k: dev.alloc(fb(cfg.index_head_dim.max(1)))?,
            idx_w: dev.alloc(fb(cfg.index_n_heads.max(1)))?,
            bf16,
            pre_a: dev.alloc(fb(hc).max(8))?,
            pre_b: dev.alloc(fb(hc).max(8))?,
            pre_c: dev.alloc(fb(hc).max(8))?,
            premix_const: dev.alloc(fb(hc).max(8))?,
            dspark_tap: dev.alloc(fb(DSPARK_TAP_SLOTS * dim).max(8))?,
            dspark_pre_mean: dev.alloc(fb(hc).max(8))?,
            eng_ids: dev.alloc(fb(eng_cols * n_eng_layers).max(8) * 2)?, // i64
            eng_rows: dev.alloc(fb(eng_cols * ehd).max(8))?,
            eng_kv: dev.alloc(fb((hc + 1) * dim))?,
            eng_xq: dev.alloc((eng_cols * ehd).max(8))?, // fp8 bytes
            eng_xsc: dev.alloc(fb((eng_cols * ehd).max(8) / 32 + 8))?,
            // B1: sized by the FULL (unsharded) wo_a output, which is also the
            // largest `ol_local` any rank produces.
            wo_q: dev.alloc(cfg.n_groups_o_lora().max(8))?,
            wo_qsc: dev.alloc(fb(cfg.n_groups_o_lora() / 32 + 8))?,
            // chain-pair-grid-sync: the [arrive, sense] barrier pair. 8 bytes; the
            // kernel self-resets it, so it only has to START at zero (see below).
            wo_bar: dev.alloc(8)?,
            // chain-pair-batch 链2: the fused shared-expert chain's swiglu fp8
            // output. Sized by the FULL `inter` (>= every rank's local slice);
            // phase 1 writes it before phase 2 ever reads it, so no zero-fill.
            sh_q: dev.alloc(inter.max(8))?,
            sh_qsc: dev.alloc(fb(inter.max(8) / 32 + 8))?,
            // swapAB K-split scratch: sized by the largest `n` any swapAB call
            // site passes (the same list `xq` is sized from, plus the engram kv
            // row and the indexer's q), times the max K split 8. The ticket array
            // needs one u32 per 16-row tile. A call whose `n` exceeds `swapab_n`
            // is not routed here (see `gemm_fp8_mx_or_swap`), so the bound is
            // enforced rather than assumed.
            swapab_n,
            swapab_part: dev.alloc(fb(8 * swapab_n))?,
            swapab_ctr: dev.alloc(4 * (swapab_n / 16 + 1))?,
            // ---- DSpark verify (`step_rows`): the m-row twins. Allocated
            // unconditionally — the whole set is a few MB (the only large member is
            // `logits_r`, one per-row head row) and a static allocation graph is
            // worth more than the bytes.
            h_r: dev.alloc(fb(VERIFY_ROWS * hc * dim))?,
            h2_r: dev.alloc(fb(VERIFY_ROWS * hc * dim))?,
            x_r: dev.alloc(fb(VERIFY_ROWS * dim))?,
            xn_r: dev.alloc(fb(VERIFY_ROWS * dim))?,
            q_r: dev.alloc(fb(VERIFY_ROWS * nh * hd))?,
            kv_r: dev.alloc(fb(VERIFY_ROWS * hd))?,
            qr_r: dev.alloc(fb(VERIFY_ROWS * ql))?,
            o_r: dev.alloc(fb(VERIFY_ROWS * nh * hd))?,
            wo_r: dev.alloc(fb(VERIFY_ROWS * cfg.n_groups_o_lora()))?,
            wo_out_r: dev.alloc(fb(VERIFY_ROWS * dim))?,
            moe_out_r: dev.alloc(fb(VERIFY_ROWS * dim))?,
            ids_r: dev.alloc(VERIFY_ROWS * 4)?,
            argmax_r: dev.alloc(VERIFY_ROWS * 4)?,
            logits_r: dev.alloc(fb(VERIFY_ROWS * cfg.vocab_size))?,
            premix_r: dev.alloc(fb(VERIFY_ROWS * hc).max(8))?,
            pre_r: dev.alloc(fb(VERIFY_ROWS * hc).max(8))?,
            pre2_r: dev.alloc(fb(VERIFY_ROWS * hc).max(8))?,
            post_r: dev.alloc(fb(VERIFY_ROWS * hc).max(8))?,
            comb_r: dev.alloc(fb(VERIFY_ROWS * hc * hc).max(8))?,
            pos_rows: dev.alloc(VERIFY_ROWS * 4)?,
            idxs_r: dev.alloc(4 * VERIFY_ROWS * (cfg.window_size + cfg.index_topk + 8).max(4))?,
            idxs_win_r: dev.alloc(4 * VERIFY_ROWS * cfg.window_size.max(4))?,
            idx_q_r: dev
                .alloc(fb(VERIFY_ROWS * cfg.index_n_heads.max(1) * cfg.index_head_dim.max(1)))?,
            idx_w_r: dev.alloc(fb(VERIFY_ROWS * cfg.index_n_heads.max(1)))?,
            kvp_r: dev.alloc(fb(VERIFY_ROWS * hd))?,
            scp_r: dev.alloc(fb(VERIFY_ROWS * hd))?,
            scores_r: dev.alloc(fb(VERIFY_ROWS * n_exp))?,
            route_idx_r: dev.alloc(4 * VERIFY_ROWS * topk)?,
            route_w_r: dev.alloc(fb(VERIFY_ROWS * topk))?,
            xq4_r: dev.alloc((VERIFY_ROWS * dim / 2).max(8))?,
            xsc4_r: dev.alloc(fb(VERIFY_ROWS * dim / 32 + 8))?,
            // `2 * inter` per slot: `inter` (the full, unsharded width) is >= every
            // rank's padded local slice, so one allocation covers tp=1 and tp=N.
            ex_act_r: dev.alloc(fb(VERIFY_ROWS * topk * 2 * inter.max(dim)))?,
            ex_down_r: dev.alloc(fb(VERIFY_ROWS * topk * dim))?,
            sh_act_r: dev.alloc(fb(VERIFY_ROWS * 2 * inter.max(dim)))?,
            eng_ids_r: dev.alloc(fb(eng_cols * n_eng_layers * VERIFY_ROWS).max(8) * 2)?, // i64
            eng_rows_r: dev.alloc(fb(VERIFY_ROWS * eng_cols * ehd).max(8))?,
            eng_kv_r: dev.alloc(fb(VERIFY_ROWS * (hc + 1) * dim))?,
            eng_xq_r: dev.alloc((VERIFY_ROWS * eng_cols * ehd).max(8))?, // fp8 bytes
            eng_xsc_r: dev.alloc(fb((VERIFY_ROWS * eng_cols * ehd).max(8) / 32 + 8))?,
            // ---- DSpark shadow mode: the verify write-set save. Allocated
            // unconditionally (a few hundred KB) so the allocation graph stays
            // static, exactly like the m-row set above.
            dspark_snap_ring: dev.alloc(fb(cfg.n_layers * VERIFY_ROWS * hd).max(8))?,
            dspark_snap_state: dev.alloc(
                fb(cfg.n_layers * 2 * cfg.compress_ratios.iter().copied().max().unwrap_or(1).max(1)
                    * hd)
                .max(8),
            )?,
            dspark_snap_latent: dev.alloc(fb(cfg.n_layers * hd).max(8))?,
            dspark_snap_clen: dev.alloc((cfg.n_layers * 4).max(4))?,
            dspark_snap_out_rows: dev.alloc((cfg.n_layers * 4).max(4))?,
        };

        // The fused route's election counter must start at 0 (cudaMalloc does
        // not zero). From here on the kernel self-resets it every call.
        dev.zero(&s.route_ctr)?;
        // chain-pair-grid-sync: the wo_a -> wo_b barrier pair must also start at
        // zero (arrive == 0, sense == 0). It is self-resetting after that, so this
        // is the ONLY initialisation it ever needs -- do not re-zero it per step
        // (that would race the kernel's own reset).
        dev.zero(&s.wo_bar)?;
        // swapAB last-block reduction: the per-tile tickets must start at zero
        // (the kernel resets each entry after the elected block consumes it, so a
        // captured graph replays clean). Like `wo_bar`, this is a ONE-TIME
        // initialisation -- re-zeroing per step would race the kernel's reset.
        dev.zero(&s.swapab_ctr)?;

        // RoPE tables covering the whole context.
        let table = max_pos;
        let half = cfg.rope_head_dim / 2;
        let cos = dev.alloc(fb(table * half))?;
        let sin = dev.alloc(fb(table * half))?;
        let cos_comp = dev.alloc(fb(table * half))?;
        let sin_comp = dev.alloc(fb(table * half))?;
        // main rope (theta=10000) for the query and window KV
        dev.rope_precompute(
            cos.ptr as *mut f32, sin.ptr as *mut f32,
            cfg.rope_head_dim as i32, table as i32,
            cfg.original_seq_len as i32, cfg.rope_theta,
            cfg.rope_factor, cfg.beta_fast, cfg.beta_slow,
        )?;
        // compressor rope (theta=160000) for the compressed latent
        dev.rope_precompute(
            cos_comp.ptr as *mut f32, sin_comp.ptr as *mut f32,
            cfg.rope_head_dim as i32, table as i32,
            cfg.original_seq_len as i32, cfg.compress_rope_theta,
            cfg.rope_factor, cfg.beta_fast, cfg.beta_slow,
        )?;
        let _ = bf16_cap;

        // engram host-side state: the hash needs the compressed token map (a pure
        // function of the tokenizer, precomputed) and keeps a token cache that
        // spans prefill + decode.
        let eng_layout = crate::dsv41::engram::EngramLayout::from_config(cfg);
        let (ngram, eng_layout, eng_map) = match (eng_layout, map) {
            (Some(lay), Some(m)) => {
                let st = crate::dsv41::engram::NgramHashState::new(cfg, &m);
                (Some(st), Some(lay), Some(m))
            }
            _ => (None, None, None),
        };

        Ok(DevChain {
            dev,
            cfg,
            w,
            opts,
            comm: None,
            layers,
            s,
            cos,
            sin,
            cos_comp,
            sin_comp,
            moe_graph: vec![None; cfg.n_layers],
            moe_graph_armed: false,
            step_count: 0,
            moe_graph_captures: 0,
            moe_graph_replays: 0,
            step_graph: None,
            decode_steps: 0,
            eng_dev: None,
            ngram,
            eng_layout,
            eng_map,
            moe_add_in: vec![None; cfg.n_layers],
        })
    }

    /// The precomputed RoPE tables, for a DsparkDev built alongside this chain
    /// (`set_rope_tables`): the draft's positions are a subset of the main
    /// chain's, so it reuses the same cos/sin instead of building its own.
    pub fn rope_tables(&self) -> (*const f32, *const f32) {
        (self.cos.as_f32(), self.sin.as_f32())
    }

    pub fn reset(&mut self) -> Result<()> {
        // the incoming premix is the CONSTANT [1,0,0,0] every step; upload it once
        // here so the per-step refresh is a device-to-device copy (no H2D)
        {
            let mut pm = vec![0f32; self.cfg.hc_mult];
            pm[0] = 1.0;
            self.dev.upload_f32_at(self.s.premix_const.ptr, 0, &pm)?;
        }
        // the dspark tap's mean weights: [1/hc; hc] — hc_collapse with these is
        // exactly the reference's `h.mean(dim=2)` over the hc copies
        {
            let mean_w = vec![1.0 / self.cfg.hc_mult as f32; self.cfg.hc_mult];
            self.dev.upload_f32_at(self.s.dspark_pre_mean.ptr, 0, &mean_w)?;
        }
        self.dev.zero_at(self.s.pos_ctr.ptr, 4)?;
        self.dev.zero_at(self.s.clen.ptr, self.cfg.n_layers * 4)?;
        // The capture must be re-armed PER REQUEST, and the captured graph must be
        // DROPPED per request. Two separate reasons, both measured:
        //  1. decode_steps carried over let the next request's PREFILL satisfy
        //     `decode_steps >= 1` and run through the decode graph (captured with
        //     decode-time host branch choices - the compressor's mode 2 instead of
        //     mode 1 at pos 0), which garbled the output.
        //  2. A graph bakes the DEVICE ADDRESSES of the buffers it recorded. That
        //     was safe while allocations came from the pool, whose whole contract
        //     is that a given size class returns the same address; the shared
        //     devrt allocator is byte-level and does not promise that, so reusing a
        //     graph from a previous request replays against addresses that may
        //     belong to something else now (measured: requests 1-3 fine, then
        //     one-step/empty outputs and faults). Dropping it costs a re-capture
        //     per request (a few ms) and makes the graph always match the state it
        //     was recorded from.
        if let Some(e) = self.step_graph.take() {
            // e is the graph EXEC (graph_instantiate's result); the captured
            // graph handle itself was already released right after instantiate.
            // graph_free destroys the exec when the second argument is non-null.
            self.dev.graph_free(std::ptr::null_mut(), e)?;
        }
        self.decode_steps = 0;
        if let Some(e) = self.eng_dev.as_ref() {
            self.dev.zero_at(e.cache.ptr, e.max_seq * 8)?;
        }
        for c in self.layers.iter_mut() {
            self.dev.zero(&c.ring)?;
            self.dev.zero(&c.state_kv)?;
            self.dev.zero(&c.state_score)?;
            c.compress_len = 0;
            // score_state is -inf except where filled; zeroing it would make an
            // empty slot look like a real (0-weight) entry, so seed it with -inf
            let neg = f32::NEG_INFINITY;
            let n = c.state_score.bytes / 4;
            let v = vec![neg; n];
            self.dev.upload_f32_at(c.state_score.ptr, 0, &v)?;
        }
        Ok(())
    }

    /// Quantise `src` (one row of `k` floats) to fp8 into `s.xq`/`s.xsc`.
    fn quant1(&self, src: *const f32, k: i32) -> Result<()> {
        self.quant1_on(src, k, self.dev.stream())
    }

    /// [`Self::quant1`] with an explicit stream. The MoE dual chain's shared half
    /// (`DSV41_MOE_DUAL`) quantises `xn` on the side stream, so its `s.xq`/`s.xsc`
    /// write overlaps the routed experts' fp4 chain. The T1/T2 pointer gating and
    /// its consume-once semantics are unchanged — there is exactly one quant1 of
    /// `xn` in `moe()`, and the flag's host-side order is the same either way.
    fn quant1_on(&self, src: *const f32, k: i32, stream: CuStream) -> Result<()> {
        // T1: when the hc tail just emitted the fp8 of `xn` alongside its f32
        // write-back, the next quant1 over `xn` is redundant. Pointer-gated so
        // every other source (qr, o, ex_act, engram rows) still quantises, and
        // the flag clears on the first hit - a later quant1(xn) (e.g. the
        // kv-early fallback path) re-quantises identical bytes, harmlessly.
        if src == self.s.xn.ptr as *const f32 && self.s.xq_of_xn_valid.get() {
            self.s.xq_of_xn_valid.set(false);
            return Ok(());
        }
        // T2: the qr rmsnorm emitted the fp8 of its own normalised output
        // through ferrite_rmsnorm_q, so the wq_b projection's quant1(qr) (and,
        // under IDX_FUSE, idx_wq_b's, which shares that one quantisation) is
        // redundant. Consume-once: s.xq is rewritten by the o/wo quantisations
        // before the indexer's separate idx_wq_b lin, so only the first qr
        // consumer may skip.
        if src == self.s.qr.ptr as *const f32 && self.s.xq_of_qr_valid.get() {
            self.s.xq_of_qr_valid.set(false);
            return Ok(());
        }
        self.dev.quant_fp8_on(
            src,
            self.s.xq.ptr as *mut u8,
            self.s.xsc.ptr as *mut f32,
            1,
            k,
            32,
            true,
            stream,
        )
    }

    /// fp8 dense linear for one row: `out[1, n_out] = a[1, k] @ w[n_out, k]^T`.
    fn lin(&self, a: *const f32, k: i32, w: &crate::dsv41::load::DevTensor, ws: &crate::dsv41::load::DevTensor, n_out: i32, out: *mut f32) -> Result<()> {
        self.quant1(a, k)?;
        self.gemm_fp8_mx_or_swap(
            self.s.xq.as_u8(),
            self.s.xsc.as_f32(),
            w.as_u8(),
            ws.as_u8(),
            std::ptr::null(),
            out,
            n_out,
            k,
        )
    }

    /// swapAB (DSV41_SWAPAB, default OFF). Routes the plain M=1 fp8 GEMV through
    /// the tensor-core `dsv41_gemm_fp8_swapab` kernel (weight on the MMA's M,
    /// activation on column 0 of B) instead of the SIMT `gemm_fp8_gemv_kernel`.
    /// NOT bit-identical to the SIMT gemv -- the tensor core sums raw fp8
    /// products and scales per k block, while the SIMT kernel does per-element
    /// `(a*sa)*(w*sb)` and reduces by shuffles -- so it is opt-in until text
    /// parity is confirmed on the machine. Read ONCE and cached.
    fn swapab() -> bool {
        static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *F.get_or_init(|| std::env::var("DSV41_SWAPAB").map(|v| v == "1").unwrap_or(false))
    }

    /// The plain M=1 fp8 GEMV with the swapAB dispatch in front of it: when the
    /// gate is on AND the loaded .so carries the kernel AND the shape can take it
    /// (n % 16 == 0, k % 32 == 0), the tensor-core form runs; otherwise the SIMT
    /// `gemm_fp8_mx` runs exactly as before. The decline is per-call and free, so
    /// no shape bookkeeping is needed on this side.
    ///
    /// The `n <= swapab_n` guard is NOT part of the kernel's own shape test: it is
    /// this side's promise that `swapab_part`/`swapab_ctr` are big enough for the
    /// K-split reduction. A larger `n` (a config the allocation did not size for)
    /// takes the SIMT path instead of overrunning the scratch.
    #[allow(clippy::too_many_arguments)]
    fn gemm_fp8_mx_or_swap(
        &self,
        a: *const u8,
        a_scale: *const f32,
        w: *const u8,
        w_scale: *const u8,
        bias: *const f32,
        out: *mut f32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        if Self::swapab() && n > 0 && n as usize <= self.s.swapab_n {
            let ok = self.dev.gemm_fp8_swapab(
                a,
                a_scale,
                w,
                w_scale,
                bias,
                out,
                n,
                k,
                self.s.swapab_part.ptr as *mut f32,
                self.s.swapab_ctr.ptr as *mut u32,
            )?;
            if ok {
                return Ok(());
            }
        }
        self.dev
            .gemm_fp8_mx(a, a_scale, w, w_scale, bias, out, 1, n, k)
    }

    /// Two projections over the SAME activation in one launch (DSV41_PROJ_FUSE).
    /// Reads ONCE and cached: this runs per attention call.
    fn proj_fuse() -> bool {
        // Default on: 15.44 against 15.99 ms in one session, same binary, with
        // the four prompts character-for-character identical. "0" still opts out.
        static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *F.get_or_init(|| {
            std::env::var("DSV41_PROJ_FUSE").map(|v| v != "0").unwrap_or(true)
        })
    }

    /// B1 (DSV41_WO_QUANT_FUSE, default OFF since round 35: the fused form's
/// 32-warp/32-block launch shape loses 148->32 active SMs and measured +0.24ms
/// in serve, outweighing the 40 saved quant1 launches; =1 re-enables).
    /// the fp8 row compression of its own output, so the `quant1(s.wo)` launch
    /// between the two projections disappears. The emitted (byte, scale) pair is
    /// `quant_kernel<0>`'s, term for term, so the model output is unchanged; "0"
    /// keeps the old (gemv, quant1) pair.
    fn wo_quant_fuse() -> bool {
        static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *F.get_or_init(|| {
            std::env::var("DSV41_WO_QUANT_FUSE").map(|v| v == "1").unwrap_or(false)
        })
    }

    /// wo_b f32-activation direct read (DSV41_WOB_F32, default ON). The wo_b GEMV
    /// takes the RAW f32 `s.wo` (via `dsv41_gemm_fp8_mx_f32`) instead of the fp8
    /// `quant1(s.wo)` pair, removing that launch (40/step) and the activation's
    /// per-block LUT decode + scale. Unlike B1 it keeps the NORMAL gemv grid shape
    /// (`g_gemv_warps` / ceil(n/warps)), so it does not pay B1's 32-warp
    /// SM-utilisation loss — only the data path changes. NOT bit-identical: it
    /// skips the fp8 round trip and is strictly more accurate (wo_b -> AR sum ->
    /// hc_post tolerates the tighter value). "0" restores the (quant1, gemm_fp8_mx)
    /// pair; a stale `.so` (no symbol) falls back the same way.
    fn wob_f32() -> bool {
        static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *F.get_or_init(|| {
            std::env::var("DSV41_WOB_F32").map(|v| v != "0").unwrap_or(true)
        })
    }

    /// chain-pair-grid-sync (DSV41_WO_PAIR, default OFF): the wo_a -> wo_b pair as
    /// ONE grid-sync launch (`dsv41_gemm_fp8_wo_pair`) instead of two, with a
    /// sense-reversing device-wide barrier between the phases. Each row is still
    /// one warp in the same lane order, so both outputs are bit-identical to the
    /// two launches this replaces; what disappears is one launch + one graph node
    /// per layer (40/step).
    ///
    /// Default OFF because it is a NEW kernel whose only verified property so far
    /// is the decline/fallback path: "=1" is the bring-up arm and the text check
    /// (`cargo test` + a serve A/B) is what flips it, like every other gate here.
    /// The fused path ALSO needs wob_f32 ON (the pair kernel's phase 2 IS the f32
    /// activation form), one wo_a group on this rank, and no B1/AR-store fusion —
    /// see the call site.
    fn wo_pair_fuse() -> bool {
        static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *F.get_or_init(|| std::env::var("DSV41_WO_PAIR").map(|v| v == "1").unwrap_or(false))
    }

    /// Two projections over the same activation: quantise once, then one gemv
    /// whose rows below n1 map to the first family and the rest to the second.
    /// Each row is still one warp in the same lane order, so both outputs are
    /// bit-identical to two `lin` calls. Ok(false) means the fused kernel
    /// declined the shape and the caller runs the two lins.
    #[allow(clippy::too_many_arguments)]
    fn lin2(
        &self,
        a: *const f32,
        k: i32,
        w1: &crate::dsv41::load::DevTensor,
        ws1: &crate::dsv41::load::DevTensor,
        n1: i32,
        out1: *mut f32,
        w2: &crate::dsv41::load::DevTensor,
        ws2: &crate::dsv41::load::DevTensor,
        n2: i32,
        out2: *mut f32,
    ) -> Result<bool> {
        if !Self::proj_fuse() {
            return Ok(false);
        }
        self.quant1(a, k)?;
        self.dev.gemm_fp8_mx2(
            self.s.xq.as_u8(),
            self.s.xsc.as_f32(),
            w1.as_u8(),
            ws1.as_u8(),
            std::ptr::null(),
            out1,
            n1,
            w2.as_u8(),
            ws2.as_u8(),
            std::ptr::null(),
            out2,
            n2,
            k,
        )
    }

    /// RoPE fusion (DSV41_ROPE_FUSE, default on): quantise once, run the
    /// single-family GEMV, and let its epilogue rotate the trailing `rope_rd`
    /// lanes of each `rope_hd`-wide head - the standalone `apply_rope` launch
    /// disappears. Ok(false) means the .so lacks the symbol or the shape cannot
    /// take it, so the caller runs `lin` + the standalone rope (bit-identical
    /// either way). The rotation uses the device position counter with mul=1,
    /// off=0, step=0, which is exactly the q / idx_q call shape.
    #[allow(clippy::too_many_arguments)]
    fn lin_rope(
        &self,
        a: *const f32,
        k: i32,
        w: &crate::dsv41::load::DevTensor,
        ws: &crate::dsv41::load::DevTensor,
        n_out: i32,
        out: *mut f32,
        rope_rd: i32,
        rope_hd: i32,
    ) -> Result<bool> {
        if !rope_fuse() || !self.dev.supports_rope_fuse() {
            return Ok(false);
        }
        self.quant1(a, k)?;
        self.dev.gemm_fp8_mx_rope(
            self.s.xq.as_u8(),
            self.s.xsc.as_f32(),
            w.as_u8(),
            ws.as_u8(),
            std::ptr::null(),
            out,
            n_out,
            k,
            self.cos.as_f32(),
            self.sin.as_f32(),
            self.s.pos_ctr.ptr as *const std::os::raw::c_int,
            1,
            0,
            0,
            false,
            rope_rd,
            rope_hd,
        )
    }

    /// RoPE fusion for the two same-activation projections (wq_b + idx_wq_b):
    /// one launch computes both and rotates both, each family with its own head
    /// width but the same rope length / cos-sin table / position counter.
    /// Ok(false) => the caller runs `lin2` + both standalone ropes.
    #[allow(clippy::too_many_arguments)]
    fn lin2_rope(
        &self,
        a: *const f32,
        k: i32,
        w1: &crate::dsv41::load::DevTensor,
        ws1: &crate::dsv41::load::DevTensor,
        n1: i32,
        out1: *mut f32,
        w2: &crate::dsv41::load::DevTensor,
        ws2: &crate::dsv41::load::DevTensor,
        n2: i32,
        out2: *mut f32,
        rope_rd: i32,
        rope_hd1: i32,
        rope_hd2: i32,
    ) -> Result<bool> {
        if !rope_fuse() || !self.dev.supports_rope_fuse() {
            return Ok(false);
        }
        self.quant1(a, k)?;
        self.dev.gemm_fp8_mx2_rope(
            self.s.xq.as_u8(),
            self.s.xsc.as_f32(),
            w1.as_u8(),
            ws1.as_u8(),
            std::ptr::null(),
            out1,
            n1,
            w2.as_u8(),
            ws2.as_u8(),
            std::ptr::null(),
            out2,
            n2,
            k,
            self.cos.as_f32(),
            self.sin.as_f32(),
            self.s.pos_ctr.ptr as *const std::os::raw::c_int,
            1,
            0,
            0,
            false,
            rope_rd,
            rope_hd1,
            rope_hd2,
        )
    }

    /// NORM_FUSE: the rope GEMV whose PROLOGUE also normalises + quantises the
    /// raw f32 row `a_raw` (k elements, norm weight `qw`, eps `eps`), instead of
    /// reading a pre-quantised `xq`/`xsc`. No `quant1` runs and neither `xq` nor
    /// `xsc` is touched, so this must only be used when `a_raw` really is the raw
    /// producer output - i.e. when `attention()` took the same fused path and
    /// left `qr` unnormalised (see `s.qr_raw`). Ok(false) when the .so lacks the
    /// symbol or the shape declines, so the caller keeps its old pair.
    #[allow(clippy::too_many_arguments)]
    fn lin_rope_norm(
        &self,
        a_raw: *const f32,
        qw: *const f32,
        eps: f32,
        k: i32,
        w: &crate::dsv41::load::DevTensor,
        ws: &crate::dsv41::load::DevTensor,
        n_out: i32,
        out: *mut f32,
        rope_rd: i32,
        rope_hd: i32,
    ) -> Result<bool> {
        if !rope_fuse() || !self.dev.supports_gemm_fp8_norm() {
            return Ok(false);
        }
        self.dev.gemm_fp8_mx_rope_norm(
            a_raw,
            qw,
            eps,
            w.as_u8(),
            ws.as_u8(),
            std::ptr::null(),
            out,
            n_out,
            k,
            self.cos.as_f32(),
            self.sin.as_f32(),
            self.s.pos_ctr.ptr as *const std::os::raw::c_int,
            1,
            0,
            0,
            false,
            rope_rd,
            rope_hd,
        )
    }

    /// f32 linear for one row. M=1 goes through our own GEMV: cuBLAS's GemmEx
    /// picked gemv2T at ~40 GFLOP/s for a single row (396us/call, 72 per step)
    /// while the weight read floors at ~112us.
    fn lin_f32(&self, a: *const f32, k: i32, w: &crate::dsv41::load::DevTensor, n_out: i32, out: *mut f32) -> Result<()> {
        if cublas_m1() {
            return self
                .dev
                .gemm_f32(a as *const c_void, w.ptr() as *const c_void, out, 1, n_out, k);
        }
        self.dev.gemv_f32(w.ptr() as *const f32, a, out, n_out, k)
    }

    /// [`Self::lin_f32`] issued on `s` instead of the main stream. Only the
    /// compressor's kvp/scp projections use it (`DSV41_COMPRESS_SIDE`), and the
    /// caller excludes the cuBLAS-M1 path (one handle, bound to the main
    /// stream), so this always takes our own GEMV.
    fn lin_f32_on(
        &self,
        a: *const f32,
        k: i32,
        w: &crate::dsv41::load::DevTensor,
        n_out: i32,
        out: *mut f32,
        s: ferrite_kernel::devrt::CuStream,
    ) -> Result<()> {
        self.dev.gemv_f32_on(w.ptr() as *const f32, a, out, n_out, k, s)
    }

    /// bf16 linear for one row. The bf16 path converts the *activation* to bf16
    /// and hands both to cuBLAS; our GEMV takes the activation in f32 and the
    /// weights natively bf16, which is both leaner and slightly more accurate
    /// (no activation rounding).
    fn lin_bf16(&self, a: *const f32, k: i32, w: &crate::dsv41::load::DevTensor, n_out: i32, out: *mut f32) -> Result<()> {
        if cublas_m1() {
            self.dev.f32_to_bf16(a, self.s.bf16.ptr as *mut c_void, k as i64)?;
            return self
                .dev
                .gemm_bf16(self.s.bf16.ptr as *const c_void, w.ptr() as *const c_void, out, 1, n_out, k);
        }
        self.dev.gemv_bf16(w.ptr() as *const c_void, a, out, n_out, k)
    }

    /// Engram: n-gram memory write-back into the hc residual stream, applied
    /// BEFORE the block at the layers the config lists (1 and 14). Mirrors the
    /// reference's `Engram.forward`: gather the `n_cols` hash rows from the
    /// (row-sharded) table, project them with `wkv` into one key per hc copy
    /// plus a shared value, gate that value by the normalised dot of the stream
    /// against the key, and add it to every copy.
    fn engram_apply(&mut self, layer: usize, li: usize) -> Result<()> {
        let cfg = self.cfg;
        let rank = self.rank();
        let (dim, hc, ehd) = (cfg.dim, cfg.hc_mult, cfg.engram_head_dim);
        let n_cols = cfg.engram_max_ngram_size.saturating_sub(1) * cfg.engram_n_heads;
        let (table, tsc, wkv, wsc, qw, kw) = {
            let ld = &self.w.layers[layer];
            (
                ld.engram_embed.as_ref(),
                ld.engram_embed_scale.as_ref(),
                ld.engram_wkv.as_ref(),
                ld.engram_wkv_scale.as_ref(),
                ld.engram_q_weight.as_ref(),
                ld.engram_k_weight.as_ref(),
            )
        };
        let (Some(table), Some(tsc), Some(wkv), Some(wsc), Some(qw), Some(kw)) =
            (table, tsc, wkv, wsc, qw, kw)
        else {
            return Ok(());
        };
        // This rank's slice of the row-parallel table: convert.py shards
        // `ceil(rows / world)` rows and zero-pads the tail.
        let world = self.world().max(1);
        let global_rows = cfg.engram_num_embeddings.get(li).copied().unwrap_or(0) as usize;
        let per = global_rows.div_ceil(world);
        let ids = (self.s.eng_ids.ptr as *const i64).wrapping_add(li * n_cols);
        self.dev.engram_gather(
            table.ptr() as *const u8,
            tsc.ptr() as *const u8,
            ids,
            self.s.eng_rows.ptr as *mut f32,
            1,
            n_cols as i32,
            ehd as i32,
            (rank * per) as i64,
            per as i64,
        )?;
        // rows another rank owns arrived as 0, so the sum yields the real row
        if let Some(c) = self.comm.clone() {
            c.all_reduce_inplace(self.s.eng_rows.ptr as *mut std::ffi::c_void, fb(n_cols * ehd))?;
        }
        // kv = wkv(gathered): [(hc + 1) * dim] = [key(hc*dim), value(dim)]
        //
        // engram f32 direct read (the same lever as wo_b's DSV41_WOB_F32): the GEMV
        // stages the RAW f32 `eng_rows` through `dsv41_gemm_fp8_mx_f32` instead of
        // the `quant_fp8` pair, so the quantisation launch disappears (one per
        // engram layer: L1 + L14 = 2/step). Only the data path changes - the grid
        // stays the plain `g_gemv_warps` / ceil(n/warps) shape - so it avoids B1's
        // 32-warp SM-utilisation loss. NOT bit-identical: it skips the
        // quantise->dequantise round trip and is strictly MORE accurate, which is
        // the right direction for a row that feeds the gated write-back.
        //
        // ⚠️ It is the POST-AR row that is read, never the pre-AR per-rank row: the
        // collective above sums the ranks in f32 and this reads that sum directly.
        // Emitting fp8 BEFORE the AR would be a different (and wrong) program, since
        // fp8(Σ rows) != Σ fp8(row) - that is the quant-final-sweep warning, and it
        // does not apply to the post-AR read.
        //
        // Gate = the symbol probe only (`supports_gemm_fp8_f32`): a stale .so has no
        // entry, and a shape decline (Ok(false), e.g. k not a multiple of 32) falls
        // through to the (quant_fp8, gemm_fp8_mx) pair below. Both are host-side and
        // shape-deterministic, so the captured decode graph stays consistent across
        // its replays (same contract as wo_b's f32 branch).
        let mut eng_f32 = false;
        if self.dev.supports_gemm_fp8_f32() {
            eng_f32 = self.dev.gemm_fp8_mx_f32(
                self.s.eng_rows.ptr as *const f32,
                wkv.ptr() as *const u8,
                wsc.ptr() as *const u8,
                std::ptr::null(),
                self.s.eng_kv.ptr as *mut f32,
                ((hc + 1) * dim) as i32,
                (n_cols * ehd) as i32,
            )?;
        }
        if !eng_f32 {
            self.dev.quant_fp8(
                self.s.eng_rows.ptr as *const f32,
                self.s.eng_xq.ptr as *mut u8,
                self.s.eng_xsc.ptr as *mut f32,
                1,
                (n_cols * ehd) as i32,
                32,
                true,
            )?;
            self.gemm_fp8_mx_or_swap(
                self.s.eng_xq.ptr as *const u8,
                self.s.eng_xsc.ptr as *const f32,
                wkv.ptr() as *const u8,
                wsc.ptr() as *const u8,
                std::ptr::null(),
                self.s.eng_kv.ptr as *mut f32,
                ((hc + 1) * dim) as i32,
                (n_cols * ehd) as i32,
            )?;
        }
        // gated write-back into h (in place)
        self.dev.engram_apply(
            self.s.h.ptr as *mut f32,
            self.s.eng_kv.ptr as *const f32,
            qw.ptr() as *const f32,
            kw.ptr() as *const f32,
            std::ptr::null(),
            1,
            hc as i32,
            dim as i32,
            cfg.norm_eps,
        )?;
        Ok(())
    }

    /// Fold the segment-C `hc_post_inplace` into the AR that just produced `s.o`.
    ///
    /// The AR is the last writer of `o` — the hc-post's `x` — and the hc-post
    /// reads/writes only the residual stream, so the two can share one launch:
    /// the fused pubred epilogue writes `res` in place from the AR result it
    /// already holds in registers. Returns `Ok(true)` when the fold ran, in
    /// which case the caller MUST skip the standalone `hc_post_inplace`. Every
    /// decline (env off, segment C not in-place, no v5 protocol, shape mismatch,
    /// or an .so without the symbol) returns `Ok(false)` and leaves the caller on
    /// the exact pair it used before.
    fn ar_hc_post_fold(
        &self,
        c: &std::sync::Arc<Collective>,
        buf: *mut std::ffi::c_void,
        len: usize,
        add_in: Option<*const f32>,
    ) -> Result<bool> {
        if !Self::hcpost_epi() || !Self::fuse_c() {
            // fuse_c() is the same gate the caller uses to choose between the
            // in-place hc-post and the h2-staging pair; folding under the
            // staging branch would apply the mix TWICE.
            return Ok(false);
        }
        // Tail-split join, placed HERE rather than after the AR in `layer`.
        // The fused epilogue below is the first consumer of the LATE half's
        // `comb`/`post` (and its `hc_res` write to `s.h` clobbers memory the
        // LATE half is still reading), so the side stream's `join_ev` must be
        // waited on the main stream BEFORE this launch — otherwise the split's
        // post-AR join in `layer` orders nothing and the graph has no edge
        // between the side LATE node and this kernel. No-op when no split is
        // armed (the `hc_split_armed` flag), or when the split is off.
        self.dev.hc_tail_join()?;
        let (dim, hc) = (self.cfg.dim, self.cfg.hc_mult);
        // ADD_EPI: the deferred shared-expert merge rides along in the store
        // epilogue. Declines (no symbol) leave the caller on add + the plain
        // hcpost fold, so nothing is ever dropped.
        if let Some(p) = add_in {
            if self.dev.supports_ar_hcpost_add() {
                return c.all_reduce_inplace_hcpost_add(
                    buf,
                    len,
                    p,
                    self.s.h.ptr as *mut f32,
                    self.s.post.as_f32(),
                    self.s.comb.as_f32(),
                    hc as i32,
                    dim as i32,
                );
            }
            return Ok(false);
        }
        c.all_reduce_inplace_hcpost(
            buf,
            len,
            self.s.h.ptr as *mut f32,
            self.s.post.as_f32(),
            self.s.comb.as_f32(),
            hc as i32,
            dim as i32,
        )
    }

    /// ADD_EPI readiness: the biased AR entry this layer's fold would actually
    /// use must exist in the loaded .so. With the default hc-post fold
    /// ([`Self::hcpost_epi`] + [`Self::fuse_c`]) the MoE AR goes through the
    /// `hcpost_add` entry; otherwise it needs the plain `_add` entry. Either way a
    /// missing symbol keeps the standalone `add_inplace` (and the fold's own
    /// shape decline has the same effect at `moe_reduce`).
    fn add_epi_ready(&self) -> bool {
        if !add_epi() {
            return false;
        }
        if Self::hcpost_epi() && Self::fuse_c() {
            self.dev.supports_ar_hcpost_add()
        } else {
            self.dev.supports_ar_add()
        }
    }

    /// The MoE's all-reduce, issued OUTSIDE any captured segment: a CUDA graph
    /// cannot contain the host barrier that this path still uses, so the segment
    /// boundary sits exactly here. Returns whether the segment-C hc-post was
    /// folded into the reduce (see [`Self::ar_hc_post_fold`]).
    fn moe_reduce(&mut self, layer: usize) -> Result<bool> {
        let dim = self.cfg.dim;
        // ADD_EPI: the merge `moe(layer, ..)` deferred (see `add_epi()`). READ,
        // never consumed: the step-body graph is captured after the first decode
        // step, so this host-side value (and the AR args it feeds) is baked at
        // capture; a `take()` would silently drop the merge on every replay.
        let add_in = self.moe_add_in[layer];
        // routed experts are expert-parallel, so each rank holds a partial sum
        if let Some(c) = self.comm.clone() {
            let folded =
                self.ar_hc_post_fold(&c, self.s.o.ptr as *mut std::ffi::c_void, fb(dim), add_in)?;
            if !folded {
                // ADD_EPI on the plain AR: publish `o + ex_out` straight from the
                // store epilogue. `added == false` (no symbol) falls back to the
                // standalone merge before the untouched AR.
                let mut added = false;
                if let Some(p) = add_in {
                    if self.dev.supports_ar_add() {
                        added = c.all_reduce_inplace_add(
                            self.s.o.ptr as *mut std::ffi::c_void,
                            fb(dim),
                            p,
                        )?;
                    }
                }
                if !added {
                    if add_in.is_some() {
                        self.dev.add_inplace(&self.s.o, &self.s.ex_out, dim as i64)?;
                    }
                    c.all_reduce_inplace(self.s.o.ptr as *mut std::ffi::c_void, fb(dim))?;
                }
            }
            c.end_round();
            return Ok(folded);
        }
        // Single device: no AR to fold into, so the merge has to run here.
        if add_in.is_some() {
            self.dev.add_inplace(&self.s.o, &self.s.ex_out, dim as i64)?;
        }
        Ok(false)
    }

    /// One decode step. Returns the logits for the fed token.
    pub fn step(&mut self, token: u32, pos: usize) -> Result<u32> {
        let ids = [token as i32];
        self.dev.upload_f32_at(self.s.ids.ptr, 0, unsafe {
            std::slice::from_raw_parts(ids.as_ptr() as *const f32, 1)
        })?;
        self.step_impl(token, pos)
    }

    /// Decode steady state: the token is ALREADY in s.ids on the device (the
    /// previous step's argmax wrote it), and the caller knows its value because
    /// that step returned it — the host value feeds only the n-gram hash, so this
    /// path does ZERO host-to-device traffic.
    pub fn step_dev(&mut self, token: u32, pos: usize) -> Result<u32> {
        self.decode_steps = self.decode_steps.wrapping_add(1);
        self.step_impl(token, pos)
    }

    /// The whole decode step as ONE CUDA graph (DSV41_GRAPH_STEP=1, the default):
    /// every per-step value now lives on the DEVICE - the position counter, the
    /// per-layer latent counters, the all-reduce epoch - so no launch argument
    /// changes from token to token and the capture is legal. The capture happens
    /// on the second step: the first warms every kernel, builds the lazy device
    /// state and sizes cublas' workspaces, all of which are illegal inside a
    /// capture. Capturing records WITHOUT executing, so the graph is launched
    /// straight afterwards to do this step's real work.
    fn step_impl(&mut self, token: u32, pos: usize) -> Result<u32> {
        let cfg = self.cfg;
        static GRAPH_STEP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let want = *GRAPH_STEP.get_or_init(|| {
            // Things that must be OFF for a whole-step capture, each because it
            // makes a host round trip inside the recorded region:
            //   DSV41_ENG_HOST  - the host n-gram hash uploads the ids per step
            //   DSV41_STATS     - the per-layer probes download tensors
            // And one thing that must be ON: the all-reduce has to be the DEVICE
            // side AR v5, because a host barrier is not a CUDA call - it would not
            // be recorded, and the replayed graph would lose the inter-rank sync.
            // ar_v5() therefore also turns on with the graph (see tp.rs).
            let host_hash = std::env::var("DSV41_ENG_HOST").map(|v| v != "0").unwrap_or(false);
            let probes = std::env::var("DSV41_STATS").map(|v| v != "0").unwrap_or(false);
            // VERIFIED: a SINGLE request with the graph on is bit-identical to the
            // per-kernel path (DSV41_TOKTRACE compared step by step, no divergence),
            // so the captured operator set is right. With SEVERAL sequential
            // requests it is not: requests one to three (short, one step each, so
            // they never reach the capture) answer correctly, and the first long
            // request then dies with an illegal memory access on a rank that varies
            // run to run, after which the engine is sticky-faulted and every later
            // request is empty. The regression and the fault were chased through the
            // whole device layer (capture mode, streams, cublas stream, graph
            // wrappers, buffer accessors, the 52 launch wrappers - all byte-equal to
            // the pre-refactor file), through a per-request graph drop, and through a
            // rank rendezvous around the capture; none of them changed the outcome,
            // SO THE DEFAULT IS NOW THE GRAPH (2026-09-11): the cause WAS located - the indexer
            // sized its dynamic shared memory from a capture-frozen per-step value while
            // scanning the live counter - and fixed by decoupling the size from the count,
            // then verified with 12/12 answers and zero faults. DSV41_GRAPH_STEP=0 opts out.
            !host_hash
                && !probes
                && std::env::var("DSV41_GRAPH_STEP").map(|v| v != "0").unwrap_or(true)
        });
        // ONLY on the decode path: capturing during prefill froze the prefill
        // branches into the graph, so the decode replays took the wrong ones (the
        // observable symptom was output that looked like a plausible continuation of
        // something else - the model was being fed a mis-processed prompt).
        if want && self.decode_steps >= 1 {
            // Rendezvous with the peers BEFORE the branch. A capture only RECORDS
            // its all-reduce kernels, while the device-side AR has no host barrier
            // of its own (end_round returns early under ar_v5, since the publish
            // chain covers the normal case). Without this the ranks keep their
            // microsecond-level skew, and a peer that is EXECUTING its AR polls for
            // a stamp that a still-recording rank is only writing down, gives up,
            // and reads staging that was never published. Measured: with the graph
            // on, one request was bit-identical to the per-kernel path, yet the
            // fourth of six sequential requests died with an illegal access on a
            // rank that varied run to run - a race, not a deterministic fault.
            if let Some(c) = self.comm.as_ref() {
                c.host_barrier();
            }
            if let Some(e) = self.step_graph {
                self.dev.graph_launch(e)?;
            } else {
                self.dev.capture_begin()?;
                self.step_body(token, pos)?;
                let g = self.dev.capture_end()?;
                // The recording is finished; align again so no rank starts
                // replaying (and thus publishing) while a peer is still capturing.
                if let Some(c) = self.comm.as_ref() {
                    c.host_barrier();
                }
                let e = self.dev.graph_instantiate(g)?;
                self.dev.graph_free(g, std::ptr::null_mut())?;
                self.dev.graph_launch(e)?; // the capture did not execute
                self.step_graph = Some(e);
            }
        } else {
            self.step_body(token, pos)?;
        }
        self.step_count = self.step_count.wrapping_add(1);
        // the ONLY host read on the decode path: 4 bytes for EOS and printing
        self.dev.sync()?;
        let tok = self.dev.download_u32(self.s.ids.ptr)?;
        // Token trace for bisecting the graph's cumulative error. It runs OUTSIDE
        // the captured region on purpose (the token is already on the host here),
        // so DSV41_TOKTRACE=1 changes nothing about the capture and the two modes'
        // token sequences can be compared step by step.
        if std::env::var("DSV41_TOKTRACE").map(|v| v != "0").unwrap_or(false) {
            eprintln!(
                "[toktr] ds={} pos={} tok={}",
                self.decode_steps, self.step_count, tok
            );
        }
        if std::env::var("DSV41_TOP5").map(|v| v != "0").unwrap_or(false) {
            let mut lg = vec![0f32; cfg.vocab_size];
            let b = Device::view(self.s.logits.ptr, cfg.vocab_size * 4);
            self.dev.download_f32(&b, &mut lg)?;
            let mut top: Vec<(usize, f32)> = lg.iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("[top5] n={} top={:?}", lg.len(), &top[..5.min(top.len())]);
        }
        Ok(tok)
    }

    /// The step's kernels with NO host round trip in between: embedding through
    /// the argmax (which is also what advances the position counter). This is the
    /// region a graph captures; everything host-side lives in step_impl.
    fn step_body(&mut self, token: u32, pos: usize) -> Result<()> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        // embedding + hyper-connection expansion lands directly in `h`
        self.dev.embed_expand_dev(
            self.w.embed.as_ref().unwrap().ptr(),
            self.s.ids.as_i32(),
            self.s.h.ptr as *mut f32,
            1,
            dim as i32,
            hc as i32,
            cfg.vocab_size as i32,
        )?;

        // probe: h right after embedding + hc expansion
        if hc_dbg() {
            let hv = self.dl(self.s.h.as_f32(), hc * dim)?;
            let r = (hv.iter().map(|v| v * v).sum::<f32>() / hv.len() as f32).sqrt();
            eprintln!("[mine] xin_rms={}", (r * 1e6).round() / 1e6);
        }

        // the initial collapse takes copy 0 of the stream: the constant [1,0,0,0]
        // lives in a device buffer uploaded once at reset; this 16-byte D2D is
        // graph-capturable where the old per-step H2D upload was not.
        self.dev.memcpy_d2d(
            self.s.pre_a.ptr,
            self.s.premix_const.ptr,
            hc * std::mem::size_of::<f32>(),
        )?;
        let mut premix_slot_idx = 0usize;
        // engram hashes for this token (all engram layers at once; the reference
        // computes them in one shot and indexes per layer). The state's token
        // cache spans prefill + decode, so this must be called every step.
        let mut eng_layer_of: Vec<(usize, usize)> = Vec::new();
        if let (Some(ng), Some(lay), Some(map)) =
            (self.ngram.as_mut(), self.eng_layout.as_ref(), self.eng_map.as_ref())
        {
            if eng_host() {
                let hs = ng.forward_row(lay, map, 0, &[token], pos, None);
                let n_cols = lay.n_hash_cols();
                if hs.len() >= lay.layers.len() * n_cols {
                    let mut bytes = Vec::with_capacity(hs.len() * 8);
                    for v in hs.iter() {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    self.dev
                        .upload_bytes_at(&self.s.eng_ids, &bytes[..lay.layers.len() * n_cols * 8])?;
                }
            } else {
                // DEVICE hash: the token is read straight from s.ids (the buffer
                // the previous step's argmax wrote, or the prefill upload) and
                // the position counter lives on the device — no H2D, and the
                // call is graph-capturable. Bit-identical to the host reference
                // (the same serial arithmetic).
                if self.eng_dev.is_none() {
                    let e = build_eng_dev(&self.dev, lay, map, ng.max_seq)?;
                    self.eng_dev = Some(e);
                }
                let e = self.eng_dev.as_ref().unwrap();
                self.dev.engram_hash_step(
                    e.map.ptr as *const i64,
                    e.cache.ptr as *mut i64,
                    e.mults.ptr as *const i64,
                    e.lms.ptr as *const u64,
                    e.offs.ptr as *const u64,
                    self.s.eng_ids.ptr as *mut i64,
                    self.s.ids.as_i32(),
                    self.s.pos_ctr.ptr as *const i32,
                    map.map.len() as i64,
                    lay.layers.len() as i32,
                    lay.max_ngram_size as i32,
                    lay.n_heads as i32,
                    ng.pad_id,
                )?;
            }
            for l in 0..cfg.n_layers {
                if let Some(li) = lay.engram_index(l) {
                    eng_layer_of.push((l, li));
                }
            }
        }
        let mut t_attn = std::time::Duration::ZERO;
        let mut t_moe = std::time::Duration::ZERO;
        for layer in 0..cfg.n_layers {
            // the engram writes into the residual stream BEFORE the block runs
            if let Some(&(_, li)) = eng_layer_of.iter().find(|(l, _)| *l == layer) {
                self.engram_apply(layer, li)?;
            }
            let _ta = std::time::Instant::now();
            premix_slot_idx = self.layer(layer, pos, premix_slot_idx)?;
            let _el = _ta.elapsed();
            if phase_dbg() {
                let _ = (&mut t_attn, &mut t_moe);
                eprintln!("[phase] L{layer} layer={:?}", _el);
            }
            if stats_dbg() && layer % stats_every() == 0 {
                self.stats(&format!("L{layer} h"), &self.s.h, hc * dim)?;
            }
        }
        self.dev.memcpy_d2d(
            self.s.pre_a.ptr,
            self.s.premix_const.ptr,
            hc * std::mem::size_of::<f32>(),
        )?;
        if Self::fuse_b1() {
            self.dev.hc_collapse_norm(
                self.s.h.ptr as *mut f32,
                self.premix_slot(1).as_f32(),
                self.w.norm.as_ref().unwrap().as_f32(),
                self.s.xn.ptr as *mut f32,
                1,
                hc as i32,
                dim as i32,
                cfg.norm_eps,
            )?;
        } else {
        self.dev.hc_collapse(
            self.s.h.ptr as *const f32,
            self.premix_slot(1).as_f32(), // attn_pre stays on the device
            self.s.x.ptr as *mut f32,
            1,
            hc as i32,
            dim as i32,
        )?;
        self.dev.rmsnorm(
            self.s.x.ptr as *const f32,
            self.w.norm.as_ref().unwrap().as_f32(),
            self.s.xn.ptr as *mut f32,
            1,
            dim as i32,
            cfg.norm_eps,
        )?;
        }
        // The head keeps the ACTIVATION in f32 (casting it to bf16 cost ~3 bits on
        // a 129280-way near-tie argmax - the reference keeps it in f32). The
        // weights now stay bf16 when the checkpoint stores them that way: the gemv
        // widens each one losslessly in-kernel (bf16 is a truncated f32) with the
        // same accumulation order as the f32 kernel, so the logits are
        // bit-identical while the step's largest single weight read - the
        // full-vocab head, replicated on every rank - halves.
        let head = self.w.head.as_ref().unwrap();
        // Vocabulary slicing (DSV41_HEAD_SLICE): the head is REPLICATED on every
        // rank, so each rank can reduce just its own 1/world slice and the ranks
        // pick the global winner from one published u64 each. The step's largest
        // single weight read (1262 MB bf16 at 129280x5120) drops to 158 MB/rank.
        // The tie rule (lowest index) is preserved exactly: the packed key is
        // monotone in (value, -index), and the final takes the max in ascending
        // rank order, which is the same order the single-rank argmax walked.
        let world = self.world();
        let rank = self.rank();
        let seg = if world > 1 { cfg.vocab_size / world } else { 0 };
        let sliced = head_slice()
            && head.dtype == "BF16"
            && world > 1
            && cfg.vocab_size % world == 0
            && self
                .comm
                .as_ref()
                .map(|c| c.uses_v5())
                .unwrap_or(false);
        if head.dtype == "BF16" {
            if sliced {
                let head_ptr =
                    (head.ptr() as *const u8).wrapping_add(rank * seg * dim as usize * 2);
                self.dev.gemv_bf16(
                    head_ptr as *const c_void,
                    self.s.xn.ptr as *const f32,
                    self.s.logits.ptr as *mut f32,
                    seg as i32,
                    dim as i32,
                )?;
            } else {
                self.dev.gemv_bf16(
                    head.ptr(),
                    self.s.xn.ptr as *const f32,
                    self.s.logits.ptr as *mut f32,
                    cfg.vocab_size as i32,
                    dim as i32,
                )?;
            }
        } else {
            self.lin_f32(
                self.s.xn.ptr as *const f32,
                dim as i32,
                head,
                cfg.vocab_size as i32,
                self.s.logits.ptr as *mut f32,
            )?;
        }
        if sliced {
            self.stats("final logits (slice)", &self.s.logits, seg)?;
        } else {
            self.stats("final logits", &self.s.logits, cfg.vocab_size)?;
        }
        // Device-side argmax (the GLM HEAD_DEV pattern): the next token lands
        // straight in s.ids, which the next step's embedding reads — no 517 KB
        // full-vocab download, no O(vocab) host scan, and the token itself never
        // crosses to the host and back.
        if sliced {
            let c = self.comm.as_ref().unwrap();
            // One round of the shared v5 epoch sequence: the local slice reduces
            // through argmax_kernel (packed key carries the GLOBAL index, ties ->
            // lowest), then the exchange kernel lands the key in every peer's
            // CURRENT parity slot, stamps, advances the device epoch, polls, and
            // maxes. stride = the slot size; the key sits at the slot's offset 0
            // under the same parity addressing the AR store uses.
            let ok = self.dev.argmax_sliced(
                self.s.logits.ptr as *const f32,
                seg as i32,
                (rank * seg) as i32,
                self.s.ids.ptr as *mut std::ffi::c_int,
                self.s.argmax_packed.ptr as *mut u64,
                self.s.pos_ctr.ptr as *mut std::ffi::c_int,
                c.peer_slots_u64(),
                c.peer_stamps_u32(),
                c.epoch_dev(),
                c.staging_dev() as *mut u64,
                c.ready_local_dev(),
                world as i32,
                rank as i32,
                c.bytes as i64,
            )?;
            if !ok {
                // The loaded .so has no cross-rank argmax. A slice-local argmax
                // would silently return the wrong token, so redo the head over the
                // full vocabulary and take the ordinary argmax - correct, just
                // slower.
                self.dev.gemv_bf16(
                    head.ptr(),
                    self.s.xn.ptr as *const f32,
                    self.s.logits.ptr as *mut f32,
                    cfg.vocab_size as i32,
                    dim as i32,
                )?;
                self.dev.argmax(
                    self.s.logits.ptr as *const f32,
                    self.s.ids.ptr as *mut std::ffi::c_int,
                    cfg.vocab_size as i32,
                    self.s.pos_ctr.ptr as *mut std::ffi::c_int,
                )?;
                return Ok(());
            }
        } else {
            self.dev.argmax(
                self.s.logits.ptr as *const f32,
                self.s.ids.ptr as *mut std::ffi::c_int,
                cfg.vocab_size as i32,
                self.s.pos_ctr.ptr as *mut std::ffi::c_int,
            )?;
        }
        Ok(())
    }

    // ========================================================================
    // DSpark verify: the m-row forward (`step_rows` and its per-block helpers)
    // ========================================================================
    //
    // # What this is
    //
    // `step_body` runs ONE token. The speculative-decoding target verify needs m
    // rows of the SAME stack in one pass: the block `[t0, d1..dm-1]`, where `t0` is
    // the anchor the previous step committed and `d*` are the draft model's
    // proposals, all placed at consecutive positions `pos_base + r`. The output is
    // one argmax per row; the host then accepts the longest matching prefix.
    //
    // # The numerical-domain rule (why this is written the way it is)
    //
    // Greedy speculative decoding is only correct if, for EVERY row, the verify
    // argmax is the token a per-token decode at that position would have produced.
    // So this path deliberately reuses the single-row kernels instead of
    // generalising the fused ones:
    //
    //   * row-INDEPENDENT ops that already have a real `rows` dimension are called
    //     ONCE with `rows = m` (`rmsnorm`, `hc_mixes`/`hc_collapse`/`hc_post`,
    //     `embed_expand_dev`, `route_topk`, `quant_fp8`/`quant_fp4`,
    //     `engram_apply`, `swiglu_limit`, the grouped `gemm_fp8_mx`, the AR);
    //   * per-row (M=1) ops are called ONCE PER ROW with the row's own pointers and
    //     position (`lin`/`lin_f32`/`lin_bf16`, `gemv_bf16`, `sparse_attn`,
    //     `indexer_topk`, `apply_rope`, the routed-expert kernels);
    //   * the single-row FUSIONS (`lin2`, `lin_rope*`, `sparse_attn_orope`,
    //     FUSE_B1/B2/C, DUEL_CHAIN, COMPRESS_SIDE, MOE_DUAL, HCPOST_EPI, ADD_EPI,
    //     the wo pair, the sliced head) are NOT taken: each is documented as
    //     bit-identical to the pair it replaces, so the naive pair IS the parity
    //     target rather than a different program.
    //
    // # The position counter
    //
    // `pos_ctr` is READ-ONLY here. The single-row step advances it from the
    // argmax kernel (the step's last kernel); a verify pass must not, because the
    // accept logic owns that decision and every kernel in the pass has to see the
    // same `pos_base`. Row r's position therefore lives in `pos_rows[r]`, and the
    // kernels that take a position POINTER (`engram_hash_step`, `ring_append`,
    // `window_idxs`) are pointed at that table; the head's argmax is called with a
    // NULL `pos_ctr` so it cannot advance either.
    //
    // # TODO (known modelling gaps — reported, not hidden)
    //
    // 1. COMPRESSOR: `compressor_fused` rejects `seqlen != 1` outright
    //    (dsv41_kernels.cu) and the state/pool kernels' decode branches carry ONE
    //    row, so `compress_rows` runs the pool+commit pair with `seqlen = m` and
    //    only the group the single-row arithmetic knows about is formed. A block
    //    that completes several groups (ratio 2, m = 6: up to 3) is not modelled
    //    yet.
    // 2. INDEXER: the candidate count is the live device counter, which includes
    //    the group this block may have just committed — whose index key is
    //    published AFTER the selection (the single-row path has the same order, it
    //    just has one row). Newly created groups are not excluded from the
    //    candidate set yet.
    // 3. HEAD: the full vocabulary is used on every rank (no `HEAD_SLICE`):
    //    `argmax_sliced` advances `pos_ctr` and consumes one v5 epoch round per
    //    call, which a 6-row block cannot afford. The head is replicated, so the
    //    unsliced argmax selects the same token; a multi-row sliced argmax is the
    //    TODO.
    // 4. The tcgen05 MXFP4 gate/up arm (`DSV41_EXPERT_TCGEN05_MXF4`, default OFF)
    //    is not wired into `moe_rows`: the rows path pins the proven GEMV arm.

    /// DSpark verify forward: run an m-row block through the layer stack in ONE
    /// pass and return each row's argmax.
    ///
    /// `toks[0]` is the anchor token, `toks[1..]` the draft proposals; row `r`
    /// sits at position `pos_base + r` where `pos_base` is the device position
    /// counter's current value (read once, never written). The returned `Vec<u32>`
    /// has one entry per row: `out[r]` is the token row `r` predicts, i.e. what
    /// the draft's `toks[r+1]` is matched against.
    ///
    /// The window ring is appended with the whole block (each row's window is
    /// causal: row r sees `[pos_base+r-window+1, pos_base+r]`, which includes the
    /// block's own rows 0..r through the ring geometry), and the KV the rest of
    /// the run will read is exactly what this pass wrote — so a verify pass leaves
    /// the caches consistent with the accepted prefix for whatever the caller
    /// decides to keep.
    ///
    /// See the section comment above for the numerical-domain rule, the position
    /// discipline and the known modelling gaps.
    pub fn step_rows(&mut self, toks: &[u32]) -> Result<Vec<u32>> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let m = toks.len();
        if m == 0 {
            return Ok(Vec::new());
        }
        if m > VERIFY_ROWS {
            return Err(FerriteError::Config(format!(
                "step_rows: {m} rows exceed the allocated verify block ({VERIFY_ROWS})"
            )));
        }
        // The row positions. One D2H: the counter is device-resident (the argmax
        // advances it), and EVERY row's position has to be materialised somewhere
        // because the kernels that take a position take a POINTER. This runs
        // between steps, so the read cannot stall anything that matters; the value
        // itself never crosses back to the device afterwards (pos_rows is an H2D of
        // m ints and is then read on-device).
        let pos_base = self.dev.download_u32(self.s.pos_ctr.ptr as *const c_void)? as i32;
        let pos_rows: Vec<i32> = (0..m).map(|r| pos_base + r as i32).collect();

        let ids: Vec<i32> = toks.iter().map(|&t| t as i32).collect();
        self.ul_i32(self.s.ids_r.ptr, &ids)?;
        self.ul_i32(self.s.pos_rows.ptr, &pos_rows)?;
        // every row's incoming premix is the one-hot [1,0,0,0] (the m-row twin of
        // the `premix_const` copy `step_body` does before its loop)
        let mut pm = vec![0f32; m * hc];
        for r in 0..m {
            pm[r * hc] = 1.0;
        }
        self.ul_f32(self.s.premix_r.ptr, &pm)?;

        // embedding + hyper-connection expansion, all m rows in one launch
        self.dev.embed_expand_dev(
            self.w.embed.as_ref().unwrap().ptr(),
            self.s.ids_r.as_i32(),
            self.s.h_r.ptr as *mut f32,
            m as i32,
            dim as i32,
            hc as i32,
            cfg.vocab_size as i32,
        )?;

        // engram hashes for the block. `engram_hash_step` is a SINGLE-token kernel
        // (it reads `*pos_ctr` and one token id), so it is called once per row with
        // the row's own position; each row's hashes land at
        // `eng_ids_r[r*n_eng*n_cols]`, the same per-token layout `step_body` fills.
        let mut eng_layer_of: Vec<(usize, usize)> = Vec::new();
        if let (Some(ng), Some(lay), Some(map)) =
            (self.ngram.as_mut(), self.eng_layout.as_ref(), self.eng_map.as_ref())
        {
            let n_cols = lay.n_hash_cols();
            let n_eng = lay.layers.len();
            if eng_host() {
                // A/B fallback (DSV41_ENG_HOST=1): the host hash + an upload. The
                // state's token cache spans prefill + decode, so the rows are hashed
                // in ascending position order.
                let mut bytes: Vec<u8> = Vec::with_capacity(m * n_eng * n_cols * 8);
                for r in 0..m {
                    let hs = ng.forward_row(
                        lay,
                        map,
                        0,
                        &[toks[r]],
                        (pos_base + r as i32) as usize,
                        None,
                    );
                    for v in hs.iter().take(n_eng * n_cols) {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }
                self.dev.upload_bytes_at(&self.s.eng_ids_r, &bytes)?;
            } else {
                if self.eng_dev.is_none() {
                    let e = build_eng_dev(&self.dev, lay, map, ng.max_seq)?;
                    self.eng_dev = Some(e);
                }
                let e = self.eng_dev.as_ref().unwrap();
                for r in 0..m {
                    self.dev.engram_hash_step(
                        e.map.ptr as *const i64,
                        e.cache.ptr as *mut i64,
                        e.mults.ptr as *const i64,
                        e.lms.ptr as *const u64,
                        e.offs.ptr as *const u64,
                        (self.s.eng_ids_r.ptr as *mut i64).wrapping_add(r * n_eng * n_cols),
                        (self.s.ids_r.as_i32()).wrapping_add(r),
                        (self.s.pos_rows.ptr as *const i32).wrapping_add(r),
                        map.map.len() as i64,
                        n_eng as i32,
                        lay.max_ngram_size as i32,
                        lay.n_heads as i32,
                        ng.pad_id,
                    )?;
                }
            }
            for l in 0..cfg.n_layers {
                if let Some(li) = lay.engram_index(l) {
                    eng_layer_of.push((l, li));
                }
            }
        }

        // ---- the layer stack ----
        // `pa` follows `layer()`'s convention (see `premix_row_slot`): 0 = the
        // constant incoming block, 2 = the previous layer's ffn_pre. Every layer
        // reports 2, so only the first layer's call starts at 0.
        let mut pa = 0usize;
        for layer in 0..cfg.n_layers {
            if let Some(&(_, li)) = eng_layer_of.iter().find(|(l, _)| *l == layer) {
                self.engram_apply_rows(layer, li, m)?;
            }
            pa = self.layer_rows(layer, m, pos_base, pa)?;
        }
        debug_assert_eq!(pa, 2, "layer_rows must report the ffn_pre slot");

        // final collapse + norm with the LAST layer's attn_pre (slot 1), exactly as
        // `step_body` does after its loop
        self.dev.hc_collapse(
            self.s.h_r.ptr as *const f32,
            self.s.pre_r.as_f32(),
            self.s.x_r.ptr as *mut f32,
            m as i32,
            hc as i32,
            dim as i32,
        )?;
        self.dev.rmsnorm(
            self.s.x_r.ptr as *const f32,
            self.w.norm.as_ref().unwrap().as_f32(),
            self.s.xn_r.ptr as *mut f32,
            m as i32,
            dim as i32,
            cfg.norm_eps,
        )?;

        // ---- head + per-row argmax ----
        // Same call `step_body` makes for its single token (f32 activation, bf16
        // weight read in-kernel), once per row into that row's logits slot.
        let head = self.w.head.as_ref().unwrap();
        for r in 0..m {
            let xnr = (self.s.xn_r.ptr as *const f32).wrapping_add(r * dim);
            let lg = (self.s.logits_r.ptr as *mut f32).wrapping_add(r * cfg.vocab_size);
            if head.dtype == "BF16" {
                self.dev
                    .gemv_bf16(head.ptr(), xnr, lg, cfg.vocab_size as i32, dim as i32)?;
            } else {
                self.lin_f32(xnr, dim as i32, head, cfg.vocab_size as i32, lg)?;
            }
            // NULL pos_ctr: this argmax must NOT advance the counter (the kernel
            // null-checks it) — the accept logic advances it once, for the accepted
            // prefix.
            self.dev.argmax(
                lg as *const f32,
                (self.s.argmax_r.ptr as *mut std::os::raw::c_int).wrapping_add(r),
                cfg.vocab_size as i32,
                std::ptr::null_mut(),
            )?;
        }
        // D2H of the m argmaxes. `download_u8` keeps this a plain byte copy, so no
        // f32 reinterpretation is involved.
        let mut bytes = vec![0u8; m * 4];
        let b = Device::view(self.s.argmax_r.ptr, m * 4);
        self.dev.download_u8(&b, &mut bytes)?;
        Ok((0..m)
            .map(|r| {
                u32::from_le_bytes([
                    bytes[4 * r],
                    bytes[4 * r + 1],
                    bytes[4 * r + 2],
                    bytes[4 * r + 3],
                ])
            })
            .collect())
    }

    /// The layers whose window ring the verify block appends to: the `owns_kv`
    /// test `attention_rows` applies, verbatim. `ring_owner_shared()` chooses
    /// between "every layer owns its own store" (the layer is its own owner, so
    /// every layer qualifies) and the `kv_owner` group mapping, where a compress
    /// consumer reads its source's ring and must not be appended to a second time.
    fn ring_owners(&self) -> Vec<usize> {
        (0..self.cfg.n_layers)
            .filter(|&l| {
                let owner = if ring_owner_shared() { self.kv_owner(l) } else { l };
                owner == l
            })
            .collect()
    }

    /// The layers whose compressor runs inside the verify block: the
    /// `compress_rows` gate in `attention_rows` (`compress_ratio > 0 &&
    /// is_kv_source`). A consumer of the same group only INHERITS the count and
    /// writes no state of its own.
    fn compress_sources(&self) -> Vec<usize> {
        (0..self.cfg.n_layers)
            .filter(|&l| self.cfg.compress_ratio(l) > 0 && self.cfg.is_kv_source(l))
            .collect()
    }

    /// Save the write set the verify block is about to touch; [`Self::dspark_rollback`]
    /// copies it back. See [`Self::dspark_shadow_step`] for the timeline.
    ///
    /// # Inventory — what `step_rows` writes, and why each entry is here
    ///
    /// For every layer that OWNS its window ring ([`Self::ring_owners`]):
    ///
    /// 1. the `m` ring slots the block lands in, `(pos + 1 + j) % window`. One
    ///    D2D per slot: the slots are `head_dim` floats each and the sequence
    ///    wraps at the ring's end, so a single contiguous copy is not available.
    ///    (The first cut is deliberately naive — ~15 launches for the production
    ///    geometry; a slot-permutation kernel is the M2 optimisation.)
    ///
    /// For every COMPRESS SOURCE ([`Self::compress_sources`]):
    ///
    /// 2. `state_kv` + `state_score` — the compressor carry, the full
    ///    `ratio * head_dim` each. The buffers are max-sized and each copy uses
    ///    its own layer's byte count (`ratio` is per layer: 1 or 2 here).
    /// 3. `latent` — **not** a per-step scratch. `indexer()`/`indexer_rows()` read
    ///    it on LATER steps as well (`indexer_owns_k` -> `lin_bf16(latent, wk)`),
    ///    and the pool rewrites it only on a step that completes a group. Leaving
    ///    the verify's latent behind would make the next non-completing step
    ///    re-publish a WRONG index key over the real chain's last key
    ///    (`index_k_publish` writes at `*clen - 1`, the same slot) — the one place
    ///    where a missed snapshot silently corrupts the main chain instead of
    ///    merely wasting work.
    /// 4. `out_rows` — assigned (not incremented) by every compressor call, and
    ///    read only by the commit inside that same call, so it is not strictly
    ///    required. It IS device state the verify mutates, it costs 4 bytes, and
    ///    restoring it makes "the main chain is untouched" a statement about the
    ///    whole buffer rather than an argument about read order.
    /// 5. the DEVICE compressed-row counter (`s.clen[layer]`, 4 bytes) and its
    ///    HOST mirror (`LayerCache::compress_len`, returned to the caller).
    ///
    /// # Deliberately NOT in the snapshot
    ///
    /// * `index_k`: the verify's indexer publishes at row `*clen - 1`; with `clen`
    ///   restored, that is exactly the row the next real commit overwrites, and
    ///   `indexer_topk` never reads past `*clen`, so the row is unreachable in
    ///   between.
    /// * the ring's COMPRESSED rows (`compress_commit` stores at row `window +
    ///   *clen`): the same argument — once `clen` is restored, that slot is the
    ///   next real commit's destination, and `sparse_attn` only reads rows
    ///   `< *clen`. The commit runs before the attention inside a step, so the
    ///   real value is in place before any read either way.
    /// * `kvp`/`scp`: pure per-call scratch (projected from `xn` and consumed by
    ///   the pool in the same call).
    /// * `LayerCache::idxs`: the verify writes only `s.idxs_r`/`s.idxs_win_r`; the
    ///   single-row path re-uploads and rewrites its own `idxs` every step.
    /// * the host-side fusion flags (`xq_of_xn_valid`, `qr_raw`, `idx_q_ready`,
    ///   `idx_q_rope`): they are pointer-gated to the SINGLE-ROW scratch (`s.xn`,
    ///   `s.qr`), which the m-row path never passes, so the verify can neither
    ///   consume nor leak one; each is cleared by the end of a real step anyway.
    /// * `pos_ctr`: never advanced by the verify — `step_rows` passes a NULL
    ///   counter to its per-row argmax.
    /// * everything in `s.*_r`: the m-row scratch is the verify's own.
    fn dspark_snapshot(&self, pos: usize, m: usize) -> Result<Vec<(usize, usize)>> {
        let cfg = self.cfg;
        let hd = cfg.head_dim;
        let win = cfg.window_size;
        let max_ratio = cfg.compress_ratios.iter().copied().max().unwrap_or(1).max(1);
        debug_assert!(m <= VERIFY_ROWS, "the snapshot is sized for a {VERIFY_ROWS}-row block");

        for &l in &self.ring_owners() {
            let ring = self.layers[l].ring.ptr as *const f32;
            let snap = (self.s.dspark_snap_ring.ptr as *mut f32)
                .wrapping_add(l * VERIFY_ROWS * hd);
            for j in 0..m {
                let slot = (pos + 1 + j) % win;
                self.dev.memcpy_d2d(
                    snap.wrapping_add(j * hd) as *mut c_void,
                    ring.wrapping_add(slot * hd) as *const c_void,
                    hd * std::mem::size_of::<f32>(),
                )?;
            }
        }

        let mut host = Vec::new();
        for &l in &self.compress_sources() {
            let ratio = cfg.compress_ratio(l).max(1);
            let bytes = ratio * hd * std::mem::size_of::<f32>();
            let cache = &self.layers[l];
            let base = (self.s.dspark_snap_state.ptr as *mut f32)
                .wrapping_add(l * 2 * max_ratio * hd);
            self.dev
                .memcpy_d2d(base as *mut c_void, cache.state_kv.ptr as *const c_void, bytes)?;
            self.dev.memcpy_d2d(
                base.wrapping_add(max_ratio * hd) as *mut c_void,
                cache.state_score.ptr as *const c_void,
                bytes,
            )?;
            self.dev.memcpy_d2d(
                (self.s.dspark_snap_latent.ptr as *mut f32).wrapping_add(l * hd) as *mut c_void,
                cache.latent.ptr as *const c_void,
                hd * std::mem::size_of::<f32>(),
            )?;
            self.dev.memcpy_d2d(
                (self.s.dspark_snap_clen.ptr as *mut i32).wrapping_add(l) as *mut c_void,
                (self.s.clen.ptr as *const i32).wrapping_add(l) as *const c_void,
                4,
            )?;
            self.dev.memcpy_d2d(
                (self.s.dspark_snap_out_rows.ptr as *mut i32).wrapping_add(l) as *mut c_void,
                cache.out_rows.ptr as *const c_void,
                4,
            )?;
            // the host mirror of the device counter, by the SAME rule the commit
            // kernel applies ((*pos_ctr + 1) % ratio == 0 commits one latent)
            host.push((l, cache.compress_len));
        }
        Ok(host)
    }

    /// Undo the verify block: copy the [`Self::dspark_snapshot`] save back over
    /// every buffer it recorded, and restore the host counters it returned.
    fn dspark_rollback(&mut self, pos: usize, m: usize, host: &[(usize, usize)]) -> Result<()> {
        // An empty mirror list means the caller skipped the snapshot (the
        // bisect modes that never run the verify): there is NOTHING to roll
        // back, and the snapshot buffers below were never written — restoring
        // from them would smear UNINITIALISED memory over the live ring/state/
        // clen and wedge the NEXT step (the exact failure the bisect run
        // caught: step 1 fine, step 2 dead).
        if host.is_empty() {
            return Ok(());
        }
        let cfg = self.cfg;
        let hd = cfg.head_dim;
        let win = cfg.window_size;
        let max_ratio = cfg.compress_ratios.iter().copied().max().unwrap_or(1).max(1);

        for &l in &self.ring_owners() {
            let ring = self.layers[l].ring.ptr as *mut f32;
            let snap = (self.s.dspark_snap_ring.ptr as *const f32)
                .wrapping_add(l * VERIFY_ROWS * hd);
            for j in 0..m {
                let slot = (pos + 1 + j) % win;
                self.dev.memcpy_d2d(
                    ring.wrapping_add(slot * hd) as *mut c_void,
                    snap.wrapping_add(j * hd) as *const c_void,
                    hd * std::mem::size_of::<f32>(),
                )?;
            }
        }

        for &l in &self.compress_sources() {
            let ratio = cfg.compress_ratio(l).max(1);
            let bytes = ratio * hd * std::mem::size_of::<f32>();
            let base = (self.s.dspark_snap_state.ptr as *const f32)
                .wrapping_add(l * 2 * max_ratio * hd);
            let cache = &self.layers[l];
            self.dev
                .memcpy_d2d(cache.state_kv.ptr, base as *const c_void, bytes)?;
            self.dev.memcpy_d2d(
                cache.state_score.ptr,
                base.wrapping_add(max_ratio * hd) as *const c_void,
                bytes,
            )?;
            self.dev.memcpy_d2d(
                cache.latent.ptr,
                (self.s.dspark_snap_latent.ptr as *const f32).wrapping_add(l * hd) as *const c_void,
                hd * std::mem::size_of::<f32>(),
            )?;
            self.dev.memcpy_d2d(
                (self.s.clen.ptr as *mut i32).wrapping_add(l) as *mut c_void,
                (self.s.dspark_snap_clen.ptr as *const i32).wrapping_add(l) as *const c_void,
                4,
            )?;
            self.dev.memcpy_d2d(
                cache.out_rows.ptr,
                (self.s.dspark_snap_out_rows.ptr as *const i32).wrapping_add(l) as *const c_void,
                4,
            )?;
        }
        for &(l, len) in host {
            self.layers[l].compress_len = len;
        }
        Ok(())
    }

    /// Shadow-mode DSpark step: run the draft and a real verify block for this
    /// step, then put the main chain back exactly where the single-row path would
    /// have left it, and report what the speculative path would have emitted.
    ///
    /// # Timeline
    ///
    /// ```text
    ///   1. snapshot     the verify's write set (`dspark_snapshot`)
    ///   2. next       = step_dev(token, pos)   the REAL step: the whole-step graph
    ///                                          (the tap hook is in it) + argmax
    ///   3. tap        -> dspark.import_tap(...) the target layers' h_mean, one D2D
    ///   4. drafts       dspark.draft_forward(token, pos), then the D2H of ids[1..=5]
    ///   5. verify_out = step_rows(&drafts)     five rows at pos + 1 .. pos + 5
    ///   6. rollback     the snapshot copied back
    ///   7. accept       host arithmetic, `DsparkShadowReport::accepted`
    /// ```
    ///
    /// `pos` is the step's position: the value of the device position counter
    /// BEFORE `step_dev` runs (`step_dev`'s own argument, and what `step_rows`
    /// reads back as `pos + 1`). The counter is read from the DEVICE here as well
    /// (one 4-byte D2H — the discipline `step_rows` already follows) and the
    /// ring-slot arithmetic uses that value, so the snapshot/rollback pair stays
    /// correct even if a caller's own bookkeeping drifts; `pos` is what goes to
    /// `step_dev`/`draft_forward`.
    ///
    /// # Accept arithmetic (the reference's rule, evaluated on the host)
    ///
    /// * `drafts[0]` is accepted iff it equals `next` — the anchor's argmax is
    ///   free, it is what the chain emits anyway.
    /// * `drafts[j]`, `j >= 1`, is accepted iff it equals `verify_out[j - 1]`: the
    ///   row fed `drafts[j - 1]` predicted it, so the checks are only meaningful
    ///   under the accepted prefix (hence the early `break`).
    /// * `accepted = 1 + <accepted draft prefix length>` (1..=DSPARK_DRAFTS + 1):
    ///   the accepted drafts plus the single bonus token the last surviving row
    ///   contributes — with nothing accepted, the emitted token is `next` alone.
    ///
    /// # What is NOT rolled back, on purpose
    ///
    /// * the DRAFT's own state (its window rings and caches): a shadow run is
    ///   meant to leave the draft where a real speculative run would, so the next
    ///   step's draft sees the same history.
    /// * `pos_ctr` and everything `step_dev` wrote: that is the REAL step, not the
    ///   verify. `step_rows` never advances the counter (its argmax gets a NULL).
    ///
    /// The function is a pure addition: no existing path calls it, so it cannot
    /// change today's decode. Wiring it into `serve`/`TpRankPool` is the caller's
    /// job (and a separate change).
    pub fn dspark_shadow_step(
        &mut self,
        dspark: &mut DsparkDev,
        token: u32,
        pos: usize,
    ) -> Result<DsparkShadowReport> {
        let cfg = self.cfg;
        let m = DSPARK_DRAFTS;
        // The tap hook only exists in the step graph when the gate was armed at
        // capture time. `dspark_armed()` caches the env in a OnceLock, so this is
        // the SAME decision `layer()` made — a process that armed the gate after
        // the first captured step would silently import a stale tap, which is why
        // the gate is a hard requirement here and not an optimisation.
        if !cfg.dspark_armed() {
            return Err(FerriteError::Config(
                "dspark_shadow_step: the chain's tap hook is off (set DSV41_DSPARK and make sure \
                 it is set before the first decoded step captures the step graph); without it the \
                 draft would read a stale target hidden"
                    .into(),
            ));
        }

        // The device counter's value BEFORE the step advances it decides which
        // ring slots the block will occupy.
        let pos_ctr = self.dev.download_u32(self.s.pos_ctr.ptr as *const c_void)? as usize;
        debug_assert_eq!(
            pos_ctr, pos,
            "dspark_shadow_step: `pos` must be the device position counter's current value"
        );

        // Bisect gate (DSV41_DSPARK_MODE): "draft" runs the draft only,
        // "verify" skips the draft and feeds a constant block, "step" runs the
        // armed step only (tap hook in-graph, no draft/verify) — for localising
        // a wedge. Default "full".
        static BISECT: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
        let bisect = *BISECT.get_or_init(|| {
            match std::env::var("DSV41_DSPARK_MODE").as_deref() {
                Ok("draft") => 1u8,
                Ok("verify") => 2u8,
                Ok("step") => 3u8,
                _ => 0u8,
            }
        });

        // 1. the snapshot, AFTER the real step and BEFORE the verify: the
        //    rollback must restore the state the REAL step left (the compressor
        //    commit it just made, the clen it just advanced), not the state from
        //    before it — restoring the pre-step state rolls the real step's
        //    commit back too, leaving clen one behind the truth every shadow
        //    step and corrupting the chain from step 2 on (the exact
        //    "step 1 fine, step 2 dead" bisect signature).
        // 2. the real step: the whole-step graph (tap hook included), the argmax,
        //    and the position counter + 1
        let next = self.step_dev(token, pos)?;
        let host_mirrors = if bisect == 1 || bisect == 3 {
            Vec::new()
        } else {
            self.dspark_snapshot(pos_ctr, m)?
        };

        // 3./4./5. the draft, from the tap of the step that just ran. `pos` is
        // the backbone token's position, the same convention `step_dev` uses and
        // the same one the host reference's `forward_spec(.., start_pos)` uses.
        let t = std::time::Instant::now();
        dspark.import_tap(self.s.dspark_tap.ptr as *const f32)?;
        let drafts = if bisect >= 2 {
            [token; DSPARK_DRAFTS]
        } else {
            dspark.draft_forward(token, pos)?;
            dspark.drafts()?
        };
        let draft_ms = t.elapsed().as_secs_f32() * 1e3;

        // 6. the verify block: one m-row forward, per-row argmax. It appends the
        //    block to the ring at `pos + 1 + j` and runs the block's compressor —
        //    all of which step 8 undoes.
        let t = std::time::Instant::now();
        let rows = if bisect == 1 || bisect == 3 {
            Vec::new()
        } else {
            self.step_rows(&drafts)?
        };
        let verify_ms = t.elapsed().as_secs_f32() * 1e3;

        // 7. rollback, immediately after the verify and BEFORE the host-only
        //    arithmetic below: the chain must not stay dirty on a report-building
        //    error path.
        self.dspark_rollback(pos_ctr, m, &host_mirrors)?;

        let mut verify_out = [0u32; DSPARK_DRAFTS];
        if bisect == 0 && rows.len() != DSPARK_DRAFTS {
            return Err(FerriteError::Config(format!(
                "dspark_shadow_step: step_rows returned {} rows for a {DSPARK_DRAFTS}-row \
                 verify block",
                rows.len()
            )));
        }
        if rows.len() == DSPARK_DRAFTS {
            verify_out.copy_from_slice(&rows);
        }

        // 8. the accept arithmetic (host, no device traffic).
        let mut acc = 0usize;
        if drafts[0] == next {
            acc = 1;
            for j in 1..DSPARK_DRAFTS {
                if drafts[j] == verify_out[j - 1] {
                    acc += 1;
                } else {
                    break;
                }
            }
        }

        Ok(DsparkShadowReport {
            next,
            drafts,
            verify_out,
            accepted: acc + 1,
            draft_ms,
            verify_ms,
        })
    }

    /// The m-row premix slots, mirroring [`Self::premix_slot`]'s convention: 0 is
    /// the incoming block (the constant `[1,0,0,0]` for a verify pass), 1 is this
    /// layer's attn_pre (written by the attention mixes, read by the FFN collapse)
    /// and 2 receives this layer's ffn_pre, i.e. the next layer's incoming premix.
    fn premix_row_slot(&self, i: usize) -> &DevBuf {
        match i % 3 {
            0 => &self.s.premix_r,
            2 => &self.s.pre2_r,
            _ => &self.s.pre_r,
        }
    }

    /// One multi-row block: the m-row twin of [`Self::layer`].
    ///
    /// The premix threading is `layer()`'s, one step up in width: `pa` indexes the
    /// m-row premix the attention collapse reads, the attention mixes land in
    /// `pre_r` (slot 1) which the FFN collapse then consumes, and the FFN mixes
    /// land in `pre2_r` (slot 2) — the next layer's incoming premix. Returns 2.
    ///
    /// `hc_post` is used in its staging form (`hc_post` + a copy back) rather than
    /// the single-row `FUSE_C` in-place form: the in-place variant is a one-row
    /// specialisation, and the staging form is the pair it was verified against.
    fn layer_rows(&mut self, layer: usize, m: usize, pos_base: i32, pa: usize) -> Result<usize> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let ld = &self.w.layers[layer];
        let hbytes = m * hc * dim * std::mem::size_of::<f32>();

        // ---------------- attention block ----------------
        // hc_mixes has a native `rows` dimension (`rows = m`, `hc_dim = hc*dim`),
        // with pre/post/comb row-major per row — the same shape the single-row call
        // uses with rows = 1.
        self.dev.hc_mixes(
            self.s.h_r.ptr as *const f32,
            ld.hc_attn_fn.as_ref().unwrap().as_f32(),
            ld.hc_attn_scale.as_ref().unwrap().as_f32(),
            ld.hc_attn_base.as_ref().unwrap().as_f32(),
            self.s.pre_r.ptr as *mut f32,  // slot 1: this layer's attn_pre
            self.s.post_r.ptr as *mut f32,
            self.s.comb_r.ptr as *mut f32,
            m as i32,
            (hc * dim) as i32,
            hc as i32,
            cfg.hc_sinkhorn_iters as i32,
            cfg.hc_eps,
        )?;
        self.dev.hc_collapse(
            self.s.h_r.ptr as *const f32,
            self.premix_row_slot(pa).as_f32(),
            self.s.x_r.ptr as *mut f32,
            m as i32,
            hc as i32,
            dim as i32,
        )?;
        self.dev.rmsnorm(
            self.s.x_r.ptr as *const f32,
            ld.attn_norm.as_ref().unwrap().as_f32(),
            self.s.xn_r.ptr as *mut f32,
            m as i32,
            dim as i32,
            cfg.norm_eps,
        )?;
        self.attention_rows(layer, m, pos_base)?;
        self.dev.hc_post(
            self.s.wo_out_r.ptr as *const f32,
            self.s.h_r.ptr as *const f32,
            self.s.post_r.as_f32(),
            self.s.comb_r.as_f32(),
            self.s.h2_r.ptr as *mut f32,
            m as i32,
            hc as i32,
            dim as i32,
        )?;
        self.dev
            .memcpy_d2d(self.s.h_r.ptr, self.s.h2_r.ptr as *const c_void, hbytes)?;

        // ---------------- FFN block ----------------
        self.dev.hc_mixes(
            self.s.h_r.ptr as *const f32,
            ld.hc_ffn_fn.as_ref().unwrap().as_f32(),
            ld.hc_ffn_scale.as_ref().unwrap().as_f32(),
            ld.hc_ffn_base.as_ref().unwrap().as_f32(),
            self.s.pre2_r.ptr as *mut f32, // slot 2: this layer's ffn_pre -> next layer
            self.s.post_r.ptr as *mut f32,
            self.s.comb_r.ptr as *mut f32,
            m as i32,
            (hc * dim) as i32,
            hc as i32,
            cfg.hc_sinkhorn_iters as i32,
            cfg.hc_eps,
        )?;
        // the FFN collapses with THIS layer's attn_pre (slot 1), exactly as `layer()`
        self.dev.hc_collapse(
            self.s.h_r.ptr as *const f32,
            self.s.pre_r.as_f32(),
            self.s.x_r.ptr as *mut f32,
            m as i32,
            hc as i32,
            dim as i32,
        )?;
        self.dev.rmsnorm(
            self.s.x_r.ptr as *const f32,
            ld.ffn_norm.as_ref().unwrap().as_f32(),
            self.s.xn_r.ptr as *mut f32,
            m as i32,
            dim as i32,
            cfg.norm_eps,
        )?;
        self.moe_rows(layer, ld, m)?;
        self.dev.hc_post(
            self.s.moe_out_r.ptr as *const f32,
            self.s.h_r.ptr as *const f32,
            self.s.post_r.as_f32(),
            self.s.comb_r.as_f32(),
            self.s.h2_r.ptr as *mut f32,
            m as i32,
            hc as i32,
            dim as i32,
        )?;
        self.dev
            .memcpy_d2d(self.s.h_r.ptr, self.s.h2_r.ptr as *const c_void, hbytes)?;
        Ok(2)
    }

    /// Multi-row attention: the m-row twin of [`Self::attention`].
    ///
    /// Every row goes through the pair-call form (`lin`, `gemm_fp8_mx_or_swap`,
    /// `apply_rope`, `sparse_attn`, `indexer_topk`) that the single-row fusions are
    /// verified against — see the section comment above for why. The two genuinely
    /// multi-row steps are `verify_ring_win` (one launch appends the whole block to
    /// the window ring and emits the per-row causal window) and the wo all-reduce,
    /// whose payload is simply m rows.
    fn attention_rows(&mut self, layer: usize, m: usize, pos_base: i32) -> Result<()> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let hd = cfg.head_dim;
        let nh = cfg.n_heads;
        let ql = cfg.q_lora_rank;
        let rd = cfg.rope_head_dim;
        let half = (cfg.rope_head_dim / 2) as i32;
        let win = cfg.window_size;
        let world = self.world();
        let nlh = nh / world;
        let ld = &self.w.layers[layer];
        let pos_ctr = self.s.pos_ctr.ptr as *const std::os::raw::c_int;

        // ---- q / kv projections, one row per call ----
        // `lin()` is the (quant1, gemm_fp8_mx_or_swap) pair: the plain shape the
        // fused `lin2`/`lin_rope*` launches are bit-identical to.
        for r in 0..m {
            let xr = (self.s.xn_r.ptr as *const f32).wrapping_add(r * dim);
            let qrr = (self.s.qr_r.ptr as *mut f32).wrapping_add(r * ql);
            let kvr = (self.s.kv_r.ptr as *mut f32).wrapping_add(r * hd);
            self.lin(
                xr,
                dim as i32,
                ld.wq_a.as_ref().unwrap(),
                ld.wq_a_scale.as_ref().unwrap(),
                ql as i32,
                qrr,
            )?;
            self.lin(
                xr,
                dim as i32,
                ld.wkv.as_ref().unwrap(),
                ld.wkv_scale.as_ref().unwrap(),
                hd as i32,
                kvr,
            )?;
            // q norm, in place (the T2 epilogue's own fallback pair: plain rmsnorm
            // into `qr`)
            self.dev.rmsnorm(
                qrr as *const f32,
                ld.q_norm.as_ref().unwrap().as_f32(),
                qrr,
                1,
                ql as i32,
                cfg.norm_eps,
            )?;
            // wq_b: this rank's `nlh` heads, written at the row's base
            self.lin(
                qrr as *const f32,
                ql as i32,
                ld.wq_b.as_ref().unwrap(),
                ld.wq_b_scale.as_ref().unwrap(),
                (nlh * hd) as i32,
                (self.s.q_r.ptr as *mut f32).wrapping_add(r * nh * hd),
            )?;
        }
        // ---- RoPE ----
        // The kernel computes `pos = *base * mul + off + row * step` for row `row`.
        // All `nlh` heads of one verify row share that row's position, so the q
        // rope rides the position in `off` with `step = 0` (exactly how the
        // single-row call keeps its heads at one position); the KV rope has one row
        // per position, so the block form works directly with `step = 1`.
        for r in 0..m {
            self.dev.apply_rope(
                (self.s.q_r.ptr as *mut f32).wrapping_add(r * nh * hd),
                self.cos.as_f32(),
                self.sin.as_f32(),
                nlh as i32,
                hd as i32,
                rd as i32,
                half,
                pos_ctr,
                1,
                r as i32,
                0,
                false,
            )?;
        }
        self.dev.rmsnorm(
            self.s.kv_r.ptr as *const f32,
            ld.kv_norm.as_ref().unwrap().as_f32(),
            self.s.kv_r.ptr as *mut f32,
            m as i32,
            hd as i32,
            cfg.norm_eps,
        )?;
        self.dev.apply_rope(
            self.s.kv_r.ptr as *mut f32,
            self.cos.as_f32(),
            self.sin.as_f32(),
            m as i32,
            hd as i32,
            rd as i32,
            half,
            pos_ctr,
            1,
            0,
            1,
            false,
        )?;

        // ---- window ring + the per-row causal window ----
        // The release shares one KV store across a group of layers; a consumer then
        // reads its owner's ring and must not append to it again (the same rule the
        // single-row `ring_append` follows).
        let owner = if ring_owner_shared() { self.kv_owner(layer) } else { layer };
        let owns_kv = owner == layer;
        let ring_ptr = self.layers[owner].ring.ptr;
        let ist = win + cfg.index_topk;
        if owns_kv {
            // PER-ROW append + PER-ROW causal window, interleaved: row r's
            // `window_idxs` must run after row r's append (its own KV is in the
            // window) but BEFORE row r+1's append (which would overwrite the
            // oldest slot that row r's window still enumerates — reading the
            // block's future row as history instead). The fused verify_ring_win
            // appended the whole block first and derived the indices from slot
            // numbers, which broke exactly there: once base+r >= window the
            // `v > start_pos` filter never fires (v is a SLOT, not a position)
            // and every row but the last read the block's own future rows while
            // losing the oldest m-1-r history — the "verify outputs garbage"
            // root cause. The single-row kernels keep the ring invariant row by
            // row, so the window is byte-identical to a single-row decode at
            // each row's position.
            for r in 0..m {
                self.dev.ring_append(
                    ring_ptr as *mut f32,
                    (self.s.kv_r.ptr as *const f32).wrapping_add(r * hd),
                    (self.s.pos_rows.ptr as *const std::os::raw::c_int).wrapping_add(r),
                    win as i32,
                    hd as i32,
                )?;
                self.dev.window_idxs(
                    (self.s.idxs_r.ptr as *mut i32).wrapping_add(r * ist),
                    (self.s.pos_rows.ptr as *const std::os::raw::c_int).wrapping_add(r),
                    win as i32,
                )?;
            }
        }
        // A consumer layer's window block is the owner's (the indices depend
        // only on the positions and the ring geometry, and the owner filled
        // idxs_r interleaved with its appends) — no append, no recompute.

        // ---- compressor + the compressed-half selection ----
        let comp_len = if cfg.compress_ratio(layer) > 0 && cfg.is_kv_source(layer) {
            self.compress_rows(layer, m, pos_base)?
        } else if cfg.compress_ratio(layer) > 0 {
            // a consumer inherits the count published by its source layer
            self.source_compress_len(layer)
        } else {
            0
        };
        if comp_len > 0 && cfg.is_index_source(layer) {
            self.indexer_rows(layer, m, win, comp_len)?;
        } else if !owns_kv && comp_len > 0 {
            // a non-index consumer reads the owner's selection, which the owner (an
            // index source, running earlier in the stack) already wrote into this
            // step's `idxs_r` compressed block
        } else if comp_len > 0 {
            // the owner has no indexer: the recency placeholder, per row
            for r in 0..m {
                self.dev.comp_placeholder(
                    (self.s.idxs_r.ptr as *mut i32).wrapping_add(r * ist + win),
                    (self.s.clen.ptr as *const std::os::raw::c_int).wrapping_add(owner),
                    win as i32,
                    cfg.index_topk as i32,
                )?;
            }
        }

        // ---- sparse attention + the inverse o-rope, one row per call ----
        let scale = 1.0f32 / (hd as f32).sqrt();
        for r in 0..m {
            self.dev.sparse_attn(
                (self.s.q_r.ptr as *const f32).wrapping_add(r * nh * hd),
                ring_ptr as *const f32,
                ld.attn_sink.as_ref().unwrap().as_f32(),
                (self.s.idxs_r.ptr as *const i32).wrapping_add(r * ist),
                (self.s.o_r.ptr as *mut f32).wrapping_add(r * nh * hd),
                1,
                1,
                nlh as i32,
                hd as i32,
                (self.s.clen.ptr as *const std::os::raw::c_int).wrapping_add(owner),
                win as i32,
                cfg.index_topk as i32,
                scale,
            )?;
        }
        for r in 0..m {
            self.dev.apply_rope(
                (self.s.o_r.ptr as *mut f32).wrapping_add(r * nh * hd),
                self.cos.as_f32(),
                self.sin.as_f32(),
                nlh as i32,
                hd as i32,
                rd as i32,
                half,
                pos_ctr,
                1,
                r as i32,
                0,
                true,
            )?;
        }

        // ---- grouped output projection + wo_b, one row per call ----
        let groups = cfg.o_groups;
        let hpg = nh / groups;
        let olg = cfg.o_lora_rank;
        let nlg = groups / world;
        let k = hpg * hd;
        let ol_total = groups * cfg.o_lora_rank;
        let ol_local = ol_total / world;
        for r in 0..m {
            // the fp8 of this row's attention output (the OROPE_Q epilogue's own
            // fallback), shared by all of the row's group blocks
            self.quant1(
                (self.s.o_r.ptr as *const f32).wrapping_add(r * nh * hd),
                (nlh * hd) as i32,
            )?;
            for g in 0..nlg {
                // The weight tensor is ALREADY this rank's local slice, so every
                // offset is local (see `attention`'s comment on the tp=8 bug).
                let a = self.s.xq.as_u8().wrapping_add(g * k);
                let asc = self.s.xsc.as_f32().wrapping_add((g * k / 32) as usize);
                let wp = ld.wo_a.as_ref().unwrap().as_u8().wrapping_add(g * olg * k);
                let wsp = ld
                    .wo_a_scale
                    .as_ref()
                    .unwrap()
                    .as_u8()
                    .wrapping_add((g * olg / 32) * (k / 32));
                let out = (self.s.wo_r.ptr as *mut f32).wrapping_add(r * ol_total + g * olg);
                self.gemm_fp8_mx_or_swap(
                    a,
                    asc,
                    wp,
                    wsp,
                    std::ptr::null(),
                    out,
                    olg as i32,
                    k as i32,
                )?;
            }
            // wo_b is RowParallel: the input is split over the ranks, so this rank
            // writes its own partial and the AR below sums them.
            self.quant1(
                (self.s.wo_r.ptr as *const f32).wrapping_add(r * ol_total),
                ol_local as i32,
            )?;
            self.gemm_fp8_mx_or_swap(
                self.s.xq.as_u8(),
                self.s.xsc.as_f32(),
                ld.wo_b.as_ref().unwrap().as_u8(),
                ld.wo_b_scale.as_ref().unwrap().as_u8(),
                std::ptr::null(),
                (self.s.wo_out_r.ptr as *mut f32).wrapping_add(r * dim),
                dim as i32,
                ol_local as i32,
            )?;
        }
        // ---- the attention all-reduce ----
        // The payload is the m rows of `wo_out_r` instead of one: the AR is a
        // byte-wise operation that sums the ranks in ascending order, so each row's
        // per-element sum is the same values in the same order as the single-row AR
        // of that row would have been. The hc-post fold (`HCPOST_EPI`) is a
        // single-row epilogue and is not taken; the standalone `hc_post` in
        // `layer_rows` is the pair it was verified against.
        if let Some(c) = self.comm.clone() {
            c.all_reduce_inplace(self.s.wo_out_r.ptr as *mut std::ffi::c_void, fb(m * dim))?;
            c.end_round();
        }
        Ok(())
    }

    /// Multi-row indexer: the m-row twin of [`Self::indexer`].
    ///
    /// The key PUBLISHING stays once per layer (it is a function of the compressor's
    /// latent, of which the multi-row compressor produces one row — see the
    /// compressor TODO); the queries, per-head weights and the top-k launch run per
    /// row, with each row's picks landing in that row's `idxs_r` block at
    /// `+offset`.
    ///
    /// `indexer_topk`'s output stride is `cols = min(topk, n_pos)` — a RUNTIME value
    /// derived from the live compressed count — so the kernel cannot be handed the
    /// `window + index_topk` row stride `idxs_r` uses. It is therefore called once
    /// per row with `m = 1`, where the stride never matters.
    fn indexer_rows(
        &mut self,
        layer: usize,
        m: usize,
        offset: usize,
        comp_len: usize,
    ) -> Result<bool> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let ql = cfg.q_lora_rank;
        let idx_nh = cfg.index_n_heads.max(1);
        let idx_hd = cfg.index_head_dim.max(1);
        let rd = cfg.rope_head_dim;
        let half = (cfg.rope_head_dim / 2) as i32;
        let ratio = cfg.compress_ratio(layer).max(1);
        let ld = &self.w.layers[layer];
        let (Some(idx_wq_b), Some(idx_wq_b_s), Some(wk), Some(kn), Some(wp)) = (
            ld.idx_wq_b.as_ref(),
            ld.idx_wq_b_scale.as_ref(),
            ld.idx_wk.as_ref(),
            ld.idx_k_norm.as_ref(),
            ld.idx_weights.as_ref(),
        ) else {
            return Ok(false);
        };
        // ---- key publishing (kv sources only, once per layer) ----
        // `indexer_owns_k` decides whether this layer publishes the index key for
        // the group the compressor produced; the roped key lands in the owner's
        // `index_k` at the slot its DEVICE counter names.
        if cfg.indexer_owns_k(layer) {
            let clen_layer = (self.s.clen.ptr as *const std::os::raw::c_int).wrapping_add(layer);
            self.lin_bf16(
                self.layers[layer].latent.ptr as *const f32,
                cfg.head_dim as i32,
                wk,
                idx_hd as i32,
                self.s.idx_k.ptr as *mut f32,
            )?;
            self.dev.rmsnorm(
                self.s.idx_k.ptr as *const f32,
                kn.as_f32(),
                self.s.idx_k.ptr as *mut f32,
                1,
                idx_hd as i32,
                cfg.norm_eps,
            )?;
            self.dev.apply_rope(
                self.s.idx_k.ptr as *mut f32,
                self.cos.as_f32(),
                self.sin.as_f32(),
                1,
                idx_hd as i32,
                rd as i32,
                (rd / 2) as i32,
                clen_layer,
                ratio as i32,
                -(ratio as i32),
                1,
                false,
            )?;
            self.dev.index_k_publish(
                self.layers[layer].index_k.ptr as *mut f32,
                self.s.idx_k.ptr as *const f32,
                clen_layer,
                idx_hd as i32,
            )?;
        }
        // ---- per-row queries, weights and selection ----
        // The q_lora stream is already normed (attention_rows ran the plain
        // rmsnorm), so `idx_wq_b` uses the plain `lin`/`lin_bf16` pair.
        for r in 0..m {
            let idxq = (self.s.idx_q_r.ptr as *mut f32).wrapping_add(r * idx_nh * idx_hd);
            self.lin(
                (self.s.qr_r.ptr as *const f32).wrapping_add(r * ql),
                ql as i32,
                idx_wq_b,
                idx_wq_b_s,
                (idx_nh * idx_hd) as i32,
                idxq,
            )?;
            self.dev.apply_rope(
                idxq,
                self.cos.as_f32(),
                self.sin.as_f32(),
                idx_nh as i32,
                idx_hd as i32,
                rd as i32,
                half,
                self.s.pos_ctr.ptr as *const std::os::raw::c_int,
                1,
                r as i32,
                0,
                false,
            )?;
            self.lin_bf16(
                (self.s.xn_r.ptr as *const f32).wrapping_add(r * dim),
                dim as i32,
                wp,
                idx_nh as i32,
                (self.s.idx_w_r.ptr as *mut f32).wrapping_add(r * idx_nh),
            )?;
        }
        let scale = 1.0f32 / (idx_hd as f32).sqrt() / (idx_nh as f32).sqrt();
        let key_owner = self.kv_owner(layer);
        let idx_lens =
            (self.s.clen.ptr as *const std::os::raw::c_int).wrapping_add(key_owner);
        let idx_k_ptr = self.layers[key_owner].index_k.ptr as *const f32;
        for r in 0..m {
            self.dev.indexer_topk(
                (self.s.idx_q_r.ptr as *const f32).wrapping_add(r * idx_nh * idx_hd),
                idx_k_ptr,
                (self.s.idx_w_r.ptr as *const f32).wrapping_add(r * idx_nh),
                std::ptr::null(),
                idx_lens,
                (self.s.idxs_r.ptr as *mut i32).wrapping_add(r * (offset + cfg.index_topk) + offset),
                1,
                1,
                idx_nh as i32,
                idx_hd as i32,
                comp_len as i32,
                cfg.index_topk as i32,
                offset as i32,
                scale,
                1.0,
                false,
            )?;
        }
        Ok(true)
    }

    /// Multi-row compressor: the m-row twin of [`Self::compress_on`].
    ///
    /// The projections are per row (the same `lin_f32` call, repeated), and the
    /// state/pool/commit trio is called with `seqlen = m`. `compressor_fused` is
    /// NOT used: its launcher rejects `b != 1 || seqlen != 1` outright. The
    /// pool+commit pair is the path the fused launch is bit-identical to, so this
    /// is a parity target and not a different program.
    ///
    /// ⚠️ KNOWN GAP (documented in `step_rows`): the state kernel's decode branch
    /// and the mode-2 pool carry ONE row each, so this models a single completed
    /// group per block. The host mirror is advanced by the same rule the commit
    /// kernel applies on the device, which keeps the two consistent for that one
    /// group.
    fn compress_rows(&mut self, layer: usize, m: usize, pos_base: i32) -> Result<usize> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let hd = cfg.head_dim;
        let ratio = cfg.compress_ratio(layer).max(1);
        let ld = &self.w.layers[layer];
        let (Some(wkv), Some(norm)) = (ld.comp_wkv.as_ref(), ld.comp_norm.as_ref()) else {
            return Ok(self.layers[layer].compress_len);
        };
        let st = self.dev.stream();
        for r in 0..m {
            let xr = (self.s.xn_r.ptr as *const f32).wrapping_add(r * dim);
            self.lin_f32_on(
                xr,
                dim as i32,
                wkv,
                hd as i32,
                (self.s.kvp_r.ptr as *mut f32).wrapping_add(r * hd),
                st,
            )?;
            if let Some(wg) = ld.comp_wgate.as_ref() {
                self.lin_f32_on(
                    xr,
                    dim as i32,
                    wg,
                    hd as i32,
                    (self.s.scp_r.ptr as *mut f32).wrapping_add(r * hd),
                    st,
                )?;
            }
        }
        if ld.comp_wgate.is_none() {
            // ratio == 1: no gate, so the pooled value IS the projection. The
            // single-row path zeroes `scp` here; in that mode every row is pooled
            // independently, so all m rows are zeroed.
            self.dev.zero_on(&self.s.scp_r, st)?;
        }
        // PER-ROW single-token pool+commit: the pool's state decode branch and
        // the commit's completion rule are per-POSITION. Running them once with
        // seqlen=m only consumed ROW 0 — rows 1..m-1's kvp/scp never entered
        // the state and the groups they completed never existed (the audit's
        // defect #2; the operator-visible symptom was "the latent the draft
        // sees never updates"). Each row now runs the SAME triple the
        // single-row decode runs, at its own position (pos_rows[r]), so the
        // compressed state after the block is exactly what m sequential
        // single-row steps would leave.
        for r in 0..m {
            self.dev.compressor_pool_on(
                (self.s.kvp_r.ptr as *const f32).wrapping_add(r * hd),
                (self.s.scp_r.ptr as *const f32).wrapping_add(r * hd),
                norm.as_f32(),
                self.layers[layer].state_kv.ptr as *mut f32,
                self.layers[layer].state_score.ptr as *mut f32,
                self.layers[layer].latent.ptr as *mut f32,
                self.layers[layer].out_rows.ptr as *mut i32,
                1,
                1,
                hd as i32,
                ratio as i32,
                pos_base + r as i32,
                (self.s.pos_rows.ptr as *const std::os::raw::c_int).wrapping_add(r),
                cfg.norm_eps,
                st,
            )?;
            self.dev.compress_commit_on(
                self.layers[layer].latent.as_f32(),
                self.cos_comp.as_f32(),
                self.sin_comp.as_f32(),
                self.layers[layer].ring.ptr as *mut f32,
                self.layers[layer].out_rows.ptr as *const std::os::raw::c_int,
                (self.s.clen.ptr as *mut std::os::raw::c_int).wrapping_add(layer),
                hd as i32,
                cfg.rope_head_dim as i32,
                (cfg.rope_head_dim / 2) as i32,
                cfg.window_size as i32,
                ratio as i32,
                st,
            )?;
            // The host MIRROR of the device counter, by the SAME deterministic
            // rule the commit kernel applies ((*pos + 1) % ratio == 0 commits
            // one latent) — per row.
            if (pos_base + r as i32 + 1) % (ratio as i32) == 0 {
                self.layers[layer].compress_len += 1;
            }
        }
        Ok(self.layers[layer].compress_len)
    }

    /// Multi-row engram write-back: the m-row twin of [`Self::engram_apply`].
    ///
    /// The gather runs once per row (the hash ids are per token); the collective is
    /// one m-row call (byte-wise, ascending rank order, so each row's part is
    /// identical to the single-row AR of that row); the projection and the gated
    /// write-back use their native `rows` dimension.
    fn engram_apply_rows(&mut self, layer: usize, li: usize, m: usize) -> Result<()> {
        let cfg = self.cfg;
        let rank = self.rank();
        let (dim, hc, ehd) = (cfg.dim, cfg.hc_mult, cfg.engram_head_dim);
        let n_cols = cfg.engram_max_ngram_size.saturating_sub(1) * cfg.engram_n_heads;
        let (table, tsc, wkv, wsc, qw, kw) = {
            let ld = &self.w.layers[layer];
            (
                ld.engram_embed.as_ref(),
                ld.engram_embed_scale.as_ref(),
                ld.engram_wkv.as_ref(),
                ld.engram_wkv_scale.as_ref(),
                ld.engram_q_weight.as_ref(),
                ld.engram_k_weight.as_ref(),
            )
        };
        let (Some(table), Some(tsc), Some(wkv), Some(wsc), Some(qw), Some(kw)) =
            (table, tsc, wkv, wsc, qw, kw)
        else {
            return Ok(());
        };
        // This rank's slice of the row-parallel table: convert.py shards
        // `ceil(rows / world)` rows and zero-pads the tail.
        let world = self.world().max(1);
        let global_rows = cfg.engram_num_embeddings.get(li).copied().unwrap_or(0) as usize;
        let per = global_rows.div_ceil(world);
        // `eng_ids_r` is [row][engram layer][column], the same per-token layer count
        // the hash wrote above.
        let n_eng = self
            .eng_layout
            .as_ref()
            .map(|l| l.layers.len())
            .unwrap_or(1)
            .max(1);
        for r in 0..m {
            let ids = (self.s.eng_ids_r.ptr as *const i64).wrapping_add((r * n_eng + li) * n_cols);
            self.dev.engram_gather(
                table.ptr() as *const u8,
                tsc.ptr() as *const u8,
                ids,
                (self.s.eng_rows_r.ptr as *mut f32).wrapping_add(r * n_cols * ehd),
                1,
                n_cols as i32,
                ehd as i32,
                (rank * per) as i64,
                per as i64,
            )?;
        }
        // rows another rank owns arrived as 0, so the sum yields the real row
        if let Some(c) = self.comm.clone() {
            c.all_reduce_inplace(
                self.s.eng_rows_r.ptr as *mut std::ffi::c_void,
                fb(m * n_cols * ehd),
            )?;
        }
        // kv = wkv(gathered): [(hc + 1) * dim] per row. The f32-direct-read and
        // swapAB variants are single-row optimisations of this (quantise, GEMM)
        // pair, which is the numerically canonical one.
        self.dev.quant_fp8(
            self.s.eng_rows_r.ptr as *const f32,
            self.s.eng_xq_r.ptr as *mut u8,
            self.s.eng_xsc_r.ptr as *mut f32,
            m as i32,
            (n_cols * ehd) as i32,
            32,
            true,
        )?;
        self.dev.gemm_fp8_mx(
            self.s.eng_xq_r.as_u8(),
            self.s.eng_xsc_r.as_f32(),
            wkv.ptr() as *const u8,
            wsc.ptr() as *const u8,
            std::ptr::null(),
            self.s.eng_kv_r.ptr as *mut f32,
            m as i32,
            ((hc + 1) * dim) as i32,
            (n_cols * ehd) as i32,
        )?;
        // gated write-back into the residual stream (in place), all m rows
        self.dev.engram_apply(
            self.s.h_r.ptr as *mut f32,
            self.s.eng_kv_r.ptr as *const f32,
            qw.ptr() as *const f32,
            kw.ptr() as *const f32,
            std::ptr::null(),
            m as i32,
            hc as i32,
            dim as i32,
            cfg.norm_eps,
        )?;
        Ok(())
    }

    /// Multi-row MoE: the m-row twin of [`Self::moe`].
    ///
    /// # The batched-expert launchers are SINGLE-ROW
    ///
    /// `dsv41_expert_gate_up_fp4_batched` / `dsv41_expert_down_fp4_batched` /
    /// `dsv41_expert_down_reduce_fp4_batched` take a `rows` argument, but it never
    /// reaches the grid: the grid is `((n_total + warps - 1)/warps, slots)` where
    /// `n_total` is the OUTPUT width (`inter` / `2*inter` / `dim`) and the slot loop
    /// is `grid.y` — i.e. each call computes ONE activation row against `slots`
    /// experts. (`dspark_dev.rs` hands them `rows = bs`, which for `bs > 1` silently
    /// computes row 0 only.) The routed half is therefore issued PER ROW, with that
    /// row's fp4-packed activation, its `ids`/`route_w` slice and its own output
    /// slot block; the gate/up output is laid out `[row][slot][act_slot]`.
    ///
    /// The routing itself IS multi-row: the gate is the same M=1 bf16 GEMV `moe()`
    /// runs (once per row), and `route_topk` has a native `rows` dimension.
    fn moe_rows(&mut self, layer: usize, ld: &LayerDev, m: usize) -> Result<()> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let inter = cfg.moe_inter_dim;
        let inter_local = crate::dsv41::weights::padded_inter(inter / self.world());
        let (n_routed, topk) = cfg.moe_config(layer);
        let topk = topk.max(1);
        let n_routed = n_routed.max(1);
        let stp = crate::dsv41::weights::shared_expert_tp();
        let sh_il = if stp { inter / self.world() } else { inter };
        let shared_rank = if stp {
            true
        } else {
            self.comm.as_ref().map(|c| c.rank == 0).unwrap_or(true)
        };
        let mdim = m * dim;

        // ---- gate + route ----
        // The gate runs on EVERY path (exactly as `moe()` does: `skip_experts` only
        // skips the expert launches); the routing is genuinely multi-row through
        // `route_topk`'s `rows` dimension.
        let gate_bias = ld
            .gate_bias
            .as_ref()
            .map(|b| b.as_f32())
            .unwrap_or(std::ptr::null());
        for r in 0..m {
            self.dev.gemv_bf16(
                ld.gate_w.as_ref().unwrap().ptr() as *const c_void,
                (self.s.xn_r.ptr as *const f32).wrapping_add(r * dim),
                (self.s.scores_r.ptr as *mut f32).wrapping_add(r * n_routed),
                n_routed as i32,
                dim as i32,
            )?;
        }
        self.dev.route_topk(
            self.s.scores_r.as_f32(),
            gate_bias,
            self.s.route_w_r.ptr as *mut f32,
            self.s.route_idx_r.ptr as *mut i32,
            std::ptr::null_mut(),
            m as i32,
            n_routed as i32,
            topk as i32,
            cfg.norm_topk_prob,
            cfg.route_scale,
            2, // sqrtsoftplus, per the checkpoint's routing
        )?;

        // The MoE accumulator: the routed half OVERWRITES it (the fused down+reduce
        // writes `out[i] = acc`, exactly like `moe_reduce`), so with no routed half
        // it has to start at zero.
        if self.opts.skip_experts {
            self.dev.zero(&self.s.moe_out_r)?;
        }

        // ---- routed experts, one activation row per launch ----
        if !self.opts.skip_experts {
            let ne = ld.experts.len();
            if ne < 2 {
                return Err(FerriteError::Config(format!(
                    "moe_rows: layer {layer} has {ne} expert tensors; the indirect \
                     per-slot weight scheme needs at least 2"
                )));
            }
            // The interleaved layout is only addressable by the FUSED batched
            // gate/up body, so refuse loudly instead of reading the wrong bytes
            // (the same contract `moe()` enforces).
            if ld.experts_ilv
                && !(gateup_fuse() && self.dev.supports_gateup_fuse() && expert_fp4_mode() == 2)
            {
                return Err(FerriteError::Config(
                    "routed expert gate/up weights are interleaved (DSV41_EXPERT_ILV) but the \
                     fused batched gate/up path is unavailable — run with DSV41_EXPERT_ILV=0, \
                     or restore DSV41_GATEUP_FUSE / DSV41_EXPERT_FP4_MODE=2 so the fused \
                     batched call is used"
                        .into(),
                ));
            }
            // The expert tensors are views into one per-layer pool with a uniform
            // stride, so the kernels derive every pointer from a base plus
            // `ids[slot] * stride` (the same derivation `moe()` uses).
            let (a, b) = (&ld.experts[0], &ld.experts[1]);
            let d = |x: *mut std::ffi::c_void, y: *mut std::ffi::c_void| (y as i64) - (x as i64);
            let (w1_base, w1_stride, w1s_base, w1s_stride, w3_base, w3_stride, w3s_base, w3s_stride) = (
                a.w1.ptr() as *const u8,
                d(a.w1.ptr(), b.w1.ptr()),
                a.w1_scale.ptr() as *const u8,
                d(a.w1_scale.ptr(), b.w1_scale.ptr()),
                a.w3.ptr() as *const u8,
                d(a.w3.ptr(), b.w3.ptr()),
                a.w3_scale.ptr() as *const u8,
                d(a.w3_scale.ptr(), b.w3_scale.ptr()),
            );
            let (w2_base, w2_stride, w2s_base, w2s_stride) = (
                a.w2.ptr() as *const u8,
                d(a.w2.ptr(), b.w2.ptr()),
                a.w2_scale.ptr() as *const u8,
                d(a.w2_scale.ptr(), b.w2_scale.ptr()),
            );
            // The gate/up fusion decision mirrors the launcher's own `fuse` test
            // (and therefore `moe()`'s expression, plus the `dim % 512 == 0` term
            // the launcher applies but the single-row call site omits). It changes
            // the slot pitch: the fused epilogue writes the swiglu'd inter-width
            // slice, so `act_slot` shrinks from `2*inter` to `inter`.
            let gateup_fused = gateup_fuse()
                && self.dev.supports_gateup_fuse()
                && expert_fp4_mode() == 2
                && (dim % 512) == 0;
            let act_slot = if gateup_fused {
                inter_local
            } else {
                2 * inter_local
            };
            let row_pitch = topk * act_slot; // usize: per-row pitch in ex_act_r
            // One fp4 packing covers all m rows (rows = m is native to the
            // quantiser); each gate/up call then reads its own row's packed bytes.
            self.dev.quant_fp4(
                self.s.xn_r.ptr as *const f32,
                self.s.xq4_r.ptr as *mut u8,
                self.s.xsc4_r.ptr as *mut f32,
                m as i32,
                dim as i32,
                32,
                true,
            )?;
            let ids_base = self.s.route_idx_r.ptr as *const i32;
            let rw_base = self.s.route_w_r.ptr as *const f32;
            for r in 0..m {
                self.dev.expert_gate_up_fp4_batched(
                    self.s.xq4_r.as_u8().wrapping_add(r * (dim / 2)),
                    self.s.xsc4_r.as_f32().wrapping_add(r * (dim / 32)),
                    (self.s.ex_act_r.ptr as *mut f32).wrapping_add(r * row_pitch),
                    act_slot as i64,
                    1,
                    dim as i32,
                    inter_local as i32,
                    cfg.swiglu_limit,
                    topk as i32,
                    w1_base,
                    w1_stride,
                    w1s_base,
                    w1s_stride,
                    w3_base,
                    w3_stride,
                    w3s_base,
                    w3s_stride,
                    ids_base.wrapping_add(r * topk),
                    ld.experts_ilv as i32,
                )?;
            }
            // The separate swiglu pass keeps the single-row call shape (one row per
            // launch) so its accumulation order is the verified one.
            if !gateup_fused {
                for r in 0..m {
                    self.dev.swiglu_limit(
                        (self.s.ex_act_r.ptr as *mut f32).wrapping_add(r * row_pitch),
                        1,
                        inter_local as i32,
                        cfg.swiglu_limit,
                    )?;
                }
            }
            for r in 0..m {
                let act = (self.s.ex_act_r.ptr as *const f32).wrapping_add(r * row_pitch);
                let ids = ids_base.wrapping_add(r * topk);
                let rw = rw_base.wrapping_add(r * topk);
                let out = (self.s.moe_out_r.ptr as *mut f32).wrapping_add(r * dim);
                if down_fuse() && self.dev.supports_down_fuse() {
                    // ONE launch: the slot loop and the fixed-order sum, verbatim.
                    self.dev.expert_down_reduce_fp4_batched(
                        act,
                        act_slot as i64,
                        out,
                        1,
                        dim as i32,
                        inter_local as i32,
                        rw,
                        1,
                        topk as i32,
                        w2_base,
                        w2_stride,
                        w2s_base,
                        w2s_stride,
                        ids,
                    )?;
                } else {
                    let scratch =
                        (self.s.ex_down_r.ptr as *mut f32).wrapping_add(r * topk * dim);
                    self.dev.expert_down_fp4_batched(
                        act,
                        act_slot as i64,
                        scratch,
                        dim as i64,
                        1,
                        dim as i32,
                        inter_local as i32,
                        rw,
                        1,
                        topk as i32,
                        w2_base,
                        w2_stride,
                        w2s_base,
                        w2s_stride,
                        ids,
                    )?;
                    self.dev
                        .moe_down_reduce(scratch as *const f32, out, dim as i32, topk as i32)?;
                }
            }
        }

        // ---- shared expert, one row per launch ----
        // Only the rank(s) that actually contribute it: all of them under
        // SHARED_TP, rank 0 alone under the replicated layout (otherwise the AR
        // would sum it `world` times).
        if !self.opts.skip_shared_expert && shared_rank {
            if let (Some(w1), Some(w1s), Some(w3), Some(w3s), Some(w2), Some(w2s)) = (
                ld.shared_w1.as_ref(),
                ld.shared_w1_scale.as_ref(),
                ld.shared_w3.as_ref(),
                ld.shared_w3_scale.as_ref(),
                ld.shared_w2.as_ref(),
                ld.shared_w2_scale.as_ref(),
            ) {
                for r in 0..m {
                    // The T1/T2 fast paths are single-row gates around this very
                    // pair, so the rows path runs the (quant1, gemm) pair.
                    self.quant1((self.s.xn_r.ptr as *const f32).wrapping_add(r * dim), dim as i32)?;
                    let sh_fused = sh_exp_mx2()
                        && self.dev.gemm_fp8_mx2(
                            self.s.xq.as_u8(),
                            self.s.xsc.as_f32(),
                            w1.as_u8(),
                            w1s.as_u8(),
                            std::ptr::null(),
                            self.s.sh_act_r.ptr as *mut f32,
                            sh_il as i32,
                            w3.as_u8(),
                            w3s.as_u8(),
                            std::ptr::null(),
                            (self.s.sh_act_r.ptr as *mut f32).wrapping_add(sh_il),
                            sh_il as i32,
                            dim as i32,
                        )?;
                    if !sh_fused {
                        self.dev.gemm_fp8_mx(
                            self.s.xq.as_u8(),
                            self.s.xsc.as_f32(),
                            w1.as_u8(),
                            w1s.as_u8(),
                            std::ptr::null(),
                            self.s.sh_act_r.ptr as *mut f32,
                            1,
                            sh_il as i32,
                            dim as i32,
                        )?;
                        self.dev.gemm_fp8_mx(
                            self.s.xq.as_u8(),
                            self.s.xsc.as_f32(),
                            w3.as_u8(),
                            w3s.as_u8(),
                            std::ptr::null(),
                            (self.s.sh_act_r.ptr as *mut f32).wrapping_add(sh_il),
                            1,
                            sh_il as i32,
                            dim as i32,
                        )?;
                    }
                    self.dev.swiglu_limit(
                        self.s.sh_act_r.ptr as *mut f32,
                        1,
                        sh_il as i32,
                        cfg.swiglu_limit,
                    )?;
                    self.quant1(self.s.sh_act_r.ptr as *const f32, sh_il as i32)?;
                    // w2 folds straight into this row's accumulator (the A5
                    // epilogue); a decline falls back to a scratch + the standalone
                    // add, which is the same pair with the same association.
                    let out = (self.s.moe_out_r.ptr as *mut f32).wrapping_add(r * dim);
                    let added = moe_epi_add()
                        && self.dev.supports_gemm_fp8_add()
                        && self.dev.gemm_fp8_mx_add(
                            self.s.xq.as_u8(),
                            self.s.xsc.as_f32(),
                            w2.as_u8(),
                            w2s.as_u8(),
                            std::ptr::null(),
                            out,
                            1,
                            dim as i32,
                            sh_il as i32,
                        )?;
                    if !added {
                        self.dev.gemm_fp8_mx(
                            self.s.xq.as_u8(),
                            self.s.xsc.as_f32(),
                            w2.as_u8(),
                            w2s.as_u8(),
                            std::ptr::null(),
                            self.s.ex_out.ptr as *mut f32,
                            1,
                            dim as i32,
                            sh_il as i32,
                        )?;
                        self.dev.add_inplace_raw(
                            out as *mut std::ffi::c_void,
                            self.s.ex_out.ptr as *const c_void,
                            dim as i64,
                        )?;
                    }
                }
            }
        }

        // ---- the MoE all-reduce ----
        // Payload = the m rows of `moe_out_r`. Byte-wise and ascending-rank, so
        // each row's sum is the single-row AR's sum for that row.
        if let Some(c) = self.comm.clone() {
            c.all_reduce_inplace(self.s.moe_out_r.ptr as *mut std::ffi::c_void, fb(mdim))?;
            c.end_round();
        }
        Ok(())
    }

    /// Tensor-parallel degree / this rank's index (1 / 0 without a collective).
    fn world(&self) -> usize {
        self.comm.as_ref().map(|c| c.world).unwrap_or(1)
    }
    fn rank(&self) -> usize {
        self.comm.as_ref().map(|c| c.rank).unwrap_or(0)
    }

    /// The three device-resident premix slots: 0 is the incoming premix the
    /// attention collapses with, 1 receives this layer's attn_pre, 2 receives the
    /// ffn_pre that the next layer uses.
    fn premix_slot(&self, i: usize) -> &DevBuf {
        match i % 3 {
            0 => &self.s.pre_a,
            1 => &self.s.pre_b,
            _ => &self.s.pre_c,
        }
    }

    /// hc_mixes, with the fused spread front end in front of it. The fused kernel
    /// answers InvalidValue when its gate is off or the row count is past the
    /// spread tables, which is reported here as Ok(false) and the single-block
    /// kernel then runs exactly as before, so both shapes are numerically
    /// identical and the switch is free to make.
    ///
    /// When the fused path runs it also does the collapse + rmsnorm of
    /// `hc_collapse_norm`, reading `pre_collapse` (the premix slot the caller
    /// collapses with, which is deliberately not the one the mixes write) and
    /// writing `out`; the returned bool then tells the caller to skip that call.
    #[allow(clippy::too_many_arguments)]
    fn hc_mixes_auto(
        &mut self,
        hc_fn: *const f32,
        hc_scale: *const f32,
        hc_base: *const f32,
        pre_slot: usize,
        hc: usize,
        dim: usize,
        sinkhorn_iters: i32,
        eps: f32,
        norm_w: *const f32,
        pre_collapse: *const f32,
        out: *mut f32,
        eps_norm: f32,
        xq: *mut u8,
        xsc: *mut f32,
    ) -> Result<bool> {
        // Stage-C persistent forms, both default OFF and both selected only when
        // the .so carries the symbol. `_MB` (multi-block) is tried first: same
        // one-launch phase structure, but the dots are spread instead of pinned
        // to one SM. The fallback chain is
        // (persist_mb -> persist -> two-launch -> hc_mixes), each step silent.
        let fused = if Self::hc_tail_split()
            && self.dev.supports_hc_tail_split()
            && !norm_w.is_null()
        {
            // Tail split (`DSV41_HC_TAIL_SPLIT`, default ON). The whole tail chain
            // — collapse + rmsnorm + fp8 (EARLY), the dots, and ss + sigmoid +
            // sinkhorn + comb (LATE) — runs on the side stream in that order; main
            // waits only the EARLY half, whose consumer is the projection group
            // immediately below. The dots are read only by the LATE branch (same
            // side stream, stream order publishes `g_hc_part`), so they no longer
            // cost main anything. `norm_w` non-null is required: with no collapse
            // half there is nothing to hand main early.
            //
            // (2026-09-11) The `!hcpost_epi()` exclusion is GONE. It existed only
            // because the AR fold consumes `comb`/`post` inside
            // attention()/moe_reduce() — i.e. EARLIER than the join that used to
            // be the only one (`layer`, after the AR). With the fold on, that join
            // was posted too late: it ordered nothing. The fix is local — the
            // AR fold now waits the split's `join_ev` itself, immediately before
            // the fused AR (`ar_hc_post_fold`), which is exactly the point where
            // the LATE half's outputs (`post`/`comb`, and `s.h` which LATE reads
            // and the fold's `hc_res` write clobbers) stop being safe to touch.
            // The post-AR join in `layer` stays for the un-folded path (a no-op
            // once the early join has consumed the armed flag).
            self.dev.hc_front_split(
                self.s.h.ptr as *const f32,
                hc_fn,
                hc_scale,
                hc_base,
                norm_w,
                pre_collapse,
                self.premix_slot(pre_slot).ptr as *mut f32,
                self.s.post.ptr as *mut f32,
                self.s.comb.ptr as *mut f32,
                out,
                1,
                hc as i32,
                dim as i32,
                sinkhorn_iters,
                eps,
                eps_norm,
                xq,
                xsc,
            )?
        } else if Self::hc_persist_mb() && self.dev.supports_hc_persist_mb() {
            self.dev.hc_front_persist_mb(
                self.s.h.ptr as *const f32,
                hc_fn,
                hc_scale,
                hc_base,
                norm_w,
                pre_collapse,
                self.premix_slot(pre_slot).ptr as *mut f32,
                self.s.post.ptr as *mut f32,
                self.s.comb.ptr as *mut f32,
                out,
                1,
                hc as i32,
                dim as i32,
                sinkhorn_iters,
                eps,
                eps_norm,
                xq,
                xsc,
            )?
        } else if Self::hc_persist() && self.dev.supports_hc_persist() {
            // Stage-C persistent prototype (DSV41_HC_PERSIST=1, default OFF): the
            // whole front end as ONE phase-machine block instead of the two-launch
            // dots+tail pair.
            self.dev.hc_front_persist(
                self.s.h.ptr as *const f32,
                hc_fn,
                hc_scale,
                hc_base,
                norm_w,
                pre_collapse,
                self.premix_slot(pre_slot).ptr as *mut f32,
                self.s.post.ptr as *mut f32,
                self.s.comb.ptr as *mut f32,
                out,
                1,
                hc as i32,
                dim as i32,
                sinkhorn_iters,
                eps,
                eps_norm,
                xq,
                xsc,
            )?
        } else {
            self.dev.hc_front(
                self.s.h.ptr as *const f32,
                hc_fn,
                hc_scale,
                hc_base,
                norm_w,
                pre_collapse,
                self.premix_slot(pre_slot).ptr as *mut f32,
                self.s.post.ptr as *mut f32,
                self.s.comb.ptr as *mut f32,
                out,
                1,
                hc as i32,
                dim as i32,
                sinkhorn_iters,
                eps,
                eps_norm,
                xq,
                xsc,
            )?
        };
        if fused {
            return Ok(true);
        }
        self.dev.hc_mixes(
            self.s.h.ptr as *const f32,
            hc_fn,
            hc_scale,
            hc_base,
            self.premix_slot(pre_slot).ptr as *mut f32,
            self.s.post.ptr as *mut f32,
            self.s.comb.ptr as *mut f32,
            1,
            (hc * dim) as i32,
            hc as i32,
            sinkhorn_iters,
            eps,
        )?;
        Ok(false)
    }


    fn dl(&self, src: *const f32, n: usize) -> Result<Vec<f32>> {
        let mut v = vec![0f32; n];
        let b = Device::view(src as *mut c_void, n * 4);
        self.dev.download_f32(&b, &mut v)?;
        Ok(v)
    }

    /// Host-upload helper (kept: the prefill and probe paths use this family).
    #[allow(dead_code)]
    fn ul_i32(&self, dst: *mut c_void, v: &[i32]) -> Result<()> {
        self.dev.upload_f32_at(dst, 0, unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len())
        })
    }

    #[allow(dead_code)]
    fn ul_f32(&self, dst: *mut c_void, v: &[f32]) -> Result<()> {
        self.dev.upload_f32_at(dst, 0, v)
    }

    /// One transformer layer. Returns the pre-mix the next block's first
    /// collapse must use (this block's *attention* mix).
    /// `pa` indexes the DEVICE-resident premix the attention collapses with; the
    /// return value indexes the one the next layer must use. Nothing here touches
    /// the host: the coefficients used to be downloaded and re-uploaded every
    /// layer, and a download is a full device sync.
/// Segment C fusion: hc_post written straight back onto the residual stream, which
/// drops the h2 staging buffer and its device-to-device copy. Read once, because
/// the hot path must never touch the environment per call.
fn fuse_c() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_FUSE_C").map(|v| v != "0").unwrap_or(true))
}

/// Segment B cluster 1 fusion: hc_collapse + rmsnorm(ffn_norm) in one kernel.
fn fuse_b1() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_FUSE_B1").map(|v| v != "0").unwrap_or(true))
}

/// Segment-C AR fold (P1 of the persistent roadmap): `hc_post_inplace` moved
/// into the AR pubred epilogue that produced its `x` (see
/// `ferrite_p2p_ar_v5_hcpost`). Saves the standalone launch at each of the two
/// per-layer AR sites. DEFAULT ON (2026-09-11) — round 31 measured it neutral
/// (the −0.15ms of the removed launches is offset by the pubred grid's wider
/// column walk), and it now coexists with the default-ON tail split because
/// [`Self::ar_hc_post_fold`] waits the split's join event before its launch.
/// The fold still rewrites a bit-exactness-sensitive chain (the AR sum feeds the
/// hc_post, and the epilogue lives in a different translation unit than
/// `dsv41_hc_post_inplace_kernel`), so `DSV41_HCPOST_EPI=0` remains the A/B arm.
fn hcpost_epi() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_HCPOST_EPI").map(|v| v != "0").unwrap_or(true))
}

/// Stage-C persistent prototype gate (docs/agent/dsv41-persistent-arch.md §1):
/// `DSV41_HC_PERSIST=1` makes the hc front end run as ONE `__syncthreads` phase
/// machine (`dsv41_hc_front_persist`) instead of the two-launch dots+tail pair.
/// DEFAULT OFF — the merge is bit-exact by construction but trades the dots'
/// 24-way block parallelism for a single block, so it must clear the same-binary
/// A/B + token-parity gate before it can be considered. A `.so` without the
/// symbol silently keeps the two-launch path (`Device::supports_hc_persist`).
fn hc_persist() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_HC_PERSIST").map(|v| v != "0").unwrap_or(false))
}

/// Stage-C persistent MULTI-BLOCK gate (`DSV41_HC_PERSIST_MB=1`, default OFF):
/// the same one-launch front end, but the dots spread over `mix * split` blocks
/// (one per projection row and K chunk) with the collapse on a parallel block
/// and the tail elected to the last-finishing dot block. This is the shape that
/// fixes the single-block prototype's parallelism loss — see
/// `docs/agent/dsv41-persistent-arch.md` §1. TAKES PRECEDENCE over
/// `DSV41_HC_PERSIST`. DEFAULT OFF because split > 1 is deterministic but NOT
/// bit-exact (the K partials recombine in ck order, not the single warp's tree),
/// so it needs a tolerance gate plus the same-binary A/B token gate before it
/// may be considered; split = 1 is bit-exact and exists as the parity target.
/// A `.so` without the symbol silently keeps the older path.
fn hc_persist_mb() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_HC_PERSIST_MB").map(|v| v != "0").unwrap_or(false))
}

/// hc TAIL SPLIT gate (`DSV41_HC_TAIL_SPLIT`, default ON). The tail kernel's two
/// halves are independent: the collapse/rmsnorm/fp8 (EARLY) feeds the next
/// projection, the ss/sigmoid/sinkhorn/comb (LATE) feeds hc_post a whole
/// projection + AR later. Splitting them lets the LATE half run on a side stream
/// underneath the projections, hiding its ~10.7us of serialised sinkhorn latency.
/// Falls back silently to the single-launch `hc_front` when the `.so` has no
/// `dsv41_hc_front_split`, when the runtime could not create the side stream or
/// events, or when there is no collapse half (`DSV41_FUSE_B1=0`). Bit-identical
/// either way, so "0" is a pure A/B arm. "1"/unset enables.
fn hc_tail_split() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_HC_TAIL_SPLIT").map(|v| v != "0").unwrap_or(true))
}

    fn layer(&mut self, layer: usize, pos: usize, pa: usize) -> Result<usize> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let ld = &self.w.layers[layer];

        // DSpark target-hidden tap: the draft consumes the per-copy mean of the
        // target layers' ATTENTION INPUT — h here, before this layer's mixes
        // run (hc_mixes only READS h). The tap buffers are fixed allocations,
        // so this one-block collapse is graph-capturable with the rest of the
        // step; it costs ~3 tiny launches per step and only when armed.
        if cfg.dspark_armed() {
            if let Some(slot) = cfg.dspark_target_slot(layer) {
                self.dev.hc_collapse(
                    self.s.h.ptr as *const f32,
                    self.s.dspark_pre_mean.as_f32(),
                    (self.s.dspark_tap.ptr as *mut f32).wrapping_add(slot * dim),
                    1,
                    hc as i32,
                    dim as i32,
                )?;
            }
        }

        // ---------------- attention block ----------------
        // The fused front end, when it runs, also collapses and normalises; the
        // collapse reads `premix_slot(pa)` rather than the slot the mixes write,
        // and only the fuse_b1 shape wants that fold at all.
        let (hc_nw, hc_pc, hc_out): (*const f32, *const f32, *mut f32) = if Self::fuse_b1() {
            (
                ld.attn_norm.as_ref().unwrap().as_f32(),
                self.premix_slot(pa).as_f32(),
                self.s.xn.ptr as *mut f32,
            )
        } else {
            (std::ptr::null(), std::ptr::null(), std::ptr::null_mut())
        };
        let hc_done = self.hc_mixes_auto(
            ld.hc_attn_fn.as_ref().unwrap().as_f32(),
            ld.hc_attn_scale.as_ref().unwrap().as_f32(),
            ld.hc_attn_base.as_ref().unwrap().as_f32(),
            1, // attn_pre
            hc,
            dim,
            cfg.hc_sinkhorn_iters as i32,
            cfg.hc_eps,
            hc_nw,
            hc_pc,
            hc_out,
            cfg.norm_eps,
            self.s.xq.ptr as *mut u8,
            self.s.xsc.ptr as *mut f32,
        )?;
        // T1: when the fused front ran the collapse, it also emitted the fp8
        // quantisation of `xn` (s.xq/s.xsc), so the next quant1(xn) - lin2's, in
        // the attention - is redundant and quant1 skips it (pointer-gated).
        self.s.xq_of_xn_valid.set(hc_done && !hc_nw.is_null());
        let _t_all = std::time::Instant::now();
        if layer == 0 && hc_dbg() {
            let attn_pre = self.dl(self.premix_slot(1).as_f32(), hc)?;
            let po = self.dl(self.s.post.as_f32(), hc)?;
            let cb = self.dl(self.s.comb.as_f32(), hc * hc)?;
            eprintln!("[mine] L0 pre={attn_pre:?}");
            eprintln!("[mine] L0 post={po:?}");
            eprintln!("[mine] L0 comb={cb:?}");
            let rs: Vec<f32> = (0..hc)
                .map(|j| (0..hc).map(|k| cb[j * hc + k]).sum())
                .collect();
            eprintln!("[mine] L0 comb_rowsum={rs:?}");
        }
        if Self::fuse_b1() {
            if !hc_done {
                self.dev.hc_collapse_norm(
                    self.s.h.ptr as *mut f32,
                    self.premix_slot(pa).as_f32(),
                    ld.attn_norm.as_ref().unwrap().as_f32(),
                    self.s.xn.ptr as *mut f32,
                    1,
                    hc as i32,
                    dim as i32,
                    cfg.norm_eps,
                )?;
            }
        } else {
        self.dev.hc_collapse(
            self.s.h.ptr as *const f32,
            self.premix_slot(pa).as_f32(),
            self.s.x.ptr as *mut f32,
            1,
            hc as i32,
            dim as i32,
        )?;
        self.dev.rmsnorm(
            self.s.x.ptr as *const f32,
            ld.attn_norm.as_ref().unwrap().as_f32(),
            self.s.xn.ptr as *mut f32,
            1,
            dim as i32,
            cfg.norm_eps,
        )?;
        }
        let attn_hc_folded = self.attention(layer, pos)?;
        // Tail split join: the attention (projections + AR) has now consumed the
        // EARLY half, so the LATE half's `comb` must be visible to hc_post. A
        // no-op unless hc_mixes_auto issued a split above.
        self.dev.hc_tail_join()?;
        if Self::fuse_c() {
            // Segment-C P1: the fold already wrote this layer's hc_post onto
            // `s.h` from the AR pubred epilogue, so only the un-folded path
            // launches the standalone kernel here.
            if !attn_hc_folded {
                self.dev.hc_post_inplace(
                    self.s.h.ptr as *mut f32,
                    self.s.o.ptr as *const f32,
                    self.s.post.as_f32(),
                    self.s.comb.as_f32(),
                    hc as i32,
                    dim as i32,
                )?;
            }
        } else {
            self.dev.hc_post(
                self.s.o.ptr as *const f32,
                self.s.h.ptr as *const f32,
                self.s.post.as_f32(),
                self.s.comb.as_f32(),
                self.s.h2.ptr as *mut f32,
                1,
                hc as i32,
                dim as i32,
            )?;
            self.copy_h_back()?;
        }

        if phase_dbg() {
            eprintln!("[phs] L{layer} attn={:?}", _t_all.elapsed());
        }
        let _t_moe = std::time::Instant::now();
        // ---------------- FFN block ----------------
        let (ffn_nw, ffn_pc, ffn_out): (*const f32, *const f32, *mut f32) = if Self::fuse_b1() {
            (
                ld.ffn_norm.as_ref().unwrap().as_f32(),
                self.premix_slot(1).as_f32(),
                self.s.xn.ptr as *mut f32,
            )
        } else {
            (std::ptr::null(), std::ptr::null(), std::ptr::null_mut())
        };
        let ffn_done = self.hc_mixes_auto(
            ld.hc_ffn_fn.as_ref().unwrap().as_f32(),
            ld.hc_ffn_scale.as_ref().unwrap().as_f32(),
            ld.hc_ffn_base.as_ref().unwrap().as_f32(),
            2, // ffn_pre -> next layer
            hc,
            dim,
            cfg.hc_sinkhorn_iters as i32,
            cfg.hc_eps,
            ffn_nw,
            ffn_pc,
            ffn_out,
            cfg.norm_eps,
            self.s.xq.ptr as *mut u8,
            self.s.xsc.ptr as *mut f32,
        )?;
        // T1 (ffn side): the tail emitted the fp8 of the ffn-norm output, so the
        // MoE's quant1(xn) - its first xq consumer - is redundant and skips.
        self.s.xq_of_xn_valid.set(ffn_done && !ffn_nw.is_null());
        if phase_dbg() {
            eprintln!("[phs] L{layer} ffn={:?}", _t_moe.elapsed());
        }
        // the FFN collapses with THIS layer's attn_pre (slot 1), which stayed on
        // the device; the FFN's own pre (slot 2) is what the NEXT layer uses.
        if Self::fuse_b1() {
            if !ffn_done {
                self.dev.hc_collapse_norm(
                    self.s.h.ptr as *mut f32,
                    self.premix_slot(1).as_f32(),
                    ld.ffn_norm.as_ref().unwrap().as_f32(),
                    self.s.xn.ptr as *mut f32,
                    1,
                    hc as i32,
                    dim as i32,
                    cfg.norm_eps,
                )?;
            }
        } else {
        self.dev.hc_collapse(
            self.s.h.ptr as *const f32,
            self.premix_slot(1).as_f32(),
            self.s.x.ptr as *mut f32,
            1,
            hc as i32,
            dim as i32,
        )?;
        self.dev.rmsnorm(
            self.s.x.ptr as *const f32,
            ld.ffn_norm.as_ref().unwrap().as_f32(),
            self.s.xn.ptr as *mut f32,
            1,
            dim as i32,
            cfg.norm_eps,
        )?;
        }
        let _t_moeonly = std::time::Instant::now();
        // DSV41_GRAPH_MOE=1 captures the host-free part of the MoE (everything up
        // to the all-reduce) into one per-layer graph. The first step warms every
        // kernel; the capture happens on the next one, then it replays.
        if self.moe_graph_armed {
            if let Some(e) = self.moe_graph.get(layer).and_then(|x| *x) {
                self.moe_graph_replays = self.moe_graph_replays.wrapping_add(1);
                if self.moe_graph_replays == 1 {
                    eprintln!("[gmo] first MoE segment replay at L{layer} (captures={})",
                        self.moe_graph_captures);
                }
                self.dev.graph_launch(e)?;
            } else {
                self.dev.capture_begin()?;
                self.moe(layer, ld)?;
                let g = self.dev.capture_end()?;
                let e = self.dev.graph_instantiate(g)?;
                self.dev.graph_free(g, std::ptr::null_mut())?;
                self.moe_graph[layer] = Some(e);
                self.moe_graph_captures = self.moe_graph_captures.wrapping_add(1);
                if self.moe_graph_captures == 1 {
                    eprintln!("[gmo] MoE segment graph armed: first capture at L{layer}");
                }
            }
        } else {
            self.moe(layer, ld)?;
        }
        let moe_hc_folded = self.moe_reduce(layer)?;
        if phase_dbg() {
            eprintln!("[phs] L{layer} moe={:?}", _t_moeonly.elapsed());
        }
        // Tail split join for the FFN front end (same contract as the attention
        // side): the MoE consumed the EARLY half, so wait the LATE half's comb.
        self.dev.hc_tail_join()?;
        if Self::fuse_c() {
            // Segment-C P1: same fold as the attention side, on the MoE AR.
            if !moe_hc_folded {
                self.dev.hc_post_inplace(
                    self.s.h.ptr as *mut f32,
                    self.s.o.ptr as *const f32,
                    self.s.post.as_f32(),
                    self.s.comb.as_f32(),
                    hc as i32,
                    dim as i32,
                )?;
            }
        } else {
            self.dev.hc_post(
                self.s.o.ptr as *const f32,
                self.s.h.ptr as *const f32,
                self.s.post.as_f32(),
                self.s.comb.as_f32(),
                self.s.h2.ptr as *mut f32,
                1,
                hc as i32,
                dim as i32,
            )?;
            self.copy_h_back()?;
        }
        if phase_dbg() {
            eprintln!("[phs] L{layer} ffn_total={:?}", _t_moe.elapsed());
        }
        Ok(2) // slot 2 holds this layer's ffn_pre = the next layer's premix
    }

    /// Diagnostic: report the magnitude of a stage's output. `DSV41_STATS=1`.
    /// Turns "the text is wrong" into "stage X is fine / stage Y exploded".
    fn stats(&self, label: &str, buf: &DevBuf, n: usize) -> Result<()> {
        if stats_dbg() {
            self.dev.sync()?;
            let mut v = vec![0f32; n];
            let b = Device::view(buf.ptr, n * 4);
            self.dev.download_f32(&b, &mut v)?;
            let mut mx = f32::NEG_INFINITY;
            let mut mn = f32::INFINITY;
            let mut ss = 0f64;
            let mut nan = 0usize;
            for &x in &v {
                if x.is_nan() {
                    nan += 1;
                } else {
                    mx = mx.max(x);
                    mn = mn.min(x);
                    ss += (x as f64) * (x as f64);
                }
            }
            eprintln!(
                "[stats] {label:<26} rms={:9.4} min={:9.4} max={:9.4} nan={nan}",
                (ss / n as f64).sqrt(),
                mn,
                mx
            );
        }
        Ok(())
    }

    fn copy_h_back(&self) -> Result<()> {
        self.dev
            .memcpy_d2d(self.s.h.ptr, self.s.h2.ptr as *const c_void, self.s.h.bytes)
    }

    /// MLA window path + grouped output projection.
    /// Decode attention for one layer. Returns whether the segment-C hc-post was
    /// folded into this layer's attention all-reduce (see
    /// [`Self::ar_hc_post_fold`]); the caller then skips the standalone launch.
    fn attention(&mut self, layer: usize, pos: usize) -> Result<bool> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let hd = cfg.head_dim;
        let nh = cfg.n_heads;
        let ql = cfg.q_lora_rank;
        let world = self.world();
        // wq_b is ColumnParallel: this rank owns a contiguous block of heads
        let nlh = nh / world;
        let ld = &self.w.layers[layer];

        // queries + window KV both read xn: one quantise and one gemv launch for
        // the pair (wkv's 128 rows ride beside wq_a's blocks instead of paying
        // their own latency-floor slot after wq_b and the rope). Both outputs
        // are bit-identical to two separate lins - each row is still one warp in
        // the same lane order.
        let kv_early = self.lin2(
            self.s.xn.ptr as *const f32,
            dim as i32,
            ld.wq_a.as_ref().unwrap(),
            ld.wq_a_scale.as_ref().unwrap(),
            ql as i32,
            self.s.qr.ptr as *mut f32,
            ld.wkv.as_ref().unwrap(),
            ld.wkv_scale.as_ref().unwrap(),
            hd as i32,
            self.s.kv.ptr as *mut f32,
        )?;
        if !kv_early {
            self.lin(
                self.s.xn.ptr as *const f32,
                dim as i32,
                ld.wq_a.as_ref().unwrap(),
                ld.wq_a_scale.as_ref().unwrap(),
                ql as i32,
                self.s.qr.ptr as *mut f32,
            )?;
        }
        // DUAL_CHAIN (DSV41_DUAL_CHAIN, default ON): fork the kv half onto the
        // runtime's second side stream HERE, so its kv norm + rope runs while the
        // q chain below (NORM_FUSE/lin_rope + wq_b gemv + rope, ~13.5us/layer)
        // owns the main stream. `s.kv` is complete at this point — `lin2`/the
        // fallback just wrote it, and nothing on the q side reads it — so this is
        // exactly the two chains' last shared node.
        //
        // Gated on `kv_early`: the unfused path above ran `lin(wq_a)` and would
        // run `lin(wkv)` below (line ~2449), whose `quant1` writes the SHARED
        // `s.xq`/`s.xsc` that the q chain's `lin_rope`/`lin2_rope` also write.
        // Overlapping those two would be a genuine data race on the activation
        // buffer, so that path stays serial.
        let dual = dual_chain() && kv_early && self.dev.supports_dual_chain();
        if dual {
            self.dev.dual_chain_fork()?;
        }
        // COMPRESS_SIDE (DSV41_COMPRESS_SIDE, default ON): the kv-source layer's
        // four compressor launches (~30us) are forked onto the THIRD side stream
        // HERE — the same point as the dual chain, right after `lin2` — so they
        // run under the q chain (~13.5us, main) AND the kv chain (side_stream2)
        // instead of serially between `ring_win_fuse` and the indexer. The
        // compressor reads `s.xn` (final since the pre-attention rmsnorm), the
        // position counter and this layer's own state; it writes only
        // layer-private buffers + the ring's COMPRESSED rows + this layer's
        // device counter, so it shares nothing with either chain. See
        // `compress_side()` for the gates.
        //
        // The compressor must be a kv source with the comp weights present (the
        // same predicate `compress()` early-returns on) — otherwise the fork
        // would open an empty side-stream window for nothing.
        let comp_side = compress_side()
            && cfg.compress_ratio(layer) > 0
            && cfg.is_kv_source(layer)
            && !cublas_m1()
            && ld.comp_wkv.is_some()
            && ld.comp_norm.is_some()
            && self.dev.supports_compress_side();
        let mut comp_len_side: Option<usize> = None;
        if comp_side {
            self.dev.compress_side_fork()?;
            let s3 = self.dev.side_stream3();
            comp_len_side = Some(self.compress_on(layer, pos, s3)?);
        }
        // L2+L3 decision, hoisted above the norm block: which wq_b launch is
        // taken decides whether NORM_FUSE may run at all. The two-family
        // IDX_FUSE launch shares ONE `xq` between wq_b and idx_wq_b, so it cannot
        // take a path that leaves `qr` raw.
        //
        // L2+L3 fusion (verified by p1p2-zero-hist): `wq_b` and the indexer's
        // `idx_wq_b` read the SAME `qr` buffer (rmsnorm writes it in place at
        // :1572, neither touches it between :1580 and :1937), both k=ql=1280,
        // and the mx2 contract is bit-identical to the two singles. Only the 8
        // index-source layers carry idx_wq_b; the other 32 fall through to the
        // single. The indexer() below skips its own lin when idx_q is already
        // computed (self.s.idx_q_ready flag).
        // DSV41_IDX_FUSE (default OFF in the code: `.unwrap_or(false)` - the
        // "default ON" note in the old comment here was stale) turns this on. It
        // is a SHAPE gate as much as an env gate: only an index-source layer
        // carries idx_wq_b, and mx2 can still decline the shape (Ok(false)) - in
        // which case BOTH singles must run, so idx_q_ready stays false and the
        // indexer computes its own query. The flag is written on EVERY path
        // here (true or false), so a layer that never calls the indexer cannot
        // leave a stale value behind.
        let idx_fused = idx_fuse()
            && cfg.is_index_source(layer)
            && ld.idx_wq_b.is_some()
            && ld.idx_wq_b_scale.is_some();
        // NORM_FUSE (DSV41_NORM_FUSE, default ON): the wq_b rope GEMV's PROLOGUE
        // computes the RMSNorm + fp8 encoding of `qr` itself, so the standalone
        // `rmsnorm_q` launch (and its graph node) disappears - 40 per step. It is
        // bit-identical by construction: the prologue's reduction tree, element
        // loop, amax shuffle and e4m3 round are `rmsnorm_q_kernel`'s, term for
        // term, at the same 1024-thread block, and the gemv then reads the same
        // bytes out of shared memory. Tried FIRST so that a decline (shape, stale
        // .so, IDX_FUSE) falls straight through to the pair below.
        //
        // ⚠️ This path leaves `qr` UNNORMALISED. Every later reader of `qr` must
        // therefore use the same fused launch - `indexer()` does, driven by
        // `s.qr_raw` (set below on EVERY path). That is why it is refused when
        // `idx_fused` (one shared `xq`) is on.
        let mut norm_fused = false;
        let mut q_roped = false;
        let mut idx_q_roped = false;
        if norm_fuse() && !idx_fused && self.dev.supports_gemm_fp8_norm() {
            norm_fused = self.lin_rope_norm(
                self.s.qr.ptr as *const f32,
                ld.q_norm.as_ref().unwrap().as_f32(),
                cfg.norm_eps,
                ql as i32,
                ld.wq_b.as_ref().unwrap(),
                ld.wq_b_scale.as_ref().unwrap(),
                (nlh * hd) as i32,
                self.s.q.ptr as *mut f32,
                cfg.rope_head_dim as i32,
                hd as i32,
            )?;
            q_roped = norm_fused;
        }
        self.s.qr_raw.set(norm_fused);
        if norm_fused {
            // No `rmsnorm_q` ran on this path, so there is no fused fp8 emission
            // to consume - and a stale `true` left by an earlier layer would make
            // some later `quant1(qr)` silently skip its launch.
            self.s.xq_of_qr_valid.set(false);
        } else {
            // T2 (attention side): the rmsnorm epilogue can emit the fp8 of its OWN
            // normalised output (`ferrite_rmsnorm_q`), which is exactly what the
            // wq_b projection's quant1(qr) - the very next qr consumer - would
            // compute. Same absmax, same fast_round_scale, same clamp/e4m3 round
            // sequence (T1's, term for term), so the emitted pair is bit-identical
            // to the launch it replaces. Falls back to the plain rmsnorm (and a
            // cleared flag) on an .so without the symbol.
            let qr_q = qr_epi()
                && self.dev.rmsnorm_q(
                    self.s.qr.ptr as *const f32,
                    ld.q_norm.as_ref().unwrap().as_f32(),
                    self.s.qr.ptr as *mut f32,
                    1,
                    ql as i32,
                    cfg.norm_eps,
                    self.s.xq.ptr as *mut u8,
                    self.s.xsc.ptr as *mut f32,
                )?;
            self.s.xq_of_qr_valid.set(qr_q);
            if !qr_q {
                self.dev.rmsnorm(
                    self.s.qr.ptr as *const f32,
                    ld.q_norm.as_ref().unwrap().as_f32(),
                    self.s.qr.ptr as *mut f32,
                    1,
                    ql as i32,
                    cfg.norm_eps,
                )?;
            }
        }
        // DSV41_ROPE_FUSE: the same launch can ALSO rotate s.q (family 1) and
        // s.idx_q (family 2) in its epilogue, so neither gets a standalone
        // apply_rope. `q_roped`/`idx_q_roped` (declared beside `norm_fused`
        // above) record which ropes the fused epilogue actually performed; every
        // path sets them, so a layer that never reaches the indexer cannot leave
        // a stale flag behind.
        if !norm_fused && idx_fused {
            // ONE launch for both: wq_b -> s.q, idx_wq_b -> s.idx_q (and, under
            // rope fuse, both rotations).
            idx_q_roped = self.lin2_rope(
                self.s.qr.ptr as *const f32,
                ql as i32,
                ld.wq_b.as_ref().unwrap(),
                ld.wq_b_scale.as_ref().unwrap(),
                (nlh * hd) as i32,
                self.s.q.ptr as *mut f32,
                ld.idx_wq_b.as_ref().unwrap(),
                ld.idx_wq_b_scale.as_ref().unwrap(),
                (cfg.index_n_heads * cfg.index_head_dim) as i32,
                self.s.idx_q.ptr as *mut f32,
                cfg.rope_head_dim as i32,
                hd as i32,
                cfg.index_head_dim as i32,
            )?;
            q_roped = idx_q_roped;
            if idx_q_roped {
                self.s.idx_q_ready.set(true);
            } else {
                let fused = self.lin2(
                    self.s.qr.ptr as *const f32,
                    ql as i32,
                    ld.wq_b.as_ref().unwrap(),
                    ld.wq_b_scale.as_ref().unwrap(),
                    (nlh * hd) as i32,
                    self.s.q.ptr as *mut f32,
                    ld.idx_wq_b.as_ref().unwrap(),
                    ld.idx_wq_b_scale.as_ref().unwrap(),
                    (cfg.index_n_heads * cfg.index_head_dim) as i32,
                    self.s.idx_q.ptr as *mut f32,
                )?;
                self.s.idx_q_ready.set(fused);
            }
        } else {
            self.s.idx_q_ready.set(false);
        }
        if !norm_fused && !self.s.idx_q_ready.get() {
            q_roped = self.lin_rope(
                self.s.qr.ptr as *const f32,
                ql as i32,
                ld.wq_b.as_ref().unwrap(),
                ld.wq_b_scale.as_ref().unwrap(),
                (nlh * hd) as i32,
                self.s.q.ptr as *mut f32,
                cfg.rope_head_dim as i32,
                hd as i32,
            )?;
            if !q_roped {
                self.lin(
                    self.s.qr.ptr as *const f32,
                    ql as i32,
                    ld.wq_b.as_ref().unwrap(),
                    ld.wq_b_scale.as_ref().unwrap(),
                    (nlh * hd) as i32,
                    self.s.q.ptr as *mut f32,
                )?;
            }
        }
        self.s.idx_q_rope.set(idx_q_roped);
        // RoPE over the trailing `rope_head_dim` lanes of each head
        // ALL heads of this token are at the SAME position — step=0. The
        // earlier step=1 gave head i position pos+i (8 different positions for
        // 8 local heads), scrambling the positional encoding: every head's
        // RoPE rotated differently, so the attention scores were positionally
        // wrong. (The KV rope uses rows=1 so step is irrelevant there.)
        // Skipped when the wq_b launch's epilogue already rotated s.q.
        if !q_roped {
            self.dev.apply_rope(
                self.s.q.ptr as *mut f32,
                self.cos.as_f32(),
                self.sin.as_f32(),
                nlh as i32,
                hd as i32,
                cfg.rope_head_dim as i32,
                (cfg.rope_head_dim / 2) as i32,
                self.s.pos_ctr.ptr as *const std::os::raw::c_int, 1, 0,
                0,
                false,
            )?;
        }

        if layer == 0 && hc_dbg() {
            let q = self.dl(self.s.q.as_f32(), nlh * hd)?;
            let r = (q.iter().map(|v| v * v).sum::<f32>() / q.len() as f32).sqrt();
            eprintln!("[mine] L0 q[0..4]={:?} q_full_rms={}", &q[..4], r);
        }
        // window KV (single shared head): already computed beside wq_a when the
        // fused launch ran; only the separate fallback computes it here.
        if !kv_early {
            self.lin(
                self.s.xn.ptr as *const f32,
                dim as i32,
                ld.wkv.as_ref().unwrap(),
                ld.wkv_scale.as_ref().unwrap(),
                hd as i32,
                self.s.kv.ptr as *mut f32,
            )?;
        }
        // DUAL_CHAIN: every launch from here to the join below is the kv half.
        // It rides the second side stream when the fork above took, so it runs
        // under the q chain instead of after it. `kv_stream` is the main stream
        // on every other path, so the serial behaviour is byte-for-byte the old
        // one (same launch order, same stream).
        let kv_stream = if dual { self.dev.side_stream2() } else { self.dev.stream() };
        // The kv norm and its rope are an adjacent pair on the same row: one
        // fused launch, bit-identical to the two (same reduction tree at
        // blockDim 1024, elementwise rope). DSV41_NR_FUSE=0 reverts, and so
        // does an .so without the symbol.
        let nr_fused = nr_fuse()
            && self.dev.rmsnorm_rope_on(
                self.s.kv.ptr as *const f32,
                ld.kv_norm.as_ref().unwrap().as_f32(),
                self.s.kv.ptr as *mut f32,
                self.cos.as_f32(),
                self.sin.as_f32(),
                1,
                hd as i32,
                cfg.rope_head_dim as i32,
                (cfg.rope_head_dim / 2) as i32,
                self.s.pos_ctr.ptr as *const std::os::raw::c_int, 1, 0,
                1,
                false,
                cfg.norm_eps,
                kv_stream,
            )?;
        if !nr_fused {
        self.dev.rmsnorm_on(
            self.s.kv.ptr as *const f32,
            ld.kv_norm.as_ref().unwrap().as_f32(),
            self.s.kv.ptr as *mut f32,
            1,
            hd as i32,
            cfg.norm_eps,
            kv_stream,
        )?;
        self.dev.apply_rope_on(
            self.s.kv.ptr as *mut f32,
            self.cos.as_f32(),
            self.sin.as_f32(),
            1,
            hd as i32,
            cfg.rope_head_dim as i32,
            (cfg.rope_head_dim / 2) as i32,
            self.s.pos_ctr.ptr as *const std::os::raw::c_int, 1, 0,
            1,
            false,
            kv_stream,
        )?;
        }
        // DUAL_CHAIN join: the kv half is fully issued (nothing below writes
        // `s.kv` until the ring append, which is its first consumer). Join the
        // side stream back so everything from here on — the debug read right
        // below, `ring_win_fuse`/`ring_append`, `sparse_attn` — sees the
        // finished `s.kv`. A no-op when the fork did not take.
        if dual {
            self.dev.dual_chain_join()?;
        }

        if layer == 0 && hc_dbg() {
            let kv = self.dl(self.s.kv.as_f32(), hd)?;
            let r = (kv.iter().map(|v| v * v).sum::<f32>() / kv.len() as f32).sqrt();
            eprintln!("[mine] L0 kv[0..4]={:?} kv_rms={}", &kv[..4], r);
        }
        let win = cfg.window_size;
        // raw pointers rather than a live borrow: `compress` below needs
        // &mut self (it updates this layer's published count and buffers)
        // The release shares one KV store across a group of layers: the kv
        // source maintains it and its consumers read it. A consumer therefore
        // must not keep its own window ring (nothing would ever put the
        // compressed rows there), it reads the owner's — which also already
        // holds this step's token, since the owner runs earlier in the stack.
        // The window KV is PER-LAYER: the reference computes `_window_kv(x, ...)`
        // with each layer's own wkv and its own input, and only the *compressed*
        // KV plus the indexer are shared group-wide ("layers sharing a ratio also
        // share one compressed KV and one indexer"). Reading the owner's ring for
        // the window part fed every consumer layer the owner's kv — which is
        // exactly why layer 2 (the owner) matched the official while layer 3, the
        // first consumer, dropped ~15%. DSV41_RING_OWNER=1 restores the old
        // shared-ring behaviour for A/B.
        let owner = if ring_owner_shared() {
            self.kv_owner(layer)
        } else {
            layer
        };
        let ring_ptr = self.layers[owner].ring.ptr;
        // index-source layers compute their OWN selection into their OWN buffer;
        // non-index layers read the owner's (shared) selection
        let idxs_ptr = if cfg.is_index_source(layer) {
            self.layers[layer].idxs.ptr
        } else {
            self.layers[owner].idxs.ptr
        };
        let cache = &self.layers[owner];
        let owns_kv = owner == layer;
        // B2 (DSV41_RING_WIN_FUSE, default ON): the ring append and the window
        // indices are two adjacent, mutually independent one-block kernels (the
        // append writes the ring at pos % win; the indices read nothing but
        // `pos_ctr`). One launch now does both. The indices may move up to here
        // because nothing between this point and `sparse_attn` writes idxs[0,
        // win) - `compress` touches the ring's compressed rows only - and
        // sparse_attn is the only reader. `ring == null` for a consumer layer
        // (it does not own its store) still performs the indices half, so the
        // later standalone `window_idxs` disappears for EVERY layer. Falls back
        // to the two launches on an .so without the symbol.
        //
        // B3 (DSV41_COMP_PLACEHOLDER_FUSE, default ON): the `comp_placeholder`
        // launch below (30/step) writes `idxs[win, win+take)` - the SAME buffer
        // this launch already fills `idxs[0, win)` in, with `sparse_attn` the
        // only reader of either block, so the two fuse. It can be hoisted here
        // because the ONLY writer that could interleave is the indexer, and an
        // index-source layer writes [win, ..) itself: that layer takes this
        // launch with the placeholder half OFF (`ph_ok` false), so nothing is
        // reordered for it. The bound stays DEVICE-derived (`*clen`), which is
        // what keeps the fused launch graph-capture safe.
        //
        // `ph_ok` also excludes a compress SOURCE: `clen[layer]` is advanced by
        // THIS step's `compress_commit`, which is issued after this point (and,
        // under COMPRESS_SIDE, concurrently on the third stream) - reading it
        // here would race. No production layer needs that combination: every
        // `kv_source` (2/8/14/20) is also an `index_source`, so those layers are
        // already excluded. A config where a compress source is NOT an index
        // source keeps the standalone launch below, unchanged.
        let ph_ok = comp_ph_fuse()
            && owns_kv
            && !cfg.is_index_source(layer)
            && !cfg.is_kv_source(layer)
            && cfg.compress_ratio(layer) > 0;
        let mut ph_fused = false;
        let rw_fused = ring_win_fuse() && {
            let ring = if owns_kv {
                cache.ring.ptr as *mut f32
            } else {
                std::ptr::null_mut()
            };
            // clen == null is the switch that turns the placeholder half off, so
            // an excluded layer reproduces `ring_win_fuse` byte for byte.
            let (clen_p, topk) = if ph_ok {
                (
                    (self.s.clen.ptr as *const std::os::raw::c_int).wrapping_add(owner),
                    cfg.index_topk as i32,
                )
            } else {
                (std::ptr::null(), 0)
            };
            if self.dev.ring_win_fuse_ph(
                ring,
                self.s.kv.ptr as *const f32,
                self.s.pos_ctr.ptr as *const std::os::raw::c_int,
                win as i32,
                hd as i32,
                idxs_ptr as *mut i32,
                clen_p,
                topk,
            )? {
                ph_fused = ph_ok;
                true
            } else {
                // .so predates `dsv41_ring_win_fuse_ph`
                self.dev.ring_win_fuse(
                    ring,
                    self.s.kv.ptr as *const f32,
                    self.s.pos_ctr.ptr as *const std::os::raw::c_int,
                    win as i32,
                    hd as i32,
                    idxs_ptr as *mut i32,
                )?
            }
        };
        if !rw_fused && owns_kv {
            // DEVICE-side slot: a host-computed destination address would be frozen
            // by the graph capture (slot = pos % win at capture time), so every
            // replay wrote the same ring row and the window went stale.
            self.dev.ring_append(
                cache.ring.ptr as *mut f32,
                self.s.kv.ptr as *const f32,
                self.s.pos_ctr.ptr as *const std::os::raw::c_int,
                win as i32,
                hd as i32,
            )?;
        }

        // selection: the window ring, oldest first (the ring index already
        // carries the ageing rotation), padded with -1
        // The window row is `win` entries in ring order with empty slots marked
        // -1, and `sparse_attn` skips negatives. The earlier revision took only
        // the LEADING `pos+1` entries — but those are the *high* slots, which
        // are exactly the invalid ones while `pos < window` — so every decode
        // step selected nothing and the attention output came out identically
        // zero (confirmed by DSV41_STATS: rms=0.0000 at pos>0 while pos=0 was
        // fine, since only then do the leading entries happen to be valid).
        // The window indices are computed ON THE DEVICE from the position counter
        // (the decode branch of ops::window_topk_idxs, verbatim) - no host
        // compute and no per-layer H2D upload any more.
        // Compressed KV. Only the kv sources run the compressor; every other
        // layer of the same group reads the latents they published, which is
        // why they all live in this layer's own copy of the sequence's rows.
        let mut comp_len = self.layers[layer].compress_len;
        // COMPRESS_SIDE: when the compressor already ran on the third side
        // stream, its host-side counter mirror is already updated (inside
        // `compress_on`) and only the JOIN is left, placed just before the
        // compressor's first consumer below. `pending_comp_join` is false on
        // every serial path.
        let pending_comp_join = if cfg.compress_ratio(layer) > 0 && cfg.is_kv_source(layer) {
            if let Some(cl) = comp_len_side {
                comp_len = cl;
                true
            } else {
                comp_len = self.compress(layer, pos)?;
                false
            }
        } else if cfg.compress_ratio(layer) > 0 {
            // a consumer inherits the count published by its source layer
            comp_len = self.source_compress_len(layer);
            false
        } else {
            false
        };
        // Selection over the compressed rows. The release scores them with the
        // indexer and keeps `index_topk`; until the indexer is wired this takes
        // the most recent ones, which is a deliberate placeholder (it is a
        // superset-free pruning that at least makes the long-range rows
        // reachable — it is NOT the learned selection).
        // ALWAYS upload the window entries: the indexer overwrites the
        // compressed block on the device, but the window block [0, win) must be
        // fresh on every step. Only the placeholder branch uploaded them before,
        // so the index-source path read stale indices — the illegal memory
        // access in sparse_attn.
        // B2: the fused launch above already wrote the window indices (for every
        // layer, owner or not); only the fallback path writes them here.
        if !rw_fused {
            self.dev
                .window_idxs(idxs_ptr as *mut i32, self.s.pos_ctr.ptr as *const i32, win as i32)?;
        }
        // COMPRESS_SIDE join. The compressor's FIRST consumers are the indexer
        // below — a kv-source index layer reads this layer's `latent` (written
        // by `compressor_pool`) and the device latent counter it advances — and,
        // for every layer, `sparse_attn` (reads the ring's compressed rows +
        // that counter). `window_idxs` above reads neither (only `pos_ctr`), so
        // the join goes here: the latest point that is still before both, which
        // lets the compressor overlap `window_idxs` too. A no-op when the fork
        // did not take.
        if pending_comp_join {
            self.dev.compress_side_join()?;
        }
        if comp_len > 0 && cfg.is_index_source(layer) {
            // EVERY index-source layer runs its own indexer (into its own
            // buffer) — the reference creates one for each, and non-source
            // layers compute their own queries/selection from the keys the
            // kv-source published. Only the KEY PUBLISHING is the source's job.
            if self.indexer(layer, pos, win, comp_len)? {
            // the kernel wrote `comp_len.min(index_topk)` entries at [win, ..)
            }
        } else if !owns_kv && comp_len > 0 {
            // a non-index consumer reads the owner's selection, which the owner
            // (an index-source) filled earlier this step — do nothing
        } else if comp_len > 0 {
            // the owner has no indexer: recency placeholder (safety net)
            // B3: `ph_fused` means the `ring_win_fuse` launch above already wrote
            // these entries with the same device-derived bound, so the standalone
            // launch (and its graph node) is skipped. It stays on every path that
            // could not fold it: an age-less .so, `DSV41_COMP_PLACEHOLDER_FUSE=0`,
            // or a layer excluded by `ph_ok`.
            if !ph_fused {
                self.dev.comp_placeholder(
                    idxs_ptr as *mut i32,
                    (self.s.clen.ptr as *const std::os::raw::c_int).wrapping_add(owner),
                    win as i32,
                    cfg.index_topk as i32,
                )?;
            }
        }

        // P1 (DSV41_SPARSE_OROPE, default ON): the sparse attention's epilogue
        // also runs the inverse o-rope and emits the fp8 of the roped output
        // (exactly what the `apply_rope_q` + `quant1` pair below produced), in
        // ONE launch - two launches per layer (80 per step) disappear. The
        // fusion is geometry-preserving: `sparse_attn_pf_kernel` and the o-rope
        // call shape of `apply_rope_kernel` own the same block, so the emitted
        // bytes are bit-identical. A decline (or a missing symbol) falls back to
        // the three-launch sequence, which is why the second call is kept.
        let s_orope = sparse_orope()
            && self.dev.sparse_attn_orope(
                self.s.q.as_f32(),
                ring_ptr as *const f32,
                ld.attn_sink.as_ref().unwrap().as_f32(),
                idxs_ptr as *const i32,
                self.s.o.ptr as *mut f32,
                1,
                1,
                nlh as i32,
                hd as i32,
                (self.s.clen.ptr as *const std::os::raw::c_int).wrapping_add(owner),
                win as i32,
                cfg.index_topk as i32,
                1.0 / (hd as f32).sqrt(),
                self.cos.as_f32(),
                self.sin.as_f32(),
                self.s.pos_ctr.ptr as *const std::os::raw::c_int,
                cfg.rope_head_dim as i32,
                (cfg.rope_head_dim / 2) as i32,
                1,
                0,
                0,
                true,
                self.s.xq.ptr as *mut u8,
                self.s.xsc.ptr as *mut f32,
            )?;
        if !s_orope {
            self.dev.sparse_attn(
                self.s.q.as_f32(),
                ring_ptr as *const f32,
                ld.attn_sink.as_ref().unwrap().as_f32(),
                idxs_ptr as *const i32,
                self.s.o.ptr as *mut f32,
                1,
                1,
                nlh as i32,
                hd as i32,
                (self.s.clen.ptr as *const std::os::raw::c_int).wrapping_add(owner),
                win as i32,
                cfg.index_topk as i32,
                1.0 / (hd as f32).sqrt(),
            )?;
        }
        // B2 (DSV41_OROPE_Q, default ON): the inverse rope's epilogue emits the
        // fp8 of the whole `s.o` region (nlh*hd) in the same launch, which is
        // exactly what the `quant1(s.o)` below would have computed - so that
        // launch (one per layer per step) is skipped when the fused call took.
        // The rope pass touches only the trailing `rope_head_dim` lanes of each
        // head, so the emission is a second, warp-per-32-block pass over the
        // WHOLE flat region, bit-identical to dsv41_quant_fp8 (rows=1, block=32).
        let o_q_epi = !s_orope
            && orope_q()
            && self.dev.apply_rope_q(
                self.s.o.ptr as *mut f32,
                self.cos.as_f32(),
                self.sin.as_f32(),
                nlh as i32,
                hd as i32,
                cfg.rope_head_dim as i32,
                (cfg.rope_head_dim / 2) as i32,
                self.s.pos_ctr.ptr as *const std::os::raw::c_int, 1, 0,
                0,
                true,
                self.s.xq.ptr as *mut u8,
                self.s.xsc.ptr as *mut f32,
            )?;
        if !s_orope && !o_q_epi {
            self.dev.apply_rope(
                self.s.o.ptr as *mut f32,
                self.cos.as_f32(),
                self.sin.as_f32(),
                nlh as i32,
                hd as i32,
                cfg.rope_head_dim as i32,
                (cfg.rope_head_dim / 2) as i32,
                self.s.pos_ctr.ptr as *const std::os::raw::c_int, 1, 0,
                0,
                true,
            )?;
        }

        if layer == 0 && hc_dbg() {
            let o = self.dl(self.s.o.as_f32(), nlh * hd)?;
            let r = (o.iter().map(|v| v * v).sum::<f32>() / o.len() as f32).sqrt();
            eprintln!("[mine] L0 sparse_o[0..4]={:?} sparse_rms={}", &o[..4], r);
        }
        // block-diagonal grouped output projection: group g owns rows
        // [g*o_lora, (g+1)*o_lora) against the head slice [g*hpg*hd, ...)
        let groups = cfg.o_groups;
        let hpg = nh / groups;
        // `o_lora_rank` is the PER-GROUP low-rank width (the reference's
        // wo_a weight is [n_groups * o_lora_rank, hpg*head_dim] viewed as
        // [n_groups, o_lora_rank, hpg*head_dim]), so a group's row block is
        // o_lora_rank tall — not o_lora_rank/groups.
        let olg = cfg.o_lora_rank;
        let nlg = groups / world; // wo_a is ColumnParallel: a block of groups each
        let k = hpg * hd;
        // B2/P1: the o-rope epilogue above already emitted `s.xq`/`s.xsc` from the
        // rotated `s.o` (or, with SPARSE_OROPE, the fused sparse kernel did);
        // only the fallback path still quantises here.
        if !s_orope && !o_q_epi {
            self.quant1(self.s.o.ptr as *const f32, (nlh * hd) as i32)?;
        }
        // B1 (DSV41_WO_QUANT_FUSE): the wo_a gemv's epilogue emits the fp8 of its
        // own output into `wo_q`/`wo_qsc` with quant_kernel's arithmetic, so the
        // `quant1(s.wo)` launch that used to sit between the two projections (one
        // per layer, 40 per step) disappears and wo_b reads `wo_q` instead of
        // `xq`. A separate buffer is REQUIRED, not a convenience: this gemv is
        // READING `xq` (the quantised attention output) as its input, so writing
        // the output fp8 back into it would race with the blocks that stage that
        // input late. `gemm_fp8_mx_q` returns false when the shape cannot take the
        // fused path (n % 32 != 0, or `mode` < 3 with no dynamic shared memory),
        // and we then run the plain call for every group; the emitted bytes are
        // bit-identical either way, which is what makes the fallback safe.
        let wo_fuse = Self::wo_quant_fuse() && (olg % 32) == 0;
        // wo_b is RowParallel: the input (groups*o_lora) is split, so this rank
        // reduces over its own slice and the ranks' partial sums are added.
        let ol_total = groups * cfg.o_lora_rank;
        let ol_local = ol_total / world;
        // this rank wrote its groups at local offsets [0, nlg*olg) = [0, ol_local)
        // == lin()'s quant1 half, hoisted so the fp8 activation exists before the
        // direct gemm_fp8_mx below (lin() would re-quantise, and cannot pass the AR
        // staging args). Under AR v5 the epilogue ALSO stores the row partial into
        // every peer's slot, so the following all_reduce becomes publish+reduce
        // only -- the standalone store kernel disappears for this site.
        // (Hoisted above the wo_a loop by chain-pair-grid-sync: the fused pair call
        // replaces BOTH projections, so it must be able to run before the loop.)
        let comm = self.comm.clone();
        // AR store fusion: gated OFF by default (round 19 showed the fused path
        // breaks the four texts even with GATEUP/DOWN_FUSE off, because this
        // changes wo_b's all_reduce to pubred-only which depends on the gemv
        // epilogue having stored the partials - a coupling that must be debugged
        // together with the GATEUP_FUSE numerical bug). DSV41_AR_STORE_FUSE=1
        // re-enables.
        let ar_store_fused = ar_store_fuse()
            && comm.as_ref().map(|c| c.uses_v5()).unwrap_or(false);
        // ---- chain-pair-grid-sync (DSV41_WO_PAIR, default OFF) ---------------
        // The (wo_a, wo_b) pair as ONE grid-sync launch. Its two phases run the
        // same GEMV bodies in the same lane order, so every row of `s.wo` and
        // `s.o` is bit-identical to what the two calls below produce; what is saved
        // is one launch + one graph node per layer (40/step).
        //
        // The kernel implements ONE arm -- wo_a as the fp8 GEMV (mode 4 + a32,
        // which is the default mode) and wo_b as the f32-activation GEMV -- and
        // one group per rank (its phase 1 stages a single block-wide activation
        // row). Every other combination is declined, so these guards mirror the
        // kernel's specialisation exactly.
        let mut wo_paired = false;
        if Self::wo_pair_fuse()
            && !wo_fuse                    // B1 epilogue is not wired into the pair
            && !ar_store_fused             // AR-store epilogue is not either
            && nlg == 1                    // one group: one staged activation row
            && (k % 16) == 0               // cp.async weight staging + LUT decode
            && (ol_local % 32) == 0        // phase-1 weight scale rows in 32-row blocks
            && Self::wob_f32()             // the pair's phase 2 IS the f32 form
            && self.dev.supports_wo_pair()
        {
            wo_paired = self.dev.gemm_fp8_wo_pair(
                self.s.xq.as_u8(),
                self.s.xsc.as_f32(),
                ld.wo_a.as_ref().unwrap().as_u8(),
                ld.wo_a_scale.as_ref().unwrap().as_u8(),
                std::ptr::null(),
                olg as i32,
                k as i32,
                ld.wo_b.as_ref().unwrap().as_u8(),
                ld.wo_b_scale.as_ref().unwrap().as_u8(),
                std::ptr::null(),
                dim as i32,
                ol_local as i32,
                self.s.wo.ptr as *mut f32,
                self.s.o.ptr as *mut f32,
                self.s.wo_bar.ptr as *mut u32,
            )?;
        }
        let mut wo_fused = false;
        if !wo_paired {
        for g in 0..nlg {
            // The weight tensor is ALREADY the rank's local slice (Shard::Groups
            // cut it at load time), so every offset must be LOCAL: group g of
            // this rank's block sits at local row g*olg. The earlier version
            // indexed with the GLOBAL group number (rank*nlg+g), which walks off
            // the end of the local buffer for any rank but 0 — the illegal
            // memory access in gemm_fp8_mx at tp=8.
            let a = self.s.xq.as_u8().wrapping_add(g * k);
            let asc = self.s.xsc.as_f32().wrapping_add((g * k / 32) as usize);
            let wp = ld.wo_a.as_ref().unwrap().as_u8().wrapping_add(g * olg * k);
            let wsp = ld
                .wo_a_scale
                .as_ref()
                .unwrap()
                .as_u8()
                .wrapping_add((g * olg / 32) * (k / 32));
            let out = (self.s.wo.ptr as *mut f32).wrapping_add(g * olg);
            // The decline is a shape/`mode` property, so group 0 decides for the
            // whole loop; a later group rides on the decision it produced.
            if wo_fuse && (g == 0 || wo_fused) {
                let ok = self.dev.gemm_fp8_mx_q(
                    a,
                    asc,
                    wp,
                    wsp,
                    std::ptr::null(),
                    out,
                    1,
                    olg as i32,
                    k as i32,
                    (self.s.wo_q.ptr as *mut u8).wrapping_add(g * olg),
                    (self.s.wo_qsc.ptr as *mut f32).wrapping_add(g * olg / 32),
                )?;
                if ok {
                    wo_fused = true;
                    continue;
                }
            }
            self.gemm_fp8_mx_or_swap(a, asc, wp, wsp, std::ptr::null(), out, olg as i32, k as i32)?;
        }
        }
        // wo_b's activation, in priority order:
        //  1. B1 (wo_fused): the wo_a epilogue already emitted `wo_q`/`wo_qsc`, so
        //     the quant1 that used to run here is skipped.
        //  2. DSV41_WOB_F32 (default ON): the wo_b gemv reads the RAW f32 `s.wo`
        //     directly, so the `quant1(s.wo)` launch disappears. It keeps the
        //     normal gemv grid shape (g_gemv_warps / ceil(n/warps)), which is why
        //     it avoids B1's +0.24ms 32-warp SM-utilisation loss — only the data
        //     path changes.
        //  3. fallback: `quant1(s.wo)` into `xq`/`xsc`, then the fp8 gemv.
        // The f32 path cannot carry the AR store fusion (that lives on the fp8
        // launcher's epilogue), so it is only taken when `ar_store_fused` is off.
        let mut wb_f32 = false;
        if !wo_paired && !wo_fused && !ar_store_fused && Self::wob_f32() && self.dev.supports_gemm_fp8_f32() {
            wb_f32 = self.dev.gemm_fp8_mx_f32(
                self.s.wo.ptr as *const f32,
                ld.wo_b.as_ref().unwrap().as_u8(),
                ld.wo_b_scale.as_ref().unwrap().as_u8(),
                std::ptr::null(),
                self.s.o.ptr as *mut f32,
                dim as i32,
                ol_local as i32,
            )?;
        }
        if !wo_paired && !wb_f32 {
            let (wb_q, wb_sc) = if wo_fused {
                (self.s.wo_q.as_u8(), self.s.wo_qsc.as_f32())
            } else {
                self.quant1(self.s.wo.ptr as *const f32, ol_local as i32)?;
                (self.s.xq.as_u8(), self.s.xsc.as_f32())
            };
            if ar_store_fused {
                let c = comm.as_ref().unwrap();
                self.dev.gemm_fp8_mx_ar(
                    wb_q,
                    wb_sc,
                    ld.wo_b.as_ref().unwrap().as_u8(),
                    ld.wo_b_scale.as_ref().unwrap().as_u8(),
                    self.s.o.ptr as *mut f32,
                    dim as i32,
                    ol_local as i32,
                    c.peer_slots_f32(),
                    c.epoch_u32(),
                    c.world as i32,
                    c.rank as i32,
                    c.slot_stride_elems(),
                )?;
            } else {
                self.gemm_fp8_mx_or_swap(
                    wb_q,
                    wb_sc,
                    ld.wo_b.as_ref().unwrap().as_u8(),
                    ld.wo_b_scale.as_ref().unwrap().as_u8(),
                    std::ptr::null(),
                    self.s.o.ptr as *mut f32,
                    dim as i32,
                    ol_local as i32,
                )?;
            }
        }
        let mut hc_folded = false;
        if let Some(c) = comm {
            if ar_store_fused {
                // store already done by the fused epilogue; publish+reduce only.
                // Must stay adjacent to the gemv above (same *epoch).
                c.all_reduce_inplace_pubred_only(self.s.o.ptr as *mut std::ffi::c_void, fb(dim))?;
            } else {
                // Segment-C fold: when it runs, the pubred epilogue has already
                // written THIS layer's hc_post onto the residual stream, so
                // `layer()` skips the standalone launch. The fused entry carries
                // its own store, which is why it cannot coexist with
                // `ar_store_fused` (that path has no store left to skip).
                hc_folded = self.ar_hc_post_fold(
                    &c,
                    self.s.o.ptr as *mut std::ffi::c_void,
                    fb(dim),
                    None,
                )?;
                if !hc_folded {
                    c.all_reduce_inplace(self.s.o.ptr as *mut std::ffi::c_void, fb(dim))?;
                }
            }
            c.end_round();
        }
        Ok(hc_folded)
    }

    /// Indexer for one decode step: publish this layer's index key for the
    /// latent the compressor just produced, then score the published keys and
    /// let the kernel write the top-k straight into the selection buffer at
    /// `offset` (`window`, so it lands after the window block).
    /// Returns false when there was nothing to select from.
    fn indexer(&mut self, layer: usize, _pos: usize, offset: usize, comp_len: usize) -> Result<bool> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let ql = cfg.q_lora_rank;
        let idx_nh = cfg.index_n_heads;
        let idx_hd = cfg.index_head_dim;
        let rd = cfg.rope_head_dim;
        let ratio = cfg.compress_ratio(layer).max(1);
        let ld = &self.w.layers[layer];
        let (Some(wq_b), Some(wq_b_s), Some(wk), Some(kn), Some(wp)) = (
            ld.idx_wq_b.as_ref(),
            ld.idx_wq_b_scale.as_ref(),
            ld.idx_wk.as_ref(),
            ld.idx_k_norm.as_ref(),
            ld.idx_weights.as_ref(),
        ) else {
            return Ok(false);
        };
        // Only kv-source layers publish index keys (they own the compressor's
        // latent). Non-source index layers compute their own queries and
        // selection from the keys their source already published.
        let owns_k = cfg.indexer_owns_k(layer);
        let group = if owns_k {
            self.layers[layer].compress_len.saturating_sub(1)
        } else {
            0
        };
        if owns_k {
        self.lin_bf16(
            self.layers[layer].latent.ptr as *const f32,
            cfg.head_dim as i32,
            wk,
            idx_hd as i32,
            self.s.idx_k.ptr as *mut f32,
        )?;
        self.dev.rmsnorm(
            self.s.idx_k.ptr as *const f32,
            kn.as_f32(),
            self.s.idx_k.ptr as *mut f32,
            1,
            idx_hd as i32,
            cfg.norm_eps,
        )?;
        self.dev.apply_rope(
            self.s.idx_k.ptr as *mut f32,
            self.cos.as_f32(),
            self.sin.as_f32(),
            1,
            idx_hd as i32,
            rd as i32,
            (rd / 2) as i32,
            (self.s.clen.ptr as *const std::os::raw::c_int).wrapping_add(layer),
            ratio as i32,
            -(ratio as i32),
            1,
            false,
        )?;
        // DEVICE-derived destination: a host-computed group slot here is exactly
        // the frozen-address bug that made the window ring go stale under the
        // graph (the index key would land in the same group every replay).
        self.dev.index_k_publish(
            self.layers[layer].index_k.ptr as *mut f32,
            self.s.idx_k.ptr as *const f32,
            (self.s.clen.ptr as *const std::os::raw::c_int).wrapping_add(layer),
            idx_hd as i32,
        )?;
        } // end owns_k (key publishing only)
        // the queries come from the q_lora stream. Skipped when attention()
        // already produced s.idx_q in the SAME mx2 launch as wq_b
        // (DSV41_IDX_FUSE); the one-shot flag is consumed here either way, so a
        // non-index-source layer's stale value can never leak into the next step.
        // DSV41_ROPE_FUSE: whichever launch produced s.idx_q can also have rotated
        // it - `lin_rope` here, the attention mx2's family-2 epilogue there - so
        // the standalone rope below runs only when neither did.
        // NORM_FUSE: `attention()` left `qr` RAW when the wq_b launch took the
        // fused path, so this idx_wq_b projection must normalise + quantise `qr`
        // in its own gemv prologue as well - `quant1(qr)` over a raw row would
        // index entirely the wrong values. The flag is consumed here either way,
        // so a layer that never reaches the indexer cannot leak it.
        let qr_is_raw = self.s.qr_raw.get();
        self.s.qr_raw.set(false);
        let idx_q_roped = if !self.s.idx_q_ready.get() {
            let mut roped = false;
            if qr_is_raw {
                roped = self.lin_rope_norm(
                    self.s.qr.ptr as *const f32,
                    ld.q_norm.as_ref().unwrap().as_f32(),
                    cfg.norm_eps,
                    ql as i32,
                    wq_b,
                    wq_b_s,
                    (idx_nh * idx_hd) as i32,
                    self.s.idx_q.ptr as *mut f32,
                    rd as i32,
                    idx_hd as i32,
                )?;
                if !roped {
                    // The fused launcher declined on a shape the plain rope
                    // launcher would reject too; materialise the norm first so the
                    // fallbacks below still quantise NORMALISED values.
                    self.dev.rmsnorm(
                        self.s.qr.ptr as *const f32,
                        ld.q_norm.as_ref().unwrap().as_f32(),
                        self.s.qr.ptr as *mut f32,
                        1,
                        ql as i32,
                        cfg.norm_eps,
                    )?;
                }
            }
            if !roped {
                roped = self.lin_rope(
                    self.s.qr.ptr as *const f32,
                    ql as i32,
                    wq_b,
                    wq_b_s,
                    (idx_nh * idx_hd) as i32,
                    self.s.idx_q.ptr as *mut f32,
                    rd as i32,
                    idx_hd as i32,
                )?;
            }
            if !roped {
                self.lin(
                    self.s.qr.ptr as *const f32,
                    ql as i32,
                    wq_b,
                    wq_b_s,
                    (idx_nh * idx_hd) as i32,
                    self.s.idx_q.ptr as *mut f32,
                )?;
            }
            roped
        } else {
            self.s.idx_q_rope.get()
        };
        self.s.idx_q_ready.set(false);
        self.s.idx_q_rope.set(false);
        if !idx_q_roped {
            self.dev.apply_rope(
                self.s.idx_q.ptr as *mut f32,
                self.cos.as_f32(),
                self.sin.as_f32(),
                idx_nh as i32,
                idx_hd as i32,
                rd as i32,
                (rd / 2) as i32,
                self.s.pos_ctr.ptr as *const std::os::raw::c_int, 1, 0,
                0,
                false,
            )?;
        }
        // per-head weights; the reference folds softmax_scale * n_heads^-0.5 into
        // them, and our kernel applies softmax_scale * head_scale to the sum, so
        // the same factor can be passed there instead
        self.lin_bf16(
            self.s.xn.ptr as *const f32,
            dim as i32,
            wp,
            idx_nh as i32,
            self.s.idx_w.ptr as *mut f32,
        )?;
        // No upload: the indexer reads the length straight from the device
        // counter the compressor's commit kernel maintains (this was the last
        // per-step H2D inside the attention path).
        // The INDEXER's scale uses index_head_dim (128), not the attention head_dim
        // (512): the reference sets `self.softmax_scale = index_head_dim**-0.5`
        // for the Indexer and folds `n_heads**-0.5` in with the per-head weights.
        // Using head_dim made every index score 2x too small, so the top-k
        // selection picked the wrong compressed positions.
        let scale = 1.0f32 / (idx_hd as f32).sqrt() / (idx_nh as f32).sqrt();
        // keys live on the KV OWNER's buffer (the source layer that published
        // them); a non-source index layer's own index_k is empty
        let key_owner = self.kv_owner(layer);
        let idx_lens_ptr =
            (self.s.clen.ptr as *const std::os::raw::c_int).wrapping_add(key_owner);
        self.dev.indexer_topk(
            self.s.idx_q.as_f32(),
            self.layers[key_owner].index_k.as_f32(),
            self.s.idx_w.as_f32(),
            std::ptr::null(),
            idx_lens_ptr,
            // the kernel writes `picked + offset` into out[row*cols + i], so
            // `out` points at the first compressed slot of the row
            (self.layers[layer].idxs.ptr as *mut i32).wrapping_add(offset),
            1,
            1,
            idx_nh as i32,
            idx_hd as i32,
            // The LIVE compressed count again (what this passed before the constant
            // existed). It is only the FALLBACK now - the kernel prefers the device
            // counter it receives as `lens`, and its shared memory is sized from the
            // fixed chunk instead of from this value, so a graph replay stays exact
            // and no candidate can be lost to a frozen bound.
            comp_len as i32,
            cfg.index_topk as i32,
            offset as i32,
            scale,
            1.0,
            false,
        )?;
        Ok(true)
    }

    /// Compressor for one decode step: the two projections in fp32 (the release
    /// promotes them), then the pooling half, then the latent lands in this
    /// layer's KV buffer at row `window + compress_len`. Returns the new count.
    fn compress(&mut self, layer: usize, pos: usize) -> Result<usize> {
        self.compress_on(layer, pos, self.dev.stream())
    }

    /// [`Self::compress`] with every launch issued on `s`. The serial path
    /// passes the main stream (byte-for-byte the old behaviour); the
    /// `DSV41_COMPRESS_SIDE` path passes the third side stream so the four
    /// launches overlap the q/kv chains that own the main stream in the same
    /// window. Kernels and operands are identical, so the two streams produce
    /// bit-identical bytes.
    ///
    /// The compressor reads `s.xn` (read-only), the device position counter and
    /// this layer's own state, and writes only this layer's private buffers plus
    /// the ring's compressed rows and this layer's device counter — never
    /// `s.qr`/`s.q`/`s.xq`/`s.xsc` (q chain) nor `s.kv` (kv chain). That
    /// disjointness is what makes the concurrent issue legal.
    fn compress_on(
        &mut self,
        layer: usize,
        pos: usize,
        s: ferrite_kernel::devrt::CuStream,
    ) -> Result<usize> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let hd = cfg.head_dim;
        let ratio = cfg.compress_ratio(layer).max(1);
        let ld = &self.w.layers[layer];
        let (Some(wkv), Some(norm)) = (ld.comp_wkv.as_ref(), ld.comp_norm.as_ref()) else {
            return Ok(self.layers[layer].compress_len);
        };
        let cache = &self.layers[layer];
        self.lin_f32_on(self.s.xn.ptr as *const f32, dim as i32, wkv, hd as i32, cache.kvp.ptr as *mut f32, s)?;
        if let Some(wg) = ld.comp_wgate.as_ref() {
            self.lin_f32_on(self.s.xn.ptr as *const f32, dim as i32, wg, hd as i32, cache.scp.ptr as *mut f32, s)?;
        } else {
            // ratio == 1: no gate; the pooling reduces to the plain projection
            self.dev.zero_on(&cache.scp, s)?;
        }
        // COMPRESS_FUSE (default ON): state carry + pool + commit as ONE launch.
        // Decode-shape gate only: the fused kernel carries the `start_pos > 0`
        // state mapping and the `mode 2` pool, so `ratio == 1` (no gate, no
        // state) and the first step (`pos == 0`) keep the three-launch sequence
        // below — which is bit-identical, so the gate costs nothing but a branch
        // that the captured graph bakes in.
        if compress_fuse() && ratio > 1 && pos > 0 && self.dev.supports_compress_fuse() {
            self.dev.compressor_fused_on(
                cache.kvp.as_f32(),
                cache.scp.as_f32(),
                norm.as_f32(),
                cache.state_kv.ptr as *mut f32,
                cache.state_score.ptr as *mut f32,
                cache.latent.ptr as *mut f32,
                cache.out_rows.ptr as *mut i32,
                self.cos_comp.as_f32(),
                self.sin_comp.as_f32(),
                cache.ring.ptr as *mut f32,
                (self.s.clen.ptr as *mut std::os::raw::c_int).wrapping_add(layer),
                1,
                1,
                hd as i32,
                ratio as i32,
                cfg.rope_head_dim as i32,
                (cfg.rope_head_dim / 2) as i32,
                cfg.window_size as i32,
                self.s.pos_ctr.ptr as *const std::os::raw::c_int,
                cfg.norm_eps,
                s,
            )?;
        } else {
            self.dev.compressor_pool_on(
                cache.kvp.as_f32(),
                cache.scp.as_f32(),
                norm.as_f32(),
                cache.state_kv.ptr as *mut f32,
                cache.state_score.ptr as *mut f32,
                cache.latent.ptr as *mut f32,
                cache.out_rows.ptr as *mut i32,
                1,
                1,
                hd as i32,
                ratio as i32,
                pos as i32,
                self.s.pos_ctr.ptr as *const std::os::raw::c_int,
                cfg.norm_eps,
                s,
            )?;
            // FUSED COMMIT (device-side): reads out_rows on the device, ropes the
            // latent, stores it into the ring and advances this layer's device
            // counter. The old form downloaded out_rows (a sync D2H per layer per
            // step), branched on the host, roped, copied and bumped a host counter -
            // all impossible to capture in a graph.
            self.dev.compress_commit_on(
                cache.latent.as_f32(),
                self.cos_comp.as_f32(),
                self.sin_comp.as_f32(),
                cache.ring.ptr as *mut f32,
                cache.out_rows.ptr as *const std::os::raw::c_int,
                (self.s.clen.ptr as *mut std::os::raw::c_int).wrapping_add(layer),
                hd as i32,
                cfg.rope_head_dim as i32,
                (cfg.rope_head_dim / 2) as i32,
                cfg.window_size as i32,
                ratio as i32,
                s,
            )?;
        }
        // The host keeps a MIRROR of the device counter using the SAME deterministic
        // rule the kernel applies ((pos + 1) % ratio == 0 commits one latent). The
        // host branches on this (whether to run the indexer, how many compressed
        // slots to expect) and the kernels read the device counter itself - so the
        // two agree by construction, without the download that used to be here.
        if (pos + 1) % ratio == 0 {
            self.layers[layer].compress_len += 1;
        }
        Ok(self.layers[layer].compress_len)
    }

    /// The layer whose KV store `layer` reads: itself, unless it is a consumer
    /// of a group whose compressed KV is maintained by the source above it.
    fn kv_owner(&self, layer: usize) -> usize {
        if self.cfg.kv_mode(layer) != KvMode::CompressConsumer {
            return layer;
        }
        for l in (0..layer).rev() {
            if self.cfg.kv_mode(l) == KvMode::CompressSource
                && self.cfg.compress_ratio(l) == self.cfg.compress_ratio(layer)
            {
                return l;
            }
        }
        layer
    }

    /// How many compressed rows the source layer of `layer` has published.
    /// Consumers share the source's cache, so they report the same count.
    fn source_compress_len(&self, layer: usize) -> usize {
        // the config lists compressors per layer; a consumer reads the most
        // recent source at or before it (the reference's kv_source mapping)
        for l in (0..=layer).rev() {
            if self.cfg.is_kv_source(l) && self.cfg.compress_ratio(l) == self.cfg.compress_ratio(layer) {
                return self.layers[l].compress_len;
            }
        }
        0
    }

    /// MoE: bf16 gate GEMM, `noaux_tc` routing, MXFP4 experts, fp8 shared expert.
    fn moe(&mut self, layer: usize, ld: &LayerDev) -> Result<()> {
        let cfg = self.cfg;
        // ADD_EPI: cleared here and set only by the deferral below (see
        // `add_epi()`); `moe_reduce(layer)` reads it right after this segment.
        self.moe_add_in[layer] = None;
        let dim = cfg.dim;
        let inter = cfg.moe_inter_dim;
        // MoE is TP-split, NOT expert-parallel: every rank holds every expert,
        // and each expert's `inter` axis is cut by world. The slice is padded up
        // to the MMA K atom (64) with zeros by the loader, so the kernels are
        // sized by the padded local width.
        let inter_local = crate::dsv41::weights::padded_inter(inter / self.world());
        let (n_routed, topk) = cfg.moe_config(layer);

        // gate: natively bf16, so a bf16 GEMM. When this rank owns the shared
        // expert, the shared expert's w1/w3 read the SAME `xn`, so gate + w1 + w3
        // share one launch and save one ~20us fixed launch floor per layer. Each
        // row keeps its own family's lane order and accumulation, so all three
        // outputs are bit-identical to the separate launches. DSV41_MIX_GATE=0
        // (or an .so without the symbol) falls back.
        // Shared expert. Under DSV41_SHARED_TP every rank computes its own
        // inter/world slice (the weights are sharded to match, see weights.rs) and
        // the MoE all-reduce below sums the partials - the routed experts'
        // structure, which removes rank 0's ~55us x 40 layers of serial work that
        // the other seven ranks waited on. Default: the historical replicated
        // layout, computed on rank 0 alone.
        let stp = crate::dsv41::weights::shared_expert_tp();
        let sh_il = if stp { inter / self.world() } else { inter };
        let shared_rank = if stp {
            true
        } else {
            self.comm.as_ref().map(|c| c.rank == 0).unwrap_or(true)
        };
        // Only the rank(s) that actually contribute the shared expert take it: all
        // of them under DSV41_SHARED_TP (each owns its inter/world slice), rank 0
        // alone under the replicated layout. Without this the fused gate+shared
        // launch would run on every rank in both cases - correct but 7/8 wasted.
        let sh_w = if !self.opts.skip_shared_expert && shared_rank {
            match (
                ld.shared_w1.as_ref(),
                ld.shared_w1_scale.as_ref(),
                ld.shared_w3.as_ref(),
                ld.shared_w3_scale.as_ref(),
            ) {
                (Some(a), Some(b), Some(c), Some(d)) => Some((a, b, c, d)),
                _ => None,
            }
        } else {
            None
        };
        let mut sh_via_mixed = false;
        // The shared half's down tensors (w2/w2s) live outside `sh_w`; whether
        // they exist decides if that half runs at all, so MOE_DUAL must know it
        // before it forks.
        let sh_w2_ok = ld.shared_w2.is_some() && ld.shared_w2_scale.is_some();
        // MOE_DUAL (DSV41_MOE_DUAL, default ON): fork the SHARED expert half onto
        // the second side stream HERE, before the gate/routed chain is issued.
        // The fork event is recorded on the main stream at a point where `xn`
        // (and the T1 fp8 of `xn`, when the hc tail emitted it) is already
        // complete, so the shared half reads exactly what the serial code read -
        // it just no longer WAITS for the gate + routed experts in between.
        //
        // The predicate mirrors `batched` below (the routed path must be the
        // batched one: the sequential loop reuses `s.ex_act`, which the shared
        // half writes). `!mix_gate_shared()` is required because the mixed
        // gate+w1+w3 launch is a main-stream writer of `s.xq`/`s.ex_act` that the
        // shared half would consume. See `moe_dual()` for the full contract.
        let dual = moe_dual()
            && !mix_gate_shared()
            && !self.opts.skip_experts
            && sh_w.is_some()
            // `sh_w` covers w1/w3 only; the shared half also needs w2 to run at
            // all. Without it the block below is skipped, yet the join would still
            // merge `ex_out` — a stale buffer — into `s.o`.
            && sh_w2_ok
            && moe_batch()
            && topk > 0
            && ld.experts.len() >= 2
            && self.dev.supports_moe_batch()
            && self.dev.supports_dual_chain();
        if dual {
            self.dev.dual_chain_fork()?;
        }
        // The stream the shared half's launches go to: the side stream under
        // MOE_DUAL, the main stream (byte-for-byte the old order) otherwise.
        let sh_st = if dual { self.dev.side_stream2() } else { self.dev.stream() };
        if let Some((w1, w1s, w3, w3s)) = sh_w {
            // T2: only the MIX_GATE path reads the fp8 of `xn` here. With
            // MIX_GATE off this quantisation has no consumer at all (the mixed
            // launch below is skipped), yet it still CONSUMED the T1 flag - which
            // is exactly why the shared expert's own quant1(xn) below (`if
            // !sh_via_mixed`) had to re-quantise. Gating it on the same cached
            // predicate the launch uses
            // keeps the flag alive for the shared expert.
            if mix_gate_shared() {
                self.quant1(self.s.xn.ptr as *const f32, dim as i32)?;
            }
            sh_via_mixed = mix_gate_shared()
                && self.dev.gemm_bf16_fp8x2(
                    ld.gate_w.as_ref().unwrap().ptr() as *const c_void,
                    std::ptr::null(),
                    self.s.scores.ptr as *mut f32,
                    n_routed as i32,
                    self.s.xq.as_u8(),
                    self.s.xsc.as_f32(),
                    w1.as_u8(),
                    w1s.as_u8(),
                    self.s.ex_act.ptr as *mut f32,
                    sh_il as i32,
                    w3.as_u8(),
                    w3s.as_u8(),
                    (self.s.ex_act.ptr as *mut f32).wrapping_add(sh_il),
                    self.s.xn.ptr as *const f32,
                    dim as i32,
                )?;
        }
        let mut routed = false;
        if !sh_via_mixed {
            // Fused gate GEMV + route: the gate's LAST block runs the top-6
            // selection over the 384 finished scores (DSV41_ROUTE_FUSE, default
            // ON) — one launch where gate + route_topk used to be two, and
            // bit-exact (device.rs::gemv_bf16_route). `routed == false` means
            // "not fused" (symbol absent / shape mismatch / gate not a bf16
            // GEMV), and the two-launch pair below still runs.
            // cuBLAS-M1 is excluded: the fusion is a GEMV-v2 launch, and an A/B
            // with DSV41_CUBLAS_M1=1 must keep its cuBLAS gate.
            if route_fuse() && !cublas_m1() {
                routed = self.dev.gemv_bf16_route(
                    ld.gate_w.as_ref().unwrap().ptr() as *const c_void,
                    self.s.xn.ptr as *const f32,
                    self.s.scores.ptr as *mut f32,
                    n_routed as i32,
                    dim as i32,
                    self.s.route_w.ptr as *mut f32,
                    self.s.route_idx.ptr as *mut i32,
                    ld.gate_bias.as_ref().map(|b| b.as_f32()).unwrap_or(std::ptr::null()),
                    topk as i32,
                    cfg.norm_topk_prob,
                    cfg.route_scale,
                    2, // sqrtsoftplus, per the checkpoint's routing
                    self.s.route_ctr.ptr as *mut std::ffi::c_uint,
                )?;
            }
            if !routed {
                self.lin_bf16(
                    self.s.xn.ptr as *const f32,
                    dim as i32,
                    ld.gate_w.as_ref().unwrap(),
                    n_routed as i32,
                    self.s.scores.ptr as *mut f32,
                )?;
            }
        }
        if !routed {
            self.dev.route_topk(
                self.s.scores.as_f32(),
                ld.gate_bias.as_ref().map(|b| b.as_f32()).unwrap_or(std::ptr::null()),
                self.s.route_w.ptr as *mut f32,
                self.s.route_idx.ptr as *mut i32,
                // P1: `hist` is dead code - the kernel only checks it for non-null to
                // issue an extra memset + route_hist_kernel per call, and no reader
                // anywhere consumes the histogram. Passing null drops 2 device ops
                // per layer (80/step) with zero numerical impact.
                std::ptr::null_mut(),
                1,
                n_routed as i32,
                topk as i32,
                cfg.norm_topk_prob,
                cfg.route_scale,
                2, // sqrtsoftplus, per the checkpoint's routing
            )?;
        }
        // DEVICE-side dispatch: the routing stays on the device and the expert
        // kernels read `ids[slot]` themselves. Both downloads here were blocking
        // cudaMemcpy calls (download_f32 uses the synchronous memcpy), i.e. a
        // per-layer host stall; they are gone, and the launch arguments no longer
        // depend on the routing (which is what a CUDA graph needs).
        let (idx, wgt) = (Vec::<i32>::new(), Vec::<f32>::new());

        // One token: every assignment shares the input row, so expert e's total
        // contribution is expert_e(x) * sum of its routing weights. Accumulating
        // the distinct experts in index order keeps the sum deterministic.
        // `expert_down_fp4` OVERWRITES its output (launch_mxf4 writes
        // out[row*n + col] = x), so the scratch AND the accumulator both start
        // from zero — `o` still held the attention output at this point, and the
        // memcpy that used to follow the loop replaced the finished sum with the
        // last expert's contribution alone.
        // P2: both zeros are redundant in the DEFAULT batched path (moe_down_reduce
        // OVERWRITES `o`: out[i] = acc), and zeroing `o` there would destroy the
        // attention residual AR#1 just summed into it. The sequential path below
        // still needs both zeros as its add base.
        let ne = ld.experts.len();
        let batched = moe_batch() && topk > 0 && ne >= 2 && self.dev.supports_moe_batch();
        // The routed experts' gate/up pools may be stored INTERLEAVED
        // (DSV41_EXPERT_ILV, decided at load time). Only the FUSED batched
        // gate/up read can address that layout, so the batched path stops being
        // optional: refuse loudly instead of running the sequential or unfused
        // fallback against bytes laid out for another reader.
        let ilv = ld.experts_ilv;
        if ilv && !batched {
            return Err(FerriteError::Config(
                "routed expert gate/up weights are interleaved (DSV41_EXPERT_ILV) but the \
                 batched MoE path is unavailable — run with DSV41_EXPERT_ILV=0, or restore \
                 DSV41_MOE_BATCH/DSV41_GATEUP_FUSE so the fused batched call is used"
                    .into(),
            ));
        }
        if !batched {
            self.dev.zero(&self.s.ex_out)?;
            self.dev.zero(&self.s.o)?;
        }
        let wsum = vec![0f32; ne.max(1)];
        if moe_dbg() {
            eprintln!("[mine] route idx={:?} wgt={:?}", &idx, &wgt);
        }
        // The experts' tensors are views into one per-layer pool with a uniform
        // per-expert stride, so the kernels can derive every pointer from a base
        // plus `ids[slot] * stride`. Taking the stride as the difference of two
        // experts' pointers keeps this correct for whatever layout the loader
        // chose.
        let (w1_base, w1_stride, w1s_base, w1s_stride, w3_base, w3_stride, w3s_base, w3s_stride) =
            if ne >= 2 {
                let (a, b) = (&ld.experts[0], &ld.experts[1]);
                let d = |x: *mut std::ffi::c_void, y: *mut std::ffi::c_void| {
                    (y as i64) - (x as i64)
                };
                (
                    a.w1.ptr() as *const u8, d(a.w1.ptr(), b.w1.ptr()),
                    a.w1_scale.ptr() as *const u8, d(a.w1_scale.ptr(), b.w1_scale.ptr()),
                    a.w3.ptr() as *const u8, d(a.w3.ptr(), b.w3.ptr()),
                    a.w3_scale.ptr() as *const u8, d(a.w3_scale.ptr(), b.w3_scale.ptr()),
                )
            } else {
                (std::ptr::null(), 0, std::ptr::null(), 0, std::ptr::null(), 0, std::ptr::null(), 0)
            };
        let (w2_base, w2_stride, w2s_base, w2s_stride) = if ne >= 2 {
            let (a, b) = (&ld.experts[0], &ld.experts[1]);
            let d = |x: *mut std::ffi::c_void, y: *mut std::ffi::c_void| {
                    (y as i64) - (x as i64)
                };
            (
                a.w2.ptr() as *const u8, d(a.w2.ptr(), b.w2.ptr()),
                a.w2_scale.ptr() as *const u8, d(a.w2_scale.ptr(), b.w2_scale.ptr()),
            )
        } else {
            (std::ptr::null(), 0, std::ptr::null(), 0)
        };
        if moe_dbg() {
            let n_active = wsum.iter().filter(|w| **w != 0.0).count();
            eprintln!(
                "[mine] L{layer} route idx={:?} active_experts={} of {} (topk={})",
                &idx, n_active, ne, topk
            );
        }
        if !self.opts.skip_experts {
            // DSV41_EXPERT_TCGEN05_MXF4=1 dispatches the routed gate/up to the
            // tcgen05 `kind::mxf4` swapAB kernel (`dsv41_experts_mxf4.cu`). The
            // 2026-09-12 ABI-gap resolution closed all three blockers, so the
            // launcher now takes exactly what this site has: the loader's four
            // weight planes (w1/w3 + their scales) as base/stride pairs, the
            // quantiser's f32 activation scales (the kernel converts them to
            // e8m0), and a device-side `ids[slot]`. Two refusals remain, both
            // reported once below rather than silently falling back to the arm
            // the operator thinks they disabled: a `.so` without the symbol
            // (stock build) and the DSV41_EXPERT_ILV layout, which only the
            // fused GEMV body can read.
            if expert_tcgen05_mxf4() {
                if !self.dev.supports_expert_tcgen05_mxf4() {
                    tcgen05_mxf4_skipped_note(
                        "the loaded .so has no such symbol (a stock build does not define \
                         DSV41_TCGEN05_GATEUP_MXF4_SKELETON)",
                        false,
                    );
                } else if ilv {
                    tcgen05_mxf4_skipped_note(
                        "the routed gate/up pools are interleaved (DSV41_EXPERT_ILV) and only \
                         the fused GEMV body can read that layout",
                        true,
                    );
                } else if !batched {
                    tcgen05_mxf4_skipped_note(
                        "the batched MoE path is unavailable (DSV41_MOE_BATCH / the batched \
                         kernel set)",
                        true,
                    );
                }
            }
            // The input row is identical for every expert, so quantise it ONCE
            // here instead of inside the loop: the fp4 path was re-quantising and
            // re-packing the same 5120-element row for each of the ~6 selected
            // experts, i.e. 6x the quant_fp4 + fp4_pack launches (two of the
            // per-expert small kernels nsys counts ~850 times).
            self.dev.quant_fp4(
                self.s.xn.ptr as *const f32,
                self.s.xq4.ptr as *mut u8,
                self.s.xsc4.ptr as *mut f32,
                1,
                dim as i32,
                32,
                true,
            )?;
            // Fixed 6-slot device-driven loop: the expert id comes from
            // route_idx on the device and the weights from route_w, so there is
            // no host round trip and the launch arguments are static.
            //
            // DSV41_MOE_BATCH=1 (default OFF) replaces the loop below with ONE
            // launch per direction (grid.y = slot). Requirements, all checked
            // here so an old .so or a degenerate layer falls back to the
            // verified sequential path instead of failing:
            //   * topk > 0 and ne >= 2 (the indirect weight-base scheme);
            //   * the batched symbols are actually present in the loaded .so.
            let batched = moe_batch()
                && topk > 0
                && ne >= 2
                && self.dev.supports_moe_batch();
            if batched {
                // Per-slot strides. `ex_act_b` holds [topk][?] where ? is
                // 2*inter_local (unfused: gate|up) or inter_local (fused: the
                // swiglu'd result written by the kernel's epilogue).
                // `ex_down_b` [topk][dim]; both disjoint.
                let ids = self.s.route_idx.ptr as *const i32;
                // ---- tcgen05 mxf4 gate/up (DSV41_EXPERT_TCGEN05_MXF4) ---------
                // Default OFF. Its epilogue only clamps (no swiglu fusion), so it
                // writes the UNFUSED [2*inter] gate|up layout — the separate
                // swiglu launch below must stay enabled whenever it runs.
                // `Ok(false)` means the `.so` gate was OFF: nothing ran, so fall
                // through to the GEMV with the layout THAT call would have used,
                // i.e. byte-for-byte the pre-tcgen05 behaviour. ILV was already
                // reported as a refusal once, above.
                let tcgen05 =
                    expert_tcgen05_mxf4() && self.dev.supports_expert_tcgen05_mxf4() && !ilv;
                let mut ran_tc = false;
                if tcgen05 {
                    ran_tc = self.dev.expert_tcgen05_gate_up_mxf4(
                        self.s.xq4.as_u8(),
                        self.s.xsc4.as_f32(),
                        self.s.ex_act_b.ptr as *mut f32,
                        (2 * inter_local) as i64,  // unfused slot pitch
                        inter_local as i32,
                        dim as i32,
                        cfg.swiglu_limit,
                        topk as i32,
                        w1_base,
                        w1_stride,
                        w1s_base,
                        w1s_stride,
                        w3_base,
                        w3_stride,
                        w3s_base,
                        w3s_stride,
                        ids,
                    )?;
                }
                let gateup_fused =
                    !ran_tc && gateup_fuse() && self.dev.supports_gateup_fuse()
                        && expert_fp4_mode() == 2;
                let act_slot = if gateup_fused {
                    inter_local as i64
                } else {
                    (2 * inter_local) as i64
                };
                let down_slot = dim as i64;
                if !ran_tc {
                    self.dev.expert_gate_up_fp4_batched(
                    self.s.xq4.as_u8(),
                    self.s.xsc4.as_f32(),
                    self.s.ex_act_b.ptr as *mut f32,
                    act_slot,
                    1,
                    dim as i32,
                    inter_local as i32,
                    cfg.swiglu_limit,
                    topk as i32,
                    w1_base,
                    w1_stride,
                    w1s_base,
                    w1s_stride,
                    w3_base,
                    w3_stride,
                    w3s_base,
                    w3s_stride,
                    ids,
                    // Interleaved gate/up pools (DSV41_EXPERT_ILV): the kernel's
                    // fused body derives the up bytes from the gate pointer and
                    // reads one LDG.128 per group. `w3_base`/`w3_stride` still
                    // point at the same doubled region (the loader makes w3's
                    // view alias w1's), they are simply not read.
                    ilv as i32,
                )?;
                }
                // W2 L2 PREWARM (DSV41_W2_PREWARM, default OFF; `=1` enables).
                //
                // The down GEMV that follows streams w2 from HBM (286-385 GB/s
                // against a ~7 TB/s part => LATENCY-bound, not bandwidth-bound)
                // and nothing earlier in the step touches w2 - the pass just
                // finished read w1/w3. So we pull every slot's w2 rows into L2
                // NOW, in the gate/up ramp-down, and the down answers from L2
                // (~200 ns) instead of HBM (~600 ns). Same-layer only: 8 slots
                // x dim x (inter/2) = 9.17 MB + 0.57 MB of scale rows is under
                // 10% of a Blackwell-class L2, while a step's worth (40 x 9.17
                // MB = 367 MB) obviously is not.
                //
                // Placement is the whole trick: AFTER the gate/up launch (so the
                // gate/up pass's own 18 MB w1/w3 stream cannot evict what we
                // warmed) and BEFORE the down launch (so the burst lands under
                // its first microseconds). The entry point is fire-and-forget -
                // it writes nothing, reads only `ids` (the router's output, not
                // the gate/up launch's) and the weight pools, and always returns
                // 0 - so it is bit-identical by construction and can never break
                // a step. `sel_bytes`/`sc_bytes` are exactly the spans the down
                // launch below reads (rows = dim, k = inter_local): row `r` sits
                // at `w2_base + e*w2_stride + r*(inter/2)` and the rows are
                // contiguous, so the warmed union has no holes and no excess.
                if ne >= 2 && self.dev.supports_w2_prewarm() {
                    let kbytes = (inter_local as i64) >> 1;   // packed fp4 bytes per row
                    let ksc = (inter_local as i64) >> 5;      // e8m0 scales per row
                    if kbytes > 0 {
                        self.dev.w2_l2_prewarm(
                            w2_base,
                            w2_stride,
                            w2s_base,
                            w2s_stride,
                            self.s.route_idx.ptr as *const i32,
                            topk as i32,
                            dim as i64 * kbytes,
                            dim as i64 * ksc,
                        )?;
                    }
                }
                // gate_up+swiglu fusion: when the fused path ran (kernel wrote
                // the swiglu'd inter-width result directly), skip the separate
                // swiglu launch. The launcher's env-gated `fuse` mirrors this:
                // both must agree. The simplest correct wiring: try the fused
                // path (the kernel sets it when mode==2 && dim%512==0), and skip
                // swiglu when it did. We can't read the kernel's decision from
                // here, so we mirror the same condition. `ran_tc` forces it OFF:
                // the tcgen05 epilogue never fuses, so its [2*inter] output needs
                // the separate swiglu launch.
                let gateup_fused = !ran_tc && gateup_fuse()
                    && self.dev.supports_gateup_fuse()
                    && expert_fp4_mode() == 2;
                if !gateup_fused {
                    self.dev.swiglu_limit_batched(
                        self.s.ex_act_b.ptr as *mut f32,
                        1,
                        inter_local as i32,
                        cfg.swiglu_limit,
                        act_slot,
                        topk as i32,
                    )?;
                }
                // row_weight is PER SLOT here: route_w is [topk] and contiguous,
                // so the kernel reads route_w[slot] (rw_stride = 1) — the exact
                // scalar the sequential call passed as `route_w + slot`.
                //
                // DSV41_DOWN_FUSE (default ON) folds this down GEMV and the
                // fixed-order sum into ONE launch
                // (dsv41_expert_down_reduce_fp4_batched): ascending serial slot
                // loop + verbatim K loop + explicitly rounded per-slot product,
                // hence bit-identical to the pair, and the [topk][dim] scratch is
                // never touched. DSV41_DOWN_FUSE=0 (or a stale .so) runs the pair
                // exactly as before. `act_slot` is passed as the fused kernel's
                // `act_stride`: it is whatever pitch the gate/up call wrote, so
                // the gate_up+swiglu fusion's inter-width layout is picked up here
                // without any change in this call.
                if down_fuse() && self.dev.supports_down_fuse() {
                    self.dev.expert_down_reduce_fp4_batched(
                        self.s.ex_act_b.ptr as *const f32,
                        act_slot,
                        self.s.o.ptr as *mut f32,
                        1,
                        dim as i32,
                        inter_local as i32,
                        self.s.route_w.ptr as *const f32,
                        1,
                        topk as i32,
                        w2_base,
                        w2_stride,
                        w2s_base,
                        w2s_stride,
                        ids,
                    )?;
                } else {
                    self.dev.expert_down_fp4_batched(
                        self.s.ex_act_b.ptr as *const f32,
                        act_slot,
                        self.s.ex_down_b.ptr as *mut f32,
                        down_slot,
                        1,
                        dim as i32,
                        inter_local as i32,
                        self.s.route_w.ptr as *const f32,
                        1,
                        topk as i32,
                        w2_base,
                        w2_stride,
                        w2s_base,
                        w2s_stride,
                        ids,
                    )?;
                    // Fixed-order sum, slot 0 first: the SAME order the sequential
                    // `o[row] += x` accumulation used (from the zeroed `o`), so the
                    // result is bit-identical (fp addition is not associative).
                    self.dev.moe_down_reduce(
                        self.s.ex_down_b.ptr as *const f32,
                        self.s.o.ptr as *mut f32,
                        dim as i32,
                        topk as i32,
                    )?;
                }
            } else {
                for slot in 0..topk {
                    let w = (self.s.route_w.ptr as *const f32).wrapping_add(slot);
                    let ids = self.s.route_idx.ptr as *const i32;
                    self.dev.expert_gate_up_fp4_indirect(
                        self.s.xq4.as_u8(),
                        self.s.xsc4.as_f32(),
                        self.s.ex_act.ptr as *mut f32,
                        1,
                        dim as i32,
                        inter_local as i32,
                        cfg.swiglu_limit,
                        w1_base,
                        w1_stride,
                        w1s_base,
                        w1s_stride,
                        w3_base,
                        w3_stride,
                        w3s_base,
                        w3s_stride,
                        ids,
                        slot as i32,
                    )?;
                    self.dev.swiglu_limit(
                        self.s.ex_act.ptr as *mut f32,
                        1,
                        inter_local as i32,
                        cfg.swiglu_limit,
                    )?;
                    self.dev.expert_down_fp4_indirect(
                        self.s.ex_act.ptr as *const f32,
                        self.s.o.ptr as *mut f32,
                        1,
                        dim as i32,
                        inter_local as i32,
                        w,
                        w2_base,
                        w2_stride,
                        w2s_base,
                        w2s_stride,
                        ids,
                        slot as i32,
                    )?;
                }
            }
        }

        // shared expert: fp8, every token. Its weights are replicated, so under
        // a collective exactly one rank may contribute it — otherwise the
        // all-reduce below would sum it `world` times. (`shared_rank` is computed
        // up at the gate, where the mixed launch decides whether to fold w1/w3 in.)
        if !self.opts.skip_shared_expert && shared_rank {
            if let (Some(w1), Some(w1s), Some(w3), Some(w3s), Some(w2), Some(w2s)) = (
                ld.shared_w1.as_ref(),
                ld.shared_w1_scale.as_ref(),
                ld.shared_w3.as_ref(),
                ld.shared_w3_scale.as_ref(),
                ld.shared_w2.as_ref(),
                ld.shared_w2_scale.as_ref(),
            ) {
                // When the mixed gate launch already produced w1/w3 from the same
                // quantised activation, only swiglu/quant/down remain here.
                if !sh_via_mixed {
                    // MOE_DUAL: issued on `sh_st` (the side stream under the dual
                    // chain, the main stream otherwise). `s.xq`/`s.xsc` are written
                    // ONLY by this half of `moe()` - the routed chain reads `xn`
                    // through the disjoint `s.xq4`/`s.xsc4`, and the mixed gate
                    // launch (the one main-stream `s.xq` writer) forces `dual` off.
                    self.quant1_on(self.s.xn.ptr as *const f32, dim as i32, sh_st)?;
                    // gate and up land contiguously so `swiglu_limit` sees [gate|up].
                    // `sh_il` (inter/world) is this rank's slice: the weights are
                    // Rows-sharded, so one launch still covers both projections and
                    // the all-reduce below reconstructs the full shared expert.
                    let sh_fused = sh_exp_mx2()
                        && self.dev.gemm_fp8_mx2_on(
                            self.s.xq.as_u8(),
                            self.s.xsc.as_f32(),
                            w1.as_u8(),
                            w1s.as_u8(),
                            std::ptr::null(),
                            self.s.ex_act.ptr as *mut f32,
                            sh_il as i32,
                            w3.as_u8(),
                            w3s.as_u8(),
                            std::ptr::null(),
                            (self.s.ex_act.ptr as *mut f32).wrapping_add(sh_il),
                            sh_il as i32,
                            dim as i32,
                            sh_st,
                        )?;
                    if !sh_fused {
                        self.dev.gemm_fp8_mx_on(
                            self.s.xq.as_u8(),
                            self.s.xsc.as_f32(),
                            w1.as_u8(),
                            w1s.as_u8(),
                            std::ptr::null(),
                            self.s.ex_act.ptr as *mut f32,
                            1,
                            sh_il as i32,
                            dim as i32,
                            sh_st,
                        )?;
                        self.dev.gemm_fp8_mx_on(
                            self.s.xq.as_u8(),
                            self.s.xsc.as_f32(),
                            w3.as_u8(),
                            w3s.as_u8(),
                            std::ptr::null(),
                            (self.s.ex_act.ptr as *mut f32).wrapping_add(sh_il),
                            1,
                            sh_il as i32,
                            dim as i32,
                            sh_st,
                        )?;
                    }
                }
                // A4: the swiglu epilogue emits the fp8 pair the w2 GEMV reads, so
                // quant1(ex_act) disappears (40 launches/step). Only the SHARED
                // expert's down needs it: the routed experts' down consumes
                // ex_act_b as f32 directly (no quant to fold).
                let act_q = swiglu_q()
                    && self.dev.supports_swiglu_q()
                    && self.dev.swiglu_limit_q_on(
                        self.s.ex_act.ptr as *mut f32,
                        1,
                        sh_il as i32,
                        cfg.swiglu_limit,
                        self.s.xq.ptr as *mut u8,
                        self.s.xsc.ptr as *mut f32,
                        sh_st,
                    )?;
                if !act_q {
                    self.dev.swiglu_limit_on(
                        self.s.ex_act.ptr as *mut f32,
                        1,
                        sh_il as i32,
                        cfg.swiglu_limit,
                        sh_st,
                    )?;
                    self.quant1_on(self.s.ex_act.ptr as *const f32, sh_il as i32, sh_st)?;
                }
                // w2 is Cols-sharded: [dim, sh_il] locally, so the reduction is over
                // this rank's slice and the output is a PARTIAL [dim] that the MoE
                // all-reduce sums with the other ranks'.
                //
                // MOE_DUAL: the w2 GEMV may NOT fold into `s.o` — the routed chain
                // is writing `s.o` on the main stream at the same time. It always
                // writes the DISJOINT `s.ex_out` here, and the join below runs the
                // same `add_inplace(&s.o, &s.ex_out)` the serial non-fused path ran
                // (same operand, same order => bit-identical). The A5 fused epilogue
                // is exactly that pair, so forcing it off under MOE_DUAL costs only
                // the one extra launch on the side stream.
                //
                // A5 (serial only): the w2 GEMV's epilogue adds straight into `s.o`
                // (the MoE accumulator AR#2 reduces), dropping the standalone
                // ferrite_add launch (40/step). Association is unchanged --
                // o + (acc + bias) either way -- so the result is bit-identical.
                let fused = !dual
                    && moe_epi_add()
                    && self.dev.supports_gemm_fp8_add()
                    && self.dev.gemm_fp8_mx_add(
                        self.s.xq.as_u8(),
                        self.s.xsc.as_f32(),
                        w2.as_u8(),
                        w2s.as_u8(),
                        std::ptr::null(),
                        self.s.o.ptr as *mut f32,
                        1,
                        dim as i32,
                        sh_il as i32,
                    )?;
                if !fused {
                    self.dev.gemm_fp8_mx_on(
                        self.s.xq.as_u8(),
                        self.s.xsc.as_f32(),
                        w2.as_u8(),
                        w2s.as_u8(),
                        std::ptr::null(),
                        self.s.ex_out.ptr as *mut f32,
                        1,
                        dim as i32,
                        sh_il as i32,
                        sh_st,
                    )?;
                    if !dual {
                        // ADD_EPI: defer the merge into the MoE all-reduce's
                        // store epilogue when the .so carries the biased entry
                        // (see `add_epi()`); otherwise the standalone add.
                        if self.add_epi_ready() {
                            self.moe_add_in[layer] = Some(self.s.ex_out.ptr as *const f32);
                        } else {
                            self.dev.add_inplace(&self.s.o, &self.s.ex_out, dim as i64)?;
                        }
                    }
                }
            }
        }
        // MOE_DUAL join: the shared half is fully issued (its last consumer of
        // `s.ex_out` here is the add just below). Wait the side stream back on the
        // main stream, then merge exactly as the serial path did — the routed
        // down-reduce wrote `s.o` on the main stream, so this add sees the same two
        // operands in the same order. A no-op when the fork did not take.
        if dual {
            self.dev.dual_chain_join()?;
            // ADD_EPI: the join just made `s.ex_out` final on the main stream, so
            // the following AR's store can publish `s.o + s.ex_out` itself.
            if self.add_epi_ready() {
                self.moe_add_in[layer] = Some(self.s.ex_out.ptr as *const f32);
            } else {
                self.dev.add_inplace(&self.s.o, &self.s.ex_out, dim as i64)?;
            }
        }
        // the block output is the attention-branch accumulator `o`
        Ok(())
    }
}
