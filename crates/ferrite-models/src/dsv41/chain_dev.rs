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

use ferrite_types::{spec_accept, FerriteError, Result, SpecStep};

use std::sync::Arc;

use crate::dsv41::config::{Dsv41Config, KvMode};
use crate::dsv41::tp::Collective;
use crate::dsv41::device::{CuStream, DevBuf, Device};
use crate::dsv41::dump_dev;
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

/// Compile-time proof that the per-row tap's PRODUCER and CONSUMERS agree on its
/// row stride. The `layer_rows` hook writes slot `s`'s row `r` at
/// `(s * VERIFY_ROWS + r) * dim` and both consumers spell the same stride with
/// this ONE constant: [`DsparkDev::note_ctx_rows`] (`(slot * VERIFY_ROWS + j) *
/// row_bytes`, dspark_dev.rs) and [`DevChain::carry_kept_tap`] (`slot *
/// VERIFY_ROWS * row_bytes`). A block is the anchor row plus `DSPARK_DRAFTS`
/// drafts, so `VERIFY_ROWS` must be exactly one more than `DSPARK_DRAFTS` — if a
/// future edit changes either number without the other, this fails the BUILD
/// instead of silently reading one row past the block (a stride mismatch is not
/// expressible in the buffer's type, so nothing else would catch it).
const _: () = assert!(
    VERIFY_ROWS == DSPARK_DRAFTS + 1,
    "VERIFY_ROWS must be the anchor row plus the DSPARK_DRAFTS draft rows"
);

/// How many row SHAPES the verify graph pool holds (see
/// [`DevChain::verify_graphs`]): the two a production request actually produces
/// — the shadow step's `DSPARK_DRAFTS` rows and the swallowed anchor's `+1`
/// (`SEED_ALIGN`/`SWALLOW`). A third shape (the parity self-test varies `m`)
/// takes the direct launches. Each slot is an independent graph (its own
/// DRY→capture schedule, its own failure latch).
const VERIFY_GRAPH_SLOTS: usize = 3;

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
    /// `k` (1..=DSPARK_DRAFTS): the number of tokens the speculative path
    /// WOULD have emitted from this step — the accepted draft prefix plus one
    /// bonus token. `1` means no draft survived, i.e. `next` alone.
    pub accepted: usize,
    /// The draft phase (`import_tap` + `draft_forward` + the drafts D2H).
    pub draft_ms: f32,
    /// The verify phase (`step_rows`, one m-row forward).
    pub verify_ms: f32,
}

/// What one REAL-COMMIT DSpark step produced. See
/// [`DevChain::dspark_spec_step`]: the anchor row plus the accepted draft prefix
/// are COMMITTED (the ring keeps their KV, the compressor is replayed up to the
/// last of them and the position counter jumps by `k_acc + 1`), so this report
/// describes engine state that has MOVED — unlike [`DsparkShadowReport`], whose
/// block is always undone.
#[derive(Debug, Clone)]
pub struct DsparkSpecReport {
    /// The anchor: `step_dev`'s argmax, i.e. the token at `pos + 1`. Its forward
    /// is verify row 0.
    pub next: u32,
    /// What the draft proposed: `drafts[j]` is the draft's guess for `pos + 2 + j`.
    pub drafts: [u32; DSPARK_DRAFTS],
    /// The verify block's per-row argmax — row `j` sits at `pos + 1 + j` (row 0
    /// is the anchor) and predicts `pos + 2 + j`.
    pub verify_out: [u32; DSPARK_DRAFTS],
    /// The accepted draft-prefix length (0..=DSPARK_DRAFTS): `drafts[0..k_acc]`
    /// all matched the verify row that predicts them.
    pub k_acc: usize,
    /// What this step EMITS, in order: `[next] ++ verify_out[0..k_acc]`, i.e.
    /// `k_acc + 1` tokens at positions `pos + 1 ..= pos + 1 + k_acc`. The last of
    /// them is the next step's input (`pos_ctr` has already been advanced to its
    /// position, and its KV is NOT yet in the ring — the next step appends it).
    pub emitted: Vec<u32>,
    /// The draft phase (`import_tap` + `draft_forward` + the drafts D2H).
    pub draft_ms: f32,
    /// The verify phase (`step_rows`, one 6-row forward).
    pub verify_ms: f32,
    /// The commit phase: partial rollback + compressor replay + the position
    /// counter's H2D + the draft's ctx rows ([`DsparkDev::note_ctx_rows`], the
    /// multi-token commit's window back-fill) — the price the shadow path never
    /// pays.
    pub commit_ms: f32,
}

/// What one `DSV41_DIFF_EAGER` probe found: the tokens a
/// [`DevChain::dspark_spec_step`] emitted against the SAME positions decoded one
/// row at a time by the plain engine.
///
/// The two vectors are index-aligned on the same position: `spec_emitted[i]` and
/// `eager[i]` are both the token at `pos + 1 + i`, the first taken from the m-row
/// verify's argmax (what the spec path emits) and the second from a single-row
/// `step_dev` replay of that position. Greedy speculative decoding makes them
/// equal by construction, so [`Self::first_mismatch`] is the first place the two
/// paths disagree — the first position at which a spec-only defect becomes
/// visible in the token stream.
#[derive(Debug, Clone)]
pub struct DiffEagerReport {
    /// The tokens the spec step emitted, the ones at
    /// `pos + 1 ..= pos + spec_emitted.len()`.
    pub spec_emitted: Vec<u32>,
    /// The single-row greedy decode of the same positions. `eager[0]` is echoed
    /// from `spec_emitted[0]` rather than re-run: the anchor IS a single-row
    /// `step_dev` argmax (every spec arm feeds the step's own argmax), so a
    /// re-run would measure reproducibility, not the two paths against each
    /// other.
    pub eager: Vec<u32>,
    /// The smallest `i` with `eager[i] != spec_emitted[i]`, else `None`. Its
    /// position in the sequence is `pos + 1 + i`.
    pub first_mismatch: Option<usize>,
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
    xq4: DevBuf,   // [dim] fp4 nibbles (dim/2 bytes) or e4m3 bytes (dim bytes, ACT_E4M3)
    xsc4: DevBuf,  // [dim/32 + 8] f32 scales (either activation format)
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
    /// R2b: the m-ROW twin of `qr_raw` — set by `attention_rows` when the fused
    /// `lin_rope_norm` wq_b launch left the block's `qr_r` RAW (it normalised into
    /// shared memory instead of in place, see [`attn_lin_fuse`]). The indexer's q
    /// half then normalises it inside its OWN gemv prologue (`indexer_front_rows`
    /// / `indexer_rows_one`), which is what removes the compensating `norm_rows`
    /// R2 had to pay. Set on EVERY `attention_rows` path — the fused arm is the
    /// only one that sets it true — and consumed by the first indexer q-half that
    /// reads `qr_r` (a layer without an indexer leaves it set; the next layer's
    /// `attention_rows` overwrites it, so it cannot leak across a layer).
    qr_raw_r: std::cell::Cell<bool>,
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
    /// Vocabulary-sliced VERIFY head only (`DSV41_VERIFY_HEAD_SLICED`): one
    /// packed comparison key per verify ROW, published TOGETHER in ONE v5 round
    /// by `dsv41_argmax_sliced_rows` (the single-row `argmax_packed` holds one).
    argmax_packed_r: DevBuf, // [VERIFY_ROWS] u64
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
    /// Per-ROW twin of `dspark_tap`, for the REAL-COMMIT path
    /// ([`Self::dspark_spec_step`]): the verify forwards `m` rows in ONE pass and
    /// the target hidden of EVERY accepted row has to reach the draft's window
    /// ([`DsparkDev::note_ctx_rows`]) — the single-row tap only ever holds the
    /// position the step consumed, so a `k > 1` commit would leave the positions
    /// it skipped as holes in that window. Layout `[DSPARK_TAP_SLOTS][m][dim]`:
    /// slot `slot` (`Dsv41Config::dspark_target_slot`), row `r` at
    /// `(slot * m + r) * dim` — the same `[hc, dim]` mean, one `hc_collapse` per
    /// row, captured by the `layer_rows` hook at the same point `layer()`'s hook
    /// uses. Written only while `spec_capture` is set, so the shadow path (whose
    /// verify is rolled back) pays nothing for it.
    dspark_tap_r: DevBuf,
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
    /// [VERIFY_ROWS, xq_row] fp8 e4m3 + [VERIFY_ROWS, xq_row/32] f32: the
    /// multi-row activation staging for `attention_rows`'s projections. The
    /// per-row loop quantised ONE row into `s.xq`/`s.xsc` and then called the m=1
    /// GEMV; the multi-row form (`gemm_fp8_mrows` / `wo_a_grouped_fp8`) takes the
    /// whole `[m, k]` block, so the block is quantised here ONCE per projection
    /// (`quant_rows`) and each projection becomes a single launch. Row `r` of
    /// this buffer is bit-identical to what the per-row `quant1` wrote there.
    xq_r: DevBuf,
    xsc_r: DevBuf,
    moe_out_r: DevBuf, // [m, dim] MoE block output (routed + shared expert)
    ids_r: DevBuf,     // [m] i32 token ids
    argmax_r: DevBuf,  // [m] i32 per-row argmax (the head's output)
    /// [m, vocab] — but the ROW PITCH is `seg = vocab/world` under
    /// `DSV41_VERIFY_HEAD_SLICED` (only the first `seg` of each row is written).
    /// `verify_head_geom` is the single place that decides which, and the head
    /// GEMV, the argmax and the probe all index it through that decision: the
    /// pitch is NOT in the type, so a site that assumes one of the two is a
    /// silent misread.
    logits_r: DevBuf,  // [m, vocab] or [m, seg] — see the doc comment
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
    /// COMPRESSOR-MROWS (`DSV41_COMPRESSOR_MROWS`): `[n_layers, VERIFY_ROWS]` i32 —
    /// the per-row device-counter snapshots `dsv41_compressor_fused_mrows` writes
    /// (`clen_rows_r[layer*VERIFY_ROWS + r]` = the counter AFTER layer `layer`'s
    /// row `r` commit). Per-LAYER indexed rather than one row-block: a consumer
    /// layer owns no compressor and reads its OWNER's snapshots, and another
    /// source layer's hoisted launch can sit between the owner and the consumer,
    /// so a shared block would be overwritten before the consumer read it.
    clen_rows_r: DevBuf,
    /// COMPRESSOR-MROWS: `[VERIFY_ROWS, hd]` f32, row r = row r's pooled+normed
    /// latent, written by the same hoisted launch. ONE row-block, never per-layer:
    /// only the layer that ran the launch reads it (`publish_index_key_rows`), and
    /// only inside its own block, before any other launch can overwrite it.
    latent_rows_r: DevBuf,
    // ---- MoE, m rows ----
    scores_r: DevBuf,   // [m, n_experts] the bf16 gate's output
    route_idx_r: DevBuf, // [m, topk] i32
    route_w_r: DevBuf,   // [m, topk] f32
    // ---- grouped (permuted) routing, DSV41_EXPERT_GROUPED (default OFF) ----
    //
    // The expert-centric routed GEMMs (`dsv41_expert_gemm_e4m3_ext`) take ONE
    // expert per launch over a DENSE row block, while `route_idx_r` is
    // per-(row, slot). These buffers hold the expert-major form of that table:
    // `grp_starts[e] .. grp_starts[e] + grp_counts[e]` is expert e's contiguous
    // block in the grouped order, `grp_rows`/`grp_slots` say which (row, slot)
    // each grouped position came from, and `grp_perm` / `grp_src` are the two
    // directions of the permutation the movers need. Layout contract:
    // `kernels/cuda/dsv41_route.cu`. Allocated unconditionally so the
    // allocation graph stays static (a few hundred KB), but written and read
    // only when the gate is armed and the `.so` carries the symbols.
    /// `[n_routed]` i32 — assignments (rows) per expert.
    grp_counts: DevBuf,
    /// `[n_routed + 1]` i32 — exclusive prefix sum of `grp_counts`;
    /// `grp_starts[n_routed]` is the total number of routable assignments.
    grp_starts: DevBuf,
    /// `[n_routed * grp_cap]` i32 — expert e's j-th grouped row's SOURCE row.
    grp_rows: DevBuf,
    /// `[n_routed * grp_cap]` i32 — expert e's j-th grouped row's SOURCE slot.
    grp_slots: DevBuf,
    /// `[VERIFY_ROWS * topk]` i32 — original flat assignment -> grouped position.
    grp_perm: DevBuf,
    /// `[VERIFY_ROWS * topk]` i32 — grouped position -> original flat assignment.
    grp_src: DevBuf,
    /// `[n_routed]` i32 — the experts that own at least one assignment, ascending,
    /// so the host iterates the ~topk*m live experts instead of all `n_routed`.
    grp_active: DevBuf,
    /// `[1]` i32 — the live length of `grp_active`.
    grp_nactive: DevBuf,
    /// `[VERIFY_ROWS * topk * dim]` u8 — the gathered e4m3 activation rows, in
    /// grouped order (each source row appears once per slot that routes to a
    /// different expert, which is what makes an expert's operand contiguous).
    grp_xq: DevBuf,
    /// `[VERIFY_ROWS * topk * dim/32 + 8]` f32 — their per-32-block scales.
    grp_xsc: DevBuf,
    /// `[VERIFY_ROWS * topk * 2 * inter]` f32 — the GROUPED gate/up output of
    /// `dsv41_expert_gemm_e4m3_grouped`, one row per ASSIGNMENT in grouped order
    /// (same shape as `ex_act_r`, hence the same capacity formula). It exists so
    /// the masked GEMM can write a CONTIGUOUS per-expert row block; the
    /// `dsv41_route_scatter_rows` call that follows moves it into `ex_act_r`, so
    /// everything downstream (swiglu, the down GEMV, the reduce) is untouched by
    /// the grouped path.
    grp_out: DevBuf,
    xq4_r: DevBuf,       // [m, dim] fp4 nibbles (dim/2 bytes/row) or e4m3 (dim)
    xsc4_r: DevBuf,      // [m, dim/32 + 8] f32 scales (either format)
    /// [m, topk, 2*inter] routed gate|up output. The batched expert launchers
    /// process ONE activation row per call (their `rows` argument is validated but
    /// never enters the grid — see `moe_rows`), so this is a per-row slot block:
    /// row `r`'s slot `t` starts at `r*(topk*act_slot) + t*act_slot`.
    ex_act_r: DevBuf,
    /// [m, topk, dim] the DOWN_FUSE=0 fallback's per-slot scratch.
    ex_down_r: DevBuf,
    /// [m, 2*inter] the shared expert's gate|up row (one row at a time).
    sh_act_r: DevBuf,
    /// [m, dim] the shared expert's w2 output for the whole block —
    /// [`DevChain::shared_expert_mrows`]'s write-then-add scratch. The per-row
    /// path writes w2 straight into `moe_out_r + r*dim` (A5) or into the
    /// single-row `ex_out` and adds; the multi-row form writes the block to a
    /// `[m, dim]` scratch and folds it in with ONE element-wise add, because
    /// `gemm_fp8_mrows` (unlike `gemm_fp8_mx_add`) has no accumulate epilogue.
    sh_out_r: DevBuf,
    /// [VERIFY_ROWS, inter] fp8 e4m3 bytes + [VERIFY_ROWS, inter/32 + 8] f32: the
    /// fused arm's (`DSV41_SH_EXP_FUSED`) phase-1 output, i.e. the `aq`/`aqsc` of
    /// `dsv41_gemm_fp8_sh_pair`'s phase 1 — the swiglu pair in fp8, which is what
    /// its phase 2 (`w2`) consumes. Row `r` uses `sh_aq_r + r*sh_il` and
    /// `sh_aqsc_r + r*(sh_il/32)`; `inter >= inter/world = sh_il`, so one row
    /// stride of `inter` covers every rank's slice.
    ///
    /// ⚠️ DELIBERATELY SEPARATE from `xq_r`/`xsc_r`. The fused kernel stages the
    /// activation it was GIVEN (`a`/`a_scale`) in EVERY block, then writes the
    /// swiglu fp8 with no barrier in between, so those two must be DISJOINT —
    /// sharing one buffer is a cross-block race inside a single grid-sync launch
    /// (the kernel's "WHY THE fp8 OUTPUT NEEDS ITS OWN BUFFER" note). The per-row
    /// path reuses `s.xq` for the same bytes only because its two launches are
    /// separated by a stream edge; that is exactly the serialisation this arm
    /// removes, so the buffer has to be its own.
    sh_aq_r: DevBuf,
    sh_aqsc_r: DevBuf,
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
    // ==================== DSpark spec commit (DSV41_SPEC) ====================
    //
    // `dspark_spec_step` keeps the ACCEPTED PREFIX of a verify block instead of
    // rolling the whole block back. The compressor cannot be rewound row by row
    // (its `state_kv`/`state_score` slots are overwritten in position order), so
    // the commit restores the pre-verify snapshot and REPLAYS the kept rows
    // through the same per-row pool+commit triple the verify used.
    //
    // Playing a row back needs that row's `kvp`/`scp`, and those live in the
    // SHARED m-row scratch (`s.kvp_r`/`s.scp_r`), which every compress-source
    // layer overwrites in turn — so the verify has to save them per layer while
    // it still can. Layout indexed BY LAYER with a `VERIFY_ROWS` row stride,
    // exactly like the shadow snapshot above.
    /// `[n_layers][VERIFY_ROWS][head_dim]` f32 — the verify's `kvp` rows.
    spec_snap_kvp: DevBuf,
    /// `[n_layers][VERIFY_ROWS][head_dim]` f32 — the verify's `scp` rows.
    spec_snap_scp: DevBuf,
    // ==================== the KV prefix snapshot (serve-side prefix cache) ====
    //
    // `DevChain::kv_snapshot` freezes the WHOLE sequence's prefix state into the
    // host, and `kv_restore` puts it back. The layout deliberately mirrors the
    // shadow save's above — indexed BY LAYER, carry stored as `state_kv` then
    // `state_score` at a `max_ratio * head_dim` stride — but at the FULL window
    // width instead of a `VERIFY_ROWS` block, and as persistent scratch rather
    // than a within-a-step save.
    //
    // ⚠️ Not the `dspark_snap_*` buffers themselves: those are `VERIFY_ROWS` rows
    // wide and must be free for a verify block to save into at any step. A prefix
    // snapshot is taken BETWEEN requests, so the two never overlap in time — but
    // sharing a buffer would make that an argument about call order instead of a
    // property of the code.
    /// `[n_layers][window][head_dim]` f32 — the staging area every ring owner's
    /// window is D2D'd into, so the window half costs ONE D2H instead of one per
    /// layer (a D2H is a full device sync; a 40-layer chain would pay 40).
    kv_snap_ring: DevBuf,
    /// `[n_layers][2][max_ratio * head_dim]` f32 — the compressor carry, same
    /// segmentation as `dspark_snap_state`.
    kv_snap_state: DevBuf,
    /// `[n_layers][head_dim]` f32 — the pooled latent rows, `dspark_snap_latent`'s
    /// layout.
    kv_snap_latent: DevBuf,
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

/// `DSV41_BF16_TRUNCATE=1` rounds every `hc_pre` collapse back to bf16 before
/// the hyper-connection norm, reproducing the official `model.py:957-960`
/// (`y = sum(...); return y.to(x.dtype)`) whose whole residual stream is bf16.
/// ferrite keeps the collapse in f32, which is ~1e-3 more precise per layer than
/// the reference; over 44 layers that drift is the prime suspect for the
/// near-tie argmax flips (docs/agent/dspark-correctness-chain.md, "hc_pre 的
/// dtype 截断对照").
///
/// DEFAULT OFF: the f32 path is the live, verified one. Read ONCE and cached
/// (the house rule — a per-call getenv is a hot-path slip), and `"0"` means OFF
/// even though it is "set". A MODULE-level free function (like `eng_host`
/// above), not an associated one: the DSpark draft calls it from
/// `dspark_dev.rs` as `chain_dev::bf16_truncate`.
pub(crate) fn bf16_truncate() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_BF16_TRUNCATE").map(|v| v != "0").unwrap_or(false))
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

/// `DSV41_EXPERT_TCGEN05_MXF4=1` (or its short alias `DSV41_EXPERT_TCGEN05=1`)
/// arms the tcgen05 MXFP4 swapAB gate/up
/// (`tc5::mxf4::expert_tcgen05_gateup_mxf4_kernel`, `dsv41_experts_mxf4.cu:3805`).
///
/// Mirror of the launcher's own gate test: either name, first char `'1'` — a
/// strict "1..." prefix, NOT the usual `!= "0"`, so the default is OFF and
/// `=0` / anything else stays OFF. Read ONCE and cached: the `.so` reads the same
/// variables once per process, so a per-call getenv here could only ever add a
/// hot-path slip and a capture hazard (plan §5), never a different decision.
///
/// ⚠️ TWO-NAME ORDER AND STRICTNESS ARE THE CONTRACT. The `.so` tests
/// `DSV41_EXPERT_TCGEN05_MXF4` then `DSV41_EXPERT_TCGEN05`
/// (`dsv41_experts_mxf4.cu`, the launcher's `enabled` lambda). If this list and
/// that one ever disagree, the Rust side believes the step runs the tcgen05 arm
/// while the `.so` keeps the paired GEMV — both A/B arms then measure the OLD
/// path (the project's #1 measurement-bias trap). Keep the two tests identical.
///
/// ⚠️ The gate can only arm a `.so` that CARRIES the symbol. Since 2026-09-12
/// `build.sh` compiles `DSV41_TCGEN05_GATEUP_MXF4_SKELETON` in BY DEFAULT (opt
/// out with `DSV41_BUILD_TCGEN05_MXF4=0`), so a stock build of that revision now
/// satisfies `Device::supports_expert_tcgen05_mxf4()`. This gate itself is still
/// default OFF, so no behaviour changes unless an operator sets it.
pub(crate) fn expert_tcgen05_mxf4() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        ["DSV41_EXPERT_TCGEN05_MXF4", "DSV41_EXPERT_TCGEN05"]
            .iter()
            .any(|n| std::env::var(n).map(|v| v.starts_with('1')).unwrap_or(false))
    })
}

/// `DSV41_EXPERT_TCGEN05_E4M3=1` arms the tcgen05 **e4m3-activation** gate/up
/// (`tc5::e4::expert_tcgen05_gateup_e4_kernel`, `dsv41_experts_mxf4.cu`), the
/// sibling of the arm above for the DIRECT e4m3 activation
/// (`DSV41_EXPERT_ACT_E4M3`). It is the ONLY way the tcgen05 gate/up can run
/// while the e4m3 activation is armed: `kind::mxf4` is e2m1 x e2m1 and the
/// `DSV41_EXPERT_ACT_E4M3` row would be decoded as fp4 nibbles.
///
/// Mirror of the launcher's own gate test: ONE name, first char `'1'` — a strict
/// "1..." prefix, NOT the usual `!= "0"`, so the default is OFF and `=0` stays
/// OFF. Read ONCE and cached, exactly like `expert_tcgen05_mxf4()`: the `.so`
/// reads the same variable once per process, so a per-call getenv here could
/// only add a hot-path slip and a capture hazard (plan §5).
///
/// ⚠️ MUST stay byte-for-byte equivalent to the `.so`'s test
/// (`dsv41_experts_mxf4.cu`, the `enabled` lambda of
/// `dsv41_expert_tcgen05_gate_up_e4m3`). If the two disagree, the Rust side
/// believes the step runs this arm while the `.so` keeps the paired GEMV — both
/// A/B arms then measure the OLD path (the project's #1 measurement-bias trap).
///
/// ⚠️ The gate can only arm a `.so` that CARRIES the symbol. `build.sh` compiles
/// `DSV41_TCGEN05_GATEUP_E4M3_SKELETON` in BY DEFAULT (opt out with
/// `DSV41_BUILD_TCGEN05_E4M3=0`), so a stock build satisfies
/// `Device::supports_expert_tcgen05_e4m3()`; this gate itself is still default
/// OFF, so no behaviour changes unless an operator sets it.
pub(crate) fn expert_tcgen05_e4m3() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("DSV41_EXPERT_TCGEN05_E4M3")
            .map(|v| v.starts_with('1'))
            .unwrap_or(false)
    })
}

/// One-shot notice for an ARMED-but-undispatchable `DSV41_EXPERT_TCGEN05_E4M3`.
/// The `.so` reads the same env var itself, so an operator who exports it
/// believes the step now runs the tcgen05 e4m3 gate/up — while `moe()` keeps
/// issuing the proven GEMV (see the call site: no symbol in the `.so`, the
/// interleaved weight layout, or no batched path). A silent no-op there is the
/// project's #1 measurement-bias trap (an "ON" arm that measures the OLD path),
/// so it is said out loud once.
fn tcgen05_e4m3_skipped_note(reason: &str, so_has_symbol: bool) {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        eprintln!(
            "warning: DSV41_EXPERT_TCGEN05_E4M3 is set, but the routed MoE still dispatches the \
             gate/up to the proven GEMV/GEMM: {reason}. Any A/B run with this gate ON measures \
             the OLD path. (symbol present in .so: {so_has_symbol})"
        );
    });
}

/// One-shot notice for an ARMED-but-undispatchable `tc5::e4x` (the dense M=128
/// tile e4m3 arm, `dsv41_expert_gemm_e4m3_ext`). Separate from
/// [`tcgen05_e4m3_skipped_note`] because the two arms ship under ONE runtime
/// gate (`DSV41_EXPERT_TCGEN05_E4M3`) and ONE `.so`: an operator who exports
/// the gate for the swapAB arm would otherwise be told nothing when the dense
/// tile arm declines (no symbol, or a tile that is not its shape).
fn tcgen05_e4m3_ext_skipped_note(reason: &str, so_has_symbol: bool) {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        eprintln!(
            "warning: DSV41_EXPERT_TCGEN05_E4M3 is set, but the routed gate/up stays on the \
             batched SIMT launch: the dense-tile tc5::e4x arm declined ({reason}). Any A/B run \
             expecting that arm measures the OLD path. (symbol present in .so: {so_has_symbol})"
        );
    });
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

/// DSV41_EXPERT_ACT_E4M3=1 runs the routed experts' activation as the OFFICIAL
/// **e4m3** one: `act_quant(e4m3, block=32)` then ONE gate/up pass.
///
/// WHY. The official `linear()` (`model.py:186-195`) feeds an fp4 WEIGHT an
/// **e4m3 activation** (`ref_inference/kernel.py::fp4_gemm_kernel` — "FP8 act x
/// FP4 weight", the fp4 sub-block cast up to fp8 and the GEMM run in fp8), so
/// ferrite's e2m1 activation carried ~4.5x the official noise on every
/// routed-expert output of all 44 layers — the measured root cause of the
/// generation degradation (docs/agent/dspark-correctness-chain.md).
///
/// HOW (direct, 2026-09-12). Quantise the activation to e4m3 with the
/// quantiser's own block-32 power-of-two scale (`dsv41_quant_fp8`, the SAME
/// entry point the dense fp8 chain uses) and hand those bytes to the fp4
/// expert gate/up with the kernel's `act_e4m3` flag: the kernel decodes one
/// e4m3 byte per value (`dsv41_e4m3_to_f`, exact) instead of two packed e2m1
/// nibbles and the rest of the GEMV — the K loop, the shuffle tree, the
/// epilogue — is untouched. ONE pass, the same launch count as the e2m1 arm:
/// only the activation format changes.
///
/// This REPLACES the e2m1x2 two-pass simulation (`q_hi + q_lo` against the same
/// fp4 weight, summed before swiglu). That arm reached 0.53x the official noise
/// but cost a full second gate/up GEMM (~+5 ms/step for ~+2% accept), which is
/// why the direct e4m3 path — exactly the official numerics, half the expert
/// GEMM — is the one that ships.
///
/// CONSEQUENCES (deliberate):
///   * gate_up+swiglu FUSION STAYS ON: this is a single pass, so the fused
///     epilogue applies swiglu to the one true gate/up pair. (The retired
///     two-pass arm had to force it OFF because swiglu(x+y) != swiglu(x)+swiglu(y).)
///   * the tcgen05 MXFP4 arm is skipped while this gate is ON: `kind::mxf4` is
///     e2m1 x e2m1 and cannot consume an e4m3 activation.
///   * the down direction is untouched: it consumes f32 activations
///     (`expert_gemv_fp4_down_reduce_kernel` does `acc += s_act[j] * w`, no fp4
///     quantisation of the act), i.e. it is already exact at e4m3 grade.
///
/// A stock .so (no `dsv41_expert_act_e4m3_cap`) leaves this OFF with a one-shot
/// notice: an armed-but-undispatchable gate that silently ran the OLD path is
/// the project's #1 measurement-bias trap.
pub(crate) fn expert_act_e4m3() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_EXPERT_ACT_E4M3").map(|v| v != "0").unwrap_or(false))
}

/// One-shot notice for an ARMED-but-undispatchable `DSV41_EXPERT_ACT_E4M3`. Called
/// at the quant/moe step when the gate is set but the loaded .so has no
/// `dsv41_expert_act_e4m3_cap` (the direct-e4m3 entry point), so the step still
/// runs the e2m1 single-pass path.
pub(crate) fn act_e4m3_skipped_note() {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        eprintln!(
            "warning: DSV41_EXPERT_ACT_E4M3 is set, but the routed experts still run the \
             e2m1 activation: the loaded .so has no `dsv41_expert_act_e4m3_cap` \
             (rebuild kernels/cuda: bash build.sh 103a). Any A/B run with this gate ON \
             measures the OLD path."
        );
    });
}

/// `DSV41_EXPERT_GROUPED=1` builds the **grouped (permuted)** routed layout:
/// the (row, slot) table is turned into expert-major row blocks
/// (`dsv41_route_group`), the activation rows are gathered into that order
/// (`dsv41_route_gather_rows`) and the expert output is scattered back to
/// `[m][topk][n]` (`dsv41_route_scatter_rows`). See
/// `kernels/cuda/dsv41_route.cu` for the layout contract.
///
/// WHY IT EXISTS. The tcgen05 e4m3 expert GEMM
/// (`dsv41_expert_gemm_e4m3_ext`, `tc5::e4x`) is expert-CENTRIC — ONE launch
/// covers ONE expert over a DENSE `[rows, k]` activation block — while
/// `moe_rows`' routing is per-(row, slot). Handing the per-(row, slot) table to
/// a dense launch would apply slot s's expert to rows that route elsewhere: a
/// SILENT WRONG ANSWER, which is why that arm declines today (`e4x_tile =
/// false` in `moe_rows`). The grouped layout is the missing precondition.
///
/// DEFAULT OFF, and read ONCE per process like every other gate here (a
/// per-call getenv is a CUDA-graph capture hazard).
///
/// STATUS: complete. The layout and the two movers are wired (`moe_route_grouped`)
/// and the consuming GEMM is issued (`moe_experts_grouped_gate_up` →
/// `dsv41_expert_gemm_e4m3_grouped`, the group-indexed MASKED M=128 tile). The
/// arm is still a numeric no-op ON ITS OWN, because that consumer also requires
/// `DSV41_EXPERT_TCGEN05_E4M3` (the `.so` reads both gates) — with only this one
/// armed the layout is built, nothing consumes it, and the proven per-(row, slot)
/// launches answer the step. That state is announced once, because an "ON" arm
/// that measures the OLD path is this project's #1 measurement-bias trap.
pub(crate) fn expert_grouped() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("DSV41_EXPERT_GROUPED")
            .map(|v| v.starts_with('1'))
            .unwrap_or(false)
    })
}

/// One-shot notice for an ARMED `DSV41_EXPERT_GROUPED`. Two distinct reasons it
/// can currently be a numeric no-op, both said out loud once:
///   * the loaded `.so` has no `dsv41_route_*` set (a stale build), or
///   * it HAS them and the grouped layout is built and used for every
///     cross-check, but the per-expert dense GEMM is not issued yet, so the
///     routed half still runs the per-(row, slot) launches.
/// In both cases an operator who exports the gate must not believe the step now
/// runs a grouped GEMM.
pub(crate) fn expert_grouped_skipped_note(reason: &str, so_has_symbols: bool) {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        eprintln!(
            "warning: DSV41_EXPERT_GROUPED is set, but the routed MoE still runs the proven \
             per-(row, slot) launches: {reason}. Any A/B run with this gate ON measures the OLD \
             path. (dsv41_route_* set present in .so: {so_has_symbols})"
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

/// COMPRESSOR-MROWS (`DSV41_COMPRESSOR_MROWS=1`, DEFAULT OFF): the verify block's
/// `m` compressor rows as ONE `dsv41_compressor_fused_mrows` launch per
/// compress-source layer instead of `m` `compressor_pool_on` + `compress_commit_on`
/// pairs (the pool entry itself launches the state carry + the pool, so the per-row
/// form is 3 kernel launches per row).
///
/// The rows are SERIAL BY CONSTRUCTION — the state carry's destination slot is
/// position-derived (rows r and r+2 share a slot at ratio 2) and the commit's
/// `*clen` / ring row is per-layer state — so the fused kernel loops `r` ascending
/// inside its single block, reproducing each row's pool/commit bytes exactly. What
/// the block-wide form does NOT reproduce is the READ SIDE's view of the shared
/// counters, and that is why the caller must take the per-row snapshot outputs:
///  * `clen_rows[owner][r]` — the counter after row r's commit — replaces the LIVE
///    `*clen` for this layer's (and its consumers') per-row readers: `sparse_attn`,
///    `comp_placeholder` and the indexer's `idx_lens`. Reading the live counter
///    after the hoisted launch hands every row the BLOCK-FINAL count, which is
///    exactly audit defect #1 (row 0 selects against its own future);
///  * `latent_rows[r]` — row r's pooled row — replaces the shared `latent` for the
///    per-row index key publish (`publish_index_key` reads it), which after the
///    launch only holds the block's LAST pooled row.
/// A layer whose compress is not hoisted keeps the live pointers, so the gate
/// being OFF (the default) is the per-row path verbatim. Declines are announced
/// once ([`Self::compress_rows_fused`]).
fn compressor_mrows() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_COMPRESSOR_MROWS").map(|v| v != "0").unwrap_or(false))
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

/// VERIFY_FORK (DSV41_VERIFY_FORK, DEFAULT OFF): the verify block (`layer_rows`
/// → `attention_rows` / `moe_rows`) takes the SAME three side-stream splits
/// `layer()` runs, so the whole verify pass (`verify_ms`) gets the decode path's
/// overlap instead of issuing every launch on one stream.
///
/// The three seams, each a faithful port of its EAGER twin (same kernels, same
/// operands, only the stream changes ⇒ the split is bit-identical by
/// construction, exactly like `DSV41_DUAL_CHAIN` / `DSV41_COMPRESS_SIDE` /
/// `DSV41_MOE_DUAL`):
///
///  1. **attention q chain vs kv chain** (`dual_chain_fork/join`,
///     `side_stream2`). The block-wide `wq_a`/`wkv` projections fill `qr_r` and
///     `kv_r`, and NOTHING after them reads both: the q chain
///     (`lin_rope_norm`/`norm_rows(qr_r)` + `proj_mrows(wq_b)` + the q rope)
///     touches only `qr_r`/`q_r`/`xq_r`/`xsc_r`, while the kv half
///     (`norm_rows(kv_r)` + the kv rope) touches only `kv_r` in place. The two
///     are therefore independent once the projections have landed — the
///     projections are the chains' last shared node — and the join sits before
///     the per-row interleave, whose `ring_append` is `kv_r`'s first consumer.
///
///  2. **compressor projections** (`compress_side_fork/join`, `side_stream3`).
///     Only `compress_proj_rows` is hoistable in the verify: it is a pure
///     function of `s.xn_r` and writes layer-private `kvp_r`/`scp_r`, so it is
///     issued on the third side stream at the SAME point as the dual chain's fork
///     and overlaps the whole attention chain. The join sits before
///     `compress_rows_fused` / the per-row `compress_row`, the first readers of
///     `kvp_r`/`scp_r`. ⚠️ NOT the pool/commit half: the verify's per-row
///     interleave (audit defects #1/#2) requires each row's commit to be issued
///     before that row's readers, so the state/pool/commit stays on the main
///     stream inside the loop.
///
///  3. **MoE routed vs shared expert** (`dual_chain_fork/join`, `side_stream2`).
///     The two halves read `xn_r` through DISJOINT quantisations (routed:
///     `xq4_r`/`xsc4_r`; shared: `xq`/`xsc`) and write disjoint scratch
///     (`ex_act_r`/`ex_down_r` vs `sh_act_r`). The one shared resource is the
///     accumulator: the routed down-reduce OVERWRITES `moe_out_r`, so the shared
///     half always writes its `w2` block to the DISJOINT `sh_out_r` scratch
///     (never `moe_out_r + r*dim`) and the join runs the same element-wise
///     `moe_out_r += sh_out_r` the serial path ran per row — same operands, same
///     order, one launch over `m*dim`.
///
/// Gated OFF (serial, byte-for-byte today's verify) on anything that would make
/// a seam a real race or a different program: `Device::supports_dual_chain()` /
/// `supports_compress_side()` (missing side stream or events), `swapAB`,
/// `cublas_m1` (one handle bound to the main stream), a compressor snapshot
/// (`spec_capture`, whose D2D copies run on the main stream), `skip_experts` /
/// `skip_shared_expert` / no shared expert on this rank, and the shared expert's
/// fused multi-row arms (`sh_exp_mrows` / `sh_pair_m` / `sh_exp_fused`), which
/// are main-stream programs this arm does not re-issue.
///
/// GRAPH SAFETY: legal inside a capture. The fork is `record(main)` +
/// `wait(side)` and the join is `record(side)` + `wait(main)` — both adjacent
/// pairs on `cudaEventDisableTiming` events (see `devrt.rs`), which is exactly
/// the pattern the whole-step graph (`DSV41_GRAPH_STEP`, ON by default) already
/// records for `layer()`'s three forks under the same
/// `cudaStreamCaptureModeRelaxed` capture, and the same one `DSV41_VERIFY_GRAPH`
/// would record here. Both sides of every pair are inside the capture, so the
/// two ops become graph edges; a capture that refuses them fails at
/// `capture_end` and the existing DRY→CAPTURE failure latch drops the request to
/// the direct launches, where the same calls are plain stream ops.
/// "0"/unset is the A/B arm (serial).
fn verify_fork() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_VERIFY_FORK").map(|v| v != "0").unwrap_or(false))
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

/// ROW-FOLD (DSV41_ROW_FOLD_ROPE=1, DEFAULT OFF): the verify block's q rope and
/// inverse o rope are `m` launches each per layer because every row owns a
/// position. The rows are INDEPENDENT (each rotates only its own `nlh` head rows
/// at `pos_rows[r]`), so the kernel can loop r ascending in ONE launch —
/// `dsv41_apply_rope_mrows`, whose body is `apply_rope_kernel`'s verbatim with
/// `t = pos_rows[r]`. The per-row loop stays as the fallback (bit-identical) for
/// an .so without the symbol. A/B arm: `DSV41_ROW_FOLD_ROPE=1`.
fn row_fold_rope() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_ROW_FOLD_ROPE").map(|v| v != "0").unwrap_or(false))
}

/// VERIFY-ROPE-MROWS (`DSV41_VERIFY_ROPE_MROWS=1`, DEFAULT OFF): `attention_rows`'
/// q rope as ONE `dsv41_apply_rope_mrows` launch instead of `m` `apply_rope`
/// calls at `off = r, step = 0` — the same kernel with the same argument pattern
/// the draft side's P3a a4 fold calls (`DsparkDev::rope_mrows`, `dspark_dev.rs`,
/// `row_stride = nh*hd`). This is the verify-plan's own arm (§10 of
/// `docs/agent/dspark-perf-400-plan.md`: the q-rope row is "semantically per-row
/// but not per-launch", `pos_rows` is already a device array).
///
/// It is also the q-rope HALF of what `DSV41_ROW_FOLD_ROPE` already covers: that
/// gate folds the q rope and the inverse o rope together behind one name, so it
/// cannot A/B the q rope alone (the o-rope side is now `DSV41_VERIFY_OROPE`'s
/// business anyway). Both gates are honoured — `row_fold_rope()` keeps its exact
/// historical behaviour, this one is additive — and with BOTH off the per-row
/// loop runs, which is the bit-identical target. Read ONCE and cached: this
/// branch runs 40x/step, inside graph capture.
fn verify_rope_mrows() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("DSV41_VERIFY_ROPE_MROWS")
            .map(|v| v != "0")
            .unwrap_or(false)
            || row_fold_rope()
    })
}

/// INDEXER-MROWS (`DSV41_INDEXER_MROWS=1`, DEFAULT OFF): `attention_rows` pays
/// the indexer's block-wide FRONT once for the whole block instead of once per
/// row — `idx_wq_b`'s fp8 projection + rope and `idx_weights`' bf16 GEMV, four
/// launches for `m` rows where the per-row form takes `4*m` (16 of the family's
/// 20 front launches per layer).
///
/// WHY ONLY THE FRONT (this is the boundary the verify-family-fusion plan draws
/// too optimistically — see the note on [`Self::indexer_rows_m`]): the SELECT
/// half (publish + `indexer_topk`) is per-row BY CONSTRUCTION inside this loop.
/// Row r's `*clen` must be row r's own (audit defect #1) and row r's window must
/// be appended immediately before row r's `sparse_attn` (audit defect #2 — the
/// window ring is exactly `window` slots wide, so row r+1's append destroys row
/// r's oldest entry). A select that consumed the block's per-row `clen` snapshot
/// would have to be issued after ALL of the block's commits, which drags it (and
/// every reader of `idxs_r`) past those appends. The front has no such coupling:
/// it is a pure function of `s.qr_r`, `s.xn_r` and `pos_rows`, all of which are
/// final before the interleave starts.
///
/// Bit-identity: every fused launch is the documented m-row twin of the per-row
/// program it replaces — `quant_rows`' row-indexed `(r, block)` map (exactly the
/// pair `attention_rows` already uses for wq_b/wo_b), `gemm_fp8_mrows_kernel`'s
/// per-row independent accumulator chain (the `C1-C6` argument on `proj_mrows`),
/// `apply_rope_kernel`'s body with the same per-row position from `pos_rows`
/// (the `DSV41_ROW_FOLD_ROPE` contract), and `gemv_bf16_nt_kernel<NT,WPR>`'s
/// per-row independent accumulator (the `DSV41_GATE_MROWS` contract). Every
/// decline (`swapab`, `cublas_m1`, an .so without a symbol, a shape outside the
/// kernel's bound) falls back to the per-row front verbatim, so the arm is
/// strictly additive. Read ONCE and cached — this branch runs 40x/step, inside
/// graph capture.
fn indexer_mrows() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("DSV41_INDEXER_MROWS")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

/// NORM-MROWS (`DSV41_NORM_MROWS=1`, DEFAULT OFF): the verify path's block norms
/// (attn_norm, ffn_norm, the q/kv norms of `attention_rows`, the final head
/// norm) run through the dsv41-side multi-row entry `dsv41_rmsnorm_rows` instead
/// of the GLM-reuse `ferrite_rmsnorm`.
///
/// **What it changes.** Both entries take a whole `[rows, dim]` block and give
/// every row its own block with its own `blockDim`-sized reduction tree, so the
/// launch count is the same either way and each row's arithmetic is the
/// single-row program's, statement for statement
/// (`dsv41_rmsnorm_rows_kernel`). What moves is OWNERSHIP: the verify's norms are
/// the only thing this chain still reaches into `ferrite_kernels.cu` for, and
/// the dsv41 entry spells the row count `rows`, like every other member of the
/// family (`dsv41_hc_collapse_norm`, `dsv41_apply_rope_mrows`,
/// `dsv41_quant_rows`). The arm exists so the A/B can prove the two kernels
/// agree row for row before any caller is moved permanently; it is additive, and
/// an `.so` without the symbol leaves every call exactly where it was.
///
/// Read ONCE and cached — this branch runs 4x/layer inside graph capture.
fn norm_mrows() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("DSV41_NORM_MROWS")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

/// ROW-FOLD (DSV41_ROW_FOLD_GATE=1, or the plan-named alias DSV41_GATE_MROWS=1;
/// both DEFAULT OFF): `moe_rows` runs the bf16 gate `gemv_bf16` once per
/// activation row. `gemv_bf16` dispatches to `ferrite_gemv_bf16_v2(...,
/// nrows = 1)` for the gate's shape (n = the routed expert count <
/// `GEMV_V2_MAX_N`), and the multi-row entry (`ferrite_gemv_bf16_v2_mrows` — the
/// same program as `ferrite_gemv_bf16_nt`, one definition in the .so) is that
/// program's m-row form: same WPR heuristic (`gv2_wpr(384) = 8`), same K-slice
/// walk, same uint4/8-element FMA groups, same smem fold order, one independent
/// accumulator per row (`gemv_bf16_nt_kernel`; `tests_gate_mrows.cu` asserts the
/// bit-identity at the gate's own WPR > 1 shape, which `tests_dsv41_head_mrows`
/// deliberately does not cover). DEFAULT OFF so the fold is an explicit A/B arm;
/// the fold itself only engages when the per-row path would take v2
/// (`gemv_bf16_v2_wanted` + the symbol present), which is what makes the parity
/// claim transfer. See `Device::gemv_bf16_v2_mrows`.
///
/// The two env names select the SAME program — `DSV41_GATE_MROWS` is the alias
/// `final-400-config` §5-P2 item 10 names the gate by, OR'd in rather than
/// replacing `DSV41_ROW_FOLD_GATE` so the arm that was already measured keeps
/// its name (the `DSV41_VERIFY_ROPE_MROWS`/`DSV41_ROW_FOLD_ROPE` precedent).
fn row_fold_gate() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("DSV41_ROW_FOLD_GATE").map(|v| v != "0").unwrap_or(false)
            || std::env::var("DSV41_GATE_MROWS").map(|v| v != "0").unwrap_or(false)
    })
}

/// DSV41_SH_EXP_MX2=0 reverts the shared expert's gate/up to two launches.
fn sh_exp_mx2() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_SH_EXP_MX2").map(|v| v != "0").unwrap_or(true))
}

/// SHARED-EXPERT ROW-FOLD (`DSV41_SH_EXP_MROWS=1`, DEFAULT OFF): the shared
/// expert's (w1/w3 -> swiglu -> w2) chain runs as ONE multi-row pass over the
/// verify block ([`DevChain::shared_expert_mrows`]) instead of the per-row loop.
///
/// **What it removes.** The shared expert is a SINGLE expert applied to every
/// row, so its weights are identical across the block — but `moe_rows` still
/// reads them once per row (4.42 MB x m per layer, 4-5 launches x m per layer).
/// At the production shape (m = 5, `sh_il` = 288, dim = 5120) that is 26.5 MB
/// and ~25 launches per layer = **~0.89 GB/step and ~960 launches/step** of pure
/// repeat traffic, the same class of redundancy the routed experts' `rows`
/// dimension and the projections' `mrows` already removed. Unlike the routed
/// half — whose rows select DIFFERENT experts, so its byte count is irreducible
/// and only its launch count can shrink — this one's bytes really do divide by m.
///
/// **Why it is not flipped on.** Every step is documented as bit-identical to
/// the call it replaces (see the method), but this project's rule is that a new
/// multi-row path ships as an explicit A/B arm first: the acceptance criterion
/// is `verify_ms` (plan §3.3/§4) plus the row-level `dspark_parity` check.
/// Read ONCE and cached — this branch runs 40x/step inside the layer loop, so a
/// per-call getenv would be exactly the hot-path slip the other gates avoid.
fn sh_exp_mrows() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_SH_EXP_MROWS").map(|v| v != "0").unwrap_or(false))
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

/// SHARED-EXPERT FUSED CHAIN (`DSV41_SH_EXP_FUSED=1`, DEFAULT OFF): the arm that
/// finally CALLS `dsv41_gemm_fp8_sh_pair` — the symbol `sh_pair()` above was
/// written for and never reached. Same three bodies, ONE launch per row instead
/// of that row's five:
///
///     5 launches/row : quant1 -> gemm_fp8_mx2(w1|w3) -> swiglu_limit
///                      -> quant1 -> gemm_fp8_mx_add(w2)
///     1 launch/row   : gemm_fp8_sh_pair (w1|w3 + swiglu + fp8 emit
///                      | grid barrier | w2)
///
/// ⚠️ WHAT IT IS **NOT**. The kernel as written is **M = 1**: phase 1's grid maps
/// onto the `inter` rows of ONE activation row (`a` is a single `k1 = dim`-byte
/// row staged block-wide) and nothing indexes the verify's `m` dimension. So this
/// arm issues `m` launches, not one — it does not reach the design's "1 launch
/// per layer" (`docs/agent/verify-family-fusion.md` §3.1 W3), which needs the
/// `template<M>` form and is a separate piece of work. What it does remove, at
/// the production shape (m = 5): 25 launches/layer -> 7 (one `quant_rows`, m
/// fused, one `add_inplace_raw`), i.e. ~720 launches/step of the audit's 3.3us
/// class, and the two serialisation points inside each row's chain (weight
/// staging + the swiglu round trip now live in one grid).
///
/// ⚠️ THE ADD IS STILL A SEPARATE LAUNCH. The kernel's phase 2 STORES
/// (`out[row] = acc`); it has no accumulate/add epilogue, so unlike the A5
/// `gemm_fp8_mx_add` path the w2 result cannot fold straight into `moe_out_r`.
/// The arm writes `sh_out_r` and keeps the mrows pass's single element-wise add.
///
/// Both `DSV41_SH_EXP_FUSED=1` AND `DSV41_SH_PAIR=1` are required (`sh_pair()` is
/// the A/B arm for the kernel itself, still default OFF), plus the .so actually
/// carrying `dsv41_gemm_fp8_sh_pair` (`supports_sh_pair()`).
///
/// Read ONCE and cached — this branch runs 40x/step inside graph capture, the
/// hot-path rule every gate here follows.
fn sh_exp_fused() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_SH_EXP_FUSED").map(|v| v == "1").unwrap_or(false))
}

/// M-ROW shared-expert chain (`DSV41_SH_PAIR_M=1`, DEFAULT OFF): the `template<M>`
/// form of the fused chain above — ONE launch for the whole verify block instead
/// of one per row.
///
///     5 launches/row  : quant1 -> gemm_fp8_mx2(w1|w3) -> swiglu_limit
///                       -> quant1 -> gemm_fp8_mx_add(w2)
///     1 launch/row    : gemm_fp8_sh_pair            (the M=1 arm above)
///     1 launch/layer  : gemm_fp8_sh_exp_fused<M>    (THIS) + `quant_rows`
///
/// Phase 1 maps the grid over (i-tile, activation row group) so the block count
/// is `ceil(sh_il/32) * ceil(m/fold_r)` instead of `ceil(sh_il/32)`, which is
/// what makes one launch cover the whole block; phase 2 carries `m` accumulators
/// over the same w2 row (the parallel axis there is `ceil(dim/32)`, so folding
/// costs nothing) and can fold the trailing `add_inplace_raw` into its store.
/// Every output is bit-identical to the per-row chain by the kernel's C1-C8
/// contract (`kernels/cuda/dsv41_kernels.cu`).
///
/// Default OFF, exactly like the other new-kernel arms here: the decline path and
/// the parity cases are what bring it up, and the four-text check is what flips
/// it. Requires the two SH_EXP envs off the critical path — this arm is tried
/// BEFORE the M=1 fused arm in `shared_expert_mrows`.
fn sh_pair_m() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_SH_PAIR_M").map(|v| v == "1").unwrap_or(false))
}

/// The phase-1 fold knob for `DSV41_SH_PAIR_M` (`DSV41_SH_PAIR_M_FOLD`, default
/// 1): how many activation rows one block owns in phase 1.
///
/// 1 (default) is the design's §3.3 winner — it keeps the M=1 kernel's exact
/// consume path (`s_af[j]`, the block-wide predecoded activation) AND its
/// parallel axis, `ceil(sh_il/32) * m` chains. >1 folds the rows into the warp's
/// registers (the mrows form) and shrinks the grid: fewer instructions, less SM
/// coverage. It is a RUNTIME kernel argument, so the sweep does not recompile.
///
/// Read ONCE and cached (hot-path rule).
fn sh_pair_m_fold() -> i32 {
    static F: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("DSV41_SH_PAIR_M_FOLD")
            .ok()
            .and_then(|v| v.trim().parse::<i32>().ok())
            .filter(|v| (1..=VERIFY_ROWS as i32).contains(v))
            .unwrap_or(1)
    })
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

/// `DSV41_VERIFY_HEAD_FOLD=1` re-enables the folded multi-row
/// `head_gemv_bf16_mrows` for the verify's head. **Default OFF (the per-row
/// `gemv_bf16`) because the folded path is measurably WRONG for this model.**
///
/// The folded kernel's header argued bit-identity with "the single-row launch
/// it replaces" (C1-C5); the production single-row head is
/// `dsv41_gemv_bf16` → `gemv_bf16_kernel` (`gemv_bf16_v2_wanted(n)` needs
/// `n < 2048`, and the head's `n = vocab_size = 129280`, so the v2/nt path is
/// NOT taken — `folded-head-korder` established this from the code). The folded
/// kernel's own scalar `for (c = lane; c < k; c += 32)` chain differs from it in
/// the fp8/fma pairing details, and the measured effect on serve is large:
///   FOLD=1: `verify_out[0] == next` (the echo) on 33% of verify rows
///   FOLD=0: 9%   (22 steps, 出师表) — and the text's adjacent repeats drop 6 → 4.
/// The echo is what makes `emitted = [next] + verify_out[..]` write `next` twice.
/// Cost of FOLD=0: the head's 1262 MB weight is streamed ~m times instead of once
/// (~+0.7 ms/step) — correctness over that, until the folded kernel is proven
/// bit-identical to v1. Read once and cached (the house rule for hot-path gates).
fn verify_head_fold() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("DSV41_VERIFY_HEAD_FOLD")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

/// `DSV41_VERIFY_HEAD_MROWS=1` (DEFAULT OFF) folds the DSpark verify's SLICED
/// head into ONE **v1-order** multi-row launch (`dsv41_gemv_bf16_v1_mrows`)
/// instead of `m` per-row `gemv_bf16` calls.
///
/// **Why v1 order, and why this is NOT [`verify_head_fold`].** The per-row
/// program the sliced verify head runs IS v1: `gemv_bf16_v2_wanted(n)` diverts
/// only `n < 2048`, and the head's per-rank `n` is `seg = vocab / world` (16160
/// at world = 8), so the per-row call lands on `dsv41_gemv_bf16` /
/// `gemv_bf16_kernel`. The existing folded entry (`head_gemv_bf16_mrows`, the
/// thing `verify_head_fold` re-enables) is the **v2** (`gemv_bf16_nt`, WPR == 1)
/// program — a NUMERICAL change against the v1 head, and it measured as one
/// (33% echo). This gate takes the **v1 fold** instead, whose row r is
/// bit-identical to the per-row launch it replaces: the same `c = lane; c += 32`
/// scan, one independent accumulator per row, v1's `__shfl_xor_sync` tree, no
/// K-split (the kernel header in dsv41_glue.cu carries the transcription
/// argument). So it changes the head's TRAFFIC (`m` passes over the slice →
/// one) and its LAUNCH COUNT (`m` → 1) and **not its values** — the design's
/// 1.12ms → 0.25-0.4ms arm (`docs/agent/verify-family-fusion.md`).
///
/// Both folds are mutually exclusive and neither is reachable from the other:
/// `verify_head_fold` only fires on the UNSLICED arm (`geom.is_none()`), this
/// one only on the SLICED arm.
///
/// Default OFF: the arm is new and needs an `.so` carrying
/// `dsv41_gemv_bf16_v1_mrows`; a stale one falls back to the per-row loop (the
/// method returns `Ok(false)`), as does `m > 8`. Read once and cached — this
/// branch runs at the verify's head, inside graph capture.
///
/// **The strict `== "1"`** (where most gates in this file test `!= "0"`) is
/// deliberate, not an oversight: this fold is an opt-in A/B arm, so a value
/// that is exported but empty, or mistyped, must not arm it.
fn verify_head_mrows() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("DSV41_VERIFY_HEAD_MROWS")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// The verify head's SLICE conditions, evaluated by `DevChain::verify_head_geom`
/// and carried as ONE value so the DECISION and its one-shot decline report
/// cannot drift: a second copy of the condition list is exactly how the report
/// would start lying about why the layout fell back.
///
/// The fields mirror `verify_head_geom`'s doc verbatim — the gate, a BF16 head,
/// `world > 1`, a vocabulary divisible by the world, the batched argmax symbol,
/// a live v5 collective, and room for the WHOLE block's keys in its slot.
struct VerifyHeadGeom {
    env: bool,
    head_bf16: bool,
    world: usize,
    vocab: usize,
    symbol: bool,
    v5: bool,
    slot_fit: bool,
    slot_bytes: usize,
}

impl VerifyHeadGeom {
    /// The predicate itself, spelled ONCE. `world > 1` short-circuits the
    /// divisibility test, so a degenerate `world == 0` cannot panic here.
    fn sliced(&self) -> bool {
        self.env
            && self.head_bf16
            && self.world > 1
            && self.vocab % self.world == 0
            && self.symbol
            && self.v5
            && self.slot_fit
    }

    /// One `&'static str` per FAILED condition, in predicate order — empty when
    /// the geometry IS available. Only ever built from the one-shot note below,
    /// so the steady-state hot path pays nothing for it.
    fn declines(&self) -> Vec<&'static str> {
        let mut v: Vec<&'static str> = Vec::new();
        if !self.env {
            v.push("DSV41_VERIFY_HEAD_SLICED=0 (the gate is off, i.e. the full head is the intended A/B arm)");
        }
        if !self.head_bf16 {
            v.push("the head weight is not BF16 (the sliced arm is the bf16 GEMV)");
        }
        if self.world <= 1 {
            v.push("world <= 1 (no cross-rank exchange to slice across)");
        } else if self.vocab % self.world != 0 {
            v.push("vocab_size is not divisible by the world");
        }
        if !self.symbol {
            v.push(
                "the loaded .so has no `dsv41_argmax_sliced_rows` (rebuild kernels/cuda: \
                 bash build.sh 103a)",
            );
        }
        if !self.v5 {
            v.push("the collective is not AR v5 (`comm.uses_v5()` is false)");
        }
        if !self.slot_fit {
            v.push(
                "the v5 slot is too small for the whole verify block (VERIFY_ROWS * 8 > \
                 comm.bytes)",
            );
        }
        v
    }

    /// One-shot decline report for the verify head's SLICE geometry. Every
    /// FAILED condition is named, so "the verify head is not sliced" stops being
    /// a fact with no cause — the mrows-decline investigation had to re-derive
    /// these seven conditions by hand because this line did not exist (the
    /// fallback path had zero observability).
    fn note_once(&self) {
        static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        ONCE.get_or_init(|| {
            eprintln!(
                "note: the DSpark verify head runs the FULL-vocabulary per-row path and \
                 `logits_r` keeps the vocab pitch: the SLICE geometry's conditions are not all \
                 met — {}. (world={}, vocab={}, comm.bytes={})",
                self.declines().join("; "),
                self.world,
                self.vocab,
                self.slot_bytes,
            );
        });
    }
}

/// One-shot notice for an ARMED `DSV41_VERIFY_HEAD_MROWS` whose fold did NOT
/// run, or ran on an arm the operator did not expect. Before it, the gate's only
/// failure mode was SILENT: the fold's call site sits inside `verify_head_geom`'s
/// `Some` branch, so ANY one of that geometry's seven conditions made the gate a
/// NO-OP that printed nothing — the STRUCTURAL DEAD GATE the mrows-decline
/// investigation found, and the project's #1 measurement-bias trap (an "ON" arm
/// that measures the OLD path).
///
/// `arm_sliced` records which arm the fold was attempted on and `performed` what
/// the kernel actually answered, giving three distinct states:
///
/// * UNSLICED arm, performed — the gate is ALIVE, just off the sliced geometry:
///   the same v1 kernel is bit-identical on the full-vocabulary head (`n = vocab`
///   is still `>= GEMV_V2_MAX_N`, so the per-row program it replaces is v1, NOT
///   the v2/nt fold `verify_head_fold` claims).
/// * UNSLICED arm, not performed, no BF16 head — STRUCTURALLY DEAD for this
///   model: the v1 kernel is bf16-only and there is no slice geometry to fall
///   back to.
/// * either arm, not performed with a BF16 head — the kernel declined
///   (`Ok(false)`): a stale .so without `dsv41_gemv_bf16_v1_mrows`, or a block
///   whose rows are outside its `1..=8` dispatch set.
fn verify_head_mrows_note(
    arm_sliced: bool,
    head_bf16: bool,
    performed: bool,
    vocab: usize,
    m: usize,
) {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        if performed {
            eprintln!(
                "note: DSV41_VERIFY_HEAD_MROWS is set: the v1 multi-row fold runs on the \
                 UNSLICED head (no slice geometry — see the note above), i.e. ONE pass over the \
                 {vocab}-row head weight serves all {m} verify rows, and row r is bit-identical \
                 to the per-row gemv_bf16 it replaces."
            );
        } else if !head_bf16 {
            eprintln!(
                "warning: DSV41_VERIFY_HEAD_MROWS is set, but the verify head is not BF16 and no \
                 slice geometry is available: the v1 multi-row kernel is bf16-only, so this gate \
                 is STRUCTURALLY DEAD for this model. The per-row head runs, and any A/B run with \
                 this gate ON measures the OLD path."
            );
        } else {
            eprintln!(
                "warning: DSV41_VERIFY_HEAD_MROWS is set, but the v1 multi-row fold did not run \
                 on the {} head: the loaded .so has no `dsv41_gemv_bf16_v1_mrows` (rebuild \
                 kernels/cuda: bash build.sh 103a), or the block's {m} rows are outside the \
                 kernel's 1..=8 dispatch set. The per-row head runs, so any A/B run with this \
                 gate ON measures the OLD path.",
                if arm_sliced { "sliced" } else { "full-vocabulary" },
            );
        }
    });
}

/// `DSV41_VERIFY_HEAD_SLICED` (DEFAULT ON) gives the DSpark verify's head the
/// same per-rank VOCABULARY SLICE the eager step takes under `DSV41_HEAD_SLICE`:
/// each rank projects its own `seg = vocab / world` rows of `head.weight`, and
/// the ranks pick the global winner PER ROW from one published u64 each.
///
/// **What it removes.** `head.weight` is `Shard::Replicated` [129280, 5120] bf16
/// = 1262 MB on EVERY rank, and the unsliced verify's head streams all of it
/// once PER ROW (m = 6 rows => 6 x 298us = 1.79ms; `verify-calc-floor.md`'s
/// 6619 MB/step). With the slice one rank reads 158 MB per row for the same m
/// launches, and the head is replicated, so the whole vocabulary is still
/// compared — the exchange's packed key carries the GLOBAL index.
///
/// **Why per-row `gemv_bf16(n = seg)` and not the folded
/// `head_gemv_bf16_mrows`.** The fold's kernel IS the v2 (`gemv_bf16_nt`)
/// program, while the production single-row head is v1 `gemv_bf16_kernel`
/// (`gemv_bf16_v2_wanted(n)` needs `n < 2048` and the head's `n` is the
/// vocabulary) — so folding the head is a NUMERICAL change, and it measured as
/// one (see [`verify_head_fold`]). The per-row call here is the SAME kernel the
/// eager sliced head runs, so each verify row keeps the eager head's exact
/// values. The fold is a separate, still-open correctness question and stays an
/// opt-in A/B arm that this gate takes precedence over.
///
/// **Why the argmax is ONE batched exchange, not m calls of `argmax_sliced`.**
/// Every `dsv41_argmax_sliced` call IS a full v5 epoch round — publish, stamp,
/// advance `*epoch`, poll every peer — and only `pos_ctr` can be nulled, never
/// the round. m rows would put m serialised cross-rank round-trips on the
/// verify's critical path (and m device epoch advances per step); the batched
/// `dsv41_argmax_sliced_rows` pays ONE for the whole block. `pos_ctr` is NULL
/// throughout: the verify must not advance it (the accept logic owns that, once
/// per accepted prefix).
///
/// The head is REPLICATED, so the full-vocabulary rounds stay available as the
/// fallback: a stale .so without the batched symbol, `world == 1`, a non-BF16
/// head or a vocabulary not divisible by the world each keep the per-row
/// full-vocab path, and [`verify_head_geom`] is the one place that decides.
/// `DSV41_VERIFY_HEAD_SLICED=0` restores it for the A/B. Read once and cached
/// (the house rule for hot-path gates).
fn verify_head_sliced() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("DSV41_VERIFY_HEAD_SLICED")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// `DSV41_SIDS_WRITEBACK=1` re-enables the s.ids write-back (emitted.last()
/// written back after the spec commit). **Default OFF** because the bisection
/// (5f774c7 vs 65d6b5a on the SAME GPU/binary pair) showed the write-back
/// makes the 出师表 collapse to LEN 12 ("出师nofollow") while its parent
/// (repetition-only) runs LEN 54 — the write-back feeds a MORE-correct token
/// (emitted.last() = verify's argmax), which AMPLIFIES verify_out's own value
/// errors into an earlier collapse. The write-back is mathematically right
/// (two independent verdicts verified it); the blocker is verify's values.
/// Re-enable once the verify-value fault is fixed (spec-eager-diff-tool is
/// building the locator). Read once and cached (the house rule).
///
/// # The invariant (ALL three spec arms)
///
/// A round leaves `s.ids` == `emitted.last()`, the token at the NEW `pos_ctr`:
///
/// | arm | block (row 0 at) | `emitted` | `emitted.last()` at | commit |
/// |---|---|---|---|---|
/// | legacy | `[d1..d5]` @ `pos+1` | `[next] ++ verify_out[..k_acc]` | `pos+1+k_acc` | `commit(pos+1, 5, k_acc)` |
/// | aligned | `[next, d1..d5]` @ `pos+1` | same (index-aligned) | `pos+1+k_acc` | `commit(pos+1, 6, k_acc)` |
/// | swallowed | `[token, d1..d5]` @ `pos` | same (== `rows[..k_emit]`) | `pos+k_emit` | `commit(pos, 6, k_emit)` |
///
/// One construction, three layouts: every arm's `verify_out[j]` is the argmax of
/// the block row fed `drafts[j]`, so `emitted[i]` is the token at `pos + 1 + i` in
/// all three — only row 0's position differs, which is what the last two columns
/// record (`k_emit = k_acc + 1`, so the swallowed arm's `pos + k_emit` is the same
/// counter the other two reach from `pos + 1`; there row 0 is the anchor itself).
/// All three must write back: the swallowed arm reads nothing from `s.ids`, but
/// the round after it may be the legacy bootstrap one (un-primed chain, or after a
/// failure), and THAT `step_dev` embeds `s.ids`.
fn sids_writeback() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("DSV41_SIDS_WRITEBACK")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

/// `DSV41_INV_CHECK=1` arms the spec path's cross-step INVARIANT ASSERTIONS.
///
/// Default OFF, so the steady-state hot path pays nothing: every check below is
/// gated on this one cached flag, and with it off each call site is a single
/// predictable branch on a `OnceLock` value (the same discipline the other gates
/// here follow — no `getenv` on a per-step path).
///
/// Each check is one of two kinds, and the kind fixes its cost:
///
/// * **static / constructive** — a `debug_assert!` or a compile-time `const`
///   assertion. Zero runtime cost in a release build, and it cannot be "passed"
///   by a wrong value: it asserts a SHAPE (a length, a row pitch, a constant
///   relationship). These catch the class of defect that a buffer's Rust type
///   cannot express — `logits_r`'s row pitch, `dspark_tap_r`'s stride, the fp8
///   staging's row geometry.
/// * **D2H spot check** — one 4-byte device→host read of the single scalar whose
///   producer/consumer relationship the check names (a position, a length, an
///   argmax). The read is the cost; the comparison is free.
///
/// The invariant behind each check and the exact site it runs at are recorded on
/// the check itself (`inv_*` in the `impl DevChain` block). A failure prints one
/// `[inv-fail]` line naming the invariant, the expected and the seen value, and
/// returns `Err` — so the first step that breaks an invariant is the step that
/// reports it, not the later step whose output merely looks wrong.
fn inv_check() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_INV_CHECK").map(|v| v != "0").unwrap_or(false))
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

/// `DSV41_TOPK_DUMP=1` arms the first-token logits probe: the head's row is read
/// back and its top `DSV41_TOPK_N` entries printed as one `[topk]` line per step
/// (see [`DevChain::dump_topk`]).
///
/// This is the ferrite half of the backbone first-token alignment experiment:
/// the official reference's top-10 on the same prompt's token sequence against
/// this line's, the first rank that disagrees being the deviation signal. The
/// step is dumped AFTER the head projection and BEFORE the argmax dispatch, so
/// the row is exactly the one the argmax consumes.
///
/// DEFAULT OFF, and — like `DSV41_STATS` — it must also disable the whole-step
/// CUDA graph: the dump is a device→host read, which is illegal inside a
/// capture. The gate is cached in a `OnceLock` (read once per process), so the
/// steady-state path pays a single predictable branch.
fn topk_dump() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_TOPK_DUMP").map(|v| v != "0").unwrap_or(false))
}

/// `DSV41_TOPK_N` (default 10): how many ranks `DSV41_TOPK_DUMP` prints.
fn topk_dump_n() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("DSV41_TOPK_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(10)
            .max(1)
    })
}

/// `DSV41_LAYER_DUMP=1` arms the per-layer residual probe: one `[layer]` line
/// per layer carrying the L2 norm of the residual stream the block just finished
/// writing (see [`DevChain::dump_layer_norm`]).
///
/// It is the bisection half of the alignment experiment: comparing the per-layer
/// norms against the reference's localises the FIRST layer whose backbone
/// departs, which the first-token logits alone cannot name.
///
/// DEFAULT OFF; like `DSV41_TOPK_DUMP` it disables the whole-step CUDA graph,
/// because the norm read is a device→host transfer.
fn layer_dump() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_LAYER_DUMP").map(|v| v != "0").unwrap_or(false))
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

/// DSV41_VERIFY_GRAPH=1 captures the DSpark verify block (`step_rows`) into its
/// OWN CUDA graph, alongside the whole-step `step_graph`.
///
/// DEFAULT OFF (A/B): the whole-step graph had to be flipped on only after a
/// measurement campaign, and this one is bigger (≈7000 nodes/verify), so the
/// opt-in is deliberate — `DSV41_VERIFY_GRAPH=1` vs unset is the A/B pair and
/// the acceptance criterion is `verify_ms` (see the perf plan §3.3/§4).
///
/// Only the env switch lives here; the per-chain conditions that must ALSO hold
/// (no `DSV41_ENG_HOST`, the P0 snapshot kernels, the device-side AR, an async
/// memset, no probe modes) are checked in [`DevChain::verify_graph_gate`],
/// because two of them are properties of the loaded `.so`.
fn verify_graph_want() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_VERIFY_GRAPH").map(|v| v != "0").unwrap_or(false))
}

/// The shape-tagged name of a verify graph: `verify_graph_m{m}`.
///
/// The pool holds up to [`VERIFY_GRAPH_SLOTS`] coexisting graphs (one per row
/// count), so the `[verify_graph]` diagnostics tag each capture line with the
/// shape it belongs to — an A/B log then shows WHICH shape engaged instead of a
/// single anonymous "captured".
fn verify_graph_name(m: usize) -> String {
    format!("verify_graph_m{m}")
}

/// The four arms of [`DevChain::step_rows_sync`] a verify block can run in, as the
/// RANK-VISIBLE code the arms vote on (Plan B, "unanimity-or-direct").
///
/// The code exists because the arms do not cost the same number of `host_barrier`
/// arrivals per block, and `SpinBarrier` counts arrivals per epoch: two ranks in
/// different arms close epochs their peers never entered, and every later barrier
/// in the request is one generation out — the stall `ar5-hang` traced. Making the
/// ranks AGREE on the arm is what keeps the counts equal; when they cannot agree,
/// the world takes [`VerifyArm::Direct`], the one arm that exists in every state
/// (the graph is an optimisation over those launches, never a different program).
///
/// Voting on the ARM rather than on "graph or not" is the point: `Dry`, `Capture`
/// and `Replay` all return a slot index, so a `Some`/`None` vote would let a rank
/// that is DRY-ing a fresh shape and a peer that is REPLAYING one "agree" while
/// walking arms with different rendezvous counts (`Capture` alone takes two — it
/// brackets its recording). The three are therefore distinct codes, and only
/// `Dry`/`Replay` — the pair with equal counts — are adjacent in meaning.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum VerifyArm {
    /// No graph: the caller's direct launches. One rendezvous whenever the graph
    /// is configured on (`DSV41_VERIFY_GRAPH`), zero when it is off.
    Direct = 0,
    /// The shape's first-use DRY execution (a REAL run, to move every lazy
    /// first-use cost outside the capture). One rendezvous.
    Dry = 1,
    /// Record the shape's graph. TWO rendezvous: the recording is bracketed.
    Capture = 2,
    /// Launch the stored graph. One rendezvous.
    Replay = 3,
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

/// R2 (DSV41_ATTN_LIN_FUSE, **DEFAULT OFF** — A/B first): `attention_rows` at
/// `m == 1` reuses the EAGER decode path's two single-row fusions instead of the
/// mrows-ified sequence it inherited when the verify path went multi-row.
///
/// The projection-family analysis found the regression: on the mrows path a
/// single-row verify costs
///   `quant_rows(xn) + proj_mrows(wq_a) + proj_mrows(wkv)`
///     + `norm_rows(qr) + quant_rows(qr) + proj_mrows(wq_b) + apply_rope`
/// = 7 projection-family launches per layer, while EAGER's `attention()` pays
/// TWO — `lin2` (wq_a + wkv over one shared `xq`, DSV41_PROJ_FUSE default ON)
/// and `lin_rope_norm` (the rmsnorm+fp8 prologue and the rope epilogue folded
/// into the wq_b GEMV, DSV41_NORM_FUSE default ON). Both kernels are already in
/// the tree and in the loaded .so; the mrows化 simply never called them at
/// `m == 1`. This gate routes exactly those two calls.
///
/// It is `m == 1`-only by construction: `lin2` and `lin_rope_norm` are the
/// `m = 1` programs (`gemm_fp8_mx2` / `gemm_fp8_mx_rope_norm`), and it is the
/// mrows path that is the right one above one row. `swapAB` is excluded because
/// `lin2` bypasses the `gemm_fp8_mx_or_swap` dispatch the per-row path honours,
/// so the fused call would silently switch a live experiment knob's numerics.
///
/// Every consumer downstream of `qr_r` sees byte-identical input: the fused
/// wq_b launch leaves `qr_r` RAW (its prologue normalised into shared memory),
/// and the indexer's q half — the only other `qr_r` reader — normalises it inside
/// its OWN gemv prologue (`indexer_qr_raw`, R2b, default ON) or, with R2b off,
/// consumes the `norm_rows`-materialised copy this path then pays for. Either way
/// the bytes the indexer's GEMV dot consumes are the same ones. A decline
/// (an .so without the symbol, a shape the launcher rejects, `PROJ_FUSE=0`)
/// falls straight through to today's mrows sequence, so OFF and ON-but-declined
/// are the same code path.
///
/// **BISECT SPLIT.** The two fusions are independent kernels with independent
/// decline codes, and the "bit-identical" argument for each is constructive (the
/// mrows kernel is *claimed* to be `gemm_fp8_mx`'s `m == 1` program) — never
/// measured on the real shapes. So the top suspect for a corruption under `=1`
/// is one SPECIFIC half, not the pair:
///   - `2` = `lin2` only  (wq_a + wkv fused; wq_b on `norm_rows + quant_rows +
///     proj_mrows + apply_rope`, 4 launches)
///   - `3` = `lin_rope_norm` only (wq_a/wkv on `quant_rows + proj_mrows ×2`;
///     wq_b on ONE `gemm_fp8_mx_rope_norm`)
/// `1` (both) is byte-for-byte the R2 that shipped; unset / `0` / anything else
/// is OFF. `2` and `3` are bisect arms, not production settings.
fn attn_lin_fuse() -> AttnLinFuse {
    static F: std::sync::OnceLock<AttnLinFuse> = std::sync::OnceLock::new();
    *F.get_or_init(|| match std::env::var("DSV41_ATTN_LIN_FUSE").as_deref() {
        Ok("1") => AttnLinFuse::Both,
        Ok("2") => AttnLinFuse::Lin2,
        Ok("3") => AttnLinFuse::RopeNorm,
        _ => AttnLinFuse::Off,
    })
}

/// K1 (DSV41_ATTN_MROWS2, **DEFAULT OFF** — A/B first): `attention_rows` runs
/// the verify's wq_a + wkv pair as ONE two-family `gemm_fp8_mrows2` launch
/// instead of two `proj_mrows` launches. This is the R2 replacement's first half
/// (`docs/agent/verify-specific-fusion-kernel-design.md` §3).
///
/// WHY IT IS NOT R2. R2 reused EAGER's `lin2` (`gemm_fp8_mx2`), a DIFFERENT
/// program from the mrows family the verify path uses, and it touched the
/// single-row scratch `s.xq`/`s.xsc`; both classes of risk are excluded here by
/// construction. K1's kernel is `gemm_fp8_mrows_kernel<M>` with the family folded
/// in as four base-pointer selections (weight / scale / output base / row index),
/// so the K walk, the `acc[r] += av * wv` chain, the `shfl_xor` tree and the
/// staged bytes are the mrows kernel's, verbatim: row `row` of family `f` is
/// bit-identical to row `row` of the `gemm_fp8_mrows` launch it replaces. The
/// activation source stays `s.xq_r`/`s.xsc_r` (the caller's own `quant_rows`),
/// never `s.xq`/`s.xsc`, and no `quant1` consume-once flag is involved.
///
/// Default OFF: unset is byte-identical to the previous behaviour. A stale .so
/// without the symbol, a declined shape, or this gate off falls straight through
/// to the two `proj_mrows` launches, which are K1's bit-exact reference.
///
/// PRECEDENCE: when K1 and R2's `lin2` half are BOTH enabled, K1 runs (it is the
/// verified-safe program) and R2's half is not reached. Both gates default OFF,
/// so this only decides the meaning of a configuration nobody ships.
fn attn_mrows2() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_ATTN_MROWS2").map(|v| v == "1").unwrap_or(false))
}

/// R2's two halves, selected by `DSV41_ATTN_LIN_FUSE` (see [`attn_lin_fuse`]).
///
/// R2 shipped as one on/off gate over TWO independent fusions, and a corruption
/// under it cannot say which half is at fault — so the gate is split into the
/// two sub-switches that make that a one-shot bisect.
#[derive(Clone, Copy)]
enum AttnLinFuse {
    /// Neither half (unset, `"0"`, or any unrecognised value).
    Off,
    /// R2's first half only: [`Self::lin2`] (wq_a + wkv over one shared `xq`);
    /// wq_b keeps its `mrows` sequence.
    Lin2,
    /// R2's second half only: [`Self::rope_norm`] (norm + wq_b + rope in one
    /// launch); wq_a/wkv keep their `mrows` sequence.
    RopeNorm,
    /// Both halves — the `=1` behaviour R2 shipped.
    Both,
}

impl AttnLinFuse {
    /// May `attention_rows` take EAGER's `lin2` (wq_a + wkv)?
    fn lin2(self) -> bool {
        matches!(self, Self::Lin2 | Self::Both)
    }

    /// May `attention_rows` take EAGER's `lin_rope_norm` (norm + wq_b + rope)?
    fn rope_norm(self) -> bool {
        matches!(self, Self::RopeNorm | Self::Both)
    }
}

/// R2b (DSV41_INDEXER_QR_RAW, **default ON, only reachable under R2**): the
/// indexer's q half consumes the RAW `qr_r` that R2's fused wq_b launch left
/// behind, normalising it inside its own gemv prologue, instead of the copy R2
/// had to materialise with a compensating `norm_rows` launch.
///
/// R2 (above) routed `m == 1`'s wq_b through EAGER's `lin_rope_norm`, which
/// normalises `qr_r` into SHARED MEMORY and never writes the row back — so R2
/// owed every later `qr_r` reader the normalisation. `attention_rows` paid it as
/// one `norm_rows` over the block, which kept the indexer's input byte-identical
/// (its stated zero-risk property) but left ONE launch per layer of the
/// projection family's budget on the table. R2b removes exactly that launch by
/// handing the indexer the fused launch EAGER's `indexer()` already uses for this
/// very purpose (`s.qr_raw` → `lin_rope_norm(idx_wq_b)`): the prologue normalises
/// the raw row with `rmsnorm_q_kernel`'s arithmetic term for term and quantises
/// it into shared memory, so the GEMV's dot — and therefore `idx_q_r` — is the
/// bytes the unfused (`norm_rows` + `quant1`) pair produced.
///
/// It is `m == 1`-only by construction: `lin_rope_norm` is the `m = 1` program,
/// and [`attn_lin_fuse`] is what sets `s.qr_raw_r` in the first place. Both
/// indexer q-half sites are covered — the block-wide front
/// ([`Self::indexer_front_rows`], `DSV41_INDEXER_MROWS`) and the per-row select
/// ([`Self::indexer_rows_one`]) — and each materialises the norm itself when the
/// fused launcher declines, so the fallbacks below still read normalised values.
///
/// `DSV41_INDEXER_QR_RAW=0` reverts to R2's compensating `norm_rows` (the same
/// code path R2 shipped); with `DSV41_ATTN_LIN_FUSE` off nothing here is
/// reachable at all.
fn indexer_qr_raw() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_INDEXER_QR_RAW").map(|v| v != "0").unwrap_or(true))
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

/// P1v (DSV41_VERIFY_OROPE, default ON): the m-row verify path takes the SAME
/// fused sparse-attention launch the EAGER path takes ([`Self::sparse_attn_orope`]:
/// the o-rope and the fp8 emission of the roped row folded into the
/// sparse-attention kernel) instead of `sparse_attn` + `apply_rope` + `quant`.
///
/// WHY THIS IS A CORRECTNESS FIX, NOT AN OPTIMISATION: the two forms are NOT
/// bit-identical, contrary to the fusion's "verbatim" claim — the `DSV41_DIFF_EAGER`
/// probe mismatches at 18 positions with the EAGER-side fusion on and at ONE with
/// `DSV41_SPARSE_OROPE=0`. So the fused launch IS the numerical divergence between
/// the spec path and the plain one, and taking the same kernel on both sides is
/// the alignment. It is also the cheap direction: the fusion stays on the EAGER
/// side, and the verify path LOSES its per-row `apply_rope` + `quant_fp8` pair.
///
/// "0" reverts the verify path to `sparse_attn` + `apply_rope` + `quant_fp8`
/// (the A/B arm). A decline (an .so without `dsv41_sparse_attn_orope`, or a
/// shape/env the launcher cannot mirror) falls back per row on its own — the same
/// decline the EAGER path would take under the same env, so the two paths stay
/// aligned either way.
fn verify_orope() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_VERIFY_OROPE").map(|v| v != "0").unwrap_or(true))
}

/// How many rows an m-row sparse-attention launch may carry — the Rust mirror of
/// `kAttnMaxBM` (`kernels/cuda/dsv41_kernels.cu`), which sizes the split arm's
/// `g_attn_part[kAttnMaxBM][...]` scratch (the launcher re-checks it).
const ATT_MROWS_MAX_BM: usize = 8;

/// W2-MROWS (`DSV41_ATTN_MROWS=1`, **DEFAULT OFF**): `attention_rows` folds the
/// block's `m` sparse-attention calls into ONE m-row launch (`b = 1, m = m`,
/// grid `(b*m, h)`) carrying a **per-row compressor-length snapshot**
/// (`clen_rows[m]`) and the `idxs_r` row pitch (`ist`, not `topk`).
///
/// **Why the snapshot is the prerequisite.** The sparse-attention kernels bound
/// themselves by the LIVE device counter `*clen` and derive `n`/`topk` from it
/// (dsv41_kernels.cu:947-949). In the verify interleave that counter ADVANCES
/// with every row's commit, so a block-wide launch would hand every row the
/// block-final value — row 0's compressed half would include the latents rows
/// 1..m-1 committed AFTER it (its own future) while losing the oldest group it
/// should still have seen. That is exactly audit defect #1 (see the comment at
/// the head of the row loop), which is what `clen_rows[r]` removes: row `r`
/// binds to its own count, the value its per-row launch read.
///
/// **Two hard preconditions, both checked at the call site** (this gate alone is
/// not enough, and getting them wrong is silent corruption):
///
/// 1. **Causal ring view.** The block's own KV rows are appended to the ring
///    BEFORE the block-wide launch, which is only harmless while the ring has
///    not turned over: for `pos + r < window` the slot index IS the position, the
///    `idx > start_pos` filter in `window_idxs_kernel` still nulls the not-yet-
///    written slots (the block's own FUTURE rows), and every slot a row reads
///    holds the single position it was written with. Once `pos + r >= window`
///    that filter never fires, so rows `0..m-2` would read the block's future
///    rows as history while losing the oldest `m-1-r` entries — audit defect #2
///    ("running the read side BLOCK-WIDE"), the exact failure the
///    `verify_ring_win` fusion had to be reverted for. The `b*m` launch is
///    therefore taken only while `pos + m - 1 < window`; the long-context case
///    needs the ring/window halves fused into the row-ascending kernel first
///    (design §3.4(b)), which is NOT part of W2.
/// 2. **Row pitch = `h*d`.** The kernels index the query/output rows as
///    `(row*h + hh)*d`. The verify call's rows live in the full `[m, nh*hd]`
///    buffers, i.e. at pitch `nh*hd`, which equals `h*d` only at `world == 1`
///    (`h = nlh = nh/world`). Batching at `world > 1` therefore needs a
///    `row_pitch` tail parameter on the five attention kernels AND the merge
///    kernel (the split arm is the default) — a follow-up ABI addition, not a
///    silent assumption. Until it exists the arm is taken at `world == 1` only,
///    which is where the parity harness runs it.
///
/// With the gate OFF (or either precondition unmet) nothing changes: the per-row
/// loop runs exactly as before, and the per-row call passes `nullptr/0` for the
/// new parameters, so the emitted bytes are the pre-W2 ones byte-for-byte.
fn attn_mrows() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("DSV41_ATTN_MROWS")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

/// One-time note when [`attn_mrows`] is ON but a precondition declines the arm —
/// the bring-up failure mode this project keeps re-hitting is "the gate is ON and
/// nothing changed", so the reason is stated once per process instead of being
/// left to inference. `OnceLock` keeps it off the hot path.
fn attn_mrows_decline(reason: &'static str) {
    static F: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    F.get_or_init(|| {
        eprintln!("[dsv41] DSV41_ATTN_MROWS=1 declined: {reason}; per-row sparse attention")
    });
}

/// B2 (DSV41_RING_WIN_FUSE, default ON): one launch does the window ring append
/// and the window indices (two adjacent, mutually independent one-block
/// kernels), saving 40 launches/step. "0" reverts; an .so without
/// `dsv41_ring_win_fuse` falls back on its own.
fn ring_win_fuse() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_RING_WIN_FUSE").map(|v| v != "0").unwrap_or(true))
}

/// R3 (DSV41_RING_WIN_FUSE, **DEFAULT OFF on the verify path** — A/B first): the
/// verify's per-row interleave calls EAGER's [`Self::ring_win_fuse`] with
/// `pos_rows + r`, i.e. the `m == 1` form of the very launch `attention()` issues
/// once per step, so the `ring_append` + `window_idxs` pair (2 launches/row, 12
/// per layer at `m = 6`) becomes ONE launch per row (6 per layer).
///
/// WHY THIS IS EXACT, not an approximation: the fused kernel's ring half is
/// `ring_append_kernel` term for term (`slot = *pos_ctr % window`, one thread per
/// `hd` element) and its indices half is `window_idxs_kernel` term for term,
/// including the `start_pos == 0` special case. The two halves write disjoint
/// buffers (`ring` vs `idxs[0, window)`) and read nothing but `*pos_ctr`, so with
/// `pos_ctr = pos_rows + r` the launch is byte-identical to the pair it replaces
/// AT THAT ROW. What made the earlier BLOCK-wide fusion (`verify_ring_win`) wrong
/// was appending the whole block up front (audit defect #2, a slot-vs-position
/// mixup); one row at a time keeps the ring invariant per row, which is exactly
/// why this variant is safe and that one was reverted.
///
/// The env var is the EAGER-side gate's, but this path reads it as an explicit
/// opt-in so the wiring can be A/B'd: unset keeps the pair here (EAGER still
/// fuses), `=1` turns it on here, `=0` turns both off. A `.so` predating
/// `dsv41_ring_win_fuse` declines inside [`Self::ring_win_fuse`] (it returns
/// `Ok(false)`) and the pair runs, so the fallback needs no symbol check here.
fn verify_ring_win_fuse() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_RING_WIN_FUSE").map(|v| v == "1").unwrap_or(false))
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

// ==================== verify-row-0 / eager parity probe =======================
//
// `DSV41_VROW0_PROBE=1`. The measured symptom it exists for: over a 154-step
// trace `drafts[0] == next` held 60% of the time (the draft is good) while
// `verify_out[0] == next` held only 16% — i.e. the verify block's row 0, the
// forward that consumes `drafts[0]` at `pos + 1`, returns an argmax that repeats
// its own input token far more often than the eager single-row path does at the
// same position. A row that re-emits its input is what the repetition rate of
// the emitted text looks like from the inside.
//
// The verify's row 0 is a RE-SCHEDULING of a plain single-row step (one m-row
// forward instead of m one-row forwards), so `verify_out[0]` must equal the
// eager argmax at `pos + 1` fed the same token. This probe puts those two
// numbers, and the two logits rows they came from, side by side, one JSONL line
// per step:
//
// ```text
//   {"seq":..,"mode":"shadow","pos":..,"token":..,"next":..,"d0":..,
//    "verify_argmax":..,"verify_host_argmax":..,"verify_top":[[idx,val]..],
//    "eager_argmax":..,"eager_host_argmax":..,"eager_top":[[idx,val]..],
//    "same":true|false,"verify_gap":val,"eager_gap":val,...}
// ```
//
// `verify_argmax == next` in that record IS the 16% metric (the verify's row 0
// emitting the token the anchor already emitted — a repetition, token for token),
// and `eager_argmax == next` is the same quantity for the plain path, so the two
// are directly comparable inside one line.
//
// Reading it: when `same` is false, the top-2 spread of BOTH rows says which kind
// of disagreement it is. `verify_gap` (top1 - top2 of the verify row) below the
// bf16/quantisation resolution means the two paths merely landed on different
// sides of a near-tie — a boundary flip that a confidence gate can absorb. A
// LARGE gap with different winners is a structural break (a kernel/geometry
// difference between the m-row and the single-row scheduling), never noise.
//
// Two hooks, because the two paths can afford different things:
//
// * [`DevChain::dspark_shadow_step`] — the FULL probe. Its rollback leaves the
//   chain exactly where the real step did (`pos_ctr == pos + 1`, the ring holding
//   only `token`'s row), which is the one state a plain `step_dev(drafts[0],
//   pos + 1)` can be compared FROM. That extra forward is a real step, so the
//   probe brackets it with its own snapshot/rollback pair — plus `pos_ctr` and
//   `s.ids`, which the verify's snapshot does not cover and the next step's
//   embedding reads.
// * [`DevChain::dspark_spec_step`] — the VERIFY side only (`"eager":null`): the
//   spec path has committed its block by the time a record could be written, so
//   there is no state left to run an eager forward against, and a commit is not
//   something a probe may undo. The record still carries the verify row's
//   logits/argmax, which is what makes the two modes diffable against each other.
//
// ⚠️ Arm it on EVERY rank: the eager forward runs the `DSV41_HEAD_SLICE` argmax
// collective, so a probe armed on one rank only would leave the others waiting.
// The gate is read once per process ([`vrow0_probe`]), so "every rank" means the
// env var, not a per-step decision.
//
// Cost, paid only while the gate is on: two vocabulary-row D2Hs per probed step
// (517 KB each at `vocab = 129280`) plus one extra single-row forward.

/// `DSV41_VROW0_PROBE=1` arms the probe; `0` or unset leaves it off. Cached in a
/// `OnceLock` for the reason every other gate here is: it is read on every
/// speculative step, and a bare `getenv` on that path is the slip this project
/// has been bitten by before.
fn vrow0_probe() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_VROW0_PROBE").map(|v| v != "0").unwrap_or(false))
}

/// `DSV41_SWALLOW_STEP=1` arms the "swallowed main-chain step": from the SECOND
/// spec round on, [`DevChain::dspark_spec_step`] drops the standalone
/// `step_dev` (6.15 ms) and lets the verify's 6-row `[anchor, d1..d5]` block
/// carry the anchor's forward instead — the block pays one extra row (~1.6 ms)
/// and saves the whole step. Default OFF so the two paths can be A/B'd in one
/// session (the switch is read once per process, like every gate here).
///
/// The FIRST round of a request always runs the legacy path: it is what
/// supplies the very first tap (see [`DevChain::spec_primed`]).
fn swallow_step() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_SWALLOW_STEP").map(|v| v != "0").unwrap_or(false))
}

/// How many verify BLOCKS a `DSV41_SWALLOW_STEP` request runs on the DIRECT
/// launches before the verify graph (`DSV41_VERIFY_GRAPH`) is allowed to engage
/// (ar5-hang's method C).
///
/// # Why the transition window has to be kept off the graph
///
/// With `SWALLOW` on, one request emits a `m = 5` verify on round 1 (the legacy
/// bootstrap) and `m = 6` on every later one, so the shape pool's engagement
/// order is forced: DRY the `m = 6` slot → CAPTURE it → REPLAY from then on. That
/// is the exact window in which the four arms of [`DevChain::step_rows_sync`]
/// coexist ACROSS RANKS — the round a rank first sees the new shape is the round
/// another rank may be recording it — and the arms' `host_barrier` counts differ
/// by construction (see `step_rows_sync`'s arm table). `SpinBarrier` counts
/// ARRIVALS per generation, so a rank that takes an arm with one rendezvous
/// fewer silently closes an epoch that a peer never entered and every later
/// barrier misphases by one — the hang `ar5-hang` traced, whose signature is a
/// stall that appears only in the first rounds of a request.
///
/// The window is also where the gate's per-RANK inputs (the `compress_len`
/// mirror behind [`DevChain::compress_branch_steady`], the shape pool, the
/// failure latch) are least likely to agree: they settle only once the swallowed
/// block has committed a few groups. Holding the graph back for the first
/// [`SWALLOW_GRAPH_WARMUP_BLOCKS`] blocks lets both the kernel warm-up and that
/// settling happen on the direct launches, where NO arm exists to diverge into.
///
/// `3` = the three blocks of the window: round 1's `m = 5` bootstrap and the two
/// `m = 6` blocks that follow (the deliberate DRY and CAPTURE of the new shape,
/// which are real executions anyway — the direct path pays exactly that cost).
const SWALLOW_GRAPH_WARMUP_BLOCKS: usize = 3;

/// `DSV41_LAZY_VERIFY=1` arms the ROW-BY-ROW lazy verify and the lazy ⇄ batched
/// route ([`DevChain::dspark_spec_lazy`], [`DevChain::lazy_route_decide`]).
/// Default OFF, so every existing path stays bit-identical.
///
/// # What it changes
///
/// The swallowed arm runs the whole `[anchor, d1..d5]` block as ONE 6-row
/// forward (~37 ms) and then rolls the rejected tail back. The lazy arm runs
/// rows `0, 1, 2, ...` one at a time (`m = 1`, ~6.15 ms each) and STOPS at the
/// first draft that misses, so a round costs `k_emit` single-row steps instead
/// of one 6-row block. Because every row it ran is inside the commit's keep
/// range, the lazy arm needs neither a rollback nor a compressor replay.
///
/// # lazy IMPLIES swallow
///
/// The block layout is the swallowed one (`[anchor, d1..d5]` at `pos .. pos+5`)
/// — the lazy arm's row 0 IS the swallowed main-chain step — so this gate also
/// routes the second-and-later rounds past the legacy arm, exactly like
/// [`swallow_step`]. The two differ only in the branch they take inside that
/// block: [`DevChain::lazy_route_decide`] picks between them per round.
///
/// The FIRST round of a request always runs the legacy path: it is what supplies
/// the very first tap (see [`DevChain::spec_primed`]).
fn lazy_verify() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_LAZY_VERIFY").map(|v| v != "0").unwrap_or(false))
}

/// `DSV41_LAZY_SDR=1`: the lazy arm's PER-ROW SYNC COALESCING — the implementable
/// part of the "L2 per-row sync" item in
/// `docs/agent/lazy-verify-optimization-path.md` §L2. Default OFF, so the lazy
/// path keeps its current every-row sync sequence bit for bit.
///
/// # What it removes
///
/// A lazy row pays ~4 host↔device round trips; this gate takes two of them out
/// without changing a single byte of the block's maths:
///
/// * **the tap carry.** [`DevChain::lazy_tap_commit`]'s `DSPARK_TAP_SLOTS`
///   device-to-device copies become ONE strided copy (`memcpy_d2d_2d`): the
///   staging buffer's slots are `dim` floats apart and so are the block's rows,
///   so the `slots × dim` block is one `cudaMemcpy2DAsync` instead of three
///   launches.
/// * **the position counter.** The per-row `set_pos_ctr(pos + i)` — a 4-byte
///   BLOCKING H2D — is dropped; the counter is parked once per round at the
///   block's row-0 position instead.
///
/// # Why dropping the per-row counter write is sound
///
/// Nothing in the m-ROW path reads `s.pos_ctr`:
///
/// * every kernel `step_rows_inner` reaches takes its position from
///   `s.pos_rows[r]` (`attention_rows`' RoPE/`sparse_attn`/`indexer`,
///   `compress_row`'s pool/commit, `engram_hash_step`'s per-row twin);
/// * the block's argmax is passed a NULL counter explicitly, because the accept
///   logic moves the counter once, for the committed prefix;
/// * `step_rows` itself does not read it back — the lazy arm passes
///   `pos_base_hint`, and [`DevChain::lazy_run_row`]'s whole point is that the
///   caller already knows the value.
///
/// The only reads of `s.pos_ctr` in this file belong to the SINGLE-ROW arms
/// (`norm_fuse`/`orope_q`'s fused RoPE, the single-row `engram_hash_step`, the
/// single-row `argmax`) and to the utilities that bracket a step
/// ([`DevChain::set_pos_ctr`]'s callers, the snapshot/shadow-step save pairs).
/// The round's final value is written by [`DevChain::dspark_commit_lazy`] (and
/// `pos` again on the error path), so the counter the NEXT round reads is the
/// same either way.
///
/// # What it deliberately does NOT do: pair two rows
///
/// The design document's headline — a speculative TWO-ROW batch, one D2H for
/// both argmaxes (its "SDR") — is REJECTED here; the reasoning is recorded on
/// [`DevChain::dspark_spec_lazy`]'s cost-shape note. In short: the drafts make
/// the pair's INPUTS available, but the second row is speculative, ~45% of
/// drafts miss, and a row costs ~6 ms of wall clock while the batched D2H saves
/// ~0.5 ms — the trade is a large net loss, and the discarded row has no exact
/// rollback on an arm that deliberately runs with `spec_capture` clear.
fn lazy_sdr() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_LAZY_SDR").map(|v| v != "0").unwrap_or(false))
}

/// `DSV41_LAZY_THRESHOLD`: the mean-k threshold of the lazy ⇄ batched route.
/// `auto` (or unset) derives it from the two calibrated step costs
/// ([`lazy_tau`]); an explicit float overrides it (`0` ⇒ always batched, a
/// value above `DSPARK_DRAFTS` ⇒ always lazy). Read once per process, like every
/// other gate here.
fn lazy_threshold_override() -> Option<f32> {
    static F: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("DSV41_LAZY_THRESHOLD")
            .ok()
            .filter(|s| s != "auto")
            .and_then(|s| s.parse::<f32>().ok())
    })
}

/// The Schmitt hysteresis of the route, in mean-k units (`lazy-batched-gate.md`
/// §2.3): the arm is STICKY, and only flips when the window's mean-k crosses the
/// threshold by this much. Without it a window sitting exactly on `τ` would
/// flip the arm every round and churn the verify graph / AR footprint.
const LAZY_HYST: f32 = 0.25;

/// The batched arm's verify cost `B` (ms), the numerator of the route rule
/// `lazy <==> (1 + mean_k) * c < B`. A PROCESS-level quantity: the row
/// parallelisation the .so does or does not get is a property of the kernel
/// image, not of a request, and every rank is a thread of the same process — so
/// one shared cell keeps all ranks on the same `τ` (`lazy-batched-gate.md` §2.2).
///
/// It starts at the measured batched verify of the status quo and is refreshed
/// by every round that RUNS the batched arm. `AtomicU32` holding an `f32`'s bits
/// because there is no `AtomicF32`; relaxed ordering is enough — the value is a
/// cost estimate, not a synchronisation token.
static LAZY_B_MS: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0x4214_0000); // 37.0f32

/// The frozen default of [`LAZY_B_MS`]: the batched verify measured at HEAD
/// (`lazy-batched-gate.md` §0-7: `B = 37 ms` ⇒ `τ ≈ 5.0`, i.e. lazy always
/// wins today, because `mrows` is not dispatching).
fn lazy_b_ms() -> f32 {
    f32::from_bits(LAZY_B_MS.load(std::sync::atomic::Ordering::Relaxed))
}

/// Refresh [`LAZY_B_MS`] from a round that actually ran the batched arm.
fn lazy_b_ms_note(ms: f32) {
    if ms.is_finite() && ms > 0.0 {
        LAZY_B_MS.store(ms.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }
}

/// The EAGER single-row step cost `c` (ms) — the denominator of the route rule.
/// A constant rather than a measurement: `lazy-batched-gate.md` §2.2 pins it to
/// the STATUS baseline, and it is the same number the row loop's own cost model
/// uses (one row = one `step_rows(m = 1)`).
const LAZY_C_MS: f32 = 6.15;

/// `τ = B / c − 1`: the mean-k at which a lazy round (`(1 + mean_k)` rows) and a
/// batched round (one `m = 6` block) cost the same. `DSV41_LAZY_THRESHOLD`
/// overrides it outright.
fn lazy_tau() -> f32 {
    if let Some(t) = lazy_threshold_override() {
        return t;
    }
    if LAZY_C_MS > 0.0 {
        lazy_b_ms() / LAZY_C_MS - 1.0
    } else {
        0.0
    }
}

/// Which verify arm the sticky route currently sits on. `None` before the first
/// decision of a request (see [`DevChain::lazy_route_decide`]'s cold start).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    Lazy,
    Batched,
}

/// The route's decision window: the last `N` rounds' `k_acc` as a ring of `u8`,
/// plus the running sum, so `mean_k` is an INTEGER average and every rank
/// computes the same value from the same window.
///
/// # Why integer, and why per-request
///
/// The decision must be bit-identical on every rank: AR v5 has no host-side
/// rendezvous, so ranks that disagreed on the arm would issue different numbers
/// of collectives in the same epoch and wedge. `k_acc` is the cross-rank
/// reduction's own output (`dsv41_argmax_sliced_rows`), so a window over it is
/// identical on every rank BY CONSTRUCTION — no float ever enters the judgement
/// (`lazy-batched-gate.md` §2.2). The window is per-request state (it resets with
/// `spec_primed`), while `B`/`c` are process-level, which is what makes `τ` a
/// rank-invariant too.
#[derive(Debug, Clone)]
struct LazyHist {
    buf: [u8; LAZY_HIST],
    head: usize,
    len: usize,
    sum: u32,
}

/// How many rounds the window averages over (`lazy-batched-gate.md` §2.2).
const LAZY_HIST: usize = 32;

impl Default for LazyHist {
    fn default() -> Self {
        Self { buf: [0; LAZY_HIST], head: 0, len: 0, sum: 0 }
    }
}

impl LazyHist {
    fn push(&mut self, k_acc: usize) {
        // `k_acc` is `k_emit - 1 <= DSPARK_DRAFTS`, so the `u8` ring never
        // saturates in practice; the clamp is here so a caller that ever hands
        // over something larger degrades to a conservative window instead of
        // wrapping the mean around.
        let k = k_acc.min(u8::MAX as usize) as u8;
        if self.len == self.buf.len() {
            self.sum -= self.buf[self.head] as u32;
        } else {
            self.len += 1;
        }
        self.buf[self.head] = k;
        self.sum += k as u32;
        self.head = (self.head + 1) % self.buf.len();
    }

    fn mean_k(&self) -> f32 {
        if self.len == 0 {
            0.0
        } else {
            self.sum as f32 / self.len as f32
        }
    }
}

/// `DSV41_SEED_ALIGN=1` arms the SEED↔TAP alignment (route A of the draft-quality
/// arbitration): [`DevChain::dspark_spec_step`] moves the draft block onto the
/// anchor the tap actually belongs to, and the verify block becomes the 6-row
/// `[next, d1..d5]` GLM layout.
///
/// # The defect it fixes
///
/// `step_dev(token, pos)` forwards `token` at position `pos` and the tap hook
/// records that forward's target-layer hidden — so the tap's position IS `pos`
/// (`layer()`'s hook runs inside the step, `dspark_dev.rs`'s `import_tap` only
/// copies it afterwards). The draft then seeded its window ring at
/// `seed_window(s, pos - 1)` while feeding the block `[embed(token), noise×4]`:
/// the ring row `{content: h(token@pos), RoPE: pos - 1}` is a one-token
/// mismatch, and the slot `pos % win` (the position the tap actually describes)
/// was written by NOBODY — every step's `p_i` was missing from the set of
/// written positions `{p_i - 1} ∪ [p_i + 1, p_i + k_acc]`. The window's contents
/// therefore sat one token BEHIND their RoPE phases: the tight predecessor
/// (distance 1, the only draft candidate that matters) appeared as distance 2,
/// while "distance 1" was the anchor itself — a duplicate of the block's row 0.
///
/// # The fix (this gate ON)
///
/// The draft is called as `draft_forward(next, pos + 1)`, where `next` is the
/// step's own argmax (the token at `pos + 1`). The seed slot is then
/// `(pos + 1) - 1 = pos` — the tap's real position, and `seed_window`'s `pos - 1`
/// argument needs no change at all (it is correct for an anchor at `pos + 1`).
/// The verify becomes the 6-row `[next, d1..d5] @ pos + 1 ..= pos + 6`, whose row
/// 0 IS the anchor's forward, so the accept chain is INDEX-ALIGNED and the shared
/// [`ferrite_types::spec_accept`] runs with `anchor_is_in_block = true` — the
/// layout GLM's MTP already uses (see [`Self::dspark_spec_swallowed`]).
///
/// The gate also fixes the ring WRAP-AROUND (`DsparkDev::win_rows`): once
/// `pos >= win` the window must rotate (the ring's slot `pos % win` is the
/// anchor's and must not be a candidate) and shrinks to `win - 1` live rows;
/// without it an anchor double-KV enters the candidate set, which the official
/// mask forbids.
///
/// Default OFF, so the two paths can be A/B'd in one session (the switch is read
/// once per process, like every gate here). This gate does NOT touch the tap
/// COLLECTION point (`layer()`'s end vs `ref_inference/model.py:1264-1266`'s
/// input) — that arbitration is a separate experiment.
pub(crate) fn seed_align() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_SEED_ALIGN").map(|v| v != "0").unwrap_or(false))
}

/// How many logits each row reports: `DSV41_VROW0_TOPK`, default 5.
fn vrow0_topk_n() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("DSV41_VROW0_TOPK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(5)
            .max(1)
    })
}

/// Where the records go: `DSV41_VROW0_PATH`, else `/tmp/vrow0.jsonl`.
fn vrow0_path() -> &'static str {
    static P: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        std::env::var("DSV41_VROW0_PATH").unwrap_or_else(|_| "/tmp/vrow0.jsonl".to_string())
    })
}

/// The top `k` entries of one logits row as `(global index, value)`, descending,
/// ties broken by the LOWER index — the rule both argmax kernels implement
/// (the sliced one packs `(value, -index)`).
///
/// `base` is the row's first entry's index in the FULL vocabulary: the verify's
/// `logits_r` row is always the whole vocabulary, while the eager step's row is
/// one rank's slice under `DSV41_HEAD_SLICE` (see [`DevChain::vrow0_eager`]).
///
/// `f32::total_cmp` rather than `partial_cmp`: a NaN is then ORDERED instead of
/// silently dropping out of every comparison, and it sorts to the top — a `NaN`
/// in a record is itself a finding, not a lost comparison.
fn vrow0_topk(lg: &[f32], base: u32, k: usize) -> Vec<(u32, f32)> {
    let k = k.min(lg.len());
    if k == 0 {
        return Vec::new();
    }
    let mut idx: Vec<u32> = (0..lg.len() as u32).collect();
    let cmp = |a: &u32, b: &u32| lg[*b as usize].total_cmp(&lg[*a as usize]).then(a.cmp(b));
    // O(n) partial select, then sort just the k winners: a full sort of 129280
    // entries per row per step would be the probe's dominant cost.
    idx.select_nth_unstable_by(k - 1, cmp);
    idx.truncate(k);
    idx.sort_by(cmp);
    idx.into_iter().map(|i| (base + i, lg[i as usize])).collect()
}

/// One probe record. Built by [`DevChain::vrow0_step`] and written by
/// [`vrow0_write`].
struct Vrow0Rec<'a> {
    /// `"shadow"` or `"spec"` — which orchestration produced the record.
    mode: &'a str,
    /// The anchor token's position (the verify's row 0 sits at `pos + 1`).
    pos: usize,
    /// The anchor token the real step consumed.
    token: u32,
    /// The real (single-row) step's argmax — the token at `pos + 1`, i.e. what
    /// `drafts[0]` was predicting. `verify_argmax == next` is the repetition
    /// fingerprint this probe was built for (measured 16% against the eager
    /// path's baseline), so it is carried in the record rather than reconstructed.
    next: u32,
    /// `drafts[0]` — the token the verify's row 0 consumed.
    d0: u32,
    /// `step_rows`' argmax for the block's row 0 (the kernel's answer).
    verify_argmax: u32,
    /// The argmax recomputed on the HOST from the same row's logits (D2H'd
    /// below). A `verify_host_argmax != verify_argmax` is an argmax-kernel
    /// disagreement, not a logits difference — the split the probe exists to
    /// make.
    verify_host_argmax: u32,
    verify_top: &'a [(u32, f32)],
    /// `(argmax, host argmax, top, row width, index base)` from the eager
    /// single-row forward at `pos + 1`. `None` on the spec path (no eager forward
    /// is legal there — see the module-level note).
    eager: Option<&'a Vrow0Eager>,
    rank: usize,
    world: usize,
}

/// The probe's eager half: what one plain single-row step produced at `pos + 1`.
struct Vrow0Eager {
    argmax: u32,
    host_argmax: u32,
    top: Vec<(u32, f32)>,
    /// The row width read: the full vocabulary, or one rank's slice.
    n: usize,
    /// The row's first entry's index in the FULL vocabulary.
    base: u32,
    /// Which of the two the row was (`step_body`'s own condition, repeated).
    sliced: bool,
}

/// Append one record. The caller LOGS a failure rather than propagating it — a
/// full disk must not answer an error for a step whose block is committed.
///
/// Floats go through Rust's shortest-roundtrip `Display`, not `serde_json`:
/// `NaN`/`inf` must stay visible rather than becoming `null`, which is the one
/// thing a numeric near-tie diff is looking for.
fn vrow0_write(rec: &Vrow0Rec<'_>) -> Result<()> {
    use std::fmt::Write as _;
    use std::io::Write as _;

    let ts_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let mut line = String::with_capacity(512);
    let _ = write!(
        line,
        "{{\"seq\":{},\"pid\":{},\"ts_ms\":{},\"mode\":\"{}\",\"pos\":{},\"token\":{},\"next\":{},\
         \"d0\":{},\
         \"verify_argmax\":{},\"verify_host_argmax\":{},\"verify_top\":[",
        VROW0_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        std::process::id(),
        ts_ms,
        rec.mode,
        rec.pos,
        rec.token,
        rec.next,
        rec.d0,
        rec.verify_argmax,
        rec.verify_host_argmax,
    );
    vrow0_push_top(&mut line, rec.verify_top);
    line.push(']');

    let (same, e_gap) = match rec.eager {
        Some(e) => {
            let _ = write!(
                line,
                ",\"eager_argmax\":{},\"eager_host_argmax\":{},\"eager_top\":[",
                e.argmax, e.host_argmax
            );
            vrow0_push_top(&mut line, &e.top);
            let _ = write!(
                line,
                "],\"eager_n\":{},\"eager_base\":{},\"eager_sliced\":{}",
                e.n, e.base, e.sliced
            );
            (Some(rec.verify_argmax == e.argmax), vrow0_gap(&e.top))
        }
        None => {
            line.push_str(",\"eager_argmax\":null,\"eager_top\":null");
            (None, f32::NAN)
        }
    };
    let v_gap = vrow0_gap(rec.verify_top);
    let _ = write!(
        line,
        ",\"verify_gap\":{v_gap},\"eager_gap\":{e_gap},\"rank\":{},\"world\":{}",
        rec.rank, rec.world
    );
    match same {
        Some(s) => line.push_str(if s { ",\"same\":true}" } else { ",\"same\":false}" }),
        None => line.push_str(",\"same\":null}"),
    }
    line.push('\n');

    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(vrow0_path())
        .map_err(|e| FerriteError::Config(format!("vrow0 probe: open {}: {e}", vrow0_path())))?;
    f.write_all(line.as_bytes())
        .map_err(|e| FerriteError::Config(format!("vrow0 probe: write {}: {e}", vrow0_path())))?;
    Ok(())
}

/// Per-process record counter, for the reason [`dump_dev`] keeps one: the file
/// is append-only and survives runs, so `pos` alone cannot separate two runs.
static VROW0_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn vrow0_push_top(line: &mut String, top: &[(u32, f32)]) {
    use std::fmt::Write as _;
    for (i, (idx, v)) in top.iter().enumerate() {
        if i > 0 {
            line.push(',');
        }
        let _ = write!(line, "[{idx},{v}]");
    }
}

/// `top1 - top2`, or `NaN` when a row has fewer than two entries — the number the
/// analysis script buckets a mismatch by.
fn vrow0_gap(top: &[(u32, f32)]) -> f32 {
    match (top.first(), top.get(1)) {
        (Some(a), Some(b)) => a.1 - b.1,
        _ => f32::NAN,
    }
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
    /// DSV41_SPEC only: while this is set, [`Self::compress_proj_rows`] also saves the
    /// verify's per-layer `kvp`/`scp` rows (see `Scratch::spec_snap_kvp`), because
    /// [`Self::dspark_spec_step`]'s commit has to REPLAY the accepted prefix
    /// through the compressor and the shared m-row scratch does not survive the
    /// layer walk. Armed only around the spec step's `step_rows` call, so the
    /// shadow path pays nothing.
    spec_capture: bool,
    /// Set ONLY while [`Self::capture_verify`] is RECORDING the verify's CUDA
    /// graph (`DSV41_VERIFY_GRAPH=1`), and cleared before the capture ends.
    ///
    /// It exists for exactly one reason: the graph is a recorded LAUNCH SEQUENCE,
    /// so nothing inside the recording may branch on a HOST value that changes
    /// from verify to verify. `committed` (the compressor's host mirror of the
    /// completion rule, `(pos_base + r + 1) % ratio == 0`) IS such a value — it
    /// flips with `pos_base mod ratio`, i.e. every single verify — and it used to
    /// decide whether `indexer_rows_one` emitted the `publish_index_key` launch
    /// at all. The capture, taken at ONE parity, therefore baked a launch
    /// sequence that was missing the publish for every row of the OPPOSITE
    /// parity, and a replay at that parity selected against `index_k` slots no
    /// launch had written.
    ///
    /// The direct path (and the DRY/fallback runs, which are the same direct
    /// launches) never sets this flag, so its behaviour is bit-for-bit what it
    /// was; only the recording is made parity-independent. See
    /// [`Self::indexer_rows_one`] for why an unconditional launch is the correct
    /// recording even though the direct path may skip it.
    verify_recording: bool,
    /// `DSV41_SWALLOW_STEP` only: the spec step's bootstrap flag. The FIRST
    /// round of a request runs the legacy path (the standalone `step_dev`
    /// supplies the anchor's forward AND the tap); every later round SWALLOWS
    /// that step and lets the verify's 6-row `[anchor, d1..d5]` block carry the
    /// anchor's forward instead (see [`Self::dspark_spec_step`]).
    ///
    /// Cleared by [`Self::reset`], so a new request re-bootstraps. Deliberately
    /// NOT part of any `KvSnapshot`: it describes THIS chain's step sequence,
    /// and the dspark spec path is explicitly mutually exclusive with the KV
    /// prefix cache (see `serve.rs`'s gate).
    spec_primed: bool,
    /// `DSV41_LAZY_VERIFY` only: which verify arm the sticky route is on
    /// ([`Self::lazy_route_decide`]). `None` until this request's first routed
    /// round — the cold start decides on the bare threshold, and from then on the
    /// arm only flips when the window crosses `τ ± LAZY_HYST` (Schmitt).
    ///
    /// Reset with `spec_primed`: both describe THIS request's spec sequence.
    lazy_mode: Option<Arm>,
    /// `DSV41_LAZY_VERIFY` only: the route's integer mean-k window. Fed from the
    /// report at the arms' COMMON exit (`dspark_spec_step`), so it averages over
    /// whatever the live arm produced — that is what lets the route see accept
    /// drift and switch.
    lazy_hist: LazyHist,
    /// `DSV41_LAZY_VERIFY` only: the "deferred tap" mode of the verify's tap hook
    /// (`layer_rows`).
    ///
    /// The tap is written INSIDE the verify's forward, once per target layer, so
    /// the row it lands on is a host-side launch argument — and the m=1 lazy block
    /// is always row 0 of its slot. With this set, `layer_rows` writes the row to
    /// the SINGLE-row tap buffer (`s.dspark_tap`) instead, and the lazy arm copies
    /// it to row `i` of `s.dspark_tap_r` after each row's forward. Two reasons the
    /// staging copy beats baking the row into the graph: the destination pointer
    /// stays IDENTICAL across rows (so one captured `m = 1` graph serves every row
    /// — the whole point of the lazy arm's graph reuse), and no `k_emit`-many
    /// graphs are needed.
    ///
    /// Cleared in [`Self::reset`]; always `false` on every non-lazy path, so the
    /// hook's behaviour there is bit-identical.
    spec_tap_deferred: bool,
    // ---- the DSpark verify's own CUDA graph (`DSV41_VERIFY_GRAPH=1`) ----
    //
    // `step_rows` is ~7000 launches per verify (40 layers x ~170 nodes), and
    // each streaming launch costs its ~2.9us of submit time on top of the
    // graph's ~0.4us/node dispatch floor. The whole block is ONE graph: the
    // per-step inputs it reads (the token ids, the row positions) are refreshed
    // on the DEVICE buffers OUTSIDE the capture, so no launch argument changes
    // from verify to verify and the replay is exact.
    //
    // A SEPARATE exec from `step_graph` on purpose: the two are captured at
    // different times (the step graph on the first decode step, this one on the
    // second verify) and cover different kernel sets. The precedent for two
    // coexisting execs is `moe_graph` above.
    //
    // The execs live in the SHAPE POOL `verify_graphs` below: ONE SLOT PER ROW
    // COUNT. With `SEED_ALIGN`/`SWALLOW` on a single request emits `m = 5` on
    // its first verify and `m = 6` on every later one, and a single slot could
    // only ever latch the first of those — see `verify_shapes`.
    verify_graphs: [Option<*mut std::ffi::c_void>; VERIFY_GRAPH_SLOTS],
    /// The row count (`m`) each graph slot was captured at, `0` = EMPTY slot.
    ///
    /// The capture bakes the launch geometry, so a verify block of a DIFFERENT
    /// length must not replay a slot's graph. ONE shared `m` field was not
    /// enough: with `SEED_ALIGN`/`SWALLOW` on, one request emits `m = 5` on its
    /// first verify and `m = 6` on every later one (`DSPARK_DRAFTS` = 5 for the
    /// shadow step, +1 for the swallowed anchor). The single field had the first
    /// DRY write `5`, so every later `m = 6` call failed the shape latch and the
    /// request quietly finished on the direct launches — the graph engaged for
    /// exactly one step and never paid for a capture at all. The pool holds the
    /// (at most) TWO shapes a production request can produce; a THIRD shape (the
    /// parity self-test varies `m`) still takes the direct path, as does a
    /// request in which the pool is full.
    verify_shapes: [usize; VERIFY_GRAPH_SLOTS],
    /// Set when a capture attempt FAILED, PER SLOT. The graph is an
    /// optimisation, so a refusal from the driver must not take the verify down
    /// with it: the request finishes on the direct launches, and this latch stops
    /// the same doomed capture from being re-attempted on every step. The slots
    /// latch INDEPENDENTLY — a refusal while capturing one shape does not disable
    /// the other (each shape has its own capture, so each gets its own verdict).
    /// Cleared by `reset`.
    verify_graph_failed: [bool; VERIFY_GRAPH_SLOTS],
    /// False until that SLOT's first `step_rows` call has executed for real: that
    /// call is the DRY run (it warms every kernel, builds the lazy engram state
    /// and sizes the cuBLAS workspaces), and only the SECOND one captures. Per
    /// slot, because each shape's geometry has to be warmed before its own
    /// capture — the second shape DRYs while the first shape's graph already
    /// exists.
    verify_dry_done: [bool; VERIFY_GRAPH_SLOTS],
    /// How many verify BLOCKS this request has run, counted at the single entry
    /// [`Self::step_rows_sync`] gives every verify.
    ///
    /// Per-request (cleared by [`Self::reset`], like the shape pool it guards)
    /// and read by ONE consumer: the `DSV41_SWALLOW_STEP` warm-up window of
    /// [`Self::verify_graph_gate`], which holds the verify graph off the first
    /// [`SWALLOW_GRAPH_WARMUP_BLOCKS`] blocks so that the DRY → CAPTURE → REPLAY
    /// transition cannot overlap the request's first rounds — see the constant's
    /// note for why that window is where the arms' rendezvous counts disagree
    /// across ranks.
    ///
    /// It counts BLOCKS rather than spec rounds on purpose: the lazy arm runs one
    /// block per row, so a block count can only ever be conservative (the gate
    /// simply waits a little longer), while a round count would let the first
    /// three lazy rows of round 1 alone consume the window.
    verify_blocks: usize,
    /// Diagnostics: captures and replays since the last `reset`.
    verify_captures: u32,
    verify_replays: u32,
    /// The verify head's geometry as the LAST [`Self::step_rows_inner`] resolved
    /// it (`None` = the full-vocabulary rows), for the vrow0 probe.
    ///
    /// The probe ([`Self::vrow0_verify`]) must read `logits_r` at the SAME pitch
    /// the head wrote, and re-deriving it at probe time is a hazard even though
    /// `verify_head_geom` no longer depends on the capture state (the v5 epoch
    /// rule): the probe is a separate call that can run after the state the
    /// geometry was resolved under (env gates, peers) has moved, and the pitch is
    /// not in the type, so a mismatch is a SILENT misread of a `seg`-pitch row as
    /// a full-vocabulary one. Recorded per `step_rows_inner` call, which is also
    /// the call the probe follows; the replay path leaves it at the capture's
    /// value, which is what the replayed graph wrote.
    verify_geom: std::cell::Cell<Option<(usize, usize)>>,
}

fn fb(n: usize) -> usize {
    n * 4
}

/// Grouped-router (DSV41_EXPERT_GROUPED) capacity in EXPERTS. The backbone and
/// the DSpark layers each carry their own routed-expert count
/// (`Dsv41Config::moe_config`), and a `Scratch` buffer is allocated once, so the
/// capacity has to cover whichever of the two is larger. ONE definition on
/// purpose: the allocation in `DevChain::new` and the launch in
/// `moe_route_grouped` would otherwise be two copies of the same expression,
/// and a drift between them is a silent out-of-bounds write.
fn grp_n_experts(cfg: &Dsv41Config) -> usize {
    cfg.n_routed_experts
        .max(1)
        .max(cfg.dspark_n_routed_experts)
}

/// Grouped-router capacity in top-k slots per row — the `topk` axis of
/// `grp_xq`/`grp_xsc` and (times `VERIFY_ROWS`) the per-expert row stride
/// `m_cap`. Same one-definition rule as [`grp_n_experts`].
fn grp_topk_max(cfg: &Dsv41Config) -> usize {
    cfg.n_activated_experts
        .max(1)
        .max(cfg.dspark_n_activated_experts)
}

/// The per-expert row stride of `grp_rows`/`grp_slots`, i.e. the `m_cap`
/// argument of `dsv41_route_group`. The worst case is every assignment landing
/// on one expert (`VERIFY_ROWS` rows x `topk` slots); the real router cannot
/// exceed `VERIFY_ROWS` because it consumes each expert it picks, so this is
/// headroom against a malformed (duplicate-id) table rather than a real bound.
fn grp_m_cap(cfg: &Dsv41Config) -> usize {
    VERIFY_ROWS * grp_topk_max(cfg)
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
            // One activation row: `dim/2` packed fp4 bytes, or `dim` e4m3 bytes
            // under DSV41_EXPERT_ACT_E4M3 (1 byte/value). `dim` covers both.
            xq4: dev.alloc(dim.max(8))?,
            xsc4: dev.alloc(fb(dim / 32 + 8))?,
            xq_of_xn_valid: std::cell::Cell::new(false),
            xq_of_qr_valid: std::cell::Cell::new(false),
            qr_raw: std::cell::Cell::new(false),
            qr_raw_r: std::cell::Cell::new(false),
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
            // the per-row twin the real commit's verify fills (see the field's
            // comment): 6 rows x 3 layers x dim f32 = 368 KB at the prod shape
            dspark_tap_r: dev.alloc(fb(DSPARK_TAP_SLOTS * VERIFY_ROWS * dim).max(8))?,
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
            // The multi-row activation staging (see the field comment): one row
            // per verify row, each sized for the LARGEST activation the block
            // quantises -- `dim` for wq_a/wkv, the full attention-output row
            // `nh*hd` for wo_a (its `a_stride` is the this-rank `nlh*hd`, which
            // is <= that). Sized unconditionally: a few hundred KB, and a static
            // allocation graph is worth more than the bytes.
            xq_r: dev.alloc((VERIFY_ROWS * dim.max(nh * hd)).max(8))?,
            xsc_r: dev.alloc(fb(VERIFY_ROWS * dim.max(nh * hd) / 32 + 8))?,
            moe_out_r: dev.alloc(fb(VERIFY_ROWS * dim))?,
            ids_r: dev.alloc(VERIFY_ROWS * 4)?,
            argmax_r: dev.alloc(VERIFY_ROWS * 4)?,
            argmax_packed_r: dev.alloc(VERIFY_ROWS * 8)?,
            // Sized by the FULL vocabulary whatever the head's arm: the sliced
            // arm only uses a `seg = vocab / world` prefix of each row, and a
            // static allocation graph is worth more than the unused 7/8.
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
            // COMPRESSOR-MROWS scratch: allocated unconditionally (a static graph
            // is worth more than the unused bytes when the gate is off).
            clen_rows_r: dev.alloc(4 * cfg.n_layers * VERIFY_ROWS)?,
            latent_rows_r: dev.alloc(fb(VERIFY_ROWS * hd))?,
            scores_r: dev.alloc(fb(VERIFY_ROWS * n_exp))?,
            route_idx_r: dev.alloc(4 * VERIFY_ROWS * topk)?,
            route_w_r: dev.alloc(fb(VERIFY_ROWS * topk))?,
            // ---- grouped (permuted) routing scratch (DSV41_EXPERT_GROUPED) ----
            // `grp_cap` is the per-expert row capacity of the `grp_rows`/
            // `grp_slots` lists. The router consumes each expert it picks
            // (`dsv41_route_topk` writes -INFINITY into the selection score), so
            // an expert can appear at most once per row and the honest bound is
            // `m`; `VERIFY_ROWS * topk` is the WORST CASE of a malformed
            // (duplicate-id) table and costs 55 KB at n_routed = 384. Sized for
            // the worst case so a malformed table truncates nothing — the
            // kernel's `j < m_cap` guard is then unreachable belt-and-braces.
            // Every capacity comes from the `grp_*` helpers above, which the
            // launch site reads back: the two MUST agree or a launch writes past
            // the block.
            grp_counts: dev.alloc(4 * grp_n_experts(cfg))?,
            grp_starts: dev.alloc(4 * (grp_n_experts(cfg) + 1))?,
            grp_rows: dev.alloc(4 * grp_n_experts(cfg) * grp_m_cap(cfg))?,
            grp_slots: dev.alloc(4 * grp_n_experts(cfg) * grp_m_cap(cfg))?,
            grp_perm: dev.alloc(4 * VERIFY_ROWS * grp_topk_max(cfg))?,
            grp_src: dev.alloc(4 * VERIFY_ROWS * grp_topk_max(cfg))?,
            grp_active: dev.alloc(4 * grp_n_experts(cfg))?,
            grp_nactive: dev.alloc(4)?,
            // One gathered row per ASSIGNMENT (m * topk), not per row: an
            // expert's operand block has to be contiguous, and a row's topk
            // slots normally go to topk different experts.
            grp_xq: dev.alloc((VERIFY_ROWS * grp_topk_max(cfg) * dim).max(8))?, // e4m3 bytes
            grp_xsc: dev.alloc(fb(VERIFY_ROWS * grp_topk_max(cfg) * dim / 32 + 8))?,
            // Same capacity as `ex_act_r` below (one row per assignment, `2*inter`
            // wide): the grouped gate/up output is the pre-scatter twin of it.
            grp_out: dev.alloc(fb(VERIFY_ROWS * grp_topk_max(cfg) * 2 * inter.max(dim)))?,
            // `dim` bytes per row covers both activation formats: the packed fp4
            // arm uses `dim/2`, the DSV41_EXPERT_ACT_E4M3 arm exactly `dim`.
            xq4_r: dev.alloc((VERIFY_ROWS * dim).max(8))?,
            xsc4_r: dev.alloc(fb(VERIFY_ROWS * dim / 32 + 8))?,
            // `2 * inter` per slot: `inter` (the full, unsharded width) is >= every
            // rank's padded local slice, so one allocation covers tp=1 and tp=N.
            ex_act_r: dev.alloc(fb(VERIFY_ROWS * topk * 2 * inter.max(dim)))?,
            ex_down_r: dev.alloc(fb(VERIFY_ROWS * topk * dim))?,
            sh_act_r: dev.alloc(fb(VERIFY_ROWS * 2 * inter.max(dim)))?,
            sh_out_r: dev.alloc(fb(VERIFY_ROWS * dim))?,
            sh_aq_r: dev.alloc((VERIFY_ROWS * inter).max(8))?, // fp8 bytes
            sh_aqsc_r: dev.alloc(fb(VERIFY_ROWS * inter / 32 + 8))?,
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
            spec_snap_kvp: dev.alloc(fb(cfg.n_layers * VERIFY_ROWS * hd).max(8))?,
            spec_snap_scp: dev.alloc(fb(cfg.n_layers * VERIFY_ROWS * hd).max(8))?,
            // ---- the KV prefix snapshot's staging area. Allocated unconditionally
            // (10.5 MB) so the allocation graph stays static, exactly like the
            // shadow save's set above; only touched when a snapshot/restore runs.
            kv_snap_ring: dev.alloc(fb(cfg.n_layers * cfg.window_size * hd).max(8))?,
            kv_snap_state: dev.alloc(
                fb(cfg.n_layers
                    * 2
                    * cfg.compress_ratios.iter().copied().max().unwrap_or(1).max(1)
                    * hd)
                .max(8),
            )?,
            kv_snap_latent: dev.alloc(fb(cfg.n_layers * hd).max(8))?,
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
            spec_capture: false,
            verify_recording: false,
            spec_primed: false,
            lazy_mode: None,
            lazy_hist: LazyHist::default(),
            spec_tap_deferred: false,
            verify_graphs: [None; VERIFY_GRAPH_SLOTS],
            verify_shapes: [0; VERIFY_GRAPH_SLOTS],
            verify_graph_failed: [false; VERIFY_GRAPH_SLOTS],
            verify_dry_done: [false; VERIFY_GRAPH_SLOTS],
            verify_blocks: 0,
            verify_captures: 0,
            verify_replays: 0,
            verify_geom: std::cell::Cell::new(None),
        })
    }

    /// The precomputed RoPE tables, for a DsparkDev built alongside this chain
    /// (`set_rope_tables`): the draft's positions are a subset of the main
    /// chain's, so it reuses the same cos/sin instead of building its own.
    pub fn rope_tables(&self) -> (*const f32, *const f32) {
        (self.cos.as_f32(), self.sin.as_f32())
    }

    /// The DSpark target-hidden tap, `[DSPARK_TAP_SLOTS, dim]` f32, as the raw
    /// pointer [`DsparkDev::import_tap`] consumes. `layer()` rewrites it on every
    /// step the gate is armed for, so it always holds the LAST real step's
    /// recording — which is exactly what the draft that follows that step needs.
    ///
    /// `pub(crate)` for the single-rank parity self-test
    /// ([`crate::dsv41::dspark_parity`]): that tool drives the draft and the
    /// verify block itself instead of through [`Self::dspark_shadow_step`],
    /// because shadow_step feeds the draft's own proposals to the verify by
    /// construction, while the parity check has to feed the TRUE continuation.
    pub(crate) fn dspark_tap_ptr(&self) -> *const f32 {
        self.s.dspark_tap.ptr as *const f32
    }

    /// Golden-comparison dump of one speculative step (`DSV41_DSPARK_DUMP=1`,
    /// see [`crate::dsv41::dump_dev`] for the format): one JSON line with the
    /// tap, the draft block, the verify block's argmax and the accepted prefix,
    /// so the official reference's per-step golden latent can be diffed against
    /// ours element by element.
    ///
    /// RANK 0 ONLY: every TP rank is a separate PROCESS appended to the same
    /// path, so an unguarded dump would interleave `world` copies of every step.
    ///
    /// A failure is LOGGED, never propagated — a debug dump must not answer an
    /// error for a step whose block is already committed.
    fn dspark_dump_step(
        &self,
        mode: &str,
        pos: usize,
        token: u32,
        next: u32,
        k_acc: usize,
        drafts: &[u32],
        verify_out: &[u32],
    ) {
        if !dump_dev::enabled() || self.rank() != 0 {
            return;
        }
        let rec = dump_dev::Step {
            mode,
            pos,
            token,
            next,
            k_acc,
            drafts,
            verify_out,
            // The tap as the step LEFT it: `import_tap` only READS it and the
            // verify's per-row hook writes `dspark_tap_r`, so this is still the
            // `[3, dim]` block the draft consumed — the golden's `main_x`
            // source, read before anything downstream could rewrite it.
            tap: self.s.dspark_tap.ptr as *const f32,
            tap_slots: DSPARK_TAP_SLOTS,
            dim: self.cfg.dim,
        };
        if let Err(e) = dump_dev::write_step(self.dev, &rec) {
            eprintln!("[dsv41] dspark dump (pos {pos}) failed: {e}");
        }
    }

    /// D2H one logits row (`n` f32) from a device pointer. A view, not an
    /// allocation: the row belongs to whichever path computed it, so the probe
    /// only ever reads it.
    fn vrow0_row(&self, ptr: *const f32, n: usize) -> Result<Vec<f32>> {
        let mut lg = vec![0f32; n];
        let b = Device::view(ptr as *mut c_void, n * 4);
        self.dev.download_f32(&b, &mut lg)?;
        Ok(lg)
    }

    /// The probe's verify half: the row 0 logits the verify's own head pass left
    /// in `logits_r` (`step_rows_inner` writes it and NOTHING else ever touches
    /// that buffer — the rollback restores the ring and the compressors, not the
    /// m-row scratch), reduced to `(kernel argmax, host argmax, top-k)`.
    ///
    /// The row's width and index base come from the geometry the SAME
    /// `step_rows_inner` call resolved for the head GEMV and the argmax — read
    /// back from [`Self::verify_geom`] rather than re-derived, because the probe
    /// is a separate call and the pitch is not carried by the type: under
    /// `DSV41_VERIFY_HEAD_SLICED` +
    /// `DSV41_VERIFY_GRAPH=1` row 0 is this rank's `seg`-wide slice (the rows
    /// packed `seg` apart in `logits_r`) and the top-k's indices are offset by
    /// the rank's slice base — the mirror of the probe's eager half below, which
    /// does the same for the single-row `s.logits`. Getting this wrong is a
    /// silent misread (the pitch is not in the type), which is why the decision
    /// lives in ONE function and its RESULT is recorded here.
    fn vrow0_verify(&self, argmax: u32) -> Result<(u32, u32, Vec<(u32, f32)>)> {
        let (n, base) = match self.verify_geom.get() {
            Some((seg, base)) => (seg, base as u32),
            None => (self.cfg.vocab_size, 0),
        };
        let lg = self.vrow0_row(self.s.logits_r.ptr as *const f32, n)?;
        let top = vrow0_topk(&lg, base, vrow0_topk_n());
        let host = top.first().map(|t| t.0).unwrap_or(0);
        Ok((argmax, host, top))
    }

    /// The probe's eager half: ONE plain single-row step at `pos + 1` fed
    /// `token` — the exact forward the verify's row 0 re-schedules — with every
    /// write it makes undone before returning.
    ///
    /// The caller's chain must be at the state the REAL step left it, which is
    /// what [`Self::dspark_shadow_step`]'s rollback guarantees: `pos_ctr ==
    /// pos + 1` and the ring holding `token`'s row at slot `pos % win`.
    ///
    /// What gets undone, and why each entry has to be:
    ///
    /// * the ring row and the compressor carry/counters — one row at `pos + 1`,
    ///   the slot `dspark_snapshot(pos, 1)` covers and `dspark_rollback(pos, 1)`
    ///   puts back (`(pos + 1 + j) % win`, j = 0, is the slot the step writes);
    /// * `pos_ctr` — the step's argmax advanced it to `pos + 2`;
    /// * `s.ids` — the NEXT step's embedding reads it (that is what makes
    ///   `step_dev` a zero-H2D path), and the eager step's argmax overwrote the
    ///   real step's answer. Restoring it is not optional: without it the next
    ///   real step emits the wrong token, which is a BEHAVIOUR change, not a
    ///   measurement.
    fn vrow0_eager(&mut self, token: u32, pos: usize) -> Result<Vrow0Eager> {
        let cfg = self.cfg;
        // The eager row lands at `pos + 1` (`step_dev` below), so the block's row
        // 0 position is `pos + 1`.
        let host = self.dspark_snapshot(pos + 1, 1)?;
        let ids_saved = self.dev.download_u32(self.s.ids.ptr as *const c_void)?;

        let argmax = self.step_dev(token, pos + 1)?;

        // The eager row: the single-row `s.logits`, the whole vocabulary unless
        // `step_body` took its `DSV41_HEAD_SLICE` arm — the SAME condition,
        // repeated here because a slice-local row must be reduced with its own
        // width and index base (the sliced argmax packs the GLOBAL index, so the
        // two sides stay comparable). `s.logits` is allocated at `vocab_size`
        // whatever the arm, so the read is always in bounds; only `n` changes.
        //
        // The one arm this repeats imperfectly: `step_body` WIDENS the row back
        // to the full vocabulary when the .so has no `dsv41_argmax_sliced`
        // (`!ok`). That fallback is invisible from here, so on it the top-k below
        // would be taken over the first `1/world` of a full row — for a probe run
        // on such an .so, set `DSV41_HEAD_SLICE=0`, which makes both sides agree.
        let world = self.world();
        let rank = self.rank();
        let seg = if world > 1 { cfg.vocab_size / world } else { 0 };
        let head_bf16 = self
            .w
            .head
            .as_ref()
            .map(|h| h.dtype == "BF16")
            .unwrap_or(false);
        let sliced = head_slice()
            && head_bf16
            && world > 1
            && cfg.vocab_size % world == 0
            && self.comm.as_ref().map(|c| c.uses_v5()).unwrap_or(false);
        let (n, base) = if sliced {
            (seg, (rank * seg) as u32)
        } else {
            (cfg.vocab_size, 0)
        };
        let lg = self.vrow0_row(self.s.logits.ptr as *const f32, n)?;
        let top = vrow0_topk(&lg, base, vrow0_topk_n());
        let host_argmax = top.first().map(|t| t.0).unwrap_or(0);

        // Undo, in the reverse order of the writes: the ring + the compressor
        // carry, then the counter, then `s.ids`.
        self.dspark_rollback(pos + 1, 1, &host)?;
        self.set_pos_ctr(pos + 1)?;
        self.ul_i32(self.s.ids.ptr, &[ids_saved as i32])?;

        Ok(Vrow0Eager {
            argmax,
            host_argmax,
            top,
            n,
            base,
            sliced,
        })
    }

    /// One probe record: the verify's row 0 against the eager single-row step at
    /// `pos + 1`.
    ///
    /// EVERY rank runs this body — the eager forward's argmax is a collective
    /// under `DSV41_HEAD_SLICE`, so a rank that skipped it would leave its peers
    /// polling — and only rank 0 appends the line (each TP rank is a separate
    /// process writing the same path, exactly as [`Self::dspark_dump_step`] does).
    ///
    /// `with_eager` is `false` on the spec path, where the block is already
    /// committed and there is no state left to compare against.
    fn vrow0_step(
        &mut self,
        mode: &str,
        pos: usize,
        token: u32,
        next: u32,
        d0: u32,
        verify_argmax: u32,
        with_eager: bool,
    ) -> Result<()> {
        let (verify_argmax, verify_host, verify_top) = self.vrow0_verify(verify_argmax)?;
        let eager = if with_eager {
            Some(self.vrow0_eager(d0, pos)?)
        } else {
            None
        };
        if self.rank() != 0 {
            return Ok(());
        }
        let rec = Vrow0Rec {
            mode,
            pos,
            token,
            next,
            d0,
            verify_argmax,
            verify_host_argmax: verify_host,
            verify_top: &verify_top,
            eager: eager.as_ref(),
            rank: self.rank(),
            world: self.world(),
        };
        vrow0_write(&rec)
    }

    /// `DSV41_DIFF_EAGER=1`: re-decode the positions a
    /// [`Self::dspark_spec_step`] just emitted ONE ROW AT A TIME and report the
    /// first position where the two paths disagree.
    ///
    /// # What it answers (and what the bisection cannot)
    ///
    /// The spec path emits `emitted[i]` = `verify_out`'s argmax — the MAIN
    /// chain's own value, read off the m-row batched verify instead of a
    /// single-row step. That substitution is correct exactly when the two agree
    /// (`step_rows`' documented contract: "the rows' per-row outputs must be
    /// BIT-IDENTICAL to the plain engine at the same positions — that parity is
    /// the spec path's correctness contract"). A bisection answers *which
    /// commit* introduced a regression; this answers *which position*, i.e.
    /// which of the emitted tokens is the first one the two paths compute
    /// differently — the first place a spec-only defect is OBSERVABLE in the
    /// token stream.
    ///
    /// # Why an in-process probe and not two serve runs
    ///
    /// Two runs (SPEC vs eager, logs pulled and diffed) cannot answer it: their
    /// KV, their capture history and their prefix state are not the same
    /// objects, so the two sequences are only comparable "from scratch" and the
    /// first divergence is smeared by whatever the prompt/prefill differed in.
    /// The probe re-decodes from the state the spec step JUST left, so both
    /// sides share the prefix bit for bit and the only thing that differs is the
    /// forward that produced the numbers.
    ///
    /// # What it compares, exactly
    ///
    /// `emitted[i]` is the token at `pos + 1 + i`. For `i >= 1` the probe runs
    /// `step_dev(emitted[i - 1], pos + i)` — the plain single-row step fed the
    /// same token at the same position — and compares its argmax against
    /// `emitted[i]`. The prefix the replay reads is the kept one the commit just
    /// left PLUS the rows this probe itself wrote for iterations `< i`, whose
    /// tokens are `emitted[0..i - 1]`; the verify's row `i - 1` read the same
    /// prefix (`drafts[j] == emitted[j]` for every accepted `j`, which is what
    /// acceptance MEANS), so the two forwards see the same context and the only
    /// difference is the batched-per-row vs single-row scheduling. That is the
    /// documented parity contract, tested at the token level rather than at the
    /// logits level.
    ///
    /// `i == 0` is deliberately NOT re-run: `emitted[0]` is the anchor, already
    /// a single-row `step_dev` argmax in every arm, so re-running it would test
    /// reproducibility instead of the two paths against each other.
    ///
    /// # The layout it assumes (holds in all three spec arms)
    ///
    /// `dspark_spec_step` emits `emitted[i]` as the token at `pos + 1 + i` and
    /// leaves the counter at `pos + emitted.len()`. The legacy 5-row block, the
    /// seed-aligned 6-row one and the swallowed one differ in where their ROW 0
    /// sits and in what `pos_base` the commit is called with, not in that
    /// contract. The counter is therefore both the invariant the probe checks
    /// and the state it has to rewind, and a mismatch is reported as an error
    /// rather than compared against the wrong positions.
    ///
    /// # Cost, and why it is a probe and not a feature
    ///
    /// `emitted.len() - 1` extra single-row forwards (~7 ms each) per spec step,
    /// plus one snapshot/rollback pair. Gated OFF by default and never enabled
    /// by any code path: it changes no engine state (see the restore list below)
    /// but it does change the timing, so it must not run under a perf A/B.
    ///
    /// # What it writes, and what is undone (the contract: leave the chain
    /// exactly where `dspark_spec_step` left it)
    ///
    /// * the m-row block's ring rows + compressor carry/counters — saved by
    ///   [`Self::dspark_snapshot`] and put back by [`Self::dspark_rollback`],
    ///   the same pair the verify uses. The slots are exactly the positions the
    ///   replay writes (`pos + 1 ..= pos + emitted.len() - 1`), and `keep == 0`
    ///   is the shadow path's full rollback;
    /// * `pos_ctr` — each replay argmax advances it, so it is re-set to
    ///   `pos + emitted.len()`;
    /// * `s.ids` — the next real step's embedding reads it (that is what makes
    ///   the step self-feeding), and the replay's argmaxes overwrote it.
    ///   Restoring it is not optional: without it the next step emits the wrong
    ///   token, which is a behaviour change, not a measurement;
    /// * `dspark_tap` — the replay re-runs the tap hook, and the SWALLOWED arm
    ///   carries this buffer into the NEXT round's draft (it has no `step_dev`
    ///   of its own to rewrite it). Saved/restored whole, 3 x `dim` f32;
    /// * `decode_steps` / `step_count` — host counters the extra forwards would
    ///   otherwise inflate (`DSV41_TOKTRACE`'s `ds=` field, the graph's
    ///   `decode_steps >= 1` gate).
    ///
    /// Deliberately NOT restored, for the reasons the verify's own rollback gives:
    /// `index_k` (unreachable past the restored `clen`, and re-published by the
    /// next real commit), the ring's compressed rows (same), the engram table
    /// (the replay writes the SAME tokens at the SAME kept positions the verify
    /// wrote, so it is idempotent there and never touches the rejected rows'
    /// slots), and the per-layer `kvp`/`scp` projections (pure functions of
    /// `s.xn`, recomputed every step).
    ///
    /// The embed/argmax path is a HEAD_SLICE collective when that gate is on, so
    /// this MUST be called on EVERY rank, exactly like [`Self::vrow0_step`]; only
    /// the printing is rank 0's.
    pub fn diff_eager_probe(&mut self, pos: usize, emitted: &[u32]) -> Result<DiffEagerReport> {
        let cfg = self.cfg;
        let n = emitted.len();
        let rows = n.saturating_sub(1);
        // Nothing to compare: the anchor is the single-row argmax by
        // construction (see the doc comment).
        if rows == 0 {
            return Ok(DiffEagerReport {
                spec_emitted: emitted.to_vec(),
                eager: emitted.to_vec(),
                first_mismatch: None,
            });
        }
        // The one invariant the probe stands on: the chain is one past the last
        // emitted token, so the block's row 0 sits at `pos + 1`.
        let pos_ctr = self.dev.download_u32(self.s.pos_ctr.ptr as *const c_void)? as usize;
        if pos_ctr != pos + n {
            return Err(FerriteError::Config(format!(
                "diff_eager_probe: the chain's position counter is {pos_ctr}, expected {} \
                 (pos {pos} + {n} emitted) — the probe must be called immediately after \
                 dspark_spec_step and on the state it left",
                pos + n
            )));
        }

        // ---- take: the block's write set, the embedding input, the tap, the
        //      host step counters ----
        let host = self.dspark_snapshot(pos + 1, rows)?;
        let ids_saved = self.dev.download_u32(self.s.ids.ptr as *const c_void)?;
        let steps_saved = (self.decode_steps, self.step_count);
        let tap_len = DSPARK_TAP_SLOTS * cfg.dim;
        let mut tap_saved = vec![0f32; tap_len];
        {
            let b = Device::view(self.s.dspark_tap.ptr, tap_len * std::mem::size_of::<f32>());
            self.dev.download_f32(&b, &mut tap_saved)?;
        }

        // ---- the replay: one plain single-row step per emitted position after
        //      the anchor ----
        self.set_pos_ctr(pos + 1)?;
        let mut eager: Vec<u32> = Vec::with_capacity(n);
        eager.push(emitted[0]);
        let mut replay_err: Option<FerriteError> = None;
        for i in 1..n {
            if let Err(e) = self.ul_i32(self.s.ids.ptr, &[emitted[i - 1] as i32]) {
                replay_err = Some(e);
                break;
            }
            match self.step_dev(emitted[i - 1], pos + i) {
                Ok(tok) => eager.push(tok),
                Err(e) => {
                    replay_err = Some(e);
                    break;
                }
            }
        }

        // ---- put it back, unconditionally: a probe may not leave the chain
        //      dirty even when its own replay failed ----
        self.dspark_rollback(pos + 1, rows, &host)?;
        self.set_pos_ctr(pos + n)?;
        self.decode_steps = steps_saved.0;
        self.step_count = steps_saved.1;
        self.dev.upload_f32_at(self.s.dspark_tap.ptr, 0, &tap_saved)?;
        self.ul_i32(self.s.ids.ptr, &[ids_saved as i32])?;
        if let Some(e) = replay_err {
            return Err(e);
        }

        let first_mismatch = (0..n).find(|&i| eager[i] != emitted[i]);
        Ok(DiffEagerReport {
            spec_emitted: emitted.to_vec(),
            eager,
            first_mismatch,
        })
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
        // The verify block's incoming premix: row r's is the CONSTANT one-hot
        // [1,0,0,0] (the m-row twin of the `premix_const` copy `step_body` makes
        // before its loop), and it does not depend on the row count or the step.
        // Written ONCE per request here — `step_rows` used to re-upload it every
        // verify, which was both a per-verify blocking H2D and the third illegal
        // op inside a capture.
        {
            let hc = self.cfg.hc_mult;
            let mut pm = vec![0f32; VERIFY_ROWS * hc];
            for r in 0..VERIFY_ROWS {
                pm[r * hc] = 1.0;
            }
            self.dev.upload_f32_at(self.s.premix_r.ptr, 0, &pm)?;
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
        // Same two reasons as `step_graph` above, and one more for the verify
        // graph specifically: its `pos_base`-derived launch arguments (the
        // compressor's mode/grid) and `spec_capture`'s host branch were frozen at
        // capture time, and a verify at a position past `window` takes a
        // different `pos_rows` residue. A stale exec would therefore replay
        // against addresses that may now belong to something else at positions it
        // was not recorded for. Drop EVERY pool slot; the next request re-DRYs and
        // re-captures each shape it uses.
        for slot in self.verify_graphs.iter_mut() {
            if let Some(e) = slot.take() {
                self.dev.graph_free(std::ptr::null_mut(), e)?;
            }
        }
        self.verify_shapes = [0; VERIFY_GRAPH_SLOTS];
        self.verify_graph_failed = [false; VERIFY_GRAPH_SLOTS];
        self.verify_dry_done = [false; VERIFY_GRAPH_SLOTS];
        // The warm-up window is per-request too: a new request re-bootstraps the
        // swallow arm (round 1 = legacy), so it re-enters the very transition the
        // window exists for. Carrying the count over would let the second
        // request's graph engage on its FIRST block, i.e. exactly in the window.
        self.verify_blocks = 0;
        // The next request's first spec step bootstraps again (its standalone
        // `step_dev` is what supplies the very first tap).
        self.spec_primed = false;
        // The lazy route is per-request too: the window must not carry the
        // PREVIOUS request's accept rate into this one's first routed round, and
        // the sticky arm must not survive a request boundary (`lazy-batched-gate.md`
        // §2.4). `B`/`c` are deliberately NOT reset — they are process-level.
        self.lazy_mode = None;
        self.lazy_hist = LazyHist::default();
        self.spec_tap_deferred = false;
        // `reset` runs at the START of a request, so this reports the PREVIOUS
        // one: the only place the capture/replay counts of a finished request can
        // be read (the alternative diagnostic is the per-request capture line in
        // `step_rows`). Silent while the switch is off or nothing engaged. With
        // the shape pool in play a request can hold TWO graphs, so `failed` is the
        // any-slot verdict and the shapes that engaged are listed alongside.
        if verify_graph_want()
            && self.rank() == 0
            && (self.verify_captures > 0
                || self.verify_replays > 0
                || self.verify_graph_failed.iter().any(|&f| f))
        {
            let shapes = self
                .verify_shapes
                .iter()
                .enumerate()
                .filter(|(_, &m)| m != 0)
                .map(|(_, &m)| format!("m={m}"))
                .collect::<Vec<_>>()
                .join(",");
            eprintln!(
                "[verify_graph] previous request: captures={} replays={} failed={} shapes=[{}]",
                self.verify_captures,
                self.verify_replays,
                self.verify_graph_failed.iter().any(|&f| f),
                shapes
            );
        }
        self.verify_captures = 0;
        self.verify_replays = 0;
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

    /// The verify path's BLOCK norm: `rows` rows of `dim`, row stride `dim`, one
    /// shared weight row `w[dim]`, each row normalised independently.
    ///
    /// `DSV41_NORM_MROWS=1` (and an `.so` exporting `dsv41_rmsnorm_rows`) routes
    /// the call through the dsv41-side multi-row entry — the same per-row
    /// program, owned by this chain's own kernel file. With the gate off, or on a
    /// stale `.so`, the pre-existing `ferrite_rmsnorm` call is issued unchanged,
    /// so OFF is bit-for-bit today's verify.
    ///
    /// Only the `m`-row sites come through here; the decode path's single-row
    /// `layer()` norms stay on `rmsnorm` (folding one row saves nothing).
    fn norm_rows(
        &self,
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        if norm_mrows() && self.dev.rmsnorm_rows(x, w, out, rows as i32, dim as i32, eps)? {
            return Ok(());
        }
        self.dev.rmsnorm(x, w, out, rows as i32, dim as i32, eps)
    }

    /// [`Self::norm_rows`] with an explicit stream. Only the verify block's
    /// attention dual chain (`DSV41_VERIFY_FORK`) calls it: its kv norm rides the
    /// second side stream under the q chain. Same gate, same two kernels, same
    /// fallback order — only the stream changes, which is precisely what makes
    /// the split bit-identical to the serial block.
    fn norm_rows_on(
        &self,
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        rows: usize,
        dim: usize,
        eps: f32,
        s: CuStream,
    ) -> Result<()> {
        if norm_mrows() && self.dev.rmsnorm_rows_on(x, w, out, rows as i32, dim as i32, eps, s)? {
            return Ok(());
        }
        self.dev.rmsnorm_on(x, w, out, rows as i32, dim as i32, eps, s)
    }

    /// Quantise `rows` FP32 rows of `cols` values each into the block scratch
    /// `s.xq_r` / `s.xsc_r` — the multi-row form of [`Self::quant1`].
    ///
    /// `quant_kernel` dispatches one thread-group per (row, 32-element block) and
    /// derives the row index from it (`r = idx / nb`), so a `rows = m` launch
    /// emits byte-for-byte what `m` separate `rows = 1` launches emit: the row
    /// index only picks the base pointers, and each row's block amax is its own
    /// shuffle reduction over its own 32 values. This is the activation staging
    /// the multi-row projections consume — [`Self::proj_mrows`] reads it as the
    /// contiguous `[m, k]` / `[m, k/32]` blocks, and `wo_a_grouped_fp8` with
    /// `a_stride` set to the row pitch.
    ///
    /// It deliberately bypasses `quant1`'s T1/T2 launch-elision flags: both are
    /// keyed on `s.xn` / `s.qr`, and no caller here ever passes those (the
    /// multi-row pass feeds `s.xn_r` / `s.qr_r`), so the elision never fired for
    /// this path and the flags' state is unchanged by using this instead.
    fn quant_rows(&self, src: *const f32, rows: usize, cols: i32) -> Result<()> {
        // (#5) The fp8 staging's ROW PITCH is `cols` bytes on `s.xq_r` and
        // `cols/32` f32 on `s.xsc_r`, and both are sized for `VERIFY_ROWS` rows of
        // the WIDEST activation the verify quantises (`dim.max(nh*hd)`, see the
        // `xq_r` field comment). A call past either bound writes past the buffer —
        // and the projection that consumes it (`proj_mrows`) reads the SAME pitch,
        // so a wrong `cols` is a silent misread, not a type error. Static: no
        // device traffic, compiled out of a release build.
        debug_assert!(
            rows <= VERIFY_ROWS,
            "quant_rows: {rows} rows exceed the {VERIFY_ROWS}-row fp8 staging"
        );
        let pitch = self.cfg.dim.max(self.cfg.n_heads * self.cfg.head_dim);
        debug_assert!(
            cols as usize <= pitch,
            "quant_rows: cols {cols} exceeds the staging row pitch ({pitch} bytes)"
        );
        self.dev.quant_fp8(
            src,
            self.s.xq_r.ptr as *mut u8,
            self.s.xsc_r.ptr as *mut f32,
            rows as i32,
            cols,
            32,
            true,
        )
    }

    /// ONE `gemm_fp8_mrows` launch for a projection whose activation is already
    /// staged in `s.xq_r` / `s.xsc_r` by [`Self::quant_rows`]: the m-row form of
    /// [`Self::lin`], weight-stationary over the block's rows.
    ///
    /// This is NOT `gemm_fp8_mx` at `m = rows`. That symbol dispatches on `m`
    /// between two different programs — the SIMT warp-per-row GEMV at `m == 1`
    /// and a 16-row tile MMA at `m > 1` — so raising `m` on it would silently
    /// change the summation. `gemm_fp8_mrows` reproduces the `m == 1` consume
    /// expression, its ascending-kb walk and its `shfl_xor` tree per row, which
    /// is what keeps "row r of an m-row launch == the m=1 decode of row r" true.
    ///
    /// `Ok(false)` = NOT performed — the caller keeps its per-row loop (the
    /// bit-exact reference it was verified against). It declines when the loaded
    /// .so lacks the symbol, the C entry declines the shape or the process's
    /// `DSV41_GEMV_FP8_MODE` / `DSV41_NO_GEMV_FP8` / `DSV41_GEMV_A32` gates refuse
    /// it, or the swapAB opt-in is on (that arm is explicitly NOT bit-identical to
    /// the SIMT gemv and keeps its own kernel).
    ///
    /// `DSV41_GEMV_A32` (Direction B, LANDED): the mrows kernel carries BOTH
    /// activation forms and selects on the same process gate the M=1 GEMV reads,
    /// so the multi-row launch is the same program at either setting —
    /// a32=1 (the default) materialises the `s_lut[byte] * sa` operand in
    /// registers before folding (the form the m=1 `gemm_fp8_mx`'s GEMV reads out
    /// of its block-wide `s_af`), a32=0 folds the same product inline. The
    /// launcher no longer declines the default, so the multi-row path is live
    /// again; `DSV41_GEMV_A32` remains a pure occupancy knob.
    fn proj_mrows(
        &self,
        w: *const u8,
        ws: *const u8,
        out: *mut f32,
        rows: usize,
        n: i32,
        k: i32,
        out_stride: i32,
    ) -> Result<bool> {
        if Self::swapab() {
            return Ok(false);
        }
        self.dev.gemm_fp8_mrows(
            self.s.xq_r.ptr as *const u8,
            self.s.xsc_r.ptr as *const f32,
            w,
            ws,
            std::ptr::null(),
            out,
            rows as i32,
            n,
            k,
            out_stride,
        )
    }

    /// K1 (DSV41_ATTN_MROWS2, default OFF): the two-family form of
    /// [`Self::proj_mrows`] — the verify's wq_a + wkv pair in ONE launch, since
    /// both project the SAME quantised activation row (`s.xq_r`/`s.xsc_r`,
    /// written once by the caller's `quant_rows`) with the same `k`. The code is
    /// `gemm_fp8_mrows_kernel<M>` with the family folded in as four base-pointer
    /// selections, so row `row` of family `f` is bit-identical to row `row` of
    /// the separate `gemm_fp8_mrows` launch it replaces (the kernel header
    /// carries C1-C6 and the warp-level family split argument).
    ///
    /// `Ok(false)` = NOT performed — the caller keeps its two `proj_mrows`
    /// launches. It declines when the loaded .so lacks the symbol, the C entry
    /// declines the shape, or the swapAB opt-in is on (that arm is explicitly
    /// NOT bit-identical to the SIMT gemv and keeps its own kernel — the same
    /// exclusion [`Self::proj_mrows`] applies).
    ///
    /// `out_stride_q`/`out_stride_kv` are separate because the wq_a output row
    /// strides by `ql` and the wkv row by `hd`. They equal `n_q`/`n_kv` at this
    /// call site (both output blocks are dense `[m, n]` rows) and are passed
    /// explicitly anyway, mirroring [`Self::proj_mrows`]'s own `out_stride`
    /// parameter rather than assuming a layout.
    #[allow(clippy::too_many_arguments)]
    fn proj_mrows2(
        &self,
        w_q: *const u8,
        ws_q: *const u8,
        out_q: *mut f32,
        n_q: i32,
        out_stride_q: i32,
        w_kv: *const u8,
        ws_kv: *const u8,
        out_kv: *mut f32,
        n_kv: i32,
        out_stride_kv: i32,
        rows: usize,
        k: i32,
    ) -> Result<bool> {
        if Self::swapab() {
            return Ok(false);
        }
        self.dev.gemm_fp8_mrows2(
            self.s.xq_r.ptr as *const u8,
            self.s.xsc_r.ptr as *const f32,
            w_q,
            ws_q,
            out_q,
            n_q,
            out_stride_q,
            w_kv,
            ws_kv,
            out_kv,
            n_kv,
            out_stride_kv,
            rows as i32,
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

    /// [`Self::ar_hc_post_fold`] for the VERIFY (m-row) chain: fold this block's
    /// `hc_post_inplace_rows` into the AR that just produced its `x`.
    ///
    /// Same contract as the single-row fold — the AR is the last writer of the
    /// payload and the hc-post reads/writes only the residual stream, so the two
    /// can share one launch — with the two multi-row differences:
    ///
    ///  * the payload is `rows * dim` floats (row stride `dim`): the AR result of
    ///    row `r` IS that row's `x`, which the standalone
    ///    `hc_post_inplace_rows` would have re-read out of the same buffer; and
    ///  * the residual / coefficients are the per-row slices the m-row
    ///    `hc_mixes` wrote — `h_r + r*hc*dim`, `post_r + r*hc`,
    ///    `comb_r + r*hc*hc` — the exact layout `hc_post_inplace_rows` takes.
    ///
    /// Returns `Ok(true)` when the fold ran, in which case the caller
    /// ([`Self::layer_rows`]) MUST skip its `hc_post_rows`. Every decline — the
    /// `DSV41_VERIFY_AR_FOLD` gate (default OFF), `fuse_c()` /
    /// `hc_verify_fuse()` off, a shape the launcher rejects, or a `.so` without
    /// `ferrite_p2p_ar_v5_hcpost_rows` — returns `Ok(false)` and leaves the
    /// caller on the `all_reduce_inplace` + `hc_post_rows` pair it used before.
    ///
    /// The tail-split join sits HERE rather than in the caller, for the same
    /// reason as the single-row fold: the fused epilogue is the first consumer of
    /// the LATE half's `comb`/`post` (and its residual write clobbers memory the
    /// LATE half is still reading), so the side stream's join event must be waited
    /// on the main stream BEFORE this launch. No-op when no split is armed.
    fn ar_hc_post_fold_rows(
        &self,
        c: &std::sync::Arc<Collective>,
        buf: *mut std::ffi::c_void,
        len: usize,
        rows: usize,
    ) -> Result<bool> {
        if !Self::verify_ar_fold()
            || !Self::fuse_c()
            || !Self::hc_verify_fuse()
            || !self.dev.supports_ar_hcpost_rows()
        {
            return Ok(false);
        }
        self.dev.hc_tail_join()?;
        let (dim, hc) = (self.cfg.dim, self.cfg.hc_mult);
        c.all_reduce_inplace_hcpost_rows(
            buf,
            len,
            self.s.h_r.ptr as *mut f32,
            self.s.post_r.as_f32(),
            self.s.comb_r.as_f32(),
            rows as i32,
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

    /// Put the continuation's first input token back on the device.
    ///
    /// `step_dev` deliberately performs NO H2D for the token: it reads `s.ids`,
    /// which the PREVIOUS step's argmax wrote (chain_dev.rs's argmax epilogue), so
    /// the steady-state decode loop is self-feeding. That contract is broken by a
    /// snapshot RESUME: `kv_restore` restores every cross-step state EXCEPT
    /// `s.ids`, which would still hold whatever token the PREVIOUS request's last
    /// step produced — so the first decode step after a prefix-cache HIT would
    /// condition on the wrong token AND write that token's KV into the restored
    /// prefix (silent, total corruption of the continuation). Callers resuming
    /// from a snapshot must call this with the continuation's first token.
    pub fn prime_input(&mut self, token: u32) -> Result<()> {
        self.ul_i32(self.s.ids.ptr, &[token as i32])
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
            // The two backbone-alignment dumps read the device from INSIDE the
            // region a capture would record (`dump_topk` in step_body before the
            // argmax, `dump_layer_norm` at each layer's end), and a device->host
            // copy is illegal inside a capture - so, exactly like `probes`, arming
            // either one forces the per-kernel path. Same cached-gate discipline:
            // this closure runs once (the OnceLock holds the answer).
            let dumps = topk_dump() || layer_dump();
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
                && !dumps
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
                bf16_truncate(),
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
        // DSV41_TOPK_DUMP: the head's row, read back HERE — after the projection
        // and before the argmax dispatch — so the probe sees exactly the logits
        // the argmax consumes. The row's geometry follows `sliced` above (the
        // full vocabulary on the plain path, one rank's slice under
        // DSV41_HEAD_SLICE with world > 1).
        if topk_dump() {
            if sliced {
                self.dump_topk(pos, (rank * seg) as u32, seg)?;
            } else {
                self.dump_topk(pos, 0, cfg.vocab_size)?;
            }
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
    //    (dsv41_kernels.cu), so the verify runs the three-launch path with ONE
    //    pool+commit pair per row ([`Self::compress_row`], interleaved inside
    //    `attention_rows`) — the state/pool kernels' decode branches carry ONE row
    //    each, which is exactly what a per-row call gives them. A block that
    //    completes several groups (ratio 2, m = 6: up to 3) is modelled: each of
    //    those groups is committed and its index key published at the row that
    //    produced it.
    // 2. INDEXER: the candidate count is the live device counter, which includes
    //    the group this row may have just committed — its index key is now
    //    published BEFORE that row's selection (per-row interleave), matching the
    //    single-row order, so newly created groups are still part of the candidate
    //    set. Excluding them is a modelling decision that has not been taken.
    // 3. HEAD: vocabulary-sliced by default (`DSV41_VERIFY_HEAD_SLICED`, ON) —
    //    each rank projects its own 1/world of the replicated head and
    //    `dsv41_argmax_sliced_rows` picks the global winner per row in ONE v5
    //    epoch round for the whole block. It is NOT wired through
    //    `argmax_sliced`: that entry consumes a full round PER CALL and only
    //    `pos_ctr` can be nulled, never the round, so m rows would cost m
    //    serialised cross-rank round-trips. `pos_ctr` stays NULL (the accept
    //    logic owns it). The full-vocabulary per-row path remains the fallback
    //    for `DSV41_VERIFY_HEAD_SLICED=0`, a stale .so, `world == 1` or a
    //    non-BF16 head — see `verify_head_geom`.
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
    ///
    /// # The CUDA graph (`DSV41_VERIFY_GRAPH=1`)
    ///
    /// ```text
    ///   [outside]  D2H pos_ctr, H2D ids_r, H2D pos_rows, H2D premix_r (reset)
    ///   [INSIDE ]  embed -> engram -> 40 x layer -> head -> per-row argmax
    ///              (writes argmax_r; no host round trip anywhere in between)
    ///   [outside]  D2H argmax_r, then the caller's snapshot/rollback/commit
    /// ```
    ///
    /// Everything the recorded region reads is a `Scratch` buffer whose ADDRESS and
    /// CONTENT are stable per verify: the ids and the row positions are refreshed
    /// OUTSIDE the capture (a blocking copy is legal there — the requirement is
    /// only that it is not inside the recording), the premix is a constant written
    /// by `reset`, and the position counter / layer latent counters / AR epoch are
    /// device-resident, so no launch argument changes from verify to verify.
    ///
    /// Order of first use (the DRY -> CAPTURE discipline `step_impl` uses, with one
    /// extra pass): the FIRST `step_rows` of a request runs direct — it warms every
    /// kernel, builds the lazy engram device state and sizes cublas' workspaces, all
    /// of which are illegal inside a capture; the SECOND captures (recording does
    /// not execute, so the graph is launched immediately after instantiation to do
    /// this verify's real work); every later one replays. The graphs are dropped by
    /// [`Self::reset`].
    ///
    /// That schedule runs PER SHAPE, because the execs live in the shape pool
    /// (`verify_graphs`): a request that emits `m = 5` first and `m = 6` afterwards
    /// (SEED_ALIGN/SWALLOW) DRYs slot 0, then DRYs slot 1 on its first 6-row block,
    /// and captures each shape on that shape's SECOND visit. Without the per-shape
    /// slot the first DRY latched `m = 5` alone and every 6-row verify — i.e. all of
    /// them — fell back to the direct launches, so the graph never engaged at all.
    ///
    /// The verify's snapshot/rollback/commit are OUTSIDE the graph on purpose (they
    /// are the caller's orchestration, and their host slot arithmetic has no device
    /// twin): the snapshot is taken before the replay and the rollback/commit after
    /// the D2H, so nothing about the accept logic changes.
    pub fn step_rows(&mut self, toks: &[u32]) -> Result<Vec<u32>> {
        // Every EXTERNAL caller keeps the historic rendezvous: the barrier in front
        // of a replay is what pairs with the CAPTURE arm of this same function (and
        // with `step_impl`'s), so a peer that is still recording cannot lose a stamp
        // an executing rank is polling for. Only the lazy row loop — where every rank
        // is provably executing the SAME replay — passes `true`; see
        // [`Self::step_rows_sync`].
        self.step_rows_sync(toks, false, None)
    }

    /// [`Self::step_rows`] with the per-replay rendezvous switchable and the
    /// `pos_ctr` read-back skippable.
    ///
    /// `skip_barrier` suppresses the ENTRY `host_barrier` of every arm that has
    /// one: the stored-graph REPLAY arm, the DRY arm and the DIRECT arm. It never
    /// touches the two barriers that bracket a CAPTURE (`capture_begin` ..
    /// `capture_end`), because those are exactly the rendezvous that keep a
    /// recording rank from being lapped by an executing peer.
    ///
    /// The suppressed rendezvous is redundant once no rank can be recording:
    /// under `ar_v5()` (which the graph gate requires whenever there are peers)
    /// the device-side protocol is self-synchronising — `p2p_ar_pubred_v5_kernel`
    /// publishes this rank's round stamp to every peer and then POLLS all peers'
    /// stamps for the same round before it reduces (ferrite_kernels.cu:8898-8928),
    /// and `end_round` already relies on exactly that ("v5 needs no host barrier",
    /// tp.rs:665-670). Only the lazy row loop passes `true`: it has already
    /// rendezvoused once before its loop, and every rank then runs the SAME replay
    /// sequence back-to-back on one stream — so the suppressed barrier is uniform
    /// across the ranks there, which is the one property the arms' counts have to
    /// keep (see the DRY arm for what happens when they do not).
    ///
    /// `pos_base_hint` is the blocking D2H of `pos_ctr` this call would otherwise
    /// make, for a caller that has JUST written it and therefore already knows the
    /// value. `None` (every external caller) reads it back, so nothing changes off
    /// the lazy path. The hint is only sound under the invariant that the caller's
    /// H2D has landed and NOTHING device-side has run since: `lazy_run_row` does
    /// exactly that (`set_pos_ctr(pos + i)` — a blocking `cudaMemcpy` H2D — with the
    /// device quiescent because the previous row ended in a blocking D2H), so the
    /// device counter equals the hint by construction.
    fn step_rows_sync(
        &mut self,
        toks: &[u32],
        skip_barrier: bool,
        pos_base_hint: Option<i32>,
    ) -> Result<Vec<u32>> {
        let m = toks.len();
        if m == 0 {
            return Ok(Vec::new());
        }
        if m > VERIFY_ROWS {
            return Err(FerriteError::Config(format!(
                "step_rows: {m} rows exceed the allocated verify block ({VERIFY_ROWS})"
            )));
        }
        // (#6) The closing D2H below reads `m` i32s out of `argmax_r`; the buffer
        // is `VERIFY_ROWS` of them. Constructive: the block SIZE is the contract
        // between `m` and the buffer, and `m > VERIFY_ROWS` is already rejected
        // above — this pins the byte count the D2H will touch. Static.
        debug_assert!(
            m * 4 <= self.s.argmax_r.bytes,
            "step_rows: {m} argmax rows exceed the {}-byte argmax_r buffer",
            self.s.argmax_r.bytes
        );
        // The row positions. One D2H: the counter is device-resident (the argmax
        // advances it), and EVERY row's position has to be materialised somewhere
        // because the kernels that take a position take a POINTER. This runs
        // between steps, so the read cannot stall anything that matters; the values
        // are then written to `pos_rows`, which the graph only ever READS.
        //
        // The read-back is skipped when the caller supplies `pos_base_hint` (the
        // lazy row loop, which has JUST written that value — see
        // [`Self::step_rows_sync`]). It is a full device sync, and paying it once
        // per row is the per-row overhead this parameter exists to remove.
        let pos_base = match pos_base_hint {
            Some(p) => p,
            None => self.dev.download_u32(self.s.pos_ctr.ptr as *const c_void)? as i32,
        };
        let pos_rows: Vec<i32> = (0..m).map(|r| pos_base + r as i32).collect();

        // The graph's per-verify inputs, refreshed before every replay: both land
        // in the SAME device buffers the capture recorded (`Scratch` members live
        // as long as the chain does), so the recorded addresses stay valid.
        let ids: Vec<i32> = toks.iter().map(|&t| t as i32).collect();
        self.ul_i32(self.s.ids_r.ptr, &ids)?;
        self.ul_i32(self.s.pos_rows.ptr, &pos_rows)?;
        // (3) the row-position table's first entry is the block's row-0 position.
        // Spot-checked right after the H2D, before anything reads it, so a table
        // the kernels read as `pos_base + r` is confirmed at its head. ONE D2H 4 B.
        self.inv_pos_rows_first(pos_base)?;

        // The gate reads `verify_blocks`, so the counter is advanced AFTER it:
        // a request's first verify block is block `0` for the `SWALLOW` warm-up
        // window (see [`Self::verify_graph_gate`]).
        let graph_slot = self.verify_graph_gate(m, pos_base);
        self.verify_blocks += 1;
        if let Some(idx) = graph_slot {
            if !self.verify_dry_done[idx] {
                // DRY: a REAL execution, so every lazy first-use cost happens
                // outside the capture. Its device effects are the caller's to
                // keep or roll back, exactly like any other verify. Run once PER
                // SHAPE: the second shape's DRY happens while the first shape's
                // graph already sits in its slot (harmless — `step_rows_inner`
                // never touches the pool).
                //
                // ★ RENDEZVOUS, at the arm's ENTRY (ar5-hang's method A). The
                // four arms of this function must each arrive at the barrier the
                // SAME number of times per epoch, because `SpinBarrier` is a
                // generation counter over ARRIVALS: one arm taking a wait fewer
                // than a peer's closes the epoch without that peer and every
                // later barrier in the request is one generation out — the
                // silent misphase `ar5-hang` traced (its four arms measured
                // DRY = 0, replay = 1, capture = 2, direct = 0 rendezvous each,
                // and the gate's inputs are per-RANK, so which arm a rank takes
                // is not guaranteed to agree across the 8 ranks of a TP8 step:
                // a capture refusal, a `compress_branch_steady` mirror that has
                // not settled, a shape slot taken on one rank and free on
                // another). Making every arm's ENTRY a rendezvous means two
                // ranks that disagree on the ARM still agree on the epoch — the
                // hazard narrows to the capture arm's second barrier (below),
                // which is the one method B — a rank-synchronised arm decision —
                // is needed for.
                //
                // The earlier note here argued against this barrier on lock-order
                // grounds ("a rank spinning device-side on a peer's stamp while
                // its peer waits for it at the host barrier"). That position is
                // the CAPTURE arm's entry barrier, which is already there and
                // already verified (`DSV41_VERIFY_GRAPH=1` runs bit-identical
                // under `ar_v5()`): a rank at a host barrier has, by construction,
                // already published the stamp its own device work owed, and the
                // v5 AR publishes BEFORE it polls (`p2p_ar_pubred_v5_kernel`), so
                // a peer's poll cannot be waiting on a rank that is merely parked
                // here. The asymmetry, not the barrier, was the bug.
                if !skip_barrier {
                    if let Some(c) = self.comm.as_ref() {
                        c.host_barrier();
                    }
                }
                self.step_rows_inner(toks, m, pos_base)?;
                self.verify_dry_done[idx] = true;
                self.verify_shapes[idx] = m;
            } else if let Some(e) = self.verify_graphs[idx] {
                // Rendezvous before the replay: a capture only RECORDS the AR
                // kernels while a peer may already be EXECUTING its own — the same
                // pair `step_impl` keeps around its capture. The lazy row loop
                // passes `skip_barrier`, because it has already rendezvoused once
                // before its loop and every rank is then executing the same replay
                // back-to-back on one stream (see [`Self::step_rows_sync`]).
                if !skip_barrier {
                    if let Some(c) = self.comm.as_ref() {
                        c.host_barrier();
                    }
                }
                self.dev.graph_launch(e)?;
                self.verify_replays += 1;
                self.advance_compress_lens(pos_base, m);
            } else {
                if let Some(c) = self.comm.as_ref() {
                    c.host_barrier();
                }
                // The capture records WITHOUT executing, so the device state stays
                // where the caller's snapshot was taken ...
                let saved = self.compress_lens();
                // The graph is an OPTIMISATION: no capture failure may take the
                // verify down with it. A driver refusal (an op the static audit
                // approved that the driver still rejects, an unsupported
                // primitive on a stale `.so`, an instantiate failure) leaves this
                // request on the DIRECT launches, and this SLOT's
                // `verify_graph_failed` latch stops the doomed capture from being
                // retried on every step (the other shape's slot is unaffected).
                let (g, cap_err) = self.capture_verify(toks, m, pos_base);
                // ... but the HOST code inside the recording ran for real either
                // way: `compress_rows` advances its `compress_len` mirror by the
                // rule the commit kernel applies on the device, and on the failure
                // arm no kernel ran at all. Leaving the mirror ahead would make
                // the NEXT step's `comp_len` branch take a path the device counter
                // does not agree with, so it is put back before either arm runs.
                self.restore_compress_lens(&saved);
                if let Some(c) = self.comm.as_ref() {
                    c.host_barrier();
                }
                if cap_err.is_some() || g.is_null() {
                    // ★ PRINT, DO NOT SILENTLY DEGRADE. The graph is an
                    // optimisation, so a refusal is *designed* to leave the request
                    // on the direct launches — which makes "graph on but never
                    // engaged" and "graph engaged" indistinguishable from the
                    // outside. That ambiguity is exactly how an A/B reports "no
                    // change" for a feature that never ran (the trap this switch
                    // was born from), so the cause crosses to the console here.
                    let why = cap_err
                        .as_ref()
                        .map(|e| format!("{e}"))
                        .unwrap_or_else(|| "capture_end returned a null graph".into());
                    eprintln!(
                        "[verify_graph] {} capture FAILED (m={m} pos={pos_base}): {why} — this \
                         request finishes on the direct launches (slot latched)",
                        verify_graph_name(m)
                    );
                    self.verify_graph_failed[idx] = true;
                    if !g.is_null() {
                        let _ = self.dev.graph_free(g, std::ptr::null_mut());
                    }
                    self.step_rows_inner(toks, m, pos_base)?;
                } else {
                    let e = self.dev.graph_instantiate(g)?;
                    self.dev.graph_free(g, std::ptr::null_mut())?;
                    self.dev.graph_launch(e)?; // the capture did not execute
                    self.verify_graphs[idx] = Some(e);
                    self.verify_captures += 1;
                    // The A/B's proof that `DSV41_VERIFY_GRAPH=1` took effect: one
                    // line per request (rank 0 only — the ranks are threads of one
                    // process and would otherwise print it `world` times). The name
                    // carries the shape, so with the pool holding two graphs the
                    // log shows WHICH one engaged.
                    if self.rank() == 0 {
                        eprintln!(
                            "[verify_graph] captured {} at pos={pos_base} — every later \
                             verify of this shape replays (DSV41_VERIFY_GRAPH=1)",
                            verify_graph_name(m)
                        );
                    }
                    self.advance_compress_lens(pos_base, m);
                }
            }
        } else {
            // ★ THE DIRECT ARM's entry rendezvous (ar5-hang's method A). This arm
            // is taken both when the graph is OFF for the whole process and when
            // it is ON but this RANK declined (a shape with no free slot, a
            // latched capture failure, an unsettled `compress_branch_steady`, a
            // driver that cannot record). Only the second case can coexist with a
            // peer in another arm, so the barrier is conditioned on the graph
            // being configured on: with `DSV41_VERIFY_GRAPH` off every rank takes
            // this arm and the count (0) already agrees, which is what keeps the
            // default path bit-for-bit what it was. With it on, this arm now
            // matches the DRY and replay arms at one rendezvous each — see the
            // DRY arm for why the counts have to agree.
            if !skip_barrier && verify_graph_want() {
                if let Some(c) = self.comm.as_ref() {
                    c.host_barrier();
                }
            }
            self.step_rows_inner(toks, m, pos_base)?;
            // (lazy verify's bare-chain path was missing this: `advance_compress_lens`
            // is called after the graph_launch but NOT after the bare chain, so the
            // host mirror of compress_len drifted from the device counter on every
            // m=1 bare-chain call — lazy's per-row step_rows(m=1) hits exactly this
            // path when no m=1 verify graph exists, and each row's committed groups
            // were invisible to the host mirror, corrupting the NEXT row's indexer
            // candidates. This is the direct source of lazy's Bristol/burdens/oqua
            // and the repetition — the KV/compute misalignment the user called out.)
            self.advance_compress_lens(pos_base, m);
        }

        // D2H of the m argmaxes, always OUTSIDE the graph (a device read is illegal
        // inside a capture). `download_u8` keeps this a plain byte copy, so no f32
        // reinterpretation is involved.
        let mut bytes = vec![0u8; m * 4];
        let b = Device::view(self.s.argmax_r.ptr, m * 4);
        self.dev.download_u8(&b, &mut bytes)?;
        let rows: Vec<u32> = (0..m)
            .map(|r| {
                u32::from_le_bytes([
                    bytes[4 * r],
                    bytes[4 * r + 1],
                    bytes[4 * r + 2],
                    bytes[4 * r + 3],
                ])
            })
            .collect();
        // (8) every verify row's argmax is a real vocabulary index. Free: the m
        // rows are already here (the closing D2H above), so this inspects exactly
        // what the caller hands to the accept chain. A failure means the head
        // wrote at the wrong pitch / the argmax read past its row.
        self.inv_argmax_rows(&rows, pos_base)?;
        Ok(rows)
    }

    /// Can THIS verify of THIS shape go through the graph right now?
    ///
    /// Returns the SHAPE POOL SLOT this call is cleared into, or `None` when it
    /// must take the direct launches. Every condition is either a concrete
    /// capture hazard or a shape mismatch:
    ///
    /// * `DSV41_VERIFY_GRAPH` on (default OFF — the A/B);
    /// * `pos_base >= 1`: the compressor's launcher picks its mode and grid from
    ///   the HOST `start_pos` it is handed (`dsv41_compressor_pool`,
    ///   dsv41_kernels.cu:6880-6890) and a capture freezes them. Mode 2 (decode)
    ///   is what a multi-row verify is; mode 1 (`start_pos == 0`, prefill) is a
    ///   different program with a different grid, so position 0 must never be the
    ///   recording. The verify always runs after its anchor step, i.e. at
    ///   `pos + 1 >= 1`, so this is a guard rather than a case. NOTE the per-step
    ///   decision (`out_rows_val`) is NOT affected: the kernel already derives it
    ///   from the DEVICE counter pointer it is given (`pos_rows[r]`, dereferenced
    ///   in-kernel at dsv41_kernels.cu:3029).
    /// * a pool SLOT for `m` (see [`Self::verify_slot`]): a slot already holding
    ///   that shape, or a free one. The capture bakes the per-row launch geometry,
    ///   so a shape with no slot (a THIRD shape — the parity self-test varies `m`
    ///   — or a pool already full) takes the direct path. This replaces the old
    ///   single-`m` latch, which let the request's FIRST shape (`m = 5`) block
    ///   every later one (`m = 6`) and silently cost the whole feature.
    /// * this SLOT has not already latched a capture failure (`verify_graph_failed`).
    /// * `!eng_host()`: the `DSV41_ENG_HOST` fallback hashes on the HOST and
    ///   uploads per row (`upload_bytes_at` = blocking H2D).
    /// * `ar_v5()` whenever there are peers: a host barrier is not a CUDA call, so
    ///   it would not be recorded and the replayed graph would silently lose the
    ///   inter-rank synchronisation. NOTE this is SATISFIED BY DEFAULT: `ar_v5()`
    ///   is `DSV41_GRAPH_STEP != 0 || DSV41_AR_V5 != 0` with both legs defaulting
    ///   ON (tp.rs), and the step graph is default ON since 2026-09-11 — so under
    ///   TP8 the condition is true unless BOTH `DSV41_GRAPH_STEP=0` and
    ///   `DSV41_AR_V5=0` are set. It is NOT an opt-in blocker, and an A/B that
    ///   never engages the graph is not explained by this clause (read the
    ///   `[verify_graph]` lines `step_rows` prints on capture / capture failure).
    /// * `supports_memset_async()`: `compress_proj_rows` zeroes `scp_r` on the
    ///   `ratio == 1` no-gate path, and `zero_at_on` would fall back to the
    ///   SYNCHRONOUS `cudaMemset` on the legacy stream without the symbol.
    /// * `supports_dspark_snapshot()`: not a capture requirement of THIS graph
    ///   (the P0 pair runs outside it, so a stale `.so` would merely be slower),
    ///   but a deliberate conservative gate: the audit that cleared this capture
    ///   was run against the kernel set that carries them.
    /// * neither `DSV41_STATS` nor `DSV41_PHASE`: both probe from the host inside
    ///   the recorded region (`step_impl` excludes them for the same reason).
    /// * the host `comp_len > 0` branch must already be in its steady state — see
    ///   [`Self::compress_branch_steady`], which is the guard for the audit's S7.
    /// The shape pool's slot for a verify block of `m` rows: the slot ALREADY
    /// holding that shape, else the first EMPTY one, else `None` (the pool is
    /// full of other shapes). Pure — the gate and `step_rows` resolve the same
    /// slot from the same state, so the slot the gate approves is the slot the
    /// call uses.
    fn verify_slot(&self, m: usize) -> Option<usize> {
        if let Some(i) = self.verify_shapes.iter().position(|&s| s == m) {
            return Some(i);
        }
        // `0` is the EMPTY marker; `step_rows` rejects `m == 0` before the gate,
        // so a zero-row block can never claim a slot here.
        self.verify_shapes.iter().position(|&s| s == 0)
    }

    /// THIS RANK's arm for a verify block of `m` rows at `pos_base` — the
    /// decision [`Self::verify_graph_gate`] used to make inline, as a value the
    /// ranks can vote on (see [`VerifyArm`]). Pure: the gate, the vote and the
    /// dispatch in [`Self::step_rows_sync`] all read the same state, so the arm
    /// that is voted on is the arm that runs.
    fn verify_arm_local(&self, m: usize, pos_base: i32) -> (VerifyArm, Option<usize>) {
        if !verify_graph_want() || pos_base < 1 {
            return (VerifyArm::Direct, None);
        }
        // (ar5-hang method C) `DSV41_SWALLOW_STEP`'s warm-up window. The swallow
        // arm's first rounds are the ONLY place a request runs TWO verify shapes
        // in sequence (`m = 5` bootstrapping verify, then `m = 6` forever), so
        // they are the only place the shape pool is driven through its full
        // DRY → CAPTURE → REPLAY transition — and the only place two ranks can
        // sit in DIFFERENT arms of `step_rows_sync`, whose rendezvous counts
        // differ (the capture arm brackets its recording with two barriers, every
        // other arm takes one). `SpinBarrier` is a generation counter, so a rank
        // that arrives a different number of times per epoch closes an epoch its
        // peers never entered and every later barrier is off by one — the stall
        // `ar5-hang` traced, appearing only in a request's first rounds.
        //
        // Holding the graph off for the first `SWALLOW_GRAPH_WARMUP_BLOCKS`
        // blocks removes that window: while it is open every rank takes the same
        // (direct) arm, and by the time it closes both the kernels and the
        // per-RANK gate inputs (`compress_branch_steady`'s host mirror, the shape
        // pool) have settled, so the arm decision below is made from a state the
        // ranks agree on. See the constant's note.
        //
        // Outside `SWALLOW` nothing changes: the gate is `swallow_step()`-only,
        // and with the switch off every verify already runs one shape.
        if swallow_step() && self.verify_blocks < SWALLOW_GRAPH_WARMUP_BLOCKS {
            return (VerifyArm::Direct, None);
        }
        // The shape pool first: without a slot there is nothing to check.
        let Some(idx) = self.verify_slot(m) else {
            return (VerifyArm::Direct, None);
        };
        if self.verify_graph_failed[idx] {
            // A capture already failed for THIS shape this request. Re-attempting
            // it every step would burn the failure on the hot path for nothing.
            // The other slot is independent and still usable.
            return (VerifyArm::Direct, None);
        }
        if self.verify_graphs[idx].is_some() {
            // The stored graph is already committed to this slot's shape and its
            // environment was checked when it was armed; the shape matches by
            // construction, so only the hazards below could still disqualify the
            // call.
            return (VerifyArm::Replay, Some(idx));
        }
        if self.verify_dry_done[idx] {
            // The DRY execution has happened and no graph is stored yet: this
            // block is the CAPTURE. Split from REPLAY on purpose — the two return
            // the same slot but take a different number of rendezvous, which is
            // exactly what the vote in [`Self::verify_graph_gate`] is about.
            return (VerifyArm::Capture, Some(idx));
        }
        let armed = !eng_host()
            && !stats_dbg()
            && !phase_dbg()
            && self.compress_branch_steady()
            && self.dev.supports_dspark_snapshot()
            && self.dev.supports_memset_async()
            && (self.comm.is_none() || crate::dsv41::tp::ar_v5());
        if armed {
            (VerifyArm::Dry, Some(idx))
        } else {
            (VerifyArm::Direct, None)
        }
    }

    /// The gate [`Self::step_rows_sync`] calls: this rank's arm made UNANIMOUS
    /// across the ranks, or the direct launches when they disagree (Plan B,
    /// "unanimity-or-direct").
    ///
    /// # Why a vote and not just a barrier
    ///
    /// Every input of the arm decision is per-RANK — `verify_graph_failed`,
    /// `verify_dry_done`, the shape pool, `compress_branch_steady`'s host mirror,
    /// a `pos_base` read off this rank's own counter — while the arms cost
    /// DIFFERENT numbers of `host_barrier` arrivals per block ([`VerifyArm`]).
    /// `SpinBarrier` counts arrivals per generation, so two ranks in different
    /// arms close an epoch their peers never entered and every later barrier in
    /// the request is one generation out. That is `ar5-hang`, and its signature
    /// is exactly what the A/B showed: `DSV41_SWALLOW_STEP` + the verify graph
    /// reproduces it (those rounds are the only place a request drives the shape
    /// pool through DRY → CAPTURE → REPLAY), while `DSV41_VERIFY_GRAPH=0` removes
    /// it for the trivial reason that every rank then walks the same arm.
    ///
    /// So the decision is broadcast, not taken: each rank votes its own arm (one
    /// `i32` code) and the ranks proceed only on UNANIMITY. A single dissent sends
    /// the whole world to the direct launches — the conservative direction, since
    /// that arm exists in every state and is a real execution of the same block,
    /// and since a majority rule would leave a minority walking an arm whose
    /// rendezvous count the majority does not share.
    ///
    /// The rendezvous this adds is a HOST atomic exchange
    /// ([`Collective::unanimous_i32`], on the shared `SpinBarrier`'s own
    /// [`crate::dsv41::tp::RankVote`] — an independent generation, so it cannot
    /// perturb the arrival epochs `host_barrier` hands out). It is NOT a CUDA call
    /// and NOT inside any capture region: the gate runs before the arm, while
    /// every launch of the block still lives on the caller's stream, which is why
    /// the vote can never be recorded into a graph.
    ///
    /// # Cost and scope
    ///
    /// One vote per gate call, taken by EVERY rank (whether or not that rank has a
    /// graph), whenever `DSV41_VERIFY_GRAPH` is configured: ~a handful of atomics
    /// on one shared line, orders below the device work it guards. With the switch
    /// off the function returns before the vote, so the default path is untouched
    /// — as it is without peers (`comm` is `None` off TP8, and there is nobody to
    /// agree with).
    fn verify_graph_gate(&self, m: usize, pos_base: i32) -> Option<usize> {
        let (arm, idx) = self.verify_arm_local(m, pos_base);
        // The env switch is process-wide, so with it off every rank is in the
        // Direct arm already and there is nothing to agree about — and no
        // rendezvous is taken, which is what keeps the default path bit-for-bit
        // what it was.
        if !verify_graph_want() {
            return idx;
        }
        let Some(c) = self.comm.as_ref() else {
            return idx;
        };
        // Unanimity includes our own vote, so on `Some(_)` the agreed code IS this
        // rank's arm and `idx` is the slot for it. `None` (any disagreement) sends
        // everyone to the direct launches.
        match c.unanimous_i32(arm as i32) {
            Some(code) if code == arm as i32 => idx,
            _ => {
                self.note_arm_dissent(arm);
                None
            }
        }
    }

    /// ★★ THE ROUND'S ARM IS A **UNANIMOUS** DECISION, AND EVERY RANK VOTES
    /// (the `ar5-hang` root cause, 2026-09-12).
    ///
    /// # The lesion this closes
    ///
    /// `spec_primed` is a **per-rank** chain flag, but the arm it selects has a
    /// **process-wide** cost: the legacy and aligned arms each call `step_dev`
    /// (the whole single-row chain — 40 layers × 2 AR ≈ 80 v5 all-reduce rounds
    /// per step) and the swallowed and lazy arms do NOT (their verify block's
    /// row 0 IS the anchor's forward). AR v5 has no host-side rendezvous, so a
    /// rank with a LARGER round footprint waits on a peer stamp its peer will
    /// never publish — forever. One rank that loses its priming (the swallowed
    /// arm's failure contract deliberately leaves the flag alone, see
    /// [`Self::dspark_spec_swallowed`]'s "Failure") therefore rifts the epochs
    /// for the REST of the request: it falls back to the legacy arm (+~80
    /// rounds per step) while its peers keep swallowing, and the rank that
    /// issues MORE rounds is the one that spins on `ar5_wait_round` — the
    /// `ar5-hang` signature.
    ///
    /// # The decision
    ///
    /// Every rank publishes its OWN `spec_primed` bit, and the round is a
    /// swallow round only when the ranks agree that ALL of them are primed. On
    /// any disagreement the whole world walks the legacy path for this round,
    /// and that path's tail re-primes **every** rank unconditionally
    /// (`spec_primed = true` at the exit of [`Self::dspark_spec_step`]) — so the
    /// next round is unanimous again: the degradation is one round wide, and it
    /// heals itself. The vote is taken BEFORE any arm's device work, because an
    /// arm must be agreed before any rank walks it.
    ///
    /// # The invariant the callers must keep
    ///
    /// **The vote must be taken by EVERY rank on EVERY round** while
    /// [`swallow_step`] is on — whether or not that rank is primed. The
    /// rendezvous is symmetric ([`crate::dsv41::tp::RankVote::unanimous_i32`]
    /// waits for `world` votes), so a rank that short-circuited the call on its
    /// own `spec_primed` would leave its primed peers spinning in the vote
    /// forever — the fix would have swapped one hang for another. That is also
    /// why this helper publishes the LOCAL bit instead of a constant `1`: a
    /// non-primed rank that voted `1` would drag its primed peers into the
    /// swallowed arm — exactly the rift this closes.
    ///
    /// # Cost and scope
    ///
    /// One rendezvous per round: `world` releases + `world` acquires on ONE
    /// shared cache line, orders of magnitude below the ~6 ms round it guards.
    /// With the gate off the caller short-circuits before this call, so the
    /// default path stays bit-for-bit what it was. Without peers (`comm` is
    /// `None` off TP>1, and a world of one is unanimity by construction) the
    /// function answers `true` with no rendezvous at all — the single-rank path
    /// is unchanged, and the caller's own `spec_primed` still decides.
    fn spec_primed_unanimous(&self) -> bool {
        let Some(c) = self.comm.as_ref() else {
            return true;
        };
        c.unanimous_i32(i32::from(self.spec_primed)) == Some(1)
    }

    /// ★ PRINT, DO NOT SILENTLY DEGRADE.
    ///
    /// A failed vote falls back to the direct launches and the request still runs,
    /// which makes "the ranks agreed all along" and "the ranks kept disagreeing,
    /// so the graph never engaged" indistinguishable from the outside. An A/B that
    /// cannot see that difference reports "no change" for a feature that never
    /// ran (the trap the capture diagnostics exist for), so the first few dissents
    /// cross to the console, with the arm that was given up. Once per process,
    /// rank 0 only — the ranks are threads of ONE process and would otherwise
    /// print the same line `world` times.
    fn note_arm_dissent(&self, arm: VerifyArm) {
        static PRINTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        if self.rank() != 0 {
            return;
        }
        let n = PRINTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 4 {
            eprintln!(
                "[verify_graph] arm vote DISAGREED at verify block {} (this rank wanted {arm:?}) \
                 — every rank runs the direct launches for this block \
                 (Plan B unanimity-or-direct)",
                self.verify_blocks
            );
        }
    }

    /// Record one [`Self::step_rows_inner`] into a fresh graph.
    ///
    /// Returns `(graph, error)`: a non-null graph with `None` on success, or a
    /// null graph with the first failure otherwise. The stream is ALWAYS taken
    /// out of capture mode before returning — when `capture_begin` succeeded and
    /// the recording failed, `capture_end` is what ends it — so the caller's
    /// fallback launch is legal. That is the whole point of this helper: the
    /// `?` on the two capture calls would leave a failed recording open and
    /// propagate, which is exactly what the fallback must not do.
    fn capture_verify(
        &mut self,
        toks: &[u32],
        m: usize,
        pos_base: i32,
    ) -> (*mut std::ffi::c_void, Option<FerriteError>) {
        if let Err(e) = self.dev.capture_begin() {
            // Nothing was recorded, so there is no capture to end.
            return (std::ptr::null_mut(), Some(e));
        }
        // The recording is a LAUNCH SEQUENCE, so no host branch inside it may
        // depend on a per-verify value: `committed` does, and it decides whether
        // the per-row index-key publish is emitted (see
        // [`Self::verify_recording`] and [`Self::indexer_rows_one`]). Armed for
        // exactly the recorded call and cleared before `capture_end`, so the
        // fallback launch below — which re-runs `step_rows_inner` on the direct
        // path — is unaffected and eager stays bit-identical.
        self.verify_recording = true;
        let inner = self.step_rows_inner(toks, m, pos_base);
        self.verify_recording = false;
        let end = self.dev.capture_end();
        match (inner, end) {
            (Ok(()), Ok(g)) => (g, None),
            (inner, end) => {
                let e = end.err().or_else(|| inner.err()).expect("one arm failed");
                (std::ptr::null_mut(), Some(e))
            }
        }
    }

    /// The guard for the audit's S7 (`comp_len` frozen into the captured branch).
    ///
    /// `attention_rows` branches on the HOST mirror `comp_len > 0` to pick one of
    /// three compressed-half behaviours ([`Self::indexer_rows_one`], nothing for a
    /// consumer, or the per-row [`Device::comp_placeholder`]), and a capture
    /// freezes that choice. The mirror is CUMULATIVE and monotonically
    /// non-decreasing across a run (this block's commits add to it, and
    /// `dspark_rollback[_keep]` puts back exactly the prefix it kept), so
    /// "every compress source has already committed something" is a steady state:
    /// from then on every verify — recorded or direct — takes the `comp_len > 0`
    /// arm for that layer. Waiting for it costs a couple of positions (with
    /// `ratio == 2` and m == 6 the first verify already commits three groups) and
    /// removes the one host value in the recorded region that would otherwise
    /// differ between the capture and a later replay.
    fn compress_branch_steady(&self) -> bool {
        self.compress_sources()
            .iter()
            .all(|&l| self.layers[l].compress_len > 0)
    }

    /// The host MIRROR of every layer's device committed-row counter — the state
    /// [`Self::compress_row`] advances from HOST code while the commit kernel
    /// advances the device counter. A capture runs the host code but no kernel, so
    /// the mirror has to be put back afterwards (see [`Self::step_rows`]).
    fn compress_lens(&self) -> Vec<usize> {
        self.layers.iter().map(|c| c.compress_len).collect()
    }

    fn restore_compress_lens(&mut self, saved: &[usize]) {
        for (c, &n) in self.layers.iter_mut().zip(saved) {
            c.compress_len = n;
        }
    }

    /// Advance the host mirrors for a verify block the DEVICE has just executed
    /// (a replayed graph runs no host code at all), by the same per-row rule
    /// [`Self::compress_row`] applies: row `r` at `pos_base + r` commits one
    /// latent when `(pos_base + r + 1) % ratio == 0`. Only the layers that OWN a
    /// compressor are counted, and only those that [`Self::compress_proj_rows`] does
    /// not decline (`comp_wkv`/`comp_norm` present) — a consumer layer inherits
    /// the count ([`Self::source_compress_len`]) and writes no state of its own.
    ///
    /// The caller's rollback restores these from its own snapshot, so an
    /// unaccepted block is undone exactly as in the direct path.
    fn advance_compress_lens(&mut self, pos_base: i32, m: usize) {
        for l in self.compress_sources() {
            let ld = &self.w.layers[l];
            if ld.comp_wkv.is_none() || ld.comp_norm.is_none() {
                continue; // `compress_row` declines the layer: it commits nothing
            }
            let ratio = self.cfg.compress_ratio(l).max(1) as i32;
            let mut add = 0usize;
            for r in 0..m {
                if (pos_base + r as i32 + 1) % ratio == 0 {
                    add += 1;
                }
            }
            self.layers[l].compress_len += add;
        }
    }

    /// The verify head's SLICE geometry — `Some((row_stride, index_base))` for
    /// `logits_r`, `None` when the verify's head is NOT sliced.
    ///
    /// ONE place decides this, because three call sites have to agree: the head
    /// GEMV writes the rows `row_stride` apart, the argmax reads them at the same
    /// pitch, and the probe ([`Self::vrow0_verify`]) reads row 0 at the same width
    /// and index base. A second copy of the condition is exactly how the two
    /// would drift apart — and `logits_r`'s row pitch is INVISIBLE to its type,
    /// so a drift is a silent misread rather than a type error.
    ///
    /// Sliced requires ALL of:
    ///
    /// * `DSV41_VERIFY_HEAD_SLICED` (default ON),
    /// * a BF16 head weight (the sliced arm is the bf16 GEMV) — the f32 arm has
    ///   no slice,
    /// * `world > 1` with a live v5 collective, because the exchange IS one round
    ///   of that epoch sequence (`comm.uses_v5()`),
    /// * a vocabulary divisible by the world (the same condition the eager
    ///   `DSV41_HEAD_SLICE` arm takes),
    /// * [`Device::supports_argmax_sliced_rows`] — tested HERE, before any layout
    ///   choice, so a stale .so keeps `logits_r` at the full-vocabulary pitch
    ///   instead of leaving a half-sliced buffer for the fallback to misread,
    /// * room for the WHOLE verify block's keys in the v5 slot
    ///   (`VERIFY_ROWS * 8 <= bytes`). That is the batched entry's own decline
    ///   arm, checked here for the worst-case row count so the geometry cannot
    ///   depend on `m` — a geometry that changed with `m` would leave the head,
    ///   the argmax and the probe disagreeing on different verifies.
    /// Every condition that fails is reported ONCE by
    /// [`VerifyHeadGeom::note_once`], so a `None` here is a cause and not just a
    /// fact. Before that report existed, "the verify head is not sliced" could
    /// only be diagnosed by re-deriving the seven conditions by hand — the
    /// fallback path had zero observability.
    ///
    /// # What a `None` here does NOT disable
    ///
    /// `DSV41_VERIFY_HEAD_MROWS`'s fold LOOKS like it lives behind this function
    /// (its call site sits inside the `Some` branch of the caller), but the
    /// geometry only decides the SLICE. [`Self::step_rows_inner`] also folds the
    /// FULL-vocabulary head with the same v1 kernel on its unsliced arm, so a
    /// `None` no longer silently no-ops that gate — see
    /// [`verify_head_mrows_note`], which says out loud which arm ran.
    fn verify_head_geom(&self) -> Option<(usize, usize)> {
        let cfg = self.cfg;
        // Capture-INDEPENDENT, deliberately: every verify path takes the SAME arm.
        //
        // The batched argmax exchange this geometry arms (`dsv41_argmax_sliced_rows`)
        // emits one round of the v5 epoch sequence, and a verify that does NOT emit
        // its round while the replayed graph does is exactly one round of permanent
        // drift per step (the DRY warm-up run of a graph slot and the plain eager
        // path with `DSV41_VERIFY_GRAPH=0` both hit this). An earlier revision
        // answered `None` outside a capture, i.e. forced those runs onto the
        // full-vocabulary head at the vocab pitch plus the ordinary per-row
        // `argmax` — a counter drift, not a topology fix. The DRY must advance the
        // epoch once, like the replay it pre-warms, so the arm is chosen by the
        // geometry alone and the head, the argmax and the probe below can never
        // disagree.
        //
        // NOTE this function is NOT a guard on the epoch either: the device entry
        // advances `*epoch` UNCONDITIONALLY whenever it is called, exactly like the
        // AR it shares the counter with (`all_reduce_inplace` → `p2p_ar_v5`, no
        // capturing check) — see [`Device::argmax_sliced_rows`].
        //
        // Deliberately NOT part of [`VerifyHeadGeom`]/`note_once`: nothing is
        // misconfigured on the unsliced arm — the env gate / symbol / v5 / slot
        // conditions already report themselves — and the full-vocabulary head is a
        // legitimate fallback, not a fault.
        let comm = self.comm.as_ref();
        let geom = VerifyHeadGeom {
            env: verify_head_sliced(),
            head_bf16: self
                .w
                .head
                .as_ref()
                .map(|h| h.dtype == "BF16")
                .unwrap_or(false),
            world: self.world(),
            vocab: cfg.vocab_size,
            symbol: self.dev.supports_argmax_sliced_rows(),
            v5: comm.map(|c| c.uses_v5()).unwrap_or(false),
            slot_fit: comm.map(|c| VERIFY_ROWS * 8 <= c.bytes).unwrap_or(false),
            slot_bytes: comm.map(|c| c.bytes).unwrap_or(0),
        };
        if !geom.sliced() {
            geom.note_once();
            return None;
        }
        let seg = cfg.vocab_size / geom.world;
        Some((seg, self.rank() * seg))
    }

    /// The verify forward as it is RECORDED: everything from the embedding to the
    /// per-row argmax, with no host round trip anywhere inside (the ids, the row
    /// positions and the premix are all on the device before this runs, and the
    /// `argmax_r` D2H is the caller's). `pos_base` is the host value the block's
    /// ROW 0 sits at — it is only read where the compressor's launcher needs a
    /// scalar, and it IS frozen by a capture (see [`Self::verify_graph_gate`]).
    fn step_rows_inner(&mut self, toks: &[u32], m: usize, pos_base: i32) -> Result<()> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let hc = cfg.hc_mult;

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
        self.norm_rows(
            self.s.x_r.ptr as *const f32,
            self.w.norm.as_ref().unwrap().as_f32(),
            self.s.xn_r.ptr as *mut f32,
            m,
            dim,
            cfg.norm_eps,
        )?;

        // ---- head + per-row argmax ----
        // Same call `step_body` makes for its single token, with two differences:
        // the m rows are one block instead of one token, and (DSV41_VERIFY_HEAD_SLICED,
        // default ON) each rank reads only its OWN 1/world of the head.
        //
        // The head is the block's single largest weight: `head.weight` is
        // Shard::Replicated [129280, 5120] bf16 = 1262 MB on every rank
        // (weights.rs:90). The unsliced arm streams all of it ONCE PER ROW (6 x
        // 298us = 1.79ms); the sliced one streams 158 MB/rank/row for the same m
        // launches and reaches the same token, because the head is REPLICATED
        // (every rank can reduce any slice) and the exchange's packed key carries
        // the GLOBAL index with the single-row tie rule (lowest index).
        //
        // The GEMV stays PER ROW on purpose: `verify_head_geom`'s doc records why
        // the folded `head_gemv_bf16_mrows` is NOT used here (it is the v2 kernel
        // program; the eager head is v1, so folding is a numerical change — see
        // `verify_head_fold`). Row r is therefore the SAME `gemv_bf16` launch the
        // eager sliced head makes, i.e. the eager head's exact values.
        //
        // The logits row PITCH follows the arm (`seg` under the slice, the full
        // vocabulary otherwise) and is fixed by `verify_head_geom` alone — the
        // head, the argmax below and the probe all index `logits_r` through it.
        let head = self.w.head.as_ref().unwrap();
        let geom = self.verify_head_geom();
        // Record the arm for the vrow0 probe: the probe is a separate call that
        // can run after `capture_end` (and after the peers the geometry was
        // resolved against have moved), so it reads the pitch back instead of
        // re-deriving it.
        self.verify_geom.set(geom);
        let lg_stride = geom.map(|(stride, _)| stride).unwrap_or(cfg.vocab_size);
        // The fold is the UNSLICED arm's alternative only: it changes the K order
        // (a numerical change), so it is never combined with the slice.
        let folded = geom.is_none()
            && verify_head_fold()
            && if head.dtype == "BF16" {
                self.dev.head_gemv_bf16_mrows(
                    head.ptr(),
                    self.s.xn_r.ptr as *const f32,
                    self.s.logits_r.ptr as *mut f32,
                    m as i32,
                    cfg.vocab_size as i32,
                    dim as i32,
                )?
            } else {
                false
            };
        if let Some((seg, _)) = geom {
            // THIS rank's vocabulary slice: `head.weight` is [vocab, dim] bf16
            // row-major, so rank `rank`'s rows start `rank * seg * dim` elements
            // (2 bytes each) into it — the same offset arithmetic `step_body`
            // uses for the eager sliced head.
            let head_ptr = (head.ptr() as *const u8).wrapping_add(self.rank() * seg * dim * 2);
            // `DSV41_VERIFY_HEAD_MROWS=1` folds the block's rows into ONE
            // V1-ORDER launch (`dsv41_gemv_bf16_v1_mrows`), which streams this
            // rank's slice once instead of `m` times. Row r of the fold is
            // BIT-IDENTICAL to the per-row `gemv_bf16` below (v1's `c = lane;
            // c += 32` chain, one accumulator per row, v1's `__shfl_xor_sync`
            // tree, no K-split — the kernel header carries the argument), so this
            // is a traffic + launch change and not a numerical one. Default OFF:
            // a stale .so (no symbol) or `m > 8` makes the method return
            // `Ok(false)` and the per-row loop below still runs.
            let armed = verify_head_mrows();
            let mrows = armed
                && self.dev.head_gemv_bf16_v1_mrows(
                    head_ptr as *const c_void,
                    self.s.xn_r.ptr as *const f32,
                    self.s.logits_r.ptr as *mut f32,
                    m as i32,
                    seg as i32,
                    dim as i32,
                )?;
            if !mrows {
                // Armed-but-declined is this gate's OTHER silent no-op (a stale
                // .so without the symbol, or `m > 8`): the per-row loop below
                // answers, so an A/B with the gate ON would otherwise measure the
                // OLD path with nothing said. One line, once.
                if armed {
                    verify_head_mrows_note(true, true, false, cfg.vocab_size, m);
                }
                for r in 0..m {
                    let xnr = (self.s.xn_r.ptr as *const f32).wrapping_add(r * dim);
                    let lg = (self.s.logits_r.ptr as *mut f32).wrapping_add(r * lg_stride);
                    self.dev.gemv_bf16(
                        head_ptr as *const c_void,
                        xnr,
                        lg,
                        seg as i32,
                        dim as i32,
                    )?;
                }
            }
        } else if !folded {
            // ---- the UNSLICED arm's fold (the structural dead gate's fallback) ----
            //
            // `DSV41_VERIFY_HEAD_MROWS`'s original call site sits inside the SLICED
            // branch above, so with a slice-less geometry it was not even
            // evaluated: an armed gate silently ran the per-row head on every A/B
            // (the mrows-decline investigation's "structural dead gate"). The SAME
            // v1 kernel folds the FULL-vocabulary head just as validly, and for the
            // same stated reason: `n = vocab` is still `>= GEMV_V2_MAX_N`, so the
            // per-row launch this replaces is v1's `gemv_bf16` — the very program
            // the kernel header transcribes (one accumulator per row, `c = lane;
            // c += 32`, v1's shuffle tree, no K-split). So this is the UNSLICED
            // arm's traffic change (m passes over the replicated 1262 MB head →
            // one) and not a numerical one, exactly like the sliced fold.
            //
            // It is NOT `verify_head_fold`: that one is the v2/`nt` program
            // (`head_gemv_bf16_mrows`) and is a numerical change, which is why it
            // is gated separately and tested first (`folded` above). A non-BF16
            // head has no v1 kernel to fold, so the `lin_f32` loop keeps it.
            let armed = verify_head_mrows();
            let mrows = armed
                && head.dtype == "BF16"
                && self.dev.head_gemv_bf16_v1_mrows(
                    head.ptr(),
                    self.s.xn_r.ptr as *const f32,
                    self.s.logits_r.ptr as *mut f32,
                    m as i32,
                    cfg.vocab_size as i32,
                    dim as i32,
                )?;
            // Say which arm ran (or why none did) once — the whole point of the
            // fix: an armed gate must never be a silent no-op.
            if armed {
                verify_head_mrows_note(false, head.dtype == "BF16", mrows, cfg.vocab_size, m);
            }
            if !mrows {
                for r in 0..m {
                    let xnr = (self.s.xn_r.ptr as *const f32).wrapping_add(r * dim);
                    let lg = (self.s.logits_r.ptr as *mut f32).wrapping_add(r * lg_stride);
                    if head.dtype == "BF16" {
                        self.dev
                            .gemv_bf16(head.ptr(), xnr, lg, cfg.vocab_size as i32, dim as i32)?;
                    } else {
                        self.lin_f32(xnr, dim as i32, head, cfg.vocab_size as i32, lg)?;
                    }
                }
            }
        }
        if let Some((seg, base)) = geom {
            // ONE cross-rank exchange for the WHOLE block: the m local slice
            // argmaxes are published together and the ranks max over them in
            // ascending rank order, so row r's global winner (and its
            // lowest-index tie rule) is the full-vocabulary argmax's.
            let c = self.comm.as_ref().unwrap();
            let ok = self.dev.argmax_sliced_rows(
                self.s.logits_r.ptr as *const f32,
                seg as i32,
                base as i32,
                lg_stride as i32,
                self.s.argmax_r.ptr as *mut std::os::raw::c_int,
                self.s.argmax_packed_r.ptr as *mut u64,
                // NULL pos_ctr: this argmax must NOT advance the counter (the
                // kernel null-checks it) — the accept logic advances it once, for
                // the accepted prefix. The v5 EPOCH advance is unconditional and
                // is exactly one round for the whole block, the same one round the
                // single-row eager head pays.
                std::ptr::null_mut(),
                c.peer_slots_u64(),
                c.peer_stamps_u32(),
                c.epoch_dev(),
                c.staging_dev() as *mut u64,
                c.ready_local_dev(),
                self.world() as i32,
                self.rank() as i32,
                m as i32,
                c.bytes as i64,
            )?;
            if !ok {
                // Unreachable by construction: `verify_head_geom` gated on the
                // symbol AND on `VERIFY_ROWS * 8 <= bytes` (>= `m * 8`), which are
                // the entry's two decline arms. A decline here would mean the two
                // sides disagree about the geometry, and the head has ALREADY
                // written `logits_r` at the slice pitch — a silent fallback would
                // then read full-vocabulary rows and return wrong tokens, which is
                // the one failure this path must not have.
                return Err(FerriteError::Config(
                    "dsv41_argmax_sliced_rows declined a shape the host gated as legal".into(),
                ));
            }
        } else {
            for r in 0..m {
                let lg = (self.s.logits_r.ptr as *mut f32).wrapping_add(r * lg_stride);
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
        }
        // The `argmax_r` D2H is deliberately NOT here: it is a device read, which a
        // capture forbids, so the caller ([`Self::step_rows`]) issues it after the
        // replay for every arm of the graph decision.
        Ok(())
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
    /// `compress_proj_rows`/`compress_row` gate in `attention_rows`
    /// (`compress_ratio > 0 && is_kv_source`). A consumer of the same group only
    /// INHERITS the count and writes no state of its own.
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
    /// 1. the `m` ring slots the block lands in, `(pos + 1 + j) % window`. ONE
    ///    launch per owner (`dsv41_dspark_ring_save`) covers all `m` slots: the
    ///    slots are `head_dim` floats each and the sequence wraps at the ring's
    ///    end, so a single contiguous copy is not available, but the kernel
    ///    computes each slot on the device. Historically this was one D2D per
    ///    slot, which made the pair the post-launch bottleneck of a verify step
    ///    (40 owners x m slots x 2 directions x ~5us) and pinned it ineligible
    ///    for a CUDA graph (a host-computed slot is a capture-time constant).
    ///
    /// For every COMPRESS SOURCE ([`Self::compress_sources`]):
    ///
    /// 2. `state_kv` + `state_score` — the compressor carry, the full
    ///    `ratio * head_dim` each. The buffers are max-sized and each copy uses
    ///    its own layer's byte count (`ratio` is per layer: 1 or 2 here).
    /// 3. `latent` — **not** a per-step scratch. `indexer()`/`indexer_rows_one()` read
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
    ///   `qr_raw_r` (R2b) is that flag's m-row twin and IS read/written by the
    ///   verify: `attention_rows` sets it (on every path) and the same layer's
    ///   indexer q half consumes it, so it is produced and retired inside one
    ///   layer and cannot leak either.
    /// * `pos_ctr`: never advanced by the verify — `step_rows` passes a NULL
    ///   counter to its per-row argmax.
    /// * everything in `s.*_r`: the m-row scratch is the verify's own.
    ///
    /// `pub(crate)` for the parity self-test ([`crate::dsv41::dspark_parity`]),
    /// which needs the same save/restore pair around its own `step_rows` call.
    ///
    /// `pos_base` is the position of the block's ROW 0 — the row `j` of the
    /// block sits at `pos_base + j`. The legacy 5-row block (`[d1..d5]`) starts
    /// at one past the single-row step's own position (`pos + 1`), while the
    /// swallowed 6-row block (`[anchor, d1..d5]`) starts AT the step's position
    /// (the anchor's forward IS row 0) — which is the one number the two callers
    /// pass differently.
    pub(crate) fn dspark_snapshot(&self, pos_base: usize, m: usize) -> Result<Vec<(usize, usize)>> {
        let cfg = self.cfg;
        let hd = cfg.head_dim;
        let win = cfg.window_size;
        let max_ratio = cfg.compress_ratios.iter().copied().max().unwrap_or(1).max(1);
        debug_assert!(m <= VERIFY_ROWS, "the snapshot is sized for a {VERIFY_ROWS}-row block");
        // P0: ONE launch per layer replaces the per-slot `cudaMemcpyAsync` loop
        // (40 owners x m slots per direction). The kernel's slot is
        // `(pos_base + j) % win`: that is the SAME residue as `(pos_base + j) % win`
        // for every `j < win` (`m <= VERIFY_ROWS` << `win`), and the reduction
        // keeps the argument inside `c_int` for positions past 2^31. The bytes are
        // identical either way; `fused == false` (a stale .so without the kernels)
        // keeps the original per-slot path.
        let fused = self.dev.supports_dspark_snapshot();
        let base = (pos_base % win) as i32;

        for &l in &self.ring_owners() {
            let ring = self.layers[l].ring.ptr as *const f32;
            let snap = (self.s.dspark_snap_ring.ptr as *mut f32)
                .wrapping_add(l * VERIFY_ROWS * hd);
            if fused {
                self.dev
                    .dspark_ring_save(snap, ring, base, win as i32, hd as i32, m as i32)?;
            } else {
                for j in 0..m {
                    let slot = (pos_base + j) % win;
                    self.dev.memcpy_d2d(
                        snap.wrapping_add(j * hd) as *mut c_void,
                        ring.wrapping_add(slot * hd) as *const c_void,
                        hd * std::mem::size_of::<f32>(),
                    )?;
                }
            }
        }

        let mut host = Vec::new();
        for &l in &self.compress_sources() {
            let ratio = cfg.compress_ratio(l).max(1);
            let cache = &self.layers[l];
            if fused {
                // One launch covers state_kv + state_score + latent + the two
                // 4-byte counters; the segments sit at the same bases the
                // memcpy path below used.
                self.dev.dspark_comp_save(
                    cache.state_kv.ptr as *const f32,
                    cache.state_score.ptr as *const f32,
                    cache.latent.ptr as *const f32,
                    (self.s.clen.ptr as *const i32).wrapping_add(l),
                    cache.out_rows.ptr as *const i32,
                    (self.s.dspark_snap_state.ptr as *mut f32)
                        .wrapping_add(l * 2 * max_ratio * hd),
                    (self.s.dspark_snap_latent.ptr as *mut f32).wrapping_add(l * hd),
                    (self.s.dspark_snap_clen.ptr as *mut i32).wrapping_add(l),
                    (self.s.dspark_snap_out_rows.ptr as *mut i32).wrapping_add(l),
                    ratio as i32,
                    max_ratio as i32,
                    hd as i32,
                )?;
                host.push((l, cache.compress_len));
                continue;
            }
            let bytes = ratio * hd * std::mem::size_of::<f32>();
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
    ///
    /// `pub(crate)` for the parity self-test, which pairs it with
    /// [`Self::dspark_snapshot`] around its own `step_rows` call.
    pub(crate) fn dspark_rollback(
        &mut self,
        pos_base: usize,
        m: usize,
        host: &[(usize, usize)],
    ) -> Result<()> {
        self.dspark_rollback_keep(pos_base, m, 0, host)
    }

    /// [`Self::dspark_rollback`] with a KEEP prefix: the first `keep` rows of the
    /// verify block STAY in the ring (they are the accepted prefix's KV — see
    /// [`Self::dspark_spec_step`]), only rows `keep..m` are restored to their
    /// pre-verify value.
    ///
    /// The compressor side is restored WHOLE regardless of `keep`: its `state_kv`
    /// / `state_score` slots are written in position order and later rows
    /// overwrite earlier ones, so there is no per-row undo — the only exact way
    /// to land on "the state after `keep` rows" is to go back to the snapshot and
    /// play those rows forward again ([`Self::compress_replay`]).
    ///
    /// `keep == 0` is exactly the shadow path's full rollback.
    ///
    /// `pos_base` is the block's ROW 0 position (see [`Self::dspark_snapshot`]):
    /// row `j` lives at slot `(pos_base + j) % win`.
    pub(crate) fn dspark_rollback_keep(
        &mut self,
        pos_base: usize,
        m: usize,
        keep: usize,
        host: &[(usize, usize)],
    ) -> Result<()> {
        debug_assert!(keep <= m, "dspark_rollback_keep: keep {keep} > block {m}");
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
        // P0: mirror of `dspark_snapshot` -- one launch per layer per side.
        // `pos_base` follows the same `pos_base % win` reduction.
        let fused = self.dev.supports_dspark_snapshot();
        let base = (pos_base % win) as i32;

        for &l in &self.ring_owners() {
            let ring = self.layers[l].ring.ptr as *mut f32;
            let snap = (self.s.dspark_snap_ring.ptr as *const f32)
                .wrapping_add(l * VERIFY_ROWS * hd);
            // Only the rows BEYOND the kept prefix go back: rows `0..keep` are
            // the accepted prefix's KV and must survive this call. `keep >= m`
            // restores nothing, which the kernel reports as a successful no-op.
            if fused {
                self.dev.dspark_ring_restore(
                    ring,
                    snap,
                    base,
                    win as i32,
                    hd as i32,
                    m as i32,
                    keep as i32,
                )?;
            } else {
                for j in keep..m {
                    let slot = (pos_base + j) % win;
                    self.dev.memcpy_d2d(
                        ring.wrapping_add(slot * hd) as *mut c_void,
                        snap.wrapping_add(j * hd) as *const c_void,
                        hd * std::mem::size_of::<f32>(),
                    )?;
                }
            }
        }

        for &l in &self.compress_sources() {
            let ratio = cfg.compress_ratio(l).max(1);
            let cache = &self.layers[l];
            if fused {
                // The whole carry is restored regardless of `keep` (see the doc
                // comment above): one launch covers all three segments and both
                // counters.
                self.dev.dspark_comp_restore(
                    cache.state_kv.ptr as *mut f32,
                    cache.state_score.ptr as *mut f32,
                    cache.latent.ptr as *mut f32,
                    (self.s.clen.ptr as *mut i32).wrapping_add(l),
                    cache.out_rows.ptr as *mut i32,
                    (self.s.dspark_snap_state.ptr as *const f32)
                        .wrapping_add(l * 2 * max_ratio * hd),
                    (self.s.dspark_snap_latent.ptr as *const f32).wrapping_add(l * hd),
                    (self.s.dspark_snap_clen.ptr as *const i32).wrapping_add(l),
                    (self.s.dspark_snap_out_rows.ptr as *const i32).wrapping_add(l),
                    ratio as i32,
                    max_ratio as i32,
                    hd as i32,
                )?;
                continue;
            }
            let bytes = ratio * hd * std::mem::size_of::<f32>();
            let base = (self.s.dspark_snap_state.ptr as *const f32)
                .wrapping_add(l * 2 * max_ratio * hd);
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

    // ======================= the KV prefix snapshot (serve side) =============
    //
    // `dspark_snapshot`/`dspark_rollback` above undo a 6-ROW block inside one
    // step. These two freeze the WHOLE sequence's prefix state and put it back
    // at a later request: the mechanism is the same one (copy the KV out, copy it
    // back, restore the counters) at the sequence scale, which is what a prefix
    // cache needs. What is saved is the union of the shadow save's inventory and
    // two entries the shadow save deliberately omits:
    //
    //   * `index_k[0 .. clen * index_head_dim)` — the shadow save can skip it
    //     because it restores `clen` in the same breath and the next real commit
    //     overwrites the row at `clen - 1`, which is the only row the verify
    //     touched. A prefix snapshot is restored from an ARBITRARY later request,
    //     so those rows are the live read source of the continuing step
    //     (`indexer_topk` scans them) and must come back.
    //   * the engram n-gram table (`EngDev::cache`) — per-sequence device state
    //     that `reset` zeroes, feeding the layer 1/14 residual write-back, so a
    //     missing table makes a resumed run diverge from a from-scratch one on a
    //     long prompt.
    //
    // The snapshot is EXACT and self-contained: restoring it and stepping on
    // reproduces a from-scratch run bit for bit. It is therefore also taken
    // literally — `pos_ctr` is read back from the device rather than assumed, and
    // a caller that finds the recorded position disagreeing with its own
    // bookkeeping must not use the result.
}

/// One layer's share of a [`KvSnapshot`]. A field is empty when the layer does
/// not play that role, so a window-only layer costs its window and nothing more.
///
/// All the float payloads are raw bytes (not `Vec<f32>`) for the reason the
/// release will need: the ring is stored fp8 on the way there, and a byte copy
/// neither reinterprets nor converts — the same block can be handed to
/// `upload_bytes_at` whatever element type the device side grows.
#[derive(Debug, Clone, Default)]
pub struct LayerKvSnapshot {
    /// The window ring, `window * head_dim` f32 as raw bytes. Empty for a layer
    /// that is not a ring owner (a shared-ring consumer has no state of its own:
    /// `sparse_attn` reads its owner's buffer).
    pub ring: Vec<u8>,
    /// The committed compressed rows, `clen * head_dim` f32 as raw bytes — the
    /// device's `ring[window, window + clen)`, contiguous right after the window.
    /// Empty for a layer that runs no compressor.
    pub compress: Vec<u8>,
    /// The published pre-RoPE index keys, `clen * index_head_dim` f32 as raw
    /// bytes. Empty for a layer that publishes none.
    pub index_k: Vec<u8>,
    /// The compressor carry, `ratio * head_dim` f32 as raw bytes.
    pub state_kv: Vec<u8>,
    /// The compressor's score carry, same shape. Never zero-seeded: the device
    /// seeds empty slots with `-inf` (`reset`), so this block must round-trip
    /// whole or an empty slot starts looking like a real 0-weight entry.
    pub state_score: Vec<u8>,
    /// The compressor's pooled latent row, `head_dim` f32 as raw bytes. Not a
    /// scratch: a later step reads it to publish its index key when the group
    /// completes (`indexer`), so a wrong latent lands in `index_k[clen - 1]`.
    pub latent: Vec<u8>,
    /// The committed-row count — the HOST mirror `LayerCache::compress_len`,
    /// which the host branches on. Restored alongside the device counter.
    pub clen: usize,
}

/// A whole chain's prefix state at one position, held on the HOST (the P0 cache
/// tier). Taken by [`DevChain::kv_snapshot`], put back by [`DevChain::kv_restore`].
#[derive(Debug, Clone)]
pub struct KvSnapshot {
    /// The device position counter as read at snapshot time: the absolute
    /// position the state sits at, i.e. the number of tokens consumed.
    pub pos_ctr: i32,
    /// [`Self::pos_ctr`] as a length — what the caller resumes from. Recorded
    /// separately so a caller can see the two disagree instead of inferring it.
    pub tokens: usize,
    /// Per layer, indexed by layer id (length `n_layers`). MTP layers are not
    /// part of this: they are not ring owners (see [`DevChain::ring_owners`]).
    pub layers: Vec<LayerKvSnapshot>,
    /// The DEVICE committed-row counters, `n_layers` i32, written back as one
    /// block (`commit`/`sparse_attn`/`indexer_topk` all read them).
    pub clen_dev: Vec<i32>,
    /// The engram n-gram table, `max_seq` i64; `None` when the engram is off.
    pub eng_cache: Option<Vec<i64>>,
}

/// `gcd`, for [`DevChain::kv_page_align`].
fn gcd(a: usize, b: usize) -> usize {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

impl<'a> DevChain<'a> {
    /// How many compressed rows this layer's ring can hold. The allocation is
    /// `window + max_comp` rows (`DevChain::new`), so the ceiling is derivable
    /// from the buffer rather than stored a second time.
    fn layer_max_comp(&self, layer: usize) -> usize {
        let row = self.cfg.head_dim * std::mem::size_of::<f32>();
        (self.layers[layer].ring.bytes / row).saturating_sub(self.cfg.window_size)
    }

    /// The largest position `<= tokens` that is a boundary for EVERY compress
    /// source, i.e. a multiple of the lcm of their ratios. `tokens` itself when
    /// all ratios are 1.
    ///
    /// Why it exists: the compressor commits a group when `(pos + 1) % ratio == 0`
    /// and only then publishes the row, so a position off the boundary carries a
    /// HALF-accumulated group. That is still exact for the same prefix — the carry
    /// round-trips whole, which is why the serving cache (an exact full-prompt
    /// match, restored at the same position) does not need this. It becomes
    /// mandatory for the P1 radix tree, where a node's state is handed to a
    /// DIFFERENT continuation and a mid-group node could not keep
    /// `(pos + 1) % ratio` aligned with the position it is resumed at.
    ///
    /// A caller combining this with a block/page granularity must also make that
    /// granularity a multiple of the returned step (`page_size % ratio == 0` for
    /// every compress source), or the two alignments cannot both hold.
    pub fn kv_page_align(&self, tokens: usize) -> usize {
        let mut step = 1usize;
        for &l in &self.compress_sources() {
            let r = self.cfg.compress_ratio(l).max(1);
            step = step / gcd(step, r) * r;
        }
        tokens - tokens % step
    }

    /// Freeze the chain's prefix state into the host. Call BETWEEN steps: the
    /// snapshot describes the state after the last completed one, and a D2H is a
    /// device sync, so an in-flight step would be captured half-applied.
    ///
    /// Cost is linear in the position (see the module's `LayerKvSnapshot`), the
    /// window being the only constant part. The reads are batched: every ring
    /// owner's window goes through one staging buffer and one D2H, the carry
    /// through two more, and only the position-sized pieces (compressed rows,
    /// index keys) are copied per source layer.
    pub fn kv_snapshot(&self) -> Result<KvSnapshot> {
        let cfg = self.cfg;
        let hd = cfg.head_dim;
        let win = cfg.window_size;
        let ihd = cfg.index_head_dim.max(1);
        let max_ratio = cfg.compress_ratios.iter().copied().max().unwrap_or(1).max(1);
        let n_layers = cfg.n_layers;
        // The host n-gram path (`DSV41_ENG_HOST=1`) keeps its rolling hash on the
        // HOST side and this increment snapshots only the device table, so a
        // resume would start from a host hash that no longer matches the prefix.
        // Refuse rather than hand back a state that cannot be restored exactly —
        // the caller's contract is "byte-identical to a from-scratch run".
        if eng_host() && self.ngram.is_some() {
            return Err(FerriteError::Config(
                "kv_snapshot: DSV41_ENG_HOST=1 keeps the n-gram hash on the host, which this \
                 snapshot does not cover (device-hash mode is the supported one)"
                    .into(),
            ));
        }
        // Drain in-flight work: everything below reads completed state.
        self.dev.sync()?;
        let pos = self.dev.download_u32(self.s.pos_ctr.ptr as *const c_void)? as i32;

        // 1. The window ring of every owner: one D2D per layer into the staging
        //    buffer, then ONE D2H for all of them.
        let owners = self.ring_owners();
        let astep = win * hd * std::mem::size_of::<f32>();
        for &l in &owners {
            self.dev.memcpy_d2d(
                (self.s.kv_snap_ring.ptr as *mut f32).wrapping_add(l * win * hd) as *mut c_void,
                self.layers[l].ring.ptr as *const c_void,
                astep,
            )?;
        }
        let mut window = vec![0u8; n_layers * astep];
        self.dev.download_u8(&self.s.kv_snap_ring, &mut window)?;

        let mut layers = vec![LayerKvSnapshot::default(); n_layers];
        for &l in &owners {
            layers[l].ring = window[l * astep..(l + 1) * astep].to_vec();
        }

        // 2. The compressor carry of every source: staged and downloaded like the
        //    window (state_kv then state_score at a max-ratio stride).
        let sources = self.compress_sources();
        let cstep = max_ratio * hd * std::mem::size_of::<f32>();
        for &l in &sources {
            let cache = &self.layers[l];
            let bytes = cfg.compress_ratio(l).max(1) * hd * std::mem::size_of::<f32>();
            let base = (self.s.kv_snap_state.ptr as *mut f32).wrapping_add(l * 2 * max_ratio * hd);
            self.dev
                .memcpy_d2d(base as *mut c_void, cache.state_kv.ptr as *const c_void, bytes)?;
            self.dev.memcpy_d2d(
                base.wrapping_add(max_ratio * hd) as *mut c_void,
                cache.state_score.ptr as *const c_void,
                bytes,
            )?;
            self.dev.memcpy_d2d(
                (self.s.kv_snap_latent.ptr as *mut f32).wrapping_add(l * hd) as *mut c_void,
                cache.latent.ptr as *const c_void,
                hd * std::mem::size_of::<f32>(),
            )?;
        }
        let mut carry = vec![0u8; n_layers * 2 * cstep];
        self.dev.download_u8(&self.s.kv_snap_state, &mut carry)?;
        let mut latents = vec![0u8; n_layers * hd * std::mem::size_of::<f32>()];
        self.dev.download_u8(&self.s.kv_snap_latent, &mut latents)?;

        // 3. The DEVICE committed-row counters FIRST (one small read): they are
        //    the authority for how much of the ring's compressed region is live,
        //    so the position-sized copies below are sized from them and not from
        //    the host mirror (which is a mirror: it tracks the same rule, but the
        //    buffer's content is what a reader actually sees).
        let mut clen_b = vec![0u8; n_layers * 4];
        self.dev
            .download_u8(&Device::view(self.s.clen.ptr, n_layers * 4), &mut clen_b)?;
        let clen_dev: Vec<i32> = clen_b
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        // 4. The position-sized pieces, one D2H each (a handful of layers).
        for &l in &sources {
            let cache = &self.layers[l];
            let ratio = cfg.compress_ratio(l).max(1);
            let clen = (clen_dev[l].max(0) as usize).min(self.layer_max_comp(l)).min(
                // a `clen` past what the host mirror claims cannot be indexed:
                // `indexer` publishes at `*clen - 1`, so the two must agree for a
                // key to exist. Clipping to the smaller keeps the snapshot
                // self-consistent rather than half-stale.
                cache.compress_len,
            );
            let cb = ratio * hd * std::mem::size_of::<f32>();
            let base = l * 2 * cstep;
            // The HOST mirror, snapshotted as-is (it is restored as-is): the two
            // counters are separate facts and neither is derived on the way back.
            layers[l].clen = cache.compress_len;
            layers[l].state_kv = carry[base..base + cb].to_vec();
            layers[l].state_score = carry[base + cstep..base + cstep + cb].to_vec();
            layers[l].latent = latents[l * hd * 4..(l + 1) * hd * 4].to_vec();
            if clen == 0 {
                continue;
            }
            let rows = clen * hd * std::mem::size_of::<f32>();
            let view = Device::view(
                (cache.ring.ptr as *mut f32).wrapping_add(win * hd) as *mut c_void,
                rows,
            );
            let mut b = vec![0u8; rows];
            self.dev.download_u8(&view, &mut b)?;
            layers[l].compress = b;
            // Only a publishing layer has keys to save (`indexer_owns_k` =
            // `is_kv_source`; a consumer's own `index_k` is never written).
            if cfg.is_index_source(l) {
                let kb = clen * ihd * std::mem::size_of::<f32>();
                let view = Device::view(cache.index_k.ptr, kb);
                let mut b = vec![0u8; kb];
                self.dev.download_u8(&view, &mut b)?;
                layers[l].index_k = b;
            }
        }

        // 5. The engram's n-gram table.
        let eng_cache = match self.eng_dev.as_ref() {
            Some(e) => {
                let mut b = vec![0u8; e.max_seq * 8];
                self.dev.download_u8(&e.cache, &mut b)?;
                Some(
                    b.chunks_exact(8)
                        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                        .collect(),
                )
            }
            None => None,
        };

        Ok(KvSnapshot {
            pos_ctr: pos,
            tokens: pos.max(0) as usize,
            layers,
            clen_dev,
            eng_cache,
        })
    }

    /// Put a [`DevChain::kv_snapshot`] back. The caller then continues with
    /// `step()` from `snap.tokens` (the tail of the prompt, if any) — the same
    /// position discipline a from-scratch run has.
    ///
    /// Two side effects the caller must not undo: the captured step graph is
    /// DROPPED and `decode_steps` reset to 0, for `reset`'s two reasons (the graph
    /// baked the previous request's device addresses and its host branch choices);
    /// and the device position counter is moved LAST, once everything it points
    /// at is in place.
    pub fn kv_restore(&mut self, snap: &KvSnapshot) -> Result<()> {
        let cfg = self.cfg;
        let hd = cfg.head_dim;
        let win = cfg.window_size;
        let ihd = cfg.index_head_dim.max(1);
        let max_ratio = cfg.compress_ratios.iter().copied().max().unwrap_or(1).max(1);
        let n_layers = cfg.n_layers;
        let bad = |m: String| FerriteError::InvalidArg(format!("kv_restore: {m}"));
        if snap.layers.len() < n_layers || snap.clen_dev.len() < n_layers {
            return Err(bad(format!(
                "snapshot covers {} layers / {} counters, the chain has {n_layers}",
                snap.layers.len(),
                snap.clen_dev.len()
            )));
        }

        // 1. Drop the graph first: it would otherwise be replayed against state
        //    (and addresses) it was not recorded from.
        if let Some(e) = self.step_graph.take() {
            self.dev.graph_free(std::ptr::null_mut(), e)?;
        }
        self.decode_steps = 0;

        // 2. The window ring: one H2D into the staging buffer, one D2D per owner.
        let owners = self.ring_owners();
        let astep = win * hd * std::mem::size_of::<f32>();
        let mut window = vec![0u8; n_layers * astep];
        for &l in &owners {
            let r = &snap.layers[l].ring;
            if r.len() != astep {
                return Err(bad(format!(
                    "layer {l} window is {} bytes, expected {astep}",
                    r.len()
                )));
            }
            window[l * astep..(l + 1) * astep].copy_from_slice(r);
        }
        self.dev.upload_bytes_at(&self.s.kv_snap_ring, &window)?;
        for &l in &owners {
            self.dev.memcpy_d2d(
                self.layers[l].ring.ptr,
                (self.s.kv_snap_ring.ptr as *const f32).wrapping_add(l * win * hd) as *const c_void,
                astep,
            )?;
        }

        // 3. The compressor carry, staged the same way, then the position-sized
        //    pieces per source (compressed rows, index keys) and the host mirror.
        let sources = self.compress_sources();
        let cstep = max_ratio * hd * std::mem::size_of::<f32>();
        let mut carry = vec![0u8; n_layers * 2 * cstep];
        let mut latents = vec![0u8; n_layers * hd * 4];
        for &l in &sources {
            let ratio = cfg.compress_ratio(l).max(1);
            let cb = ratio * hd * std::mem::size_of::<f32>();
            let li = &snap.layers[l];
            if li.state_kv.len() != cb || li.state_score.len() != cb || li.latent.len() != hd * 4 {
                return Err(bad(format!("layer {l} carry is not {cb}/{cb}/{} bytes", hd * 4)));
            }
            let base = l * 2 * cstep;
            carry[base..base + cb].copy_from_slice(&li.state_kv);
            carry[base + cstep..base + cstep + cb].copy_from_slice(&li.state_score);
            latents[l * hd * 4..(l + 1) * hd * 4].copy_from_slice(&li.latent);
        }
        self.dev.upload_bytes_at(&self.s.kv_snap_state, &carry)?;
        self.dev.upload_bytes_at(&self.s.kv_snap_latent, &latents)?;

        for &l in &sources {
            let ratio = cfg.compress_ratio(l).max(1);
            let cb = ratio * hd * std::mem::size_of::<f32>();
            let max_comp = self.layer_max_comp(l);
            let li = &snap.layers[l];
            // The row count comes from the DEVICE counter — the same authority the
            // snapshot used, and the one a reader honours. The host mirror goes
            // back unchanged but sizes nothing.
            let clen = (snap.clen_dev[l].max(0) as usize).min(max_comp);
            {
                let cache = &self.layers[l];
                let base = (self.s.kv_snap_state.ptr as *const f32)
                    .wrapping_add(l * 2 * max_ratio * hd);
                self.dev
                    .memcpy_d2d(cache.state_kv.ptr, base as *const c_void, cb)?;
                self.dev.memcpy_d2d(
                    cache.state_score.ptr,
                    base.wrapping_add(max_ratio * hd) as *const c_void,
                    cb,
                )?;
                self.dev.memcpy_d2d(
                    cache.latent.ptr,
                    (self.s.kv_snap_latent.ptr as *const f32).wrapping_add(l * hd) as *const c_void,
                    hd * 4,
                )?;
                if clen > 0 {
                    if li.compress.len() != clen * hd * 4 {
                        return Err(bad(format!(
                            "layer {l} has {} compressed-row bytes, expected {}",
                            li.compress.len(),
                            clen * hd * 4
                        )));
                    }
                    let view = Device::view(
                        (cache.ring.ptr as *mut f32).wrapping_add(win * hd) as *mut c_void,
                        clen * hd * 4,
                    );
                    self.dev.upload_bytes_at(&view, &li.compress)?;
                    if cfg.is_index_source(l) && !li.index_k.is_empty() {
                        if li.index_k.len() != clen * ihd * 4 {
                            return Err(bad(format!(
                                "layer {l} has {} index-key bytes, expected {}",
                                li.index_k.len(),
                                clen * ihd * 4
                            )));
                        }
                        let view = Device::view(cache.index_k.ptr, clen * ihd * 4);
                        self.dev.upload_bytes_at(&view, &li.index_k)?;
                    }
                }
            }
            // The host mirror last, so `clen`-gated host branches read the value
            // that goes with the buffers just written.
            self.layers[l].compress_len = li.clen;
        }

        // 4. The device counters as one block. A non-source layer cannot have
        //    committed rows — the commit kernel runs from the compressor, which only
        //    a source executes — so a non-zero counter there would mean the snapshot
        //    describes a configuration this restore does not put back. Refuse
        //    instead of leaving rows a reader would trust.
        for l in 0..n_layers {
            if snap.clen_dev[l] != 0 && !sources.contains(&l) {
                return Err(bad(format!(
                    "layer {l} reports {} committed rows but is not a compress source",
                    snap.clen_dev[l]
                )));
            }
        }
        let mut clen_b = Vec::with_capacity(n_layers * 4);
        for v in &snap.clen_dev[..n_layers] {
            clen_b.extend_from_slice(&v.to_le_bytes());
        }
        self.dev
            .upload_bytes_at(&Device::view(self.s.clen.ptr, n_layers * 4), &clen_b)?;

        // 5. The engram table. A snapshot without one means the table must be
        //    ZEROED, exactly as `reset` leaves it — a stale table would make the
        //    resumed run differ from a from-scratch one.
        match (self.eng_dev.as_ref(), snap.eng_cache.as_ref()) {
            (Some(e), Some(v)) => {
                if v.len() != e.max_seq {
                    return Err(bad(format!(
                        "engram table is {} entries, expected {}",
                        v.len(),
                        e.max_seq
                    )));
                }
                let mut b = Vec::with_capacity(v.len() * 8);
                for x in v {
                    b.extend_from_slice(&x.to_le_bytes());
                }
                self.dev.upload_bytes_at(&e.cache, &b)?;
            }
            (Some(e), None) => self.dev.zero_at(e.cache.ptr, e.max_seq * 8)?,
            (None, _) => {}
        }

        // 6. The position LAST (see the doc comment).
        self.set_pos_ctr(snap.tokens)?;
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
    /// * `accepted = 1 + <accepted draft prefix length>` (1..=DSPARK_DRAFTS):
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
        // The verify block is [d1..d5] — 5 rows (t0's forward already happened
        // in the single-row step).
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
            // The block is `[d1..d5]` and the step just advanced the counter to
            // `pos + 1`, so the block's row 0 sits there.
            self.dspark_snapshot(pos_ctr + 1, m)?
        };

        // 3./4./5. the draft, from the tap of the step that just ran. `pos` is
        // the backbone token's position, the same convention `step_dev` uses and
        // the same one the host reference's `forward_spec(.., start_pos)` uses.
        let t = std::time::Instant::now();
        dspark.import_tap(self.s.dspark_tap.ptr as *const f32)?;
        // The official model.py semantics (settled by the unit diff): the block
        // is [embed(t0), noise×4] — the JUST-CONSUMED token's embedding, placed
        // at the NEXT position's viewpoint (RoPE at start_pos + seqlen + r with
        // seqlen = 1, i.e. pos+1+r; the seed window row goes to slot
        // start_pos % win = pos). The intermediate anchor=bonus reading was the
        // sglang/DeepSpec convention, which does NOT match this checkpoint's
        // reference — the unit harness caught it as a 100% q divergence.
        let drafts = if bisect >= 2 {
            [token; DSPARK_DRAFTS]
        } else {
            // The block sits at the anchor's OWN position (row r = pos+r): the
            // accepted layout is the 5-row verify block [d1..d5] @ pos+1..pos+5
            // — row j fed drafts[j] lands at pos+1+j, so its argmax is the
            // pos+2+j prediction and drafts[j+1] (proposed for pos+2+j) is its
            // judge. This matches the shadow path, the parity oracle
            // (dspark_parity.rs:380) and the free-first-check `drafts[0]==next`.
            // The `pos + 1` form (a half-applied viewpoint experiment) fed row j
            // a token belonging to pos+2+j → a one-token GAP at pos+1 in every
            // row's context → the model's natural "fill the gap" output == next
            // → verify_out[0]==next echo + the per-1-2-char repetitions.
            dspark.draft_forward(token, pos)?;
            dspark.drafts()?
        };
        let draft_ms = t.elapsed().as_secs_f32() * 1e3;

        // 6. the verify block: one m-row forward, per-row argmax. The block is
        //    [d1..d5] at pos+1..pos+5 (t0 was already forwarded by the single-row
        //    step). Everything it appends, step 8 undoes.
        let t = std::time::Instant::now();
        let rows = if bisect == 1 || bisect == 3 {
            Vec::new()
        } else {
            self.step_rows(&drafts)?
        };
        let verify_ms = t.elapsed().as_secs_f32() * 1e3;

        // 7. rollback, immediately after the verify and BEFORE the host-only
        //    arithmetic below: the chain must not stay dirty on a report-building
        //    error path. (`pos_ctr + 1`: the block's row 0 — see the snapshot.)
        self.dspark_rollback(pos_ctr + 1, m, &host_mirrors)?;

        let mut verify_out = [0u32; DSPARK_DRAFTS];
        if bisect == 0 && rows.len() != DSPARK_DRAFTS {
            return Err(FerriteError::Config(format!(
                "dspark_shadow_step: step_rows returned {} rows for a {}-row \
                 verify block",
                rows.len(),
                DSPARK_DRAFTS
            )));
        }
        if rows.len() == DSPARK_DRAFTS {
            verify_out.copy_from_slice(&rows);
        }

        // 8. the accept arithmetic (host, no device traffic). MEASURED layout:
        //    the draft block is at the anchor's own position, so drafts[j]
        //    predicts pos+1+j; verify row j-1's argmax is at the same pos+1+j.
        //    The chain: drafts[0] vs `next` (the single-row step's argmax), then
        //    drafts[j] vs verify_out[j-1].
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

        // 9. the golden-comparison dump (no-op unless `DSV41_DSPARK_DUMP=1`).
        //    Last, and after the rollback: the chain is already back where the
        //    single-row path left it, so nothing the dump does can leak into the
        //    real decode. `k_acc` is the MATCHING draft prefix (`acc`), not the
        //    report's `accepted` (which adds the always-present anchor) — the
        //    spec path's `k_acc` is the same quantity, so the two diff cleanly.
        self.dspark_dump_step("shadow", pos, token, next, acc, &drafts, &verify_out);

        // 10. the verify-row-0 vs eager parity probe (DSV41_VROW0_PROBE=1). LAST,
        //     for the same reason the dump is: the eager forward below re-runs the
        //     step's TAP HOOK, so anything that reads `dspark_tap` (step 9) must
        //     have run first. The chain is back where the real step left it (step
        //     7), which is the one state an eager single-row forward at pos+1 can
        //     be compared from; everything that forward writes — the ring row, the
        //     compressor carry, `pos_ctr`, `s.ids` — is restored by the probe
        //     itself before it returns, so the shadow step's contract (leave the
        //     chain where the REAL step left it) still holds. Logged, never
        //     propagated: a probe must not answer an error for a step whose
        //     numbers are already fixed.
        if vrow0_probe() {
            if let Err(e) =
                self.vrow0_step("shadow", pos, token, next, drafts[0], verify_out[0], true)
            {
                eprintln!("[dsv41] vrow0 probe (shadow, pos {pos}) failed: {e}");
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

    /// REAL-COMMIT DSpark step (`DSV41_SPEC=1`): the same draft + verify block as
    /// [`Self::dspark_shadow_step`], but the accepted prefix is KEPT instead of
    /// rolled back — the engine's state moves forward by `k_acc + 1` positions.
    ///
    /// # Timeline
    ///
    /// ```text
    ///   1. next       = step_dev(token, pos)   the anchor's producer (the ONLY
    ///                                          main-chain forward this step runs)
    ///   2. snapshot    = dspark_snapshot(pos, 6)   the verify's write set, taken
    ///                                          AFTER the step (see the shadow
    ///                                          step's note on the ordering)
    ///   3. draft        dspark.draft_forward(next, pos + 1) -> d1..d5
    ///   4. verify_out = step_rows([next, d1..d5])   six rows at pos+1 .. pos+6
    ///   5. accept       k_acc = longest prefix with drafts[i] == verify_out[i]
    ///   6. commit       dspark_commit(pos, 6, k_acc, snapshot): rows 0..k_acc of
    ///                   the block survive, the rest is rolled back, the
    ///                   compressor is replayed for the survivors, and the
    ///                   position counter jumps to pos + k_acc + 1
    ///   7. emit         [next] ++ verify_out[0..k_acc]   (k_acc + 1 tokens)
    /// ```
    ///
    /// # Why the emitted tokens are the VERIFY's argmax
    ///
    /// `verify_out[j]` is the target's argmax at position `pos + 2 + j` computed
    /// from a context whose rows 0..j are the accepted tokens — i.e. exactly what
    /// a greedy single-row decode would emit there. `drafts[j]` agrees with it
    /// whenever it was accepted, so emitting `verify_out` rather than the raw
    /// drafts keeps the token stream bit-identical to the non-speculative one
    /// even at the boundary (`j == k_acc`, where the draft disagreed).
    ///
    /// # What the caller must do with `emitted`
    ///
    /// Feed every token to the driver (stop tokens included), advance its own
    /// position by `emitted.len()`, and use the LAST one as the next step's
    /// input. The engine's `pos_ctr` is already at that token's position and its
    /// KV is deliberately absent — the next step's `step_dev` appends it, exactly
    /// as it would for a plain decode step.
    ///
    /// # Failure
    ///
    /// The block is rolled back before ANY error is returned, so a failed step
    /// leaves the chain exactly where the real step left it (the caller may then
    /// fall back to `step_dev`, or answer the error).
    pub fn dspark_spec_step(
        &mut self,
        dspark: &mut DsparkDev,
        token: u32,
        pos: usize,
    ) -> Result<DsparkSpecReport> {
        let cfg = self.cfg;
        // The verify block is [d1..d5] — 5 rows, ONE batched forward (the weights
        // are read ONCE for the whole block — that is the whole point of spec
        // decoding; per-row eager calls would read them 5 times).
        let m = DSPARK_DRAFTS;
        if !cfg.dspark_armed() {
            return Err(FerriteError::Config(
                "dspark_spec_step: the chain's tap hook is off (set DSV41_DSPARK and make sure \
                 it is set before the first decoded step captures the step graph); without it the \
                 draft would read a stale target hidden"
                    .into(),
            ));
        }

        let pos_ctr = self.dev.download_u32(self.s.pos_ctr.ptr as *const c_void)? as usize;
        debug_assert_eq!(
            pos_ctr, pos,
            "dspark_spec_step: `pos` must be the device position counter's current value"
        );
        // (2) the common entry of all three arms: the device counter the driver is
        // about to step from must be the `p` it believes it is at. Reuses the D2H
        // above, so it adds no device traffic (see `inv_pos_ctr`).
        self.inv_pos_ctr(pos, pos_ctr)?;

        // A chain that has not yet bootstrapped, or a process without the gate,
        // takes the legacy 5-row block below. With the gate on, the chain is
        // primed by the round that ran the legacy path (its `step_dev` is what
        // writes the very first tap) and every later round swallows the
        // main-chain step into the verify's anchor row.
        //
        // ★ THE ARM IS A UNANIMOUS DECISION, NOT A PER-RANK ONE (`ar5-hang`'s
        // root cause — see [`Self::spec_primed_unanimous`]): `spec_primed` is a
        // per-rank flag but the arm's AR ROUND FOOTPRINT is process-wide, so one
        // rank that lost its priming must send the WHOLE world back to the
        // legacy path for this round. Otherwise that rank issues ~80 more v5
        // rounds per step than its peers and spins on their stamps forever (the
        // permanent epoch rift), while the peers — who issued FEWER — silently
        // read its previous epoch's values. The legacy path re-primes every rank
        // at its tail, so the very next round is unanimous again.
        //
        // The vote comes FIRST and is taken unconditionally: it is a symmetric
        // rendezvous, so evaluating the cheap `self.spec_primed` before it would
        // leave the primed ranks spinning alone in the vote (one hang for
        // another). With the gate off both terms short-circuit on
        // `swallow_step()` and nothing is issued — the default path is
        // bit-for-bit what it was. The trailing `self.spec_primed` is what keeps
        // the NO-PEER case identical to the old expression: without a `comm` the
        // helper answers `true` and the local flag alone decides.
        if swallow_step() && self.spec_primed_unanimous() && self.spec_primed {
            let rep = self.dspark_spec_swallowed(dspark, token, pos)?;
            lazy_b_ms_note(rep.verify_ms);
            return Ok(rep);
        }
        // ---- `DSV41_LAZY_VERIFY`: the same block, one ROW at a time ----
        //
        // The gate IMPLIES swallow (see [`lazy_verify`]): the lazy arm's row 0 is
        // the swallowed main-chain step, so a lazy round still pays no `step_dev`.
        // Inside that block the route picks per round between the row loop and the
        // single 6-row verify — the SAME choice on every rank, because it is a
        // function of the integer mean-k window and the process-level `τ`
        // (`lazy-batched-route.md` §4.3). The window is fed at this common exit,
        // so both arms update it and the route can see accept drift.
        if lazy_verify() && self.spec_primed {
            let use_lazy = self.lazy_route_decide();
            let rep = if use_lazy {
                self.dspark_spec_lazy(dspark, token, pos)?
            } else {
                let rep = self.dspark_spec_swallowed(dspark, token, pos)?;
                lazy_b_ms_note(rep.verify_ms);
                rep
            };
            self.lazy_hist.push(rep.k_acc);
            return Ok(rep);
        }
        // The seed↔tap alignment (`DSV41_SEED_ALIGN`) moves this arm's draft
        // block onto `pos + 1` and widens the verify to 6 rows — a different
        // block layout, so it is its own arm (see `dspark_spec_aligned`), NOT a
        // branch inside the legacy one: the A/B must keep the legacy arm
        // bit-identical.
        //
        // The gate is armed only AFTER one legacy round has run (`spec_primed`,
        // the same bootstrap the swallowed arm uses): the two arms must issue the
        // SAME number of all-reduce collectives per round, because AR v5 has no
        // host-side rendezvous — a rank that issues FEWER silently reads the
        // previous epoch's values for the missing ones, and a rank that issues
        // MORE spins forever. The one place the arms' AR footprints diverge is
        // the FIRST round: the legacy arm calls `draft_forward(token, pos)`, and
        // when that first round sits at `pos == 0` the call takes
        // `draft_forward`'s prefill early-exit (0 draft ARs — the window is only
        // seeded), while the aligned arm calls `draft_forward(next, pos + 1)`
        // with `pos + 1 >= 1` and always runs all three mtp blocks (3 ARs). The
        // gap is exactly `need - cur = 3` = one MoE all-reduce per mtp block.
        // Priming closes it: round 1 runs this legacy path on BOTH arms (so both
        // take the same early-exit), from round 2 on `pos >= 1` on both and the
        // three blocks run on both. See docs/agent/dspark-correctness-chain.md
        // ("AR v5 死锁", H1/F1).
        if seed_align() && self.spec_primed {
            return self.dspark_spec_aligned(dspark, token, pos);
        }

        // ---- 1. the real step: the whole-step graph (tap hook included), the
        // argmax, and the position counter + 1.
        let next = self.step_dev(token, pos)?;

        // ---- 2. the snapshot, AFTER the step and BEFORE the verify: the
        // rollback must restore the state the REAL step left. The block is
        // `[d1..d5]` and `step_dev` advanced the counter to `pos_ctr + 1`, so
        // the block's row 0 sits at `pos_ctr + 1` — one past the step's own
        // position (the swallowed branch's block starts AT `pos` instead).
        let host_mirrors = self.dspark_snapshot(pos_ctr + 1, m)?;

        // ---- 3. the draft, from the tap of the step that just ran.
        let t = std::time::Instant::now();
        dspark.import_tap(self.s.dspark_tap.ptr as *const f32)?;
        // Official model.py semantics: the block is [embed(t0), noise×4] at the
        // NEXT position's viewpoint (RoPE pos+1+r; the seed window row goes to
        // slot pos%win).
        // The block sits at the anchor's OWN position — the accepted 5-row
        // layout (see the shadow path's comment): drafts[j] is the proposal for
        // pos+1+j, verify row j (fed it at pos+1+j) judges it, so the accept
        // chain is drafts[0] vs `next` then drafts[j] vs verify_out[j-1].
        dspark.draft_forward(token, pos)?;
        let drafts = dspark.drafts()?;
        let draft_ms = t.elapsed().as_secs_f32() * 1e3;

        // ---- 4. the verify: [d1..d5] at pos+1..pos+5, ONE batched forward,
        // per-row argmax. The rows' per-row outputs must be BIT-IDENTICAL to
        // the plain engine at the same positions — that parity is the spec
        // path's correctness contract, and it is what the unit diff enforces.
        let t = std::time::Instant::now();
        self.spec_capture = true;
        let rows_res = self.step_rows(&drafts);
        self.spec_capture = false;
        let rows = match rows_res {
            Ok(r) => r,
            Err(e) => {
                let _ = self.dspark_rollback(pos_ctr, m, &host_mirrors);
                return Err(e);
            }
        };
        let verify_ms = t.elapsed().as_secs_f32() * 1e3;
        if rows.len() != DSPARK_DRAFTS {
            self.dspark_rollback(pos_ctr, m, &host_mirrors)?;
            return Err(FerriteError::Config(format!(
                "dspark_spec_step: step_rows returned {} rows for a {}-row verify block",
                rows.len(),
                DSPARK_DRAFTS
            )));
        }
        let mut verify_out = [0u32; DSPARK_DRAFTS];
        verify_out.copy_from_slice(&rows);

        // ---- 4b. the verify-row-0 parity probe (DSV41_VROW0_PROBE=1). VERIFY
        // SIDE ONLY: the block below is about to be COMMITTED, and a probe may
        // not undo a commit to go and run an eager forward — so the spec path
        // reports the row 0 logits/argmax and no eager half (`"eager":null`),
        // while the shadow path reports both. Read-only (the `logits_r` row is
        // this verify's own scratch) and gated: nothing here runs when the gate
        // is off. Logged, never propagated — a probe must not answer an error
        // for a step whose block is already committed.
        if vrow0_probe() {
            if let Err(e) =
                self.vrow0_step("spec", pos, token, next, drafts[0], verify_out[0], false)
            {
                eprintln!("[dsv41] vrow0 probe (spec, pos {pos}) failed: {e}");
            }
        }

        // ---- 5. the accept arithmetic (host, no device traffic). MEASURED
        // layout (154-step trace): drafts[0] == next 60% — the draft block sits
        // at the anchor's OWN position, so drafts[j] predicts pos+1+j and
        // verify row j-1 (fed drafts[j-1] at pos+1+j-1) has its argmax at the
        // SAME pos+1+j. The chain is therefore drafts[0] against `next` (the
        // single-row step's argmax, the pos+1 token) and then drafts[j] against
        // verify_out[j-1]. This is the historical chain; an "index-for-index
        // drafts[j]==verify_out[j]" variant was tried and measured 0.000 —
        // cross-position — confirming the draft block is at pos, not pos+1.
        //
        // The chain itself is the SHARED one ([`ferrite_types::spec_accept`],
        // bound here through `SpecStep::ANCHOR_IS_IN_BLOCK = false`): because
        // this block does NOT contain the anchor row, the anchor's judge `next`
        // has to lead the judge array and the block rows shift by one — the
        // block's LAST row's argmax is the bonus token, so it judges no draft.
        let mut judges = [0u32; DSPARK_DRAFTS];
        judges[0] = next;
        judges[1..].copy_from_slice(&verify_out[..DSPARK_DRAFTS - 1]);
        let k_acc = Self::accept(&drafts, &judges);

        // ---- 6. the commit: roll back everything past the accepted prefix,
        // replay the kept rows through the compressors, advance the counter.
        // (The block's row 0 is at `pos_ctr + 1` — see the snapshot above.)
        let t = std::time::Instant::now();
        self.dspark_commit(pos_ctr + 1, m, k_acc, &host_mirrors)?;
        dspark.note_ctx_rows(self.s.dspark_tap_r.ptr as *const f32, m, k_acc, pos + 1)?;
        // ---- 6b. and hand the NEXT round's draft its tap
        // (`DSV41_SWALLOW_STEP` only): the swallow drops `step_dev`, so this
        // round's tap in `dspark_tap` is the last one a single-row forward will
        // write. See `carry_kept_tap` for which row of the block that is.
        if swallow_step() {
            Self::carry_kept_tap(
                self.dev,
                self.s.dspark_tap.ptr,
                self.s.dspark_tap_r.ptr as *const c_void,
                cfg.dim,
                k_acc,
            )?;
        }
        let commit_ms = t.elapsed().as_secs_f32() * 1e3;

        // ---- 7. what this step emits: the anchor plus the verify's argmax for
        // every position the draft got right.
        let mut emitted = Vec::with_capacity(k_acc + 1);
        emitted.push(next);
        emitted.extend_from_slice(&verify_out[..k_acc]);
        // `emitted[i]` is the token at `pos + 1 + i`: `next` = `step_dev`'s argmax
        // at `pos + 1`, `verify_out[j]` = block row `j`'s argmax, and row `j` was
        // fed `drafts[j]` at `pos + 1 + j`, so it is the token at `pos + 2 + j`.
        // `emitted.last()` is therefore the token at `pos + k_acc + 1` — exactly
        // the counter the commit above just wrote (`pos_base = pos_ctr + 1`,
        // `keep = k_acc`) and exactly the token the NEXT round embeds. See
        // [`Self::sids_writeback`] for this invariant across the three arms.
        // ★ THE s.ids WRITE-BACK (verify-row0-systematic's root cause).
        // step_body's embedding reads `s.ids` (the `token` arg is only the
        // engram fallback), and NOTHING in the spec path updates it: the main
        // argmax wrote `next` (pos+1's token), but commit advanced the counter
        // by 1+k_acc — so after ANY k_acc>=1 round the next round would embed a
        // token k_acc positions STALE, silently corrupting the context. On the
        // digit task that self-locks into emitting every number twice (the
        // first polluted round makes the model treat repetition as the pattern
        // to continue — which is why the first ~10 chars were right and then it
        // collapsed, while emitted itself stayed item-by-item correct).
        if sids_writeback() {
            if let Some(&last) = emitted.last() {
                self.ul_i32(self.s.ids.ptr, &[last as i32])?;
            }
        }

        self.dspark_dump_step("spec", pos, token, next, k_acc, &drafts, &verify_out);
        // (1) the arms' common EXIT invariant: the round left `s.ids` holding the
        // token the NEXT round embeds. Runs after the write-back + dump, and
        // before `spec_primed` is touched, so a failure unwinds with the flag
        // alone (the caller rolls the block back).
        self.inv_ids(pos, &emitted)?;

        // The chain is now bootstrapped: with the swallowed arm's gate on, the
        // next round's 6-row block carries the anchor's forward; with the aligned
        // arm's gate on, the next round may finally take the 6-row aligned block
        // (see the gate above — the FIRST round had to run THIS path on both
        // arms). Set LAST, so a round that failed above leaves this flag alone
        // and the next round re-runs the legacy path (whose `step_dev`
        // re-supplies a valid tap).
        // (lazy-route-debug's verdict: `lazy_verify()` was MISSING from this
        // disjunction — with only LAZY_VERIFY set (SWALLOW/SEED_ALIGN both off),
        // spec_primed stayed false FOREVER, so dspark_spec_lazy was never entered
        // and every round ran the legacy 5-row path (verify 37.95 ≈ batched 38.34,
        // text identical because it WAS the same code path). The lazy arm needs the
        // same bootstrap: its row 0 is the swallowed main-chain step, so the FIRST
        // round's step_dev is what supplies the very first tap.)
        if seed_align() || swallow_step() || lazy_verify() {
            self.spec_primed = true;
        }

        Ok(DsparkSpecReport {
            next,
            drafts,
            verify_out,
            k_acc,
            emitted,
            draft_ms,
            verify_ms,
            commit_ms,
        })
    }

    /// The SEED-ALIGNED arm of [`Self::dspark_spec_step`] (`DSV41_SEED_ALIGN=1`).
    ///
    /// The same four phases as the legacy arm, with the draft block moved onto
    /// the position the imported tap actually belongs to. See [`seed_align`] for
    /// the defect and for why `seed_window(s, pos - 1)` needs no change at all.
    ///
    /// # Timeline
    ///
    /// ```text
    ///   1. next       = step_dev(token, pos)      the anchor's forward — the tap
    ///                                            is the hidden of THIS forward,
    ///                                            i.e. of the token at `pos`
    ///   2. snapshot    dspark_snapshot(pos + 1, 6)  the block's write set
    ///   3. draft        dspark.draft_forward(next, pos + 1) -> d1..d5
    ///                   the seed row lands at (pos + 1) - 1 = pos — the tap's
    ///                   position; the block's rows sit at pos + 1 .. pos + 5
    ///   4. verify_out = step_rows([next, d1..d5])   six rows at pos+1 .. pos+6
    ///   5. accept       k_emit = spec_accept(drafts, verify_out, true)
    ///   6. commit       dspark_commit(pos + 1, 6, k_acc): rows 0..k_acc survive,
    ///                   the counter lands on pos + k_acc + 1
    ///   7. emit         [next] ++ verify_out[0..k_acc]   (k_acc + 1 tokens)
    /// ```
    ///
    /// # The accept arithmetic — the SAME chain GLM's MTP runs
    ///
    /// Block row `i` is fed `[next, d1..d5][i]` at position `pos + 1 + i`, so its
    /// argmax predicts `pos + 2 + i`; the draft's `drafts[i]` is its proposal for
    /// `pos + 2 + i` too (the draft block sits at the anchor's OWN position,
    /// `pos + 1`). The two are therefore INDEX-ALIGNED, which is GLM's
    /// `[t_last, d1..d_nd]` layout (`TpCluster::mtp_step`:
    /// `while drafts[k-1] == out[k-1]`), so the shared
    /// [`ferrite_types::spec_accept`] runs with `anchor_is_in_block = true` and
    /// returns the number of tokens the step EMITS (`1..=6`) — the always-emitted
    /// anchor plus the surviving drafts. The legacy arm's 5-row block has no
    /// anchor row, which is why ITS chain (and `SpecStep::ANCHOR_IS_IN_BLOCK`)
    /// stays the shifted one: the two layouts are one chain apart by the anchor.
    /// This is the same accept binding [`Self::dspark_spec_swallowed`] uses, and
    /// the two blocks differ only by their ROW-0 POSITION.
    ///
    /// The report keeps the legacy shapes: `next` is `step_dev`'s argmax (the
    /// token at `pos + 1`, the first emitted), `verify_out` the five argmaxes
    /// AFTER the anchor row (row `i`, fed `drafts[i]` at `pos + 1 + i`) and
    /// `k_acc` the accepted DRAFT count (`k_emit - 1` = `0..=5`). So the A/B
    /// metric under this gate is `drafts[0] == verify_out[0]` — read directly off
    /// `k_acc >= 1` — NOT `drafts[0] == next` (which under this layout compares
    /// two different positions).
    ///
    /// # `emitted`, the commit and the counter
    ///
    /// The emitted tokens are the tokens at `pos + 1 ..= pos + k_acc + 1`, so the
    /// last one's position is `pos + k_acc + 1` — the value [`Self::dspark_commit`]
    /// writes for `keep = k_acc` at `pos_base = pos + 1`. `keep = k_acc` (not
    /// `k_emit`) is therefore BOTH the right counter and the right ring/compressor
    /// coverage: rows `0..k_acc` keep the KV of every emitted token EXCEPT the
    /// last, whose row (row `k_acc`, covered by the block's last accepted row or
    /// by the reversed bonus row) is rolled back and re-appended by the next
    /// step's `step_dev` — exactly the legacy arm's invariant. Keeping `k_emit`
    /// rows would leave the counter one past the last emitted token, desyncing the
    /// driver's `p += emitted.len()` from the chain's `pos_ctr`.
    ///
    /// # Failure
    ///
    /// Same contract as the legacy arm: the block is rolled back at its OWN row
    /// base (`pos + 1`, which is `pos_ctr + 1` after the step) before any error is
    /// returned, and `spec_primed` is left alone.
    fn dspark_spec_aligned(
        &mut self,
        dspark: &mut DsparkDev,
        token: u32,
        pos: usize,
    ) -> Result<DsparkSpecReport> {
        let cfg = self.cfg;
        // The block is [next, d1..d5] — 6 rows, ONE batched forward (the weights
        // are read ONCE for the whole block).
        let m = VERIFY_ROWS;
        debug_assert_eq!(
            m,
            DSPARK_DRAFTS + 1,
            "the aligned block is the anchor row plus the drafts"
        );

        let pos_ctr = self.dev.download_u32(self.s.pos_ctr.ptr as *const c_void)? as usize;
        debug_assert_eq!(
            pos_ctr, pos,
            "dspark_spec_step: `pos` must be the device position counter's current value"
        );

        // ---- 1. the real step: the whole-step graph (tap hook included), the
        // argmax (the token at `pos + 1` = the block's row 0), the counter + 1.
        let next = self.step_dev(token, pos)?;

        // ---- 2. the snapshot, AFTER the step and BEFORE the verify: row `j` of
        // the block sits at `pos_ctr + 1 + j`.
        let host_mirrors = self.dspark_snapshot(pos_ctr + 1, m)?;

        // ---- 3. the draft, from the tap of the step that just ran. The block's
        // anchor is `next` at `pos + 1`, so `draft_forward`'s seed lands on
        // `(pos + 1) - 1 = pos` — the position whose hidden the tap holds.
        let t = std::time::Instant::now();
        dspark.import_tap(self.s.dspark_tap.ptr as *const f32)?;
        dspark.draft_forward(next, pos + 1)?;
        let drafts = dspark.drafts()?;
        let draft_ms = t.elapsed().as_secs_f32() * 1e3;

        // ---- 4. the verify: [next, d1..d5] at pos+1..pos+6, ONE batched
        // forward, per-row argmax. Row 0 IS the anchor's forward, so the rows'
        // bit-parity with the plain engine at the same positions is the whole
        // correctness contract of this path.
        let t = std::time::Instant::now();
        let mut rows_in: Vec<u32> = Vec::with_capacity(m);
        rows_in.push(next);
        rows_in.extend_from_slice(&drafts);
        self.spec_capture = true;
        let rows_res = self.step_rows(&rows_in);
        self.spec_capture = false;
        let rows = match rows_res {
            Ok(r) => r,
            Err(e) => {
                let _ = self.dspark_rollback(pos_ctr + 1, m, &host_mirrors);
                return Err(e);
            }
        };
        let verify_ms = t.elapsed().as_secs_f32() * 1e3;
        if rows.len() != m {
            self.dspark_rollback(pos_ctr + 1, m, &host_mirrors)?;
            return Err(FerriteError::Config(format!(
                "dspark_spec_step (aligned): step_rows returned {} rows for a {m}-row verify block",
                rows.len()
            )));
        }

        // ---- 5. the accept: index-aligned on the anchor-carrying block (see the
        // doc comment), through the SHARED chain. `k_emit` counts the tokens the
        // step emits (the anchor plus the accepted drafts), 1..=m.
        let k_emit = spec_accept::<u32, u32>(&drafts, &rows, true);
        let k_acc = k_emit - 1;
        // The legacy report shape: the five argmaxes AFTER the anchor row. They
        // are the judges of `drafts[0..5]` (`verify_out[j]` = the token at
        // `pos + 2 + j`), and the block's LAST row's argmax — the bonus token at
        // `pos + 7` — is what this step does NOT emit.
        let mut verify_out = [0u32; DSPARK_DRAFTS];
        verify_out.copy_from_slice(&rows[..DSPARK_DRAFTS]);

        // ---- 6. the commit: roll back everything past the accepted prefix,
        // replay the kept rows through the compressors, advance the counter to
        // `pos + k_acc + 1` — the position of the LAST emitted token (whose KV the
        // rollback dropped, and which the next step's `step_dev` re-appends).
        let t = std::time::Instant::now();
        self.dspark_commit(pos_ctr + 1, m, k_acc, &host_mirrors)?;
        // The draft's window rings get the ACCEPTED rows back-filled at their own
        // positions (`pos + 1 ..= pos + k_acc`), which is the range the next
        // round's seed does not write either (`m` must be the block's row count:
        // the tap buffer's row stride is `VERIFY_ROWS`).
        dspark.note_ctx_rows(self.s.dspark_tap_r.ptr as *const f32, m, k_acc, pos + 1)?;
        // ---- 6b. and hand the NEXT round's draft its tap (`DSV41_SWALLOW_STEP`
        // only): the swallow drops `step_dev`, so this round's tap in
        // `dspark_tap` is the last one a single-row forward will write. Row
        // `k_acc - 1` sits at `pos + k_acc`, i.e. one before the swallow round's
        // own counter — the position its seed needs.
        if swallow_step() {
            Self::carry_kept_tap(
                self.dev,
                self.s.dspark_tap.ptr,
                self.s.dspark_tap_r.ptr as *const c_void,
                cfg.dim,
                k_acc,
            )?;
        }
        let commit_ms = t.elapsed().as_secs_f32() * 1e3;

        // ---- 7. what this step emits: `next` (the anchor's argmax) plus the
        // verify's argmax for every position the draft got right.
        let mut emitted = Vec::with_capacity(k_acc + 1);
        emitted.push(next);
        emitted.extend_from_slice(&verify_out[..k_acc]);
        // `emitted[i]` is the token at `pos + 1 + i`: `next` = `step_dev`'s argmax
        // at `pos + 1`, `verify_out[j]` = block row `j`'s argmax, and row `j` was
        // fed `rows_in[j]` at `pos + 1 + j`, so it is the token at `pos + 2 + j`.
        // `emitted.last()` is therefore the token at `pos + k_acc + 1` — exactly
        // the counter the commit above just wrote (`pos_base = pos_ctr + 1`,
        // `keep = k_acc`) and exactly the token the NEXT round embeds. The SAME
        // mapping as the legacy arm's 5-row block, which is why the two share one
        // construction; only the swallowed arm's row 0 differs (see there). See
        // [`Self::sids_writeback`] for this invariant across the three arms.
        // ★ THE s.ids WRITE-BACK — same root cause as the legacy arm above:
        // step_body embeds `s.ids`, and without this the next round would embed
        // a token k_acc positions stale (the digit task's self-locking
        // repetition). 4 bytes per round.
        if sids_writeback() {
            if let Some(&last) = emitted.last() {
                self.ul_i32(self.s.ids.ptr, &[last as i32])?;
            }
        }

        self.dspark_dump_step("spec", pos, token, next, k_acc, &drafts, &verify_out);
        // (1) the arms' common EXIT invariant: see the legacy arm. Idempotent
        // write-back means this check is the same statement for both arms.
        self.inv_ids(pos, &emitted)?;

        // The chain is now bootstrapped: with the swallow gate on, the next
        // round's 6-row block carries the anchor's forward. Set LAST, so a round
        // that failed above leaves this flag alone and the next round re-runs the
        // legacy path (whose `step_dev` re-supplies a valid tap).
        if swallow_step() {
            self.spec_primed = true;
        }

        Ok(DsparkSpecReport {
            next,
            drafts,
            verify_out,
            k_acc,
            emitted,
            draft_ms,
            verify_ms,
            commit_ms,
        })
    }

    /// The "swallowed main-chain step" arm of [`Self::dspark_spec_step`]
    /// (`DSV41_SWALLOW_STEP=1`; the second and every later round of a request).
    ///
    /// The round is the legacy one MINUS `step_dev`: the verify's 6-row block
    /// `[anchor, d1..d5]` sits at `pos .. pos+5`, so its row 0 IS the anchor's
    /// forward — the same token at the same position as `step_dev`'s, appending
    /// the same KV row — and its argmax IS `next`. The block pays one extra row
    /// (~1.6 ms) and saves the whole ~6.15 ms step.
    ///
    /// # Timeline
    ///
    /// ```text
    ///   1. snapshot    dspark_snapshot(pos, 6)   the block's write set; NO step
    ///                                          has advanced the counter, so the
    ///                                          block's row 0 is at `pos` itself
    ///   2. next/draft  import_tap + draft_forward(token, pos) -> d1..d5
    ///   3. verify_out  step_rows([token, d1..d5])   6 rows at pos .. pos+5
    ///   4. accept      k_emit = spec_accept(drafts, verify_out, true)
    ///   5. commit      dspark_commit(pos, 6, k_emit): rows 0..k_emit survive,
    ///                  the compressor is replayed for them, the counter jumps
    ///                  to pos + k_emit
    ///   6. emit        verify_out[0..k_emit]     (k_emit = k_acc + 1 tokens)
    /// ```
    ///
    /// # The accept arithmetic — the SAME chain GLM's MTP runs
    ///
    /// Block row `i` is fed `[token, d1..d5][i]` at position `pos + i`, so its
    /// argmax predicts `pos + 1 + i`; and `drafts[i]` is the draft's proposal for
    /// `pos + 1 + i` too (the draft block sits at `pos`). The two are therefore
    /// INDEX-ALIGNED, which is exactly GLM's `[t_last, d1..d_nd]` layout
    /// (`TpCluster::mtp_step`: `while drafts[k-1] == out[k-1]`), so the shared
    /// [`spec_accept`] runs with `anchor_is_in_block = true` and returns the
    /// number of tokens the step EMITS (`1..=6`) — the always-accepted anchor
    /// plus the surviving drafts. The legacy branch's 5-row block does not
    /// contain the anchor row, which is why ITS chain (and
    /// `SpecStep::ANCHOR_IS_IN_BLOCK`) stays the shifted one; the two layouts are
    /// one chain apart by the anchor.
    ///
    /// The report's `k_acc` keeps the LEGACY meaning (the accepted draft count =
    /// `k_emit - 1`) and `verify_out` the legacy shape (the 5 rows after the
    /// anchor, row `j` at `pos+1+j`), so the engine's "mean-k"/"tok/step"
    /// statistics and the golden dump stay comparable across the A/B.
    ///
    /// # The tap — the one approximate input of this path
    ///
    /// `draft_forward` needs the target hidden of the anchor's OWN position
    /// (`pos`); the legacy path gets it from `step_dev`'s forward of the anchor.
    /// Here the anchor's forward is row 0 of THIS round's block, i.e. it happens
    /// AFTER the draft, so the tap has to come from the previous round — and the
    /// previous round forwarded a different token at `pos`: its row `k_emit` is
    /// the first REJECTED draft, the row [`DsparkDev::note_ctx_rows`] deliberately
    /// drops ("the verify fed it a REJECTED draft token, so its hidden is not the
    /// true one").
    ///
    /// What IS exact is the LAST KEPT row (row `k_emit - 1`, at `pos - 1`): it was
    /// fed the true token of that position, and it is the same row GLM's MTP
    /// commits as `hprev <- hf_v[k-1]` (GLM's `k` = this block's `k_emit`).
    /// [`Self::carry_kept_tap`] hands that row over — the hidden of the last
    /// COMMITTED position, which is the same quantity the legacy tap holds, only
    /// evaluated at the (earlier) last committed position, because the anchor's
    /// row is not committed until this round's verify runs.
    ///
    /// To A/B the alternative (row `k_emit`, positionally exact but fed the
    /// rejected draft), add one to the `keep` argument of `carry_kept_tap` at both
    /// call sites.
    ///
    /// # Failure
    ///
    /// Same contract as the legacy arm: the block is rolled back before any error
    /// is returned, and `spec_primed` is left alone, so the next round re-runs the
    /// bootstrapping legacy path.
    fn dspark_spec_swallowed(
        &mut self,
        dspark: &mut DsparkDev,
        token: u32,
        pos: usize,
    ) -> Result<DsparkSpecReport> {
        let cfg = self.cfg;
        // The block is [anchor, d1..d5] — 6 rows, ONE batched forward (the
        // weights are read once for the whole block).
        let m = DSPARK_DRAFTS + 1;
        debug_assert_eq!(m, VERIFY_ROWS, "the swallowed block must fill VERIFY_ROWS");

        // ---- 1. the snapshot of the 6 ring slots the block will write. Taken
        // BEFORE the block and with NO step in between, so the block's row 0 is
        // the counter's current value — `pos` (the legacy branch's is `pos + 1`,
        // because its `step_dev` has already advanced the counter).
        let host_mirrors = self.dspark_snapshot(pos, m)?;

        // ---- 2. the draft, from the tap the PREVIOUS round carried over. The
        // anchor's own forward (row 0 below) has not happened yet — it is what
        // makes this path cheap and what forces the carry.
        let t = std::time::Instant::now();
        dspark.import_tap(self.s.dspark_tap.ptr as *const f32)?;
        // Draft geometry unchanged: the block is [embed(t0), noise×4] at the
        // anchor's OWN position (RoPE pos+1+r; the seed window row goes to slot
        // pos%win), so drafts[i] proposes `pos+1+i` — the same position block row
        // `i`'s argmax predicts.
        dspark.draft_forward(token, pos)?;
        let drafts = dspark.drafts()?;
        let draft_ms = t.elapsed().as_secs_f32() * 1e3;

        // ---- 3. the verify: [anchor, d1..d5] at pos..pos+5, ONE batched
        // forward, per-row argmax. Row 0's forward is the anchor's — the row
        // `step_dev` would have produced — so the rows' bit-parity with the plain
        // engine at the same positions is the whole correctness contract of this
        // path (what `dspark_parity` measures).
        let t = std::time::Instant::now();
        let mut rows_in: Vec<u32> = Vec::with_capacity(m);
        rows_in.push(token);
        rows_in.extend_from_slice(&drafts);
        self.spec_capture = true;
        let rows_res = self.step_rows(&rows_in);
        self.spec_capture = false;
        let rows = match rows_res {
            Ok(r) => r,
            Err(e) => {
                let _ = self.dspark_rollback(pos, m, &host_mirrors);
                return Err(e);
            }
        };
        let verify_ms = t.elapsed().as_secs_f32() * 1e3;
        if rows.len() != m {
            self.dspark_rollback(pos, m, &host_mirrors)?;
            return Err(FerriteError::Config(format!(
                "dspark_spec_step (swallow): step_rows returned {} rows for a {m}-row verify block",
                rows.len()
            )));
        }

        // ---- 4. the accept: index-aligned on the anchor-carrying block (see the
        // doc comment), through the SHARED chain. `k_emit` counts the tokens the
        // step emits (the anchor plus the accepted drafts), 1..=m.
        let k_emit = spec_accept::<u32, u32>(&drafts, &rows, true);
        let k_acc = k_emit - 1;
        let next = rows[0];
        // The legacy report shape: the five rows AFTER the anchor are the
        // `[d1..d5]`-style verify block, row `j` at `pos+1+j`, argmax `pos+2+j`.
        let mut verify_out = [0u32; DSPARK_DRAFTS];
        verify_out.copy_from_slice(&rows[1..m]);

        // ---- 5. the commit: rows 0..k_emit survive (the anchor's row is one of
        // them — that IS the swallowed `step_dev`), the rest is rolled back, the
        // compressor is replayed for the survivors and the counter lands on
        // `pos + k_emit` = the position of the last emitted token.
        let t = std::time::Instant::now();
        self.dspark_commit(pos, m, k_emit, &host_mirrors)?;
        dspark.note_ctx_rows(self.s.dspark_tap_r.ptr as *const f32, m, k_emit, pos)?;
        // ---- 5b. hand the NEXT round's draft its tap (see the doc comment).
        Self::carry_kept_tap(
            self.dev,
            self.s.dspark_tap.ptr,
            self.s.dspark_tap_r.ptr as *const c_void,
            cfg.dim,
            k_emit,
        )?;
        let commit_ms = t.elapsed().as_secs_f32() * 1e3;

        // ---- 6. what this step emits: every emitted token is the verify's own
        // argmax — rows[0] is `next` and the last one sits at `pos + k_emit`, the
        // new `pos_ctr`, whose KV is deliberately absent (the next round's block
        // row 0 appends it). Spelled the SAME way as the other two arms (the
        // anchor, then the accepted prefix of `verify_out`) — `rows[..k_emit]` IS
        // `[next] ++ verify_out[..k_acc]`, because `next = rows[0]` and
        // `verify_out[j] = rows[1 + j]` above — so the three arms share one
        // construction AND one invariant.
        //
        // The position mapping: `emitted[i]` is the token at `pos + 1 + i`
        // (`next` = the anchor row's argmax at `pos + 1`; `verify_out[j]` = row
        // `1 + j`'s argmax, fed `drafts[j]` at `pos + 1 + j`, so `pos + 2 + j`).
        // `emitted.last()` is therefore the token at `pos + k_emit` — exactly the
        // counter the commit above just wrote (`pos_base = pos`, `keep = k_emit`)
        // and exactly the token the NEXT round embeds. The legacy/aligned arms'
        // `pos + k_acc + 1` and this arm's `pos + k_emit` are the same number:
        // `k_emit = k_acc + 1` here, and their block's row 0 is one position
        // later. See [`Self::sids_writeback`] for the invariant.
        let mut emitted = Vec::with_capacity(k_acc + 1);
        emitted.push(next);
        emitted.extend_from_slice(&verify_out[..k_acc]);
        // ★ THE s.ids WRITE-BACK — the INVARIANT every spec arm must leave
        // behind: at the end of a round `s.ids` holds `emitted.last()`, the token
        // at the new `pos_ctr`. THIS arm does not read `s.ids` (it takes the
        // `token` arg for the draft and for the block's anchor row), so a missing
        // write-back is not immediately fatal here — but it IS a broken invariant
        // for the arms that DO read it: the bootstrap round re-enters the legacy
        // arm whenever the chain is not primed (and after every failure), and that
        // `step_dev` embeds `s.ids`, so it would embed a token `k_emit - 1`
        // positions stale. Same gate, same 4 bytes per round.
        if sids_writeback() {
            if let Some(&last) = emitted.last() {
                self.ul_i32(self.s.ids.ptr, &[last as i32])?;
            }
        }

        // The same dump shape as the legacy branch, so the golden diff compares
        // like for like. The vrow0 parity probe is deliberately NOT run here: its
        // verify half reads row 0's logits, which in this layout is the ANCHOR row
        // at `pos`, while its eager half forwards at `pos + 1` — comparing the two
        // would report a layout difference as a numerical one. The anchor row's
        // parity is what `dspark_parity`'s row-level diff measures.
        self.dspark_dump_step("spec", pos, token, next, k_acc, &drafts, &verify_out);
        // (1) the arms' common EXIT invariant: the round left `s.ids` holding the
        // token the NEXT round embeds. THIS arm never READS `s.ids`, but the
        // round after it may be the legacy bootstrap one — and that `step_dev`
        // embeds it — so the check belongs here too.
        self.inv_ids(pos, &emitted)?;

        Ok(DsparkSpecReport {
            next,
            drafts,
            verify_out,
            k_acc,
            emitted,
            draft_ms,
            verify_ms,
            commit_ms,
        })
    }

    // =========================================================================
    // `DSV41_LAZY_VERIFY` — the row-by-row verify arm and its route
    //
    // Design: `docs/agent/lazy-batched-gate.md` (route, deferred tap, commit
    // semantics) and `docs/agent/verify-fusion-arch.md` §3 (row semantics, the
    // zero-rollback argument). Default OFF: every method here is only reachable
    // through [`lazy_verify`], so the existing arms are untouched.
    // =========================================================================

    /// Which arm THIS round takes, given the sticky [`Self::lazy_mode`] and the
    /// integer mean-k window — the Schmitt trigger of `lazy-batched-gate.md` §2.3.
    ///
    /// The verdict is a pure function of (the integer window, the process-level
    /// `τ`, the sticky arm), so every rank reaches the same one and issues the
    /// same number of collectives within the round: cross-round mismatches are
    /// what wedges AR v5, and same-round synchronised switches cannot produce one
    /// (`lazy-batched-gate.md` §4.3). The two arms claim one verify-graph slot
    /// each (`m = 1` / `m = 6`), so switching costs no re-capture once both have
    /// been visited.
    fn lazy_route_decide(&mut self) -> bool {
        let tau = lazy_tau();
        let mk = self.lazy_hist.mean_k();
        // Two thresholds, one arm: a window sitting on `τ` must not flip the arm
        // every round, so the arm only moves once the mean crosses the boundary
        // `LAZY_HYST` away from the threshold.
        let (lo, hi) = (tau - LAZY_HYST, tau + LAZY_HYST);
        let use_lazy = match self.lazy_mode {
            None => mk < tau,
            Some(Arm::Lazy) => mk < hi,
            Some(Arm::Batched) => mk < lo,
        };
        self.lazy_mode = Some(if use_lazy { Arm::Lazy } else { Arm::Batched });
        use_lazy
    }

    /// The lazy arm's commit: move the counter to `pos + k_emit` and do nothing
    /// else.
    ///
    /// [`Self::dspark_commit`] is the BATCHED arm's three-stage commit — roll the
    /// block's rejected tail back, replay the kept rows through every compressor,
    /// then move the counter. None of the three applies here, and calling it would
    /// be WRONG rather than merely slow:
    ///
    /// * **no rollback.** Rows run one at a time and the loop STOPS at the first
    ///   draft that misses, so the rows it ran are exactly the commit's keep range
    ///   `0..k_emit` (`lazy-batched-gate.md` §3.4) — there is no rejected tail to
    ///   undo. (The snapshot is still taken per round, but only for the ERROR
    ///   path: a `step_rows` that fails part-way leaves rows the loop cannot
    ///   account for, and those DO have to go.)
    /// * **no replay.** Each row's `step_rows` already committed that row's
    ///   compressor state, in position order, through the same single-row
    ///   pool+commit pair `compress_replay` uses — so the compressor after
    ///   `k_emit` single-row forwards already IS the state `k_emit` sequential
    ///   decode steps would leave. Replaying the rows would advance
    ///   `state_kv`/`state_score`/`clen` a SECOND time per row: the double commit
    ///   `lazy-batched-gate.md` §0-4 calls out as the reason the design's own
    ///   pseudo-code had to be corrected.
    /// * **the counter.** `step_rows`' per-row argmax takes a NULL counter, so a
    ///   block never advances `pos_ctr`; this arm moves it row by row as it goes
    ///   and this call lands it on `pos + k_emit` — the position of the last
    ///   emitted token, which is the token the next round embeds.
    fn dspark_commit_lazy(&mut self, pos: usize, k_emit: usize) -> Result<()> {
        debug_assert!(
            (1..=VERIFY_ROWS).contains(&k_emit),
            "dspark_commit_lazy: k_emit {k_emit} is outside 1..=VERIFY_ROWS"
        );
        self.set_pos_ctr(pos + k_emit)?;
        // The invariant the batched commit also ends on: the host mirror and the
        // device counter describe ONE committed prefix. Here both advanced through
        // the same `compress_row` calls, so they agree by construction — which is
        // exactly what this (free unless `DSV41_INV_CHECK=1`) check is here to
        // notice if a later edit ever splits them.
        self.inv_compress_len()?;
        Ok(())
    }

    /// Copy the deferred tap of the row just forwarded into that row's own slot of
    /// the per-row tap block (`dspark_tap_r`), one D2D per target slot.
    ///
    /// The staging buffer is `[DSPARK_TAP_SLOTS, dim]`, so slot `s` sits at
    /// `s * dim` there and at `(s * VERIFY_ROWS + i) * dim` in the block. Those are
    /// the same two strides the hook and its consumers
    /// ([`DsparkDev::note_ctx_rows`], [`Self::carry_kept_tap`]) already use, so
    /// this copy is the ONLY place the row index enters — the consumers need no
    /// change at all.
    fn lazy_tap_commit(&self, i: usize) -> Result<()> {
        debug_assert!(i < VERIFY_ROWS, "lazy_tap_commit: row {i} past the tap block");
        let row_bytes = self.cfg.dim * std::mem::size_of::<f32>();
        // DSV41_LAZY_SDR: ONE strided copy instead of `DSPARK_TAP_SLOTS` of them.
        // The staging slots are `row_bytes` apart (contiguous `[slots, dim]`) and
        // so are the block's slots, but each block SLOT's rows are
        // `VERIFY_ROWS * row_bytes` apart — which is exactly the pitch
        // `cudaMemcpy2DAsync` carries. Byte-identical to the loop below: same
        // slots, same rows, same `row_bytes` each.
        if lazy_sdr() {
            let src = self.s.dspark_tap.ptr as *const u8;
            let dst = (self.s.dspark_tap_r.ptr as *mut u8)
                .wrapping_add(i * row_bytes) as *mut c_void;
            return self.dev.memcpy_d2d_2d(
                dst,
                VERIFY_ROWS * row_bytes,
                src as *const c_void,
                row_bytes,
                row_bytes,
                DSPARK_TAP_SLOTS,
            );
        }
        for slot in 0..DSPARK_TAP_SLOTS {
            let src = (self.s.dspark_tap.ptr as *const u8).wrapping_add(slot * row_bytes);
            let dst = (self.s.dspark_tap_r.ptr as *mut u8)
                .wrapping_add((slot * VERIFY_ROWS + i) * row_bytes);
            self.dev
                .memcpy_d2d(dst as *mut c_void, src as *const c_void, row_bytes)?;
        }
        Ok(())
    }

    /// ONE row of the lazy verify: forward `rows_in[i]` at `pos + i`, return that
    /// row's argmax.
    ///
    /// Two host-side corrections are what make a loop of `step_rows(m = 1)` calls
    /// equivalent to the batched block (`lazy-batched-gate.md` §3.2):
    ///
    /// * **the position counter.** `step_rows` reads `pos_base` off the DEVICE
    ///   counter and builds its block at `pos_base .. pos_base + m - 1`, so a
    ///   one-row block would forward every row at whatever position the counter
    ///   last held. Pushing `pos + i` first (a 4-byte H2D) is what puts row `i` at
    ///   `pos + i`; the graph only READS the counter, so one captured `m = 1` graph
    ///   serves every row.
    /// * **the tap.** `layer_rows` writes each target layer's tap at its slot's
    ///   row 0 for a one-row block, so every row would overwrite row 0 and the
    ///   commit would read row 0's hidden for all `k_emit` rows. In deferred mode
    ///   the hook stages the row in the single-row buffer and
    ///   [`Self::lazy_tap_commit`] copies it to row `i` afterwards.
    ///
    /// `spec_capture` stays CLEAR, deliberately: this arm never replays the
    /// compressor (so the `kvp`/`scp` snapshot would be pure waste) and its tap
    /// write is handled by the deferred path above.
    ///
    /// The per-row `step_rows` runs with the replay rendezvous SUPPRESSED: the
    /// caller already rendezvoused once before its loop ([`Self::dspark_spec_lazy`])
    /// and, from there to the loop's end, every rank is executing the SAME replay
    /// sequence on one stream, so the device-side AR stamps are the only
    /// cross-rank ordering that remains — and v5 carries them in-kernel
    /// ([`Self::step_rows_sync`]). Suppressing it here is what removes the
    /// ~2.85 ms/row of host rendezvous the per-row `step_rows(m = 1)` used to pay.
    fn lazy_run_row(&mut self, rows_in: &[u32], i: usize, pos: usize) -> Result<u32> {
        // DSV41_LAZY_SDR: one BLOCKING H2D fewer per row. The counter parks at the
        // block's row-0 position (written once by [`Self::dspark_spec_lazy`]) and
        // FIX (accept-degradation-rootcause): the claim "nothing in the m-row
        // path reads it" is WRONG — the per-row rope fallbacks read it:
        // `apply_rope` at :9543 (q rope), :10047 (inverse o-rope), :10589
        // (indexer q rope, NO mrows alternative in this config) all compute
        // `t = *pos_ctr + off`, so the 4-byte H2D IS load-bearing: it puts
        // row `i` at `pos + i` for the rope rotations. Skipping it made row
        // i's q/o⁻¹/idx-q rotate at `pos` instead of `pos + i` — corrupting
        // the verify argmax (k_acc 5.0→2.4). LAZY_SDR keeps only its safe
        // part (the tap-staging D2D merge at :8206); the per-row set_pos_ctr
        // H2D is restored unconditionally.
        self.set_pos_ctr(pos + i)?;
        self.spec_capture = false;
        self.spec_tap_deferred = true;
        // `Some(pos + i)` is the value `step_rows` would read straight back off the
        // device counter: the device is quiescent (the previous row ended in a
        // blocking D2H), so the hint IS the counter. (With the per-row
        // `set_pos_ctr` above, that is by construction; under
        // `DSV41_LAZY_SDR` it is by the argument that the counter is not read at
        // all inside the block.) That removes the per-row read-back, which — like
        // the argmax D2H — is a full device sync.
        let res = self.step_rows_sync(&rows_in[i..=i], true, Some(pos as i32 + i as i32));
        // Cleared BEFORE the `?`, so a failed row does not leave the hook in
        // deferred mode for whatever runs next.
        self.spec_tap_deferred = false;
        let rows = res?;
        if rows.len() != 1 {
            return Err(FerriteError::Config(format!(
                "dspark_spec_step (lazy): step_rows returned {} rows for a 1-row block",
                rows.len()
            )));
        }
        self.lazy_tap_commit(i)?;
        Ok(rows[0])
    }

    /// The row-by-row ("lazy") arm of [`Self::dspark_spec_step`]
    /// (`DSV41_LAZY_VERIFY=1`, when [`Self::lazy_route_decide`] selects it).
    ///
    /// # What it is
    ///
    /// The SAME `[anchor, d1..d5]` block as [`Self::dspark_spec_swallowed`], run
    /// as one-row forwards in order and stopped at the first draft that misses:
    ///
    /// ```text
    ///   row 0   token       @ pos       -> rows[0], judged against drafts[0]
    ///   row 1   drafts[0]   @ pos + 1   -> rows[1], judged against drafts[1]
    ///   ...
    ///   row i   drafts[i-1] @ pos + i   -> rows[i], judged against drafts[i]
    /// ```
    ///
    /// Row `i`'s argmax predicts `pos + 1 + i` and `drafts[i]` proposes exactly
    /// `pos + 1 + i`, so the chain is index-aligned and the SHARED
    /// [`spec_accept`] judges it with `anchor_is_in_block = true` — the same call
    /// the batched arm makes, which is what keeps the two arms' `k_emit`
    /// comparable.
    ///
    /// # Why the order alone is enough
    ///
    /// `step_rows` documents that row `r` sees `[pos_base + r - window + 1,
    /// pos_base + r]`: row `i`'s attention depends only on rows already in the
    /// ring. The loop runs `0, 1, 2, ...`, so row `i - 1`'s KV is in the ring
    /// before row `i` reads it — the causal chain holds with no extra
    /// synchronisation and no cross-row state to carry.
    ///
    /// # Cost shape
    ///
    /// `k_emit` single-row forwards. The best case (`drafts[0] != rows[0]`) is ONE
    /// row — the swallowed main-chain step alone — and the worst (all five drafts
    /// right) is six, i.e. the batched arm's block. The arm therefore cannot be
    /// slower than what it replaces, and on a low-accept workload it is far
    /// cheaper.
    ///
    /// # The two-row batch (SDR): analysed, and NOT taken
    ///
    /// `docs/agent/lazy-verify-optimization-path.md` §L2 proposes running rows
    /// `i`, `i+1` back to back on one stream and reading BOTH argmaxes with one
    /// D2H. The dataflow premise HOLDS — the drafts are produced before the loop
    /// ([`DsparkDev::draft_forward`] + [`DsparkDev::drafts`]), so row `i + 1`'s
    /// input token is known before row `i` has been run and the two forwards are
    /// independent. What fails is everything else:
    ///
    /// * **the second row is SPECULATIVE, and half the time it is wasted.** The
    ///   loop's early exit is the whole point of the arm: row `i + 1` runs only
    ///   when `drafts[i] == rows[i]`. At this workload's accept rate (mean
    ///   `k_emit` 2.214 ⇒ ~45% of drafts miss) fixed pairs `(0,1) (2,3) (4,5)`
    ///   run `2·⌈k_emit/2⌉` rows, i.e. 4.0 rows/step against today's 2.214 —
    ///   **+1.79 rows/step**. At the ~6 ms a row costs (`c_row` 6.15 eager / 8.18
    ///   lazy, and 6 rows = 34–37 ms for the batched block) that is **+10 ms**,
    ///   against the ~0.5 ms the halved D2H count saves. Pairing only the tail
    ///   (rows 4–5) does not rescue it either: the D2H gain is then ~0.05 ms and
    ///   the waste ~0.24 ms.
    /// * **the discarded row has no exact rollback here.** This arm runs with
    ///   `spec_capture` CLEAR on purpose ([`Self::lazy_run_row`]), so the
    ///   compressor's per-row `kvp`/`scp` inputs are never saved and
    ///   [`Self::compress_replay`] — the only exact way to land the compressor on
    ///   "the state after `keep` rows" — cannot run. `dspark_rollback_keep`
    ///   restores the compressor WHOLE, so a rollback without the replay would
    ///   leave the rejected row's committed latent in `clen`/`index_k` and hand
    ///   the indexer a KV row for a token that was never emitted (the corruption
    ///   class `lazy-batched-gate.md` §0-4 records).
    /// * **the per-row scratch is single-row by construction.** The deferred tap
    ///   hook stages ONE row into a fixed `[DSPARK_TAP_SLOTS, dim]` address, and
    ///   the block's tap/`kvp`/`scp` snapshots index rows relative to the block's
    ///   row 0 — so only the pair at `i = 0` lines up with `dspark_rollback_keep`'s
    ///   ring-slot rule, and a pair at `i >= 2` would relocate the tap and
    ///   re-index both snapshots per pair, inside the captured shape. The `m = 2`
    ///   block would also have to stay on the DIRECT arm: the `kvp`/`scp` save is
    ///   HOST code in the layer hook, and a replayed graph runs no host code, so a
    ///   captured `m = 2` shape would silently leave the rollback without its
    ///   input (and the direct arm gives back the graph's ~1.5 ms). Threading a row
    ///   offset through the hook and the snapshot is a redesign of `layer_rows` +
    ///   `dspark_snapshot` + `compress_replay`, not the "~50 行" the note assumes.
    ///
    /// What IS taken from §L2 is the sync coalescing that does not speculate:
    /// [`lazy_sdr`] folds the tap carry into one strided copy and drops the
    /// per-row counter H2D. The batched argmax D2H stays per row — it is the
    /// dependency the early exit is built on, and buying it with a speculative row
    /// costs an order of magnitude more than it saves.
    ///
    /// # Report
    ///
    /// Same shape as the batched arm's, so the driver's mean-k / tok-per-step
    /// statistics stay comparable. ONE difference to know about:
    /// `verify_out[j]` for `j >= k_acc` is UNDEFINED here (left at 0) — the
    /// batched arm fills it with the argmax of the rejected drafts' rows, which
    /// this arm never forwards. The golden diff must compare `0..k_acc` only.
    ///
    /// # Failure
    ///
    /// Same contract as the other arms: the block is rolled back before any error
    /// is returned and `spec_primed` is left alone, so the next round re-runs the
    /// bootstrapping legacy path. The rollback is the ONLY consumer of the
    /// snapshot here (see [`Self::dspark_commit_lazy`]), and it is what puts
    /// `pos_ctr` back as well: a failed loop has already advanced the counter past
    /// the rows it ran, and the next round must restart from `pos`.
    fn dspark_spec_lazy(
        &mut self,
        dspark: &mut DsparkDev,
        token: u32,
        pos: usize,
    ) -> Result<DsparkSpecReport> {
        let cfg = self.cfg;
        // The block's UPPER bound is the swallowed one; how many of its rows
        // actually run is what the early exit decides.
        let m = DSPARK_DRAFTS + 1;
        debug_assert_eq!(m, VERIFY_ROWS, "the swallowed block must fill VERIFY_ROWS");

        // ---- 1. the snapshot: the ERROR PATH's only input. The loop below keeps
        // every row it runs, so a successful round has nothing to roll back
        // (`lazy-batched-gate.md` §3.4) — this exists for a round whose
        // `step_rows` fails part-way.
        let host_mirrors = self.dspark_snapshot(pos, m)?;

        // ---- 2. the draft, from the tap the PREVIOUS round carried over: this
        // round's anchor forward is row 0 below, so it has not happened yet.
        // Identical to the batched arm — same block geometry, same draft geometry.
        let t = std::time::Instant::now();
        dspark.import_tap(self.s.dspark_tap.ptr as *const f32)?;
        dspark.draft_forward(token, pos)?;
        let drafts = dspark.drafts()?;
        let draft_ms = t.elapsed().as_secs_f32() * 1e3;

        // ---- 3. the row loop: forward, judge, stop at the first miss.
        let mut rows_in: Vec<u32> = Vec::with_capacity(m);
        rows_in.push(token);
        rows_in.extend_from_slice(&drafts);

        // ★ THE LOOP'S ONE RENDEZVOUS. Each row below suppresses the per-replay
        // barrier `step_rows` normally keeps in front of `graph_launch` (see
        // [`Self::step_rows_sync`]): with the ranks aligned HERE, and with the
        // `m = 1` graph's own CAPTURE still bracketed by the two barriers inside
        // `step_rows_sync`, no rank can be recording while a peer executes the
        // replay. From this point to the end of the loop every rank runs the same
        // replay sequence back-to-back on one stream, and the only cross-rank
        // ordering left is the v5 AR's in-kernel publish/poll — which is
        // drift-tolerant by construction (absolute epoch stamps + a read-before-
        // overwrite publish chain; see the `p2p_ar_pubred_v5_kernel` note in
        // kernels/cuda/ferrite_kernels.cu). Placed AFTER the draft forward so it
        // also covers the draft's kernels. No-op without peers (comm is None off
        // TP8), so single-rank runs are unaffected.
        if let Some(c) = self.comm.as_ref() {
            c.host_barrier();
        }

        // DSV41_LAZY_SDR: the round's ONE counter write, where the loop otherwise
        // writes it once per row. The counter ends the round at `pos + k_emit`
        // regardless (`dspark_commit_lazy`), and nothing inside the block reads it
        // — see [`lazy_sdr`] for the argument. Parking it on the block's row-0
        // position keeps "the counter is where the block starts" true for anything
        // that only ever LOOKS at it between rounds.
        if lazy_sdr() {
            self.set_pos_ctr(pos)?;
        }

        let t = std::time::Instant::now();
        let mut rows: Vec<u32> = Vec::with_capacity(m);
        let mut fail: Option<FerriteError> = None;
        match self.lazy_run_row(&rows_in, 0, pos) {
            Ok(a) => rows.push(a),
            Err(e) => fail = Some(e),
        }
        // The anchor row's argmax is the judge of `drafts[0]`: a miss here emits
        // the anchor alone (k_emit = 1) and NO other row is forwarded at all.
        if fail.is_none() && drafts[0] == rows[0] {
            for i in 1..=DSPARK_DRAFTS {
                match self.lazy_run_row(&rows_in, i, pos) {
                    Ok(a) => rows.push(a),
                    Err(e) => {
                        fail = Some(e);
                        break;
                    }
                }
                // Row `i` is judged by `drafts[i]` while `i < DSPARK_DRAFTS`. The
                // block's LAST row (i = DSPARK_DRAFTS) has no draft left to judge
                // — but it still has to be forwarded, because when every draft
                // matched its argmax IS the sixth emitted token. (The pseudo-code
                // in `lazy-batched-gate.md` §3.1 stops at i = 4, which is one row
                // short of its own `rows_run == k_emit` invariant: the full accept
                // needs both row 4's judgement of `drafts[4]` AND row 5's argmax.
                // The `debug_assert_eq!` below pins the corrected count.)
                if i < DSPARK_DRAFTS && drafts[i] != rows[i] {
                    break;
                }
            }
        }
        let verify_ms = t.elapsed().as_secs_f32() * 1e3;

        // ---- 3b. the error path: the rows the loop ran WERE forwarded and the
        // ones after the failing row never were, so the block goes back whole
        // (keep = 0) — and because the loop advances the counter row by row, the
        // counter goes back to `pos` with it.
        if let Some(e) = fail {
            let _ = self.dspark_rollback(pos, m, &host_mirrors);
            let _ = self.set_pos_ctr(pos);
            return Err(e);
        }

        // ---- 4. the accept, through the SAME shared chain as every other arm:
        // the rows this loop ran are exactly the prefix the comparison needed, so
        // the result is the EMITTED token count (the anchor plus the surviving
        // drafts), 1..=m.
        let k_emit = spec_accept::<u32, u32>(&drafts, &rows, true);
        debug_assert_eq!(
            rows.len(),
            k_emit,
            "lazy: the rows forwarded must equal the emitted count (rows_run == k_emit)"
        );
        let k_acc = k_emit - 1;
        let next = rows[0];
        // The legacy report shape: the rows AFTER the anchor, row `j` at
        // `pos + 1 + j` — the same shift the batched arm applies. Only `0..k_acc`
        // has a defined value here (see the doc comment).
        let mut verify_out = [0u32; DSPARK_DRAFTS];
        verify_out[..k_acc].copy_from_slice(&rows[1..k_emit]);

        // ---- 5. the commit: no rollback, no replay — the rows already committed
        // their own compressor state, in order (`lazy-batched-gate.md` §0-4).
        let t = std::time::Instant::now();
        self.dspark_commit_lazy(pos, k_emit)?;
        dspark.note_ctx_rows(self.s.dspark_tap_r.ptr as *const f32, m, k_emit, pos)?;
        // ---- 5b. hand the NEXT round's draft its tap, by the same rule as the
        // batched arm: the LAST KEPT row is the hidden of the last committed
        // position.
        Self::carry_kept_tap(
            self.dev,
            self.s.dspark_tap.ptr,
            self.s.dspark_tap_r.ptr as *const c_void,
            cfg.dim,
            k_emit,
        )?;
        let commit_ms = t.elapsed().as_secs_f32() * 1e3;

        // ---- 6. what this step emits: the anchor's argmax plus the argmax of
        // every accepted row, i.e. `rows[0..k_emit]` — spelled the SAME way as the
        // batched arm so the arms share one construction and one invariant
        // (`emitted[i]` is the token at `pos + 1 + i` and `emitted.last()` the
        // token at `pos + k_emit`, which is the counter the commit just wrote).
        let mut emitted = Vec::with_capacity(k_acc + 1);
        emitted.push(next);
        emitted.extend_from_slice(&verify_out[..k_acc]);
        // ★ THE s.ids WRITE-BACK — the same invariant every spec arm leaves
        // behind: at the end of a round `s.ids` holds `emitted.last()`, the token
        // at the new `pos_ctr`. See the batched arm for why a missing write-back
        // is only *immediately* fatal for the arms that READ `s.ids`.
        if sids_writeback() {
            if let Some(&last) = emitted.last() {
                self.ul_i32(self.s.ids.ptr, &[last as i32])?;
            }
        }

        self.dspark_dump_step("spec", pos, token, next, k_acc, &drafts, &verify_out);
        self.inv_ids(pos, &emitted)?;
        // Set LAST, so a round that failed above leaves the flag alone and the
        // next round re-runs the legacy bootstrap. ONE flag primes both arms of
        // the swallow family — the lazy route picks between them per round.
        self.spec_primed = true;

        Ok(DsparkSpecReport {
            next,
            drafts,
            verify_out,
            k_acc,
            emitted,
            draft_ms,
            verify_ms,
            commit_ms,
        })
    }

    /// Carry the tap the NEXT spec round's draft reads out of the block that just
    /// committed: row `keep - 1` of `dspark_tap_r` — the LAST KEPT row — is copied
    /// into `dspark_tap` (the single-row tap buffer [`DsparkDev::import_tap`]
    /// consumes), one D2D per target slot.
    ///
    /// # Which row, and why it is `keep - 1`
    ///
    /// The draft of the round AFTER a commit seeds the window ring at the new
    /// `pos_ctr` and needs the target hidden of the position just before it, i.e.
    /// of the last position the commit KEPT. Kept rows are `0..keep` from the
    /// block's row 0, so that is row `keep - 1`, at position
    /// `pos_base + keep - 1 = pos_ctr_new - 1`. In the swallowed 6-row block
    /// (`keep = k_emit`) that is index `k_acc` and in the legacy 5-row block
    /// (`keep = k_acc`) index `k_acc - 1` — the two indices the diff doc lists,
    /// which are one rule seen through the two layouts.
    ///
    /// Every kept row was fed a TRUE token (row 0 is the anchor, rows 1..=k_acc
    /// are accepted drafts), which is what makes this the exact hidden of that
    /// position's token — unlike row `keep` (the first rejected draft), whose
    /// hidden [`DsparkDev::note_ctx_rows`] drops for being the wrong token's.
    ///
    /// `keep == 0` (the legacy branch's "no draft accepted") copies nothing: the
    /// tap `step_dev` just wrote is already the hidden of the last committed
    /// position, because that step committed `pos` itself.
    fn carry_kept_tap(
        dev: &Device,
        tap: *mut c_void,
        tap_r: *const c_void,
        dim: usize,
        keep: usize,
    ) -> Result<()> {
        if keep == 0 {
            return Ok(());
        }
        debug_assert!(keep <= VERIFY_ROWS, "carry_kept_tap: row {keep} is past the tap block");
        let row_bytes = dim * std::mem::size_of::<f32>();
        let row_off = (keep - 1) * row_bytes;
        for slot in 0..DSPARK_TAP_SLOTS {
            let src = (tap_r as *const u8)
                .wrapping_add(slot * VERIFY_ROWS * row_bytes)
                .wrapping_add(row_off);
            let dst = (tap as *mut u8).wrapping_add(slot * row_bytes);
            dev.memcpy_d2d(dst as *mut c_void, src as *const c_void, row_bytes)?;
        }
        Ok(())
    }

    /// Commit the accepted prefix of a verify block: undo everything the block
    /// wrote past row `keep`, then move the engine forward by those kept rows.
    ///
    /// `keep` is the number of rows the block KEPT, counted from row 0 — and row
    /// 0 sits at `pos_base`, so the kept rows are the positions
    /// `pos_base .. pos_base + keep` and the new `pos_ctr` is `pos_base + keep`.
    /// That formulation is layout-independent on purpose: the two blocks differ
    /// only in where row 0 is and in how many rows are valid.
    ///
    /// * the LEGACY 5-row block `[d1..d5]` sits at `pos+1 .. pos+5` (row 0 is
    ///   `d1`) and `keep = k_acc`: the accepted drafts' KV plus nothing else. Row
    ///   `k_acc` (= the first REJECTED draft) is not kept — its position is the
    ///   new `pos_ctr`, so that token has not been consumed yet and the next
    ///   step's `step_dev` appends its KV. `pos_ctr = pos + k_acc + 1`.
    /// * the SWALLOWED 6-row block `[anchor, d1..d5]` sits at `pos .. pos+5` (row
    ///   0 IS the anchor, always valid) and `keep = k_emit = k_acc + 1`: the
    ///   anchor's row plus the accepted drafts. Row `k_emit` (= the first
    ///   rejected draft) is not kept, for the same reason above.
    ///   `pos_ctr = pos + k_emit`.
    ///
    /// What the block left behind, and what happens to it:
    ///
    /// * ring rows `0..keep` — the accepted prefix's KV, kept as they are;
    /// * ring rows `keep..m` — restored from the snapshot;
    /// * the compressor carry/latent/`out_rows` and BOTH compressed-row counters
    ///   (`s.clen` and the host mirror) — restored from the snapshot, after which
    ///   the kept rows are replayed ([`Self::compress_replay`]);
    /// * `index_k` — never snapshotted: the replay re-publishes a key for every
    ///   group it commits, and slots the verify wrote past the new `clen` are
    ///   unreachable (`indexer_topk` reads only `< clen`);
    /// * the ring's COMPRESSED rows (`window + clen`) — the replay rewrites the
    ///   kept ones; the rest are unreachable for the same reason;
    /// * `pos_ctr` — the one thing the verify deliberately does NOT advance
    ///   (`step_rows`' per-row argmax takes a NULL counter), so it is set here.
    fn dspark_commit(
        &mut self,
        pos_base: usize,
        m: usize,
        keep: usize,
        host: &[(usize, usize)],
    ) -> Result<()> {
        // keep == m is a LEGAL full-accept (all 5 drafts right): rollback_keep
        // then restores nothing (every slot it would touch is the committed
        // prefix's), compress_replay replays all the rows, and the counter lands
        // at pos_base + m.
        debug_assert!(
            keep <= m,
            "dspark_commit: keep {keep} exceeds the {m}-row block"
        );
        self.dspark_rollback_keep(pos_base, m, keep, host)?;
        if keep > 0 {
            self.compress_replay(pos_base as i32, keep)?;
        }
        self.set_pos_ctr(pos_base + keep)?;
        // (4) at the END of the commit the host mirror and the device *clen must
        // describe the SAME committed prefix: the rollback restored both from the
        // same snapshot and the replay advanced both by the same rule. A drift is
        // the split brain the `comp_len > 0` host branch would act on. ONE D2H 4 B.
        self.inv_compress_len()?;
        Ok(())
    }

    /// Play the KEPT rows of a verify block back through every compressor.
    ///
    /// [`Self::dspark_rollback_keep`] has just restored each compress source to
    /// its pre-verify state; this walks rows `0..rows` forward again with the
    /// `kvp`/`scp` the verify computed for them (`Scratch::spec_snap_kvp`/`_scp`),
    /// through the SAME single-row pool+commit pair the verify used, at the same
    /// positions. The carry, the pooled latent, the compressed row in the ring
    /// and this layer's device counter therefore end up exactly where `rows`
    /// sequential single-row decode steps would have left them.
    ///
    /// `pos_base` is the block's ROW 0 position (`pos + 1` — what `step_rows`
    /// read off the device counter after the real step).
    ///
    /// The index key is re-published for every row that COMPLETES a group, which
    /// is what the single-row path does (its `indexer` republishes slot
    /// `*clen - 1` every step). The m-row verify publishes only the block's LAST
    /// group, so without this the intermediate groups it committed would leave
    /// `index_k` rows that `indexer_topk` reads and never got written.
    fn compress_replay(&mut self, pos_base: i32, rows: usize) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let cfg = self.cfg;
        let hd = cfg.head_dim;
        let st = self.dev.stream();
        // Re-upload the row positions so the replay does not depend on
        // `s.pos_rows` still holding what `step_rows` put there.
        let pos_rows: Vec<i32> = (0..rows).map(|r| pos_base + r as i32).collect();
        self.ul_i32(self.s.pos_rows.ptr, &pos_rows)?;
        for layer in self.compress_sources() {
            let ld = &self.w.layers[layer];
            let (Some(_), Some(norm)) = (ld.comp_wkv.as_ref(), ld.comp_norm.as_ref()) else {
                continue;
            };
            let ratio = cfg.compress_ratio(layer).max(1);
            let kvp_base =
                (self.s.spec_snap_kvp.ptr as *const f32).wrapping_add(layer * VERIFY_ROWS * hd);
            let scp_base =
                (self.s.spec_snap_scp.ptr as *const f32).wrapping_add(layer * VERIFY_ROWS * hd);
            for r in 0..rows {
                self.dev.compressor_pool_on(
                    kvp_base.wrapping_add(r * hd),
                    scp_base.wrapping_add(r * hd),
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
                // The host MIRROR of the device counter, by the SAME rule the
                // commit kernel applies ((*pos + 1) % ratio == 0).
                if (pos_base + r as i32 + 1) % (ratio as i32) == 0 {
                    self.layers[layer].compress_len += 1;
                    // A group just completed, so its index key has to exist —
                    // `*clen - 1` is the slot the commit kernel just named.
                    if cfg.is_index_source(layer) && cfg.indexer_owns_k(layer) {
                        self.publish_index_key(layer)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Set the device position counter (a 4-byte H2D). The verify never writes
    /// it — `step_rows` passes a NULL counter to its per-row argmax precisely so a
    /// block of rows cannot advance the sequence — so a committed block has to
    /// move it here, once, for the whole accepted prefix.
    fn set_pos_ctr(&self, pos: usize) -> Result<()> {
        self.ul_i32(self.s.pos_ctr.ptr, &[pos as i32])
    }

    // =========================================================================
    // Cross-step INVARIANT ASSERTIONS (`DSV41_INV_CHECK=1`, default OFF)
    //
    // Every method here is a no-op unless [`inv_check`] is on: with the gate off
    // each is one branch, so the steady-state path pays nothing. With it on, a
    // broken invariant prints ONE `[inv-fail]` line (invariant, expected, seen)
    // and returns `Err` — the caller rolls its block back, so the step that
    // BROKE the invariant is the step that reports it, not a later step whose
    // output merely looks wrong.
    //
    // The checks are the eight root causes of the spec-chain hardening session,
    // each pinned to the exact place its producer and consumer could disagree.
    // Their costs are deliberately of two kinds only (see [`inv_check`]): a
    // static (compile-time / `debug_assert`) shape check, or a single 4-byte
    // device→host spot read.
    // =========================================================================

    /// The common failure report: one line naming the invariant, the position and
    /// the values, and an `Err` that carries the same text.
    fn inv_fail(&self, invariant: &str, pos: usize, detail: &str) -> FerriteError {
        eprintln!("[inv-fail] pos {pos}: {invariant} — {detail}");
        FerriteError::Config(format!(
            "invariant violated at pos {pos}: {invariant} — {detail}"
        ))
    }

    /// **(1)** `s.ids == emitted.last()` — the token the NEXT round's `step_body`
    /// embeds is the last token this round emitted. `step_body` reads `s.ids`
    /// (the `token` argument is only the engram fallback), and a round advances
    /// `pos_ctr` by `emitted.len()`, so a missing/stale write-back makes the next
    /// round embed a token `k_acc` positions behind (the digit task's
    /// self-locking repetition). **Cost: one D2H, 4 B.**
    ///
    /// Runs at the END of all three arms, past the `sids_writeback()` write. NOTE
    /// the write-back itself is a separate A/B gate; with it OFF the invariant is
    /// deliberately not upheld, so `DSV41_INV_CHECK=1` must be paired with
    /// `DSV41_SIDS_WRITEBACK=1` for this check to pass on a `k_acc >= 1` round —
    /// and the report below says so, which is exactly the signal wanted.
    fn inv_ids(&self, pos: usize, emitted: &[u32]) -> Result<()> {
        if !inv_check() {
            return Ok(());
        }
        let Some(&last) = emitted.last() else {
            return Ok(());
        };
        let seen = self.dev.download_u32(self.s.ids.ptr as *const c_void)?;
        if seen != last {
            return Err(self.inv_fail(
                "s.ids == emitted.last()",
                pos,
                &format!("s.ids = {seen}, emitted.last() = {last} (write-back missing or stale?)"),
            ));
        }
        Ok(())
    }

    /// **(2)** `pos_ctr == p` — the DEVICE position counter equals the `p` the
    /// driver is stepping at. The driver strides by `emitted.len()`, so any drift
    /// between the two makes every later step write the wrong ring slot. The
    /// value is passed in because the caller already paid its D2H for the entry
    /// `debug_assert`; this check therefore adds **no** device traffic.
    fn inv_pos_ctr(&self, pos: usize, seen: usize) -> Result<()> {
        if !inv_check() {
            return Ok(());
        }
        if seen != pos {
            return Err(self.inv_fail(
                "pos_ctr == p (driver position)",
                pos,
                &format!("device pos_ctr = {seen}, driver p = {pos}"),
            ));
        }
        Ok(())
    }

    /// **(3)** `pos_rows[r] == pos_base + r` — `step_rows` just uploaded the row
    /// positions; row 0 must be `pos_base`. A mismatch means the verify's kernels
    /// read a position table that does not describe the block. **Cost: one D2H,
    /// 4 B** (row 0 only — the table is filled by one host-side
    /// `(0..m).map(|r| pos_base + r)`, so a wrong row-0 value is the only failure
    /// the construction can produce).
    fn inv_pos_rows_first(&self, pos_base: i32) -> Result<()> {
        if !inv_check() {
            return Ok(());
        }
        let seen = self.dev.download_u32(self.s.pos_rows.ptr as *const c_void)? as i32;
        if seen != pos_base {
            return Err(self.inv_fail(
                "pos_rows[0] == pos_base",
                pos_base.max(0) as usize,
                &format!("pos_rows[0] = {seen}, pos_base = {pos_base}"),
            ));
        }
        Ok(())
    }

    /// **(4)** `compress_len[l] == *clen[l]` — the HOST mirror of the committed-
    /// row count agrees with the DEVICE counter for the first compressor that
    /// owns one. The mirror drives `attention_rows`' `comp_len > 0` branch and the
    /// graph-capture steady-state guard, so a drift makes the next verify take a
    /// path the device counter does not agree with. **Cost: one D2H, 4 B.**
    ///
    /// Runs at the END of [`Self::dspark_commit`]: after the rollback + replay
    /// both counters must describe the same committed prefix.
    fn inv_compress_len(&self) -> Result<()> {
        if !inv_check() {
            return Ok(());
        }
        let Some(&l) = self.compress_sources().first() else {
            return Ok(());
        };
        let seen = self.dev.download_u32(
            (self.s.clen.ptr as *const u8).wrapping_add(l * 4) as *const c_void,
        )? as i32;
        let want = self.layers[l].compress_len as i32;
        if seen != want {
            return Err(self.inv_fail(
                "compress_len[l] == *clen[l]",
                want.max(0) as usize,
                &format!("layer {l}: host mirror = {want}, device *clen = {seen}"),
            ));
        }
        Ok(())
    }

    /// **(8)** every verify row's argmax is a REAL vocabulary index
    /// (`idx < vocab`). A value outside the vocabulary means the head wrote a row
    /// at the wrong pitch (the sliced/full arm disagreeing with the argmax) or the
    /// argmax read past its row — and the accept chain would then judge drafts
    /// against garbage. The `m` argmaxes are ALREADY on the host (`step_rows`'
    /// closing D2H), so this is **free**: it inspects the rows the caller is about
    /// to hand to the accept chain.
    fn inv_argmax_rows(&self, rows: &[u32], pos_base: i32) -> Result<()> {
        if !inv_check() {
            return Ok(());
        }
        let vocab = self.cfg.vocab_size as u32;
        if let Some((r, &a)) = rows.iter().enumerate().find(|(_, &a)| a >= vocab) {
            return Err(self.inv_fail(
                "argmax_r[r] < vocab_size",
                pos_base.max(0) as usize,
                &format!("verify row {r}: argmax = {a} >= vocab {vocab}"),
            ));
        }
        Ok(())
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

    /// The m-row collapse + norm pair of [`Self::layer_rows`], as ONE launch when
    /// `FUSE_B1` (default ON) is armed.
    ///
    /// `hc_collapse_norm` has a native `rows` dimension (`grid = rows`, one block
    /// per row, each with its own `blockDim`-wide reduction) and its phase 1/2
    /// statements are `hc_collapse` + `rmsnorm` statement for statement, so row
    /// `r` of the m-row launch is the m = 1 pair's output for row `r` — the same
    /// argument `layer()`'s `FUSE_B1` fold rests on, and the same kernel decode
    /// has been running under A/B since then. It also drops the `x_r` round trip.
    ///
    /// `pre` is the premix slot the collapse reads — the caller's `pa` for the
    /// attention block, this layer's attn_pre for the FFN block, exactly as the
    /// pre-fusion pair did — and the output lands in `s.xn_r` (the `s.x_r`
    /// intermediate is not needed by the fused form, but the pair's fallback
    /// still writes it, so the buffer stays live).
    ///
    /// `DSV41_BF16_TRUNCATE` deliberately does NOT ride along (2026-09-12, the
    /// baseline-breakage fix): the fused call passes `truncate = false`, so the
    /// verify chain keeps the raw pair's historical f32 behaviour no matter what
    /// the decode-side gate does. `hc_collapse_norm` was the ONLY form that
    /// carried the gate (as in `layer()`), and arming it here made the verify the
    /// first place inside a step where BF16_TRUNCATE applied; combined with the
    /// P0-series commits that broke the zero-Latin baseline (bisect: 050c7fd, the
    /// commit that first routed the verify's collapse through `hc_collapse_norm`;
    /// the GPU A/B confirmed `HC_VERIFY_FUSE=0` restores zero Latin). With the
    /// truncate removed the fused form is a statement-for-statement superset of
    /// the raw pair's f32 path, so arming `HC_VERIFY_FUSE=1` buys the launches
    /// back without touching the verify's numerics. `layer()`'s own
    /// `hc_collapse_norm` still honours the gate — the decode path was always the
    /// one the gate was validated on.
    ///
    /// The control flow is `layer()`'s, on purpose. `fused_done` says the fused
    /// front end already collapsed this block (its EARLY half IS
    /// `hc_collapse_norm`'s body), which can only be true when `FUSE_B1` armed
    /// that collapse half — so:
    ///
    /// * `FUSE_B1` on + `fused_done` ⇒ nothing to do;
    /// * `FUSE_B1` on + not done ⇒ `hc_collapse_norm` (the A1 fold);
    /// * `FUSE_B1` **off** ⇒ the raw pair, UNCONDITIONALLY. This is the arm that
    ///   makes the combination safe: with no collapse half in the front end it can
    ///   still report "fused" while having written no `xn_r`, and skipping the
    ///   pair there would leave the block's normalised input undefined.
    fn collapse_norm_rows(
        &mut self,
        pre: *const f32,
        w: *const f32,
        m: usize,
        fused_done: bool,
    ) -> Result<()> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        if Self::fuse_b1() {
            if fused_done {
                return Ok(());
            }
            if Self::hc_verify_fuse() {
                // `truncate = false` on purpose: the verify chain must keep the
                // raw pair's historical f32 behaviour (see the doc comment above
                // and `hc_verify_fuse`). Reading the gate here is what first put
                // BF16_TRUNCATE on the verify and broke the zero-Latin baseline.
                self.dev.hc_collapse_norm(
                    self.s.h_r.ptr as *mut f32,
                    pre,
                    w,
                    self.s.xn_r.ptr as *mut f32,
                    m as i32,
                    hc as i32,
                    dim as i32,
                    cfg.norm_eps,
                    false,
                )?;
                return Ok(());
            }
        }
        self.dev.hc_collapse(
            self.s.h_r.ptr as *const f32,
            pre,
            self.s.x_r.ptr as *mut f32,
            m as i32,
            hc as i32,
            dim as i32,
        )?;
        self.norm_rows(
            self.s.x_r.ptr as *const f32,
            w,
            self.s.xn_r.ptr as *mut f32,
            m,
            dim,
            cfg.norm_eps,
        )?;
        Ok(())
    }

    /// The m-row tail of [`Self::layer_rows`]: wait the front end's LATE half and
    /// then fold this block's `hc_post` back onto the residual stream.
    ///
    /// The join is `layer()`'s contract — the attention (or MoE) just consumed the
    /// EARLY half, so the LATE half's `comb`/`post` must be visible to the post —
    /// and is a no-op when no split was issued (A2 off, or the fused front end
    /// fell back to `hc_mixes`).
    ///
    /// `FUSE_C` (default ON) + the rows entry of the segment-C kernel make the
    /// post ONE launch that writes `h_r` in place instead of `hc_post` into the
    /// `h2_r` staging buffer + a D2D copy back. Bit-identical for any `rows`: same
    /// ascending-k `__fmaf_rn` chain over the same operands, and one thread owns
    /// one four-float column of one row, so `h_r` is both read and written only
    /// inside that thread's own columns (see `dsv41_hc_post_inplace_rows_kernel`).
    /// The staging pair stays as the fallback: `FUSE_C=0`, `HC_VERIFY_FUSE=0`, or
    /// a `.so` that predates the rows symbol.
    ///
    /// `x` is this block's producer output (`wo_out_r` for the attention block,
    /// `moe_out_r` for the FFN block) — the `[m, dim]` rows the post mixes with the
    /// `hc` residual copies.
    fn hc_post_rows(&mut self, x: *const f32, m: usize) -> Result<()> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        self.dev.hc_tail_join()?;
        if Self::fuse_c()
            && Self::hc_verify_fuse()
            && self.dev.supports_hc_post_inplace_rows()
        {
            self.dev.hc_post_inplace_rows(
                self.s.h_r.ptr as *mut f32,
                x,
                self.s.post_r.as_f32(),
                self.s.comb_r.as_f32(),
                m as i32,
                hc as i32,
                dim as i32,
            )?;
        } else {
            let hbytes = m * hc * dim * std::mem::size_of::<f32>();
            self.dev.hc_post(
                x,
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
        }
        Ok(())
    }

    /// One multi-row block: the m-row twin of [`Self::layer`].
    ///
    /// The premix threading is `layer()`'s, one step up in width: `pa` indexes the
    /// m-row premix the attention collapse reads, the attention mixes land in
    /// `pre_r` (slot 1) which the FFN collapse then consumes, and the FFN mixes
    /// land in `pre2_r` (slot 2) — the next layer's incoming premix. Returns 2.
    ///
    /// The hc chain is `layer()`'s, one step up in width as well:
    /// `hc_mixes_auto` (or the raw `hc_mixes` when `DSV41_HC_FRONT_ROWS` is off)
    /// at `rows = m`, then `collapse_norm_rows` / `hc_post_rows`. The three fusions
    /// (`FUSE_B1`'s `hc_collapse_norm`, `FUSE_C`'s in-place post, the tail split)
    /// are the ones decode has been running, called with `rows = m`.
    fn layer_rows(&mut self, layer: usize, m: usize, pos_base: i32, pa: usize) -> Result<usize> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let ld = &self.w.layers[layer];

        // ---------------- DSpark tap (layer INPUT) ----------------
        // The m-row twin of the hook at the top of `layer()`, and the SAME
        // capture point as the reference: before the block, i.e. reading the
        // layer's ATTENTION INPUT (`h_r` as the previous layer's `hc_post_rows`
        // left it, with this layer's engram write-back already applied by
        // `engram_apply_rows` in `step_rows`). `DSV41_TAP_INPUT=1` only; the
        // end-of-block twin below carries the same gate.
        //
        // The row bookkeeping is the end-of-block block's, unchanged — the tap is
        // relocated in the stream, never re-shaped: the gating (`spec_capture` /
        // `spec_tap_deferred`), the `m <= VERIFY_ROWS` contract, and the deferred
        // single-row staging buffer all stay exactly as they were, so
        // `note_ctx_rows`'s indexing contract is untouched.
        if (self.spec_capture || self.spec_tap_deferred) && Self::tap_input() {
            if let Some(slot) = cfg.dspark_target_slot(layer) {
                debug_assert!(
                    m <= VERIFY_ROWS,
                    "dspark tap_r: {m}-row block overruns the {VERIFY_ROWS}-row tap slot"
                );
                let (dst, rows) = if self.spec_tap_deferred {
                    (
                        (self.s.dspark_tap.ptr as *mut f32).wrapping_add(slot * dim),
                        1,
                    )
                } else {
                    (
                        (self.s.dspark_tap_r.ptr as *mut f32)
                            .wrapping_add(slot * VERIFY_ROWS * dim),
                        m,
                    )
                };
                self.dev.hc_collapse(
                    self.s.h_r.ptr as *const f32,
                    self.s.dspark_pre_mean.as_f32(),
                    dst,
                    rows as i32,
                    hc as i32,
                    dim as i32,
                )?;
            }
        }

        // ---------------- attention block ----------------
        // hc_mixes has a native `rows` dimension (`rows = m`, `hc_dim = hc*dim`),
        // with pre/post/comb row-major per row — the same shape the single-row call
        // uses with rows = 1.
        //
        // `DSV41_HC_FRONT_ROWS` routes this call through `hc_mixes_auto` instead,
        // i.e. the fused spread front end `layer()` runs, at `rows = m`: the same
        // `rows` contract plus the EARLY collapse/rmsnorm (so the collapse+norm
        // pair below is then skipped) and the dots + LATE half on the side stream.
        // `hc_mixes_auto` falls back to `hc_mixes(rows = m)` when the `.so` or its
        // gates decline, so the gate is a pure A/B switch on the same maths.
        //
        // The T1 fp8 staging stays on `quant_rows` here even on the fused path:
        // the tail kernel emits it in a SINGLE-ROW layout (`xq[c]`, no row base),
        // so a multi-row block must not hand it a buffer — the two nulls below.
        let (hc_nw, hc_pc, hc_out): (*const f32, *const f32, *mut f32) = if Self::fuse_b1() {
            (
                ld.attn_norm.as_ref().unwrap().as_f32(),
                self.premix_row_slot(pa).as_f32(),
                self.s.xn_r.ptr as *mut f32,
            )
        } else {
            (std::ptr::null(), std::ptr::null(), std::ptr::null_mut())
        };
        let hc_done = if Self::hc_front_rows() {
            self.hc_mixes_auto(
                self.s.h_r.ptr as *const f32,
                ld.hc_attn_fn.as_ref().unwrap().as_f32(),
                ld.hc_attn_scale.as_ref().unwrap().as_f32(),
                ld.hc_attn_base.as_ref().unwrap().as_f32(),
                self.s.pre_r.ptr as *mut f32, // slot 1: this layer's attn_pre
                self.s.post_r.ptr as *mut f32,
                self.s.comb_r.ptr as *mut f32,
                hc,
                dim,
                cfg.hc_sinkhorn_iters as i32,
                cfg.hc_eps,
                hc_nw,
                hc_pc,
                hc_out,
                cfg.norm_eps,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                m,
                // A2 verify arm: `false` on purpose, exactly as A1-a's
                // `collapse_norm_rows` — the m-row chain must keep the raw pair's
                // historical f32 behaviour. Reading `bf16_truncate()` here is the
                // same trap that routed `BF16_TRUNCATE` onto the verify for the
                // first time and broke the zero-Latin baseline; only `layer()`
                // reads the gate.
                false,
            )?
        } else {
            self.dev.hc_mixes(
                self.s.h_r.ptr as *const f32,
                ld.hc_attn_fn.as_ref().unwrap().as_f32(),
                ld.hc_attn_scale.as_ref().unwrap().as_f32(),
                ld.hc_attn_base.as_ref().unwrap().as_f32(),
                self.s.pre_r.ptr as *mut f32, // slot 1: this layer's attn_pre
                self.s.post_r.ptr as *mut f32,
                self.s.comb_r.ptr as *mut f32,
                m as i32,
                (hc * dim) as i32,
                hc as i32,
                cfg.hc_sinkhorn_iters as i32,
                cfg.hc_eps,
            )?;
            false
        };
        // The fused front end did the collapse when it ran (`hc_done`, only
        // possible with FUSE_B1 armed); otherwise the m-row collapse + norm pair
        // (`hc_collapse_norm` under the A1 gate). See `collapse_norm_rows` for why
        // the "FUSE_B1 off" arm must run the pair unconditionally.
        self.collapse_norm_rows(
            self.premix_row_slot(pa).as_f32(),
            ld.attn_norm.as_ref().unwrap().as_f32(),
            m,
            hc_done,
        )?;
        let attn_folded = self.attention_rows(layer, m, pos_base)?;
        // `DSV41_VERIFY_AR_FOLD` off (default): the plain AR left `wo_out_r`
        // holding the summed rows and this is the post that consumes it.
        // On: the AR's pubred epilogue already wrote `h_r`, so the standalone
        // post (and its `hc_tail_join`, which the fold did before its launch)
        // must NOT run — otherwise the mix would be applied twice.
        if !attn_folded {
            self.hc_post_rows(self.s.wo_out_r.ptr as *const f32, m)?;
        }

        // ---------------- FFN block ----------------
        let (ffn_nw, ffn_pc, ffn_out): (*const f32, *const f32, *mut f32) = if Self::fuse_b1() {
            (
                ld.ffn_norm.as_ref().unwrap().as_f32(),
                // the FFN collapses with THIS layer's attn_pre (slot 1), exactly
                // as `layer()` does
                self.s.pre_r.as_f32(),
                self.s.xn_r.ptr as *mut f32,
            )
        } else {
            (std::ptr::null(), std::ptr::null(), std::ptr::null_mut())
        };
        let ffn_done = if Self::hc_front_rows() {
            self.hc_mixes_auto(
                self.s.h_r.ptr as *const f32,
                ld.hc_ffn_fn.as_ref().unwrap().as_f32(),
                ld.hc_ffn_scale.as_ref().unwrap().as_f32(),
                ld.hc_ffn_base.as_ref().unwrap().as_f32(),
                self.s.pre2_r.ptr as *mut f32, // slot 2: ffn_pre -> next layer
                self.s.post_r.ptr as *mut f32,
                self.s.comb_r.ptr as *mut f32,
                hc,
                dim,
                cfg.hc_sinkhorn_iters as i32,
                cfg.hc_eps,
                ffn_nw,
                ffn_pc,
                ffn_out,
                cfg.norm_eps,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                m,
                // Same A2 verify arm as the attention block above: no
                // `BF16_TRUNCATE` on the m-row chain.
                false,
            )?
        } else {
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
            false
        };
        self.collapse_norm_rows(
            self.s.pre_r.as_f32(),
            ld.ffn_norm.as_ref().unwrap().as_f32(),
            m,
            ffn_done,
        )?;
        let moe_folded = self.moe_rows(layer, ld, m)?;
        // Same `DSV41_VERIFY_AR_FOLD` contract as the attention side.
        if !moe_folded {
            self.hc_post_rows(self.s.moe_out_r.ptr as *const f32, m)?;
        }

        // ---- the DSpark tap: `layer()`'s hook, all rows in ONE launch ----
        // The real-commit path (`dspark_spec_step`) hands the ACCEPTED PREFIX of
        // this block's target hiddens to the draft (`DsparkDev::note_ctx_rows`),
        // and unlike the single-row step it needs every row, not just the one the
        // step consumed. Same capture point as `layer()`'s DEFAULT arm (the
        // layer's COMPLETED output, after this layer's whole forward — attention
        // AND ffn — hence here, after the FFN's `hc_post` has landed back in
        // `h_r`), same `1/hc` mean (`hc_collapse` IS the weighted sum over the hc
        // copies). `DSV41_TAP_INPUT=1` SKIPS this block in favour of the
        // layer-INPUT hook at the top of this function — the reference's capture
        // point (`model.py:1261-1267`). Running both would let this one overwrite
        // the input tap with the output one, silently restoring the skew.
        //
        // This used to be m single-row calls. `hc_collapse` is row-INDEPENDENT —
        // `dsv41_glue.cu:301-317`: `out[t] = Σ_i pre[r*hc + i] * x[(r*hc + i)*dim
        // + c]` with `r = t/dim`, one FMA chain per output element and no
        // cross-row term — so the m-block form is bit-identical to m single-row
        // calls. Both strides already match the multi-row layout: the source rows
        // are contiguous at `r*hc*dim` in `h_r`, and the destinations are
        // contiguous at `(slot*VERIFY_ROWS + r)*dim` in `dspark_tap_r`. Folding
        // the loop saves m-1 launches per tap layer per verify (the same class of
        // redundant per-row split as the rmsnorm bypass in `dspark_dev.rs`).
        //
        // Gated on `spec_capture`: the shadow step's verify is rolled back whole,
        // so its rows are never committed and the extra launches would buy nothing.
        // `DSV41_LAZY_VERIFY` adds the DEFERRED variant: the lazy arm runs its
        // rows one at a time, so this block is always a SINGLE row that must land
        // at the row index the LAZY ARM knows (the `m = 1` block's own row is
        // index 0 of its slot, which would overwrite row 0 every time). In
        // deferred mode the row goes to the single-row staging buffer instead and
        // the arm copies it to row `i` outside the forward — which keeps this
        // launch's destination POINTER identical from row to row, so the one
        // captured `m = 1` graph serves them all (see `spec_tap_deferred`).
        if !Self::tap_input() && (self.spec_capture || self.spec_tap_deferred) {
            if let Some(slot) = cfg.dspark_target_slot(layer) {
                // (#7) The tap's row stride. The compile-time `assert!` at
                // `VERIFY_ROWS` pins `VERIFY_ROWS == DSPARK_DRAFTS + 1`; this is
                // the WRITE side of the same contract. The consumers
                // (`DsparkDev::note_ctx_rows`, `Self::carry_kept_tap`) index row
                // `r` at `slot * VERIFY_ROWS + r`; a block with more rows than
                // that slot holds would spill into the NEXT slot. Static.
                debug_assert!(
                    m <= VERIFY_ROWS,
                    "dspark tap_r: {m}-row block overruns the {VERIFY_ROWS}-row tap slot"
                );
                let (dst, rows) = if self.spec_tap_deferred {
                    // The staging buffer is `[DSPARK_TAP_SLOTS, dim]` — the same
                    // layout this hook writes for a single-row step, so `slot`
                    // indexes it identically. ONE row by construction (the lazy
                    // arm calls `step_rows` with a one-token block).
                    (
                        (self.s.dspark_tap.ptr as *mut f32).wrapping_add(slot * dim),
                        1,
                    )
                } else {
                    (
                        (self.s.dspark_tap_r.ptr as *mut f32)
                            .wrapping_add(slot * VERIFY_ROWS * dim),
                        m,
                    )
                };
                self.dev.hc_collapse(
                    self.s.h_r.ptr as *const f32,
                    self.s.dspark_pre_mean.as_f32(),
                    dst,
                    rows as i32,
                    hc as i32,
                    dim as i32,
                )?;
            }
        }
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
    ///
    /// Returns whether the wo AR took the `DSV41_VERIFY_AR_FOLD` fold, i.e. its
    /// pubred epilogue already wrote this block's `hc_post` onto `h_r` — in which
    /// case `layer_rows` must NOT run the standalone `hc_post_rows`.
    fn attention_rows(&mut self, layer: usize, m: usize, pos_base: i32) -> Result<bool> {
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

        // ---- q / kv projections: ONE launch per projection over the whole block
        // `lin()` is the per-row (quant1, gemm_fp8_mx) pair: `2m` quant launches
        // + `2m` GEMV launches, each GEMV re-streaming the same weight matrix.
        // The multi-row form (`quant_rows` + `proj_mrows`) quantises the
        // activation block ONCE and runs ONE `gemm_fp8_mrows` per projection —
        // that kernel is the weight-stationary form of `gemm_fp8_mx`'s **m == 1**
        // program, NOT its m > 1 tile path (a different summation), so row `r`
        // stays bit-identical to the per-row call. See the kernel header's C1-C6
        // in dsv41_kernels.cu; `proj_mrows` refuses when the .so/mode/gate cannot
        // take it, and the per-row loop below is the fallback.
        //
        // The loop is split into (wq_a, wkv) -> q norm -> wq_b so the q norm can
        // be ONE `n = m` launch. It is row-INDEPENDENT (the kernel gives every row
        // its own block and its own blockDim-sized reduction), `qr_r` is the
        // contiguous [m, ql] block, and row `r`'s output is bit-identical to the
        // per-row `n = 1` call it replaces.
        let mrows = self.dev.supports_gemm_fp8_mrows() && !Self::swapab() && m <= VERIFY_ROWS;
        // K1 (DSV41_ATTN_MROWS2, default OFF): the wq_a + wkv pair as ONE
        // two-family `gemm_fp8_mrows2` launch (see [`attn_mrows2`] and
        // [`Self::proj_mrows2`]) — the R2 replacement's first half. Checked
        // BEFORE R2's `lin2_gate` below: when both gates are on K1 runs (it is the
        // same-program fusion; R2's is the EAGER-program one that was found
        // corrupt). Both default OFF, so an unset environment is unchanged.
        let mrows2 =
            attn_mrows2() && self.dev.supports_gemm_fp8_mrows2() && !Self::swapab() && m <= VERIFY_ROWS;
        // R2 (DSV41_ATTN_LIN_FUSE, default OFF): at `m == 1` reuse EAGER's
        // single-row fusions — `lin2` (wq_a + wkv over ONE shared `xq`) and,
        // below, `lin_rope_norm` (norm + wq_b + rope in one launch). Both
        // kernels are already in the .so; the mrows化 never called them at
        // one row, which is 7 projection-family launches where EAGER pays 2.
        // See [`attn_lin_fuse`]. `m > 1` keeps the mrows path verbatim. The gate
        // is split (see [`AttnLinFuse`]): `lin2_gate` is R2's first half, the
        // wq_b half is `rope_norm_gate` below. Either one off simply leaves that
        // projection on the mrows sequence, so every combination of the two
        // degrades to a path that already existed. (Named `_gate` to stay clear
        // of the unrelated `rope_fuse()` gate above.)
        let lin_gate = attn_lin_fuse();
        let lin2_gate = lin_gate.lin2() && m == 1 && !Self::swapab();
        let rope_norm_gate = lin_gate.rope_norm() && m == 1 && !Self::swapab();
        let took_akv = if mrows2 {
            // K1: ONE two-family launch for wq_a + wkv. The activation is the
            // same single `quant_rows` the mrows branch below issues, so the fp8
            // bytes the fused GEMV consumes are exactly the ones two separate
            // `proj_mrows` launches would have consumed. On a decline
            // (`proj_mrows2` -> `Ok(false)`) the whole `took_akv` is false and the
            // per-row loop below runs — the same fallback the mrows branch has,
            // and bit-identical by construction.
            self.quant_rows(self.s.xn_r.ptr as *const f32, m, dim as i32)?;
            self.proj_mrows2(
                ld.wq_a.as_ref().unwrap().as_u8(),
                ld.wq_a_scale.as_ref().unwrap().as_u8(),
                self.s.qr_r.ptr as *mut f32,
                ql as i32,
                ql as i32,
                ld.wkv.as_ref().unwrap().as_u8(),
                ld.wkv_scale.as_ref().unwrap().as_u8(),
                self.s.kv_r.ptr as *mut f32,
                hd as i32,
                hd as i32,
                m,
                dim as i32,
            )?
        } else if lin2_gate {
            // The EAGER call verbatim, with this path's `_r` buffers as the
            // destinations: `quant1` + `gemm_fp8_mx2` write row 0 of `qr_r` /
            // `kv_r`, i.e. the bytes `proj_mrows` wrote for row 0. The GEMV
            // reads the device counter with `mul = 1, off = 0, step = 0`, and
            // for the block's single row that is `pos_base` — the same position
            // the per-row path passes as `off = r = 0`.
            self.lin2(
                self.s.xn_r.ptr as *const f32,
                dim as i32,
                ld.wq_a.as_ref().unwrap(),
                ld.wq_a_scale.as_ref().unwrap(),
                ql as i32,
                self.s.qr_r.ptr as *mut f32,
                ld.wkv.as_ref().unwrap(),
                ld.wkv_scale.as_ref().unwrap(),
                hd as i32,
                self.s.kv_r.ptr as *mut f32,
            )?
        } else if mrows {
            // the block's fp8 activation: `quant_kernel` is one thread-group per
            // (row, block), so row `r` here is the byte-for-byte `quant1(xr)`
            // the per-row loop issues. wq_a and wkv share it (they read the same
            // `xn_r` row), which alone removes `m` quant launches.
            self.quant_rows(self.s.xn_r.ptr as *const f32, m, dim as i32)?;
            let ok_a = self.proj_mrows(
                ld.wq_a.as_ref().unwrap().as_u8(),
                ld.wq_a_scale.as_ref().unwrap().as_u8(),
                self.s.qr_r.ptr as *mut f32,
                m,
                ql as i32,
                dim as i32,
                ql as i32,
            )?;
            let ok_kv = self.proj_mrows(
                ld.wkv.as_ref().unwrap().as_u8(),
                ld.wkv_scale.as_ref().unwrap().as_u8(),
                self.s.kv_r.ptr as *mut f32,
                m,
                hd as i32,
                dim as i32,
                hd as i32,
            )?;
            // Both are evaluated (no short-circuit) so a decline cannot leave one
            // projection on the multi-row path and the other on the per-row one;
            // re-running either form is idempotent (same values, same layout).
            ok_a && ok_kv
        } else {
            false
        };
        if !took_akv {
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
            }
        }
        // ---- VERIFY_FORK (DSV41_VERIFY_FORK): the attention dual chain ----
        // The m-row twin of `layer()`'s DUAL_CHAIN fork. `qr_r` and `kv_r` are
        // complete at this point (the block-wide `lin2`/`proj_mrows` pair, or the
        // per-row loop above) and are the two chains' LAST SHARED NODE: from here
        // the q chain below reads/writes only `qr_r`/`q_r`/`xq_r`/`xsc_r`, and
        // the kv half reads/writes only `kv_r` in place. The fork is therefore the
        // point at which the two are independent, and it is recorded on the MAIN
        // stream so the side stream cannot start before the projections land.
        // `kv_stream` is the main stream on every other path, so the serial
        // behaviour is byte-for-byte the old one (same launches, same order, same
        // stream). See [`verify_fork`] for the whole gate + the capture argument.
        let dual = verify_fork() && self.dev.supports_dual_chain();
        let kv_stream = if dual { self.dev.side_stream2() } else { self.dev.stream() };
        if dual {
            self.dev.dual_chain_fork()?;
        }
        // ---- VERIFY_FORK: the compressor's projections on the THIRD stream ----
        // Only `compress_proj_rows` is hoisted here — it is a pure function of
        // `s.xn_r` (final since the pre-attention norm) and writes this layer's
        // private `kvp_r`/`scp_r`, so it shares nothing with either chain and can
        // be issued at the SAME fork point as the dual chain, overlapping the
        // whole q/kv window. The pool+commit half deliberately stays inside the
        // per-row interleave below: the verify's read side must see row r's OWN
        // `*clen`, which the per-row commit publishes on the main stream (audit
        // defects #1/#2). The join below sits before `compress_rows_fused` / the
        // per-row `compress_row` — the first readers of `kvp_r`/`scp_r`.
        //
        // `spec_capture` excludes the arm because `compress_proj_rows`' snapshot
        // (`s.spec_snap_kvp`/`scp`) is a `memcpy_d2d` on the MAIN stream: on the
        // side stream it would race the very projections it saves.
        let comp_side = verify_fork()
            && cfg.compress_ratio(layer) > 0
            && cfg.is_kv_source(layer)
            && !self.spec_capture
            && !cublas_m1()
            && ld.comp_wkv.is_some()
            && ld.comp_norm.is_some()
            && self.dev.supports_compress_side();
        let mut comp_proj_side = false;
        if comp_side {
            self.dev.compress_side_fork()?;
            comp_proj_side = self.compress_proj_rows(layer, m, self.dev.side_stream3())?;
        }
        // R2 (DSV41_ATTN_LIN_FUSE): at `m == 1` the wq_b projection IS the EAGER
        // path's fused launch — the rmsnorm of the raw `qr_r` and the rope of
        // its own output folded into the GEMV's prologue/epilogue (NORM_FUSE,
        // default ON). It replaces `norm_rows + quant_rows + proj_mrows + rope`
        // with ONE launch, the second half of the 7 -> 2 reduction at one row.
        //
        // ⚠️ The fused launch leaves `qr_r` RAW (it normalises into shared
        // memory and never writes the row back), so it MUST run before the
        // `norm_rows` below — that call is what materialised the normalised
        // `qr_r` R2 handed the indexer's q half (`indexer_rows_one`'s `lin`;
        // `indexer_front_rows`'s `quant_rows` + `proj_mrows`). R2b hands that
        // half the RAW row instead (`s.qr_raw_r`), so it normalises inside its
        // own gemv prologue and the `norm_rows` is skipped — same bytes into the
        // GEMV's dot either way. The rope is then already on `q_r` and the
        // standalone rope below skips the row.
        let q_norm_fused = rope_norm_gate
            && self.lin_rope_norm(
                self.s.qr_r.ptr as *const f32,
                ld.q_norm.as_ref().unwrap().as_f32(),
                cfg.norm_eps,
                ql as i32,
                ld.wq_b.as_ref().unwrap(),
                ld.wq_b_scale.as_ref().unwrap(),
                (nlh * hd) as i32,
                self.s.q_r.ptr as *mut f32,
                rd as i32,
                hd as i32,
            )?;
        // R2b (DSV41_INDEXER_QR_RAW, default ON under R2): when the fused wq_b
        // launch above took (only ever at `m == 1`), `qr_r` is RAW and the
        // indexer's q half is the one reader that has to be told. Handing it the
        // flag lets it normalise inside its own gemv prologue — the `norm_rows`
        // below then has nothing left to do for it, which is the launch R2b
        // removes. `=0` keeps R2's compensating normalisation verbatim.
        let qr_raw_r = q_norm_fused && indexer_qr_raw();
        self.s.qr_raw_r.set(qr_raw_r);
        // q norm, in place: ALL m rows in ONE launch (the T2 epilogue's own
        // fallback pair — a plain rmsnorm into `qr`). Was m launches.
        // R2b: skipped when the fused wq_b left `qr_r` raw and the indexer
        // consumes it that way (see `qr_raw_r` above); `m == 1` on that arm, so
        // there is no second row whose normalisation anything could need.
        if !qr_raw_r {
            self.norm_rows(
                self.s.qr_r.ptr as *const f32,
                ld.q_norm.as_ref().unwrap().as_f32(),
                self.s.qr_r.ptr as *mut f32,
                m,
                ql,
                cfg.norm_eps,
            )?;
        }
        // wq_b: ONE launch for the whole block (see `proj_mrows`). The activation
        // is the normalised `qr_r` block; the output row is `nh*hd` wide but this
        // rank only writes its leading `nlh*hd`, which is why `out_stride` is a
        // separate kernel parameter rather than `n`.
        let took_b = if mrows && !q_norm_fused {
            self.quant_rows(self.s.qr_r.ptr as *const f32, m, ql as i32)?;
            self.proj_mrows(
                ld.wq_b.as_ref().unwrap().as_u8(),
                ld.wq_b_scale.as_ref().unwrap().as_u8(),
                self.s.q_r.ptr as *mut f32,
                m,
                (nlh * hd) as i32,
                ql as i32,
                (nh * hd) as i32,
            )?
        } else {
            false
        };
        if !took_b && !q_norm_fused {
            for r in 0..m {
                // wq_b: this rank's `nlh` heads, written at the row's base
                self.lin(
                    (self.s.qr_r.ptr as *const f32).wrapping_add(r * ql),
                    ql as i32,
                    ld.wq_b.as_ref().unwrap(),
                    ld.wq_b_scale.as_ref().unwrap(),
                    (nlh * hd) as i32,
                    (self.s.q_r.ptr as *mut f32).wrapping_add(r * nh * hd),
                )?;
            }
        }
        // ---- RoPE ----
        // The kernel computes `pos = *base * mul + off + row * step` for row `row`.
        // All `nlh` heads of one verify row share that row's position, so the q
        // rope rides the position in `off` with `step = 0` (exactly how the
        // single-row call keeps its heads at one position); the KV rope has one row
        // per position, so the block form works directly with `step = 1`.
        //
        // ROW-FOLD (DSV41_ROW_FOLD_ROPE=1 / DSV41_VERIFY_ROPE_MROWS=1): the m
        // launches collapse into one (`apply_rope_mrows`, an in-kernel ascending
        // r loop over `pos_rows[r]` — the same positions the host passed through
        // `off = r`, read from the device array this row loop already uses, and
        // the identical kernel the draft side's P3a a4 calls). Rows are
        // independent, so the result is bit-identical; `Ok(false)` keeps the loop
        // below. See [`verify_rope_mrows`] for why the q rope has its own gate.
        // R2: a row whose wq_b took the fused `lin_rope_norm` launch already
        // carries its rotation (`q_roped` = the fusion), so neither the
        // row-folded nor the per-row rope below may touch it again.
        let q_roped = q_norm_fused
            || (verify_rope_mrows()
                && self.dev.apply_rope_mrows(
                    self.s.q_r.ptr as *mut f32,
                    self.cos.as_f32(),
                    self.sin.as_f32(),
                    m as i32,
                    nlh as i32,
                    (nh * hd) as i32,
                    hd as i32,
                    rd as i32,
                    half,
                    self.s.pos_rows.ptr as *const std::os::raw::c_int,
                    false,
                )?);
        if !q_roped {
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
        }
        // VERIFY_FORK: the kv half of the dual chain. `kv_stream` is the side
        // stream when the fork above took (so the norm + rope run UNDER the q
        // chain instead of after it) and the main stream otherwise — same two
        // launches, same operands, same order, byte-for-byte the old serial pair.
        self.norm_rows_on(
            self.s.kv_r.ptr as *const f32,
            ld.kv_norm.as_ref().unwrap().as_f32(),
            self.s.kv_r.ptr as *mut f32,
            m,
            hd,
            cfg.norm_eps,
            kv_stream,
        )?;
        self.dev.apply_rope_on(
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
            kv_stream,
        )?;
        // VERIFY_FORK join: the kv half is fully issued — nothing below writes
        // `kv_r` until the per-row `ring_append`, its first consumer. Joining here
        // makes the finished `kv_r` visible to the main stream before the
        // interleave starts. A no-op when the fork did not take.
        if dual {
            self.dev.dual_chain_join()?;
        }

        // ---- the per-row interleave: append → window → compress → select →
        //      sparse attention, ONE ROW AT A TIME ----
        //
        // ⚠️ THE ORDER IS THE WHOLE POINT (verify-parity-audit, defects #1/#2).
        // The compressor, the indexer and `sparse_attn` all bound themselves by
        // the LIVE device counter `*clen` — dsv41_kernels.cu:947-948 derive the
        // attention's `n`/`topk` from it, :2805/:2825 the indexer's `n_pos` — and
        // this pass commits up to ceil(m / ratio) groups. Running the read side
        // BLOCK-WIDE (all m appends, then the compressor's m rows, then the m
        // indexer calls, then the m attention calls) therefore handed every row
        // the BLOCK-FINAL counter: row 0's compressed half included the latents
        // that rows 1..m-1 committed *after* it — its own future — while losing
        // the oldest group it should still have seen. Its argmax then degenerated
        // into replaying a token it had just attended to, which is exactly the
        // observed double token and the collapse of the accept rate (≈0.02).
        // Interleaving the four steps per row makes row r's bound its own: the
        // commits of rows `< r` are visible, those of rows `> r` do not exist yet.
        //
        // `*clen` needs no per-row SNAPSHOT pointer for that: the commit kernel
        // for row r is issued on this stream BEFORE row r's indexer/attention, so
        // by the time they read the counter it already carries row r's value.
        // What has to hold is only "row r's compress precedes row r's readers",
        // which the loop below guarantees by construction.
        //
        // The release shares one KV store across a group of layers; a consumer
        // then reads its owner's ring (and the owner's `idxs_r`, which the owner
        // filled row-interleaved — it runs earlier in the stack) and must not
        // append to it again, the same rule the single-row `ring_append` follows.
        let owner = if ring_owner_shared() { self.kv_owner(layer) } else { layer };
        let owns_kv = owner == layer;
        let ring_ptr = self.layers[owner].ring.ptr;
        let ist = win + cfg.index_topk;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let comp_ratio = cfg.compress_ratio(layer);
        let is_comp_src = comp_ratio > 0 && cfg.is_kv_source(layer);
        let is_idx_src = cfg.is_index_source(layer);
        let clen_owner = (self.s.clen.ptr as *const std::os::raw::c_int).wrapping_add(owner);
        // The compressor's PROJECTIONS (comp_wkv/comp_wgate, the ratio-1 `scp`
        // zeroing and the DSV41_SPEC scratch snapshot) are a pure function of
        // `s.xn_r`, which no later step of this layer writes, so they stay
        // block-wide; only the state/pool/commit half is interleaved below.
        //
        // VERIFY_FORK: when the side arm took, the projections were ALREADY
        // issued on `side_stream3` back at the dual chain's fork point; this is
        // their join — the point where the main stream needs `kvp_r`/`scp_r`
        // (`compress_rows_fused` below, and the per-row `compress_row` in the
        // interleave). The serial arm is the untouched call, byte for byte.
        let comp_proj = if comp_side {
            self.dev.compress_side_join()?;
            comp_proj_side
        } else if is_comp_src {
            self.compress_proj_rows(layer, m, self.dev.stream())?
        } else {
            false
        };
        // A consumer inherits the count its source published. The source owns the
        // compressor and its whole block is finished before this layer starts, so
        // the inherited count is constant across this block.
        let comp_len_inherited = if comp_ratio > 0 && !is_comp_src {
            self.source_compress_len(layer)
        } else {
            0
        };
        // COMPRESSOR-MROWS (`DSV41_COMPRESSOR_MROWS=1`, DEFAULT OFF): this layer's
        // `m` pool+commit pairs as ONE `dsv41_compressor_fused_mrows` launch. It
        // has to be issued HERE — after `compress_proj_rows` (whose `m` projection
        // rows it consumes) and before the per-row interleave, whose `compress_row`
        // calls it replaces — and it is what makes the read side below per-row
        // aware: the hoisted launch commits all `m` rows first, so the LIVE
        // counters it advances are block-final at every row, and each reader is
        // handed the per-row snapshot instead (`mrows_clen` / `clen_row`).
        //
        // A consumer layer owns no compressor but reads its OWNER's counters, so
        // it has to know how the owner ran: `mrows_compress_ok` is a pure function
        // of (layer, m, pos_base, cfg, device), so asking it about the owner is
        // exact and needs no per-layer flag that a rollback could leave stale.
        let mut comp_len_rows = [0usize; VERIFY_ROWS];
        let mrows_compress = if is_comp_src && comp_proj {
            self.compress_rows_fused(layer, m, pos_base, &mut comp_len_rows)?
        } else {
            false
        };
        // The owner's block, for a layer that reads its counters (a consumer, or a
        // source whose own hoist declined while its owner's took).
        let mrows_own_owner = if is_comp_src {
            mrows_compress
        } else {
            owner != layer && self.mrows_compress_ok(owner, m, pos_base)
        };
        // P1v (DSV41_VERIFY_OROPE, default ON): take the EAGER path's fused
        // sparse-attention launch per row instead of the `sparse_attn` +
        // `apply_rope` + `quant_fp8` triple. See [`verify_orope`] for why this is
        // the correctness alignment (the fusion is NOT bit-identical to the
        // triple) and why the fusion is the direction that must win.
        //
        // `orope_rows[r]` records whether row r's fused launch took; the two
        // consumers are the standalone o-rope below (a row the fused launch
        // already rotated must be SKIPPED, not rotated twice) and the o-quant
        // (`sparse_attn_orope` already emitted that row's fp8).
        let vo = verify_orope() && m <= VERIFY_ROWS;
        let mut orope_rows = [false; VERIFY_ROWS];
        // INDEXER-MROWS (`DSV41_INDEXER_MROWS=1`, DEFAULT OFF): the indexer's
        // block-wide front — `idx_wq_b` + its rope and `idx_weights` — paid ONCE
        // for the block instead of once per row. It is a pure function of
        // `s.qr_r` / `s.xn_r` / `s.pos_rows`, all of which are final here, so
        // hoisting it out of the interleave changes no reader's inputs: the only
        // consumer of `idx_q_r` / `idx_w_r` is this row's select, which still runs
        // at row r's own `*clen`, between row r's commit/publish and row r's
        // window append + `sparse_attn`. `(false, false)` (gate off, a layer the
        // front declines, an .so without the symbols) leaves BOTH halves of the
        // per-row front in place, so the default path is unchanged by
        // construction.
        let idx_front = if indexer_mrows() && is_idx_src {
            self.indexer_rows_m(layer, m)?
        } else {
            (false, false)
        };
        // W2-MROWS (`DSV41_ATTN_MROWS=1`, DEFAULT OFF): ONE m-row sparse-attention
        // launch for the whole block instead of one per row. [`attn_mrows`]'s header
        // carries the full argument; this is where its two PRECONDITIONS become
        // concrete host tests, because a wrong one is silent corruption and not a
        // slow path:
        //   * `world == 1` — the kernels index the q/out rows as `(row*h + hh)*d`
        //     while this call's rows live at pitch `nh*hd` (`h = nlh = nh/world`),
        //     so a `world > 1` batch needs a `row_pitch` argument the ABI does not
        //     have yet;
        //   * `pos_base + m - 1 < win` — the block must not have turned the ring
        //     over, or rows `0..m-2` read the block's own FUTURE rows as history
        //     while losing the oldest `m-1-r` entries (audit defect #2: the
        //     `verify_ring_win` fusion was reverted for exactly this).
        // ON-but-declined announces itself once (the bring-up failure mode this
        // project keeps re-hitting is a gate that is ON and changes nothing).
        let mrows_attn = if attn_mrows() && m > 1 && m <= ATT_MROWS_MAX_BM {
            if mrows_own_owner {
                // COMPRESSOR-MROWS fence: the block-wide attention call bounds every row
                // with `clen_rows[mm]`, which THIS arm hand-builds from the host mirror
                // and passes as a HOST pointer (a separate, pre-existing defect of the
                // `DSV41_ATTN_MROWS` arm). Under the compressor hoist the live counter
                // is block-final as well, so taking both would read row r's future with
                // no snapshot that a device kernel can use — refuse rather than corrupt.
                attn_mrows_decline(
                    "COMPRESSOR-MROWS hoisted this block's compressor: its per-row \
                     counter snapshots live on the device, the W2-MROWS read has none",
                );
                false
            } else if world != 1 {
                attn_mrows_decline("world != 1: the row pitch is nh*hd, the kernels index h*d");
                false
            } else if (pos_base as i64) + (m as i64) - 1 >= win as i64 {
                attn_mrows_decline(
                    "pos + m - 1 >= window: the ring has turned over, the block's own \
                     future rows would be read as history",
                );
                false
            } else {
                true
            }
        } else {
            false
        };
        // Row `r`'s bound for that launch — filled by the loop below with the SAME
        // per-row value the per-row call read (`comp_len`, the host mirror the
        // selection is already bounded by), never the block-final counter.
        let mut clen_rows = [0i32; VERIFY_ROWS];
        for r in 0..m {
            // ---- 1) ring append + THIS row's causal window ----
            // (audit defect #2: the window half.) `window_idxs(r)` must run after
            // row r's append (its own KV is in the window) but BEFORE row r+1's
            // append (which would overwrite the oldest slot row r's window still
            // enumerates — reading the block's future row as history instead). The
            // fused verify_ring_win appended the whole block first and derived the
            // indices from slot numbers, which broke exactly there: once
            // base+r >= window the `v > start_pos` filter never fires (v is a SLOT,
            // not a position) and every row but the last read the block's own
            // future rows while losing the oldest m-1-r history. The single-row
            // kernels keep the ring invariant row by row, so the window is
            // byte-identical to a single-row decode at each row's position.
            //
            // R3 (DSV41_RING_WIN_FUSE=1, default OFF here — see
            // [`verify_ring_win_fuse`]): the two single-row kernels below are the
            // EAGER step's `ring_append` + `window_idxs` pair, which EAGER issues
            // as ONE `dsv41_ring_win_fuse`. Calling that same kernel with
            // `pos_rows + r` is this row's `m == 1` form — same arithmetic, same
            // buffers, byte-identical — and halves the per-row ring+window cost
            // (12 -> 6 launches per layer at `m = 6`). The rollback is the pair,
            // unchanged, so a decline (gate OFF, or a `.so` without the symbol)
            // reproduces today's path launch for launch.
            if owns_kv {
                let rw_fused = verify_ring_win_fuse()
                    && self.dev.ring_win_fuse(
                        ring_ptr as *mut f32,
                        (self.s.kv_r.ptr as *const f32).wrapping_add(r * hd),
                        (self.s.pos_rows.ptr as *const std::os::raw::c_int).wrapping_add(r),
                        win as i32,
                        hd as i32,
                        (self.s.idxs_r.ptr as *mut i32).wrapping_add(r * ist),
                    )?;
                if !rw_fused {
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
            //
            // ---- 2) THIS row's compressor: pool + commit (one group per `ratio`
            //         positions) + the host mirror of the device counter ----
            // (audit defect #1: the compressed half — the read side below must not
            // see the block's block-final counter.)
            let committed = if is_comp_src && comp_proj {
                if mrows_compress {
                    // the hoisted launch already committed every row's group; the
                    // rule is the deterministic one the commit kernel applied.
                    (pos_base + r as i32 + 1) % (comp_ratio as i32) == 0
                } else {
                    self.compress_row(layer, r, pos_base)?
                }
            } else {
                false
            };
            // ---- 3) the compressed-half selection, bounded by THIS row ----
            // (audit defect #1.) `comp_len` is the row's own count: for a source it
            // is the host mirror the row's commit just advanced (the device counter
            // the kernels read is the same value, advanced on this stream by the
            // same launch), for a consumer the constant the source published.
            //
            // COMPRESSOR-MROWS: the hoisted launch advanced the mirror for the WHOLE
            // block up front, so the row's own value is the `comp_len_rows` entry it
            // filled (`compress_row` is not called at all on this arm).
            let comp_len = if is_comp_src {
                if mrows_compress {
                    comp_len_rows[r]
                } else {
                    self.layers[layer].compress_len
                }
            } else {
                comp_len_inherited
            };
            // COMPRESSOR-MROWS: the per-row counter snapshot this row's readers must
            // use instead of the live counter — the owner's, since a consumer layer
            // reads the OWNER's counter. `None` on the default path keeps the live
            // pointer, byte for byte.
            let clen_row = if mrows_own_owner {
                Some(
                    (self.s.clen_rows_r.ptr as *const std::os::raw::c_int)
                        .wrapping_add(owner * VERIFY_ROWS + r),
                )
            } else {
                None
            };
            if comp_len > 0 && is_idx_src {
                // The row's own top-k, plus — for a layer that OWNS the keys — the
                // index key of the group this row's commit just produced, published
                // at `*clen - 1` BEFORE the selection reads it (the single-row
                // order). Publishing per row instead of once per block is what
                // keeps every group of the block addressable: the key of a group is
                // written while `latent` still holds that group's pooled row.
                self.indexer_rows_one(layer, r, win, comp_len, committed, idx_front, clen_row)?;
            } else if !owns_kv && comp_len > 0 {
                // a non-index consumer reads the owner's selection, which the owner
                // (an index source, running earlier in the stack) already wrote into
                // this step's `idxs_r` compressed block — per row, interleaved with
                // its own appends
            } else if comp_len > 0 {
                // the owner has no indexer: the recency placeholder, for this row,
                // bounded by the same live counter
                self.dev.comp_placeholder(
                    (self.s.idxs_r.ptr as *mut i32).wrapping_add(r * ist + win),
                    clen_row.unwrap_or(clen_owner),
                    win as i32,
                    cfg.index_topk as i32,
                )?;
            }
            // ---- 4) sparse attention: THIS row, or (W2-MROWS) the whole block ----
            // The `b = 1, m = 1` shape is exactly the single-row `attention`'s, so
            // each row is verified against the plain engine row by row — the
            // strongest parity guarantee available.
            //
            // W2-MROWS: when the block-wide arm is on, this row's launch is
            // DEFERRED to the single m-row call after the loop and all this row
            // contributes is its bound. Everything before it (append → commit →
            // select) still runs per row in ascending order, so the interleave that
            // fixes defects #1/#2 is untouched: what made the old block-wide read
            // side wrong was the BLOCK-FINAL `*clen` and the BLOCK-FIRST appends,
            // and both are answered (the snapshot here, the non-wrapped-ring
            // precondition at the gate).
            if mrows_attn {
                clen_rows[r] = comp_len as i32;
            } else {
                //
                // P1v (DSV41_VERIFY_OROPE): the EAGER path runs the FUSED launch
                // (`sparse_attn_orope`: the inverse o-rope and the fp8 emission of the
                // roped row folded into the sparse-attention kernel) while this path
                // ran the triple — and the two are NOT bit-identical (the +18
                // `DSV41_DIFF_EAGER` mismatches vs ONE with the EAGER-side fusion
                // off). Same kernel on both sides is the alignment.
                //
                // ABI (checked against `sparse_attn_orope_kernel` phase 3): the call is
                // per row (`b = m = 1`, `h = nlh`, `d = hd`), and with the row's own
                // base pointers it writes the roped row to `out + hh*hd`, the fp8 to
                // `xq + hh*hd` and the scale to `xsc + hh*(hd/32)` — `[h, d]` fp8 +
                // `[h, d/32]` f32 relative to the pointers handed in. Passing
                // `xq_r + r*nlh*hd` / `xsc_r + r*(nlh*hd/32)` therefore lands EXACTLY
                // on the per-row compact packing the `quant_fp8` loop below used to
                // write (same bytes, same offsets), so the downstream
                // `wo_a_grouped_fp8` sees an unchanged layout.
                let oroped = vo
                    && self.dev.sparse_attn_orope(
                        (self.s.q_r.ptr as *const f32).wrapping_add(r * nh * hd),
                        ring_ptr as *const f32,
                        ld.attn_sink.as_ref().unwrap().as_f32(),
                        (self.s.idxs_r.ptr as *const i32).wrapping_add(r * ist),
                        (self.s.o_r.ptr as *mut f32).wrapping_add(r * nh * hd),
                        1,
                        1,
                        nlh as i32,
                        hd as i32,
                        clen_row.unwrap_or(clen_owner),
                        win as i32,
                        cfg.index_topk as i32,
                        scale,
                        self.cos.as_f32(),
                        self.sin.as_f32(),
                        // row `r` sits at `pos_ctr + r`, i.e. `mul = 1, off = r,
                        // step = 0` — the same `tt` the per-row `apply_rope` fallback
                        // below computes off the same counter (`hd`-independent here:
                        // the fused kernel's `hh*step` term is the rope's per-head
                        // step, which is 0 on both sides).
                        pos_ctr,
                        rd as i32,
                        half,
                        1,
                        r as i32,
                        0,
                        true,
                        (self.s.xq_r.ptr as *mut u8).wrapping_add(r * nlh * hd),
                        (self.s.xsc_r.ptr as *mut f32).wrapping_add(r * (nlh * hd / 32)),
                        // W2-MROWS inputs: inert at `b = m = 1` (the row is the whole
                        // launch, `idxs` is that row's own base, `row_step = 0` keeps
                        // the `off = r` formula).
                        std::ptr::null(),
                        0,
                        0,
                    )?;
                if !oroped {
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
                        clen_row.unwrap_or(clen_owner),
                        win as i32,
                        cfg.index_topk as i32,
                        scale,
                        std::ptr::null(),
                        0,
                    )?;
                }
                if vo {
                    orope_rows[r] = oroped;
                }
            }
        }

        // ---- W2-MROWS: the block's ONE sparse-attention launch ----
        // Only reached when BOTH preconditions held (see the gate above), i.e. the
        // block's appends cannot have clobbered any slot a row's window reads, so
        // each row sees the ring image its per-row call saw, and the row pitch is
        // `h*d`, so `q_r`/`o_r`/`xq_r`/`xsc_r` are indexed exactly as the per-row
        // calls indexed them (row `mm` of the block is `q_r + mm*nh*hd`, the same
        // address the loop above passed). `clen_rows[mm]` gives row `mm` its own
        // `topk`, and `idx_stride = ist` is the `idxs_r` row pitch — NOT `topk`,
        // which is per-row now that `clen` is (`ist = win + index_topk` is the
        // block's fixed pitch; a row with `clen < index_topk` has the smaller
        // `topk`, so the historical `(bb*m+mm)*topk` base would alias the next
        // row's block).
        if mrows_attn {
            // The fused form when the row loop would have taken it. A decline is
            // all-or-nothing (it is a property of the call SHAPE and the env, not
            // of a row), which is what lets `orope_rows` be filled uniformly below.
            let oroped = vo
                && self.dev.sparse_attn_orope(
                    self.s.q_r.ptr as *const f32,
                    ring_ptr as *const f32,
                    ld.attn_sink.as_ref().unwrap().as_f32(),
                    self.s.idxs_r.ptr as *const i32,
                    self.s.o_r.ptr as *mut f32,
                    1,
                    m as i32,
                    nlh as i32,
                    hd as i32,
                    clen_owner,
                    win as i32,
                    cfg.index_topk as i32,
                    scale,
                    self.cos.as_f32(),
                    self.sin.as_f32(),
                    pos_ctr,
                    rd as i32,
                    half,
                    1,
                    0,
                    0,
                    true,
                    self.s.xq_r.ptr as *mut u8,
                    self.s.xsc_r.ptr as *mut f32,
                    clen_rows.as_ptr(),
                    ist as i32,
                    // `row_step = 1`: `tt = pos_ctr + mm`, the per-row call's
                    // `off = r` — each row ropes at its own position.
                    1,
                )?;
            if !oroped {
                self.dev.sparse_attn(
                    self.s.q_r.ptr as *const f32,
                    ring_ptr as *const f32,
                    ld.attn_sink.as_ref().unwrap().as_f32(),
                    self.s.idxs_r.ptr as *const i32,
                    self.s.o_r.ptr as *mut f32,
                    1,
                    m as i32,
                    nlh as i32,
                    hd as i32,
                    clen_owner,
                    win as i32,
                    cfg.index_topk as i32,
                    scale,
                    clen_rows.as_ptr(),
                    ist as i32,
                )?;
            }
            if vo {
                // No row was found; the block's answer is uniform.
                for slot in orope_rows[..m].iter_mut() {
                    *slot = oroped;
                }
            }
        }

        // ---- the inverse o-rope, one row per call (the attention loop above
        //      interleaves this row's selection with the block's commits, so the
        //      rope stays outside it) ----
        //
        // P1v (DSV41_VERIFY_OROPE): a row whose FUSED launch took already carries
        // the ROTATED value — `sparse_attn_orope_kernel`'s phase 2 rotates the row
        // in shared memory before storing it — so this block must not touch it
        // again (a second inverse rotation is NOT idempotent). With the default
        // gate EVERY row is fused and the whole block is skipped; the ROW_FOLD
        // fold below stays available for the all-fallback case, and a partial
        // fallback (never observed: every decline in `dsv41_sparse_attn_orope` is
        // a property of the call SHAPE and the env, not of the row) degrades to the
        // per-row form, which is what keeps a mixed block correct instead of
        // double-roping it.
        let o_oroped_any = vo && orope_rows[..m].iter().any(|&b| b);
        let o_oroped_all = vo && orope_rows[..m].iter().all(|&b| b);
        // ROW-FOLD (DSV41_ROW_FOLD_ROPE=1): same launch fold as the q rope above,
        // with `inverse = true`. Identical shape (`x = s.o_r`,
        // `row_stride = nh*hd`, `rows = nlh`) and the same per-row position array.
        let o_roped = o_oroped_all
            || (!o_oroped_any
                && row_fold_rope()
                && self.dev.apply_rope_mrows(
                    self.s.o_r.ptr as *mut f32,
                    self.cos.as_f32(),
                    self.sin.as_f32(),
                    m as i32,
                    nlh as i32,
                    (nh * hd) as i32,
                    hd as i32,
                    rd as i32,
                    half,
                    self.s.pos_rows.ptr as *const std::os::raw::c_int,
                    true,
                )?);
        if !o_roped {
            for r in 0..m {
                if vo && orope_rows[r] {
                    continue; // the fused launch already roped this row
                }
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
        }

        // ---- grouped output projection + wo_b: ONE launch per phase ----
        let groups = cfg.o_groups;
        let hpg = nh / groups;
        let olg = cfg.o_lora_rank;
        let nlg = groups / world;
        let k = hpg * hd;
        let ol_total = groups * cfg.o_lora_rank;
        let ol_local = ol_total / world;
        // The grouped wo_a as ONE launch (`dsv41_wo_a_grouped_fp8`, the draft's
        // kernel): weight-stationary over the block's rows AND over the groups
        // (`groups` rides grid.y), so the per-(row, group) loop below becomes one
        // call. `a_stride = nlh*hd` is this rank's attention-output row pitch --
        // the group's k-element segment is `k` bytes into the row, and the rows
        // are NOT `k` apart (that difference is exactly why the plain mrows kernel
        // cannot express this call site). Bit-identical per (row, group) to the
        // `gemm_fp8_mx_or_swap` pair it replaces (its kernel header carries the
        // same C1-C6 argument).
        let took_o = if mrows {
            // the fp8 of EVERY row's attention output (the OROPE_Q epilogue's own
            // fallback), quantised with the row pitch `nlh*hd` -- the per-row
            // `quant1` wrote these same bytes into `s.xq` one row at a time,
            // shared by all of the row's group blocks.
            // ⚠️ ROW STRIDE FIX (verify-value-hunt's deterministic root cause):
            // `quant_rows`'s kernel derives the SOURCE row stride from `cols`
            // (`src = x + r*cols`), but `o_r`'s real row pitch is `nh*hd`
            // (8x `nlh*hd` under TP8's head split) — so with a single block call
            // every row r>=1 quantised bytes from inside ROW 0's unwritten tail
            // (garbage scratch), which is exactly the "row 0 always right,
            // r>=1 always wrong" signature the diff probe measured. Pack row by
            // row: source at its true pitch, destination compact (the downstream
            // wo_a_grouped_fp8 / proj_mrows layouts are unchanged).
            //
            // P1v: a row whose FUSED launch took was emitted by that launch itself
            // — `sparse_attn_orope_kernel`'s phase 3 quantises the ROTATED row
            // with the same per-32-block arithmetic into the same offsets — so it
            // needs no second pass here. The gate's default fuses every row and
            // this loop is skipped entirely.
            for r in 0..m {
                if vo && orope_rows[r] {
                    continue;
                }
                self.dev.quant_fp8(
                    (self.s.o_r.ptr as *const f32).wrapping_add(r * nh * hd),
                    (self.s.xq_r.ptr as *mut u8).wrapping_add(r * nlh * hd),
                    (self.s.xsc_r.ptr as *mut f32).wrapping_add(r * (nlh * hd / 32)),
                    1,
                    (nlh * hd) as i32,
                    32,
                    true,
                )?;
            }
            self.dev.wo_a_grouped_fp8(
                self.s.xq_r.ptr as *const u8,
                self.s.xsc_r.ptr as *const f32,
                ld.wo_a.as_ref().unwrap().as_u8(),
                ld.wo_a_scale.as_ref().unwrap().as_u8(),
                std::ptr::null(),
                self.s.wo_r.ptr as *mut f32,
                nlg as i32,
                m as i32,
                olg as i32,
                k as i32,
                (nlh * hd) as i32,
                ol_total as i32,
            )?
        } else {
            false
        };
        if !took_o {
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
            }
        }
        // wo_b is RowParallel: the input is split over the ranks, so this rank
        // writes its own partial and the AR below sums them. ONE launch for the
        // block (`proj_mrows`); the activation is this rank's `ol_local`-wide
        // slice of each `wo_r` row. It reads `wo_r`, which the wo_a phase above
        // fully wrote — both forms of that phase finish before this launches.
        let took_wob = if mrows {
            // ⚠️ ROW STRIDE FIX (same class as the o_r quant above):
            // `quant_rows` derives the SOURCE row stride from `cols`, but `wo_r`'s
            // real pitch is `ol_total` (8x `ol_local` under TP8) — a block call
            // quantised garbage from row 0's tail for every r>=1. Pack row by row.
            for r in 0..m {
                self.dev.quant_fp8(
                    (self.s.wo_r.ptr as *const f32).wrapping_add(r * ol_total),
                    (self.s.xq_r.ptr as *mut u8).wrapping_add(r * ol_local),
                    (self.s.xsc_r.ptr as *mut f32).wrapping_add(r * (ol_local / 32)),
                    1,
                    ol_local as i32,
                    32,
                    true,
                )?;
            }
            self.proj_mrows(
                ld.wo_b.as_ref().unwrap().as_u8(),
                ld.wo_b_scale.as_ref().unwrap().as_u8(),
                self.s.wo_out_r.ptr as *mut f32,
                m,
                dim as i32,
                ol_local as i32,
                dim as i32,
            )?
        } else {
            false
        };
        if !took_wob {
            for r in 0..m {
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
        }
        // ---- the attention all-reduce ----
        // The payload is the m rows of `wo_out_r` instead of one: the AR is a
        // byte-wise operation that sums the ranks in ascending order, so each row's
        // per-element sum is the same values in the same order as the single-row AR
        // of that row would have been.
        //
        // `DSV41_VERIFY_AR_FOLD` (default OFF): the hc-post fold now HAS a
        // multi-row epilogue (`ferrite_p2p_ar_v5_hcpost_rows`) — it consumes the
        // AR result per row (`row = (i4<<2) / dim`, `hc_h % 4 == 0` keeps a
        // float4 inside one row) and writes `h_r` in the same ascending-k
        // `__fmaf_rn` order `hc_post_inplace_rows` does. When it runs the caller
        // (`layer_rows`) skips its `hc_post_rows`; when it declines the AR is the
        // untouched plain one and the standalone post runs as before.
        let mut folded = false;
        if let Some(c) = self.comm.clone() {
            folded = self.ar_hc_post_fold_rows(
                &c,
                self.s.wo_out_r.ptr as *mut std::ffi::c_void,
                fb(m * dim),
                m,
            )?;
            if !folded {
                c.all_reduce_inplace(self.s.wo_out_r.ptr as *mut std::ffi::c_void, fb(m * dim))?;
            }
            c.end_round();
        }
        Ok(folded)
    }

    /// Publish ONE index key: the roped `wk` projection of `layer`'s pooled
    /// compressor latent, rms-normed, written into the owner's `index_k` at the
    /// slot its DEVICE compressed-row counter names.
    ///
    /// Extracted from the two callers that used to inline it — the single-row
    /// [`Self::indexer`] and the verify's per-row [`Self::indexer_rows_one`] —
    /// because the DSpark spec commit's replay ([`Self::compress_replay`]) needs the
    /// SAME key for every group it re-commits. The verify now publishes one key per
    /// COMPLETED row, from inside the per-row interleave (so every group the block
    /// commits gets its key before the row that produced it selects); the older
    /// block-wide form could only ever name the block's LAST group, leaving the
    /// earlier groups' keys stale — and `indexer_topk` reads every row `< clen`, so
    /// a partially committed block would select against keys never written.
    ///
    /// Returns `false` when the layer carries no index-key weights, so a caller
    /// can skip it without duplicating the weight lookup. `&self`: every launch
    /// is read-only on the chain (the destination slots are the device buffers).
    fn publish_index_key(&self, layer: usize) -> Result<bool> {
        let clen_layer = (self.s.clen.ptr as *const std::os::raw::c_int).wrapping_add(layer);
        let latent_src = self.layers[layer].latent.ptr as *const f32;
        self.publish_index_key_from(layer, latent_src, clen_layer)
    }

    /// [`Self::publish_index_key`] for a HOISTED block (COMPRESSOR-MROWS): the
    /// source row `r`'s POOLED latent and row `r`'s OWN counter, instead of the
    /// shared `latent` and the live `*clen`.
    ///
    /// Both substitutions are forced: a block-wide compressor launch leaves
    /// `latent` holding the block's LAST pooled row and the live counter at the
    /// block-final value, so the per-row publishes that follow would all write the
    /// last group's key to the last group's slot — every earlier group of the block
    /// would keep whatever key it had. `latent_rows_r + r*hd` and
    /// `clen_rows_r[layer][r]` are the values the per-row interleave published from
    /// (row r's own pool output and the counter between row r's commit and row
    /// r+1's), so the bytes and the destination slots are identical.
    fn publish_index_key_rows(&self, layer: usize, r: usize) -> Result<bool> {
        let hd = self.cfg.head_dim;
        let latent_src = (self.s.latent_rows_r.ptr as *const f32).wrapping_add(r * hd);
        let clen_layer = (self.s.clen_rows_r.ptr as *const std::os::raw::c_int)
            .wrapping_add(layer * VERIFY_ROWS + r);
        self.publish_index_key_from(layer, latent_src, clen_layer)
    }

    /// The shared body of the two publish entries: `latent_src` is the pooled row
    /// to project and `clen_layer` the counter the destination slot is derived
    /// from (`*clen_layer - 1`, the same expression both callers used inline).
    fn publish_index_key_from(
        &self,
        layer: usize,
        latent_src: *const f32,
        clen_layer: *const std::os::raw::c_int,
    ) -> Result<bool> {
        let cfg = self.cfg;
        let ld = &self.w.layers[layer];
        let (Some(wk), Some(kn)) = (ld.idx_wk.as_ref(), ld.idx_k_norm.as_ref()) else {
            return Ok(false);
        };
        let idx_hd = cfg.index_head_dim.max(1);
        let rd = cfg.rope_head_dim;
        let ratio = cfg.compress_ratio(layer).max(1);
        self.lin_bf16(
            latent_src,
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
        Ok(true)
    }

    /// The m-row entry of the indexer family (`DSV41_INDEXER_MROWS=1`, default
    /// OFF): pays [`Self::indexer_front_rows`] ONCE for the block and hands the
    /// two `*_done` flags to the per-row select ([`Self::indexer_rows_one`]) that
    /// follows, so a fused half is never recomputed.
    ///
    /// WHY THE SELECT HALF IS NOT FUSED HERE. `dsv41_indexer_topk` really does
    /// carry the m-row shape (`dim3 grid(m, b)`, `cl = lens[mm]` per row,
    /// `dsv41_kernels.cu:2869/2874`), so the plan's "pure wiring" reading of the
    /// grid is correct as far as it goes — but an m-row CALL from this loop site
    /// is not expressible with the current launcher/ABI, for three reasons that
    /// only show up once you leave `b = m = 1`:
    ///
    ///   1. **the scan bound.** The kernel takes `n_pos` and then OVERRIDES it
    ///      with `*lens` (`:2854`), i.e. row 0's count. A per-row `lens[m]` with
    ///      differing counts (verify's index-source layers commit one group per
    ///      row, so `clen` ascends with `r`) therefore needs `n_pos` to be the
    ///      MAXIMUM of the rows — otherwise rows 1..m-1 are silently clamped to
    ///      row 0's count and lose their newest groups.
    ///   2. **the output row stride.** The kernel writes `out[row * cols + i]`
    ///      with `cols = min(topk, n_pos)` — a RUNTIME value — while `idxs_r`'s
    ///      row pitch is the fixed `win + index_topk`. That mismatch is exactly
    ///      why `indexer_rows_one` is called `b = m = 1` today (see its own note);
    ///      a single m-row call needs the stride as an explicit argument.
    ///   3. **causal order.** The per-row counts only exist after EVERY row of the
    ///      block has committed, so a single call can only be issued after the
    ///      block's appends — and this loop's appends are what destroy the window
    ///      row `r`'s `sparse_attn` is about to read (audit defect #2: the window
    ///      ring is exactly `window` slots). Fusing the select therefore moves
    ///      `sparse_attn` too, which is the ATTENTION family's restructure (a
    ///      device-written per-row `clen` snapshot + an explicit `idx_stride` +
    ///      the `b*m` sparse-attn arm), not a wiring change in this one.
    ///
    /// So this entry fuses what is genuinely uncoupled — the front — and leaves
    /// the select where its `*clen` is row `r`'s own.
    fn indexer_rows_m(&mut self, layer: usize, m: usize) -> Result<(bool, bool)> {
        self.indexer_front_rows(layer, m)
    }

    /// The indexer FRONT for a whole block: the m-row twin of the first three
    /// launches of [`Self::indexer_rows_one`] — `idx_wq_b`'s fp8 projection
    /// (`s.qr_r` → `s.idx_q_r`) plus its rope at `pos_rows[r]`, and `idx_weights`'
    /// bf16 GEMV (`s.xn_r` → `s.idx_w_r`) — in four launches instead of `4*m`
    /// (16 of the family's 20 front launches per layer).
    ///
    /// Returns `(q_done, w_done)`: `q_done` means `idx_q_r` rows `0..m` are
    /// projected AND roped, `w_done` means `idx_w_r` rows `0..m` are filled. A
    /// `false` half is NOT a failure — the caller passes the flags to
    /// [`Self::indexer_rows_one`], which redoes exactly that half per row. Every
    /// decline (a layer without the weights, `swapab`, `cublas_m1`, an .so
    /// without a symbol, a shape outside a kernel's bound) degrades to the
    /// per-row front verbatim, so the arm is strictly additive.
    ///
    /// The two halves are INDEPENDENT, so one may fuse while the other does not:
    /// `gemv_bf16_v2_mrows` declines on its own when the per-row `lin_bf16` would
    /// take v1 or cuBLAS (different accumulation orders, no parity to claim).
    fn indexer_front_rows(&self, layer: usize, m: usize) -> Result<(bool, bool)> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let ql = cfg.q_lora_rank;
        let idx_nh = cfg.index_n_heads.max(1);
        let idx_hd = cfg.index_head_dim.max(1);
        let rd = cfg.rope_head_dim;
        let half = (cfg.rope_head_dim / 2) as i32;
        let ld = &self.w.layers[layer];
        // The same presence gate `indexer_rows_one` applies: an indexer without
        // `idx_wq_b` / `idx_weights` runs no row at all, so there is nothing to
        // fuse and the per-row entry declines identically.
        let (Some(idx_wq_b), Some(idx_wq_b_s), Some(wp)) = (
            ld.idx_wq_b.as_ref(),
            ld.idx_wq_b_scale.as_ref(),
            ld.idx_weights.as_ref(),
        ) else {
            return Ok((false, false));
        };
        if m == 0 || m > VERIFY_ROWS {
            return Ok((false, false));
        }
        // ---- q half: ONE `quant_rows` + `proj_mrows`, then ONE rope ----
        // `quant_rows` writes `s.xq_r`/`s.xsc_r` at the `cols = ql` pitch that
        // `proj_mrows` reads back — the pair `attention_rows` already uses for
        // wq_b/wo_b, so each row's fp8 bytes are `quant1`'s and the GEMM is the
        // mrows kernel's per-row identical chain. The rope is the same kernel
        // `DSV41_ROW_FOLD_ROPE` uses, with the indexer's geometry (`rows =
        // index_n_heads` head rows of `index_head_dim`, row pitch `idx_nh*idx_hd`,
        // row r at `pos_rows[r]` — the `mul=1, off=r, step=0` the per-row call
        // evaluates off the counter).
        let mut q_done = false;
        // R2b (DSV41_INDEXER_QR_RAW, default ON under R2): `attention_rows`' fused
        // wq_b launch leaves `qr_r` RAW, so the `quant_rows` below must NOT read
        // it — staging a raw row would dot the wrong values. The fused
        // `lin_rope_norm` launch does the whole pair AND the rope in ONE launch,
        // and it is the `m == 1` program (the flag is only set on that arm) — the
        // same launch EAGER's `indexer()` takes for exactly this reason. A decline
        // normalises the row itself, so the fallback below still quantises
        // normalised values.
        let qr_raw = self.s.qr_raw_r.get();
        self.s.qr_raw_r.set(false);
        if qr_raw {
            debug_assert!(m == 1, "qr_raw_r is only set by the m == 1 fused wq_b arm");
            q_done = self.lin_rope_norm(
                self.s.qr_r.ptr as *const f32,
                ld.q_norm.as_ref().unwrap().as_f32(),
                cfg.norm_eps,
                ql as i32,
                idx_wq_b,
                idx_wq_b_s,
                (idx_nh * idx_hd) as i32,
                self.s.idx_q_r.ptr as *mut f32,
                rd as i32,
                idx_hd as i32,
            )?;
            if !q_done {
                self.norm_rows(
                    self.s.qr_r.ptr as *const f32,
                    ld.q_norm.as_ref().unwrap().as_f32(),
                    self.s.qr_r.ptr as *mut f32,
                    m,
                    ql,
                    cfg.norm_eps,
                )?;
            }
        }
        // The fused arm above already rotated `idx_q_r`'s row: `pos_ctr` with
        // `off = 0` at `m == 1` is `pos_rows[0]`, the position the mrows rope
        // below reads, so the rope goes with the rest of the pair it replaced.
        if !q_done {
            self.quant_rows(self.s.qr_r.ptr as *const f32, m, ql as i32)?;
            if self.proj_mrows(
                idx_wq_b.as_u8(),
                idx_wq_b_s.as_u8(),
                self.s.idx_q_r.ptr as *mut f32,
                m,
                (idx_nh * idx_hd) as i32,
                ql as i32,
                (idx_nh * idx_hd) as i32,
            )? {
                q_done = self.dev.apply_rope_mrows(
                    self.s.idx_q_r.ptr as *mut f32,
                    self.cos.as_f32(),
                    self.sin.as_f32(),
                    m as i32,
                    idx_nh as i32,
                    (idx_nh * idx_hd) as i32,
                    idx_hd as i32,
                    rd as i32,
                    half,
                    self.s.pos_rows.ptr as *const std::os::raw::c_int,
                    false,
                )?;
            }
        }
        // ---- w half: ONE bf16 GEMV for all m rows ----
        // `lin_bf16` takes `gemv_bf16_v2` for this shape (`index_n_heads` is far
        // under `GEMV_V2_MAX_N`) and `gemv_bf16_v2_mrows` IS that program's m-row
        // form (per-row independent accumulator, `tests_gate_mrows.cu`). The entry
        // declines by itself when v2 is not what the per-row path would run;
        // `cublas_m1` is the one branch it cannot see, so it is checked here.
        let w_done = !cublas_m1()
            && self.dev.gemv_bf16_v2_mrows(
                wp.ptr() as *const c_void,
                self.s.xn_r.ptr as *const f32,
                self.s.idx_w_r.ptr as *mut f32,
                m as i32,
                idx_nh as i32,
                dim as i32,
            )?;
        Ok((q_done, w_done))
    }

    /// ONE ROW of the multi-row indexer: the per-row query/weights/top-k triple of
    /// [`Self::indexer`], plus — when `publish_key` — the index key of the group
    /// this row's compressor just committed.
    ///
    /// Called from [`Self::attention_rows`] INSIDE the per-row interleave, right
    /// after row `r`'s [`Self::compress_row`] and before row `r`'s `sparse_attn`,
    /// so the live device counter (`idx_lens`, and the same counter `sparse_attn`
    /// reads) is row `r`'s own. Handing the whole block the block-FINAL counter is
    /// precisely the defect the interleave removes: row 0 selected against the
    /// latents rows 1..m-1 had committed after it — its own future — and lost the
    /// oldest group it should have seen.
    ///
    /// `indexer_topk`'s output stride is `cols = min(topk, n_pos)` — a RUNTIME value
    /// derived from the live compressed count — so the kernel cannot be handed the
    /// `window + index_topk` row stride `idxs_r` uses. It is therefore called with
    /// `m = 1` (`b = 1`), where the stride never matters; every offset is the row's
    /// base plus the caller's `offset`.
    ///
    /// `front = (q_done, w_done)` is [`Self::indexer_front_rows`]'s result for
    /// this block (`DSV41_INDEXER_MROWS`): a half that the block-wide front
    /// already produced is skipped here, because rows `0..m` of `idx_q_r` /
    /// `idx_w_r` hold byte-for-byte what the per-row launches below would write.
    /// `(false, false)` is the historical per-row behaviour verbatim.
    ///
    /// `mrows_clen` (COMPRESSOR-MROWS): `Some(ptr)` when this layer's or its
    /// owner's block compressor was HOISTED into one launch — `ptr` is then the
    /// owner's per-row counter snapshot (`clen_rows_r[owner][r]`), which is the
    /// value the live `*clen` carried at this row's own read point. `None` keeps
    /// the live counter pointer, i.e. the default path verbatim.
    fn indexer_rows_one(
        &mut self,
        layer: usize,
        r: usize,
        offset: usize,
        comp_len: usize,
        publish_key: bool,
        front: (bool, bool),
        mrows_clen: Option<*const std::os::raw::c_int>,
    ) -> Result<bool> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let ql = cfg.q_lora_rank;
        let idx_nh = cfg.index_n_heads.max(1);
        let idx_hd = cfg.index_head_dim.max(1);
        let rd = cfg.rope_head_dim;
        let half = (cfg.rope_head_dim / 2) as i32;
        let ld = &self.w.layers[layer];
        // `idx_wk`/`idx_k_norm` are consumed by `publish_index_key` — the key
        // publishing below AND the spec commit's replay — but an indexer without
        // them must still decline here, hence the presence check.
        let (Some(idx_wq_b), Some(idx_wq_b_s), Some(_), Some(_), Some(wp)) = (
            ld.idx_wq_b.as_ref(),
            ld.idx_wq_b_scale.as_ref(),
            ld.idx_wk.as_ref(),
            ld.idx_k_norm.as_ref(),
            ld.idx_weights.as_ref(),
        ) else {
            return Ok(false);
        };
        // ---- key publishing, PER ROW ----
        // `indexer_owns_k` decides whether this layer publishes index keys for the
        // groups its own compressor produces; the roped key lands in the owner's
        // `index_k` at the slot its DEVICE counter names (`*clen - 1`).
        //
        // Only a row that COMPLETED a group has a new latent to publish: the pool's
        // mode-2 branch writes no latent on a non-completing position, so `latent`
        // still holds the previous group's pooled row and re-publishing would
        // rewrite the very same bytes — skipping it is equivalent to the single-row
        // path, which republishes unconditionally. Publishing per row (instead of
        // once per block, which could only ever name the block's LAST group) is what
        // makes EVERY group of the block addressable, and it now happens BEFORE this
        // row's own selection reads `index_k[.. *clen]`.
        //
        // `verify_recording` (the GRAPH capture, `DSV41_VERIFY_GRAPH=1`) FORCES the
        // launch. `committed` is a HOST value that flips with `pos_base mod ratio`,
        // and a capture bakes the launch SEQUENCE it observes: recording at an even
        // `pos_base` left every odd row's publish out of the graph, so a replay at
        // an odd `pos_base` selected against `index_k` slots no launch had written
        // (and the mirror case). Recording the publish for EVERY row makes the
        // sequence parity-independent, and the replay stays correct because the
        // extra launches are byte-idempotent: a row that did not complete a group
        // leaves `latent` and the device `*clen` untouched, so the publish writes
        // the group it already wrote (`*clen - 1`, same bytes) — precisely what the
        // single-row reference does on every non-completing step. `committed` still
        // gates the DIRECT path, whose launch sequence — and therefore whose
        // device effects — are unchanged.
        if cfg.indexer_owns_k(layer) && (publish_key || self.verify_recording) {
            // COMPRESSOR-MROWS: a hoisted block leaves the shared `latent` at the
            // block's LAST pooled row, so the per-row publish has to name its own
            // row's copy and its own counter.
            if mrows_clen.is_some() {
                self.publish_index_key_rows(layer, r)?;
            } else {
                self.publish_index_key(layer)?;
            }
        }
        // ---- this row's query, per-head weights and selection ----
        // The q_lora stream is already normed (attention_rows ran the plain
        // rmsnorm), so `idx_wq_b` uses the plain `lin`/`lin_bf16` pair.
        //
        // INDEXER-MROWS: a half the block-wide front already produced is skipped
        // — `indexer_front_rows` wrote rows `0..m` of `idx_q_r` (projected AND
        // roped) and/or `idx_w_r`, and row `r` of those is exactly what the
        // per-row launches below would write. The SELECT below (publish + topk) is
        // deliberately NOT skipped: it must read row `r`'s own `*clen`.
        let (q_done, w_done) = front;
        if !q_done {
            let idxq = (self.s.idx_q_r.ptr as *mut f32).wrapping_add(r * idx_nh * idx_hd);
            // R2b (DSV41_INDEXER_QR_RAW, default ON under R2): `attention_rows`'
            // fused wq_b launch left `qr_r` RAW, so `lin` below must not quantise
            // this row as it stands — its GEMV prologue (`lin_rope_norm`) is the
            // one that normalises + quantises the raw row and rotates the result,
            // all in ONE launch (the launch R2's compensating `norm_rows` used to
            // stand in for). The flag is only ever set on the `m == 1` arm, so
            // `r == 0` and the launch's `off = 0` position is this row's own; a
            // decline materialises the norm first, exactly like EAGER's
            // `indexer()`, so the fallbacks still quantise normalised values.
            let qr_raw = self.s.qr_raw_r.get();
            self.s.qr_raw_r.set(false);
            let mut roped = false;
            if qr_raw {
                debug_assert!(r == 0, "qr_raw_r is only set by the m == 1 fused wq_b arm");
                roped = self.lin_rope_norm(
                    (self.s.qr_r.ptr as *const f32).wrapping_add(r * ql),
                    ld.q_norm.as_ref().unwrap().as_f32(),
                    cfg.norm_eps,
                    ql as i32,
                    idx_wq_b,
                    idx_wq_b_s,
                    (idx_nh * idx_hd) as i32,
                    idxq,
                    rd as i32,
                    idx_hd as i32,
                )?;
                if !roped {
                    self.dev.rmsnorm(
                        (self.s.qr_r.ptr as *const f32).wrapping_add(r * ql),
                        ld.q_norm.as_ref().unwrap().as_f32(),
                        (self.s.qr_r.ptr as *mut f32).wrapping_add(r * ql),
                        1,
                        ql as i32,
                        cfg.norm_eps,
                    )?;
                }
            }
            if !roped {
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
            }
        }
        if !w_done {
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
        // COMPRESSOR-MROWS: the owner's hoisted block left its LIVE counter at the
        // block-final value, so the selection reads the owner's per-row snapshot
        // instead — the count this row's own read point had.
        let idx_lens = match mrows_clen {
            Some(p) => p,
            None => (self.s.clen.ptr as *const std::os::raw::c_int).wrapping_add(key_owner),
        };
        let idx_k_ptr = self.layers[key_owner].index_k.ptr as *const f32;
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
        Ok(true)
    }

    /// The PROJECTION half of the multi-row compressor: this layer's per-row
    /// `comp_wkv` (`/` `comp_wgate`) projections, the ratio-1 `scp` zeroing and the
    /// DSV41_SPEC scratch snapshot, all m rows in one go.
    ///
    /// Split out of `compress_rows` so [`Self::attention_rows`] can interleave the
    /// state/pool/commit half PER ROW ([`Self::compress_row`]) while paying the
    /// projections once. They are a pure function of `s.xn_r` — which no later step
    /// of this layer writes — so their position inside the block is irrelevant, and
    /// running them up front is what lets row r's commit see row r's own
    /// projections without re-reading `xn_r` per row.
    ///
    /// Returns `false` when the layer carries no compressor (`comp_wkv`/`comp_norm`
    /// absent): no row runs at all then and the host mirror is left untouched, which
    /// is the early return `compress_rows` used to take.
    fn compress_proj_rows(
        &self,
        layer: usize,
        m: usize,
        st: ferrite_kernel::devrt::CuStream,
    ) -> Result<bool> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let hd = cfg.head_dim;
        let ld = &self.w.layers[layer];
        let (Some(wkv), Some(_)) = (ld.comp_wkv.as_ref(), ld.comp_norm.as_ref()) else {
            return Ok(false);
        };
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
        // DSV41_SPEC: the commit replays the ACCEPTED PREFIX of this block
        // through the compressor, and the only way to play a row back is with the
        // projections that row consumed. `s.kvp_r`/`s.scp_r` are SHARED m-row
        // scratch, overwritten by every compress-source layer in turn — so save
        // this layer's rows now, while they are still this layer's. Armed only
        // around the spec step's verify, so the shadow path pays nothing.
        if self.spec_capture {
            let n = m * hd * std::mem::size_of::<f32>();
            self.dev.memcpy_d2d(
                (self.s.spec_snap_kvp.ptr as *mut f32)
                    .wrapping_add(layer * VERIFY_ROWS * hd) as *mut c_void,
                self.s.kvp_r.ptr as *const c_void,
                n,
            )?;
            self.dev.memcpy_d2d(
                (self.s.spec_snap_scp.ptr as *mut f32)
                    .wrapping_add(layer * VERIFY_ROWS * hd) as *mut c_void,
                self.s.scp_r.ptr as *const c_void,
                n,
            )?;
        }
        Ok(true)
    }

    /// ONE ROW of the multi-row compressor: the single-token pool + commit pair
    /// (the SAME pair the single-row decode runs, at this row's own position
    /// `pos_base + r`), plus the host mirror of the device counter.
    ///
    /// The pair is per-POSITION, not per-block: the pool's mode-2 branch and the
    /// commit's completion rule both carry ONE row. Running them once with
    /// `seqlen = m` only consumed ROW 0 — rows 1..m-1's `kvp`/`scp` never entered
    /// the state and the groups they completed never existed (the audit's defect #2;
    /// the operator-visible symptom was "the latent the draft sees never updates").
    /// Per row, the compressed state after the block is exactly what m sequential
    /// single-row steps would leave.
    ///
    /// Returns `true` when the row COMPLETED a group — the same deterministic rule
    /// the commit kernel applies on the device (`(pos + 1) % ratio == 0`), which is
    /// the signal [`Self::attention_rows`] uses to publish that group's index key
    /// before the row's selection runs. `compressor_fused` is NOT used: its launcher
    /// rejects `b != 1 || seqlen != 1` outright, and the pool+commit pair below is
    /// the path the fused launch is bit-identical to (a parity target, not a
    /// different program).
    fn compress_row(&mut self, layer: usize, r: usize, pos_base: i32) -> Result<bool> {
        let cfg = self.cfg;
        let hd = cfg.head_dim;
        let ratio = cfg.compress_ratio(layer).max(1);
        let ratio_i = ratio as i32;
        let ld = &self.w.layers[layer];
        let (Some(_), Some(norm)) = (ld.comp_wkv.as_ref(), ld.comp_norm.as_ref()) else {
            return Ok(false);
        };
        let st = self.dev.stream();
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
            ratio_i,
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
            ratio_i,
            st,
        )?;
        // The host MIRROR of the device counter, by the SAME deterministic rule the
        // commit kernel applies ((*pos + 1) % ratio == 0 commits one latent). The
        // two agree by construction, without the download that used to be here.
        let committed = (pos_base + r as i32 + 1) % ratio_i == 0;
        if committed {
            self.layers[layer].compress_len += 1;
        }
        Ok(committed)
    }

    /// The STATIC precondition of [`Self::compress_rows_fused`] — a pure function
    /// of `(layer, m, pos_base, cfg, device)`, which is what lets
    /// [`Self::attention_rows`] ask "was my (or my owner's) block hoisted?" for any
    /// layer WITHOUT carrying a per-layer flag that a rollback could leave stale.
    ///
    /// False for: the gate off / a stale `.so`, `m < 2` (a single row is the
    /// per-row path's business and keeps its exact launch sequence), `m >
    /// VERIFY_ROWS`, the first step (`pos_base == 0` is the prefill state mapping),
    /// `ratio == 1` (no gate, no state), and a layer that carries no compressor.
    fn mrows_compress_ok(&self, layer: usize, m: usize, pos_base: i32) -> bool {
        if !compressor_mrows() || !self.dev.supports_compressor_fused_mrows() {
            return false;
        }
        if m < 2 || m > VERIFY_ROWS || pos_base <= 0 {
            return false;
        }
        if self.cfg.compress_ratio(layer) <= 1 {
            return false;
        }
        let ld = &self.w.layers[layer];
        ld.comp_wkv.is_some() && ld.comp_norm.is_some()
    }

    /// COMPRESSOR-MROWS (`DSV41_COMPRESSOR_MROWS=1`, DEFAULT OFF): ONE
    /// `dsv41_compressor_fused_mrows` launch for this layer's WHOLE block — the `m`
    /// `compressor_pool_on` + `compress_commit_on` pairs of [`Self::compress_row`]
    /// collapsed into a single kernel whose block loops `r` ascending. The kernel
    /// header carries the two cross-row dependencies that forbid anything else
    /// (the position-derived state slot, so rows r and r+2 share one at ratio 2,
    /// and the commit's `*clen` / ring row), and the per-row argument: every row's
    /// pool/commit arithmetic is the same program, so the block's bytes equal what
    /// `m` sequential rows leave.
    ///
    /// WHAT IT CHANGES FOR THE READERS (the reason the launch also writes the two
    /// snapshots). [`Self::compress_row`]'s interleave works because row r's
    /// commit is issued before row r's readers, so the LIVE device counter already
    /// carries row r's value when they read it. A block-wide launch commits every
    /// row first, so after it the live counter is the block-FINAL one at every row
    /// — `sparse_attn`, `comp_placeholder` and the indexer must be handed
    /// `clen_rows_r[layer][r]` instead (audit defect #1: otherwise row 0 selects
    /// against its own future) — and the per-row index key publish must read
    /// `latent_rows_r + r*hd`, because the shared `latent` only holds the last
    /// row's pooled row. Both consumers are wired in [`Self::attention_rows`].
    ///
    /// Returns whether the arm TOOK. `false` (gate off, a shape the kernel cannot
    /// take, an `.so` without the symbol) leaves the caller on the per-row path
    /// verbatim: `compress_row` is then still the only writer of the state, the
    /// ring and the counters. `comp_len_rows` receives the per-row HOST MIRROR —
    /// the same deterministic rule the commit kernel applies, `(pos_base + r + 1) %
    /// ratio == 0` — which the caller's per-row `comp_len` branch needs, and the
    /// mirror field itself is advanced to the block-final count that `m`
    /// `compress_row` calls would have left.
    fn compress_rows_fused(
        &mut self,
        layer: usize,
        m: usize,
        pos_base: i32,
        comp_len_rows: &mut [usize; VERIFY_ROWS],
    ) -> Result<bool> {
        if !self.mrows_compress_ok(layer, m, pos_base) {
            return Ok(false);
        }
        let cfg = self.cfg;
        let hd = cfg.head_dim;
        let ratio = cfg.compress_ratio(layer).max(1);
        let ratio_i = ratio as i32;
        let ld = &self.w.layers[layer];
        let (Some(_), Some(norm)) = (ld.comp_wkv.as_ref(), ld.comp_norm.as_ref()) else {
            return Ok(false);
        };
        let st = self.dev.stream();
        self.dev.compressor_fused_mrows_on(
            self.s.kvp_r.ptr as *const f32,
            self.s.scp_r.ptr as *const f32,
            norm.as_f32(),
            self.layers[layer].state_kv.ptr as *mut f32,
            self.layers[layer].state_score.ptr as *mut f32,
            self.layers[layer].latent.ptr as *mut f32,
            self.layers[layer].out_rows.ptr as *mut i32,
            self.cos_comp.as_f32(),
            self.sin_comp.as_f32(),
            self.layers[layer].ring.ptr as *mut f32,
            (self.s.clen.ptr as *mut std::os::raw::c_int).wrapping_add(layer),
            (self.s.clen_rows_r.ptr as *mut std::os::raw::c_int)
                .wrapping_add(layer * VERIFY_ROWS),
            self.s.latent_rows_r.ptr as *mut f32,
            1,
            m as i32,
            hd as i32,
            ratio_i,
            cfg.rope_head_dim as i32,
            (cfg.rope_head_dim / 2) as i32,
            cfg.window_size as i32,
            // `pos_rows[0] == pos_base`, and the kernel derives row r's position
            // as `start_pos + r` — the same positions the per-row calls passed
            // explicitly (`pos_base + r` with `pos_rows + r`).
            self.s.pos_rows.ptr as *const std::os::raw::c_int,
            cfg.norm_eps,
            st,
        )?;
        // The host MIRROR, row by row, by the SAME deterministic rule the commit
        // kernel applies on the device. Ended at the block-final count, which is
        // what `m` per-row `compress_row` calls leave in the field.
        let mut lens = self.layers[layer].compress_len;
        for r in 0..m {
            if (pos_base + r as i32 + 1) % ratio_i == 0 {
                lens += 1;
            }
            comp_len_rows[r] = lens;
        }
        self.layers[layer].compress_len = lens;
        Ok(true)
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
        // (lazy-text-degradation's SEVERE verdict): `gemm_fp8_mx` dispatches on
        // `m` between two different programs — SIMT warp-per-row GEMV at m==1
        // and a 16-row tile MMA at m>1 — so the batched (m=6) and lazy (m=1)
        // arms compute the engram projection through DIFFERENT summation
        // orders, making the two arms' residual streams bit-incomparable (the
        // Bristol/burdens/oqua divergence). Every other m-row projection in
        // the verify already routes through proj_mrows/gemm_fp8_mrows to avoid
        // this exact dispatch; engram was the one that slipped through. Fix:
        // m individual m==1 calls (the canonical program both arms share).
        let st = self.dev.stream();
        let k_cols = (n_cols * ehd) as i32;
        for r in 0..m {
            self.dev.gemm_fp8_mx_on(
                (self.s.eng_xq_r.ptr as *const u8).wrapping_add(r * n_cols * ehd),
                (self.s.eng_xsc_r.ptr as *const f32).wrapping_add(r * (n_cols * ehd / 32)),
                wkv.ptr() as *const u8,
                wsc.ptr() as *const u8,
                std::ptr::null(),
                (self.s.eng_kv_r.ptr as *mut f32).wrapping_add(r * (hc + 1) * dim),
                1,
                ((hc + 1) * dim) as i32,
                k_cols,
                st,
            )?;
        }
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
    /// # The batched-expert launchers are MULTI-ROW
    ///
    /// `dsv41_expert_gate_up_fp4_batched` / `dsv41_expert_down_fp4_batched` /
    /// `dsv41_expert_down_reduce_fp4_batched` / `dsv41_swiglu_limit_batched` now
    /// put `rows` into the grid's THIRD dimension (`blockIdx.z` = the activation
    /// row): the grid is `((n_total + warps - 1)/warps, slots, rows)` where
    /// `n_total` is the OUTPUT width (`inter` / `2*inter` / `dim`) and `grid.y` is
    /// the slot. Each call therefore computes ALL m activation rows against their
    /// `slots` experts, deriving every per-row pointer itself from the arguments
    /// plus `gridDim.y` (see the layout contract on the kernels). `rows == 1`
    /// leaves `blockIdx.z == 0`, i.e. the pre-existing single-row launch bit for
    /// bit — so a caller that still passes rows = 1 is unchanged.
    ///
    /// The routed half is consequently ONE call per stage instead of m, with the
    /// host passing the row-0 BASE of each `[row][slot][...]` buffer:
    ///   fp4 activation `xq4_r`/`xsc4_r` : [row][dim/2] / [row][dim/32] (the
    ///                                     quantiser's own packed layout)
    ///   `route_idx_r` / `route_w_r`     : [row][topk]      (row pitch = slots)
    ///   `ex_act_r`                      : [row][slot][act_slot]
    ///   `moe_out_r`                     : [row][dim]
    /// The per-row pointers are NOT passed any more — that is the whole point of
    /// the rewrite (m·4 launches → 4, the MoE launch count the verify audit
    /// flagged). Byte-for-byte identical to the per-row loop it replaces: see the
    /// bit-exactness note at the launch site.
    ///
    /// The routing itself IS multi-row: the gate is the same M=1 bf16 GEMV `moe()`
    /// runs (once per row), and `route_topk` has a native `rows` dimension.
    ///
    /// # Grouped (permuted) routing — DSV41_EXPERT_GROUPED
    ///
    /// Build the expert-major form of `route_idx_r[m][topk]` and gather the
    /// activation rows into it. This is the PRECONDITION of the expert-centric
    /// dense GEMMs (`dsv41_expert_gemm_e4m3_ext` covers ONE expert per launch
    /// over a dense row block); see `kernels/cuda/dsv41_route.cu` for the layout
    /// contract and docs/agent/grouped-routing-design.md for the design.
    ///
    /// Returns `Ok(true)` when the layout is LIVE in `s.grp_*`: the gate is
    /// armed, the `.so` carries the `dsv41_route_*` set, and the gather landed.
    /// Every other outcome returns `Ok(false)` after a one-shot notice, so the
    /// caller keeps the proven per-(row, slot) launches. Three rules drive that
    /// shape:
    ///   * the step must NEVER fail because of this arm (it is an optimisation,
    ///     and `Ok(false)` is the fallback contract every other arm here uses);
    ///   * an armed gate must NEVER silently measure the old path (this
    ///     project's #1 trap), hence the notice;
    ///   * it must NEVER produce a number the ungrouped path would not, so the
    ///     movers are pure copies and the table they consume is the one
    ///     `route_topk` just wrote.
    ///
    /// `row_bytes` is the activation's packed row pitch IN BYTES: `dim` for the
    /// e4m3 arm (one byte per value, the format the e4x kernel eats) and `dim/2`
    /// for the fp4 nibble packing. The scales are always one f32 per 32 values.
    ///
    /// ⚠️ NOT YET USED BY A GEMM. The layout and both movers are wired, but the
    /// per-expert dense GEMM call is not issued yet: at this engine's shapes
    /// (`m <= VERIFY_ROWS = 6`, `topk = 6`, `n_routed = 384`) the per-expert row
    /// count is O(1), while `e4x_launch_gemm` requires `rows % 128 == 0`, so a
    /// per-expert launch would pad 1-3 real rows out to 128 — 40x+ the tensor-core
    /// work of the per-(row, slot) launches it replaces. Unlocking the -6.8ms
    /// needs the GROUP-INDEXED MASKED kernel (a grid over m-tiles with a group
    /// table, DeepGEMM's `m_grouped_gemm_nt_masked`), which is a separate
    /// deliverable. Until then this arm is a numeric no-op BY CONSTRUCTION, and
    /// says so.
    fn moe_route_grouped(
        &mut self,
        m: usize,
        topk: usize,
        n_routed: usize,
        row_bytes: usize,
    ) -> Result<bool> {
        if !expert_grouped() {
            return Ok(false);
        }
        let so_has = self.dev.supports_route_group();
        if !so_has {
            expert_grouped_skipped_note(
                "the loaded .so has no `dsv41_route_group` / `dsv41_route_gather_rows` / \
                 `dsv41_route_scatter_rows` set (rebuild kernels/cuda: bash build.sh 103a)",
                false,
            );
            return Ok(false);
        }
        // Capacity guard: every grouped buffer is sized from the `grp_*` helpers
        // at allocation time, and a runtime shape past them would be an
        // out-of-bounds WRITE inside a graph. Refuse loudly instead — the
        // ungrouped path is always available.
        if n_routed > grp_n_experts(self.cfg)
            || topk > grp_topk_max(self.cfg)
            || m > VERIFY_ROWS
            || row_bytes == 0
            || (row_bytes % 16) != 0
        {
            expert_grouped_skipped_note(
                "the block is outside the grouped-router buffers' allocated capacity \
                 (see grp_n_experts / grp_topk_max / grp_m_cap in chain_dev.rs), or the \
                 activation row pitch is not a multiple of 16 bytes",
                true,
            );
            return Ok(false);
        }
        let n_assign = m * topk;
        let m_cap = grp_m_cap(self.cfg);
        let sc_row_bytes = (row_bytes / 32) * 4; // one f32 scale per 32 values
        // 1. the layout. Pure index work: counts, the prefix sum, both
        //    directions of the permutation and the compact live-expert list.
        let built = self.dev.route_group(
            self.s.route_idx_r.ptr as *const i32,
            self.s.grp_counts.ptr as *mut i32,
            self.s.grp_starts.ptr as *mut i32,
            self.s.grp_rows.ptr as *mut i32,
            self.s.grp_slots.ptr as *mut i32,
            self.s.grp_perm.ptr as *mut i32,
            self.s.grp_src.ptr as *mut i32,
            self.s.grp_active.ptr as *mut i32,
            self.s.grp_nactive.ptr as *mut i32,
            m as i32,
            topk as i32,
            n_routed as i32,
            m_cap as i32,
        )?;
        if !built {
            expert_grouped_skipped_note(
                "the `.so` exposes `dsv41_route_group` only partially",
                true,
            );
            return Ok(false);
        }
        // 2. the activation gather. Byte-verbatim: `grp_xq[g]` IS the e4m3 row
        //    of the assignment at grouped position `g`, so an expert's operand
        //    block is `grp_xq + starts[e] * row_bytes` and holds exactly the
        //    bytes the per-(row, slot) path would have fed that expert.
        let gathered = self.dev.route_gather_rows(
            self.s.xq4_r.as_u8(),
            self.s.xsc4_r.as_f32(),
            self.s.grp_xq.as_u8() as *mut u8,
            self.s.grp_xsc.ptr as *mut f32,
            self.s.grp_src.ptr as *const i32,
            n_assign as i32,
            topk as i32,
            row_bytes as i32,
            sc_row_bytes as i32,
        )?;
        if !gathered {
            expert_grouped_skipped_note(
                "the `.so` exposes `dsv41_route_gather_rows` only partially",
                true,
            );
            return Ok(false);
        }
        // The layout is live. Whether the GEMM that consumes it runs is
        // `moe_experts_grouped_gate_up`'s decision (it is the only consumer, and
        // it emits its own one-shot notice when an armed gate declines), so all
        // this step says is that the tables are ready.
        Ok(true)
    }

    /// The GROUPED routed gate/up: `dsv41_expert_gemm_e4m3_grouped` (`tc5::e4x`,
    /// DeepGEMM's `m_grouped_gemm_nt_masked`) followed by the scatter back to the
    /// per-(row, slot) `ex_act_r` layout.
    ///
    /// This is what makes the tcgen05 e4m3 arm reachable at decode shapes. The
    /// dense arm needs `rows % 128 == 0`, but `moe_rows`' routing is
    /// per-(row, slot) with O(1) rows per expert (`m <= VERIFY_ROWS = 6`,
    /// `topk = 6`, so at most 36 assignments), and handing the per-(row, slot)
    /// table to a dense launch would apply slot s's expert to rows that route
    /// elsewhere — a SILENT WRONG ANSWER (`e4x_tile = false` above). The grouped
    /// layout plus a masked M=128 tile removes the padding without changing a
    /// single rounding step (see the kernel's NUMERIC DOMAIN note).
    ///
    /// PRECONDITIONS, all checked here; every decline is LOUD ONCE, because this
    /// is only called with BOTH gates armed (`DSV41_EXPERT_GROUPED` **and**
    /// `DSV41_EXPERT_TCGEN05_E4M3`) and an armed pair that silently measured the
    /// proven per-(row, slot) launches is this project's #1 measurement-bias trap:
    ///   * `e4m3` — the kernel's A operand is `kind::f8f6f4`, i.e. the e4m3 bytes
    ///     `quant_fp8` wrote and `route_gather_rows` gathered;
    ///   * NOT `gateup_fused` — the e4x epilogue only CLAMPS, it never fuses the
    ///     swiglu, so the fused-swiglu shape is a different kernel's job;
    ///   * the launcher's shape contract: `dim % 64 == 0` and
    ///     `(2*inter_local) % 64 == 0`;
    ///   * the plain (non-interleaved) w1/w3 planes, as the dense e4x arm
    ///     requires;
    ///   * the `.so` carries the symbol (a staged-bring-up arm, like every other).
    ///
    /// On success `ex_act_r` holds the gate|up block exactly where the batched
    /// per-(row, slot) launch would have left it, so the swiglu / down / reduce
    /// chain below is untouched by this arm. `Ok(false)` means "nothing ran, run
    /// the proven launch" — the same fallback contract as every other arm here.
    #[allow(clippy::too_many_arguments)]
    fn moe_experts_grouped_gate_up(
        &mut self,
        m: usize,
        topk: usize,
        n_routed: usize,
        dim: usize,
        inter_local: usize,
        e4m3: bool,
        gateup_fused: bool,
        ld: &LayerDev,
    ) -> Result<bool> {
        // The caller guarantees `DSV41_EXPERT_GROUPED` (it only calls this with
        // the layout live). The tcgen05 e4m3 gate is this method's own business:
        // without it the `.so` would return 0 (nothing ran) and the operator would
        // never hear why the grouped arm they armed did nothing.
        if !expert_tcgen05_e4m3() {
            expert_grouped_skipped_note(
                "DSV41_EXPERT_GROUPED is on but DSV41_EXPERT_TCGEN05_E4M3 — the gate of its \
                 only GEMM consumer — is off, so the grouped tcgen05 tile arm stays \
                 compiled-in-but-inert and the proven per-(row, slot) launches answer the step",
                true,
            );
            return Ok(false);
        }
        if !e4m3 {
            expert_grouped_skipped_note(
                "the grouped tcgen05 tile arm eats e4m3 ACTIVATION bytes (kind::f8f6f4), but \
                 DSV41_EXPERT_ACT_E4M3 is off so the gathered rows are fp4 nibbles",
                true,
            );
            return Ok(false);
        }
        if gateup_fused {
            expert_grouped_skipped_note(
                "gate/up is on the FUSED swiglu shape (DSV41_GATEUP_FUSE + fp4 mode 2 + \
                 dim % 512 == 0); the tcgen05 e4x epilogue only clamps, it never fuses",
                true,
            );
            return Ok(false);
        }
        let n_total = 2 * inter_local;
        if dim % 64 != 0 || n_total % 64 != 0 {
            expert_grouped_skipped_note(
                "the grouped tcgen05 e4m3 gate/up needs the dense-tile shape contract \
                 (dim % 64 == 0 and 2*inter_local % 64 == 0)",
                true,
            );
            return Ok(false);
        }
        if ld.experts_ilv {
            expert_grouped_skipped_note(
                "the routed gate/up weights are interleaved (DSV41_EXPERT_ILV); the \
                 group-indexed e4m3 tile arm reads the plain w1/w3 planes only",
                true,
            );
            return Ok(false);
        }
        if !self.dev.supports_expert_gemm_e4m3_grouped() {
            expert_grouped_skipped_note(
                "the loaded .so has no `dsv41_expert_gemm_e4m3_grouped` \
                 (rebuild kernels/cuda: bash build.sh 103a)",
                false,
            );
            return Ok(false);
        }
        if ld.experts.len() < 2 {
            expert_grouped_skipped_note(
                "the layer has fewer than two expert tensors, so the per-expert weight \
                 strides cannot be derived",
                true,
            );
            return Ok(false);
        }
        // The expert tensors are views into one per-layer pool with a uniform
        // stride, so every plane is `base + ids[e] * stride`. Same derivation as
        // the batched call site below, repeated so this arm is self-contained.
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
        let n_assign = m * topk;
        // gate | up in ONE launch: `b = w1`, `b_hi = w3`, `b_split = inter_local`
        // (the dense arm's own convention for this pair), epi_mode 1 = the
        // gate/up clamp. `n_assign` bounds the grid's group-table cursor
        // (>= n_active by construction) and `grp_m_cap` bounds the per-expert
        // m-tiles of the masked kernel.
        let ran = self.dev.expert_gemm_e4m3_grouped(
            self.s.grp_xq.as_u8(),
            self.s.grp_xsc.as_f32(),
            self.s.grp_out.ptr as *mut f32,
            self.s.grp_active.ptr as *const i32,
            self.s.grp_nactive.ptr as *const i32,
            self.s.grp_counts.ptr as *const i32,
            self.s.grp_starts.ptr as *const i32,
            n_routed as i32,
            n_assign as i32,
            grp_m_cap(self.cfg) as i32,
            n_total as i32,
            dim as i32,
            inter_local as i32,
            1, // epi_mode: gate/up clamp (2/3 are per-(row, slot); the launcher refuses them)
            self.cfg.swiglu_limit,
            w1_base,
            w1_stride,
            w1s_base,
            w1s_stride,
            w3_base,
            w3_stride,
            w3s_base,
            w3s_stride,
        )?;
        if !ran {
            return Ok(false); // a gate was OFF after all: nothing ran, no notice owed
        }
        // Grouped `[n_assign, 2*inter_local]` -> per-(row, slot) `ex_act_r`. This
        // is the ONLY thing that has to happen for the rest of the route to be
        // unaware of the grouped arm: every (row, slot) block lands exactly where
        // the batched launch would have written it (`i = r*topk + t` is the
        // destination row and the row pitch is `topk * n_total`), and a
        // `perm_map[i] < 0` (an out-of-range expert id) is written as zeros, which
        // is what the per-(row, slot) path produces too.
        let scattered = self.dev.route_scatter_rows(
            self.s.grp_out.ptr as *const f32,
            self.s.ex_act_r.ptr as *mut f32,
            self.s.grp_perm.ptr as *const i32,
            n_assign as i32,
            n_total as i32,
        )?;
        if !scattered {
            expert_grouped_skipped_note(
                "the grouped gate/up ran but `dsv41_route_scatter_rows` is missing from the \
                 .so, so the grouped output could not be moved into `ex_act_r`",
                false,
            );
            return Ok(false); // the batched launch below re-writes `ex_act_r` in full
        }
        Ok(true)
    }

    /// The multi-row MoE: the m-row twin of [`Self::moe`], ending in the MoE
    /// all-reduce over the block's rows. Returns whether that AR took the
    /// `DSV41_VERIFY_AR_FOLD` fold, i.e. its pubred epilogue already wrote this
    /// block's `hc_post` onto `h_r` — in which case `layer_rows` must NOT run the
    /// standalone `hc_post_rows`.
    fn moe_rows(&mut self, layer: usize, ld: &LayerDev, m: usize) -> Result<bool> {
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

        // ---- VERIFY_FORK: the MoE dual chain (routed vs shared expert) ----
        // The m-row twin of `moe()`'s MOE_DUAL. The fork is recorded on the MAIN
        // stream BEFORE the gate, i.e. at a point where `xn_r` (and the T1 fp8
        // staging the hc tail may have emitted) is already complete — so the
        // shared half reads exactly what the serial code reads, it just no longer
        // WAITS for the gate + the routed experts in between. The two halves read
        // `xn_r` through DISJOINT quantisations (routed: `xq4_r`/`xsc4_r`, shared:
        // `xq`/`xsc`) and use disjoint scratch (`ex_act_r`/`ex_down_r` vs
        // `sh_act_r`). See [`verify_fork`] for the gate and the capture argument.
        //
        // The shared half may NOT fold into `moe_out_r`: the routed down-reduce
        // OVERWRITES that buffer at the same time. It therefore always writes its
        // `w2` block to the DISJOINT `sh_out_r` scratch (the same buffer
        // `shared_expert_mrows` uses for exactly this reason) and the join below
        // runs the element-wise merge the serial path ran per row.
        //
        // Excluded: the shared expert's fused multi-row arms, which are
        // main-stream programs this arm does not re-issue, and the two skip
        // switches (nothing to overlap / nothing to merge).
        let sh_w6 = ld.shared_w1.is_some()
            && ld.shared_w1_scale.is_some()
            && ld.shared_w3.is_some()
            && ld.shared_w3_scale.is_some()
            && ld.shared_w2.is_some()
            && ld.shared_w2_scale.is_some();
        let dual = verify_fork()
            && !self.opts.skip_experts
            && !self.opts.skip_shared_expert
            && shared_rank
            && sh_w6
            && !sh_exp_mrows()
            && !sh_pair_m()
            && !sh_exp_fused()
            && self.dev.supports_dual_chain();
        let sh_st = if dual { self.dev.side_stream2() } else { self.dev.stream() };
        if dual {
            self.dev.dual_chain_fork()?;
        }

        // ---- gate + route ----
        // The gate runs on EVERY path (exactly as `moe()` does: `skip_experts` only
        // skips the expert launches); the routing is genuinely multi-row through
        // `route_topk`'s `rows` dimension.
        let gate_bias = ld
            .gate_bias
            .as_ref()
            .map(|b| b.as_f32())
            .unwrap_or(std::ptr::null());
        // ROW-FOLD (DSV41_ROW_FOLD_GATE=1 or DSV41_GATE_MROWS=1, DEFAULT OFF):
        // ONE multi-row GEMV (`ferrite_gemv_bf16_v2_mrows`, m <= 8) where the
        // loop below issues `m` single-row `gemv_bf16` launches. The entry runs
        // the v2 program with a row dimension — same WPR heuristic
        // (`gv2_wpr(384) = 8`), same K-slice walk, same uint4/8-element FMA
        // groups, same smem partial fold, one independent accumulator per row —
        // so row r is bit-identical to the per-row call it replaces, PROVIDED
        // the per-row path would take v2; the wrapper enforces that
        // (`gemv_bf16_v2_wanted(n_routed)` + the symbol) and returns Ok(false)
        // otherwise, keeping the loop.
        let gate_folded = row_fold_gate()
            && self.dev.gemv_bf16_v2_mrows(
                ld.gate_w.as_ref().unwrap().ptr() as *const c_void,
                self.s.xn_r.ptr as *const f32,
                self.s.scores_r.ptr as *mut f32,
                m as i32,
                n_routed as i32,
                dim as i32,
            )?;
        if !gate_folded {
            for r in 0..m {
                self.dev.gemv_bf16(
                    ld.gate_w.as_ref().unwrap().ptr() as *const c_void,
                    (self.s.xn_r.ptr as *const f32).wrapping_add(r * dim),
                    (self.s.scores_r.ptr as *mut f32).wrapping_add(r * n_routed),
                    n_routed as i32,
                    dim as i32,
                )?;
            }
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
            // The interleaved layout (DSV41_EXPERT_ILV) is readable by the batched
            // gate/up call with EITHER epilogue: the kernel's gate/up PAIR body
            // reads both halves of a row from one region with ONE LDG.128, and its
            // epilogue is selected by the caller's slot pitch (`fuse`, bound to
            // `out_slot_stride == inter`). So the two-pass e4m3 arm - which passes
            // the RAW [2*inter] pitch - no longer conflicts with ILV; it gets
            // exactly the raw gate|up pair it accumulates in. What is still
            // required is the batched gate/up entry point itself and the pair
            // body's K contract, dim % 512 == 0 (it walks whole 512-value groups
            // and has no tail; the launcher refuses that combination loudly).
            if ld.experts_ilv && !(self.dev.supports_moe_batch() && (dim % 512) == 0) {
                return Err(FerriteError::Config(
                    "routed expert gate/up weights are interleaved (DSV41_EXPERT_ILV) but the \
                     batched gate/up path is unavailable or dim % 512 != 0 — run with \
                     DSV41_EXPERT_ILV=0, or restore DSV41_MOE_BATCH with a dim divisible by 512"
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
            // e4m3 DIRECT (`DSV41_EXPERT_ACT_E4M3`, default OFF) — see the gate's
            // doc comment. The activation is quantised with the OFFICIAL
            // `act_quant(e4m3, block=32)` and consumed in ONE gate/up pass, so
            // gate_up+swiglu fusion stays available (unlike the retired two-pass
            // arm, which had to write the unfused [2*inter] layout because
            // swiglu(x+y) != swiglu(x)+swiglu(y)).
            let e4m3 = expert_act_e4m3() && self.dev.supports_expert_act_e4m3();
            if expert_act_e4m3() && !e4m3 {
                act_e4m3_skipped_note();
            }
            let gateup_fused = gateup_fuse()
                && self.dev.supports_gateup_fuse()
                && expert_fp4_mode() == 2
                && (dim % 512) == 0;
            let act_slot = if gateup_fused {
                inter_local
            } else {
                2 * inter_local
            };
            // ex_act_r is [row][slot][act_slot], i.e. row pitch = topk*act_slot.
            // That pitch is no longer computed here - the multi-row kernels
            // derive it from `slots * out_slot_stride` - but it is the pitch the
            // kernel's per-row walk assumes, and the two agree by construction
            // (slots == topk, out_slot_stride == act_slot).
            // One quantisation covers all m rows (rows = m is native to both
            // quantisers); the single gate/up call then reads each row's bytes at
            // arow*dim (e4m3, 1 byte/value) or arow*(dim/2) (fp4) inside the
            // kernel, with the scales at arow*(dim/32) either way.
            if e4m3 {
                self.dev.quant_fp8(
                    self.s.xn_r.ptr as *const f32,
                    self.s.xq4_r.ptr as *mut u8,
                    self.s.xsc4_r.ptr as *mut f32,
                    m as i32,
                    dim as i32,
                    32,
                    true,
                )?;
            } else {
                self.dev.quant_fp4(
                    self.s.xn_r.ptr as *const f32,
                    self.s.xq4_r.ptr as *mut u8,
                    self.s.xsc4_r.ptr as *mut f32,
                    m as i32,
                    dim as i32,
                    32,
                    true,
                )?;
            }
            // ---- GROUPED (permuted) routing (DSV41_EXPERT_GROUPED) --------------
            // The e4m3 activation bytes are now in `xq4_r` for all m rows, so this
            // is the one point where the grouped layout can be built AND filled.
            // `moe_route_grouped` is a pure PREPROCESSING step: it indexes and
            // copies, never computes, and it declines (`Ok(false)`) into a no-op
            // when the gate is OFF or the `.so` lacks the `dsv41_route_*` set —
            // the `if` below therefore reads like every other optional arm here.
            // It does NOT itself replace the launches below: its consumer does.
            let row_bytes = if e4m3 { dim } else { dim / 2 };
            let _grouped = self.moe_route_grouped(m, topk, n_routed, row_bytes)?;
            // ---- GROUPED routed gate/up: the GROUP-INDEXED MASKED tcgen05 tile ---
            // `dsv41_expert_gemm_e4m3_grouped` (`tc5::e4x`,
            // `m_grouped_gemm_nt_masked`) IS the consumer: one masked M=128 tile
            // per expert, the expert coming from the group table (`active` /
            // `counts` / `starts`) and the rows past `counts[e]` masked out. It
            // needs BOTH gates armed (`DSV41_EXPERT_GROUPED` **and**
            // `DSV41_EXPERT_TCGEN05_E4M3`), the e4m3 activation, the plain
            // (non-interleaved) w1/w3 planes, the launcher's shape contract and
            // the symbol in the `.so` — otherwise it returns `false` and the
            // batched SIMT launch below runs, with a one-shot notice for every
            // decline (an armed pair must never silently measure the OLD path).
            //
            // It writes `ex_act_r` through the scatter, i.e. the very buffer the
            // batched launch writes, at the very same offsets: everything below
            // (the swiglu, the down GEMV, the reduce) is unaware of this arm.
            let grp_gu = _grouped
                && self.moe_experts_grouped_gate_up(
                    m, topk, n_routed, dim, inter_local, e4m3, gateup_fused, ld,
                )?;
            // ---- tcgen05 e4m3 DENSE-TILE gate/up (tc5::e4x) ----------------------
            // `dsv41_expert_gemm_e4m3_ext` is the e4m3 twin of the masked M=128
            // tile GEMM (`mxf4_gemm_kernel`): the M = ACTIVATION ROWS form, i.e.
            // ONE launch carries a DENSE `[rows, k]` e4m3 activation (one byte per
            // value — exactly what the `quant_fp8` branch above wrote into
            // `xq4_r`) plus its `[rows, k/32]` f32 per-row scales, and ONE expert
            // (`ids[slot]`), with `b_split = inter_local` expressing gate|up the
            // same way the batched GEMV's `b`/`b_hi` pair does. It needs the e4m3
            // activation (`DSV41_EXPERT_ACT_E4M3`, the `e4m3` term above) because
            // `kind::f8f6f4` eats e4m3 bytes, and it shares the swapAB e4m3 arm's
            // runtime gate (`DSV41_EXPERT_TCGEN05_E4M3`): which of the two a step
            // runs is a LAUNCH-SHAPE decision (dense M=128 activation tile vs the
            // single-token swapAB form), not a numeric-format one.
            //
            // ⚠️ THIS DENSE ARM NEVER QUALIFIES AT `moe_rows`' SHAPES, and that is
            // correct — it is the GROUPED masked arm above (not this one) that
            // serves the routed gate/up. Both of this arm's preconditions are
            // shape/layout, not numerics, so neither may be tolerated silently
            // (an armed gate must never mean "the OLD path measured as the new
            // one" — this repo's #1 measurement-bias trap):
            //  1. EXPERT HOMOGENEITY. The kernel derives its whole B side from
            //     `base + ids[slot] * stride`, so ONE expert covers every row of
            //     the launch, while `moe_rows`' routing is per-(row, slot)
            //     (`route_idx_r[m][topk]`). A per-slot dense launch would apply
            //     slot s's expert to rows that route to a different one, i.e. a
            //     SILENT WRONG ANSWER. A uniform tile needs the GROUPED (permuted)
            //     routed form — the permutation plus per-expert row blocks — which
            //     is now built (`moe_route_grouped`) and consumed by
            //     `moe_experts_grouped_gate_up`; checking `route_idx_r` on the host
            //     would be a D2H read (a CUDA-graph capture hazard).
            //  2. DENSE OUTPUT PITCH. The kernel writes `out[row * n_total + col]`
            //     (row pitch `n_total`), while the routed gate/up output here is
            //     `ex_act_r[row][slot][act_slot]` (row pitch `topk * act_slot`).
            //     The two pitches coincide only for `topk == 1`.
            // So this arm's own `e4x_tile` stays false: when `grp_gu` is true the
            // grouped masked launch already answered this stage, and when it is
            // false the note below says why the tcgen05 family did not.
            let e4x_armed = e4m3
                && expert_tcgen05_e4m3()
                && self.dev.supports_expert_gemm_e4m3_ext()
                && !ld.experts_ilv;
            // The launcher's own contract (`e4x_launch_gemm` refuses anything
            // else), plus 16-byte aligment of the operand bases and strides —
            // which the loader guarantees for the weight planes and the
            // quantiser's row-major buffers.
            let e4x_shape = (m % 128) == 0 && (dim % 64) == 0 && ((2 * inter_local) % 64) == 0;
            // Precondition (1)/(2) above: a dense launch would need one expert's
            // dense row block plus a `n_total`-pitched output. The GROUPED masked
            // arm covers exactly that case, so this one stays OFF by design.
            let e4x_tile = false;
            if e4x_armed && !grp_gu && !(e4x_tile && e4x_shape) {
                tcgen05_e4m3_ext_skipped_note(
                    if !e4x_shape {
                        "the routed tile is not the dense shape the launcher accepts \
                         (m % 128 == 0, dim % 64 == 0 and 2*inter % 64 == 0 are all required), \
                         and the GROUPED masked arm did not take the stage either (see its own \
                         notice above)"
                    } else {
                        "the routed tile is per-(row, slot) and the GROUPED masked arm did not \
                         take the stage (see its own notice above); the dense arm carries ONE \
                         ids[slot] expert per launch and a dense [rows, n_total] out, so it \
                         cannot serve a per-(row, slot) routing"
                    },
                    true,
                );
            }
            let ids_base = self.s.route_idx_r.ptr as *const i32;
            let rw_base = self.s.route_w_r.ptr as *const f32;
            // ---- ONE rows = m launch per stage, for all m activation rows --------
            // The kernels' `rows` argument is now the THIRD grid dimension
            // (`expert_gemv_fp4_batched_kernel`: `blockIdx.z`, see its layout
            // contract), so the launcher's grid is (n_total/warps, slots, rows)
            // and every per-row pointer is derived INSIDE the kernel from the
            // arguments below plus `gridDim.y` (= slots = topk):
            //   a / a_scale : a + arow*(k/2) / a_scale + arow*(k/32)   (quantiser's
            //                 own row-major packed layout -> no extra stride arg)
            //   ids         : ids[arow*gridDim.y + slot]      == route_idx_r[m][topk]
            //   out         : out + (arow*gridDim.y + slot)*out_slot_stride
            //   act / rw    : act + (arow*gridDim.y + slot)*act_stride, same for rw
            // Each of those IS the offset the per-row loop applied on the host
            // (r*(dim/2), r*(dim/32), ids_base + r*topk, r*row_pitch +
            // slot*act_slot with row_pitch = topk*act_slot), so the host now
            // passes the row-0 BASE of each buffer and the kernel re-derives the
            // rest. `row_pitch == slots*out_slot_stride` is the whole alignment
            // requirement, and it holds by construction (row_pitch =
            // topk*act_slot = slots*out_slot_stride).
            //
            // BIT-EXACTNESS: rows share no output element, no accumulator and no
            // shared-memory staging - `arow` only shifts base pointers (the
            // kernel's ROW INDEPENDENCE block) - so row r of this call executes
            // the identical group order / fma chain / shuffle tree / epilogue the
            // rows=1 call executed for row r. rows == 1 degenerates to the
            // previous launch exactly (grid.z == 1, every offset 0).
            // e4m3 (`e4m3`): the SAME single pass as the e2m1 arm, with `qa`
            // holding one e4m3 byte per value and the kernel's `act_e4m3` flag
            // selecting the decoder. `act_slot` follows the fusion decision above,
            // exactly as it does on the e2m1 path.
            // `grp_gu` means the grouped masked tcgen05 launch above already
            // filled every (row, slot) block of `ex_act_r`, so this proven launch
            // must NOT run: it would overwrite the grouped result with the SIMT
            // path's — that is precisely the "armed gate measured as the OLD path"
            // failure this dispatch exists to avoid. `grp_gu == false` is the
            // fallback: this launch answers the step, unchanged.
            if !grp_gu {
                let (qa, qs) = (self.s.xq4_r.as_u8(), self.s.xsc4_r.as_f32());
                self.dev.expert_gate_up_fp4_batched(
                    qa,
                    qs,
                    self.s.ex_act_r.ptr as *mut f32,
                    act_slot as i64,
                    m as i32,
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
                    ids_base,
                    ld.experts_ilv as i32,
                    e4m3 as i32,
                )?;
            }
            // The separate swiglu pass is element-wise and row-local, and its
            // kernel walks the SAME [rows][slot][slot_stride] layout
            // (`base + (blockIdx.z*gridDim.y + blockIdx.y)*slot_stride`), so the
            // m-row loop collapses into one rows=m launch with no order
            // dependency to preserve: element i of (row, slot) is read and
            // written once, and the expf/limit chain is the rows=1 one.
            if !gateup_fused {
                self.dev.swiglu_limit_batched(
                    self.s.ex_act_r.ptr as *mut f32,
                    m as i32,
                    inter_local as i32,
                    cfg.swiglu_limit,
                    act_slot as i64,
                    topk as i32,
                )?;
            }
            if down_fuse() && self.dev.supports_down_fuse() {
                // ONE rows = m launch: the fused down+reduce kernel's grid is
                // (dim/warps, 1, rows) and it derives act_row =
                // act_base + arow*slots*act_stride, out_row = out + arow*n_total
                // and ids/rw [arow*slots + slot] itself - again exactly the host
                // offsets of the old loop. Its ascending-slot sum is SERIAL PER
                // ROW, same order, same rounded product/add pair, so row r is
                // bit-identical to the rows=1 call for row r.
                self.dev.expert_down_reduce_fp4_batched(
                    self.s.ex_act_r.ptr as *const f32,
                    act_slot as i64,
                    self.s.moe_out_r.ptr as *mut f32,
                    m as i32,
                    dim as i32,
                    inter_local as i32,
                    rw_base,
                    1,
                    topk as i32,
                    w2_base,
                    w2_stride,
                    w2s_base,
                    w2s_stride,
                    ids_base,
                )?;
            } else {
                // The down WRITE is one rows = m launch too (out row pitch =
                // slots*out_slot_stride = topk*dim, which IS the scratch's row
                // pitch). The fixed-order sum stays PER ROW: `dsv41_moe_down_reduce`
                // has no `rows` dimension, and the ascending-slot summation order
                // is the numerical contract - so it keeps the exact per-row shape
                // (scratch row r, out row r) it had before.
                self.dev.expert_down_fp4_batched(
                    self.s.ex_act_r.ptr as *const f32,
                    act_slot as i64,
                    self.s.ex_down_r.ptr as *mut f32,
                    dim as i64,
                    m as i32,
                    dim as i32,
                    inter_local as i32,
                    rw_base,
                    1,
                    topk as i32,
                    w2_base,
                    w2_stride,
                    w2s_base,
                    w2s_stride,
                    ids_base,
                )?;
                for r in 0..m {
                    let scratch =
                        (self.s.ex_down_r.ptr as *const f32).wrapping_add(r * topk * dim);
                    let out = (self.s.moe_out_r.ptr as *mut f32).wrapping_add(r * dim);
                    self.dev
                        .moe_down_reduce(scratch, out, dim as i32, topk as i32)?;
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
                // MULTI-ROW (DSV41_SH_EXP_MROWS, DEFAULT OFF): one pass over the
                // whole block instead of the per-row loop below. `true` means
                // every row's shared contribution is already in `moe_out_r`, so
                // the loop has nothing left to do — see `shared_expert_mrows`
                // for the bit-identity argument and the fallback contract.
                // VERIFY_FORK: skipped under the dual arm, which issues the
                // per-row loop on the side stream itself (the mrows passes are
                // main-stream launches the arm does not re-issue, which is why
                // `sh_exp_mrows`/`sh_pair_m`/`sh_exp_fused` decline the arm).
                let sh_mrows_done = if dual {
                    false
                } else {
                    self.shared_expert_mrows(w1, w1s, w3, w3s, w2, w2s, m, sh_il, dim)?
                };
                for r in 0..m {
                    // The multi-row pass already added this layer's shared
                    // expert; running the loop would add it a second time.
                    if sh_mrows_done {
                        break;
                    }
                    // The T1/T2 fast paths are single-row gates around this very
                    // pair, so the rows path runs the (quant1, gemm) pair.
                    // VERIFY_FORK: every launch of this half takes `sh_st` (the
                    // second side stream under the dual arm, the main stream
                    // otherwise). `quant1_on(.., main)` IS `quant1(..)` — the
                    // plain entry forwards to the `_on` one — so the serial arm
                    // stays byte-for-byte the old chain.
                    self.quant1_on(
                        (self.s.xn_r.ptr as *const f32).wrapping_add(r * dim),
                        dim as i32,
                        sh_st,
                    )?;
                    let sh_fused = sh_exp_mx2()
                        && self.dev.gemm_fp8_mx2_on(
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
                            sh_st,
                        )?;
                    if !sh_fused {
                        self.dev.gemm_fp8_mx_on(
                            self.s.xq.as_u8(),
                            self.s.xsc.as_f32(),
                            w1.as_u8(),
                            w1s.as_u8(),
                            std::ptr::null(),
                            self.s.sh_act_r.ptr as *mut f32,
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
                            (self.s.sh_act_r.ptr as *mut f32).wrapping_add(sh_il),
                            1,
                            sh_il as i32,
                            dim as i32,
                            sh_st,
                        )?;
                    }
                    self.dev.swiglu_limit_on(
                        self.s.sh_act_r.ptr as *mut f32,
                        1,
                        sh_il as i32,
                        cfg.swiglu_limit,
                        sh_st,
                    )?;
                    self.quant1_on(self.s.sh_act_r.ptr as *const f32, sh_il as i32, sh_st)?;
                    if dual {
                        // VERIFY_FORK: the DISJOINT scratch, never `moe_out_r`
                        // — the routed half's down-reduce is writing that buffer
                        // on the main stream right now. Same single-row GEMV,
                        // same operands: only the destination differs, and the
                        // join below adds the block in one element-wise pass.
                        self.dev.gemm_fp8_mx_on(
                            self.s.xq.as_u8(),
                            self.s.xsc.as_f32(),
                            w2.as_u8(),
                            w2s.as_u8(),
                            std::ptr::null(),
                            (self.s.sh_out_r.ptr as *mut f32).wrapping_add(r * dim),
                            1,
                            dim as i32,
                            sh_il as i32,
                            sh_st,
                        )?;
                        continue;
                    }
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
        // VERIFY_FORK join: the shared half is fully issued. Wait the side stream
        // back on the main stream, then merge exactly as the serial path did —
        // the routed down-reduce wrote `moe_out_r` on the main stream, so this add
        // sees the same two operands in the same order. One element-wise launch
        // over `m*dim` is bit-identical to the `m` per-row adds it replaces (the
        // add is range-disjoint per row, which is the same argument
        // `shared_expert_mrows`' own merge uses). A no-op when the fork did not
        // take. `dual` implies `shared_rank && sh_w6 && !skip_shared_expert`, so
        // the shared half always ran when this branch is reached.
        if dual {
            self.dev.dual_chain_join()?;
            self.dev.add_inplace_raw(
                self.s.moe_out_r.ptr as *mut std::ffi::c_void,
                self.s.sh_out_r.ptr as *const c_void,
                (m * dim) as i64,
            )?;
        }

        // ---- the MoE all-reduce ----
        // Payload = the m rows of `moe_out_r`. Byte-wise and ascending-rank, so
        // each row's sum is the single-row AR's sum for that row. Same
        // `DSV41_VERIFY_AR_FOLD` contract as the attention side: when the fold
        // runs, `layer_rows` skips its `hc_post_rows`.
        let mut folded = false;
        if let Some(c) = self.comm.clone() {
            folded = self.ar_hc_post_fold_rows(
                &c,
                self.s.moe_out_r.ptr as *mut std::ffi::c_void,
                fb(mdim),
                m,
            )?;
            if !folded {
                c.all_reduce_inplace(self.s.moe_out_r.ptr as *mut std::ffi::c_void, fb(mdim))?;
            }
            c.end_round();
        }
        Ok(folded)
    }

    /// The shared expert as ONE multi-row pass (`DSV41_SH_EXP_MROWS`, DEFAULT
    /// OFF): the m-row twin of the per-row loop in [`Self::moe_rows`].
    ///
    /// # What it removes
    ///
    /// The shared expert is a SINGLE expert applied to every row, so its weights
    /// are identical across the block — yet `moe_rows` reads them once per row.
    /// At the production shape (m = 5, `sh_il` = 288, dim = 5120) the per-row
    /// loop streams 4.42 MB and issues 4-5 launches per row, i.e. 26.5 MB and
    /// ~25 launches per layer = ~0.89 GB/step and ~960 launches/step of pure
    /// repeat traffic. The routed experts' bytes are irreducible (their rows
    /// select DIFFERENT experts); this one's divide by m.
    ///
    /// # The four steps, each ONE launch
    ///
    /// 1. `quant_rows(xn_r, m, dim)` — the block's fp8 activation into
    ///    `xq_r`/`xsc_r`. `quant_kernel` is one thread-group per (row, block) and
    ///    each row's amax is its own 32-element shuffle reduction, so this emits
    ///    byte-for-byte what m per-row `quant1` launches emit.
    /// 2. two `gemm_fp8_mrows` into `sh_act_r` at the `[row][2*sh_il]` layout
    ///    (w1 at `+0`, w3 at `+sh_il`, `out_stride = 2*sh_il`) — the same
    ///    weight-stationary kernel the projections use, whose per-row output is
    ///    bit-identical to the `m == 1` GEMV, i.e. to `gemm_fp8_mx` and hence
    ///    (transitively, by `gemm_fp8_mx2`'s own documented contract) to the two
    ///    families of the single launch the per-row path issues.
    /// 3. `swiglu_limit_q(rows = m)` — one launch for the whole block. The kernel
    ///    indexes `gate_up + r*2*inter`, which IS the layout step 2 wrote, and
    ///    its per-(row, block) arithmetic is A4's, which the `swiglu_q` case in
    ///    `tests_dsv41_glue.cu` asserts bit-exact against (`swiglu_limit` +
    ///    `quant1`). `inter % 32 == 0` — which `sh_il % 32 == 0` implies for the
    ///    TP-sharded case and which is checked here — is what makes every scale
    ///    block land inside one 32-row tile.
    /// 4. `gemm_fp8_mrows(w2)` into the `sh_out_r` scratch, then ONE
    ///    `add_inplace_raw` of `m*dim` elements. The per-row path's default
    ///    (A5 off) is exactly this pair per row — a write plus an element-wise
    ///    add — and the add's ranges are disjoint per row, so one launch over
    ///    `m*dim` is bit-identical to m launches over `dim`.
    ///
    /// # Fallback (the reference this was verified against)
    ///
    /// Returns `Ok(false)` — leaving the per-row loop to run — when the gate is
    /// off, the shape cannot take a step (`sh_il % 32`, `dim % 32`,
    /// `m > VERIFY_ROWS`), the `.so` predates `dsv41_gemm_fp8_mrows`, or any
    /// launcher declines. A PARTIAL attempt is harmless by construction:
    /// `moe_out_r` is touched only by the final add, and everything else written
    /// (`xq_r`/`xsc_r`/`sh_act_r`/`sh_out_r`) is scratch the fallback rewrites.
    /// # The fused arm in front of it (`DSV41_SH_EXP_FUSED`, default OFF)
    ///
    /// Tried FIRST: one `dsv41_gemm_fp8_sh_pair` per row (the symbol
    /// `chain_dev.rs`'s `sh_pair()` gate was written for and never called),
    /// replacing that row's `gemm_fp8_mrows(w1)` + `gemm_fp8_mrows(w3)` +
    /// `swiglu_limit_q` + `gemm_fp8_mrows(w2)` with ONE launch — phase 1 walks
    /// both weight rows in one warp, applies the swiglu epilogue in-register and
    /// emits the fp8 pair, a grid barrier joins it to phase 2 (the w2 GEMV).
    /// `quant_rows` (step 1) and the final `add_inplace_raw` (step 4's second
    /// half) stay: the kernel consumes an fp8 activation and its phase 2 STORES
    /// into `out` (no add epilogue), so neither is removable from this side. Both
    /// envs (`DSV41_SH_EXP_FUSED=1`, `DSV41_SH_PAIR=1`) and the symbol must be
    /// present; any decline falls through to the pass below.
    #[allow(clippy::too_many_arguments)]
    fn shared_expert_mrows(
        &self,
        w1: &crate::dsv41::load::DevTensor,
        w1s: &crate::dsv41::load::DevTensor,
        w3: &crate::dsv41::load::DevTensor,
        w3s: &crate::dsv41::load::DevTensor,
        w2: &crate::dsv41::load::DevTensor,
        w2s: &crate::dsv41::load::DevTensor,
        m: usize,
        sh_il: usize,
        dim: usize,
    ) -> Result<bool> {
        let cfg = self.cfg;
        // ---- the M-row arm (`DSV41_SH_PAIR_M`) ------------------------------
        // Tried FIRST: ONE `dsv41_gemm_fp8_sh_exp_fused<m>` for the whole block,
        // replacing all m rows' (w1w3 -> swiglu -> fp8 emit | barrier | w2) with a
        // single launch that can also fold the trailing add into phase 2's store.
        // The shape gates are the kernel's own specialisation mirrored here, so a
        // shape it would decline never costs a launch: `dim % 32` is phase 1's
        // k-block count and `sh_il % 32` is both phases' 32-element scale rows
        // (and phase 2's cp.async16 row alignment). Deadlock safety is the
        // LAUNCHER's, not this caller's: it caps the grid at the residency number
        // (co_res) and declines outright when phase 1's block count does not fit
        // -- never raise the grid here.
        //
        // A decline is free by construction: the kernel is ONE launch, so either
        // the whole arm runs or `moe_out_r` is untouched (the launcher returns 2
        // before any block starts) and the arms below run exactly as before.
        if sh_pair_m()
            && self.dev.supports_sh_exp_fused()
            && m != 0
            && m <= VERIFY_ROWS
            && m <= 8
            && (sh_il % 32) == 0
            && (dim % 32) == 0
        {
            // 1) ONE fp8 quantisation of the block -- the same launch the arms
            //    below start with, so the activation is staged once either way.
            self.quant_rows(self.s.xn_r.ptr as *const f32, m, dim as i32)?;
            // The row pitch of that staging is the `cols` it was quantised with
            // (`dim` bytes / `dim/32` f32, see `quant_rows`), NOT the wider
            // `xq_r` allocation.
            let sc_pitch = dim / 32;
            let ok = self.dev.gemm_fp8_sh_exp_fused(
                self.s.xq_r.ptr as *const u8,
                self.s.xsc_r.ptr as *const f32,
                w1.as_u8(),
                w1s.as_u8(),
                w3.as_u8(),
                w3s.as_u8(),
                cfg.swiglu_limit,
                sh_il as i32,
                dim as i32,
                m as i32,
                sh_pair_m_fold(),
                // `act` is a by-product in this arm (only the fp8 pair is consumed
                // downstream) -> null drops the stores; the fallback arms below
                // rewrite the slot anyway.
                std::ptr::null_mut(),
                0,
                self.s.sh_aq_r.ptr as *mut u8,
                self.s.sh_aqsc_r.ptr as *mut f32,
                // Row pitches: the fp8 pair is `[m, sh_il]` and its scales
                // `[m, sh_il/32]` -- the same slots the M=1 arm uses.
                sh_il as i32,
                (sh_il / 32) as i32,
                w2.as_u8(),
                w2s.as_u8(),
                dim as i32,
                // out = the MoE block accumulator, `m` rows at pitch `dim`, and
                // epi_add = 1 folds `add_inplace_raw(moe_out_r, sh_out_r)` into
                // phase 2's read-modify-write (the same operand pair, so the same
                // sum).
                dim as i32,
                1,
                self.s.moe_out_r.ptr as *mut f32,
                self.dev.stream(),
            )?;
            if ok {
                return Ok(true);
            }
        }
        // ---- the fused arm (`DSV41_SH_EXP_FUSED`) ---------------------------
        // The shape gates are the kernel's own specialisation, mirrored here so a
        // shape it would decline never costs a launch: `dim % 32` is phase 1's
        // k-block count and `sh_il % 32` is both phases' 32-element scale rows
        // (and phase 2's cp.async16 row alignment). Deadlock safety is the
        // LAUNCHER's, not this caller's: `dsv41_gemm_fp8_sh_pair` caps its grid
        // at the residency number (co_res) for the barrier — never raise it here,
        // and note that phase 1 needs `ceil(sh_il/32) <= co_res` (phase 2's own
        // loop is grid-strided, so the cap can only starve phase 1).
        if sh_exp_fused()
            && sh_pair()
            && self.dev.supports_sh_pair()
            && m != 0
            && m <= VERIFY_ROWS
            && (sh_il % 32) == 0
            && (dim % 32) == 0
        {
            // 1) ONE fp8 quantisation of the block — the same launch the pass
            //    below starts with, so the activation is staged once either way.
            self.quant_rows(self.s.xn_r.ptr as *const f32, m, dim as i32)?;
            // The row pitch of that staging is the `cols` it was quantised with
            // (`dim` bytes / `dim/32` f32, see `quant_rows` (#5)), NOT the wider
            // `xq_r` allocation.
            let sc_pitch = dim / 32;
            let mut all_rows = true;
            for r in 0..m {
                let ok = self.dev.gemm_fp8_sh_pair(
                    (self.s.xq_r.ptr as *const u8).wrapping_add(r * dim),
                    (self.s.xsc_r.ptr as *const f32).wrapping_add(r * sc_pitch),
                    w1.as_u8(),
                    w1s.as_u8(),
                    w3.as_u8(),
                    w3s.as_u8(),
                    cfg.swiglu_limit,
                    sh_il as i32,
                    dim as i32,
                    // The f32 swiglu row is a by-product here (only the fp8 pair
                    // is consumed downstream); it lands in the same `[row]
                    // [2*sh_il]` slot the pass below writes, so the two layouts
                    // cannot disagree.
                    (self.s.sh_act_r.ptr as *mut f32).wrapping_add(r * 2 * sh_il),
                    (self.s.sh_aq_r.ptr as *mut u8).wrapping_add(r * sh_il),
                    (self.s.sh_aqsc_r.ptr as *mut f32).wrapping_add(r * (sh_il / 32)),
                    w2.as_u8(),
                    w2s.as_u8(),
                    dim as i32,
                    (self.s.sh_out_r.ptr as *mut f32).wrapping_add(r * dim),
                    self.dev.stream(),
                )?;
                if !ok {
                    // A mid-loop decline: fall through to the pass below. It is
                    // safe by construction — `moe_out_r` has not been touched
                    // (the add is the only write to it and it is further down)
                    // and every other buffer this arm wrote is scratch the
                    // fallback rewrites.
                    all_rows = false;
                    break;
                }
            }
            if all_rows {
                self.dev.add_inplace_raw(
                    self.s.moe_out_r.ptr as *mut std::ffi::c_void,
                    self.s.sh_out_r.ptr as *const c_void,
                    (m * dim) as i64,
                )?;
                return Ok(true);
            }
        }
        if !sh_exp_mrows()
            || m == 0
            || m > VERIFY_ROWS
            || (sh_il % 32) != 0
            || (dim % 32) != 0
            || !self.dev.supports_gemm_fp8_mrows()
        {
            return Ok(false);
        }
        // 1) ONE fp8 quantisation for the whole block.
        self.quant_rows(self.s.xn_r.ptr as *const f32, m, dim as i32)?;
        // 2) w1 | w3: one weight-stationary GEMV each, over the same activation.
        let stride = (2 * sh_il) as i32;
        let ok1 = self.dev.gemm_fp8_mrows(
            self.s.xq_r.ptr as *const u8,
            self.s.xsc_r.ptr as *const f32,
            w1.as_u8(),
            w1s.as_u8(),
            std::ptr::null(),
            self.s.sh_act_r.ptr as *mut f32,
            m as i32,
            sh_il as i32,
            dim as i32,
            stride,
        )?;
        let ok3 = self.dev.gemm_fp8_mrows(
            self.s.xq_r.ptr as *const u8,
            self.s.xsc_r.ptr as *const f32,
            w3.as_u8(),
            w3s.as_u8(),
            std::ptr::null(),
            (self.s.sh_act_r.ptr as *mut f32).wrapping_add(sh_il),
            m as i32,
            sh_il as i32,
            dim as i32,
            stride,
        )?;
        if !(ok1 && ok3) {
            return Ok(false);
        }
        // 3) swiglu + the fp8 pair the w2 GEMV reads, all m rows in one launch.
        let swiglu_ok = swiglu_q()
            && self.dev.supports_swiglu_q()
            && self.dev.swiglu_limit_q(
                self.s.sh_act_r.ptr as *mut f32,
                m as i32,
                sh_il as i32,
                cfg.swiglu_limit,
                self.s.xq_r.ptr as *mut u8,
                self.s.xsc_r.ptr as *mut f32,
            )?;
        if !swiglu_ok {
            return Ok(false);
        }
        // 4) w2 for all rows, then ONE element-wise add into the accumulator.
        let ok2 = self.dev.gemm_fp8_mrows(
            self.s.xq_r.ptr as *const u8,
            self.s.xsc_r.ptr as *const f32,
            w2.as_u8(),
            w2s.as_u8(),
            std::ptr::null(),
            self.s.sh_out_r.ptr as *mut f32,
            m as i32,
            dim as i32,
            sh_il as i32,
            dim as i32,
        )?;
        if !ok2 {
            return Ok(false);
        }
        self.dev.add_inplace_raw(
            self.s.moe_out_r.ptr as *mut std::ffi::c_void,
            self.s.sh_out_r.ptr as *const c_void,
            (m * dim) as i64,
        )?;
        Ok(true)
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
    ///
    /// The buffers are PASSED IN rather than read off the single-row scratch:
    /// `layer()` hands the single-row slots (`s.h` / `premix_slot` / `s.post` /
    /// `s.comb`) and `layer_rows()` the m-row ones (`s.h_r` / `premix_row_slot` /
    /// `s.post_r` / `s.comb_r`). Every kernel below indexes both with the same
    /// per-row strides — `x` at `r*hc*dim`, `pre_out`/`post_out` at `r*hc`,
    /// `comb_out` at `r*hc*hc` — so `rows` is the only other difference between
    /// the two callers.
    ///
    /// `xq`/`xsc` are the T1 fp8 staging. The tail kernel emits it in a
    /// SINGLE-ROW layout (`xq[c]`, no row base), so a multi-row caller passes
    /// null and quantises its own rows with `quant_rows` instead (`layer_rows`
    /// into `s.xq_r`/`s.xsc_r`).
    ///
    /// `truncate` is the fused front end's `DSV41_BF16_TRUNCATE` arm, and it is
    /// a PARAMETER rather than a direct `bf16_truncate()` read on purpose: the
    /// two callers want opposite things. `layer()` (single row) passes
    /// `bf16_truncate()` — the decode path is the one the gate was validated on.
    /// `layer_rows()` (m rows) passes `false`: letting the gate reach the verify
    /// chain put `BF16_TRUNCATE` on the verify for the first time and broke the
    /// zero-Latin baseline (the same interaction A1-a fixed by hard-coding
    /// `false` in [`Self::collapse_norm_rows`], 2026-09-12). The A2 arm must
    /// therefore keep the raw chain's historical f32 behaviour.
    #[allow(clippy::too_many_arguments)]
    fn hc_mixes_auto(
        &mut self,
        x: *const f32,
        hc_fn: *const f32,
        hc_scale: *const f32,
        hc_base: *const f32,
        pre_out: *mut f32,
        post_out: *mut f32,
        comb_out: *mut f32,
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
        rows: usize,
        truncate: bool,
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
                x,
                hc_fn,
                hc_scale,
                hc_base,
                norm_w,
                pre_collapse,
                pre_out,
                post_out,
                comb_out,
                out,
                rows as i32,
                hc as i32,
                dim as i32,
                sinkhorn_iters,
                eps,
                eps_norm,
                xq,
                xsc,
                truncate,
            )?
        } else if rows == 1 && Self::hc_persist_mb() && self.dev.supports_hc_persist_mb() {
            self.dev.hc_front_persist_mb(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                norm_w,
                pre_collapse,
                pre_out,
                post_out,
                comb_out,
                out,
                rows as i32,
                hc as i32,
                dim as i32,
                sinkhorn_iters,
                eps,
                eps_norm,
                xq,
                xsc,
                truncate,
            )?
        } else if rows == 1 && Self::hc_persist() && self.dev.supports_hc_persist() {
            // Stage-C persistent prototype (DSV41_HC_PERSIST=1, default OFF): the
            // whole front end as ONE phase-machine block instead of the two-launch
            // dots+tail pair.
            self.dev.hc_front_persist(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                norm_w,
                pre_collapse,
                pre_out,
                post_out,
                comb_out,
                out,
                rows as i32,
                hc as i32,
                dim as i32,
                sinkhorn_iters,
                eps,
                eps_norm,
                xq,
                xsc,
                truncate,
            )?
        } else {
            self.dev.hc_front(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                norm_w,
                pre_collapse,
                pre_out,
                post_out,
                comb_out,
                out,
                rows as i32,
                hc as i32,
                dim as i32,
                sinkhorn_iters,
                eps,
                eps_norm,
                xq,
                xsc,
                truncate,
            )?
        };
        if fused {
            return Ok(true);
        }
        self.dev.hc_mixes(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            pre_out,
            post_out,
            comb_out,
            rows as i32,
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

/// The verify (m-row) hc chain's A1 wiring gate (`DSV41_HC_VERIFY_FUSE`,
/// default ON; "0" is the A/B arm). The m-row path (`layer_rows`) used to run the
/// RAW ten-launch chain — `hc_mixes` / `hc_collapse` / `norm_rows` / `hc_post` +
/// the `h2_r` copy, twice per layer — even though all three decode fusions
/// (`FUSE_B1`'s `hc_collapse_norm`, `FUSE_C`'s `hc_post_inplace`, and
/// `hc_mixes_auto`'s front end) were already in the tree and already ON for the
/// single-row `layer()`. This arms the two that need no new shape:
///
///  * A1-a: the collapse + norm pair becomes `hc_collapse_norm(rows = m)` — an
///    existing kernel with a native `rows` grid, bit-exact for the same reason
///    `FUSE_B1` is (`layer()`'s A/B). **It passes `truncate = false`** (the
///    2026-09-12 fix — see `collapse_norm_rows`): the verify chain must not take
///    `DSV41_BF16_TRUNCATE`, which is what first broke the zero-Latin baseline.
///  * A1-b: `hc_post` + the `h2_r` copy become `hc_post_inplace_rows(rows = m)`
///    — the m-row entry of the segment-C kernel. Also *additionally* gated on
///    the symbol being present, so a `.so` built before this change silently
///    keeps the staging pair (see `Device::supports_hc_post_inplace_rows`).
///
/// Both halves also honor the decode gates they belong to (`DSV41_FUSE_B1=0` /
/// `DSV41_FUSE_C=0` turn off their respective half), so the four independent A/B
/// arms of `layer()` exist here too. `DSV41_HC_VERIFY_FUSE=0` restores the raw
/// chain bit for bit — the safe arm if a same-binary A/B is wanted.
fn hc_verify_fuse() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // DEFAULT OFF (baseline-breakage fix, 2026-09-12): the fuse routes the
    // verify's hc chain through the fused kernels (hc_collapse_norm with the
    // truncate param + hc_post_inplace_rows), which reaches BF16_TRUNCATE
    // into the verify for the FIRST time. The GPU A/B (9b55ea04) showed this
    // arm BREAKS the zero-Latin baseline when combined with the P0-series
    // commits (the exact interaction TBD — the fusion alone was clean at
    // 1ddff9c): HC_VERIFY_FUSE=0 restores zero Latin at HEAD. The default is
    // now OFF (=0 semantics: the historical 10-launch chain); set
    // DSV41_HC_VERIFY_FUSE=1 to re-arm after the interaction is understood.
    *F.get_or_init(|| {
        std::env::var("DSV41_HC_VERIFY_FUSE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// The verify path's AR fold (`DSV41_VERIFY_AR_FOLD`, default OFF): `layer_rows`'
/// two `hc_post_inplace_rows` launches move into the pubred epilogue of the AR
/// that produced their `x` (see `ferrite_p2p_ar_v5_hcpost_rows`), 2 launches -> 1
/// at each site — 80 of the A1 chain's 240 launches, ≈ −0.24ms at ~3µs/launch.
///
/// DEFAULT OFF: unlike `layer()`'s `HCPOST_EPI` (single-row, already A/B'd), this
/// folds a chain the verify path has never folded, and the m-row epilogue lives in
/// a different translation unit than `dsv41_hc_post_inplace_rows_kernel` — the
/// same class of change as `HCPOST_EPI`, hence the same "off is the A/B arm" rule.
///
/// The fold only runs when the caller would otherwise have taken the IN-PLACE
/// rows post: it is additionally gated on [`fuse_c`] (the staging `hc_post` +
/// `h2_r` copy pair is a different data path) and on [`Self::hc_verify_fuse`]
/// (A1) so `DSV41_HC_VERIFY_FUSE=0` keeps restoring the raw chain bit for bit,
/// and on the `.so` carrying `ferrite_p2p_ar_v5_hcpost_rows`
/// ([`Device::supports_ar_hcpost_rows`]).
fn verify_ar_fold() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_VERIFY_AR_FOLD").map(|v| v != "0").unwrap_or(false))
}

/// A2 (`DSV41_HC_FRONT_ROWS`, default OFF): route `layer_rows`'s two per-layer
/// `hc_mixes` calls through [`ChainDev::hc_mixes_auto`] at `rows = m`, i.e. give
/// the verify chain the fused spread front end (EARLY collapse/rmsnorm on main,
/// dots + LATE ss/sigmoid/sinkhorn/comb on the side stream) that `layer()` has.
/// DEFAULT OFF: this is the larger change of the pair — it needs the side-stream
/// event pairing to hold under the m-row block (it does: `hc_front_split` is
/// rows-agnostic and keeps record/wait adjacent), and it changes which kernel
/// produces `post`/`comb` on a path with no GPU-side A/B available here.
///
/// The fused front end also absorbs the collapse half, so when it runs the
/// caller skips its `hc_collapse_norm`; when it declines (no `DSV41_HC_FRONT`,
/// no `dsv41_hc_front_split` in the `.so`, `FUSE_B1=0`) `hc_mixes_auto` falls
/// back to the same `hc_mixes(rows = m)` the raw chain issues, so every
/// intermediate step of the gate is either today's chain or a bit-exact subset
/// of it.
fn hc_front_rows() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_HC_FRONT_ROWS").map(|v| v != "0").unwrap_or(false))
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

/// DSpark tap ALIGNMENT gate (`DSV41_TAP_INPUT=1`, default OFF).
///
/// Moves the target-hidden tap from the END of `layer()` / `layer_rows()` to
/// their START, where `h` is the layer's ATTENTION INPUT (the previous layer's
/// output, as the engram write-back left it) instead of the layer's COMPLETED
/// output.
///
/// The capture point is fixed by the reference, and it is the START:
/// `ref_inference/model.py:1261-1267` runs, per layer `i`, the engram injection
/// (1262-1263), then `main_hiddens.append(h.mean(dim=2))` (1265-1266), then
/// `h, pre_mix = layer(h, ...)` (1267) — and the comment directly above the tap
/// says so out loud: "the MTP head reads the attention input of its target
/// layers, not their output". `chain.rs`'s CPU reference (the path `dspark_parity`
/// compares against) is wired identically: engram at 568-609, tap at 610-621,
/// `layer_forward` at 623.
///
/// The device chain instead tapped at the END of the block, i.e. it recorded
/// layer `i`'s completed output — which is the official tap of layer `i+1`. With
/// targets 37/38/39 that is `(38/39/40)`: the draft's `main_x` and every token
/// projected from it were systematically one full layer ahead of what the head
/// was trained on.
///
/// Both arms issue the SAME one-block `hc_collapse` against the SAME `[1/hc]`
/// weights (`dspark_pre_mean`) over the SAME stream `h`; only the point in the
/// stream differs, and the two sites are mutually exclusive (the end-of-layer
/// hook is skipped when this gate is armed). `0`/unset therefore reproduces the
/// historical numbers bit for bit and the gate is a pure A/B switch on one
/// binary.
fn tap_input() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_TAP_INPUT").map(|v| v != "0").unwrap_or(false))
}

    fn layer(&mut self, layer: usize, pos: usize, pa: usize) -> Result<usize> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let ld = &self.w.layers[layer];

        // ---------------- DSpark tap (layer INPUT) ----------------
        // `DSV41_TAP_INPUT=1`: the reference's capture point — the block has not
        // run yet, so `h` still holds the previous layer's output with this
        // layer's engram write-back already folded in (`engram_apply` runs in
        // `step_body` BEFORE this call, the reference's 1262-1263 order). The
        // tap layers (37/38/39) are not engram layers, but keeping the write-back
        // ahead of the tap is what makes the ordering match the reference
        // statement for statement. The end-of-layer hook below is skipped while
        // this one is armed.
        if cfg.dspark_armed() && Self::tap_input() {
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
            self.s.h.ptr as *const f32,
            ld.hc_attn_fn.as_ref().unwrap().as_f32(),
            ld.hc_attn_scale.as_ref().unwrap().as_f32(),
            ld.hc_attn_base.as_ref().unwrap().as_f32(),
            self.premix_slot(1).ptr as *mut f32, // attn_pre
            self.s.post.ptr as *mut f32,
            self.s.comb.ptr as *mut f32,
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
            1,
            // Single-row decode form: this is the path `DSV41_BF16_TRUNCATE` was
            // validated on, so `layer()` keeps reading the gate. Only the m-row
            // verify arm passes `false` (see `hc_mixes_auto`).
            bf16_truncate(),
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
                    bf16_truncate(),
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
            self.s.h.ptr as *const f32,
            ld.hc_ffn_fn.as_ref().unwrap().as_f32(),
            ld.hc_ffn_scale.as_ref().unwrap().as_f32(),
            ld.hc_ffn_base.as_ref().unwrap().as_f32(),
            self.premix_slot(2).ptr as *mut f32, // ffn_pre -> next layer
            self.s.post.ptr as *mut f32,
            self.s.comb.ptr as *mut f32,
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
            1,
            // Same as the attention block above: the single-row decode form keeps
            // the `DSV41_BF16_TRUNCATE` gate.
            bf16_truncate(),
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
                    bf16_truncate(),
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
        // The historical capture point: the layer's COMPLETED output (after
        // attention AND ffn). Kept as the default arm; `DSV41_TAP_INPUT=1` selects
        // the reference's layer-INPUT point instead (see `tap_input`), in which
        // case this hook must NOT also run — it would overwrite the input tap with
        // the output one and silently restore the one-layer skew.
        //
        // NOTE the two are not interchangeable: this site samples layer `i`'s
        // output, which IS the reference's tap for layer `i+1`, so with targets
        // 37/38/39 this arm is the official (38/39/40).
        if !Self::tap_input() && cfg.dspark_armed() {
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
        // DSV41_LAYER_DUMP: the residual stream this block just finished writing
        // (`s.h`, the same tensor the dspark tap above collapses). LAST in the
        // layer on purpose — the norm then describes the block's COMPLETED output,
        // the quantity the bisection compares layer by layer.
        if layer_dump() {
            self.dump_layer_norm(layer)?;
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

    /// The `DSV41_TOPK_DUMP` probe: read back `n` floats of the head's logits row
    /// and print its top [`topk_dump_n`] entries as ONE `[topk]` line
    ///
    /// ```text
    /// [topk] pos=0 #1 id=12345 val=12.34 #2 id=67890 val=11.98 ...
    /// ```
    ///
    /// `base` is the row's first entry's index in the FULL vocabulary and `n` its
    /// width. With `DSV41_HEAD_SLICE` on and `world > 1` the row is this rank's
    /// slice, so the printed ids are global while the SET is slice-local (a true
    /// global top-N needs `world == 1` or `DSV41_HEAD_SLICE=0` — the alignment
    /// experiment's single-GPU eager step is the full-vocabulary case).
    /// The ranking is [`vrow0_topk`]'s: `f32::total_cmp` descending, ties to the
    /// LOWER index — the same rule both argmax kernels implement.
    fn dump_topk(&self, pos: usize, base: u32, n: usize) -> Result<()> {
        let mut lg = vec![0f32; n];
        let b = Device::view(self.s.logits.ptr, n * std::mem::size_of::<f32>());
        self.dev.download_f32(&b, &mut lg)?;
        let mut line = format!("[topk] pos={pos}");
        for (i, (id, v)) in vrow0_topk(&lg, base, topk_dump_n()).iter().enumerate() {
            line.push_str(&format!(" #{} id={} val={}", i + 1, id, v));
        }
        eprintln!("{line}");
        Ok(())
    }

    /// The `DSV41_LAYER_DUMP` probe: ONE `[layer]` line carrying the L2 norm of
    /// the residual stream the block just finished writing
    ///
    /// ```text
    /// [layer] l=0 norm=123.45
    /// ```
    ///
    /// The buffer is `s.h`, the whole `hc*dim` hyper-connection stream (the same
    /// tensor `layer()`'s dspark tap collapses over), NOT its `hc`-mean — a
    /// collapse would add a kernel to a diagnostic path and blur which of the two
    /// moved. `pos` is not printed: the caller knows which step it drove.
    fn dump_layer_norm(&self, layer: usize) -> Result<()> {
        let n = self.cfg.hc_mult * self.cfg.dim;
        let h = self.dl(self.s.h.as_f32(), n)?;
        let mut ss = 0f64;
        for &v in &h {
            ss += (v as f64) * (v as f64);
        }
        eprintln!("[layer] l={layer} norm={:.6}", ss.sqrt());
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
                // W2-MROWS: the eager row is b = m = 1 — scalar counter, `topk`
                // pitch and the old rope formula, bit-identical.
                std::ptr::null(),
                0,
                0,
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
                // W2-MROWS: b = m = 1, so both new inputs are inert.
                std::ptr::null(),
                0,
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
            // The e4m3 sibling (`DSV41_EXPERT_TCGEN05_E4M3`, default OFF) has its
            // own gate, symbol and one-shot notice, for exactly the same reasons
            // and with exactly the same three refusals. It is the ONLY tcgen05
            // arm the e4m3 activation can use: `kind::mxf4` is e2m1 x e2m1.
            if expert_tcgen05_e4m3() {
                if !self.dev.supports_expert_tcgen05_e4m3() {
                    tcgen05_e4m3_skipped_note(
                        "the loaded .so has no such symbol (rebuild with \
                         DSV41_BUILD_TCGEN05_E4M3 unset -- a stock build does define it)",
                        false,
                    );
                } else if ilv {
                    tcgen05_e4m3_skipped_note(
                        "the routed gate/up pools are interleaved (DSV41_EXPERT_ILV) and only \
                         the fused GEMV body can read that layout",
                        true,
                    );
                } else if !batched {
                    tcgen05_e4m3_skipped_note(
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
            //
            // e4m3 DIRECT (`DSV41_EXPERT_ACT_E4M3`, default OFF) — see the gate's
            // doc comment. The activation takes the OFFICIAL `act_quant(e4m3,
            // block=32)` form (ONE byte per value, f32 power-of-two scale per 32,
            // the SAME `dsv41_quant_fp8` entry the dense fp8 chain uses) and is
            // consumed by ONE gate/up pass. Allocated with the quantiser's own
            // layouts, so the expert kernels' per-row pointer derivation is the
            // only thing that changes: `xq4` is `dim` bytes (>= the `dim/2` the
            // fp4 arm packs into) and `xsc4` is `dim/32` f32 for both.
            let e4m3 = expert_act_e4m3() && self.dev.supports_expert_act_e4m3();
            if expert_act_e4m3() && !e4m3 {
                act_e4m3_skipped_note();
            }
            if e4m3 {
                self.dev.quant_fp8(
                    self.s.xn.ptr as *const f32,
                    self.s.xq4.ptr as *mut u8,
                    self.s.xsc4.ptr as *mut f32,
                    1,
                    dim as i32,
                    32,
                    true,
                )?;
            } else {
                self.dev.quant_fp4(
                    self.s.xn.ptr as *const f32,
                    self.s.xq4.ptr as *mut u8,
                    self.s.xsc4.ptr as *mut f32,
                    1,
                    dim as i32,
                    32,
                    true,
                )?;
            }
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
                // ---- tcgen05 gate/up: TWO variants, picked by the activation ----
                // Both write the same UNFUSED [2*inter] gate|up layout, take the
                // same 18 arguments and read the same activation SCALES; they
                // differ ONLY in the activation's byte layout, so the variant
                // follows the quantisation the step already chose:
                //   e2m1 packed (`!e4m3`) -> tc5::mxf4 (`kind::mxf4`, 2X)
                //   e4m3 bytes  (`e4m3`)  -> tc5::e4  (`kind::mxf8f6f4`, 1X)
                // Until `DSV41_EXPERT_TCGEN05_E4M3` existed the e4m3 arm was
                // simply REFUSED here (`let tcgen05 = !e4m3 && ...`), i.e. arming
                // the official e4m3 activation silently disabled the tcgen05
                // path. Each variant keeps its OWN runtime gate and its OWN
                // symbol, so the two are independently A/B-able and a `.so`
                // carrying only the e2m1 entry point still works (with the
                // one-shot refusal notice above).
                let tc_e4m3 = e4m3
                    && expert_tcgen05_e4m3()
                    && self.dev.supports_expert_tcgen05_e4m3()
                    && !ilv;
                let tc_mxf4 = !e4m3
                    && expert_tcgen05_mxf4()
                    && self.dev.supports_expert_tcgen05_mxf4()
                    && !ilv;
                let mut ran_tc = false;
                if tc_e4m3 {
                    // `xq4` holds ONE e4m3 byte per value here (the `quant_fp8`
                    // branch above), which is exactly the row this kernel wants;
                    // `xsc4` is the same [dim/32] f32 vector either way.
                    ran_tc = self.dev.expert_tcgen05_gate_up_e4m3(
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
                } else if tc_mxf4 {
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
                let gateup_fused = !ran_tc
                    && gateup_fuse()
                    && self.dev.supports_gateup_fuse()
                    && expert_fp4_mode() == 2;
                let act_slot = if gateup_fused {
                    inter_local as i64
                } else {
                    (2 * inter_local) as i64
                };
                let down_slot = dim as i64;
                if !ran_tc {
                    // ONE pass: `xq4`/`xsc4` hold the activation in whichever form
                    // the quantisation above produced (e4m3 bytes when `e4m3` is
                    // set, packed e2m1 otherwise) and `act_e4m3` tells the kernel
                    // which decoder to use. `act_slot` follows the fusion decision
                    // above on both arms.
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
                        // Interleaved gate/up pools (DSV41_EXPERT_ILV): the
                        // kernel's fused body derives the up bytes from the
                        // gate pointer and reads one LDG.128 per group.
                        // `w3_base`/`w3_stride` still point at the same
                        // doubled region (the loader makes w3's view alias
                        // w1's), they are simply not read.
                        ilv as i32,
                        e4m3 as i32,
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
                let gateup_fused = !ran_tc
                    && gateup_fuse()
                    && self.dev.supports_gateup_fuse()
                    && expert_fp4_mode() == 2
                    && (dim as i32) % 512 == 0;
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
                    // ONE pass: `xq4`/`xsc4` hold the activation in whichever form
                    // the quantisation above produced (e4m3 bytes when `e4m3` is
                    // set, packed e2m1 otherwise); `expert_gate_up_fp4_indirect`
                    // is epi_mode 1 (clamp + WRITE) and `act_e4m3` selects the
                    // decoder.
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
                        e4m3 as i32,
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

// ---------------------------------------------------------------------------
// The SpecStep seam — the four-phase skeleton shared with GLM's MTP chain
// ---------------------------------------------------------------------------

/// The DSV41 DSpark chain as one [`SpecStep`], the same four phases GLM's
/// `TpCluster::mtp_step` exposes (`ferrite-exec/src/tp.rs`):
/// draft → verify → accept → commit.
///
/// `ANCHOR_IS_IN_BLOCK = false` is the layout fact the shared accept chain
/// needs: this verify block is `[d1..DSPARK_DRAFTS]` and does NOT contain the
/// anchor row — the plain single-row step forwards it — so `SpecStep::accept`
/// takes the judge array with the anchor's argmax (`next`) leading it and the
/// block rows shifted by one (`drafts[j]` vs `verify_out[j - 1]`), and the
/// accepted count is the draft prefix itself (0..=DSPARK_DRAFTS).
///
/// The phases stay fused in one call on purpose: the snapshot/rollback pair
/// brackets the verify (see `DevChain::dspark_spec_step`), so the trait exposes
/// the step, not its parts.
impl<'c> SpecStep for DevChain<'c> {
    // Two lifetimes, both needed: the OUTER one borrows the draft device for
    // this step, the INNER one is `DsparkDev`'s own (it borrows the device,
    // config and weights for the whole serving scope). They cannot be collapsed
    // — `&mut T` is invariant in `T`.
    type Step<'a, 'b>
        = (&'a mut DsparkDev<'b>, u32, usize)
    where
        'b: 'a;
    type Report = DsparkSpecReport;
    const ANCHOR_IS_IN_BLOCK: bool = false;

    fn spec_step<'a, 'b>(&mut self, step: Self::Step<'a, 'b>) -> Result<Self::Report>
    where
        'b: 'a,
    {
        let (dspark, token, pos) = step;
        self.dspark_spec_step(dspark, token, pos)
    }
}
