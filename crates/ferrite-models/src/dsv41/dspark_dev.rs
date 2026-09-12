//! DSpark draft, device path: the `mtp.*` block-level draft of DeepSeek-V4.1.
//!
//! The host reference is [`crate::dsv41::dspark`] (CPU numerics) plus
//! [`crate::dsv41::chain::forward_spec`] (the call order). Both were written as
//! a CPU oracle; this module is the device twin, and it reuses the *main
//! chain's* kernels throughout ([`Device`]) because a draft block is isomorphic
//! to a backbone block: the checkpoint's `mtp.{s}.*` tensors are the same
//! shapes as a layer's, so the sequence is
//!
//! ```text
//!   forward_embed   main_proj(main_h) -> main_norm -> main_x  [1, dim]
//!                   embed[ids] hc-expanded                    [bs, hc, dim]
//!   per mtp block   hc_mixes -> collapse+norm -> DSparkAttention -> hc_post
//!                   hc_mixes -> collapse+norm -> MoE             -> hc_post
//!   forward_head    collapse -> norm -> head -> bs x markov head -> confidence
//! ```
//!
//! Only the draft *forward* exists (the reference ships no speculative loop).
//! The caller owns the orchestration: [`DsparkDev::note_target_hidden`] records
//! the target layers' attention input as the backbone walks past them, and
//! [`DsparkDev::draft_forward`] runs one draft block once the backbone has
//! sampled its next token.
//!
//! ## What this first cut deliberately does NOT do
//!
//! * No fusion gates, no side streams, no CUDA-graph-safe device counters. The
//!   per-step launch count is high on purpose: every call below is a straight
//!   transcription of the host reference onto an existing ABI, so the
//!   numerical questions and the scheduling questions stay separable.
//! * The grouping of the output projection loops one (group, row) pair at a
//!   time — see [`DsparkDev::draft_attention`].
//! * The head projection reads the whole (bf16) vocabulary head once per draft
//!   row — see [`DsparkDev::draft_head`].

use std::ffi::c_void;
use std::sync::Arc;

use ferrite_types::{FerriteError, Result};

use crate::dsv41::config::Dsv41Config;
use crate::dsv41::device::{DevBuf, Device};
use crate::dsv41::load::{Dsv41DevWeights, DevTensor, LayerDev};
use crate::dsv41::tp::Collective;
use crate::dsv41::unit_dump::{self, UnitDump};

/// Grid cap of `dsv41_dspark_markov_head`'s per-block partial scratch; must
/// match `DSPARK_MARKOV_MAX_BLOCKS` in `kernels/cuda/dsv41_glue.cu`.
const MARKOV_MAX_BLOCKS: usize = 2048;

/// The chain's target-hidden tap is a fixed `[DSPARK_TAP_SLOTS, dim]` buffer
/// (`DevChain`'s `dspark_tap`, filled by the `layer()` hook at the slot
/// `dspark_target_slot` names). It is stated here as well as there because
/// [`DsparkDev::import_tap`] copies that many slots in one go — a config
/// listing more targets than this would already overflow the CHAIN's buffer
/// inside `layer()`, not this one.
pub const DSPARK_TAP_SLOTS: usize = 3;

/// How many of the draft's sampled tokens one speculative step verifies.
/// `draft_forward` samples `dspark_block_size` (6) tokens into `ids[1..=bs]`;
/// the orchestration (`chain_dev.rs::dspark_shadow_step`) verifies the first
/// `DSPARK_DRAFTS` of them, and the accept chain's bonus token brings the
/// emitted block to `DSPARK_DRAFTS + 1` — exactly the configured block size.
/// Declared here rather than at the call site so the D2H accessor below and the
/// orchestration cannot drift apart about the width.
pub const DSPARK_DRAFTS: usize = 5;

/// `DSV41_DSPARK_TRACE=1` prints the draft's geometry once, for bring-up.
fn trace() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_DSPARK_TRACE").map(|v| v != "0").unwrap_or(false))
}

/// `DSV41_DRAFT_HEAD_FOLD=0` swaps the draft's head GEMV to the PER-ROW
/// `gemv_bf16` (the v1 program) instead of the folded multi-row
/// `head_gemv_bf16_mrows`. **Default ON (the folded path): this gate exists as
/// an A/B, not as a fallback.**
///
/// The folded kernel's header claims bit-identity with the single-row launch it
/// replaces (C1-C6); the verify's head measured it as false — see
/// [`crate::dsv41::chain_dev::verify_head_fold`]'s doc, where folded gives
/// `verify_out[0] == next` on 33% of rows vs 9% for the per-row `gemv_bf16`.
/// The production single-row head is `gemv_bf16_kernel` in `dsv41_glue.cu:368`
/// (the scalar `c += 32` chain) because `gemv_bf16_v2_wanted(n)` needs
/// `n < 2048` and the head's `n = vocab_size = 129280` (device.rs). So folding
/// the head is a NUMERICAL change, not a free scheduling one: the folded kernel
/// pairs the fma/decode differently ⇒ ~1e-3 on the logits ⇒ a near-tie argmax
/// can flip.
///
/// That matters here because the draft's top-1 IS `drafts[0]` (`ids[1]` in the
/// Markov loop below): if the fold lowers it, the accept chain loses its first
/// link. The A/B tests exactly that — same step, `drafts[0] == next` rate and
/// the `k_acc` histogram, folded vs per-row. Read once and cached (the house
/// rule for hot-path gates).
fn draft_head_fold() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("DSV41_DRAFT_HEAD_FOLD")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// ROUTED-EXPERTS ROW-FOLD (`DSV41_DRAFT_MOE_MROWS=1`, DEFAULT OFF): the draft's
/// routed-expert half ([`DsparkDev::draft_moe`]) runs as ONE `rows = bs` launch
/// per stage — gate/up, the optional separate swiglu, the fused down+reduce —
/// where the per-row arm issues `bs` `rows = 1` launches of the SAME three
/// launchers.
///
/// **What it removes.** `rows` is the launchers' THIRD grid dimension
/// (`blockIdx.z`), so the per-row form pays `bs` kernel launches per stage per
/// MTP block exactly as the pre-`moe-mrows-impl` call site did; the single call
/// derives every row's pointers internally (see the layout contract at the
/// launch site). Both arms are the same kernels with the same arguments, so the
/// A/B is a launch-shape comparison and nothing else.
///
/// **Why it ships OFF.** The kernels' ROW INDEPENDENCE block argues row r of a
/// `rows = m` call is the `rows = 1` launch for that row bit for bit (rows share
/// no output, no accumulator, no smem staging — only base pointers move). That
/// argument has to be confirmed on the real draft shape by the A/B (`dspark`
/// parity + `verify_ms`) before the per-row form is retired — the same rule
/// `DSV41_SH_EXP_MROWS` and `DSV41_ROW_FOLD_GATE` follow. Read ONCE and cached:
/// this branch runs `bs` x n_mtp times per step, so a per-call getenv would be
/// the hot-path slip every other gate in this file avoids.
fn draft_moe_mrows() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_DRAFT_MOE_MROWS").map(|v| v != "0").unwrap_or(false))
}

/// `DSV41_DRAFT_GRAPH=1` (**DEFAULT OFF** — the A/B arm) — the draft's
/// **device-level** CUDA graph: the whole kernel sequence from the `main_x`
/// projection through the sampled block is RECORDED once and rePLAYed as ONE
/// `cudaGraphLaunch` instead of ~150 launches per draft step.
///
/// # Why this is the draft's biggest single lever
///
/// `docs/agent/draft-1ms-design.md` §P1: at 4.29 ms and ~150 nodes the draft
/// pays the per-node launch+gap cost on every one of them, and the measured
/// per-node deltas (2.904 us plain vs 0.411 us inside a graph) put the ceiling
/// of the mechanical folds (P3a/P3b) well above 1 ms. The graph is not a
/// numeric change at all — the SAME launches, in the SAME order, on the same
/// stream — so it is the one lever that cannot move a bit.
///
/// # What has to be moved OUT of the recorded region
///
/// A capture RECORDS instead of executing, so every host-side effect inside it
/// would be frozen at capture time. The draft had four such effects; all four
/// are now handled in [`DsparkDev::draft_forward`]'s PROLOGUE, before
/// `capture_begin`/`graph_launch`, and every one of them writes the SAME device
/// address the recording saw (the `chain_dev.rs` verify-graph discipline):
///
/// * **D1** — the per-step blocking H2D of `ids` (`upload_i32`) became a
///   prologue upload.
/// * **D2** — `seed_window`'s ring destination was the HOST-computed
///   `window[s] + (pos % win)*hd`. A captured `cudaMemcpyAsync` bakes that
///   address, so every replay would have appended to the SAME ring slot — the
///   classic "captured but not updated" bug (`dsv41_glue.cu`'s
///   `ring_append_kernel` header records exactly this failure). It now runs
///   `dsv41_ring_append`, which derives the slot from a DEVICE counter.
/// * **D3** — the `window -> all_kv` copy's size and its `s0 == 0` branch were
///   host-computed from `win_rows(pos)`. Both settle the moment `pos >= win`
///   (`win_rows` then returns `(win, 0)` for good), which is why the gate below
///   refuses earlier positions — see [`draft_graph_arm`].
/// * **D4** — `ensure_idxs`'s H2D (the index table's `n_win` re-upload) moved
///   to the prologue, where `pos >= win` makes it a no-op anyway.
///
/// The RoPE positions needed no change: [`DsparkDev::ensure_pos_dev`] already
/// puts the base in device memory that the kernels dereference, and every
/// `rope_at` `off` is a CONSTANT relative to it (seed `-1`, query `+r`,
/// inverse `+r`, KV `0`), so a prologue upload of `pos_base` is enough to move
/// every replay's positions.
///
/// # What stays outside on purpose
///
/// * `drafts()` — the D2H of `ids[1..=bs]` is a device READ, illegal inside a
///   capture; it is a separate method the orchestration calls after the step
///   (the `step_rows` argmax-D2H pattern).
/// * the two `host_barrier`s around the capture and the one before each replay
///   (a host barrier is not a CUDA call and would not be recorded).
/// * `DSV41_DSPARK_UNIT_DUMP` / the golden injection: both probe or upload from
///   the host inside the recorded region, so the arm condition simply refuses
///   them ([`draft_graph_arm`]).
///
/// # Fallback
///
/// A capture is an OPTIMISATION: `capture_begin`/`capture_end`/
/// `graph_instantiate` refusing (a driver-rejected op, a stale runtime) latches
/// `draft_graph_failed`, which makes EVERY later draft step take the direct
/// launches — the `verify_graph_failed` pattern. Read once and cached (the
/// house rule for hot-path gates).
fn draft_graph_want() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_DRAFT_GRAPH").map(|v| v != "0").unwrap_or(false))
}

/// The four `DSV41_DRAFT_P3A` folds, resolved once (see [`draft_p3a`]).
#[derive(Clone, Copy)]
struct DraftP3a {
    /// a1: attn `hc_collapse` + `rmsnorm(attn_norm)` → `dsv41_hc_collapse_norm`.
    collapse_norm: bool,
    /// a2: the two `hc_post` destination swaps that retire both
    /// `memcpy_d2d(h <- h_out)`.
    hcpost_swap: bool,
    /// a3: premix ping-pong (retires `memcpy_d2d(pre_in <- pre_ffn)` and the
    /// pre-loop `premix_init` copy).
    premix_pp: bool,
    /// a4: `apply_rope_mrows` for the `bs` query / inverse-query rows.
    rope_mrows: bool,
}

/// `DSV41_DRAFT_P3A=1` (**DEFAULT OFF**) — the draft chain's **P3a** folds from
/// `docs/agent/draft-p3-fusion.md` §5: the zero-risk adjacent-kernel merges that
/// reuse EXISTING kernels only (no new `.cu`, nothing moves across a translation
/// unit, no instruction sequence changes).
///
/// | item | override | fold | launches saved |
/// |---|---|---|---|
/// | a1 | `DSV41_P3A_COLLAPSE_NORM` | `hc_collapse` + `rmsnorm(attn_norm)` → `dsv41_hc_collapse_norm` (the kernel the FFN half already runs) | 1/block |
/// | a2 | `DSV41_P3A_HCPOST_SWAP` | the two `hc_post`s write into each other's buffer instead of `h_out` + `memcpy_d2d` back (see `draft_forward`; the `dsv41_hc_post_inplace` variant has NO batch-row dimension and cannot carry `bs` rows) | 2/block |
/// | a3 | `DSV41_P3A_PREMIX_PP` | premix ping-pong: `pre_in`/`pre_ffn` alternate and the incoming premix is read from the previous block's own slot | 1/block + 1/step |
/// | a4 | `DSV41_P3A_ROPE_MROWS` | `bs` query rows (and `bs` inverse-query rows) → ONE `dsv41_apply_rope_mrows` | 8/block |
///
/// A per-item override, when SET, wins over the master (`=0` turns that one fold
/// off for an A/B that isolates it; any other value turns it on). An unset
/// override follows the master, so `DSV41_DRAFT_P3A=1` is the whole arm and the
/// per-item names exist to take ONE fold back.
///
/// **WHY IT SHIPS OFF.** Every fold here is meant to be bit-identical by
/// construction (same kernel, same instruction sequence) and the a1/a4 items
/// reuse kernels whose bit-identity argument is already written down in their
/// headers (`dsv41_hc_collapse_norm_kernel`:7942-7947,
/// `apply_rope_mrows_kernel`:1967-1973). That argument still has to be confirmed
/// on the real draft shape by the parity tests plus a same-binary serve A/B — the
/// house rule for a new path. Two of the six §5 items are NOT implementable with
/// the existing kernels at the production `bs = 5` and are therefore absent
/// (a5 `sparse_attn_orope`: its o-rope epilogue takes the position
/// `base*mul + off + head*step` and cannot give row `r` the position `pos + r`;
/// a6 `WOB_F32`: `dsv41_gemm_fp8_mx_f32` is an M=1 GEMV with no row batch).
///
/// Read once and cached (the house rule for hot-path gates): this branch runs
/// `bs x n_mtp` times per draft step.
/// TAP BF16 ROUND-TRIP gate (`DSV41_TAP_BF16`, DEFAULT OFF): rounds the dspark
/// draft's `main_h` input (the concatenated target-layer hidden states) back to
/// bf16 precision, matching the official model's dtype for the tensors the MTP
/// head was trained on. Read once and cached.
fn tap_bf16() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("DSV41_TAP_BF16")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

/// DRAFT ATTN BF16 gate (`DSV41_DRAFT_ATTN_BF16`, DEFAULT OFF): rounds the
/// draft block's attention output (after wo_b) back to bf16, matching the
/// official DSparkBlock's dtype for the tensor hc_post consumes. Read once.
fn draft_attn_bf16() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("DSV41_DRAFT_ATTN_BF16")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

/// DRAFT BF16 ACTIVATION DOMAIN gate (`DSV41_DRAFT_BF16_DOMAIN`, DEFAULT OFF):
/// rounds the draft's two f32 activations whose official counterparts are bf16
/// tensors — the head's `normed` input (draft-numerical-audit C#8) and the MoE
/// block's `xn` input (C#9: the gate's `F.linear`, and the act_quant that feeds
/// the routed experts, both read the SAME bf16 ffn input). ONE flag covers the
/// pair on purpose: the two sites are the same defect (an f32 activation handed
/// to weights calibrated for a bf16 one) and a half-on state would just add
/// noise to the A/B. Read once and cached (the house rule — never per-call).
fn draft_bf16_domain() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("DSV41_DRAFT_BF16_DOMAIN")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

fn draft_p3a() -> DraftP3a {
    static F: std::sync::OnceLock<DraftP3a> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let master = std::env::var("DSV41_DRAFT_P3A").map(|v| v != "0").unwrap_or(false);
        let item = |name: &str| match std::env::var(name) {
            Ok(v) => v != "0",
            Err(_) => master,
        };
        DraftP3a {
            collapse_norm: item("DSV41_P3A_COLLAPSE_NORM"),
            hcpost_swap: item("DSV41_P3A_HCPOST_SWAP"),
            premix_pp: item("DSV41_P3A_PREMIX_PP"),
            rope_mrows: item("DSV41_P3A_ROPE_MROWS"),
        }
    })
}

/// The four `DSV41_DRAFT_P3B` folds, resolved once (see [`draft_p3b`]).
#[derive(Clone, Copy)]
struct DraftP3b {
    /// b1: the shared expert's `(w1 | w3) -> swiglu+quant -> w2` chain as ONE
    /// multi-row pass ([`DsparkDev::shared_expert_mrows`]).
    sh_exp_mrows: bool,
    /// b2: the routed experts' three stages as ONE `rows = bs` launch each. This
    /// is the historical `DSV41_DRAFT_MOE_MROWS` arm; the P3B master ORs it in so
    /// one env turns the whole P3b shape on, and the old name keeps working.
    moe_mrows: bool,
    /// b3: the MoE gate's `bs` rows in ONE `ferrite_gemv_bf16_v2_mrows`.
    gate_mrows: bool,
    /// b4: the shared expert's merge folded into the w2 GEMV's epilogue
    /// (`dsv41_gemm_fp8_mx_add`). Only taken on the PER-ROW shared arm (the
    /// `gemm_fp8_mrows` w2 has no additive twin) and therefore inert whenever b1
    /// engages — see [`DsparkDev::shared_expert_epi_add`].
    sh_epi_add: bool,
}

/// `DSV41_DRAFT_P3B=1` (**DEFAULT OFF**) — the draft chain's **P3b** folds from
/// `docs/agent/draft-p3-fusion.md` §5: the multi-row passes and epilogue folds
/// that shrink each MTP block's MoE half to the size the segment kernels will
/// later eat in one bite (that document's "形态 I").
///
/// | item | override | fold | launches saved |
/// |---|---|---|---|
/// | b1 | `DSV41_P3B_SH_EXP_MROWS` | shared expert `w1/w3` + `swiglu_limit_q` + `w2` for all `bs` rows: the per-row loop's `5 x bs` launches become 4 (and the `swiglu_limit` + `quant1` pair becomes the fused `_q` epilogue) | 20/block |
/// | b2 | `DSV41_P3B_MOE_MROWS` | routed experts: one `rows = bs` launch per stage (`DSV41_DRAFT_MOE_MROWS`'s arm, kept under its own name too) | 12/block |
/// | b3 | `DSV41_P3B_GATE_MROWS` | MoE gate: `bs` `gemv_bf16` launches -> ONE `gemv_bf16_v2_mrows` | 4/block |
/// | b4 | `DSV41_P3B_SH_EPI_ADD` | shared expert merge folded into the w2 epilogue (per-row arm only) | 1/block |
///
/// A per-item override, when SET, wins over the master (`=0` turns that one fold
/// off for an A/B that isolates it; any other value turns it on). An unset
/// override follows the master, so `DSV41_DRAFT_P3B=1` is the whole arm and the
/// per-item names exist to take ONE fold back.
///
/// **WHAT IS NOT HERE, and why** (so the next reader does not re-derive it):
/// * **attention projections** (`wq_a`/`wq_b`/`wkv`/`wo_b`) are ALREADY
///   multi-row: each is one `gemm_fp8_mx` at `m = bs`. Note they take that
///   symbol's **16-row tile MMA** program, which is why they are not routed
///   through `gemm_fp8_mrows` here — the pair is not bit-identical and the
///   swap would be a numerical change, not a launch saving (the plan's P3b table
///   does not ask for it).
/// * **the hc chain** (`hc_mixes` / `hc_collapse[_norm]` / `hc_post`) is ALREADY
///   multi-row: every draft call passes `rows = bs` (the kernel's native row
///   dimension, the same one `chain_dev`'s verify side drives with `rows = m`).
/// * **rope** and the **head** are P3a a4 (`apply_rope_mrows`) and
///   `head_gemv_bf16_mrows` respectively — both already in place.
/// * **b4 does not pair with b1**: `dsv41_gemm_fp8_mrows` has no additive
///   epilogue, and folding the add into the shared `w2` would leave the
///   `shared_out` unit dump unwritten, breaking the golden comparison chain
///   (`dspark-correctness-chain.md`). It stays available for the per-row arm.
///
/// **WHY IT SHIPS OFF.** Every fold here is a bit-identity CLAIM taken from the
/// kernel headers the verify side already relies on, and each one still has to
/// be confirmed on the real draft shape (bs = 5) by the `dspark` parity tests
/// plus a same-binary serve A/B — the house rule for a new path. Read once and
/// cached: `draft_moe` runs `n_mtp` times per step and every field is read inside
/// its per-block body.
fn draft_p3b() -> DraftP3b {
    static F: std::sync::OnceLock<DraftP3b> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let master = std::env::var("DSV41_DRAFT_P3B").map(|v| v != "0").unwrap_or(false);
        let item = |name: &str| match std::env::var(name) {
            Ok(v) => v != "0",
            Err(_) => master,
        };
        DraftP3b {
            sh_exp_mrows: item("DSV41_P3B_SH_EXP_MROWS"),
            moe_mrows: draft_moe_mrows() || item("DSV41_P3B_MOE_MROWS"),
            gate_mrows: item("DSV41_P3B_GATE_MROWS"),
            sh_epi_add: item("DSV41_P3B_SH_EPI_ADD"),
        }
    })
}

/// `DSV41_MARKOV_SLICED=1` (DEFAULT OFF — the house rule: a new path ships as an
/// A/B arm) cuts the draft's Markov head across the ranks the same way
/// `DSV41_VERIFY_HEAD_SLICED` cuts the verify's head: rank `r` walks only its
/// `[r*seg, (r+1)*seg)` rows of the REPLICATED `markov_head` and the ranks fold
/// their packed keys once per step.
///
/// **Why it is the one physical enabler of the 5-step reuse.**
/// `dspark_markov_head_kernel` streams the whole `[vocab, mr]` f32 head (126 MiB)
/// once per step, 5 steps per draft block, and nothing carries over: the input
/// embedding row changes with the sampled token. 126 MiB cannot be held
/// anywhere on-chip (148 SM x 227 KB = 33.6 MB of registers + smem, 60 MB L2),
/// so all five scans are compulsory HBM passes. One rank's `vocab/world` = 16160
/// rows = 15.8 MiB DOES fit L2, which is what turns four of the five scans into
/// L2 hits (docs/agent/draft-1ms-design.md 1.2) — the slice is not a byte-count
/// trick, it is the only way the traffic is physically avoidable.
///
/// **Epoch footprint.** The Markov loop is strictly sequential, so the five steps
/// cannot be batched into one multi-row round; each step pays one
/// `dsv41_argmax_key_pub` round. That is 3 blocks x 5 = 15 extra v5 rounds per
/// `draft_forward`, and it is deadlock-free for a structural reason: the draft
/// runs on EVERY rank with the SAME launch sequence (replicated draft weights and
/// attention tensors, an MoE all-reduce per block), so every rank issues the same
/// 15 rounds plus the same 3 AR rounds. The footprint is SYMMETRIC — no rank ever
/// issues a round its peers do not (the `SEED_ALIGN` asymmetry failure). This is
/// the same property the verify's per-row argmax exchange relies on; it holds
/// here for the same reason. A sliced Markov that ran on a SUBSET of the ranks
/// would break exactly this invariant, so the gate must stay an all-ranks gate.
///
/// Fallbacks (each one keeps the full-vocabulary loop): `world == 1`,
/// `vocab % world != 0`, a non-BF16 head (the sliced head GEMV is bf16-only),
/// a stale `.so` missing either new symbol, a collective that is not on the v5
/// protocol, and a v5 slot too small for one key. [`Self::markov_head_geom`] is
/// the ONE place that decides. Read once and cached (the house rule for
/// hot-path gates).
fn markov_sliced() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_MARKOV_SLICED").map(|v| v != "0").unwrap_or(false))
}

/// The device draft. Holds the window rings (one per MTP block), every scratch
/// buffer the forward needs, and references to the shared weights/config.
pub struct DsparkDev<'a> {
    dev: &'a Device,
    cfg: &'a Dsv41Config,
    w: &'a Dsv41DevWeights,

    // ---- tensor parallelism ----
    /// TP degree / this rank's index (1 / 0 without a collective). The draft's
    /// ATTENTION tensors are replicated (every rank maps the GLOBAL head/group/
    /// vocab geometry), but its MoE is TP-split exactly like the backbone's, so
    /// the expert and shared-expert slices below are LOCAL and need an
    /// all-reduce to become the full block.
    world: usize,
    rank: usize,
    /// The collective the MoE's all-reduce runs on; `None` on a single device
    /// (and the reason a `world > 1` instance refuses to run without one).
    comm: Option<Arc<Collective>>,

    // ---- geometry ----
    dim: usize,
    hd: usize,
    nh: usize,
    ql: usize,
    /// per-group low-rank width (`o_lora_rank`)
    olg: usize,
    groups: usize,
    hpg: usize,
    /// `groups * o_lora_rank`
    ol_total: usize,
    win: usize,
    bs: usize,
    hc: usize,
    vocab: usize,
    mr: usize,
    n_target: usize,
    /// `inter / world` padded to the expert MMA K atom (the routed expert
    /// kernels are sized by it)
    inter_local: usize,
    /// The SHARED expert's local width: `inter / world` under DSV41_SHARED_TP
    /// (w1/w3 are `Shard::Rows`), the whole `inter` under the replicated
    /// rank-0-only layout. NOT padded — the loader slices the rows exactly, and
    /// the shared GEMM's `n` is a row count, not an MMA K.
    sh_il: usize,

    // ---- draft window rings, one per MTP block (`w.mtp[s]` has its own wkv) ----
    window: Vec<DevBuf>,

    // ---- forward_embed ----
    /// the recorded `h.mean(dim=hc)` of `dspark_target_layer_ids`, concatenated
    main_h: DevBuf,
    /// `main_norm(main_proj(main_h))`, `[1, dim]`
    main_x: DevBuf,
    /// the hc-expanded draft input, `[bs * hc, dim]`
    h: DevBuf,
    /// `hc_post`'s staging (it is not in-place for rows > 1)
    h_out: DevBuf,

    // ---- per-block scratch ----
    xn: DevBuf,
    qr: DevBuf,
    q: DevBuf,
    kv: DevBuf,
    /// the main stream's KV row for this step, `[hd]`
    mk: DevBuf,
    /// the window ring followed by the draft block, `[win + bs, hd]`
    all_kv: DevBuf,
    o: DevBuf,
    wo: DevBuf,
    xq: DevBuf,
    xsc: DevBuf,
    xq4: DevBuf,
    xsc4: DevBuf,
    pre_in: DevBuf,
    pre_attn: DevBuf,
    pre_ffn: DevBuf,
    post: DevBuf,
    comb: DevBuf,

    // ---- attention selection ----
    idxs: DevBuf,
    /// `[bs + 1]`: `ids[0]` is the backbone's token, the rest are the samples
    ids: DevBuf,
    /// the compressor-length stand-in `sparse_attn` reads: constant `bs`
    clen: DevBuf,
    /// The draft's RoPE base AND its per-row positions, in ONE buffer: slot 0 is
    /// the `int` every rope kernel reads as `*base`, slots `1..=bs` are
    /// `pos + r` — the `pos_rows` array the multi-row form
    /// ([`Self::rope_queries`] / [`Self::rope_queries_inv`], `DSV41_DRAFT_P3A`'s
    /// a4) reads, and slot `bs + 1` is the **window ring's seed position**
    /// (`pos - 1`, see [`Self::seed_slot_ptr`]). All three are a pure function of
    /// the ANCHOR position, so the draft step still pays exactly ONE upload (see
    /// [`Self::ensure_pos_dev`]).
    pos_base: DevBuf,
    /// Host shadow of what `pos_base` currently holds on the device, i.e. the
    /// base every `rope_at` call's `off` is measured from. `None` = unknown
    /// (nothing uploaded yet), which forces the next upload.
    pos_dev: Option<i32>,

    // ---- head ----
    collapse: DevBuf,
    normed: DevBuf,
    logits: DevBuf,
    confidence: DevBuf,
    mk_partial: DevBuf,
    mk_ctr: DevBuf,
    /// `[1]` u64: the sliced Markov step's packed slice winner (key | ~global
    /// index), the input to the step's cross-rank fold (`argmax_key_pub`). Only
    /// touched by `DSV41_MARKOV_SLICED`; dormant on the full-vocabulary path.
    mk_key: DevBuf,

    // ---- MoE ----
    scores: DevBuf,
    route_w: DevBuf,
    route_idx: DevBuf,
    ex_act: DevBuf,
    ex_act_b: DevBuf,
    moe_out: DevBuf,
    shared_out: DevBuf,

    /// `[hc]` of `1 / hc`, the mean the target-hidden recording collapses with
    pre_mean: DevBuf,
    /// `[bs * hc]` one-hot on copy 0, the first block's incoming premix
    premix_init: DevBuf,

    /// the main chain's RoPE tables (see [`DsparkDev::set_rope_tables`])
    cos: Option<*const f32>,
    sin: Option<*const f32>,

    /// `min(win, pos + 1)` the `idxs` buffer currently holds
    n_win_cached: usize,
    /// layer -> its slot in `main_h` (`None` for a layer that is not a target)
    target_slot: Vec<Option<usize>>,

    /// Golden per-unit capture (`DSV41_DSPARK_UNIT_DUMP=1`, see
    /// [`crate::dsv41::unit_dump`]). `Some` for exactly ONE forward — the first
    /// `draft_forward` with `pos > 0` on rank 0 — and `None` on every other
    /// call, so the per-dump check is one `Option` branch when the gate is off.
    unit: Option<UnitDump>,

    // ---- device-level CUDA graph (`DSV41_DRAFT_GRAPH`, see `draft_graph_want`) ----
    /// The instantiated draft graph (`cudaGraphExec_t`), once a capture has
    /// committed. `None` until then and forever after a capture failure.
    graph: Option<*mut c_void>,
    /// The DRY run has happened: every lazy first-use cost (kernel module load,
    /// stream-ordered resource setup) is behind us, so the capture that follows
    /// records a WARM sequence. Mirrors `DevChain::verify_dry_done`.
    graph_dry_done: bool,
    /// A capture failed: latch, so a capture is never retried and every later
    /// draft step takes the direct launches. Mirrors `verify_graph_failed`.
    graph_failed: bool,
    /// Diagnostics: how many captures / replays this instance has done.
    graph_captures: u64,
    graph_replays: u64,
}

fn need<'t>(t: &'t Option<DevTensor>, what: &str) -> Result<&'t DevTensor> {
    t.as_ref()
        .ok_or_else(|| FerriteError::Config(format!("dspark: {what} is not loaded")))
}

impl<'a> DsparkDev<'a> {
    /// `world`/`rank` are the tensor-parallel degree and this rank's index. The
    /// draft's attention runs on the GLOBAL geometry on every rank (its attention
    /// tensors are `Shard::Replicated`), but its MoE is TP-split like the
    /// backbone's, so `world` decides the LOCAL expert slices here — under TP8
    /// the checkpoint's `inter` (2304) is the global number and each rank holds
    /// `inter / world` (288, padded to 320 by the loader).
    pub fn new(
        dev: &'a Device,
        w: &'a Dsv41DevWeights,
        cfg: &'a Dsv41Config,
        world: usize,
        rank: usize,
    ) -> Result<Self> {
        let world = world.max(1);
        let dim = cfg.dim;
        let hd = cfg.head_dim;
        let nh = cfg.n_heads;
        let ql = cfg.q_lora_rank;
        let olg = cfg.o_lora_rank;
        let groups = cfg.o_groups;
        let hpg = nh / groups;
        let ol_total = groups * olg;
        let win = cfg.window_size;
        let bs = cfg.dspark_block_size;
        let hc = cfg.hc_mult;
        let vocab = cfg.vocab_size;
        let mr = cfg.dspark_markov_rank;
        let n_target = cfg.dspark_target_layer_ids.len();
        let inter = cfg.moe_inter_dim;
        // The routed experts are TP-split along `inter` (the loader cuts w1/w3's
        // rows and w2's columns by world, see weights.rs::load_expert_pool), so
        // every expert kernel below is sized by the PADDED LOCAL width.
        let inter_local = crate::dsv41::weights::padded_inter(inter / world);
        // The shared expert's w1/w3 are `Shard::Rows` only under
        // DSV41_SHARED_TP; otherwise they are replicated and rank 0 alone
        // computes them (weights.rs::shared_expert_tp).
        let stp = crate::dsv41::weights::shared_expert_tp();
        let sh_il = if stp { inter / world } else { inter };
        if bs == 0 || n_target == 0 || cfg.n_mtp_layers == 0 {
            return Err(FerriteError::Config(
                "dspark: the config has no draft (dspark_block_size / \
                 dspark_target_layer_ids / n_mtp_layers is empty)"
                    .into(),
            ));
        }
        let fb = |n: usize| n * 4;

        // The largest fp8 activation quantised anywhere below.
        let max_act = (n_target * dim)
            .max(bs * nh * hd)
            .max(bs * ol_total)
            .max(bs * dim)
            .max(bs * inter);

        // Defect 4 (audit-moe-seg): the MoE scratches below (`scores`, `route_w`,
        // `route_idx`, `ex_act_b`) are indexed by the DRAFT layers' routing
        // geometry — `cfg.moe_config(n_layers + s)`, i.e. the dspark 128/3 — not
        // the backbone's 384/6. Sizing them from `cfg.n_routed_experts` /
        // `cfg.n_activated_experts` happened to be the LARGER pair so nothing
        // overflowed in the shipped config, but a checkpoint whose draft routes
        // wider than its backbone would walk past the end. All `mtp` blocks share
        // one geometry; take the max over them anyway so a future per-block
        // difference cannot silently under-allocate.
        let (mo_n_routed, mo_topk) = (0..cfg.n_mtp_layers)
            .map(|s| cfg.moe_config(cfg.n_layers + s))
            .fold((1usize, 1usize), |(nr, tk), (n, k)| (nr.max(n), tk.max(k)));

        let mut window = Vec::with_capacity(cfg.n_mtp_layers);
        for _ in 0..cfg.n_mtp_layers {
            let b = dev.alloc(fb(win * hd))?;
            dev.zero(&b)?;
            window.push(b);
        }

        let main_h = dev.alloc(fb(n_target * dim))?;
        dev.zero(&main_h)?;

        let s = DsparkDev {
            dev,
            cfg,
            w,
            world,
            rank,
            comm: None,
            dim,
            hd,
            nh,
            ql,
            olg,
            groups,
            hpg,
            ol_total,
            win,
            bs,
            hc,
            vocab,
            mr,
            n_target,
            inter_local,
            sh_il,
            window,
            main_h,
            main_x: dev.alloc(fb(dim))?,
            h: dev.alloc(fb(bs * hc * dim))?,
            h_out: dev.alloc(fb(bs * hc * dim))?,
            xn: dev.alloc(fb(bs * dim))?,
            qr: dev.alloc(fb(bs * ql))?,
            q: dev.alloc(fb(bs * nh * hd))?,
            kv: dev.alloc(fb(bs * hd))?,
            mk: dev.alloc(fb(hd))?,
            all_kv: dev.alloc(fb((win + bs) * hd))?,
            o: dev.alloc(fb(bs * nh * hd))?,
            wo: dev.alloc(fb(bs * ol_total))?,
            xq: dev.alloc(max_act.max(8))?,
            xsc: dev.alloc(fb(max_act / 32 + 8))?,
            // `bs` activation rows: `dim/2` packed fp4 bytes or `dim` e4m3 bytes
            // per row — `bs*dim` covers both quantiser layouts.
            xq4: dev.alloc((bs * dim).max(8))?,
            xsc4: dev.alloc(fb(bs * dim / 32 + 8))?,
            pre_in: dev.alloc(fb(bs * hc))?,
            pre_attn: dev.alloc(fb(bs * hc))?,
            pre_ffn: dev.alloc(fb(bs * hc))?,
            post: dev.alloc(fb(bs * hc))?,
            comb: dev.alloc(fb(bs * hc * hc))?,
            idxs: dev.alloc(fb(bs * (win + bs)).max(4))?,
            ids: dev.alloc(fb(bs + 1).max(4))?,
            clen: dev.alloc(4)?,
            // base slot + the `bs` per-row positions + the ring seed position
            // (see `pos_base`'s doc)
            pos_base: dev.alloc(4 * (bs + 2))?,
            pos_dev: None,
            collapse: dev.alloc(fb(bs * dim))?,
            normed: dev.alloc(fb(bs * dim))?,
            logits: dev.alloc(fb(bs * vocab))?,
            confidence: dev.alloc(fb(bs).max(4))?,
            mk_partial: dev.alloc(MARKOV_MAX_BLOCKS * 8)?,
            mk_ctr: dev.alloc(4)?,
            mk_key: dev.alloc(8)?,
            // Sized by the DRAFT layers' routing geometry — see `mo_n_routed` /
            // `mo_topk` above (defect 4, audit-moe-seg). `draft_moe` reads them
            // with `cfg.moe_config(layer)` for `layer >= n_layers`.
            scores: dev.alloc(fb(bs * mo_n_routed))?,
            route_w: dev.alloc(fb(bs * mo_topk).max(4))?,
            route_idx: dev.alloc(fb(bs * mo_topk).max(4))?,
            // Shared by the routed SEQUENTIAL fallback ([bs][2*inter_local]) and
            // the shared expert ([2*sh_il]); under the replicated shared layout
            // (DSV41_SHARED_TP=0) `sh_il == inter` is the LARGER of the two, so
            // the size has to clear both.
            ex_act: dev.alloc(fb(bs * 2 * inter_local.max(sh_il)))?,
            ex_act_b: dev.alloc(fb(mo_topk * bs * 2 * inter_local))?,
            moe_out: dev.alloc(fb(bs * dim))?,
            shared_out: dev.alloc(fb(bs * dim))?,
            pre_mean: dev.alloc(fb(hc))?,
            premix_init: dev.alloc(fb(bs * hc))?,
            cos: None,
            sin: None,
            n_win_cached: usize::MAX,
            target_slot: (0..cfg.n_layers + cfg.n_mtp_layers)
                .map(|l| cfg.dspark_target_layer_ids.iter().position(|&t| t == l))
                .collect(),
            unit: None,
            graph: None,
            graph_dry_done: false,
            graph_failed: false,
            graph_captures: 0,
            graph_replays: 0,
        };

        // `clen` is the compressor-length stand-in `sparse_attn` reads; for the
        // draft the "compressed" block is the draft tokens themselves. The kernel
        // reads it as an `int`, so the four bytes must be the INTEGER's, not the
        // f32 the upload helper's name suggests.
        s.dev
            .upload_bytes_at(&s.clen, &(bs as i32).to_le_bytes())?;
        // The target-hidden mean and the first block's incoming premix are
        // constants of the geometry, so they are uploaded once here.
        let mean = vec![1.0f32 / hc as f32; hc];
        s.dev.upload_f32_at(s.pre_mean.ptr, 0, &mean)?;
        let mut pm = vec![0f32; bs * hc];
        for r in 0..bs {
            pm[r * hc] = 1.0;
        }
        s.dev.upload_f32_at(s.premix_init.ptr, 0, &pm)?;
        // The markov election counter must start at zero (cudaMalloc does not
        // zero) and self-resets inside the kernel from then on.
        s.dev.zero(&s.mk_ctr)?;

        if trace() {
            eprintln!(
                "[dspark] device draft: bs={bs} win={win} hc={hc} dim={dim} hd={hd} nh={nh} \
                 ql={ql} groups={groups} olg={olg} vocab={vocab} mr={mr} targets={n_target} \
                 world={world} rank={rank} inter={inter}/local={inter_local}/shared={sh_il} \
                 mtp={}",
                cfg.n_mtp_layers
            );
        }
        Ok(s)
    }

    /// Hand the draft the rank collective its MoE all-reduce runs on. Must be
    /// called before the first [`Self::draft_forward`] whenever `world > 1`;
    /// without it a multi-rank draft refuses to run rather than produce each
    /// rank's partial sum as if it were the whole block.
    pub fn set_comm(&mut self, c: Arc<Collective>) {
        self.comm = Some(c);
    }

    /// The main chain's RoPE tables (`cos`/`sin`, `[max_pos][rope_head_dim/2]`),
    /// built by `Device::rope_precompute` with the model's main theta. They are
    /// INJECTED rather than built here: a private copy would be another ~128 MiB
    /// on a 4 GiB host, and the draft's positions are a subset of the chain's.
    /// Must be called before the first [`Self::draft_forward`].
    pub fn set_rope_tables(&mut self, cos: *const f32, sin: *const f32) {
        self.cos = Some(cos);
        self.sin = Some(sin);
    }

    /// Copy the chain's target-hidden tap (`[3, dim]`, slot order =
    /// `dspark_target_layer_ids`) into this draft's `main_h` — one D2D, no host
    /// round trip.
    ///
    /// The device twin of [`Self::note_target_hidden`]: the main chain's
    /// `layer()` hook already collapses each target layer's attention input with
    /// the same `1/hc` mean (`DevChain`'s `dspark_tap`), so the orchestration
    /// hands the whole `[n_target, dim]` block over in one copy instead of
    /// calling back into the layer loop. `src` must span at least
    /// `n_target * dim` f32 in the SAME slot order the chain wrote them in
    /// (`Dsv41Config::dspark_target_slot`).
    pub fn import_tap(&mut self, src: *const f32) -> Result<()> {
        // The chain's tap is sized `DSPARK_TAP_SLOTS x dim`; a config with more
        // targets than that would have overflowed it inside `layer()` first.
        debug_assert!(
            self.n_target <= DSPARK_TAP_SLOTS,
            "dspark: the chain's tap holds {DSPARK_TAP_SLOTS} slots, but the config lists {} \
             target layers",
            self.n_target
        );
        self.dev.memcpy_d2d(
            self.main_h.ptr,
            src as *const c_void,
            self.n_target * self.dim * std::mem::size_of::<f32>(),
        )?;
        // TAP BF16 ROUND-TRIP (DSV41_TAP_BF16, default OFF): the official
        // model's hidden states carry the model dtype (bf16) — the MTP head was
        // trained on bf16 taps. ferrite's chain keeps f32, so the draft's input
        // is systematically ~1e-3 off what the head expects, which suppresses
        // the first-token acceptance (the 64% k_acc=0 mode). The round-trip is
        // exact (RN narrowing, lossless widening) and OFF keeps every bit.
        if tap_bf16() {
            let n = (self.n_target * self.dim) as i64;
            self.dev.bf16_roundtrip(self.main_h.ptr as *mut f32, n)?;
        }
        Ok(())
    }

    /// Append a COMMITTED verify block's target hiddens to the window rings:
    /// rows `0..keep` of the block, written at the positions `pos_base + j`.
    ///
    /// This is the official `_update`'s `target_hidden_states[:, :accepted + 1]`
    /// append, and it is what keeps the draft's context WHOLE across a
    /// multi-token commit. The per-step path ([`Self::draft_forward`]) seeds
    /// exactly ONE row — the position it consumes — so before this existed every
    /// position a `k > 1` commit SKIPPED stayed a hole in the ring, and the
    /// draft's attention read across it (accept degrading with the number of
    /// multi-token steps).
    ///
    /// # Layout and row range (must stay in step with `dspark_commit`'s `keep`)
    ///
    /// `tap_r` is the chain's per-row tap, `[DSPARK_TAP_SLOTS][m][dim]` with slot
    /// `slot` (`Dsv41Config::dspark_target_slot`, 0/1/2 = the target layers in
    /// config order) and row `r` at `(slot * m + r) * dim`. Verify row `r` was
    /// forwarded at position `pos + 1 + r`, so row `j`'s hidden is the one for
    /// position `pos_base + j` with `pos_base = pos + 1`.
    ///
    /// `keep` is the accepted prefix length `k_acc` — the SAME value
    /// `dspark_commit` keeps in the ring, so the two cannot disagree: rows
    /// `0..keep` sit at `pos+1 ..= pos+keep`, which is exactly the ring prefix
    /// the commit preserved. Row `keep` (position `pos+keep+1` = the new
    /// `pos_ctr`) is deliberately NOT written: the verify fed it a REJECTED
    /// draft token, so its hidden is not the true one — that position's KV comes
    /// from the next step's `step_dev` and is seeded by the next
    /// [`Self::draft_forward`], exactly as for a plain decode step. Rows past
    /// `keep` are rejected and dropped.
    ///
    /// The rows are projected ONE at a time through the same
    /// [`Self::project_main_x`] + [`Self::seed_window`] pair the per-step seed
    /// uses, so a committed row and a per-step row are the same numbers.
    pub fn note_ctx_rows(
        &mut self,
        tap_r: *const f32,
        m: usize,
        keep: usize,
        pos_base: usize,
    ) -> Result<()> {
        if keep > m {
            return Err(FerriteError::Config(format!(
                "dspark: note_ctx_rows asked for {keep} ctx rows out of a {m}-row verify block"
            )));
        }
        if keep == 0 {
            return Ok(());
        }
        let dim = self.dim;
        let n_target = self.n_target;
        let row_bytes = dim * std::mem::size_of::<f32>();
        // One RoPE base for the whole batch: row `j`'s position is `pos_base + j`
        // and `rope_at` carries that difference as the kernel's `off`, so the
        // `keep` blocking H2Ds this loop used to issue (one per `seed_window`)
        // become one.
        self.ensure_pos_dev(pos_base as i32)?;
        for j in 0..keep {
            // Row j's `forward_embed` input is the CONCATENATION of the target
            // layers' hidden[j]; the tap interleaves the layers (`m` rows apart),
            // so the `n_target * dim` block is rebuilt one `dim`-wide slice at a
            // time into the same `main_h` the per-step path fills.
            for slot in 0..n_target {
                let src = (tap_r as *const u8).wrapping_add((slot * crate::dsv41::chain_dev::VERIFY_ROWS + j) * row_bytes);
                let dst = (self.main_h.ptr as *mut u8).wrapping_add(slot * row_bytes);
                self.dev
                    .memcpy_d2d(dst as *mut c_void, src as *const c_void, row_bytes)?;
            }
            self.project_main_x()?;
            for s in 0..self.cfg.n_mtp_layers {
                // `slot_dev = false`: this is the COMMIT path, not the per-step
                // draft, so there is no device ring counter carrying row `j`'s
                // position (the buffer's slot holds `pos_base - 1`). The host
                // address is what the block loop always used here.
                self.seed_window(s, pos_base + j, false)?;
            }
        }
        Ok(())
    }

    /// D2H of the drafted block: `draft_forward` samples into `ids[1..=bs]`
    /// (`ids[0]` is the backbone's token), and one speculative step consumes the
    /// first [`DSPARK_DRAFTS`] of those samples.
    ///
    /// One 20-byte device read, outside any capture. `download_u8` keeps it a
    /// plain byte copy, so no f32 reinterpretation is involved — the discipline
    /// `step_rows` already uses for its per-row argmax.
    pub fn drafts(&self) -> Result<[u32; DSPARK_DRAFTS]> {
        if self.bs < DSPARK_DRAFTS {
            return Err(FerriteError::Config(format!(
                "dspark: the draft block samples {} tokens; the speculative step verifies \
                 {DSPARK_DRAFTS} of them",
                self.bs
            )));
        }
        // `ids` is `[i32; bs + 1]`: skip the backbone's token at index 0.
        let b = Device::view(
            (self.ids.ptr as *mut i32).wrapping_add(1) as *mut c_void,
            DSPARK_DRAFTS * 4,
        );
        let mut bytes = [0u8; DSPARK_DRAFTS * 4];
        self.dev.download_u8(&b, &mut bytes)?;
        Ok(std::array::from_fn(|i| {
            u32::from_le_bytes([
                bytes[4 * i],
                bytes[4 * i + 1],
                bytes[4 * i + 2],
                bytes[4 * i + 3],
            ])
        }))
    }

    /// Record one target layer's attention input, `h.mean(dim=hc)` (dspark.rs's
    /// `forward_embed` reads the ATTENTION INPUT of layers 37/38/39, not the
    /// output). `h` is the layer's `[hc, dim]` residual stream on the device.
    ///
    /// A no-op for a layer that is not a `dspark_target_layer_ids` entry, so the
    /// caller can invoke it unconditionally from the layer loop.
    pub fn note_target_hidden(&mut self, layer: usize, h: *const f32) -> Result<()> {
        let Some(slot) = self.target_slot.get(layer).copied().flatten() else {
            return Ok(());
        };
        let dst = (self.main_h.ptr as *mut f32).wrapping_add(slot * self.dim);
        // hc_collapse IS the weighted sum over the hc copies; with pre = 1/hc it
        // is the mean the reference computes elementwise.
        self.dev.hc_collapse(
            h,
            self.pre_mean.ptr as *const f32,
            dst,
            1,
            self.hc as i32,
            self.dim as i32,
        )
    }

    /// `main_x = main_norm(main_proj(main_h))` — the `forward_embed` projection
    /// that turns the recorded target hiddens into the row the window KV is
    /// projected from.
    ///
    /// ONE path for both callers — [`Self::draft_forward`]'s per-step seed and
    /// [`Self::note_ctx_rows`]'s committed rows — so the two kinds of window row
    /// cannot drift apart numerically.
    fn project_main_x(&mut self) -> Result<()> {
        let cfg = self.cfg;
        let dim = self.dim;
        let k_main = self.n_target * self.dim;
        let main_proj = need(&self.w.main_proj, "mtp.0.main_proj.weight")?;
        let main_proj_scale = need(&self.w.main_proj_scale, "mtp.0.main_proj.scale")?;
        let main_norm = need(&self.w.main_norm, "mtp.0.main_norm.weight")?;
        self.quant1(self.main_h.ptr as *const f32, k_main)?;
        self.dev.gemm_fp8_mx(
            self.xq.as_u8(),
            self.xsc.as_f32(),
            main_proj.as_u8(),
            main_proj_scale.as_u8(),
            std::ptr::null(),
            self.main_x.ptr as *mut f32,
            1,
            dim as i32,
            k_main as i32,
        )?;
        self.dev.rmsnorm(
            self.main_x.ptr as *const f32,
            main_norm.as_f32(),
            self.main_x.ptr as *mut f32,
            1,
            dim as i32,
            cfg.norm_eps,
        )?;
        Ok(())
    }

    /// Run the whole draft: `t0` is the token the backbone just sampled, `pos`
    /// is its position. Fills `ids[1..=bs]` (the drafted block) and
    /// `confidence[0..bs]`; both stay on the device.
    pub fn draft_forward(&mut self, t0: u32, pos: usize) -> Result<()> {
        let (Some(_), Some(_)) = (self.cos, self.sin) else {
            return Err(FerriteError::Config(
                "dspark: set_rope_tables() must be called before draft_forward()".into(),
            ));
        };
        let cfg = self.cfg;
        let dim = self.dim;
        let bs = self.bs;
        let hc = self.hc;

        // ---- golden per-unit capture: arm ONCE, on the first decoded forward ----
        // `pos == 0` is the prefill window seed (it returns before the blocks), so
        // it is not a capturable forward; rank != 0 never writes (each TP rank is
        // its own process, all pointed at the same path).
        if self.unit.is_none()
            && self.rank == 0
            && pos > 0
            && unit_dump::enabled()
            && unit_dump::arm_once()
        {
            self.unit = Some(UnitDump::new());
        }

        // ---- golden input injection (`DSV41_DSPARK_UNIT_INJECT`) --------------
        // `import_tap` has already filled `main_h` with the LIVE serve's target
        // hiddens, so an armed injection OVERWRITES it with the reference
        // harness's own input, together with `ids[0]` and `pos`. It is gated on a
        // live capture so only the ONE dumped forward is touched — the rest of the
        // serve is untouched. `main_h` stays a device buffer: the override is one
        // H2D, not a host round trip.
        // The window RING inject: the golden harness's `stage{s}.ring_before`,
        // uploaded per block so the attention diff runs against the EXACT same
        // window the reference saw (the per-step `seed_window` still runs after
        // and overwrites one slot — reproducing the golden's `ring_after`).
        if self.unit.is_some() {
            if let Some(inj) = unit_dump::inject() {
                if let Some(rings) = &inj.rings {
                    for (s, block) in rings.iter().enumerate() {
                        if s >= self.cfg.n_mtp_layers || block.len() != self.win {
                            eprintln!(
                                "[dspark] unit inject rings: block {s} has {} rows, expected {} \
                                 — rings ignored",
                                block.len(),
                                self.win
                            );
                            break;
                        }
                        let flat: Vec<f32> = block.iter().flatten().copied().collect();
                        if flat.len() != self.win * self.hd {
                            eprintln!(
                                "[dspark] unit inject rings: block {s} row width != hd — ignored"
                            );
                            break;
                        }
                        self.dev.upload_f32_at(self.window[s].ptr, 0, &flat)?;
                    }
                }
            }
        }
        let mut pos = pos;
        let mut t0 = t0;
        let mut injected = false;
        if self.unit.is_some() {
            if let Some(inj) = unit_dump::inject() {
                if inj.main_hidden.len() == self.n_target * dim {
                    self.dev.upload_f32_at(self.main_h.ptr, 0, &inj.main_hidden)?;
                    t0 = inj.token as u32;
                    if let Some(p) = inj.pos {
                        pos = p;
                    }
                    injected = true;
                } else {
                    eprintln!(
                        "[dspark] unit inject: main_hidden has {} values, expected {} \
                         (n_target {} x dim {dim}) — ignored",
                        inj.main_hidden.len(),
                        self.n_target * dim,
                        self.n_target
                    );
                }
            }
        }

        // ---- the draft's RoPE base, on the DEVICE ----------------------------
        // One 4B upload per draft step, shared by every rope site below
        // (`seed_window`'s row, the `bs` query rows, the `bs` inverse-query rows
        // and the `bs` KV rows — 36 blocking H2Ds per forward before this). Each
        // site keeps its ABSOLUTE position and `rope_at` passes the delta as the
        // kernel's `off`, so the upload is a pure emission change: the position
        // arithmetic is integer and the result is bit-identical (see
        // [`Self::ensure_pos_dev`] / [`Self::rope_at`]).
        //
        // It goes AFTER the injection override above (which may rewrite `pos`)
        // and BEFORE the `pos == 0` early return below (which seeds the window
        // rings at position 0 and ropes them).
        self.ensure_pos_dev(pos as i32)?;

        // ---- ids: the backbone's token first, the noise token for the rest ----
        // (dspark.rs::forward_embed; the noise row IS embed[noise_token_id])
        let noise = cfg.dspark_noise_token_id as i32;
        let mut ids = vec![noise; bs + 1];
        ids[0] = t0 as i32;
        self.upload_i32(&self.ids, &ids)?;

        // ---- D4: the window index table (an H2D, so OUTSIDE any capture) ----
        // It settles at `n_win == win` the moment `pos >= win` (`win_rows` then
        // returns `(win, 0)` for good, see the graph gate) and is never
        // re-uploaded, so from then on every replay reads the SAME `idxs` bytes
        // the recording saw. Skipped at `pos == 0` exactly as before: that
        // forward returns before the block loop and has no window to index.
        if pos > 0 {
            self.ensure_idxs(pos)?;
        }

        // ---- the draft's kernel sequence: direct, or the graph arm ---------
        // Everything below is ONE kernel sequence with no host effect left in
        // it, which is what lets `DSV41_DRAFT_GRAPH` RECORD it and rePLAY it as
        // a single launch. All four arms drive the SAME `draft_body`: the graph
        // must contain the sequence a normal forward would have issued, never a
        // variant of it.
        if !self.draft_graph_arm(pos) {
            self.draft_body(pos, false)?;
        } else if !self.graph_dry_done {
            // DRY: a REAL execution, so every lazy first-use cost (module load,
            // stream-ordered setup) happens OUTSIDE the recording. Its device
            // effects are the caller's to keep, exactly like any other draft.
            self.draft_body(pos, true)?;
            self.graph_dry_done = true;
        } else if let Some(e) = self.graph {
            // REPLAY. Rendezvous first: a capture only RECORDS the MoE
            // all-reduce while a peer may already be EXECUTING its own — the
            // same host-barrier pair `DevChain::step_impl` keeps around its
            // capture. A no-op without peers (`comm` is None off TP8).
            if let Some(c) = self.comm.as_ref() {
                c.host_barrier();
            }
            self.dev.graph_launch(e)?;
            self.graph_replays += 1;
        } else {
            self.draft_capture(pos)?;
        }

        // ---- serialise the capture (one file per armed forward) ----
        if let Some(u) = self.unit.take() {
            let meta = format!(
                "{{\"pos\":{pos},\"t0\":{t0},\"bs\":{bs},\"hc\":{hc},\"dim\":{dim},\"nh\":{},\
                 \"hd\":{},\"vocab\":{},\"mr\":{},\"n_target\":{},\"n_mtp\":{},\"world\":{},\
                 \"rank\":{},\"pid\":{},\"inject\":{injected}}}",
                self.nh,
                self.hd,
                self.vocab,
                self.mr,
                self.n_target,
                cfg.n_mtp_layers,
                self.world,
                self.rank,
                std::process::id(),
            );
            if let Err(e) = u.write(&meta) {
                eprintln!("[dspark] unit dump write failed: {e}");
            }
        }
        Ok(())
    }

    /// The draft's kernel sequence — the region `DSV41_DRAFT_GRAPH` records.
    ///
    /// Split out of [`Self::draft_forward`] so the recording, the DRY run and
    /// the direct path all drive EXACTLY the same launches: a capture is sound
    /// only if the recorded sequence is the one a normal forward would have
    /// issued, so there is ONE body and not two.
    ///
    /// Nothing that touches the host may live here — no `upload_*`/`download_*`
    /// (those are the caller's prologue), no allocation, no probe. The one
    /// host-side branch is `pos == 0`, which the gate excludes, and the one
    /// behavioural switch is `slot_dev` (see [`Self::seed_window`]).
    fn draft_body(&mut self, pos: usize, slot_dev: bool) -> Result<()> {
        let cfg = self.cfg;
        let dim = self.dim;
        let bs = self.bs;
        let hc = self.hc;

        // ---- forward_embed ----
        // main_x = main_norm(main_proj(concat(target hiddens)))
        self.project_main_x()?;
        self.dump_unit("main_x", self.main_x.ptr as *const f32, &[dim]);

        // h = hc_expand(embed[ids]) — the draft's residual stream starts as the
        // token embedding alone; the main stream enters through the window KV.
        need(&self.w.embed, "embed.weight")?;
        self.dev.embed_expand_dev(
            self.w.embed.as_ref().unwrap().ptr(),
            self.ids.as_i32(),
            self.h.ptr as *mut f32,
            bs as i32,
            dim as i32,
            hc as i32,
            self.vocab as i32,
        )?;
        self.dump_unit("embed", self.h.ptr as *const f32, &[bs, hc, dim]);

        // The window only seeds itself before the first decode step, exactly as
        // the reference does (`dspark_attention` with start_pos == 0).
        if pos == 0 {
            for s in 0..cfg.n_mtp_layers {
                self.seed_window(s, pos, false)?;
            }
            return Ok(());
        }

        // pos_base is uploaded per RoPE call below; make the value that never
        // changes explicit.
        let eps = cfg.norm_eps;

        // ---- three draft blocks ----
        let p3a = draft_p3a();
        // P3a a3 (premix ping-pong): a block's incoming premix IS the previous
        // block's `pre_ffn`, so instead of copying it into `pre_in` every block
        // the two scratch slots alternate and the reader follows the pointer.
        // Block 0 reads `premix_init` directly, which also retires that one
        // copy. Off the gate the copy below runs and every block still reads
        // `pre_in`, i.e. exactly the historical buffers.
        if !p3a.premix_pp {
            self.dev.memcpy_d2d(
                self.pre_in.ptr,
                self.premix_init.ptr,
                (bs * hc * 4) as usize,
            )?;
        }
        let mut premix_cur: *const f32 = if p3a.premix_pp {
            self.premix_init.ptr as *const f32
        } else {
            self.pre_in.ptr as *const f32
        };
        for s in 0..cfg.n_mtp_layers {
            let ld = &self.w.mtp[s];
            // The slot this block's `hc_mixes(ffn)` writes: the one NOT holding
            // the incoming premix. Equal to `pre_ffn` on the un-gated path (and
            // so is the dump below, which therefore keeps its exact meaning: the
            // PREVIOUS block's ffn pre).
            let ffn_out: *mut f32 = if p3a.premix_pp && s % 2 == 0 {
                self.pre_in.ptr as *mut f32
            } else {
                self.pre_ffn.ptr as *mut f32
            };
            // hc_mixes for the attention sub-block: writes THIS block's attn_pre
            // (slot 1), post and comb.
            self.hc_mixes(
                self.h.ptr as *const f32,
                ld,
                false,
                self.pre_attn.ptr as *mut f32,
                self.post.ptr as *mut f32,
                self.comb.ptr as *mut f32,
            )?;
            // the hc_mixes coefficients (pre/post/comb) — the golden harness's
            // `stage{s}.attn.hc_pre/hc_post/hc_comb`, never diffed before; the
            // ffn input's 4.4% residual (which flips top-k expert picks and
            // explodes into the 139% MoE divergence) starts somewhere in this
            // chain.
            self.dump_unit_idx("mixes_attn_pre", s, self.pre_attn.ptr as *const f32, &[bs, hc]);
            self.dump_unit_idx("mixes_attn_post", s, self.post.ptr as *const f32, &[bs, hc]);
            self.dump_unit_idx("mixes_attn_comb", s, self.comb.ptr as *const f32, &[bs, hc, hc]);
            // the FFN-side mixes coefficients — the golden's `stage{s}.ffn.hc_pre/
            // hc_post/hc_comb`. `ffn_out` is this block's `pre_ffn` slot: on the
            // un-gated path it IS `pre_ffn` (byte-identical dump), on the ping-pong
            // arm it follows the alternating slot.
            self.dump_unit_idx("mixes_ffn_pre", s, ffn_out as *const f32, &[bs, hc]);
            self.dump_unit_idx("mixes_ffn_post", s, self.post.ptr as *const f32, &[bs, hc]);
            self.dump_unit_idx("mixes_ffn_comb", s, self.comb.ptr as *const f32, &[bs, hc, hc]);
            // collapse with the INCOMING premix, then the attn norm
            let attn_norm = need(&ld.attn_norm, "mtp.*.attn_norm.weight")?;
        // THE decisive dump: the first 10 weights of this rank's attn_norm.
        // The checkpoint's mtp.0.attn_norm starts [-0.0491, -0.0457, -0.0481,
        // ...] (BF16, read straight from the safetensors). If this dump shows
        // anything else, the weight POINTER is wrong (loading/shard), which
        // would explain rmsnorm exploding while hand-computation with the
        // checkpoint's values stays normal.
        self.dump_unit_idx("attn_norm_w", s, attn_norm.as_f32(), &[16]);
            // The premix this block collapses with: `pre_in` historically, the
            // alternating slot under a3 (see the loop head).
            let collapse_pre: *const f32 = if p3a.premix_pp {
                premix_cur
            } else {
                self.pre_in.ptr as *const f32
            };
            // P3a a1 (`DSV41_DRAFT_P3A`): the FFN half's own fused kernel,
            // `dsv41_hc_collapse_norm`, is exactly this pair — its header
            // (`dsv41_kernels.cu`:7942-7947) pins the collapse's `fmaf` chain and
            // the rmsnorm's `shfl_down` tree plus in-order cross-warp sum
            // statement for statement, which is why the FFN side may use it.
            // Same kernel, same instruction sequence, one launch instead of two.
            //
            // ⚠️ The `h_premix_block` dump exists only on the two-launch arm: the
            // fused kernel keeps the collapsed row in the OUTPUT buffer and
            // normalises it in place in its phase 2, so the un-normed intermediate
            // is never in global memory. A `DSV41_DSPARK_UNIT_DUMP` capture taken
            // with this gate ON therefore has one unit fewer (and `h_norm_block`
            // unchanged) — capture goldens with the gates OFF, as always.
            if p3a.collapse_norm {
                self.dev.hc_collapse_norm(
                    self.h.ptr as *mut f32,
                    collapse_pre,
                    attn_norm.as_f32(),
                    self.xn.ptr as *mut f32,
                    bs as i32,
                    hc as i32,
                    dim as i32,
                    eps,
                    crate::dsv41::chain_dev::bf16_truncate(),
                )?;
            } else {
                self.dev.hc_collapse(
                    self.h.ptr as *const f32,
                    collapse_pre,
                    self.xn.ptr as *mut f32,
                    bs as i32,
                    hc as i32,
                    dim as i32,
                )?;
                // `h = hc_pre(x, pre_mix)`, the reference's `h(pre_mix)` — taken
                // BEFORE the in-place rmsnorm below, which is why it is recorded here.
                self.dump_unit_idx("h_premix_block", s, self.xn.ptr as *const f32, &[bs, dim]);
                // ONE multi-row rmsnorm (n = bs), in place, exactly the call
                // `chain_dev.rs`'s kv norm makes (`attention_rows` hands n = m to
                // the same launcher).
                //
                // HISTORY — this was briefly written as a per-row n=1 loop under
                // the belief that the kernel's n>1 path was broken (an m=bs call
                // appeared to produce 1e27). Reading the kernel settled it the
                // other way (`ferrite_kernels.cu:278-321`, `ferrite_rmsnorm`):
                // `grid(n)` gives one block per row, each block addresses its row
                // at `x + row*dim` / `out + row*dim`, and the cross-warp reduce is
                // sized by `blockDim`. Rows are therefore completely independent —
                // n=bs is bit-identical to bs n=1 launches. The 1e27 came from the
                // attn_norm WEIGHT POINTER being garbage (the `load.rs:925`
                // placeholder bug, since fixed), not from the kernel. The per-row
                // loop was a pure launch-count loss and is reverted here.
                self.dev.rmsnorm(
                    self.xn.ptr as *const f32,
                    attn_norm.as_f32(),
                    self.xn.ptr as *mut f32,
                    bs as i32,
                    dim as i32,
                    eps,
                )?;
            }
            // post-norm xn — the SAME semantic as the golden harness's
            // `stage{s}.attn.in` (attn_norm's output), for direct diffing.
            self.dump_unit_idx("h_norm_block", s, self.xn.ptr as *const f32, &[bs, dim]);

            self.draft_attention(s, pos, slot_dev)?;
            // the attention block's own units: q/kv are the post-RoPE projections,
            // o is the module's output (after wo_b), all still live here.
            self.dump_unit_idx("q_block", s, self.q.ptr as *const f32, &[bs, self.nh, self.hd]);
            self.dump_unit_idx("kv_block", s, self.kv.ptr as *const f32, &[bs, self.hd]);
            self.dump_unit_idx("o_block", s, self.o.ptr as *const f32, &[bs, dim]);

            // residual: h = hc_post(o, post, comb)
            //
            // P3a a2 (`DSV41_DRAFT_P3A`): the two `hc_post` + `memcpy_d2d` pairs
            // per block are retired by letting the two calls write into each
            // other's buffer instead of staging in `h_out` and copying back. The
            // attention's writes `h_out` (unchanged), the FFN's then reads `h_out`
            // and writes `h` — so `h` holds the block's final residual again and
            // the next block / `draft_head` are untouched. Same kernel, same
            // arguments but for the destination pointer, so both outputs are
            // bit-identical and the copy was a pure launch.
            //
            // WHY NOT `dsv41_hc_post_inplace`: that kernel has NO batch-row
            // dimension (`(res, x, post, comb, n, h)` — one `[n, h]` residual and
            // one `[h]` row), so a `bs = 5` draft would need `bs` launches where
            // this needs one. Its bit-exactness argument is per single row and
            // does not scale to the blocked layout.
            self.dev.hc_post(
                self.o.ptr as *const f32,
                self.h.ptr as *const f32,
                self.post.as_f32(),
                self.comb.as_f32(),
                self.h_out.ptr as *mut f32,
                bs as i32,
                hc as i32,
                dim as i32,
            )?;
            if !p3a.hcpost_swap {
                self.dev.memcpy_d2d(
                    self.h.ptr,
                    self.h_out.ptr,
                    (bs * hc * dim * 4) as usize,
                )?;
            }
            // The residual the FFN half reads: the post-attention row, which a2
            // leaves in `h_out` (the copy above is what used to move it into `h`).
            let res_cur: *const f32 = if p3a.hcpost_swap {
                self.h_out.ptr as *const f32
            } else {
                self.h.ptr as *const f32
            };

            // ---- FFN sub-block ----
            self.hc_mixes(
                res_cur,
                ld,
                true,
                ffn_out,
                self.post.ptr as *mut f32,
                self.comb.ptr as *mut f32,
            )?;
            // the FFN collapses with THIS block's attn_pre, not the incoming one
            let ffn_norm = need(&ld.ffn_norm, "mtp.*.ffn_norm.weight")?;
            self.dev.hc_collapse_norm(
                res_cur as *mut f32,
                self.pre_attn.ptr as *const f32,
                ffn_norm.as_f32(),
                self.xn.ptr as *mut f32,
                bs as i32,
                hc as i32,
                dim as i32,
                eps,
                crate::dsv41::chain_dev::bf16_truncate(),
            )?;
            // the MoE's input (post ffn-norm) — the same semantic as the golden
            // harness's `stage{s}.ffn.in`, for the MoE-segment diff.
            self.dump_unit_idx("ffn_in", s, self.xn.ptr as *const f32, &[bs, dim]);

            self.draft_moe(s, ld)?;
            // the MoE block's output, AFTER its all-reduce (full block on every
            // rank, not the rank's partial sum).
            self.dump_unit_idx("moe_out_block", s, self.moe_out.ptr as *const f32, &[bs, dim]);

            // P3a a2, FFN half: `res_cur` (= `h_out` under the gate) is read and
            // `h` is written, so no copy-back is needed and `h` carries the block's
            // final residual again (the `h_block` dump below stays on `h`).
            let (moe_res, moe_out): (*const f32, *mut f32) = if p3a.hcpost_swap {
                (res_cur, self.h.ptr as *mut f32)
            } else {
                (self.h.ptr as *const f32, self.h_out.ptr as *mut f32)
            };
            self.dev.hc_post(
                self.moe_out.ptr as *const f32,
                moe_res,
                self.post.as_f32(),
                self.comb.as_f32(),
                moe_out,
                bs as i32,
                hc as i32,
                dim as i32,
            )?;
            if !p3a.hcpost_swap {
                self.dev.memcpy_d2d(
                    self.h.ptr,
                    self.h_out.ptr,
                    (bs * hc * dim * 4) as usize,
                )?;
            }
            // the block's residual stream once both sub-blocks are in, i.e. the
            // `h` the NEXT block (or `forward_head`) reads.
            self.dump_unit_idx("h_block", s, self.h.ptr as *const f32, &[bs, hc, dim]);

            // the NEXT block's incoming premix is this block's ffn pre — P3a a3
            // makes that a pointer (`premix_cur` = the slot just written) instead
            // of this copy.
            if p3a.premix_pp {
                premix_cur = ffn_out as *const f32;
            } else {
                self.dev
                    .memcpy_d2d(self.pre_in.ptr, self.pre_ffn.ptr, (bs * hc * 4) as usize)?;
            }
        }

        // ---- forward_head: collapse, norm, head, then the Markov sampler ----
        self.draft_head()?;
        Ok(())
    }

    /// Can THIS draft forward go through the device-level graph right now?
    ///
    /// Every clause is either a concrete CAPTURE HAZARD — a host value the
    /// recorded region reads that a LATER position changes, which a capture
    /// would freeze at its recording-time value — or a missing capability of the
    /// loaded `.so`. A refused step takes the direct launches
    /// (`slot_dev = false`), bit for bit, which is also what keeps
    /// `DSV41_DRAFT_GRAPH` OFF indistinguishable from the historical path.
    fn draft_graph_arm(&self, pos: usize) -> bool {
        if !draft_graph_want() || self.graph_failed {
            return false;
        }
        // D2: the ring append must be the DEVICE-derived one. Without
        // `dsv41_ring_append` the recording would bake one ring slot (`window +
        // (pos % win)*hd`) and every replay would overwrite that same row.
        if !self.dev.supports_ring_append() {
            return false;
        }
        // The MoE zeroes `moe_out` on the sequential arm; a SYNCHRONOUS memset
        // runs on the legacy stream and would invalidate the capture.
        if !self.dev.supports_memset_async() {
            return false;
        }
        // A host barrier is not a CUDA call, so it would not be recorded and the
        // replayed graph would lose the inter-rank synchronisation of the MoE
        // all-reduce. `ar_v5` is the device-side protocol the recording CAN
        // carry — the same clause `DevChain::verify_graph_gate` carries, and
        // satisfied by default (`ar_v5` is on whenever the step graph is).
        if self.comm.is_some() && !crate::dsv41::tp::ar_v5() {
            return false;
        }
        // D3 + D4: the window geometry has to be in its STEADY state.
        // `win_rows(pos)` returns `(win, 0)` from `pos >= win` on and only then,
        // which is exactly what freezes the `window -> all_kv` copy's SIZE, its
        // `s0 == 0` branch, `sparse_attn`'s window argument and the `idxs`
        // table. Every earlier position has a different window, so a recording
        // could not follow it.
        //
        // `DSV41_SEED_ALIGN` rotates the ring's start slot (`s0`) every step,
        // turning that copy into two variable-length pieces at a host-computed
        // split — which no capture can represent. The aligned arm therefore
        // keeps the direct launches.
        if self.win < 1 || pos < self.win || crate::dsv41::chain_dev::seed_align() {
            return false;
        }
        // The golden per-unit capture probes the device from the HOST inside the
        // body (`dump_unit`'s D2H) and injects its inputs with blocking H2Ds, so
        // the whole debug arm stays on the direct launches.
        if unit_dump::enabled() || self.unit.is_some() {
            return false;
        }
        true
    }

    /// The `DSV41_DRAFT_GRAPH` CAPTURE arm: record [`Self::draft_body`] into a
    /// fresh graph, instantiate it, and run it ONCE — a capture records without
    /// executing, so the step still has to be launched.
    ///
    /// A capture is an OPTIMISATION: every failure (`capture_begin`,
    /// `capture_end` — a driver-rejected op inside the recording — the
    /// instantiate, the first launch) latches `graph_failed` and this step
    /// finishes on the direct launches, so nothing can take the draft down with
    /// it. That is the `verify_graph_failed` pattern.
    fn draft_capture(&mut self, pos: usize) -> Result<()> {
        // Rendezvous before the recording: a peer EXECUTING its MoE all-reduce
        // while this rank only RECORDS it would poll for a stamp this rank is
        // not publishing (see `Collective::host_barrier`'s note in tp.rs).
        if let Some(c) = self.comm.as_ref() {
            c.host_barrier();
        }
        let (g, mut err) = self.capture_draft(pos);
        // ... and once more after it, so no rank starts a step against a peer
        // still inside its recording.
        if let Some(c) = self.comm.as_ref() {
            c.host_barrier();
        }

        let mut exec: *mut c_void = std::ptr::null_mut();
        if err.is_none() && !g.is_null() {
            match self.dev.graph_instantiate(g) {
                Ok(e) => exec = e,
                Err(e) => err = Some(e),
            }
        }
        if !g.is_null() {
            // The captured GRAPH handle is released right after the instantiate:
            // only the EXEC is needed from here on.
            let _ = self.dev.graph_free(g, std::ptr::null_mut());
        }
        if err.is_none() {
            // The capture did not execute, so this launch IS this step's draft.
            if let Err(e) = self.dev.graph_launch(exec) {
                err = Some(e);
            }
        }
        if let Some(why) = err {
            if !exec.is_null() {
                let _ = self.dev.graph_free(std::ptr::null_mut(), exec);
            }
            // ★ PRINT, DO NOT SILENTLY DEGRADE. The graph is an optimisation, so
            // a refusal is *designed* to leave the draft on the direct launches —
            // which makes "graph on but never engaged" indistinguishable from
            // "engaged" without this line (the trap the verify switch was born
            // from).
            eprintln!(
                "[draft_graph] capture FAILED (pos={pos}): {why} — the draft stays on the \
                 direct launches (latched)"
            );
            self.graph_failed = true;
            self.draft_body(pos, true)?;
        } else {
            self.graph = Some(exec);
            self.graph_captures += 1;
            if self.rank == 0 {
                eprintln!(
                    "[draft_graph] captured the draft chain at pos={pos} (bs={}, mtp={}) — every \
                     later draft forward replays (DSV41_DRAFT_GRAPH=1)",
                    self.bs, self.cfg.n_mtp_layers
                );
            }
        }
        Ok(())
    }

    /// Record one [`Self::draft_body`] into a fresh graph.
    ///
    /// Returns `(graph, error)`: a non-null graph with `None` on success, or a
    /// null graph with the first failure otherwise. The stream is ALWAYS taken
    /// out of capture mode before returning — when `capture_begin` succeeded and
    /// the body failed, `capture_end` is what ends it — so the caller's fallback
    /// launch is legal. That is the whole point of this helper, and the same
    /// contract `DevChain::capture_verify` keeps.
    fn capture_draft(&mut self, pos: usize) -> (*mut c_void, Option<FerriteError>) {
        if let Err(e) = self.dev.capture_begin() {
            // Nothing was recorded, so there is no capture to end.
            return (std::ptr::null_mut(), Some(e));
        }
        let inner = self.draft_body(pos, true);
        let end = self.dev.capture_end();
        match (inner, end) {
            (Ok(()), Ok(g)) => (g, None),
            (inner, end) => {
                let e = end.err().or_else(|| inner.err()).expect("one arm failed");
                (std::ptr::null_mut(), Some(e))
            }
        }
    }

    /// `hc_mixes` for one sub-block of one draft block. `ffn` selects the
    /// FFN's trio; `pre_out` receives the pre the NEXT sub-block collapses with.
    /// `x` is the residual stream the dots read: `h` on the attention half, and
    /// under P3a's a2 the post-attention `h_out` on the FFN half (the residual
    /// destination swap — see `draft_forward`).
    fn hc_mixes(
        &self,
        x: *const f32,
        ld: &LayerDev,
        ffn: bool,
        pre_out: *mut f32,
        post: *mut f32,
        comb: *mut f32,
    ) -> Result<()> {
        let (f, sc, base) = if ffn {
            (
                need(&ld.hc_ffn_fn, "mtp.*.hc_ffn_fn")?,
                need(&ld.hc_ffn_scale, "mtp.*.hc_ffn_scale")?,
                need(&ld.hc_ffn_base, "mtp.*.hc_ffn_base")?,
            )
        } else {
            (
                need(&ld.hc_attn_fn, "mtp.*.hc_attn_fn")?,
                need(&ld.hc_attn_scale, "mtp.*.hc_attn_scale")?,
                need(&ld.hc_attn_base, "mtp.*.hc_attn_base")?,
            )
        };
        self.dev.hc_mixes(
            x,
            f.as_f32(),
            sc.as_f32(),
            base.as_f32(),
            pre_out,
            post,
            comb,
            self.bs as i32,
            (self.hc * self.dim) as i32,
            self.hc as i32,
            self.cfg.hc_sinkhorn_iters as i32,
            self.cfg.hc_eps,
        )
    }

    /// `DSparkAttention` (dspark.rs::dspark_attention): the draft block's own
    /// q/k/v, the window ring seeded from the main stream, and every draft query
    /// additionally attending its own block.
    ///
    /// `slot_dev` is passed straight through to [`Self::seed_window`]: it selects
    /// the device-derived ring destination that `DSV41_DRAFT_GRAPH` needs and is
    /// `false` on every direct path.
    fn draft_attention(&mut self, s: usize, pos: usize, slot_dev: bool) -> Result<()> {
        let cfg = self.cfg;
        let (dim, hd, nh, ql) = (self.dim, self.hd, self.nh, self.ql);
        let (bs, win, groups, hpg) = (self.bs, self.win, self.groups, self.hpg);
        let olg = self.olg;
        let ol_total = self.ol_total;
        let ld = &self.w.mtp[s];

        let wq_a = need(&ld.wq_a, "mtp.*.attn.wq_a.weight")?;
        let wq_a_s = need(&ld.wq_a_scale, "mtp.*.attn.wq_a.scale")?;
        let q_norm = need(&ld.q_norm, "mtp.*.attn.q_norm.weight")?;
        let wq_b = need(&ld.wq_b, "mtp.*.attn.wq_b.weight")?;
        let wq_b_s = need(&ld.wq_b_scale, "mtp.*.attn.wq_b.scale")?;
        let wkv = need(&ld.wkv, "mtp.*.attn.wkv.weight")?;
        let wkv_s = need(&ld.wkv_scale, "mtp.*.attn.wkv.scale")?;
        let kv_norm = need(&ld.kv_norm, "mtp.*.attn.kv_norm.weight")?;
        let attn_sink = need(&ld.attn_sink, "mtp.*.attn.attn_sink")?;

        // ---- the main stream's KV row goes into the ring ----
        // P0-1 FIX (draft-numerical-audit, 2026-09-12): the official seeds at
        // `start_pos` — the ANCHOR's position — for content, RoPE, and ring
        // slot alike (model.py:1039-1042 freqs_cis[start_pos], :1065
        // window_kv_cache[start_pos % win]). The historical `pos - 1` phased
        // every window row one position early relative to its content, a
        // first-order perturbation of the attention output and the #1 suspect
        // for the 1.02 accept rate (all main-chain KV relative distances
        // systematically off by one). The old comment claimed the content is
        // "hidden[anchor_pos - 1]" — the audit's read of model.py:1261-1267
        // shows the tap is captured BEFORE the target layer, i.e. at
        // start_pos, not start_pos - 1.
        // NOTE: if DSV41_SEED_ALIGN changes the calling convention to
        // draft_forward(next, pos+1), do NOT also apply this fix (the audit
        // warns the two would overshoot by one).
        debug_assert!(pos > 0, "draft_forward: the anchor is never at pos 0");
        self.seed_window(s, pos, slot_dev)?;

        // ---- q = wq_b(q_norm(wq_a(x))) with RoPE at the draft positions ----
        // D1 fix (audit-ffi-args): quantise ALL bs rows — the historical call
        // passed `dim` (ONE row) while the gemm below reads `bs × dim` bytes
        // and `bs·dim/32` scales, so rows 1..bs-1 consumed stale xq bytes and
        // the draft q/kv were silently garbage.
        self.quant1(self.xn.ptr as *const f32, bs * dim)?;
        self.dev.gemm_fp8_mx(
            self.xq.as_u8(),
            self.xsc.as_f32(),
            wq_a.as_u8(),
            wq_a_s.as_u8(),
            std::ptr::null(),
            self.qr.ptr as *mut f32,
            bs as i32,
            ql as i32,
            dim as i32,
        )?;
        self.dev.rmsnorm(
            self.qr.ptr as *const f32,
            q_norm.as_f32(),
            self.qr.ptr as *mut f32,
            bs as i32,
            ql as i32,
            cfg.norm_eps,
        )?;
        self.quant1(self.qr.ptr as *const f32, bs * ql)?;
        // wq_b is [nh * hd, ql]; one fp8 GEMM covers all bs draft rows.
        self.dev.gemm_fp8_mx(
            self.xq.as_u8(),
            self.xsc.as_f32(),
            wq_b.as_u8(),
            wq_b_s.as_u8(),
            std::ptr::null(),
            self.q.ptr as *mut f32,
            bs as i32,
            (nh * hd) as i32,
            ql as i32,
        )?;
        // the pre-RoPE projection — the unit-diff isolator between the
        // projection chain (wq_a/q_norm/wq_b) and the RoPE.
        self.dump_unit_idx("q_pre_rope", s, self.q.ptr as *const f32, &[bs, nh, hd]);
        self.rope_queries(self.q.ptr as *mut f32, pos)?;

        // ---- kv = wkv(x), normed and roped like the backbone's window KV ----
        // D1 fix (audit-ffi-args): quantise ALL bs rows — one row left rows 1..bs-1
        // reading stale xq bytes into the gemm below.
        self.quant1(self.xn.ptr as *const f32, bs * dim)?;
        self.dev.gemm_fp8_mx(
            self.xq.as_u8(),
            self.xsc.as_f32(),
            wkv.as_u8(),
            wkv_s.as_u8(),
            std::ptr::null(),
            self.kv.ptr as *mut f32,
            bs as i32,
            hd as i32,
            dim as i32,
        )?;
        self.dev.rmsnorm(
            self.kv.ptr as *const f32,
            kv_norm.as_f32(),
            self.kv.ptr as *mut f32,
            bs as i32,
            hd as i32,
            cfg.norm_eps,
        )?;
        // The draft rows sit at pos + 1 + r, one row per step — the SAME
        // per-row positions as the queries (the sglang arbitration; the host
        // comment's `seqlen` is the sequence length, not the block size).
        self.rope_at(
            self.kv.ptr as *mut f32,
            bs as i32,
            hd as i32,
            1,
            pos as i32,
            false,
        )?;

        // ---- the ring's window rows, then the draft block, contiguous ----
        // COMPACT layout: only the n_win LIVE window rows are copied, so the
        // block's rows sit at [n_win, n_win+bs) and the kernel's derived
        // geometry is self-consistent on BOTH formulas: n = window + *clen =
        // n_win + bs (the true row stride) and topk = window + min(*clen,
        // index_topk) = n_win + bs (the true candidate count). The historical
        // copy took ALL win rows (stride win+bs) while the kernel derived
        // n_win+bs from the parameters — every kv read past row n_win landed
        // `win - n_win` rows off once pos < win-1, which is exactly the
        // "draft outputs unrelated garbage" signature.
        //
        // `win_rows` also fixes the RING WRAP (`DSV41_SEED_ALIGN`): once the
        // ring has turned over, the window has to be copied in POSITION order
        // (`(pos - n_win + j) % win`), because the slot `pos % win` belongs to
        // the block's anchor row and must not appear as a window candidate —
        // the official mask allows exactly one KV per position, and the
        // slot-index-order copy handed the anchor a SECOND one while evicting
        // the oldest live row.
        let (n_win, s0) = self.win_rows(pos);
        let wbytes = (n_win * hd * 4) as usize;
        if s0 == 0 {
            self.dev
                .memcpy_d2d(self.all_kv.ptr, self.window[s].ptr, wbytes)?;
        } else {
            // The ring is circular: at most ONE wrap, so two copies cover the
            // window in position order.
            let first = (win - s0).min(n_win) * hd * 4;
            self.dev.memcpy_d2d(
                self.all_kv.ptr,
                (self.window[s].ptr as *const f32)
                    .wrapping_add(s0 * hd) as *const c_void,
                first,
            )?;
            if first < wbytes {
                self.dev.memcpy_d2d(
                    (self.all_kv.ptr as *mut u8).wrapping_add(first) as *mut c_void,
                    self.window[s].ptr as *const c_void,
                    wbytes - first,
                )?;
            }
        }
        self.dev.memcpy_d2d(
            (self.all_kv.ptr as *mut u8).wrapping_add(wbytes) as *mut c_void,
            self.kv.ptr,
            (bs * hd * 4) as usize,
        )?;
        // The anchor-KV seat: the window EXCLUDES the block's own row-0 slot
        // (n_win = min(win - 1, pos) — the ring's copy of that position either
        // does not exist yet or stays written for the NEXT round), so the
        // anchor (t0) at pos has exactly ONE kv in the candidates: the block's
        // row 0, its EMBED-derived projection — exactly the official block
        // structure (DeepSpec's draft blocks project their own inputs; the seat
        // arbitration in the earlier fix went one step further and swapped row
        // 0's bytes for the target-hidden projection mk, which is NOT what the
        // official blocks do — reverted).

        self.dev.sparse_attn(
            self.q.as_f32(),
            self.all_kv.ptr as *const f32,
            attn_sink.as_f32(),
            self.idxs.as_i32(),
            self.o.ptr as *mut f32,
            1,
            bs as i32,
            nh as i32,
            hd as i32,
            self.clen.ptr as *const std::os::raw::c_int,
            n_win as i32,
            bs as i32,
            (hd as f32).powf(-0.5),
            // W2-MROWS: the draft block's counter is one value for the whole
            // block (it is passed in as `self.clen`, not advanced per draft row),
            // so the scalar read stays - and `self.idxs` is the draft's own
            // [bs, ...] pitch (`topk` = n_win + bs here, so `idx_stride = 0`
            // reproduces the historical `topk` pitch).
            std::ptr::null(),
            0,
        )?;
        // inverse RoPE, same per-query positions
        self.rope_queries_inv(self.o.ptr as *mut f32, pos)?;

        // ---- grouped low-rank output projection ----
        // The reference's `grp` reshape is the IDENTITY (o is already
        // [bs, groups, hpg*hd] row-major), so no permute is needed. wo_a is
        // [groups * olg, hpg * hd] and each row of a group uses THAT group's
        // weight block: a BLOCK-DIAGONAL matmul, not one GEMM. The activation
        // stride between two rows of a group is nh*hd and one group's output
        // columns are strided by ol_total — both outside every GEMM launcher's
        // ABI, which is why this used to be one m=1 GEMV per (group, row): 8 x bs
        // = 40 launches per MTP block, each re-reading the group's whole weight
        // block (~500 MB/forward for a 33 MB weight set; the plan's §4 "wo_a
        // 分组投影", ~1.0ms). A "one gemm_fp8_mx over the whole wo_a" call cannot
        // express it for the same reason (see the kernel header).
        //
        // `wo_a_grouped_fp8` (dsv41_kernels.cu) is the weight-stationary form of
        // exactly the same GEMV: one launch for ALL groups (grid.y = group) in
        // which every block stages its group's weight rows ONCE and folds all
        // `bs` activation rows against them. Per (group, row) the arithmetic is
        // the m=1 gemv's order-preserving consume expression verbatim — same
        // ascending kb walk, same staged bytes, same serial `acc` chain and
        // shfl_xor tree, no cross-row recombination — so every output row is
        // BIT-IDENTICAL to the launch it replaces; the kernel header carries the
        // C1-C6 argument.
        //
        // Ok(false) = stale .so or a declined shape/mode: the per-(group, row)
        // loop below produces the same numbers by construction.
        self.quant1(self.o.ptr as *const f32, bs * nh * hd)?;
        let wo_a = need(&ld.wo_a, "mtp.*.attn.wo_a.weight")?;
        let wo_a_s = need(&ld.wo_a_scale, "mtp.*.attn.wo_a.scale")?;
        let k_grp = hpg * hd;
        let fused = self.dev.wo_a_grouped_fp8(
            self.xq.as_u8(),
            self.xsc.as_f32(),
            wo_a.as_u8(),
            wo_a_s.as_u8(),
            std::ptr::null(),
            self.wo.ptr as *mut f32,
            groups as i32,
            bs as i32,
            olg as i32,
            k_grp as i32,
            (nh * hd) as i32,   // activation row stride: one whole draft row
            ol_total as i32,    // output row stride: [bs, ol_total]
        )?;
        if !fused {
            for g in 0..groups {
                let wp = wo_a.as_u8().wrapping_add(g * olg * k_grp);
                let wsp = wo_a_s
                    .as_u8()
                    .wrapping_add((g * olg / 32) * (k_grp / 32));
                for r in 0..bs {
                    let a = self
                        .xq
                        .as_u8()
                        .wrapping_add((r * nh + g * hpg) * hd);
                    let asc = self
                        .xsc
                        .as_f32()
                        .wrapping_add(((r * nh + g * hpg) * hd / 32) as usize);
                    let out = (self.wo.ptr as *mut f32)
                        .wrapping_add(r * ol_total + g * olg);
                    self.dev.gemm_fp8_mx(
                        a,
                        asc,
                        wp,
                        wsp,
                        std::ptr::null(),
                        out,
                        1,
                        olg as i32,
                        k_grp as i32,
                    )?;
                }
            }
        }
        let wo_b = need(&ld.wo_b, "mtp.*.attn.wo_b.weight")?;
        let wo_b_s = need(&ld.wo_b_scale, "mtp.*.attn.wo_b.scale")?;
        self.quant1(self.wo.ptr as *const f32, bs * ol_total)?;
        self.dev.gemm_fp8_mx(
            self.xq.as_u8(),
            self.xsc.as_f32(),
            wo_b.as_u8(),
            wo_b_s.as_u8(),
            std::ptr::null(),
            self.o.ptr as *mut f32,
            bs as i32,
            dim as i32,
            ol_total as i32,
        )?;
        // DRAFT ATTN BF16 ROUND-TRIP (DSV41_DRAFT_ATTN_BF16, default OFF): the
        // official DSparkBlock's attention output (after wo_b) carries the model
        // dtype (bf16) — the MTP head's hc_post consumes a bf16 tensor. ferrite
        // computes in f32, so this round-trip aligns the draft's attention
        // output precision with what the head's weights are calibrated for.
        // The remaining known misalignment (after tap+hc+MoE); OFF keeps f32.
        if draft_attn_bf16() {
            let n = (bs * dim) as i64;
            self.dev.bf16_roundtrip(self.o.ptr as *mut f32, n)?;
        }
        Ok(())
    }

    /// The MoE half of a draft block: the gate, the routed experts and the
    /// shared expert. `bs` rows at a time (the host runs `bs` rows through the
    /// same `moe_forward`).
    ///
    /// # TP geometry (this is a TP-split MoE, not a replicated one)
    ///
    /// The draft's ATTENTION is replicated (every rank holds the whole wq_b /
    /// wo_a / head and maps the GLOBAL head/group/vocab geometry), but its MoE is
    /// split exactly like the backbone's — the loader cuts each routed expert's
    /// `inter` axis by `world` and the shared expert's w1/w3 rows / w2 columns
    /// when `DSV41_SHARED_TP` is on. So the expert kernels here are sized by the
    /// LOCAL `inter_local` and the shared pair by `sh_il`, and every rank's
    /// `moe_out` is only a PARTIAL sum: the all-reduce at the tail (with
    /// `end_round`, the same pair `DevChain::moe_reduce` issues) is what makes
    /// the block identical on every rank. Without it each rank would carry
    /// `1/world` of the MoE — silently, which is worse than a fault.
    fn draft_moe(&mut self, s: usize, ld: &LayerDev) -> Result<()> {
        let cfg = self.cfg;
        let (dim, bs, world, rank) = (self.dim, self.bs, self.world, self.rank);
        let (inter_local, sh_il) = (self.inter_local, self.sh_il);
        let layer = cfg.n_layers + s;
        let (n_routed, topk) = cfg.moe_config(layer);
        let topk = topk.max(1);
        let n_routed = n_routed.max(1);
        // Which ranks contribute the shared expert (the backbone's rule verbatim):
        // all of them under DSV41_SHARED_TP, where each holds its own
        // `inter / world` slice; rank 0 alone under the replicated layout, where
        // a second contributor would make the all-reduce sum it `world` times.
        let stp = crate::dsv41::weights::shared_expert_tp();
        let shared_rank = if stp { true } else { rank == 0 };

        // MOE BF16 DOMAIN (draft-numerical-audit C#9; `DSV41_DRAFT_BF16_DOMAIN`,
        // default OFF): the official block's ffn input `x` is a bf16 tensor —
        // the gate is `F.linear(x.float(), weight.float())` (model.py:811) and
        // the routed experts' `act_quant` reads that SAME bf16 `x`. ferrite's
        // `xn` is the f32 ffn-norm output, so the gate's 384-way near-tie
        // selection AND the experts' fp4/fp8 activation scale sit in a different
        // domain. ONE in-place round-trip here covers every `xn` reader below —
        // gate gemv, route, quant_fp4/quant_fp8, and the shared expert's
        // quant1: they all want the bf16 value and none needs the f32 one.
        // (Already a no-op under `DSV41_BF16_TRUNCATE`, whose `hc_collapse_norm`
        // wrote `xn` bf16 — idempotent, so the two gates compose.)
        if draft_bf16_domain() {
            let n = (bs * dim) as i64;
            self.dev.bf16_roundtrip(self.xn.ptr as *mut f32, n)?;
        }

        // ---- gate + route ----
        // The gate is bf16 and the per-row GEMV is M=1, so the historical shape is
        // `bs` launches — the same call the backbone's MoE makes.
        //
        // P3b b3 (`DSV41_DRAFT_P3B`, see [`draft_p3b`]): ONE multi-row GEMV
        // (`ferrite_gemv_bf16_v2_mrows`) where the loop below issues `bs`
        // single-row `gemv_bf16` launches. The entry runs the v2 program with a
        // row dimension — same WPR heuristic (`gv2_wpr(n_routed)`), same K-slice
        // walk, same uint4/8-element FMA groups, same smem partial fold, one
        // independent accumulator per row — so row r is bit-identical to the
        // per-row call it replaces, PROVIDED the per-row path would take v2; the
        // wrapper enforces that (`gemv_bf16_v2_wanted(n_routed)` + the symbol) and
        // returns Ok(false) otherwise, keeping the loop. This is the draft-side
        // mirror of `DevChain::moe_rows`'s `row_fold_gate` arm
        // (`chain_dev.rs`, DSV41_ROW_FOLD_GATE / DSV41_GATE_MROWS).
        let gate_w = need(&ld.gate_w, "mtp.*.ffn.gate.weight")?;
        let p3b = draft_p3b();
        let gate_folded = p3b.gate_mrows
            && self.dev.gemv_bf16_v2_mrows(
                gate_w.ptr() as *const c_void,
                self.xn.ptr as *const f32,
                self.scores.ptr as *mut f32,
                bs as i32,
                n_routed as i32,
                dim as i32,
            )?;
        if !gate_folded {
            for r in 0..bs {
                self.dev.gemv_bf16(
                    gate_w.ptr() as *const c_void,
                    (self.xn.ptr as *const f32).wrapping_add(r * dim),
                    (self.scores.ptr as *mut f32).wrapping_add(r * n_routed),
                    n_routed as i32,
                    dim as i32,
                )?;
            }
        }
        let gate_bias = ld.gate_bias.as_ref().map(|b| b.as_f32());
        self.dev.route_topk(
            self.scores.as_f32(),
            gate_bias.unwrap_or(std::ptr::null()),
            self.route_w.ptr as *mut f32,
            self.route_idx.ptr as *mut i32,
            std::ptr::null_mut(),
            bs as i32,
            n_routed as i32,
            topk as i32,
            cfg.norm_topk_prob,
            cfg.route_scale,
            2, // sqrtsoftplus, per the checkpoint's routing
        )?;
        // the routing decision — the golden harness's `stage{s}.ffn.gate.indices/
        // weights` counterparts, for the MoE-segment diff.
        self.dump_unit_idx("route_idx", s, self.route_idx.ptr as *const f32, &[bs, topk]);
        self.dump_unit_idx("route_w", s, self.route_w.ptr as *const f32, &[bs, topk]);

        // The expert weights are fp4; the input row is quantised once for all
        // slots (the reference re-quantises it per expert, which is pure waste).
        self.dev.quant_fp4(
            self.xn.ptr as *const f32,
            self.xq4.ptr as *mut u8,
            self.xsc4.ptr as *mut f32,
            bs as i32,
            dim as i32,
            32,
            true,
        )?;
        // e2m1x2 (`DSV41_EXPERT_ACT_E4M3`, default OFF): the SECOND term of the
        // activation decomposition — the same contract the backbone's `moe()`
        // runs (chain_dev.rs, the two-pass block after its `quant_fp4`). The
        // call above already IS q_hi (`xq4`/`xsc4`); here we form the f32
        // residual `xn - dequant(q_hi)` and quantise IT to e2m1 as well. Both
        // gate/up passes then run against the SAME fp4 weight bytes and their
        // outputs are summed (below, BEFORE swiglu — swiglu is non-linear). This
        // is the draft/verify precision alignment: the draft's routed experts
        // were the lone e2m1 consumer (8.4x the verify-side rel-L2), and the
        // gate is deliberately SHARED with the verify chain so both domains stay
        // in lockstep. Guarded on the .so symbol so a stock build keeps the
        // single-pass path with a one-shot notice instead of failing here.
        // e4m3 DIRECT (`DSV41_EXPERT_ACT_E4M3`, default OFF): the SAME path the
        // backbone's `moe()` runs (chain_dev.rs, the `if e4m3` quantisation).
        // The activation takes the OFFICIAL `act_quant(e4m3, block=32)` form and
        // is consumed by ONE gate/up pass, so the draft's routed experts and the
        // verify chain stay in lockstep on the one shared gate without the
        // retired e2m1x2 second pass + add. Guarded on the .so symbol so a stock
        // build keeps the e2m1 path with a one-shot notice instead of failing.
        let e4m3 = crate::dsv41::chain_dev::expert_act_e4m3()
            && self.dev.supports_expert_act_e4m3();
        if crate::dsv41::chain_dev::expert_act_e4m3() && !e4m3 {
            crate::dsv41::chain_dev::act_e4m3_skipped_note();
        }
        if e4m3 {
            self.dev.quant_fp8(
                self.xn.ptr as *const f32,
                self.xq4.ptr as *mut u8,
                self.xsc4.ptr as *mut f32,
                bs as i32,
                dim as i32,
                32,
                true,
            )?;
        } else {
            self.dev.quant_fp4(
                self.xn.ptr as *const f32,
                self.xq4.ptr as *mut u8,
                self.xsc4.ptr as *mut f32,
                bs as i32,
                dim as i32,
                32,
                true,
            )?;
        }

        let ne = ld.experts.len();
        if ne >= 2 {
            let (a, b) = (&ld.experts[0], &ld.experts[1]);
            let d = |x: *mut c_void, y: *mut c_void| (y as i64) - (x as i64);
            let strides = (
                a.w1.ptr() as *const u8,
                d(a.w1.ptr(), b.w1.ptr()),
                a.w1_scale.ptr() as *const u8,
                d(a.w1_scale.ptr(), b.w1_scale.ptr()),
                a.w3.ptr() as *const u8,
                d(a.w3.ptr(), b.w3.ptr()),
                a.w3_scale.ptr() as *const u8,
                d(a.w3_scale.ptr(), b.w3_scale.ptr()),
                a.w2.ptr() as *const u8,
                d(a.w2.ptr(), b.w2.ptr()),
                a.w2_scale.ptr() as *const u8,
                d(a.w2_scale.ptr(), b.w2_scale.ptr()),
            );
            let ids = self.route_idx.ptr as *const i32;
            // Defect 2 (audit-moe-seg): the batched launcher's own `fuse` test
            // (`dsv41_experts_mxf4.cu`: `g_fuse && g_expert_fp4_mode == 2 &&
            // dim % 512 == 0`) decides whether the epilogue writes the swiglu'd
            // inter-width slice or the full 2*inter gate|up block. A host that
            // guesses wrong walks `act_slot` off by `inter` and runs (or skips)
            // the separate swiglu against a half that was never written — silent
            // garbage. Mirror the SAME expression the backbone's `moe_rows` uses
            // (`chain_dev.rs:5378-5386`), including the `dim % 512` term the
            // single-row call site omits.
            let gateup_fused = crate::dsv41::chain_dev::gateup_fuse()
                && self.dev.supports_gateup_fuse()
                && crate::dsv41::chain_dev::expert_fp4_mode() == 2
                && (dim % 512) == 0;
            // Defect 1 (audit-moe-seg): the interleaved pool is addressable ONLY
            // by the batched gate/up reader's trailing `ilv` argument. The old
            // condition (`!ld.experts_ilv && ...`) was therefore always FALSE in
            // the default configuration (DSV41_EXPERT_ILV defaults ON and the
            // loader makes ONE layout decision shared by every layer and by the
            // `mtp` blocks), so the draft fell through to the SEQUENTIAL indirect
            // reader over interleaved bytes -> shuffled gate/up, i.e. the
            // MoE-segment 138% deviation. `experts_ilv` must not veto the batched
            // path; it is handed to the kernel instead.
            if self.dev.supports_moe_batch() {
                // The interleaved layout is readable by the batched gate/up call
                // with EITHER epilogue now (see chain_dev.rs::moe_rows): the
                // kernel's gate/up PAIR body derives the up bytes from the gate
                // pointer and reads ONE LDG.128 per group, and its epilogue is
                // chosen by the caller's slot pitch. `gateup_fused == false` no
                // longer means "unreadable pool" — it means the raw [2*inter]
                // pair is written and the separate swiglu pass below runs on it.
                // What still binds is the pair body's K contract, dim % 512 == 0
                // (no tail loop; the launcher refuses the combination loudly).
                if ld.experts_ilv && (dim % 512) != 0 {
                    return Err(FerriteError::Config(
                        "draft_moe: routed expert gate/up weights are interleaved \
                         (DSV41_EXPERT_ILV) but dim % 512 != 0 — the interleaved pair body \
                         walks whole 512-value groups and has no tail; run with \
                         DSV41_EXPERT_ILV=0 or on a dim divisible by 512"
                            .into(),
                    ));
                }
                // The batched expert launchers are MULTI-ROW as of the
                // `moe-mrows-impl` kernel change: `rows` is the grid's third
                // dimension (`blockIdx.z` = the activation row) and each row's
                // pointers are derived INSIDE the kernel from the arguments plus
                // `gridDim.y` (= slots = topk): a/a_scale at arow*(dim/2) /
                // arow*(dim/32) (the quantiser's own packed layout), ids at
                // [arow*topk + slot], out at (arow*topk + slot)*out_slot_stride,
                // act/rw at (arow*topk + slot)*act_stride. So ONE call now covers
                // all bs rows, with the host passing the row-0 BASE of each
                // [row][slot][...] buffer - exactly the offsets the old per-row
                // loop computed by hand (r*(dim/2), r*(dim/32), r*row_pitch with
                // row_pitch = topk*act_slot, ids + r*topk, out + r*dim).
                //
                // BIT-EXACTNESS: an activation row shares no output element, no
                // accumulator and no smem staging with any other row (`arow` only
                // shifts base pointers - see the kernel's ROW INDEPENDENCE block),
                // so row r of this call runs the identical instruction sequence the
                // rows=1 call ran for row r. rows == bs > 1 no longer means "row 0
                // only"; rows == 1 is the previous launch bit for bit.
                //
                // Defect 2: the fused epilogue writes only `inter` floats per
                // slot (the swiglu'd half), so `act_slot` shrinks from `2*inter`
                // to `inter` exactly when it ran. `act_slot` is also the kernel's
                // per-slot act/out stride, so ex_act_b (= [row][slot][act_slot])
                // and moe_out (= [row][dim]) keep their layout unchanged.
                let act_slot = if gateup_fused {
                    inter_local as i64
                } else {
                    (2 * inter_local) as i64
                };
                // DSV41_DRAFT_MOE_MROWS (DEFAULT OFF, see `draft_moe_mrows`):
                // ONE `rows = bs` launch per stage when armed, `bs` `rows = 1`
                // launches (one per activation row, at that row's OWN base
                // pointers) when not. The two arms deliberately share this
                // single call site — `rows` and the loop bound are the ONLY
                // difference — so the A/B cannot drift apart in their arguments.
                //
                // `row_pitch` is the [row][slot][...] pitch the multi-row
                // kernels derive internally (`slots * out_slot_stride` =
                // `topk * act_slot`, the expression quoted in their layout
                // contract); the per-row arm applies it on the host instead, as
                // the call site did before the launchers grew a row dimension.
                // Both address byte-identical locations, and every base pointer
                // below is row r's own base either way: `r == 0` when the
                // multi-row arm is armed, and the kernel then walks the
                // remaining rows itself from `gridDim.y` (= slots = topk).
                let row_pitch = (topk as usize) * (act_slot as usize);
                let mrows = p3b.moe_mrows;
                let rows = if mrows { bs as i32 } else { 1 };
                let n_launches = if mrows { 1 } else { bs };
                for r in 0..n_launches {
                    // e4m3 packs ONE byte per value, so the per-row pitch of the
                    // activation is `dim` (not `dim/2`) whenever the gate is armed.
                    let abytes = if e4m3 { dim } else { dim / 2 };
                    let xq4_r = self.xq4.as_u8().wrapping_add(r * abytes);
                    let xsc4_r = self.xsc4.as_f32().wrapping_add(r * (dim / 32));
                    let act_r = (self.ex_act_b.ptr as *mut f32).wrapping_add(r * row_pitch);
                    let ids_r = ids.wrapping_add(r * topk);
                    let out_r = (self.moe_out.ptr as *mut f32).wrapping_add(r * dim);
                    let rw_r = (self.route_w.ptr as *const f32).wrapping_add(r * topk);
                    // ONE pass: `xq4`/`xsc4` hold the activation in whichever form
                    // the quantisation above produced (e4m3 bytes when `e4m3` is
                    // set, packed e2m1 otherwise) and `act_e4m3` selects the
                    // decoder. `act_slot` follows the fusion decision above on both
                    // arms; the fp4 weight bytes are untouched either way.
                    self.dev.expert_gate_up_fp4_batched(
                        xq4_r,
                        xsc4_r,
                        act_r,
                        act_slot,
                        rows,
                        dim as i32,
                        inter_local as i32,
                        cfg.swiglu_limit,
                        topk as i32,
                        strides.0,
                        strides.1,
                        strides.2,
                        strides.3,
                        strides.4,
                        strides.5,
                        strides.6,
                        strides.7,
                        ids_r,
                        // Defect 1: the pool layout (was a hard-wired 0, which
                        // only ever matched a non-interleaved pool).
                        ld.experts_ilv as i32,
                        e4m3 as i32,
                    )?;
                    // Defect 2: the fused epilogue already applied swiglu in
                    // place, so the separate pass must be skipped exactly when
                    // it ran — it would otherwise read the never-written up half
                    // of every slot. The pass is element-wise and row-local and
                    // its kernel walks the same [rows][slot][slot_stride]
                    // layout, so one `rows = bs` launch replaces the per-row
                    // loop with no order to preserve. Under `two` it runs ONCE on
                    // the summed pair (see the add above) - `gateup_fused` is
                    // forced false, so this launch is always taken.
                    if !gateup_fused {
                        self.dev.swiglu_limit_batched(
                            act_r,
                            rows,
                            inter_local as i32,
                            cfg.swiglu_limit,
                            act_slot,
                            topk as i32,
                        )?;
                    }
                    self.dev.expert_down_reduce_fp4_batched(
                        act_r as *const f32,
                        act_slot,
                        out_r,
                        rows,
                        dim as i32,
                        inter_local as i32,
                        rw_r,
                        1,
                        topk as i32,
                        strides.8,
                        strides.9,
                        strides.10,
                        strides.11,
                        ids_r,
                    )?;
                }
            } else {
                // Defect 3 (audit-moe-seg): the sequential `*_indirect` readers
                // walk the plain [w1][w3] pools; against an interleaved pool they
                // read shuffled bytes. Refuse instead of emitting garbage.
                if ld.experts_ilv {
                    return Err(FerriteError::Config(
                        "draft_moe: the interleaved pool needs the batched path".into(),
                    ));
                }
                self.dev.zero(&self.moe_out)?;
                // Defect 3: `expert_gate_up_fp4_indirect` with `rows > 1` takes
                // the gemm body, while its `row_weight` handling belongs to the
                // M=1 body (which reads `row_weight[0]`). Issuing `rows = bs` with
                // the single `route_w + slot` pointer therefore routes every row
                // through row 0's weight. Issue one row per launch instead
                // (rows = 1), each with that row's packed fp4 bytes, its
                // `ids`/`route_w` slice and its own `[2*inter_local]` block.
                for r in 0..bs {
                    let act = (self.ex_act.ptr as *mut f32).wrapping_add(r * 2 * inter_local);
                    let out = (self.moe_out.ptr as *mut f32).wrapping_add(r * dim);
                    let ids_r = ids.wrapping_add(r * topk);
                    // e4m3 packs ONE byte per value, so this row's activation
                    // starts at `r * dim` (not `r * dim/2`) when the gate is armed.
                    let abytes = if e4m3 { dim } else { dim / 2 };
                    let xq4_r = self.xq4.as_u8().wrapping_add(r * abytes);
                    let xsc4_r = self.xsc4.as_f32().wrapping_add(r * (dim / 32));
                    for slot in 0..topk {
                        let w = (self.route_w.ptr as *const f32).wrapping_add(r * topk + slot);
                        // ONE pass: the activation form is selected by `e4m3`
                        // and `expert_gate_up_fp4_indirect` is epi_mode 1
                        // (clamp + WRITE), so the row's [2*inter_local] block is
                        // produced by a single launch.
                        self.dev.expert_gate_up_fp4_indirect(
                            xq4_r,
                            xsc4_r,
                            act,
                            1,
                            dim as i32,
                            inter_local as i32,
                            cfg.swiglu_limit,
                            strides.0,
                            strides.1,
                            strides.2,
                            strides.3,
                            strides.4,
                            strides.5,
                            strides.6,
                            strides.7,
                            ids_r,
                            slot as i32,
                            e4m3 as i32,
                        )?;
                        self.dev.swiglu_limit(act, 1, inter_local as i32, cfg.swiglu_limit)?;
                        self.dev.expert_down_fp4_indirect(
                            act as *const f32,
                            out,
                            1,
                            dim as i32,
                            inter_local as i32,
                            w,
                            strides.8,
                            strides.9,
                            strides.10,
                            strides.11,
                            ids_r,
                            slot as i32,
                        )?;
                    }
                }
            }
        } else {
            self.dev.zero(&self.moe_out)?;
        }

        // ---- shared expert (one row at a time, unless P3b's b1 folds it) ----
        // Only the rank(s) that actually contribute it run this half: all of them
        // under DSV41_SHARED_TP (each computing its own `inter / world` slice of
        // the Rows-sharded w1/w3), rank 0 alone under the replicated layout
        // (weights.rs::shared_expert_tp). `shared_rank == false` means the whole
        // block — including the merge — is skipped, so no stale `shared_out` can
        // leak into the all-reduce.
        let sh_w = if shared_rank {
            match (
                ld.shared_w1.as_ref(),
                ld.shared_w1_scale.as_ref(),
                ld.shared_w3.as_ref(),
                ld.shared_w3_scale.as_ref(),
                ld.shared_w2.as_ref(),
                ld.shared_w2_scale.as_ref(),
            ) {
                (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)) => Some((a, b, c, d, e, f)),
                _ => None,
            }
        } else {
            None
        };
        if let Some((w1, w1s, w3, w3s, w2, w2s)) = sh_w {
            // ONE fp8 activation for the whole block, whichever arm runs: the
            // per-row loop addresses row r at `+r*dim` bytes / `+r*dim/32`
            // scales, which IS `quant_fp8(rows = bs, cols = dim, block = 32)`'s
            // layout, so this single call is also what the multi-row pass below
            // consumes. It must stay OUTSIDE the two arms so an arm switch cannot
            // change the staged bytes.
            self.quant1(self.xn.ptr as *const f32, bs * dim)?;
            // P3b b1 (`DSV41_DRAFT_P3B`): the whole `(w1 | w3) -> swiglu+quant ->
            // w2` chain as ONE multi-row pass. Ok(false) = gate off, a shape the
            // kernels decline, or a stale `.so` — the per-row loop follows and
            // rewrites every buffer this attempt touched, so a partial attempt is
            // harmless by construction (only `moe_out` is not scratch, and it is
            // written by the merge `add_inplace` alone).
            if !(p3b.sh_exp_mrows
                && self.shared_expert_mrows(w1, w1s, w3, w3s, w2, w2s)?)
            {
                // P3b b4 (per-row arm only): fold the merge into w2's epilogue
                // (`out += w @ a`, straight into `moe_out`) instead of writing a
                // disjoint `shared_out` row and folding it in afterwards.
                //
                // ALL-OR-NOTHING, and that is why the decision is re-taken inside
                // the loop: the C entry's shape test is a function of (n, k, m)
                // alone, so it is row-INDEPENDENT — but if it ever declined on a
                // later row, row 0 would already sit in `moe_out` and the trailing
                // merge would either double-add it (from a stale `shared_out`
                // row) or skip the remaining rows. A decline on row 0 is clean
                // instead: the entry returns BEFORE its launch, so nothing is
                // written and the plain per-row arm below is the untouched
                // fallback.
                let mut epi_add = p3b.sh_epi_add && self.dev.supports_gemm_fp8_add();
                for r in 0..bs {
                    let a = self.xq.as_u8().wrapping_add(r * dim);
                    let asc = self.xsc.as_f32().wrapping_add(r * dim / 32);
                    // `sh_il` (inter/world under SHARED_TP, inter otherwise) is the
                    // width THIS rank's w1/w3 actually have — asking for `inter_local`
                    // here is what walked 8x past the local slice (`inter_local` is
                    // 320 under TP8, the tensors hold 288 rows).
                    self.dev.gemm_fp8_mx(
                        a,
                        asc,
                        w1.as_u8(),
                        w1s.as_u8(),
                        std::ptr::null(),
                        self.ex_act.ptr as *mut f32,
                        1,
                        sh_il as i32,
                        dim as i32,
                    )?;
                    self.dev.gemm_fp8_mx(
                        a,
                        asc,
                        w3.as_u8(),
                        w3s.as_u8(),
                        std::ptr::null(),
                        (self.ex_act.ptr as *mut f32).wrapping_add(sh_il),
                        1,
                        sh_il as i32,
                        dim as i32,
                    )?;
                    self.dev.swiglu_limit(
                        self.ex_act.ptr as *mut f32,
                        1,
                        sh_il as i32,
                        cfg.swiglu_limit,
                    )?;
                    self.quant1(self.ex_act.ptr as *const f32, sh_il)?;
                    // w2 is Cols-sharded: [dim, sh_il] locally, so its reduction runs
                    // over this rank's slice and the OUTPUT is a partial [dim].
                    if epi_add {
                        if self.dev.gemm_fp8_mx_add(
                            self.xq.as_u8(),
                            self.xsc.as_f32(),
                            w2.as_u8(),
                            w2s.as_u8(),
                            std::ptr::null(),
                            (self.moe_out.ptr as *mut f32).wrapping_add(r * dim),
                            1,
                            dim as i32,
                            sh_il as i32,
                        )? {
                            // Folded into `moe_out[r*dim .. +dim]`. The shared
                            // output is then NOT materialised, so the
                            // `shared_out` unit dump below cannot be taken (see
                            // the note there).
                            continue;
                        }
                        if r > 0 {
                            // Unreachable: the shape is row-independent and row 0
                            // already folded. Fail LOUDLY — a silent fallback here
                            // would merge a stale `shared_out` row on top of an
                            // already-folded one, i.e. a plausible wrong block.
                            return Err(FerriteError::Config(format!(
                                "draft_moe: the shared expert's epilogue fold \
                                 (DSV41_P3B_SH_EPI_ADD) took row 0 and then declined row \
                                 {r} of {bs}: the shape test is row-independent, so the \
                                 launcher/.so is inconsistent — re-run without \
                                 DSV41_P3B_SH_EPI_ADD"
                            )));
                        }
                        // Row 0 declined and wrote nothing: the plain arm below is
                        // still a clean fallback, merge included.
                        epi_add = false;
                    }
                    // `out_row` is `shared_out`'s row: the dump target.
                    self.dev.gemm_fp8_mx(
                        self.xq.as_u8(),
                        self.xsc.as_f32(),
                        w2.as_u8(),
                        w2s.as_u8(),
                        std::ptr::null(),
                        (self.shared_out.ptr as *mut f32).wrapping_add(r * dim),
                        1,
                        dim as i32,
                        sh_il as i32,
                    )?;
                }
                // `moe_out += shared_out` — the merge. Skipped exactly when every
                // row took the epilogue fold above (`epi_add`), where the sum is
                // already in `moe_out`; the per-row arm writes disjoint `dim`-wide
                // ranges, so the element-wise add is bit-identical to the `dim`
                // calls the pre-P3b code issued.
                if !epi_add {
                    self.dev.add_inplace(
                        &self.moe_out,
                        &self.shared_out,
                        (bs * dim) as i64,
                    )?;
                }
            }
            // the shared expert's output BEFORE the routed sum — the golden
            // harness's `stage{s}.ffn.shared.out`. ⚠️ Only the two non-folded
            // arms materialise `shared_out` into its own buffer: the b1 pass
            // writes it (as `[bs, dim]`, same layout) and the per-row arm writes
            // it per row, so the dump keeps its meaning on both. The b4 epilogue
            // fold has no `shared_out` to record, which is one of the reasons it
            // is not the default (see [`draft_p3b`]).
            self.dump_unit_idx("shared_out", s, self.shared_out.ptr as *const f32, &[bs, dim]);
        }

        // ---- the MoE all-reduce ----
        // Both halves above are TP-split, so `moe_out` is a PARTIAL sum and the
        // full block is its sum over the ranks. The draft must produce the SAME
        // block on every rank (the verify path re-runs it under each rank's own
        // chain), so this is not optional — and `end_round` pairs with it exactly
        // as `DevChain::moe_reduce` does, releasing the round before the next
        // block's reduce reuses the staging slots.
        if world > 1 {
            let Some(c) = self.comm.as_ref() else {
                // Fail loudly: skipping the AR would look like a working draft
                // whose MoE is `1/world` of the truth, and the accept-length
                // statistics would silently collapse.
                return Err(FerriteError::Config(format!(
                    "dspark: world={world} but no collective is attached — the draft's MoE \
                     all-reduce cannot run (call DsparkDev::set_comm before draft_forward)"
                )));
            };
            c.all_reduce_inplace(
                self.moe_out.ptr as *mut c_void,
                bs * dim * std::mem::size_of::<f32>(),
            )?;
            c.end_round();
        }
        Ok(())
    }

    /// P3b b1 (`DSV41_DRAFT_P3B`, DEFAULT OFF): the draft's shared expert as ONE
    /// multi-row pass — the draft-side mirror of `DevChain::shared_expert_mrows`
    /// (`chain_dev.rs`, `DSV41_SH_EXP_MROWS`), applied to the draft's own scratch.
    ///
    /// # What it removes
    ///
    /// The shared expert is a SINGLE expert applied to every row, so its weights
    /// are identical across the block — yet the per-row loop reads them once per
    /// row. At the production shape (`bs` = 5, `sh_il` = 288, dim = 5120) that is
    /// `5 x bs` launches and `5 x bs` weight passes per MTP block, i.e.
    /// **20 launches/block of pure repeat traffic**. Unlike the routed half —
    /// whose rows select DIFFERENT experts, so only its launch count can shrink —
    /// this one's bytes really do divide by `bs`.
    ///
    /// # The four steps, each ONE launch
    ///
    /// The caller has already staged the block's fp8 activation with
    /// `quant1(xn, bs*dim)`; that call IS `quant_fp8(rows = bs, cols = dim,
    /// block = 32)` — the per-row loop addresses row `r` at `+r*dim` bytes and
    /// `+r*dim/32` scales, which is exactly that call's layout — so the
    /// multi-row pass consumes it unchanged. Then:
    ///
    /// 1. two `gemm_fp8_mrows` (w1 at `+0`, w3 at `+sh_il`) into a `[bs,
    ///    2*sh_il]` block in `ex_act`. `gemm_fp8_mrows` is the weight-stationary
    ///    form of `gemm_fp8_mx`'s **m == 1** program (the kernel header carries
    ///    the C1-C6 argument), so row `r` is bit-identical to the `m = 1` GEMV
    ///    the loop issues for row `r`.
    /// 2. `swiglu_limit_q(rows = bs)` — ONE launch replacing the `swiglu_limit` +
    ///    `quant1` pair. The kernel indexes `gate_up + r*2*inter`, which IS the
    ///    layout step 1 wrote (`inter` here is `sh_il`), and its per-(row, block)
    ///    arithmetic is the fused epilogue `tests_dsv41_glue.cu`'s `swiglu_q` case
    ///    asserts bit-exact against (`swiglu_limit` + `quant1`). Its fp8 output
    ///    lands in the SAME `xq`/`xsc` at pitch `sh_il` / `sh_il/32`, which is the
    ///    activation step 3 reads. `sh_il % 32 == 0` is what lets every 32-element
    ///    scale block stay inside one warp's tile, and it is checked here.
    /// 3. `gemm_fp8_mrows(w2)` into `shared_out` at pitch `dim` — the same
    ///    `[bs, dim]` layout the per-row loop's `shared_out + r*dim` writes, so
    ///    the `shared_out` unit dump keeps its meaning.
    ///
    /// The merge (`add_inplace`) is left to the caller: it is the one element-wise
    /// step whose operand (`moe_out`) is not scratch, and keeping it outside means
    /// a decline at any step above is recoverable by the per-row fallback.
    ///
    /// # Fallback (the reference this was verified against)
    ///
    /// `Ok(false)` — leaving the per-row loop to run — when `bs` is outside the
    /// kernels' 1..=8 dispatch, `sh_il % 32` / `dim % 32` is non-zero, the `.so`
    /// predates `dsv41_gemm_fp8_mrows` / `dsv41_swiglu_limit_q`, or any launcher
    /// declines. A PARTIAL attempt is harmless by construction: `moe_out` is
    /// untouched by everything above (the caller's `add_inplace` is what writes
    /// it) and every buffer this pass writes (`ex_act`, `xq`/`xsc`,
    /// `shared_out`) is either re-quantised or rewritten by the per-row loop.
    fn shared_expert_mrows(
        &self,
        w1: &DevTensor,
        w1s: &DevTensor,
        w3: &DevTensor,
        w3s: &DevTensor,
        w2: &DevTensor,
        w2s: &DevTensor,
    ) -> Result<bool> {
        let cfg = self.cfg;
        let (bs, dim, sh_il) = (self.bs, self.dim, self.sh_il);
        if bs == 0
            || bs > 8
            || (sh_il % 32) != 0
            || (dim % 32) != 0
            || !self.dev.supports_gemm_fp8_mrows()
            || !self.dev.supports_swiglu_q()
        {
            return Ok(false);
        }
        // `ex_act` is `bs * 2 * max(inter_local, sh_il)` f32, so the `[bs,
        // 2*sh_il]` block this pass stages always fits — unlike the routed
        // indirect arm, whose stride is `2*inter_local`. It is DEAD by this point
        // on every arm (the batched one stages in `ex_act_b`, the indirect one
        // consumed its rows into `moe_out`), which is what makes the reuse safe.
        debug_assert!(
            (1..=8).contains(&bs),
            "shared_expert_mrows: {bs} rows exceed the mrows dispatch"
        );
        // 1) w1 | w3 over the SAME staged activation, one weight-stationary GEMV
        //    each, all `bs` rows folded against it.
        let stride = (2 * sh_il) as i32;
        let act = self.ex_act.ptr as *mut f32;
        let ok1 = self.dev.gemm_fp8_mrows(
            self.xq.as_u8(),
            self.xsc.as_f32(),
            w1.as_u8(),
            w1s.as_u8(),
            std::ptr::null(),
            act,
            bs as i32,
            sh_il as i32,
            dim as i32,
            stride,
        )?;
        let ok3 = self.dev.gemm_fp8_mrows(
            self.xq.as_u8(),
            self.xsc.as_f32(),
            w3.as_u8(),
            w3s.as_u8(),
            std::ptr::null(),
            act.wrapping_add(sh_il),
            bs as i32,
            sh_il as i32,
            dim as i32,
            stride,
        )?;
        if !(ok1 && ok3) {
            return Ok(false);
        }
        // 2) swiglu + the fp8 pair the w2 GEMV reads, all `bs` rows in one launch.
        if !self.dev.swiglu_limit_q(
            act,
            bs as i32,
            sh_il as i32,
            cfg.swiglu_limit,
            self.xq.ptr as *mut u8,
            self.xsc.ptr as *mut f32,
        )? {
            return Ok(false);
        }
        // 3) w2 for all rows — w2 is Cols-sharded ([dim, sh_il] locally), so its
        //    reduction runs over this rank's slice and the OUTPUT stays a partial
        //    [bs, dim] block, exactly as the per-row arm's rows are.
        if !self.dev.gemm_fp8_mrows(
            self.xq.as_u8(),
            self.xsc.as_f32(),
            w2.as_u8(),
            w2s.as_u8(),
            std::ptr::null(),
            self.shared_out.ptr as *mut f32,
            bs as i32,
            dim as i32,
            sh_il as i32,
            dim as i32,
        )? {
            return Ok(false);
        }
        Ok(true)
    }

    /// The vocabulary slice the Markov head walks, or `None` for the
    /// full-vocabulary path: `Some((seg, base))` with `seg = vocab / world` and
    /// `base = rank * seg`.
    ///
    /// This is [`markov_sliced`]'s ONLY consumer and the one place the geometry is
    /// decided, so the head GEMV, the Markov bias and the packed key's global
    /// index base can never disagree (the pitch is not in the type — see the
    /// head site's warning, and the verify's twin `verify_head_geom`).
    ///
    /// Gated on the whole set of fallbacks BEFORE any layout choice:
    /// * `DSV41_MARKOV_SLICED` on (default OFF — the A/B arm),
    /// * `world > 1` (a single rank has nothing to slice against),
    /// * `vocab % world == 0` (16160 x 8 = 129280 divides),
    /// * a BF16 head — the sliced head GEMV is the bf16 multi-row kernel,
    /// * BOTH new symbols present, tested HERE so a stale .so keeps the
    ///   full-vocabulary pitch instead of leaving a half-sliced `logits` buffer
    ///   for the fallback to misread,
    /// * a v5 collective, because the per-step fold IS one round of that epoch
    ///   sequence,
    /// * room for one 8-byte key in the v5 slot — the fold entry's own decline
    ///   arm, checked here so a decline can never happen after the head has
    ///   already written a sliced pitch.
    fn markov_head_geom(&self) -> Option<(usize, usize)> {
        let world = self.world;
        let vocab = self.vocab;
        if world <= 1 || vocab % world != 0 {
            return None;
        }
        let head_bf16 = self
            .w
            .head
            .as_ref()
            .map(|h| h.dtype == "BF16")
            .unwrap_or(false);
        let sliced = markov_sliced()
            && head_bf16
            && self.dev.supports_dspark_markov_head_sliced()
            && self.dev.supports_argmax_key_pub()
            && self
                .comm
                .as_ref()
                .map(|c| c.uses_v5() && 8 <= c.bytes)
                .unwrap_or(false);
        if !sliced {
            return None;
        }
        Some((vocab / world, self.rank * (vocab / world)))
    }

    /// `forward_head`: the first block's collapse, the head, then the
    /// sequential Markov loop (`dsv41_dspark_markov_head`, one launch per step).
    fn draft_head(&mut self) -> Result<()> {
        let cfg = self.cfg;
        let (dim, bs, hc, vocab, mr) = (self.dim, self.bs, self.hc, self.vocab, self.mr);
        let norm = need(&self.w.dspark_norm, "mtp.last.norm.weight")?;
        let head = need(&self.w.head, "head.weight")?;
        let markov_embed = need(&self.w.markov_embed, "mtp.last.markov_head.embed.weight")?;
        let markov_head = need(&self.w.markov_head, "mtp.last.markov_head.head.weight")?;
        let confidence_proj = need(&self.w.confidence_proj, "mtp.last.confidence_head.proj.weight")?;

        // h = hc_pre(x, premix) then rmsnorm(dspark_norm)
        // The premix is the LAST BLOCK's returned pre_mix — its ffn_pre mix,
        // exactly what the official forward_head receives
        // (`h, pre_mix = layer(...); forward_head(h, pre_mix, ...)` — the same
        // convention the backbone's own final collapse uses, `h = layer.hc_pre(h,
        // pre_mix)`). The block loop leaves it in `pre_in`. (The intermediate
        // attn_pre variant was a misreading of the backbone's premix_slot(1);
        // the official source settles it.)
        self.dev.hc_collapse(
            self.h.ptr as *const f32,
            self.pre_in.ptr as *const f32,
            self.collapse.ptr as *mut f32,
            bs as i32,
            hc as i32,
            dim as i32,
        )?;
        self.dump_unit("collapse", self.collapse.ptr as *const f32, &[bs, dim]);
        self.dev.rmsnorm(
            self.collapse.ptr as *const f32,
            norm.as_f32(),
            self.normed.ptr as *mut f32,
            bs as i32,
            dim as i32,
            cfg.norm_eps,
        )?;
        // HEAD BF16 DOMAIN (draft-numerical-audit C#8; `DSV41_DRAFT_BF16_DOMAIN`,
        // default OFF): the official `forward_head` hands the head a `normed`
        // tensor that carries the MODEL dtype (bf16) and only then casts it to
        // f32 for `F.linear(x.float(), weight_fp32)` (model.py:1008-1017).
        // ferrite's rmsnorm output is f32, so `gemv_bf16`/`head_gemv_bf16_mrows`
        // here consume a ~1e-3-different activation right before a 129280-way
        // near-tie argmax. The in-place RN round-trip is exact (narrow+wide) and
        // runs BEFORE the dump so a captured `normed` unit shows the value the
        // head actually consumes. `normed` has no other reader (this fn only).
        if draft_bf16_domain() {
            let n = (bs * dim) as i64;
            self.dev.bf16_roundtrip(self.normed.ptr as *mut f32, n)?;
        }
        self.dump_unit("normed", self.normed.ptr as *const f32, &[bs, dim]);

        // logits = head @ normed — all `bs` rows in ONE multi-row launch.
        //
        // ⚠️ COST (before): the checkpoint's head is bf16 [vocab, dim] = 1.29 GiB,
        // and the chain keeps the ACTIVATION in f32 (casting it to bf16 costs ~3
        // bits on a 129280-way near-tie argmax — see chain_dev's head site), so
        // each row used to be a separate f32-activation GEMV and the head was
        // read `bs` times: ~5 x 350 us. That single term was most of the draft
        // budget (docs/agent/dspark-verify-perf-plan.md §4).
        //
        // `head_gemv_bf16_mrows` (dsv41_glue.cu, the P1 kernel the verify's
        // `step_rows` already uses) is the WEIGHT-STATIONARY form of the same
        // GEMV: one block decodes its weight tile ONCE and folds all `bs` rows
        // against it, so the head is read once instead of `bs` times (and the
        // row loop's `bs` launches become one).
        //
        // NUMERICS — row r of the multi-row launch is BIT-IDENTICAL to the
        // per-row `gemv_bf16` launch of row r: same ascending lane->c chain (C1),
        // same decode in the same expression position (C2), the same shuffle
        // reduction tree run once per row (C3), per-row independent accumulators
        // with no cross-row recombination (C4), no K-split/fold to reorder (C5),
        // `__fmaf_rn` pinned against fast-math reassociation (C6). The kernel
        // header carries the full argument; the layouts already match what it
        // wants — `normed` is [bs, dim] row-major (x, [m, k]) and `logits` is
        // [bs, vocab] row-major (out, [m, n]).
        //
        // Ok(false) = stale .so (no symbol) or `bs` outside the kernel's 1..=8
        // dispatch: keep the per-row loop, which computes the same values by the
        // same argument. The f32-activation dtype keeps it too (the multi-row
        // kernel is bf16-only).
        //
        // `DSV41_DRAFT_HEAD_FOLD=0` forces the per-row loop even when the folded
        // launch is available: the A/B for "does the folded kernel's K-order
        // difference cost the draft its top-1" (see `draft_head_fold`).
        //
        // DSV41_MARKOV_SLICED geometry. `geom = Some((seg, base))` means this rank
        // head-projects only ITS `seg = vocab / world` rows of `head.weight` and
        // biases only the matching rows of `markov_head`, so the slope of `logits`
        // becomes `seg`. Both halves MUST take the same `(seg, base)` — a head
        // sliced at one pitch with a Markov bias at another would bias logits row
        // `v` with weight row `v + k` and still produce a plausible token (the
        // pitch is not in the type). `markov_head_geom` is the one place that
        // decides, and it is read ONCE here, before the head writes its pitch.
        //
        // The slice is taken by OFFSETTING the replicated tensor (the same thing
        // the verify's sliced head does: `head.ptr() + rank * seg * dim * 2`), not
        // by re-sharding `weights.rs`. The gate defaults OFF, so the unsliced arm
        // must keep working on the SAME loaded weights — a Rows-sharded
        // `markov_head` is exactly the "indexed vocab of a 16160-row slice" 8x
        // out-of-bounds read weights.rs records. The bandwidth win does not need
        // the shard: the kernel only ever touches its own 15.8 MiB of rows.
        //
        // `logits` stays allocated at `bs * vocab` and only its PITCH changes, so
        // the layout never depends on `set_comm` (which runs after `new`) and a
        // captured/replayed buffer keeps a stable address.
        let geom = self.markov_head_geom();
        let lg_pitch = geom.map(|(seg, _)| seg).unwrap_or(vocab);
        let lg_base = geom.map(|(_, base)| base).unwrap_or(0);
        // Element offset of this rank's first head row (bf16 = 2 bytes).
        let head_off = lg_base * dim * 2;
        let head_ptr = (head.ptr() as *const u8).wrapping_add(head_off) as *const c_void;
        let head_rows = draft_head_fold()
            && if head.dtype.as_str() == "BF16" {
                self.dev.head_gemv_bf16_mrows(
                    head_ptr,
                    self.normed.ptr as *const f32,
                    self.logits.ptr as *mut f32,
                    bs as i32,
                    lg_pitch as i32,
                    dim as i32,
                )?
            } else {
                false
            };
        if !head_rows {
            for r in 0..bs {
                match head.dtype.as_str() {
                    "BF16" => self.dev.gemv_bf16(
                        head_ptr,
                        (self.normed.ptr as *const f32).wrapping_add(r * dim),
                        (self.logits.ptr as *mut f32).wrapping_add(r * lg_pitch),
                        lg_pitch as i32,
                        dim as i32,
                    )?,
                    // Unreachable with `geom == Some`: the geometry requires a
                    // bf16 head. The full-vocabulary f32 path is unchanged.
                    _ => self.dev.gemv_f32(
                        head.as_f32(),
                        (self.normed.ptr as *const f32).wrapping_add(r * dim),
                        (self.logits.ptr as *mut f32).wrapping_add(r * lg_pitch),
                        lg_pitch as i32,
                        dim as i32,
                    )?,
                }
            }
        }

        // The Markov loop. `ids[0]` is the backbone's token; each launch biases
        // `logits[step]`, samples `ids[step + 1]` and scores `confidence[step]`.
        //
        // The unit dump takes `logits_row0` BEFORE this loop: the Markov head
        // biases the rows in place, so after the loop `logits[0]` is the biased
        // row, not the head's raw output the reference records. Under the slice
        // the recorded row is this rank's `seg`-wide slice (the same shape change
        // `DSV41_VERIFY_HEAD_SLICED` makes to the verify's `logits_r`), so the
        // dump's second dim follows the pitch rather than the vocabulary.
        self.dump_unit("logits_row0", self.logits.ptr as *const f32, &[lg_pitch]);
        for step in 0..bs {
            if let Some((seg, base)) = geom {
                // This rank's Markov head rows start `base * mr` f32 into the
                // replicated [vocab, mr] tensor — the SAME partition the head GEMV
                // just used.
                let mk_head =
                    (markov_head.as_f32() as *const u8).wrapping_add(base * mr * 4) as *const f32;
                self.dev.dspark_markov_head_sliced(
                    (self.logits.ptr as *mut f32).wrapping_add(step * seg),
                    self.collapse.ptr as *const f32,
                    // `markov_embed` NOT offset: `er` is the GLOBAL token's row.
                    markov_embed.as_f32(),
                    mk_head,
                    confidence_proj.as_f32(),
                    self.ids.ptr as *mut i32,
                    self.confidence.ptr as *mut f32,
                    dim as i32,
                    seg as i32,
                    mr as i32,
                    step as i32,
                    base as i32,
                    self.mk_key.ptr as *mut u64,
                    self.mk_partial.ptr as *mut u64,
                    self.mk_ctr.ptr as *mut u32,
                )?;
                // ONE v5 epoch round per step (the steps are strictly sequential:
                // step s+1's input token IS step s's winner). Symmetric across the
                // ranks because every rank runs this same loop — see
                // `markov_sliced`'s epoch-footprint note.
                let c = self.comm.as_ref().ok_or_else(|| {
                    FerriteError::Config(
                        "dspark: markov_head_geom accepted the slice with no collective attached"
                            .into(),
                    )
                })?;
                let ok = self.dev.argmax_key_pub(
                    self.mk_key.ptr as *const u64,
                    (self.ids.ptr as *mut i32).wrapping_add(step + 1),
                    c.peer_slots_u64(),
                    c.peer_stamps_u32(),
                    c.epoch_dev(),
                    c.staging_dev() as *mut u64,
                    c.ready_local_dev(),
                    self.world as i32,
                    self.rank as i32,
                    c.bytes as i64,
                )?;
                if !ok {
                    // Unreachable by construction: `markov_head_geom` gated on the
                    // symbol AND on `8 <= c.bytes`, the entry's two decline arms. A
                    // decline would mean the two sides disagree about the slot, and
                    // the sliced kernel has ALREADY biased this rank's row — a
                    // silent fallback would then read a slice as if it were the
                    // whole vocabulary and emit a plausible wrong token.
                    return Err(FerriteError::Config(
                        "dsv41_argmax_key_pub declined a shape the host gated as legal".into(),
                    ));
                }
            } else {
                self.dev.dspark_markov_head(
                    self.logits.ptr as *mut f32,
                    self.collapse.ptr as *const f32,
                    markov_embed.as_f32(),
                    markov_head.as_f32(),
                    confidence_proj.as_f32(),
                    self.ids.ptr as *mut i32,
                    self.confidence.ptr as *mut f32,
                    dim as i32,
                    vocab as i32,
                    mr as i32,
                    step as i32,
                    self.mk_partial.ptr as *mut u64,
                    self.mk_ctr.ptr as *mut u32,
                )?;
            }
        }
        // the drafted block the sampler produced: `[t0, d1..d_bs]`, the
        // reference's `output_ids`.
        self.dump_unit_i32("ids", self.ids.ptr as *const i32, &[bs + 1]);
        Ok(())
    }

    /// Write the main stream's KV row for the draft block into its window ring
    /// at `pos % win` (dspark.rs::dspark_attention, the `mk` half).
    ///
    /// `slot_dev` selects the destination's derivation:
    ///
    /// * `false` — the historical HOST-computed address `window + (pos % win)*hd`
    ///   as a `cudaMemcpyAsync` node. Fine for a plain launch; a CUDA-graph
    ///   capture FREEZES that address, so every replay would append the row to
    ///   the SAME ring slot ("captured but not updated" — the exact failure
    ///   `dsv41_glue.cu`'s `ring_append_kernel` header records for the main
    ///   chain). This arm is therefore only ever taken OUTSIDE the recording.
    /// * `true` — `dsv41_ring_append`, which derives the slot from the DEVICE
    ///   counter [`Self::seed_slot_ptr`] inside the kernel, so the recorded node
    ///   is position-independent. Bit-identical to the memcpy arm: same `hd`
    ///   floats to `window + ((*ctr % win))*hd`, and `ensure_pos_dev` has put
    ///   exactly this call's `pos` into that counter.
    fn seed_window(&mut self, s: usize, pos: usize, slot_dev: bool) -> Result<()> {
        let cfg = self.cfg;
        let (dim, hd) = (self.dim, self.hd);
        let rd = cfg.rope_head_dim;
        let ld = &self.w.mtp[s];
        let wkv = need(&ld.wkv, "mtp.*.attn.wkv.weight")?;
        let wkv_s = need(&ld.wkv_scale, "mtp.*.attn.wkv.scale")?;
        let kv_norm = need(&ld.kv_norm, "mtp.*.attn.kv_norm.weight")?;

        self.quant1(self.main_x.ptr as *const f32, dim)?;
        self.dev.gemm_fp8_mx(
            self.xq.as_u8(),
            self.xsc.as_f32(),
            wkv.as_u8(),
            wkv_s.as_u8(),
            std::ptr::null(),
            self.mk.ptr as *mut f32,
            1,
            hd as i32,
            dim as i32,
        )?;
        self.dev.rmsnorm(
            self.mk.ptr as *const f32,
            kv_norm.as_f32(),
            self.mk.ptr as *mut f32,
            1,
            hd as i32,
            cfg.norm_eps,
        )?;
        self.rope_at(self.mk.ptr as *mut f32, 1, hd as i32, 1, pos as i32, false)?;
        debug_assert!(rd > 0);

        if slot_dev {
            // The counter holds `pos_dev - 1`; every in-graph caller passes the
            // anchor's predecessor, and the host shadow is the one that uploaded
            // it (see `seed_slot_ptr`).
            debug_assert_eq!(
                self.pos_dev,
                Some(pos as i32 + 1),
                "seed_window(slot_dev): the device ring counter is not at this call's position"
            );
            return self.dev.ring_append(
                self.window[s].ptr as *mut f32,
                self.mk.ptr as *const f32,
                self.seed_slot_ptr(),
                self.win as i32,
                hd as i32,
            );
        }

        let slot = pos % self.win;
        let dst = (self.window[s].ptr as *mut f32).wrapping_add(slot * hd);
        self.dev
            .memcpy_d2d(dst as *mut c_void, self.mk.ptr, hd * 4)
    }

    /// Forward RoPE for the `bs` draft queries: query `r` sits at
    /// `pos + 1 + r` — the NEXT position after the anchor, one row per step
    /// (sglang arbitration: `positions_2d = prefix_lens + arange(...)` with
    /// prefix_lens already counting the anchor, i.e. row r = anchor_pos + 1 + r.
    /// The historical `pos + bs + r` misread the host comment's `seqlen` as the
    /// block size — it is the SEQUENCE length; the two differ by bs-1 = 4
    /// positions, enough to scramble the whole attention).
    /// Every head of a query shares that position (the main chain's `step = 0`
    /// convention).
    fn rope_queries(&mut self, x: *mut f32, pos: usize) -> Result<()> {
        // P3a a4: one launch for all bs rows when the gate is on (see
        // [`Self::rope_mrows`] — bit-identical to the loop below).
        if self.rope_mrows(x, pos, false)? {
            return Ok(());
        }
        let (bs, nh) = (self.bs, self.nh);
        for r in 0..bs {
            self.rope_at(
                x.wrapping_add(r * nh * self.hd),
                nh as i32,
                self.hd as i32,
                0,
                pos as i32 + r as i32,
                false,
            )?;
        }
        Ok(())
    }

    /// P3a a4 (`DSV41_DRAFT_P3A`): the `bs` query rows (or inverse-query rows) of
    /// `x` as ONE `dsv41_apply_rope_mrows` launch — `x` is row 0's base, the rows
    /// sit `nh*hd` apart, each holding `nh` head rows of `hd`, row `r` roped at
    /// `pos + r`.
    ///
    /// Bit-identical to the per-row loop it replaces: the kernel body is
    /// `apply_rope_kernel`'s with the `r` loop moved inside
    /// (`dsv41_kernels.cu`:1980-1998), row `r` indexes the tables at
    /// `pos_rows[r]`, and that array holds `pos + r` — the identical integer the
    /// loop passed through `off` (see [`Self::ensure_pos_dev`]). Row `r` touches
    /// only `[r*row_stride + h*row_len, +row_len)`, so nothing is re-associated
    /// and no cross-row reduction exists. The `pos_rows` half lives in the base
    /// counter's own buffer, which is why the arm requires `pos_dev == pos`: the
    /// array is `pos + r` only while the uploaded base IS `pos`.
    ///
    /// `Ok(false)` = NOT performed (gate off / stale `.so` without the symbol /
    /// the device base is not this row block's anchor): the caller runs the
    /// per-row loop, which is what the parity target is.
    fn rope_mrows(&mut self, x: *mut f32, pos: usize, inverse: bool) -> Result<bool> {
        if !draft_p3a().rope_mrows || self.pos_dev != Some(pos as i32) {
            return Ok(false);
        }
        let (Some(cos), Some(sin)) = (self.cos, self.sin) else {
            return Ok(false);
        };
        let (bs, nh, hd) = (self.bs, self.nh, self.hd);
        let rd = self.cfg.rope_head_dim;
        self.dev.apply_rope_mrows(
            x,
            cos,
            sin,
            bs as i32,
            nh as i32,
            (nh * hd) as i32,
            hd as i32,
            rd as i32,
            (rd / 2) as i32,
            self.pos_rows_ptr(),
            inverse,
        )
    }

    /// The inverse RoPE over the attention output, same positions as
    /// [`Self::rope_queries`].
    fn rope_queries_inv(&mut self, x: *mut f32, pos: usize) -> Result<()> {
        // P3a a4: same fold on the inverse rope over the attention output.
        if self.rope_mrows(x, pos, true)? {
            return Ok(());
        }
        let (bs, nh) = (self.bs, self.nh);
        for r in 0..bs {
            self.rope_at(
                x.wrapping_add(r * nh * self.hd),
                nh as i32,
                self.hd as i32,
                0,
                pos as i32 + r as i32,
                true,
            )?;
        }
        Ok(())
    }

    /// Upload the RoPE base into the device counter `pos_base` — at most ONCE
    /// per draft step, instead of once per `rope_at` call.
    ///
    /// `pos_base` is the draft's own little counter: `apply_rope` derives every
    /// row's position from `*base * mul + off + row * step`, so the whole draft
    /// step can share ONE uploaded value as long as each call expresses its own
    /// position as an `off` relative to it (see [`Self::rope_at`]).
    ///
    /// ⚠️ The upload is a blocking H2D (`upload_bytes_at` is a `cudaMemcpy`), so
    /// it drains the stream. That is the expensive part, and it is why the count
    /// matters: the draft's rope sites used to upload once per CALL (36 blocking
    /// H2Ds per `draft_forward`, one per seed / query row / inverse-query row /
    /// KV row — docs/agent/dspark-verify-perf-plan.md §4 ranks it
    /// "0.25 + ... 每次强制流水线排空"), and they now share one.
    ///
    /// The host keeps the shadow `pos_dev` so a value that is already on the
    /// device is not re-uploaded: the pattern is seed(pos-1) -> queries(pos) ->
    /// inverse(pos) -> KV(pos), which is two uploads per draft step, and one per
    /// `note_ctx_rows` batch (whose rows step by one), not one per rope call.
    ///
    /// Capture note: `off` still rides in the launch parameters, so this removes
    /// the per-call H2Ds but does NOT by itself make the draft graph-capturable —
    /// that needs the base to advance on the DEVICE (a device pos counter feeding
    /// `*base` with `off == 0`). This change is the sizeable half of it (the
    /// drains), and it is what makes the remaining one a single 4B upload.
    fn ensure_pos_dev(&mut self, pos: i32) -> Result<()> {
        if self.pos_dev == Some(pos) {
            return Ok(());
        }
        // The kernel reads `*base` as an `int`, so slot 0 holds the INTEGER's
        // four bytes. Slots `1..=bs` hold `pos + r`: the `pos_rows` array the
        // multi-row rope reads (`DSV41_DRAFT_P3A`'s a4 — see
        // [`Self::rope_queries`]). Slot `bs + 1` is the window ring's append
        // position (`pos - 1`, see [`Self::seed_slot_ptr`]). All three are a
        // pure function of the same `pos`, so widening the upload removes the
        // per-call H2Ds without adding an upload (the array is never uploaded
        // twice, and the `rope_mrows` arm refuses any position whose base is
        // not `pos`).
        let mut buf = Vec::with_capacity((self.bs + 2) * 4);
        buf.extend_from_slice(&pos.to_le_bytes());
        for r in 0..self.bs {
            buf.extend_from_slice(&(pos + r as i32).to_le_bytes());
        }
        // The ring append's slot is `(anchor - 1) % win` — `draft_attention`
        // seeds the block's PREDECESSOR row (see [`Self::seed_window`]), so the
        // device counter has to carry `pos - 1`, not `pos`.
        buf.extend_from_slice(&(pos - 1).to_le_bytes());
        self.dev.upload_bytes_at(&self.pos_base, &buf)?;
        self.pos_dev = Some(pos);
        Ok(())
    }

    /// The DEVICE counter `dsv41_ring_append` derives its slot from: slot
    /// `bs + 1` of `pos_base`, holding `pos_dev - 1` (the position whose ring
    /// slot the seed append lands in). Only ever dereferenced by that kernel,
    /// i.e. on the [`Self::seed_window`] arm that passes `slot_dev = true`.
    ///
    /// Its contract is [`Self::ensure_pos_dev`]'s: the value is a pure function
    /// of `pos_dev`, so the host shadow that skips a redundant upload cannot
    /// desync it.
    fn seed_slot_ptr(&self) -> *const std::os::raw::c_int {
        (self.pos_base.ptr as *const std::os::raw::c_int).wrapping_add(self.bs + 1)
    }

    /// The `pos_rows` array [`Self::rope_queries`]'s multi-row form reads: slot
    /// `r` of the half of `pos_base` that follows the base integer. Valid only
    /// while `pos_dev == pos` (the values are `pos_dev + r`).
    fn pos_rows_ptr(&self) -> *const std::os::raw::c_int {
        (self.pos_base.ptr as *const std::os::raw::c_int).wrapping_add(1)
    }

    /// `apply_rope` for the draft: `pos` is the ABSOLUTE position of row 0 (the
    /// rows then step by `step`), and the device counter only supplies the
    /// subtraction-free part of it.
    ///
    /// The kernel computes `row r -> (*base) * 1 + off + r * step`, and
    /// `ensure_pos_dev(base)` has already put `pos_dev` into `*base`, so passing
    /// `off = pos - pos_dev` yields `pos_dev + (pos - pos_dev) + r * step =
    /// pos + r * step` — the exact integer position this call always asked for.
    /// The arithmetic is INTEGER (and `off` is a tiny signed delta), so no
    /// position value can move a bit: every rope call site keeps its absolute
    /// position semantics (`rope_at(x, rows, hd, step, pos, inverse)`), and the
    /// `mul` factor the old form carried is fixed at 1 exactly as every caller
    /// passed it.
    ///
    /// `base` in the doc above is the value passed to the KERNEL (`pos_base`),
    /// which is why the caller's absolute position is an `off` here.
    #[allow(clippy::too_many_arguments)]
    fn rope_at(
        &mut self,
        x: *mut f32,
        rows: i32,
        row_len: i32,
        step: i32,
        pos: i32,
        inverse: bool,
    ) -> Result<()> {
        let cfg = self.cfg;
        let rd = cfg.rope_head_dim;
        let pos_dev = self.pos_dev.ok_or_else(|| {
            FerriteError::Config(
                "dspark: rope_at before ensure_pos_dev (no RoPE base on the device)".into(),
            )
        })?;
        let cos = self.cos.unwrap();
        let sin = self.sin.unwrap();
        self.dev.apply_rope(
            x,
            cos,
            sin,
            rows,
            row_len,
            rd as i32,
            (rd / 2) as i32,
            self.pos_base.ptr as *const std::os::raw::c_int,
            1,
            pos - pos_dev,
            step,
            inverse,
        )
    }

    /// The draft block's window geometry for a block whose row 0 sits at `pos`:
    /// how many ring rows are LIVE, and the ring slot the OLDEST of them lives
    /// in. The rows are the positions `[pos - n, pos - 1]`, slot `q % win`.
    ///
    /// Off (`DSV41_SEED_ALIGN` unset) this is the historical `min(win, pos)` rows
    /// from slot 0, which is the same set as long as the ring has not turned
    /// over — see the wrap discussion below.
    ///
    /// # Why the count is `min(win - 1, pos)` and never `win`
    ///
    /// The ring has `win` slots and the block's own row 0 owns the slot
    /// `pos % win`. Once `pos >= win` every slot is live, so a window that
    /// EXCLUDES that slot can hold at most `win - 1` rows; the historical
    /// `min(win, pos)` copied `win` rows in SLOT order, which (a) handed the
    /// anchor's position a SECOND candidate KV — the block's row 0 already
    /// provides one, and the official mask allows exactly one KV per position —
    /// and (b) silently evicted the oldest live position (`pos - win + 1`, whose
    /// slot is `(pos % win) + 1`) in exchange for the stale `pos - win` row in
    /// the anchor's slot. Copying in POSITION order (the `s0` return) is what
    /// makes the set exact; the count alone is not enough.
    fn win_rows(&self, pos: usize) -> (usize, usize) {
        let win = self.win;
        if crate::dsv41::chain_dev::seed_align() {
            let n = pos.min(win.max(1) - 1);
            (n, (pos - n) % win.max(1))
        } else {
            (win.min(pos), 0)
        }
    }

    /// Rebuild `idxs` when the valid window length changes.
    ///
    /// `dspark_topk_idxs` (dspark.rs) is `[0, n_win) ++ [win, win + bs)` repeated
    /// over the batch and over every draft query; with `bs == 1` batch that is
    /// `bs` identical rows, so the whole matrix is built on the host and cached
    /// until `n_win` moves (it settles at `win` after the first `win` positions,
    /// so the steady-state decode path is H2D-free).
    fn ensure_idxs(&mut self, pos: usize) -> Result<()> {
        let bs = self.bs;
        // The SAME window geometry `draft_attention` copies with (see
        // `win_rows`): the candidate rows are [0, n_win) plus the block at
        // [n_win, n_win + bs), in exactly the order `all_kv` holds them.
        let (n_win, _s0) = self.win_rows(pos);
        if self.n_win_cached == n_win {
            return Ok(());
        }
        // COMPACT layout: the block's rows sit at [n_win, n_win+bs) of
        // `all_kv` (only the live window rows are copied), NOT at [win, ..) —
        // the historical `win + i` indexed the full-width layout while the
        // buffer was compact, reading past every window row once pos < win-1.
        let mut row: Vec<i32> = Vec::with_capacity(n_win + bs);
        row.extend((0..n_win).map(|i| i as i32));
        row.extend((0..bs).map(|i| (n_win + i) as i32));
        let row_bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut flat = Vec::with_capacity(row_bytes.len() * bs);
        for _ in 0..bs {
            flat.extend_from_slice(&row_bytes);
        }
        self.dev.upload_bytes_at(&self.idxs, &flat)?;
        self.n_win_cached = n_win;
        Ok(())
    }

    /// `Device::quant_fp8` of one contiguous row of `k` f32 into `xq`/`xsc`.
    fn quant1(&self, src: *const f32, k: usize) -> Result<()> {
        self.dev.quant_fp8(
            src,
            self.xq.ptr as *mut u8,
            self.xsc.ptr as *mut f32,
            1,
            k as i32,
            32,
            true,
        )
    }

    fn upload_i32(&self, dst: &DevBuf, v: &[i32]) -> Result<()> {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.dev.upload_bytes_at(dst, &bytes)
    }

    // ---- golden per-unit dump (`DSV41_DSPARK_UNIT_DUMP=1`) ----------------
    //
    // Every call below is a no-op unless a capture is live, i.e. one branch on
    // `self.unit` — the gate itself is resolved once, at `draft_forward`'s top.
    // A failing D2H is LOGGED, never propagated: a debug dump must not answer an
    // error for a forward whose numbers are otherwise fine.

    /// Record one `f32` unit, `dims` naming its shape.
    #[inline]
    fn dump_unit(&mut self, name: &str, ptr: *const f32, dims: &[usize]) {
        if let Some(u) = self.unit.as_mut() {
            if let Err(e) = u.push_f32(self.dev, name, ptr, dims) {
                eprintln!("[dspark] unit dump {name}: {e}");
            }
        }
    }

    /// [`Self::dump_unit`] for a per-block unit: `{prefix}{idx}`. The name is
    /// only formatted when a capture is live, so the gate-off path allocates
    /// nothing.
    #[inline]
    fn dump_unit_idx(&mut self, prefix: &str, idx: usize, ptr: *const f32, dims: &[usize]) {
        if let Some(u) = self.unit.as_mut() {
            let name = format!("{prefix}{idx}");
            if let Err(e) = u.push_f32(self.dev, &name, ptr, dims) {
                eprintln!("[dspark] unit dump {name}: {e}");
            }
        }
    }

    /// Record one `i32` unit (the Markov sampler's `ids`).
    #[inline]
    fn dump_unit_i32(&mut self, name: &str, ptr: *const i32, dims: &[usize]) {
        if let Some(u) = self.unit.as_mut() {
            if let Err(e) = u.push_i32(self.dev, name, ptr, dims) {
                eprintln!("[dspark] unit dump {name}: {e}");
            }
        }
    }
}

/// Release the `DSV41_DRAFT_GRAPH` exec with the draft.
///
/// The draft's buffers are owned by this struct and live exactly as long as it
/// does, so a captured graph is valid for the WHOLE instance — there is no
/// per-request drop (the reason `DevChain` drops its three graph stores per
/// request is that the chain REALLOCATES its scratch, and a graph bakes device
/// addresses). The only thing left to do is hand the exec back on teardown.
impl Drop for DsparkDev<'_> {
    fn drop(&mut self) {
        if let Some(e) = self.graph.take() {
            // Best-effort: a failing teardown must not panic a process that is
            // already shutting down.
            let _ = self.dev.graph_free(std::ptr::null_mut(), e);
        }
    }
}
