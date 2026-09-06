//! MTP (multi-token prediction) × batch protocol.
//!
//! ## The protocol, end to end
//!
//! One speculative tick over a bucket of B live rows:
//!
//! 1. **Draft** — the draft head (`nextn` module) runs per row, depth `d=2`,
//!    one token each: `draft[r][0..d]`. Draft state is row-local scratch —
//!    the draft chain is never part of the committed prefix, so it lives in
//!    the graph's per-row scratch, not the radix-visible state.
//! 2. **Verify** — the target model scores `[B][d+1]` token positions in a
//!    single replay: row `r`'s input row is
//!    `[last_committed, draft[r][0], draft[r][1]]`, the graph computes
//!    logits for all `3B` positions, argmax × 3B on device. Positions are
//!    row-major so the pinned input slot is one contiguous `[3B]` write.
//! 3. **Accept** — greedy spec-accept per row on device
//!    (`ferrite_mtp_commit` is the existing single-kernel form, batched):
//!    `k[r] = run length of argmax agreeing with the draft chain`. The
//!    scheduler only reads the per-row `k[r]` (u8 pinned D2H, one memcpy).
//!    Accepted tokens *and* the first rejected argmax token (which the
//!    target itself generated) commit into the row's stream: a verify step
//!    yields `k[r] ∈ [0, 1, 2]` draft-agree tokens **plus one bonus
//!    token** from the target — the standard MTP accounting
//!    (`1 + k` tokens per step per row).
//! 4. **Commit** — per row: append `k[r]+1` tokens to the row's stream,
//!    advance the DSA page cursor (page-aligned prefix blocks may now be
//!    promotable into the radix tree — see `RadixCommit`), and if the row's
//!    live tail crossed a page boundary, freeze a radix branch so future
//!    requests (and *this* request's own verify branches — see below) can
//!    share it.
//!
//! ## Why MTP loves the radix tree
//!
//! The verify step's `[last, d0, d1]` positions are *three candidate
//! branches* of the same committed prefix: the accepted one continues the
//! row, and the rejected candidates are exactly the "cheap beam" a
//! cache-aware decider wants already resident — their DSA pages are
//! written by the verify replay regardless. In this v1 protocol the
//! rejected-candidate pages are reclaimed (refcount drop) rather than
//! retained: candidate tokens land in the radix tree only when the page
//! they sit in was *completed* and committed, which the accepted branch
//! alone determines. (Retention of rejected candidates as speculative
//! branches is a policy knob — `CandidatePolicy` — left plugged but
//! off by default: it costs snapshot slots and only pays off under
//! beam search / best-of-N sampling, which the greedy path never hits.)
//!
//! ## Accept-rate invariance (the iron law)
//!
//! Domain rule from four production regressions: any numerical-domain
//! change between draft and verify (AR order, quantization, embedding
//! table dtype) shifts argmax ties and costs accept rate, which dominates
//! every kernel win. This protocol therefore pins the *numeric* contract:
//! `ExecBackend::verify_batch` must run draft and verify in the same
//! numeric domain the single-row path used — the trait documents this as
//! a hard requirement, and nothing in this crate (padding rows, row
//! order, graph shape) may alter per-row math. Padding rows are isolated
//! by state stride and never cross-contaminate live rows.

use ferrite_types::{FerriteError, Result};

use crate::arena::SeqId;
use crate::bucket::BucketAssignment;

/// Speculative-depth knob (draft positions per row per tick).
pub const DRAFT_DEPTH: usize = 2;
/// Verify positions per row: `[last, d0, d1]`.
pub const VERIFY_WIDTH: usize = DRAFT_DEPTH + 1;

/// Per-row accept result of one verify replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcceptRow {
    /// Draft tokens accepted (0..=DRAFT_DEPTH).
    pub k: u8,
    /// The bonus token (target argmax at the last accepted position) —
    /// committed together with the k accepted draft tokens.
    pub bonus: u32,
    /// Draft tokens *as written by the draft head* (for radix promotion
    /// the caller needs the exact accepted prefix: `draft[0..k] + bonus`).
    pub accepted: [u32; DRAFT_DEPTH],
    /// Row finished (EOS or max-len) during this commit.
    pub finished: bool,
}

impl AcceptRow {
    pub fn committed_tokens(&self) -> usize {
        self.k as usize + 1
    }
}

/// Policy for rejected verify candidates (plugged, default off).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidatePolicy {
    /// Reclaim rejected-candidate pages at refcount drop (default) —
    /// greedy MTP never revisits them.
    Reclaim,
    /// Retain rejected candidates as radix branches (beam / best-of-N).
    /// Costs snapshot slots + tree nodes; pays off only when sampling.
    Retain,
}

impl Default for CandidatePolicy {
    fn default() -> Self {
        CandidatePolicy::Reclaim
    }
}

/// One MTP tick's device-facing work (what the exec backend must run).
///
/// All buffers are *described*, not owned: the exec backend binds them to
/// its captured graphs' pinned memory. Row-major `[B][VERIFY_WIDTH]`
/// layout keeps the pinned input one contiguous 3B-u32 write and the
/// argmax output one 3B-u8 read — no scatter/gather in the hot path.
#[derive(Debug, Clone)]
pub struct VerifyPlan {
    /// The bucket assignment this tick runs over (shape + live rows).
    pub bucket: BucketAssignment,
    /// Per live row: committed token to verify from (input position 0
    /// of the row's 3-slot window) — the scheduler reads it from the
    /// row's stream tail (post-commit of the previous tick).
    pub last_tokens: Vec<u32>,
    /// Draft tokens per live row `[r][d]` (draft head output, same
    /// numeric domain as verify — iron law).
    pub drafts: Vec<[u32; DRAFT_DEPTH]>,
    /// Per-row EOS check closure input (the scheduler supplies stream
    /// state; the commit math is here, stream mutation in `mtp.rs`).
    pub candidate_policy: CandidatePolicy,
}

impl VerifyPlan {
    /// Pack the verify input: row-major `[B][3]`, PAD_TOKEN for pad rows.
    /// This is the exact content the pinned `[3B]` id slot receives.
    pub fn packed_input(&self) -> Vec<u32> {
        let b = self.bucket.shape.rows() as usize;
        let mut ids = vec![crate::bucket::PAD_TOKEN; b * VERIFY_WIDTH];
        for (r, seq_window) in self.drafts.iter().enumerate() {
            let base = r * VERIFY_WIDTH;
            ids[base] = self.last_tokens.get(r).copied().unwrap_or(crate::bucket::PAD_TOKEN);
            ids[base + 1] = seq_window[0];
            ids[base + 2] = seq_window[1];
        }
        ids
    }

    pub fn live(&self) -> usize {
        self.bucket.live() as usize
    }
}

/// Post-verify commit math per row — pure function over the accept data.
///
/// Returns the token stream append (`accepted[0..k] + [bonus]`) and the
/// radix-promotion pages if the row's committed length crossed a page
/// boundary (caller performs tree insertion with `RadixCache`).
///
/// This is deliberately side-effect-free and unit-testable in isolation:
/// the numeric accept/commit logic is where correctness bugs hide, and
/// the engine integration (device reads of k/bonus) is a thin adapter.
pub fn commit_row(
    row_tokens: &mut Vec<u32>,
    committed_len: usize,
    accept: &AcceptRow,
    eos: u32,
    max_new_tokens: usize,
    page_size: usize,
) -> Result<RowCommit> {
    let append = accept.committed_tokens();
    for i in 0..accept.k as usize {
        row_tokens.push(accept.accepted[i]);
    }
    row_tokens.push(accept.bonus);
    let new_len = committed_len + append;
    // page-boundary promotion: whole pages only (radix granularity).
    // `committed_len` may already be mid-page (prefill wrote it);
    // promotion counts new full pages completed by this commit.
    let page = new_len - (new_len % page_size);
    let was = committed_len - (committed_len % page_size);
    let promoted_pages = page.saturating_sub(was);
    let finished = accept.bonus == eos || row_tokens.len() >= max_new_tokens;
    Ok(RowCommit { new_len, promoted_pages, finished, accepted_page_tokens: promoted_pages })
}

/// One row's commit outcome (drives scheduler state transitions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowCommit {
    /// Stream length after the commit (committed prefix length).
    pub new_len: usize,
    /// New full pages promotable into the radix tree this commit.
    pub promoted_pages: usize,
    /// EOS/max-len reached — retire the row after this tick.
    pub finished: bool,
    /// Alias of `promoted_pages` (tokens of promotion == pages × granularity).
    pub accepted_page_tokens: usize,
}

/// Verify-result ingest: fold `k[r]`/bonus into rows.
///
/// The scheduler calls this with the D2H'd accept rows; it performs no
/// device work. `eos`/`max_new_tokens` are per-request (closure inputs
/// the scheduler owns).
#[derive(Debug, Clone)]
pub struct AcceptBatch {
    pub accepts: Vec<AcceptRow>,
}

impl AcceptBatch {
    pub fn total_tokens(&self) -> usize {
        self.accepts.iter().map(|a| a.committed_tokens()).sum()
    }
}

/// Draft-step description (per row, before the verify replay).
/// The draft head runs inside the same tick as verify (draft is cheap:
/// 1.55 ms vs 21.7 ms verify at B=1; at B=32 both scale sub-linearly
/// in the same graph family).
#[derive(Debug, Clone)]
pub struct DraftPlan {
    pub bucket: BucketAssignment,
    /// Per live row: the committed token to draft from (same value as
    /// `VerifyPlan::last_tokens` — kept explicit for the draft graph's
    /// pinned input slot).
    pub last_tokens: Vec<u32>,
}

impl DraftPlan {
    pub fn packed_input(&self) -> Vec<u32> {
        let b = self.bucket.shape.rows() as usize;
        let mut ids = vec![crate::bucket::PAD_TOKEN; b];
        for (r, &t) in self.last_tokens.iter().enumerate() {
            ids[r] = t;
        }
        ids
    }
}

/// Error minting helper (kept local: mtp module never invents its own
/// error kind — it routes through ferrite-types).
fn err(msg: impl Into<String>) -> FerriteError {
    FerriteError::Scheduler(msg.into())
}

/// Sanity check an accept row against the draft plan (debug guard for
/// backend adapters: k must be within draft width, rows must match).
pub fn validate_accept(plan: &VerifyPlan, batch: &AcceptBatch) -> Result<()> {
    if batch.accepts.len() != plan.live() {
        return Err(err(format!(
            "accept len {} != live rows {}",
            batch.accepts.len(),
            plan.live()
        )));
    }
    for (i, a) in batch.accepts.iter().enumerate() {
        if a.k as usize > DRAFT_DEPTH {
            return Err(err(format!("row {i}: k={} > depth {DRAFT_DEPTH}", a.k)));
        }
    }
    Ok(())
}

// SeqId re-export for doc cross-refs (scheduler-facing surface).
#[allow(dead_code)]
fn _seqid_in_scope(_: SeqId) {}
