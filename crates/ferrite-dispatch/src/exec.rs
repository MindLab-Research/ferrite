//! Exec backend contract — the seam between scheduling and execution.
//!
//! The dispatcher owns *decisions* (what runs, in which bucket, with which
//! prefix resumes); the backend owns *effects* (graphs, streams, device
//! memory, NCCL groups). Everything crossing the seam is an owned
//! plan-struct — no shared mutable state, no callbacks — so:
//!
//! - the tick is a **pure function of state** (deterministic, replayable
//!   in simulation without a GPU: the whole dispatcher is unit-testable
//!   against a mock backend — the strongest property a scheduler can have
//!   and the reason the plan structs are plain data);
//! - the backend is **stateless across ticks** except captured graphs and
//!   device pools, which never appear in scheduling decisions (they are
//!   keyed by shape, not by request identity — see `graph.rs`);
//! - row identity is *physical*: plans reference registry decode rows,
//!   which are the `[B, ...]` state tensor rows the graphs address. No
//!   id translation anywhere on the hot path.
//!
//! ## Tick protocol (the whole API)
//!
//! ```text
//! loop {
//!     let tick = scheduler.plan()?;                    // pure decision
//!     match tick {
//!         Tick::Decode(d) => {
//!             backend.decode(&d)?;                     // draft+verify+commit
//!             let out = backend.readback(d.bucket.live())?;
//!             let digest = scheduler.ingest_decode(&d, &out.accepts)?; // commit math
//!         }
//!         Tick::Prefill(p) => {
//!             backend.prefill(&p)?;
//!             scheduler.ingest_prefill(&p)?;
//!         }
//!         Tick::Idle => break,
//!     }
//! }
//! ```
//!
//! One decode replay covers the full MTP step (draft → verify → commit in
//! the same graph family); the readback is a single pinned memcpy per tick
//! — the D-batch's host work is O(B) u32/u8 copies, not per-row syncs.
//!
//! ## The numeric-domain iron law
//!
//! Implementations MUST run draft and verify in the same numeric domain
//! the single-row path used (see `mtp.rs` module doc — four production
//! regressions trace to domain drift between the two). The trait cannot
//! enforce floats, so it enforces the *shape* of the evidence: readbacks
//! carry per-row accept counts, and the scheduler's digest tracks accept
//! rate per tick for the operator to alarm on.

use ferrite_types::Result;

use crate::bucket::BucketAssignment;
use crate::mtp::AcceptBatch;

/// What the scheduler asks the engine to run for one tick.
///
/// At most one decode + any number of prefill chunks per tick (prefill
/// chunks fill the verify-idle bubbles; the graph families are separate —
/// decode replays and prefill replays may interleave on different streams).
#[derive(Debug, Clone)]
pub enum Tick {
    /// One MTP decode step over the live bucket (draft + verify + commit
    /// are one replay family on device).
    Decode(DecodeWork),
    /// Prefill chunks (rows advanced from radix-resume points; the
    /// scheduler batched them to fill the token budget).
    Prefill(PrefillWork),
    /// Nothing runnable (all rows finished, queue empty).
    Idle,
}

/// Decode tick inputs — everything bound before the replay.
///
/// Owned data (no borrows): the plan crosses the scheduler/backend seam
/// by value. `verify_ids` is the packed `[3 * B]` MTP window (row-major
/// `[row][last, d0, d1]`, `PAD_TOKEN` in dead rows — see
/// `mtp::VerifyPlan::packed_input`); when the backend runs its draft head
/// *inside* the decode replay (the deployed shape: draft is a captured
/// graph too), the scheduler leaves `verify_ids` empty and the backend
/// binds from row state — the field exists for headless-draft debugging
/// (draft on host, verify on device).
#[derive(Debug, Clone)]
pub struct DecodeWork {
    /// Bucket assignment (shape + live rows + per-row seqs).
    pub bucket: BucketAssignment,
    /// Optional pre-packed MTP verify input `[3B]`.
    pub verify_ids: Vec<u32>,
    /// Per-row committed stream length (state cursor context; the commit
    /// math uses it, the graph does not).
    pub row_positions: Vec<usize>,
}

/// Prefill tick inputs — chunk graph replays, one per row, budget-sized.
///
/// Each row's chunk replays the n-generic prefill graph against its own
/// state row — the same row-identity contract decode uses. `resume_at`
/// is the radix-resumed prefix length (state was bound by
/// `StateRegistry::bind_row_to_snapshot` before the tick; the chunk
/// starts past it, never re-running a matched prefix).
#[derive(Debug, Clone, Default)]
pub struct PrefillWork {
    pub chunks: Vec<ChunkWork>,
}

/// One row's chunk of prompt ids to prefill.
#[derive(Debug, Clone)]
pub struct ChunkWork {
    /// Physical decode row (state row the chunk graph advances).
    pub row: u32,
    /// Owning sequence (scheduler bookkeeping).
    pub seq: crate::arena::SeqId,
    /// Prefix length already covered (radix resume + prior chunks).
    pub resume_at: usize,
    /// Prompt tokens `[resume_at .. resume_at + n)` — owned (the plan
    /// crosses the seam by value; the serve loop may bump-arena this).
    pub ids: Vec<u32>,
}

/// Post-replay readback: per live row accept data (k + accepted tokens +
/// bonus), already host-side (the backend did the single pinned D2H).
#[derive(Debug, Clone)]
pub struct DecodeReadback {
    pub accepts: AcceptBatch,
}

/// The execution backend — everything the dispatcher cannot decide alone.
///
/// Object-safe (plans are owned data, no generic methods): the serve
/// binary holds `Box<dyn ExecBackend>`.
pub trait ExecBackend {
    /// Warm the graph ladder (capture at startup; called once, then
    /// `GraphPool::is_warm` holds for the shapes it captured).
    fn warmup(&mut self) -> Result<()>;

    /// Run one decode tick (draft + verify + commit, single replay family).
    fn decode(&mut self, work: &DecodeWork) -> Result<()>;

    /// Single pinned readback after a decode tick (per-row k + bonus).
    fn readback(&mut self, live_rows: usize) -> Result<DecodeReadback>;

    /// Run prefill chunks (any number of rows; each replays the chunk
    /// graph against its own state row).
    fn prefill(&mut self, work: &PrefillWork) -> Result<()>;

    /// Grow a row's DSA pages by `n` (device pool alloc; the registry
    /// bookkeeping already counted them — this only performs the alloc).
    fn grow_pages(&mut self, row: u32, n: usize) -> Result<()>;

    /// Access the state-move capability (row snapshot for radix promotion,
    /// radix restore for admission resume, snapshot release for eviction).
    /// Split from `ExecBackend` so the scheduler can drive state moves
    /// without owning the whole backend (single-field borrows, no
    /// whole-self re-entrancy).
    fn state_moves(&mut self) -> &mut dyn StateStoreCap;
}

/// State-move capability slice of the backend.
///
/// All three operations go through `StateRegistry` bookkeeping (page
/// refcounts, slot domains) — the capability performs the device copies
/// and owns the registry handle. Implemented by the engine's backend as
/// a thin forward to its `StateRegistry`.
pub trait StateStoreCap {
    /// Deep-copy GDN state from a decode row into a fresh snapshot slot
    /// (returns the registry `StateId`; page lease transfer handled by
    /// the registry, this performs the device copy).
    fn snapshot_row(&mut self, row: u32, committed_tokens: usize) -> Result<crate::state::StateId>;

    /// Copy a radix snapshot's GDN state into a decode row (+ lease its
    /// pages — radix resume path).
    fn restore_row(&mut self, snap: crate::state::StateId, row: u32, tokens: usize) -> Result<()>;

    /// Release a snapshot the radix tree evicted (device slot + page refs).
    fn release_snapshot(&mut self, snap: crate::state::StateId) -> Result<()>;
}

/// Re-export for plan consumers (keeps `use` paths one-crate).
pub use crate::bucket::Bucket;

/// Compile-time sanity: plans are plain owned data (Send + no borrows),
/// the property that makes the dispatcher replayable in simulation.
trait _PlanData: Send {}
impl _PlanData for Tick {}
impl _PlanData for DecodeWork {}
impl _PlanData for PrefillWork {}
impl _PlanData for ChunkWork {}
impl _PlanData for DecodeReadback {}
