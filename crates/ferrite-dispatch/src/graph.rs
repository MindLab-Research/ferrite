//! Graph pool abstraction — capture once, replay per tick, shape-keyed.
//!
//! ## The pad-to-B graph contract
//!
//! The deployed engine (ferrite-exec `mega_chain_dev` + MTP `mega_v`)
//! captures one whole-decode-step CUDA graph per sequence today. The
//! batch path generalizes that to **one graph per ladder shape**: the
//! pool below owns `ladder.len()` graphs, each captured at shape `B`
//! with
//!
//! - input ids in a pinned `[3B]` slot (verify row-major, `PAD_TOKEN`
//!   in dead rows — state stride isolates rows, no masking math),
//! - state tensors `[B, ...]` addressed by row index — row `r` is
//!   registry decode row `r` (the identity that makes replay a pure
//!   idiom: no indirection, no per-tick pointer rewrite),
//! - output `[B]` u8 accept counts + `[B]` u32 bonus tokens pinned,
//!   one D2H memcpy per tick.
//!
//! Capture is expensive (whole-chain stream capture); replay is a single
//! `graphLaunch` per step. The pool's only job is **shape selection and
//! input binding** — everything else is the exec backend's captured
//! state.
//!
//! ## Prefill graphs
//!
//! The same pool hosts the chunked-prefill graphs (shape-keyed by chunk
//! size — `PREFILL_CHUNKS` ladder, default `[512]` growing to 4096 as
//! the kernel chain gains the n-generic prefill path). A prefill chunk
//! replay binds `[chunk]` ids + the target row's state and runs the
//! same kernels the verify path runs — the n-generic parameterization
//! already in the device chain (90% of kernels take an n).
//!
//! ## Ownership discipline
//!
//! Graphs are captured against the *physical* state layout; the pool
//! must outlive every replay. `GraphHandle` is a capability token (not
//! droppable state): the exec backend mints handles at capture and the
//! scheduler references shapes (`Bucket`), never raw handles — the
//! pool resolves shape → graph internally. This keeps a re-capture
//! (e.g. ladder extension at runtime) invisible to scheduling logic.

use ferrite_types::{FerriteError, Result};

use crate::bucket::{Bucket, BUCKET_LADDER};

/// Prefill chunk ladder (token positions per prefill replay). Single
/// entry today (512); the field is a `const` array so extending it is a
/// one-line change with compile-time enforcement that every kernel in the
/// prefill chain is n-generic.
pub const PREFILL_CHUNKS: [usize; 1] = [512];

/// A captured graph in the pool — capability token minted by the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphKind {
    /// MTP decode step at bucket shape `B` (draft + verify + commit).
    DecodeMtp(Bucket),
    /// Plain (non-MTP) decode step at bucket shape `B`.
    DecodePlain(Bucket),
    /// Prefill chunk of `tokens` positions (row = target decode row).
    PrefillChunk(usize),
}

/// The pool: shape → graph resolution + capture orchestration.
///
/// Backends implement [`GraphBackend`]; the pool is the bookkeeping half
/// that lives on the scheduler side (which shapes exist, which are warm).
#[derive(Debug, Default)]
pub struct GraphPool {
    /// Captured shapes (kind → warm status). A cold shape must be
    /// captured before first replay — the scheduler's admission of new
    /// rows is gated on shape warmth (or triggers capture synchronously).
    warm: Vec<GraphKind>,
}

impl GraphPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a captured graph (backend calls after successful capture).
    pub fn mark_warm(&mut self, kind: GraphKind) {
        if !self.warm.contains(&kind) {
            self.warm.push(kind);
        }
    }

    /// Drop a shape (re-capture required — e.g. after device reset).
    pub fn mark_cold(&mut self, kind: GraphKind) {
        self.warm.retain(|k| *k != kind);
    }

    pub fn is_warm(&self, kind: GraphKind) -> bool {
        self.warm.contains(&kind)
    }

    /// All decode shapes warm (standard admission gate for full-throughput
    /// mode; a cold ladder head is a startup-only condition).
    pub fn ladder_warm(&self) -> bool {
        BUCKET_LADDER
            .iter()
            .all(|&b| self.warm.contains(&GraphKind::DecodeMtp(b.try_into().ok().unwrap_or(Bucket::B1))))
    }

    /// The tick's replay shape for `live` rows: smallest warm shape that
    /// covers them. If the exact-cover shape is cold, fall back up the
    /// ladder to the next warm one (pad waste, never a stall).
    pub fn replay_shape(&self, live: u32) -> Result<Bucket> {
        let want = Bucket::cover(live)?;
        // ascending walk from `want`
        for &b in BUCKET_LADDER.iter().skip(want.ladder_idx()) {
            let shape = Bucket::try_from(b)?;
            if self.is_warm(GraphKind::DecodeMtp(shape)) {
                return Ok(shape);
            }
        }
        Err(FerriteError::Scheduler(format!(
            "graph pool: no warm decode shape covers {live} rows"
        )))
    }
}

/// Backend capability: capture and replay graphs of the ladder shapes.
///
/// The iron-law contract (from the MTP domain): every captured graph must
/// produce **bit-identical per-row numerics** regardless of bucket shape,
/// row occupancy, or padding content. Shape changes the *stride*, not the
/// math — backends assert this at capture time (a row at B=4 replays the
/// same kernels it would at B=1; TP all-reduce order is shape-stable
/// NCCL ring, embeddings identical).
pub trait GraphBackend {
    /// Capture a graph kind. Slow path (seconds); called only at startup
    /// and ladder extension. The graph is bound to the backend's own
    /// pinned slots — replays overwrite them in place.
    fn capture(&mut self, kind: GraphKind) -> Result<()>;

    /// Bind inputs for a decode replay: ids `[3B]` (MTP) or `[B]`
    /// (plain), row state pointers already resident (row = registry row).
    fn bind_decode(&mut self, shape: Bucket, ids: &[u32]) -> Result<()>;

    /// Replay a decode step. Returns pinned outputs for host ingest
    /// (per-row accept counts + bonus tokens) — the backend may leave
    /// them on device with a later `readback` batch.
    fn replay_decode(&mut self, shape: Bucket) -> Result<()>;

    /// D2H readback of the last replay's outputs: `[B]` accept `k`,
    /// `[B]` bonus tokens (MTP), or `[B]` next tokens (plain).
    fn readback(&mut self, shape: Bucket, k: &mut [u8], bonus: &mut [u32]) -> Result<()>;

    /// Bind + replay a prefill chunk for one row: `ids[chunk]` tokens
    /// advancing the row's state (the graph addresses state by row
    /// index — same contract as decode).
    fn replay_prefill(&mut self, chunk: usize, row: u32, ids: &[u32]) -> Result<()>;
}

// ----------------------------------------------------------------------------
// capture-shape expansion helper: the pool documents (and the backend
// macro-expands) the full ladder at startup.
// ----------------------------------------------------------------------------

/// The full capture set the engine warms at startup: decode MTP × ladder
/// + prefill chunks. Order matters for memory: ladder ascending so early
/// captures (small) land while free VRAM still allows large shapes.
pub fn warmup_plan() -> Vec<GraphKind> {
    let mut plan = Vec::with_capacity(BUCKET_LADDER.len() + PREFILL_CHUNKS.len());
    for &b in BUCKET_LADDER.iter() {
        if let Ok(shape) = Bucket::try_from(b) {
            plan.push(GraphKind::DecodeMtp(shape));
        }
    }
    for &c in PREFILL_CHUNKS.iter() {
        plan.push(GraphKind::PrefillChunk(c));
    }
    plan
}
