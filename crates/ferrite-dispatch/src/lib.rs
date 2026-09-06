//! # ferrite-dispatch — the high-throughput scheduling domain.
//!
//! Scheduling for the ferrite engine (GLM-5.3-Flash decode): continuous
//! batching over pad-to-B CUDA graph buckets, a radix prefix cache over
//! the hybrid GDN/DSA state, a three-tier hicache (GPU → RAM → NVMe), and
//! the MTP speculative protocol generalized to B rows — the pieces a
//! production engine needs to serve concurrent requests on one TP4 box
//! without giving up single-request latency.
//!
//! ## The architecture in one screen
//!
//! ```text
//!                ┌──────────────────────────────────────────────────┐
//!                │                BatchScheduler (batch.rs)            │
//!                │  tick = plan (pure) → backend.replay → ingest        │
//!                │  admission ─ prefill chunks ─ decode bucket ─ retire │
//!                └──────┬──────────────┬──────────────┬──────────────┘
//!        ┌─────────────▼──┐   ┌───────▼────────┐   ┌▼─────────────────┐
//!        │ RadixCache      │   │ GraphPool       │   │ StateRegistry    │
//!        │ (radix.rs)      │   │ (graph.rs)      │   │ (state.rs)      │
//!        │ token-block tree│   │ ladder {1..32}  │   │ rows+snapshots  │
//!        │ lock_ref pins   │   │ warm/cold shapes│   │ 3-tier hicache  │
//!        │ LRU evict       │   └───────┬────────┘   │ PageRegistry    │
//!        │ evict_for_pages │           │            └───┬────────────┘
//!        └──────┬─────────┘           │                │
//!               │  victims (node,state)                │ tier moves
//!               └──────────► demote chain ◄────────────┘
//!                  device → host → disk (prefix retained)
//!                  drop only when every tier is full
//! ```
//!
//! ## Design tenets
//!
//! **1. Plans are plain data.** `plan_into` emits a [`batch::TickPlan`]
//! the serve loop hands to [`exec::ExecBackend`] — the scheduler never
//! touches the engine, so a mock backend replays any schedule
//! deterministically (the scheduler is unit-testable without a GPU).
//!
//! **2. Row identity is physical.** Decode row `r` *is* state slot `r`
//! *is* graph row `r`. Pad-to-B bucket graphs bake row-indexed state
//! pointers at capture; the scheduler's one compaction rule (highest
//! live row moves into the retired hole, between replays only) keeps the
//! live-row domain a dense prefix `[0, live)` — no id translation
//! anywhere on the hot path.
//!
//! **3. The hicache is the cache, not a bolt-on.** A prefix snapshot's
//! tier is a field of its registry record: device (hot — prefix hits
//! restore with one D2D copy), host (RAM reservoir), disk (NVMe
//! archive). Page-budget pressure demotes **before it drops**
//! (device → host → disk, the tree intact — demotion never evicts);
//! the tree drops a leaf only when every tier refuses it. SGLang's
//! `HiMambaRadixCache` layers the same hierarchy, but as value-field
//! patches on tree nodes with async write-back — their eviction/loadback
//! races are the class this design deletes at the root: state moves are
//! **synchronous registry transitions** (no in-flight promises), the
//! tree pins (`lock_ref`) precede every copy, and one `PageRegistry`
//! owns every refcount.
//!
//! **4. Eviction is page-budget-driven.** Admission pays pages (the
//! unique suffix of a radix hit), not node counts; reclaim walks the
//! tier chain until the budget clears — the same currency vLLM/SGLang
//! charge (`num_gpu_blocks`), with the demote-first refusal to lose
//! cache while RAM/NVMe remain.
//!
//! **5. MTP is a row-batch, not a special case.** Verify is `[B][3]`
//! tokens in one replay (draft + target in one graph family), accept
//! is per-row `k + bonus`, commit math is a pure function
//! (`mtp::commit_row`), and the accepted prefix promotes into the radix
//! tree page-aligned — the speculative branch shares state with every
//! other request of the same prefix. The iron law lives in the trait
//! contract: draft/verify numerics are shape-invariant (see
//! [`exec`] docs).
//!
//! **6. Nothing allocates on the tick path.** Plans reuse caller
//! buffers; arenas own every long-lived object; handles are
//! generational (stale ids are hard misses, never dangling). The tick
//! is O(live rows + admissions + evictions), not O(system size).
//!
//! ## Module map
//!
//! | module | responsibility |
//! |---|---|
//! | [`arena`] | typed generational arenas — `SeqId`/`NodeId`/`StateId` handles |
//! | [`radix`] | token-block radix tree: lock_ref pins, LRU, page-budget evict |
//! | [`state`] | two-domain slot registry + 3-tier hicache + page refcounts |
//! | [`bucket`] | pad-to-B ladder {1,2,4,8,16,32}, `BatchDim` const generics |
//! | [`mtp`] | speculative protocol × B rows: draft/verify/accept/commit |
//! | [`graph`] | graph pool warm/cold shape bookkeeping + backend trait |
//! | [`exec`] | `ExecBackend`/`StateStoreCap` — the plan/effect seam |
//! | [`batch`] | the scheduler: admission, interleave, ingest, compaction |
//!
//! ## The tick protocol (serve loop)
//!
//! ```text
//! loop {
//!     let mut plan = TickPlan::default();
//!     scheduler.plan_into(&mut plan)?;            // pure decision
//!     match tick_of(&plan) {
//!         Decode => {
//!             backend.decode(work)?;               // draft+verify+commit replay
//!             let out = backend.readback(live)?;    // one pinned D2H
//!             let digest = scheduler.ingest_decode(&plan, &out.accepts)?;
//!         }
//!         Prefill => {
//!             backend.prefill(work)?;
//!             scheduler.ingest_prefill(&plan)?;    // promote + phase flip
//!         }
//!         Idle => break,
//!     }
//! }
//! ```
//!
//! ## Numerical-domain iron law (inherited from the engine's history)
//!
//! Four production regressions trace to draft/verify domain drift (AR
//! order, quantization, table dtype): accept rate is the canary, a
//! -0.2 drop costs ~8% throughput and dwarfs every kernel win. This
//! crate enforces the *structural* side — per-row math is row-isolated
//! by state stride, padding rows are inert, plans carry per-row accept
//! data for telemetry — and documents the trait requirement
//! ([`exec::ExecBackend`]) that backends keep the single-row numeric
//! domain under batching.
//!
//! ## Deliberately NOT here
//!
//! - Tokenizer/server I/O (the serve binary owns endpoints).
//! - Numerics (the engine executes; plans never touch tensor values).
//! - Time (no timers, no SLO algebra yet — the interleave policy is
//!   fixed decode-first/prefill-piggyback; replacing it means adding a
//!   policy struct here, not threading clocks through the crate).

pub mod arena;
pub mod batch;
pub mod bucket;
pub mod exec;
pub mod graph;
pub mod mtp;
pub mod radix;
pub mod state;

/// The prelude: one import for serve-loop integration.
pub mod prelude {
    pub use crate::arena::{NodeId, SeqId, TypedArena};
    pub use crate::batch::{
        Admission, BatchScheduler, CacheStats, PrefillMode, Reclaim, SchedConfig, TickDigest,
        TickPlan,
    };
    pub use crate::bucket::{BatchDim, Bucket, BucketAssignment, BUCKET_LADDER, PAD_TOKEN};
    pub use crate::exec::{
        ChunkWork, DecodeReadback, DecodeWork, ExecBackend, PrefillWork, StateStoreCap, Tick,
    };
    pub use crate::graph::{GraphKind, GraphPool};
    pub use crate::mtp::{
        AcceptBatch, AcceptRow, CandidatePolicy, DRAFT_DEPTH, DraftPlan, VerifyPlan, VERIFY_WIDTH,
    };
    pub use crate::radix::{EvictVictim, PrefixMatch, RadixCache, RadixNode};
    pub use crate::state::{HostStateStore, PhysStateStore, StateId, StateRegistry, Tier};
}

/// Crate version aligned with the workspace.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
