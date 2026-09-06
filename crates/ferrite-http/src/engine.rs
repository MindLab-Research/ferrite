//! Engine boundary types — the contract between HTTP handlers and the
//! driver loop.
//!
//! The HTTP side (axum, async) and the engine side (one thread, owns the
//! `BatchScheduler` — single-writer by design, see `driver.rs`) talk in
//! plain messages both directions:
//!
//! ```text
//!   HTTP (async)                     driver thread (owns scheduler)
//!   ─────────── DriverCmd::Submit ──────────▶  scheduler.submit()
//!   ─────────── DriverCmd::Cancel ─────────▶  scheduler.cancel()
//!   ◀────────── ReqEvent stream ─────────────  tick ingest (per request)
//! ```
//!
//! Events are the OpenAI-SSE vocabulary directly (token deltas, finish
//! reasons, usage with radix-hit accounting) — no tokenization here,
//! no HTTP framing there; `api.rs` renders events, `driver.rs` mints
//! them.

use std::fmt;

use ferrite_dispatch::arena::SeqId;

/// Driver-side request number (HTTP layer's correlation id; distinct
/// from the scheduler's `SeqId`, which the driver owns privately).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReqId(pub u64);

impl fmt::Display for ReqId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "req-{}", self.0)
    }
}

/// A submitted generation request (token-domain: the HTTP layer already
/// rendered the chat template and encoded the prompt).
#[derive(Debug, Clone)]
pub struct RequestSpec {
    /// Rendered + encoded prompt (chat format applied upstream).
    pub prompt_ids: Vec<u32>,
    /// OpenAI `max_tokens` (completion cap → scheduler max_new_tokens).
    pub max_new_tokens: usize,
    /// Stop token ids (tokenizer specials; the engine also honors EOS).
    pub stop_ids: Vec<u32>,
}

/// Why a request finished (`finish_reason` on the wire).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// A stop token (or EOS) was generated.
    Stop,
    /// `max_tokens` reached.
    Length,
    /// Client disconnected / API abort.
    Cancelled,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::Cancelled => "cancelled",
        }
    }
}

/// Usage accounting (the OpenAI `usage` object + our radix telemetry).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// Prompt tokens served from the radix prefix cache (prefix hit at
    /// admission — zero prefill cost; reported as
    /// `prompt_tokens_details.cached_tokens` on the wire).
    pub cached_tokens: usize,
}

/// Per-request event stream (driver → HTTP).
///
/// The driver batches per tick (one event per accept window — 1–3
/// tokens per MTP step, matching the engine's commit granularity; the
/// SSE layer renders each as one content delta).
#[derive(Debug, Clone)]
pub enum ReqEvent {
    /// Admitted (prefill start or radix resume); the prefix-hit report is
    /// the cache telemetry (first event on the stream).
    Admitted { prefix_hit: usize, prompt_tokens: usize },
    /// Generated token ids committed this tick (post-accept).
    Tokens { ids: Vec<u32> },
    /// Terminal event (always the last on the stream).
    Finished { reason: FinishReason, usage: Usage },
}

/// Commands (HTTP → driver). Owned by the mpsc queue; Submit carries the
/// response channel the HTTP side streams from.
#[derive(Debug)]
pub enum DriverCmd {
    Submit {
        spec: RequestSpec,
        /// Per-request event channel (driver keeps the sender, HTTP holds
        /// the receiver as the SSE body source).
        events: tokio::sync::mpsc::UnboundedSender<ReqEvent>,
        /// Request correlation handle (driver mints ReqId, passes SeqId
        /// back via the first event — the HTTP layer never needs it).
        reply: tokio::sync::oneshot::Sender<ReqId>,
    },
    Cancel {
        req: ReqId,
    },
}

/// The driver's public snapshot (stats endpoint source; thread-safe via
/// the command loop — the driver publishes it into an RwLock after
/// each tick).
#[derive(Debug, Clone, Copy, Default)]
pub struct DriverStats {
    pub live_rows: usize,
    pub queued: usize,
    pub tree_nodes: usize,
    pub tree_blocks: usize,
    pub evictable_tokens: usize,
    pub protected_tokens: usize,
    pub tier_census: (usize, usize, usize),
    pub pages_in_use: usize,
    pub pages_free: usize,
    pub ticks: u64,
    pub tokens_committed: u64,
}

impl DriverStats {
    /// From the scheduler's cache census + driver counters.
    pub fn from_census(c: ferrite_dispatch::batch::CacheStats, ticks: u64, tokens: u64, live: usize, queued: usize) -> Self {
        DriverStats {
            live_rows: live,
            queued,
            tree_nodes: c.tree_nodes,
            tree_blocks: c.tree_blocks,
            evictable_tokens: c.evictable_tokens,
            protected_tokens: c.protected_tokens,
            tier_census: c.tier_census,
            pages_in_use: c.pages_in_use,
            pages_free: c.pages_free,
            ticks,
            tokens_committed: tokens,
        }
    }
}

// Re-exports for the driver and api modules (one-import ergonomics).
pub use ferrite_dispatch::batch::CacheStats;
pub use ferrite_dispatch::prelude::TickPlan;

// SeqId is re-exported for drivers that want to key their own maps on the
// scheduler's handles (the HTTP layer uses ReqId only — this boundary
// type exists so engine authors never confuse the two id spaces).
#[allow(dead_code)]
fn _seq_id_boundary(_: SeqId) {}
