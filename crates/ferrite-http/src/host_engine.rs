//! Host engine: deterministic generation over the real scheduler.
//!
//! This is the bring-up backend — and the permanent architecture proof.
//! It drives the **actual** `BatchScheduler` tick protocol (admission →
//! radix resume → prefill → decode → ingest → retire) with the *same*
//! plans the CUDA backend will execute; only the compute is replaced:
//! `decode`/`readback` mint deterministic tokens instead of running the
//! transformer. Every scheduling decision — page budgets, tier demotion,
//! prefix sharing, MTP accept folding, row compaction — is real, so:
//!
//! - the HTTP/SSE layer can be developed and demoed end-to-end with no
//!   GPU (a `--mock` serve speaks real chat completions over real
//!   scheduling);
//! - the driver↔scheduler↔engine protocol is exercised in CI-shaped
//!   form (the plan structs are plain data; the mock replays them
//!   deterministically — the property `ferrite-dispatch` was designed
//!   for);
//! - the CUDA backend later plugs in by implementing the same
//!   `ExecBackend` trait with device work (see `ferrite_dispatch::exec`).
//!
//! ## Deterministic generation contract
//!
//! A request's token stream is a pure function of its prompt: seed =
//! FNV-1a(prompt ids) and token[i] = 32 + (seed mixing i) % 95 —
//! printable ASCII, so the SSE deltas decode to visible text, identical
//! on every run (golden-testable). The stream ends with the stop id at a
//! prompt-derived length; `max_new_tokens` (OpenAI `max_tokens`) caps it
//! independently through the scheduler's commit path — both finish
//! paths (stop / length) exercise `FinishReason::{Stop,Length}`.

use std::collections::HashMap;

use ferrite_dispatch::arena::SeqId;
use ferrite_dispatch::batch::{BatchScheduler, SchedConfig, TickPlan};
use ferrite_dispatch::bucket::Bucket;
use ferrite_dispatch::exec::{DecodeReadback, DecodeWork, ExecBackend, PrefillWork};
use ferrite_dispatch::graph::GraphKind;
use ferrite_dispatch::mtp::{AcceptBatch, AcceptRow, DRAFT_DEPTH};
use ferrite_dispatch::state::HostStateStore;
use ferrite_types::{FerriteError, Result};

/// The standard GLM stop id (`<|end|>` — ferrite-serve parity: 154820).
pub const STOP_ID: u32 = 154_820;

/// FNV-1a over the prompt — the generation seed.
fn seed_of(prompt: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &t in prompt {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Deterministic token at stream position `i` for `seed`.
fn token_at(seed: u64, i: usize) -> u32 {
    let mut h = seed.wrapping_add(i as u64).wrapping_mul(0x9e3779b97f4a7c15);
    h ^= h >> 29;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 32;
    32 + (h % 95) as u32 // printable ASCII
}

/// Prompt-derived stream length (16–64 tokens: enough for SSE demos,
/// bounded enough for quick tests).
fn stream_len_for(seed: u64) -> usize {
    16 + (seed % 49) as usize
}

/// Mock generation state per live sequence.
#[derive(Debug, Clone)]
struct GenState {
    seed: u64,
    /// Committed tokens so far (drives the stream position).
    emitted: usize,
    /// Stream length before the stop token.
    len: usize,
    /// Total tokens generated since admission (incl. accepted drafts).
    total: usize,
}

/// The `ExecBackend` implementation for the host engine: deterministic
/// generation, no device work. `decode` records the row→seq layout of
/// the tick; `readback` mints the MTP accept window from the generation
/// state — the scheduler's ingest folds it exactly as it will fold the
/// CUDA readback.
pub struct HostExecBackend {
    /// Row → generating sequence (from the last decode work).
    rows: HashMap<u32, SeqId>,
    /// Seq → generation state (mock stream position).
    gen: HashMap<SeqId, GenState>,
    /// Prompt per seq (seed source; recorded at submit via the driver).
    prompts: HashMap<SeqId, Vec<u32>>,
    /// Stop id emitted at stream end.
    stop_id: u32,
    /// Telemetry: ticks executed (decode replays).
    ticks: u64,
    /// The StateStoreCap placeholder (host path: state moves are
    /// scheduler-internal; the cap errors if ever reached).
    dummy_cap: HostBackendCap,
}

impl HostExecBackend {
    pub fn new(stop_id: u32) -> Self {
        HostExecBackend {
            rows: HashMap::new(),
            gen: HashMap::new(),
            prompts: HashMap::new(),
            stop_id,
            ticks: 0,
            dummy_cap: HostBackendCap,
        }
    }

    /// Register a sequence's prompt (seed + stream length source).
    /// Called at admission (driver-side hook).
    pub fn register(&mut self, seq: SeqId, prompt: &[u32]) {
        let seed = seed_of(prompt);
        self.prompts.insert(seq, prompt.to_vec());
        self.gen.insert(
            seq,
            GenState { seed, emitted: 0, len: stream_len_for(seed), total: 0 },
        );
    }

    /// Drop generation state (retirement/cancel).
    pub fn deregister(&mut self, seq: SeqId) {
        self.gen.remove(&seq);
        self.prompts.remove(&seq);
        self.rows.retain(|_, s| *s != seq);
    }

    /// Prompt length of a registered seq (page-growth input for the tick).
    pub fn prompt_len(&self, seq: SeqId) -> Option<usize> {
        self.prompts.get(&seq).map(|p| p.len())
    }

    /// The per-tick accept window for one row (deterministic MTP shape:
    /// k=2 drafts + 1 bonus = 3 tokens/step, the engine's steady accept).
    fn accept_for(&mut self, seq: SeqId) -> AcceptRow {
        let (seed, emitted, len) = {
            let g = self.gen.get(&seq).expect("gen state registered at admit");
            (g.seed, g.emitted, g.len)
        };
        // Stream end: the stop token rides the bonus slot (k collapses
        // to 0 — verify rejected the drafts; the target's own token
        // finishes the turn). Both MTP end shapes stay exercised.
        if emitted >= len {
            return AcceptRow { k: 0, bonus: self.stop_id, accepted: [0; DRAFT_DEPTH], finished: false };
        }
        let t0 = token_at(seed, emitted);
        let t1 = token_at(seed, emitted + 1);
        let t2 = token_at(seed, emitted + 2);
        // natural stop inside the step's 3-token window: truncate the
        // accepted draft run at the stop position (greedy MTP would see
        // the drafts diverge from the target at that point).
        if emitted + 1 >= len {
            return AcceptRow { k: 0, bonus: t0, accepted: [0; DRAFT_DEPTH], finished: false };
        }
        if emitted + 2 >= len {
            return AcceptRow { k: 1, bonus: t1, accepted: [t0, 0], finished: false };
        }
        AcceptRow { k: 2, bonus: t2, accepted: [t0, t1], finished: false }
    }

    /// Advance the mock stream by an accept window (post-ingest hook).
    pub fn advance(&mut self, seq: SeqId, tokens: usize) {
        if let Some(g) = self.gen.get_mut(&seq) {
            g.emitted += tokens;
            g.total += tokens;
        }
    }
}

impl ExecBackend for HostExecBackend {
    fn warmup(&mut self) -> Result<()> {
        Ok(()) // host graphs are the plan structs themselves — nothing to capture
    }

    fn decode(&mut self, work: &DecodeWork) -> Result<()> {
        self.ticks += 1;
        // Record this tick's row→seq layout (readback mints per-row
        // accepts in the same order — the bucket's dense row domain).
        self.rows.clear();
        for (i, &seq) in work.bucket.seqs.iter().enumerate() {
            self.rows.insert(work.bucket.rows[i], seq);
        }
        Ok(())
    }

    fn readback(&mut self, live_rows: usize) -> Result<DecodeReadback> {
        let mut accepts = Vec::with_capacity(live_rows);
        for i in 0..live_rows {
            let seq = *self
                .rows
                .get(&(i as u32))
                .ok_or_else(|| FerriteError::Pool(format!("readback: row {i} not decoded this tick")))?;
            accepts.push(self.accept_for(seq));
        }
        Ok(DecodeReadback { accepts: AcceptBatch { accepts } })
    }

    fn prefill(&mut self, _work: &PrefillWork) -> Result<()> {
        // Host path: prefill is pure bookkeeping (the scheduler's radix
        // resume + cursor advance IS the prefill for a mock engine — no
        // transformer to run). ChunkWork ids were already consumed by the
        // plan (optimistic cursor).
        Ok(())
    }

    fn grow_pages(&mut self, _row: u32, _n: usize) -> Result<()> {
        Ok(()) // HostStateStore's page pool is unbounded logical ids
    }

    fn state_moves(&mut self) -> &mut dyn ferrite_dispatch::exec::StateStoreCap {
        // The host engine's state moves happen inside the scheduler's own
        // registry (snapshot/restore are registry calls in ingest) — the
        // backend capability is the CUDA seam; the host path never calls it.
        &mut self.dummy_cap
    }
}

/// The engine pair (scheduler + mock exec) — one tick loop over both.
/// The driver (see `driver.rs`) owns this + the HTTP-facing channels.
pub struct HostEngine {
    pub sched: BatchScheduler<HostStateStore>,
    pub exec: HostExecBackend,
}

impl HostEngine {
    pub fn new(cfg: SchedConfig, stop_id: u32) -> Result<Self> {
        let mut sched = BatchScheduler::new(cfg, HostStateStore::new())?;
        // Warm the graph pool: the host backend "captures" every shape
        // (its graphs are the plan structs — nothing to build), so all
        // bucket replays are legal from tick 1. (The CUDA backend warms
        // the same set by real capture; a cold shape there falls back
        // up the ladder — see `GraphPool::replay_shape`.)
        for shape in <Bucket as TryFrom<u32>>::try_from(1)
            .ok()
            .into_iter()
            .chain([2, 4, 8, 16, 32].iter().filter_map(|&b| Bucket::try_from(b).ok()))
        {
            sched.graphs.mark_warm(GraphKind::DecodeMtp(shape));
        }
        for &chunk in ferrite_dispatch::graph::PREFILL_CHUNKS.iter() {
            sched.graphs.mark_warm(GraphKind::PrefillChunk(chunk));
        }
        Ok(HostEngine {
            sched,
            exec: HostExecBackend::new(stop_id),
        })
    }

    /// Submit through the scheduler; register the mock generation seed.
    pub fn submit(&mut self, prompt_ids: Vec<u32>, max_new_tokens: usize, eos: u32) -> Result<SeqId> {
        let seq = self.sched.submit(prompt_ids.clone(), eos, max_new_tokens)?;
        self.exec.register(seq, &prompt_ids);
        Ok(seq)
    }

    /// One tick: plan → prefill replay → decode replay → readback →
    /// ingest. Returns the executed plan (the driver reads admissions +
    /// per-row commits from it to mint ReqEvents).
    ///
    /// The protocol order is exactly what the CUDA driver will run —
    /// `prefill` and `decode` on the backend, one readback, one ingest
    /// per tick; the mock's readback is the only substitution.
    ///
    /// Page growth (the registry bookkeeping the device backend does
    /// inside its DSA write kernels): the mock engine grows a row's DSA
    /// pages to cover its committed stream BEFORE ingest promotes page-
    /// aligned prefixes into the radix tree (snapshot requires the pages
    /// to exist — the registry is the single page accountant, the CUDA
    /// backend's kernels allocate in-flight instead).
    pub fn tick(&mut self, plan: &mut TickPlan) -> Result<()> {
        self.sched.plan_into(plan)?;
        if !plan.prefills.chunks.is_empty() {
            let pw = PrefillWork { chunks: std::mem::take(&mut plan.prefills.chunks) };
            self.exec.prefill(&pw)?;
            // grow DSA pages to cover the post-chunk stream, then put the
            // chunks back for ingest (phase flips + radix promotion ride
            // the chunk cursor the plan already advanced).
            for chunk in &pw.chunks {
                let covered = chunk.resume_at + chunk.ids.len();
                self.sched.registry.ensure_row_pages(chunk.row, covered)?;
            }
            plan.prefills.chunks = pw.chunks;
            self.sched.ingest_prefill(plan)?;
        }
        if let Some(dw) = plan.decode.take() {
            self.exec.decode(&dw)?;
            let live = dw.bucket.seqs.len();
            let rb = self.exec.readback(live)?;
            // advance the mock streams by each row's committed window
            // (commit math runs inside ingest; the engine advances after
            // with the readback's own view — the accept windows are
            // consumed, not the folded output)
            let advances: Vec<(SeqId, usize)> = dw
                .bucket
                .seqs
                .iter()
                .zip(rb.accepts.accepts.iter())
                .map(|(s, a)| (*s, a.committed_tokens()))
                .collect();
            // page growth for the decode window (same registry contract
            // as the prefill branch above)
            for (i, &seq) in dw.bucket.seqs.iter().enumerate() {
                let row = dw.bucket.rows[i];
                let stream_len = self
                    .sched
                    .output(seq)
                    .map(|o| o.len())
                    .unwrap_or(0)
                    + advances.iter()
                        .find(|(s, _)| *s == seq)
                        .map(|(_, n)| *n)
                        .unwrap_or(0);
                let prompt_len = self.prompt_len(seq).unwrap_or(0);
                self.sched.registry.ensure_row_pages(row, prompt_len + stream_len)?;
            }
            // put the decode work back BEFORE ingest: ingest reads
            // plan.decode (bucket rows/seqs) to fold the commit — a taken
            // plan makes ingest a no-op and the tick spins (found by
            // tokens_committed=0 with 14M ticks in /v1/stats).
            plan.decode = Some(dw);
            self.sched.ingest_decode(plan, &rb.accepts)?;
            for (seq, n) in advances {
                self.exec.advance(seq, n);
            }
        }
        Ok(())
    }

    /// Prompt length of a seq (page-growth input: stream = prompt+output).
    fn prompt_len(&self, seq: SeqId) -> Option<usize> {
        self.exec.prompt_len(seq)
    }

    /// Cancel (client disconnect): retire the seq and drop mock state.
    pub fn cancel(&mut self, seq: SeqId) -> Result<bool> {
        let cancelled = self.sched.cancel(seq)?;
        if cancelled {
            self.exec.deregister(seq);
        }
        Ok(cancelled)
    }

    /// Per-seq committed output (for the non-streaming path and usage).
    pub fn output(&self, seq: SeqId) -> Result<Vec<u32>> {
        self.sched.output(seq)
    }
}

// The host backend's StateStoreCap placeholder: unreachable-by-design.
// On the host path every state move (snapshot/restore/release) is a
// scheduler-internal registry call during ingest/admission — the cap is
// the CUDA seam (the device backend implements it to move GDN snapshots
// D2D); reaching it on the host path is a wiring bug, and it says so.
#[derive(Debug, Default)]
pub struct HostBackendCap;

impl ferrite_dispatch::exec::StateStoreCap for HostBackendCap {
    fn snapshot_row(&mut self, _row: u32, _committed_tokens: usize) -> Result<ferrite_dispatch::state::StateId> {
        Err(FerriteError::Pool(
            "host engine: state moves are scheduler-internal (cap is the CUDA seam)".into(),
        ))
    }
    fn restore_row(
        &mut self,
        _snap: ferrite_dispatch::state::StateId,
        _row: u32,
        _tokens: usize,
    ) -> Result<()> {
        Err(FerriteError::Pool(
            "host engine: state moves are scheduler-internal (cap is the CUDA seam)".into(),
        ))
    }
    fn release_snapshot(&mut self, _snap: ferrite_dispatch::state::StateId) -> Result<()> {
        Err(FerriteError::Pool(
            "host engine: state moves are scheduler-internal (cap is the CUDA seam)".into(),
        ))
    }
}

/// The deterministic mock engine IS a ServeEngine: the driver runs it the
/// same way it runs the CUDA engine (ferrite-serve's GpuEngine) — one
/// trait, two backends.
impl crate::engine::ServeEngine for HostEngine {
    fn submit(
        &mut self,
        prompt_ids: Vec<u32>,
        max_new_tokens: usize,
        eos: u32,
    ) -> Result<SeqId> {
        HostEngine::submit(self, prompt_ids, max_new_tokens, eos)
    }

    fn tick(&mut self, plan: &mut TickPlan) -> Result<()> {
        HostEngine::tick(self, plan)
    }

    fn output(&self, seq: SeqId) -> Result<Vec<u32>> {
        HostEngine::output(self, seq)
    }

    fn cancel(&mut self, seq: SeqId) -> Result<bool> {
        HostEngine::cancel(self, seq)
    }

    fn deregister(&mut self, seq: SeqId) {
        self.exec.deregister(seq);
    }

    fn status(&self, seq: SeqId) -> Option<&'static str> {
        self.sched.status(seq).ok()
    }

    fn live_rows(&self) -> usize {
        self.sched.live_rows() as usize
    }

    fn queued(&self) -> usize {
        self.sched.queued()
    }

    fn cache_stats(&self) -> ferrite_dispatch::batch::CacheStats {
        self.sched.cache_stats()
    }

    fn stop_id(&self) -> u32 {
        STOP_ID
    }

    /// Mock stop set: the single deterministic STOP_ID.
    fn is_stop(&self, t: u32) -> bool {
        t == STOP_ID
    }
}
