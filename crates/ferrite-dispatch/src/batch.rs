//! BatchScheduler v2 — the tick planner: admission, P/D interleave, ingest.
//!
//! ## Design: two-phase tick, plans as plain data
//!
//! The scheduler is a **decision function**: `plan_into` reads the full
//! system state (sequences, rows, radix tree, tier budgets, graph warmth)
//! and emits a [`TickPlan`] — plain data the serve loop hands to the exec
//! backend. Nothing in `plan_into` touches the backend; a mock backend
//! can replay any schedule deterministically. This is the property that
//! makes the scheduler trustworthy at 560 tok/s: no hidden feedback
//! between decision and effect, everything observable in one struct.
//!
//! ## Sequence lifecycle (admission → retirement)
//!
//! ```text
//! Queued ──admit──▶ Prefilling ──cursor=end──▶ Decoding ──eos/max──▶ Retired
//!   │                 ▲ radix resume: cursor starts past the match      │
//!   └── radix hit ────┘   (prefix tokens are NEVER re-run)             │
//!                                                                        │
//! Retired: row released, radix path unpinned; the committed prefix is
//! first promoted into the tree (page-aligned blocks only), so the next
//! request with the same system prompt resumes at zero prefill cost —
//! through the hicache: disk → host → device promotion on hit.
//! ```
//!
//! Each phase transition is one method; illegal ones are unrepresentable
//! (the phase enum is private — outside code sees `status()` queries).
//!
//! ## Admission: page budget through the tier chain (the SGLang bug-fix)
//!
//! A queued prompt is admitted when **all** hold:
//! 1. a free decode row exists (`max_rows` = the graph ladder bound);
//! 2. the page budget covers the prompt's unique suffix (prefix hits pay
//!    only the unmatched tail: matched pages are leased zero-copy);
//! 3. budget reclaim walks the hicache **demote-before-drop**: device →
//!    host → disk demotions keep the prefix *cached* while freeing
//!    device pages; only a tree detach (LRU unpinned leaf) truly drops.
//!
//! This is the deliberate fix for the SGLang deadlock/pollution class:
//! their eviction meets in-flight loadback (async H2D holds page
//! promises the evictor frees — a wait cycle on the load lock), and the
//! tree races the pool's refcount (double free / stale prefix). Here
//! there is **no in-flight tier move**: promotion (loadback) happens
//! *inside* `bind_row_to_snapshot` — synchronous, on the admission path,
//! before the row leases anything; the tree's lock_ref pins the source
//! node before the copy starts and the single `PageRegistry` refcount
//! owns every free. An evictor cannot see a half-promoted prefix because
//! none exists — state moves are atomic registry transitions.
//!
//! ## Interleave: decode-first, prefill piggyback
//!
//! Every live row decodes every tick (MTP graph replays at the bucket
//! shape covering the live count). Prefill chunks ride the same tick up
//! to the token budget: the chunk graph and the decode graph are separate
//! captures (separate streams), and the decode replay's device-resident
//! time has host bubbles the chunk replays fill. `PrefillMode::Alternate`
//! is the conservative policy (never co-issue); `Piggyback` is the
//! high-throughput default.
//!
//! ## Ingest: commit → radix promotion → retirement
//!
//! `ingest_decode` folds the readback (per-row k/bonus) through the pure
//! commit math (`mtp::commit_row`), then promotes any page-aligned prefix
//! blocks into the radix tree — a promotion snapshots the row's GDN state
//! into the tree (device tier first; pressure demotes it), and page
//! ownership transfers to the node, so sharing begins the moment a block
//! completes, not at sequence retirement. Retirement releases the row and
//! unpins the path; **row compaction** moves the highest live row into
//! the hole so the graph-visible row domain stays a dense prefix —
//! between replays only (graph pointers are row-indexed).
//!
//! ## Row identity recap (the invariant the whole crate hangs on)
//!
//! Live decode rows are a **dense prefix** `[0, live_rows)` of slot ids.
//! Admission takes row `live_rows` (the next free row is *always* the
//! count — a Vec push); retirement compacts by moving the highest row
//! into the freed slot (one GDN device copy + a page-lease move via
//! `StateRegistry::move_row` — page-shared prefixes never copy). The
//! graph-visible row domain is dense at every tick boundary. This is the
//! one operation that moves state, and it runs only between replays.

use ferrite_types::{FerriteError, Result};

use crate::arena::{NodeId, TypedArena};
use crate::bucket::BucketAssignment;
use crate::exec::{ChunkWork, DecodeWork, PrefillWork};
use crate::graph::GraphPool;
use crate::mtp::{AcceptBatch, DRAFT_DEPTH, VERIFY_WIDTH};
use crate::radix::{PinnedPath, RadixCache};
use crate::state::{PhysStateStore, StateRegistry, Tier};

/// Prefill/decode co-issue policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillMode {
    /// Chunks ride along decode ticks (throughput mode — default).
    Piggyback,
    /// Never co-issue: alternate decode / prefill ticks (latency-debug;
    /// useful for A/B-ing the co-issue assumption on real graphs).
    Alternate,
}

/// Scheduler configuration — every knob in one struct (no env vars, no
/// globals; the serve binary owns policy).
///
/// Tier capacities follow the hicache hierarchy: device snapshot slots
/// (GPU, hot) ≪ host (RAM reservoir) ≪ disk (NVMe archive — effectively
/// unbounded; the bound is a safety valve, not a sizing goal).
#[derive(Debug, Clone)]
pub struct SchedConfig {
    /// Decode rows (= state registry rows = max bucket shape).
    pub max_rows: u32,
    /// Prefill tokens per tick (chunk ladder is 512; budget should be a
    /// multiple to land whole replays).
    pub prefill_token_budget: usize,
    /// Interleave policy.
    pub prefill_mode: PrefillMode,
    /// Radix node capacity (eviction pressure threshold).
    pub radix_max_nodes: usize,
    /// DSA page size in tokens (radix block granularity).
    pub page_size: usize,
    /// Device-tier snapshot slots (GPU — hot prefix set).
    pub max_device_snaps: usize,
    /// Host-tier snapshot slots (RAM reservoir).
    pub max_host_snaps: usize,
    /// Disk-tier snapshot slots (NVMe archive; safety valve bound).
    pub max_disk_snaps: usize,
    /// Device DSA page budget (the admission currency; SGLang
    /// `num_gpu_blocks` parity).
    pub max_pages: usize,
}

impl Default for SchedConfig {
    fn default() -> Self {
        SchedConfig {
            max_rows: 32,
            prefill_token_budget: 512,
            prefill_mode: PrefillMode::Piggyback,
            radix_max_nodes: 4096,
            page_size: 16,
            max_device_snaps: 512,
            max_host_snaps: 4096,
            max_disk_snaps: 65_536,
            max_pages: 122_880, // 512 MiB / 16-token page × 8 KiB page
        }
    }
}

/// Sequence lifecycle phases (private — transitions are methods only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Submitted, waiting for a decode row + page budget (FIFO —
    /// starvation-free by construction).
    Queued,
    /// Row held, prompt partially (or zero) processed; `cursor` = tokens
    /// the row state already covers (radix-resumed prefix included).
    Prefilling { cursor: usize },
    /// Prompt complete, MTP decoding.
    Decoding,
    /// Terminal: read out by the owner, resources already released.
    Retired,
}

/// One sequence (request). Row, prefix path and stream together — the
/// scheduler's unit of bookkeeping. Row identity is *physical* (registry
/// slot) and the radix path pins the shared prefix it resumes from.
pub struct Seq {
    pub prompt: Vec<u32>,
    pub output: Vec<u32>,
    phase: Phase,
    /// Physical decode row (once admitted).
    pub row: Option<u32>,
    /// Radix pins held by this sequence (dropped at retirement).
    pinned: Option<PinnedPath>,
    /// Radix node the row's committed stream terminates under (prefix
    /// promotion anchor: `promote` extends this node's child frontier).
    anchor: Option<NodeId>,
    /// Radix-matched prefix length at admission (never re-run).
    pub resumed_from: usize,
    /// Last GDN-snapshot token count promoted into the tree
    /// (page-aligned prefix watermark).
    promoted_to: usize,
    /// Last stream position the registry's page lease covers (the row's
    /// committed tokens backing DSA KV pages — grows in page steps).
    leased_to: usize,
    pub eos: u32,
    pub max_new_tokens: usize,
}

impl Seq {
    /// Stream = prompt + output (the committed token stream the radix
    /// tree indexes; pages cover it page-aligned prefix-first).
    fn stream_len(&self) -> usize {
        self.prompt.len() + self.output.len()
    }
}

/// One tick's plan — plain data out of `plan_into` (module doc: the
/// decision/effect seam). Owned buffers; the serve loop replays then
/// drops it. Zero-alloc steady state: `plan_into` reuses the caller's
/// TickPlan (clear + refill).
#[derive(Debug, Clone, Default)]
pub struct TickPlan {
    /// Decode work: live rows, replay shape, per-row stream positions.
    pub decode: Option<DecodeWork>,
    /// Prefill chunks (rows advancing toward cursor == prompt.len()).
    pub prefills: PrefillWork,
    /// Admission decisions taken this tick (row + radix resume length) —
    /// already bound (state moved); informational for logging/metrics.
    pub admissions: Vec<Admission>,
    /// Reclaim decisions (demote-vs-drop outcomes for the page budget) —
    /// informational: the effects are already applied.
    pub reclaims: Vec<Reclaim>,
}

/// One admission: a queued prompt bound to a row this tick.
#[derive(Debug, Clone)]
pub struct Admission {
    pub seq: crate::arena::SeqId,
    pub row: u32,
    /// Prefix hit (tokens skipped by the radix resume — hicache may have
    /// promoted it host/disk → device on the way here).
    pub prefix_hit: usize,
}

/// One page-budget reclaim outcome (which tier paid).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reclaim {
    /// A device-tier snapshot demoted to host (prefix retained).
    DemotedToHost,
    /// A host-tier snapshot demoted to disk (prefix retained).
    DemotedToDisk,
    /// A tree leaf dropped (all tiers full / unpinned LRU — prefix lost).
    Dropped { tokens: usize },
    /// No reclaim was needed (budget already satisfied).
    None,
}

/// Ingest result summary (post-readback state delta — feed to metrics).
#[derive(Debug, Clone, Copy, Default)]
pub struct TickDigest {
    pub tokens_committed: usize,
    pub rows_retired: u32,
    pub blocks_promoted: usize,
    /// Per-row accepts (MTP telemetry — the iron-law canary: a domain
    /// change between draft and verify shows here first).
    pub accept_sum: usize,
    pub accept_count: usize,
}

/// The scheduler: owns sequences, the radix tree, the tier budgets and
/// the tick plan emission. The backend is passed by trait only through
/// the registry (state moves) — plan emission itself is backend-free.
pub struct BatchScheduler<S: PhysStateStore> {
    cfg: SchedConfig,
    seqs: TypedArena<crate::arena::SeqTag, Seq>,
    /// FIFO queue of waiting seqs (admission order — starvation-free;
    /// prefix-affinity reordering is a policy knob, see `admission` doc).
    queue: Vec<crate::arena::SeqId>,
    pub radix: RadixCache,
    pub registry: StateRegistry<S>,
    pub graphs: GraphPool,
    /// Dense live-row bookkeeping: row r ↔ seq (compacted on retire —
    /// module doc "row identity recap").
    rows: Vec<Option<crate::arena::SeqId>>,
    /// Admission-time prefix-cache outcomes (the `CacheStats.hits`/
    /// `misses` counters): one sample per admission, incremented after
    /// the authoritative post-reclaim match.
    hits: u64,
    misses: u64,
}

impl<S: PhysStateStore> BatchScheduler<S> {
    pub fn new(cfg: SchedConfig, store: S) -> Result<Self> {
        let radix = RadixCache::new(cfg.radix_max_nodes, cfg.page_size)?;
        let registry = StateRegistry::new(
            store,
            cfg.max_rows,
            cfg.max_device_snaps,
            cfg.max_host_snaps,
            cfg.max_disk_snaps,
            cfg.max_pages,
            cfg.page_size,
        );
        let rows = (0..cfg.max_rows).map(|_| None).collect();
        Ok(BatchScheduler {
            cfg,
            seqs: TypedArena::with_capacity(64),
            queue: Vec::new(),
            radix,
            registry,
            graphs: GraphPool::new(),
            rows,
            hits: 0,
            misses: 0,
        })
    }

    // -- admission ----------------------------------------------------------

    /// Submit a request. The prompt is radix-matched at admission time
    /// (not now): matches that arrive between submit and admission are
    /// free wins for traffic under the same prefix.
    pub fn submit(&mut self, prompt: Vec<u32>, eos: u32, max_new_tokens: usize) -> Result<crate::arena::SeqId> {
        if prompt.is_empty() {
            return Err(FerriteError::Scheduler("submit: empty prompt".into()));
        }
        let id = self.seqs.insert(Seq {
            prompt,
            output: Vec::new(),
            phase: Phase::Queued,
            row: None,
            pinned: None,
            anchor: None,
            resumed_from: 0,
            promoted_to: 0,
            leased_to: 0,
            eos,
            max_new_tokens,
        });
        self.queue.push(id);
        Ok(id)
    }

    fn free_row(&self) -> Option<u32> {
        self.rows.iter().position(|r| r.is_none()).map(|i| i as u32)
    }

    /// Live rows (decode-batch occupancy — the bucket scheduler's input).
    pub fn live_rows(&self) -> u32 {
        self.rows.iter().filter(|r| r.is_some()).count() as u32
    }

    /// Page-budget reclaim — the hicache demote-before-drop chain.
    ///
    /// Satisfies `extra_pages` of device page budget. Order of payment
    /// (each step retains as much cache as the tiers admit):
    /// 1. device-tier radix snapshots **demote to host** (tree intact,
    ///    prefix hits promote back later — synchronous, no in-flight
    ///    state: the deadlock/pollution class is unrepresentable here);
    /// 2. host-tier snapshots **demote to disk** (NVMe archive);
    /// 3. only then tree leaves **drop** (LRU unpinned — the true loss).
    ///
    /// Victim selection is tree-lead (unpinned LRU leaf first — its GDN
    /// snapshot is what moves), SGLang `evict(num_tokens)` parity, minus
    /// the async loadback race: demote/drop are synchronous registry
    /// transitions on the admission path, and the pinned source of an
    /// in-progress promote cannot be selected as a victim (lock_ref).
    fn ensure_page_budget(&mut self, extra_pages: usize) -> Result<Vec<Reclaim>> {
        let mut reclaims = Vec::new();
        if self.registry.pages_free() >= extra_pages {
            reclaims.push(Reclaim::None);
            return Ok(reclaims);
        }
        // 1) + 2) demote chain through the hicache tiers.
        loop {
            if self.registry.pages_free() >= extra_pages {
                return Ok(reclaims);
            }
            let Some((node, state, _tokens)) = self.radix.evict_victim() else {
                break;
            };
            match self.registry.snapshot_tier(state) {
                Some(Tier::Device) if self.registry.host_can_accept() => {
                    self.registry.demote_snapshot(state)?;
                    reclaims.push(Reclaim::DemotedToHost);
                }
                Some(Tier::Host) if self.registry.disk_can_accept() => {
                    self.registry.demote_snapshot(state)?;
                    reclaims.push(Reclaim::DemotedToDisk);
                }
                _ => {
                    // disk-tier victim, or next tier full: true drop.
                    let (state, tokens) = self.radix.detach(node)?;
                    reclaims.push(Reclaim::Dropped { tokens });
                    self.registry.release_snapshot(state)?;
                }
            }
        }
        // 3) no victims at all — the budget must come from future
        // retirement (queue pressure) or is over-committed: hard error.
        if self.registry.pages_free() < extra_pages {
            return Err(FerriteError::Pool(format!(
                "page budget exhausted: need {extra_pages} pages, {} free, no evictable prefix",
                self.registry.pages_free()
            )));
        }
        Ok(reclaims)
    }

    /// Admit one queued sequence into a free row (radix match + bind).
    fn admit_one(&mut self, seq: crate::arena::SeqId) -> Result<Option<Admission>> {
        let Some(free_row) = self.free_row() else {
            return Ok(None);
        };
        let (prompt, page_size) = {
            let s = self
                .seqs
                .get(seq)
                .ok_or_else(|| FerriteError::Scheduler("admit: stale seq".into()))?;
            (s.prompt.clone(), self.cfg.page_size)
        };
        // Page budget FIRST: the reclaim (demote/drop chain) must not see
        // this seq's future pins — pinning before paying is a leak; the
        // evictor sees only committed pins. A pre-reclaim match estimate
        // sizes the budget ask (unique-suffix pages, radix hits pay less).
        let pre_match = self.radix.match_prefix(&prompt).map(|m| m.matched_tokens);
        let unique_pages = (prompt.len() - pre_match.unwrap_or(0)).div_ceil(page_size);
        let reclaims = self.ensure_page_budget(unique_pages)?;
        let _ = reclaims; // effects applied at admission; plan reports via cache_stats

        // Authoritative match AFTER reclaim: the drop path may have evicted
        // the very prefix we sized against (unpinned LRU — nothing held
        // it). Pin before reading the node state (pins gate eviction).
        let matched = self.radix.match_prefix(&prompt);
        let (matched_tokens, anchor) = match matched {
            Some(m) => (m.matched_tokens, Some(m.node)),
            None => (0, None),
        };
        // Prefix-cache outcome (one sample per admission — the /v1/stats
        // hits/misses counters).
        if matched_tokens > 0 {
            self.hits += 1;
        } else {
            self.misses += 1;
        }
        self.registry.acquire_row(free_row)?;
        let pinned = if matched_tokens > 0 {
            let anchor = anchor.expect("match without node");
            let path = self.radix.pin_path(anchor);
            let state = self
                .radix
                .node(anchor)?
                .state
                .ok_or_else(|| FerriteError::Scheduler("admit: anchor without state".into()))?;
            // bind_row promotes the snapshot through the hicache tiers
            // (disk → host → device) synchronously — no in-flight state.
            self.registry.bind_row_to_snapshot(free_row, state, matched_tokens)?;
            Some(path)
        } else {
            None
        };
        {
            let s = self
                .seqs
                .get_mut(seq)
                .ok_or_else(|| FerriteError::Scheduler("admit: stale seq".into()))?;
            s.phase = Phase::Prefilling { cursor: matched_tokens };
            s.row = Some(free_row);
            s.pinned = pinned;
            s.anchor = anchor;
            s.resumed_from = matched_tokens;
            s.promoted_to = matched_tokens;
            s.leased_to = matched_tokens;
        }
        self.rows[free_row as usize] = Some(seq);
        Ok(Some(Admission { seq, row: free_row, prefix_hit: matched_tokens }))
    }

    // -- planning (phase 1 of the tick) -------------------------------------

    /// Emit the tick plan: admissions first (rows ready for prefill in the
    /// same tick), then prefill chunks under the token budget, then the
    /// decode batch at the covering bucket shape.
    ///
    /// `plan_into` reuses the output buffers (zero-alloc steady state).
    /// Optimistic cursor: prefill chunks advance `cursor` at plan time —
    /// the backend executes the plan before the next plan; a failed
    /// replay is a panic-level fault (graph replays are not partially
    /// roll-back-able; the tick protocol owns this — see `exec` docs).
    pub fn plan_into(&mut self, out: &mut TickPlan) -> Result<()> {
        out.decode = None;
        out.prefills.chunks.clear();
        out.admissions.clear();
        out.reclaims.clear();

        // 1) admission until rows exhaust (FIFO; prefix-affinity reorder
        // is deliberately NOT implemented — cache-aware scheduling with
        // shared prefixes wins more via radix hits than via queue order).
        let queued: Vec<_> = std::mem::take(&mut self.queue);
        let mut requeued = Vec::with_capacity(queued.len());
        for seq in queued {
            match self.admit_one(seq)? {
                Some(adm) => out.admissions.push(adm),
                None => requeued.push(seq), // no rows (budget errors surface here)
            }
        }
        self.queue = requeued;

        // 2) prefill chunks under the token budget (piggyback or alternate).
        let piggyback = matches!(self.cfg.prefill_mode, PrefillMode::Piggyback)
            || self.live_rows() == 0;
        let mut budget = if piggyback { self.cfg.prefill_token_budget } else { 0 };
        let prefillable: Vec<(crate::arena::SeqId, u32, usize, usize)> = self
            .seqs
            .iter()
            .filter_map(|(id, s)| match s.phase {
                Phase::Prefilling { cursor } => Some((id, s.row?, cursor, s.prompt.len())),
                _ => None,
            })
            .collect();
        for (seq, row, cursor, plen) in prefillable {
            if budget == 0 {
                break;
            }
            let take = budget.min(plen - cursor);
            if take == 0 {
                continue;
            }
            let ids = {
                let s = self.seqs.get(seq).expect("arena-consistent");
                s.prompt[cursor..cursor + take].to_vec()
            };
            out.prefills.chunks.push(ChunkWork {
                row,
                seq,
                resume_at: cursor,
                ids,
            });
            budget -= take;
            let s = self.seqs.get_mut(seq).expect("arena-consistent");
            s.phase = Phase::Prefilling { cursor: cursor + take };
        }

        // 3) decode batch at the covering shape (dense live rows).
        let live = self.live_rows();
        if live > 0 {
            let shape = self.graphs.replay_shape(live)?;
            let mut rows = Vec::with_capacity(live as usize);
            let mut seqs = Vec::with_capacity(live as usize);
            let mut positions = Vec::with_capacity(live as usize);
            for (r, occ) in self.rows.iter().enumerate() {
                if let Some(seq) = occ {
                    rows.push(r as u32);
                    seqs.push(*seq);
                    positions.push(self.seqs.get(*seq).map(|s| s.stream_len()).unwrap_or(0));
                }
            }
            out.decode = Some(DecodeWork {
                bucket: BucketAssignment {
                    shape,
                    rows,
                    seqs,
                    pad_rows: shape.pad(live),
                },
                // draft head runs inside the decode replay (deployed
                // shape); empty = backend binds from row state — see
                // exec::DecodeWork docs (headless-draft debugging fills it).
                verify_ids: Vec::new(),
                row_positions: positions,
            });
        }
        Ok(())
    }

    /// Allocating wrapper (one-shot / integration bring-up / simulation).
    pub fn plan(&mut self) -> Result<TickPlan> {
        let mut out = TickPlan::default();
        self.plan_into(&mut out)?;
        Ok(out)
    }

    // -- ingest (phase 2 of the tick) ---------------------------------------

    /// Fold a decode readback: commit math per row, radix promotion of
    /// page-aligned blocks, retirement of finished rows (with row
    /// compaction to keep the row domain dense).
    pub fn ingest_decode(&mut self, plan: &TickPlan, readback: &AcceptBatch) -> Result<TickDigest> {
        let mut digest = TickDigest::default();
        let Some(work) = &plan.decode else {
            return Ok(digest);
        };
        if readback.accepts.len() != work.bucket.seqs.len() {
            return Err(FerriteError::Scheduler(format!(
                "readback rows {} != bucket rows {}",
                readback.accepts.len(),
                work.bucket.seqs.len()
            )));
        }
        let mut retired: Vec<crate::arena::SeqId> = Vec::new();
        for (i, accept) in readback.accepts.iter().enumerate() {
            let seq = work.bucket.seqs[i];
            let row = work.bucket.rows[i];
            let (done, promoted) = self.commit_one(seq, row, accept)?;
            digest.tokens_committed += accept.committed_tokens();
            digest.blocks_promoted += promoted;
            digest.accept_sum += accept.k as usize;
            digest.accept_count += 1;
            if done {
                digest.rows_retired += 1;
                retired.push(seq);
            }
        }
        for seq in retired {
            // Resolve the row AT RETIRE TIME, not from the bucket layout:
            // each compact_after_retire moves the highest live row into the
            // freed hole (and updates that seq's `row`), so the bucket's
            // row indices go stale the moment the first retire runs.
            // Retiring a stale index hit "retire: row N unowned" whenever
            // 2+ rows finished in one tick (concurrent completion — the
            // compaction cascade vacated later indices).
            let Some(row) = self.seqs.get(seq).and_then(|s| s.row) else {
                continue; // already terminal (defensive: double-finish edge)
            };
            self.compact_after_retire(row)?;
        }
        Ok(digest)
    }

    /// Ingest one prefill chunk batch (cursor already advanced at plan
    /// time — the optimistic-tick protocol): promote page-aligned prefix
    /// blocks the chunks completed, and flip to Decoding when a prompt is
    /// exhausted.
    pub fn ingest_prefill(&mut self, plan: &TickPlan) -> Result<TickDigest> {
        let mut digest = TickDigest::default();
        for chunk in &plan.prefills.chunks {
            let (cursor, plen) = {
                let s = self.seqs.get(chunk.seq).expect("arena-consistent");
                let Phase::Prefilling { cursor } = s.phase else {
                    continue; // retired mid-tick (eos edge)
                };
                (cursor, s.prompt.len())
            };
            let new_blocks = self.promote_row_prefix(chunk.seq, chunk.row, cursor)?;
            digest.blocks_promoted += new_blocks;
            if cursor >= plen {
                let s = self.seqs.get_mut(chunk.seq).expect("arena-consistent");
                s.phase = Phase::Decoding;
            }
        }
        Ok(digest)
    }

    /// Per-row commit (MTP accept fold + radix promotion of completed
    /// pages). Pure bookkeeping — the device committed tokens in the
    /// replay; this updates streams, tree, tiers.
    fn commit_one(
        &mut self,
        seq: crate::arena::SeqId,
        row: u32,
        accept: &crate::mtp::AcceptRow,
    ) -> Result<(bool, usize)> {
        // Output-level commit (max_new_tokens caps the COMPLETION, OpenAI
        // `max_tokens` semantics — not the prompt+output stream length)
        // followed by stream-level radix promotion (pages cover the whole
        // committed stream: prompt + output).
        let (eos, max_new, page_size, prompt_len) = {
            let s = self
                .seqs
                .get(seq)
                .ok_or_else(|| FerriteError::Scheduler("commit: stale seq".into()))?;
            (s.eos, s.max_new_tokens, self.cfg.page_size, s.prompt.len())
        };
        let commit = {
            let s = self.seqs.get_mut(seq).expect("arena-consistent");
            // `s.output` is the completion stream alone — the max-tokens
            // cap and the accepted-window append both apply to it.
            let out_len = s.output.len();
            crate::mtp::commit_row(&mut s.output, out_len, accept, eos, max_new, page_size)?
        };
        // radix promotion: whole pages of the committed stream completed
        // by this commit (prompt ++ output watermark).
        let stream_len = prompt_len + self.seqs.get(seq).expect("arena-consistent").output.len();
        let promoted = self.promote_row_prefix(seq, row, stream_len)?;
        Ok((commit.finished, promoted))
    }

    /// Promote the row's page-aligned prefix watermark to the radix tree
    /// (admission-agnostic: prefill chunks and MTP commits share the
    /// watermark logic — new full pages become shareable children).
    ///
    /// One node per page (the block granularity — a node's tokens are one
    /// DSA page, its snapshot the state at the block's end): promotion
    /// walks [`from`, watermark) one page at a time, chaining children
    /// down from the anchor — the sharing unit is exactly the page, and a
    /// mid-prefix reader matches at page boundaries only.
    fn promote_row_prefix(
        &mut self,
        seq: crate::arena::SeqId,
        row: u32,
        stream_len_now: usize,
    ) -> Result<usize> {
        let page = self.cfg.page_size;
        let (mut watermark, mut anchor) = {
            let s = self.seqs.get(seq).expect("arena-consistent");
            (s.promoted_to, s.anchor)
        };
        let mut promoted = 0usize;
        while watermark + page <= stream_len_now {
            let block_end = watermark + page;
            // Snapshot at the block boundary (the page's last-token state).
            let state = self.registry.snapshot_from_row(row, block_end)?;
            // Stream window [watermark, block_end) — prompt then output.
            let tokens = {
                let s = self.seqs.get(seq).expect("arena-consistent");
                let mut window = Vec::with_capacity(page);
                if watermark < s.prompt.len() {
                    let end = block_end.min(s.prompt.len());
                    window.extend_from_slice(&s.prompt[watermark..end]);
                }
                if block_end > s.prompt.len() {
                    let o_start = watermark.saturating_sub(s.prompt.len());
                    let o_end = (block_end - s.prompt.len()).min(s.output.len());
                    if o_start < o_end {
                        window.extend_from_slice(&s.output[o_start..o_end]);
                    }
                }
                window
            };
            let parent = anchor.unwrap_or_else(|| self.radix.root());
            let (child, inserted_new) = self.radix.insert_branch(parent, tokens, state)?;
            if !inserted_new {
                // Identical block already shared (another request promoted
                // this exact page-prefix): the existing node's state IS
                // this block's state (block-equal prefixes ⇒ equal GDN
                // snapshots); release our fresh snapshot and share the node.
                self.registry.release_snapshot(state)?;
            }
            anchor = Some(child);
            watermark = block_end;
            promoted += 1;
        }
        if promoted > 0 {
            let s = self.seqs.get_mut(seq).expect("arena-consistent");
            s.promoted_to = watermark;
            s.anchor = anchor;
        }
        Ok(promoted)
    }

    /// Retirement: unpin the radix path, release the row, terminal phase.
    /// Then compaction: the highest live row moves into the freed hole
    /// (dense row domain — module doc "row identity recap").
    fn compact_after_retire(&mut self, freed_row: u32) -> Result<()> {
        // find the seq to retire
        let seq = self
            .rows
            .get(freed_row as usize)
            .and_then(|s| *s)
            .ok_or_else(|| FerriteError::Scheduler(format!("retire: row {freed_row} unowned")))?;
        {
            let s = self
                .seqs
                .get_mut(seq)
                .ok_or_else(|| FerriteError::Scheduler("retire: stale seq".into()))?;
            let pinned = s.pinned.take();
            s.phase = Phase::Retired;
            s.row = None;
            if let Some(path) = &pinned {
                self.radix.unpin_path(path);
            }
        }
        self.registry.release_row(freed_row)?;
        self.rows[freed_row as usize] = None;
        // compaction: highest live row fills the hole
        if let Some(hi) = self.rows.iter().rposition(|r| r.is_some()) {
            let hi = hi as u32;
            if hi > freed_row {
                let seq = self.rows[hi as usize].expect("rposition found it");
                self.registry.move_row(hi, freed_row)?;
                if let Some(s) = self.seqs.get_mut(seq) {
                    s.row = Some(freed_row);
                }
                self.rows[freed_row as usize] = Some(seq);
                self.rows[hi as usize] = None;
            }
        }
        Ok(())
    }

    // -- queries ------------------------------------------------------------

    /// Abort an active sequence (client disconnect / API cancel): retire
    /// it mid-flight — release the decode row, unpin the radix path,
    /// return its pages. The partial output stays queryable (`output`).
    ///
    /// Idempotent: cancelling a retired/unknown seq is a no-op (returns
    /// false). This is the SSE-disconnect path of the HTTP server — the
    /// row freed here is admitted to the next queued request at the
    /// coming tick (no partial-tick churn; the tick is the only writer).
    pub fn cancel(&mut self, seq: crate::arena::SeqId) -> Result<bool> {
        let (phase, row) = self
            .seqs
            .get(seq)
            .map(|s| (s.phase, s.row))
            .ok_or_else(|| FerriteError::Scheduler("cancel: stale seq".into()))?;
        match phase {
            Phase::Retired => Ok(false), // already terminal (idempotent)
            Phase::Queued => {
                self.queue.retain(|&q| q != seq);
                if let Some(s) = self.seqs.get_mut(seq) {
                    s.phase = Phase::Retired;
                }
                Ok(true)
            }
            Phase::Prefilling { .. } | Phase::Decoding => {
                let row = row.ok_or_else(|| {
                    FerriteError::Scheduler("cancel: active seq without row (corrupt state)".into())
                })?;
                // committed prefix promotion happens naturally at the next
                // page boundary; a cancelled seq's partial pages unpin on
                // retire (the tree keeps only what it already promoted).
                self.compact_after_retire(row)?;
                Ok(true)
            }
        }
    }

    pub fn status(&self, seq: crate::arena::SeqId) -> Result<&'static str> {
        let phase = self
            .seqs
            .get(seq)
            .map(|s| match s.phase {
                Phase::Queued => "queued",
                Phase::Prefilling { .. } => "prefilling",
                Phase::Decoding => "decoding",
                Phase::Retired => "retired",
            })
            .ok_or_else(|| FerriteError::Scheduler("status: stale seq".into()))?;
        Ok(phase)
    }

    pub fn output(&self, seq: crate::arena::SeqId) -> Result<Vec<u32>> {
        Ok(self
            .seqs
            .get(seq)
            .map(|s| s.output.clone())
            .ok_or_else(|| FerriteError::Scheduler("output: stale seq".into()))?)
    }

    pub fn queued(&self) -> usize {
        self.queue.len()
    }

    /// Radix + hicache census (diagnostics: tree nodes/blocks, tier
    /// occupancy, page budget headroom, prefix-cache hit/miss counters).
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            tree_nodes: self.radix.live_nodes(),
            tree_blocks: self.radix.total_blocks(),
            evictable_tokens: self.radix.evictable_tokens(),
            protected_tokens: self.radix.protected_tokens(),
            tier_census: self.registry.tier_census(),
            pages_in_use: self.registry.pages_in_use(),
            pages_free: self.registry.pages_free(),
            hits: self.hits,
            misses: self.misses,
        }
    }
}

/// Cache/hicache occupancy snapshot (metrics + admission-pressure input).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    pub tree_nodes: usize,
    pub tree_blocks: usize,
    pub evictable_tokens: usize,
    pub protected_tokens: usize,
    /// (device, host, disk) snapshot census.
    pub tier_census: (usize, usize, usize),
    pub pages_in_use: usize,
    pub pages_free: usize,
    /// Prefix-cache hit/miss counters (one sample per ADMISSION — a hit is
    /// an admission whose prompt matched ≥1 cached token, not a per-lookup
    /// tally). Engines without a prefix cache report 0/0.
    pub hits: u64,
    pub misses: u64,
}

// Constant-parity guards (compile-time protocol facts the scheduler
// assumes — mtp.rs owns the values; a change there is a change here).
const _: () = {
    assert!(DRAFT_DEPTH >= 1);
    assert!(VERIFY_WIDTH == DRAFT_DEPTH + 1);
};
