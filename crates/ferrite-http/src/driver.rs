//! The driver: one thread owning the engine, HTTP on the other side.
//!
//! ## Why one thread
//!
//! The scheduler (`BatchScheduler`) is single-writer by design — the tick
//! is a pure function of state, mutations are admission → replay →
//! ingest in strict order (the two-phase tick protocol is what keeps the
//! hicache race-free; see `ferrite-dispatch::batch`). So the engine
//! thread is the *only* scheduler writer: commands (submit/cancel)
//! arrive over a tokio mpsc, tick outputs leave as per-request event
//! streams. No locks on the scheduling domain, ever.
//!
//! ```text
//!   axum handlers (async)                     engine thread (std)
//!   ───── DriverCmd::Submit ─────▶ drain ─┐
//!   ───── DriverCmd::Cancel ─────▶        │  loop {
//!   ◀──── ReqEvent (per-req mpsc) ◀───────┤    drain cmds → engine.submit/cancel
//!   ◀──── stats (RwLock snapshot) ◀──tick─┘    engine.tick(plan)
//!                                              diff outputs → Token deltas
//!                                              retire finished → Finished events
//!                                          }
//! ```
//!
//! ## Token streaming without engine coupling
//!
//! The engine commits tokens inside `tick` (MTP accept windows fold in
//! `ingest`); the driver diffs each live request's output stream against
//! the amount it already sent — no per-token engine callbacks, no shared
//! token buffers: the output query is `O(new tokens)` and the SSE layer
//! renders whatever batch the window contains (1–3 tokens per MTP step).
//!
//! ## Finish detection
//!
//! A request finishes when the scheduler retires it (EOS reached or
//! `max_tokens`): the stop token rides the stream's tail — the driver
//! strips it (OpenAI `finish_reason` semantics: the stop marker is not
//! content) and reports `Stop`; retirement without the stop marker is
//! `Length`. Client disconnects send `DriverCmd::Cancel` (the SSE body
//!'s Drop guard) → `FinishReason::Cancelled`, which the driver reports
//! on the event stream too (a graceful `event: done` before close, even
//! though nobody may be listening — the channel send is fire-and-forget).

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use ferrite_dispatch::arena::SeqId;
use ferrite_dispatch::batch::{SchedConfig, TickPlan};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::engine::{
    DriverCmd, DriverStats, FinishReason, ReqEvent, ReqId, RequestSpec, ServeEngine, Usage,
};
use crate::host_engine::HostEngine;

/// The HTTP-facing handle: submit/cancel from any async context, stats
/// without contention (RwLock written once per tick, read per request).
///
/// Submit/cancel ordering is queue-serialized on the driver thread: the
/// ReqId reply lands before the caller can issue any cancel for it, so
/// the driver's req table is the single authority (no handle-side
/// bookkeeping to race admission).
#[derive(Clone)]
pub struct EngineHandle {
    cmd: UnboundedSender<DriverCmd>,
    stats: Arc<RwLock<DriverStats>>,
}

impl EngineHandle {
    /// Submit a request; awaits the driver's admission reply (bounded by
    /// one tick — the driver drains commands before each engine tick, so
    /// worst-case latency is one tick period), then returns the request id
    /// and its event stream. A `ReqId(0)` with a closed stream means the
    /// engine refused admission (page budget with no reclaim path).
    pub async fn submit(
        &self,
        spec: RequestSpec,
    ) -> (ReqId, tokio::sync::mpsc::UnboundedReceiver<ReqEvent>) {
        let (events_tx, events_rx) = tokio::sync::mpsc::unbounded_channel();
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        if self.cmd.send(DriverCmd::Submit { spec, events: events_tx, reply: reply_tx }).is_err() {
            return (ReqId(0), events_rx);
        }
        let req = reply_rx.await.unwrap_or(ReqId(0));
        (req, events_rx)
    }

    /// Abort a request (client disconnect path).
    pub fn cancel(&self, req: ReqId) {
        let _ = self.cmd.send(DriverCmd::Cancel { req });
    }

    /// Snapshot of driver/engine telemetry (the /v1/stats source).
    pub fn stats(&self) -> DriverStats {
        self.stats.read().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// One live request on the driver side.
struct ReqCtx {
    seq: SeqId,
    events: UnboundedSender<ReqEvent>,
    /// Committed-token watermark already sent to the stream.
    sent: usize,
    /// Prompt length (usage accounting).
    prompt_tokens: usize,
    /// Radix prefix hit at admission (usage `cached_tokens`).
    cached_tokens: usize,
    /// Reported Admitted yet (first event fires post-admission).
    admitted: bool,
}

/// The engine thread's full state. Generic over the serving engine: the
/// deterministic HostEngine (mock compute over the real BatchScheduler)
/// and the CUDA engine (ferrite-serve's GpuEngine over the TpCluster)
/// both implement `ServeEngine` — the driver, the HTTP layer and the
/// event protocol are engine-agnostic.
pub struct EngineDriver<E: ServeEngine + 'static> {
    engine: E,
    cmd: UnboundedReceiver<DriverCmd>,
    stats: Arc<RwLock<DriverStats>>,
    reqs: HashMap<ReqId, ReqCtx>,
    seq_to_req: HashMap<SeqId, ReqId>,
    next_req: u64,
    ticks: u64,
    tokens_committed: u64,
    /// Reused tick plan (zero-alloc steady state — the plan buffers
    /// belong to the driver, the scheduler clears + refills them).
    plan: TickPlan,
    /// Minimum tick period (busy pacing). The real engine's tick is
    /// GPU-bound (~27ms verify); the mock tick is microseconds — without
    /// pacing it busy-spins the core. `Duration::ZERO` = full speed.
    tick_interval: std::time::Duration,
    last_tick: std::time::Instant,
}

impl EngineDriver<HostEngine> {
    /// Spawn the mock engine thread (the standalone ferrite-http binary:
    /// deterministic generation over the real scheduler — full stack on a
    /// laptop, no model files).
    pub fn spawn(cfg: SchedConfig, stop_id: u32, tick_interval: std::time::Duration) -> EngineHandle {
        let engine = HostEngine::new(cfg, stop_id).expect("host engine init");
        Self::spawn_with(engine, tick_interval)
    }
}

impl<E: ServeEngine + 'static> EngineDriver<E> {
    /// Spawn an engine thread over ANY ServeEngine (the GPU backend in
    /// ferrite-serve builds its engine and hands it here; the driver/HTTP
    /// protocol is identical). `tick_interval` = ZERO for GPU engines (the
    /// decode step is the pacing); a positive Duration paces mock engines.
    pub fn spawn_with(engine: E, tick_interval: std::time::Duration) -> EngineHandle {
        let (cmd_tx, cmd_rx) = tokio::sync::mpsc::unbounded_channel();
        let stats = Arc::new(RwLock::new(DriverStats::default()));
        let mut driver = EngineDriver {
            engine,
            cmd: cmd_rx,
            stats: stats.clone(),
            reqs: HashMap::new(),
            seq_to_req: HashMap::new(),
            next_req: 1,
            ticks: 0,
            tokens_committed: 0,
            plan: TickPlan::default(),
            tick_interval,
            last_tick: std::time::Instant::now(),
        };
        let handle = EngineHandle {
            cmd: cmd_tx,
            stats,
        };
        std::thread::Builder::new()
            .name("ferrite-engine".into())
            .spawn(move || driver.run())
            .expect("spawn engine thread");
        handle
    }

    /// The main loop: drain commands, tick, publish events; park when idle.
    fn run(&mut self) {
        loop {
            self.drain_cmds();
            let busy = self.reqs_dirty() || self.engine_busy();
            if busy {
                // Pace the busy loop (mock pacing; the real engine's
                // tick is GPU-bound and passes ZERO here).
                if !self.tick_interval.is_zero() {
                    let elapsed = self.last_tick.elapsed();
                    if elapsed < self.tick_interval {
                        std::thread::sleep(self.tick_interval - elapsed);
                    }
                }
                self.last_tick = std::time::Instant::now();
                self.tick_once();
            } else {
                // Idle: park until the next command (no busy-wait ticks;
                // a parked engine holds zero rows and an empty queue).
                match self.cmd.blocking_recv() {
                    Some(cmd) => self.handle_cmd(cmd),
                    None => return, // handle dropped — shut down
                }
                self.tick_once();
            }
        }
    }

    fn engine_busy(&self) -> bool {
        self.engine.live_rows() > 0 || self.engine.queued() > 0
    }

    fn reqs_dirty(&self) -> bool {
        // Requests still awaiting their first event (admission pending).
        self.reqs.values().any(|r| !r.admitted)
    }

    fn drain_cmds(&mut self) {
        while let Ok(cmd) = self.cmd.try_recv() {
            self.handle_cmd(cmd);
        }
    }

    fn handle_cmd(&mut self, cmd: DriverCmd) {
        match cmd {
            DriverCmd::Submit { spec, events, reply } => {
                let req = ReqId(self.next_req);
                self.next_req += 1;
                let eos = spec.stop_ids.first().copied().unwrap_or(crate::host_engine::STOP_ID);
                let prompt_tokens = spec.prompt_ids.len();
                match self.engine.submit(spec.prompt_ids, spec.max_new_tokens, eos) {
                    Ok(seq) => {
                        self.seq_to_req.insert(seq, req);
                        self.reqs.insert(
                            req,
                            ReqCtx {
                                seq,
                                events,
                                sent: 0,
                                prompt_tokens,
                                cached_tokens: 0,
                                admitted: false,
                            },
                        );
                        let _ = reply.send(req);
                    }
                    Err(e) => {
                        // Admission-refused (page budget exhausted with no
                        // reclaim path): report and close the stream.
                        let _ = events.send(ReqEvent::Finished {
                            reason: FinishReason::Cancelled,
                            usage: Usage { prompt_tokens, completion_tokens: 0, cached_tokens: 0 },
                        });
                        let _ = reply.send(ReqId(0));
                        let _ = e; // logged via stats in production wiring
                    }
                }
            }
            DriverCmd::Cancel { req } => {
                if let Some(ctx) = self.reqs.remove(&req) {
                    let reason = if self.engine.cancel(ctx.seq).unwrap_or(false) {
                        FinishReason::Cancelled
                    } else {
                        FinishReason::Cancelled // already terminal — same wire reason
                    };
                    let usage = self.usage_of(&ctx, None);
                    let _ = ctx.events.send(ReqEvent::Finished { reason, usage });
                    self.seq_to_req.remove(&ctx.seq);
                    self.engine.deregister(ctx.seq);
                }
            }
        }
    }

    /// One engine tick + event publication (the whole async-visible
    /// surface: admissions, token deltas, finishes).
    fn tick_once(&mut self) {
        self.ticks += 1;
        // Engine tick (plan → exec → ingest). Errors here are engine
        // faults — drop the affected requests' streams (Finished) rather
        // than killing the driver (the scheduler domain stays consistent;
        // production wiring would panic instead: see `main.rs` docs).
        if let Err(e) = self.engine.tick(&mut self.plan) {
            // Engine faults surface in the serve log AND to the affected
            // requests (fail_all): a silent cancelled is undebuggable.
            eprintln!("[engine] tick fault: {e}");
            self.fail_all(e.to_string());
            self.publish_stats();
            return;
        }
        // Admissions (first event per request — includes the radix hit).
        for adm in &self.plan.admissions {
            if let Some(req) = self.seq_to_req.get(&adm.seq).copied() {
                if let Some(ctx) = self.reqs.get_mut(&req) {
                    ctx.admitted = true;
                    ctx.cached_tokens = adm.prefix_hit;
                    let _ = ctx.events.send(ReqEvent::Admitted {
                        prefix_hit: adm.prefix_hit,
                        prompt_tokens: ctx.prompt_tokens,
                    });
                }
            }
        }
        // Token deltas: diff each live request's committed output.
        let live_reqs: Vec<ReqId> = self.reqs.keys().copied().collect();
        for req in live_reqs {
            let Some((seq, sent, prompt_tokens, cached)) = self
                .reqs
                .get(&req)
                .map(|c| (c.seq, c.sent, c.prompt_tokens, c.cached_tokens))
            else {
                continue;
            };
            let out = self.engine.output(seq).unwrap_or_default();
            if out.len() > sent {
                let ids = out[sent..].to_vec();
                let n = ids.len();
                if let Some(ctx) = self.reqs.get_mut(&req) {
                    ctx.sent = out.len();
                    self.tokens_committed += n as u64;
                    let _ = ctx.events.send(ReqEvent::Tokens { ids });
                }
            }
            // Retirement detection: status flips to retired when the
            // scheduler's commit math finished the request (eos/max).
            let retired = self
                .engine
                .status(seq)
                .map(|s| s == "retired")
                .unwrap_or(false);
            if retired {
                if let Some(ctx) = self.reqs.remove(&req) {
                    self.seq_to_req.remove(&seq);
                    // Finish reason: the stop marker rides the output tail
                    // (stop → `Stop`; clean cap → `Length`). The marker itself
                    // is not content: strip it from the delta the SSE sends.
                    // Read output BEFORE deregister — engines that free state
                    // on deregister (the CUDA engine drops the seq's cluster
                    // runtime) lose the final output otherwise.
                    let out = self.engine.output(seq).unwrap_or_default();
                    let stopped = out.last().map(|t| self.engine.is_stop(*t)).unwrap_or(false);
                    let reason =
                        if stopped { FinishReason::Stop } else { FinishReason::Length };
                    let completion =
                        out.len().saturating_sub(if stopped { 1 } else { 0 });
                    let _ = ctx.events.send(ReqEvent::Finished {
                        reason,
                        usage: Usage {
                            prompt_tokens,
                            completion_tokens: completion,
                            cached_tokens: cached,
                        },
                    });
                    self.engine.deregister(seq);
                }
            }
        }
        // Publish stats (once per tick — the /v1/stats source).
        self.publish_stats();
    }

    /// Snapshot engine + driver telemetry into the shared stats cell.
    /// Called from BOTH tick paths (success and fault — a fault tick
    /// still advanced the engine's admissions/partial state; /v1/stats
    /// must never report a stale world).
    fn publish_stats(&mut self) {
        let census = self.engine.cache_stats();
        let stats = DriverStats::from_census(
            census,
            self.ticks,
            self.tokens_committed,
            self.engine.live_rows(),
            self.engine.queued(),
        );
        *self.stats.write().unwrap_or_else(|e| e.into_inner()) = stats;
    }

    fn usage_of(&self, ctx: &ReqCtx, extra: Option<usize>) -> Usage {
        let completion = self
            .engine
            .output(ctx.seq)
            .map(|o| o.len())
            .unwrap_or(0)
            + extra.unwrap_or(0);
        Usage {
            prompt_tokens: ctx.prompt_tokens,
            completion_tokens: completion,
            cached_tokens: ctx.cached_tokens,
        }
    }

    fn fail_all(&mut self, msg: String) {
        let entries: Vec<(ReqId, ReqCtx)> = self.reqs.drain().collect();
        self.seq_to_req.clear();
        for (req, ctx) in entries {
            let _ = ctx.events.send(ReqEvent::Finished {
                reason: FinishReason::Cancelled,
                usage: Usage {
                    prompt_tokens: ctx.prompt_tokens,
                    completion_tokens: ctx.sent,
                    cached_tokens: ctx.cached_tokens,
                },
            });
            let _ = self.engine.cancel(ctx.seq);
            let _ = msg; // production wiring logs per-request faults
            let _ = req;
        }
    }
}
