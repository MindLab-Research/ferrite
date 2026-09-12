//! Single-flight step engines: one request at a time, one token per step.
//!
//! ## Why this exists
//!
//! `ServeEngine` (`engine.rs`) is the scheduler-shaped seam: submit many
//! sequences, tick a scheduler, read each live sequence's committed output.
//! Some backends are structurally different. A lockstep multi-rank pool —
//! deepseek-v4.1's TP8 chain, and in general any model whose runtime is a
//! `feed(token, pos) -> next token` chain rather than a batching scheduler —
//! can run exactly ONE request at a time: every rank executes the same step
//! (the collective IS the loop), and the KV state is per-sequence. That shape
//! is not GLM-specific and will recur for every single-stream checkpoint, so
//! it lives here instead of being re-implemented per model: implement
//! `StepEngine` (three methods) and `SingleFlight` supplies the rest —
//! FIFO admission, the batch-1 constraint, token accounting, retirement and
//! cancel — behind the SAME `ServeEngine` seam. The driver thread, the
//! HTTP/SSE layer, usage and stats are then the identical code GLM serves
//! through.
//!
//! ```text
//!   SingleFlight<E: StepEngine>              the engine (any single-stream backend)
//!   ── submit ──▶ FIFO queue       (batch = 1: the lockstep constraint, declared here)
//!   ── tick ────▶ admit  : prefill (reset + consume the prompt) → first token
//!                 decode : ONE step per tick; the output grows by one token
//!   ── cancel ──▶ drop the slot     (the next prefill resets the runtime)
//!   ── retire ──▶ stop token on the output tail (driver strips it, wire
//!                 finish_reason = stop) or max_new tokens reached (length)
//! ```
//!
//! ## Faults
//!
//! A lockstep pool cannot resume after a broken step (its ranks may be
//! scattered inside a collective), so a step failure POISONS the adapter:
//! every later `submit`/`tick` fails fast, and the request that faulted is
//! retired so the driver publishes a terminal event instead of hanging the
//! client. Restarting the process is the recovery path.

use std::collections::VecDeque;

use ferrite_dispatch::arena::{SeqId, SeqTag, TypedArena};
use ferrite_dispatch::batch::{Admission, CacheStats, TickPlan};
use ferrite_types::{FerriteError, Result};

use crate::engine::ServeEngine;

/// The minimal contract for a single-flight step engine.
///
/// Positions are the engine's own (a lockstep chain's KV/rope state is
/// per-sequence, so the caller must feed the position explicitly): the adapter
/// tracks them (`prompt_len + generated - 1`).
pub trait StepEngine: Send {
    /// Start a fresh sequence: reset the runtime and consume the WHOLE prompt
    /// (a per-token forward for chains whose KV is per-sequence), returning the
    /// first generated token (the argmax after the last prompt token).
    /// Called once per request.
    fn prefill(&mut self, prompt: &[u32]) -> Result<u32>;

    /// One steady-state decode step: feed `token` at `pos`, return the next
    /// sampled token.
    fn decode(&mut self, token: u32, pos: usize) -> Result<u32>;

    /// Stop-set membership (the engine's own turn-end tokens).
    fn is_stop(&self, token: u32) -> bool;

    /// The primary stop id (telemetry).
    fn stop_id(&self) -> u32;

    /// Prompt + `max_new` capacity bound (KV). `submit` refuses over it, so an
    /// over-long request fails at admission instead of mid-decode.
    fn max_ctx(&self) -> usize {
        usize::MAX
    }

    /// How many prompt tokens the MOST RECENT `prefill` served from a cached
    /// prefix instead of computing them (0 when there is no cache, it missed,
    /// or the backend has none — the honest default).
    ///
    /// Read once per admission, so the adapter can report it as
    /// `Admission::prefix_hit` → `usage.prompt_tokens_details.cached_tokens`
    /// without the engine having to know how that travels.
    fn resume_hit(&self) -> usize {
        0
    }
}

/// One request's state while it lives on the engine.
struct Live {
    /// The prompt, held until admission runs the prefill.
    prompt: Vec<u32>,
    prompt_len: usize,
    max_new: usize,
    /// Generated tokens (incl. the stop token when the stream stopped on one).
    out: Vec<u32>,
    /// The last sampled token (what the next decode step feeds).
    next: u32,
    /// Terminal — the driver reads `out` once, then deregisters.
    retired: bool,
}

/// `StepEngine` → `ServeEngine`: batch-1 admission + per-tick decode.
pub struct SingleFlight<E: StepEngine> {
    engine: E,
    arena: TypedArena<SeqTag, Live>,
    /// Awaiting admission, FIFO (the batch-1 constraint: a second request waits
    /// for the first to retire).
    queue: VecDeque<SeqId>,
    /// The one in-flight request.
    live: Option<SeqId>,
    ticks: u64,
    tokens: u64,
    /// First step fault — the adapter refuses further work (see module doc).
    fault: Option<String>,
}

impl<E: StepEngine> SingleFlight<E> {
    pub fn new(engine: E) -> Self {
        SingleFlight {
            engine,
            arena: TypedArena::with_capacity(8),
            queue: VecDeque::new(),
            live: None,
            ticks: 0,
            tokens: 0,
            fault: None,
        }
    }

    /// Start the queued request: the engine's prefill consumes the prompt and
    /// returns the first generated token.
    fn admit(&mut self, seq: SeqId) -> Result<()> {
        let Some(l) = self.arena.get(seq) else {
            return Ok(());
        };
        let prompt = l.prompt.clone();
        let first = self.engine.prefill(&prompt)?;
        let retired = {
            let Some(l) = self.arena.get_mut(seq) else {
                return Ok(());
            };
            let stopped = self.engine.is_stop(first);
            l.prompt = Vec::new(); // the KV holds it now
            l.out.push(first);
            l.next = first;
            let retired = stopped || l.out.len() >= l.max_new;
            if retired {
                l.retired = true;
            }
            retired
        };
        self.tokens += 1;
        self.live = if retired { None } else { Some(seq) };
        Ok(())
    }

    /// One decode step on the live sequence: feed the last sampled token,
    /// append the next, retire on a stop token / `max_new`.
    fn step_once(&mut self, seq: SeqId) -> Result<()> {
        let (token, pos, max_new) = match self.arena.get(seq) {
            // pos: the token produced by prefill sits at prompt_len (the chain
            // advanced one position per step, prompt included)
            Some(l) => (l.next, l.prompt_len + l.out.len() - 1, l.max_new),
            None => return Ok(()),
        };
        let next = self.engine.decode(token, pos)?;
        let retired = {
            let Some(l) = self.arena.get_mut(seq) else {
                return Ok(());
            };
            let stopped = self.engine.is_stop(next);
            l.out.push(next);
            l.next = next;
            let retired = stopped || l.out.len() >= max_new;
            if retired {
                l.retired = true;
            }
            retired
        };
        self.tokens += 1;
        if retired {
            self.live = None;
        }
        Ok(())
    }

    /// Park the adapter after a step fault: retire whatever was in flight (the
    /// driver publishes its terminal event) and refuse all further work.
    fn fail(&mut self, seq: Option<SeqId>, e: FerriteError) {
        eprintln!("[single-flight] engine fault: {e}");
        if self.fault.is_none() {
            self.fault = Some(e.to_string());
        }
        for s in seq.into_iter().chain(self.live.take()) {
            if let Some(l) = self.arena.get_mut(s) {
                l.retired = true;
            }
        }
        self.queue.clear();
    }
}

impl<E: StepEngine + 'static> ServeEngine for SingleFlight<E> {
    fn submit(&mut self, prompt_ids: Vec<u32>, max_new_tokens: usize, _eos: u32) -> Result<SeqId> {
        if let Some(f) = &self.fault {
            return Err(FerriteError::Config(format!("engine faulted: {f}")));
        }
        if prompt_ids.is_empty() {
            return Err(FerriteError::InvalidArg("empty prompt".into()));
        }
        let max_ctx = self.engine.max_ctx();
        if prompt_ids.len() + max_new_tokens > max_ctx {
            return Err(FerriteError::InvalidArg(format!(
                "context too long: prompt {} + max_new {} > {max_ctx} (engine KV bound)",
                prompt_ids.len(),
                max_new_tokens
            )));
        }
        let prompt_len = prompt_ids.len();
        let seq = self.arena.insert(Live {
            prompt: prompt_ids,
            prompt_len,
            max_new: max_new_tokens.max(1),
            out: Vec::new(),
            next: 0,
            retired: false,
        });
        self.queue.push_back(seq);
        Ok(seq)
    }

    fn tick(&mut self, plan: &mut TickPlan) -> Result<()> {
        self.ticks += 1;
        // The plan belongs to the driver; only admissions are ours to fill.
        plan.admissions.clear();
        if let Some(f) = &self.fault {
            // Already poisoned: fail fast (the driver's fault path retires
            // whatever is still pending).
            return Err(FerriteError::Config(format!("engine faulted: {f}")));
        }
        // 1. Admission — one request at a time (batch = 1: the lockstep
        //    constraint; extra submits wait FIFO in `queue`).
        if self.live.is_none() {
            if let Some(seq) = self.queue.pop_front() {
                match self.admit(seq) {
                    Ok(()) => plan.admissions.push(Admission {
                        seq,
                        row: 0,
                        // what the admission just served from a prefix cache (0 for
                        // a backend without one) — this is the number the wire
                        // reports as `cached_tokens`.
                        prefix_hit: self.engine.resume_hit(),
                    }),
                    Err(e) => self.fail(Some(seq), e),
                }
            }
        }
        // 2. Decode — one step per tick (the step IS the pacing: the driver
        //    runs with tick_interval ZERO).
        if let Some(seq) = self.live {
            if let Err(e) = self.step_once(seq) {
                self.fail(Some(seq), e);
            }
        }
        Ok(())
    }

    fn output(&self, seq: SeqId) -> Result<Vec<u32>> {
        self.arena
            .get(seq)
            .map(|l| l.out.clone())
            .ok_or_else(|| FerriteError::InvalidArg("no such seq".into()))
    }

    fn cancel(&mut self, seq: SeqId) -> Result<bool> {
        let was_live = self.live == Some(seq) || self.queue.contains(&seq);
        self.queue.retain(|s| *s != seq);
        if self.live == Some(seq) {
            self.live = None;
        }
        // Drop the slot outright: the driver reads `output` BEFORE this call
        // for a live request, and the next prefill resets the engine's state,
        // so there is nothing left to preserve. (No `deregister` follows on the
        // driver's fault path, hence the removal here too.)
        self.arena.remove(seq);
        Ok(was_live)
    }

    fn deregister(&mut self, seq: SeqId) {
        self.queue.retain(|s| *s != seq);
        if self.live == Some(seq) {
            self.live = None;
        }
        self.arena.remove(seq);
    }

    fn status(&self, seq: SeqId) -> Option<&'static str> {
        match self.arena.get(seq) {
            Some(l) if l.retired => Some("retired"),
            Some(_) => Some("live"),
            None => None,
        }
    }

    fn live_rows(&self) -> usize {
        usize::from(self.live.is_some())
    }

    fn queued(&self) -> usize {
        self.queue.len()
    }

    fn cache_stats(&self) -> CacheStats {
        // No radix / hicache on a single-flight engine yet.
        CacheStats {
            tree_nodes: 0,
            tree_blocks: 0,
            evictable_tokens: 0,
            protected_tokens: 0,
            tier_census: (0, 0, 0),
            pages_in_use: 0,
            pages_free: 0,
            hits: 0,
            misses: 0,
        }
    }

    fn stop_id(&self) -> u32 {
        self.engine.stop_id()
    }

    fn is_stop(&self, t: u32) -> bool {
        self.engine.is_stop(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy single-stream engine: token[n] = 100 + n, with a configurable stop
    /// id and a record of the positions it was fed (the batch-1 + position
    /// contract is what SingleFlight must get right).
    struct Mock {
        stop: u32,
        positions: Vec<usize>,
        steps: usize,
    }

    impl Mock {
        fn new(stop: u32) -> Self {
            Mock { stop, positions: Vec::new(), steps: 0 }
        }
    }

    impl StepEngine for Mock {
        fn prefill(&mut self, prompt: &[u32]) -> Result<u32> {
            self.steps = 0;
            Ok(100)
        }
        fn decode(&mut self, token: u32, pos: usize) -> Result<u32> {
            self.steps += 1;
            self.positions.push(pos);
            if token + 1 == self.stop {
                Ok(self.stop)
            } else {
                Ok(token + 1)
            }
        }
        fn is_stop(&self, token: u32) -> bool {
            token == self.stop
        }
        fn stop_id(&self) -> u32 {
            self.stop
        }
        fn max_ctx(&self) -> usize {
            64
        }
    }

    /// Drain the driver's view of the engine: tick until the request retires.
    fn run_until_retired(e: &mut SingleFlight<Mock>, seq: SeqId) -> Vec<u32> {
        let mut plan = TickPlan::default();
        for _ in 0..1024 {
            e.tick(&mut plan).expect("tick");
            if e.status(seq) == Some("retired") {
                return e.output(seq).expect("output");
            }
        }
        panic!("never retired");
    }

    #[test]
    fn stop_token_rides_the_tail() {
        let mut e = SingleFlight::new(Mock::new(103));
        let seq = e.submit(vec![1, 2, 3], 16, 103).expect("submit");
        let out = run_until_retired(&mut e, seq);
        // 100, 101, 102, then the stop marker — the driver strips the tail.
        assert_eq!(out, vec![100, 101, 102, 103]);
        assert!(e.is_stop(*out.last().unwrap()));
        // decode positions continue from prompt_len (3 here)
        assert_eq!(e.engine.positions, vec![3, 4, 5]);
    }

    #[test]
    fn max_new_retires_without_a_stop_token() {
        let mut e = SingleFlight::new(Mock::new(999_999));
        let seq = e.submit(vec![1, 2, 3], 3, 999_999).expect("submit");
        let out = run_until_retired(&mut e, seq);
        assert_eq!(out, vec![100, 101, 102]);
        assert!(!e.is_stop(*out.last().unwrap()));
    }

    #[test]
    fn batch_is_one_and_the_queue_is_fifo() {
        let mut e = SingleFlight::new(Mock::new(102));
        let a = e.submit(vec![1], 8, 102).expect("submit a");
        let b = e.submit(vec![2], 8, 102).expect("submit b");
        let mut plan = TickPlan::default();
        // A is admitted first; B only when A retires.
        let mut admissions = Vec::new();
        for _ in 0..64 {
            e.tick(&mut plan).expect("tick");
            admissions.extend(plan.admissions.iter().map(|a| a.seq));
            if e.live_rows() == 0 && e.queued() == 0 {
                break;
            }
        }
        assert_eq!(admissions, vec![a, b]);
        assert_eq!(e.output(a).unwrap(), vec![100, 101, 102]);
        assert_eq!(e.output(b).unwrap(), vec![100, 101, 102]);
    }

    #[test]
    fn cancel_drops_the_slot_and_the_engine_reports_it() {
        let mut e = SingleFlight::new(Mock::new(999_999));
        let seq = e.submit(vec![1], 8, 999_999).expect("submit");
        assert!(e.cancel(seq).expect("cancel"));
        assert_eq!(e.status(seq), None);
        assert_eq!(e.live_rows(), 0);
        assert_eq!(e.queued(), 0);
        // A fresh request is admitted after the cancel (the next prefill resets
        // the engine, so the runtime is reusable).
        let next = e.submit(vec![2], 2, 999_999).expect("submit after cancel");
        assert_eq!(run_until_retired(&mut e, next).len(), 2);
    }

    #[test]
    fn over_long_prompts_are_refused_at_submit() {
        let mut e = SingleFlight::new(Mock::new(1));
        assert!(e.submit(vec![0; 64], 1, 1).is_err());
        assert!(e.submit(Vec::new(), 1, 1).is_err());
    }

    #[test]
    fn a_step_fault_poisons_the_engine_and_retires_the_request() {
        struct Broken;
        impl StepEngine for Broken {
            fn prefill(&mut self, _prompt: &[u32]) -> Result<u32> {
                Err(FerriteError::Config("rank 3 died".into()))
            }
            fn decode(&mut self, _token: u32, _pos: usize) -> Result<u32> {
                Ok(0)
            }
            fn is_stop(&self, _token: u32) -> bool {
                false
            }
            fn stop_id(&self) -> u32 {
                0
            }
        }
        let mut e = SingleFlight::new(Broken);
        let seq = e.submit(vec![1], 4, 0).expect("submit");
        let mut plan = TickPlan::default();
        e.tick(&mut plan).expect("fault tick still publishes");
        // The request is retired (the driver sends a terminal event) and the
        // engine refuses new work — a lockstep pool cannot resume.
        assert_eq!(e.status(seq), Some("retired"));
        assert!(e.submit(vec![2], 1, 0).is_err());
    }
}
