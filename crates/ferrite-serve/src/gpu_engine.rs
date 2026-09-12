//! GpuEngine: the CUDA TpCluster behind the ServeEngine seam.
//!
//! The HTTP layer (ferrite-http: axum routes + SSE + the driver thread)
//! speaks `ServeEngine`; this impl drives the real GLM-5.3-Flash TP4
//! cluster — per-seq prefill (`prefill_chunk`) + per-seq mega-graph
//! decode (`decode_step`) in a round-robin. Concurrency today is
//! INTERLEAVED decode (each tick = one decode step per live seq, each
//! ~20ms of graph replay on the GPU): N live requests each get ~1/N of
//! the single-stream token rate; true batched decode (the [B,...] state
//! row engine the scheduler's ExecBackend targets) is the next phase —
//! this seam is where it lands.
//!
//! Per-seq GPU state lifecycle: every request mints a fresh cluster seq
//! (u64 counter — never reused), and `deregister`/`cancel` release its
//! DSA caches + GDN states + mega graphs via `TpCluster::free_seq`
//! (~GBs per seq — without it the serve OOMs after a handful of
//! requests).
//!
//! MTP constraint: `MtpState` is a per-rank SINGLETON (the MTP verify
//! path's ping-pong scratch) — with FERRITE_MTP=1 the engine forces
//! max_seqs=1 (multi-seq MTP corrupts the shared scratch; lifting this
//! needs per-seq MtpState buffers).

#![cfg(feature = "cuda")]

use std::collections::VecDeque;

use ferrite_dispatch::arena::{SeqId, SeqTag, TypedArena};
use ferrite_dispatch::batch::{Admission, CacheStats, TickPlan};
use ferrite_exec::tp::TpCluster;
use ferrite_http::engine::ServeEngine;
use ferrite_kernel::CudaBackend;
use ferrite_types::{FerriteError, Result};

/// DSA cache allocation bound (ferrite-kernel's max_tokens per family) —
/// prompt + generation must stay under it.
const MAX_CTX: usize = 8100;

/// Wave-4 P0-B: the chunked-prefill budget (prompt tokens per
/// `prefill_chunk` call). `0` (or unset) keeps the historical whole-segment
/// call, so the two granularities can be A/B'd on the same binary. Read once
/// and cached — the house rule for every hot-path gate.
fn prefill_token_budget() -> usize {
    static F: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("FERRITE_PREFILL_BUDGET")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
    })
}

/// One live/queued request on the engine side.
struct GpuSeq {
    /// The cluster-side sequence id (u64 counter, never reused).
    cluster_seq: u64,
    /// Prompt token ids (held until admission runs the prefill).
    prompt: Vec<u32>,
    prompt_len: usize,
    max_new: usize,
    /// rt.tokens watermark — the incremental read per tick.
    prev_len: usize,
    /// Retirement flag (status() → "retired"; the driver reads the final
    /// output once, then deregisters).
    retired: bool,
    /// GPU state released (idempotence guard for cancel→deregister).
    freed: bool,
    /// Final output snapshot (taken at retirement/cancel — the cluster
    /// runtime is freed, later output() reads serve from here).
    final_out: Option<Vec<u32>>,
}

pub struct GpuEngine {
    cluster: TpCluster<CudaBackend>,
    arena: TypedArena<SeqTag, GpuSeq>,
    /// Admitted (prefill done) — the round-robin decode set.
    live: Vec<SeqId>,
    /// Awaiting admission (prefill) — FIFO.
    queue: VecDeque<SeqId>,
    next_cluster: u64,
    /// Full stop-token set (the tokenizer's specials; retirement check on
    /// the incremental stream tail).
    stops: Vec<u32>,
    max_seqs: usize,
    ticks: u64,
    /// The current batched-decode graph's composition name
    /// ("megab_{s1}_{s2}..."). A membership change (admission / retirement /
    /// cancel) destroys the old graph — its captured kernel args embed the
    /// member seqs' per-seq state pointers, which free_seq releases. The
    /// next tick captures fresh for the new composition (~1-2s, amortized
    /// over 1000-token streams).
    batch_graph: Option<String>,
}

impl GpuEngine {
    pub fn new(cluster: TpCluster<CudaBackend>, stops: Vec<u32>, mut max_seqs: usize) -> Self {
        // MTP: MtpState (verify ping-pong scratch, hf_v/hprev) is a
        // per-rank singleton — multi-seq corrupts it. Single-seq only
        // until the scratch is per-seq.
        if std::env::var_os("FERRITE_MTP").is_some() && max_seqs > 1 {
            eprintln!(
                "[serve] FERRITE_MTP=1 with max_seqs={max_seqs}: forcing 1 (MtpState is a per-rank singleton — multi-seq MTP is unsafe)"
            );
            max_seqs = 1;
        }
        eprintln!(
            "[serve] GpuEngine: max_seqs={max_seqs} stops={stops:?} (per-seq state ~GBs; free at retire)"
        );
        GpuEngine {
            cluster,
            arena: TypedArena::with_capacity(64),
            live: Vec::new(),
            queue: VecDeque::new(),
            next_cluster: 1,
            stops,
            max_seqs,
            ticks: 0,
            batch_graph: None,
        }
    }

    /// The incremental output of a live seq (post-prompt tokens).
    fn incremental(&self, cluster_seq: u64, prompt_len: usize) -> Vec<u32> {
        self.cluster
            .shards
            .first()
            .and_then(|s| s.seq_runtime(cluster_seq))
            .map(|rt| rt.tokens[prompt_len.min(rt.tokens.len())..].to_vec())
            .unwrap_or_default()
    }

    /// Release the seq's GPU state + drop the arena entry (idempotent).
    fn free(&mut self, seq: SeqId) {
        // KEEP the per-size batched graphs across retires: the kernel args
        // reference the per-size POINTER TABLES (content-refreshed each
        // replay), not embedded seq pointers — a retired seq's slots are
        // simply overwritten on the next refresh. Destroying here forced a
        // 1-2s re-capture every membership change.
        // (free_seq still releases the seq's own GDN states / DSA caches.)
        // Read the params immutably, snapshot the output (needs &self),
        // then free the GPU state + update the arena (&mut self) —
        // sequenced to avoid the borrow conflict.
        let params = self
            .arena
            .get(seq)
            .filter(|g| !g.freed)
            .map(|g| (g.cluster_seq, g.prompt_len));
        if let Some((cluster_seq, prompt_len)) = params {
            let snapshot = self.incremental(cluster_seq, prompt_len);
            if let Some(g) = self.arena.get_mut(seq) {
                g.freed = true;
                if g.final_out.is_none() {
                    g.final_out = Some(snapshot);
                }
            }
            self.cluster.free_seq(cluster_seq);
            // DIAGNOSTIC (FERRITE_DESTROY_BG=1): the doc comment on
            // destroy_batch_graph says a retire MUST destroy the batched graphs
            // because their recorded kernel args reference the freed per-seq
            // state — but nothing calls it. Force-destroy them here to test
            // whether the first replay after a retire is what faults.
            if std::env::var_os("FERRITE_DESTROY_BG").is_some() {
                for sz in [1usize, 2, 4, 8, 16, 32, 64] {
                    self.cluster.destroy_batch_graph(&format!("megab_b{sz}"));
                }
            }
        }
        self.arena.remove(seq);
    }
}

impl ServeEngine for GpuEngine {
    fn submit(
        &mut self,
        prompt_ids: Vec<u32>,
        max_new_tokens: usize,
        _eos: u32,
    ) -> Result<SeqId> {
        if prompt_ids.len() + max_new_tokens > MAX_CTX {
            return Err(FerriteError::InvalidArg(format!(
                "context too long: prompt {} + max_new {} > {MAX_CTX} (DSA cache bound)",
                prompt_ids.len(),
                max_new_tokens
            )));
        }
        if prompt_ids.is_empty() {
            return Err(FerriteError::InvalidArg("empty prompt".into()));
        }
        let cluster_seq = self.next_cluster;
        self.next_cluster += 1;
        let prompt_len = prompt_ids.len();
        let seq = self.arena.insert(GpuSeq {
            cluster_seq,
            prompt: prompt_ids,
            prompt_len,
            max_new: max_new_tokens,
            prev_len: 0,
            retired: false,
            freed: false,
            final_out: None,
        });
        self.queue.push_back(seq);
        Ok(seq)
    }

    fn tick(&mut self, plan: &mut TickPlan) -> Result<()> {
        self.ticks += 1;
        // TICK TIMING (FERRITE_TIMING): the server-side total step time —
        // compare against the [megab] replay median to see whether the
        // ~0.9ms host gap is server-side (tick > replay → pipelining helps)
        // or client-side Python SSE parsing (tick ≈ replay → pipelining
        // is pointless).
        let tick_start = std::time::Instant::now();
        // FERRITE_NCU serve window (nsys --capture-range=cudaProfilerApi):
        // open the capture only once the batch SATURATES (live == max_seqs) —
        // skips the 80s weight load AND the admission ramp / per-size graph
        // captures; closed by run_serve's profiler_stop before exit (nsys
        // waits for cudaProfilerStop forever without it — the documented
        // "serve never exits" trap).
        if std::env::var_os("FERRITE_NCU").is_some() && self.live.len() >= self.max_seqs {
            static NCU_WIN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !NCU_WIN.swap(true, std::sync::atomic::Ordering::AcqRel) {
                eprintln!(
                    "[ncu-win] batch saturated (live={}): opening the profiler window",
                    self.live.len()
                );
                #[cfg(feature = "cuda")]
                ferrite_kernel::cuda::profiler_start();
            }
        }
        plan.admissions.clear();
        // 1. Admission — ONE prefill per tick (prefill is a blocking
        //    host-chain forward, ~0.4-2s for chat prompts; one-at-a-time
        //    bounds the stall the live decode set sees).
        if !self.queue.is_empty() && self.live.len() < self.max_seqs {
            let seq = self.queue.pop_front().expect("checked non-empty");
            // None → cancelled while queued (free already ran): drop silently.
            if let Some(g) = self.arena.get(seq) {
                let (cluster_seq, prompt) = (g.cluster_seq, g.prompt.clone());
                let t0 = std::time::Instant::now();
                // P0-B (Wave 4): feed the prompt in chunks instead of one whole
                // segment. The engine is n-variable already — `prefill_chunk`
                // accumulates the KV across successive calls on the same seq, so
                // the ONLY thing that changes is the caller's granularity. The
                // chunked path is the prerequisite for every 1M-context item
                // (bounded per-call workspace, indexer chunking, CP). `0` keeps
                // the historical whole-segment call so the two can be A/B'd.
                let budget = prefill_token_budget();
                if budget == 0 || prompt.len() <= budget {
                    self.cluster.prefill_chunk(cluster_seq, &prompt)?;
                } else {
                    for chunk in prompt.chunks(budget) {
                        self.cluster.prefill_chunk(cluster_seq, chunk)?;
                    }
                }
                if let Some(g) = self.arena.get_mut(seq) {
                    g.prev_len = g.prompt_len; // rt.tokens = prompt after prefill
                }
                self.live.push(seq);
                eprintln!(
                    "[serve] admitted seq {seq:?} cluster={cluster_seq} prompt={} prefill={:.2}s budget={} (live={} queued={})",
                    prompt.len(),
                    t0.elapsed().as_secs_f32(),
                    if budget == 0 { 0 } else { budget },
                    self.live.len(),
                    self.queue.len()
                );
                plan.admissions.push(Admission { seq, row: 0, prefix_hit: 0 });
            }
        }
        // 2. Decode — the TRUE BATCHED path (non-MTP): ONE graph step for
        //    ALL live seqs. The projections run at n=B GEMM (weights stream
        //    once per step for all B rows — the batched-GEMM directive);
        //    the per-seq recurrent state ops (GDN conv/state, DSA caches) run
        //    as B × n=1 in-graph launches with each row's own state
        //    pointers. Composition change (admission/retirement) re-captures
        //    (~1-2s, amortized over 1000-token streams).
        //    MTP: the per-seq round-robin (MtpState is a per-rank singleton —
        //    the batched MTP is Step B; FERRITE_MTP already forces max_seqs=1
        //    so this branch degenerates to a single live seq).
        let mtp_mode = std::env::var_os("FERRITE_MTP").is_some();
        let mut retired: Vec<SeqId> = Vec::new();
        if !self.live.is_empty() {
            if mtp_mode {
                // per-seq round-robin (the legacy single-seq path — MTP's
                // single-seq constraint; decode_step handles mega/MTP)
                for i in 0..self.live.len() {
                    let seq = self.live[i];
                    let (cluster_seq, prompt_len, max_new, prev_len) = match self.arena.get(seq) {
                        Some(g) => (g.cluster_seq, g.prompt_len, g.max_new, g.prev_len),
                        None => continue,
                    };
                    self.cluster.decode_step(cluster_seq)?;
                    let rt_len = self
                        .cluster
                        .shards
                        .first()
                        .and_then(|s| s.seq_runtime(cluster_seq))
                        .map(|rt| rt.tokens.len())
                        .unwrap_or(0);
                    let stopped = rt_len > prev_len
                        && self
                            .cluster
                            .shards
                            .first()
                            .and_then(|s| s.seq_runtime(cluster_seq))
                            .map(|rt| rt.tokens[prev_len..].iter().any(|t| self.stops.contains(t)))
                            .unwrap_or(false);
                    if let Some(g) = self.arena.get_mut(seq) {
                        g.prev_len = rt_len;
                    }
                    let generated = rt_len.saturating_sub(prompt_len);
                    if stopped || generated >= max_new {
                        let snapshot = self.incremental(cluster_seq, prompt_len);
                        if let Some(g) = self.arena.get_mut(seq) {
                            g.retired = true;
                            g.final_out = Some(snapshot);
                        }
                        retired.push(seq);
                    }
                }
            } else {
                // BATCHED: one decode_step_batched for the whole live set —
                // the graph composition is the ordered live cluster seqs.
                let live_seqs: Vec<u64> = self
                    .live
                    .iter()
                    .filter_map(|seq| self.arena.get(*seq).map(|g| g.cluster_seq))
                    .collect();
                if live_seqs.len() == 1 && std::env::var_os("FERRITE_FORCE_BATCHED_B1").is_none() {
                    // SINGLE seq: the per-seq mega (GEMV) path. The batched
                    // B-row GEMM graph was 1.9x SLOWER at B=1 (measured
                    // [megab] replay 17.95ms vs [mega] 9.55ms). The
                    // gdn_layer_dev_batched n==1 alignment (fused
                    // gemv_tri/gemv_qkv_conv) closes most of that gap —
                    // FERRITE_FORCE_BATCHED_B1=1 re-tests the batched path.
                    // KEEP the per-size batched graphs (the tables are
                    // content-refreshed, no embedded seq pointers) —
                    // destroying here forces a 1-2s re-capture when
                    // concurrency returns.
                    self.cluster.decode_step(live_seqs[0])?;
                } else {
                    // SGLang-style batch-size keying: tp.rs pads to
                    // 1/2/4/8/16/32 and captures ONE graph per padded size,
                    // so a membership change REUSES the graph (the per-size
                    // pointer tables' content is refreshed inside
                    // decode_step_batched) — no re-capture.
                    let size = [1usize, 2, 4, 8, 16, 32]
                        .iter()
                        .copied()
                        .find(|&s| s >= live_seqs.len())
                        .unwrap_or(live_seqs.len());
                    let batch_name = format!("megab_b{size}");
                    self.batch_graph = Some(batch_name);
                    self.cluster.decode_step_batched(&live_seqs)?;
                }
                // per-seq retirement checks (the incremental reads — same
                // logic as the per-seq loop, minus the decode_step call)
                for i in 0..self.live.len() {
                    let seq = self.live[i];
                    let (cluster_seq, prompt_len, max_new, prev_len) = match self.arena.get(seq) {
                        Some(g) => (g.cluster_seq, g.prompt_len, g.max_new, g.prev_len),
                        None => continue,
                    };
                    let rt_len = self
                        .cluster
                        .shards
                        .first()
                        .and_then(|s| s.seq_runtime(cluster_seq))
                        .map(|rt| rt.tokens.len())
                        .unwrap_or(0);
                    let stopped = rt_len > prev_len
                        && self
                            .cluster
                            .shards
                            .first()
                            .and_then(|s| s.seq_runtime(cluster_seq))
                            .map(|rt| rt.tokens[prev_len..].iter().any(|t| self.stops.contains(t)))
                            .unwrap_or(false);
                    if let Some(g) = self.arena.get_mut(seq) {
                        g.prev_len = rt_len;
                    }
                    let generated = rt_len.saturating_sub(prompt_len);
                    if stopped || generated >= max_new {
                        let snapshot = self.incremental(cluster_seq, prompt_len);
                        if let Some(g) = self.arena.get_mut(seq) {
                            g.retired = true;
                            g.final_out = Some(snapshot);
                        }
                        retired.push(seq);
                    }
                }
            }
        }
        for seq in retired {
            self.live.retain(|s| *s != seq);
        }
        if std::env::var_os("FERRITE_TIMING").is_some() {
            eprintln!("[tick] total: {:.2}ms (live={})", tick_start.elapsed().as_secs_f64() * 1e3, self.live.len());
        }
        Ok(())
    }

    fn output(&self, seq: SeqId) -> Result<Vec<u32>> {
        match self.arena.get(seq) {
            // Retired/cancelled: the frozen snapshot (the cluster runtime
            // may be freed already).
            Some(g) if g.final_out.is_some() => Ok(g.final_out.clone().expect("checked")),
            Some(g) => Ok(self.incremental(g.cluster_seq, g.prompt_len)),
            None => Err(FerriteError::InvalidArg("no such seq".into())),
        }
    }

    fn cancel(&mut self, seq: SeqId) -> Result<bool> {
        let was_live = self.live.contains(&seq) || self.queue.contains(&seq);
        self.live.retain(|s| *s != seq);
        self.queue.retain(|s| *s != seq);
        if self.arena.get(seq).is_some() {
            self.free(seq);
        }
        Ok(was_live)
    }

    fn deregister(&mut self, seq: SeqId) {
        self.live.retain(|s| *s != seq);
        self.queue.retain(|s| *s != seq);
        self.free(seq);
    }

    fn status(&self, seq: SeqId) -> Option<&'static str> {
        match self.arena.get(seq) {
            Some(g) if g.retired => Some("retired"),
            Some(_) => Some("live"),
            None => None,
        }
    }

    fn live_rows(&self) -> usize {
        self.live.len()
    }

    fn queued(&self) -> usize {
        self.queue.len()
    }

    fn cache_stats(&self) -> CacheStats {
        // No radix/hicache on this engine yet (the scheduler's prefix
        // cache lands with the batched row engine — the ExecBackend seam).
        CacheStats {
            tree_nodes: 0,
            tree_blocks: 0,
            evictable_tokens: 0,
            protected_tokens: 0,
            tier_census: (0, 0, 0),
            pages_in_use: 0,
            pages_free: 0,
        }
    }

    fn stop_id(&self) -> u32 {
        self.stops.first().copied().unwrap_or(154_820)
    }

    /// The FULL stop set decides the finish reason: the model's turn-end
    /// token (augu id 154827 — not the primary <|end|> 154820) must label
    /// genuine stops as "stop", not "length".
    fn is_stop(&self, t: u32) -> bool {
        self.stops.contains(&t)
    }
}
