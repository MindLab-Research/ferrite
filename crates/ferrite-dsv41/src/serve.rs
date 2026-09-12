//! DeepSeek-V4.1-Flash serve assembly: the TP rank pool + the shared stack.
//!
//! `--serve` mode used to be fenced inside the `dsv41-run` binary. It is
//! assembly, not model code, so it lives here (a lib module) and both callers
//! — the `dsv41-run` runner and the unified `ferrite-serve --model dsv41`
//! binary — build the SAME engine through [`build_serve_engine`].
//!
//! The only DSV41-specific pieces are:
//!
//!   * [`TpRankPool`] — the TP rank threads behind `StepEngine`: ONE request at
//!     a time, every rank running the same step (the collective is the loop).
//!     ferrite-http's `SingleFlight` adapter turns that into a full
//!     `ServeEngine` (FIFO admission, retirement, cancel, telemetry), so the
//!     driver/HTTP/SSE layer is identical to GLM's;
//!   * [`crate::frame::Dsv41Frame`] — this checkpoint's chat frame + stop set.
//!
//! Everything else (request/event protocol, SSE framing, usage, cancel-on-drop,
//! /v1/models, /health, /v1/stats, /shutdown) is the shared ferrite-http stack.
//!
//! The hand-rolled std::net server this replaces is gone. It also could not have
//! worked as written: it built the spin barrier INSIDE each rank's closure (so
//! every rank waited alone on its own barrier — an immediate hang) and skipped
//! the peer handshake (`enable_peer_access` + staging exchange + `set_peers`)
//! that the all-reduce's peer stores require. The prologue below is the one-shot
//! path's (verified) verbatim.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ferrite_http::single_flight::{SingleFlight, StepEngine};
use ferrite_http::tokenizer::{ChatTokenizer, StopSpec};
use ferrite_types::Result;

use crate::chain_dev::{DevChain, KvSnapshot, RunOpts};
use crate::config::Dsv41Config;
use crate::device::Device;
use crate::load::Loader;
use crate::tp::{self, Collective};

// ---- the rank pool ---------------------------------------------------------

/// One lockstep command, broadcast to every rank: the ranks are ONE engine (the
/// collectives make them so), so they must run the same sequence — a rank that
/// skips a step deadlocks inside the next all-reduce.
#[derive(Clone)]
enum RankCmd {
    /// Reset the chain and consume the prompt, one forward per prompt token
    /// (the KV ring is per-sequence); the reply is the first generated token.
    ///
    /// With `DSV41_KV_CACHE=1` this is ALSO the resume command: the rank consults
    /// its own frozen-prefix cache first and, on an exact prompt match, drops the
    /// snapshot back in instead of running the forwards; the pool then issues the
    /// same `DecodeRun` it would have. There is deliberately **no** `Resume`
    /// variant — the hit decision must be uniform across ranks (a rank that
    /// skipped a forward the others ran lands in the next collective out of
    /// step), and the KV is TP-sharded, so no snapshot could be broadcast anyway
    /// (a snapshot is MBs, which does not belong on the command/ack channel).
    /// See `prefill_or_resume` for the full rationale.
    Prefill(Vec<u32>),
    /// One steady-state decode step (zero H2D — the token already sits in the
    /// device's `ids` buffer). Kept as the un-batched fallback; the pool uses the
    /// batched `DecodeRun` below.
    #[allow(dead_code)]
    Decode { token: u32, pos: usize },
    /// `n` steady-state decode steps in ONE command. The rank threads are already
    /// lockstep by construction (they run the same loop), so the per-token
    /// command + world-way ack round trip was pure overhead: it dominated the step
    /// (7.75 tok/s end-to-end vs 21.7 with the old direct loop). The pool issues
    /// this and serves `decode()` from its look-ahead buffer; the engine contract
    /// is unchanged. Stops early on a stop token, exactly like the driver would.
    DecodeRun { token: u32, pos: usize, n: usize },
}

/// How many tokens one `DecodeRun` asks for. The round trip is amortized over
/// this many steps; the cost of over-running is bounded by it (the pool discards
/// the tail when the driver stops, and the next request resets the chain).
const LOOKAHEAD: usize = 16;

/// `DSV41_KV_CACHE=1` turns on the in-process KV prefix cache (P0: host memory,
/// exact full-prompt match). Read ONCE and cached — the house rule for every
/// hot-path gate; `"0"` means OFF even though it is "set".
///
/// Off by default because the snapshot has a price of its own: every prefill
/// freezes the sequence's prefix state to the host (~15-35 MB and a dozen D2H
/// syncs), which only pays off when requests actually share a prompt.
fn kv_cache_on() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_KV_CACHE").map(|v| v != "0").unwrap_or(false))
}

/// How many frozen prefixes to hold (`DSV41_KV_CACHE_SLOTS`, default 8). The
/// snapshot is linear in the prompt (see `LayerKvSnapshot`), so this is the
/// cache's memory bound: 8 x ~35 MB at a 4k prompt.
fn kv_cache_cap() -> usize {
    static F: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("DSV41_KV_CACHE_SLOTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(8)
    })
}

/// The model is ~80 s of mmap + upload; the first request waits for it.
const LOAD_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1800);
/// Watchdog on one lockstep step: a rank wedged inside a collective can be
/// reported but never unstuck (the adapter poisons the pool), so the timeout
/// keeps the HTTP layer answering instead of hanging the driver thread forever.
const STEP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1800);

/// The TP rank pool: `world` threads, each bound to its own device and holding
/// its own weight shard, all executing the same request in lockstep.
///
/// The rank threads are a `tp::run_ranks` scope parked on its own thread (they
/// outlive any single request); the pool talks to them with plain commands and
/// collects EVERY rank's reply before returning — that wait IS the lockstep
/// invariant. Rank 0's token is the result: the shared reductions make every
/// rank's logits identical, so no broadcast is needed.
pub struct TpRankPool {
    world: usize,
    /// One command channel per rank (the broadcast fan-out).
    cmd: Vec<std::sync::mpsc::Sender<RankCmd>>,
    /// Rank replies: (rank, token-or-error).
    res: std::sync::mpsc::Receiver<(usize, Result<Vec<u32>>)>,
    /// One-shot load reports — a failed load arrives as `Err` from its rank.
    ready: std::sync::mpsc::Receiver<Result<usize>>,
    /// Tokens already computed by the last `DecodeRun`, handed out one per
    /// `StepEngine::decode` call (so the round trip is per LOOKAHEAD tokens).
    lookahead: std::collections::VecDeque<u32>,
    /// The scope-owner thread, held (never joined) so the pool owns the ranks'
    /// lifetime: dropping the pool drops the command senders and each rank's
    /// `recv` then ends its loop.
    _owner: Option<std::thread::JoinHandle<Result<()>>>,
    loaded: bool,
    /// Prompt + generation bound (each prompt token is one forward into a
    /// per-sequence KV ring, so the whole sequence must fit).
    max_ctx: usize,
    /// The checkpoint's stop set (the engine retires on these ids; the wire
    /// layer strips the same ones).
    stops: Vec<u32>,
    /// The prefix-hit length the ranks reported for the most recent admission
    /// (`StepEngine::resume_hit`). Read after `broadcast` returns, i.e. after
    /// every rank has already stored its own value.
    resume_hit: usize,
    /// Written by the ranks on every prefill — the prefix cache's hit length.
    prefix_hit: Arc<AtomicUsize>,
}

impl TpRankPool {
    fn new(dir: &str, so: &str, cfg: &Dsv41Config, world: usize, stops: Vec<u32>) -> Result<Self> {
        let mut cmd = Vec::with_capacity(world);
        let mut rxs = Vec::with_capacity(world);
        for _ in 0..world {
            let (tx, rx) = std::sync::mpsc::channel::<RankCmd>();
            cmd.push(tx);
            // each rank only ever touches its own slot (no contention: the lock
            // is what makes the per-rank receivers shareable with the scope)
            rxs.push(Mutex::new(rx));
        }
        let (res_tx, res) = std::sync::mpsc::channel();
        let (ready_tx, ready) = std::sync::mpsc::channel();
        let barrier = Arc::new(tp::SpinBarrier::new(world));
        let staging = Arc::new(Mutex::new(vec![0u64; world]));
        let rxs = Arc::new(rxs);
        let prefix_hit = Arc::new(AtomicUsize::new(0));
        let prefix_hit_for_ranks = prefix_hit.clone();
        let dir = dir.to_string();
        let so = so.to_string();
        let cfg = cfg.clone();
        let max_ctx = cfg.max_seq_len;
        // the ranks must outlive this call, so the run_ranks scope parks here
        // the ranks also need the stop set (the batched run stops early on a stop
        // token, so the pool's look-ahead never runs past the driver's decision)
        let stops_for_ranks = stops.clone();
        let owner = std::thread::Builder::new()
            .name("dsv41-tp-pool".into())
            .spawn(move || {
                tp::run_ranks(world, move |rank| {
                    rank_loop(
                        rank, world, &stops_for_ranks, &cfg, &dir, &so, &rxs, &res_tx, &ready_tx, &barrier,
                        &staging, &prefix_hit_for_ranks,
                    )
                })
            })
            .map_err(|e| ferrite_types::FerriteError::Config(format!("spawn tp pool: {e}")))?;
        Ok(TpRankPool {
            lookahead: std::collections::VecDeque::new(),
            world,
            cmd,
            res,
            ready,
            _owner: Some(owner),
            loaded: false,
            max_ctx,
            stops,
            resume_hit: 0,
            prefix_hit,
        })
    }

    /// Wait for every rank's load (the first request pays it; the listener is
    /// already bound, so requests queue in `SingleFlight` meanwhile).
    fn ensure_loaded(&mut self) -> Result<()> {
        if self.loaded {
            return Ok(());
        }
        let mut loaded = 0usize;
        for _ in 0..self.world {
            match self.ready.recv_timeout(LOAD_TIMEOUT) {
                Ok(Ok(_rank)) => loaded += 1,
                Ok(Err(e)) => return Err(e),
                Err(_) => {
                    return Err(ferrite_types::FerriteError::Config(format!(
                        "tp pool: only {loaded}/{} ranks loaded (a rank died or timed out)",
                        self.world
                    )))
                }
            }
        }
        self.loaded = true;
        eprintln!("[dsv41] tp pool ready: {} ranks loaded", self.world);
        Ok(())
    }

    /// Broadcast one command, then collect EVERY rank's reply. Rank 0's token is
    /// the answer; any rank's error fails the step (a broken rank means a broken
    /// lockstep engine — the adapter poisons the pool on Err).
    fn broadcast(&mut self, cmd: RankCmd) -> Result<Vec<u32>> {
        for (r, tx) in self.cmd.iter().enumerate() {
            tx.send(cmd.clone()).map_err(|_| {
                ferrite_types::FerriteError::Config(format!("tp pool: rank {r} is gone"))
            })?;
        }
        let mut tokens: Option<Vec<u32>> = None;
        let mut fault: Option<ferrite_types::FerriteError> = None;
        for _ in 0..self.world {
            match self.res.recv_timeout(STEP_TIMEOUT) {
                Ok((rank, Ok(t))) => {
                    if rank == 0 {
                        tokens = Some(t);
                    }
                }
                Ok((rank, Err(e))) => {
                    if fault.is_none() {
                        fault = Some(ferrite_types::FerriteError::Config(format!("rank {rank}: {e}")));
                    }
                }
                Err(_) => {
                    return Err(ferrite_types::FerriteError::Config(
                        "tp pool: a rank did not answer (wedged in a collective?)".into(),
                    ))
                }
            }
        }
        if let Some(e) = fault {
            return Err(e);
        }
        tokens.ok_or_else(|| {
            ferrite_types::FerriteError::Config("tp pool: rank 0 produced no token".into())
        })
    }
}

impl StepEngine for TpRankPool {
    fn prefill(&mut self, prompt: &[u32]) -> Result<u32> {
        self.ensure_loaded()?;
        // a new request resets the chain, so any un-consumed look-ahead is stale
        self.lookahead.clear();
        self.resume_hit = 0;
        let mut v = self.broadcast(RankCmd::Prefill(prompt.to_vec()))?;
        self.resume_hit = self.prefix_hit.load(Ordering::SeqCst);
        Ok(if v.is_empty() { 0 } else { v.remove(0) })
    }

    /// Serve one decode step, but only pay a command+ack round trip once every
    /// LOOKAHEAD steps: the rank threads are lockstep by construction, so the
    /// per-token round trip was pure overhead (it dominated the step and cost
    /// ~2.8x end-to-end). The rank's n-step run produces exactly what n separate
    /// calls would, so the buffered tokens are the values the driver expects.
    fn decode(&mut self, token: u32, pos: usize) -> Result<u32> {
        if let Some(t) = self.lookahead.pop_front() {
            return Ok(t);
        }
        let n = LOOKAHEAD.min(self.max_ctx.saturating_sub(pos)).max(1);
        let mut v = self.broadcast(RankCmd::DecodeRun { token, pos, n })?;
        if v.is_empty() {
            return Err(ferrite_types::FerriteError::Config(
                "tp pool: decode run produced no token".into(),
            ));
        }
        let first = v.remove(0);
        self.lookahead.extend(v);
        Ok(first)
    }

    fn is_stop(&self, token: u32) -> bool {
        self.stops.contains(&token)
    }

    fn stop_id(&self) -> u32 {
        self.stops.first().copied().unwrap_or(0)
    }

    fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    /// The tokens the last admission served from the frozen prefix cache (0 when
    /// the cache is off, missed, or nothing is cached). `SingleFlight` puts this
    /// into `Admission::prefix_hit`, which the driver reports as
    /// `usage.prompt_tokens_details.cached_tokens`.
    fn resume_hit(&self) -> usize {
        self.resume_hit
    }
}

/// One rank's life: load its shard, join the peer handshake, then execute
/// lockstep commands until the pool is dropped.
///
/// A failed load is reported on the ready channel — its siblings are already
/// inside the shared load barrier, which cannot be left without them (the
/// documented bring-up hazard: a rank that dies during load wedges the pool).
#[allow(clippy::too_many_arguments)]
fn rank_loop(
    rank: usize,
    world: usize,
    stops: &[u32],
    cfg: &Dsv41Config,
    dir: &str,
    so: &str,
    rxs: &Arc<Vec<Mutex<std::sync::mpsc::Receiver<RankCmd>>>>,
    res_tx: &std::sync::mpsc::Sender<(usize, Result<Vec<u32>>)>,
    ready_tx: &std::sync::mpsc::Sender<Result<usize>>,
    barrier: &Arc<tp::SpinBarrier>,
    staging: &Arc<Mutex<Vec<u64>>>,
    prefix_hit: &Arc<AtomicUsize>,
) -> Result<()> {
    let r = pool_rank_body(
        rank, world, stops, cfg, dir, so, rxs, res_tx, ready_tx, barrier, staging, prefix_hit,
    );
    if let Err(e) = &r {
        let _ = ready_tx.send(Err(ferrite_types::FerriteError::Config(e.to_string())));
    }
    r
}

#[allow(clippy::too_many_arguments)]
fn pool_rank_body(
    rank: usize,
    world: usize,
    stop_set: &[u32],
    cfg: &Dsv41Config,
    dir: &str,
    so: &str,
    rxs: &Arc<Vec<Mutex<std::sync::mpsc::Receiver<RankCmd>>>>,
    res_tx: &std::sync::mpsc::Sender<(usize, Result<Vec<u32>>)>,
    ready_tx: &std::sync::mpsc::Sender<Result<usize>>,
    barrier: &Arc<tp::SpinBarrier>,
    staging: &Arc<Mutex<Vec<u64>>>,
    prefix_hit: &Arc<AtomicUsize>,
) -> Result<()> {
    // ---- bring-up: the one-shot path's prologue (rank_body) verbatim ----
    Device::bind_to(rank as i32)?;
    let dev = Arc::new(Device::open(so)?);
    let mut loader = Loader::new(std::path::Path::new(dir), &dev)?;
    if std::env::var("DSV41_SKIP_ENGRAM_WEIGHTS")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        loader.skip_prefixes.push("engram.embed.".into());
    }
    let w = loader.load(cfg, world, rank)?;
    dev.sync()?;
    let eng_map = load_eng_map(dir, cfg);
    // every rank must be loaded (and holding its CUDA context) before the peer
    // handshake: the running ranks spin inside the all-reduce until the last
    // one joins
    barrier.wait();
    let peers = dev.enable_peer_access()?;
    if rank == 0 {
        eprintln!("[dsv41] rank0: peer access to {peers} devices; entering lockstep");
    }
    barrier.wait();
    let hc_dim = cfg.hc_mult * cfg.dim;
    // The staging slot must cover the LARGEST all-reduce payload any path
    // issues. The verify chain (`step_rows`) reduces m*dim floats (m =
    // VERIFY_ROWS = 6 → 30720), larger than the single-row hc payload
    // (hc_mult*dim = 20480); an undersized slot makes the AR store overrun the
    // next peer's parity half and the protocol desyncs into a wedge (the
    // GLM-side P2P_AR_MAX_N lesson, same failure shape).
    let ar_bytes = (hc_dim.max(crate::chain_dev::VERIFY_ROWS * cfg.dim)) * 4;
    let mut comm = Collective::new(dev.clone(), world, rank, ar_bytes, barrier.clone())?;
    {
        staging.lock().unwrap()[rank] = comm.staging_base();
    }
    barrier.wait();
    let peers = staging.lock().unwrap().clone();
    barrier.wait();
    comm.set_peers(peers)?;
    let mut chain = DevChain::new(&dev, cfg, &w, RunOpts::from_env(), eng_map)?;
    chain.comm = Some(Arc::new(comm));
    // The DSpark draft (shadow mode, DSV41_DSPARK=1): the mtp.* weights are
    // already loaded (`w`); the draft reuses the chain's rope tables — its
    // positions are a subset of the main chain's.
    //
    // The draft is TP-aware: its attention tensors are replicated (global
    // geometry on every rank, no collective), but its MoE is split exactly like
    // the backbone's, so it needs `world`/`rank` for the local expert slices and
    // the chain's collective for the MoE all-reduce. `comm` has already been
    // moved into `chain.comm`, so the Arc is cloned from there.
    let mut dspark = if cfg.dspark_armed() {
        let mut d = crate::dspark_dev::DsparkDev::new(&dev, &w, cfg, world, rank)?;
        if let Some(c) = chain.comm.clone() {
            d.set_comm(c);
        }
        let (cos, sin) = chain.rope_tables();
        d.set_rope_tables(cos, sin);
        if rank == 0 {
            eprintln!(
                "[dspark] shadow mode armed: draft+verify run per step, all effects \
                 rolled back (world={world})"
            );
        }
        Some(d)
    } else {
        None
    };
    // accept-length accumulator for the periodic shadow report
    let mut dspark_acc_sum: u64 = 0;
    let mut dspark_steps: u64 = 0;
    let mut dspark_draft_ms: f64 = 0.0;
    let mut dspark_verify_ms: f64 = 0.0;
    let mut dspark_commit_ms: f64 = 0.0;
    if rank == 0 {
        eprintln!("[dsv41] rank0: chain ready, serving");
    }
    ready_tx.send(Ok(rank)).map_err(|_| {
        ferrite_types::FerriteError::Config("tp pool: coordinator gone".into())
    })?;

    // ---- the lockstep command loop ----
    // Decode timing, the GLM house style (ferrite-exec/tp.rs's `[megab] replay`
    // print): the wall of ONE full step, printed EVERY step, never averaged over
    // a segment. The accumulator this replaces carried two bugs that a segment
    // average hid for a whole session:
    //   1. the timer started AFTER the first DecodeRun batch completed while that
    //      batch's steps were still counted, so with LOOKAHEAD=16 the first
    //      report of every request divided 16 steps' wall by 32 - exactly half
    //      the true per-step time;
    //   2. the tail segment ended when the NEXT request's prefill arrived, so
    //      curl/HTTP/admission latency was spread over the tail steps.
    // Together they made a uniform ~37 ms/step read as "18 ms early, 38 ms late"
    // and got misexplained as a context-length effect. DSV41_TIMING=0 silences
    // the per-step lines.
    let timing = std::env::var("DSV41_TIMING").map(|v| v != "0").unwrap_or(true);
    let step_time = |pos: usize, dt: std::time::Duration| {
        if timing && rank == 0 {
            eprintln!(
                "[dsv41] step pos={}: {:.2}ms ({:.1} tok/s)",
                pos,
                dt.as_secs_f32() * 1e3,
                1.0 / dt.as_secs_f64().max(1e-9)
            );
        }
    };
    // ---- the KV prefix cache (DSV41_KV_CACHE=1) ----
    // Per RANK: the KV is TP-sharded, so each rank freezes and restores its own
    // share. The DECISION is nonetheless uniform — every rank sees the same
    // requests in the same order and runs the same LRU over the same prompts — so
    // a hit/miss here is one engine-wide fact, not a per-rank guess. (It has to
    // be: a rank that skipped a forward the others ran would land in the next
    // collective out of step. A divergence is therefore a loud wedge, never a
    // silent KV desync.)
    //
    // ---- 验收标准 / ACCEPTANCE (P0; the wave3-kv-mvp contract) -----------------
    // A SECOND request with the SAME prompt must serve its TTFT in **< 50 ms**
    // instead of the SECONDS a cold prefill costs: seconds = one forward per
    // prompt token (measured ~15 ms/token, so a 1k prompt ≈ 15 s); < 50 ms = the
    // hit path is `kv_restore` only (a dozen H2D/D2D + syncs, ~10 MB) plus the one
    // decode step the driver issues anyway — NOT ONE prefill forward runs.
    // Three observable points, all already wired:
    //   * `DSV41_KV_TRACE=1` → rank 0 logs one line per admission:
    //     `[kv-cache] prefill N tokens: HIT M (restored, prefill skipped) in X.Xms`
    //     (a MISS prints the same line with `MISS (full prefill)`);
    //   * the response's `usage.prompt_tokens_details.cached_tokens == prompt_len`;
    //   * the SSE `admitted prefix_hit=N` comment line.
    // With the gate unset (or `DSV41_KV_CACHE=0`) none of this engages and the
    // prefill path is the original `prefill_chain` verbatim.
    // ---------------------------------------------------------------------------
    let kv_enabled = kv_cache_on() && !cfg.dspark_armed();
    let mut kv_cache = KvCache::new(kv_cache_cap());
    let mut kv_warned = false;
    if rank == 0 && kv_cache_on() && !kv_enabled {
        eprintln!(
            "[kv-cache] DSV41_KV_CACHE is set but DSpark is armed: the draft's window rings \
             are not part of the snapshot yet, so the cache stays off"
        );
    }
    let rx = rxs[rank].lock().unwrap();
    loop {
        match rx.recv() {
            Ok(RankCmd::Prefill(ids)) => {
                let r = prefill_or_resume(
                    &mut chain,
                    &ids,
                    &mut kv_cache,
                    kv_enabled,
                    &mut kv_warned,
                    rank,
                );
                if let Ok((_, hit)) = &r {
                    // every rank stores the same value; the pool reads it once the
                    // broadcast has collected all of them.
                    prefix_hit.store(*hit, Ordering::SeqCst);
                }
                let r = r.map(|(t, _)| vec![t]);
                if res_tx.send((rank, r)).is_err() {
                    return Ok(()); // the pool is gone
                }
            }
            Ok(RankCmd::Decode { token, pos }) => {
                let st = std::time::Instant::now();
                let r = chain.step_dev(token, pos).map(|t| vec![t]);
                if r.is_ok() {
                    step_time(pos, st.elapsed());
                }
                if res_tx.send((rank, r)).is_err() {
                    return Ok(());
                }
            }
            Ok(RankCmd::DecodeRun { token, pos, n }) => {
                // n steps in one command; stops early on a stop token so the pool's
                // buffer never runs past the driver's stop decision.
                //
                // DSV41_SPEC makes a step's YIELD variable: a committed DSpark
                // block advances the engine by `k_acc + 1` positions and hands
                // back that many tokens, so the loop strides by what the step
                // returned — NOT by `i`. The old `pos + i` arithmetic is the
                // single biggest pitfall here: one multi-token step with a fixed
                // +1 stride desyncs the driver's position from the engine's
                // counter and the very next step writes the wrong slot.
                let mut out: Vec<u32> = Vec::with_capacity(n);
                let mut t = token;
                let mut p = pos;
                let mut r = Ok(());
                while out.len() < n {
                    let st = std::time::Instant::now();
                    let emitted: Vec<u32> = if let Some(d) = dspark.as_mut() {
                        if spec_mode() {
                            // REAL COMMIT: the draft+verify block is kept for the
                            // accepted prefix, so this step is the ONLY forward
                            // this step runs.
                            let rep = match chain.dspark_spec_step(d, t, p) {
                                Ok(rep) => rep,
                                Err(e) => {
                                    // NEVER `?` here: a swallowed error makes this
                                    // rank exit without answering, its siblings spin
                                    // in the next barrier forever, and the operator
                                    // sees "a rank did not answer" with NO cause —
                                    // the exact blind spot the wedge bisect hit.
                                    eprintln!(
                                        "[dsv41] rank {rank} spec step err at pos {p}: {e}"
                                    );
                                    let _ = res_tx.send((rank, Err(e)));
                                    return Ok(());
                                }
                            };
                            dspark_acc_sum += rep.k_acc as u64;
                            dspark_steps += 1;
                            dspark_draft_ms += rep.draft_ms as f64;
                            dspark_verify_ms += rep.verify_ms as f64;
                            dspark_commit_ms += rep.commit_ms as f64;
                            if std::env::var_os("DSV41_DSPARK_DEBUG").is_some() && rank == 0 {
                                eprintln!(
                                    "[dspark-dbg] pos={} next={} drafts={:?} verify={:?} \
                                     k_acc={} emitted={:?} commit={:.2}ms",
                                    p,
                                    rep.next,
                                    rep.drafts,
                                    rep.verify_out,
                                    rep.k_acc,
                                    rep.emitted,
                                    rep.commit_ms
                                );
                            }
                            if timing && rank == 0 && dspark_steps % 50 == 0 {
                                eprintln!(
                                    "[dspark] steps={} mean-k={:.3} tok/step={:.3} draft={:.2}ms \
                                     verify={:.2}ms commit={:.2}ms (per step)",
                                    dspark_steps,
                                    dspark_acc_sum as f64 / dspark_steps as f64,
                                    (dspark_acc_sum + dspark_steps) as f64 / dspark_steps as f64,
                                    dspark_draft_ms / dspark_steps as f64,
                                    dspark_verify_ms / dspark_steps as f64,
                                    dspark_commit_ms / dspark_steps as f64,
                                );
                            }
                            // `DSV41_DIFF_EAGER=1`: re-decode this step's emitted
                            // positions one row at a time and report the first one
                            // the two paths disagree on. Runs on EVERY rank (the
                            // eager forward's argmax is a HEAD_SLICE collective)
                            // and is logged, never propagated — a probe must not
                            // answer an error for a step whose tokens are already
                            // committed.
                            if diff_eager() {
                                match chain.diff_eager_probe(p, &rep.emitted) {
                                    Ok(d) => {
                                        if rank == 0 {
                                            let mm = match d.first_mismatch {
                                                Some(i) => format!(
                                                    "first_mismatch={i} mismatch_pos={}",
                                                    p + 1 + i
                                                ),
                                                None => "first_mismatch=none".to_string(),
                                            };
                                            eprintln!(
                                                "[diff] pos={} spec_emitted={:?} eager={:?} {}",
                                                p, d.spec_emitted, d.eager, mm
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!(
                                            "[dsv41] rank {rank} diff probe err at pos {p}: {e}"
                                        );
                                    }
                                }
                            }
                            rep.emitted
                        } else {
                            // shadow mode: the single-row step stays the engine's
                            // real output; draft+verify run beside it and roll back
                            let rep = match chain.dspark_shadow_step(d, t, p) {
                                Ok(rep) => rep,
                                Err(e) => {
                                    eprintln!(
                                        "[dsv41] rank {rank} shadow step err at pos {p}: {e}"
                                    );
                                    let _ = res_tx.send((rank, Err(e)));
                                    return Ok(());
                                }
                            };
                            dspark_acc_sum += rep.accepted as u64;
                            dspark_steps += 1;
                            dspark_draft_ms += rep.draft_ms as f64;
                            dspark_verify_ms += rep.verify_ms as f64;
                            // per-step token-level trace (DSV41_DSPARK_DEBUG): the
                            // draft block vs the real next token and the verify's
                            // argmax row — the fastest way to see whether a low
                            // accept is a structural break (unrelated tokens) or
                            // precision noise (neighbouring tokens).
                            if std::env::var_os("DSV41_DSPARK_DEBUG").is_some() && rank == 0 {
                                eprintln!(
                                    "[dspark-dbg] pos={} next={} drafts={:?} verify={:?} acc={}",
                                    p,
                                    rep.next,
                                    rep.drafts,
                                    rep.verify_out,
                                    rep.accepted
                                );
                            }
                            if timing && rank == 0 && dspark_steps % 50 == 0 {
                                eprintln!(
                                    "[dspark] steps={} mean-accept={:.3} draft={:.2}ms verify={:.2}ms (per step)",
                                    dspark_steps,
                                    dspark_acc_sum as f64 / dspark_steps as f64,
                                    dspark_draft_ms / dspark_steps as f64,
                                    dspark_verify_ms / dspark_steps as f64,
                                );
                            }
                            vec![rep.next]
                        }
                    } else {
                        let tok = match chain.step_dev(t, p) {
                            Ok(tok) => tok,
                            Err(e) => {
                                eprintln!("[dsv41] rank {rank} step err at pos {p}: {e}");
                                let _ = res_tx.send((rank, Err(e)));
                                return Ok(());
                            }
                        };
                        vec![tok]
                    };
                    step_time(p, st.elapsed());
                    let step_len = emitted.len();
                    let mut hit_stop = false;
                    for &tok in &emitted {
                        out.push(tok);
                        if stop_set.contains(&tok) {
                            hit_stop = true;
                            break;
                        }
                    }
                    // The engine has already advanced by the WHOLE block, so the
                    // driver's position follows it even when a stop token cut the
                    // emission short (the pool resets the chain on the next
                    // prefill, so the unreported tail is harmless).
                    if let Some(&last) = emitted.last() {
                        t = last;
                    }
                    p += step_len;
                    if hit_stop {
                        break;
                    }
                }
                if res_tx.send((rank, r.map(|_| out))).is_err() {
                    return Ok(());
                }
            }
            // the pool dropped its senders (process shutdown)
            Err(_) => return Ok(()),
        }
    }
}

/// `DSV41_SPEC=1` switches the DSpark step from SHADOW to REAL COMMIT: the draft
/// + verify block is kept for its accepted prefix and the engine advances by the
/// block's yield instead of by one. Read ONCE and cached (the house rule for
/// every hot-path gate: a per-step getenv is exactly the slip this project has
/// been bitten by), and `"0"` means OFF even though it is "set".
///
/// It still requires `DSV41_DSPARK=1`: without the tap hook there is no draft and
/// the chain falls through to the plain single-row step.
fn spec_mode() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_SPEC").map(|v| v != "0").unwrap_or(false))
}

/// `DSV41_DIFF_EAGER=1` arms the SPEC-vs-EAGER per-position diff probe: after
/// every real-commit spec round, [`DevChain::diff_eager_probe`] re-decodes the
/// positions that round emitted one row at a time and serve prints
///
/// ```text
/// [diff] pos=… spec_emitted=[…] eager=[…] first_mismatch=<i mismatch_pos=… | none>
/// ```
///
/// `first_mismatch` is the first emitted token the m-row verify disagrees with
/// the plain single-row engine about, i.e. the first position at which a
/// spec-only defect becomes observable in the stream (see the probe's doc
/// comment for what is compared and what is undone). Greedy speculative
/// decoding promises `none` on every round.
///
/// Read ONCE and cached (the house rule for every hot-path gate), `"0"` means
/// OFF. Costs `emitted.len() - 1` extra single-row forwards per round, so it is
/// a LOCALISATION tool, never a perf mode: do not leave it on under an A/B.
///
/// It needs `DSV41_SPEC=1` — the probe compares a COMMITTED step's tokens with
/// the single-row replay of their positions, and only the spec path commits
/// multi-token blocks.
fn diff_eager() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_DIFF_EAGER").map(|v| v != "0").unwrap_or(false))
}

/// Feed the prompt one token per forward (the KV ring is per-sequence) and
/// return the first generated token — `StepEngine::prefill`'s contract.
fn prefill_chain(chain: &mut DevChain<'_>, ids: &[u32]) -> Result<u32> {
    chain.reset()?;
    let mut next = 0u32;
    for (i, &t) in ids.iter().enumerate() {
        next = chain.step(t, i)?;
    }
    Ok(next)
}

// ---- the P0 prefix cache ---------------------------------------------------

/// One frozen prefix: the exact prompt it was taken from, the token the cold
/// prefill returned, and the chain state at the end of that prefill.
struct KvCacheEntry {
    /// The prompt, verbatim — the KEY. Matching is a full sequence equality
    /// test, not a prefix one: P0 deliberately caches exactly what was asked.
    tokens: Vec<u32>,
    /// What `prefill_chain` returned for this prompt. A hit replays it instead of
    /// recomputing it, which is the whole point: the state the snapshot holds is
    /// the state AFTER the last prompt token was fed, and the first generated
    /// token is that step's argmax — one value, not worth a device round trip,
    /// and not reproducible without replaying a step against the state it already
    /// consumed.
    first_token: u32,
    /// The chain state at prefill completion. Immutable once taken: the decode
    /// that follows MOVES the live KV, never this copy, which is why a later hit
    /// resumes exactly at the prefill boundary.
    snap: KvSnapshot,
    last_used: u64,
}

/// The P0 prefix cache: exact full-prompt match, LRU eviction, host memory only.
///
/// A `HashMap` keyed by a stable hash of the prompt, with each entry keeping the
/// prompt itself so a hash collision is a MISS rather than a wrong hit — the
/// failure mode of a hash-only cache here would be resuming from another
/// request's KV, which is silent and total (unlike a cache miss, which just
/// costs the prefill).
///
/// ⚠️ One cache per RANK, and its contents must stay identical across ranks: the
/// hit decision is taken independently by each rank (nothing is broadcast, and
/// nothing of the snapshot travels through the command/ack channels — a snapshot
/// is MBs, which does not belong on that path), so all ranks rely on seeing the
/// same request sequence. They do, because the pool is batch-1 and every rank
/// executes the same command stream.
struct KvCache {
    slots: std::collections::HashMap<u64, KvCacheEntry>,
    cap: usize,
    /// A monotonic tick stamped on every access — the LRU's clock.
    clock: u64,
    hits: u64,
    misses: u64,
}

impl KvCache {
    fn new(cap: usize) -> Self {
        KvCache {
            slots: std::collections::HashMap::new(),
            cap: cap.max(1),
            clock: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// FNV-1a over the token ids (little-endian). Inlined rather than pulling in
    /// a hasher: the key is only ever compared against the stored prompt, so it
    /// needs to be stable, not cryptographic.
    fn hash(tokens: &[u32]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &t in tokens {
            for b in t.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }

    /// The entry for exactly this prompt, touching it for the LRU.
    fn get(&mut self, tokens: &[u32]) -> Option<&KvCacheEntry> {
        let k = Self::hash(tokens);
        self.clock += 1;
        let clock = self.clock;
        // a collision (different tokens, same key) is a miss, never a hit
        if self.slots.get(&k).is_some_and(|e| e.tokens == tokens) {
            self.hits += 1;
            let e = self.slots.get_mut(&k).unwrap();
            e.last_used = clock;
            Some(e)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Replace-or-insert, evicting the least recently used entry when full.
    fn insert(&mut self, tokens: &[u32], first: u32, snap: KvSnapshot) {
        let k = Self::hash(tokens);
        self.clock += 1;
        let clock = self.clock;
        if let Some(e) = self.slots.get_mut(&k) {
            if e.tokens == tokens {
                e.first_token = first;
                e.snap = snap;
                e.last_used = clock;
                return;
            }
        }
        while self.slots.len() >= self.cap {
            let victim = self
                .slots
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| *k);
            match victim {
                Some(k) => {
                    self.slots.remove(&k);
                }
                None => break,
            }
        }
        self.slots.insert(
            k,
            KvCacheEntry { tokens: tokens.to_vec(), first_token: first, snap, last_used: clock },
        );
    }

    fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}

/// `StepEngine::prefill`'s body with the prefix cache in front of it. Returns
/// `(first_token, prefix_hit)`.
///
/// A HIT skips the prompt's forwards entirely: `kv_restore` drops the frozen
/// state back in and the token the cold prefill produced is replayed, so the
/// engine is standing exactly where the first request was after its prefill and
/// the driver's next `decode(token, prompt_len)` is the same step it would have
/// run. A MISS runs the unchanged [`prefill_chain`] and then freezes what it
/// produced.
///
/// The snapshot is taken at PREFILL COMPLETION, once, and never updated during
/// decode: a prefix hit only ever needs the state a prompt leaves behind, and
/// re-snapshotting per decode step would multiply the copy cost for a state no
/// hit could use.
///
/// See the KV cache section in `pool_rank_body` for the acceptance line this
/// function's timing feeds (same-prompt second request: seconds → < 50 ms).
fn prefill_or_resume(
    chain: &mut DevChain<'_>,
    ids: &[u32],
    cache: &mut KvCache,
    enabled: bool,
    warned: &mut bool,
    rank: usize,
) -> Result<(u32, usize)> {
    // The ONE number the acceptance criterion is read off: the whole admission
    // (restore+copies on a HIT, every prompt forward + the freeze on a MISS).
    let t0 = std::time::Instant::now();
    if enabled {
        if let Some(e) = cache.get(ids) {
            let first = e.first_token;
            let hit = e.snap.tokens;
            chain.kv_restore(&e.snap)?;
            // The snapshot does NOT cover `s.ids` (step_dev is zero-H2D by
            // design): without this the first decode step would feed whatever
            // token the PREVIOUS request last produced. Restore it here.
            chain.prime_input(first)?;
            kv_log(rank, ids.len(), Some(hit), cache.stats(), t0.elapsed());
            return Ok((first, hit));
        }
    }
    let first = prefill_chain(chain, ids)?;
    if enabled {
        // Refuse to cache anything the snapshot cannot represent exactly: the
        // contract a hit relies on is "byte-identical to a from-scratch run", so a
        // configuration the snapshot does not cover must degrade to a plain
        // prefill, not to a wrong answer.
        match chain.kv_snapshot() {
            Ok(snap) if snap.tokens == ids.len() => {
                cache.insert(ids, first, snap);
            }
            Ok(snap) => {
                if !*warned {
                    *warned = true;
                    eprintln!(
                        "[kv-cache] not caching: the chain's position counter reads {} after a \
                         {}-token prefill, so the snapshot would not sit where a hit resumes",
                        snap.tokens,
                        ids.len()
                    );
                }
            }
            Err(e) => {
                if !*warned {
                    *warned = true;
                    eprintln!("[kv-cache] not caching: {e}");
                }
            }
        }
    }
    kv_log(rank, ids.len(), None, cache.stats(), t0.elapsed());
    Ok((first, 0))
}

/// One line per admission, rank 0 only (`DSV41_KV_TRACE=1`): what the prefill
/// cost and whether it was served from the frozen prefix.
///
/// This is the acceptance measurement, not decoration — a same-prompt second
/// request is read straight off the two consecutive lines ("MISS (full prefill)
/// in 15_000ms" then "HIT … in 12ms"), so the `seconds → < 50 ms` criterion needs
/// no profiler. ⚠️ Rank-gated on purpose: EVERY rank runs the same prefill, so an
/// ungated line would be `world`-fold duplicate log spam (the pre-existing
/// `[kv-cache] HIT/MISS` prints had exactly that bug at TP8).
fn kv_log(rank: usize, tokens: usize, hit: Option<usize>, stats: (u64, u64), dt: std::time::Duration) {
    if rank != 0 || !kv_trace() {
        return;
    }
    let (h, m) = stats;
    match hit {
        Some(n) => eprintln!(
            "[kv-cache] prefill {tokens} tokens: HIT {n} (restored, prefill skipped) in {:.1}ms \
             (hits={h} misses={m})",
            dt.as_secs_f64() * 1e3
        ),
        None => eprintln!(
            "[kv-cache] prefill {tokens} tokens: MISS (full prefill) in {:.1}ms (hits={h} misses={m})",
            dt.as_secs_f64() * 1e3
        ),
    }
}

/// `DSV41_KV_TRACE=1` logs each hit/miss (off by default: the per-request line
/// would otherwise interleave with the per-step timing lines).
fn kv_trace() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_KV_TRACE").map(|v| v != "0").unwrap_or(false))
}

#[cfg(test)]
mod kv_cache_tests {
    use super::*;

    /// A stand-in snapshot: the cache never looks inside one, so an empty
    /// payload is enough to exercise the keying and the LRU.
    fn snap(tokens: usize) -> KvSnapshot {
        KvSnapshot {
            pos_ctr: tokens as i32,
            tokens,
            layers: Vec::new(),
            clen_dev: Vec::new(),
            eng_cache: None,
        }
    }

    /// The P0 contract: matching is a full-sequence EQUALITY, so a prompt that
    /// merely shares a prefix must miss (P0 caches exactly what was asked).
    #[test]
    fn exact_match_only() {
        let mut c = KvCache::new(4);
        c.insert(&[1, 2, 3], 7, snap(3));
        assert_eq!(c.get(&[1, 2, 3]).map(|e| e.first_token), Some(7));
        assert_eq!(c.get(&[1, 2]).map(|e| e.first_token), None);
        assert_eq!(c.get(&[1, 2, 3, 4]).map(|e| e.first_token), None);
        assert_eq!(c.stats(), (1, 2));
    }

    /// Eviction is by least-recent use, and a LOOKUP counts as a use.
    #[test]
    fn lru_eviction() {
        let mut c = KvCache::new(2);
        c.insert(&[1], 10, snap(1));
        c.insert(&[2], 20, snap(1));
        assert_eq!(c.get(&[1]).map(|e| e.first_token), Some(10));
        c.insert(&[3], 30, snap(1)); // [2] is now the oldest
        assert_eq!(c.get(&[2]).map(|e| e.first_token), None);
        assert_eq!(c.get(&[1]).map(|e| e.first_token), Some(10));
        assert_eq!(c.get(&[3]).map(|e| e.first_token), Some(30));
    }

    /// Re-freezing the same prompt replaces its state instead of growing the map
    /// (a re-prefill of a cached prompt must not leak a second snapshot).
    #[test]
    fn reinsert_replaces() {
        let mut c = KvCache::new(4);
        c.insert(&[9, 9], 1, snap(2));
        c.insert(&[9, 9], 2, snap(2));
        assert_eq!(c.slots.len(), 1);
        assert_eq!(c.get(&[9, 9]).map(|e| e.first_token), Some(2));
    }
}

/// The engram's token map (a pure function of the tokenizer, precomputed into
/// the checkpoint dir): absent means the engram is disabled for this run.
fn load_eng_map(dir: &str, cfg: &Dsv41Config) -> Option<crate::engram::TokenMap> {
    std::fs::read(format!("{dir}/engram_token_map.bin"))
        .ok()
        .filter(|b| b.len() % 8 == 0 && !b.is_empty())
        .map(|b| {
            let v: Vec<i64> = b
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            crate::engram::TokenMap::from_table(v, cfg.engram_compressed_vocab_size)
        })
}

// ---- the assembly ----------------------------------------------------------

/// Resolve the checkpoint's end-of-sequence id: `generation_config.json` →
/// `config.json` → (a `tokenizer_config.json` present ? `1` : `None`).
///
/// This checkpoint ships no generation_config.json and its
/// `text_config.eos_token_id` is null, so a single-source lookup returns None
/// and the decode loop never stops: the model answers correctly (" Paris" then
/// EOS = token 1) and then keeps generating, which looked like a degenerate
/// tail. The fallback chain is the honest answer.
pub fn resolve_eos(dir: &str) -> Option<u32> {
    let json = |p: String| {
        std::fs::read_to_string(p)
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
    };
    let id_of = |v: &serde_json::Value| v.get("eos_token_id").and_then(|e| e.as_u64());
    let from_gen = json(format!("{dir}/generation_config.json")).and_then(|v| id_of(&v));
    let from_cfg = json(format!("{dir}/config.json")).and_then(|v| id_of(&v));
    let has_tok = json(format!("{dir}/tokenizer_config.json")).is_some();
    from_gen
        .or(from_cfg)
        .map(|e| e as u32)
        .or(if has_tok { Some(1u32) } else { None })
}

/// Build the DSV41 serve stack: the checkpoint's tokenizer (with its own stop
/// set) + the TP rank pool behind the single-flight adapter. Binding happens in
/// the caller's launcher, so the listener is up immediately while the ranks
/// finish loading behind the first request.
pub fn build_serve_engine(
    dir: &str,
    so: &str,
    cfg: &Dsv41Config,
    tp: usize,
    eos: Option<u32>,
) -> Result<(ChatTokenizer, SingleFlight<TpRankPool>)> {
    // The stop set is the CHECKPOINT's: the resolved EOS PLUS the checkpoint's
    // own end-of-sentence special (its config ships eos_token_id null, so the
    // name-resolved id is the honest fallback). Engine retirement and wire
    // stripping both come from this one set.
    let eos_ids: Vec<u32> = eos.into_iter().collect();
    let spec = StopSpec::new(&eos_ids, &["<|end_of_sentence|>"]);
    let tok =
        ChatTokenizer::from_file_with(&std::path::Path::new(dir).join("tokenizer.json"), spec)?;
    let stops = tok.stop_ids().to_vec();
    let pool = TpRankPool::new(dir, so, cfg, tp, stops)?;
    Ok((tok, SingleFlight::new(pool)))
}
