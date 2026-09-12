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

use std::sync::{Arc, Mutex};

use ferrite_http::single_flight::{SingleFlight, StepEngine};
use ferrite_http::tokenizer::{ChatTokenizer, StopSpec};
use ferrite_types::Result;

use crate::chain_dev::{DevChain, RunOpts};
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
                        &staging,
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
        let mut v = self.broadcast(RankCmd::Prefill(prompt.to_vec()))?;
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
) -> Result<()> {
    let r = pool_rank_body(rank, world, stops, cfg, dir, so, rxs, res_tx, ready_tx, barrier, staging);
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
    let rx = rxs[rank].lock().unwrap();
    loop {
        match rx.recv() {
            Ok(RankCmd::Prefill(ids)) => {
                let r = prefill_chain(&mut chain, &ids).map(|t| vec![t]);
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
                // buffer never runs past the driver's stop decision
                let mut out: Vec<u32> = Vec::with_capacity(n);
                let mut t = token;
                let mut r = Ok(());
                for i in 0..n {
                    let st = std::time::Instant::now();
                    let next = if let Some(d) = dspark.as_mut() {
                        // shadow mode: the single-row step stays the engine's
                        // real output; draft+verify run beside it and roll back
                        let rep = match chain.dspark_shadow_step(d, t, pos + i) {
                            Ok(rep) => rep,
                            Err(e) => {
                                // NEVER `?` here: a swallowed error makes this
                                // rank exit without answering, its siblings spin
                                // in the next barrier forever, and the operator
                                // sees "a rank did not answer" with NO cause —
                                // the exact blind spot the wedge bisect hit.
                                eprintln!("[dsv41] rank {rank} shadow step err at pos {}: {e}", pos + i);
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
                                pos + i,
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
                        rep.next
                    } else {
                        match chain.step_dev(t, pos + i) {
                            Ok(n) => n,
                            Err(e) => {
                                eprintln!("[dsv41] rank {rank} step err at pos {}: {e}", pos + i);
                                let _ = res_tx.send((rank, Err(e)));
                                return Ok(());
                            }
                        }
                    };
                    step_time(pos + i, st.elapsed());
                    out.push(next);
                    t = next;
                    if stop_set.contains(&next) {
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
