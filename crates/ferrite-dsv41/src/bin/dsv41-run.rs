//! GPU end-to-end runner for DeepSeek-V4.1-Flash.
//!
//! ```text
//! DSV41_MODEL_DIR=/opt/dlami/nvme/models/DeepSeek-V4.1-Flash \
//! DSV41_KERNELS=$PWD/kernels/cuda/libferrite_kernels.so \
//! CUDA_VISIBLE_DEVICES=0 \
//!   ./target/release/dsv41-run --prompt "你好" --max-tokens 32
//! ```
//!
//! Loads the real checkpoint (weights stay in their native fp4/fp8 formats),
//! prefills the prompt one token at a time and then decodes greedily.
//!
//! `DSV41_SKIP_ENGRAM_WEIGHTS=1` (default in this build) skips the engram tables:
//! they are ~189 GiB and the engram write-back is still a separate increment, so
//! loading them would only add minutes to a run that cannot use them yet.

use std::sync::{Arc, Mutex};

use ferrite_dsv41::chain_dev::{DevChain, RunOpts};
use ferrite_dsv41::tp::{self, Collective};
use ferrite_dsv41::config::Dsv41Config;
use ferrite_dsv41::device::Device;
use ferrite_dsv41::load::Loader;
use ferrite_http::single_flight::{SingleFlight, StepEngine};
use ferrite_http::tokenizer::{ChatTokenizer, StopSpec};
use ferrite_http::ServeOptions;
use ferrite_types::Result;
// The checkpoint's chat frame lives WITH THE MODEL (ferrite-models) so both
// this runner and the unified `ferrite-serve --model dsv41` share one layout.
use ferrite_dsv41::Dsv41Frame;

fn arg(name: &str, default: Option<&str>) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    for (i, v) in a.iter().enumerate() {
        if v == name {
            return a.get(i + 1).cloned();
        }
        if let Some(rest) = v.strip_prefix(&format!("{name}=")) {
            return Some(rest.to_string());
        }
    }
    default.map(|d| d.to_string())
}

fn main() -> Result<()> {
    let dir = arg("--model-dir", None)
        .or_else(|| std::env::var("DSV41_MODEL_DIR").ok())
        .expect("--model-dir or DSV41_MODEL_DIR");
    let so = arg("--kernels", None)
        .or_else(|| std::env::var("DSV41_KERNELS").ok())
        .unwrap_or_else(|| "kernels/cuda/libferrite_kernels.so".to_string());
    let prompt = arg("--prompt", Some("你好")).unwrap();
    let max_tokens: usize = arg("--max-tokens", Some("32")).unwrap().parse().unwrap();
    let tp: usize = arg("--tp", Some("1")).unwrap().parse().unwrap();
    let rank: usize = arg("--rank", Some("0")).unwrap().parse().unwrap();
    let serve = std::env::args().any(|a| a == "--serve");
    let port: u16 = arg("--port", Some("8090")).unwrap().parse().unwrap();
    // End-of-sequence id. This checkpoint ships no generation_config.json and
    // its text_config.eos_token_id is null, so the previous single-source lookup
    // returned None and the decode loop NEVER stopped: the model answered
    // correctly (" Paris" then EOS = token 1) and the runner kept generating
    // past it, which is what looked like a degenerate tail. Fall back through
    // the top-level config.json, then the tokenizer's own metadata.
    let eos: Option<u32> = {
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
    };
    // The engram is an architectural component of this checkpoint, not an
    // optional extra: `text_config.engram_layer_ids = [1, 14]`, and the 48
    // shards do carry `layers.{1,14}.engram.{embed,wkv,q_weight,k_weight}`.
    // The reference applies it as `h = layer.engram(h, hashes, mask)` BEFORE the
    // block at those layers, i.e. it writes into the hc residual stream there.
    // Skipping it starves layers 1 and 14 of that write-back. Default = load.
    let skip_engram_w = std::env::var("DSV41_SKIP_ENGRAM_WEIGHTS").map(|v| v != "0").unwrap_or(false);

    // ---- config (from the checkpoint's own config.json) ----
    let cfg_txt = std::fs::read_to_string(format!("{dir}/config.json"))
        .map_err(|e| ferrite_types::FerriteError::Config(format!("config.json: {e}")))?;
    let cfg = Dsv41Config::from_json_str(&cfg_txt)?;
    // Bring-up truncation: a full single-GPU run is impossible (the checkpoint
    // is ~286 GiB without the engram tables against ~180 GiB of HBM), so the
    // pipeline is first exercised on the leading layers with real weights.
    let mut cfg = cfg;
    if let Ok(n) = std::env::var("DSV41_LAYERS") {
        let n: usize = n.parse().unwrap_or(cfg.n_layers);
        cfg.n_layers = n;
        cfg.n_mtp_layers = 0;
        eprintln!("[dsv41] TRUNCATED RUN: {n} layers, no draft layers");
    }
    eprintln!(
        "[dsv41] config: dim={} layers={} heads={} hc={} experts={}/topk={} window={} index_topk={}",
        cfg.dim, cfg.n_layers, cfg.n_heads, cfg.hc_mult, cfg.n_routed_experts,
        cfg.n_activated_experts, cfg.window_size, cfg.index_topk
    );

    // ---- device ----
    if serve {
        if tp < 2 {
            eprintln!("[dsv41] --serve needs --tp >= 2 (the ranks each hold a shard)");
            return Ok(());
        }
        let model_name = arg("--model-name", Some("deepseek-v4.1-flash")).unwrap();
        return run_serve(&dir, &so, &cfg, tp, eos, port, model_name);
    }
let dev = Device::open(&so)?;
    eprintln!("[dsv41] device ready (kernels: {so})");

    // ---- tokenizer ----
    // `DSV41_PROMPT_IDS=1,2,3` bypasses the tokenizer for the PROMPT and feeds
    // explicit ids. The reference's generate.py does not feed raw text: it wraps
    // the user turn in the checkpoint's chat template (`encode_messages(messages,
    // thinking_mode="chat")`), which for this model ends in
    // `<|Assistant|></think>` — i.e. it tells the model to answer directly.
    // Without those markers the model keeps going in thinking mode, which is the
    // "思考过程 (Thinking..." drift seen in the raw-text runs.
    let tok = tokenizers::Tokenizer::from_file(format!("{dir}/tokenizer.json"))
        .map_err(|e| ferrite_types::FerriteError::Config(format!("tokenizer: {e}")))?;
    let ids: Vec<u32> = match std::env::var("DSV41_PROMPT_IDS") {
        Ok(v) => v
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect(),
        Err(_) => {
            // Match the reference's generate.py, which never feeds raw text: it
            // wraps the user turn in the checkpoint's chat template
            // (encode_messages(messages, thinking_mode="chat")) =
            //   <|begin_of_sentence|><|User|>{prompt}<|Assistant|></think>
            // for this checkpoint = ids [0, 128803] + prompt + [128804, 128822].
            // Without the trailing </think> the model does not know it is the
            // assistant's turn: e.g. "The capital of Japan is" then answers EOS
            // immediately, while with the template it answers " Tokyo.".
            let body = tok
                .encode(prompt.clone(), false)
                .map_err(|e| ferrite_types::FerriteError::Config(format!("encode: {e}")))?
                .get_ids()
                .to_vec();
            let mut v = Vec::with_capacity(body.len() + 4);
            v.push(0u32); // <|begin_of_sentence|>
            v.push(128803u32); // <|User|>
            v.extend_from_slice(&body);
            v.push(128804u32); // <|Assistant|>
            v.push(128822u32); // </think>
            v
        }
    };
    if ids.is_empty() {
        return Err(ferrite_types::FerriteError::Config("empty prompt".into()));
    }
    eprintln!("[dsv41] prompt {} tokens: {:?}", ids.len(), &ids[..ids.len().min(8)]);

    // ---- tensor parallel path ----
    if tp > 1 {
        return run_tp(&dir, &so, &cfg, ids.clone(), prompt.len(), max_tokens, tp, eos);
    }

    // ---- weights ----
    let t0 = std::time::Instant::now();
    let mut loader = Loader::new(std::path::Path::new(&dir), &dev)?;
    if skip_engram_w {
        loader.skip_prefixes.push("engram.embed.".into());
    }
    let w = loader.load(&cfg, tp, rank)?;
    dev.sync()?;
    // The engram's n-gram hash keys tokens by a compressed id space that is a
    // pure function of the tokenizer (the reference's `build_compressed_token_map`),
    // so it is precomputed once into engram_token_map.bin (129280 i64 little-endian).
    let eng_map = std::fs::read(format!("{dir}/engram_token_map.bin"))
        .ok()
        .filter(|b| b.len() % 8 == 0 && !b.is_empty())
        .map(|b| {
            let v: Vec<i64> = b
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            ferrite_dsv41::engram::TokenMap::from_table(v, cfg.engram_compressed_vocab_size)
        });
    eprintln!(
        "[dsv41] engram: {}",
        if eng_map.is_some() { "token map loaded" } else { "NO token map (engram disabled)" }
    );
    eprintln!(
        "[dsv41] weights loaded: {:.1} GiB in {:.1}s",
        loader.uploaded as f64 / (1u64 << 30) as f64,
        t0.elapsed().as_secs_f64()
    );

    // ---- chain ----
    let opts = RunOpts::from_env();
    let mut chain = DevChain::new(&dev, &cfg, &w, opts, eng_map.clone())?;
    chain.reset()?;

    // ---- prefill (one token per step: the KV ring is per-sequence) ----
    let t1 = std::time::Instant::now();
    let mut next_tok: u32 = 0;
    for (i, &t) in ids.iter().enumerate() {
        next_tok = chain.step(t, i)?;
    }
    eprintln!(
        "[dsv41] prefill {} tokens in {:.2}s ({:.1} tok/s)",
        ids.len(),
        t1.elapsed().as_secs_f64(),
        ids.len() as f64 / t1.elapsed().as_secs_f64()
    );

    // ---- greedy decode ----
    let mut produced: Vec<u32> = Vec::new();
    let pos0 = ids.len();
    let t2 = std::time::Instant::now();
    let t_dec = std::time::Instant::now();
    for step in 0..max_tokens {
        // the token comes from the previous step's DEVICE argmax (the chain prints
        // top5 under DSV41_TOP5 itself); no host scan, no full-logits download
        produced.push(next_tok);
        if Some(next_tok) == eos {
            break;
        }
        next_tok = chain.step_dev(next_tok, pos0 + step)?;
    }
    let dt = t2.elapsed().as_secs_f64();
    eprintln!(
        "[dsv41] decode {} tokens in {:.2}s ({:.2} tok/s)",
        produced.len(),
        dt,
        produced.len() as f64 / dt
    );
    let text = tok.decode(&produced, true).unwrap_or_default();
    println!("--- generated ({}) ---", produced.len());
    println!("{text}");
    {
        let el = t_dec.elapsed();
        let n = produced.len().max(1);
        eprintln!(
            "[dsv41] DECODE {} tokens in {:?} = {:.2} tok/s ({:.1} ms/token)",
            produced.len(),
            el,
            n as f64 / el.as_secs_f64(),
            el.as_secs_f64() * 1e3 / n as f64
        );
    }
    println!("--- ids: {:?}", &produced[..produced.len().min(24)]);
    Ok(())
}

/// Tensor-parallel run: one thread per rank, each bound to its own device and
/// holding its own slice of the weights. Ranks stay in lockstep through the
/// collective's barrier, and every rank computes the same logits (the shared
/// reductions make the streams identical), so sampling needs no broadcast.
#[allow(clippy::too_many_arguments)]
/// Tensor-parallel run: one thread per rank, each bound to its own device and
/// holding its own slice of the weights. Ranks stay in lockstep through the
/// collective's barrier, and every rank arrives at the same logits (the shared
/// reductions make the streams identical), so sampling needs no broadcast.
fn run_tp(
    dir: &str,
    so: &str,
    cfg: &Dsv41Config,
    ids: Vec<u32>,
    _prompt_len: usize,
    max_tokens: usize,
    tp: usize,
    eos: Option<u32>,
) -> Result<()> {
    let cfg = cfg.clone();
    let world = tp;
    println!(
        "[dsv41] TP{world}: dim={} layers={} hc*dim={} vocab={}",
        cfg.dim,
        cfg.n_layers,
        cfg.hc_mult * cfg.dim,
        cfg.vocab_size
    );
    let barrier = Arc::new(ferrite_dsv41::tp::SpinBarrier::new(world));
    let t_small = Arc::new(Mutex::new(vec![0u64; world]));
    let t_big = Arc::new(Mutex::new(vec![0u64; world]));
    let dir = dir.to_string();
    let so = so.to_string();
    let t0 = std::time::Instant::now();

    let dir_r = dir.clone();
    let so_r = so.clone();
    let cfg_r = cfg.clone();
    let ids_r = ids.clone();
    let b_r = barrier.clone();
    let ts_r = t_small.clone();
    let tb_r = t_big.clone();
    let produced: Vec<u32> = tp::run_ranks(world, move |rank| {
        let r = rank_body(
            &dir_r, &so_r, &cfg_r, &ids_r, world, rank, max_tokens, eos, &b_r, &ts_r, &tb_r, t0,
        );
        if let Err(ref e) = r {
            eprintln!("[rank {rank}] FAILED: {e}");
        }
        r
    })?;

    let tok = tokenizers::Tokenizer::from_file(format!("{dir}/tokenizer.json"))
        .map_err(|e| ferrite_types::FerriteError::Config(format!("tokenizer: {e}")))?;
    let text = tok.decode(&produced, true).unwrap_or_default();
    println!("--- generated ({}) ---", produced.len());
    println!("{text}");
        println!("--- ids: {:?}", &produced[..produced.len().min(24)]);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn rank_body(
    dir: &str,
    so: &str,
    cfg: &Dsv41Config,
    ids: &[u32],
    world: usize,
    rank: usize,
    max_tokens: usize,
    eos: Option<u32>,
    barrier: &Arc<ferrite_dsv41::tp::SpinBarrier>,
    t_small: &Arc<Mutex<Vec<u64>>>,
    t_big: &Arc<Mutex<Vec<u64>>>,
    t0: std::time::Instant,
) -> Result<Vec<u32>> {
    Device::bind_to(rank as i32)?;
    let dev = Arc::new(Device::open(so)?);
    let mut loader = Loader::new(std::path::Path::new(dir), &dev)?;
    // the engram write-back is an architectural component of this checkpoint
    // (text_config.engram_layer_ids = [1,14]); default = load its tables.
    if std::env::var("DSV41_SKIP_ENGRAM_WEIGHTS")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        loader.skip_prefixes.push("engram.embed.".into());
    }
    let w = loader.load(cfg, world, rank)?;
    dev.sync()?;
    let eng_map = std::fs::read(format!("{dir}/engram_token_map.bin"))
        .ok()
        .filter(|b| b.len() % 8 == 0 && !b.is_empty())
        .map(|b| {
            let v: Vec<i64> = b
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            ferrite_dsv41::engram::TokenMap::from_table(v, cfg.engram_compressed_vocab_size)
        });
    if rank == 0 {
        eprintln!(
            "[dsv41] engram: {}",
            if eng_map.is_some() { "token map loaded" } else { "NO token map (engram disabled)" }
        );
    }
    // peer access needs every rank's context to exist first
    barrier.wait();
    let peers = dev.enable_peer_access()?;
    if rank == 0 {
        println!(
            "[dsv41] rank0: peer access to {peers} devices; weights {:.1} GiB in {:.1}s",
            loader.uploaded as f64 / (1u64 << 30) as f64,
            t0.elapsed().as_secs_f64()
        );
    }
    barrier.wait();

    let hc_dim = cfg.hc_mult * cfg.dim;
    let vocab = cfg.vocab_size;
    let mut c_small = Collective::new(dev.clone(), world, rank, hc_dim * 4, barrier.clone())?;
    let mut c_big = Collective::new(dev.clone(), world, rank, vocab * 4, barrier.clone())?;
    {
        t_small.lock().unwrap()[rank] = c_small.staging_base();
        t_big.lock().unwrap()[rank] = c_big.staging_base();
    }
    barrier.wait();
    let ps = t_small.lock().unwrap().clone();
    let pb = t_big.lock().unwrap().clone();
    barrier.wait();
    if rank == 0 {
        println!("[dsv41] peers_small = {:x?}", ps);
        println!("[dsv41] peers_big   = {:x?}", pb);
        eprintln!("[dsv41] rank0 own staging = {:#x}", c_small.staging_base());
    }
    c_small.set_peers(ps)?;
    c_big.set_peers(pb)?;
    // the big collective is for the vocabulary gather, which arrives with the
    // head split; keep it alive so its staging address stays valid
    let _keep_big = Arc::new(c_big);

    let mut chain = DevChain::new(&dev, cfg, &w, RunOpts::from_env(), eng_map)?;
    chain.comm = Some(Arc::new(c_small));
    chain.reset()?;
    if rank == 0 {
        eprintln!("[dsv41] rank0: chain ready, entering prefill");
    }
    let t1 = std::time::Instant::now();
    let mut next_tok: u32 = 0;
    for (i, &tk) in ids.iter().enumerate() {
        next_tok = chain.step(tk, i)?;
    }
    if rank == 0 {
        println!(
            "[dsv41] prefill {} tokens in {:.2}s",
            ids.len(),
            t1.elapsed().as_secs_f64()
        );
    }
    let mut out: Vec<u32> = Vec::new();
    let t_dec = std::time::Instant::now();
    for step in 0..max_tokens {
        out.push(next_tok);
        if Some(next_tok) == eos {
            break;
        }
        next_tok = chain.step_dev(next_tok, ids.len() + step)?;
    }
    if rank == 0 {
        let el = t_dec.elapsed();
        let n = out.len().max(1);
        eprintln!(
            "[dsv41] DECODE {} tokens in {:.2}s = {:.1} tok/s ({:.2} ms/token)",
            out.len(),
            el.as_secs_f64(),
            n as f64 / el.as_secs_f64(),
            el.as_secs_f64() * 1e3 / n as f64
        );
    }
    Ok(out)
}

// ============================================================================
// HTTP serve mode: the SHARED ferrite-http stack (axum routes + SSE + the
// engine driver thread + the tokenizer boundary — the same code GLM serves
// through; see crates/ferrite-http).
//
// The model loads ONCE and every verification round is a curl against the
// OpenAI API instead of a ~80 s cold start per prompt. The only DSV41-specific
// pieces are:
//
//   * `TpRankPool` — the TP rank threads behind `StepEngine`: ONE request at a
//     time, every rank running the same step (the collective is the loop).
//     ferrite-http's `SingleFlight` adapter turns that into a full
//     `ServeEngine` (FIFO admission, retirement, cancel, telemetry), so the
//     driver/HTTP/SSE layer is identical to GLM's;
//   * `Dsv41Frame` — this checkpoint's chat frame + stop set.
//
// Everything else (request/event protocol, SSE framing, usage, cancel-on-drop,
// /v1/models, /health, /v1/stats, /shutdown) is shared.
//
// The hand-rolled std::net server this replaces is gone. It also could not have
// worked as written: it built the spin barrier INSIDE each rank's closure (so
// every rank waited alone on its own barrier — an immediate hang) and skipped
// the peer handshake (`enable_peer_access` + staging exchange + `set_peers`)
// that the all-reduce's peer stores require. The prologue below is the one-shot
// path's (verified) verbatim.
// ============================================================================

// (Dsv41Frame — the checkpoint's chat frame — now lives WITH THE MODEL at
//  `ferrite_models::dsv41::frame` (re-exported as `ferrite_dsv41::Dsv41Frame`)
//  so this runner and `ferrite-serve --model dsv41` share one layout.)

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
struct TpRankPool {
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
    let mut comm = Collective::new(dev.clone(), world, rank, hc_dim * 4, barrier.clone())?;
    {
        staging.lock().unwrap()[rank] = comm.staging_base();
    }
    barrier.wait();
    let peers = staging.lock().unwrap().clone();
    barrier.wait();
    comm.set_peers(peers)?;
    let mut chain = DevChain::new(&dev, cfg, &w, RunOpts::from_env(), eng_map)?;
    chain.comm = Some(Arc::new(comm));
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
                    match chain.step_dev(t, pos + i) {
                        Ok(next) => {
                            step_time(pos + i, st.elapsed());
                            out.push(next);
                            t = next;
                            if stop_set.contains(&next) {
                                break;
                            }
                        }
                        Err(e) => {
                            r = Err(e);
                            break;
                        }
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
fn load_eng_map(dir: &str, cfg: &Dsv41Config) -> Option<ferrite_dsv41::engram::TokenMap> {
    std::fs::read(format!("{dir}/engram_token_map.bin"))
        .ok()
        .filter(|b| b.len() % 8 == 0 && !b.is_empty())
        .map(|b| {
            let v: Vec<i64> = b
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            ferrite_dsv41::engram::TokenMap::from_table(v, cfg.engram_compressed_vocab_size)
        })
}

/// `--serve`: the shared ferrite-http stack over the TP rank pool. Binds
/// immediately; the ranks finish loading behind the first request.
fn run_serve(
    dir: &str,
    so: &str,
    cfg: &Dsv41Config,
    tp: usize,
    eos: Option<u32>,
    port: u16,
    model_name: String,
) -> Result<()> {
    // The stop set is the CHECKPOINT's: the resolved EOS from main PLUS the
    // checkpoint's own end-of-sentence special (its config ships eos_token_id
    // null, so the name-resolved id is the honest fallback). Engine retirement
    // and wire stripping both come from this one set.
    let eos_ids: Vec<u32> = eos.into_iter().collect();
    let spec = StopSpec::new(&eos_ids, &["<|end_of_sentence|>"]);
    let tok =
        ChatTokenizer::from_file_with(&std::path::Path::new(dir).join("tokenizer.json"), spec)?;
    let stops = tok.stop_ids().to_vec();
    let pool = TpRankPool::new(dir, so, cfg, tp, stops)?;
    let engine = SingleFlight::new(pool);
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    eprintln!(
        "[dsv41] serving TP{tp} {model_name} on http://{addr}/v1/chat/completions (ranks loading)"
    );
    ferrite_http::serve::launch(
        engine,
        tok,
        Arc::new(Dsv41Frame),
        ServeOptions::new(addr, model_name),
    )
}
