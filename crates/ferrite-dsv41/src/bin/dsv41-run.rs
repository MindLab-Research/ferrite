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

use std::sync::{Arc, Barrier, Mutex};

use ferrite_dsv41::chain_dev::{DevChain, RunOpts};
use ferrite_dsv41::tp::{self, Collective};
use ferrite_dsv41::config::Dsv41Config;
use ferrite_dsv41::device::Device;
use ferrite_dsv41::load::Loader;
use ferrite_types::Result;

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
    // end-of-sequence id: prefer generation_config.json, fall back to the
    // tokenizer's own metadata
    let eos: Option<u32> = std::fs::read_to_string(format!("{dir}/generation_config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| v.get("eos_token_id").and_then(|e| e.as_u64()).map(|e| e as u32));
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
        Err(_) => tok
            .encode(prompt.clone(), true)
            .map_err(|e| ferrite_types::FerriteError::Config(format!("encode: {e}")))?
            .get_ids()
            .to_vec(),
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
    let mut logits = Vec::new();
    for (i, &t) in ids.iter().enumerate() {
        logits = chain.step(t, i)?;
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
    for step in 0..max_tokens {
        let mut best = 0usize;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        let next = best as u32;
        if step < 3 {
            let mut top: Vec<(usize, f32)> =
                logits.iter().copied().enumerate().collect::<Vec<_>>().iter().map(|&(i, v)| (i, v)).collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("[top5] step {step} n={} : {:?}", logits.len(),
                &top[..5.min(top.len())].iter().map(|(i, v)| (*i, *v)).collect::<Vec<_>>());
        }
        produced.push(next);
        if Some(next) == eos {
            break;
        }
        logits = chain.step(next, pos0 + step)?;
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
    let barrier = Arc::new(Barrier::new(world));
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
    barrier: &Arc<Barrier>,
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
    let mut logits = Vec::new();
    for (i, &tk) in ids.iter().enumerate() {
        logits = chain.step(tk, i)?;
    }
    if rank == 0 {
        println!(
            "[dsv41] prefill {} tokens in {:.2}s",
            ids.len(),
            t1.elapsed().as_secs_f64()
        );
    }
    let mut out: Vec<u32> = Vec::new();
    for step in 0..max_tokens {
        let mut best = 0usize;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        let next = best as u32;
        if rank == 0
            && step < 3
            && std::env::var("DSV41_TOP5").map(|v| v != "0").unwrap_or(false)
        {
            let mut top: Vec<(usize, f32)> =
                logits.iter().copied().enumerate().collect::<Vec<_>>();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!(
                "[top5] step {step} n={} top={:?} span={:?}",
                logits.len(),
                &top[..4.min(top.len())],
                (logits.iter().cloned().fold(f32::MIN, f32::max),
                 logits.iter().cloned().fold(f32::MAX, f32::min))
            );
        }
        out.push(next);
        if Some(next) == eos {
            break;
        }
        logits = chain.step(next, ids.len() + step)?;
    }
    Ok(out)
}
