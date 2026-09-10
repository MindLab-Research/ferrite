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

use ferrite_dsv41::chain_dev::{DevChain, RunOpts};
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
    let skip_engram_w = std::env::var("DSV41_SKIP_ENGRAM_WEIGHTS").map(|v| v != "0").unwrap_or(true);

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
    let tok = tokenizers::Tokenizer::from_file(format!("{dir}/tokenizer.json"))
        .map_err(|e| ferrite_types::FerriteError::Config(format!("tokenizer: {e}")))?;
    let enc = tok
        .encode(prompt.clone(), true)
        .map_err(|e| ferrite_types::FerriteError::Config(format!("encode: {e}")))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();
    if ids.is_empty() {
        return Err(ferrite_types::FerriteError::Config("empty prompt".into()));
    }
    eprintln!("[dsv41] prompt {} tokens: {:?}", ids.len(), &ids[..ids.len().min(8)]);

    // ---- weights ----
    let t0 = std::time::Instant::now();
    let mut loader = Loader::new(std::path::Path::new(&dir), &dev)?;
    if skip_engram_w {
        loader.skip_prefixes.push("engram.embed.".into());
    }
    let w = loader.load(&cfg, tp, rank)?;
    dev.sync()?;
    eprintln!(
        "[dsv41] weights loaded: {:.1} GiB in {:.1}s",
        loader.uploaded as f64 / (1u64 << 30) as f64,
        t0.elapsed().as_secs_f64()
    );

    // ---- chain ----
    let opts = RunOpts::from_env();
    let mut chain = DevChain::new(&dev, &cfg, &w, opts)?;
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
