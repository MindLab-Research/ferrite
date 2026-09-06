//! ferrite-serve: load the real GLM-5.3-Flash checkpoint and run inference.
//!
//! Usage:
//!   ferrite-serve --model-dir /path/to/GLM-5.3-Flash [--max-tokens N]
//!                 [--backend cpu|cuda] [--tp N] [--lib /path/libferrite_kernels.so]
//!                 [--prompt "..."]
//!
//! CPU: single-process Engine (f32, needs ~700 GB RAM).
//! CUDA: TP=N cluster — one CudaBackend per GPU (device = rank), weights
//! sharded by shard_weights_tp, per-layer all-reduce via the TpCluster.

use std::path::PathBuf;

use ferrite_model::{load_hf_checkpoint, Glm53FlashConfig};

/// GLM chat format: <|prompt|>\n...<|im_end|>\n<|answer|>\n
/// (token ids resolved from the tokenizer; falls back to raw text).
fn wrap_prompt(text: &str) -> String {
    format!("<|user|>\n{text}</s>\n<|assistant|>\n")
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut get_arg = |name: &str, default: &str| -> String {
        if let Some(p) = args.iter().position(|a| a == name) {
            if p + 1 < args.len() {
                let v = args[p + 1].clone();
                args.drain(p..=p + 1);
                return v;
            }
        }
        default.to_string()
    };
    let model_dir = PathBuf::from(get_arg("--model-dir", "."));
    let max_tokens: usize = get_arg("--max-tokens", "32").parse().unwrap_or(32);
    let backend = get_arg("--backend", "cpu");
    let tp: usize = get_arg("--tp", "8").parse().unwrap_or(8);
    let lib = get_arg("--lib", "kernels/cuda/libferrite_kernels.so");
    let prompt = get_arg("--prompt", "你好，介绍一下你自己。");

    // ---- built-in CPU profiler (Go-pprof style): FERRITE_PPROF=1 starts a
    // 1000 Hz SIGPROF sampler; on exit the flamegraph lands in
    // FERRITE_PPROF_OUT (default serve.flamegraph.svg). Replaces external
    // gdb-attach sampling — continuous, zero-perturbation, standard tooling.
    let profiler = std::env::var_os("FERRITE_PPROF").map(|_| {
        let g = pprof::ProfilerGuardBuilder::default()
            .frequency(1000)
            .blocklist(&["libc", "libgcc", "pthread", "vdso"])
            .build()
            .expect("pprof guard (FERRITE_PPROF=1)");
        println!("[serve] pprof sampling active (1000 Hz; flamegraph on exit)");
        g
    });

    // ---- config ----
    let cfg_path = model_dir.join("config.json");
    let cfg_str = std::fs::read_to_string(&cfg_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", cfg_path.display()));
    let cfg = Glm53FlashConfig::from_json_str(&cfg_str)
        .unwrap_or_else(|e| panic!("parse config.json: {e}"));
    println!(
        "[serve] config ok: {} layers, vocab {}",
        cfg.num_hidden_layers, cfg.vocab_size,
    );

    // ---- weights (FP8 dequant + name mapping; large, ~660 GB f32 in RAM) ----
    println!("[serve] loading checkpoint from {} ...", model_dir.display());
    let t0 = std::time::Instant::now();
    let (weights, rep) = load_hf_checkpoint(&model_dir, &cfg)
        .unwrap_or_else(|e| panic!("load checkpoint: {e}"));
    println!(
        "[serve] loaded {} tensors in {:.1}s (fp8-dequant: {}, fused: {}, skipped: {})",
        rep.tensors_loaded,
        t0.elapsed().as_secs_f32(),
        rep.fp8_dequantized,
        rep.fused_concat,
        rep.skipped_unsupported.len(),
    );
    println!("[serve] mem RSS after load: {:.1} GB", rss_gb());

    // ---- tokenizer ----
    let tok_path = model_dir.join("tokenizer.json");
    let tok = tokenizers::Tokenizer::from_file(tok_path)
        .unwrap_or_else(|e| panic!("load tokenizer: {e}"));
    let enc = tok
        .encode(wrap_prompt(&prompt), false)
        .unwrap_or_else(|e| panic!("encode: {e}"));
    let ids: Vec<u32> = enc.get_ids().to_vec();
    println!("[serve] prompt: {n} tokens", n = ids.len());

    // ---- inference ----
    // Stop set: primary <|end|> from generation_config.json PLUS turn
    // boundary specials — the model emits <|user|> after its answer and
    // generation must respect it (user-visible contract).
    let mut stop: Vec<u32> = vec![154820u32]; // <|end|>
    for special in ["<|user|>", "<|endoftext|>", "<|observation|>", "<|endoftext|>"] {
        if let Some(id) = tok.token_to_id(special) {
            if !stop.contains(&id) {
                stop.push(id);
            }
        }
    }
    eprintln!("[serve] stop tokens: {stop:?}");
    let t1 = std::time::Instant::now();
    let world_tp = if backend == "cuda" { tp } else { 1 };
    let new_tokens: Vec<u32> = match backend.as_str() {
        "cuda" => run_cuda(cfg, weights, &ids, max_tokens, &stop, &lib, world_tp),
        _ => run_cpu(cfg, weights, &ids, max_tokens, &stop),
    };
    let dt = t1.elapsed().as_secs_f64();
    let text = tok
        .decode(&new_tokens, false)
        .unwrap_or_else(|e| panic!("decode: {e}"));
    println!(
        "[serve] generated {} tokens in {dt:.1}s ({:.2} tok/s)",
        new_tokens.len(),
        new_tokens.len() as f64 / dt.max(1e-9)
    );
    println!("---- output ----");
    println!("{text}");

    // ---- pprof dump (after everything; the profile spans load + warmup +
    // generate — the flamegraph's self time tells the story per phase) ----
    if let Some(g) = &profiler {
        match g.report().build() {
            Ok(report) => {
                let path = std::env::var("FERRITE_PPROF_OUT")
                    .unwrap_or_else(|_| "serve.flamegraph.svg".to_string());
                match std::fs::File::create(&path).map_err(|e| e.to_string()).and_then(|f| {
                    report.flamegraph(f).map_err(|e| e.to_string())
                }) {
                    Ok(()) => println!("[serve] pprof flamegraph → {path}"),
                    Err(e) => eprintln!("[serve] pprof flamegraph write failed: {e}"),
                }
            }
            Err(e) => eprintln!("[serve] pprof report build failed: {e}"),
        }
    }

    // One-shot process: skip the exit-time teardown. Dropping the cluster
    // (1.17TB of weights + 4 CUDA contexts) SEGFAULTS at exit (EXIT 139) —
    // which also LOSES nsys's CUPTI activity buffers (no kernel data in
    // the report). The OS reclaims everything anyway.
    std::process::exit(0);
}

/// Single-process CPU inference (CpuBackend, f32).
fn run_cpu(
    cfg: Glm53FlashConfig,
    weights: ferrite_model::Weights,
    ids: &[u32],
    max_tokens: usize,
    stop: &[u32],
) -> Vec<u32> {
    use ferrite_exec::Engine;
    use ferrite_kernel::CpuBackend;
    let mut engine = Engine::new(cfg, weights, CpuBackend::new());
    engine.eos_token = stop.first().copied();
    let seq = engine
        .submit(ids.to_vec(), max_tokens)
        .unwrap_or_else(|e| panic!("submit: {e}"));
    let out = engine
        .run_until_done(seq)
        .unwrap_or_else(|e| panic!("run: {e}"));
    if out.len() > ids.len() {
        out[ids.len()..].to_vec()
    } else {
        out
    }
}

/// TP=N on-device inference: one CudaBackend per GPU, weights sharded via
/// shard_weights_tp, per-layer CPU-side all-reduce (TpCluster).
#[cfg(feature = "cuda")]
fn run_cuda(
    cfg: Glm53FlashConfig,
    weights: ferrite_model::Weights,
    ids: &[u32],
    max_tokens: usize,
    stop: &[u32],
    lib: &str,
    tp: usize,
) -> Vec<u32> {
    use ferrite_exec::tp::TpCluster;
    use ferrite_kernel::CudaBackend;

    let world = tp.max(1);
    let mut cluster = TpCluster::new(cfg, &weights, world, |rank| {
        CudaBackend::with_device(lib, rank as i32)
            .unwrap_or_else(|e| panic!("cuda backend rank {rank}: {e}"))
    });
    println!("[serve] cuda TP cluster up: {world} rank(s)");

    // Weights resident on the GPU (TileRT model): upload every shard's
    // full weight set once at startup — bf16 for 2-D matmul weights
    // (~142GB/rank at TP4 vs 285GB f32 which does not fit a 275GB B300),
    // f32 for 1-D (norms/logdecay/biases). After this, per-op traffic is
    // activations only.
    //
    // All ranks preload CONCURRENTLY — cudaSetDevice is thread-local, so
    // each rank thread binds its own device and streams its shard over
    // PCIe in parallel (serial was 606.8s; PCIe is per-device so the 4
    // uploads overlap almost perfectly).
    {
        let t0 = std::time::Instant::now();
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (rank, shard) in cluster.shards.iter().enumerate() {
                handles.push(scope.spawn(move || {
                    let mut n_2d = 0usize;
                    let mut n_1d = 0usize;
                    for (_name, t) in shard.weights.iter() {
                        shard
                            .backend
                            .preload_weight(t)
                            .unwrap_or_else(|e| panic!("preload rank {rank} weight {_name}: {e}"));
                        if t.shape.0.len() >= 2 {
                            n_2d += 1;
                        } else {
                            n_1d += 1;
                        }
                    }
                    println!(
                        "[serve] rank {rank}: {n_2d} x2d (bf16-resident) + {n_1d} x1d (f32) weights on device"
                    );
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });
        println!(
            "[serve] weights resident in {:.1}s (parallel across ranks)",
            t0.elapsed().as_secs_f32()
        );
    }

    let seq = 1u64;
    let tp0 = std::time::Instant::now();
    cluster
        .prefill_chunk(seq, ids)
        .unwrap_or_else(|e| panic!("prefill: {e}"));
    let tp1 = std::time::Instant::now();
    println!(
        "[serve] prefill {} tokens in {:.3}s",
        ids.len(),
        tp1.duration_since(tp0).as_secs_f32()
    );
    let mut out = Vec::new();
    let mut prev_rt_len = cluster
        .shards
        .first()
        .and_then(|s| s.seq_runtime(seq).map(|rt| rt.tokens.len()))
        .unwrap_or(0);
    // ncu capture window (ncu --profile-from-start off + FERRITE_NCU=1):
    // the 80s weight load (38k H2D uploads ncu intercepts at ms-level
    // cost each — stalls the run 10+ min) and prefill run at full speed;
    // only the decode loop is profiled. cuProfilerStop flushes the window.
    let ncu_win = std::env::var_os("FERRITE_NCU").is_some();
    if ncu_win {
        #[cfg(feature = "cuda")]
        ferrite_kernel::cuda::profiler_start();
    }
    for i in 0..max_tokens {
        let tok = cluster
            .decode_step(seq)
            .unwrap_or_else(|e| panic!("decode step {i}: {e}"));
        if stop.contains(&tok) {
            break;
        }
        // MTP: one decode step emits k=1..3 tokens (rt.tokens) — collect the
        // FULL incremental stream, not just the step's last token (dropping
        // accept-2/3's earlier tokens garbles the text mid-character).
        if let Some(rt) = cluster.shards.first().and_then(|s| s.seq_runtime(seq)) {
            let new_tokens: Vec<u32> = rt.tokens[prev_rt_len..].to_vec();
            if !new_tokens.is_empty() {
                if let Some(&last_new) = new_tokens.last() {
                    if stop.contains(&last_new) {
                        out.extend_from_slice(&new_tokens[..new_tokens.len() - 1]);
                        prev_rt_len = rt.tokens.len();
                        break;
                    }
                }
                out.extend_from_slice(&new_tokens);
                prev_rt_len = rt.tokens.len();
            }
        } else {
            out.push(tok);
        }
        if std::env::var_os("FERRITE_TRACE_TOK").is_some() {
            println!("[serve] tok {i}: {tok}");
        }
    }
    if ncu_win {
        #[cfg(feature = "cuda")]
        ferrite_kernel::cuda::profiler_stop();
    }
    let td1 = std::time::Instant::now();
    let decode_s = td1.duration_since(tp1).as_secs_f32();
    let gen = out.len();
    // MTP: out.len() counts decode STEPS; the real generated token count is
    // rt.tokens.len() - prompt (accept-2/3 push 2/3 tokens per step).
    let real: Option<usize> = cluster
        .shards
        .first()
        .and_then(|s| s.seq_runtime(seq).map(|rt| rt.tokens.len().saturating_sub(ids.len())));
    if gen > 0 {
        println!(
            "[serve] decode: {gen} steps in {decode_s:.3}s = {:.1} steps/s | real {} tokens = {:.1} tok/s (steady state; weights-preload + prefill excluded)",
            gen as f32 / decode_s,
            real.map(|r| r.to_string()).unwrap_or_else(|| "?".into()),
            real.map(|r| r as f32 / decode_s).unwrap_or(0.0),
        );
    }
    // Skip the exit-time teardown: dropping 1.17TB of f32 weights walks
    // ~37k large glibc chunks through munmap (~70s, observed 6/6 in gdb
    // stack samples INSIDE the generation timer), and dropping the cluster
    // cudaFrees 568GB of resident bf16 weight across 4 ranks. The OS and
    // CUDA context reclaim everything at process exit; serve is one-shot.
    std::mem::forget(cluster);
    std::mem::forget(weights);
    out
}

#[cfg(not(feature = "cuda"))]
fn run_cuda(
    _cfg: Glm53FlashConfig,
    _weights: ferrite_model::Weights,
    _ids: &[u32],
    _max_tokens: usize,
    _eos: u32,
    _lib: &str,
) -> Vec<u32> {
    panic!("ferrite-serve was built without the cuda feature (rebuild with --features ferrite-serve? no — build ferrite-kernel --features cuda first)");
}

fn rss_gb() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(v) = line.strip_prefix("VmRSS:") {
            let kb: f64 = v
                .trim()
                .trim_end_matches(" kB")
                .parse()
                .unwrap_or(0.0);
            return kb / 1024.0 / 1024.0;
        }
    }
    0.0
}
