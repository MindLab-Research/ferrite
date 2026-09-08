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

/// The CUDA GpuEngine (per-seq TpCluster decode behind the ServeEngine
/// seam) — the --serve mode's engine.
#[cfg(feature = "cuda")]
mod gpu_engine;

/// Direct mmap preload (disk→GPU without CPU materialization — the
/// FERRITE_DIRECT_LOAD=1 path): TP windows mirrored from shard_weights_tp,
/// mmap slices → the backend's direct-preload entry points.
#[cfg(feature = "cuda")]
mod direct_load;

/// GLM chat format: <|prompt|>\n...<|im_end|>\n<|answer|>\n
/// (token ids resolved from the tokenizer; falls back to raw text).
fn wrap_prompt(text: &str) -> String {
    format!("<|user|>\n{text}</s>\n<|assistant|>\n")
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let serve = args.iter().any(|a| a == "--serve");
    args.retain(|a| a != "--serve");
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
    // --serve mode flags (the OpenAI HTTP/SSE API — see run_serve).
    let port: u16 = get_arg("--port", "8080").parse().unwrap_or(8080);
    let max_seqs: usize = get_arg("--max-seqs", "4").parse().unwrap_or(4);
    let model_name = get_arg("--model-name", "glm-5.3-flash");

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

    // ---- weights ----
    // DIRECT mmap path is the DEFAULT for --backend cuda (FERRITE_LEGACY_LOAD=1
    // restores the f32-materializing legacy loader for debugging): placeholder
    // tensors (shape-real, 4-elem data stubs) feed TpCluster's shape-only
    // shard splits, and the preload phase streams the mmap slices into the
    // device caches (bf16 verbatim / fp8 GPU dequant / bf16→f32 expand / fp8
    // native for MoE experts). The CPU never materializes a weight: legacy
    // peak RSS was ~660GB of f32; the direct path maps the safetensors files
    // and the page cache is the only host-side copy.
    let use_direct = std::env::var_os("FERRITE_LEGACY_LOAD").is_none() && backend == "cuda";
    if use_direct && std::env::var_os("FERRITE_W8A8").is_some() {
        panic!("FERRITE_DIRECT_LOAD and FERRITE_W8A8 are mutually exclusive in v1 (the direct path dequants fp8 → bf16 resident)");
    }
    let t0 = std::time::Instant::now();
    let direct: Option<std::sync::Arc<ferrite_model::direct::DirectView>> = if use_direct {
        let dv = ferrite_model::direct::load_direct(&model_dir, &cfg)
            .unwrap_or_else(|e| panic!("direct mmap load: {e}"));
        println!(
            "[serve] direct mmap view: {} weights mapped ({} views, page cache = the only host copy)",
            dv.placeholders.len(),
            dv.views.len(),
        );
        Some(std::sync::Arc::new(dv))
    } else {
        None
    };
    let (weights, weights8, _rep) = if let Some(dv) = &direct {
        // placeholder table: shape-real stubs (the cluster's shard splits
        // are shape-only on them — row_split/col_split's stub branch), the
        // device caches key on their pointers; W8A8 is off (guarded above).
        (dv.placeholders.clone(), Default::default(), ferrite_model::CheckpointReport::default())
    } else {
        // legacy: FP8 dequant + name mapping on the CPU (large, ~660 GB f32)
        println!("[serve] loading checkpoint from {} ...", model_dir.display());
        let (w, w8, r) = load_hf_checkpoint(&model_dir, &cfg)
            .unwrap_or_else(|e| panic!("load checkpoint: {e}"));
        println!(
            "[serve] loaded {} tensors in {:.1}s (fp8-dequant: {}, fused: {}, skipped: {})",
            r.tensors_loaded,
            t0.elapsed().as_secs_f32(),
            r.fp8_dequantized,
            r.fused_concat,
            r.skipped_unsupported.len(),
        );
        (w, w8, r)
    };
    if direct.is_none() {
        println!("[serve] mem RSS after load: {:.1} GB", rss_gb());
    }

    // ---- serve mode (--serve): the OpenAI-compatible HTTP/SSE API ----
    // (SSE streaming + concurrent requests over the CUDA cluster; see
    // run_serve. Runs until ctrl-c — never returns.)
    if serve {
        if backend != "cuda" {
            panic!("--serve requires --backend cuda (the mock HTTP serve is the ferrite-http binary)");
        }
        #[cfg(feature = "cuda")]
        run_serve(
            cfg,
            weights,
            weights8,
            &lib,
            tp,
            &model_dir,
            port,
            max_seqs,
            model_name,
            direct,
        );
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (weights, weights8, lib, tp, model_dir, port, max_seqs, model_name);
            panic!("ferrite-serve was built without the cuda feature");
        }
    }

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
        "cuda" => run_cuda(cfg, weights, weights8, &ids, max_tokens, &stop, &lib, world_tp, direct),
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
    weights8: ferrite_model::Weights8,
    ids: &[u32],
    max_tokens: usize,
    stop: &[u32],
    lib: &str,
    tp: usize,
    direct: Option<std::sync::Arc<ferrite_model::direct::DirectView>>,
) -> Vec<u32> {
    use ferrite_exec::tp::TpCluster;
    use ferrite_kernel::CudaBackend;

    let world = tp.max(1);
    let mut cluster = TpCluster::new(cfg.clone(), &weights, world, |rank| {
        CudaBackend::with_device(lib, rank as i32)
            .unwrap_or_else(|e| panic!("cuda backend rank {rank}: {e}"))
    });
    println!("[serve] cuda TP cluster up: {world} rank(s)");

    // fp8 bypass shards (native F8 bytes + 128-block scale): same TP
    // classification as the f32 shards; engine w8() lookup misses → bf16 path
    // (safe per-weight fallback, draft/verify stay in one domain per weight).
    cluster.set_fp8(&weights8);
    println!("[serve] fp8 bypass: {} full / {} rank-0 shards", weights8.len(), cluster.shards[0].weights8.len());

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
    //
    // DIRECT path (FERRITE_DIRECT_LOAD=1): the mmap slices stream into
    // the device caches through the direct-preload entry points (bf16
    // verbatim / fp8 GPU dequant / bf16→f32 expand / pitched column
    // windows) — the placeholder stubs in shard.weights carry the shapes
    // and the runtime cache keys; the CPU f32 materialization never
    // happens (the legacy branch below is the fallback).
    {
        let t0 = std::time::Instant::now();
        let cfgref = &cfg; // &Config captured by ref into every rank closure
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (rank, shard) in cluster.shards.iter().enumerate() {
                let dv = direct.clone();
                handles.push(scope.spawn(move || {
                    if let Some(dv) = &dv {
                        let st = crate::direct_load::direct_preload_shard(
                            &shard.backend, dv, cfgref, rank, world, &shard.weights,
                        )
                        .unwrap_or_else(|e| panic!("direct preload rank {rank}: {e}"));
                        println!(
                            "[serve] rank {rank}: DIRECT mmap preload (bf16 rows {} cols {} full {} / fp8 rows {} cols {} / f32 {} / experts {})",
                            st.bf16_rows, st.bf16_cols, st.bf16_full,
                            st.fp8_rows, st.fp8_cols, st.f32_expand, st.experts,
                        );
                        return;
                    }
                    let mut n_2d = 0usize;
                    let mut n_1d = 0usize;
                    let mut n_fp8 = 0usize;
                    for (name, t) in shard.weights.iter() {
                        // fp8 bypass: skip the bf16 upload for registered hits —
                        // matmul_dev serves these from the native-F8 resident copy
                        // (registered at set_fp8). The fused-MoE kernels still read
                        // the bf16 pointer tables (routed experts + shared expert),
                        // so those weights keep their bf16 residency.
                        let moe_bf16 =
                            name.contains(".experts.") || name.contains(".shared_expert.");
                        if !moe_bf16 && shard.backend.fp8_hit(t) {
                            n_fp8 += 1;
                            continue;
                        }
                        if std::env::var_os("FERRITE_FP8_DEBUG").is_some()
                            && !moe_bf16
                            && t.shape.0.len() >= 2
                            && shard.weights8.contains_key(name)
                        {
                            // registered fp8 but preload lookup missed: the
                            // (ptr, numel) key diverged between register and here.
                            eprintln!(
                                "[fp8dbg] PRELOAD-MISS {name} ptr={:x} numel={} map={}",
                                t.as_slice().as_ptr() as usize, t.numel(), shard.backend.fp8_registered()
                            );
                        }
                        shard
                            .backend
                            .preload_weight(t)
                            .unwrap_or_else(|e| panic!("preload rank {rank} weight {name}: {e}"));
                        if t.shape.0.len() >= 2 {
                            n_2d += 1;
                        } else {
                            n_1d += 1;
                        }
                    }
                    println!(
                        "[serve] rank {rank}: {n_2d} x2d (bf16-resident) + {n_1d} x1d (f32) + {n_fp8} fp8-resident weights on device"
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
    let mut steps_done: usize = 0;
    let mut prev_rt_len = cluster
        .shards
        .first()
        .and_then(|s| s.seq_runtime(seq).map(|rt| rt.tokens.len()))
        .unwrap_or(0);
    // ncu capture window (ncu --profile-from-start off + FERRITE_NCU=1):
    // skip step 0 (the mega/verify graph CAPTURE — ncu's per-kernel replay
    // conflicts with stream capture and hangs the run) and the 80s weight
    // load (ncu intercepts each H2D at ms cost); the window opens at the
    // SECOND decode step (pure graph replays) and closes after the loop.
    let ncu_win = std::env::var_os("FERRITE_NCU").is_some();
    let mut ncu_started = false;
    for i in 0..max_tokens {
        steps_done += 1;
        if ncu_win && i == 1 {
            #[cfg(feature = "cuda")]
            ferrite_kernel::cuda::profiler_start();
            ncu_started = true;
        }
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
    if ncu_started {
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
        let acc = if steps_done > 0 { real.unwrap_or(gen) as f32 / steps_done as f32 } else { 1.0 };
        println!(
            "[serve] decode: {steps_done} steps in {decode_s:.3}s = {:.1} steps/s | real {} tokens = {:.1} tok/s | accept {:.2} (steady state; weights-preload + prefill excluded)",
            steps_done as f32 / decode_s,
            real.map(|r| r.to_string()).unwrap_or_else(|| "?".into()),
            real.map(|r| r as f32 / decode_s).unwrap_or(0.0),
            acc,
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
    _direct: Option<std::sync::Arc<ferrite_model::direct::DirectView>>,
) -> Vec<u32> {
    panic!("ferrite-serve was built without the cuda feature (rebuild with --features ferrite-serve? no — build ferrite-kernel --features cuda first)");
}

/// --serve mode: the OpenAI-compatible HTTP/SSE API over the CUDA cluster.
/// Same bring-up as run_cuda (cluster + fp8 bypass + concurrent resident
/// preload), then the GpuEngine (per-seq prefill + round-robin mega-graph
/// decode behind the ferrite-http driver) + the axum router:
/// POST /v1/chat/completions (stream=true → SSE chat.completion.chunk +
/// [DONE]; false → one JSON), /v1/models, /health, /v1/stats. Concurrency:
/// --max-seqs live requests interleaved (each ~1/N of the single-stream
/// rate; true batched decode is the next phase — the scheduler's
/// ExecBackend seam). Runs until ctrl-c, then process::exit(0) (no
/// exit-time cluster drop — same reason as the one-shot path: the 1.17TB
/// teardown segfaults).
#[cfg(feature = "cuda")]
fn run_serve(
    cfg: Glm53FlashConfig,
    weights: ferrite_model::Weights,
    weights8: ferrite_model::Weights8,
    lib: &str,
    tp: usize,
    model_dir: &std::path::Path,
    port: u16,
    max_seqs: usize,
    model_name: String,
    direct: Option<std::sync::Arc<ferrite_model::direct::DirectView>>,
) -> ! {
    use ferrite_exec::tp::TpCluster;
    use ferrite_http::api::{router, AppState};
    use ferrite_http::driver::EngineDriver;
    use ferrite_http::tokenizer::ChatTokenizer;
    use ferrite_kernel::CudaBackend;
    use std::sync::atomic::AtomicU64;
    use std::sync::Arc;

    let world = tp.max(1);
    let mut cluster = TpCluster::new(cfg.clone(), &weights, world, |rank| {
        CudaBackend::with_device(lib, rank as i32)
            .unwrap_or_else(|e| panic!("cuda backend rank {rank}: {e}"))
    });
    println!("[serve] cuda TP cluster up: {world} rank(s)");
    cluster.set_fp8(&weights8);
    println!(
        "[serve] fp8 bypass: {} full / {} rank-0 shards",
        weights8.len(),
        cluster.shards[0].weights8.len()
    );

    // Resident preload (concurrent per rank — same as run_cuda). DIRECT
    // path: the mmap slices stream into the device caches (no CPU f32
    // materialization); legacy: preload_weight over the f32 tensors.
    {
        let t0 = std::time::Instant::now();
        let cfgref = &cfg; // &Config captured by ref into every rank closure
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (rank, shard) in cluster.shards.iter().enumerate() {
                let dv = direct.clone();
                handles.push(scope.spawn(move || {
                    if let Some(dv) = &dv {
                        let st = crate::direct_load::direct_preload_shard(
                            &shard.backend, dv, cfgref, rank, world, &shard.weights,
                        )
                        .unwrap_or_else(|e| panic!("direct preload rank {rank}: {e}"));
                        println!(
                            "[serve] rank {rank}: DIRECT mmap preload (bf16 rows {} cols {} full {} / fp8 rows {} cols {} / f32 {} / experts {})",
                            st.bf16_rows, st.bf16_cols, st.bf16_full,
                            st.fp8_rows, st.fp8_cols, st.f32_expand, st.experts,
                        );
                        return;
                    }
                    let mut n_2d = 0usize;
                    let mut n_1d = 0usize;
                    let mut n_fp8 = 0usize;
                    for (name, t) in shard.weights.iter() {
                        let moe_bf16 =
                            name.contains(".experts.") || name.contains(".shared_expert.");
                        if !moe_bf16 && shard.backend.fp8_hit(t) {
                            n_fp8 += 1;
                            continue;
                        }
                        shard
                            .backend
                            .preload_weight(t)
                            .unwrap_or_else(|e| panic!("preload rank {rank} weight {name}: {e}"));
                        if t.shape.0.len() >= 2 {
                            n_2d += 1;
                        } else {
                            n_1d += 1;
                        }
                    }
                    println!(
                        "[serve] rank {rank}: {n_2d} x2d (bf16-resident) + {n_1d} x1d (f32) + {n_fp8} fp8-resident weights on device"
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

    // Tokenizer (chat template CLI-parity + the stop set) — the HTTP
    // layer renders/encodes; the engine retires on the same stops.
    let tok = ChatTokenizer::from_file(&model_dir.join("tokenizer.json"))
        .unwrap_or_else(|e| panic!("load tokenizer: {e}"));
    let stops = tok.stop_ids().to_vec();

    // The GPU engine + the driver (one engine thread owning the cluster;
    // HTTP on tokio — submit/cancel over mpsc, events back per request).
    // tick_interval ZERO: the decode step is GPU-bound (~20ms/seq/step
    // round-robin) — no pacing needed.
    let engine = crate::gpu_engine::GpuEngine::new(cluster, stops, max_seqs);
    let handle = EngineDriver::spawn_with(engine, std::time::Duration::ZERO);

    let state = AppState {
        handle,
        tok: Arc::new(tok),
        model_name: model_name.clone(),
        req_counter: Arc::new(AtomicU64::new(1)),
    };
    let app = router(state);
    let addr: std::net::SocketAddr = format!("0.0.0.0:{port}")
        .parse()
        .unwrap_or_else(|_| std::net::SocketAddr::from(([0, 0, 0, 0], port)));
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .unwrap_or_else(|e| panic!("bind {addr}: {e}"));
        println!(
            "[serve] serving {model_name} on http://{addr}/v1/chat/completions (SSE + concurrent, max_seqs={max_seqs})"
        );
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
                eprintln!("[serve] ctrl-c: shutting down");
            })
            .await
            .expect("serve");
    });
    // NO exit-time teardown: dropping the cluster (1.17TB weights + CUDA
    // contexts) segfaults (EXIT 139 — the one-shot path's known issue);
    // the driver thread + its engine leak with the process instead.
    std::process::exit(0);
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
