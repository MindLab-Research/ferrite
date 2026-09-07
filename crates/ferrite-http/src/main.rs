//! ferrite-http: the OpenAI-compatible HTTP/SSE serving layer.
//!
//! Layers (top→down):
//!
//! - `api` — axum routes: `/v1/chat/completions` (stream=true → SSE
//!   `chat.completion.chunk` frames + `[DONE]`; false → one JSON),
//!   `/v1/models`, `/health`, `/v1/stats` (radix + hicache telemetry).
//! - `sse` — the OpenAI wire objects (`chat.completion.chunk` frames,
//!   usage with `cached_tokens` — the radix prefix-hit number).
//! - `driver` — the engine thread: one writer owning the scheduler,
//!   commands in (submit/cancel) via tokio mpsc, events out per request.
//!   Client disconnects cancel through a Drop guard on the SSE body.
//! - `engine` / `host_engine` — the request/event contract and the
//!   deterministic host backend (the full scheduler path — admission,
//!   radix resume, MTP accept folding, retirement — with mock compute;
//!   the CUDA backend implements the same `ExecBackend` seam).
//! - `tokenizer` — chat template (GLM frame, ferrite-serve parity) +
//!   encode/decode (real HF tokenizer or the byte codec for mock mode).
//!
//! `main` here is the bring-up binary; `ferrite-serve` (the GPU binary)
//! embeds the same router with the CUDA backend once it lands behind
//! `ExecBackend`.

use ferrite_dispatch::batch::SchedConfig;
use ferrite_http::api::{router, AppState};
use ferrite_http::driver::EngineDriver;
use ferrite_http::tokenizer::ChatTokenizer;
use std::net::SocketAddr;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

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
    let host = get_arg("--host", "0.0.0.0");
    let port: u16 = get_arg("--port", "8080").parse().unwrap_or(8080);
    let model_dir = get_arg("--model-dir", "");
    let model_name = get_arg("--model", "glm-5.3-flash");

    // Tokenizer: real HF tokenizer when a model dir is given, the
    // byte-level codec otherwise (mock mode — full stack on a laptop).
    let tok = if model_dir.is_empty() {
        eprintln!("[http] mock mode: byte tokenizer + deterministic generation (no model files)");
        ChatTokenizer::stub()
    } else {
        let path = std::path::Path::new(&model_dir).join("tokenizer.json");
        match ChatTokenizer::from_file(&path) {
            Ok(t) => {
                eprintln!("[http] tokenizer: {}", path.display());
                t
            }
            Err(e) => {
                eprintln!("[http] tokenizer load failed ({e}); falling back to byte codec");
                ChatTokenizer::stub()
            }
        }
    };

    // Engine: the host (deterministic) backend over the real scheduler
    // (radix + hicache + MTP batching + page budgets — everything but
    // the transformer compute). The scheduler config is the deployment
    // default (32 decode rows = max bucket, 512-token prefill chunks,
    // radix 4096 nodes over 16-token pages, tier capacities).
    // Tick pacing: 5ms — the mock tick is microseconds (the real engine's
    // verify replay is ~27ms GPU-bound); unpaced busy loops would spin
    // a core and starve the async runtime.
    let sched_cfg = SchedConfig::default();
    let stop_id = ferrite_http::host_engine::STOP_ID;
    let handle = EngineDriver::spawn(sched_cfg, stop_id, std::time::Duration::from_millis(5));

    let state = AppState {
        handle: handle.clone(),
        tok: Arc::new(tok),
        model_name: model_name.clone(),
        req_counter: Arc::new(AtomicU64::new(1)),
    };

    let app = router(state);
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .unwrap_or_else(|_| SocketAddr::from(([0, 0, 0, 0], port)));

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .unwrap_or_else(|e| panic!("bind {addr}: {e}"));
        eprintln!("[http] serving {model_name} on http://{addr}/v1/chat/completions");
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
                eprintln!("[http] ctrl-c: shutting down");
            })
            .await
            .expect("serve");
    });
}
