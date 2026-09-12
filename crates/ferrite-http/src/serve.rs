//! The serve entry point: driver thread + axum + graceful shutdown.
//!
//! Every engine eventually needs the same ten lines of wiring — spawn the
//! driver over the engine, build the router with the model's chat frame, bind,
//! serve until ctrl-c (or `POST /shutdown`). It lives here so a new model
//! adds a backend + a frame, not a copy of this file.
//!
//! #### Why `launch` exits the process
//!
//! Engines that hold CUDA contexts (and terabyte weight sets) must NOT run
//! their `Drop` at exit: the teardown walks every allocation and segfaults
//! (ferrite-serve's documented EXIT 139). The OS and CUDA reclaim everything
//! at process exit, so the launcher owns the exit and `launch` never returns.

use std::net::SocketAddr;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Duration;

use crate::api::{router_with, AppState};
use crate::driver::EngineDriver;
use crate::engine::ServeEngine;
use crate::tokenizer::{ChatFrame, ChatTokenizer};

/// Where to bind, what to call the model, how to pace the driver.
pub struct ServeOptions {
    pub addr: SocketAddr,
    pub model_name: String,
    /// Busy-loop pacing for the driver: ZERO for GPU engines (the step IS the
    /// pacing — a decode step is tens of ms), a positive value for mock/CPU
    /// engines whose tick is microseconds.
    pub tick_interval: Duration,
    /// Optional teardown hook, run ONCE by [`launch`] immediately before
    /// `process::exit` — for engine-side state the OS exit would otherwise
    /// skip. GLM passes nsys/CUPTI's `cudaProfilerStop` here: nsys
    /// `--capture-range=cudaProfilerApi` waits for it and never flushes the
    /// report without it (the launcher owns the exit, so the hook must ride
    /// the options — a caller cannot do it "after launch").
    pub on_shutdown: Option<Box<dyn FnOnce() + Send>>,
}

impl ServeOptions {
    pub fn new(addr: SocketAddr, model_name: impl Into<String>) -> Self {
        ServeOptions {
            addr,
            model_name: model_name.into(),
            tick_interval: Duration::ZERO,
            on_shutdown: None,
        }
    }

    pub fn paced(mut self, tick_interval: Duration) -> Self {
        self.tick_interval = tick_interval;
        self
    }

    /// Register a teardown hook (see the field doc). Replaces any previous one.
    pub fn on_shutdown(mut self, f: impl FnOnce() + Send + 'static) -> Self {
        self.on_shutdown = Some(Box::new(f));
        self
    }
}

/// Serve `engine`'s OpenAI-compatible API until ctrl-c / `POST /shutdown`,
/// then exit the process (see the module doc). Never returns.
pub fn launch<E: ServeEngine + 'static>(
    engine: E,
    tok: ChatTokenizer,
    frame: Arc<dyn ChatFrame>,
    mut opts: ServeOptions,
) -> ! {
    // Take the teardown hook before `opts` moves into the async block below.
    let on_shutdown = opts.on_shutdown.take();
    let handle = EngineDriver::spawn_with(engine, opts.tick_interval);
    let state = AppState {
        handle,
        tok: Arc::new(tok),
        model_name: opts.model_name.clone(),
        req_counter: Arc::new(AtomicU64::new(1)),
    };
    let app = router_with(state, frame);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(opts.addr)
            .await
            .unwrap_or_else(|e| panic!("bind {}: {e}", opts.addr));
        println!(
            "[serve] serving {} on http://{}/v1/chat/completions (SSE; /v1/models, /health, /v1/stats, /shutdown)",
            opts.model_name, opts.addr
        );
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
                eprintln!("[serve] ctrl-c: shutting down");
            })
            .await
            .expect("serve");
    });
    // No teardown: dropping the engine exits through the CUDA/weight destructors
    // (the documented segfault). The driver thread and the engine leak with the
    // process by design. An engine-supplied hook runs first (e.g. CUPTI's
    // cudaProfilerStop, which nsys waits on to flush its report).
    if let Some(f) = on_shutdown {
        f();
    }
    std::process::exit(0);
}
