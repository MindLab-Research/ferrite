//! ferrite-http: the OpenAI-compatible HTTP/SSE serving layer.
//!
//! # Architecture
//!
//! ```text
//!   axum handlers (async)               engine thread (std, sole writer)
//!   ─────────────────────  DriverCmd  ─────────────────────────────▶
//!   POST /v1/chat/completions           loop: drain → submit/cancel
//!     ├─ stream=true  → SSE             │ engine.tick (plan → exec → ingest)
//!     └─ stream=false → JSON            │ diff outputs → Token events
//!   ◀──────── ReqEvent (per-req mpsc) ◀│ retire finished → Finished
//!   GET /v1/models /health /v1/stats  ◀┘ stats (RwLock, once per tick)
//! ```
//!
//! - **`api`** — OpenAI wire protocol: `/v1/chat/completions`
//!   (SSE `chat.completion.chunk` frames + `[DONE]`, or one JSON body),
//!   `/v1/models`, `/health`, `/v1/stats` (radix tree, hicache tier
//!   census, page budget, ticks — the prefix-cache observability an
//!   operator actually wants under load). Client disconnects cancel:
//!   the SSE body's Drop guard sends `DriverCmd::Cancel` → the scheduler
//!   retires the row and unpins its radix path (no leaked decode rows
//!   under connection churn).
//! - **`sse`** — the wire objects (frames, usage with `cached_tokens` —
//!   the radix prefix-hit number surfaced per request).
//! - **`driver`** — the engine thread: ONE writer for the scheduler
//!   (`BatchScheduler` is single-writer by design — the two-phase tick
//!   keeps the hicache race-free), tokio mpsc both directions, stats
//!   snapshot per tick.
//! - **`engine`** — the request/event contract (`RequestSpec` in, token
//!   deltas + finish + usage out; the async↔engine seam is plain data).
//! - **`host_engine`** — the deterministic backend over the REAL
//!   scheduler (admission, radix resume, page budgets, MTP accept
//!   folding, retirement — everything but the transformer compute):
//!   the full HTTP→SSE stack runs on a laptop (mock mode), and the CUDA
//!   backend later slots in behind the same `ExecBackend` seam.
//! - **`tokenizer`** — GLM chat frame (ferrite-serve parity: shared
//!   system prompts radix-match across CLI and HTTP clients) + HF
//!   tokenizer or the byte codec (mock mode).
//!
//! # Serving semantics
//!
//! - `max_tokens` / `max_completion_tokens` → the scheduler's per-seq
//!   cap (finish_reason: "length"); stop ids → "stop".
//! - `temperature` is accepted and ignored (greedy MTP verify — the
//!   numeric-domain iron law; sampling is a future knob behind the
//!   same engine seam).
//! - `include_usage: true` appends usage to the final SSE frame.
//! - Radix hits ride every response (`usage.cached_tokens` + the
//!   SSE `admitted prefix_hit=N` comment — `curl -N` observability).
//!
//! # The `main` binary
//!
//! `ferrite-http [--host 0.0.0.0] [--port 8080] [--model-dir DIR]`
//! `[--model NAME]` — mock mode (byte tokenizer + deterministic
//! generation) with no `--model-dir`; a real tokenizer with one
//! (generation stays deterministic until the CUDA backend lands).

pub mod api;
pub mod driver;
pub mod engine;
pub mod host_engine;
pub mod sse;
pub mod tokenizer;

/// Crate version (workspace-aligned).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
