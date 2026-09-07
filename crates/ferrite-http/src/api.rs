//! HTTP surface — OpenAI-compatible chat completions over SSE.
//!
//! Routes:
//! - `POST /v1/chat/completions` — `stream: true` renders the driver's
//!   event stream as `chat.completion.chunk` SSE frames + `[DONE]`;
//!   `stream: false` aggregates one `chat.completion` JSON. Client
//!   disconnects cancel the request (Drop-guard on the event stream
//!   → driver `Cancel` → scheduler retires the row, radix unpins —
//!   no leaked decode rows under connection churn).
//! - `GET /v1/models` — the served model (single-model engine).
//! - `GET /health` — liveness (process + driver stats snapshot).
//! - `GET /v1/stats` — the driver telemetry: radix tree, hicache tier
//!   census, page budget, ticks (radix-hit observability).
//!
//! Design notes:
//! - The tokenizer is `Arc`-shared read-only (`tokenizers`' decode takes
//!   `&self`; encode paths are content-disjoint across requests).
//! - Per-request state is one event receiver + the ReqId; everything
//!   heavy lives on the engine thread (handlers stay allocation-light:
//!   render template → encode → submit → forward frames).
//! - `include_usage` (OpenAI opt-in) appends usage on the final frame;
//!   our `cached_tokens` rides the standard `usage` extension slot.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use crate::driver::EngineHandle;
use crate::engine::{FinishReason, ReqEvent, Usage};
use crate::sse::{ChatChunk, ChatCompletionResponse, UsageDto};
use crate::tokenizer::{render_chat_template, ChatMessage, ChatTokenizer};

/// Shared handler state (cheap clones: Arc + channel sender).
#[derive(Clone)]
pub struct AppState {
    pub handle: EngineHandle,
    pub tok: Arc<ChatTokenizer>,
    pub model_name: String,
    /// Wall-clock id counter for chatcmpl- ids.
    pub req_counter: Arc<std::sync::atomic::AtomicU64>,
}

impl AppState {
    fn new_id(&self) -> String {
        let n = self.req_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        format!("chatcmpl-{:016x}", n)
    }
}

/// OpenAI-compatible request (the subset we honor; unknown fields are
/// ignored — clients send far more schema than this).
#[derive(Debug, Clone, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<WireMessage>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<usize>,
    /// OpenAI name (naming varies by client: max_tokens / max_completion_tokens).
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    /// Accepted for compatibility; the engine is greedy (MTP accept is
    /// argmax — the iron-law numeric path; sampling is a future knob).
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub include_usage: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WireMessage {
    pub role: String,
    pub content: String,
}

fn default_max_tokens() -> Option<usize> {
    Some(256)
}

/// `POST /v1/chat/completions`.
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> axum::response::Response {
    if req.messages.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "messages must be non-empty");
    }
    // Render + encode (chat template → token ids — the radix-visible form:
    // shared system prompts across requests are exactly what the prefix
    // cache is for).
    let messages: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|m| ChatMessage { role: m.role.clone(), content: m.content.clone() })
        .collect();
    let rendered = render_chat_template(&messages);
    let prompt_ids = match state.tok.encode(&rendered) {
        Ok(ids) => ids,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &e.to_string()),
    };
    let max_new = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(256)
        .max(1);
    let spec = crate::engine::RequestSpec {
        prompt_ids,
        max_new_tokens: max_new,
        stop_ids: state.tok.stop_ids().to_vec(),
    };
    let (req_id, events) = state.handle.submit(spec).await;
    let chat_id = state.new_id();

    match req.stream.unwrap_or(false) {
        true => {
            // SSE streaming: one frame per engine event (the driver's MTP
            // commit window — 1–3 tokens per frame, the OpenAI chunk
            // granularity), [DONE] appended by the stream tail. The event
            // closure owns a tokenizer clone (Arc) for delta decoding;
            // stop ids never reach content (stripped before decode).
            let tok = state.tok.clone();
            let model = state.model_name.clone();
            let include_usage = req.include_usage.unwrap_or(false);
            let handle = state.handle.clone();
            let chat_id2 = chat_id.clone();
            let body = UnboundedReceiverStream::new(events);
            let stream = body.map(move |ev: ReqEvent| -> Result<Event, std::convert::Infallible> {
                Ok(match ev {
                    ReqEvent::Admitted { prefix_hit, prompt_tokens } => Event::default()
                        .comment(format!(
                            "ferrite: admitted prefix_hit={prefix_hit} prompt_tokens={prompt_tokens}"
                        )),
                    ReqEvent::Tokens { ids } => {
                        let clean: Vec<u32> =
                            ids.into_iter().filter(|t| !tok.is_stop(*t)).collect();
                        let text = if clean.is_empty() {
                            String::new()
                        } else {
                            tok.decode(&clean).unwrap_or_default()
                        };
                        if text.is_empty() {
                            // stop-id-only window (turn end rides Finished)
                            Event::default().comment("ferrite: stop token")
                        } else {
                            let chunk = ChatChunk::content(&chat_id2, &model, &text);
                            Event::default().data(serde_json::to_string(&chunk).expect("sse"))
                        }
                    }
                    ReqEvent::Finished { reason, usage } => {
                        let chunk = ChatChunk::finish(
                            &chat_id2,
                            &model,
                            reason.as_str(),
                            include_usage.then(|| UsageDto::of(usage)),
                        );
                        Event::default().data(serde_json::to_string(&chunk).expect("sse"))
                    }
                })
            });
            // [DONE] sentinel after the Finished frame (OpenAI close).
            let stream = stream.chain(tokio_stream::iter(vec![
                Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
            ]));
            // Cancel-on-drop: the body ends — client disconnect or stream
            // complete — the driver hears Cancel (retire + unpins; cancel
            // on a finished seq is the driver's no-op path).
            let guarded = CancelGuard { handle, req: req_id, inner: stream };
            Sse::new(guarded).keep_alive(KeepAlive::default()).into_response()
        }
        false => {
            // Aggregate the event stream into one completion JSON.
            // (the stream must outlive the loop — `while let` re-evaluates
            // its scrutinee per iteration, so bind it first)
            let mut events = UnboundedReceiverStream::new(events);
            let mut usage = Usage::default();
            let mut text = String::new();
            let mut reason = FinishReason::Stop;
            while let Some(ev) = events.next().await {
                match ev {
                    ReqEvent::Admitted { .. } => {}
                    ReqEvent::Tokens { ids } => {
                        // stream-content decode in batches; drop stop ids
                        let clean: Vec<u32> =
                            ids.into_iter().filter(|t| !state.tok.is_stop(*t)).collect();
                        if !clean.is_empty() {
                            if let Ok(s) = state.tok.decode(&clean) {
                                text.push_str(&s);
                            }
                        }
                    }
                    ReqEvent::Finished { reason: r, usage: u } => {
                        reason = r;
                        usage = u;
                    }
                }
            }
            Json(ChatCompletionResponse::of(
                &chat_id,
                &state.model_name,
                text,
                reason.as_str(),
                usage,
            ))
            .into_response()
        }
    }
}

/// `POST /v1/chat/completions` (SSE rendering arm) — see the closure in
/// the stream branch: events render inline (the closure owns the Arc
/// tokenizer + chat id; `render_event` would need a 5-arg context —
/// the closure is the API).

/// GET /v1/models — the single served model (OpenAI list shape).
async fn models(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "created": 0,
            "owned_by": "ferrite",
        }]
    }))
}

/// GET /health — liveness + a stats snapshot.
async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let s = state.handle.stats();
    Json(serde_json::json!({
        "status": "ok",
        "engine": { "ticks": s.ticks, "tokens_committed": s.tokens_committed },
    }))
}

/// GET /v1/stats — the full telemetry (radix + hicache census).
async fn stats(State(state): State<AppState>) -> impl IntoResponse {
    let s = state.handle.stats();
    Json(serde_json::json!({
        "engine": {
            "ticks": s.ticks,
            "tokens_committed": s.tokens_committed,
            "live_rows": s.live_rows,
            "queued": s.queued,
        },
        "radix": {
            "tree_nodes": s.tree_nodes,
            "tree_blocks": s.tree_blocks,
            "evictable_tokens": s.evictable_tokens,
            "protected_tokens": s.protected_tokens,
        },
        "hicache": {
            "tiers": { "device": s.tier_census.0, "host": s.tier_census.1, "disk": s.tier_census.2 },
            "pages_in_use": s.pages_in_use,
            "pages_free": s.pages_free,
        },
    }))
}

fn error_response(status: StatusCode, msg: &str) -> axum::response::Response {
    (
        status,
        Json(serde_json::json!({
            "error": { "message": msg, "type": "invalid_request_error" }
        })),
    )
        .into_response()
}

/// Build the router (main.rs binds the listener).
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .route("/v1/stats", get(stats))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Cancel-on-drop guard: wraps the SSE event stream; when the HTTP body
// drops (client disconnect OR completed stream), the driver hears Cancel.
// ---------------------------------------------------------------------------

struct CancelGuard<S> {
    handle: EngineHandle,
    req: crate::engine::ReqId,
    inner: S,
}

impl<S> Drop for CancelGuard<S> {
    fn drop(&mut self) {
        self.handle.cancel(self.req);
    }
}

/// Stream passthrough with cancel-on-drop: axum drops the SSE body when
/// the client disconnects (or the stream completes) — the guard tells
/// the driver, which retires the sequence and unpins its radix path.
impl<S: tokio_stream::Stream + Unpin> tokio_stream::Stream for CancelGuard<S> {
    type Item = S::Item;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}
