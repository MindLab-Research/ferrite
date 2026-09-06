//! SSE framing — OpenAI `chat.completion.chunk` wire format.
//!
//! One frame per engine event batch: the MTP commit window (1–3 tokens)
//! becomes one content delta — the same granularity vLLM/SGLang stream
//! at (token-ish chunks; sub-word tokens coalesce into a decode round).
//!
//! The frame encoder is pure (String in → SSE frame out); `api.rs` maps
//! `ReqEvent`s onto it. Field-for-field OpenAI:
//!
//! ```text
//! data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,
//!        "model":…,"choices":[{"index":0,"delta":{"role":"assistant"},
//!        "finish_reason":null}]}
//! data: {"…","choices":[{"index":0,"delta":{"content":"tok"},"finish_reason":null}]}
//! data: {"…","choices":[{"index":0,"delta":{},"finish_reason":"stop",
//!        "usage":{…}}]}      ← usage on the last frame (include_usage)
//! data: [DONE]
//! ```

use serde::Serialize;

/// `choices[].delta` — the incremental payload.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// One streaming choice.
#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<&'static str>,
}

/// OpenAI `usage` (with the cached-tokens extension vLLM-style — our
/// radix prefix-hit accounting).
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct UsageDto {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<usize>,
}

impl UsageDto {
    pub fn of(u: crate::engine::Usage) -> Self {
        UsageDto {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.prompt_tokens + u.completion_tokens,
            cached_tokens: Some(u.cached_tokens),
        }
    }
}

/// The streaming frame.
#[derive(Debug, Clone, Serialize)]
pub struct ChatChunk {
    pub id: String,
    pub object: &'static str, // "chat.completion.chunk"
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageDto>,
}

impl ChatChunk {
    fn base() -> u64 {
        now_secs()
    }

    /// First frame: `delta: {"role":"assistant"}` (OpenAI streams the role
    /// before any content).
    pub fn role(id: &str, model: &str) -> Self {
        ChatChunk {
            id: id.to_string(),
            object: "chat.completion.chunk",
            created: Self::base(),
            model: model.to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta { role: Some("assistant"), content: None },
                finish_reason: None,
            }],
            usage: None,
        }
    }

    /// Content delta frame.
    pub fn content(id: &str, model: &str, text: &str) -> Self {
        ChatChunk {
            id: id.to_string(),
            object: "chat.completion.chunk",
            created: Self::base(),
            model: model.to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta { role: None, content: Some(text.to_string()) },
                finish_reason: None,
            }],
            usage: None,
        }
    }

    /// Terminal frame: empty delta + finish_reason (+usage when requested).
    /// `reason` is `FinishReason::as_str()` — a `&'static str`.
    pub fn finish(id: &str, model: &str, reason: &'static str, usage: Option<UsageDto>) -> Self {
        ChatChunk {
            id: id.to_string(),
            object: "chat.completion.chunk",
            created: Self::base(),
            model: model.to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta::default(),
                finish_reason: Some(reason),
            }],
            usage,
        }
    }
}

/// The terminal SSE sentinel (the stream tail after the finish frame —
/// rendered by `api.rs` as `Event::data("[DONE]")`).
pub const DONE_SENTINEL: &str = "[DONE]";

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Non-streaming response (the same wire objects, one JSON body).
// ---------------------------------------------------------------------------

/// `choices[].message` (non-streaming).
#[derive(Debug, Clone, Serialize)]
pub struct MessageDto {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseChoice {
    pub index: usize,
    pub message: MessageDto,
    pub finish_reason: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str, // "chat.completion"
    pub created: u64,
    pub model: String,
    pub choices: Vec<ResponseChoice>,
    pub usage: UsageDto,
}

impl ChatCompletionResponse {
    pub fn of(id: &str, model: &str, content: String, reason: &'static str, usage: crate::engine::Usage) -> Self {
        ChatCompletionResponse {
            id: id.to_string(),
            object: "chat.completion",
            created: ChatChunk::base(),
            model: model.to_string(),
            choices: vec![ResponseChoice {
                index: 0,
                message: MessageDto { role: "assistant", content },
                finish_reason: reason,
            }],
            usage: UsageDto::of(usage),
        }
    }
}
