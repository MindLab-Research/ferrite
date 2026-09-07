//! Tokenizer + chat-template boundary.
//!
//! The HTTP layer never touches token ids directly — this module owns
//! the text ⇄ token seam so the rest of the crate speaks `Vec<u32>`:
//!
//! - **Chat format** (GLM): messages render into the same
//!   `<|prompt|>…</s>\n<|answer|>\n` frame ferrite-serve uses — one
//!   template, one place (HTTP parity with the CLI path matters: both
//!   radix-match the same token streams, or the prefix cache fragments).
//! - **Stop tokens**: `<|end|>` plus turn-boundary specials — an OpenAI
//!   `finish_reason: "stop"` fires on any of them.
//! - **Streaming decode**: one tokenizer instance, decode-per-delta.
//!   The `tokenizers` crate is not thread-safe for mutation, but
//!   `decode` takes `&self` — one `Arc` shared across handlers.

use ferrite_types::{FerriteError, Result};
use tokenizers::Tokenizer;

/// One chat message (OpenAI wire shape, decoded straight from JSON).
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Render messages through the GLM chat format.
///
/// System + user turns stack inside `<|prompt|>…</s>`; the assistant
/// turn opens `<|answer|>` and generation continues from there. This
/// mirrors `ferrite-serve`'s `wrap_prompt` exactly (single-turn there,
/// multi-turn here — same frame, stacked turns).
pub fn render_chat_template(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    let mut last_role = "user";
    for msg in messages {
        let role = if msg.role == "assistant" { "assistant" } else { "user" };
        if role == "assistant" {
            // an in-context assistant turn: previous turn's answer text
            prompt.push_str(&msg.content);
            prompt.push_str("\n<s>\n\n");
            prompt.push_str("<|prompt|>\n");
        } else {
            if last_role == "assistant" || !prompt.is_empty() {
                // consecutive non-assistant turns stack with the frame
                prompt.push_str(&msg.content);
                prompt.push('\n');
            } else {
                prompt.push_str("<|prompt|>\n");
                prompt.push_str(&msg.content);
                prompt.push('\n');
            }
        }
        last_role = role;
    }
    // close: only if the last turn wasn't an assistant continuation
    if !prompt.starts_with("<|prompt|>") {
        prompt.insert_str(0, "<|prompt|>\n");
    }
    if !prompt.trim_end().ends_with("\n") {
        prompt.push('\n');
    }
    prompt.push_str("</s>\n\n");
    format!("{prompt}")
}

/// GLM-5.3-Flash tokenizer wrapper: encode prompts, decode deltas,
/// resolve stop ids. Shared by every handler (read-only after load).
///
/// Two vocabularies:
/// - `Real` — the HF `tokenizer.json` from the model dir (production).
/// - `Byte` — a lossless byte-level identity codec (mock mode): every
///   byte is one token id, decoding is exact — the deterministic mock
///   engine streams printable ASCII, so SSE deltas render real text
///   with zero model files (the full HTTP → scheduler → SSE path runs
///   on a laptop; the GPU backend swaps in behind the same API).
pub enum ChatTokenizer {
    Real(Tokenizer),
    Byte,
}

impl ChatTokenizer {
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let tok = Tokenizer::from_file(path)
            .map_err(|e| FerriteError::InvalidArg(format!("load tokenizer: {e}")))?;
        let mut stop: Vec<u32> = vec![154_820]; // <|end|> (ferrite-serve parity)
        for special in ["\u{FFFD}\u{FFFD}", "\u{FFFD}", "\u{FFFD}\u{FFFD}", "\u{FFFD}\u{FFFD}"] {
            if let Some(id) = tok.token_to_id(special) {
                if !stop.contains(&id) {
                    stop.push(id);
                }
            }
        }
        Ok(ChatTokenizer::Real(tok))
    }

    /// The byte-level identity codec (mock mode — see the type doc).
    pub fn stub() -> Self {
        ChatTokenizer::Byte
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            ChatTokenizer::Real(tok) => {
                let enc = tok
                    .encode(text, false)
                    .map_err(|e| FerriteError::InvalidArg(format!("encode: {e}")))?;
                Ok(enc.get_ids().to_vec())
            }
            ChatTokenizer::Byte => Ok(text.bytes().map(|b| b as u32).collect()),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        match self {
            ChatTokenizer::Real(tok) => tok
                .decode(ids, false)
                .map_err(|e| FerriteError::InvalidArg(format!("decode: {e}"))),
            ChatTokenizer::Byte => {
                let bytes: Vec<u8> = ids.iter().map(|&i| i as u8).collect();
                Ok(String::from_utf8_lossy(&bytes).into_owned())
            }
        }
    }

    /// Stop ids (chat-template specials) — the SSE layer strips them
    /// from streamed content.
    pub fn stop_ids(&self) -> &[u32] {
        match self {
            ChatTokenizer::Real(_) => STOP_IDS_REAL,
            ChatTokenizer::Byte => STOP_IDS_BYTE,
        }
    }

    /// Check whether an accepted token ends the turn.
    pub fn is_stop(&self, id: u32) -> bool {
        self.stop_ids().contains(&id)
    }
}

/// `<|end|>` + turn-boundary specials (ferrite-serve parity).
static STOP_IDS_REAL: &[u32] = &[154_820, 154_821, 154_822, 154_823];

/// Byte codec has no specials; the mock engine emits `STOP_ID` (154820)
/// outside byte range at stream end — outside id space cannot collide
/// with content bytes.
static STOP_IDS_BYTE: &[u32] = &[crate::host_engine::STOP_ID];
