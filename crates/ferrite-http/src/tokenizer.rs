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

/// Render messages through the GLM chat format — BYTE-PARITY with
/// ferrite-serve's CLI `wrap_prompt` (`<|user|>\n{text}</s>\n<|assistant|>\n`,
/// the validated frame): the same tokens radix-match the same prefix, and
/// the model sees the identical prompt the CLI path was tuned on.
///
/// Multi-turn: user/system turns open `<|user|>`, assistant turns open
/// `<|assistant|>`; every turn closes `</s>`; the generation point is the
/// trailing `<|assistant|>\n`. System turns fold into the leading user
/// frame (this model has no separate system slot).
pub fn render_chat_template(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    let mut last_was_user = false;
    for msg in messages {
        let is_asst = msg.role == "assistant";
        if is_asst {
            prompt.push_str("<|assistant|>\n");
            prompt.push_str(&msg.content);
            prompt.push_str("</s>\n");
            last_was_user = false;
        } else {
            prompt.push_str("<|user|>\n");
            prompt.push_str(&msg.content);
            prompt.push_str("</s>\n");
            let _ = last_was_user;
            last_was_user = true;
        }
    }
    // generation opens at the final assistant frame
    prompt.push_str("<|assistant|>\n");
    prompt
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
    Real(Tokenizer, Vec<u32>),
    Byte,
}

impl ChatTokenizer {
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let tok = Tokenizer::from_file(path)
            .map_err(|e| FerriteError::InvalidArg(format!("load tokenizer: {e}")))?;
        // Stop set — ferrite-serve CLI parity (byte-for-byte the same stop
        // list): primary <|end|> 154820 PLUS the turn-boundary specials
        // resolved from the tokenizer (misses skipped; dedup). The peer's
        // original literals here were U+FFFD corruption (dead resolution —
        // stop_ids fell back to a guessed static); resolved-at-load is truth.
        let mut stop: Vec<u32> = vec![154_820]; // <|end|> (eos)
        for special in ["<|user|>", "<|endoftext|>", "<|observation|>"] {
            if let Some(id) = tok.token_to_id(special) {
                if !stop.contains(&id) {
                    stop.push(id);
                }
            }
        }
        eprintln!("[http] tokenizer stops: {stop:?}");
        Ok(ChatTokenizer::Real(tok, stop))
    }

    /// The byte-level identity codec (mock mode — see the type doc).
    pub fn stub() -> Self {
        ChatTokenizer::Byte
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            ChatTokenizer::Real(tok, _) => {
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
            ChatTokenizer::Real(tok, _) => tok
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
            ChatTokenizer::Real(_, stop) => stop,
            ChatTokenizer::Byte => STOP_IDS_BYTE,
        }
    }

    /// Check whether an accepted token ends the turn.
    pub fn is_stop(&self, id: u32) -> bool {
        self.stop_ids().contains(&id)
    }
}

/// Byte codec has no specials; the mock engine emits `STOP_ID` (154820)
/// outside byte range at stream end — outside id space cannot collide
/// with content bytes.
static STOP_IDS_BYTE: &[u32] = &[crate::host_engine::STOP_ID];
