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

/// The stop set of a checkpoint: literal ids (its config's EOS — some
/// checkpoints ship no name for it) PLUS special-token names resolved from the
/// tokenizer at load (misses skipped, ids deduped).
///
/// The stop set is a property of the CHECKPOINT, not of the HTTP layer, so it
/// is data: `GLM_STOPS` is the shared stack's default (ferrite-serve parity);
/// another model supplies its own (`from_file_with`). Resolved-at-load is
/// truth: a guessed static silently mislabels finish reasons.
#[derive(Debug, Clone, Copy)]
pub struct StopSpec<'a> {
    /// Literal stop ids (the checkpoint's EOS from its own config).
    pub ids: &'a [u32],
    /// Special-token names resolved from the tokenizer at load.
    pub specials: &'a [&'a str],
}

impl<'a> StopSpec<'a> {
    pub fn new(ids: &'a [u32], specials: &'a [&'a str]) -> Self {
        StopSpec { ids, specials }
    }
}

/// GLM-5.3-Flash stops — ferrite-serve CLI parity, byte-for-byte the same list
/// (the shared stack's default; the CLI and the HTTP path must retire on the
/// same ids or the two diverge on the wire).
pub const GLM_STOPS: StopSpec<'static> = StopSpec {
    ids: &[154_820], // <|end|> (eos)
    specials: &["<|user|>", "<|endoftext|>", "<|observation|>"],
};

impl ChatTokenizer {
    /// GLM preset (ferrite-serve parity — the shared stack's default frame).
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        Self::from_file_with(path, GLM_STOPS)
    }

    /// Any checkpoint: load ITS tokenizer and resolve ITS stop set.
    pub fn from_file_with(path: &std::path::Path, spec: StopSpec<'_>) -> Result<Self> {
        let tok = Tokenizer::from_file(path)
            .map_err(|e| FerriteError::InvalidArg(format!("load tokenizer: {e}")))?;
        let mut stop: Vec<u32> = spec.ids.to_vec();
        for special in spec.specials {
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

    /// Streaming batch decode with UTF-8 tail-holdback: byte-BPE splits
    /// multi-byte characters across token boundaries — decoding a batch
    /// whose TAIL is a partial char yields U+FFFD. This holds back the
    /// trailing tokens while the decode ends in a replacement char (the
    /// held tokens stay in the caller's buffer and concatenate with the
    /// next batch; the terminal flush passes `decode` on the full buffer
    /// which accepts the tail). Returns (safe text, tokens held back).
    pub fn decode_batch(&self, buf: &[u32]) -> (String, usize) {
        if buf.is_empty() {
            return (String::new(), 0);
        }
        let mut n = buf.len();
        let mut s = self.decode(&buf[..n]).unwrap_or_default();
        while n > 1 && s.ends_with('\u{FFFD}') {
            n -= 1;
            s = self.decode(&buf[..n]).unwrap_or_default();
        }
        (s, buf.len() - n)
    }

    /// Check whether an accepted token ends the turn.
    pub fn is_stop(&self, id: u32) -> bool {
        self.stop_ids().contains(&id)
    }

    /// Single-token lookup for a chat frame's markers: a special resolves to
    /// ITS id in this checkpoint's vocab (never a hardcoded number). The byte
    /// codec has no specials.
    pub fn special_id(&self, name: &str) -> Option<u32> {
        match self {
            ChatTokenizer::Real(tok, _) => tok.token_to_id(name),
            ChatTokenizer::Byte => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Chat frame: the model-specific prompt layout
// ---------------------------------------------------------------------------

/// How an OpenAI message list becomes the checkpoint's prompt token ids.
///
/// Text ⇄ token is shared (the `ChatTokenizer`), but the FRAME — which markers
/// open a turn, what closes it, whether the assistant starts inside a thinking
/// block — is a property of the checkpoint. So the HTTP layer never hardcodes
/// markers: whoever owns the model (ferrite-serve for GLM, the DSV41 runner)
/// supplies a frame at wiring time (`api::router_with`). Everything downstream
/// — request/event protocol, SSE framing, usage, cancel-on-drop, stats — is
/// shared, which is what makes "add a model" a wiring change instead of a fork
/// of the HTTP stack.
pub trait ChatFrame: Send + Sync {
    fn encode_chat(&self, messages: &[ChatMessage], tok: &ChatTokenizer) -> Result<Vec<u32>>;
}

/// The GLM frame — ferrite-serve CLI parity, the shared stack's default.
pub struct GlmFrame;

impl ChatFrame for GlmFrame {
    fn encode_chat(&self, messages: &[ChatMessage], tok: &ChatTokenizer) -> Result<Vec<u32>> {
        tok.encode(&render_chat_template(messages))
    }
}

/// One element of an id-level frame.
#[derive(Debug, Clone, Copy)]
pub enum Seg<'a> {
    /// A tokenizer special emitted as a SINGLE token (resolved by name: the id
    /// comes from the checkpoint's own vocab). A name the tokenizer does not
    /// know falls back to literal text encode — loudly, because a mis-resolved
    /// marker means a mistemplated prompt.
    Special(&'a str),
    /// Literal text (encoded).
    Text(&'a str),
    /// The turn's message content, encoded at this position in the frame.
    Content,
}

/// Resolve + encode one segment list (`content` fills `Seg::Content`).
///
/// The shared half of an id-level frame (the DSV41/DeepSeek-style families):
/// the LAYOUT stays with the model, the marker resolution + tokenization is
/// here — one place for every model.
pub fn encode_segments(tok: &ChatTokenizer, segs: &[Seg<'_>], content: &str) -> Result<Vec<u32>> {
    let mut out: Vec<u32> = Vec::new();
    for seg in segs {
        match seg {
            Seg::Special(name) => match tok.special_id(name) {
                Some(id) => out.push(id),
                None => {
                    eprintln!(
                        "[http] frame marker {name:?} is not a special of this tokenizer — literal encode"
                    );
                    out.extend_from_slice(&tok.encode(name)?);
                }
            },
            Seg::Text(t) => out.extend_from_slice(&tok.encode(t)?),
            Seg::Content => out.extend_from_slice(&tok.encode(content)?),
        }
    }
    Ok(out)
}

/// Byte codec has no specials; the mock engine emits `STOP_ID` (154820)
/// outside byte range at stream end — outside id space cannot collide
/// with content bytes.
static STOP_IDS_BYTE: &[u32] = &[crate::host_engine::STOP_ID];
