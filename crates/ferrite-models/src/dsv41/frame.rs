//! DeepSeek-V4.1-Flash chat frame — this checkpoint's prompt layout.
//!
//! The reference's `encode_messages(messages, thinking_mode="chat")`:
//! `<|begin_of_sentence|><|User|>{prompt}<|Assistant|></think>`
//! (ids [0, 128803] + content + [128804, 128822]).
//!
//! Markers are resolved from the checkpoint's own tokenizer by name (never
//! hardcoded ids — `encode_segments` resolves them and warns when a marker is
//! not a special). A completed assistant turn closes with the checkpoint's
//! end-of-sentence marker, which reproduces the verified single-turn frame
//! exactly.
//!
//! The frame lives WITH THE MODEL (one engine, models as data): both the
//! `dsv41-run` runner and the unified `ferrite-serve --model dsv41` binary
//! share this one definition, while the `ChatFrame` trait itself stays owned
//! by `ferrite-http`.

use ferrite_http::tokenizer::{encode_segments, ChatFrame, ChatMessage, ChatTokenizer, Seg};
use ferrite_types::Result;

/// This checkpoint's chat frame.
pub struct Dsv41Frame;

impl ChatFrame for Dsv41Frame {
    fn encode_chat(&self, messages: &[ChatMessage], tok: &ChatTokenizer) -> Result<Vec<u32>> {
        let mut ids: Vec<u32> = Vec::new();
        for (i, m) in messages.iter().enumerate() {
            if i == 0 {
                // Pinned ids: this checkpoint's marker names are NOT tokenizer
                // specials, so a by-name lookup would literal-encode them and
                // change the prompt (measured earlier: the one-shot path only
                // answers correctly with the raw ids [0, 128803] + body +
                // [128804, 128822]).
                ids.extend(encode_segments(tok, &[Seg::Id(0)], "")?);
            }
            let asst = m.role == "assistant";
            let marker = if asst { 128804u32 } else { 128803u32 }; // <|Assistant|> / <|User|>
            let mut segs = vec![Seg::Id(marker), Seg::Content];
            if asst {
                segs.push(Seg::Id(1)); // <|end_of_sentence|> == eos
            }
            ids.extend(encode_segments(tok, &segs, &m.content)?);
        }
        // generation opens at the assistant position in chat mode (thinking
        // off): without the trailing </think> the model drifts into a thinking
        // block instead of answering (the raw-text runs' failure mode).
        ids.extend(encode_segments(
            tok,
            // the generation opener: <|Assistant|> then </think>, both PINNED as
            // ids for the same reason as the markers above
            &[Seg::Id(128804), Seg::Id(128822)],
            "",
        )?);
        Ok(ids)
    }
}
