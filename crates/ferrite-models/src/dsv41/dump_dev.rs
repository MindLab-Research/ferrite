//! Golden-comparison dump for the DSpark path (`DSV41_DSPARK_DUMP=1`).
//!
//! The draft's accept rate is a single number at the end of a run, which makes
//! a wrong latent and a wrong draft identical from the outside. This module is
//! the other half of the official reference's per-step capture: the reference
//! (the V4.1 PyTorch golden) records, for every decoded step, the three target
//! layers' hidden collapsed over `hc` plus its token stream; the records here
//! are the SAME quantities, taken at the SAME point, so a step can be diffed
//! element by element instead of guessed at.
//!
//! What one record holds, all from the step the chain just ran:
//!
//! * `pos` / `token` / `next` — the position, the input token and the argmax,
//!   i.e. the "which step" key.
//! * `tap` — `DevChain`'s `dspark_tap`, `[DSPARK_TAP_SLOTS][dim]` f32, read
//!   BEFORE the draft consumed it. Slot order is the config's
//!   `dspark_target_layer_ids` order (37/38/39), and each slot is the layer's
//!   COMPLETED output collapsed over `hc` — `completed.mean(dim=1)` in the
//!   reference (`deepseek_v4.py:3132-3141`). This is the golden's `main_x`
//!   source: the same `[n_target, dim]` block `import_tap` copies into the
//!   draft.
//! * `drafts` / `verify_out` / `k_acc` — the draft's proposals, the verify
//!   block's per-row argmax, and the accepted prefix length.
//!
//! # Format and cost
//!
//! One JSON object per line (JSON Lines), APPEND mode, rank 0 only (every TP
//! rank is a separate process, so an unguarded dump would interleave eight
//! identical records). Path: `DSV41_DSPARK_DUMP_PATH`, default
//! `/tmp/ferrite_tap.jsonl`.
//!
//! Floats are written with Rust's shortest-roundtrip `Display`, NOT through
//! `serde_json`: a non-finite tap value has to stay visible as `NaN`/`inf`
//! (serde_json would silently turn it into `null`, losing exactly the signal a
//! numeric diff is looking for), and the 15 360-element array is written
//! straight into one `String` instead of a `Value` tree.
//!
//! Cost is paid only while the gate is on: one `3 * dim * 4 B` D2H per step
//! (60 KB at `dim = 5120`) plus the line's formatting. Acceptable for a debug
//! tool, and the reason the whole thing is behind a `OnceLock`-cached env
//! check rather than a per-call `getenv`.

use std::ffi::c_void;
use std::fmt::Write as _;
use std::io::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use ferrite_types::{FerriteError, Result};

use crate::dsv41::device::Device;

/// `DSV41_DSPARK_DUMP=1` arms the dump; `0` or unset leaves it off. Cached in a
/// `OnceLock` for the same reason `Dsv41Config::dspark_armed` is: this is read
/// on every speculative step, and a bare `getenv` there is exactly the slip
/// this project has been bitten by before.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("DSV41_DSPARK_DUMP")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

/// Where the records go: `DSV41_DSPARK_DUMP_PATH`, else `/tmp/ferrite_tap.jsonl`.
pub fn path() -> &'static str {
    static P: OnceLock<String> = OnceLock::new();
    P.get_or_init(|| {
        std::env::var("DSV41_DSPARK_DUMP_PATH")
            .unwrap_or_else(|_| "/tmp/ferrite_tap.jsonl".to_string())
    })
}

/// Per-process record counter. The file is append-only and survives across
/// runs, so `pos` alone cannot tell two runs (or two requests) apart; `seq` and
/// `pid` can, and a comparison script picks the run it wants with them.
static SEQ: AtomicU64 = AtomicU64::new(0);

/// One step's record. Borrows the caller's slices; the tap is a DEVICE pointer
/// and is D2H'd inside [`write_step`].
pub struct Step<'a> {
    /// `"shadow"` or `"spec"` — which orchestration produced the record.
    pub mode: &'a str,
    /// The backbone token's position (the step's `pos`).
    pub pos: usize,
    /// The step's input token `t0`.
    pub token: u32,
    /// The step's real argmax.
    pub next: u32,
    /// The accepted draft-prefix length (0..=DSPARK_DRAFTS) — the same quantity
    /// in both modes, so the two are diffable against each other.
    pub k_acc: usize,
    pub drafts: &'a [u32],
    pub verify_out: &'a [u32],
    /// The chain's tap, `[tap_slots][dim]` f32 (`DevChain`'s `dspark_tap`).
    pub tap: *const f32,
    pub tap_slots: usize,
    pub dim: usize,
}

/// D2H the tap and append one JSON line for this step.
///
/// The caller is expected to LOG a failure rather than propagate it: a full
/// disk must not kill a decode step that already committed its block.
pub fn write_step(dev: &Device, s: &Step<'_>) -> Result<()> {
    let n = s.tap_slots * s.dim;
    let mut tap = vec![0f32; n];
    // A `DevBuf` VIEW over the chain's tap: the chain owns the buffer, this is
    // only the download handle (`stats()` builds its views the same way).
    let view = Device::view(s.tap as *mut c_void, n * 4);
    dev.download_f32(&view, &mut tap)?;

    let ts_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let mut line = String::with_capacity(n * 10 + 256);
    // String writes are infallible; `expect` documents that instead of
    // threading a `Result` through every formatting call.
    write!(
        line,
        "{{\"seq\":{},\"pid\":{},\"ts_ms\":{},\"mode\":\"{}\",\"pos\":{},\"token\":{},\"next\":{},\"k_acc\":{},\"tap_slots\":{},\"dim\":{}",
        SEQ.fetch_add(1, Ordering::Relaxed),
        std::process::id(),
        ts_ms,
        s.mode,
        s.pos,
        s.token,
        s.next,
        s.k_acc,
        s.tap_slots,
        s.dim,
    )
    .expect("string write");

    line.push_str(",\"drafts\":[");
    for (i, t) in s.drafts.iter().enumerate() {
        if i > 0 {
            line.push(',');
        }
        let _ = write!(line, "{t}");
    }
    line.push_str("],\"verify_out\":[");
    for (i, t) in s.verify_out.iter().enumerate() {
        if i > 0 {
            line.push(',');
        }
        let _ = write!(line, "{t}");
    }
    line.push_str("],\"tap\":[");
    for (i, v) in tap.iter().enumerate() {
        if i > 0 {
            line.push(',');
        }
        let _ = write!(line, "{v}");
    }
    line.push_str("]}\n");

    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path())
        .map_err(|e| FerriteError::Config(format!("dspark dump: open {}: {e}", path())))?;
    f.write_all(line.as_bytes())
        .map_err(|e| FerriteError::Config(format!("dspark dump: write {}: {e}", path())))?;
    Ok(())
}
