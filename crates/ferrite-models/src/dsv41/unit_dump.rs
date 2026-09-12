//! Per-UNIT golden dump of the DSpark draft (`DSV41_DSPARK_UNIT_DUMP=1`).
//!
//! The accept rate is one number at the end of a run, so a wrong latent and a
//! wrong draft look identical from the outside. [`crate::dsv41::dump_dev`] is the
//! coarse half of the answer (the tap / drafts / verify argmax of every step);
//! this module is the FINE half: the intermediate activations *inside* one
//! `draft_forward`, unit by unit, so the official PyTorch reference's per-unit
//! capture (`unit_golden.pt`, see the sibling `unit_golden.py` harness) can be
//! diffed element by element instead of guessed at.
//!
//! # Keys (official `unit_golden.py` -> this dump)
//!
//! ```text
//!   official         device key(s)                       shape
//!   ---------------  ----------------------------------  -------------------
//!   main_x           main_x                              [dim]
//!   embed            embed                               [bs, hc, dim]
//!   h(pre_mix)       h_premix_block{s}                   [bs, dim]   (per block)
//!   q                q_block{s}                          [bs, nh, hd]
//!   kv               kv_block{s}                         [bs, hd]
//!   o                o_block{s}                          [bs, dim]
//!   moe_out          moe_out_block{s}                    [bs, dim]
//!   h                h_block{s}                          [bs, hc, dim] (per block)
//!   collapse         collapse                            [bs, dim]
//!   normed           normed                              [bs, dim]
//!   logits           logits_row0                         [vocab]
//!   output_ids       ids                                 [bs + 1]
//! ```
//!
//! `{s}` is the MTP block index (`0..n_mtp_layers`). `h_premix_block{s}` is the
//! device twin of the reference's `h = hc_pre(x, pre_mix)` — the pre-norm
//! collapse the q/kv projections read (the device keeps the rmsnorm in place, so
//! the value is taken between `hc_collapse` and `rmsnorm`). It is dumped in
//! addition to the keys the task enumerated because it is one of the official
//! keys and costs a single `bs * dim` D2H per block.
//!
//! # One capture per process, rank 0 only
//!
//! The gate is armed ONCE (`DSV41_DSPARK_UNIT_DUMP=1`) and armed for exactly one
//! `draft_forward` — the first call with `pos > 0` (a `pos == 0` call is the
//! prefill window seed and returns before the blocks). Every TP rank is a
//! separate process, so only rank 0 writes; an unguarded dump would have
//! `world` identical files racing for the same path.
//!
//! # Format
//!
//! A single JSON object at `DSV41_DSPARK_UNIT_PATH` (default
//! `/tmp/unit_ferrite.json`):
//!
//! ```json
//!   {"_meta": {…}, "_shapes": {"main_x":[5120], …}, "main_x":[…], …}
//! ```
//!
//! Floats use Rust's shortest-roundtrip `Display`, NOT `serde_json`: a
//! non-finite activation has to stay VISIBLE (serde_json would silently turn it
//! into `null`, losing exactly the signal a numeric diff looks for). Non-finite
//! values are spelled `NaN` / `Infinity` / `-Infinity` — the spelling Python's
//! `json` accepts — so the comparison script stays a plain `json.load`. `_meta`
//! and `_shapes` carry the geometry the flat arrays need to be reshaped back
//! into tensors.

use std::ffi::c_void;
use std::fmt::Write as _;
use std::io::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use ferrite_types::{FerriteError, Result};

use crate::dsv41::device::Device;

/// `DSV41_DSPARK_UNIT_DUMP=1` arms the dump; `0` or unset leaves it off. Cached
/// in a `OnceLock` for the same reason [`crate::dsv41::dump_dev::enabled`] is:
/// this is read from a decode step, and a bare `getenv` there is the slip this
/// project has been bitten by before.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("DSV41_DSPARK_UNIT_DUMP")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

/// Where the capture goes: `DSV41_DSPARK_UNIT_PATH`, else
/// `/tmp/unit_ferrite.json`.
pub fn path() -> &'static str {
    static P: OnceLock<String> = OnceLock::new();
    P.get_or_init(|| {
        std::env::var("DSV41_DSPARK_UNIT_PATH")
            .unwrap_or_else(|_| "/tmp/unit_ferrite.json".to_string())
    })
}

/// The one-shot arm: `true` for exactly ONE call in the process. The capture is
/// a per-unit D2H at every step of one forward, so a per-step dump would be
/// both enormous and pointless — the official harness records one forward too.
pub fn arm_once() -> bool {
    static TAKEN: AtomicBool = AtomicBool::new(false);
    !TAKEN.swap(true, Ordering::Relaxed)
}

/// The collected units, in dump order. Built on the first eligible
/// `draft_forward` and serialised at its end.
pub struct UnitDump {
    units: Vec<(String, Vec<f32>)>,
    shapes: Vec<(String, Vec<usize>)>,
}

impl Default for UnitDump {
    fn default() -> Self {
        Self::new()
    }
}

impl UnitDump {
    pub fn new() -> Self {
        UnitDump {
            units: Vec::new(),
            shapes: Vec::new(),
        }
    }

    /// D2H one `f32` unit and record its shape. `dims`'s product is the element
    /// count, so the caller states the tensor's geometry once and it travels
    /// into `_shapes` for the Python side to reshape with.
    pub fn push_f32(
        &mut self,
        dev: &Device,
        name: &str,
        ptr: *const f32,
        dims: &[usize],
    ) -> Result<()> {
        let n: usize = dims.iter().product();
        let view = Device::view(ptr as *mut c_void, n * 4);
        let mut buf = vec![0f32; n];
        dev.download_f32(&view, &mut buf)?;
        self.units.push((name.to_string(), buf));
        self.shapes.push((name.to_string(), dims.to_vec()));
        Ok(())
    }

    /// Same, for an `i32` unit (the Markov sampler's `ids`). Widened to `f32`
    /// on the host: token ids are far below 2^24, so the round trip is exact,
    /// and it keeps one numeric type across the whole dump.
    pub fn push_i32(
        &mut self,
        dev: &Device,
        name: &str,
        ptr: *const i32,
        dims: &[usize],
    ) -> Result<()> {
        let n: usize = dims.iter().product();
        let view = Device::view(ptr as *mut c_void, n * 4);
        let mut bytes = vec![0u8; n * 4];
        dev.download_u8(&view, &mut bytes)?;
        let buf: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect();
        self.units.push((name.to_string(), buf));
        self.shapes.push((name.to_string(), dims.to_vec()));
        Ok(())
    }

    /// Serialise every unit to [`path`]. `meta` is a pre-formatted JSON object
    /// body (no braces) carrying the step's geometry.
    pub fn write(&self, meta: &str) -> Result<()> {
        let mut s = String::with_capacity(1 << 20);
        s.push_str("{\"_meta\":");
        s.push_str(meta);
        s.push_str(",\"_shapes\":{");
        for (i, (k, dims)) in self.shapes.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let _ = write!(s, "\"{k}\":[");
            for (j, d) in dims.iter().enumerate() {
                if j > 0 {
                    s.push(',');
                }
                let _ = write!(s, "{d}");
            }
            s.push(']');
        }
        s.push('}');
        for (k, v) in &self.units {
            s.push(',');
            let _ = write!(s, "\"{k}\":[");
            for (j, x) in v.iter().enumerate() {
                if j > 0 {
                    s.push(',');
                }
                fmt_f32(&mut s, *x);
            }
            s.push(']');
        }
        s.push_str("}\n");

        let p = path();
        std::fs::write(p, s.as_bytes())
            .map_err(|e| FerriteError::Config(format!("unit dump: write {p}: {e}")))?;
        Ok(())
    }
}

/// JSON number for one `f32`. Finite values keep Rust's shortest-roundtrip
/// `Display`; the non-finite ones get the spellings Python's `json` parses — and
/// that make a NaN/inf visible instead of a `null`.
fn fmt_f32(out: &mut String, v: f32) {
    if v.is_nan() {
        out.push_str("NaN");
    } else if v.is_infinite() {
        out.push_str(if v > 0.0 { "Infinity" } else { "-Infinity" });
    } else {
        let _ = write!(out, "{v}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_finite_floats_stay_visible_and_json_parsable() {
        let mut s = String::new();
        for v in [1.5f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            if !s.is_empty() {
                s.push(',');
            }
            fmt_f32(&mut s, v);
        }
        // The exact spellings Python's `json.load` accepts (Rust's own `{}`
        // would print `inf`/`-inf`, which it does not).
        assert_eq!(s, "1.5,NaN,Infinity,-Infinity");
    }
}
