//! DSpark draft/verify numerical parity — a single-rank, no-HTTP GPU self-test.
//!
//! Wave 2 built the draft and the verify block by watching `serve`+`curl` accept
//! numbers, one three-minute loop per iteration. This module is the replacement:
//! one process, one GPU, one `cargo test`, two numeric contrasts that localise a
//! draft/verify regression without a serving stack in the way.
//!
//! # What it runs
//!
//! ```text
//!   pass 1 (truth)   prefill P ids, then `steps + DSPARK_DRAFTS` single-row
//!                    decode steps
//!                    -> stream[j] = the token the SINGLE-ROW path emits at
//!                       position P + j   (the oracle both contrasts compare to)
//!
//!   pass 2 (probe)   reset, replay the same prefill + `steps` decode steps;
//!                    at every step, after the real single-row step:
//!                      a. draft    import_tap + draft_forward(t, pos) + drafts()
//!                      b. verify   step_rows(stream[pos+1 .. pos+1+k]) — the
//!                                  TRUE continuation, NOT the draft — then the
//!                                  snapshot/rollback pair, so the chain is left
//!                                  exactly where the real step left it
//! ```
//!
//! # The two contrasts
//!
//! * **draft oracle** — `drafts[j]` vs `stream[i + 1 + j]`: what the draft
//!   proposed against what the target actually produced. This is the direct
//!   *upper bound* on the accepted block length (a draft token can only survive
//!   the accept chain if it equals the token the target would have emitted).
//! * **verify parity (the iron rule)** — `step_rows(truth)[r]` must equal the
//!   single-row argmax at the same position, `stream[i + 2 + r]`, position for
//!   position. The verify block is a re-scheduling of the single-row path (one
//!   m-row forward instead of m one-row forwards), so ANY mismatch here is a
//!   kernel/geometry bug, never a modelling question.
//!
//! # Why two passes
//!
//! The verify block has to be fed the true continuation, and those tokens do not
//! exist yet at the point the verify would run — they are five single-row steps
//! in the future. Pass 2 therefore replays a chain whose every future token is
//! already known, which also makes the draft/verify arithmetic pure host code.
//!
//! [`DevChain::dspark_shadow_step`](crate::dsv41::chain_dev::DevChain::dspark_shadow_step)
//! cannot serve here: it feeds the draft's own proposals to the verify by
//! construction (that is what measures the ACCEPT rule), while parity needs the
//! true tokens in the rows. That is also why this module reaches for the chain's
//! `dspark_snapshot` / `dspark_rollback` / `dspark_tap_ptr` `pub(crate)` hooks
//! rather than a new public surface — the ordering discipline is shadow_step's,
//! the tokens fed to `step_rows` are not.
//!
//! # Running it (remote, single rank, TP=1)
//!
//! ```text
//! FERRITE_MODEL_DIR=/opt/dlami/nvme/models/DeepSeek-V4.1-Flash \
//! FERRITE_KERNELS=$PWD/kernels/cuda/libferrite_kernels.so \
//! CUDA_VISIBLE_DEVICES=0 \
//!   cargo test -p ferrite-models --lib dspark_parity -- --ignored --nocapture
//! ```
//!
//! Environment knobs (the `DSV41_*` spellings are accepted as fallbacks):
//!
//! * `FERRITE_MODEL_DIR` / `DSV41_MODEL_DIR` — the checkpoint directory.
//! * `FERRITE_KERNELS` / `DSV41_KERNELS` — `libferrite_kernels.so`; defaults to
//!   `kernels/cuda/libferrite_kernels.so` (relative to the CWD).
//! * `FERRITE_PARITY_IDS` / `DSV41_PARITY_IDS` — comma-separated prefill ids,
//!   overriding the canned [`PARITY_PREFILL_IDS`]. Pass the real prompt ids here
//!   for a meaningful draft match rate.
//! * `FERRITE_PARITY_STEPS` — probed decode steps (default [`PARITY_STEPS`]).
//! * `DSV41_SKIP_ENGRAM_WEIGHTS=1` — skip the ~189 GiB engram tables, exactly as
//!   `serve`/`dsv41-run` allow. Parity is unaffected (both paths see the same
//!   weights), but the draft match rate is a quality number on this checkpoint
//!   and the tables are part of it.
//!
//! Two caveats worth knowing before reading a failing report as a draft bug:
//!
//! * `DSV41_ENG_HOST=1` (the host n-gram fallback) has `step_rows` mutate the
//!   host-side token cache that the snapshot/rollback pair does NOT cover. The
//!   default (device engram hash) is the path `shadow_step` is exercised on and
//!   the one to test with.
//! * The draft's own window rings are deliberately NOT rolled back (the
//!   `dspark_shadow_step` rule: a probe must leave the draft where a real
//!   speculative run would). Pass 2 therefore builds the draft history from
//!   step 0 of its own replay, which is exactly what `serve` does.

use std::path::Path;

use ferrite_types::{FerriteError, Result};

use crate::dsv41::chain_dev::{DevChain, RunOpts};
use crate::dsv41::config::Dsv41Config;
use crate::dsv41::device::Device;
use crate::dsv41::dspark_dev::{DsparkDev, DSPARK_DRAFTS};
use crate::dsv41::engram::TokenMap;
use crate::dsv41::load::Loader;

/// The canned prefill prompt, 32 ids: this checkpoint's chat frame
/// (`<|begin_of_sentence|><|User|>` … `<|Assistant|></think>`, the same markers
/// `dsv41-run` and `frame.rs` pin) wrapping a run of in-vocabulary CJK-range
/// ids.
///
/// ⚠️ The body is a PLACEHOLDER, not the tokenization of any particular text:
/// it exists so the tool runs with zero arguments. `verify_parity_ok` is
/// input-independent, but `draft_match_rate` is a quality number — override this
/// with `FERRITE_PARITY_IDS=0,128803,...` to measure a real prompt.
pub const PARITY_PREFILL_IDS: [u32; 32] = [
    0, 128803, // <|begin_of_sentence|><|User|>
    49704, 12288, 86433, 30517, 97221, 15632, 60118, 74390, 24876, 51903, 38655, 83047, 27109,
    66742, 41008, 95011, 57913, 12360, 70482, 29841, 81556, 44777, 63210, 53994, 88265, 35123,
    77608, 19340, // (placeholder body)
    128804, 128822, // <|Assistant|></think>
];

/// How many decode steps are probed by default. Each probed step costs one
/// single-row step (in both passes) plus one draft and one `m`-row verify, so
/// this is the knob that trades report resolution for wall time.
pub const PARITY_STEPS: usize = 16;

/// One probed decode step: the raw numbers the two contrasts are computed from.
#[derive(Debug, Clone)]
pub struct ParityStep {
    /// 0-based index of the probed step.
    pub step: usize,
    /// The device position of the real step (`prefill_len + step`).
    pub pos: usize,
    /// The token the real (single-row) step consumed: `stream[step]`.
    pub anchor: u32,
    /// The real step's argmax: `stream[step + 1]`.
    pub next: u32,
    /// The block the draft proposed for positions `pos + 1 .. pos + 1 + slots`.
    pub drafts: [u32; DSPARK_DRAFTS],
    /// The target's real tokens at those positions: `stream[pos+1 ..]`.
    pub truth: [u32; DSPARK_DRAFTS],
    /// `k`: how many of the `DSPARK_DRAFTS` slots this step could compare.
    pub slots: usize,
    /// How many of those slots the draft got right (its accept contribution).
    pub draft_hits: usize,
    /// `step_rows(truth[..slots])` — the verify block's per-row argmax.
    pub verify_out: [u32; DSPARK_DRAFTS],
    /// The single-row argmax the same rows must reproduce:
    /// `stream[pos + 2 + r]` for row `r` — the token the single-row path emits
    /// one step after the row's own position.
    pub verify_expect: [u32; DSPARK_DRAFTS],
    /// Rows actually compared (`min(slots, known horizon)`).
    pub verify_rows: usize,
    /// Row indices whose `verify_out` differed from `verify_expect`.
    pub verify_mismatch: Vec<usize>,
}

impl ParityStep {
    /// Every compared verify row reproduced the single-row path.
    pub fn verify_ok(&self) -> bool {
        self.verify_mismatch.is_empty()
    }
}

/// What [`dspark_parity_run`] measured.
#[derive(Debug, Clone)]
pub struct ParityReport {
    pub prefill_len: usize,
    pub steps: usize,
    /// `draft_hits / draft_slots` over every probed step — the direct bound on
    /// how much of each drafted block the accept chain could take.
    pub draft_match_rate: f32,
    pub draft_hits: usize,
    pub draft_slots: usize,
    /// Every compared verify row matched the single-row path.
    pub verify_parity_ok: bool,
    pub verify_rows: usize,
    pub verify_rows_mismatched: usize,
    pub per_step_detail: Vec<ParityStep>,
}

impl ParityReport {
    /// The draft's per-slot match rate over one step, `None` when nothing was
    /// comparable.
    pub fn step_rate(&self, step: usize) -> Option<f32> {
        let s = self.per_step_detail.get(step)?;
        (s.slots > 0).then(|| s.draft_hits as f32 / s.slots as f32)
    }
}

impl std::fmt::Display for ParityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "--- dspark parity: prefill {} ids, {} probed decode steps ---",
            self.prefill_len, self.steps
        )?;
        for s in &self.per_step_detail {
            writeln!(
                f,
                "step {:>2} pos {:>3} anchor {:>6} -> {:>6} | draft {:?} truth {:?} \
                 {}/{} | verify {:?} expect {:?} {}",
                s.step,
                s.pos,
                s.anchor,
                s.next,
                s.drafts,
                &s.truth[..s.slots],
                s.draft_hits,
                s.slots,
                s.verify_out,
                &s.verify_expect[..s.verify_rows],
                if s.verify_ok() {
                    "ok".to_string()
                } else {
                    format!("MISMATCH rows {:?}", s.verify_mismatch)
                }
            )?;
        }
        writeln!(
            f,
            "draft match rate: {:.3} ({}/{})  [accept upper bound]",
            self.draft_match_rate, self.draft_hits, self.draft_slots
        )?;
        write!(
            f,
            "verify parity: {} ({}/{} rows reproduce the single-row argmax)",
            if self.verify_parity_ok { "OK" } else { "FAIL" },
            self.verify_rows - self.verify_rows_mismatched,
            self.verify_rows
        )
    }
}

/// `FERRITE_PARITY_IDS` / `DSV41_PARITY_IDS`, else the canned
/// [`PARITY_PREFILL_IDS`].
fn parity_ids() -> Vec<u32> {
    for k in ["FERRITE_PARITY_IDS", "DSV41_PARITY_IDS"] {
        if let Ok(v) = std::env::var(k) {
            let ids: Vec<u32> = v
                .split(',')
                .filter_map(|s| s.trim().parse::<u32>().ok())
                .collect();
            if !ids.is_empty() {
                return ids;
            }
        }
    }
    PARITY_PREFILL_IDS.to_vec()
}

/// `(model_dir, kernel .so)` from the environment, `FERRITE_*` first with the
/// `DSV41_*` spellings as fallbacks. Used by the `#[ignore]` GPU test.
pub fn parity_env() -> (String, String) {
    let dir = std::env::var("FERRITE_MODEL_DIR")
        .or_else(|_| std::env::var("DSV41_MODEL_DIR"))
        .expect("set FERRITE_MODEL_DIR (or DSV41_MODEL_DIR) to the checkpoint directory");
    let so = std::env::var("FERRITE_KERNELS")
        .or_else(|_| std::env::var("DSV41_KERNELS"))
        .unwrap_or_else(|_| "kernels/cuda/libferrite_kernels.so".to_string());
    (dir, so)
}

/// The engram's n-gram hash keys tokens by a compressed id space that is a pure
/// function of the tokenizer, so it is precomputed once into
/// `engram_token_map.bin` (129280 i64 little-endian) — the same load `serve` and
/// `dsv41-run` do. `None` (no file) leaves the chain's engram disabled.
fn load_eng_map(dir: &str, cfg: &Dsv41Config) -> Option<TokenMap> {
    std::fs::read(format!("{dir}/engram_token_map.bin"))
        .ok()
        .filter(|b| b.len() % 8 == 0 && !b.is_empty())
        .map(|b| {
            let v: Vec<i64> = b
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            TokenMap::from_table(v, cfg.engram_compressed_vocab_size)
        })
}

/// Run the single-rank draft/verify parity probe and return the report.
///
/// `model_dir` is the checkpoint directory (its own `config.json` is the config
/// source, exactly as `dsv41-run` does) and `so` the `libferrite_kernels.so`
/// path. The run is TP=1 (`world = 1`, `rank = 0`): [`DsparkDev`]'s only
/// collective is its MoE all-reduce, and a world of 1 needs none, so no
/// [`Collective`](crate::dsv41::tp::Collective) is built.
///
/// Returns [`Err`] — rather than an all-zero report — on the first structural
/// problem (no draft in the config, a replayed chain that diverged from the
/// truth pass, a short `step_rows` block), because a parity tool that answers
/// "0% draft match" for a mis-specified run is worse than no tool.
pub fn dspark_parity_run(model_dir: &str, so: &str) -> Result<ParityReport> {
    // The tap hook is gated by `Dsv41Config::dspark_armed()`, which caches its
    // env read in a OnceLock on FIRST use. This has to happen before anything
    // builds or steps a chain: arming late leaves `layer()`'s tap hook out of
    // the captured step graph, and the draft would silently read a stale tap.
    std::env::set_var("DSV41_DSPARK", "1");

    // ---- config ----
    let dir = Path::new(model_dir);
    let cfg_txt = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| FerriteError::Config(format!("parity: read {model_dir}/config.json: {e}")))?;
    let cfg = Dsv41Config::from_json_str(&cfg_txt)?;
    if !cfg.dspark_enabled() {
        return Err(FerriteError::Config(format!(
            "parity: {model_dir} has no DSpark draft (dspark_block_size={} n_mtp_layers={})",
            cfg.dspark_block_size, cfg.n_mtp_layers
        )));
    }
    let ids = parity_ids();
    let steps = std::env::var("FERRITE_PARITY_STEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(PARITY_STEPS)
        .max(1);

    // ---- device + weights (one rank, TP=1) ----
    let dev = Device::open(so)?;
    let mut loader = Loader::new(dir, &dev)?;
    if std::env::var("DSV41_SKIP_ENGRAM_WEIGHTS")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        loader.skip_prefixes.push("engram.embed.".into());
    }
    let w = loader.load(&cfg, 1, 0)?;
    dev.sync()?;
    let eng_map = load_eng_map(model_dir, &cfg);

    let mut chain = DevChain::new(&dev, &cfg, &w, RunOpts::from_env(), eng_map)?;
    chain.reset()?;
    // world = 1, rank = 0: `comm` stays None (no collective exists to set).
    let mut dspark = DsparkDev::new(&dev, &w, &cfg, 1, 0)?;
    // The draft's RoPE positions are a subset of the chain's, so it borrows the
    // chain's tables instead of building a private ~128 MiB copy.
    let (cos, sin) = chain.rope_tables();
    dspark.set_rope_tables(cos, sin);

    let p = ids.len();
    // One probe needs `DSPARK_DRAFTS` future tokens, so the truth pass runs
    // `steps + DSPARK_DRAFTS` decode steps: that way every verify row of every
    // probed step has a single-row argmax to be compared against.
    let horizon = steps + DSPARK_DRAFTS;

    // ---- pass 1: the single-row oracle ----
    let mut stream: Vec<u32> = Vec::with_capacity(horizon + 1);
    let mut next = 0u32;
    for (i, &t) in ids.iter().enumerate() {
        // `step` (not `step_dev`) for the prompt: it never captures the decode
        // graph, which is exactly the discipline `dsv41-run` uses for prefill.
        next = chain.step(t, i)?;
    }
    // stream[0] is the token at position `p` — the prompt's last step's argmax.
    stream.push(next);
    for i in 0..horizon {
        let t = stream[i];
        let n = chain.step_dev(t, p + i)?;
        stream.push(n);
    }
    debug_assert_eq!(stream.len(), horizon + 1);

    // ---- pass 2: replay and probe ----
    chain.reset()?;
    for (i, &t) in ids.iter().enumerate() {
        chain.step(t, i)?;
    }

    let mut detail = Vec::with_capacity(steps);
    let (mut hits, mut slots) = (0usize, 0usize);
    let (mut verify_rows, mut verify_bad) = (0usize, 0usize);

    for i in 0..steps {
        let pos = p + i;
        let anchor = stream[i];
        let next = chain.step_dev(anchor, pos)?;
        if next != stream[i + 1] {
            return Err(FerriteError::Config(format!(
                "parity: the replayed chain diverged at step {i} (pos {pos}): got {next}, pass 1 \
                 produced {} — the probe would compare against a stream this chain no longer \
                 produces",
                stream[i + 1]
            )));
        }

        // 1. the draft, from the tap the real step above just recorded.
        dspark.import_tap(chain.dspark_tap_ptr())?;
        dspark.draft_forward(anchor, pos)?;
        let drafts = dspark.drafts()?;

        // 2. the truth window this step's draft and verify are measured against.
        let k = DSPARK_DRAFTS.min(horizon - i);
        let mut truth = [0u32; DSPARK_DRAFTS];
        truth[..k].copy_from_slice(&stream[i + 1..i + 1 + k]);

        // 3. contrast 1: the draft oracle.
        let draft_hits = (0..k).filter(|&j| drafts[j] == truth[j]).count();
        hits += draft_hits;
        slots += k;

        // 4. contrast 2: verify parity. The SAME rows, fed the TRUE tokens, must
        //    reproduce the single-row argmax. The real step above already ran, so
        //    the block's row 0 is at `pos + 1` — the position `step_rows` itself
        //    reads off the (now advanced) device counter, which is what
        //    `dspark_snapshot`/`dspark_rollback` take as their base.
        let host = chain.dspark_snapshot(pos + 1, k)?;
        let rows = chain.step_rows(&truth[..k]);
        // Roll back unconditionally: a dirty ring/compressor would silently
        // change every later step's numbers, which is worse than the error.
        chain.dspark_rollback(pos + 1, k, &host)?;
        let rows = rows?;
        if rows.len() != k {
            return Err(FerriteError::Config(format!(
                "parity: step_rows returned {} rows for a {k}-row block",
                rows.len()
            )));
        }
        let mut verify_out = [0u32; DSPARK_DRAFTS];
        verify_out[..k].copy_from_slice(&rows);

        // Row r is fed `truth[r]` at position pos+1+r and predicts the token at
        // pos+2+r — `stream[i + 2 + r]`, i.e. where the single-row path is one
        // step further on. Only rows with a known successor are compared.
        let mut verify_expect = [0u32; DSPARK_DRAFTS];
        let mut mismatch = Vec::new();
        let vk = k.min(horizon.saturating_sub(i + 1));
        for r in 0..vk {
            verify_expect[r] = stream[i + 2 + r];
            if verify_out[r] != verify_expect[r] {
                mismatch.push(r);
            }
        }
        verify_rows += vk;
        verify_bad += mismatch.len();

        detail.push(ParityStep {
            step: i,
            pos,
            anchor,
            next,
            drafts,
            truth,
            slots: k,
            draft_hits,
            verify_out,
            verify_expect,
            verify_rows: vk,
            verify_mismatch: mismatch,
        });
    }

    let draft_match_rate = if slots == 0 {
        0.0
    } else {
        hits as f32 / slots as f32
    };
    Ok(ParityReport {
        prefill_len: p,
        steps,
        draft_match_rate,
        draft_hits: hits,
        draft_slots: slots,
        verify_parity_ok: verify_bad == 0,
        verify_rows,
        verify_rows_mismatched: verify_bad,
        per_step_detail: detail,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single-rank GPU parity self-test. Needs a real checkpoint and the CUDA
    /// kernels, so it is `#[ignore]`d — run it explicitly:
    ///
    /// ```text
    /// FERRITE_MODEL_DIR=/opt/dlami/nvme/models/DeepSeek-V4.1-Flash \
    /// FERRITE_KERNELS=$PWD/kernels/cuda/libferrite_kernels.so \
    /// CUDA_VISIBLE_DEVICES=0 \
    ///   cargo test -p ferrite-models --lib dspark_parity -- --ignored --nocapture
    /// ```
    ///
    /// The assertion is the iron rule, not the draft rate: a failed verify
    /// parity is a kernel/geometry bug, while a low draft match is a quality
    /// number that depends on the prompt (pass real ids via
    /// `FERRITE_PARITY_IDS`).
    #[test]
    #[ignore = "needs a GPU + the real checkpoint (run with --ignored)"]
    fn dspark_parity_gpu() {
        let (dir, so) = parity_env();
        let rep = dspark_parity_run(&dir, &so).expect("dspark_parity_run");
        // The full report is the point of --nocapture; the assert is the gate.
        println!("{rep}");
        assert!(
            rep.verify_parity_ok,
            "verify parity FAILED: {} of {} rows diverged from the single-row path",
            rep.verify_rows_mismatched, rep.verify_rows
        );
    }
}
