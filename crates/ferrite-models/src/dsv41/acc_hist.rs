//! `k_acc` histogram + p1/p_j decomposition (`DSV41_ACC_HISTOGRAM=1`).
//!
//! # Why this module exists
//!
//! The DSpark accept rate is quoted as ONE number — `mean-k`, i.e. the mean
//! accepted-draft count over the steps (`[dspark] steps=… mean-k=… tok/step=…`).
//! A mean is a lossy summary of a quantity that is **bimodal**: the accept chain
//! is a prefix chain, so every round either loses its FIRST link (`k_acc = 0`,
//! the anchor's argmax and the draft's top-1 disagree) or walks to the tail
//! (`k_acc -> DSPARK_DRAFTS`). `accept-ceiling-analysis.md` names the two modes
//! and the two rates (`p1 ≈ 0.43` for the head link, `q ≈ 0.82` for the tail);
//! the mean alone cannot tell which of the two moved.
//!
//! Worse, two different stacks can quote the same mean with completely different
//! shapes (`docs/agent/accept-ceiling-analysis.md` §1.3 made the old histogram's
//! mean-upper-bound 1.93 and the current 2.24 stack MATHEMATICALLY exclusive),
//! and the CURRENT stack's histogram has never been measured. R0 of the
//! `acc-improve-path` verdict is therefore a pure calibration step: print the
//! same-arm histogram (SWALLOW + `TAP_INPUT`) plus the head/tail decomposition,
//! and nothing else.
//!
//! # What the statistics are
//!
//! For `N` steps with `hist[k] = #steps whose k_acc was k` (`k in 0..=DSPARK_DRAFTS`):
//!
//! ```text
//!   p1   = (N - hist[0]) / N                      the FIRST link's rate
//!   p_j  = Σ_{k>=j+1} hist[k] / Σ_{k>=j} hist[k]  the rate at link j, GIVEN the
//!                                                 chain reached link j
//!   mean-k = Σ k·hist[k] / N                      what the serve line calls
//!   tok/step = mean-k + 1                         "
//! ```
//!
//! `first_match` (the anchor argmax vs `drafts[0]`, evaluated per step) is
//! reported next to `p1` as an independent count of the same event: their
//! equality is a consistency check on the accept bookkeeping, and a divergence
//! means the arm's `k_acc` is not the plain prefix chain.
//!
//! # The oracle half (R1)
//!
//! `note_oracle` accumulates the ORACLE-TAP contrast (`DSV41_ORACLE_TAP=1`, see
//! [`crate::dsv41::chain_dev::DevChain`]'s swallowed arm): the draft is re-run
//! once with the main chain's OWN hidden at the position it is asked about (the
//! verify block's anchor row), and its top-1 is compared with that row's argmax
//! — i.e. `drafts[0] == next` under the CORRECT input. `oracle_rate >= ~0.85`
//! says the head follows the backbone when fed the right tap ⇒ the online gap is
//! geometric (the tap's phase), while `~0.43` says the head cannot follow even
//! then ⇒ the input pairing itself is wrong. Both counters are local to this
//! module so the two experiments print one summary.
//!
//! # Cost and gating
//!
//! The gate `DSV41_ACC_HISTOGRAM=1` (default OFF) is READ ONCE into a `OnceLock`
//! — the house rule for every hot-path flag (a per-step `getenv` is the slip this
//! project has been bitten by before). With it off, each call site is one cached
//! boolean branch and one atomic-free early return; with it on, the per-step
//! print is one `eprintln!` per round and the summary is O(1).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// Number of `k_acc` bins: `0..=DSPARK_DRAFTS` (`DSPARK_DRAFTS = 5`, duplicated
/// here because this module must not depend on `dspark_dev`'s constants — it is
/// also the `hist` array's length in the summary line).
pub const ACC_BINS: usize = 6;

/// Number of LINKS in the accept chain (`= DSPARK_DRAFTS`): `k_acc` counts how
/// many of them matched, so the p_j ladder runs `j = 1..=ACC_LINKS` and a
/// `reached[ACC_BINS]` entry would be structurally always zero.
pub const ACC_LINKS: usize = ACC_BINS - 1;

/// `DSV41_ACC_HISTOGRAM=1` arms the histogram. `0` or unset leaves it off.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("DSV41_ACC_HISTOGRAM")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

/// `DSV41_ORACLE_TAP=1` arms the oracle half (R1). The probe that PRODUCES the
/// events lives on the chain (`DevChain`'s swallowed arm reads the same
/// variable); this is what records and summarises them, so the two gates are
/// independent — an R1-only run reports the oracle rate without paying R0's
/// per-step print.
pub fn oracle_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("DSV41_ORACLE_TAP")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Whether ANY of this module's recorders is armed — the summary is gated on
/// this, not on the histogram alone.
fn active() -> bool {
    enabled() || oracle_enabled()
}

/// The accumulated statistics. Behind a `Mutex` rather than a field on the chain
/// because the summary is printed from the serve driver (`ferrite-dsv41`), which
/// owns the loop but not the chain, and because R0's whole point is that the
/// counts cover the SAME arm over the WHOLE run (all ranks' steps are reported by
/// rank 0's chain, and a re-dispatch would otherwise split them).
struct Stats {
    /// `hist[k]` = steps whose accepted draft prefix was `k`.
    hist: [u64; ACC_BINS],
    /// Steps seen (any arm).
    steps: u64,
    /// Steps whose FIRST link matched (`drafts[0]` vs the anchor's argmax).
    first_match: u64,
    /// Per-ARM step counts, so a summary can tell a pure SWALLOW run from one
    /// that also paid bootstrap rounds on the legacy arm.
    arms: Vec<(&'static str, u64)>,
    /// The oracle arm: how many oracle re-runs ran, and how many agreed.
    oracle_steps: u64,
    oracle_hits: u64,
    /// Oracle runs whose draft top-1 differed (the ones the geometric hypothesis
    /// is about) — kept apart from `oracle_steps - oracle_hits` so the two can
    /// never disagree by accident.
    oracle_miss: u64,
}

impl Stats {
    const fn new() -> Self {
        Self {
            hist: [0; ACC_BINS],
            steps: 0,
            first_match: 0,
            arms: Vec::new(),
            oracle_steps: 0,
            oracle_hits: 0,
            oracle_miss: 0,
        }
    }
}

fn stats() -> &'static Mutex<Stats> {
    static S: OnceLock<Mutex<Stats>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(Stats::new()))
}

/// The process-global running `p1` is printed per step, so it needs a lock-free
/// read of the two counters; the histogram itself only matters at the summary.
/// Two atomics keep the per-step line cheap and exact.
static RUN_STEPS: AtomicU64 = AtomicU64::new(0);
static RUN_NONZERO: AtomicU64 = AtomicU64::new(0);

/// Record one step's accept result.
///
/// `arm` is the arm tag the dispatch chose (`"legacy"` / `"aligned"` /
/// `"swallowed"` / `"lazy"`), `k_acc` the accepted draft prefix (the LEGACY
/// meaning: `k_emit - 1`, `0..=DSPARK_DRAFTS` — all four arms report that shape),
/// and `first_match` whether `drafts[0]` equalled the anchor's argmax.
///
/// Prints `[acc-hist] step=… arm=… pos=… k_acc=… first_match=… p1_running=…`
/// once per step — the per-round trace R0 needs (a summary alone cannot show
/// whether the zeros cluster at the start of a request or are spread).
pub fn note(arm: &'static str, pos: usize, k_acc: usize, first_match: bool) {
    if !enabled() {
        return;
    }
    let k = k_acc.min(ACC_BINS - 1);
    let steps = RUN_STEPS.fetch_add(1, Ordering::Relaxed) + 1;
    if k > 0 {
        RUN_NONZERO.fetch_add(1, Ordering::Relaxed);
    }
    let p1 = RUN_NONZERO.load(Ordering::Relaxed) as f64 / steps as f64;
    eprintln!(
        "[acc-hist] step={steps} arm={arm} pos={pos} k_acc={k_acc} first_match={first_match} \
         p1_running={p1:.4}"
    );
    if let Ok(mut s) = stats().lock() {
        s.hist[k] += 1;
        s.steps += 1;
        if first_match {
            s.first_match += 1;
        }
        match s.arms.iter_mut().find(|(a, _)| *a == arm) {
            Some((_, n)) => *n += 1,
            None => s.arms.push((arm, 1)),
        }
    }
}

/// Record ONE oracle re-run's outcome (R1). `main_top` is the main chain's own
/// argmax at the position the draft was asked about (the verify block's anchor
/// row, `rows[0]`), `draft_top` the draft's top-1 fed the main chain's hidden.
pub fn note_oracle(pos: usize, draft_top: u32, main_top: u32) {
    if !oracle_enabled() {
        return;
    }
    let hit = draft_top == main_top;
    eprintln!(
        "[acc-oracle] pos={pos} draft_top={draft_top} main_top={main_top} hit={hit}"
    );
    if let Ok(mut s) = stats().lock() {
        s.oracle_steps += 1;
        if hit {
            s.oracle_hits += 1;
        } else {
            s.oracle_miss += 1;
        }
    }
}

/// Print the histogram + the p1/p_j decomposition, then leave the counters
/// alone (the caller decides whether to [`reset`]).
///
/// `tag` labels the caller (`"serve"`, `"oneshot"`, …) so a log with several
/// summaries says which run each belongs to.
pub fn print_summary(tag: &str) {
    if !active() {
        return;
    }
    let s = match stats().lock() {
        Ok(s) => s,
        Err(_) => return,
    };
    let n = s.steps;
    if n == 0 {
        // An R1-only run has no `k_acc` samples but does have oracle samples:
        // report the oracle half rather than a bare "steps=0".
        if s.oracle_steps > 0 {
            eprintln!(
                "[acc-hist-summary] tag={tag} steps=0 hist={} oracle: steps={} hit={} miss={} \
                 rate={:.4}",
                "(histogram off)",
                s.oracle_steps,
                s.oracle_hits,
                s.oracle_miss,
                s.oracle_hits as f64 / s.oracle_steps as f64
            );
        } else {
            eprintln!("[acc-hist-summary] tag={tag} steps=0 (no speculative steps recorded)");
        }
        return;
    }
    let nf = n as f64;
    // The p_j ladder. The accept chain has `ACC_LINKS = ACC_BINS - 1` links
    // (`k_acc = link count`), so link `j` (1-based) is "reached" iff
    // `k_acc >= j`: `reached[j]` = the steps that matched at least j links.
    let mut reached = [0u64; ACC_BINS + 1];
    for j in 1..=ACC_LINKS {
        reached[j] = (j..ACC_BINS).map(|k| s.hist[k]).sum();
    }
    let mean_k = (0..ACC_BINS).map(|k| k as u64 * s.hist[k]).sum::<u64>() as f64 / nf;
    let mut line = format!(
        "[acc-hist-summary] tag={tag} steps={n} mean-k={mean_k:.4} tok/step={:.4}",
        mean_k + 1.0
    );
    line.push_str(" hist={");
    for (k, c) in s.hist.iter().enumerate() {
        if k > 0 {
            line.push(' ');
        }
        line.push_str(&format!("{k}:{c}"));
    }
    line.push('}');
    // p1 comes from the histogram; `first_match` is the independent per-step
    // count of the same event (see the module doc) — print both so a mismatch is
    // visible instead of averaged away.
    let p1 = (n - s.hist[0]) as f64 / nf;
    line.push_str(&format!(
        " p1={p1:.4} first_match={:.4}",
        s.first_match as f64 / nf
    ));
    line.push_str(" p_j={");
    for j in 2..=ACC_LINKS {
        // `reached[j-1]` = the steps that got past link j-1, `reached[j]` = the
        // ones that also matched link j; the ratio is link j's conditional rate.
        let d = reached[j - 1];
        let num = reached[j];
        if d == 0 {
            line.push_str(" -");
        } else {
            line.push_str(&format!(" {:.4}", num as f64 / d as f64));
        }
        if j < ACC_LINKS {
            line.push(' ');
        }
    }
    line.push('}');
    // The tail's mean conditional rate, which is the `q ≈ 0.82` the verdict
    // contrasts against `p1`. Averaged over the links that actually ran.
    let mut tsum = 0.0f64;
    let mut tn = 0u32;
    for j in 2..=ACC_LINKS {
        let d = reached[j - 1];
        if d > 0 {
            tsum += reached[j] as f64 / d as f64;
            tn += 1;
        }
    }
    if tn > 0 {
        line.push_str(&format!(" tail_q={:.4}", tsum / tn as f64));
    }
    if !s.arms.is_empty() {
        line.push_str(" arms={");
        for (i, (a, c)) in s.arms.iter().enumerate() {
            if i > 0 {
                line.push(' ');
            }
            line.push_str(&format!("{a}:{c}"));
        }
        line.push('}');
    }
    if s.oracle_steps > 0 {
        line.push_str(&format!(
            " oracle: steps={} hit={} miss={} rate={:.4}",
            s.oracle_steps,
            s.oracle_hits,
            s.oracle_miss,
            s.oracle_hits as f64 / s.oracle_steps as f64
        ));
    }
    eprintln!("{line}");
}

/// Zero the counters (a fresh request / a fresh binary's baseline). The serve
/// driver calls this at a request boundary so the summary never mixes two
/// prompts; `RUN_*` are reset with the rest.
pub fn reset() {
    RUN_STEPS.store(0, Ordering::Relaxed);
    RUN_NONZERO.store(0, Ordering::Relaxed);
    if let Ok(mut s) = stats().lock() {
        *s = Stats::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The p1/p_j arithmetic, spelled out against a hand-computed histogram.
    ///
    /// `hist = [4, 2, 2, 1, 1, 0]` (N = 10, `ACC_LINKS = 5`):
    ///   reached = [6, 4, 2, 1, 0]
    ///   p1 = 6/10;  p2 = 4/6;  p3 = 2/4;  p4 = 1/2;  p5 = 0/1
    #[test]
    fn p_ladder_matches_the_prefix_chain_definition() {
        let hist = [4u64, 2, 2, 1, 1, 0];
        let n: u64 = hist.iter().sum();
        let reached: Vec<u64> = (1..=ACC_LINKS)
            .map(|j| (j..ACC_BINS).map(|k| hist[k]).sum())
            .collect();
        assert_eq!(n, 10);
        assert_eq!(ACC_LINKS, 5);
        assert_eq!((n - hist[0]) as f64 / n as f64, 0.6);
        assert_eq!(reached, vec![6, 4, 2, 1, 0]);
        let p: Vec<f64> = (1..ACC_LINKS)
            .map(|j| reached[j] as f64 / reached[j - 1] as f64)
            .collect();
        assert_eq!(p, vec![4.0 / 6.0, 0.5, 0.5, 0.0]);
    }

    /// `k_acc` outside the bin range can never slice out of bounds — the clamp
    /// is the same one [`note`] applies.
    #[test]
    fn bin_clamp_is_total() {
        assert_eq!(9usize.min(ACC_BINS - 1), ACC_BINS - 1);
    }
}
