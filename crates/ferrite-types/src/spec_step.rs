//! The speculative-decoding seam: ONE accept chain for every MTP / DSpark step.
//!
//! `ferrite-exec`'s GLM MTP step (`TpCluster::mtp_step`) and `ferrite-models`'s
//! DSV41 DSpark step (`DevChain::dspark_spec_step`) are two independently
//! evolved implementations of the same four phases:
//!
//! ```text
//! draft(nd) -> verify(block) -> accept(longest prefix k) -> commit(k)
//! ```
//!
//! Almost everything inside those phases is per-model and stays per-model: the
//! draft count (`FERRITE_MTP_N - 1` vs the checkpoint's 5), the verify block
//! layout (`[t_last, d1..d_nd]` vs `[d1..d5]`), where the draft's hidden state
//! comes from (in-graph ping-pong vs an imported tap), how the block's tokens
//! reach the verify (device embed vs a graph-external H2D refresh) and what a
//! commit costs (one fused kernel vs a host mirror plus a compressor replay).
//!
//! The ONE piece that is pure shared math — and the piece both chains have
//! independently tripped over — is the ACCEPT chain: the longest prefix of the
//! drafts that the verify block confirmed. A wrong index into it silently
//! degrades the accept rate (or accepts a token the verify never proposed)
//! without any error surfacing, which is exactly the failure mode this module
//! removes: there is now a single implementation, and each chain binds it to
//! its own block layout once, via [`SpecStep::ANCHOR_IS_IN_BLOCK`].
//!
//! This module lives in `ferrite-types` because that is the only crate both
//! engines already depend on (`ferrite-exec` and `ferrite-models` are siblings;
//! neither may depend on the other). It carries no device code and no CUDA
//! feature, so it stays unit-testable on a CPU-only box.

use crate::Result;

/// A token id as it travels through a spec chain.
///
/// Both chains hold their accept inputs in device-download buffers, but not in
/// the same element type: the DSV41 chain downloads its draft and verify ids as
/// `u32`, while the GLM chain's argmax kernels write `f32` buffers (the ids are
/// stored exactly — integral, non-negative, < 2^24). The conversion below is
/// character-for-character the one the GLM accept loop performed per element
/// before the chains were unified (`drafts[i] as u32 == out[i] as u32`), so no
/// comparison changes meaning.
pub trait SpecToken: Copy {
    /// This element's token id.
    fn spec_token(self) -> u32;
}

impl SpecToken for u32 {
    #[inline]
    fn spec_token(self) -> u32 {
        self
    }
}

impl SpecToken for f32 {
    #[inline]
    fn spec_token(self) -> u32 {
        self as u32
    }
}

/// Longest accepted prefix of one speculative step — THE shared accept chain.
///
/// `drafts[i]` is the draft's proposal for the `i`-th new position of the step.
/// `judges[i]` is the verify argmax that JUDGES `drafts[i]`: the caller has
/// already aligned the two slices against each other, because the alignment is
/// the only thing the two block layouts disagree about:
///
/// * `anchor_is_in_block = true` — GLM, whose verify block is `[t_last, d1..d_nd]`:
///   block row 0 IS the anchor row, so row `i`'s argmax judges `drafts[i]`
///   index-for-index and `judges` is the verify argmax exactly as downloaded.
///   The anchor is always accepted, so the returned `k` counts the tokens the
///   step EMITS: `k in 1..=drafts.len() + 1`. `k == 1` means "no draft survived
///   and the step emits the anchor row's bonus token alone"; `k == n_v` means
///   "every draft survived plus the bonus".
/// * `anchor_is_in_block = false` — DSV41, whose verify block is `[d1..d5]`: the
///   anchor row is forwarded by the plain single-row step, not by the block, so
///   its argmax (`next` — the token the step emits anyway) has to lead the
///   chain: `judges = [next] ++ verify_out[..drafts.len() - 1]` (the block's LAST
///   row contributes the bonus token and is therefore not a judge of any draft).
///   The returned `k` counts ACCEPTED DRAFTS: `k in 0..=drafts.len()`, where
///   `0` means the first draft already disagreed with the anchor's argmax.
///
/// Both conventions describe the same chain and differ only by the
/// always-accepted anchor: for the same aligned prefix,
/// `spec_accept(d, judges, false) + 1 == spec_accept(d, judges ++ [any], true)`.
///
/// At most `min(drafts.len(), judges.len())` positions are compared, so the
/// returned value is capped by the shorter slice — a truncated judge array can
/// never make the chain read past it.
#[inline]
pub fn spec_accept<D: SpecToken, J: SpecToken>(
    drafts: &[D],
    judges: &[J],
    anchor_is_in_block: bool,
) -> usize {
    // The longest common prefix of the proposals against their judges: the
    // number of drafts the verify confirmed.
    let lim = drafts.len().min(judges.len());
    let mut matched = 0;
    while matched < lim && drafts[matched].spec_token() == judges[matched].spec_token() {
        matched += 1;
    }
    if anchor_is_in_block {
        // The block's row 0 is the anchor row: the anchor itself is always
        // accepted, so the emitted count is the accepted drafts plus the bonus.
        matched + 1
    } else {
        // No anchor row in the block: the caller prepended the anchor's judge,
        // so the accepted-draft count is the matched prefix itself.
        matched
    }
}

/// One speculative-decoding step: draft → verify → accept → commit.
///
/// The trait fixes the two things the chains genuinely share — the four-phase
/// contract and the accept arithmetic — and nothing else. The per-step inputs
/// are the impl's own associated type ([`SpecStep::Step`]) precisely so neither
/// chain is forced into the other's model: GLM's step needs `(seq, plans,
/// num_dsa)`, DSV41's needs `(&mut DsparkDev, token, pos)`.
///
/// The accept chain is bound to the impl's block layout through
/// [`SpecStep::ANCHOR_IS_IN_BLOCK`] and must never be re-implemented per chain:
/// call [`SpecStep::accept`] and the layout fact lives in exactly one place per
/// impl.
pub trait SpecStep {
    /// What one step needs beyond `&mut self` — the impl's own per-step inputs,
    /// deliberately opaque.
    ///
    /// `'a` borrows those inputs for the step; `'b` is the inner lifetime of
    /// whatever THEY borrow (DSV41's `DsparkDev<'b>` borrows the device, config
    /// and weights for the whole serving scope, which strictly outlives the
    /// borrow of the draft device itself). Two lifetimes, because `&mut T` is
    /// invariant in `T`: collapsing them would make the type impossible for a
    /// caller to build.
    type Step<'a, 'b>
    where
        'b: 'a;
    /// What one step produces (GLM: the next token; DSV41: the step's report).
    type Report;
    /// Whether the verify block's row 0 IS the anchor row: `true` for GLM's
    /// `[t_last, d1..d_nd]`, `false` for DSV41's `[d1..d5]`. Read only by
    /// [`spec_accept`].
    const ANCHOR_IS_IN_BLOCK: bool;

    /// Run the four phases and return the step's result.
    ///
    /// The phases stay inside this one call for both chains (GLM fuses
    /// verify+accept+commit into a single cross-rank `fan_out`, DSV41
    /// interleaves its snapshot/rollback around the verify), so the trait does
    /// not expose them as separate methods: doing so would change when the
    /// ranks synchronise.
    fn spec_step<'a, 'b>(&mut self, step: Self::Step<'a, 'b>) -> Result<Self::Report>
    where
        'b: 'a;

    /// The shared accept chain, bound to this impl's block layout.
    ///
    /// `drafts`/`judges` follow [`spec_accept`]'s contract (the judges already
    /// aligned, the anchor's judge prepended when the block does not contain
    /// the anchor row).
    #[inline]
    fn accept<D: SpecToken, J: SpecToken>(drafts: &[D], judges: &[J]) -> usize {
        spec_accept(drafts, judges, Self::ANCHOR_IS_IN_BLOCK)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The DSV41 block width (`DSPARK_DRAFTS` in `ferrite-models`; duplicated
    /// here because `ferrite-types` may not depend on the model crate).
    const M: usize = 5;

    /// The GLM accept loop exactly as it stood in `ferrite-exec/src/tp.rs`
    /// (`TpCluster::mtp_step`) before the unification. Kept verbatim as the
    /// equivalence oracle: the shared chain must reproduce it for every input.
    fn glm_accept_before(drafts: &[f32], out: &[f32]) -> usize {
        let n_v = out.len();
        let mut k: i32 = 1;
        while (k as usize) < n_v && drafts[(k - 1) as usize] as u32 == out[(k - 1) as usize] as u32
        {
            k += 1;
        }
        k as usize
    }

    /// The DSV41 accept loop exactly as it stood in
    /// `ferrite-models/src/dsv41/chain_dev.rs` (`DevChain::dspark_spec_step`)
    /// before the unification, verbatim (with `m` standing for `DSPARK_DRAFTS`).
    fn dsv41_accept_before(drafts: &[u32], next: u32, verify_out: &[u32], m: usize) -> usize {
        let mut k_acc = 0usize;
        if drafts[0] == next {
            k_acc = 1;
            while k_acc < m && drafts[k_acc] == verify_out[k_acc - 1] {
                k_acc += 1;
            }
        }
        k_acc
    }

    /// The judge array the DSV41 call site builds: the anchor's own argmax
    /// (`next`) followed by the block rows that judge a draft (all but the last
    /// row, whose argmax is the bonus token).
    fn dsv41_judges(next: u32, verify_out: &[u32]) -> Vec<u32> {
        let mut judges = vec![0u32; M];
        judges[0] = next;
        judges[1..].copy_from_slice(&verify_out[..M - 1]);
        judges
    }

    fn digits(mut code: u32, n: usize, base: u32) -> Vec<u32> {
        (0..n)
            .map(|_| {
                let d = code % base + 1;
                code /= base;
                d
            })
            .collect()
    }

    #[test]
    fn spec_accept_reproduces_the_glm_chain_on_every_input() {
        // Every (drafts, out) pair over a 3-symbol alphabet for nd = 0..=4:
        // covers the empty block, every mismatch position and the full accept.
        for nd in 0..=4usize {
            for code in 0..3u32.pow((2 * nd + 1) as u32) {
                let d = digits(code, nd, 3);
                let o = digits(code / 3u32.pow(nd as u32), nd + 1, 3);
                let drafts: Vec<f32> = d.iter().map(|v| *v as f32).collect();
                let out: Vec<f32> = o.iter().map(|v| *v as f32).collect();
                assert_eq!(
                    spec_accept(&drafts, &out, true),
                    glm_accept_before(&drafts, &out),
                    "nd={nd} drafts={drafts:?} out={out:?}"
                );
            }
        }
    }

    #[test]
    fn spec_accept_reproduces_the_dsv41_chain_on_every_input() {
        // Every (drafts, next, verify_out) over a 2-symbol alphabet at the real
        // block width — including the full accept and the first-check failure.
        for code in 0..2u32.pow((M + 1 + M) as u32) {
            let d = digits(code, M, 2);
            let next = (code / 2u32.pow(M as u32)) % 2 + 1;
            let vo = digits(code / 2u32.pow((M + 1) as u32), M, 2);
            let drafts: Vec<u32> = d.clone();
            let judges = dsv41_judges(next, &vo);
            assert_eq!(
                spec_accept(&drafts, &judges, false),
                dsv41_accept_before(&drafts, next, &vo, M),
                "drafts={drafts:?} next={next} verify_out={vo:?}"
            );
        }
    }

    #[test]
    fn spec_accept_ignores_the_blocks_last_judge() {
        // The DSV41 chain never judges a draft with the block's LAST row (that
        // row's argmax is the bonus token) — the prepended-anchor form must
        // keep that property for every value that row could take.
        let drafts = [7u32, 8, 9, 10, 11];
        let mut vo = [8u32, 9, 10, 11, 0];
        let mut seen = Vec::new();
        for last in 0..4u32 {
            vo[M - 1] = last;
            seen.push(spec_accept(&drafts, &dsv41_judges(7, &vo), false));
        }
        assert_eq!(seen, vec![M; 4]);
    }

    #[test]
    fn the_two_layouts_are_one_chain_shifted_by_the_anchor() {
        // For the same aligned prefix, the GLM convention counts the anchor's
        // bonus token and the DSV41 convention does not: k_glm = k_dsv + 1.
        // (GLM's `out` = the prepended judge array + its own trailing bonus.)
        let drafts: Vec<u32> = vec![5, 6, 7, 8, 9];
        for cut in 0..=M {
            let hit = |i: usize| if i < cut { drafts[i] } else { 99 };
            let judges: Vec<u32> = (0..M).map(hit).collect();
            let mut out: Vec<u32> = judges.clone();
            out.push(42); // GLM's block row `n_v - 1` (the bonus row)
            assert_eq!(
                spec_accept(&drafts, &judges, false) + 1,
                spec_accept(&drafts, &out, true),
                "cut={cut}"
            );
            assert_eq!(spec_accept(&drafts, &judges, false), cut.min(M));
        }
    }

    #[test]
    fn spec_accept_empty_drafts() {
        // No draft at all: GLM still emits the anchor row's token, DSV41 has
        // nothing to accept (its anchor is emitted by the single-row step).
        assert_eq!(spec_accept::<u32, u32>(&[], &[7, 8], true), 1);
        assert_eq!(spec_accept::<u32, u32>(&[], &[7, 8], false), 0);
        assert_eq!(spec_accept::<f32, f32>(&[], &[], true), 1);
    }

    #[test]
    fn spec_accept_full_accept_stops_at_the_block_width() {
        // `k` never exceeds what the shorter slice holds: nd + 1 (GLM) / m
        // (DSV41), even when the judge array is longer than the block.
        let drafts: Vec<u32> = vec![1, 2, 3, 4];
        let out: Vec<u32> = vec![1, 2, 3, 4, 5, 6]; // longer than nd + 1
        assert_eq!(spec_accept(&drafts, &out, true), 5);
        let judges: Vec<u32> = vec![9, 1, 2, 3, 4]; // drafts[0] != 9
        assert_eq!(spec_accept(&drafts, &judges, false), 0);
        let judges: Vec<u32> = vec![1, 2, 3, 4]; // as long as drafts
        assert_eq!(spec_accept(&drafts, &judges, false), 4);
    }

    #[test]
    fn spec_accept_first_check_failure_rejects_the_whole_prefix() {
        // DSV41's off-by-one chain only accepts a suffix once the FIRST draft
        // matched the anchor's argmax, even if later drafts look right.
        let drafts: Vec<u32> = vec![9, 2, 3];
        let judges: Vec<u32> = vec![1, 2, 3, 4]; // judges[0] = next = 1
        assert_eq!(spec_accept(&drafts, &judges, false), 0);
        // GLM's index-aligned chain has no such first check: the anchor is
        // free, so a mismatch on drafts[0] still yields the bonus token.
        assert_eq!(spec_accept(&drafts, &judges, true), 1);
    }

    #[test]
    fn spec_accept_token_representations_agree() {
        // The GLM (f32) and DSV41 (u32) buffers must decide identically.
        let d_u32 = [3u32, 4, 5];
        let j_u32 = [3u32, 4, 9, 0];
        let d_f32: Vec<f32> = d_u32.iter().map(|v| *v as f32).collect();
        let j_f32: Vec<f32> = j_u32.iter().map(|v| *v as f32).collect();
        assert_eq!(
            spec_accept(&d_u32, &j_u32, true),
            spec_accept(&d_f32, &j_f32, true)
        );
        assert_eq!(spec_accept(&d_u32, &j_u32, true), 3);
    }
}
