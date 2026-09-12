//! GLM batched MTP — Step B groundwork: the verify *size class*, the graph
//! keys, the row→(seq, tok) mapping, and the design records for the B-row
//! draft chain and the batched commit.
//!
//! Wave 5 (see `docs/agent/wave5-multiseq-plan.md` §3.4/§4.2) has to turn the
//! per-seq MTP step into a batched one. Step A made [`ferrite_kernel::cuda::MtpState`]
//! per-seq (the [`ferrite_kernel::cuda::MtpStateB`] pool); this module is the
//! pure-host layer of Step B: the shape arithmetic and the naming, with NO
//! call sites yet — `gpu_engine`'s `FERRITE_MTP && max_seqs > 1 → forcing 1`
//! gate stays closed until every piece below is in place.
//!
//! # Verdict 1 — the verify graph is keyed by `(padded seqs, n_v)`, not by rows
//!
//! Today the verify graph is `mega_v{seq}` with `n_v = FERRITE_MTP_N` rows
//! (`tp.rs`'s `decode_step_mega` capture), i.e. ONE seq's block. B seqs need
//! `B × n_v` rows in ONE chain pass, laid out seq-major: row `r` belongs to
//! seq `r / n_v`, token `r % n_v` ([`seq_of_row`] / [`tok_of_row`]).
//!
//! The natural SGLang-style move is "one graph per padded row count", but that
//! key is UNSOUND here: the captured chain embeds (a) the width of the
//! per-size pointer tables ([`ferrite_kernel::cuda::CudaBackend::gdn_state_tables`],
//! `dsa_ptr_tables` — keyed by `(layer|family, size)` = the SEQ count, not the
//! row count) and (b) the row→seq divisor `n_v` baked into the row addressing.
//! A rows-only key would alias `(b=4, n_v=6)` with `(b=8, n_v=3)` — both 24
//! rows, different table widths and different divisors — and the alias would
//! silently read the wrong seq's state. So: key by `(padded seqs, n_v)`, and
//! let the row count be the *derived* padded size ([`verify_rows`]).
//!
//! Cost/benefit vs the alternatives:
//! - **per-seq graphs** (today's `mega_v{seq}`): B captures (~1-2s each, once
//!   per seq, amortized) and B replays per step. No correctness risk, no new
//!   kernel work — this is the documented fallback if the B-row capture faults
//!   (Xid 31), plan §6.3.
//! - **(B, n_v)-keyed shared graph** (chosen): `≤ |SEQ_LADDER|` captures total
//!   (one per rung, pre-warmed at boot), ONE replay per step, membership
//!   changes reuse the graph because the tables' CONTENT is refreshed (the
//!   same property that keeps `megab_b{size}` alive across retires,
//!   `gpu_engine`'s `free`). Price: intermediates scale with `b × n_v` rows
//!   (B=16, n_v=3 → 48 rows); the batched decode already pays exactly this
//!   shape class and the projections use the `small_n_rows` GEMV path
//!   (`n ≤ 16` is a GEMV; 48 rows is a tiled GEMM at ~16× the single-seq
//!   cost — see `mtp_batch::verify_rows` users for the B=16 note).
//!
//! # Verdict 2 — the draft chain graph is NOT shareable as-is
//!
//! The draft graphs `mega_d{seq}_{i}` are already per-seq, and their INPUT is
//! already device-resident (`MtpState.tokens_dev[0]`, written by a 4-byte H2D
//! before each replay — `mega_d` needs no `graph_run_ids` staging). So "each
//! step only swaps ids/pos" holds for the input half. It does NOT hold for the
//! state half: `draft_step_dev` records the seq's `MtpState` buffers
//! (`hprev`/`emb_devs`/`h_d`/`d_argmax_dev`), the MTP layer's GDN conv/gdn
//! states and the DSA family cache pointers as IMMEDIATE kernel args, so a
//! shared graph would replay seq A's pointers for every row.
//!
//! Sharing therefore requires the same conversion the batched decode already
//! made: the draft chain becomes a B-row chain whose per-seq state is reached
//! through device pointer tables (`gdn_state_tables`/`dsa_ptr_tables`), keyed
//! `mega_d_b{size}_{i}` — note the shape differs from the verify: the draft
//! chain is one row PER SEQ (B rows), not `B × n_v`. Decision for Step B:
//! **keep `mega_d{seq}_{i}` per-seq** ([`DRAFT_CHAIN_SHARED`] is false) and
//! spend the first batched landing on the verify graph, which carries the
//! whole 45-layer step; the draft chain is one MTP layer, so `B × (n_v-1)`
//! extra replays are bounded. The shared draft graph is worth revisiting only
//! if instrumented replay time says the draft share exceeds ~10% at B=16.
//!
//! # Verdict 3 — batched commit: `k` travels as a `[B]` array, one launch
//!
//! `ferrite_mtp_commit` takes ONE `k` (pinned or device pointer) and one
//! 6-pointer-per-GDN-layer plan whose A-side pointers are that seq's recurrent
//! states; the kernel derives the B-side source from `k` and writes
//! `hprev <- hf_v[k-1]`. B seqs need B distinct `k`s and B plans. Two shapes:
//!
//! - **(a) B launches, per-seq `MtpState`** — zero kernel change; the pinned
//!   `k_pin` of each state is written from the accept result. This is what the
//!   code can already express today (`mtp_commit(seq, k)` per seq) and it is
//!   the fallback if the single-launch variant's kernel work slips.
//! - **(b) ONE launch over a batched layout** (chosen for the steady state):
//!   plan `[B][n_gdn][6]` (seq-major, [`commit_plan_row`]), `k` as a `[B] i32`
//!   array (pinned host array written from the accept results, or the per-row
//!   device array the accept kernel already fills), `hprev` widened to
//!   `[B][hidden]`, `hf_v` to `[B][n_v][hidden]`. The kernel maps plan row →
//!   seq as `seq = plan_row / n_gdn` and indexes
//!   `hf_v[seq*n_v*hidden + (k[seq]-1)*hidden + r]`, `hprev[seq*hidden + r]`.
//!   Pad rows: `k[pad] = 1` (NOT 0 — the source index is `k-1`, so 0 would
//!   read `hf_v[-hidden]`); the write lands on the shared dummy A state and is
//!   discarded. Layout arithmetic: [`commit_batch_layout`].
//!
//! # Verdict 4 — `dsa_append_batched` for `n_v` rows: the mapping is seq-major, and
//! the `ntok=1` hard-code must become "ntok = the per-seq row count"
//!
//! `dsa_append_batched`'s grid is `(B, ntok)` and it addresses
//! `kvb + (seq*ntok + tok)*row`, with `t0 = *t0_tbl[seq]` — that IS the
//! seq-major `[B][ntok]` layout this module's [`seq_of_row`] uses. The
//! `cuda.rs` call site pins `ntok = 1` with a ROOT-CAUSE comment that is
//! correct for the DECODE-batched layout (`kvb` is `[B, 1, row]` there, so any
//! `ntok > 1` reads rows past `B`) but does NOT apply to the MTP verify, whose
//! rows are `[B][n_v]` by construction. Two kernel-side gaps remain before the
//! verify can pass `ntok = n_v`:
//!   1. the idx/gate tail reads `ki[seq*idm + c]` — it must read the ROW's
//!      `ki[(seq*ntok + tok)*idm + c]` (the single-seq kernel
//!      `dsa_cache_append_kernel` already does `ki[t*idm + c]`);
//!   2. the device advance (`dev_adv`) hard-codes `t0 + 1`; with `ntok` rows
//!      appended in one launch it must be `t0 + ntok`.
//! The per-seq (grid = B) kernels downstream must keep receiving `B` — the seq
//! count — while the per-row kernels (projections, sparse attention) receive
//! `B*n_v`: `kpool_compress_batched`, `indexer_topk_batched` and
//! `pool_expand_batched` are called with `ni` = the row count in the decode
//! path only because `rows == seqs` there. The sparse-attention tables stay
//! B-indexed with a `row / n_v` divisor.
//!
//! NOTE: GLM's DSA carries position IMPLICITLY through each seq's pinned
//! `t0`/`total` (`dsa_host_advance`, zero-copy reads; there is no RoPE in the
//! mega chain) — so the batched verify needs NO per-row position table, unlike
//! the DSV41 path (plan §4.2). The row→seq mapping is the whole story here.

/// The seq-count ladder the batched paths pad to (mirrors the `[1,2,4,8,16,32]`
/// search in `tp.rs::decode_step_batched` and `gpu_engine`'s scheduler). The
/// MTP verify sizes its graphs off the same rungs, so a membership change
/// inside a rung reuses the graph.
pub const SEQ_LADDER: [usize; 6] = [1, 2, 4, 8, 16, 32];

/// The padded seq count for a live set of `live` seqs: the smallest ladder rung
/// `>= live` (the exact count when it exceeds the ladder — the same fallback
/// the batched decode uses).
pub fn padded_seqs(live: usize) -> usize {
    SEQ_LADDER
        .iter()
        .copied()
        .find(|&s| s >= live)
        .unwrap_or(live)
}

/// The MTP verify graph's row count for a live set: `padded_seqs(live) * n_v`.
/// `n_v = FERRITE_MTP_N` (the verify width; drafts = `n_v - 1`). This is the
/// number the `small_n_rows` GEMV path must be asked for (`n ≤ 16` is the fast
/// path; B=16 at n_v=3 means 48 rows, i.e. the tiled-GEMM domain — see the
/// module docs).
pub fn verify_rows(live: usize, n_v: usize) -> usize {
    padded_seqs(live).max(1) * n_v.max(1)
}

/// The MTP verify graph name: `mega_v_b{padded}_{n}` — keyed by the padded seq
/// count AND the verify width (see verdict 1; a rows-only key would alias
/// different table widths).
pub fn verify_graph_name(padded: usize, n_v: usize) -> String {
    format!("mega_v_b{padded}_n{n_v}")
}

/// The seq a verify row belongs to: `row / n_v` (seq-major layout). Every
/// per-seq table in the chain (`gdn_state_tables`, `dsa_ptr_tables`, the commit
/// plan) is indexed with this.
pub fn seq_of_row(row: usize, n_v: usize) -> usize {
    row / n_v.max(1)
}

/// The slot inside its seq a verify row occupies: `row % n_v` — the index into
/// `[t_last, d1..d_{n_v-1}]` and the `ntok` axis of `dsa_append_batched`.
pub fn tok_of_row(row: usize, n_v: usize) -> usize {
    row % n_v.max(1)
}

/// Whether the draft chain's graphs (`mega_d{seq}_{i}`) are shared across seqs.
/// `false` = per-seq, the Step-B decision: the draft graphs record per-seq
/// state pointers as immediate kernel args, so sharing needs the B-row +
/// pointer-table conversion first (verdict 2).
pub const DRAFT_CHAIN_SHARED: bool = false;

/// The draft graph name a SHARED draft chain would use (one row per seq, so the
/// key is the padded SEQ count — NOT `verify_rows`). Unused while
/// [`DRAFT_CHAIN_SHARED`] is false; it exists so the naming is pinned down
/// before the conversion.
pub fn draft_graph_name(padded: usize, draft: usize) -> String {
    format!("mega_d_b{padded}_{draft}")
}

/// The row of the batched commit plan for `(seq, layer)`: the plan is
/// `[B][n_gdn][6]` seq-major (verdict 3), so the kernel's
/// `seq = plan_row / n_plans` round-trips.
pub fn commit_plan_row(seq: usize, layer: usize, n_plans: usize) -> usize {
    seq * n_plans + layer
}

/// Host-side layout of the B-row MTP commit (verdict 3): all the strides the
/// kernel needs, derived from the same numbers `MtpCommitPlan` already holds
/// (`MtpState.commit`: `plan`, `k_pin`, `mtp_n`, `conv_len`, `gdn_len`,
/// `hidden`) plus the batch width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitBatchLayout {
    /// Seq count (the plan's outer dimension).
    pub seqs: usize,
    /// `[B][n_gdn][6]` — GDN layers per seq (the same `n_plans` every state
    /// reports today; all seqs share the layer plan).
    pub n_plans: usize,
    /// Verify width `n_v` — the `hf_v` row block per seq.
    pub n_v: usize,
    /// Elements per `hf_v` seq block: `n_v * hidden`.
    pub hf_v_seq_stride: usize,
    /// Elements per `hprev` seq block (and the `hf_v` row stride): `hidden`.
    pub hidden: usize,
    /// Elements per plan row: 6 pointers = 12 f32 (the packed-pointer form
    /// `mtp_setup_bufs` writes).
    pub plan_row_elems: usize,
    /// `k` array length: one i32 per seq (pad rows must be 1, not 0).
    pub k_len: usize,
    /// GDN layers per seq (the `k == n` full-B source); convenience for the
    /// `plan_row` divisor.
    pub conv_len: usize,
    pub gdn_len: usize,
}

/// Build the batched commit's layout. Pure arithmetic — the kernel change that
/// consumes it is Step B's last piece.
pub fn commit_batch_layout(seqs: usize, n_plans: usize, n_v: usize, hidden: usize) -> CommitBatchLayout {
    CommitBatchLayout {
        seqs,
        n_plans,
        n_v,
        hf_v_seq_stride: n_v.max(1) * hidden,
        hidden,
        plan_row_elems: 12, // 6 device pointers packed 2 f32 each
        k_len: seqs,
        conv_len: 0,
        gdn_len: 0,
    }
}

impl CommitBatchLayout {
    /// Total f32 elements of the packed plan table: `seqs * n_plans * 12`.
    pub fn plan_elems(&self) -> usize {
        self.seqs * self.n_plans * self.plan_row_elems
    }

    /// Total f32 elements of the widened `hf_v`: `seqs * n_v * hidden`.
    pub fn hf_v_elems(&self) -> usize {
        self.seqs * self.hf_v_seq_stride
    }

    /// Total f32 elements of the widened `hprev` / `hf_dev`: `seqs * hidden`.
    pub fn hprev_elems(&self) -> usize {
        self.seqs * self.hidden
    }

    /// The `hf_v` source offset for `(seq, k)` — the row the commit kernel
    /// selects (`k == n_v` uses the full-B state, `k < n_v` the snapshot
    /// instead; the offset is the same either way).
    pub fn hf_v_offset(&self, seq: usize, k: usize) -> usize {
        seq * self.hf_v_seq_stride + (k.saturating_sub(1)) * self.hidden
    }

    /// The `hprev` destination offset for `seq`.
    pub fn hprev_offset(&self, seq: usize) -> usize {
        seq * self.hidden
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ladder_pads_like_the_decode_path() {
        assert_eq!(padded_seqs(0), 1, "an empty set still sizes one rung");
        assert_eq!(padded_seqs(1), 1);
        assert_eq!(padded_seqs(3), 4);
        assert_eq!(padded_seqs(4), 4);
        assert_eq!(padded_seqs(5), 8);
        assert_eq!(padded_seqs(16), 16);
        assert_eq!(padded_seqs(17), 32);
        assert_eq!(padded_seqs(33), 33, "past the ladder: the exact count");
    }

    #[test]
    fn verify_rows_is_padded_seqs_times_n_v() {
        assert_eq!(verify_rows(1, 3), 3, "the single-seq n_v=3 block");
        assert_eq!(verify_rows(2, 3), 6);
        assert_eq!(verify_rows(3, 3), 12, "padded to 4 seqs");
        assert_eq!(verify_rows(16, 3), 48, "the B=16 rung");
        assert_eq!(verify_rows(16, 5), 80, "n_v scales the block");
        assert_eq!(verify_rows(0, 3), 3, "never zero rows");
    }

    /// The graph key must distinguish `(4 seqs, n_v=6)` from `(8 seqs, n_v=3)`
    /// even though both are 24 rows — the tables' width and the row divisor
    /// differ (verdict 1).
    #[test]
    fn graph_key_keeps_rows_aliasing_apart() {
        assert_eq!(verify_rows(4, 6), verify_rows(8, 3), "same row count");
        assert_ne!(
            verify_graph_name(padded_seqs(4), 6),
            verify_graph_name(padded_seqs(8), 3),
            "the (seqs, n_v) key must not alias on equal row counts"
        );
        assert_eq!(verify_graph_name(16, 3), "mega_v_b16_n3");
        assert_eq!(verify_graph_name(1, 3), "mega_v_b1_n3");
    }

    #[test]
    fn row_mapping_is_seq_major() {
        let n_v = 3;
        // seq0: [t_last, d1, d2]; seq1 starts at row 3.
        assert_eq!(seq_of_row(0, n_v), 0);
        assert_eq!(tok_of_row(0, n_v), 0, "the anchor row of seq 0");
        assert_eq!(seq_of_row(2, n_v), 0);
        assert_eq!(tok_of_row(2, n_v), 2);
        assert_eq!(seq_of_row(3, n_v), 1);
        assert_eq!(tok_of_row(3, n_v), 0, "the anchor row of seq 1");
        assert_eq!((seq_of_row(5, n_v), tok_of_row(5, n_v)), (1, 2));
        // the mapping covers the rows exactly once
        let rows = verify_rows(2, n_v);
        let mut seen = vec![0usize; 2 * n_v];
        for r in 0..rows {
            seen[seq_of_row(r, n_v) * n_v + tok_of_row(r, n_v)] += 1;
        }
        assert!(seen.iter().all(|&c| c == 1));
    }

    #[test]
    fn draft_graphs_are_per_seq_and_named_by_the_seq_count() {
        assert!(!DRAFT_CHAIN_SHARED, "Step B keeps the per-seq draft chain");
        assert_eq!(draft_graph_name(16, 1), "mega_d_b16_1");
        // The draft chain is one row per seq — NOT verify_rows.
        assert_eq!(draft_graph_name(padded_seqs(16), 0), "mega_d_b16_0");
        assert_ne!(draft_graph_name(16, 0), verify_graph_name(16, 3));
    }

    #[test]
    fn commit_plan_rows_are_seq_major() {
        let n_plans = 38; // the GDN layer count of the production plan
        assert_eq!(commit_plan_row(0, 0, n_plans), 0);
        assert_eq!(commit_plan_row(0, n_plans - 1, n_plans), n_plans - 1);
        assert_eq!(commit_plan_row(1, 0, n_plans), n_plans, "seq 1's block starts after seq 0's");
        assert_eq!(commit_plan_row(7, 3, n_plans) / n_plans, 7, "the kernel's divisor round-trips");
        assert_eq!(commit_plan_row(7, 3, n_plans) % n_plans, 3);
    }

    #[test]
    fn commit_batch_layout_strides() {
        let l = commit_batch_layout(4, 38, 3, 5120);
        assert_eq!(l.plan_elems(), 4 * 38 * 12);
        assert_eq!(l.hf_v_elems(), 4 * 3 * 5120, "one n_v row block per seq");
        assert_eq!(l.hprev_elems(), 4 * 5120);
        assert_eq!(l.k_len, 4, "one k per seq (pad rows must carry 1)");
        assert_eq!(l.hf_v_offset(0, 1), 0);
        assert_eq!(l.hf_v_offset(0, 3), 2 * 5120, "k=3 -> row 2 of seq 0");
        assert_eq!(l.hf_v_offset(2, 1), 2 * 3 * 5120, "seq 2's block");
        assert_eq!(l.hprev_offset(3), 3 * 5120);
        // k=0 (a pad row that forgot to set 1) must not go negative.
        assert_eq!(l.hf_v_offset(1, 0), 1 * 3 * 5120);
    }
}
