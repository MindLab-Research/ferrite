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
//!
//! # Verdict 5 — the row→seq map is the ONE place the two layouts differ
//!
//! `dsa_append_batched`'s kernel grid is `(B, ntok, h)` and it addresses
//! `kvb + (seq*ntok + tok)*row` — i.e. it ALREADY implements the padded
//! seq-major map, with `ntok = n_v`. The `cuda.rs` call site pins `ntok = 1`
//! because the DECODE-batched layout is `[B][1][row]` there (verdict 4). The
//! MTP verify's rows are `[B][n_v]` by construction ([`verify_rows`]), so the
//! same kernel wants `ntok = n_v` plus two fixes: the idx/gate tail must read
//! the ROW's `ki[row*idm + c]` (today: `ki[seq*idm + c]`), and the device
//! advance must move `t0` by the seq's whole block, not by 1.
//!
//! The ragged layout (plan §3.2) cannot be expressed by a divisor at all —
//! `Σ len_r` rows have no constant pitch — so it passes a `[rows][2]`
//! `(seq, tok)` device table ([`SeqRowMap::row_table`]) plus the per-seq block
//! lengths (the advance step). [`AppendPlan`] describes both shapes in one
//! struct: `row_map = None, block_len = None, ntok = n_v` for padded,
//! `row_map = Some(_), block_len = Some(_), ntok = 0` for ragged. With
//! `ntok = 1` and no tables it degenerates to exactly today's launch — the
//! backward-compatibility invariant the mapped kernel is written against
//! (`ferrite_dsa_append_batched_mapped`, `kernels/cuda/ferrite_kernels.cu`).

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

/// Which block layout a batched verify pass uses (plan §3.1 A vs §3.2 B).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowLayout {
    /// `padded_seqs(live) * n_v` rows: every seq owns exactly `n_v` rows,
    /// seq-major. FIXED shape, so it can live inside the `(seqs, n_v)`-keyed
    /// graph; a short block's missing drafts are `PAD_TOKEN` rows that map
    /// onto the shared dummy table slots (their outputs are discarded).
    Padded,
    /// `Σ_r len_r` rows: each seq owns its own block length (`len_r = 1 + k_r`
    /// — the anchor row plus its accepted-draft block). VARIABLE shape, so it
    /// is a non-graph path only (plan §3.2): `(seq, tok)` comes from a device
    /// table, not from a divisor.
    Ragged,
}

/// `(row) -> (seq_slot, tok_in_block)` for a batched pass, plus the per-seq
/// bases the rows' positions are offset from.
///
/// This is the ONE piece of arithmetic that differs between the padded and the
/// ragged verify layout (verdict 5): every other consumer — the per-seq
/// pointer tables (`gdn_state_tables`/`dsa_ptr_tables`), the commit plan, the
/// position tables — is indexed by the `seq_slot` this returns.
///
/// Both layouts are SEQ-MAJOR: seq `s` owns the contiguous row range
/// `[prefix[s], prefix[s+1])`, so the rows of one seq stay in ascending token
/// order (which is what the per-row attention interleave depends on, plan
/// §3.1's "排序坑").
///
/// Positions: a row's position/slot is `pos_base[seq] + tok`, i.e. each seq
/// carries its OWN base. For GLM's DSA that base is the seq's pinned `t0` (the
/// cache slot the row is appended at) rather than a RoPE position — the mega
/// chain has no RoPE (the module's closing NOTE).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeqRowMap {
    layout: RowLayout,
    /// Rows per seq (padded: `seqs` copies of `n_v`).
    block_len: Vec<usize>,
    /// `[seqs + 1]` row offsets; `prefix[seqs] == rows()`. The seq-major
    /// partition — a ragged map's whole content.
    prefix: Vec<usize>,
}

impl SeqRowMap {
    /// The padded map: `seqs` seqs of `n_v` rows each. `seqs` is the PADDED
    /// seq count (callers use [`padded_seqs`]); `n_v` is the verify width.
    pub fn padded(seqs: usize, n_v: usize) -> Self {
        let n_v = n_v.max(1);
        let seqs = seqs.max(1);
        Self::from_blocks(RowLayout::Padded, &vec![n_v; seqs])
    }

    /// The padded map for `live` seqs: sizes the block off the ladder rung
    /// (the same padding the batched decode uses, so the graph is reused
    /// across membership changes inside a rung).
    pub fn padded_for_live(live: usize, n_v: usize) -> Self {
        Self::padded(padded_seqs(live), n_v)
    }

    /// The ragged map: one entry per live seq, its own block length. A seq with
    /// a block of 0 (its whole draft block was consumed/rejected) owns no row —
    /// the mapping simply skips it.
    pub fn ragged(block_lens: &[usize]) -> Self {
        Self::from_blocks(RowLayout::Ragged, block_lens)
    }

    /// The map a verify pass uses: `ragged_lens = None` → the padded default
    /// (Step-B: plan §3.3 "A 起步"), `Some(lens)` → the ragged layout.
    pub fn for_verify(live: usize, n_v: usize, ragged_lens: Option<&[usize]>) -> Self {
        match ragged_lens {
            None => Self::padded(padded_seqs(live), n_v),
            Some(lens) => Self::ragged(lens),
        }
    }

    fn from_blocks(layout: RowLayout, block_lens: &[usize]) -> Self {
        let mut prefix = Vec::with_capacity(block_lens.len() + 1);
        prefix.push(0usize);
        let mut acc = 0usize;
        for &b in block_lens {
            acc += b;
            prefix.push(acc);
        }
        Self { layout, block_len: block_lens.to_vec(), prefix }
    }

    pub fn layout(&self) -> RowLayout {
        self.layout
    }

    /// The number of seq slots (B — the width of every per-seq table).
    pub fn seqs(&self) -> usize {
        self.block_len.len()
    }

    /// The number of rows (the kvb/ki/gate row dimension, grid.x).
    pub fn rows(&self) -> usize {
        self.prefix[self.prefix.len() - 1]
    }

    /// The uniform block length, when the map has one (padded always does).
    pub fn n_v(&self) -> Option<usize> {
        match self.block_len.first() {
            Some(&first) if self.block_len.iter().all(|&b| b == first) => Some(first),
            _ => None,
        }
    }

    /// Rows owned by `seq`.
    pub fn block_len(&self, seq: usize) -> Option<usize> {
        self.block_len.get(seq).copied()
    }

    /// Whether the mapping needs a DEVICE table (ragged) or is carried by the
    /// kernel's `ntok` divisor (padded — no table, so the padded launch is
    /// bit-identical to today's `dsa_append_batched`).
    pub fn needs_row_table(&self) -> bool {
        self.layout == RowLayout::Ragged
    }

    /// The seq slot `row` belongs to. `None` past the last row.
    pub fn seq_of_row(&self, row: usize) -> Option<usize> {
        if row >= self.rows() {
            return None;
        }
        // the last prefix <= row; `prefix[0] == 0 <= row` keeps this >= 1
        let s = self.prefix.partition_point(|p| *p <= row) - 1;
        Some(s)
    }

    /// The slot inside its seq `row` occupies: the index into the seq's draft
    /// block (`[t_last, d1..]`) and the `ntok` axis of the cache append.
    pub fn tok_of_row(&self, row: usize) -> Option<usize> {
        let seq = self.seq_of_row(row)?;
        Some(row - self.prefix[seq])
    }

    /// `(seq_slot, tok_in_block)` for `row`.
    pub fn at(&self, row: usize) -> Option<(usize, usize)> {
        let seq = self.seq_of_row(row)?;
        Some((seq, row - self.prefix[seq]))
    }

    /// The position/slot of `row`: its seq's own base plus the row's token
    /// offset. `None` when `row` is out of range or `pos_base` is short.
    pub fn pos_of_row(&self, row: usize, pos_base: &[i32]) -> Option<i32> {
        let (seq, tok) = self.at(row)?;
        pos_base.get(seq)?.checked_add(i32::try_from(tok).ok()?)
    }

    /// The `[seqs]` per-seq base table to hand the device (`pos_base` itself —
    /// this validates the length, which is the mistake worth catching before
    /// the H2D: a short table silently reads a neighbour's base).
    pub fn pos_table(&self, pos_base: &[i32]) -> Option<Vec<i32>> {
        if pos_base.len() != self.seqs() {
            return None;
        }
        Some(pos_base.to_vec())
    }

    /// Rows `seq` contributes — the DEVICE advance step for its pinned
    /// `t0`/`total` (`t0 += dev_adv_step(seq)`), not a hard-coded `+1`.
    pub fn dev_adv_step(&self, seq: usize) -> Option<usize> {
        self.block_len(seq)
    }

    /// The `[rows][2]` interleaved `(seq, tok)` table the ragged kernel maps
    /// through. Also correct for padded (it is the same mapping), but padded
    /// passes `None` so the divisor path — and therefore the graph — is used.
    pub fn row_table(&self) -> Vec<i32> {
        let mut t = Vec::with_capacity(self.rows() * 2);
        for seq in 0..self.seqs() {
            for tok in 0..self.block_len[seq] {
                t.push(seq as i32);
                t.push(tok as i32);
            }
        }
        t
    }

    /// The launch arguments this map implies (verdict 5). `None` when
    /// `pos_base` does not carry exactly one base per seq.
    pub fn append_plan(&self, pos_base: &[i32]) -> Option<AppendPlan> {
        let (ntok, row_map) = if self.needs_row_table() {
            (0usize, Some(self.row_table()))
        } else {
            (self.n_v().unwrap_or(1), None)
        };
        let block_len = self
            .needs_row_table()
            .then(|| self.block_len.iter().map(|&b| b as i32).collect::<Vec<i32>>());
        Some(AppendPlan {
            rows: self.rows(),
            seqs: self.seqs(),
            ntok,
            row_map,
            block_len,
            pos_base: self.pos_table(pos_base)?,
        })
    }
}

/// The `ferrite_dsa_append_batched_mapped` launch arguments a [`SeqRowMap`]
/// derives. Both tables are OPTIONAL on purpose:
///
/// - **padded** (`ntok = n_v`, no tables) — the kernel derives
///   `seq = row / ntok, tok = row % ntok` and advances `t0` by `ntok`. This is
///   also exactly what today's `ferrite_dsa_append_batched` does at
///   `ntok = 1` (`rows == seqs`), i.e. the decode-batched launch is unchanged.
/// - **ragged** (`ntok = 0`) — the kernel reads `row_map[2*row]` /
///   `row_map[2*row+1]` and advances `t0` by `block_len[seq]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendPlan {
    /// Grid rows, and the kvb/ki/gate row dimension.
    pub rows: usize,
    /// The per-seq tables' width (the SEQ count, NOT the row count). Every
    /// per-seq kernel (kpool/indexer/expand) and pointer table keeps receiving
    /// this.
    pub seqs: usize,
    /// `> 0` → divisor mapping + advance step; `0` → the tables below.
    pub ntok: usize,
    /// `[rows][2]` interleaved `(seq, tok)` — ragged only.
    pub row_map: Option<Vec<i32>>,
    /// `[seqs]` rows per seq — ragged only (the advance step).
    pub block_len: Option<Vec<i32>>,
    /// `[seqs]` per-seq base the row's `tok` is offset from.
    pub pos_base: Vec<i32>,
}

impl AppendPlan {
    /// Elements in the device `(seq, tok)` table: `2 * rows`.
    pub fn row_map_elems(&self) -> usize {
        self.rows * 2
    }

    /// The device advance step for `seq` under this plan.
    pub fn dev_adv_step(&self, seq: usize) -> Option<usize> {
        match &self.block_len {
            Some(b) => b.get(seq).map(|&v| v as usize),
            None => Some(self.ntok),
        }
    }
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

// ---------------------------------------------------------------------------
// The per-step plan — the ONE struct the exec layer builds a batched MTP tick
// from (`TpCluster::mtp_step_batched`, the Wave 5 wiring)
// ---------------------------------------------------------------------------

/// `FERRITE_MTP_BATCHED`: the serve-layer gate for the batched-MTP sub-branch
/// (default OFF). Read once and cached — the house rule for hot-path gates.
/// Kept here (not in `gpu_engine`) so the serve layer and the exec layer read
/// the SAME switch: the gate decides whether `gpu_engine`'s `max_seqs` forcing
/// is lifted AND which step the cluster runs.
pub fn batched_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("FERRITE_MTP_BATCHED")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

/// Whether the batched-MTP **kernel half** is in place. `false` today: the
/// plan below is built and validated, but three device-side pieces are still
/// designs (the module docs' verdicts), so `mtp_step_batched` runs the per-seq
/// fallback AFTER building the plan:
///
/// 1. the `dsa_append_batched_mapped` CALL SITE at `ntok = n_v` — the kernel
///    and its Rust wrapper exist, but the MTP verify path does not call them
///    yet (`cuda.rs::dsa_append_batched` still pins `ntok = 1`, verdicts 4/5);
/// 2. the single-launch batched commit (verdict 3b) — `cuda.rs` exposes the
///    per-seq `mtp_commit(seq, k)` / `mtp_commit_dev(seq, k_dev)` only;
/// 3. the `(seqs, n_v)`-keyed B-row verify capture (verdict 1) — the per-seq
///    `mega_v{seq}` capture is what `mtp_step` uses today.
///
/// Flipping this const is the whole switch: `MtpBatchPlan` is the interface
/// those three consume, so no call site changes with it.
pub const MTP_BATCH_READY: bool = false;

/// The per-step plan a batched MTP tick is built from: the verify block's row
/// layout ([`SeqRowMap`] + [`AppendPlan`]), the graph keys and the commit
/// layout of ONE B-seq draft+verify pass. Every field is derived from the live
/// seq count and the per-seq position bases — nothing here touches the GPU, so
/// the plan is unit-testable (the tests below are the contract).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpBatchPlan {
    /// Verify width (`FERRITE_MTP_N`; drafts = `n_v - 1`).
    pub n_v: usize,
    /// Live seqs this tick.
    pub live: usize,
    /// The ladder rung the layout is padded to ([`SEQ_LADDER`]).
    pub padded: usize,
    /// The B×n_v verify block's row map.
    pub rows: SeqRowMap,
    /// The cache-append launch the verify's rows imply.
    pub append: AppendPlan,
    /// The batched commit's strides (`[B][n_gdn][6]`, `k[B]`).
    pub commit: CommitBatchLayout,
    /// The verify graph key — `mega_v_b{padded}_n{n_v}`, NOT the row count
    /// (verdict 1: the per-size pointer tables and the `n_v` divisor are baked
    /// into the capture).
    pub verify_graph: String,
}

impl MtpBatchPlan {
    /// Build the plan for `live` seqs whose own position bases are
    /// `live_bases` (one per live seq — GLM's DSA `t0`, the slot the verify's
    /// anchor row is appended at). `padded` = the ladder rung, `n_v` = the
    /// verify width, `hidden`/`n_plans` = the commit layout's dimensions
    /// (`n_plans` = the GDN layer count of `MtpState.commit`).
    ///
    /// `ragged_lens = Some(_)` selects the ragged layout (plan §3.2 — a
    /// non-graph path); `None` is the padded default (plan §3.3 "A 起步").
    ///
    /// `None` when `live_bases` does not cover the live set exactly: a short
    /// table would silently pair a seq's base with its neighbour's rows, which
    /// is the mistake worth catching BEFORE the H2D ([`SeqRowMap::pos_table`],
    /// hoisted so callers see it once). Pad slots of the padded layout carry
    /// base `0` — their rows land on the shared dummy table state and their
    /// outputs are discarded.
    pub fn build(
        live: usize,
        live_bases: &[i32],
        n_v: usize,
        hidden: usize,
        n_plans: usize,
        ragged_lens: Option<&[usize]>,
    ) -> Option<Self> {
        if live == 0 || live_bases.len() != live {
            return None;
        }
        let n_v = n_v.max(1);
        let rows = SeqRowMap::for_verify(live, n_v, ragged_lens);
        if rows.seqs() == 0 || rows.rows() == 0 {
            return None;
        }
        // Position bases are per SEQ SLOT, so the padded layout needs one
        // entry per rung slot (the pad slots included).
        let bases: Vec<i32> = match rows.layout() {
            RowLayout::Padded => {
                let mut b = vec![0i32; rows.seqs()];
                b[..live].copy_from_slice(live_bases);
                b
            }
            RowLayout::Ragged => live_bases.to_vec(),
        };
        let append = rows.append_plan(&bases)?;
        let commit = commit_batch_layout(rows.seqs(), n_plans, n_v, hidden);
        Some(Self {
            n_v,
            live,
            padded: padded_seqs(live),
            rows,
            append,
            commit,
            verify_graph: verify_graph_name(padded_seqs(live), n_v),
        })
    }

    /// The per-seq DSA advance steps the verify's rows imply: one entry per
    /// seq slot, `n_v` for the padded layout (the whole block appended in one
    /// pass) and each seq's own block length for the ragged layout. The seqs
    /// the kernel keeps counting (grid = `seqs`) advance by ONE while the
    /// per-row kernels see `rows` — the distinction [`AppendPlan::seqs`] vs
    /// [`AppendPlan::rows`] exists for.
    pub fn advance_steps(&self) -> Vec<usize> {
        (0..self.rows.seqs())
            .map(|s| self.append.dev_adv_step(s).unwrap_or(1))
            .collect()
    }

    /// The number of rows the verify graph must be asked for
    /// (`padded_seqs(live) * n_v` for the padded layout): the `small_n_rows`
    /// GEMV path is only valid at `n <= 16` (B=16, n_v=3 → 48 rows is the
    /// tiled-GEMM domain, module docs verdict 1).
    pub fn verify_rows(&self) -> usize {
        self.rows.rows()
    }

    /// One-line report (the wiring's observability; `FERRITE_MTP_BATCH_TIMING`).
    pub fn summary(&self) -> String {
        format!(
            "live={} padded={} n_v={} layout={:?} rows={} ntok={} seqs={} verify={} \
             commit=[{}x{}x{} k[{}] hf_v={} hprev={}]",
            self.live,
            self.padded,
            self.n_v,
            self.rows.layout(),
            self.append.rows,
            self.append.ntok,
            self.append.seqs,
            self.verify_graph,
            self.commit.seqs,
            self.commit.n_plans,
            self.commit.plan_row_elems,
            self.commit.k_len,
            self.commit.hf_v_elems(),
            self.commit.hprev_elems(),
        )
    }
}

/// The draft chain's graph names for this tick, seq-major: `n_v - 1` names per
/// seq, `mega_d{seq}_{i}` while [`DRAFT_CHAIN_SHARED`] is false (verdict 2 —
/// the Step-B decision), `mega_d_b{padded}_{i}` once the B-row + pointer-table
/// conversion lands. `n_v = 1` (no drafts) yields an empty vector.
pub fn draft_graphs(seqs: &[u64], n_v: usize) -> Vec<String> {
    let nd = n_v.saturating_sub(1);
    let padded = padded_seqs(seqs.len());
    let mut out = Vec::with_capacity(seqs.len() * nd);
    for &seq in seqs {
        for i in 0..nd {
            out.push(if DRAFT_CHAIN_SHARED {
                draft_graph_name(padded, i)
            } else {
                format!("mega_d{seq}_{i}")
            });
        }
    }
    out
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

    /// The identity `(seq, tok)` table a `rows == seqs, ntok == 1` launch has
    /// — i.e. today's decode-batched mapping.
    fn identity_table(seqs: usize) -> Vec<i32> {
        (0..seqs).flat_map(|s| [s as i32, 0]).collect()
    }

    #[test]
    fn padded_map_is_the_row_divide() {
        // B=4 padded seqs of n_v=3 rows: row r -> (r/3, r%3), the Step-B form.
        let m = SeqRowMap::padded(4, 3);
        assert_eq!(m.layout(), RowLayout::Padded);
        assert_eq!((m.seqs(), m.rows()), (4, 12));
        assert_eq!(m.n_v(), Some(3));
        assert_eq!(m.at(0), Some((0, 0)), "the anchor row of seq 0");
        assert_eq!(m.at(2), Some((0, 2)));
        assert_eq!(m.at(3), Some((1, 0)), "seq 1's anchor");
        assert_eq!(m.at(5), Some((1, 2)));
        assert_eq!(m.at(11), Some((3, 2)));
        assert_eq!(m.at(12), None, "past the block");
        assert_eq!(m.block_len(3), Some(3));
        assert_eq!(m.block_len(4), None);
        assert!(!m.needs_row_table(), "the divisor carries the padded map");
        // agrees with the standalone arithmetic the graph key is sized from
        for r in 0..m.rows() {
            assert_eq!(m.seq_of_row(r), Some(seq_of_row(r, 3)));
            assert_eq!(m.tok_of_row(r), Some(tok_of_row(r, 3)));
        }
    }

    #[test]
    fn padded_map_follows_the_ladder_rungs() {
        // (live, n_v) -> (padded seqs, rows) exactly as the graphs are sized.
        for (live, n_v) in [(0usize, 3usize), (1, 3), (2, 3), (3, 3), (16, 3), (16, 5), (33, 5)] {
            let m = SeqRowMap::padded_for_live(live, n_v);
            assert_eq!(m.seqs(), padded_seqs(live));
            assert_eq!(m.rows(), verify_rows(live, n_v));
            assert_eq!(m.n_v(), Some(n_v));
            // every (seq, tok) slot is covered exactly once
            let rows = m.rows();
            let mut seen = vec![0usize; m.seqs() * n_v];
            for r in 0..rows {
                let (s, t) = m.at(r).expect("row in range");
                seen[s * n_v + t] += 1;
            }
            assert!(seen.iter().all(|&c| c == 1), "live={live} n_v={n_v}");
            // the device table (if it were used) is the same mapping
            assert_eq!(m.row_table().len(), rows * 2);
        }
    }

    #[test]
    fn ragged_map_is_the_block_prefix_sum() {
        // the tail round of B=3: seq0 3 rows, seq1 1 row, seq2 5 rows.
        let m = SeqRowMap::ragged(&[3, 1, 5]);
        assert_eq!(m.layout(), RowLayout::Ragged);
        assert_eq!((m.seqs(), m.rows()), (3, 9));
        assert_eq!(m.n_v(), None, "no uniform block length");
        assert_eq!(m.block_len(1), Some(1));
        assert_eq!(m.at(0), Some((0, 0)));
        assert_eq!(m.at(2), Some((0, 2)));
        assert_eq!(m.at(3), Some((1, 0)), "seq 1 owns exactly one row");
        assert_eq!(m.at(4), Some((2, 0)), "seq 2 starts right after seq 1");
        assert_eq!(m.at(8), Some((2, 4)));
        assert_eq!(m.at(9), None);
        assert!(m.needs_row_table());
        assert_eq!(
            m.row_table(),
            vec![0, 0, 0, 1, 0, 2, 1, 0, 2, 0, 2, 1, 2, 2, 2, 3, 2, 4]
        );
    }

    #[test]
    fn ragged_map_skips_a_seq_with_no_rows() {
        // A block that was entirely consumed owns no row: the partition must
        // skip it, not hand its neighbour's rows to a phantom seq.
        let m = SeqRowMap::ragged(&[2, 0, 1]);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.at(1), Some((0, 1)));
        assert_eq!(m.at(2), Some((2, 0)), "seq 1 is skipped, seq 2 answers");
        assert_eq!(m.row_table(), vec![0, 0, 0, 1, 2, 0]);
        assert_eq!(SeqRowMap::ragged(&[]).rows(), 0, "nothing to append");
    }

    /// The alias the `(seqs, n_v)` graph key exists to prevent (verdict 1):
    /// 24 rows both ways, different seqs for the same row.
    #[test]
    fn ragged_rows_are_not_the_padded_divisor() {
        let padded = SeqRowMap::padded(8, 3); // 24 rows under 8 seqs
        let ragged = SeqRowMap::ragged(&[6, 6, 6, 6]); // 24 rows under 4 seqs
        assert_eq!(padded.rows(), ragged.rows());
        assert_ne!(padded.seqs(), ragged.seqs());
        assert_ne!(padded.seq_of_row(6), ragged.seq_of_row(6));
        assert_eq!(padded.seq_of_row(6), Some(2));
        assert_eq!(ragged.seq_of_row(6), Some(1));
        assert_ne!(padded.row_table(), ragged.row_table());
    }

    #[test]
    fn append_plan_padded_needs_no_device_table() {
        let m = SeqRowMap::padded(4, 3);
        let pos = [100, 200, 300, 400];
        let p = m.append_plan(&pos).expect("one base per seq");
        assert_eq!((p.rows, p.seqs, p.ntok), (12, 4, 3));
        assert!(p.row_map.is_none(), "padded: the ntok divisor carries it");
        assert!(p.block_len.is_none());
        assert_eq!(p.pos_base, vec![100, 200, 300, 400]);
        assert_eq!(p.dev_adv_step(1), Some(3), "advance the whole block, not +1");
        assert_eq!(m.pos_of_row(5, &pos), Some(202), "row 5 = seq 1, tok 2");
        assert_eq!(m.dev_adv_step(1), Some(3));
    }

    /// BACKWARD COMPAT — the launch the code does TODAY (decode-batched:
    /// `rows == seqs`, `ntok = 1`, no tables) must come out of the map
    /// unchanged: that is the invariant the mapped kernel is written against.
    #[test]
    fn append_plan_reproduces_todays_decode_batched_launch() {
        let live = 16;
        let m = SeqRowMap::for_verify(live, 1, None);
        let pos: Vec<i32> = (0..m.seqs() as i32).collect();
        let p = m.append_plan(&pos).expect("plan");
        assert_eq!(p.rows, live, "n == ni when ntok == 1");
        assert_eq!(p.seqs, live);
        assert_eq!(p.ntok, 1);
        assert!(p.row_map.is_none() && p.block_len.is_none());
        assert_eq!(p.pos_base, pos);
        assert_eq!(p.row_map_elems(), 2 * live);
        assert_eq!(m.row_table(), identity_table(live));
        assert_eq!(m.dev_adv_step(7), Some(1), "tok == ntok-1 == 0 advances");
    }

    #[test]
    fn append_plan_ragged_carries_both_tables() {
        let m = SeqRowMap::ragged(&[3, 2]);
        let p = m.append_plan(&[10, 20]).expect("plan");
        assert_eq!((p.rows, p.seqs, p.ntok), (5, 2, 0), "ntok 0 = table mode");
        assert_eq!(
            p.row_map.as_deref(),
            Some(&[0, 0, 0, 1, 0, 2, 1, 0, 1, 1][..])
        );
        assert_eq!(p.block_len.as_deref(), Some(&[3, 2][..]));
        assert_eq!(p.dev_adv_step(0), Some(3));
        assert_eq!(p.dev_adv_step(1), Some(2), "each seq advances by its own block");
        assert_eq!(p.row_map_elems(), 10, "2 i32 per row");
        assert_eq!(m.pos_of_row(4, &[10, 20]), Some(21), "row 4 = seq 1, tok 1");
    }

    #[test]
    fn append_plan_rejects_a_short_base_table() {
        let m = SeqRowMap::padded(4, 3);
        assert!(m.pos_table(&[0, 1, 2, 3]).is_some());
        assert!(m.pos_table(&[0, 1, 2]).is_none(), "one base per seq, not per row");
        assert!(m.append_plan(&[0, 1, 2]).is_none());
        assert_eq!(m.dev_adv_step(9), None, "no such seq");
    }

    // ---- the per-step plan (`TpCluster::mtp_step_batched`'s input) ---------

    #[test]
    fn plan_pads_the_bases_to_the_rung_and_keys_the_graph() {
        // 3 live seqs, n_v=3 -> the 4-seq rung, 12 verify rows, ntok=n_v.
        let p = MtpBatchPlan::build(3, &[10, 20, 30], 3, 5120, 38, None).expect("plan");
        assert_eq!((p.live, p.padded, p.n_v), (3, 4, 3));
        assert_eq!(p.verify_rows(), 12);
        assert_eq!(p.verify_graph, "mega_v_b4_n3");
        assert_eq!(p.rows.seqs(), 4, "the pad slot is a real table column");
        assert_eq!((p.append.rows, p.append.seqs, p.append.ntok), (12, 4, 3));
        // the real seqs keep their bases; the pad slot is 0 (its rows land on
        // the shared dummy state) and is NOT a shift of its neighbour's base.
        assert_eq!(p.append.pos_base, vec![10, 20, 30, 0]);
        assert!(p.append.row_map.is_none(), "padded: the ntok divisor carries it");
        assert_eq!(p.rows.at(3), Some((1, 0)), "seq 1's anchor row");
        assert_eq!(p.rows.pos_of_row(5, &p.append.pos_base), Some(22), "seq 1, tok 2");
        assert_eq!(p.advance_steps(), vec![3, 3, 3, 3], "the whole block, not +1");
    }

    #[test]
    fn plan_commit_layout_is_sized_off_the_rung() {
        let p = MtpBatchPlan::build(2, &[0, 100], 3, 4096, 38, None).expect("plan");
        assert_eq!(p.commit.seqs, 2, "n=2 pads to 2");
        assert_eq!(p.commit.plan_elems(), 2 * 38 * 12);
        assert_eq!(p.commit.hf_v_elems(), 2 * 3 * 4096);
        assert_eq!(p.commit.hprev_elems(), 2 * 4096);
        assert_eq!(p.commit.k_len, 2, "one k per seq — pad rows must carry 1");
        // 3 live seqs pad to 4: the commit array is widened with the rung too,
        // so the pad slots' writes land on their own dummy states.
        let q = MtpBatchPlan::build(3, &[0, 1, 2], 3, 4096, 38, None).expect("plan");
        assert_eq!((q.commit.seqs, q.live), (4, 3));
    }

    #[test]
    fn plan_rejects_a_base_table_that_misses_the_live_set() {
        assert!(MtpBatchPlan::build(3, &[10, 20], 3, 5120, 38, None).is_none(), "short");
        assert!(MtpBatchPlan::build(3, &[10, 20, 30, 40], 3, 5120, 38, None).is_none(), "long");
        assert!(MtpBatchPlan::build(0, &[], 3, 5120, 38, None).is_none(), "empty tick");
    }

    #[test]
    fn plan_ragged_carries_the_block_lengths_and_skips_the_rung() {
        // A ragged round: seq0's whole block was consumed (1 row), seq1 kept 3.
        let p = MtpBatchPlan::build(2, &[10, 20], 3, 5120, 38, Some(&[1, 3])).expect("plan");
        assert_eq!(p.rows.layout(), RowLayout::Ragged);
        assert_eq!((p.append.rows, p.append.seqs, p.append.ntok), (4, 2, 0));
        assert_eq!(p.append.row_map.as_deref(), Some(&[0, 0, 1, 0, 1, 1, 1, 2][..]));
        assert_eq!(p.append.block_len.as_deref(), Some(&[1, 3][..]));
        assert_eq!(p.advance_steps(), vec![1, 3], "each seq advances by its own block");
        // The graph key is still the (padded, n_v) pair — a ragged pass is a
        // non-graph path, but the naming must not collide with the padded one.
        assert_eq!(p.verify_graph, "mega_v_b2_n3");
    }

    #[test]
    fn plan_n_v_one_is_the_undrafted_decode_shape() {
        // n_v=1 (FERRITE_MTP_N=1): one row per seq, ntok=1 — the shape the
        // decode-batched append already launches.
        let p = MtpBatchPlan::build(4, &[0, 1, 2, 3], 1, 5120, 38, None).expect("plan");
        assert_eq!((p.verify_rows(), p.append.ntok), (4, 1));
        assert_eq!(p.append.seqs, 4);
        assert!(draft_graphs(&[7, 8, 9, 10], 1).is_empty(), "no drafts");
    }

    #[test]
    fn draft_graph_names_are_per_seq_and_seq_major() {
        assert!(!DRAFT_CHAIN_SHARED, "Step B keeps the per-seq draft chain");
        assert_eq!(
            draft_graphs(&[7, 9], 3),
            vec!["mega_d7_0", "mega_d7_1", "mega_d9_0", "mega_d9_1"]
        );
        assert_eq!(draft_graphs(&[], 3), Vec::<String>::new());
    }

    #[test]
    fn the_batched_gate_keeps_the_plan_consistent_with_the_kernel_switch() {
        // The wiring's invariant: while the kernel half is missing, the plan
        // is still built (so the data flow is exercised) and the per-seq
        // fallback runs. Flipping MTP_BATCH_READY must not change the plan.
        assert!(!MTP_BATCH_READY, "the three device-side pieces are still designs");
        let p = MtpBatchPlan::build(2, &[5, 6], 3, 5120, 38, None).expect("plan");
        assert!(p.summary().contains("verify=mega_v_b2_n3"), "{}", p.summary());
        assert!(p.summary().contains("rows=6"));
    }
}
