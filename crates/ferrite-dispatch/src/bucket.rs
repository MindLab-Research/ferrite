//! Pad-to-B static bucket scheduler — the graph-shape contract.
//!
//! ## Why static bucket shapes
//!
//! CUDA graphs (the ferrite mega-graph decode chain) bake batch size into
//! every captured kernel launch and every pointer. Replaying shape `B` with
//! `n` live rows (`n ≤ B`) wastes `B - n` rows of compute, but replays at
//! full speed; capturing arbitrary shapes stalls the pipeline for tens of
//! ms per shape. The static ladder `{1, 2, 4, 8, 16, 32}` bounds padding
//! waste at **2× amortized** (worst case `n = B/2 + 1` → pad to `2n - 2`)
//! while keeping a closed set of pre-captured graphs resident on device —
//! the same trade SGLang/TRT-LLM make with token-bucket padding, applied
//! here to *rows* (sequence slots), which is the only axis a
//! single-token-per-row MTP decode graph has.
//!
//! ## Row identity is physical
//!
//! Bucket row `r` **is** decode-row slot `r` of the state registry: the
//! captured graph's state tensors are laid out `[B, ...]` and row `i`
//! addresses slot `i` directly. Bucket assignment therefore only picks a
//! **shape** — it may not reshuffle live rows. The scheduler admits
//! sequences into the lowest free row; the bucket for a tick is
//! `min(bucket_ladder ≥ live_rows)`; rows above the live count are padded.
//! Padding rows carry dummy token ids (`PAD_TOKEN`) and write state into
//! their own (allocated) slot — no cross-row contamination, no masking
//! needed downstream because state is row-strided.
//!
//! ## MTP multiplies the verify row count
//!
//! With draft depth 2 the verify graph processes `3B` token positions
//! (each row verifies 3 candidates). That is a property of the captured
//! graph, not the scheduler: the bucket stays shape-B; the verify launch
//! reads its padded input from `[B][3]` ids (see `mtp.rs`).

use ferrite_types::{FerriteError, Result};

/// The bucket ladder. Order matters: ascending, powers of two.
pub const BUCKET_LADDER: [u32; 6] = [1, 2, 4, 8, 16, 32];

/// Padding token id fed to dummy rows (id 0 = `<|begin_of_text|>`-alike
/// placeholder; rows are isolated by state stride, the value is inert).
pub const PAD_TOKEN: u32 = 0;

/// Compile-time batch dimension — the const-generic hook the graph pool
/// specializes on. Each ladder shape implements it; the exec backend
/// captures one mega-graph per `BatchDim` instantiation (see `graph.rs`).
pub trait BatchDim {
    /// Rows in one captured graph of this shape (also the verify input's
    /// outer dim before the ×3 MTP expansion).
    const B: u32;
    /// Ladder position (index into `BUCKET_LADDER`), used for pool slots.
    const LADDER_IDX: usize;
}

/// The ladder, const-dispatched. `impl_batch_dim!` also builds the
/// runtime type registry the pool iterates — one macro keeps the ladder
/// and its impls from drifting apart.
macro_rules! impl_batch_dim {
    ($($name:ident => ($b:expr, $idx:expr)),+ $(,)?) => {
        $(
            #[derive(Debug, Clone, Copy, PartialEq, Eq)]
            pub struct $name;
            impl BatchDim for $name {
                const B: u32 = $b;
                const LADDER_IDX: usize = $idx;
            }
        )+
    };
}

impl_batch_dim! {
    B1  => (1, 0),
    B2  => (2, 1),
    B4  => (4, 2),
    B8  => (8, 3),
    B16 => (16, 4),
    B32 => (32, 5),
}

/// Runtime bucket shape (the discriminator the scheduler picks each tick).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Bucket {
    B1,
    B2,
    B4,
    B8,
    B16,
    B32,
}

impl Bucket {
    /// All shapes, ascending — iteration order for pool warm-up.
    pub const ALL: [Bucket; 6] =
        [Bucket::B1, Bucket::B2, Bucket::B4, Bucket::B8, Bucket::B16, Bucket::B32];

    pub fn rows(self) -> u32 {
        match self {
            Bucket::B1 => 1,
            Bucket::B2 => 2,
            Bucket::B4 => 4,
            Bucket::B8 => 8,
            Bucket::B16 => 16,
            Bucket::B32 => 32,
        }
    }

    /// Smallest bucket with `rows >= n` — the tick's replay shape.
    pub fn cover(n: u32) -> Result<Bucket> {
        BUCKET_LADDER
            .iter()
            .find(|&&b| b >= n)
            .map(|&b| b.try_into())
            .transpose()?
            .ok_or_else(|| FerriteError::Scheduler(format!("bucket: {n} exceeds ladder max {}", BUCKET_LADDER[BUCKET_LADDER.len() - 1])))
    }

    /// Ladder index (graph-pool slot).
    pub fn ladder_idx(self) -> usize {
        match self {
            Bucket::B1 => 0,
            Bucket::B2 => 1,
            Bucket::B4 => 2,
            Bucket::B8 => 3,
            Bucket::B16 => 4,
            Bucket::B32 => 5,
        }
    }

    /// Padding waste for `n` live rows: `rows - n`.
    pub fn pad(self, n: u32) -> u32 {
        self.rows().saturating_sub(n)
    }
}

impl From<Bucket> for u32 {
    fn from(b: Bucket) -> u32 {
        b.rows()
    }
}

impl TryFrom<u32> for Bucket {
    type Error = FerriteError;
    fn try_from(v: u32) -> Result<Bucket> {
        match v {
            1 => Ok(Bucket::B1),
            2 => Ok(Bucket::B2),
            4 => Ok(Bucket::B4),
            8 => Ok(Bucket::B8),
            16 => Ok(Bucket::B16),
            32 => Ok(Bucket::B32),
            _ => Err(FerriteError::Scheduler(format!("bucket: {v} not a ladder shape"))),
        }
    }
}

/// One tick's decode assignment — the *only* data the replay path needs.
///
/// Zero-copy by design: `rows` are registry decode-row indices (dense
/// prefix `[0, live)`), `seqs` parallel array of sequence ids in the same
/// order (slot `rows[i]` belongs to `seqs[i]`). The exec backend packs
/// token ids into the graph's pinned `[B]`/`[3B]` input slots from this.
#[derive(Debug, Clone)]
pub struct BucketAssignment {
    pub shape: Bucket,
    /// Live rows in row order (ascending; row = physical state slot).
    pub rows: Vec<u32>,
    /// Sequence handle per live row (parallel to `rows`).
    pub seqs: Vec<crate::arena::SeqId>,
    /// Padding rows the replay must fill with `PAD_TOKEN`.
    pub pad_rows: u32,
}

impl BucketAssignment {
    pub fn live(&self) -> u32 {
        self.rows.len() as u32
    }

    /// Build the padded per-row input ids for a plain decode tick
    /// (one token per row). `ids[i]` = token of row i, `PAD_TOKEN` padding.
    pub fn padded_ids(&self, token_of: impl Fn(crate::arena::SeqId) -> u32) -> Vec<u32> {
        let b = self.shape.rows() as usize;
        let mut ids = vec![PAD_TOKEN; b];
        for (i, &seq) in self.seqs.iter().enumerate() {
            ids[i] = token_of(seq);
        }
        ids
    }
}
