//! Parallel-shard axis algebra — the compile-time model for *composable*
//! parallelism across GLM-5.3-Flash's heterogeneous modules.
//!
//! The three composable attention/data axes, each with its own merge math:
//!
//! | Axis | Meaning | Merge along the axis |
//! |------|---------|----------------------|
//! | `Tp`  | head / inter / expert dim (weights slice) | **Sum** (o_proj / down_proj partial sums) or **Concat** (attention head outputs — heads are an output dim, never summed) |
//! | `Kv`  | KV page dim — the *context* axis | **LSE** (softmax denominator factorises over the page partition) |
//! | `Q`   | query token segment — the *prefill sequence* axis | **RowGather** (output rows are disjoint — no compute, only data placement) or **Pipeline** (GDN chunk-state chain) |
//!
//! ## The phase asymmetry (the "reverse" sharding)
//!
//! **Prefill CP shards Q; decode CP shards KV.** The same attention module
//! runs opposite-direction parallelism per phase:
//!
//! - **Prefill (CP)**: each rank owns a *query token segment*. The segment's
//!   output rows are independent (each query attends over the whole KV) —
//!   partial(Q_seg, KV_page) blocks merge along the KV axis (LSE) and
//!   *concatenate* along Q (rows disjoint). The data-plane cost is moving
//!   KV pages to (or ring-rotating them past) each Q segment — no output
//!   reduction communication.
//! - **Decode (DCP)**: Q is a single token (nothing to shard). Each rank
//!   owns a *KV page shard*; partial(1×page) blocks merge along KV (LSE).
//! - **DSA indexer** follows the KV axis (k_idx is page-resident,
//!   head-agnostic): page-local scores → all-gather → one global top-k
//!   (deterministic, EAGLE-safe).
//! - **GatedDeltaNet (linear attention)**: the recurrent state couples the
//!   token axis — prefill segments form a **chain** (chunk i's state out is
//!   chunk i+1's state in; the WYF chunkwise structure), i.e. a pipeline,
//!   *not* independent partials. Decode has no page axis (fixed state) —
//!   heads only.
//!
//! ## The canonical 3D composition: DSA attention (q_seg × kv_page × head)
//!
//! Every rank computes partial(Q_i, P_j, H_k). Merges:
//! 1. along KV (per Q row, per head): LSE — commutative, stable
//! 2. along head: concat (attn output) then o_proj column-slice → sum
//! 3. along Q: row gather (no compute)
//!
//! The KV and head merges **commute** (LSE is per-head; concat/sum is
//! linear) — axis order is a scheduling choice. The Q axis has no merge at
//! all. This is what makes arbitrary `(Cp × Dcp × Tp)` topologies valid:
//! every composition reduces to the same three merge kinds.
//!
//! ## Module × axis matrix (GLM-5.3-Flash, per phase)
//!
//! | Module (phase) | Tp (head) | Kv (page) | Q (token seg) | Merge |
//! |---|---|---|---|---|
//! | DSA attn (prefill) | head slice | page shard or full (data-plane ring) | **Q segment** | KV: LSE; head: concat→sum; Q: row gather |
//! | DSA attn (decode) | head slice | **page shard** | — (single token) | KV: LSE; head: concat→sum |
//! | DSA indexer | replicated (global score) | page shard (k_idx resident) | — | score all-gather → global top-k |
//! | DSA latent KV | replicated (MLA, head-agnostic) | page shard | — | none (disjoint pages) |
//! | GDN (prefill) | head slice | — (no page axis) | **token chain** | state pipeline (WYF chunks); head: sum via o_proj |
//! | GDN (decode) | head slice | — (state replicated) | — | none inside; head: sum |
//! | dense MLP / MoE | inter/expert slice | — | token dispatch (MoE) | Tp: sum |
//! | MHC / norms / embed / lm_head | replicated | replicated | replicated | none |
//!
//! ## Relation to the PDAF deployment topology
//!
//! `ShardLayout`'s layer-split (`cp_layer_range`) is a *prefill rank
//! assignment* (which layers a P-rank owns) — orthogonal to these axes;
//! a deployment instantiates ranks at (q, kv, head) coordinates *within*
//! its layer range. The P→D 2D reshard moves KV from P-rank page layouts
//! to D-rank page layouts — the axes here stay valid on both sides.

use ferrite_model::Glm53FlashConfig;

/// The three composable parallel axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
    /// Head / inter / expert dim. Weights slice; attention outputs concat,
    /// projections partial-sum.
    Head,
    /// KV page dim — the context axis. Merge: stable LSE.
    Kv,
    /// Query token segment — the prefill sequence axis. Merge: row gather
    /// (DSA) or state pipeline (GDN chunk chain).
    Q,
}

/// Deployment mesh: sizes along each axis. All >= 1 (1 = unsharded).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardAxes {
    pub tp: usize,
    pub kv: usize,
    pub q: usize,
}

/// A rank's position in the mesh: `(head, kv, q)` coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RankCoord {
    pub tp: usize,
    pub kv: usize,
    pub q: usize,
}

impl ShardAxes {
    pub fn new(tp: usize, kv: usize, q: usize) -> Self {
        assert!(tp >= 1 && kv >= 1 && q >= 1, "axis sizes must be >= 1 (1 = unsharded)");
        ShardAxes { tp, kv, q }
    }

    /// Total world size (product over axes).
    pub fn world(&self) -> usize {
        self.tp * self.kv * self.q
    }

    /// Linear rank id (row-major over `(tp, kv, q)`) → coordinates.
    pub fn coord(&self, linear: usize) -> RankCoord {
        assert!(linear < self.world(), "rank {linear} out of mesh {self:?}");
        let tp = linear / (self.kv * self.q);
        let rest = linear % (self.kv * self.q);
        RankCoord { tp, kv: rest / self.q, q: rest % self.q }
    }

    /// Coordinates → linear rank id.
    pub fn linear(&self, c: RankCoord) -> usize {
        c.tp * self.kv * self.q + c.kv * self.q + c.q
    }

    /// The communication group for a merge along `axis`, holding every other
    /// axis fixed at `at`'s coordinates. E.g. the KV LSE-merge group of
    /// `(tp=1, kv=?, q=0)` is all ranks with tp=1, q=0.
    pub fn group_along(&self, axis: Axis, at: RankCoord) -> Vec<RankCoord> {
        let n = self.size(axis);
        (0..n)
            .map(|i| {
                let mut c = at;
                match axis {
                    Axis::Head => c.tp = i,
                    Axis::Kv => c.kv = i,
                    Axis::Q => c.q = i,
                }
                c
            })
            .collect()
    }

    pub fn size(&self, axis: Axis) -> usize {
        match axis {
            Axis::Head => self.tp,
            Axis::Kv => self.kv,
            Axis::Q => self.q,
        }
    }
}

/// How results merge back along an axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeKind {
    /// No communication (replicated / disjoint output).
    None,
    /// Partial sums add (column-sliced o_proj / down_proj outputs).
    Sum,
    /// Softmax denominator factorises: stable log-sum-exp merge over the
    /// axis (KV pages). Deterministic → EAGLE-safe.
    Lse,
    /// Head outputs concatenate (attention out dim is not summed).
    Concat,
    /// Output rows are disjoint along the axis (Q segments): place rows,
    /// no compute.
    RowGather,
    /// Recurrent state chains along the axis (GDN prefill chunks): chunk i's
    /// state out feeds chunk i+1 — pipeline order, not a merge.
    Pipeline,
}

/// Which axis (or axes) a tensor/computation shards along, per phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorAxis {
    /// Shards along Tp only (head/inter/expert slice).
    Head,
    /// Shards along the KV axis only (pages; head-agnostic → replicated
    /// over Tp). DSA latent KV, indexer k_idx.
    KvPages,
    /// Shards along BOTH head and KV: the DSA attention partials —
    /// `(head_subset × page_subset)` per rank.
    HeadAndKvPages,
    /// Shards along Q (prefill token segments; GDN: chained).
    QSegments,
    /// Replicated on every rank of the relevant axis group.
    Replicated,
}

/// Per-module, per-phase parallel plan.
#[derive(Debug, Clone, Copy)]
pub struct ModuleParallel {
    /// What this module's main tensors shard along (this phase).
    pub axis: TensorAxis,
    /// Merge along the head (Tp) axis.
    pub head_merge: MergeKind,
    /// Merge along the KV (page) axis.
    pub kv_merge: MergeKind,
    /// Merge/flow along the Q (token segment) axis.
    pub q_merge: MergeKind,
}

/// The phase a plan applies to (the same module shards differently).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Prefill,
    Decode,
}

/// The GLM-5.3-Flash module × phase × axis table (compile-time, per module
/// kind). Single source of truth: `shard_weights_tp` slices per the head
/// column, cluster drivers place collectives per the merge kinds, and the
/// PDAF reshard routes per the KV column.
pub fn module_parallel(kind: ModuleKind, phase: Phase) -> ModuleParallel {
    match (kind, phase) {
        // DSA attention: prefill shards Q (row-independent) with KV as the
        // data plane; decode shards KV (single token). Head axis always
        // available on both. KV and head merges commute.
        (ModuleKind::DsaAttn, Phase::Prefill) => ModuleParallel {
            axis: TensorAxis::QSegments,
            head_merge: MergeKind::Concat, // attn out; o_proj sums after
            kv_merge: MergeKind::Lse,
            q_merge: MergeKind::RowGather,
        },
        (ModuleKind::DsaAttn, Phase::Decode) => ModuleParallel {
            axis: TensorAxis::HeadAndKvPages,
            head_merge: MergeKind::Concat,
            kv_merge: MergeKind::Lse,
            q_merge: MergeKind::None, // single token
        },
        // DSA indexer: k_idx page-resident, head-agnostic global score.
        // Page-local scores → all-gather → one global top-k.
        (ModuleKind::DsaIndexer, _) => ModuleParallel {
            axis: TensorAxis::KvPages,
            head_merge: MergeKind::None,
            kv_merge: MergeKind::Concat, // score gather, then global top-k
            q_merge: MergeKind::None,
        },
        // DSA latent KV: head-agnostic (MLA) → replicated over Tp,
        // page-sliced over KV (the 2D-reshard payload).
        (ModuleKind::DsaLatentKv, _) => ModuleParallel {
            axis: TensorAxis::KvPages,
            head_merge: MergeKind::None,
            kv_merge: MergeKind::None, // pages disjoint; placement only
            q_merge: MergeKind::None,
        },
        // GatedDeltaNet: heads slice (independent recurrence). Prefill
        // token segments form a chunk chain (WYF): state pipeline, NOT
        // independent partials. Decode: no page axis, state replicated.
        (ModuleKind::GdnAttn, Phase::Prefill) => ModuleParallel {
            axis: TensorAxis::QSegments,
            head_merge: MergeKind::Sum, // via column-sliced o_proj
            kv_merge: MergeKind::None,  // no page axis (recurrent state)
            q_merge: MergeKind::Pipeline, // chunk state chain
        },
        (ModuleKind::GdnAttn, Phase::Decode) => ModuleParallel {
            axis: TensorAxis::Head,
            head_merge: MergeKind::Sum,
            kv_merge: MergeKind::None,
            q_merge: MergeKind::None,
        },
        // MLPs: inter/expert slice along head(Tp) axis, all-reduce sum.
        (ModuleKind::DenseMlp, _) | (ModuleKind::Moe, _) => ModuleParallel {
            axis: TensorAxis::Head,
            head_merge: MergeKind::Sum,
            kv_merge: MergeKind::None,
            q_merge: MergeKind::None, // tokens independent (dispatch handled by router)
        },
        // MHC / norms / embedding / lm_head: replicated.
        (ModuleKind::Replicated, _) => ModuleParallel {
            axis: TensorAxis::Replicated,
            head_merge: MergeKind::None,
            kv_merge: MergeKind::None,
            q_merge: MergeKind::None,
        },
    }
}

/// Module kinds relevant to parallel placement (mirrors the layer plan).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleKind {
    GdnAttn,
    DsaIndexer,
    DsaAttn,
    DsaLatentKv,
    DenseMlp,
    Moe,
    Replicated,
}

/// Layer → module kinds (per layer of the config; the layer's modules all
/// appear here so a driver can wire collectives from one lookup).
pub fn layer_modules(cfg: &Glm53FlashConfig, layer: usize) -> Vec<ModuleKind> {
    let mut v = if cfg.is_dsa_layer(layer) {
        vec![
            ModuleKind::DsaLatentKv,
            ModuleKind::DsaIndexer,
            ModuleKind::DsaAttn,
        ]
    } else {
        vec![ModuleKind::GdnAttn]
    };
    let sparse = cfg
        .mlp_types
        .get(layer)
        .map(|t| *t == ferrite_model::MlpType::Sparse)
        .unwrap_or(false);
    v.push(if sparse { ModuleKind::Moe } else { ModuleKind::DenseMlp });
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_coord_roundtrip() {
        let axes = ShardAxes::new(2, 3, 4); // 24 ranks
        assert_eq!(axes.world(), 24);
        for r in 0..24 {
            let c = axes.coord(r);
            assert_eq!(axes.linear(c), r);
            assert!(c.tp < 2 && c.kv < 3 && c.q < 4);
        }
    }

    #[test]
    fn groups_are_axis_submeshes() {
        let axes = ShardAxes::new(2, 2, 2);
        let at = RankCoord { tp: 1, kv: 0, q: 1 };
        // KV merge group: same tp and q, varying kv
        let g = axes.group_along(Axis::Kv, at);
        assert_eq!(g.len(), 2);
        assert!(g.contains(&RankCoord { tp: 1, kv: 0, q: 1 }));
        assert!(g.contains(&RankCoord { tp: 1, kv: 1, q: 1 }));
        assert!(!g.contains(&RankCoord { tp: 0, kv: 0, q: 1 })); // other tp
        assert!(!g.contains(&RankCoord { tp: 1, kv: 0, q: 0 })); // other q
        // Head merge group: same kv and q, varying tp
        let t = axes.group_along(Axis::Head, at);
        assert_eq!(t.len(), 2);
        assert!(t.contains(&RankCoord { tp: 0, kv: 0, q: 1 }));
        // Q segment group: same tp and kv, varying q
        let q = axes.group_along(Axis::Q, at);
        assert_eq!(q.len(), 2);
        assert!(q.contains(&RankCoord { tp: 1, kv: 0, q: 0 }));
    }

    /// The phase asymmetry: prefill shards Q (row-gather, KV is data plane),
    /// decode shards KV (LSE merge) — the same module, opposite direction.
    #[test]
    fn dsa_phase_asymmetry() {
        let pre = module_parallel(ModuleKind::DsaAttn, Phase::Prefill);
        assert_eq!(pre.axis, TensorAxis::QSegments);
        assert_eq!(pre.q_merge, MergeKind::RowGather);
        assert_eq!(pre.kv_merge, MergeKind::Lse);
        let dec = module_parallel(ModuleKind::DsaAttn, Phase::Decode);
        assert_eq!(dec.axis, TensorAxis::HeadAndKvPages);
        assert_eq!(dec.kv_merge, MergeKind::Lse);
        assert_eq!(dec.q_merge, MergeKind::None);
        // head axis available on both, and merges commute with KV
        assert_eq!(pre.head_merge, MergeKind::Concat);
        assert_eq!(dec.head_merge, MergeKind::Concat);
    }

    /// GDN prefill is a pipeline (chunk state chain), NOT independent
    /// partials — the token axis couples through the recurrent state.
    #[test]
    fn gdn_prefill_is_pipeline() {
        let pre = module_parallel(ModuleKind::GdnAttn, Phase::Prefill);
        assert_eq!(pre.q_merge, MergeKind::Pipeline);
        assert_eq!(pre.kv_merge, MergeKind::None); // no page axis
        let dec = module_parallel(ModuleKind::GdnAttn, Phase::Decode);
        assert_eq!(dec.axis, TensorAxis::Head);
        assert_eq!(dec.q_merge, MergeKind::None);
    }

    #[test]
    fn indexer_follows_kv_latent_replicated_over_head() {
        let idx = module_parallel(ModuleKind::DsaIndexer, Phase::Decode);
        assert_eq!(idx.axis, TensorAxis::KvPages);
        assert_eq!(idx.head_merge, MergeKind::None); // head-agnostic score
        let kv = module_parallel(ModuleKind::DsaLatentKv, Phase::Decode);
        assert_eq!(kv.axis, TensorAxis::KvPages);
        assert_eq!(kv.head_merge, MergeKind::None); // MLA: head-agnostic
    }
}
