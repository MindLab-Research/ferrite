//! Shard layout & 2D reshard plan — the compile-time KV distribution model
//! for PDAF-disaggregated, CP/DCP-sharded serving.
//!
//! Design (from the page-level 2D resharding methodology, adapted to
//! GLM-5.3-Flash's hybrid attention):
//!
//! - **Prefill side** runs CP Layer-Split: `n_cp` ranks each own a
//!   contiguous layer range (div + remainder assignment). Each CP rank
//!   holds **all pages** of its owned layers.
//! - **Decode side** runs DCP: `n_dcp` ranks each own `p mod n_dcp`
//!   pages of **all** layers.
//! - The PD transfer is therefore a **2D reshard** (layer × page), solved
//!   with source-side page filtering: `mask(p, d) = [p mod n_dcp == d]`,
//!   `local_slot(p) = p div n_dcp` — the RDMA write lands directly in the
//!   destination's DCP layout, no post-processing.
//!
//! **GLM-5.3-Flash's hybrid wrinkle** (vs GLM-5.2's all-DSA layers):
//! - **DSA layers (11)**: paged latent KV → full 2D reshard applies.
//! - **GatedDeltaNet layers (34)**: fixed-size recurrent state
//!   `[heads, dk, dv]`, **no page dimension** — the reshard is a layer
//!   pass-through (state transfers atomically per layer; DCP has no
//!   meaning for it since the state does not grow with sequence length).
//!
//! Everything here is **compile-time static** (deployment topology → plan),
//! matching ferrite's specialisation philosophy: the reshard rules are
//! generated once per deployment, not decided per request. Per-request
//! metadata (`DstKvInfo`) carries only `(n_dcp, d)` — heterogeneous decode
//! groups can coexist without renegotiating a global topology.

use ferrite_model::Glm53FlashConfig;

/// Rank id (prefill CP rank or decode DCP rank).
pub type Rank = usize;

/// How a layer's KV/state is sharded — routes the reshard strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerShardKind {
    /// GatedDeltaNet (linear attention): fixed-size recurrent state
    /// `[heads, dk, dv]` — no page dimension. Reshard = layer pass-through
    /// (the state transfers atomically; DCP cannot split it).
    LinearState,
    /// DSA (sparse attention): paged latent KV. Reshard = page filtering
    /// (`p mod n_dcp == d`, local slot `p div n_dcp`).
    PagedLatent { page_size: usize },
}

/// Per-request destination metadata (decode → prefill, per request, not
/// per cluster — heterogeneous decode groups can coexist; the plan is
/// static but which `(n_dcp, d)` a request targets rides with the request).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DstKvInfo {
    /// Decode-side DCP size for this request's decode group.
    pub n_dcp: usize,
    /// This decode rank's DCP index (`= tp_rank mod n_dcp`).
    pub d: Rank,
}

/// CP Layer-Split layer range (div + remainder, contiguous):
/// rank `r` owns `[r*base + min(r, extra), (r+1)*base + min(r+1, extra))`.
pub fn cp_layer_range(n_layers: usize, n_cp: usize, r: Rank) -> (usize, usize) {
    let base = n_layers / n_cp;
    let extra = n_layers % n_cp;
    let start = r * base + r.min(extra);
    let end = (r + 1) * base + (r + 1).min(extra);
    (start, end)
}

/// The full KV distribution across the P/D deployment.
#[derive(Debug, Clone)]
pub struct ShardLayout {
    /// Prefill CP size (layer-split ranks).
    pub n_cp: usize,
    /// Decode DCP size (page-shard ranks).
    pub n_dcp: usize,
    /// Per-layer owner on the prefill side (CP rank).
    pub layer_owner: Vec<Rank>,
    /// Per-layer shard kind (routes the reshard strategy).
    pub layer_kind: Vec<LayerShardKind>,
    /// Physical page size (tokens per page; DSA latent pages).
    pub page_size: usize,
}

impl ShardLayout {
    /// Compile from a deployment topology + model config.
    pub fn compile(cfg: &Glm53FlashConfig, n_cp: usize, n_dcp: usize, page_size: usize) -> Self {
        let n = cfg.num_hidden_layers;
        let mut layer_owner = vec![0usize; n];
        for r in 0..n_cp {
            let (start, end) = cp_layer_range(n, n_cp, r);
            for l in start..end {
                layer_owner[l] = r;
            }
        }
        let layer_kind = (0..n)
            .map(|l| {
                if cfg.is_dsa_layer(l) {
                    LayerShardKind::PagedLatent { page_size }
                } else {
                    LayerShardKind::LinearState
                }
            })
            .collect();
        ShardLayout { n_cp, n_dcp, layer_owner, layer_kind, page_size }
    }

    /// Single-node trivial layout (no sharding): 1 CP rank, 1 DCP rank,
    /// everything local. This is what the current single-GPU engine uses —
    /// the reshard plan degenerates to identity, but the *types* flow.
    pub fn single_node(cfg: &Glm53FlashConfig) -> Self {
        Self::compile(cfg, 1, 1, 64)
    }

    /// Layers owned by prefill CP rank `r`.
    pub fn cp_layers(&self, r: Rank) -> Vec<usize> {
        (0..self.layer_owner.len())
            .filter(|&l| self.layer_owner[l] == r)
            .collect()
    }
}

/// One (src CP rank → dst DCP rank) transfer path.
#[derive(Debug, Clone)]
pub struct TransferPath {
    /// Source prefill CP rank.
    pub src_rank: Rank,
    /// Destination decode DCP rank.
    pub dst_rank: Rank,
    /// Layers this path carries (the src rank's owned layers).
    pub layers: Vec<usize>,
}

/// The compiled 2D reshard plan: `n_cp × n_dcp` paths, each carrying the
/// src rank's layers with per-layer routing (paged → filter; state → pass).
#[derive(Debug, Clone)]
pub struct ReshardPlan {
    pub layout: ShardLayout,
    /// All (src, dst) path pairs.
    pub paths: Vec<TransferPath>,
}

impl ReshardPlan {
    /// Compile the plan: every (CP rank × DCP rank) pair gets a path with
    /// the CP rank's layer set. 32 paths for CP=8 × DCP=4.
    pub fn compile(cfg: &Glm53FlashConfig, n_cp: usize, n_dcp: usize, page_size: usize) -> Self {
        let layout = ShardLayout::compile(cfg, n_cp, n_dcp, page_size);
        let mut paths = Vec::with_capacity(n_cp * n_dcp);
        for src in 0..n_cp {
            let layers = layout.cp_layers(src);
            for dst in 0..n_dcp {
                paths.push(TransferPath { src_rank: src, dst_rank: dst, layers: layers.clone() });
            }
        }
        ReshardPlan { layout, paths }
    }

    /// Single-node identity plan (transfer = local move, no filtering).
    pub fn single_node(cfg: &Glm53FlashConfig) -> Self {
        Self::compile(cfg, 1, 1, 64)
    }

    /// Page filter for DSA layers (source-side, lands directly in the
    /// destination's DCP layout — no post-transfer reshard):
    /// `Some(local_slot)` if page `p` belongs to DCP rank `d`, else `None`.
    ///
    /// local_slot = p div n_dcp (compress global → destination-local).
    pub fn page_mask(&self, p: usize, d: Rank) -> Option<usize> {
        let n = self.layout.n_dcp;
        if n <= 1 {
            return Some(p); // identity: single DCP rank owns everything
        }
        if p % n == d {
            Some(p / n)
        } else {
            None
        }
    }

    /// Filter a page-index list for DCP rank `d` (the batch form of
    /// `page_mask`): returns the destination-local slots for the pages
    /// this rank owns.
    pub fn filter_pages_for_dcp_rank(&self, pages: &[usize], d: Rank) -> Vec<usize> {
        if self.layout.n_dcp <= 1 {
            return pages.to_vec();
        }
        pages.iter().filter_map(|&p| self.page_mask(p, d)).collect()
    }

    /// The transfer semantics for a layer: paged (filter) vs state
    /// (pass-through). This routes the transfer worker per layer.
    pub fn layer_transfer(&self, layer: usize) -> LayerShardKind {
        self.layout.layer_kind[layer]
    }

    /// Linear-state layers owned by `src_rank` — pass-through transfers
    /// (each state is atomic; every decode rank that needs the layer's
    /// state receives it whole).
    pub fn state_layers(&self, src_rank: Rank) -> Vec<usize> {
        self.layout
            .cp_layers(src_rank)
            .into_iter()
            .filter(|&l| self.layout.layer_kind[l] == LayerShardKind::LinearState)
            .collect()
    }

    /// DSA (paged) layers owned by `src_rank` — page-filtered transfers.
    pub fn paged_layers(&self, src_rank: Rank) -> Vec<usize> {
        self.layout
            .cp_layers(src_rank)
            .into_iter()
            .filter(|&l| matches!(self.layout.layer_kind[l], LayerShardKind::PagedLatent { .. }))
            .collect()
    }

    /// Number of transfer paths (n_cp × n_dcp).
    pub fn path_count(&self) -> usize {
        self.paths.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrite_model::Glm53FlashConfig;

    #[test]
    fn cp_layer_range_div_remainder() {
        // 45 layers / 8 ranks: base=5 extra=5 → first 5 ranks get 6, last 3 get 5
        let (s, e) = cp_layer_range(45, 8, 0);
        assert_eq!((s, e), (0, 6));
        let (s, e) = cp_layer_range(45, 8, 4);
        assert_eq!((s, e), (24, 30));
        let (s, e) = cp_layer_range(45, 8, 7);
        assert_eq!((s, e), (40, 45));
        // full coverage, no overlap
        let mut all = Vec::new();
        for r in 0..8 {
            let (s, e) = cp_layer_range(45, 8, r);
            all.extend(s..e);
        }
        assert_eq!(all, (0..45).collect::<Vec<_>>());
        // 78 layers / 8 (GLM-5.2's case): 6 ranks × 10 + 2 × 9
        // 78/8 = 9 r6; rank 5: 5*9 + min(5,6) = 50, end = 6*9 + min(6,6) = 60
        let (s, e) = cp_layer_range(78, 8, 5);
        assert_eq!((s, e), (50, 60));
        let (s, e) = cp_layer_range(78, 8, 7);
        assert_eq!((s, e), (69, 78));
    }

    #[test]
    fn shard_layout_glm53_flash_hybrid_routing() {
        let cfg = Glm53FlashConfig::production_config();
        let layout = ShardLayout::compile(&cfg, 8, 4, 64);
        // 45 layers: DSA at 3,7,...,43 (11 layers); linear elsewhere (34)
        assert_eq!(layout.layer_kind.iter().filter(|k| matches!(k, LayerShardKind::PagedLatent { .. })).count(), 11);
        assert_eq!(layout.layer_kind.iter().filter(|k| matches!(k, LayerShardKind::LinearState)).count(), 34);
        // DSA layer 3 is paged; linear layer 0 is state
        assert!(matches!(layout.layer_kind[3], LayerShardKind::PagedLatent { page_size: 64 }));
        assert_eq!(layout.layer_kind[0], LayerShardKind::LinearState);
        // CP owner coverage
        let mut all = Vec::new();
        for r in 0..8 {
            all.extend(layout.cp_layers(r));
        }
        assert_eq!(all, (0..45).collect::<Vec<_>>());
    }

    #[test]
    fn reshard_plan_32_paths_and_hybrid_split() {
        let cfg = Glm53FlashConfig::production_config();
        let plan = ReshardPlan::compile(&cfg, 8, 4, 64);
        assert_eq!(plan.path_count(), 32); // 8 CP × 4 DCP
        // each path carries the src rank's layers (~5-6 layers for 45/8)
        let p0 = &plan.paths[0]; // src=0, dst=0
        assert_eq!(p0.src_rank, 0);
        assert_eq!(p0.dst_rank, 0);
        assert_eq!(p0.layers.len(), 6); // 45/8 = 5r5 → rank 0 gets 6
        // same src rank across dst ranks: identical layer sets
        let p1 = plan.paths.iter().find(|p| p.src_rank == 0 && p.dst_rank == 3).unwrap();
        assert_eq!(p0.layers, p1.layers);
        // hybrid routing within a path: rank 0 owns layers [0,6) → all linear (DSA starts at 3)
        assert!(plan.state_layers(0).len() >= 3);
        // rank 0 owns [0,6): layers 0,1,2 linear; 3,4,5 DSA (3 is DSA, 4/5 linear)
        assert_eq!(plan.state_layers(0), vec![0, 1, 2, 4, 5]);
        assert_eq!(plan.paged_layers(0), vec![3]);
    }

    #[test]
    fn page_filter_dcp4() {
        let cfg = Glm53FlashConfig::production_config();
        let plan = ReshardPlan::compile(&cfg, 8, 4, 64);
        // mask(p, d): p mod 4 == d; local slot = p div 4
        assert_eq!(plan.page_mask(0, 0), Some(0));
        assert_eq!(plan.page_mask(1, 0), None);
        assert_eq!(plan.page_mask(1, 1), Some(0));
        assert_eq!(plan.page_mask(5, 1), Some(1)); // 5 mod 4 = 1, 5 div 4 = 1
        assert_eq!(plan.page_mask(7, 3), Some(1));
        assert_eq!(plan.page_mask(8, 0), Some(2));
        // batch filter: pages 0..8 for d=1 → [0, 1] (pages 1, 5)
        let filtered = plan.filter_pages_for_dcp_rank(&(0..8).collect::<Vec<_>>(), 1);
        assert_eq!(filtered, vec![0, 1]);
        // 256 pages / DCP=4 → 64 per rank
        let all: Vec<usize> = (0..256).collect();
        for d in 0..4 {
            assert_eq!(plan.filter_pages_for_dcp_rank(&all, d).len(), 64);
        }
    }

    #[test]
    fn single_node_identity() {
        let cfg = Glm53FlashConfig::production_config();
        let plan = ReshardPlan::single_node(&cfg);
        assert_eq!(plan.path_count(), 1);
        // identity: page_mask passes everything through unchanged
        assert_eq!(plan.page_mask(42, 0), Some(42));
        let pages = vec![1, 2, 3, 100];
        assert_eq!(plan.filter_pages_for_dcp_rank(&pages, 0), pages);
        // single path carries all 45 layers
        assert_eq!(plan.paths[0].layers.len(), 45);
    }

    #[test]
    fn dst_kv_info_per_request() {
        // heterogeneous decode groups: (n_dcp, d) rides per request
        let a = DstKvInfo { n_dcp: 4, d: 1 };
        let b = DstKvInfo { n_dcp: 2, d: 0 };
        assert_ne!(a, b);
        assert_eq!(a.d, 1);
    }
}
