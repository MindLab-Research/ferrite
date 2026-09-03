//! ferrite-kv: hybrid state management for GLM-5.3-Flash.
//!
//! Two fundamentally different per-layer state families live in one pool:
//!
//! 1. **Linear attention (34/45 layers)** — Gated DeltaNet keeps a
//!    *fixed-size* recurrent state per (seq, layer): `[heads, dk, dv]` plus
//!    a short-conv tail `[3*proj, conv-1]`. It does NOT grow with sequence
//!    length, so this pool is a simple slot allocator (no paging). A
//!    snapshot of it fully captures the prefix — radix-cache-friendly by
//!    construction.
//!
//! 2. **DSA (11/45 layers)** — latent KV grows per token: `kv_lora_rank`
//!    (512, nope-only: the *entire* latent is compressible, no raw rope
//!    segment) + indexer K (`index_n_heads * index_head_dim`). Paged
//!    allocation with free-page lists and per-seq page tables.
//!
//! The hybrid pool routes per layer index and owns the full per-seq
//! lifecycle (alloc/free across both families in one call).
//!
//! `shard` module: compile-time KV distribution model (ShardLayout +
//! ReshardPlan) for CP Layer-Split (prefill) × DCP (decode) 2D resharding:
//! DSA layers page-filter (`p mod n_dcp`), GatedDeltaNet layers pass
//! through atomically (state has no page dimension).

pub mod axes;
pub mod shard;

use std::collections::HashMap;

use ferrite_model::Glm53FlashConfig;
use ferrite_types::{FerriteError, Result};

pub use shard::{
    cp_layer_range, DstKvInfo, LayerShardKind, ReshardPlan, ShardLayout, Rank,
};

/// Fixed-size slot allocator for all linear-attention layers of one engine.
///
/// Slot layout (one slot = one sequence's entire linear-attention state):
/// `[layer0_state | layer1_state | ... | layerN_state]` followed by
/// `[layer0_conv | layer1_conv | ...]` — contiguous so a future transfer
/// engine (PD move, snapshot) can copy a whole slot in one op.
pub struct LinearStatePool {
    num_linear_layers: usize,
    heads: usize,
    dk: usize,
    dv: usize,
    state_elems_per_layer: usize,
    conv_channels: usize,
    conv_hist: usize,
    conv_elems_per_layer: usize,
    slots: Vec<f32>,
    slot_len: usize,
    free_slots: Vec<usize>,
    live: HashMap<u64, usize>, // seq id -> slot
}

impl LinearStatePool {
    pub fn new(cfg: &Glm53FlashConfig, max_seqs: usize) -> Self {
        let heads = cfg.linear_attn.num_heads;
        let dk = cfg.linear_attn.head_dim;
        let dv = cfg.linear_attn.head_dim;
        let conv_ch = 3 * heads * dk;
        let conv_hist = cfg.linear_attn.short_conv_kernel_size.saturating_sub(1);
        let num_layers = cfg
            .layer_types
            .iter()
            .filter(|t| matches!(t, ferrite_model::LayerType::LinearAttention))
            .count();
        let state_elems = heads * dk * dv;
        let conv_elems = conv_ch * conv_hist;
        let slot_len = num_layers * (state_elems + conv_elems);
        LinearStatePool {
            num_linear_layers: num_layers,
            heads,
            dk,
            dv,
            state_elems_per_layer: state_elems,
            conv_channels: conv_ch,
            conv_hist,
            conv_elems_per_layer: conv_elems,
            slots: vec![0.0; max_seqs * slot_len],
            slot_len,
            free_slots: (0..max_seqs).collect(),
            live: HashMap::new(),
        }
    }

    pub fn slot_len(&self) -> usize {
        self.slot_len
    }

    pub fn num_linear_layers(&self) -> usize {
        self.num_linear_layers
    }

    /// Allocate a zeroed slot for `seq`.
    pub fn alloc(&mut self, seq: u64) -> Result<usize> {
        if self.live.contains_key(&seq) {
            return Err(FerriteError::Pool(format!("linear pool: seq {seq} already allocated")));
        }
        let slot = self
            .free_slots
            .pop()
            .ok_or_else(|| FerriteError::Pool("linear pool: out of slots".into()))?;
        let base = slot * self.slot_len;
        for v in &mut self.slots[base..base + self.slot_len] {
            *v = 0.0;
        }
        self.live.insert(seq, slot);
        Ok(slot)
    }

    pub fn free(&mut self, seq: u64) -> Result<()> {
        if let Some(slot) = self.live.remove(&seq) {
            self.free_slots.push(slot);
            Ok(())
        } else {
            Err(FerriteError::Pool(format!("linear pool: seq {seq} not allocated")))
        }
    }

    fn layer_slot_idx(&self, layer_idx: usize) -> Result<usize> {
        // layer_idx is the *linear-family* index (0..num_linear_layers)
        if layer_idx >= self.num_linear_layers {
            return Err(FerriteError::Pool(format!(
                "linear pool: layer {layer_idx} >= {}",
                self.num_linear_layers
            )));
        }
        Ok(layer_idx)
    }

    /// Split of the slot: recurrent state region for a linear layer.
    /// Layout: layer states first (all layers), then conv tails.
    pub fn state_slice(&self, seq: u64, layer_idx: usize) -> Result<&[f32]> {
        let slot = *self
            .live
            .get(&seq)
            .ok_or_else(|| FerriteError::Pool(format!("linear pool: seq {seq} not live")))?;
        self.layer_slot_idx(layer_idx)?;
        let start = slot * self.slot_len + layer_idx * self.state_elems_per_layer;
        Ok(&self.slots[start..start + self.state_elems_per_layer])
    }

    /// Mutable recurrent state `[heads, dk, dv]` for (seq, linear layer).
    pub fn state_slice_mut(&mut self, seq: u64, layer_idx: usize) -> Result<&mut [f32]> {
        let slot = *self
            .live
            .get(&seq)
            .ok_or_else(|| FerriteError::Pool(format!("linear pool: seq {seq} not live")))?;
        self.layer_slot_idx(layer_idx)?;
        let start = slot * self.slot_len + layer_idx * self.state_elems_per_layer;
        let end = start + self.state_elems_per_layer;
        Ok(&mut self.slots[start..end])
    }

    /// Mutable conv tail `[conv_channels, conv_hist]` for (seq, linear layer).
    pub fn conv_slice_mut(&mut self, seq: u64, layer_idx: usize) -> Result<&mut [f32]> {
        let slot = *self
            .live
            .get(&seq)
            .ok_or_else(|| FerriteError::Pool(format!("linear pool: seq {seq} not live")))?;
        self.layer_slot_idx(layer_idx)?;
        let states_total = self.num_linear_layers * self.state_elems_per_layer;
        let start = slot * self.slot_len
            + states_total
            + layer_idx * self.conv_elems_per_layer;
        let end = start + self.conv_elems_per_layer;
        Ok(&mut self.slots[start..end])
    }

    pub fn dims(&self) -> (usize, usize, usize, usize, usize) {
        (self.num_linear_layers, self.heads, self.dk, self.dv, self.conv_hist)
    }

    pub fn conv_layout(&self) -> (usize, usize) {
        (self.conv_channels, self.conv_hist)
    }

    pub fn live_seqs(&self) -> usize {
        self.live.len()
    }

    /// Deep-copy the whole slot of `from` into a fresh allocation for `to`
    /// (prefix snapshot / PD transfer primitive).
    pub fn snapshot_to(&mut self, from: u64, to: u64) -> Result<()> {
        let src_slot = *self
            .live
            .get(&from)
            .ok_or_else(|| FerriteError::Pool(format!("linear pool: src seq {from} not live")))?;
        let dst_slot = self.alloc(to)?;
        let (s, d) = (
            src_slot * self.slot_len,
            dst_slot * self.slot_len,
        );
        let (a, b) = if s < d {
            // copy via temp to satisfy borrow checker on the same Vec
            let tmp = self.slots[s..s + self.slot_len].to_vec();
            self.slots[d..d + self.slot_len].copy_from_slice(&tmp);
            (true, true)
        } else {
            let tmp = self.slots[s..s + self.slot_len].to_vec();
            self.slots[d..d + self.slot_len].copy_from_slice(&tmp);
            (true, true)
        };
        let _ = (a, b);
        Ok(())
    }
}

/// Paged latent KV pool for the DSA layers.
///
/// Per token per DSA layer: `latent = kv_lora_rank` floats (nope-only, the
/// whole latent is paged — no rope side-channel) plus `indexer =
/// index_n_heads * index_head_dim` floats for the top-k indexer keys.
/// Pages are `page_size` tokens; a sequence holds a page table.
pub struct DsaKvPool {
    num_dsa_layers: usize,
    latent_dim: usize,
    indexer_dim: usize,
    page_size: usize,
    /// pages * num_dsa_layers * page_size * (latent+indexer)
    storage: Vec<f32>,
    page_len: usize, // per page, spans ALL layers: num_layers * page_elems
    page_elems: usize,
    free_pages: Vec<usize>,
    seq_pages: HashMap<u64, Vec<usize>>, // seq -> per-token-range physical page ids
    token_len: Vec<HashMap<u64, usize>>,  // seq -> allocated token slots
}

impl DsaKvPool {
    pub fn new(cfg: &Glm53FlashConfig, max_pages: usize, page_size: usize) -> Self {
        let num_layers = cfg
            .layer_types
            .iter()
            .filter(|t| matches!(t, ferrite_model::LayerType::DeepseekSparseAttention))
            .count();
        let latent = cfg.dsa.kv_latent_dim();
        let indexer = cfg.dsa.index_n_heads * cfg.dsa.index_head_dim;
        let page_elems = page_size * (latent + indexer);
        DsaKvPool {
            num_dsa_layers: num_layers,
            latent_dim: latent,
            indexer_dim: indexer,
            page_size,
            storage: vec![0.0; max_pages * num_layers * page_elems],
            page_len: num_layers * page_elems,
            page_elems,
            free_pages: (0..max_pages).rev().collect(),
            seq_pages: HashMap::new(),
            token_len: vec![HashMap::new()],
        }
    }

    pub fn dims(&self) -> (usize, usize, usize, usize) {
        (self.num_dsa_layers, self.latent_dim, self.indexer_dim, self.page_size)
    }

    pub fn free_pages(&self) -> usize {
        self.free_pages.len()
    }

    /// Reserve `n` more tokens for seq (allocating pages as needed).
    pub fn reserve(&mut self, seq: u64, n: usize) -> Result<()> {
        let cur = self.token_len[0].get(&seq).copied().unwrap_or(0);
        let needed = cur + n;
        let pages_needed = needed.div_ceil(self.page_size);
        let entry = self
            .seq_pages
            .get_mut(&seq)
            .ok_or_else(|| FerriteError::Pool(format!("dsa pool: seq {seq} not allocated")))?;
        // one physical page holds ALL dsa layers' tokens for this range
        // (storage layout is [page][layer][token])
        while entry.len() < pages_needed {
            let page = self
                .free_pages
                .pop()
                .ok_or_else(|| FerriteError::Pool("dsa pool: out of pages".into()))?;
            entry.push(page);
        }
        self.token_len[0].insert(seq, needed);
        Ok(())
    }

    /// Start a sequence.
    pub fn alloc_seq(&mut self, seq: u64) -> Result<()> {
        if self.seq_pages.contains_key(&seq) {
            return Err(FerriteError::Pool(format!("dsa pool: seq {seq} exists")));
        }
        self.seq_pages.insert(seq, Vec::new());
        self.token_len[0].insert(seq, 0);
        Ok(())
    }

    /// Free a sequence, returning its pages.
    pub fn free_seq(&mut self, seq: u64) -> Result<()> {
        let pages = self
            .seq_pages
            .remove(&seq)
            .ok_or_else(|| FerriteError::Pool(format!("dsa pool: seq {seq} not allocated")))?;
        self.token_len[0].remove(&seq);
        for p in pages {
            self.free_pages.push(p);
        }
        Ok(())
    }

    pub fn token_len(&self, seq: u64) -> Option<usize> {
        self.token_len[0].get(&seq).copied()
    }

    /// Write one token's KV for one DSA layer: `latent` (kv_lora_rank floats)
    /// + `indexer` (index_n_heads*index_head_dim floats).
    pub fn write_token(
        &mut self,
        seq: u64,
        dsa_layer: usize,
        token_idx: usize,
        latent: &[f32],
        indexer: &[f32],
    ) -> Result<()> {
        let len = self
            .token_len[0]
            .get(&seq)
            .copied()
            .ok_or_else(|| FerriteError::Pool(format!("dsa pool: seq {seq} not allocated")))?;
        if token_idx >= len {
            return Err(FerriteError::Pool(format!(
                "dsa pool: token {token_idx} >= len {len}"
            )));
        }
        if dsa_layer >= self.num_dsa_layers {
            return Err(FerriteError::Pool(format!(
                "dsa pool: layer {dsa_layer} >= {}",
                self.num_dsa_layers
            )));
        }
        if latent.len() != self.latent_dim || indexer.len() != self.indexer_dim {
            return Err(FerriteError::Pool("dsa pool: token dims mismatch".into()));
        }
        let page_i = token_idx / self.page_size;
        let off = token_idx % self.page_size;
        let tables = self
            .seq_pages
            .get(&seq)
            .ok_or_else(|| FerriteError::Pool("dsa pool: missing page table".into()))?;
        let page = tables[page_i];
        // storage layout: [page][layer][token_in_page][latent+indexer]
        let base = page * self.page_len + dsa_layer * self.page_elems + off * (self.latent_dim + self.indexer_dim);
        let d = &mut self.storage;
        d[base..base + self.latent_dim].copy_from_slice(latent);
        d[base + self.latent_dim..base + self.latent_dim + self.indexer_dim].copy_from_slice(indexer);
        Ok(())
    }

    /// Read one token's KV for one DSA layer.
    pub fn read_token(
        &self,
        seq: u64,
        dsa_layer: usize,
        token_idx: usize,
    ) -> Result<(&[f32], &[f32])> {
        let len = self
            .token_len[0]
            .get(&seq)
            .copied()
            .ok_or_else(|| FerriteError::Pool(format!("dsa pool: seq {seq} not allocated")))?;
        if token_idx >= len {
            return Err(FerriteError::Pool(format!("dsa pool: token {token_idx} >= len {len}")));
        }
        let page_i = token_idx / self.page_size;
        let off = token_idx % self.page_size;
        let tables = self
            .seq_pages
            .get(&seq)
            .ok_or_else(|| FerriteError::Pool("dsa pool: missing page table".into()))?;
        let page = tables[page_i];
        let base = page * self.page_len + dsa_layer * self.page_elems + off * (self.latent_dim + self.indexer_dim);
        let row = &self.storage[base..base + self.latent_dim + self.indexer_dim];
        Ok((&row[..self.latent_dim], &row[self.latent_dim..]))
    }
}

/// Unified per-seq state across both families. Routes by model layer index
/// (the engine maps model layer -> family index via the config's layer plan).
pub struct HybridStatePool {
    pub linear: LinearStatePool,
    pub dsa: DsaKvPool,
    dsa_layer_ids: Vec<usize>,
    linear_layer_ids: Vec<usize>,
}

impl HybridStatePool {
    pub fn new(cfg: &Glm53FlashConfig, max_seqs: usize, dsa_pages: usize, page_size: usize) -> Self {
        let mut dsa_layer_ids = Vec::new();
        let mut linear_layer_ids = Vec::new();
        for (i, t) in cfg.layer_types.iter().enumerate() {
            match t {
                ferrite_model::LayerType::LinearAttention => linear_layer_ids.push(i),
                ferrite_model::LayerType::DeepseekSparseAttention => dsa_layer_ids.push(i),
            }
        }
        HybridStatePool {
            linear: LinearStatePool::new(cfg, max_seqs),
            dsa: DsaKvPool::new(cfg, dsa_pages, page_size),
            dsa_layer_ids,
            linear_layer_ids,
        }
    }

    /// Family index of a model layer (e.g. 3rd DSA layer is family idx 2).
    pub fn dsa_family_idx(&self, model_layer: usize) -> Option<usize> {
        self.dsa_layer_ids.iter().position(|&l| l == model_layer)
    }
    pub fn linear_family_idx(&self, model_layer: usize) -> Option<usize> {
        self.linear_layer_ids.iter().position(|&l| l == model_layer)
    }

    pub fn alloc_seq(&mut self, seq: u64) -> Result<()> {
        self.linear.alloc(seq)?;
        self.dsa.alloc_seq(seq).map_err(|e| {
            let _ = self.linear.free(seq);
            e
        })
    }

    pub fn free_seq(&mut self, seq: u64) -> Result<()> {
        self.linear.free(seq)?;
        self.dsa.free_seq(seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrite_model::Glm53FlashConfig;

    #[test]
    fn linear_pool_lifecycle() {
        let cfg = Glm53FlashConfig::test_config(); // 2 linear + 2 dsa layers
        let mut p = LinearStatePool::new(&cfg, 4);
        assert_eq!(p.num_linear_layers(), 2);
        let (nl, h, dk, _dv, conv_hist) = p.dims();
        assert_eq!((nl, h, dk, conv_hist), (2, 4, 32, 3));
        let slot = p.alloc(1).unwrap();
        assert_eq!(slot, 3, "free list pops from the top (LIFO)");
        // write state of layer 0, check isolation from layer 1
        let s0 = p.state_slice_mut(1, 0).unwrap();
        for v in s0.iter_mut() {
            *v = 7.0;
        }
        let s1 = p.state_slice(1, 1).unwrap();
        assert!(s1.iter().all(|v| *v == 0.0), "layer 1 untouched");
        // conv tail separate region
        let c0 = p.conv_slice_mut(1, 0).unwrap();
        assert_eq!(c0.len(), 3 * 4 * 32 * 3); // channels * hist
        for v in c0.iter_mut() {
            *v = 3.0;
        }
        // free + realloc zeroes
        p.free(1).unwrap();
        p.alloc(2).unwrap();
        let s2 = p.state_slice(2, 0).unwrap();
        assert!(s2.iter().all(|v| *v == 0.0), "realloc zeroed");
    }

    #[test]
    fn linear_pool_snapshot() {
        let cfg = Glm53FlashConfig::test_config();
        let mut p = LinearStatePool::new(&cfg, 4);
        p.alloc(1).unwrap();
        let s = p.state_slice_mut(1, 0).unwrap();
        for (i, v) in s.iter_mut().enumerate() {
            *v = i as f32 * 0.5;
        }
        p.snapshot_to(1, 2).unwrap();
        let a = p.state_slice(1, 0).unwrap().to_vec();
        let b = p.state_slice(2, 0).unwrap().to_vec();
        assert_eq!(a, b, "snapshot copies state");
    }

    #[test]
    fn dsa_pool_paging() {
        let cfg = Glm53FlashConfig::test_config(); // 2 dsa layers, latent 64, indexer 2*16=32
        let mut p = DsaKvPool::new(&cfg, 16, 8); // 16 pages, page=8 tokens
        let (nl, latent, indexer, page) = p.dims();
        assert_eq!((nl, latent, indexer, page), (2, 64, 32, 8));
        p.alloc_seq(1).unwrap();
        p.reserve(1, 20).unwrap(); // needs 3 pages (ceil(20/8))
        assert_eq!(p.free_pages(), 13);
        assert_eq!(p.token_len(1), Some(20));
        // write/read roundtrip across page boundary (token 7, 8, 15) — layer 0 only
        for t in [7usize, 8, 15] {
            let lat: Vec<f32> = (0..latent).map(|i| t as f32 + i as f32 * 0.1).collect();
            let idx: Vec<f32> = (0..indexer).map(|i| -(t as f32) - i as f32 * 0.2).collect();
            p.write_token(1, 0, t, &lat, &idx).unwrap();
            let (rl, ri) = p.read_token(1, 0, t).unwrap();
            assert_eq!(rl, &lat[..]);
            assert_eq!(ri, &idx[..]);
        }
        // layer isolation
        let (_, ri1) = p.read_token(1, 1, 7).unwrap();
        assert!(ri1.iter().all(|v| *v == 0.0), "layer 1 untouched");
        p.free_seq(1).unwrap();
        assert_eq!(p.free_pages(), 16, "pages returned");
    }

    #[test]
    fn hybrid_routing() {
        let cfg = Glm53FlashConfig::test_config();
        let mut h = HybridStatePool::new(&cfg, 8, 32, 16);
        // test config layers: 0 lin, 1 dsa, 2 lin, 3 dsa
        assert_eq!(h.linear_family_idx(0), Some(0));
        assert_eq!(h.linear_family_idx(2), Some(1));
        assert_eq!(h.dsa_family_idx(1), Some(0));
        assert_eq!(h.dsa_family_idx(3), Some(1));
        assert_eq!(h.dsa_family_idx(0), None);
        h.alloc_seq(7).unwrap();
        assert_eq!(h.linear.live_seqs(), 1);
        assert_eq!(h.dsa.token_len(7), Some(0));
        h.dsa.reserve(7, 4).unwrap();
        assert_eq!(h.dsa.token_len(7), Some(4));
        h.free_seq(7).unwrap();
        assert_eq!(h.linear.live_seqs(), 0);
    }
}
