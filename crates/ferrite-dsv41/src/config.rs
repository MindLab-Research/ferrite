//! DeepSeek-V4.1-Flash configuration.
//!
//! Parsed from the released HF `config.json` (nested `text_config` /
//! `quantization_config` / `vision_config`). Field semantics follow the
//! reference implementation (`inference/model.py::ModelArgs` + `config.json`
//! shipped next to it); every derived helper here mirrors a property the
//! reference computes inline.
//!
//! The released shapes (for documentation, all values are parsed not hardcoded):
//!   40 backbone layers + 3 DSpark draft layers under `mtp.*`
//!   dim 5120, 64 heads, head_dim 512, **1 KV head**, q_lora 1280,
//!   o_lora 1024 with o_groups 8 (block-diagonal low-rank output)
//!   window 128, 384 routed + 1 shared experts, top-k 6, sqrtsoftplus routing
//!   compress_ratios [0,0,2 x18,1 x20,0,0,0], kv_source [2,8,14,20],
//!   index_source [2,8,14,20,24,28,32,36], candidate_source 20
//!   hc_mult 4 / sinkhorn 20 / eps 1e-6 (identical to GLM-5.3-Flash)
//!   engram at layers 1 and 14 (2 x ~384M-row n-gram tables, fp8)
//!   fp8 dense weights at 32x32 ue8m0 blocks, **fp4 (e2m1)** experts

use ferrite_types::{FerriteError, Result};
use serde_json::Value;

/// How a backbone layer's KV is produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvMode {
    /// `compress_ratios[layer] == 0`: sliding window only.
    WindowOnly,
    /// ratio r > 0 and this layer is a `kv_source`: it pools `r` tokens into
    /// one latent (softmax-gated) and publishes the cache the peers read.
    CompressSource,
    /// ratio r > 0 but not a source: reads the source layer's cache.
    CompressConsumer,
}

#[derive(Debug, Clone)]
pub struct Dsv41Config {
    // ---- runtime limits (size the caches) ----
    pub max_batch_size: usize,
    pub max_seq_len: usize,
    pub temperature: f32,

    // ---- backbone shape ----
    pub vocab_size: usize,
    pub dim: usize,
    pub moe_inter_dim: usize,
    pub n_layers: usize,
    pub n_mtp_layers: usize,
    pub n_heads: usize,
    /// Per-head q/k/v width (512). The MLA latent is 1 KV head of this width.
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub norm_eps: f32,

    // ---- sparse attention ----
    pub window_size: usize,
    /// One entry per layer (backbone + MTP): 0 = window only, r = KV pooled r:1.
    pub compress_ratios: Vec<usize>,
    pub kv_source_layers: Vec<usize>,
    pub index_source_layers: Vec<usize>,
    pub compress_rope_theta: f32,
    pub original_seq_len: usize,
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,

    // ---- indexer (two-level sparse selection) ----
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    /// `< 0` disables candidate pre-filtering.
    pub candidate_source_layer: i64,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,

    // ---- MoE ----
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub n_activated_experts: usize,
    pub score_func: String,
    pub gate_temp: f32,
    pub norm_topk_prob: bool,
    pub route_scale: f32,
    pub swiglu_limit: f32,

    // ---- hyper-connections ----
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,

    // ---- quantisation ----
    /// Dense weight/activation format: "fp8" (e4m3 with ue8m0 32x32 blocks).
    pub weight_dtype: String,
    /// Expert format: `Some("fp4")` for the released checkpoint.
    pub expert_dtype: Option<String>,
    pub fp8_block: usize,
    pub fp4_block: usize,

    // ---- engram ----
    pub engram_layer_ids: Vec<usize>,
    pub engram_num_embeddings: Vec<u64>,
    pub engram_max_ngram_size: usize,
    pub engram_vocab_size: u64,
    pub engram_n_heads: usize,
    pub engram_head_dim: usize,
    pub engram_pad_id: u32,
    pub engram_compressed_vocab_size: u64,

    // ---- DSpark draft head ----
    pub dspark_block_size: usize,
    pub dspark_noise_token_id: u32,
    pub dspark_target_layer_ids: Vec<usize>,
    pub dspark_markov_rank: usize,
    pub dspark_n_routed_experts: usize,
    pub dspark_n_activated_experts: usize,

    // ---- vision ----
    pub vision_n_layers: usize,
    pub vision_dim: usize,
    pub vision_n_heads: usize,
    pub vision_inter_dim: usize,
    pub vision_patch_size: usize,
    pub vision_rope_theta: f32,
    pub vision_downsample_ratio: usize,
    pub vision_max_n_token: usize,
    pub vision_min_pixels: usize,
    pub image_token_id: u32,
}

fn u(v: &Value, k: &str) -> Option<u64> {
    v.get(k).and_then(|x| x.as_u64())
}
fn f(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(|x| x.as_f64())
}
fn b(v: &Value, k: &str) -> Option<bool> {
    v.get(k).and_then(|x| x.as_bool())
}
fn s(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(|x| x.as_str()).map(|x| x.to_string())
}
fn us(v: &Value, k: &str) -> Vec<usize> {
    v.get(k)
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|e| e.as_u64()).map(|e| e as usize).collect())
        .unwrap_or_default()
}
fn us64(v: &Value, k: &str) -> Vec<u64> {
    v.get(k)
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|e| e.as_u64()).collect())
        .unwrap_or_default()
}

impl Dsv41Config {
    /// Parse the released `config.json`. Accepts both the HF layout (shape
    /// fields under `text_config`, quantisation under `quantization_config`,
    /// vision under `vision_config`) and the flat layout shipped next to the
    /// reference implementation.
    pub fn from_json_str(txt: &str) -> Result<Self> {
        let root: Value = serde_json::from_str(txt)
            .map_err(|e| FerriteError::Config(format!("dsv41 config json: {e}")))?;
        // HF: nested; flat reference config: the root itself.
        let t = root.get("text_config").unwrap_or(&root);
        let q = root.get("quantization_config").unwrap_or(&Value::Null);
        let vis = root.get("vision_config").unwrap_or(&Value::Null);

        let n_layers = u(t, "num_hidden_layers").unwrap_or(40) as usize;
        let n_mtp = u(t, "num_nextn_predict_layers")
            .or_else(|| u(t, "n_mtp_layers"))
            .unwrap_or(0) as usize;

        // compress_ratios covers backbone + MTP layers.
        let mut compress_ratios = us(t, "compress_ratios");
        compress_ratios.resize(n_layers + n_mtp, 0);

        let wbs = q
            .get("weight_block_size")
            .and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|e| e.as_u64()).collect::<Vec<_>>())
            .unwrap_or_else(|| vec![32, 32]);

        let mut cfg = Dsv41Config {
            max_batch_size: u(t, "max_batch_size").unwrap_or(4) as usize,
            max_seq_len: u(t, "max_seq_len")
                .or_else(|| u(t, "max_position_embeddings"))
                .unwrap_or(4096) as usize,
            temperature: f(t, "temperature").unwrap_or(1.0) as f32,

            vocab_size: u(t, "vocab_size").unwrap_or(129280) as usize,
            dim: u(t, "dim").or_else(|| u(t, "hidden_size")).unwrap_or(1024) as usize,
            moe_inter_dim: u(t, "moe_inter_dim")
                .or_else(|| u(t, "moe_intermediate_size"))
                .unwrap_or(1024) as usize,
            n_layers,
            n_mtp_layers: n_mtp,
            n_heads: u(t, "n_heads")
                .or_else(|| u(t, "num_attention_heads"))
                .unwrap_or(16) as usize,
            head_dim: u(t, "head_dim").unwrap_or(128) as usize,
            rope_head_dim: u(t, "rope_head_dim")
                .or_else(|| u(t, "qk_rope_head_dim"))
                .unwrap_or(32) as usize,
            q_lora_rank: u(t, "q_lora_rank").unwrap_or(256) as usize,
            o_lora_rank: u(t, "o_lora_rank").unwrap_or(256) as usize,
            o_groups: u(t, "o_groups").unwrap_or(8) as usize,
            norm_eps: f(t, "norm_eps").or_else(|| f(t, "rms_norm_eps")).unwrap_or(1e-20) as f32,

            compress_ratios: Vec::new(), // filled below (needs n_layers + n_mtp)
            window_size: u(t, "window_size")
                .or_else(|| u(t, "sliding_window"))
                .unwrap_or(128) as usize,
            kv_source_layers: us(t, "kv_source_layers")
                .into_iter()
                .chain(us(t, "kv_source_layer_ids"))
                .collect::<Vec<_>>(),
            index_source_layers: us(t, "index_source_layers")
                .into_iter()
                .chain(us(t, "index_source_layer_ids"))
                .collect::<Vec<_>>(),
            compress_rope_theta: f(t, "compress_rope_theta").unwrap_or(40000.0) as f32,
            // YaRN is configured through `rope_scaling` in this release:
            //   {"rope_type":"yarn","factor":16,"beta_fast":32,"beta_slow":1,
            //    "original_max_position_embeddings":65536}
            // The earlier version read only a top-level `original_seq_len` and
            // defaulted to 0 — which DISABLES YaRN entirely (the reference's
            // `if original_seq_len > 0` branch), so every rope frequency was the
            // unscaled one and the positional encoding disagreed with the
            // reference from position 1 onward.
            original_seq_len: u(t, "original_seq_len")
                .or_else(|| {
                    t.get("rope_scaling")
                        .and_then(|r| r.get("original_max_position_embeddings"))
                        .and_then(|x| x.as_u64())
                        .map(|v| v as usize)
                })
                .unwrap_or(0),
            rope_theta: f(t, "rope_theta").unwrap_or(10000.0) as f32,
            rope_factor: f(t, "rope_factor")
                .or_else(|| t.get("rope_scaling").and_then(|r| r.get("factor")).and_then(|x| x.as_f64()))
                .unwrap_or(40.0) as f32,
            beta_fast: f(t, "beta_fast")
                .or_else(|| t.get("rope_scaling").and_then(|r| r.get("beta_fast")).and_then(|x| x.as_f64()))
                .unwrap_or(32.0) as f32,
            beta_slow: f(t, "beta_slow")
                .or_else(|| t.get("rope_scaling").and_then(|r| r.get("beta_slow")).and_then(|x| x.as_f64()))
                .unwrap_or(1.0) as f32,

            index_n_heads: u(t, "index_n_heads").unwrap_or(16) as usize,
            index_head_dim: u(t, "index_head_dim").unwrap_or(64) as usize,
            index_topk: u(t, "index_topk").unwrap_or(64) as usize,
            candidate_source_layer: t
                .get("candidate_source_layer")
                .or_else(|| t.get("candidate_source_layer_id"))
                .and_then(|x| x.as_i64())
                .unwrap_or(-1),
            candidate_topk_blocks: u(t, "candidate_topk_blocks").unwrap_or(0) as usize,
            candidate_block_size: u(t, "candidate_block_size").unwrap_or(0) as usize,

            n_routed_experts: u(t, "n_routed_experts").unwrap_or(8) as usize,
            n_shared_experts: u(t, "n_shared_experts").unwrap_or(1) as usize,
            n_activated_experts: u(t, "n_activated_experts")
                .or_else(|| u(t, "num_experts_per_tok"))
                .unwrap_or(2) as usize,
            score_func: s(t, "score_func")
                .or_else(|| s(t, "scoring_func"))
                .unwrap_or_else(|| "sqrtsoftplus".into()),
            gate_temp: f(t, "gate_temp").unwrap_or(1.0) as f32,
            norm_topk_prob: b(t, "norm_topk_prob").unwrap_or(true),
            route_scale: f(t, "route_scale")
                .or_else(|| f(t, "routed_scaling_factor"))
                .unwrap_or(1.0) as f32,
            swiglu_limit: f(t, "swiglu_limit").unwrap_or(0.0) as f32,

            hc_mult: u(t, "hc_mult").unwrap_or(4) as usize,
            hc_sinkhorn_iters: u(t, "hc_sinkhorn_iters").unwrap_or(20) as usize,
            hc_eps: f(t, "hc_eps").unwrap_or(1e-6) as f32,

            // quantisation format: the HF config declares it in
            // quantization_config.quant_method ("fp8"), the flat reference
            // config in `dtype` at the root.
            weight_dtype: s(q, "quant_method")
                .or_else(|| s(t, "dtype"))
                .or_else(|| s(&root, "dtype"))
                .unwrap_or_else(|| "fp8".into()),
            expert_dtype: s(t, "expert_dtype").or_else(|| s(q, "expert_dtype")),
            fp8_block: *wbs.first().unwrap_or(&32) as usize,
            fp4_block: 32,

            engram_layer_ids: us(t, "engram_layer_ids"),
            engram_num_embeddings: us64(t, "engram_num_embeddings"),
            engram_max_ngram_size: u(t, "engram_max_ngram_size").unwrap_or(1) as usize,
            engram_vocab_size: u(t, "engram_vocab_size").unwrap_or(0),
            engram_n_heads: u(t, "engram_n_heads").unwrap_or(0) as usize,
            engram_head_dim: u(t, "engram_head_dim").unwrap_or(0) as usize,
            engram_pad_id: u(t, "engram_pad_id")
                .or_else(|| u(t, "engram_pad_token_id"))
                .unwrap_or(2) as u32,
            engram_compressed_vocab_size: u(t, "engram_compressed_vocab_size").unwrap_or(0),

            dspark_block_size: u(t, "dspark_block_size").unwrap_or(0) as usize,
            dspark_noise_token_id: u(t, "dspark_noise_token_id").unwrap_or(0) as u32,
            dspark_target_layer_ids: us(t, "dspark_target_layer_ids"),
            dspark_markov_rank: u(t, "dspark_markov_rank").unwrap_or(256) as usize,
            dspark_n_routed_experts: u(t, "dspark_n_routed_experts").unwrap_or(0) as usize,
            dspark_n_activated_experts: u(t, "dspark_n_activated_experts")
                .or_else(|| u(t, "dspark_num_experts_per_tok"))
                .unwrap_or(0) as usize,

            vision_n_layers: u(vis, "num_hidden_layers").unwrap_or(0) as usize,
            vision_dim: u(vis, "hidden_size").unwrap_or(1024) as usize,
            vision_n_heads: u(vis, "num_attention_heads").unwrap_or(16) as usize,
            vision_inter_dim: u(vis, "intermediate_size").unwrap_or(2816) as usize,
            vision_patch_size: u(vis, "patch_size").unwrap_or(14) as usize,
            vision_rope_theta: f(vis, "rope_theta").unwrap_or(10000.0) as f32,
            vision_downsample_ratio: u(vis, "downsample_ratio").unwrap_or(3) as usize,
            vision_max_n_token: u(vis, "max_image_tokens").unwrap_or(1024) as usize,
            vision_min_pixels: u(vis, "min_pixels").unwrap_or(544 * 544) as usize,
            image_token_id: u(&root, "image_token_id")
                .or_else(|| u(t, "image_token_id"))
                .unwrap_or(129264) as u32,
        };
        cfg.kv_source_layers.sort_unstable();
        cfg.kv_source_layers.dedup();
        cfg.index_source_layers.sort_unstable();
        cfg.index_source_layers.dedup();
        cfg.compress_ratios = compress_ratios;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        let n = self.n_layers;
        if self.compress_ratios.len() < n + self.n_mtp_layers {
            return Err(FerriteError::Config(format!(
                "compress_ratios len {} < n_layers {} + n_mtp {}",
                self.compress_ratios.len(),
                n,
                self.n_mtp_layers
            )));
        }
        if self.head_dim == 0 || self.n_heads % self.o_groups != 0 {
            return Err(FerriteError::Config(
                "n_heads must be divisible by o_groups".into(),
            ));
        }
        for &l in &self.kv_source_layers {
            if l >= n || self.compress_ratios[l] == 0 {
                return Err(FerriteError::Config(format!(
                    "kv_source layer {l} must be a backbone layer with ratio > 0"
                )));
            }
        }
        for &l in &self.index_source_layers {
            if l >= n {
                return Err(FerriteError::Config(format!("index_source layer {l} >= n_layers")));
            }
        }
        if self.engram_layer_ids.len() != self.engram_num_embeddings.len() {
            return Err(FerriteError::Config(format!(
                "engram_layer_ids {} != engram_num_embeddings {}",
                self.engram_layer_ids.len(),
                self.engram_num_embeddings.len()
            )));
        }
        if self.engram_layer_ids.iter().any(|&l| l >= n) {
            return Err(FerriteError::Config("engram layer >= n_layers".into()));
        }
        Ok(())
    }

    // ---- derived helpers (mirror the reference's inline properties) ----

    /// `compress_ratios[layer]` (0 for pure sliding-window layers).
    /// Full low-rank width of the block-diagonal output projection:
    /// `o_groups * o_lora_rank` (each group carries its own `o_lora_rank`).
    pub fn n_groups_o_lora(&self) -> usize {
        self.o_groups * self.o_lora_rank
    }

    pub fn compress_ratio(&self, layer: usize) -> usize {
        self.compress_ratios.get(layer).copied().unwrap_or(0)
    }

    /// A backbone layer that pools its own KV and publishes it.
    pub fn is_kv_source(&self, layer: usize) -> bool {
        layer < self.n_layers && self.kv_source_layers.contains(&layer)
    }

    /// A layer that runs its own indexer (otherwise it reuses the published
    /// `topk_idxs`).
    pub fn is_index_source(&self, layer: usize) -> bool {
        layer < self.n_layers && self.index_source_layers.contains(&layer)
    }

    /// The layer that publishes coarse block candidates for the layers after it.
    pub fn is_candidate_source(&self, layer: usize) -> bool {
        self.candidate_source_layer >= 0 && layer == self.candidate_source_layer as usize
    }

    /// In the reference `Indexer.owns_k <=> layer in kv_source_layers`: only a
    /// layer producing its own latent can derive index keys from it.
    pub fn indexer_owns_k(&self, layer: usize) -> bool {
        self.is_kv_source(layer)
    }

    pub fn kv_mode(&self, layer: usize) -> KvMode {
        match self.compress_ratio(layer) {
            0 => KvMode::WindowOnly,
            _ if self.is_kv_source(layer) => KvMode::CompressSource,
            _ => KvMode::CompressConsumer,
        }
    }

    /// Routed / activated expert counts: the DSpark layers use their own.
    pub fn moe_config(&self, layer: usize) -> (usize, usize) {
        if layer < self.n_layers {
            (self.n_routed_experts, self.n_activated_experts)
        } else {
            (
                if self.dspark_n_routed_experts == 0 {
                    self.n_routed_experts
                } else {
                    self.dspark_n_routed_experts
                },
                if self.dspark_n_activated_experts == 0 {
                    self.n_activated_experts
                } else {
                    self.dspark_n_activated_experts
                },
            )
        }
    }

    pub fn engram_enabled(&self) -> bool {
        !self.engram_layer_ids.is_empty() && self.engram_max_ngram_size > 1
    }

    pub fn vision_enabled(&self) -> bool {
        self.vision_n_layers > 0
    }

    pub fn dspark_enabled(&self) -> bool {
        self.dspark_block_size > 0 && self.n_mtp_layers > 0
    }

    /// `(3 + hc_mult) * hc_mult` — a single projection produces pre (hc),
    /// post (hc) and comb (hc*hc) coefficients. The reference calls this
    /// `hc_mult3` = `hc_mult * (2 + hc_mult)`; both spellings are the same
    /// number for hc_mult 4 (24).
    pub fn hc_mix_count(&self) -> usize {
        self.hc_mult * (2 + self.hc_mult)
    }

    /// Total rows each rank owns for an engram table (rows are sharded).
    pub fn engram_part_rows(&self, world: usize, which: usize) -> u64 {
        let rows = self.engram_num_embeddings.get(which).copied().unwrap_or(0);
        rows.div_ceil(world as u64)
    }

    pub fn nope_head_dim(&self) -> usize {
        self.head_dim - self.rope_head_dim
    }

    /// Heads per rank at TP = world; o_groups per rank likewise.
    pub fn n_local_heads(&self, world: usize) -> usize {
        self.n_heads / world
    }
    pub fn n_local_groups(&self, world: usize) -> usize {
        self.o_groups / world
    }

    /// The released config, used by tests and by the weights loader when it
    /// needs the production geometry without a checkpoint present.
    pub fn production() -> Self {
        let json = include_str!("../configs/dsv41_flash.json");
        Self::from_json_str(json).expect("bundled production config must parse")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prod() -> Dsv41Config {
        Dsv41Config::production()
    }

    #[test]
    fn production_shapes() {
        let c = prod();
        assert_eq!(c.n_layers, 40);
        assert_eq!(c.n_mtp_layers, 3);
        assert_eq!(c.dim, 5120);
        assert_eq!(c.n_heads, 64);
        assert_eq!(c.head_dim, 512);
        assert_eq!(c.rope_head_dim, 64);
        assert_eq!(c.q_lora_rank, 1280);
        assert_eq!(c.o_lora_rank, 1024);
        assert_eq!(c.o_groups, 8);
        assert_eq!(c.window_size, 128);
        assert_eq!(c.moe_inter_dim, 2304);
        assert_eq!(c.n_routed_experts, 384);
        assert_eq!(c.n_activated_experts, 6);
        assert_eq!(c.vocab_size, 129280);
        assert_eq!(c.expert_dtype.as_deref(), Some("fp4"));
        assert_eq!(c.weight_dtype, "fp8");
        assert_eq!(c.fp8_block, 32);
        // hyper-connections are identical to GLM-5.3-Flash
        assert_eq!(c.hc_mult, 4);
        assert_eq!(c.hc_sinkhorn_iters, 20);
        assert_eq!(c.hc_mix_count(), 24);
    }

    #[test]
    fn layer_roles() {
        let c = prod();
        assert_eq!(c.kv_source_layers, vec![2, 8, 14, 20]);
        assert_eq!(c.index_source_layers, vec![2, 8, 14, 20, 24, 28, 32, 36]);
        assert_eq!(c.candidate_source_layer, 20);
        // window-only layers
        assert_eq!(c.kv_mode(0), KvMode::WindowOnly);
        assert_eq!(c.compress_ratio(0), 0);
        assert_eq!(c.compress_ratio(1), 0);
        // first source: ratio 2, publishes
        assert_eq!(c.kv_mode(2), KvMode::CompressSource);
        assert_eq!(c.kv_mode(3), KvMode::CompressConsumer);
        assert_eq!(c.compress_ratio(3), 2);
        // ratio-1 block: 20 is a source, 21..39 consume
        assert_eq!(c.kv_mode(20), KvMode::CompressSource);
        assert_eq!(c.compress_ratio(21), 1);
        assert_eq!(c.kv_mode(21), KvMode::CompressConsumer);
        // MTP layers are window-only
        assert_eq!(c.compress_ratio(41), 0);
        assert!(!c.is_kv_source(41));
        assert!(!c.is_index_source(41));
        assert!(c.is_index_source(24));
        assert!(c.is_candidate_source(20));
        assert!(!c.is_candidate_source(21));
        // owns_k tracks kv_source (engram/index-key ownership)
        assert!(c.indexer_owns_k(14));
        assert!(!c.indexer_owns_k(15));
    }

    #[test]
    fn moe_and_engram() {
        let c = prod();
        assert_eq!(c.moe_config(5), (384, 6));
        assert_eq!(c.moe_config(41), (128, 3));
        assert_eq!(c.engram_layer_ids, vec![1, 14]);
        assert_eq!(c.engram_num_embeddings, vec![384006168, 384016682]);
        assert_eq!(c.engram_n_heads, 8);
        assert_eq!(c.engram_head_dim, 256);
        assert_eq!(c.engram_max_ngram_size, 4);
        assert_eq!(c.engram_compressed_vocab_size, 99092);
        assert!(c.engram_enabled());
        // 24 hash columns per token: (max_ngram-1) * n_heads
        assert_eq!((c.engram_max_ngram_size - 1) * c.engram_n_heads, 24);
        // rows sharded over 8 ranks
        assert_eq!(c.engram_part_rows(8, 0), 384006168u64.div_ceil(8));
    }

    #[test]
    fn dspark_and_vision() {
        let c = prod();
        assert!(c.dspark_enabled());
        assert_eq!(c.dspark_block_size, 5);
        assert_eq!(c.dspark_target_layer_ids, vec![37, 38, 39]);
        assert_eq!(c.dspark_markov_rank, 256);
        assert_eq!(c.dspark_noise_token_id, 128799);
        assert!(c.vision_enabled());
        assert_eq!(c.vision_n_layers, 32);
        assert_eq!(c.vision_patch_size, 14);
        assert_eq!(c.vision_downsample_ratio, 3);
        assert_eq!(c.image_token_id, 129264);
    }

    #[test]
    fn flat_reference_config_parses() {
        // the layout shipped in inference/config.json (no nesting)
        let json = include_str!("../configs/dsv41_reference_flat.json");
        let c = Dsv41Config::from_json_str(json).unwrap();
        assert_eq!(c.dim, 5120);
        assert_eq!(c.n_layers, 40);
        assert_eq!(c.n_mtp_layers, 3);
        assert_eq!(c.implied_moe_inter(), 2304);
    }

    impl Dsv41Config {
        fn implied_moe_inter(&self) -> usize {
            self.moe_inter_dim
        }
    }
}
