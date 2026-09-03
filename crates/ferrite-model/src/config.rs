//! GLM-5.3-Flash (glm5_next_text) configuration, parsed from the HF
//! config.json `text_config` section. Captured from the real checkpoint on
//! the B300 cluster (45 layers: 34 GatedDeltaNet linear + 11 DSA;
//! first 3 dense FFN then 42 MoE; MHC hyper-connections; nope-only MLA).

use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    DeepseekSparseAttention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlpType {
    Dense,
    Sparse,
}

/// Raw serde view of the HF `text_config` (names match config.json 1:1).
#[derive(Debug, Clone, Deserialize)]
pub struct RawTextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    #[serde(default = "d_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "d_swiglu_limit")]
    pub swiglu_limit: f32,
    pub max_position_embeddings: usize,
    pub layer_types: Vec<LayerType>,
    pub mlp_layer_types: Vec<MlpType>,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default = "d_routed_scaling")]
    pub routed_scaling_factor: f32,
    // linear attention (Gated DeltaNet / KDA)
    pub linear_attn_config: RawLinearAttnConfig,
    // DSA (nope-only MLA + indexer)
    pub num_attention_heads: usize,
    pub kv_lora_rank: usize,
    pub q_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    #[serde(default)]
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    #[serde(default = "d_index_kpool")]
    pub index_kpool: usize,
    // hyper-connections
    #[serde(default)]
    pub mhc: bool,
    #[serde(default = "d_hc_mult")]
    pub hc_mult: usize,
    #[serde(default = "d_hc_eps")]
    pub hc_eps: f32,
    #[serde(default = "d_hc_sinkhorn")]
    pub hc_sinkhorn_iters: usize,
    // MTP (nextn / DFlash2 draft)
    #[serde(default)]
    pub num_nextn_predict_layers: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawLinearAttnConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    #[serde(default = "d_gate_lower_bound")]
    pub gate_lower_bound: f32,
    #[serde(default = "d_short_conv")]
    pub short_conv_kernel_size: usize,
}

fn d_rms_eps() -> f32 {
    1e-5
}
fn d_swiglu_limit() -> f32 {
    10.0
}
fn d_routed_scaling() -> f32 {
    2.5
}
fn d_gate_lower_bound() -> f32 {
    -5.0
}
fn d_short_conv() -> usize {
    4
}
fn d_hc_mult() -> usize {
    4
}
fn d_hc_eps() -> f32 {
    1e-6
}
fn d_hc_sinkhorn() -> usize {
    20
}
fn d_index_kpool() -> usize {
    4
}

/// Normalised strong-typed config (post-validation).
#[derive(Debug, Clone)]
pub struct Glm53FlashConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub swiglu_limit: f32,
    pub max_position_embeddings: usize,
    pub layer_types: Vec<LayerType>,
    pub mlp_types: Vec<MlpType>,
    // FFN
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: f32,
    // linear attention
    pub linear_attn: LinearAttnConfig,
    // DSA
    pub dsa: DsaConfig,
    // hyper-connections
    pub mhc: bool,
    pub hc_mult: usize,
    pub hc_eps: f32,
    pub hc_sinkhorn_iters: usize,
    // MTP
    pub num_nextn_predict_layers: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearAttnConfig {
    /// GQA-style head count of the linear attention layers (64).
    pub num_heads: usize,
    /// Per-head key/value dim of the recurrent state (128).
    pub head_dim: usize,
    /// Lower clamp for the gating logit (A), applied as exp(clamp) (−5.0).
    pub gate_lower_bound: f32,
    /// Causal short conv on q/k/v before the delta rule (4).
    pub short_conv_kernel_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DsaConfig {
    /// Number of query heads of DSA layers (64).
    pub num_attention_heads: usize,
    /// Latent KV rank (512) — nope-only: qk_rope_head_dim is 0 for 5.3-Flash.
    pub kv_lora_rank: usize,
    pub q_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    /// Indexer heads (32) & dims (128) — top-2048 page selection.
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub index_kpool: usize,
}

impl DsaConfig {
    /// Latent KV bytes per token per layer: kv_lora_rank (+ rope, 0 here).
    /// Note: nope-only means the *entire* latent is compressible — no
    /// rope segment that must stay raw (unlike GLM-5.2's 512+64).
    pub fn kv_latent_dim(&self) -> usize {
        self.kv_lora_rank + self.qk_rope_head_dim
    }
}

impl Glm53FlashConfig {
    pub fn from_text_config(raw: RawTextConfig) -> ferrite_types::Result<Self> {
        let n = raw.num_hidden_layers;
        if raw.layer_types.len() != n || raw.mlp_layer_types.len() != n {
            return Err(ferrite_types::FerriteError::Config(format!(
                "layer_types len {} / mlp_layer_types len {} != num_hidden_layers {}",
                raw.layer_types.len(),
                raw.mlp_layer_types.len(),
                n
            )));
        }
        let cfg = Glm53FlashConfig {
            hidden_size: raw.hidden_size,
            num_hidden_layers: n,
            vocab_size: raw.vocab_size,
            rms_norm_eps: raw.rms_norm_eps,
            swiglu_limit: raw.swiglu_limit,
            max_position_embeddings: raw.max_position_embeddings,
            layer_types: raw.layer_types,
            mlp_types: raw.mlp_layer_types,
            intermediate_size: raw.intermediate_size,
            moe_intermediate_size: raw.moe_intermediate_size,
            n_routed_experts: raw.n_routed_experts,
            n_shared_experts: raw.n_shared_experts,
            num_experts_per_tok: raw.num_experts_per_tok,
            routed_scaling_factor: raw.routed_scaling_factor,
            linear_attn: LinearAttnConfig {
                num_heads: raw.linear_attn_config.num_heads,
                head_dim: raw.linear_attn_config.head_dim,
                gate_lower_bound: raw.linear_attn_config.gate_lower_bound,
                short_conv_kernel_size: raw.linear_attn_config.short_conv_kernel_size,
            },
            dsa: DsaConfig {
                num_attention_heads: raw.num_attention_heads,
                kv_lora_rank: raw.kv_lora_rank,
                q_lora_rank: raw.q_lora_rank,
                qk_nope_head_dim: raw.qk_nope_head_dim,
                qk_rope_head_dim: raw.qk_rope_head_dim,
                v_head_dim: raw.v_head_dim,
                index_n_heads: raw.index_n_heads,
                index_head_dim: raw.index_head_dim,
                index_topk: raw.index_topk,
                index_kpool: raw.index_kpool,
            },
            mhc: raw.mhc,
            hc_mult: raw.hc_mult,
            hc_eps: raw.hc_eps,
            hc_sinkhorn_iters: raw.hc_sinkhorn_iters,
            num_nextn_predict_layers: raw.num_nextn_predict_layers,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Parse from a full GLM-5.3-Flash config.json (top level object with
    /// `text_config`).
    pub fn from_json_str(s: &str) -> ferrite_types::Result<Self> {
        #[derive(Deserialize)]
        struct Top {
            text_config: RawTextConfig,
        }
        let top: Top = serde_json::from_str(s)
            .map_err(|e| ferrite_types::FerriteError::Config(format!("config.json parse: {e}")))?;
        Self::from_text_config(top.text_config)
    }

    fn validate(&self) -> ferrite_types::Result<()> {
        let h = &self.linear_attn;
        if h.num_heads == 0 || h.head_dim == 0 {
            return Err(ferrite_types::FerriteError::Config("linear attn heads/head_dim = 0".into()));
        }
        let d = &self.dsa;
        if d.kv_lora_rank == 0 || d.qk_nope_head_dim == 0 {
            return Err(ferrite_types::FerriteError::Config("dsa latent dims = 0".into()));
        }
        Ok(())
    }

    /// Small deterministic config for CPU smoke tests:
    /// 4 layers [lin, dsa, lin, dsa], hidden 128, tiny heads — fast enough to
    /// run the whole engine in debug builds.
    pub fn test_config() -> Self {
        let layer_types = vec![
            LayerType::LinearAttention,
            LayerType::DeepseekSparseAttention,
            LayerType::LinearAttention,
            LayerType::DeepseekSparseAttention,
        ];
        let mlp_types = vec![
            MlpType::Dense,
            MlpType::Dense,
            MlpType::Sparse,
            MlpType::Sparse,
        ];
        Glm53FlashConfig {
            hidden_size: 128,
            num_hidden_layers: 4,
            vocab_size: 512,
            rms_norm_eps: 1e-5,
            swiglu_limit: 10.0,
            max_position_embeddings: 4096,
            layer_types: layer_types.clone(),
            mlp_types,
            intermediate_size: 256,
            moe_intermediate_size: 64,
            n_routed_experts: 8,
            n_shared_experts: 1,
            num_experts_per_tok: 2,
            routed_scaling_factor: 2.5,
            linear_attn: LinearAttnConfig {
                num_heads: 4,
                head_dim: 32,
                gate_lower_bound: -5.0,
                short_conv_kernel_size: 4,
            },
            dsa: DsaConfig {
                num_attention_heads: 4,
                kv_lora_rank: 64,
                q_lora_rank: 128,
                qk_nope_head_dim: 32,
                qk_rope_head_dim: 0,
                v_head_dim: 32,
                index_n_heads: 2,
                index_head_dim: 16,
                index_topk: 128,
                index_kpool: 2,
            },
            mhc: true,
            hc_mult: 4,
            hc_eps: 1e-6,
            hc_sinkhorn_iters: 20,
            num_nextn_predict_layers: 0,
        }
        .with_layer_types(layer_types)
    }

    fn with_layer_types(mut self, lt: Vec<LayerType>) -> Self {
        self.layer_types = lt;
        self
    }

    /// Real GLM-5.3-Flash production config (from the 1102 checkpoint).
    pub fn production_config() -> Self {
        let mut layer_types = Vec::with_capacity(45);
        let mut mlp_types = Vec::with_capacity(45);
        for i in 0..45 {
            // pattern: 3 linear then 1 DSA
            layer_types.push(if i % 4 == 3 {
                LayerType::DeepseekSparseAttention
            } else {
                LayerType::LinearAttention
            });
            mlp_types.push(if i < 3 { MlpType::Dense } else { MlpType::Sparse });
        }
        Glm53FlashConfig {
            hidden_size: 4096,
            num_hidden_layers: 45,
            vocab_size: 154880,
            rms_norm_eps: 1e-5,
            swiglu_limit: 10.0,
            max_position_embeddings: 1048576,
            layer_types: layer_types.clone(),
            mlp_types,
            intermediate_size: 12288,
            moe_intermediate_size: 2048,
            n_routed_experts: 288,
            n_shared_experts: 1,
            num_experts_per_tok: 8,
            routed_scaling_factor: 2.5,
            linear_attn: LinearAttnConfig {
                num_heads: 64,
                head_dim: 128,
                gate_lower_bound: -5.0,
                short_conv_kernel_size: 4,
            },
            dsa: DsaConfig {
                num_attention_heads: 64,
                kv_lora_rank: 512,
                q_lora_rank: 1536,
                qk_nope_head_dim: 256,
                qk_rope_head_dim: 0,
                v_head_dim: 256,
                index_n_heads: 32,
                index_head_dim: 128,
                index_topk: 2048,
                index_kpool: 4,
            },
            mhc: true,
            hc_mult: 4,
            hc_eps: 1e-6,
            hc_sinkhorn_iters: 20,
            num_nextn_predict_layers: 1,
        }
        .with_layer_types(layer_types)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_shapes() {
        let c = Glm53FlashConfig::test_config();
        assert_eq!(c.num_hidden_layers, 4);
        assert_eq!(c.layer_types.len(), 4);
        assert!(matches!(c.layer_types[1], LayerType::DeepseekSparseAttention));
        assert_eq!(c.mlp_types[0], MlpType::Dense);
        assert_eq!(c.mlp_types[2], MlpType::Sparse);
        // linear state per layer: heads * head_k * head_v
        assert_eq!(c.linear_attn.num_heads * c.linear_attn.head_dim * c.linear_attn.head_dim, 4 * 32 * 32);
    }

    #[test]
    fn production_config_matches_checkpoint() {
        let c = Glm53FlashConfig::production_config();
        let dsa_count = c
            .layer_types
            .iter()
            .filter(|t| matches!(t, LayerType::DeepseekSparseAttention))
            .count();
        assert_eq!(dsa_count, 11, "GLM-5.3-Flash has 11 DSA layers");
        assert_eq!(c.dsa.kv_lora_rank, 512);
        assert_eq!(c.dsa.qk_rope_head_dim, 0, "nope-only MLA");
        assert_eq!(c.n_routed_experts, 288);
        // DSA layers at 3,7,...,43
        assert!(matches!(c.layer_types[3], LayerType::DeepseekSparseAttention));
        assert!(matches!(c.layer_types[43], LayerType::DeepseekSparseAttention));
        assert!(matches!(c.layer_types[44], LayerType::LinearAttention));
    }

    #[test]
    fn parse_json_roundtrip() {
        let c = Glm53FlashConfig::test_config();
        // minimal valid json with text_config
        let json = format!(
            r#"{{"text_config": {{
                "hidden_size": 64, "num_hidden_layers": 2, "vocab_size": 128,
                "max_position_embeddings": 1024,
                "layer_types": ["linear_attention", "deepseek_sparse_attention"],
                "mlp_layer_types": ["dense", "sparse"],
                "intermediate_size": 128, "moe_intermediate_size": 32,
                "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
                "linear_attn_config": {{"num_heads": 2, "head_dim": 16}},
                "num_attention_heads": 2, "kv_lora_rank": 32, "q_lora_rank": 64,
                "qk_nope_head_dim": 16, "v_head_dim": 16,
                "index_n_heads": 1, "index_head_dim": 8, "index_topk": 64
            }}}}"#
        );
        let parsed = Glm53FlashConfig::from_json_str(&json).unwrap();
        assert_eq!(parsed.hidden_size, 64);
        assert_eq!(parsed.linear_attn.short_conv_kernel_size, 4, "default conv=4");
        assert_eq!(parsed.dsa.qk_rope_head_dim, 0);
        assert_eq!(parsed.dsa.index_kpool, 4, "default kpool");
        let _ = c;
    }
}
