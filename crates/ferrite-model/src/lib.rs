//! ferrite-model: static model definition for GLM-5.3-Flash (glm5_next_text).
//!
//! The model structure is fully known at load time — this crate turns a
//! HF-style config.json into a static layer plan (which layer is linear
//! attention vs DSA, dense vs MoE FFN), a weight layout (name → shape),
//! and provides a safetensors loader. Everything downstream (kernel
//! specialisation, memory planning, PDAF scheduling) is derived from
//! these static structures at *compile-of-engine* time, not per forward.

mod checkpoint;
mod config;
mod layer;
mod safetensors;
mod weights;

pub use checkpoint::{load_hf_checkpoint, CheckpointReport};
pub use config::{DsaConfig, Glm53FlashConfig, LayerType, LinearAttnConfig, MlpType};
pub use layer::{build_layer_plans, AttnKind, LayerPlan, MlpKind};
pub use safetensors::{load_safetensors_dir, load_safetensors_file};
pub use weights::{apply_fused_aliases, random_weights, weight_layout, Fp8Weight, WeightLayout, Weights, Weights8};

/// Model identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    /// GLM-5.3-Flash: hybrid GatedDeltaNet + DSA + (dense|MoE) + MHC.
    Glm53Flash,
}

impl Glm53FlashConfig {
    /// True if layer `idx` is a DSA (full sparse-attention) layer.
    pub fn is_dsa_layer(&self, idx: usize) -> bool {
        matches!(self.layer_kind(idx), AttnKind::Dsa)
    }

    /// Kind of attention at layer `idx`.
    pub fn layer_kind(&self, idx: usize) -> AttnKind {
        match self.layer_types[idx] {
            config::LayerType::LinearAttention => AttnKind::Linear,
            config::LayerType::DeepseekSparseAttention => AttnKind::Dsa,
        }
    }
}
