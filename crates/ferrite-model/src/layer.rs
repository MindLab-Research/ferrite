//! Static per-layer plan derived from the config — the backbone of
//! ferrite's compile-time-specialisation philosophy: layer roles are decided
//! once at engine-build time, never per forward.

use crate::config::{Glm53FlashConfig, LayerType, MlpType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnKind {
    /// Gated DeltaNet linear attention (fixed-size recurrent state).
    Linear,
    /// DeepSeek-style sparse attention (latent KV + indexer top-k).
    Dsa,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpKind {
    Dense,
    Moe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerPlan {
    pub layer_idx: usize,
    pub attn: AttnKind,
    pub mlp: MlpKind,
}

impl LayerPlan {
    /// Fixed-size recurrent state bytes per sequence (linear layers only).
    pub fn linear_state_elems(&self, cfg: &Glm53FlashConfig) -> Option<usize> {
        match self.attn {
            AttnKind::Linear => Some(
                cfg.linear_attn.num_heads
                    * cfg.linear_attn.head_dim
                    * cfg.linear_attn.head_dim,
            ),
            AttnKind::Dsa => None,
        }
    }
}

/// Build the full static layer table: 45 entries describing which kernel
/// family runs at each position, plus FFN kind.
pub fn build_layer_plans(cfg: &Glm53FlashConfig) -> Vec<LayerPlan> {
    cfg.layer_types
        .iter()
        .zip(cfg.mlp_types.iter())
        .enumerate()
        .map(|(i, (lt, mt))| LayerPlan {
            layer_idx: i,
            attn: match lt {
                LayerType::LinearAttention => AttnKind::Linear,
                LayerType::DeepseekSparseAttention => AttnKind::Dsa,
            },
            mlp: match mt {
                MlpType::Dense => MlpKind::Dense,
                MlpType::Sparse => MlpKind::Moe,
            },
        })
        .collect()
}

/// Count of each attention family (sanity for pool sizing).
pub fn attn_kind_counts(cfg: &Glm53FlashConfig) -> (usize, usize) {
    let plans = build_layer_plans(cfg);
    let lin = plans.iter().filter(|p| p.attn == AttnKind::Linear).count();
    let dsa = plans.iter().filter(|p| p.attn == AttnKind::Dsa).count();
    (lin, dsa)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Glm53FlashConfig;

    #[test]
    fn layer_plan_test_config() {
        let cfg = Glm53FlashConfig::test_config();
        let plans = build_layer_plans(&cfg);
        assert_eq!(plans.len(), 4);
        assert_eq!(plans[0].attn, AttnKind::Linear);
        assert_eq!(plans[1].attn, AttnKind::Dsa);
        assert_eq!(plans[0].mlp, MlpKind::Dense);
        assert_eq!(plans[2].mlp, MlpKind::Moe);
        let (lin, dsa) = attn_kind_counts(&cfg);
        assert_eq!((lin, dsa), (2, 2));
    }

    #[test]
    fn layer_plan_production() {
        let cfg = Glm53FlashConfig::production_config();
        let (lin, dsa) = attn_kind_counts(&cfg);
        assert_eq!((lin, dsa), (34, 11), "34 linear + 11 DSA");
        let plans = build_layer_plans(&cfg);
        assert_eq!(plans[3].attn, AttnKind::Dsa);
        assert_eq!(plans[43].attn, AttnKind::Dsa);
        assert_eq!(plans[44].attn, AttnKind::Linear);
        // first 3 dense then 42 MoE
        assert_eq!(plans[0].mlp, MlpKind::Dense);
        assert_eq!(plans[2].mlp, MlpKind::Dense);
        assert_eq!(plans[3].mlp, MlpKind::Moe);
    }
}
