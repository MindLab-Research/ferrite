//! Weight layout for GLM-5.3-Flash: the static name→shape table generated
//! from the config, plus deterministic random init for CPU smoke tests.
//!
//! Naming follows the HF/sglang `glm5_next` conventions so the safetensors
//! checkpoint loads 1:1. The unfused projection names are used; the fused
//! `fused_qkvbfg_a_proj` checkpoint variant maps onto the same logical
//! weights during load (TODO: fused-name aliasing when loading the real
//! B300 checkpoint).

use std::collections::HashMap;

use ferrite_types::{DType, Shape, Tensor};

use crate::config::Glm53FlashConfig;
use crate::layer::{AttnKind, MlpKind, build_layer_plans};

/// A named weight with its logical shape.
#[derive(Debug, Clone)]
pub struct WeightSpec {
    pub name: String,
    pub shape: Shape,
}

/// Full weight layout: ordered specs + total element count.
#[derive(Debug, Clone)]
pub struct WeightLayout {
    pub specs: Vec<WeightSpec>,
    pub total_elems: usize,
}

impl WeightLayout {
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.specs.iter().map(|s| s.name.as_str())
    }
}

/// Loaded/init weight set: name → tensor (f32 storage).
pub type Weights = HashMap<String, Tensor>;

fn spec(name: &str, dims: impl IntoIterator<Item = usize>) -> WeightSpec {
    WeightSpec { name: name.to_string(), shape: Shape::new(dims) }
}

fn push_layer_spans(dst: &mut Vec<WeightSpec>, pfx: &str, cfg: &Glm53FlashConfig) {
    let h = cfg.hidden_size;
    dst.push(spec(&format!("{pfx}.input_layernorm.weight"), [h]));
    if cfg.mhc {
        // Hyper-connection params (float32 in the checkpoint).
        // Shapes per sglang Glm5NextDecoderLayer:
        //   mix_hc = (2 + hc_mult) * hc_mult   (pre n + post n + comb n*n)
        //   hc_dim = hc_mult * hidden          (fn's input dim)
        //   base [mix_hc], scale [3], fn [mix_hc, hc_dim]
        let mix = (2 + cfg.hc_mult) * cfg.hc_mult;
        let hc_dim = cfg.hc_mult * cfg.hidden_size;
        dst.push(spec(&format!("{pfx}.hc_attn_base"), [mix]));
        dst.push(spec(&format!("{pfx}.hc_attn_scale"), [3]));
        dst.push(spec(&format!("{pfx}.hc_attn_fn"), [mix, hc_dim]));
        dst.push(spec(&format!("{pfx}.hc_ffn_base"), [mix]));
        dst.push(spec(&format!("{pfx}.hc_ffn_scale"), [3]));
        dst.push(spec(&format!("{pfx}.hc_ffn_fn"), [mix, hc_dim]));
    } else {
        dst.push(spec(&format!("{pfx}.post_attention_layernorm.weight"), [h]));
    }
}

fn push_linear_attn_spans(dst: &mut Vec<WeightSpec>, pfx: &str, cfg: &Glm53FlashConfig) {
    let la = &cfg.linear_attn;
    let h = cfg.hidden_size;
    let proj = la.num_heads * la.head_dim;
    let c = la.short_conv_kernel_size;
    // projections (unfused names)
    dst.push(spec(&format!("{pfx}.qkv_proj.weight"), [3 * proj, h]));
    dst.push(spec(&format!("{pfx}.f_a_proj.weight"), [la.head_dim, h]));
    dst.push(spec(&format!("{pfx}.f_b_proj.weight"), [proj, la.head_dim]));
    dst.push(spec(&format!("{pfx}.b_proj.weight"), [la.num_heads, h]));
    dst.push(spec(&format!("{pfx}.g_a_proj.weight"), [la.head_dim, h]));
    dst.push(spec(&format!("{pfx}.g_b_proj.weight"), [proj, la.head_dim]));
    // recurrent-state params
    dst.push(spec(&format!("{pfx}.A_log"), [la.num_heads]));
    dst.push(spec(&format!("{pfx}.dt_bias"), [3 * proj]));
    // short causal conv on q/k/v (conv1d, in-channels = 3*proj, kernel c)
    dst.push(spec(&format!("{pfx}.qkv_conv1d.weight"), [3 * proj, c]));
    // output norm (RMSNorm with gate over heads*head_dim)
    dst.push(spec(&format!("{pfx}.o_norm.weight"), [proj]));
    dst.push(spec(&format!("{pfx}.o_proj.weight"), [h, proj]));
}

fn push_dsa_attn_spans(dst: &mut Vec<WeightSpec>, pfx: &str, cfg: &Glm53FlashConfig) {
    let d = &cfg.dsa;
    let h = cfg.hidden_size;
    let nh = d.num_attention_heads;
    // nope-only MLA: q compressed via q_lora_rank then up-projected to nope
    dst.push(spec(&format!("{pfx}.q_a_proj.weight"), [d.q_lora_rank, h]));
    dst.push(spec(&format!("{pfx}.q_a_layernorm.weight"), [d.q_lora_rank]));
    dst.push(spec(&format!("{pfx}.q_b_proj.weight"), [nh * d.qk_nope_head_dim, d.q_lora_rank]));
    // latent KV down then up (latent contains K,V after absorb)
    dst.push(spec(&format!("{pfx}.kv_a_proj_with_mqa.weight"), [d.kv_latent_dim(), h]));
    dst.push(spec(&format!("{pfx}.kv_a_layernorm.weight"), [d.kv_lora_rank]));
    dst.push(spec(
        &format!("{pfx}.kv_b_proj.weight"),
        [nh * (d.qk_nope_head_dim + d.v_head_dim), d.kv_lora_rank],
    ));
    // indexer (lightweight q/k for top-k page selection)
    let iproj = d.index_n_heads * d.index_head_dim;
    dst.push(spec(&format!("{pfx}.indexer_q_proj.weight"), [iproj, h]));
    dst.push(spec(&format!("{pfx}.indexer_k_proj.weight"), [iproj, h]));
    dst.push(spec(&format!("{pfx}.indexer_norm.weight"), [iproj]));
    dst.push(spec(&format!("{pfx}.o_proj.weight"), [h, nh * d.v_head_dim]));
}

fn push_dense_mlp_spans(dst: &mut Vec<WeightSpec>, pfx: &str, cfg: &Glm53FlashConfig) {
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    dst.push(spec(&format!("{pfx}.gate_proj.weight"), [i, h]));
    dst.push(spec(&format!("{pfx}.up_proj.weight"), [i, h]));
    dst.push(spec(&format!("{pfx}.down_proj.weight"), [h, i]));
}

fn push_moe_mlp_spans(dst: &mut Vec<WeightSpec>, pfx: &str, cfg: &Glm53FlashConfig) {
    let h = cfg.hidden_size;
    let i = cfg.moe_intermediate_size;
    dst.push(spec(&format!("{pfx}.gate.weight"), [cfg.n_routed_experts, h]));
    if cfg.n_shared_experts > 0 {
        dst.push(spec(&format!("{pfx}.shared_expert.gate_proj.weight"), [i, h]));
        dst.push(spec(&format!("{pfx}.shared_expert.up_proj.weight"), [i, h]));
        dst.push(spec(&format!("{pfx}.shared_expert.down_proj.weight"), [h, i]));
    }
    for e in 0..cfg.n_routed_experts {
        dst.push(spec(&format!("{pfx}.experts.{e}.gate_proj.weight"), [i, h]));
        dst.push(spec(&format!("{pfx}.experts.{e}.up_proj.weight"), [i, h]));
        dst.push(spec(&format!("{pfx}.experts.{e}.down_proj.weight"), [h, i]));
    }
}

/// Build the full layout (embed, per-layer, final norm, lm_head).
pub fn weight_layout(cfg: &Glm53FlashConfig) -> WeightLayout {
    let h = cfg.hidden_size;
    let mut specs = Vec::new();
    specs.push(spec("model.embed_tokens.weight", [cfg.vocab_size, h]));
    for plan in build_layer_plans(cfg) {
        let pfx = format!("model.layers.{}", plan.layer_idx);
        push_layer_spans(&mut specs, &pfx, cfg);
        let attn_pfx = format!("{pfx}.self_attn");
        match plan.attn {
            AttnKind::Linear => push_linear_attn_spans(&mut specs, &attn_pfx, cfg),
            AttnKind::Dsa => push_dsa_attn_spans(&mut specs, &attn_pfx, cfg),
        }
        let mlp_pfx = format!("{pfx}.mlp");
        match plan.mlp {
            MlpKind::Dense => push_dense_mlp_spans(&mut specs, &mlp_pfx, cfg),
            MlpKind::Moe => push_moe_mlp_spans(&mut specs, &mlp_pfx, cfg),
        }
    }
    specs.push(spec("model.norm.weight", [h]));
    specs.push(spec("lm_head.weight", [cfg.vocab_size, h]));
    let total_elems: usize = specs.iter().map(|s| s.shape.numel()).sum();
    WeightLayout { specs, total_elems }
}

/// Deterministic PRNG (xorshift64*) — reproducible across runs/platforms.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// Uniform in (-1, 1).
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    /// Normal-ish via Box-Muller (deterministic, good enough for smoke tests).
    fn next_normal(&mut self, scale: f32) -> f32 {
        let u1 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 + 1e-12;
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() as f32 * (2.0 * std::f32::consts::PI * u2 as f32).cos() * scale
    }
}

/// Generate deterministic random weights for every spec in the layout
/// (CPU smoke tests / engine bring-up without a real checkpoint).
pub fn random_weights(cfg: &Glm53FlashConfig, seed: u64) -> Weights {
    let layout = weight_layout(cfg);
    let mut rng = Rng(seed | 1);
    let mut out = Weights::with_capacity(layout.specs.len());
    for spec in &layout.specs {
        let n = spec.shape.numel();
        // Scale by fan-in for stability (Xavier-ish).
        let fan_in = match spec.shape.rank() {
            0 | 1 => 1.0,
            _ => spec.shape.0[spec.shape.rank() - 1] as f32,
        };
        let scale = (2.0 / fan_in).sqrt().min(0.5);
        let is_bias_like = spec.name.ends_with("A_log")
            || spec.name.ends_with("dt_bias")
            || spec.name.ends_with("layernorm.weight")
            || spec.name.ends_with("o_norm.weight")
            || spec.name.ends_with("indexer_norm.weight")
            || spec.name.ends_with("hc_attn_base")
            || spec.name.ends_with("hc_ffn_base");
        let data: Vec<f32> = if is_bias_like {
            vec![0.0; n]
        } else {
            (0..n).map(|_| rng.next_normal(scale)).collect()
        };
        out.insert(spec.name.clone(), Tensor::new(spec.shape.clone(), DType::Bf16, data));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Glm53FlashConfig;

    #[test]
    fn layout_test_config() {
        let cfg = Glm53FlashConfig::test_config();
        let l = weight_layout(&cfg);
        let names: Vec<&str> = l.names().collect();
        assert!(names.contains(&"model.embed_tokens.weight"));
        assert!(names.contains(&"model.layers.0.self_attn.qkv_proj.weight"));
        assert!(names.contains(&"model.layers.1.self_attn.kv_a_proj_with_mqa.weight"));
        assert!(names.contains(&"model.layers.0.mlp.gate_proj.weight"));
        assert!(names.contains(&"model.layers.2.mlp.gate.weight"));
        assert!(names.contains(&"model.layers.2.mlp.experts.7.down_proj.weight"));
        assert!(names.contains(&"lm_head.weight"));
        // linear layer has conv + A_log + dt_bias
        assert!(names.contains(&"model.layers.0.self_attn.qkv_conv1d.weight"));
        assert!(names.contains(&"model.layers.0.self_attn.A_log"));
        // mhc params present
        assert!(names.contains(&"model.layers.0.hc_attn_base"));
    }

    #[test]
    fn layout_production_counts() {
        let cfg = Glm53FlashConfig::production_config();
        let l = weight_layout(&cfg);
        let n = l.names().count();
        // 3 top-level (embed/norm/lm_head)
        // + linear layer (34x): norm+mhc 7 + attn 11
        // + dsa layer (11x): norm+mhc 7 + attn 10
        // + dense mlp (3x): 3
        // + moe layer (42x): gate + 3 shared + 288*3 experts = 868
        let expect = 3 + 34 * (7 + 11) + 11 * (7 + 10) + 3 * 3 + 42 * (1 + 3 + cfg.n_routed_experts * 3);
        assert_eq!(n, expect, "weight name count");
    }

    #[test]
    fn random_weights_deterministic() {
        let cfg = Glm53FlashConfig::test_config();
        let w1 = random_weights(&cfg, 42);
        let w2 = random_weights(&cfg, 42);
        let t1 = &w1["model.layers.0.self_attn.qkv_proj.weight"];
        let t2 = &w2["model.layers.0.self_attn.qkv_proj.weight"];
        assert_eq!(t1.as_slice(), t2.as_slice());
        assert!(t1.as_slice().iter().any(|v| *v != 0.0));
    }
}

/// Apply fused-weight aliases: real GLM-5.3-Flash checkpoints store fused
/// projections (sglang `packed_modules_mapping`); ferrite's layout uses the
/// unfused names — this splits loaded fused tensors into them.
///
/// Mappings (per sglang Glm5NextForConditionalGeneration):
/// - `fused_qkvbfg_a_proj` [3*proj + heads + 2*head_dim, hidden] →
///   `qkv_proj` + `b_proj` + `f_a_proj` + `g_a_proj`
/// - `fused_fg_b_proj` [2, proj, head_dim] → `f_b_proj` + `g_b_proj`
/// - `fused_qkv_a_proj_with_mqa` → `q_a_proj` + `kv_a_proj_with_mqa` (DSA)
/// - `gate_up_proj` → `gate_proj` + `up_proj` (dense FFN / shared expert)
///
/// Call after `load_safetensors_dir` and before `Engine::new`.
pub fn apply_fused_aliases(weights: &mut Weights, cfg: &Glm53FlashConfig) {
    let hidden = cfg.hidden_size;
    let proj = cfg.linear_attn.num_heads * cfg.linear_attn.head_dim;
    let heads = cfg.linear_attn.num_heads;
    let head_dim = cfg.linear_attn.head_dim;
    let q_lora = cfg.dsa.q_lora_rank;
    let kv_latent = cfg.dsa.kv_latent_dim();

    for layer in 0..cfg.num_hidden_layers {
        let pfx = format!("model.layers.{layer}");

        // 1. linear-attn fused qkv+b+fa+ga → qkv_proj + b_proj + f_a_proj + g_a_proj
        if let Some(fused) =
            weights.remove(&format!("{pfx}.self_attn.fused_qkvbfg_a_proj.weight"))
        {
            let total_qkv = 3 * proj;
            let w = fused.as_slice().to_vec();
            let split_row = |start: usize, end: usize| -> Vec<f32> {
                w[start * hidden..end * hidden].to_vec()
            };
            weights.insert(
                format!("{pfx}.self_attn.qkv_proj.weight"),
                Tensor::from_f32(
                    Shape::new([total_qkv, hidden]),
                    split_row(0, total_qkv),
                ),
            );
            weights.insert(
                format!("{pfx}.self_attn.b_proj.weight"),
                Tensor::from_f32(Shape::new([heads, hidden]), split_row(total_qkv, total_qkv + heads)),
            );
            weights.insert(
                format!("{pfx}.self_attn.f_a_proj.weight"),
                Tensor::from_f32(
                    Shape::new([head_dim, hidden]),
                    split_row(total_qkv + heads, total_qkv + heads + head_dim),
                ),
            );
            weights.insert(
                format!("{pfx}.self_attn.g_a_proj.weight"),
                Tensor::from_f32(
                    Shape::new([head_dim, hidden]),
                    split_row(total_qkv + heads + head_dim, total_qkv + heads + 2 * head_dim),
                ),
            );
        }

        // 2. fused fg_b [2, proj, head_dim] (batched) → f_b_proj + g_b_proj
        if let Some(fused) = weights.remove(&format!("{pfx}.self_attn.fused_fg_b_proj.weight")) {
            let total = proj * head_dim;
            let flat = fused.as_slice().to_vec();
            // [2, proj, head_dim] batched or [2*proj, head_dim] flat: f_b first half
            weights.insert(
                format!("{pfx}.self_attn.f_b_proj.weight"),
                Tensor::from_f32(Shape::new([proj, head_dim]), flat[..total].to_vec()),
            );
            weights.insert(
                format!("{pfx}.self_attn.g_b_proj.weight"),
                Tensor::from_f32(Shape::new([proj, head_dim]), flat[total..2 * total].to_vec()),
            );
        }

        // 3. DSA fused q_a + kv_a → q_a_proj + kv_a_proj_with_mqa
        if let Some(fused) =
            weights.remove(&format!("{pfx}.self_attn.fused_qkv_a_proj_with_mqa.weight"))
        {
            let w = fused.as_slice().to_vec();
            weights.insert(
                format!("{pfx}.self_attn.q_a_proj.weight"),
                Tensor::from_f32(Shape::new([q_lora, hidden]), w[..q_lora * hidden].to_vec()),
            );
            weights.insert(
                format!("{pfx}.self_attn.kv_a_proj_with_mqa.weight"),
                Tensor::from_f32(
                    Shape::new([kv_latent, hidden]),
                    w[q_lora * hidden..(q_lora + kv_latent) * hidden].to_vec(),
                ),
            );
        }

        // 4. dense-FFN / shared-expert gate_up → gate_proj + up_proj
        for mlp_pfx in [format!("{pfx}.mlp"), format!("{pfx}.mlp.shared_expert")] {
            if let Some(fused) = weights.remove(&format!("{mlp_pfx}.gate_up_proj.weight")) {
                let inter = fused.shape.0[0] / 2;
                let w = fused.as_slice().to_vec();
                weights.insert(
                    format!("{mlp_pfx}.gate_proj.weight"),
                    Tensor::from_f32(Shape::new([inter, hidden]), w[..inter * hidden].to_vec()),
                );
                weights.insert(
                    format!("{mlp_pfx}.up_proj.weight"),
                    Tensor::from_f32(
                        Shape::new([inter, hidden]),
                        w[inter * hidden..2 * inter * hidden].to_vec(),
                    ),
                );
            }
        }
    }
}

#[cfg(test)]
mod fused_alias_tests {
    use super::*;

    #[test]
    fn fused_qkvbfg_splits_correctly() {
        let cfg = Glm53FlashConfig::test_config();
        let mut w = Weights::new();
        let (proj, heads, head_dim, hidden) = (128, 4, 32, 128); // test config dims
        let total_rows = 3 * proj + heads + 2 * head_dim;
        // deterministic values: value = row index (to verify split positions)
        let data: Vec<f32> = (0..total_rows * hidden).map(|i| i as f32).collect();
        w.insert(
            "model.layers.0.self_attn.fused_qkvbfg_a_proj.weight".to_string(),
            Tensor::from_f32(Shape::new([total_rows, hidden]), data),
        );
        apply_fused_aliases(&mut w, &cfg);
        // fused removed
        assert!(!w.contains_key("model.layers.0.self_attn.fused_qkvbfg_a_proj.weight"));
        // qkv_proj = rows [0, 3*proj)
        let qkv = &w["model.layers.0.self_attn.qkv_proj.weight"];
        assert_eq!(qkv.shape.0, vec![3 * proj, hidden]);
        assert_eq!(qkv.as_slice()[0], 0.0); // first row = fused row 0
        assert_eq!(qkv.as_slice()[3 * proj * hidden - 1], (3 * proj * hidden - 1) as f32);
        // b_proj = rows [3*proj, 3*proj+heads)
        let b = &w["model.layers.0.self_attn.b_proj.weight"];
        assert_eq!(b.shape.0, vec![heads, hidden]);
        assert_eq!(b.as_slice()[0], (3 * proj * hidden) as f32);
        // f_a = rows [3*proj+heads, +head_dim)
        let fa = &w["model.layers.0.self_attn.f_a_proj.weight"];
        assert_eq!(fa.shape.0, vec![head_dim, hidden]);
        assert_eq!(fa.as_slice()[0], ((3 * proj + heads) * hidden) as f32);
        // g_a = last head_dim rows
        let ga = &w["model.layers.0.self_attn.g_a_proj.weight"];
        assert_eq!(ga.shape.0, vec![head_dim, hidden]);
        let g_start = (3 * proj + heads + head_dim) * hidden;
        assert_eq!(ga.as_slice()[0], g_start as f32);
    }

    #[test]
    fn fused_fg_b_splits_correctly() {
        let cfg = Glm53FlashConfig::test_config();
        let mut w = Weights::new();
        let (proj, head_dim) = (128, 32);
        let total = 2 * proj * head_dim;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        w.insert(
            "model.layers.0.self_attn.fused_fg_b_proj.weight".to_string(),
            Tensor::from_f32(Shape::new([2, proj, head_dim]), data),
        );
        apply_fused_aliases(&mut w, &cfg);
        let fb = &w["model.layers.0.self_attn.f_b_proj.weight"];
        let gb = &w["model.layers.0.self_attn.g_b_proj.weight"];
        assert_eq!(fb.shape.0, vec![proj, head_dim]);
        assert_eq!(gb.shape.0, vec![proj, head_dim]);
        assert_eq!(fb.as_slice()[0], 0.0);
        assert_eq!(gb.as_slice()[0], (proj * head_dim) as f32);
    }

    #[test]
    fn fused_dsa_and_gate_up_split() {
        let cfg = Glm53FlashConfig::test_config();
        let mut w = Weights::new();
        let hidden = 128;
        // DSA fused: q_lora(128) + kv_latent(64)
        let q_lora = cfg.dsa.q_lora_rank;
        let kv_latent = cfg.dsa.kv_latent_dim();
        let rows = q_lora + kv_latent;
        let data: Vec<f32> = (0..rows * hidden).map(|i| i as f32).collect();
        w.insert(
            "model.layers.1.self_attn.fused_qkv_a_proj_with_mqa.weight".to_string(),
            Tensor::from_f32(Shape::new([rows, hidden]), data),
        );
        // dense FFN gate_up: layer 0 is dense (test config)
        let inter = cfg.intermediate_size;
        let gdata: Vec<f32> = (0..2 * inter * hidden).map(|i| i as f32).collect();
        w.insert(
            "model.layers.0.mlp.gate_up_proj.weight".to_string(),
            Tensor::from_f32(Shape::new([2 * inter, hidden]), gdata),
        );
        apply_fused_aliases(&mut w, &cfg);
        let qa = &w["model.layers.1.self_attn.q_a_proj.weight"];
        let kva = &w["model.layers.1.self_attn.kv_a_proj_with_mqa.weight"];
        assert_eq!(qa.shape.0, vec![q_lora, hidden]);
        assert_eq!(kva.shape.0, vec![kv_latent, hidden]);
        assert_eq!(qa.as_slice()[0], 0.0);
        assert_eq!(kva.as_slice()[0], (q_lora * hidden) as f32);
        let gate = &w["model.layers.0.mlp.gate_proj.weight"];
        let up = &w["model.layers.0.mlp.up_proj.weight"];
        assert_eq!(gate.shape.0, vec![inter, hidden]);
        assert_eq!(up.shape.0, vec![inter, hidden]);
        assert_eq!(up.as_slice()[0], (inter * hidden) as f32);
    }
}
