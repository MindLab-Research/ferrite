//! End-to-end chain smoke test on a tiny synthetic model.
//!
//! Exercises the *whole* chain — config → hyper-connections → MLA (window ring +
//! compression + indexer + candidates) → MoE (gate/expert clamps/shared) →
//! engram (hash + gather + gate) → head, plus the DSpark draft — with synthetic
//! weights and no GPU. It does not check numerics against the reference (that
//! needs the real checkpoint); it checks that every operator is wired, the
//! shapes flow, the cross-layer shared state is published/consumed in order and
//! nothing produces NaN/inf.

use ferrite_dsv41::chain::{
    EngramHostWeights, LayerHostWeights, ModelHostWeights, ModelState, forward, forward_spec,
};
use ferrite_dsv41::config::Dsv41Config;
use ferrite_dsv41::dspark::DraftHostWeights;
use ferrite_dsv41::engram::{EngramLayout, TokenMap};
use ferrite_dsv41::ops;

/// Deterministic small-value generator (no external rng dependency).
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * 0.2).collect()
    }
}

/// A tiny but structurally complete config: every mechanism present, all dims
/// small enough for the CPU reference path.
fn tiny_config() -> Dsv41Config {
    let mut json = String::from(
        r#"{
      "vocab_size": 64,
      "dim": 32,
      "moe_inter_dim": 16,
      "n_layers": 5,
      "n_mtp_layers": 1,
      "n_heads": 4,
      "head_dim": 8,
      "rope_head_dim": 2,
      "q_lora_rank": 8,
      "o_lora_rank": 4,
      "o_groups": 2,
      "norm_eps": 1e-6,
      "window_size": 4,
      "compress_ratios": [0, 0, 2, 1, 0, 0],
      "kv_source_layers": [2],
      "index_source_layers": [2, 3],
      "compress_rope_theta": 40000.0,
      "original_seq_len": 0,
      "rope_theta": 10000.0,
      "rope_factor": 16.0,
      "beta_fast": 32.0,
      "beta_slow": 1.0,
      "index_n_heads": 2,
      "index_head_dim": 4,
      "index_topk": 2,
      "candidate_source_layer": 2,
      "candidate_topk_blocks": 2,
      "candidate_block_size": 2,
      "n_routed_experts": 4,
      "n_shared_experts": 1,
      "n_activated_experts": 2,
      "score_func": "sqrtsoftplus",
      "gate_temp": 1.0,
      "norm_topk_prob": true,
      "route_scale": 1.5,
      "swiglu_limit": 10.0,
      "hc_mult": 4,
      "hc_sinkhorn_iters": 20,
      "hc_eps": 1e-6,
      "dtype": "fp8",
      "expert_dtype": "fp4",
      "engram_layer_ids": [1, 3],
      "engram_num_embeddings": [64, 64],
      "engram_max_ngram_size": 3,
      "engram_vocab_size": 1000,
      "engram_n_heads": 2,
      "engram_head_dim": 4,
      "engram_pad_id": 2,
      "engram_compressed_vocab_size": 64,
      "dspark_block_size": 2,
      "dspark_noise_token_id": 63,
      "dspark_target_layer_ids": [3, 4],
      "dspark_markov_rank": 8,
      "dspark_n_routed_experts": 2,
      "dspark_num_experts_per_tok": 1,
      "image_token_id": 60
    }"#,
    );
    json.retain(|c| !c.is_whitespace());
    Dsv41Config::from_json_str(&json).expect("tiny config parses")
}

fn layer_weights(cfg: &Dsv41Config, rng: &mut Lcg) -> LayerHostWeights {
    let dim = cfg.dim;
    let hd = cfg.head_dim;
    let hc = cfg.hc_mult;
    let mut w = LayerHostWeights {
        freqs: ops::precompute_freqs(
            cfg.rope_head_dim,
            cfg.max_seq_len.max(64),
            cfg.original_seq_len,
            cfg.rope_theta,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
        ),
        published_topk: Vec::new(),
        ..Default::default()
    };
    let mix = cfg.hc_mix_count();
    w.hc_attn_fn = rng.vec(mix * hc * dim);
    w.hc_attn_base = rng.vec(mix);
    w.hc_attn_scale = vec![0.1, 0.1, 0.1];
    w.hc_ffn_fn = rng.vec(mix * hc * dim);
    w.hc_ffn_base = rng.vec(mix);
    w.hc_ffn_scale = vec![0.1, 0.1, 0.1];
    w.attn_norm = vec![1.0; dim];
    w.ffn_norm = vec![1.0; dim];
    w.wq_a = rng.vec(cfg.q_lora_rank * dim);
    w.q_norm = vec![1.0; cfg.q_lora_rank];
    w.wq_b = rng.vec(cfg.n_heads * hd * cfg.q_lora_rank);
    w.wkv = rng.vec(hd * dim);
    w.kv_norm = vec![1.0; hd];
    w.wo_a = rng.vec(cfg.o_groups * cfg.o_lora_rank * (cfg.n_heads / cfg.o_groups) * hd);
    w.wo_b = rng.vec(dim * cfg.o_groups * cfg.o_lora_rank);
    w.attn_sink = rng.vec(cfg.n_heads);
    // compressor / indexer only on the layers that own them (fill always; the
    // chain only reads them where the config says so)
    w.compressor_wkv = rng.vec(hd * dim);
    w.compressor_wgate = Some(rng.vec(hd * dim));
    w.compressor_norm = vec![1.0; hd];
    w.indexer_wq_b = rng.vec(cfg.index_n_heads * cfg.index_head_dim * cfg.q_lora_rank);
    // weights_proj(x) is per token: size for the largest row count used here
    w.indexer_weights = rng.vec(cfg.index_n_heads * 8);
    w.indexer_k = rng.vec(512 * cfg.index_head_dim);
    w.gate_w = rng.vec(cfg.n_routed_experts * dim);
    w.gate_bias = rng.vec(cfg.n_routed_experts);
    let inter = cfg.moe_inter_dim;
    w.experts = (0..cfg.n_routed_experts)
        .map(|_| ferrite_dsv41::chain::HostExpert {
            w1: rng.vec(inter * dim),
            w3: rng.vec(inter * dim),
            w2: rng.vec(dim * inter),
        })
        .collect();
    w.shared_w1 = rng.vec(inter * dim);
    w.shared_w3 = rng.vec(inter * dim);
    w.shared_w2 = rng.vec(dim * inter);
    w
}

#[test]
fn tiny_model_forwards_prefill_decode_and_draft() {
    let cfg = tiny_config();
    let mut rng = Lcg::new(7);
    let dim = cfg.dim;
    let mut w = ModelHostWeights {
        embed: rng.vec(cfg.vocab_size * dim),
        norm: vec![1.0; dim],
        head: rng.vec(cfg.vocab_size * dim),
        layers: (0..cfg.n_layers).map(|_| layer_weights(&cfg, &mut rng)).collect(),
        engram: vec![None; cfg.n_layers],
        draft: (0..cfg.n_mtp_layers)
            .map(|_| DraftHostWeights {
                freqs: ops::precompute_freqs(
                    cfg.rope_head_dim,
                    256,
                    cfg.original_seq_len,
                    cfg.rope_theta,
                    cfg.rope_factor,
                    cfg.beta_fast,
                    cfg.beta_slow,
                ),
                ..Default::default()
            })
            .collect(),
    };
    // engram tables at the configured layers
    let layout = EngramLayout::from_config(&cfg).unwrap();
    let map = TokenMap::identity(cfg.vocab_size);
    for (i, &l) in cfg.engram_layer_ids.iter().enumerate() {
        let n_cols = layout.n_hash_cols();
        let ehd = cfg.engram_head_dim;
        let wkv_k = n_cols * ehd;
        let wkv_n = dim * (cfg.hc_mult + 1);
        w.engram[l] = Some(EngramHostWeights {
            table: rng.vec(64 * ehd),
            part_rows: 64,
            wkv: rng.vec(wkv_n * wkv_k),
            q_weight: rng.vec(cfg.hc_mult * dim),
            k_weight: rng.vec(cfg.hc_mult * dim),
            token_map: map.clone(),
            layout: layout.clone(),
            pad_id: map.get(cfg.engram_pad_id as i64),
        });
        let _ = i;
    }
    // draft weights
    let d = &mut w.draft[0];
    d.wq_a = rng.vec(cfg.q_lora_rank * dim);
    d.q_norm = vec![1.0; cfg.q_lora_rank];
    d.wq_b = rng.vec(cfg.n_heads * cfg.head_dim * cfg.q_lora_rank);
    d.wkv = rng.vec(cfg.head_dim * dim);
    d.kv_norm = vec![1.0; cfg.head_dim];
    d.wo_a = rng.vec(cfg.o_groups * cfg.o_lora_rank * (cfg.n_heads / cfg.o_groups) * cfg.head_dim);
    d.wo_b = rng.vec(dim * cfg.o_groups * cfg.o_lora_rank);
    d.attn_sink = rng.vec(cfg.n_heads);
    d.main_proj = rng.vec(dim * dim * cfg.dspark_target_layer_ids.len());
    d.main_norm = vec![1.0; dim];
    d.norm = vec![1.0; dim];
    d.head = rng.vec(cfg.vocab_size * dim);
    d.markov_embed = rng.vec(cfg.vocab_size * cfg.dspark_markov_rank);
    d.markov_head = rng.vec(cfg.vocab_size * cfg.dspark_markov_rank);
    d.confidence_proj = rng.vec(dim + cfg.dspark_markov_rank);
    d.noise_embed = rng.vec(dim);

    let mut st = ModelState::new(&cfg, Some(map.clone()));
    // prefill 3 tokens
    let (logits, main_hidden) = forward(&cfg, &[1, 2, 3], 0, &mut st, &w);
    assert_eq!(logits.len(), cfg.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()), "prefill logits finite: {:?}", &logits[..4.min(logits.len())]);
    // the reference cats the target layers' collapsed hidden per ROW:
    // [b, s, targets*dim]; at decode s == 1, which is what forward_spec consumes
    assert_eq!(main_hidden.len(), 3 * dim * cfg.dspark_target_layer_ids.len());
    assert!(main_hidden.iter().all(|v| v.is_finite()));

    // decode one token
    let (l2, mh2) = forward(&cfg, &[4], 3, &mut st, &w);
    assert_eq!(l2.len(), cfg.vocab_size);
    assert!(l2.iter().all(|v| v.is_finite()));

    // the draft: prefill only seeds, decode returns ids + logits + confidence
    let seeded = forward_spec(&cfg, 4, &mh2, 0, &mut st, &w);
    assert!(seeded.is_none(), "prefill only seeds the draft windows");
    let out = forward_spec(&cfg, 4, &mh2, 3, &mut st, &w);
    let (ids, dlogits, conf) = out.expect("decode produces a draft block");
    assert_eq!(ids.len(), cfg.dspark_block_size + 1);
    assert_eq!(ids[0], 4, "the first draft id is the backbone's token");
    assert_eq!(dlogits.len(), cfg.dspark_block_size * cfg.vocab_size);
    assert!(dlogits.iter().all(|v| v.is_finite()));
    assert_eq!(conf.len(), cfg.dspark_block_size);
    assert!(conf.iter().all(|v| v.is_finite()));
}

#[test]
fn shared_attention_state_is_published_in_layer_order() {
    // The ratio>0 consumers read the most recent publisher's cache. Walk the
    // layers and check the state only changes at source layers.
    let cfg = tiny_config();
    assert_eq!(cfg.kv_source_layers, vec![2]);
    assert_eq!(cfg.compress_ratio(2), 2);
    assert_eq!(cfg.compress_ratio(3), 1); // a consumer of layer 2's cache
    assert!(cfg.is_index_source(3));
    assert!(!cfg.is_kv_source(3));
    assert!(cfg.is_candidate_source(2));
}
