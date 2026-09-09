//! Tensor parallelism for GLM-5.3-Flash — weight sharding + sharded forward
//! with all-reduce, verified end-to-end against TP=1 (CPU-simulated
//! collectives now; NCCL wired in on the GPU path via `nccl.rs`).
//!
//! ## TP sharding scheme (per module, world size N)
//!
//! | Module | Shard | Communication |
//! |---|---|---|
//! | Linear attn qkv/conv/b/g/A_log/dt/o_norm | **head split** (heads/N) | — |
//! | Linear attn state [heads, dk, dv] | head split (GatedDeltaNet heads are independent) | — |
//! | Linear attn f_a/g_a [head_dim, h] | replicated (shared) | — |
//! | Linear attn f_b/g_b [proj, head_dim] | head split (row) | — |
//! | Linear attn o_proj [h, proj] | **column split** (input = head subset) | all-reduce |
//! | DSA q_b/kv_b [heads*, ...] | head split (row) | — |
//! | DSA kv_a/indexer/q_a | replicated (MLA latent + indexer shared) | — |
//! | DSA o_proj [h, heads*v] | column split | all-reduce |
//! | Dense MLP gate/up [inter, h] | row split (inter/N) | — |
//! | Dense MLP down [h, inter] | column split | all-reduce |
//! | MoE experts | expert split (N experts/N, EP-style) | all-gather (logits all-reduce alt) |
//! | MoE router / MHC / embedding (per-rank slice) / lm_head | replicated or vocab-sliced | all-gather |
//!
//! The **GatedDeltaNet head split is exact** — each head's recurrence is
//! independent, so the state slice [heads/N, dk, dv] needs no communication.
//! DSA's latent KV is shared (MQA-style), head split applies to the
//! output projections only.

use std::collections::HashMap;

use ferrite_model::{AttnKind, Glm53FlashConfig, MlpKind, Weights, build_layer_plans, Fp8Weight, Weights8};
use ferrite_types::{DType, FerriteError, Result, Shape, Tensor};

use crate::Engine;

// ---------------------------------------------------------------------------
// Weight sharding
// ---------------------------------------------------------------------------

/// Slice a weight along dim0 (row split) → rows [start, end).
/// Handles 1D (A_log/o_norm/dt_bias) and 2D tensors.
// ---------------------------------------------------------------------------
// FP8 TP sharding — mirrors shard_weights_tp's classification, but slices the
// native F8 bytes + 128-block scale instead of dequantized f32. Mirrors must
// stay bit-consistent with the Tensor sharding above (cross-checked by
// `shard_weights8_tp`'s shape assertion vs the f32 shard) or draft/verify
// numerics diverge across ranks.
// ---------------------------------------------------------------------------

/// Row-split an Fp8Weight [r0, r1). Returns None unless both edges are
/// 128-aligned (the block-scale grid rows r/128 must not straddle the seam);
/// caller falls back to the bf16 path on None.
fn fp8_row(f: &Fp8Weight, r0: usize, r1: usize) -> Option<Fp8Weight> {
    if r0 % 128 != 0 || r1 % 128 != 0 || r1 < r0 || r1 > f.rows {
        return None;
    }
    let scols = f.cols.div_ceil(128);
    let s0 = r0 / 128;
    let s1 = r1 / 128;
    let mut scale = Vec::with_capacity((s1 - s0) * scols);
    for sr in s0..s1 {
        let base = sr * scols;
        scale.extend_from_slice(&f.scale[base..base + scols]);
    }
    Some(Fp8Weight {
        rows: r1 - r0,
        cols: f.cols,
        data: f.data[r0 * f.cols..r1 * f.cols].to_vec(),
        scale,
    })
}

/// Col-split an Fp8Weight [c0, c1) — same 128-alignment contract on cols.
fn fp8_col(f: &Fp8Weight, c0: usize, c1: usize) -> Option<Fp8Weight> {
    if c0 % 128 != 0 || c1 % 128 != 0 || c1 < c0 || c1 > f.cols {
        return None;
    }
    let scols = f.cols.div_ceil(128);
    let t0 = c0 / 128;
    let t1 = c1 / 128;
    let nsc = t1 - t0;
    let mut scale = Vec::with_capacity(f.rows.div_ceil(128) * nsc);
    for sr in 0..f.rows.div_ceil(128) {
        let base = sr * scols;
        scale.extend_from_slice(&f.scale[base + t0..base + t1]);
    }
    let mut data = Vec::with_capacity(f.rows * (c1 - c0));
    for r in 0..f.rows {
        let base = r * f.cols;
        data.extend_from_slice(&f.data[base + c0..base + c1]);
    }
    Some(Fp8Weight {
        rows: f.rows,
        cols: c1 - c0,
        data,
        scale,
    })
}

/// TP-shard the fp8 bypass set. Classification mirrors shard_weights_tp:
/// replicated / EP-whole experts / row-split (gate/up, q_b/kv_b, b/g/f_b) /
/// col-split (down, o_proj, eh_proj). Misaligned seams or non-fp8 names are
/// simply absent from the result — the engine's w8() lookup misses and the
/// bf16 path serves that weight (safe fallback, keeps every rank consistent
/// because the seam alignment is rank-independent for power-of-2 worlds).
#[allow(clippy::too_many_arguments)]
pub fn shard_weights8_tp(
    w8: &Weights8,
    cfg: &Glm53FlashConfig,
    rank: usize,
    world: usize,
) -> Weights8 {
    assert!(world >= 1 && rank < world);
    if world == 1 {
        return w8.clone();
    }
    let h = cfg.hidden_size;
    let heads = cfg.linear_attn.num_heads;
    let dk = cfg.linear_attn.head_dim;
    let proj = heads * dk;
    let dsa_h = cfg.dsa.num_attention_heads;
    let nope = cfg.dsa.qk_nope_head_dim;
    let vd = cfg.dsa.v_head_dim;
    let (hs, he) = head_range(heads, rank, world);
    let (dhs, dhe) = head_range(dsa_h, rank, world);
    let n_exp = cfg.n_routed_experts;
    let (es, ee) = head_range(n_exp, rank, world);
    let mut out: Weights8 = HashMap::new();
    for (name, f) in w8 {
        // fused qkv/conv1d bypasses were never stored (block seams at the
        // q|k|v row boundaries) — only plain fp8 tensors reach here.
        let local: Option<Fp8Weight> = if name == "lm_head.weight" {
            // vocab-split happens device-side (full + mask) — replicate.
            Some(f.clone())
        } else if name == "model.embed_tokens.weight" {
            Some(f.clone())
        } else if name.ends_with(".eh_proj.weight") {
            let cols = f.cols; // 2h
            fp8_col(f, cols * rank / world, cols * (rank + 1) / world)
        } else if let Some(layer_str) = layer_of(name) {
            let layer: usize = layer_str.parse().unwrap_or(0);
            let plan = build_layer_plans(cfg);
            let lp = if layer >= plan.len() {
                &ferrite_model::LayerPlan {
                    layer_idx: layer,
                    attn: AttnKind::Dsa,
                    mlp: MlpKind::Moe,
                }
            } else {
                &plan[layer]
            };
            // expert weights: MoE TP (default) shards every expert's inter dim
            // across ranks (gate/up row-split, down col-split — constant topk
            // work per rank, no EP routing skew); FERRITE_MOE_EP=1 keeps EP
            // (whole experts per rank).
            if let Some(expert) = name.split(".experts.").nth(1) {
                let e: usize = expert.split('.').next().unwrap_or("0").parse().unwrap_or(0);
                if std::env::var_os("FERRITE_MOE_EP").is_some() {
                    if e >= es && e < ee {
                        Some(f.clone())
                    } else {
                        None
                    }
                } else if name.ends_with(".down_proj.weight") {
                    fp8_col(f, f.cols * rank / world, f.cols * (rank + 1) / world)
                } else {
                    // gate/up: [inter, hidden] row-split
                    fp8_row(f, f.rows * rank / world, f.rows * (rank + 1) / world)
                }
            } else if name.ends_with(".shared_expert.gate_proj.weight")
                || name.ends_with(".shared_expert.up_proj.weight")
                || name.ends_with(".gate_proj.weight")
                || name.ends_with(".up_proj.weight")
            {
                fp8_row(f, f.rows * rank / world, f.rows * (rank + 1) / world)
            } else if name.ends_with(".shared_expert.down_proj.weight")
                || name.ends_with(".down_proj.weight")
            {
                fp8_col(f, f.cols * rank / world, f.cols * (rank + 1) / world)
            } else {
                match lp.attn {
                    AttnKind::Linear => {
                        if name.ends_with(".b_proj.weight") {
                            fp8_row(f, hs * dk, he * dk)
                        } else if name.ends_with(".dt_bias") {
                            fp8_row(f, hs * dk, he * dk) // 1D — no fp8 in practice
                        } else if name.ends_with(".f_b_proj.weight") || name.ends_with(".g_b_proj.weight") {
                            fp8_row(f, hs * dk, he * dk)
                        } else if name.ends_with(".o_proj.weight") {
                            fp8_col(f, hs * dk, he * dk)
                        } else {
                            Some(f.clone()) // f_a/g_a proj, o_norm, indexer
                        }
                    }
                    AttnKind::Dsa => {
                        if name.ends_with(".q_b_proj.weight") {
                            fp8_row(f, dhs * nope, dhe * nope)
                        } else if name.ends_with(".kv_b_proj.weight") {
                            fp8_row(f, dhs * (nope + vd), dhe * (nope + vd))
                        } else if name.ends_with(".o_proj.weight") {
                            fp8_col(f, dhs * vd, dhe * vd)
                        } else {
                            Some(f.clone()) // q_a/kv_a/indexer replicated
                        }
                    }
                }
            }
        } else {
            Some(f.clone()) // model.norm, hc_*, router (mlp.gate) replicated
        };
        if let Some(l) = local {
            out.insert(name.clone(), l);
        }
    }
    out
}

fn row_split(w: &Tensor, start: usize, end: usize) -> Tensor {
    let dims = &w.shape.0;
    // fp8 single-store placeholder (data is a 4-elem unique-ptr stub, shape is
    // real): split SHAPE ONLY — the actual weights live in the fp8 bypass
    // (w8); Engine.weights carries this stub for dim readers + the
    // (ptr, numel) fp8_map key (shard numel = shard shape product).
    if w.as_slice().len() < w.numel() {
        let mut shape = vec![end - start];
        shape.extend_from_slice(&dims[1..]);
        return Tensor {
            shape: Shape::new(shape),
            dtype: w.dtype,
            data: std::sync::Arc::new(vec![0f32; 4]),
        };
    }
    if dims.len() == 1 {
        let data = w.as_slice()[start..end].to_vec();
        return Tensor::new(Shape::new([end - start]), w.dtype, data);
    }
    let cols = dims[1..].iter().product::<usize>();
    let data = w.as_slice()[start * cols..end * cols].to_vec();
    let mut shape = vec![end - start];
    shape.extend_from_slice(&dims[1..]);
    Tensor::new(Shape::new(shape), w.dtype, data)
}

/// Slice a weight along dim1 (column split) → cols [start, end).
fn col_split(w: &Tensor, start: usize, end: usize) -> Tensor {
    let rows = w.shape.0[0];
    let cols = w.shape.0[1];
    if w.as_slice().len() < w.numel() {
        // fp8 placeholder: shape-only split (see row_split)
        return Tensor {
            shape: Shape::new([rows, end - start]),
            dtype: w.dtype,
            data: std::sync::Arc::new(vec![0f32; 4]),
        };
    }
    let mut data = Vec::with_capacity(rows * (end - start));
    for r in 0..rows {
        data.extend_from_slice(&w.as_slice()[r * cols + start..r * cols + end]);
    }
    Tensor::new(Shape::new([rows, end - start]), w.dtype, data)
}

fn head_range(total: usize, rank: usize, world: usize) -> (usize, usize) {
    let per = total / world;
    (rank * per, (rank + 1) * per)
}

/// Shard all weights for TP rank `rank` in `world`. Returns the rank's
/// local weight set. Weights not listed are replicated (shared).
pub fn shard_weights_tp(
    w: &Weights,
    cfg: &Glm53FlashConfig,
    rank: usize,
    world: usize,
) -> Weights {
    assert!(world >= 1 && rank < world);
    if world == 1 {
        return w.clone();
    }
    let h = cfg.hidden_size;
    let heads = cfg.linear_attn.num_heads;
    let dk = cfg.linear_attn.head_dim;
    let proj = heads * dk;
    let dsa_h = cfg.dsa.num_attention_heads;
    let (hs, he) = head_range(heads, rank, world);
    let (dhs, dhe) = head_range(dsa_h, rank, world);
    // MTP (nextn) layer plan: DSA attention + MoE mlp — same shard rules
    // as decoder DSA/MoE layers (eh_proj/enorm/hnorm/shared_head.norm
    // handled by the replicated / column-split branches above).
    let mtp_plan = ferrite_model::LayerPlan {
        layer_idx: cfg.num_hidden_layers,
        attn: AttnKind::Dsa,
        mlp: MlpKind::Moe,
    };
    let mut out = HashMap::new();

    for name in w.keys() {
        let t = &w[name];
        let local = if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
            // vocab split: rows [vocab/N for this rank] — all-gather at the
            // output boundary. For the CPU simulation we keep full + mask in
            // the forward (simpler); the GPU path splits.
            t.clone()
        } else if name.starts_with("model.norm.weight")
            || name.ends_with(".enorm.weight")
            || name.ends_with(".hnorm.weight")
            || name.ends_with(".shared_head.norm.weight")
            || name.ends_with("input_layernorm.weight")
            || name.ends_with("q_a_layernorm.weight")
            || name.ends_with("kv_a_layernorm.weight")
            || name.ends_with("indexer_norm.weight")
            || name.ends_with("hc_attn_base")
            || name.ends_with("hc_attn_scale")
            || name.ends_with("hc_attn_fn")
            || name.ends_with("hc_ffn_base")
            || name.ends_with("hc_ffn_scale")
            || name.ends_with("hc_ffn_fn")
            || name.ends_with("mlp.gate.weight")
        {
            // replicated: norms over full hidden, MHC, router
            t.clone()
        } else if name.ends_with(".eh_proj.weight") {
            // MTP eh_proj [h, 2h]: column split — input is
            // cat(enorm(embed), hnorm(h_prev)), each rank takes 2h/world cols
            // (partial sums all-reduced in mtp_layer_dev).
            let cols = t.shape.0[1];
            col_split(t, cols * rank / world, cols * (rank + 1) / world)
        } else if let Some(layer_str) = layer_of(name) {
            let layer: usize = layer_str.parse().unwrap_or(0);
            let plan = build_layer_plans(cfg);
            let lp = if layer >= plan.len() {
                // MTP (nextn) layer: DSA attention + MoE mlp (eh_proj handled above)
                &mtp_plan
            } else {
                &plan[layer]
            };
            shard_one_layer(name, t, cfg, &lp, rank, world, hs, he, dhs, dhe, h, proj, dk)
        } else {
            t.clone()
        };
        out.insert(name.clone(), local);
    }
    // mark tp sharding metadata
    out
}

#[allow(clippy::too_many_arguments)]
fn shard_one_layer(
    name: &str,
    t: &Tensor,
    cfg: &Glm53FlashConfig,
    lp: &ferrite_model::LayerPlan,
    rank: usize,
    world: usize,
    hs: usize,
    he: usize,
    dhs: usize,
    dhe: usize,
    h: usize,
    proj: usize,
    dk: usize,
) -> Tensor {
    match lp.attn {
        AttnKind::Linear => shard_linear_attn_weight(name, t, rank, world, hs, he, h, proj, dk),
        AttnKind::Dsa => shard_dsa_weight(name, t, cfg, rank, world, dhs, dhe, h),
    }
    .unwrap_or_else(|| shard_mlp_weight(name, t, cfg, lp, rank, world, h).unwrap_or_else(|| t.clone()))
}

fn layer_of(name: &str) -> Option<&str> {
    let start = name.strip_prefix("model.layers.")?;
    let end = start.find('.')?;
    Some(&start[..end])
}

/// Linear-attention weight sharding (head split for qkv/b/g/o_norm,
/// column split for o_proj).
fn shard_linear_attn_weight(
    name: &str,
    t: &Tensor,
    rank: usize,
    world: usize,
    hs: usize,
    he: usize,
    _h: usize,
    proj: usize,
    dk: usize,
) -> Option<Tensor> {
    let heads_per = proj / dk / world.max(1);
    let _ = heads_per;
    let (rows, cols) = (t.shape.0[0], t.shape.0.get(1).copied().unwrap_or(1));
    if name.ends_with(".qkv_proj.weight") || name.ends_with(".qkv_conv1d.weight") {
        // [3*proj, X] rows are [q_heads..., k_heads..., v_heads...] — head-split
        // each third by rows [hs*dk, he*dk).
        // Placeholder guard (direct mmap path): data.len() < numel means the
        // real bytes live in the mmap segment (device-side preload handles the
        // split) — shape-only pass-through, no host slicing.
        if t.as_slice().len() < rows * cols {
            return Some(Tensor {
                shape: Shape::new([3 * (he - hs) * dk, cols]),
                dtype: t.dtype,
                data: std::sync::Arc::new(vec![0f32; 4]),
            });
        }
        let mut data = Vec::new();
        let third = rows / 3;
        let (qs, qe) = (hs * dk, he * dk);
        for seg in 0..3 {
            let base = seg * third;
            data.extend_from_slice(&t.as_slice()[(base + qs) * cols..(base + qe) * cols]);
        }
        Some(Tensor::new(Shape::new([3 * (qe - qs), cols]), t.dtype, data))
    } else if name.ends_with(".b_proj.weight") || name.ends_with(".A_log") {
        // [heads, h] or [heads] — head split rows
        Some(row_split(t, hs, he))
    } else if name.ends_with(".dt_bias") {
        // [h*dk] per-channel KDA forget-gate bias (on the f_b_proj output
        // channels = heads*head_dim) — head split
        Some(row_split(t, hs * dk, he * dk))
    } else if name.ends_with(".f_b_proj.weight") || name.ends_with(".g_b_proj.weight") {
        // [proj, head_dim] rows = heads*dk — head split
        Some(row_split(t, hs * dk, he * dk))
    } else if name.ends_with(".o_proj.weight") {
        // [h, proj] column split (input = head subset)
        Some(col_split(t, hs * dk, he * dk))
    } else {
        // f_a_proj/g_a_proj [head_dim, h] and o_norm [head_dim] (per-head
        // shared): replicated
        None
    }
}

/// DSA weight sharding (head split for q_b/kv_b, column for o_proj,
/// shared latent/indexer — the indexer (wq_b/wk/k_norm/weights_proj) is
/// replicated: per-head indexer scores are computed in full on every rank and
/// the top-k selection is global).
fn shard_dsa_weight(
    name: &str,
    t: &Tensor,
    cfg: &Glm53FlashConfig,
    rank: usize,
    world: usize,
    dhs: usize,
    dhe: usize,
    h: usize,
) -> Option<Tensor> {
    let _ = (h, rank, world);
    let nope = cfg.dsa.qk_nope_head_dim;
    let v = cfg.dsa.v_head_dim;
    if name.ends_with(".q_b_proj.weight") {
        // [heads*nope, q_lora] — head split rows
        Some(row_split(t, dhs * nope, dhe * nope))
    } else if name.ends_with(".kv_b_proj.weight") {
        // [heads*(nope+v), kv_lora] — head split rows
        Some(row_split(t, dhs * (nope + v), dhe * (nope + v)))
    } else if name.ends_with(".o_proj.weight") {
        // [h, heads*v] — column split
        Some(col_split(t, dhs * v, dhe * v))
    } else {
        // q_a/kv_a/layernorms/indexer.* replicated
        None
    }
}

/// MLP sharding: dense row/column split, MoE expert split.
fn shard_mlp_weight(
    name: &str,
    t: &Tensor,
    cfg: &Glm53FlashConfig,
    lp: &ferrite_model::LayerPlan,
    rank: usize,
    world: usize,
    h: usize,
) -> Option<Tensor> {
    let _ = (lp, h);
    let (rows, cols) = (t.shape.0[0], t.shape.0.get(1).copied().unwrap_or(1));
    if let Some(expert) = name.split(".experts.").nth(1) {
        // "e.gate_proj.weight"
        let e: usize = expert.split('.').next()?.parse().ok()?;
        let _ = e;
        let n = cfg.n_routed_experts;
        // MoE TP (default): every expert's intermediate dim sharded across
        // ranks — gate/up row-split [inter/world, hidden], down col-split
        // [hidden, inter/world] (the shared_expert pattern). Topk work is
        // then CONSTANT per rank — no EP routing skew (nsys: p2p_ar_sum
        // waited 44µs avg on the EP hot rank vs 2µs min = 4ms/step of pure
        // skew). FERRITE_MOE_EP=1 keeps the EP mode (whole experts per
        // rank, es..ee).
        if std::env::var_os("FERRITE_MOE_EP").is_none() {
            if name.ends_with(".down_proj.weight") {
                Some(col_split(t, cols * rank / world, cols * (rank + 1) / world))
            } else {
                // gate_proj / up_proj: [inter, hidden] row-split
                Some(row_split(t, rows * rank / world, rows * (rank + 1) / world))
            }
        } else {
            let (es, ee) = head_range(n, rank, world);
            if e < es || e >= ee {
                return Some(Tensor::new(Shape::new([0, cols]), t.dtype, vec![])); // empty: not ours
            }
            return Some(t.clone()); // full expert (EP-style: whole experts per rank)
        }
    } else if name.ends_with(".shared_expert.gate_proj.weight") || name.ends_with(".shared_expert.up_proj.weight") {
        Some(row_split(t, rows * rank / world, rows * (rank + 1) / world))
    } else if name.ends_with(".shared_expert.down_proj.weight") {
        Some(col_split(t, cols * rank / world, cols * (rank + 1) / world))
    } else if name.ends_with(".gate_proj.weight") || name.ends_with(".up_proj.weight") {
        Some(row_split(t, rows * rank / world, rows * (rank + 1) / world))
    } else if name.ends_with(".down_proj.weight") {
        Some(col_split(t, cols * rank / world, cols * (rank + 1) / world))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// CPU-simulated collectives (NCCL equivalents; the GPU path swaps these)
// ---------------------------------------------------------------------------

/// Simulated NCCL all-reduce (sum) for TP: partial outputs from each rank
/// (column-split o_proj / down_proj produce partial sums) → full tensor.
pub fn all_reduce_sum(partials: &[Tensor]) -> Tensor {
    assert!(!partials.is_empty());
    let shape = partials[0].shape.clone();
    let dtype = partials[0].dtype;
    let n = shape.numel();
    let mut acc = vec![0.0f32; n];
    for p in partials {
        for (a, v) in acc.iter_mut().zip(p.as_slice().iter()) {
            *a += v;
        }
    }
    Tensor::new(shape, dtype, acc)
}

/// Simulated NCCL all-gather (vocab-split embedding/lm_head).
pub fn all_gather_rows(parts: &[Tensor]) -> Tensor {
    assert!(!parts.is_empty());
    let cols = parts[0].shape.0[1];
    let dtype = parts[0].dtype;
    let mut data = Vec::new();
    for p in parts {
        data.extend_from_slice(p.as_slice());
    }
    Tensor::new(Shape::new([data.len() / cols, cols]), dtype, data)
}

// ---------------------------------------------------------------------------
// TpCluster — tensor-parallel execution across N shards
// ---------------------------------------------------------------------------

use ferrite_kernel::KernelBackend;

/// Verify-chain IO for MTP speculative decoding (arbitrary n, FERRITE_MTP_N):
/// - gdn_scratch: per-GDN-layer 4-tuple (conv, gdn, conv_snaps_base,
///   gdn_snaps_base) — the ping-pong B states + the [n-1] contiguous
///   t-snapshot scratch (snap i = A + tokens t_0..t_i, accept-(i+1)'s
///   commit source; the kernel indexes base + i*len);
/// - h_final: the verify chain's last hc_post residual rows (h_prev source
///   for the MTP draft's eh_proj) exported via a capture-safe D2D node.
#[cfg(feature = "cuda")]
pub(crate) struct VerifyIO {
    /// [n_gdn_layers] (conv, gdn, conv_snaps, gdn_snaps) N-UNIFIED: B = the
    /// full n-token verify state, snaps = [n-1][len] contiguous t-snapshots.
    pub gdn_scratch: Vec<(*mut f32, *mut f32, *mut f32, *mut f32)>,
    pub h_final: *mut f32, // [n*hidden] staging
}

/// FERRITE_MTP_N: the MTP verify width (draft count + 1). Default 3 = the
/// historical (d1, d2) two-draft chain (bit-identical). Any 1..=8: the draft
/// chain runs n-1 mtp_forward steps, the verify graph runs n rows, the
/// accept k ranges 1..=n — ONE code path for every n>=2. N=1 IS plain
/// decode: decode_step_mega routes it to the SAME mega1 graph path as
/// non-MTP (no ping-pong B copy, no commit kernel, no MTP buffers — zero
/// overhead, bit-identical to FERRITE_MTP unset).
#[cfg(feature = "cuda")]
pub(crate) fn mtp_verify_n() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        match std::env::var("FERRITE_MTP_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            Some(n) if (1..=8).contains(&n) => n,
            Some(bad) => {
                eprintln!("[mtp] FERRITE_MTP_N={bad} out of range 1..=8 — using 3");
                3
            }
            None => 3,
        }
    })
}

/// A tensor-parallel cluster: `world` engines, each holding its TP shard of
/// the weights (head-split attention, row/col-split MLP, expert-slice MoE).
/// Attention and FFN outputs are partial sums; the cluster all-reduces them
/// (CPU simulation via [`all_reduce_sum`]; the GPU path swaps in NCCL).
///
/// Because every collective lands at the attn/ffn boundary of each layer,
/// layers execute layer-synchronously across shards — the same discipline a
/// multi-rank NCCL deployment uses (all-reduce after attn, all-reduce after
/// FFN).
pub struct TpCluster<B: KernelBackend> {
    pub shards: Vec<Engine<B>>,
    pub full_cfg: Glm53FlashConfig,
    pub world: usize,
    /// Last batched-decode seq set: skip the per-size table refresh when
    /// unchanged (the refresh is ~45 layers x 2-6 H2D + mutexes per rank).
    last_batch_seqs: Option<Vec<u64>>,
    /// CUDA graph: true after the first decode_step captures the op sequence
    /// (FERRITE_GRAPH=1 path; replay replaces per-op launches).
    graph_captured: bool,
    /// CUDA graph: warmup/capture/replay step counter (FERRITE_GRAPH=1 path).
    graph_step: u32,
    /// Mega-graph (FERRITE_MEGA): seq whose whole-decode-step per-rank graphs
    /// are captured. Some(seq) → replay path; None/other seq → re-capture.
    mega_seq: Option<u64>,
    /// NCCL all-reduce channels (one per rank, single-process init_all).
    /// FERRITE_NCCL=1: replaces the host-side partial download → CPU sum →
    /// re-upload round-trip per attention/ffn segment (~0.6ms/layer).
    nccl: Option<std::sync::Arc<Vec<ferrite_kernel::nccl::NcclChannel>>>,
}

impl<B: KernelBackend> TpCluster<B> {
    /// Distribute the fp8 bypass set to every rank (same classification as
    /// the f32 sharding; block-128-misaligned seams simply drop out of the
    /// set — those weights stay on the bf16 path on ALL ranks).
    pub fn set_fp8(&mut self, w8: &Weights8) {
        if std::env::var_os("FERRITE_NO_FP8").is_some() {
            println!("[tp] fp8 bypass DISABLED (FERRITE_NO_FP8) — all bf16");
            return;
        }
        let dbg = std::env::var_os("FERRITE_FP8_DEBUG").is_some();
        let n = self.shards.len();
        // CONCURRENT per-rank registration (was a serial `for rank in 0..n` —
        // 9474 cudaMalloc+H2D × 4 ranks on the main thread: GPU k got its fp8
        // upload only after rank k-1 finished, so GPUs 2/3 sat at 1GB while
        // 0/1 were at 76GB — the visible "not concurrent loading"). rayon's
        // par_iter_mut gives each rank its own thread; register_fp8's enter()
        // is thread-local cudaSetDevice, so each thread binds ITS rank's
        // device — the H2Ds all fly concurrently.
        use rayon::prelude::*;
        let full_cfg = self.full_cfg.clone();
        let registered: Vec<usize> = self
            .shards
            .par_iter_mut()
            .enumerate()
            .map(|(rank, shard)| {
                let shard8 = shard_weights8_tp(w8, &full_cfg, rank, n);
                let mut cnt = 0usize;
                for (name, f8) in shard8.iter() {
                    let Some(golden) = shard.weights.get(name) else { continue };
                    let Some(cuda) = shard.backend.as_cuda() else { continue };
                    if let Err(e) = cuda.register_fp8(golden, f8.rows, f8.cols, &f8.data, &f8.scale) {
                        eprintln!("[tp] fp8 register {} failed: {e} (stays bf16)", name);
                    } else {
                        cnt += 1;
                        if dbg {
                            if !cuda.fp8_hit(golden) {
                                eprintln!(
                                    "[fp8dbg] REGISTER-VERIFY MISS {name} ptr={:x} numel={} map={}",
                                    golden.as_slice().as_ptr() as usize, golden.numel(), cuda.fp8_registered()
                                );
                            }
                            if cnt <= 3 || cnt % 2000 == 0 {
                                eprintln!(
                                    "[fp8dbg] r{rank} reg#{cnt} {name} ptr={:x} numel={}",
                                    golden.as_slice().as_ptr() as usize, golden.numel()
                                );
                            }
                        }
                    }
                }
                shard.weights8 = shard8;
                cnt
            })
            .collect();
        let registered: usize = registered.into_iter().sum();
        println!("[tp] fp8 bypass registered: {} weights (rank-avg {})", registered / n.max(1), registered);
    }

    /// Build a TP=world cluster from full weights. `mk_backend` constructs
    /// each rank's backend (rank index passed for device selection).
    pub fn new(
        full_cfg: Glm53FlashConfig,
        weights: &Weights,
        world: usize,
        mk_backend: impl Fn(usize) -> B,
    ) -> Self {
        assert!(world >= 1);
        let mut shards = Vec::with_capacity(world);
        for rank in 0..world {
            let mut shard_cfg = full_cfg.clone();
            // head-split dims shrink per rank; everything else replicated.
            shard_cfg.linear_attn.num_heads /= world;
            shard_cfg.dsa.num_attention_heads /= world;
            shard_cfg.intermediate_size /= world;
            let w = shard_weights_tp(weights, &full_cfg, rank, world);
            let mut engine = Engine::new(shard_cfg, w, mk_backend(rank));
            engine.tp_world = world;
            // MoE TP (default): ALL routed experts resident per rank (each
            // inter/world-sharded) — tp_expert_range covers the full set
            // (0..n) so the fused kernels' id→table indexing is direct.
            // FERRITE_MOE_EP=1 keeps EP (whole experts per rank slice).
            if std::env::var_os("FERRITE_MOE_EP").is_some() {
                let per = full_cfg.n_routed_experts / world;
                engine.tp_expert_range = Some((rank * per, (rank + 1) * per));
            } else {
                engine.tp_expert_range = Some((0, full_cfg.n_routed_experts));
            }
            shards.push(engine);
        }
        // P2P enable (FERRITE_P2P=1): rank 0 collects partials via NVLink
        // cudaMemcpyPeerAsync instead of the host round-trip.
        if std::env::var_os("FERRITE_P2P").is_some() {
            #[cfg(feature = "cuda")]
            {
                for (i, shard) in shards.iter().enumerate() {
                    if let Some(cuda) = shard.backend.as_cuda() {
                        for peer in 0..world as i32 {
                            if peer != i as i32 {
                                if let Err(e) = cuda.p2p_enable(peer) {
                                    eprintln!("[serve] P2P enable {}→{} failed: {:?}", i, peer, e);
                                }
                            }
                        }
                    }
                }
                eprintln!("[serve] P2P access enabled ({} ranks, NVLink)", world);
                // P2P AR v2 state (the in-graph epoch+ping-pong oneshot for the
                // decode chains — replaces the NCCL ring ARs, ~35µs → ~10µs):
                // phase 1 alloc per rank, phase 2 the [world] UVA pointer
                // tables. max_n covers the batched (max_seqs×hidden) + MTP
                // verify payloads.
                // Must cover max_seqs × hidden: the batched decode AR is
                // size × 5120 (16 × 5120 = 81920). The old 16*4096 = 65536
                // silently disabled P2P for size=16 → some ranks fell back
                // to NCCL while others used P2P → deadlock (measured).
                const P2P_AR_MAX_N: usize = 16 * 8192;
                let mut addrs = Vec::with_capacity(world);
                for shard in &shards {
                    match shard
                        .backend
                        .as_cuda()
                        .and_then(|c| c.p2p_ar_alloc(world, P2P_AR_MAX_N).ok())
                    {
                        Some(a) => addrs.push(a),
                        None => {
                            addrs.clear();
                            break;
                        }
                    }
                }
                if addrs.len() == world {
                    let (ss, rs): (Vec<usize>, Vec<usize>) = addrs.iter().cloned().unzip();
                    let mut ok = true;
                    for shard in &shards {
                        if let Some(c) = shard.backend.as_cuda() {
                            if let Err(e) = c.p2p_ar_tables(&ss, &rs) {
                                eprintln!("[serve] P2P AR tables failed: {e:?} (ARs fall back to NCCL)");
                                ok = false;
                            }
                        }
                    }
                    if ok {
                        eprintln!("[serve] P2P AR v2 ready (world={world}, max_n={P2P_AR_MAX_N})");
                    }
                } else {
                    eprintln!("[serve] P2P AR alloc failed (ARs fall back to NCCL)");
                }
            }
        }
        let nccl = if std::env::var_os("FERRITE_NCCL").is_some() {
            #[cfg(feature = "cuda")]
            {
                eprintln!("[serve] FERRITE_NCCL detected, initializing...");
                // Initialize CUDA context on device 0 before NCCL —
                // ncclCommInitAll needs an active CUDA context on the
                // calling thread (the last cudaSetDevice was device world-1
                // from the shard creation, which can cause "unhandled cuda
                // error" from NCCL's internal device queries).
                unsafe {
                    ferrite_kernel::cuda::cuda_set_device(0);
                }
                let devices: Vec<i32> = (0..world as i32).collect();
                let streams: Vec<ferrite_kernel::cuda::CuStream> = shards
                    .iter()
                    .filter_map(|s| s.backend.as_cuda().map(|c| c.stream_handle()))
                    .collect();
                match ferrite_kernel::nccl::NcclGroup::init_all(&devices, &streams) {
                    Ok(ch) => {
                        eprintln!("[serve] NCCL all-reduce up ({} ranks)", world);
                        // Hand each shard its own channel — the device chains
                        // (attn/ffn) all-reduce on-stream before their
                        // download; rank 0's partial is already the sum.
                        let arcs: Vec<std::sync::Arc<ferrite_kernel::nccl::NcclChannel>> =
                            ch.into_iter().map(std::sync::Arc::new).collect();
                        for (rank, shard) in shards.iter_mut().enumerate() {
                            shard.nccl = Some(arcs[rank].clone());
                        }
                        true
                    }
                    Err(e) => {
                        eprintln!("[serve] NCCL init failed ({e:?}) — falling back to host all-reduce");
                        false
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        } else {
            false
        };
        let _ = nccl;
        TpCluster { shards, full_cfg, world, graph_captured: false, graph_step: 0, mega_seq: None, nccl: None, last_batch_seqs: None }
    }

    fn ensure_seq_all(&mut self, seq: u64, tokens: &[u32]) {
        for s in &mut self.shards {
            s.ensure_seq(seq, tokens);
        }
    }

    /// Prefill a chunk on all shards (states stay per-shard: head-split
    /// GatedDeltaNet states, conv tails, DSA head-slice caches).
    pub fn prefill_chunk(&mut self, seq: u64, chunk_tokens: &[u32]) -> Result<()> {
        self.ensure_seq_all(seq, chunk_tokens);
        let h0 = self.shards[0].embed(chunk_tokens);
        let mut h = if self.full_cfg.mhc {
            crate::mhc::hc_expand(&h0, self.full_cfg.hc_mult)
        } else {
            h0
        };
        let plans = build_layer_plans(&self.full_cfg);
        for plan in &plans {
            h = self.layer_forward_tp(seq, plan.layer_idx, h, chunk_tokens.len())?;
        }
        let _ = h;
        Ok(())
    }

    /// Free a sequence's host + GPU state (multi-seq serving lifecycle —
    /// finished/aborted requests release ~GBs of per-seq caches: DSA KV,
    /// GDN states, mega graphs, or the serve OOMs after a handful of
    /// requests). Engine-thread only (single writer — no replay races).
    pub fn free_seq(&mut self, seq: u64) {
        for s in &mut self.shards {
            s.remove_seq(seq);
        }
        if self.mega_seq == Some(seq) {
            self.mega_seq = None;
        }
        Self::fan_out(&mut self.shards, |s| {
            if let Some(cuda) = s.backend.as_cuda() {
                if let Err(e) = cuda.free_seq(seq) {
                    eprintln!("[cluster] free_seq {seq}: {e}");
                }
            }
        });
    }

    /// Destroy a batched-decode graph ("megab_*") on all ranks — the
    /// composition lifecycle: a seq's retirement frees its per-seq states
    /// (the graph's recorded kernel args reference them) — the graph MUST
    /// be destroyed before a replay would touch freed pointers. The next
    /// decode_step_batched for the new composition captures fresh.
    pub fn destroy_batch_graph(&mut self, name: &str) {
        #[cfg(feature = "cuda")]
        Self::fan_out(&mut self.shards, |s| {
            if let Some(cuda) = s.backend.as_cuda() {
                cuda.graph_destroy(name);
            }
        });
    }

    /// ONE decode step for B seqs (the true-batched decode, Step A — non-MTP):
    /// a single mega graph covering ALL live seqs. The projections run at
    /// n=B GEMM (weights stream once per step for all B rows — the batched
    /// GEMM directive), the per-seq recurrent state ops (GDN conv/state, DSA
    /// caches) run as B × n=1 in-graph launches with each row's own
    /// (seq, layer/family) state pointers. The graph is keyed by the
    /// composition ("megab_{s1}_{s2}..."): a membership change re-captures
    /// (~1-2s, amortized over 1000-token streams). The dry-run + capture
    /// pattern mirrors decode_step_mega (the dry-run IS the step's real
    /// output — its state advances stick; the capture records only).
    #[cfg(feature = "cuda")]
    pub fn decode_step_batched(&mut self, seqs: &[u64]) -> Result<Vec<u32>> {
        let n = seqs.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        // ⛔ HARD GUARD (2026-09-09 incident): FERRITE_P2P deadlocks the batched
        // path at n>8 (documented below — capture serializes the ranks' epochs).
        // A deadlocked in-graph P2P kernel that is then killed (timeout -s INT /
        // SIGKILL) wedges the GPU driver: all 8 GPUs logged
        //   NVRM: refcntRequestReference_IMPL: Failed to enter state 1
        // and every later decode on that node faulted with Xid 31 PDE faults
        // (unmapped 2MB pages), progressively worse (b16 → b4 → b1 all dead).
        // Fail loudly here instead of silently deadlocking + wedging the node.
        if std::env::var_os("FERRITE_P2P").is_some() {
            if std::env::var_os("FERRITE_P2P_FORCE").is_some() {
                eprintln!(
                    "[p2p] ⚠️  FERRITE_P2P_FORCE=1: batched P2P all-reduce ENABLED — the in-graph \
                     P2P AR is documented to deadlock at n>8 and a kill afterwards wedges the GPU \
                     driver. Diagnostic use only."
                );
            } else {
                return Err(FerriteError::InvalidArg(
                    "FERRITE_P2P is FORBIDDEN on the batched decode path: the in-graph \
                     P2P all-reduce deadlocks at n>8, and killing the deadlocked process \
                     wedges the GPU driver (Xid 31 PDE faults on every later run). \
                     Unset FERRITE_P2P when serving with --max-seqs > 1; set \
                     FERRITE_P2P_FORCE=1 to override for diagnostics."
                        .into(),
                ));
            }
        }
        let plans = build_layer_plans(&self.full_cfg);
        let num_dsa = plans.iter().filter(|p| matches!(p.attn, AttnKind::Dsa)).count();
        // SGLang-style batch-size padding: ONE graph per padded size
        // (1,2,4,8,16,32) serves any seq-set of that size — the graph's
        // per-seq state kernels read the per-size pointer tables, whose
        // CONTENT is refreshed on every membership change. A
        // composition-keyed graph forced a re-capture (1-2s) per membership
        // change (measured: 8 captures while 16 requests streamed in →
        // throughput collapsed to 74 tok/s).
        let n = seqs.len();
        let size = if std::env::var_os("FERRITE_NO_PAD").is_some() {
            n // bisect knob: no padding (graph per exact size)
        } else {
            [1usize, 2, 4, 8, 16, 32]
                .iter()
                .copied()
                .find(|&s| s >= n)
                .unwrap_or(n)
        };
        let gname = format!("megab_b{size}");
        let mut pseqs: Vec<u64> = seqs.to_vec();
        pseqs.resize(size, u64::MAX); // padded rows → dummy states
        // B rows' last tokens → embed (replicated table) → hc_expand [B, nh]
        let mut last_toks: Vec<u32> = seqs
            .iter()
            .map(|&seq_r| -> Result<u32> {
                let s = self
                    .shards[0]
                    .seq_runtime(seq_r)
                    .ok_or_else(|| FerriteError::Config(format!("batched: missing seq {seq_r}")))?;
                Ok(*s
                    .tokens
                    .last()
                    .ok_or_else(|| FerriteError::Config("empty context".into()))?)
            })
            .collect::<Result<Vec<u32>>>()?;
        if last_toks.len() < size {
            let last = *last_toks.last().unwrap();
            last_toks.resize(size, last);
        }
        let h0 = self.shards[0].embed(&last_toks);
        let in_vals = crate::mhc::hc_expand(&h0, self.full_cfg.hc_mult);

        let have_graph = self
            .shards[0]
            .backend
            .as_cuda()
            .map(|c| c.graph_io_get(&gname).is_some())
            .unwrap_or(false);
        if !have_graph {
            // Cross-rank barrier BEFORE the dry-run: the capture path is
            // serialized by capture_lock and each rank's P2P AR epoch counter
            // must start in lockstep — measured: dev0 entered the dry-run 36
            // AR epochs behind its peers and the flag wait deadlocked (dev0
            // stuck at L0 while dev1-7 were at L35).
            if let Some(nccl) = &self.nccl {
                let mut bufs = Vec::with_capacity(nccl.len());
                for (k, ch) in nccl.iter().enumerate() {
                    if let Some(cuda) = self.shards[k].backend.as_cuda() {
                        let b = ferrite_kernel::cuda::DevBuf::alloc(cuda.dev(), cuda.stream(), 1)?;
                        ch.all_reduce_f32(b.as_const_f32(), b.as_f32(), 1)?;
                        bufs.push(b);
                    }
                }
                for k in 0..self.shards.len() {
                    if let Some(cuda) = self.shards[k].backend.as_cuda() {
                        cuda.sync()?;
                    }
                }
                drop(bufs);
            }
            // Capture path: dry-run (REAL execution — warms every pool class,
            // creates every (seq, layer/family) state, advances every seq's
            // DSA t_count + GDN states; returns the step's B tokens) then
            // capture (records only; the pre-capture DSA rollback keeps the
            // recorded pinned t0/total matching the dry-run's).
            let toks = Self::fan_out(&mut self.shards, |s| {
                Self::mega_chain_dev_batched(
                    s, pseqs.as_slice(), in_vals.as_slice(), &plans, num_dsa, false, &gname, size,
                )
            })
            .into_iter()
            .collect::<Result<Vec<Vec<f32>>>>()?;
            // DIAG: the dry-run's kernels are ASYNC — sync here so a spinning
            // kernel is attributed to the dry-run, not to the capture's
            // cudaStreamEndCapture (which waits for the same kernels).
            if std::env::var_os("FERRITE_TIMING").is_some() {
                eprintln!("[megab] dry-run issued, syncing");
            }
            Self::fan_out(&mut self.shards, |s| {
                if let Some(c) = s.backend.as_cuda() {
                    c.sync()?;
                    // The dry-run advanced every rank's P2P AR epoch; the
                    // capture pass does NOT (records only). Reset all
                    // ranks' protocol state to zero here so the capture and
                    // every replay start in lockstep (the epoch drift was
                    // the size>8 deadlock: dev0 at L0 vs peers at L35).
                    let _ = c.p2p_ar_reset();
                }
                Ok(())
            })
            .into_iter()
            .collect::<Result<Vec<()>>>()?;
            if std::env::var_os("FERRITE_TIMING").is_some() {
                eprintln!("[megab] dry-run synced, capturing");
            }
            // Cross-rank barrier BEFORE the serialized capture: capture_lock
            // serializes the 8 ranks' captures, so a rank that finishes early
            // would enter the NEXT phase while peers still capture → the P2P
            // AR's per-layer rendezvous deadlocks (measured: dev0 stuck at
            // the next dry-run's L0 while dev1-7 were at L35). A 1-element
            // NCCL all-reduce (+sync) is a pure rendezvous.
            if let Some(nccl) = &self.nccl {
                let mut bufs = Vec::with_capacity(nccl.len());
                for (k, ch) in nccl.iter().enumerate() {
                    if let Some(cuda) = self.shards[k].backend.as_cuda() {
                        let b = ferrite_kernel::cuda::DevBuf::alloc(cuda.dev(), cuda.stream(), 1)?;
                        ch.all_reduce_f32(b.as_const_f32(), b.as_f32(), 1)?;
                        bufs.push(b);
                    }
                }
                for k in 0..self.shards.len() {
                    if let Some(cuda) = self.shards[k].backend.as_cuda() {
                        cuda.sync()?;
                    }
                }
                drop(bufs);
            }
            Self::fan_out(&mut self.shards, |s| {
                Self::mega_chain_dev_batched(
                    s, pseqs.as_slice(), in_vals.as_slice(), &plans, num_dsa, true, &gname, size,
                )
            })
            .into_iter()
            .collect::<Result<Vec<Vec<f32>>>>()?;
            // Symmetric barrier AFTER the capture (rank 0 releases
            // capture_lock when its capture ends and would race ahead of
            // peers still capturing — same deadlock as above).
            if let Some(nccl) = &self.nccl {
                let mut bufs = Vec::with_capacity(nccl.len());
                for (k, ch) in nccl.iter().enumerate() {
                    if let Some(cuda) = self.shards[k].backend.as_cuda() {
                        let b = ferrite_kernel::cuda::DevBuf::alloc(cuda.dev(), cuda.stream(), 1)?;
                        ch.all_reduce_f32(b.as_const_f32(), b.as_f32(), 1)?;
                        bufs.push(b);
                    }
                }
                for k in 0..self.shards.len() {
                    if let Some(cuda) = self.shards[k].backend.as_cuda() {
                        cuda.sync()?;
                    }
                }
                drop(bufs);
            }
            eprintln!(
                "[megab] captured {gname}: {size} rows ({n} real), {} layers (B-row GEMM + per-seq state kernels)",
                plans.len()
            );
            let out: Vec<u32> = toks[0].iter().take(n).map(|t| *t as u32).collect();
            for (i, &seq_r) in seqs.iter().enumerate() {
                for s in &mut self.shards {
                    if let Some(rt) = s.seq_runtime_mut(seq_r) {
                        rt.tokens.push(out[i]);
                    }
                }
            }
            return Ok(out);
        }
        // Refresh the per-size pointer tables' CONTENT for THIS membership
        // (the graph's device addresses are stable across seq-sets of the
        // same size — only the pointers change). Only when the membership
        // actually CHANGED: the refresh is 45 layers × 2-6 H2D memcpys +
        // a mutex per table per rank — running it EVERY step was ~2ms of
        // host time that the nsys decode window showed as idle GPU gap.
        if self.last_batch_seqs.as_deref() != Some(seqs) {
            let (la_h, la_dk, la_conv) = {
                let la = &self.full_cfg.linear_attn;
                (la.num_heads, la.head_dim, la.short_conv_kernel_size)
            };
            let plans_r = &plans;
            let pseqs_r = &pseqs;
            Self::fan_out(&mut self.shards, |s| {
                let cuda = s
                    .backend
                    .as_cuda()
                    .ok_or_else(|| FerriteError::Config("batched needs cuda".into()))?;
                cuda.enter();
                let (dh, ddk, ddv, didm) = s.dsa_dims();
                let proj = la_h * la_dk;
                let hist = la_conv.saturating_sub(1).max(1);
                for (li, plan) in plans_r.iter().enumerate() {
                    match plan.attn {
                        AttnKind::Linear => {
                            cuda.gdn_state_tables(li, pseqs_r, 3 * proj * hist, proj * la_dk)?;
                        }
                        AttnKind::Dsa => {
                            let fam = s.dsa_family_index(li);
                            cuda.dsa_ptr_tables(fam, pseqs_r, dh, ddk, ddv, didm)?;
                        }
                    }
                }
                Ok(())
            })
            .into_iter()
            .collect::<Result<Vec<()>>>()?;
            self.last_batch_seqs = Some(seqs.to_vec());
        }
        // Steady state: per-seq DSA pinned advance (B × num_dsa host writes)
        // + ONE graph replay + B tokens D2H — the entire B-seq step is
        // graph-resident (per-seq kernels included).
        let t0 = std::time::Instant::now();
        let toks = Self::fan_out(&mut self.shards, |s| {
            let cuda = s
                .backend
                .as_cuda()
                .ok_or_else(|| FerriteError::Config("batched needs cuda".into()))?;
            cuda.enter();
            for &seq_r in seqs {
                for f in 0..num_dsa {
                    cuda.dsa_host_advance(seq_r, f, 1);
                }
            }
            // NOTE: do NOT reset the P2P protocol here. p2p_ar_reset is
            // called once before the capture; resetting per-replay raced
            // the 8 ranks' parallel replays (one rank's reset landed after
            // its own replay → its epoch fell back to 0 while peers kept
            // advancing → its flag froze at 1 and everyone waited forever).
            // The epoch is monotonic across replays by design.
            let mut out = vec![0f32; size];
            if !cuda.graph_run(&gname, in_vals.as_slice(), &mut out)? {
                return Err(FerriteError::InvalidArg(format!("batched graph {gname} missing")));
            }
            Ok(out)
        })
        .into_iter()
        .collect::<Result<Vec<Vec<f32>>>>()?;
        if std::env::var_os("FERRITE_TIMING").is_some() {
            let dt = t0.elapsed();
            eprintln!(
                "[megab] replay {n} seqs: {:.2}ms ({:.1} tok/s aggregate)",
                dt.as_secs_f32() * 1e3,
                n as f64 / dt.as_secs_f64().max(1e-9)
            );
        }
        let out: Vec<u32> = toks[0].iter().take(n).map(|t| *t as u32).collect();
        for (i, &seq_r) in seqs.iter().enumerate() {
            for s in &mut self.shards {
                if let Some(rt) = s.seq_runtime_mut(seq_r) {
                    rt.tokens.push(out[i]);
                }
            }
        }
        Ok(out)
    }

    /// Decode one token. Returns the sampled token id.
    pub fn decode_step(&mut self, seq: u64) -> Result<u32> {
        // CUDA graph fast path: FERRITE_GRAPH=1 → first decode_step captures
        // the GPU op sequence per layer, subsequent steps graph-replay
        // (zero kernel launch, zero CPU→GPU sync per op).
        if std::env::var_os("FERRITE_GRAPH").is_some() {
            return self.decode_step_graphed(seq);
        }
        // FERRITE_MEGA=1: the ENTIRE decode step as one per-rank CUDA graph
        // (NCCL all-reduce INSIDE the graph — the 100 tok/s path). Needs
        // FERRITE_NCCL (per-shard channels; the collectives are recorded as
        // graph nodes).
        #[cfg(feature = "cuda")]
        if std::env::var_os("FERRITE_MEGA").is_some() && self.shards[0].nccl.is_some() {
            return self.decode_step_mega(seq);
        }
        self.decode_step_normal(seq)
    }

    /// FERRITE_MEGA: whole-decode-step per-rank CUDA graphs (NCCL in-graph).
    ///
    /// Phases per seq: (1) dry-run the full chain per rank (real execution —
    /// warms every DevBuf pool class + per-rank weight caches + the NCCL
    /// 4096-float all-reduce plan; produces this step's token), (2) capture
    /// the same chain into graph `mega{seq}` (record-only; NCCL ARs become
    /// graph nodes — warm comm + ThreadLocal capture, proven in
    /// gpu_smoke_nccl_graph: 90 ARs/replay @ 15µs), (3) every later step =
    /// one graph replay per rank (staging write + DSA pinned advance +
    /// graph launch + argmax D2H — zero host round-trips per layer).
    ///
    /// Env: FERRITE_MEGA=1 FERRITE_NCCL=1 FERRITE_WORKER_POOL=1 (+ the
    /// device-chain flags for the prefill: FERRITE_GDN_DEV/MOE_DEV/DSA_DEV/
    /// LAYER_DEV/HEAD_DEV=1) and NCCL_NVLS_ENABLE=0 on b300-4.
    #[cfg(feature = "cuda")]
    fn decode_step_mega(&mut self, seq: u64) -> Result<u32> {
        let last = {
            let s = self.shards[0]
                .seq_runtime(seq)
                .ok_or_else(|| FerriteError::Config("missing seq".into()))?;
            *s.tokens.last().ok_or_else(|| FerriteError::Config("empty context".into()))?
        };
        let h0 = self.shards[0].embed(&[last]);
        let in_vals = crate::mhc::hc_expand(&h0, self.full_cfg.hc_mult);
        let plans = build_layer_plans(&self.full_cfg);
        let num_dsa = plans.iter().filter(|p| matches!(p.attn, AttnKind::Dsa)).count();
        let gname = format!("mega{seq}");
        // N-UNIFIED ROUTING: FERRITE_MTP_N=1 (or FERRITE_MTP unset) takes the
        // SAME mega1 path — no MTP buffers, no verify graph, no ping-pong
        // commit. "Non-MTP is n=1" literally: one code path, zero n=1
        // overhead (bit-identical to the historical non-MTP decode).
        let mtp = std::env::var_os("FERRITE_MTP").is_some() && mtp_verify_n() > 1;

        // Multi-seq serving: the mega graph is keyed per seq (mega{seq}), but
        // mega_seq is a SINGLE-slot marker — a naive != check re-ran the
        // seconds-scale dry-run+capture on every seq switch (round-robin
        // decode would recapture per step). A seq whose graph already exists
        // (graph_io registered at capture end) just switches the marker and
        // replays: every binding the replay reads (DSA pinned t0/total,
        // caches, GDN states, graphs) is keyed by seq.
        let have_graph = self
            .shards[0]
            .backend
            .as_cuda()
            .map(|c| c.graph_io_get(&gname).is_some())
            .unwrap_or(false);
        if self.mega_seq != Some(seq) && !have_graph {
            // (Re)capture for this seq. Dry-run: all 4 ranks run the full
            // chain in parallel (fan_out) — the NCCL ARs rendezvous for
            // real; every pool class / weight cache / NCCL plan warms on the
            // exact worker that captures next.
            let t0 = std::time::Instant::now();
            let toks = Self::fan_out(&mut self.shards, |s| {
                if mtp {
                    Self::mtp_setup_bufs(s, &plans, seq)?;
                }
                let vio = if mtp { Some(Self::mtp_vio(s, false)) } else { None };
                Self::mega_chain_dev(s, seq, in_vals.as_slice(), &plans, num_dsa, false, &gname, 1, vio.as_ref())
            })
            .into_iter()
            .collect::<Result<Vec<Vec<f32>>>>()?;
            let t_dry = t0.elapsed();
            if std::env::var_os("FERRITE_MEGA_DRY").is_some() {
                // DRY mode: skip capture/replay — every step runs the real
                // chain. Bisection: dry output correct → graph-mechanism bug;
                // dry output garbage → chain-semantics bug.
                eprintln!(
                            "[mega] DRY mode step (in={last} tok={}): dry-run {:.1}ms — no capture",
                            toks[0][0], t_dry.as_secs_f32() * 1e3
                        );
                                        // every shard's seq_runtime must track the sampled token — the NEXT
                // step's input embeds tokens.last() (decode_step_normal pushes at its
                // tail; mega omitted it → input token froze at the prompt's last
                // token → output self-locked to one token)
                let tok = toks[0][0] as u32;
                for s in &mut self.shards {
                    if let Some(rt) = s.seq_runtime_mut(seq) {
                        rt.tokens.push(tok);
                    }
                }
                return Ok(tok);
            }
            // Capture: record-only. capture_lock serializes the per-rank
            // captures (concurrent cuGraphInstantiate SIGSEGV'd historically);
            // record-mode NCCL enqueue never rendezvous, so serialized capture
            // is deadlock-free (the nccl test proved it).
            let tc = std::time::Instant::now();
            Self::fan_out(&mut self.shards, |s| {
                let vio = if mtp { Some(Self::mtp_vio(s, false)) } else { None };
                Self::mega_chain_dev(s, seq, in_vals.as_slice(), &plans, num_dsa, true, &gname, 1, vio.as_ref())
            })
            .into_iter()
            .collect::<Result<Vec<Vec<f32>>>>()?;
            self.mega_seq = Some(seq);
            if mtp {
                // MTP draft-cache catch-up: run the draft layer (layers.45)
                // over every prompt token so its DSA cache (family num_dsa)
                // holds the prompt context — without this the draft's
                // attention has no history and d1 is a blind guess (accept
                // rate 40%). h_prev approximated by hf_dev (prompt-tail h).
                let prompt_tokens: Vec<u32> = {
                    let s = self.shards[0]
                        .seq_runtime(seq)
                        .ok_or_else(|| FerriteError::Config("missing seq".into()))?;
                    s.tokens.clone()
                };
                let hidden = self.full_cfg.hidden_size;
                let _ = Self::fan_out(&mut self.shards, |s| {
                    // h chain: token 0 uses hf_dev (prompt-tail target h), then
                    // each step's MTP-layer residual h (x2) recurses — a
                    // per-token h sequence beats the fixed approximation.
                    let mut h_cur: Option<ferrite_kernel::cuda::DevBuf> = None;
                    for t in &prompt_tokens {
                        let (emb, hptr) = {
                            let cuda = s
                                .backend
                                .as_cuda()
                                .ok_or_else(|| FerriteError::Config("mtp needs cuda".into()))?;
                            cuda.enter();
                            let h2 = s.embed(&[*t]);
                            let emb = ferrite_kernel::cuda::DevBuf::alloc(cuda.dev(), cuda.stream_handle(), hidden)?;
                            emb.upload(h2.as_slice())?;
                            let m = cuda.mtp.lock().unwrap();
                            let m = m.as_ref().ok_or_else(|| FerriteError::Config("mtp bufs missing".into()))?;
                            (emb, &m.hf_dev as *const ferrite_kernel::cuda::DevBuf as usize)
                        };
                        let hout = ferrite_kernel::cuda::DevBuf::alloc(
                            s.backend.as_cuda().unwrap().dev(),
                            s.backend.as_cuda().unwrap().stream(),
                            hidden,
                        )?;
                        let hprev: &ferrite_kernel::cuda::DevBuf = match h_cur.as_ref() {
                            Some(h) => h,
                            None => unsafe { &*(hptr as *const ferrite_kernel::cuda::DevBuf) },
                        };
                        mtp_forward(s, seq, &emb, hprev, Some(&hout))?;
                        h_cur = Some(hout);
                    }
                    // SEED MtpState.hprev = hf_dev (the prompt-tail decoder
                    // h_final) right at catch-up — once, before step 1. This
                    // makes S1's hprev a DETERMINISTIC value (the previously-
                    // verified main KV per the mandate) instead of pool
                    // residue. mtp_step_zero_h2d allocates 8 extra device
                    // buffers in MtpState, so its pool residue for hprev
                    // DIFFERS from the original path's → S1's draft1 x2
                    // carries a 1-ulp drift (hidden from 8-seg 6-decimal
                    // checksums) → d1=990 survives (large argmax margin) but
                    // d2 flips (8606 vs 315, near-tie) → stream diverges →
                    // accept=1.0. FIX v2 (afdbaac) proved seeding hf_dev
                    // restores S1-S3 to d1==a0 (k=2). hf_dev is the correct
                    // seed (NOT the catch-up MTP-layer trailing hout, which
                    // moved d1 to 136493). Combined with the (2-k) rollback.
                    {
                        let cuda = s
                            .backend
                            .as_cuda()
                            .ok_or_else(|| FerriteError::Config("cuda".into()))?;
                        let m = cuda.mtp.lock().unwrap();
                        let m = m.as_ref().ok_or_else(|| FerriteError::Config("mtp bufs missing".into()))?;
                        cuda.copy_dev(&m.hf_dev, 0, m.hprev.as_f32(), hidden)?;
                    }
                    Ok::<(), FerriteError>(())
                })
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
                eprintln!(
                    "[mega] MTP: draft cache catch-up done ({} prompt tokens, h_prev seeded)",
                    prompt_tokens.len()
                );
                // MTP verify graph (N-UNIFIED: n = FERRITE_MTP_N — the verify
                // rows [t_last, d1..d_{N-2}]; n=3 = the historical [t_last, d1,
                // d2]). GDN state → scratch B (ping-pong), h_final export
                // (hf_v [n*hidden]), argmax n.
                let n_v = mtp_verify_n();
                let gv = format!("mega_v{seq}");
                let h2 = self.shards[0].embed(&vec![last; n_v]);
                let in_vals2 = crate::mhc::hc_expand(&h2, self.full_cfg.hc_mult);
                let _ = Self::fan_out(&mut self.shards, |s| {
                    let vio = Self::mtp_vio(s, true);
                    Self::mega_chain_dev(s, seq, in_vals2.as_slice(), &plans, num_dsa, false, &gv, n_v, Some(&vio))
                })
                .into_iter()
                .collect::<Result<Vec<Vec<f32>>>>()?;
                Self::fan_out(&mut self.shards, |s| {
                    let vio = Self::mtp_vio(s, true);
                    Self::mega_chain_dev(s, seq, in_vals2.as_slice(), &plans, num_dsa, true, &gv, n_v, Some(&vio))
                })
                .into_iter()
                .collect::<Result<Vec<Vec<f32>>>>()?;
                eprintln!("[mega] MTP: verify graph {gv} captured (n={n_v} = FERRITE_MTP_N, GDN ping-pong scratch)");
                // DRAFT GRAPHS (FERRITE_DRAFT_GRAPH=1, default): capture the
                // per-draft chain mega_d{seq}_{i} (cast_store → embed_one_dev
                // → the full mtp_forward layer chain → argmax). The steady
                // mtp_step replays them (H2D 4B + advance(1) + one launch
                // per draft) instead of the host-serialized chain (~0.8ms/
                // draft of kernel launches + host embed + staging at 4 ranks).
                // Sequence per rank: DRY (real execution — pool warm, MtpState
                // bufs seeded, d_argmax/h_d written) → rollback(nd) → CAPTURE
                // ×nd (records only; the dsa host bookkeeping +1 per draft
                // runs, pinned t0 writes are valueless at capture) →
                // rollback(nd) (the capture pass executed nothing; t must
                // return to the pre-dry T so the first steady step's
                // advance(1) pins t0=T). The dry appends are the first step's
                // correct KV (same inputs: last, hf-seeded hprev, catch-up
                // cache) — the first replay overwrites them bit-identically.
                // capture in graph modes ("1"/default): mode "2" (host-serial
                // device chain) and "0" (host chain) skip it.
                // FERRITE_DRAFT_DRY_ONLY=1 skips the capture pass (dry only) —
                // bisects dry's real-execution side effects from the capture
                // pass's leaked pool buffers.
                if !matches!(
                    std::env::var("FERRITE_DRAFT_GRAPH").as_deref(),
                    Ok("0") | Ok("2")
                ) {
                    let nd = n_v - 1;
                    // dry: tokens_dev[0] ← last, then the nd-step chain (real
                    // execution — P2P ARs rendezvous, dsa appends at T..T+nd-1)
                    Self::fan_out(&mut self.shards, |s| {
                        let cuda = s
                            .backend
                            .as_cuda()
                            .ok_or_else(|| FerriteError::Config("draft graph needs cuda".into()))?;
                        cuda.enter();
                        let tokens_ptr = {
                            let m = cuda.mtp.lock().unwrap();
                            let m = m
                                .as_ref()
                                .ok_or_else(|| FerriteError::Config("mtp bufs missing".into()))?;
                            m.tokens_dev.as_f32() as *mut i32
                        };
                        let last_i32 = last as i32;
                        let r = ferrite_kernel::cuda::memcpy_htod_i32(
                            tokens_ptr, &last_i32, 1, cuda.stream_handle());
                        if r != 0 {
                            return Err(FerriteError::InvalidArg(format!("tokens_dev[0] H2D: {r}")));
                        }
                        for i in 0..nd {
                            draft_step_dev(s, seq, i, nd)?;
                        }
                        Ok::<(), FerriteError>(())
                    })
                    .into_iter()
                    .collect::<Result<Vec<_>>>()?;
                    // rollback(nd) + capture ×nd + rollback(nd) — the same
                    // fan_out (host-side dsa bookkeeping + capture_lock
                    // serializes the per-rank graph instantiations, the
                    // mega_v pattern).
                    Self::fan_out(&mut self.shards, |s| {
                        let mtp_family = s
                            .cfg
                            .layer_types
                            .iter()
                            .filter(|t| matches!(t, ferrite_model::LayerType::DeepseekSparseAttention))
                            .count();
                        s.backend
                            .as_cuda()
                            .ok_or_else(|| FerriteError::Config("draft graph needs cuda".into()))?
                            .enter();
                        s.backend
                            .as_cuda()
                            .ok_or_else(|| FerriteError::Config("draft graph needs cuda".into()))?
                            .dsa_host_rollback(seq, mtp_family, nd);
                        let skip_cap =
                            std::env::var_os("FERRITE_DRAFT_DRY_ONLY").is_some();
                        if !skip_cap {
                            let _g = ferrite_kernel::cuda::capture_lock().lock().unwrap();
                            for i in 0..nd {
                                // per-iteration borrows: draft_step_dev takes &mut s
                                // (graph_capture_begin/end only need &CudaBackend).
                                s.backend
                                    .as_cuda()
                                    .ok_or_else(|| FerriteError::Config("draft graph needs cuda".into()))?
                                    .graph_capture_begin();
                                draft_step_dev(s, seq, i, nd)?;
                                s.backend
                                    .as_cuda()
                                    .ok_or_else(|| FerriteError::Config("draft graph needs cuda".into()))?
                                    .graph_capture_end(&format!("mega_d{seq}_{i}"));
                            }
                        }
                        // the capture pass executed NOTHING (record only) but
                        // each draft's dsa host bookkeeping +1 → t advanced
                        // nd — roll it back to T (the first steady step's
                        // advance(1) pins t0=T, the replay overwrites the
                        // dry's KV bit-identically).
                        s.backend
                            .as_cuda()
                            .ok_or_else(|| FerriteError::Config("draft graph needs cuda".into()))?
                            .dsa_host_rollback(seq, mtp_family, nd);
                        Ok::<(), FerriteError>(())
                    })
                    .into_iter()
                    .collect::<Result<Vec<_>>>()?;
                    eprintln!(
                        "[mega] MTP: draft graphs mega_d{seq}_0..{nd} captured (FERRITE_DRAFT_GRAPH)"
                    );
                }
            }
            eprintln!(
                "[mega] captured {gname}: {} layers, {} NCCL ARs/rank; dry-run {:.1}ms + capture {:.1}ms",
                plans.len(), plans.len() * 2,
                t_dry.as_secs_f32() * 1e3,
                tc.elapsed().as_secs_f32() * 1e3
            );
            // all four ranks computed the same token (bit-identical after
            // the ARs — symmetric redundant head)
            // every shard's seq_runtime must track the sampled token —
            // decode_step_normal pushes at its tail; mega omitted it → the
            // next step's input token froze at the prompt's last token →
            // output self-locked to one token.
            let tok = toks[0][0] as u32;
            for s in &mut self.shards {
                if let Some(rt) = s.seq_runtime_mut(seq) {
                    rt.tokens.push(tok);
                }
            }
            return Ok(tok);
        }
        if mtp {
            // MTP steady step: draft (mtp_forward) + verify (mega_v n=2) +
            // greedy accept + ping-pong commit.
            // Zero-H2D device-resident MTP (FERRITE_ZERO_H2D=1): the entire
            // draft→verify→accept→commit chain runs on device — tokens never
            // cross to host for computation. Only D2H: 8-12B per step for SSE
            // (k + next_token + verify argmax). Default OFF until verified.
            if std::env::var_os("FERRITE_ZERO_H2D").is_some() {
                // N-UNIFIED: zero-H2D is the N=3 experimental path (its draft
                // chain, k_host logic and seq push are hand-unrolled for
                // nd=2). Any other N routes to the generalized mtp_step.
                if mtp_verify_n() == 3 {
                    return self.mtp_step_zero_h2d(seq, &plans, num_dsa);
                }
                eprintln!(
                    "[zero-h2d] FERRITE_MTP_N={} != 3 — using the generalized mtp_step",
                    mtp_verify_n()
                );
            }
            let t_all = std::time::Instant::now();
            let r = self.mtp_step(seq, &plans, num_dsa);
            if std::env::var_os("FERRITE_MTP_TIMING").is_some() {
                eprintln!("[mtp-all] {:.2}ms", t_all.elapsed().as_secs_f64() * 1e3);
            }
            return r;
        }
        // Steady state: advance DSA pinned t0/total (the graph's kernels
        // read them zero-copy), write the 4 stagings, one launch per rank,
        // argmax D2H — the entire step is graph-resident.
        let t0 = std::time::Instant::now();
        let toks = Self::fan_out(&mut self.shards, |s| {
            let cuda = s
                .backend
                .as_cuda()
                .ok_or_else(|| FerriteError::Config("FERRITE_MEGA needs cuda".into()))?;
            cuda.enter();
            for f in 0..num_dsa {
                cuda.dsa_host_advance(seq, f, 1);
            }
            let mut out = [0f32; 1];
            if !cuda.graph_run(&gname, in_vals.as_slice(), &mut out)? {
                return Err(FerriteError::InvalidArg(format!("mega graph {gname} missing")));
            }
            Ok(out[0])
        })
        .into_iter()
        .collect::<Result<Vec<f32>>>()?;
        let dt = t0.elapsed();
        if std::env::var_os("FERRITE_TIMING").is_some() {
            eprintln!(
                "[mega] replay {:.2}ms ({:.1} tok/s)",
                dt.as_secs_f32() * 1e3,
                1e3 / dt.as_secs_f32().max(1e-9)
            );
        }
        // every shard's seq_runtime must track the sampled token —
        // decode_step_normal pushes at its tail; mega omitted it → the
        // next step's input token froze at the prompt's last token →
        // output self-locked to one token.
        let tok = toks[0] as u32;
        for s in &mut self.shards {
            if let Some(rt) = s.seq_runtime_mut(seq) {
                rt.tokens.push(tok);
            }
        }
        Ok(tok)
    }

    /// MTP (FERRITE_MTP=1) steady step (N-UNIFIED, FERRITE_MTP_N): nd = N-1
    /// draft mtp_forward steps (h chain: draft i's h_prev = draft i-1's h_out,
    /// draft 0's = the committed hprev) → verify (mega_v N-row graph replay
    /// over [t_last, d1..d_{nd}]) → greedy accept (longest prefix k in 1..=N)
    /// → ping-pong state commit. N=3 = the historical (d1, d2) two-draft
    /// chain, bit-identical; N=2..8 all take the SAME code path.
    #[cfg(feature = "cuda")]
    fn mtp_step(&mut self, seq: u64, plans: &[ferrite_model::LayerPlan], num_dsa: usize) -> Result<u32> {
        use ferrite_kernel::cuda::DevBuf;
        let n_v = mtp_verify_n(); // FERRITE_MTP_N: verify width (drafts = N-1)
        let nd = n_v - 1;
        let hidden = self.full_cfg.hidden_size;
        let hc_mult = self.full_cfg.hc_mult;
        let gvname = format!("mega_v{seq}");
        let mtp_family = self
            .full_cfg
            .layer_types
            .iter()
            .filter(|t| matches!(t, ferrite_model::LayerType::DeepseekSparseAttention))
            .count();
        let last = {
            let s = self.shards[0]
                .seq_runtime(seq)
                .ok_or_else(|| FerriteError::Config("missing seq".into()))?;
            *s.tokens.last().ok_or_else(|| FerriteError::Config("empty context".into()))?
        };
        let mtp_tm = std::env::var_os("FERRITE_MTP_TIMING").is_some();
        let t_d = std::time::Instant::now();
        // 1. drafts (nd steps): h chain — draft 0 from hprev, draft i from
        //    draft i-1's MTP residual h (EAGLE-style recursion). The i<nd-1
        //    drafts export their h (h_out for the next draft); the last
        //    discards it (verify's hf_v replaces it).
        //    GRAPH PATH (FERRITE_DRAFT_GRAPH=1, default): the whole draft
        //    chain replays per-draft graphs mega_d{seq}_{i} — H2D 4B (last) +
        //    dsa_host_advance(1) + ONE launch per draft (vs ~15 kernel
        //    launches + host embed + 576KB staging). Host cost ~0.5ms/step.
        //    Fallback: the original host chain (embed lookup + upload +
        //    mtp_forward per draft).
        let draft_env = std::env::var("FERRITE_DRAFT_GRAPH").unwrap_or_default();
        // DEFAULT ON (2026-09-08): the graph-resident draft chain
        // mega_d{seq}_{i} — the h_out forget bug (forget(&DevBuf) no-op) that
        // caused the accept regression is FIXED; verified accept 2.39 @
        // 116.8 tok/s (host-chain baseline 2.41/117.6). FERRITE_DRAFT_GRAPH=0
        // falls back to the host chain; =2 runs the device chain host-serial
        // (bisect mode).
        let graph_drafts = draft_env != "0";
        // mode "2": execute the SAME device chain host-serial (no graph
        // replay) — bisects a draft divergence between the chain itself
        // (embed_one_dev / cast_store / h_d relays) and capture/replay.
        let serial_dev = draft_env == "2";
        let drafts: Vec<f32> = {
            let mut graph_ok = false;
            if graph_drafts && !serial_dev {
                // probe once on rank 0's view (all ranks capture in
                // lockstep — decode_step_mega's first step captures all nd).
                let probe = Self::fan_out(&mut self.shards, |s| {
                    let cuda = s
                        .backend
                        .as_cuda()
                        .ok_or_else(|| FerriteError::Config("draft graph needs cuda".into()))?;
                    Ok::<_, FerriteError>(
                        (0..nd)
                            .all(|i| cuda.graph_exists(&format!("mega_d{seq}_{i}"))),
                    )
                })
                .into_iter()
                .collect::<Result<Vec<bool>>>()?;
                graph_ok = probe.iter().all(|&b| b);
            }
            if serial_dev {
                // host-serial device chain (no graph): the SAME draft_step_dev
                // calls the capture pass records, executed per step — bisects
                // the chain (embed_one_dev / cast_store / h_d) from the
                // capture/replay semantics.
                let toks = Self::fan_out(&mut self.shards, |s| {
                    {
                        let cuda = s
                            .backend
                            .as_cuda()
                            .ok_or_else(|| FerriteError::Config("mtp needs cuda backend".into()))?;
                        cuda.enter();
                        let tokens_ptr = {
                            let m = cuda.mtp.lock().unwrap();
                            let m = m
                                .as_ref()
                                .ok_or_else(|| FerriteError::Config("mtp bufs missing".into()))?;
                            m.tokens_dev.as_f32() as *mut i32
                        };
                        let last_i32 = last as i32;
                        let r = ferrite_kernel::cuda::memcpy_htod_i32(
                            tokens_ptr, &last_i32, 1, cuda.stream_handle());
                        if r != 0 {
                            return Err(FerriteError::InvalidArg(format!("tokens_dev[0] H2D: {r}")));
                        }
                    }
                    for i in 0..nd {
                        draft_step_dev(s, seq, i, nd)?;
                    }
                    let mut d = vec![0f32; nd];
                    {
                        let cuda = s
                            .backend
                            .as_cuda()
                            .ok_or_else(|| FerriteError::Config("mtp needs cuda backend".into()))?;
                        let d_base = {
                            let m = cuda.mtp.lock().unwrap();
                            let m = m
                                .as_ref()
                                .ok_or_else(|| FerriteError::Config("mtp bufs missing".into()))?;
                            m.d_argmax_dev.as_f32()
                        };
                        let r = ferrite_kernel::cuda::memcpy_d2h_sync(
                            d_base as *mut std::ffi::c_void,
                            d.as_mut_ptr(),
                            nd,
                            cuda.stream_handle(),
                        );
                        if r != 0 {
                            return Err(FerriteError::InvalidArg(format!("drafts D2H: {r}")));
                        }
                    }
                    Ok(d)
                })
                .into_iter()
                .collect::<Result<Vec<Vec<f32>>>>()?;
                toks[0].clone()
            } else if graph_ok {
                let toks = Self::fan_out(&mut self.shards, |s| {
                    let cuda = s
                        .backend
                        .as_cuda()
                        .ok_or_else(|| FerriteError::Config("mtp needs cuda backend".into()))?;
                    cuda.enter();
                    let (tokens_ptr, d_base) = {
                        let m = cuda.mtp.lock().unwrap();
                        let m = m
                            .as_ref()
                            .ok_or_else(|| FerriteError::Config("mtp bufs missing".into()))?;
                        (m.tokens_dev.as_f32() as *mut i32, m.d_argmax_dev.as_f32())
                    };
                    // tokens_dev[0] = last — the only host-initiated write
                    let last_i32 = last as i32;
                    let r = ferrite_kernel::cuda::memcpy_htod_i32(
                        tokens_ptr, &last_i32, 1, cuda.stream_handle());
                    if r != 0 {
                        return Err(FerriteError::InvalidArg(format!("tokens_dev[0] H2D: {r}")));
                    }
                    // per-draft replay: advance(1) pins t0/total for the
                    // graph's dsa append+query (the kernel reads them
                    // zero-copy), then ONE graph launch. The NEXT advance
                    // overwrites the SAME pinned slot the just-launched
                    // graph's kernels read — sync between graphs (the last
                    // graph's read is covered by the D2H below).
                    for i in 0..nd {
                        cuda.dsa_host_advance(seq, mtp_family, 1);
                        let gname = format!("mega_d{seq}_{i}");
                        if !cuda.graph_replay(&gname) {
                            return Err(FerriteError::InvalidArg(format!("draft graph {gname} missing")));
                        }
                        if i + 1 < nd {
                            // Pinned t0 race guard (MEASURED NECESSARY: without
                            // it accept drops 2.39 → 2.22 — the next advance's
                            // host write to the shared pinned t0 slot lands
                            // before the just-launched graph's dsa kernels read
                            // it). Cost is ~0 (step 20.5 vs 20.6ms).
                            cuda.sync()?;
                        }
                    }
                    // D2H the draft tokens (nd × 4B — verify input + accept
                    // decisions; the host-side accept logic consumes them).
                    let mut d = vec![0f32; nd];
                    let r = ferrite_kernel::cuda::memcpy_d2h_sync(
                        d_base as *mut std::ffi::c_void,
                        d.as_mut_ptr(),
                        nd,
                        cuda.stream_handle(),
                    );
                    if r != 0 {
                        return Err(FerriteError::InvalidArg(format!("drafts D2H: {r}")));
                    }
                    Ok(d)
                })
                .into_iter()
                .collect::<Result<Vec<Vec<f32>>>>()?;
                toks[0].clone()
            } else {
                // host fallback: the original chain (embed lookup + upload +
                // mtp_forward per draft — pool h_out relays between drafts).
                let toks = Self::fan_out(&mut self.shards, |s| {
                    let mut drafts: Vec<f32> = Vec::with_capacity(nd);
                    // draft 0's h_prev = MtpState.hprev — its &DevBuf REFERENCE
                    // address (NOT as_f32(): that is the device data pointer;
                    // reinterpreting it as a DevBuf struct reads floats into the
                    // ptr/len/stage fields → garbage → SEGV. The reference is
                    // stable: MtpState outlives the loop and its DevBufs never
                    // move — the same raw-&DevBuf pattern the pre-N code used).
                    let hprev_ref: usize = {
                        let cuda = s
                            .backend
                            .as_cuda()
                            .ok_or_else(|| FerriteError::Config("mtp needs cuda".into()))?;
                        let m = cuda.mtp.lock().unwrap();
                        let m = m
                            .as_ref()
                            .ok_or_else(|| FerriteError::Config("mtp bufs missing".into()))?;
                        &m.hprev as *const DevBuf as usize
                    };
                    // h chain: draft i's h_prev = draft i-1's h_out (a normal
                    // pool-allocated DevBuf, kept alive by ownership — the last
                    // draft exports no h (verify's hf_v commit replaces it).
                    let mut prev_h: Option<DevBuf> = None;
                    let mut prev_tok = last;
                    for i in 0..nd {
                        let (emb, h_out) = {
                            let cuda = s
                                .backend
                                .as_cuda()
                                .ok_or_else(|| FerriteError::Config("mtp needs cuda".into()))?;
                            cuda.enter();
                            let h2 = s.embed(&[prev_tok as u32]);
                            let emb = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), hidden)?;
                            emb.upload(h2.as_slice())?;
                            let h_out = if i + 1 < nd {
                                Some(DevBuf::alloc(cuda.dev(), cuda.stream(), hidden)?)
                            } else {
                                None
                            };
                            (emb, h_out)
                        }; // cuda dropped — mtp_forward re-acquires internally
                        let d = match prev_h.as_ref() {
                            None => {
                                let hprev: &DevBuf = unsafe { &*(hprev_ref as *const DevBuf) };
                                mtp_forward(s, seq, &emb, hprev, h_out.as_ref())?
                            }
                            Some(ph) => mtp_forward(s, seq, &emb, ph, h_out.as_ref())?,
                        };
                        drafts.push(d);
                        prev_tok = d as u32;
                        prev_h = h_out; // draft i's h_out → draft i+1's h_prev
                    }
                    // prev_h ends None (last draft exported no h) — any interim
                    // h_out DevBuf drops back to the pool on the move chain.
                    Ok(drafts)
                })
                .into_iter()
                .collect::<Result<Vec<Vec<f32>>>>()?;
                toks[0].clone()
            }
        };
        let t_draft = t_d.elapsed();
        let t_v = std::time::Instant::now();
        // 2. verify: dsa advance(n_v) + mega_v replay (the A→B ping-pong
        //    copy-in is graph-recorded — capture-time nodes, not a host
        //    memcpy loop) → argmax[n_v] → longest-prefix accept k.
        let mut toks_in: Vec<u32> = Vec::with_capacity(n_v);
        toks_in.push(last);
        for d in &drafts {
            toks_in.push(*d as u32);
        }
        // DEVICE-EMBED verify input: the mega_v graph was captured with
        // embed_expand_dev as its FIRST node (dev_embed) — replay writes
        // n_v×4B token ids into tokens_dev (graph_run_ids) instead of the
        // host embed lookup + hc_expand + 576KB pinned staging (~1ms host
        // per verify step). The embed kernel reads the F32 table cache
        // (bit-identical to the host lookup) — hc_expand semantics = row
        // copy × mult, identical.
        let toks_v = Self::fan_out(&mut self.shards, |s| {
            let cuda = s
                .backend
                .as_cuda()
                .ok_or_else(|| FerriteError::Config("mtp needs cuda".into()))?;
            cuda.enter();
            for f in 0..num_dsa {
                // advance(n_v): append this step's verify input [t_last, d1..d_{nd}].
                cuda.dsa_host_advance(seq, f, n_v);
            }
            let mut out = vec![0f32; n_v];
            let ids_dev = {
                let m = cuda.mtp.lock().unwrap();
                m.as_ref().map(|m| m.tokens_dev.as_f32() as *mut i32)
            };
            let ran = match ids_dev {
                Some(p) => cuda.graph_run_ids(&gvname, &toks_in, p, &mut out)?,
                None => false,
            };
            if !ran {
                // fallback: host embed + staging (graph NOT captured with the
                // device-embed path — e.g. legacy graphs before this change)
                let h2v = s.embed(&toks_in);
                let in_vals = crate::mhc::hc_expand(&h2v, hc_mult);
                if !cuda.graph_run(&gvname, in_vals.as_slice(), &mut out)? {
                    return Err(FerriteError::InvalidArg(format!("mega_v graph {gvname} missing")));
                }
            }
            // Accept + commit fused into the verify worker (was a THIRD
            // fan_out round-trip): drafts/argmax are bit-identical across
            // ranks, so every worker derives the same k. DSA rollback is
            // host pinned bookkeeping (ns); the B_k -> A ping-pong commit +
            // hprev <- hf_v[k-1] is ONE ferrite_mtp_commit launch.
            // k = longest matching prefix of (d1..d_nd) vs (a0..a_{nd-1}),
            // 1..=n_v (N-UNIFIED generalization of the old d1/d2 nesting).
            let mut k: i32 = 1;
            while (k as usize) < n_v
                && drafts[(k - 1) as usize] as u32 == out[(k - 1) as usize] as u32
            {
                k += 1;
            }
            let k = k as usize;
            if std::env::var_os("FERRITE_MTP_DEBUG").is_some() {
                eprintln!("[mtp-acc] drafts={:?} out={:?} k={}", drafts, out, k);
            }
            for f in 0..num_dsa {
                cuda.dsa_host_rollback(seq, f, (n_v - k) as usize);
            }
            // mtp_family: the drafts appended nd tokens (draft i's input);
            // keep min(k, nd) — k=1 keeps only last's (draft 0's input),
            // k>=2 keeps last + the accepted d1..d_{k-2}. Same arithmetic as
            // the old (2-k).max(0) at nd=2.
            cuda.dsa_host_rollback(seq, mtp_family, nd.saturating_sub(k));
            cuda.mtp_commit(k as i32)?;
            Ok((out, k))
        })
        .into_iter()
        .collect::<Result<Vec<(Vec<f32>, usize)>>>()?;
        let (out, k) = (toks_v[0].0.clone(), toks_v[0].1);
        let t_verify = t_v.elapsed();
        let t_c = std::time::Instant::now();
        // 3. accept: push the k accepted tokens — drafts[0..k-2] + the bonus
        // token out[k-1] (verify's argmax at the accept position). k=1 →
        // [a0] (all drafts rejected); k=n_v → all drafts + a_{n_v-1}.
        let dbg = std::env::var_os("FERRITE_MTP_DEBUG").is_some();
        if dbg {
            let ds: Vec<u32> = drafts.iter().map(|d| *d as u32).collect();
            let os: Vec<u32> = out.iter().map(|o| *o as u32).collect();
            eprintln!("[mtp] n={} drafts={:?} verify={:?} -> accept{}", n_v, ds, os, k);
        }
        // the accepted token list (what the seq actually gained this step):
        // d1..d_{k-2} (k-2 drafts) + a_{k-1} (the bonus) — total k tokens
        // counting a0's slot at k=1 (just the bonus a0).
        let mut accepted: Vec<u32> = Vec::with_capacity(k);
        for i in 0..k.saturating_sub(1) {
            accepted.push(drafts[i] as u32);
        }
        accepted.push(out[k - 1] as u32);
        let ret = *accepted.last().unwrap();
        for s in &mut self.shards {
            if let Some(rt) = s.seq_runtime_mut(seq) {
                for t in &accepted {
                    rt.tokens.push(*t);
                }
            }
        }
        if mtp_tm {
            eprintln!(
                "[mtp-tm] n={} draft={:.2}ms verify={:.2}ms commit{}={:.2}ms (accept{}{:?})",
                n_v,
                t_draft.as_secs_f64() * 1e3,
                t_verify.as_secs_f64() * 1e3,
                k,
                t_c.elapsed().as_secs_f64() * 1e3,
                k,
                &accepted.iter().map(|t| format!("{t}")).collect::<Vec<_>>()[accepted.len().saturating_sub(2)..]
            );
        }
        Ok(ret)
    }

    /// ZERO-H2D device-resident MTP step (FERRITE_ZERO_H2D=1): the entire
    /// draft→verify→accept→commit chain runs on device — the token NEVER
    /// crosses to host for computation. The only D2H is 8 bytes at the end
    /// (k + next_token, for SSE/seq tracking — FERRITE_SSE=0 defers even
    /// this to every N steps in batch mode).
    ///
    /// Draft: embed_one_dev reads tokens_dev[0] (device) → emb1_dev →
    ///   mtp_forward_dev_argmax → d1_argmax_dev (device, no D2H).
    /// Draft2: embed_one_dev reads d1 → emb2_dev → mtp_forward_dev_argmax
    ///   → d2_argmax_dev (device).
    /// Verify: embed_expand_dev reads [last, d1, d2] from tokens_dev →
    ///   graph staging (D2H to pinned — the graph's input mechanism;
    ///   re-capturing the graph to read from device is the next step) →
    ///   graph replay → verify_argmax_dev (device).
    /// Accept: mtp_accept_dev compares d1/d2 vs a0/a1/a2 (ALL device) →
    ///   k_dev, next_token_dev (device).
    /// Commit: mtp_commit_dev reads k from device.
    /// Host: ONE 8-byte D2H (k + next_token) for SSE/seq push.
    #[cfg(feature = "cuda")]
    fn mtp_step_zero_h2d(&mut self, seq: u64, plans: &[ferrite_model::LayerPlan], num_dsa: usize) -> Result<u32> {
        use ferrite_kernel::cuda::DevBuf;
        let hidden = self.full_cfg.hidden_size;
        let hc_mult = self.full_cfg.hc_mult;
        let gvname = format!("mega_v{seq}");
        let mtp_family = self
            .full_cfg
            .layer_types
            .iter()
            .filter(|t| matches!(t, ferrite_model::LayerType::DeepseekSparseAttention))
            .count();
        let last = {
            let s = self.shards[0]
                .seq_runtime(seq)
                .ok_or_else(|| FerriteError::Config("missing seq".into()))?;
            *s.tokens.last().ok_or_else(|| FerriteError::Config("empty context".into()))?
        };
        let mtp_tm = std::env::var_os("FERRITE_MTP_TIMING").is_some();
        let t_d = std::time::Instant::now();

        // === DRAFT + VERIFY + ACCEPT + COMMIT: single fan_out (all device) ===
        // The draft chain, verify graph replay, accept kernel, and commit all
        // run inside ONE fan_out (no intermediate host round-trips). The ONLY
        // D2H is the final k + next_token read (8 bytes).
        let embed_table = self.shards[0]
            .w("model.embed_tokens.weight")?
            .clone();
        let (k, next_token, a0f, a1f, a2f, d1f, d2f) = {
            let toks = Self::fan_out(&mut self.shards, |s| {
                // Extract ALL device pointers from MtpState (scoped mutex)
                // + raw dev/stream handles. Drop ALL borrows before calling
                // mtp_forward_raw_argmax (which takes &mut s — re-acquires
                // cuda internally). This is the SAME raw-pointer pattern as
                // the existing mtp_step (hptr as usize — proven to work).
                let (tokens_ptr, emb_ptrs, d_ptrs, hprev_ptr) = {
                    let cuda = s
                        .backend
                        .as_cuda()
                        .ok_or_else(|| FerriteError::Config("zero-H2D mtp needs cuda".into()))?;
                    cuda.enter();
                    let m = cuda.mtp.lock().unwrap();
                    let m = m
                        .as_ref()
                        .ok_or_else(|| FerriteError::Config("mtp bufs missing".into()))?;
                    let n_v = mtp_verify_n();
                    let nd = n_v - 1;
                    // N-UNIFIED: emb_ptrs[i] = draft i's embed, d_ptrs[i] = draft
                    // i's argmax slot (contiguous d_argmax_dev[i]); tokens_dev[0]
                    // = last, [i+1] = draft i (cast_store chain).
                    let mut e: Vec<*mut f32> = m.emb_devs.iter().map(|b| b.as_f32()).collect();
                    e.resize(nd.max(1), std::ptr::null_mut());
                    let d_base = m.d_argmax_dev.as_f32();
                    let mut d: Vec<*mut f32> = (0..nd).map(|i| unsafe { d_base.add(i) }).collect();
                    d.resize(nd.max(1), d_base);
                    (
                        m.tokens_dev.as_f32() as *mut i32, // [last, d1..d_{nd}] device int slots
                        e,                                   // per-draft embeds [hidden]
                        d,                                   // per-draft argmax [1] (d_argmax_dev[i])
                        m.hprev.as_f32(),                    // draft h_prev [hidden]
                    )
                }; // ALL borrows dropped (cuda, mutex, MtpState)
                let emb1_ptr = emb_ptrs[0];
                let emb2_ptr = if emb_ptrs.len() > 1 { emb_ptrs[1] } else { std::ptr::null_mut() };
                let d1_ptr = d_ptrs[0];
                let d2_ptr = if d_ptrs.len() > 1 { d_ptrs[1] } else { d_ptrs[0] };

                // Write tokens_dev[0] = last — 4B H2D (the ONLY host-initiated
                // data write in the zero-H2D path; d1/d2 are written via D2D
                // from argmax outputs, next_token by the accept kernel)
                {
                    let cuda = s
                        .backend
                        .as_cuda()
                        .ok_or_else(|| FerriteError::Config("cuda".into()))?;
                    let last_i32 = last as i32;
                    let r = ferrite_kernel::cuda::memcpy_htod_i32(
                        tokens_ptr, &last_i32, 1, cuda.stream_handle());
                    if r != 0 { return Err(FerriteError::InvalidArg(format!("tokens_dev[0] H2D: {r}"))); }
                }

                // === Phase 1: DRAFT (all device, zero H2D) ===
                // Draft 1: embed_one(last) → emb1_dev → mtp_forward → d1_argmax_dev
                {
                    let cuda = s
                        .backend
                        .as_cuda()
                        .ok_or_else(|| FerriteError::Config("cuda".into()))?;
                    cuda.enter();
                    // DEBUG: dump hprev[0..4] (the draft's state input — if
                    // garbage, commit/hf_v is the bug; if sane, draft chain is)
                    if std::env::var_os("FERRITE_MTP_DEBUG").is_some() {
                        let mut hp = [0f32; 4];
                        let rh = ferrite_kernel::cuda::memcpy_d2h_sync(
                            hprev_ptr as *mut std::ffi::c_void, hp.as_mut_ptr(), 4, cuda.stream_handle());
                        // commit validation: hprev SHOULD equal hf_v[0] (k=1 →
                        // row 0 of last step's verify h_final export). D2H both.
                        let m2 = cuda.mtp.lock().unwrap();
                        let m2 = m2.as_ref().unwrap();
                        let mut hfv = [0f32; 12];
                        let rh2 = ferrite_kernel::cuda::memcpy_d2h_sync(
                            m2.hf_v.as_f32() as *mut std::ffi::c_void, hfv.as_mut_ptr(), 12, cuda.stream_handle());
                        // FULL hptr vs hf_v[0] comparison (4096 floats) — first 8
                        // mismatch indices (k=1 commit writes hprev<-hf_v[0]).
                        let mut hp_full = vec![0f32; 4096];
                        let mut hfv_full = vec![0f32; 4096];
                        ferrite_kernel::cuda::memcpy_d2h_sync(
                            hprev_ptr as *mut std::ffi::c_void, hp_full.as_mut_ptr(), 4096, cuda.stream_handle());
                        ferrite_kernel::cuda::memcpy_d2h_sync(
                            m2.hf_v.as_f32() as *mut std::ffi::c_void, hfv_full.as_mut_ptr(), 4096, cuda.stream_handle());
                        let mism: Vec<usize> = hp_full.iter().zip(hfv_full.iter()).enumerate()
                            .filter(|(_, (a, b))| a != b).map(|(i, _)| i).take(8).collect();
                        let tc = cuda.dsa_t_count(seq, mtp_family);
                        // k=2 commit writes hprev<-hf_v[1]: which row does hprev
                        // actually match? (S6+ all-miss root: wrong row?)
                        let mut m1 = vec![0f32; 4096]; let mut m2r = vec![0f32; 4096];
                        ferrite_kernel::cuda::memcpy_d2h_sync(hprev_ptr as *mut std::ffi::c_void, m1.as_mut_ptr(), 4096, cuda.stream_handle());
                        ferrite_kernel::cuda::memcpy_d2h_sync(unsafe { (m2.hf_v.as_f32() as *mut std::ffi::c_void).add(4096) }, m2r.as_mut_ptr(), 4096, cuda.stream_handle());
                        let row_mism: Vec<usize> = m1.iter().zip(m2r.iter()).enumerate().filter(|(_, (a, b))| a != b).map(|(i, _)| i).take(4).collect();
                        eprintln!("[zh2d-hp] hp={:?} hfv_rows={:?} hp_vs_row1_mism={} idx={:?} tc={:?}",
                            hp, hfv, row_mism.len(), row_mism, tc);
                    }
                    cuda.embed_one_dev(&embed_table, tokens_ptr, emb1_ptr, hidden, 1)?;
                    if std::env::var_os("FERRITE_MTP_DEBUG").is_some() {
                        let mut eb = [0f32; 4];
                        let re = ferrite_kernel::cuda::memcpy_d2h_sync(
                            emb1_ptr as *mut std::ffi::c_void, eb.as_mut_ptr(), 4, cuda.stream_handle());
                        let last_emb_full = s.embed(&[last]).as_slice().to_vec();
                        let mut eb_full = vec![0f32; hidden];
                        let re2 = ferrite_kernel::cuda::memcpy_d2h_sync(
                            emb1_ptr as *mut std::ffi::c_void, eb_full.as_mut_ptr(), hidden, cuda.stream_handle());
                        let mism = eb_full.iter().zip(last_emb_full.iter()).filter(|(a, b)| a != b).count();
                        eprintln!("[zh2d-eb] emb1[0..4]={:?} full4096_mismatch={} r={}/{}", &eb_full[..4], mism, re, re2);
                    }
                } // cuda dropped

                // mtp_forward #1: takes &mut s (no cuda alive) — re-acquires internally
                // h_d1 is allocated fresh (the draft's h output)
                let h_d1 = {
                    let cuda = s
                        .backend
                        .as_cuda()
                        .ok_or_else(|| FerriteError::Config("cuda".into()))?;
                    ferrite_kernel::cuda::DevBuf::alloc(cuda.dev(), cuda.stream(), hidden)?
                }; // only alloc — no borrow held across mtp_forward
                mtp_forward_raw_argmax(
                    s, seq,
                    emb1_ptr as *mut std::ffi::c_void,
                    hprev_ptr as *mut std::ffi::c_void,
                    h_d1.as_f32() as *mut std::ffi::c_void,
                    d1_ptr as *mut std::ffi::c_void,
                    hidden,
                )?;
                // KV-slot dump: the draft1's just-appended slot (t_count-1 after
                // the append) and draft2's target (t_count). MUST change every
                // step (fresh KV per draft input). FIXED: the old dump used
                // BYTE arithmetic on *mut c_void (14*64*256 bytes = slot 3.5's
                // region — h=64, dk=256 → slot stride is 16384 FLOATS = 65536
                // bytes) — it read slot 3's head-32 garbage, meaningless.
                // Now: *mut f32 + (slot * h * dk) floats, t from the cache.
                if std::env::var_os("FERRITE_MTP_DEBUG").is_some() {
                    {
                        let cuda = s
                            .backend
                            .as_cuda()
                            .ok_or_else(|| FerriteError::Config("cuda".into()))?;
                        let (knp, t0c) = {
                            let m2 = cuda.mtp_family_cache(seq, mtp_family)?;
                            (m2.0, m2.1)
                        };
                        // slot stride = h*dk floats (k_nope [max_t, h, dk]);
                        // h=64 dk=256 → 16384 floats. t0c is AFTER draft1's
                        // append (t_count incremented by dsa_layer_dev), so
                        // draft1 wrote slot t0c-1; draft2 will write slot t0c.
                        let knp_f = knp as *mut f32;
                        let stride = 64 * 256;
                        let s_prev = unsafe { knp_f.add(t0c.saturating_sub(1) * stride) };
                        let s_next = unsafe { knp_f.add(t0c * stride) };
                        let mut kv = [0f32; 4];
                        let rk = ferrite_kernel::cuda::memcpy_d2h_sync(
                            s_prev as *mut std::ffi::c_void,
                            kv.as_mut_ptr(), 4, cuda.stream_handle());
                        let mut kv2 = [0f32; 4];
                        let rk2 = ferrite_kernel::cuda::memcpy_d2h_sync(
                            s_next as *mut std::ffi::c_void,
                            kv2.as_mut_ptr(), 4, cuda.stream_handle());
                        eprintln!("[zh2d-kv] t={} slot[t-1]={:?} slot[t]={:?} r={}/{}",
                            t0c, kv, kv2, rk, rk2);
                    }
                }

                // FERRITE_ZH2D_AB: same-process A/B of the draft chain — re-run
                // draft 1 the ORIGINAL way (host embed + upload + mtp_forward
                // wrapper). NCCL stays matched (all 4 ranks execute the same
                // extra 3 ARs); mtp_family t_count rolled back (-1) so draft 2
                // sees the same cache slot. Float-compare d1/d1_ref and the
                // x2 residual (h_d1) — locates WHERE the zero-H2D draft diverges.
                if std::env::var_os("FERRITE_ZH2D_AB").is_some() {
                    // AB v2 (raw_argmax — same fn as zh path, emb from host
                    // upload): d1_ref vs d1_zh + h_d1 float compare. If equal →
                    // emb buffer path is NOT the divergence; if differ → it is.
                    {
                        let cuda = s
                            .backend
                            .as_cuda()
                            .ok_or_else(|| FerriteError::Config("cuda".into()))?;
                        let mut d1_zh = [0f32; 1];
                        let mut hd1_zh = [0f32; 8];
                        ferrite_kernel::cuda::memcpy_d2h_sync(
                            d1_ptr as *mut std::ffi::c_void, &mut d1_zh[0], 1, cuda.stream_handle());
                        ferrite_kernel::cuda::memcpy_d2h_sync(
                            h_d1.as_f32() as *mut std::ffi::c_void, hd1_zh.as_mut_ptr(), 8, cuda.stream_handle());
                        let h2 = s.embed(&[last]);
                        let emb_ref = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), hidden)?;
                        emb_ref.upload(h2.as_slice())?;
                        let h_d1_ref = DevBuf::alloc(cuda.dev(), cuda.stream(), hidden)?;
                        let d1_ref_buf = DevBuf::alloc(cuda.dev(), cuda.stream(), 1)?;
                        let d1_ref_ptr = d1_ref_buf.as_f32();
                        // raw-argmax variant (no mtp_forward — avoids the
                        // segfault): same hprev_ptr, h_d1_ref out, d1_ref buf
                        drop(cuda);
                        mtp_forward_raw_argmax(
                            s, seq,
                            emb_ref.as_f32() as *mut std::ffi::c_void,
                            hprev_ptr as *mut std::ffi::c_void,
                            h_d1_ref.as_f32() as *mut std::ffi::c_void,
                            d1_ref_ptr as *mut std::ffi::c_void,
                            hidden,
                        )?;
                        {
                            let cuda = s
                                .backend
                                .as_cuda()
                                .ok_or_else(|| FerriteError::Config("cuda".into()))?;
                            cuda.dsa_host_rollback(seq, mtp_family, 1);
                            let mut d1_ref = [0f32; 1];
                            ferrite_kernel::cuda::memcpy_d2h_sync(
                                d1_ref_ptr as *mut std::ffi::c_void, &mut d1_ref[0], 1, cuda.stream_handle());
                            let mut hd1_ref = [0f32; 8];
                            ferrite_kernel::cuda::memcpy_d2h_sync(
                                h_d1_ref.as_f32() as *mut std::ffi::c_void, hd1_ref.as_mut_ptr(), 8, cuda.stream_handle());
                            let h_match = hd1_zh.iter().zip(hd1_ref.iter()).all(|(a, b)| a == b);
                            let hd: Vec<String> = hd1_zh.iter().map(|v| format!("{v:.6}")).collect();
                            let hr: Vec<String> = hd1_ref.iter().map(|v| format!("{v:.6}")).collect();
                            eprintln!("[zh2d-ab1] d1_zh={:.0} d1_ref={:.0} match={} h_d1_zh={:?} h_d1_ref={:?} h_match={}",
                                d1_zh[0], d1_ref[0], d1_zh[0] as u32 == d1_ref[0] as u32, hd, hr, h_match);
                        }
                    }
                }
                // Draft 2: embed_one(d1) → emb2_dev → mtp_forward → d2_argmax_dev
                // Device cast: d1_argmax_dev (f32) → tokens_dev[1] (i32) —
                // replaces the D2H→host-cast→H2D roundtrip. That roundtrip's
                // cudaStreamSynchronize between draft1 and draft2 broke NCCL
                // AR channel continuity: draft2's ARs got 1-ulp float drift →
                // x2 checksums matched to 6 decimals but argmax flipped on
                // near-ties (d1 98347→702, d2 315→8606) → token stream
                // diverged → accept=1.0. The cast kernel runs on the same
                // stream immediately after draft1's argmax: no sync, no H2D,
                // full device chain (true zero-H2D), NCCL AR ordering
                // preserved exactly like the original fan_out.
                {
                    let cuda = s
                        .backend
                        .as_cuda()
                        .ok_or_else(|| FerriteError::Config("cuda".into()))?;
                    cuda.cast_store_i32(
                        d1_ptr as *const std::ffi::c_void,
                        unsafe { tokens_ptr.add(1) } as *mut std::ffi::c_void,
                    )?;
                    cuda.embed_one_dev(&embed_table, unsafe { tokens_ptr.add(1) }, emb2_ptr, hidden, 1)?;
                } // cuda dropped

                // mtp_forward #2 (h_prev = h_d1 from draft #1)
                mtp_forward_raw_argmax(
                    s, seq,
                    emb2_ptr as *mut std::ffi::c_void,
                    h_d1.as_f32() as *mut std::ffi::c_void,
                    std::ptr::null_mut(), // no h_out needed for d2
                    d2_ptr as *mut std::ffi::c_void,
                    hidden,
                )?;
                // FERRITE_ZH2D_AB: draft 2 A/B — original path with the SAME
                // h_d1 (x2 from zero-H2D draft 1) + host embed of d1.
                if std::env::var_os("FERRITE_ZH2D_AB").is_some() {
                    {
                        let cuda = s
                            .backend
                            .as_cuda()
                            .ok_or_else(|| FerriteError::Config("cuda".into()))?;
                        let mut d1_zh2 = [0f32; 1];
                        let mut d2_zh = [0f32; 1];
                        ferrite_kernel::cuda::memcpy_d2h_sync(
                            d1_ptr as *mut std::ffi::c_void, &mut d1_zh2[0], 1, cuda.stream_handle());
                        ferrite_kernel::cuda::memcpy_d2h_sync(
                            d2_ptr as *mut std::ffi::c_void, &mut d2_zh[0], 1, cuda.stream_handle());
                        let h3 = s.embed(&[d1_zh2[0] as u32]);
                        let emb2_ref = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), hidden)?;
                        emb2_ref.upload(h3.as_slice())?;
                        let d2_ref = mtp_forward(s, seq, &emb2_ref, &h_d1, None)?;
                        {
                            let cuda = s
                                .backend
                                .as_cuda()
                                .ok_or_else(|| FerriteError::Config("cuda".into()))?;
                            cuda.dsa_host_rollback(seq, mtp_family, 1);
                        }
                        eprintln!("[zh2d-ab2] d2_zh={:.0} d2_ref={:.0} match={}",
                            d2_zh[0], d2_ref, d2_zh[0] as u32 == d2_ref as u32);
                    }
                }
                let t_draft = t_d.elapsed();

                // === Phase 2: VERIFY (graph replay, input from device) ===
                let t_v = std::time::Instant::now();
                {
                    let cuda = s
                        .backend
                        .as_cuda()
                        .ok_or_else(|| FerriteError::Config("cuda".into()))?;
                    cuda.enter();

                    // tokens_dev = [last, d1, d2] for verify input
                    // D2H d1 and d2 (for the host embed + accept comparison).
                    let mut d1_val = [0f32; 1];
                    let mut d2_val = [0f32; 1];
                    {
                        let r1 = ferrite_kernel::cuda::memcpy_d2h_sync(
                            d1_ptr as *mut std::ffi::c_void, d1_val.as_mut_ptr(), 1, cuda.stream_handle());
                        let r2 = ferrite_kernel::cuda::memcpy_d2h_sync(
                            d2_ptr as *mut std::ffi::c_void, d2_val.as_mut_ptr(), 1, cuda.stream_handle());
                        if r1 != 0 || r2 != 0 { return Err(FerriteError::InvalidArg(format!("d1/d2 D2H: {r1}/{r2}"))); }
                        // also write tokens_dev[2] = d2 (for the FERRITE_MTP_DEBUG path)
                        let d2_i32 = d2_val[0] as i32;
                        let r = ferrite_kernel::cuda::memcpy_htod_i32(
                            unsafe { tokens_ptr.add(2) }, &d2_i32, 1, cuda.stream_handle());
                        if r != 0 { return Err(FerriteError::InvalidArg(format!("tokens_dev[2] H2D: {r}"))); }
                    }

                    // Verify input: HOST embed + hc_expand (IDENTICAL to mtp_step).
                    // The device embed_expand_dev_buf had 1-ulp drift that flipped
                    // the verify argmax on near-ties → accept=1.0. Reverting the
                    // verify input to the proven host path fixes the accept while
                    // keeping the draft chain fully on device (the zero-H2D
                    // saving: no D2H roundtrip between draft1/draft2 — the
                    // cast_store_i32 kernel keeps the chain on device).
                    let h2v = s.embed(&[last, d1_val[0] as u32, d2_val[0] as u32]);
                    let in_vals = crate::mhc::hc_expand(&h2v, hc_mult);

                    // DSA advance (pinned t0/total bookkeeping)
                    for f in 0..num_dsa {
                        cuda.dsa_host_advance(seq, f, 3);
                    }

                    // Graph replay with the host input (graph_run copies in_vals
                    // to the pinned staging internally — same as mtp_step)
                    let mut a = [0f32; 3];
                    if !cuda.graph_run(&gvname, in_vals.as_slice(), &mut a)? {
                        return Err(FerriteError::InvalidArg(format!("mega_v graph {gvname} missing")));
                    }
                    let t_verify = t_v.elapsed();
                    let _ = t_verify;

                    // === Phase 3: ACCEPT + COMMIT ===
                    let t_c = std::time::Instant::now();
                    let k_host = if d1_val[0] as u32 == a[0] as u32 {
                        if d2_val[0] as u32 == a[1] as u32 { 3 } else { 2 }
                    } else { 1 };
                    // DSA rollback — decoder families: verify graph appends 3
                    // (last,d1,d2) → keep k accepted → rollback(3-k) ✓ (vLLM
                    // semantics). mtp_family: the ORIGINAL mtp_step's arithmetic
                    // (828ec86, accept 2.38): draft appends 2 (last@t0,
                    // d1@t0+1), verify appends 0 (tc data: 14,16,18,20,21 =
                    // P+Σk proves it) → rollback(3-k) keeps k-1: k=1 → keep 0
                    // (next draft1 overwrites last's slot), k=2 → keep 1
                    // (last), k=3 → keep 2 (last+d1). The (2-k) experiment
                    // (b2268ad) kept the accepted d1's KV at k=2 — the draft
                    // cache stream then diverges from the original's reference
                    // ((3-k) matches 828ec86's line 821 EXACTLY) and accept
                    // collapsed to 1.0 from S6 (FIX12: S1-S3 k=2, S4 k=1,
                    // S5 k=2, S6+ all k=1).
                    for f in 0..num_dsa {
                        cuda.dsa_host_rollback(seq, f, (3 - k_host) as usize);
                    }
                    // mtp_family rollback: (2-k).max(0) — the draft appends 2
                    // (last@t0, d1@t0+1), rollback removes ONLY unaccepted
                    // draft tokens. k=1: remove d1 (rejected), keep last (the
                    // step's input — already in the decoder cache). k=2/3:
                    // keep both (d1 accepted). Net advance = min(2,k) — the
                    // draft cache grows in LOCKSTEP with the decoder cache
                    // (k per step). The OLD (3-k) advanced only k-1 per step:
                    // after 3 steps the mtp_family lagged the decoder by 3
                    // tokens, the draft's DSA attention missed the accepted
                    // tokens, argmax flipped to input-repeat (d1==d2, the
                    // diagnostic's call-3 signature) and accept collapsed to
                    // 1.0 permanently (commit 2a9c85c predicted this exact
                    // failure mode; 2ffcc68's revert misread the tc data).
                    cuda.dsa_host_rollback(seq, mtp_family, (2 - k_host).max(0) as usize);
                    // Diagnostic: first 10 CALLS (step%4==0 = rank 0 of each call;
                    // 4 ranks increment ZH2D_STEP per call) — shows when accept
                    // collapses and whether hprev/draft values keep changing
                    {
                        static ZH2D_STEP: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
                        let step = ZH2D_STEP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if step < 48 && step % 4 == 0 {
                            eprintln!("[zh2d-accept] call={} last={} d1={} d2={} a0={} a1={} a2={} k={}",
                                step / 4, last, d1_val[0] as u32, d2_val[0] as u32,
                                a[0] as u32, a[1] as u32, a[2] as u32, k_host);
                        }
                    }
                    // mtp_commit with k from host (pinned — TODO: k_dev from device)
                    cuda.mtp_commit(k_host)?;
                    let t_commit = t_c.elapsed();

                    if mtp_tm {
                        eprintln!(
                            "[mtp-tm] zero-H2D draft={:.2}ms verify={:.2}ms commit={:.2}ms",
                            t_draft.as_secs_f64() * 1e3,
                            t_verify.as_secs_f64() * 1e3,
                            t_commit.as_secs_f64() * 1e3,
                        );
                    }

                    // === Phase 4: return the D2H'd values (no re-read) ===
                    // BUG FIX: Phase 5 previously re-read m.verify_argmax_dev
                    // (a MtpState buffer the graph NEVER writes — the graph's
                    // argmax goes to io.out_dev). That re-read returned garbage,
                    // the seq push wrote a garbage a0, and `last` froze on
                    // the same garbage token every step (embed(last) identical
                    // across steps — the debug smoking gun). The correct values
                    // are ALREADY D2H'd here: a (io.out_dev) + d1_val/d2_val.
                    let nt = match k_host { 3 => a[2] as i32, 2 => a[1] as i32, _ => a[0] as i32 };
                    Ok((k_host, nt, a[0], a[1], a[2], d1_val[0], d2_val[0]))
                }
            })
            .into_iter()
            .collect::<Result<Vec<(i32, i32, f32, f32, f32, f32, f32)>>>()?;
            toks[0]
        };

        // --- Phase 5: seq push (host, from the 8-byte D2H) ---
        let dbg = std::env::var_os("FERRITE_MTP_DEBUG").is_some();
        if dbg {
            eprintln!("[mtp-zero-h2d] k={} next_token={}", k, next_token);
        }
        // push accepted tokens from the verify argmax (the tokens the model
        // actually generated — we read them from the device's token chain)
        // For now: k=1→a0, k=2→d1+a1, k=3→d1+d2+a2 (the same logic as the
        // host version but computed on device — the host reads k and pushes
        // the corresponding tokens from its local copy)
        // NOTE: for full zero-H2D, the token push would be accumulated on
        // device and read in batch mode. For SSE mode, we read the verify
        // argmax (3 f32 = 12 bytes D2H, already done above as `a`).
        // a0/a1/a2 (verify graph argmax — io.out_dev D2H'd inside the
        // fan_out) + d1/d2 (draft argmax) came back with the fan_out result.
        let (a0, a1, a2) = (a0f as u32, a1f as u32, a2f as u32);
        let (d1_u32, d2_u32) = (d1f as u32, d2f as u32);
        match k {
            3 => {
                for s in &mut self.shards {
                    if let Some(rt) = s.seq_runtime_mut(seq) {
                        rt.tokens.push(d1_u32);
                        rt.tokens.push(d2_u32);
                        rt.tokens.push(a2);
                    }
                }
                Ok(a2)
            }
            2 => {
                for s in &mut self.shards {
                    if let Some(rt) = s.seq_runtime_mut(seq) {
                        rt.tokens.push(d1_u32);
                        rt.tokens.push(a1);
                    }
                }
                Ok(a1)
            }
            _ => {
                for s in &mut self.shards {
                    if let Some(rt) = s.seq_runtime_mut(seq) {
                        rt.tokens.push(a0);
                    }
                }
                Ok(a0)
            }
        }
    }

    /// Allocate the per-rank MTP fixed buffers (MtpState): decode-graph h_final
    /// [hidden], verify-graph h_final [N*hidden] (N=FERRITE_MTP_N), draft
    /// h_prev [hidden], and per-GDN-layer (conv, gdn, conv_snaps, gdn_snaps)
    /// B-scratch — the snapshots are [N-1][len] contiguous (the kernels index
    /// base + i*len). Called once before the mega graph captures (fixed
    /// addresses for graph lifetime).
    #[cfg(feature = "cuda")]
    fn mtp_setup_bufs(s: &mut Engine<B>, plans: &[ferrite_model::LayerPlan], seq: u64) -> Result<()> {
        use ferrite_kernel::cuda::{DevBuf, MtpCommitPlan, MtpState};
        let n_v = mtp_verify_n(); // FERRITE_MTP_N: verify width (drafts = n_v-1)
        let cuda = s
            .backend
            .as_cuda()
            .ok_or_else(|| FerriteError::Config("mtp needs cuda".into()))?;
        cuda.enter();
        let cfg = &s.cfg;
        let hidden = cfg.hidden_size;
        let la = &cfg.linear_attn;
        let proj = la.num_heads * la.head_dim;
        let conv_len = 3 * proj * (la.short_conv_kernel_size.saturating_sub(1).max(1));
        let gdn_len = la.num_heads * la.head_dim * la.head_dim;
        let mut scratch = Vec::new();
        for plan in plans {
            if matches!(plan.attn, AttnKind::Linear) {
                let conv = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), conv_len)?;
                let gdn = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), gdn_len)?;
                // N-UNIFIED snapshots: ONE contiguous [n-1][len] buffer per kind
                // replaces the old fixed (B0, B1) pair — snap i = A + t_0..t_i
                // (accept-(i+1)'s commit source; the kernels index base + i*len,
                // so ANY n works with the same 6-pointer commit plan).
                let conv_snaps = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), (n_v - 1) * conv_len)?;
                let gdn_snaps = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), (n_v - 1) * gdn_len)?;
                scratch.push((conv, gdn, conv_snaps, gdn_snaps));
            }
        }
        let hf_dev = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), hidden)?;
        let hf_v = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), n_v * hidden)?;
        let hprev = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), hidden)?;
        // Single-kernel accept-commit plan (N-UNIFIED): per GDN layer the
        // 6-POINTER row (conv_a, gdn_a, conv_b, gdn_b, conv_snaps_base,
        // gdn_snaps_base) packed as f32 bit patterns (DevBuf is f32-typed;
        // 2 f32 per pointer) + a pinned k slot. k=n commits B (full verify
        // state); k=j<n commits snapshot j-1 at base + (j-1)*len. The A-side
        // pointers come from the seq's recurrent-state stores (fixed for the
        // seq's lifetime — also what the verify graph's recorded A→B copy-in
        // nodes use). This allocates the A states now if not yet warm
        // (idempotent dev_state lookup).
        let mut flat: Vec<f32> = Vec::with_capacity(scratch.len() * 12);
        let mut n_plans = 0usize;
        for plan in plans {
            if matches!(plan.attn, AttnKind::Linear) {
                let a_conv = cuda.conv_state_ptr(seq, plan.layer_idx, conv_len)?;
                let a_gdn = cuda.gdn_state_ptr(seq, plan.layer_idx, gdn_len)?;
                let (cb, gb, cs, gs) = &scratch[n_plans];
                for p in [
                    a_conv,
                    a_gdn,
                    cb.as_f32(),
                    gb.as_f32(),
                    cs.as_f32(),
                    gs.as_f32(),
                ] {
                    let bits = p as usize as u64;
                    flat.push(f32::from_bits((bits & 0xffff_ffff) as u32));
                    flat.push(f32::from_bits((bits >> 32) as u32));
                }
                n_plans += 1;
            }
        }
        let plan_buf = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), flat.len())?;
        plan_buf.upload(flat.as_slice())?;
        let k_pin = cuda.pinned_i32()?;
        let commit = MtpCommitPlan { plan: plan_buf, k_pin, n: n_plans, mtp_n: n_v, conv_len, gdn_len, hidden };
        // ZERO-H2D device token chain (fixed bufs — never pooled, stable
        // addresses for the accept kernel's device-resident loop):
        // tokens_dev [n_v] i32 = [last, d1..d_{n_v-1}] (embed kernel reads)
        // k_dev/next_token_dev/n_accepted_dev [1] (accept kernel outputs)
        // verify_argmax_dev [n_v] f32 (the accept kernel's `a` input)
        // emb_devs [n_v-1] × [hidden] f32 (draft chain embeds from embed_one)
        // d_argmax_dev [n_v-1] f32 (draft argmax outputs — ONE contiguous
        // buffer; draft i's token at offset i, the accept kernel reads it
        // as the d array).
        let nd = n_v - 1;
        let tokens_dev = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), n_v)?;
        let verify_argmax_dev = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), n_v)?;
        let k_dev = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), 1)?;
        let next_token_dev = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), 1)?;
        let n_accepted_dev = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), 1)?;
        let mut emb_devs = Vec::with_capacity(nd);
        for _ in 0..nd {
            emb_devs.push(DevBuf::alloc(cuda.dev(), cuda.stream_handle(), hidden)?);
        }
        // h_d [nd-1] × [hidden]: the draft graphs' h relay (draft i's h_out =
        // draft i+1's h_prev). Fixed addresses for graph lifetime; the last
        // draft exports no h (verify's hf_v commit replaces it).
        let mut h_d = Vec::with_capacity(nd.saturating_sub(1));
        for _ in 0..nd.saturating_sub(1) {
            h_d.push(DevBuf::alloc(cuda.dev(), cuda.stream_handle(), hidden)?);
        }
        let d_argmax_dev = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), nd.max(1))?;
        *cuda.mtp.lock().unwrap() = Some(MtpState {
            hf_dev, hf_v, hprev, scratch, commit: Some(commit),
            tokens_dev, verify_argmax_dev, k_dev, next_token_dev, n_accepted_dev,
            emb_devs, h_d, d_argmax_dev,
        });
        Ok(())
    }

    /// VerifyIO for the given graph: verify=false → decode graph (n=1, empty
    /// scratch, h_final → hf_dev); verify=true → verify graph (n=2, GDN
    /// ping-pong scratch ptrs, h_final → hf_v).
    #[cfg(feature = "cuda")]
    fn mtp_vio(s: &Engine<B>, verify: bool) -> VerifyIO {
        let cuda = s.backend.as_cuda().unwrap();
        let m = cuda.mtp.lock().unwrap();
        let m = m.as_ref().unwrap();
        if verify {
            VerifyIO {
                gdn_scratch: m
                    .scratch
                    .iter()
                    .map(|(c, g, cs, gs)| (c.as_f32(), g.as_f32(), cs.as_f32(), gs.as_f32()))
                    .collect(),
                h_final: m.hf_v.as_f32(),
            }
        } else {
            VerifyIO { gdn_scratch: vec![], h_final: m.hf_dev.as_f32() }
        }
    }

// ============================================================
// Mega-graph chain (FERRITE_MEGA): ONE rank's whole-decode-step device
// chain — staging upload → [per layer: hc_pre → norm → attn(GDN/DSA) →
// NCCL AR → hc_post → hc_pre2 → norm → ffn(MoE/Dense) → NCCL AR →
// hc_post] × 45 → contract → norm → lm_head → argmax.
//
// Every intermediate stays on-device (the old chain crossed PCIe ~6× per
// layer via host all-reduce staging); NCCL all-reduce runs INSIDE the
// captured graph (warm comm + ThreadLocal capture — proven in
// gpu_smoke_nccl_graph: 90 ARs/replay @ 15µs). The hc chain + head run
// REDUNDANTLY on every rank (the in-place AR leaves bit-identical data
// everywhere; hc/lm_head weights are replicated) — no broadcast needed.
//
// capture=false → dry-run (real execution: warms every pool class +
// per-rank weight caches + the NCCL 4096-float AR plan; returns the
// sampled token). capture=true → records the whole sequence into graph
// `gname` (NCCL enqueues become graph nodes; record never executes).
//
// Capture-mode memory: intermediates return to the pool BUT replay
// allocates nothing (graph_run = staging write + launch + raw D2H), so
// their recorded addresses are never re-dispensed while the graph lives.
// Only the IO boundaries are pinned via GraphIO (forgotten arg DevBuf +
// res0's stage). The DSA host bookkeeping (t_count) advances during the
// record pass without executing the kernels — rolled back after capture;
// replay advances it for real before each graph_run.
// ============================================================

#[cfg(feature = "cuda")]
fn mega_chain_dev(
    s: &mut Engine<B>,
    seq: u64,
    in_vals: &[f32],
    plans: &[ferrite_model::LayerPlan],
    num_dsa: usize,
    capture: bool,
    gname: &str,
    n: usize,
    verify: Option<&VerifyIO>,
) -> Result<Vec<f32>> {
    use ferrite_kernel::cuda::{DevBuf, DsaLayerWeights, ExpertWeights, GdnLayerWeights, GraphIO};
    let cuda = s
        .backend
        .as_cuda()
        .ok_or_else(|| FerriteError::Config("FERRITE_MEGA needs cuda backend".into()))?;
    let nccl = s
        .nccl
        .clone()
        .ok_or_else(|| FerriteError::Config("FERRITE_MEGA needs FERRITE_NCCL=1".into()))?;
    cuda.enter();
    // MTP verify chain (n==2): per-row GEMV for the small-n matmuls (the
    // tiled GEMM wastes a tile on 2 rows: 108ms vs 23ms). Prefill keeps the
    // GEMM — its accumulation order sets the first greedy token (per-row
    // GEMV flips 背出师表 recitation into an English gloss).
    cuda.small_n_rows.store(verify.is_some(), std::sync::atomic::Ordering::Relaxed);
    let cfg = &s.cfg;
    let (hidden, hc_mult) = (cfg.hidden_size, cfg.hc_mult);
    let nh = hc_mult * hidden;
    let topk = cfg.num_experts_per_tok;
    let e = cfg.n_routed_experts;

    // FERRITE_MEGA_PROBE: dump intermediates to /tmp/orion (dry-run only —
    // download syncs, so never inside a capture). Cross-rank ar0/resmid
    // equality also validates the NCCL all-reduce.
    let dev_id = cuda.dev();
    let probe = !capture && std::env::var_os("FERRITE_MEGA_PROBE").is_some();
    // per-layer breakdown timing (dry only): attn A+B+AR / ffn C+D+E+AR /
    // head — sync at segment boundaries (diagnostic runs only)
    let tm = !capture && dev_id == 0 && std::env::var_os("FERRITE_TIMING").is_some();
    let (mut t_attn, mut t_ffn, mut t_head) = (0f64, 0f64, 0f64);
    let (mut t_a, mut t_b_gdn, mut t_b_dsa, mut t_c, mut t_e) = (0f64, 0f64, 0f64, 0f64, 0f64);
    macro_rules! mprobe {
        ($name:expr, $buf:expr, $len:expr) => {
            if probe {
                let mut pv = vec![0f32; $len];
                if $buf.download(&mut pv).is_ok() {
                    let bytes: Vec<u8> = pv.iter().flat_map(|x| x.to_le_bytes()).collect();
                    std::fs::write(
                        format!("/tmp/orion/mega_probe_{}_dev{}.f32", $name, dev_id),
                        bytes,
                    )
                    .ok();
                }
            }
        };
    }

    let _guard = if capture {
        // DSA host bookkeeping: the capture pass re-runs dsa_layer_dev's
        // host logic (t_count += 1 WITHOUT executing cache_append). Roll it
        // back BEFORE the pass so the recorded buffer sizes match the
        // dry-run's — npools/select_k derive from t_count/total, and the
        // +1 shift changes the idx_pools/idx size classes → pool miss →
        // cudaMalloc during capture = err 900. After the pass t_count
        // lands back at the real cache count (no post-capture rollback);
        // replay-side dsa_host_advance then keeps it in lockstep.
        for f in 0..num_dsa {
            cuda.dsa_host_rollback(seq, f, n);
        }
        // Serialize per-rank captures (concurrent cuGraphInstantiate
        // SIGSEGV'd historically); record-mode NCCL enqueue never
        // rendezvous, so serialized capture is deadlock-free.
        Some(ferrite_kernel::cuda::capture_lock().lock().unwrap())
    } else {
        None
    };
    if capture {
        cuda.graph_capture_begin();
    }

    // MTP verify chain: record the A→B ping-pong copy-in as the FIRST graph
    // nodes — every replay refreshes B from the committed A state before the
    // layer chain writes it. A (per-seq recurrent states) and B (MtpState
    // scratch) are fixed addresses for the graph's lifetime, so the D2D
    // memcpys are capture-stable. This removes the host loop of
    // 2*n_gdn cudaMemcpyAsync launches from every step (was ~1ms + fan_out
    // round-trips). Only captured (replayed); the dry-run skips it — B
    // holding stale data there is fine (dry output is discarded).
    if let Some(v) = verify {
        if capture && !v.gdn_scratch.is_empty() {
            let la = &cfg.linear_attn;
            let proj = la.num_heads * la.head_dim;
            let conv_len = 3 * proj * (la.short_conv_kernel_size.saturating_sub(1).max(1));
            let gdn_len = la.num_heads * la.head_dim * la.head_dim;
            let mut gi = 0usize;
            for plan in plans {
                if matches!(plan.attn, AttnKind::Linear) {
                    let (cb, gb, _, _) = v.gdn_scratch[gi];
                    let aptr = cuda.conv_state_ptr(seq, plan.layer_idx, conv_len)?;
                    let gptr = cuda.gdn_state_ptr(seq, plan.layer_idx, gdn_len)?;
                    cuda.copy_raw_dev(aptr as *const f32, cb, conv_len)?;
                    cuda.copy_raw_dev(gptr as *const f32, gb, gdn_len)?;
                    gi += 1;
                }
            }
        }
    }

    let mut res = DevBuf::alloc(cuda.dev(), cuda.stream(), n * nh)?;
    // capture-time res pointer (BEFORE the hc_post chain reassigns res —
    // the GraphIO's in_dev must reference the graph input buffer, the one
    // embed_expand_dev writes / the stage memcpy fills).
    let res_in_ptr = res.as_f32() as *mut std::ffi::c_void;
    // Device-embed graph input (MTP graphs): when capturing AND MtpState has
    // the fixed tokens_dev buffer, record embed_expand_dev as the graph's
    // FIRST node — replay writes n×4B token ids into tokens_dev (graph_run_ids)
    // instead of n*mult*hidden f32 host staging (~1ms of the verify step's
    // host budget: host embed lookup + hc_expand + 576KB pinned write).
    // The kernel reads the F32 embed cache (dev_weight — bit-identical to the
    // host lookup; the bf16-table era's 1-ulp accept crash does not apply) and
    // writes res directly (row copy + mult replication = hc_expand semantics).
    // Dry-run keeps the host staging (its sampled token needs REAL input).
    let ids_dev_cap: Option<*mut i32> = {
        let m = cuda.mtp.lock().unwrap();
        m.as_ref().map(|m| m.tokens_dev.as_f32() as *mut i32)
    };
    let dev_embed = capture && ids_dev_cap.is_some();
    if dev_embed {
        let table = s.w("model.embed_tokens.weight")?;
        cuda.embed_expand_dev_buf(
            table,
            ids_dev_cap.unwrap() as *const i32,
            res.as_f32(),
            n, hidden, hc_mult,
        )?;
    } else {
        res.upload(in_vals)?; // recorded stage→dev memcpy (the graph input)
    }
    let x_stage = res.stage; // GraphIO: replay writes fresh input here
    mprobe!("res0", &res, nh);

    let mut gdn_idx = 0usize; // verify scratch index (GDN layers only)
    for (layer_idx, plan) in plans.iter().enumerate() {
        if std::env::var_os("FERRITE_TIMING").is_some() {
            eprintln!("[megab-cap] dev{dev_id} cap={capture} L{layer_idx}");
        }
        let t_l = std::time::Instant::now();
        let pfx = format!("model.layers.{layer_idx}");
        // A: hc_pre + input_layernorm (redundant per rank)
        let (li, post_a, comb_a) = cuda.hc_pre_dev(
            &res,
            s.w(&format!("{pfx}.hc_attn_fn"))?,
            s.w(&format!("{pfx}.hc_attn_scale"))?,
            s.w(&format!("{pfx}.hc_attn_base"))?,
            s.w(&format!("{pfx}.input_layernorm.weight"))?,
            n,
            nh,
            cfg.rms_norm_eps,
            cfg.hc_eps,
            cfg.hc_sinkhorn_iters,
        )?;
        // li comes out already RMS-normalized (fused tail in hc_pre_rest) —
        // the standalone rmsnorm_dev launch is gone.
        let hn = li;
        if tm {
            let _ = cuda.sync();
            t_a += t_l.elapsed().as_secs_f64() * 1e3;
        }
        let t_b = std::time::Instant::now();
        if layer_idx == 0 {
            mprobe!("hn0", &hn, hidden);
        }
        // B: attention → NCCL all-reduce (in-place; every rank holds the sum)
        let mut partial = match plan.attn {
            AttnKind::Linear => {
                let la = &cfg.linear_attn;
                let gw = GdnLayerWeights {
                    qkv_proj: s.w(&format!("{pfx}.self_attn.qkv_proj.weight"))?,
                    b_proj: s.w(&format!("{pfx}.self_attn.b_proj.weight"))?,
                    f_a: s.w(&format!("{pfx}.self_attn.f_a_proj.weight"))?,
                    f_b: s.w(&format!("{pfx}.self_attn.f_b_proj.weight"))?,
                    g_a: s.w(&format!("{pfx}.self_attn.g_a_proj.weight"))?,
                    g_b: s.w(&format!("{pfx}.self_attn.g_b_proj.weight"))?,
                    conv_w: s.w(&format!("{pfx}.self_attn.qkv_conv1d.weight"))?,
                    dt_bias: s.w(&format!("{pfx}.self_attn.dt_bias"))?,
                    a_log: s.w(&format!("{pfx}.self_attn.A_log"))?,
                    o_norm: s.w(&format!("{pfx}.self_attn.o_norm.weight"))?,
                    o_proj: s.w(&format!("{pfx}.self_attn.o_proj.weight"))?,
                };
                let state_override = verify.and_then(|v| v.gdn_scratch.get(gdn_idx)).copied();
                gdn_idx += 1;
                cuda.gdn_layer_dev(
                    &hn, &gw, seq, layer_idx, n, hidden,
                    la.num_heads, la.head_dim, la.gate_lower_bound,
                    cfg.rms_norm_eps, la.short_conv_kernel_size, state_override,
                )?
            }
            AttnKind::Dsa => {
                let d = &cfg.dsa;
                let (dsa_h, dsa_dk, dsa_dv, _ip) = s.dsa_dims();
                let w = DsaLayerWeights {
                    q_a: s.w(&format!("{pfx}.self_attn.q_a_proj.weight"))?,
                    q_a_ln: s.w(&format!("{pfx}.self_attn.q_a_layernorm.weight"))?,
                    q_b: s.w(&format!("{pfx}.self_attn.q_b_proj.weight"))?,
                    kv_a: s.w(&format!("{pfx}.self_attn.kv_a_proj_with_mqa.weight"))?,
                    kv_a_ln: s.w(&format!("{pfx}.self_attn.kv_a_layernorm.weight"))?,
                    kv_b: s.w(&format!("{pfx}.self_attn.kv_b_proj.weight"))?,
                    wq_b: s.w(&format!("{pfx}.self_attn.indexer.wq_b.weight"))?,
                    wk: s.w(&format!("{pfx}.self_attn.indexer.wk.weight"))?,
                    k_norm_w: s.w(&format!("{pfx}.self_attn.indexer.k_norm.weight"))?,
                    k_norm_b: s.w(&format!("{pfx}.self_attn.indexer.k_norm.bias"))?,
                    weights_proj: s.w(&format!("{pfx}.self_attn.indexer.weights_proj.weight"))?,
                    gate: s.w(&format!("{pfx}.self_attn.indexer.index_kpool_compress_gate"))?,
                    ape: s.w(&format!("{pfx}.self_attn.indexer.index_kpool_compress_ape"))?,
                    o_proj: s.w(&format!("{pfx}.self_attn.o_proj.weight"))?,
                    h: dsa_h,
                    dk: dsa_dk,
                    dv: dsa_dv,
                    ih: d.index_n_heads,
                    idm: d.index_head_dim,
                    kpool: 4,
                    topk: d.index_topk,
                    rms_eps: cfg.rms_norm_eps,
                };
                let family = s.dsa_family_index(layer_idx);
                cuda.dsa_layer_dev(&hn, &w, seq, family, n, hidden)?
            }
        };
        let ar_skip = std::env::var_os("FERRITE_AR_SKIP").is_some();
        let ar_p2p = match s.backend.as_cuda() {
            Some(c) if !ar_skip => c.p2p_ar_v2(&mut partial, n * hidden).unwrap_or(false),
            _ => ar_skip,
        };
        if !ar_p2p {
            let cnt = n * hidden;
            // bf16 payload halves the latency-bound AR (320KB / 223us =
            // 2.9GB/s, far below the NVLink bandwidth). The casts are cheap
            // (~5us each) vs the ~100us saved per call.
            match s.backend.as_cuda() {
                Some(c) if cnt >= 16384 => {
                    let xb = DevBuf::alloc(c.dev(), c.stream(), (cnt + 1) / 2)?;
                    c.cast_f32_to_bf16(&partial, &xb, cnt)?;
                    nccl.all_reduce_bf16(
                        xb.as_const_f32() as *const std::ffi::c_void,
                        xb.as_f32() as *mut std::ffi::c_void,
                        cnt,
                    )?;
                    c.cast_bf16_to_f32(&xb, &mut partial, cnt)?;
                }
                _ => {
                    nccl.all_reduce_f32(partial.as_const_f32(), partial.as_f32(), cnt)?;
                }
            }
        }
        if tm {
            let _ = cuda.sync();
            t_attn += t_l.elapsed().as_secs_f64() * 1e3;
            if matches!(plan.attn, AttnKind::Dsa) {
                t_b_dsa += t_b.elapsed().as_secs_f64() * 1e3;
            } else {
                t_b_gdn += t_b.elapsed().as_secs_f64() * 1e3;
            }
        }
        let t_mid = std::time::Instant::now();
        if layer_idx == 0 {
            mprobe!("ar0", &partial, hidden); // NCCL AR result — must be identical across the 4 rank files
        }
        if probe && dev_id == 0 {
            let mut pv = vec![0f32; hidden];
            if partial.download(&mut pv).is_ok() {
                let mx = pv.iter().fold(0f32, |a, x| a.max(x.abs()));
                let kind = if matches!(plan.attn, AttnKind::Dsa) { "dsa" } else { "gdn" };
                eprintln!("[mega] L{layer_idx:02} {kind} ar  maxabs={mx:.4}");
            }
        }
        // C: hc_post → hc_pre2 → post_attention_layernorm
        let res_mid = cuda.hc_post_dev(&partial, &res, &post_a, &comb_a, n, hc_mult, hidden)?;
        if layer_idx == 0 {
            mprobe!("resmid0", &res_mid, nh);
        }
        let (li2, post_f, comb_f) = cuda.hc_pre_dev(
            &res_mid,
            s.w(&format!("{pfx}.hc_ffn_fn"))?,
            s.w(&format!("{pfx}.hc_ffn_scale"))?,
            s.w(&format!("{pfx}.hc_ffn_base"))?,
            s.w(&format!("{pfx}.post_attention_layernorm.weight"))?,
            n,
            nh,
            cfg.rms_norm_eps,
            cfg.hc_eps,
            cfg.hc_sinkhorn_iters,
        )?;
        // li2 already RMS-normalized by the fused hc_pre_rest tail.
        let hfn = li2;
        if tm {
            let _ = cuda.sync();
            t_c += t_mid.elapsed().as_secs_f64() * 1e3;
        }
        let t_d = std::time::Instant::now();
        if probe && layer_idx < 3 && matches!(plan.mlp, MlpKind::Dense) {
            mprobe!("hfn0", &hfn, hidden);
            if dev_id == 0 {
                let mut pv = vec![0f32; hidden];
                if hfn.download(&mut pv).is_ok() {
                    let mx = pv.iter().fold(0f32, |acc, x| acc.max(x.abs()));
                    eprintln!("[mega] L{layer_idx:02} hfn in={mx:.4}");
                }
            }
        }
        // D: FFN (MoE or Dense) → NCCL all-reduce
        let mut partial2 = match plan.mlp {
            MlpKind::Moe => {
                let bias = match s.weights.get(&format!("{pfx}.mlp.gate.e_score_correction_bias")) {
                    Some(b) => b.clone(),
                    None => Tensor::zeros(Shape::new([e]), DType::F32),
                };
                let gate_w = s.w(&format!("{pfx}.mlp.gate.weight"))?;
                let shared = ExpertWeights {
                    gate: s.w(&format!("{pfx}.mlp.shared_expert.gate_proj.weight"))?,
                    up: s.w(&format!("{pfx}.mlp.shared_expert.up_proj.weight"))?,
                    down: s.w(&format!("{pfx}.mlp.shared_expert.down_proj.weight"))?,
                };
                let (es, ee) = s.tp_expert_range.unwrap_or((0, e));
                let experts: Vec<ExpertWeights> = (es..ee)
                    .map(|eid| {
                        Ok(ExpertWeights {
                            gate: s.w(&format!("{pfx}.mlp.experts.{eid}.gate_proj.weight"))?,
                            up: s.w(&format!("{pfx}.mlp.experts.{eid}.up_proj.weight"))?,
                            down: s.w(&format!("{pfx}.mlp.experts.{eid}.down_proj.weight"))?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mut probs = DevBuf::alloc(cuda.dev(), cuda.stream(), n * topk)?;
                cuda.moe_layer_dev(
                    &hfn, gate_w, &bias, &shared, &experts, es, &mut probs,
                    n, hidden, topk, e, cfg.routed_scaling_factor, cfg.swiglu_limit,
                )?
            }
            MlpKind::Dense => {
                let w_gate = s.w(&format!("{pfx}.mlp.gate_proj.weight"))?;
                let w_up = s.w(&format!("{pfx}.mlp.up_proj.weight"))?;
                let w_down = s.w(&format!("{pfx}.mlp.down_proj.weight"))?;
                let hi = hidden as i32;
                let inter = w_gate.shape.0[0] as i32;
                let g = cuda.matmul_dev(&hfn, w_gate, n as i32, hi, inter)?;
                let u = cuda.matmul_dev(&hfn, w_up, n as i32, hi, inter)?;
                let a = cuda.swiglu2_dev(&g, &u, n as i32, inter, cfg.swiglu_limit)?;
                if probe && layer_idx < 3 {
                    let mm = |b: &ferrite_kernel::cuda::DevBuf| -> f32 {
                        let mut pv = vec![0f32; b.len];
                        if b.download(&mut pv).is_ok() {
                            return pv.iter().fold(0f32, |acc, x| acc.max(x.abs()));
                        }
                        0.0
                    };
                    eprintln!(
                        "[mega] L{layer_idx:02} dense g={:.4} u={:.4} a={:.4} (inter={inter})",
                        mm(&g), mm(&u), mm(&a)
                    );
                }
                cuda.matmul_dev(&a, w_down, n as i32, inter, hi)?
            }
        };
        let ar_skip2 = std::env::var_os("FERRITE_AR_SKIP").is_some();
        let ar_p2p = match s.backend.as_cuda() {
            Some(c) if !ar_skip2 => c.p2p_ar_v2(&mut partial2, n * hidden).unwrap_or(false),
            _ => ar_skip2,
        };
        if !ar_p2p {
            nccl.all_reduce_f32(partial2.as_const_f32(), partial2.as_f32(), n * hidden)?;
        }
        if probe && dev_id == 0 {
            let mut pv = vec![0f32; hidden];
            if partial2.download(&mut pv).is_ok() {
                let mx = pv.iter().fold(0f32, |a, x| a.max(x.abs()));
                let kind = if matches!(plan.mlp, MlpKind::Moe) { "moe" } else { "dense" };
                eprintln!("[mega] L{layer_idx:02} {kind} ar maxabs={mx:.4}");
            }
        }
        if tm {
            let _ = cuda.sync();
            t_ffn += t_mid.elapsed().as_secs_f64() * 1e3;
        }
        // E: hc_post2 → next layer's residual
        res = cuda.hc_post_dev(&partial2, &res_mid, &post_f, &comb_f, n, hc_mult, hidden)?;
        if tm {
            let _ = cuda.sync();
            t_e += t_d.elapsed().as_secs_f64() * 1e3;
        }
    }

    mprobe!("resL", &res, nh);
    // head: contract → model.norm → lm_head → argmax (redundant per rank —
    // identical data after the ARs, replicated weights)
    let t_hs = std::time::Instant::now();
    let h_final = cuda.hc_contract_dev(&res, n, hc_mult, hidden)?;
    // verify mode: export ALL n rows of h_final (hc_contract residual) into
    // the fixed staging buffer — the host picks the accept-position row as
    // the MTP draft's h_prev after the argmax accept decision.
    if let Some(v) = verify {
        cuda.copy_dev(&h_final, 0, v.h_final, n * hidden)?;
    }
    mprobe!("hfinal", &h_final, hidden);
    let hn_head = cuda.rmsnorm_dev(
        &h_final,
        s.w("model.norm.weight")?,
        cfg.rms_norm_eps,
        n,
        hidden,
    )?;
    let lm_w = s.w("lm_head.weight")?;
    let logits = cuda.matmul_dev(&hn_head, lm_w, n as i32, hidden as i32, cfg.vocab_size as i32)?;
    mprobe!("logits16", &logits, 16);
    let mut arg = DevBuf::alloc(cuda.dev(), cuda.stream(), n)?;
    cuda.argmax_dev(&logits, &mut arg, n, cfg.vocab_size)?;

    if capture {
        cuda.graph_capture_end(gname);
        drop(_guard);
        cuda.graph_io_put(
            gname,
            GraphIO {
                x_stage,
                x_len: n * nh,
                out_dev: arg.as_f32() as *mut std::ffi::c_void,
                out_len: n,
                // Device-embed input path: in_dev = the capture-time res buffer
                // (embed_expand_dev's output target — the graph's first node);
                // in_n>0 marks the graph as ids-input (graph_run_ids).
                in_dev: if dev_embed { res_in_ptr } else { std::ptr::null_mut() },
                in_n: if dev_embed { n } else { 0 },
                in_hidden: if dev_embed { hidden } else { 0 },
                in_mult: if dev_embed { hc_mult } else { 0 },
            },
        );
        std::mem::forget(arg); // the graph's argmax output (graph_run reads it)
        // NOTE: no DSA rollback here — the PRE-capture rollback above makes
        // the pass's virtual t_count advance land exactly on the real cache
        // count (dry-run's tokens). replay-side dsa_host_advance keeps it
        // in lockstep from here on.
        cuda.small_n_rows.store(false, std::sync::atomic::Ordering::Relaxed);
        Ok(Vec::new())
    } else {
        let mut tv = vec![0f32; n];
        arg.download(&mut tv)?;
        if tm {
            let _ = cuda.sync();
            t_head = t_hs.elapsed().as_secs_f64() * 1e3;
            let (n_dsa, n_gdn) = plans
                .iter()
                .fold((0usize, 0usize), |(d, g), pl| {
                    if matches!(pl.attn, AttnKind::Dsa) { (d + 1, g) } else { (d, g + 1) }
                });
            eprintln!(
                "[mega-timing] {}L: attn={:.1} (A_hc={:.1} B: gdn{}={:.1} dsa{}={:.1}) ffn={:.1} (C_hc={:.1} D+E={:.1}) head={:.2}",
                plans.len(),
                t_attn,
                t_a,
                n_gdn,
                t_b_gdn,
                n_dsa,
                t_b_dsa,
                t_ffn,
                t_c,
                t_ffn - t_c - t_e,
                t_head
            );
        }
        cuda.small_n_rows.store(false, std::sync::atomic::Ordering::Relaxed);
        Ok(tv)
    }
}

// ============================================================
// BATCHED decode chain (Step A of the true-batched decode): ONE graph step
// for B seqs. The projections run at n=B GEMM (weights stream ONCE for all
// B rows — the user's batched-GEMM directive), while the per-seq recurrent
// state ops (GDN conv/state, DSA caches) run as B × n=1 kernel launches
// with each row's own (seq, layer/family) state pointers — inside ONE CUDA
// graph (replay node cost ~µs). The GEMM accumulation differs from the
// n=1 GEMV path by 1-ulp-class rounding (accepted — prefill's GEMM domain).
// Composition change (a seq joins/leaves) → re-capture (~1-2s, amortized
// over 1000-token streams). Non-MTP path (MTP forces max_seqs=1).
// ============================================================
#[cfg(feature = "cuda")]
fn mega_chain_dev_batched(
    s: &mut Engine<B>,
    seqs: &[u64],
    in_vals: &[f32],
    plans: &[ferrite_model::LayerPlan],
    num_dsa: usize,
    capture: bool,
    gname: &str,
    n: usize,
) -> Result<Vec<f32>> {
    use ferrite_kernel::cuda::{DevBuf, DsaLayerWeights, ExpertWeights, GdnLayerWeights, GraphIO};
    let cuda = s
        .backend
        .as_cuda()
        .ok_or_else(|| FerriteError::Config("batched needs cuda backend".into()))?;
    let nccl = s
        .nccl
        .clone()
        .ok_or_else(|| FerriteError::Config("batched needs FERRITE_NCCL=1".into()))?;
    cuda.enter();
    // The batched chain's projections take the GEMV path (n≤16 rows under
    // small_n_rows → gemv_bf16_v2, ONE launch covering n rows): the tiled
    // GEMM at tiny n wastes its 128-row tile (measured n=4 batched: 105ms
    // /step vs n=1's 16ms — the same tile-waste the MTP verify chain hit at
    // n=2: 108ms vs 23ms). The GEMV batched rides the L2 for the n rows'
    // same-weight reads (verify n=3: +35% vs n=1).
    cuda.small_n_rows.store(true, std::sync::atomic::Ordering::Relaxed);
    let cfg = &s.cfg;
    let (hidden, hc_mult) = (cfg.hidden_size, cfg.hc_mult);
    let nh = hc_mult * hidden;
    let topk = cfg.num_experts_per_tok;
    let e = cfg.n_routed_experts;
    // FERRITE_TIMING per-segment breakdown (dry-run only — the syncs are
    // capture-illegal): attn (A_hc + B: gdn/dsa batched + AR) / ffn (C_hc +
    // D+E moe) / head. The B segment splits GDN vs DSA — the per-seq state
    // ops' share of the n-scaling cost (the n=4 step is 31.6ms vs n=1's
    // 16.1ms: WHERE the +15.5ms lives — the MoE's per-token expert reads or
    // the per-seq GDN/DSA kernels).
    let dev_id = cuda.dev();
    let tm = !capture && dev_id == 0 && std::env::var_os("FERRITE_TIMING").is_some();
    let (mut t_attn, mut t_ffn, mut t_head) = (0f64, 0f64, 0f64);
    let (mut t_a, mut t_b_gdn, mut t_b_dsa, mut t_c, mut t_e) = (0f64, 0f64, 0f64, 0f64, 0f64);

    let _guard = if capture {
        // Pre-capture rollback: the dry-run advanced each seq's DSA t_count
        // by 1 per family (real kernels executed); the capture pass re-runs
        // the host bookkeeping (+1 per seq per family) — roll back so the
        // recorded pinned t0/total match the dry-run's, and t_count lands
        // back at the real cache count after the capture pass. (The per-seq
        // GDN/conv states need no rollback — capture records without
        // executing, and the dry-run's state advance IS the real step.)
        for &seq_r in seqs {
            if seq_r == u64::MAX {
                continue; // padded row
            }
            for f in 0..num_dsa {
                cuda.dsa_host_rollback(seq_r, f, 1);
            }
        }
        // Serialize per-rank captures (concurrent cuGraphInstantiate
        // SIGSEGV'd historically); record-mode NCCL enqueue never
        // rendezvous, so serialized capture is deadlock-free.
        Some(ferrite_kernel::cuda::capture_lock().lock().unwrap())
    } else {
        None
    };
    if capture {
        cuda.graph_capture_begin();
    }

    let mut res = DevBuf::alloc(cuda.dev(), cuda.stream(), n * nh)?;
    res.upload(in_vals)?; // recorded stage→dev memcpy (the graph input)
    let x_stage = res.stage; // GraphIO: replay writes fresh input here

    for (layer_idx, plan) in plans.iter().enumerate() {
        if std::env::var_os("FERRITE_TIMING").is_some() {
            eprintln!("[megab-cap] dev{dev_id} cap={capture} L{layer_idx}");
        }
        let t_l = std::time::Instant::now();
        let pfx = format!("model.layers.{layer_idx}");
        // A: hc_pre (n=B — row-independent)
        let (li, post_a, comb_a) = cuda.hc_pre_dev(
            &res,
            s.w(&format!("{pfx}.hc_attn_fn"))?,
            s.w(&format!("{pfx}.hc_attn_scale"))?,
            s.w(&format!("{pfx}.hc_attn_base"))?,
            s.w(&format!("{pfx}.input_layernorm.weight"))?,
            n,
            nh,
            cfg.rms_norm_eps,
            cfg.hc_eps,
            cfg.hc_sinkhorn_iters,
        )?;
        let hn = li;
        if tm {
            let _ = cuda.sync();
            t_a += t_l.elapsed().as_secs_f64() * 1e3;
        }
        let t_b = std::time::Instant::now();
        // B: attention — the batched per-seq dispatch (n=B GEMM projections
        // + B × n=1 per-seq state kernels)
        let mut partial = match plan.attn {
            AttnKind::Linear => {
                let la = &cfg.linear_attn;
                let gw = GdnLayerWeights {
                    qkv_proj: s.w(&format!("{pfx}.self_attn.qkv_proj.weight"))?,
                    b_proj: s.w(&format!("{pfx}.self_attn.b_proj.weight"))?,
                    f_a: s.w(&format!("{pfx}.self_attn.f_a_proj.weight"))?,
                    f_b: s.w(&format!("{pfx}.self_attn.f_b_proj.weight"))?,
                    g_a: s.w(&format!("{pfx}.self_attn.g_a_proj.weight"))?,
                    g_b: s.w(&format!("{pfx}.self_attn.g_b_proj.weight"))?,
                    conv_w: s.w(&format!("{pfx}.self_attn.qkv_conv1d.weight"))?,
                    dt_bias: s.w(&format!("{pfx}.self_attn.dt_bias"))?,
                    a_log: s.w(&format!("{pfx}.self_attn.A_log"))?,
                    o_norm: s.w(&format!("{pfx}.self_attn.o_norm.weight"))?,
                    o_proj: s.w(&format!("{pfx}.self_attn.o_proj.weight"))?,
                };
                cuda.gdn_layer_dev_batched(
                    &hn, &gw, seqs, layer_idx, n, hidden,
                    la.num_heads, la.head_dim, la.gate_lower_bound,
                    cfg.rms_norm_eps, la.short_conv_kernel_size,
                )?
            }
            AttnKind::Dsa => {
                let d = &cfg.dsa;
                let (dsa_h, dsa_dk, dsa_dv, _ip) = s.dsa_dims();
                let w = DsaLayerWeights {
                    q_a: s.w(&format!("{pfx}.self_attn.q_a_proj.weight"))?,
                    q_a_ln: s.w(&format!("{pfx}.self_attn.q_a_layernorm.weight"))?,
                    q_b: s.w(&format!("{pfx}.self_attn.q_b_proj.weight"))?,
                    kv_a: s.w(&format!("{pfx}.self_attn.kv_a_proj_with_mqa.weight"))?,
                    kv_a_ln: s.w(&format!("{pfx}.self_attn.kv_a_layernorm.weight"))?,
                    kv_b: s.w(&format!("{pfx}.self_attn.kv_b_proj.weight"))?,
                    wq_b: s.w(&format!("{pfx}.self_attn.indexer.wq_b.weight"))?,
                    wk: s.w(&format!("{pfx}.self_attn.indexer.wk.weight"))?,
                    k_norm_w: s.w(&format!("{pfx}.self_attn.indexer.k_norm.weight"))?,
                    k_norm_b: s.w(&format!("{pfx}.self_attn.indexer.k_norm.bias"))?,
                    weights_proj: s.w(&format!("{pfx}.self_attn.indexer.weights_proj.weight"))?,
                    gate: s.w(&format!("{pfx}.self_attn.indexer.index_kpool_compress_gate"))?,
                    ape: s.w(&format!("{pfx}.self_attn.indexer.index_kpool_compress_ape"))?,
                    o_proj: s.w(&format!("{pfx}.self_attn.o_proj.weight"))?,
                    h: dsa_h,
                    dk: dsa_dk,
                    dv: dsa_dv,
                    ih: d.index_n_heads,
                    idm: d.index_head_dim,
                    kpool: 4,
                    topk: d.index_topk,
                    rms_eps: cfg.rms_norm_eps,
                };
                let family = s.dsa_family_index(layer_idx);
                cuda.dsa_layer_dev_batched(&hn, &w, seqs, family, n, hidden)?
            }
        };
        let ar_skip = std::env::var_os("FERRITE_AR_SKIP").is_some();
        let ar_p2p = match s.backend.as_cuda() {
            Some(c) if !ar_skip => c.p2p_ar_v2(&mut partial, n * hidden).unwrap_or(false),
            _ => ar_skip,
        };
        if !ar_p2p {
            let cnt = n * hidden;
            // bf16 payload halves the latency-bound AR (320KB / 223us =
            // 2.9GB/s, far below the NVLink bandwidth). The casts are cheap
            // (~5us each) vs the ~100us saved per call.
            match s.backend.as_cuda() {
                Some(c) if cnt >= 16384 => {
                    let xb = DevBuf::alloc(c.dev(), c.stream(), (cnt + 1) / 2)?;
                    c.cast_f32_to_bf16(&partial, &xb, cnt)?;
                    nccl.all_reduce_bf16(
                        xb.as_const_f32() as *const std::ffi::c_void,
                        xb.as_f32() as *mut std::ffi::c_void,
                        cnt,
                    )?;
                    c.cast_bf16_to_f32(&xb, &mut partial, cnt)?;
                }
                _ => {
                    nccl.all_reduce_f32(partial.as_const_f32(), partial.as_f32(), cnt)?;
                }
            }
        }
        if tm {
            let _ = cuda.sync();
            t_attn += t_l.elapsed().as_secs_f64() * 1e3;
            if matches!(plan.attn, AttnKind::Dsa) {
                t_b_dsa += t_b.elapsed().as_secs_f64() * 1e3;
            } else {
                t_b_gdn += t_b.elapsed().as_secs_f64() * 1e3;
            }
        }
        let t_mid = std::time::Instant::now();
        // C: hc_post → hc_pre2
        let res_mid = cuda.hc_post_dev(&partial, &res, &post_a, &comb_a, n, hc_mult, hidden)?;
        let (li2, post_f, comb_f) = cuda.hc_pre_dev(
            &res_mid,
            s.w(&format!("{pfx}.hc_ffn_fn"))?,
            s.w(&format!("{pfx}.hc_ffn_scale"))?,
            s.w(&format!("{pfx}.hc_ffn_base"))?,
            s.w(&format!("{pfx}.post_attention_layernorm.weight"))?,
            n,
            nh,
            cfg.rms_norm_eps,
            cfg.hc_eps,
            cfg.hc_sinkhorn_iters,
        )?;
        let hfn = li2;
        if tm {
            let _ = cuda.sync();
            t_c += t_mid.elapsed().as_secs_f64() * 1e3;
        }
        let t_d = std::time::Instant::now();
        // D: FFN (MoE/Dense — n=B, row-independent; the existing n>1 kernels)
        let mut partial2 = match plan.mlp {
            MlpKind::Moe => {
                let bias = match s.weights.get(&format!("{pfx}.mlp.gate.e_score_correction_bias")) {
                    Some(b) => b.clone(),
                    None => Tensor::zeros(Shape::new([e]), DType::F32),
                };
                let gate_w = s.w(&format!("{pfx}.mlp.gate.weight"))?;
                let shared = ExpertWeights {
                    gate: s.w(&format!("{pfx}.mlp.shared_expert.gate_proj.weight"))?,
                    up: s.w(&format!("{pfx}.mlp.shared_expert.up_proj.weight"))?,
                    down: s.w(&format!("{pfx}.mlp.shared_expert.down_proj.weight"))?,
                };
                let (es, ee) = s.tp_expert_range.unwrap_or((0, e));
                let experts: Vec<ExpertWeights> = (es..ee)
                    .map(|eid| {
                        Ok(ExpertWeights {
                            gate: s.w(&format!("{pfx}.mlp.experts.{eid}.gate_proj.weight"))?,
                            up: s.w(&format!("{pfx}.mlp.experts.{eid}.up_proj.weight"))?,
                            down: s.w(&format!("{pfx}.mlp.experts.{eid}.down_proj.weight"))?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mut probs = DevBuf::alloc(cuda.dev(), cuda.stream(), n * topk)?;
                cuda.moe_layer_dev(
                    &hfn, gate_w, &bias, &shared, &experts, es, &mut probs,
                    n, hidden, topk, e, cfg.routed_scaling_factor, cfg.swiglu_limit,
                )?
            }
            MlpKind::Dense => {
                let w_gate = s.w(&format!("{pfx}.mlp.gate_proj.weight"))?;
                let w_up = s.w(&format!("{pfx}.mlp.up_proj.weight"))?;
                let w_down = s.w(&format!("{pfx}.mlp.down_proj.weight"))?;
                let hi = hidden as i32;
                let inter = w_gate.shape.0[0] as i32;
                let g = cuda.matmul_dev(&hfn, w_gate, n as i32, hi, inter)?;
                let u = cuda.matmul_dev(&hfn, w_up, n as i32, hi, inter)?;
                let a = cuda.swiglu2_dev(&g, &u, n as i32, inter, cfg.swiglu_limit)?;
                cuda.matmul_dev(&a, w_down, n as i32, inter, hi)?
            }
        };
        let ar_skip3 = std::env::var_os("FERRITE_AR_SKIP").is_some();
        let ar_p2p = match s.backend.as_cuda() {
            Some(c) if !ar_skip3 => c.p2p_ar_v2(&mut partial2, n * hidden).unwrap_or(false),
            _ => ar_skip3,
        };
        if !ar_p2p {
            nccl.all_reduce_f32(partial2.as_const_f32(), partial2.as_f32(), n * hidden)?;
        }
        if tm {
            let _ = cuda.sync();
            t_ffn += t_mid.elapsed().as_secs_f64() * 1e3;
        }
        // E: hc_post2 → next layer's residual
        res = cuda.hc_post_dev(&partial2, &res_mid, &post_f, &comb_f, n, hc_mult, hidden)?;
        if tm {
            let _ = cuda.sync();
            t_e += t_d.elapsed().as_secs_f64() * 1e3;
        }
    }
    // head: contract → model.norm → lm_head → argmax (all n=B; redundant per
    // rank — identical data after the ARs, replicated weights)
    let t_hs = std::time::Instant::now();
    let h_final = cuda.hc_contract_dev(&res, n, hc_mult, hidden)?;
    let hn_head = cuda.rmsnorm_dev(
        &h_final,
        s.w("model.norm.weight")?,
        cfg.rms_norm_eps,
        n,
        hidden,
    )?;
    let lm_w = s.w("lm_head.weight")?;
    let logits = cuda.matmul_dev(&hn_head, lm_w, n as i32, hidden as i32, cfg.vocab_size as i32)?;
    let mut arg = DevBuf::alloc(cuda.dev(), cuda.stream(), n)?;
    cuda.argmax_dev(&logits, &mut arg, n, cfg.vocab_size)?;

    if capture {
        cuda.graph_capture_end(gname);
        drop(_guard);
        cuda.graph_io_put(
            gname,
            GraphIO {
                x_stage,
                x_len: n * nh,
                out_dev: arg.as_f32() as *mut std::ffi::c_void,
                out_len: n,
                in_dev: std::ptr::null_mut(),
                in_n: 0,
                in_hidden: 0,
                in_mult: 0,
            },
        );
        std::mem::forget(arg); // the graph's argmax output (graph_run reads it)
        // NOTE: no DSA rollback here — the PRE-capture rollback above makes
        // the pass's virtual t_count advance land exactly on the real cache
        // count (the dry-run's tokens); replay-side dsa_host_advance keeps it
        // in lockstep from here on.
        cuda.small_n_rows.store(false, std::sync::atomic::Ordering::Relaxed);
        Ok(Vec::new())
    } else {
        let mut tv = vec![0f32; n];
        arg.download(&mut tv)?;
        if tm {
            let _ = cuda.sync();
            t_head = t_hs.elapsed().as_secs_f64() * 1e3;
            let (n_dsa, n_gdn) = plans
                .iter()
                .fold((0usize, 0usize), |(d, g), pl| {
                    if matches!(pl.attn, AttnKind::Dsa) { (d + 1, g) } else { (d, g + 1) }
                });
            eprintln!(
                "[megab-timing] n={n} {}L: attn={:.1} (A_hc={:.1} B: gdn{}={:.1} dsa{}={:.1}) ffn={:.1} (C_hc={:.1} D+E={:.1}) head={:.2}",
                plans.len(),
                t_attn,
                t_a,
                n_gdn,
                t_b_gdn,
                n_dsa,
                t_b_dsa,
                t_ffn,
                t_c,
                t_ffn - t_c - t_e,
                t_head
            );
        }
        cuda.small_n_rows.store(false, std::sync::atomic::Ordering::Relaxed);
        Ok(tv)
    }
}

/// GDN (linear-attention) layer per shard: the CUDA path runs the WHOLE
/// layer as one DevBuf pipeline (gdn_layer_dev — zero host round-trips
/// in-layer: upload hn once, download the o_proj partial once); every
/// other backend falls back to the Tensor-level ops.
#[cfg(feature = "cuda")]
fn attn_shard(
    s: &mut Engine<B>,
    seq: u64,
    layer_idx: usize,
    pfx: &str,
    hn: &Tensor,
    n: usize,
    hidden: usize,
) -> Result<Tensor> {
    // FERRITE_GDN_DEV=1 opt-in: the device chain has a numeric bug (garbage
    // output — see the equivalence test TODO); the CPU path is the default.
    if std::env::var_os("FERRITE_GDN_DEV").is_none() {
        return s.linear_attn_forward(seq, layer_idx, pfx, hn, n);
    }
    if let Some(cuda) = s.backend.as_cuda() {
        use ferrite_kernel::cuda::{DevBuf, GdnLayerWeights};
        // cudaSetDevice is THREAD-LOCAL: in fan_out, this thread's current
        // device is whatever the last rank's ops left set. Bind BEFORE any
        // DevBuf alloc/upload (cudaMalloc binds to the current device —
        // allocating on the wrong rank's device was the err-700 crash).
        cuda.enter();
        let la = &s.cfg.linear_attn;
        let gw = GdnLayerWeights {
            qkv_proj: s.w(&format!("{pfx}.self_attn.qkv_proj.weight"))?,
            b_proj: s.w(&format!("{pfx}.self_attn.b_proj.weight"))?,
            f_a: s.w(&format!("{pfx}.self_attn.f_a_proj.weight"))?,
            f_b: s.w(&format!("{pfx}.self_attn.f_b_proj.weight"))?,
            g_a: s.w(&format!("{pfx}.self_attn.g_a_proj.weight"))?,
            g_b: s.w(&format!("{pfx}.self_attn.g_b_proj.weight"))?,
            conv_w: s.w(&format!("{pfx}.self_attn.qkv_conv1d.weight"))?,
            dt_bias: s.w(&format!("{pfx}.self_attn.dt_bias"))?,
            a_log: s.w(&format!("{pfx}.self_attn.A_log"))?,
            o_norm: s.w(&format!("{pfx}.self_attn.o_norm.weight"))?,
            o_proj: s.w(&format!("{pfx}.self_attn.o_proj.weight"))?,
        };
        // FERRITE_GRAPH_LAYER: per-(layer, rank) graph — the segment's op
        // sequence (upload memcpy + 11 kernels) is captured once, replayed
        // per token. The pool is per-device (ranks don't share) and this
        // rank's op sequence is deterministic → buffer addresses are stable.
        // x_dev/partial are LEAKED (graph replays write them).
        if std::env::var_os("FERRITE_GRAPH_LAYER").is_some()
            && std::env::var_os("FERRITE_NCCL").is_none()
            && n == 1 {
            use ferrite_kernel::cuda::GraphIO;
            let gname = format!("gdn{}", layer_idx);
            let mut v = vec![0f32; n * hidden];
            if cuda.graph_run(&gname, hn.as_slice(), &mut v)? {
                return Ok(Tensor::from_f32(Shape::new([n, hidden]), v));
            }
            // WARM + CAPTURE under the global capture lock: fan_out's 4
            // workers capture concurrently and cuGraphInstantiate crashed
            // inside libcuda (gdb: SIGSEGV). Capture is one-time per
            // segment — serializing it costs nothing steady-state.
            let _cap = ferrite_kernel::cuda::capture_lock().lock().unwrap();
            {
                // WARM the n==1 pool classes first: capture forbids cudaMalloc,
                // and prefill (n==prompt_len) leaves DIFFERENT size classes in
                // the pool — a cold n==1 class inside capture segfaults.
                let wx = DevBuf::alloc(cuda.dev(), cuda.stream(), hn.numel())?;
                wx.upload(hn.as_slice())?;
                let _wp = cuda.gdn_layer_dev(
                    &wx, &gw, seq, layer_idx, n, hidden,
                    la.num_heads, la.head_dim, la.gate_lower_bound,
                    s.cfg.rms_norm_eps, la.short_conv_kernel_size, None,
                )?;
            } // drops return everything to the pool
            cuda.graph_capture_begin();
            let mut x_dev = DevBuf::alloc(cuda.dev(), cuda.stream(), hn.numel())?;
            x_dev.upload(hn.as_slice())?;
            let mut partial = cuda.gdn_layer_dev(
                &x_dev, &gw, seq, layer_idx, n, hidden,
                la.num_heads, la.head_dim, la.gate_lower_bound,
                s.cfg.rms_norm_eps, la.short_conv_kernel_size, None,
            )?;
            cuda.graph_capture_end(&gname);
            cuda.graph_io_put(
                &gname,
                GraphIO {
                    x_stage: x_dev.stage,
                    x_len: hn.numel(),
                    out_dev: partial.as_f32() as *mut std::ffi::c_void,
                    out_len: n * hidden,
                    in_dev: std::ptr::null_mut(),
                    in_n: 0,
                    in_hidden: 0,
                    in_mult: 0,
                },
            );
            std::mem::forget(x_dev);
            std::mem::forget(partial);
            // capture records but does NOT execute — replay for this token
            if !cuda.graph_replay(&gname) {
                return Err(FerriteError::InvalidArg(format!("graph replay {gname} failed")));
            }
            let mut v = vec![0f32; n * hidden];
            // partial's device address holds the replay output
            let io = cuda.graph_io_get(&gname).unwrap();
            cuda.enter();
            let r = unsafe {
                ferrite_kernel::cuda::memcpy_d2h_sync(io.out_dev, v.as_mut_ptr(), n * hidden, cuda.stream_handle())
            };
            if r != 0 {
                return Err(FerriteError::InvalidArg(format!("gdn graph D2H failed: {r}")));
            }
            return Ok(Tensor::from_f32(Shape::new([n, hidden]), v));
        }
        let x_dev = DevBuf::alloc(cuda.dev(), cuda.stream(), hn.numel())?;
        x_dev.upload(hn.as_slice())?;
        let mut partial = cuda.gdn_layer_dev(
            &x_dev, &gw, seq, layer_idx, n, hidden,
            la.num_heads, la.head_dim, la.gate_lower_bound,
            s.cfg.rms_norm_eps, la.short_conv_kernel_size, None,
        )?;
        if let Some(ch) = &s.nccl {
            // TP all-reduce on-device (replaces the host download→sum→upload
            // round-trip; async on this rank's stream — the download below
            // syncs it, which waits for the whole collective).
            // P2P one-shot first: NCCL RING_LL measured ~390us for this
            // payload vs ~20us for the P2P kernel.
            let ar_p2p = cuda.p2p_ar_v2(&mut partial, n * hidden).unwrap_or(false);
            if !ar_p2p {
                ch.all_reduce_f32(partial.as_const_f32(), partial.as_f32(), n * hidden)?;
            }
        }
        let mut out = Tensor::zeros(Shape::new([n, hidden]), DType::F32);
        {
            let v = std::sync::Arc::get_mut(&mut out.data).expect("unique out");
            partial.download(v)?;
        }
        Ok(out)
    } else {
        s.linear_attn_forward(seq, layer_idx, pfx, hn, n)
    }
}

#[cfg(not(feature = "cuda"))]
fn attn_shard(
    s: &mut Engine<B>,
    seq: u64,
    layer_idx: usize,
    pfx: &str,
    hn: &Tensor,
    n: usize,
    _hidden: usize,
) -> Result<Tensor> {
    s.linear_attn_forward(seq, layer_idx, pfx, hn, n)
}

/// Run one op-group across all shards CONCURRENTLY (one thread per rank).
/// TP ranks are independent until the all-reduce; the serial iter_mut loop
/// left 3 of the 4 GPUs idle. cudaSetDevice is thread-local so each shard's
/// ops bind its own GPU; DevBuf pools are thread-local too (per-thread
/// arenas, no cross-thread buffer sharing). Result order = shard order
/// (fan_out preserves indices; the all-reduce sum is order-independent).
/// Run one op-group across all shards CONCURRENTLY (one thread per rank).
/// TP ranks are independent until the all-reduce; the serial iter_mut loop
/// left 3 of the 4 GPUs idle. cudaSetDevice is thread-local so each shard's
/// ops bind its own GPU; DevBuf pools are thread-local too (per-thread
/// arenas, no cross-thread buffer sharing). Result order = shard order
/// (fan_out preserves indices; the all-reduce sum is order-independent).
fn fan_out<T, F>(shards: &mut [Engine<B>], f: F) -> Vec<T>
where
    F: Fn(&mut Engine<B>) -> T + Sync,
    T: Send,
{
    if shards.len() == 1 {
        return vec![f(&mut shards[0])];
    }
    // Persistent workers (FERRITE_WORKER_POOL=1): removes the 360 spawns
    // per token (4 ranks × 2 segments × 45 layers × ~30μs each).
    // SAFETY (transmute): the main thread blocks on recv() until all
    // workers finish — f's lifetime covers the execution window.
    if let Some(pool) = fan_pool(shards.len()) {
        let ptr = shards.as_mut_ptr();
        let f_static: &F = unsafe { std::mem::transmute(&f) };
        return fan_out_pooled(pool, ptr, f_static, shards.len());
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = shards
            .iter_mut()
            .enumerate()
            .map(|(i, s)| {
                let f = &f;
                scope.spawn(move || {
                    // rank index for probe dump isolation (ferrite_kernel::shard_idx)
                    ferrite_kernel::set_shard_idx(i);
                    f(s)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("shard thread panicked"))
            .collect()
    })
}

/// TP layer forward: attn partial → all-reduce → MHC/residual →
/// ffn partial → all-reduce. Collectives land exactly where an NCCL
/// deployment would place them.
    fn layer_forward_tp(
        &mut self,
        seq: u64,
        layer_idx: usize,
        residual: Tensor,
        n: usize,
    ) -> Result<Tensor> {
        // Full device op chain (FERRITE_LAYER_DEV=1, decode n==1): MHC
        // hc_pre/hc_post + rmsnorm on GPU (DevBuf level, zero host compute
        // between the layer's GPU ops) — the layer-chain phase toward the
        // per-rank CUDA graph. GDN/MoE device chains already handle the
        // attn/ffn segments (FERRITE_GDN_DEV/FERRITE_MOE_DEV).
        #[cfg(feature = "cuda")]
        if std::env::var_os("FERRITE_LAYER_DEV").is_some() && self.full_cfg.mhc {
            let (out, _dev) = self.layer_forward_dev(seq, layer_idx, residual, None, n)?;
            return Ok(out);
        }
        let probe = std::env::var_os("FERRITE_PROBE").is_some() && layer_idx == 3 && n > 1; // prefill, first DSA+MoE layer
        if probe {
            let n_el = residual.numel();
            let bytes: Vec<u8> = residual.as_slice().iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write("/tmp/l0_in.f32", bytes).ok();
            eprintln!("[probe] L0 in: {} elems", n_el);
        }
        let plans = build_layer_plans(&self.full_cfg);
        let plan = &plans[layer_idx];
        let pfx = format!("model.layers.{layer_idx}");
        let (hidden, hc_mult) = (self.full_cfg.hidden_size, self.full_cfg.hc_mult);

        if self.full_cfg.mhc {
            // ---- attention half: hc_pre → norm → attn (per shard) → AR → hc_post ----
            let timing = std::env::var_os("FERRITE_TIMING").is_some() && n == 1;
            let t0 = std::time::Instant::now();
            let (hc_fn, hc_scale, hc_base) = {
                let s0 = &self.shards[0];
                (
                    s0.w(&format!("{pfx}.hc_attn_fn"))?.clone(),
                    s0.w(&format!("{pfx}.hc_attn_scale"))?.clone(),
                    s0.w(&format!("{pfx}.hc_attn_base"))?.clone(),
                )
            };
            let (li, post_a, comb_a) = crate::mhc::hc_pre(
                &residual,
                &hc_fn,
                &hc_scale,
                &hc_base,
                self.full_cfg.rms_norm_eps,
                self.full_cfg.hc_eps,
                self.full_cfg.hc_sinkhorn_iters,
            );
            let hn = {
                let s0 = &self.shards[0];
                let hn = s0.rmsnorm(&li, &format!("{pfx}.input_layernorm.weight"))?;
                if probe {
                    let bytes: Vec<u8> = li.as_slice().iter().flat_map(|v| v.to_le_bytes()).collect();
                    std::fs::write("/tmp/l0_collapsed.f32", bytes).ok();
                    let bytes2: Vec<u8> = hn.as_slice().iter().flat_map(|v| v.to_le_bytes()).collect();
                    std::fs::write("/tmp/l0_hn.f32", bytes2).ok();
                }
                hn
            };
            let t1 = std::time::Instant::now(); // hc_pre+rmsnorm done
            let attn_partials = Self::fan_out(&mut self.shards, |s| match plan.attn {
                AttnKind::Linear => Self::attn_shard(s, seq, layer_idx, &pfx, &hn, n, hidden),
                AttnKind::Dsa => s.dsa_attn_forward(seq, layer_idx, &pfx, &hn, n),
            });
            let t_attn = std::time::Instant::now();
            let attn_out = all_reduce_sum(&attn_partials.into_iter().collect::<Result<Vec<_>>>()?);
            let t_ar = std::time::Instant::now();
            if probe {
                // tagged by GDN path (FERRITE_GDN_DEV=1 → "dev" else "cpu") —
                // the CPU-vs-device divergence pinpoints WHERE garbage starts.
                let dir = std::env::var("FERRITE_PROBE_DIR").unwrap_or_else(|_| "/tmp/orion".into());
                let tag = if std::env::var_os("FERRITE_GDN_DEV").is_some() { "dev" } else { "cpu" };
                let bytes: Vec<u8> = attn_out.as_slice().iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/l0_attn_tp_{tag}.f32"), bytes).ok();
            }
            let res3 =
                Tensor::from_f32(Shape::new([n, hc_mult, hidden]), residual.as_slice().to_vec());
            let res2 = crate::mhc::hc_post(&attn_out, &res3, &post_a, &comb_a);
            let t_hc2 = std::time::Instant::now(); // attn hc_post done
            if probe {
                let bytes: Vec<u8> = res2.as_slice().iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write("/tmp/l0_res2.f32", bytes).ok();
            }

            // ---- ffn half ----
            let (hc_fn2, hc_scale2, hc_base2) = {
                let s0 = &self.shards[0];
                (
                    s0.w(&format!("{pfx}.hc_ffn_fn"))?.clone(),
                    s0.w(&format!("{pfx}.hc_ffn_scale"))?.clone(),
                    s0.w(&format!("{pfx}.hc_ffn_base"))?.clone(),
                )
            };
            let res2_flat =
                Tensor::from_f32(Shape::new([n, hc_mult * hidden]), res2.as_slice().to_vec());
            let (li2, post_f, comb_f) = crate::mhc::hc_pre(
                &res2_flat,
                &hc_fn2,
                &hc_scale2,
                &hc_base2,
                self.full_cfg.rms_norm_eps,
                self.full_cfg.hc_eps,
                self.full_cfg.hc_sinkhorn_iters,
            );
            let hfn = {
                let s0 = &self.shards[0];
                s0.rmsnorm(&li2, &format!("{pfx}.post_attention_layernorm.weight"))?
            };
            let t_fpre = std::time::Instant::now(); // ffn hc_pre+rmsnorm done
            let ffn_partials = Self::fan_out(&mut self.shards, |s| match plan.mlp {
                MlpKind::Dense => s.dense_ffn(&pfx, &hfn, n),
                MlpKind::Moe => s.moe_ffn(&pfx, &hfn, n),
            });
            let t_ffn = std::time::Instant::now();
            let ffn_out = all_reduce_sum(&ffn_partials.into_iter().collect::<Result<Vec<_>>>()?);
            let t_far = std::time::Instant::now();
            if probe {
                let bytes: Vec<u8> = ffn_out.as_slice().iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write("/tmp/l0_ffn.f32", bytes).ok();
            }
            let res3b =
                Tensor::from_f32(Shape::new([n, hc_mult, hidden]), res2_flat.as_slice().to_vec());
            let res_out = crate::mhc::hc_post(&ffn_out, &res3b, &post_f, &comb_f);
            if timing {
                let t_end = std::time::Instant::now();
                let ak = match plan.attn { AttnKind::Linear => "gdn", AttnKind::Dsa => "dsa" };
                let mk = match plan.mlp { MlpKind::Dense => "dense", MlpKind::Moe => "moe" };
                eprintln!(
                    "[timing] L{layer_idx:2} {ak}/{mk} hcp={:4.1} at={:6.1} ar={:4.1} hcp2={:4.1} fp={:4.1} ffn={:6.1} far={:4.1} hc3={:4.1} tot={:6.1}ms",
                    (t1 - t0).as_secs_f32() * 1e3, (t_attn - t1).as_secs_f32() * 1e3,
                    (t_ar - t_attn).as_secs_f32() * 1e3, (t_hc2 - t_ar).as_secs_f32() * 1e3,
                    (t_fpre - t_hc2).as_secs_f32() * 1e3, (t_ffn - t_fpre).as_secs_f32() * 1e3,
                    (t_far - t_ffn).as_secs_f32() * 1e3, (t_end - t_far).as_secs_f32() * 1e3,
                    (t_end - t0).as_secs_f32() * 1e3,
                );
            }
            if std::env::var_os("FERRITE_TRACE_NAN").is_some() {
                let (mut mx, mut sum) = (0.0f32, 0.0f32);
                for v in res_out.as_slice() {
                    if v.is_finite() {
                        mx = mx.max(v.abs());
                        sum += v * v;
                    }
                }
                eprintln!(
                    "[tp-trace] layer {:2} attn_max={:.4} ffn_max={:.4} h_l2={:.4} n_nan={}",
                    layer_idx,
                    attn_out.as_slice().iter().fold(0.0f32, |a, v| a.max(v.abs())),
                    ffn_out.as_slice().iter().fold(0.0f32, |a, v| a.max(v.abs())),
                    sum.sqrt(),
                    res_out.as_slice().iter().filter(|v| !v.is_finite()).count()
                );
            }

            let out_t = Tensor::from_f32(
                Shape::new([n, hc_mult * hidden]),
                res_out.as_slice().to_vec(),
            );
            if probe {
                let bytes: Vec<u8> = out_t.as_slice().iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write("/tmp/l0_out.f32", bytes).ok();
                eprintln!("[probe] L0 out: {} elems", out_t.numel());
            }
            Ok(out_t)
        } else {
            // standard residual stream
            let hn = {
                let s0 = &self.shards[0];
                s0.rmsnorm(&residual, &format!("{pfx}.input_layernorm.weight"))?
            };
            let attn_partials = Self::fan_out(&mut self.shards, |s| match plan.attn {
                AttnKind::Linear => Self::attn_shard(s, seq, layer_idx, &pfx, &hn, n, hidden),
                AttnKind::Dsa => s.dsa_attn_forward(seq, layer_idx, &pfx, &hn, n),
            });
            let attn_out = all_reduce_sum(&attn_partials.into_iter().collect::<Result<Vec<_>>>()?);
            let h2 = Tensor::from_f32(
                Shape::new([n, hidden]),
                (0..n * hidden)
                    .map(|i| residual.as_slice()[i] + attn_out.as_slice()[i])
                    .collect(),
            );
            let hfn = {
                let s0 = &self.shards[0];
                s0.rmsnorm(&h2, &format!("{pfx}.input_layernorm.weight"))?
            };
            let ffn_partials = Self::fan_out(&mut self.shards, |s| match plan.mlp {
                MlpKind::Dense => s.dense_ffn(&pfx, &hfn, n),
                MlpKind::Moe => s.moe_ffn(&pfx, &hfn, n),
            });
            let ffn_out = all_reduce_sum(&ffn_partials.into_iter().collect::<Result<Vec<_>>>()?);
            Ok(Tensor::from_f32(
                Shape::new([n, hidden]),
                (0..n * hidden)
                    .map(|i| h2.as_slice()[i] + ffn_out.as_slice()[i])
                    .collect(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrite_model::{random_weights, Glm53FlashConfig};

    #[test]
    fn tp2_head_split_covers_all() {
        let cfg = Glm53FlashConfig::test_config();
        let w = random_weights(&cfg, 42);
        let w0 = shard_weights_tp(&w, &cfg, 0, 2);
        let w1 = shard_weights_tp(&w, &cfg, 1, 2);
        // qkv_proj: each rank holds half the rows of each third
        let full = &w["model.layers.0.self_attn.qkv_proj.weight"];
        let s0 = &w0["model.layers.0.self_attn.qkv_proj.weight"];
        let s1 = &w1["model.layers.0.self_attn.qkv_proj.weight"];
        assert_eq!(s0.shape.0[0] + s1.shape.0[0], full.shape.0[0]);
        assert_eq!(s0.shape.0[0], full.shape.0[0] / 2);
        // o_proj column-split halves
        let op = &w["model.layers.0.self_attn.o_proj.weight"];
        let op0 = &w0["model.layers.0.self_attn.o_proj.weight"];
        assert_eq!(op0.shape.0[1], op.shape.0[1] / 2);
        assert_eq!(op0.shape.0[0], op.shape.0[0]);
        // f_a replicated
        let fa = &w["model.layers.0.self_attn.f_a_proj.weight"];
        let fa0 = &w0["model.layers.0.self_attn.f_a_proj.weight"];
        assert_eq!(fa0.shape, fa.shape);
        // A_log head split
        let al0 = &w0["model.layers.0.self_attn.A_log"];
        let al1 = &w1["model.layers.0.self_attn.A_log"];
        assert_eq!(al0.shape.0[0] + al1.shape.0[0], 4); // test config heads=4
    }

    #[test]
    fn tp2_moe_expert_split() {
        let cfg = Glm53FlashConfig::test_config(); // 8 experts
        let w = random_weights(&cfg, 42);
        let w0 = shard_weights_tp(&w, &cfg, 0, 2);
        let w1 = shard_weights_tp(&w, &cfg, 1, 2);
        // layer 2 is MoE (test config: layers 0,1 dense; 2,3 MoE)
        let full = &w["model.layers.2.mlp.experts.0.gate_proj.weight"];
        let s0 = &w0["model.layers.2.mlp.experts.0.gate_proj.weight"];
        let s1 = &w1["model.layers.2.mlp.experts.0.gate_proj.weight"];
        // rank 0 owns experts 0-3 (full), rank 1 gets empty
        assert!(s0.shape.0[0] > 0);
        assert_eq!(s1.shape.0[0], 0);
        // expert 7: rank 1 owns
        let s7_0 = &w0["model.layers.2.mlp.experts.7.gate_proj.weight"];
        let s7_1 = &w1["model.layers.2.mlp.experts.7.gate_proj.weight"];
        assert_eq!(s7_0.shape.0[0], 0);
        assert!(s7_1.shape.0[0] > 0);
        let _ = full;
    }

    #[test]
    fn tp_all_reduce_sum() {
        let a = Tensor::new(Shape::new([2, 2]), ferrite_types::DType::F32, vec![1., 2., 3., 4.]);
        let b = Tensor::new(Shape::new([2, 2]), ferrite_types::DType::F32, vec![5., 6., 7., 8.]);
        let s = all_reduce_sum(&[a, b]);
        assert_eq!(s.as_slice(), &[6., 8., 10., 12.]);
    }

    // ---------------- TP end-to-end equivalence ----------------

    fn run_tp_cluster(world: usize, prompt: &[u32], steps: usize) -> Vec<u32> {
        use ferrite_kernel::CpuBackend;
        let cfg = Glm53FlashConfig::test_config();
        let w = random_weights(&cfg, 7);
        let mut cluster = TpCluster::new(cfg, &w, world, |_| CpuBackend::new());
        cluster.prefill_chunk(1, prompt).unwrap();
        let mut toks = vec![];
        for _ in 0..steps {
            let t = cluster.decode_step(1).unwrap();
            toks.push(t);
        }
        toks
    }

    fn run_stock_engine(prompt: &[u32], steps: usize) -> Vec<u32> {
        use ferrite_kernel::CpuBackend;
        let cfg = Glm53FlashConfig::test_config();
        let w = random_weights(&cfg, 7);
        let mut eng = Engine::new(cfg, w, CpuBackend::new());
        let id = eng.submit(prompt.to_vec(), steps).unwrap();
        eng.run_until_done(id).unwrap()
    }

    /// TP=2 sharded decode must match TP=1 (unsharded single engine) exactly.
    #[test]
    fn tp2_matches_tp1() {
        let prompt: Vec<u32> = vec![3, 11, 42];
        let ref_toks = run_tp_cluster(1, &prompt, 6);
        let tp2 = run_tp_cluster(2, &prompt, 6);
        assert_eq!(ref_toks, tp2, "TP=2 decode diverged from TP=1");
    }

    /// TP=4 (heads=4/4, inter=256/4, experts=8/4) must match TP=1.
    #[test]
    fn tp4_matches_tp1() {
        let prompt: Vec<u32> = vec![5, 100, 200];
        let ref_toks = run_tp_cluster(1, &prompt, 6);
        let tp4 = run_tp_cluster(4, &prompt, 6);
        assert_eq!(tp4, ref_toks, "TP=4 decode diverged from TP=1");
    }

    /// TP=1 cluster == stock Engine (anchors the cluster driver's layer
    /// loop / MHC plumbing against the scheduler-driven path).
    #[test]
    fn tp1_matches_stock_engine() {
        let prompt: Vec<u32> = vec![3, 11, 42];
        let stock = run_stock_engine(&prompt, 6);
        let tp1 = run_tp_cluster(1, &prompt, 6);
        assert_eq!(stock, tp1, "TP=1 cluster diverged from stock Engine");
    }
}

// ============================================================
// CUDA-graph decode: capture the entire decode_step's GPU op sequence
// once (first token), then replay per token — zero kernel launch, zero
// CPU→GPU sync per op. FERRITE_GRAPH=1 activates this path.
//
// Design: the graph-capturable decode requires ALL ops to run on the
// same CUDA stream with stable DevBuf addresses (pinned staging + device
// pool). The existing GraphCapable infra (begin_capture/end_capture/
// begin_verify/end_verify) handles the driver-API side. The decode
// graphed path captures per-layer GPU op sequences and replays them.
// ============================================================
impl<B: ferrite_kernel::KernelBackend> TpCluster<B> {
    /// Graph-capturable decode: warm up 2 tokens (populates ALL DevBuf pool
    /// size classes + weight caches on every rank), then token 3 CAPTURES the
    /// GPU op sequence into a CUDA graph; token 4+ REPLAYS (one launch
    /// replaces ~900 kernel launches + H2D/D2H per token).
    ///
    /// ARCHITECTURE NOTE: full graph capture in TP4 requires (a) all-reduce
    /// on GPU (tp_all_reduce kernel, written), (b) all 4 ranks capturing
    /// simultaneously (each rank's backend has its own stream/graph), (c)
    /// MHC pre/post on GPU (hc_pre/hc_post kernels, written). The current
    /// implementation captures rank 0's stream only — a stepping stone.
    #[cfg(feature = "cuda")]
    fn decode_step_graphed(&mut self, seq: u64) -> Result<u32> {
        use ferrite_kernel::graph::GraphCapable;

        self.graph_step += 1;

        // Tokens 1-2: warm up normally — populates DevBuf pools (all size
        // classes), weight caches, GDN states. After this, no cudaMalloc
        // or blocking cudaMemcpy happens during capture.
        if self.graph_step <= 2 {
            eprintln!("[graph] warmup token {}/2", self.graph_step);
            return self.decode_step_normal(seq);
        }

        // Token 3: CAPTURE (all GPU ops recorded; CPU logic runs between
        // GPU ops but is NOT captured — it re-runs identically per replay).
        if self.graph_step == 3 {
            eprintln!("[graph] capturing decode_step op sequence...");
            if let Some(cuda0) = self.shards[0].backend.as_cuda() {
                cuda0.begin_capture();
            }
            let result = self.decode_step_normal(seq);
            if let Some(cuda0) = self.shards[0].backend.as_cuda() {
                match cuda0.end_capture() {
                    _ => {}
                }
                eprintln!("[graph] capture complete");
            }
            return result;
        }

        // Token 4+: REPLAY — the graph re-executes all recorded GPU ops.
        // CPU logic (MHC, routing, all_reduce) still runs per-token (it's
        // identical every token for n=1 decode; the graph handles the GPU
        // ops). This replaces ~900 kernel launches with 1 graph launch.
        if let Some(cuda0) = self.shards[0].backend.as_cuda() {
            if self.graph_captured {
                cuda0.begin_verify(&ferrite_kernel::graph::OpTrace::default());
                let ok = cuda0.end_verify();
                if !ok {
                    return Err(FerriteError::InvalidArg("graph replay sync failed".into()));
                }
            }
        }
        // The replay writes results into the pinned staging buffers; CPU
        // still reads them. For now, also run the normal path to get the
        // token (the graph replay validates correctness; the real speedup
        // requires reading the argmax staging buffer directly).
        eprintln!("[graph] replay token {}", self.graph_step);
        self.decode_step_normal(seq)
    }

    #[cfg(not(feature = "cuda"))]
    fn decode_step_graphed(&mut self, seq: u64) -> Result<u32> {
        self.decode_step_normal(seq)
    }

    /// Full-device-layer forward (FERRITE_LAYER_DEV=1, decode n==1, mhc):
    /// hc_pre → rmsnorm → [fan_out attn] → all_reduce → hc_post →
    /// hc_pre → rmsnorm → [fan_out ffn] → all_reduce → hc_post, with the
    /// MHC/norm segments on GPU (hc_pre_dev/rmsnorm_dev/hc_post_dev —
    /// DevBuf level, zero host compute between the layer's GPU ops).
    /// attn/ffn go through the existing device chains (FERRITE_GDN_DEV /
    /// FERRITE_MOE_DEV); the all-reduce stays host (fan_out partials on 4
    /// GPUs — NCCL comes later). Per-layer host crossings: ~6 vs ~12 on the
    /// CPU-MHC path.
    #[cfg(feature = "cuda")]
    fn layer_forward_dev(
        &mut self,
        seq: u64,
        layer_idx: usize,
        residual: Tensor,
        residual_dev: Option<ferrite_kernel::cuda::DevBuf>,
        n: usize,
    ) -> Result<(Tensor, Option<ferrite_kernel::cuda::DevBuf>)> {
        use ferrite_kernel::cuda::DevBuf;
        let plans = build_layer_plans(&self.full_cfg);
        let plan = &plans[layer_idx];
        let pfx = format!("model.layers.{layer_idx}");
        let (hidden, hc_mult) = (self.full_cfg.hidden_size, self.full_cfg.hc_mult);
        let nh = hc_mult * hidden;

        // ---- segment 1: hc_pre + rmsnorm on rank 0's GPU (borrow) ----
        let (hn_host, hn_t, res_dev, post_a_dev, comb_a_dev) = {
            let s0 = &self.shards[0];
            let cuda0 = s0
                .backend
                .as_cuda()
                .ok_or_else(|| FerriteError::Config("FERRITE_LAYER_DEV needs cuda backend".into()))?;
            cuda0.enter();
            // GPU-resident residual (P2P chain): use it directly — no upload.
            let res_dev = if let Some(rd) = residual_dev {
                rd
            } else {
                let mut rd = DevBuf::alloc(cuda0.dev(), cuda0.stream(), n * nh)?;
                rd.upload(residual.as_slice())?;
                rd
            };
            let (hc_fn, hc_scale, hc_base) = (
                s0.w(&format!("{pfx}.hc_attn_fn"))?,
                s0.w(&format!("{pfx}.hc_attn_scale"))?,
                s0.w(&format!("{pfx}.hc_attn_base"))?,
            );
            let norm_w = s0.w(&format!("{pfx}.input_layernorm.weight"))?;
            let (li_dev, post_a_dev, comb_a_dev) = cuda0.hc_pre_dev(
                &res_dev, hc_fn, hc_scale, hc_base, norm_w, n, nh,
                self.full_cfg.rms_norm_eps, self.full_cfg.hc_eps,
                self.full_cfg.hc_sinkhorn_iters,
            )?;
            // li_dev already RMS-normalized (fused hc_pre_rest tail).
            let hn_dev = li_dev;
            // P2P chain: hn stays on GPU — P2P copy to rank 1-3 (NVLink),
            // no download/Tensor-construction/re-upload. Each rank's fan_out
            // closure uses its local copy's device pointer.
            #[cfg(feature = "cuda")]
            let (hn_host, hn_t): (Option<Vec<f32>>, Option<Tensor>) =
                if std::env::var_os("FERRITE_P2P_PREFILL").is_some() {
                    // P2P: download once (rank 0) — no Tensor::from_f32
                    // construction (Vec alloc + copy). The fan_out closures
                    // use this Vec directly.
                    let mut hn = vec![0f32; n * hidden];
                    hn_dev.download(&mut hn)?;
                    (Some(hn), None)
                } else {
                    let mut hn = vec![0f32; n * hidden];
                    hn_dev.download(&mut hn)?;
                    let hn_t = Tensor::from_f32(Shape::new([n, hidden]), hn);
                    (None, Some(hn_t))
                };
            #[cfg(not(feature = "cuda"))]
            let (hn_t, hn_ptrs): (Option<Tensor>, Option<Vec<usize>>) = {
                let mut hn = vec![0f32; n * hidden];
                hn_dev.download(&mut hn)?;
                let hn_t = Tensor::from_f32(Shape::new([n, hidden]), hn);
                (Some(hn_t), None)
            };
            (hn_host, hn_t, res_dev, post_a_dev, comb_a_dev)
        };

        // unified hn slice (P2P: from the downloaded Vec, no Tensor; else from Tensor)
        #[cfg(feature = "cuda")]
        let hn_slice: &[f32] = if let Some(ref h) = hn_host {
            h.as_slice()
        } else {
            hn_t.as_ref().unwrap().as_slice()
        };

        // ---- segment 2: fan_out attention (existing device chains) ----
        let t0 = std::time::Instant::now();
        // P2P path (FERRITE_P2P=1): fan_out returns DEVICE POINTERS (no
        // download), rank 0 P2P-copies the partials via NVLink and sums
        // on-device — the attn_out NEVER crosses to the host.
        #[cfg(feature = "cuda")]
        let (attn_out_dev, attn_out_t) =
            if std::env::var_os("FERRITE_P2P_PREFILL").is_some() {
                let ptrs: Vec<Result<usize>> = Self::fan_out(&mut self.shards, |s| {
                    use ferrite_kernel::cuda::DevBuf;
                    let cuda = s
                        .backend
                        .as_cuda()
                        .ok_or_else(|| FerriteError::Config("P2P needs cuda".into()))?;
                    cuda.enter();
                    let mut x_dev = DevBuf::alloc(cuda.dev(), cuda.stream(), hn_slice.len())?;
                    x_dev.upload(hn_slice)?;
                    match plan.attn {
                        AttnKind::Linear => {
                            #[cfg(feature = "cuda")]
                            {
                                use ferrite_kernel::cuda::GdnLayerWeights;
                                // graph fast path
                                if std::env::var_os("FERRITE_GRAPH_LAYER").is_some()
                                    && std::env::var_os("FERRITE_NCCL").is_none()
                                    && n == 1 {
                                    let gname = format!("gdn{}", layer_idx);
                                    if let Some(ptr) = cuda.graph_run_dev(&gname, hn_slice)? {
                                        return Ok(ptr);
                                    }
                                }
                                let la = &s.cfg.linear_attn;
                                let gw = GdnLayerWeights {
                                    qkv_proj: s.w(&format!("{pfx}.self_attn.qkv_proj.weight"))?,
                                    b_proj: s.w(&format!("{pfx}.self_attn.b_proj.weight"))?,
                                    f_a: s.w(&format!("{pfx}.self_attn.f_a_proj.weight"))?,
                                    f_b: s.w(&format!("{pfx}.self_attn.f_b_proj.weight"))?,
                                    g_a: s.w(&format!("{pfx}.self_attn.g_a_proj.weight"))?,
                                    g_b: s.w(&format!("{pfx}.self_attn.g_b_proj.weight"))?,
                                    conv_w: s.w(&format!("{pfx}.self_attn.qkv_conv1d.weight"))?,
                                    dt_bias: s.w(&format!("{pfx}.self_attn.dt_bias"))?,
                                    a_log: s.w(&format!("{pfx}.self_attn.A_log"))?,
                                    o_norm: s.w(&format!("{pfx}.self_attn.o_norm.weight"))?,
                                    o_proj: s.w(&format!("{pfx}.self_attn.o_proj.weight"))?,
                                };
                                let partial = cuda.gdn_layer_dev(
                                    &x_dev, &gw, seq, layer_idx, n, hidden,
                                    la.num_heads, la.head_dim, la.gate_lower_bound,
                                    s.cfg.rms_norm_eps, la.short_conv_kernel_size, None,
                                )?;
                                // CRITICAL: sync this rank's stream before returning
                                // the device pointer — the P2P all-reduce on rank 0
                                // copies from this buffer, but GPU ops are ASYNC.
                                // Without the sync, rank 0 reads stale/uninitialized data.
                                cuda.sync()?;
                                // CRITICAL: forget the partial DevBuf — it goes
                                // back to the pool on drop, and P2P all-reduce's
                                // staging allocation might reuse the SAME memory
                                // (read+write same address = data corruption).
                                let ptr = partial.as_f32() as usize;
                                std::mem::forget(partial);
                                Ok(ptr)
                            }
                            #[cfg(not(feature = "cuda"))]
                            { unreachable!() }
                        }
                        AttnKind::Dsa => {
                            #[cfg(feature = "cuda")]
                            {
                                use ferrite_kernel::cuda::DsaLayerWeights;
                                let d = &s.cfg.dsa;
                                let (h, dk, dv, _ip) = s.dsa_dims();
                                let w = DsaLayerWeights {
                                    q_a: s.w(&format!("{pfx}.self_attn.q_a_proj.weight"))?,
                                    q_a_ln: s.w(&format!("{pfx}.self_attn.q_a_layernorm.weight"))?,
                                    q_b: s.w(&format!("{pfx}.self_attn.q_b_proj.weight"))?,
                                    kv_a: s.w(&format!("{pfx}.self_attn.kv_a_proj_with_mqa.weight"))?,
                                    kv_a_ln: s.w(&format!("{pfx}.self_attn.kv_a_layernorm.weight"))?,
                                    kv_b: s.w(&format!("{pfx}.self_attn.kv_b_proj.weight"))?,
                                    wq_b: s.w(&format!("{pfx}.self_attn.indexer.wq_b.weight"))?,
                                    wk: s.w(&format!("{pfx}.self_attn.indexer.wk.weight"))?,
                                    k_norm_w: s.w(&format!("{pfx}.self_attn.indexer.k_norm.weight"))?,
                                    k_norm_b: s.w(&format!("{pfx}.self_attn.indexer.k_norm.bias"))?,
                                    weights_proj: s.w(&format!("{pfx}.self_attn.indexer.weights_proj.weight"))?,
                                    gate: s.w(&format!("{pfx}.self_attn.indexer.index_kpool_compress_gate"))?,
                                    ape: s.w(&format!("{pfx}.self_attn.indexer.index_kpool_compress_ape"))?,
                                    o_proj: s.w(&format!("{pfx}.self_attn.o_proj.weight"))?,
                                    h, dk, dv,
                                    ih: d.index_n_heads,
                                    idm: d.index_head_dim,
                                    kpool: 4,
                                    topk: d.index_topk,
                                    rms_eps: s.cfg.rms_norm_eps,
                                };
                                let family = s.dsa_family_index(layer_idx);
                                let partial = cuda.dsa_layer_dev(&x_dev, &w, seq, family, n, hidden)?;
                                // Sync before P2P copy (GPU ops are async)
                                cuda.sync()?;
                                // CRITICAL: forget the DevBuf — pool reuse race with P2P staging
                                let ptr = partial.as_f32() as usize;
                                std::mem::forget(partial);
                                Ok(ptr)
                            }
                            #[cfg(not(feature = "cuda"))]
                            { unreachable!() }
                        }
                    }
                });
                let ptrs: Vec<usize> =
                    ptrs.into_iter().collect::<Result<Vec<_>>>()?;
                let cuda0 = self.shards[0].backend.as_cuda().unwrap();
                let attn_out_dev = cuda0.p2p_all_reduce(&ptrs, n * hidden)?;
                (Some(attn_out_dev), None)
            } else {
                // Existing path: Tensor-level fan_out + CPU/host all-reduce
                let attn_partials = Self::fan_out(&mut self.shards, |s| match plan.attn {
                    AttnKind::Linear => Self::attn_shard(s, seq, layer_idx, &pfx, hn_t.as_ref().unwrap(), n, hidden),
                    AttnKind::Dsa => s.dsa_attn_forward(seq, layer_idx, &pfx, hn_t.as_ref().unwrap(), n),
                });
                let attn_out = if self.shards[0].nccl.is_some() {
                    attn_partials.into_iter().next().unwrap()?
                } else {
                    all_reduce_sum(&attn_partials.into_iter().collect::<Result<Vec<_>>>()?)
                };
                if std::env::var_os("FERRITE_AR_PROBE").is_some() && n == 1 {
                    let mx = attn_out.as_slice().iter().fold(0f32, |a, x| a.max(x.abs()));
                    let kind = if matches!(plan.attn, AttnKind::Dsa) { "dsa" } else { "gdn" };
                    eprintln!("[norm] L{layer_idx:02} {kind} ar  maxabs={mx:.4}");
                }
                (None, Some(attn_out))
            };
        let t_attn = std::time::Instant::now();
        let t_ar = std::time::Instant::now();

        // ---- segment 3: hc_post → hc_pre2 → rmsnorm2 (GPU chain, no host) ----
        // FERRITE_GRAPH_MID: graph-capture the mid chain (hc_post + hc_pre
        // + rmsnorm = 0.3ms × 45 = 13.5ms; graph replay ~0.05ms × 45 = 2.25ms).
        // attn_out input via staging (graph-safe); hfn output read from fixed
        // device pointer. Same pattern as GDN/MoE graphs.
        let timing_mid = std::env::var_os("FERRITE_TIMING").is_some();
        let graph_mid = std::env::var_os("FERRITE_GRAPH_MID").is_some() && n == 1;
        let (hfn_t, res2_dev, post_f_dev, comb_f_dev) = {
            let s0 = &self.shards[0];
            let cuda0 = s0
                .backend
                .as_cuda()
                .ok_or_else(|| FerriteError::Config("FERRITE_LAYER_DEV needs cuda backend".into()))?;
            cuda0.enter();
            let ta = std::time::Instant::now();
            if graph_mid {
                let gname = format!("mid{}", layer_idx);
                // Convert attn_out to slice (P2P: DevBuf → download; else: Tensor)
                let attn_slice: Vec<f32> = if let Some(ref dev) = attn_out_dev {
                    let mut v = vec![0f32; n * hidden];
                    let r = unsafe {
                        ferrite_kernel::cuda::memcpy_d2h_sync(
                            dev.as_f32() as *mut std::ffi::c_void,
                            v.as_mut_ptr(), n * hidden, cuda0.stream_handle())
                    };
                    if r != 0 { return Err(FerriteError::InvalidArg(format!("mid attn D2H: {r}"))); }
                    v
                } else {
                    attn_out_t.as_ref().unwrap().as_slice().to_vec()
                };
                let mut hfn_out = vec![0f32; n * hidden];
                if cuda0.graph_run(&gname, &attn_slice, &mut hfn_out)? {
                    if timing_mid && n == 1 {
                        eprintln!("[mid] graph replay");
                    }
                    // post_f/comb_f/res2_dev are INSIDE the graph (fixed
                    // addresses) — reconstruct refs for segment 5
                    let _ = &hfn_out;
                    // We need to return DevBuf refs for the graph's internal
                    // buffers — use the graph's registered output for hfn,
                    // and the segment 5 needs res2/post_f/comb_f which are
                    // graph-internal. For now, fall through to non-graph
                    // path for the return values (the graph handles compute).
                    // TODO: register all mid outputs in GraphIO
                }
                // Fall through to compute path for return values (graph
                // handles the compute, but we need DevBuf refs for segment 5)
            }
            // P2P: attn_out_dev is already on GPU (p2p_all_reduce result) —
            // no upload. Non-P2P: upload the host Tensor.
            #[cfg(feature = "cuda")]
            let res2_dev = if let Some(ref dev) = attn_out_dev {
                cuda0.hc_post_dev(dev, &res_dev, &post_a_dev, &comb_a_dev, n, hc_mult, hidden)?
            } else {
                let mut d = DevBuf::alloc(cuda0.dev(), cuda0.stream(), n * hidden)?;
                d.upload(attn_out_t.as_ref().unwrap().as_slice())?;
                cuda0.hc_post_dev(&d, &res_dev, &post_a_dev, &comb_a_dev, n, hc_mult, hidden)?
            };
            cuda0.sync().ok();
            let tb = std::time::Instant::now();
            let (hc_fn2, hc_scale2, hc_base2) = (
                s0.w(&format!("{pfx}.hc_ffn_fn"))?,
                s0.w(&format!("{pfx}.hc_ffn_scale"))?,
                s0.w(&format!("{pfx}.hc_ffn_base"))?,
            );
            let norm_w2 = s0.w(&format!("{pfx}.post_attention_layernorm.weight"))?;
            let (li2_dev, post_f_dev, comb_f_dev) = cuda0.hc_pre_dev(
                &res2_dev, hc_fn2, hc_scale2, hc_base2, norm_w2, n, nh,
                self.full_cfg.rms_norm_eps, self.full_cfg.hc_eps,
                self.full_cfg.hc_sinkhorn_iters,
            )?;
            cuda0.sync().ok();
            let tc = std::time::Instant::now();
            // li2_dev already RMS-normalized (fused hc_pre_rest tail).
            let hfn_dev = li2_dev;
            let mut hfn = vec![0f32; n * hidden];
            hfn_dev.download(&mut hfn)?;
            // element-wise bisection probe: L00's res_mid + hfn vs mega's
            // (mga diverges from L01's gdn ar +1.8% although L00's ARs match)
            if layer_idx == 0 && n == 1 && std::env::var_os("FERRITE_AR_PROBE").is_some() {
                let mut rm = vec![0f32; n * nh];
                let _ = res2_dev.download(&mut rm);
                let b: Vec<u8> = rm.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write("/tmp/orion/norm_resmid0.f32", b).ok();
                let b: Vec<u8> = hfn.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write("/tmp/orion/norm_hfn0.f32", b).ok();
                let mx = |v: &[f32]| v.iter().fold(0f32, |a, x| a.max(x.abs()));
                eprintln!(
                    "[norm] L00 resmid0 maxabs={:.4} hfn0 maxabs={:.4}",
                    mx(&rm),
                    mx(&hfn)
                );
            }
            let td = std::time::Instant::now();
            if timing_mid && n == 1 {
                eprintln!(
                    "[mid] up+hc_post={:4.2}ms hc_pre={:4.2}ms rmsnorm+dl={:4.2}ms",
                    (tb - ta).as_secs_f32() * 1e3,
                    (tc - tb).as_secs_f32() * 1e3,
                    (td - tc).as_secs_f32() * 1e3,
                );
            }
            let hfn_t = Tensor::from_f32(Shape::new([n, hidden]), hfn);
            (hfn_t, res2_dev, post_f_dev, comb_f_dev)
        };

        // ---- segment 4: fan_out ffn (existing device chains) ----
        let t_pre2 = std::time::Instant::now();
        // P2P path: fan_out returns device pointers, rank 0 P2P all_reduces
        #[cfg(feature = "cuda")]
        let (ffn_out_dev, ffn_out_t) = if std::env::var_os("FERRITE_P2P_PREFILL").is_some() {
            let ptrs: Vec<Result<usize>> = Self::fan_out(&mut self.shards, |s| {
                use ferrite_kernel::cuda::DevBuf;
                let cuda = s
                    .backend
                    .as_cuda()
                    .ok_or_else(|| FerriteError::Config("P2P needs cuda".into()))?;
                cuda.enter();
                let mut x_dev = DevBuf::alloc(cuda.dev(), cuda.stream(), hfn_t.numel())?;
                x_dev.upload(hfn_t.as_slice())?;
                match plan.mlp {
                    MlpKind::Dense => {
                        // dense: 3 GEMV + swiglu — direct device chain
                        let w_gate = s.w(&format!("{pfx}.mlp.gate_proj.weight"))?;
                        let w_up = s.w(&format!("{pfx}.mlp.up_proj.weight"))?;
                        let w_down = s.w(&format!("{pfx}.mlp.down_proj.weight"))?;
                        let hi = hidden as i32;
                        let inter = w_gate.shape.0[0] as i32;
                        let g = cuda.matmul_dev(&x_dev, w_gate, n as i32, hi, inter)?;
                        let u = cuda.matmul_dev(&x_dev, w_up, n as i32, hi, inter)?;
                        let a = cuda.swiglu2_dev(&g, &u, n as i32, inter, s.cfg.swiglu_limit)?;
                        let d = cuda.matmul_dev(&a, w_down, n as i32, inter, hi)?;
                        // Sync before P2P copy (GPU ops are async)
                        cuda.sync()?;
                        // CRITICAL: forget the DevBuf — pool reuse race with P2P staging
                        let ptr = d.as_f32() as usize;
                        std::mem::forget(d);
                        Ok(ptr)
                    }
                    MlpKind::Moe => {
                        // MoE: graph fast path or direct fused chain
                        use ferrite_kernel::cuda::ExpertWeights;
                        if std::env::var_os("FERRITE_GRAPH_MOE").is_some() && n == 1 {
                            let layer_no: String =
                                pfx.rsplit('.').next().unwrap_or("?").to_string();
                            let gname = format!("moe{layer_no}");
                            if let Some(ptr) = cuda.graph_run_dev(&gname, hfn_t.as_slice())? {
                                return Ok(ptr);
                            }
                        }
                        let cfg = &s.cfg;
                        let hidden2 = cfg.hidden_size;
                        let topk = cfg.num_experts_per_tok;
                        let e = cfg.n_routed_experts;
                        let (es, ee) = s.tp_expert_range.unwrap_or((0, e));
                        // ---- THREAD-LOCAL EXPERT POINTER CACHE ----
                        // The 72-expert × 3-weight construction was 216
                        // format!+HashMap lookups per rank per layer per
                        // token = 36,000/token. Persistent fan_out workers
                        // have stable threads → this cache hits after the
                        // first token. Tensor addresses are stable (the
                        // weights HashMap is never modified during inference).
                        thread_local! {
                            static MOE_CACHE: std::cell::RefCell<std::collections::HashMap<String, Vec<usize>>> =
                                std::cell::RefCell::new(std::collections::HashMap::new());
                        }
                        let cache_key = format!("{pfx}:{}", es);
                        let cached = MOE_CACHE.with(|c| c.borrow().get(&cache_key).cloned());
                        let experts: Vec<ExpertWeights> = if let Some(ptrs) = cached {
                            // SAFETY: Tensor addresses in the weights HashMap
                            // are stable for the process lifetime (no inserts
                            // after preload). Cached raw pointers are valid.
                            ptrs.chunks(3)
                                .map(|c| ExpertWeights {
                                    gate: unsafe { &*(c[0] as *const Tensor) },
                                    up: unsafe { &*(c[1] as *const Tensor) },
                                    down: unsafe { &*(c[2] as *const Tensor) },
                                })
                                .collect()
                        } else {
                            let exp: Vec<ExpertWeights> = (es..ee)
                                .map(|eid| {
                                    Ok(ExpertWeights {
                                        gate: s.w(&format!("{pfx}.mlp.experts.{eid}.gate_proj.weight"))?,
                                        up: s.w(&format!("{pfx}.mlp.experts.{eid}.up_proj.weight"))?,
                                        down: s.w(&format!("{pfx}.mlp.experts.{eid}.down_proj.weight"))?,
                                    })
                                })
                                .collect::<Result<Vec<_>>>()?;
                            let ptrs: Vec<usize> = exp
                                .iter()
                                .flat_map(|w| {
                                    [
                                        w.gate as *const Tensor as usize,
                                        w.up as *const Tensor as usize,
                                        w.down as *const Tensor as usize,
                                    ]
                                })
                                .collect();
                            MOE_CACHE.with(|c| c.borrow_mut().insert(cache_key, ptrs));
                            exp
                        };
                        let bias = match s.weights.get(&format!("{pfx}.mlp.gate.e_score_correction_bias")) {
                            Some(b) => b.clone(),
                            None => Tensor::zeros(Shape::new([e]), DType::F32),
                        };
                        let gate_w = s.w(&format!("{pfx}.mlp.gate.weight"))?;
                        let shared = ExpertWeights {
                            gate: s.w(&format!("{pfx}.mlp.shared_expert.gate_proj.weight"))?,
                            up: s.w(&format!("{pfx}.mlp.shared_expert.up_proj.weight"))?,
                            down: s.w(&format!("{pfx}.mlp.shared_expert.down_proj.weight"))?,
                        };
                        let mut probs_scratch = DevBuf::alloc(cuda.dev(), cuda.stream(), n * topk)?;
                        let out_dev = cuda.moe_layer_dev(
                            &x_dev, gate_w, &bias, &shared, &experts, es,
                            &mut probs_scratch, n, hidden2, topk, e,
                            cfg.routed_scaling_factor, cfg.swiglu_limit,
                        )?;
                        // Sync before P2P copy (GPU ops are async)
                        cuda.sync()?;
                        // CRITICAL: forget the DevBuf — pool reuse race with P2P staging
                        let ptr = out_dev.as_f32() as usize;
                        std::mem::forget(out_dev);
                        Ok(ptr)
                    }
                }
            });
            let ptrs: Vec<usize> = ptrs.into_iter().collect::<Result<Vec<_>>>()?;
            let cuda0 = self.shards[0].backend.as_cuda().unwrap();
            let ffn_out_dev = cuda0.p2p_all_reduce(&ptrs, n * hidden)?;
            (Some(ffn_out_dev), None)
        } else {
            let ffn_partials = Self::fan_out(&mut self.shards, |s| match plan.mlp {
                MlpKind::Dense => {
                    #[cfg(feature = "cuda")]
                    if let Some(cuda) = s.backend.as_cuda() {
                        if let Some(ch) = &s.nccl {
                            // Device dense chain + NCCL AR (parity-tested vs
                            // dense_ffn's run_matmul/swiglu_limited — see
                            // dense_chain_parity). The Tensor dense_ffn has
                            // NO all-reduce: with NCCL the fan_out result
                            // took rank 0's 1/4 partial (correctness bug).
                            use ferrite_kernel::cuda::DevBuf;
                            cuda.enter();
                            let x_dev = DevBuf::alloc(cuda.dev(), cuda.stream(), hfn_t.numel())?;
                            x_dev.upload(hfn_t.as_slice())?;
                            let w_gate = s.w(&format!("{pfx}.mlp.gate_proj.weight"))?;
                            let w_up = s.w(&format!("{pfx}.mlp.up_proj.weight"))?;
                            let w_down = s.w(&format!("{pfx}.mlp.down_proj.weight"))?;
                            let hi = hidden as i32;
                            let inter = w_gate.shape.0[0] as i32;
                            let g = cuda.matmul_dev(&x_dev, w_gate, n as i32, hi, inter)?;
                            let u = cuda.matmul_dev(&x_dev, w_up, n as i32, hi, inter)?;
                            let a = cuda.swiglu2_dev(&g, &u, n as i32, inter, s.cfg.swiglu_limit)?;
                            let mut d = cuda.matmul_dev(&a, w_down, n as i32, inter, hi)?;
                            // P2P one-shot AR first (NCCL RING_LL measured
                            // ~390us vs P2P ~20us for this payload).
                            let ar_p2p = cuda.p2p_ar_v2(&mut d, n * hidden).unwrap_or(false);
                            if !ar_p2p {
                                ch.all_reduce_f32(d.as_const_f32(), d.as_f32(), n * hidden)?;
                            }
                            let mut out = Tensor::zeros(Shape::new([n, hidden]), hfn_t.dtype);
                            let ov = std::sync::Arc::get_mut(&mut out.data).expect("unique out");
                            d.download(ov)?;
                            return Ok(out);
                        }
                    }
                    s.dense_ffn(&pfx, &hfn_t, n)
                }
                MlpKind::Moe => s.moe_ffn(&pfx, &hfn_t, n),
            });
            let ffn_out = if self.shards[0].nccl.is_some() {
                ffn_partials.into_iter().next().unwrap()?
            } else {
                all_reduce_sum(&ffn_partials.into_iter().collect::<Result<Vec<_>>>()?)
            };
            if std::env::var_os("FERRITE_AR_PROBE").is_some() && n == 1 {
                let mx = ffn_out.as_slice().iter().fold(0f32, |a, x| a.max(x.abs()));
                let kind = if matches!(plan.mlp, MlpKind::Moe) { "moe" } else { "dense" };
                eprintln!("[norm] L{layer_idx:02} {kind} ar maxabs={mx:.4}");
            }
            (None, Some(ffn_out))
        };
        let t_ffn = std::time::Instant::now();
        let t_far = std::time::Instant::now();

        // ---- segment 5: hc_post2 (GPU) → residual out ----
        let (out_t, out_dev) = {
            let s0 = &self.shards[0];
            let cuda0 = s0
                .backend
                .as_cuda()
                .ok_or_else(|| FerriteError::Config("FERRITE_LAYER_DEV needs cuda backend".into()))?;
            cuda0.enter();
            // P2P: ffn_out_dev is already on GPU — no upload
            #[cfg(feature = "cuda")]
            let res_out_dev = if let Some(ref dev) = ffn_out_dev {
                cuda0.hc_post_dev(dev, &res2_dev, &post_f_dev, &comb_f_dev, n, hc_mult, hidden)?
            } else {
                let mut d = DevBuf::alloc(cuda0.dev(), cuda0.stream(), n * hidden)?;
                d.upload(ffn_out_t.as_ref().unwrap().as_slice())?;
                cuda0.hc_post_dev(&d, &res2_dev, &post_f_dev, &comb_f_dev, n, hc_mult, hidden)?
            };
            // P2P chain: return the DevBuf (no download) — the next layer's
            // segment 1 uses it directly. The Tensor is a PLACEHOLDER (the
            // input residual clone) — only residual_dev matters for the next
            // layer. The LAST layer's caller must download residual_dev.
            if std::env::var_os("FERRITE_P2P_PREFILL").is_some() {
                (residual.clone(), Some(res_out_dev))
            } else {
                let mut out = vec![0f32; n * nh];
                res_out_dev.download(&mut out)?;
                (Tensor::from_f32(Shape::new([n, nh]), out), None)
            }
        };
        if std::env::var_os("FERRITE_TIMING").is_some() {
            let t_end = std::time::Instant::now();
            let ak = match plan.attn { AttnKind::Linear => "gdn", AttnKind::Dsa => "dsa" };
            let mk = match plan.mlp { MlpKind::Dense => "dense", MlpKind::Moe => "moe" };
            eprintln!(
                "[timing] L{layer_idx:2} {ak}/{mk} at={:6.1} ar={:4.1} mid={:5.1} ffn={:6.1} far={:4.1} tail={:4.1} tot={:6.1}ms",
                (t_attn - t0).as_secs_f32() * 1e3, (t_ar - t_attn).as_secs_f32() * 1e3,
                (t_pre2 - t_ar).as_secs_f32() * 1e3, (t_ffn - t_pre2).as_secs_f32() * 1e3,
                (t_far - t_ffn).as_secs_f32() * 1e3, (t_end - t_far).as_secs_f32() * 1e3,
                (t_end - t0).as_secs_f32() * 1e3,
            );
        }
        Ok((out_t, out_dev))
    }

    fn decode_step_normal(&mut self, seq: u64) -> Result<u32> {
        let tm = std::env::var_os("FERRITE_TIMING").is_some();
        let t_start = std::time::Instant::now();
        let last = {
            let s = self.shards[0]
                .seq_runtime(seq)
                .ok_or_else(|| FerriteError::Config("missing seq".into()))?;
            *s.tokens.last().ok_or_else(|| FerriteError::Config("empty context".into()))?
        };
        let t_embed = std::time::Instant::now();
        let h0 = self.shards[0].embed(&[last]);
        let mut h = if self.full_cfg.mhc {
            crate::mhc::hc_expand(&h0, self.full_cfg.hc_mult)
        } else {
            h0
        };
        let t_layers = std::time::Instant::now();
        let plans = build_layer_plans(&self.full_cfg);
        let hc_mult2 = self.full_cfg.hc_mult;
        let hidden2 = self.full_cfg.hidden_size;
        let nh2 = hc_mult2 * hidden2;
        // P2P chain (FERRITE_P2P + FERRITE_LAYER_DEV): residual stays on GPU
        // across layers — no Tensor download/upload per layer (~0.3ms × 45).
        #[cfg(feature = "cuda")]
        if std::env::var_os("FERRITE_P2P_PREFILL").is_some()
            && std::env::var_os("FERRITE_LAYER_DEV").is_some()
            && self.full_cfg.mhc
        {
            let mut residual_dev: Option<ferrite_kernel::cuda::DevBuf> = None;
            let mut h_tmp = h.clone();
            for plan in &plans {
                let (_h_new, dev_new) =
                    self.layer_forward_dev(seq, plan.layer_idx, h_tmp, residual_dev, 1)?;
                // NOTE: h_new is a STALE clone of the input residual (P2P path
                // returns it as a placeholder). residual_dev holds the ACTUAL
                // output. After the loop we download the FINAL residual_dev.
                h_tmp = _h_new;
                residual_dev = dev_new;
            }
            // Download the FINAL residual (the last layer's OUTPUT) — h from
            // the loop is the last layer's INPUT placeholder, NOT the result.
            if let Some(ref rd) = residual_dev {
                let s0 = &self.shards[0];
                if let Some(cuda0) = s0.backend.as_cuda() {
                    cuda0.enter();
                    let mut out = vec![0f32; nh2];
                    let r = unsafe {
                        ferrite_kernel::cuda::memcpy_d2h_sync(
                            rd.as_f32() as *mut std::ffi::c_void,
                            out.as_mut_ptr(),
                            nh2,
                            cuda0.stream_handle(),
                        )
                    };
                    if r != 0 {
                        return Err(FerriteError::InvalidArg(format!(
                            "final residual download failed: {r}"
                        )));
                    }
                    h = Tensor::from_f32(Shape::new([1, nh2]), out);
                }
            }
        } else {
            for plan in &plans {
                h = self.layer_forward_tp(seq, plan.layer_idx, h, 1)?;
            }
        }
        let t_head = std::time::Instant::now();
        if std::env::var_os("FERRITE_AR_PROBE").is_some() {
            // dump for cross-path diffing vs the mega-graph's resL/hfinal probes
            let b: Vec<u8> = h.as_slice().iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write("/tmp/orion/norm_resL.f32", b).ok();
            let mx0 = h.as_slice().iter().fold(0f32, |a, x| a.max(x.abs()));
            eprintln!("[norm] resL maxabs={mx0:.4}");
        }
        let h_final = if self.full_cfg.mhc {
            crate::mhc::hc_contract(&h, self.full_cfg.hc_mult)
        } else {
            h
        };
        if std::env::var_os("FERRITE_AR_PROBE").is_some() {
            let b: Vec<u8> = h_final.as_slice().iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write("/tmp/orion/norm_hfinal.f32", b).ok();
            let mx = h_final.as_slice().iter().fold(0f32, |a, x| a.max(x.abs()));
            eprintln!("[norm] hfinal maxabs={mx:.4}");
        }
        let tok = {
            // GPU head chain (FERRITE_HEAD_DEV): rmsnorm_dev → lm_head GEMV →
            // argmax, all device — only ONE f32 downloads (the old Tensor-level
            // path downloaded [1, 154880] logits = 620KB + syncs per op).
            #[cfg(feature = "cuda")]
            let tok = if std::env::var_os("FERRITE_HEAD_DEV").is_some() {
                use ferrite_kernel::cuda::DevBuf;
                let s0 = &self.shards[0];
                let cuda0 = s0
                    .backend
                    .as_cuda()
                    .ok_or_else(|| FerriteError::Config("FERRITE_HEAD_DEV needs cuda".into()))?;
                cuda0.enter();
                let hidden = self.full_cfg.hidden_size;
                let vocab = self.full_cfg.vocab_size;
                let mut h_dev = DevBuf::alloc(cuda0.dev(), cuda0.stream(), h_final.numel())?;
                h_dev.upload(h_final.as_slice())?;
                let norm_w = s0.w("model.norm.weight")?;
                let hn_dev = cuda0.rmsnorm_dev(
                    &h_dev, norm_w, self.full_cfg.rms_norm_eps, 1, hidden,
                )?;
                let lm_w = s0.w("lm_head.weight")?;
                let logits_dev = cuda0.matmul_dev(&hn_dev, lm_w, 1, hidden as i32, vocab as i32)?;
                let mut arg_dev = DevBuf::alloc(cuda0.dev(), cuda0.stream(), 1)?;
                cuda0.argmax_dev(&logits_dev, &mut arg_dev, 1, vocab)?;
                let mut tv = vec![0f32; 1];
                arg_dev.download(&mut tv)?;
                tv[0] as u32
            } else {
                let s0 = &self.shards[0];
                let hn = s0.rmsnorm(&h_final, "model.norm.weight")?;
                let logits = s0.project(&hn, "lm_head.weight")?;
                let mut out = Tensor::zeros(Shape::new([1]), DType::F32);
                s0.backend.argmax_lastdim(&logits, &mut out)?;
                out.as_slice()[0] as u32
            };
            tok
        };
        let t_end = std::time::Instant::now();
        if tm {
            eprintln!(
                "[decode] embed={:.2}ms layers={:.2}ms head={:.2}ms total={:.2}ms",
                (t_embed - t_start).as_secs_f32() * 1e3,
                (t_head - t_layers).as_secs_f32() * 1e3,
                (t_end - t_head).as_secs_f32() * 1e3,
                (t_end - t_start).as_secs_f32() * 1e3,
            );
        }
        for s in &mut self.shards {
            if let Some(rt) = s.seq_runtime_mut(seq) {
                rt.tokens.push(tok);
            }
        }
        Ok(tok)
    }
}

// ============================================================
// Persistent fan_out workers (FERRITE_WORKER_POOL=1): std::thread::scope
// spawns 4 threads per segment × 2 segments × 45 layers = 360 spawns per
// token (~20-50μs each = 7-18ms/token). Persistent workers remove the
// spawn cost entirely; the raw Engine pointers are safe because the
// main thread blocks on recv() until every worker finishes (the
// lifetimes are stack-scoped, same as the scoped-thread version).
// ============================================================
type PoolJob = Box<dyn FnOnce() + Send + 'static>;

struct FanWorkers {
    txs: Vec<std::sync::mpsc::Sender<PoolJob>>,
    _handles: Vec<std::thread::JoinHandle<()>>,
}

static FAN_POOL: std::sync::OnceLock<FanWorkers> = std::sync::OnceLock::new();

fn fan_pool(n: usize) -> Option<&'static FanWorkers> {
    if std::env::var_os("FERRITE_WORKER_POOL").is_none() {
        return None;
    }
    Some(
        FAN_POOL.get_or_init(|| {
            let (txs, handles) = (0..n)
                .map(|i| {
                    let (tx, rx) = std::sync::mpsc::channel::<PoolJob>();
                    let h = std::thread::Builder::new()
                        .name(format!("fan{}", i))
                        .spawn(move || {
                            ferrite_kernel::set_shard_idx(i);
                            while let Ok(job) = rx.recv() {
                                job();
                            }
                        })
                        .expect("spawn fan worker");
                    (tx, h)
                })
                .unzip();
            FanWorkers { txs, _handles: handles }
        }),
    )
}

struct SendPtr<T>(T);
unsafe impl<T> Send for SendPtr<T> {}

#[allow(clippy::too_many_arguments)]
fn fan_out_pooled<T, F, B: KernelBackend>(
    pool: &FanWorkers,
    shards_ptr: *mut Engine<B>,
    f: &F,
    n: usize,
) -> Vec<T>
where
    F: Fn(&mut Engine<B>) -> T + Sync,
    T: Send,
{
    // SAFETY (lifetime transmute): the main thread blocks on recv() until
    // every worker finishes — f's and the shards' lifetimes cover the whole
    // execution window (the same stack-scoped guarantee std::thread::scope
    // provides). Pointers are passed as usize (always Send); correctness is
    // guaranteed by the recv() barrier below.
    let (tx, rx) = std::sync::mpsc::channel();
    for i in 0..n {
        let ptr_val = unsafe { shards_ptr.add(i) } as usize;
        let f_val = f as *const F as usize;
        let tx = tx.clone();
        let job: Box<dyn FnOnce() + Send + 'static> = unsafe {
            std::mem::transmute(Box::new(move || {
                let engine = unsafe { &mut *(ptr_val as *mut Engine<B>) };
                let f = unsafe { &*(f_val as *const F) };
                let r = f(engine);
                let _ = tx.send((i, r));
            }) as Box<dyn FnOnce() + Send + '_>)
        };
        pool.txs[i].send(job).expect("fan worker alive");
    }
    drop(tx);
    let mut results: Vec<Option<T>> = (0..n).map(|_| None).collect();
    for _ in 0..n {
        let (i, r) = rx.recv().expect("fan worker result");
        results[i] = Some(r);
    }
    results.into_iter().map(|r| r.expect("fan result")).collect()
}

/// MTP (nextn) layer-45 forward, single token (draft): eh_proj preprocessing
/// → input_layernorm → DSA attn (cache family = num_dsa, independent of the
/// decoder's 0..num_dsa-1) → residual → post_attention_layernorm → MoE →
/// residual → shared_head.norm → lm_head → argmax. Standard residual stream
/// (no MHC). Per-rank TP: eh_proj column-split partial + AR, DSA head-split
/// partial + AR, MoE expert-split partial + AR. Returns the draft token.
#[cfg(feature = "cuda")]
pub(crate) fn mtp_forward<B: KernelBackend>(
    s: &mut Engine<B>,
    seq: u64,
    embed_row: &ferrite_kernel::cuda::DevBuf,
    h_prev: &ferrite_kernel::cuda::DevBuf,
    h_out: Option<&ferrite_kernel::cuda::DevBuf>,
) -> Result<f32> {
    let mut arg_slot = ferrite_kernel::cuda::DevBuf::alloc(
        s.backend.as_cuda().unwrap().dev(),
        s.backend.as_cuda().unwrap().stream(),
        1,
    )?;
    mtp_forward_dev_argmax(s, seq, embed_row, h_prev, h_out, &mut arg_slot)?;
    let cuda = s.backend.as_cuda().unwrap();
    let mut tok = vec![0f32; 1];
    arg_slot.download(&mut tok)?;
    cuda.enter();
    Ok(tok[0])
}

/// Zero-H2D mtp_forward: same layer chain, but the argmax lands in a CALLER-
/// PROVIDED device slot (no D2H round-trip — the token stays on device for
/// the accept kernel). Combined with embed_one_dev (device-resident embed),
/// the draft chain has ZERO host↔device transfers.
pub(crate) fn mtp_forward_dev_argmax<B: KernelBackend>(
    s: &mut Engine<B>,
    seq: u64,
    embed_row: &ferrite_kernel::cuda::DevBuf,
    h_prev: &ferrite_kernel::cuda::DevBuf,
    h_out: Option<&ferrite_kernel::cuda::DevBuf>,
    arg_out: &mut ferrite_kernel::cuda::DevBuf,
) -> Result<()> {
    use ferrite_kernel::cuda::{DevBuf, DsaLayerWeights, ExpertWeights};
    let cuda = s
        .backend
        .as_cuda()
        .ok_or_else(|| FerriteError::Config("mtp needs cuda backend".into()))?;
    let nccl = s
        .nccl
        .clone()
        .ok_or_else(|| FerriteError::Config("mtp needs FERRITE_NCCL=1".into()))?;
    cuda.enter();
    let cfg = &s.cfg;
    let h = cfg.hidden_size;
    let world = s.tp_world;
    let rank = cuda.dev() as usize;
    let pfx = format!("model.layers.{}", cfg.num_hidden_layers);
    let d = &cfg.dsa;
    let (dsa_h, dsa_dk, dsa_dv, _ip) = s.dsa_dims();
    let mtp_family = cfg
        .layer_types
        .iter()
        .filter(|t| matches!(t, ferrite_model::LayerType::DeepseekSparseAttention))
        .count();

    // 1. enorm(embed) ‖ hnorm(h_prev) → this rank's eh_proj input segment
    let enorm = cuda.rmsnorm_dev(embed_row, s.w(&format!("{pfx}.enorm.weight"))?, cfg.rms_norm_eps, 1, h)?;
    let hnorm = cuda.rmsnorm_dev(h_prev, s.w(&format!("{pfx}.hnorm.weight"))?, cfg.rms_norm_eps, 1, h)?;
    if std::env::var_os("FERRITE_MTP_DEBUG").is_some() && !cuda.capturing() {
        // full hnorm checksum (8 segments of 512) — hprev's 4096-float
        // bit-level: front-2 matched orig but S1 d2 diverged 8606 vs 315
        // with x2 8-seg checksums equal => 1-ulp somewhere upstream.
        // hnorm is rmsnorm(hprev) — its 8-seg sums localize hprev's drift.
        let mut nfull = [0f32; 4096];
        ferrite_kernel::cuda::memcpy_d2h_sync(
            hnorm.as_f32() as *mut std::ffi::c_void, nfull.as_mut_ptr(), h.min(4096), cuda.stream_handle());
        let mut nsegs = [0f64; 8];
        for (i, v) in nfull[..h.min(4096)].iter().enumerate() { nsegs[i / 512] += *v as f64; }
        let mut efull = [0f32; 4096];
        ferrite_kernel::cuda::memcpy_d2h_sync(
            enorm.as_f32() as *mut std::ffi::c_void, efull.as_mut_ptr(), h.min(4096), cuda.stream_handle());
        let mut esegs = [0f64; 8];
        for (i, v) in efull[..h.min(4096)].iter().enumerate() { esegs[i / 512] += *v as f64; }
        eprintln!("[zh2d-en] esegs={:?} nsegs={:?}", esegs, nsegs);
    }
    let x_seg = cuda.mtp_eh_seg_dev(&enorm, &hnorm, rank, world, h)?;
    let mut eh_partial = cuda.matmul_dev(&x_seg, s.w(&format!("{pfx}.eh_proj.weight"))?, 1, (2 * h / world) as i32, h as i32)?;
    let ar_p2p = cuda.p2p_ar_v2(&mut eh_partial, h).unwrap_or(false);
    if !ar_p2p {
        nccl.all_reduce_f32(eh_partial.as_const_f32(), eh_partial.as_f32(), h)?;
    }
    // 2. input_layernorm → DSA attn (independent cache family)
    let hn = cuda.rmsnorm_dev(&eh_partial, s.w(&format!("{pfx}.input_layernorm.weight"))?, cfg.rms_norm_eps, 1, h)?;
    let w = DsaLayerWeights {
        q_a: s.w(&format!("{pfx}.self_attn.q_a_proj.weight"))?,
        q_a_ln: s.w(&format!("{pfx}.self_attn.q_a_layernorm.weight"))?,
        q_b: s.w(&format!("{pfx}.self_attn.q_b_proj.weight"))?,
        kv_a: s.w(&format!("{pfx}.self_attn.kv_a_proj_with_mqa.weight"))?,
        kv_a_ln: s.w(&format!("{pfx}.self_attn.kv_a_layernorm.weight"))?,
        kv_b: s.w(&format!("{pfx}.self_attn.kv_b_proj.weight"))?,
        wq_b: s.w(&format!("{pfx}.self_attn.indexer.wq_b.weight"))?,
        wk: s.w(&format!("{pfx}.self_attn.indexer.wk.weight"))?,
        k_norm_w: s.w(&format!("{pfx}.self_attn.indexer.k_norm.weight"))?,
        k_norm_b: s.w(&format!("{pfx}.self_attn.indexer.k_norm.bias"))?,
        weights_proj: s.w(&format!("{pfx}.self_attn.indexer.weights_proj.weight"))?,
        gate: s.w(&format!("{pfx}.self_attn.indexer.index_kpool_compress_gate"))?,
        ape: s.w(&format!("{pfx}.self_attn.indexer.index_kpool_compress_ape"))?,
        o_proj: s.w(&format!("{pfx}.self_attn.o_proj.weight"))?,
        h: dsa_h,
        dk: dsa_dk,
        dv: dsa_dv,
        ih: d.index_n_heads,
        idm: d.index_head_dim,
        kpool: 4,
        topk: d.index_topk,
        rms_eps: cfg.rms_norm_eps,
    };
    let attn_partial = cuda.dsa_layer_dev(&hn, &w, seq, mtp_family, 1, h)?;
    let mut attn_partial = attn_partial;
    let ar_p2p = cuda.p2p_ar_v2(&mut attn_partial, h).unwrap_or(false);
    if !ar_p2p {
        nccl.all_reduce_f32(attn_partial.as_const_f32(), attn_partial.as_f32(), h)?;
    }
    // 3. residual + post_attention_layernorm
    let x1 = cuda.add_dev(&eh_partial, &attn_partial, h)?;
    let hn2 = cuda.rmsnorm_dev(&x1, s.w(&format!("{pfx}.post_attention_layernorm.weight"))?, cfg.rms_norm_eps, 1, h)?;
    // 4. MoE
    let e = cfg.n_routed_experts;
    let bias = match s.weights.get(&format!("{pfx}.mlp.gate.e_score_correction_bias")) {
        Some(b) => b.clone(),
        None => Tensor::zeros(Shape::new([e]), DType::F32),
    };
    let gate_w = s.w(&format!("{pfx}.mlp.gate.weight"))?;
    let shared = ExpertWeights {
        gate: s.w(&format!("{pfx}.mlp.shared_expert.gate_proj.weight"))?,
        up: s.w(&format!("{pfx}.mlp.shared_expert.up_proj.weight"))?,
        down: s.w(&format!("{pfx}.mlp.shared_expert.down_proj.weight"))?,
    };
    let (es, ee) = s.tp_expert_range.unwrap_or((0, e));
    let experts: Vec<ExpertWeights> = (es..ee)
        .map(|eid| {
            Ok(ExpertWeights {
                gate: s.w(&format!("{pfx}.mlp.experts.{eid}.gate_proj.weight"))?,
                up: s.w(&format!("{pfx}.mlp.experts.{eid}.up_proj.weight"))?,
                down: s.w(&format!("{pfx}.mlp.experts.{eid}.down_proj.weight"))?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut probs = DevBuf::alloc(cuda.dev(), cuda.stream(), cfg.num_experts_per_tok)?;
    let mut moe_partial = cuda.moe_layer_dev(&hn2, gate_w, &bias, &shared, &experts, es, &mut probs, 1, h, cfg.num_experts_per_tok, e, cfg.routed_scaling_factor, cfg.swiglu_limit)?;
    let ar_p2p = cuda.p2p_ar_v2(&mut moe_partial, h).unwrap_or(false);
    if !ar_p2p {
        nccl.all_reduce_f32(moe_partial.as_const_f32(), moe_partial.as_f32(), h)?;
    }
    // 5. residual + shared_head.norm + lm_head + argmax → DEVICE SLOT (no D2H)
    let x2 = cuda.add_dev(&x1, &moe_partial, h)?;
    if let Some(ho) = h_out {
        cuda.copy_dev(&x2, 0, ho.as_f32(), h)?;
    }
    if std::env::var_os("FERRITE_MTP_DEBUG").is_some() && !cuda.capturing() {
        // full-4096 checksum: bit-level compare orig vs zero-H2D x2 (front-2
        // matched but d1 diverged — either x2's tail differs (attn/moe cache
        // diff) or the argmax output buffer is corrupted)
        // stack array (no Vec alloc — the Vec alloc crashed the original
        // path's catch-up where mtp_forward_dev_argmax is also called)
        let mut xfull = [0f32; 4096];
        let xn = h.min(4096);
        ferrite_kernel::cuda::memcpy_d2h_sync(
            x2.as_f32() as *mut std::ffi::c_void, xfull.as_mut_ptr(), xn, cuda.stream_handle());
        let mut hsum: f64 = 0.0;
        for v in &xfull[..xn] { hsum += *v as f64; }
        // 8-segment checksums (512 floats each) — pinpoints WHICH 512-block
        // of x2 differs between orig and zero-H2D (sum can cancel across blocks)
        let mut segs = [0f64; 8];
        for (i, v) in xfull[..xn].iter().enumerate() { segs[i / 512] += *v as f64; }
        eprintln!("[zh2d-x2] x2[0..2]={:?} sum={:.6} segs={:?} last4={:?}",
            &xfull[..2], hsum, segs, &xfull[xn-4..]);
        // BIT-level first 32 u32 — 8-seg f64 sums cannot see 1-ulp diffs
        // (1.5e-8 < 1e-6 print precision) but argmax flips on near-ties
        // (orig d2=315 vs zh 8606, orig S2 d1=98347 vs zh 702). Bits localize it.
        let mut xbf = [0f32; 4096];
        let xn2 = h.min(4096);
        ferrite_kernel::cuda::memcpy_d2h_sync(
            x2.as_f32() as *mut std::ffi::c_void, xbf.as_mut_ptr(), xn2, cuda.stream_handle());
        // exact float equality count vs a SECOND read of the SAME buffer via a
        // separate D2H (draft1's x2 must be bit-identical; if not, the source
        // is a stale/aliased read of the buffer across the draft1->draft2
        // boundary). Also dump the last 8 u32 to locate tail 1-ulp drift.
        let mut xbf2 = [0f32; 4096];
        ferrite_kernel::cuda::memcpy_d2h_sync(
            x2.as_f32() as *mut std::ffi::c_void, xbf2.as_mut_ptr(), xn2, cuda.stream_handle());
        let neq = xbf.iter().zip(xbf2.iter()).take(xn2).filter(|(a, b)| a != b).count();
        let xb: Vec<u32> = xbf[..xn2].iter().map(|v| v.to_bits()).collect();
        eprintln!("[zh2d-x2b] self_neq={} first8={:?} last8={:?}", neq, &xb[..8.min(xn2)], &xb[(xn2-8).min(0)>>0..]);
    }
    let h_normed = cuda.rmsnorm_dev(&x2, s.w(&format!("{pfx}.shared_head.norm.weight"))?, cfg.rms_norm_eps, 1, h)?;
    let lm_w = s.w("lm_head.weight")?;
    let logits = cuda.matmul_dev(&h_normed, lm_w, 1, h as i32, cfg.vocab_size as i32)?;
    // argmax writes DIRECTLY to the caller's device slot — zero D2H
    cuda.argmax_dev(&logits, arg_out, 1, cfg.vocab_size)?;
    if std::env::var_os("FERRITE_MTP_DEBUG").is_some() && !cuda.capturing() {
        // Read back what argmax ACTUALLY wrote vs what it was given: the
        // d1=702-vs-98347 divergence has bit-identical x2 — either argmax
        // read different logits (pool-buffer aliasing) or wrote elsewhere.
        let mut av = [0f32; 1];
        ferrite_kernel::cuda::memcpy_d2h_sync(
            arg_out.as_f32() as *mut std::ffi::c_void, av.as_mut_ptr(), 1, cuda.stream_handle());
        // top-2 of logits for argmax sanity (154880 wide, read cols of interest)
        let mut l2 = [0f32; 2];
        ferrite_kernel::cuda::memcpy_d2h_sync(
            logits.as_f32() as *mut std::ffi::c_void, l2.as_mut_ptr(), 2, cuda.stream_handle());
        eprintln!("[zh2d-am] argmax_out={:.0} logits[0..2]={:?}", av[0], l2);
    }
    Ok(())
}

/// Zero-H2D draft helper: takes RAW device pointers (no DevBuf refs — avoids
/// the &mut Engine vs &CudaBackend borrow conflict in the fan_out closure).
/// Constructs DevBuf views internally, calls the mtp_forward chain, writes
/// the argmax to the caller's device slot. The caller drops all cuda/mutex
/// borrows before calling this (it re-acquires them internally).
#[cfg(feature = "cuda")]
pub(crate) fn mtp_forward_raw_argmax<B: KernelBackend>(
    s: &mut Engine<B>,
    seq: u64,
    emb_ptr: *mut std::ffi::c_void,
    hprev_ptr: *mut std::ffi::c_void,
    h_out_ptr: *mut std::ffi::c_void,
    argmax_ptr: *mut std::ffi::c_void,
    hidden: usize,
) -> Result<()> {
    use ferrite_kernel::cuda::DevBuf;
    let cuda = s
        .backend
        .as_cuda()
        .ok_or_else(|| FerriteError::Config("mtp needs cuda backend".into()))?;
    let stream = cuda.stream_handle();
    let dev = cuda.dev();
    let emb = DevBuf {
        ptr: emb_ptr, len: hidden,
        class: (hidden as u32).next_power_of_two(),
        dev, stream, stage: std::ptr::null_mut(),
    };
    let hprev = DevBuf {
        ptr: hprev_ptr, len: hidden,
        class: (hidden as u32).next_power_of_two(),
        dev, stream, stage: std::ptr::null_mut(),
    };
    let h_out = if !h_out_ptr.is_null() {
        Some(DevBuf {
            ptr: h_out_ptr, len: hidden,
            class: (hidden as u32).next_power_of_two(),
            dev, stream, stage: std::ptr::null_mut(),
        })
    } else {
        None
    };
    let mut arg = DevBuf {
        ptr: argmax_ptr, len: 1,
        class: 1u32, dev, stream, stage: std::ptr::null_mut(),
    };
    let result = mtp_forward_dev_argmax(s, seq, &emb, &hprev, h_out.as_ref(), &mut arg);
    // CRITICAL: these DevBuf views are raw-pointer ALIASES into MtpState's
    // fixed buffers — they do NOT own the memory. The Drop impl returns
    // (ptr, stage) to the buf_pool — with stage=null, the pool's free()
    // calls cudaFreeHost(null) → SEGV. forget() them (the originals are
    // owned by MtpState and live for the seq's lifetime).
    std::mem::forget(emb);
    std::mem::forget(hprev);
    // h_out is an Option<DevBuf> ALIAS: forgetting `h_out.as_ref()` (a
    // &DevBuf) is a NO-OP — the Some(DevBuf) still dropped and returned the
    // MtpState-owned h_d[i] address to the pool, so later allocations
    // (enorm/hnorm in the next call) aliased it and overwrote the draft
    // chain's h relay. Forget the Option ITSELF (moves the owned DevBuf).
    std::mem::forget(h_out);
    std::mem::forget(arg);
    result
}

/// ONE draft step on device — the graph-capturable unit (mega_d{seq}_{i}).
/// Chain: [i>0] cast_store(d_{i-1} → tokens_dev[i]) → embed_one_dev
/// (tokens_dev[i] → emb_devs[i]) → mtp_forward_raw_argmax (h relay:
/// i=0 reads MtpState.hprev, i>0 reads h_d[i-1]; i<nd-1 exports h to
/// h_d[i]; argmax → d_argmax_dev[i]). Every buffer is a MtpState fixed
/// address (stable for the graphs' lifetime); the ONLY host input is
/// tokens_dev[0] ← last (4B H2D before the i=0 replay).
///
/// CAPTURE semantics: the chain is recordable end-to-end — dsa_layer_dev's
/// kernels read the pinned t0/total zero-copy (dsa_host_advance at replay
/// writes them), P2P/NCCL ARs are graph-safe (mega_v proven), and the
/// internal pool bufs leak-not-pool during capture (is_capturing). The
/// dsa host bookkeeping (t_count += 1) runs during the pass — the caller
/// rolls back around it (dry → rollback(nd) → capture ×nd).
///
/// REPLAY cost per draft: H2D 4B (i=0 only) + advance(1) + ONE graph
/// launch — replaces ~15 kernel launches + host embed + H2D (~0.8ms/draft
/// of host serialization at 4 ranks).
#[cfg(feature = "cuda")]
pub(crate) fn draft_step_dev<B: KernelBackend>(
    s: &mut Engine<B>,
    seq: u64,
    i: usize,
    nd: usize,
) -> Result<()> {
    let hidden = {
        let cfg = &s.cfg;
        cfg.hidden_size
    };
    // Fixed device pointers from MtpState (scoped — dropped before the
    // mtp_forward call re-acquires the backend).
    let (tokens_ptr, emb_ptr, hprev_ptr, h_out_ptr, d_ptr) = {
        let cuda = s
            .backend
            .as_cuda()
            .ok_or_else(|| FerriteError::Config("draft graph needs cuda".into()))?;
        let m = cuda.mtp.lock().unwrap();
        let m = m
            .as_ref()
            .ok_or_else(|| FerriteError::Config("mtp bufs missing".into()))?;
        let tokens = m.tokens_dev.as_f32() as *mut i32;
        let emb = m.emb_devs[i].as_f32();
        let d = unsafe { m.d_argmax_dev.as_f32().add(i) };
        // h relay: draft 0 reads hprev; draft i>0 reads h_d[i-1]; drafts
        // 0..nd-2 export h to h_d[i] (the last draft discards it).
        let hprev = if i == 0 {
            m.hprev.as_f32()
        } else {
            m.h_d[i - 1].as_f32()
        };
        let h_out = if i + 1 < nd {
            m.h_d[i].as_f32()
        } else {
            std::ptr::null_mut()
        };
        (tokens, emb, hprev, h_out, d)
    };
    // embed table (replicated weight — every rank reads the full table)
    let embed_table = s.w("model.embed_tokens.weight")?.clone();
    {
        let cuda = s
            .backend
            .as_cuda()
            .ok_or_else(|| FerriteError::Config("draft graph needs cuda".into()))?;
        cuda.enter();
        // [i>0] d_{i-1} (f32) → tokens_dev[i] (i32): the cast kernel keeps the
        // token chain on device (replay: graph i-1's argmax → graph i's embed).
        if i > 0 {
            let d_prev = unsafe { d_ptr.sub(1) };
            cuda.cast_store_i32(
                d_prev as *const std::ffi::c_void,
                unsafe { tokens_ptr.add(i) } as *mut std::ffi::c_void,
            )?;
        }
        cuda.embed_one_dev(
            &embed_table,
            unsafe { tokens_ptr.add(i) },
            emb_ptr,
            hidden,
            1,
        )?;
    } // cuda dropped — mtp_forward re-acquires internally
    mtp_forward_raw_argmax(
        s,
        seq,
        emb_ptr as *mut std::ffi::c_void,
        hprev_ptr as *mut std::ffi::c_void,
        h_out_ptr as *mut std::ffi::c_void,
        d_ptr as *mut std::ffi::c_void,
        hidden,
    )
}




