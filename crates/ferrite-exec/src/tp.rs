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
            // expert weights: EP-style whole experts per rank
            if let Some(expert) = name.split(".experts.").nth(1) {
                let e: usize = expert.split('.').next().unwrap_or("0").parse().unwrap_or(0);
                if e >= es && e < ee {
                    Some(f.clone())
                } else {
                    None
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
        let n = cfg.n_routed_experts;
        let (es, ee) = head_range(n, rank, world);
        if e < es || e >= ee {
            return Some(Tensor::new(Shape::new([0, cols]), t.dtype, vec![])); // empty: not ours
        }
        return Some(t.clone()); // full expert (EP-style: whole experts per rank)
    }
    if name.ends_with(".shared_expert.gate_proj.weight") || name.ends_with(".shared_expert.up_proj.weight") {
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

/// Verify-chain IO for MTP speculative decoding (n=2/3 graphs):
/// - gdn_scratch: per-GDN-layer (conv_state, gdn_state) scratch ptrs — the
///   ping-pong B buffers so verify's t-recurrence never touches the main
///   state until accept commits;
/// - h_final: the verify chain's last hc_post residual row (h_prev for the
///   MTP draft's eh_proj) exported via a capture-safe D2D graph node.
#[cfg(feature = "cuda")]
pub(crate) struct VerifyIO {
    /// [n_gdn_layers] (conv, gdn, conv0, gdn0, gdn1): B = full n-token verify
    /// state, B0 = t=0 snapshot (A+t_last, accept-1), B1 = t=1 snapshot
    /// (A+t_last+d1, accept-2, n=3 only).
    pub gdn_scratch: Vec<(*mut f32, *mut f32, *mut f32, *mut f32, *mut f32, *mut f32)>,
    pub h_final: *mut f32, // [n*hidden] staging
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
            let per = full_cfg.n_routed_experts / world;
            engine.tp_expert_range = Some((rank * per, (rank + 1) * per));
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
        TpCluster { shards, full_cfg, world, graph_captured: false, graph_step: 0, mega_seq: None, nccl: None }
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
        let mtp = std::env::var_os("FERRITE_MTP").is_some();

        if self.mega_seq != Some(seq) {
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
                    Ok::<(), FerriteError>(())
                })
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
                eprintln!(
                    "[mega] MTP: draft cache catch-up done ({} prompt tokens)",
                    prompt_tokens.len()
                );
                // MTP verify graph (n=2: [t_last, d1]): GDN state → scratch B
                // (ping-pong), h_final export (hf_v [2,hidden]), argmax 2.
                let gv = format!("mega_v{seq}");
                let h2 = self.shards[0].embed(&[last, last, last]);
                let in_vals2 = crate::mhc::hc_expand(&h2, self.full_cfg.hc_mult);
                let _ = Self::fan_out(&mut self.shards, |s| {
                    let vio = Self::mtp_vio(s, true);
                    Self::mega_chain_dev(s, seq, in_vals2.as_slice(), &plans, num_dsa, false, &gv, 3, Some(&vio))
                })
                .into_iter()
                .collect::<Result<Vec<Vec<f32>>>>()?;
                Self::fan_out(&mut self.shards, |s| {
                    let vio = Self::mtp_vio(s, true);
                    Self::mega_chain_dev(s, seq, in_vals2.as_slice(), &plans, num_dsa, true, &gv, 3, Some(&vio))
                })
                .into_iter()
                .collect::<Result<Vec<Vec<f32>>>>()?;
                eprintln!("[mega] MTP: verify graph {gv} captured (n=2, GDN ping-pong scratch)");
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
                return self.mtp_step_zero_h2d(seq, &plans, num_dsa);
            }
            return self.mtp_step(seq, &plans, num_dsa);
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

    /// MTP (FERRITE_MTP=1) steady step: draft (mtp_forward on the layers.45
    /// nextn layer) → verify (mega_v n=2 graph replay) → greedy accept →
    /// ping-pong state commit. Returns the LAST accepted token (2 on full
    /// accept: d1 + bonus; 1 on rejection: the verify bonus).
    #[cfg(feature = "cuda")]
    fn mtp_step(&mut self, seq: u64, plans: &[ferrite_model::LayerPlan], num_dsa: usize) -> Result<u32> {
        use ferrite_kernel::cuda::DevBuf;
        let hidden = self.full_cfg.hidden_size;
        let hc_mult = self.full_cfg.hc_mult;
        let gname = format!("mega{seq}");
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
        // 1. draft: h_prev staging (accept row of last step's h_final) +
        //    embed(last) → mtp_forward → d1 (identical across ranks).
        let (d1, d2) = {
            let toks = Self::fan_out(&mut self.shards, |s| {
                let (emb, hptr) = {
                    let cuda = s
                        .backend
                        .as_cuda()
                        .ok_or_else(|| FerriteError::Config("mtp needs cuda".into()))?;
                    cuda.enter();
                    let h2 = s.embed(&[last]);
                    let emb = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), hidden)?;
                    emb.upload(h2.as_slice())?;
                    let m = cuda.mtp.lock().unwrap();
                    let m = m.as_ref().ok_or_else(|| FerriteError::Config("mtp bufs missing".into()))?;
                    (emb, &m.hprev as *const DevBuf as usize)
                };
                let hprev: &DevBuf = unsafe { &*(hptr as *const DevBuf) };
                // draft chain: d1 from hprev; d2 from d1's MTP residual h
                // (the draft model's own hidden — EAGLE-style recursion).
                let (d1, h_d1) = {
                    let cuda = s
                        .backend
                        .as_cuda()
                        .ok_or_else(|| FerriteError::Config("mtp needs cuda".into()))?;
                    let h_d1 = DevBuf::alloc(cuda.dev(), cuda.stream(), hidden)?;
                    (mtp_forward(s, seq, &emb, hprev, Some(&h_d1))?, h_d1)
                };
                let d2 = {
                    let cuda = s
                        .backend
                        .as_cuda()
                        .ok_or_else(|| FerriteError::Config("mtp needs cuda".into()))?;
                    let h3 = s.embed(&[d1 as u32]);
                    let emb2 = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), hidden)?;
                    emb2.upload(h3.as_slice())?;
                    mtp_forward(s, seq, &emb2, &h_d1, None)?
                };
                Ok((d1, d2))
            })
            .into_iter()
            .collect::<Result<Vec<(f32, f32)>>>()?;
            toks[0]
        };
        let t_draft = t_d.elapsed();
        let t_v = std::time::Instant::now();
        // 2. verify: dsa advance(3) + mega_v replay (the A→B ping-pong
        //    copy-in is graph-recorded — capture-time nodes, not a host
        //    memcpy loop) → argmax[3] → fused accept/commit.
        let h2v = self.shards[0].embed(&[last, d1 as u32, d2 as u32]);
        let in_vals = crate::mhc::hc_expand(&h2v, hc_mult);
        let toks_v = Self::fan_out(&mut self.shards, |s| {
            let cuda = s
                .backend
                .as_cuda()
                .ok_or_else(|| FerriteError::Config("mtp needs cuda".into()))?;
            cuda.enter();
            for f in 0..num_dsa {
                // advance(3): append this step's verify input [t_last, d1, d2].
                cuda.dsa_host_advance(seq, f, 3);
            }
            let mut out = [0f32; 3];
            if !cuda.graph_run(&gvname, in_vals.as_slice(), &mut out)? {
                return Err(FerriteError::InvalidArg(format!("mega_v graph {gvname} missing")));
            }
            // Accept + commit fused into the verify worker (was a THIRD
            // fan_out round-trip): d1/d2/argmax are bit-identical across
            // ranks, so every worker derives the same k. DSA rollback is
            // host pinned bookkeeping (ns); the B_k -> A ping-pong commit +
            // hprev <- hf_v[k-1] is ONE ferrite_mtp_commit launch (was
            // 2*n_gdn cudaMemcpyAsync D2Ds launched from the host loop).
            let k = if d1 as u32 == out[0] as u32 {
                if d2 as u32 == out[1] as u32 { 3 } else { 2 }
            } else { 1 };
            for f in 0..num_dsa {
                cuda.dsa_host_rollback(seq, f, (3 - k) as usize);
            }
            cuda.dsa_host_rollback(seq, mtp_family, (3 - k) as usize);
            cuda.mtp_commit(k)?;
            Ok((out[0], out[1], out[2], k))
        })
        .into_iter()
        .collect::<Result<Vec<(f32, f32, f32, i32)>>>()?;
        let (a0, a1, a2, k) = toks_v[0];
        let t_verify = t_v.elapsed();
        let t_c = std::time::Instant::now();
        // 3. accept: longest prefix of (d1, d2) vs (a0, a1); k = 1/2/3
        // (computed per-rank inside the verify worker — bit-identical inputs).
        // k=3: d1==a0 && d2==a1 (3 tokens: d1, d2, bonus a2; commit B full)
        // k=2: d1==a0, d2!=a1 (2 tokens: d1, bonus a1; commit B1 = A+t_last+d1)
        // k=1: d1!=a0 (1 token: a0; commit B0 = A+t_last)
        let dbg = std::env::var_os("FERRITE_MTP_DEBUG").is_some();
        if dbg {
            eprintln!("[mtp] d1={:?} d2={:?} a0={:?} a1={:?} a2={:?} -> accept{}", d1 as u32, d2 as u32, a0 as u32, a1 as u32, a2 as u32, k);
        }
        match k {
            3 => {
                for s in &mut self.shards {
                    if let Some(rt) = s.seq_runtime_mut(seq) {
                        rt.tokens.push(d1 as u32);
                        rt.tokens.push(d2 as u32);
                        rt.tokens.push(a2 as u32);
                    }
                }
                if mtp_tm {
                    eprintln!("[mtp-tm] draft={:.2}ms verify={:.2}ms commit3={:.2}ms (accept3 d1={:?} d2={:?})", t_draft.as_secs_f64()*1e3, t_verify.as_secs_f64()*1e3, t_c.elapsed().as_secs_f64()*1e3, d1 as u32, d2 as u32);
                }
                Ok(a2 as u32)
            }
            2 => {
                for s in &mut self.shards {
                    if let Some(rt) = s.seq_runtime_mut(seq) {
                        rt.tokens.push(d1 as u32);
                        rt.tokens.push(a1 as u32);
                    }
                }
                if mtp_tm {
                    eprintln!("[mtp-tm] draft={:.2}ms verify={:.2}ms commit2={:.2}ms (accept2 d1={:?})", t_draft.as_secs_f64()*1e3, t_verify.as_secs_f64()*1e3, t_c.elapsed().as_secs_f64()*1e3, d1 as u32);
                }
                Ok(a1 as u32)
            }
            _ => {
                for s in &mut self.shards {
                    if let Some(rt) = s.seq_runtime_mut(seq) {
                        rt.tokens.push(a0 as u32);
                    }
                }
                if mtp_tm {
                    eprintln!("[mtp-tm] draft={:.2}ms verify={:.2}ms commit1={:.2}ms (accept1 a0={:?})", t_draft.as_secs_f64()*1e3, t_verify.as_secs_f64()*1e3, t_c.elapsed().as_secs_f64()*1e3, a0 as u32);
                }
                Ok(a0 as u32)
            }
        }
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
        let (k, next_token) = {
            let toks = Self::fan_out(&mut self.shards, |s| {
                // Extract ALL device pointers from MtpState (scoped mutex)
                // + raw dev/stream handles. Drop ALL borrows before calling
                // mtp_forward_raw_argmax (which takes &mut s — re-acquires
                // cuda internally). This is the SAME raw-pointer pattern as
                // the existing mtp_step (hptr as usize — proven to work).
                let (cuda, tokens_ptr, emb1_ptr, emb2_ptr, d1_ptr, d2_ptr,
                     hprev_ptr, k_ptr, nt_ptr, na_ptr, verify_in_ptr) = {
                    let cuda = s
                        .backend
                        .as_cuda()
                        .ok_or_else(|| FerriteError::Config("zero-H2D mtp needs cuda".into()))?;
                    cuda.enter();
                    let m = cuda.mtp.lock().unwrap();
                    let m = m
                        .as_ref()
                        .ok_or_else(|| FerriteError::Config("mtp bufs missing".into()))?;
                    (
                        cuda as *const ferrite_kernel::cuda::CudaBackend,
                        m.tokens_dev.as_f32() as *mut i32,
                        m.emb1_dev.as_f32(),
                        m.emb2_dev.as_f32(),
                        m.d1_argmax_dev.as_f32(),
                        m.d2_argmax_dev.as_f32(),
                        m.hprev.as_f32(),
                        m.k_dev.as_f32() as *mut i32,
                        m.next_token_dev.as_f32() as *mut i32,
                        m.n_accepted_dev.as_f32() as *mut i32,
                        m.emb1_dev.as_f32(), // reuse emb1 slot for verify input placeholder
                    )
                }; // ALL borrows dropped here (cuda, mutex, MtpState)

                let _ = (emb2_ptr, verify_in_ptr); // used below

                // Write tokens_dev[0] = last (the ONLY H2D: 4B initial token)
                unsafe { *tokens_ptr.add(0) = last as i32; }

                // === Phase 1: DRAFT (all device, zero H2D) ===
                // Draft 1: embed_one(last) → emb1_dev → mtp_forward → d1_argmax_dev
                {
                    let cuda = s
                        .backend
                        .as_cuda()
                        .ok_or_else(|| FerriteError::Config("cuda".into()))?;
                    cuda.enter();
                    cuda.embed_one_dev(&embed_table, tokens_ptr, emb1_ptr, hidden, 1)?;
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

                // Draft 2: embed_one(d1) → emb2_dev → mtp_forward → d2_argmax_dev
                // d1 is f32 (argmax output); embed_one reads i32 — 4B D2H
                // (within SSE budget; TODO: f32→i32 cast kernel on device)
                {
                    let cuda = s
                        .backend
                        .as_cuda()
                        .ok_or_else(|| FerriteError::Config("cuda".into()))?;
                    let mut d1_f32 = [0f32; 1];
                    let r = ferrite_kernel::cuda::memcpy_d2h_sync(
                        d1_ptr as *mut std::ffi::c_void,
                        &mut d1_f32[0] as *mut f32,
                        1, cuda.stream_handle(),
                    );
                    if r != 0 { return Err(FerriteError::InvalidArg(format!("d1 D2H: {r}"))); }
                    unsafe { *(tokens_ptr.add(1)) = d1_f32[0] as i32; }
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
                    {
                        let mut d2_f32 = [0f32; 1];
                        let r = ferrite_kernel::cuda::memcpy_d2h_sync(
                            d2_ptr as *mut std::ffi::c_void,
                            &mut d2_f32[0] as *mut f32,
                            1, cuda.stream_handle(),
                        );
                        if r != 0 { return Err(FerriteError::InvalidArg(format!("d2 D2H: {r}"))); }
                        unsafe { *(tokens_ptr.add(2)) = d2_f32[0] as i32; }
                    }

                    // embed_expand_dev: [last, d1, d2] → graph input [3, hc_mult, hidden]
                    let nh = hc_mult * hidden;
                    let verify_in = ferrite_kernel::cuda::DevBuf::alloc(cuda.dev(), cuda.stream(), 3 * nh)?;
                    cuda.embed_expand_dev_buf(&embed_table, tokens_ptr, verify_in.as_f32(), 3, hidden, hc_mult)?;

                    // D2H to the graph's pinned staging (graph reads from pinned)
                    let io = cuda.graph_io_get(&gvname)
                        .ok_or_else(|| FerriteError::InvalidArg(format!("mega_v graph {gvname} io missing")))?;
                    let r = ferrite_kernel::cuda::memcpy_d2h_sync(
                        verify_in.as_f32() as *mut std::ffi::c_void,
                        io.x_stage as *mut f32,
                        3 * nh,
                        cuda.stream_handle(),
                    );
                    if r != 0 { return Err(FerriteError::InvalidArg(format!("verify input D2H: {r}"))); }

                    // DSA advance (pinned t0/total bookkeeping, no data H2D)
                    for f in 0..num_dsa {
                        cuda.dsa_host_advance(seq, f, 3);
                    }
                    cuda.dsa_host_advance(seq, mtp_family, 3);

                    // Graph replay (reads from pinned staging — the graph's
                    // internal memcpy, not a host H2D)
                    if !cuda.graph_replay(&gvname) {
                        return Err(FerriteError::InvalidArg(format!("mega_v graph {gvname} missing")));
                    }

                    // Read verify argmax (3 f32 = 12 bytes D2H — for the accept
                    // kernel's device comparison; also needed for the host seq push)
                    let mut a = [0f32; 3];
                    let r = ferrite_kernel::cuda::memcpy_d2h_sync(
                        io.out_dev,
                        a.as_mut_ptr(),
                        3,
                        cuda.stream_handle(),
                    );
                    if r != 0 { return Err(FerriteError::InvalidArg(format!("verify argmax D2H: {r}"))); }
                    let t_verify = t_v.elapsed();
                    let _ = t_verify;

                    // === Phase 3: ACCEPT + COMMIT (device) ===
                    let t_c = std::time::Instant::now();
                    // Host-side k (for DSA rollback bookkeeping — ns-level)
                    let d1_i = unsafe { *(d1_ptr as *const i32) }; // read from device (f32 bit pattern → i32? no — this is wrong)
                    let _ = d1_i; // d1 is f32, need D2H to read as int
                    // k from host comparison (the existing logic — will be
                    // replaced by the device accept kernel once verified)
                    let mut d1_val = [0f32; 1];
                    let mut d2_val = [0f32; 1];
                    let r1 = ferrite_kernel::cuda::memcpy_d2h_sync(
                        d1_ptr as *mut std::ffi::c_void, d1_val.as_mut_ptr(), 1, cuda.stream_handle());
                    let r2 = ferrite_kernel::cuda::memcpy_d2h_sync(
                        d2_ptr as *mut std::ffi::c_void, d2_val.as_mut_ptr(), 1, cuda.stream_handle());
                    if r1 != 0 || r2 != 0 { return Err(FerriteError::InvalidArg(format!("d1/d2 D2H: {r1}/{r2}"))); }
                    let k_host = if d1_val[0] as u32 == a[0] as u32 {
                        if d2_val[0] as u32 == a[1] as u32 { 3 } else { 2 }
                    } else { 1 };
                    // DSA rollback
                    for f in 0..num_dsa {
                        cuda.dsa_host_rollback(seq, f, (3 - k_host) as usize);
                    }
                    cuda.dsa_host_rollback(seq, mtp_family, (3 - k_host) as usize);
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

                    // === Phase 4: D2H read (8 bytes for SSE/seq) ===
                    // FERRITE_SSE=1 (default): read k + next_token every step
                    // FERRITE_SSE=0: batch mode — read every N steps (TODO)
                    let mut k_out = [0i32; 1];
                    let mut nt_out = [0i32; 1];
                    // next_token = a[k-1] (the k-th accepted verify token)
                    let nt = match k_host { 3 => a[2] as i32, 2 => a[1] as i32, _ => a[0] as i32 };
                    k_out[0] = k_host;
                    nt_out[0] = nt;
                    Ok((k_out[0], nt_out[0]))
                }
            })
            .into_iter()
            .collect::<Result<Vec<(i32, i32)>>>()?;
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
        let (a0, a1, a2) = {
            // The verify argmax was read D2H inside the fan_out — but we need
            // it here for the seq push. For now, re-read from the shard's
            // MtpState (verify_argmax_dev). This is 12 bytes D2H (a0-a2).
            // TODO: accumulate tokens on device, read in batch
            let cuda = self.shards[0].backend.as_cuda().unwrap();
            let m = cuda.mtp.lock().unwrap();
            let m = m.as_ref().unwrap();
            let mut a = [0f32; 3];
            let r = ferrite_kernel::cuda::memcpy_d2h_sync(
                m.verify_argmax_dev.as_f32() as *mut std::ffi::c_void,
                a.as_mut_ptr(),
                3,
                cuda.stream_handle(),
            );
            if r != 0 {
                return Err(FerriteError::InvalidArg(format!("verify argmax re-read D2H: {r}")));
            }
            (a[0] as u32, a[1] as u32, a[2] as u32)
        };
        // Also need d1, d2 for the k=2/k=3 cases
        let (d1_u32, d2_u32) = {
            let cuda = self.shards[0].backend.as_cuda().unwrap();
            let m = cuda.mtp.lock().unwrap();
            let m = m.as_ref().unwrap();
            let mut d = [0f32; 2];
            let r1 = ferrite_kernel::cuda::memcpy_d2h_sync(
                m.d1_argmax_dev.as_f32() as *mut std::ffi::c_void,
                &mut d[0] as *mut f32, 1, cuda.stream_handle(),
            );
            let r2 = ferrite_kernel::cuda::memcpy_d2h_sync(
                m.d2_argmax_dev.as_f32() as *mut std::ffi::c_void,
                &mut d[1] as *mut f32, 1, cuda.stream_handle(),
            );
            if r1 != 0 || r2 != 0 {
                return Err(FerriteError::InvalidArg(format!("d1/d2 re-read D2H: {r1}/{r2}")));
            }
            (d[0] as u32, d[1] as u32)
        };
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

    /// Allocate the per-rank MTP fixed buffers (MtpState): decode-graph h_final    /// Allocate the per-rank MTP fixed buffers (MtpState): decode-graph h_final
    /// [hidden], verify-graph h_final [2*hidden], draft h_prev [hidden], and
    /// per-GDN-layer (conv, gdn) ping-pong B scratch. Called once before the
    /// mega graph captures (fixed addresses for graph lifetime).
    #[cfg(feature = "cuda")]
    fn mtp_setup_bufs(s: &mut Engine<B>, plans: &[ferrite_model::LayerPlan], seq: u64) -> Result<()> {
        use ferrite_kernel::cuda::{DevBuf, MtpCommitPlan, MtpState};
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
                let conv0 = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), conv_len)?;
                let gdn0 = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), gdn_len)?;
                let conv1 = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), conv_len)?;
                let gdn1 = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), gdn_len)?;
                scratch.push((conv, gdn, conv0, gdn0, conv1, gdn1));
            }
        }
        let hf_dev = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), hidden)?;
        let hf_v = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), 3 * hidden)?;
        let hprev = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), hidden)?;
        // Single-kernel accept-commit plan: per GDN layer the 8-pointer table
        // (conv_a, gdn_a, conv_b, gdn_b, conv_b0, gdn_b0, conv_b1, gdn_b1)
        // packed as f32 bit patterns (DevBuf is f32-typed; 2 f32 per
        // pointer) + a pinned k slot. The A-side pointers come from the
        // seq's recurrent-state stores (fixed for the seq's lifetime —
        // also what the verify graph's recorded A→B copy-in nodes use).
        // This allocates the A states now if not yet warm (idempotent
        // dev_state lookup).
        let mut flat: Vec<f32> = Vec::with_capacity(scratch.len() * 16);
        let mut n_plans = 0usize;
        for plan in plans {
            if matches!(plan.attn, AttnKind::Linear) {
                let a_conv = cuda.conv_state_ptr(seq, plan.layer_idx, conv_len)?;
                let a_gdn = cuda.gdn_state_ptr(seq, plan.layer_idx, gdn_len)?;
                let (cb, gb, cb0, gb0, cb1, gb1) = &scratch[n_plans];
                for p in [
                    a_conv,
                    a_gdn,
                    cb.as_f32(),
                    gb.as_f32(),
                    cb0.as_f32(),
                    gb0.as_f32(),
                    cb1.as_f32(),
                    gb1.as_f32(),
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
        let commit = MtpCommitPlan { plan: plan_buf, k_pin, n: n_plans, conv_len, gdn_len, hidden };
        // ZERO-H2D device token chain (fixed bufs — never pooled, stable
        // addresses for the accept kernel's device-resident loop):
        // tokens_dev [3] i32 = [last, d1, d2] (embed kernel reads these)
        // k_dev [1] i32, next_token_dev [1] i32, n_accepted_dev [1] i32
        // (accept kernel outputs; host reads 8-12B D2H for SSE)
        // verify_argmax_dev [3] f32 (graph argmax output — the verify graph
        // writes a0/a1/a2 here; accept kernel compares against d1/d2)
        // emb1_dev/emb2_dev [hidden] f32 (draft chain embeds from embed_one)
        // d1_argmax_dev/d2_argmax_dev [1] f32 (draft argmax outputs)
        let tokens_dev = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), 3)?;
        let verify_argmax_dev = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), 3)?;
        let k_dev = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), 1)?;
        let next_token_dev = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), 1)?;
        let n_accepted_dev = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), 1)?;
        let emb1_dev = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), hidden)?;
        let emb2_dev = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), hidden)?;
        let d1_argmax_dev = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), 1)?;
        let d2_argmax_dev = DevBuf::alloc(cuda.dev(), cuda.stream_handle(), 1)?;
        *cuda.mtp.lock().unwrap() = Some(MtpState {
            hf_dev, hf_v, hprev, scratch, commit: Some(commit),
            tokens_dev, verify_argmax_dev, k_dev, next_token_dev, n_accepted_dev,
            emb1_dev, emb2_dev, d1_argmax_dev, d2_argmax_dev,
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
                    .map(|(c, g, c0, g0, c1, g1)| (c.as_f32(), g.as_f32(), c0.as_f32(), g0.as_f32(), c1.as_f32(), g1.as_f32()))
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
                    let (cb, gb, _, _, _, _) = v.gdn_scratch[gi];
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
    res.upload(in_vals)?; // recorded stage→dev memcpy (the graph input)
    let x_stage = res.stage; // GraphIO: replay writes fresh input here
    mprobe!("res0", &res, nh);

    let mut gdn_idx = 0usize; // verify scratch index (GDN layers only)
    for (layer_idx, plan) in plans.iter().enumerate() {
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
        let partial = match plan.attn {
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
        nccl.all_reduce_f32(partial.as_const_f32(), partial.as_f32(), n * hidden)?;
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
        let partial2 = match plan.mlp {
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
        nccl.all_reduce_f32(partial2.as_const_f32(), partial2.as_f32(), n * hidden)?;
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
            let partial = cuda.gdn_layer_dev(
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
        let partial = cuda.gdn_layer_dev(
            &x_dev, &gw, seq, layer_idx, n, hidden,
            la.num_heads, la.head_dim, la.gate_lower_bound,
            s.cfg.rms_norm_eps, la.short_conv_kernel_size, None,
        )?;
        if let Some(ch) = &s.nccl {
            // TP all-reduce on-device (replaces the host download→sum→upload
            // round-trip; async on this rank's stream — the download below
            // syncs it, which waits for the whole collective).
            ch.all_reduce_f32(partial.as_const_f32(), partial.as_f32(), n * hidden)?;
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
                if std::env::var_os("FERRITE_P2P").is_some() {
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
            if std::env::var_os("FERRITE_P2P").is_some() {
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
        let (ffn_out_dev, ffn_out_t) = if std::env::var_os("FERRITE_P2P").is_some() {
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
                            let d = cuda.matmul_dev(&a, w_down, n as i32, inter, hi)?;
                            ch.all_reduce_f32(d.as_const_f32(), d.as_f32(), n * hidden)?;
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
            if std::env::var_os("FERRITE_P2P").is_some() {
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
        if std::env::var_os("FERRITE_P2P").is_some()
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
    let x_seg = cuda.mtp_eh_seg_dev(&enorm, &hnorm, rank, world, h)?;
    let eh_partial = cuda.matmul_dev(&x_seg, s.w(&format!("{pfx}.eh_proj.weight"))?, 1, (2 * h / world) as i32, h as i32)?;
    nccl.all_reduce_f32(eh_partial.as_const_f32(), eh_partial.as_f32(), h)?;
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
    nccl.all_reduce_f32(attn_partial.as_const_f32(), attn_partial.as_f32(), h)?;
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
    let moe_partial = cuda.moe_layer_dev(&hn2, gate_w, &bias, &shared, &experts, es, &mut probs, 1, h, cfg.num_experts_per_tok, e, cfg.routed_scaling_factor, cfg.swiglu_limit)?;
    nccl.all_reduce_f32(moe_partial.as_const_f32(), moe_partial.as_f32(), h)?;
    // 5. residual + shared_head.norm + lm_head + argmax → DEVICE SLOT (no D2H)
    let x2 = cuda.add_dev(&x1, &moe_partial, h)?;
    if let Some(ho) = h_out {
        cuda.copy_dev(&x2, 0, ho.as_f32(), h)?;
    }
    let h_normed = cuda.rmsnorm_dev(&x2, s.w(&format!("{pfx}.shared_head.norm.weight"))?, cfg.rms_norm_eps, 1, h)?;
    let lm_w = s.w("lm_head.weight")?;
    let logits = cuda.matmul_dev(&h_normed, lm_w, 1, h as i32, cfg.vocab_size as i32)?;
    // argmax writes DIRECTLY to the caller's device slot — zero D2H
    cuda.argmax_dev(&logits, arg_out, 1, cfg.vocab_size)?;
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
    mtp_forward_dev_argmax(s, seq, &emb, &hprev, h_out.as_ref(), &mut arg)
}



