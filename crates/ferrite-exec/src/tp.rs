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

use ferrite_model::{AttnKind, Glm53FlashConfig, MlpKind, Weights, build_layer_plans};
use ferrite_types::{DType, FerriteError, Result, Shape, Tensor};

use crate::Engine;

// ---------------------------------------------------------------------------
// Weight sharding
// ---------------------------------------------------------------------------

/// Slice a weight along dim0 (row split) → rows [start, end).
/// Handles 1D (A_log/o_norm/dt_bias) and 2D tensors.
fn row_split(w: &Tensor, start: usize, end: usize) -> Tensor {
    let dims = &w.shape.0;
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
    let mut out = HashMap::new();

    for name in w.keys() {
        let t = &w[name];
        let local = if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
            // vocab split: rows [vocab/N for this rank] — all-gather at the
            // output boundary. For the CPU simulation we keep full + mask in
            // the forward (simpler); the GPU path splits.
            t.clone()
        } else if name.starts_with("model.norm.weight")
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
        } else if let Some(layer_str) = layer_of(name) {
            let layer: usize = layer_str.parse().unwrap_or(0);
            let plan = build_layer_plans(cfg);
            let lp = plan[layer];
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
}

impl<B: KernelBackend> TpCluster<B> {
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
        TpCluster { shards, full_cfg, world }
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
        let last = {
            let s = self.shards[0]
                .seq_runtime(seq)
                .ok_or_else(|| FerriteError::Config("missing seq".into()))?;
            *s.tokens.last().ok_or_else(|| FerriteError::Config("empty context".into()))?
        };
        let h0 = self.shards[0].embed(&[last]);
        let mut h = if self.full_cfg.mhc {
            crate::mhc::hc_expand(&h0, self.full_cfg.hc_mult)
        } else {
            h0
        };
        let plans = build_layer_plans(&self.full_cfg);
        for plan in &plans {
            h = self.layer_forward_tp(seq, plan.layer_idx, h, 1)?;
        }
        let h_final = if self.full_cfg.mhc {
            crate::mhc::hc_contract(&h, self.full_cfg.hc_mult)
        } else {
            h
        };
        // final norm + lm head: replicated weights — any shard computes them.
        let s0 = &self.shards[0];
        let hn = s0.rmsnorm(&h_final, "model.norm.weight")?;
        let logits = s0.project(&hn, "lm_head.weight")?;
        let mut out = Tensor::zeros(Shape::new([1]), DType::F32);
        s0.backend.argmax_lastdim(&logits, &mut out)?;
        let tok = out.as_slice()[0] as u32;
        for s in &mut self.shards {
            if let Some(rt) = s.seq_runtime_mut(seq) {
                rt.tokens.push(tok);
            }
        }
        Ok(tok)
    }

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
    std::thread::scope(|scope| {
        let handles: Vec<_> = shards.iter_mut().map(|s| scope.spawn(|| f(s))).collect();
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
            let (hc_fn, hc_scale, hc_base) = {
                let s0 = &self.shards[0];
                (
                    s0.w(&format!("{pfx}.hc_attn_fn"))?.clone(),
                    s0.w(&format!("{pfx}.hc_attn_scale"))?.clone(),
                    s0.w(&format!("{pfx}.hc_attn_base"))?.clone(),
                )
            };
            let (li, post_a, comb_a) = self.shards[0].backend.hc_pre(
                &residual,
                &hc_fn,
                &hc_scale,
                &hc_base,
                self.full_cfg.rms_norm_eps,
                self.full_cfg.hc_eps,
                self.full_cfg.hc_sinkhorn_iters,
            )?;
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
            let attn_partials = Self::fan_out(&mut self.shards, |s| match plan.attn {
                AttnKind::Linear => s.linear_attn_forward(seq, layer_idx, &pfx, &hn, n),
                AttnKind::Dsa => s.dsa_attn_forward(seq, layer_idx, &pfx, &hn, n),
            });
            let attn_out = all_reduce_sum(&attn_partials.into_iter().collect::<Result<Vec<_>>>()?);
            if probe {
                let bytes: Vec<u8> = attn_out.as_slice().iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write("/tmp/l0_attn.f32", bytes).ok();
            }
            let res3 =
                Tensor::from_f32(Shape::new([n, hc_mult, hidden]), residual.as_slice().to_vec());
            let res2 = self.shards[0].backend.hc_post(&attn_out, &res3, &post_a, &comb_a)?;
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
            let (li2, post_f, comb_f) = self.shards[0].backend.hc_pre(
                &res2_flat,
                &hc_fn2,
                &hc_scale2,
                &hc_base2,
                self.full_cfg.rms_norm_eps,
                self.full_cfg.hc_eps,
                self.full_cfg.hc_sinkhorn_iters,
            )?;
            let hfn = {
                let s0 = &self.shards[0];
                s0.rmsnorm(&li2, &format!("{pfx}.post_attention_layernorm.weight"))?
            };
            let ffn_partials = Self::fan_out(&mut self.shards, |s| match plan.mlp {
                MlpKind::Dense => s.dense_ffn(&pfx, &hfn, n),
                MlpKind::Moe => s.moe_ffn(&pfx, &hfn, n),
            });
            let ffn_out = all_reduce_sum(&ffn_partials.into_iter().collect::<Result<Vec<_>>>()?);
            if probe {
                let bytes: Vec<u8> = ffn_out.as_slice().iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write("/tmp/l0_ffn.f32", bytes).ok();
            }
            let res3b =
                Tensor::from_f32(Shape::new([n, hc_mult, hidden]), res2_flat.as_slice().to_vec());
            let res_out = self.shards[0].backend.hc_post(&ffn_out, &res3b, &post_f, &comb_f)?;
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
                AttnKind::Linear => s.linear_attn_forward(seq, layer_idx, &pfx, &hn, n),
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
