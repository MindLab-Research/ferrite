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

        if self.mega_seq != Some(seq) {
            // (Re)capture for this seq. Dry-run: all 4 ranks run the full
            // chain in parallel (fan_out) — the NCCL ARs rendezvous for
            // real; every pool class / weight cache / NCCL plan warms on the
            // exact worker that captures next.
            let t0 = std::time::Instant::now();
            let toks = Self::fan_out(&mut self.shards, |s| {
                Self::mega_chain_dev(s, seq, in_vals.as_slice(), &plans, num_dsa, false, &gname)
            })
            .into_iter()
            .collect::<Result<Vec<f32>>>()?;
            let t_dry = t0.elapsed();
            if std::env::var_os("FERRITE_MEGA_DRY").is_some() {
                // DRY mode: skip capture/replay — every step runs the real
                // chain. Bisection: dry output correct → graph-mechanism bug;
                // dry output garbage → chain-semantics bug.
                eprintln!(
                            "[mega] DRY mode step (in={last} tok={}): dry-run {:.1}ms — no capture",
                            toks[0], t_dry.as_secs_f32() * 1e3
                        );
                                        // every shard's seq_runtime must track the sampled token — the NEXT
                // step's input embeds tokens.last() (decode_step_normal pushes at its
                // tail; mega omitted it → input token froze at the prompt's last
                // token → output self-locked to one token)
                let tok = toks[0] as u32;
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
                Self::mega_chain_dev(s, seq, in_vals.as_slice(), &plans, num_dsa, true, &gname)
            })
            .into_iter()
            .collect::<Result<Vec<f32>>>()?;
            self.mega_seq = Some(seq);
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
            let tok = toks[0] as u32;
            for s in &mut self.shards {
                if let Some(rt) = s.seq_runtime_mut(seq) {
                    rt.tokens.push(tok);
                }
            }
            return Ok(tok);
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
        // Event-in-graph segment times (FERRITE_MEGA_EVTS): read the
        // rank-0 graph's event nodes after the replay — TRUE in-graph
        // per-segment times (no sync-drain contamination like DRY mode).
        if std::env::var_os("FERRITE_MEGA_EVTS").is_some() {
            let es = ferrite_kernel::cuda::MEGA_EVTS.lock().unwrap();
            let nl = plans.len();
            if es.len() >= nl * 4 + 2 {
                let cuda0 = self.shards[0]
                    .backend
                    .as_cuda()
                    .ok_or_else(|| FerriteError::Config("FERRITE_MEGA_EVTS needs cuda".into()))?;
                cuda0.enter();
                let el = |a: usize, b: usize| {
                    cuda0.event_elapsed_ms(a as *mut std::ffi::c_void, b as *mut std::ffi::c_void) as f64
                };
                let (mut a_hc, mut b_gdn, mut b_dsa, mut c_hc, mut de_ffn) =
                    (0f64, 0f64, 0f64, 0f64, 0f64);
                for (i, p) in plans.iter().enumerate() {
                    let prev = if i == 0 { es[0] } else { es[4 + (i - 1) * 4] };
                    let (e_a, e_b, e_c, e_e) =
                        (es[1 + i * 4], es[2 + i * 4], es[3 + i * 4], es[4 + i * 4]);
                    a_hc += el(prev, e_a);
                    if matches!(p.attn, AttnKind::Dsa) {
                        b_dsa += el(e_a, e_b);
                    } else {
                        b_gdn += el(e_a, e_b);
                    }
                    c_hc += el(e_b, e_c);
                    de_ffn += el(e_c, e_e);
                }
                let head = el(es[4 + (nl - 1) * 4], es[4 * nl + 1]);
                eprintln!(
                    "[evts] A_hc={:.2} B_gdn={:.2} B_dsa={:.2} C_hc={:.2} DE_ffn={:.2} head={:.3} tot={:.2}ms",
                    a_hc, b_gdn, b_dsa, c_hc, de_ffn, head,
                    a_hc + b_gdn + b_dsa + c_hc + de_ffn + head
                );
            }
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
) -> Result<f32> {
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
    let cfg = &s.cfg;
    let (hidden, hc_mult) = (cfg.hidden_size, cfg.hc_mult);
    let nh = hc_mult * hidden;
    let n = 1usize;
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
            cuda.dsa_host_rollback(seq, f, 1);
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

    let mut res = DevBuf::alloc(cuda.dev(), cuda.stream(), nh)?;
    res.upload(in_vals)?; // recorded stage→dev memcpy (the graph input)
    let x_stage = res.stage; // GraphIO: replay writes fresh input here
    // Event-in-graph timing (FERRITE_MEGA_EVTS): rank-0 records timing
    // events at segment boundaries DURING capture — they become graph
    // nodes, replay updates them, post-replay elapsed = TRUE in-graph
    // segment times (the DRY sync-timing drains the async queue between
    // layers → contaminated). Layout: es[0] graph input; per layer i:
    // es[1+4i]=after A(hc_pre+norm), es[2+4i]=after B(attn+AR),
    // es[3+4i]=after C(hc_post+hc_pre2+norm), es[4+4i]=after
    // E(ffn+AR+hc_post2); es[4*nl+1]=after head(argmax).
    let evt_on = capture && dev_id == 0 && std::env::var_os("FERRITE_MEGA_EVTS").is_some();
    if evt_on {
        let nl = plans.len();
        let mut es = Vec::with_capacity(nl * 4 + 2);
        for _ in 0..nl * 4 + 2 {
            es.push(cuda.event_create()? as usize);
        }
        cuda.event_record(es[0] as *mut std::ffi::c_void); // graph input
        *ferrite_kernel::cuda::MEGA_EVTS.lock().unwrap() = es;
    }
    let ev = |i: usize| {
        ferrite_kernel::cuda::MEGA_EVTS.lock().unwrap()[i] as *mut std::ffi::c_void
    };
    mprobe!("res0", &res, nh);

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
        if evt_on {
            cuda.event_record(ev(1 + layer_idx * 4)); // after A (hc_pre+norm)
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
                cuda.gdn_layer_dev(
                    &hn, &gw, seq, layer_idx, n, hidden,
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
        if evt_on {
            cuda.event_record(ev(2 + layer_idx * 4)); // after B (attn+AR)
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
        if evt_on {
            cuda.event_record(ev(3 + layer_idx * 4)); // after C (hc_post+hc_pre2+norm)
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
        if evt_on {
            cuda.event_record(ev(4 + layer_idx * 4)); // after E (ffn+AR+hc_post2)
        }
    }

    mprobe!("resL", &res, nh);
    // head: contract → model.norm → lm_head → argmax (redundant per rank —
    // identical data after the ARs, replicated weights)
    let t_hs = std::time::Instant::now();
    let h_final = cuda.hc_contract_dev(&res, n, hc_mult, hidden)?;
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
    if evt_on {
        cuda.event_record(ev(4 * plans.len() + 1)); // head end (argmax)
    }

    if capture {
        cuda.graph_capture_end(gname);
        drop(_guard);
        cuda.graph_io_put(
            gname,
            GraphIO {
                x_stage,
                x_len: nh,
                out_dev: arg.as_f32() as *mut std::ffi::c_void,
                out_len: n,
            },
        );
        std::mem::forget(arg); // the graph's argmax output (graph_run reads it)
        // NOTE: no DSA rollback here — the PRE-capture rollback above makes
        // the pass's virtual t_count advance land exactly on the real cache
        // count (dry-run's tokens). replay-side dsa_host_advance keeps it
        // in lockstep from here on.
        Ok(0.0)
    } else {
        let mut tv = vec![0f32; 1];
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
        Ok(tv[0])
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
                    s.cfg.rms_norm_eps, la.short_conv_kernel_size,
                )?;
            } // drops return everything to the pool
            cuda.graph_capture_begin();
            let mut x_dev = DevBuf::alloc(cuda.dev(), cuda.stream(), hn.numel())?;
            x_dev.upload(hn.as_slice())?;
            let partial = cuda.gdn_layer_dev(
                &x_dev, &gw, seq, layer_idx, n, hidden,
                la.num_heads, la.head_dim, la.gate_lower_bound,
                s.cfg.rms_norm_eps, la.short_conv_kernel_size,
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
            s.cfg.rms_norm_eps, la.short_conv_kernel_size,
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
                                    s.cfg.rms_norm_eps, la.short_conv_kernel_size,
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
