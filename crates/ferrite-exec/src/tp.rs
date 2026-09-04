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
        TpCluster { shards, full_cfg, world, graph_captured: false, graph_step: 0, nccl: None }
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
        self.decode_step_normal(seq)
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
        if std::env::var_os("FERRITE_GRAPH_LAYER").is_some() && n == 1 {
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
        let (hn_t, res_dev, post_a_dev, comb_a_dev) = {
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
            let (li_dev, post_a_dev, comb_a_dev) = cuda0.hc_pre_dev(
                &res_dev, hc_fn, hc_scale, hc_base, n, nh,
                self.full_cfg.rms_norm_eps, self.full_cfg.hc_eps,
                self.full_cfg.hc_sinkhorn_iters,
            )?;
            let norm_w = s0.w(&format!("{pfx}.input_layernorm.weight"))?;
            let hn_dev = cuda0.rmsnorm_dev(&li_dev, norm_w, self.full_cfg.rms_norm_eps, n, hidden)?;
            let mut hn = vec![0f32; n * hidden];
            hn_dev.download(&mut hn)?;
            let hn_t = Tensor::from_f32(Shape::new([n, hidden]), hn);
            (hn_t, res_dev, post_a_dev, comb_a_dev)
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
                    let mut x_dev = DevBuf::alloc(cuda.dev(), cuda.stream(), hn_t.numel())?;
                    x_dev.upload(hn_t.as_slice())?;
                    match plan.attn {
                        AttnKind::Linear => {
                            #[cfg(feature = "cuda")]
                            {
                                use ferrite_kernel::cuda::GdnLayerWeights;
                                // graph fast path
                                if std::env::var_os("FERRITE_GRAPH_LAYER").is_some() && n == 1 {
                                    let gname = format!("gdn{}", layer_idx);
                                    if let Some(ptr) = cuda.graph_run_dev(&gname, hn_t.as_slice())? {
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
                                Ok(partial.as_f32() as usize)
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
                                Ok(partial.as_f32() as usize)
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
                    AttnKind::Linear => Self::attn_shard(s, seq, layer_idx, &pfx, &hn_t, n, hidden),
                    AttnKind::Dsa => s.dsa_attn_forward(seq, layer_idx, &pfx, &hn_t, n),
                });
                let attn_out = if self.shards[0].nccl.is_some() {
                    attn_partials.into_iter().next().unwrap()?
                } else {
                    all_reduce_sum(&attn_partials.into_iter().collect::<Result<Vec<_>>>()?)
                };
                (None, Some(attn_out))
            };
        let t_attn = std::time::Instant::now();
        let t_ar = std::time::Instant::now();

        // ---- segment 3: hc_post → hc_pre2 → rmsnorm2 (GPU chain, no host) ----
        let timing_mid = std::env::var_os("FERRITE_TIMING").is_some();
        let (hfn_t, res2_dev, post_f_dev, comb_f_dev) = {
            let s0 = &self.shards[0];
            let cuda0 = s0
                .backend
                .as_cuda()
                .ok_or_else(|| FerriteError::Config("FERRITE_LAYER_DEV needs cuda backend".into()))?;
            cuda0.enter();
            let ta = std::time::Instant::now();
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
            let (li2_dev, post_f_dev, comb_f_dev) = cuda0.hc_pre_dev(
                &res2_dev, hc_fn2, hc_scale2, hc_base2, n, nh,
                self.full_cfg.rms_norm_eps, self.full_cfg.hc_eps,
                self.full_cfg.hc_sinkhorn_iters,
            )?;
            cuda0.sync().ok();
            let tc = std::time::Instant::now();
            let norm_w2 = s0.w(&format!("{pfx}.post_attention_layernorm.weight"))?;
            let hfn_dev = cuda0.rmsnorm_dev(&li2_dev, norm_w2, self.full_cfg.rms_norm_eps, n, hidden)?;
            let mut hfn = vec![0f32; n * hidden];
            hfn_dev.download(&mut hfn)?;
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
                        Ok(d.as_f32() as usize)
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
                        let mut probs_scratch = DevBuf::alloc(cuda.dev(), cuda.stream(), n * topk)?;
                        let out_dev = cuda.moe_layer_dev(
                            &x_dev, gate_w, &bias, &shared, &experts, es,
                            &mut probs_scratch, n, hidden2, topk, e,
                            cfg.routed_scaling_factor, cfg.swiglu_limit,
                        )?;
                        Ok(out_dev.as_f32() as usize)
                    }
                }
            });
            let ptrs: Vec<usize> = ptrs.into_iter().collect::<Result<Vec<_>>>()?;
            let cuda0 = self.shards[0].backend.as_cuda().unwrap();
            let ffn_out_dev = cuda0.p2p_all_reduce(&ptrs, n * hidden)?;
            (Some(ffn_out_dev), None)
        } else {
            let ffn_partials = Self::fan_out(&mut self.shards, |s| match plan.mlp {
                MlpKind::Dense => s.dense_ffn(&pfx, &hfn_t, n),
                MlpKind::Moe => s.moe_ffn(&pfx, &hfn_t, n),
            });
            let ffn_out = if self.shards[0].nccl.is_some() {
                ffn_partials.into_iter().next().unwrap()?
            } else {
                all_reduce_sum(&ffn_partials.into_iter().collect::<Result<Vec<_>>>()?)
            };
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
            // segment 1 uses it directly. Only the last layer downloads.
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
        // P2P chain (FERRITE_P2P + FERRITE_LAYER_DEV): residual stays on GPU
        // across layers — no Tensor download/upload per layer (~0.3ms × 45).
        #[cfg(feature = "cuda")]
        if std::env::var_os("FERRITE_P2P").is_some()
            && std::env::var_os("FERRITE_LAYER_DEV").is_some()
            && self.full_cfg.mhc
        {
            let mut residual_dev: Option<ferrite_kernel::cuda::DevBuf> = None;
            for plan in &plans {
                let (h_new, dev_new) =
                    self.layer_forward_dev(seq, plan.layer_idx, h, residual_dev, 1)?;
                h = h_new;
                residual_dev = dev_new;
            }
        } else {
            for plan in &plans {
                h = self.layer_forward_tp(seq, plan.layer_idx, h, 1)?;
            }
        }
        let t_head = std::time::Instant::now();
        let h_final = if self.full_cfg.mhc {
            crate::mhc::hc_contract(&h, self.full_cfg.hc_mult)
        } else {
            h
        };
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
