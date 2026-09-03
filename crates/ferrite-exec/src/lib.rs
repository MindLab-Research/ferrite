//! ferrite-exec: the engine — wires model, kernel backend, state pools,
//! batching and PDAF scheduling into a runnable GLM-5.3-Flash inference
//! loop. `Engine<B: KernelBackend>` is monomorphised per backend.
//!
//! CPU golden path notes:
//! - Linear layers: full projection chain + causal short conv + chunkwise
//!   Gated DeltaNet via the kernel trait, states kept inline per (seq, layer).
//! - DSA layers: expanded (absorbed-at-load-time) K/V cache — mathematically
//!   equivalent to latent MLA; the latent pool layout is the B300 path.
//! - MHC: standard residual approximation on the golden path (exact sinkhorn
//!   mixing is a B300 calibration TODO).
//! - MoE: noaux-tc routing + grouped expert loop + shared expert.

use std::collections::HashMap;

use ferrite_batch::BatchScheduler;
use ferrite_kernel::KernelBackend;
use ferrite_model::{build_layer_plans, AttnKind, Glm53FlashConfig, LayerType, MlpKind, Weights};
use ferrite_scheduler::{PdafRouter, StaticPlan};
use ferrite_types::{DType, FerriteError, Result, Shape, Tensor};

pub mod graph;
pub mod mhc;
pub mod distributed;
pub mod tp;
pub use graph::GraphRunner;

/// Per-sequence runtime state (CPU golden path).
#[derive(Debug, Clone)]
pub struct DsaLayerCache {
    pub k_nope: Vec<f32>, // [t, heads, dk]
    pub v: Vec<f32>,      // [t, heads, dv]
    pub k_idx: Vec<f32>,  // [t, iproj]
}

#[derive(Debug, Clone)]
pub struct SeqRuntime {
    pub tokens: Vec<u32>,
    pub dsa_caches: Vec<DsaLayerCache>,
}

#[derive(Debug, Clone, Default)]
pub struct StepOutcome {
    pub prefilled: Vec<(u64, usize)>,
    pub decoded: Vec<(u64, u32)>,
    pub transfers: Vec<u64>,
    pub done: Vec<u64>,
}

pub struct Engine<B: KernelBackend> {
    pub cfg: Glm53FlashConfig,
    pub weights: Weights,
    pub backend: B,
    pub scheduler: BatchScheduler,
    pub router: PdafRouter,
    pub plan: StaticPlan,
    seqs: HashMap<u64, SeqRuntime>,
    /// Inline linear-attention states: (seq, layer) -> [h, dk, dv].
    /// (Mirrors HybridStatePool's linear slab; the pool is the B300 path.)
    linear_states: HashMap<(u64, usize), Vec<f32>>,
    /// Inline conv tails: (seq, layer) -> [ch, hist].
    conv_tails: HashMap<(u64, usize), Vec<f32>>,
    /// Outputs of finished (reaped) sequences.
    finished: HashMap<u64, Vec<u32>>,
    /// TP: this engine's routed-expert slice (EP-style, inclusive-exclusive).
    /// `None` = unsharded (owns all experts).
    pub tp_expert_range: Option<(usize, usize)>,
    /// TP: world size. Divides the shared-expert intermediate dim (the shared
    /// expert is row/column sharded like a dense MLP; routed experts are
    /// sliced whole). 1 = unsharded.
    pub tp_world: usize,
}

impl<B: KernelBackend> Engine<B> {
    pub fn new(cfg: Glm53FlashConfig, weights: Weights, backend: B) -> Self {
        let scheduler = BatchScheduler::new(64, 4096);
        let router = PdafRouter::new(&cfg);
        let plan = router.plan.clone();
        Engine {
            cfg,
            weights,
            backend,
            scheduler,
            router,
            plan,
            seqs: HashMap::new(),
            linear_states: HashMap::new(),
            conv_tails: HashMap::new(),
            finished: HashMap::new(),
            tp_expert_range: None,
            tp_world: 1,
        }
    }

    pub fn submit(&mut self, prompt: Vec<u32>, max_new_tokens: usize) -> Result<u64> {
        let eos = self.cfg.vocab_size.saturating_sub(1) as u32;
        let id = self.scheduler.submit(prompt.clone(), eos, max_new_tokens)?;
        let n_dsa = self
            .cfg
            .layer_types
            .iter()
            .filter(|t| matches!(t, LayerType::DeepseekSparseAttention))
            .count();
        self.seqs.insert(
            id,
            SeqRuntime {
                tokens: prompt,
                dsa_caches: (0..n_dsa).map(|_| DsaLayerCache { k_nope: Vec::new(), v: Vec::new(), k_idx: Vec::new() }).collect(),
            },
        );
        Ok(id)
    }

    pub(crate) fn dsa_dims(&self) -> (usize, usize, usize, usize) {
        let d = &self.cfg.dsa;
        (d.num_attention_heads, d.qk_nope_head_dim, d.v_head_dim, d.index_n_heads * d.index_head_dim)
    }

    pub(crate) fn w(&self, name: &str) -> Result<&Tensor> {
        self.weights
            .get(name)
            .ok_or_else(|| FerriteError::Config(format!("missing weight: {name}")))
    }

    pub(crate) fn embed(&self, tokens: &[u32]) -> Tensor {
        let table = &self.weights["model.embed_tokens.weight"];
        let hidden = self.cfg.hidden_size;
        let mut data = Vec::with_capacity(tokens.len() * hidden);
        for &t in tokens {
            let row = &table.as_slice()[(t as usize) * hidden..(t as usize + 1) * hidden];
            data.extend_from_slice(row);
        }
        Tensor::from_f32(Shape::new([tokens.len(), hidden]), data)
    }

    /// Create the per-seq runtime (tokens + DSA caches) if absent. Shared by
    /// prefill_chunk and the TP / distributed cluster drivers.
    pub(crate) fn ensure_seq(&mut self, seq: u64, initial_tokens: &[u32]) {
        if self.seqs.contains_key(&seq) {
            return;
        }
        let n_dsa = self
            .cfg
            .layer_types
            .iter()
            .filter(|t| matches!(t, LayerType::DeepseekSparseAttention))
            .count();
        self.seqs.insert(
            seq,
            SeqRuntime {
                tokens: initial_tokens.to_vec(),
                dsa_caches: (0..n_dsa)
                    .map(|_| DsaLayerCache { k_nope: Vec::new(), v: Vec::new(), k_idx: Vec::new() })
                    .collect(),
            },
        );
    }

    pub(crate) fn seq_runtime(&self, seq: u64) -> Option<&SeqRuntime> {
        self.seqs.get(&seq)
    }

    pub(crate) fn seq_runtime_mut(&mut self, seq: u64) -> Option<&mut SeqRuntime> {
        self.seqs.get_mut(&seq)
    }

    /// Run one engine step: schedule → prefill (P) → decode (D) → sample.
    pub fn step(&mut self) -> Result<StepOutcome> {
        let batch = self.scheduler.next_batch();
        let ids: Vec<u64> = self.seqs.keys().copied().collect();
        let mut info: HashMap<u64, (usize, usize, usize)> = HashMap::new();
        for id in ids.iter() {
            if let Ok(s) = self.scheduler.seq(*id) {
                info.insert(*id, (s.prompt.len(), s.prefilled, s.context_len()));
            }
        }
        let step = self.router.route(&batch, &|id| *info.get(&id).unwrap());
        let mut out = StepOutcome::default();
        for w in &step.prefill {
            let (start, chunk) = {
                let s = self.scheduler.seq(w.seq)?;
                (s.prefilled, w.chunk_tokens)
            };
            let tokens: Vec<u32> = {
                let s = self.scheduler.seq(w.seq)?;
                s.prompt[start..start + chunk].to_vec()
            };
            self.prefill_chunk(w.seq, &tokens)?;
            self.scheduler.advance_prefill(w.seq, chunk)?;
            out.prefilled.push((w.seq, chunk));
        }
        for t in &step.transfers {
            out.transfers.push(t.seq);
        }
        self.scheduler.post_step(&batch);
        for d in &step.decode {
            let token = self.decode_step(d.seq)?;
            self.scheduler.record_token(d.seq, token)?;
            {
                let s = self.seqs.get_mut(&d.seq).unwrap();
                s.tokens.push(token);
                if let Some(cache) = s.dsa_caches.first_mut() {
                    let _ = cache; // DSA cache grows lazily in decode too
                }
            }
            out.decoded.push((d.seq, token));
        }
        // snapshot finished outputs BEFORE reaping (reap removes the seqs)
        for s in self.scheduler.finished() {
            self.finished.insert(s.id, s.output.clone());
        }
        out.done = self.scheduler.reap_finished();
        for id in &out.done {
            self.linear_states.retain(|(s, _), _| s != id);
            self.conv_tails.retain(|(s, _), _| s != id);
            self.seqs.remove(id);
        }
        Ok(out)
    }

    pub fn run_until_done(&mut self, seq: u64) -> Result<Vec<u32>> {
        loop {
            if let Some(o) = self.finished.get(&seq) {
                return Ok(o.clone());
            }
            self.step()?;
        }
    }

    /// Query a finished (possibly already reaped) sequence's output.
    pub fn finished_output(&self, seq: u64) -> Option<Vec<u32>> {
        self.finished.get(&seq).cloned()
    }

    pub fn has_finished(&self, seq: u64) -> bool {
        self.finished.contains_key(&seq)
    }

    /// Export a sequence's inline states (linear GatedDeltaNet states, conv
    /// tails, DSA caches) — for the distributed (CP/DCP) path. The engine's
    /// copies are removed (moved out).
    #[allow(clippy::type_complexity)]
    pub fn take_seq_states(
        &mut self,
        seq: u64,
    ) -> (
        HashMap<(u64, usize), Vec<f32>>,
        HashMap<(u64, usize), Vec<f32>>,
        HashMap<u64, Vec<DsaLayerCache>>,
    ) {
        let mut linear = HashMap::new();
        let mut convs = HashMap::new();
        self.linear_states.retain(|&(s, l), v| {
            if s == seq {
                linear.insert((s, l), v.clone());
                false
            } else {
                true
            }
        });
        self.conv_tails.retain(|&(s, l), v| {
            if s == seq {
                convs.insert((s, l), v.clone());
                false
            } else {
                true
            }
        });
        let dsa = match self.seqs.remove(&seq) {
            Some(rt) => {
                let mut m = HashMap::new();
                m.insert(seq, rt.dsa_caches);
                m
            }
            None => HashMap::new(),
        };
        (linear, convs, dsa)
    }

    // ================= P phase =================

    pub fn prefill_chunk(&mut self, seq: u64, chunk_tokens: &[u32]) -> Result<()> {
        // ensure seq runtime exists (the distributed PdafCluster path passes
        // externally-assigned seq ids without going through submit)
        self.ensure_seq(seq, chunk_tokens);
        let h0 = self.embed(chunk_tokens);
        let mut h = if self.cfg.mhc {
            crate::mhc::hc_expand(&h0, self.cfg.hc_mult)
        } else {
            h0
        };
        let plans = build_layer_plans(&self.cfg);
        for plan in &plans {
            h = self.layer_forward(seq, plan.layer_idx, h, chunk_tokens.len())?;
        }
        let _ = h; // final logits are only consumed by decode
        Ok(())
    }

    // ================= D phase =================

    fn decode_step(&mut self, seq: u64) -> Result<u32> {
        let last = {
            let s = self.seqs.get(&seq).ok_or_else(|| FerriteError::Config("missing seq".into()))?;
            *s.tokens.last().ok_or_else(|| FerriteError::Config("empty context".into()))?
        };
        let h0 = self.embed(&[last]);
        let mut h = if self.cfg.mhc {
            crate::mhc::hc_expand(&h0, self.cfg.hc_mult)
        } else {
            h0
        };
        let plans = build_layer_plans(&self.cfg);
        for plan in &plans {
            h = self.layer_forward(seq, plan.layer_idx, h, 1)?;
        }
        let h_final = if self.cfg.mhc {
            crate::mhc::hc_contract(&h, self.cfg.hc_mult)
        } else {
            h
        };
        let h_final = self.rmsnorm(&h_final, "model.norm.weight")?;
        let logits = self.project(&h_final, "lm_head.weight")?;
        let mut out = Tensor::zeros(Shape::new([1]), DType::F32);
        self.backend.argmax_lastdim(&logits, &mut out)?;
        Ok(out.as_slice()[0] as u32)
    }

    // ================= layer forward (shared P/D) =================

    fn layer_forward(
        &mut self,
        seq: u64,
        layer_idx: usize,
        h: Tensor,
        n: usize,
    ) -> Result<Tensor> {
        if self.cfg.mhc {
            return self.layer_forward_mhc(seq, layer_idx, h, n);
        }
        let plans = build_layer_plans(&self.cfg);
        let plan = &plans[layer_idx];
        let pfx = format!("model.layers.{layer_idx}");
        let hidden = self.cfg.hidden_size;
        let hn = self.rmsnorm(&h, &format!("{pfx}.input_layernorm.weight"))?;
        let attn_out = match plan.attn {
            AttnKind::Linear => self.linear_attn_forward(seq, layer_idx, &pfx, &hn, n)?,
            AttnKind::Dsa => self.dsa_attn_forward(seq, layer_idx, &pfx, &hn, n)?,
        };
        let mut h2 = Tensor::from_f32(
            Shape::new([n, hidden]),
            (0..n * hidden)
                .map(|i| h.as_slice()[i] + attn_out.as_slice()[i])
                .collect(),
        );
        // post-attn norm for the FFN half
        let hfn = self.rmsnorm(&h2, &format!("{pfx}.input_layernorm.weight"))?;
        let ffn_out = match plan.mlp {
            MlpKind::Dense => self.dense_ffn(&pfx, &hfn, n)?,
            MlpKind::Moe => self.moe_ffn(&pfx, &hfn, n)?,
        };
        h2 = Tensor::from_f32(
            Shape::new([n, hidden]),
            (0..n * hidden)
                .map(|i| h2.as_slice()[i] + ffn_out.as_slice()[i])
                .collect(),
        );
        Ok(h2)
    }

    /// MHC layer forward: 4-flow residual stream (hc_mult flows).
    /// `residual_flat: [n, hc_mult*hidden]` in and out (flat 2-D).
    fn layer_forward_mhc(
        &mut self,
        seq: u64,
        layer_idx: usize,
        residual_flat: Tensor,
        n: usize,
    ) -> Result<Tensor> {
        let plan = &build_layer_plans(&self.cfg)[layer_idx];
        let pfx = format!("model.layers.{layer_idx}");
        let (hidden, hc_mult) = (self.cfg.hidden_size, self.cfg.hc_mult);

        // ---- attention half: hc_pre -> norm -> attention -> hc_post ----
        let hc_fn = self.w(&format!("{pfx}.hc_attn_fn"))?.clone();
        let hc_scale = self.w(&format!("{pfx}.hc_attn_scale"))?.clone();
        let hc_base = self.w(&format!("{pfx}.hc_attn_base"))?.clone();
        let (li, post_a, comb_a) = crate::mhc::hc_pre(
            &residual_flat,
            &hc_fn,
            &hc_scale,
            &hc_base,
            self.cfg.rms_norm_eps,
            self.cfg.hc_eps,
            self.cfg.hc_sinkhorn_iters,
        );
        let hn = self.rmsnorm(&li, &format!("{pfx}.input_layernorm.weight"))?;
        let attn_out = match plan.attn {
            AttnKind::Linear => self.linear_attn_forward(seq, layer_idx, &pfx, &hn, n)?,
            AttnKind::Dsa => self.dsa_attn_forward(seq, layer_idx, &pfx, &hn, n)?,
        };
        let res3 = Tensor::from_f32(
            Shape::new([n, hc_mult, hidden]),
            residual_flat.as_slice().to_vec(),
        );
        let res2 = crate::mhc::hc_post(&attn_out, &res3, &post_a, &comb_a);

        // ---- ffn half ----
        let hc_fn2 = self.w(&format!("{pfx}.hc_ffn_fn"))?.clone();
        let hc_scale2 = self.w(&format!("{pfx}.hc_ffn_scale"))?.clone();
        let hc_base2 = self.w(&format!("{pfx}.hc_ffn_base"))?.clone();
        let res2_flat = Tensor::from_f32(
            Shape::new([n, hc_mult * hidden]),
            res2.as_slice().to_vec(),
        );
        let (li2, post_f, comb_f) = crate::mhc::hc_pre(
            &res2_flat,
            &hc_fn2,
            &hc_scale2,
            &hc_base2,
            self.cfg.rms_norm_eps,
            self.cfg.hc_eps,
            self.cfg.hc_sinkhorn_iters,
        );
        let hfn = self.rmsnorm(&li2, &format!("{pfx}.input_layernorm.weight"))?;
        let ffn_out = match plan.mlp {
            MlpKind::Dense => self.dense_ffn(&pfx, &hfn, n)?,
            MlpKind::Moe => self.moe_ffn(&pfx, &hfn, n)?,
        };
        let res3b = Tensor::from_f32(
            Shape::new([n, hc_mult, hidden]),
            res2_flat.as_slice().to_vec(),
        );
        let res_out = crate::mhc::hc_post(&ffn_out, &res3b, &post_f, &comb_f);
        Ok(Tensor::from_f32(
            Shape::new([n, hc_mult * hidden]),
            res_out.as_slice().to_vec(),
        ))
    }

    pub(crate) fn rmsnorm(&self, x: &Tensor, weight_name: &str) -> Result<Tensor> {
        let w = self.w(weight_name)?;
        let mut out = Tensor::zeros(x.shape.clone(), x.dtype);
        self.backend.rmsnorm(x, w, self.cfg.rms_norm_eps, &mut out)?;
        Ok(out)
    }

    pub(crate) fn project(&self, x: &Tensor, weight_name: &str) -> Result<Tensor> {
        let w = self.w(weight_name)?;
        let rows = w.shape.0[0];
        let mut out = Tensor::zeros(Shape::new([x.shape.0[0], rows]), x.dtype);
        self.backend.matmul(x, w, None, &mut out)?;
        Ok(out)
    }

    // ---------------- GatedDeltaNet linear attention ----------------

    pub(crate) fn linear_attn_forward(
        &mut self,
        seq: u64,
        layer_idx: usize,
        pfx: &str,
        x: &Tensor,
        n: usize,
    ) -> Result<Tensor> {
        let la = &self.cfg.linear_attn;
        let (h, dk) = (la.num_heads, la.head_dim);
        let proj = h * dk;
        let conv_channels = 3 * proj;
        let hist = la.short_conv_kernel_size.saturating_sub(1);
        // projections
        let qkv = self.project(x, &format!("{pfx}.self_attn.qkv_proj.weight"))?;
        let b_raw = self.project(x, &format!("{pfx}.self_attn.b_proj.weight"))?;
        let fa = self.project(x, &format!("{pfx}.self_attn.f_a_proj.weight"))?;
        let fb = self.project(&fa, &format!("{pfx}.self_attn.f_b_proj.weight"))?;
        let ga = self.project(x, &format!("{pfx}.self_attn.g_a_proj.weight"))?;
        let gb = self.project(&ga, &format!("{pfx}.self_attn.g_b_proj.weight"))?;
        // causal short conv with carried tail (CPU golden: exact window)
        let conv_w = self.w(&format!("{pfx}.self_attn.qkv_conv1d.weight"))?;
        let prev: Vec<f32> = self
            .conv_tails
            .get(&(seq, layer_idx))
            .cloned()
            .unwrap_or_else(|| vec![0.0; conv_channels * hist]);
        let state_in = Tensor::from_f32(Shape::new([conv_channels, hist.max(1)]), prev);
        let mut conv_out = Tensor::zeros(Shape::new([n, conv_channels]), DType::F32);
        let mut state_out = Tensor::zeros(Shape::new([conv_channels, hist.max(1)]), DType::F32);
        self.backend.causal_conv1d(&qkv, conv_w, &state_in, &mut conv_out, &mut state_out)?;
        self.conv_tails
            .insert((seq, layer_idx), state_out.as_slice().to_vec());
        // split q/k/v
        let split = |i: usize| -> Tensor {
            let mut d = Vec::with_capacity(n * proj);
            for t in 0..n {
                let s = &conv_out.as_slice()[t * conv_channels + i * proj..t * conv_channels + (i + 1) * proj];
                d.extend_from_slice(s);
            }
            Tensor::from_f32(Shape::new([n, h, dk]), d)
        };
        let q = split(0);
        let k = split(1);
        let v = split(2);
        let beta = Tensor::from_f32(
            Shape::new([n, h]),
            b_raw.as_slice().iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(),
        );
        let gate = Tensor::from_f32(
            Shape::new([n, h, dk]),
            fb.as_slice().iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(),
        );
        let a_log = self.w(&format!("{pfx}.self_attn.A_log"))?;
        // recurrent state
        let state = self
            .linear_states
            .get(&(seq, layer_idx))
            .cloned()
            .unwrap_or_else(|| vec![0.0; h * dk * dk]);
        let state_in = Tensor::from_f32(Shape::new([h, dk, dk]), state);
        let mut core = Tensor::zeros(Shape::new([n, h, dk]), DType::F32);
        let mut state_out = Tensor::zeros(Shape::new([h, dk, dk]), DType::F32);
        self.backend.gated_deltanet_chunk(
            &q, &k, &v, &beta, &gate, a_log, &state_in, &mut core, &mut state_out,
        )?;
        self.linear_states
            .insert((seq, layer_idx), state_out.as_slice().to_vec());
        // gated output norm: flatten core [n,h,dk] -> [n,proj] first (the
        // o_norm weight spans the whole head*dim axis), then norm * (1+g)
        let o_norm_w = self.w(&format!("{pfx}.self_attn.o_norm.weight"))?;
        let core_flat = Tensor::from_f32(Shape::new([n, proj]), core.as_slice().to_vec());
        let mut normed = Tensor::zeros(Shape::new([n, proj]), DType::F32);
        self.backend
            .gated_rmsnorm(&core_flat, &gb, o_norm_w, self.cfg.rms_norm_eps, &mut normed)?;
        self.project(&normed, &format!("{pfx}.self_attn.o_proj.weight"))
    }

    // ---------------- DSA attention (expanded-cache golden path) ----------------

    pub(crate) fn dsa_attn_forward(
        &mut self,
        seq: u64,
        layer_idx: usize,
        pfx: &str,
        x: &Tensor,
        n: usize,
    ) -> Result<Tensor> {
        let d = &self.cfg.dsa;
        let (h, dk, dv, ip) = self.dsa_dims();
        let qa = self.project(x, &format!("{pfx}.self_attn.q_a_proj.weight"))?;
        let qa = self.rmsnorm(&qa, &format!("{pfx}.self_attn.q_a_layernorm.weight"))?;
        let qb = self.project(&qa, &format!("{pfx}.self_attn.q_b_proj.weight"))?;
        let q = Tensor::from_f32(Shape::new([n, h, dk]), qb.as_slice().to_vec());
        let latent = self.project(x, &format!("{pfx}.self_attn.kv_a_proj_with_mqa.weight"))?;
        let latent_ln = self.rmsnorm(&latent, &format!("{pfx}.self_attn.kv_a_layernorm.weight"))?;
        let kvb = self.project(&latent_ln, &format!("{pfx}.self_attn.kv_b_proj.weight"))?;
        let qi = self.project(x, &format!("{pfx}.self_attn.indexer_q_proj.weight"))?;
        let ki = self.project(x, &format!("{pfx}.self_attn.indexer_k_proj.weight"))?;
        let ki = self.rmsnorm(&ki, &format!("{pfx}.self_attn.indexer_norm.weight"))?;
        let family_idx = self.dsa_family_index(layer_idx);
        {
            let s = self.seqs.get_mut(&seq).unwrap();
            let c = &mut s.dsa_caches[family_idx];
            let t0 = c.k_nope.len() / (h * dk);
            for t in 0..n {
                let off = (t0 + t) * h * dk;
                c.k_nope.resize(off + h * dk, 0.0);
                c.k_nope[off..off + h * dk].copy_from_slice(
                    &kvb.as_slice()[t * h * (dk + dv)..t * h * (dk + dv) + h * dk],
                );
                let voff = (t0 + t) * h * dv;
                c.v.resize(voff + h * dv, 0.0);
                c.v[voff..voff + h * dv].copy_from_slice(
                    &kvb.as_slice()[t * h * (dk + dv) + h * dk..t * h * (dk + dv) + h * (dk + dv)],
                );
                let ioff = (t0 + t) * ip;
                c.k_idx.resize(ioff + ip, 0.0);
                c.k_idx[ioff..ioff + ip].copy_from_slice(&ki.as_slice()[t * ip..(t + 1) * ip]);
            }
        }
        let (k_all, v_all, kidx_all, total) = {
            let s = self.seqs.get(&seq).unwrap();
            let c = &s.dsa_caches[family_idx];
            (c.k_nope.clone(), c.v.clone(), c.k_idx.clone(), c.k_nope.len() / (h * dk))
        };
        let k_nope = Tensor::from_f32(Shape::new([total, h, dk]), k_all);
        let v = Tensor::from_f32(Shape::new([total, h, dv]), v_all);
        let k_idx_all = Tensor::from_f32(Shape::new([total, ip]), kidx_all);
        let topk = d.index_topk.min(total);
        let mut idx = Tensor::zeros(Shape::new([n, topk]), DType::F32);
        self.backend.indexer_topk(&qi, &k_idx_all, topk, &mut idx)?;
        let mut out = Tensor::zeros(Shape::new([n, h, dv]), DType::F32);
        self.backend.sparse_mla_attn(&q, &k_nope, &v, &idx, &mut out)?;
        let flat = Tensor::from_f32(Shape::new([n, h * dv]), out.as_slice().to_vec());
        self.project(&flat, &format!("{pfx}.self_attn.o_proj.weight"))
    }

    fn dsa_family_index(&self, layer_idx: usize) -> usize {
        self.cfg
            .layer_types
            .iter()
            .take(layer_idx + 1)
            .filter(|t| matches!(t, LayerType::DeepseekSparseAttention))
            .count()
            - 1
    }

    // ---------------- FFN ----------------

    pub(crate) fn dense_ffn(&self, pfx: &str, x: &Tensor, n: usize) -> Result<Tensor> {
        let inter = self.cfg.intermediate_size;
        let gate = self.project(x, &format!("{pfx}.mlp.gate_proj.weight"))?;
        let up = self.project(x, &format!("{pfx}.mlp.up_proj.weight"))?;
        let mut gate_up = Tensor::zeros(Shape::new([n, 2 * inter]), x.dtype);
        {
            let gu = std::sync::Arc::get_mut(&mut gate_up.data).unwrap();
            for t in 0..n {
                gu[t * 2 * inter..t * 2 * inter + inter]
                    .copy_from_slice(&gate.as_slice()[t * inter..(t + 1) * inter]);
                gu[t * 2 * inter + inter..(t + 1) * 2 * inter]
                    .copy_from_slice(&up.as_slice()[t * inter..(t + 1) * inter]);
            }
        }
        let mut act = Tensor::zeros(Shape::new([n, inter]), x.dtype);
        self.backend.swiglu_limited(&gate_up, self.cfg.swiglu_limit, &mut act)?;
        self.project(&act, &format!("{pfx}.mlp.down_proj.weight"))
    }

    pub(crate) fn moe_ffn(&self, pfx: &str, x: &Tensor, n: usize) -> Result<Tensor> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden_size;
        let topk = cfg.num_experts_per_tok;
        let e = cfg.n_routed_experts;
        let logits = self.project(x, &format!("{pfx}.mlp.gate.weight"))?;
        let bias = Tensor::zeros(Shape::new([e]), DType::F32);
        let mut probs = Tensor::zeros(Shape::new([n, topk]), DType::F32);
        let mut ids = Tensor::zeros(Shape::new([n, topk]), DType::F32);
        self.backend
            .moe_route(&logits, &bias, topk, cfg.routed_scaling_factor, &mut probs, &mut ids)?;
        // shared expert: row/col-sharded across TP ranks (intermediate/world),
        // so its output is a partial sum — the caller all-reduces.
        let shared_inter = cfg.moe_intermediate_size / self.tp_world.max(1);
        let mut shared = Tensor::zeros(Shape::new([n, hidden]), x.dtype);
        self.expert_ffn_impl(x, &format!("{pfx}.mlp.shared_expert"), shared_inter, &mut shared)?;
        let mut out = Tensor::zeros(Shape::new([n, hidden]), x.dtype);
        let ov = std::sync::Arc::get_mut(&mut out.data).unwrap();
        // TP: each rank computes only its routed-expert slice; the union of
        // slices over ranks covers every (token, expert) pair exactly once.
        let (es, ee) = self.tp_expert_range.unwrap_or((0, e));
        for j in 0..topk {
            for eid in es..ee {
                let sel: Vec<usize> = (0..n).filter(|&t| ids.as_slice()[t * topk + j] as usize == eid).collect();
                if sel.is_empty() {
                    continue;
                }
                let m = sel.len();
                let mut gx = Tensor::zeros(Shape::new([m, hidden]), x.dtype);
                let gxd = std::sync::Arc::get_mut(&mut gx.data).unwrap();
                for (r, &t) in sel.iter().enumerate() {
                    gxd[r * hidden..(r + 1) * hidden].copy_from_slice(&x.as_slice()[t * hidden..(t + 1) * hidden]);
                }
                let mut eout = Tensor::zeros(Shape::new([m, hidden]), x.dtype);
                self.expert_ffn_into(&gx, &format!("{pfx}.mlp.experts.{eid}"), &mut eout)?;
                for (r, &t) in sel.iter().enumerate() {
                    let w = probs.as_slice()[t * topk + j];
                    for l in 0..hidden {
                        ov[t * hidden + l] += w * eout.as_slice()[r * hidden + l];
                    }
                }
            }
        }
        for i in 0..n * hidden {
            ov[i] += shared.as_slice()[i];
        }
        Ok(out)
    }

    fn expert_ffn(&self, x: &Tensor, pfx: &str) -> Result<Tensor> {
        let n = x.shape.0[0];
        let inter = self.cfg.moe_intermediate_size;
        let mut out = Tensor::zeros(Shape::new([n, self.cfg.hidden_size]), x.dtype);
        self.expert_ffn_impl(x, pfx, inter, &mut out)?;
        Ok(out)
    }

    fn expert_ffn_into(&self, x: &Tensor, pfx: &str, out: &mut Tensor) -> Result<()> {
        let inter = self.cfg.moe_intermediate_size;
        self.expert_ffn_impl(x, pfx, inter, out)
    }

    fn expert_ffn_impl(&self, x: &Tensor, pfx: &str, inter: usize, out: &mut Tensor) -> Result<()> {
        let n = x.shape.0[0];
        let gate = self.project(x, &format!("{pfx}.gate_proj.weight"))?;
        let up = self.project(x, &format!("{pfx}.up_proj.weight"))?;
        let mut gate_up = Tensor::zeros(Shape::new([n, 2 * inter]), x.dtype);
        let gu = std::sync::Arc::get_mut(&mut gate_up.data).unwrap();
        for t in 0..n {
            gu[t * 2 * inter..t * 2 * inter + inter].copy_from_slice(&gate.as_slice()[t * inter..(t + 1) * inter]);
            gu[t * 2 * inter + inter..(t + 1) * 2 * inter].copy_from_slice(&up.as_slice()[t * inter..(t + 1) * inter]);
        }
        let mut act = Tensor::zeros(Shape::new([n, inter]), x.dtype);
        self.backend.swiglu_limited(&gate_up, self.cfg.swiglu_limit, &mut act)?;
        let down = self.project(&act, &format!("{pfx}.down_proj.weight"))?;
        let od = std::sync::Arc::get_mut(&mut out.data).unwrap();
        od.copy_from_slice(down.as_slice());
        Ok(())
    }
}
