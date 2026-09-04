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

#[cfg(all(test, feature = "cuda"))]
mod gdn_equiv_test;
pub mod tp;
pub use graph::GraphRunner;

/// Per-sequence runtime state (CPU golden path).
/// Probe dump path. FERRITE_PROBE_DIR isolates concurrent runs (shared
/// /tmp collisions cross-contaminate dumps between people — default /tmp).
fn ppath(name: &str) -> String {
    std::env::var("FERRITE_PROBE_DIR").unwrap_or_else(|_| "/tmp".to_string()) + "/" + name
}

#[derive(Debug, Clone)]
pub struct DsaLayerCache {
    pub k_nope: Vec<f32>, // [t, heads, dk]
    pub v: Vec<f32>,      // [t, heads, dv]
    pub k_idx: Vec<f32>,  // [t, iproj]
    pub k_gate: Vec<f32>, // [t, idm] kpool gate scores (indexer compression)
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
    /// EOS token for generation stopping (real model: 154820 <|end|> etc.
    /// from generation_config.json; defaults to vocab_size-1 which is wrong
    /// for GLM-5.3-Flash — serve binaries should set this).
    pub eos_token: Option<u32>,
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
            eos_token: None,
        }
    }

    pub fn submit(&mut self, prompt: Vec<u32>, max_new_tokens: usize) -> Result<u64> {
        let eos = self.eos_token.unwrap_or(self.cfg.vocab_size.saturating_sub(1) as u32);
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
                dsa_caches: (0..n_dsa).map(|_| DsaLayerCache { k_nope: Vec::new(), v: Vec::new(), k_idx: Vec::new(), k_gate: Vec::new() }).collect(),
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
                    .map(|_| DsaLayerCache { k_nope: Vec::new(), v: Vec::new(), k_idx: Vec::new(), k_gate: Vec::new() })
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
        if std::env::var_os("FERRITE_TRACE_NAN").is_some()
            && li.as_slice().iter().any(|v| !v.is_finite())
        {
            eprintln!("[trace] NaN after hc_pre at layer {layer_idx}");
        }
        let hn = self.rmsnorm(&li, &format!("{pfx}.input_layernorm.weight"))?;
        let attn_out = match plan.attn {
            AttnKind::Linear => self.linear_attn_forward(seq, layer_idx, &pfx, &hn, n)?,
            AttnKind::Dsa => self.dsa_attn_forward(seq, layer_idx, &pfx, &hn, n)?,
        };
        if std::env::var_os("FERRITE_TRACE_NAN").is_some()
            && attn_out.as_slice().iter().any(|v| !v.is_finite())
        {
            eprintln!("[trace] NaN after attn (kind={:?}) at layer {layer_idx}", plan.attn);
        }
        if std::env::var_os("FERRITE_PROBE").is_some() && layer_idx == 3 && n > 1 {
            let b: Vec<u8> = attn_out.as_slice().iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&ppath("l3_attn_out.f32"), b).ok();
        }
        let res3 = Tensor::from_f32(
            Shape::new([n, hc_mult, hidden]),
            residual_flat.as_slice().to_vec(),
        );
        let res2 = crate::mhc::hc_post(&attn_out, &res3, &post_a, &comb_a);
        if std::env::var_os("FERRITE_TRACE_NAN").is_some()
            && res2.as_slice().iter().any(|v| !v.is_finite())
        {
            eprintln!("[trace] NaN after attn hc_post at layer {layer_idx}");
        }

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
        let hfn = self.rmsnorm(&li2, &format!("{pfx}.post_attention_layernorm.weight"))?;
        let ffn_out = match plan.mlp {
            MlpKind::Dense => self.dense_ffn(&pfx, &hfn, n)?,
            MlpKind::Moe => self.moe_ffn(&pfx, &hfn, n)?,
        };
        let res3b = Tensor::from_f32(
            Shape::new([n, hc_mult, hidden]),
            res2_flat.as_slice().to_vec(),
        );
        let res_out = crate::mhc::hc_post(&ffn_out, &res3b, &post_f, &comb_f);
        if std::env::var_os("FERRITE_TRACE_NAN").is_some() {
            let (mut mx, mut sum) = (0.0f32, 0.0f32);
            for v in res_out.as_slice() {
                if v.is_finite() { mx = mx.max(v.abs()); sum += v * v; }
            }
            eprintln!(
                "[trace] layer {layer_idx:2} after: max|v|={mx:.4} l2={:.4} attn_max={:.4}",
                sum.sqrt(),
                attn_out.as_slice().iter().fold(0.0f32, |a, v| a.max(v.abs()))
            );
        }
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
        if std::env::var_os("FERRITE_PROBE").is_some() && layer_idx == 0 && n > 1 {
            let wr = |path: &str, t: &Tensor| {
                let b: Vec<u8> = t.as_slice().iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(path, b).ok();
            };
            wr(&ppath("l0_gdn_fa.f32"), &fa);
            wr(&ppath("l0_gdn_fb.f32"), &fb);
        }
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
        // SiLU activation on the conv output (Glm5NextTextLinearAttention:
        // causal_conv1d_fn(..., activation="silu"))
        // PROBE (lib.rs — the LIVE linear_attn_forward; exec_lib.rs's is dead code)
        if std::env::var_os("FERRITE_GDN_PROBE").is_some() && layer_idx == 0 && n > 1 {
            let dir = std::env::var("FERRITE_PROBE_DIR").unwrap_or_else(|_| "/tmp/orion".into());
            let d = |name: &str, v: &[f32]| {
                let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/gdn_cpu_{name}_r{}.f32", ferrite_kernel::shard_idx()), b).ok();
            };
            d("x", x.as_slice());
            d("qkv", qkv.as_slice());
            d("conv", conv_out.as_slice());
            d("braw", b_raw.as_slice());
            d("fb", fb.as_slice());
            eprintln!("[gdn_probe] cpu L0 dumped x/qkv/conv/braw/fb (lib.rs live path)");
        }
        {
            let cv = std::sync::Arc::get_mut(&mut conv_out.data).expect("unique conv");
            for v in cv.iter_mut() {
                *v = *v / (1.0 + (-*v).exp());
            }
        }
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
        // per-head L2 norm on q/k (KDA: use_qk_l2norm_in_kernel=True)
        let l2norm_heads = |t: &Tensor| -> Tensor {
            let mut d = t.as_slice().to_vec();
            for i in 0..n * h {
                let s = &d[i * dk..(i + 1) * dk];
                let norm: f32 = s.iter().map(|v| v * v).sum::<f32>().sqrt();
                let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
                for j in 0..dk {
                    d[i * dk + j] *= inv;
                }
            }
            Tensor::from_f32(Shape::new([n, h, dk]), d)
        };
        let q = l2norm_heads(&q);
        let k = l2norm_heads(&k);
        // fla KDA: q = l2norm(q) * K^-0.5 (k is NOT scaled)
        let q = {
            let scale = (dk as f32).recip().sqrt();
            let d = q.as_slice().iter().map(|v| v * scale).collect::<Vec<f32>>();
            Tensor::from_f32(Shape::new([n, h, dk]), d)
        };
        let beta = Tensor::from_f32(
            Shape::new([n, h]),
            b_raw.as_slice().iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(),
        );
        // KDA forget gate (Glm5NextTextForgetGate):
        //   g = f_b(f_a(x)) + dt_bias  → [n, h*dk]
        //   decay = gate_lower_bound * sigmoid(exp(A_log_h) * g)  (per channel)
        let dt_bias = self.w(&format!("{pfx}.self_attn.dt_bias"))?;
        let a_log = self.w(&format!("{pfx}.self_attn.A_log"))?;
        let lb = self.cfg.linear_attn.gate_lower_bound;
        let mut gate = Tensor::zeros(Shape::new([n, h, dk]), DType::F32);
        {
            let gv = std::sync::Arc::get_mut(&mut gate.data).expect("unique gate");
            let al = a_log.as_slice();
            let fbv = fb.as_slice();
            let db = dt_bias.as_slice();
            let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
            for t in 0..n {
                for hd in 0..h {
                    let a = al[hd].exp();
                    for j in 0..dk {
                        let g = fbv[t * proj + hd * dk + j] + db[hd * dk + j];
                        gv[t * proj + hd * dk + j] = lb * sig(a * g);
                    }
                }
            }
        }
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
        if std::env::var_os("FERRITE_PROBE").is_some() && layer_idx == 0 && n > 1 {
            let wr = |p: &str, t: &Tensor| {
                let b: Vec<u8> = t.as_slice().iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(p, b).ok();
            };
            wr(&ppath("l0_gdn_q.f32"), &q);
            wr(&ppath("l0_gdn_k.f32"), &k);
            wr(&ppath("l0_gdn_gate.f32"), &gate);
            wr(&ppath("l0_gdn_core.f32"), &core);
            wr(&ppath("l0_gdn_beta.f32"), &beta);
        }
        // gated output norm: per-head RMSNorm over head_dim (the real
        // checkpoint's o_norm weight is [head_dim] = [128], applied
        // head-wise with the channel gate) — reshape [n,h,dk] → [n*h, dk].
        let o_norm_w = self.w(&format!("{pfx}.self_attn.o_norm.weight"))?;
        let core_rows = Tensor::from_f32(Shape::new([n * h, dk]), core.as_slice().to_vec());
        let gate_rows = Tensor::from_f32(Shape::new([n * h, dk]), gb.as_slice().to_vec());
        let mut normed = Tensor::zeros(Shape::new([n * h, dk]), DType::F32);
        self.backend.gated_rmsnorm(
            &core_rows,
            &gate_rows,
            o_norm_w,
            self.cfg.rms_norm_eps,
            &mut normed,
        )?;
        let normed = Tensor::from_f32(Shape::new([n, proj]), normed.as_slice().to_vec());
        let partial = self.project(&normed, &format!("{pfx}.self_attn.o_proj.weight"))?;
        // PROBE: core (gdn chunk out) + partial (o_proj out) — rank-isolated
        if std::env::var_os("FERRITE_GDN_PROBE").is_some() && layer_idx == 0 && n > 1 {
            let dir = std::env::var("FERRITE_PROBE_DIR").unwrap_or_else(|_| "/tmp/orion".into());
            let d = |name: &str, v: &[f32]| {
                let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/gdn_cpu_{name}_r{}.f32", ferrite_kernel::shard_idx()), b).ok();
            };
            d("core", core.as_slice());
            d("partial", partial.as_slice());
            d("q", q.as_slice());
            d("k", k.as_slice());
            d("beta", beta.as_slice());
            d("gate", gate.as_slice());
            d("v", v.as_slice());
            eprintln!("[gdn_probe] cpu L0 core/partial/q/k/beta/gate/v dumped r{}", ferrite_kernel::shard_idx());
        }
        Ok(partial)
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
        let (h, dk, dv, _ip) = self.dsa_dims();
        let ih = d.index_n_heads; // indexer heads (32)
        let idm = d.index_head_dim; // indexer key dim (128)
        let qa = self.project(x, &format!("{pfx}.self_attn.q_a_proj.weight"))?;
        let qa = self.rmsnorm(&qa, &format!("{pfx}.self_attn.q_a_layernorm.weight"))?;
        let qb = self.project(&qa, &format!("{pfx}.self_attn.q_b_proj.weight"))?;
        let q = Tensor::from_f32(Shape::new([n, h, dk]), qb.as_slice().to_vec());
        let latent = self.project(x, &format!("{pfx}.self_attn.kv_a_proj_with_mqa.weight"))?;
        let latent_ln = self.rmsnorm(&latent, &format!("{pfx}.self_attn.kv_a_layernorm.weight"))?;
        let kvb = self.project(&latent_ln, &format!("{pfx}.self_attn.kv_b_proj.weight"))?;
        // Real-checkpoint DSA indexer: wq_b projects per-head indexer queries
        // from the (normed) q_lora latent; wk projects shared index keys from
        // hidden; k_norm is an affine norm over the 128-dim index key;
        // weights_proj gives per-head score weights.
        let qi = self.project(&qa, &format!("{pfx}.self_attn.indexer.wq_b.weight"))?;
        if std::env::var_os("FERRITE_PROBE").is_some() && layer_idx == 3 && n > 1 {
            let wr = |path: &str, t: &Tensor| {
                let b: Vec<u8> = t.as_slice().iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(path, b).ok();
            };
            wr(&ppath("l3_dsa_qa.f32"), &qa);
            wr(&ppath("l3_dsa_qi.f32"), &qi);
            wr(&ppath("l3_dsa_q.f32"), &q);
            wr(&ppath("l3_dsa_kvb.f32"), &kvb);
            wr(&ppath("l3_dsa_hn.f32"), x);
        }
        let ki_raw = self.project(x, &format!("{pfx}.self_attn.indexer.wk.weight"))?;
        let kn_w = self.w(&format!("{pfx}.self_attn.indexer.k_norm.weight"))?;
        let kn_b = self.w(&format!("{pfx}.self_attn.indexer.k_norm.bias"))?;
        // LayerNorm(k_norm affine w/b) over the [n, idm] index keys.
        let mut ki = Tensor::zeros(ki_raw.shape.clone(), DType::F32);
        {
            let src = ki_raw.as_slice();
            let wv = kn_w.as_slice();
            let bv = kn_b.as_slice();
            let ov = std::sync::Arc::get_mut(&mut ki.data).expect("unique ki");
            for i in 0..n {
                let row = &src[i * idm..(i + 1) * idm];
                let mean: f32 = row.iter().sum::<f32>() / idm as f32;
                let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / idm as f32;
                let inv = 1.0 / (var + self.cfg.rms_norm_eps).sqrt();
                for j in 0..idm {
                    ov[i * idm + j] = (row[j] - mean) * inv * wv[j] + bv[j];
                }
            }
        }
        // Per-head score weights from hidden: [n, ih], scaled by n_heads^-0.5
        // (Glm5NextTextIndexer: weights = weights_proj(x) * n_heads**-0.5)
        let w_idx = {
            let raw = self.project(x, &format!("{pfx}.self_attn.indexer.weights_proj.weight"))?;
            let scale = (ih as f32).sqrt().recip();
            let v = raw.as_slice().iter().map(|v| v * scale).collect::<Vec<f32>>();
            Tensor::from_f32(Shape::new([n, ih]), v)
        };
        // kpool gate scores per NEW token: [n, idm] (cached alongside k_idx)
        let gate_for_cache = self.project(x, &format!("{pfx}.self_attn.indexer.index_kpool_compress_gate"))?;
        let family_idx = self.dsa_family_index(layer_idx);
        {
            let s = self.seqs.get_mut(&seq).unwrap();
            let c = &mut s.dsa_caches[family_idx];
            let t0 = c.k_nope.len() / (h * dk);
            for t in 0..n {
                let off = (t0 + t) * h * dk;
                c.k_nope.resize(off + h * dk, 0.0);
                let voff = (t0 + t) * h * dv;
                c.v.resize(voff + h * dv, 0.0);
                // kvb is [t, h, dk+dv]: per-head layout — K = [hd, 0:dk], V = [hd, dk:dk+dv].
                // Extract per-head (strided), NOT contiguous blocks.
                for hd in 0..h {
                    // kvb is [n, h*(dk+dv)]: per-head layout row = hd*(dk+dv)
                    let src = t * h * (dk + dv) + hd * (dk + dv);
                    c.k_nope[off + hd * dk..off + (hd + 1) * dk]
                        .copy_from_slice(&kvb.as_slice()[src..src + dk]);
                    c.v[voff + hd * dv..voff + (hd + 1) * dv]
                        .copy_from_slice(&kvb.as_slice()[src + dk..src + dk + dv]);
                }
                let ioff = (t0 + t) * idm;
                c.k_idx.resize(ioff + idm, 0.0);
                c.k_idx[ioff..ioff + idm].copy_from_slice(&ki.as_slice()[t * idm..(t + 1) * idm]);
                let goff = (t0 + t) * idm;
                c.k_gate.resize(goff + idm, 0.0);
                c.k_gate[goff..goff + idm]
                    .copy_from_slice(&gate_for_cache.as_slice()[t * idm..(t + 1) * idm]);
            }
        }
        let (k_all, v_all, kidx_all, kgate_all, total) = {
            let s = self.seqs.get(&seq).unwrap();
            let c = &s.dsa_caches[family_idx];
            (
                c.k_nope.clone(),
                c.v.clone(),
                c.k_idx.clone(),
                c.k_gate.clone(),
                c.k_nope.len() / (h * dk),
            )
        };
        let k_nope = Tensor::from_f32(Shape::new([total, h, dk]), k_all);
        let v = Tensor::from_f32(Shape::new([total, h, dv]), v_all);
        let k_idx_all = Tensor::from_f32(Shape::new([total, idm]), kidx_all.clone());
        // ---- k-pool compression (Glm5NextTextIndexer): group index keys into
        // pools of `index_kpool` (=4) tokens; per-CHANNEL softmax over
        // (gate + ape) mixes each pool's keys; top-k selects POOLS
        // (select_k = topk/kpool), selected pools expand back to token
        // indices; the visible tail (kpool-1) is appended; -1 = padding.
        let kpool = 4usize; // config index_kpool
        let npools = (total + kpool - 1) / kpool;
        let mut pool_keys = vec![0.0f32; npools * idm];
        {
            let ks = kidx_all.as_slice();
            let gv = kgate_all.as_slice(); // [total, idm] (cached gate scores)
            let ape = self
                .w(&format!("{pfx}.self_attn.indexer.index_kpool_compress_ape"))?
                .as_slice()
                .to_vec();
            for p in 0..npools {
                for d in 0..idm {
                    let mut lmax = f32::NEG_INFINITY;
                    for j in 0..kpool {
                        let t = p * kpool + j;
                        if t < total {
                            lmax = lmax.max(gv[t * idm + d] + ape[j * idm + d]);
                        }
                    }
                    if lmax == f32::NEG_INFINITY {
                        continue;
                    }
                    let mut den = 0.0f32;
                    let mut num = 0.0f32;
                    for j in 0..kpool {
                        let t = p * kpool + j;
                        if t < total {
                            let wgt = (gv[t * idm + d] + ape[j * idm + d] - lmax).exp();
                            den += wgt;
                            num += wgt * ks[t * idm + d];
                        }
                    }
                    pool_keys[p * idm + d] = num / den;
                }
            }
        }
        // pool visibility (causal): pool p is visible to query row i iff its LAST
        // token (min((p+1)*kpool, total) - 1) is <= ctx0 + i
        let ctx0 = total - n;
        let select_k = (d.index_topk / kpool).min(npools);
        let pool_idx_all = Tensor::from_f32(Shape::new([npools, idm]), pool_keys);
        let mut idx_pools = Tensor::zeros(Shape::new([n, select_k]), DType::F32);
        self.backend.indexer_topk(&qi, &pool_idx_all, &w_idx, select_k, ctx0 / kpool, &mut idx_pools)?;
        // expand selected pools to token indices + append the visible tail
        let out_width = select_k * kpool + (kpool - 1);
        let mut idx = Tensor::zeros(Shape::new([n, out_width]), DType::F32);
        {
            let iv = std::sync::Arc::get_mut(&mut idx.data).expect("unique idx");
            let pv = idx_pools.as_slice();
            for i in 0..n {
                let mut col = 0usize;
                for r in 0..select_k {
                    let pflt = pv[i * select_k + r];
                    if pflt >= 0.0 && (pflt as usize) < npools {
                        let p = pflt as usize;
                        for j in 0..kpool {
                            let t = p * kpool + j;
                            // transformers: expand ALL kpool slots; invalid
                            // (t >= total or beyond causal) become -1
                            iv[i * out_width + col] = if t < total && t <= ctx0 + i {
                                t as f32
                            } else {
                                -1.0
                            };
                            col += 1;
                        }
                    }
                }
                // tail: transformers append_visible_tail — tail_count =
                // visible_count % kpool entries starting at tail_start, padded
                // to kpool-1 slots with -1. tail_count == 0 → all -1.
                let visible_count = ctx0 + i + 1;
                let tail_count = visible_count % kpool;
                let tail_start = visible_count - tail_count;
                for j in 0..(kpool - 1) {
                    if col >= out_width {
                        break;
                    }
                    let t = tail_start + j;
                    iv[i * out_width + col] = if j < tail_count && t <= ctx0 + i {
                        t as f32
                    } else {
                        -1.0
                    };
                    col += 1;
                }
                while col < out_width {
                    iv[i * out_width + col] = -1.0; // padding
                    col += 1;
                }
            }
        }
        let mut out = Tensor::zeros(Shape::new([n, h, dv]), DType::F32);
        if std::env::var_os("FERRITE_PROBE").is_some() && layer_idx == 3 && n > 1 {
            let b: Vec<u8> = idx.as_slice().iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&ppath("l3_dsa_idx.f32"), b).ok();
            let b2: Vec<u8> = q.as_slice().iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&ppath("l3_dsa_q_sel.f32"), b2).ok();
        }
        self.backend.sparse_mla_attn(&q, &k_nope, &v, &idx, &mut out)?;
        if std::env::var_os("FERRITE_PROBE").is_some() && layer_idx == 3 && n > 1 {
            let b: Vec<u8> = out.as_slice().iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&ppath("l3_attn_raw.f32"), b).ok();
            let b2: Vec<u8> = v.as_slice().iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&ppath("l3_attn_v.f32"), b2).ok();
        }
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
        // Fused device chain (n==1 decode, TileRT ExpertSelect idea): GPU-side
        // expert dispatch via pointer tables — routing GEMV → moe_route →
        // ferrite_moe_fused_act → ferrite_moe_fused_down_sum, ids/probs never
        // cross to the host. (NOTE: this is the LIVE Engine's moe_ffn — the
        // exec_lib.rs copy is dead code; the same branch there never ran,
        // which is why fused MoE showed no ffn speedup until now.)
        #[cfg(feature = "cuda")]
        if let Some(cuda) = self.backend.as_cuda() {
            if std::env::var_os("FERRITE_MOE_DEV").is_some() && n == 1 {
                return self.moe_ffn_dev(cuda, pfx, x, n);
            }
        }
        let cfg = &self.cfg;
        let hidden = cfg.hidden_size;
        let topk = cfg.num_experts_per_tok;
        let e = cfg.n_routed_experts;
        let logits = self.project(x, &format!("{pfx}.mlp.gate.weight"))?;
        // noaux-tc routing bias (real checkpoint: mlp.gate.e_score_correction_bias)
        let bias = match self.weights.get(&format!("{pfx}.mlp.gate.e_score_correction_bias")) {
            Some(b) => b.clone(),
            None => Tensor::zeros(Shape::new([e]), DType::F32),
        };
        let mut probs = Tensor::zeros(Shape::new([n, topk]), DType::F32);
        let mut ids = Tensor::zeros(Shape::new([n, topk]), DType::F32);
        self.backend
            .moe_route(&logits, &bias, topk, cfg.routed_scaling_factor, &mut probs, &mut ids)?;
        if std::env::var_os("FERRITE_TRACE_MOE").is_some()
            && (pfx.ends_with("layers.3") || pfx.ends_with("layers.44"))
        {
            let ids_v = ids.as_slice();
            let pr = probs.as_slice();
            eprintln!(
                "[moe] L3 ids={:?} probs={:?} logits0={:.3} bias0={:.3}",
                &ids_v[..topk.min(8)],
                &pr[..topk.min(8)],
                logits.as_slice().iter().take(4).fold(f32::MIN, |a, b| a.max(*b)),
                bias.as_slice()[0]
            );
        }
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

    /// MoE FFN via the full fused device chain (moe_layer_dev, n==1):
    /// routing GEMV → moe_route → fused act → fused down_sum, with GPU-side
    /// expert dispatch (device pointer tables) — ids/probs never cross the
    /// host. FERRITE_MOE_DEV=1 + n==1 opt-in.
    #[cfg(feature = "cuda")]
    fn moe_ffn_dev(
        &self,
        cuda: &ferrite_kernel::cuda::CudaBackend,
        pfx: &str,
        x: &Tensor,
        n: usize,
    ) -> Result<Tensor> {
        use ferrite_kernel::cuda::{DevBuf, ExpertWeights};
        cuda.enter();
        let tm = std::env::var_os("FERRITE_TIMING").is_some();
        let t0 = std::time::Instant::now();
        let cfg = &self.cfg;
        let hidden = cfg.hidden_size;
        let topk = cfg.num_experts_per_tok;
        let e = cfg.n_routed_experts;
        let bias = match self.weights.get(&format!("{pfx}.mlp.gate.e_score_correction_bias")) {
            Some(b) => b.clone(),
            None => Tensor::zeros(Shape::new([e]), DType::F32),
        };
        let gate_w = self.w(&format!("{pfx}.mlp.gate.weight"))?;
        let shared = ExpertWeights {
            gate: self.w(&format!("{pfx}.mlp.shared_expert.gate_proj.weight"))?,
            up: self.w(&format!("{pfx}.mlp.shared_expert.up_proj.weight"))?,
            down: self.w(&format!("{pfx}.mlp.shared_expert.down_proj.weight"))?,
        };
        let (es, ee) = self.tp_expert_range.unwrap_or((0, e));
        let experts: Vec<ExpertWeights> = (es..ee)
            .map(|eid| {
                Ok(ExpertWeights {
                    gate: self.w(&format!("{pfx}.mlp.experts.{eid}.gate_proj.weight"))?,
                    up: self.w(&format!("{pfx}.mlp.experts.{eid}.up_proj.weight"))?,
                    down: self.w(&format!("{pfx}.mlp.experts.{eid}.down_proj.weight"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let t1 = std::time::Instant::now();
        let mut x_dev = DevBuf::alloc(cuda.dev(), cuda.stream(), x.numel())?;
        x_dev.upload(x.as_slice())?;
        let mut probs_scratch = DevBuf::alloc(cuda.dev(), cuda.stream(), n * topk)?;
        let out_dev = cuda.moe_layer_dev(
            &x_dev, gate_w, &bias, &shared, &experts, es,
            &mut probs_scratch, n, hidden, topk, e,
            cfg.routed_scaling_factor, cfg.swiglu_limit,
        )?;
        let mut out = Tensor::zeros(Shape::new([n, hidden]), x.dtype);
        let ov = std::sync::Arc::get_mut(&mut out.data).unwrap();
        out_dev.download(ov)?;
        if tm && n == 1 {
            let t2 = std::time::Instant::now();
            eprintln!(
                "[moe-ffn-dev] r{} prep={:5.2}ms ({} experts, {} w()-lookups) exec={:5.2}ms",
                ferrite_kernel::shard_idx(),
                (t1 - t0).as_secs_f32() * 1e3,
                experts.len(),
                4 + experts.len() * 3,
                (t2 - t1).as_secs_f32() * 1e3,
            );
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
