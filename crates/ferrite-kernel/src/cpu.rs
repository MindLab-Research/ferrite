//! CPU reference backend — the numerical golden standard.
//!
//! Every op is implemented exactly (no fused approximations). The B300 CUDA
//! backend must match this within fp tolerance before it is trusted. Where
//! the real GPU kernel uses a parallel form (e.g. WYF chunkwise Gated
//! DeltaNet), the CPU keeps the *definitional* recurrence so equivalence
//! testing is meaningful.

use std::sync::Arc;
use std::sync::Mutex;

use ferrite_types::{FerriteError, Result, Tensor};

use crate::graph::{GraphCapable, OpTrace, Recorder};

/// CPU reference backend — the numerical golden standard.
///
/// Every op is implemented exactly (no fused approximations). The B300 CUDA
/// backend must match this within fp tolerance before it is trusted. Where
/// the real GPU kernel uses a parallel form (e.g. WYF chunkwise Gated
/// DeltaNet), the CPU keeps the *definitional* recurrence so equivalence
/// testing is meaningful.
///
/// Kernels are stateless (state lives in engine-owned buffers); the only
/// interior state is an op-recorder used to validate CUDA-graph
/// feasibility (op-sequence stability across steps).
#[derive(Debug, Default)]
pub struct CpuBackend {
    recorder: Mutex<Recorder>,
}

impl Clone for CpuBackend {
    fn clone(&self) -> Self {
        CpuBackend { recorder: Mutex::new(Recorder::new()) }
    }
}

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend { recorder: Mutex::new(Recorder::new()) }
    }

    /// Record one op invocation (op name only on the CPU reference — the
    /// name *sequence* is what CUDA-graph bucketing needs to be stable).
    fn rec(&self, op: &'static str) {
        self.recorder.lock().unwrap().record(op, Vec::new());
    }

    fn dims(x: &Tensor) -> Vec<usize> {
        x.shape.0.clone()
    }

    fn check_2d(x: &Tensor, what: &str) -> Result<(usize, usize)> {
        if x.shape.rank() != 2 {
            return Err(FerriteError::InvalidArg(format!(
                "{what}: expected 2-D, got {:?}",
                x.shape
            )));
        }
        Ok((x.shape.0[0], x.shape.0[1]))
    }
}

impl GraphCapable for CpuBackend {
    fn begin_capture(&self) {
        self.recorder.lock().unwrap().begin_capture();
    }
    fn end_capture(&self) -> OpTrace {
        self.recorder.lock().unwrap().end_capture()
    }
    fn begin_verify(&self, trace: &OpTrace) {
        self.recorder.lock().unwrap().begin_verify(trace.clone());
    }
    fn end_verify(&self) -> bool {
        self.recorder.lock().unwrap().end_verify()
    }
}

impl crate::KernelBackend for CpuBackend {
    // ================= dense =================

    fn matmul(
        &self,
        x: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
        out: &mut Tensor,
    ) -> Result<()> {
        self.rec("matmul");
        self.rec("matmul");
        let (n, k) = Self::check_2d(x, "matmul x")?;
        let (o, k2) = Self::check_2d(w, "matmul w")?;
        if k != k2 {
            return Err(FerriteError::ShapeMismatch { expected: x.shape.clone(), got: w.shape.clone() });
        }
        if out.shape.0 != [n, o] {
            return Err(FerriteError::InvalidArg(format!(
                "matmul out shape {:?} != [{n},{o}]",
                out.shape
            )));
        }
        let xw = x.as_slice();
        let ww = w.as_slice();
        let mut acc = vec![0f32; n * o];
        let xs = &xw[..];
        for i in 0..n {
            for j in 0..o {
                let mut s = 0f32;
                for l in 0..k {
                    s += xs[i * k + l] * ww[j * k + l];
                }
                if let Some(b) = bias {
                    s += b.as_slice()[j];
                }
                acc[i * o + j] = s;
            }
        }
        // out is pre-allocated; overwrite via replace (Arc::make_mut zero-copy if unique)
        let ovec = Arc::get_mut(&mut out.data).expect("matmul out must be uniquely owned");
        ovec.copy_from_slice(&acc);
        Ok(())
    }

    fn rmsnorm(&self, x: &Tensor, w: &Tensor, eps: f32, out: &mut Tensor) -> Result<()> {
        self.rec("rmsnorm");
        let dim = *x.shape.0.last().ok_or_else(|| FerriteError::InvalidArg("rmsnorm rank 0".into()))?;
        let n = x.numel() / dim;
        if w.numel() != dim || out.shape != x.shape {
            return Err(FerriteError::InvalidArg("rmsnorm shape mismatch".into()));
        }
        let xs = x.as_slice();
        let ws = w.as_slice();
        let ovec = Arc::get_mut(&mut out.data).expect("unique out");
        for i in 0..n {
            let row = &xs[i * dim..(i + 1) * dim];
            let ss: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let inv = 1.0 / (ss + eps).sqrt();
            for j in 0..dim {
                ovec[i * dim + j] = row[j] * inv * ws[j];
            }
        }
        Ok(())
    }

    fn gated_rmsnorm(
        &self,
        x: &Tensor,
        gate: &Tensor,
        w: &Tensor,
        eps: f32,
        out: &mut Tensor,
    ) -> Result<()> {
        self.rec("gated_rmsnorm");
        self.rec("gated_rmsnorm");
        // GLM linear-attn o_norm: y = rmsnorm(x) * w * sigmoid(gate)
        // (Glm5NextTextRMSNormGated applies sigmoid to the gate, not +1)
        let dim = *x.shape.0.last().ok_or_else(|| FerriteError::InvalidArg("gated rank 0".into()))?;
        let n = x.numel() / dim;
        if gate.numel() != x.numel() || w.numel() != dim || out.shape != x.shape {
            return Err(FerriteError::InvalidArg("gated_rmsnorm shape mismatch".into()));
        }
        let xs = x.as_slice();
        let gs = gate.as_slice();
        let ws = w.as_slice();
        let ovec = Arc::get_mut(&mut out.data).expect("unique out");
        let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
        for i in 0..n {
            let row = &xs[i * dim..(i + 1) * dim];
            let ss: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let inv = 1.0 / (ss + eps).sqrt();
            for j in 0..dim {
                ovec[i * dim + j] = row[j] * inv * ws[j] * sig(gs[i * dim + j]);
            }
        }
        Ok(())
    }

    fn swiglu_limited(&self, gate_up: &Tensor, limit: f32, out: &mut Tensor) -> Result<()> {
        self.rec("swiglu");
        // gate_up: [n, 2*inter] (gate first, then up); out: [n, inter]
        let (n, two_i) = Self::check_2d(gate_up, "swiglu")?;
        if two_i % 2 != 0 {
            return Err(FerriteError::InvalidArg("swiglu last dim must be even".into()));
        }
        let inter = two_i / 2;
        if out.shape.0 != [n, inter] {
            return Err(FerriteError::InvalidArg(format!("swiglu out {:?} != [{n},{inter}]", out.shape)));
        }
        let g = gate_up.as_slice();
        let ovec = Arc::get_mut(&mut out.data).expect("unique out");
        // transformers: gate.clamp(max=limit) (single-sided), up.clamp(±limit)
        let cl_gate = |v: f32| if v > limit { limit } else { v };
        let cl_up = |v: f32| v.clamp(-limit, limit);
        for i in 0..n {
            for j in 0..inter {
                let gi = cl_gate(g[i * two_i + j]);
                let ui = cl_up(g[i * two_i + inter + j]);
                let silu = gi / (1.0 + (-gi).exp());
                ovec[i * inter + j] = silu * ui;
            }
        }
        Ok(())
    }

    // ================= linear attention =================

    fn causal_conv1d(
        &self,
        x: &Tensor,
        w: &Tensor,
        state_in: &Tensor,
        out: &mut Tensor,
        state_out: &mut Tensor,
    ) -> Result<()> {
        self.rec("conv1d");
        self.rec("conv1d");
        // Per-channel causal conv: stream = [tail state (conv-1)] ++ x[:, c];
        // out[t, c] = Σ_i w[c, i] * stream[hist + t - (conv-1) + i]
        let (n, ch) = Self::check_2d(x, "conv x")?;
        let (ch2, conv) = Self::check_2d(w, "conv w")?;
        if ch2 != ch || conv < 1 {
            return Err(FerriteError::InvalidArg("conv shape mismatch".into()));
        }
        let hist = conv - 1;
        if state_in.shape.0 != [ch, hist.max(1)] || state_out.shape.0 != [ch, hist.max(1)] {
            return Err(FerriteError::InvalidArg("conv state shape mismatch".into()));
        }
        if out.shape.0 != [n, ch] {
            return Err(FerriteError::InvalidArg("conv out shape".into()));
        }
        if conv == 1 {
            // degenerate: identity (no history needed)
            let ovec = Arc::get_mut(&mut out.data).expect("unique out");
            ovec.copy_from_slice(x.as_slice());
            return Ok(());
        }
        let xs = x.as_slice();
        let ws = w.as_slice();
        let st = state_in.as_slice();
        let ovec = Arc::get_mut(&mut out.data).expect("unique out");
        let sovec = Arc::get_mut(&mut state_out.data).expect("unique state");
        for c in 0..ch {
            let wrow = &ws[c * conv..(c + 1) * conv];
            // padded stream: hist values of prior input + n current inputs
            let mut stream: Vec<f32> = Vec::with_capacity(hist + n);
            for h in 0..hist {
                stream.push(st[c * hist + h]);
            }
            for t in 0..n {
                stream.push(xs[t * ch + c]);
            }
            for t in 0..n {
                let mut acc = 0f32;
                for (i, wi) in wrow.iter().enumerate() {
                    acc += wi * stream[hist + t - (conv - 1) + i];
                }
                ovec[t * ch + c] = acc;
            }
            // new tail = last hist inputs of the padded stream
            let tail_start = stream.len() - hist;
            for h in 0..hist {
                sovec[c * hist + h] = stream[tail_start + h];
            }
        }
        Ok(())
    }

    fn gated_deltanet_step(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        beta: &Tensor,
        gate: &Tensor,
        a_log: &Tensor,
        state_in: &Tensor,
        out: &mut Tensor,
        state_out: &mut Tensor,
    ) -> Result<()> {
        self.rec("deltanet_step");
        self.rec("deltanet_step");
        // q,k: [n,h,dk], v: [n,h,dv], beta/gate: [n,h], a_log: [h],
        // state: [h,dk,dv], out: [n,h,dv]
        let n = q.shape.0.first().copied().unwrap_or(0);
        let h = a_log.numel();
        let dk = q.shape.0.last().copied().unwrap_or(0);
        let dv = v.shape.0.last().copied().unwrap_or(0);
        if state_in.shape.0 != [h, dk, dv] || state_out.shape.0 != [h, dk, dv] {
            return Err(FerriteError::InvalidArg("deltanet state shape".into()));
        }
        if out.shape.0 != [n, h, dv] {
            return Err(FerriteError::InvalidArg(format!("deltanet out {:?} != [{n},{h},{dv}]", out.shape)));
        }
        let qs = q.as_slice();
        let ks = k.as_slice();
        let vs = v.as_slice();
        let bs = beta.as_slice();
        let gs = gate.as_slice(); // [n, h, dk]: log-space decay (negative),
                                  // = lb * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))
                                  // KDA: S *= exp(gate)  (fla naive_recurrent_kda)
        let als = a_log.as_slice();
        let _ = als; // folded into `gate` by the engine
        // work on a local copy of the state (state may alias state_in)
        let mut s = state_in.as_slice().to_vec();
        let ovec = Arc::get_mut(&mut out.data).expect("unique out");
        let head_elems = dk * dv;
        for t in 0..n {
            for hd in 0..h {
                let bt = bs[t * h + hd];
                let qh = &qs[t * h * dk + hd * dk..t * h * dk + (hd + 1) * dk];
                let kh = &ks[t * h * dk + hd * dk..t * h * dk + (hd + 1) * dk];
                let vh = &vs[t * h * dv + hd * dv..t * h * dv + (hd + 1) * dv];
                let gh = &gs[t * h * dk + hd * dk..t * h * dk + (hd + 1) * dk];
                let sh = &mut s[hd * head_elems..(hd + 1) * head_elems];
                // S[i, :] *= exp(gate[h, i]) — KDA decay is log-space (fla:
                // S = S * g.exp(); gate = lb*sigmoid(exp(A_log)*(fb+dt_bias)))
                for i in 0..dk {
                    let decay = gh[i].exp();
                    if decay != 1.0 {
                        for j in 0..dv {
                            sh[i * dv + j] *= decay;
                        }
                    }
                }
                // kS = S^T k  -> [dv]  (S stored [dk, dv]: (S^T k)_j = sum_i k_i * S[i, j])
                let k_s: Vec<f32> = (0..dv)
                    .map(|j| (0..dk).map(|i| kh[i] * sh[i * dv + j]).sum::<f32>())
                    .collect();
                // S -= beta * k (S^T k)^T  -> S[i,j] -= beta * k_i * kS[j]
                for i in 0..dk {
                    for j in 0..dv {
                        sh[i * dv + j] -= bt * kh[i] * k_s[j];
                    }
                }
                // S += beta * k v^T
                for i in 0..dk {
                    for j in 0..dv {
                        sh[i * dv + j] += bt * kh[i] * vh[j];
                    }
                }
                // o = q^T S  -> [dv]
                for j in 0..dv {
                    let o = (0..dk).map(|i| qh[i] * sh[i * dv + j]).sum::<f32>();
                    ovec[t * h * dv + hd * dv + j] = o;
                }
            }
        }
        let sovec = Arc::get_mut(&mut state_out.data).expect("unique state");
        sovec.copy_from_slice(&s);
        Ok(())
    }

    fn gated_deltanet_chunk(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        beta: &Tensor,
        gate: &Tensor,
        a_log: &Tensor,
        state_in: &Tensor,
        out: &mut Tensor,
        state_out: &mut Tensor,
    ) -> Result<()> {
        self.rec("deltanet_chunk");
        self.rec("deltanet_chunk");
        // CPU reference: the chunk form is exactly the step recurrence run
        // over all tokens (definitional ground truth; CUDA backend
        // implements the WYF-parallel form and must match this).
        self.gated_deltanet_step(q, k, v, beta, gate, a_log, state_in, out, state_out)
    }

    // ================= DSA sparse attention =================

    fn indexer_topk(
        &self,
        q_idx: &Tensor,
        k_idx: &Tensor,
        w: &Tensor,
        topk: usize,
        ctx0: usize,
        idx: &mut Tensor,
    ) -> Result<()> {
        self.rec("indexer_topk");
        // q_idx: [n, H*D] (per-head queries), k_idx: [t, D] (shared keys),
        // w: [n, H] per-head score weights. Causal: row i selects j <= ctx0+i.
        let (n, hd) = Self::check_2d(q_idx, "q_idx")?;
        let (t, d) = Self::check_2d(k_idx, "k_idx")?;
        if hd % d != 0 {
            return Err(FerriteError::InvalidArg("indexer q rows must be H*D".into()));
        }
        let h = hd / d;
        if w.shape.0 != [n, h] {
            return Err(FerriteError::InvalidArg(format!(
                "indexer w shape {:?} != [{n},{h}]",
                w.shape.0
            )));
        }
        if idx.shape.0 != [n, topk] || topk > t {
            return Err(FerriteError::InvalidArg("indexer topk shape".into()));
        }
        let qs = q_idx.as_slice();
        let ks = k_idx.as_slice();
        let ws = w.as_slice();
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        let ovec = Arc::get_mut(&mut idx.data).expect("unique idx");
        for i in 0..n {
            // causal guard: only keys j < ctx0 + i + 1 are candidates
            let jmax = (ctx0 + i + 1).min(t);
            let mut scored: Vec<(f32, usize)> = (0..t)
                .map(|j| {
                    if j >= jmax {
                        return (f32::NEG_INFINITY, j);
                    }
                    let mut s = 0.0f32;
                    for hi in 0..h {
                        let mut dot = 0.0f32;
                        let qo = i * hd + hi * d;
                        for l in 0..d {
                            dot += qs[qo + l] * ks[j * d + l];
                        }
                        s += ws[i * h + hi] * dot.max(0.0);
                    }
                    (s * inv_sqrt_d, j)
                })
                .collect();
            // partial sort: top-k by score, ties keep lower index (stable)
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
            for (rk, (sc, j)) in scored.iter().take(topk).enumerate() {
                let sel = if *sc == f32::NEG_INFINITY {
                    // invisible pool: emit -1 (skipped at expansion time)
                    -1.0f32
                } else {
                    *j as f32
                };
                ovec[i * topk + rk] = sel;
            }
        }
        Ok(())
    }

    fn sparse_mla_attn(
        &self,
        q: &Tensor,
        k_nope: &Tensor,
        v: &Tensor,
        idx: &Tensor,
        out: &mut Tensor,
    ) -> Result<()> {
        self.rec("sparse_attn");
        self.rec("sparse_attn");
        // q: [n,h,dq], k_nope: [t,h,dk] (dq == dk for nope-only MLA),
        // v: [t,h,dv], idx: [n,topk] -> out: [n,h,dv]
        let n = q.shape.0.first().copied().unwrap_or(0);
        let h = q.shape.0.get(1).copied().unwrap_or(0);
        let dq = *q.shape.0.last().unwrap_or(&0);
        let t = k_nope.shape.0.first().copied().unwrap_or(0);
        let dk = *k_nope.shape.0.last().unwrap_or(&0);
        let dv = *v.shape.0.last().unwrap_or(&0);
        let topk = *idx.shape.0.last().unwrap_or(&0);
        if dq != dk {
            return Err(FerriteError::InvalidArg(format!(
                "sparse attn q/k dim mismatch: dq={dq} dk={dk} (nope-only expects equal)"
            )));
        }
        if out.shape.0 != [n, h, dv] {
            return Err(FerriteError::InvalidArg(format!(
                "sparse attn out {:?} != [{n},{h},{dv}]",
                out.shape
            )));
        }
        let qs = q.as_slice();
        let ks = k_nope.as_slice();
        let vs = v.as_slice();
        let is_ = idx.as_slice();
        let ovec = Arc::get_mut(&mut out.data).expect("unique out");
        ovec.iter_mut().for_each(|e| *e = 0.0);
        let scale = 1.0 / (dq as f32).sqrt();
        for i in 0..n {
            for hd in 0..h {
                let qh = &qs[i * h * dq + hd * dq..i * h * dq + (hd + 1) * dq];
                // scores over selected tokens
                let mut sc = Vec::with_capacity(topk);
                for s in 0..topk {
                    let jf = is_[i * topk + s];
                    if jf < 0.0 {
                        continue; // kpool padding slot (-1)
                    }
                    let j = jf as usize;
                    if j >= t {
                        return Err(FerriteError::IndexOutOfBounds { index: j, len: t });
                    }
                    let kj = &ks[j * h * dk + hd * dk..j * h * dk + (hd + 1) * dk];
                    let sc_j: f32 = (0..dq).map(|l| qh[l] * kj[l]).sum::<f32>() * scale;
                    sc.push((sc_j, j));
                }
                let max_s = sc.iter().map(|(s, _)| *s).fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = sc.iter().map(|(s, _)| (s - max_s).exp()).collect();
                let denom: f32 = exps.iter().sum::<f32>() + f32::EPSILON;
                for (e, (_, j)) in exps.iter().zip(sc.iter()) {
                    let wgt = e / denom;
                    let vj = &vs[j * h * dv + hd * dv..j * h * dv + (hd + 1) * dv];
                    for l in 0..dv {
                        ovec[i * h * dv + hd * dv + l] += wgt * vj[l];
                    }
                }
            }
        }
        Ok(())
    }

    // ================= MoE =================

    fn moe_route(
        &self,
        logits: &Tensor,
        bias: &Tensor,
        topk: usize,
        routed_scaling: f32,
        probs: &mut Tensor,
        ids: &mut Tensor,
    ) -> Result<()> {
        self.rec("moe_route");
        self.rec("moe_route");
        let (n, e) = Self::check_2d(logits, "logits")?;
        if bias.numel() != e {
            return Err(FerriteError::InvalidArg("router bias shape".into()));
        }
        if probs.shape.0 != [n, topk] || ids.shape.0 != [n, topk] {
            return Err(FerriteError::InvalidArg("router out shape".into()));
        }
        let ls = logits.as_slice();
        let bs = bias.as_slice();
        let pvec = Arc::get_mut(&mut probs.data).expect("unique probs");
        let ivec = Arc::get_mut(&mut ids.data).expect("unique ids");
        for i in 0..n {
            // transformers Glm5NextTextTopkRouter:
            //   scores = sigmoid(logits)
            //   scores_for_choice = scores + e_score_correction_bias  (top-k on this)
            //   topk_weights = scores.gather(idx)  (raw sigmoid, no bias)
            //   renorm + routed_scaling_factor
            let sig: Vec<f32> = (0..e).map(|j| 1.0 / (1.0 + (-ls[i * e + j]).exp())).collect();
            let mut scored: Vec<(f32, usize)> = (0..e)
                .map(|j| (sig[j] + bs[j], j))
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
            let sel: Vec<(f32, usize)> = scored.iter().take(topk).copied().collect();
            let mut sum = 0.0f32;
            for (s, j) in &sel {
                sum += sig[*j];
            }
            for (rk, (_, j)) in sel.iter().enumerate() {
                ivec[i * topk + rk] = *j as f32;
                pvec[i * topk + rk] = sig[*j] / (sum + f32::EPSILON) * routed_scaling;
            }
        }
        Ok(())
    }

    fn expert_ffn(
        &self,
        x: &Tensor,
        gate_w: &Tensor,
        up_w: &Tensor,
        down_w: &Tensor,
        swiglu_limit: f32,
        out: &mut Tensor,
    ) -> Result<()> {
        self.rec("expert_ffn");
        self.rec("expert_ffn");
        // x: [m, hidden] -> inter -> swiglu -> down
        let (m, hidden) = Self::check_2d(x, "ffn x")?;
        let (inter, h2) = Self::check_2d(gate_w, "gate_w")?;
        if h2 != hidden || up_w.shape.0 != [inter, hidden] || down_w.shape.0 != [hidden, inter] {
            return Err(FerriteError::InvalidArg("ffn weight shapes".into()));
        }
        if out.shape.0 != [m, hidden] {
            return Err(FerriteError::InvalidArg("ffn out shape".into()));
        }
        let mut gate_up = Tensor::zeros(ferrite_types::Shape::new([m, 2 * inter]), x.dtype);
        {
            let gu = Arc::get_mut(&mut gate_up.data).expect("unique");
            let xs = x.as_slice();
            let gw = gate_w.as_slice();
            let uw = up_w.as_slice();
            for i in 0..m {
                for j in 0..inter {
                    gu[i * 2 * inter + j] = (0..hidden).map(|l| xs[i * hidden + l] * gw[j * hidden + l]).sum::<f32>();
                }
                for j in 0..inter {
                    gu[i * 2 * inter + inter + j] = (0..hidden).map(|l| xs[i * hidden + l] * uw[j * hidden + l]).sum::<f32>();
                }
            }
        }
        let mut act = Tensor::zeros(ferrite_types::Shape::new([m, inter]), x.dtype);
        self.swiglu_limited(&gate_up, swiglu_limit, &mut act)?;
        let mut final_out = Tensor::zeros(ferrite_types::Shape::new([m, hidden]), x.dtype);
        {
            let fo = Arc::get_mut(&mut final_out.data).expect("unique");
            let a = act.as_slice();
            let dw = down_w.as_slice();
            for i in 0..m {
                for j in 0..hidden {
                    fo[i * hidden + j] = (0..inter).map(|l| a[i * inter + l] * dw[j * inter + l]).sum::<f32>();
                }
            }
        }
        let ovec = Arc::get_mut(&mut out.data).expect("unique out");
        ovec.copy_from_slice(final_out.as_slice());
        Ok(())
    }

    // ================= sampling =================

    fn argmax_lastdim(&self, logits: &Tensor, out: &mut Tensor) -> Result<()> {
        self.rec("argmax");
        let dim = *logits.shape.0.last().unwrap_or(&0);
        let n = logits.numel() / dim.max(1);
        if out.shape.0 != [n] {
            return Err(FerriteError::InvalidArg("argmax out shape".into()));
        }
        let ls = logits.as_slice();
        let ovec = Arc::get_mut(&mut out.data).expect("unique out");
        for i in 0..n {
            let row = &ls[i * dim..(i + 1) * dim];
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (j, v) in row.iter().enumerate() {
                if *v > bv {
                    bv = *v;
                    best = j;
                }
            }
            ovec[i] = best as f32;
        }
        Ok(())
    }

    fn softmax_lastdim(&self, logits: &Tensor, out: &mut Tensor) -> Result<()> {
        self.rec("softmax");
        let dim = *logits.shape.0.last().unwrap_or(&0);
        let n = logits.numel() / dim.max(1);
        if out.shape != logits.shape {
            return Err(FerriteError::ShapeMismatch { expected: logits.shape.clone(), got: out.shape.clone() });
        }
        let ls = logits.as_slice();
        let ovec = Arc::get_mut(&mut out.data).expect("unique out");
        for i in 0..n {
            let row = &ls[i * dim..(i + 1) * dim];
            let m = row.iter().fold(f32::NEG_INFINITY, |a: f32, b: &f32| a.max(*b));
            let sum: f32 = row.iter().map(|v| (v - m).exp()).sum::<f32>();
            for (j, v) in row.iter().enumerate() {
                ovec[i * dim + j] = (v - m).exp() / sum;
            }
        }
        Ok(())
    }
}
