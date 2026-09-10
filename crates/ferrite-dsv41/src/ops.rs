//! CPU reference implementations of the DeepSeek-V4.1-Flash operators.
//!
//! These are the numerical golden standard: every CUDA kernel in
//! `kernels/cuda/dsv41_kernels.cu` must match the corresponding function here
//! within fp tolerance. They mirror the reference implementation
//! (`inference/model.py`, `inference/kernel.py`) 1:1 and are written for
//! clarity, not speed.
//!
//! **Runtime performance contract.** The GPU path must never dequantise a
//! weight to bf16/f32 and run a bf16 GEMM. Every large matmul is a tensor-core
//! MMA over the *native* format:
//!   * dense weights: fp8 e4m3 `mma...m16n8k32.f32.e4m3.e4m3.f32`, ue8m0
//!     32x32 block scales applied per k-block in the epilogue (the reference's
//!     `fp8_gemm_kernel` accumulator scheme);
//!   * routed experts: fp4 e2m1 `mma...kind::f8f6f4.f32.e2m1.e2m1.f32`
//!     (optionally the block-scaled `kind::mxf4` form), per-row-per-32 ue8m0
//!     scales applied per k-block in the epilogue;
//!   * runtime activations stay fp8/fp4 in the same layouts.
//! The `quant` dequantisation helpers exist only for checkpoint loading and
//! for the tests below.

use crate::config::Dsv41Config;
use crate::quant::{self, FP4_MAX, FP8_MAX};

pub const NEG_INF: f32 = -1e30;

// ===========================================================================
// RoPE (YaRN) with the dual theta of the compressed path
// ===========================================================================

/// YaRN frequency table, `[seqlen][dim/2]` (cos, sin) pairs.
#[derive(Debug, Clone, Default)]
pub struct FreqTable {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub half: usize,
}

fn correction_dim(num_rotations: f32, dim: usize, base: f32, max_seq_len: f32) -> f32 {
    dim as f32 * (max_seq_len / (num_rotations * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * base.ln())
}

fn correction_range(
    low_rot: f32,
    high_rot: f32,
    dim: usize,
    base: f32,
    max_seq_len: f32,
) -> (f32, f32) {
    let low = correction_dim(low_rot, dim, base, max_seq_len);
    let high = correction_dim(high_rot, dim, base, max_seq_len);
    (low.max(0.0), high.min((dim - 1) as f32))
}

pub fn precompute_freqs(
    dim: usize,
    seqlen: usize,
    original_seq_len: usize,
    base: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> FreqTable {
    let half = dim / 2;
    let mut freq = vec![0f32; half];
    for i in 0..half {
        freq[i] = 1.0 / base.powf(2.0 * i as f32 / dim as f32);
    }
    if original_seq_len > 0 {
        let (low, high) = correction_range(beta_fast, beta_slow, dim, base, original_seq_len as f32);
        for i in 0..half {
            let t = ((i as f32 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
            let smooth = 1.0 - t; // 1 - linear_ramp
            freq[i] = freq[i] / factor * (1.0 - smooth) + freq[i] * smooth;
        }
    }
    let mut cos = vec![0f32; seqlen * half];
    let mut sin = vec![0f32; seqlen * half];
    for t in 0..seqlen {
        for i in 0..half {
            let a = t as f32 * freq[i];
            cos[t * half + i] = a.cos();
            sin[t * half + i] = a.sin();
        }
    }
    FreqTable { cos, sin, half }
}

/// Rotary embedding over the trailing `dim` lanes of each row.
pub fn apply_rope(
    x: &mut [f32],
    rows: usize,
    row_len: usize,
    dim: usize,
    freqs: &FreqTable,
    pos0: usize,
    step: usize,
    inverse: bool,
) {
    let half = dim / 2;
    let off = row_len - dim;
    for r in 0..rows {
        let t = pos0 + r * step;
        for i in 0..half {
            let c = freqs.cos[t * freqs.half + i];
            let s = freqs.sin[t * freqs.half + i] * if inverse { -1.0 } else { 1.0 };
            let a = off + 2 * i;
            let x0 = x[r * row_len + a];
            let x1 = x[r * row_len + a + 1];
            x[r * row_len + a] = x0 * c - x1 * s;
            x[r * row_len + a + 1] = x0 * s + x1 * c;
        }
    }
}

// ===========================================================================
// Sliding-window index generation
// ===========================================================================

/// Which window-cache slots each query attends to; `-1` marks an empty slot.
/// Prefill materialises one row per query; decode has a single query seeing the
/// whole ring, oldest first.
pub fn window_topk_idxs(window: usize, bsz: usize, seqlen: usize, start_pos: usize) -> Vec<i32> {
    let cols = if start_pos == 0 { seqlen.min(window) } else { window };
    let mut out = vec![-1i32; bsz * seqlen * cols];
    for b in 0..bsz {
        for q in 0..seqlen {
            for c in 0..cols {
                let v: i64 = if start_pos == 0 {
                    let end = q as i64;
                    let idx = (end - window as i64 + 1).max(0) + c as i64;
                    if idx > end {
                        -1
                    } else {
                        idx
                    }
                } else {
                    let oldest = (start_pos % window + 1) as i64;
                    let idx = if (c as i64) < window as i64 - oldest {
                        oldest + c as i64
                    } else {
                        c as i64 - (window as i64 - oldest)
                    };
                    if idx > start_pos as i64 {
                        -1
                    } else {
                        idx
                    }
                };
                out[(b * seqlen + q) * cols + c] = v as i32;
            }
        }
    }
    out
}

// ===========================================================================
// Compressor: softmax-gated pooling of `ratio` tokens into one KV latent
// ===========================================================================

/// Per-layer compressor state carried across decode steps.
#[derive(Debug, Clone)]
pub struct CompressorState {
    /// `[bsz, ratio, head_dim]`
    pub kv: Vec<f32>,
    /// `[bsz, ratio, head_dim]`, initialised to -inf
    pub score: Vec<f32>,
}

impl CompressorState {
    pub fn new(bsz: usize, ratio: usize, head_dim: usize) -> Self {
        CompressorState {
            kv: vec![0.0; bsz * ratio * head_dim],
            score: vec![f32::NEG_INFINITY; bsz * ratio * head_dim],
        }
    }
}

/// `Compressor.forward`: `Some((latent, out_rows))` on the steps that complete
/// a group, else `None`.
#[allow(clippy::too_many_arguments)]
pub fn compressor_forward(
    x: &[f32],
    bsz: usize,
    seqlen: usize,
    dim: usize,
    wkv: &[f32],
    wgate: Option<&[f32]>,
    norm_w: &[f32],
    head_dim: usize,
    ratio: usize,
    start_pos: usize,
    eps: f32,
    state: Option<&mut CompressorState>,
) -> Option<(Vec<f32>, usize)> {
    if ratio == 0 {
        return None;
    }
    let proj = |w: &[f32], x: &[f32]| -> Vec<f32> {
        let mut o = vec![0f32; bsz * seqlen * head_dim];
        for b in 0..bsz {
            for t in 0..seqlen {
                let xr = &x[(b * seqlen + t) * dim..(b * seqlen + t + 1) * dim];
                for r in 0..head_dim {
                    let wr = &w[r * dim..(r + 1) * dim];
                    let mut acc = 0f32;
                    for k in 0..dim {
                        acc += wr[k] * xr[k];
                    }
                    o[(b * seqlen + t) * head_dim + r] = acc;
                }
            }
        }
        o
    };
    let kvp = proj(wkv, x);
    if ratio == 1 {
        let mut out = kvp;
        rmsnorm_rows(&mut out, bsz * seqlen, head_dim, norm_w, eps);
        return Some((out, seqlen));
    }
    let scp = proj(wgate.unwrap(), x);
    let st = state.expect("ratio > 1 needs the carried state");
    let (should, out_rows);
    if start_pos == 0 {
        let ngroups = seqlen / ratio;
        out_rows = ngroups;
        should = seqlen >= ratio;
        let rem = seqlen % ratio;
        let cut = seqlen - rem;
        if rem > 0 {
            for b in 0..bsz {
                for t in 0..rem {
                    for c in 0..head_dim {
                        st.kv[(b * ratio + t) * head_dim + c] =
                            kvp[(b * seqlen + cut + t) * head_dim + c];
                        st.score[(b * ratio + t) * head_dim + c] =
                            scp[(b * seqlen + cut + t) * head_dim + c];
                    }
                }
            }
        }
    } else {
        should = (start_pos + 1) % ratio == 0;
        out_rows = 1;
        let slot = start_pos % ratio;
        for b in 0..bsz {
            for c in 0..head_dim {
                st.kv[(b * ratio + slot) * head_dim + c] = kvp[b * head_dim + c];
                st.score[(b * ratio + slot) * head_dim + c] = scp[b * head_dim + c];
            }
        }
    }
    if !should {
        return None;
    }
    let mut out = vec![0f32; bsz * out_rows * head_dim];
    for b in 0..bsz {
        for g in 0..out_rows {
            for c in 0..head_dim {
                // pick the ratio source slots for this group
                let mut mx = f32::NEG_INFINITY;
                let mut vals = [(0f32, 0f32); 32];
                for r in 0..ratio.min(32) {
                    let (s, v) = if start_pos == 0 {
                        if g * ratio + r < seqlen {
                            (
                                scp[(b * seqlen + g * ratio + r) * head_dim + c],
                                kvp[(b * seqlen + g * ratio + r) * head_dim + c],
                            )
                        } else {
                            let slot = ratio - (seqlen - g * ratio) + r;
                            (
                                st.score[(b * ratio + slot.min(ratio - 1)) * head_dim + c],
                                st.kv[(b * ratio + slot.min(ratio - 1)) * head_dim + c],
                            )
                        }
                    } else {
                        (
                            st.score[(b * ratio + r) * head_dim + c],
                            st.kv[(b * ratio + r) * head_dim + c],
                        )
                    };
                    vals[r] = (s, v);
                    mx = mx.max(s);
                }
                let mut den = 0f32;
                let mut acc = 0f32;
                for r in 0..ratio.min(32) {
                    let e = (vals[r].0 - mx).exp();
                    den += e;
                    acc += e * vals[r].1;
                }
                out[(b * out_rows + g) * head_dim + c] = if den > 0.0 { acc / den } else { 0.0 };
            }
        }
    }
    rmsnorm_rows(&mut out, bsz * out_rows, head_dim, norm_w, eps);
    Some((out, out_rows))
}

/// In-place RMSNorm over `cols`-wide rows (public: the chain applies it to
/// per-token KV/query rows).
pub fn rmsnorm_rows_pub(x: &mut [f32], w: &[f32], cols: usize, eps: f32) {
    let rows = x.len() / cols;
    rmsnorm_rows(x, rows, cols, w, eps);
}

fn rmsnorm_rows(x: &mut [f32], rows: usize, cols: usize, w: &[f32], eps: f32) {
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let mut ss = 0f32;
        for &v in row.iter() {
            ss += v * v;
        }
        let inv = 1.0 / (ss / cols as f32 + eps).sqrt();
        for c in 0..cols {
            row[c] = row[c] * inv * w[c];
        }
    }
}

// ===========================================================================
// Sparse attention (window + compressed positions, one shared KV head)
// ===========================================================================

/// `sparse_attn`: per (batch, query) gather `topk` positions from `kv`, run an
/// online softmax, then fold in the learnable attention sink.
///
/// `q`: `[b, m, h, d]`, `kv`: `[b, n, d]` (ONE KV head shared by all query
/// heads), `sink`: `[h]`, `idxs`: `[b, m, topk]`.
#[allow(clippy::too_many_arguments)]
pub fn sparse_attn(
    q: &[f32],
    kv: &[f32],
    sink: &[f32],
    idxs: &[i32],
    b: usize,
    m: usize,
    h: usize,
    d: usize,
    n: usize,
    topk: usize,
    scale: f32,
) -> Vec<f32> {
    let mut o = vec![0f32; b * m * h * d];
    for bb in 0..b {
        for mm in 0..m {
            for hh in 0..h {
                let qr = &q[((bb * m + mm) * h + hh) * d..((bb * m + mm) * h + hh + 1) * d];
                let mut acc = vec![0f32; d];
                let mut sum_exp = 0f32;
                // finite floor: an all-empty row yields a zero output, not NaN
                let mut smax = NEG_INF;
                for t in 0..topk {
                    let idx = idxs[(bb * m + mm) * topk + t];
                    if idx < 0 {
                        continue;
                    }
                    let kr = &kv[(bb * n + idx as usize) * d..(bb * n + idx as usize + 1) * d];
                    let mut s = 0f32;
                    for c in 0..d {
                        s += qr[c] * kr[c];
                    }
                    s *= scale;
                    let new_max = smax.max(s);
                    let corr = (smax - new_max).exp();
                    let e = (s - new_max).exp();
                    sum_exp = sum_exp * corr + e;
                    for c in 0..d {
                        acc[c] = acc[c] * corr + e * kr[c];
                    }
                    smax = new_max;
                }
                // the sink enters the denominator only, after the loop
                sum_exp += (sink[hh] - smax).exp();
                for c in 0..d {
                    o[((bb * m + mm) * h + hh) * d + c] = if sum_exp > 0.0 {
                        acc[c] / sum_exp
                    } else {
                        0.0
                    };
                }
            }
        }
    }
    o
}

// ===========================================================================
// Two-level indexer
// ===========================================================================

/// Level one: keep the `topk_blocks` best-scoring blocks per query.
pub fn select_candidate_blocks(
    logits: &[f32],
    rows: usize,
    n_pos: usize,
    compress_lens: &[usize],
    topk_blocks: usize,
    block_size: usize,
) -> Vec<bool> {
    let nb = n_pos.div_ceil(block_size.max(1));
    let mut out = vec![false; rows * n_pos];
    for r in 0..rows {
        let src = &logits[r * n_pos..(r + 1) * n_pos];
        let mut scores = vec![f32::NEG_INFINITY; nb];
        for blk in 0..nb {
            let mut mx = f32::NEG_INFINITY;
            for i in 0..block_size {
                let p = blk * block_size + i;
                if p < n_pos {
                    mx = mx.max(src[p]);
                }
            }
            scores[blk] = mx;
        }
        // pin the newest (partially filled) block
        let last = compress_lens[r].saturating_sub(1) / block_size.max(1);
        if last < nb {
            scores[last] = f32::INFINITY;
        }
        let k = topk_blocks.min(nb);
        let mut order: Vec<usize> = (0..nb).collect();
        order.sort_by(|&a, &b| {
            scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        for &blk in order.iter().take(k) {
            if scores[blk] > f32::NEG_INFINITY {
                for i in 0..block_size {
                    let p = blk * block_size + i;
                    if p < n_pos {
                        out[r * n_pos + p] = true;
                    }
                }
            }
        }
    }
    out
}

/// The indexer side attention: rectified scores combined by `weights_proj`, then
/// the top `index_topk` positions re-sorted into position order.
#[allow(clippy::too_many_arguments)]
pub fn indexer_topk(
    q: &[f32],
    index_k: &[f32],
    weights: &[f32],
    b: usize,
    m: usize,
    nh: usize,
    hd: usize,
    n_pos: usize,
    compress_lens: &[usize],
    candidates: Option<&[bool]>,
    index_topk: usize,
    offset: i32,
    softmax_scale: f32,
    head_scale: f32,
) -> Vec<i32> {
    let cols = index_topk.min(n_pos);
    let mut out = vec![-1i32; b * m * cols];
    for bb in 0..b {
        for mm in 0..m {
            let cl = compress_lens[mm.min(compress_lens.len() - 1)];
            let mut score = vec![0f32; n_pos];
            for p in 0..n_pos {
                let mut acc = 0f32;
                for h in 0..nh {
                    let qr = &q[((bb * m + mm) * nh + h) * hd..((bb * m + mm) * nh + h + 1) * hd];
                    let kr = &index_k[(bb * n_pos + p) * hd..(bb * n_pos + p + 1) * hd];
                    let mut dot = 0f32;
                    for c in 0..hd {
                        dot += qr[c] * kr[c];
                    }
                    acc += dot.max(0.0) * weights[(bb * m + mm) * nh + h];
                }
                let mut s = acc * softmax_scale * head_scale;
                if p >= cl {
                    s = f32::NEG_INFINITY;
                }
                if let Some(c) = candidates {
                    if !c[(bb * m + mm) * n_pos + p] {
                        s = f32::NEG_INFINITY;
                    }
                }
                score[p] = s;
            }
            let mut order: Vec<usize> = (0..n_pos).collect();
            order.sort_by(|&a, &b| {
                score[b].partial_cmp(&score[a]).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut picked: Vec<usize> = order.into_iter().take(cols).collect();
            picked.sort_unstable();
            for (i, &p) in picked.iter().enumerate() {
                out[(bb * m + mm) * cols + i] = if p < cl { p as i32 + offset } else { -1 };
            }
        }
    }
    out
}

// ===========================================================================
// MoE
// ===========================================================================

/// `Gate.forward`: sqrtsoftplus scores, a bias that only *selects* experts,
/// optional top-k normalisation and the route scale.
#[allow(clippy::too_many_arguments)]
pub fn moe_gate(
    x: &[f32],
    gate_w: &[f32],
    gate_bias: &[f32],
    rows: usize,
    dim: usize,
    n_experts: usize,
    topk: usize,
    gate_temp: f32,
    norm_topk_prob: bool,
    route_scale: f32,
    score_func: &str,
) -> (Vec<f32>, Vec<i32>) {
    let mut weights = vec![0f32; rows * topk];
    let mut indices = vec![0i32; rows * topk];
    for r in 0..rows {
        let xr = &x[r * dim..(r + 1) * dim];
        let mut scores = vec![0f32; n_experts];
        for e in 0..n_experts {
            let wr = &gate_w[e * dim..(e + 1) * dim];
            let mut acc = 0f32;
            for c in 0..dim {
                acc += wr[c] * xr[c];
            }
            acc /= gate_temp;
            scores[e] = match score_func {
                "softmax" => acc,
                "sigmoid" => 1.0 / (1.0 + (-acc).exp()),
                _ => (1.0 + acc.exp()).ln().sqrt(),
            };
        }
        if score_func == "softmax" {
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut den = 0f32;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                den += *s;
            }
            for s in scores.iter_mut() {
                *s /= den;
            }
        }
        let mut idx: Vec<usize> = (0..n_experts).collect();
        idx.sort_by(|&a, &b| {
            (scores[b] + gate_bias[b])
                .partial_cmp(&(scores[a] + gate_bias[a]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let sel = &idx[..topk];
        let mut w: Vec<f32> = sel.iter().map(|&e| scores[e]).collect();
        if norm_topk_prob && topk > 1 {
            let s: f32 = w.iter().sum();
            for v in w.iter_mut() {
                *v /= s + 1e-20; // the reference uses 1e-20 here, not norm_eps
            }
        }
        for (i, v) in w.iter().enumerate() {
            weights[r * topk + i] = v * route_scale;
            indices[r * topk + i] = sel[i] as i32;
        }
    }
    (weights, indices)
}

/// One SwiGLU expert with the training clamps (up clamped both sides, gate only
/// from above).
#[allow(clippy::too_many_arguments)]
pub fn expert_ffn(
    x: &[f32],
    w1: &[f32],
    w3: &[f32],
    w2: &[f32],
    weight: Option<&[f32]>,
    rows: usize,
    dim: usize,
    inter: usize,
    limit: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; rows * dim];
    for r in 0..rows {
        let xr = &x[r * dim..(r + 1) * dim];
        let mut h = vec![0f32; inter];
        for i in 0..inter {
            let w1r = &w1[i * dim..(i + 1) * dim];
            let w3r = &w3[i * dim..(i + 1) * dim];
            let mut g = 0f32;
            let mut u = 0f32;
            for c in 0..dim {
                g += w1r[c] * xr[c];
                u += w3r[c] * xr[c];
            }
            if limit > 0.0 {
                u = u.clamp(-limit, limit);
                g = g.min(limit);
            }
            let silu = g / (1.0 + (-g).exp());
            let mut v = silu * u;
            if let Some(w) = weight {
                v *= w[r];
            }
            h[i] = v;
        }
        for o in 0..dim {
            let w2r = &w2[o * inter..(o + 1) * inter];
            let mut acc = 0f32;
            for i in 0..inter {
                acc += w2r[i] * h[i];
            }
            out[r * dim + o] = acc;
        }
    }
    out
}

// ===========================================================================
// Hyper-connections (same geometry as GLM-5.3-Flash)
// ===========================================================================

pub struct HcCoeffs {
    pub pre: Vec<f32>,
    pub post: Vec<f32>,
    pub comb: Vec<f32>,
}

/// `hc_mixes` + `hc_split_sinkhorn`.
#[allow(clippy::too_many_arguments)]
pub fn hc_mixes(
    x: &[f32],
    hc_fn: &[f32],
    hc_scale: &[f32],
    hc_base: &[f32],
    rows: usize,
    hc_dim: usize,
    hc: usize,
    sinkhorn_iters: usize,
    eps: f32,
) -> HcCoeffs {
    let mix = hc * (2 + hc);
    let mut mixes = vec![0f32; rows * mix];
    for r in 0..rows {
        let xr = &x[r * hc_dim..(r + 1) * hc_dim];
        let mut ss = 0f32;
        for &v in xr {
            ss += v * v;
        }
        let inv = 1.0 / (ss / hc_dim as f32 + eps).sqrt();
        for m in 0..mix {
            let wr = &hc_fn[m * hc_dim..(m + 1) * hc_dim];
            let mut acc = 0f32;
            for c in 0..hc_dim {
                acc += wr[c] * xr[c];
            }
            mixes[r * mix + m] = acc * inv;
        }
    }
    let mut pre = vec![0f32; rows * hc];
    let mut post = vec![0f32; rows * hc];
    let mut comb = vec![0f32; rows * hc * hc];
    for r in 0..rows {
        for j in 0..hc {
            pre[r * hc + j] = sigmoid(mixes[r * mix + j] * hc_scale[0] + hc_base[j]) + eps;
            post[r * hc + j] =
                2.0 * sigmoid(mixes[r * mix + hc + j] * hc_scale[1] + hc_base[hc + j]);
        }
        let mut cm = vec![0f32; hc * hc];
        for j in 0..hc {
            for k in 0..hc {
                cm[j * hc + k] = mixes[r * mix + 2 * hc + j * hc + k] * hc_scale[2]
                    + hc_base[2 * hc + j * hc + k];
            }
        }
        for j in 0..hc {
            let mx = cm[j * hc..(j + 1) * hc].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0f32;
            for k in 0..hc {
                cm[j * hc + k] = (cm[j * hc + k] - mx).exp();
                s += cm[j * hc + k];
            }
            for k in 0..hc {
                cm[j * hc + k] = cm[j * hc + k] / s + eps;
            }
        }
        normalize_cols(&mut cm, hc, eps);
        for _ in 1..sinkhorn_iters {
            for j in 0..hc {
                let s: f32 = (0..hc).map(|k| cm[j * hc + k]).sum();
                for k in 0..hc {
                    cm[j * hc + k] /= s + eps;
                }
            }
            normalize_cols(&mut cm, hc, eps);
        }
        comb[r * hc * hc..(r + 1) * hc * hc].copy_from_slice(&cm);
    }
    HcCoeffs { pre, post, comb }
}

fn normalize_cols(cm: &mut [f32], hc: usize, eps: f32) {
    let mut col = vec![0f32; hc];
    for j in 0..hc {
        for k in 0..hc {
            col[k] += cm[j * hc + k];
        }
    }
    for j in 0..hc {
        for k in 0..hc {
            cm[j * hc + k] /= col[k] + eps;
        }
    }
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// `hc_pre`: collapse the hc copies into one sublayer input.
pub fn hc_pre(x: &[f32], pre: &[f32], rows: usize, hc: usize, dim: usize) -> Vec<f32> {
    let mut y = vec![0f32; rows * dim];
    for r in 0..rows {
        for c in 0..dim {
            let mut acc = 0f32;
            for i in 0..hc {
                acc += pre[r * hc + i] * x[(r * hc + i) * dim + c];
            }
            y[r * dim + c] = acc;
        }
    }
    y
}

/// `hc_post`: expand the sublayer output back out, mixing the residual in.
pub fn hc_post(
    x: &[f32],
    res: &[f32],
    post: &[f32],
    comb: &[f32],
    rows: usize,
    hc: usize,
    dim: usize,
) -> Vec<f32> {
    let mut y = vec![0f32; rows * hc * dim];
    for r in 0..rows {
        for i in 0..hc {
            for c in 0..dim {
                let mut acc = post[r * hc + i] * x[r * dim + c];
                for k in 0..hc {
                    acc += comb[r * hc * hc + k * hc + i] * res[(r * hc + k) * dim + c];
                }
                y[(r * hc + i) * dim + c] = acc;
            }
        }
    }
    y
}

// ===========================================================================
// Engram
// ===========================================================================

/// `Engram.forward`: the n-gram lookup written into the residual stream, gated
/// by a normalised dot product of the stream against the key.
#[allow(clippy::too_many_arguments)]
pub fn engram_forward(
    x: &[f32],
    rows_kv: &[f32],
    q_weight: &[f32],
    k_weight: &[f32],
    rows: usize,
    hc: usize,
    dim: usize,
    eps: f32,
    token_mask: Option<&[bool]>,
) -> Vec<f32> {
    let clamp_value = 1e-6f32;
    let kv_span = hc * dim + dim;
    let mut out = vec![0f32; rows * hc * dim];
    for r in 0..rows {
        let key = &rows_kv[r * kv_span..r * kv_span + hc * dim];
        let value = &rows_kv[r * kv_span + hc * dim..(r + 1) * kv_span];
        for i in 0..hc {
            let h = &x[(r * hc + i) * dim..(r * hc + i + 1) * dim];
            let k = &key[i * dim..(i + 1) * dim];
            let mut hss = 0f32;
            let mut kss = 0f32;
            for c in 0..dim {
                hss += h[c] * h[c];
                kss += k[c] * k[c];
            }
            let rstd = (1.0 / (hss / dim as f32 + eps).sqrt())
                * (1.0 / (kss / dim as f32 + eps).sqrt());
            let mut dot = 0f32;
            for c in 0..dim {
                dot += h[c] * q_weight[i * dim + c] * k_weight[i * dim + c] * k[c];
            }
            dot *= rstd * (dim as f32).powf(-0.5);
            let mag = dot.abs().max(clamp_value).sqrt() * dot.signum();
            let mut gate = sigmoid(mag);
            if let Some(m) = token_mask {
                if !m[r] {
                    gate = 0.0;
                }
            }
            for c in 0..dim {
                out[(r * hc + i) * dim + c] = h[c] + gate * value[c];
            }
        }
    }
    out
}

// ===========================================================================
// Sampling (Gumbel-max)
// ===========================================================================

/// `sample()`: temperature 0 is greedy; otherwise softmax then divide by an
/// exponential(1) draw and take the argmax.
pub fn gumbel_argmax(logits: &[f32], vocab: usize, temperature: f32, u: &[f32]) -> u32 {
    if temperature == 0.0 {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for i in 0..vocab {
            if logits[i] > bv {
                bv = logits[i];
                best = i;
            }
        }
        return best as u32;
    }
    let t = temperature.max(1e-5);
    let mx = logits.iter().take(vocab).cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut den = 0f32;
    for i in 0..vocab {
        den += ((logits[i] - mx) / t).exp();
    }
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for i in 0..vocab {
        let p = ((logits[i] - mx) / t).exp() / den;
        let v = p / u[i].max(1e-30);
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}

// ===========================================================================
// Small helpers
// ===========================================================================

pub fn rmsnorm(x: &[f32], w: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    rmsnorm_rows(&mut out, rows, cols, w, eps);
    out
}

/// fp8 e4m3 row quantisation (window KV: block 128, power-of-two scale).
pub fn kv_quant_fp8(x: &[f32], block: usize, round_scale: bool) -> (Vec<u8>, Vec<f32>) {
    quant::act_quant_fp8(x, block, round_scale)
}

/// fp4 quantisation (compressed KV: block 16 / indexer q,k: block 32).
pub fn kv_quant_fp4(x: &[f32], block: usize, round_scale: bool) -> (Vec<u8>, Vec<f32>) {
    quant::act_quant_fp4(x, block, round_scale)
}

pub const FP4_SAT: f32 = FP4_MAX;
pub const FP8_SAT: f32 = FP8_MAX;

/// Whether `layer` is a DSpark target (its attention input feeds the draft).
pub fn is_dspark_target(cfg: &Dsv41Config, layer: usize) -> bool {
    cfg.dspark_target_layer_ids.contains(&layer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_idxs_prefill_matches_reference() {
        let w = 3;
        let m = 4;
        let id = window_topk_idxs(w, 1, m, 0);
        assert_eq!(id.len(), m * w.min(m));
        assert_eq!(&id[0..3], &[0, -1, -1]);
        assert_eq!(&id[3..6], &[0, 1, -1]);
        assert_eq!(&id[6..9], &[0, 1, 2]);
        assert_eq!(&id[9..12], &[1, 2, 3]);
    }

    #[test]
    fn window_idxs_decode_is_a_ring_oldest_first() {
        let w = 4;
        let start = 6;
        let id = window_topk_idxs(w, 1, 1, start);
        let oldest = start % w + 1;
        assert_eq!(id.len(), w);
        let mut want: Vec<i32> = (oldest as i32..w as i32).collect();
        want.extend(0..oldest as i32);
        assert_eq!(id, want);
    }

    #[test]
    fn candidate_blocks_pin_the_newest_block() {
        let n_pos = 16;
        let block = 8;
        let mut logits = vec![0f32; n_pos];
        logits[8] = 100.0;
        let m = select_candidate_blocks(&logits, 1, n_pos, &[16], 1, block);
        assert!(m[8..16].iter().all(|&b| b));
        assert!(m[0..8].iter().all(|&b| !b));
        let logits0 = vec![0f32; n_pos];
        let m2 = select_candidate_blocks(&logits0, 1, n_pos, &[9], 1, block);
        assert!(m2[8..16].iter().all(|&b| b), "block 1 holds the newest position");
    }

    #[test]
    fn sparse_attn_softmax_and_sink() {
        let q = vec![1.0f32, 0.0];
        let kv = vec![1.0f32, 0.0, 0.0f32, 1.0];
        let sink = vec![f32::NEG_INFINITY];
        let o = sparse_attn(&q, &kv, &sink, &[0, 1], 1, 1, 1, 2, 2, 2, 1.0);
        let e = std::f32::consts::E;
        let w0 = e / (e + 1.0);
        assert!((o[0] - w0).abs() < 1e-6);
        assert!((o[1] - (1.0 - w0)).abs() < 1e-6);
    }

    #[test]
    fn sparse_attn_empty_row_is_all_zero() {
        let q = vec![1.0f32; 2];
        let kv = vec![0f32; 4];
        let o = sparse_attn(&q, &kv, &[0.0], &[-1i32, -1], 1, 1, 1, 2, 2, 2, 1.0);
        assert_eq!(o, vec![0.0, 0.0]);
    }

    #[test]
    fn sparse_attn_sink_only_enters_the_denominator() {
        let q = vec![1.0f32];
        let kv = vec![1.0f32];
        let o0 = sparse_attn(&q, &kv, &[0.0], &[0], 1, 1, 1, 1, 1, 1, 1.0);
        let want = std::f32::consts::E / (std::f32::consts::E + 1.0);
        assert!((o0[0] - want).abs() < 1e-6);
        let o1 = sparse_attn(&q, &kv, &[f32::NEG_INFINITY], &[0], 1, 1, 1, 1, 1, 1, 1.0);
        assert!((o1[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn gate_bias_selects_but_does_not_scale() {
        let x = vec![1.0f32, 0.0];
        let w = vec![1.0f32, 0.0, 1.0f32, 0.0];
        let bias = vec![0.0f32, 10.0];
        let (weights, idx) = moe_gate(&x, &w, &bias, 1, 2, 2, 1, 1.0, false, 1.0, "sigmoid");
        assert_eq!(idx[0], 1, "the bias picks the expert");
        let want = 1.0 / (1.0 + (-1.0f32).exp());
        assert!((weights[0] - want).abs() < 1e-6, "{} vs {want}", weights[0]);
    }

    #[test]
    fn gate_sqrtsoftplus_and_route_scale() {
        let x = vec![2.0f32];
        let w = vec![0.5f32, 0.25];
        let bias = vec![0.0f32, 0.0];
        let (weights, idx) = moe_gate(&x, &w, &bias, 1, 1, 2, 1, 1.0, false, 1.5, "sqrtsoftplus");
        assert_eq!(idx[0], 0);
        let want = (1.0 + 1.0f32.exp()).ln().sqrt() * 1.5;
        assert!((weights[0] - want).abs() < 1e-6);
    }

    #[test]
    fn expert_clamps_match_training_convention() {
        let y = expert_ffn(&[1.0], &[100.0], &[-100.0], &[1.0], None, 1, 1, 1, 10.0);
        let silu = 10.0f32 / (1.0 + (-10.0f32).exp());
        assert!((y[0] - silu * -10.0).abs() < 1e-4, "{}", y[0]);
    }

    #[test]
    fn hc_pre_post_roundtrip_with_identity_mix() {
        let (rows, hc, dim) = (1usize, 4usize, 3usize);
        let x: Vec<f32> = (0..rows * hc * dim).map(|i| i as f32).collect();
        let pre = vec![1.0f32, 0.0, 0.0, 0.0];
        assert_eq!(hc_pre(&x, &pre, rows, hc, dim), vec![0.0, 1.0, 2.0]);
        let post = vec![0f32; hc];
        let mut comb = vec![0f32; hc * hc];
        for k in 0..hc {
            comb[k * hc + k] = 1.0;
        }
        let z = hc_post(&vec![7f32; dim], &x, &post, &comb, rows, hc, dim);
        assert_eq!(&z[0..dim], &x[0..dim]);
        assert_eq!(&z[dim..2 * dim], &x[dim..2 * dim]);
    }

    #[test]
    fn engram_gate_is_signed_sqrt_of_the_dot() {
        let (hc, dim) = (1usize, 4usize);
        let h = vec![1.0f32; dim];
        let qw = vec![1.0f32; dim];
        let kw = vec![1.0f32; dim];
        let mut kv = vec![0f32; hc * dim + dim];
        kv[..dim].copy_from_slice(&h);
        let out = engram_forward(&h, &kv, &qw, &kw, 1, hc, dim, 1e-6, None);
        assert_eq!(out, vec![1.0, 1.0, 1.0, 1.0]);
        kv[dim..].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        let out2 = engram_forward(&h, &kv, &qw, &kw, 1, hc, dim, 1e-6, None);
        assert!(out2[0] > 1.0, "positive gate adds a positive value");
        assert_eq!(out2[1], 1.0);
        let out3 = engram_forward(&h, &kv, &qw, &kw, 1, hc, dim, 1e-6, Some(&[false]));
        assert_eq!(out3, h);
    }

    #[test]
    fn compressor_ratio2_pools_with_a_softmax_gate() {
        let (bsz, seqlen, dim, head_dim) = (1usize, 2usize, 1usize, 1usize);
        let x = vec![2.0f32, 4.0];
        let mut st = CompressorState::new(bsz, 2, head_dim);
        let (lat, rows) = compressor_forward(
            &x, bsz, seqlen, dim, &[1.0], Some(&[1.0]), &[1.0], head_dim, 2, 0, 1e-20,
            Some(&mut st),
        )
        .expect("a completed group yields a latent");
        assert_eq!(rows, 1);
        let (e2, e4) = ((2.0f32).exp(), (4.0f32).exp());
        let want = (2.0 * e2 + 4.0 * e4) / (e2 + e4);
        // the pooled value is RMS-normalised over a single channel
        assert!((lat[0].abs() - 1.0).abs() < 1e-4, "{}", lat[0]);
        assert!(want > 0.0);
    }

    #[test]
    fn compressor_ratio1_is_a_plain_projection() {
        let (lat, rows) =
            compressor_forward(&[3.0], 1, 1, 1, &[2.0], None, &[1.0], 1, 1, 0, 1e-20, None).unwrap();
        assert_eq!(rows, 1);
        assert!((lat[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn indexer_topk_respects_compress_lens_and_offset() {
        let (nh, hd, n_pos) = (1usize, 1usize, 3usize);
        let q = vec![1.0f32];
        let k = vec![1.0f32, 2.0, 3.0];
        let w = vec![1.0f32];
        let idx = indexer_topk(&q, &k, &w, 1, 1, nh, hd, n_pos, &[3], None, 2, 4, 1.0, 1.0);
        assert_eq!(idx, vec![5, 6]);
        let idx2 = indexer_topk(&q, &k, &w, 1, 1, nh, hd, n_pos, &[1], None, 2, 0, 1.0, 1.0);
        assert_eq!(idx2[0], 0);
        assert_eq!(idx2[1], -1);
    }
}
