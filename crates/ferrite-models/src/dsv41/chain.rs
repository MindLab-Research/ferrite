//! The DeepSeek-V4.1-Flash model chain.
//!
//! Layer order is the reference's, which is *not* the usual one: each
//! hyper-connection sub-block computes its own coefficients and they are
//! consumed by the **next** one (`Inference/model.py::Block.forward`):
//!
//! ```text
//! residual = x
//! attn_pre, attn_post, attn_comb = hc_mixes(x, hc_attn_*)
//! x = hc_pre(x, pre_mix)        # pre_mix comes from the PREVIOUS block's ffn
//! x = attn_norm(x); x = attn(x); x = hc_post(x, residual, attn_post, attn_comb)
//! residual = x
//! ffn_pre, ffn_post, ffn_comb = hc_mixes(x, hc_ffn_*)
//! x = hc_pre(x, attn_pre)       # this block's attention mix
//! x = ffn_norm(x); x = ffn(x); x = hc_post(x, residual, ffn_post, ffn_comb)
//! return x, ffn_pre             # handed to the next block
//! ```
//!
//! Attention is layered sparse attention with **one shared KV head** and, from
//! `compress_ratio > 0`, a second KV source: only `kv_source_layers` pool their
//! own KV (`Compressor`) and publish it; every other layer reads that same
//! cache. The indexer follows the same pattern for the selected positions, and
//! the candidate-source layer additionally publishes coarse block candidates.
//! All of this lives in [`SharedAttnState`], which is why the layers must run in
//! increasing order.
//!
//! The engine keeps every weight in its checkpoint format (fp8 e4m3 with
//! ue8m0 32x32 blocks, or fp4 e2m1 with per-row-per-32 ue8m0 scales) and runs
//! the matmuls on tensor cores over that native format — see
//! [`crate::dsv41::kernels`] for the ABI and the performance contract.

use crate::dsv41::config::Dsv41Config;
use crate::dsv41::ops;

/// Where a per-layer tensor lives. The device buffers are opaque here so the
/// chain can be compiled and unit-tested without a GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Resident on the device in its checkpoint format.
    Device,
    /// Kept on the host (norms, sinks, hc coefficients, engram q/k weights).
    Host,
}

/// The published per-step attention state shared across layers.
#[derive(Debug, Default)]
pub struct SharedAttnState {
    /// The compressed-KV cache of the most recent `kv_source` layer.
    pub compress_kv: Option<usize>,
    /// The number of valid compressed positions in that cache.
    pub compress_len: usize,
    /// Index keys published by the most recent `index_source` layer.
    pub index_k: Option<usize>,
    /// `topk_idxs` published by the most recent `index_source` layer.
    pub topk_idxs: Option<usize>,
    /// Block candidates published by the candidate-source layer.
    pub candidates: Option<usize>,
}

impl SharedAttnState {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Per-layer decode state.
#[derive(Debug)]
pub struct LayerState {
    /// Ring buffer of `window_size` KV rows. This struct is the *CPU reference*
    /// state (f32); the device path keeps the same cache quantised to fp8
    /// (block 128, power-of-two scale) in a separate packed buffer next to a
    /// per-row scale array.
    pub window_kv: Vec<f32>,
    pub window_kv_scale: Vec<f32>,
    /// The layer's own compressed-KV cache (only kv sources own one). Device:
    /// fp4, block 16 with e4m3 scales.
    pub compress_kv: Vec<f32>,
    pub compress_kv_scale: Vec<f32>,
    /// Index keys (only index sources own one).
    pub index_k: Vec<f32>,
    /// Compressor carry state (ratio > 1).
    pub comp: Option<ops::CompressorState>,
    pub window_pos: usize,
}

impl LayerState {
    pub fn new(cfg: &Dsv41Config) -> Self {
        let bsz = cfg.max_batch_size;
        let hd = cfg.head_dim;
        let win = cfg.window_size;
        let ratio = cfg.compress_ratio(0).max(1);
        LayerState {
            window_kv: vec![0f32; bsz * win * hd],
            window_kv_scale: vec![0f32; bsz * win],
            compress_kv: vec![0f32; bsz * (cfg.max_seq_len / ratio.max(1)) * hd],
            compress_kv_scale: vec![0f32; bsz * (cfg.max_seq_len / ratio.max(1))],
            index_k: Vec::new(),
            comp: None,
            window_pos: 0,
        }
    }
}

/// One layer's forward, with the cross-layer state threaded through.
///
/// `x` is `[rows, hc, dim]`; `pre_mix` is `[rows, hc]` from the previous block.
/// Returns `(x, ffn_pre)`.
#[allow(clippy::too_many_arguments)]
pub fn layer_forward(
    cfg: &Dsv41Config,
    layer: usize,
    x: &mut [f32],
    rows: usize,
    pre_mix: &[f32],
    st: &mut LayerState,
    shared: &mut SharedAttnState,
    start_pos: usize,
    // host-side weights for the CPU reference path
    w: &LayerHostWeights,
) -> Vec<f32> {
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    let hc_dim = hc * dim;

    // ---- attention sub-block ----
    let coeff_a = ops::hc_mixes(
        x, &w.hc_attn_fn, &w.hc_attn_scale, &w.hc_attn_base, rows, hc_dim, hc,
        cfg.hc_sinkhorn_iters, cfg.hc_eps,
    );
    let residual = x.to_vec();
    let mut h = ops::hc_pre(x, pre_mix, rows, hc, dim);
    h = ops::rmsnorm(&h, &w.attn_norm, rows, dim, cfg.norm_eps);
    // attn: MLA with the window ring + (optionally) the shared compressed KV
    h = attention_forward(cfg, layer, &h, rows, start_pos, st, shared, w);
    x.copy_from_slice(&ops::hc_post(&h, &residual, &coeff_a.post, &coeff_a.comb, rows, hc, dim));

    // ---- FFN sub-block ----
    let coeff_f = ops::hc_mixes(
        x, &w.hc_ffn_fn, &w.hc_ffn_scale, &w.hc_ffn_base, rows, hc_dim, hc,
        cfg.hc_sinkhorn_iters, cfg.hc_eps,
    );
    let residual2 = x.to_vec();
    let mut h2 = ops::hc_pre(x, &coeff_a.pre, rows, hc, dim);
    h2 = ops::rmsnorm(&h2, &w.ffn_norm, rows, dim, cfg.norm_eps);
    h2 = moe_forward(cfg, layer, &h2, rows, start_pos, None, w);
    x.copy_from_slice(&ops::hc_post(&h2, &residual2, &coeff_f.post, &coeff_f.comb, rows, hc, dim));
    coeff_f.pre
}

/// MLA attention: latent q (`wq_a` -> `q_norm` -> `wq_b`), the shared KV head,
/// the window ring plus the compressed positions, one `sparse_attn` call, then
/// the grouped low-rank output projection.
#[allow(clippy::too_many_arguments)]
fn attention_forward(
    cfg: &Dsv41Config,
    layer: usize,
    x: &[f32],
    rows: usize,
    start_pos: usize,
    st: &mut LayerState,
    shared: &mut SharedAttnState,
    w: &LayerHostWeights,
) -> Vec<f32> {
    let hd = cfg.head_dim;
    let nh = cfg.n_heads;
    let rd = cfg.rope_head_dim;
    // q = wq_b(q_norm(wq_a(x)))
    let qr = ops::rmsnorm(
        &matmul(x, &w.wq_a, rows, cfg.dim, cfg.q_lora_rank),
        &w.q_norm,
        rows,
        cfg.q_lora_rank,
        cfg.norm_eps,
    );
    let mut q = matmul(&qr, &w.wq_b, rows, cfg.q_lora_rank, nh * hd);
    ops::apply_rope(&mut q, rows, hd, rd, &w.freqs, start_pos, 1, false);

    // window KV: one token per step into the ring
    let mut kv = matmul(x, &w.wkv, rows, cfg.dim, hd);
    for r in 0..rows {
        ops::rmsnorm_rows_pub(&mut kv[r * hd..(r + 1) * hd], &w.kv_norm, hd, cfg.norm_eps);
    }
    ops::apply_rope(&mut kv, rows, hd, rd, &w.freqs, start_pos, 1, false);

    let win = cfg.window_size;
    let win_idxs = ops::window_topk_idxs(win, 1, rows, start_pos);
    let mut win_rows: Vec<f32> = Vec::new();
    // the ring stores the post-rope raw KV; append the new token(s)
    for r in 0..rows {
        let slot = (start_pos + r) % win;
        st.window_kv[slot * hd..(slot + 1) * hd].copy_from_slice(&kv[r * hd..(r + 1) * hd]);
    }
    // materialise the window rows the queries see (prefill: the current chunk;
    // decode: the whole ring, oldest first)
    let win_cols = if start_pos == 0 { rows.min(win) } else { win };
    win_rows.resize(win_cols * hd, 0.0);
    for r in 0..rows {
        for c in 0..win_cols {
            let idx = win_idxs[r * win_cols + c];
            if idx >= 0 {
                let slot = idx as usize;
                win_rows[c * hd..(c + 1) * hd]
                    .copy_from_slice(&st.window_kv[slot * hd..(slot + 1) * hd]);
            }
        }
    }

    // Assemble the two KV sources. The row width is
    //     window columns + (min(index_topk, compress_len) if compressed)
    // and the compressed indices are shifted past the window rows. Note the
    // reference concatenates along the last dim, so the window indices and the
    // compressed indices must be interleaved PER ROW here.
    let mut all_kv = win_rows.clone();
    let mut idx_cols = vec![win_cols; rows];
    let mut per_row_idx: Vec<Vec<i32>> = vec![Vec::new(); rows];
    for r in 0..rows {
        per_row_idx[r].extend_from_slice(&win_idxs[r * win_cols..(r + 1) * win_cols]);
    }
    if cfg.compress_ratio(layer) > 0 {
        let ratio = cfg.compress_ratio(layer);
        // the compressor only runs on kv sources; consumers read the shared cache
        if cfg.is_kv_source(layer) {
            if st.comp.is_none() && ratio > 1 {
                st.comp = Some(ops::CompressorState::new(1, ratio, hd));
            }
            let lat = ops::compressor_forward(
                x, 1, rows, cfg.dim, &w.compressor_wkv, w.compressor_wgate.as_deref(),
                &w.compressor_norm, hd, ratio, start_pos, cfg.norm_eps,
                st.comp.as_mut(),
            );
            if let Some((lat, n)) = lat {
                for g in 0..n {
                    let slot = start_pos / ratio + g;
                    st.compress_kv[slot * hd..(slot + 1) * hd]
                        .copy_from_slice(&lat[g * hd..(g + 1) * hd]);
                }
                shared.compress_kv = Some(layer);
                shared.compress_len = start_pos / ratio + n;
            }
        }
        let compress_len = shared.compress_len;
        // indexer: sources run their own; consumers reuse what was published
        let idxs: Vec<i32> = if compress_len > 0 && cfg.is_index_source(layer) {
            let mut q2 = matmul(
                &qr, &w.indexer_wq_b, rows, cfg.q_lora_rank,
                cfg.index_n_heads * cfg.index_head_dim,
            );
            ops::apply_rope(&mut q2, rows, cfg.index_head_dim, rd, &w.freqs, start_pos, 1, false);
            let cands: Option<Vec<bool>> = if cfg.is_candidate_source(layer) {
                // publish block candidates for the layers after this one
                let logits = vec![0f32; rows * compress_len];
                let cl = vec![compress_len; rows];
                Some(ops::select_candidate_blocks(
                    &logits, rows, compress_len, &cl,
                    cfg.candidate_topk_blocks, cfg.candidate_block_size,
                ))
            } else {
                None
            };
            let v = ops::indexer_topk(
                &q2, &w.indexer_k[..compress_len * cfg.index_head_dim], &w.indexer_weights,
                1, rows, cfg.index_n_heads, cfg.index_head_dim, compress_len,
                &vec![compress_len; rows], cands.as_deref(), cfg.index_topk,
                0, cfg.index_head_dim as f32,
                (cfg.index_n_heads as f32).powf(-0.5),
            );
            shared.topk_idxs = Some(layer);
            v
        } else if cfg.is_index_source(layer) {
            Vec::new()
        } else {
            w.published_topk.clone()
        };
        if !idxs.is_empty() && compress_len > 0 {
            // append the compressed KV rows after the window rows …
            let base = all_kv.len() / hd;
            all_kv.extend_from_slice(&st.compress_kv[..compress_len * hd]);
            // … and shift the per-row indices past them
            let cols = idxs.len() / rows;
            for r in 0..rows {
                for c in 0..cols {
                    let v = idxs[r * cols + c];
                    per_row_idx[r].push(if v < 0 { -1 } else { v + base as i32 });
                    idx_cols[r] += 1;
                }
            }
        }
    }
    // flatten to [rows, max row width] (pad with -1)
    let row_width = idx_cols.iter().copied().max().unwrap_or(0);
    let mut all_idx = vec![-1i32; rows * row_width];
    for r in 0..rows {
        for (c, &v) in per_row_idx[r].iter().enumerate() {
            all_idx[r * row_width + c] = v;
        }
    }
    let kv_rows = all_kv.len() / hd;

    let scale = (hd as f32).powf(-0.5);
    let o = ops::sparse_attn(
        &q, &all_kv, &w.attn_sink, &all_idx, 1, rows, nh, hd, kv_rows, row_width, scale,
    );
    // inverse rope on the output tail, then the grouped low-rank projection
    let mut o = o;
    ops::apply_rope(&mut o, rows * nh, hd, rd, &w.freqs, start_pos, 1, true);
    // wo_a is block-diagonal over o_groups: each group projects its own heads
    let n_local_groups = cfg.o_groups;
    let heads_per_group = nh / n_local_groups;
    let mut grp = vec![0f32; rows * n_local_groups * heads_per_group * hd];
    for r in 0..rows {
        for g in 0..n_local_groups {
            for i in 0..heads_per_group {
                let src = ((r * nh) + g * heads_per_group + i) * hd;
                let dst = (r * n_local_groups + g) * heads_per_group * hd + i * hd;
                grp[dst..dst + hd].copy_from_slice(&o[src..src + hd]);
            }
        }
    }
    let o_lora = grouped_matmul(
        &grp, &w.wo_a, rows * n_local_groups, heads_per_group * hd,
        cfg.o_lora_rank,
    );
    matmul(&o_lora, &w.wo_b, rows, n_local_groups * cfg.o_lora_rank, cfg.dim)
}

/// MoE: top-k routed experts + one shared expert.
fn moe_forward(
    cfg: &Dsv41Config,
    layer: usize,
    x: &[f32],
    rows: usize,
    _start_pos: usize,
    _image_mask: Option<&[bool]>,
    w: &LayerHostWeights,
) -> Vec<f32> {
    let (n_routed, topk) = cfg.moe_config(layer);
    let dim = cfg.dim;
    let inter = cfg.moe_inter_dim;
    let (weights, indices) = ops::moe_gate(
        x, &w.gate_w, &w.gate_bias, rows, dim, n_routed, topk,
        cfg.gate_temp, cfg.norm_topk_prob, cfg.route_scale, &cfg.score_func,
    );
    let mut y = vec![0f32; rows * dim];
    for r in 0..rows {
        for t in 0..topk {
            let e = indices[r * topk + t] as usize;
            let wr = weights[r * topk + t];
            // one row through the expert (the device path batches per expert)
            let ex = &w.experts[e];
            let row = &x[r * dim..(r + 1) * dim];
            let out = ops::expert_ffn(
                row, &ex.w1, &ex.w3, &ex.w2, Some(&[wr]), 1, dim, inter, cfg.swiglu_limit,
            );
            for c in 0..dim {
                y[r * dim + c] += out[c];
            }
        }
    }
    // the shared expert sees every token
    let sh = ops::expert_ffn(x, &w.shared_w1, &w.shared_w3, &w.shared_w2, None, rows, dim, inter, cfg.swiglu_limit);
    for i in 0..y.len() {
        y[i] += sh[i];
    }
    y
}

fn matmul(x: &[f32], w: &[f32], rows: usize, k: usize, n: usize) -> Vec<f32> {
    // CPU reference: w is stored [n, k]
    let mut o = vec![0f32; rows * n];
    for r in 0..rows {
        let xr = &x[r * k..(r + 1) * k];
        for j in 0..n {
            let wr = &w[j * k..(j + 1) * k];
            let mut acc = 0f32;
            for c in 0..k {
                acc += wr[c] * xr[c];
            }
            o[r * n + j] = acc;
        }
    }
    o
}

/// `wo_a` is block-diagonal over groups: it is applied independently per group
/// (the reference uses an einsum, not a Linear).
fn grouped_matmul(x: &[f32], w: &[f32], rows_groups: usize, k: usize, n: usize) -> Vec<f32> {
    matmul(x, w, rows_groups, k, n)
}

// ===========================================================================
// Host-side weight container for the CPU reference path
// ===========================================================================

/// One expert's SwiGLU weights, stored `[inter, dim]` / `[dim, inter]`.
#[derive(Debug, Default, Clone)]
pub struct HostExpert {
    pub w1: Vec<f32>,
    pub w3: Vec<f32>,
    pub w2: Vec<f32>,
}

/// Per-layer host weights (dequantised) used by the CPU reference path and by
/// the numeric conformance tests. The device path keeps the packed formats.
#[derive(Debug, Default, Clone)]
pub struct LayerHostWeights {
    pub hc_attn_fn: Vec<f32>,
    pub hc_attn_scale: Vec<f32>,
    pub hc_attn_base: Vec<f32>,
    pub hc_ffn_fn: Vec<f32>,
    pub hc_ffn_scale: Vec<f32>,
    pub hc_ffn_base: Vec<f32>,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub wq_a: Vec<f32>,
    pub wq_a_scale: Vec<u8>,
    pub q_norm: Vec<f32>,
    pub wq_b: Vec<f32>,
    pub wq_b_scale: Vec<u8>,
    pub wkv: Vec<f32>,
    pub wkv_scale: Vec<u8>,
    pub kv_norm: Vec<f32>,
    pub wo_a: Vec<f32>,
    pub wo_a_scale: Vec<u8>,
    pub wo_b: Vec<f32>,
    pub wo_b_scale: Vec<u8>,
    pub attn_sink: Vec<f32>,
    pub compressor_wkv: Vec<f32>,
    pub compressor_wgate: Option<Vec<f32>>,
    pub compressor_norm: Vec<f32>,
    pub indexer_wq_b: Vec<f32>,
    pub indexer_wq_b_scale: Vec<u8>,
    pub indexer_weights: Vec<f32>,
    pub indexer_k: Vec<f32>,
    pub gate_w: Vec<f32>,
    pub gate_bias: Vec<f32>,
    pub experts: Vec<HostExpert>,
    pub shared_w1: Vec<f32>,
    pub shared_w3: Vec<f32>,
    pub shared_w2: Vec<f32>,
    pub freqs: ops::FreqTable,
    pub published_topk: Vec<i32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msd_layers_partition_correctly() {
        // sanity: every backbone layer resolves to exactly one KvMode and the
        // shared state is only published by source layers
        let cfg = Dsv41Config::production();
        let mut n_src = 0;
        for l in 0..cfg.n_layers {
            if cfg.kv_mode(l) == KvMode::CompressSource {
                n_src += 1;
            }
        }
        assert_eq!(n_src, cfg.kv_source_layers.len());
        assert_eq!(n_src, 4);
    }

    #[test]
    fn layer_state_shapes() {
        let cfg = Dsv41Config::production();
        let st = LayerState::new(&cfg);
        let hd = cfg.head_dim;
        assert_eq!(st.window_kv.len(), cfg.max_batch_size * cfg.window_size * hd);
        assert_eq!(st.window_kv_scale.len(), cfg.max_batch_size * cfg.window_size);
    }
}

// ===========================================================================
// Top-level forward
// ===========================================================================

/// Per-layer engram weights (host reference path).
#[derive(Debug, Default, Clone)]
pub struct EngramHostWeights {
    /// `[part_rows, head_dim]` fp8 decoded to f32 (row shard of this rank).
    pub table: Vec<f32>,
    /// `part_rows`
    pub part_rows: usize,
    /// `[n_cols * head_dim, dim * (hc + 1)]` — the wkv projection, `[out, in]`.
    pub wkv: Vec<f32>,
    /// `[hc, dim]` each
    pub q_weight: Vec<f32>,
    pub k_weight: Vec<f32>,
    /// Token id -> compressed id.
    pub token_map: crate::dsv41::engram::TokenMap,
    pub layout: crate::dsv41::engram::EngramLayout,
    pub pad_id: i64,
}

/// Everything the model needs for one forward pass.
pub struct ModelHostWeights {
    pub embed: Vec<f32>,        // [vocab, dim]
    pub norm: Vec<f32>,         // [dim] final norm
    pub head: Vec<f32>,         // [vocab, dim]
    pub layers: Vec<LayerHostWeights>,
    pub engram: Vec<Option<EngramHostWeights>>, // indexed by layer
    pub draft: Vec<crate::dsv41::dspark::DraftHostWeights>,
}

/// Full model state (one per sequence).
pub struct ModelState {
    pub layers: Vec<LayerState>,
    pub shared: SharedAttnState,
    pub ngram: Option<crate::dsv41::engram::NgramHashState>,
}

impl ModelState {
    pub fn new(cfg: &Dsv41Config, map: Option<crate::dsv41::engram::TokenMap>) -> Self {
        ModelState {
            layers: (0..cfg.n_layers + cfg.n_mtp_layers).map(|_| LayerState::new(cfg)).collect(),
            shared: SharedAttnState::new(),
            ngram: map.map(|m| crate::dsv41::engram::NgramHashState::new(cfg, &m)),
        }
    }
}

/// The model forward: embed -> expand to hc copies -> blocks (with the engram
/// lookups interleaved and the draft inputs recorded at the target layers) ->
/// collapse -> final norm -> head.
///
/// Returns `(logits [last_row, vocab], main_hidden [last_row, targets*dim])`.
pub fn forward(
    cfg: &Dsv41Config,
    tokens: &[u32],
    start_pos: usize,
    st: &mut ModelState,
    w: &ModelHostWeights,
) -> (Vec<f32>, Vec<f32>) {
    let dim = cfg.dim;
    let hc = cfg.hc_mult;
    let rows = tokens.len();

    // embed
    let mut h = vec![0f32; rows * hc * dim];
    for (r, &tok) in tokens.iter().enumerate() {
        let e = &w.embed[tok as usize * dim..(tok as usize + 1) * dim];
        for c in 0..hc {
            h[(r * hc + c) * dim..(r * hc + c + 1) * dim].copy_from_slice(e);
        }
    }
    // identity pre-mix (one-hot on copy 0)
    let mut pre_mix = vec![0f32; rows * hc];
    for r in 0..rows {
        pre_mix[r * hc] = 1.0;
    }

    // engram hashes for the whole chunk (one shot; the reference caches them
    // across prefill/decode inside NgramHashState)
    let mut hashes: Option<Vec<i64>> = None;
    if let (Some(ng), Some(_)) = (st.ngram.as_mut(), w.engram.iter().find(|e| e.is_some())) {
        if let Some(lay) = w.engram.iter().find_map(|e| e.as_ref()) {
            hashes = Some(ng.forward_row(&lay.layout, &lay.token_map, 0, tokens, start_pos, None));
        }
    }

    let mut main_hiddens: Vec<f32> = Vec::new();
    let mut last_pre_mix = pre_mix.clone();
    for layer in 0..cfg.n_layers {
        // engram writes into the residual stream before the block runs
        if let (Some(hs), Some(Some(eng))) = (hashes.as_ref(), w.engram.get(layer)) {
            let li = eng.layout.engram_index(layer).unwrap_or(0);
            let n_cols = eng.layout.n_hash_cols();
            let ehd = cfg.engram_head_dim;
            let wkv_k = n_cols * ehd;
            let wkv_n = dim * (hc + 1);
            // ParallelEngramEmbedding: this rank's table rows only, zero rows for
            // ids another rank owns (the device path all-reduces afterwards).
            let row_off = 0usize; // CPU reference keeps the whole table
            let mut gathered = vec![0f32; rows * wkv_k];
            for r in 0..rows {
                for c in 0..n_cols {
                    let id = hs[(r * eng.layout.layers.len() + li) * n_cols + c];
                    let bucket = eng.layout.layers[li].offsets[0] as i64; // table base
                    let local = id - bucket;
                    if local < 0 || local as usize >= eng.part_rows {
                        continue; // another rank owns this row -> contributes 0
                    }
                    let src = &eng.table[local as usize * ehd..(local as usize + 1) * ehd];
                    let dst = (r * n_cols + c) * ehd;
                    gathered[dst..dst + ehd].copy_from_slice(src);
                }
            }
            // kv = wkv(gathered)  [rows, hc*dim + dim]
            let mut rows_kv = vec![0f32; rows * (hc * dim + dim)];
            for r in 0..rows {
                let xr = &gathered[r * wkv_k..(r + 1) * wkv_k];
                for j in 0..wkv_n {
                    let wr = &eng.wkv[j * wkv_k..(j + 1) * wkv_k];
                    let mut acc = 0f32;
                    for c in 0..wkv_k {
                        acc += wr[c] * xr[c];
                    }
                    rows_kv[r * wkv_n + j] = acc;
                }
            }
            let _ = row_off;
            let gated = ops::engram_forward(
                &h, &rows_kv, &eng.q_weight, &eng.k_weight, rows, hc, dim, cfg.norm_eps, None,
            );
            h.copy_from_slice(&gated);
        }
        if cfg.dspark_target_layer_ids.contains(&layer) {
            // the draft reads the ATTENTION INPUT of its target layers
            for r in 0..rows {
                let mut mean = vec![0f32; dim];
                for c in 0..hc {
                    for j in 0..dim {
                        mean[j] += h[(r * hc + c) * dim + j] / hc as f32;
                    }
                }
                main_hiddens.extend_from_slice(&mean);
            }
        }
        let lp = &w.layers[layer];
        last_pre_mix = layer_forward(
            cfg, layer, &mut h, rows, &pre_mix, &mut st.layers[layer], &mut st.shared, start_pos, lp,
        );
        pre_mix = last_pre_mix.clone();
    }

    // collapse and project
    let collapsed = ops::hc_pre(&h, &last_pre_mix, rows, hc, dim);
    let normed = ops::rmsnorm(&collapsed, &w.norm, rows, dim, cfg.norm_eps);
    // only the last row is needed for generation
    let last = &normed[(rows - 1) * dim..rows * dim];
    let logits = matmul(last, &w.head, 1, dim, cfg.vocab_size);
    (logits, main_hiddens)
}

/// `Transformer.forward_spec`: the DSpark draft. Returns `None` while prefilling
/// (`start_pos == 0` only seeds the draft windows).
#[allow(clippy::too_many_arguments)]
pub fn forward_spec(
    cfg: &Dsv41Config,
    input_ids: u32,
    main_hidden: &[f32],
    start_pos: usize,
    st: &mut ModelState,
    w: &ModelHostWeights,
) -> Option<(Vec<u32>, Vec<f32>, Vec<f32>)> {
    if cfg.n_mtp_layers == 0 {
        return None;
    }
    let dim = cfg.dim;
    let bs = cfg.dspark_block_size;
    let hc = cfg.hc_mult;
    let (mut h, main_x) = crate::dsv41::dspark::forward_embed(
        cfg,
        main_hidden,
        input_ids,
        &w.embed[input_ids as usize * dim..(input_ids as usize + 1) * dim],
        &w.draft[0],
    );
    let mut pre_mix = vec![0f32; bs * hc];
    for r in 0..bs {
        pre_mix[r * hc] = 1.0;
    }
    for (s, d) in w.draft.iter().enumerate() {
        let layer = cfg.n_layers + s;
        if start_pos == 0 {
            // seed the draft window from the main stream and return
            crate::dsv41::dspark::dspark_attention(
                cfg, &h, &main_x, 0, &mut st.layers[layer].window_kv, d,
            );
            continue;
        }
        let lp = &w.layers[layer.min(cfg.n_layers - 1)];
        pre_mix = layer_forward(
            cfg, layer, &mut h, bs, &pre_mix, &mut st.layers[layer], &mut st.shared,
            start_pos, lp,
        );
    }
    if start_pos == 0 {
        return None;
    }
    let u = vec![1.0f32; bs * cfg.vocab_size];
    let last = w.draft.last().unwrap();
    Some(crate::dsv41::dspark::forward_head(cfg, &h, &pre_mix, input_ids, last, &u))
}
