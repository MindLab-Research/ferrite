//! DSpark: the draft stage that lives under the checkpoint's `mtp.*` namespace.
//!
//! Three blocks (the released config's `num_nextn_predict_layers`) draft a
//! block of `dspark_block_size` tokens in one forward pass:
//!
//! 1. [`forward_embed`] turns the target layers' attention inputs into the
//!    draft input. The reference records `h.mean(dim=2)` at
//!    `dspark_target_layer_ids` (37, 38, 39) — the attention input, **not** the
//!    output — concatenates them and projects with `main_proj` + `main_norm`.
//!    The draft ids are all `noise_token_id` except the first, which is the
//!    token just sampled by the backbone.
//! 2. The draft blocks run through the normal block structure, but attention is
//!    [`DSparkAttention`]: it keeps a window cache seeded from the *main*
//!    stream's KV, and every draft query additionally attends to its own block
//!    (`window_size + i` ids).
//! 3. [`forward_head`] runs the first block's `hc_pre` collapse, the head, and
//!    then a `block_size`-step **sequential** loop: at step i the Markov head
//!    reads the already-sampled token i, biases the logits, samples token i+1.
//!    A confidence head scores the draft from the same block.
//!
//! Only the draft *forward* is implemented here (matching the reference, which
//! ships no speculative-decoding loop either).

use crate::config::Dsv41Config;
use crate::ops;

/// `get_dspark_topk_idxs`: the window slots plus the draft block itself.
///
/// `matrix = [arange(min(window, start_pos + 1)), window + arange(block_size)]`,
/// broadcast over the batch and over every draft query in the block.
pub fn dspark_topk_idxs(
    window: usize,
    bsz: usize,
    block_size: usize,
    start_pos: usize,
) -> Vec<i32> {
    assert!(start_pos > 0, "DSpark drafts only exist after a prefill");
    let n_win = window.min(start_pos + 1);
    let cols = n_win + block_size;
    let mut row = Vec::with_capacity(cols);
    for i in 0..n_win {
        row.push(i as i32);
    }
    for i in 0..block_size {
        row.push((window + i) as i32);
    }
    let mut out = vec![0i32; bsz * block_size * cols];
    for b in 0..bsz {
        for q in 0..block_size {
            out[(b * block_size + q) * cols..(b * block_size + q + 1) * cols]
                .copy_from_slice(&row);
        }
    }
    out
}

/// The draft block's attention: q/k/v from the draft stream, the window cache
/// seeded from the main stream.
#[allow(clippy::too_many_arguments)]
pub fn dspark_attention(
    cfg: &Dsv41Config,
    x: &[f32],      // [block_size, dim]
    main_x: &[f32], // [1, dim] the target layers' collapsed attention input
    start_pos: usize,
    window_cache: &mut [f32], // [window, head_dim]
    w: &DraftHostWeights,
) -> Vec<f32> {
    let hd = cfg.head_dim;
    let nh = cfg.n_heads;
    let rd = cfg.rope_head_dim;
    let bs = cfg.dspark_block_size;
    let win = cfg.window_size;

    if start_pos == 0 {
        // prefill only seeds the ring from the main stream
        let mut mk = matmul(main_x, &w.wkv, 1, cfg.dim, hd);
        ops::rmsnorm_rows_pub(&mut mk, &w.kv_norm, hd, cfg.norm_eps);
        ops::apply_rope(&mut mk, 1, hd, rd, &w.freqs, 0, 1, false);
        window_cache[..hd].copy_from_slice(&mk[..hd]);
        return x.to_vec();
    }

    // q/k/v of the draft tokens
    let qr = ops::rmsnorm(
        &matmul(x, &w.wq_a, bs, cfg.dim, cfg.q_lora_rank),
        &w.q_norm,
        bs,
        cfg.q_lora_rank,
        cfg.norm_eps,
    );
    let mut q = matmul(&qr, &w.wq_b, bs, cfg.q_lora_rank, nh * hd);
    // the reference rotates the queries at [start_pos + seqlen, + block_size)
    ops::apply_rope(&mut q, bs, hd, rd, &w.freqs, start_pos + bs, 1, false);
    let mut kv = matmul(x, &w.wkv, bs, cfg.dim, hd);
    ops::rmsnorm_rows_pub(&mut kv, &w.kv_norm, hd, cfg.norm_eps);
    ops::apply_rope(&mut kv, bs, hd, rd, &w.freqs, start_pos + bs, 1, false);

    // main KV goes into the ring; the draft block is appended after it
    let mut mk = matmul(main_x, &w.wkv, 1, cfg.dim, hd);
    ops::rmsnorm_rows_pub(&mut mk, &w.kv_norm, hd, cfg.norm_eps);
    ops::apply_rope(&mut mk, 1, hd, rd, &w.freqs, start_pos, 1, false);
    let slot = start_pos % win;
    window_cache[slot * hd..(slot + 1) * hd].copy_from_slice(&mk[..hd]);
    let mut all_kv = window_cache.to_vec();
    all_kv.extend_from_slice(&kv);

    let idxs = dspark_topk_idxs(win, 1, bs, start_pos);
    let cols = win.min(start_pos + 1) + bs;
    let scale = (hd as f32).powf(-0.5);
    let mut o = ops::sparse_attn(&q, &all_kv, &w.attn_sink, &idxs, 1, bs, nh, hd, win + bs, cols, scale);
    ops::apply_rope(&mut o, bs * nh, hd, rd, &w.freqs, start_pos + bs, 1, true);
    // grouped low-rank output projection (identical geometry to the backbone)
    let groups = cfg.o_groups;
    let hpg = nh / groups;
    let mut grp = vec![0f32; bs * groups * hpg * hd];
    for r in 0..bs {
        for g in 0..groups {
            for i in 0..hpg {
                let src = (r * nh + g * hpg + i) * hd;
                let dst = (r * groups + g) * hpg * hd + i * hd;
                grp[dst..dst + hd].copy_from_slice(&o[src..src + hd]);
            }
        }
    }
    let o_lora = matmul(&grp, &w.wo_a, bs * groups, hpg * hd, cfg.o_lora_rank);
    let wo_b = matmul(&o_lora, &w.wo_b, bs, groups * cfg.o_lora_rank, cfg.dim);
    wo_b
}

/// `DSparkBlock.forward_embed`: build the draft input from the target layers'
/// hidden states.
pub fn forward_embed(
    cfg: &Dsv41Config,
    main_hidden: &[f32], // [1, target_layers * dim]
    _input_ids: u32,
    embed_row: &[f32],  // [dim] the embedding of the drafted token
    w: &DraftHostWeights,
) -> (Vec<f32>, Vec<f32>) {
    let bs = cfg.dspark_block_size;
    let dim = cfg.dim;
    let hc = cfg.hc_mult;
    // main_x = main_norm(main_proj(main_hidden))
    let mut main_x = matmul(
        main_hidden,
        &w.main_proj,
        1,
        dim * cfg.dspark_target_layer_ids.len(),
        dim,
    );
    ops::rmsnorm_rows_pub(&mut main_x, &w.main_norm, dim, cfg.norm_eps);
    // draft ids: noise everywhere, the sampled token first
    let mut x = vec![0f32; bs * dim];
    for i in 0..bs {
        let row = if i == 0 { embed_row } else { &w.noise_embed };
        x[i * dim..(i + 1) * dim].copy_from_slice(row);
    }
    // expand to hc copies
    let mut xh = vec![0f32; bs * hc * dim];
    for r in 0..bs {
        for c in 0..hc {
            xh[(r * hc + c) * dim..(r * hc + c + 1) * dim]
                .copy_from_slice(&x[r * dim..(r + 1) * dim]);
        }
    }
    (xh, main_x)
}

/// `DSparkBlock.forward_head`: collapse, head, then the sequential Markov loop
/// and the confidence score.
#[allow(clippy::too_many_arguments)]
pub fn forward_head(
    cfg: &Dsv41Config,
    x: &[f32],        // [block_size, hc, dim]
    pre_mix: &[f32],  // [block_size, hc]
    input_ids: u32,
    w: &DraftHostWeights,
    u: &[f32], // exponential(1) draws for sampling, [bs, vocab]
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let bs = cfg.dspark_block_size;
    let dim = cfg.dim;
    let vocab = cfg.vocab_size;
    let h = ops::hc_pre(x, pre_mix, bs, cfg.hc_mult, dim);
    let normed = ops::rmsnorm(&h, &w.norm, bs, dim, cfg.norm_eps);
    // logits = head(normed)  (the head weight is [vocab, dim])
    let mut logits = matmul(&normed, &w.head, bs, dim, vocab);

    let mut output_ids = vec![0u32; bs + 1];
    output_ids[0] = input_ids;
    let mut markov_embeds = vec![0f32; bs * cfg.dspark_markov_rank];
    for i in 0..bs {
        // markov_head(output_ids[:, i]) -> (logits_bias, embed)
        let tok = output_ids[i] as usize;
        let er = &w.markov_embed[tok * cfg.dspark_markov_rank..(tok + 1) * cfg.dspark_markov_rank];
        markov_embeds[i * cfg.dspark_markov_rank..(i + 1) * cfg.dspark_markov_rank]
            .copy_from_slice(er);
        // bias = markov_head.head(embed)  [vocab]
        for v in 0..vocab {
            let wr = &w.markov_head[v * cfg.dspark_markov_rank..(v + 1) * cfg.dspark_markov_rank];
            let mut acc = 0f32;
            for c in 0..cfg.dspark_markov_rank {
                acc += wr[c] * er[c];
            }
            logits[i * vocab + v] += acc;
        }
        output_ids[i + 1] = ops::gumbel_argmax(
            &logits[i * vocab..(i + 1) * vocab],
            vocab,
            cfg.temperature,
            &u[i * vocab..(i + 1) * vocab],
        );
    }
    // confidence = proj(cat([h, markov_embed]))  -> [bs]
    let mr = cfg.dspark_markov_rank;
    let mut confidence = vec![0f32; bs];
    for i in 0..bs {
        let mut acc = 0f32;
        for c in 0..dim {
            acc += w.confidence_proj[c] * h[i * dim + c];
        }
        for c in 0..mr {
            acc += w.confidence_proj[dim + c] * markov_embeds[i * mr + c];
        }
        confidence[i] = acc;
    }
    (output_ids, logits, confidence)
}

fn matmul(x: &[f32], w: &[f32], rows: usize, k: usize, n: usize) -> Vec<f32> {
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

/// Host-side draft weights (the CPU reference path; the device path keeps the
/// checkpoint formats and runs the MMAs natively).
#[derive(Debug, Default, Clone)]
pub struct DraftHostWeights {
    pub wq_a: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub wq_b: Vec<f32>,
    pub wkv: Vec<f32>,
    pub kv_norm: Vec<f32>,
    pub wo_a: Vec<f32>,
    pub wo_b: Vec<f32>,
    pub attn_sink: Vec<f32>,
    pub main_proj: Vec<f32>,
    pub main_norm: Vec<f32>,
    pub norm: Vec<f32>,
    pub head: Vec<f32>,
    pub markov_embed: Vec<f32>,
    pub markov_head: Vec<f32>,
    pub confidence_proj: Vec<f32>,
    pub noise_embed: Vec<f32>,
    pub freqs: ops::FreqTable,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_idxs_are_window_plus_the_draft_block() {
        let win = 4;
        let bs = 3;
        let start = 2; // n_win = min(4, 3) = 3
        let idx = dspark_topk_idxs(win, 1, bs, start);
        let cols = 3 + bs;
        assert_eq!(idx.len(), bs * cols);
        // every query sees the same row
        for q in 0..bs {
            let row = &idx[q * cols..(q + 1) * cols];
            assert_eq!(row, &[0, 1, 2, 4, 5, 6]);
        }
    }

    #[test]
    fn topk_idxs_cap_the_window_at_start_pos() {
        // start_pos 1 -> only one valid window slot
        let bs = 5;
        let idx = dspark_topk_idxs(128, 1, bs, 1);
        let cols = 2 + bs;
        assert_eq!(idx.len(), bs * cols);
        // n_win = min(128, start_pos + 1) = 2 valid window slots
        assert_eq!(&idx[0..cols], &[0, 1, 128, 129, 130, 131, 132]);
        // every draft query in the block sees the same slots
        assert_eq!(&idx[cols..2 * cols], &idx[0..cols]);
    }

    #[test]
    fn forward_head_samples_the_first_token_from_the_input() {
        // a tiny draft: bs 2, dim 2, hc 1, vocab 3, markov rank 1
        let mut cfg = Dsv41Config::production();
        cfg.dspark_block_size = 2;
        cfg.dim = 2;
        cfg.hc_mult = 1;
        cfg.vocab_size = 8;
        cfg.dspark_markov_rank = 1;
        cfg.temperature = 0.0; // greedy
        let dim = cfg.dim;
        let bs = cfg.dspark_block_size;
        let vocab = cfg.vocab_size;
        // hc_pre with a one-hot pre gives the first copy
        let x = vec![1.0f32; bs * dim];
        let pre = vec![1.0f32; bs];
        let w = DraftHostWeights {
            norm: vec![1.0; dim],
            // head: row v has weight v+1 -> argmax is the last vocab entry
            head: (0..vocab).flat_map(|v| vec![(v + 1) as f32; dim]).collect(),
            markov_head: vec![0.0; vocab], // no bias
            markov_embed: vec![1.0; vocab],
            confidence_proj: vec![1.0; dim + 1],
            ..Default::default()
        };
        let u = vec![1.0f32; bs * vocab];
        let (ids, logits, conf) = forward_head(&cfg, &x, &pre, 7, &w, &u);
        assert_eq!(ids.len(), bs + 1);
        assert_eq!(ids[0], 7, "the first draft id is the backbone's token");
        assert!((ids[0] as usize) < vocab);
        // greedy argmax of the biased logits -> the largest head row
        assert_eq!(ids[1], (vocab - 1) as u32);
        assert_eq!(ids[2], (vocab - 1) as u32);
        assert_eq!(logits.len(), bs * vocab);
        assert_eq!(conf.len(), bs);
    }
}
