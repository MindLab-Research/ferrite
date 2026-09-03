//! DCP attention merge infrastructure — the collective math behind decode
//! context parallelism (per-rank partial attention + all-gather + LSE
//! merge), adapted from the page-level 2D-resarding methodology.
//!
//! Under DCP, each rank holds `1/N_dcp` of the KV pages and computes a
//! **partial attention** over its shard only. The full output is
//! reconstructed by merging the per-rank partials:
//!
//! ```text
//! O_full = Σ_d w_d · O_d,   w_d = exp(LSE_d) / Σ_d exp(LSE_d)
//! LSE_merged = log Σ_d exp(LSE_d)
//! ```
//!
//! This is mathematically exact (soft-max denominator factorises over the
//! page partition): each partial carries its local-softmax output `O_d`
//! and its log-sum-exp `LSE_d`; the merge is a stable log-sum-exp over
//! ranks. On multi-device deployments the partials arrive via all-gather
//! and the scalar LSEs via all-reduce; here the CPU reference implements
//! the merge directly, and the equivalence test proves
//! `merge(partials) == full attention` for any page partition.
//!
//! EAGLE/MTP verification under DCP relies on the same merge: per-rank
//! partials merge deterministically (commutative log-sum-exp), so every
//! rank arrives at identical logits → identical acceptance decisions.

use ferrite_types::{Result, Shape, Tensor};

/// One DCP rank's partial attention output.
///
/// - `o`: `[n, heads, dv]` — local-softmax weighted sum over THIS rank's
///   KV shard (normalised by the local denominator, so it is scale-free)
/// - `lse`: `[n, heads]` — the local log-sum-exp
///   `m + log Σ exp(s − m)` (max-shifted for stability)
#[derive(Debug, Clone)]
pub struct PartialAttn {
    pub o: Tensor,
    pub lse: Tensor,
}

/// Compute one rank's partial attention over its KV shard.
///
/// Inputs: `q [n, h, dq]`, `k [t, h, dq]`, `v [t, h, dv]` where `t` is
/// the number of KV tokens this rank holds (the page-filtered subset —
/// already local, no indices needed). `dq == dk` (nope-only MLA).
///
/// Per `(n, h)`: scores = q·k / √dq; m = max(scores);
/// `o = Σ softmax_local(s) · v` (local-softmax normalised);
/// `lse = m + log Σ exp(s − m)`.
pub fn sparse_attn_partial(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<PartialAttn> {
    let n = q.shape.0.first().copied().unwrap_or(0);
    let h = q.shape.0.get(1).copied().unwrap_or(0);
    let dq = *q.shape.0.last().unwrap_or(&0);
    let t = k.shape.0.first().copied().unwrap_or(0);
    let dv = *v.shape.0.last().unwrap_or(&0);
    if dq != *k.shape.0.last().unwrap_or(&0) {
        return Err(ferrite_types::FerriteError::InvalidArg(format!(
            "dcp partial: dq {dq} != dk (nope-only MLA expects equal)"
        )));
    }
    if v.shape.0.first().copied().unwrap_or(0) != t {
        return Err(ferrite_types::FerriteError::InvalidArg("k/v token count mismatch".into()));
    }
    let scale = 1.0 / (dq as f32).sqrt();
    let qs = q.as_slice();
    let ks = k.as_slice();
    let vs = v.as_slice();
    let mut o = vec![0.0f32; n * h * dv];
    let mut lse = vec![f32::NEG_INFINITY; n * h]; // -inf: empty shard contributes weight 0
    for i in 0..n {
        for hd in 0..h {
            let qh = &qs[i * h * dq + hd * dq..i * h * dq + (hd + 1) * dq];
            // pass 1: scores + max (streaming, no allocation)
            let mut m = f32::NEG_INFINITY;
            for p in 0..t {
                let kh = &ks[p * h * dq + hd * dq..p * h * dq + (hd + 1) * dq];
                let s: f32 = (0..dq).map(|l| qh[l] * kh[l]).sum::<f32>() * scale;
                if s > m {
                    m = s;
                }
            }
            if t == 0 {
                continue; // empty shard: keep -inf LSE, zero o
            }
            // pass 2: exp-sum + weighted sum
            let mut denom = 0.0f32;
            let mut acc = vec![0.0f32; dv];
            for p in 0..t {
                let kh = &ks[p * h * dq + hd * dq..p * h * dq + (hd + 1) * dq];
                let s: f32 = (0..dq).map(|l| qh[l] * kh[l]).sum::<f32>() * scale;
                let e = (s - m).exp();
                denom += e;
                let vh = &vs[p * h * dv + hd * dv..p * h * dv + (hd + 1) * dv];
                for (j, vv) in vh.iter().enumerate() {
                    acc[j] += e * vv;
                }
            }
            // local-softmax normalised output + LSE
            let inv = 1.0 / denom;
            for j in 0..dv {
                o[i * h * dv + hd * dv + j] = acc[j] * inv;
            }
            lse[i * h + hd] = m + denom.ln();
        }
    }
    Ok(PartialAttn {
        o: Tensor::from_f32(Shape::new([n, h, dv]), o),
        lse: Tensor::from_f32(Shape::new([n, h]), lse),
    })
}

/// Merge N DCP partials into the full attention output (all-gather + LSE
/// merge, numerically stable):
///
/// `M = max_d LSE_d`, `w_d = exp(LSE_d − M)`,
/// `O = Σ w_d O_d / Σ w_d`, `LSE_merged = M + log Σ w_d`.
///
/// Returns `(o [n, h, dv], lse [n, h])` — the merged LSE is useful for
/// speculative-decode verification (identical on every rank).
pub fn lse_merge(partials: &[PartialAttn]) -> Result<(Tensor, Tensor)> {
    if partials.is_empty() {
        return Err(ferrite_types::FerriteError::InvalidArg("dcp merge: no partials".into()));
    }
    let first = &partials[0];
    let n = first.o.shape.0[0];
    let h = first.o.shape.0[1];
    let dv = *first.o.shape.0.last().unwrap_or(&0);
    for (i, p) in partials.iter().enumerate() {
        if p.o.shape != first.o.shape || p.lse.shape != first.lse.shape {
            return Err(ferrite_types::FerriteError::InvalidArg(format!(
                "dcp merge: partial {i} shape mismatch"
            )));
        }
    }
    let mut o = vec![0.0f32; n * h * dv];
    let mut lse = vec![f32::NEG_INFINITY; n * h];
    for i in 0..n {
        for hd in 0..h {
            // stable log-sum-exp + weighted mean
            let mut m = f32::NEG_INFINITY;
            for p in partials {
                let l = p.lse.as_slice()[i * h + hd];
                if l > m {
                    m = l;
                }
            }
            if m == f32::NEG_INFINITY {
                continue; // all shards empty
            }
            let mut wsum = 0.0f32;
            let mut acc = vec![0.0f32; dv];
            for p in partials {
                let l = p.lse.as_slice()[i * h + hd];
                if l == f32::NEG_INFINITY {
                    continue; // empty shard: weight 0
                }
                let w = (l - m).exp();
                wsum += w;
                let oh = &p.o.as_slice()[i * h * dv + hd * dv..i * h * dv + (hd + 1) * dv];
                for (j, vv) in oh.iter().enumerate() {
                    acc[j] += w * vv;
                }
            }
            let inv = 1.0 / wsum;
            for j in 0..dv {
                o[i * h * dv + hd * dv + j] = acc[j] * inv;
            }
            lse[i * h + hd] = m + wsum.ln();
        }
    }
    Ok((
        Tensor::from_f32(Shape::new([n, h, dv]), o),
        Tensor::from_f32(Shape::new([n, h]), lse),
    ))
}

/// Convenience: split a full KV set into N shards by page round-robin
/// (`p mod n == r`, matching `ReshardPlan::page_mask`'s destination layout)
/// — the test/simulation-side counterpart of the DCP distribution.
pub fn split_pages_round_robin(
    k: &Tensor,
    v: &Tensor,
    n_shards: usize,
) -> Result<Vec<(Tensor, Tensor)>> {
    let t = k.shape.0[0];
    let mut out = Vec::with_capacity(n_shards);
    for r in 0..n_shards {
        let idx: Vec<usize> = (0..t).filter(|&p| p % n_shards == r).collect();
        let h = k.shape.0[1];
        let dk = *k.shape.0.last().unwrap_or(&0);
        let dv = *v.shape.0.last().unwrap_or(&0);
        let mut ks = Vec::with_capacity(idx.len() * h * dk);
        let mut vs = Vec::with_capacity(idx.len() * h * dv);
        for &p in &idx {
            ks.extend_from_slice(&k.as_slice()[p * h * dk..(p + 1) * h * dk]);
            vs.extend_from_slice(&v.as_slice()[p * h * dv..(p + 1) * h * dv]);
        }
        out.push((
            Tensor::from_f32(Shape::new([idx.len(), h, dk]), ks),
            Tensor::from_f32(Shape::new([idx.len(), h, dv]), vs),
        ));
    }
    Ok(out)
}

/// Split attention tensors along the **head axis** (TP shard):
/// `q/k [n, h, d]` (or `v [n, h, dv]`) → rank `r`'s head subset
/// `[n, h/w, d]`. Heads are an output dim — never merged by sum, only
/// concatenated back (see [`concat_heads`]).
pub fn split_heads(x: &Tensor, tp_rank: usize, tp_world: usize) -> Tensor {
    let n = x.shape.0[0];
    let h = x.shape.0[1];
    let d = *x.shape.0.last().unwrap_or(&0);
    assert!(h % tp_world == 0, "heads {h} not divisible by tp_world {tp_world}");
    let per = h / tp_world;
    let mut out = Vec::with_capacity(n * per * d);
    for i in 0..n {
        let start = (i * h + tp_rank * per) * d;
        out.extend_from_slice(&x.as_slice()[start..start + per * d]);
    }
    Tensor::from_f32(Shape::new([n, per, d]), out)
}

/// Split q into contiguous **token segments** (prefill CP / Q axis):
/// `q [n, h, d]` → segment `s`'s row block `[n_seg, h, d]` (div+remainder:
/// earlier segments take the extra rows).
pub fn split_q_segments(q: &Tensor, n_seg: usize, seg: usize) -> Tensor {
    let n = q.shape.0[0];
    let h = q.shape.0[1];
    let d = *q.shape.0.last().unwrap_or(&0);
    let base = n / n_seg;
    let extra = n % n_seg;
    let start = seg * base + seg.min(extra);
    let end = (seg + 1) * base + (seg + 1).min(extra);
    Tensor::from_f32(
        Shape::new([end - start, h, d]),
        q.as_slice()[start * h * d..end * h * d].to_vec(),
    )
}

/// Concatenate head-subset attention outputs back along the head axis (the
/// TP merge for attention outputs: heads are an output dim — concat, not
/// sum; the o_proj partial-sum happens after this, along the same axis).
/// Handles both `[n, h, d]` (attention `o`) and `[n, h]` (`lse`) ranks.
pub fn concat_heads(parts: &[Tensor]) -> Tensor {
    assert!(!parts.is_empty());
    let rank3 = parts[0].shape.0.len() == 3;
    let n = parts[0].shape.0[0];
    let h_total: usize = parts.iter().map(|p| p.shape.0[1]).sum();
    let last = if rank3 { *parts[0].shape.0.last().unwrap() } else { 1 };
    // row i of the output = row i of each part stitched along dim 1
    let mut data = Vec::with_capacity(parts.iter().map(|p| p.numel()).sum());
    for i in 0..n {
        for p in parts {
            let h = p.shape.0[1];
            let row_len = if rank3 { h * last } else { h };
            data.extend_from_slice(&p.as_slice()[i * row_len..(i + 1) * row_len]);
        }
    }
    let shape = if rank3 {
        Shape::new([n, h_total, last])
    } else {
        Shape::new([n, h_total])
    };
    Tensor::from_f32(shape, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(len: usize, seed: u64) -> Vec<f32> {
        let mut x = seed | 1;
        (0..len)
            .map(|_| {
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                (x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    fn full_attn(q: &Tensor, k: &Tensor, v: &Tensor) -> Vec<f32> {
        // reference: full softmax attention over all t tokens
        let n = q.shape.0[0];
        let h = q.shape.0[1];
        let dq = *q.shape.0.last().unwrap_or(&0);
        let t = k.shape.0[0];
        let dv = *v.shape.0.last().unwrap_or(&0);
        let scale = 1.0 / (dq as f32).sqrt();
        let mut out = vec![0.0f32; n * h * dv];
        for i in 0..n {
            for hd in 0..h {
                let qh = &q.as_slice()[i * h * dq + hd * dq..i * h * dq + (hd + 1) * dq];
                let mut scores = Vec::with_capacity(t);
                for p in 0..t {
                    let kh = &k.as_slice()[p * h * dq + hd * dq..p * h * dq + (hd + 1) * dq];
                    scores.push((0..dq).map(|l| qh[l] * kh[l]).sum::<f32>() * scale);
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let denom: f32 = scores.iter().map(|s| (s - m).exp()).sum();
                for j in 0..dv {
                    let mut acc = 0.0f32;
                    for (p, s) in scores.iter().enumerate() {
                        let vh = &v.as_slice()[p * h * dv + hd * dv..p * h * dv + (hd + 1) * dv];
                        acc += (s - m).exp() / denom * vh[j];
                    }
                    out[i * h * dv + hd * dv + j] = acc;
                }
            }
        }
        out
    }

    /// THE equivalence test: N-way page-sharded partials + LSE merge ==
    /// full attention over all tokens (the DCP correctness invariant).
    #[test]
    fn dcp_merge_equals_full_attention() {
        let (n, h, d, dv, t) = (3usize, 2usize, 4usize, 3usize, 17usize);
        let q = Tensor::from_f32(Shape::new([n, h, d]), rng(n * h * d, 1));
        let k = Tensor::from_f32(Shape::new([t, h, d]), rng(t * h * d, 2));
        let v = Tensor::from_f32(Shape::new([t, h, dv]), rng(t * h * dv, 3));

        // full reference
        let expect = full_attn(&q, &k, &v);

        // N-way split + partials + merge
        for n_shards in [2usize, 3, 4] {
            let shards = split_pages_round_robin(&k, &v, n_shards).unwrap();
            let mut partials = Vec::new();
            for (ks, vs) in &shards {
                partials.push(sparse_attn_partial(&q, ks, vs).unwrap());
            }
            let (o, _lse) = lse_merge(&partials).unwrap();
            for (got, want) in o.as_slice().iter().zip(expect.iter()) {
                assert!(
                    (got - want).abs() < 1e-4,
                    "n_shards={n_shards}: merged {got} vs full {want}"
                );
            }
        }
    }

    /// LSE stability under extreme scores (±1000): no NaN/inf.
    #[test]
    fn lse_merge_extreme_scores() {
        let (n, h, d, dv) = (1usize, 1usize, 2usize, 2usize);
        let q = Tensor::from_f32(Shape::new([n, h, d]), vec![1.0, 0.0]);
        // shard A: score +1000 (extreme positive), shard B: score -1000
        let ka = Tensor::from_f32(Shape::new([1, h, d]), vec![1000.0, 0.0]);
        let kb = Tensor::from_f32(Shape::new([1, h, d]), vec![-1000.0, 0.0]);
        let va = Tensor::from_f32(Shape::new([1, h, dv]), vec![1.0, 1.0]);
        let vb = Tensor::from_f32(Shape::new([1, h, dv]), vec![2.0, 2.0]);
        let pa = sparse_attn_partial(&q, &ka, &va).unwrap();
        let pb = sparse_attn_partial(&q, &kb, &vb).unwrap();
        let (o, lse) = lse_merge(&[pa, pb]).unwrap();
        // softmax([1000, -1000]) ≈ [1, 0] — output ≈ v_a (both dv slots)
        assert!(o.as_slice().iter().all(|x| x.is_finite()));
        assert!(lse.as_slice().iter().all(|x| x.is_finite()));
        assert!((o.as_slice()[0] - 1.0).abs() < 1e-3, "extreme: A dominates");
        assert!((o.as_slice()[1] - 1.0).abs() < 1e-3, "shard A's v wins (v_a=[1,1])");
    }

    /// Empty shard (t=0): LSE = -inf, contributes weight 0 to the merge.
    #[test]
    fn empty_shard_contributes_nothing() {
        let (n, h, d, dv) = (1usize, 1usize, 2usize, 2usize);
        let q = Tensor::from_f32(Shape::new([n, h, d]), vec![0.5, 0.5]);
        let k = Tensor::from_f32(Shape::new([2, h, d]), vec![1.0, 0.0, 0.0, 1.0]);
        let v = Tensor::from_f32(Shape::new([2, h, dv]), vec![3.0, 4.0, 5.0, 6.0]);
        let pa = sparse_attn_partial(&q, &k, &v).unwrap();
        // empty shard
        let ke = Tensor::from_f32(Shape::new([0, h, d]), vec![]);
        let ve = Tensor::from_f32(Shape::new([0, h, dv]), vec![]);
        let pb = sparse_attn_partial(&q, &ke, &ve).unwrap();
        let (o, _l) = lse_merge(&[pa.clone(), pb]).unwrap();
        let (o2, _l2) = lse_merge(&[pa]).unwrap();
        for (a, b) in o.as_slice().iter().zip(o2.as_slice().iter()) {
            assert!((a - b).abs() < 1e-6, "empty shard does not perturb the merge");
        }
    }

    /// Merge order independence (commutative): any permutation of partials
    /// gives the same result — the determinism property that makes EAGLE
    /// verification rank-consistent.
    #[test]
    fn merge_is_permutation_invariant() {
        let (n, h, d, dv) = (2usize, 2usize, 3usize, 2usize);
        let q = Tensor::from_f32(Shape::new([n, h, d]), rng(n * h * d, 7));
        let k = Tensor::from_f32(Shape::new([9, h, d]), rng(9 * h * d, 8));
        let v = Tensor::from_f32(Shape::new([9, h, dv]), rng(9 * h * dv, 9));
        let shards = split_pages_round_robin(&k, &v, 3).unwrap();
        let p0 = sparse_attn_partial(&q, &shards[0].0, &shards[0].1).unwrap();
        let p1 = sparse_attn_partial(&q, &shards[1].0, &shards[1].1).unwrap();
        let p2 = sparse_attn_partial(&q, &shards[2].0, &shards[2].1).unwrap();
        let (a, _) = lse_merge(&[p0.clone(), p1.clone(), p2.clone()]).unwrap();
        let (b, _) = lse_merge(&[p2.clone(), p0.clone(), p1.clone()]).unwrap();
        let (c, _) = lse_merge(&[p1, p2, p0]).unwrap();
        for i in 0..a.as_slice().len() {
            let (x, y, z) = (a.as_slice()[i], b.as_slice()[i], c.as_slice()[i]);
            assert!((x - y).abs() < 1e-5 && (y - z).abs() < 1e-5, "permutation invariance");
        }
    }

    // ------------------------------------------------------------------
    // 3D composability: (q_seg × kv_page × head) partial grid ==
    // full attention. This is THE expressibility proof for the
    // prefill-CP(Q) × DCP(KV) × TP(head) mesh: every rank holds one
    // (segment, page, head-subset) block; KV merges by LSE (per q-seg,
    // per head-subset), heads concatenate, segments are row-gathered.
    // ------------------------------------------------------------------

    /// Assemble the full attention output from a (q_seg × kv_page × head)
    /// partial grid, merging KV first (per segment & head rank), then
    /// concatenating heads, then concatenating segment rows.
    /// `partial[seg][tp][page]` — rank (seg, tp, page) of the mesh.
    fn merge_3d_kv_first(partials: &[Vec<Vec<PartialAttn>>], n_seg: usize, n_tp: usize) -> Tensor {
        let mut head_concat_per_seg: Vec<Tensor> = Vec::with_capacity(n_seg);
        for seg in 0..n_seg {
            let mut heads: Vec<Tensor> = Vec::with_capacity(n_tp);
            for tp in 0..n_tp {
                let (o, _) = lse_merge(&partials[seg][tp]).unwrap();
                heads.push(o);
            }
            head_concat_per_seg.push(concat_heads(&heads));
        }
        // row-gather along the q axis: segments have disjoint rows
        let n = head_concat_per_seg.iter().map(|t| t.shape.0[0]).sum::<usize>();
        let h = head_concat_per_seg[0].shape.0[1];
        let dv = *head_concat_per_seg[0].shape.0.last().unwrap_or(&0);
        let mut data = Vec::with_capacity(n * h * dv);
        for t in &head_concat_per_seg {
            data.extend_from_slice(t.as_slice());
        }
        Tensor::from_f32(Shape::new([n, h, dv]), data)
    }

    /// The reverse merge order: concatenate heads first (per segment &
    /// page), then LSE-merge along KV with full heads. Must equal the
    /// KV-first order — the axis merges commute (LSE is per-head;
    /// concat is linear).
    fn merge_3d_head_first(partials: &[Vec<Vec<PartialAttn>>], n_seg: usize, n_tp: usize) -> Tensor {
        let mut per_seg: Vec<Tensor> = Vec::with_capacity(n_seg);
        for seg in 0..n_seg {
            let n_pages = partials[seg][0].len();
            let mut page_full: Vec<PartialAttn> = Vec::with_capacity(n_pages);
            for page in 0..n_pages {
                let heads: Vec<Tensor> =
                    (0..n_tp).map(|tp| partials[seg][tp][page].o.clone()).collect();
                let lse_heads: Vec<Tensor> =
                    (0..n_tp).map(|tp| partials[seg][tp][page].lse.clone()).collect();
                page_full.push(PartialAttn {
                    o: concat_heads(&heads),
                    lse: concat_heads(&lse_heads),
                });
            }
            let (o, _) = lse_merge(&page_full).unwrap();
            per_seg.push(o);
        }
        let n = per_seg.iter().map(|t| t.shape.0[0]).sum::<usize>();
        let h = per_seg[0].shape.0[1];
        let dv = *per_seg[0].shape.0.last().unwrap_or(&0);
        let mut data = Vec::with_capacity(n * h * dv);
        for t in &per_seg {
            data.extend_from_slice(t.as_slice());
        }
        Tensor::from_f32(Shape::new([n, h, dv]), data)
    }

    /// (q_seg × kv_page × head) grid == full attention, both merge orders.
    #[test]
    fn q_kv_head_3d_merge_equals_full() {
        let (n, h, d, dv, t) = (6usize, 4usize, 8usize, 5usize, 17usize);
        let q = Tensor::from_f32(Shape::new([n, h, d]), rng(n * h * d, 11));
        let k = Tensor::from_f32(Shape::new([t, h, d]), rng(t * h * d, 12));
        let v = Tensor::from_f32(Shape::new([t, h, dv]), rng(t * h * dv, 13));
        let expect = full_attn(&q, &k, &v);

        let (n_seg, n_page, n_tp) = (2usize, 3usize, 2usize);
        // grid[seg][tp][page]: each rank's (Q rows × head subset × KV pages)
        let mut grid: Vec<Vec<Vec<PartialAttn>>> = vec![vec![Vec::new(); n_tp]; n_seg];
        let pages = split_pages_round_robin(&k, &v, n_page).unwrap();
        for seg in 0..n_seg {
            let q_seg = split_q_segments(&q, n_seg, seg);
            for tp in 0..n_tp {
                let q_h = split_heads(&q_seg, tp, n_tp);
                for (kp, vp) in &pages {
                    let kh = split_heads(kp, tp, n_tp);
                    let vh = split_heads(vp, tp, n_tp);
                    grid[seg][tp].push(sparse_attn_partial(&q_h, &kh, &vh).unwrap());
                }
            }
        }
        // order A: KV LSE first (per seg, per head rank) → head concat → rows
        let got_a = merge_3d_kv_first(&grid, n_seg, n_tp);
        // order B: head concat first (per seg, per page) → KV LSE with full heads → rows
        let got_b = merge_3d_head_first(&grid, n_seg, n_tp);
        for i in 0..expect.len() {
            let (a, b) = (got_a.as_slice()[i], got_b.as_slice()[i]);
            assert!(
                (a - expect[i]).abs() < 1e-5,
                "kv-first 3D merge diverged at {i}: {a} vs {}",
                expect[i]
            );
            assert!(
                (b - expect[i]).abs() < 1e-5,
                "head-first 3D merge diverged at {i}: {b} vs {}",
                expect[i]
            );
        }
    }

    /// The decode-shape corner of the same grid: q not segmented (single
    /// token = decode), KV pages + head shards only — the DCP×TP mesh.
    #[test]
    fn kv_head_2d_merge_equals_full() {
        let (n, h, d, dv, t) = (1usize, 4usize, 8usize, 5usize, 17usize);
        let q = Tensor::from_f32(Shape::new([n, h, d]), rng(n * h * d, 21));
        let k = Tensor::from_f32(Shape::new([t, h, d]), rng(t * h * d, 22));
        let v = Tensor::from_f32(Shape::new([t, h, dv]), rng(t * h * dv, 23));
        let expect = full_attn(&q, &k, &v);

        let (n_page, n_tp) = (3usize, 2usize);
        let pages = split_pages_round_robin(&k, &v, n_page).unwrap();
        let mut heads: Vec<Tensor> = Vec::with_capacity(n_tp);
        for tp in 0..n_tp {
            let q_h = split_heads(&q, tp, n_tp);
            let mut partials = Vec::new();
            for (kp, vp) in &pages {
                partials.push(sparse_attn_partial(
                    &q_h,
                    &split_heads(kp, tp, n_tp),
                    &split_heads(vp, tp, n_tp),
                ).unwrap());
            }
            let (o, _) = lse_merge(&partials).unwrap();
            heads.push(o);
        }
        let got = concat_heads(&heads);
        for i in 0..expect.len() {
            assert!(
                (got.as_slice()[i] - expect[i]).abs() < 1e-5,
                "kv×head 2D merge diverged at {i}"
            );
        }
    }
}
