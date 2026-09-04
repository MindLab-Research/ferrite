//! MHC (hyper-connections) — exact port of sglang's `_mhc_pre_torch` /
//! `_mhc_post_torch` (python/sglang/kernels/ops/layernorm/mhc.py), calibrated
//! against GLM-5.3-Flash.
//!
//! Semantics: the residual stream is n = hc_mult parallel flows
//! `[tokens, n, hidden]`. Each sublayer (attention / FFN) is bracketed by:
//!
//! - `hc_pre`: mixes = fn(residual)·rims; pre = sigmoid(mixes) + eps (input
//!   mixing weights); layer_input = Σᵢ preᵢ · residualᵢ
//! - `hc_post`: residual′ᵢ = postᵢ · x + Σₖ combᵢₖ · residualₖ
//!   (post = 2·sigmoid, comb = sinkhorn-normalised mixing matrix)
//!
//! First layer: `hc_expand` replicates x into all n flows. After the last
//! layer: `hc_contract` averages the flows back to one hidden.

use ferrite_types::{Shape, Tensor};

/// `[s, hidden] -> [s, n*hidden]` by replication (first layer init).
pub fn hc_expand(x: &Tensor, n: usize) -> Tensor {
    let s = x.shape.0[0];
    let h = x.shape.0[1];
    let mut data = Vec::with_capacity(s * n * h);
    for t in 0..s {
        for _rep in 0..n {
            data.extend_from_slice(&x.as_slice()[t * h..(t + 1) * h]);
        }
    }
    Tensor::from_f32(Shape::new([s, n * h]), data)
}

/// `[s, n, hidden] -> [s, hidden]` by averaging (after the last layer).
pub fn hc_contract(x: &Tensor, n: usize) -> Tensor {
    let s = x.shape.0[0];
    let h = x.shape.0[1] / n;
    let mut data = vec![0.0f32; s * h];
    for t in 0..s {
        for rep in 0..n {
            for (j, v) in x.as_slice()[(t * n + rep) * h..(t * n + rep + 1) * h]
                .iter()
                .enumerate()
            {
                data[t * h + j] += v;
            }
        }
    }
    for v in data.iter_mut() {
        *v /= n as f32;
    }
    Tensor::from_f32(Shape::new([s, h]), data)
}

/// One hc_pre step (sglang `_mhc_pre_torch` exact port).
///
/// `residual_flat: [s, n*h]`, `fn_w: [mix_hc, n*h]`, `scale: [3]`,
/// `base: [mix_hc]` where `mix_hc = 2n + n*n`.
///
/// Returns `(layer_input [s,h], post [s,n], comb [s,n,n])`.
pub fn hc_pre(
    residual_flat: &Tensor,
    fn_w: &Tensor,
    scale: &Tensor,
    base: &Tensor,
    rms_eps: f32,
    hc_eps: f32,
    sinkhorn_iters: usize,
) -> (Tensor, Tensor, Tensor) {
    let s = residual_flat.shape.0[0];
    let nh = residual_flat.shape.0[1]; // n * hidden
    let n = fn_w.shape.0[1] / nh * 0; // unused; n derived below
    let _ = n;
    // n from the comb block: (mix_hc - 2n) = n^2 -> solve n
    let mix_hc = fn_w.shape.0[0];
    let nn = nh_width(nh, mix_hc);
    let (slen, h) = (residual_flat.shape.0[0], nh / nn);
    let n = nn;
    let _ = s;

    // mixes = linear(x_flat, fn) * rsqrt;  x_flat row = [n*h]
    // rsqrt = 1/sqrt(mean(x^2) + rms_eps) per token
    let rf = residual_flat.as_slice();
    let fw = fn_w.as_slice();
    let sv = scale.as_slice();
    let bv = base.as_slice();

    let mut mixes = vec![0.0f32; slen * mix_hc];
    for t in 0..slen {
        let mut msq = 0.0f32;
        for &v in &rf[t * nh..(t + 1) * nh] {
            msq += v * v;
        }
        let rsqrt = 1.0 / ((msq / nh as f32 + rms_eps).sqrt());
        for m in 0..mix_hc {
            let mut acc = 0.0f32;
            let row = &fw[m * nh..(m + 1) * nh];
            for i in 0..nh {
                acc += row[i] * rf[t * nh + i];
            }
            mixes[t * mix_hc + m] = acc * rsqrt;
        }
    }

    // pre = sigmoid(mixes[:, :n]*scale[0] + base[:n]) + eps
    // post = 2*sigmoid(mixes[:, n:2n]*scale[1] + base[n:2n])
    // comb = mixes[:, 2n:]*scale[2] + base[2n:] (reshaped [n,n])
    let mut layer_input = vec![0.0f32; slen * h];
    let mut post = vec![0.0f32; slen * n];
    let mut comb = vec![0.0f32; slen * n * n];
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    for t in 0..slen {
        // layer_input = Σᵢ preᵢ · residualᵢ
        for i in 0..n {
            let pre_i = sigmoid(mixes[t * mix_hc + i] * sv[0] + bv[i]) + hc_eps;
            for j in 0..h {
                layer_input[t * h + j] += pre_i * rf[t * nh + i * h + j];
            }
        }
        // post
        for i in 0..n {
            post[t * n + i] =
                2.0 * sigmoid(mixes[t * mix_hc + n + i] * sv[1] + bv[n + i]);
        }
        // comb (+ scale, base)
        for i in 0..n {
            for k in 0..n {
                comb[t * n * n + i * n + k] =
                    mixes[t * mix_hc + 2 * n + i * n + k] * sv[2] + bv[2 * n + i * n + k];
            }
        }
        // sinkhorn: softmax(-1) + eps; /= (sum(-2) + eps); then iters-1 × { /= (sum(-1)+eps); /= (sum(-2)+eps) }
        // softmax over last dim (k) per row i
        for i in 0..n {
            let mut rowmax = f32::MIN;
            for k in 0..n {
                rowmax = rowmax.max(comb[t * n * n + i * n + k]);
            }
            let mut denom = 0.0f32;
            for k in 0..n {
                comb[t * n * n + i * n + k] =
                    (comb[t * n * n + i * n + k] - rowmax).exp();
                denom += comb[t * n * n + i * n + k];
            }
            for k in 0..n {
                comb[t * n * n + i * n + k] =
                    comb[t * n * n + i * n + k] / denom + hc_eps;
            }
        }
        // /= (sum(-2) + eps): column sums over i (per k)
        for k in 0..n {
            let mut colsum = 0.0f32;
            for i in 0..n {
                colsum += comb[t * n * n + i * n + k];
            }
            let d = colsum + hc_eps;
            for i in 0..n {
                comb[t * n * n + i * n + k] /= d;
            }
        }
        for _ in 1..sinkhorn_iters {
            // /= (sum(-1) + eps): row sums over k
            for i in 0..n {
                let mut rowsum = 0.0f32;
                for k in 0..n {
                    rowsum += comb[t * n * n + i * n + k];
                }
                let d = rowsum + hc_eps;
                for k in 0..n {
                    comb[t * n * n + i * n + k] /= d;
                }
            }
            // /= (sum(-2) + eps)
            for k in 0..n {
                let mut colsum = 0.0f32;
                for i in 0..n {
                    colsum += comb[t * n * n + i * n + k];
                }
                let d = colsum + hc_eps;
                for i in 0..n {
                    comb[t * n * n + i * n + k] /= d;
                }
            }
        }
    }
    (
        Tensor::from_f32(Shape::new([slen, h]), layer_input),
        Tensor::from_f32(Shape::new([slen, n]), post),
        Tensor::from_f32(Shape::new([slen, n, n]), comb),
    )
}

/// One hc_post step (transformers `_mhc_post_torch` port):
/// residual′[t,i,j] = post[t,i] · x[t,j] + Σₖ comb[t,**k,i] · residual[t,k,j]
/// (note the comb transpose: transformers does matmul(combᵀ, residual))
///
/// `x: [s,h]`, `residual: [s,n,h]`, `post: [s,n]`, `comb: [s,n,n]`.
pub fn hc_post(x: &Tensor, residual: &Tensor, post: &Tensor, comb: &Tensor) -> Tensor {
    let s = x.shape.0[0];
    let h = x.shape.0[1];
    let n = residual.shape.0[1];
    let mut data = vec![0.0f32; s * n * h];
    for t in 0..s {
        for i in 0..n {
            for j in 0..h {
                let mut acc = post.as_slice()[t * n + i] * x.as_slice()[t * h + j];
                for k in 0..n {
                    acc += comb.as_slice()[(t * n + k) * n + i]
                        * residual.as_slice()[(t * n + k) * h + j];
                }
                data[(t * n + i) * h + j] = acc;
            }
        }
    }
    Tensor::from_f32(Shape::new([s, n, h]), data)
}

/// n from mix_hc = 2n + n^2 (quadratic solve: n = (-2 + sqrt(4 + 4*mix)) / 2).
fn nh_width(_nh: usize, mix_hc: usize) -> usize {
    // mix = 2n + n^2  ->  n^2 + 2n - mix = 0  ->  n = (-2 + sqrt(4 + 4mix))/2
    let disc = (4 + 4 * mix_hc) as f64;
    let n = ((-2.0 + disc.sqrt()) / 2.0) as usize;
    debug_assert_eq!(2 * n + n * n, mix_hc);
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hc_expand_contract_roundtrip() {
        let x = Tensor::from_f32(Shape::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let expanded = hc_expand(&x, 4); // [2, 12]
        assert_eq!(expanded.shape.0, [2, 12]);
        assert_eq!(expanded.as_slice()[0..3], [1.0, 2.0, 3.0]);
        assert_eq!(expanded.as_slice()[3..6], [1.0, 2.0, 3.0]);
        let back = hc_contract(&expanded, 4);
        assert_eq!(back.as_slice(), x.as_slice());
    }

    #[test]
    fn hc_pre_post_shapes() {
        // n=2, h=3, mix_hc = 2*2 + 4 = 8
        let s = 2;
        let residual = Tensor::from_f32(Shape::new([s, 6]), (0..12).map(|i| i as f32 / 12.0).collect());
        let fn_w = Tensor::from_f32(Shape::new([8, 6]), vec![0.1; 48]);
        let scale = Tensor::from_f32(Shape::new([3]), vec![1.0, 1.0, 1.0]);
        let base = Tensor::from_f32(Shape::new([8]), vec![0.0; 8]);
        let (li, post, comb) = hc_pre(&residual, &fn_w, &scale, &base, 1e-5, 1e-6, 20);
        assert_eq!(li.shape.0, [s, 3]);
        assert_eq!(post.shape.0, [s, 2]);
        assert_eq!(comb.shape.0, [s, 2, 2]);
        // post in (0, 2) — sigmoid range
        for &p in post.as_slice() {
            assert!((0.0..2.0).contains(&p));
        }
        // hc_post round
        let x2 = Tensor::from_f32(Shape::new([s, 3]), vec![0.5; 6]);
        let res3 = Tensor::from_f32(Shape::new([s, 2, 3]), vec![0.25; 12]);
        let out = hc_post(&x2, &res3, &post, &comb);
        assert_eq!(out.shape.0, [s, 2, 3]);
    }

    #[test]
    fn nh_width_solve() {
        assert_eq!(nh_width(0, 24), 4); // mix 24 -> n=4 (2*4+16=24)
        assert_eq!(nh_width(0, 8), 2); // mix 8 -> n=2
    }
}
