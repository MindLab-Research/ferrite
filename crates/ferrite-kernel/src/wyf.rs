//! WYF-transform chunkwise Gated DeltaNet — the parallel-form math that the
//! CUDA kernel implements (chunk tokens in parallel instead of the
//! sequential per-token recurrence).
//!
//! This is the *reference* implementation: pure Rust, mirrors the CUDA
//! kernel's algorithm step-by-step, and is validated against the sequential
//! golden recurrence (`CpuBackend::gated_deltanet_chunk`) — the two must
//! agree to f32 tolerance for arbitrary random inputs.
//!
//! ## Derivation (per head)
//!
//! Sequential recurrence (golden, see cpu.rs):
//! ```text
//! S ← diag(D_t) · S ; kS = Sᵀk_t ; S −= β_t k_t kSᵀ ; S += β_t k_t v_tᵀ
//! ```
//! with channel-wise decay `D_t[i] = exp(gate[t,i]·a)`, `a = −exp(a_log)`.
//!
//! Unrolling with `L[t,i] = Σ_{r≤t} gate[r,i]·a` (inclusive prefix, `L[−1]=0`):
//! ```text
//! S_t = diag(e^{L_t}) S₀ + Σ_{s≤t} (k_s ⊙ e^{L_t−L_s}) w_sᵀ ,  w_s = β_s(v_s − u_s)
//! u_t = S₀ᵀ(k_t ⊙ e^{L_{t−1}}) + Σ_{s<t} c[t,s]·w_s ,
//!       c[t,s] = Σ_i k_t[i]·k_s[i]·e^{L_{t−1,i}−L_{s,i}}
//! ⇒ forward substitution (the triangular solve, O(C²) instead of O(C) serial):
//! w_t = β_t·(v_t − b_t − Σ_{s<t} c[t,s]·w_s) ,  b_t = S₀ᵀ(k_t ⊙ e^{L_{t−1}})
//! O_t = (q_t ⊙ e^{L_t})ᵀ S₀ + Σ_{s≤t} m[t,s]·w_s ,
//!       m[t,s] = Σ_i q_t[i]·k_s[i]·e^{L_{t,i}−L_{s,i}}
//! S_C = diag(e^{L_{C−1}}) S₀ + Σ_s (k_s ⊙ e^{L_{C−1}−L_s}) w_sᵀ
//! ```
//! All exponents are ≤ 0 (decay < 1) — numerically stable.
//!
//! The CUDA kernel parallelises: `c`/`m` are dense GEMMs per chunk, the
//! triangular solve is a small [C,C] system per head, and O/S_C rebuilds are
//! reductions — chunk-internal parallelism without the token loop.

/// Layouts (identical to the trait contract):
/// q,k: [n,h,dk]; v: [n,h,dv]; beta: [n,h]; gate: [n,h,dk]; a_log: [h];
/// s0, state_out: [h,dk,dv]; out: [n,h,dv].
pub fn wyf_chunk_gdn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    gate: &[f32],
    a_log: &[f32],
    s0: &[f32],
    out: &mut [f32],
    state_out: &mut [f32],
    n: usize,
    h: usize,
    dk: usize,
    dv: usize,
) {
    for hd in 0..h {
        let a = -a_log[hd].exp();
        // ---- 1. inclusive prefix L[t,i] = a * Σ_{r≤t} gate[r,hd,i] ----
        let mut l = vec![0f32; n * dk];
        for i in 0..dk {
            let mut acc = 0f32;
            for t in 0..n {
                acc += gate[(t * h + hd) * dk + i] * a;
                l[t * dk + i] = acc;
            }
        }
        let head = hd * dk * dv;
        let s = &s0[head..head + dk * dv];

        // per-token k/v/q/beta rows
        let kt = |t: usize| &k[(t * h + hd) * dk..(t * h + hd) * dk + dk];
        let vt = |t: usize| &v[(t * h + hd) * dv..(t * h + hd) * dv + dv];
        let qt = |t: usize| &q[(t * h + hd) * dk..(t * h + hd) * dk + dk];
        let bt = |t: usize| beta[t * h + hd];

        // ---- 2. b_t = S₀ᵀ (k_t ⊙ e^{L[t]}) — chunk-interaction term ----
        // NOTE: the golden recurrence decays BEFORE kS, so the state term
        // carries L[t] (inclusive of token t's decay), not L[t-1].
        let mut b = vec![vec![0f32; dv]; n];
        for t in 0..n {
            for j in 0..dv {
                let mut acc = 0f32;
                for i in 0..dk {
                    acc += s[i * dv + j] * kt(t)[i] * l[t * dk + i].exp();
                }
                b[t][j] = acc;
            }
        }

        // ---- 3. c[t,s] = Σ_i k_t·k_s·e^{L[t,i]-L[s,i]} (s < t) ----
        let mut c = vec![vec![0f32; n]; n]; // c[t][s], lower-triangular
        for t in 1..n {
            for s in 0..t {
                let mut acc = 0f32;
                for i in 0..dk {
                    acc += kt(t)[i] * kt(s)[i] * (l[t * dk + i] - l[s * dk + i]).exp();
                }
                c[t][s] = acc;
            }
        }

        // ---- 4. forward substitution: w_t = β_t(v_t − b_t − Σ_{s<t} c[t,s]·w_s) ----
        let mut w = vec![vec![0f32; dv]; n];
        for t in 0..n {
            for j in 0..dv {
                let mut acc = vt(t)[j] - b[t][j];
                for s in 0..t {
                    acc -= c[t][s] * w[s][j];
                }
                w[t][j] = bt(t) * acc;
            }
        }

        // ---- 5. O_t = (q_t ⊙ e^{L_t})ᵀ S₀ + Σ_{s≤t} m[t,s]·w_s ----
        for t in 0..n {
            let ohead = &mut out[(t * h + hd) * dv..(t * h + hd) * dv + dv];
            for j in 0..dv {
                let mut acc = 0f32;
                // state term
                for i in 0..dk {
                    acc += qt(t)[i] * l[t * dk + i].exp() * s[i * dv + j];
                }
                // write terms
                for s in 0..=t {
                    let mut m_ts = 0f32;
                    for i in 0..dk {
                        m_ts += qt(t)[i] * kt(s)[i] * (l[t * dk + i] - l[s * dk + i]).exp();
                    }
                    acc += m_ts * w[s][j];
                }
                ohead[j] = acc;
            }
        }

        // ---- 6. S_C = diag(e^{L[C-1]}) S₀ + Σ_s (k_s ⊙ e^{L[C-1]-L_s}) w_sᵀ ----
        let last = &l[(n - 1) * dk..(n - 1) * dk + dk];
        let so = &mut state_out[head..head + dk * dv];
        for i in 0..dk {
            let decay_all = last[i].exp();
            for j in 0..dv {
                let mut acc = decay_all * s[i * dv + j];
                for s_idx in 0..n {
                    let wfac = kt(s_idx)[i] * (last[i] - l[s_idx * dk + i]).exp();
                    acc += wfac * w[s_idx][j];
                }
                so[i * dv + j] = acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrite_types::{DType, Shape, Tensor};
    use crate::CpuBackend;
    use crate::KernelBackend;

    fn rng_vals(len: usize, seed: u64) -> Vec<f32> {
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

    /// WYF parallel form must match the sequential golden recurrence
    /// bit-for-bit-ish (f32 tolerance) on random inputs.
    #[test]
    fn wyf_matches_sequential_golden() {
        let (n, h, dk, dv) = (16usize, 3usize, 8usize, 6usize);
        let mk3 = |d: Vec<f32>| Tensor::new(Shape::new([n, h, d.len() / (n * h)]), DType::F32, d);
        let q = mk3(rng_vals(n * h * dk, 1));
        let k = mk3(rng_vals(n * h * dk, 2));
        let v = mk3(rng_vals(n * h * dv, 3));
        let beta = Tensor::new(
            Shape::new([n, h]),
            DType::F32,
            rng_vals(n * h, 4).iter().map(|x| 0.5 + 0.5 * x.abs()).collect(),
        );
        let gate = Tensor::new(
            Shape::new([n, h, dk]),
            DType::F32,
            rng_vals(n * h * dk, 5).iter().map(|x| 0.5 + 0.5 * x.abs()).collect(),
        );
        let a_log = Tensor::vec(vec![-0.1, -0.2, -0.05]);
        let mut s0 = rng_vals(h * dk * dv, 6);
        // start from a non-trivial state (prefix carried in)
        for (i, x) in s0.iter_mut().enumerate() {
            *x *= 0.3;
            if i % 3 == 0 {
                *x = -*x;
            }
        }
        let state_in = Tensor::from_f32(Shape::new([h, dk, dv]), s0.clone());

        // golden: sequential recurrence via the CPU backend
        let backend = CpuBackend::new();
        let mut out_gold = Tensor::zeros(Shape::new([n, h, dv]), DType::F32);
        let mut st_gold = Tensor::zeros(Shape::new([h, dk, dv]), DType::F32);
        backend
            .gated_deltanet_chunk(
                &q, &k, &v, &beta, &gate, &a_log, &state_in, &mut out_gold, &mut st_gold,
            )
            .unwrap();

        // WYF parallel form
        let mut out_wyf = vec![0f32; n * h * dv];
        let mut st_wyf = vec![0f32; h * dk * dv];
        wyf_chunk_gdn(
            q.as_slice(),
            k.as_slice(),
            v.as_slice(),
            beta.as_slice(),
            gate.as_slice(),
            a_log.as_slice(),
            &s0,
            &mut out_wyf,
            &mut st_wyf,
            n, h, dk, dv,
        );

        // diff
        let mut max_o = 0f32;
        for (g, w) in out_gold.as_slice().iter().zip(out_wyf.iter()) {
            max_o = max_o.max((g - w).abs());
        }
        let mut max_s = 0f32;
        for (g, w) in st_gold.as_slice().iter().zip(st_wyf.iter()) {
            max_s = max_s.max((g - w).abs());
        }
        // WYF reorders the arithmetic (sums in different order) — allow f32
        // accumulation noise, bounded well below 1e-3 for these magnitudes.
        assert!(max_o < 1e-3, "output mismatch: max diff {max_o}");
        assert!(max_s < 1e-3, "state mismatch: max diff {max_s}");
    }

    /// Single-token chunk: WYF degenerates to the sequential step exactly.
    #[test]
    fn wyf_single_token() {
        let (n, h, dk, dv) = (1usize, 1usize, 4usize, 3usize);
        let s0 = rng_vals(h * dk * dv, 9);
        let q = rng_vals(n * h * dk, 10);
        let k = rng_vals(n * h * dk, 11);
        let v = rng_vals(n * h * dv, 12);
        let beta = vec![0.7f32];
        let gate = rng_vals(n * h * dk, 13).iter().map(|x| 0.5 + 0.5 * x.abs()).collect::<Vec<_>>();
        let a_log = vec![-0.3f32];
        let mut out = vec![0f32; n * h * dv];
        let mut st = vec![0f32; h * dk * dv];
        wyf_chunk_gdn(&q, &k, &v, &beta, &gate, &a_log, &s0, &mut out, &mut st, n, h, dk, dv);
        // manual: t=0, L[0] = g*a (inclusive of token 0's decay);
        // golden recurrence decays FIRST then kS: u_0 = S0ᵀ(k_0 ⊙ e^{L[0]})
        let a = -a_log[0].exp();
        let mut kg = [0f32; 4];
        for i in 0..4 {
            kg[i] = (gate[i] * a).exp();
        }
        // w_0 = beta*(v - u_0), u_0[j] = Σ_i S0[i,j]·kg[i]·k0[i]
        let k0 = &k[0..4];
        let v0 = &v[0..3];
        let mut u = [0f32; 3];
        for j in 0..3 {
            u[j] = (0..4).map(|i| s0[i * 3 + j] * kg[i] * k0[i]).sum::<f32>();
        }
        let w0: Vec<f32> = (0..3).map(|j| beta[0] * (v0[j] - u[j])).collect();
        // O_0 = (q ⊙ e^{L0})^T S0 + (q·k) w0
        let q0 = &q[0..4];
        let mut o_expect = [0f32; 3];
        for j in 0..3 {
            let state_term: f32 = (0..4).map(|i| q0[i] * kg[i] * s0[i * 3 + j]).sum();
            let m00: f32 = (0..4).map(|i| q0[i] * k0[i]).sum();
            o_expect[j] = state_term + m00 * w0[j];
        }
        for j in 0..3 {
            assert!((out[j] - o_expect[j]).abs() < 1e-5, "{} vs {}", out[j], o_expect[j]);
        }
        // S_C = diag(e^{L0}) S0 + k ⊙ e^{L0-L0} w^T = diag(e^{L0}) S0 + k w^T
        for i in 0..4 {
            for j in 0..3 {
                let expect = kg[i] * s0[i * 3 + j] + k0[i] * w0[j];
                assert!((st[i * 3 + j] - expect).abs() < 1e-5);
            }
        }
    }
}
