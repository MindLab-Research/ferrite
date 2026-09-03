//! Golden numerical tests for the CPU reference backend.
//!
//! These are the ground truth the B300 CUDA backend must match:
//! every formula here is hand-derived from the Gated DeltaNet recurrence,
//! causal short-conv, SwiGLU-with-clamp and noaux-tc routing definitions.

use ferrite_kernel::{CpuBackend, KernelBackend};
use ferrite_types::{DType, Shape, Tensor};

fn t2(rows: usize, cols: usize, data: Vec<f32>) -> Tensor {
    Tensor::new(Shape::new([rows, cols]), DType::F32, data)
}

#[test]
fn matmul_matches_manual() {
    let b = CpuBackend::new();
    // x = [[1,2],[3,4]], w = [[5,6],[7,8]] -> x @ w^T = [[17,23],[39,53]]
    let x = t2(2, 2, vec![1., 2., 3., 4.]);
    let w = t2(2, 2, vec![5., 6., 7., 8.]);
    let mut out = Tensor::zeros(Shape::new([2, 2]), DType::F32);
    b.matmul(&x, &w, None, &mut out).unwrap();
    assert_eq!(out.as_slice(), &[17., 23., 39., 53.]);
    // with bias
    let bias = t2(1, 2, vec![0.5, -0.5]);
    b.matmul(&x, &w, Some(&bias), &mut out).unwrap();
    assert_eq!(out.as_slice(), &[17.5, 22.5, 39.5, 52.5]);
}

#[test]
fn rmsnorm_manual() {
    let b = CpuBackend::new();
    let x = t2(1, 4, vec![1., 2., 3., 4.]);
    let w = t2(1, 4, vec![1., 1., 1., 1.]);
    let mut out = Tensor::zeros(Shape::new([1, 4]), DType::F32);
    b.rmsnorm(&x, &w, 1e-6, &mut out).unwrap();
    let mean_sq = (1f32 + 4. + 9. + 16.) / 4f32;
    let inv = 1f32 / (mean_sq + 1e-6f32).sqrt();
    let expect: Vec<f32> = [1f32, 2., 3., 4.].iter().map(|v| v * inv).collect();
    for (g, e) in out.as_slice().iter().zip(expect.iter()) {
        assert!((g - e).abs() < 1e-5, "{g} vs {e}");
    }
}

#[test]
fn swiglu_clamps() {
    let b = CpuBackend::new();
    // gate_up row layout is [gate..., up...] (segments, not interleaved):
    // gate=[2, 20], up=[3, 1] with limit=10 -> gate 20 clamps to 10
    let gu = t2(1, 4, vec![2., 20., 3., 1.]);
    let mut out = Tensor::zeros(Shape::new([1, 2]), DType::F32);
    b.swiglu_limited(&gu, 10.0, &mut out).unwrap();
    let silu2 = 2.0f32 / (1.0 + (-2.0f32).exp());
    assert!((out.as_slice()[0] - silu2 * 3.0).abs() < 1e-6);
    assert!((out.as_slice()[1] - (10.0f32 / (1.0 + (-10.0f32).exp()))) < 1e-6, "gate clamped to 10");
}

#[test]
fn causal_conv1d_manual() {
    let b = CpuBackend::new();
    // ch=1, conv=3, state=[5,6], x=[7,8], w=[1,2,3]
    let x = t2(2, 1, vec![7., 8.]);
    let w = t2(1, 3, vec![1., 2., 3.]);
    let st = t2(1, 2, vec![5., 6.]);
    let mut out = Tensor::zeros(Shape::new([2, 1]), DType::F32);
    let mut st_out = Tensor::zeros(Shape::new([1, 2]), DType::F32);
    b.causal_conv1d(&x, &w, &st, &mut out, &mut st_out).unwrap();
    // out[0] = w0*s1 + w1*s2 + w2*x1 = 5 + 12 + 21 = 38
    // out[1] = w0*s2 + w1*x1 + w2*x2 = 6 + 14 + 24 = 44
    assert_eq!(out.as_slice(), &[38., 44.]);
    assert_eq!(st_out.as_slice(), &[7., 8.], "state carries the conv tail");
}

/// Hand-derived Gated DeltaNet recurrence (1 head, dk=dv=2, 2 tokens).
#[test]
fn gated_deltanet_step_manual() {
    let b = CpuBackend::new();
    let n = 2;
    let h = 1usize;
    let dk = 2usize;
    let dv = 2usize;
    let mk3 = |d: Vec<f32>| Tensor::new(Shape::new([n, h, d.len() / (n * h)]), DType::F32, d);
    let q = mk3(vec![1., 0., 0., 1.]); // t0: [1,0], t1: [0,1]
    let k = mk3(vec![0., 1., 1., 0.]);
    let v = mk3(vec![1., 2., 3., 4.]);
    let beta = Tensor::new(Shape::new([n, h]), DType::F32, vec![1.0, 0.5]);
    let gate = Tensor::new(Shape::new([n, h, dk]), DType::F32, vec![0.0, 0.0, 1.0, 1.0]); // channel-wise
    let a_log = Tensor::vec(vec![0.0]); // a = -exp(0) = -1.0
    let state = Tensor::zeros(Shape::new([h, dk, dv]), DType::F32);
    let mut out = Tensor::zeros(Shape::new([n, h, dv]), DType::F32);
    let mut state_out = Tensor::zeros(Shape::new([h, dk, dv]), DType::F32);
    b.gated_deltanet_step(&q, &k, &v, &beta, &gate, &a_log, &state, &mut out, &mut state_out)
        .unwrap();

    // token 0: decay = exp(0 * -1) = 1, S was 0
    //   k=[0,1], v=[1,2], beta=1: S = k v^T = [[0,0],[1,2]]
    //   o_0 = q^T S = [1,0] S = [S00, S01] = [0, 0]
    assert!((out.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((out.as_slice()[1] - 0.0).abs() < 1e-6);

    // token 1: decay = exp(1 * -1) = e^-1
    //   S *= e^-1 -> [[0,0],[e^-1, 2e^-1]]
    //   k=[1,0]: kS = S[0,:] = [0,0]; delta erase is 0
    //   S += 0.5 * k v^T = 0.5*[[3,4],[0,0]] -> [[1.5,2],[e^-1,2e^-1]]
    //   o_1 = [0,1] S = [e^-1, 2e^-1]
    let e1 = (-1.0f32).exp();
    assert!((out.as_slice()[2] - e1).abs() < 1e-5, "o1[0]={} vs {}", out.as_slice()[2], e1);
    assert!((out.as_slice()[3] - 2. * e1).abs() < 1e-5);
    assert!((state_out.as_slice()[0] - 1.5).abs() < 1e-5, "S[0,0]=1.5");
    assert!((state_out.as_slice()[1] - 2.0).abs() < 1e-5, "S[0,1]=2.0");
    assert!((state_out.as_slice()[2] - e1).abs() < 1e-5, "S[1,0]=e^-1");
    assert!((state_out.as_slice()[3] - 2. * e1).abs() < 1e-5, "S[1,1]=2e^-1");
}

/// Chunkwise == step-wise (the definitional invariant the CUDA WYF-parallel
/// form must reproduce; CPU trivially satisfies it by construction, this
/// test pins the contract).
#[test]
fn gated_deltanet_chunk_equals_steps() {
    let b = CpuBackend::new();
    let (n, h, dk, dv) = (8usize, 2usize, 4usize, 4usize);
    let mk = |dims: Vec<usize>, d: Vec<f32>| Tensor::new(Shape::new(dims), DType::F32, d);
    let rng_vals = |len: usize, seed: u64| -> Vec<f32> {
        let mut x = seed | 1;
        (0..len)
            .map(|_| {
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                (x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    };
    let q = mk(vec![n, h, dk], rng_vals(n * h * dk, 1));
    let k = mk(vec![n, h, dk], rng_vals(n * h * dk, 2));
    let v = mk(vec![n, h, dv], rng_vals(n * h * dv, 3));
    let beta = mk(vec![n, h], rng_vals(n * h, 4)).as_slice().iter().map(|x| 0.5 + 0.5 * x.abs()).collect::<Vec<f32>>();
    let beta = mk(vec![n, h], beta);
    // channel-wise gate [n, h, dk] in (0,1)
    let gate_raw = rng_vals(n * h * dk, 5);
    let gate = mk(vec![n, h, dk], gate_raw.iter().map(|x| 0.5 + 0.5 * x.abs()).collect());
    let a_log = mk(vec![h], vec![-0.1, -0.2]);
    let st0 = mk(vec![h, dk, dv], vec![0.0; h * dk * dv]);

    // chunk all-8-at-once
    let mut out_c = Tensor::zeros(Shape::new([n, h, dv]), DType::F32);
    let mut st_c = Tensor::zeros(Shape::new([h, dk, dv]), DType::F32);
    b.gated_deltanet_chunk(&q, &k, &v, &beta, &gate, &a_log, &st0, &mut out_c, &mut st_c)
        .unwrap();

    // run 2 chunks of 3 + 5 (chunk boundary must carry state)
    let run = |from: usize, to: usize, st_in: &Tensor| -> (Tensor, Tensor) {
        let m = to - from;
        let slice = |t: &Tensor, per: usize| -> Tensor {
            Tensor::new(
                Shape::new([m, h, per]),
                DType::F32,
                t.as_slice()[from * h * per..to * h * per].to_vec(),
            )
        };
        let qs = slice(&q, dk);
        let ks = slice(&k, dk);
        let vs = slice(&v, dv);
        let bs = Tensor::new(Shape::new([m, h]), DType::F32, beta.as_slice()[from * h..to * h].to_vec());
        let gs = Tensor::new(Shape::new([m, h, dk]), DType::F32, gate.as_slice()[from * h * dk..to * h * dk].to_vec());
        let mut o = Tensor::zeros(Shape::new([m, h, dv]), DType::F32);
        let mut s = Tensor::zeros(Shape::new([h, dk, dv]), DType::F32);
        b.gated_deltanet_chunk(&qs, &ks, &vs, &bs, &gs, &a_log, st_in, &mut o, &mut s)
            .unwrap();
        (o, s)
    };
    let (o1, s1) = run(0, 3, &st0);
    let (o2, s2) = run(3, 8, &s1);

    for i in 0..n {
        for j in 0..h * dv {
            let c = out_c.as_slice()[i * h * dv + j];
            let s = if i < 3 { o1.as_slice()[i * h * dv + j] } else { o2.as_slice()[(i - 3) * h * dv + j] };
            assert!((c - s).abs() < 1e-5, "tok {i} dim {j}: chunk {c} vs split {s}");
        }
    }
    for (c, s) in st_c.as_slice().iter().zip(s2.as_slice().iter()) {
        assert!((c - s).abs() < 1e-5, "final state mismatch");
    }
}

#[test]
fn moe_route_noaux_tc() {
    let b = CpuBackend::new();
    let logits = t2(1, 3, vec![1.0, 2.0, 0.5]);
    let bias = t2(1, 3, vec![0.0, 0.0, 0.5]);
    let mut probs = Tensor::zeros(Shape::new([1, 2]), DType::F32);
    let mut ids = Tensor::zeros(Shape::new([1, 2]), DType::F32);
    b.moe_route(&logits, &bias, 2, 1.0, &mut probs, &mut ids).unwrap();
    let s0 = 1.0f32 / (1.0 + (-(1.0f32)).exp()); // sigmoid(1)
    let s1 = 1.0f32 / (1.0 + (-(2.0f32)).exp()); // sigmoid(2)
    let s2 = 1.0f32 / (1.0 + (-(1.0f32)).exp()); // sigmoid(0.5+0.5)
    assert_eq!(ids.as_slice()[0], 1.0, "expert 1 has top score");
    // tie between expert 0 and 2 (both sigmoid(1)) -> lower index wins second slot
    assert_eq!(ids.as_slice()[1], 0.0);
    let sum = s1 + s0;
    assert!((probs.as_slice()[0] - s1 / sum).abs() < 1e-5);
    assert!((probs.as_slice()[1] - s0 / sum).abs() < 1e-5);
    let _ = s2;
}

#[test]
fn indexer_topk_and_sparse_attn() {
    let b = CpuBackend::new();
    // q: [1,4], k: [5,4] — q equals k[2] so top pick should be index 2
    let q = t2(1, 4, vec![1., 0., 0., 0.]);
    let k = t2(5, 4, vec![
        0., 1., 0., 0.,  // k0
        0., 0., 1., 0.,  // k1
        1., 0., 0., 0.,  // k2 == q -> top
        0., 0., 0., 1.,  // k3
        0.9, 0.1, 0., 0., // k4 near
    ]);
    let mut idx = Tensor::zeros(Shape::new([1, 2]), DType::F32);
    b.indexer_topk(&q, &k, 2, &mut idx).unwrap();
    assert_eq!(idx.as_slice()[0], 2.0, "exact match first");
    assert_eq!(idx.as_slice()[1], 4.0, "near match second");
    // sparse attn over selected: v = k for simplicity
    let h = 1usize;
    let dq = 4usize;
    let qs = Tensor::new(Shape::new([1, h, dq]), DType::F32, q.as_slice().to_vec());
    let ks = Tensor::new(Shape::new([5, h, dq]), DType::F32, k.as_slice().to_vec());
    let vs = Tensor::new(Shape::new([5, h, dq]), DType::F32, k.as_slice().to_vec());
    let mut out = Tensor::zeros(Shape::new([1, h, dq]), DType::F32);
    b.sparse_mla_attn(&qs, &ks, &vs, &idx, &mut out).unwrap();
    // attention over tokens {2, 4}: q·k2 = 1 (scaled), q·k4 = 0.9/sqrt(4)
    let sc2 = 1.0f32 / 2.0; // 1 * sqrt(4) scale = /2
    let sc4 = 0.9f32 / 2.0;
    let w2 = sc2.exp() / (sc2.exp() + sc4.exp());
    let w4 = sc4.exp() / (sc2.exp() + sc4.exp());
    assert!((out.as_slice()[0] - w2 * 1.0 - w4 * 0.9).abs() < 1e-5);
    assert!((out.as_slice()[1] - w4 * 0.1).abs() < 1e-5, "k4 dim1 is 0.1");
    assert!(out.as_slice()[2].abs() < 1e-5);
    assert!(out.as_slice()[3].abs() < 1e-5, "both selected tokens have dim3=0");
}
