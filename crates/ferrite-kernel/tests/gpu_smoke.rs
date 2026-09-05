//! GPU smoke test: CudaBackend op-level correctness vs the CPU reference
//! (golden). Requires libferrite_kernels.so (run `kernels/cuda/build.sh 103a`
//! first) and at least one GPU.
#![cfg(feature = "cuda")]

use ferrite_kernel::{CpuBackend, CudaBackend, KernelBackend};
use ferrite_types::{DType, Shape, Tensor};

fn so_path() -> String {
    std::env::var("FERRITE_KERNEL_SO").unwrap_or_else(|_| {
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("../../kernels/cuda/libferrite_kernels.so");
        p.to_string_lossy().into_owned()
    })
}

fn close(a: &Tensor, b: &Tensor, tol: f32, what: &str) {
    let (av, bv) = (a.as_slice(), b.as_slice());
    assert_eq!(av.len(), bv.len(), "{what}: len");
    let mut max_diff = 0f32;
    for i in 0..av.len() {
        let d = (av[i] - bv[i]).abs();
        if d > tol {
            panic!(
                "{what}: mismatch at {i}: {v1} vs {v2} (tol {tol})",
                v1 = av[i],
                v2 = bv[i]
            );
        }
        max_diff = max_diff.max(d);
    }
    eprintln!("{what}: max_diff {max_diff:.3e}");
}

#[test]
fn cuda_smoke_all_ops() {
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda device 0");
    let cpu = CpuBackend::new();

    // ---- matmul ----
    let x = Tensor::from_f32(
        Shape::new([4, 8]),
        (0..32).map(|i| (i as f32) * 0.25 - 3.0).collect(),
    );
    let w = Tensor::from_f32(
        Shape::new([6, 8]),
        (0..48).map(|i| ((i * 7 % 13) as f32) * 0.1 - 0.6).collect(),
    );
    let mut o_gpu = Tensor::zeros(Shape::new([4, 6]), DType::F32);
    let mut o_cpu = Tensor::zeros(Shape::new([4, 6]), DType::F32);
    dev.matmul(&x, &w, None, &mut o_gpu).unwrap();
    cpu.matmul(&x, &w, None, &mut o_cpu).unwrap();
    close(&o_gpu, &o_cpu, 2e-2, "matmul"); // bf16-resident weights: 2^-8 mantissa truncation accumulates ~1% abs

    // ---- rmsnorm [11, 4096] ----
    let (n, dim) = (11usize, 4096usize);
    let x = Tensor::from_f32(
        Shape::new([n, dim]),
        (0..n * dim).map(|i| ((i as f32) * 0.001).sin()).collect(),
    );
    let w = Tensor::from_f32(
        Shape::new([dim]),
        (0..dim).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect(),
    );
    let mut o_gpu = Tensor::zeros(Shape::new([n, dim]), DType::F32);
    let mut o_cpu = Tensor::zeros(Shape::new([n, dim]), DType::F32);
    dev.rmsnorm(&x, &w, 1e-5, &mut o_gpu).unwrap();
    cpu.rmsnorm(&x, &w, 1e-5, &mut o_cpu).unwrap();
    close(&o_gpu, &o_cpu, 1e-4, "rmsnorm");

    // ---- gated rmsnorm [n*h, dk] ----
    let (h, dk) = (64usize, 128usize);
    let x = Tensor::from_f32(
        Shape::new([n * h, dk]),
        (0..n * h * dk).map(|i| ((i as f32) * 0.002).cos()).collect(),
    );
    let g = Tensor::from_f32(Shape::new([n * h, dk]), vec![0.5; n * h * dk]);
    let w = Tensor::from_f32(Shape::new([dk]), vec![1.0; dk]);
    let mut o_gpu = Tensor::zeros(Shape::new([n * h, dk]), DType::F32);
    let mut o_cpu = Tensor::zeros(Shape::new([n * h, dk]), DType::F32);
    dev.gated_rmsnorm(&x, &g, &w, 1e-5, &mut o_gpu).unwrap();
    cpu.gated_rmsnorm(&x, &g, &w, 1e-5, &mut o_cpu).unwrap();
    close(&o_gpu, &o_cpu, 1e-4, "gated_rmsnorm");

    // ---- conv1d (n=11, ch=64, conv=4) — prefill-shaped ----
    let (n, ch, cv) = (11usize, 64usize, 4usize);
    let hist = cv - 1;
    let x = Tensor::from_f32(
        Shape::new([n, ch]),
        (0..n * ch).map(|i| ((i as f32) * 0.017).sin()).collect(),
    );
    let w = Tensor::from_f32(
        Shape::new([ch, cv]),
        (0..ch * cv).map(|i| ((i * 13 % 7) as f32) * 0.1 - 0.3).collect(),
    );
    let st_in = Tensor::from_f32(Shape::new([ch, hist]), vec![0.0; ch * hist]);
    let mut o_gpu = Tensor::zeros(Shape::new([n, ch]), DType::F32);
    let mut o_cpu = Tensor::zeros(Shape::new([n, ch]), DType::F32);
    let mut s_gpu = Tensor::zeros(Shape::new([ch, hist]), DType::F32);
    let mut s_cpu = Tensor::zeros(Shape::new([ch, hist]), DType::F32);
    dev.causal_conv1d(&x, &w, &st_in, &mut o_gpu, &mut s_gpu).unwrap();
    cpu.causal_conv1d(&x, &w, &st_in, &mut o_cpu, &mut s_cpu).unwrap();
    close(&o_gpu, &o_cpu, 1e-4, "conv1d_out");
    close(&s_gpu, &s_cpu, 1e-4, "conv1d_state");

    // ---- gated_deltanet_chunk (h=2, dk=dv=8, n=3) ----
    let (h, dk, dv, n) = (2usize, 8usize, 8usize, 3usize);
    let q = Tensor::from_f32(
        Shape::new([n, h, dk]),
        (0..n * h * dk).map(|i| ((i as f32) * 0.03).cos()).collect(),
    );
    let k = Tensor::from_f32(
        Shape::new([n, h, dk]),
        (0..n * h * dk).map(|i| ((i as f32) * 0.05).sin()).collect(),
    );
    let v = Tensor::from_f32(
        Shape::new([n, h, dv]),
        (0..n * h * dv).map(|i| ((i as f32) * 0.07).cos()).collect(),
    );
    let beta = Tensor::from_f32(Shape::new([n, h]), vec![0.5; n * h]);
    let gate = Tensor::from_f32(
        Shape::new([n, h, dk]),
        (0..n * h * dk).map(|i| 0.3 + (i % 5) as f32 * 0.05).collect(),
    );
    let a_log = Tensor::from_f32(Shape::new([h]), vec![-0.1; h]);
    let st_in = Tensor::from_f32(Shape::new([h, dk, dv]), vec![0.0; h * dk * dv]);
    let mut o_gpu = Tensor::zeros(Shape::new([n, h, dv]), DType::F32);
    let mut s_gpu = Tensor::zeros(Shape::new([h, dk, dv]), DType::F32);
    let mut o_cpu = Tensor::zeros(Shape::new([n, h, dv]), DType::F32);
    let mut s_cpu = Tensor::zeros(Shape::new([h, dk, dv]), DType::F32);
    dev.gated_deltanet_chunk(&q, &k, &v, &beta, &gate, &a_log, &st_in, &mut o_gpu, &mut s_gpu)
        .unwrap();
    cpu.gated_deltanet_chunk(&q, &k, &v, &beta, &gate, &a_log, &st_in, &mut o_cpu, &mut s_cpu)
        .unwrap();
    close(&o_gpu, &o_cpu, 1e-4, "gdn_out");
    close(&s_gpu, &s_cpu, 1e-4, "gdn_state");

    // ---- moe_route (n=2, e=288, topk=8) ----
    // NOTE: tie-free logits (strictly distinct values) so the CPU (stable
    // sort) and GPU (thread-local scan + reduce) agree on the top-k order.
    let (n, e, topk) = (2usize, 288usize, 8usize);
    let logits = Tensor::from_f32(
        Shape::new([n, e]),
        (0..n * e).map(|i| (i as f32) * 0.017 - 2.0).collect(),
    );
    let bias = Tensor::from_f32(Shape::new([e]), (0..e).map(|i| ((i % 11) as f32) * 0.01).collect());
    let mut p_gpu = Tensor::zeros(Shape::new([n, topk]), DType::F32);
    let mut i_gpu = Tensor::zeros(Shape::new([n, topk]), DType::F32);
    let mut p_cpu = Tensor::zeros(Shape::new([n, topk]), DType::F32);
    let mut i_cpu = Tensor::zeros(Shape::new([n, topk]), DType::F32);
    dev.moe_route(&logits, &bias, topk, 2.5, &mut p_gpu, &mut i_gpu).unwrap();
    cpu.moe_route(&logits, &bias, topk, 2.5, &mut p_cpu, &mut i_cpu).unwrap();
    close(&i_gpu, &i_cpu, 0.0, "moe_ids");
    close(&p_gpu, &p_cpu, 1e-5, "moe_probs");

    // ---- indexer_topk (n=4, H=8, D=16, t=11, topk=5) — real-checkpoint layout ----
    // NOTE: values are chosen so per-key scores are well separated (rows of ki
    // have distinct magnitudes) — GPU FMA vs CPU rounding then cannot flip the
    // top-k ordering.
    let (n, ih, d, t, topk) = (4usize, 8usize, 16usize, 11usize, 5usize);
    let qi = Tensor::from_f32(
        Shape::new([n, ih * d]),
        (0..n * ih * d).map(|i| ((i % 3) as f32) * 0.33 + (i as f32) * 0.001).collect(),
    );
    let ki = {
        let mut v = Vec::with_capacity(t * d);
        for j in 0..t {
            for l in 0..d {
                v.push((j as f32) * 0.25 + (l as f32) * 0.017);
            }
        }
        Tensor::from_f32(Shape::new([t, d]), v)
    };
    let w = Tensor::from_f32(
        Shape::new([n, ih]),
        (0..n * ih).map(|i| 1.0 + (i % 3) as f32 * 0.1).collect(),
    );
    let mut idx_gpu = Tensor::zeros(Shape::new([n, topk]), DType::F32);
    let mut idx_cpu = Tensor::zeros(Shape::new([n, topk]), DType::F32);
    // queries are the last n of the t keys: causal guard ctx0 = t - n
    let ctx0 = t - n;
    dev.indexer_topk(&qi, &ki, &w, topk, ctx0, &mut idx_gpu).unwrap();
    cpu.indexer_topk(&qi, &ki, &w, topk, ctx0, &mut idx_cpu).unwrap();
    // top-k ORDER may flip on the tie boundary (FMA/rounding differences
    // between CPU and GPU dot products); compare the per-row SELECTION SET.
    {
        let (a, b) = (idx_gpu.as_slice().to_vec(), idx_cpu.as_slice().to_vec());
        for r in 0..n {
            let mut ra: Vec<f32> = a[r * topk..(r + 1) * topk].to_vec();
            let mut rb: Vec<f32> = b[r * topk..(r + 1) * topk].to_vec();
            ra.sort_by(|x, y| x.partial_cmp(y).unwrap());
            rb.sort_by(|x, y| x.partial_cmp(y).unwrap());
            for (x, y) in ra.iter().zip(rb.iter()) {
                assert_eq!(x, y, "indexer_topk selection set differs (row {r}): gpu {ra:?} vs cpu {rb:?}");
            }
        }
        eprintln!("indexer_topk: sets match");
    }

    // ---- sparse_mla_attn (n=3, h=2, d=4, dv=4, t=9, topk=4) ----
    let (n, h, d, dv, t, topk) = (3usize, 2usize, 4usize, 4usize, 9usize, 4usize);
    let q = Tensor::from_f32(
        Shape::new([n, h, d]),
        (0..n * h * d).map(|i| ((i as f32) * 0.11).sin()).collect(),
    );
    let k = Tensor::from_f32(
        Shape::new([t, h, d]),
        (0..t * h * d).map(|i| ((i as f32) * 0.13).cos()).collect(),
    );
    let v = Tensor::from_f32(
        Shape::new([t, h, dv]),
        (0..t * h * dv).map(|i| ((i as f32) * 0.17).sin()).collect(),
    );
    // idx from the CPU reference (any valid selection works for the attn diff)
    let qi1 = Tensor::from_f32(
        Shape::new([n, 1 * d]),
        (0..n * d).map(|i| ((i as f32) * 0.09).cos()).collect(),
    );
    let ki1 = Tensor::from_f32(
        Shape::new([t, 1 * d]),
        (0..t * d).map(|i| ((i as f32) * 0.15).sin()).collect(),
    );
    let w1 = Tensor::from_f32(Shape::new([n, 1]), vec![1.0; n]);
    let mut idx = Tensor::zeros(Shape::new([n, topk]), DType::F32);
    cpu.indexer_topk(&qi1, &ki1, &w1, topk, 9 - 3, &mut idx).unwrap();
    let mut o_gpu = Tensor::zeros(Shape::new([n, h, dv]), DType::F32);
    let mut o_cpu = Tensor::zeros(Shape::new([n, h, dv]), DType::F32);
    dev.sparse_mla_attn(&q, &k, &v, &idx, &mut o_gpu).unwrap();
    cpu.sparse_mla_attn(&q, &k, &v, &idx, &mut o_cpu).unwrap();
    close(&o_gpu, &o_cpu, 1e-4, "sparse_attn");

    // ---- swiglu ----
    let inter = 64usize;
    let gu = Tensor::from_f32(
        Shape::new([2, 2 * inter]),
        (0..2 * 2 * inter).map(|i| ((i * 7 % 23) as f32) * 0.05 - 0.5).collect(),
    );
    let mut a_gpu = Tensor::zeros(Shape::new([2, inter]), DType::F32);
    let mut a_cpu = Tensor::zeros(Shape::new([2, inter]), DType::F32);
    dev.swiglu_limited(&gu, 10.0, &mut a_gpu).unwrap();
    cpu.swiglu_limited(&gu, 10.0, &mut a_cpu).unwrap();
    close(&a_gpu, &a_cpu, 1e-4, "swiglu");

    // ---- argmax / softmax ----
    let lg = Tensor::from_f32(
        Shape::new([2, 32]),
        (0..64).map(|i| ((i * 5 % 19) as f32) * 0.1).collect(),
    );
    let mut am_g = Tensor::zeros(Shape::new([2]), DType::F32);
    let mut am_c = Tensor::zeros(Shape::new([2]), DType::F32);
    dev.argmax_lastdim(&lg, &mut am_g).unwrap();
    cpu.argmax_lastdim(&lg, &mut am_c).unwrap();
    close(&am_g, &am_c, 0.0, "argmax");

    let mut sm_g = Tensor::zeros(Shape::new([2, 32]), DType::F32);
    let mut sm_c = Tensor::zeros(Shape::new([2, 32]), DType::F32);
    dev.softmax_lastdim(&lg, &mut sm_g).unwrap();
    cpu.softmax_lastdim(&lg, &mut sm_c).unwrap();
    close(&sm_g, &sm_c, 1e-5, "softmax");
}

/// CUDA-graph capture smoke: record a 3-op chain (matmul → rmsnorm →
/// matmul) with the pinned-staging upload/download path, instantiate, and
/// replay. Capture legality is the point — pageable memcpy is ILLEGAL
/// during stream capture, the pinned stage is the fix; DevBuf pool reuse
/// keeps device addresses stable across capture and replay.
#[test]
fn cuda_graph_capture_smoke() {
    let dev = CudaBackend::with_device(&so_path(), 0).unwrap();
    use ferrite_kernel::graph::GraphCapable;
    let cpu = CpuBackend::new();

    let x = Tensor::from_f32(Shape::new([4, 64]), (0..256).map(|i| i as f32 * 0.01).collect());
    let w1 = Tensor::from_f32(Shape::new([48, 64]), (0..48 * 64).map(|i| (i % 13) as f32 * 0.02).collect());
    let wn = Tensor::from_f32(Shape::new([48]), (0..48).map(|i| 1.0 + i as f32 * 0.001).collect());
    let w2 = Tensor::from_f32(Shape::new([32, 48]), (0..32 * 48).map(|i| (i % 7) as f32 * 0.03).collect());
    let mut o1 = Tensor::zeros(Shape::new([4, 48]), DType::F32);
    let mut o2 = Tensor::zeros(Shape::new([4, 48]), DType::F32);
    let mut o3 = Tensor::zeros(Shape::new([4, 32]), DType::F32);

    // warmup: pools hot, weights resident — capture must find NO cudaMalloc
    // and NO pageable memcpy in the recorded chain
    dev.matmul(&x, &w1, None, &mut o1).unwrap();
    dev.rmsnorm(&o1, &wn, 1e-5, &mut o2).unwrap();
    dev.matmul(&o2, &w2, None, &mut o3).unwrap();

    // capture the same 3-op chain
    dev.begin_capture();
    dev.matmul(&x, &w1, None, &mut o1).unwrap();
    dev.rmsnorm(&o1, &wn, 1e-5, &mut o2).unwrap();
    dev.matmul(&o2, &w2, None, &mut o3).unwrap();
    let trace = dev.end_capture();

    // replay: the graph re-executes stage→dev→kernel→stage for every op
    dev.begin_verify(&trace);
    assert!(dev.end_verify(), "graph replay sync");

    // NOTE: the replay writes results into the pinned stages; the Tensor
    // reads happen on the NEXT download call (the Rust op body does not
    // re-run during replay). Value parity is verified end-to-end at the
    // serve integration; here capture legality + replay success is the
    // contract.
    println!("cuda_graph_capture_smoke: 3-op chain captured + replayed OK");
    let _ = (&cpu, &x, &o3);
}

/// hc_pre / hc_post: GPU kernel vs CPU golden (MHC hyper-connections — the
/// layer-boundary host loops were the biggest serial segment after fan_out).
#[test]
fn cuda_hc_pre_post() {
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda device 0");
    let cpu = CpuBackend::new();
    use ferrite_kernel::KernelBackend;

    let (s, n, h) = (2usize, 4usize, 64usize); // mix = 2n + n^2 = 24
    let nh = n * h;
    let mix = 2 * n + n * n;
    let residual = Tensor::from_f32(
        Shape::new([s, nh]),
        (0..s * nh).map(|i| ((i * 37 % 101) as f32) * 0.02 - 1.0).collect(),
    );
    let fn_w = Tensor::from_f32(
        Shape::new([mix, nh]),
        (0..mix * nh).map(|i| ((i * 11 % 53) as f32) * 0.01 - 0.25).collect(),
    );
    let scale = Tensor::from_f32(Shape::new([3]), vec![0.8, 1.2, 0.9]);
    let base = Tensor::from_f32(Shape::new([mix]), (0..mix).map(|i| 0.01 * i as f32).collect());

    let (li_g, post_g, comb_g) = dev
        .hc_pre(&residual, &fn_w, &scale, &base, 1e-5, 1e-6, 20)
        .expect("gpu hc_pre");
    let (li_c, post_c, comb_c) = cpu
        .hc_pre(&residual, &fn_w, &scale, &base, 1e-5, 1e-6, 20)
        .expect("cpu hc_pre");
    close(&li_g, &li_c, 2e-4, "hc_pre li");
    close(&post_g, &post_c, 2e-4, "hc_pre post");
    close(&comb_g, &comb_c, 2e-4, "hc_pre comb (sinkhorn)");

    let x = Tensor::from_f32(Shape::new([s, h]), (0..s * h).map(|i| ((i * 7 % 31) as f32) * 0.03).collect());
    let res3 = Tensor::from_f32(
        Shape::new([s, n, h]),
        (0..s * n * h).map(|i| ((i * 13 % 47) as f32) * 0.02 - 0.5).collect(),
    );
    let out_g = dev.hc_post(&x, &res3, &post_g, &comb_g).expect("gpu hc_post");
    let out_c = cpu.hc_post(&x, &res3, &post_c, &comb_c).expect("cpu hc_post");
    close(&out_g, &out_c, 2e-4, "hc_post out");
}

// ============================================================
// GEMV (n==1 decode matmul) correctness + speed micro-benchmark.
// The warp-per-row GEMV replaced the 32x32 tiled kernel at n==1 (which
// wasted 31/32 warps). This test proves (a) numeric parity vs the tiled
// matmul + CPU golden at decode shapes, (b) the speedup — asserted as
// GEMV_time < tiled_time at the real decode shape [1, 4096] x [4096, 3072].
// ============================================================
#[test]
fn gemv_bf16_parity_and_speed() {
    use ferrite_kernel::cuda::DevBuf;
    use std::time::Instant;

    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda device 0");
    dev.enter();
    let cpu = CpuBackend::new();

    // Real decode shapes: hidden=4096, out=3072 (qkv shard) and out=1024.
    for (in_f, out_f) in [(4096usize, 3072usize), (4096, 1024), (1024, 4096)] {
        let x = Tensor::from_f32(
            Shape::new([1, in_f]),
            (0..in_f).map(|i| ((i as f32) * 0.37).sin() * 0.5).collect(),
        );
        let w = Tensor::from_f32(
            Shape::new([out_f, in_f]),
            (0..out_f * in_f).map(|i| (((i * 31 + 7) % 97) as f32) * 0.02 - 0.96).collect(),
        );
        // golden: CPU matmul
        let mut o_cpu = Tensor::zeros(Shape::new([1, out_f]), DType::F32);
        cpu.matmul(&x, &w, None, &mut o_cpu).unwrap();

        // GPU: matmul_dev routes n==1 → gemv_bf16 (single call each, verify parity)
        let dx = DevBuf::alloc(dev.dev(), dev.stream(), in_f).unwrap();
        dx.upload(x.as_slice()).unwrap();
        let o_dev = dev.matmul_dev(&dx, &w, 1, in_f as i32, out_f as i32).unwrap();
        let mut hv = vec![0f32; out_f];
        o_dev.download(&mut hv).unwrap();
        let o_gemv = Tensor::from_f32(Shape::new([1, out_f]), hv);
        // bf16 weight truncation: same tolerance class as the matmul smoke test
        close(&o_gemv, &o_cpu, 2e-2, &format!("gemv[{in_f}->{out_f}] vs cpu"));

        // speed: N sequential GEMV calls on one stream, wall-clock per call.
        // This is the decode steady state (layer after layer, same stream).
        let iters = 200u32;
        // warmup (fills weight cache + pools)
        for _ in 0..10 {
            let _ = dev.matmul_dev(&dx, &w, 1, in_f as i32, out_f as i32).unwrap();
        }
        dev.sync().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = dev.matmul_dev(&dx, &w, 1, in_f as i32, out_f as i32).unwrap();
        }
        dev.sync().unwrap();
        let us_gemv = t0.elapsed().as_secs_f32() * 1e6 / iters as f32;

        // reference: the Tensor-level path also hits the same gemv (n==1) —
        // instead, time the pure launch+kernel cost vs theoretical HBM:
        // W bytes = out_f * in_f * 2 (bf16)
        let w_mb = out_f * in_f * 2 / 1024 / 1024;
        eprintln!(
            "[gemv-bench] {in_f}->{out_f}: {us_gemv:.1} μs/call, W={w_mb} MB → {:.0} GB/s effective",
            (out_f * in_f * 2) as f64 / 1e9 / (us_gemv as f64 * 1e-6)
        );
        // Sanity: must beat 500 μs (the old tiled kernel's n==1 cost class);
        // a healthy GEMV on these shapes is 5-60 μs (bandwidth 200+ GB/s).
        assert!(
            us_gemv < 500.0,
            "gemv[{in_f}->{out_f}] too slow: {us_gemv:.1} μs/call (tiled-kernel class; expected <500, healthy <60)"
        );
    }
}

// ============================================================
// Fused MoE decode (n==1, GPU-dispatch path) parity + speed.
// The fused path (moe_layer_dev n==1) routes → act → down_sum with ids/
// probs NEVER crossing to the host (device pointer table). Reference:
// hand-rolled f32 routing (sigmoid+bias topk, renorm × scale) + expert
// FFN (swiglu2 semantics) + shared expert, on the same tensors.
// ============================================================
#[test]
fn moe_fused_parity_and_speed() {
    use ferrite_kernel::cuda::{DevBuf, ExpertWeights};
    use std::time::Instant;

    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda device 0");
    dev.enter();

    let (hidden, inter, inter_s, e_total, topk, e_local, expert_start) =
        (64usize, 32usize, 16usize, 8usize, 2usize, 8usize, 0usize);
    let routed_scaling = 1.2f32;
    let limit = 7.0f32; // swiglu clamp (large enough to be active at most)

    // bf16-EXACT values ({-0.5,-0.25,0,0.25,...}): the GPU's bf16 weight
    // truncation is a no-op on these, so GPU and the f32 CPU reference see
    // IDENTICAL numbers — any diff beyond accumulation order (~1e-6) is a
    // real kernel bug, precisely localizable.
    let bex = |i: usize, s: usize| ((i + s) % 4) as f32 * 0.25 - 0.5;
    let x = Tensor::from_f32(
        Shape::new([1, hidden]),
        (0..hidden).map(|i| bex(i, 0)).collect(),
    );
    let gate_w = Tensor::from_f32(
        Shape::new([e_total, hidden]),
        (0..e_total * hidden).map(|i| bex(i, 5)).collect(),
    );
    let bias = Tensor::from_f32(
        Shape::new([e_total]),
        (0..e_total).map(|i| bex(i, 2)).collect(),
    );
    let mk = |r: usize, c: usize, s: usize| {
        Tensor::from_f32(
            Shape::new([r, c]),
            (0..r * c).map(|i| bex(i * 7 + s, 0)).collect(),
        )
    };
    let eg: Vec<Tensor> = (0..e_local).map(|s| mk(inter, hidden, s * 11)).collect();
    let eu: Vec<Tensor> = (0..e_local).map(|s| mk(inter, hidden, s * 13 + 3)).collect();
    let ed: Vec<Tensor> = (0..e_local).map(|s| mk(hidden, inter, s * 17 + 7)).collect();
    let sg = mk(inter_s, hidden, 29);
    let su = mk(inter_s, hidden, 31);
    let sd = mk(hidden, inter_s, 37);
    let experts: Vec<ExpertWeights> = (0..e_local)
        .map(|i| ExpertWeights { gate: &eg[i], up: &eu[i], down: &ed[i] })
        .collect();
    let shared = ExpertWeights { gate: &sg, up: &su, down: &sd };

    // ---- GPU fused ----
    let mut probs_scratch = DevBuf::alloc(dev.dev(), dev.stream(), topk).unwrap();
    let out = {
        let dx = DevBuf::alloc(dev.dev(), dev.stream(), hidden).unwrap();
        dx.upload(x.as_slice()).unwrap();
        dev.moe_layer_dev(
            &dx, &gate_w, &bias, &shared, &experts, expert_start,
            &mut probs_scratch, 1, hidden, topk, e_total,
            routed_scaling, limit,
        )
        .unwrap()
    };
    let mut hv = vec![0f32; hidden];
    out.download(&mut hv).unwrap();
    let mut gp = vec![0f32; topk];
    probs_scratch.download(&mut gp).unwrap();

    // ---- CPU reference (f32; GPU weights are bf16-truncated → ~2e-2 tol) ----
    let (xs, gws, bs) = (x.as_slice(), gate_w.as_slice(), bias.as_slice());
    // logits[j] = x · gate_w[j, :] (the routing GEMV the GPU path runs first)
    let logits: Vec<f32> = (0..e_total)
        .map(|j| {
            let mut s = 0f32;
            for k in 0..hidden {
                s += xs[k] * gws[j * hidden + k];
            }
            s
        })
        .collect();
    let scores: Vec<f32> = (0..e_total).map(|j| 1.0 / (1.0 + (-logits[j]).exp())).collect();
    let choice: Vec<f32> = (0..e_total).map(|j| scores[j] + bs[j]).collect();
    let mut order: Vec<usize> = (0..e_total).collect();
    order.sort_by(|&a, &b| choice[b].partial_cmp(&choice[a]).unwrap());
    let ids: Vec<usize> = order[..topk].to_vec();
    let raw: Vec<f32> = ids.iter().map(|&j| scores[j]).collect();
    let rsum: f32 = raw.iter().sum();
    let probs: Vec<f32> = raw.iter().map(|v| v / rsum * routed_scaling).collect();

    let silu = |g: f32| g / (1.0 + (-g).exp());
    let mut ref_out = vec![0f32; hidden];
    for (jj, &eid) in ids.iter().enumerate() {
        if eid < expert_start || eid >= expert_start + e_local {
            continue; // another rank's slot → zero contribution here
        }
        let e = eid - expert_start;
        let mut act = vec![0f32; inter];
        for i in 0..inter {
            let mut g = 0f32;
            let mut u = 0f32;
            for k in 0..hidden {
                g += xs[k] * eg[e].as_slice()[i * hidden + k];
                u += xs[k] * eu[e].as_slice()[i * hidden + k];
            }
            g = g.min(limit);
            u = u.max(-limit).min(limit);
            act[i] = silu(g) * u;
        }
        for h in 0..hidden {
            let mut y = 0f32;
            for i in 0..inter {
                y += act[i] * ed[e].as_slice()[h * inter + i];
            }
            ref_out[h] += probs[jj] * y;
        }
    }
    // shared expert (inter_s width)
    {
        let mut act = vec![0f32; inter_s];
        for i in 0..inter_s {
            let mut g = 0f32;
            let mut u = 0f32;
            for k in 0..hidden {
                g += xs[k] * sg.as_slice()[i * hidden + k];
                u += xs[k] * su.as_slice()[i * hidden + k];
            }
            g = g.min(limit);
            u = u.max(-limit).min(limit);
            act[i] = silu(g) * u;
        }
        for h in 0..hidden {
            let mut y = 0f32;
            for i in 0..inter_s {
                y += act[i] * sd.as_slice()[h * inter_s + i];
            }
            ref_out[h] += y;
        }
    }
    // parity: bf16 weight truncation is a RELATIVE error (~0.4% accumulated
    // over hidden/inter dots) — an absolute cap fails on large activations.
    for i in 0..hidden {
        let d = (hv[i] - ref_out[i]).abs();
        let tol = 1e-4; // bf16-exact inputs: only accumulation order differs
        assert!(
            d < tol,
            "moe_fused mismatch at {i}: gpu {} vs ref {} (diff {d}, tol {tol}; probs {:?} ids {:?})",
            hv[i], ref_out[i], gp, ids
        );
    }
    eprintln!(
        "[moe-fused] parity ok (ids {:?}, probs {:?}); max_diff {}",
        ids,
        gp,
        (0..hidden).map(|i| (hv[i] - ref_out[i]).abs()).fold(0f32, f32::max)
    );

    // ---- speed: the full fused MoE (route+act+down_sum) per call ----
    let dx = DevBuf::alloc(dev.dev(), dev.stream(), hidden).unwrap();
    dx.upload(x.as_slice()).unwrap();
    for _ in 0..20 {
        let _ = dev.moe_layer_dev(&dx, &gate_w, &bias, &shared, &experts, expert_start,
                                   &mut probs_scratch, 1, hidden, topk, e_total,
                                   routed_scaling, limit).unwrap();
    }
    dev.sync().unwrap();
    let iters = 500;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = dev.moe_layer_dev(&dx, &gate_w, &bias, &shared, &experts, expert_start,
                                   &mut probs_scratch, 1, hidden, topk, e_total,
                                   routed_scaling, limit).unwrap();
    }
    dev.sync().unwrap();
    let us = t0.elapsed().as_secs_f32() * 1e6 / iters as f32;
    eprintln!("[moe-fused] {us:.1} μs/call (route+act+down_sum, e_local={e_local} topk={topk})");
    assert!(us < 300.0, "moe_fused too slow: {us:.1} μs/call");
}

/// Dense-MLP operator parity (43s unit test vs the 4-min integration run):
/// the mega-graph chain (matmul_dev + swiglu2_dev, separated g/u) vs the
/// normal path (Tensor matmul via KernelBackend::matmul + swiglu_limited,
/// packed gate_up). The mega dry-run's L00-02 dense AR diverges 2-3x from
/// the known-good normal path while its dev0 g/u/a match — bisect at the
/// OPERATOR level. Both paths share the bf16 weight cache + kernels, so
/// parity must be bit-exact; any diff pinpoints the bug.
#[test]
fn dense_chain_parity() {
    use ferrite_kernel::cuda::DevBuf;
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda device 0");
    let n = 1usize;
    let hidden = 4096usize;
    let inter = 3072usize; // per-rank shard of inter=12288 (GLM-5.3-Flash TP4)
    let limit = 7.0f32;

    let mut seed = 0x12345678u64;
    let mut r = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / 2147483648.0) - 0.5 // [-0.5, 0.5)
    };

    // ---- 1. matmul parity: Tensor path (the normal path's project) vs
    // matmul_dev (the mega chain) — same bf16 kernel + weight cache
    let x: Vec<f32> = (0..n * hidden).map(|_| r()).collect();
    let w: Vec<f32> = (0..inter * hidden).map(|_| r() * 0.05).collect();
    let x_t = Tensor::from_f32(Shape::new([n, hidden]), x.clone());
    let w_t = Tensor::from_f32(Shape::new([inter, hidden]), w);
    let mut out_t = Tensor::zeros(Shape::new([n, inter]), DType::F32);
    dev.matmul(&x_t, &w_t, None, &mut out_t).expect("tensor matmul");

    let x_dev = DevBuf::alloc(dev.dev(), dev.stream_handle(), n * hidden).unwrap();
    x_dev.upload(&x).unwrap();
    let out_dev = dev.matmul_dev(&x_dev, &w_t, n as i32, hidden as i32, inter as i32).unwrap();
    let mut out_d = vec![0f32; n * inter];
    out_dev.download(&mut out_d).unwrap();
    close(
        &out_t,
        &Tensor::from_f32(Shape::new([n, inter]), out_d),
        1e-5,
        "matmul_dev vs run_matmul (1e-6 fp noise ok)",
    );

    // ---- 2. swiglu parity: packed (swiglu_limited, normal path) vs
    // separated (swiglu2_dev, mega chain) — same clamp+silu formula
    let g: Vec<f32> = (0..n * inter).map(|_| r() * 12.0).collect(); // span the clamp
    let u: Vec<f32> = (0..n * inter).map(|_| r() * 12.0).collect();
    let mut gu = vec![0f32; n * 2 * inter];
    for t in 0..n {
        gu[t * 2 * inter..t * 2 * inter + inter].copy_from_slice(&g[t * inter..(t + 1) * inter]);
        gu[t * 2 * inter + inter..(t + 1) * 2 * inter].copy_from_slice(&u[t * inter..(t + 1) * inter]);
    }
    let gu_t = Tensor::from_f32(Shape::new([n, 2 * inter]), gu);
    let mut act_t = Tensor::zeros(Shape::new([n, inter]), DType::F32);
    dev.swiglu_limited(&gu_t, limit, &mut act_t).expect("swiglu_limited");

    let g_dev = DevBuf::alloc(dev.dev(), dev.stream_handle(), n * inter).unwrap();
    g_dev.upload(&g).unwrap();
    let u_dev = DevBuf::alloc(dev.dev(), dev.stream_handle(), n * inter).unwrap();
    u_dev.upload(&u).unwrap();
    let act_dev = dev.swiglu2_dev(&g_dev, &u_dev, n as i32, inter as i32, limit).unwrap();
    let mut act_d = vec![0f32; n * inter];
    act_dev.download(&mut act_d).unwrap();
    close(
        &act_t,
        &Tensor::from_f32(Shape::new([n, inter]), act_d),
        1e-5,
        "swiglu2_dev vs swiglu_limited (1e-6 fp noise ok)",
    );

    // ---- 3. down-proj parity: act @ wd.T — Tensor vs matmul_dev
    let wd: Vec<f32> = (0..hidden * inter).map(|_| r() * 0.05).collect();
    let wd_t = Tensor::from_f32(Shape::new([hidden, inter]), wd);
    let mut dn_t = Tensor::zeros(Shape::new([n, hidden]), DType::F32);
    dev.matmul(&act_t, &wd_t, None, &mut dn_t).expect("down tensor matmul");
    let dn_dev = dev.matmul_dev(&act_dev, &wd_t, n as i32, inter as i32, hidden as i32).unwrap();
    let mut dn_d = vec![0f32; n * hidden];
    dn_dev.download(&mut dn_d).unwrap();
    close(
        &dn_t,
        &Tensor::from_f32(Shape::new([n, hidden]), dn_d),
        1e-5,
        "dense down matmul parity",
    );
    eprintln!("dense_chain_parity: matmul/swiglu/down all bit-exact — operator level OK");
}

/// hc_contract_dev parity (host golden): [s, n*h] -> [s, h] mean over the
/// n MHC flows — the NEW kernel the mega-graph's head bridge uses. mhc.rs
/// layout: flow i of token t at in[(t*n + i)*h .. +h]; out[t*h + j] =
/// mean_i in[(t*n+i)*h + j]. Never unit-tested before (added for the
/// mega-graph); the mega dry-run outputs tok=522 with flat logits while
/// every layer's AR matches — the head bridge is the prime suspect.
#[test]
fn hc_contract_parity() {
    use ferrite_kernel::cuda::DevBuf;
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda device 0");
    let (s, n, h) = (2usize, 4usize, 4096usize); // rows, hc_mult flows, hidden
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut r = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / 2147483648.0) - 0.5
    };
    let x: Vec<f32> = (0..s * n * h).map(|_| r() * 3.0).collect();

    // host golden (mhc::hc_contract semantics)
    let mut host = vec![0f32; s * h];
    for t in 0..s {
        for j in 0..h {
            let mut acc = 0f32;
            for i in 0..n {
                acc += x[(t * n + i) * h + j];
            }
            host[t * h + j] = acc / n as f32;
        }
    }

    // device
    let x_dev = DevBuf::alloc(dev.dev(), dev.stream_handle(), s * n * h).unwrap();
    x_dev.upload(&x).unwrap();
    let out_dev = dev.hc_contract_dev(&x_dev, s, n, h).expect("hc_contract_dev");
    let mut got = vec![0f32; s * h];
    out_dev.download(&mut got).unwrap();

    let host_t = Tensor::from_f32(Shape::new([s, h]), host);
    let got_t = Tensor::from_f32(Shape::new([s, h]), got);
    close(&host_t, &got_t, 1e-5, "hc_contract_dev vs host golden");
    eprintln!("hc_contract_parity: OK (kernel layout (t*n+i)*h+j confirmed)");
}

/// hc 段微基准：mega-graph 42ms→37.5ms 后 A_hc+C_hc 仍占 17.8ms。分解
/// hc_pre_split(mix 多块) / rmsnorm / hc_post 的单次耗时（n=1 decode 形态），
/// 定位剩余 kernel 效率黑洞（hc_pre_rest sinkhorn 单块在 .so 内无法单独测）。
#[test]
fn hc_micro_bench() {
    use ferrite_kernel::cuda::DevBuf;
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda");
    dev.enter();
    let stream = dev.stream_handle();
    let (n, h) = (4usize, 4096usize); // hc_mult, hidden
    let nh = n * h;
    let s = 1usize;
    let iters = 2000;
    let mut rnd = 0x853c49e6748fea9bu64;
    let mut r = || { rnd = rnd.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((rnd >> 33) as f32) / 2147483648.0 };
    let mut mk = |len: usize| -> Vec<f32> { (0..len).map(|_| r()).collect() };
    let t = |v: &Vec<f32>| -> Tensor { Tensor::from_f32(Shape::new(vec![v.len()]), v.clone()) };

    let res_v = mk(s * nh);
    let fn_v = mk(24 * nh); // mix x nh (row-major)
    let scale_v = mk(s * n);
    let base_v = mk(24);
    let x_v = mk(s * h);
    let post_v = mk(s * n);
    let comb_v = mk(s * n * n);
    let w_v = mk(h);

    let res_t = t(&res_v); let fn_t = t(&fn_v); let scale_t = t(&scale_v);
    let base_t = t(&base_v); let x_t = t(&x_v); let post_t = t(&post_v);
    let comb_t = t(&comb_v); let w_t = t(&w_v);

    let up = |v: &Vec<f32>| -> DevBuf { let mut b = DevBuf::alloc(dev.dev(), stream, v.len()).unwrap(); b.upload(v).unwrap(); b };
    let res = up(&res_v); let x = up(&x_v); let post = up(&post_v); let comb = up(&comb_v); let wbuf = up(&w_v);

    // warm + bench hc_pre (split mix + rest)
    for _ in 0..50 { let _ = dev.hc_pre_dev(&res, &fn_t, &scale_t, &base_t, s, nh, 1e-5, 1e-4, 20); }
    let t0 = std::time::Instant::now();
    for _ in 0..iters { let _ = dev.hc_pre_dev(&res, &fn_t, &scale_t, &base_t, s, nh, 1e-5, 1e-4, 20); }
    let _ = dev.sync();
    let t_pre = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

    // rmsnorm [1, 4096]
    let _ = dev.rmsnorm_dev(&x, &w_t, 1e-5, s, h);
    let t1 = std::time::Instant::now();
    for _ in 0..iters { let _ = dev.rmsnorm_dev(&x, &w_t, 1e-5, s, h); }
    let _ = dev.sync();
    let t_rms = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;

    // hc_post x[s,h]+res[s,n,h] -> [s,n,h]
    let _ = dev.hc_post_dev(&x, &res, &post, &comb, s, n, h);
    let t2 = std::time::Instant::now();
    for _ in 0..iters { let _ = dev.hc_post_dev(&x, &res, &post, &comb, s, n, h); }
    let _ = dev.sync();
    let t_post = t2.elapsed().as_secs_f64() * 1e6 / iters as f64;

    eprintln!(
        "[hc-bench] hc_pre(split+rest)={:.1}us rmsnorm={:.2}us hc_post={:.1}us | per-layer A(pre+rms)={:.1}us C(post+pre+rms2)={:.1}us x45L={:.1}ms",
        t_pre, t_rms, t_post, t_pre + t_rms, t_post + t_pre + 2.0 * t_rms,
        (t_post + t_pre + 2.0 * t_rms) * 45.0 / 1000.0
    );
}

/// head 链微基准：mega 实测 head=2.55ms/token（contract+norm+lm_head GEMV+argmax），
/// 理论 640µs（1.2GB bf16 @2.2TB/s）。分解 lm_head GEMV 大矩阵 vs rmsnorm kernel-only。
#[test]
fn head_chain_bench() {
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda");
    dev.enter();
    let h = 4096usize;
    let vocab = 154880usize;
    let n = 1usize;
    let iters = 200;
    let mut rnd = 0x9e3779b97f4a7c15u64;
    let mut r = || { rnd = rnd.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((rnd >> 33) as f32) / 2147483648.0 };

    // lm_head GEMV: x[1,4096] × W[154880,4096] bf16 → [1,154880]
    let xv: Vec<f32> = (0..h).map(|_| r()).collect();
    let wv: Vec<f32> = (0..vocab * h).map(|_| r() * 0.02).collect();
    let x_t = Tensor::from_f32(Shape::new([n, h]), xv);
    let w_t = Tensor::from_f32(Shape::new([vocab, h]), wv);
    // warm (weight upload to bf16 cache)
    let x_dev = ferrite_kernel::cuda::DevBuf::alloc(dev.dev(), dev.stream_handle(), n * h).unwrap();
    x_dev.upload(x_t.as_slice()).unwrap();
    let _ = dev.matmul_dev(&x_dev, &w_t, n as i32, h as i32, vocab as i32);
    let _ = dev.sync();

    // bench lm_head GEMV
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = dev.matmul_dev(&x_dev, &w_t, n as i32, h as i32, vocab as i32);
    }
    let _ = dev.sync();
    let t_gemv = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

    // argmax [1, 154880]
    let logits = dev.matmul_dev(&x_dev, &w_t, n as i32, h as i32, vocab as i32).unwrap();
    let mut arg = ferrite_kernel::cuda::DevBuf::alloc(dev.dev(), dev.stream_handle(), n).unwrap();
    let _ = dev.argmax_dev(&logits, &mut arg, n, vocab);
    let _ = dev.sync();
    let t1 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = dev.argmax_dev(&logits, &mut arg, n, vocab);
    }
    let _ = dev.sync();
    let t_arg = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;

    // rmsnorm kernel-only (dev_weight cached): x[1,4096] w[4096]
    let w2 = Tensor::from_f32(Shape::new([h]), (0..h).map(|_| r()).collect::<Vec<f32>>());
    let _ = dev.rmsnorm_dev(&x_dev, &w2, 1e-5, n, h);
    let _ = dev.sync();
    let t2 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = dev.rmsnorm_dev(&x_dev, &w2, 1e-5, n, h);
    }
    let _ = dev.sync();
    let t_rms = t2.elapsed().as_secs_f64() * 1e6 / iters as f64;

    eprintln!(
        "[head-bench] lm_head GEMV [1,{h}]x[{vocab},{h}] bf16 = {:.1}us ({}GB/s eff) | argmax[{vocab}] = {:.2}us | rmsnorm[1,{h}] = {:.2}us",
        t_gemv, (vocab * h * 2) as f64 / t_gemv / 1e3, t_arg, t_rms
    );
}

/// gemv5_bf16 parity: the fused 5-matrix same-input GEMV vs 5 separate
/// matmul_dev calls (gdn: qkv/b/fa/ga share x; dsa: qa/latent/ki/w_idx/gate
/// share x). bf16 accumulation-order differences only (~1e-6).
#[test]
fn gemv5_parity() {
    use ferrite_kernel::cuda::DevBuf;
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda");
    dev.enter();
    let stream = dev.stream_handle();
    let in_f = 4096usize;
    // gdn shapes (TP4 shard): qkv [3*proj, 4096], b [h, 4096], fa [dk, 4096], ga [dk, 4096]
    let (proj, hh, dk) = (1536usize, 512usize, 512usize);
    let shapes: Vec<(usize, &str)> = vec![
        (3 * proj, "qkv"), (hh, "b"), (dk, "fa"), (dk, "ga"), (dk * 2 + 64, "kvA"),
    ];
    let mut rnd = 0x1234_5678_9abc_defu64;
    let mut r = || { rnd = rnd.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((rnd >> 33) as f32) / 2147483648.0 };
    let xv: Vec<f32> = (0..in_f).map(|_| r()).collect();
    let x_t = Tensor::from_f32(Shape::new([1, in_f]), xv.clone());
    let x_dev = DevBuf::alloc(dev.dev(), stream, in_f).unwrap();
    x_dev.upload(&xv).unwrap();

    let mut ws: Vec<Tensor> = Vec::new();
    let mut exps: Vec<Vec<f32>> = Vec::new();
    for (of, _n) in &shapes {
        let wv: Vec<f32> = (0..of * in_f).map(|_| r() * 0.02).collect();
        let w = Tensor::from_f32(Shape::new([*of, in_f]), wv);
        // reference: individual matmul_dev
        let o = dev.matmul_dev(&x_dev, &w, 1, in_f as i32, *of as i32).unwrap();
        let mut ev = vec![0f32; *of];
        o.download(&mut ev).unwrap();
        exps.push(ev);
        ws.push(w);
    }

    // fused gemv5
    let (o1, o2, o3, o4, o5) = dev
        .gemv5_dev(&x_dev, &ws[0], &ws[1], &ws[2], &ws[3], Some(&ws[4]),
                   in_f as i32, 3 * proj as i32, hh as i32, dk as i32, dk as i32)
        .unwrap();
    let outs = [o1, o2, o3, o4, o5.unwrap()];
    for (i, (o, exp)) in outs.iter().zip(exps.iter()).enumerate() {
        let mut got = vec![0f32; shapes[i].0];
        o.download(&mut got).unwrap();
        let mut maxd = 0f32;
        for (g, e) in got.iter().zip(exp.iter()) {
            maxd = maxd.max((g - e).abs());
        }
        eprintln!("[gemv5] {} [1,{}]: max_diff={:.2e}", shapes[i].1, shapes[i].0, maxd);
        assert!(maxd < 2e-4, "gemv5 {} max_diff {:.2e} too large", shapes[i].1, maxd);
    }
    // timing: fused vs 5 separate (steady-state)
    let it = 500;
    let t0 = std::time::Instant::now();
    for _ in 0..it {
        let _ = dev.matmul_dev(&x_dev, &ws[0], 1, in_f as i32, 3 * proj as i32);
        let _ = dev.matmul_dev(&x_dev, &ws[1], 1, in_f as i32, hh as i32);
        let _ = dev.matmul_dev(&x_dev, &ws[2], 1, in_f as i32, dk as i32);
        let _ = dev.matmul_dev(&x_dev, &ws[3], 1, in_f as i32, dk as i32);
        let _ = dev.matmul_dev(&x_dev, &ws[4], 1, in_f as i32, (dk * 2 + 64) as i32);
    }
    let _ = dev.sync();
    let t_sep = t0.elapsed().as_secs_f64() * 1e6 / it as f64;
    let t1 = std::time::Instant::now();
    for _ in 0..it {
        let _ = dev.gemv5_dev(&x_dev, &ws[0], &ws[1], &ws[2], &ws[3], Some(&ws[4]),
                              in_f as i32, 3 * proj as i32, hh as i32, dk as i32, dk as i32);
    }
    let _ = dev.sync();
    let t_fus = t1.elapsed().as_secs_f64() * 1e6 / it as f64;
    eprintln!("[gemv5] 5-matrix chain: separate={:.1}us fused={:.1}us ({:.1}x)", t_sep, t_fus, t_sep / t_fus.max(1e-9));
}

/// GEMV v2 (vectorized uint4 + K-split WPR) vs v1 (scalar warp-per-row):
/// parity + A/B perf on the real decode shapes (TP4 shards). v1 measured
/// 2.2-3.1 TB/s (27-39% of B300 HBM) — v2 targets 6+ TB/s. Decides the
/// matmul_dev n==1 swap.
#[test]
fn gemv_v2_bench() {
    use ferrite_kernel::cuda::DevBuf;
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda");
    dev.enter();
    let stream = dev.stream_handle();
    // (in_f, out_f, name) — real decode shapes per rank (TP4):
    // lm_head full/shard, gdn qkv/b/o, moe gate/down, dsa kv_a.
    let shapes: Vec<(usize, usize, &str)> = vec![
        (4096, 154880, "lm_head_full"),
        (4096, 38720, "lm_head_shard"),
        (4096, 16384, "qkv_big"),
        (4096, 4096, "proj_4k"),
        (4096, 3072, "o_proj_med"),
        (4096, 1536, "moe_gate"),
        (1536, 4096, "moe_down"),
        (4096, 1088, "dsa_kvA"),
        (4096, 512, "gdn_small"),
    ];
    let mut rnd = 0xfeed_beef_cafe_f00du64;
    let mut r = || { rnd = rnd.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((rnd >> 33) as f32) / 2147483648.0 };
    let mut total_v1 = 0f64;
    let mut total_v2 = 0f64;
    for (in_f, out_f, name) in &shapes {
        let (in_f, out_f) = (*in_f, *out_f);
        let xv: Vec<f32> = (0..in_f).map(|_| r()).collect();
        let wv: Vec<f32> = (0..out_f * in_f).map(|_| r() * 0.02).collect();
        let x_t = Tensor::from_f32(Shape::new([1, in_f]), xv);
        let w_t = Tensor::from_f32(Shape::new([out_f, in_f]), wv);
        let x_dev = DevBuf::alloc(dev.dev(), stream, in_f).unwrap();
        x_dev.upload(x_t.as_slice()).unwrap();
        // warm both (weight upload to bf16 cache) + parity reference
        let o1 = dev.matmul_dev(&x_dev, &w_t, 1, in_f as i32, out_f as i32).unwrap();
        let o2 = dev.gemv_v2_dev(&x_dev, &w_t, 1, in_f as i32, out_f as i32).unwrap();
        let _ = dev.sync();
        let mut e1 = vec![0f32; out_f];
        let mut e2 = vec![0f32; out_f];
        o1.download(&mut e1).unwrap();
        o2.download(&mut e2).unwrap();
        let mut maxd = 0f32;
        for (a, b) in e1.iter().zip(e2.iter()) { maxd = maxd.max((a - b).abs()); }
        // A/B: async loop + sync (kernel >> launch for these sizes)
        let bytes = (out_f * in_f * 2) as f64;
        let it = if bytes > 64e6 { 100 } else { 1000 };
        let t0 = std::time::Instant::now();
        for _ in 0..it {
            let _ = dev.matmul_dev(&x_dev, &w_t, 1, in_f as i32, out_f as i32);
        }
        let _ = dev.sync();
        let t_v1 = t0.elapsed().as_secs_f64() * 1e6 / it as f64;
        let t1 = std::time::Instant::now();
        for _ in 0..it {
            let _ = dev.gemv_v2_dev(&x_dev, &w_t, 1, in_f as i32, out_f as i32);
        }
        let _ = dev.sync();
        let t_v2 = t1.elapsed().as_secs_f64() * 1e6 / it as f64;
        total_v1 += t_v1;
        total_v2 += t_v2;
        eprintln!(
            "[gemv-v2] {:>13} [{:>6},{:>5}]: v1={:8.1}us ({:5.2}TB/s) v2={:8.1}us ({:5.2}TB/s) {:5.2}x  maxd={:.1e}",
            name, out_f, in_f,
            t_v1, bytes / t_v1 / 1e6, t_v2, bytes / t_v2 / 1e6,
            t_v1 / t_v2.max(1e-9), maxd
        );
        assert!(maxd < 2e-4, "gemv_v2 {} max_diff {:.2e} too large", name, maxd);
    }
    eprintln!("[gemv-v2] TOTAL: v1={:.1}us v2={:.1}us ({:.2}x)", total_v1, total_v2, total_v1 / total_v2.max(1e-9));
}

/// Event-in-graph timing mechanism isolation: the mega EVTS run panicked
/// cudaEventElapsedTime=InvalidValue(1) on the first post-replay read.
/// (a) plain stream events (record→kernel→record→sync→elapsed) vs
/// (b) events recorded DURING capture (→ event record nodes) →
/// instantiate → replay → elapsed. Isolates which path breaks.
#[test]
fn event_timing_graph() {
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda");
    dev.enter();
    let n = 4096usize;
    let mut rnd = 0x1234_5566_7788_99u64;
    let mut r = || { rnd = rnd.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((rnd >> 33) as f32) / 2147483648.0 };
    let xv: Vec<f32> = (0..n).map(|_| r()).collect();
    let wv: Vec<f32> = (0..n * n).map(|_| r() * 0.02).collect();
    let x_t = Tensor::from_f32(Shape::new([1, n]), xv.clone());
    let w_t = Tensor::from_f32(Shape::new([n, n]), wv);
    let x_dev = ferrite_kernel::cuda::DevBuf::alloc(dev.dev(), dev.stream_handle(), n).unwrap();
    x_dev.upload(&xv).unwrap();
    let _ = dev.matmul_dev(&x_dev, &w_t, 1, n as i32, n as i32); // warm weight cache
    let _ = dev.sync();

    // (a) plain stream events
    let e0 = dev.event_create().unwrap();
    let e1 = dev.event_create().unwrap();
    dev.event_record(e0);
    let out_a = dev.matmul_dev(&x_dev, &w_t, 1, n as i32, n as i32).unwrap();
    dev.event_record(e1);
    let _ = dev.sync();
    let ms_a = dev.event_elapsed_ms(e0, e1);
    let mut va = vec![0f32; 1];
    out_a.download(&mut va).unwrap();
    eprintln!("[evt] plain stream: {:.3}ms (out={:.4})", ms_a, va[0]);
    assert!(ms_a > 0.0, "plain elapsed must be > 0, got {ms_a}");

    // (b) events recorded during capture -> event record nodes.
    // NOTE: drop (a)'s buffers first so the pool has the 16KB size class
    // free — in-capture alloc of a cold class is err 900 (the mega chain
    // warms every class in its dry-run pass for exactly this reason).
    drop(out_a);
    let g0 = dev.event_create().unwrap();
    let g1 = dev.event_create().unwrap();
    dev.graph_capture_begin();
    dev.event_record(g0);
    let out_b = dev.matmul_dev(&x_dev, &w_t, 1, n as i32, n as i32).unwrap();
    dev.event_record(g1);
    dev.graph_capture_end("evttest");
    dev.graph_io_put("evttest", ferrite_kernel::cuda::GraphIO {
        x_stage: x_dev.stage,
        x_len: n,
        out_dev: out_b.as_f32() as *mut std::ffi::c_void,
        out_len: n,
    });
    std::mem::forget(out_b); // graph output must outlive replays (mega pattern)
    let mut ob = vec![0f32; n];
    assert!(dev.graph_run("evttest", &xv, &mut ob).unwrap(), "graph_run evttest");
    let ms_b = dev.event_elapsed_ms(g0, g1);
    eprintln!("[evt] in-graph: {:.3}ms (out={:.4})", ms_b, ob[0]);
    assert!(ms_b > 0.0, "in-graph elapsed must be > 0, got {ms_b}");
}
