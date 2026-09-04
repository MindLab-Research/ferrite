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
    close(&o_gpu, &o_cpu, 5e-3, "matmul"); // bf16-resident weights (f32 truncation ~2^-8 rel)

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
