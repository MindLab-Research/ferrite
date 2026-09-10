//! Layer-0 attention-projection parity against the reference.
//!
//! Same idea as `hc_parity`: run the GPU path on a deterministic input and print
//! the same intermediates the reference's numpy replica prints
//! (`/tmp/refcmp/attn_ref.py`), so the two can be diffed value by value.
//!
//! Covers, in one shot: the fp8 e4m3 weight decode + ue8m0 block scales, the
//! activation quantiser, the wq_a -> q_norm -> wq_b chain, and the wkv path.
//!
//!   DSV41_MODEL_DIR=... DSV41_KERNELS=... CUDA_VISIBLE_DEVICES=0 \
//!     cargo test --release -p ferrite-dsv41 --test attn_parity -- --nocapture

use ferrite_dsv41::config::Dsv41Config;
use ferrite_dsv41::device::Device;
use ferrite_dsv41::load::Loader;
use ferrite_dsv41::ops;
use ferrite_dsv41::weights::{Shard, TensorSpec};

fn env(n: &str) -> Option<String> {
    std::env::var(n).ok().filter(|v| !v.is_empty())
}

#[test]
fn layer0_attention_projections_match_reference() {
    let (Some(dir), Some(so)) = (env("DSV41_MODEL_DIR"), env("DSV41_KERNELS")) else {
        eprintln!("[attn_parity] skipped (set DSV41_MODEL_DIR / DSV41_KERNELS)");
        return;
    };
    let cfg = Dsv41Config::production();
    let dim = cfg.dim;
    let dev = Device::open(&so).expect("device");
    let mut loader = Loader::new(std::path::Path::new(&dir), &dev).expect("loader");
    let mut load = |n: &str, sh: Shard| {
        loader
            .load_single(
                &TensorSpec {
                    name: n.to_string(),
                    shape: vec![],
                    shard: sh,
                },
                1,
                0,
            )
            .unwrap()
    };
    let wq_a = load("layers.0.attn.wq_a.weight", Shard::Replicated);
    let wq_a_s = load("layers.0.attn.wq_a.scale", Shard::Replicated);
    let q_norm = load("layers.0.attn.q_norm.weight", Shard::Replicated);
    let wq_b = load("layers.0.attn.wq_b.weight", Shard::Replicated);
    let wq_b_s = load("layers.0.attn.wq_b.scale", Shard::Replicated);
    let wkv = load("layers.0.attn.wkv.weight", Shard::Replicated);
    let wkv_s = load("layers.0.attn.wkv.scale", Shard::Replicated);
    let kv_norm = load("layers.0.attn.kv_norm.weight", Shard::Replicated);

    // the same deterministic input the numpy replica uses
    let x: Vec<f32> = (0..dim).map(|i| (i % 97) as f32 * 0.01 - 0.5).collect();
    let dx = dev.upload_f32(&x).unwrap();
    let xq = dev.alloc(dim).unwrap();
    let xsc = dev.alloc((dim / 32 + 8) * 4).unwrap();
    let scratch = |n: usize| dev.alloc(n * 4).unwrap();
    let ql = cfg.q_lora_rank;
    let qr = scratch(ql);
    let q = scratch(cfg.n_heads * cfg.head_dim);
    let kv = scratch(cfg.head_dim);

    // first GEMM's activation quantisation, shared by the wq_a and wkv paths
    dev.quant_fp8(dx.as_f32(), xq.ptr as *mut u8, xsc.ptr as *mut f32, 1, dim as i32, 32, true)
        .unwrap();
    dev.gemm_fp8_mx(
        xq.as_u8(), xsc.as_f32(), wq_a.as_u8(), wq_a_s.as_u8(), std::ptr::null(),
        qr.ptr as *mut f32, 1, ql as i32, dim as i32,
    )
    .unwrap();
    dev.gemm_fp8_mx(
        xq.as_u8(), xsc.as_f32(), wkv.as_u8(), wkv_s.as_u8(), std::ptr::null(),
        kv.ptr as *mut f32, 1, cfg.head_dim as i32, dim as i32,
    )
    .unwrap();
    // q_norm then the second projection
    dev.rmsnorm(
        qr.ptr as *const f32, q_norm.as_f32(), qr.ptr as *mut f32,
        1, ql as i32, cfg.norm_eps,
    )
    .unwrap();
    dev.quant_fp8(qr.as_f32(), xq.ptr as *mut u8, xsc.ptr as *mut f32, 1, ql as i32, 32, true)
        .unwrap();
    dev.gemm_fp8_mx(
        xq.as_u8(), xsc.as_f32(), wq_b.as_u8(), wq_b_s.as_u8(), std::ptr::null(),
        q.ptr as *mut f32, 1, (cfg.n_heads * cfg.head_dim) as i32, ql as i32,
    )
    .unwrap();
    // kv_norm
    dev.rmsnorm(
        kv.ptr as *const f32, kv_norm.as_f32(), kv.ptr as *mut f32,
        1, cfg.head_dim as i32, cfg.norm_eps,
    )
    .unwrap();
    dev.sync().unwrap();

    let mut qrv = vec![0f32; ql];
    let mut qv = vec![0f32; cfg.n_heads * cfg.head_dim];
    let mut kvv = vec![0f32; cfg.head_dim];
    dev.download_f32(&qr, &mut qrv).unwrap();
    dev.download_f32(&q, &mut qv).unwrap();
    dev.download_f32(&kv, &mut kvv).unwrap();
    let rms = (qrv.iter().map(|v| v * v).sum::<f32>() / ql as f32).sqrt();
    eprintln!("[attn_parity] qr[:4]  {:?}  rms {rms}", &qrv[..4]);
    let amean = qv.iter().map(|v| v.abs()).sum::<f32>() / qv.len() as f32;
    eprintln!("[attn_parity] q[:4]   {:?}  |q|mean {amean}", &qv[..4]);
    eprintln!("[attn_parity] kv[:4]  {:?}", &kvv[..4]);
    let _ = ops::window_topk_idxs(cfg.window_size, 1, 1, 0);
}
