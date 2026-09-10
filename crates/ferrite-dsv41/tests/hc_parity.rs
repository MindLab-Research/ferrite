//! Layer-0 hyper-connection comparison against the reference implementation.
//!
//! Runs `dsv41_hc_mixes` on a deterministic input and prints pre/post/comb so
//! they can be diffed against the reference's numbers
//! (`/tmp/refcmp/hc_ref.py`, computed straight from the checkpoint with numpy).
//!
//!   DSV41_MODEL_DIR=... DSV41_KERNELS=... CUDA_VISIBLE_DEVICES=0 \
//!     cargo test --release -p ferrite-dsv41 --test hc_parity -- --nocapture

use ferrite_dsv41::config::Dsv41Config;
use ferrite_dsv41::device::Device;
use ferrite_dsv41::load::Loader;
use ferrite_dsv41::weights::{Shard, TensorSpec};

fn env(n: &str) -> Option<String> {
    std::env::var(n).ok().filter(|v| !v.is_empty())
}

#[test]
fn hc_mixes_matches_reference() {
    let (Some(dir), Some(so)) = (env("DSV41_MODEL_DIR"), env("DSV41_KERNELS")) else {
        eprintln!("[hc_parity] skipped (set DSV41_MODEL_DIR / DSV41_KERNELS)");
        return;
    };
    let cfg = Dsv41Config::production();
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    let dev = Device::open(&so).expect("device");
    let mut loader = Loader::new(std::path::Path::new(&dir), &dev).expect("loader");
    let load = |l: &mut Loader, n: &str, shape: Vec<usize>, sh: Shard| {
        l.load_single(
            &TensorSpec {
                name: n.to_string(),
                shape,
                shard: sh,
            },
            1,
            0,
        )
        .unwrap()
    };
    let mix_hc = (2 + hc) * hc;
    let fn_w = load(&mut loader, "layers.0.hc_attn_fn", vec![mix_hc, hc * dim], Shard::Replicated);
    let base = load(&mut loader, "layers.0.hc_attn_base", vec![mix_hc], Shard::Replicated);
    let scale = load(&mut loader, "layers.0.hc_attn_scale", vec![3], Shard::Replicated);

    // the SAME deterministic input the reference script used
    let xs: Vec<f32> = (0..hc * dim)
        .map(|i| (i % 97) as f32 * 0.01 - 0.5)
        .collect();
    let dx = dev.upload_f32(&xs).unwrap();
    let pre = dev.alloc(hc * 4).unwrap();
    let post = dev.alloc(hc * 4).unwrap();
    let comb = dev.alloc(hc * hc * 4).unwrap();

    dev.hc_mixes(
        dx.as_f32(),
        fn_w.as_f32(),
        scale.as_f32(),
        base.as_f32(),
        pre.ptr as *mut f32,
        post.ptr as *mut f32,
        comb.ptr as *mut f32,
        1,
        (hc * dim) as i32,
        hc as i32,
        cfg.hc_sinkhorn_iters as i32,
        cfg.hc_eps,
    )
    .expect("hc_mixes");
    dev.sync().unwrap();
    let mut p = vec![0f32; hc];
    let mut q = vec![0f32; hc];
    let mut c = vec![0f32; hc * hc];
    dev.download_f32(&pre, &mut p).unwrap();
    dev.download_f32(&post, &mut q).unwrap();
    dev.download_f32(&comb, &mut c).unwrap();
    eprintln!("[hc_parity] PRE  {p:?}");
    eprintln!("[hc_parity] POST {q:?}");
    eprintln!("[hc_parity] COMB row0 {:?}", &c[..hc]);
    let rs: f32 = c[..hc].iter().sum();
    let cs: f32 = (0..hc).map(|k| c[k * hc]).sum();
    eprintln!("[hc_parity] row0 sum {rs:.7} col0 sum {cs:.7}");
}
