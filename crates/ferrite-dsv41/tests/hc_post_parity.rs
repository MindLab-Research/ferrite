//! Bit-exactness check for the fused segment C: `hc_post` into a staging buffer
//! followed by a device copy must equal `hc_post_inplace` writing straight back
//! onto the residual, element for element.
//!
//! The serve-level check can only look at generated text, where a single
//! near-boundary logit flip shows up as a different preamble and where the engine
//! has other sources of variation; this runs the two paths on the same
//! deterministic input on one device and compares every float.
//!
//!   DSV41_KERNELS=$PWD/kernels/cuda/libferrite_kernels.so CUDA_VISIBLE_DEVICES=0 \
//!     cargo test --release -p ferrite-dsv41 --test hc_post_parity -- --nocapture

use ferrite_dsv41::device::Device;

fn env(n: &str) -> Option<String> {
    std::env::var(n).ok().filter(|v| !v.is_empty())
}

fn det(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.37 + seed).sin() * 0.5).collect()
}

#[test]
fn inplace_matches_staged_hc_post() {
    let Some(so) = env("DSV41_KERNELS") else {
        eprintln!("[hc_post_parity] skipped (set DSV41_KERNELS)");
        return;
    };
    let dev = Device::open(&so).expect("device");

    // The model's real shape (hc = 4, dim = 5120) plus a non-multiple-of-256
    // width, so the kernel's tail handling is exercised too.
    for (n, h) in [(4usize, 5120usize), (4, 1024), (2, 64)] {
        let res = det(n * h, 0.11);
        let x = det(h, 1.7);
        let post = det(n, 2.3);
        let comb = det(n * n, 3.9);

        // Path 1: the original shape - residual in, staging buffer out, then copy back.
        let d_res1 = dev.upload_f32(&res).unwrap();
        let d_x = dev.upload_f32(&x).unwrap();
        let d_post = dev.upload_f32(&post).unwrap();
        let d_comb = dev.upload_f32(&comb).unwrap();
        let d_stage = dev.alloc(n * h * 4).unwrap();
        dev.hc_post(
            d_x.as_f32(),
            d_res1.as_f32(),
            d_post.as_f32(),
            d_comb.as_f32(),
            d_stage.as_f32() as *mut f32,
            1,
            n as i32,
            h as i32,
        )
        .expect("hc_post");
        dev.memcpy_d2d(
            d_res1.ptr,
            d_stage.ptr as *const std::ffi::c_void,
            n * h * 4,
        )
        .expect("copy back");
        let mut got1 = vec![0f32; n * h];
        dev.download_f32(&d_res1, &mut got1).expect("download 1");

        // Path 2: the fused path - residual read and written in place.
        let d_res2 = dev.upload_f32(&res).unwrap();
        dev.hc_post_inplace(
            d_res2.as_f32() as *mut f32,
            d_x.as_f32(),
            d_post.as_f32(),
            d_comb.as_f32(),
            n as i32,
            h as i32,
        )
        .expect("hc_post_inplace");
        let mut got2 = vec![0f32; n * h];
        dev.download_f32(&d_res2, &mut got2).expect("download 2");

        let mut diff = 0usize;
        let mut worst = 0f32;
        for (a, b) in got1.iter().zip(got2.iter()) {
            if a.to_bits() != b.to_bits() {
                diff += 1;
                worst = worst.max((a - b).abs());
            }
        }
        println!(
            "[hc_post_parity] n={n} h={h}: differing={diff}/{} worst={worst:e}",
            n * h
        );
        assert_eq!(
            diff, 0,
            "in-place hc_post is not bit-identical for n={n} h={h} (worst |delta| {worst:e})"
        );
    }
}
