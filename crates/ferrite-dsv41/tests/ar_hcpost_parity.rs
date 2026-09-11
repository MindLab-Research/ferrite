//! Bit-exactness gate for the segment-C AR fold (`ferrite_p2p_ar_v5_hcpost`,
//! `DSV41_HCPOST_EPI=1`):
//!
//!   plain  : `all_reduce_inplace(buf)` + `hc_post_inplace(res, buf, ...)`
//!   fused  : `all_reduce_inplace_hcpost(buf, res, ...)`  (one launch)
//!
//! must agree BIT FOR BIT on BOTH outputs — the residual stream `res` it writes
//! in place and the AR output `buf` it still produces. This is the gate the
//! persistent roadmap (P1) requires before the fold may be flipped on: a single
//! ULP anywhere in the chain is a deterministic text change after 40 layers.
//!
//! The two paths live in different translation units (`dsv41_kernels.cu` vs
//! `ferrite_kernels.cu`), so the point of the test is to catch a compiler that
//! contracted/reassociated the epilogue differently despite the explicit
//! `__fmul_rn` / `__fmaf_rn` pinning.
//!
//!   DSV41_KERNELS=$PWD/kernels/cuda/libferrite_kernels.so \
//!   CUDA_VISIBLE_DEVICES=0,1,2,3 cargo test --release -p ferrite-dsv41 \
//!     --test ar_hcpost_parity -- --nocapture
//!
//! Env: AR_HCPOST_WORLD (default 4), AR_HCPOST_ROUNDS (default 4),
//!      AR_HCPOST_H (default 5120), AR_HCPOST_NHC (default 4).

use ferrite_dsv41::device::Device;
use ferrite_dsv41::tp::{Collective, SpinBarrier};
use std::sync::{Arc, Mutex};

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn det(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.37 + seed).sin() * 0.5).collect()
}

#[test]
fn fused_ar_hcpost_matches_the_two_launch_pair() {
    let so = std::env::var("DSV41_KERNELS")
        .unwrap_or_else(|_| "kernels/cuda/libferrite_kernels.so".to_string());
    let world = env_usize("AR_HCPOST_WORLD", 4);
    let rounds = env_usize("AR_HCPOST_ROUNDS", 4);
    let h = env_usize("AR_HCPOST_H", 5120);
    let nhc = env_usize("AR_HCPOST_NHC", 4);
    if world < 2 {
        eprintln!("[ar_hcpost] world={world} < 2: nothing to reduce, skipping");
        return;
    }
    assert_eq!(h % 4, 0, "the fold requires h % 4 == 0 (float4 path)");
    assert!((1..=8).contains(&nhc), "the fold requires 1 <= nhc <= 8");

    let barrier = Arc::new(SpinBarrier::new(world));
    let bases = Arc::new(Mutex::new(vec![0u64; world]));
    let failures = Arc::new(Mutex::new(Vec::<String>::new()));

    std::thread::scope(|sc| {
        for rank in 0..world {
            let barrier = barrier.clone();
            let bases = bases.clone();
            let failures = failures.clone();
            let so = so.clone();
            sc.spawn(move || {
                Device::bind_to(rank as i32).expect("bind");
                let dev = Arc::new(Device::open(&so).expect("open"));
                barrier.wait();
                dev.enable_peer_access().expect("peer access");
                barrier.wait();
                let mut c = Collective::new(dev.clone(), world, rank, h * 4, barrier.clone())
                    .expect("collective");
                bases.lock().unwrap()[rank] = c.staging_base();
                barrier.wait();
                let bs = bases.lock().unwrap().clone();
                barrier.wait();
                c.set_peers(bs).expect("set_peers");

                // The hyper-connection coefficients are NOT all-reduced: every
                // rank runs the same post/comb, so both paths see identical ones.
                let post = det(nhc, 2.3);
                let comb = det(nhc * nhc, 3.9);
                let d_post = dev.upload_f32(&post).unwrap();
                let d_comb = dev.upload_f32(&comb).unwrap();

                let res_a = dev.alloc(nhc * h * 4).unwrap();
                let res_b = dev.alloc(nhc * h * 4).unwrap();
                let buf_a = dev.alloc(h * 4).unwrap();
                let buf_b = dev.alloc(h * 4).unwrap();

                let mut ra = vec![0f32; nhc * h];
                let mut rb = vec![0f32; nhc * h];
                let mut ba = vec![0f32; h];
                let mut bb = vec![0f32; h];

                for r in 0..rounds {
                    // rank- and round-identifying payloads for the AR, and a
                    // residual that actually feeds the mix (non-trivial comb).
                    let seed = r as f32 * 17.0 + rank as f32 * 1000.0;
                    let res_h = det(nhc * h, seed + 0.11);
                    let buf_h = det(h, seed + 1.7);
                    dev.upload_f32_at(res_a.ptr, 0, &res_h).unwrap();
                    dev.upload_f32_at(res_b.ptr, 0, &res_h).unwrap();
                    dev.upload_f32_at(buf_a.ptr, 0, &buf_h).unwrap();
                    dev.upload_f32_at(buf_b.ptr, 0, &buf_h).unwrap();

                    // ---- Path A: the two-launch pair (the shipping path) ----
                    if let Err(e) = c.all_reduce_inplace(buf_a.ptr as *mut std::ffi::c_void, h * 4) {
                        failures.lock().unwrap().push(format!("rank {rank} r{r}: A ar: {e}"));
                        barrier.wait();
                        continue;
                    }
                    if let Err(e) = dev.hc_post_inplace(
                        res_a.as_f32() as *mut f32,
                        buf_a.as_f32(),
                        d_post.as_f32(),
                        d_comb.as_f32(),
                        nhc as i32,
                        h as i32,
                    ) {
                        failures.lock().unwrap().push(format!("rank {rank} r{r}: A hc_post: {e}"));
                        barrier.wait();
                        continue;
                    }

                    // ---- Path B: the fused entry (carries its own store) ----
                    match c.all_reduce_inplace_hcpost(
                        buf_b.ptr as *mut std::ffi::c_void,
                        h * 4,
                        res_b.as_f32() as *mut f32,
                        d_post.as_f32(),
                        d_comb.as_f32(),
                        nhc as i32,
                        h as i32,
                    ) {
                        Ok(true) => {}
                        Ok(false) => {
                            failures.lock().unwrap().push(format!(
                                "rank {rank} r{r}: fused entry declined (stale .so / bad shape?)"
                            ));
                            barrier.wait();
                            continue;
                        }
                        Err(e) => {
                            failures.lock().unwrap().push(format!("rank {rank} r{r}: B ar: {e}"));
                            barrier.wait();
                            continue;
                        }
                    }

                    if let Err(e) = dev.sync() {
                        failures.lock().unwrap().push(format!("rank {rank} r{r}: sync: {e}"));
                        barrier.wait();
                        continue;
                    }
                    let va = Device::view(res_a.ptr, nhc * h * 4);
                    let vb = Device::view(res_b.ptr, nhc * h * 4);
                    dev.download_f32(&va, &mut ra).expect("download res_a");
                    dev.download_f32(&vb, &mut rb).expect("download res_b");
                    let ua = Device::view(buf_a.ptr, h * 4);
                    let ub = Device::view(buf_b.ptr, h * 4);
                    dev.download_f32(&ua, &mut ba).expect("download buf_a");
                    dev.download_f32(&ub, &mut bb).expect("download buf_b");

                    for (i, (x, y)) in ra.iter().zip(rb.iter()).enumerate() {
                        if x.to_bits() != y.to_bits() {
                            failures.lock().unwrap().push(format!(
                                "rank {rank} r{r}: residual[{i}] differs: two-launch {:e} vs fused {:e}",
                                x, y
                            ));
                            break;
                        }
                    }
                    for (i, (x, y)) in ba.iter().zip(bb.iter()).enumerate() {
                        if x.to_bits() != y.to_bits() {
                            failures.lock().unwrap().push(format!(
                                "rank {rank} r{r}: AR out[{i}] differs: two-launch {:e} vs fused {:e}",
                                x, y
                            ));
                            break;
                        }
                    }
                    barrier.wait();
                }
                if rank == 0 {
                    eprintln!(
                        "[ar_hcpost] rank0 completed {rounds} rounds, h={h} nhc={nhc}, world={world}"
                    );
                }
                barrier.wait();
            });
        }
    });

    let f = failures.lock().unwrap();
    if !f.is_empty() {
        for m in f.iter().take(10) {
            eprintln!("[ar_hcpost] FAIL {m}");
        }
        panic!(
            "the fused AR + hc_post is not bit-identical to the two-launch pair ({} reports)",
            f.len()
        );
    }
    eprintln!("[ar_hcpost] OK: the fold is bit-identical to all_reduce_inplace + hc_post_inplace");
}
