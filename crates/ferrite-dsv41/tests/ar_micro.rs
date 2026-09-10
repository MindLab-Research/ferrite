//! Isolated all-reduce harness.
//!
//! Debugging the collective on the full model is hopeless: each attempt costs a
//! ~45s weight load plus a decode, and a deadlock shows up only as a timeout.
//! This drives `all_reduce_inplace` alone, in a loop, and checks every round
//! against the host-side reference (each rank's buffer must become the
//! elementwise sum of all ranks' inputs). Run it with DSV41_AR_DEV=1 to exercise
//! the opt-in device-side protocol.
//!
//!   DSV41_KERNELS=$PWD/kernels/cuda/libferrite_kernels.so \
//!   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 cargo test --release -p ferrite-dsv41 \
//!     --test ar_micro -- --nocapture
//!
//! Env: AR_MICRO_WORLD (default 8), AR_MICRO_ROUNDS (default 32), AR_MICRO_N
//! (floats per rank, default 1024).

use ferrite_dsv41::device::Device;
use ferrite_dsv41::tp::{Collective, SpinBarrier};
use std::sync::{Arc, Mutex};

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

#[test]
fn collective_all_reduce_matches_host_reference() {
    let so = std::env::var("DSV41_KERNELS").unwrap_or_else(|_| {
        "kernels/cuda/libferrite_kernels.so".to_string()
    });
    let world = env_usize("AR_MICRO_WORLD", 8);
    let rounds = env_usize("AR_MICRO_ROUNDS", 32);
    let n = env_usize("AR_MICRO_N", 1024);
    if world < 2 {
        eprintln!("[ar_micro] world={world} < 2: nothing to reduce, skipping");
        return;
    }

    let barrier = Arc::new(SpinBarrier::new(world));
    let staging = Arc::new(Mutex::new(vec![0u64; world]));
    let failures = Arc::new(Mutex::new(Vec::<String>::new()));

    std::thread::scope(|sc| {
        for rank in 0..world {
            let barrier = barrier.clone();
            let staging = staging.clone();
            let failures = failures.clone();
            let so = so.clone();
            sc.spawn(move || {
                Device::bind_to(rank as i32).expect("bind");
                let dev = Arc::new(Device::open(&so).expect("open"));

                // peer access needs every rank's context to exist first
                barrier.wait();
                let peers = dev.enable_peer_access().expect("peer access");
                barrier.wait();

                let mut c = Collective::new(dev.clone(), world, rank, n * 4, barrier.clone())
                    .expect("collective");
                staging.lock().unwrap()[rank] = c.staging_base();
                barrier.wait();
                let bases = staging.lock().unwrap().clone();
                barrier.wait();
                c.set_peers(bases).expect("set_peers");

                // this rank's buffer: a value that identifies rank and round
                let buf = dev.alloc(n * 4).expect("buf");
                let mut host = vec![0f32; n];
                let mut got = vec![0f32; n];

                for r in 0..rounds {
                    let v = r as f32 + rank as f32 * 1000.0;
                    for x in host.iter_mut() {
                        *x = v;
                    }
                    dev.upload_f32_at(buf.ptr, 0, &host).expect("upload");
                    if let Err(e) = c.all_reduce_inplace(buf.ptr as *mut std::ffi::c_void, n * 4) {
                        failures
                            .lock()
                            .unwrap()
                            .push(format!("rank {rank} round {r}: all_reduce: {e}"));
                        // keep participating in the harness barrier: returning here
                        // would leave the other ranks waiting forever and look like
                        // a protocol hang instead of reporting the real error
                        barrier.wait();
                        continue;
                    }
                    let d = ferrite_dsv41::device::Device::view(buf.ptr, n * 4);
                    dev.download_f32(&d, &mut got).expect("download");
                    // expected: the ELEMENTWISE sum over ranks of (r + rank*1000).
                    // Every element holds the same value, so the per-element sum
                    // is that value summed over the ranks — no `n` factor.
                    let expect = r as f32 * world as f32
                        + 1000.0 * (world * (world - 1) / 2) as f32;
                    if got[0] != expect || got[n - 1] != expect {
                        failures.lock().unwrap().push(format!(
                            "rank {rank} round {r}: got {:.1} (first) / {:.1} (last), expected {:.1}",
                            got[0], got[n - 1], expect
                        ));
                        barrier.wait();
                        continue;
                    }
                    // the buffers must keep advancing (a stalled round would repeat)
                    barrier.wait();
                }
                if rank == 0 {
                    eprintln!("[ar_micro] rank0 completed {rounds} rounds of {n} floats");
                }
                barrier.wait();
            });
        }
    });

    let f = failures.lock().unwrap();
    if !f.is_empty() {
        for m in f.iter().take(10) {
            eprintln!("[ar_micro] FAIL {m}");
        }
        panic!("collective all-reduce mismatched the host reference ({} reports)", f.len());
    }
    eprintln!("[ar_micro] OK: world={world} rounds={rounds} n={n}");
}
