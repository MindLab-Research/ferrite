//! NCCL-in-CUDA-graph smoke test — the mega-graph mechanism validator.
//! Run (b300-4): NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=4,5,6,7
//!   cargo test --release -p ferrite-kernel --features cuda --test gpu_smoke_nccl_graph
//!   -- --nocapture --test-threads=1
//!
//! Thread model (matches the serve fan_out architecture): ONE thread per
//! rank/comm — NCCL collectives on MULTIPLE comms from a single thread
//! require ncclGroupStart/End grouping (deadlock otherwise). Per-rank
//! threads each own their comm, no grouping needed.
#![cfg(feature = "cuda")]

use ferrite_kernel::CudaBackend;
use ferrite_kernel::cuda::{capture_lock, DevBuf};
use ferrite_kernel::nccl::NcclGroup;

fn so_path() -> String {
    std::env::var("FERRITE_KERNEL_SO").unwrap_or_else(|_| {
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("../../kernels/cuda/libferrite_kernels.so");
        p.to_string_lossy().into_owned()
    })
}

/// Scoped threads need the (Send-unsafe) channel/backend refs to cross the
/// spawn boundary — same pattern as tp.rs's fan_out SendPtr.
struct SendPtr<T>(T);
unsafe impl<T> Send for SendPtr<T> {}

const WORLD: usize = 4;
const COUNT: usize = 4096; // decode n=1 × hidden — the real collective size

#[test]
fn nccl_in_graph_all_reduce() {
    let so = so_path();
    let mut backends = Vec::with_capacity(WORLD);
    for d in 0..WORLD as i32 {
        match CudaBackend::with_device(&so, d) {
            Ok(b) => backends.push(b),
            Err(e) => {
                eprintln!("nccl_in_graph: SKIP (device {d} unavailable: {e:?}) — needs 4 GPUs");
                return;
            }
        }
    }
    let streams: Vec<_> = backends.iter().map(|b| b.stream_handle()).collect();
    let devs: Vec<i32> = (0..WORLD as i32).collect();
    backends[0].enter(); // active context on the calling thread before ncclCommInitAll
    let chans = match NcclGroup::init_all(&devs, &streams) {
        Ok(c) => c,
        Err(e) => panic!("nccl init failed ({e:?}) — run with NCCL_NVLS_ENABLE=0 on b300-4"),
    };
    assert_eq!(chans.len(), WORLD);
    eprintln!("[nccl_in_graph] ncclCommInitAll OK ({WORLD} ranks)");

    // ---- warm-up: 2 in-place collectives per rank OUTSIDE capture ----
    // (allocates NCCL's internal buffers + warms the COUNT-size DevBuf pool
    // class on every device — capture forbids cudaMalloc). Per-rank threads
    // (one thread per comm — the serve fan_out model; single-thread
    // multi-comm enqueue needs ncclGroupStart/End and deadlocks otherwise).
    std::thread::scope(|s| {
        for r in 0..WORLD {
            let b = SendPtr(&backends[r]);
            let ch = SendPtr(&chans[r]);
            let v: Vec<f32> = (0..COUNT).map(|i| (r + i % 7) as f32).collect();
            s.spawn(move || {
                let SendPtr(b) = b;
                let SendPtr(ch) = ch;
                b.enter();
                let mut wx = DevBuf::alloc(b.dev(), b.stream_handle(), COUNT).unwrap();
                wx.upload(&v).unwrap();
                ch.all_reduce_f32(wx.as_const_f32(), wx.as_f32(), COUNT).unwrap();
                // in-place AR accumulates — re-upload the initial values so
                // the second (steady-state path) round starts from v again
                wx.upload(&v).unwrap();
                ch.all_reduce_f32(wx.as_const_f32(), wx.as_f32(), COUNT).unwrap();
                b.sync().unwrap();
                let mut got = vec![0f32; COUNT];
                wx.download(&mut got).unwrap();
                for (i, g) in got.iter().enumerate() {
                    assert!(
                        // sum over r of (r + i%7) = 6 + 4*(i%7)
                        (g - (6.0 + 4.0 * (i % 7) as f32)).abs() < 1e-2,
                        "warmup allreduce wrong r{r} i{i}: {g}"
                    );
                }
            });
        }
    });
    eprintln!("[nccl_in_graph] warm-up collectives synced + sums verified");

    // ---- capture: per-rank graph, K=1 collective (correctness) ----
    // One thread per rank; capture_lock serializes the cuGraphInstantiate
    // calls (concurrent instantiate SIGSEGV'd historically).
    let inputs: Vec<Vec<f32>> = (0..WORLD)
        .map(|r| (0..COUNT).map(|i| r as f32 * 0.25 + (i % 5) as f32 * 2.0).collect())
        .collect();
    let expect: Vec<f32> = (0..COUNT)
        .map(|i| (0..WORLD).map(|r| r as f32 * 0.25 + (i % 5) as f32 * 2.0).sum())
        .collect();
    let mut out_ptrs: Vec<usize> = vec![0; WORLD];
    std::thread::scope(|s| {
        for r in 0..WORLD {
            let b = SendPtr(&backends[r]);
            let ch = SendPtr(&chans[r]);
            let input = &inputs[r];
            // slot as usize — raw pointers passed as usize are always Send
            // (tp.rs fan_out_pooled pattern); RFC-2229 precise capture
            // would otherwise grab the *mut usize field (not Send).
            let slot = unsafe { out_ptrs.as_mut_ptr().add(r) } as usize;
            s.spawn(move || {
                let SendPtr(b) = b;
                let SendPtr(ch) = ch;
                b.enter();
                let _cap = capture_lock().lock().unwrap();
                let mut buf = DevBuf::alloc(b.dev(), b.stream_handle(), COUNT).unwrap();
                b.graph_capture_begin();
                buf.upload(input).unwrap(); // recorded stage→dev memcpy
                ch.all_reduce_f32(buf.as_const_f32(), buf.as_f32(), COUNT).unwrap();
                b.graph_capture_end("nccl_g1");
                unsafe { *(slot as *mut usize) = buf.as_f32() as usize };
                std::mem::forget(buf); // stable address across replays
            });
        }
    });
    eprintln!("[nccl_in_graph] K=1 graphs captured on all {WORLD} ranks (NCCL inside capture)");

    // capture records but does NOT execute — replay concurrently
    std::thread::scope(|s| {
        for r in 0..WORLD {
            let b = SendPtr(&backends[r]);
            s.spawn(move || {
                let SendPtr(b) = b;
                b.enter();
                assert!(b.graph_replay("nccl_g1"), "replay r{r}");
                b.sync().unwrap();
            });
        }
    });
    for r in 0..WORLD {
        let b = &backends[r];
        b.enter();
        let mut got = vec![0f32; COUNT];
        let rc = ferrite_kernel::cuda::memcpy_d2h_sync(
            out_ptrs[r] as *mut std::ffi::c_void,
            got.as_mut_ptr(),
            COUNT,
            b.stream_handle(),
        );
        assert_eq!(rc, 0, "d2h r{r}");
        for (i, g) in got.iter().enumerate() {
            assert!(
                (g - expect[i]).abs() < 1e-2,
                "in-graph allreduce wrong r{r} i{i}: {g} vs {}",
                expect[i]
            );
        }
    }
    eprintln!("[nccl_in_graph] K=1 replay: in-graph all-reduce sums OK on all ranks");

    // ---- repeated re-launches of the SAME graph (node re-execution) ----
    std::thread::scope(|s| {
        for r in 0..WORLD {
            let b = SendPtr(&backends[r]);
            s.spawn(move || {
                let SendPtr(b) = b;
                b.enter();
                for _ in 0..20 {
                    assert!(b.graph_replay("nccl_g1"), "replay r{r}");
                }
                b.sync().unwrap();
            });
        }
    });
    eprintln!("[nccl_in_graph] 20 re-launches OK (values x4^20 — magnitudes only)");

    // ---- timing: K=90 collectives per graph (45 layers × 2), R replays ----
    const K: usize = 90;
    const R: usize = 50;
    std::thread::scope(|s| {
        for r in 0..WORLD {
            let b = SendPtr(&backends[r]);
            let ch = SendPtr(&chans[r]);
            let input = &inputs[r];
            s.spawn(move || {
                let SendPtr(b) = b;
                let SendPtr(ch) = ch;
                b.enter();
                let _cap = capture_lock().lock().unwrap();
                let mut buf = DevBuf::alloc(b.dev(), b.stream_handle(), COUNT).unwrap();
                b.graph_capture_begin();
                buf.upload(input).unwrap();
                for _ in 0..K {
                    ch.all_reduce_f32(buf.as_const_f32(), buf.as_f32(), COUNT).unwrap();
                }
                b.graph_capture_end("nccl_g90");
                std::mem::forget(buf);
            });
        }
    });
    eprintln!("[nccl_in_graph] K=90 graphs captured");
    let t0 = std::time::Instant::now();
    std::thread::scope(|s| {
        for r in 0..WORLD {
            let b = SendPtr(&backends[r]);
            s.spawn(move || {
                let SendPtr(b) = b;
                b.enter();
                for _ in 0..R {
                    assert!(b.graph_replay("nccl_g90"), "replay90 r{r}");
                }
                b.sync().unwrap();
            });
        }
    });
    let dt = t0.elapsed();
    eprintln!(
        "[nccl_in_graph] K={K} collectives x R={R} replays x {WORLD} ranks: {:.3?} ({:.1}us/replay, {:.2}us/collective)",
        dt,
        dt.as_secs_f64() / R as f64 * 1e6,
        dt.as_secs_f64() / (R as f64 * K as f64) * 1e6,
    );
}

/// P2P one-shot AR micro-bench (TileRT ExpertDownAllReduce mode) vs NCCL:
/// 4 GPUs, each rank writes its partial into every rank's staging via UVA
/// peer writes in-kernel, sums local rows. Measures per-AR latency for the
/// decode collective size (n=1 x hidden=4096 f32).
#[test]
fn p2p_oneshot_vs_nccl() {
    let so = so_path();
    let mut backends = Vec::with_capacity(WORLD);
    for d in 0..WORLD as i32 {
        match CudaBackend::with_device(&so, d) {
            Ok(b) => backends.push(b),
            Err(e) => {
                eprintln!("p2p_oneshot: SKIP (device {d} unavailable: {e:?}) — needs 4 GPUs");
                return;
            }
        }
    }
    let streams: Vec<_> = backends.iter().map(|b| b.stream_handle()).collect();
    let devs: Vec<i32> = (0..WORLD as i32).collect();
    backends[0].enter();
    let chans = match NcclGroup::init_all(&devs, &streams) {
        Ok(c) => c,
        Err(e) => panic!("nccl init failed ({e:?}) — run with NCCL_NVLS_ENABLE=0"),
    };
    // peer access all pairs (UVA in-kernel writes)
    for i in 0..WORLD {
        for j in 0..WORLD {
            if i != j {
                backends[i].enter();
                backends[i].p2p_enable(j as i32).expect("p2p_enable");
            }
        }
    }
    // per-rank buffers: staging [WORLD][COUNT], ready [WORLD], ctr, partial, out
    fn u64_to_f32x2(v: u64) -> [f32; 2] {
        unsafe { std::mem::transmute(v) }
    }
    let mut staging = Vec::new();
    let mut ready = Vec::new();
    let mut ctrs = Vec::new();
    let mut partials = Vec::new();
    let mut outs = Vec::new();
    let mut stg_tbls = Vec::new();
    let mut rdy_tbls = Vec::new();
    for (r, b) in backends.iter().enumerate() {
        b.enter();
        staging.push(DevBuf::alloc(b.dev(), b.stream_handle(), WORLD * COUNT).unwrap());
        ready.push(DevBuf::alloc(b.dev(), b.stream_handle(), WORLD).unwrap());
        ready[r].upload(&vec![0f32; WORLD]).unwrap();
        ctrs.push(DevBuf::alloc(b.dev(), b.stream_handle(), 1).unwrap());
        ctrs[r].upload(&[0f32]).unwrap();
        partials.push(DevBuf::alloc(b.dev(), b.stream_handle(), COUNT).unwrap());
        partials[r].upload(&vec![r as f32 + 1.0; COUNT]).unwrap();
        outs.push(DevBuf::alloc(b.dev(), b.stream_handle(), COUNT).unwrap());
    }
    // ptr tables: for rank r, table[r'] = peer r' staging/ready base (UVA ptrs)
    let stg_ptrs: Vec<u64> = (0..WORLD).map(|r| staging[r].as_f32() as u64).collect();
    let rdy_ptrs: Vec<u64> = (0..WORLD).map(|r| ready[r].as_f32() as u64).collect();
    for b in backends.iter() {
        b.enter();
        let mut t1 = Vec::with_capacity(WORLD * 2);
        for &p in &stg_ptrs { t1.extend_from_slice(&u64_to_f32x2(p)); }
        let tbl = DevBuf::alloc(b.dev(), b.stream_handle(), WORLD * 2).unwrap();
        tbl.upload(&t1).unwrap();
        stg_tbls.push(tbl);
        let mut t2 = Vec::with_capacity(WORLD * 2);
        for &p in &rdy_ptrs { t2.extend_from_slice(&u64_to_f32x2(p)); }
        let tbl = DevBuf::alloc(b.dev(), b.stream_handle(), WORLD * 2).unwrap();
        tbl.upload(&t2).unwrap();
        rdy_tbls.push(tbl);
    }
    // correctness: one iteration → every rank's out == 1+2+3+4 = 10
    std::thread::scope(|s| {
        for r in 0..WORLD {
            let b = SendPtr(&backends[r]);
            let stg = SendPtr(&staging[r]);
            let rdy = SendPtr(&ready[r]);
            let ctr = SendPtr(&ctrs[r]);
            let par = SendPtr(&partials[r]);
            let slot = unsafe { outs.as_mut_ptr().add(r) } as usize; // raw ptr as usize (Send)
            let st = SendPtr(&stg_tbls[r]);
            let rt = SendPtr(&rdy_tbls[r]);
            s.spawn(move || {
                let SendPtr(b) = b;
                let (SendPtr(stg), SendPtr(rdy), SendPtr(ctr), SendPtr(par)) = (stg, rdy, ctr, par);
                let (SendPtr(st), SendPtr(rt)) = (st, rt);
                let out = unsafe { &mut *(slot as *mut DevBuf) };
                b.enter();
                b.p2p_ar_oneshot_dev(par, st, rt, ctr, stg, rdy, out,
                                     COUNT, WORLD, r).unwrap();
                let _ = b.sync();
            });
        }
    });
    {
        backends[0].enter();
        let mut v = vec![0f32; COUNT];
        outs[0].download(&mut v).unwrap();
        let ok = v.iter().all(|x| (x - 10.0).abs() < 1e-4);
        assert!(ok, "one-shot AR wrong: {:?}", &v[..4]);
        eprintln!("[p2p_oneshot] correctness OK (sum=10.0)");
    }
    // timing: one-shot loop vs NCCL loop (same per-iter barrier cadence)
    const N: usize = 1000;
    let t0 = std::time::Instant::now();
    let barrier = std::sync::Barrier::new(WORLD);
    std::thread::scope(|s| {
        let bar = &barrier;
        for r in 0..WORLD {
            let b = SendPtr(&backends[r]);
            let stg = SendPtr(&staging[r]);
            let rdy = SendPtr(&ready[r]);
            let ctr = SendPtr(&ctrs[r]);
            let par = SendPtr(&partials[r]);
            let slot = unsafe { outs.as_mut_ptr().add(r) } as usize;
            let st = SendPtr(&stg_tbls[r]);
            let rt = SendPtr(&rdy_tbls[r]);
            s.spawn(move || {
                let SendPtr(b) = b;
                let (SendPtr(stg), SendPtr(rdy), SendPtr(ctr), SendPtr(par)) = (stg, rdy, ctr, par);
                let (SendPtr(st), SendPtr(rt)) = (st, rt);
                let out = unsafe { &mut *(slot as *mut DevBuf) };
                b.enter();
                let zeros = vec![0f32; WORLD];
                for _ in 0..N {
                    b.p2p_ar_oneshot_dev(par, st, rt, ctr, stg, rdy, out,
                                         COUNT, WORLD, r).unwrap();
                    let _ = b.sync();
                    bar.wait();
                    rdy.upload(&zeros).unwrap(); // reset flags for next iter
                    let _ = b.sync();
                    bar.wait();
                }
            });
        }
    });
    let us_oneshot = t0.elapsed().as_secs_f64() * 1e6 / N as f64;
    // NCCL baseline: same loop shape
    let t1 = std::time::Instant::now();
    let barrier2 = std::sync::Barrier::new(WORLD);
    std::thread::scope(|s| {
        let bar = &barrier2;
        for r in 0..WORLD {
            let b = SendPtr(&backends[r]);
            let ch = SendPtr(&chans[r]);
            let par = SendPtr(&partials[r]);
            let slot = unsafe { outs.as_mut_ptr().add(r) } as usize;
            s.spawn(move || {
                let SendPtr(b) = b;
                let (SendPtr(ch), SendPtr(par)) = (ch, par);
                let out = unsafe { &mut *(slot as *mut DevBuf) };
                b.enter();
                for _ in 0..N {
                    ch.all_reduce_f32(par.as_const_f32(), out.as_f32(), COUNT).unwrap();
                    let _ = b.sync();
                    bar.wait();
                }
            });
        }
    });
    let us_nccl = t1.elapsed().as_secs_f64() * 1e6 / N as f64;
    eprintln!("[p2p_oneshot] one-shot {:.2}us vs NCCL {:.2}us per AR ({} iters, {} f32)",
              us_oneshot, us_nccl, N, COUNT);

    // ---- GRAPH-MODE: capture the down+sum pair per rank, measure replay
    // latency (the production path — NCCL-in-graph measures ~15us/AR).
    for b in backends.iter() { b.enter(); }
    std::thread::scope(|s| {
        for r in 0..WORLD {
            let b = SendPtr(&backends[r]);
            let stg = SendPtr(&staging[r]);
            let rdy = SendPtr(&ready[r]);
            let ctr = SendPtr(&ctrs[r]);
            let par = SendPtr(&partials[r]);
            let slot = unsafe { outs.as_mut_ptr().add(r) } as usize;
            let st = SendPtr(&stg_tbls[r]);
            let rt = SendPtr(&rdy_tbls[r]);
            s.spawn(move || {
                let SendPtr(b) = b;
                let (SendPtr(stg), SendPtr(rdy), SendPtr(ctr), SendPtr(par)) = (stg, rdy, ctr, par);
                let (SendPtr(st), SendPtr(rt)) = (st, rt);
                let out = unsafe { &mut *(slot as *mut DevBuf) };
                b.enter();
                let _cap = capture_lock().lock().unwrap();
                b.graph_capture_begin();
                b.p2p_ar_oneshot_dev(par, st, rt, ctr, stg, rdy, out,
                                     COUNT, WORLD, r).unwrap();
                b.graph_capture_end("p2p_g");
            });
        }
    });
    // replay loop: same barrier cadence, host flag reset between replays
    let t2 = std::time::Instant::now();
    let barrier3 = std::sync::Barrier::new(WORLD);
    std::thread::scope(|s| {
        let bar = &barrier3;
        for r in 0..WORLD {
            let b = SendPtr(&backends[r]);
            let rdy_slot = unsafe { ready.as_mut_ptr().add(r) } as usize;
            s.spawn(move || {
                let SendPtr(b) = b;
                let rdy = unsafe { &*(rdy_slot as *const DevBuf) };
                b.enter();
                let zeros = vec![0f32; WORLD];
                for _ in 0..N {
                    assert!(b.graph_replay("p2p_g"), "replay");
                    let _ = b.sync();
                    bar.wait();
                    rdy.upload(&zeros).unwrap();
                    let _ = b.sync();
                    bar.wait();
                }
            });
        }
    });
    let us_graph = t2.elapsed().as_secs_f64() * 1e6 / N as f64;
    // verify replay output too
    {
        backends[0].enter();
        let mut v = vec![0f32; COUNT];
        outs[0].download(&mut v).unwrap();
        assert!(v.iter().all(|x| (x - 10.0).abs() < 1e-4), "graph one-shot wrong");
    }
    eprintln!("[p2p_oneshot] GRAPH replay: one-shot {:.2}us/AR (vs NCCL-in-graph ~15us; host-launch {:.2} vs {:.2})",
              us_graph, us_oneshot, us_nccl);
}
