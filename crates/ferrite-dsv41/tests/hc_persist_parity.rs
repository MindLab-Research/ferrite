//! Stage-C persistent prototype parity: `dsv41_hc_front_persist` (the whole hc
//! front end in ONE block, ordered by `__syncthreads` phase barriers) must
//! reproduce `dsv41_hc_front` (the two-launch dots+tail pair) BIT FOR BIT —
//! pre/post/comb, the collapsed `out`, and the T1 fp8 pair (xq/xsc).
//!
//! The merge is bit-exact *by construction* (the same lane assignment, the same
//! reduction grouping, the same arithmetic; only the storage location of the
//! weight row moves from smem to global), so any drift is a real regression and
//! must block the `DSV41_HC_PERSIST=1` flip. See
//! `docs/agent/dsv41-persistent-arch.md` §5.
//!
//!   DSV41_MODEL_DIR=... DSV41_KERNELS=... CUDA_VISIBLE_DEVICES=0 \
//!     cargo test --release -p ferrite-dsv41 --test hc_persist_parity -- --nocapture

use ferrite_dsv41::config::Dsv41Config;
use ferrite_dsv41::device::Device;
use ferrite_dsv41::load::Loader;
use ferrite_dsv41::weights::{Shard, TensorSpec};

fn env(n: &str) -> Option<String> {
    std::env::var(n).ok().filter(|v| !v.is_empty())
}

struct FrontOut {
    pre: Vec<f32>,
    post: Vec<f32>,
    comb: Vec<f32>,
    out: Vec<f32>,
    xq: Vec<u8>,
    xsc: Vec<f32>,
}

#[test]
fn persist_matches_two_launch_bit_for_bit() {
    let (Some(dir), Some(so)) = (env("DSV41_MODEL_DIR"), env("DSV41_KERNELS")) else {
        eprintln!("[hc_persist_parity] skipped (set DSV41_MODEL_DIR / DSV41_KERNELS)");
        return;
    };
    let cfg = Dsv41Config::production();
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    let dev = Device::open(&so).expect("device");
    if !dev.supports_hc_persist() {
        eprintln!("[hc_persist_parity] skipped (.so has no dsv41_hc_front_persist)");
        return;
    }
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

    // The collapse is exercised too: deterministic synthetic norm weight and
    // premix coefficients (no checkpoint tensor needed), so the T1 fp8 emission
    // (xq/xsc) is compared as well.
    let xs: Vec<f32> = (0..hc * dim).map(|i| (i % 97) as f32 * 0.01 - 0.5).collect();
    let wn: Vec<f32> = (0..dim).map(|i| 0.5 + (i % 13) as f32 * 0.05).collect();
    let pc: Vec<f32> = (0..hc).map(|i| 0.25 * (i as f32 + 1.0)).collect();

    let dx = dev.upload_f32(&xs).unwrap();
    let dwn = dev.upload_f32(&wn).unwrap();
    let dpc = dev.upload_f32(&pc).unwrap();

    let run = |persist: bool| -> FrontOut {
        let pre = dev.alloc(hc * 4).unwrap();
        let post = dev.alloc(hc * 4).unwrap();
        let comb = dev.alloc(hc * hc * 4).unwrap();
        let out = dev.alloc(dim * 4).unwrap();
        let xq = dev.alloc(dim).unwrap();
        let xsc = dev.alloc((dim / 32) * 4).unwrap();
        let ok = if persist {
            dev.hc_front_persist(
                dx.as_f32(),
                fn_w.as_f32(),
                scale.as_f32(),
                base.as_f32(),
                dwn.as_f32(),
                dpc.as_f32(),
                pre.ptr as *mut f32,
                post.ptr as *mut f32,
                comb.ptr as *mut f32,
                out.ptr as *mut f32,
                1,
                hc as i32,
                dim as i32,
                cfg.hc_sinkhorn_iters as i32,
                cfg.hc_eps,
                cfg.norm_eps,
                xq.ptr as *mut u8,
                xsc.ptr as *mut f32,
            )
            .unwrap()
        } else {
            dev.hc_front(
                dx.as_f32(),
                fn_w.as_f32(),
                scale.as_f32(),
                base.as_f32(),
                dwn.as_f32(),
                dpc.as_f32(),
                pre.ptr as *mut f32,
                post.ptr as *mut f32,
                comb.ptr as *mut f32,
                out.ptr as *mut f32,
                1,
                hc as i32,
                dim as i32,
                cfg.hc_sinkhorn_iters as i32,
                cfg.hc_eps,
                cfg.norm_eps,
                xq.ptr as *mut u8,
                xsc.ptr as *mut f32,
                // QUANT_FOLD destinations: null here, so this arm takes the
                // standalone-quant_fp4 behaviour.
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
            .unwrap()
        };
        assert!(ok, "the hc front end refused this shape");
        dev.sync().unwrap();
        let mut p = vec![0f32; hc];
        let mut q = vec![0f32; hc];
        let mut c = vec![0f32; hc * hc];
        let mut o = vec![0f32; dim];
        let mut q8 = vec![0u8; dim];
        let mut sc = vec![0f32; dim / 32];
        dev.download_f32(&pre, &mut p).unwrap();
        dev.download_f32(&post, &mut q).unwrap();
        dev.download_f32(&comb, &mut c).unwrap();
        dev.download_f32(&out, &mut o).unwrap();
        dev.download_u8(&xq, &mut q8).unwrap();
        dev.download_f32(&xsc, &mut sc).unwrap();
        FrontOut {
            pre: p,
            post: q,
            comb: c,
            out: o,
            xq: q8,
            xsc: sc,
        }
    };

    let two = run(false);
    let one = run(true);

    // Bit-for-bit, not a tolerance: the two shapes must agree exactly.
    assert_eq!(two.pre, one.pre, "pre differs");
    assert_eq!(two.post, one.post, "post differs");
    assert_eq!(two.comb, one.comb, "comb differs");
    assert_eq!(two.out, one.out, "collapsed out differs");
    assert_eq!(two.xq, one.xq, "T1 fp8 bytes differ");
    assert_eq!(two.xsc, one.xsc, "T1 fp8 scales differ");
    eprintln!(
        "[hc_persist_parity] OK — two-launch and one-block agree bit for bit \
         (pre/post {}, comb {}, out {}, xq {}, xsc {})",
        hc,
        hc * hc,
        dim,
        dim,
        dim / 32
    );
}
