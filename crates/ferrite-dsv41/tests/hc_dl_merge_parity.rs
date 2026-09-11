//! dots+LATE merge parity: `dsv41_hc_front_split` with the merged node
//! (`DSV41_HC_DL_MERGE`, default ON — `hc_dots_late_kernel`, where the LAST dot
//! block to publish `g_hc_part` is elected to run the LATE half) must reproduce
//! the two-launch path (`dsv41_hc_front`: dots then `HC_TAIL_FULL`) BIT FOR BIT
//! — pre/post/comb, the collapsed `out`, and the T1 fp8 pair (xq/xsc).
//!
//! The merge is bit-exact *by construction*: the dot branch is
//! `hc_mix_dots_kernel`'s body verbatim (same cp.async staging, same warp-0
//! float4 three-accumulator lane chain, same ss replay residue `m*32` / stride
//! `mix*32`), and the tail is `hc_mixes_tail_kernel`'s LATE branch verbatim at
//! `ss_in == 1` (warp 0 only, so the dots' 128-thread block is enough). Any
//! drift is a real regression and must block the default-ON flip.
//!
//! Run both arms (the gate is read once per process, so A/B needs two runs):
//!
//!   DSV41_MODEL_DIR=... DSV41_KERNELS=... CUDA_VISIBLE_DEVICES=0 \
//!     cargo test --release -p ferrite-dsv41 --test hc_dl_merge_parity -- --nocapture
//!   DSV41_HC_DL_MERGE=0 ... (same command)   # two-launch A/B arm: must also pass
//!
//! See `docs/agent/dsv41-layer-fusion.md` (hc tail split → dots+LATE 合并) and
//! `docs/agent/dsv41-persistent-arch.md` (P1d 机制复用).

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
    /// QUANT_FOLD: the fp4 packing of `out` emitted by the EARLY collapse
    /// epilogue (dim/2 bytes used) and its per-32-block scales (dim/32).
    xq4: Vec<u8>,
    xsc4: Vec<f32>,
}

#[test]
fn dl_merge_split_matches_two_launch_bit_for_bit() {
    let (Some(dir), Some(so)) = (env("DSV41_MODEL_DIR"), env("DSV41_KERNELS")) else {
        eprintln!("[hc_dl_merge_parity] skipped (set DSV41_MODEL_DIR / DSV41_KERNELS)");
        return;
    };
    let cfg = Dsv41Config::production();
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    let dev = Device::open(&so).expect("device");
    if !dev.supports_hc_tail_split() {
        eprintln!("[hc_dl_merge_parity] skipped (.so has no dsv41_hc_front_split)");
        return;
    }
    // The launcher reads its gate once per process, so pin the arm here: default
    // is the merged node; `DSV41_HC_DL_MERGE=0` (or `DSV41_HC_SS=0`) exercises the
    // fallback, which must agree with the two-launch path trivially.
    if std::env::var("DSV41_HC_DL_MERGE").is_err() {
        std::env::set_var("DSV41_HC_DL_MERGE", "1");
    }
    let merged_arm = std::env::var("DSV41_HC_DL_MERGE").map(|v| v != "0").unwrap_or(true)
        && std::env::var("DSV41_HC_SS").map(|v| v != "0").unwrap_or(true);
    eprintln!(
        "[hc_dl_merge_parity] arm = {}",
        if merged_arm { "merged dots+LATE (one node)" } else { "two-launch fallback" }
    );

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

    // The collapse half is exercised too, so the split path's EARLY output
    // (out/xq/xsc) and the merged node's LATE output (pre/post/comb) are both
    // compared against the two-launch path.
    let xs: Vec<f32> = (0..hc * dim).map(|i| (i % 97) as f32 * 0.01 - 0.5).collect();
    let wn: Vec<f32> = (0..dim).map(|i| 0.5 + (i % 13) as f32 * 0.05).collect();
    let pc: Vec<f32> = (0..hc).map(|i| 0.25 * (i as f32 + 1.0)).collect();

    let dx = dev.upload_f32(&xs).unwrap();
    let dwn = dev.upload_f32(&wn).unwrap();
    let dpc = dev.upload_f32(&pc).unwrap();

    let run = |split: bool| -> FrontOut {
        let pre = dev.alloc(hc * 4).unwrap();
        let post = dev.alloc(hc * 4).unwrap();
        let comb = dev.alloc(hc * hc * 4).unwrap();
        let out = dev.alloc(dim * 4).unwrap();
        let xq = dev.alloc(dim).unwrap();
        let xsc = dev.alloc((dim / 32) * 4).unwrap();
        // QUANT_FOLD destinations: the EARLY epilogue writes the fp4 pair here.
        let xq4 = dev.alloc(dim).unwrap();
        let xsc4 = dev.alloc((dim / 32) * 4).unwrap();
        let ok = if split {
            // The side-stream entry: EARLY + (dots+LATE merged) + record(join).
            let ok = dev
                .hc_front_split(
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
                    xq4.ptr as *mut u8,
                    xsc4.ptr as *mut f32,
                )
                .unwrap();
            // The caller's contract: wait `join_ev` before the hc_post that
            // consumes `comb`. Same call the model makes (`hc_tail_join`).
            dev.hc_tail_join().unwrap();
            ok
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
                xq4.ptr as *mut u8,
                xsc4.ptr as *mut f32,
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
        let mut q4 = vec![0u8; dim];
        let mut sc4 = vec![0f32; dim / 32];
        dev.download_f32(&pre, &mut p).unwrap();
        dev.download_f32(&post, &mut q).unwrap();
        dev.download_f32(&comb, &mut c).unwrap();
        dev.download_f32(&out, &mut o).unwrap();
        dev.download_u8(&xq, &mut q8).unwrap();
        dev.download_f32(&xsc, &mut sc).unwrap();
        dev.download_u8(&xq4, &mut q4).unwrap();
        dev.download_f32(&xsc4, &mut sc4).unwrap();
        // QUANT_FOLD must be BIT-IDENTICAL to the launch it replaces: run the
        // standalone `quant_fp4` over the SAME `out` value and compare the pair.
        // This is the contract `chain_dev::moe` relies on when it skips that
        // launch, so it is checked here rather than left to the serve text A/B.
        let rq4 = dev.alloc(dim).unwrap();
        let rsc4 = dev.alloc((dim / 32) * 4).unwrap();
        dev.quant_fp4(
            out.ptr as *const f32,
            rq4.ptr as *mut u8,
            rsc4.ptr as *mut f32,
            1,
            dim as i32,
            32,
            true,
        )
        .unwrap();
        dev.sync().unwrap();
        let mut rq = vec![0u8; dim];
        let mut rs = vec![0f32; dim / 32];
        dev.download_u8(&rq4, &mut rq).unwrap();
        dev.download_f32(&rsc4, &mut rs).unwrap();
        // Only the first dim/2 bytes of the fp4 row are written (two values per
        // byte); the tail of the allocation is never touched by either path.
        assert_eq!(
            q4[..dim / 2],
            rq[..dim / 2],
            "QUANT_FOLD fp4 bytes differ from a standalone quant_fp4"
        );
        assert_eq!(sc4, rs, "QUANT_FOLD fp4 scales differ from a standalone quant_fp4");
        FrontOut {
            pre: p,
            post: q,
            comb: c,
            out: o,
            xq: q8,
            xsc: sc,
            xq4: q4,
            xsc4: sc4,
        }
    };

    // `run(false)` first: the two-launch reference, then the same binary's split
    // path (whose side chain is now EARLY + one merged node).
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
        "[hc_dl_merge_parity] OK — two-launch and merged dots+LATE agree bit for bit \
         (pre/post {}, comb {}, out {}, xq {}, xsc {})",
        hc,
        hc * hc,
        dim,
        dim,
        dim / 32
    );
}
