//! Multi-row segment-C parity: `dsv41_hc_post_inplace_rows` (ONE launch, in
//! place on the m-row residual block) must reproduce `hc_post` into the `h2_r`
//! staging buffer + the device-to-device copy back **BIT FOR BIT**, for every row
//! of the block — that is the pair `layer_rows()` used before A1's in-place
//! wiring, and the pair `hc_post_rows` keeps as its fallback.
//!
//! Three comparisons, all exact (no tolerance):
//!
//! 1. rows = m, in place  ==  rows = m, `hc_post` + copy;
//! 2. rows = m, in place  ==  m single-row `hc_post_inplace` launches, one per
//!    row slice. This is the "row r of the m-row launch is the m = 1 launch of
//!    row r" contract the whole `layer_rows` chain rests on;
//! 3. rows = 1, in place  ==  the single-row entry `dsv41_hc_post_inplace`
//!    (the decode path's kernel), so the two entries cannot drift apart.
//!
//! The kernel is bit-exact *by construction*: same `post[t,i]*x[t,j]` seed, same
//! ascending-k `__fmaf_rn` chain, same four-float column per thread, and the row
//! base is the only thing the multi-row form adds. Any drift is a real
//! regression and must block the default-ON flip of `DSV41_HC_VERIFY_FUSE`.
//!
//! Run (needs a GPU and the built kernel `.so`; skips silently without them):
//!
//!   DSV41_MODEL_DIR=... DSV41_KERNELS=... CUDA_VISIBLE_DEVICES=0 \
//!     cargo test --release -p ferrite-dsv41 --test hc_post_rows_parity -- --nocapture
//!
//! See `docs/agent/hc-chain-bandwidth-analysis.md` §3 (A1-b) and §4 (H3).

use ferrite_dsv41::config::Dsv41Config;
use ferrite_dsv41::device::{DevBuf, Device};

fn env(n: &str) -> Option<String> {
    std::env::var(n).ok().filter(|v| !v.is_empty())
}

/// Deterministic filler — the values are irrelevant (every comparison is exact,
/// and both arms see the same bytes), so a cheap LCG keeps the test self-contained.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 33) as u32 % 2000) as f32 - 1000.0) / 512.0
        })
        .collect()
}

#[test]
fn hc_post_inplace_rows_matches_hc_post_bit_for_bit() {
    let (Some(dir), Some(so)) = (env("DSV41_MODEL_DIR"), env("DSV41_KERNELS")) else {
        eprintln!("[hc_post_rows_parity] skipped (set DSV41_MODEL_DIR / DSV41_KERNELS)");
        return;
    };
    let _ = dir; // no weights are needed: the post is a pure elementwise op
    let cfg = Dsv41Config::production();
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    // `VERIFY_ROWS` (drafts + 1); the exact value only has to be > 1.
    let rows = 5usize;

    let dev = Device::open(&so).expect("device");
    if !dev.supports_hc_post_inplace_rows() {
        eprintln!("[hc_post_rows_parity] skipped (.so has no dsv41_hc_post_inplace_rows)");
        return;
    }

    // [rows, hc*dim] residual, [rows, dim] producer row, [rows, hc] gate,
    // [rows, hc*hc] mix — the exact strides `layer_rows`' buffers use.
    let res_h = fill(rows * hc * dim, 0x51ed);
    let x_h = fill(rows * dim, 0x9e37);
    let post_h = fill(rows * hc, 0x1234);
    let comb_h = fill(rows * hc * hc, 0xabcd);

    let upload = |v: &[f32]| -> DevBuf { dev.upload_f32(v).unwrap() };
    let take = |b: &DevBuf, n: usize| -> Vec<f32> {
        let mut v = vec![0f32; n];
        dev.download_f32(b, &mut v).unwrap();
        v
    };

    let dx = upload(&x_h);
    let dpost = upload(&post_h);
    let dcomb = upload(&comb_h);

    // ---- arm 1: `hc_post` into the staging buffer (+ the copy the model then
    // does; the copy is a pure move, so comparing `out` and the in-place `res`
    // covers the whole pair) ----
    let dres_stage = upload(&res_h);
    let dout = dev.alloc(rows * hc * dim * 4).unwrap();
    dev.hc_post(
        dx.as_f32(),
        dres_stage.as_f32(),
        dpost.as_f32(),
        dcomb.as_f32(),
        dout.ptr as *mut f32,
        rows as i32,
        hc as i32,
        dim as i32,
    )
    .unwrap();

    // ---- arm 2: the rows entry, in place ----
    let dres_rows = upload(&res_h);
    dev.hc_post_inplace_rows(
        dres_rows.ptr as *mut f32,
        dx.as_f32(),
        dpost.as_f32(),
        dcomb.as_f32(),
        rows as i32,
        hc as i32,
        dim as i32,
    )
    .unwrap();

    // ---- arm 3: m single-row `hc_post_inplace` launches, one per row slice ----
    let dres_perrow = upload(&res_h);
    for r in 0..rows {
        let res_r = (dres_perrow.ptr as *mut f32).wrapping_add(r * hc * dim);
        let x_r = (dx.ptr as *const f32).wrapping_add(r * dim);
        let post_r = (dpost.ptr as *const f32).wrapping_add(r * hc);
        let comb_r = (dcomb.ptr as *const f32).wrapping_add(r * hc * hc);
        dev.hc_post_inplace(res_r, x_r, post_r, comb_r, hc as i32, dim as i32)
            .unwrap();
    }

    // ---- arm 4: rows = 1 through BOTH entries, on row 0's slices ----
    let dres_one_rows = upload(&res_h[..hc * dim]);
    dev.hc_post_inplace_rows(
        dres_one_rows.ptr as *mut f32,
        dx.as_f32(),
        dpost.as_f32(),
        dcomb.as_f32(),
        1,
        hc as i32,
        dim as i32,
    )
    .unwrap();
    let dres_one_single = upload(&res_h[..hc * dim]);
    dev.hc_post_inplace(
        dres_one_single.ptr as *mut f32,
        dx.as_f32(),
        dpost.as_f32(),
        dcomb.as_f32(),
        hc as i32,
        dim as i32,
    )
    .unwrap();

    dev.sync().unwrap();

    let staged = take(&dout, rows * hc * dim);
    let rows_out = take(&dres_rows, rows * hc * dim);
    let perrow_out = take(&dres_perrow, rows * hc * dim);
    let one_rows = take(&dres_one_rows, hc * dim);
    let one_single = take(&dres_one_single, hc * dim);

    assert_eq!(
        staged, rows_out,
        "hc_post_inplace_rows(rows={rows}) differs from hc_post + copy"
    );
    assert_eq!(
        perrow_out, rows_out,
        "hc_post_inplace_rows(rows={rows}) differs from m single-row launches"
    );
    assert_eq!(
        one_single, one_rows,
        "the rows entry at rows=1 differs from dsv41_hc_post_inplace"
    );
    eprintln!(
        "[hc_post_rows_parity] OK — rows={rows} in-place post agrees bit for bit \
         with hc_post + copy, with {rows} single-row launches, and with the \
         single-row entry at rows=1 (hc={hc}, dim={dim})"
    );
}
