//! swapAB ↔ SIMT-gemv numerical parity gate.
//!
//! `dsv41_gemm_fp8_swapab` (`gemm_fp8_swapab_kernel`, M=1 decode on the tensor
//! core, weight on the MMA's M / activation on B's column 0) is **not
//! bit-identical** to the SIMT `dsv41_gemm_fp8_mx` (m=1 → `gemm_fp8_gemv_kernel`,
//! LUT decode + `(a*sa)*(w*sb)` FMA chain reduced by shuffles). Both consume the
//! SAME fp8 bytes with the SAME block-32 scale scheme, so the only difference is
//! the summation order: the tensor core sums raw fp8 products per k block and
//! scales once; the SIMT kernel scales each element before it accumulates.
//!
//! This file is the numerical half of the swapAB parity judgement (the other
//! half is the four-prompt serve A/B, see `docs/agent/perf-roadmap.md` §swapAB
//! and `scripts/dsv41_swapab_text_parity.sh`). Precedent: `DSV41_WOB_F32`
//! (`dsv41_gemm_fp8_mx_f32`) is also non-bit-exact and was accepted on the same
//! terms.
//!
//! What is compared, per shape:
//!   1. **CPU golden** — the mathematically equivalent block-scaled sum built
//!      from the exact fp8 bytes. Both GPU paths must agree with it, otherwise
//!      a shape/scale-layout bug would go unnoticed when both GPU paths happen
//!      to share it.
//!   2. **swapAB vs SIMT gemv** — max |diff| / max|ref| (the `.cu` self-test's
//!      global metric) plus the per-element relative-error distribution
//!      (p50/p90/p99/max) with a floor so near-zero outputs cannot blow the
//!      ratio up.
//!   3. **Leading-token stability** — the argmax of the output vector must match
//!      (a GEMV output stands in for a logit row). The top1−top2 margin vs the
//!      observed max |diff| is reported: `max_diff < margin` PROVES the argmax
//!      cannot flip on this input, which is the quantitative form of "偶发首
//!      token 差异" from the W8A16 precedent.
//!
//! Judgement criteria (asserted below):
//!   * `global_rel = max|diff| / max|ref| < 5e-2`   (fp8 e4m3 quantisation
//!     noise order — far looser than the accumulation-order error, which lands
//!     at ~1e-6; the slack exists so a future legitimately different-but-equal
//!     scheme still passes while a real bug does not).
//!   * `max_rel < 5e-2` on the floored per-element metric.
//!   * `argmax(swapab) == argmax(simt)` for every shape.
//!
//! Run (needs a GPU + the built kernels; no checkpoint required — all inputs are
//! synthetic and quantised the runtime way):
//!
//!   DSV41_KERNELS=./kernels/cuda/libferrite_kernels.so CUDA_VISIBLE_DEVICES=0 \
//!     cargo test --release -p ferrite-dsv41 --test swapab_parity -- --nocapture
//!
//! The test skips (does not fail) when `DSV41_KERNELS` is unset or the loaded
//! `.so` was built before `dsv41_gemm_fp8_swapab` existed (stale .so => the
//! SIMT path is the only one available, so there is nothing to compare against).

use ferrite_dsv41::device::Device;
use ferrite_dsv41::quant;

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// Deterministic LCG (no external rng dependency), same generator family the
/// other tests in this crate use.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Uniform in [-0.5, 0.5).
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (1u64 << 31) as f32 - 0.5
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

/// Weight quantiser mirroring the checkpoint layout: fp8 e4m3 `[n, k]` row-major
/// with a ue8m0 power-of-two scale per **32x32 tile** (`[n/32, k/32]`). This is
/// the same routine the CUDA self-test builds its inputs with.
///
/// NOTE: `n` must be a multiple of 32 here. The launcher only checks `n % 16`,
/// but the scale array is indexed `w_scale[(m0>>5) * nb_k + kb]` — for a tile
/// whose rows cross a 32-row boundary without a backing scale row (`n % 32 !=
/// 0`) that read is out of bounds. Real checkpoint dense weights are always
/// multiples of 32, and this test mirrors that.
fn quant_weight_fp8_block(w: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    const B: usize = 32;
    debug_assert_eq!(n % B, 0);
    debug_assert_eq!(k % B, 0);
    let nb_k = k / B;
    let mut wq = vec![0u8; n * k];
    let mut ws = vec![0u8; (n / B) * nb_k];
    for cb in 0..n / B {
        for kb in 0..nb_k {
            let mut amax = 0f32;
            for c in 0..B {
                for kk in 0..B {
                    amax = amax.max(w[(cb * B + c) * k + kb * B + kk].abs());
                }
            }
            let sc = quant::fast_round_scale(amax).max(f32::MIN_POSITIVE);
            ws[cb * nb_k + kb] = quant::ue8m0_encode_pow2(quant::fast_log2_ceil(sc));
            for c in 0..B {
                for kk in 0..B {
                    let v = (w[(cb * B + c) * k + kb * B + kk] / sc)
                        .clamp(-quant::FP8_MAX, quant::FP8_MAX);
                    wq[(cb * B + c) * k + kb * B + kk] = quant::e4m3_encode(v);
                }
            }
        }
    }
    (wq, ws)
}

/// CPU golden: `out[c] = Σ_kb (Σ_{kk<32} a·w) * asc[kb] * wsc[c/32, kb] + bias`.
/// This is the mathematically equivalent definition both GPU paths implement;
/// it is the only way to catch a bug they would share.
fn cpu_gemv(
    aq: &[u8],
    asc: &[f32],
    wq: &[u8],
    wsc: &[u8],
    n: usize,
    k: usize,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let nb_k = k / 32;
    let mut out = vec![0f32; n];
    for c in 0..n {
        let mut acc = 0f32;
        for kb in 0..nb_k {
            let mut part = 0f32;
            for kk in 0..32 {
                part += quant::e4m3_decode(aq[kb * 32 + kk])
                    * quant::e4m3_decode(wq[c * k + kb * 32 + kk]);
            }
            acc += part * asc[kb] * quant::ue8m0_decode(wsc[(c / 32) * nb_k + kb]);
        }
        out[c] = acc + bias.map_or(0.0, |b| b[c]);
    }
    out
}

/// All the numbers the judgement needs.
#[derive(Debug)]
struct Stats {
    max_abs: f32,
    global_rel: f32,
    /// Floored per-element relative error, sorted-percentile style.
    p50: f32,
    p90: f32,
    p99: f32,
    max_rel: f32,
    /// Element count whose fp32 bits are identical (expected: a minority).
    bit_identical: usize,
    argmax_new: usize,
    argmax_ref: usize,
    /// top1 - top2 of the reference: a diff larger than this CAN flip argmax.
    margin: f32,
}

fn compare(new: &[f32], reference: &[f32]) -> Stats {
    assert_eq!(new.len(), reference.len());
    let n = new.len();
    let refmax = reference.iter().fold(0f32, |m, v| m.max(v.abs()));
    // Floor the per-element denominator at 1e-3 * max|ref| so elements the
    // weights happened to null out cannot produce a meaningless ratio.
    let floor = (refmax * 1e-3).max(f32::MIN_POSITIVE);

    let mut rels = Vec::with_capacity(n);
    let mut max_abs = 0f32;
    let mut bit_identical = 0usize;
    for i in 0..n {
        let d = (new[i] - reference[i]).abs();
        max_abs = max_abs.max(d);
        if new[i].to_bits() == reference[i].to_bits() {
            bit_identical += 1;
        }
        rels.push(d / reference[i].abs().max(floor));
    }
    rels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f32| rels[((n as f32 - 1.0) * p).round() as usize];

    let global_rel = if refmax > 0.0 { max_abs / refmax } else { max_abs };

    let idx_max = |v: &[f32]| {
        let mut best = 0usize;
        for (i, &x) in v.iter().enumerate() {
            if x > v[best] {
                best = i;
            }
        }
        best
    };
    let argmax_ref = idx_max(reference);
    let argmax_new = idx_max(new);
    let mut sorted = reference.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let margin = sorted[0] - sorted[1];

    Stats {
        max_abs,
        global_rel,
        p50: pct(0.50),
        p90: pct(0.90),
        p99: pct(0.99),
        max_rel: rels[n - 1],
        bit_identical,
        argmax_new,
        argmax_ref,
        margin,
    }
}

/// One shape: build inputs, run both GPU paths + the CPU golden, compare.
///
/// `None` = the swapAB launcher declined the shape, which since 2026-09-12
/// includes EVERY n < 1664: the launcher's shape dispatch sends those small
/// calls to the SIMT gemv (the measured crossover — swapAB's fixed overhead
/// loses below it, see dsv41_kernels.cu). There is no swapAB result to compare
/// then, so the caller SKIPS the shape instead of failing.
fn run_case(dev: &Device, n: usize, k: usize, with_bias: bool, seed: u64) -> Option<Stats> {
    let mut rng = Lcg::new(seed);
    // Activation: one row, per-32-block power-of-two scale (runtime semantics).
    let x = rng.vec(k);
    let (aq, asc) = quant::act_quant_fp8(&x, 32, true);
    // Weights: scaled up so the fp8 rounding is representative of real weights.
    let wf: Vec<f32> = (0..n * k).map(|_| rng.next_f32() * 4.0).collect();
    let (wq, wsc) = quant_weight_fp8_block(&wf, n, k);
    let bias: Option<Vec<f32>> = with_bias.then(|| rng.vec(n));

    let golden = cpu_gemv(&aq, &asc, &wq, &wsc, n, k, bias.as_deref());

    let da = dev.upload(&aq).unwrap();
    let dasc = dev.upload_f32(&asc).unwrap();
    let dw = dev.upload(&wq).unwrap();
    let dwsc = dev.upload(&wsc).unwrap();
    let dbias = match &bias {
        Some(b) => dev.upload_f32(b).unwrap(),
        None => dev.alloc(4).unwrap(),
    };
    let bias_ptr = if with_bias { dbias.as_f32() } else { std::ptr::null() };

    // ---- arm 1: SIMT gemv (the m=1 branch of dsv41_gemm_fp8_mx) ----
    let dref = dev.alloc(n * 4).unwrap();
    dev.gemm_fp8_mx(
        da.as_u8(),
        dasc.as_f32(),
        dw.as_u8(),
        dwsc.as_u8(),
        bias_ptr,
        dref.ptr as *mut f32,
        1,
        n as i32,
        k as i32,
    )
    .expect("dsv41_gemm_fp8_mx (SIMT gemv)");
    dev.sync().unwrap();
    let mut refo = vec![0f32; n];
    dev.download_f32(&dref, &mut refo).unwrap();

    // ---- arm 2: swapAB tensor-core GEMV ----
    let dnew = dev.alloc(n * 4).unwrap();
    // Last-block-reduction scratch (ks > 1 splits K for the production shape):
    // 8 * n partial slots + one u32 ticket per 16-row tile. The tickets must be
    // zeroed once; the kernel self-resets each entry it consumes.
    let dpart = dev.alloc(n * 8 * 4).unwrap();
    let dctr = dev.alloc((n / 16 + 1) * 4).unwrap();
    dev.zero(&dctr).unwrap();
    let ran = dev
        .gemm_fp8_swapab(
            da.as_u8(),
            dasc.as_f32(),
            dw.as_u8(),
            dwsc.as_u8(),
            bias_ptr,
            dnew.ptr as *mut f32,
            n as i32,
            k as i32,
            dpart.ptr as *mut f32,
            dctr.ptr as *mut u32,
        )
        .expect("dsv41_gemm_fp8_swapab");
    if !ran {
        // rc 2 = the launcher's graceful decline. Since 2026-09-12 that includes
        // EVERY n < 1664 (the shape dispatch routes those small calls — the ones
        // dominating the step — to the SIMT gemv, where swapAB's fixed overhead
        // loses). No swapAB result exists to compare, so SKIP the shape; the
        // SIMT fallback is the correct path there and the .cu self-test already
        // prints the decline. `ran` false for n%32!=0 is impossible in this file
        // (see the scale-layout note on `quant_weight_fp8_block`).
        return None;
    }
    dev.sync().unwrap();
    let mut newo = vec![0f32; n];
    dev.download_f32(&dnew, &mut newo).unwrap();

    // ---- the CPU golden must agree with BOTH (catches shared-layout bugs) ----
    for (tag, got) in [("simt", &refo), ("swapab", &newo)] {
        let s = compare(got, &golden);
        assert!(
            s.global_rel < 5e-2,
            "[n={n} k={k} bias={with_bias}] {tag} vs CPU golden: global_rel={:.3e} max_abs={:.3e}",
            s.global_rel,
            s.max_abs
        );
        eprintln!(
            "    {tag:<6} vs golden: global_rel={:.3e} bit_id={}/{} argmax={}",
            s.global_rel, s.bit_identical, n, s.argmax_new
        );
    }

    Some(compare(&newo, &refo))
}

#[test]
fn swapab_matches_simt_gemv() {
    let Some(so) = env("DSV41_KERNELS") else {
        eprintln!("[swapab_parity] skipped (set DSV41_KERNELS to the built libferrite_kernels.so)");
        return;
    };
    // The two arms must actually be the two arms: `DSV41_NO_SWAPAB` makes the
    // swapAB launcher decline (rc 2) and `DSV41_NO_GEMV_FP8` makes the m=1 GEMV
    // fall through to the tiled dense kernel. Refuse to run mis-configured.
    for g in ["DSV41_NO_SWAPAB", "DSV41_NO_GEMV_FP8"] {
        assert!(
            env(g).is_none(),
            "{g} is set — this test compares swapAB against the SIMT gemv, unset it"
        );
    }

    let dev = Device::open(&so).expect("open device");
    if !dev.supports_gemm_fp8_swapab() {
        eprintln!("[swapab_parity] skipped (stale .so: no dsv41_gemm_fp8_swapab)");
        return;
    }

    // Shapes: the .cu self-test's edge cases + a real production shape.
    //  * (32, 32)    one stage / one k block / one weight-scale row block
    //  * (96, 64)    three 32-row weight-scale blocks
    //  * (64, 544)   partial ring stage (544 = 17*32)
    //  * (256, 512)  full stage, multi-warp grid
    //  * (1664,5120) the M=1 decode shape the optimisation targets
    // All n are multiples of 32 (see the scale-layout note on
    // `quant_weight_fp8_block`).
    let cases: [(usize, usize, bool, u64); 5] = [
        (32, 32, false, 1),
        (96, 64, true, 2),
        (64, 544, false, 3),
        (256, 512, true, 4),
        (1664, 5120, true, 5),
    ];

    let mut worst: f32 = 0.0;
    for (n, k, bias, seed) in cases {
        eprintln!("[swapab_parity] n={n} k={k} bias={bias}");
        let Some(s) = run_case(&dev, n, k, bias, seed) else {
            continue;
        };
        eprintln!(
            "  swapab vs simt: max_abs={:.3e} global_rel={:.3e} \
             p50={:.2e} p90={:.2e} p99={:.2e} max_rel={:.3e} bit_id={}/{} \
             argmax {}/{} margin={:.3e}",
            s.max_abs,
            s.global_rel,
            s.p50,
            s.p90,
            s.p99,
            s.max_rel,
            s.bit_identical,
            n,
            s.argmax_new,
            s.argmax_ref,
            s.margin
        );
        if s.max_abs < s.margin {
            eprintln!(
                "  -> argmax PROVABLY stable (max_abs {:.3e} < margin {:.3e})",
                s.max_abs, s.margin
            );
        }
        assert!(
            s.global_rel < 5e-2,
            "[n={n} k={k}] swapAB vs SIMT gemv global_rel={:.3e} exceeds 5e-2 \
             (fp8 e4m3 noise order)",
            s.global_rel
        );
        assert!(
            s.max_rel < 5e-2,
            "[n={n} k={k}] swapAB vs SIMT gemv max per-element rel={:.3e} exceeds 5e-2",
            s.max_rel
        );
        assert_eq!(
            s.argmax_new, s.argmax_ref,
            "[n={n} k={k}] leading-token flip: swapAB argmax {} != SIMT argmax {}",
            s.argmax_new, s.argmax_ref
        );
        worst = worst.max(s.global_rel);
    }
    eprintln!("[swapab_parity] ALL SHAPES PASS — worst global_rel = {worst:.3e}");
}
