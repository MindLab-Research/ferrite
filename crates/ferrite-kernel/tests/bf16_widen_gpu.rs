//! GPU kernel unit tests for the mmap direct-load path's conversion kernels
//! (bf16→f32 widen, bf16 raw residency, f32 raw, fp8 dequant) — the serve logs
//! showed ferrite_bf16_to_f32 outputs as garbage ([48128, 15488...] vs
//! expected [-0.97, 1.11...]) while pure-memcpy paths verified correct; these
//! tests isolate every kernel with bit-exact CPU references on real hardware.
//! Run on B300: CUDA_VISIBLE_DEVICES=4 cargo test --release --test bf16_widen_gpu -- --nocapture
#![cfg(feature = "cuda")]

use ferrite_kernel::cuda::CudaBackend;
use ferrite_types::{DType, Shape, Tensor};
use std::sync::Arc;

fn so_path() -> String {
    std::env::var("FERRITE_KERNEL_SO").unwrap_or_else(|_| {
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("../../kernels/cuda/libferrite_kernels.so");
        p.to_string_lossy().into_owned()
    })
}

/// mmap-path placeholder tensor: shape-real, 4-elem zero stub (the same
/// pattern direct.rs's placeholder() produces — device caches key on ptr).
fn stub(shape: Vec<usize>) -> Tensor {
    Tensor {
        shape: Shape::new(shape),
        dtype: DType::F32,
        data: Arc::new(vec![0f32; 4]),
    }
}

fn f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// Bit-exact comparison (bf16→f32 widen is a pure bit shift — any deviation
/// is a real bug, not precision noise).
fn assert_bit_exact(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: len {} vs {}", got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if g.to_bits() != w.to_bits() {
            panic!(
                "{what}: mismatch at {i}: got {} (bits {:#x}) want {} (bits {:#x})",
                g,
                g.to_bits(),
                w,
                w.to_bits()
            );
        }
    }
}

// ============================================================
// 1. Direct kernel roundtrip: ferrite_bf16_to_f32 (the kernel both
//    preload_bf16_to_f32_raw AND the dev_weight widen call — the exact
//    kernel the serve logs showed producing garbage).
// ============================================================

#[test]
fn test_kernel_bf16_to_f32_small() {
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda");
    let vals: Vec<f32> = vec![-0.97, 1.11, 1.38, -1.16, 0.0, -0.5, 0.25, 1.0];
    let bf16 = f32_to_bf16_bytes(&vals);
    let got = dev.dbg_kernel_bf16_to_f32(&bf16).unwrap();
    let want = bf16_bytes_to_f32(&bf16);
    assert_bit_exact(&got, &want, "kernel bf16→f32 small");
}

#[test]
fn test_kernel_bf16_to_f32_4096_norm_shape() {
    // input_layernorm shape [4096] — the nw weight whose serve dump showed
    // [15808×4] instead of ~1.45.
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda");
    let vals: Vec<f32> = (0..4096).map(|i| 1.45 + (i as f32) * 1e-4).collect();
    let bf16 = f32_to_bf16_bytes(&vals);
    let got = dev.dbg_kernel_bf16_to_f32(&bf16).unwrap();
    let want = bf16_bytes_to_f32(&bf16);
    assert_bit_exact(&got, &want, "kernel bf16→f32 [4096]");
}

#[test]
fn test_kernel_bf16_to_f32_24x16384_hc_shape() {
    // hc_attn_fn shape [24, 16384] = 393216 elems — the fw weight whose serve
    // dump showed [48128, 15488, 15744, 48384] instead of ~O(1).
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda");
    let n = 24usize * 16384;
    let vals: Vec<f32> = (0..n).map(|i| ((i as i64 % 977) as f32 - 444.0) * 0.0011).collect();
    let bf16 = f32_to_bf16_bytes(&vals);
    let got = dev.dbg_kernel_bf16_to_f32(&bf16).unwrap();
    let want = bf16_bytes_to_f32(&bf16);
    assert_bit_exact(&got, &want, "kernel bf16→f32 [24,16384]");
}

#[test]
fn test_kernel_bf16_to_f32_extremes() {
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda");
    let vals: Vec<f32> = vec![
        f32::MIN_POSITIVE * 1e30,
        -3.0e38,
        3.0e38,
        -0.0,
        0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        65504.0, // f16-max-adjacent, well inside bf16 range
        2.0f32.powi(-126),
    ];
    let bf16 = f32_to_bf16_bytes(&vals);
    let got = dev.dbg_kernel_bf16_to_f32(&bf16).unwrap();
    let want = bf16_bytes_to_f32(&bf16);
    // inf/-inf/nan-preserved truncation: bit-exact on the u16 widen
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "extremes at {i}: {} vs {}", g, w);
    }
}

// ============================================================
// 2. preload_bf16_to_f32_raw roundtrip (the full mmap-path entry: H2D
//    staging → kernel → cache insert) — nw/residual norms go through this.
// ============================================================

#[test]
fn test_preload_bf16_to_f32_roundtrip() {
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda");
    let vals: Vec<f32> = (0..4096).map(|i| 0.3 + (i as f32) * 1e-5).collect();
    let bf16 = f32_to_bf16_bytes(&vals);
    let ph = stub(vec![4096]);
    dev.preload_bf16_to_f32_raw(&ph, &bf16).unwrap();
    let got = dev.dbg_dev_f32(&ph).unwrap();
    let want = bf16_bytes_to_f32(&bf16);
    assert_bit_exact(&got, &want, "preload_bf16_to_f32_raw [4096]");
}

#[test]
fn test_preload_bf16_to_f32_large_393216() {
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda");
    let n = 24usize * 16384;
    let vals: Vec<f32> = (0..n).map(|i| (((i as i64 * 7919) % 4001) as f32 - 2000.0) * 0.0005).collect();
    let bf16 = f32_to_bf16_bytes(&vals);
    let ph = stub(vec![24, 16384]);
    dev.preload_bf16_to_f32_raw(&ph, &bf16).unwrap();
    let got = dev.dbg_dev_f32(&ph).unwrap();
    let want = bf16_bytes_to_f32(&bf16);
    assert_bit_exact(&got, &want, "preload_bf16_to_f32_raw [24,16384]");
}

// ============================================================
// 3. The WIDEN RECOVERY path: preload_bf16_raw (bf16 residency) then
//    dev_weight (f32 key miss → bf16 key → GPU widen) — the exact fw/hc
//    path whose serve dump showed garbage.
// ============================================================

#[test]
fn test_widen_recovery_via_dev_weight() {
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda");
    let n = 24usize * 16384;
    let vals: Vec<f32> = (0..n).map(|i| (((i as i64 * 104729) % 8009) as f32 - 4004.0) * 0.0007).collect();
    let bf16 = f32_to_bf16_bytes(&vals);
    let ph = stub(vec![24, 16384]);
    // bf16 residency (bf16 key numel<<1|1)
    dev.preload_bf16_raw(&ph, &[&bf16]).unwrap();
    // dev_weight: f32 key miss → bf16 key hit → GPU widen → f32 readback
    let got = dev.dbg_dev_f32(&ph).unwrap();
    let want = bf16_bytes_to_f32(&bf16);
    assert_bit_exact(&got, &want, "widen recovery [24,16384]");
}

#[test]
fn test_widen_recovery_after_interleaved_preloads() {
    // Interleave preload orders the way direct_preload_shard does (bf16 raw
    // → bf16→f32 → f32 raw → another bf16 raw...) and verify no cross-
    // contamination of cache entries / staging reuse.
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda");
    let mk = |seed: u64, n: usize| -> Vec<f32> {
        (0..n).map(|i| (((i as u64 * seed + 13) % 2003) as f32 - 1000.0) * 0.003).collect()
    };
    let a_bf16 = f32_to_bf16_bytes(&mk(3, 4096));
    let b_vals = mk(5, 512);
    let c_bf16 = f32_to_bf16_bytes(&mk(7, 1024));
    let d_bf16 = f32_to_bf16_bytes(&mk(11, 2048));

    let pa = stub(vec![4096]);
    let pb = stub(vec![512]);
    let pc = stub(vec![1024]);
    let pd = stub(vec![2048]);
    dev.preload_bf16_raw(&pa, &[&a_bf16]).unwrap(); // bf16 residency (A)
    let b_bytes: Vec<u8> = b_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    dev.preload_f32_raw(&pb, &b_bytes).unwrap(); // f32 (B)
    dev.preload_bf16_to_f32_raw(&pc, &c_bf16).unwrap(); // bf16→f32 (C)
    dev.preload_bf16_raw(&pd, &[&d_bf16]).unwrap(); // bf16 residency (D)

    // A: widen recovery
    let got_a = dev.dbg_dev_f32(&pa).unwrap();
    assert_bit_exact(&got_a, &bf16_bytes_to_f32(&a_bf16), "A widen after interleave");
    // B: direct f32
    let got_b = dev.dbg_dev_f32(&pb).unwrap();
    let want_b = mk(5, 512);
    assert_eq!(got_b, want_b, "B f32 raw after interleave");
    // C: bf16→f32
    let got_c = dev.dbg_dev_f32(&pc).unwrap();
    assert_bit_exact(&got_c, &bf16_bytes_to_f32(&c_bf16), "C bf16→f32 after interleave");
    // D: bf16 residency direct
    let got_d = dev.dbg_dev_bf16_as_f32(&pd).unwrap();
    assert_bit_exact(&got_d, &bf16_bytes_to_f32(&d_bf16), "D bf16 raw after interleave");
}

// ============================================================
// 4. fp8 e4m3 block dequant (preload_fp8_dequant) vs CPU reference
// ============================================================

/// e4m3 → f32 (sign(1) exp(4) mant(3), bias 7; e==0 subnormal m/8 × 2^-6;
/// S.1111.111 (0x7F/0xFF) is e4m3 NaN — the GPU's __nv_cvt_fp8_to_halfraw
/// correctly emits NaN there, the CPU reference must too).
fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((b >> 3) & 0x0f) as i32;
    let m = (b & 0x07) as i32;
    if e == 15 && m == 7 {
        return f32::NAN; // e4m3 NaN encoding (0x7F / 0xFF)
    }
    if e == 0 {
        sign * (m as f32 / 8.0) * 2f32.powi(-6)
    } else {
        sign * (1.0 + m as f32 / 8.0) * 2f32.powi(e - 7)
    }
}

#[test]
fn test_preload_fp8_dequant_vs_cpu() {
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda");
    let (rows, cols) = (256usize, 512usize); // 2×4 scale blocks
    let mut fp8 = vec![0u8; rows * cols];
    for i in 0..fp8.len() {
        fp8[i] = ((i * 37 + 11) % 251) as u8;
    }
    let srows = rows.div_ceil(128);
    let scols = cols.div_ceil(128);
    let scale: Vec<f32> = (0..srows * scols).map(|i| 0.5 + (i % 7) as f32 * 0.01).collect();
    let ph = stub(vec![rows, cols]);
    dev.preload_fp8_dequant(&ph, &fp8, &scale, rows, cols).unwrap();

    // CPU reference: dequant → f32 → truncate bf16 (bits >> 16)
    let got = dev.dbg_dev_bf16_as_f32(&ph).unwrap();
    let mut want = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let sr = (r / 128).min(srows - 1);
            let sc = (c / 128).min(scols - 1);
            let v = e4m3_to_f32(fp8[r * cols + c]) * scale[sr * scols + sc];
            want.push(f32::from_bits(v.to_bits() & 0xffff0000));
        }
    }
    assert_eq!(got.len(), want.len(), "fp8 dequant len");
    let mut bad = 0;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        // NaN == NaN is semantically equal here (e4m3 0x7F/0xFF widen to
        // quiet-NaN 0x7FFF on GPU vs f32::NAN 0x7FC0 on CPU — same value)
        if g.is_nan() && w.is_nan() {
            continue;
        }
        if g.to_bits() != w.to_bits() {
            if bad < 5 {
                eprintln!("fp8 mismatch @{}: got {} (bits {:08x}) want {} (bits {:08x})", i, g, g.to_bits(), w, w.to_bits());
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "fp8 dequant: {}/{} mismatches", bad, want.len());
}

// ============================================================
// 5. bf16 raw residency roundtrip (the memcpy path — control group that
//    already verified correct in serve logs; proves the test harness works).
// ============================================================

#[test]
fn test_preload_bf16_raw_roundtrip_control() {
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda");
    let vals: Vec<f32> = (0..4096).map(|i| ((i as i64 % 101) as f32 - 50.0) * 0.02).collect();
    let bf16 = f32_to_bf16_bytes(&vals);
    let ph = stub(vec![4096]);
    dev.preload_bf16_raw(&ph, &[&bf16]).unwrap();
    let got = dev.dbg_dev_bf16_as_f32(&ph).unwrap();
    let want = bf16_bytes_to_f32(&bf16);
    assert_bit_exact(&got, &want, "bf16 raw roundtrip [4096] (control)");
}

#[test]
fn test_preload_bf16_raw_multi_segment_qkv() {
    // QkvHeads-style 3-segment preload (q/k/v slices concatenated)
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda");
    let seg_vals: Vec<Vec<f32>> = (0..3)
        .map(|s| (0..2048).map(|i| i as f32 * 0.001 + s as f32).collect())
        .collect();
    let segs: Vec<Vec<u8>> = seg_vals.iter().map(|v| f32_to_bf16_bytes(v)).collect();
    let refs: Vec<&[u8]> = segs.iter().map(|s| s.as_slice()).collect();
    let ph = stub(vec![3 * 2048, 1]);
    dev.preload_bf16_raw(&ph, &refs).unwrap();
    let got = dev.dbg_dev_bf16_as_f32(&ph).unwrap();
    let want: Vec<f32> = bf16_bytes_to_f32(&f32_to_bf16_bytes(&seg_vals.concat()));
    assert_bit_exact(&got, &want, "bf16 raw 3-segment qkv-style");
}

// ============================================================
// 6. f32 raw roundtrip (hc_attn_scale/base [3]/[24]-style F32Seg path)
// ============================================================

#[test]
fn test_preload_f32_raw_roundtrip() {
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda");
    let vals: Vec<f32> = vec![0.0614, 0.0446, 0.0889, -7.5831, 0.5753, -7.2021];
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ph = stub(vec![6]);
    dev.preload_f32_raw(&ph, &bytes).unwrap();
    let got = dev.dbg_dev_f32(&ph).unwrap();
    assert_eq!(got, vals, "f32 raw roundtrip (hc scale/base values)");
}

// ============================================================
// 7. Stability: repeated kernel invocations (the serve preloads 90 hc fn
//    [24,16384] + 135 norms [4096] — allocator/stream reuse patterns).
// ============================================================

#[test]
fn test_kernel_repeated_stability_90x() {
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda");
    for round in 0..90 {
        let n = if round % 2 == 0 { 24usize * 16384 } else { 4096 };
        let vals: Vec<f32> = (0..n).map(|i| ((i as i64 * 31 + round as i64 * 977) % 3001) as f32 * 0.001).collect();
        let bf16 = f32_to_bf16_bytes(&vals);
        let got = dev.dbg_kernel_bf16_to_f32(&bf16).unwrap();
        let want = bf16_bytes_to_f32(&bf16);
        assert_bit_exact(&got, &want, &format!("stability round {round} n={n}"));
    }
}

#[test]
fn test_widen_recovery_repeated_45_layers() {
    // Simulate 45 layers × (fn [24,16384] widen + norm [4096] preload):
    // alternating preload patterns like direct_preload_shard's real order.
    let dev = CudaBackend::with_device(&so_path(), 0).expect("open cuda");
    for layer in 0..45 {
        let fn_n = 24usize * 16384;
        let fn_vals: Vec<f32> = (0..fn_n).map(|i| ((i as i64 * 617 + layer as i64) % 1999) as f32 * 0.0009).collect();
        let fn_bf16 = f32_to_bf16_bytes(&fn_vals);
        let ph_fn = stub(vec![24, 16384]);
        dev.preload_bf16_raw(&ph_fn, &[&fn_bf16]).unwrap();

        let norm_vals: Vec<f32> = (0..4096).map(|i| 1.45 + (i % 7) as f32 * 0.001).collect();
        let norm_bf16 = f32_to_bf16_bytes(&norm_vals);
        let ph_norm = stub(vec![4096]);
        dev.preload_bf16_to_f32_raw(&ph_norm, &norm_bf16).unwrap();

        if layer % 15 == 0 {
            let got_fn = dev.dbg_dev_f32(&ph_fn).unwrap();
            assert_bit_exact(&got_fn, &bf16_bytes_to_f32(&fn_bf16), &format!("layer {layer} fn widen"));
            let got_norm = dev.dbg_dev_f32(&ph_norm).unwrap();
            assert_bit_exact(&got_norm, &bf16_bytes_to_f32(&norm_bf16), &format!("layer {layer} norm"));
        }
    }
}
