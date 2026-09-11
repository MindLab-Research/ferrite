//! Quantisation primitives for DeepSeek-V4.1-Flash.
//!
//! Three formats appear in the released checkpoint and in the runtime:
//!
//! | where | format | scale |
//! |---|---|---|
//! | dense weights (attn/engram-wkv/shared experts/mtp) | fp8 e4m3 | ue8m0, **32x32 blocks** |
//! | routed experts | **fp4 e2m1**, 2 per byte (I8) | ue8m0, **per row x 32-col block** |
//! | engram tables | fp8 e4m3 | ue8m0, per row x 32-col block |
//! | window KV (runtime) | fp8 e4m3 | block 128, power-of-2 (`round_scale`) |
//! | compressed KV (runtime) | **fp4** | block 16, e4m3 scales |
//! | indexer q/k (runtime) | **fp4** | block 32, e8m0 scales |
//!
//! Every routine here mirrors the reference kernels bit-for-bit
//! (`inference/kernel.py`: `fast_log2_ceil` / `fast_pow2` / `fast_round_scale`
//! / `act_quant_kernel` / `fp4_quant_kernel`; `inference/convert.py`:
//! `cast_e2m1fn_to_e4m3fn`).

/// The e2m1 code table, indexed by the 4-bit code. Codes 0..7 are the
/// positive magnitudes, 8..15 mirror them (index 8 is a negative zero and
/// encodes as 0.0).
pub const FP4_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, //
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

pub const FP4_MAX: f32 = 6.0;
pub const FP8_MAX: f32 = 448.0;

// ---------------------------------------------------------------- e2m1 (fp4)

#[inline]
pub fn e2m1_decode(code: u8) -> f32 {
    FP4_TABLE[(code & 0xF) as usize]
}

/// Nearest-code encode. Ties go to the code with the smaller magnitude-slot
/// index (the reference casts with a round-to-nearest on the value, which for
/// this non-uniform table selects the same code).
pub fn e2m1_encode(x: f32) -> u8 {
    let a = x.abs().min(FP4_MAX);
    // magnitudes available at codes 0..7
    let mag = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mut best = 0usize;
    let mut bestd = f32::INFINITY;
    for (i, &m) in mag.iter().enumerate() {
        let d = (a - m).abs();
        if d < bestd {
            bestd = d;
            best = i;
        }
    }
    let sign = if x < 0.0 { 8 } else { 0 };
    (best as u8) | sign
}

/// The I8-packed layout: element `2*i` lives in the low nibble of byte `i`,
/// element `2*i+1` in the high nibble (matches `convert.py`'s
/// `low = x & 0x0F`, `high = (x >> 4) & 0x0F`).
#[inline]
pub fn fp4_unpack_byte(b: u8) -> (u8, u8) {
    (b & 0x0F, (b >> 4) & 0x0F)
}

#[inline]
pub fn fp4_pack_byte(lo: u8, hi: u8) -> u8 {
    (lo & 0x0F) | ((hi & 0x0F) << 4)
}

/// Decode a whole packed fp4 row into f32.
pub fn fp4_decode_packed(packed: &[u8], out: &mut [f32]) {
    debug_assert_eq!(packed.len() * 2, out.len());
    for (i, &b) in packed.iter().enumerate() {
        let (lo, hi) = fp4_unpack_byte(b);
        out[2 * i] = e2m1_decode(lo);
        out[2 * i + 1] = e2m1_decode(hi);
    }
}

// ------------------------------------------------------------------ fp8 e4m3

/// Decode an e4m3 byte: sign(1) exp(4) mantissa(3), bias 7.
/// value = 2^(E-7) * (1 + m/8), subnormals (E=0) are m * 2^-9.
pub fn e4m3_decode(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0x0F) as i32;
    let m = (b & 0x07) as f32;
    if e == 0 {
        sign * m * (1.0 / 512.0)
    } else {
        sign * (1.0 + m / 8.0) * exp2i(e - 7)
    }
}

/// Round-to-nearest-even encode with saturation to +-448 (what the reference
/// gets from clamping into [-448, 448] and casting). Non-finite input
/// saturates; the reference never produces NaN here.
pub fn e4m3_encode(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    if !x.is_finite() {
        return sign | 0x7E;
    }
    let a = x.abs().min(FP8_MAX);
    if a == 0.0 {
        return sign;
    }
    if a < exp2i(-6) {
        // subnormal, step 2^-9
        let m = (a * 512.0).round_ties_even().clamp(0.0, 7.0) as u8;
        return sign | m;
    }
    // floor(log2 a), saturating the exponent at the format's top
    let mut e = (((a.to_bits() >> 23) & 0xFF) as i32 - 127).min(8);
    let mut m = ((a / exp2i(e)) - 1.0) * 8.0;
    let mut mi = m.round_ties_even() as i32;
    if mi >= 8 {
        mi = 0;
        e += 1;
    }
    if e > 8 {
        return sign | 0x7E; // 448
    }
    sign | ((((e + 7) as u8) & 0x0F) << 3) | ((mi as u8) & 0x07)
}

// -------------------------------------------------------------------- e8m0

/// `float8_e8m0fnu`: a pure power of two, `2^(b - 127)`; 0xFF is NaN.
#[inline]
pub fn ue8m0_decode(b: u8) -> f32 {
    if b == 0xFF {
        f32::NAN
    } else {
        exp2i(b as i32 - 127)
    }
}

pub fn ue8m0_encode_pow2(exp: i32) -> u8 {
    (exp + 127).clamp(0, 0xFE) as u8
}

/// Round a positive scale to the nearest power of two (`2^round(log2(s))`).
pub fn ue8m0_round(s: f32) -> f32 {
    if s <= 0.0 {
        return 0.0;
    }
    let e = s.log2().round();
    exp2i(e as i32)
}

// ------------------------------------------------------- reference bit tricks

/// `ceil(log2(x))` via IEEE-754 bit inspection (reference `fast_log2_ceil`).
pub fn fast_log2_ceil(x: f32) -> i32 {
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & ((1 << 23) - 1);
    exp - 127 + if man != 0 { 1 } else { 0 }
}

/// `2^x` for integer x (reference `fast_pow2`).
pub fn fast_pow2(e: i32) -> f32 {
    f32::from_bits(((e + 127) << 23) as u32)
}

/// `fast_pow2(fast_log2_ceil(amax * fp8_max_inv))` — the power-of-two scale
/// the reference uses whenever `round_scale=True`.
pub fn fast_round_scale(amax: f32) -> f32 {
    if amax == 0.0 {
        return 0.0;
    }
    fast_pow2(fast_log2_ceil(amax / FP8_MAX))
}

fn exp2i(e: i32) -> f32 {
    f32::from_bits((((e + 127).clamp(0, 254)) as u32) << 23)
}

// ---------------------------------------------------------- block quantisers

/// One block of activation quantisation in the reference's two flavours.
/// `round_scale` selects the power-of-two (ue8m0-compatible) scale.
#[derive(Debug, Clone, Copy)]
pub struct BlockQuant {
    pub scale: f32,
    pub max_abs: f32,
}

/// The scale a block would get (shared by ACT and FP4 quant).
pub fn block_scale(x: &[f32], maxv: f32, round_scale: bool) -> BlockQuant {
    let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let s = if round_scale {
        fast_pow2(fast_log2_ceil(amax / maxv)).max(f32::MIN_POSITIVE)
    } else {
        (amax / maxv).max(f32::MIN_POSITIVE)
    };
    BlockQuant { scale: s, max_abs: amax }
}

/// fp8 e4m3 block quantisation (reference `act_quant`).
pub fn act_quant_fp8(x: &[f32], block: usize, round_scale: bool) -> (Vec<u8>, Vec<f32>) {
    let nb = x.len() / block;
    let mut y = vec![0u8; x.len()];
    let mut s = vec![0f32; nb];
    for b in 0..nb {
        let blk = &x[b * block..(b + 1) * block];
        let q = block_scale(blk, FP8_MAX, round_scale);
        s[b] = q.scale;
        for (i, &v) in blk.iter().enumerate() {
            y[b * block + i] = e4m3_encode(v / q.scale);
        }
    }
    (y, s)
}

/// fp4 e2m1 block quantisation (reference `fp4_act_quant`); returns the packed
/// I8 layout and one scale per block.
pub fn act_quant_fp4(x: &[f32], block: usize, round_scale: bool) -> (Vec<u8>, Vec<f32>) {
    let nb = x.len() / block;
    let mut nib = vec![0u8; x.len()];
    let mut s = vec![0f32; nb];
    for b in 0..nb {
        let blk = &x[b * block..(b + 1) * block];
        let q = block_scale(blk, FP4_MAX, round_scale);
        s[b] = q.scale;
        for (i, &v) in blk.iter().enumerate() {
            nib[b * block + i] = e2m1_encode(v / q.scale);
        }
    }
    let mut packed = vec![0u8; x.len() / 2];
    for i in 0..packed.len() {
        packed[i] = fp4_pack_byte(nib[2 * i], nib[2 * i + 1]);
    }
    (packed, s)
}

// ------------------------------------------------------- weight dequantisers

/// Dense weights: fp8 e4m3 `[out, in]` with ue8m0 scales on `block x block`
/// tiles (`[out/block, in/block]`). Returns f32 `[out, in]`.
pub fn dequant_fp8_block(
    w: &[u8],
    scale: &[u8],
    out: usize,
    inn: usize,
    block: usize,
    dst: &mut [f32],
) {
    let nb_in = inn / block;
    for r in 0..out {
        for c in 0..inn {
            let s = ue8m0_decode(scale[(r / block) * nb_in + c / block]);
            dst[r * inn + c] = e4m3_decode(w[r * inn + c]) * s;
        }
    }
}

/// Routed experts: fp4, I8-packed `[out, in/2]` with ue8m0 scales
/// `[out, in/32]` (per row x 32-column blocks). Returns f32 `[out, in]`.
pub fn dequant_fp4_row32(
    w: &[u8],
    scale: &[u8],
    out: usize,
    inn: usize,
    dst: &mut [f32],
) {
    const B: usize = 32;
    let nb_in = inn / B;
    for r in 0..out {
        let row = &w[r * (inn / 2)..(r + 1) * (inn / 2)];
        for c in 0..inn {
            let byte = row[c / 2];
            let code = if c % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
            let s = ue8m0_decode(scale[r * nb_in + c / B]);
            dst[r * inn + c] = e2m1_decode(code) * s;
        }
    }
}

/// Engram tables: fp8 e4m3 `[rows, dim]` with ue8m0 `[rows, dim/32]`.
/// Only one row is decoded (the lookup path).
pub fn dequant_fp8_row32(row: &[u8], scale: &[u8], dim: usize, dst: &mut [f32]) {
    for c in 0..dim {
        dst[c] = e4m3_decode(row[c]) * ue8m0_decode(scale[c / 32]);
    }
}

// ------------------------------------------------- the fp4 -> e4m3 cast path

/// The reference's lossless fp4 -> e4m3 weight cast (`convert.py`).
///
/// An fp4 tensor with per-(row, 32-col) e8m0 scales is rewritten as an e4m3
/// tensor whose scale is **one e8m0 value per 32x32 tile**: inside a tile each
/// 32-element row segment is multiplied by `offset = seg_scale / tile_max`,
/// which is a power of two in `[1, 2^6)` because both are e8m0. fp4 values are
/// small integers times powers of two, so they stay exact in e4m3.
pub fn cast_fp4_to_e4m3(
    packed: &[u8],
    seg_scale: &[f32],
    out: usize,
    inn: usize,
    w_out: &mut [u8],
    scale_out: &mut [u8],
) {
    const T: usize = 32;
    const MAX_OFFSET_BITS: i32 = 6;
    let nseg = inn / T;
    let nb_in = inn / T;
    for tr in 0..out / T {
        for tc in 0..nb_in {
            // tile max scale over the 32 rows x 1 segment
            let mut mx = 0f32;
            for r in 0..T {
                mx = mx.max(seg_scale[(tr * T + r) * nseg + tc]);
            }
            let tile_scale = fast_pow2(fast_log2_ceil(mx) - MAX_OFFSET_BITS).max(f32::MIN_POSITIVE);
            scale_out[tr * nb_in + tc] = ue8m0_encode_pow2(fast_log2_ceil(tile_scale));
            for r in 0..T {
                let row = tr * T + r;
                let off = seg_scale[row * nseg + tc] / tile_scale;
                for c in 0..T {
                    let col = tc * T + c;
                    let byte = packed[row * (inn / 2) + col / 2];
                    let code = if col % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
                    w_out[row * inn + col] = e4m3_encode(e2m1_decode(code) * off);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp4_table_matches_reference() {
        // convert.py FP4_TABLE
        let want = [
            0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        assert_eq!(FP4_TABLE, want);
        assert_eq!(e2m1_decode(0b0111), 6.0);
        assert_eq!(e2m1_decode(0b1111), -6.0);
        assert_eq!(e2m1_decode(0b1000), 0.0); // negative zero encodes as 0.0
        assert_eq!(e2m1_decode(0b1001), -0.5);
    }

    #[test]
    fn fp4_roundtrip_all_codes() {
        for code in 0u8..16 {
            let v = e2m1_decode(code);
            let back = e2m1_encode(v);
            assert_eq!(
                e2m1_decode(back),
                v,
                "code {code} -> {v} -> {}",
                e2m1_decode(back)
            );
        }
        // nearest-value behaviour
        assert_eq!(e2m1_decode(e2m1_encode(0.2)), 0.0);
        assert_eq!(e2m1_decode(e2m1_encode(0.3)), 0.5);
        assert_eq!(e2m1_decode(e2m1_encode(-5.7)), -6.0);
        assert_eq!(e2m1_decode(e2m1_encode(100.0)), 6.0); // saturates at FP4_MAX
    }

    #[test]
    fn fp4_pack_layout() {
        // low nibble = even element, high = odd (convert.py)
        let b = fp4_pack_byte(0b0111, 0b1111);
        assert_eq!(b & 0x0F, 7);
        assert_eq!((b >> 4) & 0x0F, 15);
        let packed = [b];
        let mut out = [0f32; 2];
        fp4_decode_packed(&packed, &mut out);
        assert_eq!(out, [6.0, -6.0]);
    }

    #[test]
    fn e4m3_known_values() {
        // exponent bias 7: 0b0_1000_000 = 2^1 = 2.0
        assert_eq!(e4m3_decode(0b0100_0000), 2.0);
        // 2.5 = 2^1 * 1.25 -> E=8, m=2
        assert_eq!(e4m3_decode(0b0100_0010), 2.5);
        assert_eq!(e4m3_decode(0b1100_0000), -2.0);
        assert_eq!(e4m3_decode(0b0100_1000), 4.0);
        // 448 = exp 15, mantissa 6 -> 0b0_1111_110
        assert_eq!(e4m3_decode(0b0111_1110), 448.0);
        assert_eq!(e4m3_decode(0b0000_0000), 0.0);
        // smallest subnormal 2^-9
        assert_eq!(e4m3_decode(0b0000_0001), 1.0 / 512.0);
    }

    #[test]
    fn e4m3_roundtrip_saturating() {
        for &v in &[0.0f32, 1.0, 2.5, -2.5, 448.0, -448.0, 0.001953125, 1e9, -1e9] {
            let enc = e4m3_encode(v);
            let dec = e4m3_decode(enc);
            let want = v.clamp(-448.0, 448.0);
            // within one e4m3 ulp of the clamped value
            let ulp = (want.abs() * 2.0f32.powi(-3)).max(1.0 / 512.0);
            assert!((dec - want).abs() <= ulp, "{v} -> {dec}");
        }
        assert_eq!(e4m3_encode(1e9), 0b0111_1110); // +448
        assert_eq!(e4m3_encode(-1e9), 0b1111_1110);
    }

    #[test]
    fn ue8m0_is_pow2() {
        assert_eq!(ue8m0_decode(127), 1.0);
        assert_eq!(ue8m0_decode(128), 2.0);
        assert_eq!(ue8m0_decode(126), 0.5);
        assert_eq!(ue8m0_decode(0), exp2i(-127));
        assert!(ue8m0_decode(255).is_nan());
        assert_eq!(ue8m0_encode_pow2(0), 127);
        assert_eq!(ue8m0_encode_pow2(3), 130);
    }

    #[test]
    fn bit_tricks_match_reference() {
        // fast_log2_ceil: exact powers give the exponent, non-powers round up
        assert_eq!(fast_log2_ceil(1.0), 0);
        assert_eq!(fast_log2_ceil(2.0), 1);
        assert_eq!(fast_log2_ceil(3.0), 2);
        assert_eq!(fast_log2_ceil(0.5), -1);
        assert_eq!(fast_log2_ceil(1.5), 1);
        assert_eq!(fast_pow2(0), 1.0);
        assert_eq!(fast_pow2(3), 8.0);
        assert_eq!(fast_pow2(-1), 0.5);
        // round_scale rounds a scale up to a power of two >= x/448.  an x whose
        // amax/448 is exactly 1 keeps 1; slightly above gives 2
        assert_eq!(fast_round_scale(448.0), 1.0);
        assert_eq!(fast_round_scale(449.0), 2.0);
    }

    #[test]
    fn act_quant_fp8_block128_matches_definition() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.7).collect();
        let (y, s) = act_quant_fp8(&x, 128, false);
        assert_eq!(s.len(), 2);
        // scale = amax/448
        for b in 0..2 {
            let amax = x[b * 128..(b + 1) * 128]
                .iter()
                .fold(0f32, |a, &v| a.max(v.abs()));
            assert!((s[b] - amax / 448.0).abs() < 1e-6);
        }
        // dequantised values are within one e4m3 ulp *of the scaled value*:
        // the block scale maps `amax` onto 448, whose ulp is 32, so the error
        // bound is 0.5 * 32 * scale = 16 * scale (the format's 6.25% relative
        // resolution — this is inherent to a shared block scale, not a bug).
        for i in 0..256 {
            let b = i / 128;
            let d = e4m3_decode(y[i]) * s[b];
            assert!((d - x[i]).abs() <= 16.0 * s[b] + 1e-6, "i={i} {d} vs {}", x[i]);
        }
    }

    #[test]
    fn act_quant_fp4_block32_round_scale_is_pow2() {
        let x: Vec<f32> = (0..64).map(|i| ((i % 7) as f32 - 3.0) * 0.9).collect();
        let (packed, s) = act_quant_fp4(&x, 32, true);
        assert_eq!(packed.len(), 32);
        for &v in &s {
            // power of two
            let e = v.log2();
            assert!((e - e.round()).abs() < 1e-6, "scale {v} not a power of two");
        }
    }

    #[test]
    fn dequant_fp4_row32_layout() {
        // one row, 4 elements (2 bytes); in=4 (< 32) is not a legal block, so
        // use in=32 and check the first 4 entries
        let out = 1usize;
        let inn = 32usize;
        let mut packed = vec![0u8; inn / 2];
        // element (0,0)=6.0 (code 7), (0,1)=-0.5 (code 9)
        packed[0] = fp4_pack_byte(0b0111, 0b1001);
        let scale = vec![127u8]; // ue8m0 1.0
        let mut dst = vec![0f32; out * inn];
        dequant_fp4_row32(&packed, &scale, out, inn, &mut dst);
        assert_eq!(dst[0], 6.0);
        assert_eq!(dst[1], -0.5);
        // scale 2 multiplies through
        let scale2 = vec![128u8];
        dequant_fp4_row32(&packed, &scale2, out, inn, &mut dst);
        assert_eq!(dst[0], 12.0);
    }

    #[test]
    fn dequant_fp8_block32x32_layout() {
        let out = 32usize;
        let inn = 32usize;
        let w = vec![0b0100_0000u8; out * inn]; // 2.0 everywhere
        let scale = vec![128u8]; // x2 -> 4.0
        let mut dst = vec![0f32; out * inn];
        dequant_fp8_block(&w, &scale, out, inn, 32, &mut dst);
        assert!(dst.iter().all(|&v| v == 4.0));
    }

    #[test]
    fn cast_fp4_to_e4m3_is_lossless_per_tile() {
        // 32x32 tile, one segment; per-row segment scales differ by powers of two
        let out = 32usize;
        let inn = 32usize;
        let mut packed = vec![0u8; out * inn / 2];
        for i in 0..out * inn / 2 {
            packed[i] = fp4_pack_byte(0b0100, 0b1100); // 2.0, -2.0
        }
        let mut seg = vec![0f32; out]; // 32 rows x 1 segment
        for r in 0..out {
            seg[r] = exp2i(-(r as i32 % 7)); // 1, 1/2, ..., 1/64
        }
        let mut w_out = vec![0u8; out * inn];
        let mut s_out = vec![0u8; 1];
        cast_fp4_to_e4m3(&packed, &seg, out, inn, &mut w_out, &mut s_out);
        let tile_scale = ue8m0_decode(s_out[0]);
        for r in 0..out {
            for c in 0..inn {
                let got = e4m3_decode(w_out[r * inn + c]) * tile_scale;
                let want = e2m1_decode(if c % 2 == 0 { 0b0100 } else { 0b1100 }) * seg[r];
                assert_eq!(got, want, "r={r} c={c}");
            }
        }
    }
}
